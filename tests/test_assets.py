"""Tests for exec_sandbox.assets: where each VM file is found (override, EXEC_SANDBOX_IMAGES_DIR,
images/dist of the source tree, the download cache), and which release file is fetched when no
folder has it.

Every test goes through the public functions. A release file is "downloaded" with no request:
the folders fixture puts it in the download cache with its digest in the registry, as a
finished download (retrieve uses it when its digest is right), and the fetch unpacks it.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import re
import stat
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import aiohttp
import pytest
from aioresponses import aioresponses

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

from exec_sandbox import assets
from exec_sandbox.asset_downloader import AsyncPooch
from exec_sandbox.assets import (
    ensure_assets,
    ensure_assets_available,
    ensure_base_image,
    ensure_registry_loaded,
    fetch_gvproxy,
    get_assets,
    get_gvproxy_path,
    is_offline_mode,
    prefetch_all_assets,
)
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.images import BASE_IMAGES, kernel_dir_of
from exec_sandbox.models import Language
from exec_sandbox.platform_utils import detect_host_arch

HOST_ARCH = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}[
    platform.machine().lower()
]
"""This host's architecture in the kernel's names (macOS calls ARM64 "arm64")."""

HOST_OS = platform.system().lower()
"""This host's OS as the gvproxy-wrapper builds name it: "darwin" or "linux"."""

GO_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}
"""The Go names of the architectures (the gvproxy-wrapper builds)."""

HOST_GVPROXY = f"gvproxy-wrapper-{HOST_OS}-{GO_ARCH[HOST_ARCH]}"


@dataclass(frozen=True)
class Folders:
    """The search folders in tmp_path, in the order the module reads them (repo_bin: only for
    gvproxy-wrapper, after the override). No folder exists until a test puts a file in it."""

    override: Path
    repo_bin: Path
    env: Path
    local: Path
    cache: Path

    def put(self, folder: Path, name: str, data: bytes = b"") -> Path:
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = folder / name
        path.write_bytes(data)
        return path

    def release(self, name: str, data: bytes) -> None:
        """`name` of the release, its content `data`: a finished download in the cache, its
        digest in the registry. A fetch unpacks it (a .zst) with no request."""
        raw = zstd.compress(data) if name.endswith(".zst") else data
        self.put(self.cache, name, raw)
        get_assets().registry[name] = f"sha256:{hashlib.sha256(raw).hexdigest()}"


@pytest.fixture
def folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Folders:
    """Every search folder in tmp_path, and a new registry: empty, offline (no request to
    GitHub), its download cache in `cache` (EXEC_SANDBOX_CACHE_DIR). The source tree moves to
    tmp_path (assets.__file__): the images/dist and gvproxy-wrapper/bin of the repository
    stay out."""
    monkeypatch.setattr(assets, "__file__", str(tmp_path / "src" / "exec_sandbox" / "assets.py"))
    monkeypatch.setenv("EXEC_SANDBOX_IMAGES_DIR", str(tmp_path / "env"))
    monkeypatch.setenv("EXEC_SANDBOX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EXEC_SANDBOX_OFFLINE", "1")
    monkeypatch.setenv("EXEC_SANDBOX_ASSET_VERSION", "1.2.3")
    monkeypatch.setattr(assets, "_assets_singleton", None)  # get_assets makes a new registry from this env
    return Folders(
        override=tmp_path / "override",
        repo_bin=tmp_path / "gvproxy-wrapper" / "bin",
        env=tmp_path / "env",
        local=tmp_path / "images" / "dist",
        cache=tmp_path / "cache",
    )


def requested(m: aioresponses) -> list[str]:
    """The URLs of the requests, in order."""
    return [str(url) for _, url in m.requests]


# =============================================================================
# Lookup in the search folders
# =============================================================================


class TestLocalLookup:
    """A file in a search folder is used as it is, from the first folder that has it
    (TestFetchGvproxy shows the order)."""

    @pytest.mark.parametrize("value", ["", "   "], ids=["empty", "blank"])
    async def test_empty_env_var_is_no_folder(
        self, folders: Folders, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """An empty or blank EXEC_SANDBOX_IMAGES_DIR is no folder: not the current folder, not
        a folder with blanks as name. Both have the file here; the cache is used."""
        monkeypatch.setenv("EXEC_SANDBOX_IMAGES_DIR", value)
        monkeypatch.chdir(tmp_path)
        folders.put(tmp_path, HOST_GVPROXY)
        folders.put(tmp_path / "   ", HOST_GVPROXY)
        expected = folders.put(folders.cache, HOST_GVPROXY)

        assert await get_gvproxy_path(override=None) == expected

    async def test_symlink_is_followed(self, folders: Folders) -> None:
        """A symlink to a file counts as the file; the path of the link is returned."""
        real = folders.put(folders.local, "real-gvproxy")
        folders.env.mkdir()
        (folders.env / HOST_GVPROXY).symlink_to(real)

        assert await get_gvproxy_path(override=None) == folders.env / HOST_GVPROXY

    async def test_folder_path_with_spaces(
        self, folders: Folders, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = tmp_path / "my images"
        monkeypatch.setenv("EXEC_SANDBOX_IMAGES_DIR", str(env))
        expected = folders.put(env, HOST_GVPROXY)

        assert await get_gvproxy_path(override=None) == expected

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads every folder")
    async def test_folder_this_process_cannot_read_is_skipped(self, folders: Folders) -> None:
        """A search folder this process cannot read counts as missing (no PermissionError):
        the next folder has the file."""
        folders.put(folders.env, HOST_GVPROXY)
        expected = folders.put(folders.cache, HOST_GVPROXY)
        folders.env.chmod(0)
        try:
            assert await get_gvproxy_path(override=None) == expected
        finally:
            folders.env.chmod(0o700)

    async def test_lookups_run_off_the_event_loop(self, folders: Folders) -> None:
        """The search folders include the caller's (a network disk can take ms per stat):
        every existence check runs in a worker thread, not on the event loop thread."""
        for name in (
            f"vmlinuz-{HOST_ARCH}",
            f"initramfs-{HOST_ARCH}",
            BASE_IMAGES[Language.PYTHON].file_name(detect_host_arch()),
            HOST_GVPROXY,
        ):
            folders.put(folders.override, name)
        loop_thread = threading.get_ident()
        threads: list[int] = []
        real_exists = os.path.exists

        def exists(path: str | os.PathLike[str]) -> bool:
            threads.append(threading.get_ident())
            return real_exists(path)

        for name in (f"vmlinuz-{HOST_ARCH}", f"initramfs-{HOST_ARCH}", HOST_GVPROXY):
            folders.put(folders.cache, name)

        with patch("exec_sandbox.assets.os.path.exists", side_effect=exists):
            assert await get_gvproxy_path(override=folders.override) == folders.override / HOST_GVPROXY
            assert await ensure_assets(override=folders.override, download=False) == folders.override
            await ensure_base_image(Language.PYTHON, images_dir=folders.override)
            assert (await ensure_assets_available())[0] == folders.cache  # the cache lookups

        assert threads
        assert loop_thread not in threads


# =============================================================================
# ensure_assets
# =============================================================================


class TestEnsureAssets:
    """ensure_assets: the first search folder that holds the boot files (a kernel and the
    initramfs); else the download cache, with both downloaded into it."""

    @staticmethod
    def _put_boot_files(folders: Folders, folder: Path) -> None:
        folders.put(folder, f"vmlinuz-{HOST_ARCH}")
        folders.put(folder, f"initramfs-{HOST_ARCH}")

    async def test_returns_existing_images_dir(self, folders: Folders) -> None:
        self._put_boot_files(folders, folders.override)
        assert await ensure_assets(override=folders.override, download=False) == folders.override

    async def test_finds_env_var_dir(self, folders: Folders) -> None:
        self._put_boot_files(folders, folders.env)
        assert await ensure_assets(override=None, download=False) == folders.env

    async def test_boot_files_in_a_kernels_subfolder_count(self, folders: Folders) -> None:
        """The kernels/ layout of container images: the settings read the boot files from
        images_dir/kernels, so a folder with them there is the images folder."""
        self._put_boot_files(folders, folders.env / "kernels")
        assert await ensure_assets(override=None, download=False) == folders.env

    @pytest.mark.parametrize(
        "partial",
        [[], [f"vmlinuz-{HOST_ARCH}"], [f"initramfs-{HOST_ARCH}"]],
        ids=["empty", "kernel-only", "initramfs-only"],
    )
    async def test_a_folder_without_both_boot_files_does_not_hide_a_later_one(
        self, folders: Folders, partial: list[str]
    ) -> None:
        """The override exists with none or one of the boot files; EXEC_SANDBOX_IMAGES_DIR holds
        both: the env folder is the images folder, since VMs read both from it."""
        folders.override.mkdir()
        for name in partial:
            folders.put(folders.override, name)
        self._put_boot_files(folders, folders.env)
        assert await ensure_assets(override=folders.override, download=False) == folders.env

    async def test_an_empty_kernels_folder_in_the_cache_does_not_hide_its_boot_files(self, folders: Folders) -> None:
        """The cache holds an empty kernels/ folder: the download fills the cache's top, and
        the settings read the boot files there too, not in kernels/."""
        folders.cache.mkdir(mode=0o700)  # the downloader refuses a cache that group or others can enter
        (folders.cache / "kernels").mkdir()
        folders.release(f"vmlinuz-{HOST_ARCH}.zst", b"kernel")
        folders.release(f"initramfs-{HOST_ARCH}.zst", b"initramfs")
        folders.release(HOST_GVPROXY, b"gvproxy")

        images_dir = await ensure_assets(override=None, download=True)

        assert images_dir == folders.cache
        assert kernel_dir_of(images_dir, arch=detect_host_arch()) == folders.cache

    @pytest.mark.parametrize("download", [True, False])
    async def test_a_local_images_folder_gets_gvproxy_when_downloads_are_allowed(
        self, folders: Folders, download: bool
    ) -> None:
        """The override holds both boot files; gvproxy-wrapper is a finished download in the
        cache, not yet executable. With downloads allowed, the start fetches it where a VM
        start finds it (images folder first, then the search folders) and makes it
        executable; without, it leaves it as it is."""
        self._put_boot_files(folders, folders.override)
        folders.release(HOST_GVPROXY, b"gvproxy")

        assert await ensure_assets(override=folders.override, download=download) == folders.override

        assert await get_gvproxy_path(override=folders.override) == folders.cache / HOST_GVPROXY
        assert os.access(folders.cache / HOST_GVPROXY, os.X_OK) is download

    async def test_the_images_folder_gvproxy_comes_first(self, folders: Folders) -> None:
        """gvproxy-wrapper is in the images folder and in the cache: the start takes the images
        folder's (it makes that one executable, not the cache's), as a VM start does."""
        self._put_boot_files(folders, folders.override)
        folders.put(folders.override, HOST_GVPROXY, b"local gvproxy")
        folders.release(HOST_GVPROXY, b"gvproxy")

        await ensure_assets(override=folders.override, download=True)

        assert os.access(folders.override / HOST_GVPROXY, os.X_OK)
        assert not os.access(folders.cache / HOST_GVPROXY, os.X_OK)

    async def test_a_gvproxy_that_cannot_be_made_executable_does_not_stop_the_start(self, folders: Folders) -> None:
        """The images folder's gvproxy-wrapper refuses the chmod (a read-only folder, a file of
        another user): the start still returns the images folder. Such a folder does
        not come on demand in a test, so the chmod raises."""
        self._put_boot_files(folders, folders.override)
        folders.put(folders.override, HOST_GVPROXY, b"local gvproxy")

        with patch("exec_sandbox.assets.chmod_executable", side_effect=PermissionError(1, "Operation not permitted")):
            assert await ensure_assets(override=folders.override, download=True) == folders.override

    async def test_a_gvproxy_that_cannot_be_downloaded_does_not_stop_the_start(self, folders: Folders) -> None:
        """No gvproxy-wrapper anywhere, none in the release (offline): the start still returns
        the images folder; a VM with no network needs none."""
        self._put_boot_files(folders, folders.override)

        assert await ensure_assets(override=folders.override, download=True) == folders.override
        assert await get_gvproxy_path(override=folders.override) is None

    async def test_a_download_puts_both_boot_files_in_the_cache(self, folders: Folders) -> None:
        """No folder holds both: images/dist holds only the initramfs, the cache only a kernel.
        The download path makes the cache hold both (the initramfs comes from the release, not
        from images/dist) and returns the cache."""
        folders.put(folders.local, f"initramfs-{HOST_ARCH}", b"local initramfs")
        folders.put(folders.cache, f"vmlinuz-{HOST_ARCH}", b"cached kernel")
        folders.release(f"initramfs-{HOST_ARCH}.zst", b"released initramfs")
        folders.release(HOST_GVPROXY, b"gvproxy")

        assert await ensure_assets(override=None, download=True) == folders.cache
        assert (folders.cache / f"vmlinuz-{HOST_ARCH}").read_bytes() == b"cached kernel"
        assert (folders.cache / f"initramfs-{HOST_ARCH}").read_bytes() == b"released initramfs"

    async def test_raises_when_not_found_and_download_disabled(self, folders: Folders) -> None:
        """No folder: ConfigError ASSET_NOT_FOUND, with each folder it searched in the message."""
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND) as exc_info:
            await ensure_assets(override=folders.override, download=False)

        searched = [str(folder) for folder in (folders.override, folders.env, folders.local, folders.cache)]
        assert str(exc_info.value) == (
            f"Assets not found and auto_download_assets=False. Searched: {searched}. "
            "Set EXEC_SANDBOX_IMAGES_DIR or enable auto_download_assets."
        )

    async def test_empty_dir_triggers_download(self, folders: Folders) -> None:
        """The first folder exists but has no kernel: with download=True, the kernel, the
        initramfs and gvproxy-wrapper come from the release, and the images folder is the
        download cache."""
        folders.override.mkdir()
        folders.release(f"vmlinuz-{HOST_ARCH}.zst", b"kernel")
        folders.release(f"initramfs-{HOST_ARCH}.zst", b"initramfs")
        folders.release(HOST_GVPROXY, b"gvproxy")

        assert await ensure_assets(override=folders.override, download=True) == folders.cache
        assert (folders.cache / f"vmlinuz-{HOST_ARCH}").read_bytes() == b"kernel"


class TestEnsureAssetsAvailable:
    """ensure_assets_available: the download cache with both boot files, and the
    gvproxy-wrapper path."""

    async def test_boot_files_come_from_the_cache_gvproxy_from_the_search_folders(self, folders: Folders) -> None:
        """EXEC_SANDBOX_IMAGES_DIR holds a kernel and gvproxy-wrapper: the boot files are still
        downloaded into the cache (an images folder must hold both); gvproxy-wrapper is
        found in the env folder, where a VM start also looks."""
        for name in (f"vmlinuz-{HOST_ARCH}", HOST_GVPROXY):
            folders.put(folders.env, name)
        folders.release(f"vmlinuz-{HOST_ARCH}.zst", b"kernel")
        folders.release(f"initramfs-{HOST_ARCH}.zst", b"initramfs")

        assert await ensure_assets_available() == (folders.cache, folders.env / HOST_GVPROXY)
        assert (folders.cache / f"initramfs-{HOST_ARCH}").read_bytes() == b"initramfs"

    async def test_a_failed_fetch_ends_the_others_before_the_call_raises(self, folders: Folders) -> None:
        """The release has no kernel (ASSET_NOT_FOUND at once) while the initramfs download
        waits for a server that never answers: the call raises the kernel's error with no
        fetch left running."""
        folders.put(folders.env, HOST_GVPROXY)  # found: no gvproxy-wrapper fetch
        get_assets().registry[f"initramfs-{HOST_ARCH}.zst"] = f"sha256:{'0' * 64}"

        async def no_answer(_url: object, **_kwargs: object) -> None:
            await asyncio.Event().wait()

        others = asyncio.all_tasks()
        try:
            with aioresponses() as m:
                m.get(re.compile(r".*/initramfs-[a-z0-9_]+\.zst$"), callback=no_answer)
                with pytest.raises(
                    ConfigError,
                    check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND,
                    match=rf"^File 'vmlinuz-{HOST_ARCH}\.zst' not in registry$",
                ):
                    await ensure_assets_available()
                left = asyncio.all_tasks() - others
                assert left == set()
        finally:
            for task in asyncio.all_tasks() - others:
                task.cancel()


class TestEnsureBaseImage:
    """ensure_base_image: the base image where GuestImages reads it, the images folder or else
    the download cache; never another search folder."""

    @staticmethod
    def _name() -> str:
        return BASE_IMAGES[Language.PYTHON].file_name(detect_host_arch())

    @pytest.mark.parametrize("where", ["images", "cache"])
    async def test_a_present_image_is_not_fetched(self, folders: Folders, where: str) -> None:
        """In the images folder or in the cache: nothing is fetched (the registry is empty and
        offline, so a fetch would raise)."""
        folders.put(folders.override if where == "images" else folders.cache, self._name(), b"image")
        await ensure_base_image(Language.PYTHON, images_dir=folders.override)

    async def test_an_image_in_another_search_folder_is_downloaded_into_the_cache(self, folders: Folders) -> None:
        """EXEC_SANDBOX_IMAGES_DIR holds the image, the images folder does not: GuestImages
        would not read the env folder, so the image is downloaded into the cache."""
        folders.put(folders.env, self._name(), b"env image")
        folders.release(f"{self._name()}.zst", b"released image")

        await ensure_base_image(Language.PYTHON, images_dir=folders.override)

        assert (folders.cache / self._name()).read_bytes() == b"released image"


class TestOfflineMode:
    """Tests for is_offline_mode() function."""

    def test_offline_mode_enabled(self) -> None:
        """EXEC_SANDBOX_OFFLINE=1 enables offline mode."""
        with patch.dict(os.environ, {"EXEC_SANDBOX_OFFLINE": "1"}):
            assert is_offline_mode() is True

    def test_offline_mode_disabled_explicit(self) -> None:
        """EXEC_SANDBOX_OFFLINE=0 disables offline mode."""
        with patch.dict(os.environ, {"EXEC_SANDBOX_OFFLINE": "0"}):
            assert is_offline_mode() is False

    def test_offline_mode_disabled_by_default(self) -> None:
        """Offline mode is disabled when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_offline_mode() is False

    def test_offline_mode_other_values(self) -> None:
        """Only '1' enables offline mode, other values disable it."""
        for value in ["true", "yes", "TRUE", "2", ""]:
            with patch.dict(os.environ, {"EXEC_SANDBOX_OFFLINE": value}):
                assert is_offline_mode() is False, f"Expected False for value '{value}'"


class TestGetAssets:
    """get_assets: one registry for the process, its download cache in the cache folder."""

    def test_returns_one_registry_in_the_cache_override(self, folders: Folders) -> None:
        """With EXEC_SANDBOX_CACHE_DIR, downloads go in that folder (no version folder)."""
        registry = get_assets()

        assert isinstance(registry, AsyncPooch)
        assert get_assets() is registry
        assert registry.path == folders.cache

    def test_default_cache_has_a_folder_per_version(
        self, folders: Folders, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No EXEC_SANDBOX_CACHE_DIR: the user cache folder, then v<package version>."""
        monkeypatch.delenv("EXEC_SANDBOX_CACHE_DIR")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setattr(assets, "__version__", "9.8.7")
        user_cache = tmp_path / "home" / ("Library/Caches" if sys.platform == "darwin" else ".cache") / "exec-sandbox"

        assert get_assets().path == user_cache / "v9.8.7"


# =============================================================================
# Release names of a download
# =============================================================================


class TestReleaseNames:
    """The release file a download asks for (prefetch_all_assets: the download cache of an
    arch): the kernel of kernel_names that the registry has, and the base images by the names
    of scripts/build-qcow2.sh."""

    async def test_kernel_prefers_vmlinux_x86_64_when_in_registry(self, folders: Folders) -> None:
        """The release has vmlinux and vmlinuz: vmlinux is fetched and unpacked."""
        folders.release("vmlinux-x86_64.zst", b"vmlinux")
        folders.release("vmlinuz-x86_64.zst", b"vmlinuz")

        await prefetch_all_assets(arch="x86_64")

        assert (folders.cache / "vmlinux-x86_64").read_bytes() == b"vmlinux"
        assert not (folders.cache / "vmlinuz-x86_64").exists()

    async def test_kernel_falls_back_to_vmlinuz_when_vmlinux_not_in_registry(self, folders: Folders) -> None:
        """The release has no vmlinux: vmlinuz is fetched."""
        folders.release("vmlinuz-x86_64.zst", b"vmlinuz")

        await prefetch_all_assets(arch="x86_64")

        assert (folders.cache / "vmlinuz-x86_64").read_bytes() == b"vmlinuz"

    async def test_kernel_aarch64_always_vmlinuz(self, folders: Folders) -> None:
        """On aarch64, vmlinuz is fetched even when the release has a vmlinux."""
        folders.release("vmlinux-aarch64.zst", b"vmlinux")
        folders.release("vmlinuz-aarch64.zst", b"vmlinuz")

        await prefetch_all_assets(arch="aarch64")

        assert (folders.cache / "vmlinuz-aarch64").read_bytes() == b"vmlinuz"
        assert not (folders.cache / "vmlinux-aarch64").exists()

    @pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
    async def test_base_image_names_match_the_build_script(self, folders: Folders, arch: str) -> None:
        """The release names its images as scripts/build-qcow2.sh does: python-<Python minor> and
        node-<Bun minor> of the script's default versions. A version bump there without the same
        change here would make the download ask for a file the release does not have."""
        script = (Path(__file__).parent.parent / "scripts" / "build-qcow2.sh").read_text()

        def default_minor(name: str) -> str:
            (version,) = re.findall(rf'^{name}="\$\{{{name}:-([0-9.]+)\}}"$', script, flags=re.MULTILINE)
            return version.rsplit(".", 1)[0]  # the script's ${VERSION%.*}

        # The naming rule these expectations follow, as the script writes it
        assert 'local python_minor="${PYTHON_VERSION%.*}"' in script
        assert 'local bun_minor="${BUN_VERSION%.*}"' in script
        assert 'python) output_name="python-${python_minor}-base-$target_arch"' in script
        assert 'node)   output_name="node-${bun_minor}-base-$target_arch"' in script
        assert 'raw)    output_name="raw-base-$target_arch"' in script
        expected = [
            f"python-{default_minor('PYTHON_VERSION')}-base-{arch}.qcow2",
            f"node-{default_minor('BUN_VERSION')}-base-{arch}.qcow2",
            f"raw-base-{arch}.qcow2",
        ]
        for image in expected:
            folders.release(f"{image}.zst", image.encode())

        result = await prefetch_all_assets(arch=arch)

        assert sorted(result.downloaded) == ["javascript base", "python base", "raw base"]  # no boot files released
        for image in expected:
            assert (folders.cache / image).read_bytes() == image.encode()


class TestFetchGvproxy:
    """gvproxy-wrapper: override, gvproxy-wrapper/bin of the source tree, then the search
    folders; else the release build of this OS for the asked arch."""

    @pytest.mark.parametrize("first", ["override", "repo_bin", "env", "local", "cache"])
    async def test_get_gvproxy_path_order(self, folders: Folders, first: str) -> None:
        """The binary is in `first` and in each folder after it: the one in `first` is used."""
        order = ["override", "repo_bin", "env", "local", "cache"]
        for name in order[order.index(first) :]:
            folders.put(getattr(folders, name), HOST_GVPROXY)

        assert await get_gvproxy_path(override=folders.override) == getattr(folders, first) / HOST_GVPROXY

    async def test_get_gvproxy_path_returns_none_when_not_found(self, folders: Folders) -> None:
        """get_gvproxy_path returns None when binary not found."""
        assert await get_gvproxy_path(override=None) is None

    async def test_fetch_gvproxy_returns_cached_and_makes_executable(self, folders: Folders) -> None:
        """fetch_gvproxy returns cached binary and actually makes it executable."""
        binary = folders.put(folders.env, HOST_GVPROXY)
        binary.chmod(0o600)

        assert await fetch_gvproxy(override=None, arch=HOST_ARCH) == binary
        assert binary.stat().st_mode & stat.S_IXUSR

    @pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
    async def test_fetch_gvproxy_downloads_the_binary_of_the_asked_arch(self, folders: Folders, arch: str) -> None:
        """The gvproxy-wrapper of the asked arch (and this OS), whatever arch this host has:
        `sbx prefetch --arch` fills the cache of another arch. It is made executable."""
        name = f"gvproxy-wrapper-{HOST_OS}-{GO_ARCH[arch]}"
        folders.release(name, arch.encode())

        result = await fetch_gvproxy(override=None, arch=arch)

        assert result == folders.cache / name
        assert result.read_bytes() == arch.encode()
        assert result.stat().st_mode & stat.S_IXUSR


# =============================================================================
# ensure_registry_loaded
# =============================================================================


class TestEnsureRegistryLoaded:
    """ensure_registry_loaded: an empty registry asks GitHub for the release of the asset
    version, once; offline, no request. The GitHub route fails here (the request is what is
    checked; test_asset_downloader.py reads the release metadata)."""

    API = "https://api.github.com/repos/dualeai/exec-sandbox/releases"

    @pytest.mark.parametrize(
        ("asset_version", "package_version", "path"),
        [
            pytest.param("1.0.0", "0.9.0", "tags/v1.0.0", id="env"),
            pytest.param("v1.0.0", "0.9.0", "tags/v1.0.0", id="env-with-v"),
            pytest.param(None, "2.0.0", "tags/v2.0.0", id="package-version"),
            pytest.param(None, "2.1.0.dev4", "latest", id="dev-version-is-latest"),
        ],
    )
    async def test_asks_github_for_the_release_of_the_asset_version(
        self,
        folders: Folders,
        monkeypatch: pytest.MonkeyPatch,
        asset_version: str | None,
        package_version: str,
        path: str,
    ) -> None:
        """EXEC_SANDBOX_ASSET_VERSION (with or without "v"), else the package version; a dev
        version asks for the latest release."""
        monkeypatch.delenv("EXEC_SANDBOX_OFFLINE")
        monkeypatch.setattr(assets, "__version__", package_version)
        if asset_version is None:
            monkeypatch.delenv("EXEC_SANDBOX_ASSET_VERSION")
        else:
            monkeypatch.setenv("EXEC_SANDBOX_ASSET_VERSION", asset_version)

        with aioresponses() as m:
            m.get(f"{self.API}/{path}", exception=aiohttp.ClientConnectionError("no route"))
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.ASSET_DOWNLOAD):
                await ensure_registry_loaded()
            assert requested(m) == [f"{self.API}/{path}"]

    async def test_skips_if_already_loaded(self, folders: Folders, monkeypatch: pytest.MonkeyPatch) -> None:
        """A registry with files: no request."""
        monkeypatch.delenv("EXEC_SANDBOX_OFFLINE")
        get_assets().registry["file.txt"] = "sha256:abc"

        with aioresponses() as m:  # no route: a request fails
            await ensure_registry_loaded()
            assert requested(m) == []

    async def test_skips_in_offline_mode(self, folders: Folders) -> None:
        """Offline (the fixture sets EXEC_SANDBOX_OFFLINE=1): no request, the registry stays empty."""
        with aioresponses() as m:
            await ensure_registry_loaded()
            assert requested(m) == []
        assert get_assets().registry == {}


# =============================================================================
# prefetch_all_assets
# =============================================================================


def release_all(folders: Folders, arch: str, *, without: frozenset[str] = frozenset()) -> None:
    """Every file prefetch_all_assets fetches for `arch`, by its result name, less `without`."""
    files = {
        "kernel": f"vmlinuz-{arch}.zst",
        "initramfs": f"initramfs-{arch}.zst",
        "gvproxy": f"gvproxy-wrapper-{HOST_OS}-{GO_ARCH[arch]}",
        "python base": f"python-3.14-base-{arch}.qcow2.zst",
        "javascript base": f"node-1.4-base-{arch}.qcow2.zst",
        "raw base": f"raw-base-{arch}.qcow2.zst",
    }
    for asset, name in files.items():
        if asset not in without:
            folders.release(name, name.encode())


class TestPrefetchAllAssets:
    """prefetch_all_assets: every file of an arch in the download cache, and a report."""

    @pytest.mark.parametrize("arch", [None, "x86_64", "aarch64"], ids=["host", "x86_64", "aarch64"])
    async def test_prefetch_downloads_all_assets(self, folders: Folders, arch: str | None) -> None:
        """The kernel, the initramfs, gvproxy-wrapper and the base image of each language, of
        the asked arch (this host's without one)."""
        expected_arch = arch or HOST_ARCH
        release_all(folders, expected_arch)

        result = await prefetch_all_assets(arch=arch)

        assert (result.success, result.arch, result.errors, result.cache_dir) == (
            True,
            expected_arch,
            [],
            folders.cache,
        )
        assert sorted(result.downloaded) == sorted(
            ["kernel", "initramfs", "gvproxy", "python base", "javascript base", "raw base"]
        )
        assert sorted(p.name for p in folders.cache.iterdir() if not p.name.endswith(".lock")) == sorted(
            [
                f"vmlinuz-{expected_arch}",
                f"initramfs-{expected_arch}",
                f"gvproxy-wrapper-{HOST_OS}-{GO_ARCH[expected_arch]}",
                f"python-3.14-base-{expected_arch}.qcow2",
                f"node-1.4-base-{expected_arch}.qcow2",
                f"raw-base-{expected_arch}.qcow2",
            ]
        )

    async def test_prefetch_fills_the_cache_whatever_other_folders_hold(self, folders: Folders) -> None:
        """EXEC_SANDBOX_IMAGES_DIR holds a kernel and the python image, but no initramfs: an
        offline start reads none of them (the images folder needs both boot files), so
        prefetch still puts every file in the cache."""
        folders.put(folders.env, f"vmlinuz-{HOST_ARCH}", b"env kernel")
        folders.put(folders.env, f"python-3.14-base-{HOST_ARCH}.qcow2", b"env image")
        release_all(folders, HOST_ARCH)

        result = await prefetch_all_assets(arch=None)

        assert (result.success, result.cache_dir) == (True, folders.cache)
        for name in (f"vmlinuz-{HOST_ARCH}", f"initramfs-{HOST_ARCH}", f"python-3.14-base-{HOST_ARCH}.qcow2"):
            assert (folders.cache / name).read_bytes() == f"{name}.zst".encode(), name

    async def test_prefetch_returns_error_on_registry_failure(
        self, folders: Folders, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The release metadata cannot be read: a failed result with the one error "registry"."""
        monkeypatch.delenv("EXEC_SANDBOX_OFFLINE")

        with aioresponses() as m:
            m.get(f"{TestEnsureRegistryLoaded.API}/tags/v1.2.3", exception=aiohttp.ClientConnectionError("no route"))
            result = await prefetch_all_assets(arch="x86_64")

        assert (result.success, result.downloaded, result.cache_dir) == (False, [], None)
        assert [name for name, _ in result.errors] == ["registry"]

    @pytest.mark.parametrize(
        "missing",
        [frozenset({"initramfs"}), frozenset({"kernel", "initramfs"})],
        ids=["one", "two"],
    )
    async def test_prefetch_reports_each_failed_download(self, folders: Folders, missing: frozenset[str]) -> None:
        """Files the release does not have fail; the others are still fetched. The result
        names each failure and has no cache folder."""
        release_all(folders, "x86_64", without=missing)

        result = await prefetch_all_assets(arch="x86_64")

        assert (result.success, result.cache_dir) == (False, None)
        assert {name for name, _ in result.errors} == missing
        assert (
            set(result.downloaded)
            == {
                "kernel",
                "initramfs",
                "gvproxy",
                "python base",
                "javascript base",
                "raw base",
            }
            - missing
        )

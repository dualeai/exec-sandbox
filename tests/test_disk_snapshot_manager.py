"""Tests for DiskSnapshotManager: the L2 cache (local qcow2 layers), the L3 cache (S3), and the
build of a layer. Most go through check_cache and get_or_create_snapshot; the build's VM is a
stub (no VM boots), but for one test that needs hardware acceleration.
"""

import asyncio
import errno
import gc
import hashlib
import logging
import os
import re
import shutil
import stat
import sys
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from botocore.loaders import Loader

# Use native zstd (Python 3.14+) or backports.zstd
if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

from exec_sandbox import constants
from exec_sandbox._imports import _api_models
from exec_sandbox.disk_snapshot_manager import DiskSnapshotManager
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InternalError,
    InvalidInputError,
    OutcomeUnknownError,
    TransientError,
)
from exec_sandbox.files.locks import CacheEntry, EntryLease
from exec_sandbox.guest_agent_protocol import StreamingErrorMessage
from exec_sandbox.guest_channel import StreamDeadlineExceededError, StreamResult
from exec_sandbox.hash_utils import fingerprint
from exec_sandbox.images import BASE_IMAGES, GuestImages, image_digest
from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager
from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostArch
from exec_sandbox.qemu_vm import QemuDiagnostics
from exec_sandbox.settings import Settings
from exec_sandbox.subprocess_utils import communicate_managed_process
from exec_sandbox.vm_manager import VmManager
from tests.s3_faults import BUCKET, Hold

from .conftest import create_test_qcow2, skip_unless_hwaccel
from .instance_support import NO_VM_TIMEOUT
from .test_files_locks import evictable, leased_here, leased_in_child

_VERSION = "1.22.3"
"""The library version while a test runs (make_keyed_manager): a key holds its major.minor."""

_PYTHON_BASE = b"python base image"
"""The python base image of make_keyed_manager. SHA-256 (shasum -a 256):
6dd92b6803cad3cd9acb764c9dc1ac30dfe0a0e4de6f608f84aebab9c06e4d90."""

_NODE_BASE = b"node base image"
"""The javascript base image of make_keyed_manager."""

_PACKAGES = ["requests==2.31.0", "numpy==1.26.0"]

_KEY = (
    "python-v1.22-6dd92b6803cad3cd9acb764c9dc1ac30dfe0a0e4de6f608f84aebab9c06e4d90"
    "-f22d59c83f988d549e792451b3b5d20807f43532befdc983ef4671596cc14f05"
)
"""The L2 key of python _PACKAGES on make_keyed_manager, computed by hand (shasum -a 256): the
language, v and the library major.minor, the SHA-256 of the base image, then the SHA-256 of the
sorted packages, each after its byte length and a colon ("13:numpy==1.26.016:requests==2.31.0")."""

_BASE_KEY = "python-v1.22-6dd92b6803cad3cd9acb764c9dc1ac30dfe0a0e4de6f608f84aebab9c06e4d90-base"
"""The L2 key of the python base image with no packages."""

_L3_OBJECT = f"snapshots/{_KEY}.qcow2.zst"
"""The S3 object of the _KEY layer (the default prefix, snapshots/)."""


@pytest.fixture
async def make_keyed_manager(make_vm_settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """(DiskSnapshotManager, Settings) whose keys the tests know (_KEY, _BASE_KEY): the library
    version is _VERSION and the base images hold _PYTHON_BASE and _NODE_BASE. Its VmManager is
    not started: a lookup, a download and an upload use only its images, and a start (host
    probes, qemu-storage-daemon) costs ~0.15 s per test. The manager makes its cache folder;
    each is stopped at teardown (its uploads awaited), as Scheduler does."""
    monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.__version__", _VERSION)
    images = tmp_path / "images"
    images.mkdir()

    stack = AsyncExitStack()

    async def _make(**overrides: Any) -> tuple[DiskSnapshotManager, Settings]:
        settings = make_vm_settings(
            **{
                "base_images_dir": images,
                "kernel_path": images,
                "disk_snapshot_cache_dir": tmp_path / "cache",
                **overrides,
            }
        )
        vm_manager = VmManager(settings)
        for language, content in ((Language.PYTHON, _PYTHON_BASE), (Language.JAVASCRIPT, _NODE_BASE)):
            image = images / BASE_IMAGES[language].file_name(vm_manager.arch)
            if not image.exists():
                image.write_bytes(content)
        manager = DiskSnapshotManager(settings, vm_manager)
        stack.push_async_callback(manager.stop)
        return manager, settings

    async with stack:
        yield _make


async def _checked(manager: DiskSnapshotManager, language: Language, packages: list[str]) -> Path | None:
    """check_cache of `language` and `packages`: the image path on a hit, else None. The lease
    goes after."""
    lease = await manager.check_cache(language, packages)
    if lease is None:
        return None
    lease.release()
    return lease.path


def entry_bytes(entry: Path) -> int:
    """Disk blocks of an L2 entry and its digest sidecar, in bytes (what an eviction gives back)."""
    sidecar = entry.with_name(f"{entry.name}.sha256")
    return sum(p.stat().st_blocks * 512 for p in (entry, sidecar) if p.exists())


async def _built_path(snapshot: Awaitable[EntryLease]) -> Path:
    """The path of a snapshot that get_or_create_snapshot leased; the lease goes (no VM uses it here)."""
    lease = await snapshot
    lease.release()
    return lease.path


def _entries(cache_dir: Path) -> list[str]:
    """The cache folder listing, without the lock files of builds and leases."""
    return sorted(p.name for p in cache_dir.iterdir() if not p.name.endswith(".lock"))


async def _until(condition: Callable[[], bool], *, what: str) -> None:
    """Until `condition` holds (5 s at most): order by fact, not by sleep."""
    for _ in range(5000):
        if condition():
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"never: {what}")


async def _never_ends(*_args: object) -> None:
    """A wait that never ends (the QEMU of a fake build VM does not die)."""
    await asyncio.Event().wait()


def _enospc() -> OSError:
    """The error of a full disk."""
    return OSError(errno.ENOSPC, "No space left on device")


def _fake_builds(
    manager: DiskSnapshotManager, monkeypatch: pytest.MonkeyPatch, *, content: bytes, failures: list[BaseException]
) -> SimpleNamespace:
    """Layer builds of `manager` with no QEMU, faked at its seams. Its VmManager's ephemeral_vm
    yields `vm`, whose QEMU exits 0 when the install ends; the first starts raise `failures`,
    one each (create_vm lets the OSError of the VM working folder through). qemu-img create
    writes `content` as the image (other qemu-img commands run); the package install answers
    exit 0 (consume_stream); the cleanup of the QEMU process does nothing. Returns
    SimpleNamespace(vm, starts): a test may change `vm`; `starts` gets one item per start."""
    pending = list(failures)
    starts: list[None] = []
    vm = SimpleNamespace(
        vm_id="fake",
        process=SimpleNamespace(returncode=0, wait=_never_ends),  # exited cleanly
        channel=SimpleNamespace(connect=AsyncMock()),
        deadline_diagnostics=AsyncMock(return_value={}),
        raise_if_host_disk_full=AsyncMock(),
    )

    async def qemu_img(cmd: list[str], **kwargs: Any) -> tuple[int | None, bytes, bytes]:
        if cmd[:2] == ["qemu-img", "create"]:
            Path(cmd[-2]).write_bytes(content)
            return 0, b"", b""
        return await communicate_managed_process(cmd, **kwargs)

    @asynccontextmanager
    async def ephemeral_vm(*_args: object, **_kwargs: object) -> AsyncIterator[SimpleNamespace]:
        starts.append(None)
        if pending:
            raise pending.pop(0)
        yield vm

    monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.communicate_managed_process", qemu_img)
    monkeypatch.setattr(manager.vm_manager, "ephemeral_vm", ephemeral_vm)
    monkeypatch.setattr(
        "exec_sandbox.disk_snapshot_manager.consume_stream",
        AsyncMock(
            return_value=StreamResult(
                exit_code=0, stdout="", stderr="", execution_time_ms=None, spawn_ms=None, process_ms=None
            )
        ),
    )
    monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.cleanup_process", AsyncMock())
    return SimpleNamespace(vm=vm, starts=starts)


class _BuildStartedError(Exception):
    """Raised by the start of the build VM of _lookup: an L3 miss started a build."""


async def _lookup(manager: DiskSnapshotManager) -> Path | None:
    """get_or_create_snapshot of python _PACKAGES: the image on an L2 or L3 hit, None when the
    lookup started a build (its VM start raises _BuildStartedError: no VM boots here)."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        _fake_builds(manager, monkeypatch, content=b"built", failures=[_BuildStartedError()])
        try:
            return await _built_path(
                manager.get_or_create_snapshot(Language.PYTHON, _PACKAGES, tenant_id="t", task_id="t", memory_mb=256)
            )
        except InternalError as e:
            if isinstance(e.__cause__, _BuildStartedError):
                return None
            raise


async def _build(manager: DiskSnapshotManager, packages: list[str], content: bytes) -> Path:
    """get_or_create_snapshot of python `packages` with nothing in L2 or L3: the build (_fake_builds:
    no VM, `content` is the image) and its upload, which stop() waits for."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        _fake_builds(manager, monkeypatch, content=content, failures=[])
        path = await _built_path(
            manager.get_or_create_snapshot(Language.PYTHON, packages, tenant_id="t", task_id="t", memory_mb=256)
        )
    await manager.stop()
    return path


_DEADLINE_DIAGNOSTICS = {"console_log": "[WARN] last line", "qemu_status": "running", "qemu_cpu_seconds": 2.0}
"""What QemuVM.deadline_diagnostics adds to the context of a host-deadline error."""


def _spy_unlink(monkeypatch: pytest.MonkeyPatch, *, when: Callable[[Path], bool]) -> list[int]:
    """Path.unlink as usual; returns the thread ids of the calls on the paths `when` accepts."""
    removers: list[int] = []
    unlink = Path.unlink

    def spied(self: Path, missing_ok: bool = False) -> None:
        if when(self):
            removers.append(threading.get_ident())
        unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", spied)
    return removers


def _l3_miss_reasons(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The reasons of the "L3 cache miss" warnings of the downloads."""
    return [r.reason for r in caplog.records if r.getMessage() == "L3 cache miss"]  # type: ignore[attr-defined]


# ============================================================================
# Cache keys
# ============================================================================


class TestFingerprint:
    """fingerprint, the package part of a key: tenants choose the package specs."""

    def test_no_separator_ambiguity(self) -> None:
        """Splitting the same characters differently gives different keys (length-prefixed parts)."""
        assert fingerprint("ab", "c") != fingerprint("a", "bc")
        assert fingerprint("pkg>=1,<2") != fingerprint("pkg>=1", "<2")

    def test_chosen_collision_resistant(self) -> None:
        """A spec forged to collide with a package set under a CRC key gives another fingerprint."""
        victim = sorted(["pandas==2.2.3", "requests==2.32.3", "scipy==1.14.1"])
        forged = ["numpy>=0.206762322200000000000000000000000"]
        assert fingerprint(*victim) != fingerprint(*forged)


class TestCacheKey:
    """The key names the L2 entry and its S3 object (shared by every host): the language, the
    library major.minor, the content of the base image, and the package set. Seen through
    check_cache: a hit gives the entry file it found."""

    async def test_entry_of_packages_is_found_by_its_key_in_any_order(self, make_keyed_manager) -> None:
        manager, settings = await make_keyed_manager()
        entry = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        await create_test_qcow2(entry)

        assert await _checked(manager, Language.PYTHON, _PACKAGES) == entry
        assert await _checked(manager, Language.PYTHON, list(reversed(_PACKAGES))) == entry

    async def test_entry_without_packages_is_the_base_key(self, make_keyed_manager) -> None:
        manager, settings = await make_keyed_manager()
        entry = settings.disk_snapshot_cache_dir / f"{_BASE_KEY}.qcow2"
        await create_test_qcow2(entry)

        assert await _checked(manager, Language.PYTHON, []) == entry

    @pytest.mark.parametrize(
        ("language", "packages"),
        [
            (Language.PYTHON, ["requests==2.31.0"]),
            (Language.PYTHON, [*_PACKAGES, "flask==3.0.0"]),
            (Language.PYTHON, []),
            (Language.JAVASCRIPT, _PACKAGES),
        ],
        ids=["fewer-packages", "more-packages", "no-packages", "other-language"],
    )
    async def test_other_packages_or_language_is_a_miss(
        self, make_keyed_manager, language: Language, packages: list[str]
    ) -> None:
        """Another package set or language never gets the entry of _PACKAGES, which stays."""
        manager, settings = await make_keyed_manager()
        entry = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        await create_test_qcow2(entry)

        assert await _checked(manager, language, packages) is None
        assert entry.exists()

    async def test_rebuilt_base_image_is_a_miss(self, make_keyed_manager) -> None:
        """The key holds the SHA-256 of the base image bytes, the same on every host: new bytes
        give a new key (an entry built on the old image is never served); a new mtime with the
        same bytes keeps the key."""
        manager, settings = await make_keyed_manager()
        entry = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        await create_test_qcow2(entry)
        base = manager.vm_manager.images.base_image(Language.PYTHON)

        st = base.stat()
        os.utime(base, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert await _checked(manager, Language.PYTHON, _PACKAGES) == entry

        base.write_bytes(b"python base image, rebuilt")
        assert await _checked(manager, Language.PYTHON, _PACKAGES) is None
        rebuilt = settings.disk_snapshot_cache_dir / (
            "python-v1.22-54f1e4e67720e275bccab6754b475cdfafccb00b80f79610159d6d7d2ba8e766"
            "-f22d59c83f988d549e792451b3b5d20807f43532befdc983ef4671596cc14f05.qcow2"
        )
        await create_test_qcow2(rebuilt)
        assert await _checked(manager, Language.PYTHON, _PACKAGES) == rebuilt

    @pytest.mark.parametrize(
        ("version", "hit"),
        [("1.22.9", True), ("1.22.3.dev4", True), ("1.23.0", False), ("2.22.3", False)],
    )
    async def test_library_major_minor_is_in_the_key(
        self, make_keyed_manager, monkeypatch: pytest.MonkeyPatch, version: str, *, hit: bool
    ) -> None:
        """A patch release keeps the cache; a minor or major release starts a new one."""
        manager, settings = await make_keyed_manager()
        entry = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        await create_test_qcow2(entry)
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.__version__", version)

        assert (await _checked(manager, Language.PYTHON, _PACKAGES) == entry) is hit


# ============================================================================
# Integration Tests - Require QEMU + Images
# ============================================================================


class TestDiskSnapshotManagerIntegration:
    """Integration tests for DiskSnapshotManager with real QEMU VMs."""

    @skip_unless_hwaccel
    async def test_create_snapshot(self, make_snapshot_manager, tmp_path: Path) -> None:
        """Create snapshot with packages, verify L2 cache hit on second call.

        Requires hwaccel: boots a VM to pip-install packages, then shuts down
        and commits the overlay to a standalone qcow2.  Under TCG the combined
        boot + install wall-clock exceeds CI job timeouts (observed 60 min hang
        on linux/x64 + linux/arm64 sudo runners).
        """

        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        # Create snapshot (this boots a VM and installs packages)
        snapshot_path = await _built_path(
            snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["requests==2.31.0"],
                tenant_id="test",
                task_id=f"test-{uuid4().hex}",
                memory_mb=256,
            )
        )

        assert snapshot_path.exists()
        assert snapshot_path.suffix == ".qcow2"

        # Second call should hit L2 cache
        cached_path = await _built_path(
            snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["requests==2.31.0"],
                tenant_id="test",
                task_id=f"test-{uuid4().hex}",
                memory_mb=256,
            )
        )

        assert cached_path == snapshot_path

    def test_base_image_is_the_catalog_version(self, tmp_path: Path) -> None:
        """An older build in the folder (python-3.13) is not used: the base image is the one the
        images catalog names (one place for a new runtime version)."""
        images = tmp_path / "images"
        images.mkdir()
        host = GuestImages(images_dir=images, kernel_dir=images, download_dir=None, arch=HostArch.X86_64)
        (images / "python-3.13-base-x86_64.qcow2").touch()
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND):
            host.base_image(Language.PYTHON)
        (images / "python-3.14-base-x86_64.qcow2").touch()
        assert host.base_image(Language.PYTHON) == images / "python-3.14-base-x86_64.qcow2"

    @pytest.mark.parametrize(
        ("host_arch", "suffix"), [(HostArch.AARCH64, "aarch64"), (HostArch.X86_64, "x86_64")], ids=["aarch64", "x86_64"]
    )
    @pytest.mark.parametrize("language", list(Language), ids=[language.value for language in Language])
    def test_base_image_skips_the_other_architecture(
        self, tmp_path: Path, language: Language, host_arch: HostArch, suffix: str
    ) -> None:
        """A folder with both architectures (make build IMAGE_ARCH=all) gives the host's image,
        and the overlay pool fills for the same one; a download in progress (.zst) is no image."""
        images = tmp_path / "images"
        images.mkdir()
        stem = {Language.PYTHON: "python-3.14-base", Language.JAVASCRIPT: "node-1.4-base", Language.RAW: "raw-base"}[
            language
        ]
        for arch in ("aarch64", "x86_64"):
            (images / f"{stem}-{arch}.qcow2").touch()
        host = GuestImages(images_dir=images, kernel_dir=images, download_dir=None, arch=host_arch)
        mine = images / f"{stem}-{suffix}.qcow2"
        (images / f"{mine.name}.zst").touch()
        assert host.base_image(language) == mine
        assert host.present_base_images() == [mine]


# ============================================================================
# L2 Cache Tests - Local Disk Snapshots
# ============================================================================


class TestL2Cache:
    """Tests for L2 (local qcow2) cache operations."""

    async def test_l2_hit_keeps_mtime_so_digest_is_not_recomputed(self, make_keyed_manager) -> None:
        """An L2 hit bumps atime only: image_digest keys its cache on mtime, so a hit must not force a re-read."""
        snapshot_manager, settings = await make_keyed_manager()
        snapshot_path = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        await create_test_qcow2(snapshot_path)
        os.utime(snapshot_path, ns=(1, 2))
        digest = image_digest(snapshot_path)

        assert await _checked(snapshot_manager, Language.PYTHON, _PACKAGES) == snapshot_path

        stat = snapshot_path.stat()
        assert stat.st_mtime_ns == 2
        assert stat.st_atime_ns > 1
        with patch("exec_sandbox.images.hashlib.file_digest", side_effect=AssertionError("re-read")):
            assert image_digest(snapshot_path) == digest

    async def test_l2_check_runs_once_per_file_version(self, make_keyed_manager) -> None:
        """A passed qemu-img check is kept for the process: the next packages request on the
        same L2 runs no qemu-img. A changed file (new mtime) is checked again."""
        snapshot_manager, settings = await make_keyed_manager()
        snapshot_path = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        await create_test_qcow2(snapshot_path)

        with patch(
            "exec_sandbox.disk_snapshot_manager.communicate_managed_process", side_effect=communicate_managed_process
        ) as qemu_img:
            assert await _checked(snapshot_manager, Language.PYTHON, _PACKAGES) == snapshot_path
            assert await _checked(snapshot_manager, Language.PYTHON, _PACKAGES) == snapshot_path
            assert qemu_img.await_count == 1

            os.utime(snapshot_path, ns=(1, 2))  # rewritten in place
            assert await _checked(snapshot_manager, Language.PYTHON, _PACKAGES) == snapshot_path
            assert qemu_img.await_count == 2

    async def test_l2_check_that_did_not_run_is_not_kept(self, make_keyed_manager) -> None:
        """qemu-img could not start now (a process limit): a miss, and the next request runs the
        check. A qemu-img this user cannot run is HOST_DEPENDENCY (test_missing_qemu_img_is_a_config_error)."""
        snapshot_manager, settings = await make_keyed_manager()
        snapshot_path = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        await create_test_qcow2(snapshot_path)
        runs: list[str] = []

        async def qemu_img_fails_once(*args: Any, **kwargs: Any) -> tuple[int | None, bytes, bytes]:
            runs.append("qemu-img")
            if len(runs) == 1:
                raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
            return await communicate_managed_process(*args, **kwargs)

        with patch("exec_sandbox.disk_snapshot_manager.communicate_managed_process", side_effect=qemu_img_fails_once):
            assert await _checked(snapshot_manager, Language.PYTHON, _PACKAGES) is None
            assert not leased_here(snapshot_path)  # the miss let its lease go
            assert evictable(snapshot_path)
            assert await _checked(snapshot_manager, Language.PYTHON, _PACKAGES) == snapshot_path

        assert runs == ["qemu-img", "qemu-img"]

    async def test_evict_removes_digest_sidecar(self, make_snapshot_manager, tmp_path: Path) -> None:
        """An eviction removes the entry and its digest sidecar (.sha256), which image_digest wrote."""
        snapshot_manager, settings = await make_snapshot_manager(disk_snapshot_cache_dir=tmp_path / "cache")
        snapshot_path = settings.disk_snapshot_cache_dir / "python-evict.qcow2"
        await create_test_qcow2(snapshot_path)
        image_digest(snapshot_path)
        sidecar = snapshot_path.with_name("python-evict.qcow2.sha256")
        assert sidecar.exists()

        await snapshot_manager.evict_oldest()

        assert not snapshot_path.exists()
        assert not sidecar.exists()

    async def test_start_removes_parts_of_dead_writers_only(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """.part files of a killed process are removed at start, in a worker thread (a listing
        with no size limit, GBs per part); a live writer's are kept."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(mode=0o700)
        dead = cache_dir / "python-x.qcow2.999999999.0123abcd.part"
        live = cache_dir / f"python-y.qcow2.{os.getpid()}.0123abcd.part"
        dead.write_bytes(b"orphan")
        live.write_bytes(b"in progress")

        snapshot_manager, _settings = await make_snapshot_manager(disk_snapshot_cache_dir=cache_dir)
        assert dead.exists()  # the constructor does no file work
        removers = _spy_unlink(monkeypatch, when=lambda path: path.parent == cache_dir)
        await snapshot_manager.start()

        assert not dead.exists()
        assert live.exists()
        assert removers
        assert threading.get_ident() not in removers

    async def test_start_keeps_parts_without_a_writer_pid(self, make_snapshot_manager, tmp_path: Path) -> None:
        """A .part name with no pid field (too few fields, or not a number there) names no writer:
        start() keeps it and goes on to the other parts."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(mode=0o700)
        for name in ("stray.part", "python-z.qcow2.notapid.0123abcd.part", "python-x.qcow2.999999999.0123abcd.part"):
            (cache_dir / name).write_bytes(b"x")

        snapshot_manager, _settings = await make_snapshot_manager(disk_snapshot_cache_dir=cache_dir)
        await snapshot_manager.start()

        assert sorted(p.name for p in cache_dir.iterdir()) == ["python-z.qcow2.notapid.0123abcd.part", "stray.part"]

    async def test_l2_cache_removes_corrupt_snapshot(self, make_keyed_manager) -> None:
        """L2 cache detects and removes corrupt qcow2 (fails qemu-img check)."""
        snapshot_manager, settings = await make_keyed_manager()
        snapshot_path = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        snapshot_path.write_bytes(b"not a valid qcow2 file")
        sidecar = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2.sha256"
        sidecar.write_text("stamp digest")

        result = await _checked(snapshot_manager, Language.PYTHON, _PACKAGES)

        assert result is None, "Should return None for corrupt qcow2"
        assert not snapshot_path.exists(), "Corrupt snapshot should be removed"
        assert not sidecar.exists(), "Its digest sidecar goes with it"

    async def test_l2_cache_empty_is_a_miss(self, make_keyed_manager) -> None:
        """An empty cache is a miss, with no lease and no lock file."""
        snapshot_manager, settings = await make_keyed_manager()

        assert await snapshot_manager.check_cache(Language.PYTHON, _PACKAGES) is None
        assert list(settings.disk_snapshot_cache_dir.iterdir()) == []

    async def test_evict_oldest_removes_the_oldest_by_atime(self, make_snapshot_manager, tmp_path: Path) -> None:
        """evict_oldest removes oldest file by atime."""

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        # Create multiple snapshots
        oldest_path = settings.disk_snapshot_cache_dir / "python-oldest.qcow2"
        newest_path = settings.disk_snapshot_cache_dir / "python-newest.qcow2"

        # Create both files
        for path in [oldest_path, newest_path]:
            await create_test_qcow2(path)

        # Explicitly set atimes to ensure deterministic ordering
        # (avoids relying on filesystem atime behavior which varies across platforms,
        # especially on macOS APFS which uses relatime semantics)
        now = time.time()
        os.utime(oldest_path, (now - 100, now))  # atime=100s ago, mtime=now
        os.utime(newest_path, (now, now))  # atime=now, mtime=now

        assert oldest_path.exists()
        assert newest_path.exists()

        # Evict oldest
        assert await snapshot_manager.evict_oldest() is not None

        # Oldest should be removed, newest should remain
        assert not oldest_path.exists()
        assert newest_path.exists()

        # Admission evicts until this says no entry is left
        assert await snapshot_manager.evict_oldest() is not None
        assert not newest_path.exists()
        assert await snapshot_manager.evict_oldest() is None


# ============================================================================
# L3 Cache Tests - S3 (using moto)
# ============================================================================


@NO_VM_TIMEOUT
class TestL3Cache:
    """The L3 cache (S3, moto server mode), through get_or_create_snapshot with an empty L2: a
    download on an L3 hit (_lookup), an upload after a build (_build). The build is a stub."""

    async def test_managers_share_api_models(
        self, make_keyed_manager, moto_s3_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manager makes no botocore loader of its own: a new one lists every service folder
        and parses the S3 model again (~0.2 s per manager)."""
        endpoint_url, _ = moto_s3_server
        loaders = 0
        new_loader = Loader.__init__

        def counted(self: Loader, *args: object, **kwargs: object) -> None:
            nonlocal loaders
            loaders += 1
            new_loader(self, *args, **kwargs)  # pyright: ignore[reportArgumentType]

        async def look_up(name: str) -> None:
            manager, _ = await make_keyed_manager(
                disk_snapshot_cache_dir=tmp_path / name,
                s3_bucket=BUCKET,
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            assert await _lookup(manager) is None  # an L3 miss: S3 answered

        await look_up("first")  # makes the loader of the process, if no earlier test did
        search_paths = list(_api_models().search_paths)
        monkeypatch.setattr(Loader, "__init__", counted)
        await look_up("second")
        assert loaders == 0
        assert _api_models().search_paths == search_paths  # boto3 adds its data folder once, not per manager

    async def test_no_bucket_touches_no_s3(self, make_keyed_manager, s3_proxy) -> None:
        """Without s3_bucket, the lookup skips L3 and the build is not uploaded: no request
        reaches the S3 endpoint, though one is set."""
        snapshot_manager, settings = await make_keyed_manager(s3_bucket=None, s3_endpoint_url=s3_proxy.endpoint_url)

        built = await _build(snapshot_manager, _PACKAGES, b"image")

        assert s3_proxy.log == []
        assert built == settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        assert _entries(settings.disk_snapshot_cache_dir) == [built.name]

    async def test_upload_to_s3_success(self, make_keyed_manager, moto_s3_server) -> None:
        """A new build goes to S3 zstd-compressed, with the SHA-256 of its uncompressed bytes in
        the object metadata; no compressed part stays."""
        endpoint_url, s3_sync = moto_s3_server
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )

        await _build(snapshot_manager, _PACKAGES, b"fake qcow2 content")

        uploaded = s3_sync.get_object(Bucket=BUCKET, Key=_L3_OBJECT)
        assert zstd.decompress(uploaded["Body"].read()) == b"fake qcow2 content"
        assert uploaded["Metadata"]["sha256"] == hashlib.sha256(b"fake qcow2 content").hexdigest()
        assert _entries(settings.disk_snapshot_cache_dir) == [f"{_KEY}.qcow2"]

    async def test_download_from_s3_success(self, make_keyed_manager, moto_s3_server, tmp_path: Path) -> None:
        """An L3 hit is decompressed into L2 under its key, and no build runs."""
        endpoint_url, s3_sync = moto_s3_server
        source = tmp_path / "source.qcow2"
        await create_test_qcow2(source)
        original_content = source.read_bytes()
        s3_sync.put_object(
            Bucket=BUCKET,
            Key=_L3_OBJECT,
            Body=zstd.compress(original_content),
            Metadata={"sha256": hashlib.sha256(original_content).hexdigest()},
        )
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )

        result = await _lookup(snapshot_manager)

        assert result == settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        assert result.read_bytes() == original_content

    async def test_l3_files_have_no_group_or_other_write(
        self, make_keyed_manager, moto_s3_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under `sg qemu-vm` and umask 002, open() makes files 0664 of group qemu-vm. A layer
        from S3 is 0644 (QEMU as qemu-vm reads it, every later VM boots with it); the
        compressed parts of a download and of an upload are 0600 (only this process reads them)."""
        endpoint_url, s3_sync = moto_s3_server
        source = tmp_path / "source.qcow2"
        await create_test_qcow2(source)
        content = source.read_bytes()
        s3_sync.put_object(
            Bucket=BUCKET,
            Key=_L3_OBJECT,
            Body=zstd.compress(content),
            Metadata={"sha256": hashlib.sha256(content).hexdigest()},
        )
        snapshot_manager, _settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )
        part_modes: list[int] = []

        def compressed_part(path: Path) -> bool:
            """Notes the mode of a compressed part just before it goes."""
            if ".qcow2.zst." in path.name and path.exists():
                part_modes.append(stat.S_IMODE(path.stat().st_mode))
            return False

        _spy_unlink(monkeypatch, when=compressed_part)
        old = os.umask(0o002)
        try:
            layer = await _lookup(snapshot_manager)
            assert layer is not None
            await _build(snapshot_manager, ["other==1.0"], content)
        finally:
            os.umask(old)

        assert layer.read_bytes() == content
        assert stat.S_IMODE(layer.stat().st_mode) == 0o644
        assert part_modes == [0o600, 0o600, 0o600]  # the hit's download, the miss's download, the upload

    @pytest.mark.parametrize("verified", [True, False], ids=["verified", "wrong-digest"])
    async def test_download_removes_its_parts_off_the_loop(
        self,
        make_keyed_manager,
        moto_s3_server,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        caplog: pytest.LogCaptureFixture,
        verified: bool,
    ) -> None:
        """The downloaded .zst and the partial image (each up to GBs) are removed in worker
        threads, after a success and after a failure; no part stays."""
        endpoint_url, s3_sync = moto_s3_server
        source = tmp_path / "source.qcow2"
        await create_test_qcow2(source)
        content = source.read_bytes()
        s3_sync.put_object(
            Bucket=BUCKET,
            Key=_L3_OBJECT,
            Body=zstd.compress(content),
            Metadata={"sha256": hashlib.sha256(content).hexdigest() if verified else "0" * 64},
        )
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )
        removers = _spy_unlink(monkeypatch, when=lambda path: path.name.endswith(".part"))

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            result = await _lookup(snapshot_manager)
        if verified:
            assert result is not None
            assert result.read_bytes() == content
        else:
            assert result is None
            assert len(_l3_miss_reasons(caplog)) == 1
            assert "digest mismatch" in _l3_miss_reasons(caplog)[0]

        assert removers
        assert threading.get_ident() not in removers
        assert _entries(settings.disk_snapshot_cache_dir) == ([f"{_KEY}.qcow2"] if verified else [])

    async def test_download_from_s3_not_found(
        self, make_keyed_manager, moto_s3_server, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing object is an L3 miss (logged): the build runs, no error."""
        endpoint_url, _s3_sync = moto_s3_server
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await _lookup(snapshot_manager) is None
        assert len(_l3_miss_reasons(caplog)) == 1
        assert _entries(settings.disk_snapshot_cache_dir) == []

    @pytest.mark.parametrize(
        ("metadata", "reason"),
        [
            pytest.param({}, f"S3 object {_L3_OBJECT} has no sha256 metadata", id="no-digest"),
            pytest.param(
                {"sha256": "0" * 64},
                f"S3 snapshot digest mismatch for {_L3_OBJECT}: expected {'0' * 64}, got "
                "31209cf4dce75b9eda041796674f0a1e690fa781454a0ee40d49097093cf11d8",  # shasum -a 256 of the body
                id="wrong-digest",
            ),
        ],
    )
    async def test_download_from_s3_rejects_unverified(
        self,
        make_keyed_manager,
        moto_s3_server,
        *,
        caplog: pytest.LogCaptureFixture,
        metadata: dict[str, str],
        reason: str,
    ) -> None:
        """A missing or wrong sha256 digest is an L3 miss and leaves no file in L2."""
        endpoint_url, s3_sync = moto_s3_server
        s3_sync.put_object(
            Bucket=BUCKET, Key=_L3_OBJECT, Body=zstd.compress(b"tampered qcow2 content"), Metadata=metadata
        )
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await _lookup(snapshot_manager) is None
        assert _l3_miss_reasons(caplog) == [reason]
        assert _entries(settings.disk_snapshot_cache_dir) == []

    async def test_download_from_s3_rejects_truncated_stream(
        self, make_keyed_manager, moto_s3_server, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A cut zstd stream fails the download, even if its partial output matches the digest."""
        endpoint_url, s3_sync = moto_s3_server
        original_content = os.urandom(256 * 1024)
        compressed = zstd.compress(original_content)
        s3_sync.put_object(
            Bucket=BUCKET,
            Key=_L3_OBJECT,
            Body=compressed[: len(compressed) // 2],
            Metadata={"sha256": hashlib.sha256(original_content).hexdigest()},
        )
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await _lookup(snapshot_manager) is None
        assert _l3_miss_reasons(caplog) == ["S3 snapshot stream is truncated"]
        assert _entries(settings.disk_snapshot_cache_dir) == []

    async def test_download_rejects_qcow2_with_backing_file(
        self, make_keyed_manager, moto_s3_server, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A qcow2 that names a host backing file never reaches L2 (QEMU would serve that file to the guest)."""
        endpoint_url, s3_sync = moto_s3_server
        secret = tmp_path / "host-secret"
        secret.write_bytes(b"host secret")
        crafted = tmp_path / "crafted.qcow2"
        proc = await asyncio.create_subprocess_exec(
            "qemu-img", "create", "-f", "qcow2", "-b", str(secret), "-F", "raw", str(crafted), "1M",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
        await proc.communicate()
        assert proc.returncode == 0
        content = crafted.read_bytes()
        s3_sync.put_object(
            Bucket=BUCKET,
            Key=_L3_OBJECT,
            Body=zstd.compress(content),
            Metadata={"sha256": hashlib.sha256(content).hexdigest()},
        )
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await _lookup(snapshot_manager) is None
        assert _l3_miss_reasons(caplog) == ["S3 snapshot names a backing or data file; L2 images are standalone"]
        assert _entries(settings.disk_snapshot_cache_dir) == []

    async def test_download_stops_above_max_size(
        self,
        make_keyed_manager,
        moto_s3_server,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Decompressed output above the largest possible L2 image fails (zstd bomb guard). The
        bound is set to 1 MiB here: at its real value (576 MiB) the test would write that much."""
        endpoint_url, s3_sync = moto_s3_server
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager._L2_MAX_FILE_BYTES", 1024 * 1024)
        content = bytes(2 * 1024 * 1024)  # compresses to a few hundred bytes
        s3_sync.put_object(
            Bucket=BUCKET,
            Key=_L3_OBJECT,
            Body=zstd.compress(content),
            Metadata={"sha256": hashlib.sha256(content).hexdigest()},
        )
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await _lookup(snapshot_manager) is None
        assert _l3_miss_reasons(caplog) == ["S3 snapshot is larger than any L2 image"]
        assert _entries(settings.disk_snapshot_cache_dir) == []

    async def test_download_of_a_chunk_that_gives_several_output_steps(
        self, make_keyed_manager, moto_s3_server, tmp_path: Path
    ) -> None:
        """One 64 KiB compressed chunk whose output is larger than one 4 MiB step: every step is kept."""
        endpoint_url, s3_sync = moto_s3_server
        source = tmp_path / "source.qcow2"
        await create_test_qcow2(source)
        content = source.read_bytes() + bytes(9 * 1024 * 1024)  # trailing zeros: still a qcow2
        compressed = zstd.compress(content)
        assert len(compressed) < 64 * 1024  # the whole object is one read chunk
        s3_sync.put_object(
            Bucket=BUCKET,
            Key=_L3_OBJECT,
            Body=compressed,
            Metadata={"sha256": hashlib.sha256(content).hexdigest()},
        )
        snapshot_manager, _settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )

        result = await _lookup(snapshot_manager)

        assert result is not None
        assert result.read_bytes() == content

    async def test_download_rejects_an_object_that_is_not_qcow2(
        self, make_keyed_manager, moto_s3_server, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A verified object that is not a qcow2 image never reaches L2."""
        endpoint_url, s3_sync = moto_s3_server
        content = os.urandom(256 * 1024)
        s3_sync.put_object(
            Bucket=BUCKET,
            Key=_L3_OBJECT,
            Body=zstd.compress(content),
            Metadata={"sha256": hashlib.sha256(content).hexdigest()},
        )
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await _lookup(snapshot_manager) is None
        assert _l3_miss_reasons(caplog) == ["S3 snapshot is not qcow2 (format=raw)"]
        assert _entries(settings.disk_snapshot_cache_dir) == []

    async def test_upload_to_s3_silent_failure(
        self, make_keyed_manager, moto_s3_server, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An upload that fails (here, no such bucket) is logged, never raised: the build stays
        in L2 and its compressed part goes."""
        endpoint_url, _s3_sync = moto_s3_server
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket="nonexistent-bucket", s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            built = await _build(snapshot_manager, _PACKAGES, b"test content")

        assert [r.getMessage() for r in caplog.records].count("S3 upload failed silently") == 1
        assert built.read_bytes() == b"test content"
        assert _entries(settings.disk_snapshot_cache_dir) == [built.name]

    async def test_upload_semaphore_limits_concurrency(
        self, make_keyed_manager, s3_proxy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Five builds at once: their uploads run two at a time (max_concurrent_s3_uploads=2).

        The S3 proxy holds each upload (PUT): two reach it, and no third comes while they wait;
        once they go on, all five land."""
        snapshot_manager, _settings = await make_keyed_manager(
            s3_bucket=BUCKET,
            s3_region="us-east-1",
            s3_endpoint_url=s3_proxy.endpoint_url,
            s3_snapshot_prefix=s3_proxy.prefix,
            max_concurrent_s3_uploads=2,
        )
        held = s3_proxy.add("PUT", r"\.qcow2\.zst$", Hold(), times=None)
        _fake_builds(snapshot_manager, monkeypatch, content=b"test data", failures=[])

        await asyncio.gather(
            *(
                _built_path(
                    snapshot_manager.get_or_create_snapshot(
                        Language.PYTHON, [f"pkg{i}==1.0"], tenant_id="t", task_id=f"t{i}", memory_mb=256
                    )
                )
                for i in range(5)
            )
        )
        await _until(lambda: held.hits == 2, what="two uploads at once")  # not one at a time
        await asyncio.sleep(0.2)  # a window for a third upload, which the semaphore keeps out
        assert held.hits == 2
        held.release()
        await snapshot_manager.stop()

        assert held.hits == 5
        assert len(s3_proxy.bucket.listing()) == 5


# ============================================================================
# Builds (the build VM is a stub)
# ============================================================================


_REFUSED_OUTPUTS = [
    "ERROR: No matching distribution found for nonexistent-package",
    "ERROR: Could not find a version that satisfies the requirement foo",
    "error: 404 Not Found - GET https://registry.npmjs.org/nonexistent",
    "error: 404 Client Error: Not Found for url: https://pypi.org/simple/nonexistent/",
    "npm ERR! ERESOLVE could not resolve dependency tree",
    "npm ERR! could not resolve dependency lodash@^99.0.0",
    "ERROR: Invalid version specifier: pandas==",
    "npm ERR! 404 Not Found - not found in registry",
    "npm ERR! No matching version found for foo@^99.0.0",
    "pip ERROR: version not found for package foo==99.99.99",
    "HTTP 404 Client Error for url: https://pypi.org/simple/foo/",
    "NO MATCHING DISTRIBUTION found",
    "ERESOLVE could not resolve",
]
"""Package manager output of a refusal: the same packages fail again."""

_NETWORK_OUTPUTS = [
    "error: Failed to download: client error (Connect)",
    "error: connection refused",
    "Connection reset by peer",
    "error: connection timed out",
    "error: Temporary failure in name resolution",
    "error: Name resolution failed for files.pythonhosted.org",
    "error: network unreachable",
    "error: timed out waiting for response",
    "error: reset by peer",
    "error: request failed after 3 retries",
    "CONNECTION REFUSED",
    "Client Error (Connect)",
    "NETWORK UNREACHABLE",
]
"""Package manager output of a network failure: a retry can pass."""

_UNCLASSIFIED_OUTPUTS = [
    "error: package 'foo' has no attribute 'bar'",
    "SyntaxError: invalid syntax",
    "some unknown error occurred",
    "Segmentation fault",
    "",
]
"""Package manager output that names no known cause ("": no output, reported as "Unknown error")."""


class TestBuildRetryAfterDiskFull:
    """A build that hits ENOSPC evicts the oldest entry and retries once; the retry runs after
    the build released the creation semaphore (one build at a time), so it does not wait for itself.
    A build that fails in another way removes its image file.

    Through the public get_or_create_snapshot, with the build VM of _fake_builds. ENOSPC comes
    from the start of the build VM, after its image file exists (an OSError of the qemu-img
    runner is a spawn failure: test_missing_qemu_img_is_a_config_error)."""

    host_disk_full = False
    """What the fake build VM answers to raise_if_host_disk_full (QEMU paused it on a full disk)."""

    _TIMEOUT = 10.0
    """A retry that waits for its own semaphore hangs: the bound turns that into a failure."""

    async def _fake_build(
        self,
        make_keyed_manager,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        start_failures: list[BaseException],
    ) -> tuple[DiskSnapshotManager, Path, Path]:
        """A manager whose build makes a small image (b"built") with no VM (_fake_builds); the
        first starts of the build VM raise `start_failures`, one each. Returns (manager, oldest
        entry, newer entry)."""
        snapshot_manager, settings = await make_keyed_manager(
            disk_snapshot_cache_dir=tmp_path / "cache", s3_bucket=None
        )
        cache_dir = settings.disk_snapshot_cache_dir
        oldest, newer = cache_dir / "python-oldest.qcow2", cache_dir / "python-newer.qcow2"
        for age, entry in ((1000, oldest), (500, newer)):
            entry.write_bytes(b"entry")
            past = time.time() - age
            os.utime(entry, (past, past))

        async def deadline_diagnostics(*, operation: str) -> dict[str, object]:
            assert operation == "install_packages"
            return _DEADLINE_DIAGNOSTICS

        async def raise_if_host_disk_full() -> None:
            if TestBuildRetryAfterDiskFull.host_disk_full:
                raise TransientError(
                    "VM fake stopped: the host disk is full",
                    code=ErrorCode.LOCAL_DISK,
                    context=None,
                    hint=None,
                    instance_id=None,
                )

        builds = _fake_builds(snapshot_manager, monkeypatch, content=b"built", failures=start_failures)
        builds.vm.deadline_diagnostics = deadline_diagnostics
        builds.vm.raise_if_host_disk_full = raise_if_host_disk_full
        return snapshot_manager, oldest, newer

    async def _build(self, snapshot_manager: DiskSnapshotManager) -> Path:
        return await asyncio.wait_for(
            _built_path(
                snapshot_manager.get_or_create_snapshot(Language.PYTHON, [], tenant_id="t", task_id="t", memory_mb=512)
            ),
            self._TIMEOUT,
        )

    async def test_disk_full_once_evicts_once_off_the_loop_and_retries(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The eviction lists a folder with no size limit and removes GBs: worker threads."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[_enospc()]
        )
        removers = _spy_unlink(monkeypatch, when=lambda path: path.name.startswith(oldest.name))

        built = await self._build(snapshot_manager)

        assert built.read_bytes() == b"built"
        assert _entries(oldest.parent) == sorted([newer.name, built.name])  # one eviction: the oldest
        assert removers
        assert threading.get_ident() not in removers

    @staticmethod
    def _wrapped_qemu(
        snapshot_manager: DiskSnapshotManager, monkeypatch: pytest.MonkeyPatch, *, quit_works: bool
    ) -> AsyncMock:
        """The build's VM, with QEMU under a wrapper that ignores SIGTERM (an `unshare --fork`
        parent): its signals end QEMU with SIGKILL (-9); a QMP quit, when it works, ends it
        with 0. Returns the QMP quit mock. The wrapper's signal handling does not come on
        demand: the cleanup stub and the QMP client stand in for it."""
        process = SimpleNamespace(returncode=None, wait=_never_ends)
        quit_qemu = AsyncMock(side_effect=lambda: setattr(process, "returncode", 0))

        @asynccontextmanager
        async def ephemeral_vm(*_: object, **__: object) -> AsyncIterator[SimpleNamespace]:
            yield SimpleNamespace(
                vm_id="fake",
                process=process,
                qmp_socket=Path("/nonexistent/qmp.sock"),
                use_qemu_vm_user=False,
                channel=SimpleNamespace(connect=AsyncMock()),
                deadline_diagnostics=AsyncMock(return_value={}),
                raise_if_host_disk_full=AsyncMock(),
            )

        @asynccontextmanager
        async def qmp(*_: object) -> AsyncIterator[SimpleNamespace]:
            if not quit_works:
                raise TransientError(
                    "QMP connection failed", code=ErrorCode.MIGRATION_FAILED, context={}, hint=None, instance_id=None
                )
            yield SimpleNamespace(quit=quit_qemu)

        async def signals(proc: SimpleNamespace, *_args: object, **_kwargs: object) -> bool:
            if proc.returncode is None:
                proc.returncode = -9  # SIGTERM ignored, then SIGKILL after term_timeout
            return True

        monkeypatch.setattr(snapshot_manager.vm_manager, "ephemeral_vm", ephemeral_vm)
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.MigrationClient", qmp)
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.cleanup_process", signals)
        return quit_qemu

    async def test_a_qemu_under_a_wrapper_that_ignores_sigterm_quits_clean(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QMP quit ends the build's QEMU with 0 before any signal: the layer is published
        (under `unshare --fork`, signals alone end it -9 on every build)."""
        snapshot_manager, _oldest, _newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        quit_qemu = self._wrapped_qemu(snapshot_manager, monkeypatch, quit_works=True)

        built = await self._build(snapshot_manager)

        quit_qemu.assert_awaited_once()
        assert built.read_bytes() == b"built"

    async def test_a_build_whose_qemu_ends_on_sigkill_publishes_no_layer(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No QMP quit (its connect fails), and the signals end QEMU with SIGKILL: the layer it
        did not close is not published, QEMU_CRASHED."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        self._wrapped_qemu(snapshot_manager, monkeypatch, quit_works=False)

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED):
            await self._build(snapshot_manager)
        assert _entries(oldest.parent) == sorted([oldest.name, newer.name])  # no layer published

    async def test_built_layer_has_no_group_or_other_write(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The build target comes from qemu-img (0644 minus the umask), and QEMU as qemu-vm owned
        it during the build (it may change its mode). It goes into the cache 0644: QEMU as
        qemu-vm reads it, none may write it. Here the fake qemu-img makes it 0664 (umask 002)."""
        snapshot_manager, _oldest, _newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        old = os.umask(0o002)
        try:
            built = await self._build(snapshot_manager)
        finally:
            os.umask(old)
        assert built.read_bytes() == b"built"
        assert stat.S_IMODE(built.stat().st_mode) == 0o644

    async def test_disk_full_twice_raises(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second ENOSPC, after the one eviction and retry: TransientError LOCAL_DISK, and no
        second eviction."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[_enospc(), _enospc()]
        )
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.LOCAL_DISK,
            match=r"^Local disk error for snapshot .+: No space left on device$",
        ):
            await self._build(snapshot_manager)
        assert _entries(oldest.parent) == [newer.name]  # one eviction, then no second one; no image, no .part

    async def test_disk_full_and_a_cache_that_cannot_be_listed_is_a_local_disk_error(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The eviction after ENOSPC fails too (the cache folder cannot be listed): the build
        fails with its file error, never a raw OSError. The error comes from evict_oldest: a
        listing that fails with EIO cannot be made on demand (an unreadable folder lists as
        empty)."""
        snapshot_manager, _oldest, _newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[_enospc()]
        )
        monkeypatch.setattr(snapshot_manager, "evict_oldest", AsyncMock(side_effect=OSError(errno.EIO, "I/O error")))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK, match="I/O error"):
            await self._build(snapshot_manager)

    @pytest.mark.parametrize(
        ("failure", "raised", "code", "message"),
        [
            pytest.param(
                RuntimeError("boom"), InternalError, ErrorCode.INTERNAL, "^Failed to create snapshot: boom$", id="other"
            ),
            pytest.param(
                TransientError(
                    "guest gone", code=ErrorCode.GUEST_UNAVAILABLE, context=None, hint=None, instance_id=None
                ),
                TransientError,
                ErrorCode.GUEST_UNAVAILABLE,
                "^guest gone$",
                id="sandbox-error",
            ),
            pytest.param(
                OSError(errno.EIO, "I/O error"),
                TransientError,
                ErrorCode.LOCAL_DISK,
                "^Local disk error for snapshot .+: I/O error$",
                id="os-error",
            ),
            pytest.param(
                OSError(errno.EDQUOT, "Disc quota exceeded"),
                TransientError,
                ErrorCode.LOCAL_DISK,
                "^Local disk error for snapshot .+: Disc quota exceeded$",
                id="quota",
            ),
            pytest.param(
                PermissionError(errno.EACCES, "Permission denied"),
                ConfigError,
                ErrorCode.SETTING_INVALID,
                "^Snapshot build .+ cannot write its files: Permission denied\n"
                "Hint: Give this user write access to disk_snapshot_cache_dir, on a writable disk\\.$",
                id="no-access",
            ),
            pytest.param(
                OSError(errno.EROFS, "Read-only file system"),
                ConfigError,
                ErrorCode.SETTING_INVALID,
                "^Snapshot build .+ cannot write its files: Read-only file system\n"
                "Hint: Give this user write access to disk_snapshot_cache_dir, on a writable disk\\.$",
                id="read-only",
            ),
            pytest.param(
                TimeoutError("slow"),
                InternalError,
                ErrorCode.INTERNAL,
                "^Failed to create snapshot: slow$",
                id="timeout",
            ),
        ],
    )
    async def test_failed_build_removes_its_image_file(
        self,
        make_keyed_manager,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        failure: Exception,
        raised: type[Exception],
        code: ErrorCode | None,
        message: str,
    ) -> None:
        """A build that fails after its image file exists (here, at the start of its VM) removes
        that file: no .part and no entry stay; the other entries are kept. A SandboxError passes
        unchanged; a local OSError is LOCAL_DISK (RETRY), but no access or a read-only disk fails
        the same way again: SETTING_INVALID (FIX_CONFIG); a TimeoutError (an OSError) is not a
        disk error: INTERNAL, like any other failure."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[failure]
        )

        with pytest.raises(raised, check=lambda e: getattr(e, "code", None) is code, match=message):
            await self._build(snapshot_manager)

        assert _entries(oldest.parent) == [newer.name, oldest.name]

    @pytest.mark.parametrize(
        "spawn_error",
        [
            FileNotFoundError(errno.ENOENT, "No such file or directory", "qemu-img"),
            PermissionError(errno.EACCES, "Permission denied", "qemu-img"),
            OSError(errno.ENOEXEC, "Exec format error", "qemu-img"),
        ],
        ids=["missing", "not-executable", "other-arch"],
    )
    async def test_missing_qemu_img_is_a_config_error(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spawn_error: OSError
    ) -> None:
        """No qemu-img on the host (Debian and Ubuntu ship it in qemu-utils), or one this user
        may not run: ConfigError HOST_DEPENDENCY, which no retry fixes, never a local disk error
        that callers retry, nor a cache folder setting."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )

        async def no_qemu_img(cmd: list[str], **_kwargs: object) -> tuple[int, bytes, bytes]:
            raise spawn_error

        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.communicate_managed_process", no_qemu_img)

        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc:
            await self._build(snapshot_manager)
        assert (exc.value.context["binary"], exc.value.context["errno"]) == ("qemu-img", spawn_error.errno)
        assert exc.value.hint == "Install QEMU (on Debian and Ubuntu, the qemu-utils package)."
        assert _entries(oldest.parent) == [newer.name, oldest.name]

    async def test_qemu_img_fork_limit_is_overlay_failed(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A qemu-img fork that hits a process limit can pass later: TransientError
        OVERLAY_FAILED, not a local disk error."""
        snapshot_manager, _oldest, _newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )

        async def fork_limit(cmd: list[str], **_kwargs: object) -> tuple[int, bytes, bytes]:
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")

        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.communicate_managed_process", fork_limit)

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED) as exc:
            await self._build(snapshot_manager)
        assert exc.value.context["errno"] == errno.EAGAIN

    async def test_cache_folder_removed_at_run_time_is_made_again(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache folder removed while the worker runs (a tmp cleaner) is made again at the
        next build, which succeeds: never LOCAL_DISK (RETRY) on every build until a restart."""
        snapshot_manager, oldest, _newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        shutil.rmtree(oldest.parent)

        path = await self._build(snapshot_manager)

        assert path.parent == oldest.parent
        assert path.read_bytes() == b"built"

    @pytest.mark.skipif(os.geteuid() == 0, reason="root writes a 0o500 folder")
    async def test_read_only_cache_folder_is_a_config_error(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache folder this user cannot write fails at the build lock, the first file of a
        build: ConfigError SETTING_INVALID that names the setting, never a raw PermissionError."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        oldest.parent.chmod(0o500)
        try:
            with pytest.raises(
                ConfigError,
                check=lambda e: e.code is ErrorCode.SETTING_INVALID,
                match=r"^Snapshot build .+ cannot write its files: Permission denied\n"
                r"Hint: Give this user write access to disk_snapshot_cache_dir, on a writable disk\.$",
            ):
                await self._build(snapshot_manager)
        finally:
            oldest.parent.chmod(0o700)
        assert _entries(oldest.parent) == [newer.name, oldest.name]

    @pytest.mark.parametrize(
        ("manager", "setting"),
        [
            (DiskSnapshotManager, "disk_snapshot_cache_dir"),
            (MemorySnapshotManager, "memory_snapshot_cache_dir"),
        ],
        ids=["l2", "l1"],
    )
    async def test_cache_folder_that_cannot_be_made_is_a_config_error(
        self, make_vm_manager, tmp_path: Path, manager: type[DiskSnapshotManager | MemorySnapshotManager], setting: str
    ) -> None:
        """A cache folder path under a file: the manager raises ConfigError SETTING_INVALID that
        names its setting when it is made (Scheduler start), never a raw OSError."""
        (tmp_path / "file").write_bytes(b"")
        vm_manager = await make_vm_manager(**{setting: tmp_path / "file" / "cache"})

        with pytest.raises(
            ConfigError,
            check=lambda e: e.code is ErrorCode.SETTING_INVALID,
            match=f"^Cache folder .+ cannot be made: Not a directory\nHint: Set {setting} to a folder",
        ):
            manager(vm_manager.settings, vm_manager)

    @pytest.mark.parametrize(
        "lost",
        [
            pytest.param(
                OutcomeUnknownError(
                    "stopped: the host disk is full",
                    code=ErrorCode.GUEST_OUTCOME_UNKNOWN,
                    context=None,
                    hint=None,
                    instance_id=None,
                ),
                id="in-flight",
            ),
            pytest.param(TimeoutError(), id="no-answer"),
        ],
    )
    async def test_install_on_a_full_host_disk_is_local_disk(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost: Exception
    ) -> None:
        """QEMU paused the build VM on a full host disk: the install fails with LOCAL_DISK
        (free disk space), not GUEST_UNAVAILABLE or PACKAGE_DOWNLOAD (a guest or network fault)."""
        snapshot_manager, _oldest, _newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        monkeypatch.setattr(TestBuildRetryAfterDiskFull, "host_disk_full", True)
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", AsyncMock(side_effect=lost))

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
            await asyncio.wait_for(
                _built_path(
                    snapshot_manager.get_or_create_snapshot(
                        Language.PYTHON, ["pandas==2.2.3"], tenant_id="t", task_id="t", memory_mb=512
                    )
                ),
                self._TIMEOUT,
            )

    async def test_install_outcome_unknown_is_guest_unavailable(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An install whose transport is lost after dispatch ran no user code: the build
        raises TransientError GUEST_UNAVAILABLE (RETRY), never OutcomeUnknownError (VERIFY),
        which would reach the caller of Scheduler.new. Not retried (only PACKAGE_DOWNLOAD is)."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        lost = OutcomeUnknownError(
            "transport lost after dispatch",
            code=ErrorCode.GUEST_OUTCOME_UNKNOWN,
            context=None,
            hint=None,
            instance_id=None,
        )
        consume = AsyncMock(side_effect=lost)
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", consume)

        with pytest.raises(
            TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE and e.__cause__ is lost
        ):
            await asyncio.wait_for(
                _built_path(
                    snapshot_manager.get_or_create_snapshot(
                        Language.PYTHON, ["pandas==2.2.3"], tenant_id="t", task_id="t", memory_mb=512
                    )
                ),
                self._TIMEOUT,
            )

        consume.assert_awaited_once()
        assert _entries(oldest.parent) == [newer.name, oldest.name]

    async def test_cancelled_install_leaves_no_task_of_the_build(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A build cancelled during its package install cancels the install and the watch on
        its VM's process, and waits for both: no task of the build runs on after it."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        installing = asyncio.Event()

        async def install_never_ends(*_args: object, **_kwargs: object) -> StreamResult:
            installing.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", install_never_ends)
        before = asyncio.all_tasks()
        build = asyncio.create_task(
            snapshot_manager.get_or_create_snapshot(
                Language.PYTHON, ["pandas==2.2.3"], tenant_id="t", task_id="t", memory_mb=512
            )
        )
        await installing.wait()
        build.cancel()
        with pytest.raises(asyncio.CancelledError):
            await build

        assert asyncio.all_tasks() - before == set()
        assert _entries(oldest.parent) == [newer.name, oldest.name]  # nor its image file

    async def test_vm_death_as_the_install_fails_leaves_no_unread_error(
        self, make_keyed_manager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The build VM dies and its install fails in the same loop pass: the build raises the
        death (the install failed of it) and reads the install's error too (asyncio logs an
        error no one read)."""
        snapshot_manager, _settings = await make_keyed_manager(s3_bucket=None)
        builds = _fake_builds(snapshot_manager, monkeypatch, content=b"built", failures=[])
        builds.vm.process = SimpleNamespace(returncode=-9, wait=AsyncMock(return_value=-9))
        builds.vm.collect_diagnostics = lambda: QemuDiagnostics(
            vm_id="fake", exit_code=-9, signal_name="SIGKILL", console_log="", accel_type="hvf", host_os="macos"
        )
        lost = OSError(errno.ECONNRESET, "connection reset")
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", AsyncMock(side_effect=lost))
        loop = asyncio.get_running_loop()
        unread: list[dict[str, Any]] = []
        handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: unread.append(context))
        try:
            with pytest.raises(TransientError) as exc:
                await snapshot_manager.get_or_create_snapshot(
                    Language.PYTHON, ["pandas==2.2.3"], tenant_id="t", task_id="t", memory_mb=512
                )
            assert exc.value.code is ErrorCode.QEMU_CRASHED
            del exc
            gc.collect()
        finally:
            loop.set_exception_handler(handler)

        assert unread == []

    async def _install(self, snapshot_manager: DiskSnapshotManager) -> Path:
        """A build with one package: its install goes through consume_stream."""
        return await asyncio.wait_for(
            _built_path(
                snapshot_manager.get_or_create_snapshot(
                    Language.PYTHON, ["pandas==2.2.3"], tenant_id="t", task_id="t", memory_mb=512
                )
            ),
            self._TIMEOUT,
        )

    @staticmethod
    def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
        """Install retries without their backoff sleeps (1 to 8 s each)."""
        monkeypatch.setattr(constants, "PACKAGE_INSTALL_RETRY_MIN_SECONDS", 0)
        monkeypatch.setattr(constants, "PACKAGE_INSTALL_RETRY_MAX_SECONDS", 0)

    async def test_guest_timeout_during_install_is_package_download(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guest reports a timeout of the package manager: TransientError PACKAGE_DOWNLOAD,
        the one code the install retries (3 attempts), then raised; the build file goes."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        self._no_backoff(monkeypatch)

        async def guest_timeout(*_args: object, on_error: Any, **_kwargs: object) -> StreamResult:
            await on_error(StreamingErrorMessage(message="pip timed out", error_type="timeout_error"))
            raise AssertionError("on_error raised nothing")

        consume = AsyncMock(side_effect=guest_timeout)
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", consume)

        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.PACKAGE_DOWNLOAD,
            match=r"^install_packages: \[timeout_error\] pip timed out$",
        ):
            await self._install(snapshot_manager)

        assert consume.await_count == 3
        assert _entries(oldest.parent) == [newer.name, oldest.name]

    @pytest.mark.parametrize(
        ("stderr", "raised", "code", "message", "attempts"),
        [
            *(
                pytest.param(
                    output,
                    InvalidInputError,
                    ErrorCode.PACKAGE_INSTALL_FAILED,
                    rf"^Package installation failed permanently \(exit code 1\): {re.escape(output)}$",
                    1,
                    id=f"refused-{index}",
                )
                for index, output in enumerate(_REFUSED_OUTPUTS)
            ),
            *(
                pytest.param(
                    output,
                    TransientError,
                    ErrorCode.PACKAGE_DOWNLOAD,
                    rf"^Package installation failed with transient network error \(exit code 1\): {re.escape(output)}$",
                    3,
                    id=f"network-{index}",
                )
                for index, output in enumerate(_NETWORK_OUTPUTS)
            ),
            *(
                pytest.param(
                    output,
                    InvalidInputError,
                    ErrorCode.PACKAGE_INSTALL_FAILED,
                    rf"^Package installation failed with exit code 1: {re.escape(output or 'Unknown error')}$",
                    1,
                    id=f"unclassified-{index}",
                )
                for index, output in enumerate(_UNCLASSIFIED_OUTPUTS)
            ),
        ],
    )
    async def test_failed_install_is_classified_by_its_output(
        self,
        make_keyed_manager,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        stderr: str,
        raised: type[Exception],
        code: ErrorCode,
        message: str,
        attempts: int,
    ) -> None:
        """The package manager exits 1: its output decides, case-insensitive. A network failure
        is PACKAGE_DOWNLOAD (RETRY, retried); a refusal or an unknown failure is
        PACKAGE_INSTALL_FAILED (FIX_INPUT, not retried: the same packages fail again). A refusal
        wins over a network word in the same output ("404 Client Error")."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        self._no_backoff(monkeypatch)
        consume = AsyncMock(
            return_value=StreamResult(
                exit_code=1, stdout="", stderr=stderr, execution_time_ms=None, spawn_ms=None, process_ms=None
            )
        )
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", consume)

        with pytest.raises(raised, check=lambda e: getattr(e, "code", None) is code, match=message):
            await self._install(snapshot_manager)

        assert consume.await_count == attempts
        assert _entries(oldest.parent) == [newer.name, oldest.name]

    async def test_install_past_its_total_timeout_is_package_download(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An install still running at its total bound (all attempts and backoffs): TransientError
        PACKAGE_DOWNLOAD (RETRY), not the builtin TimeoutError."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        # One attempt, bound 50 ms: hard_timeout is PACKAGE_INSTALL_TIMEOUT_SECONDS + margin + 8.
        monkeypatch.setattr(constants, "PACKAGE_INSTALL_MAX_RETRIES", 1)
        monkeypatch.setattr(constants, "PACKAGE_INSTALL_TIMEOUT_SECONDS", 0)
        monkeypatch.setattr(constants, "EXECUTION_TIMEOUT_MARGIN_SECONDS", -7.95)

        async def never_ends(*_args: object, **_kwargs: object) -> StreamResult:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", never_ends)
        # The install's clock reads 1000.0 s at its start and 1012.3 s at its deadline: the error
        # gives the time spent (12.3 s), not the 0.05 s bound, whatever the load on the host.
        reads = iter((1000.0, 1012.3))
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.time", SimpleNamespace(monotonic=lambda: next(reads)))

        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.PACKAGE_DOWNLOAD and isinstance(e.__cause__, TimeoutError),
            match=r"^Package installation timeout after 12\.3s$",
        ):
            await self._install(snapshot_manager)

        assert _entries(oldest.parent) == [newer.name, oldest.name]

    async def test_install_stream_deadline_reports_elapsed_time_and_guest_state(
        self, make_keyed_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stream deadline of the first attempt (a host deadline, not retried) ends the
        install long before its total bound: the error gives the time spent, not that bound,
        and the guest console and QEMU state (CI: one 318 s attempt reported as 970 s)."""
        snapshot_manager, _oldest, _newer = await self._fake_build(
            make_keyed_manager, tmp_path, monkeypatch, start_failures=[]
        )
        consume = AsyncMock(side_effect=StreamDeadlineExceededError("total stream deadline"))
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", consume)
        # The install's clock reads 1000.0 s at its start and 1012.3 s at the deadline: the error
        # gives the time spent (12.3 s), not the 970 s total bound, whatever the load on the host.
        reads = iter((1000.0, 1012.3))
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.time", SimpleNamespace(monotonic=lambda: next(reads)))

        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.PACKAGE_DOWNLOAD,
            match=r"^Package installation timeout after 12\.3s$",
        ) as exc_info:
            await self._install(snapshot_manager)

        consume.assert_awaited_once()
        assert exc_info.value.context.items() >= _DEADLINE_DIAGNOSTICS.items()


# ============================================================================
# Cache hierarchy: L2, then L3, then a build
# ============================================================================


class TestCacheHierarchy:
    """get_or_create_snapshot: an L2 hit, else an L3 hit (S3), else a build. The build VM is a
    stub (_fake_builds: no VM boots); TestL3Cache has the L3 downloads and uploads."""

    async def test_l2_hit_returns_immediately_no_s3(self, make_keyed_manager, moto_s3_server, tmp_path: Path) -> None:
        """An L2 hit is returned as it is: the S3 object of the same key is not downloaded over
        it, and no build runs."""
        endpoint_url, s3_sync = moto_s3_server
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )
        entry = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        await create_test_qcow2(entry)
        l2_content = entry.read_bytes()
        other = tmp_path / "other.qcow2"
        await create_test_qcow2(other, size="2M")
        s3_content = other.read_bytes()
        assert s3_content != l2_content
        s3_sync.put_object(
            Bucket=BUCKET,
            Key=_L3_OBJECT,
            Body=zstd.compress(s3_content),
            Metadata={"sha256": hashlib.sha256(s3_content).hexdigest()},
        )

        assert await _lookup(snapshot_manager) == entry
        assert entry.read_bytes() == l2_content

    async def test_l2_populated_after_l3_download(self, make_keyed_manager, moto_s3_server, tmp_path: Path) -> None:
        """An L3 download fills L2: the next request is an L2 hit, also once the S3 object is gone."""
        endpoint_url, s3_sync = moto_s3_server
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET, s3_region="us-east-1", s3_endpoint_url=endpoint_url
        )
        real_qcow2 = tmp_path / "real-snapshot.qcow2"
        await create_test_qcow2(real_qcow2)  # an L2 hit runs qemu-img check on it
        original_content = real_qcow2.read_bytes()
        s3_sync.put_object(
            Bucket=BUCKET,
            Key=_L3_OBJECT,
            Body=zstd.compress(original_content),
            Metadata={"sha256": hashlib.sha256(original_content).hexdigest()},
        )
        l2_path = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"

        assert await _lookup(snapshot_manager) == l2_path
        assert l2_path.read_bytes() == original_content

        s3_sync.delete_object(Bucket=BUCKET, Key=_L3_OBJECT)
        assert await _lookup(snapshot_manager) == l2_path  # a download now would miss and build

    async def test_concurrent_snapshot_requests_single_creation(
        self, make_keyed_manager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """10 concurrent get_or_create_snapshot calls create the snapshot once.

        The build lock (CacheEntry.saving): the first caller takes it and builds, while the
        other 9 wait for it, then see should_create=False (the file exists) and return the same
        path. The build goes on once all 10 asked for the lock; each answer of the lock is
        recorded."""
        snapshot_manager, settings = await make_keyed_manager(s3_bucket=None)
        expected_path = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        asked: list[None] = []
        should_create: list[bool] = []

        class _RecordedEntry(CacheEntry):
            @asynccontextmanager
            async def saving(self) -> AsyncIterator[bool]:
                asked.append(None)
                async with super().saving() as answer:
                    should_create.append(answer)
                    yield answer

        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.CacheEntry", _RecordedEntry)
        builds = _fake_builds(snapshot_manager, monkeypatch, content=b"built", failures=[])

        async def all_asked(**_kwargs: object) -> None:  # the guest connection of the install, in the build
            await _until(lambda: len(asked) == 10, what="all 10 callers asked for the build lock")

        builds.vm.channel.connect = all_asked
        results = await asyncio.gather(
            *(
                _built_path(
                    snapshot_manager.get_or_create_snapshot(
                        language=Language.PYTHON,
                        packages=_PACKAGES,
                        tenant_id="test",
                        task_id=f"concurrent-{i}",
                        memory_mb=256,
                    )
                )
                for i in range(10)
            )
        )

        assert len(builds.starts) == 1
        assert should_create == [True] + [False] * 9
        assert results == [expected_path] * 10

    async def test_l2_hit_is_leased_until_released(self, make_keyed_manager) -> None:
        """The hit is leased before its check: eviction (admission's, for disk space) skips
        it until the caller lets it go (each VM on it holds its own share)."""
        snapshot_manager, settings = await make_keyed_manager()
        entry = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        await create_test_qcow2(entry)

        lease = await snapshot_manager.get_or_create_snapshot(
            language=Language.PYTHON, packages=_PACKAGES, tenant_id="t", task_id="t", memory_mb=256
        )
        assert lease.path == entry
        assert await snapshot_manager.evict_oldest() is None  # the only entry is in use
        assert entry.exists()
        lease.release()
        size = entry_bytes(entry)
        assert await snapshot_manager.evict_oldest() == size
        assert not entry.exists()

    async def test_broken_l2_entry_leaves_no_lease(self, make_keyed_manager) -> None:
        """A broken entry (qemu-img check fails) is a miss: it goes, and so does its lease."""
        snapshot_manager, settings = await make_keyed_manager()
        broken = settings.disk_snapshot_cache_dir / f"{_BASE_KEY}.qcow2"
        broken.write_bytes(b"\x00\xff\xde\xad" * 64)

        assert await snapshot_manager.check_cache(Language.PYTHON, []) is None
        assert not broken.exists()
        assert not leased_here(broken)

    async def test_broken_l2_entry_another_process_runs_on_stays(self, make_keyed_manager) -> None:
        """A check that finds the entry broken removes it as an eviction does: not while a VM
        of another process runs on it (its lease), and its own lease goes first."""
        snapshot_manager, settings = await make_keyed_manager()
        broken = settings.disk_snapshot_cache_dir / f"{_BASE_KEY}.qcow2"
        broken.write_bytes(b"\x00\xff\xde\xad" * 64)

        with leased_in_child(broken):
            assert await snapshot_manager.check_cache(Language.PYTHON, []) is None
            assert broken.exists()
            assert not leased_here(broken)
        assert await snapshot_manager.check_cache(Language.PYTHON, []) is None
        assert not broken.exists()

    async def test_leaked_clusters_only_is_a_valid_image(self, make_keyed_manager) -> None:
        """qemu-img check exit 3: leaked clusters only (space lost inside the file after an
        interrupted write), the data is intact. A hit, kept."""
        snapshot_manager, settings = await make_keyed_manager()
        entry = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        await create_test_qcow2(entry)

        async def leaked(cmd: list[str], **_kwargs: object) -> tuple[int, bytes, bytes]:
            assert cmd[:2] == ["qemu-img", "check"]
            return 3, b"", b"Leaked cluster 7 refcount=1 reference=0"

        with patch("exec_sandbox.disk_snapshot_manager.communicate_managed_process", side_effect=leaked):
            assert await _checked(snapshot_manager, Language.PYTHON, _PACKAGES) == entry
        assert entry.exists()

    async def test_upload_holds_its_share_until_it_ends(
        self, make_keyed_manager, s3_proxy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The S3 upload of a build reads the image after the request is over (it waits for its
        turn first): its own share keeps eviction off the image until it ends. The S3 proxy
        holds the upload (PUT): the window in which only its share holds the image."""
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET,
            s3_region="us-east-1",
            s3_endpoint_url=s3_proxy.endpoint_url,
            s3_snapshot_prefix=s3_proxy.prefix,
        )
        entry = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        held = s3_proxy.add("PUT", r"\.qcow2\.zst$", Hold())
        _fake_builds(snapshot_manager, monkeypatch, content=b"built", failures=[])

        lease = await snapshot_manager.get_or_create_snapshot(
            Language.PYTHON, _PACKAGES, tenant_id="t", task_id="t", memory_mb=256
        )
        lease.release()  # the request is over
        async with asyncio.timeout(5):
            await held.reached.wait()
        assert not evictable(entry)
        held.release()
        await snapshot_manager.stop()
        assert s3_proxy.bucket.listing() == [f"{_KEY}.qcow2.zst"]
        assert evictable(entry)

    async def test_stop_cancels_an_upload_still_held_after_the_grace(
        self, make_keyed_manager, s3_proxy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A store that never answers the upload (the S3 proxy holds the PUT): stop() (Scheduler
        exit) waits the grace, then returns, with no compressed file left and the image free
        for eviction again."""
        monkeypatch.setattr(constants, "L3_UPLOAD_STOP_GRACE_SECONDS", 0.2)
        snapshot_manager, settings = await make_keyed_manager(
            s3_bucket=BUCKET,
            s3_region="us-east-1",
            s3_endpoint_url=s3_proxy.endpoint_url,
            s3_snapshot_prefix=s3_proxy.prefix,
        )
        entry = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        held = s3_proxy.add("PUT", r"\.qcow2\.zst$", Hold())
        _fake_builds(snapshot_manager, monkeypatch, content=b"built", failures=[])
        lease = await snapshot_manager.get_or_create_snapshot(
            Language.PYTHON, _PACKAGES, tenant_id="t", task_id="t", memory_mb=256
        )
        lease.release()
        async with asyncio.timeout(5):
            await held.reached.wait()

        started = time.monotonic()
        async with asyncio.timeout(5):
            await snapshot_manager.stop()
        waited = time.monotonic() - started

        held.release()
        assert waited >= 0.2  # the grace, not an immediate cancel
        assert _entries(settings.disk_snapshot_cache_dir) == [f"{_KEY}.qcow2"]
        assert evictable(entry)

    async def test_stop_aborts_a_multipart_upload_held_after_the_grace(
        self, make_keyed_manager, s3_proxy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A layer above one part (10 MiB of random bytes: zstd cannot make it smaller) goes up
        in parts; the S3 proxy holds them. stop() cancels the upload after the grace and the
        upload is aborted: no incomplete multipart upload stays in the bucket (a cancel is no
        Exception, so an abort only on Exception left it there)."""
        monkeypatch.setattr(constants, "L3_UPLOAD_STOP_GRACE_SECONDS", 0.2)
        snapshot_manager, _settings = await make_keyed_manager(
            s3_bucket=BUCKET,
            s3_region="us-east-1",
            s3_endpoint_url=s3_proxy.endpoint_url,
            s3_snapshot_prefix=s3_proxy.prefix,
        )
        held = s3_proxy.add("MPU_PART", r"\.qcow2\.zst$", Hold(), times=None)
        _fake_builds(snapshot_manager, monkeypatch, content=os.urandom(10 * 1024 * 1024), failures=[])
        lease = await snapshot_manager.get_or_create_snapshot(
            Language.PYTHON, _PACKAGES, tenant_id="t", task_id="t", memory_mb=256
        )
        lease.release()
        async with asyncio.timeout(10):
            await held.reached.wait()

        async with asyncio.timeout(10):
            await snapshot_manager.stop()

        held.release()
        assert s3_proxy.count("MPU_ABORT", r"\.qcow2\.zst$") == 1
        uploads = s3_proxy.bucket.s3.list_multipart_uploads(Bucket=BUCKET, Prefix=s3_proxy.prefix)
        assert uploads.get("Uploads", []) == []

    async def test_built_snapshot_is_leased_when_returned(
        self, make_keyed_manager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A new build is leased under its build lock (that eviction needs too): the caller
        gets it held, with no moment in which an eviction could take it."""
        snapshot_manager, _settings = await make_keyed_manager(s3_bucket=None)
        _fake_builds(snapshot_manager, monkeypatch, content=b"built", failures=[])

        lease = await snapshot_manager.get_or_create_snapshot(
            language=Language.PYTHON, packages=["pandas==2.0.0"], tenant_id="t", task_id="t", memory_mb=256
        )
        try:
            assert lease.path.exists()
            assert not evictable(lease.path)
        finally:
            lease.release()

    async def test_corrupt_l2_triggers_recreation(self, make_keyed_manager, monkeypatch: pytest.MonkeyPatch) -> None:
        """A corrupt qcow2 in L2 (garbage bytes) fails qemu-img check: no false hit, a new build
        takes its place."""
        snapshot_manager, settings = await make_keyed_manager(s3_bucket=None)
        corrupt_path = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        corrupt_path.write_bytes(b"\x00\xff\xde\xad" * 64)
        builds = _fake_builds(snapshot_manager, monkeypatch, content=b"rebuilt layer", failures=[])

        result_path = await _built_path(
            snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=_PACKAGES,
                tenant_id="test",
                task_id=f"test-{uuid4().hex}",
                memory_mb=256,
            )
        )

        assert result_path == corrupt_path
        assert len(builds.starts) == 1
        assert result_path.read_bytes() == b"rebuilt layer"

    async def test_stale_lock_file_does_not_block(self, make_keyed_manager, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stale .lock file left by a crashed process does not block a build.

        CacheEntry.saving uses flock(), which goes when the process dies (fd close): a lock
        file left on disk does not keep new callers from taking the lock and building.
        """
        snapshot_manager, settings = await make_keyed_manager(s3_bucket=None)
        expected_path = settings.disk_snapshot_cache_dir / f"{_KEY}.qcow2"
        expected_path.with_suffix(".qcow2.lock").touch()  # left by a crashed process
        _fake_builds(snapshot_manager, monkeypatch, content=b"built", failures=[])

        result_path = await asyncio.wait_for(  # a wait on the stale lock file would hang
            _built_path(
                snapshot_manager.get_or_create_snapshot(
                    language=Language.PYTHON,
                    packages=_PACKAGES,
                    tenant_id="test",
                    task_id=f"test-{uuid4().hex}",
                    memory_mb=256,
                )
            ),
            timeout=10,
        )

        assert result_path == expected_path
        assert result_path.read_bytes() == b"built"

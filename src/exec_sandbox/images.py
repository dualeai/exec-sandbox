"""Guest images: the one registry of the files a VM boots, and of their digests.

The catalog names each file as the build scripts make it and the release serves it, by
architecture: the base image of each language (its runtime's minor version in the name, as
scripts/build-qcow2.sh gives it), the kernel (vmlinux for PVH direct boot on x86_64, else
vmlinuz) and the initramfs. A new runtime version changes one line here.

GuestImages finds them in this host's folders. Lookups stay lazy (a stat): the scheduler
downloads a base image on its first use (assets.ensure_base_image). Digests are cached by file
stamp (image_digest): a rebuilt or replaced file is hashed again at once.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.files.disk import create_file, file_stamp
from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostArch, arch_name

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from exec_sandbox.settings import Settings

RELEASE_SUFFIX: Final = ".zst"
"""The suffix of each image in a release (zstd); the local file has none."""


@dataclass(frozen=True)
class BaseImage:
    """The base image of one language: `stem` and, when its runtime has one, the `version`
    in its name (the runtime's minor version: Python 3.14, Bun 1.4)."""

    stem: str
    version: str | None

    def file_name(self, arch: HostArch) -> str:
        """Its file name for `arch` (python-3.14-base-x86_64.qcow2)."""
        version = "" if self.version is None else f"-{self.version}"
        return f"{self.stem}{version}-base-{arch_name(arch)}.qcow2"


BASE_IMAGES: Final[Mapping[Language, BaseImage]] = MappingProxyType(
    {
        Language.PYTHON: BaseImage(stem="python", version="3.14"),
        Language.JAVASCRIPT: BaseImage(stem="node", version="1.4"),
        Language.RAW: BaseImage(stem="raw", version=None),
    }
)
"""Each language's base image (scripts/build-qcow2.sh output_name; tests/test_assets.py checks it)."""


def kernel_names(arch: HostArch) -> tuple[str, ...]:
    """The kernel file names, preferred first: vmlinux (PVH direct boot, about 50 ms faster;
    x86_64 only), then vmlinuz."""
    name = arch_name(arch)
    return (f"vmlinux-{name}", f"vmlinuz-{name}") if arch is HostArch.X86_64 else (f"vmlinuz-{name}",)


def initramfs_name(arch: HostArch) -> str:
    """The initramfs file name for `arch` (initramfs-x86_64)."""
    return f"initramfs-{arch_name(arch)}"


def kernel_dir_of(images_dir: Path, *, arch: HostArch) -> Path:
    """The folder of the kernel and the initramfs in `images_dir`: its kernels/ subfolder when
    that holds a kernel and the initramfs of `arch` (the layout of container images), else
    images_dir itself: a kernels/ folder with none or one of them does not hide the files at
    the top. A few stats (os.path: a folder this process may not read counts as
    missing)."""
    kernels = images_dir / "kernels"
    boot_files = os.path.exists(kernels / initramfs_name(arch)) and any(  # noqa: PTH110
        os.path.exists(kernels / name)  # noqa: PTH110
        for name in kernel_names(arch)
    )
    return kernels if boot_files else images_dir


@dataclass(frozen=True)
class GuestImages:
    """This host's guest files for `arch`: the base images in `images_dir`, else in
    `download_dir` (where a base image that images_dir lacks is downloaded; None: no such
    folder), the kernel and the initramfs in `kernel_dir`. Each lookup is a stat (an image can
    come or go meanwhile)."""

    images_dir: Path
    kernel_dir: Path
    download_dir: Path | None
    arch: HostArch

    @classmethod
    def from_settings(cls, settings: Settings, *, arch: HostArch) -> GuestImages:
        """The folders of `settings`: base_images_dir, kernel_path, base_images_download_dir."""
        return cls(
            images_dir=settings.base_images_dir,
            kernel_dir=settings.kernel_path,
            download_dir=settings.base_images_download_dir,
            arch=arch,
        )

    def _base_image_or_none(self, file_name: str) -> Path | None:
        folders = (self.images_dir,) if self.download_dir is None else (self.images_dir, self.download_dir)
        return next((folder / file_name for folder in folders if (folder / file_name).exists()), None)

    def base_image(self, language: Language) -> Path:
        """The base image of `language`: in images_dir, else in download_dir.

        Raises:
            ConfigError: ASSET_NOT_FOUND: in neither folder
        """
        file_name = BASE_IMAGES[language].file_name(self.arch)
        image = self._base_image_or_none(file_name)
        if image is None:
            folders = [str(self.images_dir)] + ([] if self.download_dir is None else [str(self.download_dir)])
            raise ConfigError(
                f"Base image not found for language: {language.value}. File: {file_name}, searched: {folders}",
                code=ErrorCode.ASSET_NOT_FOUND,
                context=None,
                hint=None,
                instance_id=None,
            )
        return image

    def present_base_images(self) -> list[Path]:
        """The base image of each language that is on disk, as base_image finds it (VmManager.start
        makes the overlay template of each; Scheduler.describe_worker lists their digests)."""
        images = (self._base_image_or_none(image.file_name(self.arch)) for image in BASE_IMAGES.values())
        return [image for image in images if image is not None]

    @property
    def kernel(self) -> Path:
        """The kernel QEMU boots: the first of kernel_names that exists, else the last one (its
        absence is reported where it is opened)."""
        names = kernel_names(self.arch)
        return next(
            (self.kernel_dir / name for name in names[:-1] if (self.kernel_dir / name).exists()),
            self.kernel_dir / names[-1],
        )

    @property
    def initramfs(self) -> Path:
        """The initramfs QEMU boots (its absence is reported where it is opened)."""
        return self.kernel_dir / initramfs_name(self.arch)


# In-process digest cache: path -> (file_stamp, sha256 hex). One entry per path.
_DIGESTS: dict[str, tuple[str, str]] = {}


def image_digest(path: Path) -> str | None:
    """SHA-256 content digest of a file (the identity of a base image, kernel, initramfs or L2
    layer), or None when it does not exist.

    The digest is the same on every host for the same bytes, so cache keys built from it can be
    shared across hosts (an mtime-based fingerprint differs per host). The file is read once; the
    digest is cached in memory and in a ``<file>.sha256`` sidecar keyed by inode, mtime and size,
    so a rebuilt file is hashed again. Blocking: async callers use asyncio.to_thread.

    Raises:
        OSError: the file cannot be read (no access, an I/O error)
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    stamp = file_stamp(stat)
    cached = _DIGESTS.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        sidecar_stamp, sidecar_digest = sidecar.read_text().split()
        if sidecar_stamp == stamp:
            _DIGESTS[str(path)] = (stamp, sidecar_digest)
            return sidecar_digest
    except (OSError, ValueError):
        pass
    try:
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
    except FileNotFoundError:
        return None  # removed between stat and open (eviction)
    _DIGESTS[str(path)] = (stamp, digest)
    # Best effort: the image folder may be read-only (installed assets).
    with contextlib.suppress(OSError):
        tmp = sidecar.with_name(f"{sidecar.name}.{os.getpid()}.tmp")
        tmp.unlink(missing_ok=True)  # left by a killed process with this pid
        try:
            with create_file(tmp, mode=0o600, dir_fd=None) as f:  # only this user reads it
                f.write(f"{stamp} {digest}".encode())
            tmp.replace(sidecar)
        except OSError:  # a full disk: no temp file stays beside the image
            tmp.unlink(missing_ok=True)
            raise
    return digest


def _digest_or_disk_error(path: Path) -> str | None:
    """image_digest, with a read error as the library's LOCAL_DISK error.

    Raises:
        TransientError: LOCAL_DISK: the file cannot be read (no access, an I/O error)
    """
    try:
        return image_digest(path)
    except OSError as e:
        raise TransientError(
            f"Guest image {path.name} cannot be read: {e.strerror or type(e).__name__}",
            code=ErrorCode.LOCAL_DISK,
            context={"image": str(path), "errno": e.errno},
            hint=None,
            instance_id=None,
        ) from e


def key_part(path: Path) -> str:
    """A file's part of a cache key: its digest, or "missing0" when it is gone (a key that
    matches no entry). Blocking.

    Raises:
        TransientError: LOCAL_DISK: the file cannot be read
    """
    return _digest_or_disk_error(path) or "missing0"


def guest_digests(base_image: Path, packages_image: Path | None) -> tuple[str | None, str | None]:
    """The digests of the images a VM opens (None: missing), as a saved instance records them.
    Blocking: a worker thread.

    Raises:
        TransientError: LOCAL_DISK: an image cannot be read (no access, an I/O error)
    """
    packages = _digest_or_disk_error(packages_image) if packages_image is not None else None
    return _digest_or_disk_error(base_image), packages

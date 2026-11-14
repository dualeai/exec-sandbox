"""The guest image registry: catalog names by architecture, the files GuestImages finds, and the
digests (cached by file stamp)."""

from __future__ import annotations

import errno
import os
import stat
from typing import IO, TYPE_CHECKING
from unittest.mock import patch

import pytest

from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.files.disk import create_file
from exec_sandbox.images import (
    BASE_IMAGES,
    GuestImages,
    guest_digests,
    image_digest,
    initramfs_name,
    kernel_dir_of,
    kernel_names,
    key_part,
)
from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostArch, arch_of

if TYPE_CHECKING:
    from pathlib import Path

_SHA256_ONE = "7692c3ad3540bb803c020b3aee66cd8887123234ea0c6e7143c0add73ff431ed"
"""SHA-256 of b"one"."""
_SHA256_IMAGE_BYTES = "2c8648d103e3dd7ad87660da0f126a1443b6d21ac1bd3ec000c5e24e2373a90c"
"""SHA-256 of b"image-bytes"."""


class TestCatalog:
    @pytest.mark.parametrize("arch", [HostArch.X86_64, HostArch.AARCH64], ids=["x86_64", "aarch64"])
    def test_names_by_architecture(self, arch: HostArch) -> None:
        suffix = "x86_64" if arch is HostArch.X86_64 else "aarch64"
        assert {language: image.file_name(arch) for language, image in BASE_IMAGES.items()} == {
            Language.PYTHON: f"python-3.14-base-{suffix}.qcow2",
            Language.JAVASCRIPT: f"node-1.4-base-{suffix}.qcow2",
            Language.RAW: f"raw-base-{suffix}.qcow2",
        }
        assert initramfs_name(arch) == f"initramfs-{suffix}"

    def test_vmlinux_first_on_x86_64_only(self) -> None:
        """PVH direct boot (vmlinux) exists for x86_64; aarch64 boots vmlinuz."""
        assert kernel_names(HostArch.X86_64) == ("vmlinux-x86_64", "vmlinuz-x86_64")
        assert kernel_names(HostArch.AARCH64) == ("vmlinuz-aarch64",)

    def test_every_language_has_a_base_image(self) -> None:
        assert set(BASE_IMAGES) == set(Language)

    def test_arch_names_round_trip(self) -> None:
        assert arch_of("x86_64") is HostArch.X86_64
        assert arch_of("aarch64") is HostArch.AARCH64
        assert arch_of("riscv64") is HostArch.UNKNOWN


class TestKernelDirOf:
    @pytest.mark.parametrize(
        ("in_kernels", "expected"),
        [
            ([], "top"),
            (["vmlinuz-x86_64"], "top"),
            (["initramfs-x86_64"], "top"),
            (["vmlinuz-x86_64", "initramfs-x86_64"], "kernels"),
        ],
        ids=["empty-kernels", "kernel-only", "initramfs-only", "both"],
    )
    def test_kernels_folder_counts_only_with_the_initramfs(
        self, tmp_path: Path, in_kernels: list[str], expected: str
    ) -> None:
        """The kernels/ layout of container images counts when kernels/ holds a kernel and the
        initramfs of the arch; an empty or partial one does not hide the files at the top."""
        (tmp_path / "kernels").mkdir()
        for name in in_kernels:
            (tmp_path / "kernels" / name).touch()
        folder = kernel_dir_of(tmp_path, arch=HostArch.X86_64)
        assert folder == (tmp_path / "kernels" if expected == "kernels" else tmp_path)


class TestGuestImages:
    def test_missing_base_image_is_asset_not_found(self, tmp_path: Path) -> None:
        images = GuestImages(images_dir=tmp_path, kernel_dir=tmp_path, download_dir=None, arch=HostArch.X86_64)
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND, match=r"python-3\.14"):
            images.base_image(Language.PYTHON)
        assert images.present_base_images() == []

    def test_a_base_image_missing_from_images_dir_comes_from_the_download_dir(self, tmp_path: Path) -> None:
        """A local build with the Python image only: the JavaScript image that the scheduler
        downloaded is in download_dir; the Python one stays the images_dir one."""
        images_dir, download_dir = tmp_path / "dist", tmp_path / "cache"
        images_dir.mkdir()
        download_dir.mkdir()
        python = images_dir / "python-3.14-base-x86_64.qcow2"
        node = download_dir / "node-1.4-base-x86_64.qcow2"
        for path in (python, node, download_dir / python.name):
            path.touch()
        images = GuestImages(
            images_dir=images_dir, kernel_dir=images_dir, download_dir=download_dir, arch=HostArch.X86_64
        )

        assert images.base_image(Language.PYTHON) == python
        assert images.base_image(Language.JAVASCRIPT) == node
        assert sorted(images.present_base_images()) == sorted([python, node])
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND, match=str(download_dir)):
            images.base_image(Language.RAW)

    @pytest.mark.parametrize(
        ("arch", "files", "kernel", "initramfs"),
        [
            (HostArch.X86_64, ["vmlinux-x86_64", "vmlinuz-x86_64"], "vmlinux-x86_64", "initramfs-x86_64"),
            (HostArch.X86_64, ["vmlinuz-x86_64"], "vmlinuz-x86_64", "initramfs-x86_64"),
            (HostArch.AARCH64, ["vmlinux-aarch64", "vmlinuz-aarch64"], "vmlinuz-aarch64", "initramfs-aarch64"),
            # none: the last name (its absence is reported at the open)
            (HostArch.X86_64, [], "vmlinuz-x86_64", "initramfs-x86_64"),
        ],
        ids=["x86-both", "x86-vmlinuz", "arm-both", "x86-none"],
    )
    def test_kernel_is_the_first_present_name(
        self, tmp_path: Path, arch: HostArch, files: list[str], kernel: str, initramfs: str
    ) -> None:
        for name in files:
            (tmp_path / name).touch()
        images = GuestImages(images_dir=tmp_path, kernel_dir=tmp_path, download_dir=None, arch=arch)
        assert images.kernel == tmp_path / kernel
        assert images.initramfs == tmp_path / initramfs


class TestDigests:
    def test_digest_is_the_content_sha256_cached_by_stamp(self, tmp_path: Path) -> None:
        image = tmp_path / "base.qcow2"
        image.write_bytes(b"one")
        assert image_digest(image) == _SHA256_ONE
        with patch("exec_sandbox.images.hashlib.file_digest", side_effect=AssertionError("read again")):
            assert image_digest(image) == _SHA256_ONE  # same stamp: no read
        image.write_bytes(b"two!")  # a new size and mtime: a new stamp
        assert image_digest(image) == "a97b629a5b1da759586c3353a9bc284bf673970d8eca56e11e556b715c329614"  # b"two!"

    def test_missing_file(self, tmp_path: Path) -> None:
        assert image_digest(tmp_path / "gone") is None
        assert key_part(tmp_path / "gone") == "missing0"
        assert guest_digests(tmp_path / "gone", None) == (None, None)

    def test_unreadable_key_part_is_a_local_disk_error(self, tmp_path: Path) -> None:
        """An I/O error while an image of a cache key is read is the library's LOCAL_DISK
        error, not a raw OSError out of Scheduler.new. The read of the file is the fault."""
        image = tmp_path / "initramfs-x86_64"
        image.write_bytes(b"x")
        with (
            patch("exec_sandbox.images.hashlib.file_digest", side_effect=OSError(errno.EIO, "Input/output error")),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK) as exc,
        ):
            key_part(image)
        assert str(exc.value) == "Guest image initramfs-x86_64 cannot be read: Input/output error"
        assert exc.value.context == {"image": str(image), "errno": errno.EIO}

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file of mode 0000")
    def test_unreadable_image_is_a_local_disk_error(self, tmp_path: Path) -> None:
        """A base image this process may not read (mode 0000)."""
        image = tmp_path / "base.qcow2"
        image.write_bytes(b"x")
        image.chmod(0)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK) as exc:
            guest_digests(image, None)
        assert str(exc.value) == "Guest image base.qcow2 cannot be read: Permission denied"
        assert exc.value.context == {"image": str(image), "errno": errno.EACCES}


class TestImageDigestEdgeCases:
    """image_digest: edge cases of the content identity of an image."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """An empty file has a digest (it exists)."""

        f = tmp_path / "empty"
        f.write_bytes(b"")
        assert image_digest(f) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # b""

    def test_same_content_different_mtime_same_hash(self, tmp_path: Path) -> None:
        """A rewrite with identical bytes keeps the identity (mtime is not identity)."""

        f = tmp_path / "test"
        f.write_bytes(b"data")
        h1 = image_digest(f)
        f.write_bytes(b"data")
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # a new stamp
        assert image_digest(f) == h1

    def test_different_content_same_stat_different_hash(self, tmp_path: Path) -> None:
        """New bytes with the old mtime and size are re-hashed (stamp includes inode)."""

        f = tmp_path / "test"
        f.write_bytes(b"aaaa")
        stat1 = f.stat()
        h1 = image_digest(f)

        # Replace the file (new inode) with different bytes but the same mtime and size
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"bbbb")
        os.utime(replacement, ns=(stat1.st_atime_ns, stat1.st_mtime_ns))
        replacement.replace(f)
        assert h1 == "61be55a8e2f6b4e172338bddf184d6dbee29c98853e0a0485ecee7f27b9af0b4"  # b"aaaa"
        assert image_digest(f) == "81cc5b17018674b401b42f35ba07bb79e211239c23bffe658da1577e3e646877"  # b"bbbb"

    def test_sidecar_written(self, tmp_path: Path) -> None:
        """The digest is cached in a <file>.sha256 sidecar keyed by inode, mtime and size."""
        f = tmp_path / "image.qcow2"
        f.write_bytes(b"image-bytes")
        digest = image_digest(f)
        sidecar = tmp_path / "image.qcow2.sha256"
        assert sidecar.read_text().split()[1] == digest == _SHA256_IMAGE_BYTES

    def test_sidecar_has_no_group_or_other_access(self, tmp_path: Path) -> None:
        """Under `sg qemu-vm` and umask 002, open() made the sidecar 0664 of group qemu-vm: a
        QEMU as qemu-vm could give a base image or an L2 layer a wrong digest, and so the
        cache keys built on it. It is 0600, also where a killed process with this pid left
        its temp file (here 0666)."""

        f = tmp_path / "image.qcow2"
        f.write_bytes(b"image-bytes")
        stale = tmp_path / f"image.qcow2.sha256.{os.getpid()}.tmp"
        stale.write_text("old")
        stale.chmod(0o666)
        old = os.umask(0o002)
        try:
            digest = image_digest(f)
        finally:
            os.umask(old)
        sidecar = tmp_path / "image.qcow2.sha256"
        assert sidecar.read_text().split()[1] == digest == _SHA256_IMAGE_BYTES
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
        assert not stale.exists()

    def test_a_sidecar_that_cannot_be_written_leaves_no_file(self, tmp_path: Path) -> None:
        """The disk is full when the sidecar is written: the digest is still returned, and no
        temp file stays beside the image (no later call of another process removes it)."""
        f = tmp_path / "image.qcow2"
        f.write_bytes(b"image-bytes")
        real_create = create_file

        def full_disk(path: Path, *, mode: int, dir_fd: int | None) -> IO[bytes]:
            stream = real_create(path, mode=mode, dir_fd=dir_fd)

            def write(_data: bytes) -> int:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

            stream.write = write  # type: ignore[method-assign]
            return stream

        with patch("exec_sandbox.images.create_file", full_disk):
            assert image_digest(f) == _SHA256_IMAGE_BYTES
        assert [p.name for p in tmp_path.iterdir()] == ["image.qcow2"]

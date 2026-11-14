"""The guest image registry: catalog names by architecture, the files GuestImages finds, and the
digests (cached by file stamp)."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
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
        ("arch", "files", "kernel"),
        [
            (HostArch.X86_64, ["vmlinux-x86_64", "vmlinuz-x86_64"], "vmlinux-x86_64"),
            (HostArch.X86_64, ["vmlinuz-x86_64"], "vmlinuz-x86_64"),
            (HostArch.AARCH64, ["vmlinux-aarch64", "vmlinuz-aarch64"], "vmlinuz-aarch64"),
            (HostArch.X86_64, [], "vmlinuz-x86_64"),  # none: the last name (its absence is reported at the open)
        ],
        ids=["x86-both", "x86-vmlinuz", "arm-both", "x86-none"],
    )
    def test_kernel_is_the_first_present_name(
        self, tmp_path: Path, arch: HostArch, files: list[str], kernel: str
    ) -> None:
        for name in files:
            (tmp_path / name).touch()
        images = GuestImages(images_dir=tmp_path, kernel_dir=tmp_path, download_dir=None, arch=arch)
        assert images.kernel == tmp_path / kernel
        assert images.initramfs == tmp_path / f"initramfs-{kernel.rsplit('-', 1)[1]}"  # the kernel's arch


class TestDigests:
    def test_digest_is_the_content_sha256_cached_by_stamp(self, tmp_path: Path) -> None:
        image = tmp_path / "base.qcow2"
        image.write_bytes(b"one")
        assert image_digest(image) == hashlib.sha256(b"one").hexdigest()
        with patch("exec_sandbox.images.hashlib.file_digest", side_effect=AssertionError("read again")):
            assert image_digest(image) == hashlib.sha256(b"one").hexdigest()  # same stamp: no read
        image.write_bytes(b"two!")  # a new size and mtime: a new stamp
        assert image_digest(image) == hashlib.sha256(b"two!").hexdigest()

    def test_missing_file(self, tmp_path: Path) -> None:
        assert image_digest(tmp_path / "gone") is None
        assert key_part(tmp_path / "gone") == "missing0"
        assert guest_digests(tmp_path / "gone", None) == (None, None)

    def test_unreadable_key_part_is_a_local_disk_error(self, tmp_path: Path) -> None:
        """An I/O error at the stat of an image of a cache key is the library's LOCAL_DISK
        error, not a raw OSError out of Scheduler.new."""
        image = tmp_path / "initramfs-x86_64"
        image.write_bytes(b"x")
        with (
            patch("exec_sandbox.images.image_digest", side_effect=OSError(5, "Input/output error")),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK, match="initramfs"),
        ):
            key_part(image)

    def test_unreadable_image_is_a_local_disk_error(self, tmp_path: Path) -> None:
        image = tmp_path / "base.qcow2"
        image.write_bytes(b"x")
        with (
            patch("exec_sandbox.images.image_digest", side_effect=PermissionError(13, "Permission denied")),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK, match=r"base\.qcow2"),
        ):
            guest_digests(image, None)

"""Tests for validation: the kernel and initramfs pre-flight check (more in test_vm_manager.py)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.images import GuestImages
from exec_sandbox.platform_utils import HostArch
from exec_sandbox.validation import clear_kernel_validation_cache, validate_kernel_initramfs


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    clear_kernel_validation_cache()
    yield
    clear_kernel_validation_cache()


@pytest.mark.parametrize(("arch", "suffix"), [(HostArch.AARCH64, "aarch64"), (HostArch.X86_64, "x86_64")])
async def test_kernel_without_initramfs_names_the_missing_file(tmp_path: Path, arch: HostArch, suffix: str) -> None:
    """A kernel with no initramfs next to it: ConfigError ASSET_NOT_FOUND (FIX_CONFIG) that names
    the missing file. The failure is not cached: the check passes once the file is there."""
    initramfs = tmp_path / f"initramfs-{suffix}"
    (tmp_path / f"vmlinuz-{suffix}").touch()

    with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND) as exc:
        await validate_kernel_initramfs(
            GuestImages(images_dir=tmp_path, kernel_dir=tmp_path, download_dir=None, arch=arch)
        )
    assert exc.value.message == f"Initramfs not found: {initramfs}"
    assert exc.value.context == {"initramfs_path": str(initramfs), "arch": suffix}

    initramfs.touch()
    await validate_kernel_initramfs(GuestImages(images_dir=tmp_path, kernel_dir=tmp_path, download_dir=None, arch=arch))

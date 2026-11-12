"""Hard validation functions that raise exceptions on failure.

These are NOT probes (optional feature detection) - they validate hard requirements.
"""

import asyncio
import os
from pathlib import Path

from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.platform_utils import HostArch

__all__ = [
    "clear_kernel_validation_cache",
    "validate_kernel_initramfs",
]

# Pre-flight validation cache keyed by (kernel_path, arch)
_kernel_validated: set[tuple[Path, HostArch]] = set()


async def validate_kernel_initramfs(kernel_path: Path, arch: HostArch) -> None:
    """Pre-flight check: validate kernel and initramfs exist (cached, one-time per config).

    This is NOT a probe (optional feature) - it's a hard requirement.
    Raises ConfigError (ASSET_NOT_FOUND) if files are missing.
    """
    cache_key = (kernel_path, arch)
    if cache_key in _kernel_validated:
        return

    arch_suffix = "aarch64" if arch == HostArch.AARCH64 else "x86_64"
    # Accept either vmlinux (PVH direct boot, x86_64) or vmlinuz (compressed)
    vmlinux, vmlinuz = kernel_path / f"vmlinux-{arch_suffix}", kernel_path / f"vmlinuz-{arch_suffix}"
    initramfs = kernel_path / f"initramfs-{arch_suffix}"
    # One worker-thread job: kernel_path comes from the caller's config (images_dir,
    # EXEC_SANDBOX_IMAGES_DIR), which can be a network disk. os.path.exists: a file
    # this process may not stat counts as missing (as Path.exists does from 3.14).
    has_vmlinux, has_vmlinuz, has_initramfs = await asyncio.to_thread(
        lambda: [os.path.exists(path) for path in (vmlinux, vmlinuz, initramfs)]  # noqa: PTH110
    )
    kernel = vmlinux if has_vmlinux else vmlinuz

    if not (has_vmlinux or has_vmlinuz):
        raise ConfigError(
            f"Kernel not found: {kernel}",
            code=ErrorCode.ASSET_NOT_FOUND,
            context={"kernel_path": str(kernel), "arch": arch_suffix},
            hint=None,
            instance_id=None,
        )
    if not has_initramfs:
        raise ConfigError(
            f"Initramfs not found: {initramfs}",
            code=ErrorCode.ASSET_NOT_FOUND,
            context={"initramfs_path": str(initramfs), "arch": arch_suffix},
            hint=None,
            instance_id=None,
        )

    _kernel_validated.add(cache_key)


def clear_kernel_validation_cache() -> None:
    """Clear the kernel validation cache (for testing)."""
    _kernel_validated.clear()

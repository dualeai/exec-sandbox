"""Hard validation functions that raise exceptions on failure.

These are NOT probes (optional feature detection) - they validate hard requirements.
"""

from pathlib import Path

import aiofiles.os

from exec_sandbox.exceptions import VmDependencyError
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
    Raises VmDependencyError if files are missing.
    """
    cache_key = (kernel_path, arch)
    if cache_key in _kernel_validated:
        return

    arch_suffix = "aarch64" if arch == HostArch.AARCH64 else "x86_64"
    kernel = kernel_path / f"vmlinuz-{arch_suffix}"
    initramfs = kernel_path / f"initramfs-{arch_suffix}"

    if not await aiofiles.os.path.exists(kernel):
        raise VmDependencyError(
            f"Kernel not found: {kernel}",
            context={"kernel_path": str(kernel), "arch": arch_suffix},
        )
    if not await aiofiles.os.path.exists(initramfs):
        raise VmDependencyError(
            f"Initramfs not found: {initramfs}",
            context={"initramfs_path": str(initramfs), "arch": arch_suffix},
        )

    _kernel_validated.add(cache_key)


def clear_kernel_validation_cache() -> None:
    """Clear the kernel validation cache (for testing)."""
    _kernel_validated.clear()

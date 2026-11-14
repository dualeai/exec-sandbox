"""Hard validation functions that raise exceptions on failure.

These are NOT probes (optional feature detection) - they validate hard requirements.
"""

import asyncio
import os
from pathlib import Path

from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.images import GuestImages, kernel_names
from exec_sandbox.platform_utils import HostArch, arch_name

__all__ = [
    "clear_kernel_validation_cache",
    "validate_kernel_initramfs",
]

# Pre-flight validation cache keyed by (kernel folder, arch)
_kernel_validated: set[tuple[Path, HostArch]] = set()


async def validate_kernel_initramfs(images: GuestImages) -> None:
    """Pre-flight check: a kernel (any of kernel_names(images.arch)) and the initramfs
    exist in images.kernel_dir (cached, one time per folder and arch).

    This is NOT a probe (optional feature) - it's a hard requirement.
    Raises ConfigError (ASSET_NOT_FOUND) if files are missing.
    """
    cache_key = (images.kernel_dir, images.arch)
    if cache_key in _kernel_validated:
        return

    kernels = [images.kernel_dir / name for name in kernel_names(images.arch)]
    initramfs = images.initramfs
    # One worker-thread job: the kernel folder comes from the caller's config (images_dir,
    # EXEC_SANDBOX_IMAGES_DIR), which can be a network disk. os.path.exists: a file
    # this process may not stat counts as missing (as Path.exists does from 3.14).
    *has_kernels, has_initramfs = await asyncio.to_thread(
        lambda: [os.path.exists(path) for path in (*kernels, initramfs)]  # noqa: PTH110
    )

    if not any(has_kernels):
        raise ConfigError(
            f"Kernel not found: {kernels[-1]}",
            code=ErrorCode.ASSET_NOT_FOUND,
            context={"kernel_path": str(kernels[-1]), "arch": arch_name(images.arch)},
            hint=None,
            instance_id=None,
        )
    if not has_initramfs:
        raise ConfigError(
            f"Initramfs not found: {initramfs}",
            code=ErrorCode.ASSET_NOT_FOUND,
            context={"initramfs_path": str(initramfs), "arch": arch_name(images.arch)},
            hint=None,
            instance_id=None,
        )

    _kernel_validated.add(cache_key)


def clear_kernel_validation_cache() -> None:
    """Clear the kernel validation cache (for testing)."""
    _kernel_validated.clear()

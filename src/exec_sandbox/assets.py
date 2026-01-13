"""
exec-sandbox asset registry and fetch functions.

Provides lazy downloading of VM images and binaries from GitHub Releases.
Uses AsyncPooch for caching and checksum verification.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from exec_sandbox._logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

from exec_sandbox.asset_downloader import (
    AsyncPooch,
    decompress_zstd,
    get_cache_dir,
    get_current_arch,
    get_gvproxy_suffix,
)
from exec_sandbox.permission_utils import chmod_executable

logger = get_logger(__name__)

# GitHub repository info
GITHUB_OWNER = "dualeai"
GITHUB_REPO = "exec-sandbox"

# Get version from package
try:
    from exec_sandbox import __version__
except ImportError:
    __version__ = "0.0.0.dev0"


def _create_assets_registry() -> AsyncPooch:
    """Create the assets registry singleton."""
    return AsyncPooch(
        path=get_cache_dir("exec-sandbox"),
        base_url=f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/v{{version}}",
        version=__version__,
        version_dev="latest",
        env="EXEC_SANDBOX_CACHE_DIR",
        registry={},  # Loaded dynamically from GitHub API
    )


# Global assets registry (lazy initialization)
_assets_singleton: AsyncPooch | None = None


def get_assets() -> AsyncPooch:
    """Get or create the assets registry singleton."""
    global _assets_singleton  # noqa: PLW0603 - Singleton pattern
    if _assets_singleton is None:
        _assets_singleton = _create_assets_registry()
    return _assets_singleton


def is_offline_mode() -> bool:
    """Check if offline mode is enabled via environment variable."""
    return os.environ.get("EXEC_SANDBOX_OFFLINE", "0") == "1"


async def ensure_registry_loaded() -> None:
    """
    Ensure the asset registry is loaded from GitHub.

    In offline mode, this is a no-op (assumes assets are pre-cached).
    """
    assets = get_assets()

    # Skip if already loaded
    if assets.registry:
        return

    if is_offline_mode():
        logger.debug("Offline mode enabled, skipping registry load from GitHub")
        return

    # Get version override from environment
    version = os.environ.get("EXEC_SANDBOX_ASSET_VERSION") or f"v{__version__}"
    if not version.startswith("v"):
        version = f"v{version}"

    # Handle dev versions
    if ".dev" in __version__:
        version = "latest"

    logger.info("Loading asset registry from GitHub", extra={"version": version})
    await assets.load_registry_from_github(GITHUB_OWNER, GITHUB_REPO, version)


async def fetch_kernel(arch: str | None = None) -> Path:
    """
    Fetch kernel for the given architecture.

    Args:
        arch: Architecture ("x86_64" or "aarch64"). Defaults to current machine.

    Returns:
        Path to the decompressed kernel file.
    """
    arch = arch or get_current_arch()
    fname = f"vmlinuz-{arch}.zst"

    # Check local cache first
    if local_path := get_cached_asset_path(fname):
        logger.debug("Using cached kernel", extra={"arch": arch, "path": str(local_path)})
        return local_path

    # Not found locally, download from GitHub
    await ensure_registry_loaded()
    assets = get_assets()

    logger.debug("Fetching kernel", extra={"arch": arch, "file": fname})
    return await assets.fetch(fname, processor=decompress_zstd)


async def fetch_initramfs(arch: str | None = None) -> Path:
    """
    Fetch initramfs for the given architecture.

    Args:
        arch: Architecture ("x86_64" or "aarch64"). Defaults to current machine.

    Returns:
        Path to the decompressed initramfs file.
    """
    arch = arch or get_current_arch()
    fname = f"initramfs-{arch}.zst"

    # Check local cache first
    if local_path := get_cached_asset_path(fname):
        logger.debug("Using cached initramfs", extra={"arch": arch, "path": str(local_path)})
        return local_path

    # Not found locally, download from GitHub
    await ensure_registry_loaded()
    assets = get_assets()

    logger.debug("Fetching initramfs", extra={"arch": arch, "file": fname})
    return await assets.fetch(fname, processor=decompress_zstd)


async def fetch_base_image(language: str, arch: str | None = None) -> Path:
    """
    Fetch base qcow2 image for the given language.

    Args:
        language: Programming language ("python" or "javascript").
        arch: Architecture ("x86_64" or "aarch64"). Defaults to current machine.

    Returns:
        Path to the decompressed qcow2 image file.
    """
    arch = arch or get_current_arch()

    # Map language to image filename
    if language == "python":
        fname = f"python-3.14-base-{arch}.qcow2.zst"
    elif language == "javascript":
        fname = f"node-1.3-base-{arch}.qcow2.zst"
    else:
        fname = f"raw-base-{arch}.qcow2.zst"

    # Check local cache first
    if local_path := get_cached_asset_path(fname):
        logger.debug("Using cached base image", extra={"language": language, "arch": arch, "path": str(local_path)})
        return local_path

    # Not found locally, download from GitHub
    await ensure_registry_loaded()
    assets = get_assets()

    logger.debug("Fetching base image", extra={"language": language, "arch": arch, "file": fname})
    return await assets.fetch(fname, processor=decompress_zstd)


async def fetch_gvproxy() -> Path:
    """
    Fetch gvproxy-wrapper binary for the current platform.

    Returns:
        Path to the gvproxy-wrapper binary (executable).
    """
    suffix = get_gvproxy_suffix()
    fname = f"gvproxy-wrapper-{suffix}"

    # Check local cache first
    if local_path := get_cached_asset_path(fname):
        logger.debug("Using cached gvproxy-wrapper", extra={"path": str(local_path)})
        # Ensure executable
        await chmod_executable(local_path)
        return local_path

    # Not found locally, download from GitHub
    await ensure_registry_loaded()
    assets = get_assets()

    logger.debug("Fetching gvproxy-wrapper", extra={"file": fname})
    path = await assets.fetch(fname)

    # Make executable
    await chmod_executable(path)

    return path


async def ensure_assets_available(language: str | None = None) -> tuple[Path, Path]:
    """
    Ensure all required assets are available for the given language.

    Downloads assets from GitHub Releases if not already cached.
    In offline mode, raises FileNotFoundError if assets are missing.

    Args:
        language: Optional language to pre-fetch base image for.
                  If None, only fetches kernel and gvproxy.

    Returns:
        Tuple of (images_dir, gvproxy_path)

    Raises:
        AssetNotFoundError: Release not found on GitHub.
        AssetDownloadError: Download failed after retries.
        AssetChecksumError: Hash verification failed.
        FileNotFoundError: Offline mode and assets missing.
    """
    # Fetch required assets
    kernel_path = await fetch_kernel()
    gvproxy_path = await fetch_gvproxy()

    # Pre-fetch language base image if specified
    if language:
        await fetch_base_image(language)

    # Images directory is the parent of the kernel
    images_dir = kernel_path.parent

    logger.info(
        "Assets ready",
        extra={"images_dir": str(images_dir), "gvproxy": str(gvproxy_path)},
    )

    return images_dir, gvproxy_path


def get_cached_asset_path(fname: str) -> Path | None:
    """
    Get path to a cached asset without downloading.

    Args:
        fname: Asset filename.

    Returns:
        Path to the cached file if it exists, None otherwise.
    """
    cache_dir = get_cache_dir("exec-sandbox")
    path = cache_dir / fname

    if path.exists():
        return path

    # Check for decompressed version (without .zst)
    if fname.endswith(".zst"):
        decompressed_path = cache_dir / fname[:-4]
        if decompressed_path.exists():
            return decompressed_path

    return None

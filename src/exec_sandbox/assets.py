"""
exec-sandbox asset registry and fetch functions.

Provides lazy downloading of VM images and binaries from GitHub Releases.
Uses AsyncPooch for caching and checksum verification.

Path Resolution Priority:
    1. override (explicit from SchedulerConfig.images_dir)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. ./images/dist/ (local build directory)
    4. ~/.cache/exec-sandbox/ (download cache)
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import aiofiles.os

from exec_sandbox import __version__
from exec_sandbox._logging import get_logger
from exec_sandbox.asset_downloader import (
    AsyncPooch,
    decompress_zstd,
    get_cache_dir,
    get_current_arch,
    get_gvproxy_suffix,
)
from exec_sandbox.permission_utils import chmod_executable

logger = get_logger(__name__)

# Public API - only these should be used externally
__all__ = [
    "ensure_assets",
    "ensure_assets_available",
    "ensure_registry_loaded",
    "fetch_base_image",
    "fetch_gvproxy",
    "fetch_initramfs",
    "fetch_kernel",
    "get_assets",
    "get_gvproxy_path",
    "is_offline_mode",
]

# GitHub repository info
GITHUB_OWNER = "dualeai"
GITHUB_REPO = "exec-sandbox"


def _get_asset_version() -> tuple[str, str]:
    """Get asset version from env var or package version.

    Returns:
        Tuple of (version_for_pooch, version_for_github_tag).
        - version_for_pooch: Version without 'v' prefix for AsyncPooch
        - version_for_github_tag: Version with 'v' prefix (or 'latest' for dev)
    """
    env_version = os.environ.get("EXEC_SANDBOX_ASSET_VERSION")
    if env_version:
        version = env_version.lstrip("v")
        return version, f"v{version}"
    if ".dev" in __version__:
        return __version__, "latest"
    return __version__, f"v{__version__}"


def _create_assets_registry() -> AsyncPooch:
    """Create the assets registry singleton."""
    pooch_version, _ = _get_asset_version()
    return AsyncPooch(
        path=get_cache_dir("exec-sandbox"),
        base_url=f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/v{{version}}",
        version=pooch_version,
        env="EXEC_SANDBOX_CACHE_DIR",
        registry={},  # Loaded dynamically from GitHub API
    )


# Global assets registry (lazy initialization with thread-safe double-checked locking)
_assets_singleton: AsyncPooch | None = None
_assets_lock = threading.Lock()


def get_assets() -> AsyncPooch:
    """Get or create the assets registry singleton.

    Thread-safe using double-checked locking pattern.
    """
    global _assets_singleton  # noqa: PLW0603 - Singleton pattern
    if _assets_singleton is None:
        with _assets_lock:
            # Double-check after acquiring lock
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

    # Get version (env var takes precedence, then dev->latest, then package version)
    _, version = _get_asset_version()

    logger.info("Loading asset registry from GitHub", extra={"version": version})
    await assets.load_registry_from_github(GITHUB_OWNER, GITHUB_REPO, version)


async def fetch_kernel(arch: str | None = None, override: Path | None = None) -> Path:
    """
    Fetch kernel for the given architecture.

    Args:
        arch: Architecture ("x86_64" or "aarch64"). Defaults to current machine.
        override: Explicit path to search first (from SchedulerConfig.images_dir).

    Returns:
        Path to the decompressed kernel file.
    """
    arch = arch or get_current_arch()
    fname = f"vmlinuz-{arch}.zst"

    # Check local cache first
    if local_path := await _find_asset(fname, override):
        logger.debug("Using cached kernel", extra={"arch": arch, "path": str(local_path)})
        return local_path

    # Not found locally, download from GitHub
    await ensure_registry_loaded()
    assets = get_assets()

    logger.debug("Fetching kernel", extra={"arch": arch, "file": fname})
    return await assets.fetch(fname, processor=decompress_zstd)


async def fetch_initramfs(arch: str | None = None, override: Path | None = None) -> Path:
    """
    Fetch initramfs for the given architecture.

    Args:
        arch: Architecture ("x86_64" or "aarch64"). Defaults to current machine.
        override: Explicit path to search first (from SchedulerConfig.images_dir).

    Returns:
        Path to the decompressed initramfs file.
    """
    arch = arch or get_current_arch()
    fname = f"initramfs-{arch}.zst"

    # Check local cache first
    if local_path := await _find_asset(fname, override):
        logger.debug("Using cached initramfs", extra={"arch": arch, "path": str(local_path)})
        return local_path

    # Not found locally, download from GitHub
    await ensure_registry_loaded()
    assets = get_assets()

    logger.debug("Fetching initramfs", extra={"arch": arch, "file": fname})
    return await assets.fetch(fname, processor=decompress_zstd)


async def fetch_base_image(language: str, arch: str | None = None, override: Path | None = None) -> Path:
    """
    Fetch base qcow2 image for the given language.

    Args:
        language: Programming language ("python" or "javascript").
        arch: Architecture ("x86_64" or "aarch64"). Defaults to current machine.
        override: Explicit path to search first (from SchedulerConfig.images_dir).

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
    if local_path := await _find_asset(fname, override):
        logger.debug("Using cached base image", extra={"language": language, "arch": arch, "path": str(local_path)})
        return local_path

    # Not found locally, download from GitHub
    await ensure_registry_loaded()
    assets = get_assets()

    logger.debug("Fetching base image", extra={"language": language, "arch": arch, "file": fname})
    return await assets.fetch(fname, processor=decompress_zstd)


async def get_gvproxy_path(override: Path | None = None) -> Path | None:
    """
    Get path to gvproxy-wrapper binary for current platform (without downloading).

    Detection order:
    0. override (explicit from SchedulerConfig.images_dir)
    1. Repo-relative path (gvproxy-wrapper/bin/ - for local development builds)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. Local build directory (./images/dist/)
    4. Download cache directory (~/.cache/exec-sandbox/)

    Args:
        override: Explicit path to search first (from SchedulerConfig.images_dir).

    Returns:
        Path to the gvproxy-wrapper binary if found, None otherwise.
    """
    suffix = get_gvproxy_suffix()
    fname = f"gvproxy-wrapper-{suffix}"

    # 0. Check override path first if provided
    if override:
        binary_path = override / fname
        if await aiofiles.os.path.exists(binary_path):
            return binary_path

    # 1. Check repo-relative path (dev mode - prioritized for local builds)
    repo_root = Path(__file__).parent.parent.parent
    binary_path = repo_root / "gvproxy-wrapper" / "bin" / fname

    if await aiofiles.os.path.exists(binary_path):
        return binary_path

    # 2-4. Fall back to standard asset cache lookup
    return await _find_asset(fname, override)


async def fetch_gvproxy(override: Path | None = None) -> Path:
    """
    Fetch gvproxy-wrapper binary for the current platform.

    Args:
        override: Explicit path to search first (from SchedulerConfig.images_dir).

    Returns:
        Path to the gvproxy-wrapper binary (executable).
    """
    # Check local paths first
    if local_path := await get_gvproxy_path(override):
        logger.debug("Using cached gvproxy-wrapper", extra={"path": str(local_path)})
        # Ensure executable
        await chmod_executable(local_path)
        return local_path

    # Not found locally, download from GitHub
    await ensure_registry_loaded()
    assets = get_assets()

    suffix = get_gvproxy_suffix()
    fname = f"gvproxy-wrapper-{suffix}"

    logger.debug("Fetching gvproxy-wrapper", extra={"file": fname})
    path = await assets.fetch(fname)

    # Make executable
    await chmod_executable(path)

    return path


async def ensure_assets_available(
    language: str | None = None,
    override: Path | None = None,
) -> tuple[Path, Path]:
    """
    Ensure all required assets are available for the given language.

    Downloads assets from GitHub Releases if not already cached.
    In offline mode, raises FileNotFoundError if assets are missing.

    Args:
        language: Optional language to pre-fetch base image for.
                  If None, only fetches kernel and gvproxy.
        override: Explicit path to search first (from SchedulerConfig.images_dir).

    Returns:
        Tuple of (images_dir, gvproxy_path)

    Raises:
        AssetNotFoundError: Release not found on GitHub.
        AssetDownloadError: Download failed after retries.
        AssetChecksumError: Hash verification failed.
        FileNotFoundError: Offline mode and assets missing.
    """
    # Fetch required assets in parallel
    fetch_tasks: list[asyncio.Task[Path]] = [
        asyncio.create_task(fetch_kernel(override=override)),
        asyncio.create_task(fetch_initramfs(override=override)),
        asyncio.create_task(fetch_gvproxy(override=override)),
    ]
    if language:
        fetch_tasks.append(asyncio.create_task(fetch_base_image(language, override=override)))

    results = await asyncio.gather(*fetch_tasks)
    kernel_path, _, gvproxy_path = results[0], results[1], results[2]

    # Images directory is the parent of the kernel
    images_dir = kernel_path.parent

    logger.info(
        "Assets ready",
        extra={"images_dir": str(images_dir), "gvproxy": str(gvproxy_path)},
    )

    return images_dir, gvproxy_path


async def ensure_assets(
    override: Path | None = None,
    download: bool = True,
    language: str | None = None,
) -> Path:
    """
    Find or download assets. Single entry point for asset resolution.

    Args:
        override: Explicit path from SchedulerConfig.images_dir.
        download: If True, download missing assets. If False, raise on missing.
        language: Optional language to pre-fetch base image for.

    Returns:
        Path to images directory.

    Raises:
        FileNotFoundError: Assets not found and download=False.
        AssetNotFoundError: Release not found on GitHub.
        AssetDownloadError: Download failed after retries.
        AssetChecksumError: Hash verification failed.
    """
    # Try to find existing assets first
    if images_dir := await _find_images_dir(override):
        # Validate that essential files exist (not just the directory)
        arch = get_current_arch()
        kernel_path = images_dir / f"vmlinuz-{arch}"
        if await aiofiles.os.path.exists(kernel_path):
            logger.debug("Found existing assets", extra={"images_dir": str(images_dir)})
            return images_dir
        # Directory exists but kernel is missing - fall through to download
        logger.debug(
            "Directory exists but kernel missing",
            extra={"images_dir": str(images_dir), "kernel": str(kernel_path)},
        )

    # Not found - either download or error
    if not download:
        search_paths = _get_search_paths(override)
        raise FileNotFoundError(
            f"Assets not found and auto_download_assets=False. "
            f"Searched: {[str(p) for p in search_paths]}. "
            f"Set EXEC_SANDBOX_IMAGES_DIR or enable auto_download_assets."
        )

    # Download assets and return images directory
    images_dir, _ = await ensure_assets_available(language, override)
    return images_dir


def _get_search_paths(override: Path | None = None) -> list[Path]:
    """
    Get list of directories to search for assets.

    Priority order:
    1. override (explicit from SchedulerConfig.images_dir)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. Local build directory (./images/dist/ relative to package root)
    4. Download cache directory (~/.cache/exec-sandbox/)

    Args:
        override: Explicit path from config (highest priority).

    Returns:
        List of directories to search (may include non-existent paths).
    """
    paths: list[Path] = []

    # Priority 1: Explicit override from config
    if override:
        paths.append(override)

    # Priority 2: Env var override (skip if empty)
    if env_path := os.environ.get("EXEC_SANDBOX_IMAGES_DIR", "").strip():
        paths.append(Path(env_path))

    # Priority 3: Local build directory (for development/CI)
    local_images = Path(__file__).parent.parent.parent / "images" / "dist"
    paths.append(local_images)

    # Priority 4: Download cache directory
    paths.append(get_cache_dir("exec-sandbox"))

    return paths


async def _find_images_dir(override: Path | None = None) -> Path | None:
    """
    Find first existing images directory.

    Checks multiple locations in priority order:
    1. override (explicit from SchedulerConfig.images_dir)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. Local build directory (./images/dist/ relative to package root)
    4. Download cache directory (~/.cache/exec-sandbox/)

    Args:
        override: Explicit path from config (highest priority).

    Returns:
        Path to images directory if found, None otherwise.
    """
    for path in _get_search_paths(override):
        if await aiofiles.os.path.exists(path):
            return path
    return None


async def _find_asset(fname: str, override: Path | None = None) -> Path | None:
    """
    Find specific asset file without downloading.

    Checks multiple locations in priority order:
    1. override (explicit from SchedulerConfig.images_dir)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. Local build directory (./images/dist/ relative to package root)
    4. Download cache directory (~/.cache/exec-sandbox/)

    Also checks for decompressed versions (.zst removed).

    Args:
        fname: Asset filename.
        override: Explicit path from config (highest priority).

    Returns:
        Path to the asset file if found, None otherwise.
    """
    for directory in _get_search_paths(override):
        path = directory / fname
        if await aiofiles.os.path.exists(path):
            return path
        # Check decompressed version (without .zst)
        if fname.endswith(".zst"):
            decompressed_path = directory / fname[:-4]
            if await aiofiles.os.path.exists(decompressed_path):
                return decompressed_path
    return None

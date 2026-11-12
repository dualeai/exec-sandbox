"""
exec-sandbox asset registry and fetch functions.

Provides lazy downloading of VM images and binaries from GitHub Releases.
Uses AsyncPooch for caching and checksum verification.

Path Resolution Priority:
    1. override (explicit from SchedulerConfig.images_dir)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. ./images/dist/ (local build directory)
    4. ~/.cache/exec-sandbox/v{version}/ (download cache)
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from exec_sandbox import __version__
from exec_sandbox._logging import get_logger
from exec_sandbox.asset_downloader import (
    AsyncPooch,
    decompress_zstd,
    get_current_arch,
    get_gvproxy_suffix,
)
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.models import Language
from exec_sandbox.permission_utils import chmod_executable
from exec_sandbox.platform_utils import UserDir, get_user_dir

logger = get_logger(__name__)

# Public API - only these should be used externally
__all__ = [
    "PrefetchResult",
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
    "prefetch_all_assets",
]

# GitHub repository info
GITHUB_OWNER = "dualeai"
GITHUB_REPO = "exec-sandbox"


@dataclass
class PrefetchResult:
    """Result of prefetch_all_assets operation."""

    success: bool
    arch: str
    downloaded: list[str] = field(default_factory=lambda: [])  # noqa: PIE807
    errors: list[tuple[str, str]] = field(default_factory=lambda: [])  # noqa: PIE807
    cache_dir: Path | None = None


def _versioned_cache_dir() -> Path:
    """Get version-specific cache directory for downloaded assets."""
    base = get_user_dir(kind=UserDir.CACHE, app_name="exec-sandbox")
    # Don't add version suffix when user explicitly overrides cache dir
    if os.environ.get("EXEC_SANDBOX_CACHE_DIR"):
        return base
    return base / f"v{__version__}"


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
        path=_versioned_cache_dir(),
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


def _kernel_candidates(arch: str) -> list[str]:
    """
    Kernel filenames in preference order for the given architecture.

    For x86_64: vmlinux (PVH direct boot, ~50ms faster) is preferred over vmlinuz.
    For aarch64: only vmlinuz is used. These are the unpacked names; the release
    serves each as `<name>.zst`.
    """
    candidates: list[str] = []
    if arch == "x86_64":
        candidates.append(f"vmlinux-{arch}")
    candidates.append(f"vmlinuz-{arch}")
    return candidates


async def fetch_kernel(arch: str | None = None, override: Path | None = None) -> Path:
    """
    Fetch kernel for the given architecture.

    Prefers vmlinux (PVH direct boot, x86_64 only, ~50ms faster) over vmlinuz.
    Falls back to vmlinuz if vmlinux is not available.

    Search respects directory priority order: override > env var > images/dist/ > cache.
    Within each directory, vmlinux is preferred over vmlinuz.

    Args:
        arch: Architecture ("x86_64" or "aarch64"). Defaults to current machine.
        override: Explicit path to search first (from SchedulerConfig.images_dir).

    Returns:
        Path to the decompressed kernel file.
    """
    arch = arch or get_current_arch()
    candidates = _kernel_candidates(arch)

    # Check each search path in priority order.
    # Within each directory, prefer vmlinux (PVH direct boot, x86_64 only).
    local = await asyncio.to_thread(
        _first_existing, [directory / name for directory in _get_search_paths(override) for name in candidates]
    )
    if local is not None:
        logger.debug("Using cached kernel", extra={"arch": arch, "path": str(local)})
        return local

    # Not found locally, download from GitHub
    await ensure_registry_loaded()
    assets = get_assets()

    # Try preferred kernel from registry (vmlinux for x86_64, then vmlinuz)
    download_candidates = [f"{name}.zst" for name in candidates]
    for name in download_candidates:
        if name in assets.registry:
            logger.debug("Fetching kernel", extra={"arch": arch, "file": name})
            return await assets.fetch(name, processor=decompress_zstd)

    # Fallback to vmlinuz (will fail at fetch time if truly missing)
    fallback = download_candidates[-1]
    logger.debug("Fetching kernel", extra={"arch": arch, "file": fallback})
    return await assets.fetch(fallback, processor=decompress_zstd)


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

    # Map language to image filename: the names scripts/build-qcow2.sh gives (python-<Python
    # minor>, node-<Bun minor>); tests/test_assets.py checks them against its default versions.
    if language == "python":
        fname = f"python-3.14-base-{arch}.qcow2.zst"
    elif language == "javascript":
        fname = f"node-1.4-base-{arch}.qcow2.zst"
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
    4. Download cache directory (~/.cache/exec-sandbox/v{version}/)

    Args:
        override: Explicit path to search first (from SchedulerConfig.images_dir).

    Returns:
        Path to the gvproxy-wrapper binary if found, None otherwise.
    """
    return await _find_gvproxy(override, arch=get_current_arch())


async def _find_gvproxy(override: Path | None, *, arch: str) -> Path | None:
    """The gvproxy-wrapper binary of this OS for `arch`, in get_gvproxy_path's order; None if not found."""
    fname = f"gvproxy-wrapper-{get_gvproxy_suffix(arch)}"

    # 0. Check override path first if provided
    candidates = [override / fname] if override else []

    # 1. Check repo-relative path (dev mode - prioritized for local builds)
    repo_root = Path(__file__).parent.parent.parent
    candidates.append(repo_root / "gvproxy-wrapper" / "bin" / fname)

    # 2-4. Fall back to standard asset cache lookup
    return await asyncio.to_thread(_first_existing, [*candidates, *_asset_candidates(fname, override)])


async def fetch_gvproxy(*, override: Path | None, arch: str) -> Path:
    """
    Fetch gvproxy-wrapper binary of this OS for an architecture.

    Args:
        override: Explicit path to search first (from SchedulerConfig.images_dir), or None.
        arch: Architecture ("x86_64" or "aarch64").

    Returns:
        Path to the gvproxy-wrapper binary (executable).
    """
    # Check local paths first
    if local_path := await _find_gvproxy(override, arch=arch):
        logger.debug("Using cached gvproxy-wrapper", extra={"path": str(local_path)})
        # Ensure executable
        await chmod_executable(local_path)
        return local_path

    # Not found locally, download from GitHub
    await ensure_registry_loaded()
    assets = get_assets()

    fname = f"gvproxy-wrapper-{get_gvproxy_suffix(arch)}"

    logger.debug("Fetching gvproxy-wrapper", extra={"file": fname})
    path = await assets.fetch(fname)

    # Make executable
    await chmod_executable(path)

    return path


async def ensure_assets_available(*, override: Path | None) -> tuple[Path, Path]:
    """
    Ensure the kernel, initramfs and gvproxy are available (base images are fetched
    per language, on first use: fetch_base_image).

    Downloads assets from GitHub Releases if not already cached.
    In offline mode, missing assets raise ConfigError ASSET_NOT_FOUND (no download).

    Args:
        override: Explicit path to search first (from SchedulerConfig.images_dir).

    Returns:
        Tuple of (images_dir, gvproxy_path)

    Raises:
        ConfigError: ASSET_NOT_FOUND: release not found on GitHub, or offline mode
            and assets missing.
        TransientError: ASSET_DOWNLOAD: download or hash verification failed after retries.
    """
    # Fetch required assets in parallel
    fetch_tasks: list[asyncio.Task[Path]] = [
        asyncio.create_task(fetch_kernel(override=override)),
        asyncio.create_task(fetch_initramfs(override=override)),
        asyncio.create_task(fetch_gvproxy(override=override, arch=get_current_arch())),
    ]

    results = await asyncio.gather(*fetch_tasks)
    kernel_path, _, gvproxy_path = results[0], results[1], results[2]

    # Images directory is the parent of the kernel
    images_dir = kernel_path.parent

    logger.info(
        "Assets ready",
        extra={"images_dir": str(images_dir), "gvproxy": str(gvproxy_path)},
    )

    return images_dir, gvproxy_path


async def ensure_assets(*, override: Path | None, download: bool) -> Path:
    """
    Find or download assets. Single entry point for asset resolution.

    Args:
        override: Explicit path from SchedulerConfig.images_dir.
        download: If True, download missing assets. If False, raise on missing.

    Returns:
        Path to images directory.

    Raises:
        ConfigError: ASSET_NOT_FOUND: assets not found and download=False, or release not found on GitHub.
        TransientError: ASSET_DOWNLOAD: download or hash verification failed after retries.
    """
    # Try to find existing assets first
    if images_dir := await _find_images_dir(override):
        # Validate that essential files exist (not just the directory)
        arch = get_current_arch()
        if await asyncio.to_thread(_first_existing, [images_dir / name for name in _kernel_candidates(arch)]):
            logger.debug("Found existing assets", extra={"images_dir": str(images_dir)})
            return images_dir
        # Directory exists but kernel is missing - fall through to download
        logger.debug(
            "Directory exists but kernel missing",
            extra={"images_dir": str(images_dir)},
        )

    # Not found - either download or error
    if not download:
        search_paths = _get_search_paths(override)
        raise ConfigError(
            f"Assets not found and auto_download_assets=False. "
            f"Searched: {[str(p) for p in search_paths]}. "
            f"Set EXEC_SANDBOX_IMAGES_DIR or enable auto_download_assets.",
            code=ErrorCode.ASSET_NOT_FOUND,
            context=None,
            hint=None,
            instance_id=None,
        )

    # Download assets and return images directory
    images_dir, _ = await ensure_assets_available(override=override)
    return images_dir


def _get_search_paths(override: Path | None = None) -> list[Path]:
    """
    Get list of directories to search for assets.

    Priority order:
    1. override (explicit from SchedulerConfig.images_dir)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. Local build directory (./images/dist/ relative to package root)
    4. Download cache directory (~/.cache/exec-sandbox/v{version}/)

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
    paths.append(_versioned_cache_dir())

    return paths


async def _find_images_dir(override: Path | None = None) -> Path | None:
    """
    Find first existing images directory.

    Checks multiple locations in priority order:
    1. override (explicit from SchedulerConfig.images_dir)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. Local build directory (./images/dist/ relative to package root)
    4. Download cache directory (~/.cache/exec-sandbox/v{version}/)

    Args:
        override: Explicit path from config (highest priority).

    Returns:
        Path to images directory if found, None otherwise.
    """
    return await asyncio.to_thread(_first_existing, _get_search_paths(override))


async def _find_asset(fname: str, override: Path | None = None) -> Path | None:
    """
    Find specific asset file without downloading.

    Checks multiple locations in priority order:
    1. override (explicit from SchedulerConfig.images_dir)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. Local build directory (./images/dist/ relative to package root)
    4. Download cache directory (~/.cache/exec-sandbox/v{version}/)

    Also checks for decompressed versions (.zst removed).

    Args:
        fname: Asset filename.
        override: Explicit path from config (highest priority).

    Returns:
        Path to the asset file if found, None otherwise.
    """
    return await asyncio.to_thread(_first_existing, _asset_candidates(fname, override))


def _asset_candidates(fname: str, override: Path | None) -> list[Path]:
    """Where _find_asset looks, in order: the ready file (`.zst` removed) in each search folder."""
    return [directory / fname.removesuffix(".zst") for directory in _get_search_paths(override)]


def _first_existing(paths: list[Path]) -> Path | None:
    """The first of `paths` that exists. Blocking: the search folders include the
    caller's (images_dir, EXEC_SANDBOX_IMAGES_DIR), which can be on a network disk,
    so async callers run one lookup as one worker-thread job."""
    # os.path.exists: a folder this process may not read counts as missing
    # (Path.exists raises PermissionError before Python 3.14).
    return next((path for path in paths if os.path.exists(path)), None)  # noqa: PTH110


async def prefetch_all_assets(
    arch: str | None = None,
) -> PrefetchResult:
    """
    Pre-download all VM assets for offline use.

    Downloads kernel, initramfs, gvproxy, and all base images (from BASE_IMAGES)
    for the specified architecture. Useful for pre-populating cache in container
    builds before running with EXEC_SANDBOX_OFFLINE=1.

    Args:
        arch: Target architecture ("x86_64" or "aarch64"). Defaults to current machine.

    Returns:
        PrefetchResult with success status, downloaded assets, and any errors.
    """
    arch = arch or get_current_arch()
    logger.info("Prefetching all assets", extra={"arch": arch})

    # Load registry first
    try:
        await ensure_registry_loaded()
    except Exception as e:  # noqa: BLE001 - Catch broad exception for robustness
        logger.error("Failed to load registry", extra={"error": str(e)})
        return PrefetchResult(
            success=False,
            arch=arch,
            errors=[("registry", str(e))],
        )

    # Define all assets to fetch - core assets plus base images for all languages
    asset_tasks: list[tuple[str, asyncio.Task[Path]]] = [
        ("kernel", asyncio.create_task(fetch_kernel(arch=arch))),
        ("initramfs", asyncio.create_task(fetch_initramfs(arch=arch))),
        ("gvproxy", asyncio.create_task(fetch_gvproxy(override=None, arch=arch))),
    ]
    # Add base images for all supported languages
    asset_tasks.extend(
        (f"{lang.value} base", asyncio.create_task(fetch_base_image(lang.value, arch=arch))) for lang in Language
    )

    # Gather results with exception handling
    tasks = [task for _, task in asset_tasks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    downloaded: list[str] = []
    errors: list[tuple[str, str]] = []
    for (name, _), result in zip(asset_tasks, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("Asset download failed", extra={"asset": name, "error": str(result)})
            errors.append((name, str(result)))
        else:
            logger.debug("Asset downloaded", extra={"asset": name, "path": str(result)})
            downloaded.append(name)

    if errors:
        logger.error("Prefetch completed with errors", extra={"errors": len(errors), "downloaded": len(downloaded)})
    else:
        logger.info("Prefetch completed successfully", extra={"downloaded": len(downloaded)})

    return PrefetchResult(
        success=len(errors) == 0,
        arch=arch,
        downloaded=downloaded,
        errors=errors,
        cache_dir=_versioned_cache_dir() if len(errors) == 0 else None,
    )

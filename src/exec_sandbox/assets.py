"""
exec-sandbox asset registry and fetch functions.

Provides lazy downloading of VM images and binaries from GitHub Releases.
Uses AsyncPooch for caching and checksum verification.

Path Resolution Priority:
    1. override (explicit from SchedulerConfig.images_dir)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. ./images/dist/ (local build directory)
    4. Download cache (images_download_dir): v{version}/ in the user cache folder
       (~/.cache/exec-sandbox on Linux, ~/Library/Caches/exec-sandbox on macOS), or
       EXEC_SANDBOX_CACHE_DIR itself when it is set
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
    get_current_arch,
    get_gvproxy_suffix,
)
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.images import BASE_IMAGES, RELEASE_SUFFIX, initramfs_name, kernel_dir_of, kernel_names
from exec_sandbox.models import Language
from exec_sandbox.permission_utils import chmod_executable
from exec_sandbox.platform_utils import UserDir, arch_of, get_user_dir

logger = get_logger(__name__)

# Public API - only these should be used externally
__all__ = [
    "PrefetchResult",
    "ensure_assets",
    "ensure_assets_available",
    "ensure_registry_loaded",
    "fetch_gvproxy",
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


def images_download_dir() -> Path:
    """The folder where downloaded guest files (kernel, initramfs, base images) land: this
    version's folder in the user cache (no version folder under EXEC_SANDBOX_CACHE_DIR)."""
    base = get_user_dir(kind=UserDir.CACHE)
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
        path=images_download_dir(),
        base_url=f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/v{{version}}",
        version=pooch_version,
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


async def _download_kernel(arch: str) -> Path:
    """The kernel of `arch` from the release, unpacked into images_download_dir(): the first
    of kernel_names that the release has (vmlinux for x86_64, then vmlinuz); the last one when
    none is listed (its fetch reports it missing)."""
    await ensure_registry_loaded()
    assets = get_assets()
    download_candidates = [f"{name}{RELEASE_SUFFIX}" for name in kernel_names(arch_of(arch))]
    name = next((name for name in download_candidates if name in assets.registry), download_candidates[-1])
    logger.debug("Fetching kernel", extra={"arch": arch, "file": name})
    return await assets.fetch(name, decompress=True)


async def _download(fname: str) -> Path:
    """`fname` (a release name, .zst) from the release, unpacked into images_download_dir()."""
    await ensure_registry_loaded()
    return await get_assets().fetch(fname, decompress=True)


async def _cached(names: tuple[str, ...]) -> Path | None:
    """The first of `names` (unpacked release names) in images_download_dir(), or None. One
    worker-thread job."""
    cache = images_download_dir()
    return await asyncio.to_thread(_first_existing, [cache / name for name in names])


async def _kernel_in_cache(arch: str) -> Path:
    """The kernel of `arch` in images_download_dir(), downloaded there when missing."""
    return await _cached(kernel_names(arch_of(arch))) or await _download_kernel(arch)


async def _in_cache(name: str) -> Path:
    """`name` (an unpacked release name) in images_download_dir(), downloaded there when missing."""
    return await _cached((name,)) or await _download(name + RELEASE_SUFFIX)


async def ensure_base_image(language: Language, *, images_dir: Path) -> None:
    """Have the base image of `language` where GuestImages looks for it: in `images_dir`, else
    in images_download_dir(), downloaded there when neither holds it. Never from another
    search folder, which GuestImages does not read.

    Raises:
        ConfigError: ASSET_NOT_FOUND: not in the release, or offline mode
        TransientError: ASSET_DOWNLOAD: the download or its hash check failed after retries
    """
    arch = get_current_arch()
    name = BASE_IMAGES[language].file_name(arch_of(arch))
    if await asyncio.to_thread(_first_existing, [images_dir / name, images_download_dir() / name]):
        return
    logger.debug("Fetching base image", extra={"language": language.value, "arch": arch, "file": name})
    await _download(name + RELEASE_SUFFIX)


async def get_gvproxy_path(override: Path | None = None) -> Path | None:
    """
    Get path to gvproxy-wrapper binary for current platform (without downloading).

    Detection order:
    0. override (explicit from SchedulerConfig.images_dir)
    1. Repo-relative path (gvproxy-wrapper/bin/ - for local development builds)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. Local build directory (./images/dist/)
    4. Download cache directory (images_download_dir)

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
    candidates.extend(directory / fname for directory in _get_search_paths(override))
    return await asyncio.to_thread(_first_existing, candidates)


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
    path = await assets.fetch(fname, decompress=False)

    # Make executable
    await chmod_executable(path)

    return path


async def ensure_assets_available() -> tuple[Path, Path]:
    """
    Make the download cache hold the boot files (the kernel and the initramfs), and find or
    download gvproxy (base images are fetched per language, on first use: ensure_base_image).

    The kernel and the initramfs are taken from images_download_dir(), or downloaded into it,
    never from another search folder: the images folder must hold both. gvproxy-wrapper
    is searched with no override (SchedulerConfig.images_dir is not the images folder here, and
    a VM start searches the images folder first, then the same folders). In offline mode,
    missing assets raise ConfigError ASSET_NOT_FOUND (no download).

    Returns:
        Tuple of (images_dir: the download cache, gvproxy_path)

    Raises:
        ConfigError: ASSET_NOT_FOUND: release not found on GitHub, or offline mode
            and assets missing.
        TransientError: ASSET_DOWNLOAD: the release metadata could not be read, or a
            download or hash verification failed after retries.
    """
    arch = get_current_arch()
    kernel_path, _, gvproxy_path = await asyncio.gather(
        _kernel_in_cache(arch),
        _in_cache(initramfs_name(arch_of(arch))),
        fetch_gvproxy(override=None, arch=arch),
    )
    images_dir = kernel_path.parent

    logger.info(
        "Assets ready",
        extra={"images_dir": str(images_dir), "gvproxy": str(gvproxy_path)},
    )

    return images_dir, gvproxy_path


async def ensure_assets(*, override: Path | None, download: bool) -> Path:
    """
    Find or download assets. Single entry point for asset resolution.

    With `download`, a found images folder also gets gvproxy-wrapper where a VM start looks
    for it (best effort: a failure is logged).

    Args:
        override: Explicit path from SchedulerConfig.images_dir.
        download: If True, download missing assets. If False, raise on missing.

    Returns:
        Path to images directory.

    Raises:
        ConfigError: ASSET_NOT_FOUND: assets not found and download=False, or release not found on GitHub.
        TransientError: ASSET_DOWNLOAD: download or hash verification failed after retries.
    """
    # The first folder, in search order, that holds the boot files: a folder that exists
    # with none or one of them (an empty images/dist) must not hide one that has both
    if images_dir := await _find_images_dir(override):
        logger.debug("Found existing assets", extra={"images_dir": str(images_dir)})
        if download:
            # gvproxy-wrapper where a VM start looks for it: the images folder first.
            # Best effort: a VM with no network needs none; one with network then fails with
            # HOST_DEPENDENCY at its start. OSError: a read-only folder or a file of another
            # user refuses the chmod
            try:
                await fetch_gvproxy(override=images_dir, arch=get_current_arch())
            except (ConfigError, TransientError, OSError) as e:
                logger.warning("gvproxy-wrapper not available: VMs with network cannot start", extra={"error": str(e)})
        return images_dir

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
    images_dir, _ = await ensure_assets_available()
    return images_dir


def _get_search_paths(override: Path | None) -> list[Path]:
    """
    Get list of directories to search for assets.

    Priority order:
    1. override (explicit from SchedulerConfig.images_dir)
    2. EXEC_SANDBOX_IMAGES_DIR env var
    3. Local build directory (./images/dist/ relative to package root)
    4. Download cache directory (images_download_dir)

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
    paths.append(images_download_dir())

    return paths


async def _find_images_dir(override: Path | None) -> Path | None:
    """The first images folder, in search order (_get_search_paths: override,
    EXEC_SANDBOX_IMAGES_DIR, the local build, the download cache), that holds the boot files of
    this host's architecture: a kernel and the initramfs, in the folder or in its kernels/
    subfolder (kernel_dir_of, as the settings read them). A folder with only one of them is
    passed over: VMs read both from the images folder. None when no folder does.
    One worker-thread job (a few stat calls).
    """
    host = arch_of(get_current_arch())
    kernels = kernel_names(host)
    initramfs = initramfs_name(host)

    def first_with_boot_files() -> Path | None:
        for folder in _get_search_paths(override):
            boot = kernel_dir_of(folder, arch=host)
            if _first_existing([boot / name for name in kernels]) and _first_existing([boot / initramfs]):
                return folder
        return None

    return await asyncio.to_thread(first_with_boot_files)


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

    Puts the kernel, the initramfs and all base images of the architecture in the download
    cache (images_download_dir(): what an offline start reads when no other folder holds the
    boot files), downloading each that the cache lacks; a copy in another search folder does
    not count. Also finds or downloads gvproxy-wrapper (of this OS). Useful for
    pre-populating cache in container builds before running with EXEC_SANDBOX_OFFLINE=1.

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
        ("kernel", asyncio.create_task(_kernel_in_cache(arch))),
        ("initramfs", asyncio.create_task(_in_cache(initramfs_name(arch_of(arch))))),
        ("gvproxy", asyncio.create_task(fetch_gvproxy(override=None, arch=arch))),
    ]
    # Add base images for all supported languages
    asset_tasks.extend(
        (f"{lang.value} base", asyncio.create_task(_in_cache(BASE_IMAGES[lang].file_name(arch_of(arch)))))
        for lang in Language
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
        cache_dir=images_download_dir() if len(errors) == 0 else None,
    )

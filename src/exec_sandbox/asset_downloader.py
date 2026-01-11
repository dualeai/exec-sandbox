"""
Async file downloader with caching, inspired by Pooch.

Uses aiohttp for async HTTP, backports.zstd (PEP-784) for decompression.
Provides zero-memory-copy streaming for large files.
"""

from __future__ import annotations

import asyncio
import hashlib
import http
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
import aiohttp

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import AssetChecksumError, AssetDownloadError, AssetNotFoundError
from exec_sandbox.platform_utils import HostOS, detect_host_os, get_arch_name, get_os_name

logger = get_logger(__name__)

# Chunk size for streaming downloads and decompression (64KB)
CHUNK_SIZE = 64 * 1024

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds


def get_cache_dir(app_name: str = "exec-sandbox") -> Path:
    """
    Get platform-specific cache directory.

    Returns:
        - macOS: ~/Library/Caches/<app_name>/
        - Linux: ~/.cache/<app_name>/ (or $XDG_CACHE_HOME/<app_name>/)

    Override with EXEC_SANDBOX_CACHE_DIR environment variable.
    """
    if env_path := os.environ.get("EXEC_SANDBOX_CACHE_DIR"):
        return Path(env_path)

    match detect_host_os():
        case HostOS.MACOS:
            return Path.home() / "Library" / "Caches" / app_name
        case HostOS.LINUX:
            # XDG_CACHE_HOME takes precedence if set
            xdg_cache = os.environ.get("XDG_CACHE_HOME")
            if xdg_cache:
                return Path(xdg_cache) / app_name
            return Path.home() / ".cache" / app_name
        case _:
            # Fallback for other platforms
            return Path.home() / ".cache" / app_name


def os_cache(app_name: str) -> Path:
    """Alias for get_cache_dir (pooch compatibility)."""
    return get_cache_dir(app_name)


async def retrieve(
    url: str,
    known_hash: str,
    path: Path | None = None,
    progressbar: bool = True,
    processor: Callable[[Path], Awaitable[Path]] | None = None,
) -> Path:
    """
    Download a file and cache it locally (async version of pooch.retrieve).

    If the file exists and hash matches, returns cached path immediately.
    Otherwise downloads, verifies hash, and optionally processes (e.g., decompress).

    Args:
        url: URL to download from
        known_hash: Expected hash in format "sha256:abc123..."
        path: Local cache directory (default: platform cache dir)
        progressbar: Show download progress in logs
        processor: Post-download processor (e.g., decompress_zstd)

    Returns:
        Path to the downloaded (and optionally processed) file

    Raises:
        AssetDownloadError: Download failed after retries
        AssetChecksumError: Hash verification failed

    Example:
        fname = await retrieve(
            url="https://github.com/dualeai/exec-sandbox/releases/download/v0.1.0/vmlinuz-x86_64.zst",
            known_hash="sha256:abc123...",
            processor=decompress_zstd,
        )
    """
    cache_dir = path or get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract filename from URL
    fname = url.split("/")[-1]
    dest = cache_dir / fname

    # Check if already cached with correct hash
    if dest.exists():
        if await _verify_hash(dest, known_hash):
            logger.debug("Cache hit", extra={"file": str(dest)})
            return dest
        logger.warning("Cache file hash mismatch, re-downloading", extra={"file": str(dest)})
        dest.unlink()

    # Download with retries
    await _download_with_retry(url, dest, known_hash, progressbar)

    # Apply processor if provided (e.g., decompress)
    if processor:
        dest = await processor(dest)

    return dest


async def _verify_hash(path: Path, expected_hash: str) -> bool:
    """Verify file hash matches expected value."""
    if not expected_hash:
        return True

    algorithm, expected_digest = expected_hash.split(":", 1)

    def _compute_hash() -> str:
        hasher = hashlib.new(algorithm)
        with path.open("rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                hasher.update(chunk)
        return hasher.hexdigest()

    actual_digest = await asyncio.to_thread(_compute_hash)
    return actual_digest == expected_digest


async def _download_with_retry(
    url: str,
    dest: Path,
    expected_hash: str,
    progressbar: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> None:
    """Download file with exponential backoff retry."""
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            await _download_file(url, dest, expected_hash, progressbar)
            return
        except (aiohttp.ClientError, AssetChecksumError, OSError) as e:
            last_error = e
            if dest.exists():
                dest.unlink()

            if attempt < max_retries - 1:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    "Download failed, retrying",
                    extra={"url": url, "attempt": attempt + 1, "delay": delay, "error": str(e)},
                )
                await asyncio.sleep(delay)

    raise AssetDownloadError(
        f"Failed to download {url} after {max_retries} attempts",
        context={"url": url, "last_error": str(last_error)},
    )


async def _download_file(url: str, dest: Path, expected_hash: str, progressbar: bool = True) -> None:
    """
    Stream download with incremental hash verification.

    Uses chunked reads to minimize memory usage (~64KB per download).
    Hash is computed incrementally during download (no second pass).
    Uses atomic write pattern (temp file -> rename).
    """
    algorithm, expected_digest = expected_hash.split(":", 1) if expected_hash else ("sha256", "")
    hasher = hashlib.new(algorithm)
    temp_path = dest.with_suffix(dest.suffix + ".tmp")

    if progressbar:
        logger.info("Downloading asset", extra={"url": url, "dest": str(dest)})

    try:
        async with aiohttp.ClientSession() as session, session.get(url) as resp:
            resp.raise_for_status()

            total_size = resp.content_length
            downloaded = 0

            async with aiofiles.open(temp_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                    hasher.update(chunk)  # Hash as we go
                    await f.write(chunk)  # Write as we go
                    downloaded += len(chunk)

                    if progressbar and total_size:
                        pct = (downloaded / total_size) * 100
                        if downloaded == total_size or downloaded % (CHUNK_SIZE * 16) == 0:
                            logger.debug(
                                "Download progress",
                                extra={"percent": f"{pct:.1f}%", "downloaded": downloaded, "total": total_size},
                            )

        # Verify hash
        if expected_digest:
            actual_digest = hasher.hexdigest()
            if actual_digest != expected_digest:
                temp_path.unlink(missing_ok=True)
                raise AssetChecksumError(
                    f"Hash mismatch for {url}",
                    context={"expected": expected_hash, "actual": f"{algorithm}:{actual_digest}"},
                )

        # Atomic rename
        temp_path.rename(dest)

        if progressbar:
            logger.info("Download complete", extra={"file": str(dest)})

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


class AsyncPooch:
    """
    Async file registry manager (like pooch.Pooch but async).

    Example:
        ASSETS = AsyncPooch(
            path=get_cache_dir(),
            base_url="https://github.com/dualeai/exec-sandbox/releases/download/{version}",
            version=__version__,
            version_dev="latest",
            env="EXEC_SANDBOX_CACHE_DIR",
            registry={
                "vmlinuz-x86_64.zst": "sha256:abc123...",
                "python-3.14-base-x86_64.qcow2.zst": "sha256:def456...",
            },
        )

        kernel_path = await ASSETS.fetch("vmlinuz-x86_64.zst", processor=decompress_zstd)
    """

    def __init__(
        self,
        path: Path,
        base_url: str,
        version: str,
        version_dev: str = "latest",
        env: str | None = None,
        registry: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize AsyncPooch registry.

        Args:
            path: Local cache directory
            base_url: Base URL template with {version} placeholder
            version: Current version string
            version_dev: Version to use for dev builds (default: "latest")
            env: Environment variable to override cache path
            registry: Dict mapping filename to hash (e.g., {"file.txt": "sha256:abc123..."})
        """
        # Check env override
        if env and (env_path := os.environ.get(env)):
            self.path = Path(env_path)
        else:
            self.path = path

        self.base_url = base_url
        self.version = version
        self.version_dev = version_dev
        self.registry: dict[str, str] = registry or {}
        self._github_release_cache: dict[str, dict[str, Any]] | None = None

    async def fetch(
        self,
        fname: str,
        processor: Callable[[Path], Awaitable[Path]] | None = None,
        progressbar: bool = True,
    ) -> Path:
        """
        Fetch a file from the registry, downloading if needed.

        Args:
            fname: Filename to fetch (must be in registry)
            processor: Post-download processor (e.g., decompress_zstd)
            progressbar: Show download progress

        Returns:
            Path to the local file

        Raises:
            AssetNotFoundError: File not in registry
            AssetDownloadError: Download failed
            AssetChecksumError: Hash verification failed
        """
        if fname not in self.registry:
            raise AssetNotFoundError(
                f"File '{fname}' not in registry",
                context={"available": list(self.registry.keys())},
            )

        known_hash = self.registry[fname]

        # Build URL with version
        version = self.version if not self.version.endswith(".dev0") else self.version_dev
        url = self.base_url.format(version=version) + "/" + fname

        return await retrieve(
            url=url,
            known_hash=known_hash,
            path=self.path,
            progressbar=progressbar,
            processor=processor,
        )

    def load_registry_from_file(self, registry_file: Path) -> None:
        """
        Load registry from a file (filename hash per line).

        File format:
            vmlinuz-x86_64.zst sha256:abc123...
            python-base.qcow2.zst sha256:def456...
        """
        min_parts = 2
        with registry_file.open() as f:
            for raw_line in f:
                stripped_line = raw_line.strip()
                if not stripped_line or stripped_line.startswith("#"):
                    continue
                parts = stripped_line.split()
                if len(parts) >= min_parts:
                    self.registry[parts[0]] = parts[1]

    async def load_registry_from_github(self, owner: str, repo: str, tag: str) -> None:
        """
        Load registry from GitHub release API (uses asset digests).

        GitHub automatically computes SHA256 checksums for release assets (since June 2025).

        Args:
            owner: GitHub repository owner
            repo: GitHub repository name
            tag: Release tag (e.g., "v0.1.0" or "latest")
        """
        # Use "latest" endpoint for latest release
        if tag == "latest":
            api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        else:
            api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"

        logger.debug("Fetching release metadata from GitHub", extra={"url": api_url})

        async with aiohttp.ClientSession() as session:
            headers = {"Accept": "application/vnd.github+json"}

            # Add auth token if available (increases rate limit)
            if token := os.environ.get("GITHUB_TOKEN"):
                headers["Authorization"] = f"Bearer {token}"

            async with session.get(api_url, headers=headers) as resp:
                if resp.status == http.HTTPStatus.NOT_FOUND:
                    raise AssetNotFoundError(
                        f"Release not found: {tag}",
                        context={"owner": owner, "repo": repo, "tag": tag},
                    )
                resp.raise_for_status()
                release_data = await resp.json()

        # Cache release data for URL lookups
        self._github_release_cache = {asset["name"]: asset for asset in release_data.get("assets", [])}

        # Extract asset names and digests
        for asset in release_data.get("assets", []):
            name = asset["name"]
            # GitHub provides digest in format "sha256:abc123..."
            if digest := asset.get("digest"):
                self.registry[name] = digest
            else:
                # Fallback: no digest available (older releases)
                logger.warning("No digest available for asset", extra={"asset": name})
                self.registry[name] = ""

        logger.info(
            "Loaded registry from GitHub",
            extra={"tag": tag, "assets": len(self.registry)},
        )


# ============================================================================
# Processors (post-download transformations)
# ============================================================================


async def decompress_zstd(fname: Path) -> Path:
    """
    Decompress .zst file with streaming (zero-copy).

    Uses Python's native zstd module (3.14+) or backports.zstd for older versions.
    Streaming decompression ensures minimal memory usage regardless of file size.

    Memory usage: ~64KB (chunk size) regardless of file size.

    Args:
        fname: Path to .zst compressed file

    Returns:
        Path to decompressed file (original .zst is deleted)
    """
    import sys  # noqa: PLC0415

    # Use native zstd module (Python 3.14+) or backports.zstd
    if sys.version_info >= (3, 14):
        from compression import zstd  # noqa: PLC0415
    else:
        from backports import zstd  # noqa: PLC0415

    dest = fname.with_suffix("")  # Remove .zst

    logger.debug("Decompressing", extra={"src": str(fname), "dest": str(dest)})

    def _decompress_sync() -> None:
        with fname.open("rb") as src, dest.open("wb") as dst:
            # Use streaming decompression with incremental decompressor
            decompressor = zstd.ZstdDecompressor()
            while True:
                chunk = src.read(CHUNK_SIZE)
                if not chunk:
                    break
                decompressed = decompressor.decompress(chunk)
                if decompressed:
                    dst.write(decompressed)
        fname.unlink()  # Remove compressed file

    await asyncio.to_thread(_decompress_sync)

    logger.debug("Decompression complete", extra={"file": str(dest)})
    return dest


async def untar(fname: Path) -> Path:
    """
    Extract tar archive, return path to extracted directory.

    Args:
        fname: Path to tar archive

    Returns:
        Path to extracted directory
    """
    import tarfile  # noqa: PLC0415

    dest_dir = fname.parent / fname.stem.replace(".tar", "")

    def _extract_sync() -> None:
        with tarfile.open(fname) as tar:
            tar.extractall(path=dest_dir, filter="data")  # Secure extraction filter
        fname.unlink()

    await asyncio.to_thread(_extract_sync)
    return dest_dir


def get_current_arch() -> str:
    """
    Get current machine architecture in a normalized format.

    Returns:
        Architecture string: "x86_64" or "aarch64"
    """
    return get_arch_name("kernel")


def get_gvproxy_suffix() -> str:
    """
    Get gvproxy binary suffix for current platform.

    Returns:
        Suffix like "darwin-arm64", "darwin-amd64", "linux-arm64", "linux-amd64"
    """
    return f"{get_os_name()}-{get_arch_name('go')}"

"""
Async file downloader with caching, inspired by Pooch.

Uses aiohttp for async HTTP, backports.zstd (PEP-784) for decompression.
Provides zero-memory-copy streaming for large files.
"""

from __future__ import annotations

import asyncio
import functools
import http
import os
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import run_to_end
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.hash_utils import IncrementalHasher, file_hash, parse_hash_spec
from exec_sandbox.lock_utils import file_lock
from exec_sandbox.platform_utils import UserDir, get_arch_name, get_os_name, get_user_dir

logger = get_logger(__name__)

# Chunk size for streaming decompression (64KB)
CHUNK_SIZE = 64 * 1024

# Download chunk: one worker-thread job (hash + write) each. A job hop costs 70 us on
# an idle host, 700 us on a loaded one: with 64 KiB chunks the hops cap a download at
# 600 MB/s (idle) to 80 MB/s (loaded); with 1 MiB chunks, 2100 MB/s (idle).
_WRITE_CHUNK = 1024 * 1024

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds, doubled at each attempt


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

    # Use a file lock to prevent concurrent download+decompress of the same asset
    # across both coroutines and processes.  Without this, two callers can race:
    # one's retry cleanup (dest.unlink) deletes the .zst while another decompresses it.
    # Lock target is the final output (e.g., .qcow2 after decompressing .qcow2.zst).
    # Lock on the final output to serialize the full download+process pipeline.
    # For decompress_zstd the final file is dest without .zst; otherwise dest itself.
    final_dest = dest.with_suffix("") if processor == decompress_zstd else dest
    async with file_lock(final_dest) as should_download:
        if not should_download:
            # With a processor (e.g., decompress), the final output's existence
            # is authoritative — hash was verified on the raw download before
            # processing.  Without a processor, verify hash to detect corruption.
            if processor or await _verify_hash(final_dest, known_hash):
                logger.debug("Already cached by another caller", extra={"file": str(final_dest)})
                return final_dest
            # Corrupted cache — clean up and fall through to re-download
            logger.warning("Cache file hash mismatch (post-lock)", extra={"file": str(final_dest)})
            final_dest.unlink(missing_ok=True)

        # Check if raw download is already cached with correct hash (skip re-download)
        if not dest.exists() or not await _verify_hash(dest, known_hash):
            if dest.exists():
                logger.warning("Cache file hash mismatch, re-downloading", extra={"file": str(dest)})
                dest.unlink()
            await _download_with_retry(url, dest, known_hash, progressbar)

        # Apply processor if provided (e.g., decompress)
        if processor:
            dest = await processor(dest)

        return dest


async def _verify_hash(path: Path, expected_hash: str) -> bool:
    """Verify file hash matches expected value."""
    if not expected_hash:
        return True

    algorithm, expected_digest = parse_hash_spec(expected_hash)
    actual_digest = await file_hash(path, algorithm)
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
    algorithm, expected_digest = parse_hash_spec(expected_hash) if expected_hash else ("sha256", "")
    hasher = IncrementalHasher(algorithm)
    temp_path = dest.with_suffix(dest.suffix + ".tmp")

    if progressbar:
        logger.info("Downloading asset", extra={"url": url, "dest": str(dest)})

    try:
        async with aiohttp.ClientSession() as session, session.get(url) as resp:
            resp.raise_for_status()

            total_size = resp.content_length
            downloaded = 0

            f = temp_path.open("wb")
            try:
                async for chunk in resp.content.iter_chunked(_WRITE_CHUNK):
                    # Hash and write as we go, in one worker-thread job per chunk: a write
                    # into the page cache can wait for writeback. A cancel waits for the
                    # job, so the file never closes under a write.
                    await run_to_end(functools.partial(_hash_and_write, chunk, hasher=hasher, sink=f))
                    downloaded += len(chunk)

                    if progressbar and total_size:
                        pct = (downloaded / total_size) * 100
                        if downloaded == total_size or downloaded % _WRITE_CHUNK == 0:
                            logger.debug(
                                "Download progress",
                                extra={"percent": f"{pct:.1f}%", "downloaded": downloaded, "total": total_size},
                            )
            finally:
                # In a worker thread: the cache folder can be on NFS, where close writes
                # back every dirty page of the image. A cancel waits for the close.
                await run_to_end(f.close)

        # Verify hash
        if expected_digest:
            actual_digest = hasher.hexdigest()
            if actual_digest != expected_digest:
                temp_path.unlink(missing_ok=True)
                raise TransientError(
                    f"Hash mismatch for {url}",
                    code=ErrorCode.ASSET_DOWNLOAD,
                    context={"expected": expected_hash, "actual": f"{algorithm}:{actual_digest}"},
                    hint=None,
                    instance_id=None,
                )

        # Atomic rename
        temp_path.rename(dest)

        if progressbar:
            logger.info("Download complete", extra={"file": str(dest)})

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _hash_and_write(chunk: bytes, *, hasher: IncrementalHasher, sink: BinaryIO) -> None:
    hasher.update(chunk)
    sink.write(chunk)


class AsyncPooch:
    """
    Async file registry manager (like pooch.Pooch but async).

    Example:
        ASSETS = AsyncPooch(
            path=get_user_dir(kind=UserDir.CACHE, app_name="exec-sandbox"),
            base_url="https://github.com/dualeai/exec-sandbox/releases/download/{version}",
            version=__version__,
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
        env: str | None = None,
        registry: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize AsyncPooch registry.

        Args:
            path: Local cache directory
            base_url: Base URL template with {version} placeholder
            version: Current version string (used as fallback if GitHub API not called)
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
        self.registry: dict[str, str] = registry or {}
        self._github_release_cache: dict[str, dict[str, Any]] | None = None
        self._resolved_version: str | None = None  # Actual tag from GitHub API

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
            ConfigError: ASSET_NOT_FOUND: file not in registry
            TransientError: ASSET_DOWNLOAD: download or hash verification failed
        """
        if fname not in self.registry:
            raise ConfigError(
                f"File '{fname}' not in registry",
                code=ErrorCode.ASSET_NOT_FOUND,
                context={"available": list(self.registry.keys())},
                hint=None,
                instance_id=None,
            )

        known_hash = self.registry[fname]

        # Build URL with version
        # Use resolved version from GitHub API if available (e.g., "v0.2.1" from /releases/latest)
        # Otherwise fall back to configured version
        version = self._resolved_version.lstrip("v") if self._resolved_version else self.version
        url = self.base_url.format(version=version) + "/" + fname

        return await retrieve(
            url=url,
            known_hash=known_hash,
            path=self.path,
            progressbar=progressbar,
            processor=processor,
        )

    async def load_registry_from_file(self, registry_file: Path) -> None:
        """
        Load registry from a file (filename hash per line).

        File format:
            vmlinuz-x86_64.zst sha256:abc123...
            python-base.qcow2.zst sha256:def456...
        """
        min_parts = 2
        async with aiofiles.open(registry_file) as f:
            content = await f.read()
        for raw_line in content.splitlines():
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

        Raises:
            ConfigError: ASSET_NOT_FOUND: no such release.
            TransientError: ASSET_DOWNLOAD: the metadata could not be read (no network,
                a rate limit, a server error, a body that is not JSON).
        """
        import aiohttp  # noqa: PLC0415 - only a download needs it (~50 ms at `import exec_sandbox`)

        # Use "latest" endpoint for latest release
        if tag == "latest":
            api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        else:
            api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"

        logger.debug("Fetching release metadata from GitHub", extra={"url": api_url})

        headers = {"Accept": "application/vnd.github+json"}
        # Add auth token if available (increases rate limit)
        if token := os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with aiohttp.ClientSession() as session, session.get(api_url, headers=headers) as resp:
                if resp.status == http.HTTPStatus.NOT_FOUND:
                    raise ConfigError(
                        f"Release not found: {tag}",
                        code=ErrorCode.ASSET_NOT_FOUND,
                        context={"owner": owner, "repo": repo, "tag": tag},
                        hint=None,
                        instance_id=None,
                    )
                resp.raise_for_status()
                release_data = await resp.json()
        except (aiohttp.ClientError, TimeoutError, ValueError) as e:
            # No network, a rate limit (403), a server error or a body that is not JSON:
            # the same call can succeed later.
            raise TransientError(
                f"Could not read the release metadata of {owner}/{repo} {tag}: {e}",
                code=ErrorCode.ASSET_DOWNLOAD,
                context={"owner": owner, "repo": repo, "tag": tag},
                hint=None,
                instance_id=None,
            ) from e

        # Store actual tag name for URL construction (e.g., "v0.2.1" from /releases/latest)
        self._resolved_version = release_data.get("tag_name")
        logger.debug("Resolved release version", extra={"tag": self._resolved_version, "requested": tag})

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
    Uses atomic write pattern (temp file → rename) so partial results are never
    visible.

    Concurrency safety is provided by the caller (``retrieve``'s cross-process
    file lock).  This function must not be called without external serialization.

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

    # Use atomic write pattern: write to temp file, then rename
    temp_dest = dest.with_suffix(".tmp")

    def _decompress_sync() -> None:
        try:
            with fname.open("rb") as src, temp_dest.open("wb") as dst:
                # Use streaming decompression with incremental decompressor
                decompressor = zstd.ZstdDecompressor()
                while True:
                    chunk = src.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    decompressed = decompressor.decompress(chunk)
                    if decompressed:
                        dst.write(decompressed)
            # Atomic rename (POSIX guarantees atomicity)
            temp_dest.rename(dest)
            # Remove compressed file only after successful rename
            fname.unlink()
        except Exception:
            # Clean up temp file on failure
            temp_dest.unlink(missing_ok=True)
            raise

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
    return get_arch_name()


_GO_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}
"""get_current_arch's names to the Go names of the gvproxy-wrapper binaries."""


def get_gvproxy_suffix(arch: str) -> str:
    """
    Get gvproxy binary suffix of this OS for an architecture.

    Args:
        arch: "x86_64" or "aarch64" (get_current_arch's names).

    Returns:
        Suffix like "darwin-arm64", "darwin-amd64", "linux-arm64", "linux-amd64"
    """
    return f"{get_os_name()}-{_GO_ARCH[arch]}"

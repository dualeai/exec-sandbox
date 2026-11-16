"""
Async file downloader with caching, inspired by Pooch.

Uses aiohttp for async HTTP and the zstd module of PEP 784 (compression.zstd,
backports.zstd before Python 3.14) for decompression. Downloads and
decompression stream in chunks: memory use does not grow with the file size.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import http
import os
import sys
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable
    from pathlib import Path

from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import run_stoppable, run_to_end
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.files.disk import create_file
from exec_sandbox.files.locks import CacheEntry
from exec_sandbox.hash_utils import file_hash, parse_hash_spec
from exec_sandbox.permission_utils import make_cache_dir, setting_folder_error
from exec_sandbox.platform_utils import get_os_name

logger = get_logger(__name__)

# Chunk size for streaming decompression (64KB)
CHUNK_SIZE = 64 * 1024

# Download chunk: one worker-thread job (hash + write) each. A job hop costs 70 us on
# an idle host, 700 us on a loaded one: with 64 KiB chunks the hops cap a download at
# 600 MB/s (idle) to 80 MB/s (loaded); with 1 MiB chunks, 2100 MB/s (idle).
_WRITE_CHUNK = 1024 * 1024

_ASSET_MODE = 0o644
"""Mode of a downloaded or decompressed asset: QEMU as qemu-vm reads the kernel, the
initramfs and the base images (nothing else grants it the kernel and the initramfs). No
group write: under `sg qemu-vm` and umask 002, open() would give 0664, group qemu-vm."""

# Retry settings
MAX_RETRIES = 3

_GITHUB_API = "https://api.github.com"
_METADATA_TIMEOUT_SECONDS = 15.0
"""The release metadata request (a small JSON body): connect, send and read in all. With no
limit, a network that drops packets holds each start for aiohttp's defaults (30 s to connect,
300 s in all)."""
_DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 10.0
_DOWNLOAD_READ_TIMEOUT_SECONDS = 30.0
"""A download has no total limit (a large image on a slow link); a stall ends it: no
connection in 10 s, or no byte in 30 s. The retry loop then tries again."""
RETRY_DELAY = 1.0  # seconds, doubled at each attempt


async def retrieve(url: str, known_hash: str, *, path: Path, setting: str, decompress: bool) -> Path:
    """
    Download a file and cache it locally (async version of pooch.retrieve).

    Returns the cached file when it exists: the unpacked file as is (its download
    was verified before decompress_zstd), the file as downloaded once its hash
    matches. Otherwise downloads, verifies hash, and with `decompress` unpacks it.

    Args:
        url: URL to download from
        known_hash: Expected hash in format "sha256:abc123..."
        path: Local cache directory
        setting: The setting that chose `path`, named by its errors
        decompress: Unpack the download (a .zst) with decompress_zstd

    Returns:
        Path to the downloaded (or unpacked) file

    Raises:
        TransientError: ASSET_DOWNLOAD: download or hash verification failed after retries;
            LOCAL_DISK: a host error on the files of `path` (disk full, quota, I/O error), at
            once (no retry)
        ConfigError: SETTING_INVALID: `path` is not a private folder of this user, or this
            user cannot make it or write its files (setting_folder_error); it names `setting`

    Example:
        fname = await retrieve(
            url="https://github.com/dualeai/exec-sandbox/releases/download/v0.1.0/vmlinuz-x86_64.zst",
            known_hash="sha256:abc123...",
            path=cache_dir,
            setting="EXEC_SANDBOX_CACHE_DIR",
            decompress=True,
        )
    """
    make_cache_dir(path, setting=setting)  # 0700 whatever the umask and group (sg qemu-vm, umask 002)
    try:
        return await _retrieve_into(url, known_hash, path=path, decompress=decompress)
    except OSError as e:  # an existing folder this user cannot write, a full disk
        raise setting_folder_error(e, path=path, failure="cannot hold the download", setting=setting) from e


async def _retrieve_into(url: str, known_hash: str, *, path: Path, decompress: bool) -> Path:
    """retrieve in a folder that exists: lock, download, verify and unpack the asset."""
    fname = url.rsplit("/", maxsplit=1)[-1]
    dest = path / fname

    # The save lock of the final output (CacheEntry.saving) serializes the whole
    # download+process pipeline of one asset across coroutines and processes.
    # Without it, two callers can race: one's retry cleanup (dest.unlink) deletes
    # the .zst while another decompresses it. Unpacked, the final file is dest
    # without .zst (.qcow2 after decompressing .qcow2.zst); otherwise dest itself.
    final_dest = dest.with_suffix("") if decompress else dest
    async with CacheEntry(final_dest).saving() as should_download:
        if not should_download:
            # Unpacked, the final output's existence is authoritative — hash was
            # verified on the raw download before unpacking. Else, verify hash to
            # detect corruption.
            if decompress or await _verify_hash(final_dest, known_hash):
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
            await _download_with_retry(url, dest, expected_hash=known_hash)

        if decompress:
            dest = await decompress_zstd(dest)

        return dest


async def _verify_hash(path: Path, expected_hash: str) -> bool:
    """Whether the file's digest is `expected_hash` ("algorithm:digest"). An empty
    `expected_hash` (a release asset with no digest) always matches."""
    if not expected_hash:
        return True

    algorithm, expected_digest = parse_hash_spec(expected_hash)
    actual_digest = await file_hash(path, algorithm)
    return actual_digest == expected_digest


async def _download_with_retry(url: str, dest: Path, *, expected_hash: str) -> None:
    """Download file with exponential backoff retry. Network errors, timeouts and a hash
    mismatch are retried; a file error of the cache folder (an OSError that is not aiohttp's)
    is raised at once: a retry gets it again or fills the same disk (retrieve maps it)."""
    import aiohttp  # noqa: PLC0415 - only a download needs it (~50 ms at `import exec_sandbox`)

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            await _download_file(url, dest, expected_hash=expected_hash)
            return
        except (aiohttp.ClientError, TimeoutError, TransientError) as e:  # TransientError: hash mismatch
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2**attempt)
                logger.warning(
                    "Download failed, retrying",
                    extra={"url": url, "attempt": attempt + 1, "delay": delay, "error": str(e)},
                )
                await asyncio.sleep(delay)

    raise TransientError(
        f"Failed to download {url} after {MAX_RETRIES} attempts",
        code=ErrorCode.ASSET_DOWNLOAD,
        context={"url": url, "last_error": str(last_error)},
        hint=None,
        instance_id=None,
    )


async def _download_file(url: str, dest: Path, *, expected_hash: str) -> None:
    """
    Stream download with incremental hash verification.

    Uses chunked reads to bound memory usage (1 MiB per download).
    Hash is computed incrementally during download (no second pass).
    Uses atomic write pattern (temp file -> rename): the temp file goes on every path that
    does not rename it, a cancel too.
    """
    import aiohttp  # noqa: PLC0415 - only a download needs it (~50 ms at `import exec_sandbox`)

    algorithm, expected_digest = parse_hash_spec(expected_hash) if expected_hash else ("sha256", "")
    hasher = hashlib.new(algorithm)
    temp_path = dest.with_suffix(dest.suffix + ".tmp")

    logger.info("Downloading asset", extra={"url": url, "dest": str(dest)})

    try:
        timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=_DOWNLOAD_CONNECT_TIMEOUT_SECONDS, sock_read=_DOWNLOAD_READ_TIMEOUT_SECONDS
        )
        async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url) as resp:
            resp.raise_for_status()

            total_size = resp.content_length
            downloaded = 0

            temp_path.unlink(missing_ok=True)  # left by a killed process: create_file needs the name free
            f = create_file(temp_path, mode=_ASSET_MODE, dir_fd=None)
            try:
                async for chunk in resp.content.iter_chunked(_WRITE_CHUNK):
                    # Hash and write as we go, in one worker-thread job per chunk: a write
                    # into the page cache can wait for writeback. A cancel waits for the
                    # job, so the file never closes under a write.
                    await run_to_end(functools.partial(_hash_and_write, chunk, update=hasher.update, sink=f))
                    downloaded += len(chunk)

                    if total_size:
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

        logger.info("Download complete", extra={"file": str(dest)})

    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _hash_and_write(chunk: bytes, *, update: Callable[[bytes], object], sink: BinaryIO) -> None:
    update(chunk)
    sink.write(chunk)


class AsyncPooch:
    """
    Async file registry manager (like pooch.Pooch but async).

    Example:
        ASSETS = AsyncPooch(
            path=images_download_dir(),
            base_url="https://github.com/dualeai/exec-sandbox/releases/download/v{version}",
            version=__version__,
            setting="EXEC_SANDBOX_CACHE_DIR",
        )
        await ASSETS.load_registry_from_github("dualeai", "exec-sandbox", f"v{__version__}")

        kernel_path = await ASSETS.fetch("vmlinuz-x86_64.zst", decompress=True)
    """

    def __init__(self, path: Path, base_url: str, version: str, *, setting: str) -> None:
        """
        Initialize AsyncPooch registry: empty until load_registry_from_github fills it.

        Args:
            path: Local cache directory
            base_url: Base URL template with {version} placeholder
            version: Current version string (used as fallback if GitHub API not called)
            setting: The setting that chose `path`, for its errors
        """
        self.path = path
        self.setting = setting
        self.base_url = base_url
        self.version = version
        self.registry: dict[str, str] = {}  # file name -> hash ("sha256:abc123...")
        self._resolved_version: str | None = None  # Actual tag from GitHub API

    async def fetch(
        self,
        fname: str,
        *,
        decompress: bool,
    ) -> Path:
        """
        Fetch a file from the registry, downloading if needed.

        Args:
            fname: Filename to fetch (must be in registry)
            decompress: Unpack the download (a .zst) with decompress_zstd

        Returns:
            Path to the local file

        Raises:
            ConfigError: ASSET_NOT_FOUND: file not in registry; SETTING_INVALID: the cache
                folder is not a private folder of this user, or this user cannot make it or
                write its files (retrieve)
            TransientError: ASSET_DOWNLOAD: download or hash verification failed after
                retries; LOCAL_DISK: a host error on the cache folder (disk full, I/O error)
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
            setting=self.setting,
            decompress=decompress,
        )

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
            api_url = f"{_GITHUB_API}/repos/{owner}/{repo}/releases/latest"
        else:
            api_url = f"{_GITHUB_API}/repos/{owner}/{repo}/releases/tags/{tag}"

        logger.debug("Fetching release metadata from GitHub", extra={"url": api_url})

        headers = {"Accept": "application/vnd.github+json"}
        # Add auth token if available (increases rate limit)
        if token := os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        try:
            timeout = aiohttp.ClientTimeout(total=_METADATA_TIMEOUT_SECONDS)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(api_url, headers=headers) as resp,
            ):
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
# Decompression of a download (retrieve with decompress=True)
# ============================================================================


async def decompress_zstd(fname: Path) -> Path:
    """
    Decompress a .zst file, streaming.

    Uses Python's native zstd module (3.14+) or backports.zstd for older versions.
    Reads 64 KiB of compressed data at a time (CHUNK_SIZE): memory use depends on
    that chunk and its compression ratio, not on the file size. Uses atomic write
    pattern (temp file → rename) so partial results are never visible.

    Concurrency safety is provided by the caller (``retrieve``'s cross-process
    file lock).  This function must not be called without external serialization:
    a cancel stops the worker-thread job at its next chunk and returns once the job
    ended (its temp file removed, the .zst kept), so no write outlives the lock.

    Args:
        fname: Path to .zst compressed file

    Returns:
        Path to decompressed file (original .zst is deleted)
    """
    # Use native zstd module (Python 3.14+) or backports.zstd
    if sys.version_info >= (3, 14):
        from compression import zstd  # noqa: PLC0415
    else:
        from backports import zstd  # noqa: PLC0415

    dest = fname.with_suffix("")  # Remove .zst

    logger.debug("Decompressing", extra={"src": str(fname), "dest": str(dest)})

    # Use atomic write pattern: write to temp file, then rename
    temp_dest = dest.with_suffix(".tmp")

    def _decompress_sync(stop: threading.Event) -> None:
        try:
            temp_dest.unlink(missing_ok=True)  # left by a killed process: create_file needs the name free
            with fname.open("rb") as src, create_file(temp_dest, mode=_ASSET_MODE, dir_fd=None) as dst:
                # Use streaming decompression with incremental decompressor
                decompressor = zstd.ZstdDecompressor()
                while not stop.is_set():
                    chunk = src.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    decompressed = decompressor.decompress(chunk)
                    if decompressed:
                        dst.write(decompressed)
            if stop.is_set():  # the call was cancelled: the next fetch unpacks the .zst again
                temp_dest.unlink()
                return
            # Atomic rename (POSIX guarantees atomicity)
            temp_dest.rename(dest)
            # Remove compressed file only after successful rename
            fname.unlink()
        except Exception:
            # Clean up temp file on failure
            temp_dest.unlink(missing_ok=True)
            raise

    await run_stoppable(_decompress_sync, undo=None)

    logger.debug("Decompression complete", extra={"file": str(dest)})
    return dest


_GO_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}
"""get_arch_name's names to the Go names of the gvproxy-wrapper binaries."""


def get_gvproxy_suffix(arch: str) -> str:
    """
    Get gvproxy binary suffix of this OS for an architecture.

    Args:
        arch: "x86_64" or "aarch64" (get_arch_name's names).

    Returns:
        Suffix like "darwin-arm64", "darwin-amd64", "linux-arm64", "linux-amd64"
    """
    return f"{get_os_name()}-{_GO_ARCH[arch]}"

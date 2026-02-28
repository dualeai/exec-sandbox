"""qcow2 snapshot management for disk caching.

Implements two-tier snapshot architecture:
- L2 Cache: Local qcow2 disk snapshots (cold boot with cached packages)
- L3 Cache: S3 with zstd compression (cross-host sharing)

qcow2 optimizations:
- lazy_refcounts=on: Postpone metadata updates
- extended_l2=on: Faster CoW with subclusters
- cluster_size=128k: Balance between metadata and allocation

Snapshot structure:
- {cache_key}.qcow2: Standalone ext4 image (overlaid on EROFS base at boot)
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

import aiofiles
import aiofiles.os

# Use native zstd module (Python 3.14+) or backports.zstd
if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

import logging

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from exec_sandbox import __version__, constants
from exec_sandbox._imports import require_aioboto3
from exec_sandbox._logging import get_logger
from exec_sandbox.base_cache_manager import BaseCacheManager
from exec_sandbox.exceptions import (
    GuestAgentError,
    PackageInstallPermanentError,
    PackageInstallTransientError,
    PackageNotAllowedError,
    SnapshotError,
    VmQemuCrashError,
    VmTransientError,
)
from exec_sandbox.guest_agent_protocol import (
    InstallPackagesRequest,
    StreamingErrorMessage,
)
from exec_sandbox.guest_channel import consume_stream
from exec_sandbox.hash_utils import crc64
from exec_sandbox.models import Language
from exec_sandbox.overlay_pool import QemuImgError
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.qemu_vm import guest_error_to_exception
from exec_sandbox.settings import Settings  # noqa: TC001 - Used at runtime

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.vm_manager import VmManager

logger = get_logger(__name__)

_PERMANENT_PACKAGE_PATTERNS: Final[tuple[str, ...]] = (
    "no matching distribution",
    "could not find a version",
    "404 not found",
    "404 client error",
    "403 forbidden",
    "eresolve",
    "could not resolve dependency",
    "invalid version",
    "not found in registry",
    "no matching version",
    "version not found",
)
"""Patterns that indicate a permanent package install failure (won't succeed on retry)."""

_TRANSIENT_NETWORK_PATTERNS: Final[tuple[str, ...]] = (
    "client error",
    "connection refused",
    "connection reset",
    "connection timed out",
    "etimedout",
    "name resolution",
    "network unreachable",
    "server error",
    "ssl",
    "temporary failure in name resolution",
    "timed out",
    "reset by peer",
    "request failed after",
)
"""Patterns that indicate a transient network error (may succeed on retry)."""


def _classify_install_error(
    error_output: str,
) -> type[PackageInstallPermanentError | PackageInstallTransientError] | None:
    """Two-phase error classification for package install failures.

    Phase 1: Check permanent patterns first (no matching distribution, 404).
    Phase 2: Check transient patterns (connection reset, timed out).
    Returns None if neither matches (falls through to generic GuestAgentError).
    """
    lower = error_output.lower()
    # Phase 1: Permanent errors take priority — stop retrying immediately
    if any(pattern in lower for pattern in _PERMANENT_PACKAGE_PATTERNS):
        return PackageInstallPermanentError
    # Phase 2: Transient errors — retry with backoff
    if any(pattern in lower for pattern in _TRANSIENT_NETWORK_PATTERNS):
        return PackageInstallTransientError
    return None


class DiskSnapshotManager(BaseCacheManager):
    """Manages qcow2 snapshot cache for disk caching.

    Architecture (2-tier):
    - L2 cache: Local qcow2 disk snapshots (cold boot with cached packages)
    - L3 cache: S3 with zstd compression (cross-host sharing)

    Cache key format:
    - "{language}-v{major.minor}-base" for base images (no packages)
    - "{language}-v{major.minor}-{16char_hash}" for packages

    Simplifications:
    - ❌ No Redis (never implemented)
    - ❌ No metadata tracking (parse from cache_key)
    - ❌ No proactive eviction (lazy on disk full)
    - ✅ Pure filesystem (atime tracking only)
    - ✅ Single qcow2 file per snapshot
    """

    _cache_ext: ClassVar[str] = ".qcow2"

    def __init__(self, settings: Settings, vm_manager: VmManager):
        """Initialize qcow2 snapshot manager.

        Args:
            settings: Application settings with cache configuration
            vm_manager: VmManager for VM operations
        """
        super().__init__(settings.disk_snapshot_cache_dir)
        self.settings = settings
        self.vm_manager = vm_manager

        # L3 client (lazy init)
        self._s3_session = None

        # Concurrency control: Limit concurrent snapshot creation to prevent resource exhaustion
        # Max 1 concurrent snapshot creation (heavy operations: VM boot + package install)
        self._creation_semaphore = asyncio.Semaphore(1)

        # Per-cache-key locks to prevent race conditions during snapshot creation
        # When creating a snapshot, other VMs wanting the same snapshot wait rather than
        # trying to use a partially-created file
        self._creation_locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Protects _creation_locks dict

        # Limit concurrent S3 uploads to prevent network saturation and memory exhaustion
        # S3 PutObject is atomic - aborted uploads leave no partial blobs
        self._upload_semaphore = asyncio.Semaphore(settings.max_concurrent_s3_uploads)

    async def check_cache(
        self,
        language: Language,
        packages: list[str],
    ) -> Path | None:
        """Check L2 cache without creating snapshot.

        Use this for warm pool: creating L2 cache for base images (no packages)
        is pointless - would boot VM just to shut it down. Check-only instead.
        Returns cached snapshot if available, None if cache miss.

        Args:
            language: Programming language
            packages: Package list (empty for base image)

        Returns:
            Path to cached qcow2 snapshot, or None if cache miss.
        """
        cache_key = self._compute_cache_key(language, packages)
        return await self._check_l2_cache(cache_key)

    async def get_or_create_snapshot(
        self,
        language: Language,
        packages: list[str],
        tenant_id: str,
        task_id: str,
        memory_mb: int,
    ) -> Path:
        """Get cached snapshot or create new one.

        Cache hierarchy:
        1. Check L2 (local qcow2) → cold boot with cached disk
        2. Check L3 (S3 download) → download + cold boot
        3. Create new snapshot → package install + upload L3

        Args:
            language: Programming language
            packages: Package list with versions (e.g., ["pandas==2.1.0"])
            tenant_id: Tenant identifier
            task_id: Task identifier
            memory_mb: VM memory in MB (used for snapshot creation, not cache key)

        Returns:
            Path to snapshot qcow2 file.

        Raises:
            SnapshotError: Snapshot creation failed
        """
        cache_key = self._compute_cache_key(language, packages)

        # Fast path: Check L2 cache without lock (read-only, safe for concurrent access)
        snapshot_path = await self._check_l2_cache(cache_key)
        if snapshot_path:
            logger.debug("L2 cache hit", extra={"cache_key": cache_key})
            return snapshot_path

        # Slow path: Need to create or wait for creation
        # Use per-cache-key lock to prevent races during snapshot creation
        async with self._locks_lock:
            if cache_key not in self._creation_locks:
                self._creation_locks[cache_key] = asyncio.Lock()
            lock = self._creation_locks[cache_key]

        async with lock:
            # Re-check L2 cache under lock (another request may have created it)
            snapshot_path = await self._check_l2_cache(cache_key)
            if snapshot_path:
                logger.debug("L2 cache hit (after lock)", extra={"cache_key": cache_key})
                return snapshot_path

            # L3 cache check (S3) - only for images with packages
            # Base images are already distributed via asset downloads, no need for S3
            if packages:
                try:
                    snapshot_path = await self._download_from_s3(cache_key)
                    logger.debug("L3 cache hit", extra={"cache_key": cache_key})
                    return snapshot_path
                except SnapshotError:
                    pass  # Cache miss, create new

            # Cache miss: Create new snapshot
            logger.debug("Cache miss, creating snapshot", extra={"cache_key": cache_key})
            snapshot_path = await self._create_snapshot(language, packages, cache_key, tenant_id, task_id, memory_mb)

            # Upload to S3 (async, fire-and-forget) - only for images with packages
            # Base images don't need S3 - they're already globally distributed
            if packages:
                upload_task: asyncio.Task[None] = asyncio.create_task(self._upload_to_s3(cache_key, snapshot_path))
                self._track_task(upload_task, name=f"s3-upload-{cache_key}")

            return snapshot_path

    def _compute_cache_key(
        self,
        language: Language,
        packages: list[str],
    ) -> str:
        """Compute L2 cache key for snapshot.

        Includes:
        - Library major.minor version (invalidates on lib upgrade)
        - Base image hash (invalidates when images are rebuilt)
        - Package hash (different packages = different cache entry)

        memory_mb is NOT in the cache key because disk-only snapshots work
        with any memory allocation.

        Note: allow_network is NOT in the cache key because:
        - Snapshots are always created with network (for pip/npm install)
        - User's allow_network setting only controls gvproxy at execution time

        Format:
        - "{language}-v{major.minor}-{img_hash}-base" for base (no packages)
        - "{language}-v{major.minor}-{img_hash}-{16char_pkg_hash}" for packages

        Args:
            language: Programming language
            packages: Sorted package list with versions

        Returns:
            Cache key string
        """
        # Extract major.minor from __version__ (e.g., "0.1.0" -> "0.1")
        version_parts = __version__.split(".")
        version = f"{version_parts[0]}.{version_parts[1]}"

        # Include base image hash (first 8 chars) to invalidate cache on image rebuild
        base_image = self.vm_manager.get_base_image(language)
        img_hash = self.image_hash(base_image)

        base = f"{language.value}-v{version}-{img_hash}"

        if not packages:
            return f"{base}-base"
        packages_str = "".join(sorted(packages))
        packages_hash = crc64(packages_str)
        return f"{base}-{packages_hash}"

    async def _check_l2_cache(self, cache_key: str) -> Path | None:
        """Check L2 local cache for qcow2 snapshot.

        Validates:
        1. Snapshot file exists
        2. Valid qcow2 format (via qemu-img check)

        Args:
            cache_key: Snapshot cache key.

        Returns:
            Path to qcow2 snapshot if valid cache hit, None otherwise.
        """
        snapshot_path = self.cache_dir / f"{cache_key}.qcow2"

        if not await aiofiles.os.path.exists(snapshot_path):
            return None

        # Verify qcow2 format (standalone ext4 snapshot, no backing file expected)
        try:
            proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    "qemu-img",
                    "check",
                    str(snapshot_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            )
            await proc.communicate()

            if proc.returncode != 0:
                logger.debug("Invalid qcow2 snapshot, removing", extra={"cache_key": cache_key})
                await aiofiles.os.remove(snapshot_path)
                return None

        except (OSError, FileNotFoundError):
            return None

        # Update atime for LRU tracking
        snapshot_path.touch(exist_ok=True)

        return snapshot_path

    async def _create_snapshot(  # noqa: PLR0912, PLR0915
        self,
        language: Language,
        packages: list[str],
        cache_key: str,
        tenant_id: str,
        task_id: str,
        memory_mb: int,
        *,
        _enospc_retried: bool = False,
    ) -> Path:
        """Create new qcow2 snapshot with packages installed.

        Uses asyncio.wait racing for instant crash detection.

        Workflow:
        1. Create qcow2 with backing file (base image)
        2. Boot VM with snapshot image (ephemeral_vm handles lifecycle)
        3. Install packages via guest agent (with death monitoring)
        4. Shutdown VM (writes committed to snapshot)
        5. Return snapshot path

        Args:
            language: Programming language
            packages: Package list with versions
            cache_key: Snapshot cache key
            tenant_id: Tenant identifier
            task_id: Task identifier
            memory_mb: VM memory in MB

        Returns:
            Path to created qcow2 snapshot

        Raises:
            SnapshotError: Creation failed
            VmQemuCrashError: VM crashed during snapshot creation
        """
        start_time = asyncio.get_running_loop().time()
        snapshot_path = self.cache_dir / f"{cache_key}.qcow2"
        success = False

        # Acquire semaphore to limit concurrent snapshot creation
        async with self._creation_semaphore:
            try:
                # Step 1: Create standalone ext4 qcow2 for overlay (vdb)
                await self._create_snapshot_image(snapshot_path, cache_key, language, packages, tenant_id)

                # Step 2: Determine network configuration for snapshot creation
                # ALWAYS enable network during snapshot creation (pip/npm needs it)
                # Restrict to package registries only for security
                if packages:
                    if language == "python":
                        package_domains = list(constants.PYTHON_PACKAGE_DOMAINS)
                    elif language == "javascript":
                        package_domains = list(constants.NPM_PACKAGE_DOMAINS)
                    else:
                        package_domains = []
                else:
                    package_domains = None

                # Step 3-5: Boot VM, install packages, shutdown — ephemeral_vm handles lifecycle
                async with self.vm_manager.ephemeral_vm(
                    language,
                    tenant_id,
                    task_id,
                    memory_mb=memory_mb,
                    allow_network=True,
                    allowed_domains=package_domains,
                    direct_write_target=snapshot_path,
                    retry_profile=constants.RETRY_BACKGROUND,
                ) as vm:
                    # Install packages with death monitoring (asyncio.wait)
                    death_task = asyncio.create_task(self._monitor_vm_death(vm, cache_key))
                    install_task = asyncio.create_task(self._install_packages(vm, Language(language), packages))

                    try:
                        done, pending = await asyncio.wait(
                            {death_task, install_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        for task in pending:
                            task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task

                        completed_task = done.pop()
                        if completed_task == death_task:
                            await completed_task  # Propagate VM death exception
                        else:
                            await completed_task  # Propagate install exception if any

                    except Exception:
                        for task in [death_task, install_task]:
                            if not task.done():
                                task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await task
                        raise

                    # Shutdown QEMU process cleanly
                    we_initiated_shutdown = False
                    if vm.process.returncode is None:
                        we_initiated_shutdown = True
                        await vm.process.terminate()
                        try:
                            await asyncio.wait_for(vm.process.wait(), timeout=5.0)
                        except TimeoutError:
                            await vm.process.kill()
                            await vm.process.wait()
                    else:
                        await vm.process.wait()

                    # Verify QEMU exited cleanly
                    if we_initiated_shutdown:
                        if vm.process.returncode not in {0, -9, -15}:
                            raise VmQemuCrashError(
                                f"QEMU exited unexpectedly after terminate (exit code {vm.process.returncode})",
                                context={
                                    "cache_key": cache_key,
                                    "exit_code": vm.process.returncode,
                                    "language": language,
                                    "packages": packages,
                                },
                            )
                    elif vm.process.returncode != 0:
                        raise VmQemuCrashError(
                            f"QEMU died unexpectedly during snapshot creation (exit code {vm.process.returncode})",
                            context={
                                "cache_key": cache_key,
                                "exit_code": vm.process.returncode,
                                "language": language,
                                "packages": packages,
                            },
                        )
                    else:
                        logger.info(
                            "QEMU exited cleanly (code 0) after install completed",
                            extra={"cache_key": cache_key, "language": language},
                        )

                    success = True
                # ephemeral_vm's finally calls destroy_vm (idempotent)

            # Handle disk full (lazy eviction)
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    if _enospc_retried:
                        raise SnapshotError(
                            f"Disk full after eviction retry for snapshot {cache_key}",
                            context={"cache_key": cache_key, "language": language, "packages": packages},
                        ) from e
                    await self._evict_oldest_snapshot()
                    return await self._create_snapshot(
                        language, packages, cache_key, tenant_id, task_id, memory_mb, _enospc_retried=True
                    )
                raise

            except (SnapshotError, PackageNotAllowedError, PackageInstallPermanentError, asyncio.CancelledError):
                raise

            except VmTransientError as e:
                raise SnapshotError(
                    f"VM crashed during snapshot creation: {e}",
                    context={
                        "cache_key": cache_key,
                        "language": language,
                        "packages": packages,
                        "tenant_id": tenant_id,
                    },
                ) from e

            except Exception as e:
                raise SnapshotError(
                    f"Failed to create snapshot: {e}",
                    context={
                        "cache_key": cache_key,
                        "language": language,
                        "packages": packages,
                        "tenant_id": tenant_id,
                    },
                ) from e

            finally:
                # Cleanup snapshot file on failure
                if not success and snapshot_path.exists():
                    try:
                        snapshot_path.unlink()
                        logger.debug("Snapshot file cleaned up on failure", extra={"cache_key": cache_key})
                    except OSError as e:
                        logger.warning(
                            "Failed to cleanup snapshot file",
                            extra={"cache_key": cache_key, "error": str(e)},
                        )

        # Record snapshot creation duration
        duration_ms = round((asyncio.get_running_loop().time() - start_time) * 1000)
        logger.info(
            "Snapshot created",
            extra={
                "cache_key": cache_key,
                "language": language,
                "package_count": len(packages),
                "duration_ms": duration_ms,
            },
        )

        return snapshot_path

    async def _create_snapshot_image(
        self,
        snapshot_path: Path,
        cache_key: str,
        language: str,
        packages: list[str],
        tenant_id: str,
    ) -> None:
        """Create standalone ext4 qcow2 for snapshot overlay (vdb).

        With EROFS rootfs, snapshots are standalone ext4 qcow2 images that serve
        as the upper layer in overlayfs (merged with the read-only EROFS base on vda).
        tiny-init formats and mounts this as ext4 writable during snapshot creation.

        Args:
            snapshot_path: Path to snapshot to create
            cache_key: Snapshot cache key
            language: Programming language
            packages: Package list
            tenant_id: Tenant identifier

        Raises:
            SnapshotError: qemu-img command failed
        """
        try:
            proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    "qemu-img",
                    "create",
                    "-f",
                    "qcow2",
                    "-o",
                    "cluster_size=128k,lazy_refcounts=on,extended_l2=on",
                    str(snapshot_path),
                    "512M",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            )
            _stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise QemuImgError(f"qemu-img create failed: {stderr.decode().strip()}")
        except QemuImgError as e:
            raise SnapshotError(
                str(e),
                context={
                    "cache_key": cache_key,
                    "language": language,
                    "packages": packages,
                    "tenant_id": tenant_id,
                },
            ) from e

    async def _monitor_vm_death(self, vm: QemuVM, cache_key: str) -> None:
        """Monitor VM process for unexpected death during snapshot creation."""
        returncode = await vm.process.wait()
        diag = await vm.collect_diagnostics()
        logger.error(
            "VM died during snapshot creation",
            extra={**asdict(diag), "cache_key": cache_key},
        )
        raise VmQemuCrashError(
            f"VM process died during snapshot creation (exit code {returncode})",
            diagnostics=diag,
        )

    async def _install_packages(
        self,
        vm: QemuVM,
        language: Language,
        packages: list[str],
    ) -> None:
        """Install packages in VM via guest agent.

        Retries transient network errors (DNS, connection refused, timeouts)
        using exponential backoff. The VM stays running between retries — only
        the install request is re-sent.

        Args:
            vm: QemuVM handle
            language: Programming language
            packages: Package list with versions

        Raises:
            SnapshotError: Package installation failed
            GuestAgentError: Guest agent returned error (permanent)
            PackageInstallTransientError: Transient network error (after all retries exhausted)
            PackageInstallPermanentError: Permanent package error (no retry)
        """
        if not packages:
            return

        request = InstallPackagesRequest(
            language=language,
            packages=packages,
            timeout=constants.PACKAGE_INSTALL_TIMEOUT_SECONDS,
        )

        hard_timeout = constants.PACKAGE_INSTALL_TIMEOUT_SECONDS + constants.EXECUTION_TIMEOUT_MARGIN_SECONDS
        total_timeout = (
            hard_timeout * constants.PACKAGE_INSTALL_MAX_RETRIES
            + constants.PACKAGE_INSTALL_RETRY_MAX_SECONDS * (constants.PACKAGE_INSTALL_MAX_RETRIES - 1)
        )

        async def _handle_install_error(msg: StreamingErrorMessage) -> None:
            logger.error(
                "Guest agent install error",
                extra={"vm_id": vm.vm_id, "error": msg.message, "error_type": msg.error_type},
            )
            exc = guest_error_to_exception(msg, vm.vm_id, operation="install_packages")
            # Guest timeouts during install are transient — wrap so tenacity retries
            if isinstance(exc, VmTransientError):
                raise PackageInstallTransientError(str(exc), context={"vm_id": vm.vm_id}) from exc
            raise exc

        try:
            async with asyncio.timeout(total_timeout):
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(constants.PACKAGE_INSTALL_MAX_RETRIES),
                    wait=wait_random_exponential(
                        min=constants.PACKAGE_INSTALL_RETRY_MIN_SECONDS,
                        max=constants.PACKAGE_INSTALL_RETRY_MAX_SECONDS,
                    ),
                    retry=retry_if_exception_type(PackageInstallTransientError),
                    before_sleep=before_sleep_log(logger, logging.WARNING),
                    reraise=True,
                ):
                    with attempt:
                        await vm.channel.connect(timeout_seconds=constants.GUEST_CONNECT_TIMEOUT_SECONDS)

                        result = await consume_stream(
                            vm.channel,
                            request,
                            timeout=hard_timeout,
                            vm_id=vm.vm_id,
                            on_error=_handle_install_error,
                        )

                        if result.exit_code != 0:
                            error_output = result.stderr or "Unknown error"
                            error_class = _classify_install_error(error_output)
                            if error_class is PackageInstallPermanentError:
                                raise PackageInstallPermanentError(
                                    f"Package installation failed permanently (exit code {result.exit_code}): {error_output[:500]}",
                                    context={"exit_code": result.exit_code, "stderr": error_output[:500]},
                                )
                            if error_class is PackageInstallTransientError:
                                raise PackageInstallTransientError(
                                    f"Package installation failed with transient network error (exit code {result.exit_code}): {error_output[:500]}",
                                    context={"exit_code": result.exit_code, "stderr": error_output[:500]},
                                )
                            raise GuestAgentError(
                                f"Package installation failed with exit code {result.exit_code}: {error_output[:500]}",
                                response={"exit_code": result.exit_code, "stderr": error_output[:500]},
                            )

        except TimeoutError as e:
            raise SnapshotError(
                f"Package installation timeout after {total_timeout}s",
                context={"vm_id": vm.vm_id, "language": language, "packages": packages},
            ) from e

        except (GuestAgentError, PackageInstallTransientError, PackageInstallPermanentError, PackageNotAllowedError):
            raise

        except Exception as e:
            raise SnapshotError(
                f"Package installation failed (communication error): {e}",
                context={"vm_id": vm.vm_id, "language": language, "packages": packages},
            ) from e

    async def _download_from_s3(self, cache_key: str) -> Path:
        """Download and decompress snapshot from S3 to L2 cache.

        Args:
            cache_key: Snapshot cache key

        Returns:
            Path to downloaded qcow2 snapshot

        Raises:
            SnapshotError: Download failed
        """
        snapshot_path = self.cache_dir / f"{cache_key}.qcow2"
        compressed_path = self.cache_dir / f"{cache_key}.qcow2.zst"

        try:
            async with await self._get_s3_client() as s3:  # type: ignore[union-attr]
                # Download compressed qcow2
                s3_key = f"snapshots/{cache_key}.qcow2.zst"
                await s3.download_file(  # type: ignore[union-attr]
                    self.settings.s3_bucket,
                    s3_key,
                    str(compressed_path),
                )

            # Decompress with zstd (run in thread pool to avoid blocking)
            chunk_size = 64 * 1024  # 64KB chunks for streaming

            def _decompress() -> None:
                decompressor = zstd.ZstdDecompressor()
                with Path(compressed_path).open("rb") as src, Path(snapshot_path).open("wb") as dst:
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        decompressed = decompressor.decompress(chunk)
                        if decompressed:
                            dst.write(decompressed)

            await asyncio.to_thread(_decompress)

            # Cleanup compressed file
            await aiofiles.os.remove(compressed_path)

        except Exception as e:
            # Cleanup on failure
            if compressed_path.exists():
                await aiofiles.os.remove(compressed_path)
            if snapshot_path.exists():
                await aiofiles.os.remove(snapshot_path)

            raise SnapshotError(f"S3 download failed: {e}") from e

        return snapshot_path

    async def _upload_to_s3(self, cache_key: str, snapshot_path: Path) -> None:
        """Upload compressed snapshot to S3 (async, fire-and-forget).

        Bounded by upload_semaphore to prevent:
        - Network saturation
        - Memory exhaustion from compression buffers
        - S3 rate limiting (unlikely but possible)

        Args:
            cache_key: Snapshot cache key
            snapshot_path: Local qcow2 snapshot path
        """
        compressed_path = self.cache_dir / f"{cache_key}.qcow2.zst"

        # Acquire semaphore to limit concurrent uploads
        async with self._upload_semaphore:
            try:
                # Compress with zstd (level 3 for speed, run in thread pool to avoid blocking)
                chunk_size = 64 * 1024  # 64KB chunks for streaming

                def _compress() -> None:
                    compressor = zstd.ZstdCompressor(level=3)
                    with Path(snapshot_path).open("rb") as src, Path(compressed_path).open("wb") as dst:
                        while True:
                            chunk = src.read(chunk_size)
                            if not chunk:
                                break
                            compressed = compressor.compress(chunk)
                            if compressed:
                                dst.write(compressed)
                        # Flush remaining data
                        final = compressor.flush()
                        if final:
                            dst.write(final)

                await asyncio.to_thread(_compress)

                async with await self._get_s3_client() as s3:  # type: ignore[union-attr]
                    # Upload compressed qcow2
                    s3_key = f"snapshots/{cache_key}.qcow2.zst"
                    await s3.upload_file(  # type: ignore[union-attr]
                        str(compressed_path),
                        self.settings.s3_bucket,
                        s3_key,
                        ExtraArgs={
                            "Tagging": f"ttl_days={self.settings.snapshot_cache_ttl_days}",
                        },
                    )

                # Cleanup compressed file
                await aiofiles.os.remove(compressed_path)

            except (OSError, RuntimeError, ConnectionError, Exception) as e:  # noqa: BLE001 - Fire-and-forget S3 upload
                # Silent failure (L2 cache still works)
                # Catch all exceptions including botocore.exceptions.ClientError
                logger.warning("S3 upload failed silently", extra={"cache_key": cache_key, "error": str(e)})
                if compressed_path.exists():
                    await aiofiles.os.remove(compressed_path)

    async def _get_s3_client(self):  # type: ignore[no-untyped-def]
        """Get S3 client (lazy init).

        Raises:
            SnapshotError: If S3 backup not configured

        Returns:
            S3 client context manager from aioboto3 (untyped library)
        """
        if not self.settings.s3_bucket:
            raise SnapshotError("S3 backup disabled (s3_bucket not configured)")

        if self._s3_session is None:
            aioboto3 = require_aioboto3()
            self._s3_session = aioboto3.Session()

        return self._s3_session.client(  # type: ignore[no-any-return]
            "s3",
            region_name=self.settings.s3_region,
            endpoint_url=self.settings.s3_endpoint_url,
        )

    async def stop(self) -> None:
        """Stop DiskSnapshotManager and wait for background upload tasks to complete."""
        await super().stop()

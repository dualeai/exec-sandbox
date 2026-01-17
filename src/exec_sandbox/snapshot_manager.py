"""qcow2 snapshot management for disk caching.

Implements two-tier snapshot architecture:
- L2 Cache: Local qcow2 disk snapshots (cold boot with cached packages)
- L3 Cache: S3 with zstd compression (cross-host sharing)

qcow2 optimizations:
- lazy_refcounts=on: Postpone metadata updates
- extended_l2=on: Faster CoW with subclusters
- cluster_size=128k: Balance between metadata and allocation

Snapshot structure:
- {cache_key}.qcow2: Disk state (backing file + package changes)
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Self

import aiofiles
import aiofiles.os

# Use native zstd module (Python 3.14+) or backports.zstd
if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

from exec_sandbox import __version__, constants
from exec_sandbox._imports import require_aioboto3
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import GuestAgentError, SnapshotError, VmError
from exec_sandbox.guest_agent_protocol import (
    ExecutionCompleteMessage,
    InstallPackagesRequest,
    OutputChunkMessage,
    StreamingErrorMessage,
)
from exec_sandbox.hash_utils import crc32, crc64
from exec_sandbox.models import Language
from exec_sandbox.overlay_pool import QemuImgError, create_qcow2_overlay
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.settings import Settings  # noqa: TC001 - Used at runtime

if TYPE_CHECKING:
    from exec_sandbox.vm_manager import QemuVM, VmManager

logger = get_logger(__name__)


class SnapshotManager:
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

    def __init__(self, settings: Settings, vm_manager: VmManager):
        """Initialize qcow2 snapshot manager.

        Args:
            settings: Application settings with cache configuration
            vm_manager: VmManager for VM operations
        """
        self.settings = settings
        self.vm_manager = vm_manager
        self.cache_dir = settings.snapshot_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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

        # Track background S3 upload tasks to prevent GC
        self._background_tasks: set[asyncio.Task[None]] = set()

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
                self._background_tasks.add(upload_task)
                upload_task.add_done_callback(lambda t: self._background_tasks.discard(t))
                upload_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

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
        img_hash = self._get_base_image_hash(base_image)

        base = f"{language.value}-v{version}-{img_hash}"

        if not packages:
            return f"{base}-base"
        packages_str = "".join(sorted(packages))
        packages_hash = crc64(packages_str)
        return f"{base}-{packages_hash}"

    def _get_base_image_hash(self, base_image: Path) -> str:
        """Get hash of base image for cache key.

        Uses file modification time + size for fast hashing (avoids reading entire file).
        This detects image rebuilds while being O(1) instead of O(n).

        Args:
            base_image: Path to base qcow2 image

        Returns:
            8-character hash string (CRC32 in hex)
        """
        try:
            stat = base_image.stat()
            # Combine mtime (nanoseconds) + size for unique fingerprint
            fingerprint = f"{stat.st_mtime_ns}:{stat.st_size}"
            return crc32(fingerprint)
        except OSError:
            # If image doesn't exist, return placeholder (will fail later anyway)
            return "missing0"

    async def _check_l2_cache(self, cache_key: str) -> Path | None:
        """Check L2 local cache for qcow2 snapshot.

        Validates:
        1. Snapshot file exists
        2. Valid qcow2 format
        3. Backing file exists and matches expected base image

        Args:
            cache_key: Snapshot cache key.

        Returns:
            Path to qcow2 snapshot if valid cache hit, None otherwise.
        """
        snapshot_path = self.cache_dir / f"{cache_key}.qcow2"

        if not await aiofiles.os.path.exists(snapshot_path):
            return None

        # Verify qcow2 format and get backing file info
        try:
            proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    "qemu-img",
                    "info",
                    "--output=json",
                    str(snapshot_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            )
            stdout, _stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.debug("Invalid qcow2 snapshot, removing", extra={"cache_key": cache_key})
                await aiofiles.os.remove(snapshot_path)
                return None

            # Parse JSON output to check backing file
            info = json.loads(stdout.decode())
            backing_file = info.get("backing-filename") or info.get("full-backing-filename")

            if backing_file:
                # Verify backing file exists
                if not await aiofiles.os.path.exists(backing_file):
                    logger.warning(
                        "Snapshot backing file missing, removing stale cache",
                        extra={"cache_key": cache_key, "backing_file": backing_file},
                    )
                    await aiofiles.os.remove(snapshot_path)
                    return None

                # Verify backing file matches expected base image by checking hash
                # Extract language from cache_key (format: "{language}-v{version}-{hash|base}")
                language_str = cache_key.split("-")[0]
                try:
                    expected_base = self.vm_manager.get_base_image(Language(language_str)).resolve()
                    if Path(backing_file).resolve() != expected_base:
                        logger.warning(
                            "Snapshot backing file mismatch, removing stale cache",
                            extra={
                                "cache_key": cache_key,
                                "backing_file": backing_file,
                                "expected": str(expected_base),
                            },
                        )
                        await aiofiles.os.remove(snapshot_path)
                        return None
                except (ValueError, KeyError):
                    pass  # Can't determine expected base, skip validation

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
    ) -> Path:
        """Create new qcow2 snapshot with packages installed.

        Uses asyncio.wait racing for instant crash detection.

        Workflow:
        1. Create qcow2 with backing file (base image)
        2. Boot VM with snapshot image
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
            VmError: VM crashed during snapshot creation
        """
        start_time = asyncio.get_event_loop().time()
        snapshot_path = self.cache_dir / f"{cache_key}.qcow2"
        # Resolve to absolute path - qemu-img resolves backing file relative to snapshot location,
        # so we need absolute path when snapshot_cache_dir differs from base_images_dir
        base_image = self.vm_manager.get_base_image(language).resolve()

        # Acquire semaphore to limit concurrent snapshot creation
        async with self._creation_semaphore:
            vm = None  # Track VM for cleanup

            try:
                # Step 1: Create qcow2 with backing file
                await self._create_snapshot_image(snapshot_path, base_image, cache_key, language, packages, tenant_id)

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

                # Step 3: Create VM with config that matches cache key
                # NOTE: memory_mb should match for consistency
                # NOTE: Snapshot always created with allow_network=True (for pip/npm)
                vm = await self.vm_manager.create_vm(
                    language,
                    tenant_id,
                    task_id,
                    memory_mb=memory_mb,
                    allow_network=True,  # Always need network for package install
                    allowed_domains=package_domains,
                    direct_write_target=snapshot_path,  # Write directly to snapshot (no overlay)
                )

                # Step 4: Install packages with death monitoring (asyncio.wait)
                # Race: Install vs VM death - if VM crashes, instant detection
                # Use FIRST_COMPLETED to exit immediately when either task finishes
                death_task = asyncio.create_task(self._monitor_vm_death(vm, cache_key))
                install_task = asyncio.create_task(self._install_packages(vm, Language(language), packages))

                try:
                    done, pending = await asyncio.wait(
                        {death_task, install_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Cancel pending task
                    for task in pending:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task

                    # Check which task completed
                    completed_task = done.pop()
                    if completed_task == death_task:
                        # VM died during installation - re-raise VmError
                        await completed_task  # Propagate exception
                    else:
                        # Install succeeded - check for errors
                        await completed_task  # Propagate any exception from install

                except Exception:
                    # Cleanup: Cancel both tasks on any failure
                    for task in [death_task, install_task]:
                        if not task.done():
                            task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task
                    raise

                # Step 4: Shutdown QEMU process cleanly
                if vm.process.returncode is None:
                    await vm.process.terminate()
                    try:
                        await asyncio.wait_for(vm.process.wait(), timeout=5.0)
                    except TimeoutError:
                        await vm.process.kill()
                        await vm.process.wait()

                # Step 5: Clean up resources
                # With direct_write_target, we wrote directly to snapshot_path
                # No commit needed - just destroy VM (which cleans up cgroup, sockets, etc.)
                await self.vm_manager.destroy_vm(vm)
                vm = None

            # Handle disk full (lazy eviction)
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    # Evict oldest snapshot and retry once
                    # Cleanup handled by finally block
                    await self._evict_oldest_snapshot()
                    return await self._create_snapshot(language, packages, cache_key, tenant_id, task_id, memory_mb)
                raise

            # Handle VM death during snapshot creation
            except VmError as e:
                # Wrap VM error in SnapshotError
                # Cleanup handled by finally block
                raise SnapshotError(
                    f"VM crashed during snapshot creation: {e}",
                    context={
                        "cache_key": cache_key,
                        "language": language,
                        "packages": packages,
                        "tenant_id": tenant_id,
                    },
                ) from e

            except asyncio.CancelledError:
                logger.warning("Snapshot creation cancelled", extra={"cache_key": cache_key})
                raise  # Immediate propagation, cleanup in finally

            except Exception as e:
                # Wrap generic errors in SnapshotError
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
                # Cleanup always runs (success, error, or cancellation)
                # Step 1: Cleanup VM if still running
                if vm and vm.state != VmState.DESTROYED:
                    try:
                        await self.vm_manager.destroy_vm(vm)
                        logger.info("VM cleaned up in finally block", extra={"cache_key": cache_key})
                    except Exception as cleanup_error:
                        logger.error(
                            "VM cleanup failed in finally block",
                            extra={"cache_key": cache_key, "error": str(cleanup_error)},
                            exc_info=True,
                        )

                # Step 2: Cleanup snapshot file on failure
                # vm=None means success (VM shutdown completed), keep snapshot
                # vm!=None means failure, cleanup snapshot
                if vm is not None and snapshot_path.exists():
                    try:
                        snapshot_path.unlink()
                        logger.debug("Snapshot file cleaned up in finally block", extra={"cache_key": cache_key})
                    except OSError as e:
                        logger.warning(
                            "Failed to cleanup snapshot file",
                            extra={"cache_key": cache_key, "error": str(e)},
                        )

        # Record snapshot creation duration
        duration_ms = round((asyncio.get_event_loop().time() - start_time) * 1000)
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
        base_image: Path,
        cache_key: str,
        language: str,
        packages: list[str],
        tenant_id: str,
    ) -> None:
        """Create qcow2 snapshot image with backing file.

        Args:
            snapshot_path: Path to snapshot to create
            base_image: Base image to use as backing file
            cache_key: Snapshot cache key
            language: Programming language
            packages: Package list
            tenant_id: Tenant identifier

        Raises:
            SnapshotError: qemu-img command failed
        """
        try:
            await create_qcow2_overlay(base_image, snapshot_path)
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
        """Monitor VM process for unexpected death.

        Event-driven death detection: Waits on process exit (no polling).
        If process exits → raises VmError → TaskGroup cancels other tasks.

        Args:
            vm: QemuVM handle
            cache_key: Snapshot cache key

        Raises:
            VmError: VM process died unexpectedly
        """
        # Wait for QEMU process to exit (blocks until death)
        returncode = await vm.process.wait()

        # Process died → raise error to cancel sibling tasks
        raise VmError(
            f"VM process died during snapshot creation (exit code {returncode})",
            context={
                "cache_key": cache_key,
                "vm_id": vm.vm_id,
                "exit_code": returncode,
            },
        )

    async def _install_packages(
        self,
        vm: QemuVM,
        language: Language,
        packages: list[str],
    ) -> None:
        """Install packages in VM via guest agent.

        Event-driven architecture:
        - ZERO polling loops
        - Instant crash detection via asyncio.wait(FIRST_COMPLETED) in caller
        - Timeout via asyncio.timeout() context manager

        Args:
            vm: QemuVM handle
            language: Programming language
            packages: Package list with versions

        Raises:
            SnapshotError: Package installation failed
            GuestAgentError: Guest agent returned error
        """
        if not packages:
            return

        # Send install_packages command via TCP channel
        request = InstallPackagesRequest(
            language=language,
            packages=packages,
            timeout=constants.PACKAGE_INSTALL_TIMEOUT_SECONDS,  # Soft timeout (guest enforcement)
        )

        try:
            # Use asyncio.timeout() context manager (Python 3.14)
            async with asyncio.timeout(constants.PACKAGE_INSTALL_TIMEOUT_SECONDS):
                # Connect to guest agent (fixed init timeout)
                await vm.channel.connect(timeout_seconds=constants.GUEST_CONNECT_TIMEOUT_SECONDS)

                # Stream install output (now uses same streaming protocol as execute_code)
                # Hard timeout = soft timeout (guest) + margin (host watchdog)
                hard_timeout = constants.PACKAGE_INSTALL_TIMEOUT_SECONDS + constants.EXECUTION_TIMEOUT_MARGIN_SECONDS

                exit_code = -1
                stderr_chunks: list[str] = []

                async for msg in vm.channel.stream_messages(request, timeout=hard_timeout):
                    if isinstance(msg, OutputChunkMessage):
                        # Log install output for debugging
                        logger.info(
                            "Package install output",
                            extra={"vm_id": vm.vm_id, "stream": msg.type, "chunk": msg.chunk[:200]},
                        )
                        # Collect stderr for error reporting
                        if msg.type == "stderr":
                            stderr_chunks.append(msg.chunk)

                    elif isinstance(msg, ExecutionCompleteMessage):
                        exit_code = msg.exit_code
                        # Note: msg.execution_time_ms available but not needed for package install

                    elif isinstance(msg, StreamingErrorMessage):
                        logger.error(
                            "Guest agent install error",
                            extra={"vm_id": vm.vm_id, "error": msg.message, "error_type": msg.error_type},
                        )
                        raise GuestAgentError(
                            f"Package installation failed: {msg.message}",
                            response={"message": msg.message, "error_type": msg.error_type},
                        )

                # Check installation success
                if exit_code != 0:
                    error_output = "".join(stderr_chunks) if stderr_chunks else "Unknown error"
                    raise GuestAgentError(
                        f"Package installation failed with exit code {exit_code}: {error_output[:500]}",
                        response={"exit_code": exit_code, "stderr": error_output[:500]},
                    )

        except TimeoutError as e:
            # Timeout → package install took too long
            raise SnapshotError(
                f"Package installation timeout after {constants.PACKAGE_INSTALL_TIMEOUT_SECONDS}s",
                context={
                    "vm_id": vm.vm_id,
                    "language": language,
                    "packages": packages,
                },
            ) from e

        except GuestAgentError:
            raise  # Re-raise guest agent errors as-is

        except Exception as e:
            # Orchestrator/communication error (connection, protocol, etc)
            raise SnapshotError(
                f"Package installation failed (communication error): {e}",
                context={
                    "vm_id": vm.vm_id,
                    "language": language,
                    "packages": packages,
                },
            ) from e

    async def _evict_oldest_snapshot(self) -> None:
        """Evict single oldest snapshot (by atime).

        Called lazily when disk full (ENOSPC).
        Uses asyncio.to_thread for blocking glob and asyncio.gather for parallel stat.
        """
        # Run blocking glob in thread pool (non-blocking)
        snapshot_files = await asyncio.to_thread(lambda: list(self.cache_dir.glob("*.qcow2")))

        if not snapshot_files:
            return

        # Helper to get atime for a single file
        async def get_atime(path: Path) -> tuple[Path, float] | None:
            try:
                if await aiofiles.os.path.isfile(path):
                    stat = await aiofiles.os.stat(path)
                    return (path, stat.st_atime)
            except OSError:
                pass
            return None

        # Parallel stat calls for all files
        results = await asyncio.gather(*[get_atime(f) for f in snapshot_files])
        snapshots = [r for r in results if r is not None]

        if not snapshots:
            return

        # Find oldest (by atime)
        oldest_file, _ = min(snapshots, key=lambda x: x[1])

        # Delete oldest snapshot
        await aiofiles.os.remove(oldest_file)

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
        """Stop SnapshotManager and wait for background upload tasks to complete.

        Should be called when the SnapshotManager is no longer needed.
        Ensures all S3 uploads finish before stopping.
        """
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

    async def __aenter__(self) -> Self:
        """Enter async context manager.

        No async initialization needed - returns self immediately.
        """
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, stopping and waiting for background tasks."""
        await self.stop()


# Import VmState for type checking in finally block
from exec_sandbox.vm_manager import VmState  # noqa: E402

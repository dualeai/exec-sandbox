"""qcow2 snapshot management with CoW backing files.

Implements snapshot architecture:
- Base images: qcow2 format (Python 3.14, Node 23)
- Dynamic snapshots: qcow2 with backing file chain
- L1 Cache: Local disk with lazy LRU eviction
- L3 Cache: S3 with zstd compression (14-day TTL)

Performance targets:
- L1 hit: <400ms (qcow2 overlay creation + QEMU boot)
- L3 hit: <5s (download + decompress + overlay + boot)
- Cache miss: 10-30s (package install + snapshot creation)

qcow2 optimizations:
- lazy_refcounts=on: Postpone metadata updates
- cluster_size=2M: Reduce metadata overhead
- preallocation=metadata: Quick provisioning

Snapshot structure on disk:
- {cache_key}.qcow2 (qcow2 image with backing file)
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

from exec_sandbox.models import Language

if TYPE_CHECKING:
    from exec_sandbox.vm_manager import QemuVM, VmManager

import aiofiles
import aiofiles.os
import zstandard as zstd

from exec_sandbox import constants
from exec_sandbox._imports import require_aioboto3
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import GuestAgentError, SnapshotError, VmError
from exec_sandbox.guest_agent_protocol import (
    ExecutionCompleteMessage,
    InstallPackagesRequest,
    OutputChunkMessage,
    StreamingErrorMessage,
)
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.settings import Settings  # noqa: TC001 - Used at runtime
from exec_sandbox.subprocess_utils import drain_subprocess_output

logger = get_logger(__name__)


class SnapshotManager:
    """Manages qcow2 snapshot cache with CoW backing files.

    Architecture:
    - Base images: qcow2 format (bundled at build time)
    - Snapshot chain: base.qcow2 → snapshot.qcow2 → ephemeral_overlay.qcow2
    - L1 cache: Disk-only, lazy eviction on ENOSPC
    - L3 cache: S3 with zstd compression

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

        # Track background S3 upload tasks to prevent GC
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def get_or_create_snapshot(
        self,
        language: Language,
        packages: list[str],
        tenant_id: str,
        task_id: str,
    ) -> Path:
        """Get cached snapshot or create new one.

        Cache hierarchy:
        1. Check L1 (local disk) → <400ms
        2. Check L3 (S3 download) → <5s
        3. Create new snapshot → 10-30s

        Args:
            language: Programming language
            packages: Package list with versions (e.g., ["pandas==2.1.0"])
            tenant_id: Tenant identifier
            task_id: Task identifier

        Returns:
            Path to qcow2 snapshot

        Raises:
            SnapshotError: Snapshot creation failed
        """
        cache_key = self._compute_cache_key(language, packages)

        # L1 cache check
        snapshot_path = await self._check_l1_cache(cache_key)
        if snapshot_path:
            return snapshot_path

        # L3 cache check
        try:
            return await self._download_from_s3(cache_key)
        except SnapshotError:
            pass  # Cache miss, create new

        # Cache miss: Create new snapshot
        snapshot_path = await self._create_snapshot(language, packages, cache_key, tenant_id, task_id)

        # Upload to S3 (async, fire-and-forget, tracked to prevent GC)
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
        """Compute cache key for snapshot.

        Args:
            language: Programming language
            packages: Sorted package list with versions

        Returns:
            Cache key: {language}-{packages_hash}
        """
        packages_str = "|".join(sorted(packages))
        packages_hash = hashlib.sha256(packages_str.encode()).hexdigest()
        return f"{language.value}-{packages_hash}"

    async def _check_l1_cache(self, cache_key: str) -> Path | None:
        """Check L1 local disk cache.

        Returns:
            Snapshot qcow2 path if found, None otherwise
        """
        snapshot_path = self.cache_dir / f"{cache_key}.qcow2"

        if not await aiofiles.os.path.exists(snapshot_path):
            return None

        # Verify qcow2 format
        try:
            proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    "qemu-img",
                    "info",
                    str(snapshot_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            )
            await proc.communicate()

            if proc.returncode != 0:
                return None  # Invalid qcow2

        except (OSError, FileNotFoundError):
            return None

        # Update atime for LRU tracking
        snapshot_path.touch(exist_ok=True)

        return snapshot_path

    async def _create_snapshot(  # noqa: PLR0912, PLR0915
        self,
        language: str,
        packages: list[str],
        cache_key: str,
        tenant_id: str,
        task_id: str,
    ) -> Path:
        """Create new qcow2 snapshot with packages installed using asyncio.wait racing.

        Architecture:
        - Semaphore prevents resource exhaustion (max 1 concurrent snapshot)
        - asyncio.wait(FIRST_COMPLETED) races install vs VM death
        - Instant crash detection (not 120s timeout)
        - Proper task cancellation on completion

        Workflow:
        1. Acquire creation semaphore (prevent resource exhaustion)
        2. Create qcow2 with backing file (base image)
        3. Boot VM with snapshot image
        4. Install packages via guest agent (with death monitoring)
        5. Shutdown VM (writes committed to snapshot)
        6. Return snapshot path

        Args:
            language: Programming language
            packages: Package list with versions
            cache_key: Snapshot cache key
            tenant_id: Tenant identifier
            task_id: Task identifier

        Returns:
            Path to created qcow2 snapshot

        Raises:
            SnapshotError: Creation failed
            VmError: VM crashed during snapshot creation
        """
        start_time = time.time()
        snapshot_path = self.cache_dir / f"{cache_key}.qcow2"
        base_image = self.vm_manager.get_base_image(language)

        # Acquire semaphore to limit concurrent snapshot creation
        async with self._creation_semaphore:
            vm = None  # Track VM for cleanup

            try:
                # Step 1: Create qcow2 with backing file
                await self._create_snapshot_image(snapshot_path, base_image, cache_key, language, packages, tenant_id)

                # Step 2: Determine allowed domains for package installation
                # Restrict to package registries only during snapshot creation
                if language == "python":
                    package_domains = list(constants.PYTHON_PACKAGE_DOMAINS)
                elif language == "javascript":
                    package_domains = list(constants.NPM_PACKAGE_DOMAINS)
                else:
                    package_domains = []  # No filtering for other languages

                # Step 3: Create VM with network restricted to package registries
                vm = await self.vm_manager.create_vm(
                    language,
                    tenant_id,
                    task_id,
                    snapshot_path=snapshot_path,
                    allow_network=True,
                    allowed_domains=package_domains,  # Explicit restriction
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

                # Step 5: Shutdown QEMU process (but keep overlay for commit)
                # CRITICAL: Must shutdown QEMU before qemu-img commit (cannot commit live images)
                # but must NOT delete overlay yet (need it for commit operation)
                overlay_path = vm.overlay_image

                # Shutdown QEMU process cleanly (don't call vm.destroy() - it deletes overlay!)
                if vm.process.returncode is None:
                    await vm.process.terminate()
                    try:
                        await asyncio.wait_for(vm.process.wait(), timeout=5.0)
                    except TimeoutError:
                        await vm.process.kill()
                        await vm.process.wait()

                # Step 6: Commit overlay changes to snapshot
                # QEMU shutdown, overlay still exists on disk
                await self._commit_overlay_to_snapshot(overlay_path, snapshot_path)

                # Step 7: Clean up ALL resources (now safe to delete overlay)
                # This calls vm.destroy() which deletes overlay + cgroup
                await self.vm_manager.destroy_vm(vm)
                vm = None

            # Handle disk full (lazy eviction)
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    # Evict oldest snapshot and retry once
                    # Cleanup handled by finally block
                    await self._evict_oldest_snapshot()
                    return await self._create_snapshot(language, packages, cache_key, tenant_id, task_id)
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
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Snapshot created",
            extra={
                "cache_key": cache_key,
                "language": language,
                "package_count": len(packages),
                "duration_ms": f"{duration_ms:.1f}",
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
        cmd = [
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            str(base_image),
            "-o",
            # Performance Optimization: Extended L2 Entries (QEMU 5.2+)
            # Divides 128KB clusters into 32 x 4KB subclusters for granular allocation
            # Benefits:
            # - 10-15x faster IOPS during package install (CoW with backing files)
            # - Reduces write amplification (partial cluster updates)
            # - Smaller snapshots (less data duplication)
            # Trade-offs:
            # - Requires QEMU ≥5.2 (we have 10.1.2 ✓)
            # - Incompatible with QEMU <5.2 (not a concern, internal use only)
            # - cluster_size must be ≥16KB (128KB chosen for balance)
            # L2 cache: Default 32MB sufficient (needs only ~128KB for 500MB image)
            "lazy_refcounts=on,extended_l2=on,cluster_size=128k",
            str(snapshot_path),
        ]

        proc = ProcessWrapper(
            await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        _stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise SnapshotError(
                f"qemu-img create failed: {stderr.decode()}",
                context={
                    "cache_key": cache_key,
                    "language": language,
                    "packages": packages,
                    "tenant_id": tenant_id,
                },
            )

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

    async def _commit_overlay_to_snapshot(
        self,
        overlay_path: Path,
        snapshot_path: Path,
    ) -> None:
        """Commit overlay changes to snapshot using qemu-img commit.

        QEMU backing file chain architecture:
        - Base image (read-only, system-wide)
        - Snapshot (writable, backed by base)
        - Overlay (ephemeral, backed by snapshot)

        After package installation in VM:
        - Packages written to overlay (top layer)
        - Overlay is ephemeral (deleted on VM destroy)
        - Must commit overlay → snapshot for persistence

        This method runs `qemu-img commit` to merge overlay changes
        into the snapshot backing file, ensuring packages persist.

        CRITICAL: Must run AFTER VM shutdown (qemu-img cannot run on live images).

        Args:
            overlay_path: Ephemeral overlay file to commit from
            snapshot_path: Snapshot file to commit into

        Raises:
            SnapshotError: qemu-img commit failed
        """
        logger.info(
            "Committing overlay to snapshot",
            extra={"overlay": str(overlay_path), "snapshot": str(snapshot_path)},
        )

        cmd = ["qemu-img", "commit", str(overlay_path)]

        proc = ProcessWrapper(
            await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )

        # Drain subprocess output using subprocess_utils (prevents pipe deadlock)
        context_id = f"commit-{overlay_path.name}"
        drain_task = asyncio.create_task(
            drain_subprocess_output(
                proc,
                process_name="qemu-img",
                context_id=context_id,
                stdout_handler=lambda line: logger.info(
                    "qemu-img commit stdout", extra={"context_id": context_id, "output": line}
                ),
                stderr_handler=lambda line: logger.info(
                    "qemu-img commit stderr", extra={"context_id": context_id, "output": line}
                ),
            )
        )

        # Wait for process completion
        returncode = await proc.wait()

        # Wait for output draining to complete
        await drain_task

        if returncode != 0:
            raise SnapshotError(
                f"qemu-img commit failed with exit code {returncode}",
                context={
                    "overlay": str(overlay_path),
                    "snapshot": str(snapshot_path),
                    "exit_code": returncode,
                },
            )

        logger.info(
            "Overlay committed successfully",
            extra={"overlay": str(overlay_path), "snapshot": str(snapshot_path)},
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
        """Download and decompress snapshot from S3 to L1 cache.

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
            dctx = zstd.ZstdDecompressor()

            def _decompress():
                with Path(compressed_path).open("rb") as src, Path(snapshot_path).open("wb") as dst:
                    dctx.copy_stream(src, dst)

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

        Args:
            cache_key: Snapshot cache key
            snapshot_path: Local qcow2 snapshot path
        """
        compressed_path = self.cache_dir / f"{cache_key}.qcow2.zst"

        try:
            # Compress with zstd (level 3 for speed, run in thread pool to avoid blocking)
            cctx = zstd.ZstdCompressor(level=3)

            def _compress():
                with Path(snapshot_path).open("rb") as src, Path(compressed_path).open("wb") as dst:
                    cctx.copy_stream(src, dst)

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

        except (OSError, RuntimeError, ConnectionError):
            # Silent failure (L1 cache still works)
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


# Import VmState for type checking in finally block
from exec_sandbox.vm_manager import VmState  # noqa: E402

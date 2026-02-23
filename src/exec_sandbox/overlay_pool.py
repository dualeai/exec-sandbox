"""Pre-created qcow2 overlay pool for instant VM boot.

The overlay pool eliminates 30-430ms disk I/O from VM boot critical path by
pre-creating qcow2 overlay files. Acquisition is <1ms (atomic rename, or copy+unlink cross-filesystem) vs
30-430ms (qemu-img create under contention).

Architecture:
- Separate pool per base image (python/javascript)
- Pool directory under tempfile.gettempdir() for fast file move
- Background replenishment maintains pool size after allocations
- Graceful fallback to on-demand creation when pool is exhausted
- Uses qemu-storage-daemon for fast overlay creation (~4ms vs ~39ms)
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import Self
from uuid import uuid4

import aiofiles.os

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import VmOverlayError as QemuImgError
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.qemu_storage_daemon import QemuStorageDaemon, QemuStorageDaemonError

logger = get_logger(__name__)


async def create_qcow2_overlay(base_image: Path, overlay_path: Path) -> None:
    """Create qcow2 overlay with copy-on-write backing file.

    Subprocess-based overlay creation using qemu-img. Used by:
    - SnapshotManager (snapshot creation)

    Note: OverlayPool uses QemuStorageDaemon for faster overlay creation.

    Options rationale:
    - lazy_refcounts=on: Faster writes, metadata crash-consistent via fsync
    - extended_l2=on: Sub-cluster allocation (better for small writes)
    - cluster_size=128k: Balance between metadata size and I/O efficiency

    Args:
        base_image: Base qcow2 image (backing file, read-only)
        overlay_path: Path for new overlay file (will be created)

    Raises:
        QemuImgError: qemu-img command failed
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
        "lazy_refcounts=on,extended_l2=on,cluster_size=128k",
        str(overlay_path),
    ]

    proc = ProcessWrapper(
        await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        stderr_text = stderr.decode()
        raise QemuImgError(f"qemu-img create failed: {stderr_text}", stderr=stderr_text)


class OverlayPool:
    """Pre-created qcow2 overlay pool for instant VM boot.

    The pool maintains pre-created overlay files for each base image, enabling
    <1ms allocation via file move instead of 30-430ms qemu-img create.

    All overlay creation is centralized here - callers just request an overlay
    and the pool handles fast path (from pool) or slow path (on-demand) internally.

    Base images are auto-discovered from images_path matching pattern *-base-*.qcow2.

    Usage:
        async with OverlayPool(pool_size=5, images_path=base_images_dir) as pool:
            # Always returns an overlay - fast if from pool, slow if created on-demand
            pool_hit = await pool.acquire(base_image, target_path)
            # pool_hit indicates if it was from pool (for metrics), but overlay is ready
    """

    def __init__(
        self,
        pool_size: int = 0,
        *,
        images_path: Path | None = None,
        pool_dir: Path | None = None,
    ) -> None:
        """Initialize overlay pool.

        Args:
            pool_size: Number of pre-created overlays per base image (0 = on-demand only)
            images_path: Directory containing base images (for auto-discovery)
            pool_dir: Optional pool directory (default: tempdir/exec-sandbox-overlay-pool)
        """
        self._pool_dir = pool_dir or Path(tempfile.gettempdir()) / "exec-sandbox-overlay-pool"
        self._pool_size = pool_size
        self._images_path = images_path
        self._pools: dict[str, asyncio.Queue[Path]] = {}  # base_image_path -> queue
        self._replenish_tasks: set[asyncio.Task[None]] = set()
        self._shutdown_event = asyncio.Event()
        self._started = False
        # Semaphore to limit concurrent overlay creation (throttles both startup and replenishment)
        self._creation_sem = asyncio.Semaphore(constants.OVERLAY_POOL_REPLENISH_BATCH_SIZE)
        # qemu-storage-daemon for fast overlay creation
        self._daemon: QemuStorageDaemon | None = None

    @property
    def pool_size(self) -> int:
        """Get configured pool size per base image."""
        return self._pool_size

    @pool_size.setter
    def pool_size(self, value: int) -> None:
        """Set pool size. Must be called before start()."""
        if self._started:
            raise RuntimeError("Cannot change pool_size after start()")
        self._pool_size = value

    @property
    def daemon_enabled(self) -> bool:
        """Whether qemu-storage-daemon is active for fast overlay creation."""
        return self._daemon is not None and self._daemon.started

    def _discover_base_images(self) -> list[Path]:
        """Discover all base images in images_path.

        Uses pattern *-base-*.qcow2 to find all base images (python, javascript,
        raw, and any future languages) without needing per-language configuration.

        Returns:
            List of discovered base image paths (sorted for determinism)
        """
        if not self._images_path:
            return []

        # Single pattern catches all base images: python-*-base-*.qcow2, node-*-base-*.qcow2, etc.
        # Use absolute paths so qemu-img can find backing files from any working directory
        matches = [p.resolve() for p in self._images_path.glob("*-base-*.qcow2")]
        return sorted(matches)

    async def start(self, base_images: list[Path] | None = None) -> None:
        """Start the overlay pool and pre-create overlays for base images.

        If base_images is None, auto-discovers from images_path.

        The daemon is always started (needed for on-demand overlay creation in acquire()),
        but pre-creation and replenishment are only enabled when pool_size > 0.

        Args:
            base_images: Optional explicit list of base image paths (for testing)

        Raises:
            RuntimeError: If start() is called when pool is already started
        """
        if self._started:
            raise RuntimeError("Overlay pool already started - call stop() first")

        # Clear shutdown event to allow restart after previous shutdown
        self._shutdown_event.clear()

        # Start qemu-storage-daemon (always needed for on-demand overlay creation)
        self._daemon = QemuStorageDaemon()
        await self._daemon.start()

        self._started = True

        # Skip pre-creation when pool is disabled
        if self._pool_size <= 0:
            logger.debug("Overlay pool pre-creation disabled (pool_size=0), daemon started for on-demand creation")
            return

        # Auto-discover if not provided
        if base_images is None:
            base_images = self._discover_base_images()

        # Create pool directory (graceful degradation on permission errors)
        try:
            await aiofiles.os.makedirs(self._pool_dir, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Failed to create overlay pool directory, pool disabled",
                extra={"pool_dir": str(self._pool_dir), "error": str(e)},
            )
            return

        # Initialize pools for each base image
        for base_image in base_images:
            key = str(base_image.resolve())
            self._pools[key] = asyncio.Queue(maxsize=self._pool_size)

        # Pre-create overlays (concurrency is limited by _creation_sem in _create_and_enqueue)
        tasks = [
            self._create_and_enqueue(base_image, str(base_image.resolve()))
            for base_image in base_images
            for _ in range(self._pool_size)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Start background replenishment tasks
        for base_image in base_images:
            task = asyncio.create_task(self._replenish_loop(base_image))
            self._replenish_tasks.add(task)
            task.add_done_callback(self._replenish_tasks.discard)

        logger.info(
            "Overlay pool started",
            extra={
                "pool_size": self._pool_size,
                "base_images": [str(p) for p in base_images],
                "pool_dir": str(self._pool_dir),
                "daemon_enabled": self.daemon_enabled,
            },
        )

    async def stop(self) -> None:
        """Stop the overlay pool: cancel tasks, stop daemon, cleanup pool directory."""
        if not self._started:
            return

        # Signal shutdown to replenishment loops
        self._shutdown_event.set()

        # Cancel all replenishment tasks
        for task in list(self._replenish_tasks):
            task.cancel()

        # Wait for tasks to complete
        if self._replenish_tasks:
            await asyncio.gather(*self._replenish_tasks, return_exceptions=True)
        self._replenish_tasks.clear()

        # Stop qemu-storage-daemon
        if self._daemon:
            await self._daemon.stop()
            self._daemon = None

        # Clean up pool directory
        if self._pool_dir.exists():
            try:
                shutil.rmtree(self._pool_dir)
            except OSError as e:
                logger.warning(
                    "Failed to cleanup overlay pool directory",
                    extra={"pool_dir": str(self._pool_dir), "error": str(e)},
                )

        self._pools.clear()
        self._started = False
        logger.info("Overlay pool shutdown complete")

    async def acquire(self, base_image: Path, target_path: Path) -> bool:
        """Acquire overlay for target path.

        Always provides an overlay - from pool if available (fast), or creates
        on-demand if pool is exhausted (slow). Caller doesn't need fallback logic.

        Args:
            base_image: Base image path (backing file for overlay)
            target_path: Target path for the overlay file

        Returns:
            True if acquired from pool (fast path, <1ms),
            False if created on-demand (slow path, ~8ms via daemon)

        Raises:
            FileExistsError: target_path already exists
            RuntimeError: Daemon not started (call start() first)
            QemuStorageDaemonError: Failed to create overlay via daemon
        """
        # Prevent silent overwrite of existing files
        if target_path.exists():
            raise FileExistsError(f"Target overlay already exists: {target_path}")

        key = str(base_image.resolve())

        # Ensure pool exists for this base image (atomic check-and-set to avoid race)
        # setdefault is atomic in CPython - prevents duplicate pools from concurrent acquire()
        new_queue: asyncio.Queue[Path] = asyncio.Queue(maxsize=max(1, self._pool_size))
        pool = self._pools.setdefault(key, new_queue)
        is_new_pool = pool is new_queue

        # Start replenishment for new base image (if pool is enabled and we created it)
        if is_new_pool and self._started and self._pool_size > 0:
            task = asyncio.create_task(self._replenish_loop(base_image))
            self._replenish_tasks.add(task)
            task.add_done_callback(self._replenish_tasks.discard)

        # Try fast path: get pre-created overlay from pool
        try:
            overlay_path = pool.get_nowait()
            # Move to target (same filesystem = atomic rename, cross-filesystem = copy+unlink)
            try:
                await asyncio.to_thread(shutil.move, overlay_path, target_path)
                logger.debug(
                    "Overlay acquired from pool",
                    extra={"base_image": key, "target": str(target_path)},
                )
                return True
            except OSError as e:
                # Move failed (cross-filesystem or other error)
                logger.warning(
                    "Failed to move pooled overlay, falling back to on-demand: %s",
                    e,
                    extra={"source": str(overlay_path), "target": str(target_path)},
                )
                # Clean up the orphaned overlay file
                with contextlib.suppress(OSError):
                    if overlay_path.exists():
                        overlay_path.unlink()
                # Clean up partial target that cross-FS copy may have created
                with contextlib.suppress(OSError):
                    if target_path.exists():
                        target_path.unlink()
                # Fall through to slow path
        except asyncio.QueueEmpty:
            logger.debug(
                "Overlay pool exhausted, creating on-demand",
                extra={"base_image": key, "pool_size": self._pool_size},
            )

        # Slow path: create overlay on-demand via daemon
        if self._daemon is None:
            raise RuntimeError("Daemon must be started before acquire - call start() first")
        await self._daemon.create_overlay(base_image, target_path)
        return False

    async def _create_and_enqueue(self, base_image: Path, key: str) -> None:
        """Create overlay and add to pool queue.

        Uses _creation_sem to limit concurrent overlay creations, preventing
        disk I/O stampede during startup and replenishment.

        Args:
            base_image: Base image path
            key: Pool key (resolved base image path)
        """
        async with self._creation_sem:
            try:
                # Generate unique filename
                overlay_path = self._pool_dir / f"overlay-{uuid4()}.qcow2"

                # Create overlay via daemon
                if self._daemon is None:
                    raise RuntimeError("Daemon must be started")
                await self._daemon.create_overlay(base_image, overlay_path)

                # Add to pool (non-blocking, may fail if full)
                pool = self._pools.get(key)
                if pool:
                    try:
                        pool.put_nowait(overlay_path)
                    except asyncio.QueueFull:
                        # Pool is full, clean up extra overlay
                        with contextlib.suppress(OSError):
                            overlay_path.unlink()
            except (OSError, QemuStorageDaemonError) as e:
                logger.warning(
                    "Failed to create pooled overlay",
                    extra={"base_image": str(base_image), "error": str(e)},
                )

    async def _replenish_loop(self, base_image: Path) -> None:
        """Background task to maintain pool size.

        Args:
            base_image: Base image to create overlays for
        """
        key = str(base_image.resolve())

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(constants.OVERLAY_POOL_REPLENISH_INTERVAL_SECONDS)

                if self._shutdown_event.is_set():
                    break

                pool = self._pools.get(key)
                if not pool:
                    break

                # Calculate how many overlays to create
                current_size = pool.qsize()
                needed = self._pool_size - current_size

                if needed <= 0:
                    continue

                # Create overlays in batches
                batch_size = min(needed, constants.OVERLAY_POOL_REPLENISH_BATCH_SIZE)
                tasks = [self._create_and_enqueue(base_image, key) for _ in range(batch_size)]
                await asyncio.gather(*tasks, return_exceptions=True)

                logger.debug(
                    "Overlay pool replenished",
                    extra={
                        "base_image": key,
                        "created_count": batch_size,  # Avoid "created", conflicts with LogRecord.created
                        "current_size": pool.qsize(),
                        "target_size": self._pool_size,
                    },
                )

            except asyncio.CancelledError:
                break
            except (OSError, QemuStorageDaemonError) as e:
                logger.warning(
                    "Replenishment error",
                    extra={"base_image": key, "error": str(e)},
                )

    def get_stats(self) -> dict[str, int]:
        """Get pool sizes for monitoring.

        Returns:
            Dict mapping base image paths to current pool sizes
        """
        return {key: pool.qsize() for key, pool in self._pools.items()}

    async def __aenter__(self) -> Self:
        """Enter async context manager, starting the pool."""
        await self.start()
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, stopping the pool."""
        await self.stop()

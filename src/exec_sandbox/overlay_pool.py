"""Pre-created qcow2 overlay pool for instant VM boot.

The overlay pool takes the overlay creation out of the VM boot by pre-creating
qcow2 overlay files. Acquisition is <1ms (atomic rename, or copy+unlink cross-filesystem);
an empty pool makes the overlay on demand through qemu-storage-daemon.

Architecture:
- Separate pool per base image (one per language)
- Pool directory under tempfile.gettempdir() for fast file move
- One background loop per base image fills the pool after start(), then tops it up
  every OVERLAY_POOL_REPLENISH_INTERVAL_SECONDS
- Graceful fallback to on-demand creation when pool is exhausted
- Uses qemu-storage-daemon for fast overlay creation (a clone of the base image's template,
  ~0.3 ms, or QMP for the first, ~5 ms; ~39 ms with a qemu-img process)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from uuid import uuid4

import psutil

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import settle_and_report
from exec_sandbox.exceptions import TransientError
from exec_sandbox.permission_utils import make_owned_dir
from exec_sandbox.qemu_storage_daemon import QemuStorageDaemon

logger = get_logger(__name__)


class OverlayPool:
    """Pre-created qcow2 overlay pool for instant VM boot.

    The pool maintains pre-created overlay files for each base image, enabling
    <1ms allocation via file move instead of an on-demand creation through the daemon.

    All overlay creation is centralized here - callers just request an overlay
    and the pool handles fast path (from pool) or slow path (on-demand) internally.

    Usage:
        pool = OverlayPool(pool_size=5, pool_dir=None)
        await pool.start([base_image])  # returns once the daemon runs; the pool fills in the background
        # Always returns an overlay - fast if from pool, slow if created on-demand
        pool_hit = await pool.acquire(base_image, target_path)
        # pool_hit indicates if it was from pool (for metrics), but overlay is ready
        await pool.stop()
    """

    def __init__(self, *, pool_size: int, pool_dir: Path | None) -> None:
        """Initialize overlay pool.

        Args:
            pool_size: Number of pre-created overlays per base image (0 = on-demand only);
                set it before start()
            pool_dir: Pool directory, or None: a new folder at each start,
                tempdir/exec-sandbox-overlay-pool-<pid>-<random>, made by mkdtemp
        """
        # A given folder (tests) or, at each start, a new one from mkdtemp: a name another
        # user can guess in the shared temp folder could be made first, with planted overlays.
        self._given_pool_dir = pool_dir
        self._pool_dir: Path | None = pool_dir
        self.pool_size = pool_size
        self._pools: dict[str, asyncio.Queue[Path]] = {}  # base_image_path -> queue
        self._replenish_tasks: set[asyncio.Task[None]] = set()
        self._shutdown_event = asyncio.Event()
        self._started = False
        # Semaphore to limit concurrent overlay creation (throttles both startup and replenishment)
        self._creation_sem = asyncio.Semaphore(constants.OVERLAY_POOL_CREATION_CONCURRENCY)
        # qemu-storage-daemon for fast overlay creation
        self._daemon: QemuStorageDaemon | None = None

    @property
    def daemon_enabled(self) -> bool:
        """Whether qemu-storage-daemon is active for fast overlay creation."""
        return self._daemon is not None and self._daemon.started

    @staticmethod
    def _cleanup_stale_pool_dirs() -> int:
        """Remove stale PID-scoped pool directories from dead processes.

        Directory names encode the owner PID: exec-sandbox-overlay-pool-{pid}-{mkdtemp
        suffix}. When the owner dies (crash, OOM, kill -9), the directory becomes stale.
        Only real folders of this user: another user's folder (or a link) in the shared
        temp folder is left alone.

        Returns:
            Number of stale directories cleaned up.
        """
        tmpdir = Path(tempfile.gettempdir())
        pattern = re.compile(r"^exec-sandbox-overlay-pool-(\d+)-[a-z0-9_]+$")
        cleaned = 0

        for pool_dir in tmpdir.glob("exec-sandbox-overlay-pool-*"):
            match = pattern.match(pool_dir.name)
            if match is None:  # not a pool folder name (e.g. exec-sandbox-overlay-pool-abc)
                continue
            try:
                st = pool_dir.lstat()
            except OSError:
                continue
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
                continue
            if psutil.pid_exists(int(match.group(1))):
                continue  # its owner runs (ours or another user's)

            # Remove stale directory
            try:
                shutil.rmtree(pool_dir)
                cleaned += 1
                logger.debug("Removed stale overlay pool directory", extra={"pool_dir": str(pool_dir)})
            except OSError as e:
                logger.debug(
                    "Failed to remove stale overlay pool directory",
                    extra={"pool_dir": str(pool_dir), "error": str(e)},
                )

        return cleaned

    async def start(self, base_images: list[Path]) -> None:
        """Start the pool, rolling back any partially-started daemon lifecycle.

        Raises:
            TransientError: OVERLAY_FAILED: the daemon did not start or connect
            ConfigError: HOST_DEPENDENCY: qemu-storage-daemon is missing or cannot run;
                SETTING_INVALID: the given pool_dir is not a private folder of this user
            RuntimeError: the pool is already started (it keeps running)
        """
        if self._started:  # before the rollback below: it would stop the running pool
            raise RuntimeError("Overlay pool already started - call stop() first")
        try:
            await self._start_impl(base_images)
        except BaseException:
            rollback = asyncio.create_task(self.stop())
            if (rollback_error := await settle_and_report(rollback)) is not None:
                logger.error(
                    "Overlay pool startup rollback failed",
                    extra={"error": str(rollback_error)},
                )
            raise

    async def _start_impl(self, base_images: list[Path]) -> None:
        """Start the daemon, then fill the pool for base_images in the background.

        The daemon is always started (needed for on-demand overlay creation in acquire()).
        When pool_size > 0, one replenishment task per base image creates the overlays:
        start() does not wait for them, as acquire() creates an overlay on demand (~0.3 ms once
        the template of the base image exists, ~5 ms for the first; the fill runs
        OVERLAY_POOL_CREATION_CONCURRENCY creations at once, so it does not delay it) while the
        pool is empty. A pool folder that cannot be made (OSError) turns the pool off.

        Args:
            base_images: Base images to keep overlays for

        Raises:
            TransientError, ConfigError: as start()
        """

        # Clean up stale pool directories from dead processes
        try:
            count = await asyncio.to_thread(self._cleanup_stale_pool_dirs)
            if count:
                logger.info("Cleaned up stale overlay pool directories", extra={"count": count})
        except OSError:
            logger.debug("Stale pool directory cleanup failed (non-critical)", exc_info=True)

        # Clear shutdown event to allow restart after previous shutdown
        self._shutdown_event.clear()

        # Start qemu-storage-daemon (always needed for on-demand overlay creation)
        self._daemon = QemuStorageDaemon()
        await self._daemon.start()

        self._started = True

        # Skip pre-creation when pool is disabled
        if self.pool_size <= 0:
            logger.debug("Overlay pool pre-creation disabled (pool_size=0), daemon started for on-demand creation")
            return

        # Create pool directory (graceful degradation on permission errors)
        try:
            if self._given_pool_dir is not None:
                make_owned_dir(self._given_pool_dir, setting="pool_dir")
                self._pool_dir = self._given_pool_dir
            else:  # 0700, a new name: not writable by qemu-vm's group (its overlays go to other VMs)
                self._pool_dir = Path(tempfile.mkdtemp(prefix=f"exec-sandbox-overlay-pool-{os.getpid()}-"))
        except OSError as e:
            logger.warning(
                "Failed to create overlay pool directory, pool disabled",
                extra={"pool_dir": str(self._pool_dir), "error": str(e)},
            )
            # Off, as with pool_size 0: acquire() makes each overlay on demand and starts
            # no replenish loop (each would fail on the missing folder every 0.5 s)
            self.pool_size = 0
            return

        # Initialize pools for each base image
        for base_image in base_images:
            key = str(base_image.resolve())
            self._pools[key] = asyncio.Queue(maxsize=self.pool_size)

        # Fill in the background (concurrency is limited by _creation_sem in _create_and_enqueue)
        for base_image in base_images:
            task = asyncio.create_task(self._replenish_loop(base_image))
            self._replenish_tasks.add(task)
            task.add_done_callback(self._replenish_tasks.discard)

        logger.info(
            "Overlay pool started",
            extra={
                "pool_size": self.pool_size,
                "base_images": [str(p) for p in base_images],
                "pool_dir": str(self._pool_dir),
                "daemon_enabled": self.daemon_enabled,
            },
        )

    async def stop(self) -> None:
        """Stop the overlay pool: cancel tasks, stop daemon, cleanup pool directory."""
        if not self._started and self._daemon is None:
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
        if self._pool_dir is not None and self._pool_dir.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, self._pool_dir)
            except OSError as e:
                logger.warning(
                    "Failed to cleanup overlay pool directory",
                    extra={"pool_dir": str(self._pool_dir), "error": str(e)},
                )
        self._pool_dir = self._given_pool_dir

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
            False if created on-demand (slow path via the daemon: ~0.3 ms as a template
            clone, ~5 ms through QMP)

        Raises:
            FileExistsError: target_path already exists
            RuntimeError: Daemon not started (call start() first)
            TransientError: OVERLAY_FAILED if the daemon failed to create the overlay
        """
        # Prevent silent overwrite of existing files
        if target_path.exists():
            raise FileExistsError(f"Target overlay already exists: {target_path}")

        key = str(base_image.resolve())

        # Ensure pool exists for this base image (atomic check-and-set to avoid race)
        # setdefault is atomic in CPython - prevents duplicate pools from concurrent acquire()
        new_queue: asyncio.Queue[Path] = asyncio.Queue(maxsize=max(1, self.pool_size))
        pool = self._pools.setdefault(key, new_queue)
        is_new_pool = pool is new_queue

        # Start replenishment for new base image (if pool is enabled and we created it)
        if is_new_pool and self._started and self.pool_size > 0:
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
                # Move failed (shutil.move copies across filesystems: that copy, or the rename, failed)
                logger.warning(
                    "Failed to move pooled overlay, falling back to on-demand: %s",
                    e,
                    extra={"source": str(overlay_path), "target": str(target_path)},
                )
                # Clean up the orphaned overlay file (a new overlay: metadata only, small)
                with contextlib.suppress(OSError):
                    overlay_path.unlink(missing_ok=True)
                # Clean up partial target that cross-FS copy may have created
                with contextlib.suppress(OSError):
                    target_path.unlink(missing_ok=True)
                # Fall through to slow path
        except asyncio.QueueEmpty:
            logger.debug(
                "Overlay pool exhausted, creating on-demand",
                extra={"base_image": key, "pool_size": self.pool_size},
            )

        # Slow path: create overlay on-demand via daemon
        if self._daemon is None:
            raise RuntimeError("Daemon must be started before acquire - call start() first")
        await self._daemon.create_overlay(base_image, target_path)
        return False

    async def _create_and_enqueue(self, base_image: Path, key: str) -> None:
        """Create overlay and add it to the pool queue of `key` (the resolved base image path).
        One fill loop runs per base image and makes only the missing overlays, and stop() ends
        the loops before it removes the queues: the queue is there and has room.

        Uses _creation_sem to limit concurrent overlay creations, preventing
        disk I/O stampede during startup and replenishment.
        """
        async with self._creation_sem:
            if self._pool_dir is None or self._daemon is None:  # start() sets both before the loops
                return
            overlay_path = self._pool_dir / f"overlay-{uuid4()}.qcow2"
            try:
                await self._daemon.create_overlay(base_image, overlay_path)
            except (OSError, TransientError) as e:
                logger.warning(
                    "Failed to create pooled overlay",
                    extra={"base_image": str(base_image), "error": str(e)},
                )
                return
            self._pools[key].put_nowait(overlay_path)

    async def _replenish_loop(self, base_image: Path) -> None:
        """Background task to maintain pool size: it creates every missing overlay right after
        start(), then every OVERLAY_POOL_REPLENISH_INTERVAL_SECONDS.

        Args:
            base_image: Base image to create overlays for
        """
        key = str(base_image.resolve())
        while not self._shutdown_event.is_set() and (pool := self._pools.get(key)) is not None:
            needed = self.pool_size - pool.qsize()
            if needed > 0:
                # Create every missing overlay (_creation_sem limits how many run at once)
                await asyncio.gather(
                    *(self._create_and_enqueue(base_image, key) for _ in range(needed)), return_exceptions=True
                )
                logger.debug(
                    "Overlay pool replenished",
                    extra={
                        "base_image": key,
                        "created_count": needed,  # Not "created" — conflicts with LogRecord.created
                        "current_size": pool.qsize(),
                        "target_size": self.pool_size,
                    },
                )
            await asyncio.sleep(constants.OVERLAY_POOL_REPLENISH_INTERVAL_SECONDS)

"""Base class for snapshot cache managers.

Shared lifecycle, background task tracking, image hashing, file locking,
and eviction patterns used by both L2 disk cache (DiskSnapshotManager)
and L1 memory cache (MemorySnapshotManager).
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
from typing import TYPE_CHECKING, ClassVar, Self

from exec_sandbox._logging import get_logger
from exec_sandbox.hash_utils import crc32

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

logger = get_logger(__name__)


class BaseCacheManager:
    """Base for snapshot cache managers (L1 memory + L2 disk).

    Provides:
    - Cache directory initialization
    - Background task lifecycle (track, stop, await)
    - Stat-based image fingerprinting (O(1) mtime+size)
    - File-based flock for cross-process deduplication
    - LRU eviction by atime
    - Async context manager protocol

    Subclasses MUST set _cache_ext (e.g., ".vmstate", ".qcow2") to define
    the primary file extension for their cache entries.
    """

    _cache_ext: ClassVar[str]
    """Primary file extension for cache entries (e.g., ".vmstate", ".qcow2")."""

    _sidecar_exts: ClassVar[tuple[str, ...]] = ()
    """Extra extensions to clean up on eviction alongside the primary cache file.

    Each value is appended to the stem (filename minus _cache_ext). For example,
    if the primary file is 'key.vmstate' and _sidecar_exts = ('.vmstate.meta',),
    eviction also removes 'key.vmstate.meta'.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._background_tasks: dict[asyncio.Task[None], str] = {}

    @property
    def background_task_count(self) -> int:
        """Number of in-flight background tasks."""
        return len(self._background_tasks)

    @property
    def background_task_names(self) -> list[str]:
        """Names of in-flight background tasks (for monitoring)."""
        return list(self._background_tasks.values())

    def _track_task(self, task: asyncio.Task[None], *, name: str = "unnamed") -> None:
        """Register a background task with lifecycle logging.

        Tasks are tracked to prevent GC and awaited on stop().
        Completion and failure are logged for observability.
        """
        self._background_tasks[task] = name
        logger.info("Background task started", extra={"task_name": name, "in_flight": len(self._background_tasks)})

        def _on_done(t: asyncio.Task[None]) -> None:
            task_name = self._background_tasks.pop(t, name)
            if t.cancelled():
                logger.info("Background task cancelled", extra={"task_name": task_name})
            elif t.exception():
                logger.warning(
                    "Background task failed",
                    extra={"task_name": task_name, "error": str(t.exception())},
                )
            else:
                logger.info("Background task completed", extra={"task_name": task_name})

        task.add_done_callback(_on_done)

    @contextlib.asynccontextmanager
    async def _save_lock(self, key: str) -> AsyncIterator[bool]:
        """File-based exclusive lock for snapshot save. Cross-process safe.

        Acquires non-blocking flock on {key}{ext}.lock, then checks whether
        {key}{ext} already exists and is non-empty (detects if a concurrent
        process completed the save before us).

        Yields True if caller should proceed with save (lock acquired, file absent).
        Yields False if another process completed the save first.
        Raises BlockingIOError if lock is held by another process/task.
        """
        lock_path = self.cache_dir / f"{key}{self._cache_ext}.lock"
        target_path = self.cache_dir / f"{key}{self._cache_ext}"
        fd = lock_path.open("w")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield not (target_path.exists() and target_path.stat().st_size > 0)
        finally:
            fd.close()  # Closing fd releases the flock
            # Lock files are intentionally NOT deleted. Deleting after close creates
            # a race: another process can acquire a flock on the old inode, then a
            # third process creates a NEW file on a different inode — both hold
            # exclusive flocks simultaneously. Persistent lock files are tiny and
            # eliminate this POSIX flock race entirely.

    async def _evict_oldest_snapshot(self) -> None:
        """Evict oldest cache entry by atime (lazy, on ENOSPC).

        Removes the primary file and any sidecar files defined by _sidecar_exts.
        Skips entries with an active save lock to avoid deleting in-progress saves.
        """
        cache_files = sorted(
            self.cache_dir.glob(f"*{self._cache_ext}"),
            key=lambda p: p.stat().st_atime,
        )

        for candidate in cache_files:
            # Skip entries with an active save lock (flock held by another process).
            # Try non-blocking flock: if it succeeds, no save is in progress.
            stem = candidate.name[: -len(self._cache_ext)]
            lock_path = candidate.parent / f"{stem}{self._cache_ext}.lock"
            if lock_path.exists():
                try:
                    fd = lock_path.open("w")
                    try:
                        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        fd.close()
                except BlockingIOError:
                    logger.debug("Skipping locked cache entry for eviction", extra={"path": str(candidate)})
                    continue

            logger.info("Evicting oldest snapshot", extra={"path": str(candidate)})
            candidate.unlink(missing_ok=True)
            for ext in self._sidecar_exts:
                (candidate.parent / f"{stem}{ext}").unlink(missing_ok=True)
            return

    async def stop(self) -> None:
        """Wait for all background tasks to finish, then clear."""
        if self._background_tasks:
            task_names = list(self._background_tasks.values())
            logger.info(
                "Waiting for background tasks",
                extra={"count": len(task_names), "tasks": task_names},
            )
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        await self.stop()

    @staticmethod
    def image_hash(path: Path) -> str:
        """Stat-based fingerprint of a file for change detection (O(1), no read).

        Hashes mtime_ns + file size to detect rebuilds without reading file
        content. Sufficient for detecting image rebuilds (which always change
        mtime) but not for detecting content tampering.
        Returns "missing0" if file does not exist.
        """
        try:
            stat = path.stat()
            return crc32(f"{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            return "missing0"

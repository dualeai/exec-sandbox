"""Base class for snapshot cache managers.

Shared lifecycle, background task tracking, image hashing, file locking,
and eviction patterns used by both L2 disk cache (DiskSnapshotManager)
and L1 memory cache (MemorySnapshotManager).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
from typing import TYPE_CHECKING, ClassVar, Self

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.lock_utils import try_file_lock

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

logger = get_logger(__name__)

# In-process digest cache: path -> (file_stamp, sha256 hex). One entry per path.
_IMAGE_DIGEST_CACHE: dict[str, tuple[str, str]] = {}


def file_stamp(stat: os.stat_result) -> str:
    """One version of a file for the in-process caches: inode, mtime and size. A
    rebuilt or replaced file gets a new stamp."""
    return f"{stat.st_ino}:{stat.st_mtime_ns}:{stat.st_size}"


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

    def __init__(self, cache_dir: Path, *, setting: str) -> None:
        """Make the cache folder (_make_cache_dir); *setting* names its config field."""
        self.cache_dir = cache_dir
        self._setting = setting
        self._make_cache_dir()
        self._background_tasks: dict[asyncio.Task[None], str] = {}

    def _make_cache_dir(self) -> None:
        """Make the cache folder if it is not there: at start, then before each build or save
        lock, so a folder removed at run time (a tmp cleaner) is made again. Raises ConfigError
        SETTING_INVALID when it cannot be made (no access, a read-only disk, a file in its place)."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ConfigError(
                f"Cache folder {self.cache_dir} cannot be made: {e.strerror or type(e).__name__}",
                code=ErrorCode.SETTING_INVALID,
                context={"cache_dir": str(self.cache_dir), "errno": e.errno},
                hint=f"Set {self._setting} to a folder this user can write, on a writable disk.",
                instance_id=None,
            ) from e

    def _track_task(self, task: asyncio.Task[None], *, name: str) -> None:
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

    def _cache_path(self, key: str) -> Path:
        """Return the primary cache file path for *key*."""
        return self.cache_dir / f"{key}{self._cache_ext}"

    @contextlib.asynccontextmanager
    async def _save_lock(self, key: str) -> AsyncIterator[bool]:
        """File-based exclusive lock for snapshot save. Cross-process safe.

        Acquires non-blocking flock on {key}{ext}.lock, then checks whether
        {key}{ext} already exists and is non-empty (detects if a concurrent
        process completed the save before us).

        Yields True if caller should proceed with save (lock acquired, file absent).
        Yields False if another process completed the save first.
        Raises BlockingIOError if lock is held by another process/task, ConfigError
        SETTING_INVALID if the cache folder cannot be made again.
        """
        self._make_cache_dir()
        with try_file_lock(self._cache_path(key)) as should_proceed:
            yield should_proceed

    def _remove_entry(self, path: Path) -> None:
        """Delete every file of one cache entry: its sidecars, then the primary file `path`.
        Last, because no save of the key starts while the primary file exists (_save_lock):
        a removal in a worker thread never deletes the files of a new save of the same key.

        Blocking (an L1 .ram file is the RAM size, an L2 image up to GBs): async callers
        use asyncio.to_thread.
        """
        stem = path.name[: -len(self._cache_ext)]
        for ext in self._sidecar_exts:
            (path.parent / f"{stem}{ext}").unlink(missing_ok=True)
        path.unlink(missing_ok=True)

    async def _evict_oldest_snapshot(self) -> None:
        """Evict oldest cache entry by atime (lazy, on ENOSPC).

        Removes the primary file and any sidecar files defined by _sidecar_exts.
        Skips entries with an active save lock to avoid deleting in-progress saves.
        One worker-thread job: the listing has no size limit and an entry can be GBs.
        """
        await asyncio.to_thread(self._evict_oldest_entry)

    def _evict_oldest_entry(self) -> None:
        """_evict_oldest_snapshot itself (blocking)."""
        cache_files = sorted(
            self.cache_dir.glob(f"*{self._cache_ext}"),
            key=lambda p: p.stat().st_atime,
        )

        for candidate in cache_files:
            # Skip entries with an active save lock (flock held by another process).
            # Try non-blocking flock: if it succeeds, no save is in progress. The
            # files go under the lock: no task starts a save of the entry meanwhile.
            try:
                with try_file_lock(candidate):
                    logger.info("Evicting oldest snapshot", extra={"path": str(candidate)})
                    self._remove_entry(candidate)
            except BlockingIOError:
                logger.debug("Skipping locked cache entry for eviction", extra={"path": str(candidate)})
                continue
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
        """SHA-256 content digest of a file (identity of a base image, kernel or L2 layer).

        The digest is the same on every host for the same bytes, so cache keys
        built from it can be shared across hosts (an mtime-based fingerprint
        differs per host). The file is read once; the digest is cached in memory
        and in a ``<file>.sha256`` sidecar keyed by inode, mtime and size, so a
        rebuilt file is re-hashed. Blocking: async callers use asyncio.to_thread.
        Returns "missing0" if the file does not exist.
        """
        try:
            stat = path.stat()
        except OSError:
            return "missing0"
        stamp = file_stamp(stat)
        cached = _IMAGE_DIGEST_CACHE.get(str(path))
        if cached is not None and cached[0] == stamp:
            return cached[1]
        sidecar = path.with_name(f"{path.name}.sha256")
        try:
            sidecar_stamp, sidecar_digest = sidecar.read_text().split()
            if sidecar_stamp == stamp:
                _IMAGE_DIGEST_CACHE[str(path)] = (stamp, sidecar_digest)
                return sidecar_digest
        except (OSError, ValueError):
            pass
        try:
            with path.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
        except FileNotFoundError:
            return "missing0"  # Removed between stat and open (eviction)
        _IMAGE_DIGEST_CACHE[str(path)] = (stamp, digest)
        # Best effort: the image directory may be read-only (installed assets).
        with contextlib.suppress(OSError):
            tmp = sidecar.with_name(f"{sidecar.name}.{os.getpid()}.tmp")
            tmp.write_text(f"{stamp} {digest}")
            tmp.replace(sidecar)
        return digest

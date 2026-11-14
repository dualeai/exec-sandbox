"""Base class for snapshot cache managers.

Shared lifecycle, background task tracking, file locking and eviction patterns
used by both L2 disk cache (DiskSnapshotManager) and L1 memory cache
(MemorySnapshotManager). Image digests live in images.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections import deque
from typing import TYPE_CHECKING, ClassVar

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.files.disk import allocated_bytes, file_stamp
from exec_sandbox.files.locks import CacheEntry
from exec_sandbox.permission_utils import make_owned_dir

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from pathlib import Path

    from exec_sandbox.files.locks import EntryLease

logger = get_logger(__name__)


class EvictionOrder:
    """One pass of evictions over a cache (gate 4 of admission, an ENOSPC of a save): its
    entries listed once, oldest access first, at the first evict_next; then removed in that
    order. Each evict_next removes the next entry that no VM uses and no save holds (a lease
    of it open in any process: its blocks would not come back), and skips for the rest of the
    pass one that is in use, being saved, gone, or whose lock file cannot be made or opened (a
    full or read-only disk). An entry used after the listing keeps its place (the pass evicts by
    the access times at its start); one saved again since (a new version: its file stamp
    changed) is skipped, never removed as the old one. Calls in turn, not at once."""

    def __init__(
        self,
        *,
        oldest_first: Callable[[], list[tuple[Path, str]]],
        remove_unused: Callable[[Path, str], int | None],
    ) -> None:
        self._oldest_first = oldest_first
        self._remove_unused = remove_unused
        self._entries: deque[tuple[Path, str]] | None = None

    async def evict_next(self) -> int | None:
        """Remove the next entry that can go: the bytes its files took (what comes back to the
        free space), None when no such entry is left. One worker-thread job (the first one lists
        the cache; an entry can be GBs)."""
        return await asyncio.to_thread(self._evict_next)

    def _evict_next(self) -> int | None:
        """evict_next itself (blocking)."""
        if self._entries is None:
            self._entries = deque(self._oldest_first())
        while self._entries:
            candidate, stamp = self._entries.popleft()
            try:
                size = self._remove_unused(candidate, stamp)  # only the version listed: not a new save of the key
            except OSError as e:  # its lock file cannot be made (ENOSPC on XFS or btrfs) or opened: the next one
                logger.warning("Cache entry not evicted", extra={"path": str(candidate), "error": str(e)})
                continue
            if size is None:
                logger.debug("Skipping cache entry in use or being saved", extra={"path": str(candidate)})
                continue
            logger.info("Evicted oldest snapshot", extra={"path": str(candidate), "bytes": size})
            return size
        return None


class BaseCacheManager:
    """Base for snapshot cache managers (L1 memory + L2 disk).

    Provides:
    - Cache directory initialization
    - Background task lifecycle (track, stop, await)
    - File-based flock for cross-process deduplication (files.locks.CacheEntry)
    - Leases on entries (files.locks.EntryLease): eviction skips an entry a VM runs on
    - Eviction of the entry with the oldest access time (on ENOSPC, and for admission's
      home disk space: EvictionOrder, one listing per pass); removal of a broken entry
      under the same locks

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
        """Make the cache folder (_make_cache_dir); *setting* names its config field.

        Raises:
            ConfigError: SETTING_INVALID: the folder cannot be made
        """
        # One spelling for every path of the cache (symlinks resolved): the paths of its entries
        # compare equal (`held`); files.locks keys its locks by the folder's inode
        self.cache_dir = cache_dir.resolve()
        self._setting = setting
        self._make_cache_dir()
        self._background_tasks: dict[asyncio.Task[None], str] = {}

    def _make_cache_dir(self) -> None:
        """Make the cache folder if it is not there: at start, then before each build or save
        lock, so a folder removed at run time (a tmp cleaner) is made again. Raises ConfigError
        SETTING_INVALID when it cannot be made (no access, a read-only disk, a file in its place)."""
        try:
            make_owned_dir(
                self.cache_dir, setting=self._setting
            )  # 0700 whatever the umask and group (sg qemu-vm, umask 002)
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
    async def _save_lock(self, key: str) -> AsyncGenerator[bool]:
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
        with CacheEntry(self._cache_path(key)).try_saving() as should_proceed:
            yield should_proceed

    def _entry_files(self, path: Path) -> list[Path]:
        """Every file of the entry whose primary file is `path`: its sidecars, then `path`."""
        stem = path.name[: -len(self._cache_ext)]
        return [*(path.parent / f"{stem}{ext}" for ext in self._sidecar_exts), path]

    def _entry_bytes(self, path: Path) -> int:
        """Disk blocks of the files of one entry, in bytes (sparse files count what they hold).
        Blocking: a worker thread."""
        total = 0
        for file in self._entry_files(path):
            with contextlib.suppress(FileNotFoundError):
                total += allocated_bytes(file.stat())
        return total

    def _remove_entry(self, path: Path) -> None:
        """Delete every file of one cache entry: its sidecars, then the primary file `path`.
        Last, because no save of the key starts while the primary file exists (_save_lock):
        a removal in a worker thread never deletes the files of a new save of the same key.

        Blocking (an L1 .ram file is the RAM size, an L2 image up to GBs): async callers
        use asyncio.to_thread.
        """
        for file in self._entry_files(path):
            file.unlink(missing_ok=True)

    def _remove_unused(self, path: Path, *, stamp: str | None) -> int | None:
        """Remove the entry whose primary file is `path` under its use lock and its save lock,
        both without wait: the bytes its files took, or None when a lease of it is open (in
        any process: a VM runs on it), a save or another removal holds it, it is gone, or
        (with `stamp`) its primary file is no longer that version (a new save of the key).

        Raises:
            OSError: a lock file cannot be made or opened (a full or read-only disk, no
                access), or a file cannot be removed

        Blocking: a worker thread.
        """
        try:
            with CacheEntry(path).try_evicting():
                try:
                    current = file_stamp(path.stat())
                except FileNotFoundError:
                    return None
                if stamp is not None and current != stamp:
                    return None
                size = self._entry_bytes(path)
                self._remove_entry(path)
                return size
        except BlockingIOError:
            return None

    async def _remove_broken(self, path: Path, *, stamp: str, lease: EntryLease | None) -> None:
        """Remove the entry `path` that a check found broken (`stamp`: the version it saw),
        as an eviction does (_remove_unused): `lease`, the caller's own, goes first; an
        entry another lease holds stays (its VM runs on it). One worker-thread job; the
        caller's miss returns after it (a cancel leaves the job running)."""
        if lease is not None:
            lease.release()
        try:
            removed = await asyncio.to_thread(self._remove_unused, path, stamp=stamp)
        except OSError as e:
            logger.warning("Broken cache entry not removed", extra={"path": str(path), "error": str(e)})
            return
        if removed is None:
            logger.info("Broken cache entry kept: in use, being saved, or saved again", extra={"path": str(path)})

    def eviction_order(self) -> EvictionOrder:
        """A pass of evictions over this cache, oldest entry first (EvictionOrder): the cache is
        listed once, at the pass's first eviction, not once per eviction."""
        return EvictionOrder(
            oldest_first=self._oldest_first,
            remove_unused=lambda path, stamp: self._remove_unused(path, stamp=stamp),
        )

    async def evict_oldest(self) -> int | None:
        """One eviction (EvictionOrder.evict_next of a new order): on ENOSPC of a save. Returns
        the bytes its files took, None when no entry can go."""
        return await self.eviction_order().evict_next()

    def _oldest_first(self) -> list[tuple[Path, str]]:
        """The primary files of the entries with their version (file_stamp), oldest access time
        (st_atime) first. Blocking: the listing has no size limit."""
        entries: list[tuple[float, Path, str]] = []
        for path in self.cache_dir.glob(f"*{self._cache_ext}"):
            with contextlib.suppress(FileNotFoundError):  # evicted by another process meanwhile
                st = path.stat()
                entries.append((st.st_atime, path, file_stamp(st)))
        return [(path, stamp) for _, path, stamp in sorted(entries)]

    async def reclaimable_bytes(self, *, held: frozenset[Path]) -> int:
        """Bytes of the entries that evictions of this process can give back once its VMs are
        gone: this process may remove them (_evictable_here), no lease of another process
        holds them, and they are not in `held` (the entries the caller itself keeps leased
        meanwhile). One worker-thread job (a listing)."""
        return await asyncio.to_thread(self._reclaimable_bytes, held=held)

    def _reclaimable_bytes(self, *, held: frozenset[Path]) -> int:
        """reclaimable_bytes itself (blocking)."""
        if not os.access(self.cache_dir, os.W_OK | os.X_OK):  # no file of it can be removed here
            return 0
        total = 0
        for path in self.cache_dir.glob(f"*{self._cache_ext}"):
            if path in held or not self._evictable_here(path):
                continue
            try:
                if CacheEntry(path).leased_elsewhere():
                    continue
            except OSError:  # its use lock cannot be read: no eviction of this process can take it
                continue
            total += self._entry_bytes(path)
        return total

    @staticmethod
    def _evictable_here(path: Path) -> bool:
        """Whether an eviction of this process can take the locks of the entry `path`: it opens
        both lock files for write (CacheEntry.try_evicting), and makes them when they do
        not exist (the folder is writable: the caller checks it). A lock file of another user
        that this user may not write (0644) stops it. Blocking."""
        for lock in CacheEntry(path).lock_files:
            try:
                if not os.access(lock, os.W_OK) and lock.exists():
                    return False
            except OSError:  # no access to it: no eviction either
                return False
        return True

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

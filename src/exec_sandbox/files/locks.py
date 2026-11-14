"""Cross-process file locking: flock for the save locks, POSIX record locks for the use locks.

Lock files are intentionally never deleted.  Deleting after close creates a
POSIX race: process A acquires flock on inode X, process B unlinks the path and
creates inode Y, process C flocks inode Y — both A and C now hold "exclusive"
locks.  Persistent lock files (a few bytes each) eliminate this entirely.

Two-level locking prevents thread-pool starvation:
  1. asyncio.Lock (per-path) serializes in-process coroutines — zero threads.
  2. flock() guards cross-process access — at most 1 thread per distinct path.
Without level 1, N coroutines each submit blocking to_thread(flock) to the
default executor; the winning coroutine then needs the same executor for file
I/O (asyncio.to_thread), but every slot is occupied by flock waiters — deadlock.
See CERT TPS01-J ("Do not execute interdependent tasks in a bounded thread
pool") for the general class of bug.

A cache entry has two lock files. ``<entry>.lock`` is the save lock: one save or
build of the entry at a time (CacheEntry.saving, CacheEntry.try_saving). ``<entry>.use.lock`` is the
use lock: each VM that runs on the entry holds it shared (EntryLease) until its QEMU
is confirmed dead, and eviction tries it exclusive without wait (CacheEntry.try_evicting), so
it never removes a file a running VM still reads or maps (its blocks would not come
back). The use lock is a file of its own: a VM that held the save lock shared would
stop a rebuild of its key (the build waits for the save lock exclusive) for its whole life.

The use lock is a POSIX record lock (fcntl), not a flock. A record lock belongs to a
process: F_GETLK tells a process whether another process holds it, and in which mode
(shared: a lease; exclusive: an eviction). Two rules follow. The record locks of one
process never conflict with each other, so this module checks the leases and evictions
of this process itself (_use_locks, keyed by the folder and the name of the lock file).
Closing any file of a lock file drops every record lock of the process on it, so this
process has at most one file open on each use lock, and closes it only once it holds
nothing more there.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import functools
import os
import struct
import sys
import threading
import weakref
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator
    from pathlib import Path


@dataclass
class _PathLock:
    """The in-process lock of one path, and how many coroutines hold or wait for it."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


# Per-path asyncio locks that prevent in-process coroutines from flooding the
# thread pool with blocking flock() calls.  Only the front-of-queue coroutine
# ever enters to_thread(flock); all others wait here at zero thread cost.
# One map per event loop: a Lock binds to the loop that first waits on it, so a
# second loop (asyncio.run again in this process) must never get it. An entry
# goes when its last user leaves, so the map holds only the paths in use.
_async_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _PathLock]] = weakref.WeakKeyDictionary()


def _lock_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".lock")


def _open_lock_file(lock_path: Path) -> IO[bytes]:
    """A file open for read and write on the lock file `lock_path`, made when it does not
    exist. An existing one opens without O_CREAT: in a sticky folder that all may write, Linux
    refuses O_CREAT on a file of another user (fs.protected_regular 1 or 2, the systemd
    default), also to root and to the owner of the folder (fs/namei.c may_create_in_sticky).
    A new one is 0600: a process that can open it can hold its lock, and a QEMU that runs as
    qemu-vm must not stop the builds and evictions of this user (open() gives 0664, group
    qemu-vm, under `sg qemu-vm` and umask 002). Never O_TRUNC (0.4 ms on APFS); the file
    stays empty."""
    try:
        return lock_path.open("r+b")
    except FileNotFoundError:
        return os.fdopen(os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600), "r+b")


def _use_lock_path(target: Path) -> Path:
    """The use lock of the cache entry *target*: ``foo.qcow2`` → ``foo.qcow2.use.lock``."""
    return _lock_path(target.with_name(f"{target.name}.use"))


# This process's hold on each use lock, by _lock_key. File work (open, lock, close, stat) runs
# outside the guard: the event loop takes it too (CacheEntry.lease, share, release), and a full
# disk can make each call slow. A thread marks the entry `work` first: other threads of this
# process wait for it, so this process never opens a second file on a lock it holds, nor
# closes one while another thread uses it.
type _LockKey = tuple[int, int, str]
_use_locks: dict[_LockKey, _UseLock] = {}
_use_guard = threading.Condition()

# struct flock for F_GETLK: (l_type, l_whence, l_start, l_len, l_pid) in the field order of the OS
_FLOCK_FORMAT: Final = "=qqihh" if sys.platform == "darwin" else "=hh4xqqi4x"
_LOCK_BUSY: Final = frozenset({errno.EACCES, errno.EAGAIN})
"""errno of a record lock held by another process (POSIX allows either)."""


@dataclass
class _UseLock:
    """This process's hold on one use lock, through its only file open on it."""

    file: IO[bytes] | None = None
    """Open while this process holds the lock shared (leases)."""
    leases: int = 0
    work: Literal["lease", "close", "evict", "check"] | None = None
    """File work of one thread of this process on the lock now, or None."""


class EntryLease:
    """A lease on the cache entry `path`: eviction skips the entry while any lease of it is
    open, in any process. The caller that looked the entry up holds one until its VM launch
    is over; each VM holds its own (share) until its QEMU is confirmed dead. A lease holds
    no lock where the lock file cannot be made (a read-only disk or folder: _open_use_lock).
    release() is idempotent."""

    def __init__(self, path: Path, *, key: _LockKey | None) -> None:
        self.path = path
        self._key = key
        """Key of the lock this lease counts in (_use_locks), or None: no lock."""
        self._open = True

    def share(self) -> EntryLease:
        """Another lease of the same entry: it stays leased while either is open. No file
        work, so it never fails on an open lease."""
        with _use_guard:
            if not self._open:
                raise ValueError(f"lease of {self.path} already released")
            if self._key is not None:
                _use_locks[self._key].leases += 1
        return EntryLease(self.path, key=self._key)

    def release(self) -> None:
        """Close this lease; the last lease of this process closes its file (the lock goes)."""
        with _use_guard:
            if not self._open:
                return
            self._open = False
            if self._key is None:
                return
            held = _use_locks[self._key]
            held.leases -= 1
            if held.leases:
                return
            held.work = "close"
            file, held.file = held.file, None
        _end_work(self._key, file)


def _end_work(key: _LockKey, file: IO[bytes] | None) -> None:
    """End the file work of this thread on the lock `key`: close `file` (this process then
    holds nothing on the lock), then let the other threads at it."""
    try:
        if file is not None:
            file.close()
    finally:
        with _use_guard:
            del _use_locks[key]
            _use_guard.notify_all()


def _lock_key(lock_path: Path) -> _LockKey:
    """The key of a use lock in _use_locks: the device and inode of its folder, and its name.
    Every spelling of the folder gives the same key: a symlink, a relative path, a case
    variant on a case-insensitive filesystem (APFS), a bind mount (realpath merges only the
    first two). An inode number is not given again while a file of this process is open in
    that folder. One stat. Raises OSError (FileNotFoundError: no such folder)."""
    folder = lock_path.parent.stat()
    return folder.st_dev, folder.st_ino, lock_path.name


def _start_work(key: _LockKey, work: Literal["lease", "evict", "check"]) -> _UseLock | None:
    """Mark the lock `key` for file work of this thread (call with _use_guard held): the
    entry of this process's leases, or None when it was free and is marked now."""
    held = _use_locks.get(key)
    if held is None:
        _use_locks[key] = _UseLock(work=work)
    return held


def _open_use_lock(lock_path: Path) -> IO[bytes] | None:
    """A file on the use lock `lock_path`, or None when no lock can be held: the lock file
    cannot be made (a read-only disk or folder) and does not exist. Read and write when it
    can (an exclusive lock needs write), else read (a shared lock needs read only)."""
    try:
        return _open_lock_file(lock_path)
    except OSError as e:
        if e.errno not in (errno.EROFS, errno.EACCES, errno.EPERM):
            raise
    try:
        return lock_path.open("rb")  # made by a user who can write the folder
    except (FileNotFoundError, PermissionError):
        return None


def _lease(target: Path, *, wait: bool) -> EntryLease | None:
    """Lease *target*, or None when it is gone once the lock is held. Raises BlockingIOError
    (`wait` False) while an eviction holds the use lock, in this process or another, or
    another thread of this process works on it; with `wait`, waits for them (a worker thread)."""
    lock_path = _use_lock_path(target)
    if not os.path.exists(target):  # noqa: PTH110 - never raises; removed by hand while this process leases it
        return None
    try:
        key = _lock_key(lock_path)
    except FileNotFoundError:  # its folder removed meanwhile
        return None
    with _use_guard:
        while (held := _start_work(key, "lease")) is not None and (held.work is not None or not held.leases):
            if not wait:
                raise BlockingIOError(errno.EAGAIN, "cache entry being evicted", str(lock_path))
            _use_guard.wait()
        if held is not None:  # this process leases it: one more lease, no file work
            held.leases += 1
            return EntryLease(target, key=key)
    file: IO[bytes] | None = None
    try:
        file = _open_use_lock(lock_path)
        present = True
        if file is not None:
            try:
                fcntl.lockf(file, fcntl.LOCK_SH | (0 if wait else fcntl.LOCK_NB))
            except OSError as e:
                if e.errno in _LOCK_BUSY:
                    raise BlockingIOError(errno.EAGAIN, "cache entry being evicted", str(lock_path)) from e
                if e.errno != errno.EDEADLK:  # the kernel saw a lock cycle between two processes' evictions
                    raise
                present = False  # a miss
        # Once the lock is held (an eviction may have removed it meanwhile). os.path.exists
        # never raises (an error reads as gone); Path.exists raises on EACCES or EIO.
        present = present and os.path.exists(target)  # noqa: PTH110
    except BaseException:
        _end_work(key, file)
        raise
    if file is None or not present:
        _end_work(key, file)
        return EntryLease(target, key=None) if present else None
    with _use_guard:
        held = _use_locks[key]
        held.file, held.leases, held.work = file, 1, None
        _use_guard.notify_all()
    return EntryLease(target, key=key)


def _release_result(future: asyncio.Future[EntryLease | None]) -> None:
    """A lease its caller stopped waiting for (cancelled) goes as soon as its job ends."""
    if not future.cancelled() and future.exception() is None and (lease := future.result()) is not None:
        lease.release()


def _another_process_leases(file: IO[bytes]) -> bool:
    """Whether a process other than this one holds the record lock of `file` shared (a lease).
    F_GETLK places no lock and reads the kernel's lock table only (no disk I/O): it names one
    lock that conflicts with an exclusive one, and the locks of this process never conflict.
    An exclusive one is an eviction: no lease then."""
    query = struct.pack(_FLOCK_FORMAT, *_flock_fields(fcntl.F_WRLCK))
    found = struct.unpack(_FLOCK_FORMAT, fcntl.fcntl(file, fcntl.F_GETLK, query))
    return _flock_type(found) == fcntl.F_RDLCK


def _flock_fields(lock_type: int) -> tuple[int, ...]:
    """A struct flock over the whole file, in _FLOCK_FORMAT's field order."""
    if sys.platform == "darwin":
        return (0, 0, 0, lock_type, os.SEEK_SET)  # l_start, l_len, l_pid, l_type, l_whence
    return (lock_type, os.SEEK_SET, 0, 0, 0)  # l_type, l_whence, l_start, l_len, l_pid


def _flock_type(fields: tuple[int, ...]) -> int:
    return fields[3] if sys.platform == "darwin" else fields[0]


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """The locks of the cache entry `path` (its primary file; a downloaded asset too): two lock
    files beside it, never deleted (see the module docstring).

    - The save lock, ``<path>.lock``: one save or build of the entry at a time. `saving` waits
      for it, `try_saving` does not; both yield True while the entry is still to make (missing
      or empty), False when another caller made it meanwhile.
    - The use lock, ``<path>.use.lock``: each VM that runs on the entry holds a lease of it
      (`lease`, then EntryLease.share and release) until its QEMU is confirmed dead. A lease
      lives across tasks (the lookup takes it, each VM shares it), so it is an object, not a
      `with` block. Eviction takes the use lock exclusive, then the save lock (`try_evicting`):
      it never removes a file that a running VM still reads or maps.
    """

    path: Path

    @property
    def lock_files(self) -> tuple[Path, Path]:
        """Its use lock, then its save lock."""
        return _use_lock_path(self.path), _lock_path(self.path)

    @asynccontextmanager
    async def saving(self) -> AsyncGenerator[bool]:
        """The save lock exclusive, waiting for it (in one worker thread per path, see the
        module docstring). Yields True while the entry is still to make (missing or empty),
        False when another caller made it. The lock goes when the block ends (fd close)."""
        # Level 1: serialize in-process coroutines on an asyncio.Lock so only
        # one coroutine per path ever blocks a thread-pool worker on flock().
        lock_path = _lock_path(self.path)
        key = str(lock_path)
        locks = _async_locks.setdefault(asyncio.get_running_loop(), {})
        entry = locks.setdefault(key, _PathLock())
        entry.users += 1
        try:
            await entry.lock.acquire()
            try:
                fd = _open_lock_file(lock_path)
                try:
                    # Level 2: cross-process guard (at most 1 thread per path).
                    await asyncio.to_thread(fcntl.flock, fd.fileno(), fcntl.LOCK_EX)
                    yield not (self.path.exists() and self.path.stat().st_size > 0)
                finally:
                    fd.close()  # Closing fd releases the flock
            finally:
                entry.lock.release()
        finally:
            entry.users -= 1
            if not entry.users:
                del locks[key]

    @contextmanager
    def try_saving(self) -> Generator[bool]:
        """The save lock exclusive, without wait: raises BlockingIOError at once while it is
        held, else yields as `saving`. For sync code (a worker thread) and for async code that
        skips a held lock (the L1 save)."""
        with _open_lock_file(_lock_path(self.path)) as fd:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield not (self.path.exists() and self.path.stat().st_size > 0)

    async def lease(self) -> EntryLease | None:
        """A lease on the entry, or None when it is gone (evicted, removed or never made: then
        no lock file is made). Waits only while an eviction removes the entry, or another thread
        of this process opens or closes its lock (in a worker thread), then returns None or the
        lease.

        Raises:
            OSError: the use lock cannot be opened (out of file handles, a disk error)
        """
        if not os.path.exists(self.path):  # noqa: PTH110 - never raises (an error reads as gone): Path.exists can
            return None
        try:
            return _lease(self.path, wait=False)
        except BlockingIOError:
            pass
        future = asyncio.get_running_loop().run_in_executor(None, functools.partial(_lease, self.path, wait=True))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(_release_result)
            raise

    @contextmanager
    def try_evicting(self) -> Generator[None]:
        """Both locks exclusive, without wait, in the order every eviction takes them: the use
        lock (no lease open, in any process), then the save lock (no save of the entry). The
        caller removes the entry inside. Raises BlockingIOError at once while either is held,
        OSError when a lock file cannot be made or opened (a full or read-only disk, no
        access). Blocking (a worker thread)."""
        with self._try_unused(), self.try_saving():
            yield

    @contextmanager
    def _try_unused(self) -> Generator[None]:
        """The use lock exclusive, without wait (try_evicting's first lock): BlockingIOError
        while a lease of the entry is open, in this process or another, or another eviction
        or check holds it."""
        lock_path = _use_lock_path(self.path)
        key = _lock_key(lock_path)
        with _use_guard:
            if _start_work(key, "evict") is not None:
                raise BlockingIOError(errno.EAGAIN, "cache entry in use", str(lock_path))
        file: IO[bytes] | None = None
        try:
            file = _open_lock_file(lock_path)
            try:
                fcntl.lockf(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                if e.errno in _LOCK_BUSY:
                    raise BlockingIOError(errno.EAGAIN, "cache entry in use", str(lock_path)) from e
                raise
            yield
        finally:
            _end_work(key, file)

    def leased_elsewhere(self) -> bool:
        """Whether a lease of another process holds the entry. The leases of this process do
        not count (their VMs end, and the entry can then be evicted), nor does an eviction in
        progress (it removes the entry). Makes no lock file. Raises OSError when the use lock
        cannot be read. Blocking (a worker thread): it waits while another thread of this
        process opens or closes the lock."""
        lock_path = _use_lock_path(self.path)
        key = _lock_key(lock_path)
        with _use_guard:
            while (held := _start_work(key, "check")) is not None:
                if held.work == "evict":
                    return False  # this process removes the entry
                if held.work is None and held.file is not None:
                    return _another_process_leases(held.file)  # through this process's own file
                _use_guard.wait()
        file: IO[bytes] | None = None
        try:
            try:
                file = lock_path.open("rb")
            except FileNotFoundError:
                return False
            return _another_process_leases(file)
        finally:
            _end_work(key, file)  # this process holds no lock on it: the close drops nothing

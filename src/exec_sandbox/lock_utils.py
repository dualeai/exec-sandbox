"""Cross-process file locking using POSIX flock.

Lock files are intentionally never deleted.  Deleting after close creates a
POSIX race: process A acquires flock on inode X, process B unlinks the path and
creates inode Y, process C flocks inode Y — both A and C now hold "exclusive"
locks.  Persistent lock files (a few bytes each) eliminate this entirely.

Two-level locking prevents thread-pool starvation:
  1. asyncio.Lock (per-path) serializes in-process coroutines — zero threads.
  2. flock() guards cross-process access — at most 1 thread per distinct path.
Without level 1, N coroutines each submit blocking to_thread(flock) to the
default executor; the winning coroutine then needs the same executor for file
I/O (aiofiles), but every slot is occupied by flock waiters — deadlock.
See CERT TPS01-J ("Do not execute interdependent tasks in a bounded thread
pool") for the general class of bug.
"""

from __future__ import annotations

import asyncio
import fcntl
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator
    from pathlib import Path

# Per-path asyncio locks that prevent in-process coroutines from flooding the
# thread pool with blocking flock() calls.  Only the front-of-queue coroutine
# ever enters to_thread(flock); all others wait here at zero thread cost.
_async_locks: dict[str, asyncio.Lock] = {}


@asynccontextmanager
async def file_lock(target: Path) -> AsyncGenerator[bool]:
    """Acquire an exclusive flock for *target*, creating a sibling .lock file;
    wait in a thread until the lock is available.

    The lock file is ``target.with_suffix(target.suffix + ".lock")``, e.g.
    ``foo.qcow2`` → ``foo.qcow2.lock``.

    Yields:
        True if *target* does not yet exist (or is empty) — caller should
        proceed with producing it.  False if another caller already completed
        the work.

    The lock is released when the context manager exits (fd close).
    """
    # Level 1: serialize in-process coroutines on an asyncio.Lock so only
    # one coroutine per path ever blocks a thread-pool worker on flock().
    lock_path = _lock_path(target)
    alock = _async_locks.setdefault(str(lock_path), asyncio.Lock())
    await alock.acquire()
    try:
        fd = lock_path.open("a")  # "a", not "w": no O_TRUNC (0.4 ms on APFS), the file stays empty
        try:
            # Level 2: cross-process guard (at most 1 thread per path).
            await asyncio.to_thread(fcntl.flock, fd.fileno(), fcntl.LOCK_EX)
            yield not (target.exists() and target.stat().st_size > 0)
        finally:
            fd.close()  # Closing fd releases the flock
    finally:
        alock.release()


@contextmanager
def try_file_lock(target: Path) -> Generator[bool]:
    """The same flock as ``file_lock``, without the wait: raises ``BlockingIOError``
    at once when the lock is held, else yields the same flag. For sync code (a
    worker thread) and for async code that skips a held lock (the L1 save)."""
    with _lock_path(target).open("a") as fd:  # "a", not "w": no O_TRUNC
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield not (target.exists() and target.stat().st_size > 0)


def _lock_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".lock")

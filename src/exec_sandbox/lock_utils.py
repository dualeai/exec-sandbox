"""Cross-process file locking using POSIX flock.

Lock files are intentionally never deleted.  Deleting after close creates a
POSIX race: process A acquires flock on inode X, process B unlinks the path and
creates inode Y, process C flocks inode Y — both A and C now hold "exclusive"
locks.  Persistent lock files (a few bytes each) eliminate this entirely.
"""

from __future__ import annotations

import asyncio
import fcntl
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@asynccontextmanager
async def file_lock(
    target: Path,
    *,
    blocking: bool = True,
) -> AsyncIterator[bool]:
    """Acquire an exclusive flock for *target*, creating a sibling .lock file.

    The lock file is ``target.with_suffix(target.suffix + ".lock")``, e.g.
    ``foo.qcow2`` → ``foo.qcow2.lock``.

    Args:
        target: Path to the file being protected.
        blocking: If True (default), wait in a thread until the lock is
            available.  If False, raise ``BlockingIOError`` immediately when
            the lock is held by another caller.

    Yields:
        True if *target* does not yet exist (or is empty) — caller should
        proceed with producing it.  False if another caller already completed
        the work.

    The lock is released when the context manager exits (fd close).
    """
    lock_path = target.with_suffix(target.suffix + ".lock")
    fd = lock_path.open("w")
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        if blocking:
            # Offload to thread so the event loop keeps running while we wait.
            await asyncio.to_thread(fcntl.flock, fd.fileno(), flags)
        else:
            fcntl.flock(fd.fileno(), flags)
        yield not (target.exists() and target.stat().st_size > 0)
    finally:
        fd.close()  # Closing fd releases the flock

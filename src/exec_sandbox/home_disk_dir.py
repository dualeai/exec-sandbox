"""Per-manager folder for VM home disks, under Settings.home_disk_dir.

Each VmManager makes its own folder (mode 0700) and holds an exclusive flock
on a file in it for its whole life. A folder whose lock nobody holds belongs
to a process that died: the next manager to start removes it whole
(`remove_dead_folders`). No process needs another one's temp directory or VM
list, so managers that share the root (service and CLI, different TMPDIR,
PrivateTmp) never remove each other's disks.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import stat
import tempfile
import time
import weakref
from pathlib import Path

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.permission_utils import private_dir_problem

logger = get_logger(__name__)

_LOCK_NAME = ".lock"
_LOCKLESS_GRACE_SECONDS = 300
"""A folder with no lock file yet: its owner is between mkdtemp and creating the
lock, or died there. Removed only once it is this old."""


class HomeDiskDir:
    """This manager's home disk folder. Open with `HomeDiskDir.open`; `close` at stop."""

    def __init__(self, path: Path, lock_fd: int) -> None:
        self.path = path
        # Also runs if the manager is dropped without stop (tests, crashes in
        # __init__ callers): the folder and the lock never outlive the object.
        self._finalizer = weakref.finalize(self, _release, path, lock_fd, os.getpid())

    @classmethod
    def open(cls, root: Path) -> HomeDiskDir:
        """Make and lock a new folder under `root` (metadata calls only)."""
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _check_root(root)
        path = Path(tempfile.mkdtemp(prefix="vmm-", dir=root))
        # Lock under another name, then rename: a sweep only ever sees ".lock"
        # already held (between create and flock it would see a free lock).
        staging = path / f"{_LOCK_NAME}.new"
        fd = os.open(staging, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # a new file: nobody else can hold it
        staging.rename(path / _LOCK_NAME)
        return cls(path, fd)

    def close(self) -> None:
        """Remove the folder and release the lock. Idempotent. Blocking (the folder
        holds whole disks): async callers use asyncio.to_thread."""
        self._finalizer()


def _release(path: Path, lock_fd: int, owner_pid: int) -> None:
    # A forked child runs this finalizer too: it closes its copy of the fd (the
    # parent keeps the lock) and never removes the parent's live folder.
    if os.getpid() == owner_pid:
        shutil.rmtree(path, ignore_errors=True)
    os.close(lock_fd)


def _check_root(root: Path) -> None:
    """The root holds every tenant's disk folders: a real directory of this user, not writable by others."""
    st = root.lstat()
    if problem := private_dir_problem(root, st, forbidden=stat.S_IWGRP | stat.S_IWOTH):
        raise ConfigError(
            f"home_disk_dir {root} {problem[0]}. {problem[1]}",
            code=ErrorCode.SETTING_INVALID,
            context={"home_disk_dir": str(root), "mode": oct(stat.S_IMODE(st.st_mode)), "uid": st.st_uid},
            hint=None,
            instance_id=None,
        )


def remove_dead_folders(root: Path) -> None:
    """Remove the folders of dead managers under `root`; a live manager's (its lock
    held, this process's own included) stays. Blocking (a listing with no size limit,
    whole disks): async callers use asyncio.to_thread."""
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        try:
            fd = os.open(folder / _LOCK_NAME, os.O_RDWR)
        except FileNotFoundError:
            with contextlib.suppress(OSError):
                if time.time() - folder.stat().st_mtime > _LOCKLESS_GRACE_SECONDS:
                    shutil.rmtree(folder, ignore_errors=True)
            continue
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            continue  # held: its manager is alive
        else:
            shutil.rmtree(folder, ignore_errors=True)
            logger.info("Removed home disks of a stopped process", extra={"folder": str(folder)})
        finally:
            os.close(fd)

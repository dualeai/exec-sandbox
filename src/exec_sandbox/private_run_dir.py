"""Private folders of running processes, under a root folder of this user.

A process makes its own folder (mode 0700) under a root and holds an exclusive
flock on a file in it for its whole life. Each VmManager has one under
Settings.home_disk_dir (home disks, instance work files), and each L1 save VM
one in /dev/shm (its RAM file). A folder whose lock nobody holds belongs to a
process that died: the next process to sweep the root removes it whole
(`remove_dead_folders`). No process needs another one's temp directory or VM
list, so processes that share the root (service and CLI, different TMPDIR,
PrivateTmp) never remove each other's files.

The root is opened once for each use, never through a link, and checked: a folder
of this user with no ACL entry, in folders that only root and this user can change
(permission_utils.private_tree_problem); else another user could swap it for a
folder with records of their choice. Every step then goes through the open folders
(dir_fd), not through their paths.

A folder also records the VM cgroups of its process (`record_cgroup`). A QEMU
that runs as qemu-vm outlives this process: its parent is sudo, so
exit-with-parent does not end it, and this user cannot signal it; cgroup.kill
can. Three things end a recorded cgroup: `close`; a watcher process, started
with the first record in a cgroup of its own, as soon as this process is gone
(SIGKILL, or a cgroup.kill of this process's cgroup); and the sweep of the
folder, which also removes the empty cgroups, the watcher's among them.

It also records folders of its process outside it (`record_folder`: the VM
working folders in the temp folder). `close` and the sweep remove them once the
recorded cgroups have no process left.

A record is a symlink, made whole by one call. Only a target of the form this
module makes counts: a normalized absolute path, its last names of
[A-Za-z0-9_-] only, the last one the record's name (`_vm_cgroup`, `_own_folder`).
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import secrets
import shutil
import stat
import subprocess
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from exec_sandbox import cgroup, constants
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.files.disk import make_private_dir
from exec_sandbox.permission_utils import (
    get_qemu_vm_gid,
    private_dir_problem,
    private_tree_problem,
    sudo_errors_file,
)
from exec_sandbox.platform_utils import child_env

logger = get_logger(__name__)

_DIR_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FORBIDDEN: Final = stat.S_IWGRP | stat.S_IWOTH
"""Mode bits the root may not have: it only lists the folders (each 0700), so others may read it."""

_LOCK_NAME = ".lock"
_LOCKLESS_GRACE_SECONDS = 300
"""A folder with no lock file yet: its owner is between mkdir and creating the
lock, or died there. Removed only once it is this old."""

_CGROUP_RECORD: Final = ".cgroup"
"""Suffix of a VM cgroup record: a symlink <cgroup folder name>.cgroup to the cgroup."""

_FOLDER_RECORD: Final = ".folder"
"""Suffix of a folder record: a symlink <folder name>.folder to a folder of this process."""

_NAME: Final = re.compile(r"[A-Za-z0-9_-]+")
"""A name this module makes or records (tenant and task ids, VM ids, folder names): never "." or ".."."""

_WATCHER_TENANT: Final = "exec-sandbox-watcher"
"""The cgroup under code-exec with one cgroup per watcher, named as its folder: a
cgroup.kill of this process's cgroup (a systemd stop, an OOM group kill) misses it."""

_WATCHER: Final = r"""trap '' HUP INT TERM
echo armed
exec >/dev/null
while read -r _; do :; done
for record in ./*.cgroup; do
  [ "$record" = "./$1.cgroup" ] && continue
  (
    cd -P -- "$record" || exit
    rest=${PWD#"$2"/}
    tenant=${rest%%/*}
    vm=${rest#"$tenant"/}
    [ "$rest" = "$tenant/$vm" ] && [ -n "$tenant" ] && [ -n "$vm" ] || exit
    case $tenant$vm in *[!A-Za-z0-9_-]*) exit ;; esac
    [ "$record" = "./$vm.cgroup" ] || exit
    echo 1 > cgroup.kill
  ) 2>/dev/null
done
"""
"""/bin/sh script of the watcher, run in the folder ($1: its name, $2: the real path of the
cgroup namespace). Signals that come first (a terminal, a stop of the service) do not end
it; it says so ("armed") on its stdout, then closes it. Its stdin is a pipe whose only write
end this process holds (with its forked children), so `read` ends once they are all gone,
or at `close`. Then it writes "1" to cgroup.kill of each recorded cgroup but its own,
through the record, once the real path is <namespace>/<tenant>/<record name> (as
_vm_cgroup)."""

_WATCHER_START_SECONDS: Final = 5.0
"""Bound of the wait for the watcher to be armed (a few ms): a host that takes longer gets no watcher."""

_START_WATCHER: Final = (
    "exec 3<&0\n"
    '[ -z "$1" ] || { echo 0 > "$1/cgroup.procs"; } 2>/dev/null\n'
    '/bin/sh -c "$2" exec-sandbox-watcher "$3" "$4" 0<&3 3<&- &\n'
)
"""Moves itself into the watcher's cgroup ($1, when there is one), starts the watcher there
in the background, and exits at once: init, not this process, reaps the watcher. A
background command reads /dev/null unless its stdin is set explicitly, so the pipe goes
through fd 3 (a separate line: `a && b &` would put `a` in the background)."""


class PrivateRunDir:
    """The private folder of this process under a root. Open with `PrivateRunDir.open`; `close` at stop."""

    def __init__(self, path: Path, *, root_fd: int, fd: int, lock_fd: int) -> None:
        self.path = path
        self._fd = fd
        self._watcher_fds: list[int] = []  # write end of the watcher's pipe, once started
        self._watcher_lock = threading.Lock()  # records come from worker threads
        # Also runs if the owner is dropped without close (tests, crashes in
        # __init__ callers): the folder and the lock never outlive the object.
        self._finalizer = weakref.finalize(
            self,
            _release,
            name=path.name,
            root_fd=root_fd,
            fd=fd,
            lock_fd=lock_fd,
            owner_pid=os.getpid(),
            watcher_fds=self._watcher_fds,
        )

    @classmethod
    def open(cls, root: Path, *, name: str) -> PrivateRunDir:
        """Make and lock the new folder `name` (new_folder_name) under `root`. Blocking: the
        root check reads the user and group databases, which LDAP or SSSD can make slow;
        async callers use asyncio.to_thread.

        Raises:
            ConfigError: SETTING_INVALID: `root` is not a safe folder of this user (_open_root)
            FileExistsError: `name` is taken
            OSError: `root` or the folder cannot be made
        """
        make_private_dir(root)
        root_fd = _open_root(root)
        fd = lock_fd = -1
        try:
            os.mkdir(name, 0o700, dir_fd=root_fd)
            fd = os.open(name, _DIR_FLAGS, dir_fd=root_fd)
            # Lock under another name, then rename: a sweep only ever sees ".lock"
            # already held (between create and flock it would see a free lock).
            staging = f"{_LOCK_NAME}.new"
            lock_fd = os.open(staging, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # a new file: nobody else can hold it
            os.rename(staging, _LOCK_NAME, src_dir_fd=fd, dst_dir_fd=fd)
        except BaseException:
            for each in (lock_fd, fd, root_fd):
                if each >= 0:
                    os.close(each)
            raise
        return cls(root / name, root_fd=root_fd, fd=fd, lock_fd=lock_fd)

    def record_cgroup(self, cgroup_path: Path) -> None:
        """Record a VM cgroup of this process before any process goes in it. The first
        record also starts the watcher and waits until it is armed (_WATCHER): a few ms.
        Blocking: async callers use a worker thread. Thread-safe.

        Raises:
            OSError: the record could not be made (disk full, folder removed)
        """
        os.symlink(cgroup_path, f"{cgroup_path.name}{_CGROUP_RECORD}", dir_fd=self._fd)
        with self._watcher_lock:
            if not self._watcher_fds:
                self._start_watcher()

    def forget_cgroup(self, name: str) -> None:
        """Remove the record of the VM cgroup `name` (its folder name) once the cgroup is gone. Idempotent."""
        with contextlib.suppress(FileNotFoundError):
            os.unlink(f"{name}{_CGROUP_RECORD}", dir_fd=self._fd)

    def record_folder(self, folder: Path) -> None:
        """Record a folder of this process outside this one (absolute path): `close` and the
        sweep remove it. Sync: a metadata call.

        Raises:
            OSError: the record could not be made (disk full, folder removed)
        """
        os.symlink(folder, f"{folder.name}{_FOLDER_RECORD}", dir_fd=self._fd)

    def forget_folder(self, name: str) -> None:
        """Remove the record of the folder `name` once the folder is gone. Idempotent."""
        with contextlib.suppress(FileNotFoundError):
            os.unlink(f"{name}{_FOLDER_RECORD}", dir_fd=self._fd)

    def close(self) -> None:
        """End the VM cgroups still recorded (the watcher's too), remove the folders still
        recorded and this one, and release the lock. Idempotent. Blocking (the folder holds
        whole disks): async callers use asyncio.to_thread."""
        self._finalizer()

    def _start_watcher(self) -> None:
        """Start the watcher (_WATCHER), in a cgroup of its own when this host has cgroup v2
        (recorded: close and the sweep remove it). On failure, a warning: close and the
        sweep still end the cgroups."""
        namespace = Path(cgroup.CGROUP_V2_BASE_PATH) / cgroup.CGROUP_APP_NAMESPACE
        own_cgroup = namespace / _WATCHER_TENANT / self.path.name
        joined = ""
        if cgroup.cgroup_v2_available():
            try:
                cgroup.make_cgroup(own_cgroup, exist_ok=False)  # a new one: recorded only once it is this folder's
            except OSError as e:  # no delegated subtree: the watcher runs in this process's cgroup
                logger.debug("Watcher in the cgroup of this process", extra={"error": str(e)})
            else:
                try:
                    os.symlink(own_cgroup, f"{own_cgroup.name}{_CGROUP_RECORD}", dir_fd=self._fd)
                    joined = str(own_cgroup)
                except OSError as e:  # never an unrecorded cgroup: nothing would remove it
                    with contextlib.suppress(OSError):
                        own_cgroup.rmdir()
                    logger.debug("Watcher in the cgroup of this process", extra={"error": str(e)})
        read_fd, write_fd = os.pipe()  # other children never get either end (O_CLOEXEC)
        try:
            # Ends at the EOF of the launcher's stdout: once the launcher has exited and the
            # watcher is armed (or never started). The launcher's exit status tells nothing.
            said = subprocess.run(  # noqa: S603 - fixed argv
                ["/bin/sh", "-c", _START_WATCHER, "sh", joined, _WATCHER, self.path.name, os.path.realpath(namespace)],
                stdin=read_fd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self.path,
                env={**child_env(), "LC_ALL": "C"},  # [A-Za-z] in the script: ASCII letters only
                start_new_session=True,  # no signal to this process group reaches it
                timeout=_WATCHER_START_SECONDS,
                check=False,
            ).stdout
            problem, late = (None, False) if said == b"armed\n" else ("it did not start", False)
        except subprocess.TimeoutExpired as e:  # it may still arm later
            problem, late = str(e), True
        except OSError as e:
            problem, late = str(e), False
        finally:
            os.close(read_fd)
        if problem is None:
            self._watcher_fds.append(write_fd)
            return
        # Its cgroup goes, with the launcher or a watcher not armed in it, and its record:
        # the next record tries again in a new one (one left there would make it fail)
        if joined and not cgroup.Cgroup.end_all([cgroup.Cgroup(own_cgroup)]):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(f"{own_cgroup.name}{_CGROUP_RECORD}", dir_fd=self._fd)
            late = False  # ended with its cgroup
        if late:
            self._watcher_fds.append(write_fd)  # a watcher that arms later never sees a false EOF
        else:
            os.close(write_fd)
        logger.warning(
            "No watcher for the VM cgroups of this process: after a crash, they end at the next start",
            extra={"error": problem},
        )


def _open_root(root: Path) -> int:
    """`root` opened, never through a link, once it is a safe folder of this user
    (private_tree_problem: mode, owner, no ACL entry, parent folders only root and this
    user can change).

    Raises:
        FileNotFoundError: `root` does not exist
        ConfigError: SETTING_INVALID: `root` is not safe, or cannot be checked
        OSError: `root` cannot be opened (no access)
    """
    try:
        fd = os.open(root, _DIR_FLAGS)
    except FileNotFoundError:
        raise
    except OSError as e:  # a link (ELOOP), not a folder (ENOTDIR)
        try:
            problem = private_dir_problem(root, root.lstat(), forbidden=_FORBIDDEN)
        except OSError:
            problem = None
        if problem is None:
            raise
        raise _unsafe_root(root, problem) from e
    try:
        problem = private_tree_problem(root, fd, forbidden=_FORBIDDEN)
    except OSError as e:
        os.close(fd)
        raise _unsafe_root(root, (f"cannot be checked ({e.strerror or e})", "Give this user access to it.")) from e
    if problem is not None:
        os.close(fd)
        raise _unsafe_root(root, problem)
    return fd


def _unsafe_root(root: Path, problem: tuple[str, str]) -> ConfigError:
    return ConfigError(
        f"Folder {root} {problem[0]}. {problem[1]}",
        code=ErrorCode.SETTING_INVALID,
        context={"folder": str(root)},
        hint=None,
        instance_id=None,
    )


def new_folder_name() -> str:
    """A name for PrivateRunDir.open: 64 random bits, so no other process picks it."""
    return f"vmm-{secrets.token_hex(8)}"


def _normalized(target: str) -> Path | None:
    """`target` as a path when it is absolute and normalized (no ".", "..", "//"), else None."""
    path = Path(target)
    return path if path.is_absolute() and str(path) == target and ".." not in path.parts else None


def _vm_cgroup(target: str, record: str) -> bool:
    """Whether `target` of the cgroup record `record` is <namespace>/<tenant>/<name>, as
    the code makes it. A tenant cgroup holds the VMs of other processes, and ".." would
    name the namespace: cgroup.kill on either ends every VM in it."""
    path = _normalized(target)
    return (
        path is not None
        and path.parent.parent == Path(cgroup.CGROUP_V2_BASE_PATH) / cgroup.CGROUP_APP_NAMESPACE
        and _NAME.fullmatch(path.parent.name) is not None
        and _NAME.fullmatch(path.name) is not None
        and record == f"{path.name}{_CGROUP_RECORD}"
    )


def _own_folder(target: str, record: str) -> bool:
    """Whether `target` of the folder record `record` is an absolute normalized path whose
    last name is the record's name ("..": a folder above, never)."""
    path = _normalized(target)
    return path is not None and _NAME.fullmatch(path.name) is not None and record == f"{path.name}{_FOLDER_RECORD}"


@dataclass
class _Ending:
    """A folder whose process is gone (or closes it), with what it records (by record name)."""

    name: str
    fd: int
    cgroups: dict[str, Path]
    folders: dict[str, str]


def _ending(name: str, fd: int) -> _Ending:
    """The folder `name` open at `fd` and its records: only targets as the code makes them."""
    ending = _Ending(name=name, fd=fd, cgroups={}, folders={})
    with os.scandir(fd) as entries:  # a target may be gone: no stat, nothing followed
        records = [entry.name for entry in entries if entry.name.endswith((_CGROUP_RECORD, _FOLDER_RECORD))]
    for record in records:
        try:
            target = os.readlink(record, dir_fd=fd)
        except OSError:
            continue
        if record.endswith(_CGROUP_RECORD) and _vm_cgroup(target, record):
            ending.cgroups[record] = Path(target)
        elif record.endswith(_FOLDER_RECORD) and _own_folder(target, record):
            ending.folders[record] = target
    return ending


def _end(endings: list[_Ending], *, root_fd: int) -> None:
    """End what `endings` record, then remove each folder (in the root open at `root_fd`).
    Every recorded cgroup gets cgroup.kill first, under one deadline (Cgroup.end_all).
    A folder whose cgroup still has a process, or could not be removed, stays whole (its files
    may still be in use; the next sweep tries again).
    A folder whose recorded folder cannot be removed keeps only its lock and the records
    still pending; the next sweep tries them again."""
    cgroups = [path for ending in endings for path in ending.cgroups.values()]
    left: set[Path] = (
        {each.path for each in cgroup.Cgroup.end_all([cgroup.Cgroup(path) for path in cgroups])} if cgroups else set()
    )
    sudo = _Sudo(usable=True)  # one for all the folders of this call
    for ending in endings:
        where = {"folder": ending.name}
        if any(path in left for path in ending.cgroups.values()):
            logger.warning(
                "Kept the folder of a stopped process: a VM cgroup it records still has a process or could not be removed",
                extra=where,
            )
            continue
        pending = [record for record, folder in ending.folders.items() if not _remove_folder(folder, sudo=sudo)]
        if pending:
            _prune(ending.fd, keep={_LOCK_NAME, *pending})
            logger.warning("Kept the records of folders of a stopped process that could not be removed", extra=where)
            continue
        shutil.rmtree(ending.name, dir_fd=root_fd, ignore_errors=True)
        logger.info("Removed the folder of a stopped process", extra=where)


def _prune(fd: int, *, keep: set[str]) -> None:
    """Remove each entry of the folder open at `fd` but those named in `keep`."""
    with os.scandir(fd) as entries:
        names = [(entry.name, entry.is_dir(follow_symlinks=False)) for entry in entries if entry.name not in keep]
    for name, is_folder in names:
        with contextlib.suppress(OSError):
            if is_folder:
                shutil.rmtree(name, dir_fd=fd)
            else:
                os.unlink(name, dir_fd=fd)


@dataclass
class _Sudo:
    """Whether `sudo -n rm` may still run in one _end call: never again once it timed out
    (a stuck sudo, SSSD down, would cost its whole bound for each folder of the sweep)."""

    usable: bool


def _remove_folder(folder: str, *, sudo: _Sudo) -> bool:
    """Remove a recorded folder. True when it is gone, or is no real folder of this user (not
    one this process made: its record can go). rmtree first; what stays of a folder of group
    qemu-vm (files of a qemu-vm QEMU) goes with `sudo -n rm -rf`, as VmWorkingDirectory.cleanup
    removes it, unless a sudo of this call timed out (`sudo`): the record then stays for the
    next sweep. Under `sg qemu-vm` every new folder has group qemu-vm, even one of a QEMU that
    runs as this user: rmtree alone removes those."""
    try:
        st = os.lstat(folder)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
        return True
    shutil.rmtree(folder, ignore_errors=True)
    detail = ""
    if os.path.lexists(folder) and st.st_gid == get_qemu_vm_gid():
        if not sudo.usable:
            return False
        try:
            with sudo_errors_file(folder) as errors:
                subprocess.run(  # noqa: S603 - fixed argv
                    ["sudo", "-n", "rm", "-rf", "--", folder],  # noqa: S607 - sudo from PATH, as sudo_exec
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL if errors.file is None else errors.file,
                    env=child_env(),
                    timeout=constants.SUDO_RM_TIMEOUT_SECONDS,  # on expiry run() kills it: the record stays for the next sweep
                    check=False,
                )
                detail = errors.text()
        except subprocess.TimeoutExpired:
            sudo.usable = False
            logger.warning(
                "sudo rm of a folder of a stopped process took too long: no more sudo in this sweep",
                extra={"folder": folder},
            )
            return False
        except OSError:
            pass
    if os.path.lexists(folder):
        logger.warning("Could not remove a folder of a stopped process", extra={"folder": folder, "stderr": detail})
        return False
    return True


def _release(*, name: str, root_fd: int, fd: int, lock_fd: int, owner_pid: int, watcher_fds: list[int]) -> None:
    # A forked child runs this finalizer too: it closes its copies of the fds (the
    # parent keeps the lock and the watcher's pipe) and never touches the parent's folder.
    try:
        if os.getpid() == owner_pid:
            _end([_ending(name, fd)], root_fd=root_fd)
    finally:
        for each in (lock_fd, fd, root_fd, *watcher_fds):
            os.close(each)  # the watcher's pipe: EOF, the watcher has nothing left to end


def remove_dead_folders(root: Path) -> None:
    """Remove the folders of dead processes under `root`, once the VM cgroups they record
    have no process left, with the folders they record (_end: one deadline for all of
    them); a live process's folder (its lock held, this process's own included) stays.
    Nothing to do when `root` does not exist. Blocking (a listing with no size limit, whole
    disks, waits for killed VMs): async callers use asyncio.to_thread.

    Raises:
        ConfigError: SETTING_INVALID: `root` is not a safe folder of this user (_open_root):
            another user could have put folders and records in it
        OSError: `root` cannot be opened or listed (no access)
    """
    try:
        root_fd = _open_root(root)
    except FileNotFoundError:
        return
    held: list[int] = []  # the dead folders' fds and locks, held until they are removed
    try:
        with os.scandir(root_fd) as entries:
            names = [entry.name for entry in entries if entry.is_dir(follow_symlinks=False)]
        taken = [(name, _take_if_dead(root_fd, name, held=held)) for name in names]
        endings = [_ending(name, fd) for name, fd in taken if fd is not None]
        if endings:
            _end(endings, root_fd=root_fd)
    finally:
        for each in (*held, root_fd):
            os.close(each)


def _take_if_dead(root_fd: int, name: str, *, held: list[int]) -> int | None:
    """The open fd of the folder `name` (in the root open at `root_fd`) once this process holds
    its lock (its owner is gone), else None. Its fd and lock go to `held`. A folder with no
    lock file goes once _LOCKLESS_GRACE_SECONDS old; a lock that is a link is never followed."""
    try:
        fd = os.open(name, _DIR_FLAGS, dir_fd=root_fd)
    except OSError:
        return None
    held.append(fd)
    try:
        lock_fd = os.open(_LOCK_NAME, os.O_RDWR | os.O_NOFOLLOW, dir_fd=fd)
    except FileNotFoundError:
        with contextlib.suppress(OSError):
            if time.time() - os.fstat(fd).st_mtime > _LOCKLESS_GRACE_SECONDS:
                shutil.rmtree(name, dir_fd=root_fd, ignore_errors=True)
        return None
    except OSError:
        return None
    held.append(lock_fd)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None  # held: its process is alive
    return fd

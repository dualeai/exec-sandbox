"""Local instance store: a folder on this host, with the conditional writes of a bucket.

For one host only: the rules of store.py hold across the threads and processes
of this host, never across hosts (a fleet uses the S3 store). open() checks that
flock excludes a second open of a file in this process (Linux NFS makes flock a
POSIX lock, which one process gets twice) and refuses the folder if it does not.

Layout (keys are the bucket's keys, store.py):
    <root>/                    0700, owned by this user, not a symlink, no ACL entry,
                               no parent folder that another user can change
      .lock                    the flock of every name change (never deleted)
      .tmp/<random>.part       new files before their rename; open() removes those older than a day
      <key>                    one object: a 0600 file; each folder 0700

A key segment is 1 to 64 of [a-z0-9._-], first [a-z0-9]: no "..", no hidden
names (the store's own files), lowercase only (APFS ignores case by default).

Names: open() opens the root once (O_DIRECTORY | O_NOFOLLOW: never through a
symlink), checks it and the folders above it, and keeps it open. Each call
starts from that open folder and opens one name at a time with O_NOFOLLOW
(dir_fd): a symlink in the store is refused (ELOOP), never followed, and a
rename of the root or of a folder in it never sends a call to other files. A
missing file is "no such key" only while the root path still names the open
folder (same device and inode); else the store is gone (ConfigError INSTANCE_CONFIG).

ETags. A head's (`get`, conditional writes) is the full SHA-256 hex of its
bytes, hashed at each read: a compare-and-set needs an ETag that never comes
back for other bytes. An upload's and a listing's is the stat stamp of the
object file (inode, size, mtime, ctime): each upload renames a new file onto
the key, and a write in place changes the ctime, so other bytes give another
stamp. A listing reads no file (a stat costs 10 us; an open, up to 1 ms on a
Mac with endpoint security). The device is not in the stamp: it can change at a
remount, and every chain would then look changed.

Writes: the bytes go to a staging file, flushed to the disk (F_FULLFSYNC on
macOS: there fsync leaves them in the drive cache). Then the same thread job
takes the flock, compares, renames the staging file onto the key and flushes
the folder. A cancel never stops a job half way: the call waits for its job,
which stops at its next safe point. Readers take no lock: a rename replaces a
file whole, and an open file keeps its bytes.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import functools
import hashlib
import os
import re
import stat
import time
import weakref
from typing import TYPE_CHECKING, final

from exec_sandbox.aio_utils import run_stoppable
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.files.disk import clone_sparse_fd, flush_to_disk, make_private_dir, pwrite_all, read_all
from exec_sandbox.instance.ids import new_id
from exec_sandbox.instance.store import Applied, InstanceStore, ListedObject, NotApplied, wrong_size
from exec_sandbox.permission_utils import private_dir_problem, private_tree_problem

if TYPE_CHECKING:
    import threading
    from collections.abc import AsyncGenerator, Generator, Sequence
    from pathlib import Path

    from exec_sandbox.instance.store import CondResult

_SEGMENT = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
"""Key segment: 64 at most, the longest instance id (ids.py)."""
_LOCK_NAME = ".lock"
_STAGING_NAME = ".tmp"
_STALE_STAGING_SECONDS = 24 * 60 * 60
"""A staging file this old belongs to a writer that died: open() removes it."""
_CONFIG_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS, errno.ELOOP, errno.ENOTDIR})
"""File errors that the same call gets again: no access, a read-only disk, a symlink in the
store, a file where the store needs a folder."""
_ROOT_DISK_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EIO})
"""Errors at making or opening the root that can pass later: a full disk, a quota, an I/O error."""
_PRIVATE = 0o077
"""Mode bits the root must not have: group and others get no access."""
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

type _Parts = tuple[str, ...]
"""Key segments, checked (`_parts`)."""


class _StoppedError(Exception):
    """The job of a cancelled call stopped at a safe point, before any change."""


@final
class LocalInstanceStore(InstanceStore):
    """One folder on this host. Open with `LocalInstanceStore.open`."""

    def __init__(self, *, root: Path, root_fd: int) -> None:
        self._root = root
        self._root_fd = root_fd
        # Closed with the last reference: a job that runs on after its call was
        # cancelled holds one, so no job ever uses a closed (or reused) fd number.
        weakref.finalize(self, os.close, root_fd)

    @classmethod
    @contextlib.asynccontextmanager
    async def open(cls, *, root: Path) -> AsyncGenerator[LocalInstanceStore]:
        """Make `root` (0700) when it is missing, open and check it (owner, mode, no
        ACL entry, folders above it: permission_utils.private_tree_problem), check
        that flock is exclusive there, and remove staging files older than a day.

        Raises ConfigError (INSTANCE_CONFIG) that names the fix, or TransientError
        (INSTANCE_STORE_UNAVAILABLE) for a disk error (full, I/O).
        """
        yield await asyncio.to_thread(cls._opened, root.absolute())

    @classmethod
    def _opened(cls, root: Path) -> LocalInstanceStore:
        store = cls(root=root, root_fd=_open_root(root))
        store._prepare()
        return store

    # ------------------------------------------------------------------ objects

    async def get(self, key: str, /, *, max_bytes: int) -> tuple[bytes, str] | None:
        return await asyncio.to_thread(functools.partial(self._get, key=key, parts=_parts(key), max_bytes=max_bytes))

    async def size(self, key: str, /) -> int | None:
        return await asyncio.to_thread(functools.partial(self._size, key=key, parts=_parts(key)))

    async def upload(self, key: str, /, *, src: Path) -> str:
        """Clone `src` to a staging file (copy-on-write when the disk can), flush it,
        then rename it onto the key. The ETag is the stat stamp of the file after the rename."""
        return await run_stoppable(functools.partial(self._upload, key=key, parts=_parts(key), src=src), undo=None)

    async def download(self, key: str, /, *, dest: Path, max_bytes: int, expected_size: int | None) -> bool:
        """Clone the object to `dest` (copy-on-write when the disk can). The stat of
        the file is the truth: another size than `expected_size` is
        InstanceGoneError (INSTANCE_INTEGRITY) at once. The copy is of the file open at the start:
        a rename onto the key meanwhile does not change it."""
        return await run_stoppable(
            functools.partial(
                self._download,
                key=key,
                parts=_parts(key),
                dest=dest,
                max_bytes=max_bytes,
                expected_size=expected_size,
            ),
            undo=lambda done: dest.unlink(missing_ok=True) if done else None,
        )

    async def put_if_absent(self, key: str, /, *, data: bytes) -> CondResult:
        return await run_stoppable(
            functools.partial(self._put, key=key, parts=_parts(key), data=data, etag=None), undo=None
        )

    async def put_if_match(self, key: str, /, *, data: bytes, etag: str) -> CondResult:
        """`etag` is the SHA-256 hex that `get` or Applied gave."""
        return await run_stoppable(
            functools.partial(self._put, key=key, parts=_parts(key), data=data, etag=etag), undo=None
        )

    async def list_objects(self, prefix: str, /) -> list[ListedObject]:
        """A walk of the folder of `prefix`, without the lock: an object deleted
        during the walk may show or not."""
        return await asyncio.to_thread(functools.partial(self._list, prefix=prefix, parts=_prefix_parts(prefix)))

    async def list_children(self, prefix: str, /) -> AsyncGenerator[str]:
        """Folders directly under `prefix` that hold an object (S3 shows only those)."""
        for name in await asyncio.to_thread(
            functools.partial(self._children, prefix=prefix, parts=_prefix_parts(prefix))
        ):
            yield name

    async def delete_many(self, keys: Sequence[str], /) -> None:
        """Under the flock: remove each object, then each folder left empty (else
        each probe and deleted instance leaves folders)."""
        items = [(key, _parts(key)) for key in keys]
        if items:
            await run_stoppable(functools.partial(self._delete, items=items), undo=None)

    # ------------------------------------------------------------------ jobs (worker threads)

    def _prepare(self) -> None:
        try:
            problem = private_tree_problem(self._root, self._root_fd, forbidden=_PRIVATE)
        except OSError as e:
            raise ConfigError(
                f"Instance store folder {self._root} cannot be checked: {_reason(e)}.",
                code=ErrorCode.INSTANCE_CONFIG,
                instance_id=None,
                hint="Give this user access to the folder and to each folder above it.",
                context=None,
            ) from e
        if problem:
            raise _unsafe_root(self._root, problem)
        try:
            self._check_flock()
            staging = _make_child(self._root_fd, _STAGING_NAME)
            try:
                _remove_stale_staging(staging)
            finally:
                os.close(staging)
        except OSError as e:
            raise self._failure("OPEN", str(self._root), e) from e

    def _check_flock(self) -> None:
        """Two opens of the lock file must exclude each other (flock(2)). Linux NFS
        makes flock a POSIX lock, which one process gets twice: then writes of
        two threads would not exclude each other."""
        first = os.open(_LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=self._root_fd)
        try:
            fcntl.flock(first, fcntl.LOCK_EX)  # waits while a writer holds it
            second = os.open(_LOCK_NAME, os.O_RDWR | os.O_NOFOLLOW, dir_fd=self._root_fd)
            try:
                fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                shared = True
            except BlockingIOError:
                shared = False
            finally:
                os.close(second)
        finally:
            os.close(first)
        if shared:
            raise ConfigError(
                f"Instance store folder {self._root}: flock does not exclude a second open of a file.",
                code=ErrorCode.INSTANCE_CONFIG,
                instance_id=None,
                hint="Put instance_store_dir on a local disk (not NFS, SMB or FUSE); for many hosts, use the S3 store.",
                context=None,
            )

    def _get(self, *, key: str, parts: _Parts, max_bytes: int) -> tuple[bytes, str] | None:
        try:
            fd = self._open_key(key, parts)
            if fd is None:
                return None
            try:
                size = os.fstat(fd).st_size
                if size > max_bytes:
                    raise wrong_size(key, size, expected=f"at most {max_bytes}")
                data = read_all(fd, size)
            finally:
                os.close(fd)
        except OSError as e:
            raise self._failure("GET", key, e) from e
        return data, hashlib.sha256(data).hexdigest()

    def _size(self, *, key: str, parts: _Parts) -> int | None:
        try:
            with self._folder(parts[:-1]) as folder:
                st = None if folder is None else _lstat_at(folder, parts[-1])
        except OSError as e:
            raise self._failure("STAT", key, e) from e
        if st is None:
            self._check_present()
            return None
        if stat.S_ISDIR(st.st_mode):
            return None  # a folder of other keys
        if not stat.S_ISREG(st.st_mode):
            raise self._not_a_file(key)
        return st.st_size

    def _upload(self, stop: threading.Event, *, key: str, parts: _Parts, src: Path) -> str:
        src_fd = os.open(src, os.O_RDONLY)  # a missing or unreadable source stays an OSError (a local file)
        try:
            with self._staging_file() as (staging, staged):
                clone_sparse_fd(src_fd, staged, dst_dir_fd=staging)
                fd = os.open(staged, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=staging)
                try:
                    os.fchmod(fd, 0o600)  # clonefile copies the mode of the source
                    os.utime(fd)  # and its mtime: open() would take an old one for a dead writer's file
                    flush_to_disk(fd)
                    if stop.is_set():
                        raise _StoppedError
                    with self._locked():
                        if stop.is_set():
                            raise _StoppedError
                        with self._made_folder(parts[:-1]) as folder:
                            os.replace(staged, parts[-1], src_dir_fd=staging, dst_dir_fd=folder)
                            flush_to_disk(folder)
                            etag = _stamp(os.fstat(fd))  # after the rename: it changes the ctime
                finally:
                    os.close(fd)
        except OSError as e:
            raise self._failure("PUT", key, e) from e
        finally:
            os.close(src_fd)
        return etag

    def _download(
        self, _stop: threading.Event, *, key: str, parts: _Parts, dest: Path, max_bytes: int, expected_size: int | None
    ) -> bool:
        """A cancel needs no stop point: the clone runs to its end, then `undo` removes `dest`."""
        try:
            fd = self._open_key(key, parts)
        except OSError as e:
            raise self._failure("GET", key, e) from e
        if fd is None:
            return False
        try:
            held = os.fstat(fd)
            if expected_size is not None and held.st_size != expected_size:
                raise wrong_size(key, held.st_size, expected=str(expected_size))
            if held.st_size > max_bytes:
                raise wrong_size(key, held.st_size, expected=f"at most {max_bytes}")
            # A clone of the open file: an object file never changes in place, so this is one version.
            clone_sparse_fd(fd, dest, dst_dir_fd=None)
            dest.chmod(0o600)
        except FileExistsError:
            raise  # `dest` is another file: keep it
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        finally:
            os.close(fd)
        return True

    def _put(self, stop: threading.Event, *, key: str, parts: _Parts, data: bytes, etag: str | None) -> CondResult:
        """Stage, then under the flock: compare, rename, flush the folder. It raises
        only before the rename (nothing applied). A failed folder flush after the
        rename gives NotApplied(maybe_applied=True): a crash may undo the rename."""
        try:
            with self._staging_file() as (staging, staged):
                _write_new(staging, staged, data)
                if stop.is_set():
                    raise _StoppedError
                with self._locked():
                    if stop.is_set():
                        raise _StoppedError
                    if self._current_etag(key, parts) != etag:  # None for create-only: the key must be absent
                        return NotApplied(maybe_applied=False)
                    with self._made_folder(parts[:-1]) as folder:
                        os.replace(staged, parts[-1], src_dir_fd=staging, dst_dir_fd=folder)
                        try:
                            flush_to_disk(folder)
                        except OSError:
                            return NotApplied(maybe_applied=True)  # renamed, but a crash may undo it
        except OSError as e:
            raise self._failure("PUT", key, e) from e
        return Applied(etag=hashlib.sha256(data).hexdigest())

    def _list(self, *, prefix: str, parts: _Parts) -> list[ListedObject]:
        self._check_present()
        objects: list[ListedObject] = []
        try:
            with self._folder(parts) as top:
                if top is None:
                    return []
                for folder, segments, names in _walk(top, parts):
                    for name in names:
                        st = _lstat_at(folder, name)
                        if st is not None and stat.S_ISREG(st.st_mode):  # else gone or replaced meanwhile
                            objects.append(
                                ListedObject(key="/".join((*segments, name)), size=st.st_size, etag=_stamp(st))
                            )
        except OSError as e:
            raise self._failure("LIST", prefix, e) from e
        return sorted(objects, key=lambda o: o.key)

    def _children(self, *, prefix: str, parts: _Parts) -> list[str]:
        self._check_present()
        try:
            with self._folder(parts) as folder:
                if folder is None:
                    return []
                with os.scandir(folder) as entries:
                    names = [e.name for e in entries if _SEGMENT.fullmatch(e.name) and e.is_dir(follow_symlinks=False)]
                return sorted(name for name in names if _holds_object(folder, name))
        except OSError as e:
            raise self._failure("LIST", prefix, e) from e

    def _delete(self, stop: threading.Event, *, items: list[tuple[str, _Parts]]) -> None:
        folders: dict[_Parts, dict[str, str]] = {}  # folder: {object name: key}
        for key, parts in items:
            folders.setdefault(parts[:-1], {})[parts[-1]] = key
        key = items[0][0]
        try:
            with self._locked():
                if stop.is_set():
                    raise _StoppedError
                for segments in sorted(folders, key=len, reverse=True):  # deepest first: pruned before their parents
                    with self._chain(segments, make=False) as chain:
                        if len(chain) > len(segments):  # the folder exists
                            for name, key in folders[segments].items():  # noqa: B007  # `key` names the failed one
                                _remove_object(chain[-1], name)
                        _prune(chain, segments)
        except OSError as e:
            raise self._failure("DELETE", key, e) from e

    # ------------------------------------------------------------------ helpers (worker threads)

    @contextlib.contextmanager
    def _chain(self, parts: _Parts, *, make: bool) -> Generator[list[int]]:
        """New fds on the root and on each folder of `parts` in turn, each opened in
        the last one (a symlink is ELOOP); all closed at exit. `make`: make missing
        folders (0700), each flushed into its parent (under the flock). Else the
        chain stops before the first missing folder."""
        chain = [_open_dir(".", dir_fd=self._root_fd)]  # not the root fd itself: a scan moves the offset of its fd
        try:
            for part in parts:
                child = _make_child(chain[-1], part) if make else _open_child(chain[-1], part)
                if child is None:
                    break
                chain.append(child)
            yield chain
        finally:
            for fd in chain:
                os.close(fd)

    @contextlib.contextmanager
    def _folder(self, parts: _Parts) -> Generator[int | None]:
        """A new fd on the folder `parts`, or None when it is missing."""
        with self._chain(parts, make=False) as chain:
            yield chain[-1] if len(chain) > len(parts) else None

    @contextlib.contextmanager
    def _made_folder(self, parts: _Parts) -> Generator[int]:
        """A new fd on the folder `parts`, made with its missing parents. Under the flock."""
        with self._chain(parts, make=True) as chain:
            yield chain[-1]

    @contextlib.contextmanager
    def _staging_file(self) -> Generator[tuple[int, str]]:
        """The staging folder (open) and a new name in it. A file of that name is removed at exit (after a rename, there is none)."""
        staging = _make_child(self._root_fd, _STAGING_NAME)
        name = f"{new_id()}.part"
        try:
            yield staging, name
        finally:
            with contextlib.suppress(OSError):
                os.unlink(name, dir_fd=staging)
            os.close(staging)

    @contextlib.contextmanager
    def _locked(self) -> Generator[None]:
        """Hold the flock of the store (this thread waits for it)."""
        fd = os.open(_LOCK_NAME, os.O_RDWR | os.O_NOFOLLOW, dir_fd=self._root_fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)  # frees the flock; a close error must not hide a rename that applied

    def _current_etag(self, key: str, parts: _Parts) -> str | None:
        fd = self._open_key(key, parts)
        if fd is None:
            return None
        with os.fdopen(fd, "rb") as f:  # closes the fd
            return hashlib.file_digest(f, "sha256").hexdigest()

    def _open_key(self, key: str, parts: _Parts) -> int | None:
        """A read fd on the object file of `key`, or None when there is no object."""
        with self._folder(parts[:-1]) as folder:  # a file opened in it stays open after it closes
            try:
                # O_NONBLOCK: a FIFO in the store must not block the open.
                fd = (
                    None
                    if folder is None
                    else os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=folder)
                )
            except (FileNotFoundError, NotADirectoryError):
                fd = None
        if fd is None:
            self._check_present()
            return None
        try:
            mode = os.fstat(fd).st_mode
        except BaseException:
            os.close(fd)
            raise
        if stat.S_ISREG(mode):
            return fd
        os.close(fd)
        if stat.S_ISDIR(mode):
            return None  # a folder of other keys
        raise self._not_a_file(key)

    # ------------------------------------------------------------------ errors

    def _root_present(self) -> bool:
        """True while the path of the root names the folder this store holds open."""
        try:
            at_path = self._root.lstat()
            held = os.fstat(self._root_fd)
        except OSError:
            return False
        return (at_path.st_dev, at_path.st_ino) == (held.st_dev, held.st_ino)

    def _check_present(self) -> None:
        """A missing file is "no such key" only while the root is there (S3: NoSuchBucket)."""
        if not self._root_present():
            raise self._gone()

    def _gone(self) -> ConfigError:
        return ConfigError(
            f"Instance store folder {self._root} is gone, or another file is in its place.",
            code=ErrorCode.INSTANCE_CONFIG,
            instance_id=None,
            hint="Make the folder again or set instance_store_dir, then start the worker again.",
            context=None,
        )

    def _not_a_file(self, key: str) -> ConfigError:
        return ConfigError(
            f"Instance store file {self._root}/{key} is not a regular file.",
            code=ErrorCode.INSTANCE_CONFIG,
            instance_id=None,
            hint=f"Remove it: {self._root} holds only the files and folders of the store.",
            context=None,
        )

    def _failure(self, operation: str, key: str, error: OSError) -> ConfigError | TransientError:
        """A file error on the store side. A missing root, no access, a symlink, a file
        where a folder must be or a read-only disk fail the same way again (FIX_CONFIG);
        a full disk or an I/O error can pass later."""
        if not self._root_present():
            return self._gone()
        if error.errno in _CONFIG_ERRNOS:
            return ConfigError(
                f"Instance store {operation} {key} failed: {_reason(error)}.",
                code=ErrorCode.INSTANCE_CONFIG,
                instance_id=None,
                hint=f"Give this user read and write access to {self._root}, on a writable local disk; "
                "remove any symlink or stray file in it.",
                context=None,
            )
        return TransientError(
            f"Instance store {operation} {key} failed: {_reason(error)}.",
            code=ErrorCode.INSTANCE_STORE_UNAVAILABLE,
            instance_id=None,
            hint=f"The disk of {self._root} failed or is full. The same call can succeed later.",
            context=None,
        )


# ---------------------------------------------------------------------- root


def _open_root(root: Path) -> int:
    """Make `root` (0700) when it is missing, and open it. A symlink is refused, never followed."""
    try:
        make_private_dir(root)
    except OSError as e:
        raise _cannot_open(root, e) from e
    try:
        return os.open(root, _DIR_FLAGS)
    except OSError as e:
        try:
            problem = private_dir_problem(root, root.lstat(), forbidden=_PRIVATE)
        except OSError:
            problem = None
        if problem is None:
            raise _cannot_open(root, e) from e
        raise _unsafe_root(root, problem) from None


def _cannot_open(root: Path, error: OSError) -> ConfigError | TransientError:
    """A full disk, a quota or an I/O error can pass later; any other error (no
    access, not a folder, a read-only disk) fails the same way again."""
    if error.errno in _ROOT_DISK_ERRNOS:
        return TransientError(
            f"Instance store folder {root} cannot be made or read: {_reason(error)}.",
            code=ErrorCode.INSTANCE_STORE_UNAVAILABLE,
            instance_id=None,
            hint=f"The disk of {root} failed or is full. The same call can succeed later.",
            context=None,
        )
    return ConfigError(
        f"Instance store folder {root} cannot be made or read: {_reason(error)}.",
        code=ErrorCode.INSTANCE_CONFIG,
        instance_id=None,
        hint="Set instance_store_dir to a folder this user can make and write.",
        context=None,
    )


def _unsafe_root(root: Path, problem: tuple[str, str]) -> ConfigError:
    """The root holds every saved instance: a real folder of this user that nobody else can read or move."""
    return ConfigError(
        f"Instance store folder {root} {problem[0]}.",
        code=ErrorCode.INSTANCE_CONFIG,
        instance_id=None,
        hint=problem[1],
        context=None,
    )


def _remove_stale_staging(staging: int) -> None:
    """Staging files of writers that died (a live writer renames its file in its call)."""
    cutoff = time.time() - _STALE_STAGING_SECONDS
    with os.scandir(staging) as entries:
        for entry in entries:
            with contextlib.suppress(FileNotFoundError):
                if not entry.is_dir(follow_symlinks=False) and entry.stat(follow_symlinks=False).st_mtime < cutoff:
                    os.unlink(entry.name, dir_fd=staging)


# ---------------------------------------------------------------------- names in open folders


def _open_dir(name: str, *, dir_fd: int) -> int:
    return os.open(name, _DIR_FLAGS, dir_fd=dir_fd)


def _open_child(folder: int, name: str) -> int | None:
    """A new fd on the folder `name` in `folder`, or None when it is missing or a file.
    A symlink is ELOOP: O_DIRECTORY | O_NOFOLLOW gives ENOTDIR for it (Linux, macOS)."""
    try:
        return _open_dir(name, dir_fd=folder)
    except FileNotFoundError:
        return None
    except NotADirectoryError:
        if stat.S_ISLNK(os.stat(name, dir_fd=folder, follow_symlinks=False).st_mode):
            raise OSError(errno.ELOOP, os.strerror(errno.ELOOP), name) from None
        return None


def _make_child(folder: int, name: str) -> int:
    """`_open_child`, but a missing folder is made (0700) and flushed into `folder`."""
    child = _open_child(folder, name)
    if child is None:
        with contextlib.suppress(FileExistsError):  # made meanwhile (the staging folder), or a file: refused below
            os.mkdir(name, 0o700, dir_fd=folder)
            flush_to_disk(folder)
        child = _open_child(folder, name)
        if child is None:
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), name)
    return child


def _lstat_at(folder: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=folder, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _walk(folder: int, parts: _Parts) -> Generator[tuple[int, _Parts, list[str]]]:
    """Each folder under the open `folder` (key segments `parts`), depth first: its
    fd, its segments and its object files. A folder that went meanwhile has none."""
    with os.scandir(folder) as scan:
        entries = [(e.name, e.is_dir(follow_symlinks=False), e.is_file(follow_symlinks=False)) for e in scan]
    files = [name for name, _, is_file in entries if is_file and _SEGMENT.fullmatch(name)]
    yield folder, parts, files
    for name, is_dir, _ in entries:
        if is_dir and _SEGMENT.fullmatch(name):
            child = _open_child(folder, name)
            if child is None:
                continue
            try:
                yield from _walk(child, (*parts, name))
            finally:
                os.close(child)


def _holds_object(folder: int, name: str) -> bool:
    """True when the folder `name` in `folder` holds an object, at any depth."""
    child = _open_child(folder, name)
    if child is None:
        return False
    try:
        with contextlib.closing(_walk(child, ())) as walk:
            return any(files for _, _, files in walk)
    finally:
        os.close(child)


def _remove_object(folder: int, name: str) -> None:
    st = _lstat_at(folder, name)
    if st is None or stat.S_ISDIR(st.st_mode):
        return  # gone, or a folder of other keys, not an object
    os.unlink(name, dir_fd=folder)


def _prune(chain: list[int], parts: _Parts) -> None:
    """Remove the deepest folder of `chain` (the open folders of `parts`) and each
    parent left empty (never the root), then flush the first folder that stays:
    the removals reach the disk."""
    depth = len(chain) - 1
    while depth > 0:
        try:
            os.rmdir(parts[depth - 1], dir_fd=chain[depth - 1])
        except OSError as e:
            if e.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            break
        depth -= 1
    flush_to_disk(chain[depth])


# ---------------------------------------------------------------------- files


def _write_new(folder: int, name: str, data: bytes) -> None:
    """A new file `name` in `folder` that holds `data`, flushed to the disk."""
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=folder)
    try:
        pwrite_all(fd, data, 0)
        flush_to_disk(fd)
    finally:
        os.close(fd)


def _stamp(st: os.stat_result) -> str:
    """The ETag of an upload and of a listing (see the module docstring)."""
    return f"{st.st_ino}-{st.st_size}-{st.st_mtime_ns}-{st.st_ctime_ns}"


def _reason(error: OSError) -> str:
    name = errno.errorcode.get(error.errno or 0, type(error).__name__)
    return f"{name} ({error.strerror})" if error.strerror else name


# ---------------------------------------------------------------------- keys


def _parts(key: str) -> _Parts:
    """The segments of `key`. Refuses a key that could leave the root or reach the store's own files."""
    parts = tuple(key.split("/"))
    if not all(_SEGMENT.fullmatch(part) for part in parts):
        raise _bad_key(key)
    return parts


def _prefix_parts(prefix: str) -> _Parts:
    if prefix == "":
        return ()
    if not prefix.endswith("/"):
        raise _bad_key(prefix)
    return _parts(prefix.removesuffix("/"))


def _bad_key(key: str) -> ConfigError:
    return ConfigError(
        f"Instance store key {key[:200]!r} is not a key of the local store.",
        code=ErrorCode.INSTANCE_CONFIG,
        instance_id=None,
        hint="A key is '/'-separated segments of [a-z0-9][a-z0-9._-]{0,63}, made by the store.py helpers.",
        context=None,
    )

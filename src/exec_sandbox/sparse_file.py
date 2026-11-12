"""Sparse disk files: create and clone without allocating their holes, and find their data.

A VM's home disk is a raw sparse file the size of the disk; only the blocks
the guest wrote take host space. A clone must keep it that way: a plain copy
would write every hole as zeros and allocate the whole disk. A reader that
needs only the data (a clone, a saved instance's body) walks `data_ranges`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import fcntl
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import BinaryIO

_FICLONE = 0x40049409  # Linux ioctl: reflink the whole file (btrfs, XFS, bcachefs)
_AT_FDCWD = -2  # macOS <sys/fcntl.h>: a path relative to the current folder
_COPY_CHUNK = 8 * 1024 * 1024


def create_sparse(path: Path, size_bytes: int) -> None:
    """Create a private (0600) file of `size_bytes` that allocates no blocks (reads as zeros)."""
    with _create_private(path, dir_fd=None) as f:
        f.truncate(size_bytes)


def clone_sparse(src: Path, dst: Path) -> None:
    """Copy `src` to a new file `dst`, sharing blocks when the filesystem can (see clone_sparse_fd)."""
    fd = os.open(src, os.O_RDONLY)
    try:
        clone_sparse_fd(fd, dst, dst_dir_fd=None)
    finally:
        os.close(fd)


def clone_sparse_fd(src_fd: int, dst: str | Path, *, dst_dir_fd: int | None) -> None:
    """Copy the file open at `src_fd` to a new file `dst`, sharing blocks when the filesystem can.

    `dst` is a name in the folder open at `dst_dir_fd`, or a path when None. The
    copy is of the open file, whatever its path names meanwhile. Order:
    copy-on-write clone (APFS fclonefileat, Linux FICLONE), else a copy of the
    data ranges only (SEEK_DATA/SEEK_HOLE), so holes stay holes.

    The APFS clone copies the mode and the other attributes of the source
    (clonefile(2)): set them after, if needed. The other ways make a new file of
    mode 0600 at most. Raises FileExistsError when `dst` exists (also a symlink).
    """
    if _clonefile(src_fd, dst, dst_dir_fd=dst_dir_fd):
        return
    with _create_private(dst, dir_fd=dst_dir_fd) as fout:
        if _ficlone(src_fd, fout.fileno()):
            return
        size = os.fstat(src_fd).st_size
        _copy_data_ranges(src_fd, fout.fileno(), size)
        fout.truncate(size)


def _clonefile(src_fd: int, dst: str | Path, *, dst_dir_fd: int | None) -> bool:
    """macOS fclonefileat(2): instant copy-on-write clone on APFS."""
    if not hasattr(os, "uname") or os.uname().sysname != "Darwin":
        return False
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    at = _AT_FDCWD if dst_dir_fd is None else dst_dir_fd
    if libc.fclonefileat(src_fd, at, os.fsencode(dst), 0) == 0:
        return True
    err = ctypes.get_errno()
    if err in (errno.ENOTSUP, errno.EXDEV):
        return False  # not APFS, or another volume: fall back to a sparse copy
    raise OSError(err, os.strerror(err), str(dst))


def _ficlone(src_fd: int, dst_fd: int) -> bool:
    """Linux FICLONE: copy-on-write clone where the filesystem supports reflinks."""
    try:
        fcntl.ioctl(dst_fd, _FICLONE, src_fd)
    except OSError as e:
        if e.errno in (errno.EOPNOTSUPP, errno.EXDEV, errno.EINVAL, errno.ENOTTY):
            return False
        raise
    return True


def data_ranges(fd: int, size: int, *, grain: int) -> Iterator[tuple[int, int]]:
    """(start, end) of each range of the file open at `fd` that holds data, in
    order, below `size`: holes are skipped (SEEK_DATA/SEEK_HOLE), never read.

    Each range is widened to multiples of `grain` (`end` at most `size`), and the
    search goes on at its end: a reader of `grain`-byte blocks asks the
    filesystem once per run of blocks, not once per smaller extent (ext4 and XFS
    report 4 KiB extents). `grain` 1: the exact ranges. OSError (EINVAL) on a
    filesystem without SEEK_DATA: each caller decides."""
    offset = 0
    while offset < size:
        try:
            start = os.lseek(fd, offset, os.SEEK_DATA)
        except OSError as e:
            if e.errno == errno.ENXIO:  # no data after offset: the rest is a hole
                return
            raise
        if start >= size:  # the file grew since `size` was read
            return
        end = min(-(-os.lseek(fd, start, os.SEEK_HOLE) // grain) * grain, size)
        yield start // grain * grain, end
        offset = end


def flush_to_disk(fd: int) -> None:
    """Write a file or folder through to the disk. macOS fsync leaves the data in
    the drive cache (man 2 fsync): F_FULLFSYNC does not. On a folder, it flushes
    its names (new, renamed, removed): Linux needs it after a rename (fsync(2))."""
    if sys.platform == "darwin":
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    else:
        os.fsync(fd)


def pwrite_all(fd: int, data: bytes, offset: int) -> None:
    """Write all of `data` at `offset` (os.pwrite can write less than asked)."""
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        view, offset = view[written:], offset + written


def _copy_data_ranges(src_fd: int, dst_fd: int, size: int) -> None:
    """Copy only the allocated ranges of a sparse file (exact: a hole is never
    written as zeros). Without SEEK_DATA the OSError stays: a copy that writes the
    holes would allocate the whole disk."""
    for start, end in data_ranges(src_fd, size, grain=1):
        offset = start
        while offset < end:
            chunk = os.pread(src_fd, min(_COPY_CHUNK, end - offset), offset)
            if not chunk:
                return
            pwrite_all(dst_fd, chunk, offset)
            offset += len(chunk)


def _create_private(path: str | Path, *, dir_fd: int | None) -> BinaryIO:
    """A new file, mode 0600 at most (the umask can only remove bits); fails if `path` exists
    (O_EXCL: also when it is a symlink). `path` is relative to the folder open at `dir_fd` when given."""
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=dir_fd), "wb")

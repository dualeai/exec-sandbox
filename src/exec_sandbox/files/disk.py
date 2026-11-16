"""Disk files: create, clone and copy sparse files without allocating their holes, find their
data, flush and size them, stamp a version, ask the filesystem that holds a path, and ask the
kernel to prefetch a file.

A VM's home disk is a raw sparse file the size of the disk; only the blocks
the guest wrote take host space. A clone must keep it that way: a plain copy
would write every hole as zeros and allocate the whole disk. A reader that
needs only the data (a clone, a saved instance's body) walks `data_ranges`.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

from exec_sandbox.platform_utils import HostOS, detect_host_os

if TYPE_CHECKING:
    import threading
    from collections.abc import Iterator
    from typing import BinaryIO

_FICLONE = 0x40049409  # Linux ioctl: reflink the whole file (btrfs, XFS, bcachefs)
_AT_FDCWD = -2  # macOS <sys/fcntl.h>: a path relative to the current folder
_COPY_CHUNK = 8 * 1024 * 1024
_STAT_BLOCK_BYTES = 512  # the unit of st_blocks on Linux and macOS (POSIX leaves it open)
_libc = ctypes.CDLL(None, use_errno=True)
"""The C library of this process, loaded once: fclonefileat and F_RDADVISE (macOS), statfs (Linux)."""


def allocated_bytes(st: os.stat_result) -> int:
    """The bytes a file takes on its filesystem: its allocated blocks (st_blocks), not its
    size; a sparse file takes less. `st` from any stat call (Path.stat, os.lstat, a directory
    entry's stat), so callers choose how links are followed."""
    return st.st_blocks * _STAT_BLOCK_BYTES


def make_dirs(path: Path, *, mode: int, exist_ok: bool) -> None:
    """Make `path` and each missing folder above it with `mode`. Path.mkdir(parents=True) gives
    the parents the umask's mode instead (0775 under umask 002, the Ubuntu default), which the
    private-folder checks refuse. mkdir's mode goes through the umask, which can only remove
    bits. Folders that exist keep their mode. Blocking: a few mkdir calls.

    Raises:
        FileExistsError: `path` exists and not `exist_ok`, or it is a file
        OSError: a folder cannot be made (a file in the way, no access)
    """
    missing: list[Path] = []
    for folder in path.parents:
        if folder.exists():
            break
        missing.append(folder)
    for folder in reversed(missing):
        with contextlib.suppress(FileExistsError):  # another process made it meanwhile
            folder.mkdir(mode=mode)
    path.mkdir(mode=mode, exist_ok=exist_ok)


def make_private_dir(path: Path) -> None:
    """make_dirs with mode 0700 (the XDG rule for data folders); `path` may exist."""
    make_dirs(path, mode=0o700, exist_ok=True)


def create_sparse(path: Path, size_bytes: int) -> None:
    """Create a private (0600) file of `size_bytes` that allocates no blocks (reads as zeros)."""
    with create_file(path, mode=0o600, dir_fd=None) as f:
        f.truncate(size_bytes)


def create_file(path: str | Path, *, mode: int, dir_fd: int | None) -> BinaryIO:
    """A new file open for writing, with exactly `mode`, whatever the umask.

    open() makes a file 0666 minus the umask: 0664 under umask 002 (Ubuntu, Fedora), and
    under `sg qemu-vm` its group is qemu-vm, so a QEMU that runs as qemu-vm can write it.
    Under umask 077 a 0644 file would be 0600, which QEMU as qemu-vm cannot read (for the
    kernel, only the mode bits give qemu-vm access). Give a mode with no group or other
    write: 0644 for a file that QEMU as qemu-vm reads (kernel, base image, L2 layer), 0600
    for one that only this user reads (a lock, a digest, metadata).

    Fails if `path` exists (O_EXCL: also when it is a symlink). `path` is relative to the
    folder open at `dir_fd` when given."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode, dir_fd=dir_fd)
    try:
        os.fchmod(fd, mode)  # the umask removed bits from the create mode
        return os.fdopen(fd, "wb")
    except BaseException:
        os.close(fd)
        raise


def clone_sparse(src: Path, dst: Path, *, stop: threading.Event | None) -> bool:
    """Copy `src` to a new file `dst`, sharing blocks when the filesystem can (see clone_sparse_fd)."""
    fd = os.open(src, os.O_RDONLY)
    try:
        return clone_sparse_fd(fd, dst, dst_dir_fd=None, stop=stop)
    finally:
        os.close(fd)


def clone_sparse_fd(src_fd: int, dst: str | Path, *, dst_dir_fd: int | None, stop: threading.Event | None) -> bool:
    """Copy the file open at `src_fd` to a new file `dst`, sharing blocks when the filesystem can.

    `dst` is a name in the folder open at `dst_dir_fd`, or a path when None. The
    copy is of the open file, whatever its path names meanwhile. Order:
    copy-on-write clone (APFS fclonefileat, Linux FICLONE), else a copy of the
    data ranges only (SEEK_DATA/SEEK_HOLE), so holes stay holes.

    The APFS clone copies the mode and the other attributes of the source
    (clonefile(2)): set them after, if needed. The other ways make a new file of
    mode 0600. Raises FileExistsError when `dst` exists (also a symlink).
    Returns True for a copy-on-write clone (the blocks stay shared until written),
    False for a copy of the data (its blocks take free space at once). A copy of the data
    checks `stop` before each chunk: once it is set, `dst` is removed and InterruptedError
    raised (aio_utils.run_stoppable; a clone is one call and does not stop). Another
    OSError (no SEEK_DATA: data_ranges; a full disk) leaves `dst`: the caller removes it.
    """
    if _clonefile(src_fd, dst, dst_dir_fd=dst_dir_fd):
        return True
    with create_file(dst, mode=0o600, dir_fd=dst_dir_fd) as fout:
        if _ficlone(src_fd, fout.fileno()):
            return True
        size = os.fstat(src_fd).st_size
        try:
            _copy_data_ranges(src_fd, fout.fileno(), size, stop=stop)
        except InterruptedError:
            os.unlink(dst, dir_fd=dst_dir_fd)
            raise
        fout.truncate(size)
    return False


def _clonefile(src_fd: int, dst: str | Path, *, dst_dir_fd: int | None) -> bool:
    """macOS fclonefileat(2): instant copy-on-write clone on APFS."""
    if sys.platform != "darwin":
        return False
    at = _AT_FDCWD if dst_dir_fd is None else dst_dir_fd
    if _libc.fclonefileat(src_fd, at, os.fsencode(dst), 0) == 0:
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


def _copy_data_ranges(src_fd: int, dst_fd: int, size: int, *, stop: threading.Event | None) -> None:
    """Copy only the allocated ranges of a sparse file (exact: a hole is never
    written as zeros). Without SEEK_DATA the OSError stays: a copy that writes the
    holes would allocate the whole disk. InterruptedError once `stop` is set."""
    for start, end in data_ranges(src_fd, size, grain=1):
        offset = start
        while offset < end:
            if stop is not None and stop.is_set():
                raise InterruptedError(errno.EINTR, "The copy was stopped")
            chunk = os.pread(src_fd, min(_COPY_CHUNK, end - offset), offset)
            if not chunk:
                return
            pwrite_all(dst_fd, chunk, offset)
            offset += len(chunk)


def file_stamp(stat: os.stat_result) -> str:
    """One version of a file for the in-process caches: inode, mtime and size. A
    rebuilt or replaced file gets a new stamp."""
    return f"{stat.st_ino}:{stat.st_mtime_ns}:{stat.st_size}"


def read_all(fd: int, size: int) -> bytes:
    """Up to `size` bytes from the position of `fd`: fewer only at its end (os.read returns less)."""
    chunks: list[bytes] = []
    while size > 0 and (chunk := os.read(fd, size)):
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


LINUX_MEMORY_FILESYSTEMS: Final = {0x01021994: "tmpfs", 0x858458F6: "ramfs"}
"""statfs f_type of the memory-backed file systems (linux/magic.h TMPFS_MAGIC, RAMFS_MAGIC).
A list, because the kernel gives no other sign: statfs has no "memory" flag, and no backing
device is true of btrfs, overlay, NFS and FUSE too. These are the two filesystems of shared
memory (shmem) since Linux 2.4; glibc and systemd tell /dev/shm by the same two numbers. A
filesystem on a RAM block device (zram, brd) is a disk filesystem to the kernel."""


def statfs_type(path: Path) -> int | None:
    """The f_type of the filesystem that holds `path` (one statfs call: ``struct statfs`` starts
    with ``long f_type``), or None when statfs fails. Linux only."""
    buffer = ctypes.create_string_buffer(4096)  # larger than struct statfs
    if _libc.statfs(os.fsencode(path), buffer) != 0:
        return None
    return int.from_bytes(buffer.raw[:8], sys.byteorder) & 0xFFFFFFFF


OVERLAYFS_SUPER_MAGIC: Final = 0x794C7630
"""statfs f_type of an overlay (linux/magic.h): the type of the filesystem under it is that
of its upper folder (overlay_upper)."""
MOUNTINFO: Final = Path("/proc/self/mountinfo")


def overlay_upper(path: Path) -> Path | None:
    """The upper folder of the overlay that holds `path` (the upperdir= option of its line in
    /proc/self/mountinfo, found by the device of `path`), or None: no such line, a path that is
    not absolute, or no such folder (in a container it names a folder of the host, or one this
    user may not see). The file is read as bytes (a mount point can be any bytes); its fields
    are split on single spaces (a mount source can be empty); octal escapes are undone, and the
    option values have their commas escaped so. Blocking."""
    try:
        dev = os.stat(path).st_dev  # noqa: PTH116 - the device numbers, as mountinfo writes them
        lines = MOUNTINFO.read_bytes().split(b"\n")
    except OSError:
        return None
    for line in lines:
        fields, _, rest = line.partition(b" - ")
        try:  # <id> <parent> <major:minor> <root> <mount point> ... - <type> <source> <options>
            device, fs_type, options = fields.split(b" ")[2], *rest.split(b" ")[0:3:2]
        except (IndexError, ValueError):
            continue
        if fs_type != b"overlay" or device != f"{os.major(dev)}:{os.minor(dev)}".encode():
            continue
        for option in options.split(b","):
            if option.startswith(b"upperdir="):
                raw = re.sub(rb"\\([0-7]{3})", lambda m: bytes([int(m[1], 8)]), option[len(b"upperdir=") :])
                upper = os.fsdecode(raw)
                return Path(upper) if os.path.isabs(upper) and os.path.isdir(upper) else None  # noqa: PTH112, PTH117
    return None


def memory_filesystem(path: Path) -> str | None:
    """The memory-backed file system (tmpfs, ramfs) that holds `path` or, for a path not made
    yet, its nearest existing folder; None for a disk, or off Linux. Files there are memory
    charged to the process that writes them: for QEMU, to its VM's memory cgroup, with no way
    out but swap. statfs answers for the mount the path really is on (stacked mounts, bind
    mounts, escaped mount point names)."""
    if sys.platform != "linux":
        return None
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    fs_type = statfs_type(existing)
    return None if fs_type is None else LINUX_MEMORY_FILESYSTEMS.get(fs_type)


_F_RDADVISE = 44  # from darwin-xnu/bsd/sys/fcntl.h


class _Radvisory(ctypes.Structure):
    """macOS struct radvisory { off_t ra_offset; int ra_count; }."""

    _fields_ = [
        ("ra_offset", ctypes.c_int64),  # off_t
        ("ra_count", ctypes.c_int),  # int
    ]


def _rd_advise(fd: int, size: int) -> int:
    """fcntl(F_RDADVISE) of the first `size` bytes of `fd` (macOS): 0, or -1 with errno set.
    fcntl's third argument is variadic: on Apple Silicon it goes on the stack, which ctypes
    does only for arguments after its argtypes (without them it fails with EFAULT)."""
    fcntl = _libc.fcntl
    fcntl.argtypes = [ctypes.c_int, ctypes.c_int]
    fcntl.restype = ctypes.c_int
    return fcntl(fd, _F_RDADVISE, ctypes.byref(_Radvisory(ra_offset=0, ra_count=min(size, 2**31 - 1))))


def advise_willneed(path: Path) -> None:
    """Ask the kernel to prefetch *path* into the page cache.

    Non-blocking, zero-copy.  The kernel schedules background I/O and the call
    returns near-instantly; actual disk reads happen asynchronously.

    Best-effort: silently ignored on unsupported platforms or on failure (the
    hint is advisory, not required — QEMU then reads from disk).

    macOS: fcntl(F_RDADVISE = 44, struct radvisory) through libc, as Python's fcntl
    module does not expose it (cpython#113092); darwin-xnu bsd/sys/fcntl.h defines
    both, and PostgreSQL uses the same call ("Add prefetching support on macOS",
    2024-08-27). Linux: posix_fadvise(POSIX_FADV_WILLNEED).
    """
    fd = -1
    try:
        fd = os.open(str(path), os.O_RDONLY)
        size = os.fstat(fd).st_size
        host = detect_host_os()
        if host is HostOS.MACOS:
            _rd_advise(fd, size)
        elif host is HostOS.LINUX:
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_WILLNEED)  # type: ignore[attr-defined]
    except OSError:
        pass  # Advisory — silently degrade
    finally:
        if fd >= 0:
            os.close(fd)

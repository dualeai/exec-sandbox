"""files.disk: sparse home disk files created and cloned without allocating their holes, new
files of an exact mode, allocated sizes, memory filesystems, and private folders."""

import errno
import os
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from exec_sandbox.files import disk as files_disk
from exec_sandbox.files.disk import (
    allocated_bytes,
    clone_sparse,
    create_file,
    create_sparse,
    make_private_dir,
    memory_filesystem,
    statfs,
)
from tests.conftest import skip_unless_linux

_MIB = 1024 * 1024


def _allocated(path: Path) -> int:
    return path.stat().st_blocks * 512


class TestCreateSparse:
    def test_full_size_no_allocation(self, tmp_path: Path) -> None:
        disk = tmp_path / "home.img"
        create_sparse(disk, 64 * _MIB)
        assert disk.stat().st_size == 64 * _MIB
        assert _allocated(disk) < _MIB

    def test_refuses_existing_file(self, tmp_path: Path) -> None:
        disk = tmp_path / "home.img"
        disk.write_bytes(b"in use")
        with pytest.raises(FileExistsError):
            create_sparse(disk, _MIB)

    def test_private_whatever_the_umask(self, tmp_path: Path) -> None:
        """Tenant data: no other local account may read a home disk."""
        old = os.umask(0)
        try:
            create_sparse(tmp_path / "home.img", _MIB)
        finally:
            os.umask(old)
        assert stat.S_IMODE((tmp_path / "home.img").stat().st_mode) == 0o600


class TestAllocatedBytes:
    """allocated_bytes: what a file takes on its filesystem, from any stat result."""

    def test_a_sparse_file_takes_less_than_its_size(self, tmp_path: Path) -> None:
        path = tmp_path / "disk.img"
        create_sparse(path, 64 * 1024 * 1024)
        st = path.stat()
        assert allocated_bytes(st) < st.st_size

    def test_a_block_is_512_bytes_whatever_the_filesystem_block_size(self) -> None:
        """st_blocks counts 512-byte units on Linux and macOS, not st_blksize units (4096 here)."""
        st = os.stat_result((0o100600, 1, 2, 1, 0, 0, 1_000_000, 0, 0, 0), {"st_blocks": 8, "st_blksize": 4096})
        assert allocated_bytes(st) == 4096

    def test_written_data_counts(self, tmp_path: Path) -> None:
        path = tmp_path / "data"
        path.write_bytes(os.urandom(1024 * 1024))
        assert allocated_bytes(path.stat()) >= 1024 * 1024
        assert allocated_bytes(os.lstat(path)) == allocated_bytes(path.stat())


class TestCreateFile:
    @pytest.mark.parametrize(
        ("umask", "mode"),
        [(0o002, 0o644), (0o002, 0o600), (0o077, 0o644)],
        ids=["umask-002-0644", "umask-002-0600", "umask-077-0644"],
    )
    def test_exact_mode_whatever_the_umask(self, tmp_path: Path, umask: int, mode: int) -> None:
        """umask 002 (Ubuntu, Fedora) removes no group write bit: open() would make 0664, group
        qemu-vm under `sg qemu-vm`. umask 077 would make a 0644 kernel 0600, which QEMU as
        qemu-vm cannot read. The file gets exactly the mode asked."""
        old = os.umask(umask)
        try:
            with create_file(tmp_path / "f", mode=mode, dir_fd=None) as f:
                f.write(b"data")
        finally:
            os.umask(old)
        assert stat.S_IMODE((tmp_path / "f").stat().st_mode) == mode
        assert (tmp_path / "f").read_bytes() == b"data"

    def test_refuses_existing_name_and_symlink(self, tmp_path: Path) -> None:
        """O_EXCL: an existing file keeps its mode and content; a symlink is not followed."""
        (tmp_path / "f").write_bytes(b"old")
        (tmp_path / "link").symlink_to(tmp_path / "missing")
        for name in ("f", "link"):
            with pytest.raises(FileExistsError):
                create_file(tmp_path / name, mode=0o600, dir_fd=None)
        assert not (tmp_path / "missing").exists()


class TestCloneSparse:
    @staticmethod
    def _written_disk(path: Path) -> bytes:
        create_sparse(path, 64 * _MIB)
        with path.open("r+b") as f:
            f.seek(10 * _MIB)
            f.write(b"A" * 4096)
            f.seek(50 * _MIB)
            f.write(b"B" * 4096)
        return path.read_bytes()

    @pytest.mark.parametrize("cow", [True, False], ids=["copy-on-write", "data-ranges"])
    def test_same_bytes_holes_kept(self, tmp_path: Path, cow: bool) -> None:
        """Content is identical and the clone allocates no more than the source.

        Compared with the source, not with an absolute size: APFS allocates the
        range in front of a write in a sparse file (measured), ext4/XFS do not.
        The data-ranges case is a filesystem without copy-on-write (ext4 and others):
        _clonefile and _ficlone answer no, as they do there.
        """
        src, dst = tmp_path / "src.img", tmp_path / "dst.img"
        content = self._written_disk(src)
        if cow:
            clone_sparse(src, dst)
        else:
            with (
                patch.object(files_disk, "_clonefile", return_value=False),
                patch.object(files_disk, "_ficlone", return_value=False),
            ):
                assert clone_sparse(src, dst) is False  # a copy: its blocks are not shared
        assert dst.read_bytes() == content
        assert _allocated(dst) <= _allocated(src) + _MIB
        assert _allocated(dst) < 64 * _MIB  # holes were not written as zeros
        assert stat.S_IMODE(dst.stat().st_mode) == 0o600

    @pytest.mark.parametrize("way", ["_clonefile", "_ficlone"])
    def test_clone_says_when_blocks_are_shared(self, tmp_path: Path, way: str) -> None:
        """A copy-on-write clone (APFS or FICLONE) says so: its blocks take no free space yet.
        No host has both ways: each one answers yes in turn (_clonefile, _ficlone)."""
        src = tmp_path / "src.img"
        self._written_disk(src)
        with (
            patch.object(files_disk, "_clonefile", return_value=way == "_clonefile"),
            patch.object(files_disk, "_ficlone", return_value=way == "_ficlone"),
        ):
            assert clone_sparse(src, tmp_path / "dst.img") is True

    def test_clone_is_independent(self, tmp_path: Path) -> None:
        """Writes to the clone never reach the source (each VM owns its disk)."""
        src, dst = tmp_path / "src.img", tmp_path / "dst.img"
        content = self._written_disk(src)
        clone_sparse(src, dst)
        with dst.open("r+b") as f:
            f.seek(10 * _MIB)
            f.write(b"C" * 4096)
        assert src.read_bytes() == content

    def test_copy_without_seek_data_fails(self, tmp_path: Path) -> None:
        """No clone and no SEEK_DATA (EINVAL): the copy fails, never writes every hole as zeros.
        The patches make that filesystem: no copy-on-write, and lseek(SEEK_DATA) refused."""
        src = tmp_path / "src.img"
        self._written_disk(src)
        real_lseek = os.lseek

        def no_seek_data(fd: int, pos: int, how: int) -> int:
            if how == os.SEEK_DATA:
                raise OSError(errno.EINVAL, "Invalid argument")
            return real_lseek(fd, pos, how)

        with (
            patch.object(files_disk, "_clonefile", return_value=False),
            patch.object(files_disk, "_ficlone", return_value=False),
            patch.object(os, "lseek", no_seek_data),
            pytest.raises(OSError, match="Invalid argument"),
        ):
            clone_sparse(src, tmp_path / "dst.img")


class TestDataRanges:
    """A filesystem that reports 4 KiB extents (ext4, XFS: a RAM file with zero pages as holes).
    os.lseek is the seam: APFS reports such small holes as data, so a real file cannot show them."""

    _GRAIN = 64 * 1024
    _SIZE = 3 * _GRAIN
    _EXTENTS = [(offset, offset + 4096) for offset in range(0, _GRAIN, 8192)] + [(135168, 139264)]
    """Eight 4 KiB extents in the first 64 KiB block, one in the third; holes between them."""

    @classmethod
    def _lseek(cls, calls: list[int]) -> Callable[[int, int, int], int]:
        def lseek(fd: int, pos: int, how: int) -> int:
            calls.append(how)
            if how == os.SEEK_DATA:
                for start, end in cls._EXTENTS:
                    if end > pos:
                        return max(start, pos)
                raise OSError(errno.ENXIO, "No such device or address")
            assert how == os.SEEK_HOLE
            return next((end for start, end in cls._EXTENTS if start <= pos < end), pos)

        return lseek

    def test_grain_skips_small_extents_of_one_block(self) -> None:
        """grain 1: every extent (two lseek calls each). A grain of 64 KiB: ranges widened to
        whole blocks, the same blocks with data, and one search per run of blocks."""
        exact_calls: list[int] = []
        block_calls: list[int] = []
        with patch.object(os, "lseek", self._lseek(exact_calls)):
            exact = list(files_disk.data_ranges(0, self._SIZE, grain=1))
        with patch.object(os, "lseek", self._lseek(block_calls)):
            blocks = list(files_disk.data_ranges(0, self._SIZE, grain=self._GRAIN))
        assert exact == self._EXTENTS
        assert blocks == [(0, 65536), (131072, 196608)]
        assert (len(exact_calls), len(block_calls)) == (19, 4)

    def test_data_range_copy_never_writes_a_hole(self, tmp_path: Path) -> None:
        """A clone without copy-on-write copies the exact extents: a 4 KiB hole inside a block is
        never written as zeros (it would take disk space)."""
        src = tmp_path / "src.img"
        with src.open("wb") as f:
            f.truncate(self._SIZE)
            for start, end in self._EXTENTS:
                f.seek(start)
                f.write(b"D" * (end - start))
        written: list[tuple[int, int]] = []
        pwrite_all = files_disk.pwrite_all

        def recorded(fd: int, data: bytes, offset: int) -> None:
            written.append((offset, offset + len(data)))
            pwrite_all(fd, data, offset)

        with (
            patch.object(files_disk, "_clonefile", return_value=False),
            patch.object(files_disk, "_ficlone", return_value=False),
            patch.object(files_disk, "pwrite_all", recorded),
            patch.object(os, "lseek", self._lseek([])),
        ):
            clone_sparse(src, tmp_path / "dst.img")
        assert written == self._EXTENTS
        assert (tmp_path / "dst.img").read_bytes() == src.read_bytes()


class TestPwriteAll:
    def test_short_writes_are_continued(self, tmp_path: Path) -> None:
        """os.pwrite may write less than asked (a full disk, a signal): the rest is written."""
        path = tmp_path / "f"
        path.write_bytes(bytes(10))
        pwrite = os.pwrite

        def at_most_three(fd: int, data: bytes, offset: int) -> int:
            return pwrite(fd, data[:3], offset)

        fd = os.open(path, os.O_WRONLY)
        try:
            with patch.object(os, "pwrite", at_most_three):
                files_disk.pwrite_all(fd, b"abcdefgh", 1)
        finally:
            os.close(fd)
        assert path.read_bytes() == b"\0abcdefgh\0"


@skip_unless_linux
class TestMemoryFilesystem:
    def test_shared_memory_is_tmpfs(self) -> None:
        assert memory_filesystem(Path("/dev/shm")) == "tmpfs"

    def test_a_path_not_made_yet_answers_for_its_nearest_folder(self) -> None:
        assert memory_filesystem(Path("/dev/shm/not-made/yet/deep")) == "tmpfs"

    def test_a_kernel_filesystem_is_not_memory_for_files(self) -> None:
        assert memory_filesystem(Path("/proc/self")) is None

    def test_the_answer_is_statfs_of_the_nearest_folder_not_the_mount_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tmpfs stacked on a disk folder, or a disk bound over a tmpfs, is what statfs of the
        path says: the mount table (its first match, its escapes) is never read."""
        asked: list[Path] = []

        def fake_statfs(path: Path) -> bytes:
            asked.append(path)
            magic = 0x01021994 if path == tmp_path else 0xEF53  # tmpfs here, ext4 elsewhere
            return magic.to_bytes(8, sys.byteorder) + bytes(4088)

        real_open = Path.open

        def no_mount_table(self: Path, *args: Any, **kwargs: Any) -> Any:
            assert str(self) != "/proc/self/mounts", "memory_filesystem read the mount table"
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr("exec_sandbox.files.disk.statfs", fake_statfs)
        monkeypatch.setattr(Path, "open", no_mount_table)

        assert memory_filesystem(tmp_path / "not" / "made") == "tmpfs"
        assert asked == [tmp_path]  # the nearest existing folder
        assert memory_filesystem(tmp_path.parent) is None

    def test_statfs_sees_the_mount_a_path_is_on(self, tmp_path: Path) -> None:
        """f_type of the real mount, not a parse of /proc/self/mounts (stacked or bind mounts)."""
        raw = statfs(Path("/dev/shm"))
        assert raw is not None
        assert int.from_bytes(raw[:8], sys.byteorder) & 0xFFFFFFFF == 0x01021994  # TMPFS_MAGIC
        assert statfs(tmp_path / "missing" / "file") is None


@pytest.fixture
def group_write_umask() -> Iterator[None]:
    """umask 002: the default of user accounts on Ubuntu (user private groups)."""
    old = os.umask(0o002)
    try:
        yield
    finally:
        os.umask(old)


class TestMakePrivateDir:
    def test_missing_parents_are_private_too(self, tmp_path: Path, group_write_umask: None) -> None:
        """Path.mkdir(parents=True) gives missing parents the umask's mode (0775 here), which the
        private-folder checks refuse: every folder made is 0700."""
        target = tmp_path / "home" / ".local" / "share" / "exec-sandbox"

        make_private_dir(target)

        made = [tmp_path / "home", *(tmp_path / "home").glob("**/")]
        assert {stat.S_IMODE(folder.stat().st_mode) & 0o777 for folder in made} == {0o700}
        assert target in made

    def test_existing_folders_keep_their_mode(self, tmp_path: Path) -> None:
        tmp_path.chmod(0o755)
        make_private_dir(tmp_path / "new")
        make_private_dir(tmp_path / "new")  # exists: no error

        assert stat.S_IMODE(tmp_path.stat().st_mode) & 0o777 == 0o755
        assert stat.S_IMODE((tmp_path / "new").stat().st_mode) & 0o777 == 0o700

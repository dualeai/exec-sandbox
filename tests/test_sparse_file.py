"""Sparse home disk files: created and cloned without allocating their holes."""

import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from exec_sandbox import sparse_file
from exec_sandbox.sparse_file import clone_sparse, create_sparse

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
        """
        src, dst = tmp_path / "src.img", tmp_path / "dst.img"
        content = self._written_disk(src)
        if cow:
            clone_sparse(src, dst)
        else:
            with (
                patch.object(sparse_file, "_clonefile", return_value=False),
                patch.object(sparse_file, "_ficlone", return_value=False),
            ):
                clone_sparse(src, dst)
        assert dst.read_bytes() == content
        assert _allocated(dst) <= _allocated(src) + _MIB
        assert _allocated(dst) < 64 * _MIB  # holes were not written as zeros
        assert stat.S_IMODE(dst.stat().st_mode) == 0o600

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
        """No clone and no SEEK_DATA (EINVAL): the copy fails, never writes every hole as zeros."""
        src = tmp_path / "src.img"
        self._written_disk(src)
        real_lseek = os.lseek

        def no_seek_data(fd: int, pos: int, how: int) -> int:
            if how == os.SEEK_DATA:
                raise OSError(errno.EINVAL, "Invalid argument")
            return real_lseek(fd, pos, how)

        with (
            patch.object(sparse_file, "_clonefile", return_value=False),
            patch.object(sparse_file, "_ficlone", return_value=False),
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
            exact = list(sparse_file.data_ranges(0, self._SIZE, grain=1))
        with patch.object(os, "lseek", self._lseek(block_calls)):
            blocks = list(sparse_file.data_ranges(0, self._SIZE, grain=self._GRAIN))
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
        pwrite_all = sparse_file.pwrite_all

        def recorded(fd: int, data: bytes, offset: int) -> None:
            written.append((offset, offset + len(data)))
            pwrite_all(fd, data, offset)

        with (
            patch.object(sparse_file, "_clonefile", return_value=False),
            patch.object(sparse_file, "_ficlone", return_value=False),
            patch.object(sparse_file, "pwrite_all", recorded),
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
                sparse_file.pwrite_all(fd, b"abcdefgh", 1)
        finally:
            os.close(fd)
        assert path.read_bytes() == b"\0abcdefgh\0"

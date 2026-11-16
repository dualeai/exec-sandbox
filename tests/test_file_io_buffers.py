"""Integration tests for IO[bytes] buffer support in file CRUD interfaces.

Tests that write_file accepts IO[bytes] (e.g. BytesIO) as content and
read_file accepts IO[bytes] as destination, with full roundtrip integrity.
bytes content is sent as a BytesIO: the roundtrips across chunk sizes are in
test_file_io_integrity.py, make_executable in its TestFileIoBasicOperations.

All tests require a real VM via the scheduler fixture.
"""

import asyncio
import io
import os
import tempfile
from pathlib import Path

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec

# =============================================================================
# Roundtrip integrity — IO[bytes] combinations
# =============================================================================


@pytest.mark.slow
class TestBufferRoundtrip:
    """Write/read roundtrips mixing bytes, Path, and IO[bytes]."""

    async def test_write_bytesio_read_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write from BytesIO, read to Path — verify content match."""
        content = os.urandom(4096)
        dest = tmp_path / "out.bin"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("test.bin", io.BytesIO(content))
            await session.read_file("test.bin", destination=dest)

        assert dest.read_bytes() == content

    async def test_write_bytes_read_bytesio(self, scheduler: Scheduler) -> None:
        """Write from bytes, read to BytesIO — verify buf.getvalue(). The caller's buffer
        stays open after read_file (getvalue() raises on a closed BytesIO)."""
        content = os.urandom(4096)
        buf = io.BytesIO()

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("test.bin", content)
            await session.read_file("test.bin", destination=buf)

        assert buf.getvalue() == content


# =============================================================================
# Edge cases
# =============================================================================


@pytest.mark.slow
class TestBufferEdgeCases:
    """Edge cases for IO[bytes] buffer support."""

    async def test_non_zero_write_position(self, scheduler: Scheduler) -> None:
        """BytesIO with non-zero position — only data after position is written."""
        data = b"payload"
        buf = io.BytesIO(b"prefix" + data)
        buf.seek(6)  # Skip "prefix"
        read_buf = io.BytesIO()

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("test.bin", buf)
            await session.read_file("test.bin", destination=read_buf)

        assert read_buf.getvalue() == data

    async def test_pre_filled_read_dest(self, scheduler: Scheduler) -> None:
        """Pre-filled BytesIO read destination — content appended after existing data."""
        content = b"new_data"
        dest = io.BytesIO(b"old")
        dest.seek(3)  # Position at end of "old"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("test.bin", content)
            await session.read_file("test.bin", destination=dest)

        assert dest.getvalue() == b"old" + content

    async def test_concurrent_mixed_buffer_and_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Concurrent writes (BytesIO + bytes) and reads (BytesIO + Path) — no cross-contamination."""
        data_a = os.urandom(2048)
        data_b = os.urandom(2048)
        data_c = os.urandom(2048)
        data_d = os.urandom(2048)

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Concurrent writes: 2 from BytesIO, 2 from bytes
            await asyncio.gather(
                session.write_file("a.bin", io.BytesIO(data_a)),
                session.write_file("b.bin", data_b),
                session.write_file("c.bin", io.BytesIO(data_c)),
                session.write_file("d.bin", data_d),
            )

            # Concurrent reads: 2 to BytesIO, 2 to Path
            buf_a = io.BytesIO()
            path_b = tmp_path / "b.bin"
            buf_c = io.BytesIO()
            path_d = tmp_path / "d.bin"

            await asyncio.gather(
                session.read_file("a.bin", destination=buf_a),
                session.read_file("b.bin", destination=path_b),
                session.read_file("c.bin", destination=buf_c),
                session.read_file("d.bin", destination=path_d),
            )

        assert buf_a.getvalue() == data_a, "BytesIO read a mismatch"
        assert path_b.read_bytes() == data_b, "Path read b mismatch"
        assert buf_c.getvalue() == data_c, "BytesIO read c mismatch"
        assert path_d.read_bytes() == data_d, "Path read d mismatch"


# =============================================================================
# Non-standard streams
# =============================================================================


@pytest.mark.slow
class TestNonStandardStreams:
    """Tests with non-seekable and SpooledTemporaryFile streams."""

    async def test_non_seekable_write(self, scheduler: Scheduler) -> None:
        """A non-seekable stream (no size known before the read) writes every byte."""
        from tests.conftest import NonSeekableIO

        content = os.urandom(256)
        raw = NonSeekableIO(content)
        stream = io.BufferedReader(raw)
        read_buf = io.BytesIO()

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("test.bin", stream)  # type: ignore[arg-type]
            await session.read_file("test.bin", destination=read_buf)

        assert read_buf.getvalue() == content

    async def test_spooled_temporary_file(self, scheduler: Scheduler) -> None:
        """A SpooledTemporaryFile (seekable, held in memory under max_size) writes every byte."""
        content = os.urandom(256)
        read_buf = io.BytesIO()

        with tempfile.SpooledTemporaryFile(max_size=1024) as spool:
            spool.write(content)
            spool.seek(0)
            async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
                await session.write_file("test.bin", spool)  # type: ignore[arg-type]
                await session.read_file("test.bin", destination=read_buf)

        assert read_buf.getvalue() == content


# =============================================================================
# Error cases
# =============================================================================


@pytest.mark.slow
class TestBufferErrors:
    """Error cases for IO[bytes] buffer support."""

    async def test_closed_bytesio_write(self, scheduler: Scheduler) -> None:
        """Closed BytesIO write raises InvalidInputError ARGUMENT_INVALID before any transfer: the
        guest has no file at the path."""
        from exec_sandbox.exceptions import ErrorCode, InvalidInputError

        buf = io.BytesIO(b"data")
        buf.close()

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID):
                await session.write_file("test.bin", buf)
            assert "test.bin" not in {f.name for f in await session.list_files()}


# =============================================================================
# Ownership / lifecycle
# =============================================================================


@pytest.mark.slow
class TestBufferOwnership:
    """Verify buffer ownership: caller's buffers are never closed."""

    async def test_write_does_not_close_caller_buffer(self, scheduler: Scheduler) -> None:
        """After write_file, caller's BytesIO is still usable."""
        buf = io.BytesIO(b"hello")

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("test.bin", buf)

        # Buffer should still be open and readable
        buf.seek(0)
        assert buf.read() == b"hello"

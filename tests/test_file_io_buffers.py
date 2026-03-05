# pyright: reportPrivateUsage=false
"""Integration tests for IO[bytes] buffer support in file CRUD interfaces.

Tests that write_file accepts IO[bytes] (e.g. BytesIO) as content and
read_file accepts IO[bytes] as destination, with full roundtrip integrity.

All tests require a real VM via the scheduler fixture.
"""

import asyncio
import hashlib
import io
import os
import tempfile
from pathlib import Path

import pytest

from exec_sandbox.constants import FILE_TRANSFER_CHUNK_SIZE
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

_CHUNK = FILE_TRANSFER_CHUNK_SIZE


# =============================================================================
# Roundtrip integrity — IO[bytes] combinations
# =============================================================================


@pytest.mark.slow
class TestBufferRoundtrip:
    """Write/read roundtrips mixing bytes, Path, and IO[bytes]."""

    async def test_write_bytesio_read_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write from BytesIO, read to Path — verify content match."""
        content = os.urandom(4096)
        expected_hash = hashlib.sha256(content).hexdigest()
        dest = tmp_path / "out.bin"

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("test.bin", io.BytesIO(content))
            await session.read_file("test.bin", destination=dest)

        actual_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
        assert actual_hash == expected_hash

    async def test_write_bytes_read_bytesio(self, scheduler: Scheduler) -> None:
        """Write from bytes, read to BytesIO — verify buf.getvalue()."""
        content = os.urandom(4096)
        buf = io.BytesIO()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("test.bin", content)
            await session.read_file("test.bin", destination=buf)

        assert buf.getvalue() == content

    async def test_bytesio_full_roundtrip(self, scheduler: Scheduler) -> None:
        """Write from BytesIO, read to BytesIO — full buffer roundtrip, SHA256."""
        content = os.urandom(4096)
        expected_hash = hashlib.sha256(content).hexdigest()
        read_buf = io.BytesIO()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("test.bin", io.BytesIO(content))
            await session.read_file("test.bin", destination=read_buf)

        actual_hash = hashlib.sha256(read_buf.getvalue()).hexdigest()
        assert actual_hash == expected_hash

    @pytest.mark.parametrize(
        ("size", "label"),
        [
            (0, "0B"),
            (1, "1B"),
            (_CHUNK - 1, "chunk-1"),
            (_CHUNK, "chunk"),
            (_CHUNK + 1, "chunk+1"),
            (1024 * 1024, "1MB"),
        ],
        ids=["0B", "1B", "chunk-1", "chunk", "chunk+1", "1MB"],
    )
    async def test_bytesio_roundtrip_sizes(self, scheduler: Scheduler, size: int, label: str) -> None:
        """Parametrized BytesIO roundtrip across boundary sizes."""
        content = os.urandom(size) if size > 0 else b""
        expected_hash = hashlib.sha256(content).hexdigest()
        read_buf = io.BytesIO()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file(f"size_{label}.bin", io.BytesIO(content))
            await session.read_file(f"size_{label}.bin", destination=read_buf)

        actual_hash = hashlib.sha256(read_buf.getvalue()).hexdigest()
        assert len(read_buf.getvalue()) == size, f"[{label}] Size mismatch"
        assert actual_hash == expected_hash, f"[{label}] SHA256 mismatch"


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

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("test.bin", buf)
            await session.read_file("test.bin", destination=read_buf)

        assert read_buf.getvalue() == data

    async def test_pre_filled_read_dest(self, scheduler: Scheduler) -> None:
        """Pre-filled BytesIO read destination — content appended after existing data."""
        content = b"new_data"
        dest = io.BytesIO(b"old")
        dest.seek(3)  # Position at end of "old"

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("test.bin", content)
            await session.read_file("test.bin", destination=dest)

        assert dest.getvalue() == b"old" + content

    async def test_empty_bytesio_write(self, scheduler: Scheduler) -> None:
        """Empty BytesIO write creates empty file, list_files shows size=0."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("empty.bin", io.BytesIO(b""))
            files = await session.list_files()
            matching = [f for f in files if f.name == "empty.bin"]
            assert len(matching) == 1
            assert matching[0].size == 0

    async def test_make_executable_via_bytesio(self, scheduler: Scheduler) -> None:
        """Write shell script from BytesIO with make_executable, exec succeeds."""
        script = b"#!/bin/sh\necho hello_from_bytesio"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("run.sh", io.BytesIO(script), make_executable=True)
            result = await session.exec(
                "import subprocess; print(subprocess.check_output('./run.sh').decode().strip())"
            )
            assert "hello_from_bytesio" in result.stdout

    async def test_concurrent_buffers(self, scheduler: Scheduler) -> None:
        """asyncio.gather with multiple BytesIO writes/reads — no cross-contamination."""
        contents = {f"file_{i}.bin": os.urandom(1024) for i in range(4)}

        async with await scheduler.session(language=Language.PYTHON) as session:
            # Write all files
            for name, data in contents.items():
                await session.write_file(name, io.BytesIO(data))

            # Read all files concurrently
            bufs: dict[str, io.BytesIO] = {name: io.BytesIO() for name in contents}

            async def read_one(name: str) -> None:
                await session.read_file(name, destination=bufs[name])

            await asyncio.gather(*(read_one(name) for name in contents))

        for name, expected in contents.items():
            assert bufs[name].getvalue() == expected, f"Cross-contamination in {name}"

    async def test_concurrent_mixed_buffer_and_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Concurrent writes (BytesIO + bytes) and reads (BytesIO + Path) — no cross-contamination."""
        data_a = os.urandom(2048)
        data_b = os.urandom(2048)
        data_c = os.urandom(2048)
        data_d = os.urandom(2048)

        async with await scheduler.session(language=Language.PYTHON) as session:
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
        """Non-seekable stream write succeeds (size validation skipped)."""
        from tests.conftest import NonSeekableIO

        content = os.urandom(256)
        raw = NonSeekableIO(content)
        stream = io.BufferedReader(raw)
        read_buf = io.BytesIO()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("test.bin", stream)  # type: ignore[arg-type]
            await session.read_file("test.bin", destination=read_buf)

        assert read_buf.getvalue() == content

    async def test_spooled_temporary_file(self, scheduler: Scheduler) -> None:
        """SpooledTemporaryFile: seekable, size-validated, roundtrip works."""
        content = os.urandom(256)
        spool = tempfile.SpooledTemporaryFile(max_size=1024)
        spool.write(content)
        spool.seek(0)
        read_buf = io.BytesIO()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("test.bin", spool)  # type: ignore[arg-type]
            await session.read_file("test.bin", destination=read_buf)

        assert read_buf.getvalue() == content
        spool.close()


# =============================================================================
# Error cases
# =============================================================================


@pytest.mark.slow
class TestBufferErrors:
    """Error cases for IO[bytes] buffer support."""

    async def test_closed_bytesio_write(self, scheduler: Scheduler) -> None:
        """Closed BytesIO write raises ValueError."""
        buf = io.BytesIO(b"data")
        buf.close()

        async with await scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises(ValueError):
                await session.write_file("test.bin", buf)

    async def test_path_traversal_bytesio(self, scheduler: Scheduler) -> None:
        """Path traversal with BytesIO content raises VmPermanentError."""
        from exec_sandbox.exceptions import VmPermanentError

        async with await scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises(VmPermanentError):
                await session.write_file("../escape.txt", io.BytesIO(b"x"))


# =============================================================================
# Ownership / lifecycle
# =============================================================================


@pytest.mark.slow
class TestBufferOwnership:
    """Verify buffer ownership: caller's buffers are never closed."""

    async def test_write_does_not_close_caller_buffer(self, scheduler: Scheduler) -> None:
        """After write_file, caller's BytesIO is still usable."""
        buf = io.BytesIO(b"hello")

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("test.bin", buf)

        # Buffer should still be open and readable
        buf.seek(0)
        assert buf.read() == b"hello"

    async def test_read_does_not_close_caller_buffer(self, scheduler: Scheduler) -> None:
        """After read_file, caller's BytesIO is still usable."""
        dest = io.BytesIO()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("test.bin", b"hello")
            await session.read_file("test.bin", destination=dest)

        # Buffer should still be open and readable
        dest.seek(0)
        assert dest.read() == b"hello"

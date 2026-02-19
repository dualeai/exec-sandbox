# pyright: reportPrivateUsage=false
"""Unit tests for Session._guard(), _resolve_content(), write_file streaming protocol,
and FileOpDispatcher routing.

Tests the orchestration layer WITHOUT requiring VMs. All VM/channel interactions
are mocked. Covers:
- _guard() lifecycle checks, lock serialization, idle timer reset
- _resolve_content() sync/async file reads, boundary validation
- WriteFileRequest header serialization: validity, escaping
- UnixSocketChannel.send_raw_request()
- FileOpDispatcher message routing by op_id
"""

import asyncio
import json
import random
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from exec_sandbox.constants import FILE_TRANSFER_CHUNK_SIZE, MAX_FILE_PATH_LENGTH, MAX_FILE_SIZE_BYTES
from exec_sandbox.exceptions import SessionClosedError
from exec_sandbox.guest_agent_protocol import (
    FileChunkResponseMessage,
    FileReadCompleteMessage,
    FileWriteAckMessage,
    PongMessage,
    StreamingErrorMessage,
    StreamingMessage,
    WriteFileRequest,
)
from exec_sandbox.guest_channel import FileOpDispatcher, UnixSocketChannel
from exec_sandbox.models import ExecutionResult, FileInfo, TimingBreakdown
from exec_sandbox.session import Session

# ============================================================================
# Helpers — build a Session with mocked VM + VmManager
# ============================================================================


def _make_exec_result() -> ExecutionResult:
    """Build a minimal ExecutionResult for mocks."""
    return ExecutionResult(
        stdout="",
        stderr="",
        exit_code=0,
        execution_time_ms=1,
        external_cpu_time_ms=None,
        external_memory_peak_mb=None,
        timing=TimingBreakdown(setup_ms=0, boot_ms=0, execute_ms=1, total_ms=1, connect_ms=0),
        spawn_ms=None,
        process_ms=None,
    )


def _make_session(
    idle_timeout_seconds: int = 300,
    default_timeout_seconds: int = 30,
) -> tuple[Session, AsyncMock, AsyncMock]:
    """Create a Session with mocked VM and VmManager.

    Returns (session, mock_vm, mock_vm_manager).
    """
    mock_vm = AsyncMock()
    mock_vm.vm_id = "test-vm-001"
    mock_vm.execute = AsyncMock(return_value=_make_exec_result())
    mock_vm.write_file = AsyncMock()
    mock_vm.read_file = AsyncMock(return_value=None)
    mock_vm.list_files = AsyncMock(return_value=[FileInfo(name="a.txt", is_dir=False, size=10)])

    mock_vm_manager = AsyncMock()
    mock_vm_manager.destroy_vm = AsyncMock()

    session = Session(
        vm=mock_vm,
        vm_manager=mock_vm_manager,
        idle_timeout_seconds=idle_timeout_seconds,
        default_timeout_seconds=default_timeout_seconds,
    )
    return session, mock_vm, mock_vm_manager


# ============================================================================
# _guard() — Normal cases
# ============================================================================


class TestGuardNormal:
    """Happy-path tests for _guard() context manager."""

    async def test_exec_calls_vm_execute(self) -> None:
        """exec() passes through to VM.execute under _guard."""
        session, mock_vm, _ = _make_session()
        await session.exec("print(1)")
        mock_vm.execute.assert_awaited_once()
        assert session.exec_count == 1

    async def test_read_file_calls_vm(self, tmp_path: Path) -> None:
        """read_file() delegates to VM under _guard."""
        session, mock_vm, _ = _make_session()
        dest = tmp_path / "test.bin"
        await session.read_file("test.txt", destination=dest)
        mock_vm.read_file.assert_awaited_once_with("test.txt", destination=dest)

    async def test_list_files_calls_vm(self) -> None:
        """list_files() delegates to VM under _guard."""
        session, mock_vm, _ = _make_session()
        result = await session.list_files("subdir")
        mock_vm.list_files.assert_awaited_once_with("subdir")
        assert len(result) == 1

    async def test_write_file_bytes_calls_vm(self) -> None:
        """write_file() with bytes delegates to VM under _guard."""
        session, mock_vm, _ = _make_session()
        await session.write_file("out.bin", b"data", make_executable=True)
        mock_vm.write_file.assert_awaited_once()
        call_args = mock_vm.write_file.call_args
        assert call_args[0][0] == "out.bin"
        # Content is now an IO[bytes] stream (BytesIO wrapping the bytes)
        assert call_args[1]["make_executable"] is True


# ============================================================================
# _guard() — Closed session (fast path, before lock)
# ============================================================================


class TestGuardClosedFastPath:
    """Tests the pre-lock closed check in _guard()."""

    async def test_exec_after_close_raises(self) -> None:
        session, _, _ = _make_session()
        await session.close()
        with pytest.raises(SessionClosedError, match="Session is closed"):
            await session.exec("x = 1")

    async def test_write_file_after_close_raises(self) -> None:
        session, _, _ = _make_session()
        await session.close()
        with pytest.raises(SessionClosedError, match="Session is closed"):
            await session.write_file("f.txt", b"data")

    async def test_read_file_after_close_raises(self, tmp_path: Path) -> None:
        session, _, _ = _make_session()
        await session.close()
        with pytest.raises(SessionClosedError, match="Session is closed"):
            await session.read_file("f.txt", destination=tmp_path / "f.bin")

    async def test_list_files_after_close_raises(self) -> None:
        session, _, _ = _make_session()
        await session.close()
        with pytest.raises(SessionClosedError, match="Session is closed"):
            await session.list_files()


# ============================================================================
# _guard() — Closed while waiting for lock (race condition)
# ============================================================================


class TestGuardClosedDuringLockWait:
    """Tests the re-check inside the lock (step 3).

    Strategy: coroutine A acquires lock via a slow VM op, sets _closed=True
    before releasing lock. Coroutine B passes step 1 (not closed), blocks
    on the lock, then when A releases, B acquires and hits step 3.
    """

    async def test_read_file_closed_while_waiting_for_lock(self, tmp_path: Path) -> None:
        """read_file() raises 'closed while waiting' when session closes mid-wait."""
        session, mock_vm, _ = _make_session()
        lock_held = asyncio.Event()

        async def slow_read_and_close(path: str, *, destination: Path) -> None:
            lock_held.set()
            await asyncio.sleep(0.15)
            # Close session while still holding the lock
            session._closed = True

        mock_vm.read_file = AsyncMock(side_effect=slow_read_and_close)

        # Task A: holds lock, sets _closed = True before releasing
        task_a = asyncio.create_task(session.read_file("first.txt", destination=tmp_path / "first.bin"))

        # Wait for A to hold the lock
        await lock_held.wait()

        # Task B: passes step 1 (not closed yet), blocks on lock
        # When A releases lock, B acquires it and sees _closed=True → step 3
        with pytest.raises(SessionClosedError, match="closed while waiting"):
            await session.list_files()

        await task_a

    async def test_list_files_closed_while_waiting_for_lock(self, tmp_path: Path) -> None:
        """list_files() raises 'closed while waiting' when session closes mid-wait."""
        session, mock_vm, _ = _make_session()
        lock_held = asyncio.Event()

        async def slow_list_and_close(path: str = "") -> list[FileInfo]:
            lock_held.set()
            await asyncio.sleep(0.15)
            session._closed = True
            return [FileInfo(name="x.txt", is_dir=False, size=1)]

        mock_vm.list_files = AsyncMock(side_effect=slow_list_and_close)

        task_a = asyncio.create_task(session.list_files())
        await lock_held.wait()

        with pytest.raises(SessionClosedError, match="closed while waiting"):
            await session.read_file("blocked.txt", destination=tmp_path / "blocked.bin")

        await task_a


# ============================================================================
# _guard() — Idle timer reset by file I/O
# ============================================================================


class TestGuardIdleTimerReset:
    """Verify that file I/O operations reset the idle timer via _guard()."""

    async def test_write_file_resets_idle_timer(self) -> None:
        """write_file() resets idle timer (prevents auto-close)."""
        session, _, _ = _make_session(idle_timeout_seconds=1)

        # Wait most of the timeout, then call write_file to reset
        await asyncio.sleep(0.7)
        await session.write_file("reset.txt", b"x")

        # Wait again — if timer was NOT reset, session would be closed by now
        await asyncio.sleep(0.7)
        assert not session.closed

        # Cleanup
        await session.close()

    async def test_read_file_resets_idle_timer(self, tmp_path: Path) -> None:
        """read_file() resets idle timer."""
        session, _, _ = _make_session(idle_timeout_seconds=1)

        await asyncio.sleep(0.7)
        await session.read_file("any.txt", destination=tmp_path / "any.bin")

        await asyncio.sleep(0.7)
        assert not session.closed

        await session.close()

    async def test_list_files_resets_idle_timer(self) -> None:
        """list_files() resets idle timer."""
        session, _, _ = _make_session(idle_timeout_seconds=1)

        await asyncio.sleep(0.7)
        await session.list_files()

        await asyncio.sleep(0.7)
        assert not session.closed

        await session.close()


# ============================================================================
# _guard() — Concurrent serialization for file I/O
# ============================================================================


class TestGuardConcurrentFileIO:
    """Verify file I/O operations are serialized by the exec lock."""

    async def test_concurrent_writes_serialized(self) -> None:
        """Two concurrent write_file() calls are serialized, not interleaved."""
        session, mock_vm, _ = _make_session()
        call_order: list[str] = []

        async def tracked_write(path: str, content: bytes, *, make_executable: bool = False) -> None:
            call_order.append(f"start:{path}")
            await asyncio.sleep(0.05)
            call_order.append(f"end:{path}")

        mock_vm.write_file = AsyncMock(side_effect=tracked_write)

        await asyncio.gather(
            session.write_file("a.txt", b"a"),
            session.write_file("b.txt", b"b"),
        )

        # Serialized means one fully completes before the other starts
        assert call_order[0].startswith("start:")
        assert call_order[1].startswith("end:")
        assert call_order[2].startswith("start:")
        assert call_order[3].startswith("end:")

        await session.close()

    async def test_read_and_write_serialized(self, tmp_path: Path) -> None:
        """Concurrent read_file and write_file are serialized."""
        session, mock_vm, _ = _make_session()
        order: list[str] = []

        async def tracked_read(path: str, *, destination: Path) -> None:
            order.append("read_start")
            await asyncio.sleep(0.05)
            order.append("read_end")

        async def tracked_write(path: str, content: bytes, *, make_executable: bool = False) -> None:
            order.append("write_start")
            await asyncio.sleep(0.05)
            order.append("write_end")

        mock_vm.read_file = AsyncMock(side_effect=tracked_read)
        mock_vm.write_file = AsyncMock(side_effect=tracked_write)

        await asyncio.gather(
            session.read_file("a.txt", destination=tmp_path / "a.bin"),
            session.write_file("b.txt", b"b"),
        )

        # Verify no interleaving (one start/end pair completes before the next starts)
        for i in range(0, len(order), 2):
            assert "start" in order[i]
            assert "end" in order[i + 1]

        await session.close()


# ============================================================================
# _resolve_content() — Normal cases
# ============================================================================


class TestResolveContentNormal:
    """Happy-path tests for _resolve_content()."""

    async def test_bytes_returns_bytesio(self) -> None:
        """Bytes content is wrapped in BytesIO."""
        session, _, _ = _make_session()
        stream = await session._resolve_content(b"hello")
        assert stream.read() == b"hello"
        stream.close()

    async def test_path_returns_file_handle(self, tmp_path: Path) -> None:
        """Path returns an open file handle (streams from disk)."""
        f = tmp_path / "small.txt"
        f.write_bytes(b"small content")
        session, _, _ = _make_session()
        stream = await session._resolve_content(f)
        assert stream.read() == b"small content"
        stream.close()

    async def test_path_large_file_streams(self, tmp_path: Path) -> None:
        """Large file returns a file handle (never loaded fully by _resolve_content)."""
        f = tmp_path / "large.bin"
        content = b"x" * (2 * 1024 * 1024)  # 2MB
        f.write_bytes(content)

        session, _, _ = _make_session()
        stream = await session._resolve_content(f)
        assert stream.read() == content
        stream.close()

    async def test_write_file_resolve_before_lock(self, tmp_path: Path) -> None:
        """write_file() resolves Path BEFORE acquiring the exec lock.

        If _resolve_content ran inside the lock, a concurrent read_file
        would have to wait for disk I/O. Instead, disk I/O and lock
        acquisition are decoupled.
        """
        from typing import IO

        session, _, _ = _make_session()

        f = tmp_path / "big.bin"
        f.write_bytes(b"x" * 100)

        # Spy on _resolve_content to track call order relative to lock
        resolve_called = asyncio.Event()
        original_resolve = session._resolve_content

        async def spied_resolve(content: bytes | Path) -> IO[bytes]:
            result = await original_resolve(content)
            resolve_called.set()
            # Wait a bit to ensure read_file tries to acquire the lock
            await asyncio.sleep(0.1)
            return result

        session._resolve_content = spied_resolve  # type: ignore[method-assign]

        async def write_task() -> None:
            await session.write_file("out.bin", f)

        async def read_task() -> None:
            await resolve_called.wait()
            # This should NOT be blocked by _resolve_content
            await session.read_file("any.txt", destination=tmp_path / "any.bin")

        # Both should complete — read_file isn't blocked by file I/O
        await asyncio.gather(write_task(), read_task())

        await session.close()


# ============================================================================
# _resolve_content() — Edge cases (boundaries)
# ============================================================================


class TestResolveContentEdgeCases:
    """Boundary value tests for _resolve_content()."""

    async def test_bytes_exactly_at_max_size_accepted(self) -> None:
        """Bytes of exactly MAX_FILE_SIZE_BYTES are accepted."""
        session, _, _ = _make_session()
        content = b"x" * MAX_FILE_SIZE_BYTES
        stream = await session._resolve_content(content)
        assert stream.read() == content
        stream.close()

    async def test_bytes_one_over_max_rejected(self) -> None:
        """Bytes of MAX_FILE_SIZE_BYTES + 1 raise ValueError."""
        session, _, _ = _make_session()
        with pytest.raises(ValueError, match="exceeds"):
            await session._resolve_content(b"x" * (MAX_FILE_SIZE_BYTES + 1))

    async def test_path_exactly_at_max_size_accepted(self, tmp_path: Path) -> None:
        """Path pointing to file of exactly MAX_FILE_SIZE_BYTES is accepted."""
        f = tmp_path / "exact.bin"
        f.write_bytes(b"x" * MAX_FILE_SIZE_BYTES)
        session, _, _ = _make_session()
        stream = await session._resolve_content(f)
        assert stream.read() == b"x" * MAX_FILE_SIZE_BYTES
        stream.close()

    async def test_path_one_over_max_rejected(self, tmp_path: Path) -> None:
        """Path to file of MAX_FILE_SIZE_BYTES + 1 raises ValueError."""
        f = tmp_path / "over.bin"
        f.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))
        session, _, _ = _make_session()
        with pytest.raises(ValueError, match="exceeds"):
            await session._resolve_content(f)

    async def test_empty_bytes_accepted(self) -> None:
        """Empty bytes (0 length) are accepted."""
        session, _, _ = _make_session()
        stream = await session._resolve_content(b"")
        assert stream.read() == b""
        stream.close()

    async def test_empty_file_accepted(self, tmp_path: Path) -> None:
        """Empty file (0 bytes) is accepted."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        session, _, _ = _make_session()
        stream = await session._resolve_content(f)
        assert stream.read() == b""
        stream.close()


# ============================================================================
# _resolve_content() — Out of bounds
# ============================================================================


class TestResolveContentOutOfBounds:
    """Error handling in _resolve_content()."""

    async def test_nonexistent_path_raises_file_not_found(self, tmp_path: Path) -> None:
        """Path to non-existent file raises FileNotFoundError."""
        session, _, _ = _make_session()
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            await session._resolve_content(tmp_path / "ghost.txt")

    async def test_path_is_directory_raises(self, tmp_path: Path) -> None:
        """Path pointing to a directory raises (open fails)."""
        d = tmp_path / "subdir"
        d.mkdir()
        session, _, _ = _make_session()
        # A directory exists but open("rb") will fail
        with pytest.raises(IsADirectoryError):
            await session._resolve_content(d)


# ============================================================================
# _resolve_content() — TOCTOU guard (file grows between stat and read)
# ============================================================================


class TestResolveContentToctou:
    """TOCTOU: stat-based size check before opening file handle."""

    async def test_file_over_stat_limit_rejected(self, tmp_path: Path) -> None:
        """File whose stat() reports > MAX_FILE_SIZE_BYTES is rejected."""
        f = tmp_path / "big.bin"
        f.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))
        session, _, _ = _make_session()
        with pytest.raises(ValueError, match="exceeds"):
            await session._resolve_content(f)

    async def test_file_stays_within_limit_accepted(self, tmp_path: Path) -> None:
        """File within limit returns a readable stream."""
        f = tmp_path / "stable.bin"
        content = b"x" * 1000
        f.write_bytes(content)
        session, _, _ = _make_session()
        stream = await session._resolve_content(f)
        assert stream.read() == content
        stream.close()


# ============================================================================
# QemuVM.write_file() — Path validation (bypassed Pydantic validators)
# ============================================================================


class TestWriteFilePathValidation:
    """Path validation that mirrors WriteFileRequest Pydantic validators."""

    async def test_empty_path_rejected(self) -> None:
        """Empty path raises VmPermanentError."""
        import io

        from exec_sandbox.exceptions import VmPermanentError
        from exec_sandbox.qemu_vm import QemuVM

        vm = MagicMock(spec=QemuVM)
        vm.vm_id = "test-vm"
        vm.channel = AsyncMock()

        # Call the real write_file method
        with pytest.raises(VmPermanentError, match="must not be empty"):
            await QemuVM.write_file(vm, "", io.BytesIO(b"data"))

    async def test_path_at_max_length_accepted(self) -> None:
        """Path of exactly MAX_FILE_PATH_LENGTH chars is accepted."""
        import io

        from exec_sandbox.guest_agent_protocol import FileWriteAckMessage
        from exec_sandbox.qemu_vm import QemuVM

        vm = MagicMock(spec=QemuVM)
        vm.vm_id = "test-vm"
        vm.channel = AsyncMock()
        vm.channel.connect = AsyncMock()
        vm.channel.enqueue_raw = AsyncMock()
        vm.channel.unregister_op = AsyncMock()
        ack = FileWriteAckMessage(op_id="test", path="x" * MAX_FILE_PATH_LENGTH, bytes_written=4)
        op_queue: asyncio.Queue[FileWriteAckMessage] = asyncio.Queue()
        op_queue.put_nowait(ack)
        vm.channel.register_op = AsyncMock(return_value=op_queue)

        # Should not raise
        await QemuVM.write_file(vm, "x" * MAX_FILE_PATH_LENGTH, io.BytesIO(b"data"))

    async def test_path_over_max_length_rejected(self) -> None:
        """Path exceeding MAX_FILE_PATH_LENGTH raises VmPermanentError."""
        import io

        from exec_sandbox.exceptions import VmPermanentError
        from exec_sandbox.qemu_vm import QemuVM

        vm = MagicMock(spec=QemuVM)
        vm.vm_id = "test-vm"
        vm.channel = AsyncMock()

        with pytest.raises(VmPermanentError, match="exceeds"):
            await QemuVM.write_file(vm, "x" * (MAX_FILE_PATH_LENGTH + 1), io.BytesIO(b"data"))


# ============================================================================
# UnixSocketChannel.send_raw_request()
# ============================================================================


class TestUnixSocketSendRawRequest:
    """Tests for UnixSocketChannel.send_raw_request()."""

    async def test_not_connected_raises(self) -> None:
        """send_raw_request on unconnected channel raises RuntimeError."""
        channel = UnixSocketChannel(socket_path="/tmp/test.sock", expected_uid=1000)
        with pytest.raises(RuntimeError, match="not connected"):
            await channel.send_raw_request(b"data\n", timeout=10)

    async def test_dead_write_worker_raises(self) -> None:
        """send_raw_request detects a crashed write worker."""
        channel = UnixSocketChannel(socket_path="/tmp/test.sock", expected_uid=1000)
        channel._reader = AsyncMock()
        channel._writer = MagicMock()

        # Simulate a crashed write worker
        failed_task: asyncio.Task[None] = asyncio.ensure_future(asyncio.sleep(0))
        await failed_task  # Let it complete
        # Make it look like it raised
        dead_task = asyncio.create_task(_raise_task())
        try:
            await dead_task
        except RuntimeError:
            pass
        channel._write_task = dead_task

        with pytest.raises(RuntimeError, match="Write worker crashed"):
            await channel.send_raw_request(b"data\n", timeout=10)

    async def test_queues_write_and_reads_response(self) -> None:
        """send_raw_request queues data via write_queue and reads response."""
        channel = UnixSocketChannel(socket_path="/tmp/test.sock", expected_uid=1000)
        reader = AsyncMock()
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        channel._reader = reader
        channel._writer = writer

        # Create a live write worker that actually drains the queue
        channel._shutdown_event.clear()
        channel._write_task = asyncio.create_task(channel._write_worker())

        ack = FileWriteAckMessage(op_id="test", path="test.txt", bytes_written=42)
        reader.readuntil = AsyncMock(return_value=(ack.model_dump_json() + "\n").encode())

        result = await channel.send_raw_request(b'{"test":"data"}\n', timeout=10)
        assert isinstance(result, FileWriteAckMessage)
        assert result.bytes_written == 42

        # Cleanup
        channel._shutdown_event.set()
        channel._write_task.cancel()
        try:
            await channel._write_task
        except asyncio.CancelledError:
            pass


async def _raise_task() -> None:
    """Helper: task that raises RuntimeError (for dead-worker simulation)."""
    raise RuntimeError("simulated crash")


# ============================================================================
# WriteFileRequest header serialization
# ============================================================================


class TestWriteFileHeaderSerialization:
    """Unit tests for WriteFileRequest header serialization.

    Validates the header frame used in the streaming file write protocol,
    without requiring a VM.
    """

    @staticmethod
    def _build_header(path: str, op_id: str = "test123", make_executable: bool = False) -> bytes:
        """Build a WriteFileRequest header frame (newline-delimited JSON)."""
        req = WriteFileRequest(op_id=op_id, path=path, make_executable=make_executable)
        return req.model_dump_json(by_alias=False, exclude_none=True).encode() + b"\n"

    def test_header_is_valid_json(self) -> None:
        """The header frame is valid JSON (newline-delimited)."""
        frame = self._build_header("test.txt")
        parsed = json.loads(frame.rstrip(b"\n"))
        assert parsed["action"] == "write_file"
        assert parsed["op_id"] == "test123"
        assert parsed["path"] == "test.txt"
        assert parsed["make_executable"] is False

    def test_header_make_executable_true(self) -> None:
        """make_executable=True emits JSON true."""
        frame = self._build_header("run.sh", make_executable=True)
        parsed = json.loads(frame)
        assert parsed["make_executable"] is True

    def test_header_make_executable_false(self) -> None:
        """make_executable=False emits JSON false."""
        frame = self._build_header("data.csv")
        parsed = json.loads(frame)
        assert parsed["make_executable"] is False

    def test_header_roundtrips_via_pydantic(self) -> None:
        """Header frame round-trips through Pydantic model."""
        frame = self._build_header("payload.bin", op_id="abc123", make_executable=True)
        parsed = json.loads(frame)
        req = WriteFileRequest(**parsed)
        assert req.action == "write_file"
        assert req.op_id == "abc123"
        assert req.path == "payload.bin"
        assert req.make_executable is True

    def test_header_ends_with_newline(self) -> None:
        """Header frame ends with exactly one newline (protocol requirement)."""
        frame = self._build_header("test.txt")
        assert frame.endswith(b"\n")
        assert not frame.endswith(b"\n\n")


# ============================================================================
# WriteFileRequest header — JSON escaping (weird cases)
# ============================================================================


class TestWriteFileHeaderEscaping:
    """Test that Pydantic's JSON serialization properly escapes special characters in path."""

    _build_header = staticmethod(TestWriteFileHeaderSerialization._build_header)

    def test_double_quotes_in_path(self) -> None:
        """Path with double quotes is properly escaped."""
        frame = self._build_header('my"file.txt')
        parsed = json.loads(frame)
        assert parsed["path"] == 'my"file.txt'

    def test_backslash_in_path(self) -> None:
        """Path with backslash is properly escaped."""
        frame = self._build_header("dir\\file.txt")
        parsed = json.loads(frame)
        assert parsed["path"] == "dir\\file.txt"

    def test_newline_in_path(self) -> None:
        r"""Path with newline is escaped (would break \n-delimited protocol)."""
        frame = self._build_header("line1\nline2.txt")
        # The frame itself must be a single line (newline-delimited protocol)
        lines = frame.split(b"\n")
        # Should be exactly 2 parts: the JSON frame and the empty string after final \n
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["path"] == "line1\nline2.txt"

    def test_tab_in_path(self) -> None:
        """Path with tab character is escaped."""
        frame = self._build_header("col1\tcol2.tsv")
        parsed = json.loads(frame)
        assert parsed["path"] == "col1\tcol2.tsv"

    def test_unicode_in_path(self) -> None:
        """Path with Unicode characters round-trips correctly."""
        frame = self._build_header("données/résultat.csv")
        parsed = json.loads(frame)
        assert parsed["path"] == "données/résultat.csv"

    def test_emoji_in_path(self) -> None:
        """Path with emoji round-trips correctly."""
        frame = self._build_header("data/output_\U0001f680.json")
        parsed = json.loads(frame)
        assert parsed["path"] == "data/output_\U0001f680.json"

    def test_all_json_metacharacters(self) -> None:
        r"""Path with all JSON escape-worthy characters: " \ / \b \f \n \r \t."""
        nasty_path = 'a"b\\c/d\be\ff\ng\rh\ti'
        frame = self._build_header(nasty_path)
        parsed = json.loads(frame)
        assert parsed["path"] == nasty_path


# ============================================================================
# Zstd streaming compress/decompress round-trip (mirrors write_file/read_file)
# ============================================================================


class TestZstdStreamingRoundtrip:
    """Verify the streaming zstd pattern used by write_file and read_file.

    write_file: ZstdCompressor.compress(chunk) + flush() — incremental.
    read_file:  ZstdDecompressor.decompress(chunk)        — incremental.
    """

    @staticmethod
    def _get_zstd():  # type: ignore[no-untyped-def]
        """Import the zstd module matching the runtime."""
        import sys

        if sys.version_info >= (3, 14):
            from compression import zstd  # type: ignore[import-not-found]
        else:
            from backports import zstd  # type: ignore[import-untyped,no-redef]
        return zstd

    def test_streaming_roundtrip_small(self) -> None:
        """Small content survives streaming compress/decompress."""
        _zstd = self._get_zstd()
        content = b"hello world via streaming zstd"

        # Compress incrementally (same pattern as write_file)
        compressor = _zstd.ZstdCompressor(level=3)
        compressed = compressor.compress(content) + compressor.flush()

        # Decompress incrementally (same pattern as read_file)
        decompressor = _zstd.ZstdDecompressor()
        result = decompressor.decompress(compressed)
        assert result == content

    def test_streaming_roundtrip_chunked(self) -> None:
        """Content split into FILE_TRANSFER_CHUNK_SIZE chunks compresses and decompresses."""
        import os

        _zstd = self._get_zstd()
        content = os.urandom(256 * 1024)  # 256KB

        # Compress chunk by chunk (mirrors write_file loop)
        compressor = _zstd.ZstdCompressor(level=3)
        compressed_parts: list[bytes] = []
        chunk_size = FILE_TRANSFER_CHUNK_SIZE
        for offset in range(0, len(content), chunk_size):
            part = compressor.compress(content[offset : offset + chunk_size])
            if part:
                compressed_parts.append(part)
        remaining = compressor.flush()
        if remaining:
            compressed_parts.append(remaining)

        # Decompress the concatenated compressed stream (mirrors read_file)
        decompressor = _zstd.ZstdDecompressor()
        result = decompressor.decompress(b"".join(compressed_parts))
        assert result == content

    def test_streaming_roundtrip_empty(self) -> None:
        """Empty content survives streaming round-trip."""
        _zstd = self._get_zstd()
        compressor = _zstd.ZstdCompressor(level=3)
        compressed = compressor.compress(b"") + compressor.flush()
        result = _zstd.ZstdDecompressor().decompress(compressed)
        assert result == b""

    def test_compressed_is_smaller_for_repetitive_data(self) -> None:
        """Highly compressible content produces smaller compressed output."""
        _zstd = self._get_zstd()
        content = b"\x00" * (1024 * 1024)
        compressor = _zstd.ZstdCompressor(level=3)
        compressed = compressor.compress(content) + compressor.flush()
        assert len(compressed) < len(content)


# ============================================================================
# FileOpDispatcher routing tests
# ============================================================================


def _feed(reader: asyncio.StreamReader, msg: dict[str, object]) -> None:
    """Feed a JSON message into a StreamReader (simulates guest agent output)."""
    reader.feed_data(json.dumps(msg).encode() + b"\n")


async def _drain(queue: asyncio.Queue[StreamingMessage], timeout: float = 0.5) -> list[StreamingMessage]:
    """Drain all messages from a queue with a short timeout."""
    msgs: list[StreamingMessage] = []
    try:
        while True:
            msgs.append(await asyncio.wait_for(queue.get(), timeout=timeout))
    except TimeoutError:
        pass
    return msgs


class TestFileOpDispatcherRouting:
    """Tests the FileOpDispatcher dispatch layer in isolation.

    Uses real asyncio.StreamReader + feed_data(). No VMs, no mocks.
    We control exactly what bytes arrive and in what order.
    """

    # ------------------------------------------------------------------
    # Normal cases
    # ------------------------------------------------------------------

    async def test_error_with_op_id_routes_to_op_queue(self) -> None:
        """Error with op_id routes to the registered op queue, not default."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-A")
        _feed(reader, {"type": "error", "error_type": "io_error", "message": "fail", "op_id": "op-A"})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, StreamingErrorMessage)
        assert msg.op_id == "op-A"
        assert msg.message == "fail"

        # Default queue should be empty
        assert dispatcher._default_queue.empty()

        await dispatcher.stop()

    async def test_error_without_op_id_routes_to_default(self) -> None:
        """Error without op_id goes to default queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        await dispatcher.register_op("op-A")
        _feed(reader, {"type": "error", "error_type": "timeout", "message": "fail"})

        msg = await asyncio.wait_for(dispatcher._default_queue.get(), timeout=1.0)
        assert isinstance(msg, StreamingErrorMessage)
        assert msg.op_id is None

        await dispatcher.stop()

    async def test_file_chunk_routes_to_op_queue(self) -> None:
        """File chunk with op_id routes to the registered op queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-A")
        _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": "AAAA"})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, FileChunkResponseMessage)
        assert msg.op_id == "op-A"

        assert dispatcher._default_queue.empty()

        await dispatcher.stop()

    async def test_non_file_message_routes_to_default(self) -> None:
        """Pong (no op_id) goes to default queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        await dispatcher.register_op("op-A")
        _feed(reader, {"type": "pong", "version": "1.0"})

        msg = await asyncio.wait_for(dispatcher._default_queue.get(), timeout=1.0)
        assert isinstance(msg, PongMessage)

        await dispatcher.stop()

    # ------------------------------------------------------------------
    # Concurrent interleaved routing
    # ------------------------------------------------------------------

    async def test_interleaved_3_ops_mixed_success_and_error(self) -> None:
        """3 ops interleaved: A succeeds, B errors, C succeeds. Each queue gets only its messages."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        qa = await dispatcher.register_op("op-A")
        qb = await dispatcher.register_op("op-B")
        qc = await dispatcher.register_op("op-C")

        # Feed in worst-case interleaved order
        _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": "a1"})
        _feed(reader, {"type": "error", "error_type": "io_error", "message": "B failed", "op_id": "op-B"})
        _feed(reader, {"type": "file_chunk", "op_id": "op-C", "data": "c1"})
        _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": "a2"})
        _feed(reader, {"type": "file_read_complete", "op_id": "op-C", "path": "c.txt", "size": 10})
        _feed(reader, {"type": "file_read_complete", "op_id": "op-A", "path": "a.txt", "size": 20})

        a_msgs = await _drain(qa)
        b_msgs = await _drain(qb)
        c_msgs = await _drain(qc)

        assert len(a_msgs) == 3  # 2 chunks + 1 complete
        assert len(b_msgs) == 1  # 1 error
        assert len(c_msgs) == 2  # 1 chunk + 1 complete
        assert isinstance(b_msgs[0], StreamingErrorMessage)
        assert dispatcher._default_queue.empty()

        await dispatcher.stop()

    async def test_interleaved_writes_errors_dont_leak(self) -> None:
        """Write errors route to correct op queues, nothing leaks to default."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        q1 = await dispatcher.register_op("w1")
        q2 = await dispatcher.register_op("w2")
        q3 = await dispatcher.register_op("w3")

        _feed(reader, {"type": "error", "error_type": "validation_error", "message": "path invalid", "op_id": "w1"})
        _feed(reader, {"type": "file_write_ack", "op_id": "w2", "path": "ok.txt", "bytes_written": 100})
        _feed(reader, {"type": "error", "error_type": "io_error", "message": "disk full", "op_id": "w3"})

        m1 = await _drain(q1)
        m2 = await _drain(q2)
        m3 = await _drain(q3)

        assert len(m1) == 1
        assert isinstance(m1[0], StreamingErrorMessage)
        assert "path" in m1[0].message

        assert len(m2) == 1
        assert isinstance(m2[0], FileWriteAckMessage)

        assert len(m3) == 1
        assert isinstance(m3[0], StreamingErrorMessage)
        assert "disk" in m3[0].message

        assert dispatcher._default_queue.empty()

        await dispatcher.stop()

    async def test_concurrent_5_ops_all_error(self) -> None:
        """5 ops each receive their own error, default stays empty."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        queues = {}
        for i in range(5):
            queues[f"op-{i}"] = await dispatcher.register_op(f"op-{i}")

        for i in range(5):
            _feed(reader, {"type": "error", "error_type": "io_error", "message": f"err-{i}", "op_id": f"op-{i}"})

        for i in range(5):
            msgs = await _drain(queues[f"op-{i}"])
            assert len(msgs) == 1
            assert isinstance(msgs[0], StreamingErrorMessage)
            assert msgs[0].message == f"err-{i}"

        assert dispatcher._default_queue.empty()

        await dispatcher.stop()

    async def test_concurrent_5_ops_all_success(self) -> None:
        """5 ops each receive interleaved chunks and completions."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        queues = {}
        for i in range(5):
            queues[f"op-{i}"] = await dispatcher.register_op(f"op-{i}")

        # Interleave: chunk for each, then complete for each
        for i in range(5):
            _feed(reader, {"type": "file_chunk", "op_id": f"op-{i}", "data": f"d{i}"})
        for i in range(5):
            _feed(reader, {"type": "file_read_complete", "op_id": f"op-{i}", "path": f"f{i}.txt", "size": i * 10})

        for i in range(5):
            msgs = await _drain(queues[f"op-{i}"])
            assert len(msgs) == 2  # 1 chunk + 1 complete
            assert isinstance(msgs[0], FileChunkResponseMessage)
            assert isinstance(msgs[1], FileReadCompleteMessage)

        assert dispatcher._default_queue.empty()

        await dispatcher.stop()

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    async def test_error_with_unregistered_op_id_is_discarded(self) -> None:
        """Error with op_id not in _op_queues is discarded (not routed to default)."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        qa = await dispatcher.register_op("op-A")
        _feed(reader, {"type": "error", "error_type": "io_error", "message": "fail", "op_id": "op-UNKNOWN"})

        # Give dispatch loop time to process
        await asyncio.sleep(0.05)

        # Message should be discarded, not in default queue or op-A queue
        assert dispatcher._default_queue.empty()
        assert qa.empty()

        await dispatcher.stop()

    async def test_error_with_empty_string_op_id_goes_to_default(self) -> None:
        """Error with op_id=\"\" goes to default (empty string is falsy)."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        await dispatcher.register_op("op-A")
        _feed(reader, {"type": "error", "error_type": "io_error", "message": "fail", "op_id": ""})

        msg = await asyncio.wait_for(dispatcher._default_queue.get(), timeout=1.0)
        assert isinstance(msg, StreamingErrorMessage)

        await dispatcher.stop()

    async def test_error_after_op_unregistered_is_discarded(self) -> None:
        """Error for unregistered op is discarded (prevents default queue pollution)."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        await dispatcher.register_op("op-A")
        await dispatcher.unregister_op("op-A")
        _feed(reader, {"type": "error", "error_type": "io_error", "message": "fail", "op_id": "op-A"})

        # Give dispatch loop time to process
        await asyncio.sleep(0.05)

        # Message should be discarded, not pollute default queue
        assert dispatcher._default_queue.empty()

        await dispatcher.stop()

    async def test_rapid_register_unregister_during_dispatch(self) -> None:
        """First message routes to op queue, second (after unregister) is discarded."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-A")
        _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": "first"})

        # Wait for first message to be dispatched
        msg1 = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg1, FileChunkResponseMessage)

        await dispatcher.unregister_op("op-A")
        _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": "second"})

        # Give dispatch loop time to process
        await asyncio.sleep(0.05)

        # Second message should be discarded, not in default queue
        assert dispatcher._default_queue.empty()

        await dispatcher.stop()

    # ------------------------------------------------------------------
    # Weird / out-of-bounds cases
    # ------------------------------------------------------------------

    async def test_error_with_op_id_matching_different_message_type(self) -> None:
        """Pong goes to default, error with op_id goes to op queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-A")
        _feed(reader, {"type": "pong", "version": "1.0"})
        _feed(reader, {"type": "error", "op_id": "op-A", "error_type": "x", "message": "y"})

        pong = await asyncio.wait_for(dispatcher._default_queue.get(), timeout=1.0)
        assert isinstance(pong, PongMessage)

        err = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(err, StreamingErrorMessage)
        assert err.op_id == "op-A"

        await dispatcher.stop()

    async def test_100_ops_interleaved_all_route_correctly(self) -> None:
        """100 ops with shuffled messages all route to correct queues."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        queues: dict[str, asyncio.Queue[StreamingMessage]] = {}
        for i in range(100):
            queues[f"op-{i}"] = await dispatcher.register_op(f"op-{i}")

        # Build messages: 50 successes (file_read_complete) + 50 errors, shuffled
        messages: list[dict[str, object]] = [
            {"type": "file_read_complete", "op_id": f"op-{i}", "path": f"f{i}", "size": i} for i in range(50)
        ]
        messages.extend(
            {"type": "error", "op_id": f"op-{i}", "error_type": "io_error", "message": f"err-{i}"}
            for i in range(50, 100)
        )

        rng = random.Random(42)
        rng.shuffle(messages)

        for msg in messages:
            _feed(reader, msg)

        for i in range(100):
            msgs = await _drain(queues[f"op-{i}"])
            assert len(msgs) == 1, f"op-{i} got {len(msgs)} messages"

        assert dispatcher._default_queue.empty()

        await dispatcher.stop()

    async def test_same_op_id_receives_multiple_messages_in_order(self) -> None:
        """10 chunks for same op_id arrive in FIFO order."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-A")

        for i in range(10):
            _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": f"chunk-{i}"})

        msgs = await _drain(op_queue)
        assert len(msgs) == 10
        for i, msg in enumerate(msgs):
            assert isinstance(msg, FileChunkResponseMessage)
            assert msg.data == f"chunk-{i}"

        assert dispatcher._default_queue.empty()

        await dispatcher.stop()

# pyright: reportPrivateUsage=false
"""Unit tests for Session._guard(), _resolve_content(), and write_file JSON frame.

Tests the orchestration layer WITHOUT requiring VMs. All VM/channel interactions
are mocked. Covers:
- _guard() lifecycle checks, lock serialization, idle timer reset
- _resolve_content() sync/async file reads, boundary validation
- QemuVM.write_file() manual JSON frame: validity, escaping, round-trips
- GuestChannel.send_raw_request() on TcpChannel and UnixSocketChannel
"""

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox.constants import MAX_FILE_PATH_LENGTH, MAX_FILE_SIZE_BYTES
from exec_sandbox.exceptions import SessionClosedError
from exec_sandbox.guest_agent_protocol import (
    FileWriteAckMessage,
    StreamingErrorMessage,
    WriteFileRequest,
)
from exec_sandbox.guest_channel import TcpChannel, UnixSocketChannel
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
    mock_vm.read_file = AsyncMock(return_value=b"file-content")
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

    async def test_read_file_calls_vm(self) -> None:
        """read_file() delegates to VM under _guard."""
        session, mock_vm, _ = _make_session()
        result = await session.read_file("test.txt")
        mock_vm.read_file.assert_awaited_once_with("test.txt")
        assert result == b"file-content"

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
        mock_vm.write_file.assert_awaited_once_with("out.bin", b"data", make_executable=True)


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

    async def test_read_file_after_close_raises(self) -> None:
        session, _, _ = _make_session()
        await session.close()
        with pytest.raises(SessionClosedError, match="Session is closed"):
            await session.read_file("f.txt")

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

    async def test_read_file_closed_while_waiting_for_lock(self) -> None:
        """read_file() raises 'closed while waiting' when session closes mid-wait."""
        session, mock_vm, _ = _make_session()
        lock_held = asyncio.Event()

        async def slow_read_and_close(path: str) -> bytes:
            lock_held.set()
            await asyncio.sleep(0.15)
            # Close session while still holding the lock
            session._closed = True
            return b"data"

        mock_vm.read_file = AsyncMock(side_effect=slow_read_and_close)

        # Task A: holds lock, sets _closed = True before releasing
        task_a = asyncio.create_task(session.read_file("first.txt"))

        # Wait for A to hold the lock
        await lock_held.wait()

        # Task B: passes step 1 (not closed yet), blocks on lock
        # When A releases lock, B acquires it and sees _closed=True → step 3
        with pytest.raises(SessionClosedError, match="closed while waiting"):
            await session.list_files()

        await task_a

    async def test_list_files_closed_while_waiting_for_lock(self) -> None:
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
            await session.read_file("blocked.txt")

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

    async def test_read_file_resets_idle_timer(self) -> None:
        """read_file() resets idle timer."""
        session, _, _ = _make_session(idle_timeout_seconds=1)

        await asyncio.sleep(0.7)
        await session.read_file("any.txt")

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

    async def test_read_and_write_serialized(self) -> None:
        """Concurrent read_file and write_file are serialized."""
        session, mock_vm, _ = _make_session()
        order: list[str] = []

        async def tracked_read(path: str) -> bytes:
            order.append("read_start")
            await asyncio.sleep(0.05)
            order.append("read_end")
            return b"data"

        async def tracked_write(path: str, content: bytes, *, make_executable: bool = False) -> None:
            order.append("write_start")
            await asyncio.sleep(0.05)
            order.append("write_end")

        mock_vm.read_file = AsyncMock(side_effect=tracked_read)
        mock_vm.write_file = AsyncMock(side_effect=tracked_write)

        await asyncio.gather(
            session.read_file("a.txt"),
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

    async def test_bytes_passthrough(self) -> None:
        """Bytes content is returned as-is."""
        session, _, _ = _make_session()
        result = await session._resolve_content(b"hello")
        assert result == b"hello"

    async def test_path_small_file_read(self, tmp_path: Path) -> None:
        """Small file is read via asyncio.to_thread."""
        f = tmp_path / "small.txt"
        f.write_bytes(b"small content")
        session, _, _ = _make_session()
        result = await session._resolve_content(f)
        assert result == b"small content"

    async def test_path_large_file_read(self, tmp_path: Path) -> None:
        """Large file is read via asyncio.to_thread."""
        f = tmp_path / "large.bin"
        content = b"x" * (2 * 1024 * 1024)  # 2MB
        f.write_bytes(content)

        session, _, _ = _make_session()
        result = await session._resolve_content(f)
        assert result == content

    async def test_write_file_resolve_before_lock(self, tmp_path: Path) -> None:
        """write_file() resolves Path BEFORE acquiring the exec lock.

        If _resolve_content ran inside the lock, a concurrent read_file
        would have to wait for disk I/O. Instead, disk I/O and lock
        acquisition are decoupled.
        """
        session, _, _ = _make_session()

        f = tmp_path / "big.bin"
        f.write_bytes(b"x" * 100)

        # Spy on _resolve_content to track call order relative to lock
        resolve_called = asyncio.Event()
        original_resolve = session._resolve_content

        async def spied_resolve(content: bytes | Path) -> bytes:
            result = await original_resolve(content)
            resolve_called.set()
            # Wait a bit to ensure read_file tries to acquire the lock
            await asyncio.sleep(0.1)
            return result

        session._resolve_content = spied_resolve  # type: ignore[method-assign]

        async def write_task() -> None:
            await session.write_file("out.bin", f)

        async def read_task() -> bytes:
            await resolve_called.wait()
            # This should NOT be blocked by _resolve_content
            return await session.read_file("any.txt")

        # Both should complete — read_file isn't blocked by file I/O
        results = await asyncio.gather(write_task(), read_task())
        assert results[1] == b"file-content"

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
        result = await session._resolve_content(content)
        assert len(result) == MAX_FILE_SIZE_BYTES

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
        result = await session._resolve_content(f)
        assert len(result) == MAX_FILE_SIZE_BYTES

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
        result = await session._resolve_content(b"")
        assert result == b""

    async def test_empty_file_accepted(self, tmp_path: Path) -> None:
        """Empty file (0 bytes) is accepted."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        session, _, _ = _make_session()
        result = await session._resolve_content(f)
        assert result == b""


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
        """Path pointing to a directory raises (read_bytes fails)."""
        d = tmp_path / "subdir"
        d.mkdir()
        session, _, _ = _make_session()
        # A directory exists but read_bytes() will fail
        with pytest.raises(IsADirectoryError):
            await session._resolve_content(d)


# ============================================================================
# _resolve_content() — TOCTOU guard (file grows between stat and read)
# ============================================================================


class TestResolveContentToctou:
    """TOCTOU: file grows between stat() and read_bytes()."""

    async def test_file_grows_after_stat_rejected(self, tmp_path: Path) -> None:
        """File that grows past MAX_FILE_SIZE_BYTES after stat() is rejected."""
        f = tmp_path / "growing.bin"
        # Start small (under async threshold) so sync path is used
        f.write_bytes(b"x" * 100)

        session, _, _ = _make_session()

        # Monkey-patch read_bytes on the class — accepts self since it replaces an instance method
        def growing_read(_self: Path) -> bytes:
            return b"x" * (MAX_FILE_SIZE_BYTES + 1)

        with patch.object(type(f), "read_bytes", growing_read):
            with pytest.raises(ValueError, match="grew to"):
                await session._resolve_content(f)

    async def test_file_stays_within_limit_accepted(self, tmp_path: Path) -> None:
        """File that stays within limit after read is accepted normally."""
        f = tmp_path / "stable.bin"
        content = b"x" * 1000
        f.write_bytes(content)
        session, _, _ = _make_session()
        result = await session._resolve_content(f)
        assert result == content


# ============================================================================
# QemuVM.write_file() — Path validation (bypassed Pydantic validators)
# ============================================================================


class TestWriteFilePathValidation:
    """Path validation that mirrors WriteFileRequest Pydantic validators."""

    async def test_empty_path_rejected(self) -> None:
        """Empty path raises VmPermanentError."""
        from exec_sandbox.exceptions import VmPermanentError
        from exec_sandbox.qemu_vm import QemuVM

        vm = MagicMock(spec=QemuVM)
        vm.vm_id = "test-vm"
        vm.channel = AsyncMock()

        # Call the real write_file method
        with pytest.raises(VmPermanentError, match="must not be empty"):
            await QemuVM.write_file(vm, "", b"data")

    async def test_path_at_max_length_accepted(self) -> None:
        """Path of exactly MAX_FILE_PATH_LENGTH chars is accepted."""
        from exec_sandbox.guest_agent_protocol import FileWriteAckMessage
        from exec_sandbox.qemu_vm import QemuVM

        vm = MagicMock(spec=QemuVM)
        vm.vm_id = "test-vm"
        vm.channel = AsyncMock()
        vm.channel.connect = AsyncMock()
        ack = FileWriteAckMessage(path="x" * MAX_FILE_PATH_LENGTH, bytes_written=4)
        vm.channel.send_raw_request = AsyncMock(return_value=ack)

        # Should not raise
        await QemuVM.write_file(vm, "x" * MAX_FILE_PATH_LENGTH, b"data")

    async def test_path_over_max_length_rejected(self) -> None:
        """Path exceeding MAX_FILE_PATH_LENGTH raises VmPermanentError."""
        from exec_sandbox.exceptions import VmPermanentError
        from exec_sandbox.qemu_vm import QemuVM

        vm = MagicMock(spec=QemuVM)
        vm.vm_id = "test-vm"
        vm.channel = AsyncMock()

        with pytest.raises(VmPermanentError, match="exceeds"):
            await QemuVM.write_file(vm, "x" * (MAX_FILE_PATH_LENGTH + 1), b"data")


# ============================================================================
# TcpChannel.send_raw_request() — Normal
# ============================================================================


def _mock_tcp_channel() -> tuple[TcpChannel, AsyncMock, MagicMock]:
    """Create a TcpChannel with mocked reader/writer (already connected)."""
    channel = TcpChannel(host="127.0.0.1", port=5000)
    reader = AsyncMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    writer.transport = MagicMock()
    channel._reader = reader
    channel._writer = writer
    return channel, reader, writer


class TestTcpSendRawRequestNormal:
    """Happy path for TcpChannel.send_raw_request()."""

    async def test_sends_raw_bytes_and_parses_response(self) -> None:
        """send_raw_request sends data as-is and parses JSON response."""
        channel, reader, writer = _mock_tcp_channel()

        # Simulate a FileWriteAckMessage response
        ack = FileWriteAckMessage(path="test.txt", bytes_written=100)
        response_json = ack.model_dump_json() + "\n"
        reader.readuntil = AsyncMock(return_value=response_json.encode())

        data = b'{"action":"write_file","path":"test.txt","content_base64":"aGVsbG8=","make_executable":false}\n'
        result = await channel.send_raw_request(data, timeout=30)

        writer.write.assert_called_once_with(data)
        writer.drain.assert_awaited_once()
        assert isinstance(result, FileWriteAckMessage)
        assert result.bytes_written == 100

    async def test_does_not_modify_data(self) -> None:
        """send_raw_request sends the exact bytes provided (no re-encoding)."""
        channel, reader, writer = _mock_tcp_channel()

        ack = FileWriteAckMessage(path="x.bin", bytes_written=5)
        reader.readuntil = AsyncMock(return_value=(ack.model_dump_json() + "\n").encode())

        raw = b'{"action":"write_file","path":"x.bin","content_base64":"AQID","make_executable":true}\n'
        await channel.send_raw_request(raw, timeout=10)

        # Verify exact bytes were written
        assert writer.write.call_args[0][0] is raw


# ============================================================================
# TcpChannel.send_raw_request() — Edge / Out-of-bounds
# ============================================================================


class TestTcpSendRawRequestEdgeCases:
    """Error paths for TcpChannel.send_raw_request()."""

    async def test_not_connected_raises(self) -> None:
        """send_raw_request on unconnected channel raises RuntimeError."""
        channel = TcpChannel(host="127.0.0.1", port=5000)
        with pytest.raises(RuntimeError, match="not connected"):
            await channel.send_raw_request(b"data\n", timeout=10)

    async def test_timeout_does_not_reset_connection(self) -> None:
        """TimeoutError is re-raised without resetting reader/writer."""
        channel, reader, _ = _mock_tcp_channel()
        reader.readuntil = AsyncMock(side_effect=TimeoutError())

        with pytest.raises(TimeoutError):
            await channel.send_raw_request(b"data\n", timeout=1)

        # Connection should NOT be reset (keep-alive for retry)
        assert channel._reader is not None
        assert channel._writer is not None

    async def test_broken_pipe_resets_connection(self) -> None:
        """BrokenPipeError resets reader/writer to None."""
        channel, reader, _ = _mock_tcp_channel()
        reader.readuntil = AsyncMock(side_effect=BrokenPipeError())

        with pytest.raises(BrokenPipeError):
            await channel.send_raw_request(b"data\n", timeout=10)

        assert channel._reader is None
        assert channel._writer is None

    async def test_os_error_resets_connection(self) -> None:
        """OSError resets connection state."""
        channel, reader, _ = _mock_tcp_channel()
        reader.readuntil = AsyncMock(side_effect=OSError("socket closed"))

        with pytest.raises(OSError):
            await channel.send_raw_request(b"data\n", timeout=10)

        assert channel._reader is None
        assert channel._writer is None

    async def test_connection_error_resets_connection(self) -> None:
        """ConnectionError resets connection state."""
        channel, reader, _ = _mock_tcp_channel()
        reader.readuntil = AsyncMock(side_effect=ConnectionResetError())

        with pytest.raises(ConnectionResetError):
            await channel.send_raw_request(b"data\n", timeout=10)

        assert channel._reader is None
        assert channel._writer is None

    async def test_incomplete_read_resets_connection(self) -> None:
        """asyncio.IncompleteReadError resets connection state."""
        channel, reader, _ = _mock_tcp_channel()
        reader.readuntil = AsyncMock(side_effect=asyncio.IncompleteReadError(partial=b"", expected=100))

        with pytest.raises(asyncio.IncompleteReadError):
            await channel.send_raw_request(b"data\n", timeout=10)

        assert channel._reader is None
        assert channel._writer is None

    async def test_error_response_parsed(self) -> None:
        """StreamingErrorMessage is returned (not raised) by send_raw_request."""
        channel, reader, _ = _mock_tcp_channel()

        error = StreamingErrorMessage(
            error_type="validation_error",
            message="path traversal detected",
        )
        reader.readuntil = AsyncMock(return_value=(error.model_dump_json() + "\n").encode())

        result = await channel.send_raw_request(b'{"action":"write_file"}\n', timeout=10)
        assert isinstance(result, StreamingErrorMessage)
        assert result.error_type == "validation_error"


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

        ack = FileWriteAckMessage(path="test.txt", bytes_written=42)
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
# QemuVM.write_file() — JSON frame validity
# ============================================================================


class TestWriteFileJsonFrame:
    """Unit tests for the manual JSON frame built in QemuVM.write_file().

    Extracts the frame-building logic and validates it against the protocol,
    without requiring a VM.
    """

    @staticmethod
    def _build_frame(path: str, content: bytes = b"x", make_executable: bool = False) -> bytes:
        """Replicate the frame-building logic from QemuVM.write_file()."""
        content_b64 = base64.b64encode(content)
        return b"".join(
            [
                b'{"action":"write_file","path":',
                json.dumps(path).encode(),
                b',"content_base64":"',
                content_b64,
                b'","make_executable":',
                b"true" if make_executable else b"false",
                b"}\n",
            ]
        )

    def test_frame_is_valid_json(self) -> None:
        """The assembled frame is valid JSON (newline-delimited)."""
        frame = self._build_frame("test.txt", b"hello world")
        parsed = json.loads(frame.rstrip(b"\n"))
        assert parsed["action"] == "write_file"
        assert parsed["path"] == "test.txt"
        assert parsed["make_executable"] is False

    def test_frame_base64_roundtrips(self) -> None:
        """Base64 content in the frame decodes back to original bytes."""
        original = bytes(range(256))
        frame = self._build_frame("binary.bin", original)
        parsed = json.loads(frame)
        decoded = base64.b64decode(parsed["content_base64"])
        assert decoded == original

    def test_frame_make_executable_true(self) -> None:
        """make_executable=True emits JSON true."""
        frame = self._build_frame("run.sh", b"#!/bin/sh", make_executable=True)
        parsed = json.loads(frame)
        assert parsed["make_executable"] is True

    def test_frame_make_executable_false(self) -> None:
        """make_executable=False emits JSON false."""
        frame = self._build_frame("data.csv", b"a,b,c")
        parsed = json.loads(frame)
        assert parsed["make_executable"] is False

    def test_frame_matches_pydantic_schema(self) -> None:
        """The manual frame is parseable as a valid WriteFileRequest."""
        original = b"content payload"
        frame = self._build_frame("payload.bin", original, make_executable=True)
        parsed = json.loads(frame)

        # Validate against the Pydantic model
        req = WriteFileRequest(**parsed)
        assert req.action == "write_file"
        assert req.path == "payload.bin"
        assert req.make_executable is True
        assert base64.b64decode(req.content_base64) == original

    def test_frame_empty_content(self) -> None:
        """Empty content produces a valid frame with empty base64."""
        frame = self._build_frame("empty.txt", b"")
        parsed = json.loads(frame)
        assert parsed["content_base64"] == ""
        assert base64.b64decode(parsed["content_base64"]) == b""

    def test_frame_ends_with_newline(self) -> None:
        """Frame ends with exactly one newline (protocol requirement)."""
        frame = self._build_frame("test.txt", b"data")
        assert frame.endswith(b"\n")
        assert not frame.endswith(b"\n\n")


# ============================================================================
# QemuVM.write_file() — JSON escaping (weird cases)
# ============================================================================


class TestWriteFileJsonEscaping:
    """Test that json.dumps properly escapes special characters in path.

    These are the cases that would break if the frame were built via
    string concatenation instead of json.dumps().
    """

    # Reuse the frame builder from TestWriteFileJsonFrame
    _build_frame = staticmethod(TestWriteFileJsonFrame._build_frame)

    def test_double_quotes_in_path(self) -> None:
        """Path with double quotes is properly escaped."""
        frame = self._build_frame('my"file.txt')
        parsed = json.loads(frame)
        assert parsed["path"] == 'my"file.txt'
        WriteFileRequest(**parsed)  # Validates against schema

    def test_backslash_in_path(self) -> None:
        """Path with backslash is properly escaped."""
        frame = self._build_frame("dir\\file.txt")
        parsed = json.loads(frame)
        assert parsed["path"] == "dir\\file.txt"
        WriteFileRequest(**parsed)

    def test_newline_in_path(self) -> None:
        r"""Path with newline is escaped (would break \n-delimited protocol)."""
        frame = self._build_frame("line1\nline2.txt")
        # The frame itself must be a single line (newline-delimited protocol)
        lines = frame.split(b"\n")
        # Should be exactly 2 parts: the JSON frame and the empty string after final \n
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["path"] == "line1\nline2.txt"

    def test_tab_in_path(self) -> None:
        """Path with tab character is escaped."""
        frame = self._build_frame("col1\tcol2.tsv")
        parsed = json.loads(frame)
        assert parsed["path"] == "col1\tcol2.tsv"

    def test_unicode_in_path(self) -> None:
        """Path with Unicode characters round-trips correctly."""
        frame = self._build_frame("données/résultat.csv")
        parsed = json.loads(frame)
        assert parsed["path"] == "données/résultat.csv"

    def test_emoji_in_path(self) -> None:
        """Path with emoji round-trips correctly."""
        frame = self._build_frame("data/output_\U0001f680.json")
        parsed = json.loads(frame)
        assert parsed["path"] == "data/output_\U0001f680.json"

    def test_all_json_metacharacters(self) -> None:
        r"""Path with all JSON escape-worthy characters: " \ / \b \f \n \r \t."""
        nasty_path = 'a"b\\c/d\be\ff\ng\rh\ti'
        frame = self._build_frame(nasty_path)
        parsed = json.loads(frame)
        assert parsed["path"] == nasty_path


# ============================================================================
# QemuVM.write_file() — Large content (memory path)
# ============================================================================


class TestWriteFileMemoryPath:
    """Verify the base64 encoding path handles realistic content sizes."""

    def test_1mb_content_produces_valid_frame(self) -> None:
        """1MB content produces a valid, parseable frame."""
        content = b"\x00" * (1024 * 1024)
        frame = TestWriteFileJsonFrame._build_frame("big.bin", content)
        parsed = json.loads(frame)
        decoded = base64.b64decode(parsed["content_base64"])
        assert decoded == content

    def test_all_byte_values_roundtrip(self) -> None:
        """All 256 byte values survive the base64 round-trip in the frame."""
        content = bytes(range(256)) * 100  # 25.6KB with all byte values repeated
        frame = TestWriteFileJsonFrame._build_frame("allbytes.bin", content)
        parsed = json.loads(frame)
        decoded = base64.b64decode(parsed["content_base64"])
        assert decoded == content

    def test_base64_content_is_ascii_safe(self) -> None:
        """The base64 bytes in the frame are pure ASCII (safe for JSON string embedding)."""
        content = bytes(range(256))
        b64 = base64.b64encode(content)
        # Every byte in base64 output must be ASCII printable
        for byte in b64:
            assert 0x20 <= byte <= 0x7E, f"Non-ASCII byte in base64: {byte:#x}"

# pyright: reportPrivateUsage=false
# OperationInbox deliberately exposes no direct-enqueue/inspection surface in
# production; tests pre-fill and inspect via the private _queue instead.
"""Unit tests for Session._guard(), _resolve_content(), write_file streaming protocol,
FileOpDispatcher routing, DualPortChannel reconnection probe, and idle timer lifecycle.

Tests the orchestration layer WITHOUT requiring VMs. All VM/channel interactions
are mocked. Covers:
- _guard() lifecycle checks, lock serialization, idle timer suspension
  during operations, and fail-fast retirement of dead VMs
- _resolve_content() sync/async file reads, boundary validation
- WriteFileRequest header serialization: validity, escaping
- FileOpDispatcher message routing by op_id
- DualPortChannel._probe_guest_ready() reconnection probe
- Idle timer task lifecycle: orphan prevention, self-cancel, stress
"""

import asyncio
import base64
import io
import json
import os
import random
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from exec_sandbox import constants
from exec_sandbox.constants import FILE_TRANSFER_CHUNK_SIZE, MAX_FILE_PATH_LENGTH, MAX_FILE_SIZE_BYTES
from exec_sandbox.exceptions import (
    CommunicationError,
    CommunicationOutcomeUnknownError,
    SessionClosedError,
    VmPermanentError,
)
from exec_sandbox.guest_agent_protocol import (
    ExecutionCompleteMessage,
    FileChunkResponseMessage,
    FileListMessage,
    FileReadCompleteMessage,
    FileWriteAckMessage,
    OutputChunkMessage,
    PongMessage,
    StreamingErrorMessage,
    StreamingMessage,
    WriteFileRequest,
)
from exec_sandbox.guest_channel import (
    OP_QUEUE_DEPTH,
    DualPortChannel,
    FileOpDispatcher,
    OperationInbox,
    UnixSocketChannel,
)
from exec_sandbox.models import ExecutionResult, ExposedPort, FileInfo, TimingBreakdown
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.session import Session
from exec_sandbox.vm_types import VmState
from tests.conftest import make_destroy_mock

# ============================================================================
# Helpers — build a Session with mocked VM + VmManager
# ============================================================================


def _real_enqueue_registered(inbox: OperationInbox, _data: bytes) -> None:
    """Mirror DualPortChannel.enqueue_registered's two contract lines so mocks
    cannot drift on the ensure-open-before-mark ordering."""
    inbox.ensure_open()
    inbox.mark_command_sent()


def _make_exec_result(exit_code: int = 0) -> ExecutionResult:
    """Build a minimal ExecutionResult for mocks."""
    return ExecutionResult(
        stdout="",
        stderr="",
        exit_code=exit_code,
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
    mock_vm = AsyncMock(spec=QemuVM)
    mock_vm.vm_id = "test-vm-001"
    mock_vm.execute = AsyncMock(return_value=_make_exec_result())
    mock_vm.write_file = AsyncMock()
    mock_vm.read_file = AsyncMock(return_value=None)
    mock_vm.list_files = AsyncMock(return_value=[FileInfo(name="a.txt", is_dir=False, size=10)])
    mock_vm.exposed_ports = []
    mock_vm.state = VmState.READY
    mock_vm.process = MagicMock()
    mock_vm.process.returncode = None

    mock_vm_manager = AsyncMock()
    mock_vm_manager.destroy_vm = make_destroy_mock("confirmed")

    session = Session(
        vm=mock_vm,
        vm_manager=mock_vm_manager,
        idle_timeout_seconds=idle_timeout_seconds,
        default_timeout_seconds=default_timeout_seconds,
    )
    return session, mock_vm, mock_vm_manager


# ============================================================================
# exec() — timeout_seconds validation
# ============================================================================


class TestExecTimeoutValidation:
    """Tests for timeout_seconds validation in Session.exec()."""

    async def test_timeout_zero_rejected(self) -> None:
        """timeout_seconds=0 raises ValueError."""
        session, _, _ = _make_session()
        with pytest.raises(ValueError, match="timeout_seconds must be between 1 and 300"):
            await session.exec("print(1)", timeout_seconds=0)

    async def test_timeout_negative_rejected(self) -> None:
        """Negative timeout_seconds raises ValueError."""
        session, _, _ = _make_session()
        with pytest.raises(ValueError, match="timeout_seconds must be between 1 and 300"):
            await session.exec("print(1)", timeout_seconds=-1)

    async def test_timeout_exceeds_max_rejected(self) -> None:
        """timeout_seconds > 300 raises ValueError."""
        session, _, _ = _make_session()
        with pytest.raises(ValueError, match="timeout_seconds must be between 1 and 300"):
            await session.exec("print(1)", timeout_seconds=301)

    async def test_timeout_none_uses_default(self) -> None:
        """timeout_seconds=None uses the default from constructor."""
        session, mock_vm, _ = _make_session(default_timeout_seconds=42)
        await session.exec("print(1)")
        mock_vm.execute.assert_called_once()
        assert mock_vm.execute.call_args.kwargs["timeout_seconds"] == 42

    async def test_timeout_explicit_value_used(self) -> None:
        """Explicit timeout_seconds is passed through."""
        session, mock_vm, _ = _make_session(default_timeout_seconds=42)
        await session.exec("print(1)", timeout_seconds=10)
        mock_vm.execute.assert_called_once()
        assert mock_vm.execute.call_args.kwargs["timeout_seconds"] == 10


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

    @pytest.mark.parametrize("terminal_state", [VmState.DESTROYING, VmState.DESTROYED])
    async def test_exec_closes_when_process_exit_races_normal_result(self, terminal_state: VmState) -> None:
        """A normal terminal result cannot leave a dead session VM reusable.

        A concurrent destroy can publish DESTROYING or, if it completes before
        execute() returns, DESTROYED — both must retire the session.
        """
        session, vm, manager = _make_session()

        async def complete_while_dying(**_kwargs: object) -> ExecutionResult:
            vm.state = terminal_state
            return _make_exec_result()

        vm.execute.side_effect = complete_while_dying
        result = await session.exec("print(1)")

        assert result.exit_code == 0
        assert session.closed
        manager.destroy_vm.assert_awaited_once_with(vm)
        with pytest.raises(SessionClosedError):
            await session.exec("print(2)")

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

    async def test_bytes_returns_bytesio_owned(self) -> None:
        """Bytes content is wrapped in BytesIO, owned=True."""
        session, _, _ = _make_session()
        stream, owned = await session._resolve_content(b"hello")
        assert owned is True
        assert stream.read() == b"hello"
        stream.close()

    async def test_path_returns_file_handle_owned(self, tmp_path: Path) -> None:
        """Path returns an open file handle (streams from disk), owned=True."""
        f = tmp_path / "small.txt"
        f.write_bytes(b"small content")
        session, _, _ = _make_session()
        stream, owned = await session._resolve_content(f)
        assert owned is True
        assert stream.read() == b"small content"
        stream.close()

    async def test_path_large_file_streams(self, tmp_path: Path) -> None:
        """Large file returns a file handle (never loaded fully by _resolve_content)."""
        f = tmp_path / "large.bin"
        content = b"x" * (2 * 1024 * 1024)  # 2MB
        f.write_bytes(content)

        session, _, _ = _make_session()
        stream, owned = await session._resolve_content(f)
        assert owned is True
        assert stream.read() == content
        stream.close()

    async def test_io_bytes_returns_same_buf_not_owned(self) -> None:
        """IO[bytes] input returns the same buffer, owned=False."""
        session, _, _ = _make_session()
        buf = io.BytesIO(b"hello")
        stream, owned = await session._resolve_content(buf)
        assert owned is False
        assert stream is buf
        assert stream.read() == b"hello"

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

        async def spied_resolve(content: bytes | Path | IO[bytes]) -> tuple[IO[bytes], bool]:
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
        stream, owned = await session._resolve_content(content)
        assert owned is True
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
        stream, owned = await session._resolve_content(f)
        assert owned is True
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
        stream, owned = await session._resolve_content(b"")
        assert owned is True
        assert stream.read() == b""
        stream.close()

    async def test_empty_file_accepted(self, tmp_path: Path) -> None:
        """Empty file (0 bytes) is accepted."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        session, _, _ = _make_session()
        stream, owned = await session._resolve_content(f)
        assert owned is True
        assert stream.read() == b""
        stream.close()

    async def test_seekable_io_bytes_at_max_size_accepted(self) -> None:
        """Seekable IO[bytes] of exactly MAX_FILE_SIZE_BYTES is accepted."""
        session, _, _ = _make_session()
        buf = io.BytesIO(b"x" * MAX_FILE_SIZE_BYTES)
        stream, owned = await session._resolve_content(buf)
        assert owned is False
        assert stream is buf

    async def test_seekable_io_bytes_over_max_rejected(self) -> None:
        """Seekable IO[bytes] exceeding MAX_FILE_SIZE_BYTES raises ValueError."""
        session, _, _ = _make_session()
        buf = io.BytesIO(b"x" * (MAX_FILE_SIZE_BYTES + 1))
        with pytest.raises(ValueError, match="exceeds"):
            await session._resolve_content(buf)

    async def test_non_seekable_io_bytes_skips_validation(self) -> None:
        """Non-seekable IO[bytes] skips size validation."""
        from tests.conftest import NonSeekableIO

        session, _, _ = _make_session()
        raw = NonSeekableIO(b"x" * 100)
        buf = io.BufferedReader(raw)
        stream, owned = await session._resolve_content(buf)  # type: ignore[arg-type]
        assert owned is False
        assert stream is buf

    async def test_seekable_io_bytes_preserves_position(self) -> None:
        """Size validation restores the original stream position."""
        session, _, _ = _make_session()
        buf = io.BytesIO(b"prefix" + b"data")
        buf.seek(6)  # Position past "prefix"
        _, owned = await session._resolve_content(buf)
        assert owned is False
        assert buf.tell() == 6  # Position preserved


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
        stream, owned = await session._resolve_content(f)
        assert owned is True
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
        op_queue = OperationInbox()
        op_queue._queue.put_nowait(ack)
        vm.channel.register_op = AsyncMock(return_value=op_queue)
        vm.channel.enqueue_registered = AsyncMock(side_effect=_real_enqueue_registered)

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

    async def test_retry_restores_the_callers_initial_stream_position(self) -> None:
        """A stale-header retry resends exactly the caller-selected suffix."""
        from exec_sandbox.qemu_vm import QemuVM

        content = io.BytesIO(b"prefix-payload")
        content.seek(len(b"prefix-"))
        positions: list[int] = []

        async def protocol(*_args: object, **_kwargs: object) -> None:
            positions.append(content.tell())
            content.read(1)
            if len(positions) == 1:
                raise VmPermanentError("No active write")

        vm = MagicMock(spec=QemuVM)
        vm.vm_id = "retry-position-vm"
        vm.channel = AsyncMock()
        vm._write_file_protocol = AsyncMock(side_effect=protocol)

        await QemuVM.write_file(vm, "target.bin", content)

        assert positions == [len(b"prefix-"), len(b"prefix-")]


class TestWriteFileCommitProtocol:
    """A file_end frame is a commit record, never best-effort cleanup."""

    @staticmethod
    def _make_vm(response: StreamingMessage | None = None) -> tuple[MagicMock, OperationInbox]:
        from exec_sandbox.qemu_vm import QemuVM

        vm = MagicMock(spec=QemuVM)
        vm.vm_id = "write-commit-vm"
        vm.channel = AsyncMock()
        vm.channel.connect = AsyncMock()
        vm.channel.enqueue_raw = AsyncMock()
        vm.channel.unregister_op = AsyncMock()
        inbox = OperationInbox()
        if response is not None:
            inbox._queue.put_nowait(response)
        vm.channel.register_op = AsyncMock(return_value=inbox)
        vm.channel.enqueue_registered = AsyncMock(side_effect=_real_enqueue_registered)
        return vm, inbox

    async def test_source_failure_after_prefix_never_sends_file_end(self) -> None:
        from exec_sandbox.qemu_vm import QemuVM

        class FailingReader:
            reads = 0

            def read(self, _size: int) -> bytes:
                self.reads += 1
                if self.reads == 1:
                    return b"prefix"
                raise OSError("injected source failure")

        vm, _inbox = self._make_vm()
        with pytest.raises(CommunicationOutcomeUnknownError, match="after dispatch"):
            await QemuVM._write_file_protocol(vm, "target.bin", FailingReader())  # type: ignore[arg-type]

        actions = [json.loads(call.args[0])["action"] for call in vm.channel.enqueue_raw.await_args_list]
        assert "file_end" not in actions
        vm.channel.unregister_op.assert_awaited_once()

    async def test_failed_chunk_enqueue_never_starts_the_next_source_read(self) -> None:
        """A failed enqueue cannot return while look-ahead still owns the stream."""
        from exec_sandbox.qemu_vm import QemuVM

        class CountingReader:
            reads = 0

            def read(self, _size: int) -> bytes:
                self.reads += 1
                if self.reads == 1:
                    return os.urandom(FILE_TRANSFER_CHUNK_SIZE)
                return b""

        reader = CountingReader()
        vm, _inbox = self._make_vm()
        vm.channel.enqueue_raw = AsyncMock(side_effect=OSError("injected enqueue failure"))

        with pytest.raises(CommunicationOutcomeUnknownError, match="after dispatch"):
            await QemuVM._write_file_protocol(vm, "target.bin", reader)  # type: ignore[arg-type]

        assert reader.reads == 1

    async def test_prefailed_registration_is_reported_before_dispatch(self) -> None:
        """A dead registered inbox proves that no header reached the write queue."""
        from exec_sandbox.exceptions import VmTransientError
        from exec_sandbox.qemu_vm import QemuVM

        vm, inbox = self._make_vm()
        inbox.fail("event transport already closed")
        vm.channel.enqueue_registered = AsyncMock(side_effect=_real_enqueue_registered)

        with pytest.raises(VmTransientError, match="before dispatch"):
            await QemuVM._write_file_protocol(vm, "target.bin", io.BytesIO(b"payload"))

        vm.channel.enqueue_raw.assert_not_awaited()

    async def test_ack_timeout_after_commit_is_outcome_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from exec_sandbox.qemu_vm import QemuVM

        monkeypatch.setattr(constants, "FILE_IO_TIMEOUT_SECONDS", 0.01)
        vm, _inbox = self._make_vm()
        with pytest.raises(CommunicationOutcomeUnknownError, match="after dispatch") as exc_info:
            await QemuVM._write_file_protocol(vm, "target.bin", io.BytesIO(b"payload"))

        assert isinstance(exc_info.value.__cause__, TimeoutError)
        actions = [json.loads(call.args[0])["action"] for call in vm.channel.enqueue_raw.await_args_list]
        assert actions[-1] == "file_end"

    @pytest.mark.parametrize(
        ("ack_path", "ack_size"),
        [("different.bin", 7), ("target.bin", 6)],
    )
    async def test_ack_must_bind_path_and_source_size(self, ack_path: str, ack_size: int) -> None:
        from exec_sandbox.qemu_vm import QemuVM

        ack = FileWriteAckMessage(op_id="write-op", path=ack_path, bytes_written=ack_size)
        vm, _inbox = self._make_vm(ack)
        with pytest.raises(CommunicationOutcomeUnknownError, match="acknowledgement did not bind"):
            await QemuVM._write_file_protocol(vm, "target.bin", io.BytesIO(b"payload"))


class TestReadFileTerminalIntegrity:
    """File-read completion metadata must bind the bytes actually received."""

    async def test_declared_size_mismatch_is_communication_failure(self) -> None:
        from exec_sandbox.qemu_vm import QemuVM

        vm = MagicMock(spec=QemuVM)
        vm.vm_id = "test-vm"
        vm.channel = AsyncMock()
        inbox = OperationInbox()
        inbox._queue.put_nowait(FileReadCompleteMessage(op_id="read-op", path="file.bin", size=1))
        vm.channel.register_op = AsyncMock(return_value=inbox)
        vm.channel.enqueue_registered = AsyncMock(side_effect=_real_enqueue_registered)

        with pytest.raises(CommunicationError, match="declared_size=1, received_size=0"):
            await QemuVM._recv_file_chunks(vm, "file.bin", "read-op", io.BytesIO())

    @staticmethod
    def _file_read_vm(payload: bytes) -> MagicMock:
        from exec_sandbox.qemu_vm import QemuVM, zstd

        vm = MagicMock(spec=QemuVM)
        vm.vm_id = "test-vm"
        vm.channel = AsyncMock()
        inbox = OperationInbox()
        compressor = zstd.ZstdCompressor(level=constants.FILE_TRANSFER_ZSTD_LEVEL)
        compressed = compressor.compress(payload) + compressor.flush()
        inbox._queue.put_nowait(
            FileChunkResponseMessage(op_id="read-op", data=base64.b64encode(compressed).decode("ascii"))
        )
        inbox._queue.put_nowait(FileReadCompleteMessage(op_id="read-op", path="file.bin", size=len(payload)))
        vm.channel.register_op = AsyncMock(return_value=inbox)
        vm.channel.enqueue_registered = AsyncMock(side_effect=_real_enqueue_registered)
        return vm

    async def test_short_destination_writes_are_completed(self) -> None:
        from exec_sandbox.qemu_vm import QemuVM

        class ShortWriter:
            def __init__(self) -> None:
                self.value = bytearray()

            def write(self, data: bytes) -> int:
                count = max(1, len(data) // 2)
                self.value.extend(data[:count])
                return count

        payload = b"short writes still preserve every byte"
        writer = ShortWriter()
        vm = self._file_read_vm(payload)
        await QemuVM._recv_file_chunks(vm, "file.bin", "read-op", writer)  # type: ignore[arg-type]
        assert bytes(writer.value) == payload

    async def test_zero_progress_destination_write_is_rejected(self) -> None:
        from exec_sandbox.qemu_vm import QemuVM

        class ZeroWriter:
            def write(self, _data: bytes) -> int:
                return 0

        vm = self._file_read_vm(b"payload")
        with pytest.raises(OSError, match="no write progress"):
            await QemuVM._recv_file_chunks(vm, "file.bin", "read-op", ZeroWriter())  # type: ignore[arg-type]

    async def test_none_returning_destination_write_is_rejected(self) -> None:
        """A raw non-blocking writer returns None on would-block — OSError, not TypeError."""
        from exec_sandbox.qemu_vm import QemuVM

        class NoneWriter:
            def write(self, _data: bytes) -> None:
                return None

        vm = self._file_read_vm(b"payload")
        with pytest.raises(OSError, match="no write progress"):
            await QemuVM._recv_file_chunks(vm, "file.bin", "read-op", NoneWriter())  # type: ignore[arg-type]

    async def test_over_reporting_destination_write_is_rejected(self) -> None:
        """A writer claiming more progress than requested must not report success."""
        from exec_sandbox.qemu_vm import QemuVM

        class OverWriter:
            def write(self, data: bytes) -> int:
                return len(data) + 1

        vm = self._file_read_vm(b"payload")
        with pytest.raises(OSError, match="no write progress"):
            await QemuVM._recv_file_chunks(vm, "file.bin", "read-op", OverWriter())  # type: ignore[arg-type]


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


async def _drain(queue: OperationInbox, timeout: float = 0.5) -> list[StreamingMessage]:
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

        await dispatcher.stop()

    async def test_error_without_op_id_is_discarded(self) -> None:
        """Error without op_id is discarded (no default queue)."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-A")
        # Error without op_id — discarded
        _feed(reader, {"type": "error", "error_type": "timeout", "message": "fail"})
        # Subsequent message with op_id — routed correctly
        _feed(reader, {"type": "error", "op_id": "op-A", "error_type": "x", "message": "y"})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, StreamingErrorMessage)
        assert msg.op_id == "op-A"

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

        await dispatcher.stop()

    async def test_non_file_message_without_op_id_is_discarded(self) -> None:
        """Pong without op_id is discarded, doesn't pollute op queues."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-A")
        # Pong without op_id — discarded
        _feed(reader, {"type": "pong", "version": "1.0"})
        # Subsequent message with op_id — still routes correctly
        _feed(reader, {"type": "pong", "version": "1.0", "op_id": "op-A"})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, PongMessage)
        assert msg.op_id == "op-A"

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

        await dispatcher.stop()

    async def test_interleaved_writes_errors_dont_leak(self) -> None:
        """Write errors route to correct op queues, nothing leaks to default."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        q1 = await dispatcher.register_op("w1")
        q2 = await dispatcher.register_op("w2")
        q3 = await dispatcher.register_op("w3")

        _feed(reader, {"type": "error", "error_type": "path_error", "message": "path invalid", "op_id": "w1"})
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

        assert qa._queue.empty()

        await dispatcher.stop()

    async def test_empty_string_op_id_routes_to_op_queue(self) -> None:
        """Empty string op_id is a valid op_id and routes to its registered queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        q_empty = await dispatcher.register_op("")
        _feed(reader, {"type": "error", "error_type": "io_error", "message": "fail", "op_id": ""})

        msg = await asyncio.wait_for(q_empty.get(), timeout=1.0)
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

        await dispatcher.stop()

    # ------------------------------------------------------------------
    # Weird / out-of-bounds cases
    # ------------------------------------------------------------------

    async def test_pong_discarded_error_with_op_id_routed(self) -> None:
        """Pong without op_id is discarded, error with op_id routes to op queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-A")
        # Pong without op_id — discarded
        _feed(reader, {"type": "pong", "version": "1.0"})
        # Error with op_id — routed to op queue
        _feed(reader, {"type": "error", "op_id": "op-A", "error_type": "x", "message": "y"})

        err = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(err, StreamingErrorMessage)
        assert err.op_id == "op-A"

        await dispatcher.stop()

    async def test_100_ops_interleaved_all_route_correctly(self) -> None:
        """100 ops with shuffled messages all route to correct queues."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        queues: dict[str, OperationInbox] = {}
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

        await dispatcher.stop()


class TestDispatchLoopConnectionReset:
    """Tests that _dispatch_loop exits cleanly on OSError variants.

    The dispatch loop catches (IncompleteReadError, OSError) so that
    connection resets — normal during peer disconnect or balloon stress —
    are treated the same as EOF.

    IMPORTANT: after injecting the error, we await the task directly to
    confirm the loop exited via the except clause.  Calling stop() first
    would cancel the task, making the test exercise CancelledError instead.
    """

    # ------------------------------------------------------------------
    # Normal cases — parametrized over every error variant
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("error_cls", "error_args"),
        [
            pytest.param(ConnectionResetError, ("reset",), id="ConnectionResetError"),
            pytest.param(BrokenPipeError, ("broken",), id="BrokenPipeError"),
            pytest.param(OSError, (104, "Connection aborted"), id="OSError-custom-errno"),
        ],
    )
    async def test_loop_exits_cleanly_on_oserror(
        self, error_cls: type[OSError], error_args: tuple[object, ...]
    ) -> None:
        """Dispatch loop exits via except clause, task completes normally."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        await asyncio.sleep(0)  # let loop enter readuntil()

        reader.set_exception(error_cls(*error_args))

        # Wait for the loop to exit via the except clause (not cancellation)
        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)

        await dispatcher.stop()
        assert dispatcher._task is None

    async def test_loop_exits_cleanly_on_eof(self) -> None:
        """IncompleteReadError (EOF) exits cleanly — regression guard."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        await asyncio.sleep(0)  # let loop enter readuntil()

        reader.feed_eof()

        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)

        await dispatcher.stop()
        assert dispatcher._task is None

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    async def test_messages_before_reset_are_delivered(self) -> None:
        """Messages routed before ConnectionResetError are still in their queues."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-A")
        _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": "before-reset"})

        # Wait for message to be dispatched
        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, FileChunkResponseMessage)
        assert msg.data == "before-reset"

        # Now simulate reset — loop is back in readuntil()
        reader.set_exception(ConnectionResetError())
        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)

        await dispatcher.stop()
        assert dispatcher._task is None

    async def test_stop_after_loop_exit_is_idempotent(self) -> None:
        """Calling stop() twice after the loop already exited doesn't raise."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        await asyncio.sleep(0)  # let loop enter readuntil()

        reader.set_exception(ConnectionResetError())
        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)

        await dispatcher.stop()
        await dispatcher.stop()  # second stop — no-op
        assert dispatcher._task is None

    # ------------------------------------------------------------------
    # Weird / out-of-bounds cases
    # ------------------------------------------------------------------

    async def test_oserror_wakes_registered_queue_with_transport_failure(self) -> None:
        """An active operation receives a terminal transient transport event."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-A")

        reader.set_exception(ConnectionResetError())
        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)

        with pytest.raises(CommunicationError, match="ConnectionResetError"):
            await asyncio.wait_for(op_queue.get(), timeout=1.0)

        await dispatcher.stop()

    async def test_eof_wakes_full_queue_without_losing_buffered_messages(self) -> None:
        """Transport failure follows every buffered item when the op queue is full."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-full")
        for index in range(OP_QUEUE_DEPTH):
            op_queue._queue.put_nowait(OutputChunkMessage(type="stdout", chunk=str(index), op_id="op-full"))

        reader.feed_eof()
        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)

        messages = [await asyncio.wait_for(op_queue.get(), timeout=1.0) for _ in range(OP_QUEUE_DEPTH)]
        assert len(messages) == OP_QUEUE_DEPTH
        assert isinstance(messages[0], OutputChunkMessage)
        assert messages[0].chunk == "0"
        with pytest.raises(CommunicationError, match="EOF"):
            await op_queue.get()

        await dispatcher.stop()

    async def test_registration_after_eof_fails_immediately(self) -> None:
        """A race registering after dispatcher death cannot wait for a hard timeout."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        reader.feed_eof()
        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)

        op_queue = await dispatcher.register_op("late-op")
        with pytest.raises(CommunicationError, match="EOF"):
            op_queue.ensure_open()

        await dispatcher.stop()

    async def test_full_queue_preserves_buffered_completion_before_transport_failure(self) -> None:
        """EOF cannot turn a complete valid stream into truncated success."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-complete")
        for index in range(OP_QUEUE_DEPTH - 1):
            op_queue._queue.put_nowait(OutputChunkMessage(type="stdout", chunk=str(index), op_id="op-complete"))
        op_queue._queue.put_nowait(ExecutionCompleteMessage(exit_code=0, execution_time_ms=1, op_id="op-complete"))

        reader.feed_eof()
        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)

        messages = [await asyncio.wait_for(op_queue.get(), timeout=1.0) for _ in range(OP_QUEUE_DEPTH)]
        assert [message.chunk for message in messages[:-1] if isinstance(message, OutputChunkMessage)] == [
            str(index) for index in range(OP_QUEUE_DEPTH - 1)
        ]
        assert isinstance(messages[-1], ExecutionCompleteMessage)
        with pytest.raises(CommunicationError, match="EOF"):
            await op_queue.get()

        await dispatcher.stop()

    async def test_real_dispatch_backpressure_preserves_all_data_before_eof(self) -> None:
        """The 65th routed message waits for its consumer instead of failing."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        op_queue = await dispatcher.register_op("overflow")

        for index in range(OP_QUEUE_DEPTH + 1):
            _feed(reader, {"type": "stdout", "chunk": str(index), "op_id": "overflow"})
        reader.feed_eof()

        messages = [await asyncio.wait_for(op_queue.get(), timeout=1.0) for _ in range(OP_QUEUE_DEPTH + 1)]
        assert [message.chunk for message in messages if isinstance(message, OutputChunkMessage)] == [
            str(index) for index in range(OP_QUEUE_DEPTH + 1)
        ]
        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)
        with pytest.raises(CommunicationError, match="EOF"):
            await op_queue.get()
        await dispatcher.stop()

    async def test_unregister_full_queue_releases_dispatcher_for_later_operation(self) -> None:
        """A cancelled full inbox cannot wedge the one global dispatch task."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        abandoned = await dispatcher.register_op("abandoned")

        for index in range(OP_QUEUE_DEPTH + 1):
            _feed(reader, {"type": "stdout", "chunk": str(index), "op_id": "abandoned"})
        await asyncio.sleep(0)  # dispatcher fills 64 slots, then blocks routing the 65th
        assert abandoned._queue.full()
        await dispatcher.unregister_op("abandoned")

        later = await dispatcher.register_op("later")
        _feed(reader, {"type": "complete", "exit_code": 0, "execution_time_ms": 1, "op_id": "later"})
        message = await asyncio.wait_for(later.get(), timeout=1.0)
        assert isinstance(message, ExecutionCompleteMessage)

        reader.feed_eof()
        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)
        await dispatcher.stop()

    async def test_qemu_exit_drains_reader_buffered_terminal_before_failure(self) -> None:
        """Process death cannot overtake a terminal frame accepted by the reader."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        inbox = await dispatcher.register_op("terminal")
        inbox.mark_command_sent()
        channel = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
        channel._dispatcher = dispatcher

        _feed(reader, {"type": "complete", "exit_code": 137, "execution_time_ms": 321, "op_id": "terminal"})
        reader.feed_eof()
        await channel.fail_pending_operations("QEMU exited with code 0")

        message = await asyncio.wait_for(inbox.get(), timeout=1.0)
        assert isinstance(message, ExecutionCompleteMessage)
        assert message.exit_code == 137
        with pytest.raises(CommunicationOutcomeUnknownError, match="EOF"):
            await inbox.get()
        await dispatcher.stop()

    async def test_qemu_exit_with_full_queue_preserves_late_terminal(self) -> None:
        """Backpressure plus peer exit cannot overtake a buffered terminal."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        inbox = await dispatcher.register_op("terminal-after-overflow")
        inbox.mark_command_sent()
        channel = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
        channel._dispatcher = dispatcher

        for index in range(OP_QUEUE_DEPTH + 1):
            _feed(
                reader,
                {
                    "type": "stdout",
                    "chunk": str(index),
                    "op_id": "terminal-after-overflow",
                },
            )
        _feed(
            reader,
            {
                "type": "complete",
                "exit_code": 137,
                "execution_time_ms": 321,
                "op_id": "terminal-after-overflow",
            },
        )
        reader.feed_eof()

        failure = asyncio.create_task(channel.fail_pending_operations("QEMU exited with code 0"))
        messages = [await asyncio.wait_for(inbox.get(), timeout=1.0) for _ in range(OP_QUEUE_DEPTH + 2)]
        await asyncio.wait_for(failure, timeout=1.0)

        assert [message.chunk for message in messages[:-1] if isinstance(message, OutputChunkMessage)] == [
            str(index) for index in range(OP_QUEUE_DEPTH + 1)
        ]
        assert isinstance(messages[-1], ExecutionCompleteMessage)
        assert messages[-1].exit_code == 137
        with pytest.raises(CommunicationOutcomeUnknownError, match="EOF"):
            await inbox.get()
        await dispatcher.stop()

    async def test_total_deadline_during_registration_is_predispatch(self) -> None:
        """The total timer includes registration without inventing dispatch."""
        from exec_sandbox.guest_agent_protocol import PingRequest
        from exec_sandbox.guest_channel import StreamDeadlineExceededError

        channel = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
        channel._dispatcher = MagicMock()

        async def blocked_registration(_op_id: str) -> OperationInbox:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        channel.register_op = blocked_registration  # type: ignore[method-assign]
        stream = channel.stream_messages(PingRequest(), timeout=0)
        with pytest.raises(StreamDeadlineExceededError) as exc_info:
            await anext(stream)
        assert exc_info.value.command_dispatched is False

    async def test_transport_loss_after_dispatch_is_outcome_unknown(self) -> None:
        """Host transport loss is never fabricated as a retryable guest error."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        op_queue = await dispatcher.register_op("sent")
        op_queue.mark_command_sent()
        reader.feed_eof()

        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)
        with pytest.raises(CommunicationOutcomeUnknownError, match="EOF"):
            await op_queue.get()
        await dispatcher.stop()

    async def test_limit_overrun_wakes_ops_and_stop_remains_safe(self) -> None:
        """An oversized event frame fails active ops and cannot poison close()."""
        reader = asyncio.StreamReader(limit=16)
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        op_queue = await dispatcher.register_op("oversized")
        reader.feed_data(b"x" * 64 + b"\n")

        assert dispatcher._task is not None
        await asyncio.wait_for(dispatcher._task, timeout=1.0)
        with pytest.raises(CommunicationError, match="reader limit"):
            await op_queue.get()
        await dispatcher.stop()


# ============================================================================
# DualPortChannel.close() resiliency
# ============================================================================


def _make_close_channel(
    *,
    has_dispatcher: bool = True,
    stop_side_effect: BaseException | None = None,
) -> DualPortChannel:
    """Build a DualPortChannel with mocked internals for close() testing."""
    ch = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
    if has_dispatcher:
        mock_disp = AsyncMock(spec=FileOpDispatcher)
        if stop_side_effect is not None:
            mock_disp.stop = AsyncMock(side_effect=stop_side_effect)
        ch._dispatcher = mock_disp
    ch._cmd_channel = AsyncMock(spec=UnixSocketChannel)
    ch._event_channel = AsyncMock(spec=UnixSocketChannel)
    return ch


class TestDualPortChannelCloseResiliency:
    """Tests that DualPortChannel.close() always clears _dispatcher.

    Uses try/finally so that _dispatcher = None runs even if stop() raises,
    preventing a poisoned dispatcher from blocking subsequent health check retries.
    """

    # ------------------------------------------------------------------
    # Normal cases
    # ------------------------------------------------------------------

    async def test_close_clears_dispatcher_and_closes_channels(self) -> None:
        """Normal close: dispatcher cleared, both channels closed."""
        ch = _make_close_channel()

        await ch.close()

        assert ch._dispatcher is None
        ch._cmd_channel.close.assert_awaited_once()
        ch._event_channel.close.assert_awaited_once()

    async def test_close_with_no_dispatcher_still_closes_channels(self) -> None:
        """close() when _dispatcher is None still closes channels."""
        ch = _make_close_channel(has_dispatcher=False)
        assert ch._dispatcher is None

        await ch.close()

        assert ch._dispatcher is None
        ch._cmd_channel.close.assert_awaited_once()
        ch._event_channel.close.assert_awaited_once()

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    async def test_close_clears_dispatcher_when_stop_raises(self) -> None:
        """If stop() raises RuntimeError, _dispatcher is still cleared to None."""
        ch = _make_close_channel(stop_side_effect=RuntimeError("dispatch task exploded"))

        with pytest.raises(RuntimeError, match="dispatch task exploded"):
            await ch.close()

        # Critical: dispatcher must be cleared despite the error
        assert ch._dispatcher is None

    async def test_close_idempotent(self) -> None:
        """Calling close() twice is safe — second call is a no-op for dispatcher."""
        ch = _make_close_channel()

        await ch.close()
        await ch.close()  # second call — _dispatcher already None

        assert ch._dispatcher is None

    # ------------------------------------------------------------------
    # Weird / out-of-bounds cases
    # ------------------------------------------------------------------

    async def test_close_clears_dispatcher_on_base_exception(self) -> None:
        """Even BaseException (e.g. KeyboardInterrupt) from stop() clears _dispatcher."""
        ch = _make_close_channel(stop_side_effect=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            await ch.close()

        assert ch._dispatcher is None


# ============================================================================
# exposed_ports property — delegates to VM
# ============================================================================


class TestExposedPorts:
    """Tests for Session.exposed_ports property."""

    async def test_exposed_ports_delegates_to_vm(self) -> None:
        """exposed_ports returns the VM's exposed_ports list."""
        session, mock_vm, _ = _make_session()
        mock_vm.exposed_ports = [ExposedPort(internal=8080, external=3000)]
        assert session.exposed_ports == [ExposedPort(internal=8080, external=3000)]

    async def test_exposed_ports_empty_when_no_ports(self) -> None:
        """exposed_ports returns empty list when VM has no exposed ports."""
        session, mock_vm, _ = _make_session()
        mock_vm.exposed_ports = []
        assert session.exposed_ports == []

    async def test_exposed_ports_accessible_after_close(self) -> None:
        """exposed_ports is still accessible after session is closed."""
        session, mock_vm, _ = _make_session()
        mock_vm.exposed_ports = [ExposedPort(internal=8080, external=3000)]
        await session.close()
        assert session.exposed_ports == [ExposedPort(internal=8080, external=3000)]


# ============================================================================
# DualPortChannel reconnection probe tests
# ============================================================================
#
# Tests the _probe_guest_ready behaviour: observable outcomes (returns vs
# raises) given a simulated guest that is ready, transiently unresponsive,
# or permanently down.  No mock-interaction assertions — type-checking
# already validates the interface contracts.
# ============================================================================


def _make_probe_channel(
    send_request_side_effect: object,
) -> DualPortChannel:
    """Build a DualPortChannel with a scripted guest response sequence.

    Pass a list where each entry is either a message the guest returns
    or an exception that simulates a transport failure.  The probe's
    retry loop consumes one entry per attempt.
    """
    channel = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
    channel.send_request = AsyncMock(side_effect=send_request_side_effect)
    channel._raw_connect = AsyncMock()
    channel.close = AsyncMock()
    return channel


class TestDualPortConnectLifecycle:
    """Tests for connect() orchestration: when probing happens (and when it doesn't)."""

    async def test_first_connect_does_not_probe(self) -> None:
        """Boot-time connect has zero probe overhead."""
        channel = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
        channel._raw_connect = AsyncMock()
        channel._probe_guest_ready = AsyncMock()

        await channel.connect(timeout_seconds=5.0)

        assert channel._has_been_connected is True
        # Probe should not have fired — this is the boot path
        channel._probe_guest_ready.assert_not_awaited()

    async def test_reconnect_does_probe(self) -> None:
        """After a successful connect, subsequent connects probe the guest."""
        channel = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
        channel._raw_connect = AsyncMock()
        channel._probe_guest_ready = AsyncMock()
        channel.close = AsyncMock()

        await channel.connect(timeout_seconds=5.0)  # first
        await channel.connect(timeout_seconds=5.0)  # second — should probe

        channel._probe_guest_ready.assert_awaited_once()

    async def test_idempotent_when_already_connected(self) -> None:
        """connect() on an already-connected channel is a no-op."""
        channel = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
        channel._raw_connect = AsyncMock()
        channel._cmd_channel.is_connected = MagicMock(return_value=True)
        channel._event_channel.is_connected = MagicMock(return_value=True)

        await channel.connect(timeout_seconds=5.0)

        # _raw_connect should not have been called — already connected
        channel._raw_connect.assert_not_awaited()

    async def test_has_been_connected_survives_close(self) -> None:
        """close() does not reset the flag, so next connect will always probe."""
        channel = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
        channel._raw_connect = AsyncMock()
        channel._probe_guest_ready = AsyncMock()
        channel.close = AsyncMock()

        await channel.connect(timeout_seconds=5.0)
        await channel.close()

        assert channel._has_been_connected is True


class TestDualPortProbeSuccess:
    """Probe returns (doesn't raise) when the guest eventually responds."""

    async def test_guest_ready_immediately(self) -> None:
        """Guest responds with PongMessage on first attempt — probe returns."""
        channel = _make_probe_channel([PongMessage(version="1.0")])
        # Should not raise
        await channel._probe_guest_ready(caller_timeout=5.0)

    async def test_guest_ready_after_transient_timeouts(self) -> None:
        """Guest unresponsive twice then ready — probe returns."""
        channel = _make_probe_channel(
            [
                TimeoutError(),
                TimeoutError(),
                PongMessage(version="1.0"),
            ]
        )
        await channel._probe_guest_ready(caller_timeout=5.0)

    async def test_guest_ready_after_mixed_transport_errors(self) -> None:
        """Different transport failures then success — probe is resilient."""
        channel = _make_probe_channel(
            [
                TimeoutError(),
                OSError("reset"),
                ConnectionError("refused"),
                asyncio.IncompleteReadError(b"", 100),
                PongMessage(version="1.0"),
            ]
        )
        await channel._probe_guest_ready(caller_timeout=5.0)

    async def test_guest_ready_on_last_possible_attempt(self) -> None:
        """Guest responds on the very last retry — boundary case."""
        max_retries = constants.GUEST_RECONNECT_PROBE_MAX_RETRIES
        channel = _make_probe_channel([TimeoutError()] * (max_retries - 1) + [PongMessage(version="1.0")])
        # Should succeed, not raise
        await channel._probe_guest_ready(caller_timeout=5.0)


class TestDualPortProbeFailure:
    """Probe raises TimeoutError when the guest never becomes ready."""

    async def test_all_attempts_timeout(self) -> None:
        """Guest never responds — TimeoutError with descriptive message."""
        max_retries = constants.GUEST_RECONNECT_PROBE_MAX_RETRIES
        channel = _make_probe_channel([TimeoutError()] * max_retries)

        with pytest.raises(TimeoutError, match=f"after {max_retries} reconnection attempts"):
            await channel._probe_guest_ready(caller_timeout=5.0)

    async def test_all_attempts_connection_error(self) -> None:
        """Guest socket broken every time — same TimeoutError."""
        max_retries = constants.GUEST_RECONNECT_PROBE_MAX_RETRIES
        channel = _make_probe_channel([ConnectionError()] * max_retries)

        with pytest.raises(TimeoutError, match="reconnection attempts"):
            await channel._probe_guest_ready(caller_timeout=5.0)

    async def test_guest_returns_wrong_message_type(self) -> None:
        """Guest returns StreamingErrorMessage instead of PongMessage — treated as failure."""
        max_retries = constants.GUEST_RECONNECT_PROBE_MAX_RETRIES
        wrong_msg = StreamingErrorMessage(error_type="unknown", message="not a pong")
        channel = _make_probe_channel([wrong_msg] * max_retries)

        with pytest.raises(TimeoutError, match="reconnection attempts"):
            await channel._probe_guest_ready(caller_timeout=5.0)

    async def test_one_more_failure_than_budget_still_raises(self) -> None:
        """MAX_RETRIES-1 failures + wrong message type = still raises."""
        max_retries = constants.GUEST_RECONNECT_PROBE_MAX_RETRIES
        wrong_msg = StreamingErrorMessage(error_type="unknown", message="bad")
        channel = _make_probe_channel([TimeoutError()] * (max_retries - 1) + [wrong_msg])

        with pytest.raises(TimeoutError):
            await channel._probe_guest_ready(caller_timeout=5.0)


# ============================================================================
# op_id routing — stale message isolation
# ============================================================================


class TestOpIdDispatcherRouting:
    """Tests op_id-based message routing through the dispatcher.

    Verifies that messages with op_id are routed to per-op queues,
    and stale messages (unregistered op_id) are discarded.
    """

    async def test_pong_with_op_id_routes_to_op_queue(self) -> None:
        """PongMessage with op_id routes to registered op queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("ping-1")
        _feed(reader, {"type": "pong", "version": "1.0", "op_id": "ping-1"})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, PongMessage)
        assert msg.op_id == "ping-1"

        await dispatcher.stop()

    async def test_complete_with_op_id_routes_to_op_queue(self) -> None:
        """ExecutionCompleteMessage with op_id routes to registered op queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("exec-1")
        _feed(reader, {"type": "complete", "exit_code": 0, "execution_time_ms": 100, "op_id": "exec-1"})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, ExecutionCompleteMessage)
        assert msg.op_id == "exec-1"

        await dispatcher.stop()

    async def test_stdout_with_op_id_routes_to_op_queue(self) -> None:
        """OutputChunkMessage with op_id routes to registered op queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("exec-1")
        _feed(reader, {"type": "stdout", "chunk": "hello", "op_id": "exec-1"})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, OutputChunkMessage)
        assert msg.op_id == "exec-1"

        await dispatcher.stop()

    async def test_file_list_with_op_id_routes_to_op_queue(self) -> None:
        """FileListMessage with op_id routes to registered op queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("list-1")
        _feed(reader, {"type": "file_list", "path": "", "entries": [], "op_id": "list-1"})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, FileListMessage)
        assert msg.op_id == "list-1"

        await dispatcher.stop()

    async def test_stale_pong_discarded(self) -> None:
        """Stale PongMessage (unregistered op_id) is discarded, not queued."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        # Feed a stale PongMessage from a previous probe — no queue registered
        _feed(reader, {"type": "pong", "version": "1.0", "op_id": "probe-1"})

        # Now register a queue for list_files and feed its response
        op_queue = await dispatcher.register_op("list-1")
        _feed(reader, {"type": "file_list", "path": "", "entries": [], "op_id": "list-1"})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, FileListMessage)
        assert msg.op_id == "list-1"

        # Default queue empty — stale pong was discarded

        await dispatcher.stop()

    async def test_message_without_op_id_is_discarded(self) -> None:
        """Message without op_id is discarded (no default queue)."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("probe-1")
        # Pong without op_id — discarded
        _feed(reader, {"type": "pong", "version": "1.0"})
        # Subsequent op_id message — still routes
        _feed(reader, {"type": "pong", "version": "1.0", "op_id": "probe-1"})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, PongMessage)
        assert msg.op_id == "probe-1"

        await dispatcher.stop()

    async def test_interleaved_exec_and_list(self) -> None:
        """Two ops interleaved: exec and list_files, each gets its own messages."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        exec_queue = await dispatcher.register_op("exec-1")
        list_queue = await dispatcher.register_op("list-1")

        # Interleave messages
        _feed(reader, {"type": "stdout", "chunk": "hello", "op_id": "exec-1"})
        _feed(reader, {"type": "file_list", "path": "", "entries": [], "op_id": "list-1"})
        _feed(reader, {"type": "stderr", "chunk": "err", "op_id": "exec-1"})
        _feed(reader, {"type": "complete", "exit_code": 0, "execution_time_ms": 50, "op_id": "exec-1"})

        exec_msgs = await _drain(exec_queue)
        list_msgs = await _drain(list_queue)

        assert len(exec_msgs) == 3  # stdout, stderr, complete
        assert len(list_msgs) == 1  # file_list
        assert isinstance(exec_msgs[-1], ExecutionCompleteMessage)
        assert isinstance(list_msgs[0], FileListMessage)

        await dispatcher.stop()

    async def test_empty_op_id_routes_to_op_queue(self) -> None:
        """Empty string op_id is truthy and routes to op queue."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("")
        _feed(reader, {"type": "pong", "version": "1.0", "op_id": ""})

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, PongMessage)

        await dispatcher.stop()

    async def test_message_after_unregister_discarded(self) -> None:
        """Message arriving after unregister_op is discarded."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = await dispatcher.register_op("op-1")
        await dispatcher.unregister_op("op-1")

        _feed(reader, {"type": "pong", "version": "1.0", "op_id": "op-1"})
        # Wait a bit for the dispatch loop to process
        await asyncio.sleep(0.1)

        assert op_queue._queue.empty()

        await dispatcher.stop()


# ============================================================================
# UnixSocketChannel — write worker failure handling
# ============================================================================


# Exception scenarios for dead write worker parametrize
_DEAD_WORKER_CASES = [
    pytest.param(asyncio.CancelledError(), "Write worker was cancelled", id="cancelled"),
    pytest.param(ConnectionResetError("broken pipe"), "Write worker crashed: ConnectionResetError", id="crashed"),
    pytest.param(None, "Write worker exited unexpectedly", id="clean-exit"),
]


class TestWriteWorkerFailureHandling:
    """Tests for UnixSocketChannel behavior when the write worker is dead.

    Covers the fail-fast checks at the top of send_request() and
    stream_messages() that inspect _write_task.result() before proceeding.
    """

    @staticmethod
    def _make_channel_with_dead_writer(
        task_exception: BaseException | None = None,
    ) -> UnixSocketChannel:
        """Build a UnixSocketChannel whose write worker has already finished.

        Args:
            task_exception: If set, the write task finished with this exception.
                If None, the task finished cleanly (no exception).
        """
        ch = UnixSocketChannel("/fake/socket.sock", expected_uid=0)
        # Simulate connected state
        ch._reader = asyncio.StreamReader()
        ch._writer = MagicMock()

        # Build a pre-resolved future to simulate a dead write worker.
        # Using a Future (not Task) is fine — code under test only calls
        # .done() and .result(), which are shared API.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        if task_exception is not None:
            fut.set_exception(task_exception)
        else:
            fut.set_result(None)
        ch._write_task = fut  # type: ignore[assignment]  # Future, not Task — OK for test
        return ch

    @pytest.mark.parametrize("task_exception,expected_match", _DEAD_WORKER_CASES)
    async def test_send_request_dead_write_worker(
        self, task_exception: BaseException | None, expected_match: str
    ) -> None:
        """send_request() raises RuntimeError when write worker is dead."""
        ch = self._make_channel_with_dead_writer(task_exception)

        with pytest.raises(RuntimeError, match=expected_match):
            await ch.send_request(PongMessage(version="1.0"))  # type: ignore[arg-type]

    @pytest.mark.parametrize("task_exception,expected_match", _DEAD_WORKER_CASES)
    async def test_stream_messages_dead_write_worker(
        self, task_exception: BaseException | None, expected_match: str
    ) -> None:
        """stream_messages() raises RuntimeError when write worker is dead."""
        ch = self._make_channel_with_dead_writer(task_exception)

        with pytest.raises(RuntimeError, match=expected_match):
            async for _ in ch.stream_messages(PongMessage(version="1.0"), timeout=5):  # type: ignore[arg-type]
                pass


# ============================================================================
# Idle timer task lifecycle — orphaned task prevention
# ============================================================================


class TestIdleTimerLifecycleNormal:
    """close() and operations leave no armed idle deadline."""

    async def test_close_disarms_timer(self) -> None:
        """close() cancels the pending idle deadline."""
        session, _, _ = _make_session()
        assert session._idle_timer_handle is not None
        await session.close()
        assert session._idle_timer_handle is None

    async def test_multiple_exec_then_close_disarms(self) -> None:
        """Each exec() re-arms the deadline; close() disarms the final one."""
        session, _, _ = _make_session()
        for _ in range(5):
            await session.exec("x = 1")
        await session.close()
        assert session._idle_timer_handle is None

    async def test_context_manager_disarms(self) -> None:
        """async with Session disarms via __aexit__."""
        session, _, _ = _make_session()
        async with session:
            await session.exec("x = 1")
        assert session._idle_timer_handle is None


class TestIdleTimerLifecycleEdge:
    """Edge cases for the idle deadline handle."""

    async def test_double_close_no_error(self) -> None:
        """Second close() is idempotent — no crash, timer stays disarmed."""
        session, _, _ = _make_session()
        await session.close()
        await session.close()
        assert session._idle_timer_handle is None

    async def test_close_from_exec_error_path(self) -> None:
        """VM failure in exec() triggers close() internally — timer disarmed."""
        session, mock_vm, _ = _make_session()
        mock_vm.execute = AsyncMock(side_effect=RuntimeError("VM died"))
        with pytest.raises(RuntimeError, match="VM died"):
            await session.exec("x = 1")
        assert session.closed
        assert session._idle_timer_handle is None

    async def test_reset_replaces_pending_handle(self) -> None:
        """Re-arming cancels the prior handle and installs a fresh one."""
        session, _, _ = _make_session()
        first = session._idle_timer_handle
        session._reset_idle_timer()
        assert session._idle_timer_handle is not first
        assert first is not None and first.cancelled()
        await session.close()

    async def test_natural_idle_timeout_closes_session(self) -> None:
        """The deadline callback fires and closes the session with no crash."""
        session, _, manager = _make_session(idle_timeout_seconds=0)
        # call_later(0) fires on the next loop turn and spawns the close task;
        # await it so destroy_vm completes before asserting (session.closed
        # flips True before _close_impl reaches its destroy_vm await).
        for _ in range(10):
            await asyncio.sleep(0)
            if session._idle_close_task is not None:
                break
        assert session._idle_close_task is not None
        await session._idle_close_task
        assert session.closed
        assert session._idle_timer_handle is None
        manager.destroy_vm.assert_awaited_once()


class TestIdleTimerLifecycleStress:
    """Stress scenarios for the idle deadline."""

    async def test_rapid_reset_storm(self) -> None:
        """50 rapid re-arms leak nothing — each cancels the prior handle."""
        session, _, _ = _make_session()
        for _ in range(50):
            session._reset_idle_timer()
        await session.close()
        assert session._idle_timer_handle is None

    async def test_concurrent_close_and_exec(self) -> None:
        """Race close() against exec() — one wins, timer disarmed either way."""
        session, _, _ = _make_session()
        results = await asyncio.gather(
            session.close(),
            session.exec("x = 1"),
            return_exceptions=True,
        )
        # One of the two may raise SessionClosedError
        errors = [r for r in results if isinstance(r, SessionClosedError)]
        assert len(errors) <= 1
        assert session.closed
        assert session._idle_timer_handle is None

    async def test_concurrent_close_waits_for_shared_destroy(self) -> None:
        """All close callers wait for the one in-flight VM destruction."""
        session, _, manager = _make_session()
        destroy_started = asyncio.Event()
        allow_destroy = asyncio.Event()

        async def delayed_destroy(vm: AsyncMock) -> bool:
            destroy_started.set()
            await allow_destroy.wait()
            vm.state = VmState.DESTROYED
            return True

        manager.destroy_vm.side_effect = delayed_destroy
        first = asyncio.create_task(session.close())
        await destroy_started.wait()
        second = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert session.closed
        assert not second.done()

        allow_destroy.set()
        await asyncio.gather(first, second)
        manager.destroy_vm.assert_awaited_once()

    async def test_destroy_error_propagates_from_close(self) -> None:
        """A destroy error is propagated to the close caller."""
        session, _, manager = _make_session()
        manager.destroy_vm.side_effect = OSError("injected destroy failure")

        with pytest.raises(OSError, match="injected destroy failure"):
            await session.close()

        assert session.closed

    async def test_unconfirmed_vm_cleanup_raises(self) -> None:
        """Unconfirmed process death is reported to the close caller."""
        session, vm, manager = _make_session()
        vm.state = VmState.DESTROYING
        manager.destroy_vm = AsyncMock(return_value=False)

        with pytest.raises(VmPermanentError, match="process cleanup was not confirmed"):
            await session.close()

        assert session.closed

    async def test_terminal_result_survives_ancillary_cleanup_retry(self) -> None:
        """Exit 137 returns once processes are dead, while ancillary retry stays owned."""
        session, vm, manager = _make_session()
        vm.execute.return_value = _make_exec_result(exit_code=137)

        manager.destroy_vm = make_destroy_mock("ancillary_pending")
        result = await session.exec("raise terminal pressure")

        assert result.exit_code == 137
        assert session.closed
        manager.destroy_vm.assert_awaited_once_with(vm)

    async def test_cancelled_exec_closes_before_propagating_cancellation(self) -> None:
        """Caller cancellation cannot leave a possibly executing VM reusable."""
        session, vm, manager = _make_session()
        execute_started = asyncio.Event()

        async def blocked_execute(**_kwargs: object) -> ExecutionResult:
            execute_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        vm.execute.side_effect = blocked_execute
        task = asyncio.create_task(session.exec("while True: pass"))
        await execute_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        manager.destroy_vm.assert_awaited_once()
        assert session.closed

    async def test_idle_timer_suspended_during_operation(self) -> None:
        """An operation running longer than the idle timeout is not destroyed.

        Idle means "no operation in flight": the timer is cancelled while an
        operation runs and re-armed afterwards.
        """
        session, vm, manager = _make_session(idle_timeout_seconds=1)

        async def slow_execute(**_kwargs: object) -> ExecutionResult:
            await asyncio.sleep(1.3)
            return _make_exec_result()

        vm.execute.side_effect = slow_execute
        result = await session.exec("print(1)")

        assert result.exit_code == 0
        assert not session.closed
        manager.destroy_vm.assert_not_awaited()
        await session.close()

    async def test_idle_timer_rearmed_after_operation(self) -> None:
        """After an operation completes, the idle timer must fire again.

        Kills the mutation "suspend forever": deleting the finally re-arm in
        _guard would leak sessions that never idle-close after their first op.
        """
        session, _vm, manager = _make_session(idle_timeout_seconds=1)
        await session.exec("print(1)")
        assert not session.closed

        await asyncio.sleep(1.3)
        assert session.closed
        manager.destroy_vm.assert_awaited_once()

    @pytest.mark.parametrize(
        "state, returncode",
        [
            pytest.param(VmState.DESTROYING, None, id="state-destroying"),
            pytest.param(VmState.DESTROYED, None, id="state-destroyed"),
            pytest.param(VmState.READY, 137, id="returncode-only"),
        ],
    )
    async def test_guard_fails_fast_on_dead_vm(self, state: VmState, returncode: int | None) -> None:
        """All three death signals retire the session up front.

        The returncode leg covers L1-restored VMs that have not executed yet
        (their process-exit watcher is armed lazily on first execute).
        """
        session, vm, manager = _make_session()
        vm.state = state
        vm.process.returncode = returncode

        with pytest.raises(SessionClosedError, match="retired"):
            await session.list_files()

        assert session.closed
        manager.destroy_vm.assert_awaited_once_with(vm)

    async def test_exec_error_remains_primary_when_close_fails(self) -> None:
        """A close-time destroy error must not mask the VM error that caused it."""
        session, vm, manager = _make_session()
        vm.execute.side_effect = RuntimeError("VM died mid-execute")
        manager.destroy_vm.side_effect = OSError("destroy also failed")

        with pytest.raises(RuntimeError, match="VM died mid-execute"):
            await session.exec("print(1)")

        assert session.closed
        manager.destroy_vm.assert_awaited_once()

    async def test_cancellation_survives_close_failure(self) -> None:
        """A close-time destroy error must not replace CancelledError.

        Structured cancellation (asyncio.timeout / wait_for) relies on the
        cancelled task actually ending with CancelledError.
        """
        session, vm, manager = _make_session()
        execute_started = asyncio.Event()

        async def blocked_execute(**_kwargs: object) -> ExecutionResult:
            execute_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        vm.execute.side_effect = blocked_execute
        manager.destroy_vm.side_effect = OSError("injected destroy failure")
        task = asyncio.create_task(session.exec("while True: pass"))
        await execute_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        manager.destroy_vm.assert_awaited_once()
        assert session.closed

    async def test_cancelled_write_closes_before_propagating_cancellation(self) -> None:
        """A cancelled upload cannot commit later and leave the VM reusable."""
        session, vm, manager = _make_session()
        write_started = asyncio.Event()

        async def blocked_write(*_args: object, **_kwargs: object) -> None:
            write_started.set()
            await asyncio.Event().wait()

        vm.write_file.side_effect = blocked_write
        task = asyncio.create_task(session.write_file("target.bin", b"payload"))
        await write_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        manager.destroy_vm.assert_awaited_once()
        assert session.closed
        with pytest.raises(SessionClosedError):
            await session.list_files()

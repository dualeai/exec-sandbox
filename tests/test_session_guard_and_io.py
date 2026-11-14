# pyright: reportPrivateUsage=false
"""Unit tests of the Session calls (lifecycle checks, lock, idle deadline, write_file
sources), of the QemuVM file calls on a fake guest channel (frames on the wire, error
mapping), of FileOpDispatcher routing, and of the DualPortChannel reconnection probe.

No VM: the Session tests use a mocked QemuVM; the QemuVM tests use a real QemuVM on a
mocked channel. Covers:
- Session calls: closed checks, lock serialization, idle deadline suspended during a
  call, fail-fast retirement of dead VMs, close paths
- write_file sources (bytes, Path, stream): checks, size limits, local errors
- QemuVM write_file and read_file: frames sent, content round trip, error mapping
- FileOpDispatcher message routing by op_id, transport failure
- DualPortChannel reconnection probe and close
"""

import asyncio
import base64
import errno
import gc
import io
import json
import os
import random
import re
import tempfile
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from exec_sandbox import constants
from exec_sandbox.constants import FILE_TRANSFER_CHUNK_SIZE, MAX_CODE_SIZE, MAX_FILE_SIZE_BYTES
from exec_sandbox.exceptions import (
    CommunicationError,
    ConfigError,
    ErrorCode,
    InternalError,
    InvalidInputError,
    OutcomeUnknownError,
    RouteError,
    SandboxError,
    SessionClosedError,
    TransientError,
)
from exec_sandbox.guest_agent_protocol import (
    ExecutionCompleteMessage,
    FileChunkResponseMessage,
    FileListMessage,
    FileReadCompleteMessage,
    FileWriteAckMessage,
    OutputChunkMessage,
    PingRequest,
    PongMessage,
    StreamingErrorMessage,
    StreamingMessage,
)
from exec_sandbox.guest_channel import (
    OP_QUEUE_DEPTH,
    DualPortChannel,
    FileOpDispatcher,
    OperationInbox,
    UnixSocketChannel,
)
from exec_sandbox.models import ExecutionResult, ExposedPort, FileInfo, Language, StartupTiming, TimingBreakdown
from exec_sandbox.qemu_vm import QemuVM, local_path_error, zstd
from exec_sandbox.session import Session
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType, VmState
from tests.conftest import NonSeekableIO, make_destroy_mock

# ============================================================================
# Helpers — build a Session with mocked VM + VmManager
# ============================================================================


def _make_exec_result(exit_code: int = 0) -> ExecutionResult:
    """Build a minimal ExecutionResult for mocks."""
    return ExecutionResult(
        stdout="",
        stderr="",
        exit_code=exit_code,
        execution_time_ms=1,
        external_cpu_time_ms=None,
        external_memory_peak_mb=None,
        timing=TimingBreakdown(execute_ms=1, connect_ms=0),
        spawn_ms=None,
        process_ms=None,
    )


_STARTUP = StartupTiming(
    total_ms=1,
    admission_ms=0,
    setup_ms=1,
    boot_ms=0,
    warm_pool_hit=True,
    l1_cache_hit=False,
    l1_restore_ms=None,
    overlay_ms=0,
    qemu_cmd_build_ms=0,
    gvproxy_start_ms=0,
    qemu_fork_ms=0,
    guest_wait_ms=0,
    boot_retries=0,
    instance_load_ms=None,
)


def _closed_buffer() -> io.BytesIO:
    buffer = io.BytesIO()
    buffer.close()
    return buffer


class _SizeIgnoringReader(io.RawIOBase):
    """A binary stream that breaks the read contract: bytes whatever the size asked."""

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        return b"chunk"


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
        spec=VmSpec(language=Language.PYTHON),
        startup_timing=_STARTUP,
        instances=None,
        instance=None,
        new_instance_id=None,
        idle_timeout_seconds=idle_timeout_seconds,
        default_timeout_seconds=default_timeout_seconds,
    )
    return session, mock_vm, mock_vm_manager


@dataclass
class _Upload:
    """One write_file call that the mocked VM got."""

    path: str
    data: bytes  # the bytes from the stream position to its end
    make_executable: bool
    stream: IO[bytes]


def _record_uploads(vm: AsyncMock) -> list[_Upload]:
    """Make the mocked VM's write_file read its stream to the end (as the real one does)
    and record each call."""
    uploads: list[_Upload] = []

    async def write_file(path: str, content: IO[bytes], *, make_executable: bool = False) -> None:
        uploads.append(_Upload(path, content.read(), make_executable, content))

    vm.write_file.side_effect = write_file
    return uploads


def _record_upload_sizes(vm: AsyncMock) -> list[int]:
    """Make the mocked VM's write_file record the bytes left in its stream, without a read
    (for the 500 MiB limit cases)."""
    sizes: list[int] = []

    async def write_file(_path: str, content: IO[bytes], *, make_executable: bool = False) -> None:
        start = content.tell()
        sizes.append(content.seek(0, os.SEEK_END) - start)

    vm.write_file.side_effect = write_file
    return sizes


def _sparse_file(path: Path, size: int) -> Path:
    """A file of `size` bytes with no data blocks: the 500 MiB limit cases need no disk."""
    with path.open("wb") as f:
        f.truncate(size)
    return path


def _assert_freed(session_ref: "weakref.ref[Session]") -> None:
    """The closed session is freed when the test drops it: no armed idle deadline (a loop
    timer that holds the session) stays after the close."""
    gc.collect()
    assert session_ref() is None, "an idle deadline holds the closed session"


# ============================================================================
# exec() — timeout_seconds validation
# ============================================================================


class TestExecTimeoutValidation:
    """Tests for timeout_seconds validation in Session.exec()."""

    async def test_timeout_zero_rejected(self) -> None:
        """timeout_seconds=0 raises InvalidInputError ARGUMENT_INVALID."""
        session, _, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match="timeout_seconds must be between 1 and 300",
        ):
            await session.exec("print(1)", timeout_seconds=0)

    async def test_timeout_negative_rejected(self) -> None:
        """Negative timeout_seconds raises InvalidInputError ARGUMENT_INVALID."""
        session, _, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match="timeout_seconds must be between 1 and 300",
        ):
            await session.exec("print(1)", timeout_seconds=-1)

    async def test_timeout_exceeds_max_rejected(self) -> None:
        """timeout_seconds > 300 raises InvalidInputError ARGUMENT_INVALID."""
        session, _, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match="timeout_seconds must be between 1 and 300",
        ):
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


class TestExecCodeSizeValidation:
    """Code size is checked in Session.exec() before anything reaches the guest."""

    async def test_code_exceeds_limit_rejected(self) -> None:
        session, mock_vm, _ = _make_session()
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.CODE_INVALID, match="Code too large"):
            await session.exec("x" * (MAX_CODE_SIZE + 1))
        mock_vm.execute.assert_not_called()

    async def test_code_at_limit_accepted(self) -> None:
        session, mock_vm, _ = _make_session()
        await session.exec("x" * MAX_CODE_SIZE)
        mock_vm.execute.assert_called_once()

    async def test_startup_timing_and_spec_exposed(self) -> None:
        session, _, _ = _make_session()
        assert session.startup_timing is _STARTUP
        assert session.spec == VmSpec(language=Language.PYTHON)


# ============================================================================
# save() — which failures close the session
# ============================================================================


def _error(cls: type[SandboxError], code: ErrorCode, context: dict[str, Any] | None) -> SandboxError:
    """An error of `cls` with `code` (RouteError takes its two extra fields)."""
    if cls is RouteError:
        return RouteError("injected", code=code, required={}, found={}, context=context, hint=None, instance_id=None)
    return cls("injected", code=code, context=context, hint=None, instance_id=None)


class TestSaveCloseRule:
    """save() closes the session by error code, not by class: a failed migration closes it
    unless the VM runs again (context["vm_resumed"]); a local disk or store failure
    (TransientError too) leaves the VM running, so the session stays open."""

    @pytest.mark.parametrize(
        ("cls", "code", "context", "closes"),
        [
            pytest.param(TransientError, ErrorCode.MIGRATION_FAILED, {"vm_resumed": False}, True, id="failed"),
            pytest.param(TransientError, ErrorCode.MIGRATION_FAILED, {"vm_resumed": True}, False, id="failed-resumed"),
            pytest.param(ConfigError, ErrorCode.MIGRATION_UNSUPPORTED, None, True, id="unsupported"),
            pytest.param(
                ConfigError, ErrorCode.MIGRATION_UNSUPPORTED, {"vm_resumed": True}, False, id="unsupported-resumed"
            ),
            pytest.param(TransientError, ErrorCode.LOCAL_DISK, None, False, id="local-disk"),
            pytest.param(TransientError, ErrorCode.INSTANCE_STORE_UNAVAILABLE, None, False, id="store"),
        ],
    )
    async def test_close_rule(
        self, cls: type[SandboxError], code: ErrorCode, context: dict[str, Any] | None, closes: bool
    ) -> None:
        _, mock_vm, mock_vm_manager = _make_session()
        error = _error(cls, code, context)
        instances = AsyncMock()
        instances.save = AsyncMock(side_effect=error)
        session = Session(
            vm=mock_vm,
            vm_manager=mock_vm_manager,
            spec=VmSpec(language=Language.PYTHON),
            startup_timing=_STARTUP,
            instances=instances,
            instance=None,
            new_instance_id=None,
            idle_timeout_seconds=300,
            default_timeout_seconds=30,
        )

        with pytest.raises(cls, check=lambda e: e is error):
            await session.save()

        assert session.closed is closes
        assert mock_vm_manager.destroy_vm.await_count == (1 if closes else 0)
        await session.close()


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

    @pytest.mark.parametrize(
        ("make_destination", "match"),
        [
            pytest.param(_closed_buffer, "^read_file destination stream is closed$", id="closed"),
            pytest.param(
                lambda: io.BufferedReader(io.BytesIO(b"")),
                "^read_file destination stream is not writable$",
                id="read-only",
            ),
            pytest.param(
                io.StringIO,
                "^read_file destination stream is a text stream: open it in binary mode$",
                id="text",
            ),
            pytest.param(  # not an io.TextIOBase: only its write() refuses bytes
                lambda: tempfile.SpooledTemporaryFile(mode="w+"),
                "^read_file destination stream is a text stream: open it in binary mode$",
                id="wrapped-text",
            ),
        ],
    )
    async def test_read_file_refuses_an_unusable_destination(self, make_destination: Any, match: str) -> None:
        """A closed, read-only or text destination stream is the caller's error (FIX_INPUT),
        found before the call takes the session: the guest is not asked, the session stays
        open. A text stream would get bytes after dispatch: a builtin TypeError."""
        session, mock_vm, _ = _make_session()
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=match):
            await session.read_file("out.bin", destination=make_destination())
        mock_vm.read_file.assert_not_awaited()
        assert not session.closed

    async def test_list_files_calls_vm(self) -> None:
        """list_files() delegates to VM under _guard."""
        session, mock_vm, _ = _make_session()
        result = await session.list_files("subdir")
        mock_vm.list_files.assert_awaited_once_with("subdir")
        assert len(result) == 1

    @pytest.mark.parametrize("content", [b"data", b""], ids=["bytes", "empty"])
    async def test_write_file_bytes_calls_vm(self, content: bytes) -> None:
        """write_file() with bytes gives the VM a stream of these bytes, the path and the
        flag; the session closes the stream it made."""
        session, mock_vm, _ = _make_session()
        uploads = _record_uploads(mock_vm)
        await session.write_file("out.bin", content, make_executable=True)
        assert [(u.path, u.data, u.make_executable) for u in uploads] == [("out.bin", content, True)]
        assert uploads[0].stream.closed


# ============================================================================
# _guard() — Closed session (fast path, before lock)
# ============================================================================


class TestGuardFailureThatRetiresTheVm:
    """A failed call that left the VM non-reusable closes the session (QemuVM retires a VM
    that QEMU stopped at a full host disk, then raises LOCAL_DISK); one that left it
    usable keeps the session open."""

    @pytest.mark.parametrize("retired", [True, False], ids=["vm-retired", "vm-usable"])
    async def test_failed_file_call(self, retired: bool) -> None:
        session, vm, manager = _make_session()

        async def fail(_path: str) -> list[FileInfo]:
            if retired:
                vm.state = VmState.DESTROYING
            raise TransientError(
                "VM stopped: the host disk is full",
                code=ErrorCode.LOCAL_DISK,
                context=None,
                hint=None,
                instance_id=None,
            )

        vm.list_files.side_effect = fail
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
            await session.list_files("")

        assert session.closed is retired
        assert manager.destroy_vm.await_count == int(retired)


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

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda s, _p: s.exec("x = 1", timeout_seconds=0), id="exec-timeout"),
            pytest.param(lambda s, _p: s.exec("x" * (MAX_CODE_SIZE + 1)), id="exec-code-size"),
            pytest.param(lambda s, p: s.write_file("f.txt", p / "missing.bin"), id="write-missing-source"),
            pytest.param(lambda s, _p: s.write_file("f.txt", _closed_buffer()), id="write-closed-stream"),
            pytest.param(
                lambda s, _p: s.read_file("f.txt", destination=_closed_buffer()), id="read-closed-destination"
            ),
        ],
    )
    async def test_closed_session_comes_before_argument_errors(self, call: Any, tmp_path: Path) -> None:
        """On a closed session the caller must open a new session whatever the arguments:
        RESTART is checked before FIX_INPUT."""
        session, _, _ = _make_session()
        await session.close()
        with pytest.raises(SessionClosedError, match=r"^Session is closed$"):
            await call(session, tmp_path)

    async def test_write_file_on_closed_session_opens_no_local_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "source.bin"
        source.write_bytes(b"data")
        session, _, _ = _make_session()
        await session.close()
        opened: list[Any] = []
        open_ = os.open

        def spied_open(path: Any, *args: Any, **kwargs: Any) -> int:
            if Path(path) == source:
                opened.append(path)
            return open_(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", spied_open)
        with pytest.raises(SessionClosedError, match=r"^Session is closed$"):
            await session.write_file("f.txt", source)
        assert opened == []


# ============================================================================
# _guard() — Closed while waiting for lock (race condition)
# ============================================================================


class TestGuardClosedDuringLockWait:
    """A call that waits for the session lock checks again, when it gets the lock, that
    the session is open: close() does not take the lock."""

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda s, _p: s.exec("x = 1"), id="exec"),
            pytest.param(lambda s, _p: s.write_file("f.txt", b"data"), id="write_file"),
            pytest.param(lambda s, p: s.read_file("f.txt", destination=p / "f.bin"), id="read_file"),
            pytest.param(lambda s, _p: s.list_files(), id="list_files"),
        ],
    )
    async def test_call_waiting_for_the_lock_fails_when_the_session_closes(self, call: Any, tmp_path: Path) -> None:
        """A first call holds the lock; a second call waits for it; close() runs. The second
        call then gets SessionClosedError and does not reach the VM."""
        session, mock_vm, _ = _make_session()
        holding, release = asyncio.Event(), asyncio.Event()

        async def held_list(_path: str = "") -> list[FileInfo]:
            holding.set()
            await release.wait()
            return []

        mock_vm.list_files.side_effect = held_list
        first = asyncio.create_task(session.list_files())
        await holding.wait()
        waiting = asyncio.create_task(call(session, tmp_path))
        await asyncio.sleep(0)  # the second call passes the open check and waits for the lock
        await session.close()
        release.set()

        with pytest.raises(SessionClosedError, match=r"^Session closed while waiting for lock$"):
            await waiting
        assert await first == []
        assert mock_vm.list_files.await_count == 1
        mock_vm.execute.assert_not_awaited()
        mock_vm.write_file.assert_not_awaited()
        mock_vm.read_file.assert_not_awaited()


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
# write_file() sources — bytes, Path, caller stream
# ============================================================================


class TestWriteFileSource:
    """write_file gives the VM one readable stream of the content. The session closes a
    stream it made (bytes, Path); a stream of the caller stays open."""

    @pytest.mark.parametrize("size", [13, 2 * 1024 * 1024], ids=["small", "2MiB"])
    async def test_path_content_reaches_the_vm_and_the_file_is_closed(self, tmp_path: Path, size: int) -> None:
        content = os.urandom(size)
        source = tmp_path / "source.bin"
        source.write_bytes(content)
        session, mock_vm, _ = _make_session()
        uploads = _record_uploads(mock_vm)

        await session.write_file("out.bin", source)

        assert [(u.path, u.data) for u in uploads] == [("out.bin", content)]
        assert uploads[0].stream.closed

    async def test_caller_stream_reaches_the_vm_and_stays_open(self) -> None:
        """An IO[bytes] stream of the caller goes to the VM as it is; the session does not close it."""
        session, mock_vm, _ = _make_session()
        uploads = _record_uploads(mock_vm)
        buf = io.BytesIO(b"hello")

        await session.write_file("out.bin", buf)

        assert [(u.stream, u.data) for u in uploads] == [(buf, b"hello")]
        assert not buf.closed

    async def test_path_source_opens_without_the_session_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Path source is opened before the call takes the session lock: while its open
        waits for a slow local disk, another call of the session runs."""
        source = tmp_path / "source.bin"
        source.write_bytes(b"x" * 100)
        loop_thread = threading.get_ident()
        reached, released = threading.Event(), threading.Event()
        fdopen = os.fdopen

        def held(fd: int, *args: Any, **kwargs: Any) -> IO[Any]:
            stream = fdopen(fd, *args, **kwargs)
            reached.set()
            if threading.get_ident() != loop_thread:  # an open on the loop must not block the test
                released.wait(10)
            return stream

        monkeypatch.setattr(os, "fdopen", held)
        session, mock_vm, _ = _make_session()
        uploads = _record_uploads(mock_vm)
        writing = asyncio.create_task(session.write_file("out.bin", source))
        try:
            assert await asyncio.to_thread(reached.wait, 10)
            # The bound only ends a failed run: with the lock held, list_files waits for the open.
            listed = await asyncio.wait_for(session.list_files(), timeout=5)
        finally:
            released.set()
        await writing

        assert listed == [FileInfo(name="a.txt", is_dir=False, size=10)]
        assert [u.data for u in uploads] == [b"x" * 100]

    async def test_path_checked_and_opened_off_the_loop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Path source is opened and checked (type, size: fstat) in a worker thread: the
        caller's path can be on a network disk, where each call waits for the disk."""
        f = tmp_path / "src.bin"
        f.write_bytes(b"data")
        inode = f.stat().st_ino
        callers: list[tuple[str, int]] = []
        open_, fstat = os.open, os.fstat

        def spied_open(path: Any, *args: Any, **kwargs: Any) -> int:
            if Path(path) == f:
                callers.append(("open", threading.get_ident()))
            return open_(path, *args, **kwargs)

        def spied_fstat(fd: int) -> os.stat_result:
            st = fstat(fd)
            if st.st_ino == inode:
                callers.append(("fstat", threading.get_ident()))
            return st

        monkeypatch.setattr(os, "open", spied_open)
        monkeypatch.setattr(os, "fstat", spied_fstat)
        session, mock_vm, _ = _make_session()
        uploads = _record_uploads(mock_vm)
        await session.write_file("out.bin", f)

        assert [u.data for u in uploads] == [b"data"]
        assert {step for step, _ in callers} == {"open", "fstat"}
        assert threading.get_ident() not in {thread for _, thread in callers}

    async def test_cancel_while_opening_closes_the_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A caller cancelled while its Path source is being opened: the cancel waits for the
        open, closes the file it opened (no file object without an owner), then propagates.
        The VM is not asked and the session stays open."""
        f = tmp_path / "src.bin"
        f.write_bytes(b"data")
        loop_thread = threading.get_ident()
        reached, released = threading.Event(), threading.Event()
        opened: list[IO[Any]] = []
        fdopen = os.fdopen

        def held(fd: int, *args: Any, **kwargs: Any) -> IO[Any]:
            stream = fdopen(fd, *args, **kwargs)
            opened.append(stream)
            reached.set()
            if threading.get_ident() != loop_thread:  # an open on the loop must not block the test
                released.wait(10)
            return stream

        monkeypatch.setattr(os, "fdopen", held)
        session, mock_vm, _ = _make_session()
        writing = asyncio.create_task(session.write_file("out.bin", f))
        try:
            assert await asyncio.to_thread(reached.wait, 10)
            writing.cancel()
            await asyncio.sleep(0.05)
            assert not writing.done(), "the cancel did not wait for the open"
        finally:
            released.set()
        with pytest.raises(asyncio.CancelledError):
            await writing
        (stream,) = opened
        assert stream.closed
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed

    @pytest.mark.timeout(30)
    async def test_fifo_source_is_refused_at_once(self, tmp_path: Path) -> None:
        """A FIFO has no size to check, and its open or read can block forever (no writer):
        refused as not a regular file, without waiting for a writer."""
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        session, mock_vm, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=r"^Source pipe is not a regular file$",
        ):
            await asyncio.wait_for(session.write_file("out.bin", fifo), timeout=5)
        mock_vm.write_file.assert_not_awaited()


# ============================================================================
# write_file() sources — limits and unusable streams
# ============================================================================


class TestWriteFileSourceLimits:
    """Size limit (500 MiB) of each source kind, and streams that cannot be read as bytes.
    A refused source is the caller's error (ARGUMENT_INVALID), found before the call takes
    the session: the VM is not asked and the session stays open."""

    async def test_closed_stream_is_refused(self) -> None:
        """A closed IO source: refused before the write starts (a read would raise a builtin ValueError)."""
        session, mock_vm, _ = _make_session()
        stream = io.BytesIO(b"data")
        stream.close()
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=r"^Stream content is closed$"
        ):
            await session.write_file("out.bin", stream)
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed

    @pytest.mark.parametrize(
        ("make_source", "match"),
        [
            pytest.param(lambda: tempfile.TemporaryFile("wb"), "^Stream content is not readable$", id="write-only"),
            pytest.param(
                lambda: io.StringIO("text"), "^Stream content is a text stream: open it in binary mode$", id="text"
            ),
            pytest.param(  # not an io.TextIOBase: only its read() gives str
                lambda: tempfile.SpooledTemporaryFile(mode="w+"),
                "^Stream content is a text stream: open it in binary mode$",
                id="wrapped-text",
            ),
            pytest.param(
                _SizeIgnoringReader, "^Stream content returned bytes for a zero-byte read$", id="ignores-size"
            ),
        ],
    )
    async def test_stream_that_cannot_be_read_as_bytes_is_refused(self, make_source: Any, match: str) -> None:
        """A source stream not open for reading, a text stream, or one that ignores the read
        size: the caller's error (FIX_INPUT), found before the call takes the session. The
        guest gets no header (a failed read after it would close the session as VERIFY)."""
        session, mock_vm, _ = _make_session()
        with make_source() as source:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=match):
                await session.write_file("out.bin", source)
            assert not source.closed  # the caller's stream stays open
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed

    async def test_bytes_exactly_at_max_size_accepted(self) -> None:
        session, mock_vm, _ = _make_session()
        sizes = _record_upload_sizes(mock_vm)
        await session.write_file("out.bin", bytes(MAX_FILE_SIZE_BYTES))
        assert sizes == [MAX_FILE_SIZE_BYTES]

    async def test_bytes_one_over_max_rejected(self) -> None:
        session, mock_vm, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=r"^Content is 524288001 bytes, exceeds 524288000$",
        ):
            await session.write_file("out.bin", bytes(MAX_FILE_SIZE_BYTES + 1))
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed

    async def test_path_exactly_at_max_size_accepted(self, tmp_path: Path) -> None:
        source = _sparse_file(tmp_path / "exact.bin", MAX_FILE_SIZE_BYTES)
        session, mock_vm, _ = _make_session()
        sizes = _record_upload_sizes(mock_vm)
        await session.write_file("out.bin", source)
        assert sizes == [MAX_FILE_SIZE_BYTES]

    async def test_path_one_over_max_rejected(self, tmp_path: Path) -> None:
        source = _sparse_file(tmp_path / "over.bin", MAX_FILE_SIZE_BYTES + 1)
        session, mock_vm, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=r"^File over\.bin is 524288001 bytes, exceeds 524288000$",
        ):
            await session.write_file("out.bin", source)
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed

    async def test_empty_file_accepted(self, tmp_path: Path) -> None:
        source = tmp_path / "empty.bin"
        source.write_bytes(b"")
        session, mock_vm, _ = _make_session()
        uploads = _record_uploads(mock_vm)
        await session.write_file("out.bin", source)
        assert [u.data for u in uploads] == [b""]

    async def test_seekable_stream_at_max_size_accepted(self) -> None:
        session, mock_vm, _ = _make_session()
        sizes = _record_upload_sizes(mock_vm)
        await session.write_file("out.bin", io.BytesIO(bytes(MAX_FILE_SIZE_BYTES)))
        assert sizes == [MAX_FILE_SIZE_BYTES]

    async def test_seekable_stream_over_max_rejected(self) -> None:
        session, mock_vm, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=r"^Stream content is 524288001 bytes, exceeds 524288000$",
        ):
            await session.write_file("out.bin", io.BytesIO(bytes(MAX_FILE_SIZE_BYTES + 1)))
        mock_vm.write_file.assert_not_awaited()

    async def test_seekable_stream_limit_counts_from_its_position(self) -> None:
        """The limit counts the bytes from the stream position, not from its start: a stream
        one byte over the limit, read from offset 1, is accepted."""
        session, mock_vm, _ = _make_session()
        sizes = _record_upload_sizes(mock_vm)
        buf = io.BytesIO(bytes(MAX_FILE_SIZE_BYTES + 1))
        buf.seek(1)
        await session.write_file("out.bin", buf)
        assert sizes == [MAX_FILE_SIZE_BYTES]

    async def test_non_seekable_stream_has_no_size_check(self) -> None:
        """A non-seekable stream has no size to check before the write: it goes to the VM
        (the guest enforces the limit as it receives the bytes)."""
        session, mock_vm, _ = _make_session()
        uploads = _record_uploads(mock_vm)
        buf = io.BufferedReader(NonSeekableIO(b"x" * 100))
        await session.write_file("out.bin", buf)
        assert [(u.stream, u.data) for u in uploads] == [(buf, b"x" * 100)]

    async def test_seekable_stream_keeps_its_position(self) -> None:
        """The size check sets the stream back to its position: the VM reads from there."""
        session, mock_vm, _ = _make_session()
        uploads = _record_uploads(mock_vm)
        buf = io.BytesIO(b"prefix" + b"data")
        buf.seek(6)
        await session.write_file("out.bin", buf)
        assert [u.data for u in uploads] == [b"data"]


# ============================================================================
# write_file() sources — local errors
# ============================================================================


class TestWriteFileSourceErrors:
    """A Path source that cannot be opened: the caller's error (ARGUMENT_INVALID) when the
    same call fails again, else this host's (LOCAL_DISK). The VM is not asked and the
    session stays open."""

    async def test_nonexistent_path_raises_argument_invalid(self, tmp_path: Path) -> None:
        session, mock_vm, _ = _make_session()
        ghost = tmp_path / "ghost.txt"
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=rf"^Source file cannot be opened \(No such file or directory\): {re.escape(str(ghost))}$",
        ):
            await session.write_file("out.bin", ghost)
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed

    @pytest.mark.parametrize("case", ["no-read-access", "symlink-loop", "file-in-path"])
    async def test_unopenable_source_raises_argument_invalid(self, tmp_path: Path, case: str) -> None:
        """A source the caller cannot open is the caller's error (FIX_INPUT), not a builtin."""
        source = tmp_path / "source.bin"
        if case == "no-read-access":
            if os.geteuid() == 0:
                pytest.skip("root reads a file without read permission")
            source.write_bytes(b"data")
            source.chmod(0)
        elif case == "symlink-loop":
            source.symlink_to(tmp_path / "loop")
            (tmp_path / "loop").symlink_to(source)
        else:  # ENOTDIR
            (tmp_path / "a-file").write_bytes(b"")
            source = tmp_path / "a-file" / "source.bin"
        session, mock_vm, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=rf"^Source file cannot be opened \(.+\): {re.escape(str(source))}$",
        ) as exc_info:
            await session.write_file("out.bin", source)
        assert isinstance(exc_info.value.__cause__, OSError)
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed

    @pytest.mark.parametrize(
        ("step", "code"),
        [pytest.param("open", errno.EMFILE, id="open-EMFILE"), pytest.param("fstat", errno.EIO, id="fstat-EIO")],
    )
    async def test_source_host_error_is_local_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str, code: int
    ) -> None:
        """This host cannot open the source now (out of file handles, an I/O error): the call
        is right and can pass later (RETRY: LOCAL_DISK), not the caller's error."""
        source = tmp_path / "source.bin"
        source.write_bytes(b"data")
        inode = source.stat().st_ino
        open_, fstat = os.open, os.fstat

        def failing_open(path: Any, *args: Any, **kwargs: Any) -> int:
            if step == "open" and Path(path) == source:
                raise OSError(code, os.strerror(code), str(path))
            return open_(path, *args, **kwargs)

        def failing_fstat(fd: int) -> os.stat_result:
            st = fstat(fd)
            if step == "fstat" and st.st_ino == inode:
                raise OSError(code, os.strerror(code))
            return st

        monkeypatch.setattr(os, "open", failing_open)
        monkeypatch.setattr(os, "fstat", failing_fstat)
        session, mock_vm, _ = _make_session()
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.LOCAL_DISK and e.context["errno"] == code,
            match=rf"^Source file cannot be opened \({re.escape(os.strerror(code))}\): {re.escape(str(source))}$",
        ):
            await session.write_file("out.bin", source)
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed

    async def test_nul_byte_in_source_path_is_argument_invalid(self) -> None:
        """A NUL byte in the source path (an agent can build one from model output): the
        caller's error, not a builtin ValueError; the path is not printed."""
        session, mock_vm, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=r"^Source file cannot be opened \(.*null.*\)$",
        ) as exc_info:
            await session.write_file("out.bin", Path("a\x00b"))
        assert isinstance(exc_info.value.__cause__, ValueError)
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed

    async def test_path_is_directory_raises(self, tmp_path: Path) -> None:
        """Path pointing to a directory: not a regular file."""
        d = tmp_path / "subdir"
        d.mkdir()
        session, mock_vm, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=r"^Source subdir is not a regular file$",
        ):
            await session.write_file("out.bin", d)
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed


# ============================================================================
# QemuVM file calls — a real QemuVM on a fake guest channel
# ============================================================================


_DEADLINE_DIAGNOSTICS = {
    "console_log": "[WARN] last guest line",
    "qemu_status": "running",
    "qemu_cpu_seconds": 1.5,
    "qemu_threads": 8,
}
"""What a stubbed QemuVM.deadline_diagnostics returns: a host deadline error must carry it."""


async def _file_io_vm(vm_id: str, *answers: StreamingMessage) -> tuple[QemuVM, AsyncMock, OperationInbox]:
    """A real QemuVM on a fake guest channel. register_op gives one inbox that holds the
    guest's `answers`; enqueue_registered is the real channel code (check the inbox, mark
    the command sent, send it with enqueue_raw); enqueue_raw records each frame sent on the
    command port. deadline_diagnostics (QMP and process probes) gives _DEADLINE_DIAGNOSTICS."""
    inbox = OperationInbox()
    for answer in answers:
        assert await inbox.route(answer)
    channel = AsyncMock(spec=DualPortChannel)
    channel.register_op.return_value = inbox

    async def enqueue_registered(registered: OperationInbox, data: bytes) -> None:
        await DualPortChannel.enqueue_registered(channel, registered, data)  # type: ignore[arg-type]

    channel.enqueue_registered.side_effect = enqueue_registered
    process = MagicMock()
    process.returncode = None
    vm = QemuVM(
        vm_id,
        process=process,
        cgroup=None,
        workdir=MagicMock(use_qemu_vm_user=False),
        channel=channel,
        spec=VmSpec(language=Language.PYTHON),
        console_lines=deque(maxlen=10),
        base_image=Path("/test/base.qcow2"),
        packages_image=None,
        accel_type=AccelType.KVM,
        release_callback=AsyncMock(side_effect=AssertionError("this test does not destroy the VM")),
    )
    vm.deadline_diagnostics = AsyncMock(return_value=_DEADLINE_DIAGNOSTICS)  # type: ignore[method-assign]
    return vm, channel, inbox


def _frames(channel: AsyncMock) -> list[dict[str, Any]]:
    """The frames the VM sent on the command port, in order. Each frame is one JSON line."""
    frames: list[dict[str, Any]] = []
    for call in channel.enqueue_raw.await_args_list:
        data: bytes = call.args[0]
        assert data.endswith(b"\n")
        assert data.count(b"\n") == 1, "a frame must be one line"
        frames.append(json.loads(data))
    return frames


def _sent_content(frames: list[dict[str, Any]]) -> bytes:
    """The file content in the file_chunk frames, decoded with a new zstd decoder."""
    compressed = b"".join(base64.b64decode(f["data"]) for f in frames if f["action"] == "file_chunk")
    return zstd.ZstdDecompressor().decompress(compressed)


def _guest_file(payload: bytes, *, path: str = "file.bin", parts: int = 1) -> list[StreamingMessage]:
    """What the guest sends for read_file: the payload in one zstd stream cut in `parts`
    chunk messages, then the completion with its path and size."""
    compressor = zstd.ZstdCompressor(level=3)
    compressed = compressor.compress(payload) + compressor.flush()
    step = -(-len(compressed) // parts)
    chunks: list[StreamingMessage] = [
        FileChunkResponseMessage(op_id="read-op", data=compressed[i : i + step])
        for i in range(0, len(compressed), step)
    ]
    return [*chunks, FileReadCompleteMessage(op_id="read-op", path=path, size=len(payload))]


class TestWriteFilePathValidation:
    """QemuVM.write_file checks the path before it sends anything (the guest checks it again)."""

    async def test_empty_path_rejected(self) -> None:
        vm, channel, _ = await _file_io_vm("test-vm")
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.PATH_INVALID,
            match=r"^write_file path must not be empty$",
        ):
            await vm.write_file("", io.BytesIO(b"data"))
        channel.connect.assert_not_awaited()
        channel.enqueue_raw.assert_not_awaited()

    async def test_path_at_max_length_accepted(self) -> None:
        """A path of 4096 characters (PATH_MAX) goes to the guest."""
        path = "x" * 4096
        vm, channel, _ = await _file_io_vm("test-vm", FileWriteAckMessage(op_id="w", path=path, bytes_written=4))
        await vm.write_file(path, io.BytesIO(b"data"))
        assert _frames(channel)[0]["path"] == path

    async def test_path_over_max_length_rejected(self) -> None:
        vm, channel, _ = await _file_io_vm("test-vm")
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.PATH_INVALID,
            match=r"^write_file path is 4097 chars, exceeds 4096$",
        ):
            await vm.write_file("x" * 4097, io.BytesIO(b"data"))
        channel.connect.assert_not_awaited()
        channel.enqueue_raw.assert_not_awaited()

    async def test_dropped_header_error_is_not_retried(self) -> None:
        """The guest never drops a request (it keeps its ports open), so a "No active
        write" protocol error is a bug to report: it propagates at once, with no retry
        and no channel reset."""
        error = StreamingErrorMessage(op_id="w", error_type="protocol_error", message="No active write for op_id 'x'")
        vm, channel, _ = await _file_io_vm("protocol-error-vm", error)

        with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL, match="No active write"):
            await vm.write_file("target.bin", io.BytesIO(b"data"))

        channel.register_op.assert_awaited_once()
        assert [f["action"] for f in _frames(channel)].count("write_file") == 1
        channel.close.assert_not_awaited()


class TestWriteFileWire:
    """The frames QemuVM.write_file sends: a write_file header, the content in file_chunk
    frames (one zstd stream, base64), then file_end, all with the same op_id."""

    @pytest.mark.parametrize(
        "path",
        [
            "test.txt",
            'my"file.txt',
            "dir\\file.txt",
            "line1\nline2.txt",
            "col1\tcol2.tsv",
            "données/résultat.csv",
            "data/output_\U0001f680.json",
            'a"b\\c/d\be\ff\ng\rh\ti',
        ],
        ids=["plain", "quote", "backslash", "newline", "tab", "unicode", "emoji", "all-json-escapes"],
    )
    @pytest.mark.parametrize("make_executable", [False, True], ids=["0644", "0755"])
    async def test_header_carries_path_and_mode_on_one_line(self, path: str, make_executable: bool) -> None:
        """Each frame is one JSON line, also for a path with quotes, backslashes or control
        characters (a newline in the path would end the frame). The header carries the host
        clock at send time (the guest steps its clock forward to it)."""
        vm, channel, _ = await _file_io_vm("test-vm", FileWriteAckMessage(op_id="w", path=path, bytes_written=4))
        before = time.time_ns()
        await vm.write_file(path, io.BytesIO(b"data"), make_executable=make_executable)
        after = time.time_ns()

        header, *_, end = _frames(channel)
        op_id = end["op_id"]
        assert before <= header.pop("host_time_ns") <= after
        assert header == {"action": "write_file", "op_id": op_id, "path": path, "make_executable": make_executable}
        assert end == {"action": "file_end", "op_id": op_id}
        assert re.fullmatch(r"[0-9a-f]{32}", op_id)

    @pytest.mark.parametrize(
        "size",
        [0, 30, 2 * FILE_TRANSFER_CHUNK_SIZE + 7],
        ids=["empty", "small", "three-chunks"],
    )
    async def test_content_round_trips_through_the_chunks(self, size: int) -> None:
        """The file_chunk frames decode (base64, then one zstd stream) to the source bytes;
        the ack binds the byte count."""
        content = os.urandom(size)
        vm, channel, _ = await _file_io_vm(
            "test-vm", FileWriteAckMessage(op_id="w", path="out.bin", bytes_written=size)
        )
        await vm.write_file("out.bin", io.BytesIO(content))

        frames = _frames(channel)
        actions = [f["action"] for f in frames]
        assert actions == ["write_file", *["file_chunk"] * (len(actions) - 2), "file_end"]
        assert {f["op_id"] for f in frames} == {frames[-1]["op_id"]}
        assert _sent_content(frames) == content
        channel.unregister_op.assert_awaited_once_with(frames[-1]["op_id"])

    async def test_compressible_content_is_compressed(self) -> None:
        """1 MiB of zeros goes on the wire in far fewer bytes than the file."""
        content = bytes(1024 * 1024)
        vm, channel, _ = await _file_io_vm(
            "test-vm", FileWriteAckMessage(op_id="w", path="zeros.bin", bytes_written=len(content))
        )
        await vm.write_file("zeros.bin", io.BytesIO(content))

        frames = _frames(channel)
        wire = sum(len(f["data"]) for f in frames if f["action"] == "file_chunk")
        assert wire < len(content) // 100
        assert _sent_content(frames) == content


class TestWriteFileCommitProtocol:
    """A file_end frame is a commit record, never best-effort cleanup."""

    @pytest.mark.parametrize("seekable", [False, True], ids=["non-seekable", "seekable"])
    async def test_source_failure_after_header_aborts_as_local_disk(self, seekable: bool) -> None:
        """The local source fails after the header and a chunk, before file_end. The guest
        commits a write only on the file_end of its op_id and drops it when the channel
        closes: the host closes the channel and reports nothing written (RETRY: LOCAL_DISK),
        not an unknown outcome (VERIFY, which closes the session). A seekable source goes
        back to where the call found it, so the same call can run again."""

        class FailingSource(io.BytesIO):
            reads = 0

            def read(self, size: int | None = -1) -> bytes:
                self.reads += 1
                if self.reads == 1:
                    return super().read(size)
                raise OSError(errno.EIO, os.strerror(errno.EIO))

            def seekable(self) -> bool:
                return seekable

        source = FailingSource(b"head" + os.urandom(2 * FILE_TRANSFER_CHUNK_SIZE))
        source.seek(4)
        vm, channel, _ = await _file_io_vm("write-commit-vm")
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.LOCAL_DISK and e.context["errno"] == errno.EIO,
            match=rf"^VM write-commit-vm write_file could not read the local source \({os.strerror(errno.EIO)}\)$",
        ) as exc_info:
            await vm.write_file("target.bin", source, make_executable=False)
        assert isinstance(exc_info.value.__cause__, OSError)
        assert [f["action"] for f in _frames(channel)] == ["write_file", "file_chunk"]  # never a file_end
        channel.close.assert_awaited_once()  # the guest drops the partial write
        channel.unregister_op.assert_awaited_once()
        assert source.tell() == (4 if seekable else 4 + FILE_TRANSFER_CHUNK_SIZE)

    async def test_failed_chunk_enqueue_never_starts_the_next_source_read(self) -> None:
        """A failed enqueue cannot return while look-ahead still owns the stream."""

        class CountingReader:
            reads = 0

            def read(self, _size: int) -> bytes:
                self.reads += 1
                if self.reads == 1:
                    return os.urandom(FILE_TRANSFER_CHUNK_SIZE)
                return b""

        async def failing_chunks(data: bytes) -> None:
            if json.loads(data)["action"] == "file_chunk":
                raise OSError("injected enqueue failure")

        reader = CountingReader()
        vm, channel, _ = await _file_io_vm("write-commit-vm")
        channel.enqueue_raw.side_effect = failing_chunks

        with pytest.raises(
            OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN, match="after dispatch"
        ):
            await vm.write_file("target.bin", reader)  # type: ignore[arg-type]

        assert reader.reads == 1

    async def test_prefailed_registration_is_reported_before_dispatch(self) -> None:
        """A dead registered inbox proves that no header reached the write queue."""
        vm, channel, inbox = await _file_io_vm("write-commit-vm")
        inbox.fail("event transport already closed")

        with pytest.raises(
            TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE, match="before dispatch"
        ):
            await vm.write_file("target.bin", io.BytesIO(b"payload"))

        channel.enqueue_raw.assert_not_awaited()

    async def test_ack_timeout_after_commit_is_outcome_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(constants, "FILE_IO_TIMEOUT_SECONDS", 0.01)
        vm, channel, _ = await _file_io_vm("write-commit-vm")
        with pytest.raises(
            OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN, match="after dispatch"
        ) as exc_info:
            await vm.write_file("target.bin", io.BytesIO(b"payload"))

        assert isinstance(exc_info.value.__cause__, TimeoutError)
        assert _frames(channel)[-1]["action"] == "file_end"
        # A host deadline carries the guest console and the QEMU state.
        assert exc_info.value.context.items() >= _DEADLINE_DIAGNOSTICS.items()
        vm.deadline_diagnostics.assert_awaited_once_with(operation="write_file")  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        ("ack_path", "ack_size"),
        [("different.bin", 7), ("target.bin", 6)],
    )
    async def test_ack_must_bind_path_and_source_size(self, ack_path: str, ack_size: int) -> None:
        ack = FileWriteAckMessage(op_id="write-op", path=ack_path, bytes_written=ack_size)
        vm, _channel, _ = await _file_io_vm("write-commit-vm", ack)
        with pytest.raises(
            OutcomeUnknownError,
            check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN,
            match="acknowledgement did not bind",
        ):
            await vm.write_file("target.bin", io.BytesIO(b"payload"))


class TestReadFileTerminalIntegrity:
    """File-read completion metadata must bind the bytes actually received. A local
    destination that cannot take the bytes is the caller's error (ARGUMENT_INVALID) or this
    host's (LOCAL_DISK), never the guest's (GUEST_UNAVAILABLE)."""

    async def test_declared_size_mismatch_is_communication_failure(self) -> None:
        """A completion that declares more bytes than came: a transport fault (the read can
        run again), never a short file reported as complete."""
        complete = FileReadCompleteMessage(op_id="read-op", path="file.bin", size=1)
        vm, _channel, _ = await _file_io_vm("test-vm", complete)
        destination = io.BytesIO()
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE,
            match=r"^VM test-vm read_file communication failed: .*declared_size=1, received_size=0$",
        ):
            await vm.read_file("file.bin", destination=destination)

    async def test_request_names_the_file(self) -> None:
        vm, channel, _ = await _file_io_vm("test-vm", *_guest_file(b"payload"))
        await vm.read_file("file.bin", destination=io.BytesIO())
        (request,) = _frames(channel)
        assert isinstance(request.pop("host_time_ns"), int)
        assert request == {"action": "read_file", "op_id": request["op_id"], "path": "file.bin"}
        channel.unregister_op.assert_awaited_once_with(request["op_id"])

    @pytest.mark.parametrize("parts", [1, 3], ids=["one-chunk", "three-chunks"])
    async def test_stream_destination_gets_every_byte(self, parts: int) -> None:
        """The chunks of one zstd stream decode in order, also across chunk boundaries."""
        payload = os.urandom(2 * FILE_TRANSFER_CHUNK_SIZE + 7)
        vm, _channel, _ = await _file_io_vm("test-vm", *_guest_file(payload, parts=parts))
        destination = io.BytesIO(b"prefix")
        destination.seek(6)
        await vm.read_file("file.bin", destination=destination)
        assert destination.getvalue() == b"prefix" + payload

    @pytest.mark.parametrize("damage", ["first-chunk-lost", "last-chunk-twice"])
    async def test_a_broken_compressed_stream_is_a_guest_error(self, damage: str) -> None:
        """A chunk dropped on the way (bad base64) or sent twice breaks the one zstd stream:
        read_file raises the library's guest error, never a raw ZstdError or EOFError."""
        payload = os.urandom(2 * FILE_TRANSFER_CHUNK_SIZE + 7)
        *chunks, complete = _guest_file(payload, parts=3)
        sent = chunks[1:] if damage == "first-chunk-lost" else [*chunks, chunks[-1]]
        vm, _channel, _ = await _file_io_vm("test-vm", *sent, complete)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.read_file("file.bin", destination=io.BytesIO())
        assert "compressed stream broke" in str(exc_info.value)

    async def test_path_destination_is_written_whole(self, tmp_path: Path) -> None:
        payload = b"payload of the guest file"
        destination = tmp_path / "out" / "file.bin"
        vm, channel, _ = await _file_io_vm("test-vm", *_guest_file(payload))
        await vm.read_file("file.bin", destination=destination)
        assert destination.read_bytes() == payload
        assert sorted(p.name for p in destination.parent.iterdir()) == ["file.bin"]  # no temp file left
        channel.unregister_op.assert_awaited_once()

    @pytest.mark.parametrize("case", ["file-in-path", "no-write-access", "folder-at-destination"])
    async def test_destination_the_caller_can_fix_is_argument_invalid(self, tmp_path: Path, case: str) -> None:
        """A destination path the caller can fix (a file in its path: mkdir ENOTDIR; a folder
        with no write access: EACCES; a folder where the file goes: rename EISDIR) fails the
        same way on every retry: FIX_INPUT (ARGUMENT_INVALID), not RETRY."""
        if case == "file-in-path":
            (tmp_path / "a-file").write_bytes(b"")
            destination, expected = tmp_path / "a-file" / "sub" / "file.bin", errno.ENOTDIR
        elif case == "no-write-access":
            if os.geteuid() == 0:
                pytest.skip("root writes in a folder without write permission")
            (tmp_path / "locked").mkdir()
            (tmp_path / "locked").chmod(0o500)
            destination, expected = tmp_path / "locked" / "file.bin", errno.EACCES
        else:
            destination, expected = tmp_path / "folder", errno.EISDIR
            destination.mkdir()
        vm, channel, _ = await _file_io_vm("test-vm", *_guest_file(b"payload"))
        reason = re.escape(os.strerror(expected))
        try:
            with pytest.raises(
                InvalidInputError,
                check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID and e.context["errno"] == expected,
                match=rf"^VM test-vm read_file could not write the local destination \({reason}\)$",
            ) as exc_info:
                await vm.read_file("file.bin", destination=destination)
        finally:
            if case == "no-write-access":
                (tmp_path / "locked").chmod(0o700)
        assert isinstance(exc_info.value.__cause__, OSError)
        assert exc_info.value.context["path"] == "file.bin"
        channel.unregister_op.assert_awaited_once()
        assert list(tmp_path.rglob("*.tmp")) == []  # no temp file left

    @pytest.mark.parametrize(
        "make_destination",
        [
            pytest.param(lambda p: p / "d\x00ir" / "file.bin", id="nul-byte"),
            pytest.param(lambda _p: Path("/"), id="no-file-name"),
        ],
    )
    async def test_destination_path_value_error_is_argument_invalid(
        self, tmp_path: Path, make_destination: Any
    ) -> None:
        """A NUL byte in the destination path or a destination with no file name raises
        ValueError in Path/os calls: the caller's error, and the cleanup's own ValueError
        (unlink of the temp path) does not hide it. The guest is not asked."""
        vm, channel, _ = await _file_io_vm("test-vm", *_guest_file(b"payload"))
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=r"^VM test-vm read_file could not write the local destination \(.+\)$",
        ) as exc_info:
            await vm.read_file("file.bin", destination=make_destination(tmp_path))
        assert isinstance(exc_info.value.__cause__, ValueError)
        channel.register_op.assert_not_awaited()
        channel.unregister_op.assert_awaited_once()
        assert list(tmp_path.iterdir()) == []

    async def test_host_disk_error_is_local_disk(self) -> None:
        """A full disk can pass later: RETRY (LOCAL_DISK), named by its reason."""

        class FullDisk:
            def write(self, _data: bytes) -> int:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

        vm, _channel, _ = await _file_io_vm("test-vm", *_guest_file(b"payload"))
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.LOCAL_DISK and e.context["errno"] == errno.ENOSPC,
            match=rf"^VM test-vm read_file could not write the local destination \({os.strerror(errno.ENOSPC)}\)$",
        ):
            await vm.read_file("file.bin", destination=FullDisk())  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("make_error", "match"),
        [
            pytest.param(TimeoutError, r"^VM test-vm read_file timed out for 'file\.bin'$", id="timeout"),
            pytest.param(
                lambda: ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"),
                r"^VM test-vm read_file communication failed: .*Connection reset by peer$",
                id="oserror",
            ),
            pytest.param(
                lambda: CommunicationError("channel closed", context=None),
                r"^VM test-vm read_file communication failed: channel closed$",
                id="communication",
            ),
        ],
    )
    @pytest.mark.parametrize("kind", ["path", "stream"])
    async def test_transport_failure_is_guest_unavailable(
        self, tmp_path: Path, make_error: Any, match: str, kind: str
    ) -> None:
        """One transport mapping for both destination kinds: a timeout, a socket error
        or a channel error is the guest's (GUEST_UNAVAILABLE, RETRY), never LOCAL_DISK or a
        builtin. The temp file goes; the op is unregistered once."""
        error = make_error()
        vm, channel, _ = await _file_io_vm("test-vm", *_guest_file(b"payload"))
        channel.connect.side_effect = error
        destination: Path | IO[bytes] = tmp_path / "file.bin" if kind == "path" else io.BytesIO()
        with pytest.raises(
            TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE, match=match
        ) as exc_info:
            await vm.read_file("file.bin", destination=destination)
        assert exc_info.value.__cause__ is error
        channel.unregister_op.assert_awaited_once()
        assert list(tmp_path.iterdir()) == []
        # Only a deadline carries the guest console and the QEMU state.
        has_diagnostics = exc_info.value.context.items() >= _DEADLINE_DIAGNOSTICS.items()
        assert has_diagnostics is isinstance(error, TimeoutError)

    async def test_failed_stream_leaves_no_temp_file(self, tmp_path: Path) -> None:
        """The guest stream fails after the temp file opened: the temp file goes, the
        destination is not created, and the error stays the guest's."""
        failure = StreamingErrorMessage(op_id="read-op", error_type="io_error", message="gone")
        vm, _channel, _ = await _file_io_vm("test-vm", failure)
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
            await vm.read_file("file.bin", destination=tmp_path / "file.bin")
        assert list(tmp_path.iterdir()) == []

    async def test_short_destination_writes_are_completed(self) -> None:
        class ShortWriter:
            def __init__(self) -> None:
                self.value = bytearray()

            def write(self, data: bytes) -> int:
                count = max(1, len(data) // 2)
                self.value.extend(data[:count])
                return count

        payload = b"short writes still preserve every byte"
        writer = ShortWriter()
        vm, _channel, _ = await _file_io_vm("test-vm", *_guest_file(payload))
        await vm.read_file("file.bin", destination=writer)  # type: ignore[arg-type]
        assert bytes(writer.value) == payload

    @pytest.mark.parametrize(
        "written",
        [
            pytest.param(lambda _data: 0, id="zero"),
            pytest.param(lambda _data: None, id="none"),  # a raw non-blocking writer on would-block
            pytest.param(lambda data: len(data) + 1, id="over-reporting"),
        ],
    )
    async def test_destination_write_without_progress_is_rejected(self, written: Any) -> None:
        """A writer that reports no progress (0, or None from a raw non-blocking writer) or
        more bytes than it got cannot report success: LOCAL_DISK (an OSError), not a
        TypeError or a short file."""

        class Writer:
            def write(self, data: bytes) -> int | None:
                return written(data)

        vm, _channel, _ = await _file_io_vm("test-vm", *_guest_file(b"payload"))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK, match="no write progress"):
            await vm.read_file("file.bin", destination=Writer())  # type: ignore[arg-type]


class TestLocalPathError:
    """One errno split for a local path the caller gave (write_file's source, read_file's
    destination): an error the caller can fix is FIX_INPUT (the same call fails again); any
    other is this host's and can pass later (RETRY)."""

    @pytest.mark.parametrize(
        "code",
        [
            errno.ENOENT,
            errno.EACCES,
            errno.EPERM,
            errno.EROFS,
            errno.ENOTDIR,
            errno.EISDIR,
            errno.ELOOP,
            errno.ENAMETOOLONG,
            errno.EEXIST,
        ],
        ids=errno.errorcode.get,
    )
    def test_error_the_caller_can_fix_is_argument_invalid(self, code: int) -> None:
        error = local_path_error(
            OSError(code, os.strerror(code), "/x/y"),
            failure="Source file cannot be opened",
            path=Path("/x/y"),
            context={"k": 1},
        )
        assert isinstance(error, InvalidInputError)
        assert error.code is ErrorCode.ARGUMENT_INVALID
        assert error.message == f"Source file cannot be opened ({os.strerror(code)}): /x/y"
        assert error.context == {"k": 1, "errno": code}

    @pytest.mark.parametrize(
        "code",
        [errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EMFILE, errno.ENFILE, errno.ENOMEM],
        ids=errno.errorcode.get,
    )
    def test_host_error_is_local_disk(self, code: int) -> None:
        error = local_path_error(
            OSError(code, os.strerror(code)), failure="Destination cannot be written", path=None, context={}
        )
        assert isinstance(error, TransientError)
        assert error.code is ErrorCode.LOCAL_DISK
        assert error.message == f"Destination cannot be written ({os.strerror(code)})"
        assert error.context == {"errno": code}

    def test_error_without_errno_is_local_disk_named_by_its_text(self) -> None:
        """A destination that made no write progress (no errno): this host's, its text kept."""
        error = local_path_error(
            OSError("made no write progress"), failure="Destination cannot be written", path=None, context={}
        )
        assert isinstance(error, TransientError)
        assert error.code is ErrorCode.LOCAL_DISK
        assert error.message == "Destination cannot be written (made no write progress)"

    def test_value_error_is_argument_invalid_without_the_path(self) -> None:
        """ValueError (a NUL byte, no file name): the caller's error; the path is not printed."""
        error = local_path_error(
            ValueError("embedded null character in path"),
            failure="Source file cannot be opened",
            path=Path("a\x00b"),
            context={},
        )
        assert isinstance(error, InvalidInputError)
        assert error.code is ErrorCode.ARGUMENT_INVALID
        assert error.message == "Source file cannot be opened (embedded null character in path)"


# ============================================================================
# FileOpDispatcher routing tests
# ============================================================================


def _feed(reader: asyncio.StreamReader, msg: dict[str, object]) -> None:
    """Feed a JSON message into a StreamReader (simulates guest agent output)."""
    reader.feed_data(json.dumps(msg).encode() + b"\n")


async def _received(inbox: OperationInbox) -> list[StreamingMessage]:
    """Every message routed to the inbox, in order. The reader must be at EOF: the
    dispatcher then fails the inbox after its routed messages (CommunicationError), so the
    list is complete with no wait. The bound only ends a failed run."""
    messages: list[StreamingMessage] = []
    while True:
        try:
            messages.append(await asyncio.wait_for(inbox.get(), timeout=1.0))
        except CommunicationError:
            return messages


_WIRE_TYPES = {
    "file_chunk": ("FileChunkResponseMessage", "data", {}),
    "error": ("StreamingErrorMessage", "message", {"error_type": "io_error"}),
    "stdout": ("OutputChunkMessage", "chunk", {}),
    "file_read_complete": ("FileReadCompleteMessage", "path", {"size": 1}),
    "file_write_ack": ("FileWriteAckMessage", "path", {"bytes_written": 1}),
    "file_list": ("FileListMessage", "path", {"entries": []}),
}
"""Event types with a free text field: the message class the host parses each into, the
field that carries the test's marker, and the other required fields."""


def _wire(field: str, marker: str) -> str:
    """The marker as the guest sends that field: a file chunk's data is base64."""
    return base64.b64encode(marker.encode()).decode() if field == "data" else marker


def _marker(message: StreamingMessage) -> str:
    field = next(field for name, field, _ in _WIRE_TYPES.values() if name == type(message).__name__)
    value = getattr(message, field)
    return value.decode() if isinstance(value, bytes) else value


class TestFileOpDispatcherRouting:
    """FileOpDispatcher on a real asyncio.StreamReader fed with guest frames: each message
    goes to the inbox of its op_id, in order; a message with no op_id, or with an op_id that
    has no inbox, goes nowhere."""

    @pytest.mark.parametrize(
        ("frame", "message_type"),
        [
            pytest.param(
                {"type": "error", "error_type": "io_error", "message": "fail"}, StreamingErrorMessage, id="error"
            ),
            pytest.param({"type": "file_chunk", "data": "AAAA"}, FileChunkResponseMessage, id="file_chunk"),
            pytest.param({"type": "pong", "version": "1.0"}, PongMessage, id="pong"),
            pytest.param(
                {"type": "complete", "exit_code": 0, "execution_time_ms": 100}, ExecutionCompleteMessage, id="complete"
            ),
            pytest.param({"type": "stdout", "chunk": "hello"}, OutputChunkMessage, id="stdout"),
            pytest.param({"type": "file_list", "path": "", "entries": []}, FileListMessage, id="file_list"),
            pytest.param(
                {"type": "file_read_complete", "path": "a", "size": 1}, FileReadCompleteMessage, id="file_read_complete"
            ),
            pytest.param(
                {"type": "file_write_ack", "path": "a", "bytes_written": 1}, FileWriteAckMessage, id="file_write_ack"
            ),
        ],
    )
    @pytest.mark.parametrize("op_id", ["op-A", ""], ids=["op-id", "empty-op-id"])
    async def test_message_reaches_the_inbox_of_its_op_id(
        self, frame: dict[str, object], message_type: type[StreamingMessage], op_id: str
    ) -> None:
        """Each event type with an op_id reaches its inbox; an empty op_id is an op_id too."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        inbox = await dispatcher.register_op(op_id)
        _feed(reader, {**frame, "op_id": op_id})
        reader.feed_eof()

        assert [(type(m), m.op_id) for m in await _received(inbox)] == [(message_type, op_id)]
        await dispatcher.stop()

    @pytest.mark.parametrize(
        "stray",
        [
            pytest.param({"type": "pong", "version": "1.0"}, id="no-op-id-pong"),
            pytest.param({"type": "error", "error_type": "timeout", "message": "fail"}, id="no-op-id-error"),
            pytest.param({"type": "pong", "version": "1.0", "op_id": "op-UNKNOWN"}, id="unregistered-pong"),
            pytest.param(
                {"type": "error", "error_type": "io_error", "message": "fail", "op_id": "op-UNKNOWN"},
                id="unregistered-error",
            ),
        ],
    )
    async def test_message_without_a_registered_op_id_is_discarded(self, stray: dict[str, object]) -> None:
        """A message with no op_id, or with an op_id that has no inbox, reaches no inbox; the
        next message is routed as usual."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        inbox = await dispatcher.register_op("op-A")
        _feed(reader, stray)
        _feed(reader, {"type": "file_list", "path": "next", "entries": [], "op_id": "op-A"})
        reader.feed_eof()

        assert [(type(m), _marker(m)) for m in await _received(inbox)] == [(FileListMessage, "next")]
        await dispatcher.stop()

    async def test_message_after_unregister_is_discarded(self) -> None:
        """After unregister_op, a late message for the op reaches no inbox: the old inbox
        raises CommunicationError (it never gets the late message), the other ops go on."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        old = await dispatcher.register_op("op-A")
        _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": _wire("data", "first")})
        first = await asyncio.wait_for(old.get(), timeout=1.0)
        assert isinstance(first, FileChunkResponseMessage)
        assert first.data == b"first"

        await dispatcher.unregister_op("op-A")
        other = await dispatcher.register_op("op-B")
        _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": _wire("data", "late")})
        _feed(reader, {"type": "file_chunk", "op_id": "op-B", "data": _wire("data", "other")})
        reader.feed_eof()

        assert [_marker(m) for m in await _received(other)] == ["other"]
        with pytest.raises(CommunicationError, match=r"^operation unregistered$"):
            await old.get()
        await dispatcher.stop()

    async def test_interleaved_messages_reach_only_their_operation_in_order(self) -> None:
        """100 operations get up to 5 messages each, of every event type, in a shuffled
        order: each inbox gets exactly the messages of its op_id, in the order they came."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        rng = random.Random(42)
        ops = [f"op-{i}" for i in range(100)]
        inboxes = {op: await dispatcher.register_op(op) for op in ops}
        slots = [op for op in ops for _ in range(rng.randint(1, 5))]
        rng.shuffle(slots)
        expected: dict[str, list[tuple[str, str]]] = {op: [] for op in ops}
        for n, op in enumerate(slots):
            wire_type = rng.choice(list(_WIRE_TYPES))
            class_name, field, required = _WIRE_TYPES[wire_type]
            marker = f"{op}/{n}"
            _feed(reader, {"type": wire_type, "op_id": op, field: _wire(field, marker), **required})
            expected[op].append((class_name, marker))
        reader.feed_eof()

        for op in ops:
            received = [(type(m).__name__, _marker(m)) for m in await _received(inboxes[op])]
            assert received == expected[op], op
        await dispatcher.stop()

    async def test_cancelled_get_loses_no_message(self) -> None:
        """A get() cancelled in the loop turn that its message comes in (a wait_for timeout)
        leaves the message in the inbox: the next get() returns it."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        inbox = await dispatcher.register_op("op-A")
        waiting = asyncio.create_task(inbox.get())
        await asyncio.sleep(0)  # get() waits on its empty inbox
        assert await inbox.route(OutputChunkMessage(type="stdout", chunk="kept", op_id="op-A"))
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        message = await asyncio.wait_for(inbox.get(), timeout=1.0)
        assert isinstance(message, OutputChunkMessage)
        assert message.chunk == "kept"
        await dispatcher.stop()


class TestDispatchLoopConnectionReset:
    """The dispatch loop ends on EOF or a socket error (a peer disconnect, memory stress)
    as on a closed transport: each registered operation gets CommunicationError with the
    reason, after the messages routed before it; stop() stays safe."""

    @pytest.mark.parametrize(
        ("end", "reason"),
        [
            pytest.param(
                lambda r: r.set_exception(ConnectionResetError("reset")),
                "^event channel failed: reset$",
                id="ConnectionResetError",
            ),
            pytest.param(
                lambda r: r.set_exception(ConnectionResetError()),
                "^event channel failed: ConnectionResetError$",
                id="ConnectionResetError-no-text",
            ),
            pytest.param(
                lambda r: r.set_exception(BrokenPipeError("broken")),
                "^event channel failed: broken$",
                id="BrokenPipeError",
            ),
            pytest.param(
                lambda r: r.set_exception(OSError(104, "Connection aborted")),
                r"^event channel failed: \[Errno 104\] Connection aborted$",
                id="OSError-custom-errno",
            ),
            pytest.param(lambda r: r.feed_eof(), "^event channel reached EOF with 0 trailing bytes$", id="EOF"),
        ],
    )
    async def test_transport_end_fails_registered_ops_with_its_reason(self, end: Any, reason: str) -> None:
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        inbox = await dispatcher.register_op("op-A")
        await asyncio.sleep(0)  # the loop waits in readuntil()

        end(reader)

        with pytest.raises(CommunicationError, match=reason):
            await asyncio.wait_for(inbox.get(), timeout=1.0)
        assert not dispatcher.is_alive()
        await dispatcher.stop()
        await dispatcher.stop()  # a second stop does nothing
        assert not dispatcher.is_alive()

    async def test_messages_before_reset_are_delivered(self) -> None:
        """A message routed before ConnectionResetError stays in its inbox, ahead of the failure."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        op_queue = await dispatcher.register_op("op-A")
        _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": _wire("data", "before-reset")})
        await asyncio.sleep(0)  # the loop routes the message, then waits in readuntil()

        reader.set_exception(ConnectionResetError())

        msg = await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert isinstance(msg, FileChunkResponseMessage)
        assert msg.data == b"before-reset"
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
            assert await op_queue.route(OutputChunkMessage(type="stdout", chunk=str(index), op_id="op-full"))

        reader.feed_eof()

        messages = [await asyncio.wait_for(op_queue.get(), timeout=1.0) for _ in range(OP_QUEUE_DEPTH)]
        assert [m.chunk for m in messages if isinstance(m, OutputChunkMessage)] == [
            str(index) for index in range(OP_QUEUE_DEPTH)
        ]
        with pytest.raises(CommunicationError, match="EOF"):
            await asyncio.wait_for(op_queue.get(), timeout=1.0)

        await dispatcher.stop()

    async def test_registration_after_eof_fails_immediately(self) -> None:
        """A race registering after dispatcher death cannot wait for a hard timeout."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        first = await dispatcher.register_op("first-op")
        reader.feed_eof()
        with pytest.raises(CommunicationError, match="EOF"):  # the loop ended
            await asyncio.wait_for(first.get(), timeout=1.0)

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
            assert await op_queue.route(OutputChunkMessage(type="stdout", chunk=str(index), op_id="op-complete"))
        assert await op_queue.route(ExecutionCompleteMessage(exit_code=0, execution_time_ms=1, op_id="op-complete"))

        reader.feed_eof()

        messages = [await asyncio.wait_for(op_queue.get(), timeout=1.0) for _ in range(OP_QUEUE_DEPTH)]
        assert [message.chunk for message in messages[:-1] if isinstance(message, OutputChunkMessage)] == [
            str(index) for index in range(OP_QUEUE_DEPTH - 1)
        ]
        assert isinstance(messages[-1], ExecutionCompleteMessage)
        with pytest.raises(CommunicationError, match="EOF"):
            await asyncio.wait_for(op_queue.get(), timeout=1.0)

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
        with pytest.raises(CommunicationError, match="EOF"):
            await asyncio.wait_for(op_queue.get(), timeout=1.0)
        await dispatcher.stop()

    async def test_unregister_full_queue_releases_dispatcher_for_later_operation(self) -> None:
        """A cancelled full inbox cannot wedge the one global dispatch task.

        The setup reads the private queue once: no public call tells that the dispatcher
        waits on a full inbox, and without that state the test proves nothing."""
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
        with pytest.raises(CommunicationError, match="EOF"):
            await asyncio.wait_for(later.get(), timeout=1.0)
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
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN, match="EOF"):
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
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN, match="EOF"):
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

        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN, match="EOF"):
            await asyncio.wait_for(op_queue.get(), timeout=1.0)
        await dispatcher.stop()

    async def test_stop_raises_the_callers_cancel_once_it_has_stopped(self) -> None:
        """The caller of stop() is cancelled while the loop ends: stop() raises that cancel,
        never hides it, and only after the loop has ended and the ops are failed. The
        loop is cancelled before its first step, so it fails no op itself."""
        dispatcher = FileOpDispatcher(asyncio.StreamReader())
        inbox = await dispatcher.register_op("op-A")

        async def start_then_stop() -> None:
            dispatcher.start()
            await dispatcher.stop()  # the loop has not run yet

        stopper = asyncio.create_task(start_then_stop())
        await asyncio.sleep(0)  # stop() has cancelled the loop and waits for its end
        stopper.cancel()

        with pytest.raises(asyncio.CancelledError):
            await stopper
        assert not dispatcher.is_alive()
        with pytest.raises(CommunicationError, match=r"^event dispatcher stopped$"):
            await asyncio.wait_for(inbox.get(), timeout=1.0)

    @pytest.mark.parametrize("close_channel", [False, True], ids=["dispatcher-stop", "channel-close"])
    async def test_stop_in_a_cancel_handler_does_not_cancel_again(self, close_channel: bool) -> None:
        """A cleanup in a cancel handler (destroy_vm in a pool's handler, ephemeral_vm's
        finally) stops the dispatcher, or closes the channel, while its task's cancel count
        is already 1: it ends, with no new cancel, and the cleanup goes on."""
        dispatcher = FileOpDispatcher(asyncio.StreamReader())
        dispatcher.start()
        channel = _make_close_channel(dispatcher)
        steps: list[str] = []

        async def cancelled_then_clean_up() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await (channel.close() if close_channel else dispatcher.stop())
                steps.append("after the stop")  # destroy_vm: kill QEMU
                raise

        task = asyncio.create_task(cancelled_then_clean_up())
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert steps == ["after the stop"]
        assert not dispatcher.is_alive()

    async def test_limit_overrun_wakes_ops_and_stop_remains_safe(self) -> None:
        """An oversized event frame fails active ops and cannot poison close()."""
        reader = asyncio.StreamReader(limit=16)
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        op_queue = await dispatcher.register_op("oversized")
        reader.feed_data(b"x" * 64 + b"\n")

        with pytest.raises(CommunicationError, match="reader limit"):
            await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert not dispatcher.is_alive()
        await dispatcher.stop()


# ============================================================================
# DualPortChannel.close() resiliency
# ============================================================================


def _make_close_channel(dispatcher: FileOpDispatcher | AsyncMock | None) -> DualPortChannel:
    """A DualPortChannel with mocked ports (no sockets) and the given dispatcher."""
    ch = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
    ch._dispatcher = dispatcher  # type: ignore[assignment]
    ch._cmd_channel = AsyncMock(spec=UnixSocketChannel)
    ch._event_channel = AsyncMock(spec=UnixSocketChannel)
    return ch


class TestDualPortChannelCloseResiliency:
    """close() stops the dispatcher, closes both ports, and always drops the dispatcher,
    also when its stop raises: a later call must not use a dispatcher that failed (a
    poisoned dispatcher would block each health check retry)."""

    async def test_close_fails_ops_in_flight_and_closes_both_ports(self) -> None:
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        ch = _make_close_channel(dispatcher)
        inbox = await ch.register_op("op-A")

        await ch.close()

        with pytest.raises(CommunicationError, match=r"^event dispatcher stopped$"):
            await asyncio.wait_for(inbox.get(), timeout=1.0)
        assert not dispatcher.is_alive()
        ch._cmd_channel.close.assert_awaited_once()
        ch._event_channel.close.assert_awaited_once()
        with pytest.raises(RuntimeError, match="channel not connected"):
            await ch.register_op("op-B")

    async def test_close_with_no_dispatcher_still_closes_channels(self) -> None:
        ch = _make_close_channel(None)

        await ch.close()

        ch._cmd_channel.close.assert_awaited_once()
        ch._event_channel.close.assert_awaited_once()

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("dispatch task exploded"), KeyboardInterrupt(), asyncio.CancelledError()],
        ids=["RuntimeError", "KeyboardInterrupt", "caller-cancel"],
    )
    async def test_close_drops_dispatcher_when_stop_raises(self, error: BaseException) -> None:
        """The stop error propagates; both ports are closed all the same, and the
        channel has no dispatcher after it."""
        dispatcher = AsyncMock(spec=FileOpDispatcher)
        dispatcher.stop.side_effect = error
        ch = _make_close_channel(dispatcher)

        with pytest.raises(type(error)):
            await ch.close()

        ch._cmd_channel.close.assert_awaited_once()
        ch._event_channel.close.assert_awaited_once()
        with pytest.raises(RuntimeError, match="channel not connected"):
            await ch.register_op("op-A")

    async def test_close_idempotent(self) -> None:
        """A second close() stops nothing more and closes the ports again without error."""
        dispatcher = AsyncMock(spec=FileOpDispatcher)
        ch = _make_close_channel(dispatcher)

        await ch.close()
        await ch.close()

        dispatcher.stop.assert_awaited_once()
        with pytest.raises(RuntimeError, match="channel not connected"):
            await ch.register_op("op-A")


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
# connect() on a channel that was connected before pings the guest until a pong
# comes. No sockets: _raw_connect (open both sockets) and close are mocks, and
# send_request plays the guest. The tests check what connect() returns or raises,
# the pings sent, and each socket reset (close, then _raw_connect).
# ============================================================================


def _new_channel() -> DualPortChannel:
    """A DualPortChannel that opens no socket (_raw_connect and close are mocks)."""
    channel = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
    channel._raw_connect = AsyncMock()  # type: ignore[method-assign]
    channel.close = AsyncMock()  # type: ignore[method-assign]
    channel.send_request = AsyncMock(return_value=PongMessage(version="1.0"))  # type: ignore[method-assign]
    return channel


async def _make_probe_channel(guest: object) -> DualPortChannel:
    """A channel past its first connect, so the next connect() probes the guest. `guest`
    is the send_request side effect: a list with one answer (a message, or an exception for
    a transport failure) per ping, or a function. The mocks count from zero."""
    channel = _new_channel()
    await channel.connect(timeout_seconds=5.0)
    channel._raw_connect.reset_mock()  # type: ignore[attr-defined]
    channel.send_request = AsyncMock(side_effect=guest)  # type: ignore[method-assign]
    return channel


def _pings(channel: DualPortChannel) -> int:
    """The pings the probe sent: every request is a PingRequest."""
    requests = [call.args[0] for call in channel.send_request.await_args_list]  # type: ignore[attr-defined]
    assert all(isinstance(request, PingRequest) for request in requests), requests
    return len(requests)


class TestDualPortConnectLifecycle:
    """When connect() probes the guest (a reconnection), and when it does not."""

    async def test_first_connect_does_not_probe(self) -> None:
        """Boot-time connect sends no ping."""
        channel = _new_channel()

        await channel.connect(timeout_seconds=5.0)

        channel._raw_connect.assert_awaited_once_with(5.0)  # type: ignore[attr-defined]
        assert _pings(channel) == 0

    async def test_reconnect_does_probe(self) -> None:
        """After a successful connect, the next connect sends one ping and gets the pong."""
        channel = _new_channel()

        await channel.connect(timeout_seconds=5.0)
        await channel.connect(timeout_seconds=5.0)

        assert _pings(channel) == 1

    async def test_idempotent_when_already_connected(self) -> None:
        """connect() on an already-connected channel opens nothing and sends no ping."""
        channel = _new_channel()
        channel._cmd_channel.is_connected = MagicMock(return_value=True)  # type: ignore[method-assign]
        channel._event_channel.is_connected = MagicMock(return_value=True)  # type: ignore[method-assign]

        await channel.connect(timeout_seconds=5.0)

        channel._raw_connect.assert_not_awaited()  # type: ignore[attr-defined]
        assert _pings(channel) == 0

    async def test_connect_after_close_probes(self) -> None:
        """close() does not make the channel new: the connect after it probes the guest."""
        channel = DualPortChannel(cmd_socket="/tmp/cmd.sock", event_socket="/tmp/event.sock", expected_uid=1000)
        channel._raw_connect = AsyncMock()  # type: ignore[method-assign]
        channel.send_request = AsyncMock(return_value=PongMessage(version="1.0"))  # type: ignore[method-assign]

        await channel.connect(timeout_seconds=5.0)
        await channel.close()  # the real close: no socket is open
        await channel.connect(timeout_seconds=5.0)

        assert _pings(channel) == 1


class TestDualPortProbeSuccess:
    """connect() returns when the guest answers a ping in its budget."""

    async def test_guest_ready_immediately(self) -> None:
        """Guest responds with PongMessage on first attempt."""
        channel = await _make_probe_channel([PongMessage(version="1.0")])
        await channel.connect(timeout_seconds=5.0)
        assert _pings(channel) == 1

    async def test_guest_ready_after_transient_timeouts(self) -> None:
        """Guest unresponsive twice then ready."""
        channel = await _make_probe_channel([TimeoutError(), TimeoutError(), PongMessage(version="1.0")])
        await channel.connect(timeout_seconds=5.0)
        assert _pings(channel) == 3

    async def test_guest_ready_after_mixed_transport_errors(self) -> None:
        """Different transport failures then success — probe is resilient."""
        channel = await _make_probe_channel(
            [
                TimeoutError(),
                OSError("reset"),
                ConnectionError("refused"),
                asyncio.IncompleteReadError(b"", 100),
                PongMessage(version="1.0"),
            ]
        )
        await channel.connect(timeout_seconds=5.0)
        assert _pings(channel) == 5

    async def test_guest_ready_on_last_possible_attempt(self) -> None:
        """Guest responds on the very last retry — boundary case."""
        max_retries = len(constants.GUEST_RECONNECT_PROBE_TIMEOUTS)
        channel = await _make_probe_channel([TimeoutError()] * (max_retries - 1) + [PongMessage(version="1.0")])
        await channel.connect(timeout_seconds=5.0)
        assert _pings(channel) == max_retries

    async def test_unanswered_first_ping_costs_a_short_wait(self) -> None:
        """The first ping gets no answer in time: the next ping on the same connection
        ends the probe in ~0.1 s (the first wait is 50 ms)."""
        answered: list[bool] = []

        async def guest(*_args: object, **_kwargs: object) -> PongMessage:
            if not answered:
                answered.append(False)
                await asyncio.Event().wait()  # no answer in time
            return PongMessage(version="1.0")

        channel = await _make_probe_channel(guest)
        loop = asyncio.get_running_loop()
        start = loop.time()
        await channel.connect(timeout_seconds=5.0)

        assert loop.time() - start < 0.3

    async def test_slow_pong_never_resets_the_connection(self) -> None:
        """A ping with no answer yet is slow, not lost: the probe sends the next ping on
        the same connection. Closing it would show the guest a host disconnect for each
        slow pong."""
        channel = await _make_probe_channel(
            [TimeoutError(), TimeoutError(), TimeoutError(), PongMessage(version="1.0")]
        )

        await channel.connect(timeout_seconds=5.0)

        assert _pings(channel) == 4
        channel.close.assert_not_awaited()  # type: ignore[attr-defined]
        channel._raw_connect.assert_awaited_once_with(5.0)  # type: ignore[attr-defined]  # the reconnection only

    async def test_failed_connection_is_opened_again(self) -> None:
        """Only a connection that failed is closed and opened again before the next ping."""
        channel = await _make_probe_channel([ConnectionError("reset"), PongMessage(version="1.0")])

        await channel.connect(timeout_seconds=5.0)

        assert _pings(channel) == 2
        channel.close.assert_awaited_once()  # type: ignore[attr-defined]
        assert channel._raw_connect.await_args_list == [((5.0,),), ((5.0,),)]  # type: ignore[attr-defined]


class TestDualPortProbeFailure:
    """connect() raises TimeoutError when the guest never answers in the probe budget."""

    async def test_all_attempts_timeout(self) -> None:
        """Guest never responds — TimeoutError with descriptive message."""
        max_retries = len(constants.GUEST_RECONNECT_PROBE_TIMEOUTS)
        channel = await _make_probe_channel([TimeoutError()] * max_retries)

        with pytest.raises(TimeoutError, match=f"^Guest agent did not respond to ping after {max_retries} attempts$"):
            await channel.connect(timeout_seconds=5.0)
        # Every ping goes on the same connection.
        assert _pings(channel) == max_retries
        channel._raw_connect.assert_awaited_once()  # type: ignore[attr-defined]  # the reconnection only
        channel.close.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_all_attempts_connection_error(self) -> None:
        """Guest socket broken every time — same TimeoutError."""
        max_retries = len(constants.GUEST_RECONNECT_PROBE_TIMEOUTS)
        channel = await _make_probe_channel([ConnectionError()] * max_retries)

        with pytest.raises(TimeoutError, match="attempts"):
            await channel.connect(timeout_seconds=5.0)
        # One reconnect between attempts, none after the last: it would only delay the error.
        assert channel.close.await_count == max_retries - 1  # type: ignore[attr-defined]
        assert channel._raw_connect.await_count == 1 + (max_retries - 1)  # type: ignore[attr-defined]

    async def test_guest_returns_wrong_message_type(self) -> None:
        """Guest returns StreamingErrorMessage instead of PongMessage — treated as failure."""
        max_retries = len(constants.GUEST_RECONNECT_PROBE_TIMEOUTS)
        wrong_msg = StreamingErrorMessage(error_type="unknown", message="not a pong")
        channel = await _make_probe_channel([wrong_msg] * max_retries)

        with pytest.raises(TimeoutError, match="attempts"):
            await channel.connect(timeout_seconds=5.0)
        assert _pings(channel) == max_retries

    async def test_one_more_failure_than_budget_still_raises(self) -> None:
        """MAX_RETRIES-1 failures + wrong message type = still raises."""
        max_retries = len(constants.GUEST_RECONNECT_PROBE_TIMEOUTS)
        wrong_msg = StreamingErrorMessage(error_type="unknown", message="bad")
        channel = await _make_probe_channel([TimeoutError()] * (max_retries - 1) + [wrong_msg])

        with pytest.raises(TimeoutError):
            await channel.connect(timeout_seconds=5.0)


# ============================================================================
# Idle timer task lifecycle — orphaned task prevention
# ============================================================================


class TestIdleTimerLifecycleNormal:
    """After each close path, no idle deadline stays armed: an armed deadline is a loop
    timer that holds the session (for idle_timeout_seconds: 300 s here), so a closed
    session that the caller drops must be freed at once."""

    async def test_close_disarms_timer(self) -> None:
        session, _, _ = _make_session()
        await session.close()
        session_ref = weakref.ref(session)
        del session
        _assert_freed(session_ref)

    async def test_multiple_exec_then_close_disarms(self) -> None:
        """Each exec() re-arms the deadline; close() disarms the final one."""
        session, _, _ = _make_session()
        for _ in range(5):
            await session.exec("x = 1")
        await session.close()
        session_ref = weakref.ref(session)
        del session
        _assert_freed(session_ref)

    async def test_context_manager_disarms(self) -> None:
        """async with Session disarms via __aexit__."""
        session, _, _ = _make_session()
        async with session:
            await session.exec("x = 1")
        session_ref = weakref.ref(session)
        del session
        _assert_freed(session_ref)


class TestIdleTimerLifecycleEdge:
    """Edge cases for the idle deadline."""

    async def test_double_close_no_error(self) -> None:
        """Second close() is idempotent: no error, one destroy, no deadline armed."""
        session, _, manager = _make_session()
        await session.close()
        await session.close()
        manager.destroy_vm.assert_awaited_once()
        session_ref = weakref.ref(session)
        del session
        _assert_freed(session_ref)

    async def test_close_from_exec_error_path(self) -> None:
        """VM failure in exec() closes the session: the call does not arm a deadline again."""
        session, mock_vm, _ = _make_session()

        async def vm_died(**_kwargs: object) -> ExecutionResult:
            raise RuntimeError("VM died")  # a new error each call: the mock holds no traceback

        mock_vm.execute.side_effect = vm_died
        with pytest.raises(RuntimeError, match="VM died"):
            await session.exec("x = 1")
        assert session.closed
        session_ref = weakref.ref(session)
        del session
        _assert_freed(session_ref)

    async def test_reset_replaces_pending_handle(self) -> None:
        """Re-arming cancels the prior handle and installs a fresh one.

        Private-level: no public call re-arms an armed deadline (_guard cancels it
        first), so only a direct call reaches this defensive path."""
        session, _, _ = _make_session()
        first = session._idle_timer_handle
        session._reset_idle_timer()
        assert session._idle_timer_handle is not first
        assert first is not None and first.cancelled()
        await session.close()

    async def test_natural_idle_timeout_closes_session(self) -> None:
        """The deadline fires and closes the session: one destroy, no error."""
        session, _, manager = _make_session(idle_timeout_seconds=0)
        for _ in range(10):  # call_later(0) fires on the next loop turn
            await asyncio.sleep(0)
            if session.closed:
                break
        assert session.closed
        await session.close()  # waits for the cleanup that the deadline started
        manager.destroy_vm.assert_awaited_once()
        session_ref = weakref.ref(session)
        del session
        _assert_freed(session_ref)


class TestIdleTimerLifecycleStress:
    """Stress scenarios for the idle deadline."""

    async def test_concurrent_close_and_exec(self) -> None:
        """close() starts and does not wait for the lock: an exec() that starts while the VM
        destroy runs finds the session closed, does not reach the VM, and arms no deadline."""
        session, mock_vm, manager = _make_session()
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)  # close() marks the session closed, then waits for the destroy
        with pytest.raises(SessionClosedError, match=r"^Session is closed$"):
            await session.exec("x = 1")
        await closing
        mock_vm.execute.assert_not_awaited()
        manager.destroy_vm.assert_awaited_once()
        session_ref = weakref.ref(session)
        del session
        _assert_freed(session_ref)

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

        with pytest.raises(
            InternalError,
            check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED,
            match="process cleanup was not confirmed",
        ):
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

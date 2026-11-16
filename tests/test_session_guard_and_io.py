# pyright: reportPrivateUsage=false
"""Unit tests of the Session calls (lifecycle checks, lock, idle deadline, write_file
sources, which save() failures close the session), of the QemuVM file calls on a fake
guest channel (frames on the wire, error mapping), of FileOpDispatcher routing, and of the
DualPortChannel reconnection probe.

No VM: the Session tests use a mocked QemuVM (TestWriteFileSessionOverQemuVM: a real
QemuVM on a mocked channel); the QemuVM tests use a real QemuVM on a mocked channel; the
DualPortChannel tests use a real channel on a fake guest agent that serves the two port
sockets (_FakeGuest). Covers:
- Session calls: closed checks, lock serialization, idle deadline suspended during a
  call, fail-fast retirement of dead VMs, close paths, the save() close rule
- write_file sources (bytes, Path, stream): checks, size limits of bytes and a Path, local errors
- QemuVM write_file and read_file: frames sent, content round trip, error mapping; the
  write_file header size, its stop at a guest reply before file_end, the source set back
  after a failure; local_path_error, the errno split of a local path
- FileOpDispatcher message routing by op_id, transport failure
- DualPortChannel reconnection probe, stream deadline and close
"""

import asyncio
import base64
import contextlib
import errno
import gc
import io
import json
import logging
import os
import random
import re
import tempfile
import threading
import time
import tracemalloc
import weakref
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, ClassVar, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox import constants
from exec_sandbox.exceptions import (
    CommunicationError,
    ConfigError,
    ErrorCode,
    InstanceGoneError,
    InternalError,
    InvalidInputError,
    OutcomeUnknownError,
    RouteError,
    SandboxError,
    SessionClosedError,
    TransientError,
)
from exec_sandbox.guest_agent_protocol import (
    ExecuteCodeRequest,
    ExecutionCompleteMessage,
    FileChunkResponseMessage,
    FileListMessage,
    FileReadCompleteMessage,
    FileWriteAckMessage,
    OutputChunkMessage,
    PongMessage,
    StreamingErrorMessage,
    StreamingMessage,
)
from exec_sandbox.guest_channel import (
    OP_QUEUE_DEPTH,
    DualPortChannel,
    FileOpDispatcher,
    OperationInbox,
    StreamDeadlineExceededError,
)
from exec_sandbox.models import ExecutionResult, ExposedPort, FileInfo, Language, StartupTiming, TimingBreakdown
from exec_sandbox.qemu_vm import QemuVM, local_path_error, zstd
from exec_sandbox.session import Session
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType, VmState
from tests.conftest import LoopClock, NonSeekableIO, make_destroy_mock

_CODE_LIMIT = 1024 * 1024
"""Largest code of one exec(): 1 MiB, written out (not the library's constant)."""
_FILE_LIMIT = 500 * 1024 * 1024
"""Largest file of write_file and read_file: 500 MiB, written out (not the library's constant)."""
_CHUNK = 128 * 1024
"""QemuVM.write_file reads a Path or a stream in 128 KiB chunks, and read_file decompresses
in steps of at most 128 KiB: written out."""

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

    session = _session_on(
        mock_vm,
        mock_vm_manager,
        instances=None,
        instance=None,
        idle_timeout_seconds=idle_timeout_seconds,
        default_timeout_seconds=default_timeout_seconds,
    )
    return session, mock_vm, mock_vm_manager


def _session_on(
    vm: QemuVM | AsyncMock,
    manager: AsyncMock,
    *,
    instances: AsyncMock | None,
    instance: MagicMock | None,
    idle_timeout_seconds: int,
    default_timeout_seconds: int,
) -> Session:
    """A Python Session over `vm` and `manager` (a session of no new instance id)."""
    return Session(
        vm=vm,
        vm_manager=manager,
        spec=VmSpec(language=Language.PYTHON),
        startup_timing=_STARTUP,
        instances=instances,
        instance=instance,
        new_instance_id=None,
        idle_timeout_seconds=idle_timeout_seconds,
        default_timeout_seconds=default_timeout_seconds,
    )


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
        """Code one character over 1 MiB: CODE_INVALID with both sizes; the VM is not asked
        and the session stays open."""
        session, mock_vm, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: (
                e.code is ErrorCode.CODE_INVALID
                and e.context == {"code_size": _CODE_LIMIT + 1, "max_code_size": _CODE_LIMIT}
            ),
            match=r"^Code too large: 1048577 characters exceeds the 1048576 character limit$",
        ):
            await session.exec("x" * (_CODE_LIMIT + 1))
        mock_vm.execute.assert_not_called()
        assert not session.closed

    async def test_code_at_limit_accepted(self) -> None:
        """Code of exactly 1 MiB goes to the VM whole, and exec() returns its result."""
        session, mock_vm, _ = _make_session()
        code = "x" * _CODE_LIMIT
        result = await session.exec(code)
        assert result.exit_code == 0
        assert mock_vm.execute.call_args.kwargs["code"] == code


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
    (TransientError too) and the store's answer to the save (a conflict, an unknown outcome,
    an instance deleted meanwhile) leave the VM running, so the session stays open."""

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
            pytest.param(OutcomeUnknownError, ErrorCode.INSTANCE_CONFLICT, None, False, id="conflict"),
            pytest.param(OutcomeUnknownError, ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN, None, False, id="unknown"),
            pytest.param(InstanceGoneError, ErrorCode.INSTANCE_NOT_FOUND, None, False, id="not-found"),
        ],
    )
    async def test_close_rule(
        self, cls: type[SandboxError], code: ErrorCode, context: dict[str, Any] | None, closes: bool
    ) -> None:
        _, mock_vm, mock_vm_manager = _make_session()
        error = _error(cls, code, context)
        instances = AsyncMock()
        instances.save = AsyncMock(side_effect=error)
        session = _session_on(
            mock_vm,
            mock_vm_manager,
            instances=instances,
            instance=None,
            idle_timeout_seconds=300,
            default_timeout_seconds=30,
        )

        with pytest.raises(cls, check=lambda e: e is error):
            await session.save()

        assert session.closed is closes
        assert mock_vm_manager.destroy_vm.await_count == (1 if closes else 0)
        await session.close()

    @pytest.mark.parametrize(
        ("cls", "code"),
        [
            (OutcomeUnknownError, ErrorCode.INSTANCE_CONFLICT),
            (OutcomeUnknownError, ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN),
            (InstanceGoneError, ErrorCode.INSTANCE_NOT_FOUND),
        ],
    )
    async def test_a_close_during_the_save_keeps_the_store_answer(
        self, cls: type[OutcomeUnknownError | InstanceGoneError], code: ErrorCode
    ) -> None:
        """The VM runs again, close() runs, then the store answers: the caller gets the store's
        error with its context (the commit to look for), not SessionClosedError."""
        _, mock_vm, mock_vm_manager = _make_session()
        conflict = _error(cls, code, {"generation": 1, "commit_id": "0" * 32})
        instances = AsyncMock()
        session = _session_on(
            mock_vm,
            mock_vm_manager,
            instances=instances,
            instance=None,
            idle_timeout_seconds=300,
            default_timeout_seconds=30,
        )

        async def save(*_args: object, resumed: Callable[[], None], **_kwargs: object) -> None:
            resumed()  # as InstanceManager.save: before any store answer
            await session.close()
            raise conflict

        instances.save = AsyncMock(side_effect=save)
        with pytest.raises(cls, check=lambda e: e is conflict):
            await session.save()
        assert session.closed

    async def test_close_during_a_failing_save_releases_the_instance_files_once(self) -> None:
        """close() during a save: the save owns the loaded instance's local files, and releases
        them once it ends (here it fails, as close() destroys its VM); close() leaves them to it."""
        _, mock_vm, mock_vm_manager = _make_session()
        save_entered, fail_save = asyncio.Event(), asyncio.Event()
        destroy_entered, end_destroy = asyncio.Event(), asyncio.Event()
        confirmed = make_destroy_mock("confirmed")

        async def save_until_failed(*_args: object, **_kwargs: object) -> None:
            save_entered.set()
            await fail_save.wait()
            raise TransientError("VM gone", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None)

        async def destroy_when_released(vm: object) -> bool:
            destroy_entered.set()
            await end_destroy.wait()
            return await confirmed(vm)

        mock_vm_manager.destroy_vm = AsyncMock(side_effect=destroy_when_released)
        loaded = MagicMock(instance_id="loaded", header=MagicMock(generation=3))
        instances = AsyncMock(save=AsyncMock(side_effect=save_until_failed), release=AsyncMock())
        session = _session_on(
            mock_vm,
            mock_vm_manager,
            instances=instances,
            instance=loaded,
            idle_timeout_seconds=300,
            default_timeout_seconds=30,
        )

        save = asyncio.create_task(session.save())
        await save_entered.wait()
        close = asyncio.create_task(session.close())
        await destroy_entered.wait()
        fail_save.set()
        with pytest.raises(SessionClosedError, match="Session closed during save"):
            await save
        end_destroy.set()
        await close

        assert [c.args for c in instances.release.await_args_list] == [(loaded,)]


# ============================================================================
# _guard() — Normal cases
# ============================================================================


class TestGuardNormal:
    """Happy-path tests for _guard() context manager."""

    async def test_exec_returns_the_vm_result_with_the_session_timing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """exec() returns the VM's result with execute_ms measured by the session around the
        VM call, and the VM's connect_ms. A non-zero exit code (not 137) keeps the session
        open; exec_count counts the call. A test clock moves the loop time."""
        clock = LoopClock(monkeypatch)
        session, mock_vm, _ = _make_session()

        async def execute(**_kwargs: object) -> ExecutionResult:
            await clock.advance(0.25)  # the VM call takes 250 ms
            return ExecutionResult(
                stdout="out\n",
                stderr="err\n",
                exit_code=3,
                execution_time_ms=200,
                external_cpu_time_ms=None,
                external_memory_peak_mb=None,
                timing=TimingBreakdown(execute_ms=9, connect_ms=4),
                spawn_ms=None,
                process_ms=None,
            )

        mock_vm.execute.side_effect = execute
        result = await session.exec("print(1)")

        assert result == ExecutionResult(
            stdout="out\n",
            stderr="err\n",
            exit_code=3,
            execution_time_ms=200,
            external_cpu_time_ms=None,
            external_memory_peak_mb=None,
            timing=TimingBreakdown(execute_ms=250, connect_ms=4),
            spawn_ms=None,
            process_ms=None,
        )
        assert session.exec_count == 1
        assert not session.closed

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
        """list_files() delegates to VM under _guard and returns the VM's entries."""
        session, mock_vm, _ = _make_session()
        result = await session.list_files("subdir")
        mock_vm.list_files.assert_awaited_once_with("subdir")
        assert result == [FileInfo(name="a.txt", is_dir=False, size=10)]

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
# _guard() — A failed call that retired the VM
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


# ============================================================================
# Closed session (first check, before the lock)
# ============================================================================


class TestGuardClosedFastPath:
    """A call on a closed session raises SessionClosedError at its first check, before it
    takes the lock, whatever its arguments."""

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
            pytest.param(lambda s, _p: s.exec("x" * (_CODE_LIMIT + 1)), id="exec-code-size"),
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
# _guard() — Idle deadline re-armed by each call
# ============================================================================


class TestGuardIdleTimerReset:
    """Each call re-arms the idle deadline when it ends: the session closes
    idle_timeout_seconds after the end of its last call, not after its creation. A fake
    loop clock moves the time."""

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda s, _p: s.exec("x = 1"), id="exec"),
            pytest.param(lambda s, _p: s.write_file("reset.txt", b"x"), id="write_file"),
            pytest.param(lambda s, p: s.read_file("any.txt", destination=p / "any.bin"), id="read_file"),
            pytest.param(lambda s, _p: s.list_files(), id="list_files"),
        ],
    )
    async def test_call_rearms_the_deadline(self, call: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = LoopClock(monkeypatch)
        session, _, manager = _make_session(idle_timeout_seconds=10)

        await clock.advance(7)
        await call(session, tmp_path)
        await clock.advance(7)  # 14 s after the creation, 7 s after the call
        assert not session.closed
        await clock.advance(3.5)  # 10.5 s after the call
        assert session.closed

        await session.close()  # waits for the cleanup that the deadline started
        manager.destroy_vm.assert_awaited_once()


# ============================================================================
# _guard() — Concurrent serialization for file I/O
# ============================================================================


class TestGuardConcurrentFileIO:
    """Verify file I/O operations are serialized by the exec lock."""

    async def test_read_and_write_serialized(self, tmp_path: Path) -> None:
        """A read_file and a write_file started at once run one after the other, in call
        order: the second starts once the first has ended."""
        session, mock_vm, _ = _make_session()
        order: list[str] = []

        async def tracked_read(path: str, *, destination: Path) -> None:
            order.append("read_start")
            await asyncio.sleep(0)  # a turn for the write, were it not serialized
            order.append("read_end")

        async def tracked_write(path: str, content: bytes, *, make_executable: bool = False) -> None:
            order.append("write_start")
            await asyncio.sleep(0)
            order.append("write_end")

        mock_vm.read_file = AsyncMock(side_effect=tracked_read)
        mock_vm.write_file = AsyncMock(side_effect=tracked_write)

        await asyncio.gather(
            session.read_file("a.txt", destination=tmp_path / "a.bin"),
            session.write_file("b.txt", b"b"),
        )

        assert order == ["read_start", "read_end", "write_start", "write_end"]
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
    """Size limit (500 MiB) of bytes and of a Path, and streams that cannot be read as bytes.
    A refused source is the caller's error (ARGUMENT_INVALID), found before the call takes
    the session: the VM is not asked and the session stays open. QemuVM.write_file checks the
    size of a stream (TestWriteFileSize)."""

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
        await session.write_file("out.bin", bytes(_FILE_LIMIT))
        assert sizes == [_FILE_LIMIT]

    async def test_bytes_one_over_max_rejected(self) -> None:
        session, mock_vm, _ = _make_session()
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=r"^Content is 524288001 bytes, exceeds 524288000$",
        ):
            await session.write_file("out.bin", bytes(_FILE_LIMIT + 1))
        mock_vm.write_file.assert_not_awaited()
        assert not session.closed

    async def test_path_exactly_at_max_size_accepted(self, tmp_path: Path) -> None:
        source = _sparse_file(tmp_path / "exact.bin", _FILE_LIMIT)
        session, mock_vm, _ = _make_session()
        sizes = _record_upload_sizes(mock_vm)
        await session.write_file("out.bin", source)
        assert sizes == [_FILE_LIMIT]

    async def test_path_one_over_max_rejected(self, tmp_path: Path) -> None:
        source = _sparse_file(tmp_path / "over.bin", _FILE_LIMIT + 1)
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

    async def test_non_seekable_stream_goes_to_the_vm_as_it_is(self) -> None:
        """A non-seekable stream passes Session's checks and goes to the VM as it is (QemuVM
        sends it with no size; the guest enforces the limit as it receives the bytes)."""
        session, mock_vm, _ = _make_session()
        uploads = _record_uploads(mock_vm)
        buf = io.BufferedReader(NonSeekableIO(b"x" * 100))
        await session.write_file("out.bin", buf)
        assert [(u.stream, u.data) for u in uploads] == [(buf, b"x" * 100)]

    async def test_stream_is_measured_by_the_vm_not_before_the_lock(self) -> None:
        """Session seeks nothing in a caller stream before its lock (a seek to the end can read
        a compressed stream whole, on the event loop): QemuVM.write_file measures it, off the
        loop (TestWriteFileSize)."""
        session, mock_vm, _ = _make_session()
        buf = _at(io.BytesIO(b"prefix" + b"data"), 6)
        with patch.object(buf, "seek", wraps=buf.seek) as seek:
            await session.write_file("out.bin", buf)
        mock_vm.write_file.assert_awaited_once()
        seek.assert_not_called()


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
    """A real QemuVM on a fake guest channel. register_op gives one inbox. The guest's `answers`
    go into it when a write's file_end, or any other request, is queued (route() of a non-full
    inbox does not yield: they are in the inbox when the enqueue returns); a test that needs a
    reply sooner routes it into the returned inbox. enqueue_registered is the real channel code;
    enqueue_raw records each frame. deadline_diagnostics gives _DEADLINE_DIAGNOSTICS."""
    inbox = OperationInbox(changes_state=True)
    channel = AsyncMock(spec=DualPortChannel)
    channel.register_op.return_value = inbox

    async def answer() -> None:
        for message in answers:
            assert await inbox.route(message)

    async def enqueue_registered(registered: OperationInbox, data: bytes) -> None:
        await DualPortChannel.enqueue_registered(channel, registered, data)  # type: ignore[arg-type]
        if json.loads(data)["action"] != "write_file":
            await answer()

    async def enqueue_raw(data: bytes) -> None:
        if json.loads(data)["action"] == "file_end":
            await answer()

    channel.enqueue_registered.side_effect = enqueue_registered
    channel.enqueue_raw.side_effect = enqueue_raw
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
        gvproxy_proc=None,
        qemu_log_task=None,
        gvproxy_log_task=None,
        device_layout="",
    )
    vm.deadline_diagnostics = AsyncMock(return_value=_DEADLINE_DIAGNOSTICS)  # type: ignore[method-assign]
    return vm, channel, inbox


def _on_frame(channel: AsyncMock, *, action: str, then: Callable[[], Awaitable[object]]) -> None:
    """Run `then` when the first frame of `action` is queued (after the frame is recorded),
    before the enqueue returns: a guest reply to that frame, a transport failure or a cancel
    at that point."""
    previous = channel.enqueue_raw.side_effect
    done = False

    async def enqueue_raw(data: bytes) -> None:
        nonlocal done
        await previous(data)
        if not done and json.loads(data)["action"] == action:
            done = True
            await then()

    channel.enqueue_raw.side_effect = enqueue_raw


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


def _zeros_stream(size: int) -> list[StreamingMessage]:
    """The guest's read_file messages for `size` zero bytes in one chunk message: zstd holds
    each 128 KiB block of zeros in a few bytes. The zeros go to the compressor 1 MiB at a time
    (no file-size buffer)."""
    compressor = zstd.ZstdCompressor(level=3)
    mib = bytes(1024 * 1024)
    parts = [compressor.compress(mib) for _ in range(size // len(mib))]
    compressed = b"".join([*parts, compressor.compress(bytes(size % len(mib))), compressor.flush()])
    return [
        FileChunkResponseMessage(op_id="read-op", data=compressed),
        FileReadCompleteMessage(op_id="read-op", path="file.bin", size=size),
    ]


class _WriteSizes(io.RawIOBase):
    """A read_file destination that keeps only the size of each write."""

    def __init__(self) -> None:
        super().__init__()
        self.sizes: list[int] = []

    def writable(self) -> bool:
        return True

    def write(self, b: Any) -> int:
        self.sizes.append(len(b))
        return len(b)


class TestWriteFilePathValidation:
    """QemuVM.write_file, read_file and list_files check the path before they send anything
    (the guest checks it again): a path the protocol cannot carry is PATH_INVALID, never a raw
    pydantic error."""

    @pytest.mark.parametrize(
        ("call", "message"),
        [
            (lambda vm: vm.read_file("", destination=io.BytesIO()), "read_file path must not be empty"),
            (
                lambda vm: vm.read_file("x" * 4097, destination=io.BytesIO()),
                "read_file path is 4097 chars, exceeds 4096",
            ),
            (lambda vm: vm.list_files("x" * 4097), "list_files path is 4097 chars, exceeds 4096"),
        ],
        ids=["read-empty", "read-too-long", "list-too-long"],
    )
    async def test_read_and_list_check_the_path_first(self, call: Any, message: str) -> None:
        vm, channel, _ = await _file_io_vm("test-vm")
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID) as exc:
            await call(vm)
        assert exc.value.message == message
        channel.connect.assert_not_awaited()

    async def test_empty_path_rejected(self) -> None:
        vm, channel, _ = await _file_io_vm("test-vm")
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.PATH_INVALID,
            match=r"^write_file path must not be empty$",
        ):
            await vm.write_file("", io.BytesIO(b"data"), make_executable=False)
        channel.connect.assert_not_awaited()
        channel.enqueue_raw.assert_not_awaited()

    async def test_path_at_max_length_accepted(self) -> None:
        """A path of 4096 characters (PATH_MAX) goes to the guest."""
        path = "x" * 4096
        vm, channel, _ = await _file_io_vm("test-vm", FileWriteAckMessage(op_id="w", path=path, bytes_written=4))
        await vm.write_file(path, io.BytesIO(b"data"), make_executable=False)
        assert _frames(channel)[0]["path"] == path

    async def test_path_over_max_length_rejected(self) -> None:
        vm, channel, _ = await _file_io_vm("test-vm")
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.PATH_INVALID,
            match=r"^write_file path is 4097 chars, exceeds 4096$",
        ):
            await vm.write_file("x" * 4097, io.BytesIO(b"data"), make_executable=False)
        channel.connect.assert_not_awaited()
        channel.enqueue_raw.assert_not_awaited()

    async def test_dropped_header_error_is_not_retried(self) -> None:
        """The guest never drops a request (it keeps its ports open), so a "No active
        write" protocol error is a bug to report: it propagates at once, with no retry
        and no channel reset."""
        error = StreamingErrorMessage(op_id="w", error_type="protocol_error", message="No active write for op_id 'x'")
        vm, channel, _ = await _file_io_vm("protocol-error-vm", error)

        with pytest.raises(
            InternalError,
            check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL,
            match=r"^write_file 'target\.bin': \[protocol_error\] No active write for op_id 'x'$",
        ):
            await vm.write_file("target.bin", io.BytesIO(b"data"), make_executable=False)

        actions = [f["action"] for f in _frames(channel)]  # one write: one header, one file_end
        assert actions == ["write_file", *["file_chunk"] * (len(actions) - 2), "file_end"]
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
        clock at send time (the guest steps its clock forward to it) and the size of the source."""
        vm, channel, _ = await _file_io_vm("test-vm", FileWriteAckMessage(op_id="w", path=path, bytes_written=4))
        before = time.time_ns()
        await vm.write_file(path, io.BytesIO(b"data"), make_executable=make_executable)
        after = time.time_ns()

        header, *_, end = _frames(channel)
        op_id = end["op_id"]
        assert before <= header.pop("host_time_ns") <= after
        assert header == {
            "action": "write_file",
            "op_id": op_id,
            "path": path,
            "make_executable": make_executable,
            "size": 4,
        }
        assert end == {"action": "file_end", "op_id": op_id}
        assert re.fullmatch(r"[0-9a-f]{32}", op_id)

    @pytest.mark.parametrize(
        "size",
        [0, 30, 2 * _CHUNK + 7],
        ids=["empty", "small", "three-chunks"],
    )
    async def test_content_round_trips_through_the_chunks(self, size: int) -> None:
        """The file_chunk frames decode (base64, then one zstd stream) to the source bytes;
        the ack binds the byte count."""
        content = os.urandom(size)
        vm, channel, _ = await _file_io_vm(
            "test-vm", FileWriteAckMessage(op_id="w", path="out.bin", bytes_written=size)
        )
        await vm.write_file("out.bin", io.BytesIO(content), make_executable=False)

        frames = _frames(channel)
        actions = [f["action"] for f in frames]
        assert actions == ["write_file", *["file_chunk"] * (len(actions) - 2), "file_end"]
        assert {f["op_id"] for f in frames} == {frames[-1]["op_id"]}
        assert _sent_content(frames) == content
        channel.unregister_op.assert_called_once_with(frames[-1]["op_id"])

    async def test_compressible_content_is_compressed(self) -> None:
        """1 MiB of zeros goes on the wire in far fewer bytes than the file."""
        content = bytes(1024 * 1024)
        vm, channel, _ = await _file_io_vm(
            "test-vm", FileWriteAckMessage(op_id="w", path="zeros.bin", bytes_written=len(content))
        )
        await vm.write_file("zeros.bin", io.BytesIO(content), make_executable=False)

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

        source = FailingSource(b"head" + os.urandom(2 * _CHUNK))
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
        channel.unregister_op.assert_called_once()
        assert source.tell() == (4 if seekable else 4 + _CHUNK)

    async def test_failed_chunk_enqueue_never_starts_the_next_source_read(self) -> None:
        """A failed enqueue cannot return while look-ahead still owns the stream."""

        class CountingReader:
            reads = 0

            def read(self, _size: int) -> bytes:
                self.reads += 1
                if self.reads == 1:
                    return os.urandom(_CHUNK)
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
            await vm.write_file("target.bin", reader, make_executable=False)  # type: ignore[arg-type]

        assert reader.reads == 1

    async def test_prefailed_registration_is_reported_before_dispatch(self) -> None:
        """A dead registered inbox proves that no header reached the write queue."""
        vm, channel, inbox = await _file_io_vm("write-commit-vm")
        inbox.fail("event transport already closed")

        with pytest.raises(
            TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE, match="before dispatch"
        ):
            await vm.write_file("target.bin", io.BytesIO(b"payload"), make_executable=False)

        channel.enqueue_raw.assert_not_awaited()

    async def test_ack_timeout_after_commit_is_outcome_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(constants, "FILE_IO_TIMEOUT_SECONDS", 0.01)
        vm, channel, _ = await _file_io_vm("write-commit-vm")
        with pytest.raises(
            OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN, match="after dispatch"
        ) as exc_info:
            await vm.write_file("target.bin", io.BytesIO(b"payload"), make_executable=False)

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
        """An ack for another path or another byte count than the 7 bytes sent: an unknown
        outcome, with both sides in the context."""
        ack = FileWriteAckMessage(op_id="write-op", path=ack_path, bytes_written=ack_size)
        vm, _channel, _ = await _file_io_vm("write-commit-vm", ack)
        with pytest.raises(
            OutcomeUnknownError,
            check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN,
            match=r"^VM write-commit-vm write_file acknowledgement did not bind the requested content for 'target\.bin'$",
        ) as exc_info:
            await vm.write_file("target.bin", io.BytesIO(b"payload"), make_executable=False)
        assert exc_info.value.context == {
            "vm_id": "write-commit-vm",
            "path": "target.bin",
            "ack_path": ack_path,
            "source_bytes": 7,
            "ack_bytes_written": ack_size,
        }


class _CountingSource(io.BytesIO):
    """A seekable write_file source that counts its reads."""

    reads = 0

    def read(self, size: int | None = -1) -> bytes:
        self.reads += 1
        return super().read(size)


class _NoEndSource(io.BytesIO):
    """A seekable source with no end to seek to, as a procfs file (seq_lseek: EINVAL)."""

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_END:
            raise OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        return super().seek(offset, whence)


def _guest_refusal() -> StreamingErrorMessage:
    """The guest's refusal of a write that does not fit (its free-space check)."""
    return StreamingErrorMessage(
        op_id="w",
        error_type="io_error",
        message="No space for 'out.bin': needs 9 bytes, 4 bytes free: No space left on device (os error 28)",
    )


class TestWriteFileSize:
    """The size in the write_file header: the bytes left in a seekable source, measured by
    QemuVM.write_file in one job, after the session lock."""

    @pytest.mark.parametrize(
        ("make_source", "content", "size"),
        [
            (lambda: _at(io.BytesIO(b"abcdef"), 2), b"cdef", 4),
            (lambda: _at(io.BytesIO(b"abc"), 10), b"", 0),
            (lambda: io.BufferedReader(NonSeekableIO(b"abcdef")), b"abcdef", None),
            (lambda: _at(_NoEndSource(b"abcdef"), 1), b"bcdef", None),
        ],
        ids=["positioned", "past-its-end", "non-seekable", "no-end-to-seek"],
    )
    async def test_header_size_is_the_bytes_left(
        self, make_source: Callable[[], IO[bytes]], content: bytes, size: int | None
    ) -> None:
        """A stream past its end has 0 bytes left (never a negative size); a non-seekable
        stream, or one that cannot seek to its end (a procfs file), sends no size and is
        still written whole."""
        vm, channel, _ = await _file_io_vm(
            "size-vm", FileWriteAckMessage(op_id="w", path="out.bin", bytes_written=len(content))
        )
        await vm.write_file("out.bin", make_source(), make_executable=False)
        frames = _frames(channel)
        assert frames[0].get("size") == size
        assert ("size" in frames[0]) is (size is not None)
        assert _sent_content(frames) == content

    async def test_source_over_the_limit_is_refused_before_any_request(self) -> None:
        """A stream over the limit (its only size check; also a Path that grew after Session's
        check): the caller's error, before anything is sent, never a raw pydantic error."""
        source = io.BytesIO(bytes(_FILE_LIMIT + 1))
        vm, channel, _ = await _file_io_vm("size-vm")
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=r"^Stream content is 524288001 bytes, exceeds 524288000$",
        ):
            await vm.write_file("out.bin", source, make_executable=False)
        channel.connect.assert_not_awaited()
        assert source.tell() == 0

    async def test_limit_counts_from_the_stream_position(self) -> None:
        """One byte over the limit, read from offset 1: accepted, with the limit as its size
        (the guest refuses it at the header here, so nothing is read)."""
        source = _at(io.BytesIO(bytes(_FILE_LIMIT + 1)), 1)
        vm, channel, inbox = await _file_io_vm("size-vm")
        assert await inbox.route(_guest_refusal())
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
            await vm.write_file("out.bin", source, make_executable=False)
        assert _frames(channel)[0]["size"] == _FILE_LIMIT

    async def test_source_is_measured_and_set_back_off_the_event_loop(self) -> None:
        """Every seek of the source runs in a worker thread: the measure (a seek to the end) and
        the seek back after a refusal (a compressed stream reads itself again)."""

        class ThreadWatched(io.BytesIO):
            seeks: ClassVar[list[tuple[int, int]]] = []

            def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
                self.seeks.append((whence, threading.get_ident()))
                return super().seek(offset, whence)

        source = ThreadWatched(os.urandom(3 * _CHUNK))
        vm, channel, inbox = await _file_io_vm("size-vm")
        _on_frame(channel, action="file_chunk", then=lambda: inbox.route(_guest_refusal()))
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
            await vm.write_file("out.bin", source, make_executable=False)
        assert source.tell() == 0  # set back after the read of the first chunk
        assert os.SEEK_END in [whence for whence, _ in source.seeks]  # the measure
        assert threading.get_ident() not in [thread for _, thread in source.seeks]

    async def test_failed_seek_back_is_a_local_source_error(self) -> None:
        """The seek back to the start fails after the seek to the end: the stream moved, so
        the call cannot send it from its position (an empty file would land)."""

        class MovedSource(io.BytesIO):
            def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
                if whence == os.SEEK_SET:
                    raise OSError(errno.EIO, os.strerror(errno.EIO))
                return super().seek(offset, whence)

        vm, channel, _ = await _file_io_vm("size-vm")
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.LOCAL_DISK,
            match=r"^VM size-vm write_file could not read the local source",
        ):
            await vm.write_file("out.bin", MovedSource(b"abc"), make_executable=False)
        channel.connect.assert_not_awaited()


def _session_over(vm: QemuVM) -> tuple[Session, AsyncMock]:
    """A Session over a real QemuVM (of _file_io_vm), and its mocked VmManager."""
    manager = AsyncMock()
    manager.destroy_vm = make_destroy_mock("unconfirmed")  # a real QemuVM has no state setter
    session = _session_on(
        vm, manager, instances=None, instance=None, idle_timeout_seconds=300, default_timeout_seconds=30
    )
    return session, manager


class TestWriteFileSessionOverQemuVM:
    """Session.write_file over a real QemuVM on a fake guest channel: the size that each
    source kind sends, and the refusals after the session lock that keep the session open."""

    @pytest.mark.parametrize("kind", ["bytes", "path"])
    async def test_source_sends_its_size(self, tmp_path: Path, kind: str) -> None:
        """bytes and a Path send their size in the header (the guest reserves it)."""
        data = b"0123456789"
        vm, channel, _ = await _file_io_vm(
            "session-vm", FileWriteAckMessage(op_id="w", path="out.bin", bytes_written=len(data))
        )
        session, _ = _session_over(vm)
        source: bytes | Path = data
        if kind == "path":
            source = tmp_path / "source.bin"
            source.write_bytes(data)
        await session.write_file("out.bin", source)
        assert _frames(channel)[0]["size"] == len(data)

    async def test_refusals_keep_the_session_open(self, tmp_path: Path) -> None:
        """A stream over the limit (ARGUMENT_INVALID: QemuVM measures it after the lock, before
        any connect) and a guest refusal of the first chunk (GUEST_IO): the file was not
        written, the session and its VM stay, and the stream is back at its start."""
        vm, channel, inbox = await _file_io_vm("session-vm")
        session, manager = _session_over(vm)
        channel.connect.side_effect = AssertionError("the stream over the limit reached the channel")
        with (
            _sparse_file(tmp_path / "over.bin", _FILE_LIMIT + 1).open("rb") as over,
            pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID),
        ):
            await session.write_file("out.bin", over)
        channel.connect.side_effect = None
        _on_frame(channel, action="file_chunk", then=lambda: inbox.route(_guest_refusal()))
        stream = io.BytesIO(os.urandom(3 * _CHUNK))
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
            await session.write_file("out.bin", stream)
        assert stream.tell() == 0  # read, then set back
        assert not session.closed
        manager.destroy_vm.assert_not_awaited()


def _at[S: IO[bytes]](stream: S, position: int) -> S:
    """`stream` set to `position`: a source that the call finds away from its start."""
    stream.seek(position)
    return stream


class TestWriteFileEarlyStop:
    """Before file_end the guest replies only to end the write (an error). write_file looks
    for a reply before each chunk and before file_end: it then reads and sends nothing more,
    sends no file_end, raises the reply's error, and sets a seekable source back to where the
    call found it."""

    async def test_reply_to_the_header_stops_before_any_chunk(self) -> None:
        """A refusal of the header, in the inbox before the first chunk: no chunk is read or
        sent, the channel stays open, and the source is back at 4."""
        source = _at(_CountingSource(b"head" + os.urandom(3 * _CHUNK)), 4)
        vm, channel, inbox = await _file_io_vm("stop-vm")
        assert await inbox.route(_guest_refusal())
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.GUEST_IO,
            match=r"^write_file 'out\.bin': \[io_error\] No space for 'out\.bin'",
        ):
            await vm.write_file("out.bin", source, make_executable=False)
        assert [f["action"] for f in _frames(channel)] == ["write_file"]
        assert source.reads == 0
        assert source.tell() == 4
        channel.close.assert_not_awaited()
        channel.unregister_op.assert_called_once()

    async def test_reply_to_a_chunk_stops_the_reads(self) -> None:
        """A refusal of the first chunk: the next source read never starts, and the source is
        back at 0."""
        source = _CountingSource(os.urandom(3 * _CHUNK))  # incompressible: a frame per read
        vm, channel, inbox = await _file_io_vm("stop-vm")
        _on_frame(channel, action="file_chunk", then=lambda: inbox.route(_guest_refusal()))
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
            await vm.write_file("out.bin", source, make_executable=False)
        assert [f["action"] for f in _frames(channel)] == ["write_file", "file_chunk"]
        assert source.reads == 1
        assert source.tell() == 0

    async def test_reply_to_the_last_chunk_sends_no_file_end(self) -> None:
        """30 bytes: the compressor buffers them, so the flush frame is the only chunk."""
        vm, channel, inbox = await _file_io_vm("stop-vm")
        _on_frame(channel, action="file_chunk", then=lambda: inbox.route(_guest_refusal()))
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
            await vm.write_file("out.bin", io.BytesIO(os.urandom(30)), make_executable=False)
        assert [f["action"] for f in _frames(channel)] == ["write_file", "file_chunk"]

    async def test_ack_before_file_end_is_a_protocol_error(self) -> None:
        """Only an error comes before file_end: an ack there never reads as success."""
        vm, channel, inbox = await _file_io_vm("stop-vm")
        assert await inbox.route(FileWriteAckMessage(op_id="w", path="out.bin", bytes_written=4))
        with pytest.raises(
            InternalError,
            check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL,
            match=r"^write_file unexpected response type: FileWriteAckMessage$",
        ):
            await vm.write_file("out.bin", io.BytesIO(b"data"), make_executable=False)
        assert [f["action"] for f in _frames(channel)] == ["write_file"]
        channel.close.assert_not_awaited()

    async def test_transport_failure_is_raised_at_the_next_chunk(self) -> None:
        """The event channel fails mid-stream: the write stops before the next chunk (an unknown
        outcome: the header was sent), not after the whole source."""
        source = _CountingSource(os.urandom(3 * _CHUNK))
        vm, channel, inbox = await _file_io_vm("stop-vm")

        async def fail() -> None:
            inbox.fail("event channel reached EOF with 0 trailing bytes")

        _on_frame(channel, action="file_chunk", then=fail)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN):
            await vm.write_file("out.bin", source, make_executable=False)
        assert [f["action"] for f in _frames(channel)] == ["write_file", "file_chunk"]
        assert source.reads == 1

    async def test_reply_routed_before_a_failure_is_raised_first(self) -> None:
        """The guest's refusal, then the transport's end: the refusal is the outcome."""
        vm, channel, inbox = await _file_io_vm("stop-vm")

        async def refuse_then_close() -> None:
            assert await inbox.route(_guest_refusal())
            inbox.fail("event channel reached EOF with 0 trailing bytes")

        _on_frame(channel, action="write_file", then=refuse_then_close)
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
            await vm.write_file("out.bin", io.BytesIO(b"data"), make_executable=False)

    async def test_cancel_sets_a_seekable_source_back(self) -> None:
        """A cancel is a failure too: the source is back where the call found it."""
        source = _at(io.BytesIO(b"head" + os.urandom(3 * _CHUNK)), 4)
        vm, channel, _ = await _file_io_vm("stop-vm")
        task = asyncio.current_task()
        assert task is not None

        async def cancel() -> None:
            task.cancel()

        _on_frame(channel, action="file_chunk", then=cancel)
        with pytest.raises(asyncio.CancelledError):
            await vm.write_file("out.bin", source, make_executable=False)
        task.uncancel()
        assert source.tell() == 4

    async def test_failed_seek_back_never_replaces_the_error(self) -> None:
        """The source cannot seek back after the guest's refusal: the refusal is raised."""

        class OneWaySource(_CountingSource):
            refused_seeks = 0

            def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
                if self.reads > 0:
                    self.refused_seeks += 1
                    raise ValueError("seek after a read")
                return super().seek(offset, whence)

        source = OneWaySource(os.urandom(3 * _CHUNK))
        vm, channel, inbox = await _file_io_vm("stop-vm")
        _on_frame(channel, action="file_chunk", then=lambda: inbox.route(_guest_refusal()))
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
            await vm.write_file("out.bin", source, make_executable=False)
        assert source.refused_seeks == 1  # the seek back ran, and failed

    async def test_source_failure_with_a_failing_seek_is_still_local_disk(self) -> None:
        """The source read fails, then its seek back fails too: still LOCAL_DISK (nothing
        written: the code that keeps a session open), never an unknown outcome."""

        class BrokenSource(io.BytesIO):
            read_failed = False

            def read(self, size: int | None = -1) -> bytes:
                self.read_failed = True
                raise OSError(errno.EIO, os.strerror(errno.EIO))

            def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
                if self.read_failed:
                    raise OSError(errno.EIO, os.strerror(errno.EIO))
                return super().seek(offset, whence)

        vm, channel, _ = await _file_io_vm("stop-vm")
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
            await vm.write_file("out.bin", BrokenSource(b"abc"), make_executable=False)
        channel.close.assert_awaited_once()


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
        """One read_file frame names the file and carries the host clock at send time."""
        vm, channel, _ = await _file_io_vm("test-vm", *_guest_file(b"payload"))
        before = time.time_ns()
        await vm.read_file("file.bin", destination=io.BytesIO())
        after = time.time_ns()
        (request,) = _frames(channel)
        op_id = request["op_id"]
        assert before <= request.pop("host_time_ns") <= after
        assert request == {"action": "read_file", "op_id": op_id, "path": "file.bin"}
        assert re.fullmatch(r"[0-9a-f]{32}", op_id)
        channel.unregister_op.assert_called_once_with(op_id)

    @pytest.mark.parametrize("parts", [1, 3], ids=["one-chunk", "three-chunks"])
    @pytest.mark.parametrize("content", ["random", "16-values"])
    async def test_stream_destination_gets_every_byte(self, parts: int, content: str) -> None:
        """The chunks of one zstd stream decode in order, also across chunk boundaries, and
        also when a chunk that is not the last decodes in several steps (16 byte values: 1.5 MiB
        in about half the size, each of 3 chunks ~512 KiB decoded)."""
        if content == "random":
            payload = os.urandom(2 * _CHUNK + 7)
        else:
            payload = os.urandom(12 * _CHUNK).translate(bytes(i % 16 for i in range(256)))
        vm, _channel, _ = await _file_io_vm("test-vm", *_guest_file(payload, parts=parts))
        destination = io.BytesIO(b"prefix")
        destination.seek(6)
        await vm.read_file("file.bin", destination=destination)
        assert destination.getvalue() == b"prefix" + payload

    @pytest.mark.parametrize("damage", ["first-chunk-lost", "last-chunk-twice", "frame-end-lost", "data-after-the-end"])
    async def test_a_broken_compressed_stream_is_a_guest_error(self, damage: str) -> None:
        """A chunk dropped on the way (bad base64) or sent twice breaks the one zstd stream, and
        so do a lost frame end after the whole content and bytes after the frame end in its
        message (the size matches): read_file raises the library's guest error, never a raw
        ZstdError or EOFError, nor a file as complete."""
        payload = os.urandom(2 * _CHUNK + 7)
        *chunks, complete = _guest_file(payload, parts=3)
        if damage == "frame-end-lost":
            compressor = zstd.ZstdCompressor(level=3)
            content = compressor.compress(payload) + compressor.flush(zstd.ZstdCompressor.FLUSH_BLOCK)
            sent: list[StreamingMessage] = [FileChunkResponseMessage(op_id="read-op", data=content)]  # no flush()
        elif damage == "data-after-the-end":
            whole = zstd.compress(payload, level=3) + b"garbage"
            sent = [FileChunkResponseMessage(op_id="read-op", data=whole)]
        else:
            sent = chunks[1:] if damage == "first-chunk-lost" else [*chunks, chunks[-1]]
        vm, _channel, _ = await _file_io_vm("test-vm", *sent, complete)
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE,
            # The end is zstd's own text, or one of exec-sandbox's (frame end, data after it)
            match=r"^VM test-vm read_file communication failed: read_file 'file\.bin': the compressed stream broke: .+$",
        ):
            await vm.read_file("file.bin", destination=io.BytesIO())

    async def test_a_message_that_expands_is_written_in_chunks(self) -> None:
        """One chunk message that decodes to far more than its size (64 MiB of zeros in a few
        KB) reaches the destination in writes of at most 128 KiB, every byte: the host never
        holds the whole expansion."""
        chunk, complete = _zeros_stream(64 * 1024 * 1024)
        destination = _WriteSizes()
        vm, _channel, _ = await _file_io_vm("test-vm", chunk, complete)

        tracemalloc.start()
        try:
            await vm.read_file("file.bin", destination=destination)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert isinstance(chunk, FileChunkResponseMessage) and len(chunk.data) < 8 * 1024
        assert (max(destination.sizes), sum(destination.sizes)) == (_CHUNK, 64 * 1024 * 1024)
        assert peak < 8 * 1024 * 1024  # 0.5 MiB measured; holding the expansion is 64 MiB

    @pytest.mark.parametrize(
        ("size", "over"), [(_FILE_LIMIT, False), (_FILE_LIMIT + 1, True)], ids=["the-limit", "one-byte-more"]
    )
    async def test_more_than_the_file_limit_is_a_guest_error(self, size: int, over: bool) -> None:
        """The guest opens no file over 500 MiB: a stream that decodes to more (a file that grew
        during the read, or a hostile guest) is GUEST_UNAVAILABLE (a retry reads the file again),
        raised before the extra byte reaches the destination."""
        destination = _WriteSizes()
        vm, _channel, _ = await _file_io_vm("test-vm", *_zeros_stream(size))

        if over:
            with pytest.raises(
                TransientError,
                check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE,
                match=r"^VM test-vm read_file communication failed: read_file 'file\.bin': the guest sent more than "
                r"524288000 bytes$",
            ):
                await vm.read_file("file.bin", destination=destination)
        else:
            await vm.read_file("file.bin", destination=destination)
        assert sum(destination.sizes) == _FILE_LIMIT

    async def test_path_destination_is_written_whole(self, tmp_path: Path) -> None:
        payload = b"payload of the guest file"
        destination = tmp_path / "out" / "file.bin"
        vm, channel, _ = await _file_io_vm("test-vm", *_guest_file(payload))
        await vm.read_file("file.bin", destination=destination)
        assert destination.read_bytes() == payload
        assert sorted(p.name for p in destination.parent.iterdir()) == ["file.bin"]  # no temp file left
        channel.unregister_op.assert_called_once()

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
        channel.unregister_op.assert_called_once()
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
        assert _frames(channel) == []  # no request sent
        channel.unregister_op.assert_called_once()
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
        ("make_error", "match", "diagnostics"),
        [
            pytest.param(TimeoutError, r"^VM test-vm read_file timed out for 'file\.bin'$", True, id="timeout"),
            pytest.param(
                lambda: ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"),
                r"^VM test-vm read_file communication failed: .*Connection reset by peer$",
                False,
                id="oserror",
            ),
            pytest.param(
                lambda: CommunicationError("channel closed", context=None),
                r"^VM test-vm read_file communication failed: channel closed$",
                False,
                id="communication",
            ),
        ],
    )
    @pytest.mark.parametrize("kind", ["path", "stream"])
    async def test_transport_failure_is_guest_unavailable(
        self, tmp_path: Path, make_error: Any, match: str, diagnostics: bool, kind: str
    ) -> None:
        """One transport mapping for both destination kinds: a timeout, a socket error
        or a channel error is the guest's (GUEST_UNAVAILABLE, RETRY), never LOCAL_DISK or a
        builtin. The temp file goes; the op is unregistered once. Only a deadline carries the
        guest console and the QEMU state (`diagnostics`)."""
        error = make_error()
        vm, channel, _ = await _file_io_vm("test-vm", *_guest_file(b"payload"))
        channel.connect.side_effect = error
        destination: Path | IO[bytes] = tmp_path / "file.bin" if kind == "path" else io.BytesIO()
        with pytest.raises(
            TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE, match=match
        ) as exc_info:
            await vm.read_file("file.bin", destination=destination)
        assert exc_info.value.__cause__ is error
        channel.unregister_op.assert_called_once()
        assert list(tmp_path.iterdir()) == []
        assert (exc_info.value.context.items() >= _DEADLINE_DIAGNOSTICS.items()) is diagnostics

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
        ("written", "reported"),
        [
            pytest.param(lambda _data: 0, "0", id="zero"),
            pytest.param(lambda _data: None, "None", id="none"),  # a raw non-blocking writer on would-block
            pytest.param(lambda data: len(data) + 1, "8", id="over-reporting"),
        ],
    )
    async def test_destination_write_without_progress_is_rejected(self, written: Any, reported: str) -> None:
        """A writer that reports no progress (0, or None from a raw non-blocking writer) or
        more bytes than it got (the 7 bytes of the file) cannot report success: LOCAL_DISK
        (an OSError) that names what it reported, not a TypeError or a short file."""

        class Writer:
            def write(self, data: bytes) -> int | None:
                return written(data)

        vm, _channel, _ = await _file_io_vm("test-vm", *_guest_file(b"payload"))
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.LOCAL_DISK,
            match=(
                r"^VM test-vm read_file could not write the local destination \(read_file destination made "
                rf"no write progress: returned {reported} for 7 remaining bytes\)$"
            ),
        ):
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


class _FakeGuest:
    """The guest agent behind QEMU's two port sockets, served by this process: a real
    DualPortChannel connects to it (the peer check sees this user). Each ping on the command
    port gets the next answer of `answers`, then "pong": "pong"; "silent" (no answer yet: a
    slow guest); "error" (an error event, not a pong); "reseed-error" (the io_error event of a
    failed reseed); "unreadable" (a frame this release cannot read); "drop" (the guest side
    closes both ports). `pings` holds, for each ping, the number of the command-port connection
    it came on (1 is the first); `entropy`, for each ping, whether it carried host entropy;
    `connections` counts the command-port connections; `eofs` counts the ports that the host
    closed."""

    def __init__(self, folder: Path, answers: list[str]) -> None:
        self.cmd_socket = str(folder / "cmd.sock")
        self.event_socket = str(folder / "event.sock")
        self.answers = answers
        self.pings: list[int] = []
        self.entropy: list[bool] = []
        self.connections = 0
        self.eofs = 0
        self._event_writers: list[asyncio.StreamWriter] = []  # one for each event-port connection, in order
        self._writers: list[asyncio.StreamWriter] = []
        self._servers: list[asyncio.Server] = []
        self._changed = asyncio.Condition()

    async def __aenter__(self) -> Self:
        self._servers = [
            await asyncio.start_unix_server(self._serve_cmd, path=self.cmd_socket),
            await asyncio.start_unix_server(self._serve_event, path=self.event_socket),
        ]
        return self

    async def __aexit__(self, *_exc: object) -> None:
        for server in self._servers:
            server.close()
        for _ in range(3):  # a connection that the server has not handed to its handler yet
            self.drop()
            await asyncio.sleep(0)
        for server in self._servers:
            await asyncio.wait_for(server.wait_closed(), timeout=5)

    def channel(self) -> DualPortChannel:
        return DualPortChannel(cmd_socket=self.cmd_socket, event_socket=self.event_socket, expected_uid=os.getuid())

    async def send(self, frame: dict[str, object], *, connection: int | None = None) -> None:
        """Send one event frame to the host on the event port of `connection` (1 is the
        first; default: the latest). Waits until the server has that connection."""
        count = connection or 1
        async with self._changed:
            await asyncio.wait_for(self._changed.wait_for(lambda: len(self._event_writers) >= count), timeout=5)
        writer = self._event_writers[-1 if connection is None else connection - 1]
        writer.write(json.dumps(frame).encode() + b"\n")
        await writer.drain()

    def drop(self) -> None:
        """The guest side closes every port connection (QEMU exits)."""
        for writer in self._writers:
            writer.close()

    async def host_closed(self, ports: int) -> None:
        """Wait until the host has closed `ports` port connections in all."""
        async with self._changed:
            await asyncio.wait_for(self._changed.wait_for(lambda: self.eofs >= ports), timeout=5)

    async def _eof(self) -> None:
        async with self._changed:
            self.eofs += 1
            self._changed.notify_all()

    async def _serve_event(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.append(writer)
        async with self._changed:
            self._event_writers.append(writer)
            self._changed.notify_all()
        with contextlib.suppress(ConnectionError):
            if await reader.read() == b"":
                await self._eof()
        writer.close()

    async def _serve_cmd(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.append(writer)
        self.connections += 1
        connection = self.connections
        with contextlib.suppress(ConnectionError):
            while line := await reader.readline():
                request = json.loads(line)
                if request["action"] != "ping":
                    continue
                self.pings.append(connection)
                self.entropy.append("host_entropy" in request)
                answer = self.answers.pop(0) if self.answers else "pong"
                if answer == "pong":
                    pong = {"type": "pong", "version": "1.0", "op_id": request["op_id"]}
                    await self.send(pong, connection=connection)
                elif answer in ("error", "reseed-error"):
                    error = {
                        "type": "error",
                        "error_type": "io_error" if answer == "reseed-error" else "unknown",
                        "message": "not a pong",
                        "op_id": request["op_id"],
                    }
                    await self.send(error, connection=connection)
                elif answer == "unreadable":  # a pong this release cannot read (version not a str)
                    await self.send({"type": "pong", "version": 1, "op_id": request["op_id"]}, connection=connection)
                elif answer == "drop":
                    self.drop()
                    return
            if not writer.is_closing():
                await self._eof()
        writer.close()


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
            pytest.param({"type": "file_list", "entries": []}, FileListMessage, id="file_list"),
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
        inbox = dispatcher.register_op(op_id, changes_state=True)
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
        inbox = dispatcher.register_op("op-A", changes_state=True)
        _feed(reader, stray)
        _feed(reader, {"type": "stdout", "chunk": "next", "op_id": "op-A"})
        reader.feed_eof()

        assert [(type(m), _marker(m)) for m in await _received(inbox)] == [(OutputChunkMessage, "next")]
        await dispatcher.stop()

    async def test_message_after_unregister_is_discarded(self) -> None:
        """After unregister_op, a late message for the op reaches no inbox: the old inbox
        raises CommunicationError (it never gets the late message), the other ops go on."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        old = dispatcher.register_op("op-A", changes_state=True)
        _feed(reader, {"type": "file_chunk", "op_id": "op-A", "data": _wire("data", "first")})
        first = await asyncio.wait_for(old.get(), timeout=1.0)
        assert isinstance(first, FileChunkResponseMessage)
        assert first.data == b"first"

        dispatcher.unregister_op("op-A")
        other = dispatcher.register_op("op-B", changes_state=True)
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
        inboxes = {op: dispatcher.register_op(op, changes_state=True) for op in ops}
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
        inbox = dispatcher.register_op("op-A", changes_state=True)
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
        inbox = dispatcher.register_op("op-A", changes_state=True)
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
        op_queue = dispatcher.register_op("op-A", changes_state=True)
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

        op_queue = dispatcher.register_op("op-full", changes_state=True)
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
        first = dispatcher.register_op("first-op", changes_state=True)
        reader.feed_eof()
        with pytest.raises(CommunicationError, match="EOF"):  # the loop ended
            await asyncio.wait_for(first.get(), timeout=1.0)

        op_queue = dispatcher.register_op("late-op", changes_state=True)
        with pytest.raises(CommunicationError, match="EOF"):
            op_queue.ensure_open()

        await dispatcher.stop()

    async def test_full_queue_preserves_buffered_completion_before_transport_failure(self) -> None:
        """EOF cannot turn a complete valid stream into truncated success."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()

        op_queue = dispatcher.register_op("op-complete", changes_state=True)
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
        op_queue = dispatcher.register_op("overflow", changes_state=True)

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
        abandoned = dispatcher.register_op("abandoned", changes_state=True)

        for index in range(OP_QUEUE_DEPTH + 1):
            _feed(reader, {"type": "stdout", "chunk": str(index), "op_id": "abandoned"})
        await asyncio.sleep(0)  # dispatcher fills 64 slots, then blocks routing the 65th
        assert abandoned._queue.full()
        dispatcher.unregister_op("abandoned")

        later = dispatcher.register_op("later", changes_state=True)
        _feed(reader, {"type": "complete", "exit_code": 0, "execution_time_ms": 1, "op_id": "later"})
        message = await asyncio.wait_for(later.get(), timeout=1.0)
        assert isinstance(message, ExecutionCompleteMessage)

        reader.feed_eof()
        with pytest.raises(CommunicationError, match="EOF"):
            await asyncio.wait_for(later.get(), timeout=1.0)
        await dispatcher.stop()

    async def test_qemu_exit_drains_reader_buffered_terminal_before_failure(self, short_tmp_dir: Path) -> None:
        """Process death cannot overtake a terminal frame accepted by the reader: the guest
        sends the frame, then QEMU's sockets close."""
        async with _FakeGuest(short_tmp_dir, []) as guest:
            channel = guest.channel()
            await channel.connect(timeout_seconds=5.0)
            inbox = channel.register_op("terminal", changes_state=True)
            inbox.mark_command_sent()

            await guest.send({"type": "complete", "exit_code": 137, "execution_time_ms": 321, "op_id": "terminal"})
            guest.drop()
            await channel.fail_pending_operations("QEMU exited with code 0")

            message = await asyncio.wait_for(inbox.get(), timeout=1.0)
            assert isinstance(message, ExecutionCompleteMessage)
            assert message.exit_code == 137
            with pytest.raises(
                OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN, match="EOF"
            ):
                await inbox.get()
            await channel.close()

    async def test_qemu_exit_with_full_queue_preserves_late_terminal(self, short_tmp_dir: Path) -> None:
        """Backpressure plus peer exit cannot overtake a buffered terminal: the guest sends
        more messages than the inbox holds, then the terminal, then QEMU's sockets close."""
        async with _FakeGuest(short_tmp_dir, []) as guest:
            channel = guest.channel()
            await channel.connect(timeout_seconds=5.0)
            inbox = channel.register_op("terminal-after-overflow", changes_state=True)
            inbox.mark_command_sent()

            for index in range(OP_QUEUE_DEPTH + 1):
                await guest.send({"type": "stdout", "chunk": str(index), "op_id": "terminal-after-overflow"})
            await guest.send(
                {"type": "complete", "exit_code": 137, "execution_time_ms": 321, "op_id": "terminal-after-overflow"}
            )
            guest.drop()

            failure = asyncio.create_task(channel.fail_pending_operations("QEMU exited with code 0"))
            messages = [await asyncio.wait_for(inbox.get(), timeout=1.0) for _ in range(OP_QUEUE_DEPTH + 2)]
            await asyncio.wait_for(failure, timeout=1.0)

            assert [message.chunk for message in messages[:-1] if isinstance(message, OutputChunkMessage)] == [
                str(index) for index in range(OP_QUEUE_DEPTH + 1)
            ]
            assert isinstance(messages[-1], ExecutionCompleteMessage)
            assert messages[-1].exit_code == 137
            with pytest.raises(
                OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN, match="EOF"
            ):
                await inbox.get()
            await channel.close()

    @pytest.mark.parametrize(
        ("changes_state", "error"),
        [
            pytest.param(True, (OutcomeUnknownError, ErrorCode.GUEST_OUTCOME_UNKNOWN), id="exec-or-write"),
            pytest.param(False, (InternalError, ErrorCode.GUEST_PROTOCOL), id="read-only"),
        ],
    )
    async def test_an_unreadable_reply_fails_its_operation_at_once(
        self, caplog: pytest.LogCaptureFixture, changes_state: bool, error: tuple[type[Exception], ErrorCode]
    ) -> None:
        """A reply this release cannot read fails its operation at once, not after its
        deadline: an unknown outcome for a command that may have run user code or changed a
        file, GUEST_PROTOCOL for a read-only one. The log names field locations and types,
        never the frame's text (a frame carries the user's output); the first such frame is
        a WARNING, the next ones DEBUG. The frames before it are delivered first."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        inbox = dispatcher.register_op("op-A", changes_state=changes_state)
        later = dispatcher.register_op("op-B", changes_state=False)  # its frame comes last
        inbox.mark_command_sent()
        _feed(reader, {"type": "stdout", "chunk": "before", "op_id": "op-A"})
        for _ in range(2):
            _feed(reader, {"type": "stdout", "chunk": 42, "op_id": "op-A", "secret": "sk-user-output"})
        _feed(reader, {"type": "stdout", "chunk": "after", "op_id": "op-B"})
        with caplog.at_level(logging.DEBUG, logger="exec_sandbox.guest_channel"):
            assert isinstance(await asyncio.wait_for(inbox.get(), timeout=1.0), OutputChunkMessage)
            with pytest.raises(error[0], match="cannot read") as exc_info:
                await asyncio.wait_for(inbox.get(), timeout=1.0)
            await asyncio.wait_for(later.get(), timeout=1.0)  # both unreadable frames were read
        await dispatcher.stop()
        assert dispatcher.unreadable_frames == 2

        assert getattr(exc_info.value, "code", None) is error[1]
        records = [
            r for r in caplog.records if r.name == "exec_sandbox.guest_channel" and "cannot read" in r.getMessage()
        ]
        assert [r.levelno for r in records] == [logging.WARNING, logging.DEBUG]
        assert all("sk-user-output" not in r.getMessage() and "42" not in r.getMessage() for r in records)
        assert "sk-user-output" not in caplog.text

    async def test_transport_loss_after_dispatch_is_outcome_unknown(self) -> None:
        """Host transport loss is never fabricated as a retryable guest error."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)
        dispatcher.start()
        op_queue = dispatcher.register_op("sent", changes_state=True)
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
        inbox = dispatcher.register_op("op-A", changes_state=True)

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

    async def test_stop_in_a_cancel_handler_does_not_cancel_again(self) -> None:
        """A cleanup in a cancel handler (destroy_vm in a pool's handler, ephemeral_vm's
        finally) stops the dispatcher while its task's cancel count is already 1: it ends,
        with no new cancel, and the cleanup goes on (TestDualPortChannelCloseResiliency:
        the same for close() of a channel)."""
        dispatcher = FileOpDispatcher(asyncio.StreamReader())
        dispatcher.start()
        steps: list[str] = []

        async def cancelled_then_clean_up() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await dispatcher.stop()
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
        op_queue = dispatcher.register_op("oversized", changes_state=True)
        reader.feed_data(b"x" * 64 + b"\n")

        with pytest.raises(CommunicationError, match="reader limit"):
            await asyncio.wait_for(op_queue.get(), timeout=1.0)
        assert not dispatcher.is_alive()
        await dispatcher.stop()


# ============================================================================
# DualPortChannel.close() resiliency
# ============================================================================


class TestDualPortChannelCloseResiliency:
    """close() stops the dispatcher, closes both ports, and always drops the dispatcher,
    also when its stop raises: a later call must not use a dispatcher that failed (a
    poisoned dispatcher would block each health check retry). A real channel on a fake
    guest (_FakeGuest): the guest sees each port that the host closes."""

    async def test_close_fails_ops_in_flight_and_closes_both_ports(self, short_tmp_dir: Path) -> None:
        async with _FakeGuest(short_tmp_dir, []) as guest:
            ch = guest.channel()
            await ch.connect(timeout_seconds=5.0)
            inbox = ch.register_op("op-A", changes_state=True)

            await ch.close()

            with pytest.raises(CommunicationError, match=r"^event dispatcher stopped$"):
                await asyncio.wait_for(inbox.get(), timeout=1.0)
            await guest.host_closed(2)
            with pytest.raises(RuntimeError, match="channel not connected"):
                ch.register_op("op-B", changes_state=True)
        assert guest.eofs == 2

    async def test_close_before_connect_is_a_no_op(self) -> None:
        """close() of a channel that never connected raises nothing; the channel has no
        dispatcher after it."""
        ch = DualPortChannel(cmd_socket="/nonexistent/cmd.sock", event_socket="/nonexistent/event.sock", expected_uid=0)

        await ch.close()

        with pytest.raises(RuntimeError, match="channel not connected"):
            ch.register_op("op-A", changes_state=True)

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("dispatch task exploded"), KeyboardInterrupt(), asyncio.CancelledError()],
        ids=["RuntimeError", "KeyboardInterrupt", "caller-cancel"],
    )
    async def test_close_drops_dispatcher_when_stop_raises(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        """The dispatcher's stop raises after it stopped (a stand-in): the error propagates,
        both ports are closed all the same, and the channel has no dispatcher after it."""
        real_stop = FileOpDispatcher.stop

        async def failing_stop(dispatcher: FileOpDispatcher) -> None:
            await real_stop(dispatcher)
            raise error

        async with _FakeGuest(short_tmp_dir, []) as guest:
            ch = guest.channel()
            await ch.connect(timeout_seconds=5.0)
            monkeypatch.setattr(FileOpDispatcher, "stop", failing_stop)

            with pytest.raises(type(error)):
                await ch.close()

            await guest.host_closed(2)
            with pytest.raises(RuntimeError, match="channel not connected"):
                ch.register_op("op-A", changes_state=True)
        assert guest.eofs == 2

    async def test_close_idempotent(self, short_tmp_dir: Path) -> None:
        """A second close() raises nothing and opens or closes nothing more: the guest saw one
        connection and each of its two ports closed once; the channel has no dispatcher."""
        async with _FakeGuest(short_tmp_dir, []) as guest:
            ch = guest.channel()
            await ch.connect(timeout_seconds=5.0)

            await ch.close()
            await ch.close()

            await guest.host_closed(2)
            with pytest.raises(RuntimeError, match="channel not connected"):
                ch.register_op("op-A", changes_state=True)
        assert (guest.connections, guest.eofs) == (1, 2)

    async def test_close_in_a_cancel_handler_does_not_cancel_again(self, short_tmp_dir: Path) -> None:
        """A cleanup in a cancel handler (destroy_vm in a pool's handler, ephemeral_vm's
        finally) closes the channel while its task's cancel count is already 1: the close
        ends, with no new cancel, the ports are closed, and the cleanup goes on."""
        steps: list[str] = []
        async with _FakeGuest(short_tmp_dir, []) as guest:
            channel = guest.channel()
            await channel.connect(timeout_seconds=5.0)

            async def cancelled_then_clean_up() -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await channel.close()
                    steps.append("after the close")  # destroy_vm: kill QEMU
                    raise

            task = asyncio.create_task(cancelled_then_clean_up())
            await asyncio.sleep(0)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task
            await guest.host_closed(2)
        assert steps == ["after the close"]


# ============================================================================
# Properties — exposed_ports (from the VM), startup_timing and spec
# ============================================================================


class TestProperties:
    """Session properties: exposed_ports reads the VM's; startup_timing and spec are the
    values the session was made with."""

    async def test_startup_timing_and_spec_exposed(self) -> None:
        session, _, _ = _make_session()
        assert session.startup_timing is _STARTUP
        assert session.spec == VmSpec(language=Language.PYTHON)

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
# comes. A real channel on a fake guest (_FakeGuest): the tests check what
# connect() returns or raises, the pings the guest gets, and on which connection.
# ============================================================================


def _probe_budget(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    """Six probe attempts, as in production: 50 ms for a ping that the guest leaves with
    no answer, 5 s for the others (an answer then never misses its wait on a loaded host)."""
    slots = (answers + ["pong"] * 6)[:6]
    timeouts = tuple(0.05 if answer == "silent" else 5.0 for answer in slots)
    monkeypatch.setattr(constants, "GUEST_RECONNECT_PROBE_TIMEOUTS", timeouts)


async def _reconnect(guest: _FakeGuest) -> DualPortChannel:
    """A channel of `guest`, connected once and closed: its next connect() probes the guest."""
    channel = guest.channel()
    await channel.connect(timeout_seconds=5.0)
    await channel.close()
    return channel


class TestDualPortStreamDeadline:
    """The total deadline of stream_messages, on a real channel."""

    async def test_an_exec_the_guest_never_answers_ends_at_the_deadline(self, short_tmp_dir: Path) -> None:
        """The guest reads the exec and never answers: stream_messages raises
        StreamDeadlineExceededError at its total deadline, on a real channel. (QemuVM maps it
        to an unknown outcome: test_vm_manager test_true_stream_deadline_is_outcome_unknown_...)"""
        async with _FakeGuest(short_tmp_dir, []) as guest:
            channel = guest.channel()
            await channel.connect(timeout_seconds=5.0)
            started = time.monotonic()
            with pytest.raises(StreamDeadlineExceededError, match=r"^stream exceeded 1s total deadline$"):
                async for _message in channel.stream_messages(
                    ExecuteCodeRequest(language=Language.PYTHON, code="x", reseed=False, timeout=0, env_vars={}),
                    timeout=1,
                ):
                    pass
            elapsed = time.monotonic() - started
            await channel.close()
        assert 1.0 <= elapsed < 3.0


class TestDualPortConnectLifecycle:
    """When connect() probes the guest (a reconnection), and when it does not."""

    async def test_first_connect_does_not_probe(self, short_tmp_dir: Path) -> None:
        """Boot-time connect sends no ping."""
        async with _FakeGuest(short_tmp_dir, []) as guest:
            channel = guest.channel()
            await channel.connect(timeout_seconds=5.0)
            await channel.close()
        assert (guest.connections, guest.pings) == (1, [])

    async def test_idempotent_when_already_connected(self, short_tmp_dir: Path) -> None:
        """connect() on a connected channel opens nothing and sends no ping."""
        async with _FakeGuest(short_tmp_dir, []) as guest:
            channel = guest.channel()
            await channel.connect(timeout_seconds=5.0)
            await channel.connect(timeout_seconds=5.0)
            await channel.close()
        assert (guest.connections, guest.pings) == (1, [])

    async def test_first_connect_after_a_restore_probes(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A channel of a guest restored from a save (mark_restored, as restore_vm does): its
        first connect sends one ping, as a reconnection does, but with host entropy, and gets
        the pong; a later reconnection pings without host entropy. That restore_vm makes the
        call is checked on a VM: without the probe, the restore of a zeroed RAM template hangs
        (test_l1_damaged_entry, test_zeroed_ram_template_...)."""
        _probe_budget(monkeypatch, [])
        async with _FakeGuest(short_tmp_dir, []) as guest:
            channel = guest.channel()
            channel.mark_restored()
            await channel.connect(timeout_seconds=5.0)
            await channel.close()
            await channel.connect(timeout_seconds=5.0)  # a reconnection, not a restore
            await channel.close()
        assert (guest.connections, guest.pings, guest.entropy) == (2, [1, 2], [True, False])

    async def test_a_reseed_error_after_a_restore_is_guest_silent_after_restore(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The restored guest answers its first ping with the io_error of a failed reseed:
        TransientError GUEST_SILENT_AFTER_RESTORE at once, no second ping. Another error type
        is no reseed failure: InternalError GUEST_PROTOCOL, as without a restore."""
        _probe_budget(monkeypatch, [])  # each connect sends one ping, which waits 5 s for its answer
        async with _FakeGuest(short_tmp_dir, ["reseed-error", "error"]) as guest:
            channel = guest.channel()
            channel.mark_restored()
            with pytest.raises(
                TransientError, match="could not reseed", check=lambda e: e.code is ErrorCode.GUEST_SILENT_AFTER_RESTORE
            ) as exc:
                await channel.connect(timeout_seconds=5.0)
            assert exc.value.context == {"guest_error_type": "io_error"}
            await channel.close()
            with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL):
                await channel.connect(timeout_seconds=5.0)
            await channel.close()
        assert (guest.pings, guest.entropy) == ([1, 2], [True, True])  # one ping on each connection

    async def test_connect_after_close_probes(self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """close() does not make the channel new: the connect after it sends one ping on the
        new connection and gets the pong."""
        _probe_budget(monkeypatch, [])
        async with _FakeGuest(short_tmp_dir, []) as guest:
            channel = await _reconnect(guest)
            await channel.connect(timeout_seconds=5.0)
            await channel.close()
        assert (guest.connections, guest.pings) == (2, [2])


class TestDualPortProbeSuccess:
    """connect() returns when the guest answers a ping in its budget."""

    @pytest.mark.parametrize(
        ("answers", "pings", "connections"),
        [
            pytest.param(["silent", "drop", "silent"], [2, 2, 3, 3], 3, id="mixed-failures"),
            pytest.param(["silent"] * 5, [2] * 6, 2, id="last-possible-attempt"),
        ],
    )
    async def test_guest_ready_after_failed_attempts(
        self,
        short_tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        answers: list[str],
        pings: list[int],
        connections: int,
    ) -> None:
        """The guest answers after pings with no answer or a dropped connection: connect()
        returns at the first pong, also on the sixth and last attempt."""
        _probe_budget(monkeypatch, answers)
        async with _FakeGuest(short_tmp_dir, answers) as guest:
            channel = await _reconnect(guest)
            await channel.connect(timeout_seconds=5.0)
            await channel.close()
        assert (guest.pings, guest.connections) == (pings, connections)

    async def test_unanswered_first_ping_costs_a_short_wait(self, short_tmp_dir: Path) -> None:
        """The first ping gets no answer in time: the next ping on the same connection
        ends the probe in ~0.1 s (the production first wait is 50 ms)."""
        async with _FakeGuest(short_tmp_dir, ["silent"]) as guest:
            channel = await _reconnect(guest)
            loop = asyncio.get_running_loop()
            start = loop.time()
            await channel.connect(timeout_seconds=5.0)
            elapsed = loop.time() - start
            await channel.close()

        assert elapsed < 0.3
        assert guest.pings == [2, 2]

    async def test_slow_pong_never_resets_the_connection(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ping with no answer yet is slow, not lost: the probe sends the next ping on
        the same connection. Closing it would show the guest a host disconnect for each
        slow pong."""
        answers = ["silent"] * 3
        _probe_budget(monkeypatch, answers)
        async with _FakeGuest(short_tmp_dir, answers) as guest:
            channel = await _reconnect(guest)
            await channel.connect(timeout_seconds=5.0)
            await channel.close()
        assert (guest.pings, guest.connections) == ([2] * 4, 2)

    async def test_failed_connection_is_opened_again(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only a connection that failed is closed and opened again before the next ping."""
        _probe_budget(monkeypatch, ["drop"])
        async with _FakeGuest(short_tmp_dir, ["drop"]) as guest:
            channel = await _reconnect(guest)
            await channel.connect(timeout_seconds=5.0)
            await channel.close()
        assert (guest.pings, guest.connections) == ([2, 3], 3)


class TestDualPortProbeFailure:
    """connect() raises TimeoutError when the guest never answers in the probe budget."""

    @pytest.mark.parametrize(
        ("answers", "pings", "connections"),
        [
            # Every ping goes on the same connection.
            pytest.param(["silent"] * 6, [2] * 6, 2, id="no-answer"),
            # One reconnect between attempts, none after the last: it would only delay the error.
            pytest.param(["drop"] * 6, [2, 3, 4, 5, 6, 7], 7, id="connection-drops"),
        ],
    )
    async def test_no_pong_in_six_attempts(
        self,
        short_tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        answers: list[str],
        pings: list[int],
        connections: int,
    ) -> None:
        _probe_budget(monkeypatch, answers)
        async with _FakeGuest(short_tmp_dir, answers) as guest:
            channel = await _reconnect(guest)
            with pytest.raises(TimeoutError, match=r"^Guest agent did not respond to ping after 6 attempts$"):
                await channel.connect(timeout_seconds=5.0)
            await channel.close()
        assert (guest.pings, guest.connections) == (pings, connections)

    async def test_an_unreadable_answer_ends_the_probe(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reply to a ping that this release cannot read: GUEST_PROTOCOL at once (a ping runs
        nothing), no further ping and no reconnect."""
        _probe_budget(monkeypatch, ["unreadable"])
        async with _FakeGuest(short_tmp_dir, ["unreadable"]) as guest:
            channel = await _reconnect(guest)
            with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL, match="cannot read"):
                await channel.connect(timeout_seconds=5.0)
            await channel.close()
        assert (guest.pings, guest.connections) == ([2], 2)

    @pytest.mark.parametrize(
        ("answers", "pings"),
        [
            pytest.param(["error"], [2], id="at-once"),
            pytest.param(["silent"] * 5 + ["error"], [2] * 6, id="last"),
            pytest.param(["reseed-error"], [2], id="reseed-error-with-no-restore"),
        ],
    )
    async def test_another_answer_ends_the_probe(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str], pings: list[int]
    ) -> None:
        """The guest answers a ping with another message (an error, also the io_error of a
        reseed when no restore came first): GUEST_PROTOCOL at once, no further ping (it would
        get the same answer)."""
        _probe_budget(monkeypatch, answers)
        async with _FakeGuest(short_tmp_dir, answers) as guest:
            channel = await _reconnect(guest)
            with pytest.raises(
                InternalError,
                check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL,
                match=r'^The guest agent answered a ping with a message of type "error", not a pong\.$',
            ):
                await channel.connect(timeout_seconds=5.0)
            await channel.close()
        assert guest.pings == pings


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
        """async with Session closes it at the block's end (__aexit__): its VM is destroyed and
        no deadline stays armed."""
        session, _, manager = _make_session()
        async with session:
            await session.exec("x = 1")
            assert not session.closed
        assert session.closed
        manager.destroy_vm.assert_awaited_once()
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

    async def test_idle_timer_suspended_during_operation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operation running longer than the idle timeout is not destroyed.

        Idle means "no operation in flight": the timer is cancelled while an operation
        runs and re-armed when it ends (TestGuardIdleTimerReset: it then fires). A fake
        loop clock moves the time.
        """
        clock = LoopClock(monkeypatch)
        session, vm, manager = _make_session(idle_timeout_seconds=10)
        closed_during: list[bool] = []

        async def long_execute(**_kwargs: object) -> ExecutionResult:
            await clock.advance(15)  # the call runs past the idle deadline
            closed_during.append(session.closed)
            return _make_exec_result()

        vm.execute.side_effect = long_execute
        result = await session.exec("print(1)")
        await clock.advance(9)  # the deadline counts from the end of the call

        assert result.exit_code == 0
        assert closed_during == [False]
        assert not session.closed
        manager.destroy_vm.assert_not_awaited()
        await session.close()

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


class TestGuardCloseRule:
    """A file call whose guest outcome is unknown (OutcomeUnknownError GUEST_OUTCOME_UNKNOWN)
    closes the session: the request may have reached the guest. (A save's store answers keep
    it open: test_close_rule, through save().)"""

    @pytest.mark.parametrize("call", ["list_files", "read_file", "write_file"])
    async def test_an_unknown_guest_outcome_of_a_file_call_closes_the_session(self, call: str) -> None:
        session, mock_vm, _ = _make_session()
        error = OutcomeUnknownError(
            "injected", code=ErrorCode.GUEST_OUTCOME_UNKNOWN, context={}, hint=None, instance_id=None
        )
        setattr(mock_vm, call, AsyncMock(side_effect=error))  # exec() closes on its own errors: not this rule
        with pytest.raises(OutcomeUnknownError, check=lambda e: e is error):
            match call:
                case "list_files":
                    await session.list_files()
                case "read_file":
                    await session.read_file("f.txt", destination=io.BytesIO())
                case _:
                    await session.write_file("f.txt", b"data")
        assert session.closed
        await session.close()

"""Host-deadline diagnostics: the guest console and the QEMU state in the error context.

With only ids and timeouts, a host deadline error (no guest answer in time) cannot
tell a stalled guest from a slow one. QemuVM.deadline_diagnostics adds the console
ring buffer, the QMP run state and the QEMU CPU time and threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from unittest.mock import AsyncMock, MagicMock

import psutil
import pytest

from exec_sandbox import constants
from exec_sandbox.exceptions import ErrorCode, OutcomeUnknownError, TransientError
from exec_sandbox.guest_channel import StreamDeadlineExceededError
from exec_sandbox.models import Language
from exec_sandbox.permission_utils import get_expected_socket_uid
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.qmp_client import QMP_BUFFER_LIMIT, QmpClient
from exec_sandbox.socket_auth import connect_and_verify
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType, VmState

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from exec_sandbox.vm_manager import VmManager


async def _fake_qmp(path: Path, *, status: str | None) -> asyncio.Server:
    """A QMP server: greeting, then a reply per command. status=None never greets."""

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if status is None:
            await reader.read()  # silent until the client hangs up
            writer.close()
            return
        writer.write(b'{"QMP": {"version": {}, "capabilities": []}}\n')
        while line := await reader.readline():
            request = json.loads(line)
            result: dict[str, Any] = {}
            if request["execute"] == "query-status":
                result = {"status": status, "running": status == "running"}
            writer.write(json.dumps({"return": result, "id": request["id"]}).encode() + b"\n")
            await writer.drain()
        writer.close()

    return await asyncio.start_unix_server(serve, path=str(path))


@pytest.fixture
def qmp_path(short_tmp_dir: Path) -> Path:
    return short_tmp_dir / "qmp.sock"


class _DeadlineChannel:
    """Connected channel whose answers never come in time."""

    async def connect(self, timeout_seconds: float) -> None:
        pass

    async def send_request(self, request: object, timeout: float) -> object:
        raise TimeoutError

    async def stream_messages(self, request: object, timeout: int) -> AsyncGenerator[object]:
        raise StreamDeadlineExceededError("total stream deadline")
        yield  # an async generator


class _MakeVm(Protocol):
    def __call__(self, *, qemu: psutil.Process | None, console: list[str], channel: object = None) -> QemuVM: ...


@pytest.fixture
async def make_vm(qmp_path: Path) -> AsyncGenerator[_MakeVm]:
    """Makes real QemuVMs in state READY whose QEMU still runs: the QMP control socket at
    `qmp_path`, `qemu` as the process psutil sees (None: psutil never saw it), `console` in the
    console ring buffer. The state is set directly: only a boot makes a VM READY.

    At teardown QEMU exits and each VM is confirmed destroyed: the QEMU exit task that a VM
    starts at its first execute() ends with the test, as destroy_vm ends it."""
    exited = asyncio.Event()
    vms: list[QemuVM] = []

    async def wait() -> int:
        await exited.wait()
        return 0

    def make(*, qemu: psutil.Process | None, console: list[str], channel: object = None) -> QemuVM:
        process = AsyncMock()
        process.returncode = None
        process.psutil_proc = qemu
        process.wait = AsyncMock(side_effect=wait)
        vm = QemuVM(
            vm_id="deadline-vm",
            process=process,
            cgroup=None,
            workdir=MagicMock(qmp_socket=qmp_path, use_qemu_vm_user=False),
            channel=channel or MagicMock(),  # type: ignore[arg-type]
            spec=VmSpec(language=Language.PYTHON),
            console_lines=deque(console, maxlen=10),
            base_image=Path("/test/base.qcow2"),
            packages_image=None,
            accel_type=AccelType.KVM,
            gvproxy_proc=None,
            qemu_log_task=None,
            gvproxy_log_task=None,
            device_layout="",
        )
        vm._state = VmState.READY  # pyright: ignore[reportPrivateUsage]
        vms.append(vm)
        return vm

    yield make
    exited.set()
    for vm in vms:
        await vm.confirm_destroyed()


class TestDeadlineDiagnostics:
    """QemuVM.deadline_diagnostics: the values for the error context, and one ERROR log."""

    async def test_reads_the_qemu_state_and_logs_it(
        self, make_vm: _MakeVm, qmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """This test process stands in for QEMU: its CPU time and threads are known here."""
        own = psutil.Process()
        cpu_before = round(sum(own.cpu_times()[:2]), 3)  # user + system
        console = ["boot", "[WARN] Read timeout after 18000ms, reconnecting..."]
        vm = make_vm(qemu=own, console=console)

        async with await _fake_qmp(qmp_path, status="paused"):
            with caplog.at_level(logging.ERROR, logger="exec_sandbox.qemu_vm"):
                values = await vm.deadline_diagnostics(operation="write_file")

        assert values["qemu_status"] == "paused"
        assert values["qemu_cpu_seconds"] >= cpu_before  # this process (and its children) since its start
        assert values["qemu_threads"] >= threading.active_count()
        assert {key: values[key] for key in ("vm_id", "exit_code", "signal_name", "console_log")} == {
            "vm_id": "deadline-vm",
            "exit_code": None,
            "signal_name": "",
            "console_log": "boot\n[WARN] Read timeout after 18000ms, reconnecting...",
        }
        cpu, threads = values["qemu_cpu_seconds"], values["qemu_threads"]
        assert f"operation=write_file qemu_status=paused qemu_cpu_seconds={cpu} qemu_threads={threads}" in caplog.text
        assert "[WARN] Read timeout" in caplog.text

    async def test_no_qemu_is_unavailable(self, make_vm: _MakeVm) -> None:
        """No QMP socket, and no process that psutil saw: each value says so, nothing raises."""
        vm = make_vm(qemu=None, console=[])

        values = await vm.deadline_diagnostics(operation="read_file")

        assert values["qemu_status"].startswith("unavailable (FileNotFoundError")
        assert (values["qemu_cpu_seconds"], values["qemu_threads"], values["console_log"]) == (None, None, "(empty)")

    async def test_silent_qemu_is_bounded(
        self, make_vm: _MakeVm, qmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A QMP monitor that never answers (busy, or held by another client) costs the bound."""
        monkeypatch.setattr(constants, "DEADLINE_DIAGNOSTICS_TIMEOUT_SECONDS", 0.05)
        vm = make_vm(qemu=None, console=[])

        async with await _fake_qmp(qmp_path, status=None), asyncio.timeout(1):
            values = await vm.deadline_diagnostics(operation="execute")

        assert values["qemu_status"].startswith("unavailable (TimeoutError")


async def test_execute_deadline_carries_diagnostics(make_vm: _MakeVm, caplog: pytest.LogCaptureFixture) -> None:
    vm = make_vm(qemu=None, console=["[repl-timeout] last line"], channel=_DeadlineChannel())
    with caplog.at_level(logging.ERROR, logger="exec_sandbox.qemu_vm"), pytest.raises(OutcomeUnknownError) as exc_info:
        await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    context = exc_info.value.context  # type: ignore[attr-defined]
    assert context["console_log"] == "[repl-timeout] last line"
    assert context["qemu_status"].startswith("unavailable (FileNotFoundError")  # no QMP socket
    assert (context["qemu_cpu_seconds"], context["qemu_threads"]) == (None, None)
    assert "operation=execute qemu_status=unavailable" in caplog.text


async def test_list_files_deadline_carries_diagnostics(make_vm: _MakeVm, caplog: pytest.LogCaptureFixture) -> None:
    vm = make_vm(
        qemu=None,
        console=["[WARN] last line"],
        channel=_DeadlineChannel(),
    )
    with (
        caplog.at_level(logging.ERROR, logger="exec_sandbox.qemu_vm"),
        pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info,
    ):
        await vm.list_files("")

    context = exc_info.value.context
    assert context["console_log"] == "[WARN] last line"
    assert context["qemu_status"].startswith("unavailable (FileNotFoundError")  # no QMP socket
    assert "operation=list_files qemu_status=unavailable" in caplog.text


def _qemu_pid(vm: QemuVM) -> int:
    """QEMU's pid: the spawned process, or its descendant (through sudo when QEMU runs as qemu-vm)."""
    spawned = psutil.Process(vm.process.pid)
    for process in [spawned, *spawned.children(recursive=True)]:
        with contextlib.suppress(psutil.Error):
            if process.name().startswith("qemu-system"):
                return process.pid
    return spawned.pid


async def _qemu_stacks(vm: QemuVM) -> str:
    """Where each QEMU thread waits: `sample` on macOS; on Linux each thread's state, wchan
    and kernel stack (the stack through `sudo -n`; without sudo, state and wchan of a QEMU
    of this user)."""
    pid = _qemu_pid(vm)
    if sys.platform == "darwin":
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "sample.txt"
            proc = await asyncio.create_subprocess_exec(
                "/usr/bin/sample", str(pid), "1", "-mayDie", "-file", str(report),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.STDOUT,
            )  # fmt: skip
            await proc.wait()
            lines = report.read_text(errors="replace").splitlines() if report.exists() else ["(no sample report)"]
        start = next((i for i, line in enumerate(lines) if line.startswith("Call graph:")), 0)
        return "\n".join(lines[start : start + 300])
    script = (
        f"for t in /proc/{pid}/task/*; do "
        'echo "== ${t##*/} $(cat $t/comm) $(sed \'s/.*) //\' $t/stat | cut -d" " -f1) $(cat $t/wchan)"; '
        "cat $t/stack 2>/dev/null; done; exit 0"  # a thread that ended is no failure: only a refused sudo is
    )
    # /proc/<pid>/stack needs root even for this user's own process: sudo -n first, plain sh
    # (state and wchan only, for a QEMU of this user) when sudo is refused
    for command in (["sudo", "-n", "sh", "-c", script], ["sh", "-c", script]):
        proc = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        out, _ = await proc.communicate()
        if proc.returncode == 0:
            break
    return out.decode(errors="replace")


def _unread(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bytes:
    """The bytes QEMU sent that the exchange has not used: those the transport moved into the
    reader's buffer (after a stall of this loop, the reads run before the timeout fires),
    then those still in the socket (MSG_PEEK on a dup)."""
    buffered = bytes(reader._buffer)  # pyright: ignore[reportPrivateUsage]
    with socket.socket(fileno=os.dup(writer.get_extra_info("socket").fileno())) as peer:
        try:
            return buffered + peer.recv(4096, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        except BlockingIOError:
            return buffered


async def _qmp(vm: QemuVM, command: str) -> dict[str, Any]:
    """One QMP command on the control monitor, each step bounded by 5 s. A timeout says how
    long its step took (far above 5 s: this test's loop stalled), what QEMU sent that is not
    read yet, and where QEMU's threads wait (once, QEMU answered neither `stop` in
    5 s nor SIGTERM in 3 s)."""
    reader, writer = await connect_and_verify(
        path=str(vm.qmp_socket),
        expected_uid=get_expected_socket_uid(vm.use_qemu_vm_user),
        timeout=5,
        buffer_limit=QMP_BUFFER_LIMIT,
    )
    qmp = QmpClient(reader, writer, error_factory=OSError, id_prefix="test")
    step, started = "start (greeting, qmp_capabilities)", time.monotonic()
    try:
        await qmp.start(timeout=5)
        step, started = command, time.monotonic()
        return await qmp.execute(command, arguments=None, timeout=5, on_event=None)
    except TimeoutError as e:
        e.add_note(
            f"QMP {step}: no answer after {time.monotonic() - started:.2f} s; "
            f"unread: {_unread(reader, writer)!r}\nQEMU threads at the timeout:\n{await _qemu_stacks(vm)}"
        )
        raise
    finally:
        await qmp.close()


async def test_unread_reports_the_reply_the_loop_already_buffered() -> None:
    """After a stall of this loop the transport reads QEMU's reply into the reader before the
    timeout fires: the note shows it, then the bytes still in the socket."""
    ours, qemu = socket.socketpair()
    reader, writer = await asyncio.open_connection(sock=ours)
    try:
        qemu.sendall(b'{"return": {}, "id": "stop"}\n')
        for _ in range(100):  # the transport reads it into the reader's buffer
            await asyncio.sleep(0.01)
            if _unread(reader, writer):
                break
        assert _unread(reader, writer) == b'{"return": {}, "id": "stop"}\n'
    finally:
        writer.close()
        qemu.close()


async def test_deadline_on_a_paused_guest_carries_the_qemu_state(
    vm_manager: VmManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A guest that does not run cannot ack a write: the host deadline error says why
    (QMP run state "paused"), with the console and the QEMU CPU time."""
    vm = await vm_manager.create_vm(
        VmSpec(language=Language.RAW),
        tenant_id="test",
        task_id="deadline-diag",
        cache_leases=(),
        direct_write_target=None,
        expose_ports=None,
        on_boot_log=None,
        snapshot_drive=None,
        reservation=None,
        retry_profile=constants.RETRY_USER_FACING,
        mem_path=None,
    )
    try:
        await vm.write_file("warm.txt", io.BytesIO(b"x"))  # connected and answering
        await _qmp(vm, "stop")
        monkeypatch.setattr(constants, "FILE_IO_TIMEOUT_SECONDS", 0.3)

        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN) as exc_info:
            await vm.write_file("stalled.txt", io.BytesIO(b"y"))

        context = exc_info.value.context
        assert (context["vm_id"], context["path"]) == (vm.vm_id, "stalled.txt")
        assert context["qemu_status"] == "paused"
        assert isinstance(context["qemu_cpu_seconds"], float)
        assert context["qemu_cpu_seconds"] > 0  # the boot and the first write ran on QEMU's CPU time
        assert context["qemu_threads"] >= 2  # QEMU's main loop and a vCPU at least
        assert context["console_log"] != "(empty)"
        assert context["exit_code"] is None  # QEMU still runs
        await _qmp(vm, "cont")
    finally:
        await vm_manager.destroy_vm(vm)

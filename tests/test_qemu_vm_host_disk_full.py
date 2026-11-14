"""A full host disk under a VM's home disk fails the VM's calls at once.

The home drive has werror=enospc: a write that finds no space on the host disk
pauses the VM, and QEMU sends BLOCK_IO_ERROR with nospace true on its event
monitor. QemuVM's event watch then fails the call in flight (OutcomeUnknownError,
not at its 120 s deadline) and every later call (TransientError LOCAL_DISK).
The same watch fails the calls in flight when the guest agent closes its port
(VSERPORT_CHANGE): see TestGuestPortClosed.
Here a fake QEMU sends the events and a fake guest agent holds the sockets.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from exec_sandbox import constants, qmp_events
from exec_sandbox.exceptions import CommunicationError, ErrorCode, OutcomeUnknownError, TransientError
from exec_sandbox.guest_channel import DualPortChannel
from exec_sandbox.models import Language
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType, VmState
from tests.conftest import time_with

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

_GREETING = b'{"QMP": {"version": {"qemu": {"micro": 3, "minor": 1, "major": 11}}, "capabilities": []}}\n'

_HOME_DRIVE = {"device": "", "qdev": "/machine/peripheral-anon/device[1]/virtio-backend", "io-status": "ok"}
"""The home drive in query-block (werror=enospc gives it an io-status), while it has room."""


def _block_io_error(**data: Any) -> bytes:
    """A BLOCK_IO_ERROR event line as QEMU 11.1 sends it (data from a real run)."""
    event = {"device": "hd2", "qom-path": "/machine/peripheral-anon/device[1]", "operation": "write"} | data
    return json.dumps({"event": "BLOCK_IO_ERROR", "data": event, "timestamp": {"seconds": 0, "microseconds": 0}})


_EMPTY_LISTING: dict[str, Any] = {"type": "file_list", "entries": []}
"""The guest agent's answer to list_files of an empty folder."""


_COMPLETE: dict[str, Any] = {"type": "complete", "exit_code": 0, "execution_time_ms": 1}
"""The guest agent's result of an exec that exited 0."""


def _vserport_change(*, port_id: str, is_open: bool, at: float | None) -> Callable[[], bytes]:
    """A VSERPORT_CHANGE event line as QEMU sends it (layout from a real run), made when
    the fake QEMU sends it. Its timestamp is `at` (time.time()), or the time it is sent."""

    def line() -> bytes:
        stamp = time.time() if at is None else at
        timestamp = {"seconds": int(stamp), "microseconds": int(stamp % 1 * 1_000_000)}
        data = {"open": is_open, "id": port_id}
        return json.dumps({"timestamp": timestamp, "event": "VSERPORT_CHANGE", "data": data}).encode()

    return line


@asynccontextmanager
async def _fake_vm(
    socket_dir: Path,
    *,
    events_on_connect: list[bytes | Callable[[], bytes]],
    events: list[bytes | Callable[[], bytes]],
    events_after: str,
    answer: Callable[[dict[str, Any]], dict[str, Any] | None],
    drives: list[dict[str, Any]],
    hold_query_answer: bool,
    drop_first: bool,
) -> AsyncIterator[tuple[QemuVM, asyncio.Event]]:
    """A QemuVM over a fake QEMU event monitor and a fake guest agent.

    The monitor greets, answers qmp_capabilities and query-block (`drives`), sends
    `events_on_connect`, then sends `events` once the guest got a request whose action
    is `events_after` (the host sent all of the call, which now waits for its answer),
    then half-closes (EOF for the watch). With `hold_query_answer`, it holds the answer
    to query-block until the guest got that request, and sends `events` before that
    answer (QEMU sends events once capabilities are negotiated). With `drop_first`, it closes its first
    connection right after the greeting (a connection lost while QEMU runs). The guest
    sends answer(request) when it is not None; for the `events_after` request, only
    once the watch read every event (it closed its monitor connection). An event made
    by a function is made when it is sent. Yields the VM and an event set when the
    watch closed its monitor connection.
    """
    got_request = asyncio.Event()
    watch_closed = asyncio.Event()
    event_writer: asyncio.Future[asyncio.StreamWriter] = asyncio.get_running_loop().create_future()

    connections = 0

    async def on_monitor(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connections
        connections += 1
        writer.write(_GREETING)
        if drop_first and connections == 1:
            writer.close()
            return
        for _ in range(2):  # qmp_capabilities, then query-block: QEMU echoes each id
            request = json.loads(await reader.readline())
            if request["execute"] == "query-block" and hold_query_answer:
                await got_request.wait()
                for line in events:
                    writer.write((line() if callable(line) else line) + b"\n")
            result = drives if request["execute"] == "query-block" else {}
            writer.write(json.dumps({"return": result, "id": request["id"]}).encode() + b"\n")
        for line in events_on_connect:
            writer.write((line() if callable(line) else line) + b"\n")
        if not hold_query_answer:
            await got_request.wait()
            for line in events:
                writer.write((line() if callable(line) else line) + b"\n")
        writer.write_eof()
        await reader.read()
        watch_closed.set()

    async def on_event(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        event_writer.set_result(writer)
        await reader.read()

    async def on_cmd(reader: asyncio.StreamReader, _writer: asyncio.StreamWriter) -> None:
        while line := await reader.readline():
            request = json.loads(line)
            trigger = request["action"] == events_after
            if trigger:
                got_request.set()
            if (response := answer(request)) is not None:
                if trigger:
                    await watch_closed.wait()
                (await event_writer).write(json.dumps(response | {"op_id": request["op_id"]}).encode() + b"\n")

    paths = {name: socket_dir / f"{name}.sock" for name in ("qmp-events", "cmd", "event")}
    servers = [
        await asyncio.start_unix_server(handler, path=str(paths[name]))
        for name, handler in (("qmp-events", on_monitor), ("cmd", on_cmd), ("event", on_event))
    ]
    qemu_exits = asyncio.Event()
    process = MagicMock()
    process.wait = AsyncMock(side_effect=qemu_exits.wait)
    process.returncode = None
    workdir = MagicMock(qmp_events_socket=paths["qmp-events"], home_disk=Path("/disks/vm.img"), use_qemu_vm_user=False)
    channel = DualPortChannel(str(paths["cmd"]), str(paths["event"]), expected_uid=os.getuid())
    vm = QemuVM(
        "disk-full-vm",
        process=process,
        cgroup=None,
        workdir=workdir,
        channel=channel,
        spec=VmSpec(language=Language.RAW),
        console_lines=deque(maxlen=10),
        base_image=Path("/test/base.qcow2"),
        packages_image=None,
        accel_type=AccelType.KVM,
        gvproxy_proc=None,
        qemu_log_task=None,
        gvproxy_log_task=None,
    )
    vm.start_event_watch()
    try:
        yield vm, watch_closed
    finally:
        qemu_exits.set()
        await vm.confirm_destroyed()  # ends the watch
        await channel.close()
        for server in servers:
            server.close()


async def test_full_host_disk_fails_the_call_in_flight_then_every_call(
    short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The event comes after the guest got list_files: that call fails at once, the next one
    raises LOCAL_DISK at its start with the home disk and a hint, and the VM is not reused.
    QEMU sends one event per failed request: the repeats change nothing."""
    monkeypatch.setattr(constants, "FILE_IO_TIMEOUT_SECONDS", 2)  # a missed event fails in 2 s, not 120 s
    nospace = _block_io_error(nospace=True, action="stop", reason="No space left on device")
    stop = json.dumps({"event": "STOP", "timestamp": {"seconds": 0, "microseconds": 0}})
    async with _fake_vm(
        short_tmp_dir,
        events_on_connect=[],
        events=[nospace.encode(), nospace.encode(), stop.encode()],
        events_after="list_files",
        answer=lambda _r: None,
        drives=[_HOME_DRIVE],
        hold_query_answer=False,
        drop_first=False,
    ) as (vm, _):
        # The event fails the call: a missed event ends it at the 2 s timeout as TransientError
        # GUEST_UNAVAILABLE ("timed out"), not as this error.
        with pytest.raises(OutcomeUnknownError, match="host disk is full"):
            await vm.list_files("")  # the paused guest never answers

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK) as caught:
            await vm.list_files("")
        assert str(caught.value) == (
            "VM disk-full-vm stopped: the host disk is full (a write to its home disk found no space)"
        )
        assert caught.value.context["home_disk"] == "/disks/vm.img"
        assert caught.value.hint == "Free space on the filesystem of home_disk_dir, then open a new session."
        assert vm.state is VmState.DESTROYING  # non-reusable: a session closes on it


async def test_full_host_disk_fails_the_exec_in_flight_then_every_exec(
    short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The event comes after the guest got the exec request: the exec waits for its output."""
    monkeypatch.setattr(constants, "EXECUTION_TIMEOUT_MARGIN_SECONDS", 1)  # a missed event fails in 2 s
    nospace = _block_io_error(nospace=True, action="stop", reason="No space left on device")
    fake = _fake_vm(
        short_tmp_dir,
        events_on_connect=[],
        events=[nospace.encode()],
        events_after="exec",
        answer=lambda _r: None,
        drives=[_HOME_DRIVE],
        hold_query_answer=False,
        drop_first=False,
    )
    async with fake as (vm, _):
        await vm.transition_state(VmState.BOOTING)
        await vm.transition_state(VmState.READY)
        # A missed event ends the exec at its 2 s host deadline, without this text.
        with pytest.raises(OutcomeUnknownError, match="host disk is full"):
            await vm.execute(
                "sleep 60", timeout_seconds=1, env_vars=None, on_stdout=None, on_stderr=None
            )  # the paused guest never answers

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
            await vm.execute("echo", timeout_seconds=1, env_vars=None, on_stdout=None, on_stderr=None)
        assert vm.state is VmState.DESTROYING


async def test_full_host_disk_fails_the_write_in_flight_then_every_write(
    short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The event comes after the guest got file_end: the write waits for its ack."""
    monkeypatch.setattr(constants, "FILE_IO_TIMEOUT_SECONDS", 2)  # a missed event fails in 2 s, not 120 s
    nospace = _block_io_error(nospace=True, action="stop", reason="No space left on device")
    fake = _fake_vm(
        short_tmp_dir,
        events_on_connect=[],
        events=[nospace.encode()],
        events_after="file_end",
        answer=lambda _r: None,
        drives=[_HOME_DRIVE],
        hold_query_answer=False,
        drop_first=False,
    )
    async with fake as (vm, _):
        # A missed event ends the write at the 2 s timeout, without this text.
        with pytest.raises(OutcomeUnknownError, match="host disk is full"):
            await vm.write_file("a.txt", io.BytesIO(b"data"))  # the paused guest never acks

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
            await vm.write_file("a.txt", io.BytesIO(b"data"))
        assert vm.state is VmState.DESTROYING


async def test_full_host_disk_during_boot_ends_the_boot_wait(short_tmp_dir: Path) -> None:
    """tiny-init formats the home disk at boot: a full host disk then pauses the VM before the
    agent answers. The boot wait raises LOCAL_DISK, not a retry until its deadline."""
    nospace = _block_io_error(nospace=True, action="stop", reason="No space left on device")
    fake = _fake_vm(
        short_tmp_dir,
        events_on_connect=[],
        events=[nospace.encode()],
        events_after="ping",
        answer=lambda _r: None,
        drives=[_HOME_DRIVE],
        hold_query_answer=False,
        drop_first=False,
    )
    async with fake as (vm, _):
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
            await vm.wait_for_guest(timeout=30)


async def test_io_error_without_nospace_is_not_a_full_disk(short_tmp_dir: Path) -> None:
    """werror=report drives (overlay, packages) send no nospace: the guest got the error, the VM runs."""
    report = _block_io_error(action="report", reason="Input/output error")
    listing = {"type": "file_list", "entries": []}
    async with _fake_vm(
        short_tmp_dir,
        events_on_connect=[],
        events=[report.encode()],
        events_after="list_files",
        answer=lambda _r: listing,
        drives=[_HOME_DRIVE],
        hold_query_answer=False,
        drop_first=False,
    ) as (vm, watch_closed):
        assert await vm.list_files("") == []
        await asyncio.wait_for(watch_closed.wait(), timeout=1)  # the watch read the event, then EOF
        assert await vm.list_files("") == []


def _home_drive_full() -> dict[str, Any]:
    return _HOME_DRIVE | {"io-status": "nospace"}


async def _until_stopped(vm: QemuVM) -> None:
    """Until the watch stops the VM for a full host disk (5 s at most; it takes ms)."""
    async with asyncio.timeout(5):
        while not vm.host_disk_full:  # noqa: ASYNC110 - the VM has no event for this stop
            await asyncio.sleep(0.005)


class TestEventWatch:
    """The watch misses no full host disk: one before its connection, a bad line, a lost
    connection, a slow QEMU start."""

    async def test_disk_full_before_the_watch_connected_is_found(self, short_tmp_dir: Path) -> None:
        """QEMU sends no event before capabilities negotiation ends: query-block right after
        it finds the drive stopped (io-status nospace), so the first call raises LOCAL_DISK."""
        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=[],
            events=[],
            events_after="never",
            answer=lambda _r: None,
            drives=[_home_drive_full()],
            hold_query_answer=False,
            drop_first=False,
        )
        async with fake as (vm, _):
            await _until_stopped(vm)
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK) as caught:
                await vm.list_files("")
            assert caught.value.context["device"] == _HOME_DRIVE["qdev"]

    async def test_unreadable_line_is_skipped(self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A line that is not JSON does not end the watch: the event after it still stops the VM."""
        monkeypatch.setattr(constants, "FILE_IO_TIMEOUT_SECONDS", 2)  # a missed event fails in 2 s, not 120 s
        nospace = _block_io_error(nospace=True, action="stop", reason="No space left on device")
        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=[],
            events=[b"{not json", nospace.encode()],
            events_after="list_files",
            answer=lambda _r: None,
            drives=[_HOME_DRIVE],
            hold_query_answer=False,
            drop_first=False,
        )
        async with fake as (vm, _):
            with pytest.raises(OutcomeUnknownError, match="host disk is full"):
                await vm.list_files("")

    async def test_connection_lost_while_qemu_runs_is_made_again(self, short_tmp_dir: Path) -> None:
        """The first connection ends before its handshake while QEMU runs: the watch connects
        again, and that connection's query-block finds the full disk."""
        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=[],
            events=[],
            events_after="never",
            answer=lambda _r: None,
            drives=[_home_drive_full()],
            hold_query_answer=False,
            drop_first=True,
        )
        async with fake as (vm, _):
            await _until_stopped(vm)
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                await vm.list_files("")

    async def test_a_close_before_the_first_query_answer_is_not_lost(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QEMU sends events as soon as capabilities are negotiated, so a close can come
        before the answer to the watch's query-block (right after a (re)connect): the watch
        acts on it there too, and the exec in flight fails at once. A close after that
        answer: TestGuestPortClosed.test_close_fails_the_exec_in_flight_and_not_the_next_call."""
        monkeypatch.setattr(constants, "EXECUTION_TIMEOUT_MARGIN_SECONDS", 1)  # a missed event fails in 2 s
        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=[],
            events=[_vserport_change(port_id=constants.GUEST_PORT_IDS[0], is_open=False, at=None)],
            events_after="exec",
            answer=lambda _r: None,
            drives=[_HOME_DRIVE],
            hold_query_answer=True,
            drop_first=False,
        )
        async with fake as (vm, _):
            await vm.transition_state(VmState.BOOTING)
            await vm.transition_state(VmState.READY)
            start = asyncio.get_running_loop().time()
            # A missed event ends the exec at its 2 s host deadline, without this text.
            with pytest.raises(OutcomeUnknownError, match="closed its port guest-cmd"):
                await vm.execute(
                    "sleep 60", timeout_seconds=1, env_vars=None, on_stdout=None, on_stderr=None
                )  # the guest dropped the request
            assert asyncio.get_running_loop().time() - start < 0.5


def _paused_vm_monitor(status: str) -> Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]:
    """A QEMU command monitor whose query-status answers `status`."""

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_GREETING)
        while line := await reader.readline():
            request = json.loads(line)
            result = {"status": status, "running": False} if request["execute"] == "query-status" else {}
            writer.write(json.dumps({"return": result, "id": request["id"]}).encode() + b"\n")
        writer.close()

    return serve


class TestStopNotSeenByTheWatch:
    """Paths that meet a VM QEMU paused on a full host disk before (or without) its event."""

    @pytest.mark.parametrize(("status", "full"), [("io-error", True), ("running", False), ("paused", False)])
    async def test_run_state_io_error_is_a_full_host_disk(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch, status: str, full: bool
    ) -> None:
        """A guest that does not answer (an L1 restore probe, a package install): QEMU's run
        state io-error tells a full host disk (only the home drive stops on a write error)
        before its event reaches the watch. The check then raises LOCAL_DISK and fails the
        call in flight; another run state fails nothing."""
        monkeypatch.setattr(constants, "FILE_IO_TIMEOUT_SECONDS", 2)  # a call not failed ends in 2 s, not 120 s
        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=[],
            events=[],  # no BLOCK_IO_ERROR reaches the watch
            events_after="list_files",
            answer=lambda _r: None,
            drives=[_HOME_DRIVE],
            hold_query_answer=False,
            drop_first=False,
        )
        async with fake as (vm, watch_closed):
            vm.workdir.qmp_socket = short_tmp_dir / "qmp.sock"  # the fake workdir has no command monitor
            server = await asyncio.start_unix_server(_paused_vm_monitor(status), path=str(vm.qmp_socket))
            async with server:
                in_flight = asyncio.create_task(vm.list_files(""))  # the paused guest never answers
                await asyncio.wait_for(watch_closed.wait(), timeout=1)  # the guest got the request
                if full:
                    with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                        await vm.raise_if_host_disk_full()
                    with pytest.raises(OutcomeUnknownError, match="host disk is full"):
                        await in_flight
                    assert vm.host_disk_full
                    assert vm.state is VmState.DESTROYING
                else:
                    await vm.raise_if_host_disk_full()
                    assert not vm.host_disk_full
                    assert not in_flight.done()
                    in_flight.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await in_flight

    async def test_call_not_sent_when_the_stop_lands_is_local_disk(self, short_tmp_dir: Path) -> None:
        """The call passed its start check; the stop fails it before it was sent: LOCAL_DISK
        (the session closes), not GUEST_UNAVAILABLE (a retry on a VM that never runs again).
        A race window: a fake channel's connect lets the fake QEMU send BLOCK_IO_ERROR and
        waits until the watch stopped the VM, between the start check and the send."""
        send_event = asyncio.Event()
        nospace = _block_io_error(nospace=True, action="stop", reason="No space left on device")

        async def monitor(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.write(_GREETING)
            for _ in range(2):  # qmp_capabilities, then query-block: QEMU echoes each id
                request = json.loads(await reader.readline())
                result = [_HOME_DRIVE] if request["execute"] == "query-block" else {}
                writer.write(json.dumps({"return": result, "id": request["id"]}).encode() + b"\n")
            await send_event.wait()
            writer.write(nospace.encode() + b"\n")
            writer.write_eof()
            await reader.read()

        channel = MagicMock()
        events_socket = short_tmp_dir / "qmp-events.sock"
        workdir = MagicMock(qmp_events_socket=events_socket, home_disk=Path("/disks/vm.img"), use_qemu_vm_user=False)
        vm = QemuVM(
            "paused-vm",
            process=MagicMock(returncode=None),
            cgroup=None,
            workdir=workdir,
            channel=channel,
            spec=VmSpec(language=Language.RAW),
            console_lines=deque(maxlen=10),
            base_image=Path("/test/base.qcow2"),
            packages_image=None,
            accel_type=AccelType.KVM,
            gvproxy_proc=None,
            qemu_log_task=None,
            gvproxy_log_task=None,
        )

        async def stop_then_fail(*_args: object, **_kwargs: object) -> None:
            send_event.set()
            await _until_stopped(vm)
            raise CommunicationError("operations aborted: the host disk is full", context=None)

        channel.connect = AsyncMock(side_effect=stop_then_fail)
        server = await asyncio.start_unix_server(monitor, path=str(events_socket))
        vm.start_event_watch()
        try:
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                await vm.list_files("")
            assert vm.state is VmState.DESTROYING
        finally:
            await vm.confirm_destroyed()  # ends the watch
            server.close()


class TestGuestPortClosed:
    """The guest agent closes its port only on a port error, then opens it again: the
    calls in flight get no answer. They fail at the event, not at their deadline."""

    _CMD, _EVENT = constants.GUEST_PORT_IDS

    @staticmethod
    async def _ready(vm: QemuVM) -> None:
        await vm.transition_state(VmState.BOOTING)
        await vm.transition_state(VmState.READY)

    async def test_close_fails_the_exec_in_flight_and_not_the_next_call(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(constants, "EXECUTION_TIMEOUT_MARGIN_SECONDS", 1)  # a missed event fails in 2 s
        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=[],
            events=[_vserport_change(port_id=self._CMD, is_open=False, at=None)],
            events_after="exec",
            answer=lambda r: _EMPTY_LISTING if r["action"] == "list_files" else None,
            drives=[_HOME_DRIVE],
            hold_query_answer=False,
            drop_first=False,
        )
        async with fake as (vm, _):
            await self._ready(vm)
            start = asyncio.get_running_loop().time()
            # A missed event ends the exec at its 2 s host deadline, without this text.
            with (
                caplog.at_level(logging.DEBUG, logger="exec_sandbox"),
                pytest.raises(OutcomeUnknownError, match="closed its port guest-cmd"),
            ):
                await vm.execute(
                    "sleep 60", timeout_seconds=1, env_vars=None, on_stdout=None, on_stderr=None
                )  # the guest dropped the request
            assert asyncio.get_running_loop().time() - start < 0.5
            assert [r.levelno for r in caplog.records if "closed its port" in r.getMessage()] == [logging.ERROR]

            # The guest serves the next call on its port, open again.
            assert await vm.list_files("") == []

    async def test_open_of_an_open_port_fails_the_exec_in_flight(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QEMU sends one event per port each second: a close and the open after it in
        that second come as one open, of a port open since boot."""
        monkeypatch.setattr(constants, "EXECUTION_TIMEOUT_MARGIN_SECONDS", 1)  # a missed event fails in 2 s
        boot = [_vserport_change(port_id=port, is_open=True, at=None) for port in (self._CMD, self._EVENT)]
        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=boot,
            events=[_vserport_change(port_id=self._CMD, is_open=True, at=None)],
            events_after="exec",
            answer=lambda r: _EMPTY_LISTING if r["action"] == "list_files" else None,
            drives=[_HOME_DRIVE],
            hold_query_answer=False,
            drop_first=False,
        )
        async with fake as (vm, _):
            await self._ready(vm)
            start = asyncio.get_running_loop().time()
            with pytest.raises(OutcomeUnknownError, match="closed its port guest-cmd and opened it again"):
                await vm.execute("sleep 60", timeout_seconds=1, env_vars=None, on_stdout=None, on_stderr=None)
            assert asyncio.get_running_loop().time() - start < 0.5
            assert await vm.list_files("") == []

    async def test_first_open_of_each_port_fails_no_call(self, short_tmp_dir: Path) -> None:
        """The first open of each port (boot, or an L1 restore that loads open ports) and a
        port without a guest-agent id fail nothing: the guest answers the exec after the
        watch read the events."""
        events = [
            _vserport_change(port_id=self._CMD, is_open=True, at=None),
            _vserport_change(port_id=self._EVENT, is_open=True, at=None),
            _vserport_change(port_id="other-port", is_open=False, at=None),
        ]
        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=[],
            events=events,
            events_after="exec",
            answer=lambda _r: _COMPLETE,
            drives=[_HOME_DRIVE],
            hold_query_answer=False,
            drop_first=False,
        )
        async with fake as (vm, _):
            await self._ready(vm)
            assert (
                await vm.execute("echo", timeout_seconds=1, env_vars=None, on_stdout=None, on_stderr=None)
            ).exit_code == 0

    async def test_change_before_the_call_was_sent_does_not_fail_it(self, short_tmp_dir: Path) -> None:
        """A close and an open that QEMU made before the exec was sent (their events can
        come up to 1 s late): the guest got the exec on its open port."""
        events = [
            _vserport_change(port_id=self._CMD, is_open=False, at=time.time()),
            _vserport_change(port_id=self._EVENT, is_open=True, at=time.time()),
        ]
        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=[_vserport_change(port_id=self._EVENT, is_open=True, at=None)],
            events=events,
            events_after="exec",
            answer=lambda _r: _COMPLETE,
            drives=[_HOME_DRIVE],
            hold_query_answer=False,
            drop_first=False,
        )
        async with fake as (vm, _):
            await self._ready(vm)
            assert (
                await vm.execute("echo", timeout_seconds=1, env_vars=None, on_stdout=None, on_stderr=None)
            ).exit_code == 0

    @pytest.mark.parametrize("step", [3600.0, -5.0, -3600.0], ids=["back-1h", "forward-5s", "forward-1h"])
    async def test_close_after_a_wall_clock_step_fails_the_exec_in_flight(
        self, short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch, step: float
    ) -> None:
        """The host wall clock steps after the exec was sent and the watch took its sample: back
        before QEMU stamps the close, or forward between the stamp and its read. The step shows
        in the wall-against-monotonic offset, and the event is taken as made now."""
        monkeypatch.setattr(constants, "EXECUTION_TIMEOUT_MARGIN_SECONDS", 1)  # a missed event fails in 2 s
        real_time = time.time
        stepped = [False]
        monkeypatch.setattr(qmp_events, "time", time_with(time=lambda: real_time() - (step if stepped[0] else 0.0)))

        def close_on_the_stepped_clock() -> bytes:
            if step > 0:  # back: the step, then QEMU stamps on the stepped clock
                stepped[0] = True
                stamp = real_time() - step
            else:  # forward: QEMU stamps, then the step, before the read
                stamp = real_time()
                stepped[0] = True
            timestamp = {"seconds": int(stamp), "microseconds": int(stamp % 1 * 1_000_000)}
            data = {"open": False, "id": self._CMD}
            return json.dumps({"timestamp": timestamp, "event": "VSERPORT_CHANGE", "data": data}).encode()

        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=[],
            events=[close_on_the_stepped_clock],
            events_after="exec",
            answer=lambda _r: None,
            drives=[_HOME_DRIVE],
            hold_query_answer=False,
            drop_first=False,
        )
        async with fake as (vm, _):
            await self._ready(vm)
            start = asyncio.get_running_loop().time()
            with pytest.raises(OutcomeUnknownError, match="closed its port guest-cmd"):
                await vm.execute("sleep 60", timeout_seconds=1, env_vars=None, on_stdout=None, on_stderr=None)
            assert asyncio.get_running_loop().time() - start < 0.5

    @pytest.mark.parametrize("end", ["destroy", "qemu-shutdown"])
    async def test_close_while_the_vm_ends_fails_nothing(
        self, short_tmp_dir: Path, caplog: pytest.LogCaptureFixture, end: str
    ) -> None:
        """A close while the VM ends (a destroy started, or QEMU sent SHUTDOWN: a quit after
        a save that stops the VM, a signal) is expected: no ERROR, and the end, not the
        close, fails what it stops. Here the guest still answers the call in flight."""
        shutdown = json.dumps(
            {
                "timestamp": {"seconds": 0, "microseconds": 0},
                "event": "SHUTDOWN",
                "data": {"guest": False, "reason": "host-qmp-quit"},
            }
        ).encode()
        closes = [_vserport_change(port_id=port, is_open=False, at=None) for port in (self._CMD, self._EVENT)]
        fake = _fake_vm(
            short_tmp_dir,
            events_on_connect=[],
            events=closes if end == "destroy" else [shutdown, *closes],
            events_after="list_files",
            answer=lambda _r: _EMPTY_LISTING,
            drives=[_HOME_DRIVE],
            hold_query_answer=False,
            drop_first=False,
        )
        async with fake as (vm, _):
            if end == "destroy":
                await vm.begin_destroy()
            with caplog.at_level(logging.DEBUG, logger="exec_sandbox"):
                assert await vm.list_files("") == []
            assert [r.levelno for r in caplog.records if "closed its port" in r.getMessage()] == [logging.DEBUG] * 2

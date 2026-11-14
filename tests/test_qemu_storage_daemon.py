"""Tests for QemuStorageDaemon.

Test philosophy:
- Unit tests: Pure logic only, no real daemon process, no mocks unless unavoidable
- Integration tests: Real daemon, real files, real code paths
- Error tests: Real daemon, verify error handling works correctly

No hwaccel required: qemu-storage-daemon is a host-side tool (QMP over unix socket)
that creates qcow2 overlays — it does not boot VMs or use KVM/HVF.
"""

import asyncio
import contextlib
import errno
import json
import os
import secrets
import select
import shutil
import signal
import sys
import tempfile
import threading
from collections import deque
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from exec_sandbox import process_registry, qemu_storage_daemon
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.models import Language
from exec_sandbox.overlay_pool import OverlayPool
from exec_sandbox.qemu_storage_daemon import QemuStorageDaemon, _pid_namespace
from exec_sandbox.vm_manager import VmManager
from tests.conftest import asyncio_with, skip_unless_linux
from tests.test_subprocess_utils import write_foreign_elf

# ============================================================================
# Test Helpers & Fixtures
# ============================================================================

_DAEMON_ARGS_END = ["--monitor", "chardev=qmp0"]
"""The last arguments of each daemon launch (its QMP monitor)."""


def _daemon_sockets() -> dict[int, Path]:
    """The daemon processes that this process started and that run (no zombie), with the
    QMP socket of each, read from its command line (--chardev socket,path=...)."""
    daemons: dict[int, Path] = {}
    for child in psutil.Process().children():
        with contextlib.suppress(psutil.Error):
            cmdline = child.cmdline()
            if child.status() == psutil.STATUS_ZOMBIE or cmdline[-2:] != _DAEMON_ARGS_END:
                continue
            chardev = cmdline[cmdline.index("--chardev") + 1]
            daemons[child.pid] = Path(chardev.split(",")[1].removeprefix("path="))
    return daemons


_NO_GREETING_DAEMON = """#!{python}
import socket, sys, time
chardev = sys.argv[sys.argv.index("--chardev") + 1]
server = socket.socket(socket.AF_UNIX)
server.bind(chardev.split(",")[1].removeprefix("path="))
server.listen()
connection, _ = server.accept()
connection.sendall(b'{{"not": "a QMP greeting"}}\\n')
time.sleep(60)
"""
"""A qemu-storage-daemon that serves its socket, sends a line that is no QMP greeting, then
runs until it is ended."""


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process registry of this test only: force_kill_all() then reaches no process of
    another test of this worker."""
    monkeypatch.setattr("exec_sandbox.process_registry._process_groups", set())


async def _image_info(path: Path) -> dict[str, Any]:
    """qemu-img info of `path` (JSON): the image as QEMU reads it."""
    proc = await asyncio.create_subprocess_exec(
        "qemu-img", "info", "-U", "--output=json", str(path), stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    assert proc.returncode == 0, path
    return json.loads(stdout)


def _qcow2_header(size: int) -> bytes:
    """The first 32 bytes of a qcow2 v3 image of `size` bytes (magic, version, size at 24)."""
    return b"QFI\xfb" + (3).to_bytes(4, "big") + bytes(16) + size.to_bytes(8, "big")


class _ScriptedQmp:
    """A QMP peer in this process, as the reader and the writer of a daemon's connection.

    It answers each command as qemu-storage-daemon does (blockdev-create starts a job that
    ends at once and sends its JOB_STATUS_CHANGE events; query-jobs lists the jobs;
    job-dismiss removes one), and sends the `noise` lines before each answer. `commands`
    holds each command that the daemon sent.
    """

    def __init__(self, noise: list[dict[str, Any]]) -> None:
        self.commands: list[dict[str, Any]] = []
        self._noise = [json.dumps(line).encode() + b"\n" for line in noise]
        self._jobs: list[str] = []
        self._lines: deque[bytes] = deque()

    def write(self, data: bytes) -> None:
        command = json.loads(data)
        self.commands.append(command)
        arguments = command.get("arguments", {})
        result: Any = {}
        if command["execute"] == "blockdev-create":
            self._jobs.append(arguments["job-id"])
            for status in ("created", "running", "pending", "concluded"):
                event = {"event": "JOB_STATUS_CHANGE", "data": {"id": arguments["job-id"], "status": status}}
                self._lines.append(json.dumps(event).encode() + b"\n")
        elif command["execute"] == "query-jobs":
            result = [{"id": job, "type": "create", "status": "concluded"} for job in self._jobs]
        elif command["execute"] == "job-dismiss":
            self._jobs.remove(arguments["id"])
        self._lines.extend(self._noise)
        self._lines.append(json.dumps({"return": result, "id": command["id"]}).encode() + b"\n")

    async def drain(self) -> None:
        pass

    async def readline(self) -> bytes:
        return self._lines.popleft()


@pytest.fixture
def base_image(vm_settings) -> Path:  # type: ignore[no-untyped-def]
    """Resolve the Python base image path without starting the overlay pool."""
    return VmManager(vm_settings).images.base_image(Language.PYTHON)  # type: ignore[arg-type]


@pytest.fixture
async def started_daemon() -> AsyncGenerator[QemuStorageDaemon, None]:
    """Started QemuStorageDaemon, torn down after the test.

    Tests that verify start/stop behavior should create their own
    daemon instance instead of using this fixture.
    """
    d = QemuStorageDaemon()
    await d.start()
    yield d
    await d.stop()


# ============================================================================
# Unit Tests - Pure Logic (no daemon, no mocks)
# ============================================================================


class TestOverlayBackingPath:
    """The overlay keeps its base image path; the pool moves the overlay to the VM's folder."""

    async def test_overlay_has_the_size_of_its_base(self, started_daemon: QemuStorageDaemon, tmp_path: Path) -> None:
        """The size comes from the base's qcow2 header (no qemu-img process per overlay)."""
        base = tmp_path / "base.qcow2"
        proc = await asyncio.create_subprocess_exec("qemu-img", "create", "-q", "-f", "qcow2", str(base), "77M")
        assert await proc.wait() == 0
        overlay = tmp_path / "overlay.qcow2"

        await started_daemon.create_overlay(base, overlay)

        proc = await asyncio.create_subprocess_exec(
            "qemu-img", "info", "--output=json", str(overlay), stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        assert json.loads(stdout)["virtual-size"] == 77 * 1024 * 1024

    async def test_a_base_image_rebuilt_in_place_gives_its_new_size(
        self, started_daemon: QemuStorageDaemon, tmp_path: Path
    ) -> None:
        """The daemon reads each base image's size once, per version of the file: a rebuild at the
        same path (make build) is read again, not served from the first read."""
        base = tmp_path / "base.qcow2"
        sizes = []
        for size in ("77M", "33M"):
            base.unlink(missing_ok=True)
            proc = await asyncio.create_subprocess_exec("qemu-img", "create", "-q", "-f", "qcow2", str(base), size)
            assert await proc.wait() == 0
            overlay = tmp_path / f"overlay-{size}.qcow2"
            await started_daemon.create_overlay(base, overlay)
            proc = await asyncio.create_subprocess_exec(
                "qemu-img", "info", "--output=json", str(overlay), stdout=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            sizes.append(json.loads(stdout)["virtual-size"])

        assert sizes == [77 * 1024 * 1024, 33 * 1024 * 1024]

    async def test_a_relative_base_image_still_opens_after_the_move(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(base_image.parent)
        overlay = tmp_path / "pool" / "overlay.qcow2"
        overlay.parent.mkdir()
        await started_daemon.create_overlay(Path(base_image.name), overlay)
        moved = tmp_path / "vm" / "overlay.qcow2"
        moved.parent.mkdir()
        overlay.rename(moved)

        monkeypatch.chdir(tmp_path)
        proc = await asyncio.create_subprocess_exec(
            "qemu-img", "info", "-U", "--backing-chain", str(moved), stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        assert proc.returncode == 0, stderr.decode()


class TestQemuStorageDaemonUnit:
    """Start, stop and QMP errors, and the faults that a real daemon does not show on demand
    (each fake says why)."""

    async def test_create_overlay_when_not_started_raises_error(self) -> None:
        """create_overlay raises if daemon not started."""
        daemon = QemuStorageDaemon()
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED, match="not started"):
            await daemon.create_overlay(Path("/fake/base.qcow2"), Path("/fake/out.qcow2"))

    async def test_cancelled_exchange_drains_identified_response_before_next_command(self) -> None:
        """Cancellation cannot shift the shared QMP response stream. A race window: the cancel
        must come while the answer is on its way, so a fake stream holds that answer."""
        daemon = QemuStorageDaemon()
        first_read = asyncio.Event()
        release_first = asyncio.Event()
        calls = 0

        async def readline() -> bytes:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_read.set()
                await release_first.wait()
                return b'{"return":{"command":1},"id":"qsd-1"}\n'
            return b'{"return":{"command":2},"id":"qsd-2"}\n'

        reader = MagicMock()
        reader.readline = readline
        writer = MagicMock()
        writer.drain = AsyncMock()
        daemon._reader = reader
        daemon._writer = writer

        first = asyncio.create_task(daemon._execute("first"))
        await first_read.wait()
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done()

        release_first.set()
        with pytest.raises(asyncio.CancelledError):
            await first

        assert await daemon._execute("second") == {"command": 2}
        sent = [json.loads(call.args[0]) for call in writer.write.call_args_list]
        assert [message["id"] for message in sent] == ["qsd-1", "qsd-2"]

    async def test_stop_waits_until_the_daemon_death_is_confirmed(self) -> None:
        """A kill that is not confirmed in its timeout is tried again, 1 s later: stop() returns
        only once the death of the daemon is confirmed. An OS fault that no real daemon shows
        on demand: the process is a fake and cleanup_process (the kill) fails once."""
        daemon = QemuStorageDaemon()
        daemon._process = MagicMock()  # a daemon that a failed start left
        with (
            patch(
                "exec_sandbox.qemu_storage_daemon.cleanup_process",
                new_callable=AsyncMock,
                side_effect=[False, True],
            ) as kill,
            patch("exec_sandbox.qemu_storage_daemon.asyncio", asyncio_with(sleep=(sleep := AsyncMock()))),
        ):
            await daemon.stop()

        assert kill.await_count == 2
        sleep.assert_awaited_once_with(1.0)

    async def test_start_spawns_only_after_the_death_of_a_retained_daemon(self) -> None:
        """A failed start can leave a daemon whose death is not confirmed yet (an OS fault, so
        the process is a fake): the next start ends it first, and spawns a new daemon only
        once that death is confirmed."""
        daemon = QemuStorageDaemon()
        daemon._process = MagicMock()  # a daemon that a failed start left
        steps: list[str] = []

        async def kill(*_args: object, **_kwargs: object) -> bool:
            steps.append("kill")
            return steps.count("kill") == 2  # the second kill is confirmed

        async def spawn(*_args: object, **_kwargs: object) -> None:
            steps.append("spawn")
            raise OSError(errno.EAGAIN, os.strerror(errno.EAGAIN))  # no real daemon starts

        with (
            patch("exec_sandbox.qemu_storage_daemon.cleanup_process", new_callable=AsyncMock, side_effect=kill),
            patch("exec_sandbox.qemu_storage_daemon.start_managed_process", new_callable=AsyncMock, side_effect=spawn),
            patch("exec_sandbox.qemu_storage_daemon.asyncio", asyncio_with(sleep=AsyncMock())),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED),
        ):
            await daemon.start()

        assert steps == ["kill", "kill", "spawn"]

    async def test_stop_without_start_is_safe(self) -> None:
        """Stopping un-started daemon is a no-op (idempotent)."""
        daemon = QemuStorageDaemon()
        await daemon.stop()  # Should not raise
        assert not daemon.started

    async def test_qmp_error_keeps_its_class_and_description(self, started_daemon: QemuStorageDaemon) -> None:
        """A QMP error answer raises TransientError OVERLAY_FAILED: the message is the QMP
        description, context["error_class"] the QMP error class. The public calls send only
        valid commands: an unknown command gets a real error answer from the daemon."""
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED) as exc_info:
            await started_daemon._execute("no-such-command")

        assert str(exc_info.value) == "The command no-such-command has not been found"
        assert exc_info.value.context == {"error_class": "CommandNotFound"}

    @pytest.mark.parametrize(
        ("content", "mode", "err"),
        [
            (None, 0, errno.ENOENT),
            (b"#!/bin/sh\n", 0o644, errno.EACCES),
            (b"not a program\n", 0o755, errno.ENOEXEC),
        ],
        ids=["missing", "no-exec-bit", "not-a-program"],
    )
    async def test_binary_that_cannot_run_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes | None, mode: int, err: int
    ) -> None:
        """A qemu-storage-daemon that is missing, that this user cannot run, or that is not a
        program fails on every retry: ConfigError HOST_DEPENDENCY with the errno, not the
        transient OVERLAY_FAILED."""
        binary = tmp_path / "qemu-storage-daemon"
        if content is not None:
            binary.write_bytes(content)
            binary.chmod(mode)
        monkeypatch.setattr("exec_sandbox.qemu_storage_daemon._BINARY", str(binary))
        daemon = QemuStorageDaemon()
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc_info:
            await daemon.start()

        assert exc_info.value.context == {"binary": str(binary), "errno": err}
        assert not daemon.started

    async def test_process_limit_at_spawn_is_transient(self) -> None:
        """A fork that hits a process limit (EAGAIN: an OS state that a test cannot make, so
        the spawn is a fake) can pass later: TransientError OVERLAY_FAILED with the errno."""
        daemon = QemuStorageDaemon()
        with (
            patch(
                "exec_sandbox.qemu_storage_daemon.start_managed_process",
                new_callable=AsyncMock,
                side_effect=OSError(errno.EAGAIN, os.strerror(errno.EAGAIN)),
            ),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED) as exc_info,
        ):
            await daemon.start()

        assert exc_info.value.context == {"binary": "qemu-storage-daemon", "errno": errno.EAGAIN}
        assert not daemon.started

    @skip_unless_linux
    async def test_daemon_of_another_arch_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A qemu-storage-daemon this host cannot exec: the launch shell ends with 126 (it
        runs the exec, so Popen cannot report it): ConfigError HOST_DEPENDENCY with the
        install hint, not the transient OVERLAY_FAILED."""
        # A path, not the name: like Popen, a shell goes on to the next PATH entry after an exec error
        binary = str(write_foreign_elf(tmp_path / "qemu-storage-daemon"))
        monkeypatch.setattr("exec_sandbox.qemu_storage_daemon._BINARY", binary)
        daemon = QemuStorageDaemon()
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc_info:
            await daemon.start()
        assert str(exc_info.value).startswith(f"Failed to start {binary}: exit status 126 (")
        assert exc_info.value.context == {"binary": binary, "exit_code": 126}
        assert not daemon.started

    async def test_daemon_starts_with_no_preexec(self) -> None:
        """The daemon asks the launch to end it with this process, with no step: where setpriv
        works, the launch then has no Python preexec_fn, so Popen can vfork. The
        launch is a fake: no process shows if Popen forked or vforked it. The launch itself
        is the contract of start_managed_process; the real end: TestDaemonEndsWithItsProcess."""
        daemon = QemuStorageDaemon()
        launch = AsyncMock(side_effect=OSError(errno.EAGAIN, os.strerror(errno.EAGAIN)))
        with (
            patch("exec_sandbox.qemu_storage_daemon.start_managed_process", launch),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED),
        ):
            await daemon.start()
        assert (launch.call_args.kwargs["steps"], launch.call_args.kwargs["end_with_this_process"]) == ((), True)

    # --- QMP lines that are not the answer ---

    async def test_overlay_skips_malformed_and_unmatched_qmp_lines(self, tmp_path: Path) -> None:
        """Before each answer, the daemon reads events (malformed ones too: no data, data that
        is not an object, no id) and answers of no command it sent: it skips them, and the
        overlay completes with the same commands. A real qemu-storage-daemon sends no such
        line, so a peer in this process sends them."""
        base = tmp_path / "base.qcow2"
        base.write_bytes(_qcow2_header(77 * 1024 * 1024))
        noise: list[dict[str, Any]] = [
            {"event": "JOB_STATUS_CHANGE"},
            {"event": "JOB_STATUS_CHANGE", "data": "not-an-object"},
            {"event": "JOB_STATUS_CHANGE", "data": 42},
            {"event": "JOB_STATUS_CHANGE", "data": ["a", "b"]},
            {"event": "JOB_STATUS_CHANGE", "data": None},
            {"event": "JOB_STATUS_CHANGE", "data": {"status": "concluded"}},
            {"event": "BLOCK_IO_ERROR", "data": {"device": "", "operation": "write", "action": "report"}},
            {"data": {"id": "file-0", "status": "concluded"}},
            {"return": {}, "id": "qsd-0"},
        ]
        peer = _ScriptedQmp(noise)
        daemon = QemuStorageDaemon()
        daemon._reader = cast("asyncio.StreamReader", peer)
        daemon._writer = cast("asyncio.StreamWriter", peer)
        daemon._started = True

        await daemon.create_overlay(base, tmp_path / "overlay.qcow2")

        assert [command["execute"] for command in peer.commands] == [
            "blockdev-create",
            "query-jobs",
            "job-dismiss",
            "blockdev-add",
            "blockdev-create",
            "query-jobs",
            "job-dismiss",
            "blockdev-del",
        ]
        created = [
            command["arguments"]["job-id"] for command in peer.commands if command["execute"] == "blockdev-create"
        ]
        dismissed = [command["arguments"]["id"] for command in peer.commands if command["execute"] == "job-dismiss"]
        assert dismissed == created

    @staticmethod
    def _daemon_whose_peer_ends(*, lines_before: int, end: bytes) -> tuple[QemuStorageDaemon, _ScriptedQmp]:
        """A started daemon whose scripted peer answers `lines_before` lines, then gives `end`
        for each read (b"": the stream ended). The peer's lines for one overlay: blockdev-create
        0-4 (4 events, the answer), query-jobs 5, job-dismiss 6, blockdev-add 7, blockdev-create
        8-12, query-jobs 13, job-dismiss 14, blockdev-del 15."""
        peer = _ScriptedQmp([])
        answered = 0
        scripted_readline = peer.readline

        async def ending_readline() -> bytes:
            nonlocal answered
            if answered >= lines_before:
                return end
            answered += 1
            return await scripted_readline()

        peer.readline = ending_readline  # type: ignore[method-assign]
        daemon = QemuStorageDaemon()
        daemon._reader = cast("asyncio.StreamReader", peer)
        daemon._writer = cast("asyncio.StreamWriter", peer)
        daemon._started = True
        return daemon, peer

    @pytest.mark.parametrize(
        "lines_before",
        [0, 5, 7, 9],
        ids=["first-blockdev-create", "job-wait-query-jobs", "blockdev-add", "second-blockdev-create"],
    )
    @pytest.mark.parametrize(
        ("end", "message"),
        [
            (b"", "closed its socket"),
            (b"not json\n", "not JSON"),
            (b"[1, 2]\n", "not a JSON object"),
            (b"null\n", "not a JSON object"),
            (b"[" * 60_000 + b"\n", "not JSON"),
        ],
        ids=["end-of-stream", "torn-line", "json-array", "json-null", "nested-too-deep"],
    )
    async def test_a_daemon_that_dies_mid_overlay_is_overlay_failed(
        self, tmp_path: Path, lines_before: int, end: bytes, message: str
    ) -> None:
        """The daemon dies (OOM) during create_overlay: its stream ends, or a torn line or a
        line that is no QMP object comes. The error is TransientError OVERLAY_FAILED, never a
        raw JSONDecodeError, AttributeError or TypeError. A peer in this process
        ends on cue."""
        base = tmp_path / "base.qcow2"
        base.write_bytes(_qcow2_header(77 * 1024 * 1024))
        daemon, _ = self._daemon_whose_peer_ends(lines_before=lines_before, end=end)

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED, match=message):
            await daemon.create_overlay(base, tmp_path / "overlay.qcow2")

    async def test_a_daemon_that_dies_in_the_cleanup_leaves_a_complete_overlay(self, tmp_path: Path) -> None:
        """The stream ends first at the blockdev-del cleanup: the qcow2 job has concluded, so
        the overlay is complete and create_overlay returns; the cleanup error is suppressed."""
        base = tmp_path / "base.qcow2"
        base.write_bytes(_qcow2_header(77 * 1024 * 1024))
        daemon, peer = self._daemon_whose_peer_ends(lines_before=15, end=b"")

        await daemon.create_overlay(base, tmp_path / "overlay.qcow2")

        assert peer.commands[-1]["execute"] == "blockdev-del"

    async def test_a_qmp_error_that_is_not_an_object_is_overlay_failed(self, tmp_path: Path) -> None:
        """An answer whose "error" is not an object: OVERLAY_FAILED "Unknown QMP error", not a
        raw AttributeError."""
        base = tmp_path / "base.qcow2"
        base.write_bytes(_qcow2_header(77 * 1024 * 1024))
        peer = _ScriptedQmp([])

        async def error_readline() -> bytes:
            return json.dumps({"error": "not an object", "id": peer.commands[-1]["id"]}).encode() + b"\n"

        peer.readline = error_readline  # type: ignore[method-assign]
        daemon = QemuStorageDaemon()
        daemon._reader = cast("asyncio.StreamReader", peer)
        daemon._writer = cast("asyncio.StreamWriter", peer)
        daemon._started = True

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED, match="Unknown QMP"):
            await daemon.create_overlay(base, tmp_path / "overlay.qcow2")

    async def test_a_line_over_the_reader_limit_is_overlay_failed(self, tmp_path: Path) -> None:
        """A line longer than the stream reader's limit (64 KiB by default; 64 bytes here):
        OVERLAY_FAILED, not the raw ValueError of StreamReader.readline."""
        base = tmp_path / "base.qcow2"
        base.write_bytes(_qcow2_header(77 * 1024 * 1024))
        reader = asyncio.StreamReader(limit=64)
        reader.feed_data(b'{"return": "' + b"x" * 100 + b'"}\n')
        reader.feed_eof()  # the blockdev-del cleanup ends at once (a live daemon answers it)
        writer = MagicMock()
        writer.drain = AsyncMock()
        daemon = QemuStorageDaemon()
        daemon._reader = reader
        daemon._writer = writer
        daemon._started = True

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED, match="limit"):
            await daemon.create_overlay(base, tmp_path / "overlay.qcow2")

    async def test_a_job_still_running_after_its_concluded_event_is_waited_for(self, tmp_path: Path) -> None:
        """The JOB_STATUS_CHANGE "concluded" event comes, but query-jobs still lists the job as
        running: the wait polls again and dismisses the job once it concludes. It never takes
        the job as done on the event alone. A real daemon does not disagree on demand,
        so a peer in this process does."""
        base = tmp_path / "base.qcow2"
        base.write_bytes(_qcow2_header(77 * 1024 * 1024))
        peer = _ScriptedQmp([])
        scripted_write = peer.write
        first_query = True

        def write(data: bytes) -> None:
            nonlocal first_query
            scripted_write(data)
            if json.loads(data)["execute"] == "query-jobs" and first_query:
                first_query = False
                answer = json.loads(peer._lines.pop())  # the scripted "concluded" answer
                answer["return"] = [{**job, "status": "running"} for job in answer["return"]]
                peer._lines.append(json.dumps(answer).encode() + b"\n")

        peer.write = write  # type: ignore[method-assign]
        daemon = QemuStorageDaemon()
        daemon._reader = cast("asyncio.StreamReader", peer)
        daemon._writer = cast("asyncio.StreamWriter", peer)
        daemon._started = True

        await daemon.create_overlay(base, tmp_path / "overlay.qcow2")

        executed = [command["execute"] for command in peer.commands]
        assert executed[:3] == ["blockdev-create", "query-jobs", "query-jobs"]  # polled again
        assert executed.count("job-dismiss") == 2  # both jobs dismissed once concluded


# ============================================================================
# Integration Tests - Real daemon, real files
# ============================================================================


class TestQemuStorageDaemonIntegration:
    """Integration tests with real qemu-storage-daemon — no mocking.

    No hwaccel required: qemu-storage-daemon is a host-side tool that
    creates qcow2 overlays via QMP — it does not boot VMs or use KVM/HVF.
    """

    @pytest.mark.usefixtures("isolated_registry")
    async def test_start_stop_lifecycle(self) -> None:
        """start() runs one daemon process with its QMP socket, registered for force_kill_all;
        stop() ends the process, removes the socket and unregisters the process."""
        before = _daemon_sockets()
        daemon = QemuStorageDaemon()
        assert daemon.started is False
        await daemon.start()
        try:
            assert daemon.started is True
            started = {pid: path for pid, path in _daemon_sockets().items() if pid not in before}
            assert len(started) == 1
            (socket_path,) = started.values()
            assert socket_path.is_socket()
            assert process_registry.get_tracked_count() == 1
        finally:
            await daemon.stop()
        assert daemon.started is False
        assert not started.keys() & _daemon_sockets().keys()
        assert not socket_path.exists()
        assert process_registry.get_tracked_count() == 0

    async def test_double_start_is_idempotent(self) -> None:
        """Calling start() twice is safe - second call is no-op."""
        before = _daemon_sockets().keys()
        daemon = QemuStorageDaemon()
        await daemon.start()
        try:
            started = _daemon_sockets().keys() - before
            await daemon.start()  # Should be no-op
            assert _daemon_sockets().keys() - before == started  # the same one process
            assert len(started) == 1
            assert daemon.started is True
        finally:
            await daemon.stop()

    async def test_double_stop_is_idempotent(self) -> None:
        """Calling stop() twice is safe - second call is no-op."""
        daemon = QemuStorageDaemon()
        await daemon.start()
        await daemon.stop()
        assert daemon.started is False
        await daemon.stop()  # Should not raise
        assert daemon.started is False

    async def test_overlay_has_correct_backing_file(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Created overlay references correct base image as backing file."""
        overlay = tmp_path / "backing-test.qcow2"
        await started_daemon.create_overlay(base_image, overlay)

        info = await _image_info(overlay)

        assert info["backing-filename"] == str(base_image.absolute())
        assert info["backing-filename-format"] == "qcow2"

    async def test_overlay_has_qcow2_format(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Created overlay is valid qcow2 format, with the size of its base and the options of
        every overlay (qemu-img create -o lazy_refcounts=on,extended_l2=on,cluster_size=128k)."""
        overlay = tmp_path / "format-test.qcow2"
        await started_daemon.create_overlay(base_image, overlay)

        info = await _image_info(overlay)

        assert info["format"] == "qcow2"
        assert info["virtual-size"] == (await _image_info(base_image))["virtual-size"]
        assert info["cluster-size"] == 128 * 1024
        options = info["format-specific"]["data"]
        assert (options["lazy-refcounts"], options["extended-l2"]) == (True, True)

    # --- Job flow integration ---

    async def test_concurrent_overlays_all_complete(self, started_daemon: QemuStorageDaemon, tmp_path: Path) -> None:
        """20 overlays at once, each of its own base image (more than the pool's creation
        limit; each goes through QMP, whose commands take turns), all complete: each has the
        size of its base and that base as its backing file."""
        bases = [tmp_path / f"base-{i}.qcow2" for i in range(20)]
        for i, base in enumerate(bases):
            proc = await asyncio.create_subprocess_exec(
                "qemu-img", "create", "-q", "-f", "qcow2", str(base), f"{i + 1}M"
            )
            assert await proc.wait() == 0
        overlays = [tmp_path / f"many-{i}.qcow2" for i in range(20)]

        await asyncio.gather(*(started_daemon.create_overlay(b, o) for b, o in zip(bases, overlays, strict=True)))

        infos = [await _image_info(overlay) for overlay in overlays]
        assert [(info["virtual-size"], info["backing-filename"]) for info in infos] == [
            ((i + 1) * 1024 * 1024, str(base)) for i, base in enumerate(bases)
        ]

    async def test_overlay_after_failed_overlay_succeeds(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """A job that fails in the daemon (no write access to the overlay's folder) raises
        TransientError OVERLAY_FAILED, and the same daemon then creates a valid overlay."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)
        try:
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED, match="failed"):
                await started_daemon.create_overlay(base_image, readonly_dir / "overlay.qcow2")
        finally:
            readonly_dir.chmod(0o755)  # Restore for cleanup

        overlay = tmp_path / "success.qcow2"
        await started_daemon.create_overlay(base_image, overlay)
        assert overlay.exists()

    # --- Process registry integration ---

    @pytest.mark.usefixtures("isolated_registry")
    async def test_force_kill_all_reaches_the_daemon(self) -> None:
        """force_kill_all() (the second Ctrl+C) kills a started daemon; stop() then ends cleanly."""
        before = _daemon_sockets().keys()
        daemon = QemuStorageDaemon()
        await daemon.start()
        try:
            (pid,) = _daemon_sockets().keys() - before

            assert process_registry.force_kill_all() == 1

            async with asyncio.timeout(2):  # a SIGKILL ends it in ms; the timeout only bounds a failure
                while pid in _daemon_sockets():  # noqa: ASYNC110 - the end of a process has no event here
                    await asyncio.sleep(0.005)
        finally:
            await daemon.stop()
        assert daemon.started is False

    @pytest.mark.usefixtures("isolated_registry")
    async def test_process_unregistered_on_start_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A daemon that runs but sends no QMP greeting fails the start (TransientError
        OVERLAY_FAILED): the start ends the process and unregisters it."""
        binary = tmp_path / "qemu-storage-daemon"
        binary.write_text(_NO_GREETING_DAEMON.format(python=sys.executable))
        binary.chmod(0o755)
        monkeypatch.setattr("exec_sandbox.qemu_storage_daemon._BINARY", str(binary))
        before = _daemon_sockets().keys()
        daemon = QemuStorageDaemon()

        with pytest.raises(
            TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED, match="Invalid QMP greeting"
        ):
            await daemon.start()

        assert _daemon_sockets().keys() - before == set()
        assert process_registry.get_tracked_count() == 0
        assert not daemon.started

    # --- Orphan reaper integration ---

    @pytest.mark.parametrize("unreadable_process", [False, True], ids=["all-readable", "one-unreadable"])
    async def test_start_reaps_only_the_daemons_of_dead_owners(
        self, unreadable_process: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """start() removes each qsd-<namespace>-<pid>-<hex>.sock socket of this pid namespace in
        the temp folder whose owner pid does not run, and kills the qemu-storage-daemon that
        serves it. These stay: the socket of a running owner (this process; pid 1, of another
        user), one of another pid namespace (a container sharing /tmp: its pid means nothing
        here), one whose pid no kernel gives (no OverflowError out of start), other names,
        and a process that is not the daemon with a stale socket path in its arguments.

        one-unreadable: a process that cannot be read during the scan (on macOS psutil raises
        SystemError for a process that exits while its arguments are read) is skipped, and
        start() still succeeds. That exit race does not come on demand, so the first
        command line read raises it.

        The sockets sit in a temp folder of this test: a start in another worker reaps only its
        own temp folder."""
        tmpdir = Path(tempfile.mkdtemp(prefix="qsd-"))
        monkeypatch.setattr(tempfile, "tempdir", str(tmpdir))
        suffix = secrets.token_hex(8)  # parallel workers share the temp folder
        ns = _pid_namespace()
        dead = [tmpdir / f"qsd-{ns}-99999999-{suffix}{i}.sock" for i in range(2)]  # no pid is 99999999
        kept = [
            tmpdir / f"qsd-{ns}-{os.getpid()}-{suffix}.sock",
            tmpdir / f"qsd-{ns}-1-{suffix}.sock",
            tmpdir / f"qsd-{ns + 1}-99999999-{suffix}.sock",
            tmpdir / f"qsd-{ns}-{2**31}-{suffix}.sock",
            tmpdir / f"qsd-x{suffix}.sock",
            tmpdir / f"other-{suffix}.sock",
        ]
        for path in (dead[1], *kept):
            path.touch()
        orphan = await asyncio.create_subprocess_exec(
            "qemu-storage-daemon",
            "--chardev",
            f"socket,path={dead[0]},server=on,wait=off,id=qmp0",
            "--monitor",
            "chardev=qmp0",
            start_new_session=True,
        )
        for _ in range(200):  # the orphan serves its socket within a few ms
            if dead[0].exists():
                break
            await asyncio.sleep(0.01)
        assert dead[0].exists()
        bystander = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)", str(dead[1]), start_new_session=True
        )
        real_cmdline = psutil.Process.cmdline
        reads = 0

        def cmdline(proc: psutil.Process) -> list[str]:
            nonlocal reads
            reads += 1
            if unreadable_process and reads == 1:
                raise SystemError("<built-in function proc_cmdline> returned a result with an exception set")
            return real_cmdline(proc)

        daemon = QemuStorageDaemon()
        try:
            with patch.object(psutil.Process, "cmdline", cmdline):
                await daemon.start()

            assert await asyncio.wait_for(orphan.wait(), timeout=2) == -signal.SIGKILL
            assert bystander.returncode is None
            assert [path.exists() for path in dead] == [False, False]
            assert [path.exists() for path in kept] == [True] * len(kept)
        finally:
            await daemon.stop()
            for process in (orphan, bystander):
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            shutil.rmtree(tmpdir, ignore_errors=True)  # the test's sockets, and the daemon's own


class TestOverlayTemplates:
    """The first overlay of a base image version goes through QMP and is kept as its template;
    each later one is a clone of it, with the same bytes and mode as one made through QMP."""

    @staticmethod
    def _counted(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """The QMP commands that the daemons send from now on: a spy on _execute, the one way
        out of every command (no public path counts them; a slotted class: patched on it)."""
        sent: list[str] = []
        execute = QemuStorageDaemon._execute  # pyright: ignore[reportPrivateUsage]

        async def counted(
            self: QemuStorageDaemon, command: str, arguments: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            sent.append(command)
            return await execute(self, command, arguments)

        monkeypatch.setattr(QemuStorageDaemon, "_execute", counted)
        return sent

    async def test_later_overlays_are_clones_with_no_qmp(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first, later = tmp_path / "first.qcow2", [tmp_path / f"later-{i}.qcow2" for i in range(5)]
        await started_daemon.create_overlay(base_image, first)
        sent = self._counted(monkeypatch)

        await asyncio.gather(*(started_daemon.create_overlay(base_image, overlay) for overlay in later))

        assert sent == []
        for overlay in later:
            assert overlay.read_bytes() == first.read_bytes()
            assert overlay.stat().st_mode == first.stat().st_mode

    async def test_overlays_at_once_wait_for_the_first(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Overlays asked for at once (a burst at start) wait for the first one of their base
        image: one QMP creation, not one each."""
        sent = self._counted(monkeypatch)
        overlays = [tmp_path / f"burst-{i}.qcow2" for i in range(5)]

        await asyncio.gather(*(started_daemon.create_overlay(base_image, overlay) for overlay in overlays))

        assert sent.count("blockdev-add") == 1
        assert len({overlay.read_bytes() for overlay in overlays}) == 1

    async def test_a_template_gone_makes_the_overlay_through_qmp(
        self, short_tmp_dir: Path, base_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A template removed meanwhile (a temp folder cleaner): the overlay is made through QMP.
        The daemon's temp folder is one of this test: its template folder is the one there."""
        monkeypatch.setattr(tempfile, "tempdir", str(short_tmp_dir))
        daemon = QemuStorageDaemon()
        await daemon.start()
        try:
            first = tmp_path / "first.qcow2"
            await daemon.create_overlay(base_image, first)
            (template_dir,) = short_tmp_dir.glob("qsd-*.templates")
            for template in template_dir.iterdir():
                template.unlink()
            sent = self._counted(monkeypatch)

            await daemon.create_overlay(base_image, tmp_path / "second.qcow2")
        finally:
            await daemon.stop()

        assert "blockdev-add" in sent
        assert (tmp_path / "second.qcow2").read_bytes() == first.read_bytes()

    async def test_stop_removes_the_template_folder(
        self, short_tmp_dir: Path, base_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The template folder in the temp folder holds the template; stop() removes it."""
        monkeypatch.setattr(tempfile, "tempdir", str(short_tmp_dir))
        daemon = QemuStorageDaemon()
        await daemon.start()
        try:
            await daemon.create_overlay(base_image, tmp_path / "overlay.qcow2")
            (template_dir,) = short_tmp_dir.glob("qsd-*.templates")
            assert len(list(template_dir.iterdir())) == 1
        finally:
            await daemon.stop()
        assert list(short_tmp_dir.glob("qsd-*.templates")) == []

    @staticmethod
    async def _template_clone_held(
        daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[threading.Event, asyncio.Task[None], Path]:
        """Start the daemon and a first overlay whose template clone waits on the returned
        event. Returns (event, the creating task, the template folder), once the clone runs."""
        release = threading.Event()
        real_clone = qemu_storage_daemon._clone_overlay  # pyright: ignore[reportPrivateUsage]

        def held_clone(source: Path, target: Path) -> bool:
            if target.parent.name.endswith(".templates"):
                release.wait(10)
            return real_clone(source, target)

        monkeypatch.setattr(qemu_storage_daemon, "_clone_overlay", held_clone)
        await daemon.start()
        template_dir = daemon._template_dir  # pyright: ignore[reportPrivateUsage]
        assert template_dir is not None
        creating = asyncio.create_task(daemon.create_overlay(base_image, tmp_path / "overlay.qcow2"))
        for _ in range(500):  # until the template clone runs
            if daemon._template_writes:  # pyright: ignore[reportPrivateUsage]
                break
            await asyncio.sleep(0.01)
        assert daemon._template_writes, "the template clone never started"  # pyright: ignore[reportPrivateUsage]
        return release, creating, template_dir

    async def test_stop_waits_for_a_template_clone_in_flight(
        self, base_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """stop() during the clone that makes a template: it waits for the clone, then removes
        the folder, so the clone cannot fill it again after its removal."""
        daemon = QemuStorageDaemon()
        release, creating, template_dir = await self._template_clone_held(daemon, base_image, tmp_path, monkeypatch)
        stopping = asyncio.create_task(daemon.stop())
        for _ in range(500):  # until the daemon process is gone: only the clone then holds stop()
            if daemon._process is None:  # pyright: ignore[reportPrivateUsage]
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)

        assert not stopping.done()  # it waits for the clone
        release.set()
        await stopping
        await creating
        assert not template_dir.exists()

    async def test_a_cancelled_stop_cancels_no_clone_and_still_removes_the_folder(
        self, base_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """stop() cancelled while a template clone runs: the create_overlay that awaits the
        clone is not cancelled (its overlay is made), and the folder still goes once the clone
        ends."""
        daemon = QemuStorageDaemon()
        release, creating, template_dir = await self._template_clone_held(daemon, base_image, tmp_path, monkeypatch)
        stopping = asyncio.create_task(daemon.stop())
        await asyncio.sleep(0.1)
        stopping.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stopping

        release.set()
        await creating  # not cancelled
        assert (tmp_path / "overlay.qcow2").exists()
        await asyncio.gather(*qemu_storage_daemon._cleanup_tasks)  # pyright: ignore[reportPrivateUsage]
        assert not template_dir.exists()

    async def test_no_template_clone_starts_once_stop_began(
        self, base_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A first overlay still in its QMP step when stop() runs: once that step ends, it
        starts no template clone into the folder that stop() removes. The QMP step is
        a stand-in that waits on an event."""
        qmp_entered, qmp_done = asyncio.Event(), asyncio.Event()

        async def slow_qmp(_self: QemuStorageDaemon, _base: Path, overlay_path: Path) -> None:
            qmp_entered.set()
            await qmp_done.wait()
            overlay_path.write_bytes(b"overlay")

        clones: list[Path] = []
        monkeypatch.setattr(QemuStorageDaemon, "_create_with_qmp", slow_qmp)
        monkeypatch.setattr(qemu_storage_daemon, "_clone_overlay", lambda _source, target: clones.append(target))
        daemon = QemuStorageDaemon()
        await daemon.start()
        creating = asyncio.create_task(daemon.create_overlay(base_image, tmp_path / "overlay.qcow2"))
        await asyncio.wait_for(qmp_entered.wait(), 5)  # in its QMP step, the folder read

        await daemon.stop()
        qmp_done.set()
        await creating

        assert clones == []

    async def test_a_stop_cancelled_while_a_start_runs_still_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stop() waits for a start() and is cancelled meanwhile: the stop still runs once the
        start ends (a retained task), so no daemon or folder stays."""
        entered, release = threading.Event(), threading.Event()
        real_reap = QemuStorageDaemon._reap_stale_daemons  # pyright: ignore[reportPrivateUsage]

        def held_reap(*args: Any) -> int:
            entered.set()
            release.wait(10)
            return real_reap(*args)

        monkeypatch.setattr(QemuStorageDaemon, "_reap_stale_daemons", staticmethod(held_reap))
        daemon = QemuStorageDaemon()
        starting = asyncio.create_task(daemon.start())
        assert await asyncio.to_thread(entered.wait, 10)
        stopping = asyncio.create_task(daemon.stop())
        await asyncio.sleep(0.05)
        stopping.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stopping

        release.set()
        await starting
        await asyncio.gather(*qemu_storage_daemon._cleanup_tasks)  # pyright: ignore[reportPrivateUsage]
        assert (daemon.started, daemon._process, daemon._template_dir) == (False, None, None)  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.parametrize("held", ["reaper", "template-mkdir"])
    async def test_a_stop_during_start_stops_what_it_made(self, monkeypatch: pytest.MonkeyPatch, held: str) -> None:
        """stop() while start() still runs (its stale-daemon reaper, or the mkdir of its
        template folder): stop() waits for start(), then stops what it made: no daemon runs
        and no template folder stays. The held step waits on an event. Other daemons of this
        worker (from earlier tests) can still hold their folders: only new ones count."""
        entered, release = threading.Event(), threading.Event()
        own_folders = f"qsd-*-{os.getpid()}-*.templates"
        before = set(Path(tempfile.gettempdir()).glob(own_folders))

        def hold() -> None:
            entered.set()
            release.wait(10)

        if held == "reaper":
            real_reap = QemuStorageDaemon._reap_stale_daemons  # pyright: ignore[reportPrivateUsage]

            def held_reap(*args: Any) -> int:
                hold()
                return real_reap(*args)

            monkeypatch.setattr(QemuStorageDaemon, "_reap_stale_daemons", staticmethod(held_reap))
        else:
            real_mkdir = Path.mkdir

            def held_mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
                if path.name.endswith(".templates"):
                    hold()
                real_mkdir(path, *args, **kwargs)

            monkeypatch.setattr(Path, "mkdir", held_mkdir)
        daemon = QemuStorageDaemon()
        starting = asyncio.create_task(daemon.start())
        assert await asyncio.to_thread(entered.wait, 10)
        stopping = asyncio.create_task(daemon.stop())
        await asyncio.sleep(0.1)

        assert not stopping.done()  # it waits for start()
        release.set()
        await starting
        await stopping
        await asyncio.gather(*qemu_storage_daemon._cleanup_tasks)  # pyright: ignore[reportPrivateUsage]
        assert (daemon.started, daemon._process, daemon._template_dir) == (False, None, None)  # pyright: ignore[reportPrivateUsage]
        assert not set(Path(tempfile.gettempdir()).glob(own_folders)) - before

    async def test_start_removes_only_the_template_folders_of_dead_owners(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A template folder of a dead owner of this pid namespace goes; those of a running
        owner, of another namespace, and other names stay."""
        tmpdir = Path(tempfile.mkdtemp(prefix="qsd-"))
        monkeypatch.setattr(tempfile, "tempdir", str(tmpdir))
        ns = _pid_namespace()
        dead = tmpdir / f"qsd-{ns}-99999999-abc_12.templates"  # no pid is 99999999
        kept = [
            tmpdir / f"qsd-{ns}-{os.getpid()}-abc_12.templates",
            tmpdir / f"qsd-{ns + 1}-99999999-abc_12.templates",
            tmpdir / f"qsd-{ns}-99999999-abc_12.other",
        ]
        for folder in (dead, *kept):
            folder.mkdir()
            (folder / "t.qcow2").touch()
        daemon = QemuStorageDaemon()
        try:
            await daemon.start()
            assert not dead.exists()
            assert all(folder.exists() for folder in kept)
        finally:
            await daemon.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# OverlayPool Integration with Daemon
# ============================================================================


class TestOverlayPoolWithDaemon:
    """Integration tests for OverlayPool using QemuStorageDaemon.

    No hwaccel required: uses qemu-storage-daemon (host-side QMP tool)
    for overlay creation — no VMs are booted.
    """

    async def test_pool_uses_daemon_when_enabled(self, base_image: Path, tmp_path: Path) -> None:
        """OverlayPool uses daemon for overlay creation."""
        pool = OverlayPool(pool_size=2, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        try:
            assert pool.daemon_enabled is True

            # The pool fills in the background after start()
            queue = pool._pools[str(base_image.resolve())]
            async with asyncio.timeout(5):
                while queue.empty():  # noqa: ASYNC110 - the pool has no fill event to await
                    await asyncio.sleep(0.005)

            # Acquire overlay should work
            target = tmp_path / "acquired.qcow2"
            result = await pool.acquire(base_image, target)

            assert result is True  # From pool
            assert target.exists()
        finally:
            await pool.stop()

    async def test_daemon_stopped_on_pool_shutdown(self, base_image: Path, tmp_path: Path) -> None:
        """Daemon is stopped when pool shuts down: its process ends."""
        before = _daemon_sockets().keys()
        pool = OverlayPool(pool_size=2, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        assert pool.daemon_enabled is True
        started = _daemon_sockets().keys() - before
        assert len(started) == 1

        await pool.stop()

        assert pool.daemon_enabled is False
        assert not started & _daemon_sockets().keys()


# ============================================================================
# Error Handling Tests - Real daemon, verify error paths
# ============================================================================


class TestQemuStorageDaemonErrors:
    """Test error handling with real daemon — no mocking.

    No hwaccel required: qemu-storage-daemon is a host-side tool that
    creates qcow2 overlays via QMP — it does not boot VMs or use KVM/HVF.
    """

    async def test_create_overlay_nonexistent_base_image_fails(
        self, started_daemon: QemuStorageDaemon, tmp_path: Path
    ) -> None:
        """create_overlay with nonexistent base fails fast with clear error."""
        nonexistent = tmp_path / "does-not-exist.qcow2"
        overlay = tmp_path / "overlay.qcow2"

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED, match="Failed to stat"):
            await started_daemon.create_overlay(nonexistent, overlay)

        # Overlay should not exist
        assert not overlay.exists()

    @pytest.mark.parametrize(
        "content",
        [
            b"not a qcow2 file",
            b"",
            b"QFI\xfb" + (1).to_bytes(4, "big") + bytes(24),  # qcow v1: same magic, QEMU refuses it
            b"QFI\xfb" + (3).to_bytes(4, "big"),  # the magic and a version, then nothing
        ],
        ids=["raw", "empty", "qcow-v1", "short-header"],
    )
    async def test_create_overlay_invalid_base_image_fails(
        self, started_daemon: QemuStorageDaemon, tmp_path: Path, content: bytes
    ) -> None:
        """create_overlay with invalid qcow2 base fails fast with clear error."""
        fake_base = tmp_path / "fake-base.qcow2"
        fake_base.write_bytes(content)

        overlay = tmp_path / "overlay.qcow2"

        with pytest.raises(
            TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED, match="not qcow2 format"
        ):
            await started_daemon.create_overlay(fake_base, overlay)

    @pytest.mark.parametrize("folder", ["тест-日本語-🔥", "path with spaces"], ids=["unicode", "spaces"])
    async def test_create_overlay_with_unusual_path(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path, folder: str
    ) -> None:
        """An overlay path with unicode characters or spaces (QMP takes it as a JSON string):
        the overlay is made there, backed by its base."""
        (tmp_path / folder).mkdir()
        overlay = tmp_path / folder / f"overlay {folder}.qcow2"

        await started_daemon.create_overlay(base_image, overlay)

        assert (await _image_info(overlay))["backing-filename"] == str(base_image)

    async def test_create_overlay_symlink_base_image(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """A base image given as a symlink: the overlay keeps the symlink path as its backing
        file, and QEMU opens the chain through it."""
        symlink = tmp_path / "base-link.qcow2"
        symlink.symlink_to(base_image)

        overlay = tmp_path / "overlay.qcow2"
        await started_daemon.create_overlay(symlink, overlay)

        info = await _image_info(overlay)
        assert (info["backing-filename"], info["virtual-size"]) == (
            str(symlink),
            (await _image_info(base_image))["virtual-size"],
        )


_KILLED_OWNER = """
import asyncio, sys
from exec_sandbox.qemu_storage_daemon import QemuStorageDaemon

async def main() -> None:
    daemon = QemuStorageDaemon()
    await daemon.start()
    print(daemon._process.pid, flush=True)
    await asyncio.Event().wait()  # until the test kills this process (never a read of a shared stdin)

asyncio.run(main())
"""
"""A process with a started daemon, killed by the test (SIGKILL: no cleanup runs)."""


@skip_unless_linux
class TestDaemonEndsWithItsProcess:
    """The daemon runs as this user and is not a VM: no VM cgroup and no watcher reach it.
    Without PR_SET_PDEATHSIG, the daemon of a killed process runs on (its parent is then
    init) until a start-time reaper with the same temp folder finds its socket, if ever."""

    async def test_daemon_of_a_killed_process_ends_at_once(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="qsd-"))  # the daemon's socket: private to this test
        owner = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _KILLED_OWNER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env={**os.environ, "TMPDIR": str(tmp)},
        )
        pidfd: int | None = None
        try:
            assert owner.stdout is not None
            pidfd = os.pidfd_open(int(await owner.stdout.readline()))  # readable once the daemon ends
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            assert poller.poll(0) == []  # alive while its process lives
            owner.kill()
            assert await owner.wait() == -9
            start = asyncio.get_running_loop().time()
            assert await asyncio.to_thread(poller.poll, 2000)  # ends within ms; the timeout only bounds a failure
            assert asyncio.get_running_loop().time() - start < 1.0
        finally:
            if owner.returncode is None:
                owner.kill()
                await owner.wait()
            if pidfd is not None:
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)  # a failed test leaves no daemon
                os.close(pidfd)
            shutil.rmtree(tmp, ignore_errors=True)

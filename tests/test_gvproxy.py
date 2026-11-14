"""start_gvproxy: who owns its pre-bound socket and its process when the start fails, its
failures by code, its log task, and the level of gvproxy's stderr lines. Host only: a shell
stands in for gvproxy-wrapper (it prints the readiness line, then sleeps)."""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import re
import signal
import socket
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox import subprocess_utils
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.gvproxy import start_gvproxy
from exec_sandbox.platform_utils import HostOS, detect_host_os
from exec_sandbox.process_registry import unregister_process
from exec_sandbox.resource_cleanup import cleanup_process
from tests.test_subprocess_utils import write_foreign_elf

if TYPE_CHECKING:
    from exec_sandbox.platform_utils import ProcessWrapper


def _workdir(socket_path: Path) -> MagicMock:
    workdir = MagicMock()
    workdir.gvproxy_socket = socket_path
    return workdir


def _stand_in(tmp_path: Path) -> Path:
    """A gvproxy-wrapper stand-in: prints the readiness line, then sleeps."""
    binary = tmp_path / "gvproxy-wrapper-test"
    binary.write_text('#!/bin/sh\necho "Listening on QEMU socket"\nexec sleep 30\n')
    binary.chmod(0o755)
    return binary


def _no_qemu_vm_access(stack: ExitStack, tmp_path: Path, sock: socket.socket) -> None:
    """QEMU runs as qemu-vm and setfacl fails: start_gvproxy fails after the spawn."""
    stack.enter_context(patch("exec_sandbox.gvproxy.create_unix_socket", return_value=sock))
    stack.enter_context(
        patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=_stand_in(tmp_path))
    )
    stack.enter_context(patch("exec_sandbox.gvproxy.probe_qemu_as_qemu_vm", new_callable=AsyncMock, return_value=True))
    stack.enter_context(patch("exec_sandbox.gvproxy.grant_qemu_vm_access", new_callable=AsyncMock, return_value=False))
    stack.enter_context(
        pytest.raises(
            ConfigError,
            check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY,
            match=r"^Could not give qemu-vm access to the gvproxy socket",
        )
    )


class TestStartGvproxyOwnership:
    async def test_a_cancel_during_the_binary_lookup_closes_the_socket(self, tmp_path: Path) -> None:
        """The pre-bound socket is open while the binary is looked up (a worker-thread job):
        a cancel there closes it, as every failure after its bind does."""
        sock = socket.socket(socket.AF_UNIX)
        lookup = asyncio.Event()

        async def blocked_lookup(*, override: Path | None) -> Path | None:
            lookup.set()
            await asyncio.Event().wait()
            return None

        with (
            patch("exec_sandbox.gvproxy.create_unix_socket", return_value=sock),
            patch("exec_sandbox.gvproxy.get_gvproxy_path", blocked_lookup),
        ):
            start = asyncio.create_task(
                start_gvproxy(
                    "gvproxy-lookup",
                    allowed_domains=None,
                    language="python",
                    workdir=_workdir(tmp_path / "g"),
                    images_dir=tmp_path,
                    expose_ports=None,
                    block_outbound=False,
                    on_process_started=None,
                )
            )
            await lookup.wait()
            start.cancel()
            with pytest.raises(asyncio.CancelledError):
                await start
        assert sock.fileno() == -1

    async def test_after_the_hand_off_a_failure_leaves_the_process_to_its_owner(self, tmp_path: Path) -> None:
        """on_process_started got the process and its drain task: the start fails after it
        (setfacl), and the process still runs, its drain too. The owner stops them (the VM
        manager's cleanup of the launch), not start_gvproxy as well."""
        sock = socket.socket(socket.AF_UNIX)
        handed: list[tuple[ProcessWrapper, asyncio.Task[None] | None]] = []
        with ExitStack() as stack:
            _no_qemu_vm_access(stack, tmp_path, sock)
            await start_gvproxy(
                "gvproxy-owner",
                allowed_domains=None,
                language="python",
                workdir=_workdir(tmp_path / "g"),
                images_dir=tmp_path,
                expose_ports=None,
                block_outbound=False,
                on_process_started=lambda proc, drain: handed.append((proc, drain)),
            )
        (proc, first_drain), (same, drain) = handed
        try:
            assert (same, first_drain) == (proc, None)
            assert proc.returncode is None
            assert drain is not None
            assert not drain.done()
            assert sock.fileno() == -1
        finally:
            await cleanup_process(proc, "gvproxy", "test", term_timeout=3.0, kill_timeout=2.0, kill_first=False)
            unregister_process(proc)
            if drain is not None:
                await asyncio.wait((drain,))

    async def test_without_an_owner_a_failure_stops_the_process_and_its_drain(self, tmp_path: Path) -> None:
        """No on_process_started: start_gvproxy owns the process until it returns. A failure
        after the spawn (setfacl) stops it (SIGTERM, reaped) and waits for its drain task."""
        sock = socket.socket(socket.AF_UNIX)
        spawned: list[ProcessWrapper] = []
        real_start = subprocess_utils.start_managed_process

        async def recording_start(*args: Any, **kwargs: Any) -> ProcessWrapper:
            proc = await real_start(*args, **kwargs)
            spawned.append(proc)
            return proc

        with ExitStack() as stack:
            stack.enter_context(patch("exec_sandbox.gvproxy.start_managed_process", recording_start))
            _no_qemu_vm_access(stack, tmp_path, sock)
            await start_gvproxy(
                "gvproxy-no-owner",
                allowed_domains=None,
                language="python",
                workdir=_workdir(tmp_path / "g"),
                images_dir=tmp_path,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )
        (proc,) = spawned
        unregister_process(proc)
        assert proc.returncode == -signal.SIGTERM
        assert [
            task for task in asyncio.all_tasks() if task.get_name() == "gvproxy-wrapper-drain-gvproxy-no-owner"
        ] == []
        assert sock.fileno() == -1


class TestGvproxyStderrLevel:
    """gvisor-tap-vsock logs the close of QEMU's network socket at level=error on each VM end."""

    @pytest.mark.parametrize(
        ("line", "qemu_connected", "level"),
        [
            ('time="x" level=error msg="cannot read size from socket: EOF"', True, logging.DEBUG),
            ('time="x" level=error msg="cannot read size from socket: EOF"', False, logging.ERROR),
            ('time="x" level=error msg="cannot receive packets: broken pipe"', True, logging.ERROR),
        ],
        ids=["close-after-connect", "close-before-connect", "other-error"],
    )
    def test_level(self, line: str, qemu_connected: bool, level: int) -> None:
        """The rule on the function itself: through start_gvproxy, the "QEMU connected" line
        (stdout) and the error line (stderr) come on two pipes that two readers drain, so the
        order in which they arrive is not the order the process wrote them."""
        from exec_sandbox.gvproxy import _stderr_level

        assert _stderr_level(line, qemu_connected=qemu_connected) == level


class TestStartGvproxy:
    """start_gvproxy failures (by code), and its log task handed to the process: cleanup
    then waits for the exit without reading the pipes that task reads. Host only: a shell
    stands in for gvproxy-wrapper (it prints the readiness line, then waits)."""

    @staticmethod
    def _workdir(socket_path: Path) -> MagicMock:
        workdir = MagicMock()
        workdir.gvproxy_socket = socket_path
        return workdir

    async def test_socket_failure_is_gvproxy_failed(self, tmp_path: Path) -> None:
        """A socket path above the AF_UNIX limit (104 bytes on macOS, 108 on Linux)."""
        from exec_sandbox.gvproxy import start_gvproxy

        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.GVPROXY_FAILED,
            match=r"^Failed to create gvproxy socket: ",
        ):
            await start_gvproxy(
                "gvproxy-socket",
                allowed_domains=None,
                language="python",
                workdir=self._workdir(tmp_path / ("s" * 120)),
                images_dir=tmp_path,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )

    async def test_missing_binary_is_host_dependency(self, tmp_path: Path) -> None:
        from exec_sandbox.gvproxy import start_gvproxy

        sock = socket.socket(socket.AF_UNIX)
        with (
            patch("exec_sandbox.gvproxy.create_unix_socket", return_value=sock),
            patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=None),
        ):
            with pytest.raises(
                ConfigError,
                check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY,
                match=r"^gvproxy-wrapper binary not found",
            ):
                await start_gvproxy(
                    "gvproxy-binary",
                    allowed_domains=None,
                    language="python",
                    workdir=self._workdir(tmp_path / "g"),
                    images_dir=tmp_path,
                    expose_ports=None,
                    block_outbound=False,
                    on_process_started=None,
                )
        assert sock.fileno() == -1  # the pre-bound socket is closed

    @pytest.mark.parametrize("err", [errno.ENOENT, errno.EACCES], ids=["missing", "no-exec-bit"])
    async def test_binary_that_cannot_run_is_host_dependency(self, tmp_path: Path, err: int) -> None:
        """A gvproxy-wrapper that is gone or that this user cannot run fails on every retry:
        FIX_CONFIG, never GVPROXY_FAILED (RETRY)."""
        from exec_sandbox.gvproxy import start_gvproxy

        binary = tmp_path / "gvproxy-wrapper-test"
        if err == errno.EACCES:
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o644)
        sock = socket.socket(socket.AF_UNIX)
        with (
            patch("exec_sandbox.gvproxy.create_unix_socket", return_value=sock),
            patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=binary),
        ):
            with pytest.raises(
                ConfigError,
                check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY,
                match=r"^Failed to start .+/gvproxy-wrapper-test: ",
            ) as exc_info:
                await start_gvproxy(
                    "gvproxy-spawn",
                    allowed_domains=None,
                    language="python",
                    workdir=self._workdir(tmp_path / "g"),
                    images_dir=tmp_path,
                    expose_ports=None,
                    block_outbound=False,
                    on_process_started=None,
                )
        assert exc_info.value.context["errno"] == err
        assert exc_info.value.context["binary"] == str(binary)
        assert sock.fileno() == -1

    @pytest.mark.parametrize("kind", ["text-no-shebang", "other-arch"])
    async def test_gvproxy_that_is_no_program_here_is_host_dependency(self, tmp_path: Path, kind: str) -> None:
        """A gvproxy-wrapper that this host cannot exec: FIX_CONFIG at once, not
        GVPROXY_FAILED (RETRY) after the 5 s readiness wait. A text file never runs as a
        shell script (ENOEXEC before the spawn). Another arch: on Linux the launch shell
        execs it and ends with 126; elsewhere Popen execs it (ENOEXEC)."""
        from exec_sandbox.gvproxy import start_gvproxy

        binary = tmp_path / "gvproxy-wrapper-test"
        if kind == "other-arch":
            write_foreign_elf(binary)
        else:
            binary.write_text(f"touch {tmp_path / 'ran'}\n")  # would run as a shell script
            binary.chmod(0o755)
        sock = socket.socket(socket.AF_UNIX)
        start = asyncio.get_running_loop().time()
        with (
            patch("exec_sandbox.gvproxy.create_unix_socket", return_value=sock),
            patch(
                "exec_sandbox.gvproxy.grant_qemu_vm_access", new_callable=AsyncMock, return_value=True
            ),  # a fake socket: no file for the ACL
            patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=binary),
            pytest.raises(
                ConfigError,
                check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY,
                match=r"^Failed to start .+/gvproxy-wrapper-test: ",
            ) as exc_info,
        ):
            _, _, wait_ready = await start_gvproxy(
                "gvproxy-format",
                allowed_domains=None,
                language="python",
                workdir=self._workdir(tmp_path / "g"),
                images_dir=tmp_path,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )
            await wait_ready()
        assert asyncio.get_running_loop().time() - start < 1.0
        context = exc_info.value.context
        assert context["binary"] == str(binary)
        if kind == "other-arch" and detect_host_os() == HostOS.LINUX:
            assert context["exit_code"] == 126
            assert str(exc_info.value).endswith("Exec format error")  # the launch shell's line
        else:
            assert context["errno"] == errno.ENOEXEC
        assert not (tmp_path / "ran").exists()
        assert sock.fileno() == -1

    @pytest.mark.parametrize("drain", ["at-once", "after-the-end"])
    async def test_end_before_ready_quotes_its_last_output(self, tmp_path: Path, drain: str) -> None:
        """Go's log, log.Fatal too, writes to stdout (main.go), which the log has at DEBUG: an
        end before ready raises GVPROXY_FAILED at once with its last lines in the message and
        the context. "after-the-end": the drain reads the lines only once the process
        ended (the wait waits for them)."""
        from exec_sandbox import gvproxy, subprocess_utils
        from exec_sandbox.gvproxy import start_gvproxy

        fatal = "2026/11/13 03:00:00 Error parsing outbound allow patterns: invalid character"
        binary = tmp_path / "gvproxy-wrapper-test"
        binary.write_text(f'#!/bin/sh\necho "starting"\necho "{fatal}"\nexit 1\n')
        binary.chmod(0o755)

        def late_drain(process: Any, **kwargs: Any) -> asyncio.Task[None]:
            async def after_the_end() -> None:
                await process.wait()  # every line waits in the pipe buffers now
                for _ in range(20):  # let the start go on first (no wait: turns of the loop)
                    await asyncio.sleep(0)
                await subprocess_utils.drain_subprocess_output(
                    process, stdout_handler=kwargs["stdout_handler"], stderr_handler=kwargs["stderr_handler"]
                )

            process.drain_task = asyncio.create_task(after_the_end())
            return process.drain_task

        sock = socket.socket(socket.AF_UNIX)
        with ExitStack() as stack:
            stack.enter_context(patch("exec_sandbox.gvproxy.create_unix_socket", return_value=sock))
            stack.enter_context(
                patch("exec_sandbox.gvproxy.grant_qemu_vm_access", new_callable=AsyncMock, return_value=True)
            )  # a fake socket: no file for the ACL
            stack.enter_context(
                patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=binary)
            )
            if drain == "after-the-end":
                stack.enter_context(patch.object(gvproxy, "start_drain", late_drain))
            exc_info = stack.enter_context(
                pytest.raises(
                    TransientError,
                    check=lambda e: e.code is ErrorCode.GVPROXY_FAILED,
                    match=rf"^gvproxy-wrapper exited with code 1 before it was ready: {re.escape(fatal)}$",
                )
            )
            _, _, wait_ready = await start_gvproxy(
                "gvproxy-fatal",
                allowed_domains=None,
                language="python",
                workdir=self._workdir(tmp_path / "g"),
                images_dir=tmp_path,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )
            await wait_ready()
        assert exc_info.value.context["output"] == ["starting", fatal]
        assert exc_info.value.context["exit_code"] == 1

    @pytest.mark.parametrize(
        ("lines", "cause"),
        [
            (
                ["SIGSEGV: segmentation violation", "PC=0x0 m=0 sigcode=1", "r9     0x0"],
                "SIGSEGV: segmentation violation",
            ),
            (
                ["SIGSEGV: segmentation violation", "runtime: g 1: unexpected return pc for main.f", "r9     0x0"],
                "SIGSEGV: segmentation violation",
            ),
            (
                ["fatal error: unexpected signal", "[signal SIGSEGV: segmentation violation]"],
                "fatal error: unexpected signal",
            ),
        ],
        ids=["signal-only", "signal-then-traceback-runtime-line", "fatal-error-then-signal"],
    )
    async def test_end_before_ready_quotes_the_signal_of_a_go_crash(
        self, tmp_path: Path, lines: list[str], cause: str
    ) -> None:
        """A crash from a signal only has no panic or fatal error line: the SIGxxx: line is its
        cause, and the `runtime:` lines of its traceback do not replace it. A fatal error comes
        before its signal line (Go's throw)."""
        from exec_sandbox.gvproxy import start_gvproxy

        binary = tmp_path / "gvproxy-wrapper-test"
        echoes = "".join(f'echo "{line}" >&2\n' for line in lines)
        binary.write_text(f"#!/bin/sh\n{echoes}exit 2\n")
        binary.chmod(0o755)
        sock = socket.socket(socket.AF_UNIX)
        with ExitStack() as stack:
            stack.enter_context(patch("exec_sandbox.gvproxy.create_unix_socket", return_value=sock))
            stack.enter_context(
                patch("exec_sandbox.gvproxy.grant_qemu_vm_access", new_callable=AsyncMock, return_value=True)
            )  # a fake socket: no file for the ACL
            stack.enter_context(
                patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=binary)
            )
            exc_info = stack.enter_context(
                pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GVPROXY_FAILED)
            )
            _, _, wait_ready = await start_gvproxy(
                "gvproxy-signal",
                allowed_domains=None,
                language="python",
                workdir=self._workdir(tmp_path / "g"),
                images_dir=tmp_path,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )
            await wait_ready()
        assert exc_info.value.context["crash"] == cause
        assert str(exc_info.value).endswith(cause)

    async def test_end_before_ready_quotes_the_cause_of_a_go_crash(self, tmp_path: Path) -> None:
        """A Go runtime crash writes its cause first on stderr (here the one that `ulimit -u 5`
        gives), then stacks and registers: the message quotes the cause."""
        from exec_sandbox.gvproxy import start_gvproxy

        cause = "runtime/cgo: pthread_create failed: Resource temporarily unavailable"
        binary = tmp_path / "gvproxy-wrapper-test"
        registers = "".join(f'echo "r{i}     0x0" >&2\n' for i in range(10))
        binary.write_text(f'#!/bin/sh\necho "{cause}" >&2\necho "SIGABRT: abort" >&2\n{registers}exit 2\n')
        binary.chmod(0o755)
        sock = socket.socket(socket.AF_UNIX)
        with ExitStack() as stack:
            stack.enter_context(patch("exec_sandbox.gvproxy.create_unix_socket", return_value=sock))
            stack.enter_context(
                patch("exec_sandbox.gvproxy.grant_qemu_vm_access", new_callable=AsyncMock, return_value=True)
            )  # a fake socket: no file for the ACL
            stack.enter_context(
                patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=binary)
            )
            exc_info = stack.enter_context(
                pytest.raises(
                    TransientError,
                    check=lambda e: e.code is ErrorCode.GVPROXY_FAILED,
                    match=rf"^gvproxy-wrapper exited with code 2 before it was ready: {re.escape(cause)}$",
                )
            )
            _, _, wait_ready = await start_gvproxy(
                "gvproxy-crash",
                allowed_domains=None,
                language="python",
                workdir=self._workdir(tmp_path / "g"),
                images_dir=tmp_path,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )
            await wait_ready()
        assert exc_info.value.context["crash"] == cause
        assert exc_info.value.context["output"][-1] == "r9     0x0"  # the tail is still in the context

    async def test_fork_limit_is_gvproxy_failed(self, tmp_path: Path) -> None:
        """A fork that hits a process limit can pass later: GVPROXY_FAILED (RETRY)."""
        from exec_sandbox.gvproxy import start_gvproxy

        sock = socket.socket(socket.AF_UNIX)
        with (
            patch("exec_sandbox.gvproxy.create_unix_socket", return_value=sock),
            patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=Path("/bin/sh")),
            patch(
                "exec_sandbox.gvproxy.start_managed_process",
                new_callable=AsyncMock,
                side_effect=OSError(errno.EAGAIN, os.strerror(errno.EAGAIN)),
            ),
        ):
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GVPROXY_FAILED) as exc_info:
                await start_gvproxy(
                    "gvproxy-spawn",
                    allowed_domains=None,
                    language="python",
                    workdir=self._workdir(tmp_path / "g"),
                    images_dir=tmp_path,
                    expose_ports=None,
                    block_outbound=False,
                    on_process_started=None,
                )
        assert exc_info.value.context["errno"] == errno.EAGAIN
        assert sock.fileno() == -1

    async def test_gvproxy_ends_with_this_process_with_no_preexec(self, tmp_path: Path) -> None:
        """gvproxy-wrapper has no VM cgroup to end it when cgroups are off: it ends with
        this process (setpriv, with no Python preexec_fn). Its listener fd and arguments
        pass through. The OS spawn is faked (it fails with EAGAIN): the test reads what the
        launch gives it. The real end: TestEndWithThisProcess."""
        from exec_sandbox.gvproxy import start_gvproxy

        sock = socket.socket(socket.AF_UNIX)
        listener = sock.fileno()
        spawn = AsyncMock(side_effect=OSError(errno.EAGAIN, os.strerror(errno.EAGAIN)))
        with (
            patch("exec_sandbox.gvproxy.create_unix_socket", return_value=sock),
            patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=Path("/bin/sh")),
            patch("exec_sandbox.subprocess_utils.detect_host_os", return_value=HostOS.LINUX),
            patch("exec_sandbox.subprocess_utils.probe_setpriv_pdeathsig", new_callable=AsyncMock, return_value=True),
            patch("exec_sandbox.subprocess_utils.asyncio.create_subprocess_exec", spawn),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GVPROXY_FAILED),
        ):
            await start_gvproxy(
                "gvproxy-pdeath",
                allowed_domains=None,
                language="python",
                workdir=self._workdir(tmp_path / "g"),
                images_dir=tmp_path,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )
        cmd, kwargs = list(spawn.call_args.args), spawn.call_args.kwargs
        assert cmd[:4] == ["setpriv", "--pdeathsig", "KILL", "--"]
        assert kwargs["preexec_fn"] is None
        assert kwargs["pass_fds"] == (listener,)
        listen = cmd.index("-listen-fd")
        assert cmd[listen - 1 : listen + 2] == ["/bin/sh", "-listen-fd", str(listener)]

    async def test_the_start_returns_before_the_ready_line(self, tmp_path: Path) -> None:
        """start_gvproxy returns once gvproxy-wrapper runs, so QEMU can start meanwhile;
        wait_ready() returns once the ready line comes. The stand-in prints it when
        the test makes a file."""
        from exec_sandbox import subprocess_utils
        from exec_sandbox.gvproxy import start_gvproxy
        from exec_sandbox.resource_cleanup import cleanup_process

        real_start = subprocess_utils.start_managed_process
        go = tmp_path / "go"
        script = f'while [ ! -e "{go}" ]; do sleep 0.01; done; echo "Listening on QEMU socket"; exec sleep 30'

        async def gvproxy_stand_in(_args: object, **kwargs: Any) -> Any:
            return await real_start(["sh", "-c", script], **kwargs)

        workdir = MagicMock()
        workdir.gvproxy_socket = tmp_path / "gvproxy.sock"
        with (
            patch("exec_sandbox.gvproxy.create_unix_socket", side_effect=lambda _path: socket.socket(socket.AF_UNIX)),
            patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=Path("/bin/sh")),
            patch("exec_sandbox.gvproxy.start_managed_process", side_effect=gvproxy_stand_in),
            patch("exec_sandbox.gvproxy.grant_qemu_vm_access", new_callable=AsyncMock),
        ):
            proc, log_task, wait_ready = await start_gvproxy(
                "gvproxy-early",
                allowed_domains=None,
                language="python",
                workdir=workdir,
                images_dir=tmp_path,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )
        try:
            ready = asyncio.create_task(wait_ready())
            await asyncio.sleep(0.2)
            assert not ready.done()  # returned before the ready line
            go.touch()
            await asyncio.wait_for(ready, timeout=5)
        finally:
            await cleanup_process(
                proc, "gvproxy", "gvproxy-early", term_timeout=3.0, kill_timeout=2.0, kill_first=False
            )
            await asyncio.wait_for(log_task, timeout=5)

    async def test_process_drain_task_is_the_log_task(self, tmp_path: Path) -> None:
        """The process carries its drain task (the seam that cleanup_process reads): cleanup
        then waits for the exit without a second reader on the pipes that task reads."""
        from exec_sandbox import subprocess_utils
        from exec_sandbox.gvproxy import start_gvproxy
        from exec_sandbox.resource_cleanup import cleanup_process

        real_start = subprocess_utils.start_managed_process

        async def gvproxy_stand_in(_args: object, **kwargs: Any) -> Any:
            return await real_start(["sh", "-c", 'echo "Listening on QEMU socket"; exec sleep 30'], **kwargs)

        workdir = MagicMock()
        workdir.gvproxy_socket = tmp_path / "gvproxy.sock"
        with (
            patch("exec_sandbox.gvproxy.create_unix_socket", side_effect=lambda _path: socket.socket(socket.AF_UNIX)),
            patch("exec_sandbox.gvproxy.get_gvproxy_path", new_callable=AsyncMock, return_value=Path("/bin/sh")),
            patch("exec_sandbox.gvproxy.start_managed_process", side_effect=gvproxy_stand_in),
            patch("exec_sandbox.gvproxy.grant_qemu_vm_access", new_callable=AsyncMock),
        ):
            proc, log_task, _wait_ready = await start_gvproxy(
                "gvproxy-drain",
                allowed_domains=None,
                language="python",
                workdir=workdir,
                images_dir=tmp_path,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )
        try:
            assert proc.drain_task is log_task
        finally:
            await cleanup_process(
                proc, "gvproxy", "gvproxy-drain", term_timeout=3.0, kill_timeout=2.0, kill_first=False
            )
            await asyncio.wait_for(log_task, timeout=5)

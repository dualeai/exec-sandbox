"""Unit tests for subprocess utilities.

Tests drain_subprocess_output with real subprocesses.
No mocks - spawns actual processes.
"""

import ast
import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import exec_sandbox
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.subprocess_utils import (
    communicate_managed_process,
    drain_subprocess_output,
    start_drain,
    start_managed_process,
    wait_for_socket,
)


async def _close_writer(_r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    """Minimal handler for Unix server sockets — close immediately."""
    w.close()
    await w.wait_closed()


async def create_process(cmd: list[str]) -> ProcessWrapper:
    """Helper to create a wrapped subprocess."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return ProcessWrapper(proc)


_SPAWN_CALLS = {
    ("asyncio", "create_subprocess_exec"),
    ("asyncio", "create_subprocess_shell"),
    *(("subprocess", name) for name in ("run", "Popen", "check_output", "check_call", "call")),
}


class TestChildEnv:
    """Child processes never inherit the instance keyring or cloud credentials."""

    def test_every_spawn_site_passes_env(self) -> None:
        """AST walk over the package: each spawn call passes env= (a new site cannot forget it)."""
        missing: list[str] = []
        sites = 0
        for path in sorted(Path(exec_sandbox.__file__).parent.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if (ast.unparse(node.func.value), node.func.attr) not in _SPAWN_CALLS:
                    continue
                sites += 1
                if not any(keyword.arg == "env" for keyword in node.keywords):
                    missing.append(f"{path.name}:{node.lineno}")
        assert sites > 0
        assert missing == []

    async def test_managed_child_does_not_see_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", "secret-keyring")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-aws")
        monkeypatch.setenv("CHILD_ENV_PROBE", "kept")

        returncode, stdout, _ = await communicate_managed_process(
            ["/usr/bin/env"], process_name="env", context_id="test-child-env"
        )

        assert returncode == 0
        assert b"CHILD_ENV_PROBE=kept" in stdout
        assert b"secret-keyring" not in stdout
        assert b"secret-aws" not in stdout


class TestStartManagedProcess:
    """Process publication is atomic with respect to caller cancellation."""

    async def test_cancelled_waiter_defers_until_spawn_is_published(self) -> None:
        spawn_entered = asyncio.Event()
        release_spawn = asyncio.Event()
        async_proc = MagicMock()
        wrapped_proc = MagicMock(spec=ProcessWrapper)
        published: list[ProcessWrapper] = []

        async def blocked_spawn(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            spawn_entered.set()
            await release_spawn.wait()
            return async_proc

        with (
            patch("exec_sandbox.subprocess_utils.asyncio.create_subprocess_exec", side_effect=blocked_spawn),
            patch("exec_sandbox.subprocess_utils.ProcessWrapper", return_value=wrapped_proc),
            patch("exec_sandbox.subprocess_utils.register_process") as register,
        ):
            waiter = asyncio.create_task(start_managed_process(["qemu-test"], on_process_started=published.append))
            await spawn_entered.wait()
            waiter.cancel()
            await asyncio.sleep(0)

            assert not waiter.done()
            release_spawn.set()
            with pytest.raises(asyncio.CancelledError):
                await waiter

        register.assert_called_once_with(wrapped_proc)
        assert published == [wrapped_proc]


class TestCommunicateManagedProcess:
    """Transient children retain cleanup ownership across cancellation."""

    async def test_cancelled_communication_kills_reaps_and_unregisters(self) -> None:
        communicate_entered = asyncio.Event()
        proc = MagicMock(spec=ProcessWrapper)
        proc.returncode = None

        async def blocked_communicate() -> tuple[bytes, bytes]:
            communicate_entered.set()
            await asyncio.Event().wait()
            return b"", b""

        proc.communicate = blocked_communicate

        async def publish_process(*_args, on_process_started=None, **_kwargs):  # type: ignore[no-untyped-def]
            assert on_process_started is not None
            on_process_started(proc)
            return proc

        with (
            patch("exec_sandbox.subprocess_utils.start_managed_process", side_effect=publish_process),
            patch(
                "exec_sandbox.subprocess_utils.cleanup_process", new_callable=AsyncMock, return_value=True
            ) as cleanup,
            patch("exec_sandbox.subprocess_utils.unregister_process") as unregister,
        ):
            task = asyncio.create_task(
                communicate_managed_process(
                    ["qemu-img", "info"],
                    process_name="qemu-img",
                    context_id="cancelled",
                )
            )
            await communicate_entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        cleanup.assert_awaited_once_with(proc, "qemu-img", "cancelled")
        unregister.assert_called_once_with(proc)

    async def test_normal_completion_unregisters_after_confirmed_cleanup(self) -> None:
        proc = MagicMock(spec=ProcessWrapper)
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"out", b"err"))

        async def publish_process(*_args, on_process_started=None, **_kwargs):  # type: ignore[no-untyped-def]
            assert on_process_started is not None
            on_process_started(proc)
            return proc

        with (
            patch("exec_sandbox.subprocess_utils.start_managed_process", side_effect=publish_process),
            patch("exec_sandbox.subprocess_utils.cleanup_process", new_callable=AsyncMock, return_value=True),
            patch("exec_sandbox.subprocess_utils.unregister_process") as unregister,
        ):
            result = await communicate_managed_process(
                ["qemu-img", "info"],
                process_name="qemu-img",
                context_id="normal",
            )

        assert result == (0, b"out", b"err")
        unregister.assert_called_once_with(proc)

    async def test_unconfirmed_cleanup_blocks_return_and_retries(self) -> None:
        proc = MagicMock(spec=ProcessWrapper)
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"out", b""))
        second_cleanup = asyncio.Event()

        async def publish_process(*_args, on_process_started=None, **_kwargs):  # type: ignore[no-untyped-def]
            assert on_process_started is not None
            on_process_started(proc)
            return proc

        async def cleanup(*_args, **_kwargs) -> bool:  # type: ignore[no-untyped-def]
            if not second_cleanup.is_set():
                second_cleanup.set()
                return False
            return True

        with (
            patch("exec_sandbox.subprocess_utils.start_managed_process", side_effect=publish_process),
            patch("exec_sandbox.subprocess_utils.cleanup_process", side_effect=cleanup) as cleanup_mock,
            patch("exec_sandbox.subprocess_utils.asyncio.sleep", new_callable=AsyncMock) as sleep,
            patch("exec_sandbox.subprocess_utils.unregister_process") as unregister,
        ):
            result = await communicate_managed_process(
                ["qemu-img", "info"],
                process_name="qemu-img",
                context_id="retry",
            )

        assert result == (0, b"out", b"")
        assert cleanup_mock.await_count == 2
        sleep.assert_awaited_once_with(1.0)
        unregister.assert_called_once_with(proc)

    async def test_cancellation_during_cleanup_backoff_still_confirms_death(self) -> None:
        proc = MagicMock(spec=ProcessWrapper)
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        backoff_entered = asyncio.Event()
        release_backoff = asyncio.Event()
        cleanup_attempts = 0

        async def publish_process(*_args, on_process_started=None, **_kwargs):  # type: ignore[no-untyped-def]
            assert on_process_started is not None
            on_process_started(proc)
            return proc

        async def cleanup(*_args, **_kwargs) -> bool:  # type: ignore[no-untyped-def]
            nonlocal cleanup_attempts
            cleanup_attempts += 1
            return cleanup_attempts >= 2

        async def blocked_sleep(_delay: float) -> None:
            backoff_entered.set()
            await release_backoff.wait()

        with (
            patch("exec_sandbox.subprocess_utils.start_managed_process", side_effect=publish_process),
            patch("exec_sandbox.subprocess_utils.cleanup_process", side_effect=cleanup),
            patch("exec_sandbox.subprocess_utils.asyncio.sleep", side_effect=blocked_sleep),
            patch("exec_sandbox.subprocess_utils.unregister_process") as unregister,
        ):
            task = asyncio.create_task(
                communicate_managed_process(
                    ["qemu-img", "info"],
                    process_name="qemu-img",
                    context_id="cancel-backoff",
                )
            )
            await backoff_entered.wait()
            task.cancel()
            release_backoff.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert cleanup_attempts == 2
        unregister.assert_called_once_with(proc)


class TestDrainSubprocessOutput:
    """Tests for drain_subprocess_output function.

    Uses real subprocesses - no mocking.
    """

    async def test_drain_stdout_only(self) -> None:
        """Drain stdout from a process that only writes to stdout."""
        captured: list[str] = []

        proc = await create_process(["echo", "hello world"])

        await drain_subprocess_output(proc, stdout_handler=captured.append, stderr_handler=[].append)

        # Wait for process to finish
        await proc.wait()

        assert "hello world" in captured

    async def test_drain_stderr_only(self) -> None:
        """Drain stderr from a process that only writes to stderr."""
        captured_stderr: list[str] = []

        # bash -c to write to stderr
        proc = await create_process(["bash", "-c", "echo 'error message' >&2"])

        await drain_subprocess_output(proc, stdout_handler=[].append, stderr_handler=captured_stderr.append)

        await proc.wait()

        assert "error message" in captured_stderr

    async def test_drain_both_stdout_stderr(self) -> None:
        """Drain both stdout and stderr concurrently."""
        captured_stdout: list[str] = []
        captured_stderr: list[str] = []

        # Process that writes to both streams
        proc = await create_process(["bash", "-c", "echo 'stdout line'; echo 'stderr line' >&2"])

        await drain_subprocess_output(
            proc,
            stdout_handler=captured_stdout.append,
            stderr_handler=captured_stderr.append,
        )

        await proc.wait()

        assert "stdout line" in captured_stdout
        assert "stderr line" in captured_stderr

    async def test_drain_multiple_lines(self) -> None:
        """Drain multiple lines from stdout."""
        captured: list[str] = []

        proc = await create_process(["bash", "-c", "echo line1; echo line2; echo line3"])

        await drain_subprocess_output(proc, stdout_handler=captured.append, stderr_handler=[].append)

        await proc.wait()

        assert len(captured) == 3
        assert "line1" in captured
        assert "line2" in captured
        assert "line3" in captured

    async def test_drain_interleaved_output(self) -> None:
        """Drain interleaved stdout/stderr without deadlock."""
        captured_stdout: list[str] = []
        captured_stderr: list[str] = []

        # Interleaved output - this could deadlock without concurrent draining
        proc = await create_process(
            [
                "bash",
                "-c",
                """
            for i in 1 2 3; do
                echo "out $i"
                echo "err $i" >&2
            done
            """,
            ]
        )

        await drain_subprocess_output(
            proc,
            stdout_handler=captured_stdout.append,
            stderr_handler=captured_stderr.append,
        )

        await proc.wait()

        # All output captured without deadlock
        assert len(captured_stdout) == 3
        assert len(captured_stderr) == 3

    async def test_drain_large_output(self) -> None:
        """Drain large output without pipe buffer exhaustion."""
        captured: list[str] = []

        # Generate 1000 lines (more than typical 64KB pipe buffer)
        proc = await create_process(
            ["bash", "-c", 'for i in $(seq 1 1000); do echo "line $i with some padding text"; done']
        )

        await drain_subprocess_output(proc, stdout_handler=captured.append, stderr_handler=[].append)

        await proc.wait()

        assert len(captured) == 1000
        assert "line 1 with some padding text" in captured
        assert "line 1000 with some padding text" in captured

    async def test_drain_empty_output(self) -> None:
        """Drain from process with no output."""
        captured_stdout: list[str] = []
        captured_stderr: list[str] = []

        proc = await create_process(["true"])  # Does nothing, exits 0

        await drain_subprocess_output(
            proc,
            stdout_handler=captured_stdout.append,
            stderr_handler=captured_stderr.append,
        )

        await proc.wait()

        assert captured_stdout == []
        assert captured_stderr == []

    async def test_drain_binary_safe(self) -> None:
        """Drain handles non-UTF8 gracefully (ignores decode errors)."""
        captured: list[str] = []

        # Output some bytes that might cause decode issues
        proc = await create_process(["bash", "-c", "echo 'normal text'"])

        await drain_subprocess_output(proc, stdout_handler=captured.append, stderr_handler=[].append)

        await proc.wait()

        assert "normal text" in captured

    async def test_drain_python_subprocess(self) -> None:
        """Drain from Python subprocess with mixed output."""
        captured_stdout: list[str] = []
        captured_stderr: list[str] = []

        proc = await create_process(
            [
                "python3",
                "-c",
                """
import sys
print('stdout message')
print('stderr message', file=sys.stderr)
print('another stdout')
            """,
            ]
        )

        await drain_subprocess_output(
            proc,
            stdout_handler=captured_stdout.append,
            stderr_handler=captured_stderr.append,
        )

        await proc.wait()

        assert "stdout message" in captured_stdout
        assert "another stdout" in captured_stdout
        assert "stderr message" in captured_stderr


class TestStartDrain:
    """start_drain: its task is the process's drain_task (cleanup then never reads the
    pipes), it reads both pipes, and a failure of the task is logged."""

    async def test_task_is_the_drain_task_and_reads_both_pipes(self) -> None:
        lines: list[str] = []
        proc = await create_process(["bash", "-c", "echo out; echo err >&2"])

        task = start_drain(
            proc, process_name="bash", context_id="drain-1", stdout_handler=lines.append, stderr_handler=lines.append
        )

        assert proc.drain_task is task
        await task
        await proc.wait()
        assert sorted(lines) == ["err", "out"]

    async def test_failed_task_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        proc = await create_process(["echo", "line"])

        def fail(_line: str) -> None:
            raise RuntimeError("handler failed")

        with caplog.at_level(logging.ERROR, logger="exec_sandbox.subprocess_utils"):
            task = start_drain(
                proc, process_name="echo", context_id="drain-2", stdout_handler=fail, stderr_handler=[].append
            )
            with pytest.raises(ExceptionGroup):
                await task
            await asyncio.sleep(0)  # the done callback runs in the next loop pass
        await proc.wait()

        assert [(r.getMessage(), getattr(r, "task_name", None)) for r in caplog.records] == [
            ("Background task failed", "echo-drain-drain-2")
        ]


class TestWaitForSocket:
    """Tests for wait_for_socket function.

    Uses real Unix sockets - no mocking.
    Uses short_tmp_dir fixture (short paths) instead of tmp_path
    to avoid macOS AF_UNIX 104-byte path limit.
    """

    async def test_socket_ready_immediately(self, short_tmp_dir: Path) -> None:
        """Socket already listening before wait_for_socket is called."""
        sock_path = short_tmp_dir / "s.sock"

        server = await asyncio.start_unix_server(_close_writer, path=str(sock_path))
        try:
            result = await wait_for_socket(sock_path, timeout=2.0)
            assert result is None
        finally:
            server.close()
            await server.wait_closed()

    async def test_socket_appears_after_delay(self, short_tmp_dir: Path) -> None:
        """Socket created by a background task after a short delay."""
        sock_path = short_tmp_dir / "s.sock"

        async def _create_server_later() -> asyncio.Server:
            await asyncio.sleep(0.1)
            return await asyncio.start_unix_server(_close_writer, path=str(sock_path))

        task = asyncio.create_task(_create_server_later())
        try:
            result = await wait_for_socket(sock_path, timeout=2.0)
            assert result is None
        finally:
            server = await task
            server.close()
            await server.wait_closed()

    async def test_keep_connection_true_returns_streams(self, short_tmp_dir: Path) -> None:
        """keep_connection=True returns (reader, writer) streams."""
        sock_path = short_tmp_dir / "s.sock"

        server = await asyncio.start_unix_server(_close_writer, path=str(sock_path))
        try:
            result = await wait_for_socket(sock_path, timeout=2.0, keep_connection=True)
            assert result is not None
            r, w = result
            assert isinstance(r, asyncio.StreamReader)
            assert isinstance(w, asyncio.StreamWriter)
            w.write(b"ping")
            w.close()
            await w.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_keep_connection_false_returns_none(self, short_tmp_dir: Path) -> None:
        """keep_connection=False (default) returns None."""
        sock_path = short_tmp_dir / "s.sock"

        server = await asyncio.start_unix_server(_close_writer, path=str(sock_path))
        try:
            result = await wait_for_socket(sock_path, timeout=2.0, keep_connection=False)
            assert result is None
        finally:
            server.close()
            await server.wait_closed()

    async def test_timeout_no_socket(self, short_tmp_dir: Path) -> None:
        """TimeoutError when socket file never appears."""
        sock_path = short_tmp_dir / "s.sock"
        with pytest.raises(TimeoutError):
            await wait_for_socket(sock_path, timeout=0.1)

    async def test_socket_file_exists_but_not_listening(self, short_tmp_dir: Path) -> None:
        """Fails when path exists as regular file, not a socket.

        On Linux: connect() retries with ConnectionRefused → TimeoutError.
        On macOS: connect() raises OSError (ENOTSOCK) immediately.
        """
        sock_path = short_tmp_dir / "s.sock"
        sock_path.write_text("not a socket")
        with pytest.raises((TimeoutError, OSError)):
            await wait_for_socket(sock_path, timeout=0.2)

    async def test_abort_check_raises(self, short_tmp_dir: Path) -> None:
        """abort_check exception propagates instead of TimeoutError."""
        sock_path = short_tmp_dir / "s.sock"
        call_count = 0

        def _abort() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("process died")

        with pytest.raises(RuntimeError, match="process died"):
            await wait_for_socket(sock_path, timeout=5.0, abort_check=_abort)

    async def test_abort_check_called_each_iteration(self, short_tmp_dir: Path) -> None:
        """abort_check is invoked on every poll iteration."""
        sock_path = short_tmp_dir / "s.sock"
        call_count = 0

        def _count() -> None:
            nonlocal call_count
            call_count += 1

        with pytest.raises(TimeoutError):
            await wait_for_socket(sock_path, timeout=0.1, abort_check=_count)

        assert call_count > 0

    async def test_zero_timeout(self, short_tmp_dir: Path) -> None:
        """Zero timeout raises TimeoutError immediately."""
        sock_path = short_tmp_dir / "s.sock"
        with pytest.raises(TimeoutError):
            await wait_for_socket(sock_path, timeout=0.0)

    async def test_concurrent_wait_for_socket(self, short_tmp_dir: Path) -> None:
        """Multiple concurrent waits all succeed against the same server."""
        sock_path = short_tmp_dir / "s.sock"

        server = await asyncio.start_unix_server(_close_writer, path=str(sock_path))
        try:
            results = await asyncio.gather(
                wait_for_socket(sock_path, timeout=2.0, keep_connection=False),
                wait_for_socket(sock_path, timeout=2.0, keep_connection=False),
                wait_for_socket(sock_path, timeout=2.0, keep_connection=False),
            )
            assert all(r is None for r in results)
        finally:
            server.close()
            await server.wait_closed()

    async def test_socket_path_in_nonexistent_directory(self) -> None:
        """TimeoutError when socket path is in a directory that does not exist."""
        sock_path = Path("/tmp/nonexistent_dir_xyz/test.sock")
        with pytest.raises(TimeoutError):
            await wait_for_socket(sock_path, timeout=0.1)

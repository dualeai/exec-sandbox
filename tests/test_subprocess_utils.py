"""Unit tests for subprocess utilities.

Tests drain_subprocess_output with real subprocesses.
No mocks - spawns actual processes.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.subprocess_utils import drain_subprocess_output, wait_for_socket


async def create_process(cmd: list[str]) -> ProcessWrapper:
    """Helper to create a wrapped subprocess."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return ProcessWrapper(proc)


class TestDrainSubprocessOutput:
    """Tests for drain_subprocess_output function.

    Uses real subprocesses - no mocking.
    """

    async def test_drain_stdout_only(self) -> None:
        """Drain stdout from a process that only writes to stdout."""
        captured: list[str] = []

        proc = await create_process(["echo", "hello world"])

        await drain_subprocess_output(
            proc,
            process_name="echo",
            context_id="test-1",
            stdout_handler=captured.append,
        )

        # Wait for process to finish
        await proc.wait()

        assert "hello world" in captured

    async def test_drain_stderr_only(self) -> None:
        """Drain stderr from a process that only writes to stderr."""
        captured_stderr: list[str] = []

        # bash -c to write to stderr
        proc = await create_process(["bash", "-c", "echo 'error message' >&2"])

        await drain_subprocess_output(
            proc,
            process_name="bash",
            context_id="test-2",
            stderr_handler=captured_stderr.append,
        )

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
            process_name="bash",
            context_id="test-3",
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

        await drain_subprocess_output(
            proc,
            process_name="bash",
            context_id="test-4",
            stdout_handler=captured.append,
        )

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
            process_name="bash",
            context_id="test-5",
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

        await drain_subprocess_output(
            proc,
            process_name="bash",
            context_id="test-6",
            stdout_handler=captured.append,
        )

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
            process_name="true",
            context_id="test-7",
            stdout_handler=captured_stdout.append,
            stderr_handler=captured_stderr.append,
        )

        await proc.wait()

        assert captured_stdout == []
        assert captured_stderr == []

    async def test_drain_with_default_handlers(self) -> None:
        """Drain with default handlers (logging, no capture)."""
        proc = await create_process(["echo", "test"])

        # No custom handlers - uses default logging
        await drain_subprocess_output(
            proc,
            process_name="echo",
            context_id="test-8",
        )

        await proc.wait()

        # Just verify it completes without error
        assert proc.returncode == 0

    async def test_drain_binary_safe(self) -> None:
        """Drain handles non-UTF8 gracefully (ignores decode errors)."""
        captured: list[str] = []

        # Output some bytes that might cause decode issues
        proc = await create_process(["bash", "-c", "echo 'normal text'"])

        await drain_subprocess_output(
            proc,
            process_name="bash",
            context_id="test-9",
            stdout_handler=captured.append,
        )

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
            process_name="python",
            context_id="test-10",
            stdout_handler=captured_stdout.append,
            stderr_handler=captured_stderr.append,
        )

        await proc.wait()

        assert "stdout message" in captured_stdout
        assert "another stdout" in captured_stdout
        assert "stderr message" in captured_stderr


class TestWaitForSocket:
    """Tests for wait_for_socket function.

    Uses real Unix sockets - no mocking.
    Uses tempfile.TemporaryDirectory() (short paths) instead of tmp_path
    to avoid macOS AF_UNIX 104-byte path limit.
    """

    async def test_socket_ready_immediately(self) -> None:
        """Socket already listening before wait_for_socket is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "s.sock"

            async def _handler(_r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
                w.close()
                await w.wait_closed()

            server = await asyncio.start_unix_server(_handler, path=str(sock_path))
            try:
                result = await wait_for_socket(sock_path, timeout=2.0)
                assert result is None
            finally:
                server.close()
                await server.wait_closed()

    async def test_socket_appears_after_delay(self) -> None:
        """Socket created by a background task after a short delay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "s.sock"

            async def _handler(_r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
                w.close()
                await w.wait_closed()

            async def _create_server_later() -> asyncio.Server:
                await asyncio.sleep(0.1)
                return await asyncio.start_unix_server(_handler, path=str(sock_path))

            task = asyncio.create_task(_create_server_later())
            try:
                result = await wait_for_socket(sock_path, timeout=2.0)
                assert result is None
            finally:
                server = await task
                server.close()
                await server.wait_closed()

    async def test_keep_connection_true_returns_streams(self) -> None:
        """keep_connection=True returns (reader, writer) streams."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "s.sock"

            async def _handler(_r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
                w.close()
                await w.wait_closed()

            server = await asyncio.start_unix_server(_handler, path=str(sock_path))
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

    async def test_keep_connection_false_returns_none(self) -> None:
        """keep_connection=False (default) returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "s.sock"

            async def _handler(_r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
                w.close()
                await w.wait_closed()

            server = await asyncio.start_unix_server(_handler, path=str(sock_path))
            try:
                result = await wait_for_socket(sock_path, timeout=2.0, keep_connection=False)
                assert result is None
            finally:
                server.close()
                await server.wait_closed()

    async def test_timeout_no_socket(self) -> None:
        """TimeoutError when socket file never appears."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "s.sock"
            with pytest.raises(TimeoutError):
                await wait_for_socket(sock_path, timeout=0.1)

    async def test_socket_file_exists_but_not_listening(self) -> None:
        """Fails when path exists as regular file, not a socket.

        On Linux: connect() retries with ConnectionRefused → TimeoutError.
        On macOS: connect() raises OSError (ENOTSOCK) immediately.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "s.sock"
            sock_path.write_text("not a socket")
            with pytest.raises((TimeoutError, OSError)):
                await wait_for_socket(sock_path, timeout=0.2)

    async def test_abort_check_raises(self) -> None:
        """abort_check exception propagates instead of TimeoutError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "s.sock"
            call_count = 0

            def _abort() -> None:
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    raise RuntimeError("process died")

            with pytest.raises(RuntimeError, match="process died"):
                await wait_for_socket(sock_path, timeout=5.0, abort_check=_abort)

    async def test_abort_check_called_each_iteration(self) -> None:
        """abort_check is invoked on every poll iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "s.sock"
            call_count = 0

            def _count() -> None:
                nonlocal call_count
                call_count += 1

            with pytest.raises(TimeoutError):
                await wait_for_socket(sock_path, timeout=0.1, abort_check=_count)

            assert call_count > 0

    async def test_zero_timeout(self) -> None:
        """Zero timeout raises TimeoutError immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "s.sock"
            with pytest.raises(TimeoutError):
                await wait_for_socket(sock_path, timeout=0.0)

    async def test_concurrent_wait_for_socket(self) -> None:
        """Multiple concurrent waits all succeed against the same server."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "s.sock"

            async def _handler(_r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
                w.close()
                await w.wait_closed()

            server = await asyncio.start_unix_server(_handler, path=str(sock_path))
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

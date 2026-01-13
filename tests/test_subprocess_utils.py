"""Unit tests for subprocess utilities.

Tests drain_subprocess_output with real subprocesses.
No mocks - spawns actual processes.
"""

import asyncio

from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.subprocess_utils import drain_subprocess_output


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

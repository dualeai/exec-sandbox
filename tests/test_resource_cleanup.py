"""resource_cleanup.cleanup_process: SIGTERM, then SIGKILL."""

import asyncio
import shlex
import signal
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.resource_cleanup import cleanup_process


async def _spawn(script: str) -> ProcessWrapper:
    """A shell child with pipes, once it printed its first line (its signal setup is done)."""
    proc = ProcessWrapper(
        await asyncio.create_subprocess_exec(
            "sh", "-c", script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    )
    assert proc.stdout is not None
    assert await proc.stdout.readline() == b"ready\n"
    return proc


def _stuck_process() -> Mock:
    """A stand-in process, alive, for the states that no real process gets on demand: SIGKILL
    that does not end it in time (a QEMU in uninterruptible sleep, D state), a race window, an
    OS error."""
    proc = Mock(spec=ProcessWrapper)
    proc.pid = 12345
    proc.returncode = None
    proc.stdout = None
    proc.stderr = None
    proc.terminate = AsyncMock()
    proc.kill = AsyncMock()
    return proc


class TestResourceCleanupProcess:
    """cleanup_process: SIGTERM, then SIGKILL; True once the process is gone."""

    async def test_cleanup_process_sigterm_success(self) -> None:
        """A process that SIGTERM ends: True, and it ended on SIGTERM."""
        proc = await _spawn("echo ready; exec sleep 30")

        assert (
            await cleanup_process(
                proc=proc,
                name="graceful-process",
                context_id="test-ctx",
                term_timeout=3.0,
                kill_timeout=2.0,
                kill_first=False,
            )
            is True
        )
        assert proc.returncode == -signal.SIGTERM

    async def test_kill_first_sends_sigkill_at_once(self) -> None:
        """kill_first: a process that would wait out SIGTERM ends on SIGKILL at once, well
        before term_timeout."""
        proc = await _spawn("trap '' TERM; echo ready; exec sleep 30")
        loop = asyncio.get_running_loop()
        start = loop.time()

        result = await cleanup_process(
            proc=proc, name="killed-first", context_id="test-ctx", term_timeout=10.0, kill_timeout=2.0, kill_first=True
        )

        assert (result, proc.returncode) == (True, -signal.SIGKILL)
        assert loop.time() - start < 2.0

    async def test_cleanup_process_sigterm_timeout_sigkill_success(self, tmp_path: Path) -> None:
        """A process that outlives SIGTERM: SIGTERM first (its trap makes a file), SIGKILL after
        term_timeout, True."""
        termed = tmp_path / "termed"
        trap = shlex.quote(f": > {shlex.quote(str(termed))}")
        proc = await _spawn(f"trap {trap} TERM; echo ready; while :; do sleep 0.01; done")

        result = await cleanup_process(
            proc=proc,
            name="stubborn-process",
            context_id="test-ctx",
            term_timeout=0.3,
            kill_timeout=5.0,
            kill_first=False,
        )

        assert (result, proc.returncode, termed.exists()) == (True, -signal.SIGKILL, True)

    async def test_cleanup_process_already_dead(self) -> None:
        """A process that already exited: True at once, and its exit code stays."""
        proc = ProcessWrapper(await asyncio.create_subprocess_exec("true"))
        await proc.wait()

        assert (
            await cleanup_process(
                proc=proc,
                name="dead-process",
                context_id="test-ctx",
                term_timeout=3.0,
                kill_timeout=2.0,
                kill_first=False,
            )
            is True
        )
        assert proc.returncode == 0

    async def test_cleanup_process_none(self) -> None:
        """None process returns True (no-op)."""
        assert (
            await cleanup_process(
                proc=None,
                name="null-process",
                context_id="test-ctx",
                term_timeout=3.0,
                kill_timeout=2.0,
                kill_first=False,
            )
            is True
        )

    async def test_cleanup_process_process_lookup_error(self) -> None:
        """The process ends between the returncode check and SIGTERM (ProcessLookupError):
        True. A race window: a real process cannot be made to end inside it."""
        proc = _stuck_process()
        proc.terminate = AsyncMock(side_effect=ProcessLookupError)

        assert (
            await cleanup_process(
                proc=proc,
                name="race-condition-process",
                context_id="test-ctx",
                term_timeout=3.0,
                kill_timeout=2.0,
                kill_first=False,
            )
            is True
        )

    async def test_cleanup_process_kill_process_lookup_error(self) -> None:
        """The process ends between the SIGTERM timeout and SIGKILL (ProcessLookupError):
        True. A race window: a real process cannot be made to end inside it."""
        proc = _stuck_process()
        proc.wait_with_timeout = AsyncMock(side_effect=TimeoutError)
        proc.kill = AsyncMock(side_effect=ProcessLookupError)

        result = await cleanup_process(
            proc=proc,
            name="pgid-race-process",
            context_id="test-ctx",
            term_timeout=0.01,
            kill_timeout=0.01,
            kill_first=False,
        )

        assert result is True

    async def test_process_that_sigkill_does_not_end_in_time(self) -> None:
        """SIGKILL that does not end the process within kill_timeout (D state): False."""
        proc = _stuck_process()
        proc.wait_with_timeout = AsyncMock(side_effect=TimeoutError)

        result = await cleanup_process(
            proc=proc,
            name="stubborn-process",
            context_id="test-ctx",
            term_timeout=0.01,
            kill_timeout=0.01,
            kill_first=False,
        )

        assert (result, proc.kill.await_count) == (False, 1)

    async def test_cleanup_process_unexpected_exception(self) -> None:
        """An unexpected error of the OS call: False, never raised (cleanup paths go on)."""
        proc = _stuck_process()
        proc.terminate = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        assert (
            await cleanup_process(
                proc=proc,
                name="error-process",
                context_id="test-ctx",
                term_timeout=3.0,
                kill_timeout=2.0,
                kill_first=False,
            )
            is False
        )

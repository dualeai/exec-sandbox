"""resource_cleanup.cleanup_process: SIGTERM, then SIGKILL, then a kept reap task."""

import asyncio
import logging
import shlex
import signal
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.resource_cleanup import _zombie_reap_tasks, cleanup_process


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
        await asyncio.gather(*_zombie_reap_tasks)  # its background reap ends with the test

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


class TestZombieTaskTracking:
    """The background reap of a process that cleanup_process gave up on, or that had already
    exited. The module keeps a reference to each reap task until it ends: asyncio keeps only a
    weak one, and the garbage collector can drop a task that runs. The tests read that set
    (_zombie_reap_tasks), as a collected task cannot be made to happen on demand."""

    @pytest.fixture(autouse=True)
    def clear_zombie_tasks(self):
        """Clear zombie task set before and after each test."""
        _zombie_reap_tasks.clear()
        yield
        _zombie_reap_tasks.clear()

    async def test_already_dead_process_is_reaped_in_a_kept_task(self) -> None:
        """The reap task is kept while it runs, and released when it ends."""
        reaped = asyncio.Event()
        proc = _stuck_process()
        proc.returncode = 0

        async def wait() -> int:
            await reaped.wait()
            return 0

        proc.wait = wait

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
        assert len(_zombie_reap_tasks) == 1
        (task,) = _zombie_reap_tasks
        reaped.set()
        await task
        await asyncio.sleep(0)  # the done callback runs after the task ends
        assert not _zombie_reap_tasks

    async def test_reap_after_sigkill_timeout_waits_again(self) -> None:
        """After a SIGKILL that does not end the process in time, a kept task waits for it
        again (a zombie is reaped once the kernel lets it go); a timeout of that wait is no
        error."""
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

        assert result is False
        assert len(_zombie_reap_tasks) == 1
        (task,) = _zombie_reap_tasks
        await task
        await asyncio.sleep(0)
        assert proc.wait_with_timeout.await_count == 3  # SIGTERM wait, SIGKILL wait, reap
        assert not _zombie_reap_tasks

    async def test_failed_reap_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A reap that fails is logged as a warning and released."""
        proc = _stuck_process()
        proc.returncode = 0
        proc.wait = AsyncMock(side_effect=OSError("wait failed"))

        with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
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
            (task,) = _zombie_reap_tasks
            with pytest.raises(OSError, match="wait failed"):
                await task
            await asyncio.sleep(0)

        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
            "Zombie reaping task failed"
        ]
        assert not _zombie_reap_tasks

    async def test_cancelled_reap_is_not_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A reap cancelled (at shutdown) logs nothing, and is released."""
        proc = _stuck_process()
        proc.returncode = 0

        async def wait() -> int:
            await asyncio.Event().wait()
            return 0

        proc.wait = wait

        with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
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
            (task,) = _zombie_reap_tasks
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
        assert not _zombie_reap_tasks

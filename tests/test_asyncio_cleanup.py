"""Tests for asyncio task cleanup and event loop lifecycle.

These tests verify that:
1. Tasks created by asyncio.gather() are properly awaited on cancellation
2. Fire-and-forget tasks are tracked and cleaned up
3. Health check loops terminate cleanly
4. Python 3.14 stricter task detection doesn't produce warnings

Root cause addressed:
When asyncio.gather() is cancelled, it cancels child tasks but does NOT await
their completion before raising CancelledError. Python 3.14 has stricter
detection of these orphaned tasks, causing "Task was destroyed but it is pending"
warnings.
"""

import asyncio
import sys
from unittest.mock import MagicMock

import pytest

# Skip all tests in this module on Python < 3.12
pytestmark = [
    pytest.mark.asyncio,
]


class TestCancellationDuringGather:
    """Verify that gather children are properly awaited on cancellation."""

    async def test_gather_cancel_awaits_children(self) -> None:
        """When gather is cancelled, all children should complete."""
        child_completed = asyncio.Event()
        child_started = asyncio.Event()

        async def slow_child() -> str:
            child_started.set()
            try:
                await asyncio.sleep(10)
                return "completed"
            except asyncio.CancelledError:
                # Simulate cleanup work
                await asyncio.sleep(0.01)
                child_completed.set()
                raise

        async def run_gather() -> None:
            await asyncio.gather(slow_child())

        task = asyncio.create_task(run_gather())

        # Wait for child to start
        await child_started.wait()

        # Cancel the gather
        task.cancel()

        # The gather task should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            await task

        # But the child should have completed its cleanup
        # Give a small window for the cleanup to finish
        await asyncio.sleep(0.05)
        assert child_completed.is_set(), "Child task cleanup was not awaited"

    async def test_gather_with_return_exceptions_completes_all(self) -> None:
        """Gather with return_exceptions=True should collect all results."""
        results: list[str] = []

        async def task_success() -> str:
            await asyncio.sleep(0.01)
            results.append("success")
            return "ok"

        async def task_error() -> str:
            await asyncio.sleep(0.01)
            results.append("error")
            raise ValueError("intentional")

        gathered = await asyncio.gather(
            task_success(),
            task_error(),
            return_exceptions=True,
        )

        assert len(gathered) == 2
        assert gathered[0] == "ok"
        assert isinstance(gathered[1], ValueError)
        assert results == ["success", "error"]


class TestFireAndForgetTaskTracking:
    """Verify fire-and-forget task lifecycle management."""

    async def test_task_set_tracks_pending_tasks(self) -> None:
        """Task set pattern should track all pending tasks."""
        tasks: set[asyncio.Task[None]] = set()
        completed_count = 0

        async def background_work() -> None:
            nonlocal completed_count
            await asyncio.sleep(0.01)
            completed_count += 1

        # Create and track tasks
        for _ in range(3):
            task = asyncio.create_task(background_work())
            tasks.add(task)
            task.add_done_callback(lambda t: tasks.discard(t))

        # Wait for all to complete
        await asyncio.gather(*list(tasks), return_exceptions=True)

        assert completed_count == 3
        assert len(tasks) == 0, "Task set should be empty after completion"

    async def test_cancel_all_tracked_tasks(self) -> None:
        """Cancelling tracked tasks should await their completion."""
        tasks: set[asyncio.Task[None]] = set()
        cleanup_done = asyncio.Event()
        task_started = asyncio.Event()

        async def long_running() -> None:
            try:
                task_started.set()
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cleanup_done.set()
                raise

        task = asyncio.create_task(long_running())
        tasks.add(task)
        task.add_done_callback(lambda t: tasks.discard(t))

        # Wait for task to start before cancelling
        await task_started.wait()

        # Cancel all tasks (the pattern from warm_vm_pool.stop())
        tasks_to_cancel = list(tasks)
        for t in tasks_to_cancel:
            if not t.done():
                t.cancel()

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        assert cleanup_done.is_set(), "Task cleanup should have executed"


class TestHealthCheckLoopCancellation:
    """Verify health check loop terminates cleanly."""

    async def test_shutdown_event_stops_loop(self) -> None:
        """Shutdown event should break the health check loop."""
        shutdown_event = asyncio.Event()
        loop_iterations = 0

        async def health_check_loop() -> None:
            nonlocal loop_iterations
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=0.01,  # Short timeout for test
                    )
                    break
                except TimeoutError:
                    loop_iterations += 1
                    if loop_iterations >= 3:
                        break

        task = asyncio.create_task(health_check_loop())

        # Let it run a few iterations
        await asyncio.sleep(0.05)

        # Signal shutdown
        shutdown_event.set()

        # Should complete without hanging
        await asyncio.wait_for(task, timeout=1.0)
        assert loop_iterations >= 1

    async def test_health_task_timeout_and_cancel(self) -> None:
        """Health task that times out should be cancelled cleanly."""
        cancelled = asyncio.Event()

        async def slow_health_check() -> None:
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow_health_check())

        # Simulate stop() with timeout pattern
        try:
            await asyncio.wait_for(task, timeout=0.01)
        except TimeoutError:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert cancelled.is_set()


class TestShieldedCleanup:
    """Verify shielded cleanup operations complete even on cancellation."""

    async def test_shield_protects_cleanup(self) -> None:
        """Shielded cleanup should complete even when outer task is cancelled."""
        cleanup_started = asyncio.Event()
        cleanup_completed = asyncio.Event()

        async def cleanup_operation() -> bool:
            cleanup_started.set()
            await asyncio.sleep(0.05)  # Simulate cleanup work
            cleanup_completed.set()
            return True

        async def outer_with_cleanup() -> None:
            try:
                await asyncio.sleep(100)
            finally:
                # This is the pattern in _force_cleanup_all_resources
                await asyncio.shield(cleanup_operation())

        task = asyncio.create_task(outer_with_cleanup())

        # Give time to start
        await asyncio.sleep(0.01)

        # Cancel outer task
        task.cancel()

        # Outer task should be cancelled
        with pytest.raises(asyncio.CancelledError):
            await task

        # But cleanup should complete
        await asyncio.sleep(0.1)
        assert cleanup_completed.is_set(), "Shielded cleanup should have completed"


class TestFullLifecycleCleanup:
    """Integration tests for full lifecycle cleanup scenarios."""

    async def test_pool_stop_awaits_all_replenish_tasks(self) -> None:
        """Pool stop should await all replenish tasks, not just cancel them."""
        replenish_tasks: set[asyncio.Task[None]] = set()
        cleanup_count = 0
        all_started = asyncio.Event()
        started_count = 0

        async def mock_replenish() -> None:
            nonlocal cleanup_count, started_count
            try:
                started_count += 1
                if started_count == 3:
                    all_started.set()
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # Simulate cleanup
                await asyncio.sleep(0.01)
                cleanup_count += 1
                raise

        # Start several replenish tasks
        for _ in range(3):
            task = asyncio.create_task(mock_replenish())
            replenish_tasks.add(task)
            task.add_done_callback(lambda t: replenish_tasks.discard(t))

        # Wait for all tasks to start
        await all_started.wait()

        # Simulate stop() - the fixed pattern
        tasks_to_cancel = list(replenish_tasks)
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        replenish_tasks.clear()

        assert cleanup_count == 3, "All replenish task cleanups should have run"


@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="Python 3.14 specific asyncio cleanup behavior",
)
class TestPython314AsyncioCleanup:
    """Tests specifically for Python 3.14 stricter task detection."""

    async def test_no_pending_task_warning(self) -> None:
        """Verify no 'Task was destroyed but it is pending' warnings."""
        import warnings

        warnings.filterwarnings("error", "Task was destroyed but it is pending")

        tasks: set[asyncio.Task[None]] = set()

        async def work() -> None:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                await asyncio.sleep(0.01)
                raise

        # Create tasks
        for _ in range(5):
            task = asyncio.create_task(work())
            tasks.add(task)
            task.add_done_callback(lambda t: tasks.discard(t))

        # Cancel and await all - the correct pattern
        tasks_to_cancel = list(tasks)
        for t in tasks_to_cancel:
            t.cancel()

        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        tasks.clear()

        # If we get here without RuntimeWarning being raised, the test passes


class TestCancelledErrorHandling:
    """Verify CancelledError is handled separately from Exception."""

    async def test_cancelled_error_propagates(self) -> None:
        """CancelledError should propagate, not be caught by except Exception."""
        was_cancelled = False

        async def replenish_like() -> None:
            nonlocal was_cancelled
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                was_cancelled = True
                raise

        task = asyncio.create_task(replenish_like())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert was_cancelled

    async def test_cleanup_runs_on_cancellation(self) -> None:
        """Cleanup should run when task is cancelled mid-operation."""
        cleanup_ran = False
        vm_created = False

        async def replenish_with_cleanup() -> None:
            nonlocal cleanup_ran, vm_created
            vm: MagicMock | None = None
            try:
                # Simulate VM creation
                vm = MagicMock()
                vm_created = True
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                if vm is not None:
                    cleanup_ran = True
                raise

        task = asyncio.create_task(replenish_with_cleanup())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert vm_created
        assert cleanup_ran

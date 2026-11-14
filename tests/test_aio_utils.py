"""Unit tests for the cancellation-safe task ownership primitives.

These helpers back every owned-teardown loop in the codebase, so their
truth table is pinned directly: settled / cancelled-then-settled /
task-cancelled / task-error / cancellation + task-error (chained).
"""

import asyncio
import threading
import time

import pytest
from tenacity import retry_if_exception_type, stop_after_attempt, wait_none

from exec_sandbox.aio_utils import (
    AsyncRetrying,
    await_cancellation_safe,
    await_settled,
    gather_or_cancel,
    run_stoppable,
    run_to_end,
)


async def _wait_and_return(gate: asyncio.Event, value: str = "done") -> str:
    await gate.wait()
    return value


async def _wait_and_raise(gate: asyncio.Event) -> None:
    await gate.wait()
    raise RuntimeError("task failed")


class TestAwaitSettled:
    async def test_completed_task_returns_none(self) -> None:
        task = asyncio.create_task(asyncio.sleep(0, result="x"))
        assert await await_settled(task) is None
        assert task.result() == "x"

    async def test_absorbs_repeated_cancellation_until_task_settles(self) -> None:
        gate = asyncio.Event()
        task = asyncio.create_task(_wait_and_return(gate))

        async def waiter() -> asyncio.CancelledError | None:
            return await await_settled(task)

        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        waiter_task.cancel()
        await asyncio.sleep(0)
        waiter_task.cancel()
        gate.set()

        cancellation = await waiter_task
        assert isinstance(cancellation, asyncio.CancelledError)
        assert task.result() == "done"

    async def test_task_error_is_not_raised_here(self) -> None:
        gate = asyncio.Event()
        gate.set()
        task = asyncio.create_task(_wait_and_raise(gate))
        assert await await_settled(task) is None
        with pytest.raises(RuntimeError, match="task failed"):
            task.result()

    async def test_task_cancelled_by_third_party_reports_cancellation(self) -> None:
        gate = asyncio.Event()
        task = asyncio.create_task(_wait_and_return(gate))
        await asyncio.sleep(0)
        task.cancel()
        cancellation = await await_settled(task)
        # Third-party cancellation of the task is indistinguishable from
        # caller cancellation (documented caveat).
        assert isinstance(cancellation, asyncio.CancelledError)
        assert task.cancelled()


class TestAwaitCancellationSafe:
    async def test_returns_result_without_cancellation(self) -> None:
        assert await await_cancellation_safe(asyncio.sleep(0, result="ok")) == "ok"

    async def test_raises_operation_error_without_cancellation(self) -> None:
        gate = asyncio.Event()
        gate.set()
        with pytest.raises(RuntimeError, match="task failed"):
            await await_cancellation_safe(_wait_and_raise(gate))

    async def test_cancellation_deferred_then_reraised(self) -> None:
        gate = asyncio.Event()
        finished = False

        async def operation() -> None:
            nonlocal finished
            await gate.wait()
            finished = True

        async def caller() -> None:
            await await_cancellation_safe(operation())

        caller_task = asyncio.create_task(caller())
        await asyncio.sleep(0)
        caller_task.cancel()
        await asyncio.sleep(0)
        gate.set()

        with pytest.raises(asyncio.CancelledError):
            await caller_task
        assert finished  # the operation ran to completion despite cancellation

    async def test_cancellation_plus_operation_error_chains_cause(self) -> None:
        """The `raise cancellation from error` leg: CancelledError propagates
        with __cause__ set to the operation's own failure."""
        gate = asyncio.Event()
        observed: list[BaseException | None] = []

        async def caller() -> None:
            try:
                await await_cancellation_safe(_wait_and_raise(gate))
            except asyncio.CancelledError as cancellation:
                observed.append(cancellation.__cause__)
                raise

        caller_task = asyncio.create_task(caller())
        await asyncio.sleep(0)
        caller_task.cancel()
        await asyncio.sleep(0)
        gate.set()

        with pytest.raises(asyncio.CancelledError):
            await caller_task
        assert len(observed) == 1
        assert isinstance(observed[0], RuntimeError)
        assert str(observed[0]) == "task failed"


class TestGatherOrCancel:
    """gather_or_cancel: the results in order; on a failure, the others end before it raises."""

    @staticmethod
    async def _wait_until_cancelled(name: str, log: list[str]) -> None:
        """Wait until cancelled, then clean up (one loop pass) before it ends."""
        log.append(f"{name} started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            log.append(f"{name} ended")
            raise

    async def test_results_come_in_the_order_of_the_awaitables(self) -> None:
        gate = asyncio.Event()

        async def second() -> str:
            gate.set()
            return "second"

        assert await gather_or_cancel([_wait_and_return(gate, "first"), second()]) == ["first", "second"]

    async def test_failure_is_raised_as_is_after_the_others_end(self) -> None:
        """The failing awaitable's own error (no ExceptionGroup), raised only once each other
        awaitable was cancelled and has ended: no task is left running."""
        error = RuntimeError("fetch failed")
        log: list[str] = []

        async def fails() -> None:
            raise error

        others = asyncio.all_tasks()
        with pytest.raises(RuntimeError) as exc_info:
            await gather_or_cancel(
                [self._wait_until_cancelled("a", log), self._wait_until_cancelled("b", log), fails()]
            )
        assert exc_info.value is error
        assert log == ["a started", "b started", "a ended", "b ended"]
        assert asyncio.all_tasks() - others == set()

    async def test_cancel_of_the_caller_ends_each_awaitable_first(self) -> None:
        log: list[str] = []
        both_started = asyncio.Event()

        async def second() -> None:
            both_started.set()  # the awaitables start in their order: "a" has started
            await self._wait_until_cancelled("b", log)

        call = asyncio.create_task(gather_or_cancel([self._wait_until_cancelled("a", log), second()]))
        await both_started.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert log == ["a started", "b started", "a ended", "b ended"]


class TestRunToEnd:
    async def test_returns_the_result_of_the_call(self) -> None:
        assert await run_to_end(lambda: "ok") == "ok"

    async def test_cancel_waits_for_the_job(self) -> None:
        """A cancelled caller gets its CancelledError only once the worker job has ended."""
        started, release, finished = threading.Event(), threading.Event(), threading.Event()

        def job() -> None:
            started.set()
            release.wait(10)
            finished.set()

        caller = asyncio.create_task(run_to_end(job))
        assert await asyncio.to_thread(started.wait, 5)
        caller.cancel()
        await asyncio.sleep(0)  # the caller takes the cancel: a caller that does not wait ends here
        assert not caller.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert finished.is_set()

    def test_loop_shutdown_lets_the_job_end_before_the_owner_cleans_up(self) -> None:
        """asyncio.run cancels every task left at its end. A job run as a task (to_thread)
        is cancelled then too, while its thread still runs, and its owner cleans up first;
        the executor future of run_to_end is not a task. The job ends once the owner has
        the cancel of asyncio.run."""
        events: list[str] = []
        started = threading.Event()
        owners: list[asyncio.Task[None]] = []

        def job() -> None:
            started.set()
            deadline = time.monotonic() + 10  # only bounds a failure
            while not owners[0].cancelling() and time.monotonic() < deadline:
                time.sleep(0.001)
            events.append("job ended")

        async def owner() -> None:
            try:
                await run_to_end(job)
            finally:
                events.append("owner cleanup")

        async def main() -> None:
            owners.append(asyncio.create_task(owner()))  # left running: asyncio.run cancels it
            assert await asyncio.to_thread(started.wait, 5)

        asyncio.run(main())
        assert owners[0].cancelled()
        assert events == ["job ended", "owner cleanup"]


class TestRunStoppable:
    """run_stoppable: a cancel sets the job's stop event, waits for the job's end, then raises."""

    async def test_cancel_stops_the_job_and_waits_for_it(self) -> None:
        started = threading.Event()
        ended: list[bool] = []
        undone: list[str] = []

        def job(stop: threading.Event) -> str:
            started.set()
            stop.wait(5)  # a safe point: returns at once once stop is set
            ended.append(stop.is_set())
            return "made"

        task = asyncio.create_task(run_stoppable(job, undo=undone.append))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ended == [True]  # it saw the stop, and had ended before the cancel was raised
        assert undone == ["made"]  # the result of a job that ended after its call was cancelled

    async def test_result_without_a_cancel(self) -> None:
        assert await run_stoppable(lambda _stop: 7, undo=None) == 7


class TestAsyncRetrying:
    """The retry loop of the package: tenacity's, except that an attempt that raises
    StopAsyncIteration fails instead of ending the loop as a success."""

    @staticmethod
    def _loop() -> AsyncRetrying:
        return AsyncRetrying(retry=retry_if_exception_type(OSError), stop=stop_after_attempt(3), wait=wait_none())

    async def test_stop_async_iteration_in_an_attempt_raises(self) -> None:
        attempts = 0
        with pytest.raises(
            RuntimeError,
            check=lambda e: isinstance(e.__cause__, StopAsyncIteration),
            match=r"^A retried call raised StopAsyncIteration$",
        ):
            async for attempt in self._loop():
                with attempt:
                    attempts += 1
                    raise StopAsyncIteration
        assert attempts == 1

    async def test_retried_errors_then_success_end_the_loop(self) -> None:
        attempts = 0
        async for attempt in self._loop():
            with attempt:
                attempts += 1
                if attempts < 3:
                    raise OSError("refused")
        assert attempts == 3

    async def test_other_errors_keep_their_type(self) -> None:
        with pytest.raises(ValueError, match=r"^bad$"):
            async for attempt in self._loop():
                with attempt:
                    raise ValueError("bad")

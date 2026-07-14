"""Unit tests for the cancellation-safe task ownership primitives.

These helpers back every owned-teardown loop in the codebase, so their
truth table is pinned directly: settled / cancelled-then-settled /
task-cancelled / task-error / cancellation + task-error (chained).
"""

import asyncio

import pytest

from exec_sandbox.aio_utils import await_cancellation_safe, await_settled


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

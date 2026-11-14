"""Cancellation-safe task ownership primitives, and the retry loop.

Many owners in this codebase must bring an operation to a settled state even
while the caller is being cancelled — abandoning it would orphan a child
process, a temp directory, or a shared transport. The common shape is
"await the task through repeated CancelledError, then decide"; these helpers
hold that loop, so no caller writes it again.
"""

import asyncio
import contextlib
import threading
from collections.abc import Awaitable, Callable, Iterable

import tenacity


class AsyncRetrying(tenacity.AsyncRetrying):
    """tenacity.AsyncRetrying, except that an attempt that raises StopAsyncIteration fails.

    tenacity raises an attempt's exception from ``__anext__``, where a StopAsyncIteration
    ends the ``async for`` as if the attempt had succeeded: the code after the loop then
    runs on a call that did not finish (a guest "ready" with no pong). Python turns it
    into a RuntimeError in an async generator (PEP 525); this loop does the same.
    """

    _attempt: tenacity.AttemptManager | None = None

    async def __anext__(self) -> tenacity.AttemptManager:
        try:
            self._attempt = await super().__anext__()
        except StopAsyncIteration:
            outcome = None if self._attempt is None else self._attempt.retry_state.outcome
            if outcome is not None and isinstance(outcome.exception(), StopAsyncIteration):
                raise RuntimeError("A retried call raised StopAsyncIteration") from outcome.exception()
            raise
        return self._attempt


async def await_settled[T](task: asyncio.Task[T]) -> asyncio.CancelledError | None:
    """Await ``task`` until it settles, absorbing repeated caller cancellation.

    Returns the first :class:`asyncio.CancelledError` observed while waiting
    (caller cancellation, or third-party cancellation of the shielded task —
    the two are indistinguishable here) or ``None``. The task's own outcome is
    never raised here — on return the task is done and the caller inspects it
    via ``task.result()`` / ``task.exception()`` / ``task.cancelled()``.
    """
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        except BaseException:  # noqa: BLE001 - surfaced via task inspection by the caller
            break
    return cancellation


async def settle_and_report(task: asyncio.Task[object]) -> BaseException | None:
    """Settle a rollback task (absorbing caller cancellation) and report its
    own error, if any.

    A task cancelled from outside is treated as clean (returns ``None``): a
    cancelled rollback is not itself a failure to log. Caller cancellation is
    absorbed (a rollback must run to its end); the caller's ``except`` block
    then re-raises its primary error.
    """
    await await_settled(task)
    return None if task.cancelled() else task.exception()


async def await_cancellation_safe[T](awaitable: Awaitable[T]) -> T:
    """Run ``awaitable`` to completion even if the caller is cancelled.

    Caller cancellation is deferred until the operation settles, then
    re-raised (chained to the operation's own error, if any) so structured
    cancellation still observes CancelledError. Without cancellation this
    returns the result or raises the operation's error. Same caveat as
    await_settled: third-party cancellation of the operation is
    indistinguishable from caller cancellation and re-raises the same way.
    """
    task = asyncio.ensure_future(awaitable)
    cancellation = await await_settled(task)
    if cancellation is not None:
        if not task.cancelled() and (error := task.exception()) is not None:
            raise cancellation from error
        raise cancellation
    return task.result()


async def gather_or_cancel[T](awaitables: Iterable[Awaitable[T]]) -> list[T]:
    """Run ``awaitables`` at the same time; return their results in order.

    On the first failure (a cancel of the caller too), cancel the others, wait until each
    one ends, then raise that failure as it was raised. Unlike asyncio.gather, nothing runs
    on after the call; unlike asyncio.TaskGroup, the error is not wrapped in an
    ExceptionGroup.
    """
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def run_to_end[T](call: Callable[[], T]) -> T:
    """Run a blocking ``call`` in a worker thread to its end, even if the caller is
    cancelled (await_cancellation_safe). An executor future, not a task: a loop
    shutdown cancels every task, never this job, so the caller's cleanup runs after it.
    """
    return await await_cancellation_safe(asyncio.get_running_loop().run_in_executor(None, call))


async def run_stoppable[T](job: Callable[[threading.Event], T], *, undo: Callable[[T], None] | None) -> T:
    """Run `job(stop)` in a worker thread. A cancel sets `stop` (the job stops at its next
    safe point and removes its own files), waits until the job ends, then raises: no job
    runs on after its call, nor holds what its caller frees next (a lock, an fd). `undo`
    gets the result of a job that ended after its call was cancelled. For a long job (a
    copy of GBs) where run_to_end would make a cancel wait for all of it."""
    stop = threading.Event()
    future = asyncio.get_running_loop().run_in_executor(None, job, stop)
    try:
        return await asyncio.shield(future)  # shield marks a later job error as read
    except asyncio.CancelledError:
        stop.set()
        while not future.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait([future])
        if undo is not None and not future.cancelled() and future.exception() is None:
            undo(future.result())
        raise

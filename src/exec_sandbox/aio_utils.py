"""Cancellation-safe task ownership primitives.

Many owners in this codebase must bring an operation to a settled state even
while the caller is being cancelled — abandoning it would orphan a child
process, a temp directory, or a shared transport. The canonical shape is
"await the task through repeated CancelledError, then decide"; these helpers
replace the hand-rolled copies of that loop.
"""

import asyncio
from collections.abc import Awaitable


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
    absorbed — a wedged rollback must not be abortable — and re-raised by the
    ``except`` block that owns the primary error.
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

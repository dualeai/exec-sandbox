"""Resource cleanup utilities for VM lifecycle management.

Defensive cleanup operations that log errors but don't fail.
Used by VmManager, DiskSnapshotManager (L2 builds), qemu-storage-daemon, gvproxy and
subprocess_utils to end their child processes.
"""

import asyncio
import contextlib
from typing import Any

from exec_sandbox._logging import get_logger
from exec_sandbox.platform_utils import ProcessWrapper

logger = get_logger(__name__)

# Module-level set to hold references to background zombie reaping tasks
# Prevents garbage collection of fire-and-forget tasks
# See: https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
_zombie_reap_tasks: set[asyncio.Task[Any]] = set()


def _cleanup_zombie_task(task: asyncio.Task[Any]) -> None:
    """Remove completed task from tracking set and log any exceptions."""
    _zombie_reap_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning(
            "Zombie reaping task failed",
            extra={"error": str(task.exception())},
        )


async def cleanup_process(
    proc: ProcessWrapper | None,
    name: str,
    context_id: str,
    term_timeout: float = 3.0,
    kill_timeout: float = 2.0,
    *,
    kill_first: bool,
) -> bool:
    """Force cleanup of subprocess (SIGTERM → SIGKILL).

    - A process that already ended is reaped by a background task (no zombie)
    - SIGTERM, wait up to term_timeout, then SIGKILL, wait up to kill_timeout; with
      kill_first, SIGKILL at once (no clean exit needed)
    - A process still alive after SIGKILL is reaped by a background task (up to 30 s)
    - ProcessLookupError (the process ended meanwhile) counts as success
    - Never raises an exception (logs it and returns False); a cancellation propagates
    - Uses ProcessWrapper for PID-reuse safety and async-only operations

    Args:
        proc: ProcessWrapper to kill (None safe - returns immediately)
        name: Process name for logging (e.g., "gvproxy", "QEMU", "qemu-storage-daemon")
        context_id: Context for logging (e.g., vm_id, tenant_id)
        term_timeout: Seconds to wait after SIGTERM before SIGKILL
        kill_timeout: Seconds to wait after SIGKILL before giving up
        kill_first: SIGKILL with no SIGTERM before (a process whose clean exit keeps nothing)

    Returns:
        True if process cleaned successfully, False if issues occurred
    """
    if proc is None:
        return True

    try:
        # Check if already dead (prevents unnecessary operations)
        if proc.returncode is not None:
            logger.debug(
                f"{name} already terminated",
                extra={"context_id": context_id, "returncode": proc.returncode},
            )
            # Ensure zombie is reaped - track task to prevent GC
            task = asyncio.create_task(proc.wait())
            _zombie_reap_tasks.add(task)
            task.add_done_callback(_cleanup_zombie_task)
            return True

        if not kill_first:
            # Phase 1: SIGTERM (graceful shutdown) - async, non-blocking
            logger.debug(f"Sending SIGTERM to {name}", extra={"context_id": context_id})
            await proc.terminate()

            try:
                await proc.wait_with_timeout(timeout=term_timeout)
                logger.debug(
                    f"{name} stopped gracefully (SIGTERM)",
                    extra={"context_id": context_id, "returncode": proc.returncode},
                )
                return True
            except TimeoutError:
                logger.warning(
                    f"{name} didn't respond to SIGTERM, force killing",
                    extra={"context_id": context_id, "term_timeout": term_timeout},
                )

        # Phase 2: SIGKILL (force kill) - async, non-blocking
        logger.debug(f"Sending SIGKILL to {name}", extra={"context_id": context_id})
        await proc.kill()

        try:
            await proc.wait_with_timeout(timeout=kill_timeout)
            (logger.debug if kill_first else logger.warning)(
                f"{name} stopped (SIGKILL)" if kill_first else f"{name} force killed (SIGKILL)",
                extra={"context_id": context_id, "returncode": proc.returncode},
            )
            return True
        except TimeoutError:
            logger.error(
                f"{name} didn't respond to SIGKILL within timeout",
                extra={"context_id": context_id, "kill_timeout": kill_timeout, "pid": proc.pid},
            )

            # Still try to reap in background to prevent zombie - track task to prevent GC
            async def drain_and_wait() -> None:
                with contextlib.suppress(OSError, TimeoutError, asyncio.CancelledError):
                    await proc.wait_with_timeout(timeout=30)  # 30s timeout for zombie reaping

            task = asyncio.create_task(drain_and_wait())
            _zombie_reap_tasks.add(task)
            task.add_done_callback(_cleanup_zombie_task)
            return False

    except ProcessLookupError:
        # Process already dead (race condition between check and kill)
        logger.debug(f"{name} already dead (ProcessLookupError)", extra={"context_id": context_id})
        return True

    except Exception as e:
        # Never raise - log and return failure
        logger.error(
            f"{name} cleanup error",
            extra={"context_id": context_id, "error": str(e), "error_type": type(e).__name__},
            exc_info=True,
        )
        return False

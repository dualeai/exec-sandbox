"""Resource cleanup utilities for VM lifecycle management.

Defensive cleanup operations that log errors but don't fail.
Used by VmManager (QEMU, gvproxy), DiskSnapshotManager (L2 builds), QemuStorageDaemon and
subprocess_utils (communicate_managed_process) to end their child processes.
"""

import logging
from typing import Final

from exec_sandbox.platform_utils import ProcessWrapper

logger = logging.getLogger(__name__)

_KILL_TIMEOUT_SECONDS: Final = 2.0
"""How long a SIGKILLed process may take to end: a process still alive then is stuck (D state)."""


async def cleanup_process(
    proc: ProcessWrapper | None,
    name: str,
    context_id: str,
    *,
    term_timeout: float,
    kill_first: bool,
) -> bool:
    """Force cleanup of subprocess (SIGTERM → SIGKILL).

    - A process with a returncode is already reaped (asyncio sets it after waitpid)
    - SIGTERM, wait up to term_timeout, then SIGKILL, wait up to _KILL_TIMEOUT_SECONDS; with
      kill_first, SIGKILL at once (no clean exit needed)
    - A process still alive after SIGKILL: False; the event loop's child watcher reaps it
      when it ends (every child of create_subprocess_exec), no task here waits for it
    - ProcessLookupError (the process ended meanwhile, or ProcessWrapper.is_running can no
      longer see it) counts as success
    - Never raises an exception (logs it and returns False); a cancellation propagates
    - Uses ProcessWrapper for PID-reuse safety and async-only operations

    Args:
        proc: ProcessWrapper to kill (None safe - returns immediately)
        name: Process name for logging (e.g., "gvproxy", "QEMU", "qemu-storage-daemon")
        context_id: Context for logging (e.g., vm_id, tenant_id)
        term_timeout: Seconds to wait after SIGTERM before SIGKILL
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
            await proc.wait_with_timeout(timeout=_KILL_TIMEOUT_SECONDS)
            (logger.debug if kill_first else logger.warning)(
                f"{name} stopped (SIGKILL)" if kill_first else f"{name} force killed (SIGKILL)",
                extra={"context_id": context_id, "returncode": proc.returncode},
            )
            return True
        except TimeoutError:
            logger.error(  # the loop's child watcher reaps it when it ends
                f"{name} didn't respond to SIGKILL within timeout",
                extra={"context_id": context_id, "kill_timeout": _KILL_TIMEOUT_SECONDS, "pid": proc.pid},
            )
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

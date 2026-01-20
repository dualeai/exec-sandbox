"""Process registry for tracking child processes.

Enables force-kill of all child processes on second Ctrl+C.
Used by VmManager to register QEMU/gvproxy PIDs and by CLI for emergency cleanup.

Thread-safe: Uses a simple set with GIL protection (adequate for signal handlers).
"""

import os
import signal
from typing import TYPE_CHECKING

from exec_sandbox._logging import get_logger

if TYPE_CHECKING:
    from exec_sandbox.platform_utils import ProcessWrapper

logger = get_logger(__name__)

# Module-level set of process group IDs (PIDs of processes started with start_new_session=True)
# These processes are group leaders, so os.killpg(pid, signal) kills the entire group
_process_groups: set[int] = set()


def register_process(proc: "ProcessWrapper | None") -> None:
    """Register a process for tracking.

    Args:
        proc: ProcessWrapper to track (None safe - returns immediately)
    """
    if proc is None or proc.pid is None:
        return
    _process_groups.add(proc.pid)
    logger.debug("Registered process group", extra={"pid": proc.pid})


def unregister_process(proc: "ProcessWrapper | None") -> None:
    """Unregister a process from tracking.

    Args:
        proc: ProcessWrapper to untrack (None safe - returns immediately)
    """
    if proc is None or proc.pid is None:
        return
    _process_groups.discard(proc.pid)
    logger.debug("Unregistered process group", extra={"pid": proc.pid})


def force_kill_all() -> int:
    """Force kill all tracked process groups with SIGKILL.

    Uses os.killpg() to kill entire process groups (works because processes
    are started with start_new_session=True, making them group leaders).

    Returns:
        Number of process groups signaled
    """
    killed = 0
    for pid in list(_process_groups):
        try:
            os.killpg(pid, signal.SIGKILL)
            killed += 1
            logger.debug("Force killed process group", extra={"pid": pid})
        except ProcessLookupError:
            # Process already dead
            pass
        except PermissionError:
            logger.warning("Permission denied killing process group", extra={"pid": pid})
        except OSError as e:
            logger.warning("Error killing process group", extra={"pid": pid, "error": str(e)})
    _process_groups.clear()
    return killed


def get_tracked_count() -> int:
    """Get number of tracked process groups."""
    return len(_process_groups)

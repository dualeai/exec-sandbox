"""Subprocess output handling utilities.

Provides concurrent stdout/stderr draining to prevent pipe buffer deadlocks.
Follows best practices from Python 3.11+ asyncio.TaskGroup pattern.
"""

import asyncio
from collections.abc import Callable

import aiofiles
import aiofiles.os

from exec_sandbox._logging import get_logger
from exec_sandbox.platform_utils import ProcessWrapper

logger = get_logger(__name__)


async def read_log_tail(path: str, max_bytes: int) -> str:
    """
    Read the last max_bytes of a log file for debugging.

    Shows the most recent output which is typically most relevant for debugging
    failures (e.g., the error that caused the crash, not early boot messages).

    Args:
        path: Path to the log file
        max_bytes: Maximum bytes to read from end of file

    Returns:
        The last max_bytes of file content, or error/status message
    """
    if not await aiofiles.os.path.exists(path):
        return "(file not found)"

    try:
        async with aiofiles.open(path) as f:
            content = await f.read()
        if not content:
            return "(empty)"
        return content[-max_bytes:]
    except OSError as e:
        return f"(failed to read: {e})"


async def drain_subprocess_output(
    process: ProcessWrapper,
    *,
    process_name: str,
    context_id: str,
    stdout_handler: Callable[[str], None] | None = None,
    stderr_handler: Callable[[str], None] | None = None,
) -> None:
    """Drain subprocess stdout/stderr concurrently to prevent 64KB pipe deadlock.

    Critical: Without concurrent draining, subprocess can deadlock when:
    1. Process writes to both stdout and stderr
    2. One pipe fills (64KB buffer)
    3. Process blocks waiting for read
    4. Sequential reader stuck on other pipe

    Architecture (Nov 2025):
    - Uses asyncio.TaskGroup for concurrent reading (Python 3.11+)
    - Matches Tokio spawned task pattern in Rust (guest-agent/src/main.rs)
    - Prevents pipe buffer exhaustion in long-running processes

    Args:
        process: ProcessWrapper instance with stdout/stderr pipes
        process_name: Process identifier for logging (e.g., "QEMU", "gvproxy")
        context_id: Context identifier (e.g., vm_id) for log correlation
        stdout_handler: Optional callback for stdout lines (default: debug log)
        stderr_handler: Optional callback for stderr lines (default: warning log)

    Example:
        >>> proc = await asyncio.create_subprocess_exec(
        ...     "qemu-system-x86_64", ...,
        ...     stdout=asyncio.subprocess.PIPE,
        ...     stderr=asyncio.subprocess.PIPE,
        ... )
        >>> task = asyncio.create_task(drain_subprocess_output(
        ...     proc, process_name="QEMU", context_id=vm_id
        ... ))
        >>> # Later: task.cancel() and await task during cleanup
    """

    # Default handlers: structured logging
    if stdout_handler is None:

        def default_stdout_handler(line: str) -> None:
            logger.debug(f"[{process_name} stdout] {line}", extra={"context_id": context_id, "output": line})

        stdout_handler = default_stdout_handler

    if stderr_handler is None:

        def default_stderr_handler(line: str) -> None:
            logger.warning(f"[{process_name} stderr] {line}", extra={"context_id": context_id, "output": line})

        stderr_handler = default_stderr_handler

    # Concurrent reading with TaskGroup (Python 3.11+)
    async with asyncio.TaskGroup() as tg:

        async def read_stdout() -> None:
            """Read stdout until EOF."""
            if process.stdout:
                async for line in process.stdout:
                    try:
                        decoded = line.decode().rstrip()
                        if decoded:
                            stdout_handler(decoded)
                    except (UnicodeDecodeError, ValueError):
                        pass  # Ignore decode errors - non-UTF8 output is silently skipped

        async def read_stderr() -> None:
            """Read stderr until EOF."""
            if process.stderr:
                async for line in process.stderr:
                    try:
                        decoded = line.decode().rstrip()
                        if decoded:
                            stderr_handler(decoded)
                    except (UnicodeDecodeError, ValueError):
                        pass  # Ignore decode errors - non-UTF8 output is silently skipped

        # Launch concurrent readers (prevent deadlock)
        if process.stdout:
            tg.create_task(read_stdout())
        if process.stderr:
            tg.create_task(read_stderr())


def log_task_exception(task: asyncio.Task[None]) -> None:
    """Log exceptions from background tasks.

    Callback for asyncio.Task.add_done_callback() that properly logs any
    unhandled exceptions from background tasks. Prevents silent failures.

    Usage:
        task = asyncio.create_task(some_coroutine())
        task.add_done_callback(log_task_exception)

    Args:
        task: The completed asyncio task to check for exceptions
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Background task failed",
            extra={"task_name": task.get_name()},
            exc_info=exc,
        )

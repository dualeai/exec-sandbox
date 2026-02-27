"""Subprocess lifecycle utilities.

- drain_subprocess_output: concurrent stdout/stderr draining (prevents 64KB pipe deadlock)
- wait_for_socket: poll for a Unix socket created by a child process after fork+exec
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, overload

from exec_sandbox._logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from exec_sandbox.platform_utils import ProcessWrapper

logger = get_logger(__name__)


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


@overload
async def wait_for_socket(
    path: Path,
    *,
    timeout: float,
    poll_interval: float = ...,
    abort_check: Callable[[], None] | None = ...,
    keep_connection: Literal[True],
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]: ...


@overload
async def wait_for_socket(
    path: Path,
    *,
    timeout: float,
    poll_interval: float = ...,
    abort_check: Callable[[], None] | None = ...,
    keep_connection: Literal[False] = ...,
) -> None: ...


async def wait_for_socket(
    path: Path,
    *,
    timeout: float,
    poll_interval: float = 0.002,
    abort_check: Callable[[], None] | None = None,
    keep_connection: bool = False,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
    """Wait for a Unix socket to appear and accept connections.

    Used after fork+exec of QEMU or QSD to wait for the process to create its
    QMP control socket. Polls the filesystem because there is no async event
    to await (the file is created by an external process).

    Two-phase wait: first the socket file must exist, then a probe-connect
    must succeed. This closes the TOCTOU gap between file creation and
    listen() completion that caused spurious ConnectionRefusedError under load.

    Args:
        path: Path to the socket file.
        timeout: Maximum seconds to wait before raising TimeoutError.
        poll_interval: Seconds between checks (default 2ms).
        abort_check: Optional callable invoked each poll iteration. Should raise
            to abort the wait early (e.g. when the spawning process has died).
        keep_connection: If True, return the connected (reader, writer) streams
            instead of closing the probe connection. Eliminates the TOCTOU gap
            between probe-close and the caller's real connect for single-client
            chardev sockets (e.g. QMP).

    Returns:
        ``(reader, writer)`` when *keep_connection* is True, else ``None``.

    Raises:
        TimeoutError: Socket did not appear or accept connections within *timeout* seconds.
    """
    async with asyncio.timeout(timeout):
        # Phase 1: wait for socket file to exist
        while not path.exists():
            if abort_check is not None:
                abort_check()
            await asyncio.sleep(poll_interval)

        # Phase 2: verify socket accepts connections
        while True:
            if abort_check is not None:
                abort_check()
            try:
                r, w = await asyncio.open_unix_connection(str(path))
                if keep_connection:
                    return r, w
                w.close()
                await w.wait_closed()
                return None
            except (ConnectionRefusedError, ConnectionResetError):
                await asyncio.sleep(poll_interval)

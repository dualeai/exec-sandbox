"""Subprocess lifecycle utilities.

- start_managed_process: launch a subprocess with standard piping and process registration
- drain_subprocess_output: concurrent stdout/stderr draining (prevents 64KB pipe deadlock)
- wait_for_socket: poll for a Unix socket created by a child process after fork+exec
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, overload

from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import await_settled
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.process_registry import register_process, unregister_process
from exec_sandbox.resource_cleanup import cleanup_process

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

logger = get_logger(__name__)


async def start_managed_process(
    cmd: Sequence[str],
    *,
    pass_fds: tuple[int, ...] = (),
    preexec_fn: Callable[[], None] | None = None,
    on_process_started: Callable[[ProcessWrapper], None] | None = None,
) -> ProcessWrapper:
    """Launch a subprocess with standard piping and process registration.

    Common boilerplate for all subprocesses (QEMU, gvproxy, QSD, qemu-img):
    - Wraps asyncio subprocess with ProcessWrapper for PID-reuse safety
    - Pipes stdout/stderr (prevents 64 KiB deadlock when paired with drain_subprocess_output)
    - Registers in process_registry for emergency Ctrl+C kill (prevents orphans)
    - Creates new session (isolates signal delivery, enables killpg)

    Callers handle their own error wrapping (VmDependencyError, VmGvproxyError, etc.)
    and domain-specific post-launch steps (cgroup attachment, socket wait, etc.).

    Args:
        cmd: Command and arguments.
        pass_fds: File descriptors to inherit (socket activation).
        preexec_fn: Function called in child between fork and exec.

    Returns:
        ProcessWrapper instance with stdout/stderr pipes.

    Raises:
        OSError: Process spawn failed.
        FileNotFoundError: Binary not found.
    """

    async def spawn_and_publish() -> ProcessWrapper:
        proc = ProcessWrapper(
            await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                preexec_fn=preexec_fn,
                pass_fds=pass_fds,
            )
        )
        # Registration and the optional domain owner callback are synchronous:
        # no cancellation point can separate child creation from publication.
        register_process(proc)
        if on_process_started is not None:
            on_process_started(proc)
        return proc

    # asyncio.create_subprocess_exec() can create the OS child just before its
    # awaiting caller is cancelled. Keep spawn in a manager-owned task and defer
    # cancellation propagation until that task has either published the child or
    # failed. This prevents an unregistered/unowned process at the await boundary.
    spawn_task = asyncio.create_task(spawn_and_publish())
    try:
        return await asyncio.shield(spawn_task)
    except asyncio.CancelledError:
        await await_settled(spawn_task)
        if not spawn_task.cancelled():
            # Retrieve any spawn exception so the manager-owned task cannot emit
            # an unhandled-task warning; the caller's cancellation stays primary.
            spawn_task.exception()
        raise


async def communicate_managed_process(
    cmd: Sequence[str],
    *,
    process_name: str,
    context_id: str,
    pass_fds: tuple[int, ...] = (),
    preexec_fn: Callable[[], None] | None = None,
) -> tuple[int | None, bytes, bytes]:
    """Run a transient child with cancellation-safe kill/reap/unregistration."""

    async def run_and_cleanup() -> tuple[int | None, bytes, bytes]:
        owned_process: ProcessWrapper | None = None

        def publish_process(proc: ProcessWrapper) -> None:
            nonlocal owned_process
            owned_process = proc

        try:
            proc = await start_managed_process(
                cmd,
                pass_fds=pass_fds,
                preexec_fn=preexec_fn,
                on_process_started=publish_process,
            )
            stdout, stderr = await proc.communicate()
            return proc.returncode, stdout, stderr
        finally:
            if owned_process is not None:
                delay = 1.0
                while True:
                    try:
                        if await cleanup_process(owned_process, process_name, context_id):
                            unregister_process(owned_process)
                            break
                    except asyncio.CancelledError:
                        # Cancellation may land after communicate() completed
                        # but while cleanup is in flight. Keep this owned task
                        # alive until death is confirmed; the outer waiter still
                        # propagates its own cancellation.
                        continue
                    except Exception:
                        logger.exception(
                            "Transient process cleanup attempt failed",
                            extra={"process_name": process_name, "context_id": context_id},
                        )
                    try:
                        await asyncio.sleep(delay)
                    except asyncio.CancelledError:
                        continue
                    delay = min(delay * 2, 30.0)

    operation = asyncio.create_task(run_and_cleanup())
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        operation.cancel()
        await await_settled(operation)
        if not operation.cancelled():
            operation.exception()
        raise


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

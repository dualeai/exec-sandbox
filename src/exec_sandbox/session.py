"""Session - persistent VM for multi-step code execution.

A Session keeps a VM alive across multiple exec() calls, enabling
stateful workflows where variables, imports, and functions persist
between executions. The underlying REPL process maintains state.

Example:
    ```python
    async with Scheduler() as scheduler:
        async with await scheduler.session(language="python") as session:
            await session.exec("x = 42")
            result = await session.exec("print(x)")
            assert result.stdout.strip() == "42"
    ```

Lifecycle:
    - Created via scheduler.session() (returns Session owning a VM)
    - exec() calls execute code in the same VM (state persists)
    - close() destroys the VM (called automatically by context manager)
    - Idle timeout auto-closes after inactivity
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import (  # noqa: TC003 - Used at runtime for _guard() and on_stdout/on_stderr
    AsyncIterator,
    Callable,
)
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO, TYPE_CHECKING, Self

from exec_sandbox._logging import get_logger
from exec_sandbox.constants import MAX_CODE_SIZE, MAX_FILE_SIZE_BYTES, MAX_TIMEOUT_SECONDS
from exec_sandbox.exceptions import SessionClosedError, VmConfigError
from exec_sandbox.models import ExecutionResult, ExposedPort, FileInfo, TimingBreakdown

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.vm_manager import VmManager

logger = get_logger(__name__)


class Session:
    """Persistent VM session for multi-step code execution.

    Wraps a QemuVM and keeps it alive across multiple exec() calls.
    State (variables, imports, functions) persists between executions
    via the guest agent's persistent REPL process.

    Thread-safety: exec() calls are serialized via asyncio.Lock.
    The guest agent processes one request at a time.

    Attributes:
        closed: Whether the session has been closed.
        vm_id: ID of the underlying VM.
        exec_count: Number of successful exec() calls.
        exposed_ports: Resolved port mappings (internal→external) for this session's VM.
    """

    def __init__(
        self,
        vm: QemuVM,
        vm_manager: VmManager,
        idle_timeout_seconds: int,
        default_timeout_seconds: int,
    ) -> None:
        self._vm = vm
        self._vm_manager = vm_manager
        self._idle_timeout_seconds = idle_timeout_seconds
        self._default_timeout_seconds = default_timeout_seconds
        self._closed = False
        self._exec_count = 0
        self._exec_lock = asyncio.Lock()
        self._idle_timer_task: asyncio.Task[None] | None = None
        self._reset_idle_timer()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        """Whether the session has been closed."""
        return self._closed

    @property
    def vm_id(self) -> str:
        """ID of the underlying VM."""
        return self._vm.vm_id

    @property
    def exec_count(self) -> int:
        """Number of successful exec() calls."""
        return self._exec_count

    @property
    def exposed_ports(self) -> list[ExposedPort]:
        """Resolved port mappings (internal→external) for this session's VM."""
        return self._vm.exposed_ports

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def _guard(self) -> AsyncIterator[None]:
        """Acquire exec lock with session lifecycle checks.

        Encapsulates the 4-step preamble shared by all public methods:
        1. Check closed (fast path, no lock)
        2. Acquire _exec_lock
        3. Re-check closed (may have closed while waiting)
        4. Reset idle timer
        """
        if self._closed:
            raise SessionClosedError("Session is closed")
        async with self._exec_lock:
            if self._closed:
                raise SessionClosedError("Session closed while waiting for lock")
            self._reset_idle_timer()
            yield

    async def _resolve_content(self, content: bytes | Path) -> IO[bytes]:
        """Convert bytes | Path to a readable binary stream with size validation.

        Called BEFORE _guard() so local disk I/O doesn't hold the exec lock.

        Returns:
            IO[bytes]: BytesIO for bytes input, open file handle for Path input.
            Caller is responsible for closing the stream.

        TOCTOU note: for Path, we stat() before open(). The file could grow
        between stat() and actual reads. The guest agent also enforces the
        max file size, providing defense in depth.
        """
        if isinstance(content, Path):
            if not content.exists():
                raise FileNotFoundError(f"Source file not found: {content}")
            file_size = content.stat().st_size
            if file_size > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"File {content.name} is {file_size} bytes, exceeds {MAX_FILE_SIZE_BYTES}")
            # Return open file handle — streams from disk, never loads full content.
            return content.open("rb")
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"Content is {len(content)} bytes, exceeds {MAX_FILE_SIZE_BYTES}")
        return io.BytesIO(content)

    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------

    async def exec(
        self,
        code: str,
        *,
        timeout_seconds: int | None = None,
        env_vars: dict[str, str] | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Execute code in the session VM.

        State (variables, imports, functions) persists from previous exec() calls.
        Non-zero exit codes do NOT close the session - only VM failures do.

        Args:
            code: Source code to execute.
            timeout_seconds: Execution timeout. Default: scheduler's default_timeout_seconds.
            env_vars: Environment variables for this execution.
            on_stdout: Callback for stdout chunks (streaming).
            on_stderr: Callback for stderr chunks (streaming).

        Returns:
            ExecutionResult with stdout, stderr, exit_code, timing info.

        Raises:
            SessionClosedError: Session has been closed.
            VmPermanentError: VM communication failed (session auto-closed).
            VmTransientError: VM communication failed (session auto-closed).
            VmBootTimeoutError: Execution exceeded timeout (session auto-closed).

        Note:
            Execution timeouts (guest agent level) return a result with
            exit_code=-1 and timeout message in stderr — they do NOT raise.
            Only VM-level failures (communication errors) raise exceptions.
        """
        if timeout_seconds is not None and (timeout_seconds < 1 or timeout_seconds > MAX_TIMEOUT_SECONDS):
            raise ValueError(f"timeout_seconds must be between 1 and {MAX_TIMEOUT_SECONDS}, got {timeout_seconds}")
        if len(code) > MAX_CODE_SIZE:
            raise VmConfigError(
                f"Code too large: {len(code)} bytes exceeds {MAX_CODE_SIZE} byte limit",
                context={"code_size": len(code), "max_code_size": MAX_CODE_SIZE},
            )
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds

        async with self._guard():
            execute_start = asyncio.get_running_loop().time()
            try:
                result = await self._vm.execute(
                    code=code,
                    timeout_seconds=timeout,
                    env_vars=env_vars,
                    on_stdout=on_stdout,
                    on_stderr=on_stderr,
                )
            except Exception:
                # VM failure - auto-close session
                await self.close()
                raise

            execute_end = asyncio.get_running_loop().time()
            execute_ms = round((execute_end - execute_start) * 1000)

            self._exec_count += 1

            # Populate timing with session-measured values
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                execution_time_ms=result.execution_time_ms,
                external_cpu_time_ms=result.external_cpu_time_ms,
                external_memory_peak_mb=result.external_memory_peak_mb,
                timing=TimingBreakdown(
                    setup_ms=0,
                    boot_ms=0,
                    execute_ms=execute_ms,
                    total_ms=execute_ms,
                    connect_ms=result.timing.connect_ms,
                ),
                spawn_ms=result.spawn_ms,
                process_ms=result.process_ms,
            )

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    async def write_file(self, path: str, content: bytes | Path, *, make_executable: bool = False) -> None:
        """Write a file to the sandbox at the given path.

        Args:
            path: Relative path in sandbox (e.g., "input.csv", "data/model.pkl").
            content: File content as bytes or a local Path to read from.
                For Path, content is streamed from disk — never fully loaded.
            make_executable: Set executable permission (0o755 vs 0o644).

        Raises:
            SessionClosedError: Session has been closed.
            ValueError: Content exceeds max file size limit.
            FileNotFoundError: Path source does not exist.
            VmPermanentError: Guest agent validation or write failure.
        """
        # Resolve content to a stream BEFORE acquiring the lock —
        # local disk I/O should not block other session operations.
        stream = await self._resolve_content(content)
        try:
            async with self._guard():
                await self._vm.write_file(path, stream, make_executable=make_executable)
        finally:
            stream.close()

    async def read_file(self, path: str, *, destination: Path) -> None:
        """Read a file from the sandbox, streaming directly to a local file.

        Decompressed chunks are written directly to *destination* — peak
        memory is ~128 KB regardless of file size.

        Args:
            path: Relative path in sandbox (e.g., "output.csv").
            destination: Local file path to stream content into.

        Raises:
            SessionClosedError: Session has been closed.
            VmPermanentError: File not found or validation failure.
        """
        async with self._guard():
            await self._vm.read_file(path, destination=destination)

    async def list_files(self, path: str = "") -> list[FileInfo]:
        """List files in a directory in the sandbox.

        Args:
            path: Relative path (empty string for sandbox root).

        Returns:
            List of FileInfo entries with name, is_dir, and size.

        Raises:
            SessionClosedError: Session has been closed.
            VmPermanentError: Directory not found or validation failure.
        """
        async with self._guard():
            return await self._vm.list_files(path)

    async def close(self) -> None:
        """Close the session and destroy the VM.

        Idempotent: safe to call multiple times.

        Concurrency contract: close() does NOT acquire _exec_lock.
        Setting _closed=True is an atomic flag that _guard() checks at
        two points (before and after lock acquisition). An in-flight
        operation that already passed _guard() will complete its VM call
        before the lock is released. The next operation will see _closed=True.
        """
        if self._closed:
            return

        self._closed = True
        self._cancel_idle_timer()

        logger.info(
            "Closing session",
            extra={
                "vm_id": self._vm.vm_id,
                "exec_count": self._exec_count,
            },
        )

        try:
            await self._vm_manager.destroy_vm(self._vm)
        except (OSError, RuntimeError, TimeoutError) as e:
            logger.error(
                "Error destroying session VM",
                extra={"vm_id": self._vm.vm_id, "error": str(e)},
            )

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        """Exit async context manager - ensure cleanup."""
        await self.close()

    # -------------------------------------------------------------------------
    # Idle Timeout
    # -------------------------------------------------------------------------

    def _reset_idle_timer(self) -> None:
        """Reset the idle timeout timer."""
        self._cancel_idle_timer()
        self._idle_timer_task = asyncio.create_task(self._idle_timeout_handler())

    def _cancel_idle_timer(self) -> None:
        """Cancel the idle timeout timer."""
        if self._idle_timer_task is not None:
            if not self._idle_timer_task.done():
                self._idle_timer_task.cancel()
            self._idle_timer_task = None

    async def _idle_timeout_handler(self) -> None:
        """Auto-close session after idle timeout."""
        try:
            await asyncio.sleep(self._idle_timeout_seconds)
        except asyncio.CancelledError:
            return

        logger.info(
            "Session idle timeout",
            extra={
                "vm_id": self._vm.vm_id,
                "idle_timeout_seconds": self._idle_timeout_seconds,
                "exec_count": self._exec_count,
            },
        )
        await self.close()

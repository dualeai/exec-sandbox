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
    - Idle timeout auto-closes after inactivity (time between operations)
    - Failure closes: cancellation, unknown-outcome transport loss, exit 137,
      and a VM found dead are all terminal — see Session.exec and _guard()
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
from exec_sandbox.constants import MAX_CODE_SIZE, MAX_FILE_SIZE_BYTES, MAX_TIMEOUT_SECONDS, SIGKILL_EXIT_CODE
from exec_sandbox.exceptions import (
    CommunicationOutcomeUnknownError,
    InputValidationError,
    OutputLimitError,
    SessionClosedError,
    VmConfigError,
    VmPermanentError,
)
from exec_sandbox.models import ExecutionResult, ExposedPort, FileInfo, TimingBreakdown
from exec_sandbox.vm_types import VmState

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
        self._close_task: asyncio.Task[None] | None = None
        self._exec_count = 0
        self._exec_lock = asyncio.Lock()
        # loop.call_later handle for the idle deadline. cancel() is synchronous,
        # allocation-light, and a no-op once fired — no task, no CancelledError
        # to process, so no strong-ref set is needed.
        self._idle_timer_handle: asyncio.TimerHandle | None = None
        # Strong ref to the fire-and-forget close task the timer callback spawns.
        self._idle_close_task: asyncio.Task[None] | None = None
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

        Encapsulates the preamble shared by all public methods:
        1. Check closed (fast path, no lock)
        2. Acquire _exec_lock
        3. Re-check closed (may have closed while waiting)
        4. Retire fast if the VM process is already dead or being destroyed
        5. Suspend the idle timer for the duration of the operation

        Idle means "no operation in flight": the timer is cancelled while an
        operation runs and re-armed when it finishes, so a long exec can never
        be destroyed mid-flight by its own idle deadline.

        On the way out it also owns VM retirement for cancellation and
        unknown-outcome failures: any dispatched guest operation may still be
        running after CancelledError or CommunicationOutcomeUnknownError, so
        the session is closed before either propagates.
        """
        if self._closed:
            raise SessionClosedError("Session is closed")
        async with self._exec_lock:
            if self._closed:
                raise SessionClosedError("Session closed while waiting for lock")
            if self._vm.state in (VmState.DESTROYING, VmState.DESTROYED) or self._vm.process.returncode is not None:
                # QEMU died while the session was idle: the process-exit
                # watcher (armed lazily on first execute) publishes DESTROYING;
                # before that first execute, death is visible only via
                # returncode. Fail fast instead of letting every operation
                # time out one transient error at a time.
                await self._close_for_failure()
                raise SessionClosedError("Session VM was retired")
            self._cancel_idle_timer()
            try:
                yield
            except asyncio.CancelledError:
                # Every dispatched guest operation can outlive its cancelled
                # host waiter.  Retire the VM before permitting reuse, for
                # file/list operations as well as code execution.
                await self._close_for_failure()
                raise
            except CommunicationOutcomeUnknownError:
                # Dispatch may have reached the guest. Retire before exposing
                # the reconciliation-required error so no API can reuse a VM
                # whose command outcome is unknown.
                await self._close_for_failure()
                raise
            finally:
                if not self._closed:
                    self._reset_idle_timer()

    async def _close_for_failure(self) -> None:
        """Close the session without letting a close-time error mask the
        primary failure (or swallow the CancelledError) being propagated."""
        try:
            await self.close()
        except Exception:
            logger.exception(
                "Session close failed while handling a prior failure",
                extra={"vm_id": self._vm.vm_id},
            )

    async def _resolve_content(self, content: bytes | Path | IO[bytes]) -> tuple[IO[bytes], bool]:
        """Convert content to a readable binary stream with size validation.

        Called BEFORE _guard() so local disk I/O doesn't hold the exec lock.

        Returns:
            (stream, owned): The stream and whether the caller should close it.
            ``owned=True`` means we created the stream (bytes→BytesIO, Path→file);
            ``owned=False`` means the caller passed an IO[bytes] they own.

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
            return content.open("rb"), True
        if isinstance(content, bytes):
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"Content is {len(content)} bytes, exceeds {MAX_FILE_SIZE_BYTES}")
            return io.BytesIO(content), True
        # IO[bytes] — caller owns the stream, we do NOT close it.
        # Validate size only if the stream is seekable.
        if content.seekable():
            pos = content.tell()
            content.seek(0, 2)
            size = content.tell()
            content.seek(pos)
            if size - pos > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"Stream content is {size - pos} bytes, exceeds {MAX_FILE_SIZE_BYTES}")
        return content, False

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
        Non-zero exit codes do not normally close the session. Two cases are
        terminal: exit 137 (reserved as a terminal result) and a VM whose QEMU
        died while delivering the result — both return the result, then close
        the session.

        Args:
            code: Source code to execute.
            timeout_seconds: Execution timeout. Default: scheduler's default_timeout_seconds.
            env_vars: Environment variables for this execution.
            on_stdout: Callback for stdout chunks (streaming).
            on_stderr: Callback for stderr chunks (streaming).

        Returns:
            ExecutionResult with stdout, stderr, exit_code, timing info.

        Raises:
            CodeValidationError: Code is empty, whitespace-only, or contains null bytes.
            EnvVarValidationError: Invalid env var names/values (control chars, size limits).
            OutputLimitError: stdout/stderr exceeded guest-enforced limits (session stays alive).
            SessionClosedError: Session has been closed.
            VmPermanentError: VM communication failed (session auto-closed).
            VmTransientError: VM communication failed (session auto-closed).
            VmBootTimeoutError: Host deadline hit before command dispatch or
                guest connection timed out (session auto-closed) — not a
                user-code timeout; those return exit_code=-1.
            CommunicationOutcomeUnknownError: Transport failed after dispatch;
                the session is closed and callers must reconcile before retrying.

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
            except InputValidationError:
                # Input validation errors are caller bugs, not VM failures.
                # Session stays alive — caller can retry with valid input.
                raise
            except OutputLimitError:
                # Output limit exceeded — guest-enforced, REPL preserved.
                # Session stays alive — caller can retry with less output.
                raise
            except Exception:
                # VM failure — auto-close session. CancelledError is handled by
                # _guard, which retires the VM for every dispatched operation.
                await self._close_for_failure()
                raise

            execute_end = asyncio.get_running_loop().time()
            execute_ms = round((execute_end - execute_start) * 1000)

            self._exec_count += 1

            # Populate timing with session-measured values.
            execution_result = ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                execution_time_ms=result.execution_time_ms,
                external_cpu_time_ms=result.external_cpu_time_ms,
                external_memory_peak_mb=result.external_memory_peak_mb,
                external_cpu_nr_throttled=result.external_cpu_nr_throttled,
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
            if result.exit_code == SIGKILL_EXIT_CODE or self._vm.state in (VmState.DESTROYING, VmState.DESTROYED):
                logger.warning(
                    "Retiring session VM after terminal execution result",
                    extra={
                        "vm_id": self._vm.vm_id,
                        "exit_code": result.exit_code,
                        "vm_state": self._vm.state.value,
                    },
                )
                await self.close()
            return execution_result

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    async def write_file(self, path: str, content: bytes | Path | IO[bytes], *, make_executable: bool = False) -> None:
        """Write a file to the sandbox at the given path.

        Args:
            path: Relative path in sandbox (e.g., "input.csv", "data/model.pkl").
            content: File content as bytes, a local Path, or an IO[bytes] stream.
                For Path, content is streamed from disk — never fully loaded.
                For IO[bytes], the caller retains ownership (stream is NOT closed).
            make_executable: Set executable permission (0o755 vs 0o644).

        Raises:
            SessionClosedError: Session has been closed.
            ValueError: Content exceeds max file size limit.
            FileNotFoundError: Path source does not exist.
            VmPermanentError: Guest agent validation or write failure.
            VmTransientError: Transport failed before dispatch (retryable).
            CommunicationOutcomeUnknownError: Transport lost after dispatch —
                the write may have landed (session auto-closed).
        """
        # Resolve content to a stream BEFORE acquiring the lock —
        # local disk I/O should not block other session operations.
        stream, owned = await self._resolve_content(content)
        try:
            async with self._guard():
                await self._vm.write_file(path, stream, make_executable=make_executable)
        finally:
            if owned:
                stream.close()

    async def read_file(self, path: str, *, destination: Path | IO[bytes]) -> None:
        """Read a file from the sandbox, streaming to a local file or buffer.

        Decompressed chunks are written directly to *destination* — peak
        memory is ~128 KB regardless of file size.

        Args:
            path: Relative path in sandbox (e.g., "output.csv").
            destination: Local file path or IO[bytes] buffer to stream content into.
                For IO[bytes], content is written at the current position and
                the caller retains ownership (stream is NOT closed).

        Raises:
            SessionClosedError: Session has been closed.
            VmPermanentError: File not found or validation failure.
            VmTransientError: Timeout or transport failure (retryable on a
                live VM; cancellation and unknown outcomes auto-close).
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
            VmTransientError: Timeout or transport failure (retryable on a
                live VM); CommunicationOutcomeUnknownError after dispatch
                (session auto-closed).
        """
        async with self._guard():
            return await self._vm.list_files(path)

    async def close(self) -> None:
        """Close the session and destroy the VM.

        Idempotent: concurrent callers await the same cleanup task.

        Concurrency contract: close() does NOT acquire _exec_lock. Setting
        _closed=True immediately prevents new operations. An operation already
        inside _guard() may be interrupted by VM destruction; explicit close is
        the cancellation boundary, not a request-drain operation. Concurrent
        close callers await the same cleanup task. A cleanup failure is raised
        to every caller.
        """
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_impl())
        await asyncio.shield(self._close_task)

    async def _close_impl(self) -> None:
        """Perform the single shared cleanup operation."""
        self._cancel_idle_timer()

        logger.info(
            "Closing session",
            extra={
                "vm_id": self._vm.vm_id,
                "exec_count": self._exec_count,
            },
        )

        await self._vm_manager.destroy_vm(self._vm)
        # destroy_vm keeps failed workdir/cgroup cleanup owned for retry, and
        # publishes DESTROYED only after QEMU and gvproxy death is confirmed.
        # Only that process boundary may hide a terminal execution result.
        if self._vm.state is not VmState.DESTROYED:
            raise VmPermanentError(
                f"VM process cleanup was not confirmed for session VM {self._vm.vm_id}",
                context={"vm_id": self._vm.vm_id, "operation": "session_close"},
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
        """Arm (or re-arm) the idle deadline."""
        self._cancel_idle_timer()
        loop = asyncio.get_running_loop()
        self._idle_timer_handle = loop.call_later(self._idle_timeout_seconds, self._on_idle_timeout)

    def _cancel_idle_timer(self) -> None:
        """Cancel a pending idle deadline (no-op if it already fired)."""
        if self._idle_timer_handle is not None:
            self._idle_timer_handle.cancel()
            self._idle_timer_handle = None

    def _on_idle_timeout(self) -> None:
        """Idle deadline reached — close the session.

        Runs as a synchronous call_later callback, so it schedules the async
        close as a fire-and-forget task and holds a strong ref to it (the
        Session outlives it — close() publishes _close_task before any await).
        """
        self._idle_timer_handle = None
        logger.info(
            "Session idle timeout",
            extra={
                "vm_id": self._vm.vm_id,
                "idle_timeout_seconds": self._idle_timeout_seconds,
                "exec_count": self._exec_count,
            },
        )
        self._idle_close_task = asyncio.create_task(self._close_for_failure())

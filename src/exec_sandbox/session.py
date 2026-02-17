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
from collections.abc import Callable  # noqa: TC003 - Used at runtime for on_stdout/on_stderr parameters
from typing import TYPE_CHECKING, Self

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import SessionClosedError
from exec_sandbox.models import ExecutionResult, TimingBreakdown

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

        Note:
            Execution timeouts (guest agent level) return a result with
            exit_code=-1 and timeout message in stderr — they do NOT raise.
            Only VM-level failures (communication errors) raise exceptions.
        """
        if self._closed:
            raise SessionClosedError("Session is closed")

        timeout = timeout_seconds or self._default_timeout_seconds

        async with self._exec_lock:
            if self._closed:
                raise SessionClosedError("Session closed while waiting for exec lock")

            self._reset_idle_timer()

            execute_start = asyncio.get_event_loop().time()
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

            execute_end = asyncio.get_event_loop().time()
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

    async def close(self) -> None:
        """Close the session and destroy the VM.

        Idempotent: safe to call multiple times.
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
        if self._idle_timer_task is not None and not self._idle_timer_task.done():
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

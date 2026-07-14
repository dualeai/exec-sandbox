"""QEMU VM handle for running microVMs.

Provides the QemuVM class which represents a running QEMU microVM and handles
code execution, state management, and resource cleanup.
"""

import asyncio
import base64
import contextlib
import json
import signal
import sys
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import IO, TYPE_CHECKING, cast
from uuid import uuid4

from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    wait_random_exponential,
)

from exec_sandbox import cgroup, constants
from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import await_cancellation_safe, await_settled
from exec_sandbox.exceptions import (
    CodeValidationError,
    CommunicationError,
    CommunicationOutcomeUnknownError,
    EnvVarValidationError,
    InputValidationError,
    OutputLimitError,
    PackageNotAllowedError,
    SandboxError,
    VmBootTimeoutError,
    VmPermanentError,
    VmQemuCrashError,
    VmTransientError,
)
from exec_sandbox.guest_agent_protocol import (
    ExecuteCodeRequest,
    FileChunkResponseMessage,
    FileListMessage,
    FileReadCompleteMessage,
    FileWriteAckMessage,
    ListFilesRequest,
    PingRequest,
    PongMessage,
    ReadFileRequest,
    StreamingErrorMessage,
    WriteFileRequest,
)
from exec_sandbox.guest_channel import (
    GuestChannel,
    OperationInbox,
    StreamDeadlineExceededError,
    StreamResult,
    consume_stream,
)
from exec_sandbox.models import ExecutionResult, ExposedPort, FileInfo, Language, TimingBreakdown
from exec_sandbox.platform_utils import detect_host_os
from exec_sandbox.resource_cleanup import cleanup_vm_processes
from exec_sandbox.system_probes import detect_accel_type
from exec_sandbox.vm_timing import VmTiming
from exec_sandbox.vm_types import VALID_STATE_TRANSITIONS, VmState
from exec_sandbox.vm_working_directory import VmWorkingDirectory

if TYPE_CHECKING:
    from exec_sandbox.admission import ResourceReservation
    from exec_sandbox.platform_utils import ProcessWrapper

logger = get_logger(__name__)


async def _run_blocking_owned[T](call: Callable[[], T]) -> T:
    """Finish one owned thread operation before propagating cancellation."""
    return await await_cancellation_safe(asyncio.to_thread(call))


def _write_all_bytes(sink: IO[bytes], data: bytes) -> None:
    """Write one buffer completely or reject a non-progressing destination.

    Raw (unbuffered) IO[bytes] writers may short-write; loop until done.
    A raw non-blocking writer returns None (would-block) and a broken writer
    can over-report progress — both would silently corrupt the file, so both
    are rejected as OSError like any other non-progressing destination.
    """
    offset = 0
    while offset < len(data):
        # IO[bytes] types write() as -> int, but raw non-blocking writers
        # return None at runtime (io.RawIOBase.write would-block contract).
        written = cast("int | None", sink.write(data[offset:]))
        remaining = len(data) - offset
        if written is None or written <= 0 or written > remaining:
            raise OSError(
                f"read_file destination made no write progress: returned {written!r} for {remaining} remaining bytes"
            )
        offset += written


def guest_error_to_exception(
    msg: StreamingErrorMessage,
    vm_id: str,
    *,
    operation: str = "",
) -> SandboxError:
    """Map a guest StreamingErrorMessage to the appropriate SandboxError subclass.

    Args:
        msg: The error message from the guest agent.
        vm_id: VM identifier for context.
        operation: Optional operation name (e.g., "write_file", "read_file").

    Returns:
        A SandboxError subclass instance (not raised, caller raises).
    """
    context: dict[str, object] = {
        "vm_id": vm_id,
        "error_type": msg.error_type,
        "guest_message": msg.message,
    }
    if operation:
        context["operation"] = operation

    prefix = f"{operation}: " if operation else ""

    formatted = f"{prefix}[{msg.error_type}] {msg.message}"

    match msg.error_type:
        case constants.GuestErrorType.TIMEOUT:
            return VmTransientError(formatted, context=context)
        case constants.GuestErrorType.ENV_VAR:
            return EnvVarValidationError(formatted, context=context)
        case constants.GuestErrorType.CODE:
            return CodeValidationError(formatted, context=context)
        case constants.GuestErrorType.PACKAGE:
            return PackageNotAllowedError(formatted, context=context)
        case constants.GuestErrorType.OUTPUT_LIMIT:
            return OutputLimitError(formatted, context=context)
        case constants.GuestErrorType.EXECUTION:
            return VmTransientError(formatted, context=context)  # code never ran, transient
        case (
            constants.GuestErrorType.PATH
            | constants.GuestErrorType.IO
            | constants.GuestErrorType.REQUEST
            | constants.GuestErrorType.PROTOCOL
        ):
            return VmPermanentError(formatted, context=context)
        case _:
            return VmPermanentError(formatted, context=context)


# Use native zstd module (Python 3.14+) or backports.zstd
if sys.version_info >= (3, 14):
    from compression import zstd  # type: ignore[import-not-found]
else:
    from backports import zstd  # type: ignore[import-untyped,no-redef]


@dataclass(frozen=True, slots=True)
class QemuDiagnostics:
    """Crash diagnostics snapshot from a QEMU VM process.

    Captures console ring buffer, process stdout/stderr, signal info,
    and host environment for structured logging and error context.
    """

    vm_id: str
    exit_code: int | None
    signal_name: str  # "" if not signal-killed
    stdout: str  # truncated to QEMU_OUTPUT_MAX_BYTES
    stderr: str  # truncated to QEMU_OUTPUT_MAX_BYTES
    console_log: str  # full ring buffer join (already bounded by CONSOLE_RING_LINES)
    accel_type: str  # "hvf" | "tcg" | "kvm"
    host_os: str  # "macos" | "linux" | "unknown"


class QemuVM:
    """Handle to running QEMU microVM.

    Lifecycle managed by VmManager.
    Communicates via GuestChannel (dual-port virtio-serial).

    Context Manager Usage:
        Supports async context manager protocol for automatic cleanup:

        ```python
        async with await manager.launch_vm(...) as vm:
            result = await vm.execute(code="print('hello')", timeout_seconds=30)
            # VM automatically destroyed on exit, even if exception occurs
        ```

        Manual cleanup still available via destroy() method for explicit control.

    Attributes:
        vm_id: Unique VM identifier format: {tenant_id}-{task_id}-{uuid4}
        process: QEMU subprocess handle
        cgroup_path: cgroup v2 path for resource limits
        workdir: Working directory containing all VM temp files
        overlay_image: Ephemeral qcow2 overlay (property, from workdir)
        gvproxy_proc: Optional gvproxy-wrapper process for outbound filtering
        gvproxy_socket: Optional QEMU stream socket path (property, from workdir)
        gvproxy_log_task: Optional background task draining gvproxy stdout/stderr
    """

    def __init__(
        self,
        vm_id: str,
        process: "ProcessWrapper",
        cgroup_path: Path | None,
        workdir: VmWorkingDirectory,
        channel: GuestChannel,
        language: Language,
        console_lines: deque[str],
        gvproxy_proc: "ProcessWrapper | None" = None,
        qemu_log_task: asyncio.Task[None] | None = None,
        gvproxy_log_task: asyncio.Task[None] | None = None,
        release_callback: Callable[["QemuVM"], Awaitable[bool]] | None = None,
    ):
        """Initialize VM handle.

        Args:
            vm_id: Unique VM identifier (scoped by tenant_id)
            process: Running QEMU subprocess (ProcessWrapper for PID-reuse safety)
            cgroup_path: cgroup v2 path for cleanup
            workdir: Working directory containing overlay, sockets, and logs
            channel: Communication channel for guest agent
            language: Programming language for this VM
            console_lines: In-memory ring buffer of QEMU console output lines (for error diagnostics)
            gvproxy_proc: Optional gvproxy-wrapper process (ProcessWrapper)
            qemu_log_task: Background task draining QEMU stdout/stderr (prevents pipe deadlock)
            gvproxy_log_task: Background task draining gvproxy stdout/stderr (prevents pipe deadlock)
            release_callback: Owning manager teardown callback
        """
        self.vm_id = vm_id
        self.process = process
        self.cgroup_path = cgroup_path
        self.workdir = workdir
        self.channel = channel
        self.language = language
        self.console_lines = console_lines
        self.gvproxy_proc = gvproxy_proc
        self.qemu_log_task = qemu_log_task
        self.gvproxy_log_task = gvproxy_log_task
        self._release_callback = release_callback
        self._destroyed = False
        self._state = VmState.CREATING
        self._state_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._process_exit_task: asyncio.Task[int] | None = None
        # Timing instrumentation (set by VmManager.create_vm)
        self.timing = VmTiming()
        # Tracks if this VM holds an admission reservation (prevents double-release in destroy_vm)
        self.holds_admission_slot = False
        # Resource reservation from admission controller (set by VmManager.create_vm)
        self.resource_reservation: ResourceReservation | None = None
        # Port forwarding - set by VmManager after boot
        # Stores the resolved port mappings for result reporting
        self.exposed_ports: list[ExposedPort] = []
        # L1 memory snapshot restore flag
        self.l1_restored: bool = False
        # Previous nr_throttled from cgroup cpu.stat (for delta-based warning)
        self._prev_nr_throttled: int = 0

    # -------------------------------------------------------------------------
    # Timing properties (backwards-compatible accessors to VmTiming)
    # -------------------------------------------------------------------------

    @property
    def setup_ms(self) -> int | None:
        """Get resource setup time in milliseconds."""
        return self.timing.setup_ms

    @setup_ms.setter
    def setup_ms(self, value: int) -> None:
        """Set resource setup time in milliseconds."""
        self.timing.setup_ms = value

    @property
    def overlay_ms(self) -> int | None:
        """Get overlay acquisition time in milliseconds."""
        return self.timing.overlay_ms

    @overlay_ms.setter
    def overlay_ms(self, value: int) -> None:
        """Set overlay acquisition time in milliseconds."""
        self.timing.overlay_ms = value

    @property
    def boot_ms(self) -> int | None:
        """Get VM boot time in milliseconds."""
        return self.timing.boot_ms

    @boot_ms.setter
    def boot_ms(self, value: int) -> None:
        """Set VM boot time in milliseconds."""
        self.timing.boot_ms = value

    @property
    def qemu_cmd_build_ms(self) -> int | None:
        """Time for pre-launch setup (command build, socket cleanup, channel creation)."""
        return self.timing.qemu_cmd_build_ms

    @qemu_cmd_build_ms.setter
    def qemu_cmd_build_ms(self, value: int) -> None:
        self.timing.qemu_cmd_build_ms = value

    @property
    def gvproxy_start_ms(self) -> int | None:
        """Time to start gvproxy (0 if network disabled)."""
        return self.timing.gvproxy_start_ms

    @gvproxy_start_ms.setter
    def gvproxy_start_ms(self, value: int) -> None:
        self.timing.gvproxy_start_ms = value

    @property
    def qemu_fork_ms(self) -> int | None:
        """Time for QEMU process fork/exec."""
        return self.timing.qemu_fork_ms

    @qemu_fork_ms.setter
    def qemu_fork_ms(self, value: int) -> None:
        self.timing.qemu_fork_ms = value

    @property
    def guest_wait_ms(self) -> int | None:
        """Time waiting for guest agent (kernel + initramfs + agent init)."""
        return self.timing.guest_wait_ms

    @guest_wait_ms.setter
    def guest_wait_ms(self, value: int) -> None:
        self.timing.guest_wait_ms = value

    # -------------------------------------------------------------------------
    # Other VM properties
    # -------------------------------------------------------------------------

    @property
    def overlay_image(self) -> Path:
        """Path to overlay image (from workdir)."""
        return self.workdir.overlay_image

    @property
    def gvproxy_socket(self) -> Path | None:
        """Path to gvproxy socket (from workdir, None if no network)."""
        return self.workdir.gvproxy_socket if self.gvproxy_proc else None

    @property
    def use_qemu_vm_user(self) -> bool:
        """Whether QEMU runs as qemu-vm user."""
        return self.workdir.use_qemu_vm_user

    @property
    def qmp_socket(self) -> Path:
        """Path to QMP control socket (from workdir)."""
        return self.workdir.qmp_socket

    @property
    def cleanup_lock(self) -> asyncio.Lock:
        """Serialize all direct and manager-owned cleanup attempts."""
        return self._cleanup_lock

    async def __aenter__(self) -> "QemuVM":
        """Enter async context manager.

        Returns:
            Self for use in async with statement
        """
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> bool:
        """Exit async context manager - ensure cleanup.

        Returns:
            False to propagate exceptions
        """
        # Cleanup VM when exiting context (always runs destroy)
        # destroy() is idempotent and state-safe, will skip if already destroying/destroyed
        if not await self.destroy():
            raise VmPermanentError(
                f"VM resource cleanup was not confirmed for {self.vm_id}",
                context={"vm_id": self.vm_id, "operation": "vm_context_exit"},
            )
        return False  # Don't suppress exceptions

    @property
    def state(self) -> VmState:
        """Current VM state."""
        return self._state

    async def transition_state(self, new_state: VmState) -> None:
        """Transition VM to new state with validation.

        Validates state transition against VALID_STATE_TRANSITIONS to prevent
        invalid state changes (e.g., DESTROYED -> READY).

        Args:
            new_state: Target state to transition to

        Raises:
            VmPermanentError: If transition is invalid for current state
        """
        async with self._state_lock:
            # Validate transition is allowed from current state
            allowed_transitions = VALID_STATE_TRANSITIONS.get(self._state, set())
            if new_state not in allowed_transitions:
                raise VmPermanentError(
                    f"Invalid state transition: {self._state.value} -> {new_state.value}",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                        "target_state": new_state.value,
                        "allowed_transitions": [s.value for s in allowed_transitions],
                    },
                )

            old_state = self._state
            self._state = new_state
            logger.debug(
                "VM state transition",
                extra={
                    "debug_category": "lifecycle",
                    "vm_id": self.vm_id,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                },
            )

    async def execute(  # noqa: PLR0912, PLR0915
        self,
        code: str,
        timeout_seconds: int,
        env_vars: dict[str, str] | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Execute code via guest agent communication.

        Implementation:
        1. Connect to guest via dual-port virtio-serial channel
        2. Send execution request JSON with action, language, code, timeout, env_vars
        3. Wait for result with timeout (cgroup enforced)
        4. Parse result: stdout, stderr, exit_code, memory_mb, execution_time_ms

        Timeout Architecture (3-layer system):
        1. Init timeout (5s): Connection establishment to guest agent
        2. Soft timeout (timeout_seconds): Guest agent enforcement (sent in request)
        3. Hard timeout (timeout_seconds + EXECUTION_TIMEOUT_MARGIN_SECONDS): Host watchdog

        Example: timeout_seconds=30
        - connect(5s) - Fixed init window for socket establishment
        - send_request(timeout=40s) - 30s soft + 10s bounded-infrastructure margin
        - Guest enforces 30s of user code; the host also budgets readiness and cleanup

        Args:
            code: Code to execute in guest VM
            timeout_seconds: Maximum execution time (enforced by cgroup)
            env_vars: Environment variables for code execution (default: None)
            on_stdout: Optional callback for real-time stdout streaming.
                If the callback raises, it is disabled for the remainder of the
                execution (output collection is unaffected).
            on_stderr: Optional callback for real-time stderr streaming.
                Same defensive semantics as on_stdout.

        Returns:
            ExecutionResult with stdout, stderr, exit code, and resource usage

        Raises:
            CodeValidationError: Code is empty, whitespace-only, or contains null bytes.
            EnvVarValidationError: Invalid env vars (control chars, size limits)
            OutputLimitError: stdout or stderr exceeded guest-enforced size limits.
            VmTransientError: Failure before dispatch (connect failure, guest
                readiness gate, stdin write failure). Code never ran.
            VmPermanentError: VM not in READY state, or protocol/request corruption.
            VmBootTimeoutError: Host deadline hit before command dispatch, or
                guest connection timed out.
            CommunicationOutcomeUnknownError: Transport lost after dispatch
                (socket EOF, QEMU crash, hard deadline) — the code may have
                run; reconcile side effects before retrying.

        Failure modes:
            System/infrastructure errors (raised as exceptions):
                - VmTransientError: failure before dispatch — code never ran,
                  caller can retry
                - CommunicationOutcomeUnknownError: transport lost after
                  dispatch (socket EOF, QEMU crash, hard deadline) — reconcile
                  before retrying; a died-mid-REPL-spawn guest surfaces here
                - VmBootTimeoutError: execution exceeded host timeout
                - VmPermanentError: protocol/request corruption
            User code results (returned as ExecutionResult, not retryable):
                - exit_code=0: success
                - exit_code>0: code error
                - exit_code>128: killed by signal (128 + signal_number)
                - exit_code=-1: execution timeout (code ran too long)
        """
        # Validate VM is in READY state before execution (atomic check-and-set)
        async with self._state_lock:
            if self._state != VmState.READY:
                raise VmPermanentError(
                    f"Cannot execute in state {self._state.value}, must be READY",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                        "language": self.language,
                    },
                )

            # Validate transition to EXECUTING (inline to avoid lock re-acquisition)
            allowed_transitions = VALID_STATE_TRANSITIONS.get(self._state, set())
            if VmState.EXECUTING not in allowed_transitions:
                raise VmPermanentError(
                    f"Invalid state transition: {self._state.value} -> {VmState.EXECUTING.value}",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                        "target_state": VmState.EXECUTING.value,
                        "allowed_transitions": [s.value for s in allowed_transitions],
                    },
                )

            # Transition to EXECUTING inside same lock
            old_state = self._state
            self._state = VmState.EXECUTING
            logger.debug(
                "VM state transition",
                extra={
                    "debug_category": "lifecycle",
                    "vm_id": self.vm_id,
                    "old_state": old_state.value,
                    "new_state": self._state.value,
                },
            )

        # Prepare execution request
        timeout_phase = "connect"
        hard_timeout = timeout_seconds + constants.EXECUTION_TIMEOUT_MARGIN_SECONDS

        try:
            request = ExecuteCodeRequest(
                language=self.language,
                code=code,
                timeout=timeout_seconds,
                env_vars=env_vars or {},
            )
        except ValidationError as e:
            # Translate Pydantic structural errors through the same path as
            # guest-agent errors so all error mapping lives in one place.
            error_locs = {field for err in e.errors() for field in err.get("loc", ())}
            if "env_vars" in error_locs:
                error_type = constants.GuestErrorType.ENV_VAR
            elif "code" in error_locs:
                error_type = constants.GuestErrorType.CODE
            else:
                error_type = constants.GuestErrorType.REQUEST
            msg = StreamingErrorMessage(message=str(e), error_type=error_type.value)
            exc = guest_error_to_exception(msg, self.vm_id)
            if isinstance(exc, InputValidationError):
                # Input was invalid but VM is fine — restore READY for session reuse
                with contextlib.suppress(VmPermanentError):
                    await self.transition_state(VmState.READY)
            raise exc from e

        try:
            # Re-check state before expensive I/O operations
            # Between lock release and here, destroy() could have been called
            # which would transition state to DESTROYING or DESTROYED
            # Note: pyright doesn't understand async race conditions, so we suppress the warning
            if self._state in (VmState.DESTROYING, VmState.DESTROYED):  # type: ignore[comparison-overlap]
                raise VmPermanentError(
                    "VM destroyed during execution start",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                    },
                )

            # Connect to guest agent with timing
            # Fixed init timeout (connection establishment, independent of execution timeout)
            connect_start = asyncio.get_running_loop().time()
            await self.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
            connect_ms = round((asyncio.get_running_loop().time() - connect_start) * 1000)

            # Stream execution output to console
            # Hard timeout = soft timeout (guest enforcement) + margin (host watchdog)
            # Error handler: input validation restores READY and raises,
            # timeout falls through to consume_stream's default (exit_code=-1),
            # all other errors raise as infrastructure failures.
            async def _handle_exec_error(msg: StreamingErrorMessage) -> None:
                exc = guest_error_to_exception(msg, self.vm_id, operation="execute")
                # Recoverable input errors: VM is fine, restore READY for session reuse
                if isinstance(exc, (InputValidationError, OutputLimitError, PackageNotAllowedError)):
                    with contextlib.suppress(VmPermanentError):
                        await self.transition_state(VmState.READY)
                    raise exc
                # Timeout: code ran too long — this is a user code result, not infrastructure.
                # Fall through to consume_stream's default handling (exit_code=-1 + stderr message).
                if msg.error_type == constants.GuestErrorType.TIMEOUT:
                    logger.warning(
                        f"Execution timeout: [{msg.error_type}] {msg.message}",
                        extra={
                            "vm_id": self.vm_id,
                            "error_message": msg.message,
                            "error_type": msg.error_type,
                        },
                    )
                    return
                # All other guest errors = infrastructure/system failures.
                # Code never ran or system is broken — raise for caller to handle.
                raise exc

            timeout_phase = "execution"
            result = await self._consume_while_qemu_alive(
                consume_stream(
                    self.channel,
                    request,
                    timeout=hard_timeout,
                    vm_id=self.vm_id,
                    on_stdout=on_stdout,
                    on_stderr=on_stderr,
                    on_error=_handle_exec_error,
                ),
                soft_timeout_seconds=timeout_seconds,
                hard_timeout_seconds=hard_timeout,
            )

            # Measure external resources from host (cgroup v2)
            external_cpu_ms, external_mem_mb, external_nr_throttled = await self._read_cgroup_stats()

            if external_nr_throttled is not None and external_nr_throttled > self._prev_nr_throttled:
                new_throttles = external_nr_throttled - self._prev_nr_throttled
                logger.warning(
                    "VM CPU throttled by cgroup — consider increasing cpu_cores or DEFAULT_VM_CPU_OVERHEAD_CORES",
                    extra={
                        "vm_id": self.vm_id,
                        "nr_throttled_delta": new_throttles,
                        "nr_throttled_total": external_nr_throttled,
                    },
                )
            if external_nr_throttled is not None:
                self._prev_nr_throttled = external_nr_throttled

            # Debug log final execution output
            logger.debug(
                "Code execution complete",
                extra={
                    "vm_id": self.vm_id,
                    "exit_code": result.exit_code,
                    "execution_time_ms": result.execution_time_ms,
                    "stdout_len": len(result.stdout),
                    "stderr_len": len(result.stderr),
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500] if result.stderr else None,
                },
            )

            # Parse result with both internal (guest) and external (host) measurements
            # Note: timing is a placeholder here - scheduler will populate actual values
            exec_result = ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                execution_time_ms=result.execution_time_ms,  # Guest-reported
                external_cpu_time_ms=external_cpu_ms or None,  # Host-measured
                external_memory_peak_mb=external_mem_mb or None,  # Host-measured
                external_cpu_nr_throttled=external_nr_throttled,  # Host-measured
                timing=TimingBreakdown(setup_ms=0, boot_ms=0, execute_ms=0, total_ms=0, connect_ms=connect_ms),
                spawn_ms=result.spawn_ms,  # Guest-reported granular timing
                process_ms=result.process_ms,  # Guest-reported granular timing
            )

            # Success - transition back to READY for reuse (if not destroyed)
            try:
                await self.transition_state(VmState.READY)
            except VmPermanentError as e:
                # VM destroyed while executing, skip transition
                logger.debug(
                    "VM destroyed during execution, skipping READY transition",
                    extra={"vm_id": self.vm_id, "error": str(e)},
                )
            return exec_result

        except asyncio.CancelledError:
            logger.warning(
                "Code execution cancelled",
                extra={"vm_id": self.vm_id, "language": self.language},
            )
            # Re-raise to propagate cancellation
            raise
        except StreamDeadlineExceededError as e:
            context = {
                "vm_id": self.vm_id,
                "timeout_seconds": timeout_seconds,
                "soft_timeout_seconds": timeout_seconds,
                "hard_timeout_seconds": hard_timeout,
                "timeout_margin_seconds": constants.EXECUTION_TIMEOUT_MARGIN_SECONDS,
                "timeout_phase": timeout_phase,
                "timeout_source": "total_stream_deadline",
                "command_dispatched": e.command_dispatched,
                "language": self.language,
            }
            if not e.command_dispatched:
                raise VmBootTimeoutError(
                    f"VM {self.vm_id} execution exceeded {hard_timeout}s host deadline before command dispatch",
                    context=context,
                ) from e
            raise CommunicationOutcomeUnknownError(
                f"VM {self.vm_id} execution exceeded {hard_timeout}s host deadline "
                f"({timeout_seconds}s guest timeout + "
                f"{constants.EXECUTION_TIMEOUT_MARGIN_SECONDS}s infrastructure margin) "
                "after command dispatch",
                context=context,
            ) from e
        except TimeoutError as e:
            if timeout_phase == "connect":
                raise VmBootTimeoutError(
                    f"VM {self.vm_id} guest connection timed out before execution dispatch",
                    context={
                        "vm_id": self.vm_id,
                        "timeout_seconds": timeout_seconds,
                        "connect_attempt_timeout_seconds": constants.GUEST_CONNECT_TIMEOUT_SECONDS,
                        "timeout_phase": timeout_phase,
                        "language": self.language,
                    },
                ) from e
            raise CommunicationOutcomeUnknownError(
                f"VM {self.vm_id} execution transport timed out before a terminal event",
                context={
                    "vm_id": self.vm_id,
                    "language": self.language,
                    "soft_timeout_seconds": timeout_seconds,
                    "hard_timeout_seconds": hard_timeout,
                    "timeout_phase": timeout_phase,
                    "timeout_source": "nested_transport",
                },
            ) from e
        except (
            OSError,
            json.JSONDecodeError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            ValidationError,
            RuntimeError,
            CommunicationError,
        ) as e:
            raise VmTransientError(
                f"VM {self.vm_id} communication failed: {e}",
                context={
                    "vm_id": self.vm_id,
                    "language": self.language,
                    "error_type": type(e).__name__,
                },
            ) from e

    async def _consume_while_qemu_alive(
        self,
        stream: Coroutine[object, object, StreamResult],
        *,
        soft_timeout_seconds: int,
        hard_timeout_seconds: int,
    ) -> StreamResult:
        """Race execution streaming against QEMU process death.

        A terminal guest frame routed before socket EOF wins over process
        death. The channel uses a bounded EOF-drain window; prolonged consumer
        backpressure therefore resolves conservatively as outcome-unknown.
        QEMU death is also published through the channel's out-of-band failure
        signal instead of consuming a bounded data-queue slot.
        """
        stream_task = asyncio.create_task(stream)
        death_task = self._get_process_exit_task()
        try:
            done, _pending = await asyncio.wait(
                {stream_task, death_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stream_task in done and death_task not in done:
                result = await stream_task
                # Let the VM-lifetime waiter publish a process exit that was
                # delivered in the same scheduling turn as the terminal frame.
                await asyncio.sleep(0)
                if not death_task.done():
                    return result

            exit_code = await death_task
            await self.channel.fail_pending_operations(f"QEMU exited during execution with code {exit_code}")
            try:
                result = await stream_task
            except CommunicationOutcomeUnknownError as error:
                raise CommunicationOutcomeUnknownError(
                    f"QEMU exited during execution before a terminal guest event: {error}",
                    context={
                        "vm_id": self.vm_id,
                        "language": self.language,
                        "exit_code": exit_code,
                        "soft_timeout_seconds": soft_timeout_seconds,
                        "hard_timeout_seconds": hard_timeout_seconds,
                    },
                ) from error
            except SandboxError:
                # A parsed terminal guest error is a known operation outcome,
                # but the VM that reported it has exited and is not reusable.
                await self.begin_destroy()
                raise
            except Exception as error:
                raise CommunicationOutcomeUnknownError(
                    f"QEMU exited during execution before a trustworthy terminal guest event: "
                    f"{type(error).__name__}: {error}",
                    context={
                        "vm_id": self.vm_id,
                        "language": self.language,
                        "exit_code": exit_code,
                        "soft_timeout_seconds": soft_timeout_seconds,
                        "hard_timeout_seconds": hard_timeout_seconds,
                    },
                ) from error

            # The command result is known, but the VM is gone. Preserve the
            # terminal result and publish a non-reusable VM state; Session will
            # synchronously close/confirm cleanup before returning it.
            await self.begin_destroy()
            return result
        finally:
            if not stream_task.done():
                stream_task.cancel()
            await asyncio.gather(stream_task, return_exceptions=True)

    def _get_process_exit_task(self) -> asyncio.Task[int]:
        """Return the single VM-lifetime QEMU death notification task."""
        if self._process_exit_task is None:

            async def _wait_and_retire() -> int:
                exit_code = await self.process.wait()
                # Process death is a VM-lifetime state transition, not only an
                # execution-race signal. This also closes the post-terminal
                # window in which a dead VM could otherwise remain READY.
                await self.begin_destroy()
                return exit_code

            self._process_exit_task = asyncio.create_task(
                _wait_and_retire(),
                name=f"qemu-exit-{self.vm_id}",
            )
        return self._process_exit_task

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    async def write_file(self, path: str, content: IO[bytes], *, make_executable: bool = False) -> None:
        """Write a file to the sandbox via streaming zstd-compressed chunks.

        Protocol:
        1. Send WriteFileRequest header (with op_id)
        2. Stream-compress content chunk by chunk as FileChunkRequest messages
        3. Send FileEndRequest to finalize
        4. Await FileWriteAckMessage from guest

        Content is read from the stream in 128 KB chunks, compressed
        incrementally with a shared ZstdCompressor, and sent immediately.
        Neither the full content nor the full compressed payload is ever
        held in memory — only the current chunk pair.

        Retries once on "No active write" protocol errors, which indicate the
        WriteFile header was dropped by QEMU's virtio-serial during a guest
        agent reconnection cycle (READ_TIMEOUT_MS=18s idle timeout).

        Args:
            path: Relative path in sandbox (PATH_MAX 4096, NAME_MAX 255 per component)
            content: Readable binary stream (BytesIO for bytes, open file for Path)
            make_executable: Set executable permission (0o755 vs 0o644)

        Raises:
            VmPermanentError: On validation or write failure
            VmTransientError: On pre-dispatch timeout or communication failure
            CommunicationOutcomeUnknownError: Transport lost after the header
                was dispatched — the write may have landed
        """

        # Validate path (once, before any retry)
        if not path:
            raise VmPermanentError(
                "write_file path must not be empty",
                context={"vm_id": self.vm_id, "path": path},
            )
        if len(path) > constants.MAX_FILE_PATH_LENGTH:
            raise VmPermanentError(
                f"write_file path is {len(path)} chars, exceeds {constants.MAX_FILE_PATH_LENGTH}",
                context={"vm_id": self.vm_id, "path_length": len(path)},
            )

        try:
            retry_position = content.tell() if content.seekable() else None
        except (AttributeError, OSError):
            retry_position = None

        # Retry once for "No active write" — a transient protocol error caused
        # by QEMU dropping the WriteFile header when the guest virtio-serial
        # port is closed during the guest agent's 18s idle-timeout reconnect.
        try:
            await self._write_file_protocol(path, content, make_executable=make_executable)
        except VmPermanentError as exc:
            if "No active write" not in str(exc):
                raise
            logger.warning(
                "write_file header dropped (stale connection), forcing reconnect and retrying",
                extra={"vm_id": self.vm_id, "path": path},
            )
            # Discard the stale connection even when the retry can't proceed.
            await self.channel.close()
            if retry_position is None:
                raise
            content.seek(retry_position)
            await self._write_file_protocol(path, content, make_executable=make_executable)

    async def _write_file_protocol(  # noqa: PLR0912, PLR0915
        self, path: str, content: IO[bytes], *, make_executable: bool = False
    ) -> None:
        """Execute the write_file streaming protocol (single attempt)."""

        op_id = uuid4().hex
        op_queue: OperationInbox | None = None

        try:
            await self.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
            op_queue = await self.channel.register_op(op_id)
            # Send header frame
            header = WriteFileRequest(op_id=op_id, path=path, make_executable=make_executable)
            header_bytes = header.model_dump_json(by_alias=False, exclude_none=True).encode() + b"\n"
            await self.channel.enqueue_registered(op_queue, header_bytes)

            # Stream-compress content chunk by chunk.  Each 128 KB slice is
            # read from the stream, compressed, and sent immediately.
            # file_end is a commit record and is sent only after a proven EOF,
            # successful compressor flush, and successful chunk enqueue.
            chunk_size = constants.FILE_TRANSFER_CHUNK_SIZE
            compressor = zstd.ZstdCompressor(level=constants.FILE_TRANSFER_ZSTD_LEVEL)
            source_size = 0

            def _encode_frame(compressed: bytes) -> bytes:
                # TODO: Binary framing would eliminate this base64 encode overhead (~33% wire bloat)
                chunk_b64 = base64.b64encode(compressed).decode("ascii")
                return json.dumps({"action": "file_chunk", "op_id": op_id, "data": chunk_b64}).encode() + b"\n"

            # Keep each caller-owned read joined to this coroutine. A failed
            # enqueue or cancellation must not return while a thread still
            # touches the caller's stream.
            def _read_and_compress(reader: IO[bytes]) -> tuple[bytes, int] | None:
                raw = reader.read(chunk_size)
                if not raw:
                    return None  # EOF
                compressed = compressor.compress(raw)
                if not compressed:
                    return b"", len(raw)  # compressor buffered, no output yet
                return _encode_frame(compressed), len(raw)

            while True:
                result = await _run_blocking_owned(lambda: _read_and_compress(content))
                if result is None:
                    break  # Proven source EOF.
                frame, raw_size = result
                source_size += raw_size
                if frame:  # non-empty (not just compressor buffering)
                    await self.channel.enqueue_raw(frame)

            # Flush remaining compressed data from the compressor.
            def _flush_and_encode() -> bytes | None:
                remaining = compressor.flush()
                return _encode_frame(remaining) if remaining else None

            flush_frame = await _run_blocking_owned(_flush_and_encode)
            if flush_frame:
                await self.channel.enqueue_raw(flush_frame)

            # This is the only guest-side commit signal.  Never place it in a
            # failure/cancellation cleanup path: disconnect makes the guest
            # discard its temporary prefix.
            end_frame = json.dumps({"action": "file_end", "op_id": op_id}).encode() + b"\n"
            await self.channel.enqueue_raw(end_frame)

            # Await ack from op_queue
            response = await asyncio.wait_for(op_queue.get(), timeout=constants.FILE_IO_TIMEOUT_SECONDS)

            if isinstance(response, StreamingErrorMessage):
                raise guest_error_to_exception(response, self.vm_id, operation=f"write_file '{path}'")

            if not isinstance(response, FileWriteAckMessage):
                raise VmPermanentError(
                    f"write_file unexpected response type: {type(response).__name__}",
                    context={"vm_id": self.vm_id, "path": path},
                )

            if response.path != path or response.bytes_written != source_size:
                raise CommunicationOutcomeUnknownError(
                    f"VM {self.vm_id} write_file acknowledgement did not bind the requested content for '{path}'",
                    context={
                        "vm_id": self.vm_id,
                        "path": path,
                        "ack_path": response.path,
                        "source_bytes": source_size,
                        "ack_bytes_written": response.bytes_written,
                    },
                )

        except CommunicationOutcomeUnknownError:
            raise
        except (VmPermanentError, VmTransientError, InputValidationError, OutputLimitError, PackageNotAllowedError):
            # These are parsed terminal guest responses, so their operation
            # outcome is known even though the response is an error.
            raise
        except Exception as error:
            dispatched = op_queue is not None and op_queue.command_may_have_been_sent
            if dispatched:
                raise CommunicationOutcomeUnknownError(
                    f"VM {self.vm_id} write_file failed after dispatch for '{path}': {type(error).__name__}: {error}",
                    context={
                        "vm_id": self.vm_id,
                        "path": path,
                        "error_type": type(error).__name__,
                    },
                ) from error
            if isinstance(error, TimeoutError):
                raise VmTransientError(
                    f"VM {self.vm_id} write_file timed out before dispatch for '{path}'",
                    context={"vm_id": self.vm_id, "path": path},
                ) from error
            if isinstance(error, (OSError, json.JSONDecodeError, CommunicationError)):
                raise VmTransientError(
                    f"VM {self.vm_id} write_file communication failed before dispatch: {error}",
                    context={"vm_id": self.vm_id, "path": path},
                ) from error
            raise
        finally:
            await self.channel.unregister_op(op_id)

    async def read_file(self, path: str, *, destination: Path | IO[bytes]) -> None:
        """Read a file from the sandbox via streaming zstd-compressed chunks.

        Protocol:
        1. Send ReadFileRequest (with op_id)
        2. Receive FileChunkResponseMessage messages (zstd-compressed)
        3. Receive FileReadCompleteMessage (end of stream)

        When *destination* is a ``Path``, chunks are written to a temp sibling
        (``<dest>.<op_id>.tmp``) and atomically renamed on success.
        When *destination* is an ``IO[bytes]``, chunks are written directly
        to the buffer at its current position; the caller retains ownership.

        Peak memory is bounded by queue depths (OP_QUEUE_DEPTH items x ~200KB),
        not file size.

        Args:
            path: Relative path in sandbox (PATH_MAX 4096, NAME_MAX 255 per component)
            destination: Local file path or IO[bytes] buffer to stream into

        Raises:
            VmPermanentError: On not-found or validation failure
            VmTransientError: On timeout or communication failure (reads are
                idempotent, so post-dispatch timeouts stay retryable)
            CommunicationOutcomeUnknownError: Event transport lost after
                dispatch (propagated from the inbox)
        """
        if isinstance(destination, Path):
            await self._read_file_to_path(path, destination)
        else:
            await self._read_file_to_buffer(path, destination)

    async def _recv_file_chunks(self, path: str, op_id: str, sink: IO[bytes]) -> None:
        """Connect, send ReadFileRequest, and stream decompressed chunks into *sink*.

        Shared protocol implementation for both Path and IO[bytes] destinations.
        The caller is responsible for op registration/unregistration and sink lifecycle.
        """
        await self.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
        op_queue = await self.channel.register_op(op_id)

        request = ReadFileRequest(op_id=op_id, path=path)
        request_bytes = request.model_dump_json(by_alias=False, exclude_none=True).encode() + b"\n"
        await self.channel.enqueue_registered(op_queue, request_bytes)

        decompressor = zstd.ZstdDecompressor()
        received_size = 0
        while True:
            msg = await asyncio.wait_for(op_queue.get(), timeout=constants.FILE_IO_TIMEOUT_SECONDS)

            if isinstance(msg, FileChunkResponseMessage):
                # TODO: Binary framing would eliminate this base64 decode overhead (~33% wire bloat)
                compressed_chunk = base64.b64decode(msg.data)
                decompressed = decompressor.decompress(compressed_chunk)

                await _run_blocking_owned(partial(_write_all_bytes, sink, decompressed))
                received_size += len(decompressed)
            elif isinstance(msg, FileReadCompleteMessage):
                if msg.path != path or msg.size != received_size:
                    raise CommunicationError(
                        f"read_file terminal metadata mismatch for '{path}': "
                        f"path={msg.path!r}, declared_size={msg.size}, received_size={received_size}"
                    )
                break
            elif isinstance(msg, StreamingErrorMessage):
                raise guest_error_to_exception(msg, self.vm_id, operation=f"read_file '{path}'")
            else:
                raise VmPermanentError(
                    f"read_file unexpected message type: {type(msg).__name__}",
                    context={"vm_id": self.vm_id, "path": path},
                )

    async def _read_file_to_path(self, path: str, destination: Path) -> None:
        """Stream sandbox file to a local Path with atomic rename."""

        op_id = uuid4().hex
        # Temp path scoped by op_id: foo.bin -> foo.bin.<op_id>.tmp
        # Prevents collisions when concurrent reads target the same destination.
        tmp_dest = destination.with_suffix(f"{destination.suffix}.{op_id}.tmp")

        try:
            # Stream decompressed chunks to .tmp file.
            # All disk I/O is offloaded to the thread pool to avoid blocking
            # the event loop (decompressed chunks can be large).
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: destination.parent.mkdir(parents=True, exist_ok=True))
            sink = await loop.run_in_executor(None, tmp_dest.open, "wb")
            try:
                await self._recv_file_chunks(path, op_id, sink)
            finally:
                await loop.run_in_executor(None, sink.close)

            # Atomic rename — readers never see a partial file.
            await loop.run_in_executor(None, tmp_dest.rename, destination)

        except TimeoutError as e:
            raise VmTransientError(
                f"VM {self.vm_id} read_file timed out for '{path}'",
                context={"vm_id": self.vm_id, "path": path},
            ) from e
        except (OSError, json.JSONDecodeError, CommunicationError) as e:
            raise VmTransientError(
                f"VM {self.vm_id} read_file communication failed: {e}",
                context={"vm_id": self.vm_id, "path": path},
            ) from e
        finally:
            # Clean up partial .tmp on any failure (no-op after successful rename)
            tmp_dest.unlink(missing_ok=True)
            await self.channel.unregister_op(op_id)

    async def _read_file_to_buffer(self, path: str, sink: IO[bytes]) -> None:
        """Stream sandbox file directly into an IO[bytes] buffer."""

        op_id = uuid4().hex

        try:
            await self._recv_file_chunks(path, op_id, sink)
        except TimeoutError as e:
            raise VmTransientError(
                f"VM {self.vm_id} read_file timed out for '{path}'",
                context={"vm_id": self.vm_id, "path": path},
            ) from e
        except (OSError, json.JSONDecodeError, CommunicationError) as e:
            raise VmTransientError(
                f"VM {self.vm_id} read_file communication failed: {e}",
                context={"vm_id": self.vm_id, "path": path},
            ) from e
        finally:
            await self.channel.unregister_op(op_id)

    async def list_files(self, path: str = "") -> list[FileInfo]:
        """List files in a sandbox directory via guest agent.

        Args:
            path: Relative path (empty for sandbox root)

        Returns:
            List of FileInfo entries

        Raises:
            VmPermanentError: On validation failure
            VmTransientError: On timeout or communication failure
            CommunicationOutcomeUnknownError: Transport lost after dispatch
        """
        request = ListFilesRequest(path=path)

        try:
            await self.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
            response = await self.channel.send_request(request, timeout=constants.FILE_IO_TIMEOUT_SECONDS)
        except TimeoutError as e:
            raise VmTransientError(
                f"VM {self.vm_id} list_files timed out for '{path}'",
                context={"vm_id": self.vm_id, "path": path},
            ) from e
        except (OSError, json.JSONDecodeError, CommunicationError) as e:
            # CommunicationOutcomeUnknownError subclasses SandboxError, not
            # CommunicationError, so post-dispatch unknown outcomes still
            # propagate as themselves.
            raise VmTransientError(
                f"VM {self.vm_id} list_files communication failed: {e}",
                context={"vm_id": self.vm_id, "path": path},
            ) from e

        if isinstance(response, StreamingErrorMessage):
            raise guest_error_to_exception(response, self.vm_id, operation=f"list_files '{path}'")

        if not isinstance(response, FileListMessage):
            raise VmPermanentError(
                f"list_files unexpected response type: {type(response).__name__}",
                context={"vm_id": self.vm_id, "path": path},
            )

        return [FileInfo(name=e.name, is_dir=e.is_dir, size=e.size) for e in response.entries]

    async def _read_cgroup_stats(self) -> tuple[int | None, int | None, int | None]:
        """Read external CPU time, peak memory, and CPU throttle count from cgroup v2.

        Returns:
            Tuple of (cpu_time_ms, peak_memory_mb, nr_throttled)
            Returns (None, None, None) if cgroup not available or read fails
        """
        return await cgroup.read_cgroup_stats(self.cgroup_path)

    async def begin_destroy(self) -> None:
        """Make the VM non-reusable while allowing cleanup retries."""
        async with self._state_lock:
            if self._state in (VmState.DESTROYING, VmState.DESTROYED):
                return
            allowed_transitions = VALID_STATE_TRANSITIONS.get(self._state, set())
            if VmState.DESTROYING not in allowed_transitions:
                raise VmPermanentError(
                    f"Invalid state transition: {self._state.value} -> {VmState.DESTROYING.value}",
                    context={"vm_id": self.vm_id, "current_state": self._state.value},
                )
            self._state = VmState.DESTROYING

    async def confirm_destroyed(self) -> None:
        """Publish the terminal state only after process death is confirmed."""
        # Shield the VM-lifetime exit task from caller cancellation (a plain
        # await would cancel it), but absorb-then-re-raise the caller's own
        # cancellation instead of swallowing it: still publish DESTROYED, then
        # propagate the cancellation so structured cancellation observes it.
        cancellation: asyncio.CancelledError | None = None
        if self._process_exit_task is not None:
            cancellation = await await_settled(self._process_exit_task)
        async with self._state_lock:
            self._state = VmState.DESTROYED
            self._destroyed = True
        if cancellation is not None:
            raise cancellation

    async def destroy(self) -> bool:
        """Clean up VM and resources.

        Cleanup steps:
        1. Close communication channel
        2. Terminate QEMU and gvproxy processes (SIGTERM -> SIGKILL if needed)
        3. Remove cgroup + delete ephemeral overlay image (parallel)
        4. Re-close the channel (reaps workers resurrected by a racing op)
        5. Publish DESTROYED via confirm_destroyed()

        Called automatically by VmManager after execution or on error.
        Idempotent: safe to call multiple times.

        State Lock Strategy:
        - Cleanup lock serializes direct and manager-owned attempts
        - State lock protects DESTROYING/DESTROYED publication
        - A failed attempt remains retryable in DESTROYING
        """
        if self._release_callback is not None:
            return await self._release_callback(self)
        async with self._cleanup_lock:
            return await self._destroy_once()

    async def _destroy_once(self) -> bool:
        """Perform one serialized, retryable cleanup attempt."""
        if self._destroyed:
            logger.debug("VM already destroyed, skipping", extra={"vm_id": self.vm_id})
            return True
        await self.begin_destroy()

        # Cleanup operations outside lock (blocking I/O)
        # Step 1: Close communication channel
        with contextlib.suppress(OSError, RuntimeError):
            await self.channel.close()

        # Step 2: Terminate QEMU and gvproxy processes (SIGTERM -> SIGKILL)
        processes_destroyed = await cleanup_vm_processes(self.process, self.gvproxy_proc, self.vm_id)
        if not processes_destroyed:
            logger.error(
                "VM process cleanup is unconfirmed; retaining dependent resources",
                extra={"vm_id": self.vm_id},
            )
            return False

        # Step 3-4: Parallel cleanup (cgroup + workdir)
        # After QEMU terminates, cleanup tasks are independent
        # workdir.cleanup() removes overlay and sockets in one operation
        cgroup_cleaned, workdir_cleaned = await asyncio.gather(
            cgroup.cleanup_cgroup(self.cgroup_path, self.vm_id),
            self.workdir.cleanup(),
        )
        if not cgroup_cleaned or not workdir_cleaned:
            logger.error(
                "VM ancillary cleanup is unconfirmed; retaining retryable state",
                extra={
                    "vm_id": self.vm_id,
                    "cgroup_cleaned": cgroup_cleaned,
                    "workdir_cleaned": workdir_cleaned,
                },
            )
            return False

        # Re-close the channel after confirmed process death: an operation
        # racing this destroy can reconnect between the first close and the
        # kill, resurrecting write workers (see vm_manager.destroy_vm).
        with contextlib.suppress(OSError, RuntimeError):
            await self.channel.close()

        await self.confirm_destroyed()
        return True

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    async def capture_process_output(self) -> tuple[str, str]:
        """Capture stdout/stderr from QEMU process.

        Returns (stdout, stderr) as strings, empty if process still running.
        Must be called BEFORE destroy() — pipes are gone after cleanup.
        """
        if self.process.returncode is not None:
            try:
                stdout, stderr = await asyncio.wait_for(self.process.communicate(), timeout=1.0)
                return (stdout.decode() if stdout else "", stderr.decode() if stderr else "")
            except TimeoutError:
                pass
        return "", ""

    async def collect_diagnostics(self) -> QemuDiagnostics:
        """Collect crash diagnostics from this VM.

        Captures console ring buffer, process stdout/stderr, signal info,
        and host environment. Safe to call whether process is alive or dead:
        - Dead process: captures stdout/stderr via communicate(1s timeout)
        - Alive process: stdout/stderr will be empty (drain task owns the pipes)
        """
        stdout_text, stderr_text = await self.capture_process_output()

        console_log = "\n".join(self.console_lines) if self.console_lines else "(empty)"

        signal_name = ""
        rc = self.process.returncode
        if rc is not None and rc < 0:
            try:
                signal_name = signal.Signals(-rc).name
            except ValueError:
                signal_name = f"signal {-rc}"

        accel_type = await detect_accel_type()
        host_os = detect_host_os()

        return QemuDiagnostics(
            vm_id=self.vm_id,
            exit_code=rc,
            signal_name=signal_name,
            stdout=stdout_text[: constants.QEMU_OUTPUT_MAX_BYTES],
            stderr=stderr_text[: constants.QEMU_OUTPUT_MAX_BYTES],
            console_log=console_log,
            accel_type=accel_type.value,
            host_os=host_os.name.lower(),
        )

    # -------------------------------------------------------------------------
    # Guest readiness
    # -------------------------------------------------------------------------

    async def wait_for_guest(self, timeout: float) -> None:  # noqa: PLR0915
        """Wait for guest agent using event-driven racing.

        Races QEMU process death monitor against guest readiness checks with retry logic.

        Args:
            timeout: Maximum wait time in seconds

        Raises:
            VmQemuCrashError: QEMU process died during boot
            TimeoutError: Guest not ready within timeout
        """

        async def monitor_process_death() -> None:
            """Monitor QEMU process death - kernel-notified, instant."""
            await asyncio.shield(self._get_process_exit_task())
            diag = await self.collect_diagnostics()

            # macOS HVF: exit code 0 during boot = error (retry)
            if diag.host_os == "macos" and diag.exit_code == 0:
                logger.warning(
                    "QEMU exited with code 0 during boot on macOS (will retry)\n  vm_id=%s",
                    diag.vm_id,
                )
                raise VmQemuCrashError("QEMU process exited during boot (macOS clean exit)", diagnostics=diag)

            # TCG: exit code 0 during boot = guest reboot/panic
            if diag.accel_type == "tcg" and diag.exit_code == 0:
                logger.warning(
                    "QEMU TCG exited with code 0 during boot (will retry)\n"
                    "  vm_id=%s host_os=%s\n  stderr: %s\n  console:\n%s",
                    diag.vm_id,
                    diag.host_os,
                    diag.stderr[:500] if diag.stderr else "(empty)",
                    diag.console_log[-2000:],
                )
                raise VmQemuCrashError("QEMU TCG exited with code 0 during boot (guest reboot/panic)", diagnostics=diag)

            # General crash
            logger.error(
                "QEMU process exited unexpectedly\n"
                "  vm_id=%s exit_code=%s signal=%s\n"
                "  stderr: %s\n  stdout: %s\n  console:\n%s",
                diag.vm_id,
                diag.exit_code,
                diag.signal_name,
                diag.stderr if diag.stderr else "(empty)",
                diag.stdout if diag.stdout else "(empty)",
                diag.console_log,
            )
            raise VmQemuCrashError(
                f"QEMU process died (exit code {diag.exit_code}, {diag.signal_name}). "
                f"stderr: {diag.stderr[:200] if diag.stderr else '(empty)'}, "
                f"console: {diag.console_log[-4000:]}",
                diagnostics=diag,
            )

        async def check_guest_ready() -> None:
            """Single guest readiness check attempt."""
            await self.channel.connect(timeout_seconds=1)
            response = await self.channel.send_request(PingRequest())

            # Ping returns PongMessage
            if not isinstance(response, PongMessage):
                raise RuntimeError(f"Guest ping returned unexpected type: {type(response)}")

            logger.info("Guest agent ready", extra={"vm_id": self.vm_id, "version": response.version})

        # Race with retry logic (tenacity exponential backoff with full jitter)
        death_task: asyncio.Task[None] | None = None
        guest_task: asyncio.Task[None] | None = None
        # Store CM reference to check .expired() — CancelledError from child
        # tasks (guest_task) escapes asyncio.timeout() because the conversion
        # to TimeoutError only applies to the current task's cancellation.
        timeout_cm = asyncio.timeout(timeout)
        try:
            async with timeout_cm:
                death_task = asyncio.create_task(monitor_process_death())

                # Pre-connect to chardev sockets to trigger QEMU's poll registration.
                # Without this, QEMU may not add sockets to its poll set until after
                # guest opens virtio-serial ports, causing reads to return EOF.
                # See: https://bugs.launchpad.net/qemu/+bug/1224444 (virtio-mmio socket race)
                #
                # Timeout is short (1s vs previous 2s) because sockets are usually not ready this early.
                # The retry loop below handles actual connection with proper exponential backoff.
                # E3: Reduced pre-connect timeout from 0.1s to 0.01s — speculative, enters retry loop faster
                try:
                    await self.channel.connect(timeout_seconds=0.005)
                    logger.debug("Pre-connected to guest channel sockets", extra={"vm_id": self.vm_id})
                except (TimeoutError, OSError, CommunicationError, CommunicationOutcomeUnknownError) as e:
                    # Expected - sockets may not be ready yet, retry loop will handle
                    logger.debug("Pre-connect to sockets deferred", extra={"vm_id": self.vm_id, "reason": str(e)})

                # Retry with exponential backoff + full jitter
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(
                        (
                            TimeoutError,
                            OSError,
                            json.JSONDecodeError,
                            RuntimeError,
                            asyncio.IncompleteReadError,
                            CommunicationError,
                            CommunicationOutcomeUnknownError,
                        )
                    ),
                    # E1: Tighter retry backoff for faster guest detection
                    # E4: Reduced max from 0.2s to 0.05s — retries cap at 50ms intervals,
                    # catching guest readiness within ~10ms instead of ~150ms overshoot
                    wait=wait_random_exponential(multiplier=0.02, min=0.005, max=0.05),
                ):
                    with attempt:
                        guest_task = asyncio.create_task(check_guest_ready())

                        # Race: first one wins
                        done, _pending = await asyncio.wait(
                            {death_task, guest_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        # Check which completed
                        if death_task in done:
                            # QEMU died - cancel guest and retrieve exception
                            guest_task.cancel()
                            # Suppress ALL exceptions - we're about to re-raise VmError from death_task.
                            with contextlib.suppress(BaseException):
                                await guest_task
                            await death_task  # Re-raise VmError
                            # Safety net: monitor_process_death should always raise,
                            # but guard against future code paths that return normally.
                            diag = await self.collect_diagnostics()
                            raise VmQemuCrashError(
                                "QEMU process exited during boot (clean exit)",
                                diagnostics=diag,
                            )

                        # Guest task completed - check result (raises if failed, triggering retry)
                        # Wrap CancelledError from guest_task: the child task can be
                        # cancelled by gvproxy/socket failures or Python 3.14 asyncio
                        # internals. Convert to RuntimeError so tenacity retries it
                        # instead of letting it escape asyncio.timeout() uncaught
                        try:
                            await guest_task
                        except asyncio.CancelledError:
                            raise RuntimeError("Guest readiness check cancelled") from None

        except asyncio.CancelledError:
            # CancelledError from child tasks (guest_task) escapes asyncio.timeout()
            # because the CancelledError→TimeoutError conversion only applies to
            # the current task's cancellation, not child task cancellations.
            # Check if the timeout expired to distinguish from external cancellation.
            if timeout_cm.expired():
                raise TimeoutError(f"Guest agent not ready after {timeout}s") from None
            raise  # Genuine external cancellation — propagate as-is

        except TimeoutError:
            raise TimeoutError(f"Guest agent not ready after {timeout}s") from None

        finally:
            # Always clean up tasks to prevent "Task exception was never retrieved" warnings.
            for task in (death_task, guest_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (death_task, guest_task):
                if task is not None:
                    with contextlib.suppress(BaseException):
                        await task

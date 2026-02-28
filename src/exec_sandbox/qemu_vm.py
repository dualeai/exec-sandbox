"""QEMU VM handle for running microVMs.

Provides the QemuVM class which represents a running QEMU microVM and handles
code execution, state management, and resource cleanup.
"""

import asyncio
import contextlib
import json
import signal
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING
from uuid import uuid4

from pydantic import ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, wait_random_exponential

from exec_sandbox import cgroup, constants
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import (
    CodeValidationError,
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
from exec_sandbox.guest_channel import GuestChannel, consume_stream
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
        case (
            constants.GuestErrorType.PATH
            | constants.GuestErrorType.IO
            | constants.GuestErrorType.EXECUTION
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
        self._destroyed = False
        self._state = VmState.CREATING
        self._state_lock = asyncio.Lock()
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
        await self.destroy()
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
        - send_request(timeout=38s) - 30s soft + 8s margin
        - Guest enforces 30s, host kills at 38s if guest hangs

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
            VmPermanentError: VM not in READY state or communication failed
            VmBootTimeoutError: Execution exceeded timeout_seconds
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
            hard_timeout = timeout_seconds + constants.EXECUTION_TIMEOUT_MARGIN_SECONDS

            # Error handler: recoverable errors restore READY and raise.
            # Other errors fall through to consume_stream's default (append to stderr, break).
            async def _handle_exec_error(msg: StreamingErrorMessage) -> None:
                exc = guest_error_to_exception(msg, self.vm_id, operation="execute")
                if isinstance(exc, (InputValidationError, OutputLimitError, PackageNotAllowedError)):
                    with contextlib.suppress(VmPermanentError):
                        await self.transition_state(VmState.READY)
                    raise exc
                logger.error(
                    f"Guest agent error: [{msg.error_type}] {msg.message}",
                    extra={
                        "vm_id": self.vm_id,
                        "error_message": msg.message,
                        "error_type": msg.error_type,
                    },
                )

            result = await consume_stream(
                self.channel,
                request,
                timeout=hard_timeout,
                vm_id=self.vm_id,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
                on_error=_handle_exec_error,
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
        except TimeoutError as e:
            raise VmBootTimeoutError(
                f"VM {self.vm_id} execution exceeded {timeout_seconds}s timeout",
                context={
                    "vm_id": self.vm_id,
                    "timeout_seconds": timeout_seconds,
                    "language": self.language,
                },
            ) from e
        except (OSError, json.JSONDecodeError) as e:
            raise VmTransientError(
                f"VM {self.vm_id} communication failed: {e}",
                context={
                    "vm_id": self.vm_id,
                    "language": self.language,
                    "error_type": type(e).__name__,
                },
            ) from e

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    async def write_file(self, path: str, content: IO[bytes], *, make_executable: bool = False) -> None:  # noqa: PLR0915
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

        Args:
            path: Relative path in sandbox (PATH_MAX 4096, NAME_MAX 255 per component)
            content: Readable binary stream (BytesIO for bytes, open file for Path)
            make_executable: Set executable permission (0o755 vs 0o644)

        Raises:
            VmPermanentError: On validation or write failure
            VmTransientError: On timeout or communication failure
        """
        import base64  # noqa: PLC0415

        # Validate path
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

        op_id = uuid4().hex

        try:
            await self.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
            op_queue = await self.channel.register_op(op_id)

            # Send header frame
            header = WriteFileRequest(op_id=op_id, path=path, make_executable=make_executable)
            header_bytes = header.model_dump_json(by_alias=False, exclude_none=True).encode() + b"\n"
            await self.channel.enqueue_raw(header_bytes)

            # Stream-compress content chunk by chunk.  Each 128 KB slice is
            # read from the stream, compressed, and sent immediately.
            # file_end MUST be sent even on error, otherwise the guest agent
            # is left waiting for more chunks (stuck state).
            chunk_size = constants.FILE_TRANSFER_CHUNK_SIZE
            compressor = zstd.ZstdCompressor(level=constants.FILE_TRANSFER_ZSTD_LEVEL)
            loop = asyncio.get_running_loop()

            def _encode_frame(compressed: bytes) -> bytes:
                # TODO: Binary framing would eliminate this base64 encode overhead (~33% wire bloat)
                chunk_b64 = base64.b64encode(compressed).decode("ascii")
                return json.dumps({"action": "file_chunk", "op_id": op_id, "data": chunk_b64}).encode() + b"\n"

            # Combine read + compress into a single executor call to reduce
            # thread pool round-trips and enable look-ahead pipelining.
            def _read_and_compress(reader: IO[bytes]) -> bytes | None:
                raw = reader.read(chunk_size)
                if not raw:
                    return None  # EOF
                compressed = compressor.compress(raw)
                if not compressed:
                    return b""  # compressor buffered, no output yet
                return _encode_frame(compressed)

            try:
                # Look-ahead pipelining: start reading+compressing chunk N+1
                # in the thread pool while enqueueing chunk N on the event loop.
                prev_future = loop.run_in_executor(None, _read_and_compress, content)
                while True:
                    prev_frame = await prev_future
                    if prev_frame is None:
                        break  # EOF
                    # Start next read+compress while we enqueue the current frame
                    next_future = loop.run_in_executor(None, _read_and_compress, content)
                    if prev_frame:  # non-empty (not just compressor buffering)
                        await self.channel.enqueue_raw(prev_frame)
                    prev_future = next_future

                # Flush remaining compressed data from the compressor
                def _flush_and_encode() -> bytes | None:
                    remaining = compressor.flush()
                    return _encode_frame(remaining) if remaining else None

                flush_frame = await loop.run_in_executor(None, _flush_and_encode)
                if flush_frame:
                    await self.channel.enqueue_raw(flush_frame)
            finally:
                # Always send file_end so the guest agent exits its chunk loop.
                # Suppress errors: channel may already be broken.
                with contextlib.suppress(Exception):
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

        except TimeoutError as e:
            raise VmTransientError(
                f"VM {self.vm_id} write_file timed out for '{path}'",
                context={"vm_id": self.vm_id, "path": path},
            ) from e
        except (OSError, json.JSONDecodeError) as e:
            raise VmTransientError(
                f"VM {self.vm_id} write_file communication failed: {e}",
                context={"vm_id": self.vm_id, "path": path},
            ) from e
        finally:
            await self.channel.unregister_op(op_id)

    async def read_file(self, path: str, *, destination: Path) -> None:
        """Read a file from the sandbox via streaming zstd-compressed chunks.

        Protocol:
        1. Send ReadFileRequest (with op_id)
        2. Receive FileChunkResponseMessage messages (zstd-compressed)
        3. Receive FileReadCompleteMessage (end of stream)

        Each compressed chunk is decompressed immediately on arrival and
        written to a temp sibling of *destination* (``<dest>.<op_id>.tmp``).
        The op_id suffix prevents collisions between concurrent transfers
        targeting the same destination.  On success the temp file is
        atomically renamed to *destination*, so readers never see a
        partial / corrupted file.  On error the temp file is removed.
        Peak memory is bounded by queue depths (OP_QUEUE_DEPTH items x ~200KB),
        not file size.

        Args:
            path: Relative path in sandbox (PATH_MAX 4096, NAME_MAX 255 per component)
            destination: Local file path to stream decompressed content into

        Raises:
            VmPermanentError: On not-found or validation failure
            VmTransientError: On timeout or communication failure
        """
        import base64  # noqa: PLC0415

        op_id = uuid4().hex
        # Temp path scoped by op_id: foo.bin -> foo.bin.<op_id>.tmp
        # Prevents collisions when concurrent reads target the same destination.
        tmp_dest = destination.with_suffix(f"{destination.suffix}.{op_id}.tmp")

        try:
            await self.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
            op_queue = await self.channel.register_op(op_id)

            # Send read request
            request = ReadFileRequest(op_id=op_id, path=path)
            request_bytes = request.model_dump_json(by_alias=False, exclude_none=True).encode() + b"\n"
            await self.channel.enqueue_raw(request_bytes)

            # Stream decompressed chunks to .tmp file.
            # All disk I/O is offloaded to the thread pool to avoid blocking
            # the event loop (decompressed chunks can be large).
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: destination.parent.mkdir(parents=True, exist_ok=True))
            sink = await loop.run_in_executor(None, tmp_dest.open, "wb")
            decompressor = zstd.ZstdDecompressor()
            try:
                while True:
                    msg = await asyncio.wait_for(op_queue.get(), timeout=constants.FILE_IO_TIMEOUT_SECONDS)

                    if isinstance(msg, FileChunkResponseMessage):
                        # TODO: Binary framing would eliminate this base64 decode overhead (~33% wire bloat)
                        compressed_chunk = base64.b64decode(msg.data)
                        decompressed = decompressor.decompress(compressed_chunk)
                        await loop.run_in_executor(None, sink.write, decompressed)
                    elif isinstance(msg, FileReadCompleteMessage):
                        break
                    elif isinstance(msg, StreamingErrorMessage):
                        raise guest_error_to_exception(msg, self.vm_id, operation=f"read_file '{path}'")
                    else:
                        raise VmPermanentError(
                            f"read_file unexpected message type: {type(msg).__name__}",
                            context={"vm_id": self.vm_id, "path": path},
                        )
            finally:
                await loop.run_in_executor(None, sink.close)

            # Atomic rename — readers never see a partial file.
            await loop.run_in_executor(None, tmp_dest.rename, destination)

        except TimeoutError as e:
            raise VmTransientError(
                f"VM {self.vm_id} read_file timed out for '{path}'",
                context={"vm_id": self.vm_id, "path": path},
            ) from e
        except (OSError, json.JSONDecodeError) as e:
            raise VmTransientError(
                f"VM {self.vm_id} read_file communication failed: {e}",
                context={"vm_id": self.vm_id, "path": path},
            ) from e
        finally:
            # Clean up partial .tmp on any failure (no-op after successful rename)
            tmp_dest.unlink(missing_ok=True)
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
        except (OSError, json.JSONDecodeError) as e:
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

    async def destroy(self) -> None:
        """Clean up VM and resources.

        Cleanup steps:
        1. Close communication channel
        2. Terminate QEMU process (SIGTERM -> SIGKILL if needed)
        3. Remove cgroup
        4. Delete ephemeral overlay image

        Called automatically by VmManager after execution or on error.
        Idempotent: safe to call multiple times.

        State Lock Strategy:
        - Lock held during state check + transition to DESTROYING
        - Released during cleanup (blocking I/O operations)
        - DESTROYING state prevents concurrent destroy() from proceeding
        """
        # Atomic state check and transition (prevent concurrent destroy)
        async with self._state_lock:
            if self._destroyed:
                logger.debug("VM already destroyed, skipping", extra={"vm_id": self.vm_id})
                return

            # Set destroyed flag immediately to prevent concurrent destroy
            self._destroyed = True

            # Validate transition to DESTROYING (inline to avoid lock re-acquisition)
            allowed_transitions = VALID_STATE_TRANSITIONS.get(self._state, set())
            if VmState.DESTROYING not in allowed_transitions:
                raise VmPermanentError(
                    f"Invalid state transition: {self._state.value} -> {VmState.DESTROYING.value}",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                        "target_state": VmState.DESTROYING.value,
                        "allowed_transitions": [s.value for s in allowed_transitions],
                    },
                )

            old_state = self._state
            self._state = VmState.DESTROYING
            logger.debug(
                "VM state transition",
                extra={
                    "debug_category": "lifecycle",
                    "vm_id": self.vm_id,
                    "old_state": old_state.value,
                    "new_state": self._state.value,
                },
            )

        # Cleanup operations outside lock (blocking I/O)
        # Step 1: Close communication channel
        with contextlib.suppress(OSError, RuntimeError):
            await self.channel.close()

        # Step 2: Terminate QEMU and gvproxy processes (SIGTERM -> SIGKILL)
        await cleanup_vm_processes(self.process, self.gvproxy_proc, self.vm_id)

        # Step 3-4: Parallel cleanup (cgroup + workdir)
        # After QEMU terminates, cleanup tasks are independent
        # workdir.cleanup() removes overlay and sockets in one operation
        await asyncio.gather(
            cgroup.cleanup_cgroup(self.cgroup_path, self.vm_id),
            self.workdir.cleanup(),
            return_exceptions=True,
        )

        # Final state transition (acquires lock again - safe for same task)
        await self.transition_state(VmState.DESTROYED)

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
            await self.process.wait()
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
                except (TimeoutError, OSError) as e:
                    # Expected - sockets may not be ready yet, retry loop will handle
                    logger.debug("Pre-connect to sockets deferred", extra={"vm_id": self.vm_id, "reason": str(e)})

                # Retry with exponential backoff + full jitter
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(
                        (TimeoutError, OSError, json.JSONDecodeError, RuntimeError, asyncio.IncompleteReadError)
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

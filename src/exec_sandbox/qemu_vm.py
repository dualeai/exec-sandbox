"""QEMU VM handle for running microVMs.

Provides the QemuVM class which represents a running QEMU microVM and handles
code execution, state management, and resource cleanup.
"""

import asyncio
import contextlib
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, TYPE_CHECKING, TextIO
from uuid import uuid4

from exec_sandbox import cgroup, constants
from exec_sandbox.exceptions import VmBootTimeoutError, VmPermanentError, VmTransientError
from exec_sandbox.guest_agent_protocol import (
    ExecuteCodeRequest,
    FileChunkResponseMessage,
    FileListMessage,
    FileReadCompleteMessage,
    FileWriteAckMessage,
    ListFilesRequest,
    ReadFileRequest,
    StreamingErrorMessage,
    WriteFileRequest,
)
from exec_sandbox.guest_channel import GuestChannel
from exec_sandbox.models import ExecutionResult, ExposedPort, FileInfo, Language, TimingBreakdown
from exec_sandbox.resource_cleanup import cleanup_vm_processes
from exec_sandbox.vm_timing import VmTiming
from exec_sandbox.vm_types import VALID_STATE_TRANSITIONS, VmState
from exec_sandbox.vm_working_directory import VmWorkingDirectory

if TYPE_CHECKING:
    from exec_sandbox.platform_utils import ProcessWrapper

logger = logging.getLogger(__name__)

# Use native zstd module (Python 3.14+) or backports.zstd
if sys.version_info >= (3, 14):
    from compression import zstd  # type: ignore[import-not-found]
else:
    from backports import zstd  # type: ignore[import-untyped,no-redef]


class QemuVM:
    """Handle to running QEMU microVM.

    Lifecycle managed by VmManager.
    Communicates via GuestChannel (dual-port virtio-serial).

    Security:
    - Layer 1: Hardware isolation (KVM) or TCG software emulation
    - Layer 2: Unprivileged user (qemu-vm if available, optional)
    - Layer 3: Seccomp syscall filtering
    - Layer 4: cgroup v2 resource limits
    - Layer 5: Linux namespaces (PID, net, mount, UTS, IPC)
    - Layer 6: SELinux/AppArmor (optional production hardening)

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
        cgroup_path: Path,
        workdir: VmWorkingDirectory,
        channel: GuestChannel,
        language: Language,
        gvproxy_proc: "ProcessWrapper | None" = None,
        qemu_log_task: asyncio.Task[None] | None = None,
        gvproxy_log_task: asyncio.Task[None] | None = None,
        console_log: TextIO | None = None,
    ):
        """Initialize VM handle.

        Args:
            vm_id: Unique VM identifier (scoped by tenant_id)
            process: Running QEMU subprocess (ProcessWrapper for PID-reuse safety)
            cgroup_path: cgroup v2 path for cleanup
            workdir: Working directory containing overlay, sockets, and logs
            channel: Communication channel for guest agent
            language: Programming language for this VM
            gvproxy_proc: Optional gvproxy-wrapper process (ProcessWrapper)
            qemu_log_task: Background task draining QEMU stdout/stderr (prevents pipe deadlock)
            gvproxy_log_task: Background task draining gvproxy stdout/stderr (prevents pipe deadlock)
            console_log: Optional file handle for QEMU console log
        """
        self.vm_id = vm_id
        self.process = process
        self.cgroup_path = cgroup_path
        self.workdir = workdir
        self.channel = channel
        self.language = language
        self.gvproxy_proc = gvproxy_proc
        self.qemu_log_task = qemu_log_task
        self.gvproxy_log_task = gvproxy_log_task
        self.console_log: TextIO | None = console_log
        self._destroyed = False
        self._state = VmState.CREATING
        self._state_lock = asyncio.Lock()
        # Timing instrumentation (set by VmManager.create_vm)
        self.timing = VmTiming()
        # Tracks if this VM owns a semaphore permit (prevents double-release in destroy)
        self.holds_semaphore_slot = False
        # Port forwarding - set by VmManager after boot
        # Stores the resolved port mappings for result reporting
        self.exposed_ports: list[ExposedPort] = []

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
            on_stdout: Optional callback for real-time stdout streaming
            on_stderr: Optional callback for real-time stderr streaming

        Returns:
            ExecutionResult with stdout, stderr, exit code, and resource usage

        Raises:
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
        request = ExecuteCodeRequest(
            language=self.language,
            code=code,
            timeout=timeout_seconds,
            env_vars=env_vars or {},
        )

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
            connect_start = asyncio.get_event_loop().time()
            await self.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
            connect_ms = round((asyncio.get_event_loop().time() - connect_start) * 1000)

            # Stream execution output to console
            # Hard timeout = soft timeout (guest enforcement) + margin (host watchdog)
            hard_timeout = timeout_seconds + constants.EXECUTION_TIMEOUT_MARGIN_SECONDS

            # Stream messages and collect output
            exit_code = -1
            execution_time_ms: int | None = None
            spawn_ms: int | None = None
            process_ms: int | None = None
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            async for msg in self.channel.stream_messages(request, timeout=hard_timeout):
                # Type-safe message handling
                from exec_sandbox.guest_agent_protocol import (  # noqa: PLC0415
                    ExecutionCompleteMessage,
                    OutputChunkMessage,
                    StreamingErrorMessage,
                )

                if isinstance(msg, OutputChunkMessage):
                    # Collect chunk for return to user
                    if msg.type == "stdout":
                        stdout_chunks.append(msg.chunk)
                        # Call streaming callback if provided
                        if on_stdout:
                            on_stdout(msg.chunk)
                    else:  # stderr
                        stderr_chunks.append(msg.chunk)
                        # Call streaming callback if provided
                        if on_stderr:
                            on_stderr(msg.chunk)

                    # Also log for debugging (truncated)
                    logger.debug(
                        "VM output",
                        extra={
                            "vm_id": self.vm_id,
                            "stream": msg.type,
                            "chunk": msg.chunk[:200],
                        },
                    )
                elif isinstance(msg, ExecutionCompleteMessage):
                    # Execution complete - capture all timing fields
                    exit_code = msg.exit_code
                    execution_time_ms = msg.execution_time_ms
                    spawn_ms = msg.spawn_ms
                    process_ms = msg.process_ms
                elif isinstance(msg, StreamingErrorMessage):
                    # Streaming error from guest - include details in log message
                    logger.error(
                        f"Guest agent error: [{msg.error_type}] {msg.message}",
                        extra={
                            "vm_id": self.vm_id,
                            "error_message": msg.message,
                            "error_type": msg.error_type,
                        },
                    )
                    # Store error in stderr so callers can see what went wrong
                    stderr_chunks.append(f"[{msg.error_type}] {msg.message}")
                    exit_code = -1
                    break

            # Measure external resources from host (cgroup v2)
            external_cpu_ms, external_mem_mb = await self._read_cgroup_stats()

            # Concatenate collected chunks
            stdout_full = "".join(stdout_chunks)
            stderr_full = "".join(stderr_chunks)

            # Truncate to limits
            stdout_truncated = stdout_full[: constants.MAX_STDOUT_SIZE]
            stderr_truncated = stderr_full[: constants.MAX_STDERR_SIZE]

            # Debug log final execution output
            logger.debug(
                "Code execution complete",
                extra={
                    "vm_id": self.vm_id,
                    "exit_code": exit_code,
                    "execution_time_ms": execution_time_ms,
                    "stdout_len": len(stdout_full),
                    "stderr_len": len(stderr_full),
                    "stdout": stdout_truncated[:500],  # First 500 chars for debug
                    "stderr": stderr_truncated[:500] if stderr_truncated else None,
                },
            )

            # Parse result with both internal (guest) and external (host) measurements
            # Note: timing is a placeholder here - scheduler will populate actual values
            exec_result = ExecutionResult(
                stdout=stdout_truncated,  # Return to user
                stderr=stderr_truncated,  # Return to user
                exit_code=exit_code,
                execution_time_ms=execution_time_ms,  # Guest-reported
                external_cpu_time_ms=external_cpu_ms or None,  # Host-measured
                external_memory_peak_mb=external_mem_mb or None,  # Host-measured
                timing=TimingBreakdown(setup_ms=0, boot_ms=0, execute_ms=0, total_ms=0, connect_ms=connect_ms),
                spawn_ms=spawn_ms,  # Guest-reported granular timing
                process_ms=process_ms,  # Guest-reported granular timing
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
            try:
                while True:
                    # Offload to thread pool: content may be a SpooledTemporaryFile
                    # that has spilled to disk, blocking the event loop.
                    input_chunk = await loop.run_in_executor(None, content.read, chunk_size)
                    if not input_chunk:
                        break
                    compressed = compressor.compress(input_chunk)
                    if compressed:
                        chunk_b64 = base64.b64encode(compressed).decode("ascii")
                        chunk_frame = (
                            json.dumps({"action": "file_chunk", "op_id": op_id, "data": chunk_b64}).encode() + b"\n"
                        )
                        await self.channel.enqueue_raw(chunk_frame)

                # Flush remaining compressed data from the compressor
                remaining = compressor.flush()
                if remaining:
                    chunk_b64 = base64.b64encode(remaining).decode("ascii")
                    chunk_frame = (
                        json.dumps({"action": "file_chunk", "op_id": op_id, "data": chunk_b64}).encode() + b"\n"
                    )
                    await self.channel.enqueue_raw(chunk_frame)
            finally:
                # Always send file_end so the guest agent exits its chunk loop.
                # Suppress errors: channel may already be broken.
                with contextlib.suppress(Exception):
                    end_frame = json.dumps({"action": "file_end", "op_id": op_id}).encode() + b"\n"
                    await self.channel.enqueue_raw(end_frame)

            # Await ack from op_queue
            response = await asyncio.wait_for(op_queue.get(), timeout=constants.FILE_IO_TIMEOUT_SECONDS)

            if isinstance(response, StreamingErrorMessage):
                raise VmPermanentError(
                    f"write_file failed for '{path}': [{response.error_type}] {response.message}",
                    context={"vm_id": self.vm_id, "path": path, "error_type": response.error_type},
                )

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
        Peak memory is bounded by queue depths (4 items x ~200KB = ~800KB),
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
                        compressed_chunk = base64.b64decode(msg.data)
                        decompressed = decompressor.decompress(compressed_chunk)
                        await loop.run_in_executor(None, sink.write, decompressed)
                    elif isinstance(msg, FileReadCompleteMessage):
                        break
                    elif isinstance(msg, StreamingErrorMessage):
                        raise VmPermanentError(
                            f"read_file failed for '{path}': [{msg.error_type}] {msg.message}",
                            context={"vm_id": self.vm_id, "path": path, "error_type": msg.error_type},
                        )
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
            raise VmPermanentError(
                f"list_files failed for '{path}': [{response.error_type}] {response.message}",
                context={"vm_id": self.vm_id, "path": path, "error_type": response.error_type},
            )

        if not isinstance(response, FileListMessage):
            raise VmPermanentError(
                f"list_files unexpected response type: {type(response).__name__}",
                context={"vm_id": self.vm_id, "path": path},
            )

        return [FileInfo(name=e.name, is_dir=e.is_dir, size=e.size) for e in response.entries]

    async def _read_cgroup_stats(self) -> tuple[int | None, int | None]:
        """Read external CPU time and peak memory from cgroup v2.

        Returns:
            Tuple of (cpu_time_ms, peak_memory_mb)
            Returns (None, None) if cgroup not available or read fails
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
        # workdir.cleanup() removes overlay, sockets, and console log in one operation
        await asyncio.gather(
            cgroup.cleanup_cgroup(self.cgroup_path, self.vm_id),
            self.workdir.cleanup(),
            return_exceptions=True,
        )

        # Final state transition (acquires lock again - safe for same task)
        await self.transition_state(VmState.DESTROYED)

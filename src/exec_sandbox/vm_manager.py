"""QEMU microVM lifecycle management with multi-layer security.

Architecture:
- Supports Linux with KVM or TCG acceleration
- 6-layer security: KVM + unprivileged + seccomp + cgroups + namespaces + MAC
- qcow2 snapshot-based boot <400ms
- TCP host-guest communication

Performance Optimizations (QEMU 10.0+):
- CPU host passthrough (KVM): Enables all host CPU features (AVX2, AES-NI)
- Memory preallocation: Eliminates page fault latency during code execution
- virtio-blk: 4K blocks, num-queues=1, queue-size=256
- virtio-net: multiqueue off, TCP offload disabled (simpler for short VMs)
- Drive tuning: detect-zeroes=unmap, copy-on-read off, werror/rerror explicit
- Machine: mem-merge off (no KSM), dump-guest-core off
- io_uring AIO: Modern Linux async I/O (probed at startup, threads fallback)
- cache=unsafe: Safe for ephemeral VMs, major I/O performance boost
- microvm fast shutdown: -no-reboot + triple-fault for ~1-2s cleanup
"""

import asyncio
import ctypes
import errno
import hashlib
import json
import logging
import os
import platform
import signal
import time
from enum import Enum
from pathlib import Path
from uuid import uuid4

import aiofiles
import aiofiles.os
from tenacity import AsyncRetrying, before_sleep_log, retry_if_exception_type, wait_random_exponential

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.dns_filter import generate_dns_zones_json
from exec_sandbox.exceptions import VmError, VmTimeoutError
from exec_sandbox.guest_agent_protocol import (
    ExecuteCodeRequest,
    PingRequest,
    PongMessage,
)
from exec_sandbox.guest_channel import DualPortChannel, GuestChannel
from exec_sandbox.models import ExecutionResult, Language
from exec_sandbox.platform_utils import HostOS, ProcessWrapper, detect_host_os
from exec_sandbox.resource_cleanup import (
    cleanup_cgroup,
    cleanup_file,
    cleanup_overlay,
    cleanup_process,
)
from exec_sandbox.settings import Settings
from exec_sandbox.subprocess_utils import drain_subprocess_output

logger = get_logger(__name__)


def _get_gvproxy_wrapper_path() -> Path:
    """Get path to gvproxy-wrapper binary for current platform.

    Returns:
        Path to the platform-specific gvproxy-wrapper binary

    Raises:
        VmError: Binary not found for current platform
    """
    # Detect OS
    host_os = detect_host_os()
    if host_os == HostOS.MACOS:
        os_name = "darwin"
    elif host_os == HostOS.LINUX:
        os_name = "linux"
    else:
        raise VmError("Unsupported OS for gvproxy-wrapper")

    # Detect architecture
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        arch = "arm64"
    elif machine in ("x86_64", "amd64"):
        arch = "amd64"
    else:
        raise VmError(f"Unsupported architecture for gvproxy-wrapper: {machine}")

    # Construct path relative to package
    binary_name = f"gvproxy-wrapper-{os_name}-{arch}"
    # Look in gvproxy-wrapper/bin/ relative to repo root
    repo_root = Path(__file__).parent.parent.parent
    binary_path = repo_root / "gvproxy-wrapper" / "bin" / binary_name

    if not binary_path.exists():
        raise VmError(f"gvproxy-wrapper binary not found: {binary_path}. Run 'make build' to build it.")

    return binary_path


class VmState(Enum):
    """VM lifecycle states."""

    CREATING = "creating"
    BOOTING = "booting"
    READY = "ready"
    EXECUTING = "executing"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


# Valid state transitions for VM lifecycle
VALID_STATE_TRANSITIONS: dict[VmState, set[VmState]] = {
    VmState.CREATING: {VmState.BOOTING, VmState.DESTROYING},
    VmState.BOOTING: {VmState.READY, VmState.DESTROYING},
    VmState.READY: {VmState.EXECUTING, VmState.DESTROYING},
    VmState.EXECUTING: {VmState.READY, VmState.DESTROYING},
    VmState.DESTROYING: {VmState.DESTROYED},
    VmState.DESTROYED: set(),  # Terminal state - no transitions allowed
}

# Validate all states have transition rules defined
assert set(VmState) == set(VALID_STATE_TRANSITIONS.keys()), (
    f"Missing states in transition table: {set(VmState) - set(VALID_STATE_TRANSITIONS.keys())}"
)


def _check_kvm_available() -> bool:
    """Check if KVM acceleration is available.

    KVM vs TCG: Virtualization modes with vastly different characteristics
    ======================================================================

    KVM (Kernel-based Virtual Machine) - Production mode:
    - Hardware-assisted virtualization (Intel VT-x / AMD-V)
    - VM boot time: <400ms (with snapshot cache)
    - CPU overhead: near-native performance (~5% penalty)
    - Security: Hardware-enforced memory isolation (EPT/NPT)
    - Requirements: Linux host + KVM kernel module + /dev/kvm device
    - Use case: Production deployments, CI/CD

    TCG (Tiny Code Generator) - Development fallback:
    - Software-based CPU emulation (no hardware virtualization)
    - VM boot time: 2-5s (5-10x slower than KVM)
    - CPU overhead: 10-50x slower (instruction-level emulation)
    - Security: Software-based isolation (weaker than hardware)
    - Requirements: Any platform (Linux, macOS, Windows)
    - Use case: Development/testing only (macOS Docker Desktop)

    Production requirement: KVM is MANDATORY for performance and security.
    TCG is acceptable ONLY for local development and testing.

    Returns:
        True if /dev/kvm exists and is accessible (enables KVM mode)
        False otherwise (falls back to TCG software emulation)
    """
    return Path("/dev/kvm").exists()


class QemuVM:
    """Handle to running QEMU microVM.

    Lifecycle managed by VmManager.
    Communicates via TCP.

    Security:
    - Layer 1: Hardware isolation (KVM) or TCG software emulation
    - Layer 2: Unprivileged user (UID 1000, no root)
    - Layer 3: Seccomp syscall filtering
    - Layer 4: cgroup v2 resource limits
    - Layer 5: Linux namespaces (PID, net, mount, UTS, IPC)
    - Layer 6: SELinux/AppArmor (optional production hardening)

    Context Manager Usage:
        Supports async context manager protocol for automatic cleanup:

        ```python
        async with await manager.launch_vm(...) as vm:
            result = await vm.execute(code="print('hello')", language="python", timeout_seconds=30)
            # VM automatically destroyed on exit, even if exception occurs
        ```

        Manual cleanup still available via destroy() method for explicit control.

    Attributes:
        vm_id: Unique VM identifier format: {tenant_id}-{task_id}-{uuid4}
        process: QEMU subprocess handle
        cgroup_path: cgroup v2 path for resource limits
        overlay_image: Ephemeral qcow2 overlay (deleted on destroy)
        gvproxy_proc: Optional gvproxy-wrapper process for DNS filtering
        gvproxy_socket: Optional QEMU stream socket path
        gvproxy_log_task: Optional background task draining gvproxy stdout/stderr
    """

    def __init__(
        self,
        vm_id: str,
        process: ProcessWrapper,
        cgroup_path: Path,
        overlay_image: Path,
        channel: GuestChannel,
        language: str,
        gvproxy_proc: ProcessWrapper | None = None,
        gvproxy_socket: Path | None = None,
        qemu_log_task: asyncio.Task[None] | None = None,
        gvproxy_log_task: asyncio.Task[None] | None = None,
        console_log: object | None = None,
    ):
        """Initialize VM handle.

        Args:
            vm_id: Unique VM identifier (scoped by tenant_id)
            process: Running QEMU subprocess (ProcessWrapper for PID-reuse safety)
            cgroup_path: cgroup v2 path for cleanup
            overlay_image: Ephemeral qcow2 overlay for cleanup
            channel: Communication channel for TCP guest agent
            language: Programming language for this VM
            gvproxy_proc: Optional gvproxy-wrapper process (ProcessWrapper)
            gvproxy_socket: Optional QEMU stream socket path
            qemu_log_task: Background task draining QEMU stdout/stderr (prevents pipe deadlock)
            gvproxy_log_task: Background task draining gvproxy stdout/stderr (prevents pipe deadlock)
            console_log: Optional file handle for QEMU console log
        """
        self.vm_id = vm_id
        self.process = process
        self.cgroup_path = cgroup_path
        self.overlay_image = overlay_image
        self.channel = channel
        self.language = language
        self.gvproxy_proc = gvproxy_proc
        self.gvproxy_socket = gvproxy_socket
        self.qemu_log_task = qemu_log_task
        self.gvproxy_log_task = gvproxy_log_task
        self.console_log = console_log
        self._destroyed = False
        self._state = VmState.CREATING
        self._state_lock = asyncio.Lock()
        self._creation_time = asyncio.get_event_loop().time()

    async def __aenter__(self) -> "QemuVM":
        """Enter async context manager.

        Returns:
            Self for use in async with statement
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        """Exit async context manager - ensure cleanup.

        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred

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

    async def _transition_state(self, new_state: VmState) -> None:
        """Transition VM to new state with validation.

        Validates state transition against VALID_STATE_TRANSITIONS to prevent
        invalid state changes (e.g., DESTROYED → READY).

        Args:
            new_state: Target state to transition to

        Raises:
            VmError: If transition is invalid for current state
        """
        async with self._state_lock:
            # Validate transition is allowed from current state
            allowed_transitions = VALID_STATE_TRANSITIONS.get(self._state, set())
            if new_state not in allowed_transitions:
                raise VmError(
                    f"Invalid state transition: {self._state.value} → {new_state.value}",
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

    async def execute(
        self,
        code: str,
        language: Language,
        timeout_seconds: int,
        env_vars: dict[str, str] | None = None,
        on_stdout: callable | None = None,
        on_stderr: callable | None = None,
    ) -> ExecutionResult:
        """Execute code via TCP guest agent communication.

        Implementation:
        1. Connect to guest via TCP (127.0.0.1 + allocated port)
        2. Send execution request JSON with action, language, code, timeout, env_vars
        3. Wait for result with timeout (cgroup enforced)
        4. Parse result: stdout, stderr, exit_code, memory_mb, execution_time_ms

        Timeout Architecture (3-layer system):
        1. Init timeout (5s): Connection establishment to guest agent
        2. Soft timeout (timeout_seconds): Guest agent enforcement (sent in request)
        3. Hard timeout (timeout_seconds + 2s): Host watchdog protection

        Example: timeout_seconds=30
        - connect(5s) - Fixed init window for socket establishment
        - send_request(timeout=32s) - 30s soft + 2s margin
        - Guest enforces 30s, host kills at 32s if guest hangs

        Args:
            code: Code to execute in guest VM
            language: Programming language (python or javascript)
            timeout_seconds: Maximum execution time (enforced by cgroup)
            env_vars: Environment variables for code execution (default: None)
            on_stdout: Optional callback for real-time stdout streaming
            on_stderr: Optional callback for real-time stderr streaming

        Returns:
            ExecutionResult with stdout, stderr, exit code, and resource usage

        Raises:
            VmError: VM not in READY state or communication failed
            VmTimeoutError: Execution exceeded timeout_seconds
        """
        # Validate VM is in READY state before execution (atomic check-and-set)
        async with self._state_lock:
            if self._state != VmState.READY:
                raise VmError(
                    f"Cannot execute in state {self._state.value}, must be READY",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                        "language": language,
                    },
                )

            # Validate transition to EXECUTING (inline to avoid lock re-acquisition)
            allowed_transitions = VALID_STATE_TRANSITIONS.get(self._state, set())
            if VmState.EXECUTING not in allowed_transitions:
                raise VmError(
                    f"Invalid state transition: {self._state.value} → {VmState.EXECUTING.value}",
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

        # Start timing
        exec_start_time = time.time()

        # Prepare execution request
        request = ExecuteCodeRequest(
            language=language,
            code=code,
            timeout=timeout_seconds,
            env_vars=env_vars or {},
        )

        try:
            # Connect to guest via TCP
            # Fixed init timeout (connection establishment, independent of execution timeout)
            await self.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)

            # Stream execution output to console
            # Hard timeout = soft timeout (guest enforcement) + margin (host watchdog)
            hard_timeout = timeout_seconds + constants.EXECUTION_TIMEOUT_MARGIN_SECONDS

            # Stream messages and collect output
            exit_code = -1
            execution_time_ms = None
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            async for msg in self.channel.stream_messages(request, timeout=hard_timeout):
                # Type-safe message handling
                from exec_sandbox.guest_agent_protocol import (
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
                    # Execution complete
                    exit_code = msg.exit_code
                    execution_time_ms = msg.execution_time_ms
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
            external_cpu_ms, external_mem_mb = self._read_cgroup_stats()

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
            exec_result = ExecutionResult(
                stdout=stdout_truncated,  # Return to user
                stderr=stderr_truncated,  # Return to user
                exit_code=exit_code,
                execution_time_ms=execution_time_ms,  # Guest-reported
                external_cpu_time_ms=external_cpu_ms or None,  # Host-measured
                external_memory_peak_mb=external_mem_mb or None,  # Host-measured
            )

            # Success - transition back to READY for reuse (if not destroyed)
            try:
                await self._transition_state(VmState.READY)
            except VmError as e:
                # VM destroyed while executing, skip transition
                logger.debug(
                    "VM destroyed during execution, skipping READY transition",
                    extra={"vm_id": self.vm_id, "error": str(e)},
                )
            return exec_result

        except asyncio.CancelledError:
            logger.warning(
                "Code execution cancelled",
                extra={"vm_id": self.vm_id, "language": language},
            )
            # Re-raise to propagate cancellation
            raise
        except TimeoutError as e:
            raise VmTimeoutError(
                f"VM {self.vm_id} execution exceeded {timeout_seconds}s timeout",
                context={
                    "vm_id": self.vm_id,
                    "timeout_seconds": timeout_seconds,
                    "language": language,
                },
            ) from e
        except (OSError, json.JSONDecodeError) as e:
            raise VmError(
                f"VM {self.vm_id} communication failed: {e}",
                context={
                    "vm_id": self.vm_id,
                    "language": language,
                    "error_type": type(e).__name__,
                },
            ) from e

    def _read_cgroup_stats(self) -> tuple[int | None, int | None]:
        """Read external CPU time and peak memory from cgroup v2.

        Returns:
            Tuple of (cpu_time_ms, peak_memory_mb)
            Returns (None, None) if cgroup not available or read fails
        """
        if not self.cgroup_path or not self.cgroup_path.exists():
            return (None, None)

        cpu_time_ms: int | None = None
        peak_memory_mb: int | None = None

        try:
            # Read cpu.stat for usage_usec (microseconds)
            cpu_stat_file = self.cgroup_path / "cpu.stat"
            if cpu_stat_file.exists():
                cpu_stat = cpu_stat_file.read_text()
                for line in cpu_stat.splitlines():
                    if line.startswith("usage_usec"):
                        usage_usec = int(line.split()[1])
                        cpu_time_ms = usage_usec // 1000  # Convert to milliseconds
                        break

            # Read memory.peak for peak memory usage (bytes)
            memory_peak_file = self.cgroup_path / "memory.peak"
            if memory_peak_file.exists():
                peak_bytes = int(memory_peak_file.read_text().strip())
                peak_memory_mb = peak_bytes // (1024 * 1024)  # Convert to MB

        except (OSError, ValueError) as e:
            logger.debug(
                f"Failed to read cgroup stats: {e}",
                extra={"vm_id": self.vm_id, "cgroup_path": str(self.cgroup_path)},
            )

        return (cpu_time_ms, peak_memory_mb)

    async def destroy(self) -> None:
        """Clean up VM and resources.

        Cleanup steps:
        1. Close communication channel
        2. Terminate QEMU process (SIGTERM → SIGKILL if needed)
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
                raise VmError(
                    f"Invalid state transition: {self._state.value} → {VmState.DESTROYING.value}",
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
        try:
            await self.channel.close()
        except (OSError, RuntimeError):
            pass

        # Step 3-4: Parallel cleanup (cgroup + overlay)
        # After QEMU terminates, cleanup tasks are independent
        async def cleanup_cgroup_fn():
            """Remove cgroup asynchronously."""
            if self.cgroup_path.exists():
                try:
                    # Move all processes to parent cgroup first
                    parent_cgroup = self.cgroup_path.parent / "cgroup.procs"
                    if parent_cgroup.exists():
                        async with aiofiles.open(self.cgroup_path / "cgroup.procs") as f:
                            pids = (await f.read()).strip().split("\n")
                        for pid in pids:
                            if pid:
                                try:
                                    async with aiofiles.open(parent_cgroup, "w") as f:
                                        await f.write(pid)
                                except (OSError, PermissionError):
                                    pass

                    # Remove cgroup directory
                    await aiofiles.os.rmdir(self.cgroup_path)
                except (OSError, PermissionError):
                    pass

        async def cleanup_overlay_fn():
            """Delete overlay image asynchronously."""
            if self.overlay_image.exists():
                try:
                    await aiofiles.os.remove(self.overlay_image)
                except OSError:
                    pass

        # Run cleanup tasks in parallel
        await asyncio.gather(
            cleanup_cgroup_fn(),
            cleanup_overlay_fn(),
            return_exceptions=True,
        )

        # Final state transition (acquires lock again - safe for same task)
        await self._transition_state(VmState.DESTROYED)


class VmManager:
    """QEMU microVM lifecycle manager with cross-platform support.

    Architecture:
    - Runtime detection: KVM or TCG acceleration
    - qcow2 snapshot-based boot with CoW overlays
    - 6-layer security architecture
    - TCP guest agent communication
    - cgroup v2 resource limits

    Usage:
        manager = VmManager(settings)
        vm = await manager.create_vm("python", "tenant-123", "task-456")
        result = await vm.execute("print('hello')", "python", timeout_seconds=30)
        await manager.destroy_vm(vm)
    """

    def __init__(self, settings: Settings):
        """Initialize QEMU manager.

        Args:
            settings: Service configuration (paths, limits, etc.)

        Note on crash recovery:
            VM registry is in-memory only. If service crashes, registry is lost
            but QEMU processes may still be running. On restart:
            - Registry initializes empty (logged below)
            - Zombie QEMU processes are orphaned (no cleanup attempted)
            - Orphaned VMs timeout naturally (max runtime: 2 min)
        """
        self.settings = settings
        self.machine = platform.machine()  # "x86_64", "arm64", "aarch64"

        # Probe io_uring support ONCE at startup
        self._io_uring_available = self._probe_io_uring_support()

        self._vms: dict[str, QemuVM] = {}  # vm_id → VM object
        self._vms_lock = asyncio.Lock()  # Protect registry access

        # Log registry initialization (empty on startup, even after crash)
        logger.info(
            "VM registry initialized",
            extra={
                "max_concurrent_vms": self.settings.max_concurrent_vms,
                "io_uring_available": self._io_uring_available,
                "note": "Any zombie QEMU processes from crashes will timeout naturally (max 2min)",
            },
        )

    def get_active_vms(self) -> dict[str, QemuVM]:
        """Get snapshot of active VMs (for debugging/metrics).

        Returns:
            Copy of VM registry (vm_id → QemuVM)
        """
        return dict(self._vms)

    async def create_vm(
        self,
        language: str,
        tenant_id: str,
        task_id: str,
        snapshot_path: Path | None = None,
        memory_mb: int = constants.DEFAULT_MEMORY_MB,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
    ) -> QemuVM:
        """Create and boot QEMU microVM from snapshot.

        Workflow:
        1. Generate unique VM ID and CID
        2. Create ephemeral qcow2 overlay from snapshot
        3. Set up cgroup v2 resource limits
        4. Build QEMU command (platform-specific)
        5. Launch QEMU subprocess
        6. Wait for guest agent ready

        Args:
            language: Programming language (python or javascript)
            tenant_id: Tenant identifier for isolation
            task_id: Task identifier
            snapshot_path: Optional snapshot base image (default: use base image)
            memory_mb: Memory limit in MB (128-2048, default 512)
            allow_network: Enable network access (default: False, isolated)
            allowed_domains: Whitelist of allowed domains if allow_network=True

        Returns:
            QemuVM handle for code execution

        Raises:
            VmError: VM creation failed
            asyncio.TimeoutError: VM boot timeout (>5s)
        """
        # Start timing
        start_time = asyncio.get_event_loop().time()

        # Step 1: Generate VM identifiers
        vm_id = f"{tenant_id}-{task_id}-{uuid4()}"

        # Domain whitelist semantics:
        # - None or [] = no filtering (full internet access)
        # - list with domains = whitelist filtering via gvproxy
        logger.debug(
            "Network configuration",
            extra={
                "debug_category": "network",
                "vm_id": vm_id,
                "allow_network": allow_network,
                "allowed_domains": allowed_domains,
                "will_enable_filtering": bool(allowed_domains and len(allowed_domains) > 0),
            },
        )

        # Step 2-4: Parallel resource setup (overlay + cgroup + gvproxy)
        # These operations are independent and can run concurrently
        overlay_image = Path(f"/tmp/qemu-{vm_id}.qcow2")
        base_image = snapshot_path or self._get_base_image(language)

        # Initialize ALL tracking variables before try block for finally cleanup
        cgroup_path: Path | None = None
        gvproxy_proc: ProcessWrapper | None = None
        gvproxy_socket: Path | None = None
        gvproxy_log_task: asyncio.Task[None] | None = None
        qemu_proc: ProcessWrapper | None = None
        vm_created = False  # Flag to skip cleanup if VM successfully created

        # IMPORTANT: Always use gvproxy for network-enabled VMs
        # SLIRP user networking has reliability issues with containerized unprivileged execution
        enable_dns_filtering = allow_network  # Force gvproxy for all network-enabled VMs

        try:
            # Run independent setup tasks in parallel
            tasks = [
                self._create_overlay(base_image, overlay_image),
                self._setup_cgroup(vm_id, tenant_id),
            ]

            # Add gvproxy startup to parallel tasks if network enabled
            logger.debug(
                "Checking gvproxy condition",
                extra={
                    "debug_category": "network",
                    "vm_id": vm_id,
                    "allow_network": allow_network,
                    "allowed_domains": allowed_domains,
                    "domains_count": len(allowed_domains) if allowed_domains else 0,
                },
            )
            if allow_network:
                # Always use gvproxy for network-enabled VMs
                # Pass allowed_domains to gvproxy - None means use language defaults
                logger.info(
                    "Adding gvproxy-wrapper to parallel tasks",
                    extra={"vm_id": vm_id, "allowed_domains": allowed_domains},
                )
                tasks.append(self._start_gvproxy(vm_id, allowed_domains, language))

            # Wait for all setup tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Extract results and check for failures
            overlay_result = results[0]
            cgroup_result = results[1]

            # Check for overlay/cgroup failures
            if isinstance(overlay_result, Exception):
                raise overlay_result
            if isinstance(cgroup_result, Exception):
                raise cgroup_result

            # Type assertion for cgroup_result
            assert isinstance(cgroup_result, Path)
            cgroup_path = cgroup_result

            if len(results) == 3:  # gvproxy was started
                gvproxy_result = results[2]
                if isinstance(gvproxy_result, Exception):
                    raise gvproxy_result
                # Unpack tuple from _start_gvproxy
                assert isinstance(gvproxy_result, tuple) and len(gvproxy_result) == 3
                proc_item, socket_item, task_item = gvproxy_result
                assert isinstance(proc_item, ProcessWrapper)
                assert isinstance(socket_item, Path)
                assert isinstance(task_item, asyncio.Task)
                gvproxy_proc = proc_item
                gvproxy_socket = socket_item
                gvproxy_log_task = task_item

            # Step 5: Build QEMU command (always Linux in container)
            qemu_cmd = self._build_linux_cmd(
                language,
                vm_id,
                overlay_image,
                memory_mb,
                allow_network,
                enable_dns_filtering,
                gvproxy_socket,
            )

            # Step 6: Create dual-port Unix socket communication channel for guest agent
            cmd_socket = self._get_cmd_socket_path(vm_id)
            event_socket = self._get_event_socket_path(vm_id)
            channel: GuestChannel = DualPortChannel(cmd_socket, event_socket)

            # If cgroups unavailable, wrap with ulimit for host resource control
            # ulimit works on Linux, macOS, BSD (POSIX)
            cgroups_available = str(cgroup_path).startswith("/sys/fs/cgroup")
            if not cgroups_available:
                qemu_cmd = self._wrap_with_ulimit(qemu_cmd, memory_mb)

            # Step 7: Launch QEMU
            try:
                # DEBUG: Log the exact command
                last_arg = qemu_cmd[-1]
                last_arg_preview = last_arg[:200] if len(last_arg) > 200 else last_arg
                logger.debug(
                    "QEMU command with network",
                    extra={
                        "debug_category": "network",
                        "vm_id": vm_id,
                        "cmd_length": len(qemu_cmd),
                        "first_args": qemu_cmd[:3],
                        "last_arg_preview": last_arg_preview,
                    },
                )

                qemu_proc = ProcessWrapper(
                    await asyncio.create_subprocess_exec(
                        *qemu_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True,  # Create new process group for proper cleanup
                    )
                )

                # Background task to drain QEMU output (prevent 64KB pipe deadlock)
                console_log_path = Path(f"/tmp/vm-{vm_id}-console.log")
                console_log = console_log_path.open("w", buffering=1)  # Line buffering

                def write_to_console(line: str) -> None:
                    """Write line to console log file and structured logs."""
                    try:
                        console_log.write(f"[{vm_id}] {line}\n")
                    except Exception as e:
                        logger.error(f"Console write failed: {e}", extra={"context_id": vm_id})

                qemu_log_task = asyncio.create_task(
                    drain_subprocess_output(
                        qemu_proc,
                        process_name="QEMU",
                        context_id=vm_id,
                        stdout_handler=write_to_console,
                        stderr_handler=write_to_console,
                    )
                )
                qemu_log_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

                # Attach process to cgroup (only if cgroups available)
                if cgroups_available and qemu_proc.pid is not None:
                    await self._attach_to_cgroup(cgroup_path, qemu_proc.pid)

                # Check if process crashed immediately
                await asyncio.sleep(0.1)
                if qemu_proc.returncode is not None:
                    stdout_text, stderr_text = await self._capture_qemu_output(qemu_proc)

                    max_bytes = constants.QEMU_OUTPUT_MAX_BYTES
                    raise VmError(
                        f"QEMU crashed immediately (exit code {qemu_proc.returncode}). "
                        f"stderr: {stderr_text[:max_bytes]}, stdout: {stdout_text[:max_bytes]}",
                        context={
                            "vm_id": vm_id,
                            "language": language,
                            "exit_code": qemu_proc.returncode,
                            "memory_mb": memory_mb,
                            "allow_network": allow_network,
                            "qemu_cmd_preview": qemu_cmd[:5] if len(qemu_cmd) > 5 else qemu_cmd,
                        },
                    )

            except (OSError, FileNotFoundError) as e:
                raise VmError(
                    f"Failed to launch QEMU: {e}",
                    context={
                        "vm_id": vm_id,
                        "language": language,
                        "memory_mb": memory_mb,
                    },
                ) from e

            # Step 8: Wait for guest agent ready
            vm = QemuVM(
                vm_id,
                qemu_proc,
                cgroup_path,
                overlay_image,
                channel,
                language,
                gvproxy_proc,
                gvproxy_socket,
                qemu_log_task,
                gvproxy_log_task,
                console_log,
            )

            # Step 8a: Register VM in registry (before BOOTING to ensure tracking)
            async with self._vms_lock:
                if len(self._vms) >= self.settings.max_concurrent_vms:
                    raise VmError(f"VM pool full: {len(self._vms)}/{self.settings.max_concurrent_vms} VMs active")
                self._vms[vm.vm_id] = vm

            # Transition to BOOTING state
            await vm._transition_state(VmState.BOOTING)

            try:
                await self._wait_for_guest(vm, timeout=constants.VM_BOOT_TIMEOUT_SECONDS)
                # Transition to READY state after boot completes
                await vm._transition_state(VmState.READY)
            except TimeoutError as e:
                # Capture QEMU output for debugging
                stdout_text, stderr_text = await self._capture_qemu_output(qemu_proc)

                # Log output if available
                if stderr_text or stdout_text:
                    logger.error(
                        "QEMU output captured",
                        extra={
                            "stderr": stderr_text[: constants.QEMU_OUTPUT_MAX_BYTES],
                            "stdout": stdout_text[: constants.QEMU_OUTPUT_MAX_BYTES],
                        },
                    )
                else:
                    logger.error("QEMU process still running but guest agent not responding")

                await vm.destroy()

                raise VmError(
                    f"Guest agent not ready after {constants.VM_BOOT_TIMEOUT_SECONDS}s: {e}. stderr: {stderr_text[:200] if stderr_text else 'none'}",
                    context={
                        "vm_id": vm_id,
                        "language": language,
                        "timeout_seconds": constants.VM_BOOT_TIMEOUT_SECONDS,
                    },
                ) from e

            # Log boot time
            boot_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.info(
                "VM created",
                extra={
                    "vm_id": vm_id,
                    "language": language,
                    "boot_time_ms": boot_time_ms,
                },
            )

            # Mark VM as successfully created to skip cleanup in finally
            vm_created = True
            return vm

        finally:
            # Comprehensive cleanup on failure (vm_created flag prevents cleanup on success)
            if not vm_created:
                logger.info(
                    "VM creation failed, cleaning up resources",
                    extra={
                        "vm_id": vm_id,
                        "qemu_started": qemu_proc is not None,
                        "gvproxy_started": gvproxy_proc is not None,
                    },
                )
                # Remove from registry if it was added (defensive - always try)
                async with self._vms_lock:
                    self._vms.pop(vm_id, None)

                await self._force_cleanup_all_resources(
                    vm_id=vm_id,
                    qemu_proc=qemu_proc,
                    gvproxy_proc=gvproxy_proc,
                    gvproxy_socket=gvproxy_socket,
                    overlay_image=overlay_image,
                    cgroup_path=cgroup_path,
                )

    async def _start_gvproxy(
        self,
        vm_id: str,
        allowed_domains: list[str] | None,
        language: str,
    ) -> tuple[ProcessWrapper, Path, asyncio.Task[None]]:
        r"""Start gvproxy-wrapper with DNS filtering for this VM.

        Architecture Decision: gvisor-tap-vsock over alternatives
        ========================================================

        Chosen: gvisor-tap-vsock
        - ✅ Built-in DNS filtering via zones (regex-based)
        - ✅ Production-ready (Podman default since 2022)
        - ✅ 10MB memory overhead per VM
        - ✅ Simple JSON zone configuration
        - ✅ Zero CVEs (vs SLIRP: CVE-2021-3592/3/4/5, CVE-2020-29129/30)

        Args:
            vm_id: Unique VM identifier
            allowed_domains: Whitelist of allowed domains
            language: Programming language (for default registries)

        Returns:
            Tuple of (gvproxy_process, socket_path, gvproxy_log_task)

        Raises:
            VmError: Failed to start gvproxy-wrapper
        """
        socket_path = self._get_socket_path(vm_id, prefix="gvproxy")

        # Generate DNS zones JSON configuration
        dns_zones_json = generate_dns_zones_json(allowed_domains, language)

        logger.info(
            "Starting gvproxy-wrapper with DNS filtering",
            extra={
                "vm_id": vm_id,
                "allowed_domains": allowed_domains,
                "language": language,
                "dns_zones_json": dns_zones_json,
            },
        )

        # Start gvproxy-wrapper
        gvproxy_binary = _get_gvproxy_wrapper_path()
        try:
            proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    str(gvproxy_binary),
                    "-listen-qemu",
                    f"unix://{socket_path}",
                    "-dns-zones",
                    dns_zones_json,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,  # Create new process group for proper cleanup
                )
            )
        except (OSError, FileNotFoundError) as e:
            raise VmError(
                f"Failed to start gvproxy-wrapper: {e}",
                context={
                    "vm_id": vm_id,
                    "language": language,
                    "allowed_domains": allowed_domains,
                    "binary_path": str(gvproxy_binary),
                },
            ) from e

        # Background task to drain gvproxy output (prevent pipe deadlock)
        gvproxy_log_task = asyncio.create_task(
            drain_subprocess_output(
                proc,
                process_name="gvproxy-wrapper",
                context_id=vm_id,
                stdout_handler=lambda line: logger.debug("[gvproxy-wrapper]", extra={"vm_id": vm_id, "output": line}),
                stderr_handler=lambda line: logger.error(
                    "[gvproxy-wrapper error]", extra={"vm_id": vm_id, "output": line}
                ),
            )
        )
        gvproxy_log_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

        # Wait for socket creation (max 5 seconds)
        for _ in range(50):
            if socket_path.exists():
                break
            await asyncio.sleep(0.1)
        else:
            await proc.terminate()
            await proc.wait()
            raise VmError(
                f"gvproxy-wrapper socket not created: {socket_path}",
                context={
                    "vm_id": vm_id,
                    "language": language,
                    "socket_path": str(socket_path),
                    "allowed_domains": allowed_domains,
                },
            )

        # Fix socket permissions for qemu-vm user (UID 1000) to connect
        chmod_proc = ProcessWrapper(
            await asyncio.create_subprocess_exec(
                "chmod",
                "666",
                str(socket_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        )
        await chmod_proc.communicate()

        logger.info(
            "gvproxy-wrapper started successfully",
            extra={
                "vm_id": vm_id,
                "socket": str(socket_path),
                "dns_filtering": True,
            },
        )

        return proc, socket_path, gvproxy_log_task

    async def _force_cleanup_all_resources(
        self,
        vm_id: str,
        qemu_proc: ProcessWrapper | None = None,
        gvproxy_proc: ProcessWrapper | None = None,
        gvproxy_socket: Path | None = None,
        overlay_image: Path | None = None,
        cgroup_path: Path | None = None,
    ) -> dict[str, bool]:
        """Comprehensive cleanup of ALL VM resources in reverse dependency order.

        This is the MAIN cleanup method used in finally blocks.

        Best practices:
        - Cleans in reverse dependency order (processes → files → directories)
        - NEVER raises exceptions (logs errors instead)
        - Safe to call multiple times (idempotent)
        - Handles None/already-cleaned resources
        - Returns status dict for monitoring/debugging
        - Uses standardized generic cleanup functions

        Cleanup order (reverse dependencies):
        1. QEMU process (depends on: overlay, cgroup, networking)
        2. gvproxy process (QEMU networking dependency)
        3. gvproxy socket (gvproxy dependency)
        4. Overlay file (QEMU disk dependency)
        5. Cgroup directory (QEMU process was in it)

        Args:
            vm_id: VM identifier for logging
            qemu_proc: QEMU subprocess (can be None)
            gvproxy_proc: gvproxy subprocess (can be None)
            gvproxy_socket: gvproxy socket path (can be None)
            overlay_image: qcow2 overlay path (can be None)
            cgroup_path: cgroup directory path (can be None)

        Returns:
            Dictionary with cleanup status for each resource
        """
        logger.info("Starting comprehensive resource cleanup", extra={"vm_id": vm_id})
        results: dict[str, bool] = {}

        # Phase 1: Kill processes in parallel (independent operations)
        process_results = await asyncio.gather(
            cleanup_process(
                proc=qemu_proc,
                name="QEMU",
                context_id=vm_id,
                term_timeout=5.0,
                kill_timeout=2.0,
            ),
            cleanup_process(
                proc=gvproxy_proc,
                name="gvproxy",
                context_id=vm_id,
                term_timeout=3.0,
                kill_timeout=2.0,
            ),
            return_exceptions=True,
        )
        results["qemu"] = process_results[0] if isinstance(process_results[0], bool) else False
        results["gvproxy"] = process_results[1] if isinstance(process_results[1], bool) else False

        # Phase 2: Cleanup files in parallel (after processes dead)
        file_results = await asyncio.gather(
            cleanup_file(
                file_path=gvproxy_socket,
                context_id=vm_id,
                description="gvproxy socket",
            ),
            cleanup_overlay(
                overlay_path=overlay_image,
                context_id=vm_id,
            ),
            cleanup_cgroup(
                cgroup_path=cgroup_path,
                context_id=vm_id,
            ),
            return_exceptions=True,
        )
        results["socket"] = file_results[0] if isinstance(file_results[0], bool) else False
        results["overlay"] = file_results[1] if isinstance(file_results[1], bool) else False
        results["cgroup"] = file_results[2] if isinstance(file_results[2], bool) else False

        # Log summary
        success_count = sum(results.values())
        total_count = len(results)
        if success_count == total_count:
            logger.info("Cleanup completed successfully", extra={"vm_id": vm_id, "results": results})
        else:
            logger.warning(
                "Cleanup completed with errors",
                extra={
                    "vm_id": vm_id,
                    "results": results,
                    "success": success_count,
                    "total": total_count,
                },
            )

        return results

    async def destroy_vm(self, vm: QemuVM) -> None:
        """Destroy VM and clean up resources using defensive generic cleanup.

        This method uses the comprehensive cleanup orchestrator to ensure
        all resources are properly cleaned up even if some operations fail.

        Args:
            vm: QemuVM handle to destroy
        """
        try:
            # Close console log file before cancelling tasks
            if vm.console_log:
                try:
                    vm.console_log.close()
                except Exception:
                    pass

            # Cancel output reader tasks (prevent pipe deadlock during cleanup)
            if vm.qemu_log_task and not vm.qemu_log_task.done():
                vm.qemu_log_task.cancel()
                try:
                    await vm.qemu_log_task
                except asyncio.CancelledError:
                    pass  # Expected when cancelling

            if vm.gvproxy_log_task and not vm.gvproxy_log_task.done():
                vm.gvproxy_log_task.cancel()
                try:
                    await vm.gvproxy_log_task
                except asyncio.CancelledError:
                    pass  # Expected when cancelling

            # Destroy VM (transitions state, closes channel)
            await vm.destroy()

            # Comprehensive cleanup using defensive generic functions
            await self._force_cleanup_all_resources(
                vm_id=vm.vm_id,
                qemu_proc=vm.process,
                gvproxy_proc=vm.gvproxy_proc,
                gvproxy_socket=vm.gvproxy_socket,
                overlay_image=vm.overlay_image,
                cgroup_path=vm.cgroup_path,
            )
        finally:
            # ALWAYS remove from registry, even on failure
            async with self._vms_lock:
                self._vms.pop(vm.vm_id, None)

    async def _capture_qemu_output(self, process: ProcessWrapper) -> tuple[str, str]:
        """Capture stdout/stderr from QEMU process.

        Args:
            process: QEMU subprocess

        Returns:
            Tuple of (stdout, stderr) as strings, empty if process still running
        """
        if process.returncode is not None:
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=1.0)
                return (stdout.decode() if stdout else "", stderr.decode() if stderr else "")
            except TimeoutError:
                pass
        return "", ""

    def _get_socket_path(self, vm_id: str, prefix: str = "qemu-serial") -> Path:
        """Generate short socket path from VM ID using full SHA-256 hash.

        UNIX domain sockets have 108-byte path limit.

        Args:
            vm_id: Unique VM identifier
            prefix: Socket file prefix (default: qemu-serial)

        Returns:
            Path to socket file with hashed vm_id
        """
        hash_hex = hashlib.sha256(vm_id.encode()).hexdigest()
        return Path(f"/tmp/{prefix}-{hash_hex}.sock")

    def _get_cmd_socket_path(self, vm_id: str) -> str:
        """Get path to virtio-serial Unix socket for command channel (host → guest).

        Args:
            vm_id: Unique VM identifier

        Returns:
            /tmp/cmd-{hash}.sock
        """
        return str(self._get_socket_path(vm_id, prefix="cmd"))

    def _get_event_socket_path(self, vm_id: str) -> str:
        """Get path to virtio-serial Unix socket for event channel (guest → host).

        Args:
            vm_id: Unique VM identifier

        Returns:
            /tmp/event-{hash}.sock
        """
        return str(self._get_socket_path(vm_id, prefix="event"))

    def _get_base_image(self, language: str) -> Path:
        """Get base image path for language via auto-discovery.

        Auto-discovers images matching patterns:
        - python: python-*-base-*.qcow2
        - javascript: node-*-base-*.qcow2
        - raw: raw-base-*.qcow2

        Args:
            language: Programming language (python, javascript, or raw)

        Returns:
            Path to base qcow2 image

        Raises:
            VmError: Base image not found
        """
        # Pattern prefixes for each language
        patterns = {
            "python": "python-*-base-*.qcow2",
            "javascript": "node-*-base-*.qcow2",
            "raw": "raw-base-*.qcow2",
        }

        pattern = patterns.get(language)
        if not pattern:
            raise VmError(f"Unknown language: {language}")

        # Find matching images
        matches = list(self.settings.base_images_dir.glob(pattern))
        if not matches:
            raise VmError(
                f"Base image not found for language: {language}. "
                f"Pattern: {pattern}, dir: {self.settings.base_images_dir}"
            )

        # Return first match (sorted for determinism)
        return sorted(matches)[0]

    async def _create_overlay(self, base_image: Path, overlay_image: Path) -> None:
        """Create ephemeral qcow2 overlay with CoW backing file.

        Args:
            base_image: Base qcow2 image (backing file)
            overlay_image: Ephemeral overlay to create

        Raises:
            VmError: qemu-img command failed
        """
        cmd = [
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            str(base_image),
            "-o",
            "lazy_refcounts=on,extended_l2=on,cluster_size=128k",
            str(overlay_image),
        ]

        proc = ProcessWrapper(
            await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        )
        _stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise VmError(f"qemu-img create failed: {stderr.decode()}")

        # Change ownership to qemu-vm user (UID 1000) so QEMU process can read it
        # Skip on macOS (local development) where this isn't needed
        if detect_host_os() != HostOS.MACOS:
            chown_proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    "chown",
                    "1000:1000",
                    str(overlay_image),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            )
            await chown_proc.communicate()
            if chown_proc.returncode != 0:
                raise VmError("Failed to chown overlay image to qemu-vm user")

    async def _setup_cgroup(self, vm_id: str, tenant_id: str) -> Path:
        """Set up cgroup v2 resource limits.

        Limits:
        - memory.max: 512MB
        - cpu.max: 100000 (1 vCPU)
        - pids.max: 100 (fork bomb prevention)

        Args:
            vm_id: Unique VM identifier
            tenant_id: Tenant identifier

        Returns:
            Path to cgroup directory (dummy path if cgroups unavailable)

        Note:
            Gracefully degrades to no resource limits on Docker Desktop (read-only /sys/fs/cgroup)
            or environments without cgroup v2 support
        """
        cgroup_path = Path(f"/sys/fs/cgroup/code-exec/{tenant_id}/{vm_id}")

        try:
            await aiofiles.os.makedirs(cgroup_path, exist_ok=True)

            # Set memory limit (512MB default + overhead)
            memory_mb = 512
            async with aiofiles.open(cgroup_path / "memory.max", "w") as f:
                await f.write(str((memory_mb + constants.CGROUP_MEMORY_OVERHEAD_MB) * 1024 * 1024))

            # Set CPU limit (1 vCPU)
            async with aiofiles.open(cgroup_path / "cpu.max", "w") as f:
                await f.write("100000 100000")

            # Set PID limit (fork bomb prevention)
            async with aiofiles.open(cgroup_path / "pids.max", "w") as f:
                await f.write(str(constants.CGROUP_PIDS_LIMIT))

        except OSError as e:
            # Gracefully degrade if cgroups unavailable (e.g., Docker Desktop)
            if e.errno == constants.ERRNO_READ_ONLY_FILESYSTEM:
                logger.warning(
                    "cgroup v2 unavailable (read-only filesystem), resource limits disabled",
                    extra={"vm_id": vm_id, "path": str(cgroup_path)},
                )
                return Path(f"/tmp/cgroup-{vm_id}")
            raise VmError(f"Failed to setup cgroup: {e}") from e
        except PermissionError as e:
            raise VmError(f"Failed to setup cgroup: {e}") from e

        return cgroup_path

    async def _attach_to_cgroup(self, cgroup_path: Path, pid: int) -> None:
        """Attach process to cgroup.

        Args:
            cgroup_path: cgroup directory
            pid: Process ID to attach

        Raises:
            VmError: Failed to attach process
        """
        try:
            async with aiofiles.open(cgroup_path / "cgroup.procs", "w") as f:
                await f.write(str(pid))
        except (OSError, PermissionError) as e:
            raise VmError(f"Failed to attach PID {pid} to cgroup: {e}") from e

    def _probe_io_uring_support(self) -> bool:
        """Probe for io_uring support using syscall test.

        Returns:
            True if io_uring fully available, False otherwise
        """
        # Check 1: Sysctl restrictions (kernel 5.12+)
        sysctl_path = Path("/proc/sys/kernel/io_uring_disabled")
        if sysctl_path.exists():
            try:
                disabled_value = int(sysctl_path.read_text().strip())
                if disabled_value == 2:
                    logger.info(
                        "io_uring disabled via sysctl",
                        extra={"sysctl_value": disabled_value},
                    )
                    return False
                if disabled_value == 1:
                    logger.debug(
                        "io_uring restricted to CAP_SYS_ADMIN",
                        extra={"sysctl_value": disabled_value},
                    )
            except (ValueError, OSError) as e:
                logger.warning("Failed to read io_uring_disabled sysctl", extra={"error": str(e)})

        # Check 2: Syscall probe (most reliable)
        try:
            libc = ctypes.CDLL(None, use_errno=True)

            # __NR_io_uring_setup syscall number: 425
            NR_io_uring_setup = 425

            # Call io_uring_setup(0, NULL) - should fail with EINVAL if supported
            result = libc.syscall(NR_io_uring_setup, 0, None)

            if result == -1:
                err = ctypes.get_errno()
                if err == errno.ENOSYS:
                    logger.info(
                        "io_uring syscall not available (ENOSYS)",
                        extra={"kernel": platform.release()},
                    )
                    return False
                if err == errno.EINVAL:
                    logger.info(
                        "io_uring syscall available",
                        extra={"kernel": platform.release()},
                    )
                    return True
                if err == errno.EPERM:
                    logger.warning(
                        "io_uring blocked by seccomp/container policy",
                        extra={"errno": err},
                    )
                    return False
                logger.warning(
                    "io_uring probe failed with unexpected error",
                    extra={"errno": err, "error": os.strerror(err)},
                )
                return False

            return True

        except Exception as e:
            logger.warning(
                "io_uring syscall probe failed",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            return False

    def _wrap_with_ulimit(self, qemu_cmd: list[str], memory_mb: int) -> list[str]:
        """Wrap QEMU command with ulimit for resource control (cgroups alternative).

        Args:
            qemu_cmd: Original QEMU command
            memory_mb: Guest memory in MB

        Returns:
            Command wrapped with ulimit via sh -c
        """
        import shlex

        qemu_cmd_str = " ".join(shlex.quote(arg) for arg in qemu_cmd)

        # QEMU overhead: ~1.5-2x guest memory for KVM
        virtual_mem_kb = memory_mb * 1024 * constants.ULIMIT_MEMORY_MULTIPLIER

        # Platform-specific: macOS kernel doesn't support modifying virtual memory limits
        if detect_host_os() == HostOS.MACOS:
            shell_cmd = f"exec {qemu_cmd_str}"
        else:
            shell_cmd = f"ulimit -v {virtual_mem_kb} && exec {qemu_cmd_str}"

        return ["sh", "-c", shell_cmd]

    def _build_linux_cmd(
        self,
        language: str,
        vm_id: str,
        overlay_image: Path,
        memory_mb: int,
        allow_network: bool,
        enable_dns_filtering: bool = False,
        gvproxy_socket: Path | None = None,
    ) -> list[str]:
        """Build QEMU command for Linux (KVM + unshare + namespaces).

        Args:
            language: Programming language
            vm_id: Unique VM identifier
            overlay_image: Ephemeral qcow2 overlay
            memory_mb: Guest VM memory in MB
            allow_network: Enable network access
            enable_dns_filtering: Enable DNS filtering via gvisor-tap-vsock
            gvproxy_socket: QEMU stream socket path for gvproxy connection

        Returns:
            QEMU command as list of strings
        """
        # Determine QEMU binary, machine type, and kernel based on architecture
        is_macos = detect_host_os() == HostOS.MACOS

        if self.machine in ("arm64", "aarch64"):
            arch_suffix = "aarch64"
            qemu_bin = "qemu-system-aarch64"
            if is_macos:
                machine_type = "virt,mem-merge=off"
            else:
                machine_type = "virt,mem-merge=off,dump-guest-core=off"
        else:
            arch_suffix = "x86_64"
            qemu_bin = "qemu-system-x86_64"
            if is_macos:
                machine_type = "microvm,x-option-roms=off,pit=off,pic=off,rtc=off,mem-merge=off"
            else:
                machine_type = "microvm,x-option-roms=off,pit=off,pic=off,rtc=off,mem-merge=off,dump-guest-core=off"

        # Auto-discover kernel and initramfs based on architecture
        kernel_path = self.settings.kernel_path / f"vmlinuz-{arch_suffix}"
        initramfs_path = self.settings.kernel_path / f"initramfs-{arch_suffix}"

        # Layer 5: Linux namespaces
        cmd = []
        if detect_host_os() != HostOS.MACOS:
            if allow_network:
                unshare_args = ["unshare", "--pid", "--mount", "--uts", "--ipc", "--fork"]
                cmd.extend([*unshare_args, "--"])
            else:
                unshare_args = ["unshare", "--pid", "--net", "--mount", "--uts", "--ipc", "--fork"]
                cmd.extend([*unshare_args, "--"])

        # Detect hardware acceleration availability
        use_kvm = _check_kvm_available()

        if is_macos:
            accel = "hvf"
        elif use_kvm:
            accel = "kvm"
        else:
            accel = "tcg"

        # Build QEMU command arguments
        qemu_args = [
            qemu_bin,
            "-accel",
            accel,
            "-cpu",
            ("host" if accel in ("hvf", "kvm") else "cortex-a57" if self.machine in ("arm64", "aarch64") else "qemu64"),
            "-M",
            machine_type,
            "-no-reboot",
            "-m",
            f"{memory_mb}M",
            "-smp",
            "1",
            "-kernel",
            str(kernel_path),
            "-initrd",
            str(initramfs_path),
            "-append",
            # Optimized boot params: quiet console, no logging, skip PS/2 probing
            "console=hvc0 loglevel=0 quiet root=/dev/vda rootflags=rw,noatime rootfstype=ext4 rootwait=2 fsck.mode=skip reboot=t preempt=none nomodules i8042.noaux i8042.nomux i8042.nopnp init=/init",
        ]

        # Platform-specific memory configuration
        # Note: -mem-prealloc removed for faster boot (demand-paging is fine for ephemeral VMs)
        host_os = detect_host_os()

        # Layer 3: Seccomp sandbox - Linux only
        if detect_host_os() != HostOS.MACOS:
            qemu_args.extend(
                [
                    "-sandbox",
                    "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny",
                ]
            )

        # Determine AIO mode based on cached startup probe
        aio_mode = "io_uring" if self._io_uring_available else "threads"
        if not self._io_uring_available:
            logger.debug(
                "Using aio=threads (io_uring not available)",
                extra={"reason": "syscall_probe_failed", "vm_id": vm_id},
            )

        # IOThread configuration
        match host_os:
            case HostOS.LINUX:
                use_iothread = True
            case HostOS.MACOS | HostOS.UNKNOWN:
                use_iothread = False

        iothread_id = f"iothread0-{vm_id}" if use_iothread else None
        if use_iothread:
            qemu_args.extend(["-object", f"iothread,id={iothread_id}"])

        # Disk configuration
        qemu_args.extend(
            [
                "-drive",
                f"file={overlay_image},"
                f"format=qcow2,"
                f"if=none,"
                f"id=hd0,"
                f"cache=unsafe,"
                f"aio={aio_mode},"
                f"discard=unmap,"
                f"detect-zeroes=unmap,"
                f"werror=report,"
                f"rerror=report,"
                f"copy-on-read=off,"
                f"bps={constants.DISK_BPS_LIMIT},"
                f"bps_max={constants.DISK_BPS_BURST},"
                f"iops={constants.DISK_IOPS_LIMIT},"
                f"iops_max={constants.DISK_IOPS_BURST}",
            ]
        )

        # Platform-specific block device
        match host_os:
            case HostOS.MACOS:
                qemu_args.extend(
                    [
                        "-device",
                        "virtio-blk-device,drive=hd0,num-queues=1,queue-size=128",
                    ]
                )
            case HostOS.LINUX | HostOS.UNKNOWN:
                qemu_args.extend(
                    [
                        "-device",
                        f"virtio-blk-device,drive=hd0,iothread={iothread_id},num-queues=1,queue-size=128,logical_block_size=4096,physical_block_size=4096",
                    ]
                )

        # Display/console configuration
        qemu_args.extend(
            [
                "-nographic",
            ]
        )

        # virtio-serial device for guest agent communication
        cmd_socket_path = self._get_cmd_socket_path(vm_id)
        event_socket_path = self._get_event_socket_path(vm_id)
        qemu_args.extend(
            [
                "-chardev",
                f"socket,id=cmd0,path={cmd_socket_path},server=on,wait=off",
                "-chardev",
                f"socket,id=event0,path={event_socket_path},server=on,wait=off",
                "-device",
                # Reduced from 31 to 4: only using 2 ports (cmd+event), saves ~4KB/port + virtqueue init time
                "virtio-serial-device,max_ports=4",
                "-device",
                "virtserialport,chardev=cmd0,name=org.dualeai.cmd,nr=1",
                "-device",
                "virtserialport,chardev=event0,name=org.dualeai.event,nr=2",
            ]
        )

        # virtio-net configuration (optional, internet access only)
        if allow_network:
            if not gvproxy_socket:
                raise RuntimeError(f"gvproxy socket required for network-enabled VM but not provided (vm_id={vm_id})")
            qemu_args.extend(
                [
                    "-netdev",
                    f"stream,id=net0,addr.type=unix,addr.path={gvproxy_socket}",
                    "-device",
                    "virtio-net-device,netdev=net0,mq=off,csum=off,gso=off,host_tso4=off,host_tso6=off,mrg_rxbuf=off,ctrl_rx=off,guest_announce=off",
                ]
            )

        # Run QEMU as unprivileged user (Linux production) or directly (macOS development)
        if detect_host_os() != HostOS.MACOS:
            cmd.extend(["sudo", "-u", "qemu-vm"])

        cmd.extend(qemu_args)

        return cmd

    async def _wait_for_guest(self, vm: QemuVM, timeout: float) -> None:
        """Wait for guest agent using event-driven racing.

        Races QEMU process death monitor against guest readiness checks with retry logic.

        Args:
            vm: QemuVM handle
            timeout: Maximum wait time in seconds

        Raises:
            VmError: QEMU process died
            asyncio.TimeoutError: Guest not ready within timeout
        """

        async def monitor_process_death() -> None:
            """Monitor QEMU process death - kernel-notified, instant."""
            await vm.process.wait()

            # macOS HVF: Clean QEMU exit (code 0) is expected with -no-reboot
            host_os = detect_host_os()
            match host_os:
                case HostOS.MACOS if vm.process.returncode == 0:
                    logger.info(
                        "QEMU process exited cleanly (expected on macOS HVF with -no-reboot)",
                        extra={"vm_id": vm.vm_id, "exit_code": 0},
                    )
                    return
                case _:
                    pass

            # Process died - capture output
            stdout_text, stderr_text = await self._capture_qemu_output(vm.process)
            signal_name = ""
            if vm.process.returncode and vm.process.returncode < 0:
                sig = -vm.process.returncode
                signal_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else f"signal {sig}"

            logger.error(
                "QEMU process exited unexpectedly",
                extra={
                    "vm_id": vm.vm_id,
                    "exit_code": vm.process.returncode,
                    "signal": signal_name,
                    "stdout": stdout_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stdout_text else "",
                    "stderr": stderr_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stderr_text else "",
                },
            )
            stderr_preview = stderr_text[:200] if stderr_text else ""
            raise VmError(
                f"QEMU process died (exit code {vm.process.returncode}, {signal_name}). stderr: {stderr_preview}"
            )

        async def check_guest_ready() -> None:
            """Single guest readiness check attempt."""
            await vm.channel.connect(timeout_seconds=5)
            response = await vm.channel.send_request(PingRequest(), timeout=5)

            # Ping returns PongMessage
            if not isinstance(response, PongMessage):
                raise RuntimeError(f"Guest ping returned unexpected type: {type(response)}")

            logger.info("Guest agent ready", extra={"vm_id": vm.vm_id, "version": response.version})

        # Race with retry logic (tenacity exponential backoff with full jitter)
        death_task: asyncio.Task[None] | None = None
        guest_task: asyncio.Task[None] | None = None
        try:
            async with asyncio.timeout(timeout):
                death_task = asyncio.create_task(monitor_process_death())

                # Retry with exponential backoff + full jitter
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(
                        (TimeoutError, OSError, json.JSONDecodeError, RuntimeError, asyncio.IncompleteReadError)
                    ),
                    # Reduced min from 0.1s to 0.01s for faster guest detection (agent ready in ~200-300ms)
                    wait=wait_random_exponential(multiplier=0.05, min=0.01, max=1.0),
                    before_sleep=before_sleep_log(logger, logging.DEBUG),
                ):
                    with attempt:
                        guest_task = asyncio.create_task(check_guest_ready())

                        # Race: first one wins
                        done, pending = await asyncio.wait(
                            {death_task, guest_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        # Check which completed
                        if death_task in done:
                            # QEMU died - cancel guest and retrieve exception
                            guest_task.cancel()
                            try:
                                await guest_task
                            except (asyncio.CancelledError, Exception):
                                pass
                            await death_task  # Re-raise VmError

                        # Guest task completed - check result (raises if failed, triggering retry)
                        await guest_task

                # Success - cancel death monitor
                death_task.cancel()
                try:
                    await death_task
                except asyncio.CancelledError:
                    pass

        except TimeoutError:
            # Clean up in-flight tasks
            for task in (death_task, guest_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (death_task, guest_task):
                if task is not None:
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
            console_log_path = Path(f"/tmp/vm-{vm.vm_id}-console.log")
            console_output = "none"
            if console_log_path.exists():
                try:
                    console_output = console_log_path.read_text()[: constants.CONSOLE_LOG_MAX_BYTES]
                except Exception:
                    console_output = "failed to read console log"

            logger.error(
                "Guest agent timeout",
                extra={
                    "vm_id": vm.vm_id,
                    "timeout": timeout,
                    "qemu_running": vm.process.returncode is None,
                    "console_output": console_output,
                },
            )

            raise TimeoutError(f"Guest agent not ready after {timeout}s")

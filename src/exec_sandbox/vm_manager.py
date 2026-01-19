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
import contextlib
import logging
import os
import re
import signal
from pathlib import Path
from typing import TextIO
from uuid import uuid4

import aiofiles.os
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from exec_sandbox import cgroup, constants
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import (
    VmBootTimeoutError,
    VmConfigError,
    VmDependencyError,
    VmOverlayError,
    VmQemuCrashError,
)
from exec_sandbox.guest_agent_protocol import PingRequest, PongMessage
from exec_sandbox.guest_channel import DualPortChannel, GuestChannel
from exec_sandbox.gvproxy import start_gvproxy
from exec_sandbox.models import Language
from exec_sandbox.overlay_pool import OverlayPool, QemuImgError
from exec_sandbox.permission_utils import (
    chmod_async,
    chown_to_qemu_vm,
    ensure_traversable,
    get_qemu_vm_uid,
    probe_sudo_as_qemu_vm,
)
from exec_sandbox.platform_utils import HostArch, HostOS, ProcessWrapper, detect_host_arch, detect_host_os
from exec_sandbox.qemu_cmd import build_qemu_cmd
from exec_sandbox.qemu_storage_daemon import QemuStorageDaemonError
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.resource_cleanup import cleanup_process
from exec_sandbox.settings import Settings
from exec_sandbox.subprocess_utils import drain_subprocess_output, log_task_exception, read_log_tail
from exec_sandbox.system_probes import (
    check_tsc_deadline,
    detect_accel_type,
    probe_io_uring_support,
    probe_qemu_version,
    probe_unshare_support,
)
from exec_sandbox.validation import validate_kernel_initramfs
from exec_sandbox.vm_types import AccelType, VmState
from exec_sandbox.vm_working_directory import VmWorkingDirectory

logger = get_logger(__name__)

# Security: Identifier validation pattern
# Only alphanumeric, underscore, and hyphen allowed to prevent:
# - Shell command injection via malicious tenant_id/task_id
# - Path traversal attacks (no '..', '/')
# - Socket path manipulation
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_IDENTIFIER_MAX_LENGTH = 128  # Reasonable limit for identifiers

# QEMU binary extraction pattern for error diagnostics
# Extracts binary name from shell wrapper commands (e.g., "qemu-system-x86_64")
_QEMU_BINARY_PATTERN = re.compile(r"(qemu-system-[^\s]+)")


def _validate_identifier(value: str, name: str) -> None:
    """Validate identifier contains only safe characters.

    Prevents shell injection and path traversal attacks by ensuring identifiers
    (tenant_id, task_id) contain only alphanumeric characters, underscores, and hyphens.

    Args:
        value: The identifier value to validate
        name: Human-readable name for error messages

    Raises:
        ValueError: If identifier contains invalid characters or is too long
    """
    if not value:
        raise ValueError(f"{name} cannot be empty")
    if len(value) > _IDENTIFIER_MAX_LENGTH:
        raise ValueError(f"{name} too long: {len(value)} > {_IDENTIFIER_MAX_LENGTH}")
    if not _IDENTIFIER_PATTERN.match(value):
        raise ValueError(f"{name} contains invalid characters (only [a-zA-Z0-9_-] allowed): {value!r}")


class VmManager:
    """QEMU microVM lifecycle manager with cross-platform support.

    Architecture:
    - Runtime detection: KVM or TCG acceleration
    - qcow2 snapshot-based boot with CoW overlays
    - 6-layer security architecture
    - TCP guest agent communication
    - cgroup v2 resource limits

    Usage:
        async with VmManager(settings) as manager:
            vm = await manager.create_vm(Language.PYTHON, "tenant-123", "task-456")
            result = await vm.execute("print('hello')", timeout_seconds=30)
            await manager.destroy_vm(vm)
    """

    def __init__(self, settings: Settings):
        """Initialize QEMU manager (sync part only).

        Args:
            settings: Service configuration (paths, limits, etc.)

        Note: Call `await start()` after construction to run async system probes.

        Note on crash recovery:
            VM registry is in-memory only. If service crashes, registry is lost
            but QEMU processes may still be running. On restart:
            - Registry initializes empty (logged below)
            - Zombie QEMU processes are orphaned (no cleanup attempted)
            - Orphaned VMs timeout naturally (max runtime: 2 min)
        """
        self.settings = settings
        self.arch = detect_host_arch()
        self._initialized = False

        self._vms: dict[str, QemuVM] = {}  # vm_id -> VM object
        self._vms_lock = asyncio.Lock()  # Protect registry access
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_vms)  # Backpressure

        # Overlay pool for fast VM boot (auto-manages base image discovery and pooling)
        self._overlay_pool = OverlayPool(
            max_concurrent_vms=settings.max_concurrent_vms,
            images_path=settings.base_images_dir,
        )

    async def start(self) -> None:
        """Start VmManager and run async system probes.

        This method runs all async system capability probes and caches their results
        at module level. This prevents cache stampede when multiple VMs start
        concurrently - all probes are pre-warmed here instead of racing during VM creation.

        Must be called before creating VMs.
        """
        if self._initialized:
            return

        # Run all async probes concurrently (they cache their results at module level)
        # This prevents cache stampede when multiple VMs start concurrently
        accel_type, io_uring_available, unshare_available, qemu_version = await asyncio.gather(
            self._detect_accel_type(),  # Pre-warms HVF/KVM + QEMU accelerator caches
            probe_io_uring_support(),
            probe_unshare_support(),
            probe_qemu_version(),  # Pre-warm QEMU version for netdev reconnect
        )

        # Pre-warm TSC deadline (unified function handles arch/OS dispatch)
        await check_tsc_deadline()

        # Pre-flight check: validate kernel and initramfs exist (cached)
        await validate_kernel_initramfs(self.settings.kernel_path, self.arch)

        # Start overlay pool (discovers base images internally)
        await self._overlay_pool.start()

        self._initialized = True

        # Log registry initialization (empty on startup, even after crash)
        logger.info(
            "VM registry initialized",
            extra={
                "max_concurrent_vms": self.settings.max_concurrent_vms,
                "accel_type": accel_type.value,
                "io_uring_available": io_uring_available,
                "unshare_available": unshare_available,
                "qemu_version": ".".join(map(str, qemu_version)) if qemu_version else None,
                "note": "All system probes pre-warmed (stampede prevention)",
            },
        )

    def get_active_vms(self) -> dict[str, QemuVM]:
        """Get snapshot of active VMs (for debugging/metrics).

        Returns:
            Copy of VM registry (vm_id -> QemuVM)
        """
        return dict(self._vms)

    async def stop(self) -> None:
        """Stop VmManager and cleanup resources (overlay pool).

        Should be called when the VmManager is no longer needed.
        """
        await self._overlay_pool.stop()

    async def __aenter__(self) -> "VmManager":
        """Enter async context manager, starting the manager."""
        await self.start()
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, stopping the manager."""
        await self.stop()

    async def create_vm(
        self,
        language: Language,
        tenant_id: str,
        task_id: str,
        backing_image: Path | None = None,
        memory_mb: int = constants.DEFAULT_MEMORY_MB,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
        direct_write_target: Path | None = None,
    ) -> QemuVM:
        """Create and boot QEMU microVM with automatic retry on transient failures.

        Wraps _create_vm_impl with tenacity retry logic to handle CPU contention
        during boot. Uses exponential backoff with full jitter to prevent
        thundering herd on retry.

        Args:
            language: Programming language (python or javascript)
            tenant_id: Tenant identifier for isolation
            task_id: Task identifier
            backing_image: Base image for overlay (default: language base image)
            memory_mb: Memory limit in MB (128-2048, default 512)
            allow_network: Enable network access (default: False, isolated)
            allowed_domains: Whitelist of allowed domains if allow_network=True
            direct_write_target: If set, write directly to this file (no overlay).
                Used for snapshot creation. Mutually exclusive with backing_image.

        Returns:
            QemuVM handle for code execution

        Raises:
            VmTransientError: VM creation failed (retried, then re-raised)
            VmPermanentError: VM creation failed (not retryable)
            asyncio.TimeoutError: VM boot timeout after all retries
        """
        from exec_sandbox.exceptions import VmTransientError  # noqa: PLC0415

        # Acquire semaphore manually - hold until VM is destroyed (lifecycle-bound)
        await self._semaphore.acquire()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(constants.VM_BOOT_MAX_RETRIES),
                wait=wait_random_exponential(
                    min=constants.VM_BOOT_RETRY_MIN_SECONDS,
                    max=constants.VM_BOOT_RETRY_MAX_SECONDS,
                ),
                # Only retry transient errors - permanent errors (config, capacity, dependency) should fail immediately
                retry=retry_if_exception_type((VmTransientError, TimeoutError)),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    vm = await self._create_vm_impl(
                        language=language,
                        tenant_id=tenant_id,
                        task_id=task_id,
                        backing_image=backing_image,
                        memory_mb=memory_mb,
                        allow_network=allow_network,
                        allowed_domains=allowed_domains,
                        direct_write_target=direct_write_target,
                    )
                    # Track retry count (attempt.retry_state.attempt_number is 1-indexed)
                    vm.timing.boot_retries = attempt.retry_state.attempt_number - 1
                    # Mark VM as holding semaphore slot (released in destroy_vm)
                    vm.holds_semaphore_slot = True
                    return vm

            # Unreachable: AsyncRetrying either returns or raises
            raise AssertionError("Unreachable: AsyncRetrying exhausted without exception")
        except BaseException:
            # Release semaphore on failure - VM was not created successfully
            self._semaphore.release()
            raise

    async def _create_vm_impl(  # noqa: PLR0912, PLR0915
        self,
        language: Language,
        tenant_id: str,
        task_id: str,
        backing_image: Path | None = None,
        memory_mb: int = constants.DEFAULT_MEMORY_MB,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
        direct_write_target: Path | None = None,
    ) -> QemuVM:
        """Create and boot QEMU microVM (implementation).

        Workflow:
        1. Generate unique VM ID and CID
        2. Create ephemeral qcow2 overlay from backing image (or write directly)
        3. Set up cgroup v2 resource limits
        4. Build QEMU command (platform-specific)
        5. Launch QEMU subprocess
        6. Wait for guest agent ready

        Args:
            language: Programming language (python or javascript)
            tenant_id: Tenant identifier for isolation
            task_id: Task identifier
            backing_image: Base image for overlay (default: language base image)
            memory_mb: Memory limit in MB (128-2048, default 512)
            allow_network: Enable network access (default: False, isolated)
            allowed_domains: Whitelist of allowed domains if allow_network=True
            direct_write_target: If set, write directly to this file (no overlay).
                Used for snapshot creation. Mutually exclusive with backing_image.

        Returns:
            QemuVM handle for code execution

        Raises:
            VmConfigError: Invalid configuration (mutually exclusive args)
            VmDependencyError: Missing kernel, image, or qemu-vm user
            VmOverlayError: Overlay creation failed
            VmQemuCrashError: QEMU crashed during startup
            VmBootTimeoutError: Guest agent not ready in time
            VmGvproxyError: gvproxy startup failed
            asyncio.TimeoutError: VM boot timeout (>5s)
        """
        # Start timing
        start_time = asyncio.get_event_loop().time()

        # Security: Validate identifiers to prevent shell injection and path traversal
        # Must be done BEFORE any use of tenant_id/task_id in paths, commands, or IDs
        _validate_identifier(tenant_id, "tenant_id")
        _validate_identifier(task_id, "task_id")

        # Step 0: Validate kernel and initramfs exist (cached, one-time check)
        await validate_kernel_initramfs(self.settings.kernel_path, self.arch)
        arch_suffix = "aarch64" if self.arch == HostArch.AARCH64 else "x86_64"
        kernel_path = self.settings.kernel_path / f"vmlinuz-{arch_suffix}"
        initramfs_path = self.settings.kernel_path / f"initramfs-{arch_suffix}"

        # Step 1: Generate VM identifiers
        vm_id = f"{tenant_id}-{task_id}-{uuid4()}"

        # Validate mutual exclusivity
        if backing_image and direct_write_target:
            raise VmConfigError("backing_image and direct_write_target are mutually exclusive")

        # Step 1.5: Create working directory for all VM temp files
        # Uses tempfile.mkdtemp() for atomic, secure directory creation (mode 0700)
        # For direct_write_target mode, use it as the overlay path
        workdir = await VmWorkingDirectory.create(
            vm_id,
            custom_overlay_path=direct_write_target,
        )

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

        # Step 2: Determine virtualization mode early (needed for cgroup memory sizing)
        # TCG mode requires significantly more memory for translation block cache
        accel_type = await self._detect_accel_type()
        use_tcg = accel_type == AccelType.TCG

        # Step 3-4: Parallel resource setup (overlay + cgroup)
        # These operations are independent and can run concurrently
        # Note: gvproxy moved to boot phase to reduce contention under high concurrency
        # Resolve to absolute path - qemu-img resolves backing file relative to overlay location,
        # and VmWorkingDirectory places overlay in a temp dir, so relative paths would break
        base_image = (backing_image or self.get_base_image(language)).resolve()

        # Initialize ALL tracking variables before try block for finally cleanup
        cgroup_path: Path | None = None
        gvproxy_proc: ProcessWrapper | None = None
        gvproxy_log_task: asyncio.Task[None] | None = None
        qemu_proc: ProcessWrapper | None = None
        qemu_log_task: asyncio.Task[None] | None = None
        console_log: TextIO | None = None
        vm_created = False  # Flag to skip cleanup if VM successfully created

        try:
            # Log network configuration for debugging
            logger.debug(
                "Network configuration",
                extra={
                    "debug_category": "network",
                    "vm_id": vm_id,
                    "allow_network": allow_network,
                    "allowed_domains": allowed_domains,
                    "domains_count": len(allowed_domains) if allowed_domains else 0,
                },
            )

            # Unified setup phase: overlay + cgroup (gvproxy moved to boot phase)
            # This reduces setup contention under high concurrency
            overlay_ms = 0  # Default for direct_write mode (no overlay)
            if direct_write_target:
                # Direct write mode - VM writes directly to target file (no overlay)
                # Used for L2 snapshot creation where disk changes are written directly
                workdir.use_qemu_vm_user = False  # target file owned by current user
                cgroup_path = await cgroup.setup_cgroup(vm_id, tenant_id, memory_mb, use_tcg)
            # Normal mode - create overlay backed by base image
            # This allows the backing image to remain read-only and shareable
            # Pool handles fast path (from pool) or slow path (on-demand) internally
            else:
                overlay_start = asyncio.get_event_loop().time()
                try:
                    await self._overlay_pool.acquire(base_image, workdir.overlay_image)
                except (QemuImgError, QemuStorageDaemonError) as e:
                    raise VmOverlayError(str(e)) from e
                overlay_ms = round((asyncio.get_event_loop().time() - overlay_start) * 1000)
                # Apply permissions in parallel with cgroup setup
                perm_result, cgroup_result = await asyncio.gather(
                    self._apply_overlay_permissions(base_image, workdir.overlay_image),
                    cgroup.setup_cgroup(vm_id, tenant_id, memory_mb, use_tcg),
                    return_exceptions=True,
                )
                if isinstance(perm_result, BaseException):
                    raise perm_result
                if isinstance(cgroup_result, BaseException):
                    raise cgroup_result
                cgroup_path = cgroup_result
                workdir.use_qemu_vm_user = perm_result
            setup_complete_time = asyncio.get_event_loop().time()

            # Step 5: Build QEMU command (always Linux in container)
            qemu_cmd_start = asyncio.get_event_loop().time()
            qemu_cmd = await build_qemu_cmd(
                self.settings,
                self.arch,
                vm_id,
                workdir,
                memory_mb,
                allow_network,
            )

            # Step 6: Create dual-port Unix socket communication channel for guest agent
            # Socket paths are now in workdir (shorter, under 108-byte Unix socket limit)
            cmd_socket = workdir.cmd_socket
            event_socket = workdir.event_socket

            # Clean up any stale socket files before QEMU creates new ones
            # This ensures QEMU gets a clean state for chardev sockets
            for socket_path in [cmd_socket, event_socket, str(workdir.qmp_socket)]:
                with contextlib.suppress(OSError):
                    await aiofiles.os.remove(socket_path)

            # Determine expected UID for socket authentication (mandatory)
            # Verifies QEMU process identity before sending commands
            if workdir.use_qemu_vm_user:
                expected_uid = get_qemu_vm_uid()
                if expected_uid is None:
                    # qemu-vm user expected but doesn't exist - configuration error
                    raise VmDependencyError(
                        "qemu-vm user required for socket authentication but not found",
                        {"use_qemu_vm_user": True},
                    )
            else:
                expected_uid = os.getuid()

            channel: GuestChannel = DualPortChannel(cmd_socket, event_socket, expected_uid=expected_uid)

            # If cgroups unavailable, wrap with ulimit for host resource control
            # ulimit works on Linux, macOS, BSD (POSIX)
            if not cgroup.is_cgroup_available(cgroup_path):
                qemu_cmd = cgroup.wrap_with_ulimit(qemu_cmd, memory_mb)
            qemu_cmd_build_ms = round((asyncio.get_event_loop().time() - qemu_cmd_start) * 1000)

            # Boot phase: Start gvproxy BEFORE QEMU (if network enabled)
            gvproxy_start_time = asyncio.get_event_loop().time()
            gvproxy_start_ms = 0
            # gvproxy must create socket before QEMU connects to it
            # Moved from setup phase to boot phase to reduce contention under high concurrency
            if allow_network:
                logger.info(
                    "Starting gvproxy-wrapper in boot phase (before QEMU)",
                    extra={"vm_id": vm_id, "allowed_domains": allowed_domains},
                )
                gvproxy_proc, gvproxy_log_task = await start_gvproxy(vm_id, allowed_domains, language, workdir)
                # Attach gvproxy to cgroup for resource limits
                await cgroup.attach_if_available(cgroup_path, gvproxy_proc.pid)
                gvproxy_start_ms = round((asyncio.get_event_loop().time() - gvproxy_start_time) * 1000)

            # Step 7: Launch QEMU
            qemu_start_time = asyncio.get_event_loop().time()
            try:
                # Set umask 007 for qemu-vm user to create sockets with 0660 permissions
                # This is done via preexec_fn to avoid shell injection (no 'sh -c' needed)
                def _set_umask_007() -> None:
                    os.umask(0o007)

                qemu_proc = ProcessWrapper(
                    await asyncio.create_subprocess_exec(
                        *qemu_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True,  # Create new process group for proper cleanup
                        preexec_fn=_set_umask_007 if workdir.use_qemu_vm_user else None,
                    )
                )
                # Capture fork time immediately after subprocess creation
                qemu_fork_ms = round((asyncio.get_event_loop().time() - qemu_start_time) * 1000)

                # Attach process to cgroup (only if cgroups available)
                await cgroup.attach_if_available(cgroup_path, qemu_proc.pid)

                # Check if process crashed immediately BEFORE starting drain task
                # This ensures we can capture stderr/stdout on immediate crash
                await asyncio.sleep(0.02)
                if qemu_proc.returncode is not None:
                    stdout_text, stderr_text = await self._capture_qemu_output(qemu_proc)

                    # Build full command string for debugging
                    import shlex  # noqa: PLC0415

                    qemu_cmd_str = " ".join(shlex.quote(arg) for arg in qemu_cmd)

                    # Log detailed error for debugging
                    logger.error(
                        "QEMU crashed immediately",
                        extra={
                            "vm_id": vm_id,
                            "exit_code": qemu_proc.returncode,
                            "stderr": stderr_text[:2000] if stderr_text else "(empty)",
                            "stdout": stdout_text[:2000] if stdout_text else "(empty)",
                            "qemu_cmd": qemu_cmd_str[:2000],
                        },
                    )

                    max_bytes = constants.QEMU_OUTPUT_MAX_BYTES
                    raise VmQemuCrashError(
                        f"QEMU crashed immediately (exit code {qemu_proc.returncode}). "
                        f"stderr: {stderr_text[:max_bytes] if stderr_text else '(empty)'}, "
                        f"stdout: {stdout_text[:max_bytes] if stdout_text else '(empty)'}",
                        context={
                            "vm_id": vm_id,
                            "language": language,
                            "exit_code": qemu_proc.returncode,
                            "memory_mb": memory_mb,
                            "allow_network": allow_network,
                            "qemu_cmd": qemu_cmd_str,
                        },
                    )

                # Background task to drain QEMU output (prevent 64KB pipe deadlock)
                # Started AFTER crash check to ensure we can capture error output
                console_log_path = workdir.console_log

                # Open file asynchronously to avoid blocking the event loop
                # The file handle itself stays synchronous for simple write callbacks
                loop = asyncio.get_running_loop()
                console_log = await loop.run_in_executor(
                    None,
                    lambda: console_log_path.open("w", buffering=1),  # Line buffering
                )

                # Capture in local variable for type narrowing (console_log is definitely not None here)
                # Type assertion: we just opened the file above, so console_log is a valid TextIO
                assert console_log is not None  # noqa: S101 - type narrowing, not runtime check
                _console_log: TextIO = console_log

                def write_to_console(line: str) -> None:
                    """Write line to console log file and structured logs."""
                    try:
                        _console_log.write(f"[{vm_id}] {line}\n")
                    except OSError as e:
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
                qemu_log_task.add_done_callback(log_task_exception)

            except (OSError, FileNotFoundError) as e:
                raise VmDependencyError(
                    f"Failed to launch QEMU: {e}",
                    context={
                        "vm_id": vm_id,
                        "language": language,
                        "memory_mb": memory_mb,
                    },
                ) from e

            # Step 8: Wait for guest agent ready
            guest_wait_start = asyncio.get_event_loop().time()
            vm = QemuVM(
                vm_id,
                qemu_proc,
                cgroup_path,
                workdir,
                channel,
                language,
                gvproxy_proc,
                qemu_log_task,
                gvproxy_log_task,
                console_log,
            )

            # Register VM in registry (before BOOTING to ensure tracking)
            # Note: Capacity is enforced by semaphore in create_vm(), not here
            async with self._vms_lock:
                self._vms[vm.vm_id] = vm

            # Transition to BOOTING state
            await vm.transition_state(VmState.BOOTING)

            try:
                await self._wait_for_guest(vm, timeout=constants.VM_BOOT_TIMEOUT_SECONDS)
                boot_complete_time = asyncio.get_event_loop().time()
                guest_wait_ms = round((boot_complete_time - guest_wait_start) * 1000)
                # Store timing on VM for scheduler to use
                vm.setup_ms = round((setup_complete_time - start_time) * 1000)
                vm.boot_ms = round((boot_complete_time - setup_complete_time) * 1000)
                # Granular setup timing
                vm.overlay_ms = overlay_ms
                # Granular boot timing
                vm.qemu_cmd_build_ms = qemu_cmd_build_ms
                vm.gvproxy_start_ms = gvproxy_start_ms
                vm.qemu_fork_ms = qemu_fork_ms
                vm.guest_wait_ms = guest_wait_ms
                # Transition to READY state after boot completes
                await vm.transition_state(VmState.READY)
            except TimeoutError as e:
                # Capture QEMU output for debugging
                stdout_text, stderr_text = await self._capture_qemu_output(qemu_proc)

                # Flush console log before reading to ensure all buffered content is written
                if console_log:
                    with contextlib.suppress(OSError):
                        console_log.flush()

                # Read console log (last N bytes for debugging)
                console_output = await read_log_tail(str(console_log_path), constants.CONSOLE_LOG_MAX_BYTES)

                # Build command string for debugging
                import shlex  # noqa: PLC0415

                qemu_cmd_str = " ".join(shlex.quote(arg) for arg in qemu_cmd)

                # Log output if available
                logger.error(
                    "Guest agent boot timeout",
                    extra={
                        "vm_id": vm_id,
                        "stderr": stderr_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stderr_text else "(empty)",
                        "stdout": stdout_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stdout_text else "(empty)",
                        "console_log": console_output,
                        "qemu_running": qemu_proc.returncode is None,
                        "qemu_returncode": qemu_proc.returncode,
                        "qemu_cmd": qemu_cmd_str[:1000],
                        "overlay_image": str(workdir.overlay_image),
                        "kernel_path": str(kernel_path),
                        "initramfs_path": str(initramfs_path),
                    },
                )

                await vm.destroy()

                # Get QEMU binary from command - handle ulimit wrapper case
                # When wrapped with bash -c, qemu_cmd is ["bash", "-c", "ulimit ... && exec qemu-system-..."]
                qemu_binary = "(unknown)"
                if qemu_cmd:
                    if qemu_cmd[0] == "bash" and len(qemu_cmd) > 2:  # noqa: PLR2004
                        # Extract actual QEMU binary from shell command string
                        shell_cmd_str = qemu_cmd[2]
                        qemu_match = _QEMU_BINARY_PATTERN.search(shell_cmd_str)
                        qemu_binary = qemu_match.group(1) if qemu_match else f"bash -c '{shell_cmd_str[:100]}...'"
                    else:
                        qemu_binary = qemu_cmd[0]

                raise VmBootTimeoutError(
                    f"Guest agent not ready after {constants.VM_BOOT_TIMEOUT_SECONDS}s: {e}. "
                    f"qemu_binary={qemu_binary}, qemu_running={qemu_proc.returncode is None}, "
                    f"returncode={qemu_proc.returncode}, "
                    f"stderr: {stderr_text[:200] if stderr_text else '(empty)'}, "
                    f"console: {console_output[-constants.CONSOLE_LOG_PREVIEW_BYTES :] if console_output else '(empty)'}",
                    context={
                        "vm_id": vm_id,
                        "language": language,
                        "timeout_seconds": constants.VM_BOOT_TIMEOUT_SECONDS,
                        "console_log": console_output,
                        "qemu_running": qemu_proc.returncode is None,
                        "qemu_returncode": qemu_proc.returncode,
                        "qemu_cmd": qemu_cmd_str[:1000],
                        "kernel_path": str(kernel_path),
                        "initramfs_path": str(initramfs_path),
                        "overlay_image": str(workdir.overlay_image),
                    },
                ) from e

            # Log boot time with breakdown
            total_boot_ms = round((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.info(
                "VM created",
                extra={
                    "vm_id": vm_id,
                    "language": language,
                    "setup_ms": vm.setup_ms,
                    "boot_ms": vm.boot_ms,
                    "total_boot_ms": total_boot_ms,
                    # Granular setup breakdown
                    "overlay_ms": overlay_ms,
                    # Granular boot breakdown
                    "qemu_cmd_build_ms": qemu_cmd_build_ms,
                    "gvproxy_start_ms": gvproxy_start_ms,
                    "qemu_fork_ms": qemu_fork_ms,
                    "guest_wait_ms": guest_wait_ms,
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

                # Close console log file if opened (prevent resource leak)
                if console_log is not None:
                    with contextlib.suppress(OSError):
                        console_log.close()

                # Cancel log drain task if started
                if qemu_log_task is not None and not qemu_log_task.done():
                    qemu_log_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await qemu_log_task

                # Remove from registry if it was added (defensive - always try)
                async with self._vms_lock:
                    self._vms.pop(vm_id, None)

                await self._force_cleanup_all_resources(
                    vm_id=vm_id,
                    qemu_proc=qemu_proc,
                    gvproxy_proc=gvproxy_proc,
                    workdir=workdir,
                    cgroup_path=cgroup_path,
                )

    async def _force_cleanup_all_resources(
        self,
        vm_id: str,
        qemu_proc: ProcessWrapper | None = None,
        gvproxy_proc: ProcessWrapper | None = None,
        workdir: VmWorkingDirectory | None = None,
        cgroup_path: Path | None = None,
    ) -> dict[str, bool]:
        """Comprehensive cleanup of ALL VM resources in reverse dependency order.

        This is the MAIN cleanup method used in finally blocks.

        Best practices:
        - Cleans in reverse dependency order (processes -> workdir -> cgroup)
        - NEVER raises exceptions (logs errors instead)
        - Safe to call multiple times (idempotent)
        - Handles None/already-cleaned resources
        - Returns status dict for monitoring/debugging

        Cleanup order (reverse dependencies):
        1. QEMU process (depends on: workdir files, cgroup, networking)
        2. gvproxy process (QEMU networking dependency)
        3. Working directory (contains overlay, sockets, logs - single rmtree)
        4. Cgroup directory (QEMU process was in it)

        Args:
            vm_id: VM identifier for logging
            qemu_proc: QEMU subprocess (can be None)
            gvproxy_proc: gvproxy subprocess (can be None)
            workdir: VM working directory containing all temp files (can be None)
            cgroup_path: cgroup directory path (can be None)

        Returns:
            Dictionary with cleanup status for each resource
        """
        logger.info("Starting comprehensive resource cleanup", extra={"vm_id": vm_id})
        results: dict[str, bool] = {}
        was_cancelled = False

        # Phase 1: Kill processes in parallel (independent operations)
        # Shield cleanup from cancellation to ensure resources are fully released
        # NOTE: asyncio.shield() still raises CancelledError AFTER the shielded operation
        # completes if the outer task was cancelled. We must catch this to ensure Phase 2 runs.
        try:
            process_results = await asyncio.shield(
                asyncio.gather(
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
            )
            results["qemu"] = process_results[0] if isinstance(process_results[0], bool) else False
            results["gvproxy"] = process_results[1] if isinstance(process_results[1], bool) else False
        except asyncio.CancelledError:
            # Shield completed but outer task was cancelled - continue to Phase 2 anyway
            logger.debug(
                "Cleanup Phase 1 completed but task was cancelled, continuing to Phase 2", extra={"vm_id": vm_id}
            )
            results["qemu"] = False
            results["gvproxy"] = False
            was_cancelled = True

        # Phase 2: Cleanup workdir and cgroup in parallel (after processes dead)
        # workdir.cleanup() removes overlay, sockets, and console log in one operation
        async def cleanup_workdir() -> bool:
            if workdir is None:
                return True
            return await workdir.cleanup()

        # Shield file cleanup from cancellation to ensure resources are fully released
        try:
            file_results = await asyncio.shield(
                asyncio.gather(
                    cleanup_workdir(),
                    cgroup.cleanup_cgroup(
                        cgroup_path=cgroup_path,
                        context_id=vm_id,
                    ),
                    return_exceptions=True,
                )
            )
            results["workdir"] = file_results[0] if isinstance(file_results[0], bool) else False
            results["cgroup"] = file_results[1] if isinstance(file_results[1], bool) else False
        except asyncio.CancelledError:
            logger.debug("Cleanup Phase 2 completed but task was cancelled", extra={"vm_id": vm_id})
            results["workdir"] = False
            results["cgroup"] = False
            was_cancelled = True

        # Log summary
        success_count = sum(results.values())
        total_count = len(results)
        if success_count == total_count and not was_cancelled:
            logger.info("Cleanup completed successfully", extra={"vm_id": vm_id, "results": results})
        else:
            logger.warning(
                "Cleanup completed with errors" if not was_cancelled else "Cleanup completed (task was cancelled)",
                extra={
                    "vm_id": vm_id,
                    "results": results,
                    "success": success_count,
                    "total": total_count,
                    "was_cancelled": was_cancelled,
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
                with contextlib.suppress(OSError):
                    vm.console_log.close()

            # Cancel output reader tasks (prevent pipe deadlock during cleanup)
            if vm.qemu_log_task and not vm.qemu_log_task.done():
                vm.qemu_log_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await vm.qemu_log_task

            if vm.gvproxy_log_task and not vm.gvproxy_log_task.done():
                vm.gvproxy_log_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await vm.gvproxy_log_task

            # Destroy VM (transitions state, closes channel)
            await vm.destroy()

            # Comprehensive cleanup using defensive generic functions
            await self._force_cleanup_all_resources(
                vm_id=vm.vm_id,
                qemu_proc=vm.process,
                gvproxy_proc=vm.gvproxy_proc,
                workdir=vm.workdir,
                cgroup_path=vm.cgroup_path,
            )
        finally:
            # ALWAYS remove from registry, even on failure
            async with self._vms_lock:
                self._vms.pop(vm.vm_id, None)
            # Release semaphore slot only if this VM held one (prevents double-release)
            if vm.holds_semaphore_slot:
                vm.holds_semaphore_slot = False
                self._semaphore.release()

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

    def get_base_image(self, language: str) -> Path:
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
            VmConfigError: Unknown language
            VmDependencyError: Base image not found
        """
        # Pattern prefixes for each language
        patterns = {
            "python": "python-*-base-*.qcow2",
            "javascript": "node-*-base-*.qcow2",
            "raw": "raw-base-*.qcow2",
        }

        pattern = patterns.get(language)
        if not pattern:
            raise VmConfigError(f"Unknown language: {language}")

        # Find matching images
        matches = list(self.settings.base_images_dir.glob(pattern))
        if not matches:
            raise VmDependencyError(
                f"Base image not found for language: {language}. "
                f"Pattern: {pattern}, dir: {self.settings.base_images_dir}"
            )

        # Return first match (sorted for determinism)
        return sorted(matches)[0]

    async def _detect_accel_type(self) -> AccelType:
        """Detect which QEMU accelerator to use.

        This is the single source of truth for virtualization mode detection.
        Used for both cgroup memory sizing (TCG needs more) and QEMU command building.

        Returns:
            AccelType.KVM if Linux KVM available
            AccelType.HVF if macOS HVF available
            AccelType.TCG if software emulation needed (or force_emulation=True)
        """
        return await detect_accel_type(force_emulation=self.settings.force_emulation)

    async def _apply_overlay_permissions(self, base_image: Path, overlay_image: Path) -> bool:
        """Apply permissions to overlay (chown/chmod for qemu-vm isolation).

        Args:
            base_image: Base qcow2 image (needs read permission for qemu-vm)
            overlay_image: Overlay image (will be chowned to qemu-vm if possible)

        Returns:
            True if overlay was chowned to qemu-vm (QEMU should run as qemu-vm),
            False if overlay is owned by current user (QEMU should run as current user)
        """
        # Change ownership to qemu-vm user for process isolation (optional hardening)
        # Only if: Linux + qemu-vm user exists + can run sudo -u qemu-vm
        # The stronger probe_sudo_as_qemu_vm() ensures we can actually execute as qemu-vm
        # (probe_qemu_vm_user only checks if user exists, not sudo permissions)
        # Returns whether QEMU should run as qemu-vm user (based on chown success)
        if await probe_sudo_as_qemu_vm():
            # Make base image accessible to qemu-vm user
            # qemu-vm needs: read on file + execute on all parent directories
            # This is safe because the base image is read-only (writes go to overlay)

            # Make all parent directories traversable (a+x) up to /tmp or root
            current = base_image.parent
            dirs_to_chmod: list[Path] = []
            while current != current.parent and str(current) not in ("/", "/tmp"):  # noqa: S108
                dirs_to_chmod.append(current)
                current = current.parent

            await ensure_traversable(dirs_to_chmod)

            # Make base image readable
            if not await chmod_async(base_image, "a+r"):
                logger.debug("Could not chmod base image (qemu-vm may not have access)")

            if await chown_to_qemu_vm(overlay_image):
                # Make workdir accessible to qemu-vm for socket creation
                # mkdtemp creates with mode 0700, but qemu-vm needs access to create sockets
                workdir_path = overlay_image.parent
                if not await chmod_async(workdir_path, "a+rwx"):
                    logger.debug("Could not chmod workdir for qemu-vm access")
                return True  # Overlay chowned to qemu-vm, QEMU should run as qemu-vm
            logger.debug("Could not chown overlay to qemu-vm user (optional hardening)")
            return False  # Chown failed, QEMU should run as current user

        return False  # qemu-vm not available, QEMU should run as current user

    async def _wait_for_guest(self, vm: QemuVM, timeout: float) -> None:  # noqa: PLR0915
        """Wait for guest agent using event-driven racing.

        Races QEMU process death monitor against guest readiness checks with retry logic.

        Args:
            vm: QemuVM handle
            timeout: Maximum wait time in seconds

        Raises:
            VmQemuCrashError: QEMU process died during boot
            asyncio.TimeoutError: Guest not ready within timeout
        """

        async def monitor_process_death() -> None:
            """Monitor QEMU process death - kernel-notified, instant."""
            await vm.process.wait()

            # macOS HVF: Clean QEMU exit (code 0) is expected with -no-reboot
            # when VM shuts down normally after execution completes.
            host_os = detect_host_os()
            if host_os == HostOS.MACOS and vm.process.returncode == 0:
                logger.info(
                    "QEMU process exited cleanly (expected on macOS with -no-reboot)",
                    extra={"vm_id": vm.vm_id, "exit_code": 0},
                )
                return

            # TCG emulation: Exit code 0 during boot indicates timing race on
            # ARM64 GIC/virtio-MMIO initialization (translation cache pressure,
            # single-threaded TCG throughput limits). Log as warning for visibility,
            # then raise VmQemuCrashError to trigger outer retry with fresh VM.
            accel_type = await detect_accel_type()
            if accel_type == AccelType.TCG and vm.process.returncode == 0:
                logger.warning(
                    "QEMU TCG exited with code 0 during boot (timing race, will retry)",
                    extra={"vm_id": vm.vm_id, "exit_code": 0, "host_os": host_os.value},
                )
                raise VmQemuCrashError(
                    "QEMU TCG exited with code 0 during boot (timing race on virtio-mmio init)",
                    context={"vm_id": vm.vm_id, "exit_code": 0, "accel_type": "tcg"},
                )

            # Process died - capture output
            stdout_text, stderr_text = await self._capture_qemu_output(vm.process)
            signal_name = ""
            if vm.process.returncode and vm.process.returncode < 0:
                sig = -vm.process.returncode
                signal_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else f"signal {sig}"

            # Read console log (last N bytes for debugging)
            console_output = await read_log_tail(str(vm.workdir.console_log), constants.CONSOLE_LOG_MAX_BYTES)

            logger.error(
                "QEMU process exited unexpectedly",
                extra={
                    "vm_id": vm.vm_id,
                    "exit_code": vm.process.returncode,
                    "signal": signal_name,
                    "stdout": stdout_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stdout_text else "(empty)",
                    "stderr": stderr_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stderr_text else "(empty)",
                    "console_log": console_output,
                },
            )
            stderr_preview = stderr_text[:200] if stderr_text else "(empty)"
            console_preview = console_output[-constants.CONSOLE_LOG_PREVIEW_BYTES :] if console_output else "(empty)"
            raise VmQemuCrashError(
                f"QEMU process died (exit code {vm.process.returncode}, {signal_name}). "
                f"stderr: {stderr_preview}, console: {console_preview}"
            )

        async def check_guest_ready() -> None:
            """Single guest readiness check attempt."""
            await vm.channel.connect(timeout_seconds=5)
            response = await vm.channel.send_request(PingRequest(), timeout=5)

            # Ping returns PongMessage
            if not isinstance(response, PongMessage):
                raise RuntimeError(f"Guest ping returned unexpected type: {type(response)}")

            logger.info("Guest agent ready", extra={"vm_id": vm.vm_id, "version": response.version})

        import json  # noqa: PLC0415

        # Race with retry logic (tenacity exponential backoff with full jitter)
        death_task: asyncio.Task[None] | None = None
        guest_task: asyncio.Task[None] | None = None
        try:
            async with asyncio.timeout(timeout):
                death_task = asyncio.create_task(monitor_process_death())

                # Pre-connect to chardev sockets to trigger QEMU's poll registration.
                # Without this, QEMU may not add sockets to its poll set until after
                # guest opens virtio-serial ports, causing reads to return EOF.
                # See: https://bugs.launchpad.net/qemu/+bug/1224444 (virtio-mmio socket race)
                #
                # Timeout is short (1s vs previous 2s) because sockets are usually not ready this early.
                # The retry loop below handles actual connection with proper exponential backoff.
                try:
                    await vm.channel.connect(timeout_seconds=1)
                    logger.debug("Pre-connected to guest channel sockets", extra={"vm_id": vm.vm_id})
                except (TimeoutError, OSError) as e:
                    # Expected - sockets may not be ready yet, retry loop will handle
                    logger.debug("Pre-connect to sockets deferred", extra={"vm_id": vm.vm_id, "reason": str(e)})

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
                        done, _pending = await asyncio.wait(
                            {death_task, guest_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        # Check which completed
                        if death_task in done:
                            # QEMU died - cancel guest and retrieve exception
                            guest_task.cancel()
                            # Suppress ALL exceptions - we're about to re-raise VmError from death_task.
                            # Race condition: guest_task may also have completed with an exception
                            # (e.g., IncompleteReadError) which we must suppress to avoid masking VmError.
                            # Use BaseException to also catch CancelledError (not a subclass of Exception in Python 3.8+).
                            with contextlib.suppress(BaseException):
                                await guest_task
                            await death_task  # Re-raise VmError

                        # Guest task completed - check result (raises if failed, triggering retry)
                        await guest_task

        except TimeoutError:
            # Flush console log before reading to ensure all buffered content is written
            if vm.console_log:
                with contextlib.suppress(OSError):
                    vm.console_log.flush()

            console_output = await read_log_tail(str(vm.workdir.console_log), constants.CONSOLE_LOG_MAX_BYTES)

            logger.error(
                "Guest agent timeout",
                extra={
                    "vm_id": vm.vm_id,
                    "timeout": timeout,
                    "qemu_running": vm.process.returncode is None,
                    "console_output": console_output,
                    "overlay_image": str(vm.overlay_image) if vm.overlay_image else "(none)",
                },
            )

            raise TimeoutError(f"Guest agent not ready after {timeout}s") from None

        finally:
            # Always clean up tasks to prevent "Task exception was never retrieved" warnings.
            # This handles all exit paths: success, TimeoutError, VmError, and any other exception.
            # Use BaseException to catch CancelledError (which is not a subclass of Exception in Python 3.8+).
            for task in (death_task, guest_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (death_task, guest_task):
                if task is not None:
                    with contextlib.suppress(BaseException):
                        await task

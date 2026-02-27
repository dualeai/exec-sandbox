"""QEMU microVM lifecycle management.

Architecture:
- Supports Linux with KVM, macOS with HVF, or TCG (software emulation)
- qcow2 snapshot-based boot <400ms, L1 memory snapshot restore ~100ms
- Dual-port virtio-serial guest communication

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
import json
import logging
import os
import re
import shlex
import signal
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
from exec_sandbox.admission import ResourceAdmissionController
from exec_sandbox.exceptions import (
    VmBootTimeoutError,
    VmConfigError,
    VmDependencyError,
    VmOverlayError,
    VmQemuCrashError,
    VmTransientError,
)
from exec_sandbox.guest_agent_protocol import PingRequest, PongMessage
from exec_sandbox.guest_channel import DualPortChannel, GuestChannel
from exec_sandbox.gvproxy import start_gvproxy
from exec_sandbox.migration_client import MigrationClient
from exec_sandbox.models import ExposedPort, Language
from exec_sandbox.overlay_pool import OverlayPool, QemuImgError
from exec_sandbox.permission_utils import (
    chmod_async,
    get_qemu_vm_uid,
    grant_qemu_vm_file_access,
    probe_sudo_as_qemu_vm,
)
from exec_sandbox.platform_utils import HostArch, HostOS, ProcessWrapper, detect_host_arch, detect_host_os
from exec_sandbox.process_registry import register_process, unregister_process
from exec_sandbox.qemu_cmd import build_qemu_cmd
from exec_sandbox.qemu_storage_daemon import QemuStorageDaemonError
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.resource_cleanup import cleanup_process
from exec_sandbox.settings import Settings
from exec_sandbox.subprocess_utils import drain_subprocess_output, log_task_exception, wait_for_socket
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


@dataclass
class _VmInfra:
    """Pre-launch VM infrastructure assembled by VmManager._setup_vm_infra.

    Holds all shared state from the setup phase (validation → workdir → overlay →
    cgroup → channel). Both _create_vm_impl and restore_vm build this, then diverge
    for their specific launch/boot logic.
    """

    vm_id: str
    workdir: VmWorkingDirectory
    base_image: Path
    cgroup_path: Path | None
    use_tcg: bool
    expected_uid: int
    channel: GuestChannel


class VmManager:
    """QEMU microVM lifecycle manager with cross-platform support.

    Architecture:
    - Runtime detection: KVM or TCG acceleration
    - qcow2 snapshot-based boot with CoW overlays
    - Dual-port virtio-serial guest communication
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
        self._admission = ResourceAdmissionController(
            memory_overcommit_ratio=settings.memory_overcommit_ratio,
            cpu_overcommit_ratio=settings.cpu_overcommit_ratio,
            host_memory_reserve_ratio=settings.host_memory_reserve_ratio,
            host_memory_mb=settings.host_memory_mb,
            host_cpu_count=settings.host_cpu_count,
            available_memory_floor_mb=settings.available_memory_floor_mb,
        )

        # Overlay pool for fast VM boot (auto-manages base image discovery and pooling)
        # pool_size=0 initially; computed from admission budget in start()
        self._overlay_pool = OverlayPool(
            pool_size=0,
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

        # Probe host resources for admission control (psutil)
        await self._admission.start()

        # Compute overlay pool size from admission budget (after probing host resources)
        self._overlay_pool.pool_size = self._compute_overlay_pool_size()

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
                "overlay_pool_size": self._overlay_pool.pool_size,
                "accel_type": accel_type.value,
                "io_uring_available": io_uring_available,
                "unshare_available": unshare_available,
                "qemu_version": ".".join(map(str, qemu_version)) if qemu_version else None,
                "note": "All system probes pre-warmed (stampede prevention)",
            },
        )

    @property
    def admission(self) -> ResourceAdmissionController:
        """Access the resource admission controller (for ResourceMonitor)."""
        return self._admission

    def get_active_vms(self) -> dict[str, QemuVM]:
        """Get snapshot of active VMs (for debugging/metrics).

        Returns:
            Copy of VM registry (vm_id -> QemuVM)
        """
        return dict(self._vms)

    async def stop(self) -> None:
        """Stop VmManager and cleanup resources (admission probe, overlay pool).

        Should be called when the VmManager is no longer needed.
        """
        await self._admission.stop()
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
        memory_mb: int = constants.DEFAULT_MEMORY_MB,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
        direct_write_target: Path | None = None,
        expose_ports: list[ExposedPort] | None = None,
        on_boot_log: Callable[[str], None] | None = None,
        snapshot_drive: Path | None = None,
    ) -> QemuVM:
        """Create and boot QEMU microVM with automatic retry on transient failures.

        Wraps _create_vm_impl with tenacity retry logic to handle CPU contention
        during boot. Uses exponential backoff with full jitter to prevent
        thundering herd on retry.

        Args:
            language: Programming language (python or javascript)
            tenant_id: Tenant identifier for isolation
            task_id: Task identifier
            memory_mb: Memory limit in MB (minimum 128, default 256)
            allow_network: Enable network access (default: False, isolated)
            allowed_domains: Whitelist of allowed domains if allow_network=True
            direct_write_target: If set, path to ext4 qcow2 for snapshot creation.
                Attached as second drive (serial=snap, writable) for overlayfs
                upper layer on EROFS base.
            expose_ports: List of ports to expose from guest to host.
                Mode 1: Works without allow_network (QEMU hostfwd, no internet).
                Mode 2: Works with allow_network (gvproxy API, with internet).
            on_boot_log: Optional callback for streaming boot console output.
                When provided, enables verbose kernel/init logging and calls
                the callback with each line of boot output.
            snapshot_drive: Path to ext4 qcow2 snapshot (serial=snap, read-only).
                When set, tiny-init discovers drives by serial number and mounts
                EROFS base + ext4 snapshot via overlayfs.

        Returns:
            QemuVM handle for code execution

        Raises:
            VmTransientError: VM creation failed (retried, then re-raised)
            VmPermanentError: VM creation failed (not retryable)
            asyncio.TimeoutError: VM boot timeout after all retries
        """
        # Detect acceleration type to calculate accurate memory overhead
        accel_type = await self._detect_accel_type()
        use_tcg = accel_type == AccelType.TCG

        # Acquire resource reservation - blocks if insufficient resources
        reservation = await self._admission.acquire(
            vm_id=f"{tenant_id}-{task_id}",
            memory_mb=memory_mb,
            cpu_cores=constants.DEFAULT_VM_CPU_CORES,
            timeout=constants.RESOURCE_ADMISSION_TIMEOUT_SECONDS,
            use_tcg=use_tcg,
        )
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
                        memory_mb=memory_mb,
                        allow_network=allow_network,
                        allowed_domains=allowed_domains,
                        direct_write_target=direct_write_target,
                        expose_ports=expose_ports,
                        on_boot_log=on_boot_log,
                        snapshot_drive=snapshot_drive,
                    )
                    # Track retry count (attempt.retry_state.attempt_number is 1-indexed)
                    vm.timing.boot_retries = attempt.retry_state.attempt_number - 1
                    # Mark VM as holding semaphore slot (released in destroy_vm)
                    vm.holds_semaphore_slot = True
                    # Store resource reservation on VM for release in destroy_vm
                    vm.resource_reservation = reservation
                    return vm

            # Unreachable: AsyncRetrying either returns or raises
            raise AssertionError("Unreachable: AsyncRetrying exhausted without exception")
        except BaseException:
            # Release reservation on failure - VM was not created successfully
            await self._admission.release(reservation)
            raise

    async def _setup_vm_infra(
        self,
        language: Language,
        tenant_id: str,
        task_id: str,
        memory_mb: int,
    ) -> _VmInfra:
        """Common VM infrastructure setup shared by _create_vm_impl and restore_vm.

        1. Validate identifiers and kernel/initramfs
        2. Generate VM ID and working directory
        3. Detect acceleration type
        4. Create overlay from base image
        5. Set up cgroup and permissions (parallel)
        6. Prepare guest communication channel

        Returns:
            _VmInfra with all pre-launch state. Caller must handle cleanup
            via _cleanup_failed_launch on failure.

        Raises:
            VmDependencyError: Missing kernel, images, or qemu-vm user
            VmOverlayError: Overlay creation failed
        """
        _validate_identifier(tenant_id, "tenant_id")
        _validate_identifier(task_id, "task_id")
        await validate_kernel_initramfs(self.settings.kernel_path, self.arch)

        vm_id = f"{tenant_id}-{task_id}-{uuid4()}"
        workdir = await VmWorkingDirectory.create(vm_id)
        base_image = self.get_base_image(language).resolve()
        accel_type = await self._detect_accel_type()
        use_tcg = accel_type == AccelType.TCG

        # Overlay + cgroup + permissions (parallel where possible)
        try:
            await self._overlay_pool.acquire(base_image, workdir.overlay_image)
        except (QemuImgError, QemuStorageDaemonError) as e:
            raise VmOverlayError(str(e)) from e

        perm_result, cgroup_result = await asyncio.gather(
            self._apply_overlay_permissions(base_image, workdir.overlay_image),
            cgroup.setup_cgroup(vm_id, tenant_id, memory_mb, constants.DEFAULT_VM_CPU_CORES, use_tcg),
            return_exceptions=True,
        )
        if isinstance(perm_result, BaseException):
            raise perm_result
        if isinstance(cgroup_result, BaseException):
            raise cgroup_result
        cgroup_path: Path | None = cgroup_result
        workdir.use_qemu_vm_user = perm_result

        # Socket cleanup + channel
        for socket_path in [workdir.cmd_socket, workdir.event_socket, str(workdir.qmp_socket)]:
            with contextlib.suppress(OSError):
                await aiofiles.os.remove(socket_path)

        if workdir.use_qemu_vm_user:
            expected_uid = get_qemu_vm_uid()
            if expected_uid is None:
                raise VmDependencyError(
                    "qemu-vm user required for socket authentication but not found",
                    {"use_qemu_vm_user": True},
                )
        else:
            expected_uid = os.getuid()

        channel: GuestChannel = DualPortChannel(workdir.cmd_socket, workdir.event_socket, expected_uid=expected_uid)

        return _VmInfra(
            vm_id=vm_id,
            workdir=workdir,
            base_image=base_image,
            cgroup_path=cgroup_path,
            use_tcg=use_tcg,
            expected_uid=expected_uid,
            channel=channel,
        )

    async def _start_gvproxy_for_vm(
        self,
        infra: _VmInfra,
        language: Language,
        allow_network: bool,
        allowed_domains: list[str] | None,
        expose_ports: list[ExposedPort] | None,
    ) -> tuple[ProcessWrapper | None, asyncio.Task[None] | None, int]:
        """Start gvproxy-wrapper for a VM (shared by create and restore paths).

        Computes network mode, starts gvproxy, registers process, and attaches
        to cgroup.  Returns (None, None, 0) when gvproxy is not needed.

        Returns:
            (gvproxy_proc, gvproxy_log_task, gvproxy_start_ms)
        """
        needs_gvproxy = allow_network or bool(expose_ports)
        if not needs_gvproxy:
            return None, None, 0

        gvproxy_start_time = asyncio.get_running_loop().time()
        is_mode1 = bool(expose_ports) and not allow_network
        effective_allowed_domains = allowed_domains if allow_network else []

        logger.info(
            "Starting gvproxy-wrapper",
            extra={
                "vm_id": infra.vm_id,
                "allowed_domains": effective_allowed_domains,
                "mode": "Mode 1 (port-forward only)" if is_mode1 else "Mode 2/3 (internet)",
                "block_outbound": is_mode1,
            },
        )

        gvproxy_proc, gvproxy_log_task = await start_gvproxy(
            infra.vm_id,
            effective_allowed_domains,
            language,
            infra.workdir,
            expose_ports=expose_ports,
            block_outbound=is_mode1,
        )
        register_process(gvproxy_proc)
        await cgroup.attach_if_available(infra.cgroup_path, gvproxy_proc.pid)
        gvproxy_start_ms = round((asyncio.get_running_loop().time() - gvproxy_start_time) * 1000)
        return gvproxy_proc, gvproxy_log_task, gvproxy_start_ms

    async def _launch_qemu_vm(
        self,
        infra: _VmInfra,
        qemu_cmd: list[str],
        language: Language,
        memory_mb: int,
        on_boot_log: Callable[[str], None] | None = None,
        gvproxy_proc: ProcessWrapper | None = None,
        gvproxy_log_task: asyncio.Task[None] | None = None,
    ) -> QemuVM:
        """Launch QEMU subprocess, create VM object, register in tracking.

        Handles ulimit wrapping, subprocess fork, cgroup attach, output drain,
        QemuVM creation, and registry insertion.

        Returns VM in BOOTING state. On failure, caller must clean up via
        _cleanup_failed_launch.

        Raises:
            VmDependencyError: QEMU binary not found
        """
        if not cgroup.is_cgroup_available(infra.cgroup_path):
            qemu_cmd = cgroup.wrap_with_ulimit(qemu_cmd, memory_mb)

        try:

            def _set_umask_007() -> None:
                os.umask(0o007)

            qemu_proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    *qemu_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                    preexec_fn=_set_umask_007 if infra.workdir.use_qemu_vm_user else None,
                )
            )
            register_process(qemu_proc)
            await cgroup.attach_if_available(infra.cgroup_path, qemu_proc.pid)
        except (OSError, FileNotFoundError) as e:
            raise VmDependencyError(
                f"Failed to launch QEMU: {e}",
                context={"vm_id": infra.vm_id, "language": language},
            ) from e

        console_lines: deque[str] = deque(maxlen=constants.CONSOLE_RING_LINES)

        def write_to_console(line: str) -> None:
            nonlocal on_boot_log
            console_lines.append(line)
            if on_boot_log is not None:
                try:
                    on_boot_log(line)
                except Exception:  # noqa: BLE001 - user-provided callback, must not kill drain task
                    logger.warning(
                        "on_boot_log callback raised, disabling", extra={"vm_id": infra.vm_id}, exc_info=True
                    )
                    on_boot_log = None

        qemu_log_task = asyncio.create_task(
            drain_subprocess_output(
                qemu_proc,
                process_name="QEMU",
                context_id=infra.vm_id,
                stdout_handler=write_to_console,
                stderr_handler=write_to_console,
            )
        )
        qemu_log_task.add_done_callback(log_task_exception)

        vm = QemuVM(
            infra.vm_id,
            qemu_proc,
            infra.cgroup_path,
            infra.workdir,
            infra.channel,
            language,
            console_lines,
            gvproxy_proc,
            qemu_log_task,
            gvproxy_log_task,
        )

        async with self._vms_lock:
            self._vms[vm.vm_id] = vm

        await vm.transition_state(VmState.BOOTING)
        return vm

    async def _cleanup_failed_launch(
        self,
        vm_id: str,
        qemu_proc: ProcessWrapper | None = None,
        qemu_log_task: asyncio.Task[None] | None = None,
        gvproxy_proc: ProcessWrapper | None = None,
        workdir: VmWorkingDirectory | None = None,
        cgroup_path: Path | None = None,
    ) -> None:
        """Clean up after failed VM launch: cancel drains, unregister, cleanup resources."""
        if qemu_log_task is not None and not qemu_log_task.done():
            qemu_log_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await qemu_log_task

        async with self._vms_lock:
            self._vms.pop(vm_id, None)

        await self._force_cleanup_all_resources(
            vm_id=vm_id,
            qemu_proc=qemu_proc,
            gvproxy_proc=gvproxy_proc,
            workdir=workdir,
            cgroup_path=cgroup_path,
        )

    async def _create_vm_impl(  # noqa: PLR0915
        self,
        language: Language,
        tenant_id: str,
        task_id: str,
        memory_mb: int = constants.DEFAULT_MEMORY_MB,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
        direct_write_target: Path | None = None,
        expose_ports: list[ExposedPort] | None = None,
        on_boot_log: Callable[[str], None] | None = None,
        snapshot_drive: Path | None = None,
    ) -> QemuVM:
        """Create and boot QEMU microVM (implementation).

        Uses shared helpers _setup_vm_infra / _launch_qemu_vm for the common
        infrastructure, then handles create-specific logic: gvproxy, QEMU cmd
        building, guest agent wait, and detailed timing.

        Args:
            language: Programming language (python or javascript)
            tenant_id: Tenant identifier for isolation
            task_id: Task identifier
            memory_mb: Memory limit in MB (minimum 128, default 256)
            allow_network: Enable network access (default: False, isolated)
            allowed_domains: Whitelist of allowed domains if allow_network=True
            direct_write_target: If set, path to ext4 qcow2 for snapshot creation.
            expose_ports: List of ports to expose from guest to host.
            on_boot_log: Optional callback for streaming boot console output.
            snapshot_drive: Path to ext4 qcow2 snapshot (serial=snap, read-only).

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
        start_time = asyncio.get_running_loop().time()

        # Validate mutual exclusivity (create-specific)
        if snapshot_drive and direct_write_target:
            raise VmConfigError("snapshot_drive and direct_write_target are mutually exclusive")

        # Phase 1: Shared infrastructure (validation, workdir, overlay, cgroup, channel)
        infra = await self._setup_vm_infra(language, tenant_id, task_id, memory_mb)
        setup_complete_time = asyncio.get_running_loop().time()

        # Grant qemu-vm access to extra drives (snapshot creation target or read-only snapshot).
        # _setup_vm_infra only handles overlay/base images; extra drives need separate permissions.
        if infra.workdir.use_qemu_vm_user and direct_write_target:
            await grant_qemu_vm_file_access(direct_write_target, writable=True)
        elif infra.workdir.use_qemu_vm_user and snapshot_drive:
            await grant_qemu_vm_file_access(snapshot_drive, writable=False)

        # Phase 2: Build QEMU command (create-specific params)
        vdb_path = (
            str(direct_write_target) if direct_write_target else (str(snapshot_drive) if snapshot_drive else None)
        )
        qemu_cmd = await build_qemu_cmd(
            self.settings,
            self.arch,
            infra.vm_id,
            infra.workdir,
            memory_mb,
            constants.DEFAULT_VM_CPU_CORES,
            allow_network,
            expose_ports=expose_ports,
            direct_write=direct_write_target is not None,
            debug_boot=on_boot_log is not None,
            snapshot_drive=vdb_path,
        )

        # Phase 3: Start gvproxy BEFORE QEMU
        gvproxy_proc, gvproxy_log_task, gvproxy_start_ms = await self._start_gvproxy_for_vm(
            infra, language, allow_network, allowed_domains, expose_ports
        )

        # Phase 4: Launch QEMU (shared helper)
        vm_created = False
        try:
            qemu_start_time = asyncio.get_running_loop().time()
            vm = await self._launch_qemu_vm(
                infra,
                qemu_cmd,
                language,
                memory_mb,
                on_boot_log,
                gvproxy_proc,
                gvproxy_log_task,
            )
            qemu_fork_ms = round((asyncio.get_running_loop().time() - qemu_start_time) * 1000)

            # Phase 5: Wait for guest agent ready (create-specific)
            guest_wait_start = asyncio.get_running_loop().time()
            try:
                await self._wait_for_guest(vm, timeout=constants.VM_BOOT_TIMEOUT_SECONDS)
                boot_complete_time = asyncio.get_running_loop().time()
                guest_wait_ms = round((boot_complete_time - guest_wait_start) * 1000)

                # Timing breakdown
                vm.setup_ms = round((setup_complete_time - start_time) * 1000)
                vm.boot_ms = round((boot_complete_time - setup_complete_time) * 1000)
                vm.gvproxy_start_ms = gvproxy_start_ms
                vm.qemu_fork_ms = qemu_fork_ms
                vm.guest_wait_ms = guest_wait_ms

                if expose_ports:
                    vm.exposed_ports = expose_ports

                await vm.transition_state(VmState.READY)
            except TimeoutError as e:
                stdout_text, stderr_text = await self._capture_qemu_output(vm.process)
                console_snapshot = "\n".join(vm.console_lines) if vm.console_lines else "(empty)"
                qemu_cmd_str = " ".join(shlex.quote(arg) for arg in qemu_cmd)

                arch_suffix = "aarch64" if self.arch == HostArch.AARCH64 else "x86_64"
                kernel_path = self.settings.kernel_path / f"vmlinuz-{arch_suffix}"
                initramfs_path = self.settings.kernel_path / f"initramfs-{arch_suffix}"

                logger.error(
                    "Guest agent boot timeout",
                    extra={
                        "vm_id": infra.vm_id,
                        "stderr": stderr_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stderr_text else "(empty)",
                        "stdout": stdout_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stdout_text else "(empty)",
                        "console_log": console_snapshot,
                        "qemu_running": vm.process.returncode is None,
                        "qemu_returncode": vm.process.returncode,
                        "qemu_cmd": qemu_cmd_str[:1000],
                        "overlay_image": str(infra.workdir.overlay_image),
                        "kernel_path": str(kernel_path),
                        "initramfs_path": str(initramfs_path),
                    },
                )

                await vm.destroy()

                qemu_binary = "(unknown)"
                if qemu_cmd:
                    if qemu_cmd[0] == "bash" and len(qemu_cmd) > 2:  # noqa: PLR2004
                        shell_cmd_str = qemu_cmd[2]
                        qemu_match = _QEMU_BINARY_PATTERN.search(shell_cmd_str)
                        qemu_binary = qemu_match.group(1) if qemu_match else f"bash -c '{shell_cmd_str[:100]}...'"
                    else:
                        qemu_binary = qemu_cmd[0]

                raise VmBootTimeoutError(
                    f"Guest agent not ready after {constants.VM_BOOT_TIMEOUT_SECONDS}s: {e}. "
                    f"qemu_binary={qemu_binary}, qemu_running={vm.process.returncode is None}, "
                    f"returncode={vm.process.returncode}, "
                    f"stderr: {stderr_text[:200] if stderr_text else '(empty)'}, "
                    f"console: {console_snapshot[-4000:]}",
                    context={
                        "vm_id": infra.vm_id,
                        "language": language,
                        "timeout_seconds": constants.VM_BOOT_TIMEOUT_SECONDS,
                        "console_log": console_snapshot,
                        "qemu_running": vm.process.returncode is None,
                        "qemu_returncode": vm.process.returncode,
                        "qemu_cmd": qemu_cmd_str[:1000],
                        "kernel_path": str(kernel_path),
                        "initramfs_path": str(initramfs_path),
                        "overlay_image": str(infra.workdir.overlay_image),
                    },
                ) from e

            total_boot_ms = round((asyncio.get_running_loop().time() - start_time) * 1000)
            logger.info(
                "VM created",
                extra={
                    "vm_id": infra.vm_id,
                    "language": language,
                    "setup_ms": vm.setup_ms,
                    "boot_ms": vm.boot_ms,
                    "total_boot_ms": total_boot_ms,
                    "gvproxy_start_ms": gvproxy_start_ms,
                    "qemu_fork_ms": qemu_fork_ms,
                    "guest_wait_ms": guest_wait_ms,
                },
            )

            vm_created = True
            return vm

        finally:
            if not vm_created:
                await self._cleanup_failed_launch(
                    infra.vm_id,
                    gvproxy_proc=gvproxy_proc,
                    workdir=infra.workdir,
                    cgroup_path=infra.cgroup_path,
                )

    async def restore_vm(
        self,
        language: Language,
        tenant_id: str,
        task_id: str,
        vmstate_path: Path,
        memory_mb: int = constants.DEFAULT_MEMORY_MB,
        snapshot_drive: Path | None = None,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
        expose_ports: list[ExposedPort] | None = None,
    ) -> QemuVM:
        """Restore VM from L1 memory snapshot (full CPU+RAM+device state).

        Uses shared helpers _setup_vm_infra / _launch_qemu_vm, then restores
        state via MigrationClient instead of waiting for guest boot.

        Args:
            language: Programming language
            tenant_id: Tenant identifier
            task_id: Task identifier
            vmstate_path: Path to vmstate file from save_snapshot()
            memory_mb: VM memory in MB (must match save-time value)
            snapshot_drive: Optional L2 qcow2 for disk state
            allow_network: Network topology must match save-time config.
                L1 snapshots with network (allow_network or expose_ports) have
                virtio-net device in migration stream; restoring without matching
                topology causes migration failure.
            allowed_domains: Whitelist of allowed domains for DNS/TLS filtering.
                None for language defaults, empty list to block all outbound.
            expose_ports: List of ports to expose from guest to host.

        Returns:
            QemuVM in READY state with l1_restored=True

        Raises:
            VmTransientError: Restore failed (retryable)
            VmPermanentError: Configuration error
        """
        start_time = asyncio.get_running_loop().time()

        # Early existence check: avoids wasting an admission slot + QEMU launch
        # if the vmstate was evicted between check_cache() and this call (TOCTOU).
        if not vmstate_path.exists():
            raise VmTransientError(
                f"vmstate file missing (evicted?): {vmstate_path}",
                context={"vmstate_path": str(vmstate_path)},
            )

        # Acquire resource reservation FIRST (matches create_vm ordering).
        # Acquiring overlay before admission would hold an overlay slot during
        # backpressure waits, starving other callers.
        reservation = await self._admission.acquire(
            vm_id=f"{tenant_id}-{task_id}",
            memory_mb=memory_mb,
            cpu_cores=constants.DEFAULT_VM_CPU_CORES,
            timeout=constants.RESOURCE_ADMISSION_TIMEOUT_SECONDS,
        )

        # Phase 1: Shared infrastructure (validation, workdir, overlay, cgroup, channel)
        try:
            infra = await self._setup_vm_infra(language, tenant_id, task_id, memory_mb)
        except BaseException:
            await self._admission.release(reservation)
            raise

        # Grant qemu-vm read access to snapshot drive if present
        if infra.workdir.use_qemu_vm_user and snapshot_drive:
            await grant_qemu_vm_file_access(snapshot_drive, writable=False)

        # Phase 2: Build QEMU command (restore-specific: defer_incoming, matching network topology)
        vdb_path = str(snapshot_drive) if snapshot_drive else None
        qemu_cmd = await build_qemu_cmd(
            self.settings,
            self.arch,
            infra.vm_id,
            infra.workdir,
            memory_mb,
            constants.DEFAULT_VM_CPU_CORES,
            allow_network=allow_network,
            expose_ports=expose_ports,
            snapshot_drive=vdb_path,
            defer_incoming=True,  # VM starts paused, waiting for migration stream
        )

        # Phase 3: Start gvproxy if network topology requires it
        gvproxy_proc, gvproxy_log_task, _ = await self._start_gvproxy_for_vm(
            infra, language, allow_network, allowed_domains, expose_ports
        )

        # Phase 4: Launch QEMU (shared helper) and restore snapshot
        vm_created = False
        try:
            vm = await self._launch_qemu_vm(
                infra,
                qemu_cmd,
                language,
                memory_mb,
                gvproxy_proc=gvproxy_proc,
                gvproxy_log_task=gvproxy_log_task,
            )

            # Wait for QMP socket to exist (QEMU needs a few ms after fork)
            qemu_version = await probe_qemu_version()
            try:
                await wait_for_socket(vm.qmp_socket, timeout=5.0)
            except TimeoutError as e:
                await vm.destroy()
                raise VmTransientError(
                    f"QMP socket never appeared for {infra.vm_id}",
                    context={"vm_id": infra.vm_id, "qmp_socket": str(vm.qmp_socket)},
                ) from e

            # Restore VM state via QMP migration (load vmstate → resume vCPU)
            try:
                async with MigrationClient(vm.qmp_socket, infra.expected_uid) as client:
                    await client.restore_snapshot(vmstate_path, qemu_version=qemu_version)
            except Exception as e:
                await vm.destroy()
                raise VmTransientError(
                    f"L1 restore failed for {infra.vm_id}: {e}",
                    context={"vm_id": infra.vm_id, "vmstate_path": str(vmstate_path)},
                ) from e

            # Connect guest channel with reconnection probe.
            # L1 restore needs probing because the virtio-serial chardev was connected
            # at save time — the restored VM needs the host to re-sync the connection.
            infra.channel._has_been_connected = True  # type: ignore[attr-defined]  # noqa: SLF001
            await infra.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)

            # VM is now running with REPL already warm
            vm.l1_restored = True
            vm.holds_semaphore_slot = True
            vm.resource_reservation = reservation

            restore_ms = round((asyncio.get_running_loop().time() - start_time) * 1000)
            vm.timing.setup_ms = restore_ms  # L1 restore replaces cold boot + REPL warm
            vm.timing.boot_ms = 0  # No kernel boot on restore

            if expose_ports:
                vm.exposed_ports = expose_ports

            await vm.transition_state(VmState.READY)

            logger.info(
                "VM restored from L1 snapshot",
                extra={
                    "vm_id": infra.vm_id,
                    "language": language,
                    "restore_ms": restore_ms,
                    "vmstate_path": str(vmstate_path),
                    "allow_network": allow_network,
                    "expose_ports": [(p.internal, p.external) for p in expose_ports] if expose_ports else None,
                },
            )

            vm_created = True
            return vm

        except BaseException:
            if not vm_created:
                await self._admission.release(reservation)
                await self._cleanup_failed_launch(
                    infra.vm_id,
                    gvproxy_proc=gvproxy_proc,
                    workdir=infra.workdir,
                    cgroup_path=infra.cgroup_path,
                )
            raise

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
            # Unregister from emergency cleanup registry
            unregister_process(qemu_proc)
            unregister_process(gvproxy_proc)
        except asyncio.CancelledError:
            # Shield completed but outer task was cancelled - continue to Phase 2 anyway
            logger.debug(
                "Cleanup Phase 1 completed but task was cancelled, continuing to Phase 2", extra={"vm_id": vm_id}
            )
            results["qemu"] = False
            results["gvproxy"] = False
            was_cancelled = True
            # Still unregister even on cancellation
            unregister_process(qemu_proc)
            unregister_process(gvproxy_proc)

        # Phase 2: Cleanup workdir and cgroup in parallel (after processes dead)
        # workdir.cleanup() removes overlay and sockets in one operation
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
            # Release resource reservation only if this VM held one (prevents double-release)
            if vm.holds_semaphore_slot:
                vm.holds_semaphore_slot = False
                if vm.resource_reservation is not None:
                    await self._admission.release(vm.resource_reservation)
                    vm.resource_reservation = None

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

    def _compute_overlay_pool_size(self) -> int:
        """Compute overlay pool size from admission controller budget.

        Derives the effective max VMs from host resources (memory + CPU),
        then applies OVERLAY_POOL_SIZE_RATIO to determine pool size.

        Returns:
            Pool size per base image
        """
        effective_max = self._admission.effective_max_vms(
            guest_memory_mb=constants.DEFAULT_MEMORY_MB,
            cpu_per_vm=constants.DEFAULT_VM_CPU_CORES,
        )
        if effective_max < 0:
            logger.warning(
                "Cannot compute overlay pool size from host resources (psutil probe failed), using fallback",
                extra={"fallback_pool_size": constants.OVERLAY_POOL_FALLBACK_SIZE},
            )
            return constants.OVERLAY_POOL_FALLBACK_SIZE
        pool_size = max(0, int(effective_max * constants.OVERLAY_POOL_SIZE_RATIO))
        logger.info(
            "Overlay pool size computed from admission budget",
            extra={"effective_max_vms": effective_max, "pool_size": pool_size},
        )
        return pool_size

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
            # Make base image readable for qemu-vm (read-only, writes go to overlay)
            if not await grant_qemu_vm_file_access(base_image, writable=False):
                logger.debug("Could not grant qemu-vm read access to base image")

            if await grant_qemu_vm_file_access(overlay_image, writable=True):
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

            # macOS HVF: QEMU exits with code 0 when -no-reboot is set.
            # This closure is only used during boot (inside _wait_for_guest),
            # where any QEMU exit is an error — raise to trigger tenacity retry.
            host_os = detect_host_os()
            if host_os == HostOS.MACOS and vm.process.returncode == 0:
                logger.warning(
                    "QEMU exited with code 0 during boot on macOS (will retry)",
                    extra={"vm_id": vm.vm_id, "exit_code": 0, "host_os": "macos"},
                )
                raise VmQemuCrashError(
                    "QEMU process exited during boot (macOS clean exit)",
                    context={"vm_id": vm.vm_id, "exit_code": 0, "host_os": "macos"},
                )

            # TCG emulation: Exit code 0 during boot means -no-reboot caught a
            # guest reboot/panic. Capture diagnostics before retrying so CI
            # failures are debuggable.
            accel_type = await detect_accel_type()
            if accel_type == AccelType.TCG and vm.process.returncode == 0:
                console_snapshot = "\n".join(vm.console_lines) if vm.console_lines else "(empty)"
                stdout_text, stderr_text = await self._capture_qemu_output(vm.process)
                logger.warning(
                    "QEMU TCG exited with code 0 during boot (will retry)",
                    extra={
                        "vm_id": vm.vm_id,
                        "exit_code": 0,
                        "host_os": host_os.value,
                        "console_log": console_snapshot[-2000:],
                        "stderr": stderr_text[:500] if stderr_text else "(empty)",
                    },
                )
                raise VmQemuCrashError(
                    "QEMU TCG exited with code 0 during boot (guest reboot/panic)",
                    context={"vm_id": vm.vm_id, "exit_code": 0, "accel_type": "tcg"},
                )

            # Process died - capture output
            stdout_text, stderr_text = await self._capture_qemu_output(vm.process)
            signal_name = ""
            if vm.process.returncode and vm.process.returncode < 0:
                sig = -vm.process.returncode
                signal_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else f"signal {sig}"

            # Snapshot console lines from in-memory ring buffer
            console_snapshot = "\n".join(vm.console_lines) if vm.console_lines else "(empty)"

            logger.error(
                "QEMU process exited unexpectedly",
                extra={
                    "vm_id": vm.vm_id,
                    "exit_code": vm.process.returncode,
                    "signal": signal_name,
                    "stdout": stdout_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stdout_text else "(empty)",
                    "stderr": stderr_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stderr_text else "(empty)",
                    "console_log": console_snapshot,
                },
            )
            stderr_preview = stderr_text[:200] if stderr_text else "(empty)"
            raise VmQemuCrashError(
                f"QEMU process died (exit code {vm.process.returncode}, {signal_name}). "
                f"stderr: {stderr_preview}, console: {console_snapshot[-4000:]}"
            )

        async def check_guest_ready() -> None:
            """Single guest readiness check attempt."""
            await vm.channel.connect(timeout_seconds=1)
            response = await vm.channel.send_request(PingRequest())

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

                # Pre-connect to chardev sockets to trigger QEMU's poll registration.
                # Without this, QEMU may not add sockets to its poll set until after
                # guest opens virtio-serial ports, causing reads to return EOF.
                # See: https://bugs.launchpad.net/qemu/+bug/1224444 (virtio-mmio socket race)
                #
                # Timeout is short (1s vs previous 2s) because sockets are usually not ready this early.
                # The retry loop below handles actual connection with proper exponential backoff.
                # E3: Reduced pre-connect timeout from 0.1s to 0.01s — speculative, enters retry loop faster
                try:
                    await vm.channel.connect(timeout_seconds=0.005)
                    logger.debug("Pre-connected to guest channel sockets", extra={"vm_id": vm.vm_id})
                except (TimeoutError, OSError) as e:
                    # Expected - sockets may not be ready yet, retry loop will handle
                    logger.debug("Pre-connect to sockets deferred", extra={"vm_id": vm.vm_id, "reason": str(e)})

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
                            # Race condition: guest_task may also have completed with an exception
                            # (e.g., IncompleteReadError) which we must suppress to avoid masking VmError.
                            # Use BaseException to also catch CancelledError (not a subclass of Exception in Python 3.8+).
                            with contextlib.suppress(BaseException):
                                await guest_task
                            await death_task  # Re-raise VmError
                            # Safety net: monitor_process_death should always raise,
                            # but guard against future code paths that return normally.
                            raise VmQemuCrashError(
                                "QEMU process exited during boot (clean exit)",
                                context={"vm_id": vm.vm_id, "exit_code": vm.process.returncode},
                            )

                        # Guest task completed - check result (raises if failed, triggering retry)
                        await guest_task

        except TimeoutError:
            # Snapshot console lines from in-memory ring buffer
            console_snapshot = "\n".join(vm.console_lines) if vm.console_lines else "(empty)"

            logger.error(
                "Guest agent timeout",
                extra={
                    "vm_id": vm.vm_id,
                    "timeout": timeout,
                    "qemu_running": vm.process.returncode is None,
                    "console_log": console_snapshot,
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

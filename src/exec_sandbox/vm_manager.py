"""QEMU microVM lifecycle management.

Architecture:
- Supports Linux with KVM, macOS with HVF, or TCG (software emulation)
- qcow2 snapshot-based boot ~200 ms, L1 memory snapshot restore ~50-60 ms
- Dual-port virtio-serial guest communication

QEMU settings (qemu_cmd.build_qemu_cmd):
- CPU: x86 guests get one fixed x86-64-v3 (AVX2-level) model, so a saved instance
  loads on any such host; ARM guests get "host" (KVM, HVF) or "neoverse-v1,pauth=off"
  (TCG: qemu_cmd.guest_cpu_model)
- virtio-blk: num-queues=1, queue-size=64
- virtio-net: multiqueue off, TCP offload disabled (simpler for short VMs)
- Drive tuning: detect-zeroes=unmap, copy-on-read off, werror/rerror explicit
- Machine: mem-merge on (KSM page dedup, Linux only; off on macOS HVF), dump-guest-core off
- io_uring AIO: Modern Linux async I/O (probed at startup, threads fallback)
- Drive flushes ignored (cache=unsafe, or O_DIRECT with no-flush where it helps: qemu_cmd)
- -no-reboot: a guest reboot ends QEMU
"""

import asyncio
import functools
import grp
import logging
import os
import pwd
import re
import shlex
import socket
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from exec_sandbox import cgroup, constants
from exec_sandbox._logging import get_logger
from exec_sandbox.admission import EvictableCache, ResourceAdmissionController, ResourceReservation
from exec_sandbox.aio_utils import await_cancellation_safe as _await_cancellation_safe
from exec_sandbox.aio_utils import run_to_end
from exec_sandbox.cgroup import Cgroup
from exec_sandbox.exceptions import (
    CommunicationError,
    ConfigError,
    ErrorCode,
    InternalError,
    InvalidInputError,
    TransientError,
    spawn_error,
)
from exec_sandbox.files.disk import allocated_bytes, clone_sparse, create_sparse
from exec_sandbox.files.locks import EntryLease
from exec_sandbox.guest_channel import DualPortChannel, GuestChannel, raise_send_buffer
from exec_sandbox.gvproxy import start_gvproxy
from exec_sandbox.images import GuestImages, guest_digests
from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager, SnapshotMetadata, remove_dead_live_ram_dirs
from exec_sandbox.migration_client import MigrationClient, migration_timeout
from exec_sandbox.models import ExposedPort
from exec_sandbox.overlay_pool import OverlayPool
from exec_sandbox.permission_utils import (
    chown_async,
    get_qemu_vm_uid,
    grant_qemu_vm_acl,
    grant_qemu_vm_dir_access,
    grant_qemu_vm_file_access,
    grant_qemu_vm_private_file_access,
    probe_qemu_as_qemu_vm,
)
from exec_sandbox.platform_utils import ProcessWrapper, detect_host_arch
from exec_sandbox.private_run_dir import PrivateRunDir, new_folder_name, remove_dead_folders
from exec_sandbox.process_registry import unregister_process
from exec_sandbox.qemu_cmd import build_qemu_cmd
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.resource_cleanup import cleanup_process
from exec_sandbox.settings import Settings
from exec_sandbox.socket_auth import create_unix_socket
from exec_sandbox.subprocess_utils import (
    exec_failed_error,
    start_drain,
    start_managed_process,
    wait_for_socket,
)
from exec_sandbox.system_probes import (
    check_tsc_deadline,
    detect_accel_type,
    probe_io_uring_support,
    probe_qemu_runnable,
    probe_qemu_sandbox_support,
    probe_qemu_version,
    probe_unshare_support,
    qemu_system_binary,
    qemu_unrunnable_error,
)
from exec_sandbox.validation import validate_kernel_initramfs
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType, VmState
from exec_sandbox.vm_working_directory import VmWorkingDirectory, drain_workdir_cleanup_tasks

logger = get_logger(__name__)
LOG_DRAIN_AFTER_PROCESS_EXIT_TIMEOUT_SECONDS = 1.0


async def _finish_process_log_drain(task: asyncio.Task[None] | None) -> None:
    """Let a dead process's pipe reader consume EOF before cancelling it."""
    if task is None:
        return
    if task.done():
        # The done callback owns exception reporting. Retrieving the result here
        # is unnecessary and could turn a logging failure into cleanup failure.
        return
    done, _pending = await asyncio.wait(
        {task},
        timeout=LOG_DRAIN_AFTER_PROCESS_EXIT_TIMEOUT_SECONDS,
    )
    if task not in done:
        # A confirmed-dead process should close both pipes promptly. Keep cleanup
        # bounded if an unexpected inherited descriptor prevents EOF.
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# Security: Identifier validation pattern
# Only alphanumeric, underscore, and hyphen allowed to prevent:
# - Shell command injection via malicious tenant_id/task_id
# - Path traversal attacks (no '..', '/')
# - Socket path manipulation
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_IDENTIFIER_MAX_LENGTH = 128  # Reasonable limit for identifiers


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
    if not _IDENTIFIER_PATTERN.fullmatch(value):  # match with `$` would let a final newline in
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
    cgroup: Cgroup | None  # None: this host gave the VM no cgroup (ulimits only)
    overlay_ms: int  # its overlay acquisition (pool hit or on-demand creation)
    accel_type: AccelType  # as built into the QEMU command (force_emulation applied)
    expected_uid: int
    channel: GuestChannel
    # The pre-launch owner published during setup; launch/boot logic reuses it
    # instead of re-publishing.
    pending: "_PendingLaunch"


@dataclass
class _PendingLaunch:
    """Immediate owner for pre-QemuVM processes and dependent resources."""

    vm_id: str
    # None once the caller took the reservation back: the launch's processes are
    # dead and only its files remain for the reaper (_cleanup_pending_launch).
    reservation: ResourceReservation | None
    workdir: VmWorkingDirectory
    # The VM's cgroup path from before its create (where cgroup v2 is mounted): a cancel or an
    # error of the create can leave a folder there, which the cleanup removes (a refused create
    # removes its own)
    cgroup: Cgroup | None
    qemu_proc: ProcessWrapper | None = None
    gvproxy_proc: ProcessWrapper | None = None
    qemu_log_task: asyncio.Task[None] | None = None
    gvproxy_log_task: asyncio.Task[None] | None = None
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # The launch's own shares of the cache leases (_launch_qemu_vm), from before the spawn
    # until QEMU's death is confirmed or the QemuVM takes them
    cache_leases: tuple[EntryLease, ...] = ()

    def release_cache_leases(self) -> None:
        """Release the launch's shares: call it only once QEMU's death is confirmed."""
        leases, self.cache_leases = self.cache_leases, ()
        for lease in leases:
            lease.release()


def _share_all(leases: tuple[EntryLease, ...]) -> tuple[EntryLease, ...]:
    """A share of each lease: all of them, or none (the shares taken go when one fails)."""
    shares: list[EntryLease] = []
    try:
        shares.extend(lease.share() for lease in leases)  # keeps the shares taken before a failure
    except BaseException:
        for share in shares:
            share.release()
        raise
    return tuple(shares)


def _record_vm(run_dir: PrivateRunDir, *, workdir: Path, vm_cgroup: Cgroup | None) -> None:
    """_setup_vm_infra's records (blocking): the working folder, and the VM cgroup where
    cgroup v2 is mounted (before the cgroup exists)."""
    run_dir.record_folder(workdir)
    if vm_cgroup is not None:
        run_dir.record_cgroup(vm_cgroup.path)


def _sudo_wraps_qemu(workdir: VmWorkingDirectory) -> bool:
    """Whether sudo starts QEMU (as qemu-vm, from another user): sudo >= 1.9.14 closes all
    fds >= 3 (closefrom), so QEMU inherits no socket and makes its own at the paths.
    See: https://www.sudo.ws/docs/man/sudo.man/"""
    qemu_vm_uid = get_qemu_vm_uid()
    return workdir.use_qemu_vm_user and (qemu_vm_uid is None or os.getuid() != qemu_vm_uid)


def _guest_port_listeners(workdir: VmWorkingDirectory) -> list[socket.socket]:
    """The cmd and event listening sockets, made here for QEMU to inherit, or none when sudo
    starts QEMU. QEMU sends the guest's output on the socket it accepts. On macOS that socket
    takes the listener's send buffer: 8 KiB by default, raised here to SOCKET_SEND_BUFFER_BYTES
    (a 30 MB read_file: 120 -> 55 ms). On Linux an accepted socket has its own default
    (about 208 KiB) and the raise changes nothing. The caller closes them once QEMU started
    (_close_listeners)."""
    listeners: list[socket.socket] = []
    if _sudo_wraps_qemu(workdir):
        return listeners
    try:
        for path in (workdir.cmd_socket, workdir.event_socket):
            listeners.append(create_unix_socket(path))
            raise_send_buffer(listeners[-1])
    except BaseException:
        _close_listeners(listeners)
        raise
    return listeners


def _listener_fds(listeners: list[socket.socket]) -> tuple[int, int] | None:
    """The cmd and event fds of _guest_port_listeners, or None when QEMU makes the sockets."""
    if not listeners:
        return None
    cmd_listener, event_listener = listeners
    return cmd_listener.fileno(), event_listener.fileno()


def _close_listeners(listeners: list[socket.socket]) -> None:
    """Close this process's copies of the listening sockets (idempotent)."""
    for listener in listeners:
        with suppress(OSError):
            listener.close()


async def _stop_qemu(proc: ProcessWrapper | None, *, vm_id: str, as_qemu_vm: bool) -> bool:
    """End a VM's QEMU: True once its exit is confirmed (cleanup_process). Every drive of
    the VM goes with its folder after this, so the clean exit of SIGTERM keeps nothing (an L2
    build and an L1 save end their QEMU before). A QEMU of this user gets SIGKILL first: it
    ends in about 3 ms, not 13 ms (macOS HVF). sudo does not pass SIGKILL on: a qemu-vm QEMU
    gets the SIGTERM of cleanup_process, then its cgroup kill."""
    return await cleanup_process(
        proc=proc, name="QEMU", context_id=vm_id, term_timeout=5.0, kill_timeout=2.0, kill_first=not as_qemu_vm
    )


def _host_dependency(message: str, *, context: dict[str, Any] | None) -> ConfigError:
    """The ConfigError (HOST_DEPENDENCY) of a user or ACL support this host lacks."""
    return ConfigError(message, code=ErrorCode.HOST_DEPENDENCY, context=context, hint=None, instance_id=None)


async def _grant_drive(path: Path, *, writable: bool, vm_id: str) -> None:
    """Give qemu-vm an extra drive of the VM (L2 layer: owner to write it, read to use it).

    Raises:
        ConfigError: HOST_DEPENDENCY: it could not (QEMU as qemu-vm would not open it)
    """
    if not await grant_qemu_vm_file_access(path, writable=writable):
        raise _host_dependency(
            f"Could not give qemu-vm {'write' if writable else 'read'} access to a VM drive",
            context={"vm_id": vm_id, "drive": str(path)},
        )


async def _take_back_drive(path: Path, *, vm_id: str) -> None:
    """Give this user back the drive that _grant_drive gave qemu-vm to write (an L2 build
    target), once its QEMU is dead. Owned by qemu-vm, it stays writable to every later QEMU
    as qemu-vm (a guest of any tenant that escaped into it), and every later VM boots with it.

    Raises:
        ConfigError: HOST_DEPENDENCY: `sudo -n chown` failed
    """
    me = pwd.getpwuid(os.geteuid())
    try:
        group = grp.getgrgid(me.pw_gid).gr_name
    except KeyError:  # a primary group with no name: chown takes the number
        group = str(me.pw_gid)
    qemu_vm_uid = get_qemu_vm_uid()
    if qemu_vm_uid is None or not await chown_async(path, me.pw_name, group, owner_uid=qemu_vm_uid):
        raise _host_dependency(
            "Could not take back from qemu-vm the drive it wrote (sudo -n chown)",
            context={"vm_id": vm_id, "drive": str(path)},
        )


class VmManager:
    """QEMU microVM lifecycle manager with cross-platform support.

    Architecture:
    - Runtime detection: KVM, HVF or TCG acceleration
    - qcow2 snapshot-based boot with CoW overlays
    - Dual-port virtio-serial guest communication
    - cgroup v2 resource limits

    Usage:
        async with VmManager(settings) as manager:
            vm = await manager.create_vm(
                VmSpec(language="python"), tenant_id="tenant-123", task_id="task-456", cache_leases=()
            )
            result = await vm.execute("print('hello')", timeout_seconds=30)
            await manager.destroy_vm(vm)
    """

    def __init__(self, settings: Settings):
        """Initialize QEMU manager (sync part only).

        Args:
            settings: Service configuration (paths, limits, etc.)

        Note: Call `await start()` after construction to run async system probes.

        Note on crash recovery:
            VM registry is in-memory only. Each VM cgroup is recorded in this manager's
            folder (work_dir) before any process goes in it. If this process ends
            without stop() (a crash, SIGKILL), a watcher process kills those cgroups
            at once: a QEMU that runs as qemu-vm would outlive it (its parent is sudo).
            The next VmManager to start removes the cgroups and the folder.

        No file work here (the constructor runs on the event loop): start() makes the folder.
        """
        self.settings = settings
        self.arch = detect_host_arch()
        self.images = GuestImages.from_settings(settings, arch=self.arch)  # the guest files VMs boot
        self._run_dir_path = settings.home_disk_dir / new_folder_name()
        self._run_dir: PrivateRunDir | None = None  # made by start (_open_run_dir)
        self._initialized = False

        self._vms: dict[str, QemuVM] = {}  # vm_id -> VM object
        self._pending_launches: dict[str, _PendingLaunch] = {}
        self._vms_lock = asyncio.Lock()  # Protect registry access
        self._failed_cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._active_launches = 0
        self._launches_drained = asyncio.Event()
        self._launches_drained.set()
        self._stopping = False
        self._stop_task: asyncio.Task[None] | None = None
        self._admission = ResourceAdmissionController(
            memory_overcommit_ratio=settings.memory_overcommit_ratio,
            cpu_overcommit_ratio=settings.cpu_overcommit_ratio,
            disk_overcommit_ratio=settings.disk_overcommit_ratio,
            host_memory_reserve_ratio=settings.host_memory_reserve_ratio,
            host_memory_mb=settings.host_memory_mb,
            host_cpu_count=settings.host_cpu_count,
            available_memory_floor_mb=settings.available_memory_floor_mb,
            host_cpu_reserve_cores=settings.host_cpu_reserve_cores,
            disk_dir=self._run_dir_path,
            disk_free_floor_mb=settings.disk_free_floor_mb,
        )

        # Overlay pool for fast VM boot
        # pool_size=0 initially; computed from admission budget in start()
        self._overlay_pool = OverlayPool(pool_size=0)

    def add_cache(self, cache: EvictableCache) -> None:
        """Let admission evict the oldest entries of `cache` when the home disks lack space."""
        self._admission.add_cache(cache)

    @property
    def work_dir(self) -> Path:
        """This manager's private folder on host disk (0700), made by start(): stop() removes
        it; after a crash, the next VmManager to start removes it. Holds the home disks, the
        saved-instance work files (instances/) and the records of the VMs."""
        return self._run_dir_path

    async def start(self) -> None:
        """Start VmManager: make its folder (work_dir), sweep the folders of dead
        processes, probe host resources, run the async system probes, check the
        kernel and initramfs, and start the overlay pool.

        The probes cache their results at module level. This prevents cache stampede
        when multiple VMs start concurrently - all probes are pre-warmed here instead
        of racing during VM creation.

        Must be called before creating VMs.
        """
        if self._initialized:
            return

        # This manager's folder, then the folders of dead processes (worker-thread jobs: the
        # root check reads the user and group databases; whole disks, and the VMs they
        # record are killed and waited for), then the RAM folders of their L1 saves
        # (memory), before admission measures the free disk space and memory
        await run_to_end(self._open_run_dir)
        await asyncio.to_thread(self._remove_dead_folders)

        # Probe host resources for admission control (psutil)
        await self._admission.start()

        # Compute overlay pool size from admission budget (after probing host resources)
        self._overlay_pool.pool_size = self._compute_overlay_pool_size()

        # Run all async probes concurrently (they cache their results at module level)
        # This prevents cache stampede when multiple VMs start concurrently
        (
            accel_type,
            io_uring_available,
            unshare_available,
            qemu_version,
            _sandbox,
            qemu_as_qemu_vm,
        ) = await asyncio.gather(
            self._detect_accel_type(),  # Pre-warms HVF/KVM + QEMU accelerator caches
            probe_io_uring_support(),
            probe_unshare_support(),
            probe_qemu_version(),  # Pre-warm QEMU version for netdev reconnect
            probe_qemu_sandbox_support(),  # Pre-warm seccomp sandbox availability
            probe_qemu_as_qemu_vm(),  # The QEMU identity of every VM of this process (one warning)
        )
        # A QEMU binary this host cannot run: no VM could start (the version probe answered it)
        if (qemu_unrunnable := await probe_qemu_runnable()) is not None:
            raise qemu_unrunnable_error(qemu_unrunnable) from qemu_unrunnable[1]

        # Pre-warm TSC deadline (unified function handles arch/OS dispatch)
        await check_tsc_deadline()

        # Pre-flight check: validate kernel and initramfs exist (cached)
        await validate_kernel_initramfs(self.images)

        # Start overlay pool: overlays for this host's base images, filled in the background
        await self._overlay_pool.start(self.images.present_base_images())

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
                "qemu_as_qemu_vm": qemu_as_qemu_vm,
                "note": "All system probes pre-warmed (stampede prevention)",
            },
        )

    def _qemu_exec_failed(self, vm: QemuVM, *, qemu_cmd: list[str], context: dict[str, Any]) -> ConfigError | None:
        """The ConfigError (HOST_DEPENDENCY) of a QEMU launch that ended with 126 or 127 (an
        exec in it failed: QEMU, or a program that starts it), else None. The shell of the
        launch runs the exec, so Popen cannot report it; the console names the program."""
        return exec_failed_error(
            vm.process.returncode,
            binary=qemu_system_binary(self.settings, arch=self.arch),
            detail=None,  # the console is in the context
            context={**context, "qemu_cmd": " ".join(shlex.quote(arg) for arg in qemu_cmd)[:1000]},
            hint=None,
        )

    def _open_run_dir(self) -> None:
        """Make this manager's folder (work_dir) unless it is made. Blocking (start runs it in
        a worker thread)."""
        if self._run_dir is None:
            self._run_dir = PrivateRunDir.open(self.settings.home_disk_dir, name=self._run_dir_path.name)

    def _remove_dead_folders(self) -> None:
        """start's sweep (blocking): the folders of dead processes under home_disk_dir (and
        the VMs and working folders they record), then the RAM folders of their L1 saves."""
        remove_dead_folders(self.settings.home_disk_dir)
        remove_dead_live_ram_dirs()

    def get_active_vms(self) -> dict[str, QemuVM]:
        """Get snapshot of active VMs (for debugging/metrics).

        Returns:
            Copy of VM registry (vm_id -> QemuVM)
        """
        return dict(self._vms)

    async def stop(self) -> None:
        """Stop VmManager and cleanup resources (tracked VMs, admission, overlay pool).

        Destroys all tracked VMs (including checked-out ones that were never
        returned to a pool) before tearing down overlay services. Concurrent
        callers share one manager-owned stop task. Cancelling one waiter does
        not cancel process cleanup; normal return proves all tracked QEMU and
        gvproxy processes have confirmed death.
        """
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop_impl())
        await asyncio.shield(self._stop_task)

    async def _stop_impl(self) -> None:
        """Close launch ingress, drain owners, then tear down shared services."""
        # Synchronous flag flip: single event loop, no await between check
        # sites, so no lock is needed (see _launch_operation).
        self._stopping = True

        # Reject and wake admission waiters before taking a closed snapshot.
        # Existing reservations remain releasable by their current owners.
        await self._admission.stop()

        # A launch that linearized before STOPPING either publishes a tracked VM
        # or cleans up its partial resources before decrementing this counter.
        await self._launches_drained.wait()

        # Snapshot the registry under lock, then destroy outside lock
        # (destroy_vm acquires _vms_lock internally to remove each entry).
        async with self._vms_lock:
            vms_to_destroy = list(self._vms.values())
        pending_to_destroy = list(self._pending_launches.values())
        await asyncio.gather(
            *(self.destroy_vm(vm) for vm in vms_to_destroy),
            *(self._cleanup_pending_launch(pending) for pending in pending_to_destroy),
            return_exceptions=True,
        )
        # A successful stop is a proof that no tracked VM process remains.
        # Shield independently owned reapers so cancellation of any stop waiter
        # cannot turn an unconfirmed process into an unowned orphan.
        while True:
            async with self._vms_lock:
                remaining_vms = list(self._vms.values())
            remaining_pending = list(self._pending_launches.values())
            for vm in remaining_vms:
                self._schedule_failed_cleanup_retry(vm)
            for pending in remaining_pending:
                self._schedule_failed_pending_cleanup_retry(pending)

            # Scheduling above either reused a live task or synchronously
            # created a fresh (never-done) one under the same key, so a nonempty
            # snapshot always yields nonempty retry_tasks — no empty-tasks arm.
            retry_tasks = [task for task in self._failed_cleanup_tasks.values() if not task.done()]
            if not retry_tasks:
                break
            await asyncio.shield(asyncio.gather(*retry_tasks, return_exceptions=True))
        await drain_workdir_cleanup_tasks()
        await self._overlay_pool.stop()
        if self._run_dir is not None:
            await asyncio.to_thread(self._run_dir.close)  # the home disks: up to GBs

    @asynccontextmanager
    async def _launch_operation(self) -> AsyncGenerator[None]:
        """Linearize launch/restore work against manager shutdown.

        The check-and-increment is synchronous (no await), so on a single
        event loop it cannot interleave with _stop_impl's flag flip — plain
        attributes suffice, no lock.
        """
        if self._stopping:
            raise InvalidInputError(
                "VmManager is stopping and cannot launch new VMs",
                code=ErrorCode.SCHEDULER_STATE,
                context=None,
                hint=None,
                instance_id=None,
            )
        if self._active_launches == 0:
            self._launches_drained.clear()
        self._active_launches += 1
        try:
            yield
        finally:
            self._active_launches -= 1
            if self._active_launches == 0:
                self._launches_drained.set()

    async def __aenter__(self) -> Self:
        """Enter async context manager, starting the manager. A start that fails is stopped
        (its run folder and what it started go), as no __aexit__ follows."""
        try:
            await self.start()
        except BaseException:
            await self.stop()
            raise
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, stopping the manager."""
        await self.stop()

    @asynccontextmanager
    async def reservation_context(
        self,
        spec: VmSpec,
        *,
        vm_id: str,
        held: frozenset[Path],
    ) -> AsyncGenerator[ResourceReservation]:
        """Acquire an admission reservation for a restore→create fallback sequence.

        Use when a caller may try restore_vm then fall back to create_vm,
        to avoid blocking on admission twice. The yielded reservation is
        passed to create_vm/restore_vm via their ``reservation`` parameter.
        `held`: the cache entries (lease paths) and folders the caller keeps while
        it waits for admission (ResourceAdmissionController.acquire).

        Contract:
        - On exception before process launch, reservation is released automatically.
          If launched-process death is unconfirmed, the VM/reaper retains it.
        - On normal exit: the caller MUST have created a VM that took
          ownership (``vm.resource_reservation = reservation``).
          destroy_vm() releases it later. Exiting normally without
          creating a VM is a usage error that leaks the reservation.

        Usage::

            async with vm_manager.reservation_context(spec, vm_id=vm_id, held=frozenset()) as reservation:
                try:
                    vm = await vm_manager.restore_vm(..., reservation=reservation)
                except Exception:
                    vm = None
                if vm is None:
                    vm = await vm_manager.create_vm(..., reservation=reservation)
        """
        async with self._launch_operation():
            accel_type = await self._detect_accel_type()
            use_tcg = accel_type == AccelType.TCG
            reservation = await self._admission.acquire(
                vm_id=vm_id,
                memory_mb=spec.memory_mb,
                cpu_cores=spec.cpu_cores,
                timeout=constants.RESOURCE_ADMISSION_TIMEOUT_SECONDS,
                use_tcg=use_tcg,
                disk_mb=spec.disk_mb,
                held=held,
            )
            try:
                yield reservation
            except BaseException:
                # The caller will never receive any process published under
                # this reservation. Destroy each such owner; unconfirmed death
                # remains registry/reaper-owned.
                await self._cleanup_reservation_owners(reservation, release_admission=True)
                if not await self._reservation_is_retained(reservation):
                    await _await_cancellation_safe(self._admission.release(reservation))
                raise

    async def _reservation_is_retained(self, reservation: ResourceReservation) -> bool:
        """Whether a VM or pending launch still owns a reservation."""
        async with self._vms_lock:
            retained_by_vm = any(
                vm.holds_admission_slot and vm.resource_reservation is reservation for vm in self._vms.values()
            )
        return retained_by_vm or any(pending.reservation is reservation for pending in self._pending_launches.values())

    async def _cleanup_reservation_owners(
        self,
        reservation: ResourceReservation,
        *,
        release_admission: bool,
    ) -> None:
        """Destroy every resource owner published under one reservation."""
        async with self._vms_lock:
            retained_vms = [
                vm for vm in self._vms.values() if vm.holds_admission_slot and vm.resource_reservation is reservation
            ]
        retained_pending = [
            pending for pending in self._pending_launches.values() if pending.reservation is reservation
        ]
        await asyncio.gather(
            *(self.destroy_vm(vm, _release_admission=release_admission) for vm in retained_vms),
            *(
                self._cleanup_pending_launch(pending, _release_admission=release_admission)
                for pending in retained_pending
            ),
            return_exceptions=True,
        )

    @asynccontextmanager
    async def ephemeral_vm(
        self,
        spec: VmSpec,
        *,
        tenant_id: str,
        task_id: str,
        retry_profile: constants.RetryProfile,
        direct_write_target: Path | None,
        snapshot_drive: Path | None,
        cache_leases: tuple[EntryLease, ...],
        mem_path: str | None,
        mem_path_share: bool,
    ) -> AsyncGenerator[QemuVM]:
        """Create a VM, yield it, destroy it on exit.

        Encapsulates the common create→use→destroy lifecycle pattern.
        The VM is always destroyed on exit (success or failure). After a clean exit, a
        direct_write_target that QEMU wrote as qemu-vm belongs to this user again.

        Args:
            spec: Creation settings of the VM
            tenant_id: Tenant identifier
            task_id: Task identifier
            retry_profile: Retry parameters for boot
            direct_write_target: Path for snapshot creation, or None
            snapshot_drive: Path to ext4 qcow2 snapshot, or None
            cache_leases: Leases on the cache entries the VM runs on (create_vm)
            mem_path: File that backs guest RAM (L1 template save), or None
            mem_path_share: Whether QEMU writes guest RAM to mem_path (share=on)

        Yields:
            QemuVM ready for use
        """
        vm = await self.create_vm(
            spec,
            tenant_id=tenant_id,
            task_id=task_id,
            direct_write_target=direct_write_target,
            expose_ports=None,
            snapshot_drive=snapshot_drive,
            cache_leases=cache_leases,
            retry_profile=retry_profile,
            mem_path=mem_path,
            mem_path_share=mem_path_share,
        )
        primary_error: BaseException | None = None
        try:
            yield vm
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                cleanup_confirmed = await self.destroy_vm(vm)
            except BaseException:
                if primary_error is None:
                    raise
                logger.exception(
                    "Ephemeral VM cleanup failed while preserving primary error",
                    extra={"vm_id": vm.vm_id},
                )
            else:
                # Only process death decides: with QEMU dead and only files left, the
                # reaper removes them (and keeps the admission slot until then).
                if not cleanup_confirmed and primary_error is None and vm.state is not VmState.DESTROYED:
                    raise InternalError(
                        f"Ephemeral VM resource cleanup was not confirmed for {vm.vm_id}",
                        code=ErrorCode.CLEANUP_UNCONFIRMED,
                        context={"vm_id": vm.vm_id, "operation": "ephemeral_vm_cleanup"},
                        hint=None,
                        instance_id=None,
                    )
        # Here only after a clean exit, with QEMU dead. A failed build removes its target.
        if direct_write_target is not None and vm.use_qemu_vm_user:
            await _take_back_drive(direct_write_target, vm_id=vm.vm_id)

    async def create_vm(
        self,
        spec: VmSpec,
        *,
        tenant_id: str,
        task_id: str,
        direct_write_target: Path | None = None,
        expose_ports: list[ExposedPort] | None = None,
        on_boot_log: Callable[[str], None] | None = None,
        snapshot_drive: Path | None = None,
        cache_leases: tuple[EntryLease, ...],
        reservation: ResourceReservation | None = None,
        retry_profile: constants.RetryProfile = constants.RETRY_USER_FACING,
        mem_path: str | None = None,
        mem_path_share: bool = False,
    ) -> QemuVM:
        """Create and boot QEMU microVM with automatic retry on transient failures.

        Wraps _create_vm_impl with tenacity retry logic to handle CPU contention
        during boot. Uses exponential backoff with full jitter to prevent
        thundering herd on retry.

        Args:
            spec: Creation settings (language, memory, vCPUs, network policy)
            tenant_id: Tenant identifier for isolation
            task_id: Task identifier
            direct_write_target: If set, path to ext4 qcow2 for snapshot creation.
                Attached as second drive (serial=snap, writable) for overlayfs
                upper layer on EROFS base.
            expose_ports: List of ports to expose from guest to host (gvproxy
                forwards them).
                Mode 1: without allow_network (gvproxy -block-outbound, no internet).
                Mode 2: with allow_network (outbound as allowed_domains permits).
            on_boot_log: Optional callback for streaming boot console output.
                When provided, enables verbose kernel/init logging and calls
                the callback with each line of boot output.
            snapshot_drive: Path to ext4 qcow2 snapshot (serial=snap, read-only).
                When set, tiny-init discovers drives by serial number and mounts
                EROFS base + ext4 snapshot via overlayfs.
            cache_leases: Leases on the cache entries the VM runs on (an L2 snapshot
                drive), held by the caller from its cache lookup. Each launch takes its
                own share before QEMU opens them and keeps it until QEMU's death is
                confirmed: eviction skips those entries meanwhile. The caller keeps and
                releases its own leases.
            reservation: Pre-acquired admission reservation from reservation_context().
                When provided, skips admission acquire (caller owns the reservation).
                When None, acquires its own reservation.
            retry_profile: Retry parameters for boot (default: RETRY_USER_FACING).
            mem_path: File that backs guest RAM (L1 template save), or None
            mem_path_share: Whether QEMU writes guest RAM to mem_path (share=on)

        Returns:
            QemuVM handle for code execution

        Raises:
            TransientError: VM creation failed (retried, then re-raised; VM_BOOT_TIMEOUT
                when the guest never became ready)
            SandboxError: VM creation failed (not retryable)
        """
        async with self._launch_operation():
            if reservation is not None:
                try:
                    return await self._create_vm_with_reservation(
                        reservation,
                        spec,
                        tenant_id=tenant_id,
                        task_id=task_id,
                        direct_write_target=direct_write_target,
                        expose_ports=expose_ports,
                        on_boot_log=on_boot_log,
                        snapshot_drive=snapshot_drive,
                        cache_leases=cache_leases,
                        retry_profile=retry_profile,
                        mem_path=mem_path,
                        mem_path_share=mem_path_share,
                    )
                except BaseException:
                    await self._cleanup_reservation_owners(reservation, release_admission=False)
                    raise

            # reservation_context.__aexit__ owns owner cleanup and admission
            # release on failure — no second sweep here.
            held = frozenset(lease.path for lease in cache_leases)
            async with self.reservation_context(spec, vm_id=f"{tenant_id}-{task_id}", held=held) as res:
                return await self._create_vm_with_reservation(
                    res,
                    spec,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    direct_write_target=direct_write_target,
                    expose_ports=expose_ports,
                    on_boot_log=on_boot_log,
                    snapshot_drive=snapshot_drive,
                    cache_leases=cache_leases,
                    retry_profile=retry_profile,
                    mem_path=mem_path,
                    mem_path_share=mem_path_share,
                )

    async def _create_vm_with_reservation(
        self,
        reservation: ResourceReservation,
        spec: VmSpec,
        *,
        tenant_id: str,
        task_id: str,
        direct_write_target: Path | None,
        expose_ports: list[ExposedPort] | None,
        on_boot_log: Callable[[str], None] | None,
        snapshot_drive: Path | None,
        cache_leases: tuple[EntryLease, ...],
        retry_profile: constants.RetryProfile,
        mem_path: str | None,
        mem_path_share: bool,
    ) -> QemuVM:
        """Boot VM with retries. Reservation ownership transfers to the VM on success."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(retry_profile.max_attempts),
            wait=wait_random_exponential(
                min=retry_profile.retry_min_seconds,
                max=retry_profile.retry_max_seconds,
            ),
            # Only retry transient errors - other errors (config, internal) should fail immediately
            retry=retry_if_exception_type((TransientError, TimeoutError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                vm = await self._create_vm_impl(
                    spec,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    reservation=reservation,
                    direct_write_target=direct_write_target,
                    expose_ports=expose_ports,
                    on_boot_log=on_boot_log,
                    snapshot_drive=snapshot_drive,
                    cache_leases=cache_leases,
                    boot_timeout_seconds=retry_profile.boot_timeout_seconds,
                    mem_path=mem_path,
                    mem_path_share=mem_path_share,
                )
                # Track retry count (attempt.retry_state.attempt_number is 1-indexed).
                # Admission ownership was already claimed inside _launch_qemu_vm.
                vm.timing.boot_retries = attempt.retry_state.attempt_number - 1
                return vm

        # Unreachable: AsyncRetrying either returns or raises
        raise AssertionError("Unreachable: AsyncRetrying exhausted without exception")

    async def _setup_vm_infra(
        self,
        spec: VmSpec,
        tenant_id: str,
        task_id: str,
        reservation: ResourceReservation,
        *,
        base_image: Path,
        home_template: Path | None,
    ) -> _VmInfra:
        """Common VM infrastructure setup shared by _create_vm_impl and restore_vm.

        1. Validate identifiers and kernel/initramfs
        2. Generate VM ID and working directory; record both (and the cgroup path)
        3. Detect acceleration type
        4. Create the overlay from the base image and the cgroup (in parallel)
        5. Give qemu-vm its files, once the overlay exists (_apply_overlay_permissions)
        6. Create the home disk (blank, or a clone of home_template)
        7. Prepare guest communication channel

        Args:
            spec: Creation settings of the VM
            tenant_id: Tenant identifier
            task_id: Task identifier
            reservation: Pre-acquired admission reservation with effective limits
            base_image: Backing file of the overlay
            home_template: Home disk to clone (L1 restore or saved instance), or
                None for a blank disk of spec.disk_mb (cold boot: tiny-init formats it)

        Returns:
            _VmInfra with all pre-launch state. Once returned, the caller must
            publish a _PendingLaunch owner before another fallible await.

        Raises:
            ValueError: tenant_id or task_id is empty, too long or has other characters
                than [a-zA-Z0-9_-]
            InvalidInputError: SCHEDULER_STATE: start() was not called
            ConfigError: ASSET_NOT_FOUND: missing kernel or initramfs; HOST_DEPENDENCY:
                missing qemu-vm user, qemu-vm could not get access to the home disk, or
                QEMU runs as qemu-vm and the VM got no cgroup v2
            TransientError: OVERLAY_FAILED: overlay creation failed; LOCAL_DISK: the
                home disk, or the record of the working folder or cgroup, could not be made
        """
        _validate_identifier(tenant_id, "tenant_id")
        _validate_identifier(task_id, "task_id")
        await validate_kernel_initramfs(self.images)

        run_dir = self._run_dir
        if run_dir is None:
            raise InvalidInputError(
                "VmManager is not started: call start() before creating VMs",
                code=ErrorCode.SCHEDULER_STATE,
                context=None,
                hint=None,
                instance_id=None,
            )
        vm_id = f"{tenant_id}-{task_id}-{uuid4()}"
        workdir = await VmWorkingDirectory.create(vm_id, home_dir=run_dir.path)
        # Publish an owner before any overlay/cgroup/permission await. The real
        # cgroup path is deterministic, so even cancellation during setup can
        # remove a partially-created hierarchy.
        expected_cgroup = (
            Cgroup(Cgroup.path_of(tenant_id=tenant_id, vm_id=vm_id)) if cgroup.cgroup_v2_available() else None
        )
        pending = self._publish_pending_resources(vm_id, reservation, workdir, expected_cgroup)

        try:
            # Recorded before QEMU starts, the cgroup before it exists: if this process ends
            # without a cleanup (SIGKILL), its watcher and the next start kill the VM (a
            # qemu-vm QEMU outlives this process), and the next start removes the working
            # folder (PrivateRunDir). A worker thread: the first record starts the watcher.
            try:
                await run_to_end(
                    functools.partial(_record_vm, run_dir, workdir=workdir.path, vm_cgroup=expected_cgroup)
                )
            except OSError as e:
                raise TransientError(
                    f"Could not record the VM working folder or cgroup: {e.strerror or type(e).__name__}",
                    code=ErrorCode.LOCAL_DISK,
                    context={"vm_id": vm_id, "folder": str(run_dir.path)},
                    hint=None,
                    instance_id=None,
                ) from e
            base_image = base_image.resolve()
            accel_type = await self._detect_accel_type()

            # Overlay + cgroup in parallel (overlay doesn't depend on reservation;
            # cgroup needs reservation limits but both can run concurrently).
            # Permissions must wait for overlay to exist, so it runs after.
            overlay_ms = 0

            async def acquire_overlay() -> bool:  # timed alone: the cgroup create runs beside it
                nonlocal overlay_ms
                started = asyncio.get_running_loop().time()
                try:
                    return await self._overlay_pool.acquire(base_image, workdir.overlay_image)
                finally:
                    overlay_ms = round((asyncio.get_running_loop().time() - started) * 1000)

            overlay_result, cgroup_result = await asyncio.gather(
                acquire_overlay(),
                Cgroup.create(
                    tenant_id=tenant_id,
                    vm_id=vm_id,
                    memory_mb=int(reservation.memory_mb),
                    cpu_cores=reservation.cpu_cores,
                ),
                return_exceptions=True,
            )
            if isinstance(overlay_result, BaseException):
                raise overlay_result  # qemu-img or the daemon: already TransientError OVERLAY_FAILED
            if isinstance(cgroup_result, BaseException):
                raise cgroup_result
            vm_cgroup = cgroup_result
            if vm_cgroup is None and await probe_qemu_as_qemu_vm():
                # This user cannot signal a QEMU of qemu-vm. Only cgroup.kill ends one that
                # outlives sudo (sudo killed at the stop timeout), and only cgroup.events
                # proves its end (Cgroup.kill): without them its leases, reservation
                # and files would go while it may still run.
                raise _host_dependency(
                    "QEMU runs as qemu-vm, but this VM got no cgroup v2 (no delegated cgroup "
                    "subtree, or a read-only one): the end of a qemu-vm QEMU cannot be confirmed",
                    context={"vm_id": vm_id},
                )

            # Permissions need the overlay file to exist (changes ownership/mode)
            workdir.use_qemu_vm_user = await self._apply_overlay_permissions(base_image, workdir.overlay_image)

            await self._prepare_home_disk(workdir, spec, home_template)

            # Socket cleanup + channel
            for socket_path in [
                workdir.cmd_socket,
                workdir.event_socket,
                workdir.qmp_socket,
                workdir.qmp_events_socket,
            ]:
                with suppress(OSError):
                    Path(socket_path).unlink()

            if workdir.use_qemu_vm_user:
                expected_uid = get_qemu_vm_uid()
                if expected_uid is None:
                    raise _host_dependency(
                        "qemu-vm user required for socket authentication but not found",
                        context={"use_qemu_vm_user": True},
                    )
            else:
                expected_uid = os.getuid()

            channel: GuestChannel = DualPortChannel(
                workdir.cmd_socket,
                workdir.event_socket,
                expected_uid=expected_uid,
            )

            return _VmInfra(
                vm_id=vm_id,
                workdir=workdir,
                base_image=base_image,
                cgroup=vm_cgroup,
                overlay_ms=overlay_ms,
                accel_type=accel_type,
                expected_uid=expected_uid,
                channel=channel,
                pending=pending,
            )
        except BaseException as error:
            await self._cleanup_failed_owner(vm_id, operation="infrastructure_cleanup", error=error)
            raise

    async def _start_gvproxy_for_vm(
        self,
        infra: _VmInfra,
        pending: _PendingLaunch,
        spec: VmSpec,
        expose_ports: list[ExposedPort] | None,
    ) -> tuple[ProcessWrapper | None, asyncio.Task[None] | None, int, Callable[[], Awaitable[None]] | None]:
        """Start gvproxy-wrapper for a VM (shared by create and restore paths).

        Computes network mode, starts gvproxy, registers process, and attaches
        to cgroup.  Returns (None, None, 0, None) when gvproxy is not needed.

        Returns:
            (gvproxy_proc, gvproxy_log_task, gvproxy_start_ms, wait_ready): the caller
            awaits wait_ready() once QEMU runs, before the VM is ready (start_gvproxy)
        """
        needs_gvproxy = spec.allow_network or bool(expose_ports)
        if not needs_gvproxy:
            return None, None, 0, None

        gvproxy_start_time = asyncio.get_running_loop().time()
        is_mode1 = bool(expose_ports) and not spec.allow_network
        effective_allowed_domains = spec.allowed_domains if spec.allow_network else ()

        logger.info(
            "Starting gvproxy-wrapper",
            extra={
                "vm_id": infra.vm_id,
                "allowed_domains": effective_allowed_domains,
                "mode": "Mode 1 (port-forward only)" if is_mode1 else "Mode 2/3 (internet)",
                "block_outbound": is_mode1,
            },
        )

        def record_process(proc: ProcessWrapper, log_task: asyncio.Task[None] | None) -> None:
            pending.gvproxy_proc = proc
            if log_task is not None:
                pending.gvproxy_log_task = log_task

        # record_process publishes proc + log_task to `pending` before any
        # cancellation point inside start_gvproxy; no post-return re-publish.
        gvproxy_proc, gvproxy_log_task, wait_ready = await start_gvproxy(
            infra.vm_id,
            allowed_domains=effective_allowed_domains,
            language=spec.language,
            workdir=infra.workdir,
            images_dir=self.images.images_dir,
            expose_ports=expose_ports,
            block_outbound=is_mode1,
            on_process_started=record_process,
        )
        if infra.cgroup is not None and gvproxy_proc.pid is not None:
            await infra.cgroup.attach(gvproxy_proc.pid)
        gvproxy_start_ms = round((asyncio.get_running_loop().time() - gvproxy_start_time) * 1000)
        return gvproxy_proc, gvproxy_log_task, gvproxy_start_ms, wait_ready

    async def _launch_qemu_vm(
        self,
        infra: _VmInfra,
        pending: _PendingLaunch,
        *,
        qemu_cmd: list[str],
        spec: VmSpec,
        packages_image: Path | None,
        reservation: ResourceReservation,
        on_boot_log: Callable[[str], None] | None,
        gvproxy_proc: ProcessWrapper | None,
        gvproxy_log_task: asyncio.Task[None] | None,
        pass_fds: tuple[int, ...],
        cache_leases: tuple[EntryLease, ...],
    ) -> QemuVM:
        """Launch QEMU subprocess, create VM object, register in tracking.

        Handles ulimit wrapping, subprocess fork, cgroup attach, output drain,
        QemuVM creation, and registry insertion.

        Returns VM in BOOTING state. On failure, caller must clean up whichever
        owner is published in _pending_launches or _vms.

        Raises:
            ConfigError: HOST_DEPENDENCY: the QEMU binary is missing or cannot run, or a
                QEMU of qemu-vm could not join its VM cgroup (it did not start)
            TransientError: QEMU_CRASHED: the fork hit a process, memory or file-handle limit
        """
        if infra.cgroup is None:
            qemu_cmd = cgroup.wrap_with_ulimit(qemu_cmd, int(reservation.memory_mb))

        # The bytes QEMU opens next, for a saved instance (the leases keep these files)
        base_digest, packages_digest = await asyncio.to_thread(guest_digests, infra.base_image, packages_image)
        # Before QEMU opens them: the launch's own shares of the cache leases. The pending
        # owner, then the VM, releases them once QEMU's death is confirmed.
        pending.cache_leases = _share_all(cache_leases)
        as_qemu_vm = infra.workdir.use_qemu_vm_user

        def not_joined() -> ConfigError:
            return _host_dependency(
                "QEMU runs as qemu-vm, but its process could not join the VM cgroup (this process is "
                "outside the delegated cgroup subtree): the end of a qemu-vm QEMU could not be confirmed",
                context={"vm_id": infra.vm_id, "cgroup_path": None if infra.cgroup is None else str(infra.cgroup.path)},
            )

        if as_qemu_vm and infra.cgroup is None:
            raise not_joined()  # no real cgroup: nothing starts
        try:
            # The child joins the VM cgroup before it execs: sudo, QEMU and QEMU's
            # teardown process are born in it (limits, Cgroup.kill). As qemu-vm, the
            # socket umask comes from the command (QEMU_VM_SUDO).
            # A qemu-vm QEMU joins or stops, and only its cgroup ends it (sudo is setuid: the
            # death signal does not cross it). A QEMU of this user joins where it can and ends
            # with this process: the first program is unshare when namespaces work, which
            # outlives a killed host process otherwise, and so does QEMU (exit-with-parent ties
            # QEMU to unshare); without a real cgroup, nothing else ends it.
            if as_qemu_vm and infra.cgroup is not None:
                steps, ends = [infra.cgroup.join_or_stop_step()], False
            else:
                steps, ends = ([] if infra.cgroup is None else [infra.cgroup.join_step()]), True
            # The callback publishes qemu_proc to `pending` synchronously inside
            # the owned spawn task, before any cancellation point — no re-publish.
            qemu_proc = await start_managed_process(
                qemu_cmd,
                steps=steps,
                end_with_this_process=ends,
                pass_fds=pass_fds,
                on_process_started=lambda proc: setattr(pending, "qemu_proc", proc),
            )
        except OSError as e:
            raise spawn_error(
                e,
                binary=qemu_cmd[0],
                code=ErrorCode.QEMU_CRASHED,
                context={"vm_id": infra.vm_id, "language": spec.language},
                hint=None,
            ) from e
        # Its first line, or the end of file when the move failed and the shell stopped (no
        # sudo, no QEMU ran; `pending` owns the process)
        if as_qemu_vm and (qemu_proc.stdout is None or await qemu_proc.stdout.readline() != cgroup.JOINED):
            raise not_joined()

        console_lines: deque[str] = deque(maxlen=constants.CONSOLE_RING_LINES)

        def write_to_console(line: str) -> None:
            nonlocal on_boot_log
            console_lines.append(line)
            if on_boot_log is not None:
                try:
                    on_boot_log(line)
                except Exception:
                    logger.warning(
                        "on_boot_log callback raised, disabling", extra={"vm_id": infra.vm_id}, exc_info=True
                    )
                    on_boot_log = None

        qemu_log_task = start_drain(
            qemu_proc,
            process_name="QEMU",
            context_id=infra.vm_id,
            stdout_handler=write_to_console,
            stderr_handler=write_to_console,
        )
        pending.qemu_log_task = qemu_log_task

        vm = QemuVM(
            infra.vm_id,
            process=qemu_proc,
            cgroup=infra.cgroup,
            workdir=infra.workdir,
            channel=infra.channel,
            spec=spec,
            console_lines=console_lines,
            base_image=infra.base_image,
            packages_image=packages_image,
            accel_type=infra.accel_type,
            gvproxy_proc=gvproxy_proc,
            qemu_log_task=qemu_log_task,
            gvproxy_log_task=gvproxy_log_task,
            release_callback=self.destroy_vm,
        )

        # Process launch transfers admission ownership immediately, not only
        # after guest readiness. A boot/restore failure can still leave a live
        # QEMU process whose reservation must remain held until confirmed death.
        vm.holds_admission_slot = True
        vm.resource_reservation = reservation
        vm.cache_leases, pending.cache_leases = pending.cache_leases, ()
        vm.base_image_digest, vm.packages_image_digest = base_digest, packages_digest

        # The event loop cannot interleave these synchronous dictionary writes:
        # ownership transfers atomically from the pre-launch record to QemuVM
        # before any post-spawn fallible await.
        self._vms[vm.vm_id] = vm
        self._pending_launches.pop(vm.vm_id, None)
        vm.start_event_watch()  # from the spawn on: a full host disk stops the VM (werror=enospc)

        # A qemu-vm QEMU proved its move (JOINED); a QEMU of this user may have had its join
        # refused, and its attach retries the move
        if infra.cgroup is not None and not as_qemu_vm and qemu_proc.pid is not None:
            await infra.cgroup.attach(qemu_proc.pid)
        await vm.transition_state(VmState.BOOTING)
        return vm

    def _publish_pending_resources(
        self,
        vm_id: str,
        reservation: ResourceReservation,
        workdir: VmWorkingDirectory,
        vm_cgroup: Cgroup | None,
    ) -> _PendingLaunch:
        """Publish the single pre-QemuVM owner for one launch attempt.

        Called once per attempt from _setup_vm_infra, which mints a fresh
        uuid4-suffixed vm_id, so the key is always new — no refine path.
        """
        pending = _PendingLaunch(
            vm_id=vm_id,
            reservation=reservation,
            workdir=workdir,
            cgroup=vm_cgroup,
        )
        self._pending_launches[vm_id] = pending
        return pending

    async def _cleanup_failed_owner(self, vm_id: str, *, operation: str, error: BaseException) -> None:
        """Clean a failed attempt without consuming its caller-owned reservation.

        When the attempt's processes are dead but some files remain, the reaper owns
        the files and the caller gets the reservation back: the caller then raises
        `error` (a boot retry or an L1 cold-boot fallback can run).

        Raises:
            InternalError: CLEANUP_UNCONFIRMED (from `error`, the failure of the attempt)
                when process death is not confirmed; the owner keeps the reservation
                and the reaper retries.
        """
        async with self._vms_lock:
            vm = self._vms.get(vm_id)
        if vm is not None:
            confirmed = await self.destroy_vm(vm, _release_admission=False)
            retained = vm.holds_admission_slot
        elif (pending := self._pending_launches.get(vm_id)) is not None:
            confirmed = await self._cleanup_pending_launch(pending, _release_admission=False)
            retained = pending.reservation is not None
        else:
            confirmed, retained = True, False
        if not confirmed and retained:
            raise InternalError(
                f"VM resource cleanup was not confirmed for {vm_id} ({operation})",
                code=ErrorCode.CLEANUP_UNCONFIRMED,
                context={"vm_id": vm_id, "operation": operation},
                hint=None,
                instance_id=None,
            ) from error

    async def _cleanup_pending_launch(
        self,
        pending: _PendingLaunch,
        *,
        _schedule_retry: bool = True,
        _release_admission: bool = True,
    ) -> bool:
        """Confirm pre-VM process death before releasing launch ownership."""
        try:
            async with pending.cleanup_lock:
                cleanup_results = await self._force_cleanup_all_resources(
                    vm_id=pending.vm_id,
                    qemu_proc=pending.qemu_proc,
                    gvproxy_proc=pending.gvproxy_proc,
                    workdir=pending.workdir,
                    vm_cgroup=pending.cgroup,
                )
                processes_destroyed = all(cleanup_results.get(resource) is True for resource in ("qemu", "gvproxy"))
                if processes_destroyed:  # QEMU no longer reads the cache entries
                    pending.release_cache_leases()
                await asyncio.gather(
                    *(
                        _finish_process_log_drain(task)
                        for resource, task in (
                            ("qemu", pending.qemu_log_task),
                            ("gvproxy", pending.gvproxy_log_task),
                        )
                        if cleanup_results.get(resource) is True
                    )
                )
                owner_cleaned = processes_destroyed and all(
                    cleanup_results.get(resource) is True for resource in ("workdir", "cgroup")
                )
                if processes_destroyed and not owner_cleaned and not _release_admission:
                    # The caller keeps the reservation; the reaper only removes files.
                    pending.reservation = None

        except (asyncio.CancelledError, Exception):
            self._pending_launches[pending.vm_id] = pending
            if _schedule_retry:
                self._schedule_failed_pending_cleanup_retry(pending)
            raise

        if owner_cleaned:

            async def commit_cleanup() -> None:
                # Keep the owner published until reservation release completes.
                if _release_admission and pending.reservation is not None:
                    await self._admission.release(pending.reservation)
                if self._pending_launches.get(pending.vm_id) is pending:
                    self._pending_launches.pop(pending.vm_id, None)

            # A cancel comes out only once commit_cleanup completed (await_cancellation_safe)
            try:
                await _await_cancellation_safe(commit_cleanup())
            except Exception:
                self._pending_launches[pending.vm_id] = pending
                if _schedule_retry:
                    self._schedule_failed_pending_cleanup_retry(pending)
                raise
            return True

        self._pending_launches[pending.vm_id] = pending
        if _schedule_retry:
            self._schedule_failed_pending_cleanup_retry(pending)
        return False

    async def _create_vm_impl(  # noqa: PLR0915
        self,
        spec: VmSpec,
        *,
        tenant_id: str,
        task_id: str,
        reservation: ResourceReservation,
        direct_write_target: Path | None,
        expose_ports: list[ExposedPort] | None,
        on_boot_log: Callable[[str], None] | None,
        snapshot_drive: Path | None,
        cache_leases: tuple[EntryLease, ...],
        boot_timeout_seconds: int,
        mem_path: str | None,
        mem_path_share: bool,
    ) -> QemuVM:
        """Create and boot QEMU microVM (implementation).

        Uses shared helpers _setup_vm_infra / _launch_qemu_vm for the common
        infrastructure, then handles create-specific logic: gvproxy, QEMU cmd
        building, guest agent wait, and detailed timing.

        Args:
            spec: Creation settings (language, memory, vCPUs, network policy)
            tenant_id: Tenant identifier for isolation
            task_id: Task identifier
            reservation: Pre-acquired admission reservation with effective limits
            direct_write_target: If set, path to ext4 qcow2 for snapshot creation.
            expose_ports: List of ports to expose from guest to host.
            on_boot_log: Optional callback for streaming boot console output.
            snapshot_drive: Path to ext4 qcow2 snapshot (serial=snap, read-only).
            cache_leases: Leases on the cache entries the VM runs on (see create_vm)
            boot_timeout_seconds: Longest wait for the guest agent
            mem_path: File that backs guest RAM (L1 template save), or None
            mem_path_share: Whether QEMU writes guest RAM to mem_path (share=on)

        Returns:
            QemuVM handle for code execution

        Raises:
            InternalError: INTERNAL: invalid configuration (mutually exclusive args)
            ConfigError: ASSET_NOT_FOUND: missing kernel, initramfs or base image;
                HOST_DEPENDENCY: missing QEMU, gvproxy-wrapper or qemu-vm user, or QEMU runs
                as qemu-vm and the VM got no cgroup v2
            TransientError: OVERLAY_FAILED (overlay creation failed), QEMU_CRASHED
                (QEMU crashed during startup), VM_BOOT_TIMEOUT (guest agent not ready
                in time), GVPROXY_FAILED (gvproxy startup failed)
        """
        start_time = asyncio.get_running_loop().time()

        # Validate mutual exclusivity (create-specific)
        if snapshot_drive and direct_write_target:
            raise InternalError(
                "snapshot_drive and direct_write_target are mutually exclusive",
                code=ErrorCode.INTERNAL,
                context=None,
                hint=None,
                instance_id=None,
            )

        # Phase 1: Shared infrastructure (validation, workdir, overlay, cgroup, channel)
        infra = await self._setup_vm_infra(
            spec, tenant_id, task_id, reservation, base_image=self.images.base_image(spec.language), home_template=None
        )
        setup_complete_time = asyncio.get_running_loop().time()
        pending = infra.pending

        # Grant qemu-vm access to extra drives (snapshot creation target or read-only snapshot).
        # _setup_vm_infra only handles overlay/base images; extra drives need separate permissions.
        listeners: list[socket.socket] = []
        try:
            if infra.workdir.use_qemu_vm_user and direct_write_target:
                await _grant_drive(direct_write_target, writable=True, vm_id=infra.vm_id)
            elif infra.workdir.use_qemu_vm_user and snapshot_drive:
                await _grant_drive(snapshot_drive, writable=False, vm_id=infra.vm_id)

            # Phase 2: Build QEMU command (create-specific params — use guest values for QEMU)
            vdb_path = (
                str(direct_write_target) if direct_write_target else (str(snapshot_drive) if snapshot_drive else None)
            )
            cmd_started = asyncio.get_running_loop().time()
            listeners = _guest_port_listeners(infra.workdir)
            qemu_cmd = await build_qemu_cmd(
                self.settings,
                self.arch,
                infra.vm_id,
                infra.workdir,
                spec,
                expose_ports=expose_ports,
                direct_write=direct_write_target is not None,
                debug_boot=on_boot_log is not None,
                snapshot_drive=vdb_path,
                mem_path=mem_path,
                mem_path_share=mem_path_share,
                guest_port_fds=_listener_fds(listeners),
            )
            qemu_cmd_build_ms = round((asyncio.get_running_loop().time() - cmd_started) * 1000)
        except BaseException as error:
            _close_listeners(listeners)
            await self._cleanup_failed_owner(infra.vm_id, operation="pre_boot_resource_cleanup", error=error)
            raise

        # Phase 3: Start gvproxy BEFORE QEMU
        try:
            gvproxy_proc, gvproxy_log_task, gvproxy_start_ms, gvproxy_ready = await self._start_gvproxy_for_vm(
                infra,
                pending,
                spec,
                expose_ports,
            )
        except BaseException as error:
            _close_listeners(listeners)
            await self._cleanup_failed_owner(infra.vm_id, operation="pre_boot_cleanup", error=error)
            raise

        # Phase 4: Launch QEMU (shared helper)
        try:
            qemu_start_time = asyncio.get_running_loop().time()
            try:
                vm = await self._launch_qemu_vm(
                    infra,
                    pending,
                    qemu_cmd=qemu_cmd,
                    spec=spec,
                    packages_image=snapshot_drive,
                    reservation=reservation,
                    on_boot_log=on_boot_log,
                    gvproxy_proc=gvproxy_proc,
                    gvproxy_log_task=gvproxy_log_task,
                    pass_fds=tuple(listener.fileno() for listener in listeners),
                    cache_leases=cache_leases,
                )
            finally:
                _close_listeners(listeners)  # QEMU has its own copies
            qemu_fork_ms = round((asyncio.get_running_loop().time() - qemu_start_time) * 1000)

            # Phase 5: Wait for guest agent ready (create-specific)
            guest_wait_start = asyncio.get_running_loop().time()
            try:
                await vm.wait_for_guest(timeout=boot_timeout_seconds)
                if gvproxy_ready is not None:  # it got ready during the boot
                    await gvproxy_ready()
                boot_complete_time = asyncio.get_running_loop().time()
                guest_wait_ms = round((boot_complete_time - guest_wait_start) * 1000)

                # Timing breakdown
                vm.timing.setup_ms = round((setup_complete_time - start_time) * 1000)
                vm.timing.boot_ms = round((boot_complete_time - setup_complete_time) * 1000)
                vm.timing.overlay_ms = infra.overlay_ms
                vm.timing.qemu_cmd_build_ms = qemu_cmd_build_ms
                vm.timing.gvproxy_start_ms = gvproxy_start_ms
                vm.timing.qemu_fork_ms = qemu_fork_ms
                vm.timing.guest_wait_ms = guest_wait_ms

                if expose_ports:
                    vm.exposed_ports = expose_ports

                await vm.transition_state(VmState.READY)
            except TimeoutError as e:
                diag = await vm.collect_diagnostics()
                # The console (QEMU output, kernel, init) goes in the context, never the
                # message: str(error) is one line.
                context = {
                    **asdict(diag),
                    "language": spec.language,
                    "timeout_seconds": boot_timeout_seconds,
                    "qemu_cmd": " ".join(shlex.quote(arg) for arg in qemu_cmd)[:1000],
                    "kernel_path": str(self.images.kernel),
                    "initramfs_path": str(self.images.initramfs),
                    "overlay_image": str(infra.workdir.overlay_image),
                }
                logger.error("Guest agent boot timeout", extra=context)
                raise TransientError(
                    f"Guest agent not ready after {boot_timeout_seconds}s (qemu_running={diag.exit_code is None})",
                    code=ErrorCode.VM_BOOT_TIMEOUT,
                    context=context,
                    hint=None,
                    instance_id=None,
                ) from e
            except TransientError as e:  # QEMU_CRASHED: an end with 126 or 127 is no crash
                failed = self._qemu_exec_failed(vm, qemu_cmd=qemu_cmd, context=e.context)
                if failed is None:
                    raise
                raise failed from e

            total_boot_ms = round((asyncio.get_running_loop().time() - start_time) * 1000)
            logger.info(
                "VM created",
                extra={
                    "vm_id": infra.vm_id,
                    "language": spec.language,
                    "setup_ms": vm.timing.setup_ms,
                    "boot_ms": vm.timing.boot_ms,
                    "total_boot_ms": total_boot_ms,
                    "gvproxy_start_ms": gvproxy_start_ms,
                    "qemu_fork_ms": qemu_fork_ms,
                    "guest_wait_ms": guest_wait_ms,
                },
            )

            return vm

        except BaseException as error:
            await self._cleanup_failed_owner(infra.vm_id, operation="boot_failure_cleanup", error=error)
            raise

    async def restore_vm(
        self,
        spec: VmSpec,
        *,
        tenant_id: str,
        task_id: str,
        vmstate_path: Path,
        base_image: Path,
        snapshot_drive: Path | None,
        cache_leases: tuple[EntryLease, ...],
        expose_ports: list[ExposedPort] | None,
        reservation: ResourceReservation,
    ) -> QemuVM:
        """Restore a VM from a saved state (L1 template or saved instance): full CPU, RAM and device state.

        Uses shared helpers _setup_vm_infra / _launch_qemu_vm, then restores
        state via MigrationClient instead of waiting for guest boot.

        Args:
            spec: Creation settings; must match the saved VM (memory, vCPUs, and
                the network device: a state saved with virtio-net loads only into
                a VM with virtio-net)
            tenant_id: Tenant identifier
            task_id: Task identifier
            vmstate_path: Migration file. Its home disk must sit next to it with
                the .home suffix (MemorySnapshotManager.home_path_from_vmstate);
                it is cloned as the VM's home disk.
            base_image: Backing file of the overlay (for a saved instance: the
                image with the saved digest)
            snapshot_drive: Optional L2 qcow2 for disk state
            cache_leases: Leases on the cache entries the VM runs on (the L1 entry, the
                L2 snapshot drive): see create_vm
            expose_ports: List of ports to expose from guest to host.
            reservation: Pre-acquired admission reservation from reservation_context()
                (caller owns the reservation; a failed restore does not release it).

        Returns:
            QemuVM in READY state with l1_restored=True

        Raises:
            TransientError: Restore failed (retryable): LOCAL_DISK, QEMU_CRASHED,
                MIGRATION_FAILED (its cause chain holds MIGRATION_REJECTED or
                MIGRATION_UNSUPPORTED when QEMU refused the state),
                GUEST_SILENT_AFTER_RESTORE, GUEST_UNAVAILABLE (the guest channel failed)
            ConfigError: ASSET_NOT_FOUND: missing kernel or initramfs; HOST_DEPENDENCY:
                QEMU cannot run, or qemu-vm could not get access to a file or a cgroup
            InternalError: CLEANUP_UNCONFIRMED: the failed restore's QEMU may still run
                (it keeps the reservation)
        """
        async with self._launch_operation():
            start_time = asyncio.get_running_loop().time()

            # Early existence check: avoids wasting an admission slot + QEMU
            # launch if the vmstate was evicted between check_cache() and this
            # call (TOCTOU).
            if not vmstate_path.exists():
                raise TransientError(
                    f"vmstate file missing (evicted?): {vmstate_path}",
                    code=ErrorCode.LOCAL_DISK,
                    context={"vmstate_path": str(vmstate_path)},
                    hint=None,
                    instance_id=None,
                )

            try:
                return await self._restore_vm_with_reservation(
                    reservation,
                    spec,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    vmstate_path=vmstate_path,
                    base_image=base_image,
                    snapshot_drive=snapshot_drive,
                    cache_leases=cache_leases,
                    expose_ports=expose_ports,
                    start_time=start_time,
                )
            except BaseException:
                await self._cleanup_reservation_owners(reservation, release_admission=False)
                raise

    async def _restore_vm_with_reservation(  # noqa: PLR0912, PLR0915
        self,
        reservation: ResourceReservation,
        spec: VmSpec,
        *,
        tenant_id: str,
        task_id: str,
        vmstate_path: Path,
        base_image: Path,
        snapshot_drive: Path | None,
        cache_leases: tuple[EntryLease, ...],
        expose_ports: list[ExposedPort] | None,
        start_time: float,
    ) -> QemuVM:
        """Restore VM with an already-acquired reservation. Ownership transfers to VM on success."""
        # Phase 1: Shared infrastructure (validation, workdir, overlay, cgroup, channel)
        home_template = MemorySnapshotManager.home_path_from_vmstate(vmstate_path)
        if not home_template.exists():
            raise TransientError(
                f"L1 home disk missing (evicted?): {home_template}",
                code=ErrorCode.LOCAL_DISK,
                context={"vmstate_path": str(vmstate_path)},
                hint=None,
                instance_id=None,
            )
        infra = await self._setup_vm_infra(
            spec, tenant_id, task_id, reservation, base_image=base_image, home_template=home_template
        )
        pending = infra.pending

        # VM Templating: check metadata to see if save used template mode.
        # Template restore only when the save skipped RAM: the RAM template exists
        # and the metadata says use_template=True.
        ram_path = MemorySnapshotManager.ram_path_from_vmstate(vmstate_path)
        meta_path = vmstate_path.with_suffix(".vmstate.meta")

        # qemu-vm reads the saved state (and the RAM template) by ACL: the host owns
        # them, whichever QEMU saved them; then the snapshot drive, if present.
        try:
            use_template = False
            if ram_path.exists() and meta_path.exists():
                meta = SnapshotMetadata.from_json(meta_path.read_text())
                use_template = meta is not None and meta.use_template
            if infra.workdir.use_qemu_vm_user:
                state_files = [vmstate_path, ram_path] if use_template else [vmstate_path]
                if not await grant_qemu_vm_acl(state_files, perms="r"):
                    raise _host_dependency(
                        "Could not give qemu-vm read access to the saved VM state (needs setfacl, package acl, "
                        "files of this user, and a+x on the folders above them)",
                        context={"vm_id": infra.vm_id, "vmstate_path": str(vmstate_path)},
                    )
                if snapshot_drive:
                    await _grant_drive(snapshot_drive, writable=False, vm_id=infra.vm_id)
        except BaseException as error:
            await self._cleanup_failed_owner(infra.vm_id, operation="pre_restore_file_access_cleanup", error=error)
            raise

        # Phase 2: Socket activation for QMP (eliminates TOCTOU race)
        # Pre-create the QMP listening socket (and the guest port sockets:
        # _guest_port_listeners) so QEMU inherits them via pass_fds.
        # Only disabled when sudo wraps the QEMU command — sudo >= 1.9.14
        # closes all fds >= 3 (closefrom security policy). When the process is
        # already running as qemu-vm user, no sudo is needed and fd inheritance
        # works normally. See: https://www.sudo.ws/docs/man/sudo.man/
        qmp_parent_sock = None
        qmp_fd: int | None = None
        listeners: list[socket.socket] = []
        try:
            listeners = _guest_port_listeners(infra.workdir)
            if not _sudo_wraps_qemu(infra.workdir):
                qmp_parent_sock = create_unix_socket(str(infra.workdir.qmp_socket))
                qmp_fd = qmp_parent_sock.fileno()
        except BaseException as error:
            _close_listeners(listeners)
            await self._cleanup_failed_owner(infra.vm_id, operation="pre_restore_qmp_socket_cleanup", error=error)
            raise

        # Phase 3-5 wrapped in try so qmp_parent_sock is always cleaned up
        try:
            # Phase 3: Build QEMU command (restore-specific — use guest values for QEMU)
            vdb_path = str(snapshot_drive) if snapshot_drive else None
            mem_path_str = str(ram_path) if use_template else None

            qemu_cmd = await build_qemu_cmd(
                self.settings,
                self.arch,
                infra.vm_id,
                infra.workdir,
                spec,
                expose_ports=expose_ports,
                snapshot_drive=vdb_path,
                defer_incoming=True,  # VM starts paused, waiting for migration stream
                qmp_fd=qmp_fd,
                guest_port_fds=_listener_fds(listeners),
                mem_path=mem_path_str,
                mem_path_share=False,  # Clone: MAP_PRIVATE (COW)
            )

            # Phase 4: Start gvproxy if network topology requires it
            gvproxy_proc, gvproxy_log_task, _, gvproxy_ready = await self._start_gvproxy_for_vm(
                infra,
                pending,
                spec,
                expose_ports,
            )

            # Phase 5: Launch QEMU (shared helper) and restore snapshot
            vm = await self._launch_qemu_vm(
                infra,
                pending,
                qemu_cmd=qemu_cmd,
                spec=spec,
                packages_image=snapshot_drive,
                reservation=reservation,
                on_boot_log=None,
                gvproxy_proc=gvproxy_proc,
                gvproxy_log_task=gvproxy_log_task,
                pass_fds=tuple(sock.fileno() for sock in (*listeners, qmp_parent_sock) if sock is not None),
                cache_leases=cache_leases,
            )

            # Close parent's copy of the inherited fds after QEMU inherits them
            _close_listeners(listeners)
            if qmp_parent_sock is not None:
                qmp_parent_sock.close()
                qmp_parent_sock = None

            # Abort-check: fail fast if QEMU dies during startup
            # (as QemuStorageDaemon._wait_and_connect_qmp does)
            def _check_qemu_alive() -> None:
                if vm.process.returncode is not None:
                    context = {"vm_id": infra.vm_id, "returncode": vm.process.returncode}
                    raise self._qemu_exec_failed(vm, qemu_cmd=qemu_cmd, context=context) or TransientError(
                        f"QEMU died during L1 restore for {infra.vm_id} (rc={vm.process.returncode})",
                        code=ErrorCode.QEMU_CRASHED,
                        context=context,
                        hint=None,
                        instance_id=None,
                    )

            _check_qemu_alive()

            # Wait for QMP socket (only needed on sudo path where QEMU creates the socket)
            qemu_version = await probe_qemu_version()
            if qmp_fd is None:
                # Sudo path: QEMU creates the socket — must poll for it
                try:
                    await wait_for_socket(
                        vm.qmp_socket,
                        timeout=constants.MEMORY_SNAPSHOT_QMP_SOCKET_TIMEOUT_SECONDS,
                        abort_check=_check_qemu_alive,
                        keep_connection=False,
                    )
                except TimeoutError as e:
                    raise TransientError(
                        f"QMP socket never appeared for {infra.vm_id}",
                        code=ErrorCode.QEMU_CRASHED,
                        context={"vm_id": infra.vm_id, "qmp_socket": str(vm.qmp_socket)},
                        hint=None,
                        instance_id=None,
                    ) from e
            # else: socket activation — socket is pre-created, no wait needed

            # Restore VM state via QMP migration (load vmstate → resume vCPU)
            try:
                async with MigrationClient(vm.qmp_socket, infra.expected_uid) as client:
                    await client.restore_snapshot(
                        vmstate_path,
                        qemu_version=qemu_version,
                        timeout=migration_timeout(spec.memory_mb),
                        use_template=use_template,
                    )
            except Exception as e:
                # Check if QEMU died — provides a more informative error
                _check_qemu_alive()
                diag = await vm.collect_diagnostics()
                logger.error(
                    "L1 restore failed",
                    extra={**asdict(diag), "vmstate_path": str(vmstate_path)},
                )
                raise TransientError(
                    f"L1 restore failed for {infra.vm_id}: {e}",
                    code=ErrorCode.MIGRATION_FAILED,
                    context={**asdict(diag), "vmstate_path": str(vmstate_path)},
                    hint=None,
                    instance_id=None,
                ) from e

            # Connect guest channel with reconnection probe.
            # L1 restore needs probing because the virtio-serial chardev was connected
            # at save time — the restored VM needs the host to re-sync the connection.
            infra.channel._has_been_connected = True  # type: ignore[attr-defined]  # noqa: SLF001
            try:
                await infra.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
            except (TimeoutError, OSError) as e:  # no answer, or QEMU closed its port (it stopped)
                # `cont` replayed a home disk write that finds no space: QEMU paused the
                # VM again. LOCAL_DISK, so the L1 layer cold boots and keeps a good entry.
                await vm.raise_if_host_disk_full()
                # The saved state does not run (a torn RAM template, for example): the L1
                # layer cold boots and removes the entry, as for a stream QEMU refuses.
                raise TransientError(
                    f"L1 restore of {infra.vm_id}: the guest did not answer after its state loaded",
                    code=ErrorCode.GUEST_SILENT_AFTER_RESTORE,
                    context={"vm_id": infra.vm_id, "vmstate_path": str(vmstate_path)},
                    hint=None,
                    instance_id=None,
                ) from e
            except CommunicationError as e:  # socket peer check failed (internal class): nothing ran
                raise TransientError(
                    f"L1 restore of {infra.vm_id}: the guest channel failed: {e}",
                    code=ErrorCode.GUEST_UNAVAILABLE,
                    context={"vm_id": infra.vm_id, "error_type": type(e).__name__},
                    hint=None,
                    instance_id=None,
                ) from e

            if gvproxy_ready is not None:  # it got ready during the restore
                await gvproxy_ready()

            # VM is now running with REPL already warm.
            # Admission ownership was already claimed inside _launch_qemu_vm.
            vm.l1_restored = True

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
                    "language": spec.language,
                    "restore_ms": restore_ms,
                    "vmstate_path": str(vmstate_path),
                    "allow_network": spec.allow_network,
                    "expose_ports": [(p.internal, p.external) for p in expose_ports] if expose_ports else None,
                },
            )

            return vm

        except BaseException as error:
            _close_listeners(listeners)
            if qmp_parent_sock is not None:
                with suppress(OSError):
                    qmp_parent_sock.close()
            await self._cleanup_failed_owner(infra.vm_id, operation="restore_failure_cleanup", error=error)
            raise

    async def _force_cleanup_all_resources(
        self,
        vm_id: str,
        qemu_proc: ProcessWrapper | None = None,
        gvproxy_proc: ProcessWrapper | None = None,
        workdir: VmWorkingDirectory | None = None,
        vm_cgroup: Cgroup | None = None,
    ) -> dict[str, bool]:
        """Comprehensive cleanup of ALL VM resources in reverse dependency order.

        This is the MAIN cleanup method used in finally blocks.

        Best practices:
        - Cleans in reverse dependency order (processes -> workdir and cgroup)
        - Raises only CancelledError (the shielded work goes on); other errors are
          logged and give False for their resource
        - Safe to call multiple times (idempotent)
        - Handles None/already-cleaned resources
        - Returns status dict for monitoring/debugging

        Cleanup order (reverse dependencies):
        1. QEMU and gvproxy processes, in parallel
        2. Every process left in the VM cgroup: killed, then waited for
           (Cgroup.kill). The spawned process is not the whole VM: QEMU's
           async-teardown process outlives it, and through sudo the qemu-vm QEMU
           is not this user's to signal. results["qemu"] is True only once the
           cgroup is empty.
        3. Only once every process is confirmed dead, in parallel: the working
           directory (overlay, sockets) with the home disk, and the cgroup directory

        Args:
            vm_id: VM identifier for logging
            qemu_proc: QEMU subprocess (can be None)
            gvproxy_proc: gvproxy subprocess (can be None)
            workdir: VM working directory containing all temp files (can be None)
            vm_cgroup: the VM cgroup (None: the VM has none)

        Returns:
            Dictionary with cleanup status for each resource
        """
        logger.info("Starting comprehensive resource cleanup", extra={"vm_id": vm_id})
        results: dict[str, bool] = {}

        # Phase 1: Kill processes in parallel (independent operations). Each shield keeps the
        # cleanup going when the caller is cancelled; the cancel still comes out here, and the
        # caller keeps the registry and emergency ownership for a new attempt.
        process_results = await asyncio.shield(
            asyncio.gather(
                # No working folder: no known identity, SIGTERM as for qemu-vm
                _stop_qemu(qemu_proc, vm_id=vm_id, as_qemu_vm=workdir is None or workdir.use_qemu_vm_user),
                cleanup_process(
                    proc=gvproxy_proc,
                    name="gvproxy",
                    context_id=vm_id,
                    term_timeout=3.0,
                    kill_timeout=2.0,
                    kill_first=False,
                ),
                return_exceptions=True,
            )
        )
        results["qemu"] = process_results[0] if isinstance(process_results[0], bool) else False
        results["gvproxy"] = process_results[1] if isinstance(process_results[1], bool) else False
        # Keep any unconfirmed process in the emergency registry. Removing
        # it here would turn a failed SIGKILL/reap into an unowned orphan.
        if results["qemu"]:
            unregister_process(qemu_proc)
        if results["gvproxy"]:
            unregister_process(gvproxy_proc)
        # cgroup.kill whatever the cleanup above said: a qemu-vm QEMU outlives a killed sudo
        # and only its cgroup ends it, also while gvproxy is stuck
        if vm_cgroup is not None:
            drained = await asyncio.shield(asyncio.gather(vm_cgroup.kill(), return_exceptions=True))
            results["qemu"] = results["qemu"] and drained[0] is True

        # Phase 2: Cleanup workdir and cgroup in parallel (after processes dead)
        # workdir.cleanup() removes the home disk, then the overlay and sockets in one rmtree
        async def cleanup_workdir() -> bool:
            if workdir is None:
                return True
            return await workdir.cleanup()

        async def remove_cgroup() -> bool:
            return vm_cgroup is None or await vm_cgroup.remove()

        # Never remove a live QEMU's overlay, sockets, or cgroup. A retrying
        # reaper owns those resources until process death is confirmed.
        if not (results["qemu"] and results["gvproxy"]):
            logger.error(
                "Skipping dependent VM resource cleanup because process termination is unconfirmed",
                extra={"vm_id": vm_id},
            )
            results["workdir"] = False
            results["cgroup"] = False
        else:
            # Shield file cleanup from cancellation to ensure resources are fully released
            file_results = await asyncio.shield(
                asyncio.gather(
                    cleanup_workdir(),
                    remove_cgroup(),
                    return_exceptions=True,
                )
            )
            results["workdir"] = file_results[0] if isinstance(file_results[0], bool) else False
            results["cgroup"] = file_results[1] if isinstance(file_results[1], bool) else False
            # Gone: nothing left for the watcher or a sweep to end or remove
            if results["cgroup"] and self._run_dir is not None:
                with suppress(OSError):
                    self._run_dir.forget_cgroup(vm_id)
            if results["workdir"] and workdir is not None and self._run_dir is not None:
                with suppress(OSError):
                    self._run_dir.forget_folder(workdir.path.name)

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

    def _schedule_failed_cleanup_retry(self, vm: QemuVM) -> None:
        """Retain ownership and retry an unconfirmed process cleanup."""
        self._schedule_cleanup_retry(vm.vm_id, lambda: self.destroy_vm(vm, _schedule_retry=False))

    def _schedule_failed_pending_cleanup_retry(self, pending: _PendingLaunch) -> None:
        """Retry cleanup for a process that failed before QemuVM handoff."""
        self._schedule_cleanup_retry(
            pending.vm_id, lambda: self._cleanup_pending_launch(pending, _schedule_retry=False)
        )

    def _schedule_cleanup_retry(self, key: str, attempt: Callable[[], Awaitable[bool]]) -> None:
        """Own and retry one failed cleanup, deduped by key.

        A live task under the same key is left running; otherwise a fresh reaper
        is published and self-discards from the registry when it completes.
        """
        existing = self._failed_cleanup_tasks.get(key)
        if existing is not None and not existing.done():
            return

        task = asyncio.create_task(self._retry_cleanup_forever(key, attempt))
        self._failed_cleanup_tasks[key] = task

        def discard(completed: asyncio.Task[None]) -> None:
            if self._failed_cleanup_tasks.get(key) is completed:
                self._failed_cleanup_tasks.pop(key, None)

        task.add_done_callback(discard)

    async def _retry_cleanup_forever(self, key: str, attempt: Callable[[], Awaitable[bool]]) -> None:
        """Retry `attempt` (bounded backoff) until it confirms cleanup (True)."""
        delay = 1.0
        while True:
            await asyncio.sleep(delay)
            try:
                if await attempt():
                    return
            except Exception as error:
                logger.exception(
                    "Cleanup reaper attempt failed",
                    extra={"vm_id": key, "error": str(error)},
                )
            delay = min(delay * 2, 30.0)

    async def destroy_vm(
        self,
        vm: QemuVM,
        *,
        _schedule_retry: bool = True,
        _release_admission: bool = True,
    ) -> bool:
        """Destroy VM and clean up resources using defensive generic cleanup.

        This method uses the comprehensive cleanup orchestrator to ensure
        all resources are properly cleaned up even if some operations fail.

        Args:
            vm: QemuVM handle to destroy
            _schedule_retry: Start a reaper task when cleanup is not confirmed (the
                reaper itself passes False)
            _release_admission: Release the VM's reservation once cleanup is confirmed
                (False: the caller keeps it, as for a failed attempt it may retry)

        Returns:
            True only when process, working-directory, and cgroup cleanup are
            all confirmed. Any failed resource remains registry-owned for retry.

        State contract (what callers should key on): vm.state is published as
        DESTROYED once QEMU and gvproxy death is confirmed — even when this
        method returns False because workdir/cgroup cleanup is still pending.
        The bool attests full resource cleanup; DESTROYED attests process death.
        """
        try:
            async with vm.cleanup_lock:
                # Make direct VM APIs non-reusable before touching processes. This does
                # not claim DESTROYED yet: confirm_destroyed does, once QEMU is dead.
                await vm.begin_destroy()
                with suppress(OSError, RuntimeError):
                    await vm.channel.close()

                cleanup_results = await self._force_cleanup_all_resources(
                    vm_id=vm.vm_id,
                    qemu_proc=vm.process,
                    gvproxy_proc=vm.gvproxy_proc,
                    workdir=vm.workdir,
                    vm_cgroup=vm.cgroup,
                )
                processes_destroyed = all(cleanup_results.get(resource) is True for resource in ("qemu", "gvproxy"))
                await asyncio.gather(
                    *(
                        _finish_process_log_drain(task)
                        for resource, task in (
                            ("qemu", vm.qemu_log_task),
                            ("gvproxy", vm.gvproxy_log_task),
                        )
                        if cleanup_results.get(resource) is True
                    )
                )
                all_resources_destroyed = all(
                    cleanup_results.get(resource) is True for resource in ("qemu", "gvproxy", "workdir", "cgroup")
                )
                if processes_destroyed:
                    vm.release_cache_leases()  # QEMU no longer reads the cache entries
                    # Re-close the channel: an in-flight operation racing this
                    # destroy can reconnect to QEMU's chardevs between the
                    # first close and the process kill, resurrecting write
                    # workers. QEMU is dead now, so one re-close reaps them.
                    with suppress(OSError, RuntimeError):
                        await vm.channel.close()
                    await vm.confirm_destroyed()
                if not all_resources_destroyed:
                    logger.error(
                        "VM cleanup remains owned after unconfirmed resource cleanup",
                        extra={"vm_id": vm.vm_id, "cleanup_results": cleanup_results},
                    )
                    if processes_destroyed and not _release_admission:
                        # The caller keeps the reservation (a failed attempt it may
                        # retry): the processes are dead, the reaper only removes files.
                        vm.holds_admission_slot = False
                        vm.resource_reservation = None

        except (asyncio.CancelledError, Exception):
            # Event-loop dictionary publication is synchronous, matching the
            # pending->VM launch handoff. A repeated cancel cannot interrupt it.
            self._vms[vm.vm_id] = vm
            if _schedule_retry:
                self._schedule_failed_cleanup_retry(vm)
            raise

        if all_resources_destroyed:

            async def commit_cleanup() -> None:
                # Admission is released before the VM is unpublished. No await
                # may follow clearing the ownership fields.
                if vm.holds_admission_slot and vm.resource_reservation is not None and _release_admission:
                    await self._admission.release(vm.resource_reservation)
                async with self._vms_lock:
                    self._vms.pop(vm.vm_id, None)
                    if vm.holds_admission_slot:
                        vm.holds_admission_slot = False
                        vm.resource_reservation = None

            # A cancel comes out only once commit_cleanup completed (await_cancellation_safe)
            try:
                await _await_cancellation_safe(commit_cleanup())
            except Exception:
                self._vms[vm.vm_id] = vm
                if _schedule_retry:
                    self._schedule_failed_cleanup_retry(vm)
                raise
            return True

        # Keep the registry entry, admission reservation, and every dependent
        # resource. A dedicated reaper retries until the full cleanup result is
        # confirmed, even when QEMU and gvproxy are already dead.
        self._vms[vm.vm_id] = vm
        if _schedule_retry:
            self._schedule_failed_cleanup_retry(vm)
        return False

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

    async def _prepare_home_disk(self, workdir: VmWorkingDirectory, spec: VmSpec, home_template: Path | None) -> None:
        """Create the VM's private home disk: sparse, so only written blocks take space.

        Blank (spec.disk_mb, tiny-init formats it) at a cold boot; a clone of the
        template's disk at a restore (L1 template or saved instance). Raises
        TransientError (LOCAL_DISK) when the file cannot be made, ConfigError
        (HOST_DEPENDENCY) when qemu-vm cannot get access to it.
        """
        try:
            if home_template is None:
                await asyncio.to_thread(create_sparse, workdir.home_disk, spec.disk_mb * 1024 * 1024)
            else:
                # A copy-on-write clone shares its blocks with the template until the guest
                # overwrites them. Noted before the file exists, so no gate 4 probe meanwhile
                # counts them as written; a plain copy (ext4) takes them from free space at
                # once, and a failed clone shares nothing: the note goes.
                self._admission.note_shared_blocks(workdir.home_disk.name, allocated_bytes(home_template.stat()))
                cloned = False
                try:
                    cloned = await asyncio.to_thread(clone_sparse, home_template, workdir.home_disk)
                finally:
                    if not cloned:
                        self._admission.forget_shared_blocks(workdir.home_disk.name)
        except OSError as e:
            # Template evicted meanwhile, disk full, no reflink support...: on
            # TransientError the caller falls back to a cold boot (L1), retries
            # (cold boot), or fails reuse() (saved instance).
            raise TransientError(
                f"Could not prepare the home disk: {e.strerror or type(e).__name__}",
                code=ErrorCode.LOCAL_DISK,
                context={"vm_id": workdir.vm_id, "home_disk": str(workdir.home_disk)},
                hint=None,
                instance_id=None,
            ) from e
        # ACL, not chown: the host keeps owning the 0600 file (it clones it for
        # L1 and deletes it without sudo); no account except this user and
        # qemu-vm can read it. Every QEMU shares the qemu-vm uid: see the TODO
        # (security) in qemu_cmd.py.
        if workdir.use_qemu_vm_user and not await grant_qemu_vm_private_file_access(workdir.home_disk):
            raise _host_dependency(
                "Could not give qemu-vm read-write access to the home disk (needs setfacl, package acl)",
                context={"vm_id": workdir.vm_id, "home_disk": str(workdir.home_disk)},
            )

    async def _apply_overlay_permissions(self, base_image: Path, overlay_image: Path) -> bool:
        """Whether QEMU runs as qemu-vm for this VM: the decision of this process
        (probe_qemu_as_qemu_vm), never a fallback of one VM. As qemu-vm, first give it
        the base image (read), the overlay (owner) and the working folder (group
        qemu-vm, mode 2770: QEMU makes its sockets there; other users get no access).

        Raises:
            ConfigError: HOST_DEPENDENCY: qemu-vm could not get one of them
        """
        if not await probe_qemu_as_qemu_vm():
            return False
        # The folder comes last: the overlay grant makes it a+x (ensure_traversable),
        # and the folder grant sets its whole mode.
        if not (
            await grant_qemu_vm_file_access(base_image, writable=False)
            and await grant_qemu_vm_file_access(overlay_image, writable=True)
            and grant_qemu_vm_dir_access(overlay_image.parent)
        ):
            raise _host_dependency(
                "Could not give qemu-vm the base image (read), the overlay (sudo -n chown) "
                "or the VM folder (group qemu-vm, mode 2770)",
                context={"base_image": str(base_image), "overlay_image": str(overlay_image)},
            )
        return True

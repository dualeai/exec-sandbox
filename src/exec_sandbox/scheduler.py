"""Scheduler: the entry point of exec-sandbox. It starts the VMs, runs code in
them and destroys them; use it as an async context manager.

Architecture:
- VM lifecycle: a VM process is never shared. Each new() or reuse() starts a fresh VM,
  destroyed at close().
- Warm pool: Pre-started VMs waiting for commands (faster than cold boot).
- Snapshot cache: L1 memory snapshots (a restore in place of a boot); L2 disk
  snapshots with the packages installed, and the L3 snapshot cache in S3.
- Admission: a VM starts only when the host has the memory, CPU and disk for it.
- Saved instances (with instance_store and instance_keyring): Session.save() stores
  a VM's state; reuse() starts a VM from it.

Example:
    ```python
    from exec_sandbox import Scheduler, SchedulerConfig, VmSpec

    # One session, one execution
    async with Scheduler() as scheduler:
        async with await scheduler.new(VmSpec(language="python")) as session:
            result = await session.exec("print('hello')")
            print(result.stdout)  # "hello\\n"

    # With packages and streaming
    DATA = VmSpec(language="python", packages=("pandas==3.0.1",))
    async with Scheduler() as scheduler:
        async with await scheduler.new(DATA) as session:
            await session.exec("import pandas; print(pandas.__version__)", on_stdout=print)

    # With the L3 snapshot cache in S3
    async with Scheduler(SchedulerConfig(s3_bucket="my-snapshots")) as scheduler:
        ...
    ```
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable  # noqa: TC003 - Used at runtime for on_stdout/on_stderr parameters
from pathlib import Path  # noqa: TC003 - Used at runtime
from typing import TYPE_CHECKING, Self

from pydantic import ValidationError

from exec_sandbox import constants
from exec_sandbox._imports import require_instance_admin, require_instance_manager
from exec_sandbox._logging import get_logger
from exec_sandbox._version import __version__
from exec_sandbox.aio_utils import gather_or_cancel, settle_and_report
from exec_sandbox.assets import ensure_assets, ensure_base_image, images_download_dir
from exec_sandbox.config import MAX_SESSION_IDLE_TIMEOUT_SECONDS, MIN_SESSION_IDLE_TIMEOUT_SECONDS, SchedulerConfig
from exec_sandbox.disk_snapshot_manager import DiskSnapshotManager
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InternalError,
    InvalidInputError,
    RouteKey,
    SandboxError,
    TransientError,
    named_instance,
    no_instance_store,
)
from exec_sandbox.files.locks import EntryLease  # noqa: TC001 - Used at runtime
from exec_sandbox.images import image_digest, kernel_dir_of
from exec_sandbox.instance.ids import check_instance_id
from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager
from exec_sandbox.migration_client import refusal_cause
from exec_sandbox.models import ExposedPort, Language, StartupTiming
from exec_sandbox.package_validator import BUNDLED_NPM_ALLOW_LIST, BUNDLED_PYPI_ALLOW_LIST, PackageValidator
from exec_sandbox.platform_utils import detect_host_arch
from exec_sandbox.port_forward import resolve_port_mappings
from exec_sandbox.session import Session
from exec_sandbox.settings import Settings
from exec_sandbox.system_probes import (
    raise_fd_limit,
    raise_nproc_limit,
    require_qemu_machine_types,
    warn_cgroup_pids_limit,
    warn_damon,
    warn_max_map_count,
    warn_overcommit,
    warn_threads_max,
    warn_zswap,
)
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.vm_types import GuestBoot
from exec_sandbox.warm_vm_pool import WarmVMPool
from exec_sandbox.worker_facts import WorkerFacts, counter_fact, machine_facts, qemu_version_fact

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from exec_sandbox.instance.manager import InstanceAdmin, InstanceManager, InstanceState
    from exec_sandbox.instance.models import InstanceInfo
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.vm_spec import VmSpec

logger = get_logger(__name__)


def _scheduler_state(message: str) -> InvalidInputError:
    """The InvalidInputError (SCHEDULER_STATE) of a call the scheduler cannot take now."""
    return InvalidInputError(message, code=ErrorCode.SCHEDULER_STATE, context=None, hint=None, instance_id=None)


def _internal(message: str) -> InternalError:
    """The InternalError (INTERNAL) of a scheduler invariant that does not hold."""
    return InternalError(message, code=ErrorCode.INTERNAL, context=None, hint=None, instance_id=None)


_NOT_STARTED = "Scheduler not started. Use: async with Scheduler() as scheduler:"
_NOT_INITIALIZED = "Scheduler resources not initialized"
_STILL_STARTING = "Scheduler is still starting"


class Scheduler:
    """Starts the VMs, runs code in them, and destroys them.

    Lifecycle:
        async with Scheduler(config) as scheduler:
            # The Scheduler is ready
            async with await scheduler.new(spec) as session:
                result = await session.exec(...)
        # All VMs are destroyed and their resources released

    Methods:
        new(spec): start a new VM; returns the Session that owns it.
        reuse(instance_id): start a VM from a saved instance; returns the
            Session that continues it.
        wait_pool_ready(): wait until the warm pool is full.
        describe_instance(instance_id), list_instances(),
            delete_instance(instance_id): read and delete saved instances.
        check_reuse(instance_id): what reuse() checks, with no download and no VM.
        describe_worker(): this worker's facts, for a broker.

    Use one Scheduler per process: each Scheduler counts only its own VMs for
    admission. Run calls at the same time with asyncio.

    Attributes:
        config: SchedulerConfig (immutable)
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        """Make a Scheduler from a configuration.

        Args:
            config: Optional SchedulerConfig. None uses the defaults.
        """
        self.config = config or SchedulerConfig()
        self._instances: InstanceManager | None = None
        self._instances_exit: contextlib.AsyncExitStack | None = None
        self._vm_manager: VmManager | None = None
        self._snapshot_manager: DiskSnapshotManager | None = None
        self._memory_snapshot_manager: MemorySnapshotManager | None = None
        self._warm_pool: WarmVMPool | None = None
        self._package_validator: PackageValidator | None = None
        self._started = False
        # A start is in progress: a second start or a stop meanwhile would take or stop its components
        self._starting = False
        self._shutdown_task: asyncio.Task[None] | None = None

    @property
    def _vms(self) -> VmManager:
        """The VM manager of a started scheduler (InternalError before: a bug)."""
        if self._vm_manager is None:
            raise _internal(_NOT_INITIALIZED)
        return self._vm_manager

    @property
    def _snapshots(self) -> DiskSnapshotManager:
        """The L2 snapshot manager of a started scheduler (InternalError before: a bug)."""
        if self._snapshot_manager is None:
            raise _internal(_NOT_INITIALIZED)
        return self._snapshot_manager

    async def __aenter__(self) -> Self:
        """Start the Scheduler.

        Checks the host (open files, processes; warns about unsafe kernel settings)
        and QEMU; downloads the missing kernel and initramfs from GitHub Releases (unless
        auto_download_assets=False), and the base images of a warm pool (a base image
        otherwise downloads at the first new() of its language); opens the instance store (with instance_store:
        keyring, settings, QEMU version, store check) and the snapshot caches; starts
        the warm pool (wait_pool_ready() waits until it is full).

        Raises:
            TransientError: ASSET_DOWNLOAD (a VM image download failed),
                INSTANCE_STORE_UNAVAILABLE (the instance store check got no answer),
                LOCAL_DISK (a host error on a cache folder or the home disk folder: disk full, I/O
                error; or a read of the host's clock counter rate, CPU identity or KVM TSC
                scaling hit a process, memory or file-handle limit or got no answer in time),
                QEMU_CRASHED (a QEMU probe hit a process, memory or file-handle limit, or
                gave no answer in time), OVERLAY_FAILED (qemu-storage-daemon did not start)
            InvalidInputError: SCHEDULER_STATE: already started, still starting, or still
                shutting down
            ConfigError: ASSET_NOT_FOUND (a VM image is missing and was not downloaded),
                EXTRA_MISSING (instance_store set without the extra it needs),
                INSTANCE_CONFIG (the instance store check failed), MIGRATION_UNSUPPORTED
                (instance_store set and QEMU below 11.0), HOST_DEPENDENCY,
                SETTING_INVALID (an EXEC_SANDBOX_* variable is not valid: the message names it, never
                its value; EXEC_SANDBOX_HOME_DISK_DIR is not a folder of this user that only it can write;
                or a cache folder cannot be made or written: instance_asset_dir, disk_snapshot_cache_dir,
                memory_snapshot_cache_dir, or the download cache under EXEC_SANDBOX_CACHE_DIR)
        """
        if self._started:
            raise _scheduler_state("Scheduler already started")
        if self._starting:
            raise _scheduler_state(_STILL_STARTING)
        if self._shutdown_task is not None:
            if not self._shutdown_task.done():
                raise _scheduler_state("Scheduler is still shutting down")
            # A completed lifecycle may be reused only after its shutdown was
            # successful. result() preserves a prior cleanup failure.
            self._shutdown_task.result()
            self._shutdown_task = None

        self._starting = True
        try:
            await self._start()
        finally:
            self._starting = False
        self._started = True
        logger.info("Scheduler started successfully")
        return self

    async def _start(self) -> None:
        """__aenter__ after its state checks: the host checks, then each component. A failure
        (a cancel too) rolls back what was made, through the shutdown task, then raises."""
        # A completed lifecycle never lends component references to the next
        # one. This also makes a pre-manager startup failure an inert state.
        self._vm_manager = None
        self._snapshot_manager = None
        self._memory_snapshot_manager = None
        self._warm_pool = None
        self._instances = None
        self._instances_exit = None

        # System limit probes: raise safe limits, warn about unsafe ones.
        raise_fd_limit()
        raise_nproc_limit()
        warn_overcommit()
        warn_max_map_count()
        warn_threads_max()
        warn_cgroup_pids_limit()
        warn_zswap()
        warn_damon()

        # Resolve images directory (downloads if needed and allowed)
        images_dir = await ensure_assets(
            override=self.config.images_dir,
            download=self.config.auto_download_assets,
        )

        # Create Settings from SchedulerConfig
        settings = settings_from_config(self.config, images_dir=images_dir)

        logger.info(
            "Starting scheduler",
            extra={
                "warm_pool_size": self.config.warm_pool_size,
                "s3_enabled": settings.s3_bucket is not None,
                "auto_download_assets": self.config.auto_download_assets,
            },
        )

        # Initialize VmManager (handles backpressure internally via admission controller).
        # From this publication onward, every startup failure rolls the whole
        # partial lifecycle back through one scheduler-owned shutdown task.
        self._vm_manager = VmManager(settings)
        try:
            await self._vm_manager.start()  # Pre-warms all system probe caches

            # Saved instances: keyring, store settings, QEMU floor and store probe fail the start (fail fast)
            if settings.instance_store is not None:
                self._instances_exit = contextlib.AsyncExitStack()
                manager = require_instance_manager(store=settings.instance_store)
                self._instances = await self._instances_exit.enter_async_context(
                    manager.open(
                        settings, work_dir=self._vm_manager.work_dir / "instances", vm_manager=self._vm_manager
                    )
                )

            # Initialize DiskSnapshotManager (L2 local cache always available, L3 S3 optional)
            self._snapshot_manager = DiskSnapshotManager(settings, self._vm_manager)
            await self._snapshot_manager.start()  # removes .part files of killed processes

            # Initialize MemorySnapshotManager (L1 cache)
            self._memory_snapshot_manager = MemorySnapshotManager(settings, self._vm_manager)

            # Admission evicts L2 and L1 entries before a VM waits for home disk space
            self._vm_manager.add_cache(self._snapshot_manager)
            self._vm_manager.add_cache(self._memory_snapshot_manager)
            if self._instances is not None:  # images that loads fetched from the instance store
                self._vm_manager.add_cache(self._instances.assets)

            # Initialize WarmVMPool (optional)
            if self.config.warm_pool_size > 0:
                # The pool pre-boots warm_pool_size VMs per language, and its startup
                # hashes each base image (compute_cache_key -> VmManager.images.base_image).
                # On a fresh cache ensure_assets() fetched only kernel/initramfs, so
                # download every pooled base image first, else start() raises
                # ConfigError (ASSET_NOT_FOUND). Mirrors the per-run fetch in _prepare_vm
                # and _launch_for.
                if self.config.auto_download_assets:
                    await gather_or_cancel(ensure_base_image(language, images_dir=images_dir) for language in Language)
                self._warm_pool = WarmVMPool(
                    self._vm_manager, self.config, memory_snapshot_manager=self._memory_snapshot_manager
                )
                await self._warm_pool.start()
        except BaseException:
            # Caller cancellation (Ctrl-C too) waits for the rollback: leaked VM processes
            # would be worse than a slow interrupt
            self._shutdown_task = asyncio.create_task(self._shutdown())
            if (rollback_error := await settle_and_report(self._shutdown_task)) is not None:
                logger.error(
                    "Scheduler startup rollback failed",
                    extra={"error": str(rollback_error)},
                )
            raise

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Stop the Scheduler: wait for the saves in progress, destroy every VM, then close the instance store.

        Raises InvalidInputError SCHEDULER_STATE while a start is in progress (cancel the start
        instead: it stops what it made).
        """
        if self._starting:
            raise _scheduler_state(_STILL_STARTING)
        self._started = False
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self) -> None:
        """Shutdown scheduler and clean up all resources.

        Shutdown sequence:
        1. Stop WarmVMPool (drains and destroys pre-booted VMs, frees resource slots)
        2. Stop MemorySnapshotManager (awaits in-flight background L1 saves)
        3. Stop DiskSnapshotManager (awaits in-flight L3 uploads up to
           L3_UPLOAD_STOP_GRACE_SECONDS, then cancels the ones left)
        4. Wait for instance saves in flight (InstanceManager.stop: new ones are refused)
        5. Stop VmManager: refuses new launches, waits for those in flight (new() and
           reuse() calls), then destroys the remaining VMs (sessions not closed) and
           stops the overlay daemon
        6. Close the instance store (after every VM that could save is gone)

        IMPORTANT: WarmVMPool must stop BEFORE MemorySnapshotManager. Background
        L1 save tasks (from schedule_background_save) create sacrificial VMs via
        vm_manager.create_vm(), which blocks on admission.acquire() until a
        resource slot is available. If the warm pool holds all slots, stopping
        MemorySnapshotManager first deadlocks: stop() awaits the background task,
        the task waits for a slot, and slots are only freed when the pool drains.

        Always completes cleanup, even on exceptions.
        """
        logger.info("Shutting down scheduler")
        deferred_error: BaseException | None = None
        # Snapshot this lifecycle's components. Re-entry is rejected while this
        # task is pending, and local references make that boundary explicit.
        warm_pool = self._warm_pool
        memory_snapshot_manager = self._memory_snapshot_manager
        snapshot_manager = self._snapshot_manager
        vm_manager = self._vm_manager
        instances = self._instances
        instances_exit = self._instances_exit

        # Stop WarmVMPool (drains VMs, frees resource slots for background saves)
        if warm_pool:
            logger.info("Draining warm VM pool, this may take a moment")
            deferred_error = await _cleanup_step(warm_pool.stop())  # core VM cleanup still runs after it

        # Stop MemorySnapshotManager (wait for background saves: the drained pool freed their resource slots)
        if memory_snapshot_manager:
            logger.info("Waiting for in-flight background snapshot saves to complete")
            save_error = await _cleanup_step(memory_snapshot_manager.stop())  # runs after a failed pool stop too
            deferred_error = deferred_error or save_error

        # Wait for the L3 uploads in flight: a task left running is cancelled when the loop closes
        if snapshot_manager is not None:
            upload_error = await _cleanup_step(snapshot_manager.stop())  # runs even after an earlier failed step
            deferred_error = deferred_error or upload_error

        # Let instance saves in flight end before their VMs and the work folder go
        if instances is not None:
            drain_error = await _cleanup_step(instances.stop())  # runs even after an earlier failed step
            deferred_error = deferred_error or drain_error

        # Stop VmManager: it refuses new launches and closes admission, waits for the launches
        # in flight (a new() or reuse() past its start check: the VM it makes is tracked),
        # then destroys every VM it tracks (the overlay daemon stops last). The count below is
        # for the log only: it can miss a VM of a launch in flight.
        if vm_manager:
            if active_vms := vm_manager.get_active_vms():
                logger.warning("Destroying remaining VMs on shutdown", extra={"count": len(active_vms)})
            stop_error = await _cleanup_step(vm_manager.stop())
            deferred_error = deferred_error or stop_error

        # Close the instance store (its clients) after every VM that could save is gone
        if instances_exit is not None:
            close_error = await _cleanup_step(instances_exit.aclose())  # runs even after an earlier failed step
            deferred_error = deferred_error or close_error

        logger.info("Scheduler shutdown complete")
        if deferred_error is not None:
            raise deferred_error

    async def wait_pool_ready(self, timeout: float = 120) -> None:
        """Wait until the warm pool is full, or until `timeout` seconds pass (then log a
        warning and return; no error). Returns at once when the pool is off."""
        if self._warm_pool:
            await self._warm_pool.wait_until_ready(timeout)

    async def new(
        self,
        spec: VmSpec,
        *,
        idle_timeout_seconds: int | None = None,
        on_boot_log: Callable[[str], None] | None = None,
        instance_id: str | None = None,
    ) -> Session:
        """Start a new VM made from `spec` and return the session that owns it.

        The VM comes from the warm pool, an L1 memory snapshot or a cold boot.
        It is never shared: close() (or the context manager, the idle timeout,
        or a failure that closes the session) destroys it.

        Args:
            spec: What the VM is made of (language, memory, disk, packages, network, ports).
            idle_timeout_seconds: Close after this long without a call, 10 to 3600 seconds
                (the range of SchedulerConfig.session_idle_timeout_seconds). None: that config value.
            on_boot_log: Callback for the boot console output (kernel, init, guest agent).
                When given, it turns on verbose boot logging (and skips the warm pool and the L1
                restore).
            instance_id: Id the session's first save() gives the new instance, unless that save
                names another id (save(instance_id=...)) (1 to 64
                lowercase letters, digits, ".", "_" or "-", starting with a letter or
                digit), checked before any VM starts. None: the library makes a random
                id. An id that already exists makes that save fail with
                OutcomeUnknownError (INSTANCE_CONFLICT); nothing is overwritten, and the
                session stays open (save it under another id with save(instance_id=...)).

        Returns:
            Session that runs the code. Use it as an async context manager.

        Raises:
            TransientError: CAPACITY (no resources within the admission timeout),
                LOCAL_DISK (the home disk cannot fit on its filesystem, raised at once),
                or VM creation failed and can succeed later.
            InvalidInputError: SCHEDULER_STATE (scheduler not started),
                INSTANCE_INVALID_ID (`instance_id` breaks the id rule), ARGUMENT_INVALID
                (`idle_timeout_seconds` out of range; a package without a version or with a
                bad name; packages for raw), PACKAGE_NOT_ALLOWED (package not in the
                allow-list), PACKAGE_INSTALL_FAILED.
            ConfigError: INSTANCE_CONFIG (`instance_id` is set, but no instance store
                is configured), ASSET_NOT_FOUND (the base image is not on this host and was
                not downloaded: auto_download_assets off, offline mode, or not in the release),
                HOST_DEPENDENCY, SETTING_INVALID (no write access to
                disk_snapshot_cache_dir or, when a base image is downloaded, to the download
                cache under EXEC_SANDBOX_CACHE_DIR; a read-only disk; or the folder cannot be
                made again).
            RouteError: CAPACITY_NEVER_FITS: the spec is larger than the whole budget
                (raised at once).
            InternalError: VM creation failed (a bug or a broken host).
        """
        if instance_id is not None:
            self._require_instances(check_instance_id(instance_id))  # before any VM starts
        idle_timeout = self._idle_timeout(idle_timeout_seconds)
        loop = asyncio.get_running_loop()
        start = loop.time()
        vm, is_cold_boot, admission_ms = await self._prepare_vm(
            spec, task_id=f"session-{id(self)}", on_boot_log=on_boot_log
        )
        total_ms = round((loop.time() - start) * 1000)

        session = Session(
            vm=vm,
            vm_manager=self._vms,
            spec=spec,
            startup_timing=_startup_timing(vm, is_cold_boot=is_cold_boot, total_ms=total_ms, admission_ms=admission_ms),
            instances=self._instances,
            instance=None,
            new_instance_id=instance_id,
            idle_timeout_seconds=idle_timeout,
            default_timeout_seconds=self.config.default_timeout_seconds,
        )
        logger.info(
            "Session created",
            extra={
                "vm_id": vm.vm_id,
                "language": spec.language,
                "idle_timeout_seconds": idle_timeout,
                "startup_ms": total_ms,
            },
        )
        return session

    async def reuse(self, instance_id: str, *, idle_timeout_seconds: int | None = None) -> Session:
        """Start a VM from a saved instance and return the session that continues it.

        The VM resumes where the last save left it: memory, processes and the
        files in /home/user. It is made from the VmSpec stored with the instance
        (you cannot change the spec). `session.instance` is set, and the
        next `save()` adds one generation.

        Args:
            instance_id: An id from `InstanceRef.instance_id`.
            idle_timeout_seconds: Close after this long without a call, as in new().

        Raises:
            TransientError: INSTANCE_STORE_UNAVAILABLE, CAPACITY (no resources within
                the admission timeout), LOCAL_DISK (a local file error of the load: disk
                full, work folder gone; or the home disk cannot fit on its filesystem), or
                the VM could not start from the saved state (the stored instance is unchanged).
            InvalidInputError: INSTANCE_INVALID_ID (not a valid instance id, no request
                is sent), SCHEDULER_STATE (scheduler not started), ARGUMENT_INVALID
                (`idle_timeout_seconds` out of range, no request is sent).
            ConfigError: INSTANCE_CONFIG (no instance store, or the store refuses this
                worker), INSTANCE_KEY_MISSING (this worker's keyring lacks the instance's
                key; do not delete the instance), HOST_DEPENDENCY (for example qemu-vm
                cannot get access to the loaded migration file).
            RouteError: INSTANCE_HOST_MISMATCH: this worker cannot run it: arch,
                accelerator, guest CPU model, host CPU (ARM with KVM or HVF) or guest clock
                rate (counter_hz) differ from the saved ones, its
                QEMU lacks the saved machine type (checked before any download), a missing
                guest image, data that another exec-sandbox release wrote and this one
                cannot read, or a QEMU older than the saver's that refuses the saved state
                (a refusal on the same or a newer QEMU is InternalError INTERNAL).
                CAPACITY_NEVER_FITS: the spec is larger than the whole budget (raised before
                the download).
            InstanceGoneError: INSTANCE_NOT_FOUND, INSTANCE_EXPIRED, INSTANCE_INTEGRITY
                (the stored data was changed or damaged): start again with new().
            InternalError: The VM could not start (a bug or a broken host).
        """
        instances = self._require_instances(check_instance_id(instance_id))
        idle_timeout = self._idle_timeout(idle_timeout_seconds)
        loop = asyncio.get_running_loop()
        start = loop.time()
        snapshots = self._snapshots

        # The local L2 image is leased from its lookup on: admission (below) may evict
        # cache entries for disk space, never this one. The VM takes its own share.
        packages_lease: EntryLease | None = None

        async def packages_image(spec: VmSpec) -> Path | None:
            # A local L2 is used only if its digest matches; else load() downloads the
            # image by digest from the instance store.
            nonlocal packages_lease
            lease = await _packages_lease(snapshots, spec)
            if packages_lease is not None:  # an earlier attempt of the load
                packages_lease.release()
            packages_lease = lease
            return lease.path if lease is not None else None

        try:
            with named_instance(instance_id):  # VM and guest errors name no instance
                state = await instances.load(instance_id, packages_image=packages_image)
                return await self._reuse_loaded(
                    instances,
                    instance_id,
                    state=state,
                    packages_lease=packages_lease,
                    start=start,
                    idle_timeout=idle_timeout,
                )
        except SandboxError as e:
            # One outcome record per load; the values are in the text (the CLI log format shows no extra)
            logger.warning(
                f"Instance {instance_id} not reused: {e.code} ({e.action}): {e.message}",
                extra={"instance_id": instance_id, "error_code": str(e.code), "action": e.action, "context": e.context},
            )
            raise
        finally:
            if packages_lease is not None:
                packages_lease.release()

    async def check_reuse(self, instance_id: str) -> None:
        """What reuse() checks before its download, with no download and no VM: the instance
        exists, this keyring and this release open it, this worker has the saved host facts
        and the budget, and each guest image is on this worker or in the store. QEMU still
        decides at a reuse() (a refused state: RouteError on an older QEMU).

        Raises what reuse() raises before its download: InvalidInputError
        INSTANCE_INVALID_ID or SCHEDULER_STATE; InstanceGoneError INSTANCE_NOT_FOUND or
        INSTANCE_INTEGRITY; ConfigError INSTANCE_CONFIG or INSTANCE_KEY_MISSING; RouteError
        INSTANCE_HOST_MISMATCH or CAPACITY_NEVER_FITS; TransientError
        INSTANCE_STORE_UNAVAILABLE or LOCAL_DISK (a lock file of the guest image cache cannot
        be opened).
        """
        instances = self._require_instances(check_instance_id(instance_id))
        snapshots = self._snapshots
        leases: list[EntryLease] = []

        async def packages_image(spec: VmSpec) -> Path | None:
            if (lease := await _packages_lease(snapshots, spec)) is None:
                return None
            leases.append(lease)  # kept while the check reads its digest
            return lease.path

        try:
            await instances.check(instance_id, packages_image=packages_image)
        finally:
            for lease in leases:
                lease.release()

    async def _reuse_loaded(
        self,
        instances: InstanceManager,
        instance_id: str,
        *,
        state: InstanceState,
        packages_lease: EntryLease | None,
        start: float,
        idle_timeout: int,
    ) -> Session:
        """reuse() after the load: restore the VM of `state`, then the session."""
        loop = asyncio.get_running_loop()
        load_ms = round((loop.time() - start) * 1000)
        spec = state.body.spec
        own_layer = packages_lease is not None and state.packages_image == packages_lease.path
        leases = (*state.asset_leases, *((packages_lease,) if packages_lease is not None and own_layer else ()))
        try:
            loader_qemu = await qemu_version_fact()  # for the outcome record; before a VM exists
            resolved_ports = resolve_port_mappings(list(spec.expose_ports)) if spec.expose_ports else []
            task_id = f"reuse-{id(state)}"
            admission_start = loop.time()
            # Kept while it waits: the loaded state's folder and the packages layer
            held = frozenset({state.folder, *(lease.path for lease in leases)})
            async with self._vms.reservation_context(
                spec, vm_id=f"{constants.SCHEDULER_TENANT_ID}-{task_id}", held=held
            ) as reservation:
                admission_ms = round((loop.time() - admission_start) * 1000)
                try:
                    vm = await self._vms.restore_vm(
                        spec,
                        tenant_id=constants.SCHEDULER_TENANT_ID,
                        task_id=task_id,
                        vmstate_path=state.vmstate,
                        base_image=state.base_image,
                        snapshot_drive=state.packages_image,
                        cache_leases=leases,
                        expose_ports=resolved_ports or None,
                        reservation=reservation,
                        machine_type=state.body.host.machine_type,
                        egress_domains=state.body.egress_domains,  # the policy at save
                        boot=GuestBoot(
                            kernel=state.body.guest.kernel,
                            initramfs=state.body.guest.initramfs,
                            counter_hz=state.body.guest.counter_hz,
                        ),
                        use_template=False,  # the RAM is in the saved vmstate
                    )
                except TransientError as e:
                    if (cause := refusal_cause(e)) is not None:
                        # The saved state cannot run here (QEMU refused it, aborted on it, or its
                        # guest panicked): a worker with the saver's facts and QEMU can load it.
                        raise await instances.refused_by_qemu(state, qemu_error=cause) from e
                    raise
        except BaseException:
            await instances.release(state, wait=True)
            raise
        for lease in state.asset_leases:  # the VM holds its own shares now
            lease.release()
        total_ms = round((loop.time() - start) * 1000)
        saver = state.body.host
        logger.info(
            f"Instance {instance_id} reused: generation {state.header.generation}, saved by release "
            f'"{saver.release}" with QEMU "{saver.qemu_version}", loaded by release "{__version__}" with '
            f'QEMU "{loader_qemu}" in VM {vm.vm_id}.',
            extra={
                "instance_id": instance_id,
                "generation": state.header.generation,
                "vm_id": vm.vm_id,
                "saved_release": saver.release,
                "saved_qemu_version": saver.qemu_version,
                "qemu_version": loader_qemu,
                "load_ms": load_ms,
            },
        )
        return Session(
            vm=vm,
            vm_manager=self._vms,
            spec=spec,
            startup_timing=StartupTiming(
                total_ms=total_ms,
                admission_ms=admission_ms,
                setup_ms=max(0, total_ms - load_ms - admission_ms),  # restore of the saved state
                boot_ms=0,
                warm_pool_hit=False,
                l1_cache_hit=False,
                l1_restore_ms=None,
                overlay_ms=0,
                qemu_cmd_build_ms=0,
                gvproxy_start_ms=0,
                qemu_fork_ms=0,
                guest_wait_ms=0,
                boot_retries=0,
                instance_load_ms=load_ms,
            ),
            instances=instances,
            instance=state,
            new_instance_id=None,
            idle_timeout_seconds=idle_timeout,
            default_timeout_seconds=self.config.default_timeout_seconds,
        )

    async def describe_instance(self, instance_id: str) -> InstanceInfo:
        """What the store holds for an instance (InstanceInfo: `access` says how much this
        worker could read; `spec` and `host` only when it is "full"); no VM starts.

        Raises TransientError (INSTANCE_STORE_UNAVAILABLE), InvalidInputError
        (INSTANCE_INVALID_ID, SCHEDULER_STATE: not started), ConfigError (INSTANCE_CONFIG;
        INSTANCE_KEY_MISSING when this keyring lacks the key and the public header is
        another release's), RouteError (INSTANCE_HOST_MISMATCH: the header itself is
        another release's), InstanceGoneError (INSTANCE_NOT_FOUND, INSTANCE_INTEGRITY:
        the head failed authentication).
        """
        return await self._require_instances(check_instance_id(instance_id)).describe(instance_id)

    async def describe_worker(self) -> WorkerFacts:
        """This worker's facts, one value per RouteKey in the value forms of RouteError.required
        (host facts, the clock counter rate of its guests, QEMU version and machine types, the
        guest images it holds, its whole admission budget), and its instance key ids. A broker
        reads them once per worker, and again after a change of QEMU, guest images or budget,
        and sends a call that raised RouteError to a worker whose facts satisfy the error's
        `required` (WorkerFacts.satisfies). The library places nothing. No VM starts; the
        probes are the cached ones of the start (run again after a change of the QEMU binary).

        Raises:
            InvalidInputError: SCHEDULER_STATE: not started
            ConfigError: HOST_DEPENDENCY, or TransientError: QEMU_CRASHED: a QEMU probe that
                runs again after a change of the QEMU binary fails (detect_accel_type,
                require_qemu_version, require_qemu_machine_types, and on x86 KVM
                require_guest_cpu through counter_rate)
        """
        if not self._started or self._vm_manager is None:
            raise _scheduler_state(_NOT_STARTED)
        vm_manager, instances = self._vm_manager, self._instances
        force_emulation = vm_manager.settings.force_emulation
        facts = {
            **await machine_facts(force_emulation=force_emulation),
            RouteKey.COUNTER_HZ: await counter_fact(force_emulation=force_emulation),
        }
        own = await asyncio.to_thread(_image_digests, vm_manager.images.present_base_images())
        # The L2 images of this worker's cache: a load uses one whose digest matches
        layers = self._snapshot_manager
        own_layers = (
            frozenset[str]()
            if layers is None
            else await asyncio.to_thread(lambda: _image_digests(layers.entry_paths()))
        )
        downloaded = frozenset[str]() if instances is None else await asyncio.to_thread(instances.asset_digests)
        memory_mb, cpu = vm_manager.budget_facts()
        return WorkerFacts(
            values={
                **{key: value for key, value in facts.items() if value is not None},
                RouteKey.QEMU_VERSION: await qemu_version_fact(),
                RouteKey.MACHINE_TYPE: frozenset(await require_qemu_machine_types()),
                RouteKey.BASE_IMAGE: own | downloaded,
                RouteKey.PACKAGES_IMAGE: own_layers | downloaded,
                RouteKey.MEMORY_MB: None if memory_mb is None else str(round(memory_mb)),
                RouteKey.CPU: None if cpu is None else f"{cpu:.2f}",
            },
            key_ids=() if instances is None else tuple(key.id for key in instances.keyring.keys),
        )

    async def delete_instance(self, instance_id: str) -> bool:
        """Delete a saved instance; return True if it existed, False if not (a second
        call is safe). The head goes first (gone and unreadable at once), then the
        stored objects. An open session of it fails at its next save() with
        InstanceGoneError (INSTANCE_NOT_FOUND).
        Needs read, list and delete access to the store (S3: s3:GetObject and
        s3:DeleteObject on the instance prefix, s3:ListBucket on the bucket). If it fails
        after the head is gone, a retry returns False and still removes the leftovers.

        Raises TransientError (INSTANCE_STORE_UNAVAILABLE), InvalidInputError
        (INSTANCE_INVALID_ID, SCHEDULER_STATE: not started), ConfigError (INSTANCE_CONFIG:
        no instance store, or the store refuses this worker).
        """
        return await self._require_instances(check_instance_id(instance_id)).delete(instance_id)

    async def list_instances(self) -> AsyncGenerator[str]:
        """Ids that have objects in the store, in store order:
        `async for iid in scheduler.list_instances()`.

        An id can have no head (a first save that did not finish, or a delete that
        did not finish): describe_instance then raises InstanceGoneError (INSTANCE_NOT_FOUND). Errors
        (not started, no store) come at the first iteration, not at the call.
        Needs list access to the store (S3: s3:ListBucket).
        """
        async for instance_id in self._require_instances(None).list_ids():
            yield instance_id

    def _idle_timeout(self, idle_timeout_seconds: int | None) -> int:
        """The idle timeout of a new session: the argument, or the config's when None.
        InvalidInputError ARGUMENT_INVALID outside the config field's range (0 included)."""
        if idle_timeout_seconds is None:
            return self.config.session_idle_timeout_seconds
        if not MIN_SESSION_IDLE_TIMEOUT_SECONDS <= idle_timeout_seconds <= MAX_SESSION_IDLE_TIMEOUT_SECONDS:
            raise InvalidInputError(
                f"idle_timeout_seconds must be between {MIN_SESSION_IDLE_TIMEOUT_SECONDS} and "
                f"{MAX_SESSION_IDLE_TIMEOUT_SECONDS}, got {idle_timeout_seconds}",
                code=ErrorCode.ARGUMENT_INVALID,
                context=None,
                hint=None,
                instance_id=None,
            )
        return idle_timeout_seconds

    def _require_instances(self, instance_id: str | None) -> InstanceManager:
        """The instance manager. InvalidInputError SCHEDULER_STATE before the start (checked
        first), ConfigError INSTANCE_CONFIG without an instance store."""
        if not self._started:
            raise _scheduler_state(_NOT_STARTED)
        if self._instances is None:
            raise no_instance_store(instance_id=instance_id)
        return self._instances

    async def _prepare_vm(
        self,
        spec: VmSpec,
        *,
        task_id: str,
        on_boot_log: Callable[[str], None] | None,
    ) -> tuple[QemuVM, bool, int]:
        """Create and boot a VM made from `spec`: warm pool, L1 restore or cold boot.

        Args:
            spec: What the VM is made of.
            task_id: Identifier for this VM (e.g. "session-...").
            on_boot_log: Optional callback for streaming boot console output.

        Returns:
            Tuple of (vm, is_cold_boot, admission_ms); is_cold_boot is False only for a warm
            pool VM (an L1 restore is True). The VM holds its forwarded ports
            (QemuVM.exposed_ports).

        Raises:
            InvalidInputError: SCHEDULER_STATE (scheduler not started), PACKAGE_NOT_ALLOWED
                (package not in allowlist), ARGUMENT_INVALID (a package spec without a version,
                packages for a language that has none).
            SandboxError: VM creation failed (TransientError, ConfigError, RouteError,
                InternalError).
        """
        if not self._started:
            raise _scheduler_state(_NOT_STARTED)

        # Resolve port mappings (allocate ephemeral ports for those without explicit external;
        # resolve_port_mappings logs them)
        resolved_ports: list[ExposedPort] = []
        if spec.expose_ports:
            resolved_ports = resolve_port_mappings(list(spec.expose_ports))

        # Validate packages against allowlist. The bundled catalogs never change: parsed on
        # first use, then kept. Two first requests at the same instant may both parse them
        # (same result).
        if spec.packages and self.config.enable_package_validation:
            if self._package_validator is None:
                self._package_validator = await PackageValidator.create(
                    pypi_allow_list_path=BUNDLED_PYPI_ALLOW_LIST, npm_allow_list_path=BUNDLED_NPM_ALLOW_LIST
                )
            self._package_validator.validate(list(spec.packages), spec.language)

        # Check L2 snapshot cache for disk caching (only if packages specified)
        # Only use snapshots when packages need to be installed: without packages the VM runs
        # on the base image (its overlay is a clone of the template VmManager.start made)
        # The L2 layer is leased from its lookup on: admission (below) may evict cache
        # entries for disk space, never this one. Each VM launch takes its own share.
        snapshot: EntryLease | None = None
        if spec.packages:
            # The L2 cache key hashes the base image: it must be on disk first
            if self.config.auto_download_assets:
                await ensure_base_image(spec.language, images_dir=self._vms.images.images_dir)
            # L2 (local qcow2), then L3 (S3), else a build: DiskSnapshotManager.get_or_create_snapshot
            snapshot = await self._snapshots.get_or_create_snapshot(
                language=spec.language,
                packages=list(spec.packages),
                tenant_id=constants.SCHEDULER_TENANT_ID,
                task_id=f"snapshot-{hash(tuple(sorted(spec.packages)))}",
                memory_mb=spec.memory_mb,
            )
        try:
            return await self._launch_for(
                spec, task_id=task_id, on_boot_log=on_boot_log, snapshot=snapshot, resolved_ports=resolved_ports
            )
        finally:
            if snapshot is not None:
                snapshot.release()

    async def _launch_for(
        self,
        spec: VmSpec,
        *,
        task_id: str,
        on_boot_log: Callable[[str], None] | None,
        snapshot: EntryLease | None,
        resolved_ports: list[ExposedPort],
    ) -> tuple[QemuVM, bool, int]:
        """_prepare_vm after the L2 lookup: warm pool, L1 restore or cold boot on `snapshot`."""
        snapshot_path = snapshot.path if snapshot is not None else None
        if snapshot_path is not None:
            logger.info(
                "Snapshot ready",
                extra={"snapshot_path": str(snapshot_path)},
            )

        # Try warm pool first (instant allocation). The pool serves only its own
        # spec (VmSpec defaults: no packages, no network); skip it when a boot
        # log is requested (pool VMs booted without on_boot_log).
        vm: QemuVM | None = None
        is_cold_boot = False
        admission_ms = 0
        if self._warm_pool and on_boot_log is None:
            vm = await self._warm_pool.get_vm(spec)

        if vm is None:
            is_cold_boot = True

            # Base image must be on disk before the L1 cache key hashes it
            # (compute_cache_key -> VmManager.images.base_image) and before cold create_vm.
            # Skip when snapshot_path already fetched it.
            if self.config.auto_download_assets and not snapshot_path:
                await ensure_base_image(spec.language, images_dir=self._vms.images.images_dir)

            # L1 key, once per request (it hashes the base image, kernel and L2 layer).
            # It includes the network device (spec.has_network_device):
            # expose_ports also adds a virtio-net device.
            memory = self._memory_snapshot_manager
            l1_key = await memory.compute_cache_key(spec, l2_path=snapshot_path) if memory else None
            # The L1 entry is leased before admission, like the L2 layer: admission may evict
            # cache entries for disk space, never these. Skipped when on_boot_log is requested
            # (L1 skips boot, the callback would never fire).
            l1_entry = await memory.lease(l1_key) if memory and l1_key and on_boot_log is None else None
            try:
                vm, admission_ms = await self._admit_and_launch(
                    spec,
                    task_id=task_id,
                    on_boot_log=on_boot_log,
                    snapshot=snapshot,
                    l1_key=l1_key,
                    l1_entry=l1_entry,
                    resolved_ports=resolved_ports,
                )
            finally:
                if l1_entry is not None:  # restore() releases it; this covers a failed admission
                    l1_entry.release()

        return vm, is_cold_boot, admission_ms

    async def _admit_and_launch(
        self,
        spec: VmSpec,
        *,
        task_id: str,
        on_boot_log: Callable[[str], None] | None,
        snapshot: EntryLease | None,
        l1_key: str | None,
        l1_entry: EntryLease | None,
        resolved_ports: list[ExposedPort],
    ) -> tuple[QemuVM, int]:
        """_launch_for after a warm pool miss: admission, then an L1 restore of `l1_entry`
        (taken over by restore) or a cold boot on `snapshot`. Returns the VM and the
        admission time in ms."""
        memory = self._memory_snapshot_manager
        snapshot_path = snapshot.path if snapshot is not None else None
        held = frozenset(lease.path for lease in (snapshot, l1_entry) if lease is not None)
        admission_start = asyncio.get_running_loop().time()
        async with self._vms.reservation_context(
            spec, vm_id=f"{constants.SCHEDULER_TENANT_ID}-{task_id}", held=held
        ) as reservation:
            admission_ms = round((asyncio.get_running_loop().time() - admission_start) * 1000)
            vm: QemuVM | None = None
            # L1 memory snapshot restore (~50-60 ms)
            if memory and l1_entry is not None:
                vm = await memory.restore(
                    spec,
                    entry=l1_entry,
                    tenant_id=constants.SCHEDULER_TENANT_ID,
                    task_id=task_id,
                    snapshot_drive=snapshot,
                    expose_ports=resolved_ports if resolved_ports else None,
                    reservation=reservation,
                )
                if vm is not None:
                    logger.info(
                        "VM restored from L1 cache",
                        extra={"vm_id": vm.vm_id, "language": spec.language},
                    )

            # Cold boot fallback: L1 miss (or L1 skipped for a boot log)
            if vm is None:
                vm = await self._vms.create_vm(
                    spec,
                    tenant_id=constants.SCHEDULER_TENANT_ID,
                    task_id=task_id,
                    expose_ports=resolved_ports if resolved_ports else None,
                    on_boot_log=on_boot_log,
                    snapshot_drive=snapshot_path,
                    cache_leases=(snapshot,) if snapshot is not None else (),
                    reservation=reservation,
                    direct_write_target=None,
                    retry_profile=constants.RETRY_USER_FACING,
                    mem_path=None,
                )

                # Schedule background L1 save for next time
                if memory and l1_key:
                    try:
                        await memory.schedule_background_save(spec, key=l1_key, snapshot_drive=snapshot)
                    except BaseException:
                        # The caller never receives the newly-created VM when
                        # post-acquisition scheduling is interrupted. Return
                        # ownership to VmManager before propagating the error.
                        await self._vms.destroy_vm(vm)
                        raise
        return vm, admission_ms


def _image_digests(paths: list[Path]) -> frozenset[str]:
    """SHA-256 of each readable image (image_digest: cached by the file's stamp); an image
    this worker cannot read is not one it can serve."""
    digests: set[str] = set()
    for path in paths:
        try:
            digest = image_digest(path)
        except OSError as e:
            logger.warning("Guest image unreadable", extra={"image": str(path), "error": str(e)})
            continue
        if digest is not None:  # None: the file went away since it was listed
            digests.add(digest)
    return frozenset(digests)


def _startup_timing(vm: QemuVM, *, is_cold_boot: bool, total_ms: int, admission_ms: int) -> StartupTiming:
    """Startup phases of a VM new() just prepared. Boot fields are 0 unless it cold booted."""

    def cold(value: int | None) -> int:
        return value if is_cold_boot and value is not None else 0

    boot_ms = cold(vm.timing.boot_ms)
    return StartupTiming(
        total_ms=total_ms,
        admission_ms=admission_ms,
        # Infra time only (cache, overlay, cgroup): admission wait and boot excluded
        setup_ms=max(0, total_ms - boot_ms - admission_ms),
        boot_ms=boot_ms,
        warm_pool_hit=not is_cold_boot,
        l1_cache_hit=vm.l1_restored,
        # setup_ms of an L1-restored VM is its restore time
        l1_restore_ms=vm.timing.setup_ms if vm.l1_restored else None,
        overlay_ms=cold(vm.timing.overlay_ms),
        qemu_cmd_build_ms=cold(vm.timing.qemu_cmd_build_ms),
        gvproxy_start_ms=cold(vm.timing.gvproxy_start_ms),
        qemu_fork_ms=cold(vm.timing.qemu_fork_ms),
        guest_wait_ms=cold(vm.timing.guest_wait_ms),
        boot_retries=cold(vm.timing.boot_retries),
        instance_load_ms=None,
    )


async def _packages_lease(snapshots: DiskSnapshotManager, spec: VmSpec) -> EntryLease | None:
    """A lease on this worker's L2 image of `spec`, or None: no packages, or no L2 of its own
    (a load then downloads the image by digest from the instance store)."""
    if not spec.packages:
        return None
    try:
        return await snapshots.check_cache(language=spec.language, packages=list(spec.packages))
    except ConfigError as e:
        if e.code is not ErrorCode.ASSET_NOT_FOUND:
            raise
        return None  # no base image of its own (the L2 key needs it), so no L2 here: a miss


def instance_admin(config: SchedulerConfig) -> contextlib.AbstractAsyncContextManager[InstanceAdmin]:
    """Describe, list and delete saved instances without a Scheduler: no VM, no guest image,
    no QEMU (a cleanup job on any host). The store and keyring come from `config`, then the
    EXEC_SANDBOX_* variables, as for a Scheduler. Use `async with instance_admin(config) as
    admin:`, then `admin.describe(id)` (InstanceInfo, as Scheduler.describe_instance),
    `admin.list_ids()` (async iterator, as Scheduler.list_instances) and `admin.delete(id)`
    (as Scheduler.delete_instance).

    Raises at the call: ConfigError SETTING_INVALID (an EXEC_SANDBOX_* variable),
    EXTRA_MISSING (cryptography is not installed). Raises when the block opens: ConfigError
    INSTANCE_CONFIG (no instance store, or a store or keyring that is not valid),
    EXTRA_MISSING (the s3 extra, for an S3 store), TransientError INSTANCE_STORE_UNAVAILABLE
    (a disk error of the local store). An S3 store that does not answer fails at the first
    call (TransientError INSTANCE_STORE_UNAVAILABLE).
    """
    settings = settings_from_config(config, images_dir=images_download_dir())
    return require_instance_admin(store=settings.instance_store or "local")(settings)


def settings_from_config(config: SchedulerConfig, *, images_dir: Path) -> Settings:
    """Bridge the public SchedulerConfig to internal Settings.

    Only the fields the caller set on SchedulerConfig are passed: for every shared
    field, a value set in code wins, else the EXEC_SANDBOX_* env var, else the default.
    The config is valid already, so what Settings refuses comes from the variables:
    ConfigError SETTING_INVALID (_invalid_variables).
    """
    bridged = SchedulerConfig.model_fields.keys() & Settings.model_fields.keys()
    try:
        return Settings(
            base_images_dir=images_dir,
            kernel_path=kernel_dir_of(images_dir, arch=detect_host_arch()),  # kernels/ when it has the files
            base_images_download_dir=images_download_dir(),
            **config.model_dump(include=bridged, exclude_unset=True),
        )
    except ValidationError as e:
        raise _invalid_variables(e) from None


def _invalid_variables(error: ValidationError) -> ConfigError:
    """ConfigError SETTING_INVALID for the EXEC_SANDBOX_* variables Settings refused: each
    variable's name and why. Never a value: a variable can hold a secret (instance_keyring).
    A rule across fields (overlapping S3 prefixes) has no variable: its line says why only."""
    prefix = str(Settings.model_config.get("env_prefix", ""))
    problems = [
        (f"{prefix}{'__'.join(map(str, err['loc'])).upper()}" if err["loc"] else None, err["msg"])
        for err in error.errors(include_url=False)
    ]
    return ConfigError(
        f"Invalid EXEC_SANDBOX_* variables: {'; '.join(f'{name}: {msg}' if name else msg for name, msg in problems)}.",
        code=ErrorCode.SETTING_INVALID,
        context={"variables": [name for name, _ in problems if name]},
        hint="Correct or unset these variables, then start again.",
        instance_id=None,
    )


async def _cleanup_step(step: Awaitable[object]) -> BaseException | None:
    """Await one shutdown step and return its error: later steps must still run."""
    try:
        await step
    except BaseException as e:  # noqa: BLE001 - the caller raises the first error after all steps
        return e
    return None

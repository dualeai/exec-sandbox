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
from exec_sandbox._imports import require_instance_manager
from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import await_settled
from exec_sandbox.asset_downloader import get_current_arch
from exec_sandbox.assets import ensure_assets, ensure_base_image, images_download_dir
from exec_sandbox.config import MAX_SESSION_IDLE_TIMEOUT_SECONDS, MIN_SESSION_IDLE_TIMEOUT_SECONDS, SchedulerConfig
from exec_sandbox.disk_snapshot_manager import DiskSnapshotManager
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InternalError,
    InvalidInputError,
    TransientError,
    host_mismatch,
    no_instance_store,
)
from exec_sandbox.files.locks import EntryLease  # noqa: TC001 - Used at runtime
from exec_sandbox.images import kernel_dir_of
from exec_sandbox.instance.ids import check_instance_id
from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager
from exec_sandbox.migration_client import rejected_by_qemu
from exec_sandbox.models import ExposedPort, Language, StartupTiming
from exec_sandbox.package_validator import PackageValidator
from exec_sandbox.platform_utils import arch_of
from exec_sandbox.port_forward import resolve_port_mappings
from exec_sandbox.session import Session
from exec_sandbox.settings import Settings
from exec_sandbox.system_probes import (
    raise_fd_limit,
    raise_nproc_limit,
    warn_cgroup_pids_limit,
    warn_damon,
    warn_max_map_count,
    warn_overcommit,
    warn_threads_max,
    warn_zswap,
)
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.warm_vm_pool import WarmVMPool

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from exec_sandbox.instance.manager import InstanceManager, InstanceState
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


async def _wait_for_owned_task(task: asyncio.Task[None]) -> BaseException | None:
    """Wait through repeated caller cancellation and return an owned task error.

    Deliberate trade-off: caller cancellation (including Ctrl-C) is absorbed
    until the owned task finishes, so a wedged startup rollback cannot be
    aborted — leaked VM processes would be worse than a slow interrupt.
    """
    await await_settled(task)
    if task.cancelled():
        return asyncio.CancelledError()
    return task.exception()


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
        self._settings: Settings | None = None
        self._vm_manager: VmManager | None = None
        self._snapshot_manager: DiskSnapshotManager | None = None
        self._memory_snapshot_manager: MemorySnapshotManager | None = None
        self._warm_pool: WarmVMPool | None = None
        self._package_validator: PackageValidator | None = None
        self._started = False
        self._shutdown_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        """Start the Scheduler.

        Checks the host (open files, processes; warns about unsafe kernel settings)
        and QEMU; downloads missing VM images from GitHub Releases (unless
        auto_download_assets=False); opens the instance store (with instance_store:
        keyring, settings, QEMU version, store check) and the snapshot caches; starts
        the warm pool (wait_pool_ready() waits until it is full).

        Raises:
            TransientError: ASSET_DOWNLOAD (a VM image download failed),
                INSTANCE_STORE_UNAVAILABLE (the instance store check got no answer)
            InvalidInputError: SCHEDULER_STATE: already started, or still shutting down
            ConfigError: ASSET_NOT_FOUND (a VM image is missing and was not downloaded),
                EXTRA_MISSING (instance_store set without the extra it needs),
                INSTANCE_CONFIG (the instance store check failed), MIGRATION_UNSUPPORTED
                (instance_store set and QEMU below 11.0), HOST_DEPENDENCY,
                SETTING_INVALID (an EXEC_SANDBOX_* variable is not valid: the message names it, never
                its value; EXEC_SANDBOX_HOME_DISK_DIR is not a folder of this user that only it can write;
                or a cache folder, disk_snapshot_cache_dir or memory_snapshot_cache_dir, cannot be made)
        """
        if self._started:
            raise _scheduler_state("Scheduler already started")
        if self._shutdown_task is not None:
            if not self._shutdown_task.done():
                raise _scheduler_state("Scheduler is still shutting down")
            # A completed lifecycle may be reused only after its shutdown was
            # successful. result() preserves a prior cleanup failure.
            self._shutdown_task.result()
            self._shutdown_task = None

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
        self._settings = settings_from_config(self.config, images_dir=images_dir)

        logger.info(
            "Starting scheduler",
            extra={
                "warm_pool_size": self.config.warm_pool_size,
                "s3_enabled": self._settings.s3_bucket is not None,
                "auto_download_assets": self.config.auto_download_assets,
            },
        )

        # Initialize VmManager (handles backpressure internally via admission controller).
        # From this publication onward, every startup failure rolls the whole
        # partial lifecycle back through one scheduler-owned shutdown task.
        self._vm_manager = VmManager(self._settings)
        try:
            await self._vm_manager.start()  # Pre-warms all system probe caches

            # Saved instances: keyring, store settings, QEMU floor and store probe fail the start (fail fast)
            if self._settings.instance_store is not None:
                self._instances_exit = contextlib.AsyncExitStack()
                manager = require_instance_manager(store=self._settings.instance_store)
                self._instances = await self._instances_exit.enter_async_context(
                    manager.open(
                        self._settings, work_dir=self._vm_manager.work_dir / "instances", vm_manager=self._vm_manager
                    )
                )

            # Initialize DiskSnapshotManager (L2 local cache always available, L3 S3 optional)
            self._snapshot_manager = DiskSnapshotManager(self._settings, self._vm_manager)
            await self._snapshot_manager.start()  # removes .part files of killed processes

            # Initialize MemorySnapshotManager (L1 cache)
            self._memory_snapshot_manager = MemorySnapshotManager(self._settings, self._vm_manager)

            # Admission evicts L2 and L1 entries before a VM waits for home disk space
            self._vm_manager.add_cache(self._snapshot_manager)
            self._vm_manager.add_cache(self._memory_snapshot_manager)

            # Initialize WarmVMPool (optional)
            if self.config.warm_pool_size > 0:
                # The pool pre-boots warm_pool_size VMs per language, and its startup
                # hashes each base image (compute_cache_key -> VmManager.images.base_image).
                # On a fresh cache ensure_assets() fetched only kernel/initramfs, so
                # download every pooled base image first, else start() raises
                # ConfigError (ASSET_NOT_FOUND). Mirrors the per-run fetch in _launch_for
                # and _get_or_create_snapshot.
                if self.config.auto_download_assets:
                    await asyncio.gather(*(ensure_base_image(language, images_dir=images_dir) for language in Language))
                self._warm_pool = WarmVMPool(
                    self._vm_manager,
                    self.config,
                    self._snapshot_manager,
                    memory_snapshot_manager=self._memory_snapshot_manager,
                )
                await self._warm_pool.start()
        except BaseException:
            self._shutdown_task = asyncio.create_task(self._shutdown())
            rollback_error = await _wait_for_owned_task(self._shutdown_task)
            if rollback_error is not None:
                logger.error(
                    "Scheduler startup rollback failed",
                    extra={"error": str(rollback_error)},
                )
            raise

        self._started = True
        logger.info("Scheduler started successfully")
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Stop the Scheduler: wait for the saves in progress, destroy every VM, then close the instance store."""
        self._started = False
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self) -> None:
        """Shutdown scheduler and clean up all resources.

        Shutdown sequence:
        1. Stop WarmVMPool (drains and destroys pre-booted VMs, frees resource slots)
        2. Stop MemorySnapshotManager (awaits in-flight background L1 saves)
        3. Wait for instance saves in flight (InstanceManager.drain)
        4. Destroy any remaining active VMs
        5. Stop VmManager (overlay pool cleanup)
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
        vm_manager = self._vm_manager
        instances = self._instances
        instances_exit = self._instances_exit

        # Stop WarmVMPool (drains VMs, frees resource slots for background saves)
        if warm_pool:
            logger.info("Draining warm VM pool, this may take a moment")
            try:
                await warm_pool.stop()
            except (OSError, RuntimeError, TimeoutError) as e:
                logger.error("WarmVMPool stop error", extra={"error": str(e)})
            except BaseException as e:  # noqa: BLE001 - core VM cleanup must still run
                deferred_error = e
                logger.error("WarmVMPool stop aborted", extra={"error": str(e)})

        # Stop MemorySnapshotManager (wait for background saves — now has resource slots)
        if memory_snapshot_manager:
            logger.info("Waiting for in-flight background snapshot saves to complete")
            try:
                await memory_snapshot_manager.stop()
            except (OSError, RuntimeError, TimeoutError) as e:
                logger.error("MemorySnapshotManager stop error", extra={"error": str(e)})
            except BaseException as e:  # noqa: BLE001 - core VM cleanup must still run
                if deferred_error is None:
                    deferred_error = e
                logger.error("MemorySnapshotManager stop aborted", extra={"error": str(e)})

        # Let instance saves in flight end before their VMs and the work folder go
        if instances is not None:
            drain_error = await _cleanup_step(instances.drain())  # runs even after an earlier failed step
            deferred_error = deferred_error or drain_error

        # Destroy any remaining VMs
        if vm_manager:
            active_vms = vm_manager.get_active_vms()
            if active_vms:
                logger.warning(
                    "Destroying remaining VMs on shutdown",
                    extra={"count": len(active_vms)},
                )
                destroy_tasks = [vm_manager.destroy_vm(vm) for vm in active_vms.values()]
                await asyncio.gather(*destroy_tasks, return_exceptions=True)

            # Stop VmManager (includes overlay pool cleanup)
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
                When given, it turns on verbose boot logging (and skips the warm pool and L1).
            instance_id: Id the session's first save() gives the new instance (1 to 64
                lowercase letters, digits, ".", "_" or "-", starting with a letter or
                digit), checked before any VM starts. None: the library makes a random
                id. An id that already exists makes that save fail with
                OutcomeUnknownError (INSTANCE_CONFLICT) and close the session (nothing is
                overwritten).

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
                is configured), HOST_DEPENDENCY, SETTING_INVALID (no write access to
                disk_snapshot_cache_dir, a read-only disk, or the folder cannot be made again).
            RouteError: CAPACITY_NEVER_FITS: the spec is larger than the whole budget
                (raised at once).
            InternalError: VM creation failed (a bug or a broken host).
        """
        if instance_id is not None:
            self._require_instances(check_instance_id(instance_id))  # before any VM starts
        idle_timeout = self._idle_timeout(idle_timeout_seconds)
        loop = asyncio.get_running_loop()
        start = loop.time()
        vm, resolved_ports, is_cold_boot, admission_ms = await self._prepare_vm(
            spec, task_id=f"session-{id(self)}", on_boot_log=on_boot_log
        )
        total_ms = round((loop.time() - start) * 1000)

        # Store resolved ports on VM so Session.exposed_ports can read them
        vm.exposed_ports = resolved_ports

        # _vm_manager guaranteed non-None by _prepare_vm
        if self._vm_manager is None:
            raise _internal(_NOT_INITIALIZED)

        session = Session(
            vm=vm,
            vm_manager=self._vm_manager,
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
                key; do not delete the instance).
            RouteError: INSTANCE_HOST_MISMATCH: this worker cannot run it: arch,
                accelerator, CPU vendor, guest CPU model or exec-sandbox release differ
                from the saved ones (checked before any download), a missing guest image,
                or a QEMU that refuses the saved state.
                CAPACITY_NEVER_FITS: the spec is larger than the whole budget (raised at once).
            InstanceGoneError: INSTANCE_NOT_FOUND, INSTANCE_EXPIRED, INSTANCE_INTEGRITY
                (the stored data was changed or damaged): start again with new().
            InternalError: The VM could not start (a bug or a broken host).
        """
        instances = self._require_instances(check_instance_id(instance_id))
        idle_timeout = self._idle_timeout(idle_timeout_seconds)
        if self._vm_manager is None or self._snapshot_manager is None:
            raise _internal(_NOT_INITIALIZED)
        loop = asyncio.get_running_loop()
        start = loop.time()
        snapshots = self._snapshot_manager

        # The local L2 image is leased from its lookup on: admission (below) may evict
        # cache entries for disk space, never this one. The VM takes its own share.
        packages_lease: EntryLease | None = None

        async def packages_image(spec: VmSpec) -> Path | None:
            # A local L2 is used only if its digest matches; else load() downloads the
            # image by digest from the instance store.
            nonlocal packages_lease
            if not spec.packages:
                return None
            lease = await snapshots.check_cache(language=spec.language, packages=list(spec.packages))
            if packages_lease is not None:  # an earlier attempt of the load
                packages_lease.release()
            packages_lease = lease
            return lease.path if lease is not None else None

        try:
            state = await instances.load(instance_id, packages_image=packages_image)
            return await self._reuse_loaded(
                instances,
                instance_id,
                state=state,
                packages_lease=packages_lease,
                start=start,
                idle_timeout=idle_timeout,
            )
        finally:
            if packages_lease is not None:
                packages_lease.release()

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
        if self._vm_manager is None:
            raise _internal(_NOT_INITIALIZED)
        loop = asyncio.get_running_loop()
        load_ms = round((loop.time() - start) * 1000)
        spec = state.body.spec
        leases = (packages_lease,) if packages_lease is not None and state.packages_image == packages_lease.path else ()
        try:
            resolved_ports = resolve_port_mappings(list(spec.expose_ports)) if spec.expose_ports else []
            task_id = f"reuse-{id(state)}"
            admission_start = loop.time()
            # Kept while it waits: the loaded state's folder and the packages layer
            held = frozenset({state.folder, *(lease.path for lease in leases)})
            async with self._vm_manager.reservation_context(
                spec, vm_id=f"{constants.SCHEDULER_TENANT_ID}-{task_id}", held=held
            ) as reservation:
                admission_ms = round((loop.time() - admission_start) * 1000)
                try:
                    vm = await self._vm_manager.restore_vm(
                        spec,
                        tenant_id=constants.SCHEDULER_TENANT_ID,
                        task_id=task_id,
                        vmstate_path=state.vmstate,
                        base_image=state.base_image,
                        snapshot_drive=state.packages_image,
                        cache_leases=leases,
                        expose_ports=resolved_ports or None,
                        reservation=reservation,
                    )
                except TransientError as e:
                    if rejected_by_qemu(e):
                        # QEMU refused the saved state: a worker with the QEMU that wrote it can load it.
                        raise host_mismatch(
                            instance_id=instance_id,
                            required={"qemu_version": state.body.host.qemu_version},
                            found={"qemu_version": "this worker's QEMU refused the saved state"},
                        ) from e
                    raise
        except BaseException:
            await instances.release(state)
            raise
        vm.exposed_ports = resolved_ports
        total_ms = round((loop.time() - start) * 1000)
        logger.info(
            "Instance reused",
            extra={"instance_id": instance_id, "generation": state.header.generation, "load_ms": load_ms},
        )
        return Session(
            vm=vm,
            vm_manager=self._vm_manager,
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
        """What the store holds for an instance (InstanceInfo: `spec` and `host` only
        when this keyring has the instance's key); no VM starts.

        Raises TransientError (INSTANCE_STORE_UNAVAILABLE), InvalidInputError
        (INSTANCE_INVALID_ID, SCHEDULER_STATE: not started), ConfigError (INSTANCE_CONFIG),
        RouteError (INSTANCE_HOST_MISMATCH: another release wrote it), InstanceGoneError
        (INSTANCE_NOT_FOUND, INSTANCE_INTEGRITY: the head failed authentication).
        """
        return await self._require_instances(check_instance_id(instance_id)).describe(instance_id)

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
    ) -> tuple[QemuVM, list[ExposedPort], bool, int]:
        """Create and boot a VM made from `spec`: warm pool, L1 restore or cold boot.

        Args:
            spec: What the VM is made of.
            task_id: Identifier for this VM (e.g. "session-...").
            on_boot_log: Optional callback for streaming boot console output.

        Returns:
            Tuple of (vm, resolved_ports, is_cold_boot, admission_ms).

        Raises:
            InvalidInputError: SCHEDULER_STATE (scheduler not started), PACKAGE_NOT_ALLOWED
                (package not in allowlist), ARGUMENT_INVALID (a package spec without a version,
                packages for a language that has none).
            SandboxError: VM creation failed (TransientError, ConfigError, RouteError,
                InternalError).
        """
        if not self._started:
            raise _scheduler_state(_NOT_STARTED)
        if self._vm_manager is None or self._settings is None:
            raise _internal(_NOT_INITIALIZED)

        # Resolve port mappings (allocate ephemeral ports for those without explicit external;
        # resolve_port_mappings logs them)
        resolved_ports: list[ExposedPort] = []
        if spec.expose_ports:
            resolved_ports = resolve_port_mappings(list(spec.expose_ports))

        # Validate packages against allowlist
        if spec.packages and self.config.enable_package_validation:
            await self._validate_packages(list(spec.packages), spec.language)

        # Check L2 snapshot cache for disk caching (only if packages specified)
        # Only use snapshots when packages need to be installed - empty packages case
        # benefits from overlay pool pre-caching which requires using the original base image
        # The L2 layer is leased from its lookup on: admission (below) may evict cache
        # entries for disk space, never this one. Each VM launch takes its own share.
        snapshot: EntryLease | None = None
        if spec.packages:
            snapshot = await self._get_or_create_snapshot(spec.language, list(spec.packages), spec.memory_mb)
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
    ) -> tuple[QemuVM, list[ExposedPort], bool, int]:
        """_prepare_vm after the L2 lookup: warm pool, L1 restore or cold boot on `snapshot`."""
        if self._vm_manager is None:
            raise _internal(_NOT_INITIALIZED)
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
                await ensure_base_image(spec.language, images_dir=self._vm_manager.images.images_dir)

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

        return vm, resolved_ports, is_cold_boot, admission_ms

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
        if self._vm_manager is None:
            raise _internal(_NOT_INITIALIZED)
        memory = self._memory_snapshot_manager
        snapshot_path = snapshot.path if snapshot is not None else None
        held = frozenset(lease.path for lease in (snapshot, l1_entry) if lease is not None)
        admission_start = asyncio.get_running_loop().time()
        async with self._vm_manager.reservation_context(
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
                vm = await self._vm_manager.create_vm(
                    spec,
                    tenant_id=constants.SCHEDULER_TENANT_ID,
                    task_id=task_id,
                    expose_ports=resolved_ports if resolved_ports else None,
                    on_boot_log=on_boot_log,
                    snapshot_drive=snapshot_path,
                    cache_leases=(snapshot,) if snapshot is not None else (),
                    reservation=reservation,
                )

                # Schedule background L1 save for next time
                if memory and l1_key:
                    try:
                        await memory.schedule_background_save(spec, key=l1_key, snapshot_drive=snapshot)
                    except BaseException:
                        # The caller never receives the newly-created VM when
                        # post-acquisition scheduling is interrupted. Return
                        # ownership to VmManager before propagating the error.
                        await self._vm_manager.destroy_vm(vm)
                        raise
        return vm, admission_ms

    async def _validate_packages(self, packages: list[str], language: Language) -> None:
        """Validate packages against allowlist.

        Args:
            packages: List of package specifiers (e.g., ["pandas==3.0.1"])
            language: Programming language

        Raises:
            InvalidInputError: PACKAGE_NOT_ALLOWED: package not in allowlist; ARGUMENT_INVALID:
                a package spec without a version, packages for a language that has none
        """
        # The bundled catalogs never change: parsed on first use, then kept. Two
        # first requests at the same instant may both parse them (same result).
        if self._package_validator is None:
            self._package_validator = await PackageValidator.create()
        self._package_validator.validate(packages, language)

    async def _get_or_create_snapshot(self, language: Language, packages: list[str], memory_mb: int) -> EntryLease:
        """Get cached snapshot or create new one with packages.

        Checks L2 (local qcow2) and L3 (S3) caches before building.

        Args:
            language: Programming language
            packages: List of packages to install
            memory_mb: VM memory in MB of the VM that builds the snapshot (not in the cache key)

        Returns:
            A lease on the snapshot qcow2 (lease.path): the caller releases it.

        Raises:
            InternalError: INTERNAL: snapshot manager not configured.
            SandboxError: Snapshot creation failed (see
                DiskSnapshotManager.get_or_create_snapshot).
        """
        if not self._snapshot_manager:
            raise _internal("Packages requested but snapshot manager is not configured")

        # Ensure base image is on disk before computing snapshot cache key
        # (_compute_cache_key hashes the base image that VmManager.images names)
        if self.config.auto_download_assets:
            if self._vm_manager is None:
                raise _internal(_NOT_INITIALIZED)
            await ensure_base_image(language, images_dir=self._vm_manager.images.images_dir)

        snapshot = await self._snapshot_manager.get_or_create_snapshot(
            language=language,
            packages=packages,
            tenant_id=constants.SCHEDULER_TENANT_ID,
            task_id=f"snapshot-{hash(tuple(sorted(packages)))}",
            memory_mb=memory_mb,
        )
        logger.debug(
            "Snapshot ready",
            extra={
                "language": language,
                "packages": packages,
                "path": str(snapshot.path),
            },
        )
        return snapshot


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
            kernel_path=kernel_dir_of(images_dir, arch=arch_of(get_current_arch())),  # kernels/ when it has the files
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

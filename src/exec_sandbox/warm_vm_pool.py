"""Warm VM pool for instant code execution startup.

Pre-boots VMs at service startup for default-image executions.
Provides 200-400x faster execution start (1-2ms vs 400ms cold boot).

Architecture:
- Pool size: Configured via warm_pool_size (per language)
- Languages: all (python, javascript, raw)
- Lifecycle: Pre-boot → allocate → execute → destroy → replenish
- Security: One-time use (no cross-tenant reuse)

Performance:
- Default image (packages=[]): 1-2ms allocation (vs 400ms cold boot)
- Custom packages: Fallback to cold boot (no change)
- Memory overhead: ~140MB idle (balloon inflated) / ~192MB active per VM

L2 Disk Snapshots:
- Uses L2 cache (local qcow2) for faster warm pool boots
- snapshot_manager: Optional for L2 cache (graceful degradation to cold boot if None)

Memory Optimization (Balloon):
- Idle pool VMs have balloon inflated (guest has BALLOON_INFLATE_TARGET_MB)
- Before execution, balloon deflates (guest gets full memory back)
- Reduces idle memory per VM (140MB idle vs 192MB active)

Example:
    ```python
    # In Scheduler
    async with WarmVMPool(vm_manager, config, snapshot_manager) as warm_pool:
        # Per execution
        vm = await warm_pool.get_vm("python", packages=[])
        if vm:  # Warm hit (1-2ms)
            result = await vm.execute(...)
        else:  # Cold fallback (400ms)
            vm = await vm_manager.create_vm(...)
    ```
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, Self

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from exec_sandbox import cgroup, constants
from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import await_settled, settle_and_report
from exec_sandbox.balloon_client import BalloonClient, BalloonError
from exec_sandbox.exceptions import SocketAuthError, VmTransientError
from exec_sandbox.guest_agent_protocol import PingRequest, PongMessage, WarmReplAckMessage, WarmReplRequest
from exec_sandbox.models import Language
from exec_sandbox.permission_utils import get_expected_socket_uid
from exec_sandbox.platform_utils import advise_willneed
from exec_sandbox.vm_types import VmState

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from tenacity.wait import wait_base

    from exec_sandbox.config import SchedulerConfig
    from exec_sandbox.disk_snapshot_manager import DiskSnapshotManager
    from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.vm_manager import VmManager

logger = get_logger(__name__)

# Transient exceptions during health checks that should trigger retry/unhealthy status.
# These indicate temporary communication failures, not permanent VM problems:
# - OSError/ConnectionError/EOFError: Socket/network issues
# - TimeoutError: Guest agent slow to respond
# - SocketAuthError: SO_PEERCRED returns pid=0 when QEMU frozen (SIGSTOP) due to kernel race
_HEALTH_CHECK_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    TimeoutError,
    ConnectionError,
    EOFError,
    SocketAuthError,
)


class WarmVMPool:
    """Manages pre-booted VMs for instant execution.

    Single Responsibility: VM pool lifecycle management
    - Startup: Pre-boot VMs in parallel (non-blocking when called from main.py)
    - Allocation: Get VM from pool (no wait for availability; dead pooled
      VMs are screened out and destroyed at checkout)
    - Replenishment: Background task to maintain pool size
    - Shutdown: Drain and destroy all VMs

    Thread-safety: Uses asyncio.Queue (thread-safe for async)

    Attributes:
        vm_manager: VmManager for VM lifecycle
        config: Scheduler configuration
        pool_size_per_language: Number of VMs per language
        pools: Dict[language, Queue[QemuVM]] for each language
    """

    def __init__(
        self,
        vm_manager: VmManager,
        config: SchedulerConfig,
        snapshot_manager: DiskSnapshotManager | None = None,
        memory_snapshot_manager: MemorySnapshotManager | None = None,
    ):
        """Initialize warm VM pool.

        Args:
            vm_manager: VmManager for VM lifecycle
            config: Scheduler configuration
            snapshot_manager: Optional DiskSnapshotManager for L2 cache (faster refill)
            memory_snapshot_manager: Optional MemorySnapshotManager for L1 cache
        """
        self.vm_manager = vm_manager
        self.config = config
        self.snapshot_manager = snapshot_manager
        self.memory_snapshot_manager = memory_snapshot_manager

        # Pool size is explicitly configured via warm_pool_size
        # warm_pool_size=0 means warm pool disabled (caller should not create WarmVMPool)
        self.pool_size_per_language = config.warm_pool_size

        # Pools: asyncio.Queue for thread-safe async access
        self.pools: dict[Language, asyncio.Queue[QemuVM]] = {
            lang: asyncio.Queue(maxsize=self.pool_size_per_language) for lang in Language
        }

        # Track background replenish tasks (prevent GC)
        self._replenish_tasks: set[asyncio.Task[None]] = set()
        self._initial_boot_tasks: set[asyncio.Task[None]] = set()

        # Semaphore that throttles concurrent replenishment boots per language,
        # up to ~50% of pool_size for faster catch-up under load. It bounds boot
        # concurrency; it does NOT make the pool.full() check atomic with boot.
        # With a count > 1, two tasks can both pass full() at qsize == maxsize-1
        # and both boot; the loser then blocks in pool.put() holding a live VM
        # until a checkout frees a slot (transient over-boot, self-correcting).
        self._replenish_max_concurrent = max(
            1,  # Minimum 1 concurrent boot
            int(self.pool_size_per_language * constants.WARM_POOL_REPLENISH_CONCURRENCY_RATIO),
        )
        self._replenish_semaphores: dict[Language, asyncio.Semaphore] = {
            lang: asyncio.Semaphore(self._replenish_max_concurrent) for lang in Language
        }

        # Health check task
        self._health_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._start_task: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Task[None] | None = None

        logger.info(
            "Warm VM pool initialized",
            extra={
                "pool_size_per_language": self.pool_size_per_language,
                "languages": [lang.value for lang in Language],
                "total_vms": self.pool_size_per_language * len(Language),
            },
        )

    async def start(self) -> None:
        """Start the pool and roll back every partially-created VM on failure."""
        if self._stop_task is not None:
            if not self._stop_task.done():
                raise RuntimeError("Cannot start warm VM pool while shutdown is in progress")
            self._stop_task.result()
            self._stop_task = None
            self._start_task = None

        if self._start_task is None:
            self._shutdown_event.clear()
            self._start_task = asyncio.create_task(self._start_impl())
        start_task = self._start_task
        try:
            await asyncio.shield(start_task)
        except BaseException:
            rollback = asyncio.create_task(self.stop())
            if (rollback_error := await settle_and_report(rollback)) is not None:
                logger.error(
                    "Warm pool startup rollback failed",
                    extra={"error": str(rollback_error)},
                )
            raise

    async def _start_impl(self) -> None:
        """Start the warm VM pool by pre-booting VMs (parallel).

        Boots all VMs in parallel for faster startup.
        Logs progress for operational visibility.

        Raises:
            VmTransientError: If critical number of VMs fail to boot
        """
        logger.info(
            "Starting warm VM pool",
            extra={"total_vms": self.pool_size_per_language * len(Language)},
        )
        boot_start = asyncio.get_running_loop().time()

        # Pre-warm vmstate files into kernel page cache so concurrent L1
        # restores hit RAM instead of storage.  advise_willneed() issues an
        # async, zero-copy kernel readahead (F_RDADVISE on macOS,
        # posix_fadvise WILLNEED on Linux) and returns near-instantly; the
        # 15ms boot stagger below gives the kernel time to complete the I/O.
        if self.memory_snapshot_manager:
            for language in Language:
                vmstate = await self.memory_snapshot_manager.check_cache(language, [], constants.DEFAULT_MEMORY_MB)
                if vmstate:
                    advise_willneed(vmstate)

        # Build list of all VMs to boot, staggered by 15ms to avoid macOS dyld
        # contention.  Each QEMU fork+exec resolves ~170 dylibs under a global
        # dyld lock; overlapping spawns serialize on that lock and degrade from
        # ~150ms to 250-400ms per VM.  15ms ≈ single-process dyld resolve time,
        # so the next spawn starts just as the previous one releases the lock.
        boot_coroutines: list[Coroutine[Any, Any, None]] = []
        boot_order = 0
        for language in Language:
            logger.info(f"Pre-booting {self.pool_size_per_language} {language.value} VMs (parallel)")
            for i in range(self.pool_size_per_language):
                boot_coroutines.append(self._boot_and_add_vm(language, index=i, boot_delay=boot_order * 0.015))
                boot_order += 1

        # Publish every initial boot task before awaiting. Shutdown owns this set
        # separately from the parent start task because a child may need to
        # finish cancellation cleanup after gather itself is cancelled.
        boot_tasks = [asyncio.create_task(coroutine) for coroutine in boot_coroutines]
        self._initial_boot_tasks.update(boot_tasks)
        for task in boot_tasks:
            task.add_done_callback(self._initial_boot_tasks.discard)

        # Boot all VMs in parallel
        results: list[None | BaseException] = await asyncio.gather(*boot_tasks, return_exceptions=True)

        # Log failures (graceful degradation)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Failed to boot warm VM",
                    extra={"task_index": i, "error": str(result)},
                    exc_info=result,
                )

        boot_duration = asyncio.get_running_loop().time() - boot_start

        # Start health check background task
        self._health_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            "Warm VM pool startup complete",
            extra={
                "boot_duration_s": f"{boot_duration:.2f}",
                **{f"{lang.value}_vms": self.pools[lang].qsize() for lang in Language},
            },
        )

    async def wait_until_ready(self, timeout: float = 120) -> None:
        """Wait until all language pools are fully populated.

        Polls pool sizes every 200ms until all queues report full().
        Returns on timeout with a warning if pools never fill (e.g. boot failures).

        Args:
            timeout: Maximum seconds to wait before returning.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if all(pool.full() for pool in self.pools.values()):
                return
            await asyncio.sleep(0.2)
        logger.warning(
            "wait_until_ready timed out",
            extra={
                "timeout": timeout,
                **{
                    f"{lang.value}_pool": f"{self.pools[lang].qsize()}/{self.pool_size_per_language}"
                    for lang in Language
                },
            },
        )

    async def get_vm(
        self,
        language: Language,
        packages: list[str],
    ) -> QemuVM | None:
        """Get warm VM if eligible (no wait for availability).

        Eligibility: packages=[] (default image only)
        Graceful degradation: Pool empty → return None (cold boot fallback)

        Side-effect: Triggers background replenishment

        Args:
            language: Programming language enum
            packages: Package list (must be empty for warm pool)

        Returns:
            Warm VM if available, None otherwise
        """
        # Only serve default-image executions
        if packages:
            logger.debug("Warm pool ineligible (custom packages)", extra={"language": language.value})
            return None

        vm: QemuVM | None = None
        try:
            # Non-blocking get (raises QueueEmpty if pool exhausted).
            # Skip VMs whose QEMU died while pooled: the process-exit watcher
            # flips state to DESTROYING, and L1-restored VMs (no watcher)
            # expose death via process.returncode. Handing one out fails the
            # caller (VmPermanentError from the execute state gate, or a
            # connect failure for the returncode-only case) instead of
            # cold-boot fallback; the health check only evicts every 15s.
            while True:
                vm = self.pools[language].get_nowait()
                if vm.state is VmState.READY and vm.process.returncode is None:
                    break
                logger.warning(
                    "Discarding dead warm VM at checkout",
                    extra={
                        "language": language.value,
                        "vm_id": vm.vm_id,
                        "vm_state": vm.state.value,
                        "returncode": vm.process.returncode,
                    },
                )
                self._schedule_replenishment(language)
                dead_vm = vm
                vm = None
                try:
                    await self.vm_manager.destroy_vm(dead_vm)
                except Exception:
                    logger.exception(
                        "Failed to destroy dead warm VM at checkout",
                        extra={"language": language.value, "vm_id": dead_vm.vm_id},
                    )

            # Deflate balloon to restore memory before code execution.
            # Skip for L1-restored VMs: no balloon device (COW file-backed memory).
            if not vm.l1_restored:
                await self._deflate_balloon(vm)

            logger.debug(
                "Warm VM allocated",
                extra={
                    "debug_category": "lifecycle",
                    "language": language.value,
                    "vm_id": vm.vm_id,
                    "pool_remaining": self.pools[language].qsize(),
                },
            )

            # Trigger background replenishment (fire-and-forget)
            self._schedule_replenishment(language)

            return vm

        except asyncio.QueueEmpty:
            logger.warning(
                "Warm pool exhausted (cold boot fallback)",
                extra={"language": language.value, "pool_size": self.pool_size_per_language},
            )
            return None

        except BaseException:
            # Ownership transferred out of the queue before the first await.
            # If checkout cannot complete, destroy that VM and replenish rather
            # than leaving a live, unreachable owner outside both pool and caller.
            if vm is None:
                raise
            self._schedule_replenishment(language)
            try:
                await self.vm_manager.destroy_vm(vm)
            except BaseException:
                logger.exception(
                    "Failed to destroy warm VM after interrupted checkout",
                    extra={"language": language.value, "vm_id": vm.vm_id},
                )
            raise

    def _schedule_replenishment(self, language: Language) -> None:
        """Publish one pool-owned replenish task without a cancellation point."""
        if self._shutdown_event.is_set():
            return
        replenish_task = asyncio.create_task(self._replenish_pool(language))
        self._replenish_tasks.add(replenish_task)
        replenish_task.add_done_callback(self._replenish_tasks.discard)

    async def stop(self) -> None:
        """Stop the warm VM pool: drain and destroy all VMs.

        The shared shutdown task owns teardown to completion. Cancelling any
        individual waiter delays cancellation propagation until every queued VM
        and replenishment task has been handled.
        """
        self._shutdown_event.set()
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop_impl())
        stop_task = self._stop_task
        cancellation = await await_settled(stop_task)

        if cancellation is not None:
            if not stop_task.cancelled() and (error := stop_task.exception()) is not None:
                logger.error("Warm VM pool shutdown failed after waiter cancellation", exc_info=error)
            raise asyncio.CancelledError
        await stop_task

    async def _stop_initial_boots(self) -> None:
        """Cancel and await every owner that can still enqueue an initial VM."""
        start_task = self._start_task
        if start_task is not None and not start_task.done():
            start_task.cancel()
        initial_boot_tasks = list(self._initial_boot_tasks)
        for task in initial_boot_tasks:
            if not task.done():
                task.cancel()
        startup_tasks = [task for task in (start_task, *initial_boot_tasks) if task is not None]
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

    async def _stop_health_check(self) -> None:
        """Stop the health owner without letting its failure abort pool drain."""
        if self._health_task is None:
            return
        try:
            await asyncio.wait_for(
                self._health_task,
                timeout=constants.WARM_POOL_HEALTH_CHECK_INTERVAL + 2.0,
            )
        except asyncio.CancelledError:
            if not self._health_task.cancelled():
                raise
            logger.debug("Health check task was already cancelled during shutdown")
        except TimeoutError:
            logger.warning("Health check task timed out during shutdown, cancelling")
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task
        except Exception as error:
            logger.exception(
                "Health check task failed during shutdown; continuing teardown",
                exc_info=error,
            )

    async def _stop_impl(self) -> None:
        """Owned warm-pool shutdown implementation.

        Stop sequence:
        1. Signal health check to stop
        2. Wait for health check task
        3. Cancel and await pending replenish tasks
        4. Drain all pools and destroy VMs (parallel)
        """
        logger.info("Shutting down warm VM pool")

        # Initial-fill children are owned by the shared start task. Cancel and
        # await that task before taking the final queue snapshot so no boot can
        # enqueue after a successful stop.
        await self._stop_initial_boots()

        # Stop health check with timeout to prevent indefinite wait
        # Timeout must be > health check interval (15s) to allow current iteration to complete
        await self._stop_health_check()

        # Close replenish ingress before draining. Otherwise a boot can enqueue
        # behind the drain snapshot and survive a successful stop().
        tasks_to_cancel = list(self._replenish_tasks)
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        # No explicit clear: each task's done-callback (add_done_callback at
        # spawn) already discards itself, and those callbacks run before gather
        # resolves, so the set is empty here.

        # Drain and destroy all VMs in parallel
        destroy_tasks: list[asyncio.Task[bool]] = []
        destroyed_count = 0
        for language, pool in self.pools.items():
            while not pool.empty():
                try:
                    vm = pool.get_nowait()
                    # Spawn parallel destruction task
                    destroy_tasks.append(asyncio.create_task(self._destroy_vm_with_logging(vm, language)))
                except asyncio.QueueEmpty:
                    break

        # Wait for all destructions to complete
        if destroy_tasks:
            results: list[bool | BaseException] = await asyncio.gather(*destroy_tasks, return_exceptions=True)
            destroyed_count = sum(1 for r in results if r is True)

        logger.info("Warm VM pool shutdown complete", extra={"destroyed_vms": destroyed_count})

    async def _destroy_vm_with_logging(
        self,
        vm: QemuVM,
        language: Language,
    ) -> bool:
        """Destroy VM with logging (helper for parallel shutdown).

        Args:
            vm: VM to destroy
            language: Programming language (for logging)

        Returns:
            True if destroyed successfully, False otherwise
        """
        try:
            cleanup_confirmed = await self.vm_manager.destroy_vm(vm)
            if cleanup_confirmed:
                logger.debug("Warm VM destroyed", extra={"language": language.value, "vm_id": vm.vm_id})
            else:
                logger.error(
                    "Warm VM cleanup left retryable resources unconfirmed",
                    extra={"language": language.value, "vm_id": vm.vm_id},
                )
            return cleanup_confirmed
        except Exception as e:
            logger.error(
                "Failed to destroy warm VM",
                extra={"language": language.value, "error": str(e)},
                exc_info=True,
            )
            return False

    async def _boot_and_add_vm(
        self,
        language: Language,
        index: int,
        boot_delay: float = 0,
    ) -> None:
        """Boot VM and add to pool (used for parallel startup).

        For index=0: If L1 cache is empty, saves L1 from the first warmed VM
        (sacrificial save — VM is killed, then replaced from L1 cache).

        Args:
            language: Programming language enum
            index: VM index in pool (for unique ID)
            boot_delay: Seconds to wait before booting (staggers fork/exec)
        """
        if boot_delay > 0:
            await asyncio.sleep(boot_delay)
        vm: QemuVM | None = None
        try:
            vm = await self._boot_warm_vm(language, index)

            # L1-restored VMs already have warm REPL — skip warm_repl
            if not vm.l1_restored:
                # Pre-warm REPL at full memory (hides ~10s Python/Bun startup on HVF).
                await self._warm_repl(vm, language)

                # index=0 only: save L1 for future restores (one-time sacrificial save)
                if self.memory_snapshot_manager and index == 0:
                    l1_hit = await self.memory_snapshot_manager.check_cache(
                        language,
                        [],
                        constants.DEFAULT_MEMORY_MB,
                    )
                    if not l1_hit:
                        saved = await self.memory_snapshot_manager.save_snapshot(
                            vm,
                            language,
                            [],
                            constants.DEFAULT_MEMORY_MB,
                        )
                        if saved:
                            # VM is dead after save — boot replacement (now from L1 cache)
                            with contextlib.suppress(Exception):
                                await self.vm_manager.destroy_vm(vm)
                            vm = await self._boot_warm_vm(language, index)
                            if not vm.l1_restored:
                                await self._warm_repl(vm, language)

            await self._prepare_and_enqueue_vm(vm, language, repl_is_ready=True)
            logger.info(
                "Warm VM ready",
                extra={
                    "language": language.value,
                    "vm_id": vm.vm_id,
                    "index": index,
                    "total": self.pool_size_per_language,
                    "l1_restored": vm.l1_restored,
                },
            )
        except BaseException as e:
            # CRITICAL: destroy VM to release admission slot if creation succeeded
            if vm is not None:
                with contextlib.suppress(BaseException):
                    await self.vm_manager.destroy_vm(vm)
            if not isinstance(e, asyncio.CancelledError):
                logger.error(
                    "Failed to boot warm VM",
                    extra={"language": language.value, "index": index, "error": str(e)},
                    exc_info=True,
                )
            raise  # Propagate for gather(return_exceptions=True)

    async def _boot_warm_vm(
        self,
        language: Language,
        index: int,
    ) -> QemuVM:
        """Boot single warm VM with placeholder IDs.

        Tries L1 restore first (REPL already warm), falls back to L2/cold boot.
        Uses reservation_context() to acquire admission once for both paths,
        avoiding double admission wait when L1 restore fails and cold boot is
        the fallback.

        Args:
            language: Programming language enum
            index: VM index in pool (for unique ID)

        Returns:
            Booted QemuVM in READY state
        """
        tenant_id = constants.WARM_POOL_TENANT_ID
        task_id = f"warm-{language.value}-{index}"

        async with self.vm_manager.reservation_context(
            vm_id=f"{tenant_id}-{task_id}",
            memory_mb=constants.DEFAULT_MEMORY_MB,
        ) as reservation:
            # Try L1 restore first (REPL already warm — skip _warm_repl)
            if self.memory_snapshot_manager:
                try:
                    vmstate = await self.memory_snapshot_manager.check_cache(
                        language,
                        [],
                        constants.DEFAULT_MEMORY_MB,
                    )
                    if vmstate:
                        vm = await self.vm_manager.restore_vm(
                            language,
                            tenant_id,
                            task_id,
                            vmstate_path=vmstate,
                            memory_mb=constants.DEFAULT_MEMORY_MB,
                            reservation=reservation,
                        )
                        logger.debug(
                            "L1 cache hit for warm pool VM",
                            extra={"language": language.value, "vm_id": vm.vm_id},
                        )
                        return vm
                except VmTransientError as e:
                    logger.warning(
                        "L1 restore failed for warm pool, falling back to cold boot",
                        extra={"language": language.value, "error": str(e)},
                    )

            # Fall back to L2/cold boot
            snapshot_path = None
            if self.snapshot_manager:
                try:
                    snapshot_path = await self.snapshot_manager.check_cache(
                        language=language,
                        packages=[],
                    )
                    if snapshot_path:
                        logger.debug(
                            "L2 cache hit for warm pool VM",
                            extra={"language": language.value, "snapshot_path": str(snapshot_path)},
                        )
                except (OSError, RuntimeError) as e:
                    logger.warning(
                        "L2 cache check failed for warm pool, falling back to cold boot",
                        extra={"language": language.value, "error": str(e)},
                    )

            return await self.vm_manager.create_vm(
                language=language,
                tenant_id=tenant_id,
                task_id=task_id,
                memory_mb=constants.DEFAULT_MEMORY_MB,
                allow_network=False,
                allowed_domains=None,
                snapshot_drive=snapshot_path,
                reservation=reservation,
                retry_profile=constants.RETRY_BACKGROUND,
            )

    async def _replenish_pool(self, language: Language) -> None:
        """Replenish pool in background (non-blocking).

        A per-language semaphore throttles how many replenish boots run at once
        (see __init__); it does not make the pool.full() check atomic with boot.

        Replenishes ONE VM to maintain pool size.
        Logs failures but doesn't propagate (graceful degradation).

        Args:
            language: Programming language enum to replenish
        """
        async with self._replenish_semaphores[language]:
            vm: QemuVM | None = None
            try:
                if self._shutdown_event.is_set():
                    return
                # Best-effort skip when already full. Not atomic with boot: a
                # concurrent replenish can still over-boot by one (see __init__).
                if self.pools[language].full():
                    logger.debug("Warm pool already full (skip replenish)", extra={"language": language.value})
                    return

                # Boot new VM
                index = self.pools[language].maxsize - self.pools[language].qsize()
                vm = await self._boot_warm_vm(language, index=index)

                # Use the same readiness and idle-memory path as initial fill.
                await self._prepare_and_enqueue_vm(vm, language)

                logger.info(
                    "Warm pool replenished",
                    extra={"language": language.value, "vm_id": vm.vm_id, "pool_size": self.pools[language].qsize()},
                )

            except asyncio.CancelledError:
                # CancelledError is BaseException, not caught by 'except Exception'
                # Cleanup VM if creation succeeded before cancellation
                if vm is not None:
                    with contextlib.suppress(Exception):
                        await self.vm_manager.destroy_vm(vm)
                logger.debug("Replenish task cancelled", extra={"language": language.value})
                raise  # Re-raise cancellation to propagate shutdown

            except Exception as e:
                # CRITICAL: destroy VM to release admission slot if creation succeeded
                if vm is not None:
                    with contextlib.suppress(Exception):
                        await self.vm_manager.destroy_vm(vm)
                logger.error(
                    "Failed to replenish warm pool",
                    extra={"language": language.value, "error": str(e)},
                    exc_info=True,
                )
                # Don't propagate - graceful degradation

    async def _prepare_and_enqueue_vm(
        self,
        vm: QemuVM,
        language: Language,
        *,
        repl_is_ready: bool = False,
    ) -> None:
        """Apply the one readiness/memory contract for every pool insertion."""
        if self._shutdown_event.is_set():
            raise asyncio.CancelledError
        if not vm.l1_restored and not repl_is_ready:
            await self._warm_repl(vm, language)

        # L1 uses file-backed MAP_PRIVATE memory and has no balloon device.
        if not vm.l1_restored:
            await self._inflate_balloon(vm)
            await cgroup.reclaim_memory(vm.cgroup_path)

        if self._shutdown_event.is_set():
            raise asyncio.CancelledError
        await self.pools[language].put(vm)

    async def _inflate_balloon(self, vm: QemuVM) -> None:
        """Inflate balloon to reduce guest memory for idle pool VM.

        Inflating the balloon takes memory FROM the guest, reducing idle footprint.
        Graceful degradation: logs warning and continues if balloon fails.

        Args:
            vm: QemuVM to inflate balloon for
        """
        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            async with BalloonClient(vm.qmp_socket, expected_uid) as client:
                await client.inflate(target_mb=constants.BALLOON_INFLATE_TARGET_MB)
                logger.debug(
                    "Balloon inflated for warm pool VM",
                    extra={"vm_id": vm.vm_id, "target_mb": constants.BALLOON_INFLATE_TARGET_MB},
                )
        except (BalloonError, OSError, TimeoutError) as e:
            # Graceful degradation: log and continue
            logger.warning(
                "Balloon inflation failed (VM will use full memory)",
                extra={"vm_id": vm.vm_id, "error": str(e)},
            )

    async def _deflate_balloon(self, vm: QemuVM) -> None:
        """Deflate balloon to restore guest memory before code execution.

        Deflating the balloon returns memory TO the guest. Uses fire-and-forget
        mode (wait_for_target=False) to avoid blocking - the balloon command is
        sent immediately and memory is restored progressively while code runs.

        This eliminates up to 5s of polling overhead on slow systems (nested
        virtualization) where balloon operations are degraded. Most code doesn't
        need the full 192MB immediately - the idle memory (BALLOON_INFLATE_TARGET_MB)
        is sufficient for runtime startup, and full memory becomes available within ~1s.

        Graceful degradation: logs warning and continues if balloon fails.

        Args:
            vm: QemuVM to deflate balloon for
        """
        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            async with BalloonClient(vm.qmp_socket, expected_uid) as client:
                await client.deflate(target_mb=constants.DEFAULT_MEMORY_MB, wait_for_target=False)
                logger.debug(
                    "Balloon deflated for warm pool VM",
                    extra={"vm_id": vm.vm_id, "target_mb": constants.DEFAULT_MEMORY_MB},
                )
        except (BalloonError, OSError, TimeoutError) as e:
            # Graceful degradation: log and continue (VM may be memory-constrained)
            logger.warning(
                "Balloon deflation failed (VM may be memory-constrained)",
                extra={"vm_id": vm.vm_id, "error": str(e)},
            )

    async def _warm_repl(self, vm: QemuVM, language: Language) -> None:
        """Pre-warm REPL in guest VM for faster first execution.

        A failed warm-up makes this VM ineligible for pooling or L1 capture.
        The caller destroys it through the normal _boot_and_add_vm error path.
        """
        response = await vm.channel.send_request(
            WarmReplRequest(language=language),
            timeout=constants.WARM_REPL_TIMEOUT_SECONDS,
        )
        if not (isinstance(response, WarmReplAckMessage) and response.status == "ok"):
            raise RuntimeError(f"REPL pre-warm failed for {language.value}: {response}")
        logger.debug("REPL pre-warmed", extra={"vm_id": vm.vm_id, "language": language.value})

    async def _health_check_loop(self) -> None:
        """Background health check for warm VMs.

        Pings guest agents every WARM_POOL_HEALTH_CHECK_INTERVAL seconds
        (currently 15s) to detect crashes. Replaces unhealthy VMs automatically.
        """
        logger.info("Warm pool health check started")

        while not self._shutdown_event.is_set():
            try:
                # Wait WARM_POOL_HEALTH_CHECK_INTERVAL or until shutdown
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=constants.WARM_POOL_HEALTH_CHECK_INTERVAL,
                )
                break  # Shutdown signaled
            except TimeoutError:
                pass  # Continue health check

            # Check all VMs in all pools
            for language, pool in self.pools.items():
                await self._health_check_pool(language, pool)

        logger.info("Warm pool health check stopped")

    async def _health_check_pool(self, language: Language, pool: asyncio.Queue[QemuVM]) -> None:
        """Perform health check on a single pool.

        Strategy: Remove VMs, check in parallel, restore immediately when healthy.
        Each VM is restored as soon as its check completes - unhealthy VMs don't
        block healthy ones from returning to the pool.
        """
        pool_size = pool.qsize()
        if pool_size == 0:
            return

        check_start = asyncio.get_running_loop().time()
        logger.info(
            "Health check iteration starting",
            extra={"language": language.value, "pool_size": pool_size},
        )

        # Remove all VMs from pool (atomic snapshot)
        vms_to_check = self._drain_pool_for_check(pool, pool_size, language)
        if not vms_to_check:
            return

        # Health check all VMs in parallel - each VM restored immediately when healthy
        results = await asyncio.gather(
            *[self._check_and_restore_vm(vm, pool, language) for vm in vms_to_check],
            return_exceptions=True,
        )

        # Count results (True = healthy, False = unhealthy, Exception = error)
        healthy_count = sum(1 for r in results if r is True)
        unhealthy_count = len(results) - healthy_count

        check_duration = asyncio.get_running_loop().time() - check_start
        logger.info(
            "Health check iteration complete",
            extra={
                "language": language.value,
                "duration_ms": round(check_duration * 1000),
                "healthy": healthy_count,
                "unhealthy": unhealthy_count,
                "pool_size": pool.qsize(),
            },
        )

    def _drain_pool_for_check(self, pool: asyncio.Queue[QemuVM], pool_size: int, language: Language) -> list[QemuVM]:
        """Drain VMs from pool for health checking."""
        vms_to_check: list[QemuVM] = []
        for _ in range(pool_size):
            try:
                vm = pool.get_nowait()
                vms_to_check.append(vm)
            except asyncio.QueueEmpty:
                break

        logger.debug(
            "Pool drained for health check",
            extra={"language": language.value, "vms_removed": len(vms_to_check)},
        )
        return vms_to_check

    async def _check_and_restore_vm(
        self,
        vm: QemuVM,
        pool: asyncio.Queue[QemuVM],
        language: Language,
    ) -> bool:
        """Check VM health and immediately restore to pool if healthy.

        This is called in parallel for all VMs. Each healthy VM is restored
        immediately without waiting for other checks to complete, minimizing
        the window where the pool is depleted.

        Returns:
            True if healthy (restored to pool), False if unhealthy (destroyed).
        """
        try:
            healthy = await self._check_vm_health(vm)
            if healthy:
                await pool.put(vm)  # Immediately back in pool
                return True
            await self._handle_unhealthy_vm(vm, language)
            return False
        except _HEALTH_CHECK_TRANSIENT_ERRORS as e:
            logger.error(
                "Health check exception",
                extra={"language": language.value, "vm_id": vm.vm_id, "error": str(e)},
                exc_info=e,
            )
            await self._handle_unhealthy_vm(vm, language)
            return False
        except asyncio.CancelledError:
            # The VM was removed from the queue before the health await. A
            # cancelled health pass must not strand that checkout.
            with contextlib.suppress(Exception):
                await self.vm_manager.destroy_vm(vm)
            raise
        except Exception as e:
            logger.exception(
                "Unexpected warm VM health-check failure",
                extra={"language": language.value, "vm_id": vm.vm_id, "error": str(e)},
            )
            await self._handle_unhealthy_vm(vm, language)
            return False

    async def _handle_unhealthy_vm(self, vm: QemuVM, language: Language) -> None:
        """Handle an unhealthy VM by destroying and triggering replenishment."""
        logger.warning(
            "Unhealthy warm VM detected",
            extra={"language": language.value, "vm_id": vm.vm_id},
        )
        with contextlib.suppress(Exception):
            await self.vm_manager.destroy_vm(vm)

        self._schedule_replenishment(language)

    async def _check_vm_health(
        self,
        vm: QemuVM,
        *,
        _wait: wait_base | None = None,
    ) -> bool:
        """Check if VM is healthy (guest agent responsive).

        Uses retry with exponential backoff to prevent false positives from
        transient failures. Matches Kubernetes failureThreshold=3 pattern.

        Uses QEMU GA industry standard pattern: connect → command → disconnect
        (same as libvirt, QEMU GA reference implementation).

        Why reconnect per command:
        - virtio-serial: No way to detect if guest agent disconnected (limitation)
        - If guest closed FD after boot ping, our writes queue but never read
        - Result: TimeoutError or IncompleteReadError (EOF)
        - Reconnect ensures fresh connection state each health check

        Libvirt best practice: "guest-sync command prior to every useful command"
        Our implementation: connect() achieves same - fresh channel state

        Args:
            vm: QemuVM to check
            _wait: Optional wait strategy override (for testing with wait_none())

        Returns:
            True if healthy, False otherwise
        """
        # Check stopped first, if stopped, process exists but can't communicate
        if await vm.process.is_stopped():
            logger.warning(
                "VM process is stopped (SIGSTOP/frozen)",
                extra={"vm_id": vm.vm_id},
            )
            return False

        # Then check running, catches terminated processes
        if not await vm.process.is_running():
            logger.warning(
                "VM process not running (killed or crashed)",
                extra={"vm_id": vm.vm_id},
            )
            return False

        async def _ping_guest() -> bool:
            """Single ping attempt - may raise on transient failure."""
            # QEMU GA standard pattern: connect before each command
            logger.debug("Health check: closing existing connection", extra={"vm_id": vm.vm_id})
            await vm.channel.close()
            logger.debug("Health check: establishing fresh connection", extra={"vm_id": vm.vm_id})
            await vm.channel.connect(timeout_seconds=5)
            logger.debug("Health check: sending ping request", extra={"vm_id": vm.vm_id})
            response = await vm.channel.send_request(PingRequest())
            logger.debug(
                "Health check: received response",
                extra={"vm_id": vm.vm_id, "response_type": type(response).__name__},
            )
            return isinstance(response, PongMessage)

        # Use injected wait strategy or default exponential backoff
        wait_strategy = _wait or wait_random_exponential(
            min=constants.WARM_POOL_HEALTH_CHECK_RETRY_MIN_SECONDS,
            max=constants.WARM_POOL_HEALTH_CHECK_RETRY_MAX_SECONDS,
        )

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(constants.WARM_POOL_HEALTH_CHECK_MAX_RETRIES),
                wait=wait_strategy,
                retry=retry_if_exception_type(_HEALTH_CHECK_TRANSIENT_ERRORS),
                before_sleep=before_sleep_log(logger, logging.DEBUG),
                reraise=True,
            ):
                with attempt:
                    return await _ping_guest()
        except _HEALTH_CHECK_TRANSIENT_ERRORS as e:
            # All retries exhausted - log and return unhealthy
            logger.warning(
                "Health check failed after retries",
                extra={
                    "vm_id": vm.vm_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "max_retries": constants.WARM_POOL_HEALTH_CHECK_MAX_RETRIES,
                },
            )
            return False
        except asyncio.CancelledError:
            # Don't retry on cancellation - propagate immediately
            logger.debug("Health check cancelled", extra={"vm_id": vm.vm_id})
            raise

        # Unreachable: AsyncRetrying either returns from within or raises
        # But required for type checker (mypy/pyright) to see all paths return
        raise AssertionError("Unreachable: AsyncRetrying exhausted without exception")

    async def __aenter__(self) -> Self:
        """Enter async context manager, starting the pool."""
        await self.start()
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, stopping the pool."""
        await self.stop()

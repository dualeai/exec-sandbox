"""Warm VM pool for instant code execution startup.

Pre-boots VMs at service startup for default-image executions.
A pool hit starts in under 1 ms, against about 200 ms for a cold boot plus the REPL start.

Architecture:
- Pool size: Configured via warm_pool_size (per language)
- Languages: all (python, javascript, raw)
- Lifecycle: Pre-boot → allocate → execute → destroy → replenish
- Security: One-time use (no cross-tenant reuse)

Performance:
- A spec equal to VmSpec(language=...) defaults: under 1 ms allocation; each pool VM ran
  its REPL warm-up after its reclaim, so the first exec of a user is not the REPL's first
- Any other spec: L1 restore or cold boot
- Memory: memory.reclaim moves the pages of idle pool VMs to zswap (Linux); the
  guest kernel zeroes freed pages (init_on_free), so they cost almost nothing there

Example:
    ```python
    # In Scheduler
    warm_pool = WarmVMPool(vm_manager, config, memory_snapshot_manager=memory_snapshot_manager)
    await warm_pool.start()
    # Per execution
    vm = await warm_pool.get_vm(VmSpec(language="python"))
    if vm:  # Warm hit (under 1 ms)
        result = await vm.execute(...)
    else:  # Cold fallback (about 200 ms + the REPL start)
        vm = await vm_manager.create_vm(...)
    await warm_pool.stop()
    ```
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from typing import TYPE_CHECKING

from tenacity import (
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import AsyncRetrying, await_settled
from exec_sandbox.exceptions import CommunicationError, ErrorCode, OutcomeUnknownError
from exec_sandbox.files.disk import advise_willneed
from exec_sandbox.guest_agent_protocol import PingRequest, PongMessage, WarmReplAckMessage, WarmReplRequest
from exec_sandbox.models import Language
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import VmState

if TYPE_CHECKING:
    from exec_sandbox.config import SchedulerConfig
    from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.vm_manager import VmManager

logger = get_logger(__name__)

# Transport failures of a health ping: the guest cannot answer (QEMU died, froze or
# closed its sockets). The ping is tried again; when every try fails, the VM is unhealthy.
# - OSError: a socket refuses or breaks; TimeoutError (a subclass): no pong in time
# - CommunicationError: the transport failed before the ping was sent (a socket EOF
#   among them). SocketAuthError (a subclass): SO_PEERCRED returns pid=0 when QEMU is
#   frozen (SIGSTOP) due to a kernel race
_HEALTH_CHECK_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (OSError, CommunicationError)


def _is_failed_health_ping(error: BaseException) -> bool:
    """Whether a health ping error means the guest cannot answer (tenacity passes the error).

    A transport failure, or OutcomeUnknownError GUEST_OUTCOME_UNKNOWN (the transport
    was lost after the ping was sent), means True. Any other error is a bug or a
    setup fault (InternalError, ConfigError, RuntimeError, TypeError, ...): False,
    so it propagates. Cancellation also propagates.
    """
    if isinstance(error, OutcomeUnknownError):
        return error.code is ErrorCode.GUEST_OUTCOME_UNKNOWN
    return isinstance(error, _HEALTH_CHECK_TRANSIENT_ERRORS)


class WarmVMPool:
    """Manages pre-booted VMs for instant execution.

    Single Responsibility: VM pool lifecycle management
    - Startup: Pre-boot VMs in parallel
    - Allocation: Get VM from pool (no wait for availability; dead pooled
      VMs are screened out and destroyed at checkout)
    - Replenishment: Background task to maintain pool size
    - Shutdown: Drain and destroy all VMs

    Concurrency: one event loop; the pools are asyncio.Queue objects (not thread-safe)

    Attributes:
        vm_manager: VmManager for VM lifecycle
        pool_size_per_language: Number of VMs per language
        pools: Dict[language, Queue[QemuVM]] for each language
    """

    def __init__(
        self,
        vm_manager: VmManager,
        config: SchedulerConfig,
        *,
        memory_snapshot_manager: MemorySnapshotManager | None,
    ):
        """Initialize warm VM pool.

        Args:
            vm_manager: VmManager for VM lifecycle
            config: Scheduler configuration
            memory_snapshot_manager: MemorySnapshotManager for L1 cache (None: no L1 restore or save)
        """
        self.vm_manager = vm_manager
        self.memory_snapshot_manager = memory_snapshot_manager

        # Pool size is explicitly configured via warm_pool_size
        # warm_pool_size=0 means warm pool disabled (caller should not create WarmVMPool)
        self.pool_size_per_language = config.warm_pool_size

        # Pools: one bounded asyncio.Queue per language
        self.pools: dict[Language, asyncio.Queue[QemuVM]] = {
            lang: asyncio.Queue(maxsize=self.pool_size_per_language) for lang in Language
        }

        # Track background replenish tasks (prevent GC)
        self._replenish_tasks: set[asyncio.Task[None]] = set()

        # Semaphore that throttles concurrent replenishment boots per language,
        # up to ~50% of pool_size for faster catch-up under load. It bounds boot
        # concurrency; it does NOT make the pool.full() check atomic with boot.
        # With a count > 1, two tasks can both pass full() at qsize == maxsize-1
        # and both boot; the loser then blocks in pool.put() holding a live VM
        # until a checkout frees a slot (transient over-boot, self-correcting).
        replenish_max_concurrent = max(
            1,  # Minimum 1 concurrent boot
            int(self.pool_size_per_language * constants.WARM_POOL_REPLENISH_CONCURRENCY_RATIO),
        )
        self._replenish_semaphores: dict[Language, asyncio.Semaphore] = {
            lang: asyncio.Semaphore(replenish_max_concurrent) for lang in Language
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
        """Start the pool, once. On a failure the caller calls stop(): it also ends a start
        in progress and destroys the VMs it made."""
        self._start_task = asyncio.create_task(self._start_impl())
        await asyncio.shield(self._start_task)

    async def _start_impl(self) -> None:
        """Start the warm VM pool by pre-booting VMs (parallel).

        Boots all VMs in parallel for faster startup. A VM that fails to boot is
        logged and left out (the pool runs with fewer VMs): a failed boot does not raise.
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
                key = await self.memory_snapshot_manager.compute_cache_key(self.spec_for(language), l2_path=None)
                vmstate = await self.memory_snapshot_manager.check_cache(key)
                if vmstate:
                    advise_willneed(vmstate)

        # Boots staggered by 15ms to avoid macOS dyld contention.  Each QEMU
        # fork+exec resolves ~170 dylibs under a global dyld lock; overlapping
        # spawns serialize on that lock and degrade from ~150ms to 250-400ms per
        # VM.  15ms ≈ single-process dyld resolve time, so the next spawn starts
        # just as the previous one releases the lock.
        # Boot all VMs in parallel (_boot_and_add_vm logs each failure: graceful degradation).
        # A cancel of this start task cancels each boot, and gather ends only once each
        # boot ended (its cleanup too): stop() awaits this task alone.
        await asyncio.gather(
            *(
                self._boot_and_add_vm(language, index=i, boot_delay=order * 0.015)
                for order, (language, i) in enumerate(itertools.product(Language, range(self.pool_size_per_language)))
            ),
            return_exceptions=True,
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

    async def wait_until_ready(self, timeout: float) -> None:
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

    def spec_for(self, language: Language) -> VmSpec:
        """The one VM shape this pool holds for a language: VmSpec defaults
        (no packages, no network)."""
        return VmSpec(language=language)

    async def get_vm(self, spec: VmSpec) -> QemuVM | None:
        """Get warm VM if eligible (no wait for availability).

        Eligibility: the request's spec equals the pool's spec for its language.
        Graceful degradation: Pool empty → return None (cold boot fallback)

        Side-effect: Triggers background replenishment

        Args:
            spec: Creation settings of the requested VM

        Returns:
            Warm VM if available, None otherwise
        """
        if spec != self.spec_for(spec.language):
            logger.debug("Warm pool ineligible (spec differs from pool)", extra={"language": spec.language.value})
            return None

        vm: QemuVM | None = None
        try:
            # Non-blocking get (raises QueueEmpty if pool exhausted).
            # Skip VMs whose QEMU died while pooled: the process-exit watcher
            # flips state to DESTROYING, and L1-restored VMs (no watcher)
            # expose death via process.returncode. Handing one out fails the
            # caller (InternalError from the execute state gate, or a
            # connect failure for the returncode-only case) instead of
            # cold-boot fallback; the health check only evicts every 15s.
            # Skip also a VM that QEMU paused on a full host disk: it never runs again.
            while True:
                vm = self.pools[spec.language].get_nowait()
                if vm.state is VmState.READY and vm.process.returncode is None and not vm.host_disk_full:
                    break
                logger.warning(
                    "Discarding dead warm VM at checkout",
                    extra={
                        "language": spec.language.value,
                        "vm_id": vm.vm_id,
                        "vm_state": vm.state.value,
                        "returncode": vm.process.returncode,
                        "host_disk_full": vm.host_disk_full,
                    },
                )
                self._schedule_replenishment(spec.language)
                dead_vm = vm
                vm = None
                try:
                    await self.vm_manager.destroy_vm(dead_vm)
                except Exception:
                    logger.exception(
                        "Failed to destroy dead warm VM at checkout",
                        extra={"language": spec.language.value, "vm_id": dead_vm.vm_id},
                    )

            logger.debug(
                "Warm VM allocated",
                extra={
                    "debug_category": "lifecycle",
                    "language": spec.language.value,
                    "vm_id": vm.vm_id,
                    "pool_remaining": self.pools[spec.language].qsize(),
                },
            )

            # Trigger background replenishment (fire-and-forget)
            self._schedule_replenishment(spec.language)

            return vm

        except asyncio.QueueEmpty:
            logger.warning(
                "Warm pool exhausted (cold boot fallback)",
                extra={"language": spec.language.value, "pool_size": self.pool_size_per_language},
            )
            return None

        except BaseException:
            # Ownership transferred out of the queue before the first await.
            # If checkout cannot complete, destroy that VM and replenish rather
            # than leaving a live, unreachable owner outside both pool and caller.
            if vm is None:
                raise
            self._schedule_replenishment(spec.language)
            try:
                await self.vm_manager.destroy_vm(vm)
            except BaseException:
                logger.exception(
                    "Failed to destroy warm VM after interrupted checkout",
                    extra={"language": spec.language.value, "vm_id": vm.vm_id},
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
        """Cancel and await the start task: no initial boot can enqueue a VM after it
        (its gather ends once each boot ended)."""
        if (start_task := self._start_task) is not None:
            start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)

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

        Stop sequence (stop() has set the shutdown event, which ends the health check):
        1. Cancel and await the start task and the initial boots
        2. Wait for health check task
        3. Cancel and await pending replenish tasks
        4. Drain all pools and destroy VMs (parallel)
        """
        logger.info("Shutting down warm VM pool")

        # Initial-fill children are owned by the shared start task. Cancel and
        # await that task before taking the final queue snapshot so no boot can
        # enqueue after a successful stop.
        await self._stop_initial_boots()

        # The shutdown event ends the wait between passes at once; a pass in progress ends
        # first, at most WARM_POOL_HEALTH_CHECK_INTERVAL + 2 s, then it is cancelled
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
                vm = pool.get_nowait()  # never QueueEmpty: no await after empty()
                destroy_tasks.append(asyncio.create_task(self._destroy_vm_with_logging(vm, language)))

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

    async def _boot_and_add_vm(self, language: Language, *, index: int, boot_delay: float) -> None:
        """Boot VM and add to pool (used for parallel startup).

        For index=0 of a cold boot (no L1 entry was restored): schedules a background
        L1 save (a template entry with a warm REPL, made by a sacrificial VM); this VM
        stays in the pool.

        Args:
            language: Programming language enum
            index: VM index in pool (for unique ID)
            boot_delay: Seconds to wait before booting (staggers fork/exec)
        """
        if boot_delay > 0:
            await asyncio.sleep(boot_delay)
        try:
            vm = await self._boot_and_enqueue(language, index=index)
        except BaseException as e:
            if not isinstance(e, asyncio.CancelledError):
                logger.error(
                    "Failed to boot warm VM",
                    extra={"language": language.value, "index": index, "error": str(e)},
                    exc_info=True,
                )
            raise  # Propagate for gather(return_exceptions=True)
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

        # index=0 of a cold boot whose REPL warm-up passed: make the L1 entry for future
        # restores and refills. A background save boots its own VM and saves a template
        # (RAM shared copy-on-write by every restore); it is a no-op when the entry
        # exists, and dedups with the scheduler's saves of the key. The pool owns the VM
        # now (get_vm may have handed it out): a failure here never destroys it.
        if not vm.l1_restored and self.memory_snapshot_manager and index == 0:
            try:
                spec = self.spec_for(language)
                key = await self.memory_snapshot_manager.compute_cache_key(spec, l2_path=None)
                await self.memory_snapshot_manager.schedule_background_save(spec, key=key, snapshot_drive=None)
            except Exception as e:
                logger.warning(
                    "Could not schedule the L1 save of a warm pool language",
                    extra={"language": language.value, "error": str(e)},
                    exc_info=True,
                )

    async def _boot_warm_vm(
        self,
        language: Language,
        index: int,
    ) -> QemuVM:
        """Boot single warm VM with placeholder IDs.

        Tries L1 restore first (its REPL runs), falls back to a cold boot.
        Uses reservation_context() to acquire admission once for both paths,
        avoiding double admission wait when L1 restore fails and cold boot is
        the fallback.

        Args:
            language: Programming language enum
            index: VM index in pool (for unique ID)

        Returns:
            Booted QemuVM in READY state

        Raises:
            What VmManager.reservation_context and create_vm raise; from the L1 restore,
            only InternalError CLEANUP_UNCONFIRMED (any other failed restore gives None:
            the cold boot follows)
        """
        tenant_id = constants.WARM_POOL_TENANT_ID
        task_id = f"warm-{language.value}-{index}"
        spec = self.spec_for(language)

        # The L1 entry is leased before admission: admission may evict cache entries for
        # disk space, never a leased one. The pool's spec has no packages: no L2 layer.
        memory = self.memory_snapshot_manager
        l1_entry = await memory.lease(await memory.compute_cache_key(spec, l2_path=None)) if memory else None
        try:
            held = frozenset({l1_entry.path} if l1_entry is not None else ())
            async with self.vm_manager.reservation_context(
                spec, vm_id=f"{tenant_id}-{task_id}", held=held
            ) as reservation:
                # Try L1 restore first (its REPL runs: the warm-up only exercises it)
                if memory and l1_entry is not None:
                    vm = await memory.restore(
                        spec,
                        entry=l1_entry,
                        tenant_id=tenant_id,
                        task_id=task_id,
                        snapshot_drive=None,
                        expose_ports=None,
                        reservation=reservation,
                    )
                    if vm is not None:
                        logger.debug(
                            "L1 cache hit for warm pool VM",
                            extra={"language": language.value, "vm_id": vm.vm_id},
                        )
                        return vm
                return await self.vm_manager.create_vm(
                    spec,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    snapshot_drive=None,
                    cache_leases=(),
                    reservation=reservation,
                    retry_profile=constants.RETRY_BACKGROUND,
                    direct_write_target=None,
                    expose_ports=None,
                    on_boot_log=None,
                    mem_path=None,
                )
        finally:
            if l1_entry is not None:  # restore() releases it; this covers a failed admission
                l1_entry.release()

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
            try:
                if self._shutdown_event.is_set():
                    return
                # Best-effort skip when already full. Not atomic with boot: a
                # concurrent replenish can still over-boot by one (see __init__).
                if self.pools[language].full():
                    logger.debug("Warm pool already full (skip replenish)", extra={"language": language.value})
                    return

                # Boot new VM (the same readiness and idle-memory path as initial fill)
                index = self.pools[language].maxsize - self.pools[language].qsize()
                vm = await self._boot_and_enqueue(language, index=index)

                logger.info(
                    "Warm pool replenished",
                    extra={"language": language.value, "vm_id": vm.vm_id, "pool_size": self.pools[language].qsize()},
                )

            except asyncio.CancelledError:
                logger.debug("Replenish task cancelled", extra={"language": language.value})
                raise  # Re-raise cancellation to propagate shutdown

            except Exception as e:
                logger.error(
                    "Failed to replenish warm pool",
                    extra={"language": language.value, "error": str(e)},
                    exc_info=True,
                )
                # Don't propagate - graceful degradation

    async def _boot_and_enqueue(self, language: Language, *, index: int) -> QemuVM:
        """Boot a warm VM and put it in its pool (_prepare_and_enqueue_vm). On a failure, a
        cancel too, destroy the VM it booted (its admission slot comes back), then raise."""
        vm: QemuVM | None = None
        try:
            vm = await self._boot_warm_vm(language, index)
            await self._prepare_and_enqueue_vm(vm, language)
        except BaseException:
            if vm is not None:
                with contextlib.suppress(Exception):
                    await self.vm_manager.destroy_vm(vm)
            raise
        return vm

    async def _prepare_and_enqueue_vm(self, vm: QemuVM, language: Language) -> None:
        """Apply the one readiness/memory contract for every pool insertion."""
        if self._shutdown_event.is_set():
            raise asyncio.CancelledError

        # A cold boot starts its REPL before the reclaim, so the pages of its start
        # (heap, freed import churn) can go to zswap with the rest
        if not vm.l1_restored:
            await self._warm_repl(vm, language)
        # Every idle VM: with init_on_free its free pages are zero, so reclaiming up
        # to the whole guest RAM moves them to zswap at almost no cost.
        if vm.cgroup is not None:
            await vm.cgroup.reclaim(amount_mb=constants.DEFAULT_MEMORY_MB)
        # Then every VM runs the warm-up on its live REPL (an L1 restore has one): the
        # pages the no-op uses come back after the reclaim, so the first exec of a user
        # is not the REPL's first. Measured (KVM), the user's first exec: on a VM restored
        # from L1, Python 17.7 -> 2.5 ms, JavaScript 54.3 -> 3.9 ms (TCG adds the
        # translation, about 100 and 240 ms); on a cold boot, JavaScript 5.5 -> 2.3 ms
        # against one warm-up after the reclaim (Python 2.1 either way).
        await self._warm_repl(vm, language)

        if self._shutdown_event.is_set():
            raise asyncio.CancelledError
        await self.pools[language].put(vm)

    async def _warm_repl(self, vm: QemuVM, language: Language) -> None:
        """Warm the REPL of `language` in the VM (WarmReplRequest: the guest starts the REPL
        when none runs, then runs a no-op on it).

        A failed warm-up (RuntimeError, or the error of send_request) makes this VM
        ineligible for pooling (and index 0 then schedules no L1 save):
        _boot_and_enqueue destroys it.
        """
        response = await vm.channel.send_request(
            WarmReplRequest(language=language),
            timeout=constants.WARM_REPL_TIMEOUT_SECONDS,
        )
        if not (isinstance(response, WarmReplAckMessage) and response.status == "ok"):
            raise RuntimeError(f"REPL pre-warm failed for {language.value}: {response}")
        vm.reseed_next_exec = False  # the warm-up ran the REPL's reseed (WarmReplRequest)
        logger.debug("REPL pre-warmed", extra={"vm_id": vm.vm_id, "language": language.value})

    async def _health_check_loop(self) -> None:
        """Background health check for warm VMs.

        Pings guest agents every WARM_POOL_HEALTH_CHECK_INTERVAL seconds
        (15 s) to detect crashes. Replaces unhealthy VMs automatically.
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

        # Remove all VMs from pool (atomic snapshot: no await since qsize(), each get succeeds)
        vms_to_check = [pool.get_nowait() for _ in range(pool_size)]

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
            if await self._check_vm_health(vm):
                await pool.put(vm)  # Immediately back in pool
                return True
        except asyncio.CancelledError:
            # The VM was removed from the queue before the health await. A
            # cancelled health pass must not strand that checkout.
            with contextlib.suppress(Exception):
                await self.vm_manager.destroy_vm(vm)
            raise
        except Exception as e:
            logger.exception(
                "Health check exception",
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

    async def _check_vm_health(self, vm: QemuVM) -> bool:
        """Check if VM is healthy (guest agent responsive).

        Uses retry with exponential backoff to prevent false positives from
        transient failures. Matches Kubernetes failureThreshold=3 pattern.

        Pings on the VM's open connection. The guest agent keeps its ports open
        for its whole life, so a ping is held until the agent reads it, never
        lost. connect() does nothing on a working connection and opens a new one
        only when the old one failed (its event dispatcher stopped): a close and
        reconnect per check would show the guest a host disconnect each time.

        Args:
            vm: QemuVM to check

        Returns:
            True if healthy, False otherwise. A ping that fails on every try because
            the guest cannot answer (see _is_failed_health_ping) gives False; any
            other error propagates.
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
            await vm.channel.connect(timeout_seconds=5)  # reconnects only a failed connection
            logger.debug("Health check: sending ping request", extra={"vm_id": vm.vm_id})
            response = await vm.channel.send_request(
                PingRequest(host_entropy=None), timeout=constants.GUEST_REQUEST_TIMEOUT_SECONDS
            )
            logger.debug(
                "Health check: received response",
                extra={"vm_id": vm.vm_id, "response_type": type(response).__name__},
            )
            return isinstance(response, PongMessage)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(constants.WARM_POOL_HEALTH_CHECK_MAX_RETRIES),
                wait=wait_random_exponential(
                    min=constants.WARM_POOL_HEALTH_CHECK_RETRY_MIN_SECONDS,
                    max=constants.WARM_POOL_HEALTH_CHECK_RETRY_MAX_SECONDS,
                ),
                retry=retry_if_exception(_is_failed_health_ping),
                before_sleep=before_sleep_log(logger, logging.DEBUG),
                reraise=True,
            ):
                with attempt:
                    return await _ping_guest()
        except Exception as e:
            if not _is_failed_health_ping(e):
                raise  # a bug or a setup fault, not a VM that cannot answer
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

        # Unreachable: AsyncRetrying either returns from within or raises
        # But required for type checker (mypy/pyright) to see all paths return
        raise AssertionError("Unreachable: AsyncRetrying exhausted without exception")

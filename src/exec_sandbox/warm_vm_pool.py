"""Warm VM pool for instant code execution startup.

Pre-boots VMs at service startup for default-image executions.
Provides 200-400x faster execution start (1-2ms vs 400ms cold boot).

Architecture:
- Pool size: 25% of max_concurrent_vms (2-3 VMs per language)
- Languages: python, javascript
- Lifecycle: Pre-boot → allocate → execute → destroy → replenish
- Security: One-time use (no cross-tenant reuse)

Performance:
- Default image (packages=[]): 1-2ms allocation (vs 400ms cold boot)
- Custom packages: Fallback to cold boot (no change)
- Memory overhead: ~1GB for 4 VMs (256MB x 4, based on 25% of max_concurrent_vms=10)

L2 Disk Snapshots:
- Uses L2 cache (local qcow2) for faster warm pool boots
- snapshot_manager: Optional for L2 cache (graceful degradation to cold boot if None)

Memory Optimization (Balloon):
- Idle pool VMs have balloon inflated (guest has ~64MB free)
- Before execution, balloon deflates (guest gets full memory back)
- Reduces idle memory from ~1GB to ~256MB for 4 VMs (75% reduction)

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

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.balloon_client import BalloonClient, BalloonError
from exec_sandbox.exceptions import SocketAuthError
from exec_sandbox.models import Language
from exec_sandbox.permission_utils import get_expected_socket_uid

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from tenacity.wait import wait_base

    from exec_sandbox.config import SchedulerConfig
    from exec_sandbox.snapshot_manager import SnapshotManager
    from exec_sandbox.vm_manager import QemuVM, VmManager

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
    - Allocation: Get VM from pool (non-blocking)
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
        snapshot_manager: SnapshotManager | None = None,
    ):
        """Initialize warm VM pool.

        Args:
            vm_manager: VmManager for VM lifecycle
            config: Scheduler configuration
            snapshot_manager: Optional SnapshotManager for L2 cache (faster refill)
        """
        self.vm_manager = vm_manager
        self.config = config
        self.snapshot_manager = snapshot_manager

        # Calculate pool size: use explicit warm_pool_size if set, else 25% of max_concurrent_vms
        if config.warm_pool_size > 0:
            self.pool_size_per_language = config.warm_pool_size
        else:
            self.pool_size_per_language = max(
                1,  # Minimum 1 VM per language
                int(config.max_concurrent_vms * constants.WARM_POOL_SIZE_RATIO),
            )

        # Pools: asyncio.Queue for thread-safe async access
        self.pools: dict[Language, asyncio.Queue[QemuVM]] = {
            lang: asyncio.Queue(maxsize=self.pool_size_per_language) for lang in constants.WARM_POOL_LANGUAGES
        }

        # Track background replenish tasks (prevent GC)
        self._replenish_tasks: set[asyncio.Task[None]] = set()

        # Semaphore to limit concurrent replenishment per language (prevents race condition
        # where multiple tasks pass the pool.full() check before any VM is booted)
        # Allows parallel boots up to 50% of pool_size for faster replenishment under load
        self._replenish_max_concurrent = max(
            1,  # Minimum 1 concurrent boot
            int(self.pool_size_per_language * constants.WARM_POOL_REPLENISH_CONCURRENCY_RATIO),
        )
        self._replenish_semaphores: dict[Language, asyncio.Semaphore] = {
            lang: asyncio.Semaphore(self._replenish_max_concurrent) for lang in constants.WARM_POOL_LANGUAGES
        }

        # Health check task
        self._health_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

        logger.info(
            "Warm VM pool initialized",
            extra={
                "pool_size_per_language": self.pool_size_per_language,
                "languages": [lang.value for lang in constants.WARM_POOL_LANGUAGES],
                "total_vms": self.pool_size_per_language * len(constants.WARM_POOL_LANGUAGES),
            },
        )

    async def start(self) -> None:
        """Start the warm VM pool by pre-booting VMs (parallel).

        Boots all VMs in parallel for faster startup.
        Logs progress for operational visibility.

        Raises:
            VmTransientError: If critical number of VMs fail to boot
        """
        logger.info(
            "Starting warm VM pool",
            extra={"total_vms": self.pool_size_per_language * len(constants.WARM_POOL_LANGUAGES)},
        )

        boot_start = asyncio.get_event_loop().time()

        # Build list of all VMs to boot (parallel execution)
        boot_coroutines: list[Coroutine[Any, Any, None]] = []
        for language in constants.WARM_POOL_LANGUAGES:
            logger.info(f"Pre-booting {self.pool_size_per_language} {language.value} VMs (parallel)")
            boot_coroutines.extend(self._boot_and_add_vm(language, index=i) for i in range(self.pool_size_per_language))

        # Boot all VMs in parallel
        results: list[None | BaseException] = await asyncio.gather(*boot_coroutines, return_exceptions=True)

        # Log failures (graceful degradation)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Failed to boot warm VM",
                    extra={"task_index": i, "error": str(result)},
                    exc_info=result,
                )

        boot_duration = asyncio.get_event_loop().time() - boot_start

        # Start health check background task
        self._health_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            "Warm VM pool startup complete",
            extra={
                "boot_duration_s": f"{boot_duration:.2f}",
                "python_vms": self.pools[Language.PYTHON].qsize(),
                "javascript_vms": self.pools[Language.JAVASCRIPT].qsize(),
            },
        )

    async def get_vm(
        self,
        language: Language,
        packages: list[str],
    ) -> QemuVM | None:
        """Get warm VM if eligible (non-blocking).

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

        try:
            # Non-blocking get (raises QueueEmpty if pool exhausted)
            vm = self.pools[language].get_nowait()

            # Deflate balloon to restore memory before code execution
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
            replenish_task: asyncio.Task[None] = asyncio.create_task(self._replenish_pool(language))
            self._replenish_tasks.add(replenish_task)
            replenish_task.add_done_callback(lambda t: self._replenish_tasks.discard(t))

            return vm

        except asyncio.QueueEmpty:
            logger.warning(
                "Warm pool exhausted (cold boot fallback)",
                extra={"language": language.value, "pool_size": self.pool_size_per_language},
            )
            return None

    async def stop(self) -> None:
        """Stop the warm VM pool: drain and destroy all VMs.

        Stop sequence:
        1. Signal health check to stop
        2. Wait for health check task
        3. Drain all pools and destroy VMs (parallel)
        4. Cancel pending replenish tasks
        """
        logger.info("Shutting down warm VM pool")

        # Stop health check with timeout to prevent indefinite wait
        # Timeout must be > health check interval (10s) to allow current iteration to complete
        self._shutdown_event.set()
        if self._health_task:
            try:
                await asyncio.wait_for(
                    self._health_task,
                    timeout=constants.WARM_POOL_HEALTH_CHECK_INTERVAL + 2.0,
                )
            except TimeoutError:
                logger.warning("Health check task timed out during shutdown, cancelling")
                self._health_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._health_task

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

        # Cancel pending replenish tasks and await all together
        # CRITICAL: Using asyncio.gather() ensures all gather children complete before continuing.
        # When asyncio.gather() is cancelled, it cancels child tasks but does NOT await their
        # completion. Python 3.14 has stricter detection of these orphaned tasks.
        tasks_to_cancel = list(self._replenish_tasks)
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()

        # Await ALL cancelled tasks together to ensure gather futures complete
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        self._replenish_tasks.clear()

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
            await self.vm_manager.destroy_vm(vm)
            logger.debug("Warm VM destroyed", extra={"language": language.value, "vm_id": vm.vm_id})
            return True
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
    ) -> None:
        """Boot VM and add to pool (used for parallel startup).

        Args:
            language: Programming language enum
            index: VM index in pool (for unique ID)
        """
        vm: QemuVM | None = None
        try:
            vm = await self._boot_warm_vm(language, index)

            # Inflate balloon to reduce idle memory footprint
            await self._inflate_balloon(vm)

            await self.pools[language].put(vm)
            logger.info(
                "Warm VM ready",
                extra={
                    "language": language.value,
                    "vm_id": vm.vm_id,
                    "index": index,
                    "total": self.pool_size_per_language,
                },
            )
        except Exception as e:
            # CRITICAL: destroy VM to release semaphore slot if creation succeeded
            if vm is not None:
                with contextlib.suppress(Exception):
                    await self.vm_manager.destroy_vm(vm)
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

        Uses L2 cache for disk caching if available.

        Args:
            language: Programming language enum
            index: VM index in pool (for unique ID)

        Returns:
            Booted QemuVM in READY state
        """
        # Placeholder IDs for warm pool VMs
        tenant_id = constants.WARM_POOL_TENANT_ID
        task_id = f"warm-{language.value}-{index}"

        # Check L2 cache for base image (check-only, no create)
        # For base images (no packages), creating L2 cache is pointless - would boot VM
        # just to shut it down with no data to cache. Boot from base image directly.
        snapshot_path = None
        if self.snapshot_manager:
            try:
                snapshot_path = await self.snapshot_manager.check_cache(
                    language=language,
                    packages=[],  # Base image, no packages
                )
                if snapshot_path:
                    logger.debug(
                        "L2 cache hit for warm pool VM",
                        extra={"language": language.value, "snapshot_path": str(snapshot_path)},
                    )
            except (OSError, RuntimeError) as e:
                # Graceful degradation: log and continue with cold boot
                logger.warning(
                    "L2 cache check failed for warm pool, falling back to cold boot",
                    extra={"language": language.value, "error": str(e)},
                )

        return await self.vm_manager.create_vm(
            language=language,
            tenant_id=tenant_id,
            task_id=task_id,
            backing_image=snapshot_path,
            memory_mb=constants.DEFAULT_MEMORY_MB,
            allow_network=False,  # Warm pool VMs don't need network
            allowed_domains=None,
        )

    async def _replenish_pool(self, language: Language) -> None:
        """Replenish pool in background (non-blocking).

        Uses semaphore to serialize replenishment per language, preventing race
        condition where multiple tasks pass pool.full() before any VM is booted.

        Replenishes ONE VM to maintain pool size.
        Logs failures but doesn't propagate (graceful degradation).

        Args:
            language: Programming language enum to replenish
        """
        async with self._replenish_semaphores[language]:
            vm: QemuVM | None = None
            try:
                # Check if pool already full (now atomic with boot due to semaphore)
                if self.pools[language].full():
                    logger.debug("Warm pool already full (skip replenish)", extra={"language": language.value})
                    return

                # Boot new VM
                index = self.pools[language].maxsize - self.pools[language].qsize()
                vm = await self._boot_warm_vm(language, index=index)

                # Add to pool
                await self.pools[language].put(vm)

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
                # CRITICAL: destroy VM to release semaphore slot if creation succeeded
                if vm is not None:
                    with contextlib.suppress(Exception):
                        await self.vm_manager.destroy_vm(vm)
                logger.error(
                    "Failed to replenish warm pool",
                    extra={"language": language.value, "error": str(e)},
                    exc_info=True,
                )
                # Don't propagate - graceful degradation

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
        need the full 256MB immediately - the 64MB idle memory is sufficient
        for runtime startup, and additional memory becomes available within ~1s.

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

    async def _health_check_loop(self) -> None:
        """Background health check for warm VMs.

        Pings guest agents every 30s to detect crashes.
        Replaces unhealthy VMs automatically.
        """
        logger.info("Warm pool health check started")

        while not self._shutdown_event.is_set():
            try:
                # Wait 30s or until shutdown
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

        check_start = asyncio.get_event_loop().time()
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

        check_duration = asyncio.get_event_loop().time() - check_start
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

    async def _handle_unhealthy_vm(self, vm: QemuVM, language: Language) -> None:
        """Handle an unhealthy VM by destroying and triggering replenishment."""
        logger.warning(
            "Unhealthy warm VM detected",
            extra={"language": language.value, "vm_id": vm.vm_id},
        )
        with contextlib.suppress(Exception):
            await self.vm_manager.destroy_vm(vm)

        # Trigger replenishment
        task: asyncio.Task[None] = asyncio.create_task(self._replenish_pool(language))
        self._replenish_tasks.add(task)
        task.add_done_callback(lambda t: self._replenish_tasks.discard(t))

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

        from exec_sandbox.guest_agent_protocol import (  # noqa: PLC0415
            PingRequest,
            PongMessage,
        )

        async def _ping_guest() -> bool:
            """Single ping attempt - may raise on transient failure."""
            # QEMU GA standard pattern: connect before each command
            logger.debug("Health check: closing existing connection", extra={"vm_id": vm.vm_id})
            await vm.channel.close()
            logger.debug("Health check: establishing fresh connection", extra={"vm_id": vm.vm_id})
            await vm.channel.connect(timeout_seconds=5)
            logger.debug("Health check: sending ping request", extra={"vm_id": vm.vm_id})
            response = await vm.channel.send_request(PingRequest(), timeout=5)
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

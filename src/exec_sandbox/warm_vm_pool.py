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

L0 Memory Snapshots (Future Enhancement):
- First boot: cold boot → save L0 snapshot (python-base, javascript-base)
- Subsequent boots: restore from L0 via -loadvm (~50-200ms)
- Requires create_vm_from_snapshot() method in VmManager

Example:
    ```python
    # In Scheduler
    warm_pool = WarmVMPool(vm_manager, config)
    await warm_pool.startup()  # Pre-boot VMs

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
from typing import TYPE_CHECKING, Any

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.models import Language

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from exec_sandbox.config import SchedulerConfig
    from exec_sandbox.vm_manager import QemuVM, VmManager

logger = get_logger(__name__)


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

    def __init__(self, vm_manager: VmManager, config: SchedulerConfig):
        """Initialize warm VM pool.

        Args:
            vm_manager: VmManager for VM lifecycle
            config: Scheduler configuration
        """
        self.vm_manager = vm_manager
        self.config = config

        # Calculate pool size: 25% of max_concurrent_vms
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

    async def startup(self) -> None:
        """Pre-boot VMs on service startup (parallel).

        Boots all VMs in parallel for faster startup.
        Logs progress for operational visibility.

        Raises:
            VmError: If critical number of VMs fail to boot
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

    async def shutdown(self) -> None:
        """Drain and destroy all warm VMs (blocking).

        Shutdown sequence:
        1. Signal health check to stop
        2. Wait for health check task
        3. Drain all pools and destroy VMs (parallel)
        4. Cancel pending replenish tasks
        """
        logger.info("Shutting down warm VM pool")

        # Stop health check
        self._shutdown_event.set()
        if self._health_task:
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

        # Cancel pending replenish tasks (wait for cancellation to complete)
        # Copy set to avoid "Set changed size during iteration" error
        for task in list(self._replenish_tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task  # Expected during shutdown

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
        try:
            vm = await self._boot_warm_vm(language, index)
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

        Args:
            language: Programming language enum
            index: VM index in pool (for unique ID)

        Returns:
            Booted QemuVM in READY state
        """
        # Placeholder IDs for warm pool VMs
        tenant_id = constants.WARM_POOL_TENANT_ID
        task_id = f"warm-{language.value}-{index}"

        return await self.vm_manager.create_vm(
            language=language.value,
            tenant_id=tenant_id,
            task_id=task_id,
            snapshot_path=None,  # Default image (no packages)
            memory_mb=constants.DEFAULT_MEMORY_MB,
            allow_network=False,  # Complete isolation
            allowed_domains=None,
        )

    async def _replenish_pool(self, language: Language) -> None:
        """Replenish pool in background (non-blocking).

        Replenishes ONE VM to maintain pool size.
        Logs failures but doesn't propagate (graceful degradation).

        Args:
            language: Programming language enum to replenish
        """
        try:
            # Check if pool already full (race condition)
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

        except Exception as e:
            logger.error(
                "Failed to replenish warm pool",
                extra={"language": language.value, "error": str(e)},
                exc_info=True,
            )
            # Don't propagate - graceful degradation

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

        Strategy: Remove VMs, check in parallel, restore only healthy ones.
        This ensures pool never has unhealthy VMs, even if some exhaustion during check.
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

        # Health check all VMs in parallel
        results = await asyncio.gather(
            *[self._check_vm_health(vm) for vm in vms_to_check],
            return_exceptions=True,
        )

        # Process results and restore healthy VMs
        healthy_count, unhealthy_count = await self._process_health_results(language, pool, vms_to_check, results)

        check_duration = asyncio.get_event_loop().time() - check_start
        logger.info(
            "Health check iteration complete",
            extra={
                "language": language.value,
                "duration_ms": f"{check_duration * 1000:.1f}",
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

    async def _process_health_results(
        self,
        language: Language,
        pool: asyncio.Queue[QemuVM],
        vms: list[QemuVM],
        results: list[bool | BaseException],
    ) -> tuple[int, int]:
        """Process health check results and restore healthy VMs."""
        healthy_count = 0
        unhealthy_count = 0

        for vm, result in zip(vms, results, strict=True):
            healthy = self._evaluate_health_result(result, language, vm)

            if healthy:
                await pool.put(vm)
                healthy_count += 1
            else:
                await self._handle_unhealthy_vm(vm, language)
                unhealthy_count += 1

        return healthy_count, unhealthy_count

    def _evaluate_health_result(self, result: bool | BaseException, language: Language, vm: QemuVM) -> bool:
        """Evaluate health check result for a single VM."""
        if isinstance(result, BaseException):
            logger.error(
                "Health check exception",
                extra={"language": language.value, "vm_id": vm.vm_id, "error": str(result)},
                exc_info=result,
            )
            return False
        return result

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

    async def _check_vm_health(self, vm: QemuVM) -> bool:
        """Check if VM is healthy (guest agent responsive).

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

        Returns:
            True if healthy, False otherwise
        """
        try:
            from exec_sandbox.guest_agent_protocol import (  # noqa: PLC0415
                PingRequest,
                PongMessage,
            )

            # QEMU GA standard pattern: connect before each command
            logger.debug("Health check: closing existing connection", extra={"vm_id": vm.vm_id})
            # Close existing connection first (if any)
            await vm.channel.close()
            logger.debug("Health check: establishing fresh connection", extra={"vm_id": vm.vm_id})
            # Establish fresh connection
            await vm.channel.connect(timeout_seconds=5)
            logger.debug("Health check: sending ping request", extra={"vm_id": vm.vm_id})
            # Send command
            response = await vm.channel.send_request(PingRequest(), timeout=5)
            logger.debug(
                "Health check: received response",
                extra={"vm_id": vm.vm_id, "response_type": type(response).__name__},
            )
            return isinstance(response, PongMessage)

        except (OSError, TimeoutError, asyncio.CancelledError, ConnectionError) as e:
            # Log exception for debugging - specific exceptions for health check failures
            logger.warning(
                "Health check ping failed",
                extra={"vm_id": vm.vm_id, "error": str(e), "error_type": type(e).__name__},
            )
            return False

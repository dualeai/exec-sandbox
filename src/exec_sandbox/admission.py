"""Resource-aware admission controller for VM scheduling.

Replaces the simple asyncio.Semaphore with two resource-based admission gates:
1. Memory budget - host_total * (1 - reserve_ratio) * overcommit_ratio
2. CPU budget - host_cpu_count * cpu_overcommit_ratio

Both gates must pass for a VM to be admitted. Uses asyncio.Condition
for blocking/signaling when resources are insufficient.

Graceful degradation: if psutil probe fails, budgets are set to infinity
(no admission control, equivalent to unlimited capacity).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Final
from uuid import uuid4

from exec_sandbox._logging import get_logger
from exec_sandbox.cgroup import CGROUP_MEMORY_OVERHEAD_MB, TCG_TB_CACHE_SIZE_MB
from exec_sandbox.exceptions import VmCapacityError

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResourceReservation:
    """Tracks resources reserved by a single VM admission."""

    vm_id: str
    memory_mb: float
    cpu_cores: float
    # Unique key in _reservations dict — prevents collision when
    # multiple VMs share the same vm_id string (e.g. concurrent requests
    # with the same tenant_id + task_id).
    reservation_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ResourceSnapshot:
    """Point-in-time view of admission controller state."""

    # Host resources (detected at start)
    host_memory_mb: float
    host_cpu_count: float

    # Budget limits (after overcommit)
    memory_budget_mb: float
    cpu_budget: float

    # Current allocations
    allocated_memory_mb: float = 0.0
    allocated_cpu: float = 0.0
    allocated_vm_slots: int = 0

    # Computed availability
    available_memory_mb: float = field(init=False)
    available_cpu: float = field(init=False)

    def __post_init__(self) -> None:
        self.available_memory_mb = max(0.0, self.memory_budget_mb - self.allocated_memory_mb)
        self.available_cpu = max(0.0, self.cpu_budget - self.allocated_cpu)


# Sentinel for "no limit" when psutil probe fails
_UNLIMITED: Final[float] = float("inf")


class ResourceAdmissionController:
    """Resource-aware admission controller replacing asyncio.Semaphore.

    Two gates (both must pass):
    1. Memory budget: host_total * (1 - reserve_ratio) * overcommit_ratio
    2. CPU budget: host_cpus * overcommit_ratio

    If psutil is unavailable, both gates degrade to unlimited
    (no admission control).
    """

    def __init__(
        self,
        memory_overcommit_ratio: float,
        cpu_overcommit_ratio: float,
        host_memory_reserve_ratio: float,
        host_memory_mb: float | None = None,
        host_cpu_count: float | None = None,
    ) -> None:
        self._memory_overcommit_ratio = memory_overcommit_ratio
        self._cpu_overcommit_ratio = cpu_overcommit_ratio
        self._host_memory_reserve_ratio = host_memory_reserve_ratio

        # Set by start() or constructor override (for testing)
        self._host_memory_mb: float = host_memory_mb if host_memory_mb is not None else 0.0
        self._host_cpu_count: float = host_cpu_count if host_cpu_count is not None else 0.0

        # Budget limits (computed in start())
        self._memory_budget_mb: float = _UNLIMITED
        self._cpu_budget: float = _UNLIMITED

        # Current allocations (protected by _condition's lock)
        self._allocated_memory_mb: float = 0.0
        self._allocated_cpu: float = 0.0
        self._allocated_vm_slots: int = 0
        self._reservations: dict[str, ResourceReservation] = {}

        # Condition for blocking acquire / signaling release
        self._condition = asyncio.Condition()
        self._started = False
        self._start_lock = asyncio.Lock()

        # If host resources were provided at construction, compute budgets now
        if host_memory_mb is not None and host_cpu_count is not None:
            self._compute_budgets()
            self._started = True

    async def start(self) -> None:
        """Probe host resources via psutil and compute budgets.

        Safe to call multiple times (idempotent after first probe).
        If psutil fails, degrades gracefully to unlimited budgets.
        Serialized via asyncio.Lock to prevent concurrent probe races.
        """
        async with self._start_lock:
            if self._started:
                return

            # Probe host resources
            try:
                import psutil  # noqa: PLC0415

                loop = asyncio.get_running_loop()
                vmem = await loop.run_in_executor(None, psutil.virtual_memory)
                cpu_count = await loop.run_in_executor(None, psutil.cpu_count)

                self._host_memory_mb = vmem.total / (1024 * 1024)
                self._host_cpu_count = float(cpu_count or 1)
                self._compute_budgets()

                reserve_mb = round(self._host_memory_mb * self._host_memory_reserve_ratio)
                logger.info(
                    "Host resources detected",
                    extra={
                        "host_memory_mb": round(self._host_memory_mb),
                        "host_cpu_count": self._host_cpu_count,
                        "host_memory_reserve_ratio": self._host_memory_reserve_ratio,
                        "host_memory_reserve_mb": reserve_mb,
                        "memory_budget_mb": round(self._memory_budget_mb),
                        "cpu_budget": round(self._cpu_budget, 1),
                        "memory_overcommit_ratio": self._memory_overcommit_ratio,
                        "cpu_overcommit_ratio": self._cpu_overcommit_ratio,
                    },
                )
            except (ImportError, OSError, AttributeError):
                logger.warning(
                    "psutil probe failed, using unlimited memory/CPU budgets (no admission control)",
                )
                self._memory_budget_mb = _UNLIMITED
                self._cpu_budget = _UNLIMITED

            self._started = True

    def _compute_budgets(self) -> None:
        """Compute effective budgets from host resources and overcommit ratios.

        Memory: host_total * (1 - reserve_ratio) * overcommit_ratio
        CPU:    host_cpu_count * overcommit_ratio
        """
        available_mb = self._host_memory_mb * (1.0 - self._host_memory_reserve_ratio)
        self._memory_budget_mb = max(0.0, available_mb) * self._memory_overcommit_ratio
        self._cpu_budget = self._host_cpu_count * self._cpu_overcommit_ratio

    async def acquire(
        self,
        vm_id: str,
        memory_mb: float,
        cpu_cores: float,
        timeout: float,
        use_tcg: bool = False,
    ) -> ResourceReservation:
        """Acquire resources for a VM, blocking if insufficient.

        Checks both gates under lock. Blocks on asyncio.Condition if
        insufficient resources. Raises VmCapacityError after timeout.

        Args:
            vm_id: VM identifier (for logging, does not need to be unique)
            memory_mb: Guest memory in MB (overhead is added automatically)
            cpu_cores: CPU cores to reserve
            timeout: Max seconds to wait for resources
            use_tcg: Whether TCG emulation is used (adds extra memory overhead)

        Returns:
            ResourceReservation tracking the allocated resources

        Raises:
            VmCapacityError: Resources not available within timeout
        """
        # Calculate total memory needed: guest + QEMU overhead (+TCG if applicable)
        total_memory_mb = memory_mb + CGROUP_MEMORY_OVERHEAD_MB
        if use_tcg:
            total_memory_mb += TCG_TB_CACHE_SIZE_MB

        reservation = ResourceReservation(
            vm_id=vm_id,
            memory_mb=total_memory_mb,
            cpu_cores=cpu_cores,
        )

        try:
            async with asyncio.timeout(timeout):
                async with self._condition:
                    await self._condition.wait_for(lambda: self._can_admit(total_memory_mb, cpu_cores))
                    # Resources available - commit reservation
                    self._allocated_memory_mb += total_memory_mb
                    self._allocated_cpu += cpu_cores
                    self._allocated_vm_slots += 1
                    self._reservations[reservation.reservation_id] = reservation

                    logger.debug(
                        "Resource reservation acquired",
                        extra={
                            "vm_id": vm_id,
                            "reservation_id": reservation.reservation_id,
                            "reserved_memory_mb": round(total_memory_mb),
                            "reserved_cpu": cpu_cores,
                            "total_allocated_memory_mb": round(self._allocated_memory_mb),
                            "total_allocated_cpu": round(self._allocated_cpu, 1),
                            "vm_slots": self._allocated_vm_slots,
                        },
                    )
                    return reservation

        except TimeoutError:
            mem_budget_str = "unlimited" if math.isinf(self._memory_budget_mb) else str(round(self._memory_budget_mb))
            cpu_budget_str = "unlimited" if math.isinf(self._cpu_budget) else str(round(self._cpu_budget, 1))
            raise VmCapacityError(
                f"Resource admission timeout after {timeout}s for VM {vm_id}. "
                f"Requested: {round(total_memory_mb)}MB memory, {cpu_cores} CPU. "
                f"Budget: {mem_budget_str}MB memory "
                f"({round(self._allocated_memory_mb)}/{mem_budget_str} allocated), "
                f"{cpu_budget_str} CPU "
                f"({round(self._allocated_cpu, 1)}/{cpu_budget_str} allocated), "
                f"{self._allocated_vm_slots} VM slots in use",
                context={
                    "vm_id": vm_id,
                    "requested_memory_mb": round(total_memory_mb),
                    "requested_cpu": cpu_cores,
                    "allocated_memory_mb": round(self._allocated_memory_mb),
                    "memory_budget_mb": mem_budget_str,
                    "allocated_cpu": round(self._allocated_cpu, 1),
                    "cpu_budget": cpu_budget_str,
                    "vm_slots": self._allocated_vm_slots,
                },
            ) from None

    async def release(self, reservation: ResourceReservation) -> None:
        """Release resources for a VM and signal waiters.

        Args:
            reservation: The reservation returned by acquire()
        """
        async with self._condition:
            if reservation.reservation_id not in self._reservations:
                logger.debug(
                    "Reservation already released (idempotent)",
                    extra={"vm_id": reservation.vm_id, "reservation_id": reservation.reservation_id},
                )
                return

            self._allocated_memory_mb -= reservation.memory_mb
            self._allocated_cpu -= reservation.cpu_cores
            self._allocated_vm_slots -= 1
            del self._reservations[reservation.reservation_id]

            # Snap counters to zero when empty (prevents IEEE 754 drift)
            if self._allocated_vm_slots == 0:
                self._allocated_memory_mb = 0.0
                self._allocated_cpu = 0.0

            logger.debug(
                "Resource reservation released",
                extra={
                    "vm_id": reservation.vm_id,
                    "reservation_id": reservation.reservation_id,
                    "freed_memory_mb": round(reservation.memory_mb),
                    "freed_cpu": reservation.cpu_cores,
                    "remaining_allocated_memory_mb": round(self._allocated_memory_mb),
                    "remaining_allocated_cpu": round(self._allocated_cpu, 1),
                    "vm_slots": self._allocated_vm_slots,
                },
            )

            # Wake ALL waiters - multiple may now fit
            self._condition.notify_all()

    def snapshot(self) -> ResourceSnapshot:
        """Return a point-in-time snapshot of admission state.

        SYNC-ONLY: No await points. Atomicity relies on asyncio single-thread
        cooperative scheduling. If this method ever becomes async, it MUST
        acquire self._condition before reading the allocation fields.
        """
        return ResourceSnapshot(
            host_memory_mb=self._host_memory_mb,
            host_cpu_count=self._host_cpu_count,
            memory_budget_mb=self._memory_budget_mb,
            cpu_budget=self._cpu_budget,
            allocated_memory_mb=self._allocated_memory_mb,
            allocated_cpu=self._allocated_cpu,
            allocated_vm_slots=self._allocated_vm_slots,
        )

    def effective_max_vms(
        self,
        guest_memory_mb: float,
        cpu_per_vm: float,
    ) -> int:
        """Compute how many VMs could theoretically fit given the current budget.

        Uses total budget (not available), i.e. ignores current allocations.
        Useful for sizing pools at startup before any VMs are running.

        Memory per VM is computed as guest_memory_mb + CGROUP_MEMORY_OVERHEAD_MB
        (same formula used by acquire()).

        Args:
            guest_memory_mb: Guest memory per VM (without cgroup overhead)
            cpu_per_vm: CPU cores per VM

        Returns:
            Effective max VMs, or -1 if budgets are unlimited (psutil unavailable)
        """
        if math.isinf(self._memory_budget_mb) or math.isinf(self._cpu_budget):
            return -1
        total_memory_per_vm = guest_memory_mb + CGROUP_MEMORY_OVERHEAD_MB
        max_by_memory = int(self._memory_budget_mb / total_memory_per_vm) if total_memory_per_vm > 0 else 0
        max_by_cpu = int(self._cpu_budget / cpu_per_vm) if cpu_per_vm > 0 else 0
        return min(max_by_memory, max_by_cpu)

    def _can_admit(self, memory_mb: float, cpu_cores: float) -> bool:
        """Check if both admission gates pass.

        Args:
            memory_mb: Total memory needed (guest + overhead)
            cpu_cores: CPU cores needed

        Returns:
            True if both gates pass
        """
        # Gate 1: Memory budget
        if self._allocated_memory_mb + memory_mb > self._memory_budget_mb:
            return False

        # Gate 2: CPU budget
        return self._allocated_cpu + cpu_cores <= self._cpu_budget

"""Resource-aware admission controller for VM scheduling.

Three admission gates (all must pass):
1. Memory budget - host_total * (1 - reserve_ratio) * overcommit_ratio
2. CPU budget - host_cpu_count * cpu_overcommit_ratio
3. Available-memory floor - reject when system available memory < threshold

Gates 1-2 are budget-based (stable). Gate 3 is dynamic (probed every 5s).
All gates block via asyncio.Condition, timing out with VmCapacityError.

Capacity detection priority chain:
  manual override > cgroup limit > psutil total

Graceful degradation: if psutil probe fails, budgets are set to infinity
(no admission control, equivalent to unlimited capacity).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from dataclasses import dataclass, field
from typing import Final, Literal
from uuid import uuid4

import psutil

from exec_sandbox._logging import get_logger
from exec_sandbox.cgroup import (
    CGROUP_MEMORY_OVERHEAD_MB,
    TCG_TB_CACHE_SIZE_MB,
    detect_cgroup_cpu_limit,
    detect_cgroup_memory_limit_mb,
    read_container_available_memory_mb,
)
from exec_sandbox.constants import AVAILABLE_MEMORY_PROBE_INTERVAL_SECONDS
from exec_sandbox.exceptions import VmCapacityError

logger = get_logger(__name__)

CapacitySource = Literal["manual", "cgroup", "psutil", "none", "unknown"]


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

    # Capacity source (for observability)
    capacity_source: CapacitySource = "psutil"

    # Gate 3: available-memory floor
    available_memory_floor_mb: int = 0
    system_available_memory_mb: float | None = None

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

    Three gates (all must pass):
    1. Memory budget: host_total * (1 - reserve_ratio) * overcommit_ratio
    2. CPU budget: host_cpus * overcommit_ratio
    3. Available-memory floor: system available memory >= threshold

    Gate 3 is disabled by default (available_memory_floor_mb=0).
    If psutil is unavailable, gates 1-2 degrade to unlimited.
    """

    def __init__(
        self,
        memory_overcommit_ratio: float,
        cpu_overcommit_ratio: float,
        host_memory_reserve_ratio: float,
        host_memory_mb: float | None = None,
        host_cpu_count: float | None = None,
        available_memory_floor_mb: int = 0,
    ) -> None:
        self._memory_overcommit_ratio = memory_overcommit_ratio
        self._cpu_overcommit_ratio = cpu_overcommit_ratio
        self._host_memory_reserve_ratio = host_memory_reserve_ratio
        self._available_memory_floor_mb = available_memory_floor_mb

        # Set by start() or constructor override (for testing)
        self._host_memory_mb: float = host_memory_mb if host_memory_mb is not None else 0.0
        self._host_cpu_count: float = host_cpu_count if host_cpu_count is not None else 0.0

        # Budget limits (computed in start())
        self._memory_budget_mb: float = _UNLIMITED
        self._cpu_budget: float = _UNLIMITED

        # Capacity source for observability
        self._capacity_source: CapacitySource = "unknown"

        # Current allocations (protected by _condition's lock)
        self._allocated_memory_mb: float = 0.0
        self._allocated_cpu: float = 0.0
        self._allocated_vm_slots: int = 0
        self._reservations: dict[str, ResourceReservation] = {}

        # Gate 3: available memory (updated by background probe)
        # None = not yet probed or not available
        self._system_available_memory_mb: float | None = None

        # Condition for blocking acquire / signaling release
        self._condition = asyncio.Condition()
        self._started = False
        self._start_lock = asyncio.Lock()

        # Background probe task (Gate 3)
        self._probe_task: asyncio.Task[None] | None = None

        # If host resources were provided at construction, compute budgets now
        if host_memory_mb is not None and host_cpu_count is not None:
            self._capacity_source = "manual"
            self._compute_budgets()
            self._started = True
            logger.warning(
                "Host resources manually overridden, admission budgets may not reflect actual capacity",
                extra={
                    "host_memory_mb": host_memory_mb,
                    "host_cpu_count": host_cpu_count,
                    "memory_budget_mb": round(self._memory_budget_mb),
                    "cpu_budget": round(self._cpu_budget, 1),
                },
            )

    async def start(self) -> None:  # noqa: PLR0912
        """Probe host resources and compute budgets.

        Detection priority: cgroup limit > psutil total.
        Manual overrides (host_memory_mb/host_cpu_count) skip this entirely.

        Safe to call multiple times (idempotent after first probe).
        If all probes fail, degrades gracefully to unlimited budgets.
        Serialized via asyncio.Lock to prevent concurrent probe races.
        """
        async with self._start_lock:
            if self._started:
                return

            # --- L1: Cgroup-aware capacity detection ---
            cgroup_mem = detect_cgroup_memory_limit_mb()
            cgroup_cpu = detect_cgroup_cpu_limit()

            if cgroup_mem is not None:
                self._host_memory_mb = cgroup_mem
                self._capacity_source = "cgroup"
                logger.info(
                    "Using cgroup memory limit for admission budget",
                    extra={"cgroup_memory_mb": round(cgroup_mem)},
                )

            if cgroup_cpu is not None:
                self._host_cpu_count = cgroup_cpu
                logger.info(
                    "Using cgroup CPU limit for admission budget",
                    extra={"cgroup_cpus": round(cgroup_cpu, 2)},
                )

            # --- Fallback to psutil for any missing dimension ---
            need_psutil_mem = cgroup_mem is None
            need_psutil_cpu = cgroup_cpu is None

            if need_psutil_mem or need_psutil_cpu:
                try:
                    loop = asyncio.get_running_loop()

                    if need_psutil_mem:
                        vmem = await loop.run_in_executor(None, psutil.virtual_memory)
                        self._host_memory_mb = vmem.total / (1024 * 1024)
                        if self._capacity_source == "unknown":
                            self._capacity_source = "psutil"

                    if need_psutil_cpu:
                        cpu_count = await loop.run_in_executor(None, psutil.cpu_count)
                        self._host_cpu_count = float(cpu_count or 1)

                    self._compute_budgets()

                    reserve_mb = round(self._host_memory_mb * self._host_memory_reserve_ratio)
                    logger.info(
                        "Host resources detected",
                        extra={
                            "capacity_source": self._capacity_source,
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
                except (OSError, AttributeError):
                    if cgroup_mem is not None or cgroup_cpu is not None:
                        # Partial cgroup detection - compute budgets with what we have
                        if cgroup_mem is None:
                            self._host_memory_mb = 0.0
                        if cgroup_cpu is None:
                            self._host_cpu_count = 1.0
                        self._compute_budgets()
                    else:
                        logger.warning(
                            "All resource probes failed, using unlimited budgets (no admission control)",
                        )
                        self._memory_budget_mb = _UNLIMITED
                        self._cpu_budget = _UNLIMITED
                        self._capacity_source = "none"
            else:
                # Both dimensions from cgroup
                self._compute_budgets()
                logger.info(
                    "Host resources detected (all from cgroup)",
                    extra={
                        "capacity_source": "cgroup",
                        "host_memory_mb": round(self._host_memory_mb),
                        "host_cpu_count": self._host_cpu_count,
                        "memory_budget_mb": round(self._memory_budget_mb),
                        "cpu_budget": round(self._cpu_budget, 1),
                    },
                )

            self._started = True

            # --- L3: Start background probe if floor is enabled ---
            if self._available_memory_floor_mb > 0:
                # Seed initial value synchronously so Gate 3 is active immediately
                # (not deferred until first background tick)
                self._system_available_memory_mb = await self._probe_available_memory()
                self._probe_task = asyncio.create_task(self._probe_available_memory_loop())
                logger.info(
                    "Available-memory floor enabled (Gate 3)",
                    extra={
                        "floor_mb": self._available_memory_floor_mb,
                        "probe_interval_seconds": AVAILABLE_MEMORY_PROBE_INTERVAL_SECONDS,
                        "initial_available_mb": (
                            round(self._system_available_memory_mb)
                            if self._system_available_memory_mb is not None
                            else None
                        ),
                    },
                )

    async def stop(self) -> None:
        """Stop the background memory probe task.

        Safe to call multiple times. No-op if probe was never started.
        """
        if self._probe_task is not None:
            self._probe_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._probe_task
            self._probe_task = None

    async def _probe_available_memory(self) -> float | None:
        """Probe system available memory (cgroup → psutil fallback).

        Tries cgroup first (sync file reads, fast). Falls back to
        psutil.virtual_memory via executor to avoid blocking the loop.

        Returns:
            Available memory in MB, or None if all probes fail.
        """
        avail = read_container_available_memory_mb()

        if avail is None:
            try:
                loop = asyncio.get_running_loop()
                vmem = await loop.run_in_executor(None, psutil.virtual_memory)
                avail = vmem.available / (1024 * 1024)
            except (OSError, AttributeError):
                avail = None

        if avail is None:
            logger.warning(
                "Failed to probe available memory (cgroup and psutil both failed), "
                "Gate 3 floor enforcement is degraded",
            )

        return avail

    async def _probe_available_memory_loop(self) -> None:
        """Background loop that refreshes system available memory.

        Runs every AVAILABLE_MEMORY_PROBE_INTERVAL_SECONDS.
        When memory recovers above floor, wakes blocked waiters.
        """
        while True:
            try:
                avail = await self._probe_available_memory()

                old_value = self._system_available_memory_mb
                self._system_available_memory_mb = avail

                # If memory recovered above floor, wake waiters
                if (
                    avail is not None
                    and avail >= self._available_memory_floor_mb
                    and (old_value is None or old_value < self._available_memory_floor_mb)
                ):
                    async with self._condition:
                        self._condition.notify_all()

            except (OSError, ValueError, AttributeError):
                logger.debug("Available memory probe failed", exc_info=True)

            await asyncio.sleep(AVAILABLE_MEMORY_PROBE_INTERVAL_SECONDS)

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

        Checks all three gates under lock. Blocks on asyncio.Condition if
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
            floor_info = ""
            if self._available_memory_floor_mb > 0:
                avail_str = (
                    str(round(self._system_available_memory_mb))
                    if self._system_available_memory_mb is not None
                    else "unknown"
                )
                floor_info = f" Floor: {avail_str}/{self._available_memory_floor_mb}MB available."
            raise VmCapacityError(
                f"Resource admission timeout after {timeout}s for VM {vm_id}. "
                f"Requested: {round(total_memory_mb)}MB memory, {cpu_cores} CPU. "
                f"Budget: {mem_budget_str}MB memory "
                f"({round(self._allocated_memory_mb)}/{mem_budget_str} allocated), "
                f"{cpu_budget_str} CPU "
                f"({round(self._allocated_cpu, 1)}/{cpu_budget_str} allocated), "
                f"{self._allocated_vm_slots} VM slots in use.{floor_info}",
                context={
                    "vm_id": vm_id,
                    "requested_memory_mb": round(total_memory_mb),
                    "requested_cpu": cpu_cores,
                    "allocated_memory_mb": round(self._allocated_memory_mb),
                    "memory_budget_mb": mem_budget_str,
                    "allocated_cpu": round(self._allocated_cpu, 1),
                    "cpu_budget": cpu_budget_str,
                    "vm_slots": self._allocated_vm_slots,
                    "available_memory_floor_mb": self._available_memory_floor_mb,
                    "system_available_memory_mb": (
                        round(self._system_available_memory_mb)
                        if self._system_available_memory_mb is not None
                        else None
                    ),
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
            capacity_source=self._capacity_source,
            available_memory_floor_mb=self._available_memory_floor_mb,
            system_available_memory_mb=self._system_available_memory_mb,
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
        """Check if all admission gates pass.

        Args:
            memory_mb: Total memory needed (guest + overhead)
            cpu_cores: CPU cores needed

        Returns:
            True if all gates pass
        """
        # Gate 1: Memory budget
        if self._allocated_memory_mb + memory_mb > self._memory_budget_mb:
            return False

        # Gate 2: CPU budget
        if self._allocated_cpu + cpu_cores > self._cpu_budget:
            return False

        # Gate 3: Available-memory floor (disabled when floor_mb == 0)
        return not (
            self._available_memory_floor_mb > 0
            and self._system_available_memory_mb is not None
            and self._system_available_memory_mb < self._available_memory_floor_mb
        )

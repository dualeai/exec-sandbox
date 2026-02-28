"""Resource-aware admission controller for VM scheduling.

Four admission gates (all must pass):
1. Memory budget - host_total * (1 - reserve_ratio) * overcommit_ratio
2. CPU budget - host_cpu_count * cpu_overcommit_ratio
3a. Available-memory floor - reject when system available memory is insufficient
3b. Memory pressure - reject when system is thrashing
    - Linux 4.20+: PSI full avg10 >= 10% from /proc/pressure/memory
    - macOS: kern.memorystatus_vm_pressure_level at CRITICAL

Gates 1-2 are budget-based (stable). Gates 3a/3b are dynamic (inline probe,
~3-5µs each, cached 100ms). A self-wake timer periodically notifies blocked
waiters so Gate 3 re-probes even without local release() events.
All gates block via asyncio.Condition, timing out with VmCapacityError.

Capacity detection priority chain:
  manual override > cgroup limit > psutil total

Graceful degradation: if psutil probe fails, budgets are set to infinity
(no admission control, equivalent to unlimited capacity).
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.util
import functools
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
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
from exec_sandbox.constants import GATE3_SELF_WAKE_INTERVAL_SECONDS
from exec_sandbox.exceptions import VmCapacityError
from exec_sandbox.platform_utils import HostOS, detect_host_os

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


# Sentinel for "no limit" when psutil probe fails
_UNLIMITED: Final[float] = float("inf")

# Probe cache TTL: collapses N identical probes into 1 when
# notify_all() wakes multiple waiters simultaneously
_PROBE_CACHE_TTL_SECONDS: Final[float] = 0.1

# Linux PSI threshold: percentage of last 10s ALL tasks were stalled on memory.
# 10% means 1 second out of 10 every task was blocked on memory I/O — serious thrashing.
_PSI_FULL_THRESHOLD_PCT: Final[float] = 10.0

# macOS memory pressure levels (from dispatch/source.h).
# Returned by kern.memorystatus_vm_pressure_level sysctl.
_MACOS_PRESSURE_NORMAL: Final[int] = 1
_MACOS_PRESSURE_WARN: Final[int] = 2
_MACOS_PRESSURE_CRITICAL: Final[int] = 4

# macOS: reject at CRITICAL only.  WARN is common during normal operation
# (active compressor, many apps open) — blocking at WARN makes the tool
# unusable on most dev machines.
_MACOS_REJECT_AT_LEVEL: Final[int] = _MACOS_PRESSURE_CRITICAL

# Lazily loaded libc handle for macOS sysctl (None on Linux)
_libc: ctypes.CDLL | None = None


def _read_macos_memory_pressure_level() -> int | None:
    """Read macOS kernel memory pressure level via sysctlbyname.

    Returns:
        1 (NORMAL), 2 (WARN), 4 (CRITICAL), or None on failure.
        Cost: ~3-5µs per call (no subprocess, direct syscall).
    """
    global _libc  # noqa: PLW0603
    if _libc is None:
        libc_path = ctypes.util.find_library("c")
        if libc_path is None:
            return None
        _libc = ctypes.CDLL(libc_path)

    val = ctypes.c_int32(0)
    sz = ctypes.c_size_t(ctypes.sizeof(val))
    ret = _libc.sysctlbyname(
        b"kern.memorystatus_vm_pressure_level",
        ctypes.byref(val),
        ctypes.byref(sz),
        None,
        ctypes.c_size_t(0),
    )
    if ret != 0:
        return None
    return val.value


def _ttl_cache(ttl_seconds: float):
    """Per-instance TTL cache for zero-arg methods.

    Stores ``(monotonic_timestamp, value)`` on the instance.
    *ttl_seconds* < 0 means infinite (never expires).
    """

    def decorator(func):
        attr = f"_ttl_{func.__name__}"

        @functools.wraps(func)
        def wrapper(self):
            now = time.monotonic()
            cached = getattr(self, attr, None)
            if cached is not None:
                ts, val = cached
                if ttl_seconds < 0 or now - ts < ttl_seconds:
                    return val
            result = func(self)
            setattr(self, attr, (now, result))
            return result

        return wrapper

    return decorator


class ResourceAdmissionController:
    """Resource-aware admission controller replacing asyncio.Semaphore.

    Four gates (all must pass):
    1. Memory budget: host_total * (1 - reserve_ratio) * overcommit_ratio
    2. CPU budget: host_cpus * overcommit_ratio
    3a. Available-memory floor: system_available - requested >= floor
    3b. Memory pressure: reject when system is thrashing
        (Linux: PSI full avg10 >= 10% / macOS: sysctl level == CRITICAL)

    Gates 3a/3b probe system memory inline (~3-5µs each, cached 100ms).
    A self-wake timer periodically notifies blocked waiters so Gate 3
    re-probes even when no local release() fires (external recovery).
    With floor=0 (default), 3a rejects only when the system literally
    cannot fit the requested VM — providing cross-process backpressure.
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

        # Condition for blocking acquire / signaling release
        self._condition = asyncio.Condition()
        self._started = False
        self._start_lock = asyncio.Lock()

        # Shutdown flag: set by stop() to unblock acquire() waiters immediately
        self._stopped = False

        # Self-wake timer: periodically notifies blocked waiters so Gate 3
        # re-probes system memory (closes wakeup gap for external recovery)
        self._self_wake_task: asyncio.Task[None] | None = None

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
            # Start self-wake timer if not already running (idempotent)
            if self._self_wake_task is None:
                self._self_wake_task = asyncio.create_task(self._self_wake_loop())
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

    async def stop(self) -> None:
        """Stop the admission controller and unblock all waiting acquire() callers.

        Sets a stopped flag and notifies all waiters so they raise
        VmCapacityError immediately instead of blocking until timeout.
        """
        self._stopped = True

        # Wake all blocked waiters so they see _stopped and exit
        async with self._condition:
            self._condition.notify_all()

        if self._self_wake_task is not None:
            self._self_wake_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._self_wake_task
            self._self_wake_task = None

    @_ttl_cache(_PROBE_CACHE_TTL_SECONDS)
    def _probe_available_memory(self) -> float | None:
        """Probe system available memory.

        Probe chain: cgroup → psutil → None (graceful degradation).
        """
        avail = read_container_available_memory_mb()
        if avail is None:
            try:
                vmem = psutil.virtual_memory()
                avail = vmem.available / (1024 * 1024)
            except (OSError, AttributeError, RuntimeError, psutil.Error):
                avail = None
        return avail

    @_ttl_cache(_PROBE_CACHE_TTL_SECONDS)
    def _probe_memory_pressure(self) -> bool | None:
        """Check if system memory pressure warrants rejecting admission.

        Each platform has its own decision logic — no shared threshold:

        - **Linux** (4.20+, CONFIG_PSI=y): reads ``/proc/pressure/memory``
          ``full avg10`` and rejects when >= 10% (serious thrashing).
        - **macOS**: reads ``kern.memorystatus_vm_pressure_level`` sysctl
          and rejects at CRITICAL (4). WARN (2) is common during normal
          operation and does not trigger rejection.

        Returns:
            True to reject (system under pressure), False to admit,
            None if unavailable (gate skipped).
            Cost: ~3-5µs per call (procfs read or sysctl, cached 100ms).

        Note:
            We opened https://github.com/giampaolo/psutil/issues/2725 to add
            cross-platform memory-pressure metrics to psutil. Once merged,
            replace ``_probe_psi_linux`` / ``_probe_pressure_macos`` /
            ``_read_macos_memory_pressure_level`` with a single psutil call.
        """
        host_os = detect_host_os()
        if host_os is HostOS.LINUX:
            return self._probe_psi_linux()
        if host_os is HostOS.MACOS:
            return self._probe_pressure_macos()
        return None

    @staticmethod
    def _probe_psi_linux() -> bool | None:
        """Read PSI full avg10 from /proc/pressure/memory (Linux).

        Returns True if pressure >= threshold (reject), False otherwise,
        None if PSI is unavailable.
        """
        try:
            content = Path("/proc/pressure/memory").read_text()
            for line in content.splitlines():
                if line.startswith("full "):
                    for token in line.split():
                        if token.startswith("avg10="):
                            val = float(token[6:])
                            if not math.isfinite(val):
                                return None
                            return val >= _PSI_FULL_THRESHOLD_PCT
        except (FileNotFoundError, OSError, ValueError):
            pass
        return None

    @staticmethod
    def _probe_pressure_macos() -> bool | None:
        """Read kernel memory pressure level via sysctl (macOS).

        Returns True at CRITICAL (reject), False at NORMAL/WARN (admit),
        None if sysctl fails or returns unknown level.
        """
        level = _read_macos_memory_pressure_level()
        if level is None:
            return None
        if level not in {_MACOS_PRESSURE_NORMAL, _MACOS_PRESSURE_WARN, _MACOS_PRESSURE_CRITICAL}:
            return None
        return level >= _MACOS_REJECT_AT_LEVEL

    async def _self_wake_loop(self) -> None:
        """Periodically notify blocked waiters so Gate 3 re-probes.

        Closes the wakeup gap: if external processes free memory and no
        local release() fires, blocked waiters would otherwise wait until
        timeout. This timer ensures they re-check every N seconds.
        """
        while True:
            await asyncio.sleep(GATE3_SELF_WAKE_INTERVAL_SECONDS)
            async with self._condition:
                self._condition.notify_all()

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

        logger.debug(
            "Acquiring resources",
            extra={
                "vm_id": vm_id,
                "requested_memory_mb": round(total_memory_mb),
                "requested_cpu": cpu_cores,
                "timeout": timeout,
                "allocated_memory_mb": round(self._allocated_memory_mb),
                "allocated_cpu": round(self._allocated_cpu, 1),
                "vm_slots": self._allocated_vm_slots,
            },
        )

        try:
            async with asyncio.timeout(timeout):
                async with self._condition:
                    await self._condition.wait_for(lambda: self._stopped or self._can_admit(total_memory_mb, cpu_cores))
                    if self._stopped:
                        raise VmCapacityError(
                            f"Admission controller stopped while waiting for VM {vm_id}",
                            context={"vm_id": vm_id},
                        )
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
            system_avail = self._probe_available_memory()
            if system_avail is not None:
                floor_info = f" System available: {round(system_avail)}MB (floor: {self._available_memory_floor_mb}MB)."
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
                    "system_available_memory_mb": (round(system_avail) if system_avail is not None else None),
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

    def effective_max_vms(
        self,
        guest_memory_mb: float,
        cpu_per_vm: float,
        use_tcg: bool = False,
    ) -> int:
        """Compute how many VMs could theoretically fit given the current budget.

        Uses total budget (not available), i.e. ignores current allocations.
        Useful for sizing pools at startup before any VMs are running.

        Memory per VM is computed as guest_memory_mb + CGROUP_MEMORY_OVERHEAD_MB
        (+ TCG_TB_CACHE_SIZE_MB when use_tcg=True), same formula used by acquire().

        Args:
            guest_memory_mb: Guest memory per VM (without cgroup overhead)
            cpu_per_vm: CPU cores per VM
            use_tcg: Whether TCG emulation is used (adds extra memory overhead)

        Returns:
            Effective max VMs, or -1 if both budgets are unlimited (psutil unavailable)
        """
        total_memory_per_vm = guest_memory_mb + CGROUP_MEMORY_OVERHEAD_MB
        if use_tcg:
            total_memory_per_vm += TCG_TB_CACHE_SIZE_MB

        # Compute per-dimension max, treating infinite budget as unbounded
        max_by_memory: int | None = None
        if not math.isinf(self._memory_budget_mb):
            max_by_memory = int(self._memory_budget_mb / total_memory_per_vm) if total_memory_per_vm > 0 else 0

        max_by_cpu: int | None = None
        if not math.isinf(self._cpu_budget):
            max_by_cpu = int(self._cpu_budget / cpu_per_vm) if cpu_per_vm > 0 else 0

        if max_by_memory is None and max_by_cpu is None:
            return -1  # Both unlimited
        if max_by_memory is None:
            return max_by_cpu  # type: ignore[return-value]
        if max_by_cpu is None:
            return max_by_memory
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
            logger.debug(
                "Gate 1 (memory budget) blocked",
                extra={
                    "requested_mb": round(memory_mb),
                    "allocated_mb": round(self._allocated_memory_mb),
                    "budget_mb": round(self._memory_budget_mb),
                },
            )
            return False

        # Gate 2: CPU budget
        if self._allocated_cpu + cpu_cores > self._cpu_budget:
            logger.debug(
                "Gate 2 (CPU budget) blocked",
                extra={
                    "requested_cpu": cpu_cores,
                    "allocated_cpu": round(self._allocated_cpu, 1),
                    "budget_cpu": round(self._cpu_budget, 1),
                },
            )
            return False

        # Gate 3a: System available memory (always-on, inline probe ~3µs, cached 100ms)
        # Provides cross-process backpressure: sees memory consumed by
        # other Scheduler instances, xdist workers, or external processes.
        system_available = self._probe_available_memory()
        if system_available is not None and system_available - memory_mb < self._available_memory_floor_mb:
            logger.debug(
                "Gate 3a (available-memory floor) blocked",
                extra={
                    "requested_mb": round(memory_mb),
                    "system_available_mb": round(system_available),
                    "after_admit_mb": round(system_available - memory_mb),
                    "floor_mb": self._available_memory_floor_mb,
                },
            )
            return False

        # Gate 3b: Memory pressure (Linux PSI / macOS sysctl, ~3-5µs, cached 100ms)
        # Each platform has its own rejection logic (no shared threshold).
        pressure = self._probe_memory_pressure()
        if pressure is True:
            logger.debug("Gate 3b (memory pressure) blocked")
            return False

        return True

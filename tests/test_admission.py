"""Tests for ResourceAdmissionController.

Tests the four admission gates (memory budget, CPU budget, available-memory floor,
memory pressure), blocking/signaling behavior, timeout, cgroup-aware startup,
probe cache, self-wake timer, and graceful degradation.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from exec_sandbox.admission import ResourceAdmissionController
from exec_sandbox.cgroup import CGROUP_MEMORY_OVERHEAD_MB, TCG_TB_CACHE_SIZE_MB
from exec_sandbox.constants import (
    DEFAULT_CPU_OVERCOMMIT_RATIO,
    DEFAULT_HOST_MEMORY_RESERVE_RATIO,
    DEFAULT_MEMORY_OVERCOMMIT_RATIO,
    DEFAULT_VM_CPU_CORES,
    RESOURCE_ADMISSION_TIMEOUT_SECONDS,
)
from exec_sandbox.exceptions import VmCapacityError
from tests.conftest import skip_unless_linux, skip_unless_macos

# ============================================================================
# Helpers
# ============================================================================

# Shorthand for test acquire calls
_CPU = DEFAULT_VM_CPU_CORES
_TIMEOUT = RESOURCE_ADMISSION_TIMEOUT_SECONDS


def _make_controller(
    host_memory_mb: float = 16_000.0,
    host_cpu_count: float = 8.0,
    memory_overcommit: float = DEFAULT_MEMORY_OVERCOMMIT_RATIO,
    cpu_overcommit: float = DEFAULT_CPU_OVERCOMMIT_RATIO,
    reserve_ratio: float = DEFAULT_HOST_MEMORY_RESERVE_RATIO,
    available_memory_floor_mb: int = 0,
) -> ResourceAdmissionController:
    """Create an admission controller with known host resources (no psutil)."""
    return ResourceAdmissionController(
        memory_overcommit_ratio=memory_overcommit,
        cpu_overcommit_ratio=cpu_overcommit,
        host_memory_reserve_ratio=reserve_ratio,
        host_memory_mb=host_memory_mb,
        host_cpu_count=host_cpu_count,
        available_memory_floor_mb=available_memory_floor_mb,
    )


def _invalidate_probe_caches(*controllers: ResourceAdmissionController) -> None:
    """Clear TTL probe caches so next call sees fresh system state."""
    for ctrl in controllers:
        for attr in list(vars(ctrl)):
            if attr.startswith("_ttl_"):
                delattr(ctrl, attr)


# ============================================================================
# Budget Calculation
# ============================================================================


def test_budget_calculation() -> None:
    """Memory budget = host_total * (1 - reserve_ratio) * overcommit."""
    ctrl = _make_controller(
        host_memory_mb=16_000.0,
        reserve_ratio=0.1,
        memory_overcommit=1.5,
        host_cpu_count=8.0,
        cpu_overcommit=4.0,
    )
    # 16000 * 0.9 * 1.5 = 21600
    assert ctrl._memory_budget_mb == pytest.approx(21_600.0)
    # 8 * 4 = 32
    assert ctrl._cpu_budget == pytest.approx(32.0)


def test_budget_reserve_ratio_scales() -> None:
    """Verify reserve ratio scales with host size."""
    # 4GB host, 10% reserve → 400MB reserved → 3600MB available → *1.5 = 5400
    ctrl_small = _make_controller(host_memory_mb=4_000.0, reserve_ratio=0.1, memory_overcommit=1.5)
    assert ctrl_small._memory_budget_mb == pytest.approx(5_400.0)

    # 64GB host, 10% reserve → 6400MB reserved → 57600MB available → *1.5 = 86400
    ctrl_large = _make_controller(host_memory_mb=64_000.0, reserve_ratio=0.1, memory_overcommit=1.5)
    assert ctrl_large._memory_budget_mb == pytest.approx(86_400.0)


# ============================================================================
# Gate 1: Memory Budget
# ============================================================================


async def test_memory_gate_blocks_when_budget_exhausted() -> None:
    """Acquire blocks when memory budget would be exceeded."""
    # 4GB host, 10% reserve, 1.0x overcommit → 3600MB budget
    # Each VM: 256 + CGROUP_MEMORY_OVERHEAD_MB overhead
    # Compute how many VMs fit, then the next one should block
    total_per_vm = 256 + CGROUP_MEMORY_OVERHEAD_MB
    budget_mb = 3_600  # 4000 * 0.9 * 1.0
    n_fit = budget_mb // total_per_vm  # VMs that fit in budget

    ctrl = _make_controller(
        host_memory_mb=4_000.0,
        reserve_ratio=0.1,
        memory_overcommit=1.0,
    )

    reservations = []
    for i in range(n_fit):
        r = await ctrl.acquire(f"vm-{i}", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        reservations.append(r)

    # Next should block (budget exhausted)
    acquire_task = asyncio.create_task(ctrl.acquire("vm-block", memory_mb=256, cpu_cores=_CPU, timeout=1.0))
    await asyncio.sleep(0.1)
    assert not acquire_task.done()

    # Release one → unblocks
    await ctrl.release(reservations[0])
    r_new = await asyncio.wait_for(acquire_task, timeout=5.0)
    assert r_new.vm_id == "vm-block"

    # Cleanup
    for r in reservations[1:]:
        await ctrl.release(r)
    await ctrl.release(r_new)


# ============================================================================
# Gate 2: CPU Budget
# ============================================================================


async def test_cpu_gate_blocks_when_budget_exhausted() -> None:
    """Acquire blocks when CPU budget would be exceeded."""
    # 2 host CPUs, 1.0x overcommit → 2.0 CPU budget
    # Each VM: 1.0 CPU → 2 fit
    ctrl = _make_controller(
        host_memory_mb=100_000.0,
        host_cpu_count=2.0,
        cpu_overcommit=1.0,
    )

    r1 = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
    r2 = await ctrl.acquire("vm-2", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)

    # 3rd should block
    acquire_task = asyncio.create_task(ctrl.acquire("vm-3", memory_mb=256, cpu_cores=_CPU, timeout=1.0))
    await asyncio.sleep(0.1)
    assert not acquire_task.done()

    await ctrl.release(r1)
    r3 = await asyncio.wait_for(acquire_task, timeout=5.0)
    assert r3.vm_id == "vm-3"

    await ctrl.release(r2)
    await ctrl.release(r3)


# ============================================================================
# CPU Budget - Multi-Core VMs
# ============================================================================


async def test_multi_core_vms_consume_proportional_budget() -> None:
    """2-core VMs consume 2x budget, so only 2 fit in a 4-core budget."""
    ctrl = _make_controller(
        host_memory_mb=100_000.0,
        host_cpu_count=4.0,
        cpu_overcommit=1.0,
    )

    r1 = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=2.0, timeout=_TIMEOUT)
    r2 = await ctrl.acquire("vm-2", memory_mb=256, cpu_cores=2.0, timeout=_TIMEOUT)

    # 3rd 2-core VM should block (4.0 budget fully used)
    acquire_task = asyncio.create_task(ctrl.acquire("vm-3", memory_mb=256, cpu_cores=2.0, timeout=0.3))
    await asyncio.sleep(0.1)
    assert not acquire_task.done()

    # Release one → unblocks
    await ctrl.release(r1)
    r3 = await asyncio.wait_for(acquire_task, timeout=5.0)
    assert r3.vm_id == "vm-3"

    await ctrl.release(r2)
    await ctrl.release(r3)


async def test_mixed_core_vms_share_budget() -> None:
    """Mix of 1-core and 2-core VMs shares the same budget correctly."""
    # 3 host CPUs, 1.0x overcommit → 3.0 CPU budget
    ctrl = _make_controller(
        host_memory_mb=100_000.0,
        host_cpu_count=3.0,
        cpu_overcommit=1.0,
    )

    # 1-core + 2-core = 3.0 (fits exactly)
    r1 = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=1.0, timeout=_TIMEOUT)
    r2 = await ctrl.acquire("vm-2", memory_mb=256, cpu_cores=2.0, timeout=_TIMEOUT)

    assert ctrl._allocated_cpu == pytest.approx(3.0)
    assert ctrl._cpu_budget - ctrl._allocated_cpu == pytest.approx(0.0)

    # Even a 1-core VM should block now
    acquire_task = asyncio.create_task(ctrl.acquire("vm-3", memory_mb=256, cpu_cores=1.0, timeout=0.3))
    await asyncio.sleep(0.1)
    assert not acquire_task.done()

    await ctrl.release(r1)
    r3 = await asyncio.wait_for(acquire_task, timeout=5.0)

    await ctrl.release(r2)
    await ctrl.release(r3)


async def test_overcommit_allows_more_cpu_than_physical() -> None:
    """CPU overcommit ratio allows admitting more cores than physical host has."""
    # 2 physical CPUs, 4.0x overcommit → 8.0 CPU budget
    ctrl = _make_controller(
        host_memory_mb=100_000.0,
        host_cpu_count=2.0,
        cpu_overcommit=4.0,
    )

    reservations = []
    for i in range(8):
        r = await ctrl.acquire(f"vm-{i}", memory_mb=256, cpu_cores=1.0, timeout=_TIMEOUT)
        reservations.append(r)

    assert ctrl._allocated_cpu == pytest.approx(8.0)
    assert ctrl._cpu_budget - ctrl._allocated_cpu == pytest.approx(0.0)

    # 9th should block
    acquire_task = asyncio.create_task(ctrl.acquire("vm-block", memory_mb=256, cpu_cores=1.0, timeout=0.3))
    await asyncio.sleep(0.1)
    assert not acquire_task.done()

    await ctrl.release(reservations[0])
    r_new = await asyncio.wait_for(acquire_task, timeout=5.0)

    for r in reservations[1:]:
        await ctrl.release(r)
    await ctrl.release(r_new)


# ============================================================================
# Timeout
# ============================================================================


async def test_acquire_timeout_raises_capacity_error() -> None:
    """VmCapacityError raised when timeout expires waiting for resources."""
    # Tight CPU budget: only 1 VM fits
    ctrl = _make_controller(host_cpu_count=1.0, cpu_overcommit=1.0)
    r1 = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)

    with pytest.raises(VmCapacityError, match="Resource admission timeout"):
        await ctrl.acquire("vm-2", memory_mb=256, cpu_cores=_CPU, timeout=0.2)

    await ctrl.release(r1)


# ============================================================================
# TCG Overhead
# ============================================================================


async def test_tcg_adds_extra_memory_overhead() -> None:
    """use_tcg=True adds TCG_TB_CACHE_SIZE_MB to the reservation."""
    ctrl = _make_controller()

    r_kvm = await ctrl.acquire("vm-kvm", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT, use_tcg=False)
    assert r_kvm.memory_mb == 256 + CGROUP_MEMORY_OVERHEAD_MB

    r_tcg = await ctrl.acquire("vm-tcg", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT, use_tcg=True)
    assert r_tcg.memory_mb == 256 + CGROUP_MEMORY_OVERHEAD_MB + TCG_TB_CACHE_SIZE_MB

    await ctrl.release(r_kvm)
    await ctrl.release(r_tcg)


# ============================================================================
# Release & Idempotency
# ============================================================================


async def test_release_frees_resources() -> None:
    """Release returns resources to the budget."""
    ctrl = _make_controller()
    before_slots = ctrl._allocated_vm_slots

    r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
    assert ctrl._allocated_vm_slots == 1

    await ctrl.release(r)
    assert ctrl._allocated_vm_slots == before_slots
    assert ctrl._allocated_memory_mb == pytest.approx(0.0)
    assert ctrl._allocated_cpu == pytest.approx(0.0)


async def test_double_release_is_idempotent() -> None:
    """Releasing the same reservation twice is safe (no-op)."""
    ctrl = _make_controller()
    r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
    await ctrl.release(r)
    # Second release should be a no-op
    await ctrl.release(r)
    assert ctrl._allocated_vm_slots == 0


# ============================================================================
# Availability
# ============================================================================


def test_availability_reports_correct_values() -> None:
    """Available memory and CPU are budget minus allocated."""
    ctrl = _make_controller()
    available_mem = ctrl._memory_budget_mb - ctrl._allocated_memory_mb
    available_cpu = ctrl._cpu_budget - ctrl._allocated_cpu
    assert available_mem == ctrl._memory_budget_mb
    assert available_cpu == ctrl._cpu_budget


# ============================================================================
# Reservation Key Uniqueness
# ============================================================================


async def test_concurrent_same_vm_id_no_collision() -> None:
    """Multiple reservations with the same vm_id get unique reservation_ids."""
    ctrl = _make_controller()

    # Acquire two reservations with the same vm_id (simulates concurrent requests)
    r1 = await ctrl.acquire("same-id", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
    r2 = await ctrl.acquire("same-id", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)

    # Both should have different reservation_ids
    assert r1.reservation_id != r2.reservation_id
    assert ctrl._allocated_vm_slots == 2

    # Release first — should NOT affect second
    await ctrl.release(r1)
    assert ctrl._allocated_vm_slots == 1

    # Release second
    await ctrl.release(r2)
    assert ctrl._allocated_vm_slots == 0
    assert ctrl._allocated_memory_mb == pytest.approx(0.0)
    assert ctrl._allocated_cpu == pytest.approx(0.0)


# ============================================================================
# Graceful Degradation (psutil failure)
# ============================================================================


async def test_psutil_failure_degrades_to_unlimited() -> None:
    """When psutil fails, budgets are unlimited (no admission control).

    Gate 3a/3b probes also fail → all four gates are bypassed.
    """
    # Construct WITHOUT providing host resources (simulates needing psutil)
    ctrl = ResourceAdmissionController(
        memory_overcommit_ratio=DEFAULT_MEMORY_OVERCOMMIT_RATIO,
        cpu_overcommit_ratio=DEFAULT_CPU_OVERCOMMIT_RATIO,
        host_memory_reserve_ratio=DEFAULT_HOST_MEMORY_RESERVE_RATIO,
    )
    # Manually set started (simulating psutil failure path)
    ctrl._memory_budget_mb = float("inf")
    ctrl._cpu_budget = float("inf")
    ctrl._started = True

    # Mock Gate 3a + 3b probes to fail (consistent with psutil-unavailable scenario)
    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=OSError("no psutil")),
        patch.object(type(ctrl), "_probe_memory_pressure", return_value=None),
    ):
        # Can acquire any amount of memory/CPU (no budget limit, no Gate 3a/3b)
        r1 = await ctrl.acquire("vm-1", memory_mb=10_000, cpu_cores=_CPU, timeout=_TIMEOUT)
        r2 = await ctrl.acquire("vm-2", memory_mb=10_000, cpu_cores=_CPU, timeout=_TIMEOUT)
        r3 = await ctrl.acquire("vm-3", memory_mb=10_000, cpu_cores=_CPU, timeout=_TIMEOUT)

    assert ctrl._allocated_vm_slots == 3

    # Release outside the mock context — probe returns real values but release doesn't call _can_admit
    await ctrl.release(r1)
    await ctrl.release(r2)
    await ctrl.release(r3)


# ============================================================================
# Float Drift Prevention
# ============================================================================


async def test_counters_snap_to_zero_when_empty() -> None:
    """After releasing all reservations, counters are exactly zero (no float drift)."""
    ctrl = _make_controller()

    # Acquire and release many times
    for i in range(50):
        r = await ctrl.acquire(f"vm-{i}", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        await ctrl.release(r)

    assert ctrl._allocated_vm_slots == 0
    assert ctrl._allocated_memory_mb == 0.0  # Exact zero, not approx
    assert ctrl._allocated_cpu == 0.0


# ============================================================================
# Cgroup-Aware Startup (L1)
# ============================================================================


async def test_cgroup_memory_limit_used_as_capacity() -> None:
    """When cgroup memory limit detected, it becomes host_memory_mb."""
    ctrl = ResourceAdmissionController(
        memory_overcommit_ratio=1.0,
        cpu_overcommit_ratio=1.0,
        host_memory_reserve_ratio=0.1,
    )

    with (
        patch("exec_sandbox.admission.detect_cgroup_memory_limit_mb", return_value=4096.0),
        patch("exec_sandbox.admission.detect_cgroup_cpu_limit", return_value=2.0),
    ):
        await ctrl.start()

    assert ctrl._host_memory_mb == pytest.approx(4096.0)
    assert ctrl._host_cpu_count == pytest.approx(2.0)
    assert ctrl._capacity_source == "cgroup"
    # Budget = 4096 * 0.9 * 1.0 = 3686.4
    assert ctrl._memory_budget_mb == pytest.approx(3686.4)


async def test_cgroup_falls_through_to_psutil_when_unlimited() -> None:
    """When cgroup returns None, psutil is used as fallback."""
    ctrl = ResourceAdmissionController(
        memory_overcommit_ratio=1.0,
        cpu_overcommit_ratio=1.0,
        host_memory_reserve_ratio=0.1,
    )

    mock_vmem = type("vmem", (), {"total": 8 * 1024 * 1024 * 1024})()  # 8GB
    with (
        patch("exec_sandbox.admission.detect_cgroup_memory_limit_mb", return_value=None),
        patch("exec_sandbox.admission.detect_cgroup_cpu_limit", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", return_value=mock_vmem),
        patch("exec_sandbox.admission.psutil.cpu_count", return_value=4),
    ):
        await ctrl.start()

    assert ctrl._host_memory_mb == pytest.approx(8192.0)
    assert ctrl._host_cpu_count == pytest.approx(4.0)
    assert ctrl._capacity_source == "psutil"


async def test_manual_override_beats_cgroup() -> None:
    """Manual host_memory_mb/host_cpu_count bypass cgroup + psutil."""
    ctrl = _make_controller(host_memory_mb=32_000.0, host_cpu_count=16.0)

    assert ctrl._host_memory_mb == pytest.approx(32_000.0)
    assert ctrl._host_cpu_count == pytest.approx(16.0)
    assert ctrl._capacity_source == "manual"


async def test_capacity_source_set_correctly() -> None:
    """Capacity source reports which method was used for detection."""
    ctrl = _make_controller()
    assert ctrl._capacity_source == "manual"


# ============================================================================
# Gate 3: Available-Memory Floor (inline probe)
# ============================================================================


def _mock_vmem(available_mb: float) -> object:
    """Create a mock psutil vmem result with given available MB."""
    return type("vmem", (), {"available": int(available_mb * 1024 * 1024)})()


@contextmanager
def _mock_system_memory(available_mb: float) -> Generator[None]:
    """Mock Gate 3 inline probe to report given available MB (no cgroup)."""
    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", return_value=_mock_vmem(available_mb)),
    ):
        yield


@contextmanager
def _mock_dynamic_system_memory(initial_mb: float) -> Generator[list[float]]:
    """Mock Gate 3 probe with mutable available memory.

    Returns a single-element list ``[mb]`` — mutate ``ref[0]`` to simulate
    system memory changing between acquire/release calls.
    """
    ref = [initial_mb]

    def _fake_vmem() -> object:
        return _mock_vmem(ref[0])

    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=_fake_vmem),
    ):
        yield ref


async def test_floor_rejects_when_system_cannot_fit_requested() -> None:
    """Gate 3 rejects when system available < requested + floor."""
    ctrl = _make_controller(available_memory_floor_mb=512)

    # System has 300MB available, request needs ~384MB + 512MB floor → reject
    with _mock_system_memory(300.0), pytest.raises(VmCapacityError, match="Resource admission timeout"):
        await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=0.2)


async def test_floor_allows_when_sufficient_available() -> None:
    """Gate 3 allows when system available - requested >= floor."""
    ctrl = _make_controller(available_memory_floor_mb=512)

    # System has 2048MB, request needs ~384MB, leaving 1664MB > 512MB floor → allow
    with _mock_system_memory(2048.0):
        r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        assert r.vm_id == "vm-1"
        await ctrl.release(r)


async def test_floor_zero_rejects_when_literally_no_memory() -> None:
    """Floor=0: rejects when system available < requested memory."""
    ctrl = _make_controller(available_memory_floor_mb=0)

    # System has 100MB available, request needs ~384MB → 100 - 384 < 0 → reject
    with _mock_system_memory(100.0), pytest.raises(VmCapacityError, match="Resource admission timeout"):
        await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=0.2)


async def test_floor_zero_allows_when_available_sufficient() -> None:
    """Floor=0: allows when system available >= requested memory."""
    ctrl = _make_controller(available_memory_floor_mb=0)

    # System has 2048MB, request needs ~384MB → 2048 - 384 >= 0 → allow
    with _mock_system_memory(2048.0):
        r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        assert r.vm_id == "vm-1"
        await ctrl.release(r)


async def test_probe_failure_allows_admission() -> None:
    """When all probes fail (return None), Gate 3 allows (graceful degradation)."""
    ctrl = _make_controller(available_memory_floor_mb=512)

    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=OSError("no psutil")),
    ):
        r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        assert r.vm_id == "vm-1"
        await ctrl.release(r)


async def test_cgroup_probe_preferred_over_psutil() -> None:
    """Gate 3 uses cgroup available memory when present."""
    ctrl = _make_controller(available_memory_floor_mb=0)

    # Cgroup reports 2048MB available — psutil should NOT be called
    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=2048.0),
        patch("exec_sandbox.admission.psutil.virtual_memory") as mock_psutil,
    ):
        r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        mock_psutil.assert_not_called()
        await ctrl.release(r)


async def test_floor_recovery_wakes_waiters() -> None:
    """When memory recovers, blocked waiters are unblocked."""
    ctrl = _make_controller(available_memory_floor_mb=512)

    # Start with low memory — acquire will block in wait_for loop
    with _mock_dynamic_system_memory(100.0) as available_mb:
        acquire_task = asyncio.create_task(ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=5.0))
        await asyncio.sleep(0.1)
        assert not acquire_task.done()

        # Simulate memory recovery + notify
        available_mb[0] = 2048.0
        async with ctrl._condition:
            ctrl._condition.notify_all()

        r = await asyncio.wait_for(acquire_task, timeout=5.0)
        assert r.vm_id == "vm-1"
        await ctrl.release(r)


async def test_floor_setting_accessible() -> None:
    """Available-memory floor setting is accessible on the controller."""
    ctrl = _make_controller(available_memory_floor_mb=512)

    # Trigger a probe by acquiring (which calls _can_admit → _probe_available_memory)
    with _mock_system_memory(2048.0):
        r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        assert ctrl._probe_available_memory() == pytest.approx(2048.0)

    assert ctrl._available_memory_floor_mb == 512
    await ctrl.release(r)


async def test_cross_process_backpressure_rejects_when_system_low() -> None:
    """Gate 3 provides cross-process backpressure even with budget available.

    Simulates the scenario where Gate 1/2 pass (per-process budget) but
    the system is under memory pressure from other processes.
    """
    # Large budget (Gates 1-2 will pass easily)
    ctrl = _make_controller(
        host_memory_mb=100_000.0,
        available_memory_floor_mb=0,
    )

    # System only has 200MB available (e.g. other xdist workers consumed memory)
    # Request needs ~384MB → 200 - 384 < 0 → reject
    with _mock_system_memory(200.0), pytest.raises(VmCapacityError, match="Resource admission timeout"):
        await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=0.2)


# ============================================================================
# Stop
# ============================================================================


async def test_stop_is_idempotent() -> None:
    """stop() is safe to call multiple times."""
    ctrl = _make_controller()
    # start() creates self-wake task; stop() cancels it
    await ctrl.start()
    await ctrl.stop()
    await ctrl.stop()  # Second stop is a no-op
    assert ctrl._self_wake_task is None


async def test_stop_cancels_self_wake_task() -> None:
    """stop() cancels the self-wake timer task."""
    ctrl = _make_controller()
    await ctrl.start()
    assert ctrl._self_wake_task is not None
    assert not ctrl._self_wake_task.done()

    await ctrl.stop()
    assert ctrl._self_wake_task is None


# ============================================================================
# Self-Wake Timer
# ============================================================================


async def test_self_wake_timer_unblocks_gate3_waiters() -> None:
    """Self-wake timer re-evaluates Gate 3 when external memory recovers."""
    ctrl = _make_controller(available_memory_floor_mb=0)
    # Start controller to create self-wake task
    await ctrl.start()

    with (
        _mock_dynamic_system_memory(100.0) as available_mb,  # initially low → Gate 3 rejects
        # Speed up self-wake for testing (100ms instead of 10s)
        patch("exec_sandbox.admission.GATE3_SELF_WAKE_INTERVAL_SECONDS", 0.1),
    ):
        # Restart self-wake with patched interval
        await ctrl.stop()
        ctrl._stopped = False  # Reset stopped flag so acquire() doesn't bail out
        ctrl._self_wake_task = asyncio.create_task(ctrl._self_wake_loop())

        # Acquire blocks because 100MB < 384MB needed
        acquire_task = asyncio.create_task(
            ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=5.0),
        )
        await asyncio.sleep(0.05)
        assert not acquire_task.done()

        # Simulate external memory recovery (no local release!)
        available_mb[0] = 2048.0

        # Self-wake timer should fire within ~100ms and unblock
        r = await asyncio.wait_for(acquire_task, timeout=2.0)
        assert r.vm_id == "vm-1"
        await ctrl.release(r)

    await ctrl.stop()


# ============================================================================
# Probe Cache
# ============================================================================


async def test_probe_cache_deduplicates_calls() -> None:
    """Probe cache returns cached value within 100ms TTL."""
    ctrl = _make_controller()
    call_count = 0
    original_vmem = _mock_vmem(4096.0)

    def counting_vmem():
        nonlocal call_count
        call_count += 1
        return original_vmem

    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=counting_vmem),
    ):
        # First probe — cache miss
        result1 = ctrl._probe_available_memory()
        assert result1 is not None
        assert call_count == 1

        # Second probe — cache hit (within 100ms)
        result2 = ctrl._probe_available_memory()
        assert result2 == result1
        assert call_count == 1  # psutil NOT called again


# ============================================================================
# Memory Pressure (Gate 3b)
# ============================================================================


class TestMemoryPressureGate:
    """Gate 3b: memory pressure detection (cross-platform).

    _probe_memory_pressure() returns bool | None:
      True = under pressure (reject), False = ok (admit), None = unavailable.
    Platform-specific tests use skip_unless_linux / skip_unless_macos.
    """

    # ---- Gate 3b integration (acquire/reject behavior) ----

    async def test_rejects_when_under_pressure(self) -> None:
        """Gate 3b rejects when _probe_memory_pressure returns True."""
        ctrl = _make_controller()

        with (
            _mock_system_memory(8192.0),
            patch.object(type(ctrl), "_probe_memory_pressure", return_value=True),
            pytest.raises(VmCapacityError, match="Resource admission timeout"),
        ):
            await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=0.2)

    async def test_allows_when_no_pressure(self) -> None:
        """Gate 3b allows when _probe_memory_pressure returns False."""
        ctrl = _make_controller()

        with (
            _mock_system_memory(8192.0),
            patch.object(type(ctrl), "_probe_memory_pressure", return_value=False),
        ):
            r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
            assert r.vm_id == "vm-1"
            await ctrl.release(r)

    async def test_allows_when_unavailable(self) -> None:
        """Gate 3b skipped when _probe_memory_pressure returns None."""
        ctrl = _make_controller()

        with (
            _mock_system_memory(8192.0),
            patch.object(type(ctrl), "_probe_memory_pressure", return_value=None),
        ):
            r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
            assert r.vm_id == "vm-1"
            await ctrl.release(r)

    async def test_pressure_recovery_unblocks_waiter(self) -> None:
        """Waiter blocked by Gate 3b is unblocked when pressure clears."""
        ctrl = _make_controller()
        under_pressure = [True]

        def fake_probe(_self: object) -> bool:
            return under_pressure[0]

        with (
            _mock_system_memory(8192.0),
            patch.object(type(ctrl), "_probe_memory_pressure", fake_probe),
        ):
            # Blocks because pressure=True
            acquire_task = asyncio.create_task(
                ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=5.0),
            )
            await asyncio.sleep(0.1)
            assert not acquire_task.done()

            # Pressure clears + notify
            under_pressure[0] = False
            async with ctrl._condition:
                ctrl._condition.notify_all()

            r = await asyncio.wait_for(acquire_task, timeout=5.0)
            assert r.vm_id == "vm-1"
            await ctrl.release(r)

    async def test_probe_caches_value(self) -> None:
        """Memory pressure probe returns cached value within TTL."""
        ctrl = _make_controller()

        with (
            _mock_system_memory(2048.0),
            patch.object(type(ctrl), "_probe_memory_pressure", return_value=False),
        ):
            r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
            assert ctrl._probe_memory_pressure() is False

        await ctrl.release(r)

    # ---- Dispatcher: _probe_memory_pressure() platform routing ----

    async def test_dispatcher_unknown_os_returns_none(self) -> None:
        """On unknown OS, _probe_memory_pressure returns None (gate skipped)."""
        ctrl = _make_controller()
        with patch("exec_sandbox.admission.detect_host_os", return_value=None):
            # Invalidate TTL cache so the probe runs fresh
            cache_attr = "_ttl__probe_memory_pressure"
            if hasattr(ctrl, cache_attr):
                delattr(ctrl, cache_attr)
            assert ctrl._probe_memory_pressure() is None

    # ---- Linux: PSI (Pressure Stall Information) ----

    @skip_unless_linux
    async def test_linux_psi_real_probe(self) -> None:
        """On real Linux, _probe_psi_linux() returns a bool or None."""
        result = ResourceAdmissionController._probe_psi_linux()
        assert result is None or isinstance(result, bool)

    @pytest.mark.parametrize(
        ("avg10", "expected"),
        [
            ("0.00", False),  # zero pressure
            ("9.99", False),  # just below threshold
            ("10.00", True),  # exactly at threshold (>=)
            ("10.01", True),  # just above threshold
            ("50.00", True),  # heavy pressure
            ("100.00", True),  # maximum
        ],
        ids=["zero", "just-below", "at-threshold", "just-above", "heavy", "max"],
    )
    async def test_linux_psi_threshold_boundary(self, avg10: str, expected: bool) -> None:
        """PSI full avg10 boundary values around the 10% threshold."""
        psi_content = (
            f"some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10={avg10} avg60=0.00 avg300=0.00 total=0\n"
        )
        with patch("exec_sandbox.admission.Path.read_text", return_value=psi_content):
            assert ResourceAdmissionController._probe_psi_linux() is expected

    @pytest.mark.parametrize(
        ("avg10", "reason"),
        [
            ("inf", "infinity"),
            ("-inf", "negative infinity"),
            ("nan", "not a number"),
        ],
        ids=["inf", "neg-inf", "nan"],
    )
    async def test_linux_psi_non_finite_returns_none(self, avg10: str, reason: str) -> None:
        """PSI probe returns None for non-finite values: {reason}."""
        psi_content = (
            f"some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10={avg10} avg60=0.00 avg300=0.00 total=0\n"
        )
        with patch("exec_sandbox.admission.Path.read_text", return_value=psi_content):
            assert ResourceAdmissionController._probe_psi_linux() is None

    @pytest.mark.parametrize(
        "content",
        [
            "",  # empty file
            "some avg10=1.00 avg60=0.00 avg300=0.00 total=0\n",  # no "full" line
            "garbage data that doesn't match any pattern",  # garbage
            "full\n",  # "full" line with no tokens
            "full avg60=5.00 avg300=2.00 total=0\n",  # "full" line but no avg10
        ],
        ids=["empty", "no-full-line", "garbage", "full-no-tokens", "full-no-avg10"],
    )
    async def test_linux_psi_malformed_content(self, content: str) -> None:
        """PSI probe returns None for malformed /proc/pressure/memory content."""
        with patch("exec_sandbox.admission.Path.read_text", return_value=content):
            assert ResourceAdmissionController._probe_psi_linux() is None

    @pytest.mark.parametrize(
        "exception",
        [FileNotFoundError, OSError("permission denied"), PermissionError],
        ids=["file-not-found", "os-error", "permission-denied"],
    )
    async def test_linux_psi_file_errors(self, exception: type | Exception) -> None:
        """PSI probe returns None for filesystem errors."""
        with patch("exec_sandbox.admission.Path.read_text", side_effect=exception):
            assert ResourceAdmissionController._probe_psi_linux() is None

    async def test_linux_psi_avg10_not_a_number(self) -> None:
        """PSI probe returns None when avg10 value is not parseable as float."""
        psi_content = "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=abc avg60=0.00 avg300=0.00 total=0\n"
        with patch("exec_sandbox.admission.Path.read_text", return_value=psi_content):
            assert ResourceAdmissionController._probe_psi_linux() is None

    # ---- macOS: kern.memorystatus_vm_pressure_level sysctl ----

    @skip_unless_macos
    async def test_macos_real_probe(self) -> None:
        """On real macOS, _probe_pressure_macos() returns a bool or None."""
        result = ResourceAdmissionController._probe_pressure_macos()
        assert result is None or isinstance(result, bool)

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (1, False),  # NORMAL → admit
            (2, False),  # WARN → admit (common during normal use)
            (4, True),  # CRITICAL → reject
        ],
        ids=["NORMAL", "WARN", "CRITICAL"],
    )
    async def test_macos_known_levels(self, level: int, expected: bool) -> None:
        """Known macOS pressure levels map to correct bool decisions."""
        with patch("exec_sandbox.admission._read_macos_memory_pressure_level", return_value=level):
            assert ResourceAdmissionController._probe_pressure_macos() is expected

    @pytest.mark.parametrize(
        "level",
        [0, 3, 5, 6, 7, 8, 16, -1, 2**31 - 1, 99],
        ids=[
            "zero",
            "between-warn-critical",
            "five",
            "six",
            "seven",
            "eight",
            "sixteen",
            "negative",
            "int32-max",
            "99",
        ],
    )
    async def test_macos_unexpected_levels_return_none(self, level: int) -> None:
        """Unknown/unexpected sysctl values return None (defensive)."""
        with patch("exec_sandbox.admission._read_macos_memory_pressure_level", return_value=level):
            assert ResourceAdmissionController._probe_pressure_macos() is None

    async def test_macos_sysctl_failure_returns_none(self) -> None:
        """Returns None when sysctl call fails."""
        with patch("exec_sandbox.admission._read_macos_memory_pressure_level", return_value=None):
            assert ResourceAdmissionController._probe_pressure_macos() is None


# ============================================================================
# Multi-Scheduler (cross-process backpressure)
# ============================================================================


async def test_two_controllers_share_system_memory() -> None:
    """Two controllers with full budgets are both limited by system available memory.

    This is the core scenario: two xdist workers each create a Scheduler with
    100% host budget.  Gate 1/2 pass for both, but Gate 3a sees the real system
    available memory shrinking as VMs are admitted across controllers.
    """
    # Both controllers think they own a 16GB machine
    ctrl_a = _make_controller(host_memory_mb=16_000.0)
    ctrl_b = _make_controller(host_memory_mb=16_000.0)

    # Simulate 2048MB system available — enough for a few VMs (384MB each)
    with _mock_dynamic_system_memory(2048.0) as available_mb:
        # Controller A grabs a VM — system available drops
        r_a = await ctrl_a.acquire("vm-a1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        available_mb[0] -= 384.0  # 2048 → 1664
        _invalidate_probe_caches(ctrl_a, ctrl_b)

        # Controller B grabs a VM — system available drops further
        r_b = await ctrl_b.acquire("vm-b1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        available_mb[0] -= 384.0  # 1664 → 1280
        _invalidate_probe_caches(ctrl_a, ctrl_b)

        # Both can still go (plenty of system memory left)
        r_a2 = await ctrl_a.acquire("vm-a2", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        available_mb[0] -= 384.0  # 1280 → 896
        _invalidate_probe_caches(ctrl_a, ctrl_b)

        r_b2 = await ctrl_b.acquire("vm-b2", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        available_mb[0] -= 384.0  # 896 → 512
        _invalidate_probe_caches(ctrl_a, ctrl_b)

        # Now system only has 512MB — next VM needs 384MB, leaving 128MB
        # With floor=0, 512 - 384 = 128 >= 0, so this still passes
        r_a3 = await ctrl_a.acquire("vm-a3", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        available_mb[0] -= 384.0  # 512 → 128
        _invalidate_probe_caches(ctrl_a, ctrl_b)

        # 128MB left — request needs 384MB → 128 - 384 < 0 → Gate 3a rejects
        # Gate 1/2 still pass (each controller has 21600MB budget, only used ~1152)
        with pytest.raises(VmCapacityError, match="Resource admission timeout"):
            await ctrl_b.acquire("vm-b3", memory_mb=256, cpu_cores=_CPU, timeout=0.2)

    await ctrl_a.release(r_a)
    await ctrl_a.release(r_a2)
    await ctrl_a.release(r_a3)
    await ctrl_b.release(r_b)
    await ctrl_b.release(r_b2)


async def test_controller_release_unblocks_other_controller() -> None:
    """When one controller releases, the other can proceed.

    Verifies cross-controller coordination via shared system memory.
    """
    ctrl_a = _make_controller(host_memory_mb=16_000.0)
    ctrl_b = _make_controller(host_memory_mb=16_000.0)

    with _mock_dynamic_system_memory(500.0) as available_mb:  # tight — only 1 VM fits
        # Controller A grabs the only VM that fits
        r_a = await ctrl_a.acquire("vm-a1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        available_mb[0] -= 384.0  # 500 → 116

        # Controller B blocks (116 - 384 < 0)
        acquire_b = asyncio.create_task(
            ctrl_b.acquire("vm-b1", memory_mb=256, cpu_cores=_CPU, timeout=5.0),
        )
        await asyncio.sleep(0.1)
        assert not acquire_b.done()

        # Controller A releases — system memory recovers
        await ctrl_a.release(r_a)
        available_mb[0] += 384.0  # 116 → 500

        # Notify B's condition (in real usage, self-wake timer does this)
        async with ctrl_b._condition:
            ctrl_b._condition.notify_all()

        r_b = await asyncio.wait_for(acquire_b, timeout=5.0)
        assert r_b.vm_id == "vm-b1"
        await ctrl_b.release(r_b)


async def test_n_controllers_converge_to_system_limit() -> None:
    """N controllers each claiming full budget converge to the real system capacity.

    With 4 controllers and 2048MB system memory, at most ~5 VMs total (384MB each)
    should be admitted regardless of per-controller budgets.
    """
    n_controllers = 4
    controllers = [_make_controller(host_memory_mb=16_000.0) for _ in range(n_controllers)]

    total_admitted = 0
    reservations: list[tuple[ResourceAdmissionController, object]] = []

    with _mock_dynamic_system_memory(2048.0) as available_mb:
        # Round-robin across controllers until system is full
        for i in range(20):  # try up to 20 VMs
            ctrl = controllers[i % n_controllers]
            _invalidate_probe_caches(*controllers)
            try:
                r = await ctrl.acquire(
                    f"vm-{i}",
                    memory_mb=256,
                    cpu_cores=_CPU,
                    timeout=0.2,
                )
                available_mb[0] -= 384.0
                total_admitted += 1
                reservations.append((ctrl, r))
            except VmCapacityError:
                break  # System full

    # Per-controller budget allows 56 VMs each (224 total), but system only fits ~5
    assert total_admitted == 5  # 2048 / 384 = 5.33 → 5
    assert available_mb[0] == pytest.approx(2048.0 - 5 * 384.0)

    for ctrl, r in reservations:
        await ctrl.release(r)


async def test_gate1_still_limits_single_controller() -> None:
    """Gate 1/2 still enforces per-controller budget even with plenty of system memory.

    Ensures system memory probe doesn't bypass the per-controller budget —
    both gates must pass.
    """
    # Tiny budget: only 2 VMs fit per Gate 1
    ctrl = _make_controller(
        host_memory_mb=1_000.0,
        memory_overcommit=1.0,
        reserve_ratio=0.0,
    )
    # Budget = 1000MB, per-VM = 384MB → max 2 VMs

    with _mock_system_memory(100_000.0):  # system has plenty
        r1 = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        r2 = await ctrl.acquire("vm-2", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)

        # Third VM: Gate 3a passes (100GB available) but Gate 1 rejects (768/1000 + 384 > 1000)
        with pytest.raises(VmCapacityError, match="Resource admission timeout"):
            await ctrl.acquire("vm-3", memory_mb=256, cpu_cores=_CPU, timeout=0.2)

    await ctrl.release(r1)
    await ctrl.release(r2)


async def test_floor_adds_headroom_in_multi_controller() -> None:
    """Non-zero floor reserves headroom that all controllers respect.

    With floor=512, controllers stop admitting earlier — leaving breathing room
    for OS/other services.
    """
    ctrl_a = _make_controller(host_memory_mb=16_000.0, available_memory_floor_mb=512)
    ctrl_b = _make_controller(host_memory_mb=16_000.0, available_memory_floor_mb=512)

    with _mock_dynamic_system_memory(2048.0) as available_mb:
        reservations = []
        admitted = 0

        for i in range(10):
            ctrl = ctrl_a if i % 2 == 0 else ctrl_b
            _invalidate_probe_caches(ctrl_a, ctrl_b)
            try:
                r = await ctrl.acquire(f"vm-{i}", memory_mb=256, cpu_cores=_CPU, timeout=0.2)
                available_mb[0] -= 384.0
                admitted += 1
                reservations.append((ctrl, r))
            except VmCapacityError:
                break

    # floor=512: reject when available - 384 < 512, i.e. available < 896
    # 2048 → 1664 → 1280 → 896 → STOP (896 - 384 = 512, exactly at floor, passes)
    # → 512 → STOP (512 - 384 = 128 < 512, reject)
    # So 4 VMs admitted
    assert admitted == 4
    assert available_mb[0] == pytest.approx(2048.0 - 4 * 384.0)

    for ctrl, r in reservations:
        await ctrl.release(r)


async def test_probe_failure_removes_cross_process_protection() -> None:
    """When probes fail, Gate 3a is skipped — controllers lose cross-process visibility.

    This is the degraded mode: each controller falls back to per-process budget only.
    """
    ctrl_a = _make_controller(host_memory_mb=16_000.0)
    ctrl_b = _make_controller(host_memory_mb=16_000.0)

    # Both probes fail
    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=OSError("broken")),
        patch.object(type(ctrl_a), "_probe_memory_pressure", return_value=None),
        patch.object(type(ctrl_b), "_probe_memory_pressure", return_value=None),
    ):
        # Both can acquire freely — no cross-process protection
        r_a = await ctrl_a.acquire("vm-a", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
        r_b = await ctrl_b.acquire("vm-b", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)

    await ctrl_a.release(r_a)
    await ctrl_b.release(r_b)


# ============================================================================
# Shutdown / Stop
# ============================================================================


async def test_stop_unblocks_waiting_acquires() -> None:
    """stop() immediately unblocks blocked acquire() callers with VmCapacityError."""
    # Tight budget: only 1 VM fits
    ctrl = _make_controller(host_cpu_count=1.0, cpu_overcommit=1.0)

    r1 = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=1.0, timeout=_TIMEOUT)

    # Second acquire blocks (no CPU left)
    acquire_task = asyncio.create_task(
        ctrl.acquire("vm-2", memory_mb=256, cpu_cores=1.0, timeout=120.0),
    )
    await asyncio.sleep(0.05)
    assert not acquire_task.done()

    # stop() should immediately unblock the waiter
    await ctrl.stop()

    with pytest.raises(VmCapacityError, match="stopped"):
        await asyncio.wait_for(acquire_task, timeout=2.0)

    await ctrl.release(r1)


# ============================================================================
# effective_max_vms
# ============================================================================


def test_effective_max_vms_basic() -> None:
    """effective_max_vms returns correct value for standard config."""
    ctrl = _make_controller(host_memory_mb=16_000.0, host_cpu_count=8.0)
    result = ctrl.effective_max_vms(guest_memory_mb=256, cpu_per_vm=1.0)
    # Memory: 16000 * 0.9 * 1.5 = 21600, per-VM = 256+128=384, max_mem = 56
    # CPU: 8 * 4.0 = 32, per-VM = 1.0, max_cpu = 32
    # min(56, 32) = 32
    assert result == 32


def test_effective_max_vms_with_tcg() -> None:
    """effective_max_vms accounts for TCG overhead when use_tcg=True."""
    # Use high CPU count so memory is the bottleneck (not CPU)
    ctrl = _make_controller(host_memory_mb=16_000.0, host_cpu_count=100.0)
    result_kvm = ctrl.effective_max_vms(guest_memory_mb=256, cpu_per_vm=1.0, use_tcg=False)
    result_tcg = ctrl.effective_max_vms(guest_memory_mb=256, cpu_per_vm=1.0, use_tcg=True)
    # KVM: per-VM = 256+128=384, budget=21600, max_mem=56, max_cpu=400 → 56
    # TCG: per-VM = 256+128+256=640, budget=21600, max_mem=33, max_cpu=400 → 33
    assert result_tcg < result_kvm
    assert result_kvm == 56
    assert result_tcg == 33


def test_effective_max_vms_one_dimension_unlimited() -> None:
    """effective_max_vms uses the finite dimension when only one is unlimited."""
    ctrl = _make_controller(host_memory_mb=16_000.0, host_cpu_count=8.0)
    # Manually set memory unlimited, CPU finite
    ctrl._memory_budget_mb = float("inf")
    result = ctrl.effective_max_vms(guest_memory_mb=256, cpu_per_vm=1.0)
    # CPU: 8 * 4.0 = 32
    assert result == 32

    # Manually set CPU unlimited, memory finite
    ctrl._memory_budget_mb = 21600.0
    ctrl._cpu_budget = float("inf")
    result = ctrl.effective_max_vms(guest_memory_mb=256, cpu_per_vm=1.0)
    # Memory: 21600 / 384 = 56
    assert result == 56


def test_effective_max_vms_both_unlimited() -> None:
    """effective_max_vms returns -1 when both budgets are unlimited."""
    ctrl = _make_controller(host_memory_mb=16_000.0, host_cpu_count=8.0)
    ctrl._memory_budget_mb = float("inf")
    ctrl._cpu_budget = float("inf")
    assert ctrl.effective_max_vms(guest_memory_mb=256, cpu_per_vm=1.0) == -1


async def test_psutil_error_in_probe_degrades_gracefully() -> None:
    """psutil.Error and RuntimeError in _probe_available_memory are caught."""
    import psutil as _psutil

    ctrl = _make_controller()

    # psutil.Error (covers AccessDenied, etc.)
    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=_psutil.Error("access denied")),
    ):
        assert ctrl._probe_available_memory() is None

    # RuntimeError (observed on some platforms)
    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=RuntimeError("platform error")),
    ):
        assert ctrl._probe_available_memory() is None

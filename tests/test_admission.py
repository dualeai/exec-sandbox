"""Tests for ResourceAdmissionController.

Tests the three admission gates (memory budget, CPU budget, available-memory floor),
blocking/signaling behavior, timeout, cgroup-aware startup, and graceful degradation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

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
    snap = ctrl.snapshot()
    # 16000 * 0.9 * 1.5 = 21600
    assert snap.memory_budget_mb == pytest.approx(21_600.0)
    # 8 * 4 = 32
    assert snap.cpu_budget == pytest.approx(32.0)


def test_budget_reserve_ratio_scales() -> None:
    """Verify reserve ratio scales with host size."""
    # 4GB host, 10% reserve → 400MB reserved → 3600MB available → *1.5 = 5400
    ctrl_small = _make_controller(host_memory_mb=4_000.0, reserve_ratio=0.1, memory_overcommit=1.5)
    assert ctrl_small.snapshot().memory_budget_mb == pytest.approx(5_400.0)

    # 64GB host, 10% reserve → 6400MB reserved → 57600MB available → *1.5 = 86400
    ctrl_large = _make_controller(host_memory_mb=64_000.0, reserve_ratio=0.1, memory_overcommit=1.5)
    assert ctrl_large.snapshot().memory_budget_mb == pytest.approx(86_400.0)


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

    snap = ctrl.snapshot()
    assert snap.allocated_cpu == pytest.approx(3.0)
    assert snap.available_cpu == pytest.approx(0.0)

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

    snap = ctrl.snapshot()
    assert snap.allocated_cpu == pytest.approx(8.0)
    assert snap.available_cpu == pytest.approx(0.0)

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
    snap_before = ctrl.snapshot()

    r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
    snap_during = ctrl.snapshot()
    assert snap_during.allocated_vm_slots == 1

    await ctrl.release(r)
    snap_after = ctrl.snapshot()
    assert snap_after.allocated_vm_slots == snap_before.allocated_vm_slots
    assert snap_after.allocated_memory_mb == pytest.approx(snap_before.allocated_memory_mb)
    assert snap_after.allocated_cpu == pytest.approx(snap_before.allocated_cpu)


async def test_double_release_is_idempotent() -> None:
    """Releasing the same reservation twice is safe (no-op)."""
    ctrl = _make_controller()
    r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
    await ctrl.release(r)
    # Second release should be a no-op
    await ctrl.release(r)
    assert ctrl.snapshot().allocated_vm_slots == 0


# ============================================================================
# Snapshot
# ============================================================================


def test_snapshot_reports_availability() -> None:
    """Snapshot computed fields (available_*) are correct."""
    ctrl = _make_controller()
    snap = ctrl.snapshot()
    assert snap.available_memory_mb == snap.memory_budget_mb
    assert snap.available_cpu == snap.cpu_budget


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
    assert ctrl.snapshot().allocated_vm_slots == 2

    # Release first — should NOT affect second
    await ctrl.release(r1)
    assert ctrl.snapshot().allocated_vm_slots == 1

    # Release second
    await ctrl.release(r2)
    assert ctrl.snapshot().allocated_vm_slots == 0
    assert ctrl.snapshot().allocated_memory_mb == pytest.approx(0.0)
    assert ctrl.snapshot().allocated_cpu == pytest.approx(0.0)


# ============================================================================
# Graceful Degradation (psutil failure)
# ============================================================================


async def test_psutil_failure_degrades_to_unlimited() -> None:
    """When psutil fails, budgets are unlimited (no admission control)."""
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

    # Can acquire any amount of memory/CPU (no budget limit)
    r1 = await ctrl.acquire("vm-1", memory_mb=10_000, cpu_cores=_CPU, timeout=_TIMEOUT)
    r2 = await ctrl.acquire("vm-2", memory_mb=10_000, cpu_cores=_CPU, timeout=_TIMEOUT)
    r3 = await ctrl.acquire("vm-3", memory_mb=10_000, cpu_cores=_CPU, timeout=_TIMEOUT)

    assert ctrl.snapshot().allocated_vm_slots == 3

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

    snap = ctrl.snapshot()
    assert snap.allocated_vm_slots == 0
    assert snap.allocated_memory_mb == 0.0  # Exact zero, not approx
    assert snap.allocated_cpu == 0.0


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

    snap = ctrl.snapshot()
    assert snap.host_memory_mb == pytest.approx(4096.0)
    assert snap.host_cpu_count == pytest.approx(2.0)
    assert snap.capacity_source == "cgroup"
    # Budget = 4096 * 0.9 * 1.0 = 3686.4
    assert snap.memory_budget_mb == pytest.approx(3686.4)


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

    snap = ctrl.snapshot()
    assert snap.host_memory_mb == pytest.approx(8192.0)
    assert snap.host_cpu_count == pytest.approx(4.0)
    assert snap.capacity_source == "psutil"


async def test_manual_override_beats_cgroup() -> None:
    """Manual host_memory_mb/host_cpu_count bypass cgroup + psutil."""
    ctrl = _make_controller(host_memory_mb=32_000.0, host_cpu_count=16.0)

    snap = ctrl.snapshot()
    assert snap.host_memory_mb == pytest.approx(32_000.0)
    assert snap.host_cpu_count == pytest.approx(16.0)
    assert snap.capacity_source == "manual"


async def test_snapshot_includes_capacity_source() -> None:
    """ResourceSnapshot reports which source was used for capacity detection."""
    ctrl = _make_controller()
    snap = ctrl.snapshot()
    assert snap.capacity_source == "manual"


# ============================================================================
# Gate 3: Available-Memory Floor
# ============================================================================


async def test_floor_rejects_when_available_below_threshold() -> None:
    """Gate 3 rejects when system available memory < floor."""
    ctrl = _make_controller(available_memory_floor_mb=512)
    # Simulate low available memory
    ctrl._system_available_memory_mb = 256.0

    with pytest.raises(VmCapacityError, match="Resource admission timeout"):
        await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=0.2)


async def test_floor_allows_when_available_above_threshold() -> None:
    """Gate 3 allows when system available memory >= floor."""
    ctrl = _make_controller(available_memory_floor_mb=512)
    ctrl._system_available_memory_mb = 1024.0

    r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
    assert r.vm_id == "vm-1"
    await ctrl.release(r)


async def test_floor_zero_disables_gate3() -> None:
    """Floor=0 disables Gate 3 (backward compat)."""
    ctrl = _make_controller(available_memory_floor_mb=0)
    # Even with very low available memory, Gate 3 is skipped
    ctrl._system_available_memory_mb = 10.0

    r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
    assert r.vm_id == "vm-1"
    await ctrl.release(r)


async def test_floor_none_available_allows_admission() -> None:
    """When available memory hasn't been probed (None), Gate 3 allows."""
    ctrl = _make_controller(available_memory_floor_mb=512)
    assert ctrl._system_available_memory_mb is None

    r = await ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=_TIMEOUT)
    assert r.vm_id == "vm-1"
    await ctrl.release(r)


async def test_floor_recovery_wakes_waiters() -> None:
    """When memory recovers above floor, blocked waiters are unblocked."""
    ctrl = _make_controller(available_memory_floor_mb=512)
    ctrl._system_available_memory_mb = 100.0  # Below floor

    # Start blocked acquire
    acquire_task = asyncio.create_task(ctrl.acquire("vm-1", memory_mb=256, cpu_cores=_CPU, timeout=5.0))
    await asyncio.sleep(0.1)
    assert not acquire_task.done()

    # Simulate memory recovery + notify
    ctrl._system_available_memory_mb = 1024.0
    async with ctrl._condition:
        ctrl._condition.notify_all()

    r = await asyncio.wait_for(acquire_task, timeout=5.0)
    assert r.vm_id == "vm-1"
    await ctrl.release(r)


async def test_snapshot_includes_floor_fields() -> None:
    """ResourceSnapshot includes Gate 3 fields."""
    ctrl = _make_controller(available_memory_floor_mb=512)
    ctrl._system_available_memory_mb = 2048.0

    snap = ctrl.snapshot()
    assert snap.available_memory_floor_mb == 512
    assert snap.system_available_memory_mb == pytest.approx(2048.0)


# ============================================================================
# Stop
# ============================================================================


async def test_stop_cancels_probe_task() -> None:
    """stop() cancels the background probe task."""
    ctrl = _make_controller(available_memory_floor_mb=512)

    # Simulate a running probe task
    async def fake_probe() -> None:
        await asyncio.sleep(3600)

    ctrl._probe_task = asyncio.create_task(fake_probe())
    await ctrl.stop()
    assert ctrl._probe_task is None


async def test_stop_is_idempotent() -> None:
    """stop() is safe to call multiple times."""
    ctrl = _make_controller()
    await ctrl.stop()
    await ctrl.stop()  # No error

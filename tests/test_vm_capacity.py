"""Tests for VM capacity enforcement (admission-based concurrency control).

These tests verify that the VmManager correctly enforces resource budgets
using the ResourceAdmissionController:
- Resources acquired on VM creation
- Resources released on VM destruction
- Proper blocking behavior when at capacity
- No double-release bugs
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.exceptions import VmCapacityError, VmOverlayError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.settings import Settings
from exec_sandbox.vm_manager import VmManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def capacity_settings(images_dir: Path) -> Settings:
    """Settings with tight CPU budget for testing capacity.

    Uses host_cpu_count=2.0 with cpu_overcommit=1.0 so that only
    2 VMs (at 1.0 CPU each) can be admitted concurrently.
    Memory is large enough to not be the bottleneck.
    """
    return Settings(
        base_images_dir=images_dir,
        kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
        host_memory_mb=100_000.0,  # Large, not the bottleneck
        host_cpu_count=2.0,  # Tight: 2 VMs fit, 3 don't
        cpu_overcommit_ratio=1.0,  # No overcommit
    )


@pytest.fixture
async def capacity_vm_manager(capacity_settings: Settings) -> AsyncGenerator[VmManager, None]:
    """VmManager with tight CPU budget (2 VMs max) for capacity testing."""
    async with VmManager(capacity_settings) as manager:
        yield manager


# ============================================================================
# Integration Tests - Concurrent VM Capacity
# ============================================================================


async def test_create_vm_blocks_at_capacity(capacity_vm_manager: VmManager) -> None:
    """When at capacity, create_vm blocks until slot available (not fails).

    The admission controller should cause blocking behavior rather than raising
    VmCapacityError immediately.
    """
    # Create first VM (at limit since CPU budget=2.0, each VM=1.0)
    vm1 = await capacity_vm_manager.create_vm(
        language=Language.PYTHON,
        tenant_id="test",
        task_id="task-1",
    )

    vm2 = await capacity_vm_manager.create_vm(
        language=Language.PYTHON,
        tenant_id="test",
        task_id="task-2",
    )

    # Start third creation (should block, not fail)
    create_task = asyncio.create_task(
        capacity_vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="task-3",
        )
    )

    # Give it time to potentially complete (should stay blocked)
    await asyncio.sleep(0.5)
    assert not create_task.done(), "Should be blocked waiting for resources, not completed"

    # Destroy one VM (releases resources)
    await capacity_vm_manager.destroy_vm(vm1)

    # Third creation should now complete
    vm3 = await asyncio.wait_for(create_task, timeout=120)
    assert vm3 is not None
    assert vm3.holds_semaphore_slot is True

    # Cleanup
    await capacity_vm_manager.destroy_vm(vm2)
    await capacity_vm_manager.destroy_vm(vm3)


async def test_admission_released_on_destroy(capacity_vm_manager: VmManager) -> None:
    """Resources must be released when VM is destroyed."""
    snap_before = capacity_vm_manager.admission.snapshot()

    # Create VM (acquires resources)
    vm = await capacity_vm_manager.create_vm(
        language=Language.PYTHON,
        tenant_id="test",
        task_id="task-1",
    )

    snap_during = capacity_vm_manager.admission.snapshot()
    assert snap_during.allocated_vm_slots == snap_before.allocated_vm_slots + 1
    assert vm.holds_semaphore_slot is True

    # Destroy VM (releases resources)
    await capacity_vm_manager.destroy_vm(vm)

    snap_after = capacity_vm_manager.admission.snapshot()
    assert snap_after.allocated_vm_slots == snap_before.allocated_vm_slots
    assert vm.holds_semaphore_slot is False


async def test_admission_released_on_create_failure(capacity_vm_manager: VmManager) -> None:
    """Resources must be released when create_vm fails mid-way."""
    snap_before = capacity_vm_manager.admission.snapshot()

    # Mock _create_vm_impl to fail after resources acquired
    with patch.object(capacity_vm_manager, "_create_vm_impl", side_effect=VmOverlayError("mock failure")):
        with pytest.raises(VmOverlayError, match="mock failure"):
            await capacity_vm_manager.create_vm(
                language=Language.PYTHON,
                tenant_id="test",
                task_id="task-1",
            )

    # Resources should be restored
    snap_after = capacity_vm_manager.admission.snapshot()
    assert snap_after.allocated_vm_slots == snap_before.allocated_vm_slots


async def test_double_destroy_does_not_double_release(capacity_vm_manager: VmManager) -> None:
    """Calling destroy_vm twice must not corrupt resource count."""
    snap_before = capacity_vm_manager.admission.snapshot()

    vm = await capacity_vm_manager.create_vm(
        language=Language.PYTHON,
        tenant_id="test",
        task_id="task-1",
    )

    snap_during = capacity_vm_manager.admission.snapshot()
    assert snap_during.allocated_vm_slots == snap_before.allocated_vm_slots + 1

    # First destroy
    await capacity_vm_manager.destroy_vm(vm)
    snap_after1 = capacity_vm_manager.admission.snapshot()
    assert snap_after1.allocated_vm_slots == snap_before.allocated_vm_slots

    # Second destroy should be safe (no-op for resources)
    await capacity_vm_manager.destroy_vm(vm)
    snap_after2 = capacity_vm_manager.admission.snapshot()
    assert snap_after2.allocated_vm_slots == snap_before.allocated_vm_slots, "Double destroy corrupted resource count"


async def test_vm_holds_semaphore_flag_set_correctly(capacity_vm_manager: VmManager) -> None:
    """Verify holds_semaphore_slot flag is set on successful creation."""
    vm = await capacity_vm_manager.create_vm(
        language=Language.PYTHON,
        tenant_id="test",
        task_id="task-1",
    )

    assert vm.holds_semaphore_slot is True
    assert vm.resource_reservation is not None

    await capacity_vm_manager.destroy_vm(vm)

    assert vm.holds_semaphore_slot is False
    assert vm.resource_reservation is None


# ============================================================================
# Slow Tests - Load Testing
# ============================================================================


@pytest.mark.slow
async def test_no_capacity_errors_under_load(images_dir: Path) -> None:
    """Verify VmCapacityError no longer raised under concurrent load.

    Regression test: previously the capacity check in _create_vm_impl could
    race with the semaphore, causing VmCapacityError even with available slots.
    """
    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    capacity_errors = 0
    total_runs = 10

    async with Scheduler(config) as scheduler:
        tasks = [scheduler.run(code="print('ok')", language=Language.PYTHON) for _ in range(total_runs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, VmCapacityError):
                capacity_errors += 1

    assert capacity_errors == 0, f"Got {capacity_errors} VmCapacityError - regression in capacity tracking"


@pytest.mark.slow
async def test_many_sequential_create_destroy_cycles(capacity_vm_manager: VmManager) -> None:
    """Verify resource count stays consistent after many create/destroy cycles."""
    snap_initial = capacity_vm_manager.admission.snapshot()
    cycles = 20

    for i in range(cycles):
        vm = await capacity_vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id=f"task-{i}",
        )
        await capacity_vm_manager.destroy_vm(vm)

    snap_final = capacity_vm_manager.admission.snapshot()
    assert snap_final.allocated_vm_slots == snap_initial.allocated_vm_slots, (
        f"Resource count drifted after {cycles} cycles"
    )

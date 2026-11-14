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
import contextlib
import dataclasses
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from exec_sandbox import constants
from exec_sandbox.config import SchedulerConfig
from exec_sandbox.exceptions import ErrorCode, SandboxError, TransientError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.settings import Settings
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def capacity_settings(images_dir: Path) -> Settings:
    """Settings with tight CPU budget for testing capacity.

    Uses host_cpu_count=3.0 with cpu_overcommit=1.0 and no CPU reserve so that
    only 2 VMs (at 1.25 effective CPU each = 1.0 guest + 0.25 overhead) can be
    admitted concurrently (2 * 1.25 = 2.5 <= 3.0, 3 * 1.25 = 3.75 > 3.0).
    Memory is large enough to not be the bottleneck.
    """
    return Settings(
        base_images_dir=images_dir,
        kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
        host_memory_mb=100_000.0,  # Large, not the bottleneck
        host_cpu_count=3.0,  # Tight: 2 VMs fit (2.5 effective), 3 don't (3.75)
        cpu_overcommit_ratio=1.0,  # No overcommit
        host_cpu_reserve_cores=0.0,  # No reserve (simplifies test math)
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
    TransientError CAPACITY immediately.
    """
    # Create first VM (budget=3.0, each VM=1.25 effective)
    vm1 = await capacity_vm_manager.create_vm(
        VmSpec(language=Language.PYTHON), tenant_id="test", task_id="task-1", cache_leases=()
    )

    vm2 = await capacity_vm_manager.create_vm(
        VmSpec(language=Language.PYTHON), tenant_id="test", task_id="task-2", cache_leases=()
    )

    # Start third creation (should block, not fail)
    create_task = asyncio.create_task(
        capacity_vm_manager.create_vm(
            VmSpec(language=Language.PYTHON), tenant_id="test", task_id="task-3", cache_leases=()
        )
    )

    # Give it time to potentially complete (should stay blocked)
    await asyncio.sleep(0.5)
    assert not create_task.done(), "Should be blocked waiting for resources, not completed"

    # Destroy one VM (releases resources once its cleanup is confirmed)
    assert await capacity_vm_manager.destroy_vm(vm1) is True

    # Third creation should now complete
    vm3 = await asyncio.wait_for(create_task, timeout=120)
    assert vm3 is not None
    assert vm3.holds_admission_slot is True

    # Cleanup
    assert await capacity_vm_manager.destroy_vm(vm2) is True
    assert await capacity_vm_manager.destroy_vm(vm3) is True


class _GiveBackError(Exception):
    """Ends the probe's reservations: reservation_context gives back a reservation on an error."""


async def _vms_admitted_now(manager: VmManager, *, most: int) -> int:
    """How many more VMs the budget admits at once (at most `most`): the probe takes
    reservations through VmManager.reservation_context until one waits (0.1 s), then an
    error ends each reservation_context, which gives its reservation back."""
    taken = 0
    with contextlib.suppress(_GiveBackError):
        async with contextlib.AsyncExitStack() as stack:
            for i in range(most):
                try:
                    async with asyncio.timeout(0.1):
                        await stack.enter_async_context(
                            manager.reservation_context(
                                VmSpec(language=Language.PYTHON), vm_id=f"probe-{i}", held=frozenset()
                            )
                        )
                except TimeoutError:
                    break
                taken += 1
            raise _GiveBackError
    return taken


async def test_destroy_gives_the_slot_back(capacity_vm_manager: VmManager) -> None:
    """A VM takes one of the 2 slots while it runs; destroy_vm gives it back once the whole
    cleanup (processes, cgroup, files) is confirmed, and says so (False: the reaper still
    holds the slot)."""
    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 2

    vm = await capacity_vm_manager.create_vm(
        VmSpec(language=Language.PYTHON), tenant_id="test", task_id="task-1", cache_leases=()
    )
    assert vm.holds_admission_slot is True
    assert vm.resource_reservation is not None
    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 1

    assert await capacity_vm_manager.destroy_vm(vm) is True
    assert vm.holds_admission_slot is False
    assert vm.resource_reservation is None
    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 2


async def test_admission_released_on_create_failure(capacity_vm_manager: VmManager) -> None:
    """A create_vm that fails after admission, at each boot attempt, gives its slot back. The
    fault replaces _create_vm_impl, the step between admission and the VM: no public input
    fails there. The attempts follow each other with no pause."""
    failure = TransientError("mock failure", code=ErrorCode.OVERLAY_FAILED, context=None, hint=None, instance_id=None)
    no_pause = dataclasses.replace(constants.RETRY_USER_FACING, retry_min_seconds=0.0, retry_max_seconds=0.0)
    with patch.object(capacity_vm_manager, "_create_vm_impl", side_effect=failure) as attempt:
        with pytest.raises(TransientError, check=lambda e: e is failure):
            await capacity_vm_manager.create_vm(
                VmSpec(language=Language.PYTHON),
                tenant_id="test",
                task_id="task-1",
                cache_leases=(),
                retry_profile=no_pause,
            )
    assert attempt.call_count == constants.RETRY_USER_FACING.max_attempts > 1

    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 2


async def test_double_destroy_does_not_double_release(capacity_vm_manager: VmManager) -> None:
    """A second destroy_vm of the same VM gives nothing back: the budget stays at 2 VMs, not 3."""
    vm = await capacity_vm_manager.create_vm(
        VmSpec(language=Language.PYTHON), tenant_id="test", task_id="task-1", cache_leases=()
    )

    assert await capacity_vm_manager.destroy_vm(vm) is True
    await capacity_vm_manager.destroy_vm(vm)

    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 2, "Double destroy corrupted resource count"


# ============================================================================
# Slow Tests - Load Testing
# ============================================================================


@pytest.mark.slow
async def test_no_capacity_errors_under_load(images_dir: Path) -> None:
    """Ten concurrent runs on the default budget get no TransientError CAPACITY: a request
    that does not fit yet waits for admission instead of failing.
    """
    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    capacity_errors = 0
    total_runs = 10

    async with Scheduler(config) as scheduler:
        tasks = [run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')") for _ in range(total_runs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, SandboxError) and r.code is ErrorCode.CAPACITY:
                capacity_errors += 1

    assert capacity_errors == 0, f"Got {capacity_errors} CAPACITY errors - regression in capacity tracking"


@pytest.mark.slow
async def test_many_sequential_create_destroy_cycles(capacity_vm_manager: VmManager) -> None:
    """Verify resource count stays consistent after many create/destroy cycles."""
    cycles = 20

    for i in range(cycles):
        vm = await capacity_vm_manager.create_vm(
            VmSpec(language=Language.PYTHON), tenant_id="test", task_id=f"task-{i}", cache_leases=()
        )
        assert await capacity_vm_manager.destroy_vm(vm) is True

    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 2, f"Resource count drifted after {cycles} cycles"

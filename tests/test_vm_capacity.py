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
import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from exec_sandbox import constants
from exec_sandbox.config import SchedulerConfig
from exec_sandbox.exceptions import ErrorCode, TransientError
from exec_sandbox.models import Language
from exec_sandbox.qemu_storage_daemon import QemuStorageDaemon
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.settings import Settings
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator
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
    manager = VmManager(capacity_settings)
    try:
        await manager.start()
        yield manager
    finally:
        await manager.stop()


# ============================================================================
# Integration Tests - Concurrent VM Capacity
# ============================================================================


class _AdmissionLog(logging.Handler):
    """The admission log lines of one VM, in order; `asked` is set at its "Acquiring resources" line.

    The controller logs "Acquiring resources", then, with no await between, either "Resource
    reservation acquired" (admitted) or nothing more until a release wakes it (it waits)."""

    def __init__(self, vm_id: str) -> None:
        super().__init__(logging.DEBUG)
        self.vm_id = vm_id
        self.lines: list[str] = []
        self.asked = asyncio.Event()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "vm_id", None) == self.vm_id:
            self.lines.append(record.getMessage())
            if record.getMessage() == "Acquiring resources":
                self.asked.set()

    async def waits(self) -> bool:
        """Wait for the "Acquiring resources" line; True when the request was not admitted then."""
        async with asyncio.timeout(10):
            await self.asked.wait()
        return self.lines == ["Acquiring resources"]


@contextlib.contextmanager
def _admission_log(vm_id: str) -> Iterator[_AdmissionLog]:
    """The admission log of `vm_id` (DEBUG lines) while the block runs."""
    admission_logger = logging.getLogger("exec_sandbox.admission")
    log, level = _AdmissionLog(vm_id), admission_logger.level
    admission_logger.addHandler(log)
    admission_logger.setLevel(logging.DEBUG)
    try:
        yield log
    finally:
        admission_logger.setLevel(level)
        admission_logger.removeHandler(log)


async def test_create_vm_blocks_at_capacity(capacity_vm_manager: VmManager) -> None:
    """A create_vm that does not fit (two VMs of 1.25 CPU hold 2.5 of the 3.0 budget) waits for a
    slot, with no TransientError CAPACITY: it is admitted once destroy_vm gives a slot back."""
    vm1 = await capacity_vm_manager.create_vm(
        VmSpec(language=Language.PYTHON),
        tenant_id="test",
        task_id="task-1",
        cache_leases=(),
        direct_write_target=None,
        expose_ports=None,
        on_boot_log=None,
        snapshot_drive=None,
        reservation=None,
        retry_profile=constants.RETRY_USER_FACING,
        mem_path=None,
    )
    vm2 = await capacity_vm_manager.create_vm(
        VmSpec(language=Language.PYTHON),
        tenant_id="test",
        task_id="task-2",
        cache_leases=(),
        direct_write_target=None,
        expose_ports=None,
        on_boot_log=None,
        snapshot_drive=None,
        reservation=None,
        retry_profile=constants.RETRY_USER_FACING,
        mem_path=None,
    )

    with _admission_log("test-task-3") as admission:
        create_task = asyncio.create_task(
            capacity_vm_manager.create_vm(
                VmSpec(language=Language.PYTHON),
                tenant_id="test",
                task_id="task-3",
                cache_leases=(),
                direct_write_target=None,
                expose_ports=None,
                on_boot_log=None,
                snapshot_drive=None,
                reservation=None,
                retry_profile=constants.RETRY_USER_FACING,
                mem_path=None,
            )
        )
        try:
            assert await admission.waits()
            assert not create_task.done()

            assert await capacity_vm_manager.destroy_vm(vm1) is True  # gives its slot back
            vm3 = await asyncio.wait_for(create_task, timeout=120)
        finally:  # a failed assert leaves no waiting create_vm (the manager's stop destroys a VM it made)
            create_task.cancel()
            await asyncio.gather(create_task, return_exceptions=True)
        assert admission.lines == ["Acquiring resources", "Resource reservation acquired"]
    reservation = vm3.resource_reservation
    assert reservation is not None
    assert (reservation.vm_id, reservation.cpu_cores) == ("test-task-3", 1.25)  # its own admitted request

    assert await capacity_vm_manager.destroy_vm(vm2) is True
    assert await capacity_vm_manager.destroy_vm(vm3) is True


class _GiveBackError(Exception):
    """Ends the probe's reservations: reservation_context gives back a reservation on an error."""


async def _vms_admitted_now(manager: VmManager, *, most: int) -> int:
    """How many more VMs the budget admits at once (at most `most`). The probe takes
    reservations through VmManager.reservation_context, each in a task of its own, until one is
    not admitted at once (_AdmissionLog.waits): that one is cancelled, then an error ends each
    reservation_context, which gives its reservation back."""
    taken = 0
    with contextlib.suppress(_GiveBackError):
        async with contextlib.AsyncExitStack() as stack:
            for i in range(most):
                admission = stack.enter_context(_admission_log(f"probe-{i}"))
                reserving = asyncio.ensure_future(
                    stack.enter_async_context(
                        manager.reservation_context(
                            VmSpec(language=Language.PYTHON), vm_id=f"probe-{i}", held=frozenset()
                        )
                    )
                )
                if await admission.waits():
                    reserving.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await reserving
                    break
                await reserving
                taken += 1
            raise _GiveBackError
    return taken


async def test_destroy_gives_the_slot_back(capacity_vm_manager: VmManager) -> None:
    """A VM takes one of the 2 slots while it runs; destroy_vm gives it back once the whole
    cleanup (processes, cgroup, files) is confirmed, and says so (False: the reaper still
    holds the slot)."""
    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 2

    vm = await capacity_vm_manager.create_vm(
        VmSpec(language=Language.PYTHON),
        tenant_id="test",
        task_id="task-1",
        cache_leases=(),
        direct_write_target=None,
        expose_ports=None,
        on_boot_log=None,
        snapshot_drive=None,
        reservation=None,
        retry_profile=constants.RETRY_USER_FACING,
        mem_path=None,
    )
    assert vm.resource_reservation is not None
    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 1

    assert await capacity_vm_manager.destroy_vm(vm) is True
    assert vm.resource_reservation is None
    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 2


async def test_admission_released_on_create_failure(capacity_vm_manager: VmManager) -> None:
    """A create_vm that fails after admission, at each boot attempt, gives its slot back. The
    fault is the overlay of each attempt (QemuStorageDaemon.create_overlay, OVERLAY_FAILED: retried), after
    the attempt published its pending owner: each attempt's cleanup runs, and the slot comes
    back once. The attempts follow each other with no pause."""
    failure = TransientError("mock failure", code=ErrorCode.OVERLAY_FAILED, context=None, hint=None, instance_id=None)
    no_pause = dataclasses.replace(constants.RETRY_USER_FACING, retry_min_seconds=0.0, retry_max_seconds=0.0)
    # start() makes the overlay templates in the background, through the same method: they
    # end first, so the patch counts the boot attempts only
    if (templates := capacity_vm_manager._templates_task) is not None:  # pyright: ignore[reportPrivateUsage]
        await templates
    with patch.object(QemuStorageDaemon, "create_overlay", AsyncMock(side_effect=failure)) as attempt:
        with pytest.raises(TransientError, check=lambda e: e is failure):
            await capacity_vm_manager.create_vm(
                VmSpec(language=Language.PYTHON),
                tenant_id="test",
                task_id="task-1",
                cache_leases=(),
                retry_profile=no_pause,
                direct_write_target=None,
                expose_ports=None,
                on_boot_log=None,
                snapshot_drive=None,
                reservation=None,
                mem_path=None,
            )
    assert attempt.call_count == constants.RETRY_USER_FACING.max_attempts > 1

    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 2
    assert not capacity_vm_manager._pending_launches  # pyright: ignore[reportPrivateUsage]  # no owner left


async def test_double_destroy_does_not_double_release(capacity_vm_manager: VmManager) -> None:
    """A second destroy_vm of the same VM gives nothing back: the budget stays at 2 VMs, not 3."""
    vm = await capacity_vm_manager.create_vm(
        VmSpec(language=Language.PYTHON),
        tenant_id="test",
        task_id="task-1",
        cache_leases=(),
        direct_write_target=None,
        expose_ports=None,
        on_boot_log=None,
        snapshot_drive=None,
        reservation=None,
        retry_profile=constants.RETRY_USER_FACING,
        mem_path=None,
    )

    assert await capacity_vm_manager.destroy_vm(vm) is True
    assert await capacity_vm_manager.destroy_vm(vm) is True  # nothing left to clean

    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 2, "Double destroy corrupted resource count"


# ============================================================================
# Slow Tests - Load Testing
# ============================================================================


@pytest.mark.slow
async def test_no_capacity_errors_under_load(images_dir: Path) -> None:
    """Ten concurrent runs on the default budget all run their code: a request that does not
    fit yet waits for admission instead of failing (no TransientError CAPACITY, nor any other).
    """
    config = SchedulerConfig(
        images_dir=images_dir,
        auto_download_assets=False,
        warm_pool_size=0,
    )
    total_runs = 10

    async with Scheduler(config) as scheduler:
        tasks = [run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')") for _ in range(total_runs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    assert [r if isinstance(r, BaseException) else (r.exit_code, r.stdout) for r in results] == [
        (0, "ok\n")
    ] * total_runs


@pytest.mark.slow
async def test_many_sequential_create_destroy_cycles(capacity_vm_manager: VmManager) -> None:
    """Verify resource count stays consistent after many create/destroy cycles."""
    cycles = 20

    for i in range(cycles):
        vm = await capacity_vm_manager.create_vm(
            VmSpec(language=Language.PYTHON),
            tenant_id="test",
            task_id=f"task-{i}",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
        )
        assert await capacity_vm_manager.destroy_vm(vm) is True

    assert await _vms_admitted_now(capacity_vm_manager, most=3) == 2, f"Resource count drifted after {cycles} cycles"

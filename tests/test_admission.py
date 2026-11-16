"""Tests for ResourceAdmissionController.

Tests the admission gates (memory budget, CPU budget, available-memory floor, memory
pressure, home disk space and its cache evictions per filesystem), blocking/signaling
behavior, timeout, cgroup-aware startup, probe cache, self-wake timer, and graceful
degradation.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import uuid4

import psutil
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

from exec_sandbox.admission import (
    _AFTER_COMMIT,  # pyright: ignore[reportPrivateUsage]
    _AT_REMOVAL,  # pyright: ignore[reportPrivateUsage]
    _BACKGROUND,  # pyright: ignore[reportPrivateUsage]
    ResourceAdmissionController,
    ResourceReservation,
    _DiskSpace,  # pyright: ignore[reportPrivateUsage]
    _Evictions,  # pyright: ignore[reportPrivateUsage]
    _Pending,  # pyright: ignore[reportPrivateUsage]
    _read_macos_memory_pressure_level,  # pyright: ignore[reportPrivateUsage]
    _Settle,  # pyright: ignore[reportPrivateUsage]
    _settle_removals,  # pyright: ignore[reportPrivateUsage]
    _zfs_loaded,  # pyright: ignore[reportPrivateUsage]
    _zfs_settle,  # pyright: ignore[reportPrivateUsage]
)
from exec_sandbox.constants import (
    DEFAULT_CPU_OVERCOMMIT_RATIO,
    DEFAULT_DISK_OVERCOMMIT_RATIO,
    DEFAULT_HOST_CPU_RESERVE_CORES,
    DEFAULT_HOST_MEMORY_RESERVE_RATIO,
    DEFAULT_MEMORY_OVERCOMMIT_RATIO,
    RESOURCE_ADMISSION_TIMEOUT_SECONDS,
)
from exec_sandbox.exceptions import ErrorAction, ErrorCode, InvalidInputError, RouteError, RouteKey, TransientError
from exec_sandbox.files.locks import CacheEntry
from exec_sandbox.platform_utils import HostOS, detect_host_os
from exec_sandbox.worker_facts import WorkerFacts
from tests.conftest import skip_unless_linux, skip_unless_macos, time_with
from tests.test_files_locks import SizedCache, evicted_in_child, leased_in_child

# ============================================================================
# Helpers
# ============================================================================

_CPU = 1.0
"""Guest cores of most test acquire calls: 1.25 cores with the overhead, as the hand-computed
figures below count."""
# No such folder: gate 4 reads no free space and admits, so the free disk of the host
# never changes the outcome of the tests of the other gates (TestDiskGate covers gate 4)
_DISK_DIR = Path(tempfile.gettempdir()) / f"no-home-disks-{uuid4().hex}"
_MIB = 1024 * 1024
_TIMEOUT = RESOURCE_ADMISSION_TIMEOUT_SECONDS
_REFUSED = 0.05
"""Timeout of an acquire() that a gate refuses: the test reads the gate from its error."""

# Hand-computed figures below use the per-VM overheads that acquire() adds: 80 MB of memory
# (QEMU and gvproxy), 128 MB more with TCG (its translation cache), 0.25 CPU core. A guest of
# 256 MB reserves 336 MB.


def _make_controller(
    *,
    host_memory_mb: float = 16_000.0,
    host_cpu_count: float = 8.0,
    memory_overcommit: float = DEFAULT_MEMORY_OVERCOMMIT_RATIO,
    cpu_overcommit: float = DEFAULT_CPU_OVERCOMMIT_RATIO,
    reserve_ratio: float = DEFAULT_HOST_MEMORY_RESERVE_RATIO,
    available_memory_floor_mb: int = 0,
    cpu_reserve_cores: float = DEFAULT_HOST_CPU_RESERVE_CORES,
) -> ResourceAdmissionController:
    """Create an admission controller with known host resources (no psutil)."""
    return ResourceAdmissionController(
        disk_dir=_DISK_DIR,
        disk_free_floor_mb=0,
        memory_overcommit_ratio=memory_overcommit,
        cpu_overcommit_ratio=cpu_overcommit,
        disk_overcommit_ratio=DEFAULT_DISK_OVERCOMMIT_RATIO,
        host_memory_reserve_ratio=reserve_ratio,
        host_memory_mb=host_memory_mb,
        host_cpu_count=host_cpu_count,
        available_memory_floor_mb=available_memory_floor_mb,
        host_cpu_reserve_cores=cpu_reserve_cores,
    )


def _probed_controller(*, memory_overcommit: float = 1.0, cpu_overcommit: float = 1.0) -> ResourceAdmissionController:
    """A controller with no host figures: start() probes them (cgroup, then psutil).
    Memory reserve 10 %, CPU reserve 0.5 core."""
    return ResourceAdmissionController(
        disk_dir=_DISK_DIR,
        disk_free_floor_mb=0,
        memory_overcommit_ratio=memory_overcommit,
        cpu_overcommit_ratio=cpu_overcommit,
        disk_overcommit_ratio=DEFAULT_DISK_OVERCOMMIT_RATIO,
        host_memory_reserve_ratio=0.1,
        host_memory_mb=None,
        host_cpu_count=None,
        available_memory_floor_mb=0,
        host_cpu_reserve_cores=0.5,
    )


@contextmanager
def _probe_clock() -> Generator[Callable[[], None]]:
    """The clock of the 100 ms probe caches (time.monotonic of the admission module) for the
    block. Each call of the returned function moves it 1 s on: the next probe reads the (faked)
    host again. It stands for the cache time that passes between two steps of a test."""
    offset = [0.0]

    def later() -> None:
        offset[0] += 1.0

    with patch("exec_sandbox.admission.time", time_with(monotonic=lambda: time.monotonic() + offset[0])):
        yield later


async def _budgets(ctrl: ResourceAdmissionController) -> dict[str, str | None]:
    """The memory and CPU budgets of `ctrl`, as the error of a request above both gives them
    (RouteError CAPACITY_NEVER_FITS, `found`: memory in whole MB, CPU with two decimals;
    None: no limit)."""
    with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.CAPACITY_NEVER_FITS) as exc_info:
        await ctrl.acquire(
            "vm-huge", memory_mb=10**12, cpu_cores=10**6, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
        )
    return dict(exc_info.value.found)


async def _blocking_gate_of(ctrl: ResourceAdmissionController, *, memory_mb: float = 256) -> str | None:
    """One acquire() on `ctrl`: the gate that refused it (TransientError CAPACITY, its
    `blocking_gate`), or None when it is admitted (then released)."""
    try:
        reservation = await ctrl.acquire(
            "vm-1", memory_mb=memory_mb, cpu_cores=_CPU, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
        )
    except TransientError as e:
        assert e.code is ErrorCode.CAPACITY
        return e.context["blocking_gate"]
    await ctrl.release(reservation)
    return None


# ============================================================================
# Budget Calculation
# ============================================================================


async def test_budget_calculation() -> None:
    """Memory budget = host_total * (1 - reserve_ratio) * overcommit.
    CPU budget = (host_cpus - cpu_reserve) * overcommit.
    """
    ctrl = _make_controller(
        host_memory_mb=16_000.0,
        reserve_ratio=0.1,
        memory_overcommit=1.5,
        host_cpu_count=8.0,
        cpu_overcommit=4.0,
        cpu_reserve_cores=0.5,
    )
    # 16000 * 0.9 * 1.5 = 21600; (8 - 0.5) * 4 = 30
    assert await _budgets(ctrl) == {"memory_mb": "21600", "cpu": "30.00"}


async def test_budget_facts_round_down_and_the_target_rounds_up() -> None:
    """budget_facts (the route facts of the budget) round down and CAPACITY_NEVER_FITS's
    `required` rounds up: a worker whose budget facts are not below `required` admits the
    VM, and the worker that raised the error never satisfies its own `required`.
    Budgets: 900.5 * 0.9 * 1.0 = 810.45 MB (a fraction: the facts give 810); (1.7 - 0.5) *
    1.05 = 1.26 cores."""
    ctrl = _make_controller(
        host_memory_mb=900.5,
        reserve_ratio=0.1,
        memory_overcommit=1.0,
        host_cpu_count=1.7,
        cpu_overcommit=1.05,
        cpu_reserve_cores=0.5,
    )
    assert ctrl.budget_facts() == (810.0, 1.26)
    with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.CAPACITY_NEVER_FITS) as exc_info:
        # 730.6 + 80 MB of overhead = 810.6 > 810.45; 1.01 + 0.25 = 1.26 cores fit
        await ctrl.acquire(
            "vm-big", memory_mb=730.6, cpu_cores=1.01, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
        )
    required = exc_info.value.required
    assert required == {"memory_mb": "811", "cpu": "1.26"}
    worker = WorkerFacts(values=dict(exc_info.value.found), key_ids=())  # this worker, as the error found it
    assert not worker.satisfies(required)
    bigger = WorkerFacts(values={RouteKey.MEMORY_MB: "811", RouteKey.CPU: "1.26"}, key_ids=())
    assert bigger.satisfies(required)


@pytest.mark.parametrize(
    ("host_cpu_count", "cpu_reserve_cores", "cpu_overcommit", "cpu_cores"),
    [pytest.param(1.4, 0.5, 12.5, 11.0, id="11.25-cores"), pytest.param(47.5, 0.0, 2.3, 109.0, id="109.25-cores")],
)
async def test_a_vm_at_the_budget_fits_whatever_the_float_noise(
    host_cpu_count: float, cpu_reserve_cores: float, cpu_overcommit: float, cpu_cores: float
) -> None:
    """A budget that is a float product ((1.4 - 0.5) * 12.5 is 11.249999999999998) admits a VM
    that needs exactly that budget (11 + 0.25 cores): the budget facts give the same value, and
    acquire() admits it, so a worker never refuses what its own facts say it holds."""
    ctrl = _make_controller(
        host_cpu_count=host_cpu_count, cpu_reserve_cores=cpu_reserve_cores, cpu_overcommit=cpu_overcommit
    )
    need, have = ctrl.check_fits("vm", 256, cpu_cores, use_tcg=False, required={}, found={})
    assert need[RouteKey.CPU] == have[RouteKey.CPU] == f"{cpu_cores + 0.25:.2f}"
    with _mock_system_memory(100_000):  # host memory never decides here
        reservation = await ctrl.acquire(
            "vm", memory_mb=256, cpu_cores=cpu_cores, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
        )
    await ctrl.release(reservation)


def test_check_fits_adds_the_checked_target() -> None:
    """check_fits (the load's check before the download): the route facts checked before
    (this worker has them) are in `required` and `found` with the budget; a VM that fits
    passes. Budgets as in test_budget_calculation: 21600 MB of memory, 30 cores."""
    ctrl = _make_controller(
        host_memory_mb=16_000.0,
        reserve_ratio=0.1,
        memory_overcommit=1.5,
        host_cpu_count=8.0,
        cpu_overcommit=4.0,
        cpu_reserve_cores=0.5,
    )
    # This worker takes every counter rate: `found` gives its own value, not the saved one
    target = {RouteKey.ARCH: "aarch64", RouteKey.MACHINE_TYPE: "virt-11.1", RouteKey.COUNTER_HZ: "2700000000"}
    here = {**target, RouteKey.COUNTER_HZ: "any"}
    ctrl.check_fits("my-vm", 512, 1.0, use_tcg=False, required=target, found=here)  # fits: no error
    with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.CAPACITY_NEVER_FITS) as exc_info:
        ctrl.check_fits("my-vm", 512, 64.0, use_tcg=False, required=target, found=here)
    assert dict(exc_info.value.required) == {**target, "memory_mb": "592", "cpu": "64.25"}
    assert dict(exc_info.value.found) == {**here, "memory_mb": "21600", "cpu": "30.00"}


async def test_budget_reserve_ratio_scales() -> None:
    """Verify reserve ratio scales with host size."""
    # 4GB host, 10% reserve → 400MB reserved → 3600MB available → *1.5 = 5400
    ctrl_small = _make_controller(host_memory_mb=4_000.0, reserve_ratio=0.1, memory_overcommit=1.5)
    assert (await _budgets(ctrl_small))["memory_mb"] == "5400"

    # 64GB host, 10% reserve → 6400MB reserved → 57600MB available → *1.5 = 86400
    ctrl_large = _make_controller(host_memory_mb=64_000.0, reserve_ratio=0.1, memory_overcommit=1.5)
    assert (await _budgets(ctrl_large))["memory_mb"] == "86400"


# ============================================================================
# Gate 1: Memory Budget
# ============================================================================


async def test_memory_gate_blocks_when_budget_exhausted() -> None:
    """Acquire blocks when memory budget would be exceeded, though the host has memory to
    spare (gate 3a passes): the gate is "memory_budget". A release admits the waiter."""
    with _mock_system_memory(100_000):  # host memory never decides: the budget gates do
        # 4000 MB host, 10 % reserve, 1.0x overcommit: a budget of 3600 MB. Each VM of 256 MB
        # reserves 336 MB: 10 fit (3360 MB), the 11th (3696 MB) blocks.
        ctrl = _make_controller(
            host_memory_mb=4_000.0,
            reserve_ratio=0.1,
            memory_overcommit=1.0,
        )

        reservations = []
        for i in range(10):
            r = await ctrl.acquire(
                f"vm-{i}", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
            )
            reservations.append(r)
        assert await _blocking_gate_of(ctrl) == "memory_budget"

        # Next should block (budget exhausted)
        acquire_task = asyncio.create_task(
            ctrl.acquire(
                "vm-block", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=1.0, use_tcg=False, held=frozenset()
            )
        )
        await _until_waiting_or_done(ctrl, acquire_task)
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
    """Acquire blocks when CPU budget would be exceeded.

    Each VM uses its guest cores + 0.25 core of overhead.
    With 3.0 host CPUs, 0 reserve, 1.0x overcommit → 3.0 budget.
    Each VM: 1.25 effective → 2 fit, 3rd blocks.
    """
    with _mock_system_memory(100_000):  # host memory never decides: the budget gates do
        ctrl = _make_controller(
            host_memory_mb=100_000.0,
            host_cpu_count=3.0,
            cpu_overcommit=1.0,
            cpu_reserve_cores=0.0,
        )

        r1 = await ctrl.acquire(
            "vm-1", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
        )
        r2 = await ctrl.acquire(
            "vm-2", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
        )

        # 3rd should block (2.5 + 1.25 = 3.75 > 3.0)
        acquire_task = asyncio.create_task(
            ctrl.acquire("vm-3", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=1.0, use_tcg=False, held=frozenset())
        )
        await _until_waiting_or_done(ctrl, acquire_task)
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
    """2-core VMs consume proportional budget including overhead.

    Each 2-core VM uses 2.0 + 0.25 = 2.25 effective CPU.
    With 5.0 host CPUs, 0 reserve, 1.0x overcommit → 5.0 budget.
    2 VMs use 4.5, 3rd would need 6.75 → blocks.
    """
    with _mock_system_memory(100_000):  # host memory never decides: the budget gates do
        ctrl = _make_controller(
            host_memory_mb=100_000.0,
            host_cpu_count=5.0,
            cpu_overcommit=1.0,
            cpu_reserve_cores=0.0,
        )

        r1 = await ctrl.acquire(
            "vm-1", memory_mb=256, cpu_cores=2.0, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
        )
        r2 = await ctrl.acquire(
            "vm-2", memory_mb=256, cpu_cores=2.0, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
        )

        # 3rd 2-core VM should block (4.5 + 2.25 = 6.75 > 5.0)
        acquire_task = asyncio.create_task(
            ctrl.acquire("vm-3", memory_mb=256, cpu_cores=2.0, disk_mb=0, timeout=1.0, use_tcg=False, held=frozenset())
        )
        await _until_waiting_or_done(ctrl, acquire_task)
        assert not acquire_task.done()

        # Release one → unblocks
        await ctrl.release(r1)
        r3 = await asyncio.wait_for(acquire_task, timeout=5.0)
        assert r3.vm_id == "vm-3"

        await ctrl.release(r2)
        await ctrl.release(r3)


async def test_mixed_core_vms_share_budget() -> None:
    """Mix of 1-core and 2-core VMs shares the same budget correctly.

    Each VM gets 0.25 core of overhead added to its guest cores.
    With 4.0 host CPUs, 0 reserve, 1.0x overcommit → 4.0 budget.
    """
    with _mock_system_memory(100_000):  # host memory never decides: the budget gates do
        ctrl = _make_controller(
            host_memory_mb=100_000.0,
            host_cpu_count=4.0,
            cpu_overcommit=1.0,
            cpu_reserve_cores=0.0,
        )

        # 1-core (1.25 eff) + 2-core (2.25 eff) = 3.50 (fits in 4.0 budget)
        r1 = await ctrl.acquire(
            "vm-1", memory_mb=256, cpu_cores=1.0, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
        )
        r2 = await ctrl.acquire(
            "vm-2", memory_mb=256, cpu_cores=2.0, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
        )

        # Even a 1-core VM should block now (1.25 > 0.50 remaining)
        acquire_task = asyncio.create_task(
            ctrl.acquire("vm-3", memory_mb=256, cpu_cores=1.0, disk_mb=0, timeout=1.0, use_tcg=False, held=frozenset())
        )
        await _until_waiting_or_done(ctrl, acquire_task)
        assert not acquire_task.done()

        # The 0.50 left holds a 0.25-core VM (0.50 effective), and nothing more (0.01 core: 0.26)
        r_fill = await ctrl.acquire(
            "vm-fill", memory_mb=256, cpu_cores=0.25, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
        )
        with pytest.raises(TransientError, check=lambda e: e.context["blocking_gate"] == "cpu_budget"):
            await ctrl.acquire(
                "vm-more", memory_mb=256, cpu_cores=0.01, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
            )
        await ctrl.release(r_fill)

        await ctrl.release(r1)
        r3 = await asyncio.wait_for(acquire_task, timeout=5.0)
        assert r3.vm_id == "vm-3"

        await ctrl.release(r2)
        await ctrl.release(r3)


async def test_overcommit_allows_more_cpu_than_physical() -> None:
    """CPU overcommit ratio allows admitting more cores than physical host has.

    3 physical CPUs, 0.5 reserve, 4.0x overcommit → (3-0.5)*4 = 10.0 CPU budget.
    Each 1-core VM effective: 1.25. 8 VMs * 1.25 = 10.0 (exact fit).
    """
    with _mock_system_memory(100_000):  # host memory never decides: the budget gates do
        ctrl = _make_controller(
            host_memory_mb=100_000.0,
            host_cpu_count=3.0,
            cpu_overcommit=4.0,
            cpu_reserve_cores=0.5,
        )

        reservations = []
        for i in range(8):
            r = await ctrl.acquire(
                f"vm-{i}", memory_mb=256, cpu_cores=1.0, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
            )
            reservations.append(r)

        # 9th should block
        acquire_task = asyncio.create_task(
            ctrl.acquire(
                "vm-block", memory_mb=256, cpu_cores=1.0, disk_mb=0, timeout=1.0, use_tcg=False, held=frozenset()
            )
        )
        await _until_waiting_or_done(ctrl, acquire_task)
        assert not acquire_task.done()

        await ctrl.release(reservations[0])
        r_new = await asyncio.wait_for(acquire_task, timeout=5.0)

        for r in reservations[1:]:
            await ctrl.release(r)
        await ctrl.release(r_new)


# ============================================================================
# Timeout
# ============================================================================


@pytest.mark.parametrize(
    ("memory_mb", "cpu_cores", "required", "found"),
    [
        # 1000000 + 80 MB of overhead, above the 21600 MB budget
        pytest.param(
            1_000_000,
            _CPU,
            {"memory_mb": "1000080", "cpu": "1.25"},
            {"memory_mb": "21600", "cpu": "30.00"},
            id="memory",
        ),
        # 64 + 0.25 cores of overhead, above the 30-core budget
        pytest.param(256, 64.0, {"memory_mb": "336", "cpu": "64.25"}, {"memory_mb": "21600", "cpu": "30.00"}, id="cpu"),
    ],
)
async def test_acquire_never_fits_fails_at_once(
    memory_mb: float, cpu_cores: float, required: dict[str, str], found: dict[str, str]
) -> None:
    """A request above the whole budget fails at once, without waiting for the timeout.

    It is a RouteError (a worker with a larger budget can run it); `required` holds the
    whole target (both dimensions, the VM overhead included), `found` the budgets. Budgets as in test_budget_calculation:
    21600 MB of memory, 30 cores.
    """
    ctrl = _make_controller(
        host_memory_mb=16_000.0,
        reserve_ratio=0.1,
        memory_overcommit=1.5,
        host_cpu_count=8.0,
        cpu_overcommit=4.0,
        cpu_reserve_cores=0.5,
    )

    async with asyncio.timeout(1.0):
        with pytest.raises(
            RouteError, check=lambda e: e.code is ErrorCode.CAPACITY_NEVER_FITS, match="can never fit"
        ) as exc_info:
            await ctrl.acquire(
                "vm-huge",
                memory_mb=memory_mb,
                cpu_cores=cpu_cores,
                disk_mb=0,
                timeout=_TIMEOUT,
                use_tcg=False,
                held=frozenset(),
            )

    error = exc_info.value
    assert error.action is ErrorAction.ROUTE
    assert (dict(error.required), dict(error.found)) == (required, found)
    async with asyncio.timeout(1.0):  # nothing was reserved: a request that fits is admitted at once
        await ctrl.release(
            await ctrl.acquire(
                "vm-ok", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
            )
        )


async def test_acquire_timeout_raises_capacity_error() -> None:
    """TransientError CAPACITY raised when timeout expires waiting for resources. The budgets
    show in whole MB and 0.1 core."""
    # Tight CPU budget: only 1 VM fits (1.37 host CPUs, 0 reserve, each VM 1.25 effective)
    ctrl = _make_controller(
        host_memory_mb=1000.0,
        reserve_ratio=0.0,
        memory_overcommit=1.0,
        host_cpu_count=1.37,
        cpu_overcommit=1.0,
        cpu_reserve_cores=0.0,
    )
    r1 = await ctrl.acquire(
        "vm-1", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
    )

    with pytest.raises(
        TransientError,
        check=lambda e: e.code is ErrorCode.CAPACITY,
        match=r"Resource admission timeout after 0\.2s for VM vm-2, blocked by CPU budget",
    ) as exc_info:
        await ctrl.acquire(
            "vm-2", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=0.2, use_tcg=False, held=frozenset()
        )

    context = exc_info.value.context
    assert (context["blocking_gate"], context["memory_budget_mb"], context["cpu_budget"]) == (
        "cpu_budget",
        "1000",
        "1.4",
    )
    await ctrl.release(r1)


# ============================================================================
# TCG Overhead
# ============================================================================


async def test_tcg_adds_extra_memory_overhead() -> None:
    """use_tcg=True adds the 128 MB TCG translation cache to the 80 MB overhead of the VM."""
    ctrl = _make_controller()

    r_kvm = await ctrl.acquire(
        "vm-kvm", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
    )
    assert r_kvm.memory_mb == 336  # 256 + 80

    r_tcg = await ctrl.acquire(
        "vm-tcg", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=True, held=frozenset()
    )
    assert r_tcg.memory_mb == 464  # 256 + 80 + 128

    await ctrl.release(r_kvm)
    await ctrl.release(r_tcg)


# ============================================================================
# Release & Idempotency
# ============================================================================


def _one_vm_controller(count: int = 1) -> ResourceAdmissionController:
    """A controller whose memory and CPU budgets hold exactly `count` VMs of 256 MB and one
    core (336 MB and 1.25 cores each)."""
    return _make_controller(
        host_memory_mb=336.0 * count,
        reserve_ratio=0.0,
        memory_overcommit=1.0,
        host_cpu_count=1.25 * count,
        cpu_reserve_cores=0.0,
        cpu_overcommit=1.0,
    )


async def test_release_frees_resources() -> None:
    """Release returns the memory and the CPU of the VM to the budget: the same VM fits again."""
    ctrl = _one_vm_controller()
    r = await ctrl.acquire(
        "vm-1", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
    )
    assert await _blocking_gate_of(ctrl) == "memory_budget"

    await ctrl.release(r)
    assert await _blocking_gate_of(ctrl) is None


async def test_double_release_is_idempotent() -> None:
    """Releasing the same reservation twice gives its resources back once. Another VM stays
    admitted meanwhile (the counters snap to zero only once no VM is left): after the double
    release, one VM fits, not two."""
    ctrl = _one_vm_controller(count=2)
    other = await ctrl.acquire(
        "vm-other", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
    )
    r = await ctrl.acquire(
        "vm-1", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
    )
    await ctrl.release(r)
    await ctrl.release(r)  # a no-op

    again = await ctrl.acquire(
        "vm-2", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
    )
    assert await _blocking_gate_of(ctrl) == "memory_budget"
    await ctrl.release(again)
    await ctrl.release(other)


# ============================================================================
# Reservation Key Uniqueness
# ============================================================================


async def test_concurrent_same_vm_id_no_collision() -> None:
    """Multiple reservations with the same vm_id get unique reservation_ids, and each release
    gives back its own: once both are released, two VMs fit again."""
    ctrl = _one_vm_controller(count=2)

    # Acquire two reservations with the same vm_id (simulates concurrent requests)
    r1 = await ctrl.acquire(
        "same-id", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
    )
    r2 = await ctrl.acquire(
        "same-id", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
    )
    assert r1.reservation_id != r2.reservation_id

    # Release first: r2 still holds its share, so one VM fits, not two
    await ctrl.release(r1)
    r3 = await ctrl.acquire(
        "vm-3", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
    )
    assert await _blocking_gate_of(ctrl) == "memory_budget"

    await ctrl.release(r2)
    await ctrl.release(r3)
    both = [
        await ctrl.acquire(
            f"vm-{i}", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
        )
        for i in range(2)
    ]
    for r in both:
        await ctrl.release(r)


# ============================================================================
# Graceful Degradation (psutil failure)
# ============================================================================


async def test_psutil_failure_degrades_to_unlimited() -> None:
    """When cgroup and psutil both fail at start(), the budgets are unlimited: no admission
    control. Gates 3a and 3b read nothing either: any request is admitted."""
    ctrl = _probed_controller()
    with (
        patch("exec_sandbox.admission.detect_cgroup_memory_limit_mb", return_value=None),
        patch("exec_sandbox.admission.detect_cgroup_cpu_limit", return_value=None),
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=OSError("no psutil")),
        patch("exec_sandbox.admission.psutil.cpu_count", side_effect=OSError("no psutil")),
        patch("exec_sandbox.admission.detect_host_os", return_value=HostOS.UNKNOWN),  # no pressure probe
    ):
        await ctrl.start()
        reservations = [
            await ctrl.acquire(
                f"vm-{i}",
                memory_mb=10**9,
                cpu_cores=10**4,
                disk_mb=0,
                timeout=_REFUSED,
                use_tcg=False,
                held=frozenset(),
            )
            for i in range(3)
        ]
    for r in reservations:
        await ctrl.release(r)
    await ctrl.stop()


# ============================================================================
# Float Drift Prevention
# ============================================================================


async def test_counters_snap_to_zero_when_empty() -> None:
    """After releasing all reservations, the counters are exactly zero: a VM that takes the
    whole budget fits.

    The reservations of 0.7, 0.4 and 0.3 (MB and cores; 80.7 MB and 0.95 core with the
    overheads), added then taken off in that order, leave +4.3e-14 MB and +3.3e-16 core in
    floats. With no snap to zero, a VM of exactly the budget (336 MB, 2.5 cores) would not fit.
    """
    ctrl = _make_controller(
        host_memory_mb=336.0,
        reserve_ratio=0.0,
        memory_overcommit=1.0,
        host_cpu_count=2.5,
        cpu_reserve_cores=0.0,
        cpu_overcommit=1.0,
    )
    sizes = [0.7, 0.4, 0.3]
    held = [
        await ctrl.acquire(
            f"vm-{i}", memory_mb=size, cpu_cores=size, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
        )
        for i, size in enumerate(sizes)
    ]
    for r in held:
        await ctrl.release(r)

    whole = await ctrl.acquire(
        "vm-whole", memory_mb=256, cpu_cores=2.25, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
    )
    await ctrl.release(whole)


# ============================================================================
# Cgroup-Aware Startup (L1)
# ============================================================================


async def test_cgroup_memory_limit_used_as_capacity() -> None:
    """When cgroup limits are detected, they are the host memory and CPUs of the budgets."""
    ctrl = _probed_controller()

    with (
        patch("exec_sandbox.admission.detect_cgroup_memory_limit_mb", return_value=4096.0),
        patch("exec_sandbox.admission.detect_cgroup_cpu_limit", return_value=2.0),
    ):
        await ctrl.start()

    # 4096 * 0.9 * 1.0 = 3686.4; (2 - 0.5) * 1.0 = 1.5
    assert await _budgets(ctrl) == {"memory_mb": "3686", "cpu": "1.50"}
    await ctrl.stop()


async def test_cgroup_falls_through_to_psutil_when_unlimited() -> None:
    """When cgroup returns None, psutil is used as fallback."""
    ctrl = _probed_controller()

    mock_vmem = SimpleNamespace(total=8 * 1024 * 1024 * 1024)  # 8GB
    with (
        patch("exec_sandbox.admission.detect_cgroup_memory_limit_mb", return_value=None),
        patch("exec_sandbox.admission.detect_cgroup_cpu_limit", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", return_value=mock_vmem),
        patch("exec_sandbox.admission.psutil.cpu_count", return_value=4),
    ):
        await ctrl.start()

    # 8192 * 0.9 * 1.0 = 7372.8; (4 - 0.5) * 1.0 = 3.5
    assert await _budgets(ctrl) == {"memory_mb": "7372", "cpu": "3.50"}
    await ctrl.stop()


async def test_manual_override_beats_cgroup() -> None:
    """Manual host_memory_mb/host_cpu_count bypass cgroup + psutil: start() keeps them."""
    ctrl = _make_controller(
        host_memory_mb=32_000.0,
        host_cpu_count=16.0,
        reserve_ratio=0.1,
        memory_overcommit=1.0,
        cpu_reserve_cores=0.5,
        cpu_overcommit=1.0,
    )

    with (
        patch("exec_sandbox.admission.detect_cgroup_memory_limit_mb", return_value=4096.0),
        patch("exec_sandbox.admission.detect_cgroup_cpu_limit", return_value=2.0),
    ):
        await ctrl.start()

    # 32000 * 0.9 * 1.0 = 28800; (16 - 0.5) * 1.0 = 15.5
    assert await _budgets(ctrl) == {"memory_mb": "28800", "cpu": "15.50"}
    await ctrl.stop()


_GIB_VMEM = SimpleNamespace(total=8 * 1024 * 1024 * 1024)


@pytest.mark.parametrize(
    ("given", "cgroup", "psutil_probes", "budgets"),
    [
        # memory given: 32000 * 0.9 = 28800; CPUs from the cgroup: 2 - 0.5 = 1.5
        pytest.param(
            {"host_memory_mb": 32_000.0},
            (4096.0, 2.0),
            (_GIB_VMEM, 4),
            {"memory_mb": "28800", "cpu": "1.50"},
            id="memory-given-cgroup",
        ),
        # memory from the cgroup: 4096 * 0.9 = 3686.4; CPUs given: 16 - 0.5 = 15.5
        pytest.param(
            {"host_cpu_count": 16.0},
            (4096.0, 2.0),
            (_GIB_VMEM, 4),
            {"memory_mb": "3686", "cpu": "15.50"},
            id="cpus-given-cgroup",
        ),
        # no cgroup limit: CPUs from psutil, 4 - 0.5 = 3.5
        pytest.param(
            {"host_memory_mb": 32_000.0},
            (None, None),
            (_GIB_VMEM, 4),
            {"memory_mb": "28800", "cpu": "3.50"},
            id="memory-given-psutil",
        ),
        # no cgroup limit: memory from psutil, 8192 * 0.9 = 7372.8
        pytest.param(
            {"host_cpu_count": 16.0},
            (None, None),
            (_GIB_VMEM, 4),
            {"memory_mb": "7372", "cpu": "15.50"},
            id="cpus-given-psutil",
        ),
        # the psutil CPU probe fails: no CPU limit (not a budget from 1 CPU)
        pytest.param(
            {"host_memory_mb": 32_000.0},
            (None, None),
            (_GIB_VMEM, OSError("no psutil")),
            {"memory_mb": "28800", "cpu": None},
            id="memory-given-psutil-fails",
        ),
        # the psutil memory probe fails: no memory limit (not a budget of 0)
        pytest.param(
            {"host_cpu_count": 16.0},
            (None, None),
            (OSError("no psutil"), 4),
            {"memory_mb": None, "cpu": "15.50"},
            id="cpus-given-psutil-fails",
        ),
        # nothing given, the memory probe fails: the CPU probe still runs, 4 - 0.5 = 3.5
        pytest.param(
            {},
            (None, None),
            (OSError("no psutil"), 4),
            {"memory_mb": None, "cpu": "3.50"},
            id="memory-probe-fails-cpus-counted",
        ),
        # psutil cannot count the CPUs (None): no CPU limit, not a budget from 1 CPU
        pytest.param(
            {"host_memory_mb": 32_000.0},
            (None, None),
            (_GIB_VMEM, None),
            {"memory_mb": "28800", "cpu": None},
            id="cpu-count-none",
        ),
    ],
)
async def test_one_manual_override_stands_and_the_other_is_probed(
    given: dict[str, float],
    cgroup: tuple[float | None, float | None],
    psutil_probes: tuple[object, object],
    budgets: dict[str, str | None],
) -> None:
    """Only one of host_memory_mb and host_cpu_count given: start() keeps it and probes only
    the other (cgroup limit, else psutil), never replaces the given one. A dimension no
    probe can size has no limit, as when every probe fails: the error of a request
    above both budgets then gives None for it."""
    ctrl = ResourceAdmissionController(
        disk_dir=_DISK_DIR,
        disk_free_floor_mb=0,
        memory_overcommit_ratio=1.0,
        cpu_overcommit_ratio=1.0,
        disk_overcommit_ratio=DEFAULT_DISK_OVERCOMMIT_RATIO,
        host_memory_reserve_ratio=0.1,
        available_memory_floor_mb=0,
        host_cpu_reserve_cores=0.5,
        **{"host_memory_mb": None, "host_cpu_count": None, **given},
    )

    def probe(result: object) -> dict[str, object]:
        return {"side_effect": result} if isinstance(result, BaseException) else {"return_value": result}

    with (
        patch("exec_sandbox.admission.detect_cgroup_memory_limit_mb", return_value=cgroup[0]),
        patch("exec_sandbox.admission.detect_cgroup_cpu_limit", return_value=cgroup[1]),
        patch("exec_sandbox.admission.psutil.virtual_memory", **probe(psutil_probes[0])),
        patch("exec_sandbox.admission.psutil.cpu_count", **probe(psutil_probes[1])),
    ):
        await ctrl.start()

    assert await _budgets(ctrl) == budgets
    await ctrl.stop()


@pytest.mark.parametrize(
    ("given_mb", "cgroup_mb", "vmem", "source"),
    [
        pytest.param(32_000.0, 4096.0, _GIB_VMEM, "manual", id="manual"),
        pytest.param(None, 4096.0, _GIB_VMEM, "cgroup", id="cgroup"),
        pytest.param(None, None, _GIB_VMEM, "psutil", id="psutil"),
        pytest.param(None, None, OSError("no psutil"), "none", id="none"),
    ],
)
async def test_start_logs_where_the_memory_size_came_from(
    given_mb: float | None,
    cgroup_mb: float | None,
    vmem: object,
    source: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The "Host resources detected" line names the source of the memory size: the given
    setting, the cgroup limit, psutil, or none (no memory limit). The CPUs are probed."""
    ctrl = ResourceAdmissionController(
        disk_dir=_DISK_DIR,
        disk_free_floor_mb=0,
        memory_overcommit_ratio=1.0,
        cpu_overcommit_ratio=1.0,
        disk_overcommit_ratio=DEFAULT_DISK_OVERCOMMIT_RATIO,
        host_memory_reserve_ratio=0.1,
        available_memory_floor_mb=0,
        host_cpu_reserve_cores=0.5,
        host_memory_mb=given_mb,
        host_cpu_count=None,
    )
    with (
        patch("exec_sandbox.admission.detect_cgroup_memory_limit_mb", return_value=cgroup_mb),
        patch("exec_sandbox.admission.detect_cgroup_cpu_limit", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=[vmem]),
        patch("exec_sandbox.admission.psutil.cpu_count", return_value=4),
        caplog.at_level("INFO", logger="exec_sandbox.admission"),
    ):
        await ctrl.start()
    await ctrl.stop()

    [detected] = [record for record in caplog.records if record.getMessage() == "Host resources detected"]
    assert detected.capacity_source == source  # pyright: ignore[reportAttributeAccessIssue]


# ============================================================================
# Gate 3: Available-Memory Floor (inline probe)
# ============================================================================


def _mock_vmem(available_mb: float) -> object:
    """Create a mock psutil vmem result with given available MB."""
    return SimpleNamespace(available=int(available_mb * 1024 * 1024))


@contextmanager
def _mock_system_memory(available_mb: float) -> Generator[None]:
    """Gate 3a reads `available_mb` (psutil, no cgroup limit).

    Gate 3b reads nothing (a host OS with no pressure probe), so that real host pressure
    on CI runners does not change the outcome of the Gate 3a tests.
    """
    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", return_value=_mock_vmem(available_mb)),
        patch("exec_sandbox.admission.detect_host_os", return_value=HostOS.UNKNOWN),
    ):
        yield


@contextmanager
def _mock_dynamic_system_memory(initial_mb: float) -> Generator[list[float]]:
    """Mock Gate 3a probe with mutable available memory.

    Returns a single-element list ``[mb]`` — mutate ``ref[0]`` to simulate
    system memory changing between acquire/release calls.

    Gate 3b reads nothing (a host OS with no pressure probe), as in _mock_system_memory.
    """
    ref = [initial_mb]

    def _fake_vmem() -> object:
        return _mock_vmem(ref[0])

    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=_fake_vmem),
        patch("exec_sandbox.admission.detect_host_os", return_value=HostOS.UNKNOWN),
    ):
        yield ref


@pytest.mark.parametrize(
    ("floor_mb", "available_mb", "admitted"),
    [
        pytest.param(512, 848.0, True, id="floor-512-left-at-the-floor"),  # 848 - 336 = 512
        pytest.param(512, 847.0, False, id="floor-512-left-below-the-floor"),
        pytest.param(512, 300.0, False, id="floor-512-no-room"),
        pytest.param(0, 336.0, True, id="floor-0-exact-fit"),
        pytest.param(0, 335.0, False, id="floor-0-short-by-1-mb"),
    ],
)
async def test_available_memory_floor(floor_mb: int, available_mb: float, admitted: bool) -> None:
    """Gate 3a admits while the system available memory less the request (336 MB) stays at the
    floor or above. A large budget (gates 1-2 pass): the gate gives backpressure across
    processes. A refusal names the gate and its figures."""
    ctrl = _make_controller(host_memory_mb=100_000.0, available_memory_floor_mb=floor_mb)

    with _mock_system_memory(available_mb):
        try:
            reservation = await ctrl.acquire(
                "vm-1", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
            )
        except TransientError as e:
            assert not admitted
            assert e.code is ErrorCode.CAPACITY
            assert e.context["blocking_gate"] == "available_memory"
            assert e.context["available_memory_floor_mb"] == floor_mb
            assert e.context["system_available_memory_mb"] == round(available_mb)
        else:
            assert admitted
            await ctrl.release(reservation)


@pytest.mark.parametrize(
    "error",
    [OSError("no psutil"), psutil.Error("access denied"), RuntimeError("platform error"), AttributeError("no field")],
    ids=["os-error", "psutil-error", "runtime-error", "attribute-error"],
)
async def test_probe_failure_allows_admission(error: Exception) -> None:
    """When the cgroup and psutil probes fail, Gate 3a reads no figure and admits, even with a
    floor (graceful degradation): the error does not come out of acquire()."""
    ctrl = _make_controller(available_memory_floor_mb=512)

    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=error),
        patch("exec_sandbox.admission.detect_host_os", return_value=HostOS.UNKNOWN),
    ):
        assert await _blocking_gate_of(ctrl) is None


async def test_cgroup_probe_preferred_over_psutil() -> None:
    """Gate 3a uses the cgroup available memory when present: a container with 300 MB left is
    refused, though psutil sees a host with 100 GB."""
    ctrl = _make_controller(available_memory_floor_mb=0)

    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=300.0),
        patch("exec_sandbox.admission.psutil.virtual_memory", return_value=_mock_vmem(100_000.0)),
        patch("exec_sandbox.admission.detect_host_os", return_value=HostOS.UNKNOWN),
        pytest.raises(TransientError, check=lambda e: e.context["blocking_gate"] == "available_memory") as exc_info,
    ):
        await ctrl.acquire(
            "vm-1", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
        )
    assert exc_info.value.context["system_available_memory_mb"] == 300


# ============================================================================
# Stop
# ============================================================================


async def test_stop_is_idempotent() -> None:
    """start() runs the self-wake timer; stop() ends it, and a second stop() is a no-op. No
    task of the controller is left, and a later acquire() is refused (SCHEDULER_STATE)."""
    ctrl = _make_controller()
    before = asyncio.all_tasks()
    await ctrl.start()
    assert len(asyncio.all_tasks() - before) == 1  # the self-wake timer

    await ctrl.stop()
    await ctrl.stop()  # Second stop is a no-op
    assert asyncio.all_tasks() == before
    with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE):
        await ctrl.acquire(
            "vm-1", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
        )


async def test_a_cancelled_stop_raises_its_cancel() -> None:
    """A cancel of the stop() caller while stop() waits for the end of its timer comes out as
    CancelledError: the wait for the timer must not take the caller's cancel for the timer's."""
    ctrl = _make_controller()
    await ctrl.start()
    stopping = asyncio.create_task(ctrl.stop())
    await asyncio.sleep(0)  # stop() now waits for the end of the timer it cancelled
    stopping.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await stopping
    finally:
        await ctrl.stop()


async def test_a_start_after_stop_starts_no_timer() -> None:
    """start() after stop() leaves no task: nothing would end the self-wake timer of a stopped
    controller (stop() ran already)."""
    ctrl = _make_controller()
    before = asyncio.all_tasks()
    await ctrl.stop()
    await ctrl.start()
    assert asyncio.all_tasks() == before


# ============================================================================
# Self-Wake Timer
# ============================================================================


@pytest.mark.parametrize("gate", ["available_memory", "memory_pressure"])
async def test_self_wake_timer_unblocks_gate3_waiters(gate: str) -> None:
    """The self-wake timer makes a waiter that gate 3a or 3b blocks probe the host again: once
    the host recovers (another process frees memory, the pressure ends), it is admitted with
    no release here. The timer runs every 50 ms here, not every 10 s."""
    ctrl = _make_controller(available_memory_floor_mb=0)
    available_mb = [100.0 if gate == "available_memory" else 8192.0]  # 100 MB: under the 336 MB asked
    level = [4 if gate == "memory_pressure" else 1]  # macOS CRITICAL, then NORMAL

    with (
        patch("exec_sandbox.admission.GATE3_SELF_WAKE_INTERVAL_SECONDS", 0.05),
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=lambda: _mock_vmem(available_mb[0])),
        patch("exec_sandbox.admission.detect_host_os", return_value=HostOS.MACOS),
        patch("exec_sandbox.admission._read_macos_memory_pressure_level", side_effect=lambda: level[0]),
    ):
        await ctrl.start()
        acquire_task = asyncio.create_task(
            ctrl.acquire(
                "vm-1", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=5.0, use_tcg=False, held=frozenset()
            ),
        )
        await _until_waiting_or_done(ctrl, acquire_task)
        assert not acquire_task.done()

        # The host recovers (no local release)
        available_mb[0] = 8192.0
        level[0] = 1

        r = await asyncio.wait_for(acquire_task, timeout=2.0)
        assert r.vm_id == "vm-1"
        await ctrl.release(r)
        await ctrl.stop()


# ============================================================================
# Probe Cache
# ============================================================================


async def test_probe_cache_deduplicates_calls() -> None:
    """Gates 3a and 3b probe the host once per 100 ms: requests in a burst share one psutil
    read and one sysctl read."""
    ctrl = _make_controller()
    reads = {"psutil": 0, "sysctl": 0}

    def counting_vmem() -> object:
        reads["psutil"] += 1
        return _mock_vmem(4096.0)

    def counting_level() -> int:
        reads["sysctl"] += 1
        return 1

    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=None),
        patch("exec_sandbox.admission.psutil.virtual_memory", side_effect=counting_vmem),
        patch("exec_sandbox.admission.detect_host_os", return_value=HostOS.MACOS),
        patch("exec_sandbox.admission._read_macos_memory_pressure_level", side_effect=counting_level),
    ):
        reservations = [
            await ctrl.acquire(
                f"vm-{i}", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
            )
            for i in range(5)
        ]
    for r in reservations:
        await ctrl.release(r)
    assert reads == {"psutil": 1, "sysctl": 1}


# ============================================================================
# Memory Pressure (Gate 3b)
# ============================================================================


@contextmanager
def _pressure(
    host_os: HostOS, *, level: int | None = None, psi: str | type[Exception] | Exception = ""
) -> Generator[None]:
    """Gate 3b on a `host_os` host: the macOS sysctl reads `level` (None: the call fails), the
    Linux PSI file holds `psi` (an exception: its read raises it). Gate 3a reads 8192 MB
    available (a cgroup figure). Only /proc/pressure/memory holds `psi`: a probe that reads
    another file (/proc/pressure/cpu, /proc/pressure/io) reads that real file."""
    real_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) != "/proc/pressure/memory":
            return real_read_text(path, *args, **kwargs)
        if isinstance(psi, str):
            return psi
        raise psi

    with (
        patch("exec_sandbox.admission.read_container_available_memory_mb", return_value=8192.0),
        patch("exec_sandbox.admission.detect_host_os", return_value=host_os),
        patch("exec_sandbox.admission._read_macos_memory_pressure_level", return_value=level),
        patch("exec_sandbox.admission.Path.read_text", autospec=True, side_effect=read_text),
    ):
        yield


def _psi(full_avg10: str) -> str:
    """/proc/pressure/memory with `full_avg10` as its full avg10 value."""
    return f"some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10={full_avg10} avg60=0.00 avg300=0.00 total=0\n"


class TestMemoryPressureGate:
    """Gate 3b: memory pressure detection (cross-platform), through acquire().

    The host OS and its pressure source are faked at the OS call: the macOS sysctl
    (kern.memorystatus_vm_pressure_level) and the read of /proc/pressure/memory on Linux.
    A refusal names the gate: "memory_pressure".
    """

    # ---- macOS: kern.memorystatus_vm_pressure_level sysctl ----

    @pytest.mark.parametrize(
        ("level", "blocked"),
        [
            (1, False),  # NORMAL → admit
            (2, False),  # WARN → admit (common during normal use)
            (4, True),  # CRITICAL → reject
        ],
        ids=["NORMAL", "WARN", "CRITICAL"],
    )
    async def test_macos_known_levels(self, level: int, blocked: bool) -> None:
        """Known macOS pressure levels: only CRITICAL refuses."""
        with _pressure(HostOS.MACOS, level=level):
            assert await _blocking_gate_of(_make_controller()) == ("memory_pressure" if blocked else None)

    @pytest.mark.parametrize(
        "level",
        [0, 3, 5, 6, 7, 8, 16, -1, 2**31 - 1, 99, None],
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
            "sysctl-failure",
        ],
    )
    async def test_macos_unexpected_levels_skip_the_gate(self, level: int | None) -> None:
        """Unknown sysctl values, or a failed sysctl call, skip the gate (defensive)."""
        with _pressure(HostOS.MACOS, level=level):
            assert await _blocking_gate_of(_make_controller()) is None

    @skip_unless_macos
    def test_macos_real_probe(self) -> None:
        """The real sysctl call returns a known level on this host (NORMAL, WARN or CRITICAL).
        It is read directly: through acquire() the outcome depends on the pressure of the host."""
        assert _read_macos_memory_pressure_level() in {1, 2, 4}

    # ---- Linux: PSI (Pressure Stall Information) ----

    @pytest.mark.parametrize(
        ("avg10", "blocked"),
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
    async def test_linux_psi_threshold_boundary(self, avg10: str, blocked: bool) -> None:
        """PSI full avg10 boundary values around the 10% threshold."""
        with _pressure(HostOS.LINUX, psi=_psi(avg10)):
            assert await _blocking_gate_of(_make_controller()) == ("memory_pressure" if blocked else None)

    @pytest.mark.parametrize(
        "psi",
        [
            _psi("inf"),
            _psi("-inf"),
            _psi("nan"),
            _psi("abc"),  # not a number
            "",  # empty file
            "some avg10=1.00 avg60=0.00 avg300=0.00 total=0\n",  # no "full" line
            "garbage data that doesn't match any pattern",  # garbage
            "full\n",  # "full" line with no tokens
            "full avg60=50.00 avg300=50.00 total=0\n",  # "full" line but no avg10
            FileNotFoundError,
            OSError("permission denied"),
            PermissionError,
        ],
        ids=[
            "inf",
            "neg-inf",
            "nan",
            "not-a-number",
            "empty",
            "no-full-line",
            "garbage",
            "full-no-tokens",
            "full-no-avg10",
            "file-not-found",
            "os-error",
            "permission-denied",
        ],
    )
    async def test_linux_psi_unreadable_skips_the_gate(self, psi: str | type[Exception] | Exception) -> None:
        """A PSI file that is missing, unreadable, malformed, or gives a value that is not finite
        skips the gate."""
        with _pressure(HostOS.LINUX, psi=psi):
            assert await _blocking_gate_of(_make_controller()) is None

    @skip_unless_linux
    def test_linux_psi_real_probe(self) -> None:
        """On real Linux, the PSI probe gives a verdict wherever the kernel serves the PSI file
        (CONFIG_PSI, not psi=0). It is read directly: through acquire() the outcome depends on
        the pressure of the host."""
        result = ResourceAdmissionController._probe_psi_linux()  # pyright: ignore[reportPrivateUsage]
        try:
            Path("/proc/pressure/memory").read_text()
        except OSError:
            assert result is False  # no reading: the gate admits
        else:
            assert isinstance(result, bool)

    # ---- Other host OS ----

    async def test_unknown_os_skips_the_gate(self) -> None:
        """On a host OS with no pressure probe, the gate is skipped, whatever the sysctl or the
        PSI file would say."""
        with _pressure(HostOS.UNKNOWN, level=4, psi=_psi("100.00")):
            assert await _blocking_gate_of(_make_controller()) is None


# ============================================================================
# Multi-Scheduler (cross-process backpressure)
# ============================================================================


@pytest.mark.parametrize(("floor_mb", "expected"), [(0, 6), (512, 4)], ids=["floor-0", "floor-512"])
async def test_n_controllers_converge_to_system_limit(floor_mb: int, expected: int) -> None:
    """N controllers each claiming full budget converge to the real system capacity.

    Four controllers think they own a 16 GB machine; the system has 2048 MB available, and
    each admitted VM takes its 336 MB from it. Floor 0: 2048, 1712, 1376, 1040, 704, 368 MB
    admit (368 - 336 = 32), 32 MB does not: 6 VMs. Floor 512: a VM needs 848 MB available:
    4 VMs, then 704 MB refuses. Gates 1-2 would admit 12 VMs per controller (the CPU
    budget: (8 - 0.5) * 2 = 15 cores, 1.25 per VM).
    """
    n_controllers = 4
    controllers = [
        _make_controller(host_memory_mb=16_000.0, available_memory_floor_mb=floor_mb) for _ in range(n_controllers)
    ]

    reservations: list[tuple[ResourceAdmissionController, ResourceReservation]] = []
    refused: TransientError | None = None

    with _mock_dynamic_system_memory(2048.0) as available_mb, _probe_clock() as later:
        # Round-robin across controllers until system is full
        for i in range(20):  # try up to 20 VMs
            ctrl = controllers[i % n_controllers]
            later()
            try:
                r = await ctrl.acquire(
                    f"vm-{i}",
                    memory_mb=256,
                    cpu_cores=_CPU,
                    disk_mb=0,
                    timeout=_REFUSED,
                    use_tcg=False,
                    held=frozenset(),
                )
            except TransientError as e:
                refused = e
                break  # System full
            available_mb[0] -= 336.0
            reservations.append((ctrl, r))

    assert len(reservations) == expected
    assert refused is not None
    assert refused.context["blocking_gate"] == "available_memory"

    for ctrl, r in reservations:
        await ctrl.release(r)


# ============================================================================
# Shutdown / Stop
# ============================================================================


async def test_stop_unblocks_waiting_acquires() -> None:
    """stop() immediately unblocks blocked acquire() callers with InvalidInputError SCHEDULER_STATE."""
    # Tight budget: only 1 VM fits (1.5 host CPUs, 0 reserve, each VM 1.25 effective)
    ctrl = _make_controller(host_cpu_count=1.5, cpu_overcommit=1.0, cpu_reserve_cores=0.0)

    r1 = await ctrl.acquire(
        "vm-1", memory_mb=256, cpu_cores=1.0, disk_mb=0, timeout=_TIMEOUT, use_tcg=False, held=frozenset()
    )

    # Second acquire blocks (no CPU left)
    acquire_task = asyncio.create_task(
        ctrl.acquire("vm-2", memory_mb=256, cpu_cores=1.0, disk_mb=0, timeout=120.0, use_tcg=False, held=frozenset()),
    )
    await _until_waiting_or_done(ctrl, acquire_task)
    assert not acquire_task.done()

    # stop() should immediately unblock the waiter
    await ctrl.stop()

    with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE, match="stopped"):
        await asyncio.wait_for(acquire_task, timeout=2.0)

    await ctrl.release(r1)


# ============================================================================
# Gate 4: home disk space
# ============================================================================


class _Disk:
    """The free space gate 4 reads (shutil.disk_usage), set by the test; `written_mb`: what the
    home disks of the controller wrote (their st_blocks, faked by `patched`)."""

    def __init__(self, free_mb: float) -> None:
        self.free_mb = free_mb
        self.written_mb = 0.0
        self.read = asyncio.Event()  # set at each read of the free space

    def usage(self, _path: object) -> SimpleNamespace:
        self.read.set()
        return SimpleNamespace(free=int(self.free_mb * _MIB))

    def allocated(self, _st: os.stat_result) -> int:
        """allocated_bytes of the one home disk in disk_dir: `written_mb`."""
        return int(self.written_mb * _MIB)

    @contextmanager
    def patched(self, disk_dir: Path) -> Generator[None]:
        """Gate 4 reads this disk: its free space (statvfs), and one home disk file in
        `disk_dir` whose allocated blocks (st_blocks) are `written_mb`."""
        (disk_dir / "vm.img").touch()
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", self.usage),
            patch("exec_sandbox.admission.allocated_bytes", self.allocated),
        ):
            yield


class _Cache:
    """A cache whose entries, oldest first, give `gives_back` of their size back to `disk` when
    evicted, `after` seconds later (XFS frees removed files in the background); 0: nothing back
    (their blocks still in use: a CoW clone, a reader without a lease). With `holds`, the
    space of each eviction shows only at show() (a ZFS transaction sync): the passes before
    it see nothing, whatever the clock."""

    def __init__(
        self,
        cache_dir: Path,
        disk: _Disk,
        sizes_mb: list[float],
        *,
        gives_back: float = 1.0,
        after: float = 0.0,
        holds: bool = False,
    ) -> None:
        cache_dir.mkdir(mode=0o700, exist_ok=True)
        self.cache_dir = cache_dir
        self._disk = disk
        self._gives_back = gives_back
        self._after = after
        self._holds = holds
        self._held_mb = 0.0  # evicted, held until show()
        self.entries = list(sizes_mb)
        self.evicted: list[float] = []
        self.size_read = asyncio.Event()
        self.size_read_calls: list[float] = []
        self.evicted_one = asyncio.Event()
        self.orders = 0  # eviction passes asked of this cache
        self.passes: list[list[float]] = []  # the sizes evicted by each pass (each order)

    async def evict_oldest(self) -> int | None:
        if not self.entries:
            return None
        size = self.entries.pop(0)
        self.evicted.append(size)
        if self.passes:
            self.passes[-1].append(size)
        self.evicted_one.set()

        if self._holds:
            self._held_mb += size * self._gives_back
            return int(size * _MIB)

        def give_back() -> None:
            self._disk.free_mb += size * self._gives_back

        if self._after:
            asyncio.get_running_loop().call_later(self._after, give_back)
        else:
            give_back()
        return int(size * _MIB)

    def show(self) -> None:
        """Give back the space held so far (`holds`)."""
        self._disk.free_mb += self._held_mb
        self._held_mb = 0.0

    def eviction_order(self) -> SimpleNamespace:
        """A pass: each evict_next is this fake's evict_oldest (it keeps no listing)."""
        self.orders += 1
        self.passes.append([])
        return SimpleNamespace(evict_next=self.evict_oldest)

    async def reclaimable_bytes(self, *, held: frozenset[Path]) -> int:
        """Yields once: requests made together register before the first pass sizes itself.
        The real listing is a thread job, which yields nothing on 3.13+ when it ends before
        its await: there the first pass can size itself only (the next passes cover the rest)."""
        del held  # no lease paths here: every entry counts
        await asyncio.sleep(0)
        self.size_read.set()
        self.size_read_calls.append(asyncio.get_running_loop().time())
        return int(sum(self.entries) * _MIB)


_OVERLAY = _Settle(quiet=0.02, limit=0.1, sync_once=True, later=0.0, counts_writes=False)
"""_settle_removals of an overlay (_OTHER), with test-sized times."""


def _zfs(later: float) -> _Settle:
    """_settle_removals of ZFS (one step at the next transaction sync, at most `later` s away)."""
    return _Settle(quiet=0.0, limit=0.0, sync_once=False, later=later, counts_writes=True)


class _GivesBack(_Cache):
    """A cache whose entries are (size MB, part of it given back at the eviction), oldest first."""

    def __init__(self, cache_dir: Path, disk: _Disk, entries: list[tuple[float, float]]) -> None:
        super().__init__(cache_dir, disk, [size for size, _ in entries], gives_back=0)
        self._ratios = [ratio for _, ratio in entries]

    async def evict_oldest(self) -> int | None:
        if not self.entries:
            return None
        ratio = self._ratios.pop(0)
        freed = await super().evict_oldest()
        self._disk.free_mb += self.evicted[-1] * ratio
        return freed


class _SizedDisk:
    """Free space of a filesystem of `capacity_mb` that holds the entries of `caches`."""

    def __init__(self, capacity_mb: float, caches: list[SizedCache]) -> None:
        self._capacity_mb = capacity_mb
        self._caches = caches

    def usage(self, _path: object) -> SimpleNamespace:
        used = sum(  # the sizes the cache keeps: a read of an entry would set its atime
            size for cache in self._caches for name, size in cache.sizes.items() if (cache.cache_dir / name).exists()
        )
        return SimpleNamespace(free=int((self._capacity_mb - used) * _MIB))


class TestDiskGate:
    """Sparse home disks: each admitted disk reserves a part of what its guest has not written yet.

    The free space (shutil.disk_usage), the caches (add_cache) and syncfs are faked at their
    calls. How the filesystem shows removals (_settle_removals, private) is patched with
    test-sized times: the filesystems (btrfs, XFS, ZFS, overlay) are not on the test host, and
    their real times are seconds. TestSettleRemovals covers the real call."""

    @pytest.fixture(autouse=True)
    async def _executor_thread_started(self) -> None:
        """The default executor of this test's event loop has its first thread: the start of a
        thread holds the event loop (15 ms under load), longer than the time windows here."""
        await asyncio.to_thread(int)

    @pytest.fixture(autouse=True)
    def _space_shows_at_removal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fake disks here show a removal in their free space when the fake cache evicts
        (a test that patches it again shows it later). TestSettleRemovals runs the real call."""
        monkeypatch.setattr("exec_sandbox.admission._settle_removals", lambda _path: _AT_REMOVAL)

    @staticmethod
    def _controller(disk_dir: Path, *, floor_mb: int, ratio: float = 4.0) -> ResourceAdmissionController:
        """Memory and CPU budgets that hold every request here. Disk overcommit ratio 4 unless
        `ratio`: a 1024 MB disk reserves 256 MB, a 400 MB disk 100 MB."""
        return ResourceAdmissionController(
            disk_dir=disk_dir,
            disk_free_floor_mb=floor_mb,
            memory_overcommit_ratio=1.0,
            cpu_overcommit_ratio=1.0,
            disk_overcommit_ratio=ratio,
            host_memory_reserve_ratio=0.0,
            host_memory_mb=1_000_000.0,
            host_cpu_count=1000.0,
            available_memory_floor_mb=0,
            host_cpu_reserve_cores=DEFAULT_HOST_CPU_RESERVE_CORES,
        )

    @pytest.mark.parametrize(("ratio", "fits"), [(1.0, 1), (8.0, 12)])
    async def test_the_disk_overcommit_ratio_sets_the_reserve(self, tmp_path: Path, ratio: float, fits: int) -> None:
        """Each admitted disk reserves its part not written yet / disk_overcommit_ratio. In 1542
        MB of room, 1.0 reserves the whole 1024 MB disk (1 VM), 8.0 an eighth, 128 MB (12 VMs).
        test_sparse_disks_do_not_turn_free_disk_into_one_vm covers 4.0."""
        ctrl = self._controller(tmp_path, floor_mb=1024, ratio=ratio)
        with patch("exec_sandbox.admission.shutil.disk_usage", _Disk(2566).usage):
            held = [
                await ctrl.acquire(
                    f"vm-{i}",
                    memory_mb=1,
                    cpu_cores=_CPU,
                    disk_mb=1024,
                    timeout=_REFUSED,
                    use_tcg=False,
                    held=frozenset(),
                )
                for i in range(fits)
            ]
            with pytest.raises(TransientError, check=lambda e: e.context["blocking_gate"] == "home_disk_space"):
                await ctrl.acquire(
                    "vm-next",
                    memory_mb=1,
                    cpu_cores=_CPU,
                    disk_mb=1024,
                    timeout=_REFUSED,
                    use_tcg=False,
                    held=frozenset(),
                )
        for reservation in held:
            await ctrl.release(reservation)

    async def test_sparse_disks_do_not_turn_free_disk_into_one_vm(self, tmp_path: Path) -> None:
        """The CI case: 2566 MB free, floor 1024 MB, 1024 MB disks. A count of each whole disk admits 1 VM;
        the reserve (256 MB each) admits 6 at once (1536 of the 1542 MB of room), then names the gate that
        blocks."""
        ctrl = self._controller(tmp_path, floor_mb=1024)
        with patch("exec_sandbox.admission.shutil.disk_usage", _Disk(2566).usage):
            held = [
                await ctrl.acquire(
                    f"vm-{i}",
                    memory_mb=1,
                    cpu_cores=_CPU,
                    disk_mb=1024,
                    timeout=_REFUSED,
                    use_tcg=False,
                    held=frozenset(),
                )
                for i in range(6)
            ]
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.CAPACITY) as exc_info:
                await ctrl.acquire(
                    "vm-next",
                    memory_mb=1,
                    cpu_cores=_CPU,
                    disk_mb=1024,
                    timeout=_REFUSED,
                    use_tcg=False,
                    held=frozenset(),
                )
        for reservation in held:
            await ctrl.release(reservation)
        error = exc_info.value
        assert "blocked by home disk space" in str(error)
        assert "2566MB free, floor 1024MB, 6144MB of disks admitted (1536MB reserved)" in str(error)
        assert error.context["blocking_gate"] == "home_disk_space"
        assert error.context["disk_free_mb"] == 2566
        assert error.context["disk_free_floor_mb"] == 1024
        assert error.context["disk_admitted_mb"] == 6144
        assert error.context["disk_requested_mb"] == 1024

    async def test_written_blocks_are_not_counted_twice(self, tmp_path: Path) -> None:
        """A guest that wrote its whole disk took it from free space: no reserve is left for it.
        vm-1's 2000 MB disk is written in full, and the free space is down to 1100 MB. vm-2
        reserves 500 MB (floor 500 + 500 <= 1100); a reserve for vm-1 too (500 more) would not
        fit."""
        disk = _Disk(1100)
        disk.written_mb = 2000
        ctrl = self._controller(tmp_path, floor_mb=500)
        with disk.patched(tmp_path):
            first = await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=2000, timeout=_REFUSED, use_tcg=False, held=frozenset()
            )
            second = await ctrl.acquire(
                "vm-2", memory_mb=1, cpu_cores=_CPU, disk_mb=2000, timeout=_REFUSED, use_tcg=False, held=frozenset()
            )
        await ctrl.release(second)
        await ctrl.release(first)

    async def test_cloned_disk_shared_blocks_are_not_written(self, tmp_path: Path) -> None:
        """A clone's blocks shared with its L1 template take no space until overwritten: they
        give no room once the VM is gone."""
        disk_file = tmp_path / "vm.img"
        disk_file.write_bytes(os.urandom(4 * _MIB))
        ctrl = self._controller(tmp_path, floor_mb=0, ratio=256.0)
        with patch("exec_sandbox.admission.shutil.disk_usage", _Disk(2).usage), _probe_clock() as later:
            ctrl.note_shared_blocks("vm.img", disk_file.stat().st_blocks * 512)
            assert _written_mb(ctrl) == 0  # the shared blocks are not written
            disk_file.unlink()
            later()
            assert _written_mb(ctrl) == 0  # no disk file left (the next step shows the note went)
            disk_file.write_bytes(os.urandom(4 * _MIB))  # a new, fully written disk, same name
            later()
            assert _written_mb(ctrl) == pytest.approx(4, abs=0.1)  # the old note did not stay

    async def test_a_note_made_before_the_clone_survives_a_probe(self, tmp_path: Path) -> None:
        """The VM manager notes the shared blocks before the clone exists, so a probe in that
        window does not count the clone as written: that probe keeps the note of a disk
        not made yet, and the next one counts the clone's shared blocks as not written."""
        ctrl = self._controller(tmp_path, floor_mb=0, ratio=256.0)
        with patch("exec_sandbox.admission.shutil.disk_usage", _Disk(2).usage), _probe_clock() as later:
            ctrl.note_shared_blocks("vm.img", 4 * _MIB)
            assert _written_mb(ctrl) == 0  # a probe before the clone is made
            (tmp_path / "vm.img").write_bytes(os.urandom(4 * _MIB))
            later()
            assert _written_mb(ctrl) == 0  # the clone's shared blocks
            ctrl.forget_shared_blocks("vm.img")  # the clone was a plain copy
            later()
            assert _written_mb(ctrl) == pytest.approx(4, abs=0.1)

    async def test_unreadable_disk_dir_skips_the_gate(self, tmp_path: Path) -> None:
        """No free-space figure (disk_dir gone): the gate admits, it does not block every VM."""
        ctrl = self._controller(tmp_path / "gone", floor_mb=10**9)
        reservation = await ctrl.acquire(
            "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=2000, timeout=0.2, use_tcg=False, held=frozenset()
        )
        await ctrl.release(reservation)

    async def test_probe_reads_free_space_and_written_blocks(self, tmp_path: Path) -> None:
        """The probe reads the blocks the home disks (*.img) use; other files in disk_dir do not
        count."""
        (tmp_path / "vm.img").write_bytes(os.urandom(4 * _MIB))
        (tmp_path / "other.file").write_bytes(os.urandom(4 * _MIB))  # not a home disk
        ctrl = self._controller(tmp_path, floor_mb=0, ratio=256.0)
        with patch("exec_sandbox.admission.shutil.disk_usage", _Disk(2).usage):
            assert _written_mb(ctrl) == pytest.approx(4, abs=0.1)  # with other.file it would be 8
            probe = ctrl._probe_disk()  # pyright: ignore[reportPrivateUsage]
            assert probe is not None
            assert probe[0] == 2  # the free space, as the filesystem gives it

    async def test_eviction_gives_space_back_before_waiting(self, tmp_path: Path) -> None:
        """Gate 4 blocks: the request evicts the oldest entries, each cache in turn, until it fits;
        it neither waits nor empties the caches."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        l2 = _Cache(tmp_path / "l2", disk, [100, 100, 100])
        l1 = _Cache(tmp_path / "l1", disk, [200, 200])
        ctrl.add_cache(l2)
        ctrl.add_cache(l1)
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            await ctrl.release(
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=0.2, use_tcg=False, held=frozenset()
                )
            )
        assert (l2.evicted, l1.evicted) == ([100], [200])
        assert (l2.entries, l1.entries) == ([100, 100], [200])

    async def test_entries_in_use_are_skipped_and_free_ones_make_room(self, tmp_path: Path) -> None:
        """Real caches, the oldest entry of each in use by a VM (leased): eviction skips
        it and removes the free entries behind it, in either cache, until the request
        fits. The entries in use stay."""
        l2 = SizedCache(tmp_path / "l2")
        l1 = SizedCache(tmp_path / "l1")
        held_l2 = l2.add("held", 100)
        l2_free = [l2.add(f"free-{i}", 200) for i in range(2)]
        held_l1 = l1.add("held", 50)
        l1_free = l1.add("free", 300)
        disk = _SizedDisk(1000 + 850, [l2, l1])  # 1000 MB free; floor 1000: a 1024 MB disk needs 256 MB more
        ctrl = self._controller(tmp_path, floor_mb=1000)
        ctrl.add_cache(l2)
        ctrl.add_cache(l1)
        leases = [await CacheEntry(held_l2).lease(), await CacheEntry(held_l1).lease()]
        try:
            with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=1, use_tcg=False, held=frozenset()
                    )
                )
        finally:
            for lease in leases:
                assert lease is not None
                lease.release()
        assert held_l2.exists()
        assert held_l1.exists()
        assert (l2_free[0].exists(), l1_free.exists(), l2_free[1].exists()) == (False, False, True)

    async def test_oldest_entry_in_use_does_not_stop_the_evictions(self, tmp_path: Path) -> None:
        """The oldest entry of the only cache in use, or the whole L2 in use and the L1 free.
        Both admit (no LOCAL_DISK while free entries remain)."""
        only = SizedCache(tmp_path / "only")
        held = only.add("held", 50)
        only.add("free-1", 200)
        only.add("free-2", 200)
        lease = await CacheEntry(held).lease()
        assert lease is not None
        ctrl = self._controller(tmp_path, floor_mb=1000)
        ctrl.add_cache(only)
        try:
            with patch("exec_sandbox.admission.shutil.disk_usage", _SizedDisk(1000 + 450, [only]).usage):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=1, use_tcg=False, held=frozenset()
                    )
                )
        finally:
            lease.release()
        assert held.exists()

        l2 = SizedCache(tmp_path / "l2")
        l1 = SizedCache(tmp_path / "l1")
        lease = await CacheEntry(l2.add("held", 100)).lease()
        assert lease is not None
        l1.add("free", 300)
        ctrl = self._controller(tmp_path, floor_mb=1000)
        ctrl.add_cache(l2)
        ctrl.add_cache(l1)
        try:
            with patch("exec_sandbox.admission.shutil.disk_usage", _SizedDisk(1000 + 400, [l2, l1]).usage):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-2", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=1, use_tcg=False, held=frozenset()
                    )
                )
        finally:
            lease.release()
        assert lease.path.exists()

    async def test_a_pass_that_gives_nothing_back_ends_the_evictions_of_the_request(self, tmp_path: Path) -> None:
        """Evicted entries whose blocks stay in use: one pass removes at most what the
        request lacks (1000 MB), then no pass and no wake-up evicts more for it (else the next
        passes empty both caches: 2000 MB, 0 MB back)."""
        disk = _Disk(1000)
        disk.written_mb = 5000  # VMs of this controller wrote 5000 MB: their end makes room
        caches = [_Cache(tmp_path / name, disk, [100] * 10, gives_back=0) for name in ("l2", "l1")]
        ctrl = self._controller(tmp_path, floor_mb=0)
        for cache in caches:
            ctrl.add_cache(cache)

        with disk.patched(tmp_path):
            # 8000 MB disk: needs 2000 MB, 1000 MB short
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=8000, timeout=30, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting(ctrl, 1)
            assert sum(len(cache.evicted) for cache in caches) == 10  # 1000 MB: what it lacked
            for _ in range(12):  # 12 wake-ups that change the room (self-wakes)
                await _until_waiting(ctrl, 1)
                disk.read.clear()
                await _self_wake(ctrl)  # taken once the waiter waits: a change of room
                async with asyncio.timeout(5):
                    await disk.read.wait()  # the waiter woke and read the free space to make room again
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert sum(len(cache.evicted) for cache in caches) == 10

    async def test_a_request_that_can_never_fit_evicts_nothing(self, tmp_path: Path) -> None:
        """disk_mb has no upper bound: a request larger than the free space, the end of its
        VMs and every cache entry together fails at once, before any eviction."""
        disk = _Disk(1000)
        caches = [_Cache(tmp_path / name, disk, [100] * 20) for name in ("l2", "l1")]
        ctrl = self._controller(tmp_path, floor_mb=0)
        for cache in caches:
            ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=10**9, timeout=1, use_tcg=False, held=frozenset()
            )
        assert [cache.evicted for cache in caches] == [[], []]

    async def test_wake_up_evicts_entries_saved_meanwhile(self, tmp_path: Path) -> None:
        """A waiter whose first eviction found nothing evicts again on a later wake-up (the
        self-wake timer): entries saved while it waits give their space back."""
        disk = _Disk(1300)
        ctrl = self._controller(tmp_path, floor_mb=1000)
        cache = _Cache(tmp_path / "l1", disk, [])
        ctrl.add_cache(cache)
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            held = await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=0.2, use_tcg=False, held=frozenset()
            )
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-2", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting(ctrl, 1)
            cache.entries.append(300)  # an L1 save while vm-2 waits
            await _self_wake(ctrl)
            async with asyncio.timeout(1):
                await ctrl.release(await waiter)
        await ctrl.release(held)
        assert cache.evicted == [300]

    async def test_eviction_leaves_a_cache_on_another_filesystem(self, tmp_path: Path) -> None:
        """Evicting a cache on another filesystem frees nothing on the home disk filesystem."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)
        elsewhere = _Cache(tmp_path / "l2", disk, [500])
        elsewhere.cache_dir = Path("/dev")  # devfs/devtmpfs: never the filesystem of tmp_path
        ctrl.add_cache(elsewhere)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=1, use_tcg=False, held=frozenset()
            )
        assert elsewhere.evicted == []

    async def test_fails_at_once_when_nothing_can_make_room(self, tmp_path: Path) -> None:
        """No VM of its own to wait for, and the cache entries are too small: LOCAL_DISK at
        once, not a CAPACITY timeout, and no entry goes for nothing. The error names the
        gate and its figures."""
        disk = _Disk(1100)
        ctrl = self._controller(tmp_path, floor_mb=1000)
        cache = _Cache(tmp_path / "l1", disk, [50])
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK) as exc_info,
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=1, use_tcg=False, held=frozenset()
            )
        assert cache.evicted == []  # 50 MB could not make the 156 MB it lacks
        error = exc_info.value
        assert error.action is ErrorAction.RETRY
        assert str(error) == (
            "VM vm-1 does not fit on the filesystem of the home disks: neither cache evictions nor the end "
            "of this worker's VMs and sessions can make room. Home disk space: 1100MB free, floor 1000MB, "
            "0MB of disks admitted (0MB reserved), 1024MB requested (256MB reserved)."
        )
        assert error.context == {
            "vm_id": "vm-1",
            "blocking_gate": "home_disk_space",
            "disk_free_mb": 1100,
            "disk_free_floor_mb": 1000,
            "disk_admitted_mb": 0,
            "disk_reserved_mb": 0,
            "disk_requested_mb": 1024,
            "disk_needed_mb": 256,
        }
        disk.free_mb = 1300  # room for one 1024 MB disk: the failed request reserved nothing
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage), _probe_clock() as later:
            later()
            await ctrl.release(
                await ctrl.acquire(
                    "vm-2", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=_REFUSED, use_tcg=False, held=frozenset()
                )
            )

    async def test_entry_leased_by_another_process_does_not_count(self, tmp_path: Path) -> None:
        """An entry a VM of another process runs on gives nothing back at its end in this
        process: a request that only it could make room for fails at once, and it stays."""
        cache = SizedCache(tmp_path / "l2")
        entry = cache.add("other", 500)
        ctrl = self._controller(tmp_path, floor_mb=1000)
        ctrl.add_cache(cache)
        with (
            leased_in_child(entry),
            patch("exec_sandbox.admission.shutil.disk_usage", _SizedDisk(1000 + 500, [cache]).usage),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=1, use_tcg=False, held=frozenset()
            )
        assert entry.exists()

    async def test_entry_under_another_process_eviction_still_counts(self, tmp_path: Path) -> None:
        """Another process holds a free entry's use lock exclusive for a moment (its eviction or
        its own size check): no VM runs on it, so the request waits for the space instead of
        failing with LOCAL_DISK."""
        cache = SizedCache(tmp_path / "l2")
        entry = cache.add("free", 500)
        ctrl = self._controller(tmp_path, floor_mb=1000)
        ctrl.add_cache(cache)
        with (
            evicted_in_child(entry),
            patch("exec_sandbox.admission.shutil.disk_usage", _SizedDisk(1000 + 500, [cache]).usage),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.CAPACITY),
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=0.2, use_tcg=False, held=frozenset()
            )
        assert entry.exists()  # this process could not take it meanwhile

    async def test_entry_the_request_keeps_leased_does_not_count(self, tmp_path: Path) -> None:
        """The request's own lookup lease (`held`) stays while it waits: the entry gives
        nothing back meanwhile, so the request fails at once with LOCAL_DISK, not with
        CAPACITY after its whole timeout."""
        cache = SizedCache(tmp_path / "l2")
        entry = cache.add("mine", 500)
        ctrl = self._controller(tmp_path, floor_mb=1000)
        ctrl.add_cache(cache)
        lease = await CacheEntry(entry).lease()
        assert lease is not None
        try:
            with (
                patch("exec_sandbox.admission.shutil.disk_usage", _SizedDisk(1000 + 500, [cache]).usage),
                pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
            ):
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset({entry})
                )
        finally:
            lease.release()
        assert entry.exists()

    async def test_entry_leased_here_and_by_another_process_does_not_count(self, tmp_path: Path) -> None:
        """A VM of this process and one of another process run on the entry: the end of this
        process's VMs gives nothing back, so LOCAL_DISK at once."""
        cache = SizedCache(tmp_path / "l2")
        entry = cache.add("shared", 500)
        ctrl = self._controller(tmp_path, floor_mb=1000)
        ctrl.add_cache(cache)
        lease = await CacheEntry(entry).lease()  # a VM of this process
        assert lease is not None
        try:
            with (
                leased_in_child(entry),
                patch("exec_sandbox.admission.shutil.disk_usage", _SizedDisk(1000 + 500, [cache]).usage),
                pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
            ):
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
        finally:
            lease.release()

    async def test_evicted_space_that_shows_at_the_filesystem_commit_admits(self, tmp_path: Path) -> None:
        """btrfs frees an evicted entry's blocks at its next commit (up to 30 s later): the
        commit after the pass (_settle_removals) makes them show, and the request is admitted
        at once, with no LOCAL_DISK and no wait for a wake-up."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300], gives_back=0)
        ctrl.add_cache(cache)

        def commit(_path: Path) -> _Settle:  # what a btrfs commit does
            disk.free_mb = 1000 + sum(cache.evicted)
            return _AFTER_COMMIT

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", side_effect=commit) as sync,
        ):
            async with asyncio.timeout(1):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                    )
                )
        sync.assert_called_once_with(tmp_path)
        assert cache.evicted == [300]

    async def test_a_pass_that_gives_nothing_back_fails_at_once(self, tmp_path: Path) -> None:
        """A copy-on-write clone keeps the evicted blocks: after the pass the free space has
        not grown. The request evicts no more and counts on no cache entry: LOCAL_DISK at
        once, not CAPACITY after its whole timeout."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=0)  # an 8000 MB disk needs 1000 MB more
        cache = _Cache(tmp_path / "l2", disk, [100] * 20, gives_back=0)
        ctrl.add_cache(cache)
        loop = asyncio.get_running_loop()
        started = loop.time()
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=8000, timeout=2, use_tcg=False, held=frozenset()
            )
        assert loop.time() - started < 1
        assert sum(cache.evicted) == 1000  # what it lacked, no more

    async def test_space_that_shows_slowly_is_not_evicted_for_again(self, tmp_path: Path) -> None:
        """The evicted space shows 30 MB at a time, after the pass. The request (it waits: the end of its
        controller's VMs makes room) does not evict the whole shortfall again at each wake-up (else 3300 MB go
        for a 512 MB shortfall): one pass, then no more."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 2048 MB disk needs 512 MB more
        cache = _Cache(tmp_path / "l2", disk, [100] * 40, gives_back=0)
        ctrl.add_cache(cache)

        disk.written_mb = 5000  # VMs of this controller wrote 5000 MB: their end makes room
        with disk.patched(tmp_path):
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=2048, timeout=30, use_tcg=False, held=frozenset()
                )
            )
            for _ in range(6):
                await _until_waiting(ctrl, 1)
                disk.free_mb += 30  # the filesystem frees 30 MB more of the evicted blocks
                await _self_wake(ctrl)  # taken once the waiter waits
            await _until_waiting(ctrl, 1)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert sum(cache.evicted) == 600  # whole entries up to the 512 MB it lacked, once

    async def test_space_that_shows_after_the_pass_admits(self, tmp_path: Path) -> None:
        """XFS frees a removed file in the background, and no call of an unprivileged process
        waits for it: the space shows ms after the pass (on EC2: 2 MB after syncfs, all of it
        0.6 s later, and LOCAL_DISK at once). The free space is read again while it grows: the
        request is admitted, fast, with one entry evicted."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300, 300], after=0.03)
        ctrl.add_cache(cache)
        loop = asyncio.get_running_loop()
        started = loop.time()
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_BACKGROUND),
        ):
            await ctrl.release(
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
            )
        assert loop.time() - started < 0.5
        assert cache.evicted == [300]

    async def test_a_write_of_another_process_during_the_reads_is_no_lack_of_gain(self, tmp_path: Path) -> None:
        """XFS shows 150 of the 300 MB a pass evicted, then another process writes 300 MB before
        the reads stop: the last read shows less than at the start, but the pass gave space
        back. The verdict uses the largest gain seen: the next passes run and admit (the last
        read made it a pass of nothing, and LOCAL_DISK). The write comes right after the first
        read that shows the gain (not on a timer: under load the two could swap)."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300] * 6, gives_back=0.5, after=0.02)
        ctrl.add_cache(cache)
        wrote = False

        def usage(path: object) -> SimpleNamespace:
            nonlocal wrote
            read = disk.usage(path)
            if not wrote and disk.free_mb > 1000:  # this read shows the pass's gain
                wrote = True
                disk.free_mb -= 300  # another process writes before the next read
            return read

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_BACKGROUND),
        ):
            await ctrl.release(
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                )
            )
        assert wrote
        # The passes after the first one ran: +150 MB (406 MB lacking), then 2 entries for +300
        # (106 MB lacking), then 1 entry that makes it fit
        assert cache.evicted == [300] * 4

    async def test_space_that_never_shows_fails_once_it_stops_growing(self, tmp_path: Path) -> None:
        """A clone keeps the evicted blocks on XFS too, where the space is read again after the
        pass: LOCAL_DISK once the space has not grown for its quiet time, not after the whole
        limit."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=0)  # an 8000 MB disk needs 1000 MB more
        cache = _Cache(tmp_path / "l2", disk, [100] * 20, gives_back=0)
        ctrl.add_cache(cache)
        loop = asyncio.get_running_loop()
        started = loop.time()
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_BACKGROUND),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=8000, timeout=5, use_tcg=False, held=frozenset()
            )
        assert 0.15 <= loop.time() - started < 1.0  # XFS: no growth for 0.15 s ends the reads, 1 s at most
        assert sum(cache.evicted) == 1000

    async def test_metadata_a_clone_gives_back_is_no_gain(self, tmp_path: Path) -> None:
        """The removal of a reflink clone frees only its metadata (0.6 % of its bytes, measured on
        XFS): the pass counts as no gain. The request, which waits for its own VMs to end, evicts
        no more than it lacked; with no VM of its own it fails at once."""
        disk = _Disk(1000)
        cache = _Cache(tmp_path / "l2", disk, [100] * 20, gives_back=0.006)
        ctrl = self._controller(tmp_path, floor_mb=0)  # an 8000 MB disk needs 1000 MB more
        ctrl.add_cache(cache)
        disk.written_mb = 5000  # VMs of this controller wrote 5000 MB: their end makes room
        with disk.patched(tmp_path):
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=8000, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting_or_done(ctrl, waiter)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert sum(cache.evicted) == 1000  # one pass: what it lacked

        disk = _Disk(1000)
        cache = _Cache(tmp_path / "l1", disk, [100] * 20, gives_back=0.006)
        ctrl = self._controller(tmp_path, floor_mb=0)
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=8000, timeout=2, use_tcg=False, held=frozenset()
            )
        assert sum(cache.evicted) == 1000

    async def test_compressed_entries_give_back_part_of_their_bytes(self, tmp_path: Path) -> None:
        """btrfs compression: an evicted entry gives back as little as 1/32 of the bytes its
        st_blocks count (zeros; an L1 RAM file gave 8 %). Each pass counts, the next follows at
        once, and the request is admitted."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [100] * 100, gives_back=1 / 32)
        ctrl.add_cache(cache)
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            async with asyncio.timeout(1):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                    )
                )
        assert 256 * 32 <= sum(cache.evicted) < 256 * 32 + 300  # what it lacked, at 1/32

    async def test_a_pass_asks_each_cache_for_one_order(self, tmp_path: Path) -> None:
        """The evictions of one pass take the next entries of one order per cache: the cache is
        listed once per pass, not once per eviction."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [100] * 5)
        ctrl.add_cache(cache)
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            await ctrl.release(
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
            )
        assert (cache.evicted, cache.orders) == ([100, 100, 100], 1)

    async def test_a_pass_that_leaves_the_request_short_is_followed_at_once(self, tmp_path: Path) -> None:
        """A pass gave space back, but less than the request lacks (a third of each entry here): the next pass
        follows at once, so the request does not wait for the 10 s self-wake (else admitted at 10.00 s, with
        the space shown at 0.5 s)."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300] * 5, gives_back=1 / 3)
        ctrl.add_cache(cache)
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            await ctrl.release(
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
            )
        assert cache.evicted == [300, 300, 300]

    async def test_writes_of_its_own_vms_do_not_hide_the_gain(self, tmp_path: Path) -> None:
        """A guest of this controller writes its home disk while a pass evicts: those blocks take
        the free space the eviction gave back. The gain counts net of them, so the pass is no
        "no gain" (the request would then wait for its timeout): the next pass follows at once
        and the request is admitted."""
        disk = _Disk(1500)
        ctrl = self._controller(tmp_path, floor_mb=1000)

        class _WrittenMeanwhile(_Cache):
            async def evict_oldest(self) -> int | None:
                freed = await super().evict_oldest()
                if len(self.evicted) == 1:  # the guest of vm-1 writes 400 MB during the first pass
                    disk.free_mb -= 400
                    disk.written_mb += 400
                return freed

        cache = _WrittenMeanwhile(tmp_path / "l2", disk, [300, 300])
        ctrl.add_cache(cache)
        with disk.patched(tmp_path):
            first = await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=2000, timeout=0.2, use_tcg=False, held=frozenset()
            )
            # vm-2 needs 256 MB more; after the first pass the free space is 100 MB lower
            second = await ctrl.acquire(
                "vm-2", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
            )
        await ctrl.release(second)
        await ctrl.release(first)
        assert cache.evicted == [300, 300]

    async def test_overlay_space_shows_after_one_sync(self, tmp_path: Path) -> None:
        """An overlay frees a removed file when its upper filesystem commits (a btrfs below: at +30 s, long
        after a LOCAL_DISK at 0.2 s). Its statfs type does not say which filesystem that is: when the space
        read again shows nothing, one syncfs makes it show, and the request is admitted with one entry
        evicted."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300, 300], gives_back=0)
        ctrl.add_cache(cache)

        def sync(_path: Path) -> None:  # what syncfs does on an overlay over btrfs
            disk.free_mb = 1000 + sum(cache.evicted)

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_OVERLAY),
            patch("exec_sandbox.admission._syncfs", side_effect=sync) as syncfs,
        ):
            await ctrl.release(
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
            )
        syncfs.assert_called_once_with(tmp_path)
        assert cache.evicted == [300]

    async def test_overlay_gain_hidden_by_dirty_data_is_measured_after_a_first_sync(self, tmp_path: Path) -> None:
        """An overlay over btrfs shows the removals and the blocks of other files' dirty data in the same
        commit: after the one syncfs, 300 MB of dirty data of another process make a 200 MB gain read as -100
        MB, a verdict that would stop the request (LOCAL_DISK). That pass gives no verdict: the next one syncs
        before it evicts, its gain shows alone, and the request is admitted."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=894)  # a 1024 MB disk needs 150 MB more
        cache = _Cache(tmp_path / "l2", disk, [200] * 4, gives_back=0)
        ctrl.add_cache(cache)
        shown: list[float] = []
        dirty_mb = [300.0]  # written by another process, not synced yet

        def sync(_path: Path) -> None:  # the upper btrfs commits the removals and the dirty data together
            disk.free_mb += sum(cache.evicted) - sum(shown) - dirty_mb[0]
            shown[:] = cache.evicted
            dirty_mb[0] = 0.0

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_OVERLAY),
            patch("exec_sandbox.admission._syncfs", side_effect=sync),
        ):
            await ctrl.release(
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
            )
        assert cache.evicted == [200, 200, 200]  # 150 MB lacking, then 250 MB once the dirty data took 300

    async def test_overlay_pass_that_gives_nothing_back_after_a_first_sync_stops(self, tmp_path: Path) -> None:
        """A clone on an overlay: no sync shows its blocks. The pass with no verdict, then the
        pass measured alone after a first sync, whose space counts as still coming up to one ZFS
        transaction sync (an overlay over ZFS shows it only then): LOCAL_DISK then, with
        two passes evicted, not the whole cache. Here zfs_txg_timeout is 0: 1 s."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=894)  # a 1024 MB disk needs 150 MB more
        cache = _Cache(tmp_path / "l2", disk, [200] * 4, gives_back=0)
        ctrl.add_cache(cache)
        txg = tmp_path / "zfs_txg_timeout"
        txg.write_text("0\n")
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_OVERLAY),
            patch("exec_sandbox.admission._ZFS_TXG_TIMEOUT", txg),
            patch("exec_sandbox.admission._syncfs") as syncfs,
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=3, use_tcg=False, held=frozenset()
            )
        assert cache.evicted == [200, 200]
        assert syncfs.call_count == 3  # after pass 1; before and after pass 2

    async def test_hidden_overlay_without_zfs_does_not_wait_for_a_transaction_sync(self, tmp_path: Path) -> None:
        """A container on an ext4 or XFS host (no ZFS module): the overlay's upper folder is hidden, and
        evicted clones give nothing. A wait for a ZFS sync there gives LOCAL_DISK at 11.57 s instead of 0.66
        s: the wait is only where the ZFS module is loaded."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=894)  # a 1024 MB disk needs 150 MB more
        cache = _Cache(tmp_path / "l2", disk, [200] * 4, gives_back=0)
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_OVERLAY),
            patch("exec_sandbox.admission._ZFS_TXG_TIMEOUT", tmp_path / "no-zfs-module"),
            patch("exec_sandbox.admission._syncfs"),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
        ):
            async with asyncio.timeout(1):
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=3, use_tcg=False, held=frozenset()
                )
        assert cache.evicted == [200, 200]

    async def test_an_unreadable_filesystem_type_reads_as_unknown(self, tmp_path: Path) -> None:
        """An error out of _settle_removals (a mountinfo line that is not UTF-8, an upper folder under a 0700
        folder) does not come out of acquire(): it reads as an unknown type (_OTHER), and the request is
        admitted."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300] * 2, after=0.05)  # it shows after the removal: settle runs
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", side_effect=PermissionError(13, "Permission denied")),
        ):
            await ctrl.release(
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
            )
        assert cache.evicted == [300]

    async def test_overlay_space_that_shows_at_a_transaction_sync_admits(self, tmp_path: Path) -> None:
        """An overlay over ZFS (OpenZFS 2.2+ takes overlay upper folders) whose upper folder this process does
        not see: its syncfs commits the intent log only, so two passes see nothing (on EC2: LOCAL_DISK at 0.39
        s, and 600 MB shown 12 s later). The second pass's space counts as still coming up to one ZFS
        transaction sync: admitted once it shows. Here the space of both removals shows 50 ms after the
        second pass took that wait (_zfs_settle), so no pass sees it before, whatever the runner's speed (a
        wall-clock delay after each removal let a slow runner's second pass see the space: no wait taken);
        zfs_txg_timeout 0: 1 s."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=894)  # a 1024 MB disk needs 150 MB more
        cache = _Cache(tmp_path / "l2", disk, [200] * 4, holds=True)
        ctrl.add_cache(cache)
        txg = tmp_path / "zfs_txg_timeout"
        txg.write_text("0\n")
        loop = asyncio.get_running_loop()

        def txg_wait() -> _Settle:
            loop.call_later(0.05, cache.show)  # the next transaction sync
            return _zfs_settle()

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_OVERLAY),
            patch("exec_sandbox.admission._ZFS_TXG_TIMEOUT", txg),
            patch("exec_sandbox.admission._zfs_settle", side_effect=txg_wait),
            patch("exec_sandbox.admission._syncfs"),
        ):
            async with asyncio.timeout(1.5):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=3, use_tcg=False, held=frozenset()
                    )
                )
        assert cache.evicted == [200, 200]

    async def test_a_sync_not_done_in_time_leaves_the_space_coming(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A syncfs writes the dirty data of the filesystem first (8-16 s with 1-2 GB on gp3): it
        is not waited for past _SYNCFS_WAIT_SECONDS. The removed space counts as still coming,
        so the request waits for a wake-up instead of failing, and is admitted once the sync
        showed it."""
        monkeypatch.setattr("exec_sandbox.admission._SYNCFS_WAIT_SECONDS", 0.05)
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300, 300], gives_back=0)
        ctrl.add_cache(cache)
        go_on, synced = threading.Event(), threading.Event()

        def slow_sync(_path: Path) -> None:
            go_on.wait(5)
            disk.free_mb = 1000 + sum(cache.evicted)
            synced.set()

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_OVERLAY),
            patch("exec_sandbox.admission._syncfs", side_effect=slow_sync),
        ):
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting_or_done(ctrl, waiter)
            assert not waiter.done()  # it waits for a wake-up: no LOCAL_DISK
            go_on.set()
            assert await asyncio.to_thread(synced.wait, 5)
            await _self_wake(ctrl)
            async with asyncio.timeout(1):
                await ctrl.release(await waiter)
        assert cache.evicted == [300]

    async def test_a_slow_sync_with_dirty_data_admits_once_it_ends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On gp3 the syncfs of an overlay over btrfs writes 1 GB of dirty data in 8 s, past
        _SYNCFS_WAIT_SECONDS, and its commit shows the removals less those new blocks.
        The removed space counts as coming only until that sync ends, not for _UNSEEN_SECONDS:
        then the next pass syncs first, measures its own gain, and the request is admitted."""
        monkeypatch.setattr("exec_sandbox.admission._SYNCFS_WAIT_SECONDS", 0.05)
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=894)  # a 1024 MB disk needs 150 MB more
        cache = _Cache(tmp_path / "l2", disk, [200] * 4, gives_back=0)
        ctrl.add_cache(cache)
        shown: list[float] = []
        dirty_mb = [300.0]
        go_on = threading.Event()

        def sync(_path: Path) -> None:
            if dirty_mb[0]:
                go_on.wait(5)  # the first sync writes the dirty data: slow
            disk.free_mb += sum(cache.evicted) - sum(shown) - dirty_mb[0]
            shown[:] = cache.evicted
            dirty_mb[0] = 0.0

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_OVERLAY),
            patch("exec_sandbox.admission._syncfs", side_effect=sync),
        ):
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting_or_done(ctrl, waiter)
            assert not waiter.done()
            go_on.set()
            async with asyncio.timeout(1):  # well before the 10 s the space still coming counts
                await ctrl.release(await waiter)
        assert cache.evicted == [200, 200, 200]

    async def test_a_request_that_fits_is_woken_before_all_the_space_has_shown(self, tmp_path: Path) -> None:
        """ZFS: the transaction sync shows most of the evicted space, the rest still trickles in:
        the waiting request fits, and the re-check timer wakes it then, not once the pending
        pass ends (4 s here; ADMITTED at 0.506 s on EC2 with a VM writing)."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300, 300], gives_back=0.9, after=0.2)
        ctrl.add_cache(cache)
        loop = asyncio.get_running_loop()
        for step in range(1, 100):  # 0.3 MB every 20 ms: it never stops growing for long
            loop.call_later(0.2 + step / 50, lambda: setattr(disk, "free_mb", disk.free_mb + 0.3))
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_zfs(4.0)),
        ):
            async with asyncio.timeout(1):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                    )
                )
        assert cache.evicted == [300]

    async def test_space_shown_as_far_as_it_will_lets_the_next_pass_start(self, tmp_path: Path) -> None:
        """ZFS: the transaction sync shows 90 % of the evicted space, and the rest never comes
        (the data a VM wrote meanwhile hides it). Once the space has grown, then stopped, the
        pass has shown as far as it will: the next request's pass does not wait for the end of
        its time (4 s here)."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300, 300], gives_back=0.9, after=0.2)
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_zfs(4.0)),
        ):
            async with asyncio.timeout(1.5):
                first = await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
                second = await ctrl.acquire(
                    "vm-2", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
        await ctrl.release(second)
        await ctrl.release(first)
        assert cache.evicted == [300, 300]

    async def test_btrfs_writes_of_its_own_vms_are_no_gain(self, tmp_path: Path) -> None:
        """btrfs counts a write in st_blocks at once, in its free space only at writeback. For a pass of
        clones (no space back), a gain net of the writes of this controller's VMs grows at their write speed:
        the request would evict the whole cache and get CAPACITY at its timeout. On btrfs the free space alone
        judges the gain: one pass, then the request waits for its VMs (or fails with LOCAL_DISK without them)."""
        disk = _Disk(1000)
        disk.written_mb = 5000  # VMs of this controller wrote 5000 MB: their end makes room
        ctrl = self._controller(tmp_path, floor_mb=0)  # an 8000 MB disk needs 1000 MB more

        class _ClonesWhileAVmWrites(_Cache):
            async def evict_oldest(self) -> int | None:
                freed = await super().evict_oldest()
                disk.written_mb += 5  # delalloc: st_blocks grows, the free space does not
                return freed

        cache = _ClonesWhileAVmWrites(tmp_path / "l2", disk, [100] * 20, gives_back=0)
        ctrl.add_cache(cache)
        with (
            disk.patched(tmp_path),
            patch("exec_sandbox.admission._settle_removals", return_value=_AFTER_COMMIT),
        ):
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=8000, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting_or_done(ctrl, waiter)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert sum(cache.evicted) == 1000  # one pass

    async def test_overlay_space_still_growing_is_synced_too(self, tmp_path: Path) -> None:
        """On an overlay the free space can still grow a little at the end of the reads (another
        file freed meanwhile) while the removal shows only at a sync of the filesystem below: the
        pass syncs then too, instead of counting the space as still coming for 10 s."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300, 300], gives_back=0)
        ctrl.add_cache(cache)
        loop = asyncio.get_running_loop()
        # 0.1 MB every 5 ms for 0.2 s: still growing at the pass's 0.1 s limit, and under 1/64
        # of the pass (300 / 64 = 4.7 MB) even once every step ran (4 MB): a read that a loaded
        # runner makes late gives the same verdict (with 0.2 MB steps, a read 30 ms late saw
        # 5.2 MB, and the pass never synced)
        for step in range(1, 41):
            loop.call_later(step / 200, lambda: setattr(disk, "free_mb", disk.free_mb + 0.1))

        def sync(_path: Path) -> None:
            disk.free_mb += sum(cache.evicted)

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_OVERLAY),
            patch("exec_sandbox.admission._syncfs", side_effect=sync) as syncfs,
        ):
            async with asyncio.timeout(1):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                    )
                )
        syncfs.assert_called_once_with(tmp_path)

    async def test_requests_that_gate_4_blocks_together_are_admitted_after_one_sync(self, tmp_path: Path) -> None:
        """ZFS shows evictions at its next transaction sync. With a pass for each request, each holding the
        lock of the passes until its own gain shows, one request is admitted per sync (10 requests with a 3 s
        timeout: 5 CAPACITY). One pass evicts for every request that gate 4 blocks, and does not hold the lock
        until the sync: all are admitted after it."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # each 400 MB disk needs 100 MB
        shown: list[float] = []
        cache = _Cache(tmp_path / "l2", disk, [100] * 40, gives_back=0)
        ctrl.add_cache(cache)

        async def transaction_syncs() -> None:  # every 0.3 s, the evictions since the last sync show
            while True:
                await asyncio.sleep(0.3)
                disk.free_mb += sum(cache.evicted) - sum(shown)
                shown[:] = cache.evicted

        syncs = asyncio.create_task(transaction_syncs())
        try:
            with (
                patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
                patch("exec_sandbox.admission._settle_removals", return_value=_zfs(0.7)),
            ):
                reservations = await asyncio.gather(
                    *(
                        ctrl.acquire(
                            f"vm-{i}",
                            memory_mb=1,
                            cpu_cores=_CPU,
                            disk_mb=400,
                            timeout=1,
                            use_tcg=False,
                            held=frozenset(),
                        )
                        for i in range(5)
                    )
                )
        finally:
            syncs.cancel()
        for reservation in reservations:
            await ctrl.release(reservation)
        assert sum(cache.evicted) == 500  # one pass for the five

    async def test_space_that_never_shows_at_a_transaction_sync_stops_the_request(self, tmp_path: Path) -> None:
        """ZFS: an evicted clone or open file shows nothing at the next transaction sync. Once
        the space counted as still coming has had its time, the pass gets its verdict: no gain,
        LOCAL_DISK, with one pass evicted."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [100] * 20, gives_back=0)
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_zfs(0.2)),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
            )
        assert sum(cache.evicted) == 300  # the 256 MB it lacked, in whole entries: one pass

    async def test_space_still_coming_of_another_request_is_not_a_pass_gain(self, tmp_path: Path) -> None:
        """A's evicted entry still frees on XFS (10 MB every 30 ms) while B evicts clones that give nothing
        back: B's passes could read A's space as their own gain (CAPACITY at B's timeout, not LOCAL_DISK). No
        pass starts while the space of an earlier one still shows: B's pass then measures its own, gets no
        gain, and fails at once."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)

        class _FreedSlowly(_Cache):
            async def evict_oldest(self) -> int | None:
                freed = await super().evict_oldest()
                if freed is not None:
                    loop = asyncio.get_running_loop()
                    for step in range(1, 31):
                        loop.call_later(0.03 * step, lambda: setattr(disk, "free_mb", disk.free_mb + 10))
                return freed

        slow = _FreedSlowly(tmp_path / "l2", disk, [300], gives_back=0)
        clones = _Cache(tmp_path / "l1", disk, [100] * 10, gives_back=0)
        ctrl.add_cache(slow)
        ctrl.add_cache(clones)
        xfs = _Settle(quiet=0.05, limit=0.1, sync_once=False, later=0.0, counts_writes=True)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=xfs),
        ):
            a = asyncio.create_task(
                ctrl.acquire(
                    "vm-a", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=3, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting_or_done(ctrl, a)  # A's pass ended, its space still shows
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                await ctrl.acquire(
                    "vm-b", memory_mb=1, cpu_cores=_CPU, disk_mb=3000, timeout=2, use_tcg=False, held=frozenset()
                )
            await ctrl.release(await a)
        # One pass: the 706 MB B lacked once A's space showed (1300 MB free; floor 1000, A's 256 MB
        # reserve and B's 750 MB), in whole entries
        assert clones.evicted == [100] * 8

    async def test_space_that_shows_in_one_step_at_a_transaction_sync_admits(self, tmp_path: Path) -> None:
        """ZFS shows a removal in one step at its next transaction sync (4-5 s on EC2), nothing
        before, and syncfs does not start one (LOCAL_DISK at 0.18 s). The pass reads it once
        and counts it as still coming until the sync, without the lock of the passes: the
        request is admitted once it shows, within the 50 ms reads of the re-check timer. Here
        one sync is 0.2 s."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300, 300], after=0.2)
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_zfs(0.4)),
        ):
            async with asyncio.timeout(1):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                    )
                )
        assert cache.evicted == [300]

    async def test_compressed_entries_that_cannot_make_room_fail_after_one_pass(self, tmp_path: Path) -> None:
        """btrfs compression: the entries give back 1/32 of their st_blocks, so all of them give less than the
        request lacks. After its first pass the rest counts at what that pass gave back: LOCAL_DISK then, with
        no more than the shortfall evicted (else all 10 of 10 go for +19 MB)."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1044)  # a 1024 MB disk needs 300 MB more
        cache = _Cache(tmp_path / "l2", disk, [100] * 10, gives_back=1 / 32)
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
        ):
            await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
            )
        assert sum(cache.evicted) == 300

    async def test_a_request_checks_its_room_after_the_pass_of_another(self, tmp_path: Path) -> None:
        """Two requests short of space at once. A's pass removed an entry that the free space shows only at
        A's commit; a B that checks its room meanwhile counts neither and fails with LOCAL_DISK (4 of 6 runs
        on btrfs). B checks under the lock of the passes, after A's, and waits for A's VM instead."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300, 100], gives_back=0)
        ctrl.add_cache(cache)
        committing, go_on = threading.Event(), threading.Event()

        def commit(_path: Path) -> _Settle:
            committing.set()
            go_on.wait(5)
            disk.free_mb = 1000 + sum(cache.evicted)
            return _AT_REMOVAL

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", side_effect=commit),
        ):
            a = asyncio.create_task(
                ctrl.acquire(
                    "vm-a", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await asyncio.to_thread(committing.wait, 5)
            b = asyncio.create_task(
                ctrl.acquire(
                    "vm-b", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await _until_queued_for_the_passes(ctrl, b)  # B fails at once, or waits for the lock A holds
            assert not b.done()
            go_on.set()
            first = await a
            await _until_waiting(ctrl, 1)  # B waits for A's VM to end
            await ctrl.release(first)
            async with asyncio.timeout(1):
                await ctrl.release(await b)

    async def test_space_still_growing_at_the_limit_counts_until_it_shows(self, tmp_path: Path) -> None:
        """XFS frees fragmented entries for seconds: the free space still grows when the pass
        stops reading it (LOCAL_DISK at 1.12 s with +268 of +889 MB shown). The rest
        counts as still coming: the request waits instead of failing, also at a wake-up before
        the space has shown, and checks again soon (not at the 10 s self-wake: ADMITTED at
        10.01 s on EC2): it is admitted once the space has shown."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more

        checks: list[float] = []  # the free space at each check of the room (each listing of the cache)

        class _FreedSlowly(_Cache):
            async def evict_oldest(self) -> int | None:
                freed = await super().evict_oldest()
                if freed is not None:  # 10 MB every 10 ms, for 0.3 s
                    loop = asyncio.get_running_loop()
                    for step in range(1, 31):
                        loop.call_later(step / 100, lambda: setattr(disk, "free_mb", disk.free_mb + 10))
                return freed

            async def reclaimable_bytes(self, *, held: frozenset[Path]) -> int:
                checks.append(disk.free_mb)
                return await super().reclaimable_bytes(held=held)

        cache = _FreedSlowly(tmp_path / "l2", disk, [300], gives_back=0)
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch(
                "exec_sandbox.admission._settle_removals",
                return_value=_Settle(quiet=0.05, limit=0.1, sync_once=False, later=0.0, counts_writes=True),
            ),
        ):
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting_or_done(ctrl, waiter)
            assert not waiter.done()  # it waits: no LOCAL_DISK
            await _self_wake(ctrl)  # a wake-up while the space still comes: it still counts
            await _until(lambda: len(checks) > 1 or waiter.done(), what="the waiter checked its room again")
            assert checks[-1] < 1300  # checked before all of it showed
            await _until_waiting_or_done(ctrl, waiter)
            assert not waiter.done()
            async with asyncio.timeout(1):  # the re-check timer, not the 10 s self-wake
                await ctrl.release(await waiter)
        assert cache.evicted == [300]

    async def test_a_dip_of_the_free_space_after_the_commit_is_read_past(self, tmp_path: Path) -> None:
        """btrfs: 1-2 ms after its commit the free space can dip by a data chunk, then come back. A single
        read then sees the pass give nothing back and stops it: the space is read again until it has not
        changed for a moment."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300], gives_back=0)
        ctrl.add_cache(cache)
        loop = asyncio.get_running_loop()

        def commit(_path: Path) -> _Settle:
            disk.free_mb = 1300 - 1024  # the gain, less a 1 GiB data chunk for 2 ms
            loop.call_soon_threadsafe(loop.call_later, 0.002, lambda: setattr(disk, "free_mb", 1300))
            return _AFTER_COMMIT

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", side_effect=commit),
        ):
            await ctrl.release(
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
            )
        assert cache.evicted == [300]

    async def test_space_that_stops_growing_at_the_end_of_the_reads_lets_the_next_pass_start(
        self, tmp_path: Path
    ) -> None:
        """XFS frees a shared entry in steps up to 0.06 s, within the last quiet time (0.05 s) before the 0.1
        s limit of the reads, and only part of it: the pass ends "growing", and no growth comes after. A
        pending space that counts no growth then never settles, and no pass starts for 10 s (admitted at 11.74
        s). It grew to the end of the reads: it settles once it has not grown for a moment."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=994)  # a 1024 MB disk needs 250 MB more: one entry a pass

        class _SharedFreedSlowly(_Cache):
            async def evict_oldest(self) -> int | None:
                freed = await super().evict_oldest()
                if freed is not None:  # 200 of its 300 MB come back, in steps until 0.06 s
                    loop = asyncio.get_running_loop()
                    for step in range(1, 13):
                        loop.call_later(0.005 * step, lambda: setattr(disk, "free_mb", disk.free_mb + 200 / 12))
                return freed

        cache = _SharedFreedSlowly(tmp_path / "l2", disk, [300] * 4, gives_back=0)
        ctrl.add_cache(cache)
        xfs = _Settle(quiet=0.05, limit=0.1, sync_once=False, later=0.0, counts_writes=True)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=xfs),
        ):
            async with asyncio.timeout(1.5):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=3, use_tcg=False, held=frozenset()
                    )
                )
        assert cache.evicted == [300, 300]

    async def test_a_pass_for_several_waiters_wakes_the_one_that_fits(self, tmp_path: Path) -> None:
        """A pass made room for A and the waiters W1 (too large still) and W2. An admission that wakes one
        waiter wakes W1, which does not fit, and W2 sleeps to the 10 s self-wake. With waiters of mixed sizes,
        an admission wakes them all: W2 is admitted at once."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)
        disk.written_mb = 10_000  # VMs of this controller wrote much: the waiters wait, no LOCAL_DISK
        cache = _Cache(tmp_path / "l2", disk, [])
        ctrl.add_cache(cache)
        with disk.patched(tmp_path):
            w1 = asyncio.create_task(
                ctrl.acquire(
                    "w1", memory_mb=1, cpu_cores=_CPU, disk_mb=8000, timeout=3, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting_or_done(ctrl, w1)
            w2 = asyncio.create_task(
                ctrl.acquire("w2", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset())
            )
            await _until_waiting(ctrl, 2)
            cache.entries.append(300)  # a VM ends and saves an entry: A's pass takes it
            async with asyncio.timeout(1):
                a = await ctrl.acquire(
                    "a", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                )
                second = await w2  # 300 MB: A's 100 MB reserve and W2's 100 MB
            assert not w1.done()
            w1.cancel()
            with pytest.raises(asyncio.CancelledError):
                await w1
        await ctrl.release(second)
        await ctrl.release(a)

    async def test_a_no_gain_verdict_stops_every_request_that_gate_4_blocks(self, tmp_path: Path) -> None:
        """ZFS with a snapshot: no eviction ever shows. With a verdict for the pass's request only, and no
        pass while one is pending, each request waits its own 11 s in turn (LOCAL_DISK at 11/22/33 s, then
        CAPACITY). Once two passes in a row gave nothing, the cause is the filesystem's (one pass alone can
        meet entries of its own): every request gate 4 blocks then stops, after two passes for all of them."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # each 400 MB disk needs 100 MB
        cache = _Cache(tmp_path / "l2", disk, [100] * 30, gives_back=0)
        ctrl.add_cache(cache)
        loop = asyncio.get_running_loop()
        started = loop.time()
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_zfs(0.3)),
        ):
            results = await asyncio.gather(
                *(
                    ctrl.acquire(
                        f"vm-{i}", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                    )
                    for i in range(3)
                ),
                return_exceptions=True,
            )
        assert all(isinstance(r, TransientError) and r.code is ErrorCode.LOCAL_DISK for r in results)
        assert loop.time() - started < 1.5
        # Two passes: the first for the three requests, the second (after a pass of nothing) for
        # its own request only
        assert cache.passes == [[100, 100, 100], [100]]

    async def test_one_pass_of_nothing_does_not_stop_the_other_requests(self, tmp_path: Path) -> None:
        """The oldest entry is a clone (its removal gives nothing), the next ones give all. A's pass removes
        the clone; a verdict that stops B too gives A and B LOCAL_DISK while C, 50 ms later, is admitted from
        the same cache. One pass of nothing stops only its own request from evicting: B's pass takes the next
        entries for both, and all three are admitted."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # each 400 MB disk needs 100 MB

        class _ClonedFirst(_Cache):
            async def evict_oldest(self) -> int | None:
                first = not self.evicted
                freed = await super().evict_oldest()
                if first and freed is not None:
                    disk.free_mb -= 200  # the clone kept its blocks
                return freed

        cache = _ClonedFirst(tmp_path / "l2", disk, [200] * 5)
        ctrl.add_cache(cache)
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            a = asyncio.create_task(
                ctrl.acquire(
                    "vm-a", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=2, use_tcg=False, held=frozenset()
                )
            )
            b = asyncio.create_task(
                ctrl.acquire(
                    "vm-b", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=2, use_tcg=False, held=frozenset()
                )
            )
            async with asyncio.timeout(1):
                reservations = await asyncio.gather(a, b)
                reservations.append(
                    await ctrl.acquire(
                        "vm-c", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=2, use_tcg=False, held=frozenset()
                    )
                )
        for reservation in reservations:
            await ctrl.release(reservation)

    async def test_requests_after_two_passes_of_nothing_evict_nothing(self, tmp_path: Path) -> None:
        """ZFS with a snapshot: after two passes in a row gave nothing, a request that comes later and starts
        from zero runs one more pass (100 MB more evicted, 11 s on real ZFS). It inherits the stop: LOCAL_DISK
        at once, with nothing evicted."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # each 400 MB disk needs 100 MB
        cache = _Cache(tmp_path / "l2", disk, [100] * 40, gives_back=0)
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_zfs(0.2)),
        ):
            await asyncio.gather(
                *(
                    ctrl.acquire(
                        f"vm-{i}", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                    )
                    for i in range(3)
                ),
                return_exceptions=True,
            )
            evicted = sum(cache.evicted)
            loop = asyncio.get_running_loop()
            started = loop.time()
            for i in range(3):
                with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                    await ctrl.acquire(
                        f"late-{i}",
                        memory_mb=1,
                        cpu_cores=_CPU,
                        disk_mb=400,
                        timeout=3,
                        use_tcg=False,
                        held=frozenset(),
                    )
            assert loop.time() - started < 0.5
        assert sum(cache.evicted) == evicted

    async def test_the_self_wake_ends_a_shared_stop(self, tmp_path: Path) -> None:
        """No VM of this controller runs (a pool refill): two passes of nothing share a stop. If only a
        release ended it, every later request would fail with LOCAL_DISK, with 4000 MB of good entries left.
        The self-wake ends it too: the next request evicts."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # each 400 MB disk needs 100 MB
        cache = _GivesBack(tmp_path / "l2", disk, [(200, 0.0), (200, 0.0)] + [(200, 1.0)] * 10)
        ctrl.add_cache(cache)
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            for name in ("a", "b", "stopped"):  # two passes of nothing, then a request under the shared stop
                with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                    await ctrl.acquire(
                        name, memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=2, use_tcg=False, held=frozenset()
                    )
            assert cache.evicted == [200, 200]  # the request under the stop evicted nothing
            await _self_wake(ctrl)
            async with asyncio.timeout(1):
                reservation = await ctrl.acquire(
                    "c", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=2, use_tcg=False, held=frozenset()
                )
            await ctrl.release(reservation)
        assert cache.evicted == [200, 200, 200]

    @pytest.mark.parametrize("shows", ["at_removal", "at_commit"])
    async def test_passes_that_make_the_requests_fit_end_a_run_of_passes_of_nothing(
        self, tmp_path: Path, shows: str
    ) -> None:
        """Entries: a clone, a good one, a clone, good ones. A's pass gives nothing, B's pass makes B fit, C's
        pass gives nothing: not two in a row. A count that stays at 1 after B (its pass returns before any
        verdict) makes C's pass share a stop, and D gets LOCAL_DISK at once. A pass that makes the requests
        fit ends the run, whether the space shows at the removal (ext4) or at the end of the pass (a btrfs
        commit): D is admitted."""
        disk = _Disk(900)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # each 400 MB disk needs 100 MB: 200 MB short
        cache = _GivesBack(tmp_path / "l2", disk, [(200, 0.0), (200, 1.0), (200, 0.0)] + [(200, 1.0)] * 5)
        ctrl.add_cache(cache)
        hidden: list[float] = []  # at_commit: what the removals give back, shown at the commit
        if shows == "at_commit":
            evict = cache.evict_oldest

            async def evict_until_the_commit() -> int | None:
                before = disk.free_mb
                freed = await evict()
                hidden.append(disk.free_mb - before)
                disk.free_mb = before
                return freed

            cache.evict_oldest = evict_until_the_commit  # type: ignore[method-assign]

        def commit(_path: Path) -> _Settle:
            disk.free_mb += sum(hidden)
            hidden.clear()
            return _AFTER_COMMIT

        outcomes: list[str] = []
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", side_effect=commit),
            _probe_clock() as later,
        ):
            for name in ("a", "b", "c", "d"):
                try:
                    reservation = await ctrl.acquire(
                        name, memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=1, use_tcg=False, held=frozenset()
                    )
                except TransientError as e:
                    outcomes.append(e.code.name)
                else:
                    outcomes.append("ADMITTED")
                    await ctrl.release(reservation)
                    disk.free_mb = 900  # its VM used the space meanwhile
                    later()
        assert outcomes == ["LOCAL_DISK", "ADMITTED", "LOCAL_DISK", "ADMITTED"]

    async def test_the_end_of_a_sync_makes_the_request_make_room_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request whose first pass showed nothing after its sync (dirty data of another process took the
        space of the removals) syncs before each next pass, and waits for a slow syncfs. If the end of that
        sync wakes the request with no change of room, the request waits for the self-wake, whose sync is slow
        again: CAPACITY in a loop (8 s). The end of the sync is a change of room: the pass follows at once."""
        monkeypatch.setattr("exec_sandbox.admission._SYNCFS_WAIT_SECONDS", 0.05)
        disk = _Disk(900)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 400 MB disk needs 100 MB: 200 MB short
        cache = _Cache(tmp_path / "l2", disk, [100] * 10, gives_back=0)
        ctrl.add_cache(cache)
        shown: list[float] = []
        syncs: list[float] = []

        def sync(_path: Path) -> None:  # the overlay frees removed files at a sync
            syncs.append(1)
            if len(syncs) == 1:  # dirty data of another process takes the space of the removals
                shown[:] = cache.evicted
                return
            if len(syncs) == 2:  # the sync before the next pass writes much dirty data
                time.sleep(0.3)
            disk.free_mb += sum(cache.evicted) - sum(shown)
            shown[:] = cache.evicted

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_OVERLAY),
            patch("exec_sandbox.admission._syncfs", side_effect=sync),
        ):
            async with asyncio.timeout(1.5):
                await ctrl.release(
                    await ctrl.acquire(
                        "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                    )
                )
        # pass 1 and its sync; the slow sync; the next sync, pass 2 and its sync
        assert (len(syncs), cache.passes) == (4, [[100, 100], [100, 100]])
        await ctrl.stop()

    async def test_a_release_during_make_room_is_not_lost(self, tmp_path: Path) -> None:
        """W runs _make_room (out of the waiting list) while the VM V ends and releases its lease on the only
        entry that can make room: a release that wakes nobody leaves W to the self-wake (10.002 s). The room
        changed since W made room: W makes room again."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 400 MB disk needs 100 MB
        leased = _Cache(tmp_path / "l2", disk, [300])
        in_listing = asyncio.Event()
        go_on = asyncio.Event()

        class _LeasedThenFree(_Cache):
            def __init__(self) -> None:
                super().__init__(tmp_path / "l1", disk, [])

            async def evict_oldest(self) -> int | None:  # W's pass reaches it while V still leases L2
                in_listing.set()
                await go_on.wait()
                return None

        lease_held = [True]
        evict_l2 = leased.evict_oldest

        async def l2_evict() -> int | None:
            return None if lease_held[0] else await evict_l2()

        leased.evict_oldest = l2_evict  # type: ignore[method-assign]
        ctrl.add_cache(leased)
        ctrl.add_cache(_LeasedThenFree())
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            v = await ctrl.acquire(
                "v", memory_mb=1, cpu_cores=_CPU, disk_mb=0, timeout=1, use_tcg=False, held=frozenset()
            )
            w = asyncio.create_task(
                ctrl.acquire("w", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=2, use_tcg=False, held=frozenset())
            )
            await in_listing.wait()
            lease_held[0] = False  # V ends: its lease goes, then its admission
            await ctrl.release(v)
            go_on.set()
            async with asyncio.timeout(1):
                await ctrl.release(await w)
        assert leased.evicted == [300]

    async def test_after_stop_no_sync_starts_and_no_queued_request_lists_the_caches(self, tmp_path: Path) -> None:
        """After stop(), no pass starts a syncfs (asyncio.run would wait for its thread: 8-16 s
        on gp3), and the requests queued for the lock of the passes list no cache."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        in_pass = asyncio.Event()
        go_on = asyncio.Event()

        class _SlowRemoval(_Cache):
            async def evict_oldest(self) -> int | None:
                in_pass.set()
                await go_on.wait()
                return await super().evict_oldest()

        cache = _SlowRemoval(tmp_path / "l2", disk, [300] * 4, gives_back=0)
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_OVERLAY),
            patch("exec_sandbox.admission._syncfs") as syncfs,
        ):
            first = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=3, use_tcg=False, held=frozenset()
                )
            )
            await in_pass.wait()
            second = asyncio.create_task(
                ctrl.acquire(
                    "vm-2", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=3, use_tcg=False, held=frozenset()
                )
            )
            await _until_queued_for_the_passes(ctrl, second)
            listings = len(cache.size_read_calls)
            await ctrl.stop()
            go_on.set()
            await asyncio.gather(first, second, return_exceptions=True)
        syncfs.assert_not_called()  # vm-1's pass ended after stop(): its space showed nothing, and no sync started
        assert len(cache.size_read_calls) == listings

    async def test_a_release_lifts_the_stop_another_request_gave(self, tmp_path: Path) -> None:
        """Two passes in a row gave nothing (B's, then C's): D, which waits for its VMs too, is
        stopped with them. After a release (the VM that ended can leave new cache entries) D
        evicts again and is admitted; B and C keep their own stops."""
        disk = _Disk(1000)
        disk.written_mb = 10_000  # VMs of this controller wrote much: the requests wait
        ctrl = self._controller(tmp_path, floor_mb=1000)
        clones = _Cache(tmp_path / "l2", disk, [100] * 10, gives_back=0)
        ctrl.add_cache(clones)
        with (
            disk.patched(tmp_path),
            patch("exec_sandbox.admission._settle_removals", return_value=_AT_REMOVAL),
        ):
            first = await ctrl.acquire(
                "vm-0", memory_mb=1, cpu_cores=_CPU, disk_mb=0, timeout=1, use_tcg=False, held=frozenset()
            )
            requests = {
                name: asyncio.create_task(
                    ctrl.acquire(
                        name, memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=3, use_tcg=False, held=frozenset()
                    )
                )
                for name in ("vm-b", "vm-c", "vm-d")
            }
            await _until_waiting(ctrl, 3)
            assert sum(clones.evicted) == 1000  # two passes of nothing: all the clones
            fresh = _Cache(tmp_path / "l1", disk, [300, 300])  # saved meanwhile: they give space back
            ctrl.add_cache(fresh)
            await ctrl.release(first)
            async with asyncio.timeout(1):
                done, _ = await asyncio.wait(set(requests.values()), return_when=asyncio.FIRST_COMPLETED)
            assert fresh.evicted
            admitted = done.pop()
            for task in requests.values():
                if task is not admitted and not task.done():
                    task.cancel()
            await asyncio.gather(*requests.values(), return_exceptions=True)
        await ctrl.release(await admitted)

    def test_lifts_of_the_shared_stop_come_less_often_until_a_pass_gives_space(self, tmp_path: Path) -> None:
        """A ZFS or btrfs snapshot keeps every block: each lift of the shared stop (a release, the self-wake)
        lets one more pass evict for nothing (one lift every 10 s: 360 passes an hour). One release or self-
        wake a second: the lifts come 10 s apart, then twice as far each time, up to 640 s; after a pass that
        gives space back, the next comes at once. The verdicts and the lifts are called directly with the
        event loop time they take: through acquire(), the schedule takes 32 minutes."""
        ctrl = self._controller(tmp_path, floor_mb=0)
        space = _DiskSpace(
            free_mb=1000, written_mb=0, admitted_mb=0, reserved_mb=0, requested_mb=0, needed_mb=0, floor_mb=0
        )

        def share_a_stop() -> None:  # passes of nothing until the stop is shared
            while ctrl._no_gain_passes < 2:  # pyright: ignore[reportPrivateUsage]
                ctrl._judge(_Evictions(), evicted_mb=100, gain_mb=0, space=space)  # pyright: ignore[reportPrivateUsage]

        lifts: list[int] = []
        share_a_stop()
        for now in range(2000):
            ctrl._ease_shared_stop(now=float(now))  # pyright: ignore[reportPrivateUsage]
            if ctrl._no_gain_passes < 2:  # pyright: ignore[reportPrivateUsage]
                lifts.append(now)
                share_a_stop()
        assert lifts == [0, 10, 30, 70, 150, 310, 630, 1270, 1910]
        ctrl._judge(_Evictions(), evicted_mb=100, gain_mb=100, space=space)  # pyright: ignore[reportPrivateUsage]
        share_a_stop()
        ctrl._ease_shared_stop(now=1911.0)  # pyright: ignore[reportPrivateUsage]
        assert ctrl._no_gain_passes == 1  # pyright: ignore[reportPrivateUsage]

    async def test_the_pass_after_a_lift_evicts_for_its_own_request_only(self, tmp_path: Path) -> None:
        """A snapshot keeps every block. After a lift of the shared stop, a pass of the next request for what
        every waiter lacks evicts for nothing again: it evicts what its own request lacks only, until a pass
        gives space back."""
        disk = _Disk(1000)
        disk.written_mb = 10_000  # VMs of this controller wrote much: the requests wait
        ctrl = self._controller(tmp_path, floor_mb=1000)  # each 400 MB disk needs 100 MB
        cache = _Cache(tmp_path / "l2", disk, [100] * 40, gives_back=0)
        ctrl.add_cache(cache)
        with (
            disk.patched(tmp_path),
            patch("exec_sandbox.admission._settle_removals", return_value=_AT_REMOVAL),
        ):
            vm = await ctrl.acquire(
                "vm-0", memory_mb=1, cpu_cores=_CPU, disk_mb=0, timeout=1, use_tcg=False, held=frozenset()
            )
            requests = [
                asyncio.create_task(
                    ctrl.acquire(
                        name, memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                    )
                )
                for name in ("vm-b", "vm-c", "vm-d")
            ]
            await _until_waiting(ctrl, 3)
            # B's pass for the three, C's for C only (after one pass of nothing): the stop is
            # shared, and D waits with it
            assert cache.passes == [[100, 100, 100], [100]]
            await ctrl.release(vm)
            await _until(lambda: len(cache.passes) == 3, what="the pass after the lift")
            await _until_waiting(ctrl, 3)  # D waits again: its pass has its verdict
            for task in requests:
                task.cancel()
            await asyncio.gather(*requests, return_exceptions=True)
        assert cache.passes == [[100, 100, 100], [100], [100]]

    async def test_after_a_pass_that_gives_space_the_passes_evict_for_every_waiter_again(self, tmp_path: Path) -> None:
        """B's and C's passes take clones (nothing back): the stop is shared, D, E and F wait with it. Once
        B's pass gave nothing, C's evicts only what C lacks (passes sized for every waiter empty a reflink
        cache). A release lifts the stop; D's pass evicts what D lacks only, gives space back and admits D.
        From then on a pass evicts for every waiter again (one ZFS transaction sync then shows the space of
        all): E's pass covers the four others."""
        disk = _Disk(20_000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # each 400 MB disk needs 100 MB
        cache = _GivesBack(tmp_path / "l2", disk, [(100, 0.0)] * 2 + [(100, 1.0)] * 10)  # B's and C's clones
        ctrl.add_cache(cache)
        with (
            disk.patched(tmp_path),
            patch("exec_sandbox.admission._settle_removals", return_value=_AT_REMOVAL),
            _probe_clock() as later,
        ):
            vm = await ctrl.acquire(
                "vm-0", memory_mb=1, cpu_cores=_CPU, disk_mb=0, timeout=1, use_tcg=False, held=frozenset()
            )
            long = await ctrl.acquire(
                "long", memory_mb=1, cpu_cores=_CPU, disk_mb=40_000, timeout=1, use_tcg=False, held=frozenset()
            )
            disk.free_mb = 11_000  # its reserve: 10000 MB; each request then lacks 100 MB, and waits for it
            later()
            requests = []
            for count, names in enumerate((["vm-b"], ["vm-c"], ["vm-d", "vm-e", "vm-f"]), start=1):
                requests += [
                    asyncio.create_task(
                        ctrl.acquire(
                            name, memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                        )
                    )
                    for name in names
                ]
                await _until_waiting(ctrl, len(requests) if count < 3 else 5)
            await ctrl.release(vm)
            async with asyncio.timeout(1):
                reservations = await asyncio.gather(*requests)
        for reservation in [*reservations, long]:
            await ctrl.release(reservation)
        assert cache.passes == [[100], [100], [100], [100] * 4]  # B's, C's, D's; E's for B, C, E and F

    @pytest.mark.parametrize("first", ["leaves", "waits"])
    async def test_a_waiter_that_cannot_use_a_release_passes_it_on(self, tmp_path: Path, first: str) -> None:
        """A release wakes one waiter when all ask the same size. Its pass took a clone (nothing
        back) and stopped it: it leaves with LOCAL_DISK, or waits for its VMs. The wake-up goes
        on to the next waiter, which the good entries behind the clone admit (else it sleeps
        until the 10 s self-wake)."""
        disk = _Disk(1000)
        if first == "waits":
            disk.written_mb = 10_000  # VMs of this controller wrote much: a stopped request waits
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 400 MB disk needs 100 MB
        leased = [True]

        class _LeasedByV(_GivesBack):
            async def evict_oldest(self) -> int | None:
                return None if leased[0] else await super().evict_oldest()

        cache = _LeasedByV(tmp_path / "l2", disk, [(300, 0.0), (200, 1.0), (200, 1.0)])
        ctrl.add_cache(cache)
        with (
            disk.patched(tmp_path),
            patch("exec_sandbox.admission._settle_removals", return_value=_AT_REMOVAL),
        ):
            v = await ctrl.acquire(
                "v", memory_mb=1, cpu_cores=_CPU, disk_mb=0, timeout=1, use_tcg=False, held=frozenset()
            )
            w_a = asyncio.create_task(
                ctrl.acquire(
                    "w-a", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting(ctrl, 1)
            w_b = asyncio.create_task(
                ctrl.acquire(
                    "w-b", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting(ctrl, 2)
            leased[0] = False  # V ends: its leases go, then its admission
            await ctrl.release(v)
            async with asyncio.timeout(1):
                reservation = await w_b
            await ctrl.release(reservation)
            w_a.cancel()
            (outcome,) = await asyncio.gather(w_a, return_exceptions=True)
        assert cache.evicted == [300, 200]  # W_a's pass took the clone, W_b's the good entry behind it
        if first == "leaves":
            assert isinstance(outcome, TransientError) and outcome.code is ErrorCode.LOCAL_DISK

    async def test_one_pass_of_nothing_after_a_release_shares_the_stop_again(self, tmp_path: Path) -> None:
        """A release lifts the shared stop of two passes of nothing, and the next pass of nothing
        shares it again: the count goes back to one, not to zero. Every entry is a clone
        (nothing back), and no VM of this controller holds a disk: a request fails with
        LOCAL_DISK after its pass. A and B evict; C, under the shared stop, evicts nothing; the
        release lifts it, and D evicts once more; E, under the stop D shared again, evicts
        nothing."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # each 400 MB disk needs 100 MB
        cache = _Cache(tmp_path / "l2", disk, [100] * 10, gives_back=0)
        ctrl.add_cache(cache)
        evicted: dict[str, float] = {}
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            vm = await ctrl.acquire(
                "vm", memory_mb=1, cpu_cores=_CPU, disk_mb=0, timeout=1, use_tcg=False, held=frozenset()
            )
            for name in ("a", "b", "c", "release", "d", "e"):
                if name == "release":
                    await ctrl.release(vm)
                    continue
                before = sum(cache.evicted)
                with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                    await ctrl.acquire(
                        name, memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=1, use_tcg=False, held=frozenset()
                    )
                evicted[name] = sum(cache.evicted) - before
        assert evicted == {"a": 100, "b": 100, "c": 0, "d": 100, "e": 0}

    async def test_the_self_wake_lifts_the_stop_of_a_request_that_waits_behind_a_vm(self, tmp_path: Path) -> None:
        """A VM of this controller runs and does not end; two passes of nothing shared a stop,
        and W, which came after, waits with it. The self-wake lifts it: W evicts the good
        entries behind the clones and is admitted, with no release."""
        disk = _Disk(1000)
        disk.written_mb = 10_000  # VMs of this controller wrote much: the requests wait
        ctrl = self._controller(tmp_path, floor_mb=1000)  # each 400 MB disk needs 100 MB
        cache = _GivesBack(tmp_path / "l2", disk, [(100, 0.0)] * 2 + [(200, 1.0)] * 5)  # A's and B's clones
        ctrl.add_cache(cache)
        with (
            disk.patched(tmp_path),
            patch("exec_sandbox.admission._settle_removals", return_value=_AT_REMOVAL),
        ):
            vm = await ctrl.acquire(
                "vm-0", memory_mb=1, cpu_cores=_CPU, disk_mb=0, timeout=1, use_tcg=False, held=frozenset()
            )
            stopped = []
            for count, name in enumerate(("vm-a", "vm-b"), start=1):  # one pass of nothing each
                stopped.append(
                    asyncio.create_task(
                        ctrl.acquire(
                            name, memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                        )
                    )
                )
                await _until_waiting(ctrl, count)
            w = asyncio.create_task(
                ctrl.acquire(
                    "vm-w", memory_mb=1, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting(ctrl, 3)
            assert sum(cache.evicted) == 200  # the clones: after A's, B's pass evicts for B only; W evicts nothing
            await _self_wake(ctrl)
            async with asyncio.timeout(1):
                await ctrl.release(await w)
            for task in stopped:
                task.cancel()
            await asyncio.gather(*stopped, return_exceptions=True)
            await ctrl.release(vm)
        assert sum(cache.evicted) == 400  # and one good entry for W

    def test_a_pass_that_gives_space_back_ends_the_run_of_passes_of_nothing(self, tmp_path: Path) -> None:
        """Passes of nothing count in a row: a pass that gave space back between two of them
        resets the count, so the second stops only its own request, not the others. The
        verdicts are called directly: through acquire(), the pass that gives space back is
        followed by one that makes its request fit, which resets the count too, so the reset by
        the verdict alone does not show."""
        ctrl = self._controller(tmp_path, floor_mb=0)
        space = _DiskSpace(
            free_mb=1000, written_mb=0, admitted_mb=0, reserved_mb=0, requested_mb=0, needed_mb=0, floor_mb=0
        )
        other = _Evictions()
        ctrl._disk_needs["other"] = (100, other)  # pyright: ignore[reportPrivateUsage]
        ctrl._judge(_Evictions(), evicted_mb=100, gain_mb=0, space=space)  # pyright: ignore[reportPrivateUsage]
        ctrl._judge(_Evictions(), evicted_mb=100, gain_mb=100, space=space)  # pyright: ignore[reportPrivateUsage]
        ctrl._judge(_Evictions(), evicted_mb=100, gain_mb=0, space=space)  # pyright: ignore[reportPrivateUsage]
        assert not other.stopped
        ctrl._judge(_Evictions(), evicted_mb=100, gain_mb=0, space=space)  # pyright: ignore[reportPrivateUsage]
        assert other.stopped and other.shared_stop

    async def test_stop_ends_the_passes_of_evictions(self, tmp_path: Path) -> None:
        """stop() while a request runs its passes (btrfs compression: each eviction gives 1/32 back): passes
        that go on evict 156 entries after stop(). The passes end at the next eviction, and no re-check timer
        starts after stop(). The timer is asked for directly: a pass reaches it after stop() only in a race (a
        sync that stop() ends)."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)

        class _SlowRemovals(_Cache):
            async def evict_oldest(self) -> int | None:
                await asyncio.sleep(0.02)  # a removal in a worker thread
                return await super().evict_oldest()

        cache = _SlowRemovals(tmp_path / "l2", disk, [100] * 200, gives_back=1 / 32)
        ctrl.add_cache(cache)
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=8000, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await cache.evicted_one.wait()
            await ctrl.stop()
            evicted_at_stop = len(cache.evicted)
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE):
                async with asyncio.timeout(1):
                    await waiter
        assert len(cache.evicted) <= evicted_at_stop + 1
        ctrl._recheck_soon()  # pyright: ignore[reportPrivateUsage]
        assert ctrl._recheck_task is None  # pyright: ignore[reportPrivateUsage]

    async def test_a_request_that_fits_at_the_lock_lists_nothing(self, tmp_path: Path) -> None:
        """A pass of another request made room while this one waited for the lock of the passes: it fits at
        the lock, and goes on with no listing of the caches (else each queued request lists every cache before
        it sees that it fits). A's pass evicts one 600 MB entry: room for A and B."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        in_pass, go_on = asyncio.Event(), asyncio.Event()

        class _SlowRemoval(_Cache):
            async def evict_oldest(self) -> int | None:
                in_pass.set()
                await go_on.wait()
                return await super().evict_oldest()

        cache = _SlowRemoval(tmp_path / "l2", disk, [600])
        ctrl.add_cache(cache)
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            a = asyncio.create_task(
                ctrl.acquire(
                    "vm-a", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
            )
            await in_pass.wait()  # A holds the lock of the passes
            b = asyncio.create_task(
                ctrl.acquire(
                    "vm-b", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=2, use_tcg=False, held=frozenset()
                )
            )
            await _until_queued_for_the_passes(ctrl, b)
            listings = len(cache.size_read_calls)
            go_on.set()
            async with asyncio.timeout(1):
                reservations = await asyncio.gather(a, b)
        for reservation in reservations:
            await ctrl.release(reservation)
        assert (listings, len(cache.size_read_calls)) == (1, 1)  # A's listing only
        assert cache.evicted == [600]

    def test_a_pass_evicts_only_for_the_waiters_that_can_fit(self, tmp_path: Path) -> None:
        """The other waiters' reserves count only when each can fit after this request's
        listing: a 16 GB reserve on a filesystem with 2 GB to gain adds nothing (a pass evicted
        1899 MB where 829 MB served every request that could fit). The sum is read directly:
        through acquire(), a waiter that cannot fit after the listing of another request has
        counted other room at its own check (other leases in `held`, VMs that ended since), a
        state of many steps."""
        ctrl = self._controller(tmp_path, floor_mb=1000)
        start = _DiskSpace(
            free_mb=1000, written_mb=0, admitted_mb=0, reserved_mb=0, requested_mb=400, needed_mb=100, floor_mb=1000
        )
        ctrl._disk_needs.update(  # pyright: ignore[reportPrivateUsage]
            {"own": (100.0, _Evictions()), "small": (100.0, _Evictions()), "huge": (16_000.0, _Evictions())}
        )
        assert ctrl._others_mb(start, later_mb=2000) == 100  # pyright: ignore[reportPrivateUsage]

    async def test_an_admission_does_not_make_the_disk_waiters_list_the_caches(self, tmp_path: Path) -> None:
        """Requests that gate 4 blocks wait for their VMs; small requests that fit come and go. Each admission
        wakes them all (mixed sizes); if each woken waiter listed the caches again, 100 waiters and 20
        admissions would give 2000 listings. They make room again only when the room changed (a release, a
        pass that gave space, the self-wake), and at one room version only until a pass finds nothing to
        evict."""
        disk = _Disk(1000)
        disk.written_mb = 10_000  # VMs of this controller wrote much: the requests wait
        ctrl = self._controller(tmp_path, floor_mb=1000)
        cache = _Cache(tmp_path / "l2", disk, [])
        ctrl.add_cache(cache)
        with disk.patched(tmp_path):
            waiters = [
                asyncio.create_task(
                    ctrl.acquire(
                        f"w{i}",
                        memory_mb=1,
                        cpu_cores=_CPU,
                        disk_mb=4000 + 1000 * (i % 2),
                        timeout=5,
                        use_tcg=False,
                        held=frozenset(),
                    )
                )
                for i in range(10)
            ]
            await _until_waiting(ctrl, 10)
            listings = len(cache.size_read_calls)

            def all_wait() -> bool:  # every waiter is back in the wait, no pass runs
                return sum(ctrl._waiting.values()) >= 10 and not ctrl._evict_lock.locked()  # pyright: ignore[reportPrivateUsage]

            for i in range(5):
                small = await ctrl.acquire(
                    f"s{i}", memory_mb=1, cpu_cores=_CPU, disk_mb=0, timeout=1, use_tcg=False, held=frozenset()
                )
                await ctrl.release(small)  # a release: the room changed
                await _until(lambda i=i: len(cache.size_read_calls) - listings > i, what="the listing of a waiter")
                await _until(all_wait, what="all waiters in the wait again")
            # one per release: the first waiter finds nothing to evict, the others skip until the
            # room changes again (not one per release and waiter); none per admission
            assert len(cache.size_read_calls) - listings == 5
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)

    async def test_a_waiter_that_fits_the_disk_only_wakes_nobody(self, tmp_path: Path) -> None:
        """While its pass is pending, S fits gate 4 (another process freed space) but waits on the memory
        budget: a timer that wakes all the waiters every 50 ms for the whole pending time costs 954
        listings/s. It wakes them only when a waiting request passes every gate. The wake-ups are counted on
        the condition: a woken waiter whose room did not change lists nothing, so no file call shows them;
        their cost is CPU."""
        disk = _Disk(1050)
        ctrl = ResourceAdmissionController(
            disk_dir=tmp_path,
            disk_free_floor_mb=1000,
            memory_overcommit_ratio=1.0,
            cpu_overcommit_ratio=1.0,
            disk_overcommit_ratio=4.0,  # a 400 MB disk reserves 100 MB: 50 MB lacking in 1050 - 1000
            host_memory_reserve_ratio=0.0,
            host_memory_mb=1000.0,
            host_cpu_count=1000.0,
            available_memory_floor_mb=0,
            host_cpu_reserve_cores=DEFAULT_HOST_CPU_RESERVE_CORES,
        )
        cache = _Cache(tmp_path / "l2", disk, [5000] * 4, gives_back=0)  # 60 MB is no gain of a 5000 MB pass
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_zfs(5.0)),
        ):
            s_task = asyncio.create_task(
                ctrl.acquire(
                    "s", memory_mb=400, cpu_cores=_CPU, disk_mb=400, timeout=3, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting_or_done(ctrl, s_task)  # its pass is pending: 50 MB lacking
            held = await ctrl.acquire(
                "held", memory_mb=600, cpu_cores=_CPU, disk_mb=0, timeout=1, use_tcg=False, held=frozenset()
            )
            disk.free_mb += 60  # another process frees 60 MB: S fits gate 4, not the memory budget
            wakes = 0
            notify_all = ctrl._condition.notify_all  # pyright: ignore[reportPrivateUsage]

            def counted() -> None:
                nonlocal wakes
                wakes += 1
                notify_all()

            with patch.object(ctrl._condition, "notify_all", counted):  # pyright: ignore[reportPrivateUsage]
                await asyncio.sleep(0.5)
            assert wakes == 0
            s_task.cancel()
            await asyncio.gather(s_task, return_exceptions=True)
            await ctrl.release(held)
        await ctrl.stop()

    @pytest.mark.parametrize("slow", ["reads", "sync"])
    async def test_stop_ends_a_pass_that_reads_the_space_or_waits_for_a_sync(self, tmp_path: Path, slow: str) -> None:
        """stop() during a pass that reads the space while it grows (XFS, up to 1 s) or waits for a syncfs (up
        to 1 s): a pass that goes on ends acquire() 1-2 s after stop(). It ends at once."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300] * 4, gives_back=0)
        ctrl.add_cache(cache)
        loop = asyncio.get_running_loop()
        # Reads: the space grows for 1 s (0.2 MB every 10 ms), and the pass reads it to its limit. Sync:
        # no growth, so the pass gains nothing and syncs; with growth, a loop stall that ends 0.235 s
        # after the pass's first read makes the gain reach 1/64 of the pass, which skips the sync
        if slow == "reads":
            for step in range(1, 100):
                loop.call_later(step / 100, lambda: setattr(disk, "free_mb", disk.free_mb + 0.2))
        settle = (
            _Settle(quiet=0.15, limit=1.0, sync_once=False, later=0.0, counts_writes=True)
            if slow == "reads"
            else _OVERLAY
        )
        sync_started, sync_ends = threading.Event(), threading.Event()  # a sync that writes much dirty data
        reads = 0

        def usage(path: object) -> SimpleNamespace:
            nonlocal reads
            reads += 1
            return disk.usage(path)

        def slow_sync(_path: Path) -> None:
            sync_started.set()
            sync_ends.wait(5)  # it ends with the test

        with (
            patch("exec_sandbox.admission.shutil.disk_usage", usage),
            patch("exec_sandbox.admission._settle_removals", return_value=settle),
            patch("exec_sandbox.admission._syncfs", side_effect=slow_sync),
        ):
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await cache.evicted_one.wait()
            if slow == "reads":  # one read after the eviction, then the reads while the space grows
                evicted_at = reads
                await _until(lambda: reads >= evicted_at + 3, what="the reads of the space after the pass")
            else:
                assert await asyncio.to_thread(sync_started.wait, 5)
            stopped_at = loop.time()
            await ctrl.stop()
            try:
                with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE):
                    await waiter
                assert loop.time() - stopped_at < 0.3
            finally:
                sync_ends.set()

    async def test_waiters_are_not_woken_while_a_pass_is_pending(self, tmp_path: Path) -> None:
        """While a pass is pending and no waiter fits, a re-check timer that wakes them all twice a second
        makes each list the caches again (100 waiters: 197 listings/s). It wakes them once the pending pass
        ends, or one fits. The wake-ups are counted on the condition: a woken waiter whose room did not change
        lists nothing, so no file call shows them."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300] * 4, gives_back=0)
        ctrl.add_cache(cache)
        with (
            patch("exec_sandbox.admission.shutil.disk_usage", disk.usage),
            patch("exec_sandbox.admission._settle_removals", return_value=_zfs(30.0)),
        ):
            waiter = asyncio.create_task(
                ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting_or_done(ctrl, waiter)
            wakes = 0
            notify_all = ctrl._condition.notify_all  # pyright: ignore[reportPrivateUsage]

            def counted() -> None:
                nonlocal wakes
                wakes += 1
                notify_all()

            with patch.object(ctrl._condition, "notify_all", counted):  # pyright: ignore[reportPrivateUsage]
                await asyncio.sleep(1.2)  # more than two 0.5 s periods of a fixed re-check
            assert wakes == 0
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        await ctrl.stop()

    async def test_a_request_stopped_by_the_verdict_on_a_pending_pass_evicts_nothing(self, tmp_path: Path) -> None:
        """The verdict on a pending pass can come as the next pass starts, and stop that
        pass's own request: then that pass evicts nothing. The pass runs directly on a
        pending pass whose time is over: through acquire(), the re-check timer gives that
        verdict first, so the window is a race."""
        disk = _Disk(1000)
        ctrl = self._controller(tmp_path, floor_mb=1000)  # a 1024 MB disk needs 256 MB more
        cache = _Cache(tmp_path / "l2", disk, [300] * 4, gives_back=0)
        ctrl.add_cache(cache)
        evictions = _Evictions()
        with patch("exec_sandbox.admission.shutil.disk_usage", disk.usage):
            start = ctrl._disk_space(1024)  # pyright: ignore[reportPrivateUsage]
            assert start is not None
            ctrl._pending = _Pending(  # pyright: ignore[reportPrivateUsage]
                evictions=evictions,
                start=start,
                evicted_mb=300,
                net_of_writes=True,
                until=asyncio.get_running_loop().time() - 1,  # its time is over: nothing showed
                sync=None,
                shown_mb=0.0,
                grew_at=None,
            )
            assert not await ctrl._evict_for_disk(1024, evictions=evictions, later_mb=0.0)  # pyright: ignore[reportPrivateUsage]
        assert evictions.stopped
        assert cache.evicted == []

    async def test_instance_work_files_come_back_when_sessions_end(self, tmp_path: Path) -> None:
        """Saved-instance work files (disk_dir/instances) go when their session closes: a
        request that fits once they are gone waits instead of failing with LOCAL_DISK; the
        loaded state the request itself keeps (`held`) does not count."""
        session = tmp_path / "instances" / "session-1"
        session.mkdir(parents=True)
        (session / "home").write_bytes(os.urandom(20 * _MIB))
        ctrl = self._controller(tmp_path, floor_mb=1000)
        with patch("exec_sandbox.admission.shutil.disk_usage", _Disk(1000).usage):  # a 64 MB disk needs 16 MB more
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.CAPACITY):
                await ctrl.acquire(
                    "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=64, timeout=_REFUSED, use_tcg=False, held=frozenset()
                )
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                await ctrl.acquire(
                    "vm-2", memory_mb=1, cpu_cores=_CPU, disk_mb=64, timeout=2, use_tcg=False, held=frozenset({session})
                )

    async def test_waits_when_the_end_of_its_own_vm_makes_room(self, tmp_path: Path) -> None:
        """The reserve of an admitted VM comes back at its release: the request waits for it
        instead of failing at once."""
        ctrl = self._controller(tmp_path, floor_mb=1000)
        with patch("exec_sandbox.admission.shutil.disk_usage", _Disk(1300).usage):
            first = await ctrl.acquire(
                "vm-1", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=0.2, use_tcg=False, held=frozenset()
            )
            second = asyncio.create_task(
                ctrl.acquire(
                    "vm-2", memory_mb=1, cpu_cores=_CPU, disk_mb=1024, timeout=5, use_tcg=False, held=frozenset()
                )
            )
            await _until_waiting(ctrl, 1)
            await ctrl.release(first)
            async with asyncio.timeout(1):
                await ctrl.release(await second)


# ============================================================================
# Wake-up with mixed request sizes (VmSpec sizes differ)
# ============================================================================


def _written_mb(ctrl: ResourceAdmissionController) -> float:
    """The MB the home disks of disk_dir have written, as gate 4's probe reads it (private:
    the gate admits on it with the reserves of its own VMs, which these tests hold none of)."""
    probe = ctrl._probe_disk()  # pyright: ignore[reportPrivateUsage]
    assert probe is not None
    return probe[1]


async def _self_wake(ctrl: ResourceAdmissionController) -> None:
    """One wake-up of the self-wake timer, now: the body of its loop (private). The timer runs
    every 10 s, so no test waits for it; test_self_wake_timer_unblocks_gate3_waiters runs it."""
    await ctrl._self_wake()  # pyright: ignore[reportPrivateUsage]


async def _until_waiting_or_done(ctrl: ResourceAdmissionController, task: asyncio.Task[object]) -> None:
    """Until `task` (an acquire) has ended or blocks in the controller (2 s at most). The
    controller has no public waiter count: its counter (private) is polled."""
    for _ in range(2000):
        if task.done() or ctrl._waiting:  # pyright: ignore[reportPrivateUsage]
            return
        await asyncio.sleep(0.001)
    raise AssertionError("the acquire neither ended nor blocked")


async def _until_queued_for_the_passes(ctrl: ResourceAdmissionController, task: asyncio.Task[object]) -> None:
    """Until `task` (an acquire) has ended or waits for the lock of the passes of evictions (2 s
    at most). No public fact shows that wait: the waiters of that lock (private) are polled."""
    for _ in range(2000):
        if task.done() or ctrl._evict_lock._waiters:  # pyright: ignore[reportPrivateUsage]
            return
        await asyncio.sleep(0.001)
    raise AssertionError("the acquire neither ended nor waited for the lock of the passes")


async def _until(condition: Callable[[], bool], *, what: str) -> None:
    """Until `condition` holds (5 s at most): order by fact, not by sleep."""
    for _ in range(5000):
        if condition():
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"never: {what}")


async def _until_waiting(ctrl: ResourceAdmissionController, count: int) -> None:
    """Until `count` acquire() calls block in the controller (order by fact, not by sleep)."""
    for _ in range(5000):  # the controller has no waiter event: poll its counter (5 s at most)
        if sum(ctrl._waiting.values()) >= count:  # pyright: ignore[reportPrivateUsage]
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"{count} waiters never blocked")


class TestMixedSizeWakeUp:
    """A release must reach the waiters that now fit, not only the first in line."""

    _PROMPT = 2.0  # well under the 10 s self-wake: only a real wake-up admits in time

    async def test_small_waiter_behind_large_one_is_admitted(self) -> None:
        """A budget of 672 MB holds two VMs of 256 MB (336 MB each). A large request (592 MB,
        672 with its overhead) waits first, a small one behind it; one release admits the
        small one."""
        ctrl = _make_controller(host_memory_mb=672.0, memory_overcommit=1.0, reserve_ratio=0.0, cpu_overcommit=100.0)
        held = [
            await ctrl.acquire(
                f"hold-{i}", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
            )
            for i in range(2)
        ]
        large = asyncio.create_task(
            ctrl.acquire("large", memory_mb=592, cpu_cores=_CPU, disk_mb=0, timeout=30, use_tcg=False, held=frozenset())
        )
        await _until_waiting(ctrl, 1)  # the large one waits first
        small = asyncio.create_task(
            ctrl.acquire("small", memory_mb=256, cpu_cores=_CPU, disk_mb=0, timeout=30, use_tcg=False, held=frozenset())
        )
        await _until_waiting(ctrl, 2)

        await ctrl.release(held[0])

        await ctrl.release(await asyncio.wait_for(small, self._PROMPT))
        assert not large.done()
        large.cancel()
        await ctrl.release(held[1])

    async def test_one_large_release_admits_several_same_size_waiters(self) -> None:
        """A budget of 816 MB holds three VMs of 192 MB (272 MB each). One VM of 736 MB (816
        with its overhead) takes it all; its release admits the three waiters of the same size."""
        ctrl = _make_controller(host_memory_mb=816.0, memory_overcommit=1.0, reserve_ratio=0.0, cpu_overcommit=100.0)
        big = await ctrl.acquire(
            "big", memory_mb=736, cpu_cores=_CPU, disk_mb=0, timeout=_REFUSED, use_tcg=False, held=frozenset()
        )
        waiters = [
            asyncio.create_task(
                ctrl.acquire(
                    f"w-{i}", memory_mb=192, cpu_cores=_CPU, disk_mb=0, timeout=30, use_tcg=False, held=frozenset()
                )
            )
            for i in range(3)
        ]
        await _until_waiting(ctrl, 3)

        await ctrl.release(big)

        for reservation in await asyncio.wait_for(asyncio.gather(*waiters), self._PROMPT):
            await ctrl.release(reservation)
        assert not ctrl._waiting  # pyright: ignore[reportPrivateUsage]


_SHOWN_AT_REMOVAL = _Settle(quiet=0.0, limit=0.0, sync_once=False, later=0.0, counts_writes=True)
"""ext4, tmpfs, APFS: read once."""
_SHOWN_AFTER_COMMIT = _Settle(quiet=0.01, limit=0.05, sync_once=False, later=0.0, counts_writes=False)
"""btrfs after its commit: read again for 10 ms of no change, 50 ms at most."""
_SHOWN_IN_BACKGROUND = _Settle(quiet=0.15, limit=1.0, sync_once=False, later=0.0, counts_writes=True)
"""XFS: read again until no growth for 150 ms, 1 s at most."""
_SHOWN_AT_A_SYNC = _Settle(quiet=0.15, limit=1.0, sync_once=True, later=0.0, counts_writes=False)
"""An overlay with no upper folder in sight: read again as on XFS, then syncfs once."""
_BTRFS, _XFS, _ZFS, _OVERLAY_FS = 0x9123683E, 0x58465342, 0x2FC12FC1, 0x794C7630
"""statfs f_type values (linux/magic.h; ZFS: its own ZFS_SUPER_MAGIC)."""
_START_SYNC, _WAIT_SYNC = 0x80089418, 0x40089416
"""BTRFS_IOC_START_SYNC: _IOR(0x94, 24, __u64); BTRFS_IOC_WAIT_SYNC: _IOW(0x94, 22, __u64)."""


class TestSettleRemovals:
    """_settle_removals: how long the free space of each filesystem can take to show removals.
    It is called directly (private): through acquire(), each case needs that filesystem on the
    host. The statfs type and the ioctl are faked at the OS call."""

    def test_the_real_call_on_a_real_folder(self, tmp_path: Path) -> None:
        """No error on a real folder. macOS (APFS) shows a removal at once."""
        settled = _settle_removals(tmp_path)
        if detect_host_os() is HostOS.MACOS:
            assert settled == _SHOWN_AT_REMOVAL

    @staticmethod
    def _on_linux(fs_type: int) -> contextlib.AbstractContextManager[object]:
        """A Linux host whose statfs names `fs_type`."""
        stack = contextlib.ExitStack()
        stack.enter_context(patch("exec_sandbox.admission.detect_host_os", return_value=HostOS.LINUX))
        stack.enter_context(patch("exec_sandbox.admission.statfs_type", return_value=fs_type))
        return stack

    def test_btrfs_commits_and_waits_for_the_commit(self, tmp_path: Path) -> None:
        """btrfs frees removed blocks at its commit: start one and wait for that transaction
        (its id from START_SYNC goes to WAIT_SYNC). No syncfs: it writes every dirty page of the
        filesystem first (16 s with 2 GB on gp3)."""
        calls: list[tuple[int, bytes]] = []

        def ioctl(_fd: int, request: int, arg: bytearray) -> int:
            calls.append((request, bytes(arg)))
            if request == _START_SYNC:
                arg[:] = (42).to_bytes(8, sys.byteorder)
            return 0

        with self._on_linux(_BTRFS), patch("exec_sandbox.admission.fcntl.ioctl", side_effect=ioctl):
            assert _settle_removals(tmp_path) == _SHOWN_AFTER_COMMIT
        assert calls == [(_START_SYNC, bytes(8)), (_WAIT_SYNC, (42).to_bytes(8, sys.byteorder))]

    def test_a_failed_btrfs_commit_reads_the_space_again(self, tmp_path: Path) -> None:
        """A commit ioctl that fails (not btrfs after all): the space is read again, then syncfs
        once, as for an unknown type."""
        with (
            self._on_linux(_BTRFS),
            patch("exec_sandbox.admission.fcntl.ioctl", side_effect=OSError(errno.ENOTTY, "not btrfs")),
        ):
            assert _settle_removals(tmp_path) == _SHOWN_AT_A_SYNC

    @pytest.mark.parametrize(
        ("fs_type", "settled"),
        [
            (0xEF53, _SHOWN_AT_REMOVAL),
            (0x01021994, _SHOWN_AT_REMOVAL),
            (_XFS, _SHOWN_IN_BACKGROUND),
            (_OVERLAY_FS, _SHOWN_AT_A_SYNC),  # overlay, whatever filesystem it writes to
        ],
        ids=["ext4", "tmpfs", "xfs", "overlay"],
    )
    def test_other_filesystems_need_no_call(self, tmp_path: Path, fs_type: int, settled: _Settle) -> None:
        """ext4 and tmpfs show a removal at once; XFS frees it in the background (read again);
        an overlay frees it when its upper filesystem commits (read again, then syncfs once)."""
        with self._on_linux(fs_type), patch("exec_sandbox.admission.fcntl.ioctl") as ioctl:
            assert _settle_removals(tmp_path) == settled
        ioctl.assert_not_called()

    @pytest.mark.parametrize(
        ("upper_type", "settled"),
        [(0xEF53, _SHOWN_AT_REMOVAL), (_XFS, _SHOWN_IN_BACKGROUND), (_BTRFS, _SHOWN_AFTER_COMMIT)],
        ids=["ext4", "xfs", "btrfs"],
    )
    def test_an_overlay_settles_as_its_upper_filesystem(
        self, tmp_path: Path, upper_type: int, settled: _Settle
    ) -> None:
        """An overlay reports its own statfs type whatever filesystem it writes to: where this
        process sees the upper folder, that filesystem decides (an overlay over ZFS showed its
        removals only at the transaction sync, over ext4 at once) and whether the free
        space drops as data is written. btrfs commits through the upper folder."""
        upper = tmp_path / "upper"
        upper.mkdir()
        calls: list[int] = []

        def statfs_type(path: Path) -> int:
            return upper_type if path == upper else _OVERLAY_FS

        with (
            patch("exec_sandbox.admission.detect_host_os", return_value=HostOS.LINUX),
            patch("exec_sandbox.admission.statfs_type", side_effect=statfs_type),
            patch("exec_sandbox.admission.overlay_upper", return_value=upper),
            patch("exec_sandbox.admission.fcntl.ioctl", side_effect=lambda _fd, request, _arg: calls.append(request)),
        ):
            assert _settle_removals(tmp_path) == settled
        assert calls == ([_START_SYNC, _WAIT_SYNC] if upper_type == _BTRFS else [])

    def test_a_zfs_module_this_user_may_not_see_reads_as_none(self, tmp_path: Path) -> None:
        """A zfs_txg_timeout path this user cannot reach reads as no ZFS module, never as a PermissionError
        out of acquire() (Path.exists raises it)."""
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0)
        try:
            with patch("exec_sandbox.admission._ZFS_TXG_TIMEOUT", locked / "zfs" / "parameters" / "zfs_txg_timeout"):
                assert _zfs_loaded() is False
        finally:
            locked.chmod(0o700)

    def test_zfs_shows_a_removal_at_its_next_transaction_sync(self, tmp_path: Path) -> None:
        """ZFS shows a removal only at its next transaction sync (zfs_txg_timeout, 5 s by
        default), in one step, and syncfs does not start one: the pass reads it once and counts
        it as still coming for two sync intervals and a second (that sync can end late)."""
        txg = tmp_path / "zfs_txg_timeout"
        txg.write_text("5\n")
        with self._on_linux(_ZFS), patch("exec_sandbox.admission._ZFS_TXG_TIMEOUT", txg):
            assert _settle_removals(tmp_path) == _Settle(
                quiet=0, limit=0, sync_once=False, later=11, counts_writes=True
            )

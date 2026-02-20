"""Tests for ResourceMonitor.

Verifies the background monitoring task starts, ticks, and stops cleanly.
Uses mocked VmManager to avoid needing real VMs.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from exec_sandbox.admission import ResourceAdmissionController
from exec_sandbox.constants import (
    DEFAULT_CPU_OVERCOMMIT_RATIO,
    DEFAULT_HOST_MEMORY_RESERVE_RATIO,
    DEFAULT_MEMORY_OVERCOMMIT_RATIO,
)
from exec_sandbox.resource_monitor import ResourceMonitor

# ============================================================================
# Helpers
# ============================================================================


def _make_admission() -> ResourceAdmissionController:
    """Create an admission controller with known host resources."""
    return ResourceAdmissionController(
        memory_overcommit_ratio=DEFAULT_MEMORY_OVERCOMMIT_RATIO,
        cpu_overcommit_ratio=DEFAULT_CPU_OVERCOMMIT_RATIO,
        host_memory_reserve_ratio=DEFAULT_HOST_MEMORY_RESERVE_RATIO,
        host_memory_mb=16_000.0,
        host_cpu_count=8.0,
    )


def _make_vm_manager_mock(active_vms: dict | None = None) -> MagicMock:
    """Create a mock VmManager that returns given active VMs."""
    mock = MagicMock()
    mock.get_active_vms.return_value = active_vms or {}
    return mock


# ============================================================================
# Start / Stop
# ============================================================================


async def test_monitor_starts_and_stops() -> None:
    """Monitor starts a background task and stops it cleanly."""
    admission = _make_admission()
    vm_manager = _make_vm_manager_mock()

    monitor = ResourceMonitor(vm_manager, admission, interval_seconds=0.1)
    await monitor.start()
    assert monitor._task is not None
    assert not monitor._task.done()

    await monitor.stop()
    assert monitor._task is None


async def test_monitor_start_is_idempotent() -> None:
    """Calling start() twice doesn't create a second task."""
    admission = _make_admission()
    vm_manager = _make_vm_manager_mock()

    monitor = ResourceMonitor(vm_manager, admission, interval_seconds=0.1)
    await monitor.start()
    task1 = monitor._task

    await monitor.start()
    assert monitor._task is task1

    await monitor.stop()


async def test_monitor_stop_without_start_is_safe() -> None:
    """Calling stop() before start() is a no-op."""
    admission = _make_admission()
    vm_manager = _make_vm_manager_mock()
    monitor = ResourceMonitor(vm_manager, admission)
    await monitor.stop()  # Should not raise


# ============================================================================
# Tick Behavior
# ============================================================================


async def test_monitor_tick_with_no_vms() -> None:
    """Tick with no active VMs is a no-op (no crash)."""
    admission = _make_admission()
    vm_manager = _make_vm_manager_mock(active_vms={})

    monitor = ResourceMonitor(vm_manager, admission, interval_seconds=0.05)
    await monitor.start()
    await asyncio.sleep(0.15)  # Allow a few ticks
    await monitor.stop()

    # Should have called get_active_vms at least once
    assert vm_manager.get_active_vms.called


async def test_monitor_tick_with_active_vms() -> None:
    """Tick with active VMs reads stats and logs."""
    admission = _make_admission()

    # Create mock VM with cgroup_path
    mock_vm = MagicMock()
    mock_vm.vm_id = "test-vm-1"
    mock_vm.language = "python"
    mock_vm.cgroup_path = None  # No real cgroup
    mock_vm.process.pid = None

    vm_manager = _make_vm_manager_mock(active_vms={"test-vm-1": mock_vm})

    monitor = ResourceMonitor(vm_manager, admission, interval_seconds=0.05)
    await monitor.start()
    await asyncio.sleep(0.15)
    await monitor.stop()

    assert vm_manager.get_active_vms.call_count >= 1


# ============================================================================
# Error Resilience
# ============================================================================


async def test_monitor_survives_tick_exception() -> None:
    """Monitor keeps running even if a tick fails."""
    admission = _make_admission()
    vm_manager = _make_vm_manager_mock()
    # First call raises, second succeeds
    vm_manager.get_active_vms.side_effect = [RuntimeError("boom"), {}, {}]

    monitor = ResourceMonitor(vm_manager, admission, interval_seconds=0.05)
    await monitor.start()
    await asyncio.sleep(0.2)
    await monitor.stop()

    # Should have retried after the error
    assert vm_manager.get_active_vms.call_count >= 2

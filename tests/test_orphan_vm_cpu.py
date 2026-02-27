"""Tests for orphan VM CPU behavior.

These tests verify that orphan VMs (VMs where the host disconnected without
proper cleanup) consume minimal CPU instead of busy-looping at 100%.

The fix involves two components:
1. vm_manager.py: QemuVM.destroy() now properly terminates processes
2. guest-agent: Detects EPOLLHUP on virtio-serial disconnect and backs off

Background:
- When host disconnects from virtio-serial, kernel returns POLLHUP immediately
- Without backoff, guest-agent would busy-loop consuming 100% CPU
- With the fix, guest-agent uses exponential backoff (50ms -> 1s) allowing
  the CPU to enter WFI (Wait For Interrupt) idle state

These tests intentionally orphan VMs to verify the fix works under various
scenarios including:
- Simple orphan (no destroy called)
- Orphan with network (gvproxy)
- Multiple concurrent orphans
"""

import asyncio
from typing import TypedDict

import psutil
import pytest

from exec_sandbox.models import Language
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.resource_cleanup import cleanup_vm_processes
from exec_sandbox.vm_manager import VmManager
from tests.conftest import skip_unless_hwaccel


class ProcessInfo(TypedDict):
    """Process information from psutil."""

    status: str
    cpu: float


def get_process_info(proc: ProcessWrapper | None) -> ProcessInfo | None:
    """Get process status and CPU usage from ProcessWrapper (cross-platform).

    Uses the ProcessWrapper's internal psutil_proc for PID-safe monitoring.

    Returns:
        Dict with 'status' and 'cpu' keys, or None if process doesn't exist.
    """
    if proc is None or proc.psutil_proc is None:
        return None
    try:
        # cpu_percent needs a small interval for accurate reading
        cpu = proc.psutil_proc.cpu_percent(interval=0.1)
        status = proc.psutil_proc.status()
        return ProcessInfo(status=status, cpu=cpu)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def is_process_alive(proc: ProcessWrapper | None) -> bool:
    """Check if process is alive using ProcessWrapper's psutil_proc."""
    if proc is None or proc.psutil_proc is None:
        return False
    try:
        return proc.psutil_proc.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def is_process_idle(status: str) -> bool:
    """Check if process status indicates idle/sleeping (not busy-looping).

    Idle states (psutil cross-platform):
    - sleeping: Interruptible sleep (Linux S, macOS S)
    - idle: Idle (macOS specific)
    - stopped: Stopped by signal (Linux/macOS T)
    - disk-sleep: Uninterruptible disk sleep (Linux D)

    Running states (bad for orphan VMs):
    - running: Running or runnable (Linux/macOS R)
    """
    return status in (
        psutil.STATUS_SLEEPING,  # "sleeping"
        psutil.STATUS_IDLE,  # "idle" (macOS)
        psutil.STATUS_STOPPED,  # "stopped"
        psutil.STATUS_DISK_SLEEP,  # "disk-sleep"
    )


async def wait_for_process_idle(proc: ProcessWrapper, timeout: float = 10.0, threshold_cpu: float = 5.0) -> bool:
    """Wait for a process to settle into idle state.

    Args:
        proc: ProcessWrapper to monitor
        timeout: Maximum time to wait in seconds
        threshold_cpu: CPU percentage threshold for "idle"

    Returns:
        True if process is idle (sleeping with low CPU), False otherwise
    """
    start = asyncio.get_event_loop().time()
    samples: list[ProcessInfo] = []

    while asyncio.get_event_loop().time() - start < timeout:
        info = get_process_info(proc)
        if info is None:
            return False  # Process died

        samples.append(info)

        # Need at least 3 samples showing idle behavior
        if len(samples) >= 3:
            recent: list[ProcessInfo] = samples[-3:]
            all_idle = all(is_process_idle(s["status"]) for s in recent)
            all_low_cpu = all(s["cpu"] < threshold_cpu for s in recent)
            if all_idle and all_low_cpu:
                return True

        await asyncio.sleep(0.5)

    return False


async def kill_vm_processes(vm: QemuVM) -> None:
    """Kill VM QEMU and gvproxy processes for cleanup."""
    await cleanup_vm_processes(
        vm.process,
        vm.gvproxy_proc,
        vm.vm_id,
        qemu_term_timeout=1.0,
        qemu_kill_timeout=1.0,
        gvproxy_term_timeout=1.0,
        gvproxy_kill_timeout=1.0,
    )


@pytest.mark.asyncio
@skip_unless_hwaccel
class TestOrphanVmCpu:
    """Integration tests for orphan VM CPU behavior.

    Requires hwaccel: CPU idle measurement is meaningless under TCG — the
    TCG JIT thread always spins, so the orphan idle invariant cannot hold.
    """

    async def test_orphan_vm_stays_idle(self, vm_manager: VmManager) -> None:
        """Orphan VM (no destroy called) should use minimal CPU.

        This test creates a VM, then lets it go out of scope without
        calling destroy. The VM becomes orphaned when the host closes
        the virtio-serial connection. The guest-agent should detect
        POLLHUP and back off, keeping CPU usage minimal.
        """
        # Create VM
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test-orphan",
            task_id="idle-test",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        assert vm.process.pid is not None, "QEMU process should have a PID"

        try:
            # Let the host connection close (simulate orphan)
            # Close the channel but don't call destroy
            await vm.channel.close()

            # Wait a bit for guest-agent to detect disconnect
            await asyncio.sleep(2)

            # Monitor CPU for several seconds
            idle = await wait_for_process_idle(vm.process, timeout=10.0, threshold_cpu=5.0)

            # Get final state
            info = get_process_info(vm.process)
            assert info is not None, "QEMU process died unexpectedly"

            # Verify idle behavior
            assert idle, f"Orphan VM should be idle, got status={info['status']}, cpu={info['cpu']}%"
            assert is_process_idle(info["status"]), f"Expected idle status, got {info['status']}"
            assert info["cpu"] < 5.0, f"Expected <5% CPU, got {info['cpu']}%"

        finally:
            # Clean up - kill the orphan
            await kill_vm_processes(vm)

    async def test_orphan_vm_with_network_stays_idle(self, vm_manager: VmManager) -> None:
        """Orphan VM with network (gvproxy) should use minimal CPU.

        Tests that both QEMU and gvproxy processes stay idle when orphaned.
        """
        # Create VM with network
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test-orphan-net",
            task_id="idle-net-test",
            memory_mb=256,
            allow_network=True,
            allowed_domains=["example.com"],
        )

        assert vm.process.pid is not None, "QEMU process should have a PID"
        assert vm.gvproxy_proc is not None, "Network VM should have gvproxy"
        assert vm.gvproxy_proc.pid is not None, "gvproxy should have a PID"

        try:
            # Close host connection (simulate orphan)
            await vm.channel.close()
            await asyncio.sleep(2)

            # Check QEMU
            qemu_idle = await wait_for_process_idle(vm.process, timeout=10.0, threshold_cpu=5.0)
            qemu_info = get_process_info(vm.process)
            assert qemu_info is not None, "QEMU process died unexpectedly"
            assert qemu_idle, f"QEMU should be idle, got status={qemu_info['status']}, cpu={qemu_info['cpu']}%"

            # Check gvproxy
            gvproxy_info = get_process_info(vm.gvproxy_proc)
            assert gvproxy_info is not None, "gvproxy process died unexpectedly"
            assert gvproxy_info["cpu"] < 5.0, f"gvproxy should use <5% CPU, got {gvproxy_info['cpu']}%"

        finally:
            # Clean up
            await kill_vm_processes(vm)

    async def test_multiple_orphan_vms_stay_idle(self, vm_manager: VmManager) -> None:
        """Multiple orphan VMs should all stay idle.

        Creates several VMs concurrently, orphans them, and verifies
        all stay at low CPU usage.
        """
        num_vms = 3
        vms: list[QemuVM] = []

        try:
            # Create VMs concurrently
            create_tasks = [
                vm_manager.create_vm(
                    language=Language.PYTHON,
                    tenant_id="test-multi-orphan",
                    task_id=f"vm-{i}",
                    memory_mb=256,
                    allow_network=False,
                    allowed_domains=None,
                )
                for i in range(num_vms)
            ]
            vms = list(await asyncio.gather(*create_tasks))

            # Close all channels (orphan them)
            for vm in vms:
                await vm.channel.close()

            # Wait for guest-agents to detect disconnect
            await asyncio.sleep(3)

            # Verify all are idle
            # Threshold at 12% to tolerate psutil rounding on noisy CI runners
            for i, vm in enumerate(vms):
                assert vm.process.pid is not None, f"VM {i} should have a PID"
                info = get_process_info(vm.process)
                assert info is not None, f"VM {i} process died unexpectedly"
                assert is_process_idle(info["status"]), f"VM {i} should be idle, got {info['status']}"
                assert info["cpu"] < 12.0, f"VM {i} should use <12% CPU, got {info['cpu']}%"

        finally:
            # Clean up all VMs
            for vm in vms:
                await kill_vm_processes(vm)

    async def test_orphan_vm_cpu_over_time(self, vm_manager: VmManager) -> None:
        """Orphan VM CPU should remain low over extended period.

        This test monitors CPU usage over 10 seconds to ensure
        the backoff mechanism prevents CPU spikes.
        """
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test-orphan-time",
            task_id="extended-test",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        assert vm.process.pid is not None, "QEMU process should have a PID"

        try:
            # Orphan the VM
            await vm.channel.close()
            await asyncio.sleep(2)

            # Collect CPU samples over 10 seconds
            samples: list[float] = []
            for _ in range(10):
                info = get_process_info(vm.process)
                if info:
                    samples.append(info["cpu"])
                await asyncio.sleep(1)

            # Verify no sample exceeds threshold
            # Threshold at 12% to tolerate psutil rounding on noisy CI runners
            max_cpu = max(samples) if samples else 0.0
            avg_cpu = sum(samples) / len(samples) if samples else 0.0

            assert max_cpu < 12.0, f"Max CPU spike was {max_cpu}%, expected <12%"
            assert avg_cpu < 3.0, f"Average CPU was {avg_cpu:.1f}%, expected <3%"

        finally:
            await kill_vm_processes(vm)


@pytest.mark.asyncio
@pytest.mark.slow
class TestDestroyProcessCleanup:
    """Tests for QemuVM.destroy() process termination.

    Slow under TCG: process kill correctness tests — OS-level, not
    speed-dependent, but VM boot overhead makes them too slow for
    the default suite.
    """

    async def test_destroy_kills_qemu_process(self, vm_manager: VmManager) -> None:
        """QemuVM.destroy() should terminate the QEMU process."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test-destroy",
            task_id="kill-test",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        assert vm.process.pid is not None, "QEMU process should have a PID"

        # Verify process is running
        assert is_process_alive(vm.process), "QEMU should be running before destroy"

        # Call destroy (should kill process)
        await vm.destroy()

        # Wait a moment for process to die
        await asyncio.sleep(1)

        # Verify process is dead
        assert not is_process_alive(vm.process), "QEMU process should be dead after destroy"

    async def test_destroy_kills_gvproxy_process(self, vm_manager: VmManager) -> None:
        """QemuVM.destroy() should terminate the gvproxy process."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test-destroy-net",
            task_id="kill-gvproxy-test",
            memory_mb=256,
            allow_network=True,
            allowed_domains=["example.com"],
        )

        assert vm.process.pid is not None, "QEMU process should have a PID"
        assert vm.gvproxy_proc is not None, "Network VM should have gvproxy"
        assert vm.gvproxy_proc.pid is not None, "gvproxy should have a PID"

        # Verify both are running
        assert is_process_alive(vm.process), "QEMU should be running"
        assert is_process_alive(vm.gvproxy_proc), "gvproxy should be running"

        # Call destroy
        await vm.destroy()
        await asyncio.sleep(1)

        # Verify both are dead
        assert not is_process_alive(vm.process), "QEMU should be dead"
        assert not is_process_alive(vm.gvproxy_proc), "gvproxy should be dead"

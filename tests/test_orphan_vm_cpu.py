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

import psutil
import pytest

from exec_sandbox.models import Language
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.resource_cleanup import cleanup_vm_processes
from exec_sandbox.vm_manager import VmManager
from tests.conftest import skip_unless_hwaccel
from tests.cpu_helpers import (
    assert_cpu_idle,
    collect_cpu_samples,
    collect_cpu_samples_bulk,
)


def is_process_alive(proc: ProcessWrapper | None) -> bool:
    """Check if process is alive using ProcessWrapper's psutil_proc."""
    if proc is None or proc.psutil_proc is None:
        return False
    try:
        return proc.psutil_proc.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
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
            # Close the channel but don't call destroy (simulate orphan)
            await vm.channel.close()

            # Wait for guest-agent to detect disconnect and back off
            await asyncio.sleep(2)

            # Percentile-based CPU measurement (absorbs CI noise)
            samples = await collect_cpu_samples(vm.process, n_samples=10)
            assert_cpu_idle(samples, label="orphan QEMU")

        finally:
            await kill_vm_processes(vm)

    async def test_orphan_vm_with_network_stays_idle(self, vm_manager: VmManager) -> None:
        """Orphan VM with network (gvproxy) should use minimal CPU.

        Tests that both QEMU and gvproxy processes stay idle when orphaned.
        """
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

            # Percentile-based CPU measurement for both processes
            qemu_samples = await collect_cpu_samples(vm.process, n_samples=10)
            assert_cpu_idle(qemu_samples, label="orphan QEMU")

            gvproxy_samples = await collect_cpu_samples(vm.gvproxy_proc, n_samples=10)
            assert_cpu_idle(gvproxy_samples, label="orphan gvproxy")

        finally:
            await kill_vm_processes(vm)

    async def test_multiple_orphan_vms_stay_idle(self, vm_manager: VmManager) -> None:
        """Multiple orphan VMs should all stay idle.

        Creates several VMs concurrently, orphans them, and verifies
        all stay at low CPU usage using bulk percentile measurement.
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

            # Bulk percentile-based CPU measurement (same time window)
            procs = [vm.process for vm in vms]
            all_samples = await collect_cpu_samples_bulk(procs, n_samples=10)

            for i, samples in enumerate(all_samples):
                assert_cpu_idle(samples, label=f"orphan VM {i}")

        finally:
            for vm in vms:
                await kill_vm_processes(vm)

    async def test_orphan_vm_cpu_over_time(self, vm_manager: VmManager) -> None:
        """Orphan VM CPU should remain low over extended period.

        This test monitors CPU usage over ~15 seconds to ensure
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

            # Extended percentile-based CPU measurement
            samples = await collect_cpu_samples(vm.process, n_samples=15)
            assert_cpu_idle(samples, label="orphan QEMU over time")

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

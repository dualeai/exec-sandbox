"""Tests for orphan VM CPU behavior.

These tests verify that orphan VMs (VMs where the host disconnected without
proper cleanup) consume minimal CPU instead of busy-looping at 100%.

Two components keep it idle:
1. vm_manager.py: VmManager.destroy_vm() ends the QEMU and gvproxy processes
2. guest-agent: Detects EPOLLHUP on virtio-serial disconnect and backs off

Background:
- When host disconnects from virtio-serial, kernel returns POLLHUP immediately
- Without backoff, guest-agent would busy-loop consuming 100% CPU
- The guest agent backs off exponentially (50ms -> 1s), so the CPU can enter
  WFI (Wait For Interrupt) idle state

These tests orphan VMs on purpose, in these cases:
- Single orphan over ~15 s (no destroy called)
- Orphan with network (gvproxy)
- Multiple concurrent orphans
"""

import asyncio
from typing import TYPE_CHECKING

import psutil
import pytest

from exec_sandbox import constants
from exec_sandbox.models import Language
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import skip_unless_hwaccel
from tests.cpu_helpers import (
    assert_cpu_idle,
    collect_cpu_samples,
    collect_cpu_samples_bulk,
)

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuVM


def is_process_alive(proc: ProcessWrapper | None) -> bool:
    """Check if process is alive using ProcessWrapper's psutil_proc."""
    if proc is None or proc.psutil_proc is None:
        return False
    try:
        return proc.psutil_proc.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


@pytest.mark.asyncio
@skip_unless_hwaccel
class TestOrphanVmCpu:
    """Integration tests for orphan VM CPU behavior.

    Requires hwaccel: CPU idle measurement is meaningless under TCG — the
    TCG JIT thread always spins, so the orphan idle invariant cannot hold.
    """

    async def test_orphan_vm_with_network_stays_idle(self, vm_manager: VmManager) -> None:
        """Orphan VM with network (gvproxy) should use minimal CPU.

        Tests that both QEMU and gvproxy processes stay idle when orphaned.
        """
        vm = await vm_manager.create_vm(
            VmSpec(language=Language.PYTHON, memory_mb=256, allow_network=True, allowed_domains=("example.com",)),
            tenant_id="test-orphan-net",
            task_id="idle-net-test",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
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
            await vm_manager.destroy_vm(vm)

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
                    VmSpec(language=Language.PYTHON, memory_mb=256, allow_network=False, allowed_domains=None),
                    tenant_id="test-multi-orphan",
                    task_id=f"vm-{i}",
                    cache_leases=(),
                    direct_write_target=None,
                    expose_ports=None,
                    on_boot_log=None,
                    snapshot_drive=None,
                    reservation=None,
                    retry_profile=constants.RETRY_USER_FACING,
                    mem_path=None,
                    mem_path_share=False,
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
                await vm_manager.destroy_vm(vm)

    async def test_orphan_vm_cpu_over_time(self, vm_manager: VmManager) -> None:
        """Orphan VM CPU should remain low over extended period.

        This test monitors CPU usage over ~15 seconds to ensure
        the backoff mechanism prevents CPU spikes.
        """
        vm = await vm_manager.create_vm(
            VmSpec(language=Language.PYTHON, memory_mb=256, allow_network=False, allowed_domains=None),
            tenant_id="test-orphan-time",
            task_id="extended-test",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
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
            await vm_manager.destroy_vm(vm)


@pytest.mark.asyncio
@pytest.mark.slow
class TestDestroyProcessCleanup:
    """Tests for VmManager.destroy_vm() process termination.

    Slow under TCG: process kill correctness tests — OS-level, not
    speed-dependent, but VM boot overhead makes them too slow for
    the default suite.
    """

    async def test_destroy_kills_qemu_process(self, vm_manager: VmManager) -> None:
        """VmManager.destroy_vm() should terminate the QEMU process."""
        vm = await vm_manager.create_vm(
            VmSpec(language=Language.PYTHON, memory_mb=256, allow_network=False, allowed_domains=None),
            tenant_id="test-destroy",
            task_id="kill-test",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
        )

        assert vm.process.pid is not None, "QEMU process should have a PID"

        # Verify process is running
        assert is_process_alive(vm.process), "QEMU should be running before destroy"

        # destroy_vm() returns once the process is gone: no wait after it
        assert await vm_manager.destroy_vm(vm) is True
        assert not is_process_alive(vm.process), "QEMU process should be dead after destroy"

    async def test_destroy_kills_gvproxy_process(self, vm_manager: VmManager) -> None:
        """VmManager.destroy_vm() should terminate the gvproxy process."""
        vm = await vm_manager.create_vm(
            VmSpec(language=Language.PYTHON, memory_mb=256, allow_network=True, allowed_domains=("example.com",)),
            tenant_id="test-destroy-net",
            task_id="kill-gvproxy-test",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
        )

        assert vm.process.pid is not None, "QEMU process should have a PID"
        assert vm.gvproxy_proc is not None, "Network VM should have gvproxy"
        assert vm.gvproxy_proc.pid is not None, "gvproxy should have a PID"

        # Verify both are running
        assert is_process_alive(vm.process), "QEMU should be running"
        assert is_process_alive(vm.gvproxy_proc), "gvproxy should be running"

        # destroy_vm() returns once both processes are gone: no wait after it
        assert await vm_manager.destroy_vm(vm) is True
        assert not is_process_alive(vm.process), "QEMU should be dead"
        assert not is_process_alive(vm.gvproxy_proc), "gvproxy should be dead"

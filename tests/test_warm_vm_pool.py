"""Tests for WarmVMPool.

Unit tests: Pool data structures, config handling, healthcheck pure functions.
Integration tests: Real VM pool operations (requires QEMU + images).
"""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.models import Language

from .conftest import skip_unless_hwaccel
from .cpu_helpers import (
    CPU_CONSECUTIVE_SPIKE_THRESHOLD,
    CPU_MAX_CONSECUTIVE_SPIKES,
    CPU_SAMPLES_SUSTAINED,
    assert_cpu_idle,
    collect_cpu_samples,
    collect_cpu_samples_bulk,
)

# ============================================================================
# Unit Tests - No QEMU needed
# ============================================================================


class TestWarmVMPoolConfig:
    """Tests for WarmVMPool configuration."""

    def test_explicit_warm_pool_size(self, unit_test_vm_manager) -> None:
        """When warm_pool_size > 0, it directly sets pool size."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=5)
        pool = WarmVMPool(unit_test_vm_manager, config)
        assert pool.pool_size_per_language == 5

        config = SchedulerConfig(warm_pool_size=50)
        pool = WarmVMPool(unit_test_vm_manager, config)
        assert pool.pool_size_per_language == 50

    def test_warm_pool_has_all_languages(self, unit_test_vm_manager) -> None:
        """Warm pool creates a pool for every Language."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)
        assert set(pool.pools.keys()) == set(Language)


class TestLanguageEnum:
    """Tests for Language enum."""

    def test_language_values(self) -> None:
        """Language enum has expected values."""
        assert Language.PYTHON.value == "python"
        assert Language.JAVASCRIPT.value == "javascript"

    def test_language_from_string(self) -> None:
        """Language can be created from string."""
        assert Language("python") == Language.PYTHON
        assert Language("javascript") == Language.JAVASCRIPT


# ============================================================================
# Unit Tests - Healthcheck Pure Functions (No QEMU, No Mocks)
# ============================================================================


class TestWarmReplUnit:
    """Unit tests for _warm_repl - mocked channel, no QEMU."""

    async def test_warm_repl_ok(self, unit_test_vm_manager) -> None:
        """_warm_repl succeeds when guest returns ok ack."""
        from exec_sandbox.guest_agent_protocol import WarmReplAckMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = AsyncMock()
        vm.vm_id = "test-vm"
        vm.channel.send_request = AsyncMock(return_value=WarmReplAckMessage(language="python", status="ok"))

        # Should complete without error
        await pool._warm_repl(vm, Language.PYTHON)
        vm.channel.send_request.assert_called_once()

    async def test_warm_repl_error_response_non_fatal(self, unit_test_vm_manager) -> None:
        """_warm_repl logs warning but doesn't raise on error response."""
        from exec_sandbox.guest_agent_protocol import WarmReplAckMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = AsyncMock()
        vm.vm_id = "test-vm"
        vm.channel.send_request = AsyncMock(
            return_value=WarmReplAckMessage(language="python", status="error", message="spawn failed")
        )

        # Should NOT raise
        await pool._warm_repl(vm, Language.PYTHON)

    async def test_warm_repl_timeout_non_fatal(self, unit_test_vm_manager) -> None:
        """_warm_repl swallows timeout exceptions (graceful degradation)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = AsyncMock()
        vm.vm_id = "test-vm"
        vm.channel.send_request = AsyncMock(side_effect=TimeoutError("timed out"))

        # Should NOT raise
        await pool._warm_repl(vm, Language.PYTHON)

    async def test_warm_repl_connection_error_non_fatal(self, unit_test_vm_manager) -> None:
        """_warm_repl swallows connection errors (graceful degradation)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = AsyncMock()
        vm.vm_id = "test-vm"
        vm.channel.send_request = AsyncMock(side_effect=ConnectionError("broken pipe"))

        # Should NOT raise
        await pool._warm_repl(vm, Language.PYTHON)

    async def test_warm_repl_unexpected_response_type(self, unit_test_vm_manager) -> None:
        """_warm_repl handles unexpected response type (logs warning, doesn't crash)."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = AsyncMock()
        vm.vm_id = "test-vm"
        # Return a PongMessage instead of WarmReplAckMessage
        vm.channel.send_request = AsyncMock(return_value=PongMessage(version="1.0"))

        # Should NOT raise — logs warning about non-ok response
        await pool._warm_repl(vm, Language.PYTHON)

    async def test_warm_repl_passes_correct_language(self, unit_test_vm_manager) -> None:
        """_warm_repl sends the correct language in request."""
        from exec_sandbox.guest_agent_protocol import WarmReplAckMessage, WarmReplRequest
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = AsyncMock()
        vm.vm_id = "test-vm"
        vm.channel.send_request = AsyncMock(return_value=WarmReplAckMessage(language="javascript", status="ok"))

        await pool._warm_repl(vm, Language.JAVASCRIPT)

        # Verify the request sent had language=javascript
        call_args = vm.channel.send_request.call_args
        request = call_args[0][0]
        assert isinstance(request, WarmReplRequest)
        assert request.language == Language.JAVASCRIPT


class TestDrainPoolForCheck:
    """Tests for _drain_pool_for_check - pure queue draining logic."""

    async def test_drain_empty_pool(self, unit_test_vm_manager) -> None:
        """Draining empty pool returns empty list."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        # Create pool with minimal config (no VMs booted)
        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Pool is empty (no startup called)
        result = pool._drain_pool_for_check(
            pool.pools[Language.PYTHON],
            pool_size=0,
            language=Language.PYTHON,
        )

        assert result == []

    async def test_drain_respects_pool_size_parameter(self, unit_test_vm_manager) -> None:
        """Drain only removes up to pool_size items."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Manually add items to test drain logic
        # Using simple objects since we're testing queue behavior, not VM behavior
        test_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        await test_queue.put("vm1")
        await test_queue.put("vm2")
        await test_queue.put("vm3")

        # Drain only 2 items even though 3 exist
        result = pool._drain_pool_for_check(
            test_queue,  # type: ignore[arg-type]
            pool_size=2,
            language=Language.PYTHON,
        )

        assert len(result) == 2
        assert result == ["vm1", "vm2"]
        assert test_queue.qsize() == 1  # One item remains

    async def test_drain_more_than_exists(self, unit_test_vm_manager) -> None:
        """Drain handles request for more items than queue contains."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        test_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        await test_queue.put("vm1")
        await test_queue.put("vm2")

        # Request 5 items but only 2 exist
        result = pool._drain_pool_for_check(
            test_queue,  # type: ignore[arg-type]
            pool_size=5,
            language=Language.PYTHON,
        )

        # Should only get what exists, not crash
        assert len(result) == 2
        assert result == ["vm1", "vm2"]
        assert test_queue.qsize() == 0

    async def test_drain_exact_size(self, unit_test_vm_manager) -> None:
        """Drain exactly the number of items in queue (boundary)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        test_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        await test_queue.put("vm1")
        await test_queue.put("vm2")
        await test_queue.put("vm3")

        # Request exactly 3 items
        result = pool._drain_pool_for_check(
            test_queue,  # type: ignore[arg-type]
            pool_size=3,
            language=Language.PYTHON,
        )

        assert len(result) == 3
        assert result == ["vm1", "vm2", "vm3"]
        assert test_queue.qsize() == 0


# ============================================================================
# Unit Tests - Health Check Pool Empty Case (No QEMU, No Mocks)
# ============================================================================


class TestHealthCheckPoolUnit:
    """Unit tests for _health_check_pool edge cases."""

    async def test_health_check_empty_pool_returns_early(self, unit_test_vm_manager) -> None:
        """Health check on empty pool returns immediately without error."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Pool is empty (no startup called)
        # Should return early without error
        await pool._health_check_pool(
            Language.PYTHON,
            pool.pools[Language.PYTHON],
        )

        # Pool should still be empty
        assert pool.pools[Language.PYTHON].qsize() == 0


# ============================================================================
# Integration Tests - Require QEMU + Images
# ============================================================================


@pytest.mark.slow
class TestWarmVMPoolIntegration:
    """Integration tests for WarmVMPool with real QEMU VMs.

    Slow under TCG: pool start/stop/checkout correctness tests — work under
    TCG but VM boot overhead (~5-8x) makes them too slow for the default suite.
    """

    async def test_pool_start_stop(self, vm_manager) -> None:
        """Pool starts and stops cleanly."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        # Pools should be populated
        assert pool.pools[Language.PYTHON].qsize() > 0

        await pool.stop()

        # Pools should be empty
        assert pool.pools[Language.PYTHON].qsize() == 0
        assert pool.pools[Language.JAVASCRIPT].qsize() == 0
        assert pool.pools[Language.RAW].qsize() == 0

    async def test_get_vm_from_pool(self, vm_manager) -> None:
        """Get VM from warm pool."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        try:
            # Get VM from pool (should be instant)
            vm = await pool.get_vm(Language.PYTHON, packages=[])

            assert vm is not None
            assert vm.vm_id is not None

            # Destroy VM after use
            await vm_manager.destroy_vm(vm)

        finally:
            await pool.stop()

    async def test_warm_pool_vm_has_prewarmed_repl(self, vm_manager) -> None:
        """VM from warm pool has pre-warmed REPL (fast first execution)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)
        await pool.start()

        try:
            vm = await pool.get_vm(Language.PYTHON, packages=[])
            assert vm is not None

            # First execution should be fast — REPL was pre-warmed
            result = await vm.execute(
                code="print('warm')",
                timeout_seconds=30,
            )
            assert result.exit_code == 0
            assert "warm" in result.stdout

            await vm_manager.destroy_vm(vm)
        finally:
            await pool.stop()

    async def test_get_vm_with_packages_returns_none(self, vm_manager) -> None:
        """Get VM with packages returns None (not eligible for warm pool)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        try:
            # Get VM with packages - should return None
            vm = await pool.get_vm(Language.PYTHON, packages=["pandas==2.0.0"])
            assert vm is None

        finally:
            await pool.stop()


# ============================================================================
# Integration Tests - Healthcheck Workflow (Require QEMU + Images)
# ============================================================================


@pytest.mark.slow
class TestHealthcheckIntegration:
    """Integration tests for healthcheck with real QEMU VMs.

    Slow under TCG: health check correctness tests with no timing
    assertions — work under TCG but boot overhead makes them too slow
    for the default suite.
    """

    async def test_check_vm_health_healthy_vm(self, vm_manager) -> None:
        """_check_vm_health returns True for healthy VM."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        try:
            # Get a VM from pool
            vm = await pool.get_vm(Language.PYTHON, packages=[])
            assert vm is not None

            # Health check should pass for a freshly booted VM
            is_healthy = await pool._check_vm_health(vm)
            assert is_healthy is True

            await vm_manager.destroy_vm(vm)

        finally:
            await pool.stop()

    async def test_health_check_pool_preserves_healthy_vms(self, vm_manager) -> None:
        """_health_check_pool keeps healthy VMs in pool."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        try:
            # Wait for VMs to stabilize after startup balloon inflation
            # On slow CI, VMs may need time to adjust to reduced memory
            await asyncio.sleep(0.5)

            initial_size = pool.pools[Language.PYTHON].qsize()
            assert initial_size > 0

            # Run health check on Python pool
            await pool._health_check_pool(
                Language.PYTHON,
                pool.pools[Language.PYTHON],
            )

            # All healthy VMs should be preserved
            final_size = pool.pools[Language.PYTHON].qsize()
            assert final_size == initial_size

        finally:
            await pool.stop()

    async def test_drain_pool_restores_vms_after_health_check(self, vm_manager) -> None:
        """VMs drained for health check are restored to pool."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        try:
            python_pool = pool.pools[Language.PYTHON]
            initial_size = python_pool.qsize()

            # Drain all VMs
            vms = pool._drain_pool_for_check(
                python_pool,
                pool_size=initial_size,
                language=Language.PYTHON,
            )

            assert python_pool.qsize() == 0
            assert len(vms) == initial_size

            # Check and restore each VM immediately (new architecture)
            results = await asyncio.gather(
                *[pool._check_and_restore_vm(vm, python_pool, Language.PYTHON) for vm in vms],
                return_exceptions=True,
            )

            # Count results (True = healthy, False = unhealthy)
            healthy_count = sum(1 for r in results if r is True)
            unhealthy_count = len(results) - healthy_count

            assert healthy_count == initial_size
            assert unhealthy_count == 0
            assert python_pool.qsize() == initial_size

        finally:
            await pool.stop()

    async def test_health_check_loop_stops_on_stop(self, vm_manager) -> None:
        """Health check loop exits cleanly when stop is signaled."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        # Health task should be running
        assert pool._health_task is not None
        assert not pool._health_task.done()

        # stop() should stop health task
        await pool.stop()

        assert pool._health_task.done()
        assert pool._shutdown_event.is_set()

    # -------------------------------------------------------------------------
    # Edge Cases - Real VM Tests (NO MOCKS)
    # -------------------------------------------------------------------------

    async def test_killed_vm_detected_as_unhealthy(self, vm_manager) -> None:
        """Health check detects killed VM process as unhealthy.

        This is a critical test - verifies that when QEMU process dies,
        the health check correctly identifies the VM as unhealthy.
        """
        import signal

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        try:
            # Get a VM from pool
            vm = await pool.get_vm(Language.PYTHON, packages=[])
            assert vm is not None

            # Verify it's healthy first
            is_healthy = await pool._check_vm_health(vm)
            assert is_healthy is True

            # Kill the QEMU process (simulate crash)
            assert vm.process.pid is not None
            import os

            os.kill(vm.process.pid, signal.SIGKILL)

            # Wait for process to die
            await asyncio.sleep(0.1)

            # Health check should now detect unhealthy
            is_healthy = await pool._check_vm_health(vm)
            assert is_healthy is False

        finally:
            await pool.stop()

    async def test_mixed_pool_healthy_and_killed_vms(self, vm_manager) -> None:
        """Health check correctly handles mix of healthy and killed VMs.

        Tests the real-world scenario where some VMs in pool have crashed
        while others are still healthy. Verifies selective detection.
        """
        import signal

        from exec_sandbox.warm_vm_pool import WarmVMPool

        # Create 2 VMs manually (warm pool only creates 1 per language with max_concurrent=4)
        vm1 = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="mixed-test-1",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )
        vm2 = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="mixed-test-2",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        try:
            # Kill first VM
            killed_vm_id = vm1.vm_id
            assert vm1.process.pid is not None
            import os

            os.kill(vm1.process.pid, signal.SIGKILL)
            await asyncio.sleep(0.1)

            # Check both VMs
            result1 = await pool._check_vm_health(vm1)
            result2 = await pool._check_vm_health(vm2)

            # vm1 should be unhealthy (killed), vm2 should be healthy
            assert result1 is False, "Killed VM should be unhealthy"
            assert result2 is True, "Live VM should be healthy"

        finally:
            # Clean up
            with contextlib.suppress(Exception):
                await vm_manager.destroy_vm(vm1)
            with contextlib.suppress(Exception):
                await vm_manager.destroy_vm(vm2)

    async def test_health_check_pool_removes_killed_vm(self, vm_manager) -> None:
        """Full _health_check_pool correctly removes killed VM from pool.

        Tests the complete health check flow, not just individual VM checks.
        Uses _check_and_restore_vm to verify killed VM is not restored.
        """
        import signal

        from exec_sandbox.warm_vm_pool import WarmVMPool

        # Create 2 VMs manually
        vm1 = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="pool-test-1",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )
        vm2 = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="pool-test-2",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)
        python_pool = pool.pools[Language.PYTHON]

        try:
            # Kill first VM
            killed_vm_id = vm1.vm_id
            assert vm1.process.pid is not None
            import os

            os.kill(vm1.process.pid, signal.SIGKILL)
            await asyncio.sleep(0.1)

            # Run check and restore on both VMs
            result1 = await pool._check_and_restore_vm(vm1, python_pool, Language.PYTHON)
            result2 = await pool._check_and_restore_vm(vm2, python_pool, Language.PYTHON)

            # vm1 should fail (not restored), vm2 should succeed (restored)
            assert result1 is False, "Killed VM should not be restored"
            assert result2 is True, "Healthy VM should be restored"

            # Pool should only contain vm2
            assert python_pool.qsize() == 1, f"Pool should have 1 VM, got {python_pool.qsize()}"

            # Get the VM and verify it's vm2
            restored_vm = python_pool.get_nowait()
            assert restored_vm.vm_id == vm2.vm_id, f"Expected {vm2.vm_id}, got {restored_vm.vm_id}"
            assert restored_vm.vm_id != killed_vm_id, "Killed VM should not be in pool"

        finally:
            # Clean up any remaining VMs
            with contextlib.suppress(Exception):
                await vm_manager.destroy_vm(vm1)
            with contextlib.suppress(Exception):
                await vm_manager.destroy_vm(vm2)

    async def test_frozen_vm_detected_as_unhealthy(self, vm_manager) -> None:
        """Health check detects frozen VM (SIGSTOP) as unhealthy via timeout.

        SIGSTOP freezes the QEMU process, making it unresponsive.
        The health check should timeout and mark it unhealthy.
        """
        import os
        import signal

        import psutil

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        try:
            # Get a VM from pool
            vm = await pool.get_vm(Language.PYTHON, packages=[])
            assert vm is not None

            # Wait for VM to stabilize after balloon deflation
            # On slow CI, the guest needs time to reclaim memory after deflate
            await asyncio.sleep(0.5)

            # Verify healthy first (with retries for slow CI where balloon
            # operations may leave the guest temporarily slow)
            is_healthy = False
            for _ in range(3):
                is_healthy = await pool._check_vm_health(vm)
                if is_healthy:
                    break
                await asyncio.sleep(0.3)
            assert is_healthy is True, "VM should be healthy before SIGSTOP"

            # Freeze the QEMU process (simulate hang)
            assert vm.process.pid is not None
            os.kill(vm.process.pid, signal.SIGSTOP)

            # Wait for process to actually stop - SIGSTOP is async, os.kill()
            # returns before the kernel fully stops the process. Without this,
            # there's a race where QEMU can respond to the health check ping
            # before being frozen.
            proc = psutil.Process(vm.process.pid)
            for _ in range(100):  # 1s max
                if proc.status() == psutil.STATUS_STOPPED:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail(f"QEMU process did not stop within 1s (status: {proc.status()})")

            try:
                # Health check should timeout and return unhealthy
                # Uses retry with backoff, so give it time
                is_healthy = await pool._check_vm_health(vm)
                assert is_healthy is False
            finally:
                # Unfreeze so cleanup can proceed
                os.kill(vm.process.pid, signal.SIGCONT)
                await asyncio.sleep(0.1)

            # Clean up
            await vm_manager.destroy_vm(vm)

        finally:
            await pool.stop()

    async def test_vm_killed_during_health_check(self, vm_manager) -> None:
        """Health check handles VM dying mid-check gracefully.

        Tests that the health check doesn't crash when VM dies during the check.
        The result may be True (if check completed before kill) or False (if kill
        happened first) - the key assertion is no crash/exception.
        """
        import signal

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        try:
            vm = await pool.get_vm(Language.PYTHON, packages=[])
            assert vm is not None

            # Kill VM immediately (no delay) to maximize chance of race
            async def kill_vm_now():
                if vm.process.pid:
                    import os

                    os.kill(vm.process.pid, signal.SIGKILL)

            # Start health check and kill concurrently
            health_task = asyncio.create_task(pool._check_vm_health(vm))
            kill_task = asyncio.create_task(kill_vm_now())

            # Both should complete without exception
            results = await asyncio.gather(health_task, kill_task, return_exceptions=True)

            # Health check should return a boolean (True or False), not crash
            health_result = results[0]
            assert isinstance(health_result, bool), f"Expected bool, got {type(health_result)}: {health_result}"

        finally:
            await pool.stop()

    async def test_multiple_consecutive_health_checks_after_kill(self, vm_manager) -> None:
        """Multiple health checks on killed VM all return False.

        Verifies consistent behavior across repeated checks on dead VM.
        """
        import signal

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        try:
            vm = await pool.get_vm(Language.PYTHON, packages=[])
            assert vm is not None

            # Kill VM
            assert vm.process.pid is not None
            import os

            os.kill(vm.process.pid, signal.SIGKILL)
            await asyncio.sleep(0.1)

            # Multiple health checks should all return False
            for i in range(3):
                is_healthy = await pool._check_vm_health(vm)
                assert is_healthy is False, f"Check {i + 1} should return False"

        finally:
            await pool.stop()

    async def test_check_and_restore_only_restores_healthy(self, vm_manager) -> None:
        """_check_and_restore_vm only puts healthy VMs back in pool.

        Verifies the new immediate-restore architecture works correctly.
        """
        import signal

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)

        await pool.start()

        try:
            python_pool = pool.pools[Language.PYTHON]

            # Get VM from pool
            vm = await python_pool.get()
            assert python_pool.qsize() == 0  # Pool now empty

            # Kill VM
            assert vm.process.pid is not None
            import os

            os.kill(vm.process.pid, signal.SIGKILL)
            await asyncio.sleep(0.1)

            # _check_and_restore_vm should NOT put killed VM back
            result = await pool._check_and_restore_vm(vm, python_pool, Language.PYTHON)

            assert result is False  # Unhealthy
            assert python_pool.qsize() == 0  # VM NOT restored to pool

        finally:
            await pool.stop()


# ============================================================================
# Integration Tests - Warm Pool Idle CPU (Require QEMU + Images)
# ============================================================================

SETTLE_SLEEP_S: float = 3.0
"""Post-start settling time (balloon inflate + REPL warmup)."""

WAKEUPS_MAX_PER_SECOND: float = 50.0
"""Max idle context switches/sec. Observed baseline: ~19-21/sec on macOS HVF
(LAPIC timer + balloon free-page-reporting + QEMU event loop polling).
Linux KVM expected similar (~15-25/sec). Threshold at ~2.5x baseline catches
spin-loops while absorbing CI noise and health-check bursts."""


@skip_unless_hwaccel
class TestWarmPoolIdleCpu:
    """Integration tests verifying warm pool VMs consume zero CPU when idle.

    Requires hwaccel: CPU idle measurement is meaningless under TCG — the
    TCG JIT thread always spins, so the "no execution = no CPU" invariant
    cannot hold.

    Measurement: sample cpu_percent every 1s over 10-15s.
    Assertions: median < 2%, p95 < 10%.
    """

    @pytest.mark.parametrize(
        "lang",
        [
            pytest.param(Language.PYTHON, id="python"),
            pytest.param(Language.JAVASCRIPT, id="javascript"),
            pytest.param(Language.RAW, id="raw"),
        ],
    )
    async def test_idle_zero_cpu(self, vm_manager, lang: Language) -> None:
        """Each language VM idles at near-zero CPU and low wakeup rate."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)
        await pool.start()

        try:
            await asyncio.sleep(SETTLE_SLEEP_S)

            # Get VM directly from internal queue (no balloon deflation, no replenishment)
            vm = pool.pools[lang].get_nowait()

            # Snapshot context switches before CPU sampling window
            assert vm.process.psutil_proc is not None
            cs_before = vm.process.psutil_proc.num_ctx_switches()
            t_before = time.monotonic()

            samples = await collect_cpu_samples(vm.process)

            # Snapshot after — measures wakeups over the same window as CPU
            cs_after = vm.process.psutil_proc.num_ctx_switches()
            t_after = time.monotonic()

            assert_cpu_idle(samples, label=lang.value)

            delta_cs = (cs_after.voluntary + cs_after.involuntary) - (cs_before.voluntary + cs_before.involuntary)
            wakeups_per_sec = delta_cs / (t_after - t_before)
            assert wakeups_per_sec < WAKEUPS_MAX_PER_SECOND, (
                f"{lang.value} idle wakeups/sec {wakeups_per_sec:.1f} >= {WAKEUPS_MAX_PER_SECOND}"
            )

            pool.pools[lang].put_nowait(vm)
        finally:
            await pool.stop()

    async def test_idle_after_health_check_ping(self, vm_manager) -> None:
        """CPU settles back to ~0% after health check wakes guest agent."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)
        await pool.start()

        try:
            await asyncio.sleep(SETTLE_SLEEP_S)

            vm = pool.pools[Language.PYTHON].get_nowait()

            # Manually trigger health check (connect + ping + pong wakes guest)
            healthy = await pool._check_vm_health(vm)
            assert healthy, "VM should be healthy"

            # CPU should settle back to idle after the ping
            samples = await collect_cpu_samples(vm.process, n_samples=10)
            assert_cpu_idle(samples, label="post-healthcheck")

            pool.pools[Language.PYTHON].put_nowait(vm)
        finally:
            await pool.stop()

    @pytest.mark.slow  # pool_size=2 may not fully boot within SETTLE_SLEEP_S on loaded CI runners
    async def test_multiple_vms_idle_no_interference(self, vm_manager) -> None:
        """Multiple VMs don't interfere with each other's idle CPU (thundering herd)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=2)
        pool = WarmVMPool(vm_manager, config)
        await pool.start()

        vms = []
        try:
            await asyncio.sleep(SETTLE_SLEEP_S)

            # Get all VMs from all language pools (2 each = 6 total)
            for lang in Language:
                for _ in range(pool.pool_size_per_language):
                    try:
                        vm = pool.pools[lang].get_nowait()
                        vms.append((lang, vm))
                    except asyncio.QueueEmpty:
                        break

            assert len(vms) >= 2, f"Expected at least 2 VMs, got {len(vms)}"

            all_samples = await collect_cpu_samples_bulk(
                [vm.process for _, vm in vms],
            )

            for i, ((lang, _), samples) in enumerate(zip(vms, all_samples, strict=True)):
                assert_cpu_idle(samples, label=f"VM {i} ({lang.value})")

            # Put VMs back
            for lang, vm in vms:
                pool.pools[lang].put_nowait(vm)
        finally:
            await pool.stop()

    async def test_balloon_inflation_cpu_is_transient(self, vm_manager) -> None:
        """Balloon inflation CPU spike settles within bounded time."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)
        await pool.start()

        try:
            # Do NOT wait — start measuring immediately after start()
            vm = pool.pools[Language.PYTHON].get_nowait()

            # Balloon inflation happened during start(); p95 absorbs initial spike
            samples = await collect_cpu_samples(vm.process)
            assert_cpu_idle(samples, label="post-balloon")

            pool.pools[Language.PYTHON].put_nowait(vm)
        finally:
            await pool.stop()

    @pytest.mark.slow
    async def test_idle_cpu_sustained_over_time(self, vm_manager) -> None:
        """No CPU drift over 30s (spans ~3 health check cycles)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)
        await pool.start()

        try:
            await asyncio.sleep(SETTLE_SLEEP_S)

            vm = pool.pools[Language.PYTHON].get_nowait()

            samples = await collect_cpu_samples(vm.process, n_samples=CPU_SAMPLES_SUSTAINED)
            assert_cpu_idle(samples, label="sustained")

            # Additional check: no sustained busy period
            consecutive_high = 0
            max_consecutive_high = 0
            for cpu in samples:
                if cpu > CPU_CONSECUTIVE_SPIKE_THRESHOLD:
                    consecutive_high += 1
                    max_consecutive_high = max(max_consecutive_high, consecutive_high)
                else:
                    consecutive_high = 0

            assert max_consecutive_high < CPU_MAX_CONSECUTIVE_SPIKES, (
                f"Sustained busy period: {max_consecutive_high} consecutive samples "
                f">{CPU_CONSECUTIVE_SPIKE_THRESHOLD}% (sorted: {sorted(samples)})"
            )

            pool.pools[Language.PYTHON].put_nowait(vm)
        finally:
            await pool.stop()


# ============================================================================
# Integration Tests - CPU Idle After Execution
# ============================================================================


@pytest.mark.slow  # warm pool boot + CPU sampling requires quiet runner; flaky under contention
@skip_unless_hwaccel
class TestPostExecutionCpuIdle:
    """Integration tests verifying VMs return to idle CPU after code execution.

    Requires hwaccel: post-execution CPU idle invariant doesn't hold under
    TCG — the TCG JIT thread always spins, making idle measurement meaningless.

    Pattern: get warm VM → execute code → sleep(SETTLE) → sample CPU → assert idle.
    Run with -n 0 for accurate CPU measurement.
    """

    @pytest.mark.parametrize(
        "lang, code",
        [
            pytest.param(Language.PYTHON, "print('hello')", id="python"),
            pytest.param(Language.JAVASCRIPT, "console.log('hello')", id="javascript"),
            pytest.param(Language.RAW, "echo hello", id="raw"),
        ],
    )
    async def test_cpu_idle_after_trivial_execution(self, vm_manager, lang: Language, code: str) -> None:
        """CPU settles to idle after a trivial code execution."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)
        await pool.start()

        try:
            await asyncio.sleep(SETTLE_SLEEP_S)

            vm = await pool.get_vm(lang, packages=[])
            assert vm is not None, f"No warm VM available for {lang.value}"

            result = await vm.execute(code=code, timeout_seconds=30)
            assert result.exit_code == 0

            await asyncio.sleep(SETTLE_SLEEP_S)

            samples = await collect_cpu_samples(vm.process, n_samples=10)
            assert_cpu_idle(samples, label=f"post-exec-{lang.value}")
        finally:
            await pool.stop()

    async def test_cpu_idle_after_large_output(self, vm_manager) -> None:
        """CPU settles to idle after generating ~100KB of stdout."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)
        await pool.start()

        try:
            await asyncio.sleep(SETTLE_SLEEP_S)

            vm = await pool.get_vm(Language.PYTHON, packages=[])
            assert vm is not None, "No warm VM available for Python"

            # ~100KB of output
            result = await vm.execute(
                code="print('x' * 1000, end='\\n')\n" * 100,
                timeout_seconds=30,
            )
            assert result.exit_code == 0

            await asyncio.sleep(SETTLE_SLEEP_S)

            samples = await collect_cpu_samples(vm.process, n_samples=10)
            assert_cpu_idle(samples, label="post-large-output")
        finally:
            await pool.stop()

    async def test_cpu_idle_after_failed_execution(self, vm_manager) -> None:
        """CPU settles to idle even after a failed execution (non-zero exit)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)
        await pool.start()

        try:
            await asyncio.sleep(SETTLE_SLEEP_S)

            vm = await pool.get_vm(Language.PYTHON, packages=[])
            assert vm is not None, "No warm VM available for Python"

            result = await vm.execute(
                code="raise RuntimeError('intentional failure')",
                timeout_seconds=30,
            )
            assert result.exit_code != 0

            await asyncio.sleep(SETTLE_SLEEP_S)

            samples = await collect_cpu_samples(vm.process, n_samples=10)
            assert_cpu_idle(samples, label="post-failed-exec")
        finally:
            await pool.stop()

    async def test_cpu_idle_after_sequential_executions(self, vm_manager) -> None:
        """CPU settles to idle after 3 sequential executions on the same VM."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(vm_manager, config)
        await pool.start()

        try:
            await asyncio.sleep(SETTLE_SLEEP_S)

            vm = await pool.get_vm(Language.PYTHON, packages=[])
            assert vm is not None, "No warm VM available for Python"

            for i in range(3):
                result = await vm.execute(
                    code=f"print('iteration {i}')",
                    timeout_seconds=30,
                )
                assert result.exit_code == 0

            await asyncio.sleep(SETTLE_SLEEP_S)

            samples = await collect_cpu_samples(vm.process, n_samples=10)
            assert_cpu_idle(samples, label="post-sequential-exec")
        finally:
            await pool.stop()


# ============================================================================
# Unit Tests - Replenish Race Condition Fix
# ============================================================================


class TestReplenishRaceCondition:
    """Tests for replenish race condition fix using semaphore serialization.

    These tests use asyncio.Event for deterministic synchronization instead of
    timing-based sleeps, ensuring reliable CI execution.
    """

    async def test_concurrent_replenish_serialized_by_semaphore(self, unit_test_vm_manager) -> None:
        """Prove only 1 boot runs at a time with semaphore."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Tracking variables
        max_concurrent = 0
        current_concurrent = 0
        boot_count = 0

        # Gates for deterministic control
        boot_started = asyncio.Event()  # Signals "a boot began"
        boot_can_finish = asyncio.Event()  # Test controls when boots complete

        async def controlled_boot(language: Language, index: int) -> AsyncMock:
            nonlocal max_concurrent, current_concurrent, boot_count

            # Track concurrency
            current_concurrent += 1
            boot_count += 1
            max_concurrent = max(max_concurrent, current_concurrent)

            boot_started.set()  # Tell test "I started"
            await boot_can_finish.wait()  # Wait for test permission

            current_concurrent -= 1

            vm = AsyncMock()
            vm.vm_id = f"vm-{boot_count}"
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=controlled_boot):
            # Spawn 3 concurrent replenish tasks
            tasks = [asyncio.create_task(pool._replenish_pool(Language.PYTHON)) for _ in range(3)]

            # Wait for first boot to start (deterministic, no sleep)
            await asyncio.wait_for(boot_started.wait(), timeout=1.0)

            # Release the gate - let all proceed
            boot_can_finish.set()

            await asyncio.gather(*tasks)

        # THE KEY ASSERTIONS:
        # With semaphore: max_concurrent == 1 (serialized)
        # Without semaphore: max_concurrent == 3 (racing)
        assert max_concurrent == 1, f"Expected 1 concurrent boot, got {max_concurrent}"

        # Only 1 boot needed - others see pool full and skip
        assert boot_count == 1, f"Expected 1 boot, got {boot_count}"

    async def test_replenish_skips_when_pool_full(self, unit_test_vm_manager) -> None:
        """After first replenish, subsequent calls skip (pool full)."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)  # pool_size = 1
        pool = WarmVMPool(unit_test_vm_manager, config)

        boot_count = 0

        async def mock_boot(language: Language, index: int) -> AsyncMock:
            nonlocal boot_count
            boot_count += 1
            vm = AsyncMock()
            vm.vm_id = f"vm-{boot_count}"
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=mock_boot):
            # First replenish - should boot
            await pool._replenish_pool(Language.PYTHON)
            assert boot_count == 1
            assert pool.pools[Language.PYTHON].qsize() == 1

            # Second replenish - pool is full, should skip
            await pool._replenish_pool(Language.PYTHON)
            assert boot_count == 1  # No additional boot

    async def test_per_language_semaphore_independence(self, unit_test_vm_manager) -> None:
        """Python replenish must not block JavaScript (would deadlock if shared)."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        py_started = asyncio.Event()
        js_started = asyncio.Event()

        async def mock_boot(language: Language, index: int) -> AsyncMock:
            if language == Language.PYTHON:
                py_started.set()
                # Would deadlock here if JS is blocked by Python's semaphore
                await asyncio.wait_for(js_started.wait(), timeout=1.0)
            else:
                js_started.set()
                # Would deadlock here if Python is blocked by JS's semaphore
                await asyncio.wait_for(py_started.wait(), timeout=1.0)

            vm = AsyncMock()
            vm.vm_id = f"{language.value}"
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=mock_boot):
            # This times out (fails) if languages block each other
            await asyncio.wait_for(
                asyncio.gather(
                    pool._replenish_pool(Language.PYTHON),
                    pool._replenish_pool(Language.JAVASCRIPT),
                ),
                timeout=2.0,  # Would hang forever if blocking
            )

    async def test_semaphore_released_on_boot_failure(self, unit_test_vm_manager) -> None:
        """Semaphore is released even when boot fails, allowing retry."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        call_count = 0

        async def failing_then_succeeding_boot(language: Language, index: int) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated boot failure")
            vm = AsyncMock()
            vm.vm_id = f"vm-{call_count}"
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=failing_then_succeeding_boot):
            # First replenish fails
            await pool._replenish_pool(Language.PYTHON)
            assert pool.pools[Language.PYTHON].qsize() == 0  # Failed, no VM added

            # Second replenish should succeed (semaphore was released)
            await pool._replenish_pool(Language.PYTHON)
            assert pool.pools[Language.PYTHON].qsize() == 1  # Success

        assert call_count == 2

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    async def test_semaphore_released_on_cancellation(self, unit_test_vm_manager) -> None:
        """Semaphore is released when task is cancelled during boot."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        boot_started = asyncio.Event()

        async def slow_boot(language: Language, index: int) -> AsyncMock:
            boot_started.set()
            await asyncio.sleep(10)  # Will be cancelled
            vm = AsyncMock()
            vm.vm_id = "never-returned"
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=slow_boot):
            task = asyncio.create_task(pool._replenish_pool(Language.PYTHON))

            # Wait for boot to start
            await asyncio.wait_for(boot_started.wait(), timeout=1.0)

            # Semaphore should be held
            assert pool._replenish_semaphores[Language.PYTHON].locked()

            # Cancel the task
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # Semaphore should be released after cancellation
            assert not pool._replenish_semaphores[Language.PYTHON].locked()

    async def test_vm_destroyed_on_cancellation_after_creation(self, unit_test_vm_manager) -> None:
        """VM is destroyed when task is cancelled AFTER VM creation (during pool.put)."""
        from unittest.mock import AsyncMock, patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        created_vm = AsyncMock()
        created_vm.vm_id = "created-then-cancelled"
        destroy_called = asyncio.Event()
        put_started = asyncio.Event()

        async def instant_boot(language: Language, index: int) -> AsyncMock:
            return created_vm

        async def mock_destroy(vm) -> None:
            destroy_called.set()

        # Mock put() to block after signaling it started
        async def slow_put(vm):
            put_started.set()
            await asyncio.sleep(10)  # Will be cancelled

        with (
            patch.object(pool, "_boot_warm_vm", side_effect=instant_boot),
            patch.object(pool.vm_manager, "destroy_vm", side_effect=mock_destroy),
            patch.object(pool.pools[Language.PYTHON], "put", side_effect=slow_put),
        ):
            task = asyncio.create_task(pool._replenish_pool(Language.PYTHON))

            # Wait for put() to start
            await asyncio.wait_for(put_started.wait(), timeout=1.0)

            # Cancel while blocked in put() - VM has been created but not added to pool
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # VM should have been destroyed due to CancelledError handling
            await asyncio.sleep(0.01)
            assert destroy_called.is_set(), "VM should be destroyed on cancellation"

    async def test_empty_pool_replenish(self, unit_test_vm_manager) -> None:
        """Replenish on completely empty pool works correctly."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)  # pool_size = 1
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Pool starts empty
        assert pool.pools[Language.PYTHON].qsize() == 0

        async def mock_boot(language: Language, index: int) -> AsyncMock:
            vm = AsyncMock()
            vm.vm_id = "new-vm"
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=mock_boot):
            await pool._replenish_pool(Language.PYTHON)

        assert pool.pools[Language.PYTHON].qsize() == 1

    # -------------------------------------------------------------------------
    # Boundary Cases
    # -------------------------------------------------------------------------

    async def test_larger_pool_multiple_replenishes(self, unit_test_vm_manager) -> None:
        """With pool_size > 1, multiple sequential replenishes fill the pool."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        # warm_pool_size=5
        config = SchedulerConfig(warm_pool_size=5)
        pool = WarmVMPool(unit_test_vm_manager, config)

        assert pool.pool_size_per_language == 5

        boot_count = 0

        async def mock_boot(language: Language, index: int) -> AsyncMock:
            nonlocal boot_count
            boot_count += 1
            vm = AsyncMock()
            vm.vm_id = f"vm-{boot_count}"
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=mock_boot):
            # Replenish 5 times to fill pool
            for i in range(5):
                await pool._replenish_pool(Language.PYTHON)
                assert pool.pools[Language.PYTHON].qsize() == i + 1

            # 6th replenish should skip (pool full)
            await pool._replenish_pool(Language.PYTHON)
            assert boot_count == 5  # No additional boot

    async def test_pool_size_one_boundary(self, unit_test_vm_manager) -> None:
        """Pool size = 1 is the minimum boundary case."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        # warm_pool_size=1
        config = SchedulerConfig(warm_pool_size=1)
        pool = WarmVMPool(unit_test_vm_manager, config)

        assert pool.pool_size_per_language == 1

        boot_count = 0

        async def mock_boot(language: Language, index: int) -> AsyncMock:
            nonlocal boot_count
            boot_count += 1
            vm = AsyncMock()
            vm.vm_id = f"vm-{boot_count}"
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=mock_boot):
            # First replenish fills the pool
            await pool._replenish_pool(Language.PYTHON)
            assert pool.pools[Language.PYTHON].qsize() == 1
            assert boot_count == 1

            # Second replenish skips
            await pool._replenish_pool(Language.PYTHON)
            assert boot_count == 1

    # -------------------------------------------------------------------------
    # Stress Cases
    # -------------------------------------------------------------------------

    async def test_many_concurrent_replenishes_small_pool(self, unit_test_vm_manager) -> None:
        """10 concurrent replenishes on pool_size=2 → only 2 boots, max 1 concurrent."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        # warm_pool_size=2
        # replenish_max_concurrent = max(1, int(2 * 0.5)) = 1
        config = SchedulerConfig(warm_pool_size=2)
        pool = WarmVMPool(unit_test_vm_manager, config)

        assert pool.pool_size_per_language == 2
        assert pool._replenish_max_concurrent == 1  # Small pool = serialized

        max_concurrent = 0
        current_concurrent = 0
        boot_count = 0
        boot_started = asyncio.Event()
        boot_can_finish = asyncio.Event()

        async def controlled_boot(language: Language, index: int) -> AsyncMock:
            nonlocal max_concurrent, current_concurrent, boot_count

            current_concurrent += 1
            boot_count += 1
            max_concurrent = max(max_concurrent, current_concurrent)

            boot_started.set()
            await boot_can_finish.wait()

            current_concurrent -= 1

            vm = AsyncMock()
            vm.vm_id = f"vm-{boot_count}"
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=controlled_boot):
            # Spawn 10 concurrent replenish tasks
            tasks = [asyncio.create_task(pool._replenish_pool(Language.PYTHON)) for _ in range(10)]

            # Wait for first boot to start
            await asyncio.wait_for(boot_started.wait(), timeout=1.0)

            # Release gate
            boot_can_finish.set()

            await asyncio.gather(*tasks)

        # Only 2 boots needed (pool_size=2), others skip
        assert boot_count == 2, f"Expected 2 boots, got {boot_count}"
        # Max concurrent = 1 for small pools (serialized by semaphore)
        assert max_concurrent == 1, f"Expected max 1 concurrent, got {max_concurrent}"
        # Pool should be full
        assert pool.pools[Language.PYTHON].qsize() == 2

    async def test_concurrent_replenish_large_pool_allows_parallelism(self, unit_test_vm_manager) -> None:
        """Large pool (size=5) allows 2 concurrent boots for faster replenishment."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        # warm_pool_size=5
        # replenish_max_concurrent = max(1, int(5 * 0.5)) = 2
        config = SchedulerConfig(warm_pool_size=5)
        pool = WarmVMPool(unit_test_vm_manager, config)

        assert pool.pool_size_per_language == 5
        assert pool._replenish_max_concurrent == 2  # Large pool = parallel boots

        max_concurrent = 0
        current_concurrent = 0
        boot_count = 0
        boots_started = asyncio.Event()
        boot_can_finish = asyncio.Event()

        async def controlled_boot(language: Language, index: int) -> AsyncMock:
            nonlocal max_concurrent, current_concurrent, boot_count

            current_concurrent += 1
            boot_count += 1
            max_concurrent = max(max_concurrent, current_concurrent)

            # Signal when 2 boots are running concurrently
            if current_concurrent >= 2:
                boots_started.set()

            await boot_can_finish.wait()

            current_concurrent -= 1

            vm = AsyncMock()
            vm.vm_id = f"vm-{boot_count}"
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=controlled_boot):
            # Spawn 10 concurrent replenish tasks
            tasks = [asyncio.create_task(pool._replenish_pool(Language.PYTHON)) for _ in range(10)]

            # Wait for 2 concurrent boots (proves parallelism works)
            await asyncio.wait_for(boots_started.wait(), timeout=1.0)

            # Release gate
            boot_can_finish.set()

            await asyncio.gather(*tasks)

        # Only 5 boots needed (pool_size=5), others skip
        assert boot_count == 5, f"Expected 5 boots, got {boot_count}"
        # Max concurrent should be 2 (limited by semaphore, not 10)
        assert max_concurrent == 2, f"Expected max 2 concurrent, got {max_concurrent}"
        # Pool should be full
        assert pool.pools[Language.PYTHON].qsize() == 5

    # -------------------------------------------------------------------------
    # Weird Cases
    # -------------------------------------------------------------------------

    async def test_pool_filled_externally_during_semaphore_wait(self, unit_test_vm_manager) -> None:
        """If pool becomes full while waiting for semaphore, skip boot."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(warm_pool_size=1)  # pool_size = 1
        pool = WarmVMPool(unit_test_vm_manager, config)

        boot_count = 0
        first_boot_done = asyncio.Event()

        async def mock_boot(language: Language, index: int) -> AsyncMock:
            nonlocal boot_count
            boot_count += 1
            vm = AsyncMock()
            vm.vm_id = f"vm-{boot_count}"
            first_boot_done.set()
            return vm

        with patch.object(pool, "_boot_warm_vm", side_effect=mock_boot):
            # Start first replenish (will acquire semaphore and boot)
            task1 = asyncio.create_task(pool._replenish_pool(Language.PYTHON))

            # Wait for first boot to complete
            await asyncio.wait_for(first_boot_done.wait(), timeout=1.0)
            await task1

            # Pool is now full
            assert pool.pools[Language.PYTHON].qsize() == 1

            # Second replenish should see pool is full and skip
            await pool._replenish_pool(Language.PYTHON)

            # Only 1 boot should have happened
            assert boot_count == 1

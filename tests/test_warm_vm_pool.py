"""Tests for WarmVMPool.

Unit tests: Pool data structures, config handling, healthcheck pure functions.
Integration tests: Real VM pool operations (requires QEMU + images).
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from tenacity import wait_none

from exec_sandbox import constants
from exec_sandbox.config import SchedulerConfig
from exec_sandbox.models import Language

# ============================================================================
# Unit Tests - No QEMU needed
# ============================================================================


class TestWarmVMPoolConfig:
    """Tests for WarmVMPool configuration."""

    def test_pool_size_calculation(self) -> None:
        """Pool size is 25% of max_concurrent_vms."""
        # The calculation: max(1, int(max_concurrent_vms * 0.25))

        # max_concurrent_vms=10 → pool_size=2
        expected = max(1, int(10 * constants.WARM_POOL_SIZE_RATIO))
        assert expected == 2

        # max_concurrent_vms=100 → pool_size=25
        expected = max(1, int(100 * constants.WARM_POOL_SIZE_RATIO))
        assert expected == 25

        # max_concurrent_vms=1 → pool_size=1 (minimum)
        expected = max(1, int(1 * constants.WARM_POOL_SIZE_RATIO))
        assert expected == 1

    def test_warm_pool_languages(self) -> None:
        """Warm pool supports python and javascript."""
        assert Language.PYTHON in constants.WARM_POOL_LANGUAGES
        assert Language.JAVASCRIPT in constants.WARM_POOL_LANGUAGES
        assert len(constants.WARM_POOL_LANGUAGES) == 2


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


class TestDrainPoolForCheck:
    """Tests for _drain_pool_for_check - pure queue draining logic."""

    async def test_drain_empty_pool(self, unit_test_vm_manager) -> None:
        """Draining empty pool returns empty list."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        # Create pool with minimal config (no VMs booted)
        config = SchedulerConfig(max_concurrent_vms=4)
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

        config = SchedulerConfig(max_concurrent_vms=4)
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

        config = SchedulerConfig(max_concurrent_vms=4)
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

        config = SchedulerConfig(max_concurrent_vms=4)
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


class TestEvaluateHealthResult:
    """Tests for _evaluate_health_result - pure result evaluation logic."""

    async def test_evaluate_true_returns_true(self, unit_test_vm_manager) -> None:
        """True result means healthy."""
        from exec_sandbox.vm_manager import QemuVM
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Create a minimal VM object for testing
        vm = QemuVM.__new__(QemuVM)
        vm.vm_id = "test-vm"

        result = pool._evaluate_health_result(True, Language.PYTHON, vm)
        assert result is True

    async def test_evaluate_false_returns_false(self, unit_test_vm_manager) -> None:
        """False result means unhealthy."""
        from exec_sandbox.vm_manager import QemuVM
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = QemuVM.__new__(QemuVM)
        vm.vm_id = "test-vm"

        result = pool._evaluate_health_result(False, Language.PYTHON, vm)
        assert result is False

    async def test_evaluate_exception_returns_false(self, unit_test_vm_manager) -> None:
        """Exception result means unhealthy."""
        from exec_sandbox.vm_manager import QemuVM
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = QemuVM.__new__(QemuVM)
        vm.vm_id = "test-vm"

        # Any exception should be treated as unhealthy
        result = pool._evaluate_health_result(
            TimeoutError("connection timeout"),
            Language.PYTHON,
            vm,
        )
        assert result is False

    async def test_evaluate_oserror_returns_false(self, unit_test_vm_manager) -> None:
        """OSError (connection failed) means unhealthy."""
        from exec_sandbox.vm_manager import QemuVM
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = QemuVM.__new__(QemuVM)
        vm.vm_id = "test-vm"

        result = pool._evaluate_health_result(
            OSError("Connection refused"),
            Language.PYTHON,
            vm,
        )
        assert result is False

    async def test_evaluate_connection_error_returns_false(self, unit_test_vm_manager) -> None:
        """ConnectionError means unhealthy."""
        from exec_sandbox.vm_manager import QemuVM
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = QemuVM.__new__(QemuVM)
        vm.vm_id = "test-vm"

        result = pool._evaluate_health_result(
            ConnectionError("Connection reset by peer"),
            Language.PYTHON,
            vm,
        )
        assert result is False

    async def test_evaluate_cancelled_error_returns_false(self, unit_test_vm_manager) -> None:
        """CancelledError (task cancelled) means unhealthy."""
        from exec_sandbox.vm_manager import QemuVM
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = QemuVM.__new__(QemuVM)
        vm.vm_id = "test-vm"

        result = pool._evaluate_health_result(
            asyncio.CancelledError(),
            Language.PYTHON,
            vm,
        )
        assert result is False

    async def test_evaluate_base_exception_returns_false(self, unit_test_vm_manager) -> None:
        """BaseException (keyboard interrupt, system exit) means unhealthy."""
        from exec_sandbox.vm_manager import QemuVM
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm = QemuVM.__new__(QemuVM)
        vm.vm_id = "test-vm"

        # BaseException that's not an Exception (KeyboardInterrupt, SystemExit)
        # Implementation correctly handles this as unhealthy
        result = pool._evaluate_health_result(
            KeyboardInterrupt(),
            Language.PYTHON,
            vm,
        )
        assert result is False


# ============================================================================
# Unit Tests - Process Health Results (No QEMU, No Mocks)
# ============================================================================


class TestProcessHealthResults:
    """Tests for _process_health_results - result processing logic."""

    async def test_process_empty_results(self, unit_test_vm_manager) -> None:
        """Processing empty results list returns zeros."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        test_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)

        healthy, unhealthy = await pool._process_health_results(
            Language.PYTHON,
            test_queue,  # type: ignore[arg-type]
            vms=[],
            results=[],
        )

        assert healthy == 0
        assert unhealthy == 0
        assert test_queue.qsize() == 0

    async def test_process_all_healthy(self, unit_test_vm_manager) -> None:
        """All healthy results restores all VMs to pool."""
        from exec_sandbox.vm_manager import QemuVM
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Create minimal VM objects
        vm1 = QemuVM.__new__(QemuVM)
        vm1.vm_id = "vm1"
        vm2 = QemuVM.__new__(QemuVM)
        vm2.vm_id = "vm2"

        test_queue: asyncio.Queue[QemuVM] = asyncio.Queue(maxsize=10)

        healthy, unhealthy = await pool._process_health_results(
            Language.PYTHON,
            test_queue,
            vms=[vm1, vm2],
            results=[True, True],
        )

        assert healthy == 2
        assert unhealthy == 0
        assert test_queue.qsize() == 2

    async def test_process_all_unhealthy(self, unit_test_vm_manager) -> None:
        """All unhealthy results triggers cleanup for all VMs."""
        from exec_sandbox.vm_manager import QemuVM
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm1 = QemuVM.__new__(QemuVM)
        vm1.vm_id = "vm1"
        vm2 = QemuVM.__new__(QemuVM)
        vm2.vm_id = "vm2"

        test_queue: asyncio.Queue[QemuVM] = asyncio.Queue(maxsize=10)

        healthy, unhealthy = await pool._process_health_results(
            Language.PYTHON,
            test_queue,
            vms=[vm1, vm2],
            results=[False, False],
        )

        assert healthy == 0
        assert unhealthy == 2
        assert test_queue.qsize() == 0  # No VMs restored

    async def test_process_mixed_results(self, unit_test_vm_manager) -> None:
        """Mixed results restores only healthy VMs."""
        from exec_sandbox.vm_manager import QemuVM
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        vm1 = QemuVM.__new__(QemuVM)
        vm1.vm_id = "vm1"
        vm2 = QemuVM.__new__(QemuVM)
        vm2.vm_id = "vm2"
        vm3 = QemuVM.__new__(QemuVM)
        vm3.vm_id = "vm3"

        test_queue: asyncio.Queue[QemuVM] = asyncio.Queue(maxsize=10)

        # vm1: healthy, vm2: unhealthy (False), vm3: unhealthy (exception)
        healthy, unhealthy = await pool._process_health_results(
            Language.PYTHON,
            test_queue,
            vms=[vm1, vm2, vm3],
            results=[True, False, TimeoutError("timeout")],
        )

        assert healthy == 1
        assert unhealthy == 2
        assert test_queue.qsize() == 1

        # Verify the healthy VM was restored
        restored_vm = await test_queue.get()
        assert restored_vm.vm_id == "vm1"


# ============================================================================
# Unit Tests - Health Check Pool Empty Case (No QEMU, No Mocks)
# ============================================================================


class TestHealthCheckPoolUnit:
    """Unit tests for _health_check_pool edge cases."""

    async def test_health_check_empty_pool_returns_early(self, unit_test_vm_manager) -> None:
        """Health check on empty pool returns immediately without error."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
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


class TestWarmVMPoolIntegration:
    """Integration tests for WarmVMPool with real QEMU VMs."""

    async def test_pool_startup_shutdown(self, vm_manager) -> None:
        """Pool starts and shuts down cleanly."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(vm_manager, config)

        await pool.startup()

        # Pools should be populated
        assert pool.pools[Language.PYTHON].qsize() > 0

        await pool.shutdown()

        # Pools should be empty
        assert pool.pools[Language.PYTHON].qsize() == 0
        assert pool.pools[Language.JAVASCRIPT].qsize() == 0

    async def test_get_vm_from_pool(self, vm_manager) -> None:
        """Get VM from warm pool."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(vm_manager, config)

        await pool.startup()

        try:
            # Get VM from pool (should be instant)
            vm = await pool.get_vm(Language.PYTHON, packages=[])

            assert vm is not None
            assert vm.vm_id is not None

            # Destroy VM after use
            await vm_manager.destroy_vm(vm)

        finally:
            await pool.shutdown()

    async def test_get_vm_with_packages_returns_none(self, vm_manager) -> None:
        """Get VM with packages returns None (not eligible for warm pool)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(vm_manager, config)

        await pool.startup()

        try:
            # Get VM with packages - should return None
            vm = await pool.get_vm(Language.PYTHON, packages=["pandas==2.0.0"])
            assert vm is None

        finally:
            await pool.shutdown()


# ============================================================================
# Integration Tests - Healthcheck Workflow (Require QEMU + Images)
# ============================================================================


class TestHealthcheckIntegration:
    """Integration tests for healthcheck with real QEMU VMs."""

    async def test_check_vm_health_healthy_vm(self, vm_manager) -> None:
        """_check_vm_health returns True for healthy VM."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(vm_manager, config)

        await pool.startup()

        try:
            # Get a VM from pool
            vm = await pool.get_vm(Language.PYTHON, packages=[])
            assert vm is not None

            # Health check should pass for a freshly booted VM
            is_healthy = await pool._check_vm_health(vm)
            assert is_healthy is True

            await vm_manager.destroy_vm(vm)

        finally:
            await pool.shutdown()

    async def test_health_check_pool_preserves_healthy_vms(self, vm_manager) -> None:
        """_health_check_pool keeps healthy VMs in pool."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(vm_manager, config)

        await pool.startup()

        try:
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
            await pool.shutdown()

    async def test_drain_pool_restores_vms_after_health_check(self, vm_manager) -> None:
        """VMs drained for health check are restored to pool."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(vm_manager, config)

        await pool.startup()

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

            # Check health of all VMs
            results = await asyncio.gather(
                *[pool._check_vm_health(vm) for vm in vms],
                return_exceptions=True,
            )

            # Process results - should restore all healthy VMs
            healthy_count, unhealthy_count = await pool._process_health_results(
                Language.PYTHON,
                python_pool,
                vms,
                results,
            )

            assert healthy_count == initial_size
            assert unhealthy_count == 0
            assert python_pool.qsize() == initial_size

        finally:
            await pool.shutdown()

    async def test_health_check_loop_stops_on_shutdown(self, vm_manager) -> None:
        """Health check loop exits cleanly when shutdown is signaled."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(vm_manager, config)

        await pool.startup()

        # Health task should be running
        assert pool._health_task is not None
        assert not pool._health_task.done()

        # Shutdown should stop health task
        await pool.shutdown()

        assert pool._health_task.done()
        assert pool._shutdown_event.is_set()


# ============================================================================
# Unit Tests - Health Check Retry Logic (No QEMU, uses mocks)
# ============================================================================


class TestCheckVmHealthRetry:
    """Unit tests for _check_vm_health retry logic."""

    @pytest.fixture
    def mock_vm(self):
        """Create a mock VM with mocked channel."""
        vm = Mock()
        vm.vm_id = "test-vm"
        vm.channel = Mock()
        vm.channel.close = AsyncMock()
        vm.channel.connect = AsyncMock()
        vm.channel.send_request = AsyncMock()
        return vm

    async def test_success_on_first_attempt(self, unit_test_vm_manager, mock_vm) -> None:
        """Health check succeeds on first attempt without retry."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        mock_vm.channel.send_request = AsyncMock(return_value=PongMessage(version="1.0"))

        # Inject wait_none() for instant retries (no delays)
        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is True
        assert mock_vm.channel.send_request.call_count == 1

    async def test_retry_succeeds_after_transient_timeout(self, unit_test_vm_manager, mock_vm) -> None:
        """Retry succeeds after transient TimeoutError."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # First 2 calls: TimeoutError, Third call: success
        mock_vm.channel.send_request = AsyncMock(
            side_effect=[
                TimeoutError("timeout"),
                TimeoutError("timeout"),
                PongMessage(version="1.0"),
            ]
        )

        # Inject wait_none() - retries happen instantly
        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is True
        assert mock_vm.channel.send_request.call_count == 3

    async def test_retry_succeeds_after_transient_oserror(self, unit_test_vm_manager, mock_vm) -> None:
        """Retry succeeds after transient OSError."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        mock_vm.channel.send_request = AsyncMock(
            side_effect=[OSError("connection refused"), PongMessage(version="1.0")]
        )

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is True
        assert mock_vm.channel.send_request.call_count == 2

    async def test_retry_succeeds_after_connection_error(self, unit_test_vm_manager, mock_vm) -> None:
        """Retry succeeds after transient ConnectionError."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        mock_vm.channel.send_request = AsyncMock(side_effect=[ConnectionError("reset"), PongMessage(version="1.0")])

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is True
        assert mock_vm.channel.send_request.call_count == 2

    async def test_retry_exhausted_returns_false(self, unit_test_vm_manager, mock_vm) -> None:
        """Returns False after all retries exhausted."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # All attempts fail with TimeoutError
        mock_vm.channel.send_request = AsyncMock(side_effect=TimeoutError("always timeout"))

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is False
        assert mock_vm.channel.send_request.call_count == constants.WARM_POOL_HEALTH_CHECK_MAX_RETRIES

    async def test_cancellation_not_retried(self, unit_test_vm_manager, mock_vm) -> None:
        """CancelledError propagates immediately without retry."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        mock_vm.channel.connect = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await pool._check_vm_health(mock_vm, _wait=wait_none())

        # Should only attempt once - no retry on cancellation
        assert mock_vm.channel.connect.call_count == 1

    @pytest.mark.parametrize(
        "exception_type",
        [
            OSError("connection refused"),
            TimeoutError("timeout"),
            ConnectionError("connection reset"),
        ],
    )
    async def test_retries_on_all_transient_exception_types(
        self, unit_test_vm_manager, mock_vm, exception_type
    ) -> None:
        """Retries on all transient exception types then succeeds."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        mock_vm.channel.send_request = AsyncMock(side_effect=[exception_type, PongMessage(version="1.0")])

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is True
        assert mock_vm.channel.send_request.call_count == 2

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    async def test_wrong_response_type_returns_false_no_retry(self, unit_test_vm_manager, mock_vm) -> None:
        """Wrong response type returns False without retry (protocol error, not transient)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Return something that's not a PongMessage
        wrong_response = Mock()
        wrong_response.__class__.__name__ = "UnexpectedMessage"
        mock_vm.channel.send_request = AsyncMock(return_value=wrong_response)

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is False
        # Should not retry - wrong response type is not a transient error
        assert mock_vm.channel.send_request.call_count == 1

    async def test_connect_failure_is_retried(self, unit_test_vm_manager, mock_vm) -> None:
        """Failure during connect() is retried."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Connect fails twice, then succeeds
        mock_vm.channel.connect = AsyncMock(side_effect=[OSError("refused"), TimeoutError("timeout"), None])
        mock_vm.channel.send_request = AsyncMock(return_value=PongMessage(version="1.0"))

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is True
        assert mock_vm.channel.connect.call_count == 3
        assert mock_vm.channel.send_request.call_count == 1

    async def test_close_failure_is_retried(self, unit_test_vm_manager, mock_vm) -> None:
        """Failure during close() is retried."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Close fails once, then succeeds
        mock_vm.channel.close = AsyncMock(side_effect=[OSError("broken pipe"), None])
        mock_vm.channel.send_request = AsyncMock(return_value=PongMessage(version="1.0"))

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is True
        assert mock_vm.channel.close.call_count == 2

    async def test_mixed_exception_types_across_retries(self, unit_test_vm_manager, mock_vm) -> None:
        """Different exception types across retries still succeed."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Different exceptions on each retry
        mock_vm.channel.send_request = AsyncMock(
            side_effect=[
                OSError("connection refused"),
                TimeoutError("timeout"),
                PongMessage(version="1.0"),
            ]
        )

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is True
        assert mock_vm.channel.send_request.call_count == 3

    async def test_success_on_last_retry_boundary(self, unit_test_vm_manager, mock_vm) -> None:
        """Success on exactly the last retry attempt (boundary condition)."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # Fail exactly (MAX_RETRIES - 1) times, succeed on last attempt
        failures = [TimeoutError("timeout")] * (constants.WARM_POOL_HEALTH_CHECK_MAX_RETRIES - 1)
        mock_vm.channel.send_request = AsyncMock(side_effect=[*failures, PongMessage(version="1.0")])

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is True
        assert mock_vm.channel.send_request.call_count == constants.WARM_POOL_HEALTH_CHECK_MAX_RETRIES

    async def test_failure_on_last_retry_returns_false(self, unit_test_vm_manager, mock_vm) -> None:
        """Failure on last retry returns False (boundary condition)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # All MAX_RETRIES attempts fail
        mock_vm.channel.send_request = AsyncMock(side_effect=TimeoutError("always fails"))

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is False
        assert mock_vm.channel.send_request.call_count == constants.WARM_POOL_HEALTH_CHECK_MAX_RETRIES

    # -------------------------------------------------------------------------
    # Weird Cases
    # -------------------------------------------------------------------------

    async def test_none_response_returns_false(self, unit_test_vm_manager, mock_vm) -> None:
        """None response returns False (edge case)."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        mock_vm.channel.send_request = AsyncMock(return_value=None)

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is False
        assert mock_vm.channel.send_request.call_count == 1

    async def test_exception_subclass_is_retried(self, unit_test_vm_manager, mock_vm) -> None:
        """Exception subclasses (e.g., ConnectionRefusedError) are retried."""
        from exec_sandbox.guest_agent_protocol import PongMessage
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        # ConnectionRefusedError is a subclass of ConnectionError
        mock_vm.channel.send_request = AsyncMock(
            side_effect=[ConnectionRefusedError("refused"), PongMessage(version="1.0")]
        )

        result = await pool._check_vm_health(mock_vm, _wait=wait_none())

        assert result is True
        assert mock_vm.channel.send_request.call_count == 2

    async def test_non_retryable_exception_propagates(self, unit_test_vm_manager, mock_vm) -> None:
        """Non-retryable exceptions (e.g., ValueError) propagate immediately."""
        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)
        pool = WarmVMPool(unit_test_vm_manager, config)

        mock_vm.channel.send_request = AsyncMock(side_effect=ValueError("unexpected"))

        with pytest.raises(ValueError, match="unexpected"):
            await pool._check_vm_health(mock_vm, _wait=wait_none())

        # Should only attempt once - ValueError is not retryable
        assert mock_vm.channel.send_request.call_count == 1


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

        config = SchedulerConfig(max_concurrent_vms=4)
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

        config = SchedulerConfig(max_concurrent_vms=4)  # pool_size = 1
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

        config = SchedulerConfig(max_concurrent_vms=4)
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

        config = SchedulerConfig(max_concurrent_vms=4)
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

        config = SchedulerConfig(max_concurrent_vms=4)
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

    async def test_empty_pool_replenish(self, unit_test_vm_manager) -> None:
        """Replenish on completely empty pool works correctly."""
        from unittest.mock import patch

        from exec_sandbox.warm_vm_pool import WarmVMPool

        config = SchedulerConfig(max_concurrent_vms=4)  # pool_size = 1
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

        # max_concurrent_vms=20 → pool_size = max(1, int(20 * 0.25)) = 5
        config = SchedulerConfig(max_concurrent_vms=20)
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

        # max_concurrent_vms=1 → pool_size = max(1, int(1 * 0.25)) = max(1, 0) = 1
        config = SchedulerConfig(max_concurrent_vms=1)
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

        # max_concurrent_vms=8 → pool_size = max(1, int(8 * 0.25)) = 2
        # replenish_max_concurrent = max(1, int(2 * 0.5)) = 1
        config = SchedulerConfig(max_concurrent_vms=8)
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

        # max_concurrent_vms=20 → pool_size = max(1, int(20 * 0.25)) = 5
        # replenish_max_concurrent = max(1, int(5 * 0.5)) = 2
        config = SchedulerConfig(max_concurrent_vms=20)
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

        config = SchedulerConfig(max_concurrent_vms=4)  # pool_size = 1
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

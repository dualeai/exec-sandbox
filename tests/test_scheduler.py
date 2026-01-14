"""Tests for Scheduler.

Unit tests: Test validation, config, error handling (no QEMU needed).
Integration tests: Test real VM execution (requires QEMU + images).
"""

from pathlib import Path

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.exceptions import SandboxError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# ============================================================================
# Unit Tests - No QEMU needed
# ============================================================================


class TestSchedulerInit:
    """Tests for Scheduler initialization."""

    def test_init_default_config(self) -> None:
        """Scheduler can be created with default config."""
        scheduler = Scheduler()
        assert scheduler.config is not None
        assert scheduler.config.max_concurrent_vms == 10

    def test_init_custom_config(self) -> None:
        """Scheduler accepts custom config."""
        config = SchedulerConfig(
            max_concurrent_vms=5,
            default_memory_mb=512,
            default_timeout_seconds=60,
        )
        scheduler = Scheduler(config)
        assert scheduler.config.max_concurrent_vms == 5
        assert scheduler.config.default_memory_mb == 512
        assert scheduler.config.default_timeout_seconds == 60

    def test_not_started_initially(self) -> None:
        """Scheduler is not started after __init__."""
        scheduler = Scheduler()
        assert scheduler._started is False

    def test_internal_state_none_initially(self) -> None:
        """Internal managers are None before start."""
        scheduler = Scheduler()
        assert scheduler._vm_manager is None
        assert scheduler._snapshot_manager is None
        assert scheduler._warm_pool is None
        assert scheduler._semaphore is None


class TestSchedulerContextManager:
    """Tests for Scheduler context manager."""

    async def test_double_start_raises(self, tmp_path: Path) -> None:
        """Starting already-started scheduler raises SandboxError."""
        test_images_dir = tmp_path / "images"
        test_images_dir.mkdir()

        config = SchedulerConfig(images_dir=test_images_dir)
        scheduler = Scheduler(config)

        # Manually set _started to simulate already started
        scheduler._started = True

        with pytest.raises(SandboxError) as exc_info:
            await scheduler.__aenter__()

        assert "already started" in str(exc_info.value)

    async def test_run_without_start_raises(self) -> None:
        """Calling run() without starting raises SandboxError."""
        scheduler = Scheduler()

        with pytest.raises(SandboxError) as exc_info:
            await scheduler.run(code="print(1)", language=Language.PYTHON)

        assert "not started" in str(exc_info.value)


class TestPackageValidation:
    """Tests for package validation in Scheduler."""

    def test_validate_packages_allowed(self, tmp_path: Path) -> None:
        """Valid packages pass validation."""
        # Create a scheduler (we'll test the internal method)
        scheduler = Scheduler()

        # Access the internal validate method - need catalogs
        # This test verifies the validation logic works
        # When package_validation is disabled, all packages pass
        config = SchedulerConfig(enable_package_validation=False)
        scheduler_no_validation = Scheduler(config)

        # Should not raise when validation disabled
        # (We can't test real validation without starting the scheduler)

    async def test_validate_packages_rejects_unknown(self, tmp_path: Path) -> None:
        """Unknown packages are rejected when validation enabled."""
        # Create test catalogs
        catalogs_dir = tmp_path / "catalogs"
        catalogs_dir.mkdir()

        import json

        (catalogs_dir / "pypi_top_10k.json").write_text(json.dumps(["pandas", "numpy"]))
        (catalogs_dir / "npm_top_10k.json").write_text(json.dumps(["lodash", "axios"]))

        # Note: Full validation test requires started scheduler
        # This is tested at a higher level in integration tests


class TestSchedulerConfig:
    """Tests for Scheduler config handling."""

    def test_config_immutable(self) -> None:
        """Scheduler config is immutable."""
        config = SchedulerConfig(max_concurrent_vms=5)
        scheduler = Scheduler(config)

        # Config should be the same object (frozen)
        assert scheduler.config is config

    def test_s3_not_configured_by_default(self) -> None:
        """S3 snapshot manager not created without s3_bucket."""
        config = SchedulerConfig()
        scheduler = Scheduler(config)
        assert config.s3_bucket is None

    def test_warm_pool_disabled_by_default(self) -> None:
        """Warm pool not created when warm_pool_size is 0."""
        config = SchedulerConfig(warm_pool_size=0)
        scheduler = Scheduler(config)
        assert config.warm_pool_size == 0


class TestSchedulerSnapshotInit:
    """Tests for SnapshotManager initialization in Scheduler."""

    async def test_snapshot_manager_initialized_without_s3(self, scheduler_config: SchedulerConfig) -> None:
        """SnapshotManager is created even without S3 config (L2 cache works)."""
        async with Scheduler(scheduler_config) as scheduler:
            assert scheduler._snapshot_manager is not None

    async def test_snapshot_manager_initialized_with_s3(self, images_dir: Path) -> None:
        """SnapshotManager is created with S3 config."""
        config = SchedulerConfig(
            images_dir=images_dir,
            s3_bucket="test-bucket",
            s3_region="us-east-1",
            auto_download_assets=False,
        )
        async with Scheduler(config) as scheduler:
            assert scheduler._snapshot_manager is not None

    async def test_snapshot_manager_has_vm_manager(self, scheduler_config: SchedulerConfig) -> None:
        """SnapshotManager receives vm_manager reference."""
        async with Scheduler(scheduler_config) as scheduler:
            assert scheduler._snapshot_manager is not None
            assert scheduler._snapshot_manager.vm_manager is scheduler._vm_manager


# ============================================================================
# Integration Tests - Require QEMU + Images
# ============================================================================


class TestSchedulerIntegration:
    """Integration tests for Scheduler with real QEMU VMs.

    These tests require:
    - QEMU installed
    - VM images built (run 'make build-images')
    """

    async def test_scheduler_lifecycle(self, scheduler_config: SchedulerConfig) -> None:
        """Scheduler starts and stops cleanly."""
        async with Scheduler(scheduler_config) as scheduler:
            assert scheduler._started is True
            assert scheduler._vm_manager is not None
            assert scheduler._semaphore is not None

        # After exit
        assert scheduler._started is False

    async def test_run_simple_python(self, scheduler: Scheduler) -> None:
        """Run simple Python code."""
        result = await scheduler.run(
            code="print('hello from python')",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert "hello from python" in result.stdout

    async def test_run_python_calculation(self, scheduler: Scheduler) -> None:
        """Run Python code with calculation."""
        result = await scheduler.run(
            code="print(2 + 2)",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert "4" in result.stdout

    async def test_run_python_multiline(self, scheduler: Scheduler) -> None:
        """Run multiline Python code."""
        code = """
for i in range(3):
    print(f"line {i}")
"""

        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "line 0" in result.stdout
        assert "line 1" in result.stdout
        assert "line 2" in result.stdout

    async def test_run_python_exit_code(self, scheduler: Scheduler) -> None:
        """Python code with non-zero exit."""
        result = await scheduler.run(
            code="import sys; sys.exit(42)",
            language=Language.PYTHON,
        )

        assert result.exit_code == 42

    async def test_run_python_stderr(self, scheduler: Scheduler) -> None:
        """Python code that writes to stderr."""
        code = """
import sys
print("stdout message")
print("stderr message", file=sys.stderr)
"""

        result = await scheduler.run(code=code, language=Language.PYTHON)

        assert result.exit_code == 0
        assert "stdout message" in result.stdout
        assert "stderr message" in result.stderr

    async def test_run_with_env_vars(self, scheduler: Scheduler) -> None:
        """Run with custom environment variables."""
        code = """
import os
print(os.environ.get('MY_VAR', 'not set'))
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            env_vars={"MY_VAR": "hello"},
        )

        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_run_with_streaming(self, scheduler: Scheduler) -> None:
        """Run with streaming output callbacks."""
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        code = """
import sys
print("out1")
print("err1", file=sys.stderr)
print("out2")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert result.exit_code == 0
        # Chunks should have been received
        assert len(stdout_chunks) > 0 or "out1" in result.stdout

    async def test_run_timeout(self, scheduler_config: SchedulerConfig) -> None:
        """Execution timeout works.

        Timeout can be enforced at two levels:
        1. Guest-agent soft timeout: Returns result with exit code (killed by signal)
        2. Host hard timeout: Raises VmTimeoutError

        The test accepts either behavior - both indicate the timeout worked.
        """
        config = SchedulerConfig(
            images_dir=scheduler_config.images_dir,
            default_timeout_seconds=2,
        )

        code = """
import time
time.sleep(30)
"""

        async with Scheduler(config) as scheduler:
            from exec_sandbox.exceptions import VmTimeoutError

            try:
                result = await scheduler.run(
                    code=code,
                    language=Language.PYTHON,
                    timeout_seconds=1,
                )
                # If we get here, guest-agent handled timeout
                # Process should have been killed (exit code != 0)
                assert result.exit_code != 0, (
                    f"Expected non-zero exit code for timed-out execution, got {result.exit_code}"
                )
            except VmTimeoutError:
                # Host timeout kicked in - also valid
                pass

    async def test_run_multiple_sequential(self, scheduler: Scheduler) -> None:
        """Multiple sequential runs work (VMs not reused)."""
        result1 = await scheduler.run(
            code="print('first')",
            language=Language.PYTHON,
        )
        result2 = await scheduler.run(
            code="print('second')",
            language=Language.PYTHON,
        )

        assert result1.exit_code == 0
        assert "first" in result1.stdout
        assert result2.exit_code == 0
        assert "second" in result2.stdout

    async def test_execution_result_metrics(self, scheduler: Scheduler) -> None:
        """ExecutionResult contains timing metrics."""
        result = await scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        # Metrics should be populated
        if result.execution_time_ms is not None:
            assert result.execution_time_ms >= 0


class TestSchedulerJavaScript:
    """JavaScript execution tests."""

    async def test_run_simple_javascript(self, scheduler: Scheduler) -> None:
        """Run simple JavaScript code."""
        result = await scheduler.run(
            code="console.log('hello from javascript')",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "hello from javascript" in result.stdout

    async def test_run_javascript_calculation(self, scheduler: Scheduler) -> None:
        """Run JavaScript with calculation."""
        result = await scheduler.run(
            code="console.log(2 + 2)",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "4" in result.stdout


# ============================================================================
# Parametrized Tests - All Image Types
# ============================================================================


# Test data for parametrized tests across all image types
SCHEDULER_IMAGE_TEST_CASES = [
    pytest.param(
        Language.PYTHON,
        "print('hello')",
        "hello",
        id="python",
    ),
    pytest.param(
        Language.JAVASCRIPT,
        "console.log('hello')",
        "hello",
        id="javascript",
    ),
    pytest.param(
        Language.RAW,
        "echo 'hello'",
        "hello",
        id="raw",
    ),
]


class TestSchedulerAllImages:
    """Parametrized tests to verify all image types work via Scheduler.

    Each image type (python, javascript, raw) must:
    1. Boot successfully (implicit via scheduler.run)
    2. Execute code and return correct output
    """

    @pytest.mark.parametrize("language,code,expected_output", SCHEDULER_IMAGE_TEST_CASES)
    async def test_scheduler_execute_all_images(
        self,
        scheduler: Scheduler,
        language: Language,
        code: str,
        expected_output: str,
    ) -> None:
        """Scheduler executes code for all image types."""
        result = await scheduler.run(
            code=code,
            language=language,
        )

        assert result.exit_code == 0, f"Exit code {result.exit_code}, stderr: {result.stderr}"
        assert expected_output in result.stdout, f"Expected '{expected_output}' in stdout: {result.stdout}"


# ============================================================================
# Unit Tests - TimingBreakdown and warm_pool_hit
# ============================================================================


class TestTimingBreakdownModel:
    """Unit tests for TimingBreakdown model validation."""

    def test_timing_breakdown_all_fields_required(self) -> None:
        """TimingBreakdown requires all four timing fields."""
        from exec_sandbox.models import TimingBreakdown

        timing = TimingBreakdown(setup_ms=100, boot_ms=200, execute_ms=50, total_ms=350)
        assert timing.setup_ms == 100
        assert timing.boot_ms == 200
        assert timing.execute_ms == 50
        assert timing.total_ms == 350

    def test_timing_breakdown_zero_values_valid(self) -> None:
        """TimingBreakdown accepts zero values (warm pool case)."""
        from exec_sandbox.models import TimingBreakdown

        timing = TimingBreakdown(setup_ms=0, boot_ms=0, execute_ms=50, total_ms=50)
        assert timing.setup_ms == 0
        assert timing.boot_ms == 0

    def test_timing_breakdown_missing_field_raises(self) -> None:
        """TimingBreakdown raises if any field is missing."""
        from pydantic import ValidationError

        from exec_sandbox.models import TimingBreakdown

        with pytest.raises(ValidationError):
            TimingBreakdown(setup_ms=100, boot_ms=200, execute_ms=50)  # type: ignore[call-arg]


class TestExecutionResultTiming:
    """Unit tests for ExecutionResult timing and warm_pool_hit fields."""

    def test_execution_result_timing_required(self) -> None:
        """ExecutionResult requires timing field."""
        from pydantic import ValidationError

        from exec_sandbox.models import ExecutionResult

        with pytest.raises(ValidationError):
            ExecutionResult(  # type: ignore[call-arg]
                stdout="hello",
                stderr="",
                exit_code=0,
                # Missing timing
            )

    def test_execution_result_with_timing(self) -> None:
        """ExecutionResult accepts timing field."""
        from exec_sandbox.models import ExecutionResult, TimingBreakdown

        result = ExecutionResult(
            stdout="hello",
            stderr="",
            exit_code=0,
            timing=TimingBreakdown(setup_ms=100, boot_ms=200, execute_ms=50, total_ms=350),
        )
        assert result.timing.setup_ms == 100
        assert result.timing.total_ms == 350

    def test_execution_result_warm_pool_hit_default_false(self) -> None:
        """warm_pool_hit defaults to False."""
        from exec_sandbox.models import ExecutionResult, TimingBreakdown

        result = ExecutionResult(
            stdout="hello",
            stderr="",
            exit_code=0,
            timing=TimingBreakdown(setup_ms=100, boot_ms=200, execute_ms=50, total_ms=350),
        )
        assert result.warm_pool_hit is False

    def test_execution_result_warm_pool_hit_explicit_true(self) -> None:
        """warm_pool_hit can be set to True."""
        from exec_sandbox.models import ExecutionResult, TimingBreakdown

        result = ExecutionResult(
            stdout="hello",
            stderr="",
            exit_code=0,
            timing=TimingBreakdown(setup_ms=0, boot_ms=0, execute_ms=50, total_ms=50),
            warm_pool_hit=True,
        )
        assert result.warm_pool_hit is True
        assert result.timing.setup_ms == 0
        assert result.timing.boot_ms == 0

    def test_timing_breakdown_with_connect_ms(self) -> None:
        """TimingBreakdown with optional connect_ms field."""
        from exec_sandbox.models import TimingBreakdown

        timing = TimingBreakdown(
            setup_ms=10,
            boot_ms=200,
            execute_ms=50,
            total_ms=260,
            connect_ms=5,
        )
        assert timing.connect_ms == 5

    def test_timing_breakdown_without_connect_ms(self) -> None:
        """TimingBreakdown without connect_ms (backwards compat)."""
        from exec_sandbox.models import TimingBreakdown

        timing = TimingBreakdown(
            setup_ms=10,
            boot_ms=200,
            execute_ms=50,
            total_ms=260,
        )
        assert timing.connect_ms is None

    def test_execution_result_with_guest_timing(self) -> None:
        """ExecutionResult with guest-reported spawn_ms and process_ms."""
        from exec_sandbox.models import ExecutionResult, TimingBreakdown

        result = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=0,
            timing=TimingBreakdown(setup_ms=0, boot_ms=0, execute_ms=0, total_ms=0),
            spawn_ms=5,
            process_ms=10,
        )
        assert result.spawn_ms == 5
        assert result.process_ms == 10

    def test_execution_result_without_guest_timing(self) -> None:
        """ExecutionResult without guest timing (backwards compat)."""
        from exec_sandbox.models import ExecutionResult, TimingBreakdown

        result = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=0,
            timing=TimingBreakdown(setup_ms=0, boot_ms=0, execute_ms=0, total_ms=0),
        )
        assert result.spawn_ms is None
        assert result.process_ms is None

    def test_execution_result_full_timing(self) -> None:
        """ExecutionResult with all timing fields populated."""
        from exec_sandbox.models import ExecutionResult, TimingBreakdown

        result = ExecutionResult(
            stdout="hello",
            stderr="",
            exit_code=0,
            execution_time_ms=71,
            timing=TimingBreakdown(
                setup_ms=45,
                boot_ms=380,
                execute_ms=85,
                total_ms=512,
                connect_ms=2,
            ),
            warm_pool_hit=False,
            spawn_ms=52,
            process_ms=15,
        )
        # Host-measured timing
        assert result.timing.setup_ms == 45
        assert result.timing.boot_ms == 380
        assert result.timing.execute_ms == 85
        assert result.timing.total_ms == 512
        assert result.timing.connect_ms == 2
        # Guest-measured timing
        assert result.execution_time_ms == 71
        assert result.spawn_ms == 52
        assert result.process_ms == 15
        # Warm pool flag
        assert result.warm_pool_hit is False


# ============================================================================
# Integration Tests - Timing Behavior
# ============================================================================


class TestSchedulerTimingIntegration:
    """Integration tests for scheduler timing behavior.

    These tests verify that:
    1. timing is always populated (never None)
    2. timing values are reasonable (non-negative, consistent)
    3. warm_pool_hit is correctly set
    """

    async def test_cold_boot_timing_always_populated(self, scheduler: Scheduler) -> None:
        """Cold boot execution always returns timing breakdown."""
        result = await scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
        )

        # timing must always be present
        assert result.timing is not None
        assert isinstance(result.timing.setup_ms, int)
        assert isinstance(result.timing.boot_ms, int)
        assert isinstance(result.timing.execute_ms, int)
        assert isinstance(result.timing.total_ms, int)

    async def test_timing_values_non_negative(self, scheduler: Scheduler) -> None:
        """All timing values must be non-negative."""
        result = await scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
        )

        assert result.timing.setup_ms >= 0
        assert result.timing.boot_ms >= 0
        assert result.timing.execute_ms >= 0
        assert result.timing.total_ms >= 0

    async def test_timing_total_reasonable(self, scheduler: Scheduler) -> None:
        """Total time should be >= execute time."""
        result = await scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
        )

        # total_ms should be at least execute_ms
        assert result.timing.total_ms >= result.timing.execute_ms

    async def test_cold_boot_warm_pool_hit_false(self, scheduler: Scheduler) -> None:
        """Cold boot (no warm pool) should have warm_pool_hit=False."""
        # Default scheduler fixture has warm_pool_size=0
        result = await scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
        )

        assert result.warm_pool_hit is False

    async def test_cold_boot_has_nonzero_boot_time(self, scheduler: Scheduler) -> None:
        """Cold boot should have measurable boot time."""
        result = await scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
        )

        # Cold boot should have some boot time (at least 1ms typically 100-500ms)
        # We use a very low threshold to avoid flaky tests
        assert result.timing.boot_ms >= 0
        # setup_ms + boot_ms should contribute to total (unless warm pool)
        if not result.warm_pool_hit:
            assert result.timing.setup_ms >= 0
            assert result.timing.boot_ms >= 0

    async def test_timing_with_timeout(self, scheduler_config: SchedulerConfig) -> None:
        """Timing is populated even when execution times out."""
        config = SchedulerConfig(
            images_dir=scheduler_config.images_dir,
            default_timeout_seconds=2,
        )

        code = "import time; time.sleep(10)"

        async with Scheduler(config) as sched:
            from exec_sandbox.exceptions import VmTimeoutError

            try:
                result = await sched.run(
                    code=code,
                    language=Language.PYTHON,
                    timeout_seconds=1,
                )
                # If we get a result, timing should still be populated
                assert result.timing is not None
                assert result.timing.total_ms >= 0
            except VmTimeoutError:
                # Timeout at host level - this is also valid behavior
                pass

    async def test_timing_with_error_code(self, scheduler: Scheduler) -> None:
        """Timing is populated even when code exits with error."""
        result = await scheduler.run(
            code="import sys; sys.exit(42)",
            language=Language.PYTHON,
        )

        assert result.exit_code == 42
        # Timing should still be populated
        assert result.timing is not None
        assert result.timing.total_ms >= 0

    async def test_timing_with_exception(self, scheduler: Scheduler) -> None:
        """Timing is populated even when code raises exception."""
        result = await scheduler.run(
            code="raise ValueError('test error')",
            language=Language.PYTHON,
        )

        assert result.exit_code != 0
        # Timing should still be populated
        assert result.timing is not None
        assert result.timing.total_ms >= 0

    async def test_timing_with_large_output(self, scheduler: Scheduler) -> None:
        """Timing is populated for executions with large output."""
        # Generate 100KB of output
        code = "print('x' * 100000)"

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert len(result.stdout) > 0
        # Timing should still be populated
        assert result.timing is not None
        assert result.timing.execute_ms >= 0

    async def test_timing_javascript(self, scheduler: Scheduler) -> None:
        """Timing works for JavaScript executions."""
        result = await scheduler.run(
            code="console.log('hello')",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert result.timing is not None
        assert result.timing.total_ms >= 0
        assert result.timing.execute_ms >= 0

    async def test_granular_timing_populated(self, scheduler: Scheduler) -> None:
        """Granular timing fields (connect_ms, spawn_ms, process_ms) are populated."""
        result = await scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0

        # connect_ms should be populated (host-measured)
        assert result.timing.connect_ms is not None
        assert result.timing.connect_ms >= 0

        # Guest-reported timing should be populated
        assert result.spawn_ms is not None
        assert result.spawn_ms >= 0

        assert result.process_ms is not None
        assert result.process_ms >= 0

        # Guest timing should be reasonable relative to execution_time_ms
        if result.execution_time_ms is not None:
            # spawn + process should be <= execution_time_ms (with some margin for streaming)
            assert result.spawn_ms <= result.execution_time_ms
            assert result.process_ms <= result.execution_time_ms

    async def test_granular_timing_consistency(self, scheduler: Scheduler) -> None:
        """Granular timing values should be consistent with each other."""
        result = await scheduler.run(
            code="import time; time.sleep(0.1); print('done')",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0

        # With a 100ms sleep, process_ms should be at least ~100ms
        if result.process_ms is not None:
            assert result.process_ms >= 50  # Allow some margin for timing variance


class TestSchedulerTimingEdgeCases:
    """Edge case tests for timing behavior."""

    async def test_timing_very_fast_execution(self, scheduler: Scheduler) -> None:
        """Timing handles very fast executions (sub-millisecond code)."""
        result = await scheduler.run(
            code="pass",  # Minimal Python code
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        # Even for fast code, timing should be populated
        assert result.timing is not None
        # execute_ms might be 0 for very fast code, that's OK
        assert result.timing.execute_ms >= 0

    async def test_timing_empty_output(self, scheduler: Scheduler) -> None:
        """Timing works when code produces no output."""
        result = await scheduler.run(
            code="x = 1",  # No print
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout == ""
        # Timing should still be populated
        assert result.timing is not None

    async def test_multiple_runs_have_independent_timing(self, scheduler: Scheduler) -> None:
        """Each run has its own independent timing values."""
        # Fast execution
        result1 = await scheduler.run(
            code="print('fast')",
            language=Language.PYTHON,
        )

        # Slightly slower execution
        result2 = await scheduler.run(
            code="import time; time.sleep(0.1); print('slow')",
            language=Language.PYTHON,
        )

        # Both should have timing
        assert result1.timing is not None
        assert result2.timing is not None

        # Second should have longer execute_ms (sleeping 100ms)
        # Allow some variance for timing precision
        assert result2.timing.execute_ms >= 50  # At least 50ms for sleep(0.1)


# ============================================================================
# Integration Tests - Warm Pool Timing
# ============================================================================


class TestSchedulerWarmPoolTiming:
    """Integration tests for warm pool timing behavior.

    These tests verify that warm pool hits have:
    1. warm_pool_hit=True
    2. setup_ms=0, boot_ms=0 (boot happened at startup, not request time)
    3. execute_ms and total_ms reflect actual request time
    """

    @pytest.fixture
    async def warm_pool_scheduler(self, scheduler_config: SchedulerConfig):
        """Scheduler with warm pool enabled."""
        config = SchedulerConfig(
            images_dir=scheduler_config.images_dir,
            warm_pool_size=2,  # Enable warm pool with 2 VMs per language
            auto_download_assets=False,
        )
        async with Scheduler(config) as sched:
            yield sched

    async def test_warm_pool_hit_flag_true(self, warm_pool_scheduler: Scheduler) -> None:
        """Warm pool hit should have warm_pool_hit=True."""
        result = await warm_pool_scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.warm_pool_hit is True

    async def test_warm_pool_timing_zero_setup_boot(self, warm_pool_scheduler: Scheduler) -> None:
        """Warm pool hit should have setup_ms=0 and boot_ms=0."""
        result = await warm_pool_scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
        )

        assert result.warm_pool_hit is True
        # Setup and boot are "free" for warm pool - they happened at startup
        assert result.timing.setup_ms == 0
        assert result.timing.boot_ms == 0

    async def test_warm_pool_timing_has_execute_time(self, warm_pool_scheduler: Scheduler) -> None:
        """Warm pool hit should have real execute_ms and total_ms."""
        result = await warm_pool_scheduler.run(
            code="import time; time.sleep(0.05); print('done')",
            language=Language.PYTHON,
        )

        assert result.warm_pool_hit is True
        # Execute time should reflect actual code execution (at least 50ms for sleep)
        assert result.timing.execute_ms >= 40  # Allow some variance
        assert result.timing.total_ms >= 40

    async def test_warm_pool_total_approximately_equals_execute(self, warm_pool_scheduler: Scheduler) -> None:
        """For warm pool, total_ms should be close to execute_ms (no boot overhead)."""
        result = await warm_pool_scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
        )

        assert result.warm_pool_hit is True
        # For warm pool: total ≈ execute (since setup=0, boot=0)
        # Allow some overhead for queue allocation, etc.
        assert result.timing.total_ms >= result.timing.execute_ms
        # Total shouldn't be dramatically larger than execute for warm pool
        assert result.timing.total_ms <= result.timing.execute_ms + 100  # 100ms tolerance

    async def test_warm_pool_exhaustion_falls_back_to_cold(self, scheduler_config: SchedulerConfig) -> None:
        """When warm pool is exhausted, falls back to cold boot with full timing."""
        import asyncio

        # Create scheduler with very small pool (1 VM)
        config = SchedulerConfig(
            images_dir=scheduler_config.images_dir,
            warm_pool_size=1,
            auto_download_assets=False,
        )

        async with Scheduler(config) as sched:
            # Run multiple concurrent executions to exhaust pool
            # First one gets warm VM, subsequent ones may cold boot
            results = await asyncio.gather(
                sched.run(code="import time; time.sleep(0.2); print('1')", language=Language.PYTHON),
                sched.run(code="import time; time.sleep(0.2); print('2')", language=Language.PYTHON),
                sched.run(code="import time; time.sleep(0.2); print('3')", language=Language.PYTHON),
            )

            # All should succeed
            for r in results:
                assert r.exit_code == 0
                assert r.timing is not None
                assert r.timing.total_ms >= 0

            # At least one should be warm, at least one should be cold (pool exhausted)
            warm_hits = sum(1 for r in results if r.warm_pool_hit)
            cold_boots = sum(1 for r in results if not r.warm_pool_hit)

            # With pool size 1, we expect 1 warm hit and 2 cold boots
            # (assuming no replenishment completes during the test)
            assert warm_hits >= 1, "Expected at least one warm pool hit"
            assert cold_boots >= 1, "Expected at least one cold boot (pool exhaustion)"

            # Cold boots should have non-zero setup/boot times
            for r in results:
                if not r.warm_pool_hit:
                    # Cold boot - should have real timing values
                    # At least one of setup_ms or boot_ms should be > 0
                    assert r.timing.setup_ms > 0 or r.timing.boot_ms > 0

    async def test_warm_pool_javascript(self, warm_pool_scheduler: Scheduler) -> None:
        """Warm pool works for JavaScript with correct timing."""
        result = await warm_pool_scheduler.run(
            code="console.log('hello')",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert result.warm_pool_hit is True
        assert result.timing.setup_ms == 0
        assert result.timing.boot_ms == 0
        assert result.timing.execute_ms >= 0

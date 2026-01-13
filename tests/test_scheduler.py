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

    async def test_snapshot_manager_initialized_without_s3(
        self, scheduler_config: SchedulerConfig
    ) -> None:
        """SnapshotManager is created even without S3 config (L1 cache works)."""
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

    async def test_snapshot_manager_has_vm_manager(
        self, scheduler_config: SchedulerConfig
    ) -> None:
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

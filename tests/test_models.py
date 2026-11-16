"""exec_sandbox.models: the result and timing models."""

import pytest
from pydantic import ValidationError

from exec_sandbox.models import ExecutionResult, StartupTiming, TimingBreakdown


class TestTimingModels:
    """Execution timing (per exec) and startup timing (per VM) are separate models."""

    def test_timing_breakdown_is_execution_only(self) -> None:
        timing = TimingBreakdown(execute_ms=50, connect_ms=2)
        assert (timing.execute_ms, timing.connect_ms) == (50, 2)
        with pytest.raises(ValidationError):
            TimingBreakdown(execute_ms=50)  # type: ignore[call-arg]  # connect_ms is required (None when not measured)

    def test_startup_timing_requires_every_field(self) -> None:
        """Built only by the library: a missing phase is a bug, never a silent default."""
        with pytest.raises(ValidationError):
            StartupTiming(total_ms=1, admission_ms=0, setup_ms=1, boot_ms=0)  # type: ignore[call-arg]

    def test_execution_result_has_no_startup_fields(self) -> None:
        result = ExecutionResult(
            stdout="hello", stderr="", exit_code=0, timing=TimingBreakdown(execute_ms=1, connect_ms=None)
        )
        assert not {"warm_pool_hit", "l1_cache_hit", "exposed_ports"} & set(result.model_dump())
        assert (result.spawn_ms, result.process_ms) == (None, None)

    def test_execution_result_timing_required(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionResult(stdout="hello", stderr="", exit_code=0)  # type: ignore[call-arg]

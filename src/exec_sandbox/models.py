"""Data models for exec-sandbox."""

from enum import Enum

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    RAW = "raw"


class TimingBreakdown(BaseModel):
    """Detailed timing breakdown for VM cold start and execution.

    All times are in milliseconds, measured from the host side.
    Follows Firecracker/AWS Lambda SnapStart conventions for phase separation.
    """

    setup_ms: int = Field(description="Resource setup time (overlay, cgroup, gvproxy - parallel)")
    boot_ms: int = Field(description="VM boot time (QEMU start + kernel + initramfs + guest-agent ready)")
    execute_ms: int = Field(description="Code execution time (connect + run + response)")
    total_ms: int = Field(description="Total end-to-end time (setup + boot + execute)")


class ExecutionResult(BaseModel):
    """Result from code execution inside microVM."""

    stdout: str = Field(max_length=1_000_000, description="Standard output (truncated at 1MB)")
    stderr: str = Field(max_length=100_000, description="Standard error (truncated at 100KB)")
    exit_code: int = Field(description="Process exit code (0=success)")
    execution_time_ms: int | None = Field(default=None, description="Execution time in ms (guest-reported)")
    external_cpu_time_ms: int | None = Field(default=None, description="CPU time in ms (host cgroup)")
    external_memory_peak_mb: int | None = Field(default=None, description="Peak memory in MB (host cgroup)")
    timing: TimingBreakdown | None = Field(default=None, description="Detailed timing breakdown (cold start only)")

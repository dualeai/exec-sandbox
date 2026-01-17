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

    Note: total_ms is independently measured end-to-end and may be slightly
    larger than setup_ms + boot_ms + execute_ms due to orchestration overhead
    (warm pool checks, semaphore acquisition, etc.).

    For warm pool hits, setup_ms and boot_ms are 0 since those costs were
    pre-paid at pool startup time.

    Granular boot timing note: The sum of (qemu_cmd_build_ms + gvproxy_start_ms +
    qemu_fork_ms + guest_wait_ms) will be ~20-30ms less than boot_ms due to
    unmeasured overhead between QEMU fork and guest wait (crash-check sleep,
    console log setup, VM registration, state transitions).
    """

    setup_ms: int = Field(description="Resource setup time (overlay, cgroup, gvproxy - parallel)")
    boot_ms: int = Field(description="VM boot time (QEMU start + kernel + initramfs + guest-agent ready)")
    execute_ms: int = Field(description="Code execution time (connect + run + response)")
    total_ms: int = Field(description="Total end-to-end time (setup + boot + execute)")
    connect_ms: int | None = Field(
        default=None,
        description="Time for channel.connect() in milliseconds (host-measured)",
    )
    # Granular setup timing (for tracing/profiling)
    overlay_ms: int | None = Field(
        default=None,
        description="Time for overlay acquisition (pool hit <1ms, on-demand 30-400ms)",
    )
    # Granular boot timing (for tracing/profiling)
    qemu_cmd_build_ms: int | None = Field(
        default=None,
        description="Time for pre-launch setup (command build, socket cleanup, channel creation)",
    )
    gvproxy_start_ms: int | None = Field(
        default=None,
        description="Time to start gvproxy (0 if network disabled)",
    )
    qemu_fork_ms: int | None = Field(
        default=None,
        description="Time for QEMU process fork/exec (subprocess creation)",
    )
    guest_wait_ms: int | None = Field(
        default=None,
        description="Time waiting for guest agent to become ready (kernel + initramfs + agent init)",
    )


class ExecutionResult(BaseModel):
    """Result from code execution inside microVM."""

    stdout: str = Field(max_length=1_000_000, description="Standard output (truncated at 1MB)")
    stderr: str = Field(max_length=100_000, description="Standard error (truncated at 100KB)")
    exit_code: int = Field(description="Process exit code (0=success)")
    execution_time_ms: int | None = Field(default=None, description="Execution time in ms (guest-reported)")
    external_cpu_time_ms: int | None = Field(default=None, description="CPU time in ms (host cgroup)")
    external_memory_peak_mb: int | None = Field(default=None, description="Peak memory in MB (host cgroup)")
    timing: TimingBreakdown = Field(description="Detailed timing breakdown (setup, boot, execute, total)")
    warm_pool_hit: bool = Field(default=False, description="True if VM was allocated from warm pool (instant start)")
    # Guest-reported granular timing (pass-through from guest agent)
    spawn_ms: int | None = Field(
        default=None,
        description="Time for process spawn (fork/exec) in milliseconds (guest-reported)",
    )
    process_ms: int | None = Field(
        default=None,
        description="Time from spawn to process exit in milliseconds (guest-reported)",
    )

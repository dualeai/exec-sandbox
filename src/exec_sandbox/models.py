"""Data models for exec-sandbox."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FileInfo(BaseModel):
    """File entry from directory listing in the sandbox.

    Returned by Session.list_files() and QemuVM.list_files().
    """

    name: str = Field(description="File or directory name")
    is_dir: bool = Field(description="True if entry is a directory")
    size: int = Field(description="File size in bytes (0 for directories)")


class PortMapping(BaseModel):
    """Port mapping configuration: internal (guest) → external (host).

    Used to specify which ports to expose from the guest VM to the host.

    Examples:
        # Expose guest port 8080 on host port 3000
        PortMapping(internal=8080, external=3000)

        # Expose guest port 5000 on a dynamically allocated host port
        PortMapping(internal=5000)  # external will be auto-assigned
    """

    internal: int = Field(ge=1, le=65535, description="Port inside VM (guest listens on)")
    external: int | None = Field(
        default=None,
        ge=1024,
        le=65535,
        description="Port on host (None=dynamic, auto-assigned by OS)",
    )
    protocol: Literal["tcp", "udp"] = Field(default="tcp", description="Transport protocol")


class ExposedPort(BaseModel):
    """Active port mapping after allocation.

    Represents a resolved port mapping with the actual allocated host port.
    """

    internal: int = Field(description="Port inside VM")
    external: int = Field(description="Allocated port on host")
    host: str = Field(default="127.0.0.1", description="Host bind address")
    protocol: Literal["tcp", "udp"] = Field(default="tcp", description="Transport protocol")

    @property
    def url(self) -> str:
        """Get HTTP URL for this exposed port."""
        return f"http://{self.host}:{self.external}"


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
    # L1 memory snapshot timing
    l1_restore_ms: int | None = Field(
        default=None,
        description="L1 memory snapshot restore time in milliseconds",
    )
    # Retry tracking (for CPU contention resilience)
    boot_retries: int = Field(
        default=0,
        description="Number of boot retries (0=first attempt succeeded or warm pool, 1+=retries needed due to CPU contention)",
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
    l1_cache_hit: bool = Field(default=False, description="VM restored from L1 memory snapshot")
    # Guest-reported granular timing (pass-through from guest agent)
    spawn_ms: int | None = Field(
        default=None,
        description="Time to acquire REPL process in milliseconds (guest-reported). Includes fork/exec on cold start, ~0ms on cache hit.",
    )
    process_ms: int | None = Field(
        default=None,
        description="Time from spawn to process exit in milliseconds (guest-reported)",
    )
    # Port forwarding: exposed ports accessible from host
    exposed_ports: list[ExposedPort] = Field(
        default_factory=lambda: [],  # noqa: PIE807 - lambda needed for pyright type inference
        description="List of ports exposed from guest to host (empty if no port forwarding configured)",
    )

"""Data models for exec-sandbox."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class FileInfo(BaseModel):
    """One entry of Session.list_files()."""

    name: str = Field(description="File or folder name")
    is_dir: bool = Field(description="True if the entry is a folder")
    size: int = Field(
        description="Bytes of the file: its length, or its allocated bytes when less (a sparse file); 0 for folders"
    )


class PortMapping(BaseModel):
    """A guest port (internal) to expose on a host port (external). Give it to VmSpec.expose_ports.

    Examples:
        # Expose guest port 8080 on host port 3000
        PortMapping(internal=8080, external=3000)

        # Expose guest port 5000 on a host port that the OS picks
        PortMapping(internal=5000)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    internal: int = Field(ge=1, le=65535, description="Port that the guest listens on")
    external: int | None = Field(
        default=None,
        ge=1024,
        le=65535,
        description="Port on the host (None: the OS picks one)",
    )
    protocol: Literal["tcp", "udp"] = Field(default="tcp", description="Transport protocol")


class ExposedPort(BaseModel):
    """A port mapping with its host port (for a dynamic port, the one the OS picked)."""

    internal: int = Field(description="Port in the VM")
    external: int = Field(description="Port on the host")
    host: str = Field(default="127.0.0.1", description="Host bind address")
    protocol: Literal["tcp", "udp"] = Field(default="tcp", description="Transport protocol")

    @computed_field
    @property
    def url(self) -> str:
        """HTTP URL of this exposed port (also in serialized output)."""
        return f"http://{self.host}:{self.external}"


class Language(StrEnum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    RAW = "raw"


class TimingBreakdown(BaseModel):
    """Host-measured time of one execution (Session.exec). Startup is in Session.startup_timing."""

    execute_ms: int = Field(description="Code execution time (connect + run + response)")
    connect_ms: int | None = Field(description="Time to connect to the guest agent (None when not measured)")


class StartupTiming(BaseModel):
    """How the session's VM started (Session.startup_timing). All times in ms, host-measured.

    total_ms covers the new() or reuse() call only; each exec() has its own
    time (ExecutionResult.timing.execute_ms).

    Phases do not overlap: total_ms = admission_ms + setup_ms + boot_ms
    (+ instance_load_ms after reuse()); setup_ms is the rest of total_ms. A warm pool hit
    has admission_ms = boot_ms = 0; an L1 restore has boot_ms = 0 and
    l1_restore_ms set; reuse() has boot_ms = 0 and setup_ms = the restore of the
    saved state.

    After a cold boot, the first exec() starts the interpreter: its execute_ms
    includes that start (about 0.2-0.7 s with KVM or HVF, 3-5 s under TCG) and
    can be longer than total_ms. A warm-pool or L1 VM has its REPL started.

    The cold-boot parts add up to about 20-30 ms less than boot_ms, because some
    steps are not measured.
    """

    model_config = ConfigDict(frozen=True)

    total_ms: int = Field(description="The whole new() or reuse() call (exec() calls not included)")
    admission_ms: int = Field(description="Wait for a resource slot (0 for warm pool hits)")
    setup_ms: int = Field(
        description="Cache lookup, overlay, cgroup (after reuse(): the restore of the saved state); "
        "excludes admission and boot"
    )
    boot_ms: int = Field(description="QEMU start + kernel + guest agent (0 for warm pool, L1 and reuse())")
    warm_pool_hit: bool = Field(description="The VM came from the warm pool")
    l1_cache_hit: bool = Field(description="The VM was restored from the L1 memory snapshot")
    l1_restore_ms: int | None = Field(description="L1 restore time (None unless l1_cache_hit)")
    overlay_ms: int = Field(description="Overlay acquisition (cold boot only, else 0)")
    qemu_cmd_build_ms: int = Field(description="QEMU command build (cold boot only, else 0)")
    gvproxy_start_ms: int = Field(
        description="gvproxy start up to its process (cold boot with network only; its ready wait is in guest_wait_ms)"
    )
    qemu_fork_ms: int = Field(description="QEMU process fork/exec (cold boot only)")
    guest_wait_ms: int = Field(description="Wait for the guest agent (cold boot only)")
    boot_retries: int = Field(description="Boot retries after CPU contention (0 = first try or no boot)")
    instance_load_ms: int | None = Field(
        description="reuse(): head read, host checks, download and rebuild of the saved state (None for new())"
    )


class ExecutionResult(BaseModel):
    """The result of one exec() call."""

    stdout: str = Field(description="Standard output (at most 1 MB, a limit of the guest)")
    stderr: str = Field(description="Standard error (at most 100 KB, a limit of the guest)")
    exit_code: int = Field(
        description="Process status 0-255 (0 = success, 128+N = usually killed by signal N), or -1 for a timeout"
    )
    execution_time_ms: int | None = Field(default=None, description="Execution time in ms (guest-reported)")
    external_cpu_time_ms: int | None = Field(default=None, description="CPU time in ms (host cgroup)")
    external_memory_peak_mb: int | None = Field(default=None, description="Peak memory in MB (host cgroup)")
    external_cpu_nr_throttled: int | None = Field(
        default=None,
        description="Number of CFS bandwidth throttle events (host cgroup cpu.stat nr_throttled)",
    )
    timing: TimingBreakdown = Field(description="Execution timing (startup is in Session.startup_timing)")
    # Guest-reported granular timing (pass-through from guest agent)
    spawn_ms: int | None = Field(
        default=None,
        description="Time to get the REPL process in milliseconds (guest-reported)",
    )
    process_ms: int | None = Field(
        default=None,
        description=(
            "Time from the code's write to the REPL to the end of its output, in milliseconds (guest-reported)"
        ),
    )


class InstanceRef(BaseModel):
    """A saved instance and its generation, returned by Session.save(). Keep the id and pass it to Scheduler.reuse()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance_id: str = Field(
        description="The id given to Scheduler.new(instance_id=...) or Session.save(instance_id=...), else "
        "one the library made at the first save (32 lowercase hex characters)"
    )
    generation: int = Field(description="1 after the first save; +1 for each later save by any caller")

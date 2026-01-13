"""Cgroup v2 and ulimit resource limiting utilities.

Provides:
- Cgroup setup, attachment, stats reading, and cleanup
- ulimit fallback for environments without cgroups (Docker Desktop, macOS)
- Graceful degradation when cgroups unavailable

References:
- Kernel cgroup v2 docs: https://docs.kernel.org/admin-guide/cgroup-v2.html
- pids.max limits both processes AND threads (goroutines in Go)
"""

import contextlib
from pathlib import Path
from typing import Final

import aiofiles
import aiofiles.os

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import VmError
from exec_sandbox.platform_utils import HostOS, detect_host_os

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

CGROUP_V2_BASE_PATH: Final[str] = "/sys/fs/cgroup"
"""Base path for cgroup v2 filesystem."""

CGROUP_APP_NAMESPACE: Final[str] = "code-exec"
"""Application cgroup namespace under /sys/fs/cgroup."""

CGROUP_MEMORY_OVERHEAD_MB: Final[int] = 200
"""QEMU process overhead added to guest memory for cgroup limits."""

TCG_TB_CACHE_SIZE_MB: Final[int] = 512
"""TCG translation block cache size in MB (must match tb-size in vm_manager.py).
QEMU 5.0+ defaults to 1GB, we use 512MB for better CI compatibility.
See: https://blueprints.launchpad.net/nova/+spec/control-qemu-tb-cache"""

CGROUP_PIDS_LIMIT: Final[int] = 100
"""Maximum PIDs in cgroup (fork bomb prevention).
Note: pids.max limits both processes AND threads, so this also limits goroutines."""

ULIMIT_MEMORY_MULTIPLIER: Final[int] = 14
"""Virtual memory multiplier for ulimit (guest_mb * 14 for TCG overhead)."""

ERRNO_READ_ONLY_FILESYSTEM: Final[int] = 30
"""errno for read-only filesystem (EROFS)."""

ERRNO_PERMISSION_DENIED: Final[int] = 13
"""errno for permission denied (EACCES)."""


# =============================================================================
# Availability Check
# =============================================================================


class _CgroupCache:
    """Cache for cgroup v2 availability check result."""

    def __init__(self) -> None:
        self.available: bool | None = None

    def reset(self) -> None:
        """Reset cache (for testing)."""
        self.available = None


_cgroup_cache = _CgroupCache()


def _check_cgroup_v2_mounted() -> bool:
    """Check if cgroup v2 filesystem is mounted and usable.

    Checks:
    1. /sys/fs/cgroup exists and is a directory
    2. cgroup.controllers file exists (cgroup v2 indicator)
    3. Not cgroup v1 (would have separate controllers like cpu, memory dirs)

    Returns:
        True if cgroup v2 is mounted and usable, False otherwise
    """
    # Return cached result if already checked
    if _cgroup_cache.available is not None:
        return _cgroup_cache.available

    cgroup_base = Path(CGROUP_V2_BASE_PATH)

    # Check 1: Base path exists and is a directory
    if not cgroup_base.is_dir():
        logger.debug("cgroup v2 not available: /sys/fs/cgroup is not a directory")
        _cgroup_cache.available = False
        return False

    # Check 2: cgroup.controllers exists (cgroup v2 unified hierarchy indicator)
    # In cgroup v1, this file doesn't exist at the root
    controllers_file = cgroup_base / "cgroup.controllers"
    if not controllers_file.exists():
        logger.debug("cgroup v2 not available: cgroup.controllers not found (likely cgroup v1)")
        _cgroup_cache.available = False
        return False

    # Check 3: Verify we can read controllers (not a permission issue)
    try:
        controllers = controllers_file.read_text().strip()
        # Should contain at least some controllers like "cpu memory pids"
        if not controllers:
            logger.warning("cgroup v2 mounted but no controllers enabled")
        else:
            logger.debug(f"cgroup v2 available with controllers: {controllers}")
    except (OSError, PermissionError) as e:
        logger.debug(f"cgroup v2 not available: cannot read controllers: {e}")
        _cgroup_cache.available = False
        return False

    _cgroup_cache.available = True
    return True


def is_cgroup_available(cgroup_path: Path | None) -> bool:
    """Check if cgroup_path is a usable cgroup v2 path.

    Performs multiple checks:
    1. Path is not None
    2. Path is under /sys/fs/cgroup (not a fallback dummy path)
    3. cgroup v2 filesystem is actually mounted and usable

    Args:
        cgroup_path: Path to check (None-safe)

    Returns:
        True if path is a valid cgroup v2 path and cgroups are available
    """
    # Check 1: Not None
    if cgroup_path is None:
        return False

    # Check 2: Path is under cgroup filesystem (not fallback like /tmp/cgroup-vm123)
    if not str(cgroup_path).startswith(CGROUP_V2_BASE_PATH):
        return False

    # Check 3: cgroup v2 is actually mounted and usable
    return _check_cgroup_v2_mounted()


# =============================================================================
# Setup
# =============================================================================


async def setup_cgroup(
    vm_id: str,
    tenant_id: str,
    memory_mb: int,
    use_tcg: bool = False,
) -> Path:
    """Set up cgroup v2 resource limits for a VM.

    Limits:
    - memory.max: guest_mb + overhead (+ TCG TB cache if software emulation)
    - cpu.max: 100000 (1 vCPU)
    - pids.max: 100 (fork bomb prevention, also limits goroutines)

    Args:
        vm_id: Unique VM identifier
        tenant_id: Tenant identifier
        memory_mb: Guest VM memory in MB
        use_tcg: True if using TCG software emulation (needs extra memory for TB cache)

    Returns:
        Path to cgroup directory (dummy path if cgroups unavailable)

    Note:
        Gracefully degrades to no resource limits on Docker Desktop (read-only /sys/fs/cgroup)
        or environments without cgroup v2 support.

        TCG mode requires significantly more memory due to the translation block (TB) cache.
        QEMU 5.0+ defaults to 1GB TB cache; we use 512MB (tb-size=512) but still need to
        account for this overhead. See: https://blueprints.launchpad.net/nova/+spec/control-qemu-tb-cache
    """
    tenant_cgroup = Path(f"{CGROUP_V2_BASE_PATH}/{CGROUP_APP_NAMESPACE}/{tenant_id}")
    cgroup_path = tenant_cgroup / vm_id

    try:
        # Create tenant cgroup and enable controllers for nested VM cgroups
        # In cgroup v2, subtree_control only affects immediate children,
        # so we must enable controllers at each level of the hierarchy
        await aiofiles.os.makedirs(tenant_cgroup, exist_ok=True)
        async with aiofiles.open(tenant_cgroup / "cgroup.subtree_control", "w") as f:
            await f.write("+memory +cpu +pids")

        # Create VM cgroup
        await aiofiles.os.makedirs(cgroup_path, exist_ok=True)

        # Calculate memory limit based on virtualization mode:
        # - KVM/HVF: guest_mb + process overhead (~200MB)
        # - TCG: guest_mb + TB cache (512MB) + process overhead (~200MB)
        # TCG needs the TB cache for JIT-compiled code translation blocks
        cgroup_memory_mb = memory_mb + CGROUP_MEMORY_OVERHEAD_MB
        if use_tcg:
            cgroup_memory_mb += TCG_TB_CACHE_SIZE_MB

        async with aiofiles.open(cgroup_path / "memory.max", "w") as f:
            await f.write(str(cgroup_memory_mb * 1024 * 1024))

        # Set CPU limit (1 vCPU)
        async with aiofiles.open(cgroup_path / "cpu.max", "w") as f:
            await f.write("100000 100000")

        # Set PID limit (fork bomb prevention)
        async with aiofiles.open(cgroup_path / "pids.max", "w") as f:
            await f.write(str(CGROUP_PIDS_LIMIT))

        # Verify cgroup.procs is writable (required for attaching processes)
        # Writing to control files (memory.max, etc.) requires different privileges
        # than writing to cgroup.procs, which needs proper systemd delegation
        async with aiofiles.open(cgroup_path / "cgroup.procs", "a") as f:
            pass  # Just test we can open for writing

    except OSError as e:
        # Gracefully degrade if cgroups unavailable (e.g., Docker Desktop, CI runners)
        # Note: PermissionError is a subclass of OSError
        if e.errno in (ERRNO_READ_ONLY_FILESYSTEM, ERRNO_PERMISSION_DENIED):
            logger.warning(
                "cgroup v2 unavailable, resource limits disabled",
                extra={"vm_id": vm_id, "path": str(cgroup_path), "errno": e.errno},
            )
            return Path(f"/tmp/cgroup-{vm_id}")  # noqa: S108
        raise VmError(f"Failed to setup cgroup: {e}") from e

    return cgroup_path


# =============================================================================
# Process Management
# =============================================================================


async def attach_to_cgroup(cgroup_path: Path, pid: int) -> None:
    """Attach process to cgroup.

    Args:
        cgroup_path: cgroup directory
        pid: Process ID to attach

    Raises:
        VmError: Failed to attach process
    """
    try:
        async with aiofiles.open(cgroup_path / "cgroup.procs", "w") as f:
            await f.write(str(pid))
    except (OSError, PermissionError) as e:
        raise VmError(f"Failed to attach PID {pid} to cgroup: {e}") from e


async def attach_if_available(cgroup_path: Path | None, pid: int | None) -> bool:
    """Attach process to cgroup if available.

    Convenience wrapper that handles None values and availability check.

    Args:
        cgroup_path: cgroup directory (may be dummy path if unavailable)
        pid: Process ID to attach (may be None if process failed to start)

    Returns:
        True if attached, False if cgroups unavailable or pid is None
    """
    if not is_cgroup_available(cgroup_path) or pid is None:
        return False
    await attach_to_cgroup(cgroup_path, pid)  # type: ignore[arg-type]
    return True


# =============================================================================
# Stats
# =============================================================================


async def read_cgroup_stats(cgroup_path: Path | None) -> tuple[int | None, int | None]:
    """Read external CPU time and peak memory from cgroup v2.

    Args:
        cgroup_path: cgroup directory path

    Returns:
        Tuple of (cpu_time_ms, peak_memory_mb)
        Returns (None, None) if cgroup not available or read fails
    """
    if not cgroup_path or not await aiofiles.os.path.exists(cgroup_path):
        return (None, None)

    cpu_time_ms: int | None = None
    peak_memory_mb: int | None = None

    try:
        # Read cpu.stat for usage_usec (microseconds)
        cpu_stat_file = cgroup_path / "cpu.stat"
        if await aiofiles.os.path.exists(cpu_stat_file):
            async with aiofiles.open(cpu_stat_file) as f:
                cpu_stat = await f.read()
            for line in cpu_stat.splitlines():
                if line.startswith("usage_usec"):
                    usage_usec = int(line.split()[1])
                    cpu_time_ms = usage_usec // 1000  # Convert to milliseconds
                    break

        # Read memory.peak for peak memory usage (bytes)
        memory_peak_file = cgroup_path / "memory.peak"
        if await aiofiles.os.path.exists(memory_peak_file):
            async with aiofiles.open(memory_peak_file) as f:
                peak_bytes = int((await f.read()).strip())
            peak_memory_mb = peak_bytes // (1024 * 1024)  # Convert to MB

    except (OSError, ValueError) as e:
        logger.debug(
            f"Failed to read cgroup stats: {e}",
            extra={"cgroup_path": str(cgroup_path)},
        )

    return (cpu_time_ms, peak_memory_mb)


# =============================================================================
# Cleanup
# =============================================================================


async def cleanup_cgroup(cgroup_path: Path | None, context_id: str) -> bool:
    """Remove cgroup directory after moving processes to parent.

    Per kernel docs (https://docs.kernel.org/admin-guide/cgroup-v2.html):
    A cgroup can only be removed when it has no children and no live processes.
    Writing "" to cgroup.procs does NOT work - each PID must be explicitly
    written to the parent's cgroup.procs file.

    Args:
        cgroup_path: Path to cgroup to remove (None safe - returns immediately)
        context_id: Context identifier for logging

    Returns:
        True if cgroup cleaned successfully, False if issues occurred
    """
    if cgroup_path is None:
        return True

    try:
        # For non-cgroup paths (fallback dummy), just try rmdir
        if not is_cgroup_available(cgroup_path):
            with contextlib.suppress(FileNotFoundError, OSError):
                await aiofiles.os.rmdir(cgroup_path)
            return True

        # Move all PIDs to parent cgroup first (required before rmdir)
        parent_procs = cgroup_path.parent / "cgroup.procs"
        procs_file = cgroup_path / "cgroup.procs"

        if await aiofiles.os.path.exists(parent_procs) and await aiofiles.os.path.exists(procs_file):
            async with aiofiles.open(procs_file) as f:
                pids = (await f.read()).strip().split("\n")

            for pid in pids:
                if pid:
                    try:
                        async with aiofiles.open(parent_procs, "w") as f:
                            await f.write(pid)
                    except (OSError, PermissionError):
                        # PID may have already exited
                        pass

        # Now safe to remove cgroup directory
        await aiofiles.os.rmdir(cgroup_path)
        logger.debug(
            "cgroup removed",
            extra={"context_id": context_id, "path": str(cgroup_path)},
        )
        return True

    except FileNotFoundError:
        # Already deleted (race condition) - success
        return True

    except OSError as e:
        # Directory not empty, permission denied, etc.
        logger.error(
            "cgroup removal error",
            extra={
                "context_id": context_id,
                "path": str(cgroup_path),
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        return False


# =============================================================================
# ulimit Fallback
# =============================================================================


ULIMIT_CPU_TIME_SECONDS: Final[int] = 3600
"""CPU time limit for ulimit fallback (1 hour safety net for long-running VMs)."""


def wrap_with_ulimit(cmd: list[str], memory_mb: int) -> list[str]:
    """Wrap command with ulimit for resource control (cgroups alternative).

    Used as fallback when cgroups are unavailable (Docker Desktop, macOS).

    Platform-specific limits:
    - Linux: -v (virtual memory), -t (CPU time), -u (max processes)
    - macOS: -u (max processes) only - virtual memory not supported by kernel,
             and -t (CPU time) breaks subprocess stdout pipe

    Args:
        cmd: Original command
        memory_mb: Memory limit in MB

    Returns:
        Command wrapped with ulimit via bash -c (bash required for -u support)
    """
    import shlex  # noqa: PLC0415

    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)

    # Memory overhead: ~14x guest memory for TCG worst case
    virtual_mem_kb = memory_mb * 1024 * ULIMIT_MEMORY_MULTIPLIER

    # Platform-specific limits based on kernel support
    if detect_host_os() == HostOS.MACOS:
        # macOS: Use process limit (-u) only
        # - Virtual memory (-v) not supported by macOS kernel (setrlimit fails)
        # - CPU time (-t) breaks subprocess stdout pipe on macOS (QEMU output lost)
        # Note: -u requires bash (POSIX sh doesn't support it)
        shell_cmd = f"ulimit -u {CGROUP_PIDS_LIMIT} && exec {cmd_str}"
    else:
        # Linux: Full resource limits
        # - Virtual memory (-v) is the primary memory control
        # - CPU time (-t) and processes (-u) as safety nets
        shell_cmd = f"ulimit -v {virtual_mem_kb} && ulimit -t {ULIMIT_CPU_TIME_SECONDS} && ulimit -u {CGROUP_PIDS_LIMIT} && exec {cmd_str}"

    return ["bash", "-c", shell_cmd]

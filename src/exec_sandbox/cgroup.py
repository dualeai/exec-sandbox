"""Cgroup v2 and ulimit resource limiting utilities.

Provides:
- Cgroup setup, attachment, stats reading, and cleanup
- ulimit fallback for environments without cgroups (Docker Desktop, macOS)
- Graceful degradation when cgroups unavailable

Delegation model
----------------
The orchestrator process must live inside the delegated cgroup subtree
**before** spawning VMs.  cgroupv2 PID migration (writing a PID to
``cgroup.procs``) requires two permission checks:

1. VFS write permission on the **destination** ``cgroup.procs`` — checked
   at ``open()`` time by the normal VFS path.
2. Write permission on the ``cgroup.procs`` of the **common ancestor** of
   the source and destination cgroups — checked at ``write()`` time by the
   kernel's ``cgroup_procs_write_permission()`` using the credentials
   captured at open.

When the orchestrator is in ``code-exec/runner/`` and the VM cgroup is
``code-exec/exec-sandbox/session-xxx/``, the common ancestor is
``code-exec/`` — owned by the runner user, so both checks pass without
root privileges.

If the orchestrator is outside the subtree (e.g. in a system slice), the
common ancestor is ``/sys/fs/cgroup/`` (root-owned) and check 2 fails
with EACCES.  The CI workflow handles this by moving the runner process
into the subtree via ``sudo`` before any sandboxed execution.

References:
- Kernel cgroup v2 docs: https://docs.kernel.org/admin-guide/cgroup-v2.html
- pids.max limits both processes AND threads (goroutines in Go)
"""

import contextlib
import shlex
from pathlib import Path
from typing import Final

import aiofiles
import aiofiles.os

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import VmDependencyError
from exec_sandbox.platform_utils import HostOS, detect_host_os

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

CGROUP_V2_BASE_PATH: Final[str] = "/sys/fs/cgroup"
"""Base path for cgroup v2 filesystem."""

CGROUP_APP_NAMESPACE: Final[str] = "code-exec"
"""Application cgroup namespace under /sys/fs/cgroup."""

CGROUP_PIDS_LIMIT: Final[int] = 256
"""Maximum PIDs in cgroup (fork bomb prevention).

256 provides ample headroom for QEMU + gvproxy threads in both aio=io_uring
(~38 peak) and aio=threads (~102 peak) modes, while still preventing fork
bombs (thousands of processes).  Guest code runs inside the VM and does NOT
count toward this host cgroup limit.

Note: pids.max limits both processes AND threads (including goroutines)."""

CGROUP_CPU_PERIOD_US: Final[int] = 100_000
"""CFS bandwidth period in microseconds (100ms). Standard Linux CFS period."""

ULIMIT_MEMORY_MULTIPLIER: Final[int] = 14
"""Virtual memory multiplier for ulimit (guest_mb * 14 for TCG overhead)."""

ERRNO_READ_ONLY_FILESYSTEM: Final[int] = 30
"""errno for read-only filesystem (EROFS)."""

ERRNO_PERMISSION_DENIED: Final[int] = 13
"""errno for permission denied (EACCES)."""

ERRNO_OPERATION_NOT_PERMITTED: Final[int] = 1
"""errno for operation not permitted (EPERM)."""


# cgroup v1 "unlimited" sentinel: values at or above this threshold are treated
# as "no limit set".  The kernel uses PAGE_COUNTER_MAX which on 64-bit is
# 2^63 / PAGE_SIZE ≈ 9.2e18 / 4096 ≈ 2.25e15.  Checking > 2^62 safely covers
# both raw byte values (memory.limit_in_bytes) and page-counter representations.
_CGROUP_V1_UNLIMITED_THRESHOLD: Final[int] = 2**62


# =============================================================================
# Self Cgroup Resolution
# =============================================================================


def resolve_self_cgroup_v2() -> str | None:
    """Return the current process's cgroup v2 relative path, or None.

    Parses ``/proc/self/cgroup`` which may contain multiple lines on hybrid
    v1+v2 systems (e.g. RHEL 8).  The cgroup v2 entry is always ``0::/path``.

    Returns:
        Relative cgroup path (e.g. ``/system.slice/docker-xxx.scope``),
        or None if not on Linux, not cgroup v2, or ``/proc`` is unavailable.
    """
    if detect_host_os() != HostOS.LINUX:
        return None

    try:
        content = Path("/proc/self/cgroup").read_text()
    except (OSError, ValueError):
        return None

    for line in content.splitlines():
        parts = line.split(":")
        if len(parts) == 3 and parts[0] == "0":  # noqa: PLR2004
            return parts[2]
    return None


# =============================================================================
# Container Cgroup Detection
# =============================================================================


def detect_cgroup_memory_limit_mb(
    cgroup_v2_base: str | None = None,
) -> float | None:
    """Detect the container cgroup memory limit.

    Priority: cgroup v2 ``memory.max`` → cgroup v1
    ``memory.limit_in_bytes`` → None (fall through to psutil).

    All reads are synchronous (tiny procfs files, no I/O).

    Args:
        cgroup_v2_base: Override base path for testing (default: ``/sys/fs/cgroup``).

    Returns:
        Memory limit in MB, or None if no cgroup limit is set.
    """
    base = cgroup_v2_base or CGROUP_V2_BASE_PATH

    # --- cgroup v2 ---
    v2_path = Path(base) / "memory.max"
    try:
        raw = v2_path.read_text().strip()
        if raw != "max":
            limit_bytes = int(raw)
            limit_mb = limit_bytes / (1024 * 1024)
            logger.info(
                "Detected cgroup v2 memory limit",
                extra={"limit_mb": round(limit_mb), "path": str(v2_path)},
            )
            return limit_mb
    except (FileNotFoundError, OSError, ValueError):
        pass

    # --- cgroup v1 ---
    v1_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    try:
        limit_bytes = int(v1_path.read_text().strip())
        if limit_bytes < _CGROUP_V1_UNLIMITED_THRESHOLD:
            limit_mb = limit_bytes / (1024 * 1024)
            logger.info(
                "Detected cgroup v1 memory limit",
                extra={"limit_mb": round(limit_mb), "path": str(v1_path)},
            )
            return limit_mb
    except (FileNotFoundError, OSError, ValueError):
        pass

    return None


def detect_cgroup_cpu_limit(
    cgroup_v2_base: str | None = None,
) -> float | None:
    """Detect the container cgroup CPU limit (in logical CPUs).

    Priority: cgroup v2 ``cpu.max`` → cgroup v1
    ``cpu.cfs_quota_us / cpu.cfs_period_us`` → None.

    Args:
        cgroup_v2_base: Override base path for testing.

    Returns:
        CPU limit as a float (e.g. 2.0 = 2 CPUs), or None if unlimited.
    """
    base = cgroup_v2_base or CGROUP_V2_BASE_PATH

    # --- cgroup v2: "quota period" e.g. "200000 100000" ---
    v2_path = Path(base) / "cpu.max"
    try:
        parts = v2_path.read_text().strip().split()
        if len(parts) == 2 and parts[0] != "max":  # noqa: PLR2004
            quota = int(parts[0])
            period = int(parts[1])
            if period > 0:
                cpus = quota / period
                logger.info(
                    "Detected cgroup v2 CPU limit",
                    extra={"cpus": round(cpus, 2), "path": str(v2_path)},
                )
                return cpus
    except (FileNotFoundError, OSError, ValueError):
        pass

    # --- cgroup v1: cpu.cfs_quota_us / cpu.cfs_period_us ---
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        if quota == -1:
            return None  # unlimited
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if period > 0:
            cpus = quota / period
            logger.info(
                "Detected cgroup v1 CPU limit",
                extra={"cpus": round(cpus, 2)},
            )
            return cpus
    except (FileNotFoundError, OSError, ValueError):
        pass

    return None


def read_container_available_memory_mb(
    cgroup_v2_base: str | None = None,
) -> float | None:
    """Read available memory inside a container (cgroup limit minus current usage).

    Uses ``memory.max - memory.current`` (cgroup v2) or
    ``memory.limit_in_bytes - memory.usage_in_bytes`` (cgroup v1).

    Args:
        cgroup_v2_base: Override base path for testing.

    Returns:
        Available memory in MB, or None if not in a cgroup.
    """
    base = cgroup_v2_base or CGROUP_V2_BASE_PATH

    # --- cgroup v2 ---
    try:
        max_raw = Path(base, "memory.max").read_text().strip()
        if max_raw != "max":
            limit = int(max_raw)
            current = int(Path(base, "memory.current").read_text().strip())
            return max(0, limit - current) / (1024 * 1024)
    except (FileNotFoundError, OSError, ValueError):
        pass

    # --- cgroup v1 ---
    try:
        limit = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip())
        if limit < _CGROUP_V1_UNLIMITED_THRESHOLD:
            usage = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip())
            return max(0, limit - usage) / (1024 * 1024)
    except (FileNotFoundError, OSError, ValueError):
        pass

    return None


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
# Internal Helpers
# =============================================================================


async def _write_cgroup_optional(path: Path, value: str) -> bool:
    """Write to a cgroup control file, silently ignoring if unsupported.

    Used for optional cgroup features (memory.high, memory.oom.group,
    memory.zswap.writeback) that may not exist on older kernels.

    Returns:
        True if written successfully, False if file doesn't exist or write failed.
    """
    try:
        async with aiofiles.open(path, "w") as f:
            await f.write(value)
        return True
    except (FileNotFoundError, OSError):
        return False


# =============================================================================
# Setup
# =============================================================================


async def setup_cgroup(
    vm_id: str,
    tenant_id: str,
    cgroup_memory_mb: int,
    cgroup_cpu_cores: float,
) -> Path:
    """Set up cgroup v2 resource limits for a VM.

    Applies pre-computed effective limits from the admission controller.
    The admission controller is the single source of truth for overhead
    calculations (memory overhead, TCG TB cache, CPU overhead).

    Limits:
    - memory.max: cgroup_memory_mb (effective limit, pre-computed by admission)
    - cpu.max: quota proportional to cgroup_cpu_cores (effective, includes overhead)
    - pids.max: 256 (fork bomb prevention, also limits goroutines)

    Args:
        vm_id: Unique VM identifier
        tenant_id: Tenant identifier
        cgroup_memory_mb: Effective memory limit in MB (guest + overhead, pre-computed by admission)
        cgroup_cpu_cores: Effective CPU limit (guest + overhead, pre-computed by admission)

    Returns:
        Path to cgroup directory (dummy path if cgroups unavailable)

    Note:
        Gracefully degrades to no resource limits on Docker Desktop (read-only /sys/fs/cgroup)
        or environments without cgroup v2 support.
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

        # Apply effective memory limit (pre-computed by admission controller)
        memory_max_bytes = cgroup_memory_mb * 1024 * 1024
        async with aiofiles.open(cgroup_path / "memory.max", "w") as f:
            await f.write(str(memory_max_bytes))

        # Soft memory limit at 85% of max — triggers kernel reclaim before OOM.
        # Creates a buffer zone: between high and max, the kernel aggressively
        # reclaims (page cache eviction, zswap compression) without killing.
        memory_high_bytes = int(memory_max_bytes * 0.85)
        await _write_cgroup_optional(cgroup_path / "memory.high", str(memory_high_bytes))

        # Kill entire cgroup as a unit on OOM (QEMU + gvproxy together).
        # Without this, OOM may kill only gvproxy, leaving QEMU orphaned.
        await _write_cgroup_optional(cgroup_path / "memory.oom.group", "1")

        # Prevent zswap from writing back to disk swap — keeps VM latency
        # predictable.  Without this, under host pressure zswap may evict
        # compressed pages to disk, causing multi-ms stalls.
        await _write_cgroup_optional(cgroup_path / "memory.zswap.writeback", "0")

        # Apply effective CPU limit (pre-computed by admission controller)
        cpu_quota = int(CGROUP_CPU_PERIOD_US * cgroup_cpu_cores)
        async with aiofiles.open(cgroup_path / "cpu.max", "w") as f:
            await f.write(f"{cpu_quota} {CGROUP_CPU_PERIOD_US}")

        # Set PID limit (fork bomb prevention)
        async with aiofiles.open(cgroup_path / "pids.max", "w") as f:
            await f.write(str(CGROUP_PIDS_LIMIT))

        # Sanity-check that we can open cgroup.procs for writing.
        # This validates the VFS permission (check 1 in module docstring) but
        # NOT the common-ancestor check (check 2), which fires at write() time
        # inside the kernel's cgroup_attach_task().  If the orchestrator is
        # outside the delegated subtree, this passes but attach_to_cgroup()
        # fails later — attach_if_available() handles that gracefully.
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
        raise VmDependencyError(f"Failed to setup cgroup: {e}") from e

    return cgroup_path


# =============================================================================
# Process Management
# =============================================================================


async def attach_to_cgroup(cgroup_path: Path, pid: int) -> None:
    """Attach process to cgroup.

    Writes *pid* to ``cgroup_path/cgroup.procs``.  The kernel checks
    write permission on the **common ancestor's** ``cgroup.procs`` at
    write time (see module docstring for the full delegation model).
    If the orchestrator lives inside the delegated subtree, the common
    ancestor is runner-owned and no root privileges are needed.

    Args:
        cgroup_path: cgroup directory
        pid: Process ID to attach

    Raises:
        VmDependencyError: Failed to attach process (e.g. EACCES when
            the orchestrator is outside the delegated subtree)
    """
    try:
        async with aiofiles.open(cgroup_path / "cgroup.procs", "w") as f:
            await f.write(str(pid))
    except OSError as e:  # includes PermissionError (EACCES)
        raise VmDependencyError(f"Failed to attach PID {pid} to cgroup: {e}") from e


async def attach_if_available(cgroup_path: Path | None, pid: int | None) -> bool:
    """Attach process to cgroup if available, gracefully degrading on failure.

    Convenience wrapper that handles None values, availability check,
    and permission errors.  The latter can occur when the orchestrator
    process is outside the delegated cgroup subtree (e.g. sandboxed CI
    runners like CodSpeed/valgrind where the initial ``sudo`` move into
    the subtree was not possible).  In that case resource limits are not
    enforced for this VM, but execution continues.

    Args:
        cgroup_path: cgroup directory (may be dummy path if unavailable)
        pid: Process ID to attach (may be None if process failed to start)

    Returns:
        True if attached, False if cgroups unavailable, pid is None,
        or attachment failed
    """
    if not is_cgroup_available(cgroup_path) or pid is None:
        return False
    try:
        await attach_to_cgroup(cgroup_path, pid)  # type: ignore[arg-type]
        return True
    except VmDependencyError as e:
        # Only degrade gracefully for permission errors (delegation not
        # possible).  Other errors (ENOENT, EINVAL) indicate a real fault
        # and should propagate.
        cause = e.__cause__
        if isinstance(cause, OSError) and cause.errno in (
            ERRNO_PERMISSION_DENIED,
            ERRNO_OPERATION_NOT_PERMITTED,
        ):
            logger.warning(
                "cgroup attachment failed, resource limits not enforced for this VM",
                extra={
                    "cgroup_path": str(cgroup_path),
                    "pid": pid,
                    "errno": cause.errno,
                },
            )
            return False
        raise


# =============================================================================
# Stats
# =============================================================================


async def read_cgroup_stats(cgroup_path: Path | None) -> tuple[int | None, int | None, int | None]:
    """Read external CPU time, peak memory, and CPU throttle count from cgroup v2.

    Args:
        cgroup_path: cgroup directory path

    Returns:
        Tuple of (cpu_time_ms, peak_memory_mb, nr_throttled)
        Returns (None, None, None) if cgroup not available or read fails
    """
    if not cgroup_path or not await aiofiles.os.path.exists(cgroup_path):
        return (None, None, None)

    cpu_time_ms: int | None = None
    peak_memory_mb: int | None = None
    nr_throttled: int | None = None

    try:
        # Read cpu.stat for usage_usec (microseconds) and nr_throttled
        cpu_stat_file = cgroup_path / "cpu.stat"
        if await aiofiles.os.path.exists(cpu_stat_file):
            async with aiofiles.open(cpu_stat_file) as f:
                cpu_stat = await f.read()
            for line in cpu_stat.splitlines():
                if line.startswith("usage_usec"):
                    usage_usec = int(line.split()[1])
                    cpu_time_ms = usage_usec // 1000  # Convert to milliseconds
                elif line.startswith("nr_throttled "):
                    nr_throttled = int(line.split()[1])

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

    return (cpu_time_ms, peak_memory_mb, nr_throttled)


# =============================================================================
# Memory Reclaim
# =============================================================================


async def reclaim_memory(cgroup_path: Path | None, amount_mb: int = 32) -> bool:
    """Proactively reclaim memory from a VM's cgroup via memory.reclaim.

    Triggers the kernel to compress anonymous pages into zswap/zram, reducing
    RSS without killing the process.  This is the mechanism Meta uses in TMO
    (Transparent Memory Offloading) for 20-32% fleet-wide memory savings.

    Requires Linux >= 6.9 for the ``swappiness`` parameter.  Falls back to
    plain reclaim on older kernels, and is a no-op on macOS or when cgroups
    are unavailable.

    Args:
        cgroup_path: VM cgroup directory
        amount_mb: Amount of memory to reclaim (default 32 MB)

    Returns:
        True if reclaim was issued, False if unavailable or failed
    """
    if not is_cgroup_available(cgroup_path) or cgroup_path is None:
        return False

    reclaim_file = cgroup_path / "memory.reclaim"
    amount_bytes = amount_mb * 1024 * 1024

    try:
        # Try with swappiness=200 first (kernel 6.9+, aggressively compress anon pages)
        async with aiofiles.open(reclaim_file, "w") as f:
            await f.write(f"{amount_bytes} swappiness=200")
        return True
    except OSError:
        pass

    try:
        # Fall back to plain reclaim (kernel 6.1+)
        async with aiofiles.open(reclaim_file, "w") as f:
            await f.write(str(amount_bytes))
        return True
    except OSError as e:
        logger.debug(
            "memory.reclaim unavailable",
            extra={"cgroup_path": str(cgroup_path), "error": str(e)},
        )
        return False


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


def wrap_with_ulimit(cmd: list[str], cgroup_memory_mb: int) -> list[str]:
    """Wrap command with ulimit for resource control (cgroups alternative).

    Used as fallback when cgroups are unavailable (Docker Desktop, macOS).

    Platform-specific limits:
    - Linux: -v (virtual memory), -t (CPU time)
    - macOS: no limits (virtual memory not supported, CPU time breaks pipes)

    Note: ``ulimit -u`` (RLIMIT_NPROC) is intentionally NOT set here.
    Unlike cgroup ``pids.max`` (which is per-cgroup, per-VM), RLIMIT_NPROC is
    per-UID: ALL processes under the same UID share a single counter.  Setting
    it to 256 (the per-VM cgroup pids limit) caused ``qemu_thread_create:
    Resource temporarily unavailable`` at ~23 concurrent VMs because
    23 QEMUs x 12 threads = 276 > 256.  The system's default RLIMIT_NPROC
    (typically 126K+ on Linux, 2784 on macOS) already prevents fork bombs.

    Args:
        cmd: Original command
        cgroup_memory_mb: Effective memory limit in MB (guest + overhead, pre-computed by admission)

    Returns:
        Command wrapped with ulimit via bash -c (bash required for -u support)
    """
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)

    # Memory overhead: ~14x effective memory for TCG worst case
    # ULIMIT_MEMORY_MULTIPLIER accounts for TCG virtual address space expansion
    # (separate from cgroup overhead which is already included in cgroup_memory_mb)
    virtual_mem_kb = cgroup_memory_mb * 1024 * ULIMIT_MEMORY_MULTIPLIER

    # Platform-specific limits based on kernel support
    if detect_host_os() == HostOS.MACOS:
        # macOS: No ulimit controls available
        # - Virtual memory (-v) not supported by macOS kernel (setrlimit fails)
        # - CPU time (-t) breaks subprocess stdout pipe on macOS (QEMU output lost)
        # - Process limit (-u / RLIMIT_NPROC) is per-UID, not per-process — cannot
        #   provide per-VM isolation. Rely on system default.
        shell_cmd = f"exec {cmd_str}"
    else:
        # Linux: Virtual memory and CPU time limits only
        # - Virtual memory (-v) is the primary memory control
        # - CPU time (-t) as safety net for runaway VMs
        # - No -u (RLIMIT_NPROC): per-UID, not per-process — see docstring
        shell_cmd = f"ulimit -v {virtual_mem_kb} && ulimit -t {ULIMIT_CPU_TIME_SECONDS} && exec {cmd_str}"

    return ["bash", "-c", shell_cmd]

"""Cgroup v2 and ulimit resource limiting utilities.

Provides:
- Cgroup: one cgroup that this library makes (a VM's): create, join or attach, stats,
  reclaim, kill, remove; None where the host gives no cgroup (graceful degradation)
- the memory and CPU limits and the free memory of the cgroup at the mount root (a
  container's own), for admission; and cgroup_v2_available
- ulimit fallback for environments without cgroups (Docker Desktop, macOS)

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

from __future__ import annotations

import asyncio
import contextlib
import errno
import functools
import os
import select
import shlex
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

import psutil

from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import await_cancellation_safe, run_to_end
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.files.disk import make_dirs
from exec_sandbox.platform_utils import HostOS, child_env, detect_host_os

if TYPE_CHECKING:
    from collections.abc import Sequence

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
"""Virtual memory multiplier of the ulimit fallback: the effective memory (guest + overhead)
times 14, for the address space of TCG."""

CGROUP_DRAIN_TIMEOUT_SECONDS: Final[float] = 2.0
"""How long a VM destroy waits for its cgroup to empty after cgroup.kill. SIGKILL ends
each process at once; the last one (QEMU's async-teardown process) first unmaps the
guest RAM, which takes milliseconds to some hundreds of ms."""


# cgroup v1 "unlimited" sentinel: values at or above this threshold are treated
# as "no limit set".  memory.limit_in_bytes shows "no limit" as PAGE_COUNTER_MAX
# pages in bytes: 2^63 rounded down to a page on 64-bit (9223372036854771712), above 2^62.
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


def detect_cgroup_memory_limit_mb() -> float | None:
    """Detect the container cgroup memory limit.

    Priority: cgroup v2 ``memory.max`` → cgroup v1
    ``memory.limit_in_bytes`` → None (fall through to psutil).

    All reads are synchronous (small cgroupfs files: no disk I/O).

    Returns:
        Memory limit in MB, or None if no cgroup limit is set.
    """
    # --- cgroup v2 ---
    v2_path = Path(CGROUP_V2_BASE_PATH) / "memory.max"
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
    except (OSError, ValueError):
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
    except (OSError, ValueError):
        pass

    return None


def detect_cgroup_cpu_limit() -> float | None:
    """Detect the container cgroup CPU limit (in logical CPUs).

    Priority: cgroup v2 ``cpu.max`` → cgroup v1
    ``cpu.cfs_quota_us / cpu.cfs_period_us`` → None.

    Returns:
        CPU limit as a float (e.g. 2.0 = 2 CPUs), or None if unlimited.
    """
    # --- cgroup v2: "quota period" e.g. "200000 100000" ---
    v2_path = Path(CGROUP_V2_BASE_PATH) / "cpu.max"
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
    except (OSError, ValueError):
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
    except (OSError, ValueError):
        pass

    return None


def read_container_available_memory_mb() -> float | None:
    """Read available memory inside a container (cgroup limit minus current usage).

    Uses ``memory.max - memory.current`` (cgroup v2) or
    ``memory.limit_in_bytes - memory.usage_in_bytes`` (cgroup v1).

    Returns:
        Available memory in MB, or None if not in a cgroup.
    """
    # --- cgroup v2 ---
    try:
        max_raw = Path(CGROUP_V2_BASE_PATH, "memory.max").read_text().strip()
        if max_raw != "max":
            limit = int(max_raw)
            current = int(Path(CGROUP_V2_BASE_PATH, "memory.current").read_text().strip())
            return max(0, limit - current) / (1024 * 1024)
    except (OSError, ValueError):
        pass

    # --- cgroup v1 ---
    try:
        limit = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip())
        if limit < _CGROUP_V1_UNLIMITED_THRESHOLD:
            usage = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip())
            return max(0, limit - usage) / (1024 * 1024)
    except (OSError, ValueError):
        pass

    return None


# =============================================================================
# Availability Check
# =============================================================================


@functools.cache
def cgroup_v2_available() -> bool:
    """Whether the cgroup v2 filesystem is mounted and usable (cached; tests clear it with
    cgroup_v2_available.cache_clear()).

    Checks:
    1. /sys/fs/cgroup exists and is a directory
    2. cgroup.controllers file exists (cgroup v2 indicator; cgroup v1 has no such file at the root)
    3. cgroup.controllers can be read (an empty list is logged, not refused)

    Returns:
        True if cgroup v2 is mounted and usable, False otherwise
    """
    cgroup_base = Path(CGROUP_V2_BASE_PATH)

    # Check 1: Base path exists and is a directory
    if not cgroup_base.is_dir():
        logger.debug("cgroup v2 not available: /sys/fs/cgroup is not a directory")
        return False

    # Check 2: cgroup.controllers exists (cgroup v2 unified hierarchy indicator)
    # In cgroup v1, this file doesn't exist at the root
    controllers_file = cgroup_base / "cgroup.controllers"
    if not controllers_file.exists():
        logger.debug("cgroup v2 not available: cgroup.controllers not found (likely cgroup v1)")
        return False

    # Check 3: Verify we can read controllers (not a permission issue)
    try:
        controllers = controllers_file.read_text().strip()
        # Should contain at least some controllers like "cpu memory pids"
        if not controllers:
            logger.warning("cgroup v2 mounted but no controllers enabled")
        else:
            logger.debug(f"cgroup v2 available with controllers: {controllers}")
    except OSError as e:  # PermissionError too
        logger.debug(f"cgroup v2 not available: cannot read controllers: {e}")
        return False

    return True


# =============================================================================
# Internal Helpers
# =============================================================================


_SWAPPINESS: Final = Path("/proc/sys/vm/swappiness")


def _cgroup_can_swap(cgroup_path: Path) -> bool:
    """Whether anon pages of a cgroup at `cgroup_path` can go to swap: the host has swap,
    vm.swappiness is not 0 (cgroup v2 reclaim is then file-only), and no cgroup above it up
    to the mount root sets memory.swap.max to 0. A container started without swap sees the
    host's swap (psutil.swap_memory), but its cgroup namespace root (the mount root, as the
    container sees it) forbids swap."""
    if not _host_has_swap() or _swappiness() == 0:
        return False
    base = Path(CGROUP_V2_BASE_PATH)
    for folder in cgroup_path.parents:
        if folder != base and base not in folder.parents:
            break  # above the mount root
        try:
            if (folder / "memory.swap.max").read_text().strip() == "0":
                return False
        except OSError:
            continue  # no swap controller file at this level
    return True


def _swappiness() -> int | None:
    """vm.swappiness, or None if it cannot be read."""
    try:
        return int(_SWAPPINESS.read_text().strip())
    except (OSError, ValueError):
        return None


def _host_has_swap() -> bool:
    """Whether the host has swap (psutil.swap_memory: SwapTotal > 0): where anon pages above a
    cgroup's memory.high can go."""
    try:
        return psutil.swap_memory().total > 0
    except OSError:  # no /proc/meminfo
        return False


def _write_cgroup_optional(path: Path, value: str) -> None:
    """Write to a cgroup control file, silently ignoring if unsupported.

    Used for optional cgroup features (memory.high, memory.oom.group) that may
    not exist on older kernels.
    """
    with contextlib.suppress(OSError):
        path.write_text(value)


# =============================================================================
# Setup
# =============================================================================


CGROUP_FOLDER_MODE: Final[int] = 0o755
"""Mode of each cgroup folder this module makes: no group or other write. Under `sg qemu-vm`
(the README's service setup) and umask 002, a new folder gets group qemu-vm with group write:
qemu-vm could then make child cgroups in it (its rmdir fails, EBUSY) or remove an empty VM
cgroup and make its own in its place."""


def make_cgroup(path: Path, *, exist_ok: bool) -> None:
    """Make the cgroup folder `path`, and each missing one above it, with CGROUP_FOLDER_MODE
    (mkdir's mode goes through the umask, which can only remove bits). A folder of this user
    that is there already loses its group and other write bits. Blocking: a few mkdir calls.

    Raises:
        FileExistsError: `path` exists and not `exist_ok`
        OSError: a folder cannot be made (no delegated subtree, a read-only cgroupfs)
    """
    try:
        make_dirs(path, mode=CGROUP_FOLDER_MODE, exist_ok=False)
    except FileExistsError:
        if not exist_ok:
            raise
        st = path.stat()
        if st.st_uid == os.geteuid() and st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            path.chmod(stat.S_IMODE(st.st_mode) & ~(stat.S_IWGRP | stat.S_IWOTH))


@dataclass(frozen=True)
class CgroupStats:
    """What a cgroup used (cpu.stat, memory.peak); each None when it cannot be read."""

    cpu_time_ms: int | None
    peak_memory_mb: int | None
    nr_throttled: int | None


NO_CGROUP_STATS: Final = CgroupStats(cpu_time_ms=None, peak_memory_mb=None, nr_throttled=None)
"""The stats of a VM with no cgroup."""


JOINED: Final = b"joined\n"
"""The first line of a child that Cgroup.join_or_stop_step moved: it is in its cgroup and
execs its command now."""


@dataclass(frozen=True, slots=True)
class Cgroup:
    """One cgroup folder that this library makes under code-exec: a VM's (its limits; QEMU,
    gvproxy and QEMU's teardown process in it) or the probe's (can_contain_processes).

    Its life: create; a child joins it before its exec (join_step, join_or_stop_step) or a
    process is attached after its spawn (attach); stats and reclaim while it runs; then kill
    (every process left, waited for) and remove (the folder), or close for both. `async with`
    closes it at the end of the block, made or not. Logs name it by its folder name (the VM
    id). Cgroup(path) is also the handle of one that may exist (a cleanup after a cancelled
    create, a recorded one of a dead process): kill and remove of a missing folder succeed.
    https://docs.kernel.org/admin-guide/cgroup-v2.html
    """

    path: Path

    @staticmethod
    def path_of(*, tenant_id: str, vm_id: str) -> Path:
        """Where the cgroup of VM `vm_id` of `tenant_id` is: code-exec/<tenant_id>/<vm_id>."""
        return Path(CGROUP_V2_BASE_PATH) / CGROUP_APP_NAMESPACE / tenant_id / vm_id

    @classmethod
    async def create(cls, *, tenant_id: str, vm_id: str, memory_mb: int, cpu_cores: float) -> Cgroup | None:
        """Make the cgroup of a VM with the effective limits that admission computed (the
        single source of the overheads): memory.max (memory_mb), cpu.max (cpu_cores), pids.max
        (CGROUP_PIDS_LIMIT: fork bombs, goroutines), memory.high where it can swap.

        None when this host gives this process no cgroup (no cgroup v2, a read-only one as on
        Docker Desktop, no delegated subtree: EROFS, EACCES): the VM then runs with ulimits only
        (wrap_with_ulimit). A refused create removes the VM folder it made before it returns; a
        folder that a cancel or another error left is the caller's to remove
        (Cgroup(Cgroup.path_of(...)).close(); the VM's pending owner does it).

        One worker-thread job: mkdir and the cgroup.subtree_control write take the kernel's
        cgroup locks, which other cgroup work can hold for long. A cancel waits for the job: the
        owner's cleanup (rmdir of the path it published) never runs while the job still makes
        the hierarchy.

        Raises:
            ConfigError: HOST_DEPENDENCY: another error made the cgroup fail
        """
        if not cgroup_v2_available():
            return None
        made = cls(cls.path_of(tenant_id=tenant_id, vm_id=vm_id))
        try:
            await run_to_end(functools.partial(_create_cgroup, made.path, memory_mb=memory_mb, cpu_cores=cpu_cores))
        except OSError as e:  # PermissionError too
            if e.errno in (errno.EROFS, errno.EACCES):
                logger.warning(
                    "cgroup v2 unavailable, resource limits disabled",
                    extra={"vm_id": vm_id, "path": str(made.path), "errno": e.errno},
                )
                # The VM folder may exist (a write refused after its mkdir): nothing would own it
                await run_to_end(functools.partial(_remove_cgroup, made.path))
                return None
            raise ConfigError(
                f"Failed to setup cgroup: {e}",
                code=ErrorCode.HOST_DEPENDENCY,
                context=None,
                hint=None,
                instance_id=None,
            ) from e
        return made

    async def attach(self, pid: int) -> bool:
        """Move process `pid` into this cgroup (its cgroup.procs), after its spawn. False when
        the kernel refuses (EACCES, EPERM: this process is outside the delegated subtree, as on
        a sandboxed CI runner; logged): the limits then do not hold for that process, and the
        VM goes on. The kernel checks write permission on the common ancestor's cgroup.procs
        (see the module docstring).

        False too when the process is gone (ESRCH: it exited and was reaped meanwhile; its
        exit shows where it is waited for).

        One worker-thread write: the kernel's move can wait for an RCU grace period
        (cgroup_threadgroup_rwsem), often several ms. A cancel waits for it (run_to_end): a
        write that lands later could move a process that reused the pid.

        Raises:
            ConfigError: HOST_DEPENDENCY: the write failed for another reason (ENOENT, EINVAL)
        """
        try:
            await run_to_end(functools.partial((self.path / "cgroup.procs").write_text, str(pid)))
        except OSError as e:
            if e.errno == errno.ESRCH:
                logger.debug(
                    "cgroup attachment skipped: the process is gone", extra={"cgroup_path": str(self.path), "pid": pid}
                )
                return False
            if e.errno in (errno.EACCES, errno.EPERM):
                logger.warning(
                    "cgroup attachment failed, resource limits not enforced for this VM",
                    extra={"cgroup_path": str(self.path), "pid": pid, "errno": e.errno},
                )
                return False
            raise ConfigError(
                f"Failed to attach PID {pid} to cgroup: {e}",
                code=ErrorCode.HOST_DEPENDENCY,
                context=None,
                hint=None,
                instance_id=None,
            ) from e
        return True

    def join_step(self) -> tuple[str, str]:
        """The shell step (subprocess_utils.child_shell) that moves the new child into this
        cgroup before it runs its command.

        Attaching the PID after the spawn is too late for a launcher: sudo forks QEMU, and QEMU
        clones its async-teardown process, at once; both then stay in this process's cgroup,
        with no limits and out of reach of kill. Writing "0" to cgroup.procs moves the writing
        process: the shell, which then execs the command. A shell step, not a Python
        preexec_fn, so Popen can vfork. A refused move (outside the delegated subtree) is
        ignored here and left to attach after the spawn.
        """
        return '{ echo 0 > "$1/cgroup.procs"; } 2>/dev/null', str(self.path)

    def join_or_stop_step(self) -> tuple[str, str]:
        """The shell step (subprocess_utils.child_shell) of a child that must run in this
        cgroup, as a QEMU that runs as qemu-vm: the shell writes "0" to its cgroup.procs (the
        kernel moves the shell), prints JOINED and execs its command (sudo). When the write
        fails (this process outside the delegated subtree), the shell exits at once: no sudo,
        no QEMU. The write succeeds only when the kernel moved the task, so it is the proof: no
        /proc read, which would assume /sys/fs/cgroup is the root of this cgroup namespace.
        This user cannot signal a qemu-vm QEMU: only its cgroup can end it (cgroup.kill) and
        prove its end (cgroup.events), so it never starts outside it. A shell step, not a
        Python preexec_fn, so Popen can vfork; the launch reads the line."""
        return 'echo 0 > "$1/cgroup.procs" || exit; echo joined', str(self.path)

    def stats(self) -> CgroupStats:
        """CPU time, peak memory and CPU throttle count, seen from the host (cpu.stat
        usage_usec and nr_throttled, memory.peak); each None when it cannot be read.

        Sync, called on the event loop: the kernel answers these small cgroupfs reads at once
        (no disk).
        """
        cpu_time_ms: int | None = None
        peak_memory_mb: int | None = None
        nr_throttled: int | None = None
        try:
            for line in (self.path / "cpu.stat").read_text().splitlines():
                if line.startswith("usage_usec"):
                    cpu_time_ms = int(line.split()[1]) // 1000
                elif line.startswith("nr_throttled "):
                    nr_throttled = int(line.split()[1])
        except (OSError, ValueError) as e:
            logger.debug(f"Failed to read cgroup cpu.stat: {e}", extra={"cgroup_path": str(self.path)})
        try:  # memory.peak: Linux 5.19+
            peak_memory_mb = int((self.path / "memory.peak").read_text().strip()) // (1024 * 1024)
        except (OSError, ValueError) as e:
            logger.debug(f"Failed to read cgroup memory.peak: {e}", extra={"cgroup_path": str(self.path)})
        return CgroupStats(cpu_time_ms=cpu_time_ms, peak_memory_mb=peak_memory_mb, nr_throttled=nr_throttled)

    async def reclaim(self, *, amount_mb: int) -> bool:
        """Ask the kernel to reclaim `amount_mb` from this cgroup (memory.reclaim): where the
        host has swap, it moves anonymous pages out (zswap or zram compresses them), which
        lowers RSS with no kill (Meta's TMO, Transparent Memory Offloading: 20-32% fleet-wide
        memory saved). Without swap only file pages go: after a reclaim an idle Python pool VM
        kept 206 MiB, 10 MiB with swap and zswap. swappiness=200 needs Linux 6.9+; plain
        reclaim (6.1+) before it.

        Returns:
            True if the reclaim was issued, False when it failed

        One worker-thread job: the kernel reclaims inside the write (10 to 500 ms).
        """
        return await asyncio.to_thread(_reclaim, self.path, amount_bytes=amount_mb * 1024 * 1024)

    async def kill(self) -> bool:
        """Kill every process left in this cgroup and wait until it is empty.

        The spawned process (QEMU, or sudo) is reaped first, but the cgroup can still hold
        processes nobody waits for: QEMU's async-teardown process (cleanup/<pid>, which unmaps
        the guest RAM after QEMU exits and checks its parent once per second) and, through sudo,
        a qemu-vm QEMU this user may not signal. cgroup.kill (Linux 5.14+) sends SIGKILL to each
        of them, with no per-process permission check; cgroup.events shows "populated 0" once
        the last one has left.

        Returns:
            True when the cgroup is empty or gone, False when a process is still in it after
            CGROUP_DRAIN_TIMEOUT_SECONDS (or on a read error)

        One worker-thread job; a cancel waits for it (see create).
        """
        return await run_to_end(functools.partial(_drain_from_now, self.path))

    async def remove(self) -> bool:
        """Remove this cgroup's folder (and the empty child cgroups a QEMU made in it). Kill
        first: a cgroup with a process in it cannot be removed (EBUSY), and its processes cannot
        move to the parent either (the parent enables controllers: cgroup v2 "no internal
        process" rule).

        Returns:
            True when the folder is gone, False when it stays (logged)

        One worker-thread job: rmdir takes the kernel's cgroup lock. A cancel waits for it.
        """
        return await run_to_end(functools.partial(_remove_cgroup, self.path))

    async def close(self) -> bool:
        """kill, then remove once empty: the end of a cgroup. True when it is gone."""
        return await self.kill() and await self.remove()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """close, to its end even when the caller is cancelled meanwhile (its cancel is raised
        after). A failed close is logged and the folder stays: its owner removes it later (the
        probe: the next probe, _stale_probes)."""
        await await_cancellation_safe(self.close())

    @staticmethod
    def end_all(cgroups: Sequence[Cgroup]) -> list[Cgroup]:
        """kill, then remove, for many cgroups at once (blocking): the sweep of the folders of
        dead processes (private_run_dir). Each gets cgroup.kill first, so their waits overlap,
        under one deadline: N cgroups whose processes do not leave cost
        CGROUP_DRAIN_TIMEOUT_SECONDS, not N times it. A cgroup with no process left is then
        removed (a failed rmdir is logged).

        Returns:
            The cgroups that still have a process CGROUP_DRAIN_TIMEOUT_SECONDS after the kills
            (or whose state cannot be read), or whose folder could not be removed (logged): the
            caller keeps their records for the next sweep; empty when all are gone
        """
        for each in cgroups:
            _send_kill(each.path)
        deadline = time.monotonic() + CGROUP_DRAIN_TIMEOUT_SECONDS
        # In order: each wait, then its removal (a comprehension with the work in its test)
        return [
            each for each in cgroups if not (_drain_cgroup(each.path, deadline=deadline) and _remove_cgroup(each.path))
        ]


def _create_cgroup(cgroup_path: Path, *, memory_mb: int, cpu_cores: float) -> None:
    """Cgroup.create's file work (blocking)."""
    tenant_cgroup = cgroup_path.parent

    # Create tenant cgroup and enable controllers for nested VM cgroups
    # In cgroup v2, subtree_control only affects immediate children,
    # so we must enable controllers at each level of the hierarchy
    make_cgroup(tenant_cgroup, exist_ok=True)
    (tenant_cgroup / "cgroup.subtree_control").write_text("+memory +cpu +pids")

    # Create VM cgroup
    make_cgroup(cgroup_path, exist_ok=True)

    # Apply effective memory limit (pre-computed by admission controller)
    memory_max_bytes = memory_mb * 1024 * 1024
    (cgroup_path / "memory.max").write_text(str(memory_max_bytes))

    # Soft memory limit at 85% of max, only when this cgroup can swap: above it the kernel
    # moves the VM's idle anon pages out (zswap compression in front of the swap), which
    # memory overcommit counts on. Without swap the guest RAM cannot leave, and a soft
    # limit below guest RAM + QEMU + their page cache only throttles the VM (measured on
    # nested KVM: an L1 or instance save then misses its 32 s deadline). zswap writeback
    # stays on then: with it off, a page the zswap pool refuses (pool full, incompressible)
    # has nowhere to go, and the VM is throttled again.
    if _cgroup_can_swap(cgroup_path):
        memory_high_bytes = int(memory_max_bytes * 0.85)
        _write_cgroup_optional(cgroup_path / "memory.high", str(memory_high_bytes))

    # Kill entire cgroup as a unit on OOM (QEMU + gvproxy together).
    # Without this, OOM may kill only gvproxy, leaving QEMU orphaned.
    _write_cgroup_optional(cgroup_path / "memory.oom.group", "1")

    # Apply effective CPU limit (pre-computed by admission controller)
    cpu_quota = int(CGROUP_CPU_PERIOD_US * cpu_cores)
    (cgroup_path / "cpu.max").write_text(f"{cpu_quota} {CGROUP_CPU_PERIOD_US}")

    # Set PID limit (fork bomb prevention)
    (cgroup_path / "pids.max").write_text(str(CGROUP_PIDS_LIMIT))

    # Sanity-check that we can open cgroup.procs for writing.
    # This validates the VFS permission (check 1 in module docstring) but
    # NOT the common-ancestor check (check 2), which fires at write() time
    # (cgroup_procs_write_permission()).  If the orchestrator is outside the
    # delegated subtree, this passes but the join step or Cgroup.attach fails
    # later, and each handles it.
    with (cgroup_path / "cgroup.procs").open("a"):
        pass  # Just test we can open for writing


async def can_contain_processes() -> bool:
    """Whether this process can make a VM cgroup and put a process in it, as each VM launch
    does (a delegated cgroup v2 subtree it runs inside): a QEMU that runs as qemu-vm needs
    it, as only cgroup.kill ends one and only cgroup.events proves its end. One probe cgroup
    and one short child: both go on every end (an error, a cancel); any failure is False.
    First, the probe cgroups of dead processes go (_stale_probes): nothing else records them."""
    vm_id = f"probe-{os.getpid()}-{time.monotonic_ns()}"
    tenant_id = _PROBE_TENANT
    for stale in await asyncio.to_thread(_stale_probes):
        await stale.close()  # a failure is logged: the next probe tries again
    # The path is known before create: a cancel or a later failure can leave the folder
    async with Cgroup(Cgroup.path_of(tenant_id=tenant_id, vm_id=vm_id)) as probe:
        child: asyncio.subprocess.Process | None = None
        try:
            # cgroup.kill: Linux 5.14+; before it, nothing could end a QEMU this user may not signal
            if await Cgroup.create(tenant_id=tenant_id, vm_id=vm_id, memory_mb=64, cpu_cores=0.1) is None or (
                not await asyncio.to_thread((probe.path / "cgroup.kill").exists)
            ):
                return False
            child = await asyncio.create_subprocess_exec(
                "sleep", "60", stdin=asyncio.subprocess.DEVNULL, env=child_env()
            )
            return await probe.attach(child.pid)
        except (ConfigError, OSError):
            return False
        finally:
            if child is not None:  # before the cgroup's end: a child that never joined it stays out of its kill
                await await_cancellation_safe(_end_child(child))


_PROBE_TENANT: Final = "exec-sandbox-probe"


def _stale_probes() -> list[Cgroup]:
    """The probe cgroups (probe-<pid>-<time>) whose process is gone: a probe whose process died
    mid-way (SIGKILL), or whose removal failed. A cgroup.kill ends what is left in
    them (the probe's child). Blocking: one listing."""
    if not cgroup_v2_available():
        return []
    tenant = Cgroup.path_of(tenant_id=_PROBE_TENANT, vm_id="x").parent
    try:
        names = [entry.name for entry in os.scandir(tenant) if entry.is_dir(follow_symlinks=False)]
    except OSError:  # not made yet, or no access: nothing of this user's there
        return []
    stale: list[Cgroup] = []
    for name in names:
        prefix, _, rest = name.partition("-")
        pid, _, _when = rest.partition("-")
        if prefix == "probe" and pid.isdigit() and int(pid) != os.getpid() and not psutil.pid_exists(int(pid)):
            stale.append(Cgroup(tenant / name))
    return stale


async def _end_child(child: asyncio.subprocess.Process) -> None:
    """can_contain_processes' child: killed and reaped."""
    with contextlib.suppress(ProcessLookupError):
        child.kill()
    await child.wait()


def _reclaim(cgroup_path: Path, *, amount_bytes: int) -> bool:
    """Cgroup.reclaim's writes (blocking)."""
    reclaim_file = cgroup_path / "memory.reclaim"

    # EAGAIN = the kernel reclaimed less than asked: normal for a request sized to
    # the whole guest. Only EINVAL (no swappiness support, kernel < 6.9) falls back.
    try:
        # Try with swappiness=200 first (kernel 6.9+, aggressively compress anon pages)
        reclaim_file.write_text(f"{amount_bytes} swappiness=200")
        return True
    except OSError as e:
        if e.errno == errno.EAGAIN:
            return True
        if e.errno != errno.EINVAL:
            logger.debug("memory.reclaim failed", extra={"cgroup_path": str(cgroup_path), "error": str(e)})
            return False

    try:
        # Fall back to plain reclaim (kernel 6.1+)
        reclaim_file.write_text(str(amount_bytes))
        return True
    except OSError as e:
        if e.errno == errno.EAGAIN:
            return True
        logger.debug(
            "memory.reclaim unavailable",
            extra={"cgroup_path": str(cgroup_path), "error": str(e)},
        )
        return False


def _drain_from_now(cgroup_path: Path) -> bool:
    """Cgroup.kill's job: the wait's deadline starts when the job runs, not when it was
    queued (a busy executor would eat the wait of a kill that works)."""
    return _drain_cgroup(cgroup_path, deadline=time.monotonic() + CGROUP_DRAIN_TIMEOUT_SECONDS)


def _populated(fd: int) -> bool:
    """The "populated" value of the cgroup.events file open at `fd` (read from the start)."""
    for line in os.pread(fd, 4096, 0).decode().splitlines():
        key, _, value = line.partition(" ")
        if key == "populated":
            return value.strip() != "0"
    return False


def _drain_cgroup(cgroup_path: Path, *, deadline: float) -> bool:
    """Cgroup.kill's work (blocking): poll(2) on cgroup.events wakes at each change of it,
    until `deadline` (time.monotonic)."""
    context = {"context_id": cgroup_path.name, "path": str(cgroup_path)}
    try:
        fd = os.open(cgroup_path / "cgroup.events", os.O_RDONLY)
    except FileNotFoundError:
        return True  # The cgroup is gone (removed only when empty)
    except OSError as e:
        logger.error(f"cgroup drain error: {_errno_text(e)}", extra=context)
        return False
    try:
        if not _populated(fd):
            return True
        _send_kill(cgroup_path)  # Linux < 5.14 has no cgroup.kill: processes may exit by themselves
        poller = select.poll()
        poller.register(fd, select.POLLPRI | select.POLLERR)
        while _populated(fd):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pids = (cgroup_path / "cgroup.procs").read_text().split()
                logger.error(
                    f"cgroup still has processes at the end of the wait after cgroup.kill: pids {' '.join(pids)}",
                    extra=context,
                )
                return False
            poller.poll(remaining * 1000)
        return True
    except OSError as e:
        if e.errno in (errno.ENOENT, errno.ENODEV):
            return True  # Removed meanwhile
        logger.error(f"cgroup drain error: {_errno_text(e)}", extra=context)
        return False
    finally:
        os.close(fd)


def _send_kill(cgroup_path: Path) -> None:
    """Write "1" to the cgroup.kill file of `cgroup_path`, never creating it. Errors (the
    cgroup is gone, Linux < 5.14) are left to _drain_cgroup."""
    try:
        fd = os.open(cgroup_path / "cgroup.kill", os.O_WRONLY)
    except OSError:
        return
    try:
        os.write(fd, b"1")
    except OSError:
        pass
    finally:
        os.close(fd)


def _errno_text(error: OSError) -> str:
    """'[Errno 16] EBUSY: Device or resource busy' (the log formats drop `extra`)."""
    name = errno.errorcode.get(error.errno, "?") if error.errno is not None else "?"
    return f"[Errno {error.errno}] {name}: {error.strerror}"


def _remove_cgroup(cgroup_path: Path) -> bool:
    """Cgroup.remove's work (blocking): its empty child cgroups go first (a QEMU may make
    some where it can write: rmdir of a cgroup with a child fails, EBUSY)."""
    context = {"context_id": cgroup_path.name, "path": str(cgroup_path)}
    try:
        for parent, children, _files in os.walk(cgroup_path, topdown=False):
            for child in children:
                os.rmdir(os.path.join(parent, child))  # noqa: PTH106, PTH118 - os.walk gives str paths
        cgroup_path.rmdir()
    except FileNotFoundError:
        return True  # Already deleted (race condition)
    except OSError as e:
        logger.error(
            f"cgroup removal error: {_errno_text(e)}",
            extra={**context, "errno": e.errno},
        )
        return False
    logger.debug("cgroup removed", extra=context)
    return True


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
    per-UID: ALL processes under the same UID share a single counter.  At 256
    (the per-VM cgroup pids limit), QEMU fails with ``qemu_thread_create:
    Resource temporarily unavailable`` at ~23 concurrent VMs:
    23 QEMUs x 12 threads = 276 > 256.  The system's default RLIMIT_NPROC
    (typically 126K+ on Linux, 2784 on macOS) already prevents fork bombs.

    Args:
        cmd: Original command
        cgroup_memory_mb: Effective memory limit in MB (guest + overhead, pre-computed by admission)

    Returns:
        Linux: the command run by bash -c after ulimit -v and -t. macOS: the command itself
        (no limit applies there; a shell that only execs it costs ~5 ms per QEMU start)
    """
    if detect_host_os() == HostOS.MACOS:
        # macOS: No ulimit controls available
        # - Virtual memory (-v) not supported by macOS kernel (setrlimit fails)
        # - CPU time (-t) breaks subprocess stdout pipe on macOS (QEMU output lost)
        # - Process limit (-u / RLIMIT_NPROC) is per-UID, not per-process — cannot
        #   provide per-VM isolation. Rely on system default.
        return list(cmd)

    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)

    # Memory overhead: ~14x effective memory for TCG worst case
    # ULIMIT_MEMORY_MULTIPLIER accounts for TCG virtual address space expansion
    # (separate from cgroup overhead which is already included in cgroup_memory_mb)
    virtual_mem_kb = cgroup_memory_mb * 1024 * ULIMIT_MEMORY_MULTIPLIER

    # Linux: Virtual memory and CPU time limits only
    # - Virtual memory (-v) is the primary memory control
    # - CPU time (-t) as safety net for runaway VMs
    # - No -u (RLIMIT_NPROC): per-UID, not per-process — see docstring
    shell_cmd = f"ulimit -v {virtual_mem_kb} && ulimit -t {ULIMIT_CPU_TIME_SECONDS} && exec {cmd_str}"
    return ["bash", "-c", shell_cmd]

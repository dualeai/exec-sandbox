"""System capability probes for VM acceleration and feature detection.

These probes detect system capabilities once and cache the results.
Async probes use a shared cache container to avoid global statements.
The Scheduler start also raises the process limits it can (raise_fd_limit,
raise_nproc_limit) and warns about unsafe host settings (warn_*).
"""

import asyncio
import contextlib
import errno
import os
import platform
import re
import resource
import sys
import time
import weakref
from collections.abc import Callable, Coroutine
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar, Final, ParamSpec, TypeVar, cast

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.cgroup import CGROUP_V2_BASE_PATH, cgroup_v2_available, resolve_self_cgroup_v2
from exec_sandbox.exceptions import ErrorCode, SandboxError, spawn_error
from exec_sandbox.files.disk import LINUX_MEMORY_FILESYSTEMS, OVERLAYFS_SUPER_MAGIC, overlay_upper, statfs_type
from exec_sandbox.platform_utils import HostArch, HostOS, child_env, detect_host_arch, detect_host_os
from exec_sandbox.settings import QemuBinaries
from exec_sandbox.vm_types import AccelType

P = ParamSpec("P")
T = TypeVar("T")

logger = get_logger(__name__)

__all__ = [
    "NOT_CACHED",
    "ProbeCache",
    "can_access",
    "check_hvf_available",
    "check_hwaccel_available",
    "check_kvm_available",
    "check_tsc_deadline",
    "detect_accel_type",
    "posix_acl_xattr",
    "probe_cache",
    "probe_direct_io",
    "probe_drive_direct_io",
    "probe_io_uring_support",
    "probe_qemu_accelerators",
    "probe_qemu_runnable",
    "probe_qemu_sandbox_support",
    "probe_qemu_version",
    "probe_setpriv_pdeathsig",
    "probe_unshare_support",
    "qemu_system_binary",
    "qemu_unrunnable_error",
    "raise_fd_limit",
    "raise_nproc_limit",
    "warn_cgroup_pids_limit",
    "warn_max_map_count",
    "warn_overcommit",
    "warn_threads_max",
]

# ============================================================================
# Recommended System Limit Thresholds
# ============================================================================
# Centralized thresholds for all startup probes. Each probe warns when
# the corresponding system limit is below these values.

_MIN_RECOMMENDED_FD_LIMIT = 8192  # RLIMIT_NOFILE — each VM needs ~7-10 fds
_MIN_RECOMMENDED_NPROC_LIMIT = 4096  # RLIMIT_NPROC — 50 VMs x 15 threads = 750
_MIN_RECOMMENDED_MAX_MAP_COUNT = 262_144  # vm.max_map_count — each VM uses ~340 VMAs
_MIN_RECOMMENDED_THREADS_MAX = 100_000  # kernel.threads-max — default ~60K on 7GB RAM
_MIN_RECOMMENDED_CGROUP_PIDS = 8192  # cgroup pids.max — Docker default is 4096

# ============================================================================
# File Descriptor Limit Tuning
# ============================================================================


def raise_fd_limit() -> None:
    """Raise the soft file descriptor limit to the hard limit (no privileges needed).

    On most Linux/macOS systems, the default soft limit is 1024 (inherited from
    ancient ``select()`` compat) while the hard limit is much higher (e.g., 524288
    on Debian, 1048576 on Ubuntu). Each QEMU VM consumes ~7-10 fds from the
    orchestrator process for subprocess pipes, sockets, and lock files.

    At 100 concurrent VMs the default 1024 soft limit is exhausted, causing QEMU
    to die with SIGABRT ("qemu_thread_create: Resource temporarily unavailable").

    This follows the same pattern as Go 1.19+ which automatically raises
    ``RLIMIT_NOFILE`` soft to hard at runtime startup (go#46279).

    POSIX guarantees that any unprivileged process can raise its soft limit up to
    the hard limit — no root/``CAP_SYS_RESOURCE`` needed. Only raising the hard
    limit requires privileges.

    Called once during Scheduler startup, before any VMs are created.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft >= hard:
            logger.debug("fd limit already at maximum", extra={"soft": soft, "hard": hard})
            return

        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        logger.info(
            "Raised fd soft limit to hard limit",
            extra={"old_soft": soft, "new_soft": hard, "hard": hard},
        )

        if hard < _MIN_RECOMMENDED_FD_LIMIT:
            logger.warning(
                "Hard fd limit is low — high-concurrency workloads (>50 VMs) may hit fd exhaustion. "
                "Raise with: ulimit -n 65536 (or set in /etc/security/limits.conf)",
                extra={"hard": hard, "recommended_min": _MIN_RECOMMENDED_FD_LIMIT},
            )
    except (ValueError, OSError) as e:
        # Not fatal — log and continue. Worst case: QEMU dies under high concurrency
        # and the existing retry logic handles it.
        logger.warning("Failed to raise fd limit", extra={"error": str(e)})


# ============================================================================
# Virtual Memory Overcommit Check
# ============================================================================


def warn_overcommit() -> None:
    """Warn if Linux ``vm.overcommit_memory=0`` (heuristic mode) is active.

    With heuristic overcommit (mode 0, the default), the kernel may reject
    ``mmap`` calls for QEMU thread stacks during burst VM startup, even when
    the system has ample free RAM.  glibc converts the resulting ``ENOMEM``
    to ``EAGAIN``, which QEMU treats as fatal (``abort()`` with zero retry).
    This manifests as ``qemu_thread_create: Resource temporarily unavailable``.

    Setting ``vm.overcommit_memory=1`` ("always allow") eliminates the
    heuristic check.  The OOM killer remains as the backstop, which is
    appropriate for ephemeral VM workloads.

    Empirically, switching from mode 0 to 1 reduced 100-VM burst failures
    from 14 % to 1 % on a 16-core / 32 GB EC2 instance.

    Called once during Scheduler startup on Linux only.  No-op on macOS.
    """
    if detect_host_os() != HostOS.LINUX:
        return

    try:
        overcommit = int(Path("/proc/sys/vm/overcommit_memory").read_text().strip())
    except (OSError, ValueError):
        return  # /proc not available or unreadable — skip silently

    if overcommit == 0:
        logger.warning(
            "vm.overcommit_memory=0 (heuristic) — QEMU thread creation may fail under "
            "burst VM startup. Set to 1 for reliable high-concurrency operation: "
            "sudo sysctl -w vm.overcommit_memory=1",
            extra={"current": overcommit, "recommended": 1},
        )


# ============================================================================
# Process/Thread Limit Tuning
# ============================================================================


def raise_nproc_limit() -> None:
    """Raise the soft process/thread limit to the hard limit (no privileges needed).

    RLIMIT_NPROC limits the total number of processes+threads for the current UID.
    Each QEMU VM spawns ~15 threads, so the default soft limit (e.g., 63281 on
    Linux, 5568 on macOS) can be exhausted under high concurrency.

    When exhausted, ``pthread_create`` fails with ``EAGAIN``, which QEMU converts
    to a fatal ``abort()`` — "qemu_thread_create: Resource temporarily unavailable".

    Called once during Scheduler startup, before any VMs are created.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            logger.info(
                "Raised nproc soft limit to hard limit",
                extra={"old_soft": soft, "new_soft": hard, "hard": hard},
            )
        else:
            logger.debug("nproc limit already at maximum", extra={"soft": soft, "hard": hard})

        # Warn unconditionally if hard limit is low (even if soft == hard)
        if hard < _MIN_RECOMMENDED_NPROC_LIMIT:
            logger.warning(
                "Hard nproc limit is low — QEMU thread creation may fail under high concurrency. "
                "Raise with: ulimit -u 65536",
                extra={"hard": hard, "recommended_min": _MIN_RECOMMENDED_NPROC_LIMIT},
            )
    except (ValueError, OSError) as e:
        logger.warning("Failed to raise nproc limit", extra={"error": str(e)})


# ============================================================================
# Virtual Memory Map Count Check
# ============================================================================


def warn_max_map_count() -> None:
    """Warn if Linux ``vm.max_map_count`` is below the recommended threshold.

    Each QEMU VM creates ~340 virtual memory areas (VMAs) for virtio rings,
    guest RAM mappings, and device MMIO regions. The kernel default of 65530
    is exhausted at ~190 concurrent VMs, causing ``mmap`` failures.

    This is a well-known limit also raised by Elasticsearch, Docker, and
    Kata Containers.

    Called once during Scheduler startup on Linux only. No-op on macOS.
    """
    if detect_host_os() != HostOS.LINUX:
        return

    try:
        max_map_count = int(Path("/proc/sys/vm/max_map_count").read_text().strip())
    except (OSError, ValueError):
        return

    if max_map_count < _MIN_RECOMMENDED_MAX_MAP_COUNT:
        logger.warning(
            "vm.max_map_count is low — QEMU mmap may fail under high VM concurrency. "
            "Raise with: sudo sysctl -w vm.max_map_count=262144",
            extra={"current": max_map_count, "recommended_min": _MIN_RECOMMENDED_MAX_MAP_COUNT},
        )


# ============================================================================
# Kernel Threads-Max Check
# ============================================================================


def warn_threads_max() -> None:
    """Warn if Linux ``kernel.threads-max`` is below the recommended threshold.

    ``kernel.threads-max`` is the system-wide limit on total threads (all UIDs).
    On low-memory hosts (e.g., 7 GB CI runners) the default (~60K) can be
    reached before per-UID ``RLIMIT_NPROC``, causing ``pthread_create`` failures
    across all processes — not just QEMU.

    Called once during Scheduler startup on Linux only. No-op on macOS.
    """
    if detect_host_os() != HostOS.LINUX:
        return

    try:
        threads_max = int(Path("/proc/sys/kernel/threads-max").read_text().strip())
    except (OSError, ValueError):
        return

    if threads_max < _MIN_RECOMMENDED_THREADS_MAX:
        logger.warning(
            "kernel.threads-max is low — system-wide thread creation may fail under high VM concurrency. "
            "Raise with: sudo sysctl -w kernel.threads-max=120000",
            extra={"current": threads_max, "recommended_min": _MIN_RECOMMENDED_THREADS_MAX},
        )


# ============================================================================
# Cgroup PIDs Limit Check
# ============================================================================


def warn_cgroup_pids_limit() -> None:
    """Warn if any cgroup v2 ancestor has a low ``pids.max`` limit.

    In containerized environments (Docker, Kubernetes, systemd slices), the
    effective PID limit is the *minimum* ``pids.max`` across all ancestor
    cgroups. Docker defaults to ``--pids-limit=4096`` and systemd sets
    ``DefaultTasksMax`` based on ``kernel.pid_max``.

    Only the tightest ancestor is reported to avoid log spam.

    Called once during Scheduler startup on Linux only. No-op on macOS.
    """
    cgroup_root = Path(CGROUP_V2_BASE_PATH)
    if not cgroup_v2_available():
        return

    cgroup_rel = resolve_self_cgroup_v2()
    if cgroup_rel is None:
        return

    # Walk from current cgroup up to root, find the tightest pids.max
    tightest_limit: int | None = None
    tightest_path: str | None = None

    current = cgroup_root / cgroup_rel.lstrip("/")
    while current != current.parent:  # Stop at filesystem root (safety bound)
        try:
            content = (current / "pids.max").read_text().strip()
            if content != "max":
                limit = int(content)
                if tightest_limit is None or limit < tightest_limit:
                    tightest_limit = limit
                    tightest_path = str(current.relative_to(cgroup_root))
        except (OSError, ValueError):
            pass  # File doesn't exist or isn't readable at this level

        if current == cgroup_root:
            break
        current = current.parent

    if tightest_limit is not None and tightest_limit < _MIN_RECOMMENDED_CGROUP_PIDS:
        logger.warning(
            "cgroup pids.max is low — container/systemd PID limit may cause thread creation failures. "
            "Fix with: docker run --pids-limit=-1, or systemd TasksMax=infinity",
            extra={
                "current": tightest_limit,
                "cgroup": tightest_path,
                "recommended_min": _MIN_RECOMMENDED_CGROUP_PIDS,
            },
        )


_ZSWAP_PARAM_BASE: Final[str] = "/sys/module/zswap/parameters"


def _read_zswap_param(name: str) -> str | None:
    """Read a zswap sysfs parameter, or None if unavailable."""
    try:
        return Path(f"{_ZSWAP_PARAM_BASE}/{name}").read_text().strip()
    except (OSError, ValueError):
        return None


def warn_zswap() -> None:
    """Warn if Linux zswap is disabled or suboptimally configured.

    zswap provides a compressed write-back cache for swap pages, absorbing
    transient host memory pressure spikes without disk I/O.  Combined with
    per-VM ``memory.reclaim``, it enables proactive compression of idle VM
    memory (Meta TMO: 20-32% fleet-wide savings).

    Recommended config for VM density::

        echo 1 > /sys/module/zswap/parameters/enabled
        echo lz4 > /sys/module/zswap/parameters/compressor
        echo zsmalloc > /sys/module/zswap/parameters/zpool
        echo 40 > /sys/module/zswap/parameters/max_pool_percent

    LZ4 is preferred over zstd: lower latency (1-2µs vs 5-10µs per page),
    better for interactive VM workloads.  40% pool allows 2-3x effective
    memory capacity under pressure.

    Called once during Scheduler startup on Linux only.  No-op on macOS.
    """
    if detect_host_os() != HostOS.LINUX:
        return

    enabled = _read_zswap_param("enabled")
    if enabled is None:
        return  # zswap module not loaded or /sys unavailable

    if enabled != "Y":
        logger.warning(
            "zswap disabled — enable for better VM density under memory pressure: "
            "echo 1 > /sys/module/zswap/parameters/enabled && "
            "echo lz4 > /sys/module/zswap/parameters/compressor && "
            "echo zsmalloc > /sys/module/zswap/parameters/zpool && "
            "echo 40 > /sys/module/zswap/parameters/max_pool_percent",
        )
        return

    # Check tuning of enabled zswap
    pool_pct = _read_zswap_param("max_pool_percent")
    if pool_pct is not None and int(pool_pct) < 30:  # noqa: PLR2004
        logger.info(
            "zswap max_pool_percent is low — raise to 40 for better VM density: "
            "echo 40 > /sys/module/zswap/parameters/max_pool_percent",
            extra={"current": pool_pct, "recommended": 40},
        )


def warn_damon() -> None:
    """Log DAMON availability for idle VM page reclamation.

    DAMON (Data Access MONitor, kernel 5.16+) with DAMOS can automatically
    pageout guest memory pages not accessed for a configurable period,
    providing continuous host-driven reclamation without guest cooperation.
    This supplements ``memory.reclaim`` with proactive kernel-driven reclaim.

    Called once during Scheduler startup on Linux only.  No-op on macOS.
    """
    if detect_host_os() != HostOS.LINUX:
        return

    damon_dir = Path("/sys/kernel/mm/damon/admin")
    if not damon_dir.is_dir():
        logger.info(
            "DAMON not available — enable CONFIG_DAMON + CONFIG_DAMON_SYSFS in host "
            "kernel for automatic idle VM memory reclamation",
        )
        return

    logger.debug("DAMON available for idle VM page reclamation")


# ============================================================================
# Probe Cache
# ============================================================================

# Sentinel value for "not yet cached" (distinguishes from None results)
NOT_CACHED: Any = object()

# KVM ioctl constants for probing
# See: linux/kvm.h - these are stable ABI
_KVM_GET_API_VERSION = 0xAE00
_KVM_API_VERSION_EXPECTED = 12  # Stable since Linux 2.6.38


class ProbeCache:
    """Container for cached system probe results.

    Uses a class to avoid global statements while maintaining module-level caching.
    Locks are made on first use (get_lock), one per probe name and event loop.

    The locks prevent cache stampede when multiple VMs start concurrently - without
    them, all VMs would run the detection subprocess simultaneously instead of
    sharing the cached result.
    """

    __slots__ = (
        "_locks",
        "direct_io",
        "drive_direct_io",
        "hvf",
        "io_uring",
        "kvm",
        "posix_acls",
        "qemu_accels",
        "qemu_sandbox",
        "qemu_unrunnable",
        "qemu_version",
        "setpriv",
        "tsc_deadline",
        "unshare",
    )
    _RESULTS: ClassVar[tuple[str, ...]] = tuple(name for name in __slots__ if name != "_locks")

    def __init__(self) -> None:
        self.direct_io: dict[int, bool] = {}  # per device (st_dev): its filesystem takes O_DIRECT
        self.drive_direct_io: dict[Path, bool] = {}  # per folder: VM drive files there use O_DIRECT
        self.posix_acls: dict[int, bool] = {}  # per device (st_dev): its filesystem keeps POSIX ACLs
        self.hvf: bool | None = NOT_CACHED
        self.io_uring: bool | None = NOT_CACHED
        self.kvm: bool | None = NOT_CACHED
        self.qemu_accels: set[str] | None = NOT_CACHED  # Accelerators available in QEMU binary
        self.qemu_sandbox: bool | None = NOT_CACHED  # QEMU compiled with CONFIG_SECCOMP
        self.qemu_version: tuple[int, int, int] | None = NOT_CACHED  # QEMU semver (major, minor, patch)
        # The QEMU binary and the exec error of the version probe, when this host cannot run it
        self.qemu_unrunnable: tuple[str, OSError] | None = None
        self.setpriv: bool | None = NOT_CACHED  # setpriv --pdeathsig works
        self.tsc_deadline: bool | None = NOT_CACHED
        self.unshare: bool | None = NOT_CACHED
        # One map per event loop: an asyncio.Lock binds to the loop that first waits on it,
        # so a second loop (asyncio.run again) must never get it. A map goes with its loop.
        self._locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = (
            weakref.WeakKeyDictionary()
        )

    def get_lock(self, name: str) -> asyncio.Lock:
        """The lock of the probe `name` in the running event loop, made on first use (reset()
        keeps it)."""
        locks = self._locks.setdefault(asyncio.get_running_loop(), {})
        if name not in locks:
            locks[name] = asyncio.Lock()
        return locks[name]

    def reset(self, attr: str | None = None) -> None:
        """Put cached probe result(s) back to their uncached value (for testing): the value a new
        cache has, so a per-device dict is an empty dict again, not the sentinel.

        Args:
            attr: Specific attribute to reset, or None to reset all.
        """
        fresh = ProbeCache()
        for name in self._RESULTS if attr is None else (attr,):
            setattr(self, name, getattr(fresh, name))


# Module-level cache instance
probe_cache = ProbeCache()


def _async_cached_probe(
    cache_attr: str,
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Decorator for async probes with double-checked locking cache.

    Eliminates boilerplate pattern:
        if cache.attr is not NOT_CACHED: return cache.attr
        async with cache.get_lock("attr"):
            if cache.attr is not NOT_CACHED: return cache.attr
            result = await actual_work()
            cache.attr = result
            return result

    Uses a sentinel value (NOT_CACHED) to distinguish "not yet cached" from
    cached None results, allowing probes that return None on failure to be cached.

    Args:
        cache_attr: Name of the attribute on probe_cache to use for caching

    Returns:
        Decorated async function with automatic caching
    """

    def decorator(func: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Fast path: return cached result (no lock needed)
            cached = getattr(probe_cache, cache_attr)
            if cached is not NOT_CACHED:
                return cached
            # Slow path: acquire lock to prevent stampede, then check cache again
            async with probe_cache.get_lock(cache_attr):
                # Double-check after acquiring lock (another task may have populated cache)
                cached = getattr(probe_cache, cache_attr)
                if cached is not NOT_CACHED:
                    return cached
                result = await func(*args, **kwargs)
                setattr(probe_cache, cache_attr, result)
                return result

        return wrapper

    return decorator


def qemu_system_binary(binaries: QemuBinaries, *, arch: HostArch) -> str:
    """The qemu-system binary for `arch`: the configured file (EXEC_SANDBOX_QEMU_BIN_X86 /
    _ARM, /usr/bin by default) when it exists, else the bare name, found on PATH by exec.
    The one resolver: VMs (qemu_cmd) and the QEMU probes run the binary it gives."""
    configured = binaries.qemu_bin_arm if arch == HostArch.AARCH64 else binaries.qemu_bin_x86
    if configured.is_file():
        return str(configured)
    return "qemu-system-aarch64" if arch == HostArch.AARCH64 else "qemu-system-x86_64"


def _probed_qemu_binary() -> str:
    """The binary the QEMU probes run: VMs read the same two variables (Settings is a
    QemuBinaries; SchedulerConfig does not set them)."""
    return qemu_system_binary(QemuBinaries(), arch=detect_host_arch())


@_async_cached_probe("qemu_accels")
async def probe_qemu_accelerators() -> set[str]:
    """Probe QEMU binary for available accelerators (cached).

    This provides a 2nd layer of verification beyond OS-level checks (ioctl/sysctl).
    Even if /dev/kvm exists and responds to ioctl, QEMU may not have the accelerator
    compiled in, or may fail to initialize it in certain environments.

    Uses `qemu-system-xxx -accel help` to get the list of accelerators that QEMU
    actually supports. This is the same method recommended by QEMU documentation.

    References:
        - QEMU docs: "help can also be passed as an argument to another option"
        - libvirt probes QEMU binary presence + /dev/kvm (drvqemu.html)
        - GitHub Actions KVM issues: https://github.com/orgs/community/discussions/8305

    Returns:
        Set of available accelerator names (e.g., {"tcg", "kvm"} or {"tcg", "hvf"})
    """
    qemu_bin = _probed_qemu_binary()

    try:
        proc = await asyncio.create_subprocess_exec(
            qemu_bin,
            "-accel",
            "help",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env(),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)

        if proc.returncode != 0:
            logger.warning(
                "QEMU accelerator probe failed",
                extra={"qemu_bin": qemu_bin, "returncode": proc.returncode},
            )
            return set()

        # Parse output: "Accelerators supported in QEMU binary:\ntcg\nkvm\n"
        # or "tcg\nhvf\n" on macOS
        output = stdout.decode().strip()
        accels: set[str] = set()
        for raw_line in output.split("\n"):
            accel_name = raw_line.strip().lower()
            # Skip header line and empty lines
            if accel_name and not accel_name.startswith("accelerator"):
                accels.add(accel_name)

        logger.debug(
            "QEMU accelerator probe complete",
            extra={"qemu_bin": qemu_bin, "accelerators": sorted(accels)},
        )
        return accels

    except FileNotFoundError:
        logger.warning(
            "QEMU binary not found for accelerator probe",
            extra={"qemu_bin": qemu_bin},
        )
        return set()
    except (OSError, TimeoutError) as e:
        logger.warning(
            "QEMU accelerator probe failed",
            extra={"qemu_bin": qemu_bin, "error": str(e)},
        )
        return set()


@_async_cached_probe("qemu_version")
async def probe_qemu_version() -> tuple[int, int, int] | None:
    """Probe QEMU binary version (cached).

    Returns:
        Semver tuple (major, minor, patch), or None if detection fails.
        Example: (8, 2, 0) for QEMU 8.2.0

    Tuples compare naturally as semver: (9, 2, 0) >= (9, 2, 0) works correctly.
    """
    qemu_bin = _probed_qemu_binary()
    probe_cache.qemu_unrunnable = None

    try:
        proc = await asyncio.create_subprocess_exec(
            qemu_bin,
            "--version",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=child_env(),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)

        if proc.returncode != 0:
            logger.warning("QEMU version probe failed", extra={"returncode": proc.returncode})
            return None

        # Parse "QEMU emulator version X.Y.Z" (handles suffixes like "(Homebrew)")
        output = stdout.decode().strip()
        match = re.search(r"QEMU emulator version (\d+)\.(\d+)(?:\.(\d+))?", output)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            patch = int(match.group(3)) if match.group(3) else 0
            logger.debug("QEMU version detected", extra={"version": f"{major}.{minor}.{patch}"})
            return (major, minor, patch)

        logger.warning("Could not parse QEMU version", extra={"output": output[:100]})
        return None

    except FileNotFoundError:
        logger.warning("QEMU binary not found", extra={"qemu_bin": qemu_bin})
        return None
    except (OSError, TimeoutError) as e:
        if e.errno in _NOT_A_PROGRAM_HERE:  # TimeoutError: no errno
            probe_cache.qemu_unrunnable = (qemu_bin, e)  # probe_qemu_runnable gives it
        logger.warning("QEMU version probe failed", extra={"error": str(e)})
        return None


_NOT_A_PROGRAM_HERE: Final = frozenset({errno.ENOEXEC, getattr(errno, "EBADARCH", errno.ENOEXEC)})
"""Exec errors of a file that this host cannot run at all: another CPU architecture or no
program (ENOEXEC; EBADARCH on macOS)."""


async def probe_qemu_runnable() -> tuple[str, OSError] | None:
    """The QEMU binary and its exec error when this host cannot run it at all (another CPU
    architecture, no program: _NOT_A_PROGRAM_HERE), else None (cached). A capability, as the
    other probes: VmManager.start asks it with them, and a use that needs QEMU raises
    qemu_unrunnable_error. A launch cannot always tell: behind unshare or stdbuf (glibc
    execvp), or under busybox ash, such a file runs as a shell script and QEMU_CRASHED comes
    after the boot retries. The exec of probe_qemu_version answers it (one exec for both)."""
    await probe_qemu_version()
    return probe_cache.qemu_unrunnable


def qemu_unrunnable_error(unrunnable: tuple[str, OSError]) -> SandboxError:
    """The error of a QEMU binary that this host cannot run (probe_qemu_runnable): ConfigError
    HOST_DEPENDENCY that names the binary and the CPU build to install. No VM can start."""
    binary, error = unrunnable
    return spawn_error(
        error,
        binary=binary,
        code=ErrorCode.QEMU_CRASHED,
        context={},
        hint=f"Install the QEMU build for this host's CPU ({detect_host_arch().name.lower()}).",
    )


_SANDBOX_PROBE_TIMEOUT_SECONDS: Final = 10.0
"""Upper bound for the QMP greeting of the sandbox probe; it arrives in ~10 ms."""


@_async_cached_probe("qemu_sandbox")
async def probe_qemu_sandbox_support() -> bool:
    """Probe whether the QEMU binary supports -sandbox (seccomp) (cached).

    The -sandbox option requires CONFIG_SECCOMP at QEMU compile time.  When
    absent, QEMU rejects the flag at CLI parsing with "There is no option
    group 'sandbox'" (util/qemu-config.c:find_list).  Neither ``-sandbox help``
    nor ``--help`` output can reliably detect this (both produce false results),
    and ``-version`` exits before QEMU installs the filter.

    Strategy: launch ``qemu-system-xxx -machine none -sandbox <QEMU_SANDBOX_POLICY> -S
    -display none -qmp stdio``.  QEMU prints the QMP greeting only after its start-up, which
    installs the seccomp filter, so the greeting proves both that the option is
    compiled in and that this kernel accepts the filter (~10 ms).  If -sandbox
    is unrecognised or the filter fails, QEMU exits with code 1 and no greeting.

    Returns:
        True if -sandbox is available, False otherwise.
    """
    # Seccomp is a Linux kernel feature — skip the subprocess on macOS.
    if detect_host_os() != HostOS.LINUX:
        return False

    qemu_bin = _probed_qemu_binary()

    try:
        proc = await asyncio.create_subprocess_exec(
            qemu_bin,
            "-machine",
            "none",
            "-sandbox",
            constants.QEMU_SANDBOX_POLICY,  # the filter VMs install
            "-S",
            "-display",
            "none",
            "-qmp",
            "stdio",
            stdin=asyncio.subprocess.PIPE,  # kept open: an EOF on the QMP chardev is not our signal
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env(),
        )
        assert proc.stdout is not None and proc.stderr is not None  # noqa: S101 - PIPE above
        # stderr drains while the probe waits: a full pipe would block QEMU before its greeting
        stderr_task = asyncio.create_task(proc.stderr.read())
        running = True  # still running at the bound, without a greeting
        try:
            async with asyncio.timeout(_SANDBOX_PROBE_TIMEOUT_SECONDS):
                while line := await proc.stdout.readline():
                    if line.startswith(b'{"QMP"'):
                        break
                else:
                    running = False  # stdout EOF: QEMU exited before its greeting
        except (TimeoutError, ValueError):  # ValueError: a line over the stream limit, not the greeting
            logger.warning(
                "QEMU sandbox probe: no QMP greeting, assuming supported",
                extra={"qemu_bin": qemu_bin, "timeout_seconds": _SANDBOX_PROBE_TIMEOUT_SECONDS},
            )
        finally:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()  # every path: the probe never leaves a QEMU behind
            await proc.wait()
            stderr = await stderr_task
        if running:
            # The greeting comes after start-up, which installs the filter. A silent QEMU that
            # still runs accepted -sandbox (an unknown option or a failed filter exits at once).
            logger.debug("QEMU sandbox probe: supported (-sandbox on, QEMU started)")
            return True

        stderr_text = stderr.decode(errors="replace").strip() if stderr else ""
        if "option group" in stderr_text or "sandbox" in stderr_text.lower():
            logger.info(
                "QEMU sandbox not available (compiled without CONFIG_SECCOMP)",
                extra={"qemu_bin": qemu_bin, "stderr": stderr_text[:200]},
            )
        else:
            logger.warning(
                "QEMU sandbox probe failed",
                extra={"qemu_bin": qemu_bin, "returncode": proc.returncode, "stderr": stderr_text[:200]},
            )
        return False

    except FileNotFoundError:
        logger.warning("QEMU binary not found for sandbox probe", extra={"qemu_bin": qemu_bin})
        return False
    except OSError as e:
        logger.warning("QEMU sandbox probe failed", extra={"qemu_bin": qemu_bin, "error": str(e)})
        return False


@_async_cached_probe("kvm")
async def check_kvm_available() -> bool:
    """Check if KVM acceleration is available and accessible (cached).

    Two-layer verification approach
    ===============================
    Layer 1 (Kernel): Verify /dev/kvm exists, is accessible, and responds to ioctl
    Layer 2 (QEMU):   Verify QEMU binary has KVM support via `-accel help`

    This 2-layer approach catches edge cases where:
    - /dev/kvm exists but KVM module is broken (nested VMs, containers)
    - KVM ioctl works but QEMU doesn't have KVM compiled in
    - GitHub Actions runners with inconsistent KVM availability

    References:
    - GitHub Actions KVM issues: https://github.com/orgs/community/discussions/8305
    - libvirt capability probing: https://libvirt.org/drvqemu.html

    KVM vs TCG: Virtualization modes with vastly different characteristics
    ======================================================================

    KVM (Kernel-based Virtual Machine) - Production mode:
    - Hardware-assisted virtualization (Intel VT-x / AMD-V)
    - VM boot time: <400ms (with snapshot cache)
    - CPU overhead: near-native performance (~5% penalty)
    - Security: Hardware-enforced memory isolation (EPT/NPT)
    - Requirements: Linux host + KVM kernel module + /dev/kvm device
    - Use case: Production deployments, CI/CD

    TCG (Tiny Code Generator) - Development fallback:
    - Software-based CPU emulation (no hardware virtualization)
    - VM boot time: 2-5s (5-10x slower than KVM)
    - CPU overhead: ~5-8x slower (instruction-level emulation)
    - Security: Software-based isolation (weaker than hardware)
    - Requirements: Any platform (Linux, macOS, Windows)
    - Use case: Development/testing only (macOS Docker Desktop)

    Production requirement: KVM is MANDATORY for performance and security.
    TCG is acceptable ONLY for local development and testing.

    Returns:
        True if both kernel and QEMU verify KVM is available
        False otherwise (falls back to TCG software emulation)
    """
    kvm_path = "/dev/kvm"
    if not Path(kvm_path).exists():
        logger.debug("KVM not available: /dev/kvm does not exist")
        return False

    # Check if we can actually access /dev/kvm (not just that it exists)
    # This catches permission issues that would cause QEMU to fail or hang
    # See: https://github.com/actions/runner-images/issues/8542
    # A WARNING, once per process (this probe is cached): else VMs silently run under TCG.
    if not can_access(kvm_path, os.R_OK | os.W_OK):
        logger.warning(
            "KVM not available: this user cannot read and write /dev/kvm, so VMs use software emulation "
            "(TCG, about 5-8 times slower). Add the user to the kvm group (sudo usermod -aG kvm $USER), "
            "then log in again.",
            extra={"path": kvm_path},
        )
        return False

    # Actually try to open /dev/kvm and check API version via subprocess
    # Some environments (nested VMs, containers) have /dev/kvm but it doesn't work
    # Uses subprocess to avoid blocking the event loop with ioctl()
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            f"import fcntl; f=open('{kvm_path}','rb'); print(fcntl.ioctl(f.fileno(), {_KVM_GET_API_VERSION}))",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=child_env(),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode != 0:
            logger.warning("KVM device accessible but ioctl failed")
            return False

        api_version = int(stdout.decode().strip())
        if api_version != _KVM_API_VERSION_EXPECTED:
            logger.warning(
                "KVM available but unexpected API version",
                extra={"api_version": api_version, "expected": _KVM_API_VERSION_EXPECTED},
            )
            return False

        logger.debug("KVM ioctl check passed", extra={"api_version": api_version})

    except (OSError, TimeoutError, ValueError) as e:
        logger.debug("KVM not available: failed to verify /dev/kvm", extra={"error": str(e)})
        return False

    # Layer 2: Verify QEMU binary has KVM support compiled in
    # Even if /dev/kvm works, QEMU may not have KVM support or may fail to initialize it
    # See: https://github.com/orgs/community/discussions/8305 (GitHub Actions KVM issues)
    qemu_accels = await probe_qemu_accelerators()
    if "kvm" not in qemu_accels:
        logger.warning(
            "KVM not available: QEMU binary does not support KVM accelerator",
            extra={"available_accelerators": sorted(qemu_accels)},
        )
        return False

    logger.debug("KVM available and working (kernel + QEMU verified)")
    return True


@_async_cached_probe("hvf")
async def check_hvf_available() -> bool:
    """Check if HVF (Hypervisor.framework) acceleration is available on macOS (cached).

    HVF requires:
    - macOS host (automatically implied by caller)
    - CPU with virtualization extensions
    - Hypervisor entitlement (usually available)
    - NOT running inside a VM without nested virtualization

    GitHub Actions macOS runners run inside VMs without nested virtualization,
    so HVF is not available there. This check detects that case.

    Returns:
        True if HVF is available and can be used
        False otherwise (falls back to TCG software emulation)
    """
    try:
        # sysctl kern.hv_support returns 1 if Hypervisor.framework is available
        proc = await asyncio.create_subprocess_exec(
            "/usr/sbin/sysctl",
            "-n",
            "kern.hv_support",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=child_env(),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        hvf_kernel_support = proc.returncode == 0 and stdout.decode().strip() == "1"

        if not hvf_kernel_support:
            logger.debug("HVF not available: kern.hv_support is not enabled")
            return False

        logger.debug("HVF kernel support check passed")

    except (OSError, TimeoutError) as e:
        logger.debug("HVF not available: sysctl check failed", extra={"error": str(e)})
        return False

    # Layer 2: Verify QEMU binary has HVF support compiled in
    # Even if kern.hv_support is enabled, QEMU may not have HVF support
    qemu_accels = await probe_qemu_accelerators()
    if "hvf" not in qemu_accels:
        logger.warning(
            "HVF not available: QEMU binary does not support HVF accelerator",
            extra={"available_accelerators": sorted(qemu_accels)},
        )
        return False

    logger.debug("HVF available and working (kernel + QEMU verified)")
    return True


async def check_hwaccel_available() -> bool:
    """Check if hardware acceleration (KVM or HVF) is available.

    TCG (software emulation) is ~5-8x slower than hardware virtualization,
    making timing-sensitive tests unreliable.

    Returns:
        True if KVM (Linux) or HVF (macOS) is available
        False otherwise (will use TCG software emulation)
    """
    host_os = detect_host_os()

    if host_os == HostOS.LINUX:
        return await check_kvm_available()
    if host_os == HostOS.MACOS:
        return await check_hvf_available()
    return False


@_async_cached_probe("tsc_deadline")
async def check_tsc_deadline() -> bool:
    """Check if TSC_DEADLINE CPU feature is available (cached).

    TSC_DEADLINE is required to disable PIT (i8254) and PIC (i8259) in microvm.
    Without TSC_DEADLINE, the APIC timer cannot use deadline mode, and the system
    needs the legacy PIT for timer interrupts.

    In nested virtualization (e.g., GitHub Actions runners), TSC_DEADLINE may not
    be exposed to the guest, causing boot hangs if PIT/PIC are disabled.

    See: https://www.qemu.org/docs/master/system/i386/microvm.html

    Returns:
        True if TSC_DEADLINE is available, False otherwise
    """
    # TSC_DEADLINE is x86-only
    if detect_host_arch() != HostArch.X86_64:
        return False

    # Dispatch to platform-specific implementation
    host_os = detect_host_os()
    if host_os == HostOS.LINUX:
        return await _check_tsc_deadline_linux()
    if host_os == HostOS.MACOS:
        return await _check_tsc_deadline_macos()

    # Unknown platform
    return False


async def _check_tsc_deadline_linux() -> bool:
    """Linux-specific TSC_DEADLINE check via /proc/cpuinfo.

    Note: Called from check_tsc_deadline() which handles caching and locking.
    """
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return False

    try:
        # In a worker thread: the kernel builds this file for every CPU (large on
        # big hosts; some kernels sample each CPU's frequency while doing so)
        cpuinfo = await asyncio.to_thread(cpuinfo_path.read_text)
        # Look for tsc_deadline_timer in the flags line
        # Format: "flags : fpu vme ... tsc_deadline_timer ..."
        for line in cpuinfo.split("\n"):
            if line.startswith("flags"):
                has_tsc = "tsc_deadline_timer" in line.split()
                if has_tsc:
                    logger.debug("TSC_DEADLINE available (can disable PIT/PIC)")
                else:
                    logger.debug("TSC_DEADLINE not available (keeping PIT/PIC enabled)")
                return has_tsc
    except OSError as e:
        logger.warning("Failed to read /proc/cpuinfo for TSC_DEADLINE check", extra={"error": str(e)})

    return False


async def _check_tsc_deadline_macos() -> bool:
    """macOS-specific TSC_DEADLINE check via sysctl.

    Note: Called from check_tsc_deadline() which handles caching and locking.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "/usr/sbin/sysctl",
            "-n",
            "machdep.cpu.features",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=child_env(),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode == 0:
            features = stdout.decode().upper()
            has_tsc = "TSC_DEADLINE" in features or "TSCDEAD" in features
            if has_tsc:
                logger.debug("TSC_DEADLINE available on macOS (can disable PIT/PIC)")
            else:
                logger.debug("TSC_DEADLINE not available on macOS (keeping legacy timers)")
            return has_tsc
    except (OSError, TimeoutError):
        pass

    return False


def can_access(path: Path | str, mode: int) -> bool:
    """Whether this user can access `path` with `mode` (os.R_OK, os.W_OK, os.X_OK, or a
    combination): one metadata call."""
    return os.access(path, mode)


_GETXATTR: Final = cast("Callable[[Path, str], bytes] | None", getattr(os, "getxattr", None))
"""The xattr read call, or None: this OS has none in Python (macOS)."""

POSIX_ACL_ACCESS_XATTR: Final = "system.posix_acl_access"
"""Where a filesystem keeps the POSIX access ACL of a file (acl(5)): a version, then
(tag, perm, id) entries (include/uapi/linux/posix_acl_xattr.h)."""


def posix_acl_xattr(path: Path) -> bytes | None:
    """The POSIX access ACL of `path` as the filesystem keeps it (POSIX_ACL_ACCESS_XATTR), or
    None: no ACL beyond the mode bits, or no POSIX ACLs here. Whether the filesystem keeps
    POSIX ACLs is probed once per device (ProbeCache.posix_acls): an OS with no xattr call
    (macOS: its ACLs are another model, read with acl_get_fd_np) or a filesystem that refuses
    the name (EOPNOTSUPP: no ACL support, mounted noacl) has none, whatever the OS is called.
    Another error (access, a file gone) is this call's only."""
    if _GETXATTR is None:
        return None
    try:
        device = path.stat().st_dev
    except OSError:
        return None
    if probe_cache.posix_acls.get(device) is False:
        return None
    try:
        raw = _GETXATTR(path, POSIX_ACL_ACCESS_XATTR)
    except OSError as e:
        if e.errno == errno.EOPNOTSUPP:  # ENOTSUP too (one value on Linux)
            probe_cache.posix_acls[device] = False
        elif e.errno == errno.ENODATA:
            probe_cache.posix_acls[device] = True
        return None
    probe_cache.posix_acls[device] = True
    return raw


_O_DIRECT: Final = cast("int | None", getattr(os, "O_DIRECT", None))
"""The open flag of direct IO, or None: this OS has none (macOS, where F_NOCACHE is another
thing, and QEMU's direct-io needs O_DIRECT)."""


async def probe_direct_io(folder: Path) -> bool:
    """Whether a file in `folder` takes O_DIRECT, as QEMU opens its multifd channels for
    direct-io: the filesystem answers (cached per device, st_dev), whatever the OS; False where
    the OS has no O_DIRECT. One probe file per device (the kernel makes it before it refuses
    O_DIRECT, so it goes on every path). EINVAL is the filesystem's answer, cached and warned
    once per device; another error (no space, quota, file handles, access) is this call's
    only: False, not cached."""
    if _O_DIRECT is None:
        return False
    try:
        device = (await asyncio.to_thread(folder.stat)).st_dev
    except OSError:
        return False
    if (known := probe_cache.direct_io.get(device)) is not None:
        return known
    async with probe_cache.get_lock(f"direct_io:{device}"):
        if (known := probe_cache.direct_io.get(device)) is not None:
            return known
        takes = await asyncio.to_thread(_opens_direct, folder, flag=_O_DIRECT)
        if takes is not None:
            probe_cache.direct_io[device] = takes
        return bool(takes)


_CHECKSUMMED_FILESYSTEMS: Final = {0x9123683E: "btrfs", 0x2FC12FC1: "zfs"}
"""statfs f_type of the filesystems that checksum the data of a direct write from the writer's
own memory (linux/magic.h BTRFS_SUPER_MAGIC; ZFS's ZFS_SUPER_MAGIC)."""


async def probe_drive_direct_io(folder: Path) -> bool:
    """Whether QEMU writes the drive files of a VM in `folder` (overlay, home disk, the layer
    of a snapshot build) with O_DIRECT (qemu_cmd), cached per folder (a fixed one: the caller
    passes the folder of the VM folders, not one per VM): once known, a VM start asks no
    thread. Yes when the filesystem takes O_DIRECT (probe_direct_io) and is neither memory
    (tmpfs, ramfs: the file is memory charged to the VM either way, and O_DIRECT only adds
    read-modify-writes) nor btrfs or ZFS (they checksum a direct write from the writer's
    memory: a guest page that changes meanwhile is stored with a bad checksum or fails with
    EIO; btrfs from Linux 6.14 writes buffered anyway). No off Linux (no memory cgroup to
    protect). A probe that failed for another reason (no space, file handles) is not cached:
    the next VM start asks again."""
    if sys.platform != "linux":
        return False
    if (known := probe_cache.drive_direct_io.get(folder)) is not None:
        return known
    try:
        device = await asyncio.to_thread(_drive_filesystem_device, folder)
    except OSError:  # the folder cannot be read (not made yet): no answer, asked again
        return False
    if device is None:
        probe_cache.drive_direct_io[folder] = False
        return False
    takes = await probe_direct_io(folder)
    if device in probe_cache.direct_io:  # a definite answer of the filesystem
        probe_cache.drive_direct_io[folder] = takes
    return takes


def _drive_filesystem_device(folder: Path) -> int | None:
    """The device (st_dev) of the filesystem of `folder` when O_DIRECT drive files can help
    there (not memory-backed, not checksummed: probe_drive_direct_io), else None. An overlay
    (a container's root) is judged by the filesystem of its upper folder, where its writes go;
    one whose upper folder this process cannot see: None. Raises OSError when the folder
    cannot be read."""
    fs_type = statfs_type(folder)
    if fs_type is None:
        raise OSError(f"statfs of {folder} failed")
    if fs_type == OVERLAYFS_SUPER_MAGIC:
        upper = overlay_upper(folder)
        fs_type = None if upper is None else statfs_type(upper)
    if fs_type is None or fs_type in LINUX_MEMORY_FILESYSTEMS or fs_type in _CHECKSUMMED_FILESYSTEMS:
        return None
    return folder.stat().st_dev


def _opens_direct(folder: Path, *, flag: int) -> bool | None:
    """Open a new probe file in `folder` with `flag` (O_DIRECT): True, False (the filesystem
    refuses it: EINVAL), or None (another error, this call's only). The file goes either way."""
    probe = folder / f".direct-io-probe-{os.getpid()}-{time.monotonic_ns()}"
    try:
        os.close(os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | flag, 0o600))
    except OSError as e:
        if e.errno != errno.EINVAL:
            logger.warning(
                "O_DIRECT probe of the folder failed: writes there are buffered this time",
                extra={"folder": str(folder), "error": e.strerror or str(e)},
            )
            return None
        logger.warning("Folder does not take O_DIRECT: writes there are buffered", extra={"folder": str(folder)})
        return False
    finally:
        with contextlib.suppress(OSError):
            probe.unlink()
    return True


@_async_cached_probe("io_uring")
async def probe_io_uring_support() -> bool:
    """Probe for io_uring support using syscall test (cached).

    Returns:
        True if io_uring fully available, False otherwise
    """
    # io_uring is Linux-only - immediately return False on other platforms
    if detect_host_os() != HostOS.LINUX:
        return False

    # Check 1: Sysctl restrictions (kernel 6.6+)
    sysctl_path = Path("/proc/sys/kernel/io_uring_disabled")
    if sysctl_path.exists():
        try:
            disabled_value = int(sysctl_path.read_text().strip())  # a sysctl: answered at once
            # io_uring_disabled sysctl values: 0=enabled, 1=restricted, 2=disabled
            if disabled_value == 2:  # noqa: PLR2004
                logger.info(
                    "io_uring disabled via sysctl",
                    extra={"sysctl_value": disabled_value},
                )
                return False
            if disabled_value == 1:
                logger.debug(
                    "io_uring restricted to CAP_SYS_ADMIN",
                    extra={"sysctl_value": disabled_value},
                )
        except (ValueError, OSError) as e:
            logger.warning("Failed to read io_uring_disabled sysctl", extra={"error": str(e)})

    # Check 2: Syscall probe via subprocess (avoids blocking event loop)
    # Uses subprocess to prevent blocking - ctypes syscall would block the event loop
    # Exit codes: 0=available (EINVAL/EFAULT), 1=not available (ENOSYS), 2=blocked (EPERM), 3=error
    try:
        probe_script = """
import ctypes
import errno
import sys

try:
    libc = ctypes.CDLL(None, use_errno=True)
    # __NR_io_uring_setup = 425
    result = libc.syscall(425, 0, None)
    if result == -1:
        err = ctypes.get_errno()
        if err == errno.ENOSYS:
            sys.exit(1)  # Not available
        if err in (errno.EINVAL, errno.EFAULT):
            sys.exit(0)  # Available (kernel recognized syscall)
        if err == errno.EPERM:
            sys.exit(2)  # Blocked by seccomp/container
        sys.exit(3)  # Unexpected error
    sys.exit(0)  # Available
except Exception:
    sys.exit(3)  # Error
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            probe_script,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=child_env(),
        )
        await asyncio.wait_for(proc.wait(), timeout=5)

        if proc.returncode == 0:
            logger.info(
                "io_uring syscall available",
                extra={"kernel": platform.release()},
            )
            return True
        if proc.returncode == 1:
            logger.info(
                "io_uring syscall not available (ENOSYS)",
                extra={"kernel": platform.release()},
            )
            return False
        if proc.returncode == 2:  # noqa: PLR2004
            logger.warning(
                "io_uring blocked by seccomp/container policy",
                extra={"kernel": platform.release()},
            )
            return False

        logger.warning(
            "io_uring probe failed with unexpected result",
            extra={"exit_code": proc.returncode},
        )
        return False

    except (OSError, TimeoutError) as e:
        logger.warning(
            "io_uring syscall probe failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return False


@_async_cached_probe("unshare")
async def probe_unshare_support() -> bool:
    """Probe for unshare (Linux namespace) support (cached).

    Tests if the current environment allows creating new namespaces via unshare.
    This requires either:
    - Root privileges
    - CAP_SYS_ADMIN capability
    Unprivileged user namespaces do not help: neither the probe nor the VM command
    makes a user namespace (no --user).

    Returns:
        True if unshare works, False otherwise (skip namespace isolation)
    """
    # Skip on non-Linux - unshare is Linux-specific
    if detect_host_os() == HostOS.MACOS:
        return False

    try:
        # Test unshare with minimal namespaces (pid requires fork)
        proc = await asyncio.create_subprocess_exec(
            "/usr/bin/unshare",
            "--pid",
            "--fork",
            "--kill-child=SIGKILL",  # the VM command uses it: util-linux 2.32+
            "--",
            "/usr/bin/true",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=child_env(),
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)

        if proc.returncode == 0:
            logger.info("unshare available (namespace isolation enabled)")
            return True

        stderr_text = stderr.decode().strip() if stderr else ""
        logger.warning(
            "unshare unavailable (namespace isolation disabled)",
            extra={"exit_code": proc.returncode, "stderr": stderr_text[:200]},
        )
        return False
    except (OSError, TimeoutError) as e:
        logger.warning(
            "unshare probe failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return False


@_async_cached_probe("setpriv")
async def probe_setpriv_pdeathsig() -> bool:
    """Whether `setpriv --pdeathsig` works (util-linux 2.33+; cached). Children that end with
    this process (subprocess_utils.start_managed_process) then need no Python preexec_fn, so
    Popen can vfork. Else one warning: each such launch forks this process. False off Linux."""
    if detect_host_os() != HostOS.LINUX:
        return False
    try:
        # The launch form: setpriv, then /bin/sh (subprocess_utils._launch_command)
        proc = await asyncio.create_subprocess_exec(
            "setpriv",
            "--pdeathsig",
            "KILL",
            "--",
            "/bin/sh",
            "-c",
            ":",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=child_env(),
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
    except (OSError, TimeoutError) as e:
        reason: dict[str, object] = {"error": str(e), "error_type": type(e).__name__}
    else:
        if proc.returncode == 0:
            return True
        reason = {"exit_code": proc.returncode, "stderr": stderr.decode(errors="replace").strip()[:200]}
    logger.warning(
        "setpriv --pdeathsig unavailable (util-linux 2.33+): each child that ends with this process "
        "(QEMU as this user, gvproxy, qemu-storage-daemon) starts through a fork of this process",
        extra=reason,
    )
    return False


async def detect_accel_type(*, force_emulation: bool = False) -> AccelType:
    """Detect which QEMU accelerator to use.

    This is the single source of truth for virtualization mode detection.
    Used for both cgroup memory sizing (TCG needs more) and QEMU command building.

    Args:
        force_emulation: Force TCG software emulation

    Returns:
        AccelType.KVM if Linux KVM available
        AccelType.HVF if macOS HVF available
        AccelType.TCG if software emulation needed (or force_emulation=True)
    """
    if force_emulation:
        return AccelType.TCG
    if await check_kvm_available():
        return AccelType.KVM
    if detect_host_os() == HostOS.MACOS and await check_hvf_available():
        return AccelType.HVF
    return AccelType.TCG

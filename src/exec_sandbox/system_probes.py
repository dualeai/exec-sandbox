"""System capability probes for VM acceleration and feature detection.

These probes detect system capabilities once and cache the results.
Async probes use a shared cache container to avoid global statements.
The Scheduler start also raises the process limits it can (raise_fd_limit,
raise_nproc_limit) and warns about unsafe host settings (warn_*).
"""

import asyncio
import contextlib
import ctypes
import errno
import fcntl
import functools
import json
import mmap
import os
import platform
import re
import resource
import shutil
import signal
import sys
import time
import weakref
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar, Final, ParamSpec, TypeVar, cast

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import await_cancellation_safe
from exec_sandbox.cgroup import CGROUP_V2_BASE_PATH, cgroup_v2_available, resolve_self_cgroup_v2
from exec_sandbox.exceptions import (
    HOST_LIMIT_ERRNOS,
    ConfigError,
    ErrorCode,
    SandboxError,
    TransientError,
    spawn_error,
)
from exec_sandbox.files.disk import LINUX_MEMORY_FILESYSTEMS, OVERLAYFS_SUPER_MAGIC, overlay_upper, statfs_type
from exec_sandbox.platform_utils import HostArch, HostOS, child_env, detect_host_arch, detect_host_os
from exec_sandbox.settings import QemuBinaries
from exec_sandbox.vm_types import AccelType

P = ParamSpec("P")
T = TypeVar("T")

logger = get_logger(__name__)

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

    At 100 concurrent VMs the default 1024 soft limit is exhausted: the opens, pipes and
    sockets of this process then fail with EMFILE ("Too many open files").

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
    if (overcommit := _kernel_int("/proc/sys/vm/overcommit_memory")) == 0:
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
    max_map_count = _kernel_int("/proc/sys/vm/max_map_count")
    if max_map_count is not None and max_map_count < _MIN_RECOMMENDED_MAX_MAP_COUNT:
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
    threads_max = _kernel_int("/proc/sys/kernel/threads-max")
    if threads_max is not None and threads_max < _MIN_RECOMMENDED_THREADS_MAX:
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
    """Read a zswap sysfs parameter, or None if unavailable (not Linux, no zswap module)."""
    try:
        return Path(f"{_ZSWAP_PARAM_BASE}/{name}").read_text().strip()
    except (OSError, ValueError):
        return None


def _kernel_int(path: str) -> int | None:
    """The integer in the kernel file `path` (/proc/sys, /sys), or None: no such file (not
    Linux, a module not loaded), no access, or not an integer. The warn_* checks skip then."""
    try:
        return int(Path(path).read_text().strip())
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
    pool_pct = _kernel_int(f"{_ZSWAP_PARAM_BASE}/max_pool_percent")
    if pool_pct is not None and pool_pct < 30:  # noqa: PLR2004
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
_KVM_CHECK_EXTENSION = 0xAE03
_KVM_CAP_TSC_CONTROL = 60


@dataclass(frozen=True)
class _Stamped[R]:
    """A QEMU probe's result and the stamp of the binary it ran (_qemu_binary_stamp)."""

    stamp: tuple[str, int, int, int, int] | None
    result: R


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
        "cpu_id",
        "direct_io",
        "drive_direct_io",
        "failures",
        "guest_cpu",
        "host_counter_hz",
        "hvf",
        "io_uring",
        "kvm",
        "kvm_tsc_scaling",
        "qemu_accels",
        "qemu_as_qemu_vm",
        "qemu_machine_types",
        "qemu_sandbox",
        "qemu_version",
        "qemu_vm_user",
        "setpriv",
        "tsc_deadline",
        "unshare",
    )
    _RESULTS: ClassVar[tuple[str, ...]] = tuple(name for name in __slots__ if name != "_locks")

    def __init__(self) -> None:
        self.cpu_id: str | None = NOT_CACHED  # the host CPU a -cpu host guest sees (ARM64)
        self.direct_io: dict[int, bool] = {}  # per device (st_dev): its filesystem takes O_DIRECT
        self.drive_direct_io: dict[Path, bool] = {}  # per folder: VM drive files there use O_DIRECT
        # Per (QEMU binary stamp, accelerator, -cpu value): the guest TSC rate QEMU reads (require_guest_cpu)
        self.guest_cpu: dict[tuple[tuple[str, int, int, int, int] | None, str, str], int | None] = {}
        self.host_counter_hz: int | None = NOT_CACHED  # the clock counter rate, where exact
        self.hvf: bool | None = NOT_CACHED
        self.io_uring: bool | None = NOT_CACHED
        self.kvm: bool | None = NOT_CACHED
        self.kvm_tsc_scaling: bool = NOT_CACHED  # KVM_CAP_TSC_CONTROL
        # The QEMU probes keep the binary's stamp with their result (_Stamped): a new binary
        # at the same path (a package upgrade) runs them again
        self.qemu_accels: _Stamped[set[str]] = NOT_CACHED  # Accelerators available in QEMU binary
        self.qemu_machine_types: _Stamped[dict[str, str]] = NOT_CACHED  # Machine types of the QEMU binary
        self.qemu_sandbox: _Stamped[bool] = NOT_CACHED  # -sandbox works: CONFIG_SECCOMP, a kernel that takes its filter
        self.qemu_version: _Stamped[tuple[int, int, int] | None] = NOT_CACHED  # QEMU semver (major, minor, patch)
        self.qemu_vm_user: bool = NOT_CACHED  # the qemu-vm user exists (permission_utils)
        self.qemu_as_qemu_vm: bool = NOT_CACHED  # this process runs QEMU as qemu-vm (one decision)
        # The exception of the last failed run of each probe (by cache name), when one says why
        self.failures: dict[str, BaseException | None] = {}
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


class _ProbeKilled(Exception):  # noqa: N818 - a cause kept for _qemu_probe_failure
    """A probe QEMU that a signal ended (the OOM killer, a process limit): no answer, and a
    failure that can pass."""


def _exit_error(returncode: int) -> BaseException | None:
    """The cause to keep for a probe that exited with `returncode` (not 0): _ProbeKilled for a
    signal, else None (QEMU's own exit: its answer is no answer)."""
    return _ProbeKilled(f"killed by signal {-returncode}") if returncode < 0 else None


class _Uncached(Exception):  # noqa: N818 - a result, not an error
    """Raised by a cached probe that failed (a timeout, a fork error, a binary half written by
    an upgrade): the decorator returns `result` and caches nothing, so the next call probes
    again. A real answer (QEMU built without seccomp) is returned and cached."""

    def __init__(self, *, result: object, error: BaseException | None) -> None:
        super().__init__()
        self.result = result
        self.error = error
        """Why the probe failed, when an exception says it (probe_cache.failures keeps it)."""


def _qemu_binary_stamp() -> tuple[str, int, int, int, int] | None:
    """The QEMU binary the probes run, as (path, device, inode, size, modification time in
    ns); None when no file is found. A package upgrade writes a new file: its stamp differs."""
    binary = _probed_qemu_binary()
    path = binary if os.sep in binary else _on_path(binary)
    try:
        stat = Path(path).stat() if path is not None else None
    except OSError:
        return None
    return None if stat is None else (binary, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def async_cached_probe(
    cache_attr: str,
    *,
    per_qemu_binary: bool,
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Decorator for async probes with double-checked locking cache.

    Eliminates boilerplate pattern:
        if cache.attr is not NOT_CACHED: return cache.attr
        async with cache.get_lock("attr"):
            if cache.attr is not NOT_CACHED: return cache.attr
            result = await actual_work()
            cache.attr = result
            return result

    Uses a sentinel value (NOT_CACHED) to distinguish "not yet cached" from a cached None.
    A probe that failed raises _Uncached: the call returns its `result`, nothing is cached,
    and the next call runs the probe again. Any value the probe returns is cached.

    Args:
        cache_attr: Name of the attribute on probe_cache to use for caching
        per_qemu_binary: the result belongs to the QEMU binary: it is kept with the
            binary's stamp, taken before the probe runs, and a call that finds another
            stamp runs the probe again (a stat and cached lookups per call: about 5 us)

    Returns:
        Decorated async function with automatic caching
    """

    def decorator(func: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Coroutine[Any, Any, T]]:
        def cached(stamp: tuple[str, int, int, int, int] | None) -> Any:
            value = getattr(probe_cache, cache_attr)
            if not per_qemu_binary or value is NOT_CACHED:
                return value
            return value.result if value.stamp == stamp else NOT_CACHED

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            stamp = _qemu_binary_stamp() if per_qemu_binary else None
            # Fast path: return cached result (no lock needed)
            if (value := cached(stamp)) is not NOT_CACHED:
                return value
            # Slow path: acquire lock to prevent stampede, then check cache again
            async with probe_cache.get_lock(cache_attr):
                # Double-check after acquiring lock (another task may have populated cache)
                if (value := cached(stamp)) is not NOT_CACHED:
                    return value
                try:
                    result = await func(*args, **kwargs)
                except _Uncached as failed:  # a failure can pass: the next call runs the probe again
                    probe_cache.failures[cache_attr] = failed.error
                    return cast("T", failed.result)
                probe_cache.failures.pop(cache_attr, None)
                setattr(probe_cache, cache_attr, _Stamped(stamp, result) if per_qemu_binary else result)
                return result

        return wrapper

    return decorator


def qemu_system_binary(binaries: QemuBinaries, *, arch: HostArch) -> str:
    """The qemu-system binary for `arch`: the configured file (EXEC_SANDBOX_QEMU_BIN_X86 /
    _ARM, /usr/bin by default) when it exists, else the file of the bare name on PATH (its
    full path, _on_path), else the bare name (no such file: exec reports it). The one
    resolver: VMs (qemu_cmd) and the QEMU probes run the binary it gives, so the probe
    answers (by its stamp) are those of the file each VM runs, also behind sudo, whose PATH
    differs."""
    configured = binaries.qemu_bin_arm if arch == HostArch.AARCH64 else binaries.qemu_bin_x86
    if configured.is_file():
        return str(configured)
    name = "qemu-system-aarch64" if arch == HostArch.AARCH64 else "qemu-system-x86_64"
    return _on_path(name) or name


def _probed_qemu_binary() -> str:
    """The binary the QEMU probes run: VMs read the same two variables (Settings is a
    QemuBinaries; SchedulerConfig does not set them). The settings are read once for each
    value of the variables (40 us each, on every QEMU probe call)."""
    return qemu_system_binary(_qemu_binaries(_qemu_variables()), arch=detect_host_arch())


_QEMU_VARIABLES: Final = tuple(
    name for base in ("EXEC_SANDBOX_QEMU_BIN_X86", "EXEC_SANDBOX_QEMU_BIN_ARM") for name in (base, base.lower())
)
"""The two QemuBinaries variables, upper and lower case (QemuBinaries reads any case)."""


def _qemu_variables() -> tuple[str | None, ...]:
    """The values of _QEMU_VARIABLES: four lookups (reading all of os.environ costs 15 us)."""
    return tuple(os.environ.get(name) for name in _QEMU_VARIABLES)


@functools.lru_cache(maxsize=8)
def _qemu_binaries(_variables: tuple[str | None, ...]) -> QemuBinaries:
    """QemuBinaries for these values of its variables (the cache key: it reads them)."""
    return QemuBinaries()


_FOUND_ON_PATH: dict[tuple[str, str | None], str] = {}
"""Files found on PATH, by (name, PATH): the stamp stats this file; a name not found is
looked up again at the next call."""


def _on_path(name: str) -> str | None:
    """The file of `name` on PATH, executable or not (a file this user cannot run is no
    missing binary)."""
    key = (name, os.environ.get("PATH"))
    if (found := _FOUND_ON_PATH.get(key)) is not None and not Path(found).exists():
        del _FOUND_ON_PATH[key]  # moved or removed: look it up again
        found = None
    if found is None and (found := shutil.which(name, mode=os.F_OK)) is not None:
        _FOUND_ON_PATH[key] = found
    return found


_PROBE_TIMEOUT_SECONDS: Final = 5.0
"""Upper bound of a probe command (sysctl, `qemu -accel help`, a sudo): each ends in ms."""


async def run_probe(cmd: Sequence[str], *, stdin: bytes | None, stdout: int, stderr: int) -> tuple[int, bytes, bytes]:
    """Run the probe command `cmd` to its end (child_env()), at most _PROBE_TIMEOUT_SECONDS:
    (exit status, stdout, stderr). On the timeout or a cancel, the command is killed and
    reaped before the call ends: no probe leaves a process behind. `stdin`: the bytes the
    command reads, then end of file; None: /dev/null. stdout and stderr:
    asyncio.subprocess.PIPE or DEVNULL.

    Raises:
        OSError: the command did not start (FileNotFoundError: not found)
        TimeoutError: it did not end within the bound
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL if stdin is None else asyncio.subprocess.PIPE,
        stdout=stdout,
        stderr=stderr,
        env=child_env(),
    )
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
            out, err = await proc.communicate(stdin)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await await_cancellation_safe(proc.wait())
        raise
    return cast("int", proc.returncode), out, err  # communicate() reaped it: never None


@async_cached_probe("qemu_accels", per_qemu_binary=True)
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
        returncode, stdout, _ = await run_probe(
            (qemu_bin, "-accel", "help"), stdin=None, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        if returncode != 0:
            logger.warning(
                "QEMU accelerator probe failed",
                extra={"qemu_bin": qemu_bin, "returncode": returncode},
            )
            raise _Uncached(result=set[str](), error=_exit_error(returncode))

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

    except OSError as e:  # TimeoutError too; check_kvm_available and check_hvf_available raise it
        logger.warning(
            "QEMU accelerator probe failed",
            extra={"qemu_bin": qemu_bin, "error": str(e)},
        )
        raise _Uncached(result=set[str](), error=e) from None


@async_cached_probe("qemu_version", per_qemu_binary=True)
async def probe_qemu_version() -> tuple[int, int, int] | None:
    """Probe QEMU binary version (cached).

    Returns:
        Semver tuple (major, minor, patch), or None if detection fails.
        Example: (8, 2, 0) for QEMU 8.2.0

    Tuples compare naturally as semver: (9, 2, 0) >= (9, 2, 0) works correctly.
    """
    qemu_bin = _probed_qemu_binary()

    try:
        returncode, stdout, _ = await run_probe(
            (qemu_bin, "--version"), stdin=None, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )

        if returncode != 0:
            logger.warning("QEMU version probe failed", extra={"returncode": returncode})
            raise _Uncached(result=None, error=_exit_error(returncode))

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
        raise _Uncached(result=None, error=None)

    except OSError as e:  # TimeoutError too; require_qemu_version raises it
        logger.warning("QEMU version probe failed", extra={"error": str(e)})
        raise _Uncached(result=None, error=e) from None


async def require_qemu_version() -> tuple[int, int, int]:
    """This worker's QEMU version (probe_qemu_version, cached). Every VM needs it (command
    line, migration, saved facts): VmManager.start asks it, so no VM starts without one.

    Raises:
        ConfigError: HOST_DEPENDENCY: the probe found no version (no binary, or one this host
            or this user cannot run, or an answer it cannot read)
        TransientError: QEMU_CRASHED: the probe hit a process, memory or file-handle limit,
            the binary was being written, a signal ended it, or QEMU did not answer in time
            (the next call probes again)
    """
    if (version := await probe_qemu_version()) is None:
        raise _qemu_probe_failure("qemu_version", "version", "--version", expected="'QEMU emulator version X.Y.Z'")
    return version


def _qemu_probe_failure(cache_attr: str, what: str, args: str, *, expected: str) -> SandboxError:
    """The error of a QEMU probe (`args`) that found no `what`, by the exception of its last
    run (probe_cache.failures): a binary this host cannot run at all (another CPU) names the
    build to install; a fork or file limit, or no answer in time, can pass (TransientError
    QEMU_CRASHED); else _qemu_probe_error (ConfigError HOST_DEPENDENCY)."""
    error = probe_cache.failures.get(cache_attr)
    binary = _probed_qemu_binary()
    if isinstance(error, OSError) and error.errno in _NOT_A_PROGRAM_HERE:
        # Only a probe tells: a launch behind unshare or stdbuf (glibc execvp), or under
        # busybox ash, runs such a file as a shell script, and QEMU_CRASHED comes after the
        # boot retries
        return spawn_error(
            error,
            binary=binary,
            code=ErrorCode.QEMU_CRASHED,
            context={},
            hint=f"Install the QEMU build for this host's CPU ({detect_host_arch().name.lower()}).",
        )
    if isinstance(error, (TimeoutError, _ProbeKilled)):
        return TransientError(
            f"QEMU did not answer {binary} {args} ({str(error) or 'no answer in time'}).",
            code=ErrorCode.QEMU_CRASHED,
            context={"binary": binary},
            hint=None,
            instance_id=None,
        )
    if isinstance(error, OSError) and error.errno == errno.ETXTBSY:  # an upgrade writes the binary now
        return TransientError(
            f"The QEMU binary {binary} is being written ({args}).",
            code=ErrorCode.QEMU_CRASHED,
            context={"binary": binary},
            hint=None,
            instance_id=None,
        )
    if isinstance(error, OSError) and error.errno in HOST_LIMIT_ERRNOS:
        return spawn_error(error, binary=binary, code=ErrorCode.QEMU_CRASHED, context={}, hint=None)
    return _qemu_probe_error(what, args, expected=expected)


def _qemu_probe_error(what: str, args: str, *, expected: str) -> ConfigError:
    """ConfigError HOST_DEPENDENCY of a QEMU probe (`args`) that found no `what`: no QEMU
    binary on this host, or one this user cannot run or whose answer is not `expected`."""
    binary = _probed_qemu_binary()
    if _qemu_binary_stamp() is None:
        variable = (
            "EXEC_SANDBOX_QEMU_BIN_ARM" if detect_host_arch() == HostArch.AARCH64 else "EXEC_SANDBOX_QEMU_BIN_X86"
        )
        return ConfigError(
            f"No QEMU binary {binary} on this host.",
            code=ErrorCode.HOST_DEPENDENCY,
            context={"binary": binary},
            hint=f"Install QEMU, or set {variable} to its path.",
            instance_id=None,
        )
    return ConfigError(
        f"The QEMU {what} could not be read ({binary} {args}).",
        code=ErrorCode.HOST_DEPENDENCY,
        context={"binary": binary},
        hint=f"`{binary} {args}` must print {expected} when this user runs it.",
        instance_id=None,
    )


async def require_qemu_machine_types() -> dict[str, str]:
    """This QEMU's machine types (probe_qemu_machine_types, cached per binary). Every VM needs
    them (the type its -M resolves to, kept for a save; the machine type check of a load):
    VmManager.start asks them, so no VM starts without them.

    Raises:
        ConfigError: HOST_DEPENDENCY: the probe found no machine type (no binary, or one this
            host or this user cannot run, or an answer it cannot read)
        TransientError: QEMU_CRASHED: the probe hit a process, memory or file-handle limit,
            the binary was being written, a signal ended it, or QEMU did not answer in time
            (the next call probes again)
    """
    if not (types := await probe_qemu_machine_types()):
        raise _qemu_probe_failure("qemu_machine_types", "machine types", "-machine help", expected="its machine types")
    return types


@async_cached_probe("qemu_machine_types", per_qemu_binary=True)
async def probe_qemu_machine_types() -> dict[str, str]:
    """The machine types this QEMU binary has (`-machine help`, cached), each with the type
    it resolves to: an alias ("virt") its versioned type ("virt-11.1"), any other type
    itself. Empty when the probe fails, and not cached: the next call probes again
    (require_qemu_machine_types raises then, so no VM starts)."""
    qemu_bin = _probed_qemu_binary()
    try:
        returncode, stdout, _ = await run_probe(
            (qemu_bin, "-machine", "help"),
            stdin=None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as e:  # TimeoutError too; require_qemu_machine_types raises it
        logger.warning("QEMU machine type probe failed", extra={"qemu_bin": qemu_bin, "error": str(e)})
        raise _Uncached(result={}, error=e) from None
    if returncode != 0:
        logger.warning("QEMU machine type probe failed", extra={"qemu_bin": qemu_bin, "returncode": returncode})
        raise _Uncached(result={}, error=_exit_error(returncode))
    # "Supported machines are:" then one "<name>  <description>" line per type; an alias
    # line ends with "(alias of <type>)"
    types: dict[str, str] = {}
    for line in stdout.decode(errors="replace").splitlines()[1:]:
        if line.strip():
            alias = re.search(r"\(alias of (\S+)\)\s*$", line)
            types[line.split()[0]] = alias.group(1) if alias else line.split()[0]
    return types


_NOT_A_PROGRAM_HERE: Final = frozenset({errno.ENOEXEC, getattr(errno, "EBADARCH", errno.ENOEXEC)})
"""Exec errors of a file that this host cannot run at all: another CPU architecture or no
program (ENOEXEC; EBADARCH on macOS)."""


_SANDBOX_PROBE_TIMEOUT_SECONDS: Final = 10.0
"""Upper bound for the QMP greeting of the sandbox probe; it arrives in ~10 ms."""


@async_cached_probe("qemu_sandbox", per_qemu_binary=True)
async def probe_qemu_sandbox_support() -> bool | None:
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
        True if -sandbox is available, or when QEMU still runs with no greeting after
        _SANDBOX_PROBE_TIMEOUT_SECONDS or wrote a line over the stream limit (assumed: an
        unknown option or a failed filter exits at once). False off Linux, or if QEMU ended
        before its greeting with a seccomp error (built without seccomp, a filter the kernel
        refuses) or its filter's SIGSYS. Both are kept. None: no answer (the probe did not start, a signal other than SIGSYS
        killed it, a limit, or another exit), nothing cached (require_qemu_sandbox_support
        raises).
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
            returncode = await proc.wait()
            stderr = await stderr_task
        if running:
            # The greeting comes after start-up, which installs the filter. A silent QEMU that
            # still runs accepted -sandbox (an unknown option or a failed filter exits at once).
            logger.debug("QEMU sandbox probe: supported (-sandbox on, QEMU started)")
            return True

        return _sandbox_answer(qemu_bin, returncode, stderr.decode(errors="replace").strip() if stderr else "")

    except OSError as e:
        logger.warning("QEMU sandbox probe failed", extra={"qemu_bin": qemu_bin, "error": str(e)})
        raise _Uncached(result=None, error=e) from None


def _sandbox_answer(qemu_bin: str, returncode: int, stderr_text: str) -> bool:
    """The answer of a sandbox probe QEMU that ended before its greeting. A False is kept, and
    every later VM starts without seccomp: only an answer about seccomp is one. Any other end
    is no answer (_Uncached: never kept; require_qemu_sandbox_support raises, so no VM starts
    on it)."""
    log = {"qemu_bin": qemu_bin, "returncode": returncode, "stderr": stderr_text[:200]}
    if returncode < 0 and -returncode != signal.SIGSYS:
        logger.warning("QEMU sandbox probe killed", extra=log)  # the OOM killer, a process limit
        raise _Uncached(result=None, error=_exit_error(returncode))
    if returncode == -signal.SIGSYS:
        logger.info("QEMU sandbox not available (its seccomp filter killed QEMU)", extra=log)
        return False
    if (limit := _limit_errno(stderr_text)) is not None:
        logger.warning("QEMU sandbox probe hit a limit", extra=log)
        raise _Uncached(result=None, error=OSError(limit, stderr_text[:200]))
    # QEMU names the option in each -sandbox error: "-sandbox on,...: There is no option group
    # 'sandbox'" (built without seccomp), "...: failed to install seccomp syscall filter in the
    # kernel" (a kernel that refuses it)
    if "sandbox" in stderr_text.lower() or "seccomp" in stderr_text.lower():
        logger.info("QEMU sandbox not available", extra=log)
        return False
    logger.warning("QEMU sandbox probe failed", extra=log)  # a library, memory, a file limit
    raise _Uncached(result=None, error=None)


_LIMIT_TEXTS: Final = {
    os.strerror(errno.ENOMEM): errno.ENOMEM,
    os.strerror(errno.EAGAIN): errno.EAGAIN,
    os.strerror(errno.EMFILE): errno.EMFILE,
    os.strerror(errno.ENFILE): errno.ENFILE,
}
"""QEMU's text of a memory, process or file limit (error_setg_errno appends strerror)."""


def _limit_errno(stderr_text: str) -> int | None:
    """The errno of a limit that QEMU's error text names (_LIMIT_TEXTS), else None."""
    return next((number for text, number in _LIMIT_TEXTS.items() if text in stderr_text), None)


async def require_qemu_sandbox_support() -> bool:
    """Whether VMs run under QEMU's seccomp filter (probe_qemu_sandbox_support, cached). False
    only as a final answer: not Linux, or a probe QEMU that ended before its greeting with a
    seccomp error (built without seccomp, a filter the kernel refuses) or its filter's SIGSYS.
    A probe with no answer starts no VM: without it, the VM would run without its filter.

    Raises:
        ConfigError: HOST_DEPENDENCY: the probe could not run the QEMU binary (no binary, or
            one this host or this user cannot run), or it ended before its greeting for
            another reason (a missing library, for example)
        TransientError: QEMU_CRASHED: the probe hit a process, memory or file-handle limit,
            the binary was being written, or a signal other than SIGSYS ended it (the next
            call probes again)
    """
    if (supported := await probe_qemu_sandbox_support()) is None:
        raise _qemu_probe_failure(
            "qemu_sandbox",
            "seccomp support",
            f"-machine none -sandbox {constants.QEMU_SANDBOX_POLICY} -qmp stdio",
            expected="a QMP greeting",
        )
    return supported


@async_cached_probe("kvm", per_qemu_binary=True)  # reads the accelerator probe
async def check_kvm_available() -> bool | None:
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
        None: the /dev/kvm check could not run or answer (a process, memory or file-handle
        limit, no answer in time): nothing cached, detect_accel_type raises (a kept False
        would start every later VM under TCG)

    Raises:
        TransientError: QEMU_CRASHED, or ConfigError HOST_DEPENDENCY: the accelerator probe
            gave no answer (_qemu_probe_failure): no answer is cached
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
        returncode, stdout, _ = await run_probe(
            (
                sys.executable,
                "-c",
                f"import fcntl; f=open('{kvm_path}','rb'); print(fcntl.ioctl(f.fileno(), {_KVM_GET_API_VERSION}))",
            ),
            stdin=None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if returncode != 0:
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

    except OSError as e:  # TimeoutError too; the check did not run or answer: no answer, never kept
        logger.warning("KVM check got no answer", extra={"error": str(e)})
        raise _Uncached(result=None, error=e) from None
    except ValueError as e:  # an answer that is no API version: no usable KVM
        logger.debug("KVM not available: failed to verify /dev/kvm", extra={"error": str(e)})
        return False

    # Layer 2: Verify QEMU binary has KVM support compiled in
    # Even if /dev/kvm works, QEMU may not have KVM support or may fail to initialize it
    # See: https://github.com/orgs/community/discussions/8305 (GitHub Actions KVM issues)
    qemu_accels = await probe_qemu_accelerators()
    if not qemu_accels:  # the probe failed (a QEMU lists tcg at least): no answer to keep
        raise _qemu_probe_failure("qemu_accels", "accelerators", "-accel help", expected="its accelerators")
    if "kvm" not in qemu_accels:
        logger.warning(
            "KVM not available: QEMU binary does not support KVM accelerator",
            extra={"available_accelerators": sorted(qemu_accels)},
        )
        return False

    logger.debug("KVM available and working (kernel + QEMU verified)")
    return True


@async_cached_probe("hvf", per_qemu_binary=True)  # reads the accelerator probe
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

    Raises:
        TransientError: QEMU_CRASHED, or ConfigError HOST_DEPENDENCY: the accelerator probe
            gave no answer (_qemu_probe_failure): no answer is cached
    """
    try:
        # sysctl kern.hv_support returns 1 if Hypervisor.framework is available
        returncode, stdout, _ = await run_probe(
            ("/usr/sbin/sysctl", "-n", "kern.hv_support"),
            stdin=None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        hvf_kernel_support = returncode == 0 and stdout.decode().strip() == "1"

        if not hvf_kernel_support:
            logger.debug("HVF not available: kern.hv_support is not enabled")
            return False

        logger.debug("HVF kernel support check passed")

    except OSError as e:  # TimeoutError too
        logger.debug("HVF not available: sysctl check failed", extra={"error": str(e)})
        return False

    # Layer 2: Verify QEMU binary has HVF support compiled in
    # Even if kern.hv_support is enabled, QEMU may not have HVF support
    qemu_accels = await probe_qemu_accelerators()
    if not qemu_accels:  # the probe failed (a QEMU lists tcg at least): no answer to keep
        raise _qemu_probe_failure("qemu_accels", "accelerators", "-accel help", expected="its accelerators")
    if "hvf" not in qemu_accels:
        logger.warning(
            "HVF not available: QEMU binary does not support HVF accelerator",
            extra={"available_accelerators": sorted(qemu_accels)},
        )
        return False

    logger.debug("HVF available and working (kernel + QEMU verified)")
    return True


@async_cached_probe("tsc_deadline", per_qemu_binary=False)
async def check_tsc_deadline() -> bool:
    """Check if TSC_DEADLINE CPU feature is available (cached).

    TSC_DEADLINE is required to disable PIT (i8254) and PIC (i8259) in microvm.
    Without TSC_DEADLINE, the APIC timer cannot use deadline mode, and the system
    needs the legacy PIT for timer interrupts.

    In nested virtualization (e.g., GitHub Actions runners), TSC_DEADLINE may not
    be exposed to the guest, causing boot hangs if PIT/PIC are disabled. Under HVF
    (macOS) the guest never has it: False there, with no probe.

    See: https://www.qemu.org/docs/master/system/i386/microvm.html

    Returns:
        True if TSC_DEADLINE is available, False otherwise
    """
    # TSC_DEADLINE is x86-only. macOS: HVF never passes the TSC deadline timer to an x86
    # guest (QEMU target/i386/hvf/x86_cpuid.c masks it out of CPUID leaf 1 ECX), whatever the
    # host CPU has; with enforce, asking for it would refuse every VM
    if detect_host_arch() != HostArch.X86_64 or detect_host_os() != HostOS.LINUX:
        return False
    try:
        # In a worker thread: the kernel builds this file for every CPU (large on
        # big hosts; some kernels sample each CPU's frequency while doing so)
        cpuinfo = await asyncio.to_thread(Path("/proc/cpuinfo").read_text)
    except OSError as e:
        logger.warning("Failed to read /proc/cpuinfo for TSC_DEADLINE check", extra={"error": str(e)})
        return False
    # The first "flags : fpu vme ... tsc_deadline_timer ..." line (every CPU lists the same)
    flags = next((line for line in cpuinfo.splitlines() if line.startswith("flags")), "")
    has_tsc = "tsc_deadline_timer" in flags.split()
    logger.debug(
        "TSC_DEADLINE available (can disable PIT/PIC)"
        if has_tsc
        else "TSC_DEADLINE not available (keeping PIT/PIC enabled)"
    )
    return has_tsc


@async_cached_probe("kvm_tsc_scaling", per_qemu_binary=False)
async def probe_kvm_tsc_scaling() -> bool:
    """KVM runs each guest's TSC at the rate the guest booted with, whatever this host's TSC
    rate is (KVM_CAP_TSC_CONTROL: Intel TSC scaling, AMD TSC ratio; cached): a state saved on
    a host with another rate loads here with its clock right (QEMU sets the saved rate).
    False without /dev/kvm.

    See: https://docs.kernel.org/virt/kvm/api.html (KVM_CAP_TSC_CONTROL, KVM_SET_TSC_KHZ)

    Raises:
        TransientError: LOCAL_DISK: a file-handle or memory limit (nothing kept)
    """

    def ask() -> bool:
        with Path("/dev/kvm").open("rb") as kvm:
            return fcntl.ioctl(kvm.fileno(), _KVM_CHECK_EXTENSION, _KVM_CAP_TSC_CONTROL) > 0

    try:
        return await asyncio.to_thread(ask)
    except OSError as e:
        if e.errno in (errno.ENOENT, errno.ENXIO, errno.ENODEV, errno.EACCES, errno.EPERM):
            return False  # no /dev/kvm, or no access: no KVM guest here to scale
        # A limit: TransientError, nothing kept; another error (EIO, EBUSY): False, kept and logged
        _no_host_source(e, what="KVM TSC scaling")
        return False


def _no_host_source(error: OSError, *, what: str) -> None:
    """A probe's failed read of the host's `what`: None, the probe's kept answer (no source
    here: a file it may not read, for example), logged once.

    Raises:
        TransientError: LOCAL_DISK: a limit (file handles, memory, processes) or no answer in
            time: nothing is kept, the next call reads again
    """
    if isinstance(error, TimeoutError) or error.errno in HOST_LIMIT_ERRNOS:
        raise TransientError(
            f"The {what} of this host could not be read now ({error.strerror or 'no answer in time'}).",
            code=ErrorCode.LOCAL_DISK,
            context={"errno": error.errno},
            hint=None,
            instance_id=None,
        ) from error
    logger.warning(f"The {what} of this host could not be read: none is used", extra={"error": str(error)})


_CNTFRQ_EL0_READ: Final = bytes.fromhex("00e03bd5c0035fd6")
"""arm64 code: mrs x0, cntfrq_el0; ret."""


def _read_cntfrq_el0() -> int | None:
    """CNTFRQ_EL0, the rate of the ARM generic timer as the firmware set it: user space reads
    it with one instruction (not trapped), run here from a page made executable. None: this
    host refuses executable memory (SELinux execmem, PaX MPROTECT), so no exact source.

    Raises:
        OSError: mprotect failed for another reason
    """
    with mmap.mmap(-1, mmap.PAGESIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE) as page:
        page.write(_CNTFRQ_EL0_READ)
        # The c_char view goes at once: no buffer export is left when the page closes
        address = ctypes.addressof(ctypes.c_char.from_buffer(page))
        mprotect = ctypes.CDLL(None, use_errno=True).mprotect
        mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        if mprotect(address, mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_EXEC) != 0:
            if (error := ctypes.get_errno()) in (errno.EACCES, errno.EPERM):
                return None
            raise OSError(error, "mprotect of the CNTFRQ_EL0 read")
        return ctypes.CFUNCTYPE(ctypes.c_uint64)(address)()


def _bogomips_hz(cpuinfo: str) -> int | None:
    """The x86 TSC rate in Hz from the first BogoMIPS line, when the CPU or the hypervisor
    states the rate (flag tsc_known_freq): the kernel then sets loops_per_jiffy from tsc_khz,
    and prints it as BogoMIPS with two decimals, tsc_khz / 500 (CONFIG_HZ 100, 250 or 1000):
    the rate in steps of 5 kHz. None without that flag: the kernel measured the rate at its
    boot, and it differs from boot to boot."""
    lines = cpuinfo.splitlines()
    flags = next((line for line in lines if line.startswith("flags")), "")
    bogomips = next((line for line in lines if line.startswith("bogomips")), None)
    if "tsc_known_freq" not in flags.split() or bogomips is None:
        return None
    match = re.fullmatch(r"bogomips\s*:\s*(\d+)\.(\d\d)", bogomips.strip())
    return None if match is None else (int(match[1]) * 100 + int(match[2])) * 5000


@async_cached_probe("host_counter_hz", per_qemu_binary=False)
async def probe_host_counter_hz() -> int | None:
    """Rate in Hz of the counter a guest's clock runs on here, from an exact source (cached);
    None: this host has none. A guest calibrates its clock against this rate once, at its
    boot: on a host with another rate its time runs fast or slow.

    - Linux x86: the TSC rate when the CPU or the hypervisor states it (_bogomips_hz).
    - Linux ARM64: CNTFRQ_EL0 (a KVM guest reads the same register).
    - macOS ARM64: sysctl hw.tbfrequency (24 MHz on Apple chips: the rate HVF guests see).
    - macOS x86: sysctl machdep.tsc.frequency.

    Raises:
        TransientError: LOCAL_DISK: the read hit a host limit or got no answer in time
            (_no_host_source: nothing kept)
    """
    arch, host_os = detect_host_arch(), detect_host_os()
    try:
        if host_os == HostOS.MACOS:
            name = "hw.tbfrequency" if arch == HostArch.AARCH64 else "machdep.tsc.frequency"
            returncode, stdout, _ = await run_probe(
                ("/usr/sbin/sysctl", "-n", name),
                stdin=None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            return int(stdout) if returncode == 0 and stdout.strip().isdigit() else None
        if arch == HostArch.AARCH64:
            if (hz := await asyncio.to_thread(_read_cntfrq_el0)) is None:
                logger.warning("No exact clock counter rate: this host refuses executable memory (CNTFRQ_EL0 read)")
            return hz
        # In a worker thread: the kernel builds this file for every CPU
        return _bogomips_hz(await asyncio.to_thread(Path("/proc/cpuinfo").read_text))
    except OSError as e:  # TimeoutError too
        return _no_host_source(e, what="clock counter rate")


_HVF_ID_REGISTERS: Final = (
    ("dfr0", 0),
    ("dfr1", 1),
    ("isar0", 2),
    ("isar1", 3),
    ("mmfr0", 4),
    ("mmfr1", 5),
    ("mmfr2", 6),
    ("pfr0", 7),
    ("pfr1", 8),
    ("smfr0", 12),
    ("zfr0", 13),
)
"""(name, hv_feature_reg_t) of the ID_AA64 registers QEMU's -cpu host takes under HVF
(target/arm/hvf/hvf.c); smfr0 and zfr0 only where macOS has them (15.2 and later)."""


def _hvf_cpu_id() -> str | None:
    """The HVF guest's CPU feature registers (hv_vcpu_config_get_feature_reg, no
    entitlement needed), as name=0x<16 hex digits> joined by commas; None when none reads
    (a Mac with no HVF, such as a macOS VM)."""
    hv = ctypes.CDLL("/System/Library/Frameworks/Hypervisor.framework/Hypervisor")
    hv.hv_vcpu_config_create.restype = ctypes.c_void_p
    hv.hv_vcpu_config_get_feature_reg.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
    hv.hv_vcpu_config_get_feature_reg.restype = ctypes.c_int32
    config = hv.hv_vcpu_config_create()  # one per process (the probe is cached): not released
    parts: list[str] = []
    for name, register in _HVF_ID_REGISTERS:
        value = ctypes.c_uint64()
        if hv.hv_vcpu_config_get_feature_reg(config, register, ctypes.byref(value)) == 0:  # HV_SUCCESS
            parts.append(f"{name}=0x{value.value:016x}")
    return ",".join(parts) or None


def _linux_arm_cpu_id(cpus: Path) -> str | None:
    """The distinct midr_el1 and revidr_el1 of the CPUs under `cpus` (sysfs), sorted, as
    midr_el1=0x...,revidr_el1=0x... joined by semicolons (one entry on a server whose cores
    are all alike); None without them. AIDR_EL1 is left out: Linux lists it from 6.16 only."""
    found: set[str] = set()
    for folder in cpus.glob("cpu[0-9]*/regs/identification"):
        midr, revidr = ((folder / name).read_text().strip() for name in ("midr_el1", "revidr_el1"))
        found.add(f"midr_el1={midr},revidr_el1={revidr}")
    return ";".join(sorted(found)) or None


@async_cached_probe("cpu_id", per_qemu_binary=False)
async def probe_host_cpu_id() -> str | None:
    """Identity of the host CPU as an ARM64 guest with -cpu host sees it (cached): the CPU
    part and revision on Linux (_linux_arm_cpu_id), the HVF feature registers on macOS
    (_hvf_cpu_id). A state saved on one CPU loads only on the same one: QEMU refuses
    another after the download, or HVF loads it into other features. None on x86 (the guest
    CPU is a fixed model) and when the registers cannot be read.

    Raises:
        TransientError: LOCAL_DISK: the read hit a host limit (_no_host_source: nothing kept)
    """
    if detect_host_arch() != HostArch.AARCH64:
        return None
    try:
        if detect_host_os() == HostOS.MACOS:
            return await asyncio.to_thread(_hvf_cpu_id)
        return await asyncio.to_thread(_linux_arm_cpu_id, Path("/sys/devices/system/cpu"))
    except OSError as e:  # TimeoutError too
        return _no_host_source(e, what="CPU identity")


_GUEST_CPU_QMP: Final = (
    b'{"execute": "qmp_capabilities"}\n'
    b'{"execute": "qom-get", "arguments": {"path": "/machine/unattached/device[0]", "property": "tsc-frequency"}}\n'
    b'{"execute": "quit"}\n'
)
"""The QMP commands of require_guest_cpu: the vCPU's TSC rate (QEMU reads it from KVM when it
creates the vCPU; 0 under TCG), then quit."""


async def require_guest_cpu(cpu: str, *, accel: str) -> int | None:
    """Start this QEMU once, paused, with the x86 guest CPU `cpu` (a -cpu value) under `accel`
    (a -accel value) on the microvm machine (QEMU 11 runs no x86 CPU on the 'none' machine),
    and read the vCPU's TSC rate: cached per QEMU binary, accelerator and CPU. VmManager.start
    asks it, so a host that cannot run the guest CPU (enforce: a flag it lacks) fails there,
    with QEMU's reasons, not at each VM start.

    Returns: the guest TSC rate in Hz under KVM (the rate KVM gives a new vCPU); None under
    TCG and HVF.

    Raises:
        ConfigError: HOST_DEPENDENCY: QEMU refused the CPU (its lines name the features this
            host lacks), or there is no QEMU binary or it cannot run here
        TransientError: QEMU_CRASHED: another QEMU error (its lines), a process, memory or
            file-handle limit, a signal, the binary being written, or no answer in time
            (nothing is kept: the next call probes again)
    """
    stamp = _qemu_binary_stamp()
    key = (stamp, accel, cpu)
    if key in probe_cache.guest_cpu:
        return probe_cache.guest_cpu[key]
    async with probe_cache.get_lock("guest_cpu"):
        if key in probe_cache.guest_cpu:
            return probe_cache.guest_cpu[key]
        binary = _probed_qemu_binary()
        args = (
            "-M", "microvm,acpi=off,x-option-roms=off,isa-serial=off", "-accel", accel, "-cpu", cpu,
            "-m", "16", "-nodefaults", "-no-user-config", "-display", "none", "-S", "-qmp", "stdio",
        )  # fmt: skip
        try:
            returncode, stdout, stderr = await run_probe(
                (binary, *args), stdin=_GUEST_CPU_QMP, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError as e:  # TimeoutError too
            probe_cache.failures["guest_cpu"] = e
            raise _qemu_probe_failure("guest_cpu", "guest CPU", f"-cpu {cpu}", expected="a QMP greeting") from None
        if returncode < 0:
            probe_cache.failures["guest_cpu"] = _exit_error(returncode)
            raise _qemu_probe_failure("guest_cpu", "guest CPU", f"-cpu {cpu}", expected="a QMP greeting")
        if returncode != 0:
            reasons = " ".join(line.strip() for line in stderr.decode(errors="replace").splitlines() if line.strip())
            context = {"binary": binary, "accel": accel, "cpu": cpu}
            if "doesn't support requested feature" in reasons:  # enforce: KVM's or TCG's refusal
                raise ConfigError(
                    f"This host cannot run the guest CPU ({binary} -accel {accel} -cpu {cpu}): {reasons}",
                    code=ErrorCode.HOST_DEPENDENCY,
                    context=context,
                    hint="x86 guests need an x86-64-v3 host CPU (Intel Haswell or AMD Zen 1, or newer) "
                    "that passes these flags to its VMs.",
                    instance_id=None,
                )
            # Another QEMU error (KVM_CREATE_VM out of memory, for example): it can pass, and
            # nothing is kept, so the next start probes again
            raise TransientError(
                f"QEMU did not start the guest CPU ({binary} -accel {accel} -cpu {cpu}, exit code "
                f"{returncode}): {reasons or 'no reason given'}",
                code=ErrorCode.QEMU_CRASHED,
                context=context,
                hint=None,
                instance_id=None,
            )
        replies = [json.loads(line) for line in stdout.decode(errors="replace").splitlines() if line.startswith("{")]
        rates = [reply["return"] for reply in replies if isinstance(reply.get("return"), int)]
        rate = rates[0] if rates and rates[0] > 0 and accel == "kvm" else None
        probe_cache.guest_cpu[key] = rate
        return rate


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
    None: no ACL beyond the mode bits (ENODATA), or no POSIX ACLs here: an OS with no xattr
    call (macOS: its ACLs are another model, read with acl_get_fd_np) or a filesystem that
    refuses the name (EOPNOTSUPP: no ACL support, mounted noacl), whatever the OS is called.
    Another error (access, a file gone) gives None too."""
    if _GETXATTR is None:
        return None
    try:
        return _GETXATTR(path, POSIX_ACL_ACCESS_XATTR)
    except OSError:
        return None


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


@async_cached_probe("io_uring", per_qemu_binary=False)
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
        returncode, _, _ = await run_probe(
            (sys.executable, "-c", probe_script),
            stdin=None,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        if returncode == 0:
            logger.info(
                "io_uring syscall available",
                extra={"kernel": platform.release()},
            )
            return True
        if returncode == 1:
            logger.info(
                "io_uring syscall not available (ENOSYS)",
                extra={"kernel": platform.release()},
            )
            return False
        if returncode == 2:  # noqa: PLR2004
            logger.warning(
                "io_uring blocked by seccomp/container policy",
                extra={"kernel": platform.release()},
            )
            return False

        logger.warning(
            "io_uring probe failed with unexpected result",
            extra={"exit_code": returncode},
        )
        return False

    except OSError as e:  # TimeoutError too
        logger.warning(
            "io_uring syscall probe failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return False


@async_cached_probe("unshare", per_qemu_binary=False)
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
    if detect_host_os() != HostOS.LINUX:
        return False

    try:
        # Test unshare with minimal namespaces (pid requires fork)
        returncode, _, stderr = await run_probe(
            (
                "/usr/bin/unshare",
                "--pid",
                "--fork",
                "--kill-child=SIGKILL",  # the VM command uses it: util-linux 2.32+
                "--",
                "/usr/bin/true",
            ),
            stdin=None,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        if returncode == 0:
            logger.info("unshare available (namespace isolation enabled)")
            return True

        stderr_text = stderr.decode().strip() if stderr else ""
        logger.warning(
            "unshare unavailable (namespace isolation disabled)",
            extra={"exit_code": returncode, "stderr": stderr_text[:200]},
        )
        return False
    except OSError as e:  # TimeoutError too
        logger.warning(
            "unshare probe failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return False


@async_cached_probe("setpriv", per_qemu_binary=False)
async def probe_setpriv_pdeathsig() -> bool:
    """Whether `setpriv --pdeathsig` works (util-linux 2.33+; cached). Children that end with
    this process (subprocess_utils.start_managed_process) then need no Python preexec_fn, so
    Popen can vfork. Else one warning: each such launch forks this process. False off Linux."""
    if detect_host_os() != HostOS.LINUX:
        return False
    try:
        # The launch form: setpriv, then /bin/sh (subprocess_utils._launch_command)
        returncode, _, stderr = await run_probe(
            ("setpriv", "--pdeathsig", "KILL", "--", "/bin/sh", "-c", ":"),
            stdin=None,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:  # TimeoutError too
        reason: dict[str, object] = {"error": str(e), "error_type": type(e).__name__}
    else:
        if returncode == 0:
            return True
        reason = {"exit_code": returncode, "stderr": stderr.decode(errors="replace").strip()[:200]}
    logger.warning(
        "setpriv --pdeathsig unavailable (util-linux 2.33+): each child that ends with this process "
        "(QEMU as this user, gvproxy, qemu-storage-daemon) starts through a fork of this process",
        extra=reason,
    )
    return False


def _kvm_check_failure() -> SandboxError:
    """The error of a /dev/kvm check that got no answer (probe_cache.failures["kvm"]): no VM
    starts on a guess. A process, memory or file-handle limit at its spawn can pass
    (TransientError QEMU_CRASHED); a check that did not answer in time too."""
    error = probe_cache.failures.get("kvm")
    context = {"check": "/dev/kvm ioctl"}
    if isinstance(error, OSError) and not isinstance(error, TimeoutError):
        return spawn_error(error, binary=sys.executable, code=ErrorCode.QEMU_CRASHED, context=context, hint=None)
    return TransientError(
        "The /dev/kvm check gave no answer in time.",
        code=ErrorCode.QEMU_CRASHED,
        context=context,
        hint=None,
        instance_id=None,
    )


async def detect_accel_type(*, force_emulation: bool) -> AccelType:
    """Detect which QEMU accelerator to use.

    This is the single source of truth for virtualization mode detection.
    Used for both cgroup memory sizing (TCG needs more) and QEMU command building.

    Args:
        force_emulation: Force TCG software emulation

    Returns:
        AccelType.KVM if Linux KVM available
        AccelType.HVF if macOS HVF available
        AccelType.TCG if software emulation needed (or force_emulation=True)

    Raises:
        TransientError: QEMU_CRASHED, or ConfigError HOST_DEPENDENCY: the host has the
            accelerator, but the QEMU accelerator probe gave no answer (check_kvm_available,
            check_hvf_available)
    """
    if force_emulation:
        return AccelType.TCG
    if (kvm := await check_kvm_available()) is None:
        raise _kvm_check_failure()
    if kvm:
        return AccelType.KVM
    if detect_host_os() == HostOS.MACOS and await check_hvf_available():
        return AccelType.HVF
    return AccelType.TCG

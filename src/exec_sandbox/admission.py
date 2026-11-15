"""Resource-aware admission controller for VM scheduling.

Four admission gates, gate 3 in two checks (all must pass):
1. Memory budget - host_total * (1 - reserve_ratio) * overcommit_ratio
2. CPU budget - (host_cpu_count - host_cpu_reserve) * cpu_overcommit_ratio
3a. Available-memory floor - reject when system available memory is insufficient
3b. Memory pressure - reject when system is thrashing
    - Linux 4.20+: PSI full avg10 >= 10% from /proc/pressure/memory
    - macOS: kern.memorystatus_vm_pressure_level at CRITICAL
4. Home disk space - free space of the home disk folder, minus the reserve of
   admitted home disks and of the request, stays >= disk_free_floor_mb. Home
   disks are sparse: the reserve is the part not written yet / disk_overcommit_ratio.

Gates 1-2 are budget-based (stable). Gates 3a/3b and 4 are dynamic (inline
probe, cached 100ms). A self-wake timer periodically notifies blocked
waiters so Gate 3 re-probes even without local release() events.
When gate 4 blocks, the request first evicts the oldest entries of the caches
on the same filesystem (add_cache) that no VM uses, then waits. Each pass of
evictions stops once the request fits, or once its entries cover what the
request lacked, and what the other requests that gate 4 blocks need. One request
checks and evicts at a time (_evict_lock). After a pass the free space is read
again once it shows the removals (_settle_removals: a btrfs commit; XFS frees
them later, so the space is read again while it grows; overlay: one syncfs).
Removed space still coming when the pass ends (ZFS shows it at its next
transaction sync) counts as room until it shows (_Pending), and no pass starts
meanwhile, so what the free space gains then is that pass's own. A pass that
gave space back and left the request short is followed by the next one at once.
A pass that gave almost nothing back ends the evictions of that request (a
copy-on-write clone or an open file keeps the blocks), and the cache entries no
longer count as room it can make; else they count at the part of their bytes
the passes gave back (btrfs compression). The gain is net of what this
controller's VMs wrote meanwhile where the free space counts written data at
once (ext4, XFS, ZFS, tmpfs, APFS); btrfs counts it only at writeback. When
neither evictions nor the end of this controller's VMs and sessions can make
room, it fails at once (TransientError LOCAL_DISK): before any eviction, or
after the first pass where the removed entries gave back only part of their
size.
Waiters block on an asyncio.Condition. A release wakes one waiter when all
waiters ask for the same size, else all of them (mixed VmSpec sizes). A
timeout raises TransientError (CAPACITY) that names the gate that blocked.

Capacity detection priority chain:
  manual override > cgroup limit > psutil total

Graceful degradation: a dimension (memory or CPUs) that neither the cgroup nor the
psutil probe can size gets an infinite budget (no admission control for it).
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import fcntl
import functools
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeVar, cast
from uuid import uuid4

import psutil

from exec_sandbox._logging import get_logger
from exec_sandbox.cgroup import (
    detect_cgroup_cpu_limit,
    detect_cgroup_memory_limit_mb,
    read_container_available_memory_mb,
)
from exec_sandbox.constants import (
    DEFAULT_TCG_TB_CACHE_SIZE_MB,
    DEFAULT_VM_CPU_OVERHEAD_CORES,
    DEFAULT_VM_MEMORY_OVERHEAD_MB,
    GATE3_SELF_WAKE_INTERVAL_SECONDS,
)
from exec_sandbox.exceptions import ErrorCode, InvalidInputError, RouteError, RouteKey, TransientError
from exec_sandbox.files.disk import OVERLAYFS_SUPER_MAGIC, allocated_bytes, overlay_upper, statfs_type
from exec_sandbox.platform_utils import HostOS, detect_host_os

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = get_logger(__name__)

CapacitySource = Literal["manual", "cgroup", "psutil", "none", "unknown"]


@dataclass(frozen=True)
class ResourceReservation:
    """Tracks resources reserved by a single VM admission: the effective limits
    (guest + overhead) that budget tracking counts, and the home disk size.
    """

    vm_id: str
    # Effective limits (guest + overhead) — used by cgroup/ulimit enforcers and budget tracking
    memory_mb: float
    cpu_cores: float
    # Home disk size (sparse file: the guest can fill it up to this)
    disk_mb: float
    # Unique key in _reservations dict — prevents collision when
    # multiple VMs share the same vm_id string (e.g. concurrent requests
    # with the same tenant_id + task_id).
    reservation_id: str = field(default_factory=lambda: str(uuid4()))


class EvictionPass(Protocol):
    """One pass of evictions over a cache (BaseCacheManager.eviction_order)."""

    async def evict_next(self) -> int | None:
        """Remove the next least recently used entry that no VM uses and no save holds: the
        bytes its files took; None when no such entry is left."""
        ...


class EvictableCache(Protocol):
    """A cache on disk that gives space back when asked: the L1 and L2 caches, and the guest
    images that loads fetched from the instance store (AssetCache)."""

    @property
    def cache_dir(self) -> Path: ...

    def eviction_order(self) -> EvictionPass:
        """A pass of evictions, oldest entry first; the cache is listed once per pass."""
        ...

    async def reclaimable_bytes(self, *, held: frozenset[Path]) -> int:
        """Bytes of the entries that evictions can give back once this process's VMs are
        gone: no VM of another process uses them, and they are not in `held` (the entries
        the waiting request keeps leased)."""
        ...


class _Gate(StrEnum):
    """An admission gate, by the name errors give it."""

    MEMORY_BUDGET = "memory budget"
    CPU_BUDGET = "CPU budget"
    AVAILABLE_MEMORY = "available memory"
    MEMORY_PRESSURE = "memory pressure"
    HOME_DISK_SPACE = "home disk space"


_MIB: Final[int] = 1024 * 1024


@dataclass(frozen=True)
class _DiskSpace:
    """Gate 4 figures for one request, in MB."""

    free_mb: float
    """Free space of the filesystem of disk_dir."""
    written_mb: float
    """Blocks this controller's home disks use (not those shared with an L1 template)."""
    admitted_mb: float
    """Total size of the admitted home disks."""
    reserved_mb: float
    """Reserve of the admitted disks: their part not written yet / the disk overcommit ratio."""
    requested_mb: float
    """Size of the requested home disk."""
    needed_mb: float
    """Reserve of the requested disk (nothing written yet): requested_mb / the ratio."""
    floor_mb: int

    @property
    def shortfall_mb(self) -> float:
        """Space the request lacks; 0 or less when it fits."""
        return self.floor_mb + self.reserved_mb + self.needed_mb - self.free_mb

    def fits_after(self, need_mb: float, *, later_mb: float) -> bool:
        """A request whose reserve is `need_mb` (this one: needed_mb; a waiter of _disk_needs)
        fits once every VM of this controller is gone (their reserve and the blocks of their
        disks come back) and `later_mb` more comes back (_later_mb): else only other processes
        can make room."""
        return self.floor_mb + need_mb - self.free_mb - self.written_mb - later_mb <= 0

    def gain_since(self, start: _DiskSpace, *, net_of_writes: bool) -> float:
        """Free space that came back since `start`. With `net_of_writes`, net of the blocks this
        controller's home disks wrote meanwhile (a guest that writes hides the space a removal
        gave back): only where the free space drops as data is written (_Settle.counts_writes);
        btrfs counts a write in st_blocks at once but in its free space only at writeback, so
        the sum would grow at the speed of any writing VM there (measured: +143 MB in 3 s at
        50 MB/s, from nothing)."""
        gain = self.free_mb - start.free_mb
        return gain + self.written_mb - start.written_mb if net_of_writes else gain

    def describe(self) -> str:
        return (
            f"Home disk space: {round(self.free_mb)}MB free, floor {self.floor_mb}MB, "
            f"{round(self.admitted_mb)}MB of disks admitted ({round(self.reserved_mb)}MB reserved), "
            f"{round(self.requested_mb)}MB requested ({round(self.needed_mb)}MB reserved)."
        )

    def context(self) -> dict[str, int]:
        return {
            "disk_free_mb": round(self.free_mb),
            "disk_free_floor_mb": self.floor_mb,
            "disk_admitted_mb": round(self.admitted_mb),
            "disk_reserved_mb": round(self.reserved_mb),
            "disk_requested_mb": round(self.requested_mb),
            "disk_needed_mb": round(self.needed_mb),
        }


@dataclass
class _Evictions:
    """The gate 4 evictions of one waiting request, across its wake-ups."""

    stopped: bool = False
    """A pass gave almost no space back (_GAIN_MIN_RATIO) once the free space showed its
    removals: a copy-on-write clone or an open file keeps the blocks. This request evicts no
    more, and counts on no cache entry for room."""
    evicted_mb: float = 0.0
    """Bytes (st_blocks) that the passes of this request removed, once the free space showed them."""
    gained_mb: float = 0.0
    """What the free space showed of them, net of what this controller's VMs wrote meanwhile."""
    shared_stop: bool = False
    """`stopped` came from the verdict on a pass of another request that gate 4 blocked at the
    same time (the same cache gave nothing back: a ZFS snapshot keeps the blocks); a pass that
    gives space back, a release or the self-wake lifts it (_gave_space, _ease_shared_stop)."""
    checked: bool = False
    """Its room was checked once (_make_room ran _check_room): a later wake-up may skip the
    listing where nothing can be evicted at this room version (_nothing_to_evict_at)."""
    sync_first: bool = False
    """A pass showed almost nothing after its syncfs (_Settle.sync_once), and no verdict came of
    it: that sync also wrote the dirty data of other files, and on an overlay over btrfs their
    new blocks and the removals show in the same commit (300 MB of dirty data made a 200 MB
    gain read as -100 MB). The next passes sync before they evict, so the sync after each pass
    writes only the data written meanwhile, and its gain is measured alone."""

    @property
    def cache_share(self) -> float:
        """The part of the bytes of a cache entry (st_blocks) that its eviction gives back, as the
        passes of this request measured it: btrfs compression gives back as little as 1/32 (it
        counts st_blocks before compression). 1 before the first pass, 0 once stopped."""
        if self.stopped:
            return 0.0
        if not self.evicted_mb:
            return 1.0
        return min(1.0, max(0.0, self.gained_mb / self.evicted_mb))


@dataclass
class _Pending:
    """The space the last pass removed, when the free space did not show it all as the pass
    ended: XFS still frees a fragmented file (for seconds), ZFS shows it at its next transaction
    sync, a syncfs still runs. It counts as room until it shows, until it has grown (by
    _GAIN_MIN_RATIO of the pass at least) and then not for _BACKGROUND.quiet (it showed as far
    as it will: 90 % on ZFS while a VM writes), until `until`, or until that sync ends. No pass
    starts meanwhile, so what the free space gains is this pass's own."""

    evictions: _Evictions
    """The request of the pass: the verdict on the pass goes to it (_settle_pending)."""
    start: _DiskSpace
    """The space before the pass."""
    evicted_mb: float
    net_of_writes: bool
    """_Settle.counts_writes of the filesystem."""
    until: float
    """Event loop time after which the space that did not show counts no more."""
    sync: asyncio.Task[None] | None
    """The syncfs the pass did not wait for to the end, or None. Once it ends, no verdict: its
    writes of other files' dirty data can hide the gain (_Evictions.sync_first)."""
    shown_mb: float
    """The largest gain seen since the start of the pass."""
    grew_at: float | None
    """Event loop time when that gain grew last after the pass, or None: not yet."""


@dataclass(frozen=True)
class _PassEnd:
    """The space once the free space shows what a pass evicted (_space_after_pass)."""

    space: _DiskSpace | None
    more_s: float
    """How long the rest can still take to show (_Pending), or 0: it showed as far as it will."""
    growing: bool
    """It still grew when the reads stopped (XFS), so it has shown part already."""
    synced: bool
    """A syncfs ran for it and ended (_Settle.sync_once)."""
    net_of_writes: bool
    """_Settle.counts_writes of the filesystem."""
    gain_mb: float
    """The largest gain since the start of the pass that its reads saw: another process that
    writes meanwhile can hide part of it in the last read, which `space` is."""


@dataclass(frozen=True)
class _Settle:
    """How long the free space can take to show removals once _settle_removals returns."""

    quiet: float
    """The space shows no more once it has not grown for this long (0: read it once)."""
    limit: float
    """At most this long; space still growing then is still coming (_Pending)."""
    sync_once: bool
    """A pass that showed almost nothing by then makes the filesystem sync once (_syncfs), then
    reads again: an overlay frees removed files when its upper filesystem commits, and its
    statfs type does not say which filesystem that is (a btrfs commits every 30 s)."""
    later: float
    """The space shows in one step this long after the pass at most (ZFS: at the next
    transaction sync), so the pass reads it once and counts the rest as still coming for that
    long (_Pending), without the lock of the passes; 0: no such step."""
    counts_writes: bool
    """The free space drops as data is written (ext4, XFS, ZFS, tmpfs, APFS): the gain of a pass
    is net of what this controller's VMs wrote meanwhile (_DiskSpace.gain_since)."""


_GAIN_MIN_RATIO: Final[float] = 1 / 64
"""A pass of evictions gives space back when at least this part of the bytes it removed
(st_blocks) shows in the free space. The removal of a reflink clone frees only metadata (0.6 %
measured on XFS). btrfs compression gives back at least 1/32 (st_blocks counts the bytes before
compression, and a 128 KiB extent takes at least one 4 KiB block; an L1 RAM file gave 8 %)."""

_GAIN_FIRST_POLL_SECONDS: Final[float] = 0.001
"""The first wait before the free space is read again; each next one is twice as long, up to a
quarter of the quiet time of the filesystem and _GAIN_MAX_POLL_SECONDS."""
_GAIN_MAX_POLL_SECONDS: Final[float] = 0.05

_AT_REMOVAL: Final = _Settle(quiet=0.0, limit=0.0, sync_once=False, later=0.0, counts_writes=True)
"""ext4, tmpfs, macOS (APFS): a removal shows at once."""
_AFTER_COMMIT: Final = _Settle(quiet=0.01, limit=0.05, sync_once=False, later=0.0, counts_writes=False)
"""btrfs after its commit, or an overlay after syncfs: the free space can change once more 1-2
ms after the commit (an emptied data block group). A write shows in its free space only at
writeback."""
_BACKGROUND: Final = _Settle(quiet=0.15, limit=1.0, sync_once=False, later=0.0, counts_writes=True)
"""XFS frees a removed file in the background (inode inactivation): its free space grows from
1 ms after the removal (77 ms under 2x CPU overcommit), then in steps at most 35 ms apart; an
entry of a few extents (the real L1 and L2 files) frees in 3 ms, a 455 MB file of 37k extents
in 0.75 s at 0.5 % free (EC2, Linux 7.0)."""
_OTHER: Final = _Settle(quiet=0.15, limit=1.0, sync_once=True, later=0.0, counts_writes=False)
"""overlay and the other types: read again as on XFS, then syncfs once. The free space alone
judges the gain: over btrfs, a write shows in it only at writeback."""
_SYNCFS_WAIT_SECONDS: Final[float] = 1.0
"""At most this long a pass waits for syncfs, which writes every dirty page of the filesystem
first: 1 ms on an idle overlay, 108 ms with 300 MB of dirty data on a RAM disk, 8-16 s with 1-2
GB on gp3. A sync not done by then leaves the removed space still coming (_Pending)."""
_SHARED_NO_GAIN_PASSES: Final[int] = 2
"""Passes in a row (of any requests) that gave nothing back before every request that waits
for disk space stops evicting: one pass can give nothing for a cause of its own entries (the
oldest is open, cloned; another process wrote meanwhile) while the next entries give space; two
point at the filesystem (a ZFS snapshot keeps every block). At least 2: a lift of the shared stop
sets the count to this minus 1, which _others_mb reads as a pass of nothing."""
_EASE_MAX_SECONDS: Final[float] = 640.0
"""The longest gap between two lifts of the shared stop (_ease_shared_stop): 10 s, then twice as
long after each lift, while no pass gives space back. A ZFS or btrfs snapshot keeps every block,
so each pass after a lift removes good entries for nothing (a lift every 10 s: 360 passes an hour,
1 GB each; at most 6 passes an hour with this cap)."""
_UNSEEN_SECONDS: Final[float] = 10.0
"""Removed space still coming counts as room this long (one self-wake): XFS freed 911 MB of
fragmented entries in 3 s."""
_ZFS_TXG_SYNCS: Final[int] = 2
"""ZFS: a removal counts as still coming for this many transaction intervals and a second: it
shows at the end of the next sync, which can end late when it writes much dirty data."""

_XFS_SUPER_MAGIC: Final = 0x58465342
_ZFS_SUPER_MAGIC: Final = 0x2FC12FC1
_ZFS_TXG_TIMEOUT: Final = Path("/sys/module/zfs/parameters/zfs_txg_timeout")
"""Seconds between two ZFS transaction syncs (default 5): a removal shows at the next one, and
syncfs does not start one (it commits the intent log only: zfs_sync)."""
_BTRFS_SUPER_MAGIC: Final = 0x9123683E
_BTRFS_IOC_START_SYNC: Final = 0x80089418
"""_IOR(0x94, 24, __u64): start the commit of the running transaction; its id comes out."""
_BTRFS_IOC_WAIT_SYNC: Final = 0x40089416
"""_IOW(0x94, 22, __u64): wait for the commit of the transaction with that id."""
_FREED_AT_REMOVAL: Final = frozenset({0xEF53, 0x01021994})
"""statfs f_type of the Linux filesystems whose free space shows a removal at once
(linux/magic.h EXT4_SUPER_MAGIC, also ext2 and ext3; TMPFS_MAGIC)."""


def _zfs_loaded() -> bool:
    """Whether the ZFS module is loaded on this host (its zfs_txg_timeout parameter is there)."""
    return os.path.exists(_ZFS_TXG_TIMEOUT)  # noqa: PTH110 - never raises (no access reads as no module)


def _zfs_settle() -> _Settle:
    """ZFS: a removal shows in one step at its next transaction sync (zfs_txg_timeout)."""
    try:
        txg_seconds = int(_ZFS_TXG_TIMEOUT.read_text())
    except (OSError, ValueError):
        txg_seconds = 5
    return _Settle(quiet=0.0, limit=0.0, sync_once=False, later=_ZFS_TXG_SYNCS * txg_seconds + 1, counts_writes=True)


def _settle_removals(path: Path) -> _Settle:
    """Make the blocks of the files removed on the filesystem of `path` show in its free space
    where a call can, and tell how long they can still take. ext4, tmpfs and macOS (APFS) free
    them at the removal. btrfs frees them at its next transaction commit (up to 30 s later):
    this starts one and waits for it (BTRFS_IOC_START_SYNC, WAIT_SYNC: 1-4 ms with 2 GB of
    dirty data; it writes no file data, unlike syncfs: 16 s on gp3). XFS frees them in the
    background, and no call waits for that without CAP_SYS_ADMIN (XFS_IOC_FREE_EOFBLOCKS:
    EPERM). ZFS shows them at its next transaction sync. An overlay frees them as its upper
    filesystem does: its type is read from the upper folder where this process sees it
    (overlay_upper), else _OTHER. Blocking: a worker thread (a btrfs commit)."""
    if detect_host_os() is not HostOS.LINUX:
        return _AT_REMOVAL
    fs_type = statfs_type(path)
    if fs_type == OVERLAYFS_SUPER_MAGIC:
        upper = overlay_upper(path)
        if upper is None or (upper_type := statfs_type(upper)) in (None, OVERLAYFS_SUPER_MAGIC):
            return _OTHER
        path, fs_type = upper, upper_type  # a btrfs commit goes through the upper folder
    if fs_type in _FREED_AT_REMOVAL:
        return _AT_REMOVAL
    if fs_type == _XFS_SUPER_MAGIC:
        return _BACKGROUND
    if fs_type == _ZFS_SUPER_MAGIC:
        return _zfs_settle()
    if fs_type != _BTRFS_SUPER_MAGIC:
        return _OTHER  # the types not named here
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return _OTHER
    try:
        transaction = bytearray(8)
        fcntl.ioctl(fd, _BTRFS_IOC_START_SYNC, transaction)
        fcntl.ioctl(fd, _BTRFS_IOC_WAIT_SYNC, transaction)
    except OSError:
        return _OTHER
    finally:
        os.close(fd)
    return _AFTER_COMMIT


_LIBC: Final = ctypes.CDLL(None, use_errno=True)
"""The C library of this process: syncfs (Linux) and sysctlbyname (macOS)."""


def _syncfs(path: Path) -> None:
    """syncfs(2) on the filesystem of `path`: it writes every dirty page of that filesystem
    (of the upper filesystem for an overlay), then commits it. Best effort. Blocking: a worker
    thread."""
    if detect_host_os() is not HostOS.LINUX:
        return
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        _LIBC.syncfs(fd)
    finally:
        os.close(fd)


# Sentinel for "no limit" of a dimension that no probe sizes
_UNLIMITED: Final[float] = float("inf")

# Probe cache TTL: collapses repeated probes into 1 when
# self-wake timer or stop() wakes multiple waiters simultaneously
_PROBE_CACHE_TTL_SECONDS: Final[float] = 0.1

# Linux PSI threshold: percentage of last 10s ALL tasks were stalled on memory.
# 10% means 1 second out of 10 every task was blocked on memory I/O — serious thrashing.
_PSI_FULL_THRESHOLD_PCT: Final[float] = 10.0

# macOS memory pressure level CRITICAL (dispatch/source.h; NORMAL 1, WARN 2), read from the
# kern.memorystatus_vm_pressure_level sysctl
_MACOS_PRESSURE_CRITICAL: Final[int] = 4


def _read_macos_memory_pressure_level() -> int | None:
    """Read macOS kernel memory pressure level via sysctlbyname.

    Returns:
        1 (NORMAL), 2 (WARN), 4 (CRITICAL), or None on failure.
        Cost: ~3-5µs per call (no subprocess, direct syscall).
    """
    val = ctypes.c_int32(0)
    sz = ctypes.c_size_t(ctypes.sizeof(val))
    ret = _LIBC.sysctlbyname(
        b"kern.memorystatus_vm_pressure_level",
        ctypes.byref(val),
        ctypes.byref(sz),
        None,
        ctypes.c_size_t(0),
    )
    if ret != 0:
        return None
    return val.value


_R = TypeVar("_R")


def _ttl_cache(ttl_seconds: float) -> Callable[[Callable[[Any], _R]], Callable[[Any], _R]]:
    """Per-instance TTL cache for zero-arg methods.

    Stores ``(monotonic_timestamp, value)`` on the instance.
    """

    def decorator(func: Callable[[Any], _R]) -> Callable[[Any], _R]:
        attr = f"_ttl_{func.__name__}"

        @functools.wraps(func)
        def wrapper(self: Any) -> _R:
            now = time.monotonic()
            cached: tuple[float, _R] | None = getattr(self, attr, None)
            if cached is not None:
                ts, val = cached
                if now - ts < ttl_seconds:
                    return val
            result = func(self)
            setattr(self, attr, (now, result))
            return result

        return cast("Callable[[Any], _R]", wrapper)

    return decorator


def _forget_cached(owner: object, method: Callable[[Any], object]) -> None:
    """Drop the value a _ttl_cache method keeps on `owner`: its next call probes again."""
    vars(owner).pop(f"_ttl_{method.__name__}", None)


def _vm_totals(memory_mb: float, cpu_cores: float, *, use_tcg: bool) -> tuple[float, float]:
    """Memory (MB) and CPU (cores) a VM reserves: the guest's plus its overheads (single
    source of truth; TCG adds the translation block cache)."""
    total_memory_mb = memory_mb + DEFAULT_VM_MEMORY_OVERHEAD_MB
    if use_tcg:
        total_memory_mb += DEFAULT_TCG_TB_CACHE_SIZE_MB
    return total_memory_mb, cpu_cores + DEFAULT_VM_CPU_OVERHEAD_CORES


def budget_need(memory_mb: float, cpu_cores: float, *, use_tcg: bool) -> dict[RouteKey, str]:
    """The budget facts a VM needs, its overheads counted, in the route value forms rounded
    up (memory_mb whole MB, cpu two decimals): a worker whose budget facts (budget_facts,
    rounded down) are not below them admits the VM."""
    total_memory_mb, total_cpu = _vm_totals(memory_mb, cpu_cores, use_tcg=use_tcg)
    return {
        RouteKey.MEMORY_MB: str(math.ceil(round(total_memory_mb, 6))),
        RouteKey.CPU: f"{math.ceil(round(total_cpu * 100, 6)) / 100:.2f}",
    }


class ResourceAdmissionController:
    """Resource-aware admission controller for VM lifecycle: the gates of the module docstring
    (gate 3a: system_available - requested >= floor). With floor=0 (default), 3a rejects only
    when the system cannot fit the requested VM: backpressure across processes.
    """

    def __init__(
        self,
        *,
        memory_overcommit_ratio: float,
        cpu_overcommit_ratio: float,
        disk_overcommit_ratio: float,
        host_memory_reserve_ratio: float,
        host_memory_mb: float | None,
        host_cpu_count: float | None,
        available_memory_floor_mb: int,
        host_cpu_reserve_cores: float,
        disk_dir: Path,
        disk_free_floor_mb: int,
    ) -> None:
        self._disk_dir = disk_dir
        self._disk_free_floor_mb = disk_free_floor_mb
        self._memory_overcommit_ratio = memory_overcommit_ratio
        self._cpu_overcommit_ratio = cpu_overcommit_ratio
        self._disk_overcommit_ratio = disk_overcommit_ratio
        self._host_memory_reserve_ratio = host_memory_reserve_ratio
        self._available_memory_floor_mb = available_memory_floor_mb
        self._host_cpu_reserve_cores = host_cpu_reserve_cores

        # The host_memory_mb and host_cpu_count settings: each one given stands, start() probes
        # only the other; both given, the budgets are set here
        self._given_memory_mb = host_memory_mb
        self._given_cpu_count = host_cpu_count
        self._host_memory_mb: float = host_memory_mb if host_memory_mb is not None else 0.0
        self._host_cpu_count: float = host_cpu_count if host_cpu_count is not None else 0.0

        # Budget limits (computed in start())
        self._memory_budget_mb: float = _UNLIMITED
        self._cpu_budget: float = _UNLIMITED

        # Where the memory size came from (the CPU count has its own source; logs only)
        self._capacity_source: CapacitySource = "unknown"

        # Current allocations (protected by _condition's lock)
        self._allocated_memory_mb: float = 0.0
        self._allocated_cpu: float = 0.0
        self._allocated_disk_mb: float = 0.0
        # Home disk name -> bytes it shares with its L1 template at clone time, and the
        # home disks the last probe saw (a note of a disk not made yet stays)
        self._shared_disk_bytes: dict[str, int] = {}
        self._disks_seen: set[str] = set()
        # Caches on the filesystem of disk_dir that gate 4 evicts from (add_cache);
        # one request checks its room and evicts at a time
        self._caches: list[EvictableCache] = []
        self._evict_lock = asyncio.Lock()
        self._pending: _Pending | None = None
        self._no_gain_passes = 0  # passes in a row that gave nothing back (_SHARED_NO_GAIN_PASSES)
        # The lifts of the shared stop (_ease_shared_stop): the event loop time before which none
        # comes, and the gap after the last one (0: none since a pass gave space back)
        self._ease_at = 0.0
        self._ease_gap = 0.0
        # The room version at which a pass found no cache entry it could evict: until the room
        # changes, a waiter that checked its room once does not list the caches again
        self._nothing_to_evict_at = -1
        # Incremented when the room for gate 4 can have changed (a release, a pass that gave
        # space, the end of a pending pass or of a syncfs, the self-wake): a request that gate 4
        # blocks makes room again (_make_room: a listing of each cache) only then, not at each
        # wake-up
        self._room_version = 0
        self._stop_event = asyncio.Event()
        # reservation id -> the reserve (MB) of each request that gate 4 blocks: a pass evicts
        # for all of them at once (on ZFS, one transaction sync then shows the space of all)
        self._disk_needs: dict[str, tuple[float, _Evictions]] = {}
        self._syncfs_job: asyncio.Task[None] | None = None
        self._recheck_task: asyncio.Task[None] | None = None
        self._allocated_vm_slots: int = 0
        self._reservations: dict[str, ResourceReservation] = {}
        # (memory, cpu, disk) of each blocked acquire -> how many wait with it
        self._waiting: Counter[tuple[float, float, float]] = Counter()

        # Condition for blocking acquire / signaling release
        self._condition = asyncio.Condition()
        self._started = False
        self._start_lock = asyncio.Lock()

        # Shutdown flag: set by stop() to unblock acquire() waiters immediately

        # Self-wake timer: periodically notifies blocked waiters so Gate 3
        # re-probes system memory (closes wakeup gap for external recovery)
        self._self_wake_task: asyncio.Task[None] | None = None

        # If host resources were provided at construction, compute budgets now
        if host_memory_mb is not None and host_cpu_count is not None:
            self._capacity_source = "manual"
            self._compute_budgets()
            self._started = True
            logger.warning(
                "Host resources manually overridden, admission budgets may not reflect actual capacity",
                extra={
                    "host_memory_mb": host_memory_mb,
                    "host_cpu_count": host_cpu_count,
                    "memory_budget_mb": round(self._memory_budget_mb),
                    "cpu_budget": round(self._cpu_budget, 1),
                },
            )

    async def start(self) -> None:  # noqa: PLR0912
        """Probe host resources and compute budgets.

        Detection priority: given override > cgroup limit > psutil total, for memory and CPUs
        each. Both overrides given: the budgets are already set, and this only starts the timer.

        Safe to call multiple times (idempotent after first probe).
        If all probes fail, degrades gracefully to unlimited budgets.
        Serialized via asyncio.Lock to prevent concurrent probe races.
        """
        async with self._start_lock:
            # Start self-wake timer if not already running (idempotent); none after stop(),
            # which would not end it
            if self._self_wake_task is None and not self._stop_event.is_set():
                self._self_wake_task = asyncio.create_task(self._self_wake_loop())
            if self._started:
                return

            # --- L1: a given override, else the cgroup limit ---
            known_mem = self._given_memory_mb
            if known_mem is not None:
                self._capacity_source = "manual"
            elif (known_mem := detect_cgroup_memory_limit_mb()) is not None:
                self._host_memory_mb = known_mem
                self._capacity_source = "cgroup"
                logger.info(
                    "Using cgroup memory limit for admission budget",
                    extra={"cgroup_memory_mb": round(known_mem)},
                )

            known_cpu = self._given_cpu_count
            if known_cpu is None and (known_cpu := detect_cgroup_cpu_limit()) is not None:
                self._host_cpu_count = known_cpu
                logger.info(
                    "Using cgroup CPU limit for admission budget",
                    extra={"cgroup_cpus": round(known_cpu, 2)},
                )

            # --- Fallback to psutil for any missing dimension, each probe on its own: one that
            # fails leaves the other's dimension sized ---
            loop = asyncio.get_running_loop()
            if known_mem is None:
                try:
                    vmem = await loop.run_in_executor(None, psutil.virtual_memory)
                    known_mem = self._host_memory_mb = vmem.total / (1024 * 1024)
                    self._capacity_source = "psutil"
                except (OSError, AttributeError):
                    logger.debug("The psutil memory probe failed", exc_info=True)
            if known_cpu is None:
                try:
                    cpu_count = await loop.run_in_executor(None, psutil.cpu_count)
                except (OSError, AttributeError):
                    logger.debug("The psutil CPU probe failed", exc_info=True)
                else:
                    if cpu_count:  # None: psutil cannot count them (unsized, not 1 CPU)
                        known_cpu = self._host_cpu_count = float(cpu_count)

            self._compute_budgets()
            # A dimension that no probe sized has no limit, as when every probe fails: never a
            # budget computed from a host size of 0
            unsized = [name for name, size in (("memory", known_mem), ("cpu", known_cpu)) if size is None]
            if known_mem is None:
                self._memory_budget_mb = _UNLIMITED
                self._capacity_source = "none"
            if known_cpu is None:
                self._cpu_budget = _UNLIMITED
            if unsized:
                logger.warning(
                    "Resource probes failed: no admission limit for these resources",
                    extra={"unsized": unsized},
                )
            logger.info(
                "Host resources detected",
                extra={
                    "capacity_source": self._capacity_source,
                    "host_memory_mb": round(self._host_memory_mb),
                    "host_cpu_count": self._host_cpu_count,
                    "host_memory_reserve_ratio": self._host_memory_reserve_ratio,
                    "host_memory_reserve_mb": round(self._host_memory_mb * self._host_memory_reserve_ratio),
                    "memory_budget_mb": self._memory_budget_mb,
                    "cpu_budget": self._cpu_budget,
                    "memory_overcommit_ratio": self._memory_overcommit_ratio,
                    "cpu_overcommit_ratio": self._cpu_overcommit_ratio,
                    "disk_overcommit_ratio": self._disk_overcommit_ratio,
                },
            )

            self._started = True

    async def stop(self) -> None:
        """Stop the admission controller and unblock all waiting acquire() callers.

        Sets a stopped flag and notifies all waiters so they raise
        InvalidInputError (SCHEDULER_STATE) immediately instead of blocking until timeout.
        """
        self._stop_event.set()  # stopped; a pass waiting for a syncfs ends at once

        # Wake all blocked waiters so they see the stop and exit
        async with self._condition:
            self._condition.notify_all()

        timers = [task for task in (self._self_wake_task, self._recheck_task) if task is not None]
        for task in timers:
            task.cancel()
        # gather, not suppress(CancelledError): a cancel of this caller comes out
        await asyncio.gather(*timers, return_exceptions=True)
        self._self_wake_task = self._recheck_task = None

    @_ttl_cache(_PROBE_CACHE_TTL_SECONDS)
    def _probe_available_memory(self) -> float | None:
        """Probe system available memory.

        Probe chain: cgroup → psutil → None (graceful degradation).
        """
        avail = read_container_available_memory_mb()
        if avail is None:
            try:
                vmem = psutil.virtual_memory()
                avail = vmem.available / (1024 * 1024)
            except (OSError, AttributeError, RuntimeError, psutil.Error):
                avail = None
        return avail

    @_ttl_cache(_PROBE_CACHE_TTL_SECONDS)
    def _probe_memory_pressure(self) -> bool:
        """Check if system memory pressure warrants rejecting admission.

        Each platform has its own decision logic — no shared threshold:

        - **Linux** (4.20+, CONFIG_PSI=y): reads ``/proc/pressure/memory``
          ``full avg10`` and rejects when >= 10% (serious thrashing).
        - **macOS**: reads ``kern.memorystatus_vm_pressure_level`` sysctl
          and rejects at CRITICAL (4). WARN (2) is common during normal
          operation and does not trigger rejection.

        Returns:
            True to reject (system under pressure), False to admit, also when the platform
            gives no reading (gate skipped). Cost: ~3-5µs per call (procfs read or sysctl,
            cached 100ms).

        Note:
            We opened https://github.com/giampaolo/psutil/issues/2725 to add
            cross-platform memory-pressure metrics to psutil. Once merged,
            replace ``_probe_psi_linux`` / ``_probe_pressure_macos`` /
            ``_read_macos_memory_pressure_level`` with a single psutil call.
        """
        host_os = detect_host_os()
        if host_os is HostOS.LINUX:
            return self._probe_psi_linux()
        if host_os is HostOS.MACOS:
            return self._probe_pressure_macos()
        return False

    @staticmethod
    def _probe_psi_linux() -> bool:
        """Read PSI full avg10 from /proc/pressure/memory (Linux): True if pressure >= threshold
        (reject), False otherwise, also when PSI is unavailable or its value is not finite."""
        try:
            content = Path("/proc/pressure/memory").read_text()
            for line in content.splitlines():
                if line.startswith("full "):
                    for token in line.split():
                        if token.startswith("avg10="):
                            val = float(token[6:])
                            return math.isfinite(val) and val >= _PSI_FULL_THRESHOLD_PCT
        except (OSError, ValueError):
            pass
        return False

    @staticmethod
    def _probe_pressure_macos() -> bool:
        """Read kernel memory pressure level via sysctl (macOS): True at CRITICAL (reject),
        False otherwise (NORMAL, WARN, an unknown level, a failed sysctl)."""
        level = _read_macos_memory_pressure_level()
        # CRITICAL only: WARN is common during normal operation (active compressor, many apps
        # open), and blocking at WARN makes the tool unusable on most dev machines
        return level == _MACOS_PRESSURE_CRITICAL

    async def _self_wake_loop(self) -> None:
        """Periodically notify blocked waiters so Gate 3 re-probes.

        Closes the wakeup gap: if external processes free memory and no
        local release() fires, blocked waiters would otherwise wait until
        timeout. This timer ensures they re-check every N seconds.
        """
        while True:
            await asyncio.sleep(GATE3_SELF_WAKE_INTERVAL_SECONDS)
            await self._self_wake()

    async def _self_wake(self) -> None:
        """One wake-up of _self_wake_loop: every waiter checks again, and gate 4 makes room again
        (a cache may hold new entries). It lifts the shared stop too (_ease_shared_stop): no VM
        may end to lift it (none runs: a pool refill; long sessions)."""
        async with self._condition:
            self._room_version += 1
            self._ease_shared_stop(now=asyncio.get_running_loop().time())
            self._condition.notify_all()

    def _compute_budgets(self) -> None:
        """Compute effective budgets from host resources and overcommit ratios.

        Memory: host_total * (1 - reserve_ratio) * overcommit_ratio
        CPU:    (host_cpu_count - host_cpu_reserve) * overcommit_ratio
        Rounded to 6 decimals once: the float noise of the products (1.4 - 0.5 cores at 12.5x
        is 11.249999999999998) would refuse a VM that the budget facts (budget_facts) admit.
        """
        available_mb = self._host_memory_mb * (1.0 - self._host_memory_reserve_ratio)
        self._memory_budget_mb = round(max(0.0, available_mb) * self._memory_overcommit_ratio, 6)
        available_cpus = max(0.0, self._host_cpu_count - self._host_cpu_reserve_cores)
        self._cpu_budget = round(available_cpus * self._cpu_overcommit_ratio, 6)

    async def acquire(
        self,
        vm_id: str,
        memory_mb: float,
        cpu_cores: float,
        timeout: float,
        *,
        use_tcg: bool,
        disk_mb: int,
        held: frozenset[Path],
    ) -> ResourceReservation:
        """Acquire resources for a VM, blocking if insufficient.

        Checks all gates under lock (memory, CPU, available memory, memory
        pressure, home disk space). Blocks on asyncio.Condition if resources are
        insufficient; when the home disk space blocks, evicts cache entries that no
        VM uses first (add_cache), again at each wake-up where the room can have
        changed (_room_version), until a pass of evictions gives no space back.
        Raises at once when the request exceeds the
        whole memory or CPU budget, or does not fit on the disk even with no VM or
        session of this controller left and every entry no other process uses
        evicted, `held` left out (then before any eviction, or after the first pass
        where the evicted entries gave back only part of their size); after
        `timeout` otherwise, or when the controller stops.

        Args:
            vm_id: VM identifier (for logging, does not need to be unique)
            memory_mb: Guest memory in MB (overhead is added automatically)
            cpu_cores: Guest CPU cores (DEFAULT_VM_CPU_OVERHEAD_CORES is added automatically)
            timeout: Max seconds to wait for resources
            use_tcg: Whether TCG emulation is used (adds extra memory overhead)
            disk_mb: Home disk size the guest can fill
            held: Cache entries (primary files) and folders the request keeps while it
                waits: its cache leases, its loaded instance's folder. Their space does
                not come back meanwhile, so gate 4 does not count on it.

        Returns:
            ResourceReservation tracking the allocated resources

        Raises:
            TransientError: CAPACITY: resources not available within timeout (the
                message and `context["blocking_gate"]` name the gate); LOCAL_DISK: the
                home disk does not fit on its filesystem and this controller cannot
                make room
            RouteError: CAPACITY_NEVER_FITS: a request larger than the whole budget
            InvalidInputError: SCHEDULER_STATE: a stopped controller
        """
        # Budgets are static: a request above the whole budget can never be
        # admitted, so fail now instead of after `timeout`.
        self.check_fits(vm_id, memory_mb, cpu_cores, use_tcg=use_tcg, target={})
        total_memory_mb, total_cpu = _vm_totals(memory_mb, cpu_cores, use_tcg=use_tcg)
        reservation = ResourceReservation(
            vm_id=vm_id,
            memory_mb=total_memory_mb,
            cpu_cores=total_cpu,
            disk_mb=disk_mb,
        )

        logger.debug(
            "Acquiring resources",
            extra={
                "vm_id": vm_id,
                "requested_memory_mb": round(total_memory_mb),
                "requested_cpu": round(total_cpu, 2),
                "timeout": timeout,
                "allocated_memory_mb": round(self._allocated_memory_mb),
                "allocated_cpu": round(self._allocated_cpu, 1),
                "vm_slots": self._allocated_vm_slots,
            },
        )

        request = (total_memory_mb, total_cpu, float(disk_mb))
        blocked: _Gate | None = None  # the gate of the last check
        made_room = False  # gate 4 made room for this request, and the room has not changed since
        room_seen = -1  # self._room_version when it last made room
        evictions = _Evictions()
        admitted = False
        pass_on = False  # its pass stopped only itself: another waiter's pass may still make room
        try:
            async with asyncio.timeout(timeout):
                while True:
                    async with self._condition:
                        if pass_on:  # the wake-up that made it make room goes to the next waiter
                            self._condition.notify(1)
                        self._waiting[request] += 1
                        try:
                            while not self._stop_event.is_set():
                                blocked = self._blocking_gate(total_memory_mb, total_cpu, disk_mb)
                                if blocked is None:
                                    break
                                # the room changed since it made room (a release while it ran
                                # _make_room, out of _waiting): make room again
                                made_room = made_room and room_seen == self._room_version
                                if blocked is _Gate.HOME_DISK_SPACE and not made_room:
                                    break  # make room first, out of the lock: file work takes time
                                await self._condition.wait()
                        finally:  # the lock is held again here, also on timeout
                            self._waiting[request] -= 1
                            if not self._waiting[request]:
                                del self._waiting[request]
                        if self._stop_event.is_set():
                            raise InvalidInputError(
                                f"Admission controller stopped while waiting for VM {vm_id}",
                                code=ErrorCode.SCHEDULER_STATE,
                                context={"vm_id": vm_id},
                                hint=None,
                                instance_id=None,
                            )
                        if blocked is None:
                            admitted = True
                            return self._commit(reservation)
                    self._disk_needs[reservation.reservation_id] = (disk_mb / self._disk_overcommit_ratio, evictions)
                    room_seen = self._room_version
                    await self._make_room(vm_id, disk_mb, held=held, evictions=evictions)
                    made_room = True
                    pass_on = evictions.stopped and not evictions.shared_stop

        except TimeoutError:
            raise self._timeout_error(reservation, timeout=timeout, blocked=blocked) from None
        finally:
            self._disk_needs.pop(reservation.reservation_id, None)
            if not admitted and not self._stop_event.is_set():
                # A release wakes one waiter of a size (release): one that leaves (LOCAL_DISK, a
                # timeout, cancelled even once notified) passes that wake-up on, else the next one
                # waits for the self-wake (10 s)
                async with self._condition:
                    self._condition.notify(1)

    def _commit(self, reservation: ResourceReservation) -> ResourceReservation:
        """Count an admitted reservation (the condition's lock held)."""
        self._allocated_memory_mb += reservation.memory_mb
        self._allocated_cpu += reservation.cpu_cores
        self._allocated_disk_mb += reservation.disk_mb
        self._allocated_vm_slots += 1
        self._reservations[reservation.reservation_id] = reservation

        logger.debug(
            "Resource reservation acquired",
            extra={
                "vm_id": reservation.vm_id,
                "reservation_id": reservation.reservation_id,
                "reserved_memory_mb": round(reservation.memory_mb),
                "reserved_cpu": round(reservation.cpu_cores, 2),
                "total_allocated_memory_mb": round(self._allocated_memory_mb),
                "total_allocated_cpu": round(self._allocated_cpu, 1),
                "vm_slots": self._allocated_vm_slots,
            },
        )
        # One release or pass of evictions can make room for several waiters: pass the wake-up on
        self._wake_waiters()
        return reservation

    def _wake_waiters(self) -> None:
        """Wake one waiter when all waiters ask for the same size: the first fits whenever any
        does, and each admission wakes the next (_commit), so a large release still admits
        several, with no O(N²) wake-ups under burst load (500 waiters x 500 releases: 250k).
        Mixed sizes (VmSpec): the first waiter may be too large while a smaller one fits, so
        all check again. The condition's lock held."""
        if len(self._waiting) > 1:
            self._condition.notify_all()
        else:
            self._condition.notify(1)

    def check_fits(
        self, vm_id: str, memory_mb: float, cpu_cores: float, *, use_tcg: bool, target: Mapping[RouteKey, str]
    ) -> tuple[dict[RouteKey, str], dict[RouteKey, str | None]]:
        """The budget facts a VM of this size needs (budget_need) and this budget's (None: no
        limit), as a RouteError's `required` and `found` give them. Raise when the VM, with
        its overheads, is larger than the whole budget: it can never be admitted. `target`:
        route facts the caller checked before (this worker has them), added to the error's
        `required` and `found`.

        Raises:
            RouteError: CAPACITY_NEVER_FITS
        """
        # The whole target (both dimensions) rounded up, the budget rounded down (as
        # budget_facts): a worker whose budget is not below `required` admits it.
        memory_budget, cpu_budget = self.budget_facts()
        need = budget_need(memory_mb, cpu_cores, use_tcg=use_tcg)
        have = {
            RouteKey.MEMORY_MB: None if memory_budget is None else str(round(memory_budget)),
            RouteKey.CPU: None if cpu_budget is None else f"{cpu_budget:.2f}",
        }
        total_memory_mb, total_cpu = _vm_totals(memory_mb, cpu_cores, use_tcg=use_tcg)
        if total_memory_mb <= self._memory_budget_mb and total_cpu <= self._cpu_budget:
            return need, have
        # The message and context give the values of required and found (one may be unlimited)
        mem_budget_str = have[RouteKey.MEMORY_MB] or "unlimited"
        cpu_budget_str = have[RouteKey.CPU] or "unlimited"
        raise RouteError(
            f"VM {vm_id} can never fit: requested {need[RouteKey.MEMORY_MB]}MB memory, "
            f"{need[RouteKey.CPU]} CPU; total budget {mem_budget_str}MB memory, "
            f"{cpu_budget_str} CPU.",
            code=ErrorCode.CAPACITY_NEVER_FITS,
            required={**target, **need},
            found={**target, **have},
            context={
                "vm_id": vm_id,
                "requested_memory_mb": need[RouteKey.MEMORY_MB],
                "requested_cpu": need[RouteKey.CPU],
                "memory_budget_mb": mem_budget_str,
                "cpu_budget": cpu_budget_str,
            },
            hint="Run it on a worker whose budget holds it (see `required`), or ask for a smaller VM.",
            instance_id=None,
        )

    def budget_facts(self) -> tuple[float | None, float | None]:
        """The whole memory budget (whole MB) and CPU budget (cores, two decimals), rounded
        down, VM overheads counted as in CAPACITY_NEVER_FITS; None for a dimension no probe
        sized (no limit). A request whose `required` is not above them fits the budget."""
        return (
            None if math.isinf(self._memory_budget_mb) else float(math.floor(self._memory_budget_mb)),
            None if math.isinf(self._cpu_budget) else math.floor(round(self._cpu_budget * 100, 6)) / 100,
        )

    def _budget_texts(self) -> tuple[str, str]:
        """The memory budget (whole MB) and the CPU budget (0.1 core) as error texts;
        "unlimited" for a dimension that no probe sized."""
        return (
            "unlimited" if math.isinf(self._memory_budget_mb) else str(round(self._memory_budget_mb)),
            "unlimited" if math.isinf(self._cpu_budget) else str(round(self._cpu_budget, 1)),
        )

    def _timeout_error(
        self, reservation: ResourceReservation, *, timeout: float, blocked: _Gate | None
    ) -> TransientError:
        """TransientError CAPACITY of a request that waited `timeout`: the gate that
        blocked it last, the budgets, and the disk figures."""
        mem_budget_str, cpu_budget_str = self._budget_texts()
        floor_info = ""
        system_avail = self._probe_available_memory()
        if system_avail is not None:
            floor_info = f" System available: {round(system_avail)}MB (floor: {self._available_memory_floor_mb}MB)."
        space = self._disk_space(reservation.disk_mb)
        disk_info = (
            space.describe()
            if space is not None
            else f"Home disk space: unknown, {round(self._allocated_disk_mb)}MB of disks admitted, "
            f"{round(reservation.disk_mb)}MB requested."
        )
        gate_info = f", blocked by {blocked}" if blocked is not None else ""
        return TransientError(
            f"Resource admission timeout after {timeout}s for VM {reservation.vm_id}{gate_info}. "
            f"Requested: {round(reservation.memory_mb)}MB memory, {round(reservation.cpu_cores, 2)} CPU. "
            f"Budget: {mem_budget_str}MB memory "
            f"({round(self._allocated_memory_mb)}/{mem_budget_str} allocated), "
            f"{cpu_budget_str} CPU "
            f"({round(self._allocated_cpu, 1)}/{cpu_budget_str} allocated), "
            f"{self._allocated_vm_slots} VM slots in use.{floor_info} {disk_info}",
            code=ErrorCode.CAPACITY,
            context={
                "vm_id": reservation.vm_id,
                "blocking_gate": blocked.name.lower() if blocked is not None else None,
                "requested_memory_mb": round(reservation.memory_mb),
                "requested_cpu": round(reservation.cpu_cores, 2),
                "allocated_memory_mb": round(self._allocated_memory_mb),
                "memory_budget_mb": mem_budget_str,
                "allocated_cpu": round(self._allocated_cpu, 1),
                "cpu_budget": cpu_budget_str,
                "vm_slots": self._allocated_vm_slots,
                "available_memory_floor_mb": self._available_memory_floor_mb,
                "system_available_memory_mb": (round(system_avail) if system_avail is not None else None),
                **(
                    space.context()
                    if space is not None
                    else {
                        "disk_admitted_mb": round(self._allocated_disk_mb),
                        "disk_requested_mb": round(reservation.disk_mb),
                    }
                ),
            },
            hint=None,
            instance_id=None,
        )

    def _disk_full_error(self, vm_id: str, space: _DiskSpace) -> TransientError:
        """TransientError LOCAL_DISK of a home disk that only other processes can make room for."""
        logger.warning("Home disk does not fit on its filesystem", extra={"vm_id": vm_id, **space.context()})
        return TransientError(
            f"VM {vm_id} does not fit on the filesystem of the home disks: neither cache evictions "
            f"nor the end of this worker's VMs and sessions can make room. {space.describe()}",
            code=ErrorCode.LOCAL_DISK,
            context={"vm_id": vm_id, "blocking_gate": _Gate.HOME_DISK_SPACE.name.lower(), **space.context()},
            hint="Free space on the filesystem of home_disk_dir, or ask for a smaller disk_mb.",
            instance_id=None,
        )

    async def release(self, reservation: ResourceReservation) -> None:
        """Release resources for a VM and signal waiters.

        Args:
            reservation: The reservation returned by acquire()
        """
        async with self._condition:
            if reservation.reservation_id not in self._reservations:
                logger.debug(
                    "Reservation already released (idempotent)",
                    extra={"vm_id": reservation.vm_id, "reservation_id": reservation.reservation_id},
                )
                return

            self._allocated_memory_mb -= reservation.memory_mb
            self._allocated_cpu -= reservation.cpu_cores
            self._allocated_disk_mb -= reservation.disk_mb
            self._allocated_vm_slots -= 1
            del self._reservations[reservation.reservation_id]

            # Snap counters to zero when empty (prevents IEEE 754 drift)
            if self._allocated_vm_slots == 0:
                self._allocated_memory_mb = 0.0
                self._allocated_cpu = 0.0
                self._allocated_disk_mb = 0.0

            logger.debug(
                "Resource reservation released",
                extra={
                    "vm_id": reservation.vm_id,
                    "reservation_id": reservation.reservation_id,
                    "freed_memory_mb": round(reservation.memory_mb),
                    "freed_cpu": reservation.cpu_cores,
                    "remaining_allocated_memory_mb": round(self._allocated_memory_mb),
                    "remaining_allocated_cpu": round(self._allocated_cpu, 1),
                    "vm_slots": self._allocated_vm_slots,
                },
            )

            # Gate 3 recovery (external memory pressure drop) is handled by _self_wake_loop
            # (notify_all() every 10s)
            self._room_version += 1
            self._ease_shared_stop(now=asyncio.get_running_loop().time())
            self._wake_waiters()

    def _blocking_gate(self, memory_mb: float, cpu_cores: float, disk_mb: float) -> _Gate | None:
        """The first admission gate that refuses the request.

        Args:
            memory_mb: Total memory needed (guest + overhead)
            cpu_cores: CPU cores needed
            disk_mb: Home disk size needed

        Returns:
            None if all gates pass
        """
        # Gate 1: Memory budget
        if self._allocated_memory_mb + memory_mb > self._memory_budget_mb:
            logger.debug(
                "Gate 1 (memory budget) blocked",
                extra={
                    "requested_mb": round(memory_mb),
                    "allocated_mb": round(self._allocated_memory_mb),
                    "budget_mb": round(self._memory_budget_mb),
                },
            )
            return _Gate.MEMORY_BUDGET

        # Gate 2: CPU budget
        if self._allocated_cpu + cpu_cores > self._cpu_budget:
            logger.debug(
                "Gate 2 (CPU budget) blocked",
                extra={
                    "requested_cpu": cpu_cores,
                    "allocated_cpu": round(self._allocated_cpu, 1),
                    "budget_cpu": round(self._cpu_budget, 1),
                },
            )
            return _Gate.CPU_BUDGET

        # Gate 3a: System available memory (always-on, inline probe ~3µs, cached 100ms)
        # Provides cross-process backpressure: sees memory consumed by
        # other Scheduler instances, xdist workers, or external processes.
        system_available = self._probe_available_memory()
        if system_available is not None and system_available - memory_mb < self._available_memory_floor_mb:
            logger.debug(
                "Gate 3a (available-memory floor) blocked",
                extra={
                    "requested_mb": round(memory_mb),
                    "system_available_mb": round(system_available),
                    "after_admit_mb": round(system_available - memory_mb),
                    "floor_mb": self._available_memory_floor_mb,
                },
            )
            return _Gate.AVAILABLE_MEMORY

        # Gate 3b: Memory pressure (Linux PSI / macOS sysctl, ~3-5µs, cached 100ms)
        # Each platform has its own rejection logic (no shared threshold).
        if self._probe_memory_pressure():
            logger.debug("Gate 3b (memory pressure) blocked")
            return _Gate.MEMORY_PRESSURE

        # Gate 4: Home disk space (statvfs + one stat per disk, cached 100ms)
        space = self._disk_space(disk_mb)
        if space is not None and space.shortfall_mb > 0:
            logger.debug("Gate 4 (home disk space) blocked", extra=space.context())
            return _Gate.HOME_DISK_SPACE

        return None

    def _disk_space(self, disk_mb: float) -> _DiskSpace | None:
        """Gate 4 figures for a home disk of `disk_mb`; None when the free space cannot be read."""
        disk = self._probe_disk()
        if disk is None:
            return None
        free_mb, written_mb = disk
        unwritten_mb = max(0.0, self._allocated_disk_mb - written_mb)
        return _DiskSpace(
            free_mb=free_mb,
            written_mb=written_mb,
            admitted_mb=self._allocated_disk_mb,
            reserved_mb=unwritten_mb / self._disk_overcommit_ratio,
            requested_mb=disk_mb,
            needed_mb=disk_mb / self._disk_overcommit_ratio,
            floor_mb=self._disk_free_floor_mb,
        )

    def add_cache(self, cache: EvictableCache) -> None:
        """Let gate 4 evict entries of `cache` before a request waits for disk space.
        A cache on another filesystem than disk_dir is left out: its space is not the
        space gate 4 lacks. Metadata calls only."""
        try:
            same_filesystem = cache.cache_dir.stat().st_dev == self._disk_dir.stat().st_dev
        except OSError:
            same_filesystem = False
        if not same_filesystem:
            logger.debug(
                "Cache not on the home disk filesystem: gate 4 leaves it", extra={"cache_dir": str(cache.cache_dir)}
            )
        elif cache not in self._caches:
            self._caches.append(cache)

    async def _make_room(self, vm_id: str, disk_mb: float, *, held: frozenset[Path], evictions: _Evictions) -> None:
        """Gate 4 blocks a request. Raise TransientError (LOCAL_DISK) when neither the end of
        this controller's VMs and sessions nor evictions can make room (`held` left out), before
        each pass of evictions and after the last. The check and the pass hold _evict_lock: a
        pass of another request ends first, so its removals show (or count: _Pending). A pass
        that gave space back and left the request short is followed by the next one at once,
        with no wait for a wake-up; none follows a pass that gave almost nothing back
        (`evictions`)."""
        if self._no_gain_passes >= _SHARED_NO_GAIN_PASSES and not evictions.stopped:
            evictions.stopped = evictions.shared_stop = True  # the caches gave nothing to the requests before
        while not self._stop_event.is_set():  # acquire raises SCHEDULER_STATE then
            async with self._evict_lock:
                if self._stop_event.is_set():  # it waited for the lock behind a pass that stop() ended
                    return
                if evictions.checked and self._nothing_to_evict_at == self._room_version:
                    return  # a pass at this room version found nothing to evict: wait for a change
                _forget_cached(self, ResourceAdmissionController._probe_disk)  # the free space now
                if (space := self._disk_space(disk_mb)) is not None:
                    self._pending_mb(space)  # its verdict can stop this request: then no cache entry counts
                    if space.shortfall_mb <= 0:
                        return  # it fits now (a pass of another request made room): no listing
                later_mb = await self._later_mb(held=held, cache_share=evictions.cache_share)
                self._check_room(vm_id, disk_mb, later_mb=later_mb)
                evictions.checked = True
                if evictions.stopped or not await self._evict_for_disk(disk_mb, evictions=evictions, later_mb=later_mb):
                    return

    def _check_room(self, vm_id: str, disk_mb: float, *, later_mb: float) -> None:
        """Raise TransientError (LOCAL_DISK) when only other processes can make room: `later_mb`
        comes back once this controller's VMs and sessions are gone, and the removed space still
        coming (_pending_mb) once it shows."""
        space = self._disk_space(disk_mb)
        if space is not None and not space.fits_after(space.needed_mb, later_mb=later_mb + self._pending_mb(space)):
            raise self._disk_full_error(vm_id, space)

    def _pending_mb(self, space: _DiskSpace) -> float:
        """The removed space still coming (_Pending) that `space` does not show yet, in MB. Once
        it has shown, its time is over, or its sync has ended, the pending pass ends: its request
        gets the verdict on it (_judge), except after a sync (no verdict: _Pending.sync)."""
        pending = self._pending
        if pending is None:
            return 0.0
        now = asyncio.get_running_loop().time()
        gain_mb = space.gain_since(pending.start, net_of_writes=pending.net_of_writes)
        if gain_mb > pending.shown_mb:
            pending.shown_mb, pending.grew_at = gain_mb, now
        left_mb = pending.evicted_mb - gain_mb
        settled = (
            pending.grew_at is not None
            and now - pending.grew_at >= _BACKGROUND.quiet
            and gain_mb >= pending.evicted_mb * _GAIN_MIN_RATIO
        )
        over = now > pending.until or settled or (pending.sync is not None and pending.sync.done())
        if left_mb > 0 and not over:
            return left_mb
        self._pending = None  # the timer (_recheck_later) sees it and tells the waiters: a change of room
        if pending.sync is None:
            self._judge(pending.evictions, evicted_mb=pending.evicted_mb, gain_mb=gain_mb, space=space)
        return 0.0

    def _judge(self, evictions: _Evictions, *, evicted_mb: float, gain_mb: float, space: _DiskSpace) -> None:
        """The verdict on a pass of `evictions` once its space has shown: what it gave back
        counts for the cache entries left (_Evictions.cache_share), and less than
        _GAIN_MIN_RATIO of its bytes stops the evictions of the request, and of every other
        request that gate 4 blocks now: the same caches would give them nothing either (else on
        ZFS each request learns it alone, one 11 s pending pass after the other)."""
        evictions.evicted_mb += evicted_mb
        evictions.gained_mb += gain_mb
        if gain_mb >= evicted_mb * _GAIN_MIN_RATIO:
            self._gave_space()
            return
        evictions.stopped, evictions.shared_stop = True, False
        self._no_gain_passes += 1
        shared = self._no_gain_passes >= _SHARED_NO_GAIN_PASSES
        if shared:
            for _, other in self._disk_needs.values():
                if not other.stopped:
                    other.stopped = other.shared_stop = True
        logger.info(
            "Cache eviction gave no home disk space back: "
            + ("the requests waiting for disk space evict no more" if shared else "this request evicts no more"),
            extra={"evicted_mb": round(evicted_mb), "gain_mb": round(gain_mb), "free_mb": round(space.free_mb)},
        )

    def _gave_space(self) -> None:
        """A pass gave space back (a gain, or the requests fit after it): the run of passes of
        nothing ends, the shared stops are lifted, the next lift comes at once (_ease_shared_stop),
        and the room changed."""
        self._no_gain_passes = 0
        self._ease_at = self._ease_gap = 0.0
        self._room_version += 1
        self._lift_shared_stops()

    def _ease_shared_stop(self, *, now: float) -> None:
        """A release (the VM that ended can leave new cache entries) or the self-wake, at event
        loop time `now`, while passes of nothing share a stop: the shared stops are lifted, and one
        more pass evicts, for its own request only (_no_gain_passes stays above 0: _others_mb); if
        it gives nothing back too, the stop is shared again. A ZFS or btrfs snapshot keeps every
        block: each such pass removes good entries for nothing. So a lift comes
        GATE3_SELF_WAKE_INTERVAL_SECONDS after the last at the earliest, then twice as long after
        each, up to _EASE_MAX_SECONDS, until a pass gives space back (_gave_space)."""
        if self._no_gain_passes < _SHARED_NO_GAIN_PASSES or now < self._ease_at:
            return
        self._ease_gap = min(_EASE_MAX_SECONDS, 2 * self._ease_gap or GATE3_SELF_WAKE_INTERVAL_SECONDS)
        self._ease_at = now + self._ease_gap
        self._no_gain_passes = _SHARED_NO_GAIN_PASSES - 1
        self._lift_shared_stops()

    def _lift_shared_stops(self) -> None:
        """The requests stopped by the verdicts on the passes of others may evict again (a VM
        that ends can leave new cache entries); their own verdicts stay."""
        for _, evictions in self._disk_needs.values():
            if evictions.shared_stop:
                evictions.stopped = evictions.shared_stop = False

    async def _later_mb(self, *, held: frozenset[Path], cache_share: float) -> float:
        """What comes back once this controller's VMs and sessions are gone, beside their home
        disks, in MB: the files in the folders of disk_dir (saved-instance work files of the
        sessions), and `cache_share` of the bytes of the cache entries evictions can then take
        (_Evictions.cache_share). `held` is left out."""
        total = await asyncio.to_thread(self._work_bytes, held=held)
        for cache in list(self._caches) if cache_share else []:
            try:
                total += cache_share * await cache.reclaimable_bytes(held=held)
            except OSError as e:  # a file error of the cache (folder removed, no access): it gives nothing
                logger.warning(
                    "Cache size for home disk space unreadable",
                    extra={"cache_dir": str(cache.cache_dir), "error": str(e)},
                )
        return total / _MIB

    def _work_bytes(self, *, held: frozenset[Path]) -> int:
        """Blocks of the files in the folders of disk_dir (the home disks are its files; the
        work files of saved-instance saves and loads are in its folders), the folders in
        `held` left out. They go when their session closes. Blocking: a worker thread."""
        total = 0
        try:
            folders = [Path(entry.path) for entry in os.scandir(self._disk_dir) if entry.is_dir(follow_symlinks=False)]
        except OSError:
            return 0
        for folder in folders:
            for root, dirs, files in os.walk(folder):
                dirs[:] = [name for name in dirs if Path(root, name) not in held]
                if Path(root) in held:
                    continue
                for name in files:
                    with contextlib.suppress(OSError):  # removed meanwhile
                        total += allocated_bytes(os.lstat(os.path.join(root, name)))  # noqa: PTH118
        return total

    async def _evict_for_disk(self, disk_mb: float, *, evictions: _Evictions, later_mb: float) -> bool:
        """One pass for a request that gate 4 blocks (the caller holds _evict_lock): evict the
        oldest entry that no VM uses of each cache in turn until no cache has such an entry
        left, or the entries evicted cover what the request lacked at the start and what the
        other requests that gate 4 blocks need (_disk_needs). No pass while the space of an
        earlier one still shows (_Pending). Then the free space is read once it shows the
        removals (_space_after_pass). Space still coming then (_Pending) gets its verdict once
        it has shown; else a pass that gave back less than _GAIN_MIN_RATIO of its bytes stops
        the evictions of the request (`evictions`). A pass whose syncfs may hide its gain
        (_Evictions.sync_first) gives no verdict: the next one syncs before it evicts. Returns
        whether the request checks its room again at once: the pass evicted, its space
        showed (or its sync hid the gain: the next pass syncs first, _Evictions.sync_first),
        and the request does not fit."""
        if evictions.sync_first and not await self._sync_filesystem():
            self._recheck_soon()  # the sync still writes the dirty data: check again once it may be done
            return False
        _forget_cached(self, ResourceAdmissionController._probe_disk)  # the pass of another request may have ended
        start = self._disk_space(disk_mb)
        if start is None or start.shortfall_mb <= 0:
            return False
        if self._pending_mb(start):
            self._recheck_soon()  # the space of an earlier pass still shows: check again once it may have
            return False
        if evictions.stopped:  # the verdict on that pending pass stopped this request too
            return False
        others_mb = self._others_mb(start, later_mb=later_mb)
        room_version = self._room_version
        evicted_mb = await self._evict_entries(disk_mb, lacking_mb=start.shortfall_mb + others_mb, others_mb=others_mb)
        if not evicted_mb:  # nothing evicted, or the requests fit (then the room version changed)
            if self._room_version == room_version:
                self._nothing_to_evict_at = room_version
            return False
        after = await self._space_after_pass(disk_mb, start=start, evicted_mb=evicted_mb)
        end = after.space
        if end is None or self._stop_event.is_set():
            return False
        if end.shortfall_mb <= 0:
            self._gave_space()  # it fits: the run of passes of nothing ends
            return False
        gain_mb = after.gain_mb  # the largest seen: a writer of another process lowers the last read
        more_s, grown = after.more_s, after.growing
        if after.synced and gain_mb < evicted_mb * _GAIN_MIN_RATIO:
            if not evictions.sync_first:
                evictions.sync_first = True
                logger.info(
                    "Cache eviction showed no home disk space after a sync: the next pass syncs first",
                    extra={"evicted_mb": round(evicted_mb), "gain_mb": round(gain_mb), "free_mb": round(end.free_mb)},
                )
                return True
            # measured alone after a sync, and still nothing: an overlay over ZFS shows it only at
            # the next transaction sync (its syncfs commits the intent log only); no ZFS here: no wait
            more_s, grown = (_zfs_settle().later if await asyncio.to_thread(_zfs_loaded) else 0.0), False
        if more_s:
            sync = self._syncfs_job if self._syncfs_job is not None and not self._syncfs_job.done() else None
            evictions.sync_first = evictions.sync_first or sync is not None  # no verdict before that sync ends
            now = asyncio.get_running_loop().time()
            self._pending = _Pending(
                evictions=evictions,
                start=start,
                evicted_mb=evicted_mb,
                net_of_writes=after.net_of_writes,
                until=now + more_s,
                sync=sync,
                shown_mb=gain_mb,
                grew_at=now if grown else None,  # it grew to the end of the reads: it may stop there
            )
            logger.info(
                "Evicted space still shows: the request waits for it",
                extra={"evicted_mb": round(evicted_mb), "gain_mb": round(gain_mb), "free_mb": round(end.free_mb)},
            )
            self._recheck_soon()
            return False
        self._judge(evictions, evicted_mb=evicted_mb, gain_mb=gain_mb, space=end)
        if others_mb and not evictions.stopped:
            async with self._condition:  # the pass made room for other waiters too
                self._condition.notify_all()
        return True

    def _others_mb(self, start: _DiskSpace, *, later_mb: float) -> float:
        """What a pass evicts for the other requests that gate 4 blocks, beyond its own
        shortfall: their reserves (_disk_needs), each only when it can fit after this request's
        listing (`later_mb`: a 16 GB request on a 4 GB disk gets nothing). None once a pass gave
        nothing back (_no_gain_passes; a lift of the shared stop leaves it above 0): until a
        pass gives space back, a pass evicts only for its own request (a reflink or snapshot
        filesystem keeps every block: each pass sized for every waiter would empty the cache)."""
        if self._no_gain_passes:
            return 0.0
        fitting = sum(need for need, _ in self._disk_needs.values() if start.fits_after(need, later_mb=later_mb))
        return max(0.0, fitting - start.needed_mb)

    async def _evict_entries(self, disk_mb: float, *, lacking_mb: float, others_mb: float) -> float:
        """Evict the oldest entry that no VM uses of each cache in turn, until `lacking_mb` is
        evicted or no cache has such an entry left: the MB evicted, or 0 as soon as the request
        and the others' `others_mb` fit (the free space showed the removals at once)."""
        orders = [(cache, cache.eviction_order()) for cache in self._caches]  # each cache listed once
        evicted_mb = 0.0
        while orders and evicted_mb < lacking_mb and not self._stop_event.is_set():
            cache, order = orders.pop(0)
            try:
                freed = await order.evict_next()
            except OSError as e:  # a file error of the cache (folder removed, no access)
                logger.warning(
                    "Cache eviction for home disk space failed",
                    extra={"cache_dir": str(cache.cache_dir), "error": str(e)},
                )
                freed = None
            if freed is None:
                continue
            orders.append((cache, order))
            evicted_mb += freed / _MIB
            _forget_cached(self, ResourceAdmissionController._probe_disk)
            space = self._disk_space(disk_mb)
            if space is None:
                return 0.0
            if space.shortfall_mb + others_mb <= 0:
                self._gave_space()  # the requests fit: the run of passes of nothing ends
                return 0.0
        return evicted_mb

    async def _space_after_pass(self, disk_mb: float, *, start: _DiskSpace, evicted_mb: float) -> _PassEnd:
        """The space once the free space shows what a pass evicted (_PassEnd). After
        _settle_removals, the space is read again while it grows (_wait_for_gain), or once where
        it shows in one step later (_Settle.later). Where the filesystem asks for it
        (_Settle.sync_once) and almost nothing showed, also when the space still grew (an
        overlay over btrfs shows nothing before its commit), one _sync_filesystem follows; a
        sync not done in time can still show the space."""
        try:
            settle = await asyncio.to_thread(_settle_removals, self._disk_dir)
        except (OSError, ValueError) as e:  # an odd /proc/self/mountinfo, no access: read as an unknown type
            logger.warning("Home disk filesystem type unreadable", extra={"error": str(e)})
            settle = _OTHER
        net = settle.counts_writes
        end, growing, gain = await self._wait_for_gain(disk_mb, start=start, evicted_mb=evicted_mb, settle=settle)
        shown = end is None or end.shortfall_mb <= 0 or gain >= evicted_mb
        if settle.later and not shown:
            return _PassEnd(
                space=end, more_s=settle.later, growing=False, synced=False, net_of_writes=net, gain_mb=gain
            )
        if not settle.sync_once or end is None or end.shortfall_mb <= 0 or gain >= evicted_mb * _GAIN_MIN_RATIO:
            more_s = _UNSEEN_SECONDS if growing else 0.0
            return _PassEnd(space=end, more_s=more_s, growing=growing, synced=False, net_of_writes=net, gain_mb=gain)
        if not await self._sync_filesystem():
            return _PassEnd(
                space=end, more_s=_UNSEEN_SECONDS, growing=False, synced=False, net_of_writes=net, gain_mb=gain
            )
        end, growing, gain = await self._wait_for_gain(
            disk_mb, start=start, evicted_mb=evicted_mb, settle=_AFTER_COMMIT
        )
        more_s = _UNSEEN_SECONDS if growing else 0.0
        return _PassEnd(
            space=end,
            more_s=more_s,
            growing=growing,
            synced=True,
            net_of_writes=_AFTER_COMMIT.counts_writes,
            gain_mb=gain,
        )

    async def _wait_for_gain(
        self, disk_mb: float, *, start: _DiskSpace, evicted_mb: float, settle: _Settle
    ) -> tuple[_DiskSpace | None, bool, float]:
        """The space for a request once the free space shows what a pass evicted, whether it
        still grew at `settle.limit` (more still comes), and the largest gain the reads saw (the
        verdict: another process that writes meanwhile lowers the last read). It is read from
        1 ms on, at doubling intervals, until the request fits, the gain since `start` covers
        `evicted_mb`, the gain has not grown for `settle.quiet`, or `settle.limit` passes. Each
        read is a gate 4 probe (one statfs, a listing of disk_dir); the waits are asyncio sleeps
        (cancellable)."""
        loop = asyncio.get_running_loop()
        started = grew_at = loop.time()
        step = _GAIN_FIRST_POLL_SECONDS
        best = -math.inf
        while True:
            _forget_cached(self, ResourceAdmissionController._probe_disk)
            space = self._disk_space(disk_mb)
            gain = None if space is None else space.gain_since(start, net_of_writes=settle.counts_writes)
            now = loop.time()
            if gain is not None and gain > best:
                best, grew_at = gain, now
            seen = max(best, 0.0)
            if (
                self._stop_event.is_set()
                or space is None
                or gain is None
                or space.shortfall_mb <= 0
                or gain >= evicted_mb
            ):
                return space, False, seen
            if now - grew_at >= settle.quiet:
                return space, False, seen
            if now - started >= settle.limit:
                return space, True, seen
            await asyncio.sleep(step)
            step = min(step * 2, settle.quiet / 4, _GAIN_MAX_POLL_SECONDS)

    def _recheck_soon(self) -> None:
        """Wake every waiter once the removed space still coming (_Pending) has shown or its
        pass has ended, once one waiting request fits, or once a running syncfs has ended: the
        requests that wait for it check again then. One timer at a time; none after stop()."""
        if not self._stop_event.is_set() and (self._recheck_task is None or self._recheck_task.done()):
            self._recheck_task = asyncio.get_running_loop().create_task(self._recheck_later())

    async def _recheck_later(self) -> None:
        """The timer of _recheck_soon: it reads the free space every _GAIN_MAX_POLL_SECONDS (a
        gate 4 probe) while a pass is pending (_pending_mb), or waits for a running syncfs. It
        wakes the waiters when a waiting request passes every gate (it is admitted within that
        time: ZFS, of its transaction sync), and once the pending pass or the syncfs has ended
        (the room changed: they make room again, so a request that waits for its sync before
        a pass runs that pass then, not at the self-wake); not before (100 waiters woken twice
        a second each listed the caches: 0.3 core per cache)."""
        while not self._stop_event.is_set():
            await asyncio.sleep(_GAIN_MAX_POLL_SECONDS)
            if (pending := self._pending) is None:
                if self._syncfs_job is None or self._syncfs_job.done():
                    break
                continue
            _forget_cached(self, ResourceAdmissionController._probe_disk)
            space = self._disk_space(0)
            if space is None:
                continue
            self._pending_mb(space)  # it ends once it has shown, or has grown and then stopped
            if self._pending is not pending:
                break
            if any(self._blocking_gate(*size) is None for size in self._waiting):
                async with self._condition:  # admitted at once; the others see no change of room
                    self._condition.notify_all()
        async with self._condition:
            self._room_version += 1  # the pending pass or the syncfs ended: they make room again
            self._condition.notify_all()

    async def _sync_filesystem(self) -> bool:
        """syncfs on the filesystem of disk_dir in a worker thread (_syncfs): whether it ended
        within _SYNCFS_WAIT_SECONDS. One runs at a time: a call while one runs waits for that
        one. A sync not done in time, or when stop() comes, goes on in its thread; none starts
        after stop() (asyncio.run would wait for its thread: 8-16 s on gp3)."""
        if self._stop_event.is_set():
            return False
        if self._syncfs_job is None or self._syncfs_job.done():
            self._syncfs_job = asyncio.get_running_loop().create_task(asyncio.to_thread(_syncfs, self._disk_dir))
        stopping = asyncio.ensure_future(self._stop_event.wait())
        try:
            await asyncio.wait(
                {asyncio.shield(self._syncfs_job), stopping},
                timeout=_SYNCFS_WAIT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            stopping.cancel()
        return self._syncfs_job.done() and not self._stop_event.is_set()

    def note_shared_blocks(self, disk_name: str, shared_bytes: int) -> None:
        """A home disk about to be cloned from an L1 template: its first `shared_bytes` of
        allocation take no free space yet (copy-on-write), so they are not counted as written.
        Call it before the clone, so no gate 4 probe meanwhile counts them, then
        forget_shared_blocks when the clone fails or is a plain copy (its blocks take free space
        at once: they are written)."""
        self._shared_disk_bytes[disk_name] = shared_bytes

    def forget_shared_blocks(self, disk_name: str) -> None:
        """Undo note_shared_blocks: the clone failed, or it copied the blocks."""
        self._shared_disk_bytes.pop(disk_name, None)

    @_ttl_cache(_PROBE_CACHE_TTL_SECONDS)
    def _probe_disk(self) -> tuple[float, float] | None:
        """Free space of disk_dir and the space its home disks use, in MB.

        Only this controller's disks are in disk_dir (one folder per manager),
        so their allocated blocks are what admitted guests have written. Blocks
        a clone still shares with its template are not written (the guest can
        still need that space when it overwrites them). Saved-instance work
        files (disk_dir/instances/) use free space too: this gate reserves none
        for them, and counts on them when it checks whether room can come back
        (_later_mb).
        """
        try:
            free = shutil.disk_usage(self._disk_dir).free
            written = 0
            present: set[str] = set()
            with os.scandir(self._disk_dir) as entries:
                for entry in entries:
                    if entry.name.endswith(".img"):
                        with contextlib.suppress(FileNotFoundError):  # VM destroyed meanwhile
                            allocated = allocated_bytes(entry.stat(follow_symlinks=False))
                            written += max(0, allocated - self._shared_disk_bytes.get(entry.name, 0))
                            present.add(entry.name)
        except OSError:
            return None
        # The notes of the disks gone since the last probe go; a note of a disk not made yet stays
        for gone in self._shared_disk_bytes.keys() & (self._disks_seen - present):
            del self._shared_disk_bytes[gone]
        self._disks_seen = present
        return free / (1024 * 1024), written / (1024 * 1024)

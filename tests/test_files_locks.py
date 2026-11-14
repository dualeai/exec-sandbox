"""CacheEntry.saving's in-process level (one asyncio.Lock per path and per event loop), and the
leases on cache entries (use lock) that eviction skips."""

import asyncio
import contextlib
import errno
import fcntl
import os
import stat
import subprocess
import sys
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

from exec_sandbox.base_cache_manager import BaseCacheManager
from exec_sandbox.files import locks as files_locks
from exec_sandbox.files.disk import file_stamp
from exec_sandbox.files.locks import CacheEntry
from tests.conftest import skip_unless_linux

_CHILD_LOCK = """
import fcntl, sys
lock = open(sys.argv[1], "a+b")
fcntl.lockf(lock, fcntl.LOCK_SH if sys.argv[2] == "lease" else fcntl.LOCK_EX)
print("held", flush=True)
sys.stdin.read()
"""


def _use_lock(entry: Path) -> Path:
    """The use lock file of `entry`, written out: a process of another version finds it by this name."""
    return entry.with_name(f"{entry.name}.use.lock")


@contextlib.contextmanager
def _locked_in_child(entry: Path, *, mode: str) -> Generator[None]:
    """The use lock of `entry` held by another process inside the block: shared ("lease")
    or exclusive ("evict"), as files.locks takes it (a record lock)."""
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", _CHILD_LOCK, str(_use_lock(entry)), mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline() == "held\n"
        yield
    finally:
        child.communicate()  # closes its stdin: it exits, its lock goes


def leased_in_child(entry: Path) -> contextlib.AbstractContextManager[None]:
    """`entry` leased by another process inside the block."""
    return _locked_in_child(entry, mode="lease")


def evicted_in_child(entry: Path) -> contextlib.AbstractContextManager[None]:
    """`entry` under the eviction of another process (its use lock exclusive) inside the block."""
    return _locked_in_child(entry, mode="evict")


class SizedCache(BaseCacheManager):
    """A real cache (locks, leases, LRU by atime) of tiny files that each declare their size
    in MB: tests count them as that many MB of used space."""

    _cache_ext = ".qcow2"

    def __init__(self, cache_dir: Path) -> None:
        super().__init__(cache_dir, setting="test_cache_dir")
        self._next_atime = 1_000_000
        self.sizes: dict[str, int] = {}
        """MB of each entry, by file name. Never read from the file: a read sets its atime
        (Linux relatime: the atimes here are of 1970, more than 24 h old), and the entry became
        the youngest, so an eviction took another one."""

    def add(self, name: str, size_mb: int) -> Path:
        """A new entry, younger than every earlier one."""
        path = self.cache_dir / f"{name}{self._cache_ext}"
        path.write_text(str(size_mb))
        self.sizes[path.name] = size_mb
        self._next_atime += 10
        os.utime(path, (self._next_atime, self._next_atime))
        return path

    def _entry_bytes(self, path: Path) -> int:
        return self.sizes[path.name] * (1024 * 1024) if path.exists() else 0


def leased_here(entry: Path) -> bool:
    """Whether a lease of this process on `entry` is still open (not released). Private state:
    no public call tells this process's leases apart from those of another process
    (evictable sees both, leased_elsewhere only the other ones)."""
    held = files_locks._use_locks.get(files_locks._lock_key(_use_lock(entry)))  # pyright: ignore[reportPrivateUsage]
    return held is not None and held.leases > 0


def evictable(entry: Path) -> bool:
    """Whether eviction could take the use lock of `entry` now (from a file handle of its own)."""
    try:
        with CacheEntry(entry).try_evicting():
            return True
    except BlockingIOError:
        return False


async def _contend(target: Path) -> None:
    """Two coroutines on one path: the second waits on the asyncio.Lock (it binds to this loop)."""
    holding = asyncio.Event()

    async def holder() -> None:
        async with CacheEntry(target).saving():
            holding.set()
            await asyncio.sleep(0)

    async def waiter() -> None:
        await holding.wait()
        async with CacheEntry(target).saving():
            pass

    await asyncio.gather(holder(), waiter())


def test_a_second_event_loop_gets_its_own_lock(tmp_path: Path) -> None:
    """asyncio.run twice in one process on the same path: the first loop's lock (bound to it
    by its waiter) is never handed to the second loop ("bound to a different event loop")."""
    target = tmp_path / "image.qcow2"

    asyncio.run(_contend(target))
    asyncio.run(_contend(target))


async def test_an_unused_path_leaves_no_lock(tmp_path: Path) -> None:
    """The map holds only the paths in use: a long run that touches many cache keys does not grow it.
    The map is private: no public call shows its size, and a leak there shows only as the memory
    of a long run."""
    await _contend(tmp_path / "image.qcow2")

    assert files_locks._async_locks[asyncio.get_running_loop()] == {}


class TestEntryLease:
    """A VM's lease keeps eviction off its cache entry until the last share is released."""

    async def test_missing_entry_is_a_miss_and_makes_no_lock_file(self, tmp_path: Path) -> None:
        entry = tmp_path / "gone.qcow2"
        assert await CacheEntry(entry).lease() is None
        assert list(tmp_path.iterdir()) == []

    async def test_lease_and_its_shares_hold_the_entry_until_the_last_release(self, tmp_path: Path) -> None:
        entry = tmp_path / "entry.qcow2"
        entry.write_bytes(b"x")
        lease = await CacheEntry(entry).lease()
        assert lease is not None
        share = lease.share()
        assert not evictable(entry)
        lease.release()
        lease.release()  # idempotent
        assert not evictable(entry)  # the share keeps the lock: no LOCK_UN on release
        share.release()
        assert evictable(entry)

    async def test_entry_removed_during_an_eviction_is_a_miss(self, tmp_path: Path) -> None:
        """An eviction holds the use lock: the lease waits for it, then finds the entry gone."""
        entry = tmp_path / "entry.qcow2"
        entry.write_bytes(b"x")
        with CacheEntry(entry).try_evicting():
            leasing = asyncio.create_task(CacheEntry(entry).lease())
            await asyncio.sleep(0)  # its try without wait met the eviction: the rest runs in a worker thread
            assert not leasing.done()
            entry.unlink()
        assert await leasing is None

    async def test_cancelled_lease_is_released_when_its_wait_ends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller stops waiting while an eviction holds the entry: the worker thread then
        gets the lease, and the event loop releases it. The private _lease tells when the worker
        got it: before that, the entry is evictable whether or not the release comes."""
        entry = tmp_path / "entry.qcow2"
        entry.write_bytes(b"x")
        leased = threading.Event()
        lease_now = files_locks._lease  # pyright: ignore[reportPrivateUsage]

        def lease_and_tell(target: Path, *, wait: bool) -> files_locks.EntryLease | None:
            lease = lease_now(target, wait=wait)
            if wait:
                leased.set()
            return lease

        monkeypatch.setattr(files_locks, "_lease", lease_and_tell)
        with CacheEntry(entry).try_evicting():
            leasing = asyncio.create_task(CacheEntry(entry).lease())
            await asyncio.sleep(0)
            leasing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await leasing
        for _ in range(5000):  # the worker thread leases the entry, then the loop releases it (5 s at most)
            if leased.is_set() and evictable(entry):
                break
            await asyncio.sleep(0.001)
        assert leased.is_set()
        assert evictable(entry)  # no lease left, in this process or another

    async def test_own_leases_are_not_another_process(self, tmp_path: Path) -> None:
        """This process's leases end with its VMs: the entry still counts as reclaimable."""
        entry = tmp_path / "entry.qcow2"
        entry.write_bytes(b"x")
        assert not CacheEntry(entry).leased_elsewhere()
        assert not (tmp_path / "entry.qcow2.use.lock").exists()  # the check makes no lock file
        lease = await CacheEntry(entry).lease()
        assert lease is not None
        assert not CacheEntry(entry).leased_elsewhere()
        lease.release()

    async def test_lease_of_another_process_stops_eviction(self, tmp_path: Path) -> None:
        """Across processes: the cache skips the entry while the other process holds it."""
        cache = SizedCache(tmp_path / "l2")
        entry = cache.add("held", 100)
        with leased_in_child(entry):
            assert CacheEntry(entry).leased_elsewhere()
            assert await cache.evict_oldest() is None
            assert entry.exists()
        assert not CacheEntry(entry).leased_elsewhere()
        assert await cache.evict_oldest() == 100 * 1024 * 1024
        assert not entry.exists()

    def test_an_eviction_of_another_process_is_no_lease(self, tmp_path: Path) -> None:
        """Another process holds the use lock exclusive (its eviction, or its own check): no VM
        runs on the entry, so it still counts as space evictions give back."""
        entry = tmp_path / "entry.qcow2"
        entry.write_bytes(b"x")
        with evicted_in_child(entry):
            assert not CacheEntry(entry).leased_elsewhere()
            assert not evictable(entry)  # this process cannot evict it meanwhile

    async def test_lease_of_another_process_is_seen_while_this_process_leases_too(self, tmp_path: Path) -> None:
        """Both processes lease the entry: the end of this process's VMs gives nothing back,
        the other process's VM still runs on it."""
        entry = tmp_path / "entry.qcow2"
        entry.write_bytes(b"x")
        lease = await CacheEntry(entry).lease()
        assert lease is not None
        try:
            with leased_in_child(entry):
                assert CacheEntry(entry).leased_elsewhere()
            assert not CacheEntry(entry).leased_elsewhere()
        finally:
            lease.release()

    async def test_one_file_per_use_lock_in_this_process(self, tmp_path: Path) -> None:
        """Leases, shares and checks of this process share one open file on the use lock:
        closing any other file of it would drop this process's lock (record locks)."""
        entry = tmp_path / "entry.qcow2"
        entry.write_bytes(b"x")
        first = await CacheEntry(entry).lease()
        second = await CacheEntry(entry).lease()
        assert first is not None
        assert second is not None
        share = first.share()
        assert not CacheEntry(entry).leased_elsewhere()  # a check while this process leases
        first.release()
        second.release()
        # the share alone holds the lock (the check closed no file of it): an eviction of another process fails
        assert not _child_can_evict(entry)
        share.release()
        assert _child_can_evict(entry)

    async def test_read_only_folder_leases_without_a_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read-only disk (EROFS): the lock file cannot be made and nothing can evict there.
        A hit still gets its lease; eviction skips the entry (its lock cannot be made)."""
        cache = SizedCache(tmp_path / "l2")
        entry = cache.add("ro", 10)
        real_open = Path.open

        def read_only_disk(self: Path, mode: str = "r", *args: object, **kwargs: object) -> object:
            if self == _use_lock(entry) and ("+" in mode or "a" in mode or "w" in mode):
                raise OSError(errno.EROFS, "Read-only file system", str(self))
            return real_open(self, mode, *args, **kwargs)  # type: ignore[call-overload]

        monkeypatch.setattr(Path, "open", read_only_disk)
        lease = await CacheEntry(entry).lease()
        assert lease is not None
        assert await cache.evict_oldest() is None
        assert entry.exists()
        lease.release()

    async def test_lock_file_of_a_folder_this_user_cannot_write_is_still_locked(self, tmp_path: Path) -> None:
        """A cache folder made by another user (no write access here): the lock file it has
        is opened for read, and the lease holds a real shared lock."""
        folder = tmp_path / "l2"
        folder.mkdir()
        entry = folder / "entry.qcow2"
        entry.write_bytes(b"x")
        _use_lock(entry).touch()
        _use_lock(entry).chmod(0o444)
        folder.chmod(0o555)
        try:
            lease = await CacheEntry(entry).lease()
            assert lease is not None
            assert _child_sees_a_lease(entry)  # read from a read-only file too (F_GETLK)
            lease.release()
            assert not _child_sees_a_lease(entry)
        finally:
            folder.chmod(0o755)

    async def test_a_pass_lists_the_cache_once(self, tmp_path: Path) -> None:
        """Each eviction of a pass takes the next entry of one listing, oldest access first: the
        cache is not listed, stat-ed and sorted again per eviction. So an entry added
        after the first eviction is not in the pass, and an entry in use when the pass tried it
        is not tried again in the same pass, even once it is free."""
        cache = SizedCache(tmp_path / "l2")
        held = cache.add("held", 50)
        oldest = cache.add("oldest", 100)
        newer = cache.add("newer", 200)
        for age, entry in enumerate((held, oldest, newer)):  # held is the oldest access
            os.utime(entry, (1_000 + age, 1_000 + age))
        order = cache.eviction_order()
        lease = await CacheEntry(held).lease()
        assert lease is not None

        assert await order.evict_next() == 100 * 1024 * 1024  # held skipped: in use
        lease.release()
        late = cache.add("late", 400)
        os.utime(late, (1, 1))  # the oldest access of all: first in a new listing
        assert await order.evict_next() == 200 * 1024 * 1024  # not late (not listed), not held (tried)
        assert await order.evict_next() is None

        assert [entry.exists() for entry in (held, oldest, newer, late)] == [True, False, False, True]

    async def test_an_entry_saved_again_during_the_pass_stays(self, tmp_path: Path) -> None:
        """Another process evicts an entry this pass listed and saves the key again: the new
        version (another file stamp) is not removed as the old one."""
        cache = SizedCache(tmp_path / "l2")
        first = cache.add("first", 100)
        second = cache.add("second", 200)
        os.utime(first, (1_000, 1_000))
        os.utime(second, (1_001, 1_001))
        order = cache.eviction_order()
        assert await order.evict_next() == 100 * 1024 * 1024
        second.unlink()
        again = cache.add("second", 200)
        os.utime(again, (2_000, 2_000))  # a new save: another mtime (and inode)
        assert await order.evict_next() is None
        assert again.exists()

    async def test_eviction_skips_an_entry_whose_lock_cannot_be_made(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full XFS or btrfs disk refuses the new lock file (ENOSPC): eviction goes on to the
        next entry instead of failing the whole pass. The disk stand-in refuses to make the use
        lock of the oldest entry."""
        cache = SizedCache(tmp_path / "l2")
        oldest = cache.add("oldest", 100)
        newer = cache.add("newer", 200)
        real_open = os.open

        def full_disk(path: str | os.PathLike[str], flags: int, *args: int) -> int:
            if Path(path) == _use_lock(oldest) and flags & os.O_CREAT:
                raise OSError(errno.ENOSPC, "No space left on device", str(path))
            return real_open(path, flags, *args)

        monkeypatch.setattr(os, "open", full_disk)
        assert await cache.evict_oldest() == 200 * 1024 * 1024
        assert oldest.exists()
        assert not newer.exists()

    async def test_removal_keeps_a_new_version_of_the_entry(self, tmp_path: Path) -> None:
        """A broken entry is removed only while it is the version the check saw: a new save
        of the key in between stays. _remove_broken is what the L1 and L2 caches call
        when their check finds an entry broken: a base cache has no public call to it."""
        cache = SizedCache(tmp_path / "l2")
        entry = cache.add("key", 10)
        seen = file_stamp(entry.stat())
        entry.unlink()
        cache.add("key", 20)  # saved again: another inode
        await cache._remove_broken(entry, stamp=seen, lease=None)  # pyright: ignore[reportPrivateUsage]
        assert entry.read_text() == "20"
        await cache._remove_broken(entry, stamp=file_stamp(entry.stat()), lease=None)  # pyright: ignore[reportPrivateUsage]
        assert not entry.exists()

    async def test_an_error_after_the_lock_leaves_no_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An error once the wait path holds the lock (a disk error): the lock goes (another
        process can evict), this process keeps no mark of the work (the next lease does not
        wait for it), and the next lease works."""
        entry = tmp_path / "entry.qcow2"
        entry.write_bytes(b"x")
        key = files_locks._lock_key(_use_lock(entry))  # pyright: ignore[reportPrivateUsage]
        waiting = threading.Event()
        real_lockf = fcntl.lockf

        def lock_then_fail(file: object, flags: int) -> None:
            if flags == fcntl.LOCK_SH:  # the wait path: another process evicts
                waiting.set()
                real_lockf(file, flags)
                raise OSError(errno.EIO, "Input/output error")
            real_lockf(file, flags)

        monkeypatch.setattr(fcntl, "lockf", lock_then_fail)
        try:
            with evicted_in_child(entry):
                leasing = asyncio.create_task(CacheEntry(entry).lease())
                await asyncio.to_thread(waiting.wait, 5)
            with pytest.raises(OSError, match="Input/output error"):
                async with asyncio.timeout(5):
                    await leasing
            monkeypatch.setattr(fcntl, "lockf", real_lockf)
            assert _child_can_evict(entry)
            async with asyncio.timeout(2):
                lease = await CacheEntry(entry).lease()
            assert lease is not None
            lease.release()
        finally:  # a mutant left the mark: let its waiting thread go (private: only for cleanup)
            with files_locks._use_guard:  # pyright: ignore[reportPrivateUsage]
                files_locks._use_locks.pop(key, None)  # pyright: ignore[reportPrivateUsage]
                files_locks._use_guard.notify_all()  # pyright: ignore[reportPrivateUsage]

    async def test_two_spellings_of_the_folder_are_one_lock(self, tmp_path: Path) -> None:
        """A symlink to the cache folder: a lease through one spelling, an eviction through the
        other. The eviction skips the entry, and the lease still holds for other processes
        after it."""
        real = tmp_path / "cache"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        (real / "entry.qcow2").write_bytes(b"x")
        lease = await CacheEntry(real / "entry.qcow2").lease()
        assert lease is not None
        try:
            assert not evictable(link / "entry.qcow2")
            assert not _child_can_evict(real / "entry.qcow2")
        finally:
            lease.release()
        assert evictable(link / "entry.qcow2")

    async def test_spellings_realpath_keeps_apart_are_one_lock(self, tmp_path: Path) -> None:
        """A case-insensitive filesystem (APFS by default): "Cache" and "cache" name one folder,
        and realpath keeps each spelling. A lease through one, an eviction through the other:
        the eviction skips the entry, and the lease still holds for other processes after its
        file closes."""
        real = tmp_path / "Cache"
        real.mkdir()
        other = tmp_path / "cache"
        if not other.exists():
            pytest.skip("case-sensitive filesystem: test_two_bind_mounts_of_the_folder_are_one_lock")
        assert os.path.realpath(other) != os.path.realpath(real)  # a realpath key would give two locks
        (real / "entry.qcow2").write_bytes(b"x")
        lease = await CacheEntry(real / "entry.qcow2").lease()
        assert lease is not None
        try:
            assert not evictable(other / "entry.qcow2")
            assert (real / "entry.qcow2").exists()
            assert not _child_can_evict(real / "entry.qcow2")
        finally:
            lease.release()
        assert evictable(other / "entry.qcow2")

    @skip_unless_linux
    def test_two_bind_mounts_of_the_folder_are_one_lock(self, tmp_path: Path) -> None:
        """Two bind mounts of one cache folder (realpath keeps both paths): a lease through one,
        an eviction through the other, in one process. The eviction skips the entry. A
        child process in its own user and mount namespace makes the bind mount (no root)."""
        if subprocess.run(["unshare", "-rm", "true"], check=False, capture_output=True).returncode:
            pytest.skip("no unprivileged user namespace on this host")
        real, bind = tmp_path / "cache", tmp_path / "bind"
        real.mkdir()
        bind.mkdir()
        (real / "entry.qcow2").write_bytes(b"x")
        child = subprocess.run(
            [
                *("unshare", "-rm", "sh", "-c", 'mount --bind "$1" "$2" && shift 2 && exec "$@"'),
                *("sh", str(real), str(bind), sys.executable, "-c", _CHILD_LEASES_AND_EVICTS),
                *(str(real / "entry.qcow2"), str(bind / "entry.qcow2")),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert (real / "entry.qcow2").exists()

    async def test_the_loop_never_waits_for_file_work_of_a_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker's eviction is stuck in a slow open of its lock file (a full disk): the
        event loop leases, shares and releases other entries while the open still waits."""
        slow, other = tmp_path / "slow.qcow2", tmp_path / "other.qcow2"
        slow.write_bytes(b"x")
        other.write_bytes(b"x")
        opening = threading.Event()
        go_on = threading.Event()
        opened = threading.Event()
        real_open = Path.open

        def slow_disk(self: Path, *args: object, **kwargs: object) -> object:
            if self == _use_lock(slow):
                opening.set()
                go_on.wait(5)
                opened.set()
            return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

        def evict() -> None:
            with CacheEntry(slow).try_evicting():
                pass

        monkeypatch.setattr(Path, "open", slow_disk)
        worker = asyncio.create_task(asyncio.to_thread(evict))
        try:
            await asyncio.to_thread(opening.wait, 5)
            lease = await CacheEntry(other).lease()
            assert lease is not None
            lease.share().release()
            lease.release()
            assert not opened.is_set()  # all done while the worker's open still waits
        finally:
            go_on.set()
            await worker

    async def test_only_entries_this_process_can_evict_count(self, tmp_path: Path) -> None:
        """Admission counts the space evictions of this process can give back: not an entry
        whose lock file it may not write (another user's), nor any entry of a folder it may
        not write."""
        cache = SizedCache(tmp_path / "l2")
        cache.add("free", 100)
        foreign = cache.add("foreign", 200)
        _use_lock(foreign).touch()
        _use_lock(foreign).chmod(0o444)
        assert await cache.reclaimable_bytes(held=frozenset()) == 100 * 1024 * 1024
        cache.cache_dir.chmod(0o555)
        try:
            assert await cache.reclaimable_bytes(held=frozenset()) == 0
        finally:
            cache.cache_dir.chmod(0o755)

    async def test_new_lock_files_have_no_group_or_other_access(self, tmp_path: Path) -> None:
        """Under `sg qemu-vm` and umask 002, open() would make lock files 0664 of group qemu-vm. A
        process that can open a lock file can hold its lock: a QEMU that runs as qemu-vm could
        stop every build, save and eviction of this user. A new lock file is 0600."""
        cache = SizedCache(tmp_path / "l2")
        built, leased = cache.add("built", 1), cache.add("leased", 1)
        old = os.umask(0o002)
        try:
            async with CacheEntry(built).saving():
                pass
            lease = await CacheEntry(leased).lease()
            assert lease is not None
            lease.release()
            with CacheEntry(leased).try_saving():
                pass
        finally:
            os.umask(old)
        locks = [*CacheEntry(leased).lock_files, CacheEntry(built).lock_files[1]]
        assert [stat.S_IMODE(lock.stat().st_mode) for lock in locks] == [0o600] * 3


_CHILD_LEASES_AND_EVICTS = """
import asyncio, sys
from pathlib import Path
from exec_sandbox.files.locks import CacheEntry

async def main():
    lease = await CacheEntry(Path(sys.argv[1])).lease()
    assert lease is not None
    try:
        with CacheEntry(Path(sys.argv[2])).try_evicting():
            sys.exit("the eviction took the use lock over the lease")
    except BlockingIOError:
        pass
    lease.release()

asyncio.run(main())
"""

_CHILD_EVICTS = """
import fcntl, sys
lock = open(sys.argv[1], "a+b")
try:
    fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)
"""


def _child_sees_a_lease(entry: Path) -> bool:
    """Whether another process sees a lease of this process on `entry`."""
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; from pathlib import Path; from exec_sandbox.files.locks import CacheEntry; "
            "sys.exit(0 if CacheEntry(Path(sys.argv[1])).leased_elsewhere() else 1)",
            str(entry),
        ],
        check=False,
        timeout=30,
    )
    return child.returncode == 0


def _child_can_evict(entry: Path) -> bool:
    """Whether another process could take the use lock of `entry` exclusive now (an eviction)."""
    child = subprocess.run([sys.executable, "-I", "-c", _CHILD_EVICTS, str(_use_lock(entry))], check=False, timeout=30)
    return child.returncode == 0


async def _enter_saving(target: Path) -> None:
    async with CacheEntry(target).saving():
        pass


@contextlib.contextmanager
def _save_lock_held(target: Path) -> Generator[threading.Event]:
    """The save lock of `target` held as another process holds it (flock belongs to an open
    file: a file of its own) until the block ends, or for 5 s at most: a build that ends. The
    event is set when the holder let go on its own."""
    holder = target.with_name(f"{target.name}.lock").open("a+b")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    let_go = threading.Event()

    def build_ends() -> None:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        let_go.set()

    timer = threading.Timer(5.0, build_ends)
    timer.start()
    try:
        yield let_go
    finally:
        timer.cancel()
        timer.join()
        holder.close()


def _flock_waits(monkeypatch: pytest.MonkeyPatch) -> threading.Semaphore:
    """Released once for each waiting flock() (LOCK_EX without LOCK_NB: the wait of
    CacheEntry.saving), just before the call."""
    waits = threading.Semaphore(0)
    real_flock = fcntl.flock

    def flock(fd: int, operation: int) -> None:
        if operation == fcntl.LOCK_EX:
            waits.release()
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", flock)
    return waits


class TestSaving:
    """CacheEntry.saving and try_saving: the save lock of one entry."""

    async def test_yields_true_when_target_missing(self, tmp_path: Path):
        """Should yield True when target file does not exist."""
        target = tmp_path / "missing.txt"
        async with CacheEntry(target).saving() as should_proceed:
            assert should_proceed is True

    async def test_yields_false_when_target_exists(self, tmp_path: Path):
        """Should yield False when target file already exists with content."""
        target = tmp_path / "existing.txt"
        target.write_text("content")
        async with CacheEntry(target).saving() as should_proceed:
            assert should_proceed is False

    async def test_try_lock_raises_when_held(self, tmp_path: Path):
        """try_saving raises BlockingIOError at once while saving holds the lock."""
        target = tmp_path / "locked.txt"
        async with CacheEntry(target).saving():
            with pytest.raises(BlockingIOError), CacheEntry(target).try_saving():
                pass  # Should not reach here

    @pytest.mark.parametrize("wait", [True, False], ids=["saving", "try_saving"])
    async def test_lock_file_is_not_truncated(self, tmp_path: Path, wait: bool):
        """Taking the lock opens the existing lock file without O_TRUNC (open("w") costs
        about 0.4 ms on APFS): bytes already in it stay, the lock is the same."""
        target = tmp_path / "kept.txt"
        lock_file = tmp_path / "kept.txt.lock"
        lock_file.write_bytes(b"kept")

        async with contextlib.AsyncExitStack() as stack:
            if wait:
                should_proceed = await stack.enter_async_context(CacheEntry(target).saving())
            else:
                should_proceed = stack.enter_context(CacheEntry(target).try_saving())
            assert should_proceed is True
            assert lock_file.read_bytes() == b"kept"
            with (
                pytest.raises(BlockingIOError),
                CacheEntry(target).try_saving(),
            ):  # still an exclusive lock on the same file
                pass

        assert lock_file.read_bytes() == b"kept"

    async def test_cancel_during_the_wait_returns_while_another_holds_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another process holds the save lock (a build of the same entry): a caller cancelled
        while it waits returns at once, not once the other build ends. macOS close(2) waits for
        a flock(2) in progress on the same file: a close of the lock file on the event loop would
        stop every task of the loop until the holder lets go."""
        target = tmp_path / "entry.qcow2"
        with _save_lock_held(target) as let_go:
            waiting = _flock_waits(monkeypatch)
            first = asyncio.create_task(_enter_saving(target))
            assert await asyncio.to_thread(waiting.acquire, timeout=5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not let_go.is_set()  # the holder still holds the lock: the cancel did not wait for it
        async with asyncio.timeout(5):
            await _enter_saving(target)  # the lock is free once the holder lets go

    async def test_a_cancelled_wait_keeps_its_place_until_its_flock_returns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At most one thread of this process waits in flock() per path: the next caller of the
        path waits on the in-process lock until the thread of the cancelled caller got the flock
        (then its lock file closes), never in a second thread."""
        target = tmp_path / "entry.qcow2"
        with _save_lock_held(target):
            waiting = _flock_waits(monkeypatch)
            first = asyncio.create_task(_enter_saving(target))
            assert await asyncio.to_thread(waiting.acquire, timeout=5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(_enter_saving(target))
            assert not await asyncio.to_thread(waiting.acquire, timeout=0.2)
        async with asyncio.timeout(5):
            await second
        assert files_locks._async_locks[asyncio.get_running_loop()] == {}

    @pytest.mark.parametrize("n_waiters", [1, 10, 100])
    async def test_blocking_waits_then_proceeds(self, tmp_path: Path, n_waiters: int):
        """Callers that come while the holder saves wait for it, then see the produced file."""
        target = tmp_path / "produced.txt"
        order: list[str] = []
        holding = asyncio.Event()
        all_waiting = asyncio.Event()
        entering = [0]

        async def holder() -> None:
            async with CacheEntry(target).saving() as should_proceed:
                assert should_proceed is True
                order.append("holder-acquired")
                holding.set()
                await all_waiting.wait()  # every waiter now waits on the lock
                target.write_text("done")
                order.append("holder-released")

        async def waiter(idx: int) -> None:
            await holding.wait()
            entering[0] += 1
            if entering[0] == n_waiters:
                all_waiting.set()
            # No await before the lock wait inside saving(): this waiter waits on it in this step
            async with CacheEntry(target).saving() as should_proceed:
                order.append(f"waiter-{idx}-acquired")
                # File was produced by holder, so should_proceed is False
                assert should_proceed is False

        await asyncio.gather(holder(), *(waiter(i) for i in range(n_waiters)))
        assert order[:2] == ["holder-acquired", "holder-released"]
        assert sorted(order[2:]) == sorted(f"waiter-{i}-acquired" for i in range(n_waiters))  # each once, after

"""Local instance store only: races across processes, crashes, cancels, full disks, root checks.

Children are spawned processes (each its own store on the same root), lined up
with a Barrier or an Event: no sleep orders them.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import grp
import hashlib
import multiprocessing
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from exec_sandbox.exceptions import ConfigError, ErrorCode, SandboxError, TransientError
from exec_sandbox.instance import local_store
from exec_sandbox.instance.local_store import LocalInstanceStore
from exec_sandbox.instance.store import Applied, ListedObject, NotApplied

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Barrier, Event

_MAX = 1024 * 1024
_HEAD = "s/i/head"
_CHILDREN = 8
_WAIT = 120  # seconds: an upper bound for a hung child, never an order


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture
async def store(root: Path) -> AsyncGenerator[LocalInstanceStore]:
    async with LocalInstanceStore.open(root=root) as s:
        yield s


def _lock_is_held(root: Path) -> bool:
    fd = os.open(root / ".lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def _signal_lock_requests(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """From now on, the event is set when a thread asks for the store flock and may wait
    for it (a blocking LOCK_EX); the lock itself works as before."""
    asked = threading.Event()
    real = fcntl.flock

    def flock(fd: int, operation: int) -> None:
        if operation == fcntl.LOCK_EX:
            asked.set()
        real(fd, operation)

    monkeypatch.setattr(local_store.fcntl, "flock", flock)
    return asked


def _tree(base: Path) -> dict[str, bytes | None]:
    """Each entry under `base`: the bytes of a file, else None. Symlinks are not followed."""
    return {
        str(path.relative_to(base)): path.read_bytes() if stat.S_ISREG(path.lstat().st_mode) else None
        for folder, dirs, files in base.walk()
        for path in (folder / name for name in (*dirs, *files))
    }


def _add_acl(path: Path, entry: str) -> None:
    """Add a macOS ACL entry (chmod +a)."""
    subprocess.run(["/bin/chmod", "+a", entry, str(path)], check=True)


# ---------------------------------------------------------------- children (spawned processes)


type _Outcome = tuple[bytes, str, str | None]
"""(bytes written, "applied" | "refused" | "maybe", the ETag of an applied write)."""


def _child_put(root: str, data: bytes, etag: str | None, barrier: Barrier, results: Queue[_Outcome]) -> None:
    """One conditional write, started with the other children."""

    async def main() -> _Outcome:
        async with LocalInstanceStore.open(root=Path(root)) as store:
            barrier.wait(_WAIT)
            if etag is None:
                result = await store.put_if_absent(_HEAD, data=data)
            else:
                result = await store.put_if_match(_HEAD, data=data, etag=etag)
        if isinstance(result, Applied):
            return data, "applied", result.etag
        return data, "maybe" if result.maybe_applied else "refused", None

    results.put(asyncio.run(main()))


def _payload(version: int) -> bytes:
    """1 MiB in which every 4-byte word is the version: a torn read mixes two words."""
    return version.to_bytes(4, "big") * (256 * 1024)


def _child_rewrite(root: str, rounds: int, started: Event) -> None:
    """Replace the head `rounds` times, each from the ETag of the last write."""

    async def main() -> None:
        async with LocalInstanceStore.open(root=Path(root)) as store:
            got = await store.get(_HEAD, max_bytes=_MAX)
            assert got is not None
            etag = got[1]
            started.set()
            for version in range(rounds):
                result = await store.put_if_match(_HEAD, data=_payload(version), etag=etag)
                assert isinstance(result, Applied)
                etag = result.etag

    asyncio.run(main())


def _child_hold_lock(root: str, held: Event) -> None:
    """Hold the store flock as a writer does, until killed."""
    fd = os.open(Path(root) / ".lock", os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    held.set()
    threading.Event().wait()


def _race(root: Path, *, etag: str | None) -> list[_Outcome]:
    ctx = multiprocessing.get_context("spawn")
    results: Queue[_Outcome] = ctx.Queue()
    barrier = ctx.Barrier(_CHILDREN)
    children = [
        ctx.Process(target=_child_put, args=(str(root), f"child-{n}".encode(), etag, barrier, results))
        for n in range(_CHILDREN)
    ]
    for child in children:
        child.start()
    try:
        return [results.get(timeout=_WAIT) for _ in children]
    finally:
        for child in children:
            child.join(_WAIT)
            assert child.exitcode == 0


# ---------------------------------------------------------------- races


class TestRaces:
    @pytest.mark.parametrize("condition", ["create-only", "if-match"])
    async def test_processes_one_applies(self, store: LocalInstanceStore, root: Path, condition: str) -> None:
        """Children of other processes race one conditional write: one applies, the others are
        refused, and the head is the winner's bytes with the ETag its write returned."""
        etag: str | None = None
        if condition == "if-match":
            base = await store.put_if_absent(_HEAD, data=b"base")
            assert isinstance(base, Applied)
            etag = base.etag
        outcomes = await asyncio.to_thread(_race, root, etag=etag)
        winners = [(data, applied) for data, outcome, applied in outcomes if outcome == "applied"]
        assert len(winners) == 1
        assert sorted(outcome for _, outcome, _ in outcomes) == ["applied"] + ["refused"] * (_CHILDREN - 1)
        assert await store.get(_HEAD, max_bytes=_MAX) == winners[0]

    @pytest.mark.parametrize("condition", ["create-only", "if-match"])
    async def test_coroutines_one_applies(self, store: LocalInstanceStore, condition: str) -> None:
        """Threads of one process exclude each other too (flock, never fcntl locks)."""
        etag: str | None = None
        if condition == "if-match":
            base = await store.put_if_absent(_HEAD, data=b"base")
            assert isinstance(base, Applied)
            etag = base.etag

        async def write(n: int) -> Applied | NotApplied:
            data = f"writer-{n}".encode()
            if etag is None:
                return await store.put_if_absent(_HEAD, data=data)
            return await store.put_if_match(_HEAD, data=data, etag=etag)

        results = await asyncio.gather(*(write(n) for n in range(20)))
        (winner,) = [n for n, result in enumerate(results) if isinstance(result, Applied)]
        assert results[:winner] + results[winner + 1 :] == [NotApplied(maybe_applied=False)] * 19
        assert await store.get(_HEAD, max_bytes=_MAX) == (f"writer-{winner}".encode(), results[winner].etag)

    async def test_readers_never_see_a_half_written_object(self, store: LocalInstanceStore, root: Path) -> None:
        await store.put_if_absent(_HEAD, data=_payload(0xFFFFFFFF))
        ctx = multiprocessing.get_context("spawn")
        started = ctx.Event()
        writer = ctx.Process(target=_child_rewrite, args=(str(root), 200, started))
        writer.start()
        seen: set[int] = set()
        reads = 0
        try:
            assert await asyncio.to_thread(started.wait, _WAIT)
            while writer.exitcode is None:
                got = await store.get(_HEAD, max_bytes=_MAX)
                assert got is not None
                data, etag = got
                assert hashlib.sha256(data).hexdigest() == etag
                assert data == data[:4] * (256 * 1024), "a read mixed two versions"
                seen.add(int.from_bytes(data[:4], "big"))
                reads += 1
        finally:
            await asyncio.to_thread(writer.join, _WAIT)
        assert writer.exitcode == 0
        assert reads > 0
        assert (await store.get(_HEAD, max_bytes=_MAX) or (b"", ""))[0] == _payload(199)


# ---------------------------------------------------------------- crashes and cancels


class TestCrashAndCancel:
    async def test_killed_writer_frees_the_lock(
        self, store: LocalInstanceStore, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A writer of another process holds the flock and is killed: the waiting write then applies."""
        ctx = multiprocessing.get_context("spawn")
        held = ctx.Event()
        holder = ctx.Process(target=_child_hold_lock, args=(str(root), held))
        holder.start()
        try:
            assert await asyncio.to_thread(held.wait, _WAIT)
            asked = _signal_lock_requests(monkeypatch)
            write = asyncio.create_task(store.put_if_absent(_HEAD, data=b"after"))
            assert await asyncio.to_thread(asked.wait, _WAIT)  # the write asked for the lock
            done, _ = await asyncio.wait({write}, timeout=0.2)
            assert not done, "the write did not wait for the lock"
            assert holder.pid is not None
            os.kill(holder.pid, signal.SIGKILL)
            assert isinstance(await asyncio.wait_for(write, _WAIT), Applied)
        finally:
            holder.kill()
            await asyncio.to_thread(holder.join, _WAIT)
        assert (await store.get(_HEAD, max_bytes=_MAX) or (b"", ""))[0] == b"after"

    async def test_cancel_inside_the_lock_keeps_the_lock(
        self, store: LocalInstanceStore, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancel between compare and rename: the call waits for its job, so the
        lock is never free while the job can still rename (two writes applied)."""
        base = await store.put_if_absent(_HEAD, data=b"base")
        assert isinstance(base, Applied)
        inside, release = threading.Event(), threading.Event()
        rename = os.replace

        def paused(*args: Any, **kwargs: Any) -> None:
            """The first rename waits: the job has compared under the lock and not renamed yet."""
            if not inside.is_set():
                inside.set()
                release.wait(_WAIT)
            rename(*args, **kwargs)

        monkeypatch.setattr(os, "replace", paused)
        first = asyncio.create_task(store.put_if_match(_HEAD, data=b"first", etag=base.etag))
        assert await asyncio.to_thread(inside.wait, _WAIT)
        first.cancel()
        done, _ = await asyncio.wait({first}, timeout=0.2)
        assert not done, "the cancelled call returned while its job holds the lock"
        assert _lock_is_held(root)
        second = asyncio.create_task(store.put_if_match(_HEAD, data=b"second", etag=base.etag))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second == NotApplied(maybe_applied=False)
        assert (await store.get(_HEAD, max_bytes=_MAX) or (b"", ""))[0] == b"first"  # applied whole
        assert not _lock_is_held(root)

    async def test_cancel_before_the_lock_applies_nothing(
        self, store: LocalInstanceStore, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conditional write cancelled while it waits for the flock: nothing is written, no
        staging file stays."""
        fd = os.open(root / ".lock", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # another writer holds the lock
            asked = _signal_lock_requests(monkeypatch)
            write = asyncio.create_task(store.put_if_absent(_HEAD, data=b"never"))
            assert await asyncio.to_thread(asked.wait, _WAIT)  # the write waits for the lock
            assert not write.done()
            write.cancel()
            await asyncio.sleep(0)  # one loop turn: the call takes the cancel and stops its job
        finally:
            os.close(fd)
        with pytest.raises(asyncio.CancelledError):
            await write
        assert await store.get(_HEAD, max_bytes=_MAX) is None
        assert not list((root / ".tmp").iterdir())

    async def test_cancelled_upload_stops_before_the_lock(
        self, store: LocalInstanceStore, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancel that reaches the upload's job while it copies: the job stops before the
        flock, so the call ends while another writer still holds the lock, and nothing is written."""
        (tmp_path / "src").write_bytes(b"body")
        copying, release = threading.Event(), threading.Event()
        clone = local_store.clone_sparse_fd

        def paused(*args: Any, **kwargs: Any) -> None:
            copying.set()
            release.wait(_WAIT)
            clone(*args, **kwargs)

        monkeypatch.setattr(local_store, "clone_sparse_fd", paused)
        fd = os.open(root / ".lock", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # another writer holds the lock
            upload = asyncio.create_task(store.upload("s/i/b/c.p", src=tmp_path / "src"))
            assert await asyncio.to_thread(copying.wait, _WAIT)
            upload.cancel()
            await asyncio.sleep(0)  # one loop turn: the call takes the cancel and stops its job
            release.set()
            done, _ = await asyncio.wait({upload}, timeout=_WAIT)
            assert done, "the cancelled upload waited for the lock"
        finally:
            release.set()
            os.close(fd)
        with pytest.raises(asyncio.CancelledError):
            await upload
        assert not await store.exists("s/i/b/c.p")
        assert not list((root / ".tmp").iterdir())

    @pytest.mark.parametrize("call", ["upload", "delete"])
    async def test_cancel_while_waiting_for_the_lock_changes_nothing(
        self, store: LocalInstanceStore, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: str
    ) -> None:
        """The job checks for a cancel again once it has the flock: an upload or a delete
        cancelled while another writer holds the lock leaves the key as it was."""
        (tmp_path / "src").write_bytes(b"new")
        written = await store.put_if_absent("s/i/b/c.p", data=b"old")
        assert isinstance(written, Applied)
        fd = os.open(root / ".lock", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # another writer holds the lock
            asked = _signal_lock_requests(monkeypatch)
            if call == "upload":
                job = asyncio.create_task(store.upload("s/i/b/c.p", src=tmp_path / "src"))
            else:
                job = asyncio.create_task(store.delete_many(["s/i/b/c.p"]))
            assert await asyncio.to_thread(asked.wait, _WAIT)  # the job waits for the lock
            job.cancel()
            await asyncio.sleep(0)  # one loop turn: the call takes the cancel and stops its job
        finally:
            os.close(fd)
        with pytest.raises(asyncio.CancelledError):
            await job
        assert await store.get("s/i/b/c.p", max_bytes=_MAX) == (b"old", written.etag)
        assert not list((root / ".tmp").iterdir())
        assert not _lock_is_held(root)

    async def test_cancelled_download_leaves_no_file(
        self, store: LocalInstanceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cancel comes while the copy job runs; the job ends after it (a clone has no stop
        point): the file it made is removed."""
        (tmp_path / "src").write_bytes(os.urandom(1000))
        await store.upload("k", src=tmp_path / "src")
        dest = tmp_path / "dest"
        started, go_on = threading.Event(), threading.Event()
        clone = local_store.clone_sparse_fd

        def held_clone(*args: Any, **kwargs: Any) -> bool:
            started.set()
            go_on.wait(_WAIT)
            return clone(*args, **kwargs)

        monkeypatch.setattr(local_store, "clone_sparse_fd", held_clone)
        task = asyncio.create_task(store.download("k", dest=dest, max_bytes=_MAX, expected_size=None))
        assert await asyncio.to_thread(started.wait, _WAIT)
        task.cancel()
        go_on.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not dest.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["src", "store"]


# ---------------------------------------------------------------- disk failures


def _fail_flush(monkeypatch: pytest.MonkeyPatch, *, kind: Callable[[int], bool], error: int) -> None:
    """From now on, the OS call that writes a file or folder through to the disk (fsync;
    fcntl F_FULLFSYNC on macOS) fails with `error` when `kind(st_mode)` is true."""
    real_fsync, real_fcntl = os.fsync, fcntl.fcntl
    full_fsync = getattr(fcntl, "F_FULLFSYNC", None)

    def check(fd: int) -> None:
        if kind(os.fstat(fd).st_mode):
            raise OSError(error, os.strerror(error))

    def fsync(fd: int) -> None:
        check(fd)
        real_fsync(fd)

    def fcntl_call(fd: int, cmd: int, arg: Any = 0) -> Any:
        if full_fsync is not None and cmd == full_fsync:
            check(fd)
        return real_fcntl(fd, cmd, arg)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(fcntl, "fcntl", fcntl_call)


class TestDiskFailures:
    async def test_full_disk_before_rename_applies_nothing(
        self, store: LocalInstanceStore, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ENOSPC before the rename: an error (the session stays usable), never "maybe applied"."""
        base = await store.put_if_absent(_HEAD, data=b"base")
        assert isinstance(base, Applied)
        _fail_flush(monkeypatch, kind=stat.S_ISREG, error=errno.ENOSPC)
        with pytest.raises(
            TransientError, match="ENOSPC", check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE
        ) as exc:
            await store.put_if_match(_HEAD, data=b"new", etag=base.etag)
        assert exc.value.action == "RETRY"
        with pytest.raises(
            TransientError, match="ENOSPC", check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE
        ):
            await store.put_if_absent("s/j/head", data=b"new")
        assert await store.get(_HEAD, max_bytes=_MAX) == (b"base", base.etag)
        assert await store.get("s/j/head", max_bytes=_MAX) is None
        assert not list((root / ".tmp").iterdir())

    async def test_flush_failure_after_rename_may_have_applied(
        self, store: LocalInstanceStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = await store.put_if_absent(_HEAD, data=b"base")
        assert isinstance(base, Applied)
        _fail_flush(monkeypatch, kind=stat.S_ISDIR, error=errno.EIO)
        assert await store.put_if_match(_HEAD, data=b"new", etag=base.etag) == NotApplied(maybe_applied=True)
        assert (await store.get(_HEAD, max_bytes=_MAX) or (b"", ""))[0] == b"new"

    async def test_full_disk_during_upload_keeps_the_key(
        self, store: LocalInstanceStore, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "src").write_bytes(b"body")
        _fail_flush(monkeypatch, kind=stat.S_ISREG, error=errno.ENOSPC)
        with pytest.raises(
            TransientError, match="ENOSPC", check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE
        ):
            await store.upload("s/i/b/c.p", src=tmp_path / "src")
        assert not await store.exists("s/i/b/c.p")
        assert not list((root / ".tmp").iterdir())

    async def test_read_only_store_is_config_error(
        self, store: LocalInstanceStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fail_flush(monkeypatch, kind=stat.S_ISREG, error=errno.EROFS)
        with pytest.raises(ConfigError, match="EROFS", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG) as exc:
            await store.put_if_absent(_HEAD, data=b"x")
        assert exc.value.action == "FIX_CONFIG"

    async def test_removed_root_is_config_error(self, store: LocalInstanceStore, root: Path) -> None:
        await store.put_if_absent(_HEAD, data=b"x")
        (root / "s" / "i" / "head").unlink()
        for folder in (root / "s" / "i", root / "s", root / ".tmp"):
            folder.rmdir()
        (root / ".lock").unlink()
        root.rmdir()
        with pytest.raises(ConfigError, match="is gone", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            await store.get(_HEAD, max_bytes=_MAX)
        with pytest.raises(ConfigError, match="is gone", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            await store.put_if_absent(_HEAD, data=b"x")
        with pytest.raises(ConfigError, match="is gone", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            await store.list_objects("")


# ---------------------------------------------------------------- open: root checks, staging


class TestOpen:
    async def test_missing_parents_under_a_group_write_umask(self, tmp_path: Path) -> None:
        """umask 002 (Ubuntu's default for users): the folders made above a new store are 0700
        too, so the store's own check of the folders above it accepts them."""
        old_umask = os.umask(0o002)
        try:
            async with LocalInstanceStore.open(root=tmp_path / "home" / ".local" / "share" / "store"):
                pass
        finally:
            os.umask(old_umask)
        assert stat.S_IMODE((tmp_path / "home" / ".local").stat().st_mode) & 0o777 == 0o700

    async def test_missing_root_is_made_private(self, tmp_path: Path) -> None:
        root = tmp_path / "a" / "b" / "store"
        async with LocalInstanceStore.open(root=root):
            pass
        assert stat.S_IMODE(root.stat().st_mode) & 0o777 == 0o700
        assert stat.S_IMODE((root / ".lock").stat().st_mode) == 0o600
        assert stat.S_IMODE((root / ".tmp").stat().st_mode) & 0o777 == 0o700

    async def test_symlink_root_is_refused(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir(mode=0o700)
        (tmp_path / "link").symlink_to(real)
        with pytest.raises(ConfigError, match="symlink", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            async with LocalInstanceStore.open(root=tmp_path / "link"):
                pass

    async def test_file_root_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "file").write_bytes(b"")
        with pytest.raises(ConfigError, match="cannot be made", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            async with LocalInstanceStore.open(root=tmp_path / "file"):
                pass

    @pytest.mark.parametrize(
        ("code", "error", "action"),
        [
            (errno.ENOSPC, (TransientError, ErrorCode.INSTANCE_STORE_UNAVAILABLE), "RETRY"),
            (errno.EIO, (TransientError, ErrorCode.INSTANCE_STORE_UNAVAILABLE), "RETRY"),
            (errno.EACCES, (ConfigError, ErrorCode.INSTANCE_CONFIG), "FIX_CONFIG"),
        ],
        ids=["disk-full", "io", "no-access"],
    )
    async def test_root_that_cannot_be_made(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        code: int,
        error: tuple[type[SandboxError], ErrorCode],
        action: str,
    ) -> None:
        """A full disk or an I/O error while the root is made can pass later; no access cannot."""

        def mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
            raise OSError(code, os.strerror(code), str(self))

        monkeypatch.setattr(Path, "mkdir", mkdir)
        with pytest.raises(error[0], match="cannot be made or read", check=lambda e: e.code is error[1]) as exc:
            async with LocalInstanceStore.open(root=tmp_path / "store"):
                pass
        assert exc.value.action == action

    @pytest.mark.parametrize("mode", [0o750, 0o705, 0o740])
    async def test_open_root_is_refused(self, tmp_path: Path, mode: int) -> None:
        root = tmp_path / "store"
        root.mkdir()
        root.chmod(mode)
        with pytest.raises(
            ConfigError, match="group or others", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ) as exc:
            async with LocalInstanceStore.open(root=root):
                pass
        assert f"chmod 700 {root}" in str(exc.value.hint)

    async def test_root_of_another_user_names_the_sudo_fix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        uid = os.geteuid()
        monkeypatch.setattr(os, "geteuid", lambda: uid + 1)
        with pytest.raises(
            ConfigError, match=f"belongs to uid {uid}", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ) as exc:
            async with LocalInstanceStore.open(root=tmp_path / "store"):
                pass
        assert "sudo chown -R" in str(exc.value.hint)

    @pytest.mark.skipif(
        sys.platform != "darwin", reason="chmod +a sets macOS ACLs (Linux: setfacl, on a disk with ACLs)"
    )
    @pytest.mark.parametrize("on", ["root", "inherited"])
    async def test_root_with_acl_is_refused(self, tmp_path: Path, on: str) -> None:
        """Mode bits do not show ACL entries: a 0700 root with an "everyone" entry is open to all.
        "inherited": the parent's inheritable entry reaches the root that open() makes."""
        parent = tmp_path / "parent"
        parent.mkdir()
        root = parent / "store"
        if on == "root":
            root.mkdir(mode=0o700)
            _add_acl(root, "everyone allow list,search,add_file")
        else:
            _add_acl(parent, "everyone allow list,search,add_file,file_inherit,directory_inherit")
        with pytest.raises(
            ConfigError, match="has ACL entries", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ) as exc:
            async with LocalInstanceStore.open(root=root):
                pass
        assert f"chmod -R -N {root}" in str(exc.value.hint)
        assert sorted(p.name for p in root.iterdir()) == []  # refused before .lock and .tmp

    @pytest.mark.parametrize("mode", [0o777, 0o770, 0o707], ids=oct)
    async def test_parent_that_others_can_change_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: int
    ) -> None:
        """Another user who can write a parent folder can put another folder in place of the root
        (OpenSSH StrictModes): every folder above the root must be writable only by root or this
        user. The group database lists another member in the folder's group here, so it is not
        this user's private group (that one passes on Linux; macOS never reads the database)."""

        def shared_group(gid: int) -> grp.struct_group:
            return grp.struct_group(("shared", "x", gid, ["another-user"]))

        monkeypatch.setattr(grp, "getgrgid", shared_group)
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(mode)
        root = shared / "a" / "store"
        with pytest.raises(
            ConfigError, match="which another user can change", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ) as exc:
            async with LocalInstanceStore.open(root=root):
                pass
        assert f"chmod go-w {shared.resolve()}" in str(exc.value.hint)
        assert f"chmod +t {shared.resolve()}" in str(exc.value.hint)  # a shared volume: the sticky bit passes
        assert sorted(p.name for p in root.iterdir()) == []

    async def test_sticky_parent_is_accepted(self, tmp_path: Path) -> None:
        """As /tmp: other users can add names there, but not rename or remove the names of this user."""
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o1777)
        async with LocalInstanceStore.open(root=shared / "store") as store:
            assert isinstance(await store.put_if_absent(_HEAD, data=b"x"), Applied)

    async def test_shared_flock_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Linux NFS turns flock into a POSIX lock: one process gets it twice."""
        monkeypatch.setattr(local_store.fcntl, "flock", lambda fd, operation: None)
        with pytest.raises(ConfigError, match="flock", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            async with LocalInstanceStore.open(root=tmp_path / "store"):
                pass

    async def test_old_staging_files_are_removed(self, root: Path) -> None:
        async with LocalInstanceStore.open(root=root):
            pass
        old, new = root / ".tmp" / "old.part", root / ".tmp" / "new.part"
        old.write_bytes(b"dead writer")
        new.write_bytes(b"live writer")
        two_days_ago = time.time() - 2 * 24 * 3600
        os.utime(old, (two_days_ago, two_days_ago))
        async with LocalInstanceStore.open(root=root):
            pass
        assert not old.exists()
        assert new.exists()

    async def test_upload_of_an_old_file_is_not_stale(
        self, store: LocalInstanceStore, root: Path, tmp_path: Path
    ) -> None:
        """clonefile keeps the mtime of the source: the staging file gets a new one,
        else an open() in another process could remove it as a dead writer's."""
        src = tmp_path / "src"
        src.write_bytes(b"old image")
        os.utime(src, (0, 0))
        await store.upload("assets/sha256/x", src=src)
        assert (root / "assets" / "sha256" / "x").stat().st_mtime > time.time() - 3600


# ---------------------------------------------------------------- keys and modes


class TestKeys:
    @pytest.mark.parametrize(
        "key",
        ["../x", "s/../x", "/abs", "s//x", "s/x/", "", ".lock", ".tmp/x", "s/.x", "S/x", "a/./b", "a\\b", "a" * 65],
    )
    async def test_bad_keys_are_refused(self, store: LocalInstanceStore, tmp_path: Path, key: str) -> None:
        (tmp_path / "src").write_bytes(b"x")
        calls = [
            lambda: store.get(key, max_bytes=_MAX),
            lambda: store.size(key),
            lambda: store.put_if_absent(key, data=b"x"),
            lambda: store.put_if_match(key, data=b"x", etag="0" * 64),
            lambda: store.upload(key, src=tmp_path / "src"),
            lambda: store.download(key, dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None),
            lambda: store.delete_many(["s/ok", key]),
        ]
        for call in calls:
            with pytest.raises(
                ConfigError, match="not a key of the local store", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
            ):
                await call()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["src", "store"]

    async def test_longest_instance_id_is_a_key_segment(self, store: LocalInstanceStore) -> None:
        """An instance id is 1 to 64 characters (ids.py): the longest one names a head and a listing."""
        longest = "a" * 64
        written = await store.put_if_absent(f"s/{longest}/head", data=b"x")
        assert isinstance(written, Applied)
        assert [name async for name in store.list_children("s/")] == [longest]

    @pytest.mark.parametrize("prefix", ["s", "../", "/", "s//", ".tmp/", "s/../"])
    async def test_bad_prefixes_are_refused(self, store: LocalInstanceStore, prefix: str) -> None:
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            await store.list_objects(prefix)
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            [name async for name in store.list_children(prefix)]

    async def test_symlink_at_a_key_is_refused(self, store: LocalInstanceStore, root: Path, tmp_path: Path) -> None:
        (tmp_path / "secret").write_bytes(b"secret")
        (root / "s").mkdir(mode=0o700)
        (root / "s" / "head").symlink_to(tmp_path / "secret")
        with pytest.raises(ConfigError, match="ELOOP", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            await store.get("s/head", max_bytes=_MAX)

    @pytest.mark.parametrize("kind", ["fifo", "symlink"])
    async def test_special_file_at_a_key_is_refused(
        self, store: LocalInstanceStore, root: Path, tmp_path: Path, kind: str
    ) -> None:
        """A FIFO or a symlink planted at a key is never taken for an object, nor for a
        missing key: the error names the file to remove (FIX_CONFIG). A get of a symlink is
        ELOOP (test_symlink_at_a_key_is_refused)."""
        (tmp_path / "outside").write_bytes(b"outside")
        if kind == "fifo":
            os.mkfifo(root / "a", 0o600)
        else:
            (root / "a").symlink_to(tmp_path / "outside")
        calls = [lambda: store.size("a")]
        if kind == "fifo":
            calls += [
                lambda: store.get("a", max_bytes=_MAX),
                lambda: store.download("a", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None),
            ]
        for call in calls:
            with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG) as exc:
                await call()
            assert str(exc.value) == f"Instance store file {root}/a is not a regular file."
            assert exc.value.hint == f"Remove it: {root} holds only the files and folders of the store."
            assert exc.value.action == "FIX_CONFIG"
        assert not (tmp_path / "dest").exists()
        assert (tmp_path / "outside").read_bytes() == b"outside"

    async def test_object_where_a_folder_must_be_is_refused(
        self, store: LocalInstanceStore, root: Path, tmp_path: Path
    ) -> None:
        """A write of "a/b" while "a" is an object (a file, not a folder) applies nothing and
        fails the same way again (FIX_CONFIG). The S3 store has no folders and accepts it;
        the store.py helpers never make such keys."""
        (tmp_path / "src").write_bytes(b"body")
        written = await store.put_if_absent("a", data=b"x")
        assert isinstance(written, Applied)
        calls = [lambda: store.put_if_absent("a/b", data=b"y"), lambda: store.upload("a/b", src=tmp_path / "src")]
        for call in calls:
            with pytest.raises(
                ConfigError, match=r"PUT a/b failed: ENOTDIR", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
            ) as exc:
                await call()
            assert exc.value.action == "FIX_CONFIG"  # the same call fails the same way again
        assert await store.get("a", max_bytes=_MAX) == (b"x", written.etag)
        assert not list((root / ".tmp").iterdir())

    async def test_modes(self, store: LocalInstanceStore, root: Path, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.write_bytes(b"body")
        src.chmod(0o644)
        await store.put_if_absent(_HEAD, data=b"head")
        await store.upload("s/i/b/c.p", src=src)
        for folder in (root, root / "s", root / "s" / "i", root / "s" / "i" / "b"):
            assert stat.S_IMODE(folder.stat().st_mode) & 0o777 == 0o700, folder
        for file in (root / "s" / "i" / "head", root / "s" / "i" / "b" / "c.p"):
            assert stat.S_IMODE(file.stat().st_mode) == 0o600, file

    async def test_delete_removes_empty_folders(self, store: LocalInstanceStore, root: Path, tmp_path: Path) -> None:
        (tmp_path / "src").write_bytes(b"body")
        await store.put_if_absent(_HEAD, data=b"head")
        await store.upload("s/i/b/c.p", src=tmp_path / "src")
        await store.put_if_absent("s/j/head", data=b"other")
        await store.delete_many(["s/i/b/c.p", _HEAD])
        assert sorted(p.name for p in (root / "s").iterdir()) == ["j"]
        await store.delete_many(["s/j/head"])
        assert sorted(p.name for p in root.iterdir()) == [".lock", ".tmp"]


# ---------------------------------------------------------------- symlinks: never followed


class TestSymlinks:
    async def test_root_swapped_for_a_symlink_after_open(self, tmp_path: Path) -> None:
        """Whoever can rename in the parent of the root swaps the root for a symlink after
        open (here this user plays them). Each call starts from the folder that open()
        holds, never from the path: no file outside the store is read, written or deleted."""
        parent = tmp_path / "parent"
        parent.mkdir(mode=0o700)  # writable by this user only, whatever the umask: open() checks it
        root = parent / "store"
        victim = tmp_path / "victim"
        (victim / "b").mkdir(parents=True)
        (victim / "head").write_bytes(b"victim head")
        (victim / "b" / "c.p").write_bytes(b"victim body")
        evil = tmp_path / "evil"
        (evil / "s").mkdir(parents=True)
        (evil / ".tmp").mkdir()
        (evil / ".lock").touch()
        (evil / "s" / "i").symlink_to(victim)
        outside = _tree(victim), _tree(evil)
        (tmp_path / "src").write_bytes(b"new body")
        async with LocalInstanceStore.open(root=root) as store:
            mine = await store.put_if_absent("s/mine/head", data=b"mine")
            assert isinstance(mine, Applied)
            root.rename(parent / "moved")
            root.symlink_to(evil)
            with pytest.raises(ConfigError, match="is gone", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
                await store.get(_HEAD, max_bytes=_MAX)
            with pytest.raises(ConfigError, match="is gone", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
                await store.list_objects("s/i/")
            with pytest.raises(ConfigError, match="is gone", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
                await store.put_if_absent(_HEAD, data=b"x")
            await store.delete_many([_HEAD, "s/i/b/c.p"])  # not in the folder held open: nothing to delete
            await store.upload("s/i/b/c.p", src=tmp_path / "src")
            assert await store.get("s/mine/head", max_bytes=_MAX) == (b"mine", mine.etag)
        assert (_tree(victim), _tree(evil)) == outside
        assert (parent / "moved" / "s" / "i" / "b" / "c.p").read_bytes() == b"new body"

    @pytest.mark.parametrize("link", ["s", "s/i"])
    async def test_symlink_folder_in_the_root_is_refused(
        self, store: LocalInstanceStore, root: Path, tmp_path: Path, link: str
    ) -> None:
        """A symlink planted for a folder of keys: each call through it is refused (ELOOP);
        a listing skips it. Nothing outside the store is read, written or deleted."""
        target = tmp_path / "outside"
        for key, data in {"s/i/head": b"outside head", "s/i/b/c.p": b"outside body"}.items():
            path = target.joinpath(*key.split("/")[link.count("/") + 1 :])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (root / link).parent.mkdir(mode=0o700, exist_ok=True)
        (root / link).symlink_to(target)
        outside = _tree(target)
        (tmp_path / "src").write_bytes(b"new")
        calls = [
            lambda: store.get("s/i/head", max_bytes=_MAX),
            lambda: store.size("s/i/head"),
            lambda: store.put_if_absent("s/i/new", data=b"x"),
            lambda: store.put_if_match("s/i/head", data=b"x", etag=hashlib.sha256(b"outside head").hexdigest()),
            lambda: store.upload("s/i/b/c.p", src=tmp_path / "src"),
            lambda: store.download("s/i/b/c.p", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None),
            lambda: store.list_objects("s/i/"),
            lambda: store.delete_many(["s/i/head", "s/i/b/c.p"]),
        ]
        for call in calls:
            with pytest.raises(ConfigError, match="ELOOP", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
                await call()
        assert await store.list_objects("") == []
        assert [name async for name in store.list_children("")] == []
        assert _tree(target) == outside
        assert not (tmp_path / "dest").exists()
        assert not list((root / ".tmp").iterdir())

    @pytest.mark.parametrize("name", [".tmp", ".lock"])
    async def test_symlink_for_a_file_of_the_store_is_refused(
        self, store: LocalInstanceStore, root: Path, tmp_path: Path, name: str
    ) -> None:
        target = tmp_path / "outside"
        target.mkdir()
        if name == ".lock":
            (root / name).unlink()
        else:
            (root / name).rmdir()
        (root / name).symlink_to(target)
        with pytest.raises(ConfigError, match="ELOOP", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            await store.put_if_absent(_HEAD, data=b"x")
        assert not list(target.iterdir())


# ---------------------------------------------------------------- listing ETags


class TestListingEtags:
    async def test_listing_never_opens_an_object(
        self, store: LocalInstanceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ETag of an upload is the stat stamp of its file after the rename onto the
        key: a listing gives the same one from a stat, without opening (reading) the object.
        (The store opens every name with os.open; the ETag match itself is in the contract tests.)"""
        (tmp_path / "src").write_bytes(os.urandom(1000))
        etag = await store.upload("s/i/b/c.f", src=tmp_path / "src")
        opened: list[str] = []
        real_open = os.open

        def recording_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            opened.append(os.path.basename(os.fsdecode(path)))  # noqa: PTH119
            return real_open(path, flags, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(os, "open", recording_open)
            listed = await store.list_objects("s/i/b/")
        assert listed == [ListedObject(key="s/i/b/c.f", size=1000, etag=etag)]
        assert "c.f" not in opened

    async def test_same_bytes_uploaded_again_give_a_new_etag(self, store: LocalInstanceStore, tmp_path: Path) -> None:
        """Each upload renames a new file onto the key: a new ETag, also for the same bytes."""
        (tmp_path / "src").write_bytes(b"image")
        first = await store.upload("assets/sha256/x", src=tmp_path / "src")
        second = await store.upload("assets/sha256/x", src=tmp_path / "src")
        assert second != first
        assert await store.list_objects("assets/") == [ListedObject(key="assets/sha256/x", size=5, etag=second)]

    @pytest.mark.parametrize("change", ["touch", "replaced"])
    async def test_a_changed_file_gets_a_new_etag(
        self, store: LocalInstanceStore, root: Path, tmp_path: Path, change: str
    ) -> None:
        """A write outside the store (same size): the listing no longer gives the upload's
        ETag, so the next save writes a full copy."""
        (tmp_path / "src").write_bytes(b"body")
        etag = await store.upload("s/i/b/c.f", src=tmp_path / "src")
        stored = root / "s" / "i" / "b" / "c.f"
        if change == "touch":
            os.utime(stored, ns=(0, 0))
        else:
            other = stored.with_name("other")
            other.write_bytes(b"BODY")
            other.replace(stored)
        (listed,) = await store.list_objects("s/i/b/")
        assert listed.size == 4
        assert listed.etag != etag

    async def test_head_etag_is_the_sha256_of_its_bytes(self, store: LocalInstanceStore, root: Path) -> None:
        """A head keeps a content ETag (a stat stamp can come back for other bytes and break
        the compare-and-set): a touch of the head file changes nothing for If-Match."""
        written = await store.put_if_absent(_HEAD, data=b"head")
        assert written == Applied(
            etag="9f2e6d33a3717ee826353a404ba4618d1aeeb6879ad7936bce8ed5f46814924d"
        )  # SHA-256 of b"head"
        os.utime(root / "s" / "i" / "head", ns=(0, 0))
        assert isinstance(await store.put_if_match(_HEAD, data=b"next", etag=written.etag), Applied)

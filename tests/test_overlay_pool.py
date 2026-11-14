"""Tests for OverlayPool.

Test philosophy:
- Unit tests: the public calls (start, acquire, stop) on a fake qemu-storage-daemon
  (_FakeDaemon): the files in the pool folder and at the targets are the facts to check
- Error tests: mock only to simulate failures that can't be triggered otherwise
- Integration tests: Real qemu-img, real files, real code paths
"""

import asyncio
import errno
import os
import shutil
import stat
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox import constants
from exec_sandbox.exceptions import ErrorCode, TransientError
from exec_sandbox.models import Language
from exec_sandbox.overlay_pool import OverlayPool
from exec_sandbox.vm_spec import VmSpec

_BASE = Path("/fake/base.qcow2")
"""The base image of the unit tests: the fake daemon never reads it."""


def _owner_alive(alive: bool) -> Any:
    """psutil.pid_exists of the overlay pool module answers `alive` for every owner."""
    return patch("exec_sandbox.overlay_pool.psutil", SimpleNamespace(pid_exists=lambda _pid: alive))


class _FakeDaemon:
    """qemu-storage-daemon stand-in (no QEMU process). create_overlay writes the overlay path
    into the new file: a target that names a path in the pool folder was moved from the pool,
    a target that names itself was made on demand. As the real daemon, a creation that fails
    raises TransientError OVERLAY_FAILED."""

    def __init__(self) -> None:
        self.started = False
        self.stops = 0
        self.failures: list[Exception] = []
        """Raised by the next creations, in order."""
        self.empty_successes = 0
        """The next creations return with no file made."""

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False
        self.stops += 1

    async def create_overlay(self, base_image: Path, overlay_path: Path) -> None:
        del base_image
        if self.failures:
            raise self.failures.pop(0)
        if self.empty_successes:
            self.empty_successes -= 1
            return
        try:
            overlay_path.write_text(str(overlay_path))
        except OSError as e:
            raise TransientError(
                f"overlay {overlay_path}: {e}", code=ErrorCode.OVERLAY_FAILED, context=None, hint=None, instance_id=None
            ) from e


@pytest.fixture
def temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The temp folder of the process for this test (tempfile.gettempdir()): start() sweeps the
    stale pool folders there and makes its default pool folder there."""
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    return temp


@pytest.fixture
def fake_daemon(temp_dir: Path) -> Iterator[_FakeDaemon]:
    """The daemon that each OverlayPool.start() of the test starts (in an empty temp folder)."""
    del temp_dir
    daemon = _FakeDaemon()
    with patch("exec_sandbox.overlay_pool.QemuStorageDaemon", return_value=daemon):
        yield daemon


def _overlays(pool_dir: Path) -> set[Path]:
    return set(pool_dir.glob("*.qcow2"))


async def _wait_overlays(pool_dir: Path, count: int) -> None:
    """Until the pool folder holds `count` overlays. With the fake daemon, an overlay file is
    queued in the same step that writes it: a file in the pool folder is a queued overlay."""
    async with asyncio.timeout(5):
        while len(_overlays(pool_dir)) < count:  # noqa: ASYNC110
            await asyncio.sleep(0.005)


@pytest.fixture
async def start_pool(
    fake_daemon: _FakeDaemon, tmp_path: Path
) -> AsyncIterator[Callable[[int], Awaitable[OverlayPool]]]:
    """Start a pool of `size` overlays of _BASE in tmp_path/pool, and wait until it is full.
    The pool does not fill again (replenish interval of one hour): each acquire takes an
    overlay for good. Stopped at teardown."""
    del fake_daemon
    started: list[OverlayPool] = []

    async def start(size: int) -> OverlayPool:
        pool = OverlayPool(pool_size=size, pool_dir=tmp_path / "pool")
        started.append(pool)
        await pool.start([_BASE])
        await _wait_overlays(tmp_path / "pool", size)
        return pool

    with patch.object(constants, "OVERLAY_POOL_REPLENISH_INTERVAL_SECONDS", 3600):
        try:
            yield start
        finally:
            for pool in started:
                await pool.stop()


# ============================================================================
# Unit Tests - lifecycle and pool folder
# ============================================================================


class TestOverlayPoolPureLogic:
    """start(), stop() and acquire() on the fake daemon."""

    async def test_acquire_before_start_fails(self, tmp_path: Path) -> None:
        """Acquire before start raises RuntimeError - daemon required."""
        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")
        # No start() called - daemon not started
        target = tmp_path / "target.qcow2"

        with pytest.raises(RuntimeError, match="Daemon must be started"):
            await pool.acquire(Path("/fake/base.qcow2"), target)

    @pytest.mark.parametrize("size", [0, -5])
    async def test_pool_size_zero_or_less_makes_overlays_on_demand_only(
        self, fake_daemon: _FakeDaemon, tmp_path: Path, size: int
    ) -> None:
        """start() starts the daemon (acquire needs it), but makes no pool folder and no
        overlay: acquire makes each overlay on demand at its target."""
        del fake_daemon
        pool_dir = tmp_path / "pool"
        pool = OverlayPool(pool_size=size, pool_dir=pool_dir)
        await pool.start([_BASE])
        try:
            assert pool.daemon_enabled is True
            target = tmp_path / "target.qcow2"
            assert await pool.acquire(_BASE, target) is False
            assert target.read_text() == str(target)
            assert not pool_dir.exists()
        finally:
            await pool.stop()

    async def test_empty_base_images_list(self, tmp_path: Path) -> None:
        """start() with no base image makes the pool folder, and no overlay in it."""
        pool_dir = tmp_path / "pool"
        pool = OverlayPool(pool_size=5, pool_dir=pool_dir)
        await pool.start([])
        try:
            assert pool.daemon_enabled is True
            assert list(pool_dir.iterdir()) == []
        finally:
            await pool.stop()

    async def test_pool_folder_is_private_whatever_the_umask(self, tmp_path: Path) -> None:
        """Made 0700 under umask 002 (the Ubuntu default) and any group (sg qemu-vm): a group
        that can write it could swap the overlays of other VMs."""
        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "tmp" / "pool")
        old = os.umask(0o002)
        try:
            await pool.start([])
            modes = [stat.S_IMODE(p.stat().st_mode) & 0o777 for p in (tmp_path / "tmp", tmp_path / "tmp" / "pool")]
        finally:
            os.umask(old)
            await pool.stop()
        assert modes == [0o700, 0o700]

    async def test_double_stop_is_safe(self, tmp_path: Path) -> None:
        """Calling stop() twice doesn't error (idempotent)."""
        pool = OverlayPool(pool_size=0, pool_dir=tmp_path / "pool")
        await pool.start([])

        await pool.stop()
        await pool.stop()  # Should not raise

    async def test_double_start_raises_error(self, tmp_path: Path) -> None:
        """Calling start() twice without stop() raises RuntimeError."""
        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")
        await pool.start([])

        with pytest.raises(RuntimeError, match="already started"):
            await pool.start([])

        await pool.stop()

    async def test_start_restart_after_stop(self, fake_daemon: _FakeDaemon, tmp_path: Path) -> None:
        """After stop(), start() runs the pool again: the daemon starts again and the pool
        fills again (stop() set the shutdown event, start() clears it)."""
        pool_dir = tmp_path / "pool"
        pool = OverlayPool(pool_size=1, pool_dir=pool_dir)
        for attempt in range(2):
            await pool.start([_BASE])
            try:
                assert pool.daemon_enabled is True
                await _wait_overlays(pool_dir, 1)
                assert await pool.acquire(_BASE, tmp_path / f"target-{attempt}.qcow2") is True
            finally:
                await pool.stop()
            assert pool.daemon_enabled is False
        assert fake_daemon.stops == 2

    async def test_acquire_existing_target_raises_error(self, tmp_path: Path) -> None:
        """Acquire raises FileExistsError if target_path already exists."""
        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")
        target = tmp_path / "existing.qcow2"
        target.write_text("existing content")

        with pytest.raises(FileExistsError, match="already exists"):
            await pool.acquire(Path("/fake/base.qcow2"), target)

    async def test_default_pool_dir_is_new_and_private_at_each_start(
        self, fake_daemon: _FakeDaemon, temp_dir: Path
    ) -> None:
        """A name no other user can make first in the shared temp folder (mkdtemp, 0700); the
        pid stays in the name for the stale sweep. stop() removes the folder."""
        del fake_daemon
        pool = OverlayPool(pool_size=5)
        names: list[str] = []
        for _ in range(2):
            await pool.start([])
            try:
                (made,) = temp_dir.iterdir()
                names.append(made.name)
                assert made.name.startswith(f"exec-sandbox-overlay-pool-{os.getpid()}-")
                assert stat.S_IMODE(made.stat().st_mode) == 0o700
            finally:
                await pool.stop()
            assert list(temp_dir.iterdir()) == []
        assert names[0] != names[1]

    async def test_given_pool_dir_is_used_as_is(self, fake_daemon: _FakeDaemon, temp_dir: Path, tmp_path: Path) -> None:
        """A given pool_dir holds the overlays: no folder is made in the temp folder."""
        del fake_daemon
        custom = tmp_path / "custom"
        pool = OverlayPool(pool_size=1, pool_dir=custom)
        await pool.start([_BASE])
        try:
            await _wait_overlays(custom, 1)
            assert list(temp_dir.iterdir()) == []
        finally:
            await pool.stop()

    async def test_mkdir_permission_error_disables_pool_precreation(
        self, fake_daemon: _FakeDaemon, tmp_path: Path
    ) -> None:
        """The pool folder cannot be made: start() still starts the daemon, makes no overlay,
        and acquire makes each overlay on demand."""
        del fake_daemon
        pool_dir = tmp_path / "pool"
        pool = OverlayPool(pool_size=5, pool_dir=pool_dir)
        real_mkdir = Path.mkdir

        def mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
            if path == pool_dir:
                raise PermissionError(errno.EACCES, "Access denied")
            real_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", autospec=True, side_effect=mkdir):
            await pool.start([_BASE])
        try:
            assert pool.daemon_enabled is True
            assert not pool_dir.exists()
            target = tmp_path / "target.qcow2"
            assert await pool.acquire(_BASE, target) is False
            assert target.read_text() == str(target)
        finally:
            await pool.stop()


# ============================================================================
# Stale Pool Directory Cleanup Tests
# ============================================================================


class TestOverlayPoolStaleCleanup:
    """start() removes the pool folders that dead processes left in the temp folder."""

    @staticmethod
    async def _start_and_stop() -> None:
        """Run the sweep of start(). Pool size 0: the pool makes no folder of its own."""
        pool = OverlayPool(pool_size=0)
        await pool.start([])
        await pool.stop()

    async def test_removes_dir_for_dead_pid(self, fake_daemon: _FakeDaemon, temp_dir: Path) -> None:
        """Directory for a nonexistent PID is cleaned up."""
        del fake_daemon
        stale_dir = temp_dir / "exec-sandbox-overlay-pool-99999-abc"
        stale_dir.mkdir()
        (stale_dir / "overlay.qcow2").write_text("stale")

        with _owner_alive(False):
            await self._start_and_stop()

        assert not stale_dir.exists()

    async def test_preserves_dir_for_alive_pid(self, fake_daemon: _FakeDaemon, temp_dir: Path) -> None:
        """Directory for current PID is NOT cleaned up."""
        del fake_daemon
        alive_dir = temp_dir / f"exec-sandbox-overlay-pool-{os.getpid()}-abc"
        alive_dir.mkdir()
        (alive_dir / "overlay.qcow2").write_text("active")

        await self._start_and_stop()

        assert (alive_dir / "overlay.qcow2").read_text() == "active"

    async def test_a_running_owner_keeps_its_folder(self, fake_daemon: _FakeDaemon, temp_dir: Path) -> None:
        """Its owner runs (ours or another user's: psutil.pid_exists): the folder stays."""
        del fake_daemon
        dir_other = temp_dir / "exec-sandbox-overlay-pool-12345-abc"
        dir_other.mkdir()

        with _owner_alive(True):
            await self._start_and_stop()

        assert dir_other.exists()

    async def test_skips_non_matching_dirs(self, fake_daemon: _FakeDaemon, temp_dir: Path) -> None:
        """Directories that don't match the pool naming pattern are ignored."""
        del fake_daemon
        other_dir = temp_dir / "other-directory-123"
        other_dir.mkdir()

        with _owner_alive(False):
            await self._start_and_stop()

        assert other_dir.exists()

    async def test_skips_non_numeric_suffix(self, fake_daemon: _FakeDaemon, temp_dir: Path) -> None:
        """Directory with non-numeric PID suffix is ignored."""
        del fake_daemon
        bad_dir = temp_dir / "exec-sandbox-overlay-pool-abc"
        bad_dir.mkdir()

        with _owner_alive(False):
            await self._start_and_stop()

        assert bad_dir.exists()

    async def test_stale_sweep_leaves_links_and_folders_of_others(
        self, fake_daemon: _FakeDaemon, temp_dir: Path, tmp_path: Path
    ) -> None:
        """A link named like a dead process's pool folder is not followed: its target stays."""
        del fake_daemon
        target = tmp_path / "precious"
        target.mkdir()
        (target / "data").write_bytes(b"x")
        (temp_dir / "exec-sandbox-overlay-pool-999999999-abc").symlink_to(target)
        dead = temp_dir / "exec-sandbox-overlay-pool-999999998-def"
        dead.mkdir()

        with _owner_alive(False):
            await self._start_and_stop()

        assert (target / "data").exists()
        assert not dead.exists()

    async def test_handles_rmtree_permission_error(self, fake_daemon: _FakeDaemon, temp_dir: Path) -> None:
        """A stale folder that cannot be removed stays, and start() does not fail."""
        del fake_daemon
        stale_dir = temp_dir / "exec-sandbox-overlay-pool-99999-abc"
        stale_dir.mkdir()

        with _owner_alive(False), patch("shutil.rmtree", side_effect=PermissionError("Access denied")):
            await self._start_and_stop()

        assert stale_dir.exists()


# ============================================================================
# Error Handling Tests - Mocks needed to simulate failures
# ============================================================================


class TestOverlayPoolErrorHandling:
    """Failures of the daemon, of the moves and of the pool folder."""

    async def test_cancelled_start_rolls_back_started_daemon(self, temp_dir: Path, tmp_path: Path) -> None:
        """start() cancelled while the daemon starts: the daemon is stopped, the pool has none."""
        del temp_dir
        pool = OverlayPool(pool_size=1, pool_dir=tmp_path / "pool")
        daemon = MagicMock()
        daemon.stop = AsyncMock()
        daemon.started = True
        start_entered = asyncio.Event()

        async def blocked_start() -> None:
            start_entered.set()
            await asyncio.Event().wait()

        daemon.start = AsyncMock(side_effect=blocked_start)

        with patch("exec_sandbox.overlay_pool.QemuStorageDaemon", return_value=daemon):
            startup = asyncio.create_task(pool.start([Path("/tmp/base.qcow2")]))
            await start_entered.wait()
            startup.cancel()
            with pytest.raises(asyncio.CancelledError):
                await startup

        daemon.stop.assert_awaited_once_with()
        assert pool.daemon_enabled is False

    async def test_a_pool_folder_that_cannot_be_made_turns_the_pool_off(
        self, fake_daemon: _FakeDaemon, tmp_path: Path
    ) -> None:
        """A file in place of the pool folder: the pool is off (size 0) and each acquire()
        makes its overlay on demand, with no replenish loop retrying the folder."""
        blocker = tmp_path / "pool"
        blocker.write_text("not a folder")
        pool = OverlayPool(pool_size=2, pool_dir=blocker)
        made: list[Path] = []
        real_create = fake_daemon.create_overlay

        async def counted(base: Path, overlay: Path) -> None:
            made.append(overlay)
            await real_create(base, overlay)

        fake_daemon.create_overlay = counted  # type: ignore[method-assign]
        await pool.start([_BASE])
        try:
            assert pool.pool_size == 0
            for name in ("a.qcow2", "b.qcow2"):
                assert await pool.acquire(_BASE, tmp_path / name) is False  # made on demand
            for _ in range(20):  # a replenish loop would try at once
                await asyncio.sleep(0)
            assert made == [tmp_path / "a.qcow2", tmp_path / "b.qcow2"]
        finally:
            await pool.stop()

    async def test_start_does_not_wait_for_the_pool_to_fill(self, fake_daemon: _FakeDaemon, tmp_path: Path) -> None:
        """start() returns once the daemon runs, and the fill begins at once (not after the
        replenish interval): the first VMs get pool hits as soon as the overlays exist.
        stop() cancels a creation in progress."""
        pool = OverlayPool(pool_size=2, pool_dir=tmp_path / "pool")
        creation_entered = asyncio.Event()

        async def blocked_create(_base: Path, _overlay: Path) -> None:
            creation_entered.set()
            await asyncio.Event().wait()

        fake_daemon.create_overlay = blocked_create  # type: ignore[method-assign]
        # A loop that sleeps before its first fill never reaches creation in this test
        with patch.object(constants, "OVERLAY_POOL_REPLENISH_INTERVAL_SECONDS", 3600):
            # Creation never finishes: a start that waits for it would hang
            async with asyncio.timeout(5):
                await pool.start([Path("/tmp/base.qcow2")])
            assert pool.daemon_enabled is True
            # ...and the background fill did begin
            async with asyncio.timeout(5):
                await creation_entered.wait()

            async with asyncio.timeout(5):
                await pool.stop()

        assert fake_daemon.stops == 1

    async def test_move_failure_falls_back_to_ondemand(
        self, start_pool: Callable[[int], Awaitable[OverlayPool]], tmp_path: Path
    ) -> None:
        """Failed move (e.g. cross-filesystem) removes the pooled overlay and makes one on demand."""
        pool = await start_pool(1)
        target = tmp_path / "target.qcow2"

        with patch("shutil.move", side_effect=OSError(errno.EXDEV, "Cross-device link")):
            assert await pool.acquire(_BASE, target) is False

        assert _overlays(tmp_path / "pool") == set()  # the pooled overlay is removed
        assert target.read_text() == str(target)  # made on demand

    async def test_move_succeeds_same_filesystem(
        self, start_pool: Callable[[int], Awaitable[OverlayPool]], tmp_path: Path
    ) -> None:
        """Real move: the pooled overlay is at the target, none is made on demand."""
        pool = await start_pool(1)
        (pooled,) = _overlays(tmp_path / "pool")
        target = tmp_path / "target.qcow2"

        assert await pool.acquire(_BASE, target) is True

        assert target.read_text() == str(pooled)
        assert not pooled.exists()

    async def test_source_vanishes_before_move(
        self, start_pool: Callable[[int], Awaitable[OverlayPool]], tmp_path: Path
    ) -> None:
        """Pooled overlay deleted before acquire: the move fails, the overlay is made on demand."""
        pool = await start_pool(1)
        (pooled,) = _overlays(tmp_path / "pool")
        pooled.unlink()
        target = tmp_path / "target.qcow2"

        assert await pool.acquire(_BASE, target) is False

        assert target.read_text() == str(target)

    async def test_target_parent_missing(
        self, start_pool: Callable[[int], Awaitable[OverlayPool]], tmp_path: Path
    ) -> None:
        """Target folder missing: the move fails, the pooled overlay is removed, and the
        daemon's TransientError OVERLAY_FAILED of the on-demand creation propagates."""
        pool = await start_pool(1)
        target = tmp_path / "nonexistent" / "target.qcow2"

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED):
            await pool.acquire(_BASE, target)

        assert _overlays(tmp_path / "pool") == set()
        assert not target.parent.exists()

    async def test_orphan_cleanup_failure_suppressed(
        self, start_pool: Callable[[int], Awaitable[OverlayPool]], tmp_path: Path
    ) -> None:
        """Move EIO + unlink EIO -> suppress(OSError) handles both."""
        pool = await start_pool(1)
        target = tmp_path / "target.qcow2"

        with (
            patch("shutil.move", side_effect=OSError(errno.EIO, "I/O error")),
            patch("os.unlink", side_effect=OSError(errno.EIO, "I/O error")),
        ):
            assert await pool.acquire(_BASE, target) is False

        assert target.read_text() == str(target)

    async def test_first_move_fails_queue_retains_rest(
        self, start_pool: Callable[[int], Awaitable[OverlayPool]], tmp_path: Path
    ) -> None:
        """1st fails -> on-demand; 2nd acquire gets the next overlay of the pool."""
        pool = await start_pool(2)
        pool_dir = tmp_path / "pool"
        first = tmp_path / "t1.qcow2"

        with patch("shutil.move", side_effect=OSError("disk error")):
            assert await pool.acquire(_BASE, first) is False
        assert first.read_text() == str(first)
        (left,) = _overlays(pool_dir)

        second = tmp_path / "t2.qcow2"
        assert await pool.acquire(_BASE, second) is True
        assert second.read_text() == str(left)

    async def test_non_oserror_propagates(
        self, start_pool: Callable[[int], Awaitable[OverlayPool]], tmp_path: Path
    ) -> None:
        """RuntimeError from move propagates (not caught by except OSError)."""
        pool = await start_pool(1)

        with (
            patch("shutil.move", side_effect=RuntimeError("unexpected")),
            pytest.raises(RuntimeError, match="unexpected"),
        ):
            await pool.acquire(_BASE, tmp_path / "target.qcow2")

    async def test_stop_handles_rmtree_failure(self, tmp_path: Path) -> None:
        """stop() completes even if directory cleanup fails."""
        pool_dir = tmp_path / "pool"
        pool = OverlayPool(pool_size=5, pool_dir=pool_dir)

        # Use empty startup (no base images = no qemu-img calls needed)
        await pool.start([])

        with patch("shutil.rmtree", side_effect=OSError("Permission denied")):
            await pool.stop()  # Should not raise

        assert pool.daemon_enabled is False
        assert pool_dir.exists()
        await pool.start([])  # stopped for good: it starts again
        await pool.stop()
        assert not pool_dir.exists()

    # --- C: File existence verification after daemon creation ---

    async def test_a_creation_without_a_file_takes_no_pool_place(
        self, fake_daemon: _FakeDaemon, tmp_path: Path
    ) -> None:
        """The daemon reports success but made no file: nothing is queued, so the next fill
        makes a real overlay and acquire gets it (a queued missing file would hold the place,
        and every acquire would make its overlay on demand)."""
        fake_daemon.empty_successes = 1
        pool_dir = tmp_path / "pool"
        pool = OverlayPool(pool_size=1, pool_dir=pool_dir)
        with patch.object(constants, "OVERLAY_POOL_REPLENISH_INTERVAL_SECONDS", 0.01):
            await pool.start([_BASE])
            try:
                await _wait_overlays(pool_dir, 1)
                assert await pool.acquire(_BASE, tmp_path / "target.qcow2") is True
            finally:
                await pool.stop()

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                TransientError(
                    "daemon error", code=ErrorCode.OVERLAY_FAILED, context=None, hint=None, instance_id=None
                ),
                id="overlay-failed",
            ),
            pytest.param(OSError(errno.ENOSPC, "No space left on device"), id="oserror"),
        ],
    )
    async def test_a_failed_creation_is_made_again_at_the_next_fill(
        self, fake_daemon: _FakeDaemon, tmp_path: Path, error: Exception
    ) -> None:
        """A creation fails: nothing is queued, the fill goes on, and the next pass makes the
        overlay that acquire then gets."""
        fake_daemon.failures = [error]
        pool_dir = tmp_path / "pool"
        pool = OverlayPool(pool_size=1, pool_dir=pool_dir)
        with patch.object(constants, "OVERLAY_POOL_REPLENISH_INTERVAL_SECONDS", 0.01):
            await pool.start([_BASE])
            try:
                await _wait_overlays(pool_dir, 1)
                assert await pool.acquire(_BASE, tmp_path / "target.qcow2") is True
            finally:
                await pool.stop()
        assert fake_daemon.failures == []

    # --- D: Cross-process isolation ---

    async def test_stop_only_deletes_own_pool_dir(self, tmp_path: Path) -> None:
        """Pool A's stop() removes dir-A but NOT dir-B."""
        dir_a = tmp_path / "pool-a"
        dir_b = tmp_path / "pool-b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "overlay.qcow2").write_text("a")
        (dir_b / "overlay.qcow2").write_text("b")

        pool_a = OverlayPool(pool_size=0, pool_dir=dir_a)
        await pool_a.start([])
        await pool_a.stop()

        assert not dir_a.exists()
        assert dir_b.exists()
        assert (dir_b / "overlay.qcow2").read_text() == "b"

    async def test_pool_dir_deleted_externally_all_fallback(
        self, start_pool: Callable[[int], Awaitable[OverlayPool]], tmp_path: Path
    ) -> None:
        """Pool folder deleted by another process: all concurrent acquires make their
        overlays on demand."""
        pool = await start_pool(5)
        shutil.rmtree(tmp_path / "pool")

        targets = [tmp_path / f"target-{i}.qcow2" for i in range(5)]
        results = await asyncio.gather(*[pool.acquire(_BASE, t) for t in targets])

        assert results == [False] * 5
        assert [t.read_text() for t in targets] == [str(t) for t in targets]

    # --- E: _create_and_enqueue branches that no public call reaches ---

    @staticmethod
    def _pool_with_queue(pool_dir: Path, daemon: _FakeDaemon, *, maxsize: int) -> tuple[OverlayPool, str]:
        """A pool in the state that start() leaves (daemon, pool folder, a queue for _BASE),
        set by hand, and the queue's key."""
        pool_dir.mkdir()
        pool = OverlayPool(pool_size=maxsize, pool_dir=pool_dir)
        key = str(_BASE.resolve())
        pool._pools[key] = asyncio.Queue(maxsize=maxsize)
        pool._started = True
        daemon.started = True
        pool._daemon = daemon  # type: ignore[assignment]
        return pool, key

    async def test_create_enqueue_queue_full_cleans_file(self, tmp_path: Path) -> None:
        """Queue full when the overlay is made: the new file is deleted. A private-level test:
        one fill loop runs for each base image and makes only the missing overlays, so no
        public call fills the queue during a creation."""
        pool_dir = tmp_path / "pool"
        pool, key = self._pool_with_queue(pool_dir, _FakeDaemon(), maxsize=1)
        existing = pool_dir / "existing.qcow2"
        existing.write_text("existing")
        await pool._pools[key].put(existing)

        await pool._create_and_enqueue(_BASE, key)

        assert _overlays(pool_dir) == {existing}

    async def test_create_enqueue_pool_removed_during_creation(self, tmp_path: Path) -> None:
        """The queue of the base image is gone when the overlay is made: the call returns
        without error and the file stays in the pool folder (stop() removes the folder).
        A private-level test: stop() cancels the creations before it removes the queues, so
        no public call reaches this branch."""
        pool_dir = tmp_path / "pool"
        daemon = _FakeDaemon()
        pool, key = self._pool_with_queue(pool_dir, daemon, maxsize=1)
        real_create = daemon.create_overlay

        async def create_then_drop_queue(base: Path, overlay: Path) -> None:
            await real_create(base, overlay)
            pool._pools.pop(key)

        daemon.create_overlay = create_then_drop_queue  # type: ignore[method-assign]

        await pool._create_and_enqueue(_BASE, key)

        assert len(_overlays(pool_dir)) == 1


# ============================================================================
# Integration Tests - Real qemu-img, real files, real code
# ============================================================================


async def _wait_filled(pool: Any, base_image: Path) -> None:
    """Wait for the background fill that start() begins. It reads the private queue: the pool
    has no fill event to await, and the real daemon makes the overlay file before its qcow2
    data is written and the overlay is queued, so the files of the pool folder do not tell."""
    queue = pool._pools[str(base_image.resolve())]
    async with asyncio.timeout(5):
        while queue.qsize() < pool.pool_size:  # noqa: ASYNC110
            await asyncio.sleep(0.005)


class TestOverlayPoolIntegration:
    """Integration tests with real qemu-img — no mocking.

    No hwaccel required: uses qemu-img and qemu-storage-daemon on the host
    to create/acquire qcow2 overlays — no VMs are booted.
    """

    async def test_full_lifecycle(self, vm_settings, tmp_path: Path) -> None:
        """Test complete lifecycle: start → acquire → stop."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.images.base_image(Language.PYTHON)

        pool_dir = tmp_path / "pool"
        pool = OverlayPool(pool_size=2, pool_dir=pool_dir)
        await pool.start([base_image])
        await _wait_filled(pool, base_image)
        before = _overlays(pool_dir)
        assert len(before) == 2

        # Acquire one
        target = tmp_path / "acquired.qcow2"
        result = await pool.acquire(base_image, target)

        assert result is True
        assert target.exists()
        assert len(before - _overlays(pool_dir)) == 1  # one pooled overlay moved out (a refill adds new names)

        await pool.stop()
        assert not pool_dir.exists()  # Cleaned up

    async def test_acquire_with_zero_pool_size_creates_on_demand(self, vm_settings, tmp_path: Path) -> None:
        """Test acquire works with pool_size=0 (CLI single-VM mode).

        This is a regression test for the bug where pool_size=0, and start()
        didn't initialize the daemon, causing acquire() to fail with
        "Daemon must be started before acquire".
        """
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.images.base_image(Language.PYTHON)

        # pool_size=0
        pool = OverlayPool(pool_size=0, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        # Daemon should be started even with pool_size=0
        assert pool.daemon_enabled is True
        # No pre-created overlays (pool_size=0)
        assert not (tmp_path / "pool").exists()

        # Acquire should work via on-demand creation
        target = tmp_path / "acquired.qcow2"
        result = await pool.acquire(base_image, target)

        # Returns False because created on-demand (not from pool)
        assert result is False
        assert target.exists()

        await pool.stop()

    async def test_acquired_overlay_has_correct_backing_file(self, vm_settings, tmp_path: Path) -> None:
        """Acquired overlay references correct base image."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.images.base_image(Language.PYTHON)

        pool = OverlayPool(pool_size=1, pool_dir=tmp_path / "pool")
        await pool.start([base_image])
        await _wait_filled(pool, base_image)

        target = tmp_path / "acquired.qcow2"
        assert await pool.acquire(base_image, target) is True  # a pooled overlay, moved

        # Verify backing file using qemu-img info
        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "info",
            str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        assert str(base_image) in stdout.decode()

        await pool.stop()

    async def test_pool_exhaustion_creates_ondemand(self, vm_settings, tmp_path: Path) -> None:
        """Acquiring more than pool_size creates on-demand (returns False but succeeds)."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.images.base_image(Language.PYTHON)

        pool = OverlayPool(pool_size=2, pool_dir=tmp_path / "pool")
        await pool.start([base_image])
        await _wait_filled(pool, base_image)

        # Acquire all 2 from pool, then 1 on-demand
        t1 = tmp_path / "t1.qcow2"
        t2 = tmp_path / "t2.qcow2"
        t3 = tmp_path / "t3.qcow2"

        assert await pool.acquire(base_image, t1) is True  # From pool
        assert await pool.acquire(base_image, t2) is True  # From pool
        assert await pool.acquire(base_image, t3) is False  # Created on-demand

        # All 3 overlays should exist and be valid
        assert t1.exists()
        assert t2.exists()
        assert t3.exists()

        await pool.stop()

    async def test_concurrent_acquires_all_succeed(self, vm_settings, tmp_path: Path) -> None:
        """Concurrent acquires all succeed (some from pool, rest on-demand)."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.images.base_image(Language.PYTHON)

        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")
        await pool.start([base_image])
        await _wait_filled(pool, base_image)

        # 10 concurrent acquires for 5 in pool
        targets = [tmp_path / f"target-{i}.qcow2" for i in range(10)]
        results = await asyncio.gather(*[pool.acquire(base_image, t) for t in targets])

        # 5 from pool (True), 5 on-demand (False)
        assert sum(results) == 5

        # All 10 overlays should exist and be valid
        assert all(t.exists() for t in targets)
        sizes = [t.stat().st_size for t in targets]
        assert all(s > 0 for s in sizes)

        await pool.stop()

    async def test_multiple_base_images(self, vm_settings, tmp_path: Path) -> None:
        """Pool handles multiple different base images."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        python_base = vm_manager.images.base_image(Language.PYTHON)
        js_base = vm_manager.images.base_image(Language.JAVASCRIPT)

        pool = OverlayPool(pool_size=2, pool_dir=tmp_path / "pool")
        await pool.start([python_base, js_base])
        await _wait_filled(pool, python_base)
        await _wait_filled(pool, js_base)

        # Acquire from each: both pools are full
        py_target = tmp_path / "py.qcow2"
        js_target = tmp_path / "js.qcow2"

        assert await pool.acquire(python_base, py_target) is True
        assert await pool.acquire(js_base, js_target) is True

        # Verify each has correct backing file
        for target, base in [(py_target, python_base), (js_target, js_base)]:
            proc = await asyncio.create_subprocess_exec(
                "qemu-img",
                "info",
                str(target),
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            assert str(base) in stdout.decode()

        await pool.stop()

    async def test_vm_boots_with_pooled_overlay(self, vm_manager, vm_settings, execute_timeout) -> None:
        """Full integration: VM boots successfully with pooled overlay."""
        from exec_sandbox.models import Language

        # vm_manager fixture already calls start() and stop()
        vm = await vm_manager.create_vm(
            VmSpec(language=Language.PYTHON), tenant_id="test", task_id="pool-test", cache_leases=()
        )

        try:
            result = await vm.execute("print('hello from pool')", timeout_seconds=execute_timeout)
            assert "hello from pool" in result.stdout
        finally:
            await vm_manager.destroy_vm(vm)

"""Per-manager home disk folders: locked while alive, removed whole once their process is gone."""

import fcntl
import gc
import os
import shutil
import stat
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from exec_sandbox import home_disk_dir
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.home_disk_dir import HomeDiskDir, remove_dead_folders
from exec_sandbox.vm_manager import VmManager


def _lock_held(folder: Path) -> bool:
    """True when another open file description cannot take the folder's lock."""
    fd = os.open(folder / ".lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        return False
    finally:
        os.close(fd)


def _dead_folder(root: Path, name: str) -> Path:
    """A folder as a crashed process leaves it: lock file present, lock not held."""
    folder = root / name
    folder.mkdir()
    (folder / ".lock").touch()
    (folder / "vm.img").write_bytes(b"tenant data")
    return folder


class TestHomeDiskDir:
    def test_open_makes_private_locked_folder_and_close_removes_it(self, tmp_path: Path) -> None:
        home = HomeDiskDir.open(tmp_path)
        assert stat.S_IMODE(home.path.stat().st_mode) == 0o700
        assert _lock_held(home.path)
        home.close()
        home.close()  # idempotent
        assert not home.path.exists()

    def test_dead_folder_removed_live_folder_kept(self, tmp_path: Path) -> None:
        live = HomeDiskDir.open(tmp_path)
        (live.path / "vm.img").write_bytes(b"in use")
        dead = _dead_folder(tmp_path, "vmm-dead")
        second = HomeDiskDir.open(tmp_path)
        try:
            assert dead.exists()  # open only makes its own folder: the sweep is a separate job
            remove_dead_folders(tmp_path)
            assert not dead.exists()
            assert (live.path / "vm.img").read_bytes() == b"in use"
            assert second.path.exists()
        finally:
            live.close()
            second.close()

    def test_lockless_folder_removed_only_after_grace(self, tmp_path: Path) -> None:
        young, old = tmp_path / "vmm-young", tmp_path / "vmm-old"
        young.mkdir()
        old.mkdir()
        past = time.time() - home_disk_dir._LOCKLESS_GRACE_SECONDS - 60
        os.utime(old, (past, past))
        remove_dead_folders(tmp_path)
        assert young.exists()
        assert not old.exists()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root opens a file of mode 0")
    def test_folder_whose_lock_cannot_be_opened_is_kept(self, tmp_path: Path) -> None:
        """A lock file this process may not open (another user's manager), or an entry that is
        not a folder: the sweep keeps it and goes on to the other folders."""
        foreign = _dead_folder(tmp_path, "vmm-foreign")
        (foreign / ".lock").chmod(0)
        dead = _dead_folder(tmp_path, "vmm-dead")
        (tmp_path / "stray").write_bytes(b"not a folder")
        try:
            remove_dead_folders(tmp_path)
            assert sorted(p.name for p in tmp_path.iterdir()) == ["stray", "vmm-foreign"]
            assert (foreign / "vm.img").read_bytes() == b"tenant data"
            assert not dead.exists()
        finally:
            (foreign / ".lock").chmod(0o600)

    def test_dropped_without_close_releases(self, tmp_path: Path) -> None:
        home = HomeDiskDir.open(tmp_path)
        path = home.path
        del home
        gc.collect()
        assert not path.exists()

    def test_forked_child_keeps_parents_folder(self, tmp_path: Path) -> None:
        """The child inherits the finalizer: on its exit it must not remove the parent's live
        folder, and closing its copy of the lock fd leaves the parent's lock held."""
        home = HomeDiskDir.open(tmp_path)
        pid = os.fork()
        if pid == 0:  # child: run the finalizer as an exiting child would
            home.close()
            os._exit(0)
        os.waitpid(pid, 0)
        try:
            assert home.path.exists()
            assert _lock_held(home.path)
        finally:
            home.close()

    @pytest.mark.parametrize("mode", [0o755, 0o750], ids=["world-readable", "group-readable"])
    def test_root_others_can_read_accepted(self, tmp_path: Path, mode: int) -> None:
        """The root only lists the per-manager folders (each 0700): only write access is refused."""
        root = tmp_path / "root"
        root.mkdir()
        root.chmod(mode)
        HomeDiskDir.open(root).close()

    @pytest.mark.parametrize("mode", [0o777, 0o770], ids=["world-writable", "group-writable"])
    def test_unsafe_root_refused(self, tmp_path: Path, mode: int) -> None:
        root = tmp_path / "root"
        root.mkdir()
        root.chmod(mode)
        with pytest.raises(
            ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID, match="group or others have access"
        ):
            HomeDiskDir.open(root)

    def test_symlink_root_refused(self, tmp_path: Path) -> None:
        (tmp_path / "real").mkdir(mode=0o700)
        (tmp_path / "link").symlink_to(tmp_path / "real")
        with pytest.raises(
            ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID, match="is a symlink or not a folder"
        ):
            HomeDiskDir.open(tmp_path / "link")

    def test_sweep_between_lock_steps_never_takes_a_new_folder(self, tmp_path: Path) -> None:
        """A sweep that runs while a manager is still creating its lock leaves the folder alone."""
        real_rename = Path.rename
        swept: list[bool] = []

        def sweep_then_rename(self: Path, target: Path) -> Path:
            if not swept:  # once: the inner open renames its own lock
                swept.append(True)
                remove_dead_folders(tmp_path)  # another manager starts and sweeps right now
            return real_rename(self, target)

        with patch.object(Path, "rename", sweep_then_rename):
            home = HomeDiskDir.open(tmp_path)
        try:
            assert swept, "no sweep ran between the lock steps"
            assert _lock_held(home.path)
        finally:
            home.close()


class TestVmManagerSweeps:
    async def test_start_and_stop_remove_folders_off_the_loop(
        self, make_vm_settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A VmManager removes dead managers' folders at start and its own folder at stop, each
        in a worker thread: a folder holds whole disks and the sweep lists a folder with no
        size limit, too long for the event loop."""
        root = tmp_path / "hd"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        loop_thread = threading.get_ident()
        removers: dict[Path, int] = {}
        rmtree = shutil.rmtree

        def spied(path: Path, *args: Any, **kwargs: Any) -> None:
            removers[Path(path)] = threading.get_ident()
            rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", spied)
        manager = VmManager(make_vm_settings(home_disk_dir=root))
        own = manager.work_dir
        assert dead.exists()  # the constructor makes this manager's folder only
        try:
            await manager.start()
            assert not dead.exists()
        finally:
            await manager.stop()
        assert not own.exists()
        assert removers[dead] != loop_thread
        assert removers[own] != loop_thread

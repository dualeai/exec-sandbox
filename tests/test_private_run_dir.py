"""Private folders of running processes: locked while alive, removed whole once their process
is gone; the VM cgroups they record end first (a qemu-vm QEMU outlives its process)."""

import asyncio
import contextlib
import errno
import fcntl
import gc
import logging
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import psutil
import pytest

from exec_sandbox import cgroup, constants, permission_utils, private_run_dir
from exec_sandbox.cgroup import CGROUP_APP_NAMESPACE, CGROUP_V2_BASE_PATH, Cgroup
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.permission_utils import QEMU_VM_SUDO, grant_qemu_vm_dir_access
from exec_sandbox.private_run_dir import PrivateRunDir, new_folder_name, remove_dead_folders
from exec_sandbox.vm_manager import VmManager
from tests.conftest import skip_unless_linux

_TENANT = Path(CGROUP_V2_BASE_PATH) / CGROUP_APP_NAMESPACE / "test-private-run-dir"
"""Where the recorded VM cgroups of these tests would be (Cgroup.end_all is a stand-in)."""


def _open(root: Path) -> PrivateRunDir:
    return PrivateRunDir.open(root, name=new_folder_name())


def _stuck(cgroups: list[Cgroup]) -> list[Cgroup]:
    """A stand-in for Cgroup.end_all: every cgroup keeps a process."""
    return list(cgroups)


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


class TestPrivateRunDir:
    def test_open_makes_private_locked_folder_and_close_removes_it(self, tmp_path: Path) -> None:
        home = _open(tmp_path)
        assert stat.S_IMODE(home.path.stat().st_mode) & 0o777 == 0o700
        assert _lock_held(home.path)
        home.close()
        home.close()  # idempotent
        assert not home.path.exists()

    def test_dead_folder_removed_live_folder_kept(self, tmp_path: Path) -> None:
        live = _open(tmp_path)
        (live.path / "vm.img").write_bytes(b"in use")
        dead = _dead_folder(tmp_path, "vmm-dead")
        second = _open(tmp_path)
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
        """A folder without a lock file (its owner is between mkdir and the lock) has a grace
        of 5 minutes from its last change: a 4-minute-old one stays, a 6-minute-old one goes."""
        young, older, old = tmp_path / "vmm-young", tmp_path / "vmm-older", tmp_path / "vmm-old"
        for folder, age in ((young, 0), (older, 240), (old, 360)):
            folder.mkdir()
            past = time.time() - age
            os.utime(folder, (past, past))
        remove_dead_folders(tmp_path)
        assert young.exists()
        assert older.exists()
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
        home = _open(tmp_path)
        path = home.path
        del home
        gc.collect()
        assert not path.exists()

    @pytest.mark.filterwarnings("ignore:This process .* is multi-threaded, use of fork:DeprecationWarning")
    def test_forked_child_keeps_parents_folder(self, tmp_path: Path) -> None:
        """The child inherits the finalizer: on its exit it must not remove the parent's live
        folder, and closing its copy of the lock fd leaves the parent's lock held. A fork is
        the case under test; the child only closes fds (no lock another thread may hold)."""
        home = _open(tmp_path)
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

    def test_lock_that_is_a_link_is_never_followed(self, tmp_path: Path) -> None:
        """No process makes a ".lock" link: the folder is not one of these, and stays."""
        folder = tmp_path / "vmm-linked"
        folder.mkdir()
        (tmp_path / "free.lock").touch()
        (folder / ".lock").symlink_to(tmp_path / "free.lock")
        (folder / "vm.img").write_bytes(b"tenant data")
        remove_dead_folders(tmp_path)
        assert (folder / "vm.img").exists()

    def test_records_go_in_the_open_folder(self, tmp_path: Path) -> None:
        """The folder's path may name another folder later (a rename): records go in the folder
        this process opened, never through the path."""
        run_dir = _open(tmp_path)
        try:
            moved = tmp_path / "moved"
            run_dir.path.rename(moved)
            run_dir.path.mkdir()  # a decoy at the old path
            run_dir.record_cgroup(_TENANT / "vm-1")
            run_dir.record_folder(tmp_path / "vm-2")
            assert sorted(p.name for p in moved.iterdir() if p.is_symlink() and p.name.startswith("vm-")) == [
                "vm-1.cgroup",
                "vm-2.folder",
            ]
            assert [p.name for p in run_dir.path.iterdir() if p.name.startswith("vm-")] == []
            run_dir.forget_cgroup("vm-1")
        finally:
            run_dir.close()

    @pytest.mark.parametrize("mode", [0o755, 0o750], ids=["world-readable", "group-readable"])
    def test_root_others_can_read_accepted(self, tmp_path: Path, mode: int) -> None:
        """The root only lists the per-manager folders (each 0700): only write access is refused."""
        root = tmp_path / "root"
        root.mkdir()
        root.chmod(mode)
        _open(root).close()

    @pytest.mark.parametrize("mode", [0o777, 0o770], ids=["world-writable", "group-writable"])
    def test_unsafe_root_refused(self, tmp_path: Path, mode: int) -> None:
        root = tmp_path / "root"
        root.mkdir()
        root.chmod(mode)
        with pytest.raises(
            ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID, match="group or others have access"
        ):
            _open(root)

    def test_symlink_root_refused(self, tmp_path: Path) -> None:
        (tmp_path / "real").mkdir(mode=0o700)
        (tmp_path / "link").symlink_to(tmp_path / "real")
        with pytest.raises(
            ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID, match="is a symlink or not a folder"
        ):
            _open(tmp_path / "link")

    @pytest.mark.parametrize("step", ["flock", "rename"])
    def test_sweep_between_lock_steps_never_takes_a_new_folder(self, tmp_path: Path, step: str) -> None:
        """A sweep that runs while a manager is still creating its lock (before the flock: a
        ".lock" made there would be free to take) leaves the folder alone."""
        module = fcntl if step == "flock" else os
        real = getattr(module, step)
        swept: list[bool] = []

        def sweep_then_step(*args: Any, **kwargs: Any) -> Any:
            if not swept:  # once: the sweep itself takes locks too
                swept.append(True)
                remove_dead_folders(tmp_path)  # another manager starts and sweeps right now
            return real(*args, **kwargs)

        with patch.object(module, step, sweep_then_step):
            home = _open(tmp_path)
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
        removers: dict[str, int] = {}
        rmtree = shutil.rmtree

        def spied(path: str, *args: Any, **kwargs: Any) -> None:
            removers[Path(path).name] = threading.get_ident()
            rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", spied)
        manager = VmManager(make_vm_settings(home_disk_dir=root))
        own = manager.work_dir
        assert dead.exists()  # the sweep is start's job
        try:
            await manager.start()
            assert not dead.exists()
        finally:
            await manager.stop()
        assert not own.exists()
        assert removers[dead.name] != loop_thread
        assert removers[own.name] != loop_thread

    async def test_start_makes_the_folder_off_the_loop(
        self, make_vm_settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The constructor runs on the event loop (Scheduler builds the manager there) and makes
        no file: start makes the folder in a worker thread, as its root check reads the user
        and group databases (LDAP or SSSD can take seconds)."""
        root = tmp_path / "hd"
        root.mkdir(mode=0o700)
        loop_thread = threading.get_ident()
        openers: list[int] = []
        real_open = PrivateRunDir.open

        def spied(folder_root: Path, *, name: str) -> PrivateRunDir:
            openers.append(threading.get_ident())
            return real_open(folder_root, name=name)

        monkeypatch.setattr(PrivateRunDir, "open", spied)
        manager = VmManager(make_vm_settings(home_disk_dir=root))
        assert not manager.work_dir.exists()
        assert openers == []
        try:
            await manager.start()
            assert manager.work_dir.is_dir()
            assert len(openers) == 1
            assert openers[0] != loop_thread
        finally:
            await manager.stop()

    async def test_start_removes_the_ram_folders_of_dead_l1_saves(
        self, make_vm_settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The RAM file of a killed L1 save holds memory (/dev/shm): it goes at the next start,
        off the event loop, not only at the next L1 save. A test folder stands in for /dev/shm
        through the module's private root: no setting moves it."""
        shm = tmp_path / "shm"
        shm.mkdir()
        shm.chmod(0o1777)  # as /dev/shm: sticky, open to every user
        user_root = shm / f"exec-sandbox-{os.geteuid()}"
        user_root.mkdir(mode=0o700)
        dead = _dead_folder(user_root, "vmm-dead")
        monkeypatch.setattr("exec_sandbox.memory_snapshot_files._LIVE_RAM_ROOT", shm)
        loop_thread = threading.get_ident()
        removers: dict[str, int] = {}
        rmtree = shutil.rmtree

        def spied(path: str, *args: Any, **kwargs: Any) -> None:
            removers[Path(path).name] = threading.get_ident()
            rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", spied)
        root = tmp_path / "hd"
        root.mkdir(mode=0o700)
        manager = VmManager(make_vm_settings(home_disk_dir=root))
        try:
            await manager.start()
            assert not dead.exists()
            assert removers[dead.name] != loop_thread
        finally:
            await manager.stop()


_OWNER = """
import sys, time
from pathlib import Path
from exec_sandbox import cgroup
from exec_sandbox.private_run_dir import PrivateRunDir, new_folder_name
cgroup.CGROUP_V2_BASE_PATH = sys.argv[2]
if len(sys.argv) > 4:  # the service cgroup this process runs in
    Path(sys.argv[4], "cgroup.procs").write_text("0")
run_dir = PrivateRunDir.open(Path(sys.argv[1]), name=new_folder_name())  # held: no finalizer before the kill
for target in sys.argv[3].split(","):
    run_dir.record_cgroup(Path(target))
sys.stdout.write(run_dir.path.name + "\\n")
sys.stdout.flush()
time.sleep(60)
"""
"""The owner process of the watcher tests (a fresh interpreter, not a fork of this
multi-threaded one): with argv[2] as the cgroup base folder, it records the cgroups of argv[3]
(comma-separated), prints its folder's name, then waits to be killed."""


def _start_owner(
    tmp_path: Path, base: Path, targets: list[Path], *, service: Path | None
) -> tuple[subprocess.Popen[bytes], str]:
    """The owner (_OWNER) in a new session, as a service under a supervisor is, and its folder's name."""
    owner = subprocess.Popen(
        [sys.executable, "-c", _OWNER, str(tmp_path / "root"), str(base), ",".join(map(str, targets))]
        + ([] if service is None else [str(service)]),
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert owner.stdout is not None
    return owner, owner.stdout.readline().decode().strip()


def _end_owner(owner: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(owner.pid, signal.SIGKILL)
    owner.wait()
    assert owner.stdout is not None
    owner.stdout.close()


def _kill_written(fifo: Path, act: Callable[[], None]) -> list[str]:
    """What the watcher writes to `fifo` (a stand-in for cgroup.kill) after `act`. The watcher
    writes within ms; the timeout only bounds a failure."""
    written: list[str] = []
    reader = threading.Thread(target=lambda: written.append(fifo.read_text()))
    reader.start()
    try:
        act()
        reader.join(timeout=5)
    finally:
        if reader.is_alive():  # no write came: end the reader's open
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            reader.join()
    return written


def _watchers(name: str) -> list[psutil.Process]:
    """The watcher of the folder `name` (its arguments name it)."""
    return [
        process
        for process in psutil.process_iter(["cmdline"])
        if "exec-sandbox-watcher" in (cmdline := process.info["cmdline"] or []) and name in cmdline
    ]


@pytest.fixture
def fake_cgroups(tmp_path: Path) -> Iterator[Path]:
    """A folder that stands in for /sys/fs/cgroup in this process: its code-exec/test/vm-1
    has a FIFO for cgroup.kill."""
    base = (tmp_path / "cg").resolve()
    (base / "code-exec" / "test" / "vm-1").mkdir(parents=True)
    os.mkfifo(base / "code-exec" / "test" / "vm-1" / "cgroup.kill")
    with patch.object(cgroup, "CGROUP_V2_BASE_PATH", str(base)):
        yield base


@pytest.fixture
def no_watcher_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    """cgroup v2 unavailable: no watcher gets a cgroup of its own (on Linux, it would be a real
    one; these tests replace Cgroup.end_all, which would remove it)."""
    monkeypatch.setattr(cgroup, "cgroup_v2_available", lambda: False)


@pytest.mark.usefixtures("no_watcher_cgroup")
class TestCgroupRecords:
    """A folder records the VM cgroups of its process: a QEMU that runs as qemu-vm outlives the
    process (its parent is sudo) and only cgroup.kill ends it. The sweep, close and a watcher
    process end them. Cgroup.end_all is a stand-in here; the kernel side is in
    test_cgroup.py and the SIGKILL of a real VM's process in test_vm_manager.py."""

    def test_record_is_a_symlink_to_the_cgroup_until_forgotten(self, tmp_path: Path) -> None:
        run_dir = _open(tmp_path)
        try:
            run_dir.record_cgroup(_TENANT / "vm-1")
            assert (run_dir.path / "vm-1.cgroup").readlink() == _TENANT / "vm-1"
            run_dir.forget_cgroup("vm-1")
            run_dir.forget_cgroup("vm-1")  # idempotent
            assert not (run_dir.path / "vm-1.cgroup").is_symlink()
        finally:
            run_dir.close()

    def test_sweep_ends_the_recorded_cgroups_before_it_removes_a_dead_folder(self, tmp_path: Path) -> None:
        """Only records that name a VM cgroup (<namespace>/<tenant>/<record name>) count: a tenant
        cgroup holds the VMs of other processes, and cgroup.kill on it would kill them too."""
        dead = _dead_folder(tmp_path, "vmm-dead")
        for name, target in [
            ("vm-1", _TENANT / "vm-1"),
            ("vm-2", _TENANT / "vm-2"),
            (_TENANT.name, _TENANT),  # named as its target, at the tenant level
            ("vm-3", _TENANT / "another-vm"),
            ("vm-4", Path("test-private-run-dir") / "vm-4"),
            ("..", _TENANT / ".."),  # its name is "..": the namespace, every VM and the service in it
            ("vm-5", f"{_TENANT}//vm-5"),  # not normalized
            ("vm.6", _TENANT / "vm.6"),  # not a VM id
            ("vm-7", _TENANT.parent / "t.x" / "vm-7"),  # not a tenant id
        ]:
            (dead / f"{name}.cgroup").symlink_to(target)
        ended: list[list[Path]] = []

        def end_all(cgroups: list[Cgroup]) -> list[Cgroup]:
            assert dead.exists()
            ended.append(sorted(each.path for each in cgroups))
            return []

        with patch.object(Cgroup, "end_all", side_effect=end_all):
            remove_dead_folders(tmp_path)
        assert ended == [[_TENANT / "vm-1", _TENANT / "vm-2"]]
        assert not dead.exists()

    def test_one_wait_for_every_dead_folder(self, tmp_path: Path) -> None:
        """Dead folders whose VMs do not end: one Cgroup.end_all call for all of them (one deadline),
        not one each (each start would wait N times CGROUP_DRAIN_TIMEOUT_SECONDS)."""
        for name in ("vmm-dead-1", "vmm-dead-2"):
            (_dead_folder(tmp_path, name) / f"vm-{name[-1]}.cgroup").symlink_to(_TENANT / f"vm-{name[-1]}")
        with patch.object(Cgroup, "end_all", side_effect=_stuck) as end:
            remove_dead_folders(tmp_path)
        assert [sorted(each.path for each in call.args[0]) for call in end.call_args_list] == [
            [_TENANT / "vm-1", _TENANT / "vm-2"]
        ]
        assert sorted(p.name for p in tmp_path.iterdir()) == ["vmm-dead-1", "vmm-dead-2"]

    def test_dead_folder_whose_cgroup_keeps_a_process_stays(self, tmp_path: Path) -> None:
        """Its files may still be in use: the folder stays, unlocked, for the next sweep."""
        dead = _dead_folder(tmp_path, "vmm-dead")
        (dead / "vm-1.cgroup").symlink_to(_TENANT / "vm-1")
        with patch.object(Cgroup, "end_all", side_effect=_stuck):
            remove_dead_folders(tmp_path)
        assert (dead / "vm.img").read_bytes() == b"tenant data"
        assert not _lock_held(dead)

    def test_close_ends_the_cgroups_still_recorded(self, tmp_path: Path) -> None:
        """An owner dropped without its VM cleanup: the record would go with the folder, and
        then nothing could end the VM."""
        run_dir = _open(tmp_path)
        run_dir.record_cgroup(_TENANT / "vm-1")
        with patch.object(Cgroup, "end_all", return_value=[]) as end:
            run_dir.close()
        end.assert_called_once_with([Cgroup(_TENANT / "vm-1")])
        assert not run_dir.path.exists()

    def test_close_keeps_a_folder_whose_cgroup_keeps_a_process(self, tmp_path: Path) -> None:
        run_dir = _open(tmp_path)
        run_dir.record_cgroup(_TENANT / "vm-1")
        with patch.object(Cgroup, "end_all", side_effect=_stuck):
            run_dir.close()
        assert (run_dir.path / "vm-1.cgroup").is_symlink()
        assert not _lock_held(run_dir.path)  # released: the next sweep tries again

    def test_watcher_kills_the_recorded_cgroups_when_the_owner_group_is_killed(
        self, tmp_path: Path, fake_cgroups: Path
    ) -> None:
        """SIGKILL of the owner's whole process group (a supervisor, a shell job; no finalizer
        runs): the watcher, in a session of its own, writes "1" to cgroup.kill of each recorded
        cgroup. A FIFO stands in for cgroup.kill: its read returns the watcher's write."""
        vm = fake_cgroups / "code-exec" / "test" / "vm-1"
        owner, _name = _start_owner(tmp_path, fake_cgroups, [vm], service=None)
        try:
            assert _kill_written(vm / "cgroup.kill", lambda: _end_owner(owner)) == ["1\n"]
        finally:
            _end_owner(owner)

    def test_watcher_ignores_hangup_interrupt_and_term(self, tmp_path: Path, fake_cgroups: Path) -> None:
        """A closed terminal, Ctrl-C or a stop (SIGTERM to every process) can reach the watcher
        before the owner is gone: it stays, and acts when the owner is gone."""
        vm = fake_cgroups / "code-exec" / "test" / "vm-1"
        owner, name = _start_owner(tmp_path, fake_cgroups, [vm], service=None)
        try:
            [watcher] = _watchers(name)
            for each in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
                watcher.send_signal(each)
            assert _kill_written(vm / "cgroup.kill", lambda: _end_owner(owner)) == ["1\n"]
        finally:
            _end_owner(owner)

    def test_watcher_kills_only_vm_cgroups(self, tmp_path: Path, fake_cgroups: Path) -> None:
        """A record whose real path is not <namespace>/<tenant>/<record name>: "..", the namespace
        itself, or a tenant (each cgroup.kill would end the VMs of every process, and the
        service). The watcher skips them. Records go in name order: the skips come before the
        write that ends the wait."""
        namespace = fake_cgroups / "code-exec"
        vm = namespace / "test" / "vm-1"
        owner, name = _start_owner(tmp_path, fake_cgroups, [vm], service=None)
        traps = {
            "vm-0": namespace / "test" / "vm-2",  # another VM's cgroup: not the record's name
            "vm-00": namespace / "t.x" / "vm-00",  # not a tenant id
            "vm-01": namespace / "test" / "..",  # the namespace
            "test": namespace / "test",  # a tenant, named as it
            "...": namespace / "test" / "..",
        }
        decoys = [namespace, namespace / "test", namespace / "test" / "vm-2", namespace / "t.x" / "vm-00"]
        for each in decoys:
            each.mkdir(parents=True, exist_ok=True)
            (each / "cgroup.kill").write_text("")
        for record, target in traps.items():  # planted after the owner's records, before its end
            (tmp_path / "root" / name / f"{record}.cgroup").symlink_to(target)
        try:
            assert _kill_written(vm / "cgroup.kill", lambda: _end_owner(owner)) == ["1\n"]
        finally:
            _end_owner(owner)
        assert [(each / "cgroup.kill").read_text() for each in decoys] == ["", "", "", ""]

    @skip_unless_linux
    def test_first_record_returns_once_the_watcher_is_armed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A watcher slow to start (here 0.1 s): record_cgroup returns only once it ignores
        HUP, INT and TERM (SigIgn in /proc), so a stop right after cannot end it. The delay is
        put in the private watcher script: no public path makes the watcher slow."""
        monkeypatch.setattr(private_run_dir, "_WATCHER", "sleep 0.1\n" + private_run_dir._WATCHER)
        run_dir = _open(tmp_path)
        try:
            run_dir.record_cgroup(_TENANT / "vm-1")
            [watcher] = _watchers(run_dir.path.name)
            status = Path(f"/proc/{watcher.pid}/status").read_text()
            ignored = int(next(line.split()[1] for line in status.splitlines() if line.startswith("SigIgn:")), 16)
            assert [bool(ignored & 1 << (each - 1)) for each in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)] == [
                True,
                True,
                True,
            ]
            run_dir.forget_cgroup("vm-1")
        finally:
            run_dir.close()

    def test_failed_watcher_start_leaves_no_cgroup(
        self, tmp_path: Path, fake_cgroups: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A launcher that starts no watcher: its cgroup and record go. The next record then
        starts the watcher in a new cgroup of its own (a cgroup left there would make it fail,
        and the watcher would run in this process's cgroup). The private launcher script is
        replaced: no public path makes it fail."""
        monkeypatch.setattr(cgroup, "cgroup_v2_available", lambda: True)
        real_start = private_run_dir._START_WATCHER
        monkeypatch.setattr(private_run_dir, "_START_WATCHER", "exit 0\n")
        run_dir = _open(tmp_path / "root")
        own = fake_cgroups / "code-exec" / "exec-sandbox-watcher" / run_dir.path.name
        try:
            run_dir.record_cgroup(fake_cgroups / "code-exec" / "test" / "vm-8")  # no FIFO: close never waits
            assert not own.exists()
            assert not (run_dir.path / f"{own.name}.cgroup").is_symlink()
            monkeypatch.setattr(private_run_dir, "_START_WATCHER", real_start)
            run_dir.record_cgroup(fake_cgroups / "code-exec" / "test" / "vm-9")
            assert (own / "cgroup.procs").read_text() == "0\n"  # the launcher joined its new cgroup
        finally:
            run_dir.close()

    def test_watcher_cgroup_goes_when_its_record_fails(
        self, tmp_path: Path, fake_cgroups: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cgroup with no record: nothing would ever remove it. The watcher then runs in this
        process's cgroup."""
        monkeypatch.setattr(cgroup, "cgroup_v2_available", lambda: True)
        real_symlink = os.symlink

        def symlink(target: Any, record: Any, *args: Any, **kwargs: Any) -> None:
            if str(record).startswith("vmm-"):  # the watcher's own record
                raise OSError(errno.ENOSPC, "No space left on device")
            real_symlink(target, record, *args, **kwargs)

        monkeypatch.setattr(os, "symlink", symlink)
        run_dir = _open(tmp_path / "root")
        try:
            run_dir.record_cgroup(fake_cgroups / "code-exec" / "test" / "vm-9")  # no FIFO: close never waits on it
            assert not (fake_cgroups / "code-exec" / "exec-sandbox-watcher" / run_dir.path.name).exists()
            assert len(_watchers(run_dir.path.name)) == 1
        finally:
            run_dir.close()

    def test_watcher_waits_while_the_owner_lives_and_exits_at_close(self, tmp_path: Path) -> None:
        """One watcher per folder, started by the first record; close ends it (no process left)."""
        run_dir = _open(tmp_path)
        run_dir.record_cgroup(_TENANT / "vm-1")
        run_dir.record_cgroup(_TENANT / "vm-2")
        run_dir.forget_cgroup("vm-1")
        run_dir.forget_cgroup("vm-2")
        watchers = _watchers(run_dir.path.name)
        run_dir.close()
        assert len(watchers) == 1
        _, alive = psutil.wait_procs(watchers, timeout=5)  # exits within ms; the timeout only bounds a failure
        assert not alive


def _stand_in_alive(pid_file: Path) -> bool:
    """Whether the stand-in sudo that wrote `pid_file` still runs (its pid may be reused by
    another program after its end, and a stand-in killed before it wrote the file never ran)."""
    try:
        pid = int(pid_file.read_text())
        return "sleep" in " ".join(psutil.Process(pid).cmdline())
    except (OSError, ValueError, psutil.Error):
        return False


class TestFolderRecords:
    """A folder records folders of its process outside it (the VM working folders in the temp
    folder): close and the sweep remove them once the recorded cgroups have no process left."""

    @staticmethod
    def _workdir(tmp_path: Path, name: str) -> Path:
        workdir = tmp_path / "tmp" / name
        workdir.mkdir(parents=True)
        (workdir / "overlay.qcow2").write_bytes(b"guest writes")
        return workdir

    def test_record_is_a_symlink_to_the_folder_until_forgotten(self, tmp_path: Path) -> None:
        run_dir = _open(tmp_path / "root")
        try:
            workdir = self._workdir(tmp_path, "vm-1")
            run_dir.record_folder(workdir)
            assert (run_dir.path / "vm-1.folder").readlink() == workdir
            run_dir.forget_folder("vm-1")
            run_dir.forget_folder("vm-1")  # idempotent
            assert not (run_dir.path / "vm-1.folder").is_symlink()
        finally:
            run_dir.close()
        assert workdir.exists()  # forgotten: close leaves it

    def test_sweep_removes_the_recorded_folders_of_a_dead_folder_after_its_vms(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        workdir = self._workdir(tmp_path, "vm-1")
        (dead / "vm-1.folder").symlink_to(workdir)
        (dead / "vm-1.cgroup").symlink_to(_TENANT / "vm-1")
        workdir_at_end: list[bool] = []

        def end_all(_cgroups: list[Cgroup]) -> list[Cgroup]:
            workdir_at_end.append(workdir.exists())  # QEMU may still use it until then
            return []

        with patch.object(Cgroup, "end_all", side_effect=end_all):
            remove_dead_folders(root)
        assert workdir_at_end == [True]
        assert not workdir.exists()
        assert not dead.exists()

    def test_recorded_folder_stays_while_a_recorded_cgroup_keeps_a_process(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        workdir = self._workdir(tmp_path, "vm-1")
        (dead / "vm-1.folder").symlink_to(workdir)
        (dead / "vm-1.cgroup").symlink_to(_TENANT / "vm-1")
        with patch.object(Cgroup, "end_all", side_effect=_stuck):
            remove_dead_folders(root)
        assert (workdir / "overlay.qcow2").exists()
        assert dead.exists()

    def test_only_a_real_folder_named_as_its_record_is_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record whose target is a link, has another name or a relative path (it would
        resolve from the sweeping process's working folder) names no folder of this process:
        the sweep never follows it."""
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        real = self._workdir(tmp_path, "real")
        (tmp_path / "tmp" / "vm-link").symlink_to(real)
        (dead / "vm-link.folder").symlink_to(tmp_path / "tmp" / "vm-link")
        (dead / "vm-2.folder").symlink_to(real)
        (dead / "real.folder").symlink_to(Path("tmp") / "real")
        monkeypatch.chdir(tmp_path)  # where the relative target names `real`
        remove_dead_folders(root)
        assert (real / "overlay.qcow2").exists()
        assert not dead.exists()

    def test_dot_dot_names_no_recorded_folder(self, tmp_path: Path) -> None:
        """A record named "..", whose target ends in "..": the folder above would go."""
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        home = self._workdir(tmp_path, "Documents").parent
        (dead / "...folder").symlink_to(home / "Documents" / "..")
        remove_dead_folders(root)
        assert (home / "Documents" / "overlay.qcow2").exists()
        assert not dead.exists()

    def test_dot_dot_inside_a_folder_record_names_no_folder(self, tmp_path: Path) -> None:
        """A path with ".." inside it, though its last name is the record's name."""
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        workdir = self._workdir(tmp_path, "vm-1")
        (tmp_path / "tmp" / "Documents").mkdir()
        (dead / "vm-1.folder").symlink_to(f"{tmp_path}/tmp/Documents/../vm-1")
        remove_dead_folders(root)
        assert (workdir / "overlay.qcow2").exists()

    @pytest.mark.parametrize("kept_by_rmtree", [False, True], ids=["files-of-this-user", "files-of-qemu-vm"])
    def test_folder_of_group_qemu_vm_goes_by_rmtree_then_sudo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kept_by_rmtree: bool
    ) -> None:
        """Under `sg qemu-vm` every new folder has group qemu-vm, also those of a QEMU that runs
        as this user: rmtree removes them, with no sudo. Only what rmtree leaves (files of a
        qemu-vm QEMU) goes with `sudo -n rm -rf`. A folder only root may enter stands in for
        those files; a stand-in for sudo removes it."""
        if os.geteuid() == 0:
            pytest.skip("root enters every folder")
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        workdir = self._workdir(tmp_path, "vm-1")
        if kept_by_rmtree:
            (workdir / "private").mkdir()
            (workdir / "private" / "f").touch()
            (workdir / "private").chmod(0)
        (dead / "vm-1.folder").symlink_to(workdir)
        monkeypatch.setattr(private_run_dir, "get_qemu_vm_gid", lambda: workdir.stat().st_gid)
        commands: list[list[str]] = []

        def sudo(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            commands.append(command)
            (workdir / "private").chmod(0o700)
            shutil.rmtree(command[-1])
            return subprocess.CompletedProcess(command, 0, b"", b"")

        monkeypatch.setattr(private_run_dir.subprocess, "run", sudo)
        remove_dead_folders(root)
        assert not workdir.exists()
        assert commands == ([["sudo", "-n", "rm", "-rf", "--", str(workdir)]] if kept_by_rmtree else [])

    def test_stuck_sudo_is_bounded_and_the_record_stays(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A sudo that hangs (a hostname, PAM or SSSD lookup) is killed after
        constants.SUDO_RM_TIMEOUT_SECONDS (here 0.2 s): close and the start sweep never hang on
        it, and the record stays for the next sweep. A stand-in sudo sleeps 30 s."""
        if os.geteuid() == 0:
            pytest.skip("root enters every folder")
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        workdir = self._workdir(tmp_path, "vm-1")
        (workdir / "private").mkdir()
        (workdir / "private").chmod(0)  # rmtree leaves it: sudo's turn
        (dead / "vm-1.folder").symlink_to(workdir)
        stand_in = tmp_path / "bin" / "sudo"
        stand_in.parent.mkdir()
        pid_file = tmp_path / "sudo.pid"
        stand_in.write_text(f'#!/bin/sh\necho $$ > "{pid_file}"\nexec sleep 30\n')
        stand_in.chmod(0o755)
        monkeypatch.setenv("PATH", f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setattr(private_run_dir, "get_qemu_vm_gid", lambda: workdir.stat().st_gid)
        monkeypatch.setattr(constants, "SUDO_RM_TIMEOUT_SECONDS", 0.2)
        try:
            start = time.monotonic()
            remove_dead_folders(root)
            assert (
                time.monotonic() - start < 5.0
            )  # far below the stand-in's 30 s (the root checks read the user databases)
            assert sorted(p.name for p in dead.iterdir()) == [".lock", "vm-1.folder"]
            assert not _stand_in_alive(pid_file)  # killed and reaped
        finally:
            (workdir / "private").chmod(0o700)

    def test_one_stuck_sudo_per_sweep(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stuck sudo (SSSD down) costs its bound once per sweep, not once per folder (50
        folders at 2 s: 100 s at each start and close): after its first timeout, the other
        folders keep their records for the next sweep, with no sudo. A stand-in sudo sleeps; the
        sudo starts are counted at the spawn (a log line from the stand-in itself may come after
        the bound under load)."""
        if os.geteuid() == 0:
            pytest.skip("root enters every folder")
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        workdirs = []
        for index in range(3):
            dead = _dead_folder(root, f"vmm-dead-{index}")
            workdir = self._workdir(tmp_path, f"vm-{index}")
            (workdir / "private").mkdir()
            (workdir / "private").chmod(0)  # rmtree leaves it: sudo's turn
            (dead / f"vm-{index}.folder").symlink_to(workdir)
            workdirs.append(workdir)
        stand_in = tmp_path / "bin" / "sudo"
        stand_in.parent.mkdir()
        stand_in.write_text("#!/bin/sh\nexec sleep 5\n")
        stand_in.chmod(0o755)
        calls: list[list[str]] = []

        def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            if args[0] == "sudo":
                calls.append(args)
            return subprocess.run(args, **kwargs)  # noqa: PLW1510 - check comes from the code under test

        monkeypatch.setattr(private_run_dir, "subprocess", SimpleNamespace(**{**vars(subprocess), "run": run}))
        monkeypatch.setenv("PATH", f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setattr(private_run_dir, "get_qemu_vm_gid", lambda: workdirs[0].stat().st_gid)
        monkeypatch.setattr(constants, "SUDO_RM_TIMEOUT_SECONDS", 0.2)
        try:
            remove_dead_folders(root)
            assert len(calls) == 1
            assert sorted(p.name for folder in root.iterdir() for p in folder.iterdir()) == [
                ".lock",
                ".lock",
                ".lock",
                "vm-0.folder",
                "vm-1.folder",
                "vm-2.folder",
            ]
        finally:
            for workdir in workdirs:
                (workdir / "private").chmod(0o700)

    def test_a_failed_sudo_rm_logs_its_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """rm's stderr reaches the warning: a busy mount and a missing sudo rule both exit 1."""
        if os.geteuid() == 0:
            pytest.skip("root enters every folder")
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead-0")
        workdir = self._workdir(tmp_path, "vm-0")
        (workdir / "private").mkdir()
        (workdir / "private").chmod(0)  # rmtree leaves it: sudo's turn
        (dead / "vm-0.folder").symlink_to(workdir)
        stand_in = tmp_path / "bin" / "sudo"
        stand_in.parent.mkdir()
        # sudo -n rm -rf -- <folder>: the folder is $5
        stand_in.write_text("#!/bin/sh\necho \"rm: cannot remove '$5/x': Device or resource busy\" >&2\nexit 1\n")
        stand_in.chmod(0o755)
        monkeypatch.setenv("PATH", f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setattr(private_run_dir, "get_qemu_vm_gid", lambda: workdir.stat().st_gid)
        try:
            with caplog.at_level(logging.WARNING, logger="exec_sandbox.private_run_dir"):
                remove_dead_folders(root)
            [record] = [r for r in caplog.records if r.getMessage() == "Could not remove a folder of a stopped process"]
            assert record.__dict__["stderr"] == f"rm: cannot remove '{workdir}/x': Device or resource busy"
        finally:
            (workdir / "private").chmod(0o700)

    def test_no_room_for_the_error_file_still_removes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No inode left in the temp folder for sudo's error file: the sweep still runs sudo
        (its errors to /dev/null) and removes the folder and its record."""
        if os.geteuid() == 0:
            pytest.skip("root enters every folder")
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead-0")
        workdir = self._workdir(tmp_path, "vm-0")
        (workdir / "private").mkdir()
        (workdir / "private").chmod(0)  # rmtree leaves it: sudo's turn
        (dead / "vm-0.folder").symlink_to(workdir)
        stand_in = tmp_path / "bin" / "sudo"
        stand_in.parent.mkdir()
        stand_in.write_text('#!/bin/sh\nchmod -R u+rwx "$5"\n/bin/rm -rf -- "$5"\n')
        stand_in.chmod(0o755)
        monkeypatch.setenv("PATH", f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setattr(private_run_dir, "get_qemu_vm_gid", lambda: workdir.stat().st_gid)

        def no_room(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(
            permission_utils, "tempfile", SimpleNamespace(**{**vars(tempfile), "TemporaryFile": no_room})
        )
        try:
            remove_dead_folders(root)
            assert not workdir.exists()
            assert list(root.iterdir()) == []
        finally:
            if (workdir / "private").exists():
                (workdir / "private").chmod(0o700)

    def test_folder_that_cannot_be_removed_keeps_its_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dead folder keeps its lock and the records still pending (the next sweep tries
        them again); the rest goes now: the home disks, the records of what has ended."""
        if os.geteuid() == 0:
            pytest.skip("root enters every folder")
        monkeypatch.setattr(private_run_dir, "get_qemu_vm_gid", lambda: None)  # no sudo rm (no qemu-vm here)
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        stuck = self._workdir(tmp_path, "vm-1")
        (stuck / "private").mkdir()
        (stuck / "private" / "f").touch()
        (stuck / "private").chmod(0)  # rmtree cannot empty it
        done = self._workdir(tmp_path, "vm-2")
        for each in (stuck, done):
            (dead / f"{each.name}.folder").symlink_to(each)
        (dead / "vm-3.cgroup").symlink_to(_TENANT / "vm-3")
        try:
            with patch.object(Cgroup, "end_all", return_value=[]):
                remove_dead_folders(root)
            assert sorted(p.name for p in dead.iterdir()) == [".lock", "vm-1.folder"]
            assert not done.exists()
        finally:
            (stuck / "private").chmod(0o700)
        remove_dead_folders(root)  # now it can go: the next sweep removes it, then the folder
        assert not stuck.exists()
        assert not dead.exists()

    def test_folder_of_another_user_is_never_removed(self, tmp_path: Path) -> None:
        """A recorded folder that is not this user's: not one this process made."""
        run_dir = _open(tmp_path / "root")
        workdir = self._workdir(tmp_path, "vm-1")
        run_dir.record_folder(workdir)
        with patch("os.geteuid", return_value=os.geteuid() + 1):  # the folder now belongs to someone else
            run_dir.close()
        assert (workdir / "overlay.qcow2").exists()

    def test_close_removes_the_folders_still_recorded(self, tmp_path: Path) -> None:
        """An owner dropped without its VM cleanup: the record would go with the folder."""
        run_dir = _open(tmp_path / "root")
        workdir = self._workdir(tmp_path, "vm-1")
        run_dir.record_folder(workdir)
        run_dir.close()
        assert not workdir.exists()

    def test_sweep_refuses_a_root_others_can_write(self, tmp_path: Path) -> None:
        """Anyone could have put folders and records in it: nothing in it is removed."""
        root = tmp_path / "root"
        root.mkdir()
        root.chmod(0o777)
        dead = _dead_folder(root, "vmm-dead")
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID):
            remove_dead_folders(root)
        assert dead.exists()

    def test_sweep_of_a_missing_root_does_nothing(self, tmp_path: Path) -> None:
        """No error, and the sweep does not make the root."""
        remove_dead_folders(tmp_path / "never-made")
        assert not (tmp_path / "never-made").exists()


class TestRootChecks:
    """The root holds every process's folder and records: another user who could swap it
    could make the sweep kill VMs of their choice and remove folders of this user."""

    def test_sweep_works_in_the_root_it_checked(self, tmp_path: Path) -> None:
        """A user who can write a parent of the root swaps it for a link right after its check:
        the sweep goes on in the folder it opened and checked, and reads no planted record."""
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        other = tmp_path / "other"
        other.mkdir(mode=0o700)
        planted = _dead_folder(other, "vmm-planted")
        victim = tmp_path / "projects"
        victim.mkdir()
        (victim / "src.py").write_text("code")
        (planted / "projects.folder").symlink_to(victim)
        (planted / "vm-x.cgroup").symlink_to(_TENANT / "vm-x")
        real_check = private_run_dir.private_tree_problem

        def check_then_swap(path: Path, fd: int, *, forbidden: int) -> tuple[str, str] | None:
            problem = real_check(path, fd, forbidden=forbidden)
            root.rename(tmp_path / "moved")
            root.symlink_to(other)
            return problem

        with (
            patch.object(private_run_dir, "private_tree_problem", check_then_swap),
            patch.object(Cgroup, "end_all", return_value=[]) as end,
        ):
            remove_dead_folders(root)
        end.assert_not_called()
        assert (victim / "src.py").exists()
        assert planted.exists()
        assert not (tmp_path / "moved" / dead.name).exists()

    def test_root_in_a_folder_others_can_change_is_refused(self, tmp_path: Path) -> None:
        """Another user could rename the root and put another folder in its place."""
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o777)  # not sticky
        for use in (_open, remove_dead_folders):
            with pytest.raises(
                ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID, match="another user can change"
            ):
                use(shared / "root")

    def test_root_in_a_sticky_shared_folder_is_accepted(self, tmp_path: Path) -> None:
        """As /tmp and /dev/shm: others cannot rename what they do not own."""
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o1777)
        _open(shared / "root").close()
        remove_dead_folders(shared / "root")


def _wait_until_empty(cgroup_path: Path, *, timeout: float) -> bool:
    """Whether `cgroup_path` has no process left within `timeout` s: poll(2) on cgroup.events,
    which wakes at each change. It kills nothing itself."""
    import select

    fd = os.open(cgroup_path / "cgroup.events", os.O_RDONLY)
    try:
        poller = select.poll()
        poller.register(fd, select.POLLPRI | select.POLLERR)
        deadline = time.monotonic() + timeout
        while b"populated 1" in os.pread(fd, 4096, 0):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            poller.poll(remaining * 1000)
        return True
    finally:
        os.close(fd)


def _delegated_namespace() -> Path:
    """The cgroup namespace (code-exec) when this process runs inside it (the README's
    delegated subtree: it can move processes between its cgroups), else a skip."""
    if not (cgroup.resolve_self_cgroup_v2() or "").startswith(f"/{CGROUP_APP_NAMESPACE}/"):
        pytest.skip("this process does not run inside the delegated cgroup v2 subtree")
    return Path(CGROUP_V2_BASE_PATH) / CGROUP_APP_NAMESPACE


@skip_unless_linux
class TestWatcherOwnCgroup:
    """The watcher runs in a cgroup of its own (code-exec/exec-sandbox-watcher/<folder>): a
    cgroup.kill of the service's cgroup (a systemd stop with KillMode=control-group, an OOM
    group kill, an operator) ends the service, not the watcher, which then ends its VMs."""

    def test_vms_end_when_the_service_cgroup_is_killed(self, tmp_path: Path) -> None:
        namespace = _delegated_namespace()
        suffix = secrets.token_hex(4)
        service = namespace / f"test-watcher-service-{suffix}"
        vm = namespace / "test-watcher" / f"vz-{suffix}"  # its record comes after the watcher's own (vmm-)
        made: list[Path] = []
        sleeper: subprocess.Popen[bytes] | None = None
        owner: subprocess.Popen[bytes] | None = None
        try:
            for each in (service, vm):
                each.mkdir(parents=True)
                made.append(each)
            sleeper = subprocess.Popen(["sleep", "60"])  # stands in for a qemu-vm QEMU
            (vm / "cgroup.procs").write_text(str(sleeper.pid))
            owner, name = _start_owner(tmp_path, Path(CGROUP_V2_BASE_PATH), [vm], service=service)
            watcher_cgroup = namespace / "exec-sandbox-watcher" / name
            made.append(watcher_cgroup)
            assert (service / "cgroup.procs").read_text().split() == [str(owner.pid)]
            assert len((watcher_cgroup / "cgroup.procs").read_text().split()) == 1

            (service / "cgroup.kill").write_text("1")
            assert owner.wait(timeout=5) == -signal.SIGKILL
            assert _wait_until_empty(vm, timeout=2.0)  # the watcher acts in ms
            assert sleeper.wait(timeout=5) == -signal.SIGKILL

            remove_dead_folders(tmp_path / "root")  # the next start: the VM's and the watcher's cgroups go
            assert not vm.exists()
            assert not watcher_cgroup.exists()
            assert list((tmp_path / "root").iterdir()) == []
        finally:
            if owner is not None:
                _end_owner(owner)
            if sleeper is not None and sleeper.poll() is None:
                sleeper.kill()
                sleeper.wait()
            Cgroup.end_all([Cgroup(each) for each in made if each.exists()])
            with contextlib.suppress(OSError):
                (namespace / "test-watcher").rmdir()

    def test_watcher_that_does_not_arm_in_time_goes_with_its_cgroup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A launcher that joins its cgroup, then hangs (here past a 0.2 s bound): what is in the
        cgroup is killed and the cgroup removed, with its record; record_cgroup returns. The
        private launcher script and bound are replaced: no public path makes the launcher hang."""
        namespace = _delegated_namespace()
        monkeypatch.setattr(
            private_run_dir, "_START_WATCHER", 'exec 3<&0\n{ echo 0 > "$1/cgroup.procs"; } 2>/dev/null\nsleep 5\n'
        )
        monkeypatch.setattr(private_run_dir, "_WATCHER_START_SECONDS", 0.2)
        run_dir = _open(tmp_path)
        watcher_cgroup = namespace / "exec-sandbox-watcher" / run_dir.path.name
        try:
            start = time.monotonic()
            run_dir.record_cgroup(namespace / "test-watcher" / f"vm-{secrets.token_hex(4)}")  # never made
            assert time.monotonic() - start < 2.0
            assert not watcher_cgroup.exists()
            assert not (run_dir.path / f"{watcher_cgroup.name}.cgroup").is_symlink()
        finally:
            run_dir.close()
            if watcher_cgroup.exists():
                Cgroup.end_all([Cgroup(watcher_cgroup)])

    def test_close_removes_the_watcher_and_its_cgroup(self, tmp_path: Path) -> None:
        namespace = _delegated_namespace()
        run_dir = _open(tmp_path)
        watcher_cgroup = namespace / "exec-sandbox-watcher" / run_dir.path.name
        try:
            run_dir.record_cgroup(namespace / "test-watcher" / f"vm-{secrets.token_hex(4)}")  # never made
            watchers = _watchers(run_dir.path.name)
            assert [str(process.pid) for process in watchers] == (watcher_cgroup / "cgroup.procs").read_text().split()
        finally:
            run_dir.close()
        assert not watcher_cgroup.exists()
        _, alive = psutil.wait_procs(watchers, timeout=5)  # killed by close; the timeout only bounds a failure
        assert not alive


@skip_unless_linux
class TestQemuVmWorkingFolder:
    async def test_sweep_removes_a_folder_qemu_vm_files_are_in(self, tmp_path: Path) -> None:
        """A VM working folder of a qemu-vm QEMU (group qemu-vm, 2770) holds files of qemu-vm:
        the sweep removes it with sudo, as VmWorkingDirectory.cleanup does. A folder only
        qemu-vm can open stands in for those files."""
        from exec_sandbox.permission_utils import ensure_traversable, probe_qemu_as_qemu_vm

        if not await probe_qemu_as_qemu_vm():
            pytest.skip("QEMU does not run as qemu-vm here")
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        dead = _dead_folder(root, "vmm-dead")
        workdir = tmp_path / "tmp" / "vm-test-qemu-vm"
        workdir.mkdir(parents=True)
        assert await ensure_traversable(workdir / "private")  # as for a VM's files: qemu-vm reaches the folder
        assert grant_qemu_vm_dir_access(workdir)
        await asyncio.to_thread(
            subprocess.run,
            [
                *QEMU_VM_SUDO,
                "/bin/sh",
                "-c",
                'mkdir "$1/private" && touch "$1/private/f" && chmod 700 "$1/private"',
                "sh",
                str(workdir),
            ],
            check=True,
        )
        (dead / f"{workdir.name}.folder").symlink_to(workdir)
        try:
            remove_dead_folders(root)
            assert not workdir.exists()
            assert not dead.exists()
        finally:  # a failure leaves files this user cannot remove (nor pytest's cleanup)
            if workdir.exists():
                await asyncio.to_thread(subprocess.run, ["sudo", "-n", "rm", "-rf", "--", str(workdir)], check=False)

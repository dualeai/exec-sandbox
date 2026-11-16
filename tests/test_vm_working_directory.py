"""Tests for VmWorkingDirectory class."""

from __future__ import annotations

import asyncio
import errno
import os
import re
import shutil
import socket
import stat
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

import exec_sandbox.vm_working_directory as workdir_module
from exec_sandbox import constants
from exec_sandbox.exceptions import ErrorCode, TransientError
from exec_sandbox.models import Language
from exec_sandbox.permission_utils import get_qemu_vm_gid, get_qemu_vm_uid, probe_qemu_as_qemu_vm
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_working_directory import VmWorkingDirectory
from tests.conftest import skip_unless_linux

if TYPE_CHECKING:
    from exec_sandbox.vm_manager import VmManager

# Home disks of test workdirs (no VM runs here, so no file is created)
_HOME_DIR = Path(tempfile.gettempdir())


def _sudo_stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_first: bool) -> Path:
    """Put a stand-in for sudo on PATH: it adds its arguments to a log (one line for each
    call), then runs them without the first one (-n) as this user. With `fail_first`, its
    first call removes nothing and exits 1. Returns the log."""
    log, failed = tmp_path / "sudo.log", tmp_path / "sudo.failed"
    stand_in = tmp_path / "bin" / "sudo"
    stand_in.parent.mkdir()
    fail = f'if [ ! -e "{failed}" ]; then : > "{failed}"; exit 1; fi\n' if fail_first else ""
    stand_in.write_text(f'#!/bin/sh\necho "$*" >> "{log}"\n{fail}shift\nexec "$@"\n')
    stand_in.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}")
    return log


class TestVmWorkingDirectoryCreate:
    """Test VmWorkingDirectory.create() method."""

    async def test_create_makes_a_private_directory_in_the_temp_folder(self):
        """create() makes a directory in the temp folder, of mode 0700: other users get no
        access to the overlay and the sockets."""
        workdir = await VmWorkingDirectory.create("test-vm-123", home_dir=_HOME_DIR)
        try:
            st = workdir.path.stat()
            assert (workdir.path.parent, stat.S_ISDIR(st.st_mode), stat.S_IMODE(st.st_mode)) == (
                Path(tempfile.gettempdir()),
                True,
                0o700,
            )
        finally:
            await workdir.cleanup()

    async def test_a_folder_that_cannot_be_made_is_local_disk(self) -> None:
        """mkdtemp fails (a full disk, no inodes): TransientError LOCAL_DISK with the OS reason,
        never a raw OSError out of create_vm. A fake mkdtemp: only a full disk makes the real
        one fail with ENOSPC."""

        def full_disk(*, prefix: str) -> str:
            raise OSError(errno.ENOSPC, "No space left on device")

        fake = SimpleNamespace(mkdtemp=full_disk, gettempdir=tempfile.gettempdir)
        with (
            patch("exec_sandbox.vm_working_directory.tempfile", fake),
            pytest.raises(
                TransientError,
                check=lambda e: e.code is ErrorCode.LOCAL_DISK and isinstance(e.__cause__, OSError),
                match=r"^Could not make the VM working folder: No space left on device$",
            ),
        ):
            await VmWorkingDirectory.create("test-vm-full", home_dir=_HOME_DIR)

    async def test_create_sets_vm_id(self):
        """create() stores the vm_id."""
        workdir = await VmWorkingDirectory.create("test-vm-456", home_dir=_HOME_DIR)
        try:
            assert workdir.vm_id == "test-vm-456"
        finally:
            await workdir.cleanup()

    async def test_create_uses_prefix(self):
        """The directory name is "vm-", the first 8 characters of the vm_id and "-", then the
        random part of mkdtemp."""
        workdir = await VmWorkingDirectory.create("test-vm-789", home_dir=_HOME_DIR)
        try:
            assert re.fullmatch(r"vm-test-vm--[a-z0-9_]+", workdir.path.name) is not None, workdir.path.name
        finally:
            await workdir.cleanup()

    async def test_cancelled_thread_creation_removes_abandoned_directory(self, tmp_path: Path):
        """mkdtemp cannot publish a directory after its cancelled caller exits."""
        abandoned = tmp_path / "abandoned"
        created = threading.Event()
        release_thread = threading.Event()

        def blocked_mkdtemp(*_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            abandoned.mkdir()
            created.set()
            release_thread.wait()
            return str(abandoned)

        with patch("exec_sandbox.vm_working_directory.tempfile.mkdtemp", side_effect=blocked_mkdtemp):
            creation = asyncio.create_task(VmWorkingDirectory.create("cancelled-create", home_dir=_HOME_DIR))
            assert await asyncio.to_thread(created.wait, 1)
            creation.cancel()
            release_thread.set()
            with pytest.raises(asyncio.CancelledError):
                await creation

        assert not abandoned.exists()

    async def test_cancelled_creation_retains_failed_cleanup_until_retry_succeeds(self, tmp_path: Path):
        """A cancelled create() whose first removal of the new folder fails: one module retry
        owns the folder and removes it, then leaves the module's set of retry tasks. Only
        that private set shows a retry task that stays after its end
        (drain_workdir_cleanup_tasks clears the set)."""
        abandoned = tmp_path / "abandoned-retry"
        created = threading.Event()
        release_thread = threading.Event()
        real_rmtree = shutil.rmtree
        attempts = 0
        before = set(workdir_module._WORKDIR_CLEANUP_TASKS)

        def blocked_mkdtemp(*_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            abandoned.mkdir()
            created.set()
            release_thread.wait()
            return str(abandoned)

        def fail_once(target: Path) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("simulated removal failure")
            real_rmtree(target)

        with (
            patch("exec_sandbox.vm_working_directory.tempfile.mkdtemp", side_effect=blocked_mkdtemp),
            patch("exec_sandbox.vm_working_directory.shutil.rmtree", side_effect=fail_once),
        ):
            creation = asyncio.create_task(VmWorkingDirectory.create("cancelled-retry", home_dir=_HOME_DIR))
            assert await asyncio.to_thread(created.wait, 1)
            creation.cancel()
            release_thread.set()
            with pytest.raises(asyncio.CancelledError):
                await creation
            retries = set(workdir_module._WORKDIR_CLEANUP_TASKS) - before
            assert len(retries) == 1  # the folder of the failed removal has an owner
            await asyncio.wait_for(asyncio.gather(*retries), timeout=1)

        assert attempts == 2
        assert not abandoned.exists()
        assert not retries & workdir_module._WORKDIR_CLEANUP_TASKS

    async def test_cancelled_creation_whose_cleanup_is_cancelled_retains_the_folder(self, tmp_path: Path):
        """A cancelled create() whose removal of the new folder is cancelled too: the folder
        gets a retry owner (_retain_cleanup), as after a failed removal."""
        abandoned = tmp_path / "abandoned-cancelled-cleanup"
        created = threading.Event()
        release_thread = threading.Event()

        def blocked_mkdtemp(*_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            abandoned.mkdir()
            created.set()
            release_thread.wait()
            return str(abandoned)

        with (
            patch("exec_sandbox.vm_working_directory.tempfile.mkdtemp", side_effect=blocked_mkdtemp),
            patch.object(VmWorkingDirectory, "cleanup", AsyncMock(side_effect=asyncio.CancelledError)),
            patch.object(VmWorkingDirectory, "_retain_cleanup", autospec=True) as retain,
        ):
            creation = asyncio.create_task(VmWorkingDirectory.create("cancelled-cleanup", home_dir=_HOME_DIR))
            assert await asyncio.to_thread(created.wait, 1)
            creation.cancel()
            release_thread.set()
            with pytest.raises(asyncio.CancelledError):
                await creation

        assert [call.args[0].path for call in retain.call_args_list] == [abandoned]

    async def test_cancelled_failed_cleanup_retries_and_the_drain_waits_for_it(self, tmp_path: Path):
        """A cleanup() cancelled while its rmtree worker runs: the cancel waits for the worker,
        and its failed removal goes to a module retry (the folder keeps an owner). The
        shutdown drain does not return while that retry still removes the folder."""
        path = tmp_path / "drained-retry"
        path.mkdir()
        workdir = VmWorkingDirectory("drained-retry", path, home_dir=_HOME_DIR)
        first_entered, release_first = threading.Event(), threading.Event()
        retry_entered, allow_retry = threading.Event(), threading.Event()
        real_rmtree = shutil.rmtree
        attempts = 0

        def fail_then_block(target: Path) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                first_entered.set()
                release_first.wait()
                raise OSError("simulated removal failure")
            retry_entered.set()
            allow_retry.wait()
            real_rmtree(target)

        try:
            with patch("exec_sandbox.vm_working_directory.shutil.rmtree", side_effect=fail_then_block):
                cleanup = asyncio.create_task(workdir.cleanup())
                assert await asyncio.to_thread(first_entered.wait, 1)
                cleanup.cancel()
                release_first.set()
                with pytest.raises(asyncio.CancelledError):
                    await cleanup
                drain = asyncio.create_task(workdir_module.drain_workdir_cleanup_tasks())
                assert await asyncio.to_thread(retry_entered.wait, 1)
                done, _ = await asyncio.wait({drain}, timeout=0.1)
                assert not done  # the retry still removes the folder
                assert path.exists()

                allow_retry.set()
                await asyncio.wait_for(drain, timeout=1)
        finally:
            release_first.set()
            allow_retry.set()

        assert attempts == 2
        assert not path.exists()

    async def test_drain_takes_retries_that_ended_before_their_callback(self):
        """Retry tasks that ended but are still in the module's set (their done callback, which
        removes them, has not run yet): the drain returns and empties the set."""
        ended = [asyncio.create_task(asyncio.sleep(0)) for _ in range(2)]
        await asyncio.gather(*ended)
        workdir_module._WORKDIR_CLEANUP_TASKS.update(ended)  # no done callback: never removed
        try:
            await asyncio.wait_for(workdir_module.drain_workdir_cleanup_tasks(), timeout=1)
            assert not workdir_module._WORKDIR_CLEANUP_TASKS & set(ended)
        finally:
            workdir_module._WORKDIR_CLEANUP_TASKS.difference_update(ended)


class TestVmWorkingDirectoryProperties:
    """Test VmWorkingDirectory path properties."""

    async def test_vm_files_are_distinct_entries_of_the_folder(self):
        """The overlay and each socket are their own entry of the folder: cleanup removes them
        all, and the folder mode guards them. The channel sockets are str (QEMU args)."""
        workdir = await VmWorkingDirectory.create("test-vm-files", home_dir=_HOME_DIR)
        try:
            assert isinstance(workdir.cmd_socket, str)
            assert isinstance(workdir.event_socket, str)
            files = [
                workdir.overlay_image,
                Path(workdir.cmd_socket),
                Path(workdir.event_socket),
                workdir.qmp_socket,
                workdir.qmp_events_socket,
                workdir.gvproxy_socket,
            ]
            assert {f.parent for f in files} == {workdir.path}
            assert len(set(files)) == len(files)
        finally:
            await workdir.cleanup()


class TestVmWorkingDirectorySocketPathLengths:
    """Each socket path fits in sockaddr_un.sun_path (108 bytes on Linux, 104 on macOS)."""

    async def test_every_socket_binds_for_a_long_vm_id(self):
        """The folder name takes 8 characters of the vm_id: the host binds each socket
        (OSError for a path too long), also for a long vm_id."""
        workdir = await VmWorkingDirectory.create("t" * 200, home_dir=_HOME_DIR)
        try:
            for path in (
                workdir.cmd_socket,
                workdir.event_socket,
                str(workdir.qmp_socket),
                str(workdir.qmp_events_socket),
                str(workdir.gvproxy_socket),
            ):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.bind(path)
        finally:
            await workdir.cleanup()


class TestVmWorkingDirectoryCleanup:
    """Test VmWorkingDirectory.cleanup() method."""

    async def test_cleanup_removes_files_inside(self):
        """cleanup() removes files inside the directory."""
        workdir = await VmWorkingDirectory.create("test-vm-cleanup-files", home_dir=_HOME_DIR)
        path = workdir.path
        # Create a test file
        (path / "testfile.txt").write_text("test content")
        assert await workdir.cleanup() is True
        assert not path.exists()

    async def test_cleanup_is_idempotent(self):
        """cleanup() can be called multiple times safely."""
        workdir = await VmWorkingDirectory.create("test-vm-idempotent", home_dir=_HOME_DIR)
        assert await workdir.cleanup() is True
        assert await workdir.cleanup() is True  # Second call succeeds

    async def test_cleanup_returns_true_if_already_removed(self):
        """cleanup() returns True if directory already removed."""
        workdir = await VmWorkingDirectory.create("test-vm-removed", home_dir=_HOME_DIR)
        # Manually remove directory
        shutil.rmtree(workdir.path)
        # cleanup() should still succeed
        result = await workdir.cleanup()
        assert result is True

    async def test_failed_sudo_cleanup_remains_retryable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """With files of qemu-vm, cleanup() removes the folder with `sudo -n rm -rf` (-n: never
        a password prompt). A sudo rm that fails leaves the folder: False; the next cleanup()
        runs it again and removes the folder."""
        path = tmp_path / "retryable"
        path.mkdir()
        workdir = VmWorkingDirectory("retryable", path, home_dir=_HOME_DIR)
        workdir.use_qemu_vm_user = True
        log = _sudo_stand_in(tmp_path, monkeypatch, fail_first=True)

        assert await workdir.cleanup() is False
        assert path.exists()
        assert await workdir.cleanup() is True
        assert not path.exists()
        assert log.read_text().splitlines() == [f"-n rm -rf -- {path}"] * 2


class TestVmWorkingDirectoryQemuVmUser:
    """Test use_qemu_vm_user flag handling."""

    async def test_use_qemu_vm_user_defaults_false(self):
        """use_qemu_vm_user defaults to False."""
        workdir = await VmWorkingDirectory.create("test-vm-user-default", home_dir=_HOME_DIR)
        try:
            assert workdir.use_qemu_vm_user is False
        finally:
            await workdir.cleanup()


class TestHomeDisk:
    """The home disk lives in home_dir (host disk), named after the workdir."""

    async def test_home_disk_outside_workdir(self, tmp_path: Path) -> None:
        """A .img file of home_dir (admission counts the .img files there), one for each
        workdir, also for the same vm_id."""
        first = await VmWorkingDirectory.create("test-vm-home", home_dir=tmp_path)
        second = await VmWorkingDirectory.create("test-vm-home", home_dir=tmp_path)
        try:
            assert first.home_disk.parent == tmp_path
            assert first.home_disk.suffix == ".img"
            assert first.home_disk != second.home_disk
        finally:
            await first.cleanup()
            await second.cleanup()

    async def test_cleanup_removes_home_disk(self, tmp_path: Path) -> None:
        """cleanup() removes the home disk in home_dir and the working folder."""
        workdir = await VmWorkingDirectory.create("test-vm-home-cleanup", home_dir=tmp_path)
        workdir.home_disk.write_bytes(b"disk")
        assert await workdir.cleanup() is True
        assert not workdir.home_disk.exists()
        assert not workdir.path.exists()


class TestQemuVmFolder:
    """VmManager._apply_overlay_permissions on a real VM folder: on the qemu-vm path the
    folder gets group qemu-vm and mode 2770 (other users: no access) after the overlay
    grant. The qemu-vm group is a stand-in: a group this process is in. The file grants
    are stand-ins that make the folder a+x, as ensure_traversable does for the overlay.
    No public path sets up this case on a host without the qemu-vm user: create_vm then
    starts QEMU as qemu-vm (TestQemuVmFolderReal does that on a Linux host with the user)."""

    @staticmethod
    async def _apply(manager: VmManager, workdir: VmWorkingDirectory, *, gid: int) -> bool:
        async def grant(path: Path, *, writable: bool) -> bool:
            if writable:  # the overlay: ensure_traversable makes its folder a+x
                path.parent.chmod(stat.S_IMODE(path.parent.stat().st_mode) | 0o111)
            return True

        workdir.overlay_image.touch()
        with (
            patch("exec_sandbox.vm_manager.probe_qemu_as_qemu_vm", AsyncMock(return_value=True)),
            patch("exec_sandbox.vm_manager.grant_qemu_vm_file_access", side_effect=grant),
            patch("exec_sandbox.permission_utils.get_qemu_vm_gid", return_value=gid),
        ):
            return await manager._apply_overlay_permissions(Path("/base.qcow2"), workdir.overlay_image)  # pyright: ignore[reportPrivateUsage]

    async def test_folder_gets_group_qemu_vm_and_mode_2770(
        self, unit_test_vm_manager: VmManager, tmp_path: Path
    ) -> None:
        workdir = await VmWorkingDirectory.create("test-vm-qemu-vm-folder", home_dir=tmp_path)
        try:
            gid = next((g for g in sorted({os.getegid(), *os.getgroups()}) if g != workdir.path.stat().st_gid), None)
            if gid is None:
                pytest.skip("this process is in one group only")
            assert await self._apply(unit_test_vm_manager, workdir, gid=gid) is True
            st = workdir.path.stat()
            assert (stat.S_IMODE(st.st_mode), st.st_gid) == (0o2770, gid)
        finally:
            await workdir.cleanup()


async def _as_nobody(*command: str) -> int | None:
    """Exit code of `command` run as the user nobody (sudo -n), or None where this
    process may not run commands as nobody."""
    proc = await asyncio.create_subprocess_exec(
        "sudo", "-n", "-u", "nobody", *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 and stderr.startswith(b"sudo"):
        return None
    return proc.returncode


@skip_unless_linux
class TestQemuVmFolderReal:
    """A VM whose QEMU runs as qemu-vm (the CI setup: qemu-vm user, sudoers rule, this
    process in group qemu-vm through `sg qemu-vm`). One VM: QEMU makes its sockets in
    the 2770 folder, this process makes the gvproxy socket there and connects to QEMU's
    sockets, and another user (nobody) cannot remove or replace a socket."""

    _SPEC = VmSpec(language=Language.PYTHON, allow_network=True)  # gvproxy: a socket of this process

    @pytest.fixture(autouse=True)
    async def _qemu_vm_host(self) -> None:
        if not await probe_qemu_as_qemu_vm():
            pytest.skip("this process does not run QEMU as qemu-vm here (needs the qemu-vm setup and its group)")

    async def test_folder_is_group_qemu_vm_2770_and_closed_to_other_users(self, vm_manager: VmManager) -> None:
        as_nobody = await _as_nobody("true") is not None
        vm = await vm_manager.create_vm(
            self._SPEC,
            tenant_id="workdir",
            task_id="mode",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
        )
        folder = vm.workdir.path
        cmd_socket, planted = Path(vm.workdir.cmd_socket), folder / "planted"
        try:
            st = folder.stat()
            assert (stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid) == (0o2770, os.getuid(), get_qemu_vm_gid())
            qemu_sockets = [
                cmd_socket,
                Path(vm.workdir.event_socket),
                vm.workdir.qmp_socket,
                vm.workdir.qmp_events_socket,
            ]
            assert {(s.stat().st_uid, s.stat().st_gid) for s in qemu_sockets} == {
                (get_qemu_vm_uid(), get_qemu_vm_gid())
            }
            gvproxy = vm.workdir.gvproxy_socket.stat()
            assert (gvproxy.st_uid, gvproxy.st_gid) == (os.getuid(), get_qemu_vm_gid())
            if as_nobody:
                # nobody reaches the folder (the overlay grant makes the folders above it
                # a+x): the folder mode itself refuses the changes
                assert await _as_nobody("/bin/sh", "-c", 'cd "$1"', "sh", str(folder.parent)) == 0
                assert await _as_nobody("rm", "-f", str(cmd_socket)) == 1
                assert await _as_nobody("mkdir", str(planted)) == 1
                assert cmd_socket.is_socket()
                assert not planted.exists()
            result = await vm.execute("print(6 * 7)", timeout_seconds=60, env_vars=None, on_stdout=None, on_stderr=None)
            assert (result.exit_code, result.stdout.strip()) == (0, "42")
        finally:
            assert await vm_manager.destroy_vm(vm) is True
        assert not folder.exists()
        if not as_nobody:
            pytest.skip("checked the folder, not another user's access: needs `sudo -n -u nobody`")

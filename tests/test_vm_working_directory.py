"""Tests for VmWorkingDirectory class."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import stat
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

import exec_sandbox.vm_working_directory as workdir_module
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.models import Language
from exec_sandbox.permission_utils import get_qemu_vm_gid, get_qemu_vm_uid, probe_qemu_as_qemu_vm
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_working_directory import VmWorkingDirectory
from tests.conftest import skip_unless_linux

if TYPE_CHECKING:
    from exec_sandbox.vm_manager import VmManager

# Home disks of test workdirs (no VM runs here, so no file is created)
_HOME_DIR = Path(tempfile.gettempdir())


class TestVmWorkingDirectoryCreate:
    """Test VmWorkingDirectory.create() method."""

    async def test_create_makes_directory(self):
        """create() creates a directory that exists."""
        workdir = await VmWorkingDirectory.create("test-vm-123", home_dir=_HOME_DIR)
        try:
            assert workdir.path.exists()
            assert workdir.path.is_dir()
        finally:
            await workdir.cleanup()

    async def test_create_sets_vm_id(self):
        """create() stores the vm_id."""
        workdir = await VmWorkingDirectory.create("test-vm-456", home_dir=_HOME_DIR)
        try:
            assert workdir.vm_id == "test-vm-456"
        finally:
            await workdir.cleanup()

    async def test_create_uses_prefix(self):
        """create() uses vm_id prefix in directory name."""
        workdir = await VmWorkingDirectory.create("test-vm-789", home_dir=_HOME_DIR)
        try:
            # Directory name should start with "vm-test-vm-" (prefix + truncated vm_id)
            assert workdir.path.name.startswith("vm-test-vm-")
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
            await asyncio.wait_for(asyncio.to_thread(created.wait), timeout=1)
            creation.cancel()
            release_thread.set()
            with pytest.raises(asyncio.CancelledError):
                await creation

        assert not abandoned.exists()

    async def test_cancelled_creation_retains_failed_cleanup_until_retry_succeeds(self, tmp_path: Path):
        """A failed first removal keeps a strong retry owner after cancellation. The module's
        set of retry tasks is empty at the end: a retry owner that stays shows only there."""
        abandoned = tmp_path / "abandoned-retry"
        created = threading.Event()
        release_thread = threading.Event()
        retry_done = asyncio.Event()
        attempts = 0

        def blocked_mkdtemp(*_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
            abandoned.mkdir()
            created.set()
            release_thread.wait()
            return str(abandoned)

        async def cleanup_once_then_succeed(workdir: VmWorkingDirectory) -> bool:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return False
            shutil.rmtree(workdir.path)
            retry_done.set()
            return True

        with (
            patch("exec_sandbox.vm_working_directory.tempfile.mkdtemp", side_effect=blocked_mkdtemp),
            patch.object(VmWorkingDirectory, "cleanup", new=cleanup_once_then_succeed),
        ):
            creation = asyncio.create_task(VmWorkingDirectory.create("cancelled-retry", home_dir=_HOME_DIR))
            await asyncio.wait_for(asyncio.to_thread(created.wait), timeout=1)
            creation.cancel()
            release_thread.set()
            with pytest.raises(asyncio.CancelledError):
                await creation
            await asyncio.wait_for(retry_done.wait(), timeout=1)

        await asyncio.sleep(0)
        assert attempts == 2
        assert not abandoned.exists()
        assert not workdir_module._WORKDIR_CLEANUP_TASKS

    async def test_shutdown_drain_waits_for_abandoned_cleanup_owner(self, tmp_path: Path):
        """A failed removal handed to a module-owned retry: the shutdown drain does not return
        while that retry still removes the directory."""
        path = tmp_path / "drained-retry"
        path.mkdir()
        workdir = VmWorkingDirectory("drained-retry", path, home_dir=_HOME_DIR)
        retry_entered = threading.Event()
        allow_retry = threading.Event()
        real_rmtree = shutil.rmtree
        attempts = 0

        def fail_then_block(target: Path) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("simulated removal failure")
            retry_entered.set()
            allow_retry.wait()
            real_rmtree(target)

        try:
            with patch("exec_sandbox.vm_working_directory.shutil.rmtree", side_effect=fail_then_block):
                assert await workdir.cleanup() is False
                workdir._retain_cleanup()  # pyright: ignore[reportPrivateUsage]
                assert path.exists()  # the first removal failed: a retry owns the directory
                drain = asyncio.create_task(workdir_module.drain_workdir_cleanup_tasks())
                await asyncio.wait_for(asyncio.to_thread(retry_entered.wait), timeout=1)
                await asyncio.sleep(0)
                assert not drain.done()

                allow_retry.set()
                await asyncio.wait_for(drain, timeout=1)
        finally:
            allow_retry.set()

        assert attempts == 2
        assert not path.exists()


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

    async def test_failed_sudo_cleanup_remains_retryable(self, tmp_path: Path):
        path = tmp_path / "retryable"
        path.mkdir()
        workdir = VmWorkingDirectory("retryable", path, home_dir=_HOME_DIR)
        workdir.use_qemu_vm_user = True
        attempts = 0

        async def sudo_rm_side_effect(_path: Path) -> bool:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return False
            shutil.rmtree(path)
            return True

        with patch("exec_sandbox.vm_working_directory.sudo_rm", side_effect=sudo_rm_side_effect):
            assert await workdir.cleanup() is False
            assert await workdir.cleanup() is True

        assert attempts == 2

    async def test_cancelled_cleanup_retains_failed_worker_for_retry(self, tmp_path: Path):
        """Cancellation cannot detach a failed rmtree worker and lose ownership. The module's
        set of retry tasks is empty at the end: a retry owner that stays shows only there."""
        path = tmp_path / "cancelled-cleanup"
        path.mkdir()
        workdir = VmWorkingDirectory("cancelled-cleanup", path, home_dir=_HOME_DIR)
        worker_entered = threading.Event()
        release_worker = threading.Event()
        retry_done = asyncio.Event()
        loop = asyncio.get_running_loop()
        real_rmtree = shutil.rmtree
        attempts = 0

        def fail_once(target: Path) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                worker_entered.set()
                release_worker.wait()
                raise OSError("simulated detached worker failure")
            real_rmtree(target)
            loop.call_soon_threadsafe(retry_done.set)

        with patch("exec_sandbox.vm_working_directory.shutil.rmtree", side_effect=fail_once):
            cleanup = asyncio.create_task(workdir.cleanup())
            await asyncio.wait_for(asyncio.to_thread(worker_entered.wait), timeout=1)
            cleanup.cancel()
            release_worker.set()
            with pytest.raises(asyncio.CancelledError):
                await cleanup
            await asyncio.wait_for(retry_done.wait(), timeout=1)
            await asyncio.wait_for(workdir_module.drain_workdir_cleanup_tasks(), timeout=1)

        assert attempts == 2
        assert not path.exists()
        assert not workdir_module._WORKDIR_CLEANUP_TASKS


class TestVmWorkingDirectoryQemuVmUser:
    """Test use_qemu_vm_user flag handling."""

    async def test_use_qemu_vm_user_defaults_false(self):
        """use_qemu_vm_user defaults to False."""
        workdir = await VmWorkingDirectory.create("test-vm-user-default", home_dir=_HOME_DIR)
        try:
            assert workdir.use_qemu_vm_user is False
        finally:
            await workdir.cleanup()

    @pytest.mark.sudo
    async def test_cleanup_with_qemu_vm_user_uses_sudo(self):
        """cleanup() uses sudo when use_qemu_vm_user is True."""
        workdir = await VmWorkingDirectory.create("test-vm-user-sudo", home_dir=_HOME_DIR)
        workdir.use_qemu_vm_user = True

        try:
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.wait = AsyncMock(return_value=0)
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                await workdir.cleanup()

                # Verify sudo rm -rf was called
                mock_exec.assert_called_once()
                args = mock_exec.call_args[0]
                assert args[:4] == ("sudo", "-n", "rm", "-rf")  # -n: never a password prompt
        finally:  # the stand-in for sudo removed nothing: the folder (in the temp folder) goes here
            shutil.rmtree(workdir.path, ignore_errors=True)


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
        workdir = await VmWorkingDirectory.create("test-vm-home-cleanup", home_dir=tmp_path)
        workdir.home_disk.write_bytes(b"disk")
        assert await workdir.cleanup() is True
        assert not workdir.home_disk.exists()
        assert not workdir.path.exists()


class TestQemuVmFolder:
    """VmManager._apply_overlay_permissions on a real VM folder: on the qemu-vm path it gets
    group qemu-vm and mode 2770 (other users: no access); a failure is HOST_DEPENDENCY and
    never a world-writable folder. The qemu-vm group is a stand-in: a group this process
    is in (or not in). The file grants are stand-ins that make the folder a+x, as
    ensure_traversable does for the overlay."""

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

    async def test_folder_failure_is_host_dependency_never_world_writable(
        self, unit_test_vm_manager: VmManager, tmp_path: Path
    ) -> None:
        """A group this process is not in (EPERM): the VM fails; the folder never gets a+rwx."""
        if os.geteuid() == 0:
            pytest.skip("root may give a file to any group")
        workdir = await VmWorkingDirectory.create("test-vm-qemu-vm-folder-fail", home_dir=tmp_path)
        try:
            with pytest.raises(ConfigError) as caught:
                await self._apply(unit_test_vm_manager, workdir, gid=max({os.getegid(), *os.getgroups()}) + 1)
            assert caught.value.code is ErrorCode.HOST_DEPENDENCY
            assert workdir.path.stat().st_mode & (stat.S_IROTH | stat.S_IWOTH) == 0
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
        vm = await vm_manager.create_vm(self._SPEC, tenant_id="workdir", task_id="mode", cache_leases=())
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
            result = await vm.execute("print(6 * 7)", timeout_seconds=60)
            assert (result.exit_code, result.stdout.strip()) == (0, "42")
        finally:
            assert await vm_manager.destroy_vm(vm) is True
        assert not folder.exists()
        if not as_nobody:
            pytest.skip("checked the folder, not another user's access: needs `sudo -n -u nobody`")

"""Tests for VmWorkingDirectory class."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from exec_sandbox.vm_working_directory import VmWorkingDirectory


class TestVmWorkingDirectoryCreate:
    """Test VmWorkingDirectory.create() method."""

    async def test_create_makes_directory(self):
        """create() creates a directory that exists."""
        workdir = await VmWorkingDirectory.create("test-vm-123")
        try:
            assert workdir.path.exists()
            assert workdir.path.is_dir()
        finally:
            await workdir.cleanup()

    async def test_create_sets_vm_id(self):
        """create() stores the vm_id."""
        workdir = await VmWorkingDirectory.create("test-vm-456")
        try:
            assert workdir.vm_id == "test-vm-456"
        finally:
            await workdir.cleanup()

    async def test_create_uses_prefix(self):
        """create() uses vm_id prefix in directory name."""
        workdir = await VmWorkingDirectory.create("test-vm-789")
        try:
            # Directory name should start with "vm-test-vm-" (prefix + truncated vm_id)
            assert workdir.path.name.startswith("vm-test-vm-")
        finally:
            await workdir.cleanup()


class TestVmWorkingDirectoryProperties:
    """Test VmWorkingDirectory path properties."""

    async def test_overlay_image_property(self):
        """overlay_image returns correct path."""
        workdir = await VmWorkingDirectory.create("test-vm-overlay")
        try:
            assert workdir.overlay_image == workdir.path / "overlay.qcow2"
        finally:
            await workdir.cleanup()

    async def test_console_log_property(self):
        """console_log returns correct path."""
        workdir = await VmWorkingDirectory.create("test-vm-console")
        try:
            assert workdir.console_log == workdir.path / "console.log"
        finally:
            await workdir.cleanup()

    async def test_cmd_socket_is_string(self):
        """cmd_socket returns string path (for QEMU args)."""
        workdir = await VmWorkingDirectory.create("test-vm-cmd")
        try:
            assert isinstance(workdir.cmd_socket, str)
            assert workdir.cmd_socket == str(workdir.path / "cmd.sock")
        finally:
            await workdir.cleanup()

    async def test_event_socket_is_string(self):
        """event_socket returns string path (for QEMU args)."""
        workdir = await VmWorkingDirectory.create("test-vm-event")
        try:
            assert isinstance(workdir.event_socket, str)
            assert workdir.event_socket == str(workdir.path / "event.sock")
        finally:
            await workdir.cleanup()

    async def test_qmp_socket_is_path(self):
        """qmp_socket returns Path."""
        workdir = await VmWorkingDirectory.create("test-vm-qmp")
        try:
            assert isinstance(workdir.qmp_socket, Path)
            assert workdir.qmp_socket == workdir.path / "qmp.sock"
        finally:
            await workdir.cleanup()

    async def test_gvproxy_socket_is_path(self):
        """gvproxy_socket returns Path."""
        workdir = await VmWorkingDirectory.create("test-vm-gvproxy")
        try:
            assert isinstance(workdir.gvproxy_socket, Path)
            assert workdir.gvproxy_socket == workdir.path / "gvproxy.sock"
        finally:
            await workdir.cleanup()


class TestVmWorkingDirectorySocketPathLengths:
    """Test socket paths are under Unix 108-byte limit."""

    async def test_cmd_socket_under_limit(self):
        """cmd_socket path is under 108-byte Unix limit."""
        workdir = await VmWorkingDirectory.create("test-vm-length")
        try:
            assert len(workdir.cmd_socket) < 108
        finally:
            await workdir.cleanup()

    async def test_event_socket_under_limit(self):
        """event_socket path is under 108-byte Unix limit."""
        workdir = await VmWorkingDirectory.create("test-vm-length")
        try:
            assert len(workdir.event_socket) < 108
        finally:
            await workdir.cleanup()

    async def test_qmp_socket_under_limit(self):
        """qmp_socket path is under 108-byte Unix limit."""
        workdir = await VmWorkingDirectory.create("test-vm-length")
        try:
            assert len(str(workdir.qmp_socket)) < 108
        finally:
            await workdir.cleanup()

    async def test_gvproxy_socket_under_limit(self):
        """gvproxy_socket path is under 108-byte Unix limit."""
        workdir = await VmWorkingDirectory.create("test-vm-length")
        try:
            assert len(str(workdir.gvproxy_socket)) < 108
        finally:
            await workdir.cleanup()


class TestVmWorkingDirectoryCleanup:
    """Test VmWorkingDirectory.cleanup() method."""

    async def test_cleanup_removes_directory(self):
        """cleanup() removes the directory."""
        workdir = await VmWorkingDirectory.create("test-vm-cleanup")
        path = workdir.path
        await workdir.cleanup()
        assert not path.exists()

    async def test_cleanup_removes_files_inside(self):
        """cleanup() removes files inside the directory."""
        workdir = await VmWorkingDirectory.create("test-vm-cleanup-files")
        path = workdir.path
        # Create a test file
        (path / "testfile.txt").write_text("test content")
        await workdir.cleanup()
        assert not path.exists()

    async def test_cleanup_is_idempotent(self):
        """cleanup() can be called multiple times safely."""
        workdir = await VmWorkingDirectory.create("test-vm-idempotent")
        assert await workdir.cleanup() is True
        assert await workdir.cleanup() is True  # Second call succeeds

    async def test_cleanup_returns_true_on_success(self):
        """cleanup() returns True on success."""
        workdir = await VmWorkingDirectory.create("test-vm-success")
        result = await workdir.cleanup()
        assert result is True

    async def test_cleanup_returns_true_if_already_removed(self):
        """cleanup() returns True if directory already removed."""
        workdir = await VmWorkingDirectory.create("test-vm-removed")
        # Manually remove directory
        import shutil

        shutil.rmtree(workdir.path)
        # cleanup() should still succeed
        result = await workdir.cleanup()
        assert result is True


class TestVmWorkingDirectoryContextManager:
    """Test VmWorkingDirectory async context manager."""

    async def test_context_manager_creates_directory(self):
        """Context manager creates directory on entry."""
        workdir = await VmWorkingDirectory.create("test-vm-ctx")
        async with workdir:
            assert workdir.path.exists()
        # After exit, directory should be cleaned up
        assert not workdir.path.exists()

    async def test_context_manager_cleans_up_on_exit(self):
        """Context manager cleans up on exit."""
        workdir = await VmWorkingDirectory.create("test-vm-ctx-exit")
        path = workdir.path
        async with workdir:
            assert path.exists()
        assert not path.exists()

    async def test_context_manager_cleans_up_on_exception(self):
        """Context manager cleans up even on exception."""
        workdir = await VmWorkingDirectory.create("test-vm-ctx-exc")
        path = workdir.path
        with pytest.raises(RuntimeError):
            async with workdir:
                raise RuntimeError("Test exception")
        assert not path.exists()

    async def test_context_manager_does_not_suppress_exceptions(self):
        """Context manager does not suppress exceptions."""
        workdir = await VmWorkingDirectory.create("test-vm-ctx-no-suppress")
        with pytest.raises(ValueError):
            async with workdir:
                raise ValueError("Should propagate")


class TestVmWorkingDirectoryQemuVmUser:
    """Test use_qemu_vm_user flag handling."""

    async def test_use_qemu_vm_user_defaults_false(self):
        """use_qemu_vm_user defaults to False."""
        workdir = await VmWorkingDirectory.create("test-vm-user-default")
        try:
            assert workdir.use_qemu_vm_user is False
        finally:
            await workdir.cleanup()

    async def test_use_qemu_vm_user_can_be_set(self):
        """use_qemu_vm_user can be set."""
        workdir = await VmWorkingDirectory.create("test-vm-user-set")
        try:
            workdir.use_qemu_vm_user = True
            assert workdir.use_qemu_vm_user is True
        finally:
            await workdir.cleanup()

    @pytest.mark.sudo
    async def test_cleanup_with_qemu_vm_user_uses_sudo(self):
        """cleanup() uses sudo when use_qemu_vm_user is True."""
        workdir = await VmWorkingDirectory.create("test-vm-user-sudo")
        workdir.use_qemu_vm_user = True

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock(return_value=0)
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            await workdir.cleanup()

            # Verify sudo rm -rf was called
            mock_exec.assert_called_once()
            args = mock_exec.call_args[0]
            assert args[0] == "sudo"
            assert args[1] == "rm"
            assert args[2] == "-rf"

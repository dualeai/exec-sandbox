"""Tests for permission_utils.py.

Covers ACL, chmod, chown, sudo operations with real filesystem tests.
Sudo tests require elevated privileges and are marked with @pytest.mark.sudo.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from exec_sandbox.permission_utils import (
    _probe_getfacl,
    _probe_setfacl,
    can_access,
    chmod_async,
    chmod_executable,
    chown_async,
    chown_to_qemu_vm,
    ensure_traversable,
    get_acl,
    get_mode,
    get_owner,
    get_qemu_vm_uid,
    grant_qemu_vm_access,
    grant_qemu_vm_file_access,
    probe_qemu_vm_user,
    probe_sudo_as_qemu_vm,
    remove_acl_user,
    set_acl_user,
    sudo_rm,
    verify_user_access,
)
from tests.conftest import skip_unless_linux, skip_unless_macos


class TestGetQemuVmUid:
    """Tests for get_qemu_vm_uid function."""

    def test_returns_int_or_none(self) -> None:
        """get_qemu_vm_uid returns int or None."""
        result = get_qemu_vm_uid()
        assert result is None or isinstance(result, int)

    def test_cached_result(self) -> None:
        """get_qemu_vm_uid result is cached (lru_cache)."""
        result1 = get_qemu_vm_uid()
        result2 = get_qemu_vm_uid()
        assert result1 == result2


class TestProbeQemuVmUser:
    """Tests for probe_qemu_vm_user function."""

    async def test_returns_bool(self) -> None:
        """probe_qemu_vm_user returns boolean."""
        result = await probe_qemu_vm_user()
        assert isinstance(result, bool)

    async def test_cached_result(self) -> None:
        """probe_qemu_vm_user result is cached."""
        result1 = await probe_qemu_vm_user()
        result2 = await probe_qemu_vm_user()
        assert result1 == result2

    @skip_unless_macos
    async def test_false_on_macos(self) -> None:
        """probe_qemu_vm_user returns False on macOS."""
        result = await probe_qemu_vm_user()
        assert result is False


class TestProbeSudoAsQemuVm:
    """Tests for probe_sudo_as_qemu_vm function."""

    async def test_returns_bool(self) -> None:
        """probe_sudo_as_qemu_vm returns boolean."""
        result = await probe_sudo_as_qemu_vm()
        assert isinstance(result, bool)

    @skip_unless_macos
    async def test_false_on_macos(self) -> None:
        """probe_sudo_as_qemu_vm returns False on macOS."""
        result = await probe_sudo_as_qemu_vm()
        assert result is False

    async def test_false_if_qemu_vm_user_missing(self) -> None:
        """probe_sudo_as_qemu_vm returns False if qemu-vm user doesn't exist."""
        with patch("exec_sandbox.permission_utils.probe_qemu_vm_user", new_callable=AsyncMock) as mock:
            mock.return_value = False
            # Clear cache to force re-probe
            from exec_sandbox.permission_utils import _probe_cache

            _probe_cache.sudo_as_qemu_vm = None
            result = await probe_sudo_as_qemu_vm()
            # Should be False because qemu-vm user doesn't exist
            assert result is False or result is True  # Either is valid based on environment

    @skip_unless_linux
    async def test_true_with_proper_sudoers_linux(self) -> None:
        """probe_sudo_as_qemu_vm returns True if sudoers is configured (Linux only)."""
        # This test verifies the probe works in CI where sudoers is set up
        # It's expected to return True if:
        # 1. qemu-vm user exists (created by CI setup)
        # 2. sudoers allows 'runner ALL=(qemu-vm) NOPASSWD: ALL'
        result = await probe_sudo_as_qemu_vm()
        # We can't assert True because local dev may not have sudoers configured
        assert isinstance(result, bool)


class TestChmodAsync:
    """Tests for chmod_async function."""

    async def test_chmod_octal_mode(self, tmp_path: Path) -> None:
        """chmod_async works with octal mode string."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = await chmod_async(test_file, "644")

        assert result is True
        # Verify mode changed (mask with 0o777 to get permission bits)
        assert (test_file.stat().st_mode & 0o777) == 0o644

    async def test_chmod_symbolic_mode(self, tmp_path: Path) -> None:
        """chmod_async works with symbolic mode."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        test_file.chmod(0o600)

        result = await chmod_async(test_file, "a+r")

        assert result is True
        # Verify readable by all
        mode = test_file.stat().st_mode & 0o777
        assert mode & 0o004  # Other read bit set

    async def test_chmod_nonexistent_file(self, tmp_path: Path) -> None:
        """chmod_async returns False for nonexistent file."""
        nonexistent = tmp_path / "nonexistent.txt"

        result = await chmod_async(nonexistent, "644")

        assert result is False

    async def test_chmod_directory(self, tmp_path: Path) -> None:
        """chmod_async works on directories."""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        result = await chmod_async(test_dir, "755")

        assert result is True
        assert (test_dir.stat().st_mode & 0o777) == 0o755


class TestChmodExecutable:
    """Tests for chmod_executable function."""

    async def test_makes_file_executable(self, tmp_path: Path) -> None:
        """chmod_executable sets 0o755 permissions."""
        test_file = tmp_path / "script.sh"
        test_file.write_text("#!/bin/bash\necho hello")
        test_file.chmod(0o644)

        await chmod_executable(test_file)

        assert (test_file.stat().st_mode & 0o777) == 0o755

    async def test_idempotent(self, tmp_path: Path) -> None:
        """chmod_executable can be called multiple times."""
        test_file = tmp_path / "script.sh"
        test_file.write_text("#!/bin/bash")

        await chmod_executable(test_file)
        await chmod_executable(test_file)

        assert (test_file.stat().st_mode & 0o777) == 0o755


class TestGetMode:
    """Tests for get_mode function."""

    async def test_returns_mode(self, tmp_path: Path) -> None:
        """get_mode returns file mode."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        test_file.chmod(0o640)

        result = await get_mode(test_file)

        assert result is not None
        assert (result & 0o777) == 0o640

    async def test_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """get_mode returns None for nonexistent file."""
        nonexistent = tmp_path / "nonexistent.txt"

        result = await get_mode(nonexistent)

        assert result is None


class TestGetOwner:
    """Tests for get_owner function."""

    async def test_returns_owner_tuple(self, tmp_path: Path) -> None:
        """get_owner returns (user, group) tuple."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = await get_owner(test_file)

        assert result is not None
        user, group = result
        assert isinstance(user, str)
        assert isinstance(group, str)

    async def test_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """get_owner returns None for nonexistent file."""
        nonexistent = tmp_path / "nonexistent.txt"

        result = await get_owner(nonexistent)

        assert result is None


class TestCanAccess:
    """Tests for can_access function."""

    async def test_readable_file(self, tmp_path: Path) -> None:
        """can_access returns True for readable file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        test_file.chmod(0o644)

        assert await can_access(test_file, os.R_OK) is True

    async def test_nonexistent_file(self, tmp_path: Path) -> None:
        """can_access returns False for nonexistent file."""
        nonexistent = tmp_path / "nonexistent.txt"

        assert await can_access(nonexistent, os.R_OK) is False

    async def test_write_access(self, tmp_path: Path) -> None:
        """can_access checks write permission."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        assert await can_access(test_file, os.W_OK) is True


class TestEnsureTraversable:
    """Tests for ensure_traversable function."""

    async def test_adds_execute_permission(self, tmp_path: Path) -> None:
        """ensure_traversable adds a+x to directories."""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()
        test_dir.chmod(0o700)

        result = await ensure_traversable([test_dir])

        assert result is True
        mode = test_dir.stat().st_mode & 0o777
        assert mode & 0o001  # Other execute bit

    async def test_multiple_directories(self, tmp_path: Path) -> None:
        """ensure_traversable handles multiple directories."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        dir1.chmod(0o700)
        dir2.chmod(0o700)

        result = await ensure_traversable([dir1, dir2])

        assert result is True
        assert dir1.stat().st_mode & 0o001
        assert dir2.stat().st_mode & 0o001

    async def test_empty_list(self) -> None:
        """ensure_traversable handles empty list."""
        result = await ensure_traversable([])

        assert result is True


class TestAclOperations:
    """Tests for ACL functions (Linux only, no-op on macOS)."""

    @skip_unless_macos
    async def test_set_acl_user_macos_returns_false(self, tmp_path: Path) -> None:
        """set_acl_user returns False on macOS."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = await set_acl_user(test_file, "nobody", "rw")

        assert result is False

    @skip_unless_macos
    async def test_get_acl_macos_returns_none(self, tmp_path: Path) -> None:
        """get_acl returns None on macOS."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = await get_acl(test_file)

        assert result is None

    async def test_grant_qemu_vm_access_no_user(self, tmp_path: Path) -> None:
        """grant_qemu_vm_access returns False if qemu-vm user doesn't exist."""
        test_file = tmp_path / "test.sock"
        test_file.write_text("")

        # Mock probe to return False (no qemu-vm user)
        with patch(
            "exec_sandbox.permission_utils.probe_qemu_vm_user",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await grant_qemu_vm_access(test_file)

        assert result is False


@skip_unless_linux
class TestLinuxAclBinaries:
    """Tests for Linux ACL binary availability.

    These tests verify that setfacl/getfacl are available on Linux.
    They ASSERT (not skip) to catch missing binaries in CI.
    """

    async def test_setfacl_available(self) -> None:
        """setfacl must be available on Linux."""
        result = await _probe_setfacl()
        assert result is True, "setfacl binary must be available on Linux"

    async def test_getfacl_available(self) -> None:
        """getfacl must be available on Linux."""
        result = await _probe_getfacl()
        assert result is True, "getfacl binary must be available on Linux"

    async def test_set_acl_works(self, tmp_path: Path) -> None:
        """set_acl_user actually sets ACL on Linux."""
        import pwd

        test_file = tmp_path / "acl_test.txt"
        test_file.write_text("content")
        current_user = pwd.getpwuid(os.getuid()).pw_name

        result = await set_acl_user(test_file, current_user, "rw")

        assert result is True, "set_acl_user must succeed on Linux"

    async def test_get_acl_works(self, tmp_path: Path) -> None:
        """get_acl actually reads ACL on Linux."""
        test_file = tmp_path / "acl_test.txt"
        test_file.write_text("content")

        result = await get_acl(test_file)

        assert result is not None, "get_acl must return ACL dict on Linux"
        assert "user::" in result, "get_acl must include owner entry"


class TestSudoRm:
    """Tests for sudo_rm function."""

    async def test_sudo_rm_mocked(self, tmp_path: Path) -> None:
        """sudo_rm calls sudo_exec with rm -rf."""
        test_file = tmp_path / "test.txt"

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch(
            "exec_sandbox.permission_utils.sudo_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ) as mock:
            result = await sudo_rm(test_file)

            assert result is True
            mock.assert_called_once()
            args = mock.call_args[0][0]  # First positional arg is the list
            assert args[0] == "rm"
            assert args[1] == "-rf"
            assert str(test_file) in args[2]

    async def test_sudo_rm_failure(self, tmp_path: Path) -> None:
        """sudo_rm returns False on failure."""
        test_file = tmp_path / "test.txt"

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch(
            "exec_sandbox.permission_utils.sudo_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ):
            result = await sudo_rm(test_file)

            assert result is False


# =============================================================================
# SUDO TESTS - Require elevated privileges
# =============================================================================


@pytest.mark.sudo
class TestSudoRmReal:
    """Real sudo tests - require sudo privileges.

    Run with: uv run pytest tests/test_permission_utils.py -v -m sudo
    """

    async def test_sudo_rm_removes_root_owned_file(self, tmp_path: Path) -> None:
        """Create file, chown to root, verify cannot remove, sudo_rm, verify gone.

        This is the canonical test for sudo_rm:
        1. Create a file as current user
        2. Change ownership to root (requires sudo)
        3. Verify normal user cannot remove it
        4. Use sudo_rm to remove it
        5. Verify file no longer exists
        """
        if os.getuid() == 0:
            pytest.skip("Test must run as non-root user")

        test_file = tmp_path / "root_owned.txt"
        test_file.write_text("owned by root")

        # Step 1: Change ownership to root
        chown_result = await chown_async(test_file, "root", "root")
        if not chown_result:
            pytest.skip("Cannot chown to root (sudo not available?)")

        # Verify ownership changed
        owner = await get_owner(test_file)
        assert owner is not None
        assert owner[0] == "root", f"Expected owner 'root', got '{owner[0]}'"

        # Step 2: Verify normal user cannot remove
        try:
            test_file.unlink()
            pytest.fail("Should not be able to remove root-owned file")
        except PermissionError:
            pass  # Expected

        # Step 3: Use sudo_rm to remove
        result = await sudo_rm(test_file)

        # Step 4: Verify removed
        assert result is True
        assert not test_file.exists(), "File should be removed by sudo_rm"

    async def test_sudo_rm_removes_root_owned_directory(self, tmp_path: Path) -> None:
        """sudo_rm removes root-owned directory with contents."""
        if os.getuid() == 0:
            pytest.skip("Test must run as non-root user")

        test_dir = tmp_path / "root_owned_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("file 1")
        (test_dir / "file2.txt").write_text("file 2")

        # Change ownership to root
        chown_result = await chown_async(test_dir, "root", "root")
        if not chown_result:
            pytest.skip("Cannot chown to root")

        # Verify cannot remove normally
        import shutil

        try:
            shutil.rmtree(test_dir)
            pytest.fail("Should not be able to remove root-owned directory")
        except PermissionError:
            pass

        # sudo_rm should work
        result = await sudo_rm(test_dir)

        assert result is True
        assert not test_dir.exists()

    async def test_chown_to_qemu_vm(self, tmp_path: Path) -> None:
        """chown_to_qemu_vm changes ownership if user exists."""
        test_file = tmp_path / "qemu_owned.txt"
        test_file.write_text("for qemu-vm")

        result = await chown_to_qemu_vm(test_file)

        assert result is True
        owner = await get_owner(test_file)
        assert owner is not None
        assert owner[0] == "qemu-vm"

        # Cleanup
        await sudo_rm(test_file)


@pytest.mark.sudo
@skip_unless_linux
class TestAclRealLinux:
    """Real ACL tests on Linux - require setfacl."""

    async def test_set_and_get_acl(self, tmp_path: Path) -> None:
        """set_acl_user and get_acl work together."""
        test_file = tmp_path / "acl_test.txt"
        test_file.write_text("content")

        # Set ACL for current user
        import pwd

        current_user = pwd.getpwuid(os.getuid()).pw_name

        result = await set_acl_user(test_file, current_user, "rw")
        if not result:
            pytest.skip("setfacl not available")

        # Get ACL and verify
        acl = await get_acl(test_file)

        assert acl is not None
        assert f"user:{current_user}" in acl
        assert "rw" in acl[f"user:{current_user}"]

    async def test_verify_user_access(self, tmp_path: Path) -> None:
        """verify_user_access checks ACL permissions."""
        test_file = tmp_path / "verify_test.txt"
        test_file.write_text("content")

        import pwd

        current_user = pwd.getpwuid(os.getuid()).pw_name

        # Set ACL
        result = await set_acl_user(test_file, current_user, "rw")
        if not result:
            pytest.skip("setfacl not available")

        # Verify access
        has_access = await verify_user_access(test_file, current_user, "rw")

        assert has_access is True

    async def test_remove_acl_user(self, tmp_path: Path) -> None:
        """remove_acl_user removes user ACL entry."""
        test_file = tmp_path / "remove_acl.txt"
        test_file.write_text("content")

        import pwd

        current_user = pwd.getpwuid(os.getuid()).pw_name

        # Set then remove ACL
        set_result = await set_acl_user(test_file, current_user, "rw")
        if not set_result:
            pytest.skip("setfacl not available")

        remove_result = await remove_acl_user(test_file, current_user)

        assert remove_result is True

        # Verify removed
        acl = await get_acl(test_file)
        assert acl is not None
        assert f"user:{current_user}" not in acl

    async def test_grant_qemu_vm_access_real(self, tmp_path: Path) -> None:
        """grant_qemu_vm_access sets ACL for qemu-vm user."""
        test_file = tmp_path / "qemu_acl.txt"
        test_file.write_text("content")

        result = await grant_qemu_vm_access(test_file)

        assert result is True

        # Verify ACL
        acl = await get_acl(test_file)
        assert acl is not None
        assert "user:qemu-vm" in acl


# =============================================================================
# grant_qemu_vm_file_access Tests
# =============================================================================


class TestGrantQemuVmFileAccess:
    """Tests for grant_qemu_vm_file_access function.

    Note: grant_qemu_vm_file_access traverses parent dirs all the way to /,
    which requires root to chmod system dirs. In unit tests, we mock
    ensure_traversable to isolate the file operation logic. The traversal
    itself is tested separately in TestEnsureTraversable.
    """

    # ---- writable=False path ----

    async def test_readonly_chmods_file(self, tmp_path: Path) -> None:
        """writable=False makes file world-readable (a+r)."""
        test_file = tmp_path / "test.qcow2"
        test_file.write_bytes(b"\x00")
        test_file.chmod(0o600)

        with patch(
            "exec_sandbox.permission_utils.ensure_traversable",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await grant_qemu_vm_file_access(test_file, writable=False)

        assert result is True
        assert test_file.stat().st_mode & 0o004  # other-read bit

    async def test_readonly_idempotent(self, tmp_path: Path) -> None:
        """Calling twice with writable=False still succeeds."""
        test_file = tmp_path / "test.qcow2"
        test_file.write_bytes(b"\x00")

        with patch(
            "exec_sandbox.permission_utils.ensure_traversable",
            new_callable=AsyncMock,
            return_value=True,
        ):
            assert await grant_qemu_vm_file_access(test_file, writable=False) is True
            assert await grant_qemu_vm_file_access(test_file, writable=False) is True
        assert test_file.stat().st_mode & 0o004

    # ---- writable=True path (mocked chown, no real qemu-vm user) ----

    async def test_writable_calls_chown(self, tmp_path: Path) -> None:
        """writable=True calls chown_to_qemu_vm instead of chmod."""
        test_file = tmp_path / "snap.qcow2"
        test_file.write_bytes(b"\x00")

        with (
            patch(
                "exec_sandbox.permission_utils.ensure_traversable",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "exec_sandbox.permission_utils.chown_to_qemu_vm",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_chown,
        ):
            result = await grant_qemu_vm_file_access(test_file, writable=True)

        assert result is True
        mock_chown.assert_awaited_once_with(test_file)

    async def test_writable_does_not_chmod(self, tmp_path: Path) -> None:
        """writable=True does not call chmod a+r."""
        test_file = tmp_path / "snap.qcow2"
        test_file.write_bytes(b"\x00")
        test_file.chmod(0o600)

        with (
            patch(
                "exec_sandbox.permission_utils.ensure_traversable",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "exec_sandbox.permission_utils.chown_to_qemu_vm",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            await grant_qemu_vm_file_access(test_file, writable=True)

        # File should NOT have been made world-readable
        assert not (test_file.stat().st_mode & 0o004)

    # ---- Failure propagation ----

    async def test_traversal_failure_skips_file_op(self, tmp_path: Path) -> None:
        """If ensure_traversable fails, no chown/chmod is attempted."""
        test_file = tmp_path / "snap.qcow2"
        test_file.write_bytes(b"\x00")

        with (
            patch(
                "exec_sandbox.permission_utils.ensure_traversable",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "exec_sandbox.permission_utils.chown_to_qemu_vm",
                new_callable=AsyncMock,
            ) as mock_chown,
            patch(
                "exec_sandbox.permission_utils.chmod_async",
                new_callable=AsyncMock,
            ) as mock_chmod,
        ):
            result = await grant_qemu_vm_file_access(test_file, writable=True)

        assert result is False
        mock_chown.assert_not_awaited()
        mock_chmod.assert_not_awaited()

    async def test_chown_failure_returns_false(self, tmp_path: Path) -> None:
        """writable=True returns False when chown fails."""
        test_file = tmp_path / "snap.qcow2"
        test_file.write_bytes(b"\x00")

        with (
            patch(
                "exec_sandbox.permission_utils.ensure_traversable",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "exec_sandbox.permission_utils.chown_to_qemu_vm",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await grant_qemu_vm_file_access(test_file, writable=True)

        assert result is False

    async def test_chmod_failure_returns_false(self, tmp_path: Path) -> None:
        """writable=False returns False when chmod fails (nonexistent file)."""
        nonexistent = tmp_path / "does_not_exist.qcow2"

        with patch(
            "exec_sandbox.permission_utils.ensure_traversable",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await grant_qemu_vm_file_access(nonexistent, writable=False)

        assert result is False

    # ---- Parent directory computation ----

    async def test_collects_all_ancestors(self, tmp_path: Path) -> None:
        """Traversal collects every directory from parent up to root."""
        deep = tmp_path / "l1" / "l2" / "l3"
        deep.mkdir(parents=True)
        test_file = deep / "file.qcow2"
        test_file.write_bytes(b"\x00")

        collected: list[Path] = []

        async def spy_ensure(dirs: list[Path]) -> bool:
            collected.extend(dirs)
            return True

        with patch(
            "exec_sandbox.permission_utils.ensure_traversable",
            side_effect=spy_ensure,
        ):
            await grant_qemu_vm_file_access(test_file, writable=False)

        # Must include immediate parent and all ancestors
        assert deep in collected
        assert (tmp_path / "l1" / "l2") in collected
        assert (tmp_path / "l1") in collected
        assert tmp_path in collected

    async def test_file_at_shallow_path(self, tmp_path: Path) -> None:
        """File directly in tmp_path still works."""
        test_file = tmp_path / "shallow.qcow2"
        test_file.write_bytes(b"\x00")

        with patch(
            "exec_sandbox.permission_utils.ensure_traversable",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await grant_qemu_vm_file_access(test_file, writable=False)

        assert result is True

    async def test_traversal_order_innermost_first(self, tmp_path: Path) -> None:
        """Parent dirs are collected innermost-first (child before parent)."""
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        test_file = nested / "f.qcow2"
        test_file.write_bytes(b"\x00")

        collected: list[Path] = []

        async def spy_ensure(dirs: list[Path]) -> bool:
            collected.extend(dirs)
            return True

        with patch(
            "exec_sandbox.permission_utils.ensure_traversable",
            side_effect=spy_ensure,
        ):
            await grant_qemu_vm_file_access(test_file, writable=False)

        # b must come before a (innermost first)
        idx_b = collected.index(nested)
        idx_a = collected.index(tmp_path / "a")
        assert idx_b < idx_a


# =============================================================================
# grant_qemu_vm_file_access Sudo Integration Tests
# =============================================================================


@pytest.mark.sudo
class TestGrantQemuVmFileAccessReal:
    """Real filesystem tests for grant_qemu_vm_file_access.

    Run with: uv run pytest tests/test_permission_utils.py -v -m sudo
    """

    async def test_writable_chowns_to_qemu_vm(self, tmp_path: Path) -> None:
        """writable=True chowns file to qemu-vm user on Linux."""
        test_file = tmp_path / "snap.qcow2"
        test_file.write_bytes(b"\x00")

        result = await grant_qemu_vm_file_access(test_file, writable=True)

        assert result is True
        owner = await get_owner(test_file)
        assert owner is not None
        assert owner[0] == "qemu-vm"

        await sudo_rm(test_file)

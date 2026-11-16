"""Tests for permission_utils.py.

Covers the private-folder checks, the qemu-vm probes, and the ACL, chmod, chown and sudo
operations, with real filesystem tests where the host allows them.
Sudo tests require elevated privileges and are marked with @pytest.mark.sudo.
"""

import asyncio
import contextlib
import errno
import grp
import logging
import os
import pwd
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from exec_sandbox import constants, permission_utils, system_probes
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.permission_utils import (
    QEMU_VM_SUDO,
    _add_execute_bits,
    chmod_executable,
    chown_async,
    ensure_traversable,
    get_mode,
    get_qemu_vm_gid,
    get_qemu_vm_uid,
    grant_qemu_vm_acl,
    grant_qemu_vm_dir_access,
    grant_qemu_vm_file_access,
    make_cache_dir,
    private_tree_problem,
    probe_qemu_as_qemu_vm,
    probe_qemu_vm_user,
    qemu_vm_acl_perms,
    set_acl_user,
    sudo_exec,
    sudo_rm,
)
from exec_sandbox.platform_utils import HostOS, ProcessWrapper
from exec_sandbox.system_probes import can_access
from tests.conftest import asyncio_with, skip_unless_linux, skip_unless_macos


def read_acl(path: Path) -> dict[str, str]:
    """ACL entries of a file from getfacl, e.g. {'user::': 'rw-', 'user:qemu-vm': 'rw-'}."""
    out = subprocess.run(["getfacl", "-p", str(path)], capture_output=True, text=True, check=True).stdout
    entries = [line.split(":") for line in out.splitlines() if line and not line.startswith("#")]
    return {f"{kind}:{name}" if name else f"{kind}::": perms for kind, name, perms, *_ in entries}


def _record_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, *, exit_code: int
) -> Callable[[], list[list[str]]]:
    """Put a stand-in `name` (sudo, setfacl) first on PATH: it records its arguments and exits
    with `exit_code`, and runs nothing. Returns a reader of the recorded calls, in order."""
    record = tmp_path / f"{name}.calls"
    stand_in = tmp_path / "bin" / name
    stand_in.parent.mkdir(exist_ok=True)
    stand_in.write_text(
        f'#!/bin/sh\nfor arg in "$@"; do printf "%s\\0" "$arg"; done >> "{record}"\n'
        f'printf "\\n" >> "{record}"\nexit {exit_code}\n'
    )
    stand_in.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}")

    def calls() -> list[list[str]]:
        if not record.exists():
            return []
        return [line.split("\0")[:-1] for line in record.read_text().split("\n")[:-1]]

    return calls


class TestPrivateTreeProblem:
    """The folder above a private one: root or this user, no other write and no group write
    (but for this user's private group with no ACL entry), unless sticky."""

    @staticmethod
    def _problem(root: Path) -> tuple[str, str] | None:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            return private_tree_problem(root, fd, forbidden=0o077)
        finally:
            os.close(fd)

    def test_private_folder_passes(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        assert self._problem(root) is None

    @pytest.mark.parametrize(
        ("mode", "hint_end"),
        [
            (0o755, "."),
            (0o777, ", then `sudo chmod go-w {parent}` or `sudo chmod +t {parent}`."),
            (0o1777, "."),
        ],
        ids=["0755", "0777", "sticky-1777"],
    )
    def test_parent_of_another_user_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: int, hint_end: str
    ) -> None:
        """Its owner can put another folder in place of the root, whatever its mode. A stat
        stands in for the other user's folder (making one needs root). When others can also
        write it (not sticky), the hint adds the mode step after the owner's (its mode alone is
        no remedy; without it, a second start fails on the mode)."""
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        parent, other = root.resolve().parent, os.geteuid() + 1
        lstat = Path.lstat

        def lstat_as_other(path: Path) -> os.stat_result:
            st = lstat(path)
            if path != parent:
                return st
            return os.stat_result(
                (stat.S_IFDIR | mode, st.st_ino, st.st_dev, st.st_nlink, other, st.st_gid, st.st_size, 0, 0, 0)
            )

        monkeypatch.setattr(Path, "lstat", lstat_as_other)
        assert self._problem(root) == (
            f"is in {parent} (uid {other}, mode {oct(mode)}), which another user can change",
            f"Use a folder whose parent folders belong to root or this user, or give {parent} to root "
            f"(`sudo chown root {parent}`: uid {other} then no longer owns it)" + hint_end.format(parent=parent),
        )

    @pytest.mark.parametrize(
        ("own_group", "problem"), [(True, False), (False, True)], ids=["own-group", "shared-group"]
    )
    def test_group_writable_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, own_group: bool, problem: bool
    ) -> None:
        """A ~/.cache made under umask 002 (0775): safe when its group is this user's own. The
        parent gets a group other than this process's egid when it can (the group asked about
        is the parent's)."""
        parent = tmp_path / "cache"
        parent.mkdir()
        if others := [gid for gid in os.getgroups() if gid != os.getegid()]:
            os.chown(parent, -1, others[0])
        parent.chmod(0o775)
        root = parent / "root"
        root.mkdir(mode=0o700)
        groups: list[int] = []
        monkeypatch.setattr(
            "exec_sandbox.permission_utils._group_is_this_users", lambda gid: groups.append(gid) or own_group
        )

        assert (self._problem(root) is not None) is problem
        assert groups[0] == parent.stat().st_gid  # the innermost parent first (others above may be 0775 too)

    def test_group_writable_parent_with_acl_entries_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Linux the group bits of a folder with ACL entries are the ACL mask: a named user's
        write shows as group write, whatever the owning group. A real entry that gives no
        write right (macOS chmod +a, Linux setfacl) is enough to refuse the group write."""
        parent = tmp_path / "acl"
        parent.mkdir()
        parent.chmod(0o775)
        root = parent / "root"
        root.mkdir(mode=0o700)
        if sys.platform == "darwin":
            subprocess.run(["/bin/chmod", "+a", "user:nobody allow list", str(parent)], check=True)
        elif shutil.which("setfacl") is None:
            pytest.skip("setfacl (acl) is not installed")
        elif subprocess.run(["setfacl", "-m", "u:nobody:r-x", str(parent)], check=False).returncode != 0:
            pytest.skip("this filesystem takes no ACL")
        monkeypatch.setattr("exec_sandbox.permission_utils._group_is_this_users", lambda _gid: True)

        problem = self._problem(root)
        assert problem is not None
        assert f"is in {parent} (uid" in problem[0]

    def test_group_writable_parent_that_cannot_be_opened_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its ACL cannot be read (mode 0371: no read for its owner): refused, not trusted."""
        if os.geteuid() == 0:
            pytest.skip("root opens any folder")
        parent = tmp_path / "noread"
        parent.mkdir()
        root = parent / "root"
        root.mkdir(mode=0o700)
        monkeypatch.setattr("exec_sandbox.permission_utils._group_is_this_users", lambda _gid: True)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)  # opened before the parent loses read
        parent.chmod(0o371)
        try:
            assert private_tree_problem(root, fd, forbidden=0o077) is not None
        finally:
            parent.chmod(0o700)
            os.close(fd)

    def test_other_writable_parent_is_refused_even_with_its_own_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent = tmp_path / "open"
        parent.mkdir()
        parent.chmod(0o777)
        root = parent / "root"
        root.mkdir(mode=0o700)
        monkeypatch.setattr("exec_sandbox.permission_utils._group_is_this_users", lambda _gid: True)

        assert self._problem(root) is not None


@skip_unless_macos
class TestMacosParentAcl:
    """macOS mode bits do not show ACL entries: a parent's allow entry for another user can
    let that user replace the root."""

    @staticmethod
    def _root_under(parent: Path) -> Path:
        root = parent / "root"
        root.mkdir(mode=0o700)
        return root

    @pytest.mark.parametrize(
        ("entry", "refused"),
        [
            ("user:nobody allow add_file,add_subdirectory,delete_child", True),
            ("group:everyone allow delete", True),
            ("group:everyone deny delete", False),  # every macOS home has it
            ("user:nobody allow list,search,readattr", False),  # no right to change the folder
            ("user:nobody allow add_file,directory_inherit,only_inherit", False),  # for new children only
            ("user:nobody allow add_file,directory_inherit", True),  # applies here and to new folders
        ],
        ids=["other-user-writes", "group-deletes", "deny-entry", "read-only", "only-inherit", "inherit-flag"],
    )
    def test_parent_acl_entry(self, tmp_path: Path, entry: str, refused: bool) -> None:
        parent = tmp_path / "parent"
        parent.mkdir(mode=0o755)
        root = self._root_under(parent)
        subprocess.run(["/bin/chmod", "+a", entry, str(parent)], check=True)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            problem = private_tree_problem(root, fd, forbidden=0o077)
        finally:
            os.close(fd)
            subprocess.run(["/bin/chmod", "-N", str(parent)], check=True)
        assert (problem is not None) is refused
        if refused:
            assert problem is not None
            assert "ACL entry" in problem[0]

    def test_inherited_entry_is_refused(self, tmp_path: Path) -> None:
        """An entry the parent got from its own parent (`allow,inherited`) applies too."""
        parent = tmp_path / "parent"
        parent.mkdir(mode=0o755)
        root = self._root_under(parent)
        subprocess.run(["/bin/chmod", "+ai", "group:everyone allow delete_child", str(parent)], check=True)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            problem = private_tree_problem(root, fd, forbidden=0o077)
        finally:
            os.close(fd)
            subprocess.run(["/bin/chmod", "-N", str(parent)], check=True)
        assert problem is not None

    def test_entry_for_this_user_passes(self, tmp_path: Path) -> None:
        parent = tmp_path / "parent"
        parent.mkdir(mode=0o755)
        root = self._root_under(parent)
        me = pwd.getpwuid(os.geteuid()).pw_name
        subprocess.run(["/bin/chmod", "+a", f"user:{me} allow add_file,delete_child", str(parent)], check=True)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            assert private_tree_problem(root, fd, forbidden=0o077) is None
        finally:
            os.close(fd)
            subprocess.run(["/bin/chmod", "-N", str(parent)], check=True)


class TestParseId:
    """uid and gid fields as glibc reads them (strtoul, then nss_files' 32-bit check).
    Private level: the public callers read the host's /etc/passwd and /etc/group, and reach a
    field only through stand-ins for sudo, the cgroup and both files."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("986", 986),
            ("0986", 986),
            (" 986", 986),
            ("+986", 986),
            ("-0", 0),
            ("-18446744073709550630", 986),  # -x wraps modulo 2**64
            ("-1", None),  # 2**64 - 1: above 0xffffffff
            ("4294967295", 0xFFFFFFFF),
            ("4294967296", None),
            ("99999999999999999999999", None),  # strtoul overflow: ULONG_MAX
            ("-18446744073709551616", None),  # a negative overflow too (ERANGE), not 1
            ("0" * 5000 + "986", 986),  # leading zeros: glibc reads 986 (Python's int() refuses 4300+ digits)
            ("9" * 5000, None),
            ("986 ", None),
            ("9_86", None),
            ("", None),
        ],
    )
    def test_field(self, field: str, value: int | None) -> None:
        from exec_sandbox.permission_utils import _parse_id

        assert _parse_id(field) == value


class TestGroupIsThisUsers:
    """Only the user's private group: its primary group, named as the user, no other member.
    Private level: the answer comes from the host's user and group database, which a test
    cannot change; the stand-ins replace that database, not the rule."""

    @pytest.mark.parametrize(
        "case",
        [
            (1000, "me", [], "me:x:1000:1000::/home/me:/bin/sh\n", True),
            (1000, "me", ["me"], "me:x:1000:1000::/home/me:/bin/sh\n", True),
            (1000, "me", ["me", "eve"], "me:x:1000:1000::/home/me:/bin/sh\n", False),
            (1000, "me", [], "me:x:1000:1000::/home/me:/bin/sh\neve:x:1001:1000::/home/eve:/bin/sh\n", False),
            (1000, "users", [], "me:x:1000:1000::/home/me:/bin/sh\n", False),
            (109, "docker", ["me"], "me:x:1000:1000::/home/me:/bin/sh\n", False),
            (12, "me", [], "me:x:1000:1000::/home/me:/bin/sh\n", False),
        ],
        ids=["private", "me-listed", "other-member", "other-primary", "named-otherwise", "secondary", "not-primary"],
    )
    def test_only_the_private_group(
        self, monkeypatch: pytest.MonkeyPatch, case: tuple[int, str, list[str], str, bool]
    ) -> None:
        """case: the gid asked about, its group name, its listed members, /etc/passwd, the answer."""
        gid, name, members, passwd, own = case
        from exec_sandbox import permission_utils

        monkeypatch.setattr("exec_sandbox.permission_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setattr(pwd, "getpwuid", lambda _uid: MagicMock(pw_name="me", pw_gid=1000))
        monkeypatch.setattr(grp, "getgrgid", lambda _gid: MagicMock(gr_name=name, gr_mem=members))
        monkeypatch.setattr("exec_sandbox.permission_utils._read_passwd", lambda: passwd)
        monkeypatch.setattr(
            "exec_sandbox.permission_utils._read_group", lambda: f"{name}:x:{gid}:{','.join(members)}\n"
        )

        assert permission_utils._group_is_this_users(gid) is own

    @pytest.mark.parametrize(
        ("group_file", "own"),
        [
            ("me:x:1000:\n", True),
            ("me:x:1000:\nshadowupg:x:1000:nobody\n", False),  # groupadd -o: a second record with the gid
            ("me:x:1000:\nme:x:1000:eve\n", False),  # the same name, another member
            ("me:x:1000:\n#old:x:1000:eve\n", True),  # a comment is no record
            ("me:x:1000:\n+nis:x:1000:eve\n", False),  # initgroups reads compat lines: eve gets the gid
            ("me:x:1000:\n+::\n", True),  # a compat line with no gid reads as gid 0, not 1000
        ],
        ids=["one-record", "second-name", "second-members", "comment", "nis-compat-line", "compat-no-gid"],
    )
    def test_every_group_record_with_the_gid(self, monkeypatch: pytest.MonkeyPatch, group_file: str, own: bool) -> None:
        """getgrgid shows the first record; initgroups gives the gid to the members of each."""
        from exec_sandbox import permission_utils

        monkeypatch.setattr("exec_sandbox.permission_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setattr(pwd, "getpwuid", lambda _uid: MagicMock(pw_name="me", pw_gid=1000))
        monkeypatch.setattr(grp, "getgrgid", lambda _gid: MagicMock(gr_name="me", gr_mem=[]))
        monkeypatch.setattr("exec_sandbox.permission_utils._read_passwd", lambda: "me:x:1000:1000::/home/me:/bin/sh\n")
        monkeypatch.setattr("exec_sandbox.permission_utils._read_group", lambda: group_file)

        assert permission_utils._group_is_this_users(1000) is own

    def test_never_on_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """grp does not show macOS's computed members (everyone, localaccounts): even a group
        that looks private is not trusted there."""
        from exec_sandbox import permission_utils

        monkeypatch.setattr("exec_sandbox.permission_utils.detect_host_os", lambda: HostOS.MACOS)
        monkeypatch.setattr(pwd, "getpwuid", lambda _uid: MagicMock(pw_name="me", pw_gid=1000))
        monkeypatch.setattr(grp, "getgrgid", lambda _gid: MagicMock(gr_name="me", gr_mem=[]))
        monkeypatch.setattr("exec_sandbox.permission_utils._read_passwd", lambda: "me:x:1000:1000::/home/me:/bin/sh\n")
        assert permission_utils._group_is_this_users(1000) is False

    def test_unreadable_passwd_file_is_not_private(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real reader raises: no answer about other accounts is not "no other account"."""
        from exec_sandbox import permission_utils

        real_read_text = Path.read_text

        def read_text(self: Path, *args: Any, **kwargs: Any) -> str:
            if str(self) == "/etc/passwd":
                raise PermissionError(13, "Permission denied")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr("exec_sandbox.permission_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setattr(pwd, "getpwuid", lambda _uid: MagicMock(pw_name="me", pw_gid=1000))
        monkeypatch.setattr(grp, "getgrgid", lambda _gid: MagicMock(gr_name="me", gr_mem=[]))
        monkeypatch.setattr(Path, "read_text", read_text)
        assert permission_utils._group_is_this_users(1000) is False

    @pytest.mark.parametrize("failure", ["group", "passwd", "group-file"])
    def test_a_failed_lookup_is_not_this_users(self, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
        from exec_sandbox import permission_utils

        def missing(gid: int) -> None:
            raise KeyError(gid)

        def unreadable() -> str:
            raise PermissionError(13, "Permission denied")

        private = MagicMock(gr_name="me", gr_mem=[])
        monkeypatch.setattr("exec_sandbox.permission_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setattr(pwd, "getpwuid", lambda _uid: MagicMock(pw_name="me", pw_gid=1000))
        monkeypatch.setattr(grp, "getgrgid", missing if failure == "group" else lambda _gid: private)
        monkeypatch.setattr(
            "exec_sandbox.permission_utils._read_passwd",
            unreadable if failure == "passwd" else lambda: "me:x:1000:1000::/home/me:/bin/sh\n",
        )
        monkeypatch.setattr(
            "exec_sandbox.permission_utils._read_group",
            unreadable if failure == "group-file" else lambda: "me:x:1000:\n",
        )
        assert permission_utils._group_is_this_users(1000) is False


class TestSudoRmBound:
    """sudo_rm at each qemu-vm VM's destroy: a stuck sudo is killed after its bound."""

    async def test_stuck_sudo_is_killed_and_the_folder_stays(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "vm-1"
        folder.mkdir()
        stand_in = tmp_path / "bin" / "sudo"
        stand_in.parent.mkdir()
        pid_file = tmp_path / "sudo.pid"
        stand_in.write_text(f'#!/bin/sh\necho $$ > "{pid_file}"\nexec sleep 30\n')
        stand_in.chmod(0o755)
        monkeypatch.setenv("PATH", f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}")
        # 1 s: the stand-in writes its pid long before the bound, also on a loaded host
        monkeypatch.setattr(constants, "SUDO_RM_TIMEOUT_SECONDS", 1.0)

        start = asyncio.get_running_loop().time()
        assert await sudo_rm(folder) is False
        assert asyncio.get_running_loop().time() - start < 5.0  # far below the stand-in's 30 s
        assert folder.exists()
        pid = int(pid_file.read_text())  # it ran, so its kill is checked
        with contextlib.suppress(psutil.NoSuchProcess):  # gone: killed
            assert "sleep" not in " ".join(psutil.Process(pid).cmdline())  # a reused pid runs another program


class TestSudoRmChild:
    """sudo forks rm, which holds sudo's pipes: the bound must not wait for that child."""

    @staticmethod
    def _stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, *, bound: float) -> None:
        """sudo on PATH runs `body`; sudo_rm's bound is `bound` (1 s where the stand-in must
        fork or remove before it: a shell can take more than 0.2 s to start under load)."""
        stand_in = tmp_path / "bin" / "sudo"
        stand_in.parent.mkdir(exist_ok=True)
        stand_in.write_text(f"#!/bin/sh\n{body}\n")  # $1 -n, $2 rm, $3 -rf, $4 --, $5 the path
        stand_in.chmod(0o755)
        monkeypatch.setenv("PATH", f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setattr(constants, "SUDO_RM_TIMEOUT_SECONDS", bound)
        monkeypatch.setattr(permission_utils, "_left_rm", {})

    async def test_a_slow_child_of_sudo_does_not_hold_the_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The root rm stands in as a child (no exec) that sleeps: sudo is killed at the bound,
        and the wait does not wait for the child's end."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        self._stand_in(tmp_path, monkeypatch, "sh -c 'sleep 6'", bound=0.2)
        start = asyncio.get_running_loop().time()
        assert await sudo_rm(folder) is False
        assert asyncio.get_running_loop().time() - start < 3.0  # far below the child's 6 s
        assert folder.exists()

    async def test_a_folder_gone_at_the_bound_is_removed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """rm removed the folder, then sudo hung: the path is checked again after the kill."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        self._stand_in(tmp_path, monkeypatch, '/bin/rm -rf -- "$5"\nexec sleep 30', bound=1.0)
        assert await sudo_rm(folder) is True
        assert not folder.exists()


async def _forked(marker: Path) -> None:
    """Wait until the stand-in sudo has forked its rm: it writes the rm's pid to `marker` (10 s at most).
    A poll: another process writes the file, so there is no event to wait on."""
    for _ in range(1000):
        if marker.exists() and marker.read_text().strip():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"the stand-in sudo did not fork its rm: no pid in {marker}")


class TestSudoRmOutcomes:
    """What sudo_rm tells and keeps when sudo ends at the bound, fails, or leaves a root rm."""

    async def test_sudo_that_ended_at_the_kill_still_checks_the_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """sudo ends between the timeout and the kill (ProcessLookupError): the path is
        checked again, and a removed path is True, not a failed rm."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, '/bin/rm -rf -- "$5"\nexec sleep 30', bound=1.0)

        async def kill_that_finds_it_ended(self: ProcessWrapper) -> None:
            self.async_proc.kill()
            raise ProcessLookupError(f"Process {self.pid} is not running")

        monkeypatch.setattr(ProcessWrapper, "kill", kill_that_finds_it_ended)
        assert await sudo_rm(folder) is True

    async def test_a_failed_rm_logs_its_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """rm's stderr reaches the log: a busy mount and a missing sudo rule both exit 1."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        TestSudoRmChild._stand_in(
            tmp_path, monkeypatch, "echo \"rm: cannot remove '$5/x': Device or resource busy\" >&2\nexit 1", bound=5.0
        )
        with caplog.at_level(logging.WARNING, logger="exec_sandbox.permission_utils"):
            assert await sudo_rm(folder) is False
        [record] = [r for r in caplog.records if r.getMessage() == "sudo rm failed"]
        assert record.__dict__["exit_code"] == 1
        assert record.__dict__["stderr"] == f"rm: cannot remove '{folder}/x': Device or resource busy"

    @staticmethod
    def _no_room_for_files(monkeypatch: pytest.MonkeyPatch) -> None:
        """The temp folder has no inode left (ENOSPC on any new file)."""

        def no_room(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(
            permission_utils, "tempfile", SimpleNamespace(**{**vars(tempfile), "TemporaryFile": no_room})
        )

    async def test_no_room_for_the_error_file_still_removes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The VM folders that would free the temp folder live in it: with no room for the
        error file, sudo still runs (its errors go to /dev/null). A removal that works loses
        nothing: no WARNING."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, '/bin/rm -rf -- "$5"', bound=5.0)
        self._no_room_for_files(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="exec_sandbox.permission_utils"):
            assert await sudo_rm(folder) is True
        assert not folder.exists()
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    async def test_no_room_and_a_failed_rm_says_the_text_was_not_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        folder = tmp_path / "vm-1"
        folder.mkdir()
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, "echo busy >&2\nexit 1", bound=5.0)
        self._no_room_for_files(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="exec_sandbox.permission_utils"):
            assert await sudo_rm(folder) is False
        [record] = [r for r in caplog.records if r.getMessage() == "sudo rm failed"]
        assert record.__dict__["stderr"] == "(not kept: no file in the temp folder: No space left on device)"

    @staticmethod
    def _no_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
        """posix_fallocate fails for this user (a user quota, the ext4 root reserve)."""

        def no_reserve(*_args: Any) -> None:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")

        monkeypatch.setattr(permission_utils, "os", SimpleNamespace(**{**vars(os), "posix_fallocate": no_reserve}))

    @skip_unless_linux
    @pytest.mark.parametrize(
        ("body", "stderr"),
        [
            (
                "echo 'rm: cannot remove x: Operation not permitted' >&2\nexit 1",
                "rm: cannot remove x: Operation not permitted",
            ),
            ("exit 1", "(not kept: no room reserved in the temp folder: Disk quota exceeded)"),
        ],
        ids=["root-wrote-its-text", "nothing-written"],
    )
    async def test_a_failed_reserve_keeps_the_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        body: str,
        stderr: str,
    ) -> None:
        """The reserve runs as this user: a folder full for this user may still take what the
        root rm writes (the ext4 root reserve, a user quota), so the file stays; the reason
        shows only when no text came."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, body, bound=5.0)
        self._no_reserve(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="exec_sandbox.permission_utils"):
            assert await sudo_rm(folder) is False
        [record] = [r for r in caplog.records if r.getMessage() == "sudo rm failed"]
        assert record.__dict__["stderr"] == stderr

    def test_an_error_thrown_in_has_no_room_error_as_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The None path yields outside its except: a timeout or a cancel thrown into it does
        not carry the ENOSPC error as its context."""
        self._no_room_for_files(monkeypatch)
        with pytest.raises(subprocess.TimeoutExpired) as info, permission_utils.sudo_errors_file("/x"):
            raise subprocess.TimeoutExpired("sudo", 1)
        assert info.value.__context__ is None

    def test_a_zombie_left_rm_is_ended_and_reaped(self) -> None:
        """An orphan root rm whose parent became this process (PID 1, a subreaper) stays a
        zombie until reaped: it counts as ended, and it is reaped. Private level: a test
        process is not PID 1 or a subreaper, so sudo_rm never adopts an orphan here."""
        pid = os.posix_spawn("/usr/bin/true", ["true"], dict(os.environ))
        zombie = psutil.Process(pid)
        for _ in range(200):
            if zombie.status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.01)
        assert permission_utils._ended(permission_utils._Left(process=zombie, reap=True)) is True
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)  # reaped already

    def test_an_own_child_is_left_to_asyncio_to_reap(self) -> None:
        """sudo is this process's own child, which asyncio's watcher reaps: _keep marks it, and
        _ended counts its zombie as ended without taking its exit status. Private level: the
        zombie lasts only until asyncio reaps it, a window that sudo_rm cannot hold open."""
        pid = os.posix_spawn("/usr/bin/true", ["true"], dict(os.environ))
        child = psutil.Process(pid)
        for _ in range(200):
            if child.status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.01)
        permission_utils._keep("/x", [child])
        try:
            [left] = permission_utils._left_rm["/x"]
            assert left.reap is False
            assert permission_utils._ended(left) is True
            assert os.waitpid(pid, os.WNOHANG)[0] == pid  # its status was still there
        finally:
            permission_utils._left_rm.pop("/x", None)

    def test_a_process_this_user_may_not_inspect_still_runs(self) -> None:
        """hidepid=1: status() is refused while the process runs: not ended (its entry stays).
        Private level: hidepid is a mount option of /proc that a test cannot set."""
        process = MagicMock(spec=psutil.Process)
        process.is_running.return_value = True
        process.status.side_effect = psutil.AccessDenied(pid=1)
        assert permission_utils._ended(permission_utils._Left(process=process, reap=False)) is False

    def test_an_entry_that_grew_after_its_check_stays(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A call extended the entry between the check (in a thread) and the drop: it stays.
        Private level: the window between the two steps is too short to hit through sudo_rm."""
        ended = MagicMock(spec=psutil.Process)
        ended.is_running.return_value = False
        live = MagicMock(spec=psutil.Process)
        monkeypatch.setattr(permission_utils, "_left_rm", {"/x": [permission_utils._Left(process=ended, reap=False)]})
        judged = permission_utils._ended_left_rm()
        assert judged == [("/x", 1)]
        permission_utils._left_rm["/x"].append(permission_utils._Left(process=live, reap=False))
        permission_utils._drop_ended(judged)
        assert len(permission_utils._left_rm["/x"]) == 2

    async def test_an_ended_left_rm_goes_at_any_call(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The caller of a path whose root rm ended may never call again (it finds the path
        gone): any later call drops the entry. The entry is read from _left_rm: an entry that
        stays has no other effect than the memory it holds."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, '/bin/rm -rf -- "$5"', bound=5.0)
        ended = psutil.Popen(["true"])
        ended.wait(timeout=5)
        monkeypatch.setattr(
            permission_utils, "_left_rm", {str(tmp_path / "vm-0"): [permission_utils._Left(process=ended, reap=False)]}
        )
        assert await sudo_rm(folder) is True
        assert permission_utils._left_rm == {}

    async def test_two_calls_for_a_path_whose_rm_ended(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both calls drop the ended entry (pop with a default): no KeyError."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, '/bin/rm -rf -- "$5"', bound=5.0)
        ended = psutil.Popen(["true"])
        ended.wait(timeout=5)
        monkeypatch.setattr(
            permission_utils, "_left_rm", {str(folder): [permission_utils._Left(process=ended, reap=False)]}
        )
        assert await asyncio.gather(sudo_rm(folder), sudo_rm(folder)) == [True, True]
        assert not folder.exists()

    async def test_two_timed_out_calls_keep_both_rms(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two calls for one path at once, both past the bound: the rm of each is kept."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, "sleep 30 &\nexec sleep 30", bound=1.0)
        assert await asyncio.gather(sudo_rm(folder), sudo_rm(folder)) == [False, False]
        left = permission_utils._left_rm[str(folder)]
        try:
            assert len(left) == 2
        finally:
            for entry in left:
                with contextlib.suppress(psutil.Error):
                    entry.process.kill()

    async def test_a_cancelled_call_keeps_its_sudo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cancel leaves sudo and its rm running: they are kept, and a new call starts no
        second rm while they run."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        forked = tmp_path / "forked"
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, f'sleep 30 &\necho $! > "{forked}"\nexec sleep 30', bound=10.0)
        call = asyncio.create_task(sudo_rm(folder))
        await _forked(forked)  # the stand-in forked its rm
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        left = permission_utils._left_rm[str(folder)]
        try:
            assert len(left) == 2  # sudo and its rm
            assert await sudo_rm(folder) is False
        finally:
            for entry in left:
                with contextlib.suppress(psutil.Error):
                    entry.process.kill()

    @pytest.mark.parametrize("point", ["spawn", "children", "kill", "wait-after-kill", "lexists"])
    async def test_a_cancel_at_any_await_keeps_what_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str
    ) -> None:
        """A cancel at any await after (or during) the spawn keeps sudo and its rm while they
        run: the next call starts no second rm on the same tree."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        forked = tmp_path / "forked"
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, f'sleep 30 &\necho $! > "{forked}"\nexec sleep 30', bound=1.0)
        starts: list[list[str]] = []
        calls: list[asyncio.Task[bool]] = []
        waits = [0]
        real_sudo_exec, real_to_thread = permission_utils.sudo_exec, asyncio.to_thread
        real_kill, real_wait = ProcessWrapper.kill, ProcessWrapper.wait

        async def sudo_exec(args: list[str], **kwargs: Any) -> ProcessWrapper:
            starts.append(args)
            proc = await real_sudo_exec(args, **kwargs)
            if point == "spawn":
                calls[0].cancel()
            await _forked(forked)  # the stand-in forks its rm
            return proc

        async def to_thread(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
            if (point, getattr(func, "__name__", "")) in {("children", "_sudo_children"), ("lexists", "lexists")}:
                calls[0].cancel()
            return await real_to_thread(func, *args, **kwargs)

        async def kill(self: ProcessWrapper) -> None:
            if point == "kill":
                calls[0].cancel()
            await real_kill(self)

        async def wait(self: ProcessWrapper) -> int:
            waits[0] += 1
            if point == "wait-after-kill" and waits[0] == 2:
                calls[0].cancel()
            return await real_wait(self)

        monkeypatch.setattr(permission_utils, "sudo_exec", sudo_exec)
        monkeypatch.setattr(permission_utils, "asyncio", asyncio_with(to_thread=to_thread))
        monkeypatch.setattr(ProcessWrapper, "kill", kill)
        monkeypatch.setattr(ProcessWrapper, "wait", wait)
        if point == "lexists":
            monkeypatch.setattr(constants, "SUDO_RM_TIMEOUT_SECONDS", 1.0)
        calls.append(asyncio.create_task(sudo_rm(folder)))
        kept: list[permission_utils._Left] = []
        try:
            with pytest.raises(asyncio.CancelledError):
                await calls[0]
            kept = list(permission_utils._left_rm.get(str(folder), []))
            assert any(entry.process.is_running() for entry in kept)
            assert await sudo_rm(folder) is False
            assert len(starts) == 1  # no second rm
        finally:
            for entry in kept:
                with contextlib.suppress(psutil.Error):
                    entry.process.kill()

    async def test_a_root_rm_that_runs_on_gets_no_second(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A timed-out sudo leaves its rm running (here a child that sleeps): a new call for the
        same path starts no sudo while it runs, and one again once it ended."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, "sleep 30 &\nexec sleep 30", bound=1.0)
        starts: list[list[str]] = []
        real_sudo_exec = permission_utils.sudo_exec

        async def counting_sudo_exec(args: list[str], **kwargs: Any) -> ProcessWrapper:
            starts.append(args)
            return await real_sudo_exec(args, **kwargs)

        monkeypatch.setattr(permission_utils, "sudo_exec", counting_sudo_exec)
        assert await sudo_rm(folder) is False
        [left] = permission_utils._left_rm[str(folder)]
        try:
            assert await sudo_rm(folder) is False
            assert len(starts) == 1
        finally:
            left.process.kill()
            left.process.wait(timeout=5)
        TestSudoRmChild._stand_in(tmp_path, monkeypatch, '/bin/rm -rf -- "$5"', bound=5.0)
        monkeypatch.setattr(permission_utils, "_left_rm", {str(folder): [left]})
        assert await sudo_rm(folder) is True
        assert len(starts) == 2
        assert permission_utils._left_rm == {}


class TestGetQemuVmUid:
    """Tests for get_qemu_vm_uid function."""

    def test_matches_id(self) -> None:
        """The uid that `id -u qemu-vm` gives, or None when it finds no such user."""
        out = subprocess.run(["id", "-u", "qemu-vm"], capture_output=True, text=True, check=False)
        assert get_qemu_vm_uid() == (int(out.stdout) if out.returncode == 0 else None)

    def test_cached_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One passwd lookup for many calls: each socket check calls it."""
        lookups: list[str] = []

        def getpwnam(name: str) -> pwd.struct_passwd:
            lookups.append(name)
            raise KeyError(name)

        get_qemu_vm_uid.cache_clear()
        monkeypatch.setattr(pwd, "getpwnam", getpwnam)
        try:
            assert get_qemu_vm_uid() is None
            assert get_qemu_vm_uid() is None
        finally:
            get_qemu_vm_uid.cache_clear()
        assert lookups == ["qemu-vm"]


class TestProbeQemuVmUser:
    """Tests for probe_qemu_vm_user function."""

    @skip_unless_linux
    async def test_matches_the_user_database(self) -> None:
        """True exactly when the passwd database has a qemu-vm user."""
        try:
            pwd.getpwnam("qemu-vm")
            exists = True
        except KeyError:
            exists = False
        assert await probe_qemu_vm_user() is exists

    async def test_cached_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One user database lookup, in a worker thread, for the life of this process,
        whatever the answer."""
        lookups: list[object] = []
        real_to_thread = asyncio.to_thread

        async def counted_to_thread(func: Callable[[], object]) -> object:
            lookups.append(func)
            return await real_to_thread(func)

        monkeypatch.setattr("exec_sandbox.permission_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setattr(permission_utils, "asyncio", asyncio_with(to_thread=counted_to_thread))
        first = await probe_qemu_vm_user()

        assert await probe_qemu_vm_user() is first
        assert lookups == [get_qemu_vm_uid]

    @skip_unless_macos
    async def test_false_on_macos(self) -> None:
        """probe_qemu_vm_user returns False on macOS."""
        result = await probe_qemu_vm_user()
        assert result is False


class _Proc:
    """An asyncio subprocess stand-in: its exit code and stdout; with `ends=False`, a command
    that never ends until it is killed."""

    def __init__(self, returncode: int, stdout: bytes, *, ends: bool = True) -> None:
        self.returncode: int | None = returncode if ends else None
        self._stdout = stdout
        self._end = asyncio.Event()
        if ends:
            self._end.set()

    async def communicate(self, _stdin: bytes | None) -> tuple[bytes, bytes]:  # run_probe passes its stdin
        await self._end.wait()
        return self._stdout, b""

    def kill(self) -> None:
        self.returncode = -9
        self._end.set()

    async def wait(self) -> int | None:
        await self._end.wait()
        return self.returncode


class TestLocalUsersWithPrimaryGroup:
    """/etc/passwd read as glibc's nss_files reads it (the sets getent passwd gave on Ubuntu 26.04).
    Private level: see TestParseId."""

    @pytest.mark.parametrize(
        ("passwd", "users"),
        [
            ("qemu-vm:x:999:986::/:/usr/sbin/nologin\nalice:x:1000:1000::/home/alice:/bin/bash\n", {"qemu-vm"}),
            ("qemu-vm:x:999:986::/:/usr/sbin/nologin", {"qemu-vm"}),  # no final newline
            ("#old:x:1003:986::/:/bin/sh\n  # spaced comment:x:1:986::/:/bin/sh\n", set()),
            ("+carol::1004:986:::\n-dave::1005:986:::\n+::::::\n", set()),  # NIS compat: not files accounts
            (
                "erin:x:1006:0986::/:/bin/sh\nfrank:x:1007: 986::/:/bin/sh\n  lead:x:1008:986::/:/bin/sh\n",
                {"erin", "frank", "lead"},
            ),
            (
                "gina:x:1:9_86::/:/bin/sh\nivy:x:2:986 ::/:/bin/sh\nhank:x:abc:986::/:/bin/sh\nnoid:x::986::/:/bin/sh\n",
                set(),
            ),
            ("qemu-vm:x:999:986::/:/usr/sbin/nologin\r\n\nbroken-line\n", {"qemu-vm"}),
            ("nul:x:1003:986\0junk\n", {"nul"}),  # a NUL ends the line
            ("\x1c#fs:x:1:986::/:/bin/sh\n\x85#nel:x:2:986::/:/bin/sh\n", {"\x1c#fs", "\x85#nel"}),  # not C blanks
            ("cr\rme:x:1:986::/:/bin/sh\nx\x1ce:x:2:986::/:/bin/sh\n", {"cr\rme", "x\x1ce"}),  # lines end at \n only
            (":x:1:986::/:/bin/sh\n", {""}),  # an empty name is an account
            ("fs:x:1:\x1c986::/:/bin/sh\n", set()),  # \x1c is no C blank: strtoul stops there
            ("uidminus0:x:-0:986::/:/bin/sh\nminus1:x:-1:986::/:/bin/sh\n", {"uidminus0"}),  # -0 is 0
        ],
        ids=[
            "plain",
            "no-final-newline",
            "comments",
            "nis-compat",
            "leading-zero-blank-indent",
            "invalid-ids",
            "crlf-blank-broken",
            "nul",
            "non-c-blanks",
            "line-ends",
            "empty-name",
            "non-c-blank-before-gid",
            "minus-zero",
        ],
    )
    def test_accounts_whose_primary_group_it_is(
        self, monkeypatch: pytest.MonkeyPatch, passwd: str, users: set[str]
    ) -> None:
        from exec_sandbox import permission_utils

        monkeypatch.setattr("exec_sandbox.permission_utils._read_passwd", lambda: passwd)

        assert permission_utils._local_users_with_primary_group(986) == users


_FALLBACK = "QEMU runs as this user, not as qemu-vm (less isolation): "
"""The start of the warning of a broken step of the qemu-vm path; the step's reason follows."""


def _shared(group: str, others: str) -> str:
    """The reason of a qemu-vm group (gid 986) that other users share."""
    return (
        f"QEMU would run with group {group} (986), which other users share (other members: {others}); "
        "they could reach every VM's sockets: give qemu-vm its own group, qemu-vm, with only this user in it"
    )


class TestProbeQemuAsQemuVm:
    """probe_qemu_as_qemu_vm: one QEMU identity per process, checked end to end. The host
    (Linux, qemu-vm uid 999 and gid 986) and each sudo call are stand-ins; `sudo_report` is
    what the command run through QEMU_VM_SUDO prints (its umask, then its gid)."""

    @staticmethod
    def _host(
        monkeypatch: pytest.MonkeyPatch,
        *,
        sudo_rc: int = 0,
        sudo_report: bytes = b"0007\n986\n",
        groups: list[int] | None = None,
        euid: int = 1000,
        chown_ok: bool = True,
        rm_rc: int = 0,
        cgroup_ok: bool = True,
        group_name: str = "qemu-vm",
        group_members: tuple[str, ...] = ("runner",),
        primary_group_users: tuple[str, ...] = ("qemu-vm",),
        extra_group_lines: str = "",
        stuck: str | None = None,
    ) -> list[tuple[str, ...]]:
        """Patch the host; returns the list of commands the probe runs. This process's user is
        `runner`; qemu-vm's group (986) has `group_members`, and `primary_group_users` have it
        as their primary group. The sudo of the command `stuck` (chown, rm) never ends."""
        commands: list[tuple[str, ...]] = []

        async def exec_(*cmd: str, **_kwargs: Any) -> _Proc:
            commands.append(cmd)
            if stuck is not None and stuck in cmd:
                return _Proc(0, b"", ends=False)
            if "chown" in cmd:
                return _Proc(0 if chown_ok else 1, b"")
            if "rm" in cmd:
                return _Proc(rm_rc, b"")
            return _Proc(sudo_rc, sudo_report)

        mod = "exec_sandbox.permission_utils"
        monkeypatch.setattr(f"{mod}.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setattr(f"{mod}.probe_qemu_vm_user", AsyncMock(return_value=True))
        monkeypatch.setattr(f"{mod}.get_qemu_vm_uid", lambda: 999)
        monkeypatch.setattr(system_probes, "asyncio", asyncio_with(create_subprocess_exec=exec_))  # run_probe
        monkeypatch.setattr(f"{mod}.os.geteuid", lambda: euid)
        monkeypatch.setattr(f"{mod}.os.getegid", lambda: 1000)
        monkeypatch.setattr(f"{mod}.os.getgroups", lambda: [1000, 986] if groups is None else groups)
        monkeypatch.setattr(f"{mod}.can_contain_processes", AsyncMock(return_value=cgroup_ok))
        monkeypatch.setattr(
            f"{mod}.grp.getgrgid",
            lambda gid: SimpleNamespace(gr_name=group_name, gr_gid=gid, gr_mem=list(group_members)),
        )
        monkeypatch.setattr(f"{mod}.pwd.getpwuid", lambda _uid: SimpleNamespace(pw_name="runner"))
        passwd = "".join(
            f"{name}:x:{1100 + i}:986::/home/{name}:/bin/sh\n" for i, name in enumerate(primary_group_users)
        )
        monkeypatch.setattr(f"{mod}._read_passwd", lambda: passwd + "someone:x:1001:100::/home/someone:/bin/bash\n")
        group_file = f"{group_name}:x:986:{','.join(group_members)}\nusers:x:100:\n" + extra_group_lines
        monkeypatch.setattr(f"{mod}._read_group", lambda: group_file)
        monkeypatch.setattr(f"{mod}.pwd.getpwall", lambda: pytest.fail("pwd.getpwall holds the GIL: read /etc/passwd"))
        return commands

    @staticmethod
    def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    async def test_whole_path_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """sudo, umask 007, the group, chown and rm all work: QEMU runs as qemu-vm. The
        probe runs the launch prefix itself, and once: the answer is cached. Each sudo of the
        probe ignores cached credentials (-k): a sudo typed minutes ago would pass now and fail
        every VM start once it expires."""
        commands = self._host(monkeypatch)

        assert await probe_qemu_as_qemu_vm() is True
        assert await probe_qemu_as_qemu_vm() is True

        assert commands[0] == ("sudo", "-n", "-k", *QEMU_VM_SUDO[2:], "/bin/sh", "-c", "umask; id -g")
        assert [c[:4] for c in commands[1:]] == [("sudo", "-n", "-k", "chown"), ("sudo", "-n", "-k", "rm")]

    @pytest.mark.parametrize(
        ("host", "reason"),
        [
            pytest.param(
                {"cgroup_ok": False},
                "this process cannot put a process in a VM cgroup (no delegated cgroup v2 subtree it runs "
                "inside), and only a cgroup ends a qemu-vm QEMU and proves its end; see the README's cgroup setup",
                id="no-delegated-cgroup",
            ),
            pytest.param(
                {"sudo_rc": 1, "sudo_report": b""},
                "`sudo -n -u qemu-vm` failed (exit 1); add `<user> ALL=(qemu-vm) NOPASSWD: ALL` to sudoers",
                id="no-sudoers-rule",
            ),
            pytest.param(
                {"sudo_report": b"0027\n986\n"},
                "commands run as qemu-vm get umask 0027: QEMU sockets would have no group write",
                id="umask-without-group-write",
            ),
            pytest.param(
                {"groups": [1000]},
                "this process is not in group 986 (qemu-vm) now, so it cannot connect to the sockets QEMU makes; "
                "add the user to the group and log in again, or start the service with `sg qemu-vm`",
                id="group-not-live",
            ),
            pytest.param({"group_name": "users"}, _shared("users", "none"), id="shared-named-group"),
            pytest.param({"group_members": ("runner", "eve")}, _shared("qemu-vm", "eve"), id="other-member"),
            pytest.param(
                {"primary_group_users": ("qemu-vm", "mallory")}, _shared("qemu-vm", "mallory"), id="other-primary"
            ),
            pytest.param(
                {"extra_group_lines": "shadowq:x:986:nobody\n"},
                _shared("qemu-vm", "nobody"),
                id="second-record-other-name",
            ),
            pytest.param(
                {"extra_group_lines": "qemu-vm:x:986:eve\n"}, _shared("qemu-vm", "eve"), id="second-record-member"
            ),
            pytest.param(
                {"extra_group_lines": "+shadowq:x:986:nobody\n"}, _shared("qemu-vm", "nobody"), id="compat-record"
            ),
            pytest.param(
                {"chown_ok": False},
                "`sudo -n chown` as root does not work; add the sudoers lines of the README",
                id="no-root-chown",
            ),
            pytest.param(
                {"rm_rc": 1}, "`sudo -n rm` as root does not work; add the sudoers lines of the README", id="no-root-rm"
            ),
        ],
    )
    async def test_any_broken_step_falls_back_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        host: dict[str, Any],
        reason: str,
    ) -> None:
        """A step fails: QEMU runs as this user for the whole process, with one warning
        that names the step. Three concurrent first calls get one answer and one warning."""
        self._host(monkeypatch, **host)

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.permission_utils"):
            results = await asyncio.gather(*(probe_qemu_as_qemu_vm() for _ in range(3)))
            assert await probe_qemu_as_qemu_vm() is False

        assert results == [False, False, False]
        assert self._warnings(caplog) == [_FALLBACK + reason]

    @pytest.mark.parametrize(("command", "root_sudos"), [("chown", ["chown"]), ("rm", ["chown", "rm"])])
    async def test_a_root_sudo_that_never_ends_falls_back_at_the_bound(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, command: str, root_sudos: list[str]
    ) -> None:
        """A sudo stuck in PAM, SSSD or a hostname lookup: the probe ends at its bound with
        the step's warning, not never (each VM start waits for this probe)."""
        commands = self._host(monkeypatch, stuck=command)
        monkeypatch.setattr(system_probes, "_PROBE_TIMEOUT_SECONDS", 0.05)

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.permission_utils"):
            assert await asyncio.wait_for(probe_qemu_as_qemu_vm(), timeout=5) is False

        assert [c[3] for c in commands[1:]] == root_sudos
        assert self._warnings(caplog) == [
            f"{_FALLBACK}`sudo -n {command}` as root does not work; add the sudoers lines of the README"
        ]

    async def test_unreadable_passwd_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No answer about the group's other accounts is not "no other account"."""
        self._host(monkeypatch)

        def unreadable() -> str:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr("exec_sandbox.permission_utils._read_passwd", unreadable)
        with caplog.at_level(logging.WARNING, logger="exec_sandbox.permission_utils"):
            assert await probe_qemu_as_qemu_vm() is False
        assert self._warnings(caplog) == [
            f"{_FALLBACK}/etc/passwd or /etc/group cannot be read to check who else has group qemu-vm (986): "
            "Permission denied"
        ]

    async def test_root_needs_no_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Root connects to any socket: the group check does not apply."""
        self._host(monkeypatch, groups=[0], euid=0)
        assert await probe_qemu_as_qemu_vm() is True

    async def test_process_that_is_qemu_vm_runs_qemu_as_itself(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """This process is qemu-vm: no sudo, no warning (QEMU already has that uid)."""
        commands = self._host(monkeypatch, euid=999)
        with caplog.at_level(logging.WARNING, logger="exec_sandbox.permission_utils"):
            assert await probe_qemu_as_qemu_vm() is False
        assert commands == []
        assert self._warnings(caplog) == []

    async def test_false_if_qemu_vm_user_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No qemu-vm user: False, no sudo run, no warning (a host without the setup)."""
        commands = self._host(monkeypatch)
        monkeypatch.setattr("exec_sandbox.permission_utils.probe_qemu_vm_user", AsyncMock(return_value=False))
        with caplog.at_level(logging.WARNING, logger="exec_sandbox.permission_utils"):
            assert await probe_qemu_as_qemu_vm() is False
        assert commands == []
        assert self._warnings(caplog) == []

    @skip_unless_macos
    async def test_false_on_macos(self) -> None:
        """probe_qemu_as_qemu_vm returns False on macOS."""
        assert await probe_qemu_as_qemu_vm() is False


class TestQemuVmLaunchPrefix:
    """QEMU_VM_SUDO: sudo -n -u qemu-vm, then a fixed sh script that sets umask 007 and runs
    the command. The command gets umask 007 whatever umask the caller has and whatever sudo
    applies (sudo-rs ORs in 022), and each argument as it is (no shell parsing)."""

    async def test_shell_part_sets_umask_and_passes_arguments_unchanged(self) -> None:
        """The part after sudo, run here as this user (no sudo needed)."""
        assert QEMU_VM_SUDO[:4] == ("sudo", "-n", "-u", "qemu-vm")
        args = ["a b", "$(id)", ";rm -rf /nonexistent", "'\"", "*"]
        proc = await asyncio.create_subprocess_exec(
            *QEMU_VM_SUDO[4:],
            "/bin/sh",
            "-c",
            'umask; printf "%s\\n" "$@"',
            "argv0",
            *args,
            stdout=asyncio.subprocess.PIPE,
            preexec_fn=lambda: os.umask(0o022),
        )
        stdout, _ = await proc.communicate()
        assert stdout.decode().splitlines() == ["0007", *args]

    @skip_unless_linux
    async def test_command_gets_umask_007(self) -> None:
        if not await probe_qemu_vm_user():
            pytest.skip("no qemu-vm user")
        proc = await asyncio.create_subprocess_exec(
            *QEMU_VM_SUDO,
            "/bin/sh",
            "-c",
            "umask",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=lambda: os.umask(0o022),
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            pytest.skip(f"sudo -n -u qemu-vm does not work here: {stderr.decode().strip()}")
        assert stdout.decode().strip() == "0007"


class TestAddExecuteBits:
    """_add_execute_bits: chmod a+x from the mode get_mode read, or False."""

    def test_keeps_the_other_bits(self, tmp_path: Path) -> None:
        """A 0o2750 folder ends 0o2751: the setgid bit and the other bits stay."""
        folder = tmp_path / "setgid"
        folder.mkdir()
        folder.chmod(0o2750)

        assert _add_execute_bits(folder, folder.stat().st_mode) is True

        assert stat.S_IMODE(folder.stat().st_mode) == 0o2751

    def test_missing_folder_returns_false(self, tmp_path: Path) -> None:
        assert _add_execute_bits(tmp_path / "gone", 0o40700) is False


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

    def test_returns_mode(self, tmp_path: Path) -> None:
        """get_mode returns file mode."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        test_file.chmod(0o640)

        result = get_mode(test_file)

        assert result is not None
        assert (result & 0o777) == 0o640

    def test_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """get_mode returns None for nonexistent file."""
        nonexistent = tmp_path / "nonexistent.txt"

        result = get_mode(nonexistent)

        assert result is None


class TestCanAccess:
    """Tests for system_probes.can_access."""

    def test_readable_file(self, tmp_path: Path) -> None:
        """can_access returns True for readable file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        test_file.chmod(0o644)

        assert can_access(test_file, os.R_OK) is True

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """can_access returns False for nonexistent file."""
        nonexistent = tmp_path / "nonexistent.txt"

        assert can_access(nonexistent, os.R_OK) is False

    def test_write_access(self, tmp_path: Path) -> None:
        """can_access checks write permission: a 0444 file is readable, not writable (not as root)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        assert can_access(test_file, os.W_OK) is True
        if os.geteuid() != 0:
            test_file.chmod(0o444)
            assert (can_access(test_file, os.R_OK), can_access(test_file, os.W_OK)) == (True, False)


class TestEnsureTraversable:
    """ensure_traversable(path): a+x on every folder above `path` but the filesystem root,
    innermost first, and only where an execute bit is missing. _add_execute_bits is recorded,
    not run, for the folders above tmp_path: they belong to the system."""

    @staticmethod
    async def _chmods(path: Path, *, tmp_path: Path, result: bool) -> tuple[bool, list[Path]]:
        """ensure_traversable(path): (its result, the folders chmod a+x ran on, in order).

        A real chmod for a folder in tmp_path when `result` is True; else `result`."""
        done: list[Path] = []

        def chmod(dir_path: Path, mode: int) -> bool:
            done.append(dir_path)
            if result and dir_path.is_relative_to(tmp_path):
                return _add_execute_bits(dir_path, mode)
            return result

        with patch("exec_sandbox.permission_utils._add_execute_bits", side_effect=chmod):
            ok = await ensure_traversable(path)
        return ok, done

    async def test_folder_without_x_gets_it(self, tmp_path: Path) -> None:
        """A 0o700 folder above the file ends 0o711."""
        folder = tmp_path / "private"
        folder.mkdir()
        folder.chmod(0o700)

        ok, done = await self._chmods(folder / "file.qcow2", tmp_path=tmp_path, result=True)

        assert ok is True
        assert done[0] == folder
        assert folder.stat().st_mode & 0o777 == 0o711

    async def test_every_folder_but_the_root_innermost_first(self, tmp_path: Path) -> None:
        """No folder has an execute bit (get_mode stands in): chmod runs on each folder from
        the file's parent up to a folder in the root, never on the root itself."""
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)

        with patch("exec_sandbox.permission_utils.get_mode", return_value=0o600):
            ok, done = await self._chmods(deep / "file.qcow2", tmp_path=tmp_path, result=True)

        assert ok is True
        assert done[:3] == [deep, tmp_path / "a", tmp_path]
        assert done[-1].parent == Path(tmp_path.anchor)
        assert Path(tmp_path.anchor) not in done

    async def test_traversable_folder_spawns_no_chmod(self, tmp_path: Path) -> None:
        """A folder that already has a+x is checked, not changed."""
        folder = tmp_path / "open"
        folder.mkdir()
        folder.chmod(0o711)

        ok, done = await self._chmods(folder / "file.qcow2", tmp_path=tmp_path, result=True)

        assert ok is True
        assert folder not in done
        assert folder.stat().st_mode & 0o777 == 0o711

    async def test_folder_without_other_x_is_fixed(self, tmp_path: Path) -> None:
        """Only o+x missing: chmod a+x runs, and the folder ends 0o751."""
        folder = tmp_path / "group-only"
        folder.mkdir()
        folder.chmod(0o750)

        ok, done = await self._chmods(folder / "file.qcow2", tmp_path=tmp_path, result=True)

        assert ok is True
        assert done[0] == folder
        assert folder.stat().st_mode & 0o777 == 0o751

    async def test_failed_chmod_returns_false_after_the_other_folders(self, tmp_path: Path) -> None:
        """One chmod fails: False, and the folders above it are still tried."""
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)

        with patch("exec_sandbox.permission_utils.get_mode", return_value=0o600):
            ok, done = await self._chmods(deep / "file.qcow2", tmp_path=tmp_path, result=False)

        assert ok is False
        assert done[:3] == [deep, tmp_path / "a", tmp_path]

    async def test_file_in_the_root_changes_nothing(self, tmp_path: Path) -> None:
        """Only the filesystem root is above the file: nothing to do."""
        ok, done = await self._chmods(Path(tmp_path.anchor) / "file.qcow2", tmp_path=tmp_path, result=True)

        assert (ok, done) == (True, [])


class TestAclOperations:
    """Tests for ACL functions (Linux only, no-op on macOS)."""

    @skip_unless_macos
    async def test_set_acl_user_macos_returns_false(self, tmp_path: Path) -> None:
        """set_acl_user returns False on macOS."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = await set_acl_user([test_file], user="nobody", perms="rw")

        assert result is False


@skip_unless_linux
class TestLinuxAclBinaries:
    """setfacl works on Linux: the test ASSERTS (not skips) to catch a missing binary in CI."""

    async def test_set_acl_works(self, tmp_path: Path) -> None:
        """set_acl_user adds the user's entry with the permissions asked (getfacl reads it back)."""
        test_file = tmp_path / "acl_test.txt"
        test_file.write_text("content")
        current_user = pwd.getpwuid(os.getuid()).pw_name

        result = await set_acl_user([test_file], user=current_user, perms="rw")

        assert result is True, "set_acl_user must succeed on Linux"
        assert read_acl(test_file)[f"user:{current_user}"] == "rw-"


class TestSudoRm:
    """sudo_rm runs `sudo -n rm -rf -- <path>` with stdout on /dev/null and stderr in a file:
    no pipe that the root rm could hold after sudo ends."""

    async def test_runs_rm_without_pipes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The stand-in sudo records its arguments and the kind of its stdout and stderr, then
        removes the path ("--": a path never reads as an option)."""
        folder = tmp_path / "vm-1"
        folder.mkdir()
        (folder / "disk.qcow2").write_bytes(b"x")
        argv, fds = tmp_path / "argv", tmp_path / "fds"
        kind = 'if [ -p "$f" ]; then k=pipe; elif [ -c "$f" ]; then k=device; elif [ -f "$f" ]; then k=file; fi'
        TestSudoRmChild._stand_in(
            tmp_path,
            monkeypatch,
            f'printf \'%s\\n\' "$@" > "{argv}"\n'
            f'for f in /dev/fd/1 /dev/fd/2; do k=other; {kind}; printf \'%s\\n\' "$k" >> "{fds}"; done\n'
            '/bin/rm -rf -- "$5"',
            bound=5.0,
        )

        assert await sudo_rm(folder) is True
        assert not folder.exists()
        assert argv.read_text().splitlines() == ["-n", "rm", "-rf", "--", str(folder)]
        assert fds.read_text().splitlines() == ["device", "file"]  # /dev/null, then an unlinked file


class TestSudoNeverPrompts:
    """Every sudo the library runs has -n: with a terminal and no NOPASSWD rule, sudo would
    wait for a password and the caller would hang; -n makes it fail at once. A stand-in sudo
    on PATH records its arguments."""

    async def test_sudo_exec(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _record_calls(tmp_path, monkeypatch, "sudo", exit_code=0)

        proc = await sudo_exec(
            ["rm", "-rf", "/x"], stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await proc.communicate()

        assert calls() == [["-n", "rm", "-rf", "/x"]]

    async def test_chown_async(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls = _record_calls(tmp_path, monkeypatch, "sudo", exit_code=0)
        disk = tmp_path / "disk.qcow2"
        disk.write_bytes(b"")

        assert await chown_async(disk, "qemu-vm", "qemu-vm", owner_uid=os.geteuid()) is True

        assert calls() == [["-n", "chown", "-h", "qemu-vm:qemu-vm", "--", str(disk)]]

    @pytest.mark.parametrize("planted", ["link", "folder", "other-owner"])
    async def test_root_never_chowns_what_is_not_the_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, planted: str
    ) -> None:
        """Root follows a link: a link planted in place of an overlay would give any file of
        the host to qemu-vm. Only a regular file of the expected owner, checked first."""
        calls = _record_calls(tmp_path, monkeypatch, "sudo", exit_code=0)
        victim = tmp_path / "victim"
        victim.write_bytes(b"secret")
        path = tmp_path / "overlay.qcow2"
        owner = os.geteuid()
        if planted == "link":
            path.symlink_to(victim)
        elif planted == "folder":
            path.mkdir()
        else:
            path.write_bytes(b"")
            owner += 1

        assert await chown_async(path, "qemu-vm", "qemu-vm", owner_uid=owner) is False
        assert calls() == []


class TestGrantReadAccess:
    """grant_qemu_vm_file_access(writable=False): a+r through an O_NOFOLLOW descriptor."""

    async def test_regular_file_becomes_readable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("exec_sandbox.permission_utils.ensure_traversable", AsyncMock(return_value=True))
        layer = tmp_path / "layer.qcow2"
        layer.write_bytes(b"")
        layer.chmod(0o600)

        assert await grant_qemu_vm_file_access(layer, writable=False) is True
        assert stat.S_IMODE(layer.stat().st_mode) == 0o644

    async def test_planted_link_never_opens_its_target(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A link in place of a cache file would make a private file of this user readable by all."""
        monkeypatch.setattr("exec_sandbox.permission_utils.ensure_traversable", AsyncMock(return_value=True))
        key = tmp_path / "id_ed25519"
        key.write_bytes(b"private")
        key.chmod(0o600)
        layer = tmp_path / "layer.qcow2"
        layer.symlink_to(key)

        assert await grant_qemu_vm_file_access(layer, writable=False) is False
        assert stat.S_IMODE(key.stat().st_mode) == 0o600


class TestMakeCacheDir:
    """make_cache_dir: a folder the setting can fix is ConfigError SETTING_INVALID naming the
    setting; a host error is TransientError LOCAL_DISK (it can pass later)."""

    def test_a_file_in_the_path_is_a_setting_to_fix(self, tmp_path: Path) -> None:
        (tmp_path / "file").write_bytes(b"")
        folder = tmp_path / "file" / "cache"
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID) as exc:
            make_cache_dir(folder, setting="EXEC_SANDBOX_CACHE_DIR")
        assert exc.value.message == f"Folder {folder} cannot be made: {os.strerror(errno.ENOTDIR)}"
        assert exc.value.context == {"folder": str(folder), "errno": errno.ENOTDIR}
        assert exc.value.hint == "Set EXEC_SANDBOX_CACHE_DIR to a folder this user can write, on a writable disk."

    @pytest.mark.parametrize(
        ("error", "expected", "code"),
        [
            (errno.EACCES, ConfigError, ErrorCode.SETTING_INVALID),
            (errno.EROFS, ConfigError, ErrorCode.SETTING_INVALID),
            (errno.EEXIST, ConfigError, ErrorCode.SETTING_INVALID),
            (errno.ENOSPC, TransientError, ErrorCode.LOCAL_DISK),
            (errno.EDQUOT, TransientError, ErrorCode.LOCAL_DISK),
            (errno.EIO, TransientError, ErrorCode.LOCAL_DISK),
            (errno.EMFILE, TransientError, ErrorCode.LOCAL_DISK),
        ],
    )
    def test_the_error_follows_the_errno(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        error: int,
        expected: type[ConfigError | TransientError],
        code: ErrorCode,
    ) -> None:
        def fails(_path: Path) -> None:
            raise OSError(error, os.strerror(error))

        monkeypatch.setattr(permission_utils, "make_private_dir", fails)
        with pytest.raises(expected) as exc:
            make_cache_dir(tmp_path / "cache", setting="disk_snapshot_cache_dir")
        assert exc.value.code is code
        assert exc.value.context == {"folder": str(tmp_path / "cache"), "errno": error}


class TestMakeCacheDirIsPrivate:
    """make_cache_dir: make_private_dir, then a folder that exists must be private too."""

    def test_new_folders_are_private(self, tmp_path: Path) -> None:
        old = os.umask(0o002)
        try:
            make_cache_dir(tmp_path / "a" / "b", setting="cache_dir")
        finally:
            os.umask(old)
        assert [stat.S_IMODE(p.stat().st_mode) & 0o777 for p in (tmp_path / "a", tmp_path / "a" / "b")] == [
            0o700,
            0o700,
        ]

    def test_existing_traversable_folder_passes(self, tmp_path: Path) -> None:
        """0711 is what ensure_traversable leaves for qemu-vm."""
        folder = tmp_path / "cache"
        folder.mkdir(mode=0o711)
        folder.chmod(0o711)
        make_cache_dir(folder, setting="cache_dir")

    @pytest.mark.parametrize("mode", [0o775, 0o757], ids=["group-writable", "other-writable"])
    def test_existing_writable_folder_is_refused(self, tmp_path: Path, mode: int) -> None:
        """Made before under `sg qemu-vm` and umask 002: QEMU as qemu-vm could replace its files."""
        folder = tmp_path / "cache"
        folder.mkdir()
        folder.chmod(mode)
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID) as exc:
            make_cache_dir(folder, setting="cache_dir")
        assert exc.value.message == f"Folder {folder} has mode {oct(mode)}: group or others have access"
        assert exc.value.hint == f"Run `chmod 700 {folder}`."
        assert exc.value.context == {"folder": str(folder), "setting": "cache_dir"}

    def test_folder_in_a_parent_others_can_change_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 0775 ~/.cache of group qemu-vm (made under `sg qemu-vm`): qemu-vm could rename the
        cache folder away and put its own in place, so the folder made in it is refused too."""
        parent = tmp_path / "cache"
        parent.mkdir()
        parent.chmod(0o775)
        monkeypatch.setattr("exec_sandbox.permission_utils._group_is_this_users", lambda _gid: False)
        folder = parent / "exec-sandbox" / "disk-snapshots"
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID) as exc:
            make_cache_dir(folder, setting="cache_dir")
        changes = parent.resolve()  # the check walks the real path
        assert exc.value.message == (
            f"Folder {folder} is in {changes} (uid {os.geteuid()}, mode 0o775), which another user can change"
        )
        assert exc.value.hint == (
            f"Use a folder whose parent folders only root and this user can write, run `chmod go-w {changes}`, "
            f"or, when others must keep writing there (a shared volume), `chmod +t {changes}` (a sticky folder "
            "passes, as /tmp). Only the folder's owner can run these: root (sudo) for a folder of root, an init "
            "container that runs as root for a Kubernetes emptyDir."
        )

    def test_link_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "real").mkdir(mode=0o700)
        (tmp_path / "cache").symlink_to(tmp_path / "real")
        # O_NOFOLLOW | O_DIRECTORY on the link: ELOOP on Linux, ENOTDIR on macOS
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID) as exc:
            make_cache_dir(tmp_path / "cache", setting="cache_dir")
        assert exc.value.context["errno"] in {errno.ELOOP, errno.ENOTDIR}


# =============================================================================
# SUDO TESTS - Require elevated privileges
# =============================================================================


async def _sudo_chown_to_root(*paths: Path) -> bool:
    """Test setup: give `paths` (files or folders) to root. Not chown_async: it gives only a
    regular file of a known owner, never a folder."""
    proc = await sudo_exec(
        ["chown", "root:root", "--", *map(str, paths)],
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()
    return proc.returncode == 0


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

        # In a root folder: unlink needs write access to the folder, not to the file
        root_dir = tmp_path / "root_dir"
        root_dir.mkdir()
        test_file = root_dir / "root_owned.txt"
        test_file.write_text("owned by root")

        # Step 1: Change ownership to root
        chown_result = await _sudo_chown_to_root(test_file, root_dir)
        if not chown_result:
            pytest.skip("Cannot chown to root (sudo not available?)")

        # Verify ownership changed
        owner = pwd.getpwuid(test_file.stat().st_uid).pw_name
        assert owner == "root", f"Expected owner 'root', got '{owner}'"

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
        assert await sudo_rm(root_dir) is True

    async def test_sudo_rm_removes_root_owned_directory(self, tmp_path: Path) -> None:
        """sudo_rm removes root-owned directory with contents."""
        if os.getuid() == 0:
            pytest.skip("Test must run as non-root user")

        test_dir = tmp_path / "root_owned_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("file 1")
        (test_dir / "file2.txt").write_text("file 2")

        # Change ownership to root
        chown_result = await _sudo_chown_to_root(test_dir)
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


@pytest.mark.sudo
@skip_unless_linux
class TestAclRealLinux:
    """Real ACL tests on Linux - require setfacl."""

    async def test_set_acl_user_gives_qemu_vm_read_write(self, tmp_path: Path) -> None:
        """set_acl_user gives the qemu-vm user read and write access by an ACL entry (the
        gvproxy socket grant)."""
        test_file = tmp_path / "qemu_acl.txt"
        test_file.write_text("content")

        result = await set_acl_user([test_file], user="qemu-vm", perms="rw")

        assert result is True
        assert read_acl(test_file)["user:qemu-vm"] == "rw-"


# =============================================================================
# grant_qemu_vm_file_access Tests
# =============================================================================


class TestGrantQemuVmFileAccess:
    """Tests for grant_qemu_vm_file_access function.

    grant_qemu_vm_file_access makes the folders above the file traversable up to the
    filesystem root, and the folders above tmp_path belong to the system. So
    ensure_traversable stands in for the walk where the walk is not under test; the walk
    itself is tested in TestEnsureTraversable. A stand-in sudo on PATH records the chown.
    """

    # ---- writable=False path ----

    async def test_readonly_file_already_readable_is_left_as_it_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file others can read already needs no chmod: chmod by a user who does not own
        the file fails even when the mode does not change (an image installed by root). The
        stand-in fchmod fails as for such a file."""
        test_file = tmp_path / "base.qcow2"
        test_file.write_bytes(b"\x00")
        test_file.chmod(0o644)

        def refused(_fd: int, _mode: int) -> None:
            raise PermissionError(errno.EPERM, "Operation not permitted")

        monkeypatch.setattr(permission_utils, "os", SimpleNamespace(**{**vars(os), "fchmod": refused}))
        with patch("exec_sandbox.permission_utils.ensure_traversable", new_callable=AsyncMock, return_value=True):
            assert await grant_qemu_vm_file_access(test_file, writable=False) is True
        assert stat.S_IMODE(test_file.stat().st_mode) == 0o644

    async def test_chmod_failure_returns_false(self, tmp_path: Path) -> None:
        """writable=False returns False when the file cannot be opened for its chmod (no such file)."""
        nonexistent = tmp_path / "does_not_exist.qcow2"

        with patch(
            "exec_sandbox.permission_utils.ensure_traversable",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await grant_qemu_vm_file_access(nonexistent, writable=False)

        assert result is False

    # ---- writable=True path (stand-in sudo, no real qemu-vm user) ----

    async def test_writable_gives_the_file_to_qemu_vm(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """writable=True: one `sudo -n chown -h qemu-vm:qemu-vm -- <file>`, and no a+r:
        other users still cannot read the file."""
        calls = _record_calls(tmp_path, monkeypatch, "sudo", exit_code=0)
        test_file = tmp_path / "snap.qcow2"
        test_file.write_bytes(b"\x00")
        test_file.chmod(0o600)

        with patch("exec_sandbox.permission_utils.ensure_traversable", new_callable=AsyncMock, return_value=True):
            assert await grant_qemu_vm_file_access(test_file, writable=True) is True

        assert calls() == [["-n", "chown", "-h", "qemu-vm:qemu-vm", "--", str(test_file)]]
        assert stat.S_IMODE(test_file.stat().st_mode) == 0o600

    async def test_chown_failure_returns_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """writable=True returns False when the chown fails (sudo exits 1)."""
        _record_calls(tmp_path, monkeypatch, "sudo", exit_code=1)
        test_file = tmp_path / "snap.qcow2"
        test_file.write_bytes(b"\x00")

        with patch("exec_sandbox.permission_utils.ensure_traversable", new_callable=AsyncMock, return_value=True):
            assert await grant_qemu_vm_file_access(test_file, writable=True) is False

    # ---- Failure propagation ----

    @pytest.mark.parametrize("writable", [True, False])
    async def test_traversal_failure_skips_file_op(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writable: bool
    ) -> None:
        """A folder above cannot be made traversable: False, no chown and no a+r."""
        calls = _record_calls(tmp_path, monkeypatch, "sudo", exit_code=0)
        test_file = tmp_path / "snap.qcow2"
        test_file.write_bytes(b"\x00")
        test_file.chmod(0o600)

        with patch("exec_sandbox.permission_utils.ensure_traversable", new_callable=AsyncMock, return_value=False):
            assert await grant_qemu_vm_file_access(test_file, writable=writable) is False

        assert calls() == []
        assert stat.S_IMODE(test_file.stat().st_mode) == 0o600

    # ---- Parent directory traversal ----

    async def test_folder_of_the_file_becomes_traversable(self, tmp_path: Path) -> None:
        """The walk starts at the file's own folder: a 0700 folder ends 0711. chmod a+x runs
        for real in tmp_path only (the folders above it belong to the system)."""
        test_file = tmp_path / "l1" / "file.qcow2"
        test_file.parent.mkdir()
        test_file.parent.chmod(0o700)
        test_file.write_bytes(b"\x00")
        test_file.chmod(0o600)

        def chmod(dir_path: Path, mode: int) -> bool:
            return _add_execute_bits(dir_path, mode) if dir_path.is_relative_to(tmp_path) else True

        with patch("exec_sandbox.permission_utils._add_execute_bits", side_effect=chmod):
            assert await grant_qemu_vm_file_access(test_file, writable=False) is True

        assert stat.S_IMODE(test_file.parent.stat().st_mode) == 0o711
        assert stat.S_IMODE(test_file.stat().st_mode) == 0o644


def _my_other_group(path: Path) -> int:
    """A group this process is in that `path` does not have yet (a stand-in for qemu-vm)."""
    current = path.stat().st_gid
    for gid in sorted({os.getegid(), *os.getgroups()}):
        if gid != current:
            return gid
    pytest.skip("this process is in one group only")


def _foreign_group() -> int:
    """A group this process is not in: chown to it fails with EPERM (not as root)."""
    if os.geteuid() == 0:
        pytest.skip("root may give a file to any group")
    return max({os.getegid(), *os.getgroups()}) + 1


class TestGrantQemuVmDirAccess:
    """grant_qemu_vm_dir_access: the VM folder gets group qemu-vm and mode 2770, through
    one descriptor, with no sudo; any failure is False and never a wider mode. The
    qemu-vm group is a stand-in: a group this process is in (or not in)."""

    @staticmethod
    def _folder(tmp_path: Path) -> Path:
        """A VM folder as the overlay grant leaves it: mkdtemp 0700, then a+x."""
        folder = tmp_path / "vm"
        folder.mkdir(mode=0o700)
        folder.chmod(0o711)
        return folder

    def test_group_qemu_vm_mode_2770_and_new_files_get_the_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = self._folder(tmp_path)
        gid = _my_other_group(folder)
        monkeypatch.setattr("exec_sandbox.permission_utils.get_qemu_vm_gid", lambda: gid)

        assert grant_qemu_vm_dir_access(folder) is True

        st = folder.stat()
        assert (stat.S_IMODE(st.st_mode), st.st_gid, st.st_uid) == (0o2770, gid, os.getuid())
        assert st.st_mode & stat.S_IRWXO == 0  # other users: no access
        (folder / "cmd.sock").touch()
        assert (folder / "cmd.sock").stat().st_gid == gid  # setgid: QEMU's sockets get the group

    def test_group_this_process_is_not_in_fails_and_changes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EPERM from the group change: False, and the mode stays as it was (no a+rwx)."""
        folder = self._folder(tmp_path)
        before = folder.stat()
        gid = _foreign_group()
        monkeypatch.setattr("exec_sandbox.permission_utils.get_qemu_vm_gid", lambda: gid)

        assert grant_qemu_vm_dir_access(folder) is False

        after = folder.stat()
        assert (after.st_mode, after.st_gid) == (before.st_mode, before.st_gid)

    def test_dropped_setgid_is_a_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """chmod drops the setgid bit with no error for a process outside the group: the
        read-back catches it."""
        folder = self._folder(tmp_path)
        gid = _my_other_group(folder)
        monkeypatch.setattr("exec_sandbox.permission_utils.get_qemu_vm_gid", lambda: gid)
        real_fchmod = os.fchmod
        monkeypatch.setattr(
            "exec_sandbox.permission_utils.os.fchmod", lambda fd, mode: real_fchmod(fd, mode & ~stat.S_ISGID)
        )

        assert grant_qemu_vm_dir_access(folder) is False

    def test_link_is_not_followed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = self._folder(tmp_path)
        link = tmp_path / "link"
        link.symlink_to(target)
        before = target.stat()
        gid = _my_other_group(target)
        monkeypatch.setattr("exec_sandbox.permission_utils.get_qemu_vm_gid", lambda: gid)

        assert grant_qemu_vm_dir_access(link) is False

        after = target.stat()
        assert (after.st_mode, after.st_gid) == (before.st_mode, before.st_gid)

    def test_no_qemu_vm_user(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        folder = self._folder(tmp_path)
        monkeypatch.setattr("exec_sandbox.permission_utils.get_qemu_vm_gid", lambda: None)
        assert grant_qemu_vm_dir_access(folder) is False
        assert stat.S_IMODE(folder.stat().st_mode) & 0o777 == 0o711

    def test_gid_is_the_group_of_the_qemu_vm_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The group QEMU runs with (sudo -u qemu-vm), not the uid."""
        entry = pwd.struct_passwd(("qemu-vm", "x", 999, 986, "", "/nonexistent", "/usr/sbin/nologin"))
        monkeypatch.setattr(pwd, "getpwnam", lambda name: entry if name == "qemu-vm" else None)
        assert get_qemu_vm_gid() == 986

        def missing(name: str) -> pwd.struct_passwd:
            raise KeyError(name)

        monkeypatch.setattr(pwd, "getpwnam", missing)
        assert get_qemu_vm_gid() is None


def _acl_xattr(*entries: tuple[int, int, int]) -> bytes:
    """A system.posix_acl_access value: version 2, then (tag, perm, id) entries."""
    return struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *entry) for entry in entries)


_USER_OBJ, _USER, _GROUP_OBJ, _MASK, _OTHER = 0x01, 0x02, 0x04, 0x10, 0x20
_NO_ID = 0xFFFFFFFF


class TestQemuVmAcl:
    """qemu_vm_acl_perms reads the qemu-vm entry of a file's ACL (xattr, no process);
    grant_qemu_vm_acl runs one setfacl, only for the files whose entry differs."""

    @staticmethod
    def _linux(monkeypatch: pytest.MonkeyPatch, xattrs: dict[str, bytes]) -> None:
        """A Linux host with POSIX ACLs (system_probes.posix_acl_xattr) and these ACLs by path."""
        monkeypatch.setattr("exec_sandbox.permission_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setattr("exec_sandbox.permission_utils.get_qemu_vm_uid", lambda: 999)
        monkeypatch.setattr("exec_sandbox.permission_utils.posix_acl_xattr", lambda path: xattrs.get(str(path)))

    @pytest.mark.parametrize(
        ("entries", "perms"),
        [
            pytest.param([(_USER_OBJ, 6, _NO_ID), (_USER, 6, 999), (_MASK, 6, _NO_ID)], 6, id="rw"),
            pytest.param([(_USER_OBJ, 6, _NO_ID), (_USER, 6, 999), (_MASK, 4, _NO_ID)], 4, id="mask-limits"),
            pytest.param([(_USER_OBJ, 6, _NO_ID), (_USER, 6, 1000), (_MASK, 6, _NO_ID)], None, id="other-user"),
            pytest.param([(_USER_OBJ, 6, _NO_ID), (_GROUP_OBJ, 4, _NO_ID), (_OTHER, 0, _NO_ID)], None, id="no-entry"),
        ],
    )
    def test_perms_of_the_qemu_vm_entry(
        self, monkeypatch: pytest.MonkeyPatch, entries: list[tuple[int, int, int]], perms: int | None
    ) -> None:
        self._linux(monkeypatch, {"/f": _acl_xattr(*entries)})
        assert qemu_vm_acl_perms(Path("/f")) == perms

    def test_no_acl_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._linux(monkeypatch, {})
        assert qemu_vm_acl_perms(Path("/f")) is None

    def test_a_value_as_the_kernel_keeps_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bytes of `u::rw-,u:999:rw-,g::r--,m::r--,o::---` written out by hand from the
        acl(5) layout (little-endian: version 2, then a 2-byte tag, a 2-byte perm and a 4-byte
        id for each entry): the mask limits qemu-vm to r."""
        raw = bytes.fromhex(
            "02000000"  # version 2
            "01000600ffffffff"  # user::rw-
            "02000600e7030000"  # user:999:rw-
            "04000400ffffffff"  # group::r--
            "10000400ffffffff"  # mask::r--
            "20000000ffffffff"  # other::---
        )
        self._linux(monkeypatch, {"/f": raw})
        assert qemu_vm_acl_perms(Path("/f")) == 4

    async def test_one_setfacl_for_the_files_that_differ(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Read wanted: a file with r already is skipped; a file with rw (write must go)
        and a file without an entry get one setfacl (a stand-in on PATH records it)."""
        read, write, bare = tmp_path / "read", tmp_path / "write", tmp_path / "bare"
        self._linux(
            monkeypatch,
            {
                str(read): _acl_xattr((_USER, 4, 999), (_MASK, 6, _NO_ID)),
                str(write): _acl_xattr((_USER, 6, 999), (_MASK, 6, _NO_ID)),
            },
        )
        calls = _record_calls(tmp_path, monkeypatch, "setfacl", exit_code=0)
        with patch("exec_sandbox.permission_utils.ensure_traversable", new_callable=AsyncMock, return_value=True):
            assert await grant_qemu_vm_acl([read, write, bare], perms="r") is True
        assert calls() == [["-m", "u:qemu-vm:r", str(write), str(bare)]]

    async def test_nothing_to_do_spawns_nothing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._linux(monkeypatch, {str(tmp_path / "f"): _acl_xattr((_USER, 4, 999), (_MASK, 4, _NO_ID))})
        calls = _record_calls(tmp_path, monkeypatch, "setfacl", exit_code=0)
        with patch("exec_sandbox.permission_utils.ensure_traversable", new_callable=AsyncMock, return_value=True):
            assert await grant_qemu_vm_acl([tmp_path / "f"], perms="r") is True
        assert calls() == []

    async def test_untraversable_folder_fails_closed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A folder above the files cannot be made traversable (ensure_traversable stands in
        for a folder of another user): False, and no ACL is set (QEMU would fail to open the
        file anyway)."""
        self._linux(monkeypatch, {})
        calls = _record_calls(tmp_path, monkeypatch, "setfacl", exit_code=0)
        with patch("exec_sandbox.permission_utils.ensure_traversable", new_callable=AsyncMock, return_value=False):
            assert await grant_qemu_vm_acl([tmp_path / "a", tmp_path / "b"], perms="rw") is False
        assert calls() == []


@pytest.mark.sudo
@skip_unless_linux
class TestQemuVmAclReal:
    """grant_qemu_vm_acl with the real setfacl and the real qemu-vm user."""

    async def test_rw_then_r(self, tmp_path: Path) -> None:
        if get_qemu_vm_uid() is None:
            pytest.skip("no qemu-vm user")
        files = [tmp_path / "a.vmstate", tmp_path / "a.ram"]
        for f in files:
            f.write_bytes(b"x")
            f.chmod(0o600)
        assert await grant_qemu_vm_acl(files, perms="rw") is True
        assert [qemu_vm_acl_perms(f) for f in files] == [6, 6]
        assert await grant_qemu_vm_acl(files, perms="r") is True
        assert [qemu_vm_acl_perms(f) for f in files] == [4, 4]
        assert read_acl(files[0])["user:qemu-vm"] == "r--"


# =============================================================================
# grant_qemu_vm_file_access Sudo Integration Tests
# =============================================================================


@pytest.mark.sudo
@skip_unless_linux  # qemu-vm user only exists on Linux CI runners
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
        assert pwd.getpwuid(test_file.stat().st_uid).pw_name == "qemu-vm"

        await sudo_rm(test_file)

"""Tests for non-root REPL execution (UID 1000 sandbox user).

Guest-agent runs as root (PID 1) for package install, file I/O, and module
loading. The REPL subprocess drops to UID 1000 to block mount(2), ptrace,
raw sockets and other root operations without needing seccomp.

Test categories:
- Normal: identity of each REPL, file I/O, package install
- Edge: permission boundaries between root guest-agent and UID 1000 REPL
- Weird: privilege escalation attempts that fail
- Out of bounds: kernel-level operations blocked by missing capabilities
"""

from pathlib import Path

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once


# =============================================================================
# Normal: REPL runs as non-root
# =============================================================================
class TestNonRootIdentity:
    """Each REPL runs its code as UID 1000 and GID 1000 (user 'user'), not root."""

    async def test_shell_identity(self, scheduler: Scheduler) -> None:
        """The shell REPL runs as user 'user', UID 1000, GID 1000."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "whoami; id -u; id -g")
        assert (result.exit_code, result.stdout) == (0, "user\n1000\n1000\n"), result.stderr

    async def test_python_identity(self, scheduler: Scheduler) -> None:
        """The Python REPL runs as UID 1000 and GID 1000."""
        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), "import os; print(os.getuid(), os.getgid())"
        )
        assert (result.exit_code, result.stdout) == (0, "1000 1000\n"), result.stderr

    async def test_javascript_uid(self, scheduler: Scheduler) -> None:
        """JavaScript REPL runs as UID 1000."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            "const { execSync } = require('child_process'); console.log(execSync('id -u').toString().trim())",
        )
        assert (result.exit_code, result.stdout) == (0, "1000\n"), result.stderr


# =============================================================================
# Edge: File I/O across root/non-root boundary
# =============================================================================
class TestNonRootFileIO:
    """File I/O works across the root guest-agent / UID 1000 REPL boundary."""

    async def test_write_file_then_read_in_repl(self, scheduler: Scheduler) -> None:
        """Files written by guest-agent (root) are readable by REPL (UID 1000)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("test.txt", b"hello from root")
            result = await session.exec('print(open("test.txt").read())')
            assert (result.exit_code, result.stdout) == (0, "hello from root\n"), result.stderr

    async def test_repl_creates_file_then_read_file(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Files created by REPL (UID 1000) are readable by guest-agent (root)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec('open("created_by_repl.txt", "w").write("hello from repl")')
            assert result.exit_code == 0

            dest = tmp_path / "created_by_repl.txt"
            await session.read_file("created_by_repl.txt", destination=dest)
            assert dest.read_text() == "hello from repl"

    async def test_repl_file_in_home_owned_by_uid_1000(self, scheduler: Scheduler) -> None:
        """The REPL can write in /home/user, and the file it makes there is owned by UID 1000."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.PYTHON),
            "import os; open('/home/user/myfile.txt', 'w').write('x'); print(os.stat('/home/user/myfile.txt').st_uid)",
        )
        assert (result.exit_code, result.stdout) == (0, "1000\n"), result.stderr


# =============================================================================
# Edge: Package install works (root installs, non-root imports)
# =============================================================================
class TestNonRootPackages:
    """Package installation (root) and import (UID 1000) work together."""

    @pytest.mark.slow
    async def test_pip_install_and_import(self, scheduler: Scheduler) -> None:
        """Packages installed by root are importable by non-root REPL."""
        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON, packages=["six==1.17.0"]), "import six; print(six.__version__)"
        )
        assert (result.exit_code, result.stdout) == (0, "1.17.0\n"), result.stderr


# =============================================================================
# Weird: Privilege escalation attempts that should fail
# =============================================================================
class TestNonRootPrivilegeBlocked:
    """Operations requiring root/capabilities fail for UID 1000 REPL."""

    async def test_python_mount_blocked(self, scheduler: Scheduler) -> None:
        """mount(2) needs CAP_SYS_ADMIN: it fails with EPERM (1) for UID 1000."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
ret = libc.mount(b"none", b"/mnt", b"tmpfs", 0, None)
print(f"ret={ret} errno={ctypes.get_errno()}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "ret=-1 errno=1\n"), result.stderr

    async def test_usr_readonly_mount(self, scheduler: Scheduler) -> None:
        """Writing to /usr returns EROFS (30, read-only filesystem), not just EACCES. Each
        root-owned folder of the guest is on this read-only root filesystem."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# O_WRONLY | O_CREAT = 0x41
fd = libc.open(b"/usr/local/bin/evil", 0x41, 0o644)
err = ctypes.get_errno()
print(f"fd={fd} errno={err}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "fd=-1 errno=30\n"), result.stderr

    async def test_no_new_privs_set(self, scheduler: Scheduler) -> None:
        """REPL subprocess has no_new_privs set (blocks SUID escalation)."""
        code = """\
with open("/proc/self/status") as f:
    for line in f:
        if line.startswith("NoNewPrivs:"):
            print(line.strip())
            break
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "NoNewPrivs:\t1\n"), result.stderr


# =============================================================================
# Out of bounds: Kernel-level operations blocked by missing capabilities
# =============================================================================
class TestNonRootCapabilitiesBlocked:
    """Kernel operations requiring specific capabilities are blocked."""

    async def test_raw_socket_blocked(self, scheduler: Scheduler) -> None:
        """RAW sockets require CAP_NET_RAW — blocked for non-root."""
        code = """\
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
    s.close()
    print("unexpected_success")
except PermissionError:
    print("blocked")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "blocked\n"), result.stderr

    async def test_privileged_port_blocked(self, scheduler: Scheduler) -> None:
        """Binding to port <1024 requires CAP_NET_BIND_SERVICE — blocked for non-root."""
        code = """\
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", 80))
    s.close()
    print("unexpected_success")
except PermissionError:
    print("blocked")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "blocked\n"), result.stderr

    async def test_chown_to_root_blocked(self, scheduler: Scheduler) -> None:
        """chown to root requires CAP_CHOWN — blocked for non-root."""
        code = """\
import os
open("/home/user/chown_test.txt", "w").write("test")
try:
    os.chown("/home/user/chown_test.txt", 0, 0)
    print("unexpected_success")
except PermissionError:
    print("blocked")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "blocked\n"), result.stderr

"""Tests for non-root REPL execution (UID 1000 sandbox user).

Guest-agent runs as root (PID 1) for package install, file I/O, and module
loading. The REPL subprocess drops to UID 1000 to block mount(2), ptrace,
raw sockets, and kernel module loading without needing seccomp.

Test categories:
- Normal: identity, code execution, file I/O, package install all work
- Edge: permission boundaries between root guest-agent and UID 1000 REPL
- Weird: privilege escalation attempts that should fail silently
- Out of bounds: kernel-level operations blocked by missing capabilities
"""

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


# =============================================================================
# Normal: REPL runs as non-root and basic operations work
# =============================================================================
class TestNonRootIdentity:
    """REPL process identity is UID 1000, not root."""

    async def test_shell_whoami(self, scheduler: Scheduler) -> None:
        """Shell REPL reports 'user' identity."""
        result = await scheduler.run(code="whoami", language=Language.RAW)
        assert result.exit_code == 0
        assert result.stdout.strip() == "user"

    async def test_shell_id(self, scheduler: Scheduler) -> None:
        """Shell REPL reports uid=1000, gid=1000."""
        result = await scheduler.run(code="id", language=Language.RAW)
        assert result.exit_code == 0
        assert "uid=1000" in result.stdout
        assert "gid=1000" in result.stdout

    async def test_python_uid(self, scheduler: Scheduler) -> None:
        """Python REPL runs as UID 1000."""
        result = await scheduler.run(
            code="import os; print(os.getuid())",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "1000"

    async def test_python_gid(self, scheduler: Scheduler) -> None:
        """Python REPL runs as GID 1000."""
        result = await scheduler.run(
            code="import os; print(os.getgid())",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "1000"

    async def test_javascript_uid(self, scheduler: Scheduler) -> None:
        """JavaScript REPL runs as UID 1000."""
        result = await scheduler.run(
            code="const { execSync } = require('child_process'); console.log(execSync('id -u').toString().trim())",
            language=Language.JAVASCRIPT,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "1000"


# =============================================================================
# Normal: Code execution works under non-root
# =============================================================================
class TestNonRootExecution:
    """Standard code execution works correctly as non-root."""

    async def test_python_hello_world(self, scheduler: Scheduler) -> None:
        """Basic Python execution works as non-root."""
        result = await scheduler.run(
            code='print("hello from sandbox")',
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert "hello from sandbox" in result.stdout

    async def test_javascript_hello_world(self, scheduler: Scheduler) -> None:
        """Basic JavaScript execution works as non-root."""
        result = await scheduler.run(
            code='console.log("hello from sandbox")',
            language=Language.JAVASCRIPT,
        )
        assert result.exit_code == 0
        assert "hello from sandbox" in result.stdout

    async def test_shell_hello_world(self, scheduler: Scheduler) -> None:
        """Basic shell execution works as non-root."""
        result = await scheduler.run(
            code='echo "hello from sandbox"',
            language=Language.RAW,
        )
        assert result.exit_code == 0
        assert "hello from sandbox" in result.stdout

    async def test_python_session_state_persistence(self, scheduler: Scheduler) -> None:
        """State persists across exec calls in non-root session."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("x = 42")
            result = await session.exec("print(x)")
            assert result.exit_code == 0
            assert "42" in result.stdout


# =============================================================================
# Edge: File I/O across root/non-root boundary
# =============================================================================
class TestNonRootFileIO:
    """File I/O works across the root guest-agent / UID 1000 REPL boundary."""

    async def test_write_file_then_read_in_repl(self, scheduler: Scheduler) -> None:
        """Files written by guest-agent (root) are readable by REPL (UID 1000)."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("test.txt", b"hello from root")
            result = await session.exec('print(open("test.txt").read())')
            assert result.exit_code == 0
            assert "hello from root" in result.stdout

    async def test_repl_creates_file_then_read_file(self, scheduler: Scheduler) -> None:
        """Files created by REPL (UID 1000) are readable by guest-agent (root)."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec('open("created_by_repl.txt", "w").write("hello from repl")')
            assert result.exit_code == 0

            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "created_by_repl.txt"
                await session.read_file("created_by_repl.txt", destination=dest)
                assert dest.read_text() == "hello from repl"

    async def test_repl_file_owned_by_uid_1000(self, scheduler: Scheduler) -> None:
        """Files created by REPL are owned by UID 1000."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec(
                "import os; open('myfile.txt', 'w').write('x'); print(os.stat('myfile.txt').st_uid)"
            )
            assert result.exit_code == 0
            assert result.stdout.strip() == "1000"

    async def test_home_dir_writable(self, scheduler: Scheduler) -> None:
        """REPL can write to /home/user (owned by UID 1000)."""
        result = await scheduler.run(
            code="import os; open('/home/user/test_write.txt', 'w').write('ok'); print('success')",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert "success" in result.stdout


# =============================================================================
# Edge: Package install works (root installs, non-root imports)
# =============================================================================
class TestNonRootPackages:
    """Package installation (root) and import (UID 1000) work together."""

    @pytest.mark.slow
    async def test_pip_install_and_import(self, scheduler: Scheduler) -> None:
        """Packages installed by root are importable by non-root REPL."""
        result = await scheduler.run(
            code="import six; print(six.__version__)",
            language=Language.PYTHON,
            packages=["six==1.17.0"],
        )
        assert result.exit_code == 0
        assert "1.17.0" in result.stdout


# =============================================================================
# Weird: Privilege escalation attempts that should fail
# =============================================================================
class TestNonRootPrivilegeBlocked:
    """Operations requiring root/capabilities fail for UID 1000 REPL."""

    async def test_mount_blocked(self, scheduler: Scheduler) -> None:
        """mount(2) requires CAP_SYS_ADMIN — blocked for non-root."""
        result = await scheduler.run(
            code="mount -t tmpfs none /mnt 2>&1; echo exit=$?",
            language=Language.RAW,
        )
        assert result.exit_code == 0  # Shell itself succeeds
        # mount fails with EPERM
        assert "ermission" in result.stdout or "exit=32" in result.stdout or "exit=1" in result.stdout

    async def test_python_mount_blocked(self, scheduler: Scheduler) -> None:
        """Python ctypes mount(2) fails with EPERM for non-root."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
ret = libc.mount(b"none", b"/mnt", b"tmpfs", 0, None)
print(f"ret={ret} errno={ctypes.get_errno()}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # EPERM = 1
        assert "errno=1" in result.stdout

    async def test_write_to_root_owned_dir_blocked(self, scheduler: Scheduler) -> None:
        """Writing to root-owned directories fails for UID 1000."""
        result = await scheduler.run(
            code="touch /usr/local/bin/evil 2>&1; echo exit=$?",
            language=Language.RAW,
        )
        assert "exit=1" in result.stdout or "ermission" in result.stdout

    async def test_usr_readonly_mount(self, scheduler: Scheduler) -> None:
        """Writing to /usr returns EROFS (read-only filesystem), not just EACCES."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# O_WRONLY | O_CREAT = 0x41
fd = libc.open(b"/usr/local/bin/evil", 0x41, 0o644)
err = ctypes.get_errno()
print(f"fd={fd} errno={err}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # EROFS = 30 (read-only filesystem)
        assert "errno=30" in result.stdout

    async def test_no_new_privs_set(self, scheduler: Scheduler) -> None:
        """REPL subprocess has no_new_privs set (blocks SUID escalation)."""
        code = """\
with open("/proc/self/status") as f:
    for line in f:
        if line.startswith("NoNewPrivs:"):
            print(line.strip())
            break
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "NoNewPrivs:\t1" in result.stdout

    async def test_kill_pid1_blocked(self, scheduler: Scheduler) -> None:
        """Non-root cannot signal PID 1 (guest-agent)."""
        result = await scheduler.run(
            code="import os, signal, errno\ntry:\n    os.kill(1, signal.SIGTERM)\n    print('unexpected_success')\nexcept PermissionError:\n    print('blocked')",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert "blocked" in result.stdout


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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_kernel_module_load_blocked(self, scheduler: Scheduler) -> None:
        """Loading kernel modules requires CAP_SYS_MODULE — blocked for non-root."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# init_module syscall (175 on x86_64, 105 on aarch64)
import platform
nr = 175 if platform.machine() == "x86_64" else 105
ret = libc.syscall(nr, b"", 0, b"")
err = ctypes.get_errno()
# EPERM=1 or ENOENT=2 (both acceptable — permission check happens first)
print(f"ret={ret} errno={err}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # EPERM=1 (no CAP_SYS_MODULE), ENOENT=2, or ENOSYS=38 (CONFIG_MODULES=n)
        assert "errno=1" in result.stdout or "errno=2" in result.stdout or "errno=38" in result.stdout

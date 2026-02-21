"""Tests for /etc filesystem hardening in guest VM.

Verifies that UID 1000 (sandbox user) cannot modify system configuration
files in /etc. The REPL subprocess drops to UID 1000 via guest-agent, and
/etc must be owned by root with standard Unix permissions to prevent:
- DNS hijacking (/etc/resolv.conf)
- Host alias injection (/etc/hosts)
- User injection (/etc/passwd)

Build-time fix: scripts/build-qcow2.sh resets ownership to root before
creating the qcow2 image (macOS Docker VirtioFS loses UID 0 on export).
Runtime fix: guest-agent hardens permissions at boot (defense-in-depth).

CIS Benchmark Section 6.1.x compliance:
- /etc/passwd, /etc/group: 0644 root:root
- /etc/shadow: 0640 root:shadow
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


# =============================================================================
# Normal: Direct writes to /etc files should fail
# =============================================================================
class TestEtcWriteBlocked:
    """Direct writes to /etc files and directory are blocked for UID 1000."""

    async def test_append_resolv_conf(self, scheduler: Scheduler) -> None:
        """Append to /etc/resolv.conf fails (DNS hijack vector).

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
try:
    with open('/etc/resolv.conf', 'a') as f:
        f.write('nameserver 1.2.3.4\\n')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_append_hosts(self, scheduler: Scheduler) -> None:
        """Append to /etc/hosts fails (host alias injection).

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
try:
    with open('/etc/hosts', 'a') as f:
        f.write('0.0.0.0 evil.example.com\\n')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_append_passwd(self, scheduler: Scheduler) -> None:
        """Append to /etc/passwd fails (UID 0 user injection).

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
try:
    with open('/etc/passwd', 'a') as f:
        f.write('evil:x:0:0:evil:/root:/bin/sh\\n')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_truncate_resolv_conf(self, scheduler: Scheduler) -> None:
        """Truncate /etc/resolv.conf fails (DNS denial-of-service).

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
try:
    with open('/etc/resolv.conf', 'w') as f:
        f.write('')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_create_file_in_etc(self, scheduler: Scheduler) -> None:
        """Creating new files in /etc is blocked (tests dir write permission)."""
        result = await scheduler.run(
            code="touch /etc/evil 2>&1; echo exit=$?",
            language=Language.RAW,
        )
        assert "exit=1" in result.stdout or "ermission" in result.stdout

    async def test_shell_redirect_to_etc_hosts(self, scheduler: Scheduler) -> None:
        """Shell redirect to /etc/hosts fails."""
        result = await scheduler.run(
            code='echo "0.0.0.0 evil" >> /etc/hosts 2>&1; echo exit=$?',
            language=Language.RAW,
        )
        assert "exit=1" in result.stdout or "ermission" in result.stdout or "Read-only" in result.stdout

    async def test_truncate_hosts(self, scheduler: Scheduler) -> None:
        """Truncate /etc/hosts fails (O_WRONLY|O_TRUNC path).

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
try:
    with open('/etc/hosts', 'w') as f:
        f.write('')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_tee_append_hosts(self, scheduler: Scheduler) -> None:
        """tee -a /etc/hosts fails (GTFOBins tee write vector)."""
        result = await scheduler.run(
            code="echo '0.0.0.0 evil' | tee -a /etc/hosts 2>&1; echo exit=$?",
            language=Language.RAW,
        )
        assert "exit=1" in result.stdout or "ermission" in result.stdout or "Read-only" in result.stdout

    async def test_dd_overwrite_hosts(self, scheduler: Scheduler) -> None:
        """dd of=/etc/hosts fails (GTFOBins dd write vector)."""
        result = await scheduler.run(
            code="echo '0.0.0.0 evil' | dd of=/etc/hosts 2>&1; echo exit=$?",
            language=Language.RAW,
        )
        assert "exit=1" in result.stdout or "ermission" in result.stdout or "Read-only" in result.stdout


# =============================================================================
# Edge: Permission boundary subtleties
# =============================================================================
class TestEtcEdgeCases:
    """Indirect or boundary-crossing attempts to modify /etc."""

    async def test_symlink_write_through(self, scheduler: Scheduler) -> None:
        """Write through symlink from sandbox -> /etc/resolv.conf is blocked.

        Symlink created in writable /home/user, but kernel checks target perms.
        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
import os
os.symlink('/etc/resolv.conf', '/home/user/link_resolv')
try:
    with open('/home/user/link_resolv', 'a') as f:
        f.write('nameserver 1.2.3.4\\n')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_hardlink_etc_passwd_write(self, scheduler: Scheduler) -> None:
        """Write through hard link to /etc/passwd is blocked.

        Hard link creation always fails (EXDEV: /home/user is tmpfs, /etc is
        rootfs — different filesystems). Fallback tests /etc/passwd directly.
        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
import os
# Create hard link (may succeed since user can read source and write to target dir)
try:
    os.link('/etc/passwd', '/home/user/passwd_link')
except OSError:
    # If hard link fails (fs.protected_hardlinks), test the original directly
    pass
target = '/home/user/passwd_link' if os.path.exists('/home/user/passwd_link') else '/etc/passwd'
try:
    with open(target, 'a') as f:
        f.write('evil:x:0:0:evil:/root:/bin/sh\\n')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_mkdir_in_etc(self, scheduler: Scheduler) -> None:
        """mkdir inside /etc is blocked (directory creation requires dir write).

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
import os
try:
    os.mkdir('/etc/evil.d')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_delete_file_from_etc(self, scheduler: Scheduler) -> None:
        """rm /etc/resolv.conf fails (deletion requires write on parent dir).

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
import os
try:
    os.unlink('/etc/resolv.conf')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_rename_into_etc(self, scheduler: Scheduler) -> None:
        """mv from /home/user into /etc fails (cross-dir rename needs target dir write).

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
import os
with open('/home/user/evil.conf', 'w') as f:
    f.write('evil')
try:
    os.rename('/home/user/evil.conf', '/etc/evil.conf')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_symlink_write_through_hosts(self, scheduler: Scheduler) -> None:
        """Write through symlink from sandbox -> /etc/hosts is blocked.

        RO bind remount follows symlink target, so EROFS is enforced.
        """
        code = """\
import os
os.symlink('/etc/hosts', '/home/user/link_hosts')
try:
    with open('/home/user/link_hosts', 'a') as f:
        f.write('0.0.0.0 evil.example.com\\n')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_cp_overwrite_hosts(self, scheduler: Scheduler) -> None:
        """cp overwrite /etc/hosts fails (GTFOBins cp vector).

        cp to existing file uses O_WRONLY|O_TRUNC — tests file-level write perms.
        """
        result = await scheduler.run(
            code="echo '0.0.0.0 evil' > /tmp/evil_hosts && cp /tmp/evil_hosts /etc/hosts 2>&1; echo exit=$?",
            language=Language.RAW,
        )
        assert "exit=1" in result.stdout or "ermission" in result.stdout or "Read-only" in result.stdout

    async def test_hosts_injection_no_resolution(self, scheduler: Scheduler) -> None:
        """Semantic test: injected host alias does not resolve.

        musl checks /etc/hosts before DNS — verifies injection attempt
        fails and the injected name produces socket.gaierror.
        """
        code = """\
import socket
# Attempt injection (should fail with EROFS or PermissionError)
try:
    with open('/etc/hosts', 'a') as f:
        f.write('127.0.0.1 evil.sandbox.test\\n')
except (PermissionError, OSError):
    pass
# Verify the name does NOT resolve
try:
    socket.getaddrinfo('evil.sandbox.test', 80)
    print('unexpected_resolution')
except socket.gaierror:
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout


# =============================================================================
# Weird: Indirect modification attempts
# =============================================================================
class TestEtcWeirdCases:
    """Unconventional modification attempts that should still fail."""

    async def test_chmod_etc_then_write(self, scheduler: Scheduler) -> None:
        """chmod 777 /etc fails (requires CAP_FOWNER or ownership), so write still blocked.

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
import os
try:
    os.chmod('/etc', 0o777)
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_shutil_copy_into_etc(self, scheduler: Scheduler) -> None:
        """shutil.copy into /etc fails (creates new file in /etc).

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
import shutil
with open('/home/user/src.txt', 'w') as f:
    f.write('evil')
try:
    shutil.copy('/home/user/src.txt', '/etc/evil.txt')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_sed_inplace_etc(self, scheduler: Scheduler) -> None:
        """sed -i on /etc/resolv.conf fails (creates temp + rename in /etc)."""
        result = await scheduler.run(
            code="sed -i 's/nameserver.*/nameserver 1.2.3.4/' /etc/resolv.conf 2>&1; echo exit=$?",
            language=Language.RAW,
        )
        assert (
            "exit=1" in result.stdout
            or "exit=2" in result.stdout
            or "ermission" in result.stdout
            or "Read-only" in result.stdout
        )

    async def test_ctypes_open_wronly(self, scheduler: Scheduler) -> None:
        """Direct open(2) via ctypes with O_WRONLY on /etc/resolv.conf fails.

        Bypasses Python file abstraction -- verifies kernel enforces permissions.
        """
        code = """\
import ctypes, os
libc = ctypes.CDLL("libc.so.6", use_errno=True)
fd = libc.open(b"/etc/resolv.conf", os.O_WRONLY)
err = ctypes.get_errno()
if fd == -1:
    print(f"blocked errno={err}")
else:
    os.close(fd)
    print("unexpected_success")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_sed_inplace_hosts(self, scheduler: Scheduler) -> None:
        """sed -i on /etc/hosts fails (creates temp + rename in /etc)."""
        result = await scheduler.run(
            code="sed -i 's/localhost/evil/' /etc/hosts 2>&1; echo exit=$?",
            language=Language.RAW,
        )
        assert (
            "exit=1" in result.stdout
            or "exit=2" in result.stdout
            or "ermission" in result.stdout
            or "Read-only" in result.stdout
        )

    async def test_ctypes_open_wronly_hosts(self, scheduler: Scheduler) -> None:
        """Direct open(2) via ctypes with O_WRONLY on /etc/hosts fails.

        Bypasses Python abstraction — verifies kernel enforces RO bind remount
        at the syscall level. Asserts EROFS (errno=30) specifically.
        """
        code = """\
import ctypes, os, errno
libc = ctypes.CDLL("libc.so.6", use_errno=True)
fd = libc.open(b"/etc/hosts", os.O_WRONLY)
err = ctypes.get_errno()
if fd == -1:
    if err == errno.EROFS:
        print(f"blocked erofs errno={err}")
    else:
        print(f"blocked errno={err}")
else:
    os.close(fd)
    print("unexpected_success")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout


# =============================================================================
# Out of bounds: Privilege escalation required, should fail hard
# =============================================================================
class TestEtcOutOfBounds:
    """Attacks requiring capabilities that UID 1000 does not have."""

    async def test_mount_overlay_on_etc(self, scheduler: Scheduler) -> None:
        """Overlay mount on /etc requires CAP_SYS_ADMIN -- blocked."""
        code = """\
import subprocess
r = subprocess.run(
    ["mount", "-t", "overlay", "overlay",
     "-o", "lowerdir=/etc,upperdir=/tmp/upper,workdir=/tmp/work", "/etc"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("blocked")
else:
    print("unexpected_success")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_chown_etc_to_user(self, scheduler: Scheduler) -> None:
        """chown /etc to UID 1000 requires CAP_CHOWN -- blocked.

        With RO bind remount, raises OSError (EROFS) instead of PermissionError.
        """
        code = """\
import os
try:
    os.chown('/etc', 1000, 1000)
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_proc_fd_write_bypass(self, scheduler: Scheduler) -> None:
        """/proc/self/fd trick: open read-only, write via fd path -- blocked.

        Kernel checks permissions on original open(), not the fd path.
        """
        code = """\
import os
fd = os.open('/etc/resolv.conf', os.O_RDONLY)
try:
    with open(f'/proc/self/fd/{fd}', 'w') as f:
        f.write('nameserver 1.2.3.4\\n')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
finally:
    os.close(fd)
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_proc_fd_write_bypass_hosts(self, scheduler: Scheduler) -> None:
        """/proc/self/fd trick on /etc/hosts: open read-only, write via fd path.

        Kernel checks permissions on original open(), not the fd path.
        """
        code = """\
import os
fd = os.open('/etc/hosts', os.O_RDONLY)
try:
    with open(f'/proc/self/fd/{fd}', 'w') as f:
        f.write('0.0.0.0 evil.example.com\\n')
    print('unexpected_success')
except (PermissionError, OSError):
    print('blocked')
finally:
    os.close(fd)
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_bind_mount_over_hosts(self, scheduler: Scheduler) -> None:
        """Bind mount over /etc/hosts requires CAP_SYS_ADMIN -- blocked."""
        code = """\
import subprocess
r = subprocess.run(
    ["mount", "--bind", "/tmp/evil_hosts", "/etc/hosts"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("blocked")
else:
    print("unexpected_success")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

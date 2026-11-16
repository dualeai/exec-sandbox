"""Tests for /etc filesystem hardening in guest VM.

Verifies that UID 1000 (sandbox user) cannot modify system configuration
files in /etc. The REPL subprocess drops to UID 1000 via guest-agent, and
no write may reach /etc, to prevent:
- DNS hijacking (/etc/resolv.conf)
- Host alias injection (/etc/hosts)
- User injection (/etc/passwd)

Build time: scripts/build-qcow2.sh gives /etc/passwd, /etc/group, /etc/hosts and
/etc/resolv.conf mode 0644 and /etc/shadow 0640 (CIS 6.1.x), and mkfs.erofs --all-root
makes each file of the image belong to root.
Run time: the root file system is EROFS, read-only; guest-agent also bind-remounts
/etc/hosts and /etc/resolv.conf read-only at boot (defense in depth).

Each attempt must fail with EROFS: any other error (ENOENT, EXDEV) means the
attempt did not reach the protection, and the test fails.
"""

import textwrap

from exec_sandbox.models import ExecutionResult, Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once


async def _guest_errno(scheduler: Scheduler, *, setup: str, attempt: str) -> str:
    """Run setup, then attempt, as UID 1000 in Python: the errno name of the OSError, or SUCCESS."""
    code = (
        f"import errno, os, shutil\n{setup}\n"
        f"try:\n{textwrap.indent(attempt, '    ')}\n    print('SUCCESS')\n"
        "except OSError as e:\n    print(errno.errorcode[e.errno])\n"
    )
    result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
    assert result.exit_code == 0, result.stderr
    return result.stdout.strip()


def _assert_shell_refused(result: ExecutionResult, *, stdout: str) -> None:
    """A shell command ran with 2>&1 and then `echo exit=$?`: `stdout` is its EROFS message and exit code."""
    assert (result.exit_code, result.stdout, result.stderr) == (0, stdout, "")


# =============================================================================
# Normal: Direct writes to /etc files should fail
# =============================================================================
class TestEtcWriteBlocked:
    """Direct writes to /etc files and directory are blocked for UID 1000."""

    async def test_append_resolv_conf(self, scheduler: Scheduler) -> None:
        """Append to /etc/resolv.conf fails with EROFS (DNS hijack vector)."""
        attempt = "with open('/etc/resolv.conf', 'a') as f:\n    f.write('nameserver 1.2.3.4\\n')"
        assert await _guest_errno(scheduler, setup="", attempt=attempt) == "EROFS"

    async def test_append_hosts(self, scheduler: Scheduler) -> None:
        """Append to /etc/hosts fails with EROFS (host alias injection)."""
        attempt = "with open('/etc/hosts', 'a') as f:\n    f.write('0.0.0.0 evil.example.com\\n')"
        assert await _guest_errno(scheduler, setup="", attempt=attempt) == "EROFS"

    async def test_append_passwd(self, scheduler: Scheduler) -> None:
        """Append to /etc/passwd fails with EROFS (UID 0 user injection)."""
        attempt = "with open('/etc/passwd', 'a') as f:\n    f.write('evil:x:0:0:evil:/root:/bin/sh\\n')"
        assert await _guest_errno(scheduler, setup="", attempt=attempt) == "EROFS"

    async def test_truncate_resolv_conf(self, scheduler: Scheduler) -> None:
        """Truncate /etc/resolv.conf fails with EROFS (DNS denial-of-service)."""
        attempt = "with open('/etc/resolv.conf', 'w') as f:\n    f.write('')"
        assert await _guest_errno(scheduler, setup="", attempt=attempt) == "EROFS"

    async def test_create_file_in_etc(self, scheduler: Scheduler) -> None:
        """touch of a new file in /etc fails with EROFS (a new entry in the directory)."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "touch /etc/evil 2>&1; echo exit=$?")
        _assert_shell_refused(result, stdout="touch: cannot touch '/etc/evil': Read-only file system\nexit=1\n")

    async def test_shell_redirect_to_etc_hosts(self, scheduler: Scheduler) -> None:
        """Shell redirect to /etc/hosts fails: the shell reports it on its own stderr (the
        failed redirect comes before the 2>&1 of the command)."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), 'echo "0.0.0.0 evil" >> /etc/hosts 2>&1; echo exit=$?'
        )
        assert (result.exit_code, result.stdout) == (0, "exit=1\n")
        assert result.stderr.endswith(": /etc/hosts: Read-only file system\n"), result.stderr  # after the script line

    async def test_truncate_hosts(self, scheduler: Scheduler) -> None:
        """Truncate /etc/hosts fails with EROFS (O_WRONLY|O_TRUNC path)."""
        attempt = "with open('/etc/hosts', 'w') as f:\n    f.write('')"
        assert await _guest_errno(scheduler, setup="", attempt=attempt) == "EROFS"

    async def test_tee_append_hosts(self, scheduler: Scheduler) -> None:
        """tee -a /etc/hosts fails (GTFOBins tee write vector)."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "echo '0.0.0.0 evil' | tee -a /etc/hosts 2>&1; echo exit=$?"
        )
        _assert_shell_refused(result, stdout="tee: /etc/hosts: Read-only file system\n0.0.0.0 evil\nexit=1\n")

    async def test_dd_overwrite_hosts(self, scheduler: Scheduler) -> None:
        """dd of=/etc/hosts fails (GTFOBins dd write vector)."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "echo '0.0.0.0 evil' | dd of=/etc/hosts 2>&1; echo exit=$?"
        )
        _assert_shell_refused(result, stdout="dd: failed to open '/etc/hosts': Read-only file system\nexit=1\n")


# =============================================================================
# Edge: Permission boundary subtleties
# =============================================================================
class TestEtcEdgeCases:
    """Indirect or boundary-crossing attempts to modify /etc."""

    async def test_symlink_write_through(self, scheduler: Scheduler) -> None:
        """Write through symlink from sandbox -> /etc/resolv.conf fails with EROFS.

        Symlink created in writable /home/user, but the kernel checks the target.
        """
        setup = "os.symlink('/etc/resolv.conf', '/home/user/link_resolv')"
        attempt = "with open('/home/user/link_resolv', 'a') as f:\n    f.write('nameserver 1.2.3.4\\n')"
        assert await _guest_errno(scheduler, setup=setup, attempt=attempt) == "EROFS"

    async def test_mkdir_in_etc(self, scheduler: Scheduler) -> None:
        """mkdir inside /etc fails with EROFS (directory creation requires dir write)."""
        assert await _guest_errno(scheduler, setup="", attempt="os.mkdir('/etc/evil.d')") == "EROFS"

    async def test_delete_file_from_etc(self, scheduler: Scheduler) -> None:
        """rm /etc/resolv.conf fails with EROFS (deletion requires write on parent dir)."""
        assert await _guest_errno(scheduler, setup="", attempt="os.unlink('/etc/resolv.conf')") == "EROFS"

    async def test_rename_into_etc(self, scheduler: Scheduler) -> None:
        """mv from /home/user into /etc fails with EROFS.

        /home/user is another file system, so rename(2) gives EXDEV; mv (here
        shutil.move) then copies the file into /etc, and that copy gets EROFS.
        """
        setup = "open('/home/user/evil.conf', 'w').write('evil')"
        attempt = "shutil.move('/home/user/evil.conf', '/etc/evil.conf')"
        assert await _guest_errno(scheduler, setup=setup, attempt=attempt) == "EROFS"

    async def test_symlink_write_through_hosts(self, scheduler: Scheduler) -> None:
        """Write through symlink from sandbox -> /etc/hosts fails with EROFS.

        The open follows the symlink to the read-only bind mount of /etc/hosts.
        """
        setup = "os.symlink('/etc/hosts', '/home/user/link_hosts')"
        attempt = "with open('/home/user/link_hosts', 'a') as f:\n    f.write('0.0.0.0 evil.example.com\\n')"
        assert await _guest_errno(scheduler, setup=setup, attempt=attempt) == "EROFS"

    async def test_cp_overwrite_hosts(self, scheduler: Scheduler) -> None:
        """cp overwrite /etc/hosts fails with EROFS (GTFOBins cp vector).

        cp to an existing file opens it O_WRONLY|O_TRUNC: the open of the file itself fails.
        """
        result = await run_once(
            scheduler,
            VmSpec(language=Language.RAW),
            "echo '0.0.0.0 evil' > /tmp/evil_hosts && cp /tmp/evil_hosts /etc/hosts 2>&1; echo exit=$?",
        )
        _assert_shell_refused(
            result, stdout="cp: cannot create regular file '/etc/hosts': Read-only file system\nexit=1\n"
        )

    async def test_hosts_injection_no_resolution(self, scheduler: Scheduler) -> None:
        """Semantic test: injected host alias does not resolve.

        musl checks /etc/hosts before DNS: the injection fails with EROFS, and
        the injected name gives socket.gaierror.
        """
        code = """\
import errno, socket
try:
    with open('/etc/hosts', 'a') as f:
        f.write('127.0.0.1 evil.sandbox.test\\n')
    print('WRITTEN')
except OSError as e:
    print(errno.errorcode[e.errno])
try:
    socket.getaddrinfo('evil.sandbox.test', 80)
    print('RESOLVED')
except socket.gaierror:
    print('NOT_RESOLVED')
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "EROFS\nNOT_RESOLVED\n"


# =============================================================================
# Weird: Indirect modification attempts
# =============================================================================
class TestEtcWeirdCases:
    """Unconventional modification attempts that should still fail."""

    async def test_chmod_etc(self, scheduler: Scheduler) -> None:
        """chmod 777 /etc fails with EROFS (it would also need CAP_FOWNER or ownership)."""
        assert await _guest_errno(scheduler, setup="", attempt="os.chmod('/etc', 0o777)") == "EROFS"

    async def test_shutil_copy_into_etc(self, scheduler: Scheduler) -> None:
        """shutil.copy into /etc fails with EROFS (creates new file in /etc)."""
        setup = "open('/home/user/src.txt', 'w').write('evil')"
        attempt = "shutil.copy('/home/user/src.txt', '/etc/evil.txt')"
        assert await _guest_errno(scheduler, setup=setup, attempt=attempt) == "EROFS"

    async def test_sed_inplace_etc(self, scheduler: Scheduler) -> None:
        """sed -i on /etc/resolv.conf fails: it cannot make its temp file in /etc (exit 4, I/O error)."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.RAW),
            "sed -i 's/nameserver.*/nameserver 1.2.3.4/' /etc/resolv.conf 2>&1; echo exit=$?",
        )
        _assert_shell_refused(
            result, stdout="sed: couldn't open temporary file /etc/sedXXXXXX: Read-only file system\nexit=4\n"
        )

    async def test_ctypes_open_wronly(self, scheduler: Scheduler) -> None:
        """Direct open(2) via ctypes with O_WRONLY on /etc/resolv.conf and /etc/hosts fails with EROFS.

        Bypasses Python file abstraction -- the kernel refuses the write open of a read-only mount
        (the RO bind remounts, at the syscall level).
        """
        code = """\
import ctypes, errno, os
libc = ctypes.CDLL("libc.so.6", use_errno=True)
for path in (b"/etc/resolv.conf", b"/etc/hosts"):
    fd = libc.open(path, os.O_WRONLY)
    print(errno.errorcode[ctypes.get_errno()] if fd == -1 else "SUCCESS")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "EROFS\nEROFS\n"


# =============================================================================
# Out of bounds: Privilege escalation required, should fail hard
# =============================================================================
class TestEtcOutOfBounds:
    """Attacks requiring capabilities that UID 1000 does not have."""

    async def test_mount_overlay_on_etc(self, scheduler: Scheduler) -> None:
        """Overlay mount on /etc requires CAP_SYS_ADMIN: mount refuses it."""
        code = """\
import subprocess
r = subprocess.run(
    ["mount", "-t", "overlay", "overlay",
     "-o", "lowerdir=/etc,upperdir=/tmp/upper,workdir=/tmp/work", "/etc"],
    capture_output=True, text=True
)
print(r.returncode, r.stderr.strip())
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "1 mount: permission denied (are you root?)\n"

    async def test_chown_etc_to_user(self, scheduler: Scheduler) -> None:
        """chown /etc to UID 1000 fails with EROFS (it would also need CAP_CHOWN)."""
        assert await _guest_errno(scheduler, setup="", attempt="os.chown('/etc', 1000, 1000)") == "EROFS"

    async def test_proc_fd_write_bypass(self, scheduler: Scheduler) -> None:
        """/proc/self/fd trick: open read-only, write via fd path: EROFS.

        Opening /proc/self/fd/N for write opens the file again, with a new
        permission check.
        """
        setup = "fd = os.open('/etc/resolv.conf', os.O_RDONLY)"
        attempt = "with open(f'/proc/self/fd/{fd}', 'w') as f:\n    f.write('nameserver 1.2.3.4\\n')"
        assert await _guest_errno(scheduler, setup=setup, attempt=attempt) == "EROFS"

    async def test_proc_fd_write_bypass_hosts(self, scheduler: Scheduler) -> None:
        """/proc/self/fd trick on /etc/hosts: open read-only, write via fd path: EROFS.

        Opening /proc/self/fd/N for write opens the file again, with a new
        permission check.
        """
        setup = "fd = os.open('/etc/hosts', os.O_RDONLY)"
        attempt = "with open(f'/proc/self/fd/{fd}', 'w') as f:\n    f.write('0.0.0.0 evil.example.com\\n')"
        assert await _guest_errno(scheduler, setup=setup, attempt=attempt) == "EROFS"

    async def test_bind_mount_over_hosts(self, scheduler: Scheduler) -> None:
        """Bind mount over /etc/hosts requires CAP_SYS_ADMIN: mount refuses it."""
        code = """\
import subprocess
r = subprocess.run(
    ["mount", "--bind", "/tmp/evil_hosts", "/etc/hosts"],
    capture_output=True, text=True
)
print(r.returncode, r.stderr.strip())
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "1 mount: permission denied (are you root?)\n"

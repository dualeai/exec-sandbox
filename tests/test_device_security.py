"""Tests for guest VM device-layer security hardening.

Verifies that dangerous device nodes are removed and /dev/shm is mounted
with restrictive flags. These are defense-in-depth measures that apply
regardless of UID — even root (guest-agent) cannot access /dev/mem after
these changes.

Complements test_nonroot_repl.py (privilege escalation from UID 1000) with
kernel/device-layer hardening that blocks access at the filesystem level.
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


# =============================================================================
# Dangerous device nodes should not exist
# =============================================================================
class TestDangerousDeviceNodes:
    """Dangerous device nodes are removed by tiny-init after devtmpfs mount."""

    async def test_dev_mem_not_exists(self, scheduler: Scheduler) -> None:
        """/dev/mem (raw physical memory) must not exist."""
        result = await scheduler.run(
            code="import os; print(os.path.exists('/dev/mem'))",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "False"

    async def test_dev_kmem_not_exists(self, scheduler: Scheduler) -> None:
        """/dev/kmem (kernel virtual memory) must not exist."""
        result = await scheduler.run(
            code="import os; print(os.path.exists('/dev/kmem'))",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "False"

    async def test_dev_port_not_exists(self, scheduler: Scheduler) -> None:
        """/dev/port (raw I/O port access) must not exist."""
        result = await scheduler.run(
            code="import os; print(os.path.exists('/dev/port'))",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "False"

    async def test_dev_mem_open_fails(self, scheduler: Scheduler) -> None:
        """open('/dev/mem') fails even if node is somehow recreated (defense-in-depth)."""
        code = """\
try:
    f = open('/dev/mem', 'rb')
    f.close()
    print("unexpected_success")
except (FileNotFoundError, PermissionError, OSError):
    print("blocked")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout


# =============================================================================
# /dev/shm mount hardening
# =============================================================================
class TestDevShmHardening:
    """/dev/shm is mounted with nosuid, nodev, noexec flags."""

    SHM_MOUNT_LINE = """\
with open('/proc/mounts') as f:
    for line in f:
        if '/dev/shm' in line:
            print(line)
            break
"""

    async def test_noexec_flag(self, scheduler: Scheduler) -> None:
        """/dev/shm has noexec mount flag."""
        result = await scheduler.run(code=self.SHM_MOUNT_LINE, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "noexec" in result.stdout

    async def test_nosuid_flag(self, scheduler: Scheduler) -> None:
        """/dev/shm has nosuid mount flag."""
        result = await scheduler.run(code=self.SHM_MOUNT_LINE, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "nosuid" in result.stdout

    async def test_nodev_flag(self, scheduler: Scheduler) -> None:
        """/dev/shm has nodev mount flag."""
        result = await scheduler.run(code=self.SHM_MOUNT_LINE, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "nodev" in result.stdout

    async def test_exec_blocked_on_dev_shm(self, scheduler: Scheduler) -> None:
        """Script execution is blocked on /dev/shm (noexec enforcement)."""
        code = """\
import os, stat, subprocess
path = '/dev/shm/test_exec.sh'
with open(path, 'w') as f:
    f.write('#!/bin/sh\\necho pwned\\n')
os.chmod(path, stat.S_IRWXU)
try:
    r = subprocess.run([path], capture_output=True, timeout=5)
    print(f"exit={r.returncode}")
except PermissionError:
    print("blocked")
except OSError as e:
    if e.errno == 13:  # EACCES
        print("blocked")
    else:
        print(f"error={e}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_multiprocessing_pool_still_works(self, scheduler: Scheduler) -> None:
        """multiprocessing.Pool works despite noexec on /dev/shm (regression canary).

        POSIX semaphores use shm_open() + mmap(), not execve(), so noexec
        does not affect them.
        """
        code = """\
from multiprocessing import Pool

def square(x):
    return x ** 2

with Pool(processes=2) as pool:
    result = pool.map(square, [1, 2, 3, 4, 5])
print(result)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.strip() == "[1, 4, 9, 16, 25]"

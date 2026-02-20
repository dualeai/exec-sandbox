"""Tests for guest VM device-layer security hardening.

Verifies that dangerous device nodes are removed, /dev/shm is mounted
with restrictive flags, and /tmp has nosuid/nodev flags with an explicit
inode cap (nr_inodes=16384). These are defense-in-depth measures that
apply regardless of UID — even root (guest-agent) cannot access /dev/mem
after these changes.

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


# =============================================================================
# /tmp tmpfs hardening
# =============================================================================
class TestTmpHardening:
    """/tmp is mounted with nosuid, nodev, explicit nr_inodes=16384, mode=1777."""

    TMP_MOUNT_LINE = """\
with open('/proc/mounts') as f:
    for line in f:
        if ' /tmp ' in line:
            print(line.strip())
            break
"""

    # --- Normal cases: verify mount options are applied ---

    async def test_nosuid_flag(self, scheduler: Scheduler) -> None:
        """/tmp has nosuid mount flag (CIS 1.1.4)."""
        result = await scheduler.run(code=self.TMP_MOUNT_LINE, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "nosuid" in result.stdout

    async def test_nodev_flag(self, scheduler: Scheduler) -> None:
        """/tmp has nodev mount flag (CIS 1.1.3)."""
        result = await scheduler.run(code=self.TMP_MOUNT_LINE, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "nodev" in result.stdout

    async def test_no_noexec_flag(self, scheduler: Scheduler) -> None:
        """/tmp does NOT have noexec (would break uv wheel install, PyInstaller)."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        if ' /tmp ' in line:
            # Options are the 4th space-delimited field, comma-separated
            opts = line.split()[3].split(',')
            print(f'NOEXEC_PRESENT:{"noexec" in opts}')
            break
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "NOEXEC_PRESENT:False" in result.stdout

    async def test_sticky_bit(self, scheduler: Scheduler) -> None:
        """/tmp has sticky bit (mode 1777) preventing cross-user file deletion."""
        code = "import os, stat; s = os.stat('/tmp'); print(f'{stat.S_IMODE(s.st_mode):o}')"
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip() == "1777"

    async def test_inode_limit_is_explicit(self, scheduler: Scheduler) -> None:
        """Verify /tmp has an explicit inode limit via statvfs (not unlimited)."""
        code = "import os; s = os.statvfs('/tmp'); print(f'INODES:{s.f_files}')"
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip() == "INODES:16384"

    async def test_tmpfs_size_128m(self, scheduler: Scheduler) -> None:
        """Verify /tmp size is 128MB."""
        code = """\
import os
s = os.statvfs('/tmp')
size_mb = (s.f_blocks * s.f_frsize) / (1024 * 1024)
print(f'SIZE_MB:{size_mb:.0f}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip() == "SIZE_MB:128"

    # --- Normal cases: verify functionality is preserved ---

    async def test_exec_works_on_tmp(self, scheduler: Scheduler) -> None:
        """Script execution works on /tmp (noexec NOT set)."""
        code = """\
import os, stat, subprocess
path = '/tmp/test_exec.sh'
with open(path, 'w') as f:
    f.write('#!/bin/sh\\necho EXEC_OK\\n')
os.chmod(path, stat.S_IRWXU)
r = subprocess.run([path], capture_output=True, text=True, timeout=5)
print(r.stdout.strip())
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXEC_OK" in result.stdout

    async def test_tempfile_module_works(self, scheduler: Scheduler) -> None:
        """Python tempfile module works normally on /tmp."""
        code = """\
import tempfile, os
with tempfile.NamedTemporaryFile(dir='/tmp', delete=False, suffix='.txt') as f:
    f.write(b'hello')
    name = f.name
print(f'CREATED:{os.path.exists(name)}')
os.unlink(name)
print(f'DELETED:{not os.path.exists(name)}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "CREATED:True" in result.stdout
        assert "DELETED:True" in result.stdout

    # --- Edge cases: boundary conditions ---

    async def test_inode_exhaustion_produces_oserror(self, scheduler: Scheduler) -> None:
        """Creating files beyond inode limit produces OSError (ENOSPC), not crash."""
        code = """\
import os
count = 0
try:
    for i in range(20000):
        with open(f'/tmp/f_{i}', 'w') as f:
            pass
        count += 1
    print(f'FILE_LIMIT:NONE:{count}')
except OSError as e:
    print(f'FILE_LIMIT:{count}')
    print(f'ERRNO:{e.errno}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "FILE_LIMIT:NONE:" not in result.stdout
        assert "FILE_LIMIT:" in result.stdout
        # Should hit limit near 16384 (minus a few for . and existing entries)
        count = int(result.stdout.split("FILE_LIMIT:")[1].split("\n")[0])
        assert 16000 <= count <= 16384
        # ENOSPC = errno 28
        assert "ERRNO:28" in result.stdout

    async def test_files_still_writable_below_inode_limit(self, scheduler: Scheduler) -> None:
        """Can create ~1000 files and write data to them (well within 16K limit)."""
        code = """\
import os
for i in range(1000):
    with open(f'/tmp/data_{i}.txt', 'w') as f:
        f.write(f'content_{i}')
# Verify a sample
with open('/tmp/data_999.txt') as f:
    print(f'CONTENT:{f.read()}')
print(f'COUNT:{len(os.listdir("/tmp"))}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=30)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "CONTENT:content_999" in result.stdout
        assert "COUNT:" in result.stdout, f"stdout: {result.stdout}"
        count = int(result.stdout.split("COUNT:")[1].strip())
        assert count >= 1000

    async def test_subdirectories_consume_inodes(self, scheduler: Scheduler) -> None:
        """Directories consume inodes too — mkdir counts against nr_inodes."""
        code = """\
import os
count = 0
try:
    for i in range(20000):
        os.mkdir(f'/tmp/dir_{i}')
        count += 1
    print(f'NO_LIMIT:{count}')
except OSError:
    print(f'DIR_LIMIT:{count}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "DIR_LIMIT:" in result.stdout, f"stdout: {result.stdout}"
        count = int(result.stdout.split("DIR_LIMIT:")[1].strip())
        assert 16000 <= count <= 16384

    # --- Weird cases: unusual patterns ---

    async def test_rapid_create_delete_cycle(self, scheduler: Scheduler) -> None:
        """Rapid create/delete cycles reclaim inodes correctly."""
        code = """\
import os
# Create and delete 50K files — should never hit limit if inodes are reclaimed
for i in range(50000):
    path = f'/tmp/cycle_{i % 100}'
    with open(path, 'w') as f:
        f.write('x')
    os.unlink(path)
print('CYCLE_OK')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0
        assert "CYCLE_OK" in result.stdout

    async def test_symlinks_consume_inodes(self, scheduler: Scheduler) -> None:
        """Symlinks consume inodes from the same pool."""
        code = """\
import os
with open('/tmp/target', 'w') as f:
    f.write('data')
count = 0
try:
    for i in range(20000):
        os.symlink('/tmp/target', f'/tmp/link_{i}')
        count += 1
    print(f'NO_LIMIT:{count}')
except OSError:
    print(f'SYMLINK_LIMIT:{count}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "SYMLINK_LIMIT:" in result.stdout, f"stdout: {result.stdout}"
        count = int(result.stdout.split("SYMLINK_LIMIT:")[1].strip())
        # 1 target file + N symlinks; should exhaust near 16384
        assert 16000 <= count <= 16384

    async def test_hardlinks_consume_inodes(self, scheduler: Scheduler) -> None:
        """Hard links on tmpfs consume inodes (each dentry needs its own inode).

        Unlike on-disk filesystems, tmpfs allocates a new inode per dentry,
        so hard links count against nr_inodes the same as regular files.
        """
        code = """\
import os
s_before = os.statvfs('/tmp')
# Create a source file (1 inode)
with open('/tmp/src', 'w') as f:
    f.write('data')
# Create 100 hard links — each consumes an inode on tmpfs
for i in range(100):
    os.link('/tmp/src', f'/tmp/hard_{i}')
s_after = os.statvfs('/tmp')
inodes_used = s_before.f_ffree - s_after.f_ffree
print(f'INODES_USED:{inodes_used}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "INODES_USED:" in result.stdout, f"stdout: {result.stdout}"
        inodes_used = int(result.stdout.split("INODES_USED:")[1].strip())
        # 1 source file + 100 hard links = 101 inodes on tmpfs
        # (tmpfs calls shmem_reserve_inode() per link, unlike on-disk filesystems)
        assert inodes_used == 101

    async def test_empty_files_same_cost_as_large_files(self, scheduler: Scheduler) -> None:
        """Empty files consume 1 inode each — same as files with data."""
        code = """\
import os
s_before = os.statvfs('/tmp')
# Create 100 empty files and 100 files with 1KB data
for i in range(100):
    with open(f'/tmp/empty_{i}', 'w') as f:
        pass
for i in range(100):
    with open(f'/tmp/data_{i}', 'w') as f:
        f.write('x' * 1024)
s_after = os.statvfs('/tmp')
inodes_used = s_before.f_ffree - s_after.f_ffree
print(f'INODES_USED:{inodes_used}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=30)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "INODES_USED:" in result.stdout, f"stdout: {result.stdout}"
        inodes_used = int(result.stdout.split("INODES_USED:")[1].strip())
        # Should be exactly 200 (100 empty + 100 with data)
        assert inodes_used == 200

    # --- Out of bounds: security enforcement ---

    async def test_mknod_blocked(self, scheduler: Scheduler) -> None:
        """Device node creation on /tmp fails (nodev + non-root, defense-in-depth).

        Non-root lacks CAP_MKNOD (primary block). The nodev mount flag is a
        second layer that would block even a root process from creating device
        nodes on this mount.
        """
        code = """\
import os, errno
try:
    os.mknod('/tmp/fake_null', 0o666 | 0o020000, os.makedev(1, 3))
    print('CREATED')
except PermissionError:
    print(f'BLOCKED:{errno.EPERM}')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.startswith("BLOCKED:")

    async def test_suid_bit_ignored_on_tmp(self, scheduler: Scheduler) -> None:
        """SUID bit on /tmp binaries is ignored during execve() due to nosuid.

        Copies a real ELF binary (id) to /tmp, sets SUID, and executes it.
        With nosuid, the effective UID remains the caller (1000), not root (0).
        Uses a subdirectory to preserve the basename — Alpine's busybox
        dispatches applets by argv[0] basename.
        """
        code = """\
import shutil, os, stat, subprocess
# Copy a real binary — shell scripts ignore SUID (kernel only honors it on ELF).
# Preserve basename "id" because Alpine uses busybox, which dispatches by argv[0].
src = shutil.which('id')
os.makedirs('/tmp/suid_test', exist_ok=True)
dst = '/tmp/suid_test/id'
shutil.copy2(src, dst)
os.chmod(dst, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH | stat.S_ISUID)
actual = os.stat(dst)
print(f'SUID_BIT_SET:{bool(actual.st_mode & stat.S_ISUID)}')
# Execute it — nosuid means kernel ignores SUID during execve()
r = subprocess.run([dst], capture_output=True, text=True, timeout=5)
print(f'OUTPUT:{r.stdout.strip()}')
# nosuid should prevent euid escalation to root.
# id(1) prints "euid=0(root)" only when effective UID differs from real UID.
print(f'NO_ROOT_EUID:{"euid=0" not in r.stdout}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=30)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "SUID_BIT_SET:True" in result.stdout
        assert "NO_ROOT_EUID:True" in result.stdout

    async def test_inode_exhaustion_doesnt_crash_vm(self, scheduler: Scheduler) -> None:
        """Exhausting /tmp inodes doesn't crash the VM — other paths still work."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            # Step 1: Exhaust inodes
            exhaust = """\
import os
count = 0
try:
    for i in range(20000):
        with open(f'/tmp/exhaust_{i}', 'w') as f:
            pass
        count += 1
except OSError:
    pass
print(f'EXHAUSTED:{count}')
"""
            r1 = await session.exec(exhaust, timeout_seconds=60)
            assert r1.exit_code == 0, f"stderr: {r1.stderr}"
            assert "EXHAUSTED:" in r1.stdout

            # Step 2: Verify /tmp writes fail
            r2 = await session.exec(
                """\
try:
    with open('/tmp/should_fail', 'w') as f:
        f.write('test')
    print('UNEXPECTED_SUCCESS')
except OSError:
    print('CORRECTLY_BLOCKED')
""",
                timeout_seconds=30,
            )
            assert r2.exit_code == 0, f"stderr: {r2.stderr}"
            assert "CORRECTLY_BLOCKED" in r2.stdout

            # Step 3: Verify rootfs (/home/user) still works
            r3 = await session.exec(
                """\
with open('/home/user/still_works.txt', 'w') as f:
    f.write('alive')
with open('/home/user/still_works.txt') as f:
    print(f'ROOTFS:{f.read()}')
""",
                timeout_seconds=30,
            )
            assert r3.exit_code == 0, f"stderr: {r3.stderr}"
            assert "ROOTFS:alive" in r3.stdout

            # Step 4: Verify Python execution still works (REPL uses rootfs, not /tmp)
            r4 = await session.exec("print(f'MATH:{2 + 2}')", timeout_seconds=30)
            assert r4.exit_code == 0, f"stderr: {r4.stderr}"
            assert "MATH:4" in r4.stdout

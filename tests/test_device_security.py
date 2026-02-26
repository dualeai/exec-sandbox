"""Tests for guest VM device-layer security hardening.

Verifies that dangerous device nodes are removed, block device nodes
(/dev/vda, /dev/zram0) are removed after use, /dev is bind-remounted
read-only (preventing mknod with EROFS), /dev/shm is mounted with
restrictive flags, /tmp has nosuid/nodev flags with an explicit inode
cap (nr_inodes=16384), /proc/sys is read-only, and /bin + /sbin are
bind-mounted read-only with nosuid (blocking path hijack attacks).
These are defense-in-depth measures that apply regardless of UID — even
root (guest-agent) cannot access /dev/mem or raw-read the root disk
after these changes.

Complements test_nonroot_repl.py (privilege escalation from UID 1000) with
kernel/device-layer hardening that blocks access at the filesystem level.
"""

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


# =============================================================================
# Dangerous device nodes should not exist
# =============================================================================
class TestDangerousDeviceNodes:
    """Dangerous device nodes are removed by tiny-init after devtmpfs mount."""

    async def test_dev_mem_not_exists(self, dual_scheduler: Scheduler) -> None:
        """/dev/mem (raw physical memory) must not exist."""
        result = await dual_scheduler.run(
            code="import os; print(os.path.exists('/dev/mem'))",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "False"

    async def test_dev_kmem_not_exists(self, dual_scheduler: Scheduler) -> None:
        """/dev/kmem (kernel virtual memory) must not exist."""
        result = await dual_scheduler.run(
            code="import os; print(os.path.exists('/dev/kmem'))",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "False"

    async def test_dev_port_not_exists(self, dual_scheduler: Scheduler) -> None:
        """/dev/port (raw I/O port access) must not exist."""
        result = await dual_scheduler.run(
            code="import os; print(os.path.exists('/dev/port'))",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "False"

    async def test_dev_mem_open_fails(self, dual_scheduler: Scheduler) -> None:
        """open('/dev/mem') fails even if node is somehow recreated (defense-in-depth)."""
        code = """\
try:
    f = open('/dev/mem', 'rb')
    f.close()
    print("unexpected_success")
except (FileNotFoundError, PermissionError, OSError):
    print("blocked")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
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

    async def test_noexec_flag(self, dual_scheduler: Scheduler) -> None:
        """/dev/shm has noexec mount flag."""
        result = await dual_scheduler.run(code=self.SHM_MOUNT_LINE, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "noexec" in result.stdout

    async def test_nosuid_flag(self, dual_scheduler: Scheduler) -> None:
        """/dev/shm has nosuid mount flag."""
        result = await dual_scheduler.run(code=self.SHM_MOUNT_LINE, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "nosuid" in result.stdout

    async def test_nodev_flag(self, dual_scheduler: Scheduler) -> None:
        """/dev/shm has nodev mount flag."""
        result = await dual_scheduler.run(code=self.SHM_MOUNT_LINE, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "nodev" in result.stdout

    async def test_exec_blocked_on_dev_shm(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_multiprocessing_pool_still_works(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(
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

    async def test_nosuid_flag(self, dual_scheduler: Scheduler) -> None:
        """/tmp has nosuid mount flag (CIS 1.1.4)."""
        result = await dual_scheduler.run(code=self.TMP_MOUNT_LINE, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "nosuid" in result.stdout

    async def test_nodev_flag(self, dual_scheduler: Scheduler) -> None:
        """/tmp has nodev mount flag (CIS 1.1.3)."""
        result = await dual_scheduler.run(code=self.TMP_MOUNT_LINE, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "nodev" in result.stdout

    async def test_no_noexec_flag(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "NOEXEC_PRESENT:False" in result.stdout

    async def test_sticky_bit(self, dual_scheduler: Scheduler) -> None:
        """/tmp has sticky bit (mode 1777) preventing cross-user file deletion."""
        code = "import os, stat; s = os.stat('/tmp'); print(f'{stat.S_IMODE(s.st_mode):o}')"
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip() == "1777"

    async def test_inode_limit_is_explicit(self, dual_scheduler: Scheduler) -> None:
        """Verify /tmp has an explicit inode limit via statvfs (not unlimited)."""
        code = "import os; s = os.statvfs('/tmp'); print(f'INODES:{s.f_files}')"
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip() == "INODES:16384"

    async def test_tmpfs_size_128m(self, dual_scheduler: Scheduler) -> None:
        """Verify /tmp size is 128MB."""
        code = """\
import os
s = os.statvfs('/tmp')
size_mb = (s.f_blocks * s.f_frsize) / (1024 * 1024)
print(f'SIZE_MB:{size_mb:.0f}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip() == "SIZE_MB:128"

    # --- Normal cases: verify functionality is preserved ---

    async def test_exec_works_on_tmp(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXEC_OK" in result.stdout

    async def test_tempfile_module_works(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "CREATED:True" in result.stdout
        assert "DELETED:True" in result.stdout

    # --- Edge cases: boundary conditions ---

    async def test_inode_exhaustion_produces_oserror(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "FILE_LIMIT:NONE:" not in result.stdout
        assert "FILE_LIMIT:" in result.stdout
        # Should hit limit near 16384 (minus a few for . and existing entries)
        count = int(result.stdout.split("FILE_LIMIT:")[1].split("\n")[0])
        assert 16000 <= count <= 16384
        # ENOSPC = errno 28
        assert "ERRNO:28" in result.stdout

    async def test_files_still_writable_below_inode_limit(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=30)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "CONTENT:content_999" in result.stdout
        assert "COUNT:" in result.stdout, f"stdout: {result.stdout}"
        count = int(result.stdout.split("COUNT:")[1].strip())
        assert count >= 1000

    async def test_subdirectories_consume_inodes(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "DIR_LIMIT:" in result.stdout, f"stdout: {result.stdout}"
        count = int(result.stdout.split("DIR_LIMIT:")[1].strip())
        assert 16000 <= count <= 16384

    # --- Weird cases: unusual patterns ---

    async def test_rapid_create_delete_cycle(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0
        assert "CYCLE_OK" in result.stdout

    async def test_symlinks_consume_inodes(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "SYMLINK_LIMIT:" in result.stdout, f"stdout: {result.stdout}"
        count = int(result.stdout.split("SYMLINK_LIMIT:")[1].strip())
        # 1 target file + N symlinks; should exhaust near 16384
        assert 16000 <= count <= 16384

    async def test_hardlinks_consume_inodes(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "INODES_USED:" in result.stdout, f"stdout: {result.stdout}"
        inodes_used = int(result.stdout.split("INODES_USED:")[1].strip())
        # 1 source file + 100 hard links = 101 inodes on tmpfs
        # (tmpfs calls shmem_reserve_inode() per link, unlike on-disk filesystems)
        assert inodes_used == 101

    async def test_empty_files_same_cost_as_large_files(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=30)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "INODES_USED:" in result.stdout, f"stdout: {result.stdout}"
        inodes_used = int(result.stdout.split("INODES_USED:")[1].strip())
        # Should be exactly 200 (100 empty + 100 with data)
        assert inodes_used == 200

    # --- Out of bounds: security enforcement ---

    async def test_mknod_blocked(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.startswith("BLOCKED:")

    async def test_suid_bit_ignored_on_tmp(self, dual_scheduler: Scheduler) -> None:
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=30)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "SUID_BIT_SET:True" in result.stdout
        assert "NO_ROOT_EUID:True" in result.stdout

    async def test_inode_exhaustion_doesnt_crash_vm(self, dual_scheduler: Scheduler) -> None:
        """Exhausting /tmp inodes doesn't crash the VM — other paths still work."""
        async with await dual_scheduler.session(language=Language.PYTHON) as session:
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

            # Step 3: Verify /home/user (tmpfs) still works
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


# =============================================================================
# Block device hardening
# =============================================================================
class TestBlockDeviceHardening:
    """Block device nodes (/dev/vda, /dev/zram0) are removed by tiny-init after use."""

    # --- Normal cases: device nodes removed and filesystem still works ---

    async def test_dev_vda_not_exists(self, dual_scheduler: Scheduler) -> None:
        """/dev/vda (root block device) must not exist after mount."""
        result = await dual_scheduler.run(
            code="import os; print(os.path.exists('/dev/vda'))",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "False"

    async def test_zram_swap_active(self, dual_scheduler: Scheduler) -> None:
        """zram swap is active — guest-agent sets up /dev/zram0 then removes the device node for security."""
        result = await dual_scheduler.run(
            code="print('zram0' in open('/proc/swaps').read())",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "True"

    async def test_filesystem_still_works(self, dual_scheduler: Scheduler) -> None:
        """Read/write on the root filesystem works after /dev/vda removal."""
        code = """\
import os
path = '/home/user/test_blkdev.txt'
with open(path, 'w') as f:
    f.write('block device node removed')
with open(path) as f:
    data = f.read()
os.unlink(path)
print(f'OK:{data}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "OK:block device node removed" in result.stdout

    # --- Edge cases: alternative paths to block device are closed ---

    async def test_dev_block_symlinks_broken(self, dual_scheduler: Scheduler) -> None:
        """/dev/block/ directory is removed by tiny-init (eliminates symlink access path)."""
        code = """\
import os
print(f'EXISTS:{os.path.exists("/dev/block")}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXISTS:False" in result.stdout

    async def test_dev_root_not_accessible(self, dual_scheduler: Scheduler) -> None:
        """/dev/root (sometimes auto-created by kernel) doesn't exist or is broken."""
        code = """\
import os
exists = os.path.exists('/dev/root')
if exists:
    # If it exists, verify it's not a working block device
    try:
        f = open('/dev/root', 'rb')
        f.close()
        print('ACCESSIBLE')
    except (PermissionError, OSError):
        print('BLOCKED')
else:
    print('NOT_EXISTS')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip() in ("NOT_EXISTS", "BLOCKED")

    async def test_proc_partitions_readable_but_no_data_access(self, dual_scheduler: Scheduler) -> None:
        """/proc/partitions shows vda metadata but the device node is gone."""
        code = """\
# /proc/partitions should still list vda (kernel metadata, not a device node)
with open('/proc/partitions') as f:
    content = f.read()
has_vda = 'vda' in content
print(f'PROC_HAS_VDA:{has_vda}')

# But opening /dev/vda should fail
try:
    f = open('/dev/vda', 'rb')
    f.close()
    print('DEV_VDA_OPEN:success')
except FileNotFoundError:
    print('DEV_VDA_OPEN:not_found')
except (PermissionError, OSError):
    print('DEV_VDA_OPEN:blocked')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "PROC_HAS_VDA:True" in result.stdout
        assert "DEV_VDA_OPEN:not_found" in result.stdout

    # --- Weird cases: attempts to recreate block device access ---

    async def test_mknod_blocked(self, dual_scheduler: Scheduler) -> None:
        """mknod with block device type fails — UID 1000 lacks CAP_MKNOD."""
        code = """\
import os, stat
try:
    # Attempt to create a block device node (major 253 = virtblk)
    os.mknod('/home/user/vda', 0o660 | stat.S_IFBLK, os.makedev(253, 0))
    print('CREATED')
except PermissionError:
    print('BLOCKED:EPERM')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip().startswith("BLOCKED:")

    async def test_dev_vda_open_fails(self, dual_scheduler: Scheduler) -> None:
        """open('/dev/vda') fails (defense-in-depth, same pattern as /dev/mem test)."""
        code = """\
try:
    f = open('/dev/vda', 'rb')
    f.close()
    print("unexpected_success")
except (FileNotFoundError, PermissionError, OSError):
    print("blocked")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    # --- Out of bounds cases: no block device nodes remain anywhere in /dev ---

    async def test_no_block_devices_in_dev(self, dual_scheduler: Scheduler) -> None:
        """Only expected block device nodes exist under /dev (zram for swap)."""
        code = """\
import os, stat

# zram0 is expected: used for compressed swap in microVM
allowed = {'/dev/zram0'}
block_devices = []
for dirpath, dirnames, filenames in os.walk('/dev'):
    for name in filenames:
        path = os.path.join(dirpath, name)
        try:
            s = os.lstat(path)
            if stat.S_ISBLK(s.st_mode) and path not in allowed:
                block_devices.append(path)
        except OSError:
            pass

if block_devices:
    print(f'FOUND:{",".join(block_devices)}')
else:
    print('NONE')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip() == "NONE", f"Unexpected block device nodes found: {result.stdout.strip()}"


# =============================================================================
# Loop device + filesystem creation blocked
# =============================================================================
class TestLoopDeviceBlocked:
    """Loop device infrastructure is absent — no module, no device nodes, no tools.

    Loop devices are a proven attack vector: CVE-2025-8067 (udisks loop device
    LPE, CVSS 8.5), CVE-2025-6019 (SUID shell via loop-mounted XFS without
    nosuid), and multiple ext4/btrfs/xfs kernel bugs triggered by mounting
    crafted filesystem images (CVE-2025-38220, CVE-2025-38222, CVE-2024-39472).

    5 independent layers block loop device + mkfs operations:
    1. No loop.ko in initramfs (module list: virtio, ext4, zram only)
    2. modules_disabled=1 (irreversible sysctl set by tiny-init)
    3. No /dev/loop* nodes (kernel never creates them without loop module)
    4. /dev is read-only (mknod returns EROFS)
    5. No CAP_MKNOD (UID 1000, verified in TestDevReadonlyHardening)
    """

    # --- Normal: verify loop infrastructure absent ---

    async def test_no_loop_devices_exist(self, dual_scheduler: Scheduler) -> None:
        """No /dev/loop* nodes exist anywhere under /dev."""
        code = """\
import os

loop_devices = []
for dirpath, dirnames, filenames in os.walk('/dev'):
    for name in filenames:
        if name.startswith('loop'):
            path = os.path.join(dirpath, name)
            loop_devices.append(path)

if loop_devices:
    print(f'FOUND:{",".join(loop_devices)}')
else:
    print('NONE')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip() == "NONE", f"Loop device nodes found: {result.stdout.strip()}"

    async def test_loop_module_not_loaded(self, dual_scheduler: Scheduler) -> None:
        """loop kernel module is not loaded (or modules disabled entirely)."""
        code = """\
try:
    with open('/proc/modules') as f:
        modules = [line.split()[0] for line in f]
    print(f'LOOP_LOADED:{"loop" in modules}')
except FileNotFoundError:
    # CONFIG_MODULES=n: /proc/modules doesn't exist — no modules at all
    print('MODULES_DISABLED')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "LOOP_LOADED:False" in result.stdout or "MODULES_DISABLED" in result.stdout

    async def test_dev_loop_control_not_exists(self, dual_scheduler: Scheduler) -> None:
        """/dev/loop-control (dynamic loop allocation, Linux 3.1+) does not exist."""
        result = await dual_scheduler.run(
            code="import os; print(os.path.exists('/dev/loop-control'))",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "False"

    # --- Edge: tools that need loop devices fail gracefully ---

    async def test_losetup_associate_fails(self, dual_scheduler: Scheduler) -> None:
        """losetup cannot associate a file with a loop device — no /dev/loop* nodes."""
        code = """\
import subprocess

# Create a small file to use as backing store
subprocess.run(
    ["dd", "if=/dev/zero", "of=/tmp/lo.img", "bs=1k", "count=64"],
    capture_output=True, timeout=5,
)

# Try to associate it with a loop device
r = subprocess.run(
    ["losetup", "/dev/loop0", "/tmp/lo.img"],
    capture_output=True, text=True, timeout=5,
)
print(f'LOSETUP_RC:{r.returncode}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=30)
        assert result.exit_code == 0
        assert "LOSETUP_RC:0" not in result.stdout, f"Expected losetup associate to fail.\nstdout: {result.stdout}"

    async def test_mkfs_ext4_not_exploitable(self, dual_scheduler: Scheduler) -> None:
        """mkfs.ext4 exists (needed by tiny-init for snapshots) but cannot format block devices.

        e2fsprogs was added in b836e56 for ext4 overlay drive formatting by
        tiny-init (running as root).  The sandbox user (UID 1000) cannot
        exploit it because loop devices are blocked (CONFIG_BLK_DEV_LOOP=n)
        and /dev/vd* nodes are not accessible.
        """
        code = """\
import subprocess, shutil
has_mkfs = shutil.which("mkfs.ext4") is not None
print(f'HAS_MKFS:{has_mkfs}')
# Attempt to format the rootfs block device — must fail
r = subprocess.run(
    ["mkfs.ext4", "-F", "/dev/vda"],
    capture_output=True, text=True, timeout=5,
)
print(f'FORMAT_RC:{r.returncode}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "HAS_MKFS:True" in result.stdout
        assert "FORMAT_RC:0" not in result.stdout, (
            f"Sandbox user should not be able to format block devices.\nstdout: {result.stdout}"
        )

    # --- Out of bounds: full attack chains blocked ---

    async def test_loop_module_load_blocked(self, dual_scheduler: Scheduler) -> None:
        """modprobe loop fails — modules_disabled=1 is irreversible."""
        result = await dual_scheduler.run(
            code="modprobe loop 2>&1; echo EXIT:$?",
            language=Language.RAW,
        )
        assert result.exit_code == 0
        assert "EXIT:0" not in result.stdout, (
            f"Expected modprobe loop to fail with modules_disabled=1.\nstdout: {result.stdout}"
        )

    async def test_file_backed_loop_mount_blocked(self, dual_scheduler: Scheduler) -> None:
        """Full CVE-2025-6019 attack chain: create file, losetup, mount — every step fails."""
        code = """\
import subprocess

results = []

# Step 1: Create a 1MB file (should succeed — /tmp is writable)
r = subprocess.run(
    ["dd", "if=/dev/zero", "of=/tmp/fake.img", "bs=1M", "count=1"],
    capture_output=True, text=True, timeout=10,
)
results.append(f'DD_EXIT:{r.returncode}')

# Step 2: Try losetup (should fail — no loop devices)
r = subprocess.run(
    ["losetup", "-f", "/tmp/fake.img"],
    capture_output=True, text=True, timeout=5,
)
results.append(f'LOSETUP_EXIT:{r.returncode}')

# Step 3: Try direct mount (should fail — no CAP_SYS_ADMIN)
r = subprocess.run(
    ["mount", "-o", "loop", "/tmp/fake.img", "/mnt"],
    capture_output=True, text=True, timeout=5,
)
results.append(f'MOUNT_EXIT:{r.returncode}')

for line in results:
    print(line)
"""
        result = await dual_scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )
        assert result.exit_code == 0
        # dd may succeed (just creates a regular file), but losetup and mount must fail
        assert "LOSETUP_EXIT:0" not in result.stdout, f"Expected losetup to fail.\nstdout: {result.stdout}"
        assert "MOUNT_EXIT:0" not in result.stdout, f"Expected mount to fail.\nstdout: {result.stdout}"


# =============================================================================
# /proc/sys read-only hardening
# =============================================================================
class TestProcSysHardening:
    """/proc/sys is bind-mounted read-only to prevent sysctl modification.

    Blocks the most dangerous privilege escalation vectors:
    - core_pattern pipe-to-binary (arbitrary root code execution on crash)
    - modprobe path hijack (root code execution on unknown binary format)
    - randomize_va_space=0 (disables ASLR system-wide)
    - ip_forward (network pivoting), hostname, panic, panic_on_oom (DoS)

    All sysctl tuning is done by tiny-init during setup_zram() before
    switch_root, so no writes to /proc/sys are needed at runtime.

    See: CVE-2025-31133, CVE-2025-52565, CVE-2025-52881 (runc /proc/sys escapes)
    """

    # --- Normal: direct writes to dangerous sysctl paths ---

    async def test_proc_sys_mount_is_readonly(self, dual_scheduler: Scheduler) -> None:
        """/proc/sys has ro flag in /proc/mounts."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/proc/sys':
            opts = parts[3].split(',')
            print(f'RO:{"ro" in opts}')
            break
    else:
        print('MOUNT_NOT_FOUND')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "RO:True" in result.stdout, f"Expected /proc/sys to be mounted read-only.\nstdout: {result.stdout}"

    async def test_write_core_pattern_blocked(self, dual_scheduler: Scheduler) -> None:
        """Critical: pipe-to-binary root execution via core_pattern cannot be set."""
        code = """\
try:
    with open('/proc/sys/kernel/core_pattern', 'w') as f:
        f.write('|/tmp/pwned')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected write to core_pattern to be blocked.\nstdout: {result.stdout}"
        )

    async def test_write_modprobe_blocked(self, dual_scheduler: Scheduler) -> None:
        """Critical: module autoloader path cannot be hijacked."""
        code = """\
try:
    with open('/proc/sys/kernel/modprobe', 'w') as f:
        f.write('/tmp/pwned')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected write to modprobe to be blocked.\nstdout: {result.stdout}"
        )

    async def test_write_randomize_va_space_blocked(self, dual_scheduler: Scheduler) -> None:
        """ASLR cannot be disabled via randomize_va_space."""
        code = """\
try:
    with open('/proc/sys/kernel/randomize_va_space', 'w') as f:
        f.write('0')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected write to randomize_va_space to be blocked.\nstdout: {result.stdout}"
        )

    async def test_write_ip_forward_blocked(self, dual_scheduler: Scheduler) -> None:
        """IP forwarding cannot be enabled (prevents network pivoting)."""
        code = """\
try:
    with open('/proc/sys/net/ipv4/ip_forward', 'w') as f:
        f.write('1')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected write to ip_forward to be blocked.\nstdout: {result.stdout}"
        )

    async def test_write_hostname_blocked(self, dual_scheduler: Scheduler) -> None:
        """Hostname sysctl cannot be changed."""
        code = """\
try:
    with open('/proc/sys/kernel/hostname', 'w') as f:
        f.write('pwned')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected write to hostname to be blocked.\nstdout: {result.stdout}"
        )

    async def test_write_overcommit_memory_blocked(self, dual_scheduler: Scheduler) -> None:
        """OOM policy cannot be changed via overcommit_memory."""
        code = """\
try:
    with open('/proc/sys/vm/overcommit_memory', 'w') as f:
        f.write('1')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected write to overcommit_memory to be blocked.\nstdout: {result.stdout}"
        )

    async def test_write_panic_blocked(self, dual_scheduler: Scheduler) -> None:
        """Panic behavior cannot be changed."""
        code = """\
try:
    with open('/proc/sys/kernel/panic', 'w') as f:
        f.write('1')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), f"Expected write to panic to be blocked.\nstdout: {result.stdout}"

    async def test_write_panic_on_oom_blocked(self, dual_scheduler: Scheduler) -> None:
        """Kernel panic on OOM cannot be enabled."""
        code = """\
try:
    with open('/proc/sys/vm/panic_on_oom', 'w') as f:
        f.write('1')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected write to panic_on_oom to be blocked.\nstdout: {result.stdout}"
        )

    async def test_read_sysctl_still_works(self, dual_scheduler: Scheduler) -> None:
        """Reading sysctl values still works (read-only, not hidden)."""
        code = """\
with open('/proc/sys/vm/swappiness') as f:
    val = f.read().strip()
print(f'SWAPPINESS:{val}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "SWAPPINESS:" in result.stdout
        val = int(result.stdout.split("SWAPPINESS:")[1].strip())
        assert 0 <= val <= 200  # valid swappiness range

    # --- Edge: indirect write vectors ---

    async def test_shell_redirect_write_blocked(self, dual_scheduler: Scheduler) -> None:
        """Shell redirect to /proc/sys is blocked too."""
        code = """\
import subprocess
r = subprocess.run(
    ["sh", "-c", "echo 0 > /proc/sys/kernel/randomize_va_space 2>&1; echo EXIT:$?"],
    capture_output=True, text=True, timeout=5,
)
print(r.stdout)
print(r.stderr, end='')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        out = result.stdout + result.stderr
        assert "EXIT:0" not in result.stdout or "Read-only" in out or "Permission denied" in out

    async def test_open_modes_all_blocked(self, dual_scheduler: Scheduler) -> None:
        """O_WRONLY, O_RDWR, and O_WRONLY|O_APPEND are all rejected on /proc/sys files."""
        code = """\
import os

path = '/proc/sys/kernel/hostname'
results = []
for mode_name, mode_flag in [('O_WRONLY', os.O_WRONLY), ('O_RDWR', os.O_RDWR), ('O_APPEND', os.O_WRONLY | os.O_APPEND)]:
    try:
        fd = os.open(path, mode_flag)
        os.close(fd)
        results.append(f'{mode_name}:OPEN')
    except OSError as e:
        results.append(f'{mode_name}:BLOCKED:{e.errno}')
for r in results:
    print(r)
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "O_WRONLY:BLOCKED:" in result.stdout
        assert "O_RDWR:BLOCKED:" in result.stdout
        assert "O_APPEND:BLOCKED:" in result.stdout

    async def test_nested_path_write_blocked(self, dual_scheduler: Scheduler) -> None:
        """Read-only applies recursively to deeply nested sysctl paths."""
        code = """\
import os

# Try a deeply nested path under /proc/sys
nested_paths = [
    '/proc/sys/net/ipv4/conf/all/accept_redirects',
    '/proc/sys/net/ipv4/conf/all/send_redirects',
    '/proc/sys/net/core/somaxconn',
]
for path in nested_paths:
    if os.path.exists(path):
        try:
            with open(path, 'w') as f:
                f.write('0')
            print(f'{path}:WRITTEN')
        except OSError as e:
            print(f'{path}:BLOCKED:{e.errno}')
        break
else:
    print('NO_NESTED_PATH_FOUND')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "BLOCKED:" in result.stdout, f"Expected nested sysctl write to be blocked.\nstdout: {result.stdout}"

    async def test_symlink_write_through_blocked(self, dual_scheduler: Scheduler) -> None:
        """Symlink pointing to /proc/sys path — write through symlink is blocked."""
        code = """\
import os

os.symlink('/proc/sys/kernel/hostname', '/tmp/sys_link')
try:
    with open('/tmp/sys_link', 'w') as f:
        f.write('pwned')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
finally:
    os.unlink('/tmp/sys_link')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected symlink write-through to be blocked.\nstdout: {result.stdout}"
        )

    # --- Weird: unconventional bypass attempts ---

    async def test_proc_self_fd_reopen_blocked(self, dual_scheduler: Scheduler) -> None:
        """Open O_RDONLY, reopen via /proc/self/fd/{fd} O_WRONLY — still blocked."""
        code = """\
import os

fd_ro = os.open('/proc/sys/kernel/hostname', os.O_RDONLY)
try:
    fd_rw = os.open(f'/proc/self/fd/{fd_ro}', os.O_WRONLY)
    os.close(fd_rw)
    print('REOPENED')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
finally:
    os.close(fd_ro)
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected /proc/self/fd reopen to be blocked.\nstdout: {result.stdout}"
        )

    async def test_ctypes_direct_syscall_blocked(self, dual_scheduler: Scheduler) -> None:
        """libc.open(path, O_WRONLY) — kernel enforcement, not Python."""
        code = """\
import ctypes, os, errno

libc = ctypes.CDLL("libc.so.6", use_errno=True)
fd = libc.open(b"/proc/sys/kernel/core_pattern", os.O_WRONLY)
err = ctypes.get_errno()
if fd == -1:
    if err == errno.EROFS:
        print("BLOCKED:erofs")
    elif err == errno.EACCES:
        print("BLOCKED:eacces")
    else:
        print(f"BLOCKED:errno_{err}")
else:
    os.close(fd)
    print("OPENED")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), f"Expected ctypes open to be blocked.\nstdout: {result.stdout}"

    async def test_truncate_sysctl_blocked(self, dual_scheduler: Scheduler) -> None:
        """os.truncate() on a sysctl file is blocked at VFS layer."""
        code = """\
import os

try:
    os.truncate('/proc/sys/kernel/hostname', 0)
    print('TRUNCATED')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), f"Expected truncate to be blocked.\nstdout: {result.stdout}"

    async def test_binfmt_misc_register_blocked(self, dual_scheduler: Scheduler) -> None:
        """Cannot register custom binary format interpreters via binfmt_misc."""
        code = """\
import os

# binfmt_misc register path (may or may not exist in minimal VM)
register_path = '/proc/sys/fs/binfmt_misc/register'
if not os.path.exists(register_path):
    # binfmt_misc not mounted — also safe
    print('BLOCKED:not_mounted')
else:
    try:
        with open(register_path, 'w') as f:
            f.write(':test:M::MZ::/tmp/handler:')
        print('REGISTERED')
    except OSError as e:
        print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected binfmt_misc register to be blocked.\nstdout: {result.stdout}"
        )

    # --- Out of bounds: privilege escalation to remount ---

    async def test_remount_rw_blocked(self, dual_scheduler: Scheduler) -> None:
        """mount -o remount,rw /proc/sys requires CAP_SYS_ADMIN."""
        code = """\
import subprocess
r = subprocess.run(
    ["mount", "-o", "remount,rw", "/proc/sys"],
    capture_output=True, text=True, timeout=5,
)
print(f"EXIT:{r.returncode}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXIT:0" not in result.stdout, (
            f"Expected remount to fail without CAP_SYS_ADMIN.\nstdout: {result.stdout}"
        )

    async def test_bind_mount_shadow_blocked(self, dual_scheduler: Scheduler) -> None:
        """Bind tmpfs over /proc/sys requires CAP_SYS_ADMIN."""
        code = """\
import subprocess
r = subprocess.run(
    ["mount", "-t", "tmpfs", "none", "/proc/sys"],
    capture_output=True, text=True, timeout=5,
)
print(f"EXIT:{r.returncode}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXIT:0" not in result.stdout, f"Expected tmpfs shadow mount to fail.\nstdout: {result.stdout}"

    async def test_umount_proc_sys_blocked(self, dual_scheduler: Scheduler) -> None:
        """umount /proc/sys requires CAP_SYS_ADMIN."""
        code = """\
import subprocess
r = subprocess.run(
    ["umount", "/proc/sys"],
    capture_output=True, text=True, timeout=5,
)
print(f"EXIT:{r.returncode}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXIT:0" not in result.stdout, f"Expected umount to fail without CAP_SYS_ADMIN.\nstdout: {result.stdout}"

    async def test_mount_second_proc_blocked(self, dual_scheduler: Scheduler) -> None:
        """Mount new procfs at /tmp/proc requires CAP_SYS_ADMIN."""
        code = """\
import subprocess, os
os.makedirs("/tmp/proc", exist_ok=True)
r = subprocess.run(
    ["mount", "-t", "proc", "proc", "/tmp/proc"],
    capture_output=True, text=True, timeout=5,
)
print(f"EXIT:{r.returncode}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXIT:0" not in result.stdout, f"Expected secondary procfs mount to fail.\nstdout: {result.stdout}"

    async def test_sysrq_disabled(self, dual_scheduler: Scheduler) -> None:
        """kernel.sysrq must be 0 (or absent with CONFIG_MAGIC_SYSRQ=n)."""
        code = """\
try:
    with open('/proc/sys/kernel/sysrq') as f:
        val = f.read().strip()
    print(f'SYSRQ:{val}')
except FileNotFoundError:
    # CONFIG_MAGIC_SYSRQ=n: sysctl doesn't exist — SysRq fully compiled out
    print('SYSRQ_ABSENT')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "SYSRQ:0" in result.stdout or "SYSRQ_ABSENT" in result.stdout, (
            f"Expected kernel.sysrq=0 or absent. stdout: {result.stdout}"
        )


# =============================================================================
# /dev read-only hardening
# =============================================================================
class TestDevReadonlyHardening:
    """/dev is bind-mounted read-only to prevent device node creation.

    Blocks the CVE-2020-2023 (Kata Containers) attack chain:
    discover major:minor from /proc/partitions -> mknod -> debugfs -w /dev/vda.
    Even if CAP_MKNOD were somehow available, mknod(2) returns EROFS on a
    read-only filesystem. Does NOT break device I/O: VFS skips write-access
    counting for special files (char/block devices go through device driver).
    Does NOT affect /dev/shm (separate tmpfs mount, preserved by recursive bind).
    """

    # --- Normal: standard write attempts fail ---

    async def test_dev_vda_write_fails(self, dual_scheduler: Scheduler) -> None:
        """open('/dev/vda', 'wb') fails — node removed + /dev read-only."""
        code = """\
try:
    f = open('/dev/vda', 'wb')
    f.close()
    print('unexpected_success')
except (FileNotFoundError, PermissionError, OSError):
    print('blocked')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_dev_vda_readwrite_os_open(self, dual_scheduler: Scheduler) -> None:
        """os.open('/dev/vda', O_RDWR) fails (different syscall flags than Python open)."""
        code = """\
import os
try:
    fd = os.open('/dev/vda', os.O_RDWR)
    os.close(fd)
    print('unexpected_success')
except (FileNotFoundError, PermissionError, OSError):
    print('blocked')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_dev_mounted_readonly(self, dual_scheduler: Scheduler) -> None:
        """/dev is mounted read-only; mknod returns EROFS."""
        code = """\
import os, stat

# Verify /dev is mounted read-only.
# /proc/mounts may have multiple entries for /dev (original + bind mount).
# Check the LAST entry which reflects the bind-remount.
with open('/proc/mounts') as f:
    dev_ro = False
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/dev':
            opts = parts[3].split(',')
            dev_ro = 'ro' in opts
print(f'DEV_RO:{dev_ro}')

# Attempt mknod — should get EROFS (errno 30)
try:
    os.mknod('/dev/test_blk', 0o660 | stat.S_IFBLK, os.makedev(253, 0))
    print('MKNOD:success')
except OSError as e:
    print(f'MKNOD:errno_{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "DEV_RO:True" in result.stdout
        # EROFS (30) or EPERM (1) — both acceptable, EROFS preferred
        assert "MKNOD:errno_" in result.stdout
        assert "MKNOD:success" not in result.stdout

    # --- Normal: mmap attacks blocked (requires open() first) ---

    async def test_dev_vda_mmap_blocked(self, dual_scheduler: Scheduler) -> None:
        """open('/dev/vda', 'rb') + mmap fails — node removed."""
        code = """\
import mmap
try:
    f = open('/dev/vda', 'rb')
    m = mmap.mmap(f.fileno(), 4096)
    m.close()
    f.close()
    print('unexpected_success')
except (FileNotFoundError, PermissionError, OSError):
    print('blocked')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_dev_mem_mmap_blocked(self, dual_scheduler: Scheduler) -> None:
        """open('/dev/mem', 'rb') + mmap fails — node removed."""
        code = """\
import mmap
try:
    f = open('/dev/mem', 'rb')
    m = mmap.mmap(f.fileno(), 4096)
    m.close()
    f.close()
    print('unexpected_success')
except (FileNotFoundError, PermissionError, OSError):
    print('blocked')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_dev_vda_mmap_write_blocked(self, dual_scheduler: Scheduler) -> None:
        """open('/dev/vda', 'r+b') + writable mmap fails — node removed."""
        code = """\
import mmap
try:
    f = open('/dev/vda', 'r+b')
    m = mmap.mmap(f.fileno(), 4096, mmap.MAP_SHARED, mmap.PROT_WRITE)
    m.close()
    f.close()
    print('unexpected_success')
except (FileNotFoundError, PermissionError, OSError):
    print('blocked')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    # --- Edge: alternative access paths closed ---

    async def test_proc_self_root_traversal_blocked(self, dual_scheduler: Scheduler) -> None:
        """/proc/self/root/dev/vda fails (WithSecure Labs traversal vector)."""
        code = """\
try:
    f = open('/proc/self/root/dev/vda', 'rb')
    f.close()
    print('unexpected_success')
except (FileNotFoundError, PermissionError, OSError):
    print('blocked')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_mknod_tmp_nodev(self, dual_scheduler: Scheduler) -> None:
        """mknod in /tmp fails — nodev mount flag blocks block device creation."""
        code = """\
import os, stat
try:
    os.mknod('/tmp/fake_blk', 0o660 | stat.S_IFBLK, os.makedev(253, 0))
    print('CREATED')
except PermissionError:
    print('BLOCKED:EPERM')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:")

    async def test_open_by_handle_at_blocked(self, dual_scheduler: Scheduler) -> None:
        """open_by_handle_at syscall fails without CAP_DAC_READ_SEARCH (Shocker exploit)."""
        code = """\
import ctypes, ctypes.util, errno, struct

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

# open_by_handle_at requires CAP_DAC_READ_SEARCH (the "Shocker" exploit)
# We don't even need a valid handle — the capability check happens first
# Construct a minimal file_handle struct (8 bytes header + 0 bytes handle)
handle_bytes = struct.pack('II', 0, 0)  # handle_bytes=0, handle_type=0
handle_buf = ctypes.create_string_buffer(handle_bytes)

# SYS_open_by_handle_at = 304 on x86_64, 265 on aarch64
import platform
if platform.machine() == 'x86_64':
    SYS_open_by_handle_at = 304
else:
    SYS_open_by_handle_at = 265

ret = libc.syscall(SYS_open_by_handle_at, 3, handle_buf, 0)
err = ctypes.get_errno()
if ret == -1 and err == errno.EPERM:
    print('BLOCKED:EPERM')
elif ret == -1:
    print(f'BLOCKED:errno_{err}')
else:
    print('unexpected_success')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "BLOCKED:" in result.stdout
        assert "unexpected_success" not in result.stdout

    # --- Weird: creative bypass attempts ---

    async def test_mknod_raw_syscall_blocked(self, dual_scheduler: Scheduler) -> None:
        """ctypes SYS_mknod proves kernel-level deny (Kata CVE-2020-2023 chain)."""
        code = """\
import ctypes, ctypes.util, errno, os, stat, platform

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

# SYS_mknodat = 259 on x86_64, 33 on aarch64
if platform.machine() == 'x86_64':
    SYS_mknodat = 259
else:
    SYS_mknodat = 33

# AT_FDCWD = -100
AT_FDCWD = -100
path = b'/dev/test_mknod'
mode = stat.S_IFBLK | 0o660
dev = os.makedev(253, 0)

ret = libc.syscall(SYS_mknodat, AT_FDCWD, path, mode, dev)
err = ctypes.get_errno()
if ret == -1:
    if err == errno.EROFS:
        print('BLOCKED:EROFS')
    elif err == errno.EPERM:
        print('BLOCKED:EPERM')
    else:
        print(f'BLOCKED:errno_{err}')
else:
    print('unexpected_success')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "BLOCKED:" in result.stdout
        assert "unexpected_success" not in result.stdout

    async def test_debugfs_not_available(self, dual_scheduler: Scheduler) -> None:
        """debugfs binary doesn't exist in minimal Alpine rootfs."""
        code = """\
import shutil
print(f'DEBUGFS:{shutil.which("debugfs") is not None}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "DEBUGFS:False" in result.stdout

    async def test_sysfs_block_no_data_path(self, dual_scheduler: Scheduler) -> None:
        """/sys/dev/block/ exposes metadata only, no writable raw IO interface."""
        code = """\
import os

# Find any block device entries in sysfs
sysfs_block = '/sys/dev/block'
if not os.path.isdir(sysfs_block):
    print('NO_SYSFS_BLOCK')
else:
    dangerous_files = []
    for entry in os.listdir(sysfs_block):
        dev_path = os.path.join(sysfs_block, entry)
        if os.path.islink(dev_path):
            resolved = os.path.realpath(dev_path)
            for name in ['data', 'raw']:
                check = os.path.join(resolved, name)
                if os.path.exists(check):
                    dangerous_files.append(check)
    if dangerous_files:
        print(f'DANGEROUS:{",".join(dangerous_files)}')
    else:
        print('SAFE')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        out = result.stdout.strip()
        assert out in ("NO_SYSFS_BLOCK", "SAFE"), f"Dangerous sysfs paths found: {out}"

    async def test_uevent_helper_not_writable(self, dual_scheduler: Scheduler) -> None:
        """/sys/kernel/uevent_helper not writable by UID 1000."""
        code = """\
import os

path = '/sys/kernel/uevent_helper'
if not os.path.exists(path):
    print('NOT_EXISTS')
else:
    try:
        with open(path, 'w') as f:
            f.write('/tmp/pwned')
        print('WRITTEN')
    except (PermissionError, OSError) as e:
        print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "WRITTEN" not in result.stdout

    # --- Out of bounds: comprehensive/adversarial ---

    async def test_no_block_devices_outside_dev(self, dual_scheduler: Scheduler) -> None:
        """No block device nodes in /tmp or /home (catches smuggled nodes)."""
        code = """\
import os, stat

block_devices = []
for search_dir in ['/tmp', '/home']:
    if not os.path.isdir(search_dir):
        continue
    for dirpath, dirnames, filenames in os.walk(search_dir):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                s = os.lstat(path)
                if stat.S_ISBLK(s.st_mode):
                    block_devices.append(path)
            except OSError:
                pass

if block_devices:
    print(f'FOUND:{",".join(block_devices)}')
else:
    print('NONE')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.strip() == "NONE", f"Block devices outside /dev: {result.stdout}"

    async def test_effective_caps_no_mknod(self, dual_scheduler: Scheduler) -> None:
        """CAP_MKNOD (bit 27) and CAP_DAC_READ_SEARCH (bit 2) are both clear."""
        code = """\
with open('/proc/self/status') as f:
    for line in f:
        if line.startswith('CapEff:'):
            cap_eff = int(line.split(':')[1].strip(), 16)
            cap_mknod = bool(cap_eff & (1 << 27))
            cap_dac_read_search = bool(cap_eff & (1 << 2))
            print(f'CAP_MKNOD:{cap_mknod}')
            print(f'CAP_DAC_READ_SEARCH:{cap_dac_read_search}')
            break
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "CAP_MKNOD:False" in result.stdout
        assert "CAP_DAC_READ_SEARCH:False" in result.stdout

    async def test_sysrq_trigger_not_writable(self, dual_scheduler: Scheduler) -> None:
        """/proc/sysrq-trigger bind-mounted read-only (EROFS).

        Writing 'c' = kernel crash, 'b' = reboot (DoS). kernel.sysrq=0 does NOT
        protect this file — write_sysrq_trigger() bypasses the sysrq_enabled bitmask.
        See: CVE-2025-31133, CVE-2025-52565, CVE-2025-52881.
        """
        code = """\
try:
    with open('/proc/sysrq-trigger', 'w') as f:
        f.write('h')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected sysrq-trigger write to be blocked.\nstdout: {result.stdout}"
        )


# =============================================================================
# Root filesystem read-only hardening
# =============================================================================
class TestRootFsReadonly:
    """Root filesystem is mounted read-only with nosuid and nodev flags.

    Matches AWS Lambda rootfs flags (ro,nosuid,nodev,noatime). The rootfs
    never enters rw state — tiny-init mounts it with MS_RDONLY from boot.
    Prevents writing to /etc/passwd, /sbin, planting SUID binaries, or
    remounting rw to DoS subsequent code.
    """

    ROOT_MOUNT_LINE = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/':
            print(line.strip())
            break
"""

    async def test_root_has_ro_flag(self, dual_scheduler: Scheduler) -> None:
        """Root filesystem has ro mount flag."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/':
            opts = parts[3].split(',')
            print(f'RO:{"ro" in opts}')
            break
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "RO:True" in result.stdout, f"Expected / to be mounted read-only.\nstdout: {result.stdout}"

    async def test_root_has_nosuid_flag(self, dual_scheduler: Scheduler) -> None:
        """Root filesystem has nosuid mount flag."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/':
            opts = parts[3].split(',')
            print(f'NOSUID:{"nosuid" in opts}')
            break
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "NOSUID:True" in result.stdout, f"Expected / to have nosuid.\nstdout: {result.stdout}"

    async def test_root_has_nodev_flag(self, dual_scheduler: Scheduler) -> None:
        """Root filesystem has nodev mount flag."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/':
            opts = parts[3].split(',')
            print(f'NODEV:{"nodev" in opts}')
            break
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "NODEV:True" in result.stdout, f"Expected / to have nodev.\nstdout: {result.stdout}"

    async def test_write_to_var_fails(self, dual_scheduler: Scheduler) -> None:
        """Writing to /var fails with EROFS on read-only rootfs."""
        code = """\
try:
    with open('/var/test_file', 'w') as f:
        f.write('test')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), f"Expected write to /var to fail.\nstdout: {result.stdout}"

    async def test_create_file_in_sbin_fails(self, dual_scheduler: Scheduler) -> None:
        """Writing to /sbin fails with EROFS on read-only rootfs."""
        code = """\
try:
    with open('/sbin/test_file', 'w') as f:
        f.write('test')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), f"Expected write to /sbin to fail.\nstdout: {result.stdout}"

    async def test_remount_rw_root_fails(self, dual_scheduler: Scheduler) -> None:
        """mount -o remount,rw / requires CAP_SYS_ADMIN — fails as UID 1000."""
        code = """\
import subprocess
r = subprocess.run(
    ["mount", "-o", "remount,rw", "/"],
    capture_output=True, text=True, timeout=5,
)
print(f"EXIT:{r.returncode}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXIT:0" not in result.stdout, f"Expected remount rw / to fail.\nstdout: {result.stdout}"

    async def test_home_user_still_writable(self, dual_scheduler: Scheduler) -> None:
        """/home/user is writable (tmpfs overlay on read-only rootfs)."""
        code = """\
import os
path = '/home/user/test_rw.txt'
with open(path, 'w') as f:
    f.write('writable')
with open(path) as f:
    data = f.read()
os.unlink(path)
print(f'OK:{data}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "OK:writable" in result.stdout

    async def test_tmp_still_writable(self, dual_scheduler: Scheduler) -> None:
        """/tmp is writable (separate tmpfs mount)."""
        code = """\
import os
path = '/tmp/test_rw.txt'
with open(path, 'w') as f:
    f.write('writable')
with open(path) as f:
    data = f.read()
os.unlink(path)
print(f'OK:{data}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "OK:writable" in result.stdout

    async def test_dev_shm_still_writable(self, dual_scheduler: Scheduler) -> None:
        """/dev/shm is writable (separate tmpfs mount)."""
        code = """\
import os
path = '/dev/shm/test_rw.txt'
with open(path, 'w') as f:
    f.write('writable')
with open(path) as f:
    data = f.read()
os.unlink(path)
print(f'OK:{data}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "OK:writable" in result.stdout


# =============================================================================
# /home/user tmpfs mount hardening
# =============================================================================
class TestHomeUserTmpfs:
    """/home/user is mounted as tmpfs with nosuid and nodev flags.

    Zero-copy scratch space: /home/user on rootfs is empty (cloudpickle
    installed to /usr/lib/python3/site-packages), so the tmpfs mount
    overlays an empty directory. This provides writable user space without
    compromising the read-only rootfs.
    """

    HOME_MOUNT_LINE = """\
with open('/proc/mounts') as f:
    for line in f:
        if ' /home/user ' in line:
            print(line.strip())
            break
"""

    async def test_is_tmpfs(self, dual_scheduler: Scheduler) -> None:
        """/home/user filesystem type is tmpfs."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/home/user':
            print(f'FSTYPE:{parts[2]}')
            break
    else:
        print('MOUNT_NOT_FOUND')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "FSTYPE:tmpfs" in result.stdout, f"Expected /home/user to be tmpfs.\nstdout: {result.stdout}"

    async def test_nosuid_flag(self, dual_scheduler: Scheduler) -> None:
        """/home/user has nosuid mount flag."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/home/user':
            opts = parts[3].split(',')
            print(f'NOSUID:{"nosuid" in opts}')
            break
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "NOSUID:True" in result.stdout

    async def test_nodev_flag(self, dual_scheduler: Scheduler) -> None:
        """/home/user has nodev mount flag."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/home/user':
            opts = parts[3].split(',')
            print(f'NODEV:{"nodev" in opts}')
            break
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "NODEV:True" in result.stdout

    async def test_no_noexec_flag(self, dual_scheduler: Scheduler) -> None:
        """/home/user does NOT have noexec (uv/bun install executables)."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/home/user':
            opts = parts[3].split(',')
            print(f'NOEXEC_PRESENT:{"noexec" in opts}')
            break
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "NOEXEC_PRESENT:False" in result.stdout

    async def test_cloudpickle_importable(self, dual_scheduler: Scheduler) -> None:
        """cloudpickle is importable (loaded from /usr/lib/python3/site-packages)."""
        code = """\
import cloudpickle
print(f'VERSION:{cloudpickle.__version__}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "VERSION:" in result.stdout

    async def test_owned_by_uid_1000(self, dual_scheduler: Scheduler) -> None:
        """/home/user is owned by UID 1000 (sandbox user)."""
        code = """\
import os
s = os.stat('/home/user')
print(f'UID:{s.st_uid}')
print(f'GID:{s.st_gid}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "UID:1000" in result.stdout
        assert "GID:1000" in result.stdout


# =============================================================================
# /bin and /sbin read-only hardening
# =============================================================================
class TestSystemBinaryHardening:
    """/bin and /sbin are bind-mounted read-only with nosuid.

    Blocks path hijack attacks where user code (UID 1000) replaces /bin/sh
    (a busybox symlink) with an arbitrary script, hijacking subsequent shebang
    (#!/bin/sh) execution within the same session. Primary protection is the
    RO rootfs from tiny-init; the bind-mount adds nosuid as defense-in-depth.

    Alpine 3.23+ uses usrmerge (/bin -> /usr/bin, /sbin -> /usr/sbin); mount(2)
    resolves symlinks, so the bind-mount targets the resolved /usr/bin directory.

    See: Microsoft Copilot Enterprise path hijack (Eye Security, April 2025).
    """

    # --- Normal: verify read-only mount flags ---

    async def test_bin_mounted_readonly(self, dual_scheduler: Scheduler) -> None:
        """/bin has 'ro' flag in /proc/mounts."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/bin':
            opts = parts[3].split(',')
            print(f'RO:{"ro" in opts}')
            break
    else:
        # usrmerge: /bin -> /usr/bin, check /usr instead
        for line in open('/proc/mounts'):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == '/usr':
                opts = parts[3].split(',')
                print(f'RO:{"ro" in opts}')
                break
        else:
            print('NOT_FOUND')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "RO:True" in result.stdout, f"Expected /bin to be read-only.\nstdout: {result.stdout}"

    async def test_sbin_mounted_readonly(self, dual_scheduler: Scheduler) -> None:
        """/sbin has 'ro' flag in /proc/mounts."""
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == '/sbin':
            opts = parts[3].split(',')
            print(f'RO:{"ro" in opts}')
            break
    else:
        # usrmerge: /sbin -> /usr/sbin, check /usr instead
        for line in open('/proc/mounts'):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == '/usr':
                opts = parts[3].split(',')
                print(f'RO:{"ro" in opts}')
                break
        else:
            print('NOT_FOUND')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "RO:True" in result.stdout, f"Expected /sbin to be read-only.\nstdout: {result.stdout}"

    # --- Normal: write attempts blocked ---

    async def test_cannot_replace_bin_sh(self, dual_scheduler: Scheduler) -> None:
        """os.unlink('/bin/sh') is blocked with EROFS."""
        code = """\
import os, errno
try:
    os.unlink('/bin/sh')
    print('REMOVED')
except OSError as e:
    if e.errno == errno.EROFS:
        print('BLOCKED:erofs')
    elif e.errno == errno.EACCES:
        print('BLOCKED:eacces')
    else:
        print(f'BLOCKED:errno_{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), f"Expected unlink /bin/sh to be blocked.\nstdout: {result.stdout}"

    async def test_cannot_overwrite_busybox(self, dual_scheduler: Scheduler) -> None:
        """open('/bin/busybox', 'wb') is blocked."""
        code = """\
import errno
try:
    f = open('/bin/busybox', 'wb')
    f.close()
    print('OPENED')
except OSError as e:
    if e.errno == errno.EROFS:
        print('BLOCKED:erofs')
    elif e.errno == errno.EACCES:
        print('BLOCKED:eacces')
    else:
        print(f'BLOCKED:errno_{e.errno}')
except FileNotFoundError:
    # usrmerge: busybox may be at /usr/bin/busybox
    print('BLOCKED:not_found')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected busybox overwrite to be blocked.\nstdout: {result.stdout}"
        )

    async def test_cannot_create_file_in_bin(self, dual_scheduler: Scheduler) -> None:
        """File creation in /bin is blocked."""
        code = """\
import errno
try:
    with open('/bin/evil', 'w') as f:
        f.write('#!/bin/sh\\necho pwned')
    print('CREATED')
except OSError as e:
    if e.errno == errno.EROFS:
        print('BLOCKED:erofs')
    elif e.errno == errno.EACCES:
        print('BLOCKED:eacces')
    else:
        print(f'BLOCKED:errno_{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected file creation in /bin to be blocked.\nstdout: {result.stdout}"
        )

    async def test_cannot_create_file_in_sbin(self, dual_scheduler: Scheduler) -> None:
        """File creation in /sbin is blocked."""
        code = """\
import errno
try:
    with open('/sbin/evil', 'w') as f:
        f.write('#!/bin/sh\\necho pwned')
    print('CREATED')
except OSError as e:
    if e.errno == errno.EROFS:
        print('BLOCKED:erofs')
    elif e.errno == errno.EACCES:
        print('BLOCKED:eacces')
    else:
        print(f'BLOCKED:errno_{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected file creation in /sbin to be blocked.\nstdout: {result.stdout}"
        )

    async def test_cannot_create_symlink_in_bin(self, dual_scheduler: Scheduler) -> None:
        """Symlink creation in /bin is blocked."""
        code = """\
import os, errno
try:
    os.symlink('/tmp/evil', '/bin/evil_link')
    print('CREATED')
except OSError as e:
    if e.errno == errno.EROFS:
        print('BLOCKED:erofs')
    elif e.errno == errno.EACCES:
        print('BLOCKED:eacces')
    else:
        print(f'BLOCKED:errno_{e.errno}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert result.stdout.startswith("BLOCKED:"), (
            f"Expected symlink creation in /bin to be blocked.\nstdout: {result.stdout}"
        )

    # --- Edge: the specific attack vector ---

    async def test_bin_sh_replacement_attack_blocked(self, dual_scheduler: Scheduler) -> None:
        """rm /bin/sh via RAW language fails (the specific attack vector)."""
        result = await dual_scheduler.run(
            code="rm /bin/sh 2>&1; echo EXIT:$?",
            language=Language.RAW,
        )
        assert result.exit_code in {0, 1}
        assert "EXIT:0" not in result.stdout, f"Expected rm /bin/sh to fail.\nstdout: {result.stdout}"

    # --- Normal: system binaries still functional ---

    async def test_bin_sh_still_executable(self, dual_scheduler: Scheduler) -> None:
        """/bin/sh -c 'echo OK' still works (read-only does not imply noexec)."""
        code = """\
import subprocess
r = subprocess.run(
    ["/bin/sh", "-c", "echo OK"],
    capture_output=True, text=True, timeout=5,
)
print(f'OUTPUT:{r.stdout.strip()}')
print(f'EXIT:{r.returncode}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "OUTPUT:OK" in result.stdout
        assert "EXIT:0" in result.stdout

    async def test_busybox_applets_still_work(self, dual_scheduler: Scheduler) -> None:
        """Busybox applets (ls, cat) still work from /bin."""
        code = """\
import subprocess

# Test ls
r = subprocess.run(["ls", "/tmp"], capture_output=True, text=True, timeout=5)
print(f'LS_EXIT:{r.returncode}')

# Test cat
r = subprocess.run(
    ["cat", "/etc/hostname"],
    capture_output=True, text=True, timeout=5,
)
print(f'CAT_EXIT:{r.returncode}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "LS_EXIT:0" in result.stdout
        assert "CAT_EXIT:0" in result.stdout

    @pytest.mark.parametrize(
        "binary, version_marker",
        [
            pytest.param("grep", "GNU grep", id="grep"),
            pytest.param("find", "GNU findutils", id="find"),
            pytest.param("sed", "GNU sed", id="sed"),
            pytest.param("gawk", "GNU Awk", id="gawk"),
            pytest.param("diff", "diffutils", id="diff"),
        ],
    )
    async def test_gnu_tool_installed(self, dual_scheduler: Scheduler, binary: str, version_marker: str) -> None:
        """GNU coreutils variant is installed, not busybox."""
        code = f"""\
import subprocess
r = subprocess.run(["{binary}", "--version"], capture_output=True, text=True, timeout=5)
print(f'EXIT:{{r.returncode}}')
print(f'GNU:{{{version_marker!r} in r.stdout}}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXIT:0" in result.stdout
        assert "GNU:True" in result.stdout

    # --- Weird: mount bypass attempts ---

    async def test_remount_bin_rw_blocked(self, dual_scheduler: Scheduler) -> None:
        """mount -o remount,rw /bin requires CAP_SYS_ADMIN."""
        code = """\
import subprocess
r = subprocess.run(
    ["mount", "-o", "remount,rw", "/bin"],
    capture_output=True, text=True, timeout=5,
)
print(f"EXIT:{r.returncode}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXIT:0" not in result.stdout, (
            f"Expected remount to fail without CAP_SYS_ADMIN.\nstdout: {result.stdout}"
        )

    async def test_remount_sbin_rw_blocked(self, dual_scheduler: Scheduler) -> None:
        """mount -o remount,rw /sbin requires CAP_SYS_ADMIN."""
        code = """\
import subprocess
r = subprocess.run(
    ["mount", "-o", "remount,rw", "/sbin"],
    capture_output=True, text=True, timeout=5,
)
print(f"EXIT:{r.returncode}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXIT:0" not in result.stdout, (
            f"Expected remount to fail without CAP_SYS_ADMIN.\nstdout: {result.stdout}"
        )

    async def test_bind_mount_shadow_bin_blocked(self, dual_scheduler: Scheduler) -> None:
        """Bind tmpfs over /bin requires CAP_SYS_ADMIN."""
        code = """\
import subprocess
r = subprocess.run(
    ["mount", "-t", "tmpfs", "none", "/bin"],
    capture_output=True, text=True, timeout=5,
)
print(f"EXIT:{r.returncode}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXIT:0" not in result.stdout, f"Expected tmpfs shadow mount to fail.\nstdout: {result.stdout}"


# =============================================================================
# Thread/fork bomb mitigation (RLIMIT_NPROC + kernel.threads-max)
# =============================================================================
class TestThreadBombMitigation:
    """RLIMIT_NPROC and kernel.threads-max cap thread/fork bombs.

    Defense-in-depth: RLIMIT_NPROC=1024 (per-UID, set via prlimit64 from the
    guest-agent after REPL spawn) is the primary defense. kernel.threads-max=1200
    (system-wide, set in tiny-init) is the secondary defense.
    """

    # --- Normal: verify limits are applied ---

    async def test_rlimit_nproc_is_set(self, dual_scheduler: Scheduler) -> None:
        """RLIMIT_NPROC soft and hard limits are both 1024."""
        code = """\
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
print(f'NPROC_SOFT:{soft}')
print(f'NPROC_HARD:{hard}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "NPROC_SOFT:1024" in result.stdout
        assert "NPROC_HARD:1024" in result.stdout

    async def test_threadpool_executor_still_works(self, dual_scheduler: Scheduler) -> None:
        """ThreadPoolExecutor(max_workers=4) works within the limit."""
        code = """\
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(lambda x: x**2, range(100)))
print(f'RESULT:{len(results)}:{results[-1]}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "RESULT:100:9801" in result.stdout

    # --- Edge: thread bomb is capped ---

    async def test_thread_bomb_capped(self, dual_scheduler: Scheduler) -> None:
        """Thread bomb hits RLIMIT_NPROC well below the old ~1,659 ceiling."""
        code = """\
import threading
import time

threads = []
try:
    while True:
        t = threading.Thread(target=lambda: time.sleep(60), daemon=True)
        t.start()
        threads.append(t)
except RuntimeError:
    pass

print(f'THREAD_COUNT:{len(threads)}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=30)
        assert result.exit_code == 0
        count_line = [line for line in result.stdout.splitlines() if line.startswith("THREAD_COUNT:")]
        assert count_line, f"Expected THREAD_COUNT in output.\nstdout: {result.stdout}"
        count = int(count_line[0].split(":")[1])
        assert count <= 1100, f"Thread bomb created {count} threads, expected <= 1100 with RLIMIT_NPROC=1024"

    # --- Out of bounds: cannot escalate limits ---

    async def test_cannot_raise_nproc_limit(self, dual_scheduler: Scheduler) -> None:
        """User code cannot raise RLIMIT_NPROC above the hard limit."""
        code = """\
import resource
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (9999, 9999))
    print('RAISED:True')
except (ValueError, PermissionError, OSError) as e:
    print(f'RAISED:False:{type(e).__name__}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "RAISED:False" in result.stdout

    async def test_fork_bomb_hits_nproc(self, dual_scheduler: Scheduler) -> None:
        """Fork bomb hits RLIMIT_NPROC, count is <= 1024."""
        code = """\
import os

count = 0
try:
    while True:
        pid = os.fork()
        if pid == 0:
            # Child: exit immediately (becomes zombie until reaped)
            os._exit(0)
        count += 1
except OSError:
    pass

# Reap all zombies
while True:
    try:
        os.waitpid(-1, 0)
    except ChildProcessError:
        break

print(f'FORK_COUNT:{count}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=30)
        assert result.exit_code == 0
        count_line = [line for line in result.stdout.splitlines() if line.startswith("FORK_COUNT:")]
        assert count_line, f"Expected FORK_COUNT in output.\nstdout: {result.stdout}"
        count = int(count_line[0].split(":")[1])
        assert count <= 1024, f"Fork bomb created {count} children, expected <= 1024 with RLIMIT_NPROC=1024"

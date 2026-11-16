"""Tests for guest VM device-layer security hardening.

Verifies that dangerous device nodes do not exist, block device nodes
(/dev/vda, /dev/zram0) are removed after use, /dev is bind-remounted
read-only (preventing mknod with EROFS), /dev/shm is mounted with
restrictive flags, /tmp has nosuid/nodev flags with an explicit inode
cap (nr_inodes=16384) and a per-UID tmpfs quota (usrquota, the size of
the mount), /proc/sys is read-only, and /bin + /sbin are bind-mounted
read-only with nosuid and nodev (blocking path hijack attacks).
These are defense-in-depth measures that apply regardless of UID — even
root (the guest agent) cannot open /dev/mem or read the root disk raw.

Complements test_nonroot_repl.py (privilege escalation from UID 1000) with
kernel/device-layer hardening that blocks access at the filesystem level.
"""

import re

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

_SIZE_AND_MEMTOTAL = """\
import os
with open('/proc/meminfo') as f:
    mem_total_kb = next(int(line.split()[1]) for line in f if line.startswith('MemTotal:'))
s = os.statvfs('{}')
print(mem_total_kb * 1024, os.sysconf('SC_PAGESIZE'), s.f_blocks * s.f_frsize)
"""
"""Guest code: MemTotal in bytes, the page size, and the size of a tmpfs mount."""


_MOUNT_OPTIONS = """\
for line in open('/proc/mounts'):
    _device, mountpoint, fstype, options = line.split()[:4]
    if mountpoint == '{}':
        print(fstype, ','.join(o for o in options.split(',') if not o.startswith(('size=', 'usrquota_block_hardlimit='))))
"""
"""Guest code: the file system type and mount options of each mount at a path, but for the
values that follow the VM memory (size=, usrquota_block_hardlimit=: the size tests check them)."""


def _assert_page_aligned_share(stdout: str, *, share: float) -> None:
    """The tmpfs size is `share` of MemTotal, rounded down to a whole page."""
    mem_total, page, size = (int(field) for field in stdout.split())
    assert size % page == 0, (size, page)
    assert mem_total * share - page < size <= mem_total * share, (size, mem_total, page)


# =============================================================================
# Dangerous device nodes should not exist
# =============================================================================
class TestDangerousDeviceNodes:
    """Dangerous device nodes do not exist: the kernel has no DEVMEM and no DEVPORT
    (images/kernel/exec-sandbox.config), and /dev/kmem left Linux in 5.13."""

    async def test_dangerous_nodes_not_exist(self, dual_scheduler: Scheduler) -> None:
        """/dev/mem (raw physical memory), /dev/kmem (kernel virtual memory) and /dev/port (raw
        I/O port access) do not exist."""
        code = "import os; print([p for p in ('/dev/mem', '/dev/kmem', '/dev/port') if os.path.lexists(p)])"
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "[]\n"), result.stderr


# =============================================================================
# Entropy device hardening (least-privilege)
# =============================================================================
class TestEntropyDeviceHardening:
    """Entropy devices are read-only for unprivileged users (UID 1000).

    /dev/urandom and /dev/random are character devices auto-created by devtmpfs
    with mode 0666. Even though /dev is bind-remounted read-only by guest-agent,
    Linux VFS allows writes to char devices on RO filesystems (sb_permission()
    only blocks regular files, dirs, and symlinks). Mode 0644 restricts write
    access to root (guest-agent needs it for RNDRESEEDCRNG ioctl), while
    UID 1000 only needs read access (or getrandom() syscall).
    """

    async def test_write_blocked(self, dual_scheduler: Scheduler) -> None:
        """UID 1000 cannot open /dev/urandom or /dev/random for write (entropy injection): EACCES,
        as the mode is 0644 and the owner root."""
        code = """\
import errno, os
for path in ('/dev/urandom', '/dev/random'):
    try:
        os.close(os.open(path, os.O_WRONLY))
        print(f'{path}:OPENED')
    except OSError as e:
        print(f'{path}:{errno.errorcode[e.errno]}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "/dev/urandom:EACCES\n/dev/random:EACCES\n"), result.stderr

    async def test_urandom_read_still_works(self, dual_scheduler: Scheduler) -> None:
        """Reading /dev/urandom still works for UID 1000."""
        code = """\
data = open('/dev/urandom', 'rb').read(32)
print(f'READ:{len(data)}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "READ:32\n"

    async def test_urandom_permissions(self, dual_scheduler: Scheduler) -> None:
        """/dev/urandom has mode 0644 (owner rw, others read-only)."""
        code = """\
import os, stat
st = os.stat('/dev/urandom')
mode = stat.S_IMODE(st.st_mode)
print(f'MODE:{oct(mode)}')
print(f'UID:{st.st_uid}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "MODE:0o644\nUID:0\n"


# =============================================================================
# /dev/shm mount hardening
# =============================================================================
class TestDevShmHardening:
    """/dev/shm is mounted with nosuid, nodev, noexec flags, and usrquota."""

    async def test_mount_options(self, dual_scheduler: Scheduler) -> None:
        """/dev/shm is a tmpfs mounted nosuid, nodev and noexec, with a per-UID quota (usrquota)."""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), _MOUNT_OPTIONS.format("/dev/shm"))
        assert result.exit_code == 0, result.stderr
        # Twice: the recursive read-only bind remount of /dev holds /dev/shm too, with its options
        assert result.stdout == "tmpfs rw,nosuid,nodev,noexec,relatime,inode64,noswap,usrquota\n" * 2

    async def test_shm_size_is_forty_pct_memtotal(self, dual_scheduler: Scheduler) -> None:
        """/dev/shm size is 40% of MemTotal (memory-proportional), rounded down to a page."""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), _SIZE_AND_MEMTOTAL.format("/dev/shm"))
        assert result.exit_code == 0, result.stderr
        _assert_page_aligned_share(result.stdout, share=0.4)

    async def test_exec_blocked_on_dev_shm(self, dual_scheduler: Scheduler) -> None:
        """Script execution is blocked on /dev/shm (noexec enforcement): execve() fails with EACCES."""
        code = """\
import errno, os, stat, subprocess
path = '/dev/shm/test_exec.sh'
with open(path, 'w') as f:
    f.write('#!/bin/sh\\necho pwned\\n')
os.chmod(path, stat.S_IRWXU)
try:
    r = subprocess.run([path], capture_output=True, timeout=5)
    print(f"exit={r.returncode}")
except OSError as e:
    print(errno.errorcode[e.errno])
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "EACCES\n"), result.stderr

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.strip() == "[1, 4, 9, 16, 25]"


# =============================================================================
# /tmp tmpfs hardening
# =============================================================================
class TestTmpHardening:
    """/tmp is mounted with nosuid, nodev, nr_inodes=16384, mode=1777, usrquota."""

    async def test_mount_options(self, dual_scheduler: Scheduler) -> None:
        """/tmp is a tmpfs mounted nosuid (CIS 1.1.4) and nodev (CIS 1.1.3) but not noexec (uv
        wheel installs and PyInstaller run files there), with 16,384 inodes and a per-UID quota."""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), _MOUNT_OPTIONS.format("/tmp"))
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "tmpfs rw,nosuid,nodev,relatime,nr_inodes=16384,inode64,noswap,usrquota\n"

    async def test_sticky_bit(self, dual_scheduler: Scheduler) -> None:
        """/tmp has sticky bit (mode 1777) preventing cross-user file deletion."""
        code = "import os, stat; s = os.stat('/tmp'); print(f'{stat.S_IMODE(s.st_mode):o}')"
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "1777"

    async def test_tmpfs_size_is_half_memtotal(self, dual_scheduler: Scheduler) -> None:
        """/tmp size is 50% of MemTotal (memory-proportional, not hardcoded), rounded down to a page."""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), _SIZE_AND_MEMTOTAL.format("/tmp"))
        assert result.exit_code == 0, result.stderr
        _assert_page_aligned_share(result.stdout, share=0.5)

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "EXEC_OK\n"

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "CREATED:True\nDELETED:True\n"

    # --- Edge cases: boundary conditions ---

    async def test_inode_exhaustion_produces_oserror(self, dual_scheduler: Scheduler) -> None:
        """Files stop at the inode limit with OSError ENOSPC, not a crash: 16,383 files, as /tmp
        itself takes one of the 16,384 inodes."""
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "FILE_LIMIT:16383\nERRNO:28\n"  # ENOSPC

    async def test_files_still_writable_below_inode_limit(self, dual_scheduler: Scheduler) -> None:
        """1000 files, well under the inode limit, take data and list (the only files of /tmp)."""
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "CONTENT:content_999\nCOUNT:1000\n"

    async def test_subdirectories_consume_inodes(self, dual_scheduler: Scheduler) -> None:
        """Directories take inodes too: mkdir stops at the same 16,383 as files."""
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "DIR_LIMIT:16383\n"

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "CYCLE_OK\n"

    async def test_symlinks_consume_inodes(self, dual_scheduler: Scheduler) -> None:
        """Symlinks take inodes from the same pool: 16,382 (/tmp and the target file take two)."""
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "SYMLINK_LIMIT:16382\n"

    async def test_hardlinks_consume_inodes(self, dual_scheduler: Scheduler) -> None:
        """Hard links on tmpfs consume inodes: each link takes one of nr_inodes.

        Unlike on-disk filesystems, where a link takes no inode, tmpfs reserves one
        per link (shmem_link), so hard links count the same as regular files.
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        # 1 source file + 100 hard links = 101 inodes on tmpfs
        # (tmpfs calls shmem_reserve_inode() per link, unlike on-disk filesystems)
        assert result.stdout == "INODES_USED:101\n"

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "INODES_USED:200\n"  # 100 empty + 100 with data

    # --- Out of bounds: security enforcement ---

    async def test_inode_exhaustion_doesnt_crash_vm(self, dual_scheduler: Scheduler) -> None:
        """Exhausting /tmp inodes doesn't crash the VM — other paths still work."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
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
            r1 = await session.exec(exhaust)
            assert (r1.exit_code, r1.stdout) == (0, "EXHAUSTED:16383\n"), r1.stderr

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
            )
            assert (r2.exit_code, r2.stdout) == (0, "CORRECTLY_BLOCKED\n"), r2.stderr

            # Step 3: Verify /home/user (the ext4 home disk) still works
            r3 = await session.exec(
                """\
with open('/home/user/still_works.txt', 'w') as f:
    f.write('alive')
with open('/home/user/still_works.txt') as f:
    print(f'HOME:{f.read()}')
""",
            )
            assert (r3.exit_code, r3.stdout) == (0, "HOME:alive\n"), r3.stderr

            # Step 4: Verify Python execution still works (the REPL runs from /home/user, not /tmp)
            r4 = await session.exec("print(f'MATH:{2 + 2}')")
            assert (r4.exit_code, r4.stdout) == (0, "MATH:4\n"), r4.stderr


# =============================================================================
# Block device hardening
# =============================================================================
class TestBlockDeviceHardening:
    """Setup-only block device nodes are absent before untrusted execution."""

    # --- Normal cases: device nodes removed and filesystem still works ---

    async def test_setup_and_alias_nodes_not_exist(self, dual_scheduler: Scheduler) -> None:
        """/dev/zram0 (the guest agent removes it after activating swap), /dev/block (devtmpfs
        makes no such symlinks: udev does, and the guest has no udev) and /dev/root (sometimes
        auto-created by the kernel) do not exist."""
        code = "import os; print([p for p in ('/dev/zram0', '/dev/block', '/dev/root') if os.path.lexists(p)])"
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "[]\n"), result.stderr

    # --- Edge cases: alternative paths to block device are closed ---

    async def test_proc_partitions_readable_but_no_data_access(self, dual_scheduler: Scheduler) -> None:
        """/proc/partitions shows vda metadata but the device node /dev/vda is gone."""
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
except OSError:
    print('DEV_VDA_OPEN:blocked')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "PROC_HAS_VDA:True\nDEV_VDA_OPEN:not_found\n"

    # --- Weird cases: attempts to recreate block device access ---

    async def test_mknod_blocked(self, dual_scheduler: Scheduler) -> None:
        """mknod of a block device node fails with EPERM: UID 1000 lacks CAP_MKNOD (on any
        writable mount; nodev on /tmp and /home/user is a second layer)."""
        code = """\
import errno, os, stat
try:
    # Attempt to create a block device node (major 253 = virtblk)
    os.mknod('/home/user/vda', 0o660 | stat.S_IFBLK, os.makedev(253, 0))
    print('CREATED')
except OSError as e:
    print(f'BLOCKED:{errno.errorcode[e.errno]}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "BLOCKED:EPERM\n"), result.stderr

    # --- Out of bounds cases: no block device nodes remain anywhere ---

    async def test_no_block_devices(self, dual_scheduler: Scheduler) -> None:
        """No block device nodes remain under /dev, and none are smuggled into /tmp or /home."""
        code = """\
import os, stat

for top in ('/dev', '/tmp', '/home'):
    for dirpath, dirnames, filenames in os.walk(top):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                if stat.S_ISBLK(os.lstat(path).st_mode):
                    print(path)
            except OSError:
                pass
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, ""), result.stderr


# =============================================================================
# Loop device + filesystem creation blocked
# =============================================================================
class TestLoopDeviceBlocked:
    """Loop device infrastructure is absent — no module, no device nodes, no tools.

    Loop devices are a proven attack vector: CVE-2025-8067 (udisks loop device
    LPE, CVSS 8.5), CVE-2025-6019 (SUID shell via loop-mounted XFS without
    nosuid), and multiple ext4/btrfs/xfs kernel bugs triggered by mounting
    crafted filesystem images (CVE-2025-38220, CVE-2025-38222, CVE-2024-39472).

    4 independent layers block loop device + mkfs operations:
    1. CONFIG_MODULES=n, so no loadable loop driver exists
    2. No /dev/loop* nodes (the kernel has no loop device support)
    3. /dev is read-only (mknod returns EROFS)
    4. No CAP_MKNOD (UID 1000, verified in TestDevReadonlyHardening)
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "NONE", f"Loop device nodes found: {result.stdout.strip()}"

    async def test_loop_module_not_loaded(self, dual_scheduler: Scheduler) -> None:
        """No loop module can be loaded: CONFIG_MODULES=n (layer 1), so /proc/modules does not exist."""
        result = await run_once(
            dual_scheduler, VmSpec(language=Language.PYTHON), "import os; print(os.path.exists('/proc/modules'))"
        )
        assert (result.exit_code, result.stdout) == (0, "False\n"), result.stderr

    # --- Edge: tools that need loop devices fail gracefully ---

    async def test_mkfs_ext4_not_exploitable(self, dual_scheduler: Scheduler) -> None:
        """mkfs.ext4 exists but cannot format a block device.

        e2fsprogs is in the image for tiny-init (root) to format the home disk and the
        snapshot drive. The sandbox user (UID 1000) cannot exploit it: the kernel has no loop
        devices (CONFIG_BLK_DEV_LOOP=n) and tiny-init removes the /dev/vd* nodes after mount.
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "HAS_MKFS:True\nFORMAT_RC:1\n"

    # --- Out of bounds: full attack chains blocked ---

    async def test_loop_module_load_blocked(self, dual_scheduler: Scheduler) -> None:
        """modprobe loop fails because the kernel has no module support."""
        result = await run_once(dual_scheduler, VmSpec(language=Language.RAW), "modprobe loop 2>&1; echo EXIT:$?")
        assert result.exit_code == 0
        assert (
            result.stdout == "modprobe: can't change directory to '/lib/modules': No such file or directory\nEXIT:1\n"
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        # dd makes the file (a regular file in /tmp); losetup and mount fail
        assert result.stdout == "DD_EXIT:0\nLOSETUP_EXIT:1\nMOUNT_EXIT:255\n"


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

    Required VM tuning is applied by guest-agent before /proc/sys is sealed
    read-only, so untrusted code needs no writes there at runtime.

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "RO:True\n"

    async def test_write_dangerous_sysctls_blocked(self, dual_scheduler: Scheduler) -> None:
        """No write to a dangerous sysctl gets through. One VM tries each:
        - kernel.core_pattern (pipe-to-binary root execution) and kernel.modprobe (module
          autoloader path): absent (ENOENT), as CONFIG_COREDUMP=n and CONFIG_MODULES=n;
        - randomize_va_space (ASLR off), ip_forward (network pivoting), hostname, overcommit_memory
          (OOM policy), panic, panic_on_oom: EROFS (the truncating open needs a writable mount).
        """
        code = """\
for path, value in [
    ('kernel/core_pattern', '|/tmp/pwned'),
    ('kernel/modprobe', '/tmp/pwned'),
    ('kernel/randomize_va_space', '0'),
    ('net/ipv4/ip_forward', '1'),
    ('kernel/hostname', 'pwned'),
    ('vm/overcommit_memory', '1'),
    ('kernel/panic', '1'),
    ('vm/panic_on_oom', '1'),
]:
    try:
        with open(f'/proc/sys/{path}', 'w') as f:
            f.write(value)
        print(f'{path}:WRITTEN')
    except OSError as e:
        print(f'{path}:BLOCKED:{e.errno}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.splitlines() == [
            "kernel/core_pattern:BLOCKED:2",
            "kernel/modprobe:BLOCKED:2",
            "kernel/randomize_va_space:BLOCKED:30",
            "net/ipv4/ip_forward:BLOCKED:30",
            "kernel/hostname:BLOCKED:30",
            "vm/overcommit_memory:BLOCKED:30",
            "kernel/panic:BLOCKED:30",
            "vm/panic_on_oom:BLOCKED:30",
        ]

    async def test_read_sysctl_still_works(self, dual_scheduler: Scheduler) -> None:
        """Reading sysctl values still works (read-only, not hidden): vm.swappiness is the guest
        agent's 100."""
        code = """\
with open('/proc/sys/vm/swappiness') as f:
    val = f.read().strip()
print(f'SWAPPINESS:{val}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "SWAPPINESS:100\n"

    # --- Edge: indirect write vectors ---

    async def test_shell_redirect_write_blocked(self, dual_scheduler: Scheduler) -> None:
        """Shell redirect to /proc/sys is blocked too: the redirect fails, the value stays."""
        code = """\
import subprocess
r = subprocess.run(
    ["sh", "-c", "echo 0 > /proc/sys/kernel/randomize_va_space; echo EXIT:$?"],
    capture_output=True, text=True, timeout=5,
)
print(r.stdout.strip())
print(open('/proc/sys/kernel/randomize_va_space').read().strip())
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "EXIT:1\n2\n"  # the redirect failed; full ASLR (2), unchanged

    async def test_open_modes_all_blocked(self, dual_scheduler: Scheduler) -> None:
        """O_WRONLY, O_RDWR, and O_WRONLY|O_APPEND opens (open(2), no truncate) are all rejected on
        /proc/sys files with EACCES: the permission check of the root-owned file comes first."""
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "O_WRONLY:BLOCKED:13\nO_RDWR:BLOCKED:13\nO_APPEND:BLOCKED:13\n"

    async def test_nested_path_write_blocked(self, dual_scheduler: Scheduler) -> None:
        """Read-only applies recursively to deeply nested sysctl paths: EROFS on
        net/ipv4/conf/all/accept_redirects."""
        code = """\
try:
    with open('/proc/sys/net/ipv4/conf/all/accept_redirects', 'w') as f:
        f.write('0')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "BLOCKED:30\n"), result.stderr

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "BLOCKED:30\n"

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "BLOCKED:13\n"

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "BLOCKED:13\n"

    async def test_binfmt_misc_register_blocked(self, dual_scheduler: Scheduler) -> None:
        """Cannot register custom binary format interpreters: binfmt_misc is absent
        (CONFIG_BINFMT_MISC=n), so its register file does not exist."""
        code = "import os; print(os.path.exists('/proc/sys/fs/binfmt_misc/register'))"
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "False\n"), result.stderr

    # --- Out of bounds: privilege escalation to remount ---

    async def test_mount_and_umount_blocked(self, dual_scheduler: Scheduler) -> None:
        """A rw remount of /proc/sys, a tmpfs over it, its umount and a new procfs at /tmp/proc
        each require CAP_SYS_ADMIN: the command exits 1."""
        code = """\
import os, subprocess
os.makedirs('/tmp/proc', exist_ok=True)
for cmd in (
    'mount -o remount,rw /proc/sys',
    'mount -t tmpfs none /proc/sys',
    'umount /proc/sys',
    'mount -t proc proc /tmp/proc',
):
    try:
        print(f'{cmd}:{subprocess.run(cmd.split(), capture_output=True, timeout=5).returncode}')
    except subprocess.TimeoutExpired:  # the other commands are still tried
        print(f'{cmd}:TIMEOUT')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.splitlines() == [
            "mount -o remount,rw /proc/sys:1",
            "mount -t tmpfs none /proc/sys:1",
            "umount /proc/sys:1",
            "mount -t proc proc /tmp/proc:1",
        ]

    async def test_sysrq_disabled(self, dual_scheduler: Scheduler) -> None:
        """kernel.sysrq does not exist: CONFIG_MAGIC_SYSRQ=n compiles SysRq out."""
        code = "import os; print(os.path.exists('/proc/sys/kernel/sysrq'))"
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "False\n"), result.stderr


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

    async def test_dev_mounted_readonly(self, dual_scheduler: Scheduler) -> None:
        """/dev is mounted read-only; mknod returns EROFS (the read-only check comes before the
        CAP_MKNOD check, so EROFS shows this layer, not the capability)."""
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.splitlines() == ["DEV_RO:True", "MKNOD:errno_30"]  # EROFS

    # --- Normal: every access path to the removed nodes fails ---

    async def test_removed_nodes_refuse_every_access_path(self, dual_scheduler: Scheduler) -> None:
        """/dev/vda and /dev/mem refuse each access path: read, write and read-write opens,
        os.open(O_RDWR), read and write mmap (an open comes first), and the /proc/self/root
        traversal (WithSecure Labs vector). One VM tries each path in turn."""
        code = """\
import mmap, os

def mapped(path, mode, prot):
    with open(path, mode) as f:
        mmap.mmap(f.fileno(), 4096, mmap.MAP_SHARED, prot).close()

attempts = {
    'open-rb /dev/vda': lambda: open('/dev/vda', 'rb').close(),
    'open-wb /dev/vda': lambda: open('/dev/vda', 'wb').close(),
    'os.open-rdwr /dev/vda': lambda: os.close(os.open('/dev/vda', os.O_RDWR)),
    'mmap-read /dev/vda': lambda: mapped('/dev/vda', 'rb', mmap.PROT_READ),
    'mmap-write /dev/vda': lambda: mapped('/dev/vda', 'r+b', mmap.PROT_WRITE),
    'open-rb /proc/self/root/dev/vda': lambda: open('/proc/self/root/dev/vda', 'rb').close(),
    'open-rb /dev/mem': lambda: open('/dev/mem', 'rb').close(),
    'mmap-read /dev/mem': lambda: mapped('/dev/mem', 'rb', mmap.PROT_READ),
}
for label, attempt in attempts.items():
    try:
        attempt()
        print(f'{label}:OPENED')
    except OSError:
        print(f'{label}:blocked')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.splitlines() == [
            "open-rb /dev/vda:blocked",
            "open-wb /dev/vda:blocked",
            "os.open-rdwr /dev/vda:blocked",
            "mmap-read /dev/vda:blocked",
            "mmap-write /dev/vda:blocked",
            "open-rb /proc/self/root/dev/vda:blocked",
            "open-rb /dev/mem:blocked",
            "mmap-read /dev/mem:blocked",
        ]

    # --- Edge: alternative access paths closed ---

    async def test_open_by_handle_at_blocked(self, dual_scheduler: Scheduler) -> None:
        """open_by_handle_at fails with EPERM without CAP_DAC_READ_SEARCH (Shocker exploit).

        name_to_handle_at needs no capability: the test gets a real handle of /home/user and a
        real mount fd, so only the capability check can refuse the open.
        """
        code = """\
import ctypes, ctypes.util, errno, os, platform

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

class FileHandle(ctypes.Structure):
    _fields_ = [('handle_bytes', ctypes.c_uint), ('handle_type', ctypes.c_int), ('f_handle', ctypes.c_ubyte * 128)]

# name_to_handle_at, open_by_handle_at: 303, 304 on x86_64; 264, 265 on aarch64
name_nr, open_nr = (303, 304) if platform.machine() == 'x86_64' else (264, 265)
handle = FileHandle(handle_bytes=128)
mount_id = ctypes.c_int()
AT_FDCWD = -100
if libc.syscall(name_nr, AT_FDCWD, b'/home/user', ctypes.byref(handle), ctypes.byref(mount_id), 0) != 0:
    print(f'NO_HANDLE:{errno.errorcode[ctypes.get_errno()]}')
else:
    mount_fd = os.open('/home/user', os.O_RDONLY)
    ret = libc.syscall(open_nr, mount_fd, ctypes.byref(handle), os.O_RDONLY)
    print('unexpected_success' if ret >= 0 else f'BLOCKED:{errno.errorcode[ctypes.get_errno()]}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "BLOCKED:EPERM"

    # --- Weird: creative bypass attempts ---

    async def test_mknod_raw_syscall_blocked(self, dual_scheduler: Scheduler) -> None:
        """The raw mknodat syscall (ctypes) is refused by the kernel (Kata CVE-2020-2023 chain)."""
        code = """\
import ctypes, ctypes.util, errno, os, stat, platform

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

# SYS_mknodat = 259 on x86_64, 33 on aarch64
SYS_mknodat = 259 if platform.machine() == 'x86_64' else 33
AT_FDCWD = -100
ret = libc.syscall(SYS_mknodat, AT_FDCWD, b'/dev/test_mknod', stat.S_IFBLK | 0o660, os.makedev(253, 0))
print('unexpected_success' if ret == 0 else f'BLOCKED:{errno.errorcode[ctypes.get_errno()]}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "BLOCKED:EROFS\n"), result.stderr  # /dev is read-only

    async def test_debugfs_not_available(self, dual_scheduler: Scheduler) -> None:
        """debugfs binary doesn't exist in minimal Alpine rootfs."""
        code = """\
import shutil
print(f'DEBUGFS:{shutil.which("debugfs") is not None}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "DEBUGFS:False\n"

    async def test_uevent_helper_not_writable(self, dual_scheduler: Scheduler) -> None:
        """/sys/kernel/uevent_helper not writable by UID 1000."""
        code = """\
try:
    with open('/sys/kernel/uevent_helper', 'w') as f:
        f.write('/tmp/pwned')
    print('WRITTEN')
except OSError as e:
    print(f'BLOCKED:{e.errno}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "BLOCKED:13"

    # --- Out of bounds: comprehensive/adversarial ---

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "CAP_MKNOD:False\nCAP_DAC_READ_SEARCH:False\n"

    async def test_sysrq_trigger_not_writable(self, dual_scheduler: Scheduler) -> None:
        """/proc/sysrq-trigger does not exist (CONFIG_MAGIC_SYSRQ is off): ENOENT (2).

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "BLOCKED:2", (
            f"Expected sysrq-trigger write to be blocked.\nstdout: {result.stdout}"
        )


# =============================================================================
# Root filesystem read-only hardening
# =============================================================================
class TestRootFsReadonly:
    """Root filesystem is mounted read-only with nosuid and nodev flags.

    Matches AWS Lambda rootfs flags (ro,nosuid,nodev,noatime). tiny-init mounts the
    EROFS root read-only from boot (only the VM that creates a disk snapshot gets a
    writable overlay). Prevents writing to /etc/passwd, /sbin, planting SUID binaries, or
    remounting rw to DoS subsequent code.
    """

    async def test_mount_options(self, dual_scheduler: Scheduler) -> None:
        """The root file system is mounted ro, nosuid and nodev."""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), _MOUNT_OPTIONS.format("/"))
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "erofs ro,nosuid,nodev,noatime,user_xattr,acl,cache_strategy=readaround\n"

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "BLOCKED:30\n"  # EROFS

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "EXIT:1\n"

    async def test_home_user_and_dev_shm_still_writable(self, dual_scheduler: Scheduler) -> None:
        """/home/user (the home disk) and /dev/shm (a tmpfs) are separate mounts: still writable."""
        code = """\
import errno, os
for path in ('/home/user/test_rw.txt', '/dev/shm/test_rw.txt'):
    try:
        with open(path, 'w') as f:
            f.write('writable')
        with open(path) as f:
            print(f'{path}:{f.read()}')
        os.unlink(path)
    except OSError as e:  # the other path is still tried
        print(f'{path}:{errno.errorcode[e.errno]}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (
            0,
            "/home/user/test_rw.txt:writable\n/dev/shm/test_rw.txt:writable\n",
        ), result.stderr


# =============================================================================
# /home/user disk mount hardening
# =============================================================================
class TestHomeUserDisk:
    """/home/user is the VM's own ext4 home disk (serial=home), mounted by
    tiny-init with nosuid, nodev, noatime and discard.

    The disk size caps what user code can write there (no guest RAM is used).
    /home/user on rootfs is empty (cloudpickle is in
    /usr/lib/python3/site-packages), so the mount overlays an empty directory.
    """

    async def test_mount_options(self, dual_scheduler: Scheduler) -> None:
        """/home/user is the ext4 home disk, mounted nosuid and nodev but not noexec (uv and bun
        install executables), noatime (reads write nothing) and discard (deleted files free host
        disk blocks)."""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), _MOUNT_OPTIONS.format("/home/user"))
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "ext4 rw,nosuid,nodev,noatime,discard\n"

    async def test_no_lost_and_found(self, dual_scheduler: Scheduler) -> None:
        """mke2fs's root-only lost+found is removed: user tools see only the REPL scripts (as on
        tmpfs)."""
        code = "ls -A /home/user; tar -C /home/user -cf /tmp/home.tar . && echo TAR_OK"
        result = await run_once(dual_scheduler, VmSpec(language=Language.RAW), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "_repl.mjs\n_repl.py\n_repl.sh\nTAR_OK\n"

    async def test_size_matches_disk(self, dual_scheduler: Scheduler) -> None:
        """The default home disk (1024 MB) has 975 MB of usable space: ext4 metadata takes the rest."""
        code = """\
import os
s = os.statvfs('/home/user')
print(f'MB:{s.f_blocks * s.f_frsize // (1 << 20)}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "MB:975\n"), result.stderr

    async def test_cloudpickle_importable(self, dual_scheduler: Scheduler) -> None:
        """cloudpickle is importable (loaded from /usr/lib/python3/site-packages)."""
        code = """\
import cloudpickle
print(cloudpickle.__file__)
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "/usr/lib/python3/site-packages/cloudpickle/__init__.py\n"

    async def test_owned_by_uid_1000(self, dual_scheduler: Scheduler) -> None:
        """/home/user is owned by UID 1000 (sandbox user)."""
        code = """\
import os
s = os.stat('/home/user')
print(f'UID:{s.st_uid}')
print(f'GID:{s.st_gid}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "UID:1000\nGID:1000\n"


# =============================================================================
# Per-UID tmpfs quota enforcement (/dev/shm: RAM-backed)
# =============================================================================
class TestTmpfsQuotaEnforcement:
    """Real writes to a RAM-backed tmpfs stop at its size; sparse files take no pages.

    /dev/shm (40% of MemTotal) and /tmp (50%) are tmpfs: size= caps what all users write,
    and the per-UID quota (usrquota_block_hardlimit) has the same value. ftruncate() to a
    huge size takes no pages, so it succeeds, and list_files reports such a file at
    min(size, blocks * 512) (guest-agent/src/file_io.rs). /home/user is a disk: its size
    is the cap.
    """

    # --- Normal: sparse files don't consume quota ---

    async def test_sparse_truncate_succeeds(self, dual_scheduler: Scheduler) -> None:
        """ftruncate() to a huge apparent size succeeds and takes no block (metadata only): 50 GiB
        on the home disk, 1 GiB on /tmp and /dev/shm (the quota counts real blocks only)."""
        code = """\
import errno, os
for path, size in (('/home/user/sparse', 50 * 1024**3), ('/tmp/sparse', 1024**3), ('/dev/shm/sparse', 1024**3)):
    try:
        open(path, 'w').close()
        os.truncate(path, size)
        s = os.stat(path)
        print(f'{path}:{s.st_size}:{s.st_blocks}')
    except OSError as e:  # the other paths are still tried
        print(f'{path}:{errno.errorcode[e.errno]}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.splitlines() == [
            "/home/user/sparse:53687091200:0",
            "/tmp/sparse:1073741824:0",
            "/dev/shm/sparse:1073741824:0",
        ]

    # --- Edge: real writes hit quota/size limit ---

    async def test_real_write_exceeding_limit_fails(self, dual_scheduler: Scheduler) -> None:
        """Writing real data beyond the /dev/shm size fails with ENOSPC.

        /dev/shm is 40% of MemTotal (about 68 MiB with the default 192 MB of VM memory):
        300 MB of real data cannot fit. The per-UID quota has the same limit, and the kernel
        checks the size before the quota (shmem_inode_acct_blocks), so the error is ENOSPC,
        never EDQUOT.
        """
        code = """\
import errno
# Write in 10MB chunks to /dev/shm until failure
written = 0
try:
    with open('/dev/shm/bigfile', 'wb') as f:
        for _ in range(30):  # 300MB attempted
            f.write(b'A' * (10 * 1024 * 1024))
            f.flush()
            written += 10
    print(f'NO_LIMIT:{written}')
except OSError as e:
    print(f'BLOCKED:{written}')
    print(errno.errorcode[e.errno])
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        # MB written before the limit (it follows the VM memory), then the error
        assert re.fullmatch(r"BLOCKED:\d+\nENOSPC\n", result.stdout), result.stdout

    # --- Weird: mix sparse and real ---

    async def test_sparse_then_real_write_still_bounded(self, dual_scheduler: Scheduler) -> None:
        """Creating huge sparse files doesn't allow bypassing real write limits.

        Sparse inflation is free (no pages), but real writes still hit the
        size=/quota limit. This verifies the two are independent.
        """
        code = """\
import os
# Create 10 sparse files with 10GB apparent each (100GB total apparent)
for i in range(10):
    path = f'/dev/shm/sparse_{i}'
    open(path, 'w').close()
    os.truncate(path, 10 * 1024**3)

# Now try to write real data — should still hit limit
written = 0
try:
    with open('/dev/shm/realdata', 'wb') as f:
        for _ in range(30):
            f.write(b'B' * (10 * 1024 * 1024))
            f.flush()
            written += 10
    print(f'NO_LIMIT:{written}')
except OSError:
    print(f'BOUNDED:{written}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert re.fullmatch(r"BOUNDED:\d+\n", result.stdout), result.stdout  # MB written before the limit

    # --- Out of bounds: VM doesn't crash ---

    async def test_quota_exhaustion_doesnt_crash_vm(self, dual_scheduler: Scheduler) -> None:
        """Filling /dev/shm to its size doesn't crash the VM — exec still works after."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Step 1: Fill /dev/shm until failure
            exhaust = """\
written = 0
try:
    with open('/dev/shm/fill', 'wb') as f:
        for _ in range(50):
            f.write(b'X' * (10 * 1024 * 1024))
            f.flush()
            written += 10
except OSError:
    pass
print(f'FILLED:{written}MB')
"""
            r1 = await session.exec(exhaust)
            assert r1.exit_code == 0, r1.stderr
            assert re.fullmatch(r"FILLED:\d+MB\n", r1.stdout), r1.stdout

            # Step 2: Python execution still works (REPL doesn't need /dev/shm)
            r2 = await session.exec("print(f'MATH:{6 * 7}')")
            assert (r2.exit_code, r2.stdout) == (0, "MATH:42\n"), r2.stderr

            # Step 3: Clean up and verify writes work again
            r3 = await session.exec(
                """\
import os
os.unlink('/dev/shm/fill')
with open('/dev/shm/recovered.txt', 'w') as f:
    f.write('back to normal')
with open('/dev/shm/recovered.txt') as f:
    print(f'RECOVERED:{f.read()}')
""",
            )
            assert (r3.exit_code, r3.stdout) == (0, "RECOVERED:back to normal\n"), r3.stderr


# =============================================================================
# /bin and /sbin read-only hardening
# =============================================================================
class TestSystemBinaryHardening:
    """/bin and /sbin are bind-mounted read-only with nosuid.

    Blocks path hijack attacks where user code (UID 1000) replaces /bin/sh
    (a busybox symlink) with an arbitrary script, hijacking subsequent shebang
    (#!/bin/sh) execution within the same session. Primary protection is the
    RO rootfs from tiny-init; the guest agent's bind-mount adds nosuid as
    defense-in-depth (a root from a disk snapshot is an overlay without nosuid).

    /bin and /sbin are directories of the image (no usrmerge symlinks): each is its
    own mount.

    See: Microsoft Copilot Enterprise path hijack (Eye Security, April 2025).
    """

    # --- Normal: verify read-only mount flags ---

    async def test_bin_and_sbin_mounted_readonly(self, dual_scheduler: Scheduler) -> None:
        """/bin and /sbin are each a read-only bind mount of the EROFS root, with nosuid and nodev
        (as the root)."""
        # Each path's line comes after the path: a failure names the path it is about
        code = "print('/bin')\n" + _MOUNT_OPTIONS.format("/bin") + "print('/sbin')\n" + _MOUNT_OPTIONS.format("/sbin")
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        options = "erofs ro,nosuid,nodev,noatime,user_xattr,acl,cache_strategy=readaround"
        assert result.stdout.splitlines() == ["/bin", options, "/sbin", options]

    # --- Normal: write attempts blocked ---

    async def test_writes_blocked(self, dual_scheduler: Scheduler) -> None:
        """Each write to /bin or /sbin fails with EROFS: unlink of /bin/sh, a write open of
        /bin/busybox, a new file in /bin or /sbin, a symlink in /bin. One VM tries each."""
        code = """\
import errno, os
attempts = {
    'unlink /bin/sh': lambda: os.unlink('/bin/sh'),
    'open-wb /bin/busybox': lambda: open('/bin/busybox', 'wb').close(),
    'create /bin/evil': lambda: open('/bin/evil', 'w').close(),
    'create /sbin/evil': lambda: open('/sbin/evil', 'w').close(),
    'symlink /bin/evil_link': lambda: os.symlink('/tmp/evil', '/bin/evil_link'),
}
for label, attempt in attempts.items():
    try:
        attempt()
        print(f'{label}:DONE')
    except OSError as e:
        print(f'{label}:{errno.errorcode[e.errno]}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.splitlines() == [
            "unlink /bin/sh:EROFS",
            "open-wb /bin/busybox:EROFS",
            "create /bin/evil:EROFS",
            "create /sbin/evil:EROFS",
            "symlink /bin/evil_link:EROFS",
        ]

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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "OUTPUT:OK\nEXIT:0\n"

    async def test_busybox_applets_still_work(self, dual_scheduler: Scheduler) -> None:
        """ls and cat (GNU coreutils) still run from /bin."""
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "LS_EXIT:0\nCAT_EXIT:0\n"

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
        """The GNU version of the tool is installed, not the BusyBox applet."""
        code = f"""\
import subprocess
r = subprocess.run(["{binary}", "--version"], capture_output=True, text=True, timeout=5)
print(f'EXIT:{{r.returncode}}')
print(f'GNU:{{{version_marker!r} in r.stdout}}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "EXIT:0\nGNU:True\n"

    # --- Weird: mount bypass attempts ---

    async def test_mounts_blocked(self, dual_scheduler: Scheduler) -> None:
        """A rw remount of /bin or /sbin and a tmpfs over /bin each require CAP_SYS_ADMIN: the
        command exits 1."""
        code = """\
import subprocess
for cmd in ('mount -o remount,rw /bin', 'mount -o remount,rw /sbin', 'mount -t tmpfs none /bin'):
    try:
        print(f'{cmd}:{subprocess.run(cmd.split(), capture_output=True, timeout=5).returncode}')
    except subprocess.TimeoutExpired:  # the other commands are still tried
        print(f'{cmd}:TIMEOUT')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.splitlines() == [
            "mount -o remount,rw /bin:1",
            "mount -o remount,rw /sbin:1",
            "mount -t tmpfs none /bin:1",
        ]


# =============================================================================
# Thread/fork bomb mitigation (RLIMIT_NPROC + kernel.threads-max)
# =============================================================================
_UID_1000_TASKS = """\
import os

def uid_1000_tasks():
    tasks = 0
    for pid in filter(str.isdigit, os.listdir('/proc')):  # hidepid=2: the processes of this user
        try:
            if os.stat(f'/proc/{pid}').st_uid == 1000:
                tasks += len(os.listdir(f'/proc/{pid}/task'))
        except FileNotFoundError:  # it ended
            pass
    return tasks
"""
"""Guest code: uid_1000_tasks(), the tasks (threads) that RLIMIT_NPROC counts for UID 1000."""


class TestThreadBombMitigation:
    """RLIMIT_NPROC and kernel.threads-max cap thread/fork bombs.

    Defense-in-depth: RLIMIT_NPROC=1024 (per-UID, set via prlimit64 from the
    guest-agent after REPL spawn) is the primary defense. kernel.threads-max=1200
    (system-wide, a sysctl of the guest agent) is the secondary defense.
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
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "NPROC_SOFT:1024\nNPROC_HARD:1024\n"

    async def test_threadpool_executor_still_works(self, dual_scheduler: Scheduler) -> None:
        """ThreadPoolExecutor(max_workers=4) works within the limit."""
        code = """\
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(lambda x: x**2, range(100)))
print(f'RESULT:{len(results)}:{results[-1]}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "RESULT:100:9801\n"

    # --- Edge: thread bomb is capped ---

    async def test_thread_bomb_capped(self, dual_scheduler: Scheduler) -> None:
        """Thread bomb stops at RLIMIT_NPROC: threads count as processes of UID 1000, so the
        tasks of UID 1000 before the bomb and its threads are 1024."""
        code = (
            _UID_1000_TASKS
            + """\
import threading
import time

before = uid_1000_tasks()
threads = []
try:
    while True:
        t = threading.Thread(target=lambda: time.sleep(60), daemon=True)
        t.start()
        threads.append(t)
except RuntimeError:
    pass

print(f'BEFORE:{before} THREADS:{len(threads)}')
"""
        )
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        counts = re.fullmatch(r"BEFORE:(\d+) THREADS:(\d+)\n", result.stdout)
        assert counts, result.stdout
        assert int(counts[1]) + int(counts[2]) == 1024, result.stdout

    # --- Out of bounds: cannot escalate limits ---

    async def test_cannot_raise_nproc_limit(self, dual_scheduler: Scheduler) -> None:
        """User code cannot raise RLIMIT_NPROC above the hard limit."""
        code = """\
import resource
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (9999, 9999))
    print('RAISED:True')
except (ValueError, OSError) as e:
    print(f'RAISED:False:{type(e).__name__}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "RAISED:False:ValueError\n"

    async def test_fork_bomb_hits_nproc(self, dual_scheduler: Scheduler) -> None:
        """Fork bomb stops at RLIMIT_NPROC: the tasks of UID 1000 before the bomb and its children
        (zombies until reaped) are 1024."""
        code = (
            _UID_1000_TASKS
            + """\
import os

before = uid_1000_tasks()
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

print(f'BEFORE:{before} CHILDREN:{count}')
"""
        )
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        counts = re.fullmatch(r"BEFORE:(\d+) CHILDREN:(\d+)\n", result.stdout)
        assert counts, result.stdout
        assert int(counts[1]) + int(counts[2]) == 1024, result.stdout

"""Tests that psutil system introspection works inside the hardened VM.

psutil is a top-100 PyPI package used by monitoring agents, Jupyter (ipykernel
>= 6.9.2 depends on it for kernel resource tracking), ML frameworks (resource-
aware training in PyTorch Lightning / HuggingFace Accelerate), and virtually
every Python ops tool.  It reads from /proc/stat, /proc/meminfo, /proc/cpuinfo,
/proc/[pid]/*, /sys/class/hwmon, and other kernel interfaces.

These tests are regression canaries: if hardening (/proc/sys read-only, device
node removal, UID 1000, noexec on /dev/shm, etc.) inadvertently blocks any of
the /proc or /sys paths psutil needs, these tests will catch it.

Where we control the VM configuration (CPU count, memory size), tests
cross-validate that psutil reports values consistent with what QEMU was given.

Sensor functions (temperatures, fans, battery) are expected to gracefully return
empty results in a VM -- the tests verify they don't crash.

See:
- https://github.com/giampaolo/psutil
- https://psutil.readthedocs.io/
- https://github.com/jupyter-server/jupyter-resource-usage (uses psutil)
"""

import json

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.constants import DEFAULT_VM_CPU_CORES
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# Slow under TCG: psutil correctness tests inside guest — package install
# requires snapshot VM (slow under TCG). All assertions are correctness-only
# (0-100 range, non-None values), no tight timing thresholds.
pytestmark = pytest.mark.slow

PSUTIL_PACKAGES = ["psutil==7.2.1"]

# The default memory allocated to each VM by the scheduler (QEMU -m flag).
# We use this to cross-validate psutil.virtual_memory().total inside the guest.
_DEFAULT_MEMORY_MB = SchedulerConfig().default_memory_mb

# Kernel reserves memory for its own data structures (page tables, slab caches,
# reserved memory regions).  On a minimal Alpine Linux VM this overhead is
# typically 20-40 MB, so /proc/meminfo MemTotal < QEMU -m value.
# Kernel 6.18 on microvm with virtio_balloon + zram modules loaded shows
# ~38 MB overhead on 128 MB VMs. We cap at 30% for small VMs so a severe
# regression (e.g., QEMU allocating half the requested memory) does not pass.
_KERNEL_OVERHEAD_MB = 50
_MAX_KERNEL_OVERHEAD_RATIO = 0.30


def _memory_lower_bound_mb(configured_mb: int) -> float:
    """Minimum expected MemTotal for a given QEMU -m value.

    Uses the lesser of a fixed overhead (50 MB) and 25% of configured size
    so that small VMs (128 MB) get a tighter bound than large ones.
    """
    overhead = min(_KERNEL_OVERHEAD_MB, configured_mb * _MAX_KERNEL_OVERHEAD_RATIO)
    return configured_mb - overhead


# =============================================================================
# CPU introspection — reads /proc/stat, /proc/cpuinfo
# =============================================================================
class TestPsutilCpu:
    """psutil CPU functions read /proc/stat and /proc/cpuinfo."""

    async def test_cpu_count_logical(self, scheduler: Scheduler) -> None:
        """psutil.cpu_count(logical=True) matches QEMU -smp setting."""
        code = """\
import psutil
count = psutil.cpu_count(logical=True)
print(f'CPU_COUNT:{count}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        count = int(result.stdout.split("CPU_COUNT:")[1].strip())
        assert count == DEFAULT_VM_CPU_CORES, f"Expected {DEFAULT_VM_CPU_CORES} vCPU(s) (QEMU -smp), got {count}"

    async def test_cpu_count_physical(self, scheduler: Scheduler) -> None:
        """psutil.cpu_count(logical=False) matches QEMU -smp setting."""
        code = """\
import psutil
count = psutil.cpu_count(logical=False)
print(f'PHYSICAL_CPUS:{count}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        val = result.stdout.split("PHYSICAL_CPUS:")[1].strip()
        assert val != "None", "Expected physical CPU count, got None"
        assert int(val) == DEFAULT_VM_CPU_CORES, f"Expected {DEFAULT_VM_CPU_CORES} physical core(s), got {val}"

    async def test_cpu_percent(self, scheduler: Scheduler) -> None:
        """psutil.cpu_percent(interval=0.1) returns 0-100 float."""
        code = """\
import psutil
pct = psutil.cpu_percent(interval=0.1)
print(f'CPU_PERCENT:{pct}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        pct = float(result.stdout.split("CPU_PERCENT:")[1].strip())
        assert 0.0 <= pct <= 100.0

    async def test_cpu_times(self, scheduler: Scheduler) -> None:
        """psutil.cpu_times() returns user/system/idle from /proc/stat."""
        code = """\
import psutil
t = psutil.cpu_times()
print(f'USER:{t.user}')
print(f'SYSTEM:{t.system}')
print(f'IDLE:{t.idle}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        for label in ("USER:", "SYSTEM:", "IDLE:"):
            val = float(result.stdout.split(label)[1].split("\n")[0])
            assert val >= 0.0

    async def test_cpu_times_percent(self, scheduler: Scheduler) -> None:
        """psutil.cpu_times_percent() returns per-state CPU breakdown."""
        code = """\
import psutil
tp = psutil.cpu_times_percent(interval=0.1)
print(f'USER_PCT:{tp.user}')
print(f'SYSTEM_PCT:{tp.system}')
print(f'IDLE_PCT:{tp.idle}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        for label in ("USER_PCT:", "SYSTEM_PCT:", "IDLE_PCT:"):
            val = float(result.stdout.split(label)[1].split("\n")[0])
            assert 0.0 <= val <= 100.0, f"{label} {val} not in [0, 100]"

    async def test_cpu_stats(self, scheduler: Scheduler) -> None:
        """psutil.cpu_stats() returns ctx_switches, interrupts, etc."""
        code = """\
import psutil
stats = psutil.cpu_stats()
print(f'CTX_SWITCHES:{stats.ctx_switches}')
print(f'INTERRUPTS:{stats.interrupts}')
print(f'SOFT_INTERRUPTS:{stats.soft_interrupts}')
print(f'SYSCALLS:{stats.syscalls}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        ctx = int(result.stdout.split("CTX_SWITCHES:")[1].split("\n")[0])
        assert ctx >= 0

    async def test_cpu_freq(self, scheduler: Scheduler) -> None:
        """psutil.cpu_freq() returns frequency or None (both acceptable)."""
        code = """\
import psutil
freq = psutil.cpu_freq()
if freq is not None:
    print(f'CURRENT_MHZ:{freq.current}')
    print(f'HAS_FREQ:True')
else:
    # Some VMs don't expose frequency — that's fine
    print('HAS_FREQ:False')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "HAS_FREQ:" in result.stdout

    async def test_getloadavg(self, scheduler: Scheduler) -> None:
        """psutil.getloadavg() returns 1/5/15-min load averages."""
        code = """\
import psutil
load1, load5, load15 = psutil.getloadavg()
print(f'LOAD1:{load1}')
print(f'LOAD5:{load5}')
print(f'LOAD15:{load15}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        for label in ("LOAD1:", "LOAD5:", "LOAD15:"):
            val = float(result.stdout.split(label)[1].split("\n")[0])
            assert val >= 0.0


# =============================================================================
# Memory introspection — reads /proc/meminfo
# =============================================================================
class TestPsutilMemory:
    """psutil memory functions read /proc/meminfo."""

    async def test_virtual_memory(self, scheduler: Scheduler) -> None:
        """psutil.virtual_memory().total matches QEMU -m setting (minus kernel overhead)."""
        code = """\
import psutil
mem = psutil.virtual_memory()
print(f'TOTAL:{mem.total}')
print(f'AVAILABLE:{mem.available}')
print(f'PERCENT:{mem.percent}')
print(f'USED:{mem.used}')
print(f'FREE:{mem.free}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        total_bytes = int(result.stdout.split("TOTAL:")[1].split("\n")[0])
        total_mb = total_bytes / (1024 * 1024)
        # Kernel reserves memory for page tables, slab, etc.
        # MemTotal should be within [configured - overhead, configured].
        lower = _memory_lower_bound_mb(_DEFAULT_MEMORY_MB)
        assert total_mb >= lower, (
            f"psutil total {total_mb:.0f} MB too low for {_DEFAULT_MEMORY_MB} MB VM (floor {lower:.0f})"
        )
        assert total_mb <= _DEFAULT_MEMORY_MB, f"psutil total {total_mb:.0f} MB exceeds {_DEFAULT_MEMORY_MB} MB VM"
        available = int(result.stdout.split("AVAILABLE:")[1].split("\n")[0])
        assert 0 < available < total_bytes, "available must be strictly less than total (kernel uses memory)"
        pct = float(result.stdout.split("PERCENT:")[1].split("\n")[0])
        assert 0.0 <= pct <= 100.0
        used = int(result.stdout.split("USED:")[1].split("\n")[0])
        assert used >= 0
        free = int(result.stdout.split("FREE:")[1].split("\n")[0])
        assert free >= 0

    async def test_virtual_memory_linux_fields(self, scheduler: Scheduler) -> None:
        """Linux-specific fields (buffers, cached, shared, active, inactive)."""
        code = """\
import psutil
mem = psutil.virtual_memory()
fields = {}
for attr in ('buffers', 'cached', 'shared', 'active', 'inactive'):
    val = getattr(mem, attr, None)
    if val is not None:
        fields[attr] = val
import json
print(f'FIELDS:{json.dumps(fields)}')
print(f'FIELD_COUNT:{len(fields)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        count = int(result.stdout.split("FIELD_COUNT:")[1].strip())
        # Linux should expose at least buffers and cached
        assert count >= 2, f"Expected Linux-specific memory fields, got {count}"

    async def test_swap_memory(self, scheduler: Scheduler) -> None:
        """psutil.swap_memory() returns swap stats (VM uses zram swap)."""
        code = """\
import psutil
swap = psutil.swap_memory()
print(f'SWAP_TOTAL:{swap.total}')
print(f'SWAP_USED:{swap.used}')
print(f'SWAP_FREE:{swap.free}')
print(f'SWAP_PERCENT:{swap.percent}')
print(f'SWAP_SIN:{swap.sin}')
print(f'SWAP_SOUT:{swap.sout}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        total = int(result.stdout.split("SWAP_TOTAL:")[1].split("\n")[0])
        # VM has zram swap
        assert total > 0
        used = int(result.stdout.split("SWAP_USED:")[1].split("\n")[0])
        assert used >= 0
        free = int(result.stdout.split("SWAP_FREE:")[1].split("\n")[0])
        assert free >= 0
        pct = float(result.stdout.split("SWAP_PERCENT:")[1].split("\n")[0])
        assert 0.0 <= pct <= 100.0

    @pytest.mark.parametrize(
        "memory_mb",
        [
            pytest.param(128, id="128mb"),
            pytest.param(256, id="256mb"),
            pytest.param(512, id="512mb"),
        ],
    )
    async def test_virtual_memory_matches_configured_size(self, scheduler: Scheduler, memory_mb: int) -> None:
        """psutil.virtual_memory().total scales with the memory_mb we pass to QEMU."""
        code = """\
import psutil
total_mb = psutil.virtual_memory().total / (1024 * 1024)
print(f'TOTAL_MB:{total_mb:.1f}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
            memory_mb=memory_mb,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        total_mb = float(result.stdout.split("TOTAL_MB:")[1].strip())
        lower = _memory_lower_bound_mb(memory_mb)
        assert total_mb >= lower, f"psutil total {total_mb:.0f} MB too low for {memory_mb} MB VM (floor {lower:.0f})"
        assert total_mb <= memory_mb, f"psutil total {total_mb:.0f} MB exceeds {memory_mb} MB VM"


# =============================================================================
# Disk introspection — reads /proc/mounts, statvfs()
# =============================================================================
class TestPsutilDisk:
    """psutil disk functions read /proc/mounts and call statvfs()."""

    async def test_disk_usage_root(self, scheduler: Scheduler) -> None:
        """psutil.disk_usage('/') returns total/used/free/percent."""
        code = """\
import psutil
usage = psutil.disk_usage('/')
print(f'TOTAL:{usage.total}')
print(f'USED:{usage.used}')
print(f'FREE:{usage.free}')
print(f'PERCENT:{usage.percent}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        total = int(result.stdout.split("TOTAL:")[1].split("\n")[0])
        assert total > 0
        used = int(result.stdout.split("USED:")[1].split("\n")[0])
        assert used >= 0
        free = int(result.stdout.split("FREE:")[1].split("\n")[0])
        assert free >= 0
        pct = float(result.stdout.split("PERCENT:")[1].split("\n")[0])
        assert 0.0 <= pct <= 100.0

    async def test_disk_usage_home(self, scheduler: Scheduler) -> None:
        """psutil.disk_usage('/home/user') works for the REPL user dir."""
        code = """\
import psutil
usage = psutil.disk_usage('/home/user')
print(f'TOTAL:{usage.total}')
print(f'FREE:{usage.free}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        total = int(result.stdout.split("TOTAL:")[1].split("\n")[0])
        assert total > 0

    async def test_disk_partitions(self, scheduler: Scheduler) -> None:
        """psutil.disk_partitions() returns at least the root partition.

        Uses all=True because the VM's EROFS/overlayfs root is marked
        "nodev" in /proc/filesystems, so all=False filters it out.
        """
        code = """\
import psutil
parts = psutil.disk_partitions(all=True)
mountpoints = [p.mountpoint for p in parts]
print(f'COUNT:{len(parts)}')
print(f'HAS_ROOT:{"/" in mountpoints}')
for p in parts:
    print(f'PART:{p.mountpoint}|{p.fstype}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        count = int(result.stdout.split("COUNT:")[1].split("\n")[0])
        assert count >= 1
        assert "HAS_ROOT:True" in result.stdout

    async def test_disk_io_counters(self, scheduler: Scheduler) -> None:
        """psutil.disk_io_counters() returns I/O stats or None (diskless)."""
        code = """\
import psutil
counters = psutil.disk_io_counters()
if counters is not None:
    print(f'READ_COUNT:{counters.read_count}')
    print(f'WRITE_COUNT:{counters.write_count}')
    print(f'READ_BYTES:{counters.read_bytes}')
    print(f'WRITE_BYTES:{counters.write_bytes}')
    print(f'HAS_IO:True')
else:
    # /dev/vda removed — psutil may return None, that's acceptable
    print('HAS_IO:False')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "HAS_IO:" in result.stdout


# =============================================================================
# Network introspection — reads /proc/net/*, /sys/class/net/
# =============================================================================
class TestPsutilNetwork:
    """psutil network functions read /proc/net/ and /sys/class/net/."""

    async def test_net_if_addrs(self, scheduler: Scheduler) -> None:
        """psutil.net_if_addrs() returns at least loopback interface."""
        code = """\
import psutil
addrs = psutil.net_if_addrs()
interfaces = list(addrs.keys())
print(f'IFACE_COUNT:{len(interfaces)}')
print(f'HAS_LO:{"lo" in interfaces}')
print(f'INTERFACES:{",".join(interfaces)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        count = int(result.stdout.split("IFACE_COUNT:")[1].split("\n")[0])
        assert count >= 1
        assert "HAS_LO:True" in result.stdout

    async def test_net_if_stats(self, scheduler: Scheduler) -> None:
        """psutil.net_if_stats() returns per-NIC stats (isup, mtu, etc.)."""
        code = """\
import psutil
stats = psutil.net_if_stats()
for name, st in stats.items():
    print(f'NIC:{name}|isup={st.isup}|mtu={st.mtu}')
print(f'NIC_COUNT:{len(stats)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        count = int(result.stdout.split("NIC_COUNT:")[1].strip())
        assert count >= 1

    async def test_net_io_counters(self, scheduler: Scheduler) -> None:
        """psutil.net_io_counters() returns bytes/packets sent/received."""
        code = """\
import psutil
counters = psutil.net_io_counters()
if counters is not None:
    print(f'BYTES_SENT:{counters.bytes_sent}')
    print(f'BYTES_RECV:{counters.bytes_recv}')
    print(f'PACKETS_SENT:{counters.packets_sent}')
    print(f'PACKETS_RECV:{counters.packets_recv}')
    print(f'HAS_NET_IO:True')
else:
    print('HAS_NET_IO:False')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "HAS_NET_IO:" in result.stdout

    async def test_net_connections(self, scheduler: Scheduler) -> None:
        """psutil.net_connections() works for UID 1000 (non-root)."""
        code = """\
import psutil
try:
    conns = psutil.net_connections(kind='inet')
    print(f'CONN_COUNT:{len(conns)}')
    print('NET_CONN:OK')
except psutil.AccessDenied:
    # Non-root may get AccessDenied — that's acceptable
    print('NET_CONN:ACCESS_DENIED')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "NET_CONN:" in result.stdout


# =============================================================================
# Process introspection — reads /proc/[pid]/*
# =============================================================================
class TestPsutilProcess:
    """psutil process functions read /proc/[pid]/ subtree."""

    async def test_current_process_info(self, scheduler: Scheduler) -> None:
        """psutil.Process() returns pid, name, status, uid."""
        code = """\
import psutil, os
proc = psutil.Process()
print(f'PID:{proc.pid}')
print(f'NAME:{proc.name()}')
print(f'STATUS:{proc.status()}')
print(f'UID:{proc.uids().real}')
print(f'MATCHES_OS_PID:{proc.pid == os.getpid()}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "MATCHES_OS_PID:True" in result.stdout
        assert "STATUS:" in result.stdout

    async def test_process_memory_info(self, scheduler: Scheduler) -> None:
        """psutil.Process().memory_info() returns RSS and VMS."""
        code = """\
import psutil
proc = psutil.Process()
mem = proc.memory_info()
print(f'RSS:{mem.rss}')
print(f'VMS:{mem.vms}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        rss = int(result.stdout.split("RSS:")[1].split("\n")[0])
        vms = int(result.stdout.split("VMS:")[1].split("\n")[0])
        assert rss > 0
        assert vms > 0

    async def test_process_memory_percent(self, scheduler: Scheduler) -> None:
        """Process.memory_percent() returns 0-100 (used by jupyter-resource-usage)."""
        code = """\
import psutil
proc = psutil.Process()
pct = proc.memory_percent(memtype='rss')
print(f'MEM_PERCENT:{pct}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        pct = float(result.stdout.split("MEM_PERCENT:")[1].strip())
        assert 0.0 < pct <= 100.0

    async def test_process_cpu_times(self, scheduler: Scheduler) -> None:
        """Process.cpu_times() returns user/system times."""
        code = """\
import psutil
proc = psutil.Process()
times = proc.cpu_times()
print(f'USER:{times.user}')
print(f'SYSTEM:{times.system}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        user = float(result.stdout.split("USER:")[1].split("\n")[0])
        assert user >= 0.0

    async def test_process_oneshot(self, scheduler: Scheduler) -> None:
        """Process.oneshot() context manager caches multiple reads efficiently.

        This is the recommended pattern for monitoring agents that read many
        attributes at once (1.3x-6.5x speedup per the psutil docs).
        """
        code = """\
import psutil
proc = psutil.Process()
with proc.oneshot():
    info = {
        'pid': proc.pid,
        'name': proc.name(),
        'status': proc.status(),
        'username': proc.username(),
        'cpu_times': str(proc.cpu_times()),
        'memory_info': str(proc.memory_info()),
        'num_threads': proc.num_threads(),
    }
import json
print(f'ONESHOT:{json.dumps(info)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout.split("ONESHOT:")[1].strip())
        assert data["pid"] > 0
        assert data["num_threads"] >= 1

    async def test_process_as_dict(self, scheduler: Scheduler) -> None:
        """Process.as_dict() bulk-reads attributes (used by monitoring dashboards)."""
        code = """\
import psutil, json
proc = psutil.Process()
d = proc.as_dict(attrs=['pid', 'name', 'status', 'username', 'memory_info', 'cpu_times', 'num_threads', 'num_fds', 'create_time'])
# Serialize memory_info and cpu_times
d['memory_info'] = str(d['memory_info'])
d['cpu_times'] = str(d['cpu_times'])
print(f'AS_DICT:{json.dumps(d)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout.split("AS_DICT:")[1].strip())
        assert data["pid"] > 0
        assert data["num_fds"] >= 0

    async def test_pids_and_process_iter(self, scheduler: Scheduler) -> None:
        """psutil.pids() and process_iter() enumerate running processes."""
        code = """\
import psutil
pids = psutil.pids()
print(f'PID_COUNT:{len(pids)}')
print(f'HAS_PID_1:{1 in pids}')

# process_iter with attrs — the recommended pattern
procs = list(psutil.process_iter(['pid', 'name']))
print(f'ITER_COUNT:{len(procs)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        pid_count = int(result.stdout.split("PID_COUNT:")[1].split("\n")[0])
        assert pid_count >= 1
        # PID 1 hidden by hidepid=2 (root-owned, invisible to UID 1000)
        assert "HAS_PID_1:False" in result.stdout
        iter_count = int(result.stdout.split("ITER_COUNT:")[1].strip())
        assert iter_count >= 1

    async def test_process_children_and_parent(self, scheduler: Scheduler) -> None:
        """Process.children() and parent() navigate the process tree."""
        code = """\
import psutil
proc = psutil.Process()
children = proc.children(recursive=True)
print(f'CHILDREN_COUNT:{len(children)}')
parent = proc.parent()
print(f'HAS_PARENT:{parent is not None}')
if parent:
    print(f'PARENT_PID:{parent.pid}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "CHILDREN_COUNT:" in result.stdout
        assert "HAS_PARENT:" in result.stdout

    async def test_process_cwd(self, scheduler: Scheduler) -> None:
        """Process.cwd() returns the working directory."""
        code = """\
import psutil
proc = psutil.Process()
cwd = proc.cwd()
print(f'CWD:{cwd}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        cwd = result.stdout.split("CWD:")[1].strip()
        assert len(cwd) > 0

    async def test_process_environ(self, scheduler: Scheduler) -> None:
        """Process.environ() returns env vars dict.

        Note: psutil reads /proc/<pid>/environ which requires dumpable=1.
        Our REPL wrapper sets PR_SET_DUMPABLE=0 (CVE-2022-30594 mitigation),
        so psutil.Process().environ() raises AccessDenied. We use os.environ
        as a fallback to verify environment propagation, then confirm psutil
        correctly raises AccessDenied as expected under our hardening.
        """
        code = """\
import os, psutil
# os.environ works regardless of dumpable flag
env = dict(os.environ)
print(f'HAS_PATH:{"PATH" in env}')
print(f'ENV_COUNT:{len(env)}')
# Verify psutil raises AccessDenied due to PR_SET_DUMPABLE=0
try:
    psutil.Process().environ()
    print('PSUTIL_DUMPABLE:allowed')
except psutil.AccessDenied:
    print('PSUTIL_DUMPABLE:blocked')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "HAS_PATH:True" in result.stdout
        # Parse ENV_COUNT from the line (not the whole output)
        for line in result.stdout.strip().split("\n"):
            if line.startswith("ENV_COUNT:"):
                env_count = int(line.split(":")[1])
                assert env_count >= 1
                break
        else:
            pytest.fail("ENV_COUNT not found in output")
        # PR_SET_DUMPABLE=0 must block psutil's /proc/PID/environ read
        assert "PSUTIL_DUMPABLE:blocked" in result.stdout

    async def test_process_open_files(self, scheduler: Scheduler) -> None:
        """Process.open_files() lists open file descriptors."""
        code = """\
import psutil, tempfile
# Open a file so there's something to detect
with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    f.write('test')
    fname = f.name

with open(fname) as fh:
    proc = psutil.Process()
    files = proc.open_files()
    paths = [f.path for f in files]
    print(f'OPEN_COUNT:{len(files)}')
    print(f'FOUND_FILE:{fname in paths}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        count = int(result.stdout.split("OPEN_COUNT:")[1].split("\n")[0])
        assert count >= 1, "Expected at least one open file"
        assert "FOUND_FILE:True" in result.stdout, "Temp file not found in open_files()"

    async def test_process_io_counters(self, scheduler: Scheduler) -> None:
        """Process.io_counters() returns read/write byte counts.

        The method may not exist when the guest kernel lacks
        CONFIG_TASK_IO_ACCOUNTING (/proc/[pid]/io absent), in which
        case psutil never defines it on the Process class.
        """
        code = """\
import psutil
proc = psutil.Process()
if not hasattr(proc, 'io_counters'):
    # Kernel lacks /proc/[pid]/io — psutil omits the method entirely
    print('IO_OK:NOT_AVAILABLE')
else:
    try:
        io = proc.io_counters()
        print(f'READ_BYTES:{io.read_bytes}')
        print(f'WRITE_BYTES:{io.write_bytes}')
        print(f'IO_OK:True')
    except psutil.AccessDenied:
        # Acceptable if /proc/[pid]/io is restricted
        print('IO_OK:ACCESS_DENIED')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "IO_OK:" in result.stdout

    async def test_process_num_threads_and_fds(self, scheduler: Scheduler) -> None:
        """Process.num_threads() and num_fds() return counts."""
        code = """\
import psutil
proc = psutil.Process()
print(f'NUM_THREADS:{proc.num_threads()}')
print(f'NUM_FDS:{proc.num_fds()}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        threads = int(result.stdout.split("NUM_THREADS:")[1].split("\n")[0])
        assert threads >= 1
        fds = int(result.stdout.split("NUM_FDS:")[1].strip())
        assert fds >= 0

    async def test_process_threads(self, scheduler: Scheduler) -> None:
        """Process.threads() returns per-thread id and CPU times."""
        code = """\
import psutil
proc = psutil.Process()
threads = proc.threads()
print(f'THREAD_COUNT:{len(threads)}')
for t in threads[:3]:
    print(f'THREAD:{t.id}|user={t.user_time}|sys={t.system_time}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        count = int(result.stdout.split("THREAD_COUNT:")[1].split("\n")[0])
        assert count >= 1


# =============================================================================
# Sensors — reads /sys/class/hwmon/, /sys/class/thermal/
# =============================================================================
class TestPsutilSensors:
    """Sensor functions should gracefully return empty results in a VM.

    VMs don't have real hardware sensors, so these functions should return
    empty dicts or None — not crash or raise unexpected errors.
    """

    async def test_sensors_temperatures_graceful(self, scheduler: Scheduler) -> None:
        """sensors_temperatures() returns empty dict (no hwmon in VM)."""
        code = """\
import psutil
temps = psutil.sensors_temperatures()
print(f'TYPE:{type(temps).__name__}')
print(f'COUNT:{len(temps)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "TYPE:dict" in result.stdout

    async def test_sensors_fans_graceful(self, scheduler: Scheduler) -> None:
        """sensors_fans() returns empty dict (no fan sensors in VM)."""
        code = """\
import psutil
fans = psutil.sensors_fans()
print(f'TYPE:{type(fans).__name__}')
print(f'COUNT:{len(fans)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "TYPE:dict" in result.stdout

    async def test_sensors_battery_graceful(self, scheduler: Scheduler) -> None:
        """sensors_battery() returns None (no battery in VM)."""
        code = """\
import psutil
battery = psutil.sensors_battery()
print(f'BATTERY:{battery}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "BATTERY:None" in result.stdout


# =============================================================================
# System-wide info — reads /proc/uptime, utmp
# =============================================================================
class TestPsutilSystem:
    """psutil system-wide functions."""

    async def test_boot_time(self, scheduler: Scheduler) -> None:
        """psutil.boot_time() returns a valid timestamp."""
        code = """\
import psutil, time
boot = psutil.boot_time()
now = time.time()
print(f'BOOT_TIME:{boot}')
print(f'REASONABLE:{0 < boot <= now}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "REASONABLE:True" in result.stdout

    async def test_users(self, scheduler: Scheduler) -> None:
        """psutil.users() does not crash (may return empty list in VM)."""
        code = """\
import psutil
users = psutil.users()
print(f'USER_COUNT:{len(users)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "USER_COUNT:" in result.stdout

    async def test_pid_exists(self, scheduler: Scheduler) -> None:
        """psutil.pid_exists() checks PID existence."""
        code = """\
import psutil
print(f'PID1_EXISTS:{psutil.pid_exists(1)}')
print(f'BOGUS_PID:{psutil.pid_exists(999999)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        # PID 1 hidden by hidepid=2 (root-owned, invisible to UID 1000)
        assert "PID1_EXISTS:False" in result.stdout
        assert "BOGUS_PID:False" in result.stdout


# =============================================================================
# Realistic application patterns
# =============================================================================
class TestPsutilRealisticUsage:
    """Simulate how real applications use psutil for system discovery."""

    async def test_monitoring_agent_snapshot(self, scheduler: Scheduler) -> None:
        """Collect a full system overview like a monitoring agent would.

        This pattern is used by jupyter-resource-usage, Datadog agents,
        Prometheus exporters, and custom health-check endpoints.

        Cross-validates CPU count and memory against known VM configuration.
        """
        code = """\
import psutil
import json

info = {
    'cpu_count_logical': psutil.cpu_count(logical=True),
    'cpu_count_physical': psutil.cpu_count(logical=False),
    'memory_total_mb': round(psutil.virtual_memory().total / 1024 / 1024),
    'memory_available_mb': round(psutil.virtual_memory().available / 1024 / 1024),
    'memory_percent': psutil.virtual_memory().percent,
    'swap_total_mb': round(psutil.swap_memory().total / 1024 / 1024),
    'disk_total_mb': round(psutil.disk_usage('/').total / 1024 / 1024),
    'disk_percent': psutil.disk_usage('/').percent,
    'pid_count': len(psutil.pids()),
    'net_interfaces': len(psutil.net_if_addrs()),
    'boot_time': psutil.boot_time(),
    'load_avg_1min': psutil.getloadavg()[0],
}

# Validate all values are present and reasonable
all_ok = all(v is not None for v in info.values())
print(f'ALL_OK:{all_ok}')
print(f'OVERVIEW:{json.dumps(info)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
            timeout_seconds=30,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "ALL_OK:True" in result.stdout, f"Some psutil values missing:\n{result.stdout}"
        overview = json.loads(result.stdout.split("OVERVIEW:")[1].strip())
        # Cross-validate against known VM configuration
        assert overview["cpu_count_logical"] == DEFAULT_VM_CPU_CORES
        lower = _memory_lower_bound_mb(_DEFAULT_MEMORY_MB)
        assert overview["memory_total_mb"] >= lower
        assert overview["memory_total_mb"] <= _DEFAULT_MEMORY_MB
        assert 0 < overview["memory_available_mb"] < overview["memory_total_mb"]
        assert overview["swap_total_mb"] > 0, "VM should have zram swap"

    async def test_jupyter_resource_tracking_pattern(self, scheduler: Scheduler) -> None:
        """Simulate ipykernel/jupyter-resource-usage resource tracking.

        jupyter-resource-usage polls psutil every 5s for kernel memory (PSS/RSS)
        and optionally CPU percent.  ipykernel >= 6.11.0 uses psutil.Process()
        to report kernel resource usage in the sidebar.

        Cross-validates system totals against known VM configuration.
        """
        code = """\
import psutil, os

proc = psutil.Process(os.getpid())
# jupyter-resource-usage reads PSS on Linux, falls back to RSS
mem = proc.memory_info()
rss_mb = mem.rss / 1024 / 1024
print(f'RSS_MB:{rss_mb:.1f}')

# CPU percent with interval (as jupyter polls)
cpu_pct = proc.cpu_percent(interval=0.1)
print(f'CPU_PCT:{cpu_pct}')

# Memory percent of total system memory
mem_pct = proc.memory_percent(memtype='rss')
print(f'MEM_PCT:{mem_pct}')

# System-wide memory for limit display
sys_mem = psutil.virtual_memory()
print(f'SYS_TOTAL_MB:{sys_mem.total / 1024 / 1024:.0f}')
print(f'SYS_AVAIL_MB:{sys_mem.available / 1024 / 1024:.0f}')
print(f'JUPYTER_OK:True')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "JUPYTER_OK:True" in result.stdout
        rss = float(result.stdout.split("RSS_MB:")[1].split("\n")[0])
        assert rss > 0
        # Validate system total matches VM config
        sys_total = int(result.stdout.split("SYS_TOTAL_MB:")[1].split("\n")[0])
        lower = _memory_lower_bound_mb(_DEFAULT_MEMORY_MB)
        assert sys_total >= lower
        assert sys_total <= _DEFAULT_MEMORY_MB

    async def test_ml_resource_detection_pattern(self, scheduler: Scheduler) -> None:
        """Simulate ML framework resource detection (PyTorch Lightning, Accelerate).

        Before training, ML frameworks query CPU count + available memory to
        configure data loaders (num_workers) and batch sizes.

        Cross-validates that the detected resources match our VM settings.
        """
        code = """\
import psutil

# Detect available compute resources
cpu_count = psutil.cpu_count(logical=True)
mem = psutil.virtual_memory()
total_mb = mem.total / (1024 * 1024)
avail_mb = mem.available / (1024 * 1024)

# Typical heuristic: num_workers = min(cpu_count, 4)
num_workers = min(cpu_count, 4)

print(f'CPU_COUNT:{cpu_count}')
print(f'TOTAL_MB:{total_mb:.1f}')
print(f'AVAIL_MB:{avail_mb:.1f}')
print(f'NUM_WORKERS:{num_workers}')
print(f'ML_DETECT_OK:True')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "ML_DETECT_OK:True" in result.stdout
        cpu = int(result.stdout.split("CPU_COUNT:")[1].split("\n")[0])
        assert cpu == DEFAULT_VM_CPU_CORES, f"ML framework would see {cpu} CPUs, expected {DEFAULT_VM_CPU_CORES}"
        total_mb = float(result.stdout.split("TOTAL_MB:")[1].split("\n")[0])
        lower = _memory_lower_bound_mb(_DEFAULT_MEMORY_MB)
        assert total_mb >= lower
        assert total_mb <= _DEFAULT_MEMORY_MB

    async def test_process_tree_enumeration(self, scheduler: Scheduler) -> None:
        """Enumerate all processes — used by top/htop-like tools and debuggers."""
        code = """\
import psutil

procs = []
for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_info', 'cpu_percent']):
    try:
        info = proc.info
        procs.append({
            'pid': info['pid'],
            'name': info['name'],
            'user': info['username'],
        })
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

print(f'PROC_COUNT:{len(procs)}')
# PID 1 hidden by hidepid=2 (root-owned, invisible to UID 1000)
pid1_found = any(p['pid'] == 1 for p in procs)
print(f'PID1_FOUND:{pid1_found}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        count = int(result.stdout.split("PROC_COUNT:")[1].split("\n")[0])
        assert count >= 1
        assert "PID1_FOUND:False" in result.stdout

    @pytest.mark.parametrize(
        "code,check",
        [
            pytest.param(
                f"import psutil; print(f'OK:{{psutil.cpu_count() == {DEFAULT_VM_CPU_CORES}}}')",
                "OK:True",
                id="cpu-count-exact",
            ),
            pytest.param(
                "import psutil; print(f'OK:{psutil.virtual_memory().total > 0}')",
                "OK:True",
                id="memory-total",
            ),
            pytest.param(
                "import psutil; print(f'OK:{psutil.virtual_memory().available > 0}')",
                "OK:True",
                id="memory-available",
            ),
            pytest.param(
                "import psutil; print(f'OK:{psutil.disk_usage(\"/\").total > 0}')",
                "OK:True",
                id="disk-total",
            ),
            pytest.param(
                "import psutil; print(f'OK:{len(psutil.pids()) >= 1}')",
                "OK:True",
                id="pids-non-empty",
            ),
            pytest.param(
                "import psutil; print(f'OK:{psutil.boot_time() > 0}')",
                "OK:True",
                id="boot-time",
            ),
        ],
    )
    async def test_one_liner_checks(self, scheduler: Scheduler, code: str, check: str) -> None:
        """Quick smoke tests for common psutil one-liner patterns."""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=PSUTIL_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert check in result.stdout

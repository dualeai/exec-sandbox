"""Tests that Dask parallel computing works inside the hardened VM.

Dask is a top-100 PyPI package used for parallel/distributed computing in data
science, ML pipelines, and ETL workloads.  It reads system context from multiple
kernel interfaces to size its thread and process pools:

- os.cpu_count() / os.sched_getaffinity() for CPU detection
- /proc/meminfo via psutil.virtual_memory() for memory detection
- /proc/self/cgroup for cgroups v2 path resolution
- /sys/fs/cgroup/.../cpu.cfs_quota_us for cgroups v1 CPU quota
- /sys/fs/cgroup/.../cpu.max for cgroups v2 CPU quota
- /sys/fs/cgroup/memory/memory.limit_in_bytes for cgroups v1 memory
- /dev/shm for POSIX semaphores (multiprocessing.Pool)

These tests are regression canaries: if hardening (/proc/sys read-only, device
node removal, UID 1000, noexec on /dev/shm, etc.) inadvertently blocks any of
the system paths Dask needs, these tests will catch it.

Note: Dask's scheduler='processes' (ProcessPoolExecutor, spawn mode) is
incompatible with the REPL wrapper's set_start_method("fork") — spawned
children re-execute _repl.py and hit "context already set".  Fork-based
multiprocessing is tested separately in test_multiprocessing.py.

See:
- https://github.com/dask/dask/blob/main/dask/system.py
- https://github.com/dask/distributed/blob/main/distributed/system.py
- https://docs.dask.org/en/stable/scheduling.html
"""

from exec_sandbox.constants import DEFAULT_VM_CPU_CORES
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

from .conftest import skip_unless_hwaccel

DASK_PACKAGES = ["dask==2026.1.2"]


# =============================================================================
# System context detection — dask.system reads /proc/*, /sys/fs/cgroup/*
# =============================================================================
@skip_unless_hwaccel  # Snapshot creation (pip install dask + deps) routinely exceeds 940s under TCG
class TestDaskSystemContext:
    """dask.system CPU/memory detection works inside the hardened VM.

    Raw /proc and /sys reads (os.cpu_count, /proc/meminfo, /proc/self/status)
    are already covered by test_psutil_compatibility.py.  These tests verify
    that Dask's own detection layer — which adds affinity, cgroups quota, and
    cgroups memory limit on top of the OS primitives — works correctly.
    """

    async def test_cpu_count(self, scheduler: Scheduler) -> None:
        """dask.system.cpu_count() matches QEMU -smp (reads affinity + cgroups).

        Internally calls os.sched_getaffinity(), reads /proc/self/cgroup and
        /sys/fs/cgroup/.../cpu.max, and takes the minimum.
        """
        code = """\
from dask.system import cpu_count
count = cpu_count()
print(f'DASK_CPU:{count}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=DASK_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        count = int(result.stdout.split("DASK_CPU:")[1].strip())
        assert count == DEFAULT_VM_CPU_CORES, (
            f"dask.system.cpu_count() returned {count}, expected {DEFAULT_VM_CPU_CORES}"
        )

    async def test_cpu_count_constant(self, scheduler: Scheduler) -> None:
        """dask.system.CPU_COUNT module constant matches runtime cpu_count()."""
        code = """\
from dask.system import cpu_count, CPU_COUNT
print(f'FN:{cpu_count()}')
print(f'CONST:{CPU_COUNT}')
print(f'MATCH:{cpu_count() == CPU_COUNT}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=DASK_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "MATCH:True" in result.stdout


# =============================================================================
# Scheduler backends — synchronous, threaded
# =============================================================================
class TestDaskSchedulers:
    """Dask scheduler backends produce correct results.

    Tests synchronous and threaded schedulers.  The processes scheduler
    (ProcessPoolExecutor, spawn mode) is incompatible with the REPL
    wrapper's set_start_method("fork") — see module docstring.
    """

    async def test_synchronous_scheduler(self, scheduler: Scheduler) -> None:
        """Dask synchronous scheduler computes correctly (baseline)."""
        code = """\
import dask.bag as db

bag = db.from_sequence(range(100), npartitions=4)
result = bag.map(lambda x: x ** 2).sum().compute(scheduler='synchronous')
print(f'RESULT:{result}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=DASK_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        # sum(x**2 for x in range(100)) = 99*100*199/6 = 328350
        assert "RESULT:328350" in result.stdout

    async def test_threaded_scheduler(self, scheduler: Scheduler) -> None:
        """Dask threaded scheduler (ThreadPoolExecutor) computes correctly."""
        code = """\
import dask.bag as db

bag = db.from_sequence(range(100), npartitions=4)
result = bag.map(lambda x: x ** 2).sum().compute(scheduler='threads')
print(f'RESULT:{result}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=DASK_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "RESULT:328350" in result.stdout

    async def test_threaded_scheduler_multiple_partitions(self, scheduler: Scheduler) -> None:
        """Threaded scheduler with many partitions (exercises thread pool sizing)."""
        code = """\
import dask.bag as db

bag = db.from_sequence(range(1000), npartitions=16)
result = bag.map(lambda x: x * 2).filter(lambda x: x % 3 == 0).sum().compute(scheduler='threads')
print(f'RESULT:{result}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=DASK_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        # sum(x*2 for x in range(1000) if (x*2) % 3 == 0)
        # = sum(2*x for x in range(1000) if x % 3 == 0)
        # x in {0, 3, 6, ..., 999}: count = 334, sum of x = 3*(0+1+...+333) = 3*333*334/2 = 166833
        # sum of 2*x = 333666
        expected = sum(x * 2 for x in range(1000) if (x * 2) % 3 == 0)
        assert f"RESULT:{expected}" in result.stdout


# =============================================================================
# Dask.delayed — task graph API
# =============================================================================
class TestDaskDelayed:
    """dask.delayed builds and executes task graphs."""

    async def test_delayed_basic(self, scheduler: Scheduler) -> None:
        """dask.delayed computes a simple task graph."""
        code = """\
import dask

@dask.delayed
def add(x, y):
    return x + y

@dask.delayed
def mul(x, y):
    return x * y

# Build graph: (1+2) * (3+4) = 3 * 7 = 21
result = mul(add(1, 2), add(3, 4)).compute(scheduler='synchronous')
print(f'RESULT:{result}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=DASK_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "RESULT:21" in result.stdout

    async def test_delayed_threaded(self, scheduler: Scheduler) -> None:
        """dask.delayed executes task graph with threaded scheduler."""
        code = """\
import dask

@dask.delayed
def square(x):
    return x ** 2

# Build graph: sum of squares 0..9
tasks = [square(i) for i in range(10)]
total = dask.delayed(sum)(tasks)
result = total.compute(scheduler='threads')
print(f'RESULT:{result}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=DASK_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        # sum(x**2 for x in range(10)) = 9*10*19/6 = 285
        assert "RESULT:285" in result.stdout


# =============================================================================
# Realistic application patterns
# =============================================================================
class TestDaskRealisticUsage:
    """Simulate how real applications use Dask for parallel computing."""

    async def test_data_pipeline(self, scheduler: Scheduler) -> None:
        """ETL-style data pipeline with dask.bag (JSON-like record processing)."""
        code = """\
import dask.bag as db

records = [
    {'name': 'alice', 'score': 85},
    {'name': 'bob', 'score': 92},
    {'name': 'carol', 'score': 78},
    {'name': 'dave', 'score': 95},
    {'name': 'eve', 'score': 88},
]

bag = db.from_sequence(records, npartitions=2)
high_scores = (
    bag
    .filter(lambda r: r['score'] >= 85)
    .map(lambda r: r['name'])
    .compute(scheduler='threads')
)
print(f'COUNT:{len(high_scores)}')
print(f'NAMES:{sorted(high_scores)}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=DASK_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "COUNT:4" in result.stdout
        assert "NAMES:['alice', 'bob', 'dave', 'eve']" in result.stdout

    async def test_system_aware_worker_sizing(self, scheduler: Scheduler) -> None:
        """Dask uses detected CPU count to dynamically size partitions.

        This simulates how ML frameworks and data pipelines use Dask's
        system detection to configure parallelism at runtime.
        """
        code = """\
import dask.bag as db
from dask.system import CPU_COUNT

# Use detected CPU count to size a computation
bag = db.from_sequence(range(50), npartitions=CPU_COUNT)
result = bag.map(lambda x: x ** 2).sum().compute(scheduler='threads')
# sum(x**2 for x in range(50)) = 49*50*99/6 = 40425
print(f'NPARTITIONS:{CPU_COUNT}')
print(f'RESULT:{result}')
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            packages=DASK_PACKAGES,
        )
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        npartitions = int(result.stdout.split("NPARTITIONS:")[1].split("\n")[0])
        assert npartitions >= 1
        assert "RESULT:40425" in result.stdout

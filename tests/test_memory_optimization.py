"""Integration tests for memory optimization features (zram).

These tests verify that zram compression works correctly
in real QEMU VMs.

Run with: uv run pytest tests/test_memory_optimization.py -v
"""

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from exec_sandbox import Scheduler, SchedulerConfig, constants
from exec_sandbox.models import Language
from exec_sandbox.vm_spec import VmSpec

if TYPE_CHECKING:
    from exec_sandbox.models import ExecutionResult
    from exec_sandbox.vm_manager import VmManager

from tests.conftest import run_once, run_timed

from .conftest import skip_unless_fast_virt, skip_unless_hwaccel, skip_unless_macos_arm64

# Memory-intensive tests that push VMs to near-maximum utilization (250MB in 256MB VM)
# require fast memory operations. On x64 CI runners with nested virtualization
# (GitHub Actions on Azure), these operations are 50-100x slower due to missing
# TSC_DEADLINE timer support, causing timeouts and VM crashes. These tests are skipped
# on degraded environments via skip_unless_fast_virt and run on:
# - ARM64 Linux (native KVM, no nested virt penalty)
# - macOS (HVF, no nested virt)
# - Local development with native KVM/HVF

# ============================================================================
# zram Tests
# ============================================================================


class TestZramConfiguration:
    """Tests for the effective zram configuration in a cold guest."""

    async def test_zram_and_vm_configuration(
        self,
        vm_manager: "VmManager",
        execute_timeout: int,
    ) -> None:
        """A cold 192 MiB VM exposes the required effective zram geometry."""
        vm = await vm_manager.create_vm(
            VmSpec(language=Language.PYTHON, memory_mb=192, allow_network=False, allowed_domains=None),
            tenant_id="test",
            task_id="zram-config-192mb",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
        )
        try:
            result = await vm.execute(
                code=r"""
import os

with open('/proc/swaps') as f:
    swaps = f.read()
expected_path = r'/dev/zram0\040(deleted)'
rows = [
    line.split()
    for line in swaps.splitlines()[1:]
    if line.split() and line.split()[0] == expected_path
]
assert len(rows) == 1, f'expected one active unlinked zram swap, got: {swaps}'
path, swap_type, size_kib_raw, _used_kib, priority_raw = rows[0]
assert path == expected_path
assert swap_type == 'partition', f'unexpected zram swap type: {swap_type}'
assert int(priority_raw) >= 100, f'zram priority below 100: {priority_raw}'
swap_size_kib = int(size_kib_raw)

with open('/sys/block/zram0/disksize') as f:
    disksize = int(f.read().strip())
with open('/sys/block/zram0/comp_algorithm') as f:
    algorithm = f.read().strip()
assert '[lz4]' in algorithm, f'lz4 is not selected: {algorithm}'

with open('/proc/meminfo') as f:
    mem_total_kib = next(
        int(line.split()[1]) for line in f if line.startswith('MemTotal:')
    )
with open('/sys/block/zram0/mm_stat') as f:
    mm_stat = [int(value) for value in f.read().split()]
effective_mem_limit = mm_stat[3]

page_size = os.sysconf('SC_PAGE_SIZE')
memory_bytes = mem_total_kib * 1024
requested_disksize = memory_bytes * 40 // 100
requested_mem_limit = memory_bytes * 20 // 100
for name, effective, requested in (
    ('disksize', disksize, requested_disksize),
    ('mem_limit', effective_mem_limit, requested_mem_limit),
):
    assert effective % page_size == 0, f'{name} is not page-aligned: {effective}'
    assert requested <= effective < requested + page_size, (
        f'{name} is not the page-rounded requested value: '
        f'requested={requested}, effective={effective}, page_size={page_size}'
    )

expected_swap_size_kib = (disksize - page_size) // 1024
assert swap_size_kib == expected_swap_size_kib, (
    f'/proc/swaps size excludes more than the header page: '
    f'expected={expected_swap_size_kib}, actual={swap_size_kib}'
)
print('PASS: effective zram geometry')
""",
                timeout_seconds=execute_timeout,
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )
            assert result.exit_code == 0, result.stderr
            assert result.stdout == "PASS: effective zram geometry\n"
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_compression_actually_compresses(self, scheduler: Scheduler) -> None:
        """zram compresses what goes to swap: 130 % of MemAvailable, in 10 MiB chunks of a repeated
        256-byte sequence, swaps out, and lz4 stores it at least 2x smaller.

        Not one byte repeated: zram keeps a page of one repeated word without compression (same_pages),
        and compr_data_size would not grow."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.PYTHON),
            """
def zram():
    with open('/sys/block/zram0/mm_stat') as f:  # orig_data_size compr_data_size ...
        orig, compr = (int(v) for v in f.read().split()[:2])
    return orig, compr

with open('/proc/meminfo') as f:
    available_mb = next(int(line.split()[1]) for line in f if line.startswith('MemAvailable:')) // 1024
before = zram()
page = bytearray(range(256)) * 16
chunks = [page * 2560 for _ in range(available_mb * 13 // 100)]  # 10 MiB each
after = zram()
print(after[0] - before[0], after[1] - before[1])
""",
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        swapped, compressed = (int(v) for v in result.stdout.split())
        assert swapped >= 10 * 1024 * 1024, result.stdout  # the data went over RAM
        assert 0 < compressed <= swapped / 2, result.stdout


class TestZramMemoryExpansion:
    """Tests for zram enabling memory expansion beyond physical RAM."""

    async def test_allocate_well_beyond_physical_ram(self, scheduler: Scheduler) -> None:
        """VM should allocate well beyond available RAM using zram."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.PYTHON),
            """
import gc
gc.collect()

# Get available memory before allocation
with open('/proc/meminfo') as f:
    for line in f:
        if 'MemAvailable' in line:
            available_kb = int(line.split()[1])
            break

available_mb = available_kb // 1024
print(f'Available memory: {available_mb}MB')

# Allocate 130% of available — must exceed physical RAM via zram
target_mb = int(available_mb * 1.3)
# Round up to nearest 10MB chunk boundary
target_mb = ((target_mb + 9) // 10) * 10
print(f'Target allocation: {target_mb}MB')

chunks = []
allocated = 0
try:
    for i in range(target_mb // 10):
        chunk = bytearray(10 * 1024 * 1024)
        # Touch every page to force allocation
        for j in range(0, len(chunk), 4096):
            chunk[j] = 42
        chunks.append(chunk)
        allocated += 10

    # Verify we actually exceeded available RAM
    assert allocated > available_mb, f'Did not exceed available RAM: {allocated}MB <= {available_mb}MB'
    excess = allocated - available_mb
    print(f'PASS: Allocated {allocated}MB, exceeded available by {excess}MB')
except MemoryError:
    print(f'FAIL: MemoryError after {allocated}MB')
    raise
""",
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert re.fullmatch(
            r"Available memory: \d+MB\nTarget allocation: \d+MB\nPASS: Allocated \d+MB, exceeded available by \d+MB\n",
            result.stdout,
        ), result.stdout

    async def test_memory_survives_repeated_cycles(self, scheduler: Scheduler) -> None:
        """Memory allocation should work reliably across multiple cycles."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.PYTHON),
            """
import gc

def allocate_and_verify(size_mb, pattern_byte):
    '''Allocate memory, write pattern, verify it.'''
    chunks = []
    for i in range(size_mb // 10):
        chunk = bytearray(10 * 1024 * 1024)
        for j in range(0, len(chunk), 4096):
            chunk[j] = pattern_byte
        chunks.append(chunk)

    # Verify pattern
    for chunk in chunks:
        for j in range(0, len(chunk), 4096):
            assert chunk[j] == pattern_byte, f'Data corruption detected'

    return chunks

# Run 3 allocation cycles
for cycle in range(3):
    pattern = (cycle + 1) * 42  # Different pattern each cycle
    print(f'Cycle {cycle + 1}: Allocating 150MB with pattern {pattern}')

    chunks = allocate_and_verify(150, pattern)

    # Force garbage collection
    del chunks
    gc.collect()

    print(f'Cycle {cycle + 1}: PASS')

print('PASS: All 3 allocation cycles completed without corruption')
""",
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert (
            result.stdout
            == "".join(
                f"Cycle {cycle}: Allocating 150MB with pattern {cycle * 42}\nCycle {cycle}: PASS\n"
                for cycle in (1, 2, 3)
            )
            + "PASS: All 3 allocation cycles completed without corruption\n"
        )


# ============================================================================
# zram mem_limit Tests (OOM instead of thrashing)
# ============================================================================


_INCOMPRESSIBLE_PRESSURE = """
import os

print('WORKLOAD_STARTED', flush=True)
chunks = []
while True:
    chunks.append(os.urandom(10 * 1024 * 1024))
"""


# slow: not for its time (about 1 s) but to run alone (-n 0): its memory pressure slows the VMs of
# parallel tests, and their time limits fail (a QMP stop, a sudo bound: 2 of 3 runs)
@pytest.mark.slow
@skip_unless_hwaccel
class TestZramMemLimit:
    """Tests for retiring one-shot executions under capped-zram pressure.

    Without mem_limit, the kernel reclaim loop thrashes (compress/decompress
    in zram) when memory is exhausted, causing the VM to appear hung until
    timeout. Either the kernel OOM killer or the PSI guard may terminate the
    allocator first; the public result and confirmed host teardown are the
    stable contract.

    Refs:
        - https://docs.kernel.org/admin-guide/blockdev/zram.html (mem_limit)
        - https://lwn.net/Articles/612763/ (zram-full thrashing analysis)
    """

    @staticmethod
    def _pressure_config(images_dir: Path, tmp_path: Path) -> SchedulerConfig:
        return SchedulerConfig(
            images_dir=images_dir,
            auto_download_assets=False,
            warm_pool_size=0,
            default_timeout_seconds=60,
            disk_snapshot_cache_dir=tmp_path / "l2",
            memory_snapshot_cache_dir=tmp_path / "l1",
        )

    async def test_incompressible_pressure_retires_cold_192mb_vm(
        self,
        images_dir: Path,
        tmp_path: Path,
    ) -> None:
        """A cold-boot execution under incompressible pressure returns 137 fast."""
        config = self._pressure_config(images_dir, tmp_path)
        async with Scheduler(config) as scheduler:
            cold, cold_startup, _ = await asyncio.wait_for(
                run_timed(
                    scheduler,
                    VmSpec(language=Language.PYTHON, memory_mb=192),
                    _INCOMPRESSIBLE_PRESSURE,
                    timeout_seconds=60,
                ),
                timeout=30,
            )
            assert (cold.exit_code, cold.stdout) == (137, "WORKLOAD_STARTED\n"), cold.stderr
            assert cold_startup.warm_pool_hit is False
            assert cold_startup.l1_cache_hit is False

    @skip_unless_macos_arm64
    async def test_incompressible_pressure_retires_l1_restored_192mb_vm(
        self,
        images_dir: Path,
        tmp_path: Path,
    ) -> None:
        """A real L1-restored execution under incompressible pressure returns 137."""
        config = self._pressure_config(images_dir, tmp_path)
        async with Scheduler(config) as scheduler:
            # First run cold-boots and schedules a background L1 save; retry
            # through the public API until the save has landed and a run
            # restores from L1 (no reach into private snapshot-manager state).
            restored = None
            for _ in range(10):
                result, startup, _ = await asyncio.wait_for(
                    run_timed(
                        scheduler,
                        VmSpec(language=Language.PYTHON, memory_mb=192),
                        _INCOMPRESSIBLE_PRESSURE,
                        timeout_seconds=60,
                    ),
                    timeout=30,
                )
                assert (result.exit_code, result.stdout) == (137, "WORKLOAD_STARTED\n"), result.stderr
                assert startup.warm_pool_hit is False
                if startup.l1_cache_hit:
                    restored = result
                    break
                await asyncio.sleep(1)
            assert restored is not None, "no run restored from L1 within retry budget"


# ============================================================================
# Concurrent VM Tests
# ============================================================================


class TestConcurrentVMs:
    """Tests for multiple VMs running concurrently with memory features."""

    # slow: not for its time (about 1 s) but to run alone (-n 0): its memory pressure slows the VMs of
    # parallel tests, and their time limits fail (a QMP stop, a sudo bound: 2 of 3 runs)
    @pytest.mark.slow
    @skip_unless_fast_virt
    async def test_concurrent_vms_with_heavy_memory_pressure(self, images_dir: Path) -> None:
        """Concurrent VMs should each handle 180MB allocation.

        Runs 3 VMs simultaneously, each allocating 180MB in a 256MB VM (70%
        utilization). Tests that zram enables memory expansion under concurrent
        load. Skipped on nested virtualization (x64 CI) where memory operations
        are too slow.
        """
        config = SchedulerConfig(
            default_timeout_seconds=90,
            images_dir=images_dir,
        )

        async with Scheduler(config) as sched:
            code = """
import os

# Allocate 180MB per VM (requires zram to succeed)
chunks = []
for i in range(18):  # 180MB
    chunk = bytearray(10 * 1024 * 1024)
    for j in range(0, len(chunk), 4096):
        chunk[j] = (i * 7) % 256  # Unique pattern per chunk
    chunks.append(chunk)

# Verify data integrity
for i, chunk in enumerate(chunks):
    expected = (i * 7) % 256
    assert chunk[0] == expected, f'Chunk {i} corrupted'

# Report swap usage
with open('/proc/swaps') as f:
    lines = f.readlines()
    swap_used = int(lines[1].split()[3]) // 1024 if len(lines) > 1 else 0

print(f'PASS: 180MB allocated, swap_used={swap_used}MB')
"""
            tasks = [run_once(sched, VmSpec(language=Language.PYTHON, memory_mb=256), code) for _ in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            for i, r in enumerate(results):
                if isinstance(r, BaseException):
                    pytest.fail(f"VM {i + 1} failed with exception: {r}")
                result: ExecutionResult = r
                assert result.exit_code == 0, f"VM {i + 1} exit_code={result.exit_code}, stderr={result.stderr}"
                assert re.fullmatch(r"PASS: 180MB allocated, swap_used=\d+MB\n", result.stdout), result.stdout

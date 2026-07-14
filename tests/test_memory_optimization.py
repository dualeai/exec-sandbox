"""Integration tests for memory optimization features (balloon + zram).

These tests verify that virtio-balloon and zram compression work correctly
in real QEMU VMs.

Run with: uv run pytest tests/test_memory_optimization.py -v
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from exec_sandbox import Scheduler, SchedulerConfig
from exec_sandbox.warm_vm_pool import Language

if TYPE_CHECKING:
    from exec_sandbox.models import ExecutionResult
    from exec_sandbox.vm_manager import VmManager

from .conftest import skip_unless_fast_balloon, skip_unless_hwaccel, skip_unless_macos_arm64

# Memory-intensive tests that push VMs to near-maximum utilization (250MB in 256MB VM)
# require fast balloon/memory operations. On x64 CI runners with nested virtualization
# (GitHub Actions on Azure), these operations are 50-100x slower due to missing
# TSC_DEADLINE timer support, causing timeouts and VM crashes. These tests are skipped
# on degraded environments via skip_unless_fast_balloon and run on:
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
            language=Language.PYTHON,
            tenant_id="test",
            task_id="zram-config-192mb",
            memory_mb=192,
            allow_network=False,
            allowed_domains=None,
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
            )
            assert result.exit_code == 0, result.stderr
            assert "PASS: effective zram geometry" in result.stdout
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_compression_actually_compresses(self, scheduler: Scheduler) -> None:
        """zram should achieve real compression on compressible data."""
        result = await scheduler.run(
            code="""
import gc
gc.collect()

def get_zram_stats():
    '''Get orig_data_size and compr_data_size from mm_stat.'''
    with open('/sys/block/zram0/mm_stat') as f:
        parts = f.read().strip().split()
        # mm_stat format: orig_data_size compr_data_size mem_used_total ...
        return int(parts[0]), int(parts[1])

initial_orig, initial_compr = get_zram_stats()
print(f'Initial: orig={initial_orig}, compr={initial_compr}')

# Allocate compressible data (repetitive pattern compresses well)
# 120MB exceeds available RAM but fits within zram capacity thanks to compression.
# (zram mem_limit is 20% of RAM; repetitive patterns compress >10x with lz4.)
chunks = []
for i in range(12):  # 120MB of compressible data
    chunk = bytearray(10 * 1024 * 1024)
    # Fill with repetitive pattern (highly compressible)
    pattern = bytes([i % 256] * 4096)
    for j in range(0, len(chunk), 4096):
        chunk[j:j+4096] = pattern
    chunks.append(chunk)

# Force some to swap by accessing in reverse order
for chunk in reversed(chunks):
    _ = chunk[0]

final_orig, final_compr = get_zram_stats()
print(f'Final: orig={final_orig}, compr={final_compr}')

# If data was swapped, compression should be significant
if final_orig > initial_orig:
    data_swapped = final_orig - initial_orig
    data_compressed = final_compr - initial_compr
    if data_compressed > 0:
        ratio = data_swapped / data_compressed
        print(f'Compression ratio: {ratio:.2f}x')
        # lz4 should achieve at least 2x on repetitive data
        assert ratio >= 1.5, f'Compression ratio {ratio:.2f}x too low'
        print(f'PASS: Compression ratio {ratio:.2f}x')
    else:
        print('PASS: No compression needed (data fit in RAM)')
else:
    print('PASS: No swap used (data fit in RAM)')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout


class TestZramMemoryExpansion:
    """Tests for zram enabling memory expansion beyond physical RAM."""

    async def test_allocate_well_beyond_physical_ram(self, scheduler: Scheduler) -> None:
        """VM should allocate well beyond available RAM using zram."""
        result = await scheduler.run(
            code="""
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
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout
        assert "exceeded available by" in result.stdout

    async def test_memory_survives_repeated_cycles(self, scheduler: Scheduler) -> None:
        """Memory allocation should work reliably across multiple cycles."""
        result = await scheduler.run(
            code="""
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
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS: All 3 allocation cycles" in result.stdout


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
            cold = await asyncio.wait_for(
                scheduler.run(
                    code=_INCOMPRESSIBLE_PRESSURE,
                    language=Language.PYTHON,
                    memory_mb=192,
                    timeout_seconds=60,
                ),
                timeout=30,
            )
            assert cold.exit_code == 137, cold.stderr
            assert "WORKLOAD_STARTED" in cold.stdout
            assert cold.warm_pool_hit is False
            assert cold.l1_cache_hit is False

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
                result = await asyncio.wait_for(
                    scheduler.run(
                        code=_INCOMPRESSIBLE_PRESSURE,
                        language=Language.PYTHON,
                        memory_mb=192,
                        timeout_seconds=60,
                    ),
                    timeout=30,
                )
                assert result.exit_code == 137, result.stderr
                assert "WORKLOAD_STARTED" in result.stdout
                assert result.warm_pool_hit is False
                if result.l1_cache_hit:
                    restored = result
                    break
                await asyncio.sleep(1)
            assert restored is not None, "no run restored from L1 within retry budget"


# ============================================================================
# Balloon Tests
# ============================================================================


class TestBalloonDevice:
    """Tests for virtio-balloon device in guest.

    Balloon is only present on cold-boot VMs (anonymous memory).  Template-restored
    VMs (L1 cache hit) use file-backed COW memory which is incompatible with balloon.
    """

    async def test_balloon_device_and_driver(self, scheduler: Scheduler) -> None:
        """Balloon device should be visible with correct type (5), bound to a driver, and expose features.

        On L1-restored VMs (template COW), balloon is absent by design — verify
        that instead.
        """
        result = await scheduler.run(
            code="""
import os

virtio_path = '/sys/bus/virtio/devices'
assert os.path.exists(virtio_path), f'{virtio_path} not found'

found_balloon = False
for dev in os.listdir(virtio_path):
    modalias_path = os.path.join(virtio_path, dev, 'modalias')
    if os.path.exists(modalias_path):
        with open(modalias_path) as f:
            modalias = f.read().strip()
            if 'd00000005' not in modalias:
                continue

        found_balloon = True
        print(f'Found balloon device: {dev}')
        print(f'  modalias: {modalias}')

        # Driver must be bound (proves driver is working)
        driver_path = os.path.join(virtio_path, dev, 'driver')
        assert os.path.islink(driver_path), f'Balloon device {dev} not bound to driver'
        driver = os.path.basename(os.readlink(driver_path))
        print(f'  driver: {driver}')

        # Check features sysfs attribute
        features_path = os.path.join(virtio_path, dev, 'features')
        if os.path.exists(features_path):
            with open(features_path) as f:
                features = f.read().strip()
                print(f'  features: {features}')

        break

if found_balloon:
    print('PASS: Balloon device visible and driver functional')
else:
    print('NO_BALLOON: Balloon device absent (expected for template COW VMs)')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        if result.l1_cache_hit:
            # Template-restored VM: balloon absent by design (COW file-backed memory)
            assert "NO_BALLOON" in result.stdout
        else:
            # Cold-boot VM: balloon must be present
            assert "PASS" in result.stdout
            assert "d00000005" in result.stdout


# ============================================================================
# Concurrent VM Tests
# ============================================================================


class TestConcurrentVMs:
    """Tests for multiple VMs running concurrently with memory features."""

    @pytest.mark.slow
    @skip_unless_fast_balloon
    async def test_concurrent_vms_with_heavy_memory_pressure(self, images_dir: Path) -> None:
        """Concurrent VMs should each handle 180MB allocation.

        Runs 3 VMs simultaneously, each allocating 180MB in a 256MB VM (70%
        utilization). Tests that zram enables memory expansion under concurrent
        load. Skipped on nested virtualization (x64 CI) where memory operations
        are too slow.
        """
        config = SchedulerConfig(
            default_memory_mb=256,
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
            tasks = [sched.run(code=code, language=Language.PYTHON) for _ in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            for i, r in enumerate(results):
                if isinstance(r, BaseException):
                    pytest.fail(f"VM {i + 1} failed with exception: {r}")
                result: ExecutionResult = r
                assert result.exit_code == 0, f"VM {i + 1} exit_code={result.exit_code}, stderr={result.stderr}"
                assert "PASS" in result.stdout, f"VM {i + 1} output: {result.stdout}"

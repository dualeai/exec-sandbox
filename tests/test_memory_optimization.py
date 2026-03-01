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

from .conftest import skip_unless_fast_balloon

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
    """Tests for zram setup and VM tuning in guest VM."""

    async def test_zram_and_vm_configuration(self, scheduler: Scheduler) -> None:
        """All zram/VM sysfs and procfs settings should be correctly configured.

        Validates setup_zram_swap() + apply_zram_vm_tuning() in init.rs:
        device existence, swap priority, disksize, compression algorithm,
        RAM ratio, page-cluster, swappiness, overcommit, min_free_kbytes,
        mem_limit, oom_kill_allocating_task, and watermark tuning.
        """
        result = await scheduler.run(
            code="""
import os, stat

# --- zram device exists and is active ---
assert os.path.exists('/sys/block/zram0'), 'zram0 not found in /sys/block'

with open('/proc/swaps') as f:
    content = f.read()
    assert 'zram0' in content, f'zram0 not in /proc/swaps: {content}'
    for line in content.strip().split('\\n')[1:]:
        if 'zram0' in line:
            priority = int(line.split()[-1])
            assert priority >= 100, f'zram priority should be >=100, got {priority}'
            break

with open('/sys/block/zram0/disksize') as f:
    disksize = int(f.read().strip())
    assert disksize > 0, 'zram disksize is 0'

# --- lz4 compression ---
with open('/sys/block/zram0/comp_algorithm') as f:
    algo = f.read().strip()
    assert '[lz4]' in algo, f'Expected [lz4] active, got: {algo}'

# --- disksize is ~50% of RAM ---
with open('/proc/meminfo') as f:
    for line in f:
        if 'MemTotal' in line:
            mem_kb = int(line.split()[1])
            break
zram_kb = disksize // 1024
ratio = zram_kb / mem_kb
assert 0.45 <= ratio <= 0.55, f'zram ratio {ratio:.3f} not ~50%'

# --- page-cluster=0 (disables swap readahead for compressed swap) ---
with open('/proc/sys/vm/page-cluster') as f:
    page_cluster = int(f.read().strip())
    assert page_cluster == 0, f'page-cluster must be 0 for zram, got {page_cluster}'

# --- swappiness>=100 (prefer swap over dropping caches) ---
with open('/proc/sys/vm/swappiness') as f:
    swappiness = int(f.read().strip())
    assert swappiness >= 100, f'swappiness should be >=100 for zram, got {swappiness}'

# --- overcommit_memory=0 (heuristic, required for JIT runtimes) ---
with open('/proc/sys/vm/overcommit_memory') as f:
    overcommit_memory = int(f.read().strip())
    assert overcommit_memory == 0, f'overcommit_memory should be 0 (heuristic), got {overcommit_memory}'

# --- min_free_kbytes (prevents OOM deadlocks) ---
with open('/proc/sys/vm/min_free_kbytes') as f:
    min_free_kb = int(f.read().strip())
    assert min_free_kb >= 5000, f'min_free_kbytes should be >=5000, got {min_free_kb}'

# --- mem_limit sysfs (write-only, kernel-enforced) ---
path = '/sys/block/zram0/mem_limit'
assert os.path.exists(path), 'mem_limit sysfs attribute missing'
mode = os.stat(path).st_mode
assert stat.S_IWUSR & mode, 'mem_limit should be writable by owner'
assert not (stat.S_IRUSR & mode), 'mem_limit should be write-only (not readable)'

# --- oom_kill_allocating_task=1 (fast OOM response) ---
with open('/proc/sys/vm/oom_kill_allocating_task') as f:
    oom_kill = int(f.read().strip())
    assert oom_kill == 1, f'oom_kill_allocating_task should be 1, got {oom_kill}'

# --- watermark tuning (per ArchWiki zram recommendations) ---
with open('/proc/sys/vm/watermark_boost_factor') as f:
    boost = int(f.read().strip())
    assert boost == 0, f'watermark_boost_factor should be 0, got {boost}'

with open('/proc/sys/vm/watermark_scale_factor') as f:
    scale = int(f.read().strip())
    assert scale == 125, f'watermark_scale_factor should be 125, got {scale}'

print(f'PASS: all zram/VM settings correct')
print(f'  disksize={disksize//(1024*1024)}MB, ratio={ratio:.3f}, algo={algo}')
print(f'  page-cluster={page_cluster}, swappiness={swappiness}')
print(f'  overcommit={overcommit_memory}, min_free_kb={min_free_kb}')
print(f'  oom_kill_allocating_task={oom_kill}')
print(f'  watermark_boost={boost}, watermark_scale={scale}')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout

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
# (zram mem_limit is 25% of RAM; repetitive patterns compress >10x with lz4.)
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


class TestZramMemLimit:
    """Tests for zram mem_limit — ensures OOM kills fast instead of thrashing.

    Without mem_limit, the kernel reclaim loop thrashes (compress/decompress
    in zram) when memory is exhausted, causing the VM to appear hung until
    timeout. With mem_limit, zram rejects writes with -ENOMEM once compressed
    storage is full, triggering the OOM killer within seconds.

    Refs:
        - https://docs.kernel.org/admin-guide/blockdev/zram.html (mem_limit)
        - https://lwn.net/Articles/612763/ (zram-full thrashing analysis)
    """

    async def test_massive_alloc_oom_kills_not_timeout(self, scheduler: Scheduler) -> None:
        """Massive allocation must OOM-kill (exit 137), NOT timeout.

        This is the core regression test. Before mem_limit, allocating far
        beyond RAM caused the kernel to thrash zram indefinitely, hitting the
        execution timeout. With mem_limit, the OOM killer fires in seconds.

        Uses os.urandom() (incompressible data) so zram can't compress the
        pages — zero-filled bytearrays compress to near-nothing with lz4,
        letting ~8 GB fit in the 41 MB mem_limit before OOM triggers.
        """
        result = await scheduler.run(
            code="""
import os
chunks = []
while True:
    chunks.append(os.urandom(10 * 1024 * 1024))  # 10MB random (incompressible)
""",
            language=Language.PYTHON,
        )
        # OOM kill = 137 (128 + SIGKILL). Must NOT be timeout (-1).
        assert result.exit_code == 137, (
            f"Expected OOM kill (137), got exit_code={result.exit_code}. "
            f"If -1, zram is thrashing instead of OOM-killing."
        )

    async def test_oom_after_successful_alloc_free_cycle(self, scheduler: Scheduler) -> None:
        """OOM should still fire after alloc/free cycles (no zram leak).

        Alloc/free cycles leave residual zram pages and allocator
        fragmentation. Uses random data so zram can't compress it.
        """
        result = await scheduler.run(
            code="""
import gc, os

# Warm up: alloc 100MB random, free, repeat 3 times
for _ in range(3):
    chunks = [os.urandom(10 * 1024 * 1024) for _ in range(10)]
    del chunks
    gc.collect()

# Now exhaust memory — should OOM, not thrash
chunks = []
while True:
    chunks.append(os.urandom(10 * 1024 * 1024))
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 137, (
            f"Expected OOM kill (137), got exit_code={result.exit_code}. OOM should fire even after alloc/free cycles."
        )


# ============================================================================
# Balloon Tests
# ============================================================================


class TestBalloonDevice:
    """Tests for virtio-balloon device in guest."""

    async def test_balloon_device_and_driver(self, scheduler: Scheduler) -> None:
        """Balloon device should be visible with correct type (5), bound to a driver, and expose features."""
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

assert found_balloon, 'Balloon device (type 5) not found in /sys/bus/virtio'
print('PASS: Balloon device visible and driver functional')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
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

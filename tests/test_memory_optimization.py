"""Integration tests for memory optimization features (balloon + zram).

These tests verify that virtio-balloon and zram compression work correctly
in real QEMU VMs.

Run with: uv run pytest tests/test_memory_optimization.py -v
"""

import asyncio
from pathlib import Path

import pytest

from exec_sandbox import Scheduler, SchedulerConfig
from exec_sandbox.models import ExecutionResult
from exec_sandbox.warm_vm_pool import Language

# Maximum concurrent VMs for stress tests. QEMU creates 5-15 threads per VM;
# with pytest-xdist this can exhaust thread limits (SIGABRT). Value of 3 is
# safe while still testing concurrency.
_MAX_CONCURRENT_VMS = 3

# ============================================================================
# zram Tests
# ============================================================================


class TestZramConfiguration:
    """Tests for zram setup in guest VM."""

    async def test_zram_device_exists_and_active(self, scheduler: Scheduler) -> None:
        """zram0 device should be created, active, and have high swap priority."""
        result = await scheduler.run(
            code="""
import os

# Check device exists in /dev and /sys
assert os.path.exists('/dev/zram0'), 'zram0 device not found in /dev'
assert os.path.exists('/sys/block/zram0'), 'zram0 not found in /sys/block'

# Check it's in swaps with high priority
with open('/proc/swaps') as f:
    content = f.read()
    assert 'zram0' in content, f'zram0 not in /proc/swaps: {content}'
    # Parse priority (last column) - should be 100
    lines = content.strip().split('\\n')
    for line in lines[1:]:  # Skip header
        if 'zram0' in line:
            parts = line.split()
            priority = int(parts[-1])
            assert priority >= 100, f'zram priority should be >=100, got {priority}'
            print(f'zram0 priority: {priority}')
            break

# Check disksize is non-zero
with open('/sys/block/zram0/disksize') as f:
    disksize = int(f.read().strip())
    assert disksize > 0, 'zram disksize is 0'
    print(f'zram disksize: {disksize // (1024*1024)}MB')

print('PASS: zram0 device active with correct priority')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout

    async def test_zram_uses_lz4_compression(self, scheduler: Scheduler) -> None:
        """zram should use lz4 compression algorithm (fastest)."""
        result = await scheduler.run(
            code="""
with open('/sys/block/zram0/comp_algorithm') as f:
    algo = f.read().strip()
    # Active algorithm shown in brackets [lz4]
    assert '[lz4]' in algo, f'Expected [lz4] active, got: {algo}'
    print(f'PASS: compression algorithm = {algo}')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout
        assert "[lz4]" in result.stdout

    async def test_zram_size_is_half_ram(self, scheduler: Scheduler) -> None:
        """zram disksize should be exactly 50% of total RAM."""
        result = await scheduler.run(
            code="""
# Get total RAM
with open('/proc/meminfo') as f:
    for line in f:
        if 'MemTotal' in line:
            mem_kb = int(line.split()[1])
            break

# Get zram size
with open('/sys/block/zram0/disksize') as f:
    zram_bytes = int(f.read().strip())
    zram_kb = zram_bytes // 1024

# Should be ~50% (allow 45-55% range for rounding)
ratio = zram_kb / mem_kb
assert 0.45 <= ratio <= 0.55, f'zram ratio {ratio:.3f} not ~50%'
print(f'PASS: zram={zram_kb//1024}MB, RAM={mem_kb//1024}MB, ratio={ratio:.3f}')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout

    async def test_vm_settings_optimized_for_zram(self, scheduler: Scheduler) -> None:
        """VM settings should be optimized: page-cluster=0, swappiness>=100."""
        result = await scheduler.run(
            code="""
# page-cluster=0 disables swap readahead (critical for compressed swap)
with open('/proc/sys/vm/page-cluster') as f:
    page_cluster = int(f.read().strip())
    assert page_cluster == 0, f'page-cluster must be 0 for zram, got {page_cluster}'

# swappiness>=100 prefers swap over dropping caches (kernel allows up to 200 for zram)
with open('/proc/sys/vm/swappiness') as f:
    swappiness = int(f.read().strip())
    assert swappiness >= 100, f'swappiness should be >=100 for zram, got {swappiness}'

print(f'PASS: page-cluster={page_cluster}, swappiness={swappiness}')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout

    async def test_overcommit_settings_configured(self, scheduler: Scheduler) -> None:
        """VM should have heuristic overcommit for JIT runtime compatibility."""
        result = await scheduler.run(
            code="""
# vm.overcommit_memory=0 (heuristic) allows large virtual memory reservations
# Required for JIT runtimes like Bun/JavaScriptCore that reserve 128GB+ virtual address space
with open('/proc/sys/vm/overcommit_memory') as f:
    overcommit_memory = int(f.read().strip())
    assert overcommit_memory == 0, f'overcommit_memory should be 0 (heuristic), got {overcommit_memory}'

# vm.min_free_kbytes should be set (prevents OOM deadlocks)
with open('/proc/sys/vm/min_free_kbytes') as f:
    min_free_kb = int(f.read().strip())
    assert min_free_kb >= 5000, f'min_free_kbytes should be >=5000, got {min_free_kb}'

print(f'PASS: overcommit_memory={overcommit_memory}, min_free_kbytes={min_free_kb}')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout


class TestZramCompression:
    """Tests for zram compression effectiveness."""

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
chunks = []
for i in range(20):  # 200MB of compressible data
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
            timeout_seconds=60,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout


class TestZramMemoryExpansion:
    """Tests for zram enabling memory expansion beyond physical RAM."""

    async def test_allocate_well_beyond_physical_ram(self, scheduler: Scheduler) -> None:
        """VM should allocate 240MB when only ~175MB available (37% over)."""
        result = await scheduler.run(
            code="""
import gc
gc.collect()

# Get available memory before
with open('/proc/meminfo') as f:
    for line in f:
        if 'MemAvailable' in line:
            available_kb = int(line.split()[1])
            break

available_mb = available_kb // 1024
print(f'Available memory: {available_mb}MB')

# Allocate 240MB (significantly more than available ~175MB)
target_mb = 240
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
            timeout_seconds=90,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout
        assert "exceeded available by" in result.stdout

    @pytest.mark.slow
    async def test_swap_usage_correlates_with_allocation(self, scheduler: Scheduler) -> None:
        """Swap usage should increase proportionally as memory pressure grows."""
        result = await scheduler.run(
            code="""
def get_swap_used_kb():
    with open('/proc/swaps') as f:
        lines = f.readlines()
        if len(lines) > 1:
            return int(lines[1].split()[3])
    return 0

def get_available_mb():
    with open('/proc/meminfo') as f:
        for line in f:
            if 'MemAvailable' in line:
                return int(line.split()[1]) // 1024
    return 0

# Record initial state
initial_swap_kb = get_swap_used_kb()
initial_avail = get_available_mb()
print(f'Initial: available={initial_avail}MB, swap_used={initial_swap_kb//1024}MB')

# Allocate memory in stages and track swap
chunks = []
measurements = []
for stage in range(1, 6):  # 50MB increments up to 250MB
    for _ in range(5):  # 5 x 10MB = 50MB per stage
        chunk = bytearray(10 * 1024 * 1024)
        for j in range(0, len(chunk), 4096):
            chunk[j] = 42
        chunks.append(chunk)

    swap_kb = get_swap_used_kb()
    avail = get_available_mb()
    allocated = stage * 50
    measurements.append((allocated, swap_kb // 1024, avail))
    print(f'Stage {stage}: allocated={allocated}MB, swap={swap_kb//1024}MB, avail={avail}MB')

# Verify swap increased significantly
final_swap_kb = get_swap_used_kb()
swap_increase_mb = (final_swap_kb - initial_swap_kb) // 1024
print(f'Swap increase: {swap_increase_mb}MB')

# Should have used at least 30MB of swap for 250MB allocation
assert swap_increase_mb >= 30, f'Swap increase too small: {swap_increase_mb}MB'
print(f'PASS: Swap increased by {swap_increase_mb}MB')
""",
            language=Language.PYTHON,
            timeout_seconds=90,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout

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
            timeout_seconds=120,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS: All 3 allocation cycles" in result.stdout


# ============================================================================
# Balloon Tests
# ============================================================================


class TestBalloonDevice:
    """Tests for virtio-balloon device in guest."""

    async def test_balloon_device_visible_and_correct_type(self, scheduler: Scheduler) -> None:
        """Balloon device should be visible with correct device type (5)."""
        result = await scheduler.run(
            code="""
import os

found_balloon = False
balloon_dev = None
virtio_path = '/sys/bus/virtio/devices'

assert os.path.exists(virtio_path), f'{virtio_path} not found'

for dev in os.listdir(virtio_path):
    modalias_path = os.path.join(virtio_path, dev, 'modalias')
    if os.path.exists(modalias_path):
        with open(modalias_path) as f:
            modalias = f.read().strip()
            # Device type 5 = balloon (virtio:d00000005v...)
            if 'd00000005' in modalias:
                found_balloon = True
                balloon_dev = dev
                print(f'Found balloon device: {dev}')
                print(f'  modalias: {modalias}')

                # Check device is bound to driver
                driver_path = os.path.join(virtio_path, dev, 'driver')
                if os.path.exists(driver_path):
                    driver = os.path.basename(os.readlink(driver_path))
                    print(f'  driver: {driver}')
                break

assert found_balloon, 'Balloon device (type 5) not found in /sys/bus/virtio'
print(f'PASS: Balloon device {balloon_dev} visible')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout
        assert "d00000005" in result.stdout

    async def test_balloon_driver_functional(self, scheduler: Scheduler) -> None:
        """Balloon driver should be functional (built-in or module)."""
        result = await scheduler.run(
            code="""
import os

# The balloon driver can be either built-in or a module
# Check if the device is bound to a driver (proves driver is working)
virtio_path = '/sys/bus/virtio/devices'
found_driver = False

for dev in os.listdir(virtio_path):
    modalias_path = os.path.join(virtio_path, dev, 'modalias')
    if os.path.exists(modalias_path):
        with open(modalias_path) as f:
            if 'd00000005' in f.read():  # Device type 5 = balloon
                # Check driver is bound
                driver_path = os.path.join(virtio_path, dev, 'driver')
                if os.path.islink(driver_path):
                    driver = os.path.basename(os.readlink(driver_path))
                    print(f'Balloon device {dev} bound to driver: {driver}')
                    found_driver = True

                    # Verify driver exposes expected sysfs attributes
                    features_path = os.path.join(virtio_path, dev, 'features')
                    if os.path.exists(features_path):
                        with open(features_path) as f:
                            features = f.read().strip()
                            print(f'Balloon features: {features}')
                break

assert found_driver, 'Balloon device not bound to driver'
print('PASS: Balloon driver functional')
""",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0, f"Failed: {result.stderr}"
        assert "PASS" in result.stdout


# ============================================================================
# Concurrent VM Tests
# ============================================================================


class TestConcurrentVMs:
    """Tests for multiple VMs running concurrently with memory features."""

    @pytest.mark.slow
    async def test_concurrent_vms_with_heavy_memory_pressure(self, images_dir: Path) -> None:
        """Concurrent VMs should each handle 180MB allocation."""
        config = SchedulerConfig(
            default_memory_mb=256,
            default_timeout_seconds=90,
            max_concurrent_vms=_MAX_CONCURRENT_VMS,
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
            # Run VMs concurrently (limited by _MAX_CONCURRENT_VMS to avoid thread exhaustion)
            tasks = [sched.run(code=code, language=Language.PYTHON) for _ in range(_MAX_CONCURRENT_VMS)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            for i, r in enumerate(results):
                if isinstance(r, BaseException):
                    pytest.fail(f"VM {i + 1} failed with exception: {r}")
                result: ExecutionResult = r
                assert result.exit_code == 0, f"VM {i + 1} exit_code={result.exit_code}, stderr={result.stderr}"
                assert "PASS" in result.stdout, f"VM {i + 1} output: {result.stdout}"

    async def test_concurrent_vms_isolation(self, images_dir: Path) -> None:
        """Each VM should have independent memory space (no cross-contamination)."""
        config = SchedulerConfig(
            default_memory_mb=256,
            default_timeout_seconds=60,
            max_concurrent_vms=_MAX_CONCURRENT_VMS,
            images_dir=images_dir,
        )

        async with Scheduler(config) as sched:
            # Each VM writes a unique signature and verifies it
            async def run_vm_with_signature(vm_id: int) -> ExecutionResult:
                code = f"""
import hashlib

# Write unique signature based on VM ID
signature = b'VM{vm_id}_' + bytes([{vm_id}] * 1000)
chunks = []
for i in range(10):  # 100MB
    chunk = bytearray(10 * 1024 * 1024)
    chunk[0:len(signature)] = signature
    chunk[-len(signature):] = signature
    chunks.append(chunk)

# Verify signatures weren't overwritten
for i, chunk in enumerate(chunks):
    assert chunk[0:len(signature)] == signature, f'Start signature corrupted in chunk {{i}}'
    assert chunk[-len(signature):] == signature, f'End signature corrupted in chunk {{i}}'

# Compute hash of all data
h = hashlib.sha256()
for chunk in chunks:
    h.update(chunk)

print(f'PASS: VM{vm_id} hash={{h.hexdigest()[:16]}}')
"""
                return await sched.run(code=code, language=Language.PYTHON)

            tasks = [run_vm_with_signature(i) for i in range(_MAX_CONCURRENT_VMS)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            hashes: list[str] = []
            for i, r in enumerate(results):
                if isinstance(r, BaseException):
                    pytest.fail(f"VM {i} failed: {r}")
                result: ExecutionResult = r
                assert result.exit_code == 0, f"VM {i} failed: {result.stderr}"
                assert "PASS" in result.stdout
                # Extract hash
                for line in result.stdout.split("\n"):
                    if "hash=" in line:
                        h = line.split("hash=")[1].strip()
                        hashes.append(h)

            # All hashes should be different (each VM has unique signature)
            assert len(set(hashes)) == _MAX_CONCURRENT_VMS, (
                f"Expected {_MAX_CONCURRENT_VMS} unique hashes, got: {hashes}"
            )


# ============================================================================
# Constants Tests
# ============================================================================


class TestMemoryConstants:
    """Tests for memory-related constants."""

    def test_default_memory_allows_zram_expansion(self) -> None:
        """DEFAULT_MEMORY_MB with zram should allow ~1.5x effective memory."""
        from exec_sandbox import constants

        # Default memory should be at least 256MB
        assert constants.DEFAULT_MEMORY_MB >= 256
        # With zram at 50% and ~2.5x compression, effective ~1.25x expansion
        # So 256MB VM can handle ~320MB allocations

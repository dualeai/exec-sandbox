"""
Memory leak detection tests for sustained VM usage.

These tests verify that host memory returns to baseline after many VM executions,
catching issues like:
- VM references not cleaned up
- Asyncio tasks accumulating
- Cache/registry unbounded growth
- File descriptor leaks

Run with: make test-slow
"""

import asyncio
import gc
import os
from pathlib import Path

import psutil
import pytest

from exec_sandbox import Scheduler, SchedulerConfig
from exec_sandbox.warm_vm_pool import Language

# Use half CPU count for max concurrency - avoids boot timeouts under load
_MAX_CONCURRENT = (os.cpu_count() or 4) // 2 or 1

# Memory growth threshold (MB) - allows for GC jitter, allocator overhead, and initialization costs
# Per-VM overhead varies by architecture:
# - arm64: ~0.25MB/VM (smaller pointers, lighter psutil.Process caching)
# - x64: ~0.53MB/VM (8-byte pointers, larger process structures, heavier QEMU footprint)
# For 50 iterations: arm64 ~12.5MB, x64 ~26.5MB
# For 200 iterations: arm64 ~50MB, x64 ~106MB
# Threshold set to accommodate x64 worst case with headroom for GC timing variance
_LEAK_THRESHOLD_MB = 120

# Peak RAM per VM threshold (MB) - measured ~9.5MB for single VM on arm64
# x64 has ~2MB higher overhead per VM due to architecture differences
_PEAK_RAM_PER_VM_MB = 12

# Code that exercises network stack without external dependencies
_NETWORK_TEST_CODE = """
import socket
# Create socket and resolve DNS (exercises gvproxy network path)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5)
try:
    # Just resolve DNS and attempt connect - don't need success
    s.connect(('1.1.1.1', 53))
    print('OK: connected')
except Exception as e:
    print(f'Connect: {e}')
finally:
    s.close()
"""


@pytest.fixture(params=[50, 200])
def iterations(request: pytest.FixtureRequest) -> int:
    """Parametrized iteration counts for memory leak tests."""
    return request.param


@pytest.mark.slow
async def test_no_memory_leak_without_network(iterations: int, images_dir: Path) -> None:
    """Verify host memory returns to baseline after N VM executions."""
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
        max_concurrent_vms=_MAX_CONCURRENT,
    )

    async with Scheduler(config) as scheduler:
        tasks = [scheduler.run(code="print('ok')", language=Language.PYTHON) for _ in range(iterations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Use BaseException to catch CancelledError (BaseException, not Exception in Python 3.8+)
        successes = sum(1 for r in results if not isinstance(r, BaseException) and r.exit_code == 0)
        assert successes >= iterations * 0.9, f"Only {successes}/{iterations} succeeded"

    gc.collect()
    gc.collect()

    final_rss = process.memory_info().rss
    growth_mb = (final_rss - baseline_rss) / 1024 / 1024

    assert growth_mb < _LEAK_THRESHOLD_MB, (
        f"Memory leak: {growth_mb:.1f}MB growth after {iterations} runs (threshold: {_LEAK_THRESHOLD_MB}MB)"
    )


@pytest.mark.slow
async def test_no_memory_leak_with_network(iterations: int, images_dir: Path) -> None:
    """Verify no leak with network enabled (gvproxy) over N executions."""
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
        max_concurrent_vms=_MAX_CONCURRENT,
        default_timeout_seconds=30,
    )

    async with Scheduler(config) as scheduler:
        tasks = [
            scheduler.run(code=_NETWORK_TEST_CODE, language=Language.PYTHON, allow_network=True)
            for _ in range(iterations)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Use BaseException to catch CancelledError (BaseException, not Exception in Python 3.8+)
        successes = sum(1 for r in results if not isinstance(r, BaseException) and r.exit_code == 0)
        assert successes >= iterations * 0.9, f"Only {successes}/{iterations} succeeded"

    gc.collect()
    gc.collect()

    final_rss = process.memory_info().rss
    growth_mb = (final_rss - baseline_rss) / 1024 / 1024

    assert growth_mb < _LEAK_THRESHOLD_MB, (
        f"Memory leak: {growth_mb:.1f}MB growth after {iterations} network runs (threshold: {_LEAK_THRESHOLD_MB}MB)"
    )


# =============================================================================
# Peak RAM Tests - measure memory overhead per VM during execution
# =============================================================================


@pytest.fixture(params=[4, 8])
def concurrent_vms(request: pytest.FixtureRequest) -> int:
    """Number of concurrent VMs to run for peak RAM measurement."""
    return request.param


@pytest.fixture(params=[False, True], ids=["no_network", "with_network"])
def allow_network(request: pytest.FixtureRequest) -> bool:
    """Whether to enable network access for peak RAM tests."""
    return request.param


class PeakMemoryTracker:
    """Track peak RSS memory during async operations."""

    def __init__(self, sample_interval: float = 0.05):
        self.process = psutil.Process()
        self.sample_interval = sample_interval
        self.peak_rss = 0
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def _monitor(self) -> None:
        while self._running:
            current_rss = self.process.memory_info().rss
            self.peak_rss = max(self.peak_rss, current_rss)
            await asyncio.sleep(self.sample_interval)

    def start(self, baseline_rss: int) -> None:
        self.peak_rss = baseline_rss
        self._running = True
        self._task = asyncio.create_task(self._monitor())

    async def stop(self) -> int:
        self._running = False
        if self._task:
            await self._task
        return self.peak_rss


# =============================================================================
# File I/O Memory Tests - detect leaks in write_file/read_file paths
# =============================================================================

# File sizes (MB) shared across all file I/O memory tests
_FILE_IO_SIZES_MB = [1, 5, 10]

# Overhead ratios — thresholds = file_size_mb x ratio.
# Start at the tightest defensible values; raise only with CI profiling data.
#
# Peak: write path holds raw(1x) + base64(1.33x) + JSON frame(1.37x) = 3.7x
# simultaneously (qemu_vm.py:566-577).  Read path is lighter (~2.3x).
_FILE_IO_PEAK_RATIO = 3
#
# Leak: after teardown + gc.collect(), all buffers must be freed.
# Growth should be zero; ratio of 1 gives a noise ceiling equal to one file.
_FILE_IO_LEAK_RATIO = 1


@pytest.fixture(params=_FILE_IO_SIZES_MB, ids=[f"{s}MB" for s in _FILE_IO_SIZES_MB])
def file_io_size_mb(request: pytest.FixtureRequest) -> int:
    """File size in MB, shared across all file I/O memory tests."""
    return request.param


async def _file_io_cycle(scheduler: Scheduler, index: int, size_bytes: int) -> None:
    """Write and read back a file in a fresh session."""
    content = os.urandom(size_bytes)
    async with await scheduler.session(language=Language.PYTHON) as session:
        await session.write_file(f"leak_test_{index}.bin", content)
        result = await session.read_file(f"leak_test_{index}.bin")
        assert len(result) == len(content)


@pytest.mark.slow
async def test_no_memory_leak_file_io(iterations: int, file_io_size_mb: int, images_dir: Path) -> None:
    """Verify host memory returns to baseline after repeated file I/O cycles.

    Each iteration opens a fresh session, writes a file, reads it back,
    and closes the session.  Threshold reuses _LEAK_THRESHOLD_MB because
    VM lifecycle overhead dominates (same as existing non-file-IO tests).
    """
    size_bytes = file_io_size_mb * 1024 * 1024
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
        max_concurrent_vms=_MAX_CONCURRENT,
    )

    async with Scheduler(config) as scheduler:
        sem = asyncio.Semaphore(_MAX_CONCURRENT)

        async def guarded(i: int) -> None:
            async with sem:
                await _file_io_cycle(scheduler, i, size_bytes)

        results = await asyncio.gather(
            *(guarded(i) for i in range(iterations)),
            return_exceptions=True,
        )

        successes = sum(1 for r in results if not isinstance(r, BaseException))
        assert successes >= iterations * 0.9, f"Only {successes}/{iterations} succeeded"

    gc.collect()
    gc.collect()

    final_rss = process.memory_info().rss
    growth_mb = (final_rss - baseline_rss) / 1024 / 1024

    assert growth_mb < _LEAK_THRESHOLD_MB, (
        f"Memory leak: {growth_mb:.1f}MB growth after {iterations} x {file_io_size_mb}MB "
        f"file I/O cycles (threshold: {_LEAK_THRESHOLD_MB}MB)"
    )


@pytest.mark.slow
async def test_no_memory_leak_file_io_in_session(file_io_size_mb: int, images_dir: Path) -> None:
    """Verify no buffer accumulation across repeated file I/O within one session.

    Single VM, 50 write+read cycles.  Catches asyncio StreamReader buffer
    inflation, Pydantic model caching, and unreleased base64 temporaries.
    Threshold: file_size x _FILE_IO_LEAK_RATIO (tight — only buffer leaks).
    """
    size_bytes = file_io_size_mb * 1024 * 1024
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
        max_concurrent_vms=1,
    )

    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            for i in range(50):
                content = os.urandom(size_bytes)
                await session.write_file(f"intra_test_{i}.bin", content)
                result = await session.read_file(f"intra_test_{i}.bin")
                assert len(result) == len(content)

    gc.collect()
    gc.collect()

    final_rss = process.memory_info().rss
    growth_mb = (final_rss - baseline_rss) / 1024 / 1024

    threshold_mb = file_io_size_mb * _FILE_IO_LEAK_RATIO
    assert growth_mb < threshold_mb, (
        f"Memory leak: {growth_mb:.1f}MB growth after 50 x {file_io_size_mb}MB "
        f"intra-session file I/O (threshold: {threshold_mb}MB = "
        f"{file_io_size_mb}MB x {_FILE_IO_LEAK_RATIO})"
    )


# =============================================================================
# Peak RAM Tests - measure memory overhead per VM during execution
# =============================================================================


@pytest.mark.slow
async def test_peak_ram_file_io(file_io_size_mb: int, images_dir: Path) -> None:
    """Measure peak RSS during a single write+read cycle.

    Write holds raw + base64 + JSON frame simultaneously (~3.7x).
    Threshold: file_size x _FILE_IO_PEAK_RATIO.
    """
    size_bytes = file_io_size_mb * 1024 * 1024
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
        max_concurrent_vms=1,
    )

    tracker = PeakMemoryTracker()

    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            tracker.start(baseline_rss)
            content = os.urandom(size_bytes)
            await session.write_file("peak_test.bin", content)
            result = await session.read_file("peak_test.bin")
            peak_rss = await tracker.stop()
            assert len(result) == len(content)

    peak_growth_mb = (peak_rss - baseline_rss) / 1024 / 1024

    threshold_mb = file_io_size_mb * _FILE_IO_PEAK_RATIO
    assert peak_growth_mb < threshold_mb, (
        f"Peak RAM too high: {peak_growth_mb:.1f}MB for {file_io_size_mb}MB file I/O "
        f"(threshold: {threshold_mb}MB = {file_io_size_mb}MB x {_FILE_IO_PEAK_RATIO})"
    )


@pytest.mark.slow
async def test_peak_ram_per_vm(concurrent_vms: int, allow_network: bool, images_dir: Path) -> None:
    """Measure peak RAM overhead per concurrent VM execution."""
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
        max_concurrent_vms=concurrent_vms,
        default_timeout_seconds=30 if allow_network else 10,
    )

    tracker = PeakMemoryTracker()
    code = _NETWORK_TEST_CODE if allow_network else "print('ok')"

    async with Scheduler(config) as scheduler:
        tracker.start(baseline_rss)

        tasks = [
            scheduler.run(code=code, language=Language.PYTHON, allow_network=allow_network)
            for _ in range(concurrent_vms)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        peak_rss = await tracker.stop()

        # Use BaseException to catch CancelledError (BaseException, not Exception in Python 3.8+)
        successes = sum(1 for r in results if not isinstance(r, BaseException) and r.exit_code == 0)
        assert successes >= concurrent_vms * 0.9, f"Only {successes}/{concurrent_vms} succeeded"

    peak_growth_mb = (peak_rss - baseline_rss) / 1024 / 1024
    per_vm_mb = peak_growth_mb / concurrent_vms
    network_label = " with network" if allow_network else ""

    assert per_vm_mb < _PEAK_RAM_PER_VM_MB, (
        f"Peak RAM too high: {per_vm_mb:.1f}MB/VM for {concurrent_vms} VMs{network_label} "
        f"(total: {peak_growth_mb:.1f}MB, threshold: {_PEAK_RAM_PER_VM_MB}MB/VM)"
    )

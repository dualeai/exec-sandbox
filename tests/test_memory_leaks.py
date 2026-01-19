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
_LEAK_THRESHOLD_MB = 15

# Peak RAM per VM threshold (MB) - measured ~9.5MB for single VM, includes fixed overhead
_PEAK_RAM_PER_VM_MB = 10

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

        successes = sum(1 for r in results if not isinstance(r, Exception) and r.exit_code == 0)
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

        successes = sum(1 for r in results if not isinstance(r, Exception) and r.exit_code == 0)
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

        successes = sum(1 for r in results if not isinstance(r, Exception) and r.exit_code == 0)
        assert successes >= concurrent_vms * 0.9, f"Only {successes}/{concurrent_vms} succeeded"

    peak_growth_mb = (peak_rss - baseline_rss) / 1024 / 1024
    per_vm_mb = peak_growth_mb / concurrent_vms
    network_label = " with network" if allow_network else ""

    assert per_vm_mb < _PEAK_RAM_PER_VM_MB, (
        f"Peak RAM too high: {per_vm_mb:.1f}MB/VM for {concurrent_vms} VMs{network_label} "
        f"(total: {peak_growth_mb:.1f}MB, threshold: {_PEAK_RAM_PER_VM_MB}MB/VM)"
    )

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
import tracemalloc
from pathlib import Path

import psutil
import pytest

from exec_sandbox import Scheduler, SchedulerConfig, constants
from exec_sandbox.exceptions import OutputLimitError
from exec_sandbox.guest_channel import OP_QUEUE_DEPTH
from exec_sandbox.warm_vm_pool import Language
from tests.conftest import skip_unless_hwaccel

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


@skip_unless_hwaccel
@pytest.mark.slow
async def test_no_memory_leak_without_network(iterations: int, images_dir: Path) -> None:
    """Verify host memory returns to baseline after N VM executions."""
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
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


@skip_unless_hwaccel
@pytest.mark.slow
async def test_no_memory_leak_with_network(iterations: int, images_dir: Path) -> None:
    """Verify no leak with network enabled (gvproxy) over N executions."""
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
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

# --- Streaming pipeline overhead thresholds ---
#
# Both write and read paths use bounded queues (maxsize=OP_QUEUE_DEPTH).
# Per queued item: FILE_TRANSFER_CHUNK_SIZE bytes compressed → base64 (4/3 expansion).
# Queue capacity dominates; +2MB covers decode/decompress temporaries + StreamReader.
# Memory is O(queue_capacity) — bounded, must NOT scale with file size.
_QUEUE_CAPACITY_MB = OP_QUEUE_DEPTH * constants.FILE_TRANSFER_CHUNK_SIZE * 4 / 3 / (1024 * 1024)
_PIPELINE_JITTER_MB = 2
# Write path: look-ahead pipelining fills the bounded write queue before
# backpressure kicks in.
_FILE_IO_TRACEMALLOC_WRITE_OVERHEAD_MB = _QUEUE_CAPACITY_MB + _PIPELINE_JITTER_MB
# Tracemalloc overhead for bytes-input write: BytesIO copy is tracked at ~file_size,
# plus ~3MB from in-flight queue frames and compressor state.
_FILE_IO_TRACEMALLOC_BYTES_OVERHEAD_MB = 3
# Read path: op_queue (maxsize=OP_QUEUE_DEPTH) buffers FileChunkResponseMessage
# models while consumer awaits disk I/O per chunk.
_FILE_IO_TRACEMALLOC_READ_OVERHEAD_MB = _QUEUE_CAPACITY_MB + _PIPELINE_JITTER_MB
# Pipeline overhead for psutil RSS tests (includes C allocators, page cache,
# write queue fill during sustained transfers, and free-threaded Python overhead).
_FILE_IO_RSS_OVERHEAD_MB = 14
# Streaming chunk pipeline headroom for RSS tests (~500KB real, padded for jitter).
_FILE_IO_RSS_STREAMING_OVERHEAD_MB = 2
# Intra-session accumulation ceiling (tracemalloc current, NOT RSS).
# RSS is unreliable here: allocators never return pages to the OS after
# many large alloc/free cycles (allocator hysteresis), inflating RSS by
# the peak allocation high-water mark (BytesIO copy + queue fill ≈ 22 MB
# for 10 MB files).  tracemalloc.get_traced_memory()[0] measures live
# Python objects only — immune to OS allocator behavior.
_FILE_IO_INTRA_SESSION_LEAK_MB = 5

# --- Output cap overhead thresholds ---
#
# Guest enforces stdout cap at 1 MB (MAX_STDOUT_BYTES).  On exceed the guest
# stops transmitting chunks and raises output_limit_error.  Host-side peak
# allocation is: ~1 MB of accumulated stdout_chunks + per-message Pydantic
# parsing + channel read buffers.  The threshold must be FLAT — independent
# of how much the guest process actually wrote.
_OUTPUT_CAP_TRACEMALLOC_OVERHEAD_MB = 5


@pytest.fixture(params=_FILE_IO_SIZES_MB, ids=[f"{s}MB" for s in _FILE_IO_SIZES_MB])
def file_io_size_mb(request: pytest.FixtureRequest) -> int:
    """File size in MB, shared across all file I/O memory tests."""
    return request.param


async def _file_io_cycle(scheduler: Scheduler, index: int, size_bytes: int, tmp_dir: Path) -> None:
    """Write and read back a file in a fresh session."""
    content = os.urandom(size_bytes)
    dest = tmp_dir / f"leak_test_{index}.bin"
    async with await scheduler.session(language=Language.PYTHON) as session:
        await session.write_file(f"leak_test_{index}.bin", content)
        await session.read_file(f"leak_test_{index}.bin", destination=dest)
        assert dest.stat().st_size == len(content)
    dest.unlink(missing_ok=True)


@pytest.mark.slow
async def test_no_memory_leak_file_io(iterations: int, file_io_size_mb: int, images_dir: Path, tmp_path: Path) -> None:
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
    )

    async with Scheduler(config) as scheduler:
        sem = asyncio.Semaphore(_MAX_CONCURRENT)

        async def guarded(i: int) -> None:
            async with sem:
                await _file_io_cycle(scheduler, i, size_bytes, tmp_path)

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
async def test_no_memory_leak_file_io_in_session(file_io_size_mb: int, images_dir: Path, tmp_path: Path) -> None:
    """Verify no buffer accumulation across repeated file I/O within one session.

    Single VM, 50 write+read cycles.  Catches asyncio StreamReader buffer
    inflation, Pydantic model caching, and unreleased base64 temporaries.

    Uses tracemalloc (current allocation, not peak) instead of psutil RSS.
    RSS is unreliable here: OS allocators keep the high-water mark after
    many large alloc/free cycles (allocator hysteresis), inflating the
    measurement by the peak allocation (~BytesIO copy + queue fill) even
    though all objects are freed.  tracemalloc measures live Python objects
    only, making the test deterministic across platforms and allocators.

    First iteration warms up caches and JIT; measurement starts after.
    """
    size_bytes = file_io_size_mb * 1024 * 1024

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            # Warm-up iteration: populates Pydantic caches, zstd contexts, etc.
            warmup_content = os.urandom(size_bytes)
            warmup_dest = tmp_path / "warmup.bin"
            await session.write_file("intra_test.bin", warmup_content)
            await session.read_file("intra_test.bin", destination=warmup_dest)
            warmup_dest.unlink(missing_ok=True)
            del warmup_content

            gc.collect()
            tracemalloc.start()
            baseline_traced = tracemalloc.get_traced_memory()[0]

            try:
                for i in range(50):
                    content = os.urandom(size_bytes)
                    dest = tmp_path / f"intra_test_{i}.bin"
                    # Reuse same guest path to avoid filling disk (50 x 10MB = 500MB)
                    await session.write_file("intra_test.bin", content)
                    await session.read_file("intra_test.bin", destination=dest)
                    assert dest.stat().st_size == len(content)
                    dest.unlink(missing_ok=True)
                    del content
                    gc.collect()

                final_traced = tracemalloc.get_traced_memory()[0]
            finally:
                tracemalloc.stop()

    growth_mb = (final_traced - baseline_traced) / 1024 / 1024

    assert growth_mb < _FILE_IO_INTRA_SESSION_LEAK_MB, (
        f"Memory leak: {growth_mb:.1f}MB tracemalloc growth after 50 x {file_io_size_mb}MB "
        f"intra-session file I/O (threshold: {_FILE_IO_INTRA_SESSION_LEAK_MB}MB — "
        f"streaming buffers are O(chunk_size), growth should be near zero)"
    )


# =============================================================================
# Peak RAM Tests - measure memory overhead per VM during execution
# =============================================================================


@pytest.mark.slow
async def test_peak_ram_file_io(file_io_size_mb: int, images_dir: Path, tmp_path: Path) -> None:
    """Measure peak RSS during a single write+read cycle.

    os.urandom(size_bytes) is called AFTER baseline, so content allocation is
    counted in RSS growth (~1x file_size).  Chunk pipeline is O(chunk_size)
    with bounded queues.  Threshold: file_size + flat pipeline overhead.
    """
    size_bytes = file_io_size_mb * 1024 * 1024
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    tracker = PeakMemoryTracker()
    dest = tmp_path / "peak_test.bin"

    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            tracker.start(baseline_rss)
            content = os.urandom(size_bytes)
            await session.write_file("peak_test.bin", content)
            await session.read_file("peak_test.bin", destination=dest)
            peak_rss = await tracker.stop()
            assert dest.stat().st_size == len(content)

    peak_growth_mb = (peak_rss - baseline_rss) / 1024 / 1024

    threshold_mb = file_io_size_mb + _FILE_IO_RSS_OVERHEAD_MB
    assert peak_growth_mb < threshold_mb, (
        f"Peak RAM too high: {peak_growth_mb:.1f}MB for {file_io_size_mb}MB file I/O "
        f"(threshold: {threshold_mb}MB = {file_io_size_mb}MB BytesIO + "
        f"{_FILE_IO_RSS_OVERHEAD_MB}MB pipeline overhead)"
    )


@skip_unless_hwaccel
@pytest.mark.slow
async def test_peak_ram_per_vm(concurrent_vms: int, allow_network: bool, images_dir: Path) -> None:
    """Measure peak RAM overhead per concurrent VM execution."""
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
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


# =============================================================================
# Peak RAM Tests - read-only path
# =============================================================================


@pytest.mark.slow
async def test_peak_ram_read_file(file_io_size_mb: int, images_dir: Path, tmp_path: Path) -> None:
    """Measure peak RSS during a read_file-only cycle (separate from write+read).

    Baseline is taken AFTER write_file completes, so content + BytesIO are
    already accounted for.  Bounded op_queue (maxsize=4) and StreamReader
    buffer (512KB) keep read-path memory at O(chunk_size).
    Threshold: 5 MB flat (chunk pipeline + GC/allocator jitter).
    Must NOT scale with file size.
    """
    size_bytes = file_io_size_mb * 1024 * 1024
    process = psutil.Process()

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    dest = tmp_path / "peak_read_test.bin"
    content = os.urandom(size_bytes)

    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            # Write first (not measured)
            await session.write_file("peak_read_test.bin", content)

            gc.collect()
            baseline_rss = process.memory_info().rss

            # Measure peak during read only
            tracker = PeakMemoryTracker()
            tracker.start(baseline_rss)
            await session.read_file("peak_read_test.bin", destination=dest)
            peak_rss = await tracker.stop()

            assert dest.stat().st_size == len(content)

    peak_growth_mb = (peak_rss - baseline_rss) / 1024 / 1024

    assert peak_growth_mb < _FILE_IO_RSS_OVERHEAD_MB, (
        f"Peak RAM too high during read: {peak_growth_mb:.1f}MB for {file_io_size_mb}MB file "
        f"(flat threshold: {_FILE_IO_RSS_OVERHEAD_MB}MB — streaming is O(chunk_size), not O(file_size))"
    )


# =============================================================================
# Concurrent Transfer Memory Tests
# =============================================================================


@pytest.fixture(params=[3, 5])
def concurrent_sessions(request: pytest.FixtureRequest) -> int:
    """Number of concurrent sessions for concurrent file I/O tests."""
    return request.param


@pytest.mark.slow
async def test_peak_ram_concurrent_file_io(
    concurrent_sessions: int, file_io_size_mb: int, images_dir: Path, tmp_path: Path
) -> None:
    """Measure peak RSS with N sessions each doing write+read concurrently.

    Uses Path input so _resolve_content opens a file handle (no BytesIO copy).
    Source files are created BEFORE baseline measurement, so os.urandom bytes
    are freed and don't pollute the tracked region.

    Per-session peak during transfer:
      - File handle from _resolve_content: ~0 bytes (not a copy)
      - Chunk pipeline (128 KB read + compress + base64 + JSON): ~500 KB
      - VM/channel overhead: ~_PEAK_RAM_PER_VM_MB
    Threshold: N * (per_vm_overhead + 1 MB streaming headroom).
    Flat per file size — proves streaming is O(chunk_size), not O(file_size).
    """
    size_bytes = file_io_size_mb * 1024 * 1024

    # Create source files BEFORE baseline — os.urandom bytes freed after write_bytes
    sources: dict[int, Path] = {}
    for i in range(concurrent_sessions):
        src = tmp_path / f"concurrent_src_{i}.bin"
        src.write_bytes(os.urandom(size_bytes))
        sources[i] = src

    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    async def session_cycle(scheduler: Scheduler, idx: int) -> None:
        src = sources[idx]
        dest = tmp_path / f"concurrent_{idx}.bin"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file(f"concurrent_{idx}.bin", src)
            await session.read_file(f"concurrent_{idx}.bin", destination=dest)
            assert dest.stat().st_size == size_bytes
        dest.unlink(missing_ok=True)

    tracker = PeakMemoryTracker()

    async with Scheduler(config) as scheduler:
        tracker.start(baseline_rss)
        results = await asyncio.gather(
            *(session_cycle(scheduler, i) for i in range(concurrent_sessions)),
            return_exceptions=True,
        )
        peak_rss = await tracker.stop()

        successes = sum(1 for r in results if not isinstance(r, BaseException))
        assert successes >= concurrent_sessions * 0.9, f"Only {successes}/{concurrent_sessions} succeeded"

    peak_growth_mb = (peak_rss - baseline_rss) / 1024 / 1024

    # Flat per session: VM overhead + streaming pipeline (~500 KB)
    threshold_mb = concurrent_sessions * (_PEAK_RAM_PER_VM_MB + _FILE_IO_RSS_STREAMING_OVERHEAD_MB)
    assert peak_growth_mb < threshold_mb, (
        f"Peak RAM too high: {peak_growth_mb:.1f}MB for {concurrent_sessions} sessions x {file_io_size_mb}MB "
        f"(threshold: {threshold_mb}MB = {concurrent_sessions} x "
        f"({_PEAK_RAM_PER_VM_MB}MB/VM + {_FILE_IO_RSS_STREAMING_OVERHEAD_MB}MB streaming))"
    )


# =============================================================================
# tracemalloc-based Peak Detection Tests
# =============================================================================


@pytest.mark.slow
async def test_tracemalloc_peak_write_file(file_io_size_mb: int, images_dir: Path) -> None:
    """Measure Python-level allocation peak during write_file with bytes input.

    Uses tracemalloc for exact Python allocation tracking (catches sub-ms
    spikes that 50ms psutil sampling misses).  Note: zstd C allocations
    are NOT tracked — this is complementary to psutil RSS tests.

    content (os.urandom) is allocated BEFORE tracemalloc.start(), so NOT tracked.
    Tracked: BytesIO(content) copy in _resolve_content (~1x file_size) + chunk
    pipeline (in-flight queue frames + compressor state).  Threshold: file_size + 3 MB.
    """
    size_bytes = file_io_size_mb * 1024 * 1024

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            content = os.urandom(size_bytes)
            tracemalloc.start()
            try:
                await session.write_file("tracemalloc_write.bin", content)
                _, peak_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

    peak_mb = peak_bytes / 1024 / 1024
    threshold_mb = file_io_size_mb + _FILE_IO_TRACEMALLOC_BYTES_OVERHEAD_MB

    assert peak_mb < threshold_mb, (
        f"tracemalloc peak too high for write_file (bytes): {peak_mb:.1f}MB "
        f"(threshold: {threshold_mb}MB = {file_io_size_mb}MB BytesIO copy + "
        f"{_FILE_IO_TRACEMALLOC_BYTES_OVERHEAD_MB}MB chunk overhead)"
    )


@pytest.mark.slow
async def test_tracemalloc_peak_write_file_from_path(file_io_size_mb: int, images_dir: Path, tmp_path: Path) -> None:
    """Measure Python-level allocation peak during write_file with Path input.

    Path input: _resolve_content opens a file handle, no full-content copy.
    Write queue (maxsize=64) with look-ahead pipelining can hold up to
    ~11MB of in-flight compressed+encoded frames.  Memory is O(queue_capacity),
    bounded but not O(chunk_size).
    """
    size_bytes = file_io_size_mb * 1024 * 1024
    src = tmp_path / "tracemalloc_src.bin"
    src.write_bytes(os.urandom(size_bytes))

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            tracemalloc.start()
            try:
                await session.write_file("tracemalloc_path_write.bin", src)
                _, peak_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

    peak_mb = peak_bytes / 1024 / 1024

    assert peak_mb < _FILE_IO_TRACEMALLOC_WRITE_OVERHEAD_MB, (
        f"tracemalloc peak too high for write_file (Path): {peak_mb:.1f}MB "
        f"(threshold: {_FILE_IO_TRACEMALLOC_WRITE_OVERHEAD_MB}MB — "
        f"write queue capacity bounds in-flight data)"
    )


@pytest.mark.slow
async def test_tracemalloc_peak_read_file(file_io_size_mb: int, images_dir: Path, tmp_path: Path) -> None:
    """Measure Python-level allocation peak during read_file.

    Read loop: op_queue.get() → base64.b64decode → decompress → disk write,
    per 128 KB chunk.  Bounded op_queue (maxsize=OP_QUEUE_DEPTH) caps in-flight
    data at O(queue_capacity).
    Threshold derived from OP_QUEUE_DEPTH x chunk wire size + pipeline jitter.
    Must NOT scale with file size — proves streaming is bounded.
    """
    size_bytes = file_io_size_mb * 1024 * 1024
    content = os.urandom(size_bytes)
    dest = tmp_path / "tracemalloc_read.bin"

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("tracemalloc_read.bin", content)

            tracemalloc.start()
            try:
                await session.read_file("tracemalloc_read.bin", destination=dest)
                _, peak_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

            assert dest.stat().st_size == len(content)

    peak_mb = peak_bytes / 1024 / 1024

    assert peak_mb < _FILE_IO_TRACEMALLOC_READ_OVERHEAD_MB, (
        f"tracemalloc peak too high for read_file: {peak_mb:.1f}MB "
        f"(flat threshold: {_FILE_IO_TRACEMALLOC_READ_OVERHEAD_MB}MB — "
        f"streaming is O(chunk_size), not O(file_size))"
    )


# =============================================================================
# Large File Peak RAM Tests
# =============================================================================


@skip_unless_hwaccel
@pytest.mark.slow
@pytest.mark.parametrize("large_file_mb", [30, 50], ids=["30MB", "50MB"])
async def test_peak_ram_large_file_io(large_file_mb: int, images_dir: Path, tmp_path: Path) -> None:
    """Measure peak RSS for large files — threshold is FLAT, not proportional.

    Uses Path input so _resolve_content opens a file handle (no copy).
    Streaming processes 128 KB chunks — ~500 KB alive at any point.
    Baseline is taken before Scheduler creation, so growth includes VM
    lifecycle overhead (~_PEAK_RAM_PER_VM_MB) + streaming (~500 KB).
    Flat threshold proves streaming is O(chunk_size), not O(file_size):
    a 30 MB and 50 MB file must fit under the SAME threshold.
    """
    size_bytes = large_file_mb * 1024 * 1024
    src = tmp_path / "large_src.bin"
    src.write_bytes(os.urandom(size_bytes))
    dest = tmp_path / "large_dest.bin"

    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    tracker = PeakMemoryTracker()

    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            tracker.start(baseline_rss)
            await session.write_file("large_file.bin", src)
            await session.read_file("large_file.bin", destination=dest)
            peak_rss = await tracker.stop()

            assert dest.stat().st_size == size_bytes

    peak_growth_mb = (peak_rss - baseline_rss) / 1024 / 1024

    # Flat threshold: VM lifecycle + full pipeline RSS overhead.
    # During sustained large-file transfers the write queue (64 x ~175 KB)
    # fills up, and C allocators / kernel buffers add further overhead.
    # _FILE_IO_RSS_OVERHEAD_MB captures this (vs the minimal per-session
    # _FILE_IO_RSS_STREAMING_OVERHEAD_MB used in the concurrent test).
    threshold_mb = _PEAK_RAM_PER_VM_MB + _FILE_IO_RSS_OVERHEAD_MB
    assert peak_growth_mb < threshold_mb, (
        f"Peak RAM too high for {large_file_mb}MB file: {peak_growth_mb:.1f}MB "
        f"(flat threshold: {threshold_mb}MB = {_PEAK_RAM_PER_VM_MB}MB VM + "
        f"{_FILE_IO_RSS_OVERHEAD_MB}MB pipeline overhead). "
        f"Streaming is O(chunk_size), not O(file_size)."
    )


# =============================================================================
# bytes vs Path Input Memory Comparison
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("comparison_size_mb", [5, 10], ids=["5MB", "10MB"])
async def test_bytes_vs_path_memory_difference(comparison_size_mb: int, images_dir: Path, tmp_path: Path) -> None:
    """Compare tracemalloc peaks for bytes vs Path write_file inputs.

    Both paths share the same bounded chunk pipeline (write queue maxsize=64).
    Bytes input: BytesIO copy (~file_size) + in-flight queue frames.
    Path input: file handle (no copy) + in-flight queue frames.
    Both bounded by write queue capacity (~11MB).
    """
    size_bytes = comparison_size_mb * 1024 * 1024
    src = tmp_path / "compare_src.bin"
    src.write_bytes(os.urandom(size_bytes))

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    # Measure bytes input peak
    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            content = src.read_bytes()
            tracemalloc.start()
            try:
                await session.write_file("compare_bytes.bin", content)
                _, bytes_peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            del content

    # Measure Path input peak
    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            tracemalloc.start()
            try:
                await session.write_file("compare_path.bin", src)
                _, path_peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

    bytes_peak_mb = bytes_peak / 1024 / 1024
    path_peak_mb = path_peak / 1024 / 1024

    bytes_threshold_mb = comparison_size_mb + _FILE_IO_TRACEMALLOC_BYTES_OVERHEAD_MB
    assert bytes_peak_mb < bytes_threshold_mb, (
        f"bytes input peak ({bytes_peak_mb:.1f}MB) should be < {bytes_threshold_mb}MB — "
        f"BytesIO copy ({comparison_size_mb}MB) + {_FILE_IO_TRACEMALLOC_BYTES_OVERHEAD_MB}MB pipeline overhead"
    )
    assert path_peak_mb < _FILE_IO_TRACEMALLOC_WRITE_OVERHEAD_MB, (
        f"Path input peak ({path_peak_mb:.1f}MB) should be < {_FILE_IO_TRACEMALLOC_WRITE_OVERHEAD_MB}MB — "
        f"bounded by write queue capacity"
    )


# =============================================================================
# Output Cap Memory Tests — verify guest-side cap bounds host memory
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("output_mb", [2, 100], ids=["2MB", "100MB"])
async def test_output_cap_bounds_host_memory(output_mb: int, images_dir: Path) -> None:
    """Guest-side output cap prevents host memory from scaling with output size.

    Generates output_mb of stdout in the guest.  The guest agent caps at 1 MB
    (MAX_STDOUT_BYTES) and raises output_limit_error.  Host-side tracemalloc
    peak must be FLAT regardless of output_mb — a 2 MB and 100 MB output must
    fit under the SAME threshold.  This proves the cap is enforced in the guest
    and the host never buffers unbounded output.

    Uses tracemalloc for precise Python heap measurement (immune to allocator
    hysteresis).  A warm-up exec populates Pydantic caches and channel state
    so only the capped execution's allocations are measured.
    """
    target_bytes = output_mb * 1024 * 1024

    # Write in 64KB chunks to avoid a single huge Python string in the guest
    code = f"""
import sys
chunk = "X" * 65536
remaining = {target_bytes}
while remaining > 0:
    n = min(65536, remaining)
    sys.stdout.write(chunk[:n])
    remaining -= n
sys.stdout.flush()
"""

    config = SchedulerConfig(
        images_dir=images_dir,
        warm_pool_size=0,
    )

    async with Scheduler(config) as scheduler:
        async with await scheduler.session(language=Language.PYTHON) as session:
            # Warm-up: populates Pydantic caches, channel state, etc.
            await session.exec('print("warmup")')

            gc.collect()
            tracemalloc.start()
            try:
                with pytest.raises(OutputLimitError):
                    await session.exec(code, timeout_seconds=300)
                _, peak_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

    peak_mb = peak_bytes / 1024 / 1024

    assert peak_mb < _OUTPUT_CAP_TRACEMALLOC_OVERHEAD_MB, (
        f"Output cap not enforced on host: {peak_mb:.1f}MB tracemalloc peak "
        f"for {output_mb}MB guest output (flat threshold: "
        f"{_OUTPUT_CAP_TRACEMALLOC_OVERHEAD_MB}MB — guest caps stdout at 1 MB, "
        f"host should never see more)"
    )

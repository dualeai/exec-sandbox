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
import contextlib
import gc
import io
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any, Self

import psutil
import pytest

from exec_sandbox import Scheduler, SchedulerConfig
from exec_sandbox.exceptions import ErrorCode, InvalidInputError
from exec_sandbox.models import Language
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once, skip_unless_hwaccel

# Use half CPU count for max concurrency - avoids boot timeouts under load
_MAX_CONCURRENT = (os.cpu_count() or 4) // 2 or 1

# Memory growth threshold (MB) - allows for GC jitter, allocator overhead, and initialization costs
# Per-VM overhead varies by architecture and runtime:
# - arm64: ~0.25MB/VM (smaller pointers, lighter psutil.Process caching)
# - x64: ~0.53MB/VM (8-byte pointers, larger process structures, heavier QEMU footprint)
# - x64 3.14t: ~0.82MB/VM (mimalloc per-thread arenas, biased refcounting overhead)
# For 200 iterations: arm64 ~50MB, x64 ~106MB, x64 3.14t ~164MB
# Threshold covers 3.14t worst case (164MB observed) with headroom for GC timing variance
_LEAK_THRESHOLD_MB = 200

# Peak RAM per VM threshold (MB):
#   arm64/HVF macOS: ~2-4MB/VM
#   x64/KVM Linux:   ~11.5MB/VM
#   x64/KVM 3.14t:   ~17-18MB/VM (mimalloc arenas + biased refcounting)
# 21MB gives ~20% headroom over 3.14t worst case for CI variance.
_PEAK_RAM_PER_VM_MB = 21

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


async def _warmup_and_baseline(config: SchedulerConfig, *, allow_network: bool) -> tuple[psutil.Process, int]:
    """Run a warmup Scheduler cycle (the network code when allow_network), gc.collect, and
    return (process, baseline_rss).

    Absorbs one-time init costs (Scheduler, Pydantic, psutil, mimalloc arenas)
    before taking the RSS baseline.  See test_no_memory_leak_without_network
    docstring for full rationale.
    """
    code = _NETWORK_TEST_CODE if allow_network else "print('warmup')"
    async with Scheduler(config) as scheduler:
        await run_once(scheduler, VmSpec(language=Language.PYTHON, allow_network=allow_network), code)
    process = psutil.Process()
    gc.collect()
    baseline_rss = process.memory_info().rss
    return process, baseline_rss


def _assert_success_rate(results: list[Any]) -> None:
    """Assert >=90% of concurrent results succeeded, logging failures on error.

    A result is an ExecutionResult (a success at exit code 0), None (a success) or the
    exception. Uses int() to floor the threshold so e.g. 4/5 passes (tolerates 1 failure
    under CI load).
    """
    failures = [r for r in results if isinstance(r, BaseException) or (r is not None and r.exit_code != 0)]
    total = len(results)
    successes = total - len(failures)
    min_required = int(total * 0.9)
    assert successes >= min_required, f"Only {successes}/{total} succeeded (need {min_required}): " + "; ".join(
        repr(f) for f in failures[:10]
    )


_ITERATIONS = 200
"""VM executions (or file I/O cycles) of each leak test."""


@skip_unless_hwaccel
@pytest.mark.slow
async def test_no_memory_leak_without_network(scheduler_config: SchedulerConfig) -> None:
    """Verify host memory returns to baseline after N VM executions.

    Requires hwaccel: 90% success threshold — TCG timeouts cause too many
    failures, inflating the failure rate below the threshold.

    A warmup run absorbs one-time initialization costs (Scheduler startup,
    Pydantic model compilation, psutil caching, asyncio thread-pool warmup)
    before taking the RSS baseline.  Without warmup, the first test of a
    process counts all init overhead as growth, while a later one starts from
    a warm process — a false positive on free-threaded Python, where mimalloc
    per-thread arenas inflate the init cost.
    """
    process, baseline_rss = await _warmup_and_baseline(scheduler_config, allow_network=False)

    async with Scheduler(scheduler_config) as scheduler:
        tasks = [run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')") for _ in range(_ITERATIONS)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        _assert_success_rate(results)

    gc.collect()
    gc.collect()

    final_rss = process.memory_info().rss
    growth_mb = (final_rss - baseline_rss) / 1024 / 1024

    assert growth_mb < _LEAK_THRESHOLD_MB, (
        f"Memory leak: {growth_mb:.1f}MB growth after {_ITERATIONS} runs (threshold: {_LEAK_THRESHOLD_MB}MB)"
    )


@skip_unless_hwaccel
@pytest.mark.slow
async def test_no_memory_leak_with_network(scheduler_config: SchedulerConfig) -> None:
    """Verify no leak with network enabled (gvproxy) over N executions.

    Requires hwaccel: same as test_no_memory_leak_without_network, plus
    gvproxy overhead amplifies TCG timeout failures.

    Warmup run absorbs one-time init costs — see test_no_memory_leak_without_network
    docstring for rationale.  The network warmup additionally primes gvproxy binary
    resolution, socket allocation paths, and outbound-allow JSON generation.
    """
    process, baseline_rss = await _warmup_and_baseline(scheduler_config, allow_network=True)

    async with Scheduler(scheduler_config) as scheduler:
        tasks = [
            run_once(scheduler, VmSpec(language=Language.PYTHON, allow_network=True), _NETWORK_TEST_CODE)
            for _ in range(_ITERATIONS)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        _assert_success_rate(results)

    gc.collect()
    gc.collect()

    final_rss = process.memory_info().rss
    growth_mb = (final_rss - baseline_rss) / 1024 / 1024

    assert growth_mb < _LEAK_THRESHOLD_MB, (
        f"Memory leak: {growth_mb:.1f}MB growth after {_ITERATIONS} network runs (threshold: {_LEAK_THRESHOLD_MB}MB)"
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
    """Peak RSS of this process while its `async with` block runs (a sample every 50 ms), from
    `baseline_rss` up. The sampling task ends with the block, also when the block raises."""

    def __init__(self, baseline_rss: int) -> None:
        self.process = psutil.Process()
        self.peak_rss = baseline_rss
        self._task: asyncio.Task[None] | None = None

    async def _monitor(self) -> None:
        while True:
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            await asyncio.sleep(0.05)

    async def __aenter__(self) -> Self:
        self._task = asyncio.create_task(self._monitor())
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        assert self._task is not None
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


# =============================================================================
# File I/O Memory Tests - detect leaks in write_file/read_file paths
# =============================================================================

# File sizes (MB) shared across all file I/O memory tests
_FILE_IO_SIZES_MB = [1, 10]

# --- Write pipeline bound (tracemalloc) ---
#
# write_file() makes one frame per 128 KiB chunk (zstd, then base64 in a JSON
# line) and queues it for the write worker of the channel. The frames that can
# be alive at the same time:
# - the write queue: 64 frames;
# - the transport: the batch of 16 frames that the worker writes before one
#   drain(), plus the partly sent frame of the batch before (drain() returns at
#   the low-water mark, and the transport keeps that frame whole);
# - the producer: the frame it builds, with the raw chunk, the zstd output and
#   four copies of the frame (base64 str, JSON str, encoded bytes, + newline).
# The bound does not grow with the file size. The guest sets where the peak
# falls under it: a slow drain (TCG) brings it close, a fast one keeps it
# lower. Bytes input is not copied (BytesIO shares the buffer of a bytes
# object), so bytes and Path input have the same bound.
#
# The bounds are numbers, computed by hand from the design values above: a
# bound made from the code's constants grows with them, so a larger queue
# would pass. The numbers:
# - largest zstd output of a 131,072 B chunk (ZSTD_COMPRESSBOUND):
#   131,072 + 131,072 / 256 = 131,584 B;
# - largest frame: base64 of it, 4 * ceil(131,584 / 3) = 175,448 B, plus the
#   JSON envelope {"action": "file_chunk", "op_id": <32 chars>, "data": ""} and
#   its newline, 82 B: 175,530 B;
# - the frame being built: 131,072 + 131,584 + 4 * 175,530 = 964,776 B;
# - 81 frames in flight (64 queued + 16 + 1 in the transport):
#   81 * 175,530 + 964,776 = 15,182,706 B = 14.48 MiB, plus 0.5 MiB of fixed
#   objects (request models, futures, executor work items): 14.98 MiB.
_WRITE_PIPELINE_PEAK_MB = 15.0
_WRITE_BOUND_TEXT = "81 frames in flight of 175,530 B, one frame being built (964,776 B), 0.5 MB fixed"

# The encoded write frames of one write that can be alive at once (MB), by file
# size: the whole file when it fits in the pipeline, plus the frame being built.
# 1 MiB: 8 frames, 8 * 175,530 + 964,776 B = 2.26 MiB; 10 MiB: 80 frames, 14.31 MiB.
_WRITE_FRAMES_IN_FLIGHT_MB = {1: 2.3, 10: 14.4}


# File sizes of the tracemalloc peak tests. 32 MiB is above the capacity of each
# pipeline (about 10 MiB of chunks for a write, 8 MiB for a read): a peak that grows
# with the file goes over the bound.
_PEAK_SIZES_MB = [1, 10, 32]

# --- Read pipeline bound (tracemalloc) ---
#
# read_file(): the operation queue (64 messages) holds the chunk messages of
# the guest while the consumer writes each chunk to its destination. Pydantic
# decodes the base64 at the parse: a queued message holds one read of the
# guest's zstd stream into its 128 KiB buffer, at most 131,072 B (random bytes
# come out as raw blocks, a full read then the 3 B rest of the block: the test
# content, over 225 values, makes each read one whole block of ~127 KB). A full
# queue stops the dispatcher, and the StreamReader of the port socket fills:
# asyncio pauses the transport after the read that takes the buffer over twice
# its 512 KiB limit, and one read takes up to 262,144 B. The numbers:
# - the queue: 64 * 131,072 = 8,388,608 B;
# - the dispatcher: the 65th message, waiting for a slot, and its line (base64
#   of 131,072 B and the JSON envelope, 174,839 B): 305,911 B;
# - the consumer: the message in hand and its output (one 128 KiB input piece
#   of the guest): 262,144 B;
# - the StreamReader: 2 * 524,288 + 262,144 = 1,310,720 B, allocated with 1/8
#   more and 6 B (bytearray growth): 1,474,566 B. Reading stops past twice the
#   limit and starts again only at the limit or under, after a del that shrank
#   the block to its size: the old block of a growth that copies (after
#   readuntil() took a line off the front) grew while reading ran, to at most
#   1,048,576 * 1.125 + 6 = 1,179,654 B. Both blocks and the bytes of the read:
#   2,916,364 B;
# - total 11,873,027 B = 11.32 MiB, plus 0.5 MiB of fixed objects: 11.82 MiB.
# The dispatcher and the consumer are counted at their awaits: the copies that
# readuntil(), the parse and a decompress step make come when a queue slot is
# free or the reader holds one block, and those moments stay under the total.
# Measured at 32 MiB (content over 225 values, both destinations): 10.0-10.7 MiB
# on HVF (3.12, 3.14t), 10.1-10.8 MiB on x86_64 KVM (3.14).
_FILE_IO_TRACEMALLOC_READ_OVERHEAD_MB = 11.9

# The read messages that the operation queue holds for one read_file() (MB), by file size:
# at most 64 queued, each counted at 4/3 of a 128 KiB chunk (its base64 frame), although a
# queued message holds the decoded chunk (at most 131,072 B): _FILE_IO_RSS_FIXED_MB was
# measured with these numbers. 1 MiB: 8 chunks, 1.33 MiB; 10 MiB: 64 of its 80 chunks, 10.67 MiB.
_READ_MESSAGES_QUEUED_MB = {1: 1.4, 10: 10.7}

# RSS of one write then one read beyond their bounded buffers: interpreter objects, allocator
# bookkeeping, the free-threaded build's per-object locks. Measured on x86_64 KVM (20 runs each on
# 3.12 and 3.14t): at most 1.6 MB for a 1 MiB file, 2.7 MB for a 10 MiB file.
_FILE_IO_RSS_FIXED_MB = 5

# Pipeline overhead for psutil RSS tests (includes C allocators, page cache,
# write queue fill during sustained transfers, and free-threaded Python overhead).
# 3.14t adds ~2-4MB residual overhead from per-object locks even after warmup.
# Threshold covers 3.14t worst case with headroom.
_FILE_IO_RSS_OVERHEAD_MB = 24
# Intra-session accumulation ceiling (tracemalloc current, NOT RSS).
# RSS is unreliable here: allocators never return pages to the OS after
# many large alloc/free cycles (allocator hysteresis), inflating RSS by
# the peak allocation high-water mark (the content and the write frames in
# flight: about 25 MB for 10 MB files).  tracemalloc.get_traced_memory()[0]
# measures live Python objects only — immune to OS allocator behavior.
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
    async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
        await session.write_file(f"leak_test_{index}.bin", content)
        await session.read_file(f"leak_test_{index}.bin", destination=dest)
        assert dest.stat().st_size == len(content)
    dest.unlink(missing_ok=True)


@skip_unless_hwaccel
@pytest.mark.slow
async def test_no_memory_leak_file_io(file_io_size_mb: int, scheduler_config: SchedulerConfig, tmp_path: Path) -> None:
    """Verify host memory returns to baseline after repeated file I/O cycles.

    Requires hwaccel: the TCG JIT inflates the RSS of each VM, and 200 cycles
    under TCG on free-threaded Python grew past the threshold.

    Each iteration opens a fresh session, writes a file, reads it back,
    and closes the session.  Threshold reuses _LEAK_THRESHOLD_MB because
    VM lifecycle overhead dominates (same as existing non-file-IO tests).

    Warmup run absorbs one-time init costs — see test_no_memory_leak_without_network
    docstring for rationale.
    """
    size_bytes = file_io_size_mb * 1024 * 1024

    process, baseline_rss = await _warmup_and_baseline(scheduler_config, allow_network=False)

    async with Scheduler(scheduler_config) as scheduler:
        sem = asyncio.Semaphore(_MAX_CONCURRENT)

        async def guarded(i: int) -> None:
            async with sem:
                await _file_io_cycle(scheduler, i, size_bytes, tmp_path)

        results = await asyncio.gather(
            *(guarded(i) for i in range(_ITERATIONS)),
            return_exceptions=True,
        )

        _assert_success_rate(results)

    gc.collect()
    gc.collect()

    final_rss = process.memory_info().rss
    growth_mb = (final_rss - baseline_rss) / 1024 / 1024

    assert growth_mb < _LEAK_THRESHOLD_MB, (
        f"Memory leak: {growth_mb:.1f}MB growth after {_ITERATIONS} x {file_io_size_mb}MB "
        f"file I/O cycles (threshold: {_LEAK_THRESHOLD_MB}MB)"
    )


@pytest.mark.slow
async def test_no_memory_leak_file_io_in_session(
    file_io_size_mb: int, scheduler_config: SchedulerConfig, tmp_path: Path
) -> None:
    """Verify no buffer accumulation across repeated file I/O within one session.

    Single VM, 50 write+read cycles.  Catches asyncio StreamReader buffer
    inflation, Pydantic model caching, and unreleased base64 temporaries.

    Uses tracemalloc (current allocation, not peak) instead of psutil RSS.
    RSS is unreliable here: OS allocators keep the high-water mark after
    many large alloc/free cycles (allocator hysteresis), inflating the
    measurement by the peak allocation (~content + write frames in flight) even
    though all objects are freed.  tracemalloc measures live Python objects
    only, making the test deterministic across platforms and allocators.

    First iteration warms up caches and JIT; measurement starts after.
    """
    size_bytes = file_io_size_mb * 1024 * 1024

    async with Scheduler(scheduler_config) as scheduler:
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
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
async def test_peak_ram_file_io(file_io_size_mb: int, scheduler_config: SchedulerConfig, tmp_path: Path) -> None:
    """Measure peak RSS during a single write+read cycle.

    os.urandom(size_bytes) is called AFTER baseline, so content allocation is
    counted in RSS growth (~1x file_size). Threshold: the content, the write
    frames in flight (_WRITE_FRAMES_IN_FLIGHT_MB), the read messages queued
    (_READ_MESSAGES_QUEUED_MB) and _FILE_IO_RSS_FIXED_MB. Both pipeline peaks
    count: the allocator keeps the pages that the write freed, so the read
    buffers can add on top in RSS. A flat 24 MB beyond the content sat at the
    median for 10 MiB: p50 33.6 and 33.8 MB, max 37.2 and 37.8 MB (3.12 and
    3.14t, x86_64 KVM).

    Warmup run absorbs one-time init costs — see test_no_memory_leak_without_network
    docstring for rationale.
    """
    size_bytes = file_io_size_mb * 1024 * 1024

    _, baseline_rss = await _warmup_and_baseline(scheduler_config, allow_network=False)

    dest = tmp_path / "peak_test.bin"

    async with Scheduler(scheduler_config) as scheduler:
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            async with PeakMemoryTracker(baseline_rss) as tracker:
                content = os.urandom(size_bytes)
                await session.write_file("peak_test.bin", content)
                await session.read_file("peak_test.bin", destination=dest)
            assert dest.stat().st_size == len(content)

    peak_growth_mb = (tracker.peak_rss - baseline_rss) / 1024 / 1024

    write_mb = _WRITE_FRAMES_IN_FLIGHT_MB[file_io_size_mb]
    read_mb = _READ_MESSAGES_QUEUED_MB[file_io_size_mb]
    threshold_mb = file_io_size_mb + write_mb + read_mb + _FILE_IO_RSS_FIXED_MB
    assert peak_growth_mb < threshold_mb, (
        f"Peak RAM too high: {peak_growth_mb:.1f}MB for {file_io_size_mb}MB file I/O "
        f"(threshold: {threshold_mb:.1f}MB = {file_io_size_mb}MB content + {write_mb}MB write frames + "
        f"{read_mb}MB read messages + {_FILE_IO_RSS_FIXED_MB}MB fixed)"
    )


@skip_unless_hwaccel
@pytest.mark.slow
async def test_peak_ram_per_vm(concurrent_vms: int, allow_network: bool, images_dir: Path) -> None:
    """Measure peak RAM overhead per concurrent VM execution.

    Requires hwaccel: RSS thresholds are calibrated for KVM/HVF — TCG JIT
    compilation inflates per-VM RSS, causing false positives.

    Warmup run absorbs one-time init costs and, when allow_network=True,
    also primes the gvproxy/network path — see test_no_memory_leak_without_network
    docstring for rationale.

    Baseline is taken inside the measurement Scheduler (after startup but
    before launching VMs) so Scheduler creation cost isn't counted as per-VM
    growth.  gc.collect() before baseline and before stopping the tracker
    reduces RSS noise from dead objects.
    """
    config = SchedulerConfig(images_dir=images_dir, default_timeout_seconds=30 if allow_network else 10)

    # Warmup absorbs one-time init costs (Pydantic, psutil, mimalloc arenas)
    await _warmup_and_baseline(config, allow_network=allow_network)

    code = _NETWORK_TEST_CODE if allow_network else "print('ok')"
    process = psutil.Process()

    async with Scheduler(config) as scheduler:
        # Baseline inside the Scheduler so startup cost isn't counted as per-VM growth
        gc.collect()
        baseline_rss = process.memory_info().rss
        async with PeakMemoryTracker(baseline_rss) as tracker:
            tasks = [
                run_once(scheduler, VmSpec(language=Language.PYTHON, allow_network=allow_network), code)
                for _ in range(concurrent_vms)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            gc.collect()

        _assert_success_rate(results)

    peak_growth_mb = (tracker.peak_rss - baseline_rss) / 1024 / 1024
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
async def test_peak_ram_read_file(file_io_size_mb: int, scheduler_config: SchedulerConfig, tmp_path: Path) -> None:
    """Measure peak RSS during a read_file-only cycle (separate from write+read).

    Baseline is taken AFTER write_file completes, so the content is already
    accounted for.  Bounded op_queue (64 messages) and
    StreamReader buffer keep read-path memory at O(chunk_size).
    Threshold: _FILE_IO_RSS_OVERHEAD_MB flat (VM lifecycle + queue fill +
    C allocator overhead + 3.14t mimalloc residual).
    Must NOT scale with file size.

    No external warmup Scheduler is needed: the baseline is captured
    mid-session after write_file completes, so all one-time init costs
    (Scheduler, Pydantic, VM boot) are already absorbed.
    """
    size_bytes = file_io_size_mb * 1024 * 1024
    process = psutil.Process()

    dest = tmp_path / "peak_read_test.bin"
    content = os.urandom(size_bytes)

    async with Scheduler(scheduler_config) as scheduler:
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Write first (not measured)
            await session.write_file("peak_read_test.bin", content)

            gc.collect()
            baseline_rss = process.memory_info().rss

            # Measure peak during read only
            async with PeakMemoryTracker(baseline_rss) as tracker:
                await session.read_file("peak_read_test.bin", destination=dest)

            assert dest.stat().st_size == len(content)

    peak_growth_mb = (tracker.peak_rss - baseline_rss) / 1024 / 1024

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


@skip_unless_hwaccel
@pytest.mark.slow
async def test_peak_ram_concurrent_file_io(
    concurrent_sessions: int, file_io_size_mb: int, scheduler_config: SchedulerConfig, tmp_path: Path
) -> None:
    """Measure peak RSS with N sessions each doing write+read concurrently.

    Uses Path input: _resolve_content opens a file handle, so the content is not in host memory.
    Source files are created BEFORE baseline measurement, so os.urandom bytes
    are freed and don't pollute the tracked region.

    Warmup run absorbs one-time init costs — see test_no_memory_leak_without_network
    docstring for rationale.

    Per-session peak during transfer:
      - File handle from _resolve_content: ~0 bytes (not a copy)
      - Encoded write frames in flight (_WRITE_FRAMES_IN_FLIGHT_MB): the whole
        file when it fits in the write pipeline, up to about 14.5 MiB per write
        in progress (64 queued + 16 + 1 transport frames
        of ~175 KB, plus the frame being built). The read that follows holds
        less (64 messages).
      - VM/channel overhead: ~_PEAK_RAM_PER_VM_MB
    Threshold: N * (per_vm_overhead + write frames in flight).
    """
    size_bytes = file_io_size_mb * 1024 * 1024

    # Create source files BEFORE baseline — os.urandom bytes freed after write_bytes
    sources: dict[int, Path] = {}
    for i in range(concurrent_sessions):
        src = tmp_path / f"concurrent_src_{i}.bin"
        src.write_bytes(os.urandom(size_bytes))
        sources[i] = src

    _, baseline_rss = await _warmup_and_baseline(scheduler_config, allow_network=False)

    async def session_cycle(scheduler: Scheduler, idx: int) -> None:
        src = sources[idx]
        dest = tmp_path / f"concurrent_{idx}.bin"
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file(f"concurrent_{idx}.bin", src)
            await session.read_file(f"concurrent_{idx}.bin", destination=dest)
            assert dest.stat().st_size == size_bytes
        dest.unlink(missing_ok=True)

    async with Scheduler(scheduler_config) as scheduler:
        async with PeakMemoryTracker(baseline_rss) as tracker:
            results = await asyncio.gather(
                *(session_cycle(scheduler, i) for i in range(concurrent_sessions)),
                return_exceptions=True,
            )

        _assert_success_rate(results)

    peak_growth_mb = (tracker.peak_rss - baseline_rss) / 1024 / 1024

    # Per session: VM overhead + the write frames in flight
    streaming_mb = _WRITE_FRAMES_IN_FLIGHT_MB[file_io_size_mb]
    threshold_mb = concurrent_sessions * (_PEAK_RAM_PER_VM_MB + streaming_mb)
    assert peak_growth_mb < threshold_mb, (
        f"Peak RAM too high: {peak_growth_mb:.1f}MB for {concurrent_sessions} sessions x {file_io_size_mb}MB "
        f"(threshold: {threshold_mb:.1f}MB = {concurrent_sessions} x "
        f"({_PEAK_RAM_PER_VM_MB}MB/VM + {streaming_mb:.1f}MB write frames in flight))"
    )


# =============================================================================
# tracemalloc-based Peak Detection Tests
# =============================================================================


@pytest.fixture(params=_PEAK_SIZES_MB, ids=[f"{s}MB" for s in _PEAK_SIZES_MB])
def peak_size_mb(request: pytest.FixtureRequest) -> int:
    """File size in MB of the tracemalloc peak tests (32 MiB is above each pipeline's capacity)."""
    return request.param


def _write_bound_message(*, source: str, size_mb: int, peak_mb: float) -> str:
    """The assertion message of a write_file peak test: the peak and how the bound is made."""
    return (
        f"tracemalloc peak too high for write_file ({source}, {size_mb}MB): {peak_mb:.2f}MB, bound "
        f"{_WRITE_PIPELINE_PEAK_MB:.2f}MB = {_WRITE_BOUND_TEXT}"
    )


@pytest.mark.slow
async def test_tracemalloc_peak_write_file(peak_size_mb: int, scheduler_config: SchedulerConfig) -> None:
    """Python allocation peak of write_file with bytes input: under the write pipeline bound.

    tracemalloc catches sub-ms spikes that 50 ms psutil sampling misses. The
    zstd C allocations are not traced: the psutil RSS tests cover them.

    content (os.urandom) is allocated BEFORE tracemalloc.start(), so it is not
    traced, and BytesIO(content) does not copy it. Traced: the frames in flight
    (_WRITE_PIPELINE_PEAK_MB). One bound for every size: at 32 MiB, a copy of
    the content or a pipeline that grows with the file goes over it.
    """
    size_bytes = peak_size_mb * 1024 * 1024

    async with Scheduler(scheduler_config) as scheduler:
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            content = os.urandom(size_bytes)
            tracemalloc.start()
            try:
                await session.write_file("tracemalloc_write.bin", content)
                _, peak_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

    peak_mb = peak_bytes / 1024 / 1024
    assert peak_mb < _WRITE_PIPELINE_PEAK_MB, _write_bound_message(
        source="bytes", size_mb=peak_size_mb, peak_mb=peak_mb
    )


@pytest.mark.slow
async def test_tracemalloc_peak_write_file_from_path(
    peak_size_mb: int, scheduler_config: SchedulerConfig, tmp_path: Path
) -> None:
    """Python allocation peak of write_file with Path input: the same bound as bytes input.

    _resolve_content opens a file handle; write_file reads it chunk by chunk.
    Traced: the frames in flight (_WRITE_PIPELINE_PEAK_MB), whatever the file size.
    """
    size_bytes = peak_size_mb * 1024 * 1024
    src = tmp_path / "tracemalloc_src.bin"
    src.write_bytes(os.urandom(size_bytes))

    async with Scheduler(scheduler_config) as scheduler:
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            tracemalloc.start()
            try:
                await session.write_file("tracemalloc_path_write.bin", src)
                _, peak_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

    peak_mb = peak_bytes / 1024 / 1024
    assert peak_mb < _WRITE_PIPELINE_PEAK_MB, _write_bound_message(source="Path", size_mb=peak_size_mb, peak_mb=peak_mb)


_225_VALUES = bytes(i % 225 for i in range(256))
"""Random bytes over 225 values compress by ~3 %: zstd writes compressed blocks, not raw ones,
so each read of the guest's stream is one whole block of ~127 KB (97 % of its 128 KiB buffer;
raw blocks of random bytes come out as 131,072 B and 3 B in turn: the queue would hold half)."""


class _SlowSink(io.RawIOBase):
    """A read_file destination that keeps nothing (no traced bytes) and takes 5 ms per write:
    the consumer falls behind, so the operation queue and the StreamReader of the port fill."""

    def __init__(self) -> None:
        super().__init__()
        self.size = 0

    def writable(self) -> bool:
        return True

    def write(self, b: Any) -> int:
        time.sleep(0.005)
        self.size += len(b)
        return len(b)


@pytest.mark.slow
@pytest.mark.parametrize("destination", ["file", "slow-stream"])
async def test_tracemalloc_peak_read_file(
    peak_size_mb: int, scheduler_config: SchedulerConfig, tmp_path: Path, destination: str
) -> None:
    """Measure Python-level allocation peak during read_file.

    Read loop, per 128 KiB chunk: op_queue.get() (pydantic decoded the base64 at
    the parse) → decompress → destination write. Bounded op_queue (64 messages) caps
    in-flight data at O(queue_capacity).
    Threshold: the full queue, the StreamReader buffer and the messages out of the queue
    (hand-computed: _FILE_IO_TRACEMALLOC_READ_OVERHEAD_MB). One bound for every size: at
    32 MiB, a read that holds the file goes over it. The content (over 225 values) makes each
    queued message a whole 128 KiB read; the slow stream keeps the consumer behind, a local
    file is the common case.
    """
    size_bytes = peak_size_mb * 1024 * 1024
    content = os.urandom(size_bytes).translate(_225_VALUES)  # each guest read: one whole block (~127 KB)
    sink = _SlowSink()
    dest = tmp_path / "tracemalloc_read.bin" if destination == "file" else sink

    async with Scheduler(scheduler_config) as scheduler:
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("tracemalloc_read.bin", content)

            tracemalloc.start()
            try:
                await session.read_file("tracemalloc_read.bin", destination=dest)
                _, peak_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

            assert (dest.stat().st_size if isinstance(dest, Path) else sink.size) == len(content)

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
async def test_peak_ram_large_file_io(large_file_mb: int, scheduler_config: SchedulerConfig, tmp_path: Path) -> None:
    """Measure peak RSS for large files — threshold is FLAT, not proportional.

    Requires hwaccel: RSS thresholds are calibrated for KVM/HVF — TCG JIT
    compilation inflates per-VM RSS, causing false positives.

    Uses Path input: _resolve_content opens a file handle, so the content is not in host memory.
    Streaming reads 128 KB chunks; up to about 14.5 MiB of encoded write
    frames are in flight, whatever the file size (both sizes fill the pipeline).
    Baseline is taken before Scheduler creation, so growth includes VM
    lifecycle overhead (~_PEAK_RAM_PER_VM_MB) + the frames in flight.
    Flat threshold proves streaming is bounded, not O(file_size):
    a 30 MB and 50 MB file must fit under the SAME threshold.

    Warmup run absorbs one-time init costs — see test_no_memory_leak_without_network
    docstring for rationale.
    """
    size_bytes = large_file_mb * 1024 * 1024
    src = tmp_path / "large_src.bin"
    src.write_bytes(os.urandom(size_bytes))
    dest = tmp_path / "large_dest.bin"

    _, baseline_rss = await _warmup_and_baseline(scheduler_config, allow_network=False)

    async with Scheduler(scheduler_config) as scheduler:
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            async with PeakMemoryTracker(baseline_rss) as tracker:
                await session.write_file("large_file.bin", src)
                await session.read_file("large_file.bin", destination=dest)

            assert dest.stat().st_size == size_bytes

    peak_growth_mb = (tracker.peak_rss - baseline_rss) / 1024 / 1024

    # Flat threshold: VM lifecycle + full pipeline RSS overhead.
    # During sustained large-file transfers the write pipeline (about 14.5 MiB:
    # _WRITE_FRAMES_IN_FLIGHT_MB) fills up, and C allocators / kernel buffers
    # add further overhead. _FILE_IO_RSS_OVERHEAD_MB captures this.
    threshold_mb = _PEAK_RAM_PER_VM_MB + _FILE_IO_RSS_OVERHEAD_MB
    assert peak_growth_mb < threshold_mb, (
        f"Peak RAM too high for {large_file_mb}MB file: {peak_growth_mb:.1f}MB "
        f"(flat threshold: {threshold_mb}MB = {_PEAK_RAM_PER_VM_MB}MB VM + "
        f"{_FILE_IO_RSS_OVERHEAD_MB}MB pipeline overhead). "
        f"Streaming is O(chunk_size), not O(file_size)."
    )


# =============================================================================
# Output Cap Memory Tests — verify guest-side cap bounds host memory
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("output_mb", [2, 100], ids=["2MB", "100MB"])
async def test_output_cap_bounds_host_memory(output_mb: int, scheduler_config: SchedulerConfig) -> None:
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

    async with Scheduler(scheduler_config) as scheduler:
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Warm-up: populates Pydantic caches, channel state, etc.
            await session.exec('print("warmup")')

            gc.collect()
            tracemalloc.start()
            try:
                with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.OUTPUT_LIMIT):
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

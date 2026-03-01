"""Shared CPU measurement helpers for idle-CPU tests.

Used by test_warm_vm_pool.py and test_orphan_vm_cpu.py to assert that
QEMU / gvproxy processes stay idle using percentile-based sampling
(median + p95) instead of flaky point-in-time readings.
"""

import asyncio
from statistics import median as _median

from exec_sandbox.platform_utils import ProcessWrapper

# --- Constants ---

CPU_SAMPLE_INTERVAL_S: float = 1.0
"""Interval between cpu_percent samples (seconds)."""

CPU_SAMPLES_DEFAULT: int = 15
"""Default sample count for standard tests (~15 s window)."""

CPU_SAMPLES_SUSTAINED: int = 30
"""Sample count for the sustained / slow test (~30 s window)."""

CPU_MEDIAN_THRESHOLD: float = 2.0
"""Idle median CPU must be below this (%). Observed: Python ~0.2 %, Raw ~0.2 %."""

CPU_P95_THRESHOLD: float = 10.0
"""Idle p95 CPU must be below this (%). Absorbs transient health-check spikes."""

CPU_CONSECUTIVE_SPIKE_THRESHOLD: float = 5.0
"""Per-sample threshold for the consecutive-spike detector."""

CPU_MAX_CONSECUTIVE_SPIKES: int = 3
"""Max consecutive samples > CPU_CONSECUTIVE_SPIKE_THRESHOLD before failure."""

CPU_SAMPLES_MIN: int = 5
"""Minimum samples required for meaningful p95 (guard against small-N bugs)."""


# --- Helpers ---


def p95(samples: list[float]) -> float:
    """95th percentile (allows ~1 outlier per 15-20 samples)."""
    assert len(samples) >= CPU_SAMPLES_MIN, f"Need >= {CPU_SAMPLES_MIN} samples for meaningful p95, got {len(samples)}"
    s = sorted(samples)
    idx = int(0.95 * (len(s) - 1))
    return s[idx]


async def collect_cpu_samples(
    proc: ProcessWrapper,
    n_samples: int = CPU_SAMPLES_DEFAULT,
    interval_s: float = CPU_SAMPLE_INTERVAL_S,
) -> list[float]:
    """Collect CPU % samples using non-blocking psutil pattern.

    Primes cpu_percent (first call returns 0.0, discarded),
    then collects n_samples at interval_s intervals.
    Each sample represents average CPU over the preceding interval.
    """
    assert proc.psutil_proc is not None
    proc.psutil_proc.cpu_percent(interval=None)  # Prime (discard 0.0)
    await asyncio.sleep(interval_s)

    samples: list[float] = []
    for _ in range(n_samples):
        cpu = proc.psutil_proc.cpu_percent(interval=None)
        samples.append(cpu)
        await asyncio.sleep(interval_s)
    return samples


async def collect_cpu_samples_bulk(
    procs: list[ProcessWrapper],
    n_samples: int = CPU_SAMPLES_DEFAULT,
    interval_s: float = CPU_SAMPLE_INTERVAL_S,
) -> list[list[float]]:
    """Collect CPU % samples from multiple processes in the same time window.

    Non-blocking bulk pattern: primes all, then samples all each interval.
    """
    for proc in procs:
        assert proc.psutil_proc is not None
        proc.psutil_proc.cpu_percent(interval=None)  # Prime all
    await asyncio.sleep(interval_s)

    all_samples: list[list[float]] = [[] for _ in procs]
    for _ in range(n_samples):
        for i, proc in enumerate(procs):
            assert proc.psutil_proc is not None
            cpu = proc.psutil_proc.cpu_percent(interval=None)
            all_samples[i].append(cpu)
        await asyncio.sleep(interval_s)
    return all_samples


def assert_cpu_idle(
    samples: list[float],
    *,
    label: str = "VM",
    median_threshold: float = CPU_MEDIAN_THRESHOLD,
    p95_threshold: float = CPU_P95_THRESHOLD,
) -> None:
    """Assert CPU samples indicate idle behavior (median + p95)."""
    med = _median(samples)
    p95_val = p95(samples)
    assert med < median_threshold, f"{label} median CPU {med:.1f}% >= {median_threshold}% (sorted: {sorted(samples)})"
    assert p95_val < p95_threshold, f"{label} p95 CPU {p95_val:.1f}% >= {p95_threshold}% (sorted: {sorted(samples)})"

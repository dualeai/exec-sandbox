"""Shared CPU measurement helpers for idle-CPU tests.

Used by test_warm_vm_pool.py and test_orphan_vm_cpu.py to assert that
QEMU / gvproxy processes stay idle using percentile-based sampling
(median + p95) instead of flaky point-in-time readings.
"""

import asyncio
import contextlib
from statistics import median as _median

import psutil

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


def sampled(proc: ProcessWrapper) -> psutil.Process:
    """The process whose CPU a sample measures and whose life a test checks: `proc`, or, when
    `proc` only starts QEMU (unshare, or sudo for a QEMU of qemu-vm), its qemu-system
    descendant: the wrapper itself idles, and can outlive or die before QEMU."""
    assert proc.psutil_proc is not None
    with contextlib.suppress(psutil.Error):
        if proc.psutil_proc.name().startswith("qemu-system"):
            return proc.psutil_proc
        for child in proc.psutil_proc.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                if child.name().startswith("qemu-system"):
                    return child
        # A wrapper with no QEMU found under it (hidepid): measuring the idle wrapper would
        # pass whatever QEMU does
        assert proc.psutil_proc.name() not in _QEMU_WRAPPERS, f"no qemu-system under {proc.psutil_proc.name()}"
    return proc.psutil_proc


_QEMU_WRAPPERS = frozenset({"sudo", "unshare", "sh", "stdbuf"})
"""The programs that start QEMU (qemu_cmd): a process of one of them is not QEMU itself."""


def gone(process: psutil.Process) -> bool:
    """`process` ended: no process, or a zombie (a QEMU under unshare dies before its reaper,
    init or a subreaper, collects it)."""
    try:
        return not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def p95(samples: list[float]) -> float:
    """95th percentile (allows ~1 outlier per 15-20 samples)."""
    assert len(samples) >= CPU_SAMPLES_MIN, f"Need >= {CPU_SAMPLES_MIN} samples for meaningful p95, got {len(samples)}"
    s = sorted(samples)
    idx = int(0.95 * (len(s) - 1))
    return s[idx]


async def collect_cpu_samples(proc: ProcessWrapper, n_samples: int = CPU_SAMPLES_DEFAULT) -> list[float]:
    """Collect CPU % samples of one process (collect_cpu_samples_bulk of one)."""
    (samples,) = await collect_cpu_samples_bulk([proc], n_samples=n_samples)
    return samples


async def collect_cpu_samples_bulk(
    procs: list[ProcessWrapper], n_samples: int = CPU_SAMPLES_DEFAULT
) -> list[list[float]]:
    """Collect CPU % samples from multiple processes in the same time window.

    Non-blocking psutil pattern: primes cpu_percent of all (the first call returns 0.0,
    discarded), then samples all each CPU_SAMPLE_INTERVAL_S. Each sample is the average
    CPU over the interval before it.
    """
    measured = [sampled(proc) for proc in procs]
    for process in measured:
        process.cpu_percent(interval=None)  # Prime all
    await asyncio.sleep(CPU_SAMPLE_INTERVAL_S)

    all_samples: list[list[float]] = [[] for _ in procs]
    for _ in range(n_samples):
        for i, process in enumerate(measured):
            all_samples[i].append(process.cpu_percent(interval=None))
        await asyncio.sleep(CPU_SAMPLE_INTERVAL_S)
    return all_samples


def assert_cpu_idle(samples: list[float], *, label: str) -> None:
    """Assert CPU samples indicate idle behavior (median + p95)."""
    med = _median(samples)
    p95_val = p95(samples)
    assert med < CPU_MEDIAN_THRESHOLD, (
        f"{label} median CPU {med:.1f}% >= {CPU_MEDIAN_THRESHOLD}% (sorted: {sorted(samples)})"
    )
    assert p95_val < CPU_P95_THRESHOLD, (
        f"{label} p95 CPU {p95_val:.1f}% >= {CPU_P95_THRESHOLD}% (sorted: {sorted(samples)})"
    )

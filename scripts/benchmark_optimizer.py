#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["exec-sandbox", "scipy", "numpy"]
#
# [tool.uv.sources]
# exec-sandbox = { path = ".." }
# ///
"""Overcommit optimizer for exec-sandbox.

Runs a 2D grid sweep over CPU and memory overcommit ratios, firing N VMs
per combination.  Optimizes across three latency dimensions:

  admission_p95 — time blocked in the admission queue
  setup_p95     — infra setup (overlay, cgroup)
  exec_p95      — VM boot + code execution

Computes a 3D Pareto front (non-dominated on all three), ranks by a
latency efficiency score (10^6 / geometric mean of the three p95s, per
Fleming & Wallace 1986 / SPEC convention), and fits an RBF surrogate
surface via scipy to predict the optimal beyond the sampled grid.

The sweep results inform the defaults in constants.py
(DEFAULT_CPU_OVERCOMMIT_RATIO, DEFAULT_MEMORY_OVERCOMMIT_RATIO).

Re-run after changing VM sizing, admission logic, or target hardware:
    make bench-optimizer              # 200 VMs per combo (default)
    make bench-optimizer N_VMS=500    # heavier load
    uv run --script scripts/benchmark_optimizer.py -n 50   # quick local test
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

import numpy as np  # type: ignore[reportMissingImports]
import psutil
from scipy.interpolate import RBFInterpolator  # type: ignore[reportMissingImports]
from scipy.optimize import minimize  # type: ignore[reportMissingImports]

from exec_sandbox import Scheduler, SchedulerConfig, VmSpec
from exec_sandbox._logging import configure_logging
from exec_sandbox.constants import DEFAULT_CPU_OVERCOMMIT_RATIO, DEFAULT_MEMORY_OVERCOMMIT_RATIO
from exec_sandbox.models import Language

configure_logging()

# ============================================================================
# Workload definitions
# ============================================================================
# Each entry: language + {code: proportion}.  Each burst draws its commands
# with these proportions (fixed seed, see _build_task_list) to model a
# realistic consumer mix.


@dataclass(frozen=True)
class Workload:
    """A workload definition: language + weighted command mix."""

    lang: Language
    commands: dict[str, float]  # code -> proportion (must sum to 1.0)


WORKLOADS: Final[list[Workload]] = [
    Workload(
        lang=Language.PYTHON,
        commands={
            "print('hello world')": 0.4,
            "import math; print(math.factorial(20))": 0.3,
            "print(sum(range(1_000)))": 0.3,
        },
    ),
    Workload(
        lang=Language.JAVASCRIPT,
        commands={
            "console.log('hello world')": 0.4,
            "console.log(Math.sqrt(144))": 0.3,
            "console.log(JSON.stringify({a: 1, b: [2,3]}))": 0.3,
        },
    ),
    Workload(
        lang=Language.RAW,
        commands={
            "echo hello world": 0.5,
            "cat /etc/os-release | head -1": 0.3,
            "uname -a": 0.2,
        },
    ),
]

# ============================================================================
# Sweep grid + result dataclass
# ============================================================================

SWEEP_CPU_OC: Final[list[float]] = [2.0, 4.0, 8.0, 12.0]
SWEEP_MEM_OC: Final[list[float]] = [1.5, 3.0, 5.0, 8.0]
SWEEP_DEFAULT_N_VMS: Final[int] = 200


@dataclass(frozen=True)
class BurstResult:
    """Summary metrics from a single burst run.

    Latency is split into three dimensions:
    - **setup**: cache lookup + overlay + cgroup (excludes admission)
    - **admission**: time blocked in admission queue (not part of setup, 0 for warm pool hits)
    - **exec**: VM boot + code execution (the actual work)
    """

    cpu_oc: float
    mem_oc: float
    n_vms: int
    n_ok: int
    wall_s: float
    # Admission queue latency (0 for warm pool hits)
    admission_p50_ms: int
    admission_p95_ms: int
    # Setup latency — infra only (cache + overlay + cgroup), excludes admission
    setup_p50_ms: int
    setup_p95_ms: int
    # Exec latency (boot + execute, hardware-dominated)
    exec_p50_ms: int
    exec_p95_ms: int
    # Total: wall clock of new() + exec() + close() over all VMs (failed ones too)
    total_p50_ms: int
    total_p95_ms: int
    # System pressure
    peak_rss_mb: int
    mem_total_mb: int

    @property
    def success_pct(self) -> int:
        return self.n_ok * 100 // self.n_vms if self.n_vms else 0

    @property
    def throughput(self) -> float:
        """Completed VMs per second."""
        return self.n_ok / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def rss_fraction(self) -> float:
        """Peak QEMU RSS as fraction of total RAM."""
        return self.peak_rss_mb / self.mem_total_mb if self.mem_total_mb > 0 else 0.0

    @property
    def efficiency(self) -> float:
        """Latency efficiency: 10^6 / geometric mean of p95 tail latencies.

        Uses the geometric mean of (admission_p95, setup_p95, exec_p95) so
        equal-percentage improvements in any dimension contribute equally,
        regardless of absolute scale (admission ~10s, setup ~1s, exec ~70ms).
        This follows the Fleming & Wallace (1986) / SPEC convention for
        combining benchmark metrics: geomean is normalization-invariant.

        Formula: efficiency = 10^6 / (admission * setup * exec)^(1/3)

        Higher is better.  When admission_p95 is 0 (no queueing), uses
        only setup and exec for a 2D geomean.  Returns NaN when setup or
        exec is zero (for example a burst with no successful VM).
        """
        a, s, e = self.admission_p95_ms, self.setup_p95_ms, self.exec_p95_ms
        if s <= 0 or e <= 0:
            return float("nan")
        if a <= 0:
            # No admission queueing — use 2D geomean of setup and exec
            return 1_000_000 / (s * e) ** (1 / 2)
        return 1_000_000 / (a * s * e) ** (1 / 3)


def _build_task_list(n: int) -> list[tuple[Language, str]]:
    """Build a deterministic list of (language, code) tuples from WORKLOADS.

    Draws *n* tasks with a fixed seed: each workload has an equal weight (its
    proportions sum to 1), and each command its proportion of that weight.
    Counts follow the weights only approximately.
    """
    population = [(wl.lang, code) for wl in WORKLOADS for code in wl.commands]
    weights = [proportion for wl in WORKLOADS for proportion in wl.commands.values()]
    return random.Random(42).choices(population, weights=weights, k=n)  # deterministic for reproducibility


# ============================================================================
# System monitor (100 ms background thread)
# ============================================================================


class SystemMonitor:
    """Background sampler of the total QEMU RSS (MB) at 100 ms resolution, while its `with` block runs."""

    def __init__(self, *, interval: float) -> None:
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self.samples.append(_qemu_rss_mb())
            self._stop.wait(self.interval)


def _qemu_rss_mb() -> int:
    """Total RSS of the QEMU processes via psutil, in MB."""
    qemu_rss_bytes = 0
    for proc in psutil.process_iter(["cmdline"]):  # type: ignore[reportUnknownMemberType]
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            cmdline: list[str] = proc.info.get("cmdline") or []  # type: ignore[union-attr]
            if any("qemu-system" in arg for arg in cmdline):
                qemu_rss_bytes += proc.memory_info().rss
    return qemu_rss_bytes // (1024 * 1024)


# ============================================================================
# VM runner
# ============================================================================


@dataclass(frozen=True)
class VMResult:
    """Result from a single VM execution."""

    ok: bool
    error: str | None  # why the run failed (an exception or an exit code), None when ok
    elapsed_ms: int
    setup_ms: int
    admission_ms: int  # Time blocked in admission queue (0 for warm pool hits)
    exec_ms: int  # boot_ms + execute_ms (includes VM boot, not pure execution)


async def _run_single(scheduler: Scheduler, lang: Language, code: str) -> VMResult:
    """Run a single VM and return timing/status."""
    t0 = time.monotonic()
    try:
        async with await scheduler.new(VmSpec(language=lang)) as vm:
            result = await vm.exec(code)
            startup = vm.startup_timing
        elapsed = time.monotonic() - t0
        return VMResult(
            ok=result.exit_code == 0,
            error=None if result.exit_code == 0 else f"exit code {result.exit_code}",
            elapsed_ms=round(elapsed * 1000),
            setup_ms=startup.setup_ms,
            admission_ms=startup.admission_ms,
            exec_ms=startup.boot_ms + result.timing.execute_ms,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        return VMResult(
            ok=False,
            error=f"{type(e).__name__}: {str(e)[:160]}",
            elapsed_ms=round(elapsed * 1000),
            setup_ms=0,
            admission_ms=0,
            exec_ms=0,
        )


def _percentile(data: list[int], pct: float) -> int:
    """Percentile (pct in 0-100) of sorted data, linearly interpolated; 0 when empty."""
    if not data:
        return 0
    k = (len(data) - 1) * pct / 100
    lo = int(k)
    hi = lo + 1
    if hi >= len(data):
        return data[-1]
    return round(data[lo] + (k - lo) * (data[hi] - data[lo]))


# ============================================================================
# Burst runner
# ============================================================================


async def _run_burst(n_vms: int, cpu_oc: float, mem_oc: float, *, images_dir: Path) -> BurstResult:
    """Execute a single burst and return summary metrics."""
    tasks_list = _build_task_list(n_vms)
    # The monitor runs until the Scheduler has stopped (also on an error or Ctrl-C)
    with SystemMonitor(interval=0.1) as monitor:
        wall_start = time.monotonic()
        async with Scheduler(
            SchedulerConfig(
                images_dir=images_dir,
                auto_download_assets=False,
                cpu_overcommit_ratio=cpu_oc,
                memory_overcommit_ratio=mem_oc,
            )
        ) as scheduler:
            coros = [_run_single(scheduler, lang, code) for lang, code in tasks_list]
            results = list(await asyncio.gather(*coros))
        wall_elapsed = time.monotonic() - wall_start

    ok_results = [r for r in results if r.ok]
    for error, count in Counter(r.error for r in results if r.error is not None).most_common(3):
        print(f"    Warning: {count} runs failed: {error}")
    # Total latency includes ALL VMs (failed VMs use wall-clock elapsed_ms,
    # reflecting the user-visible wait time including timeout).
    e2e_sorted = sorted(r.elapsed_ms for r in results)
    # Setup/admission/exec only from successes (failed VMs have setup_ms=0, admission_ms=0, exec_ms=0).
    setup_sorted = sorted(r.setup_ms for r in ok_results)
    admission_sorted = sorted(r.admission_ms for r in ok_results)
    exec_sorted = sorted(r.exec_ms for r in ok_results)
    peak_rss = max(monitor.samples, default=0)
    mem_total = psutil.virtual_memory().total // (1024 * 1024)

    return BurstResult(
        cpu_oc=cpu_oc,
        mem_oc=mem_oc,
        n_vms=n_vms,
        n_ok=len(ok_results),
        wall_s=wall_elapsed,
        admission_p50_ms=_percentile(admission_sorted, 50),
        admission_p95_ms=_percentile(admission_sorted, 95),
        setup_p50_ms=_percentile(setup_sorted, 50),
        setup_p95_ms=_percentile(setup_sorted, 95),
        exec_p50_ms=_percentile(exec_sorted, 50),
        exec_p95_ms=_percentile(exec_sorted, 95),
        total_p50_ms=_percentile(e2e_sorted, 50),
        total_p95_ms=_percentile(e2e_sorted, 95),
        peak_rss_mb=peak_rss,
        mem_total_mb=mem_total,
    )


# ============================================================================
# Efficient frontier analysis (Markowitz-inspired)
# ============================================================================


def _pareto_front(results: list[BurstResult]) -> set[tuple[float, float]]:
    """3D non-dominated sorting on (admission_p95, setup_p95, exec_p95).

    Finds configs where no other config is simultaneously better on all
    three latency dimensions.  O(M*N^2) with M=3, N=16 combos — trivial.

    Only configs with 100% success participate — a config with failures should
    never appear on the efficient frontier regardless of survivor latency.

    Dominance: b dominates a iff b <= a on all 3 objectives AND b < a on
    at least one (b != a).  The Pareto front is the set of non-dominated points.

    Returns set of (cpu_oc, mem_oc) tuples on the Pareto front.
    """
    points = {
        (r.cpu_oc, r.mem_oc): (r.admission_p95_ms, r.setup_p95_ms, r.exec_p95_ms)
        for r in results
        if r.success_pct == 100
    }
    return {
        combo
        for combo, a in points.items()
        if not any(b != a and all(x <= y for x, y in zip(b, a, strict=True)) for b in points.values())
    }


def _print_ranking(results: list[BurstResult]) -> None:
    """Print efficiency ranking with Pareto front markers."""
    pareto = _pareto_front(results)
    # NaN efficiency (no setup or exec time) sorts to the bottom
    ranked = sorted(results, key=lambda r: -r.efficiency if math.isfinite(r.efficiency) else float("inf"))

    print(f"\n{'=' * 140}")
    print("  LATENCY FRONTIER RANKING (10^6 / geomean(admission, setup, exec) p95)")
    print(f"{'=' * 140}")
    print(
        f"  {'Rank':>4}  {'CPU_OC':>6}  {'MEM_OC':>6}  {'OK%':>4}  {'VMs/s':>6}  "
        f"{'Effic':>7}  {'Pareto':>6}  "
        f"{'admit50':>8}  {'admit95':>8}  {'setup50':>8}  {'setup95':>8}  "
        f"{'exec50':>8}  {'exec95':>8}  "
        f"{'total50':>8}  {'total95':>8}  {'RSS%':>5}"
    )
    print(f"  {'─' * 138}")

    for rank, r in enumerate(ranked, 1):
        star = "★" if (r.cpu_oc, r.mem_oc) in pareto else ""
        eff_str = f"{r.efficiency:7.1f}" if math.isfinite(r.efficiency) else "    n/a"
        print(
            f"  {rank:4d}  {r.cpu_oc:6.1f}  {r.mem_oc:6.1f}  {r.success_pct:3d}%  "
            f"{r.throughput:6.1f}  {eff_str}  "
            f"{star:>6}  "
            f"{r.admission_p50_ms:6d}ms  {r.admission_p95_ms:6d}ms  "
            f"{r.setup_p50_ms:6d}ms  {r.setup_p95_ms:6d}ms  "
            f"{r.exec_p50_ms:6d}ms  {r.exec_p95_ms:6d}ms  "
            f"{r.total_p50_ms:6d}ms  {r.total_p95_ms:6d}ms  "
            f"{r.rss_fraction * 100:4.1f}%"
        )

    print(f"{'=' * 140}")

    # Pareto front explanation
    pareto_results = [r for r in ranked if (r.cpu_oc, r.mem_oc) in pareto]
    if pareto_results:
        print("\n  ★ Pareto-optimal: no other config is better on ALL of admission, setup, exec p95")
        for r in sorted(pareto_results, key=lambda r: r.admission_p95_ms + r.setup_p95_ms + r.exec_p95_ms):
            print(
                f"    CPU={r.cpu_oc:.0f}x MEM={r.mem_oc:.1f}x → "
                f"admit_p95={r.admission_p95_ms}ms  setup_p95={r.setup_p95_ms}ms  "
                f"exec_p95={r.exec_p95_ms}ms  "
                f"(geomean={int((r.admission_p95_ms * r.setup_p95_ms * r.exec_p95_ms) ** (1 / 3))}ms)"
            )


def _surrogate_analysis(results: list[BurstResult]) -> None:
    """RBF surrogate surface + scipy.optimize to find predicted optimal.

    Fits a radial basis function interpolant on the sampled grid using
    the latency efficiency score (10^6 / geomean of p95 tail latencies),
    then runs multi-start L-BFGS-B to locate the global maximum.
    """
    # Drop results with NaN efficiency (no setup or exec time): cannot train on them
    valid = [r for r in results if math.isfinite(r.efficiency)]
    if len(valid) < 4:
        print(f"\n  (need >= 4 valid grid points for surrogate, got {len(valid)} -- skipping)")
        return

    # Training data: (cpu_oc, mem_oc) -> latency efficiency
    coords = np.array([[r.cpu_oc, r.mem_oc] for r in valid])  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
    values = np.array([r.efficiency for r in valid])  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]

    try:
        rbf = RBFInterpolator(coords, values, kernel="thin_plate_spline")  # type: ignore[reportUnknownVariableType]
    except np.linalg.LinAlgError:  # type: ignore[reportUnknownMemberType]
        print(f"\n  (surrogate: singular matrix with {len(valid)} points — data may be collinear, skipping)")
        return

    # Search bounds from grid extremes
    cpu_lo = float(coords[:, 0].min())  # type: ignore[reportUnknownMemberType]
    cpu_hi = float(coords[:, 0].max())  # type: ignore[reportUnknownMemberType]
    mem_lo = float(coords[:, 1].min())  # type: ignore[reportUnknownMemberType]
    mem_hi = float(coords[:, 1].max())  # type: ignore[reportUnknownMemberType]
    bounds = [(cpu_lo, cpu_hi), (mem_lo, mem_hi)]

    # Multi-start optimization to avoid local optima
    best_x: object | None = None
    best_val = float("-inf")
    for cpu_s in np.linspace(cpu_lo, cpu_hi, 4):  # type: ignore[reportUnknownMemberType]
        for mem_s in np.linspace(mem_lo, mem_hi, 4):  # type: ignore[reportUnknownMemberType]
            res = minimize(  # type: ignore[reportUnknownVariableType]
                lambda x: -float(rbf(x.reshape(1, -1))[0]),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                x0=np.array([float(cpu_s), float(mem_s)]),  # type: ignore[reportUnknownMemberType]
                bounds=bounds,
                method="L-BFGS-B",
            )
            if res.success and -res.fun > best_val:  # type: ignore[reportUnknownMemberType]
                best_val = float(-res.fun)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                best_x = res.x  # type: ignore[reportUnknownMemberType]

    if best_x is not None:
        _print_surrogate_result(valid, best_x, best_val)  # type: ignore[reportUnknownArgumentType]
    else:
        print("\n  (surrogate optimization: all starts failed to converge -- skipping)")


def _print_surrogate_result(
    results: list[BurstResult],
    best_x: object,
    best_val: float,
) -> None:
    """Print RBF surrogate prediction vs best sampled point."""
    x = np.asarray(best_x)  # type: ignore[reportUnknownMemberType]
    print(f"\n  RBF SURROGATE PREDICTION (thin-plate spline, {len(results)} samples)")
    opt_cpu = float(x[0])  # type: ignore[reportUnknownArgumentType]
    opt_mem = float(x[1])  # type: ignore[reportUnknownArgumentType]
    print(f"    Predicted optimal: CPU_OC={opt_cpu:.1f}, MEM_OC={opt_mem:.1f}")
    print(f"    Predicted latency efficiency: {best_val:.1f}")
    best_sampled = max(results, key=lambda r: r.efficiency)
    delta_pct = (
        (best_val - best_sampled.efficiency) / best_sampled.efficiency * 100 if best_sampled.efficiency > 0 else 0
    )
    print(
        f"    Best sampled: CPU_OC={best_sampled.cpu_oc:.1f}, "
        f"MEM_OC={best_sampled.mem_oc:.1f}, latency efficiency={best_sampled.efficiency:.1f}"
    )
    if delta_pct > 2.0:
        print(f"    → Surrogate suggests {delta_pct:.1f}% better beyond grid (consider refining)")
    else:
        print("    → Grid already covers the optimum")


# ============================================================================
# Main
# ============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overcommit optimizer for exec-sandbox")
    parser.add_argument("-n", type=int, default=SWEEP_DEFAULT_N_VMS, help="VMs per combo (default: %(default)s)")
    return parser.parse_args()


async def main() -> int:
    """Run 2D overcommit sweep and compute efficient frontier."""
    args = _parse_args()
    n_vms: int = args.n
    # The images of this checkout (make build): `uv run --script` installs the package
    # in its own environment, where the default would be the download cache
    images_dir = Path(__file__).parent.parent / "images" / "dist"
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}. Run 'make build' first.")
        return 1
    combos = [(cpu, mem) for cpu in SWEEP_CPU_OC for mem in SWEEP_MEM_OC]
    print("=" * 80)
    print(f"  2D OVERCOMMIT SWEEP: {len(combos)} combos x {n_vms} VMs each")
    print(f"  CPU_OC = {SWEEP_CPU_OC}")
    print(f"  MEM_OC = {SWEEP_MEM_OC}")
    print(f"  Current defaults: CPU={DEFAULT_CPU_OVERCOMMIT_RATIO}x, MEM={DEFAULT_MEMORY_OVERCOMMIT_RATIO}x")
    print("=" * 80)

    all_results: list[BurstResult] = []
    for i, (cpu_oc, mem_oc) in enumerate(combos, 1):
        print(f"\n  [{i}/{len(combos)}] CPU_OC={cpu_oc:.1f} MEM_OC={mem_oc:.1f}...")
        burst = await _run_burst(n_vms, cpu_oc, mem_oc, images_dir=images_dir)
        all_results.append(burst)
        print(
            f"    → {burst.n_ok}/{burst.n_vms} ok ({burst.success_pct}%) | "
            f"wall={burst.wall_s:.1f}s | "
            f"admit={burst.admission_p50_ms}/{burst.admission_p95_ms}ms "
            f"setup={burst.setup_p50_ms}/{burst.setup_p95_ms}ms "
            f"exec={burst.exec_p50_ms}/{burst.exec_p95_ms}ms "
            f"total={burst.total_p50_ms}/{burst.total_p95_ms}ms | "
            f"RSS={burst.peak_rss_mb}MB ({burst.rss_fraction * 100:.1f}%) | "
            f"eff={'n/a' if not math.isfinite(burst.efficiency) else f'{burst.efficiency:.1f}'}"
        )

    # Efficient frontier analysis
    _print_ranking(all_results)
    _surrogate_analysis(all_results)

    # Verdict
    all_ok = all(r.success_pct == 100 for r in all_results)
    print(f"\n{'=' * 80}")
    if all_ok:
        print(f"  SWEEP VERDICT: ALL {len(all_results)} combos achieved 100% success")
    else:
        failed = [r for r in all_results if r.success_pct < 100]
        print(f"  SWEEP VERDICT: {len(failed)}/{len(all_results)} combos had failures")
    print("=" * 80)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["exec-sandbox", "psutil"]
#
# [tool.uv.sources]
# exec-sandbox = { path = ".." }
# ///
"""VM density benchmark for exec-sandbox.

Measures host-level memory efficiency when running concurrent VMs.
Answers: "How much host RAM does each additional VM actually cost?"

Methodology:
  1. Measure host baseline memory (no VMs)
  2. Boot N VMs, keep them alive
  3. Optionally execute a workload on each VM
  4. Sample host memory at steady state
  5. Compute per-VM marginal cost = (used_after - used_before) / N

Runs multiple "scenarios" to compare density techniques:
  - Baseline (current config)
  - With free page reporting (if supported)
  - With balloon inflated (idle VMs)

Supports both macOS (vm_stat) and Linux (/proc/meminfo, KSM, PSI).

Usage:
    uv run --script scripts/benchmark_density.py               # Quick: 5 VMs
    uv run --script scripts/benchmark_density.py -n 10         # 10 VMs
    uv run --script scripts/benchmark_density.py --workload     # Run code on each VM
    uv run --script scripts/benchmark_density.py --lang python  # Python only
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import psutil

from exec_sandbox import Scheduler, SchedulerConfig
from exec_sandbox._logging import configure_logging
from exec_sandbox.constants import DEFAULT_CPU_OVERCOMMIT_RATIO, DEFAULT_MEMORY_OVERCOMMIT_RATIO
from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostOS, detect_host_os

if TYPE_CHECKING:
    from exec_sandbox.session import Session

configure_logging()

# ============================================================================
# Constants
# ============================================================================

_HOST_OS: Final[HostOS] = detect_host_os()

WORKLOAD_CODE: Final[dict[Language, str]] = {
    Language.PYTHON: "import json, math, os; print(json.dumps({'pid': os.getpid(), 'fact': math.factorial(100)}))",
    Language.JAVASCRIPT: "console.log(JSON.stringify({pid: process.pid, sqrt: Math.sqrt(2)}))",
    Language.RAW: "echo ok && cat /proc/meminfo 2>/dev/null || vm_stat 2>/dev/null || echo no-meminfo",
}

LANG_DISPLAY: Final[dict[Language, str]] = {
    Language.PYTHON: "Python",
    Language.JAVASCRIPT: "JavaScript",
    Language.RAW: "Raw",
}

# Stabilization: wait for memory to settle after VM operations
SETTLE_SECONDS: Final[float] = 3.0


# ============================================================================
# Host memory sampling
# ============================================================================


@dataclass(frozen=True)
class HostMemorySample:
    """Single host memory measurement."""

    timestamp: float
    total_mb: int
    available_mb: int
    used_mb: int  # total - available
    # QEMU-specific
    qemu_count: int
    qemu_rss_mb: int  # Sum of RSS across all QEMU processes
    qemu_pss_mb: int | None  # Sum of PSS (Linux only, needs root)
    # Linux-specific
    ksm_pages_sharing: int | None  # Pages saved by KSM
    ksm_pages_shared: int | None  # Unique pages being shared
    psi_some_avg10: float | None  # Memory pressure (some, avg10)
    swap_used_mb: int | None
    committed_mb: int | None  # Committed_AS from /proc/meminfo


def _collect_qemu_metrics() -> tuple[int, int, int | None]:
    """Collect QEMU process count, total RSS, and PSS (Linux only)."""
    count = 0
    rss_bytes = 0
    pss_bytes: int | None = 0 if _HOST_OS == HostOS.LINUX else None

    for proc in psutil.process_iter(["cmdline"]):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            cmdline: list[str] = proc.info.get("cmdline") or []  # type: ignore[union-attr]
            if not any("qemu-system" in arg for arg in cmdline):
                continue
            count += 1
            rss_bytes += proc.memory_info().rss
            # PSS from smaps_rollup (Linux, may need root)
            if pss_bytes is not None:
                try:
                    smaps = Path(f"/proc/{proc.pid}/smaps_rollup").read_text()
                    for line in smaps.splitlines():
                        if line.startswith("Pss:"):
                            pss_bytes += int(line.split()[1]) * 1024  # KB -> bytes
                            break
                except (OSError, ValueError):
                    pass  # Permission denied or process exited

    rss_mb = rss_bytes // (1024 * 1024)
    pss_mb = pss_bytes // (1024 * 1024) if pss_bytes is not None else None
    return count, rss_mb, pss_mb


def _read_ksm_stats() -> tuple[int | None, int | None]:
    """Read KSM stats from sysfs (Linux only)."""
    if _HOST_OS != HostOS.LINUX:
        return None, None
    try:
        sharing = int(Path("/sys/kernel/mm/ksm/pages_sharing").read_text().strip())
        shared = int(Path("/sys/kernel/mm/ksm/pages_shared").read_text().strip())
        return sharing, shared
    except (OSError, ValueError):
        return None, None


def _read_psi_memory() -> float | None:
    """Read memory PSI some avg10 (Linux only)."""
    if _HOST_OS != HostOS.LINUX:
        # macOS: try kern.memorystatus_vm_pressure_level
        try:
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            # 1=normal, 2=warn, 4=critical -> map to percentage-like
            level = int(result.stdout.strip())
            return {1: 0.0, 2: 25.0, 4: 75.0}.get(level, 0.0)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return None
    try:
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            if line.startswith("some"):
                # "some avg10=0.50 avg60=0.20 avg300=0.10 total=12345"
                for part in line.split():
                    if part.startswith("avg10="):
                        return float(part.split("=")[1])
    except (OSError, ValueError):
        pass
    return None


def sample_host_memory() -> HostMemorySample:
    """Take a single host memory snapshot."""
    vmem = psutil.virtual_memory()
    total_mb = vmem.total // (1024 * 1024)
    avail_mb = vmem.available // (1024 * 1024)
    used_mb = total_mb - avail_mb

    qemu_count, qemu_rss_mb, qemu_pss_mb = _collect_qemu_metrics()
    ksm_sharing, ksm_shared = _read_ksm_stats()
    psi = _read_psi_memory()

    swap_used_mb: int | None = None
    committed_mb: int | None = None
    if _HOST_OS == HostOS.LINUX:
        swap = psutil.swap_memory()
        swap_used_mb = swap.used // (1024 * 1024)
        with contextlib.suppress(OSError, ValueError):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("Committed_AS:"):
                    committed_mb = int(line.split()[1]) // 1024
                    break

    return HostMemorySample(
        timestamp=time.monotonic(),
        total_mb=total_mb,
        available_mb=avail_mb,
        used_mb=used_mb,
        qemu_count=qemu_count,
        qemu_rss_mb=qemu_rss_mb,
        qemu_pss_mb=qemu_pss_mb,
        ksm_pages_sharing=ksm_sharing,
        ksm_pages_shared=ksm_shared,
        psi_some_avg10=psi,
        swap_used_mb=swap_used_mb,
        committed_mb=committed_mb,
    )


# ============================================================================
# Background sampler (continuous memory tracking during VM operations)
# ============================================================================


class MemorySampler:
    """Background thread that samples host memory at fixed intervals."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.samples: list[HostMemorySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self.samples.append(sample_host_memory())
            self._stop.wait(self.interval)

    @property
    def peak_used_mb(self) -> int:
        return max((s.used_mb for s in self.samples), default=0)

    @property
    def peak_qemu_rss_mb(self) -> int:
        return max((s.qemu_rss_mb for s in self.samples), default=0)

    @property
    def peak_qemu_count(self) -> int:
        return max((s.qemu_count for s in self.samples), default=0)


# ============================================================================
# Density measurement result
# ============================================================================


@dataclass
class DensityResult:
    """Result from a single density measurement scenario."""

    scenario: str
    n_vms: int
    language: Language
    # Baseline (before VMs)
    baseline_used_mb: int
    baseline_available_mb: int
    # Steady state (VMs alive)
    steady_used_mb: int
    steady_available_mb: int
    steady_qemu_rss_mb: int
    steady_qemu_pss_mb: int | None
    steady_qemu_count: int
    # Peak during lifetime
    peak_used_mb: int
    peak_qemu_rss_mb: int
    # KSM (Linux only)
    ksm_pages_sharing: int | None
    ksm_pages_shared: int | None
    # Pressure
    psi_some_avg10: float | None
    # Timing
    boot_total_ms: int  # Wall clock to boot all VMs
    exec_total_ms: int | None  # Wall clock to execute on all VMs (if workload)
    # Per-VM latencies
    boot_latencies_ms: list[int] = field(default_factory=list[int])
    exec_latencies_ms: list[int] = field(default_factory=list[int])

    @property
    def delta_used_mb(self) -> int:
        """Host memory consumed by VMs = steady - baseline."""
        return self.steady_used_mb - self.baseline_used_mb

    @property
    def marginal_mb_per_vm(self) -> float:
        """Marginal host memory cost per VM."""
        return self.delta_used_mb / self.n_vms if self.n_vms > 0 else 0.0

    @property
    def rss_per_vm_mb(self) -> float:
        """Average QEMU RSS per VM."""
        return self.steady_qemu_rss_mb / self.n_vms if self.n_vms > 0 else 0.0

    @property
    def pss_per_vm_mb(self) -> float | None:
        """Average QEMU PSS per VM (Linux only)."""
        if self.steady_qemu_pss_mb is None:
            return None
        return self.steady_qemu_pss_mb / self.n_vms if self.n_vms > 0 else 0.0

    @property
    def ksm_saved_mb(self) -> float | None:
        """Memory saved by KSM in MB.

        pages_sharing = extra virtual mappings pointing to shared physical pages (= savings).
        pages_shared = unique physical pages being shared (metadata cost, not savings).
        Per kernel docs: savings = pages_sharing * page_size.
        """
        if self.ksm_pages_sharing is None:
            return None
        return self.ksm_pages_sharing * 4096 / (1024 * 1024)

    @property
    def estimated_max_vms(self) -> int | None:
        """Estimated max VMs this host could sustain (based on marginal cost)."""
        if self.marginal_mb_per_vm <= 0:
            return None
        host_total = self.baseline_available_mb + self.baseline_used_mb
        usable = host_total * 0.9  # 10% reserve
        return int(usable / self.marginal_mb_per_vm)


# ============================================================================
# Helpers
# ============================================================================


async def _median_sample(n: int = 5, interval: float = 0.5) -> HostMemorySample:
    """Take n samples and return the one with median used_mb.

    Sorting sample objects (not just values) ensures all fields on the
    selected sample are from the same instant — no Franken-sample.
    """
    samples: list[HostMemorySample] = []
    for _ in range(n):
        samples.append(sample_host_memory())
        await asyncio.sleep(interval)
    by_used = sorted(samples, key=lambda s: s.used_mb)
    return by_used[len(by_used) // 2]


# ============================================================================
# Scenario runner
# ============================================================================


async def measure_density(  # noqa: PLR0915
    n_vms: int,
    language: Language,
    scenario: str,
    *,
    run_workload: bool = False,
    memory_mb: int = 192,
    overcommit_cpu: float = DEFAULT_CPU_OVERCOMMIT_RATIO,
    overcommit_mem: float = DEFAULT_MEMORY_OVERCOMMIT_RATIO,
) -> DensityResult:
    """Boot N VMs, hold them alive, measure host memory impact."""
    images_dir = Path(__file__).parent.parent / "images" / "dist"
    code = WORKLOAD_CODE.get(language, "echo ok")

    # -- Baseline sample (no VMs from us) --
    await asyncio.sleep(SETTLE_SECONDS)
    baseline = sample_host_memory()
    print(
        f"    Baseline: {baseline.used_mb} MB used, {baseline.available_mb} MB available, "
        f"{baseline.qemu_count} existing QEMU processes"
    )

    # -- Start background sampler --
    sampler = MemorySampler(interval=0.25)
    sampler.start()

    config = SchedulerConfig(
        images_dir=images_dir,
        auto_download_assets=False,
        warm_pool_size=0,  # Cold boot everything for fair measurement
        cpu_overcommit_ratio=overcommit_cpu,
        memory_overcommit_ratio=overcommit_mem,
        default_memory_mb=memory_mb,
    )

    boot_latencies: list[int] = []
    exec_latencies: list[int] = []
    exec_wall_ms: int | None = None

    async with Scheduler(config) as scheduler:
        # -- Boot all VMs concurrently using session() --
        # Use sessions to keep VMs alive for measurement
        sessions: list[Session] = []
        boot_errors = 0
        boot_wall_start = time.monotonic()

        async def boot_one(idx: int) -> None:
            # Safe: asyncio is single-threaded cooperative multitasking.
            # list.append() and += have no await between read and write.
            nonlocal boot_errors
            t0 = time.monotonic()
            try:
                session = await scheduler.session(language=language, memory_mb=memory_mb)
                sessions.append(session)
                boot_latencies.append(round((time.monotonic() - t0) * 1000))
            except Exception as e:
                boot_errors += 1
                print(f"    Warning: VM {idx} boot failed: {type(e).__name__}: {e}")

        await asyncio.gather(*[boot_one(i) for i in range(n_vms)])
        boot_wall_ms = round((time.monotonic() - boot_wall_start) * 1000)

        if boot_errors:
            print(f"    {boot_errors}/{n_vms} VMs failed to boot")

        n_booted = len(sessions)
        print(f"    Booted {n_booted}/{n_vms} VMs in {boot_wall_ms}ms")

        # -- Let memory settle --
        await asyncio.sleep(SETTLE_SECONDS)

        # -- Steady-state sample (VMs alive, idle, median of 5) --
        steady = await _median_sample()
        print(
            f"    Steady state: {steady.used_mb} MB used, {steady.available_mb} MB available, "
            f"{steady.qemu_count} QEMU processes, RSS={steady.qemu_rss_mb} MB"
            + (f", PSS={steady.qemu_pss_mb} MB" if steady.qemu_pss_mb is not None else "")
        )

        # -- Optional workload execution --
        if run_workload and sessions:
            print(f"    Executing workload on {n_booted} VMs...")
            exec_wall_start = time.monotonic()

            async def exec_one(sess: Session, idx: int) -> None:
                t0 = time.monotonic()
                try:
                    result = await sess.exec(code=code)
                    exec_latencies.append(round((time.monotonic() - t0) * 1000))
                    if result.exit_code != 0:
                        print(f"    Warning: VM {idx} exec failed: {result.stderr[:100]}")
                except Exception as e:
                    print(f"    Warning: VM {idx} exec error: {type(e).__name__}: {e}")

            await asyncio.gather(*[exec_one(sess, i) for i, sess in enumerate(sessions)])
            exec_wall_ms = round((time.monotonic() - exec_wall_start) * 1000)

            # Settle after workload
            await asyncio.sleep(SETTLE_SECONDS)

            # Post-workload sample
            post_exec = sample_host_memory()
            print(f"    Post-exec: {post_exec.used_mb} MB used, {post_exec.qemu_rss_mb} MB QEMU RSS")

        # -- Teardown: close all sessions --
        print(f"    Tearing down {n_booted} VMs...")
        for sess in sessions:
            with contextlib.suppress(Exception):
                await sess.close()

    sampler.stop()

    # Use the idle steady-state sample for the result, not the post-workload
    # final sample.  This ensures marginal_mb_per_vm always measures the idle
    # VM memory footprint, regardless of whether --workload was passed.
    return DensityResult(
        scenario=scenario,
        n_vms=n_booted,
        language=language,
        baseline_used_mb=baseline.used_mb,
        baseline_available_mb=baseline.available_mb,
        steady_used_mb=steady.used_mb,
        steady_available_mb=steady.available_mb,
        steady_qemu_rss_mb=steady.qemu_rss_mb,
        steady_qemu_pss_mb=steady.qemu_pss_mb,
        steady_qemu_count=steady.qemu_count,
        peak_used_mb=sampler.peak_used_mb,
        peak_qemu_rss_mb=sampler.peak_qemu_rss_mb,
        boot_total_ms=boot_wall_ms,
        exec_total_ms=exec_wall_ms,
        ksm_pages_sharing=steady.ksm_pages_sharing,
        ksm_pages_shared=steady.ksm_pages_shared,
        psi_some_avg10=steady.psi_some_avg10,
        boot_latencies_ms=boot_latencies,
        exec_latencies_ms=exec_latencies,
    )


# ============================================================================
# Report printing
# ============================================================================


def _fmt_latency(values: list[int]) -> str:
    """Format latency list as median/p95 (nearest-rank method)."""
    if not values:
        return "-"
    s = sorted(values)
    med = statistics.median(s)
    # Nearest-rank p95, matching benchmark_latency.py's _percentile()
    p95_idx = min(math.ceil(len(s) * 0.95) - 1, len(s) - 1)
    return f"{med:.0f} / {s[p95_idx]}"


def print_result(r: DensityResult) -> None:
    """Print a single density measurement result."""
    print(f"\n  {'─' * 70}")
    print(f"  {r.scenario} ({LANG_DISPLAY[r.language]}, {r.n_vms} VMs)")
    print(f"  {'─' * 70}")

    print("  Host memory:")
    print(f"    Baseline:          {r.baseline_used_mb:>6} MB used  /  {r.baseline_available_mb:>6} MB available")
    print(f"    With VMs:          {r.steady_used_mb:>6} MB used  /  {r.steady_available_mb:>6} MB available")
    print(
        f"    Delta:             {r.delta_used_mb:>+6} MB  ({r.delta_used_mb / (r.baseline_used_mb + r.baseline_available_mb) * 100:.1f}% of total)"
    )
    print(f"    Peak used:         {r.peak_used_mb:>6} MB")

    print("  Per-VM cost:")
    print(f"    Marginal (host):   {r.marginal_mb_per_vm:>6.1f} MB/VM  (delta_used / N)")
    print(f"    RSS (QEMU sum):    {r.rss_per_vm_mb:>6.1f} MB/VM  (total QEMU RSS / N)")
    if r.pss_per_vm_mb is not None:
        print(f"    PSS (proportional):{r.pss_per_vm_mb:>6.1f} MB/VM  (shared pages counted fractionally)")

    if r.ksm_saved_mb is not None:
        print("  KSM:")
        print(f"    Pages sharing:     {r.ksm_pages_sharing}")
        print(f"    Pages shared:      {r.ksm_pages_shared}")
        print(f"    Memory saved:      {r.ksm_saved_mb:>6.1f} MB")

    if r.psi_some_avg10 is not None:
        print(f"  Memory pressure:     {r.psi_some_avg10:.2f}%")

    if r.estimated_max_vms is not None:
        print(
            f"  Estimated max VMs:   ~{r.estimated_max_vms} (linear extrapolation at N={r.n_vms}, "
            f"conservative at low N due to page-sharing amortization)"
        )

    print("  Timing:")
    print(f"    Boot (wall):       {r.boot_total_ms:>6} ms  (all {r.n_vms} VMs concurrent)")
    print(f"    Boot (per-VM):     {_fmt_latency(r.boot_latencies_ms):>12} ms  (median / p95)")
    if r.exec_total_ms is not None:
        print(f"    Exec (wall):       {r.exec_total_ms:>6} ms")
        print(f"    Exec (per-VM):     {_fmt_latency(r.exec_latencies_ms):>12} ms  (median / p95)")


def print_comparison(results: list[DensityResult]) -> None:
    """Print a comparison table across scenarios."""
    if len(results) < 2:
        return

    print(f"\n{'=' * 90}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 90}")

    header = (
        f"  {'Scenario':<30} {'VMs':>4} {'Delta MB':>9} {'MB/VM':>7} {'RSS/VM':>7}"
        f" {'PSS/VM':>7} {'KSM MB':>7} {'Boot ms':>8} {'MaxVMs':>7}"
    )
    print(header)
    print(f"  {'─' * 88}")

    baseline_marginal = results[0].marginal_mb_per_vm if results else 0

    for r in results:
        pss_str = f"{r.pss_per_vm_mb:.1f}" if r.pss_per_vm_mb is not None else "-"
        ksm_str = f"{r.ksm_saved_mb:.1f}" if r.ksm_saved_mb is not None else "-"
        max_str = f"~{r.estimated_max_vms}" if r.estimated_max_vms else "-"
        boot_med = statistics.median(r.boot_latencies_ms) if r.boot_latencies_ms else 0

        # Show cost change vs first result (negative = cheaper, positive = more expensive)
        savings = ""
        same_n = len({res.n_vms for res in results}) == 1
        if same_n and baseline_marginal > 0 and r is not results[0]:
            pct = (r.marginal_mb_per_vm / baseline_marginal - 1) * 100
            savings = f" ({pct:+.0f}%)"

        print(
            f"  {r.scenario:<30} {r.n_vms:>4} {r.delta_used_mb:>+8} "
            f"{r.marginal_mb_per_vm:>6.1f}{savings:>0} {r.rss_per_vm_mb:>6.1f}"
            f" {pss_str:>7} {ksm_str:>7} {boot_med:>7.0f} {max_str:>7}"
        )

    print(f"{'=' * 90}")


def print_json_summary(results: list[DensityResult]) -> None:
    """Print machine-readable JSON summary."""
    summary: list[dict[str, Any]] = []
    for r in results:
        entry = {
            "scenario": r.scenario,
            "language": r.language.value,
            "n_vms": r.n_vms,
            "baseline_used_mb": r.baseline_used_mb,
            "steady_used_mb": r.steady_used_mb,
            "delta_used_mb": r.delta_used_mb,
            "marginal_mb_per_vm": round(r.marginal_mb_per_vm, 1),
            "rss_per_vm_mb": round(r.rss_per_vm_mb, 1),
            "pss_per_vm_mb": round(r.pss_per_vm_mb, 1) if r.pss_per_vm_mb is not None else None,
            "peak_used_mb": r.peak_used_mb,
            "peak_qemu_rss_mb": r.peak_qemu_rss_mb,
            "ksm_saved_mb": round(r.ksm_saved_mb, 1) if r.ksm_saved_mb is not None else None,
            "psi_some_avg10": r.psi_some_avg10,
            "estimated_max_vms": r.estimated_max_vms,
            "boot_total_ms": r.boot_total_ms,
            "boot_median_ms": round(statistics.median(r.boot_latencies_ms)) if r.boot_latencies_ms else None,
            "exec_total_ms": r.exec_total_ms,
            "exec_median_ms": round(statistics.median(r.exec_latencies_ms)) if r.exec_latencies_ms else None,
        }
        summary.append(entry)
    print(f"\n  JSON: {json.dumps(summary, indent=2)}")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VM density benchmark — measure host memory cost per VM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Quick: 5 Python VMs
  %(prog)s -n 10                    10 VMs
  %(prog)s -n 10 --workload         Run code on each VM
  %(prog)s --lang python raw        Compare Python vs Raw
  %(prog)s --sweep 5 10 20          Sweep N=5,10,20 VMs
  %(prog)s --json                   Include machine-readable output
  %(prog)s --memory 128             Use 128MB VMs (denser)
""",
    )
    parser.add_argument("-n", type=int, default=5, metavar="N", help="number of VMs (default: 5)")
    parser.add_argument("--workload", action="store_true", help="execute code on each VM after boot")
    parser.add_argument(
        "--lang",
        nargs="+",
        choices=["python", "javascript", "raw"],
        default=["python"],
        metavar="LANG",
        help="languages to test (default: python)",
    )
    parser.add_argument(
        "--sweep",
        nargs="+",
        type=int,
        metavar="N",
        help="sweep multiple VM counts (e.g., --sweep 5 10 20)",
    )
    parser.add_argument("--memory", type=int, default=192, metavar="MB", help="guest memory per VM (default: 192)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON summary")
    parser.add_argument(
        "--cpu-oc",
        type=float,
        default=DEFAULT_CPU_OVERCOMMIT_RATIO,
        metavar="RATIO",
        help=f"CPU overcommit ratio (default: {DEFAULT_CPU_OVERCOMMIT_RATIO})",
    )
    parser.add_argument(
        "--mem-oc",
        type=float,
        default=DEFAULT_MEMORY_OVERCOMMIT_RATIO,
        metavar="RATIO",
        help=f"memory overcommit ratio (default: {DEFAULT_MEMORY_OVERCOMMIT_RATIO})",
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================


async def main() -> None:
    args = parse_args()
    languages = [Language(lang) for lang in args.lang]
    vm_counts = args.sweep if args.sweep else [args.n]

    # System info
    vmem = psutil.virtual_memory()
    total_mb = vmem.total // (1024 * 1024)
    cpus = os.cpu_count() or 1

    print("=" * 70)
    print("  exec-sandbox VM Density Benchmark")
    print("=" * 70)
    print(f"  Host:       {_HOST_OS.value}, {cpus} CPUs, {total_mb} MB RAM")
    print(f"  VM counts:  {vm_counts}")
    print(f"  Languages:  {', '.join(LANG_DISPLAY[lang] for lang in languages)}")
    print(f"  Memory/VM:  {args.memory} MB")
    print(f"  Overcommit: CPU={args.cpu_oc}x, MEM={args.mem_oc}x")
    print(f"  Workload:   {'yes' if args.workload else 'no'}")

    # Check KSM status (Linux only)
    if _HOST_OS == HostOS.LINUX:
        try:
            ksm_run = int(Path("/sys/kernel/mm/ksm/run").read_text().strip())
            ksm_status = "enabled" if ksm_run else "disabled"
            print(f"  KSM:        {ksm_status}")
        except (OSError, ValueError):
            print("  KSM:        unavailable")

    all_results: list[DensityResult] = []

    for n_vms in vm_counts:
        for lang in languages:
            scenario = f"N={n_vms}"
            if len(languages) > 1:
                scenario += f" {LANG_DISPLAY[lang]}"

            print(f"\n  [{scenario}] Measuring density with {n_vms} {LANG_DISPLAY[lang]} VMs...")

            result = await measure_density(
                n_vms=n_vms,
                language=lang,
                scenario=scenario,
                run_workload=args.workload,
                memory_mb=args.memory,
                overcommit_cpu=args.cpu_oc,
                overcommit_mem=args.mem_oc,
            )
            all_results.append(result)
            print_result(result)

    # Comparison (if multiple measurements)
    print_comparison(all_results)

    if args.json:
        print_json_summary(all_results)

    # Final verdict
    print(f"\n{'=' * 70}")
    if all_results:
        best = min(all_results, key=lambda r: r.marginal_mb_per_vm if r.marginal_mb_per_vm > 0 else float("inf"))
        print(f"  BEST DENSITY: {best.scenario} at {best.marginal_mb_per_vm:.1f} MB/VM marginal cost")
        if best.estimated_max_vms:
            print(f"  ESTIMATED MAX: ~{best.estimated_max_vms} VMs on this host ({total_mb} MB RAM)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

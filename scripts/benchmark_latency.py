#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["exec-sandbox"]
# ///
"""Benchmark VM latency for exec-sandbox.

Measures:
- Cold boot latency: Time to create a fresh VM and execute code
- Warm pool latency: Time with pre-warmed VMs (sequential and concurrent)

Uses TimingBreakdown from exec-sandbox for detailed phase timings:
- setup_ms: Resource setup (overlay, cgroup, gvproxy)
- boot_ms: VM boot (QEMU + kernel + initramfs + guest-agent)
- execute_ms: Code execution (connect + run + response)

Usage:
    uv run python scripts/benchmark_latency.py           # Quick benchmark
    uv run python scripts/benchmark_latency.py -n 20    # More iterations
    uv run python scripts/benchmark_latency.py --pool 8 # With warm pool
"""

import argparse
import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from operator import attrgetter
from pathlib import Path

from exec_sandbox import ExecutionResult, Scheduler, SchedulerConfig
from exec_sandbox._logging import configure_logging
from exec_sandbox.constants import DEFAULT_MEMORY_MB
from exec_sandbox.models import Language

# Honor EXEC_SANDBOX_LOG_LEVEL env var (e.g. DEBUG) by wiring up a real handler
configure_logging()

# ============================================================================
# Module Constants
# ============================================================================

CODE_MAP: dict[Language, str] = {
    Language.PYTHON: "print('ok')",
    Language.JAVASCRIPT: "console.log('ok')",
    Language.RAW: "echo ok",
}

ALL_LANGUAGES: list[Language] = [Language.PYTHON, Language.JAVASCRIPT, Language.RAW]

# Display names for languages (capitalize() mangles "JavaScript" -> "Javascript")
LANG_DISPLAY: dict[Language, str] = {
    Language.PYTHON: "Python",
    Language.JAVASCRIPT: "JavaScript",
    Language.RAW: "Raw",
}

# ============================================================================
# Timing Data Collection
# ============================================================================

# (source_path on ExecutionResult, stats_attr on TimingStats)
_ALWAYS_FIELDS: tuple[tuple[str, str], ...] = (
    ("timing.total_ms", "total"),
    ("timing.setup_ms", "setup"),
    ("timing.boot_ms", "boot"),
    ("timing.execute_ms", "execute"),
    ("timing.boot_retries", "boot_retries"),
)

_OPTIONAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("execution_time_ms", "guest_exec"),
    ("timing.overlay_ms", "overlay"),
    ("timing.connect_ms", "connect"),
    ("timing.l1_restore_ms", "l1_restore"),
    ("spawn_ms", "spawn"),
    ("process_ms", "process"),
    ("timing.qemu_cmd_build_ms", "qemu_cmd_build"),
    ("timing.gvproxy_start_ms", "gvproxy_start"),
    ("timing.qemu_fork_ms", "qemu_fork"),
    ("timing.guest_wait_ms", "guest_wait"),
)


@dataclass
class TimingStats:
    """Collected timing measurements."""

    e2e: list[float] = field(default_factory=list[float])  # End-to-end (measured)
    total: list[float] = field(default_factory=list[float])  # From TimingBreakdown
    setup: list[float] = field(default_factory=list[float])
    boot: list[float] = field(default_factory=list[float])
    execute: list[float] = field(default_factory=list[float])
    guest_exec: list[float] = field(default_factory=list[float])
    # Granular setup timing
    overlay: list[float] = field(default_factory=list[float])  # Overlay acquisition (pool or on-demand)
    # Granular execute timing
    connect: list[float] = field(default_factory=list[float])  # Host: channel.connect()
    spawn: list[float] = field(default_factory=list[float])  # Guest: cmd.spawn() fork/exec
    process: list[float] = field(default_factory=list[float])  # Guest: actual process runtime
    # Granular boot timing
    qemu_cmd_build: list[float] = field(default_factory=list[float])
    gvproxy_start: list[float] = field(default_factory=list[float])
    qemu_fork: list[float] = field(default_factory=list[float])
    guest_wait: list[float] = field(default_factory=list[float])
    # L1 memory snapshot timing
    l1_restore: list[float] = field(default_factory=list[float])
    # Retry tracking
    boot_retries: list[int] = field(default_factory=list[int])
    warm_hits: int = 0
    l1_hits: int = 0
    cold_boots: int = 0


def collect_timing(result: ExecutionResult, stats: TimingStats, e2e_ms: float) -> None:
    """Extract timing info from ExecutionResult."""
    stats.e2e.append(e2e_ms)
    for source_path, stats_attr in _ALWAYS_FIELDS:
        getattr(stats, stats_attr).append(attrgetter(source_path)(result))
    for source_path, stats_attr in _OPTIONAL_FIELDS:
        value = attrgetter(source_path)(result)
        if value is not None:
            getattr(stats, stats_attr).append(value)
    if result.warm_pool_hit:
        stats.warm_hits += 1
    elif result.l1_cache_hit:
        stats.l1_hits += 1
    else:
        stats.cold_boots += 1


def _collect_results(results: list[tuple[ExecutionResult, float]]) -> TimingStats:
    """Collect timing from results, skipping failures."""
    stats = TimingStats()
    failures = 0
    for result, e2e_ms in results:
        if result.exit_code != 0:
            failures += 1
            stderr_preview = result.stderr[:200].strip()
            print(f"  Warning: run failed (exit_code={result.exit_code}): {stderr_preview}")
            continue
        collect_timing(result, stats, e2e_ms)
    if failures:
        print(f"  Warning: {failures}/{len(results)} runs failed, excluded from stats")
    return stats


# ============================================================================
# Benchmark Runner
# ============================================================================


async def benchmark_concurrent(
    scheduler: Scheduler,
    language: Language,
    concurrency: int,
    *,
    memory_mb: int | None = None,
    allow_network: bool = False,
) -> TimingStats:
    """Benchmark VM boot + execution latency with concurrent requests."""
    code = CODE_MAP.get(language, "echo ok")

    async def single_run() -> tuple[ExecutionResult, float]:
        start = time.perf_counter()
        result = await scheduler.run(
            code=code,
            language=language,
            timeout_seconds=60,
            memory_mb=memory_mb,
            allow_network=allow_network,
        )
        e2e_ms = (time.perf_counter() - start) * 1000
        return result, e2e_ms

    # Launch all requests concurrently
    results = await asyncio.gather(*[single_run() for _ in range(concurrency)])
    return _collect_results(results)


# ============================================================================
# Statistics & Display
# ============================================================================


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute the p-th percentile using the nearest-rank method (0-indexed)."""
    idx = min(math.ceil(len(sorted_vals) * p) - 1, len(sorted_vals) - 1)
    return sorted_vals[idx]


def fmt_stats(values: list[float]) -> str:
    """Format list of values as median / p95."""
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.0f}"
    sorted_vals = sorted(values)
    median = statistics.median(sorted_vals)
    p95 = _percentile(sorted_vals, 0.95)
    return f"{median:.0f} / {p95:.0f}"


# Tree breakdown tables: (label, stats_attr, annotation | None)
_MAIN_TREE: list[tuple[str, str, str | None]] = [
    ("Setup", "setup", None),
    ("Boot", "boot", None),
    ("Execute", "execute", None),
]

_SETUP_BREAKDOWN: list[tuple[str, str, str | None]] = [
    ("Overlay", "overlay", "pool hit <1ms"),
    ("L1 restore", "l1_restore", "memory snapshot"),
]

_BOOT_BREAKDOWN: list[tuple[str, str, str | None]] = [
    ("Pre-launch", "qemu_cmd_build", None),
    ("gvproxy", "gvproxy_start", None),
    ("QEMU fork", "qemu_fork", None),
    ("Guest wait", "guest_wait", "kernel+agent"),
]

_EXEC_BREAKDOWN: list[tuple[str, str, str | None]] = [
    ("Connect", "connect", None),
    ("Spawn", "spawn", None),
    ("Process", "process", None),
]


def _render_tree(
    indent: str,
    rows: list[tuple[str, str, str | None]],
    stats: TimingStats,
) -> None:
    """Render a tree breakdown, auto-computing glyphs from filtered list."""
    active = [(label, attr, ann) for label, attr, ann in rows if getattr(stats, attr)]
    if not active:
        return
    col_width = max(len(label) for label, _, _ in active) + 2  # label + colon + space
    for i, (label, attr, ann) in enumerate(active):
        is_last = i == len(active) - 1
        glyph = "└─" if is_last else "├─"
        annotation = f"  ← {ann}" if ann else ""
        print(f"{indent}{glyph} {label + ':':<{col_width}}{fmt_stats(getattr(stats, attr))} ms{annotation}")


def print_stats(name: str, stats: TimingStats) -> None:
    """Print timing statistics (all metrics are per-VM)."""
    if not stats.e2e:
        print(f"\n{name}: No data")
        return

    n = len(stats.e2e)
    parts: list[str] = []
    if stats.warm_hits:
        parts.append(f"{stats.warm_hits} warm")
    if stats.l1_hits:
        parts.append(f"{stats.l1_hits} L1")
    if stats.cold_boots:
        parts.append(f"{stats.cold_boots} cold")
    source_info = f" [{', '.join(parts)}]" if parts else ""
    print(f"\n{name} ({n} VMs concurrent){source_info}:")
    print("  Per-VM latency (median / p95):")
    print(f"    E2E:        {fmt_stats(stats.e2e)} ms")
    _render_tree("    ", _MAIN_TREE, stats)

    if any(getattr(stats, attr) for _, attr, _ in _SETUP_BREAKDOWN):
        print("  Setup breakdown:")
        _render_tree("       ", _SETUP_BREAKDOWN, stats)

    if any(getattr(stats, attr) for _, attr, _ in _BOOT_BREAKDOWN):
        print("  Boot breakdown:")
        _render_tree("       ", _BOOT_BREAKDOWN, stats)

    if any(getattr(stats, attr) for _, attr, _ in _EXEC_BREAKDOWN):
        print("  Execute breakdown:")
        _render_tree("       ", _EXEC_BREAKDOWN, stats)

    if stats.guest_exec:
        print(f"    Guest time: {fmt_stats(stats.guest_exec)} ms")

    # Retry stats
    total_retries = sum(stats.boot_retries)
    retried_count = sum(1 for r in stats.boot_retries if r > 0)
    print(f"  Boot retries: {total_retries} total ({retried_count}/{len(stats.boot_retries)} VMs needed retry)")


def print_comparison(cold_boot_results: dict[Language, TimingStats]) -> None:
    """Print hyperfine-style comparison summary table."""
    entries: list[tuple[Language, float, float]] = []
    for lang, stats in cold_boot_results.items():
        if not stats.e2e:
            continue
        sorted_vals = sorted(stats.e2e)
        median = statistics.median(sorted_vals)
        p95 = _percentile(sorted_vals, 0.95)
        entries.append((lang, median, p95))

    if len(entries) < 2:
        return

    # Sort by median ascending
    entries.sort(key=lambda x: x[1])
    min_median = entries[0][1]

    print("\nSummary (cold boot, median E2E):")
    print(f"  {'Language':<14} {'Median':>9}  {'p95':>9}  {'Relative':>8}")
    for lang, median, p95 in entries:
        relative = median / min_median
        print(f"  {LANG_DISPLAY[lang]:<14} {f'{median:.0f} ms':>9}  {f'{p95:.0f} ms':>9}  {f'{relative:.2f}x':>8}")


# ============================================================================
# CLI & Orchestration
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark VM latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                            Quick benchmark (10 iterations, cold boot only)
  %(prog)s -n 20                      More iterations for stable results
  %(prog)s --pool 8                   Enable warm pool with 8 pre-warmed VMs
  %(prog)s --network                  Benchmark with network enabled (gvproxy overhead)
  %(prog)s --lang javascript          Benchmark JS only
  %(prog)s --lang python javascript   Compare Python vs JS
  %(prog)s -n 20 --pool 8             Full benchmark
""",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        metavar="N",
        help="iterations per benchmark (default: 10)",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=0,
        metavar="SIZE",
        help="warm pool size, 0=disabled (default: 0)",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="enable network access for VMs (tests gvproxy overhead)",
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        choices=["python", "javascript", "raw"],
        default=None,
        metavar="LANG",
        help="languages to benchmark: python, javascript, raw (default: all)",
    )
    args = parser.parse_args()

    # Convert --lang strings to Language enum
    if args.lang is None:
        args.langs = list(ALL_LANGUAGES)
    else:
        args.langs = [Language(lang) for lang in args.lang]

    return args


async def run_benchmarks(
    scheduler: Scheduler,
    n: int,
    pool: int,
    network: bool,
    langs: list[Language],
) -> dict[str, TimingStats]:
    """Run all benchmarks, return results dict."""
    all_results: dict[str, TimingStats] = {}

    # Cold boot benchmarks (concurrent, per-language)
    for lang in langs:
        print(f"\nRunning {LANG_DISPLAY[lang]} cold boot benchmark ({n} concurrent)...")
        all_results[f"Cold Boot ({LANG_DISPLAY[lang]})"] = await benchmark_concurrent(
            scheduler,
            lang,
            n,
            memory_mb=None,  # Use scheduler default
            allow_network=network,
        )

    # Warm pool benchmark (if enabled)
    if pool > 0:
        print("\nWaiting for warm pool to replenish...")
        await scheduler.wait_pool_ready()
        print("Warm pool ready.")

        # Use pool size as concurrency to ensure all VMs come from pool
        for lang in langs:
            print(f"\nRunning {LANG_DISPLAY[lang]} warm pool benchmark ({pool} concurrent = pool size)...")
            all_results[f"Warm Pool ({LANG_DISPLAY[lang]})"] = await benchmark_concurrent(
                scheduler,
                lang,
                pool,
                allow_network=network,
            )

    return all_results


def print_report(all_results: dict[str, TimingStats], langs: list[Language]) -> None:
    """Print full benchmark report."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name, stats in all_results.items():
        print_stats(name, stats)

    # Comparison table for cold boot results
    cold_boot_results: dict[Language, TimingStats] = {}
    for lang in langs:
        key = f"Cold Boot ({LANG_DISPLAY[lang]})"
        if key in all_results:
            cold_boot_results[lang] = all_results[key]
    print_comparison(cold_boot_results)

    print("\n" + "=" * 60)


async def main() -> None:
    """Entry point."""
    args = parse_args()

    # Determine images directory
    images_dir = Path(__file__).parent.parent / "images" / "dist"
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        print("Run 'make build-images' first.")
        return

    pool_enabled = args.pool > 0

    print("=" * 60)
    print("exec-sandbox VM Latency Benchmark")
    print("=" * 60)
    print(f"Concurrency:  {args.n}")
    if pool_enabled:
        total_warm_pool_vms = args.pool * len(args.langs)
        lang_names = " + ".join(LANG_DISPLAY[lang] for lang in args.langs)
        print(
            f"Warm pool:    {args.pool} VMs/lang x {len(args.langs)} languages ({lang_names}) = {total_warm_pool_vms} VMs"
        )
    else:
        print("Warm pool:    disabled")
    print(f"Network:      {'enabled' if args.network else 'disabled'}")
    print(f"Memory/VM:    {DEFAULT_MEMORY_MB} MB")

    config = SchedulerConfig(
        images_dir=images_dir,
        auto_download_assets=False,
        warm_pool_size=args.pool,
    )

    async with Scheduler(config) as scheduler:
        all_results = await run_benchmarks(scheduler, args.n, args.pool, args.network, args.langs)

    print_report(all_results, args.langs)


if __name__ == "__main__":
    asyncio.run(main())

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
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from exec_sandbox import ExecutionResult, Scheduler, SchedulerConfig
from exec_sandbox.models import Language


def _float_list() -> list[float]:
    return []


@dataclass
class TimingStats:
    """Collected timing measurements."""

    e2e: list[float] = field(default_factory=_float_list)  # End-to-end (measured)
    total: list[float] = field(default_factory=_float_list)  # From TimingBreakdown
    setup: list[float] = field(default_factory=_float_list)
    boot: list[float] = field(default_factory=_float_list)
    execute: list[float] = field(default_factory=_float_list)
    guest_exec: list[float] = field(default_factory=_float_list)
    # Granular timing (new)
    connect: list[float] = field(default_factory=_float_list)  # Host: channel.connect()
    spawn: list[float] = field(default_factory=_float_list)  # Guest: cmd.spawn() fork/exec
    process: list[float] = field(default_factory=_float_list)  # Guest: actual process runtime
    warm_hits: int = 0
    cold_boots: int = 0


def collect_timing(result: ExecutionResult, stats: TimingStats, e2e_ms: float) -> None:
    """Extract timing info from ExecutionResult."""
    stats.e2e.append(e2e_ms)
    stats.total.append(result.timing.total_ms)
    stats.setup.append(result.timing.setup_ms)
    stats.boot.append(result.timing.boot_ms)
    stats.execute.append(result.timing.execute_ms)
    if result.execution_time_ms is not None:
        stats.guest_exec.append(result.execution_time_ms)
    # Granular timing (new)
    if result.timing.connect_ms is not None:
        stats.connect.append(result.timing.connect_ms)
    if result.spawn_ms is not None:
        stats.spawn.append(result.spawn_ms)
    if result.process_ms is not None:
        stats.process.append(result.process_ms)
    if result.warm_pool_hit:
        stats.warm_hits += 1
    else:
        stats.cold_boots += 1


async def benchmark_cold_boot(
    scheduler: Scheduler,
    language: Language,
    concurrency: int,
) -> TimingStats:
    """Benchmark cold VM boot + execution latency with concurrent requests."""
    code_map = {
        Language.PYTHON: "print('ok')",
        Language.JAVASCRIPT: "console.log('ok')",
        Language.RAW: "echo ok",
    }
    code = code_map.get(language, "echo ok")

    async def single_run() -> tuple[ExecutionResult, float]:
        start = time.perf_counter()
        result = await scheduler.run(
            code=code,
            language=language,
            timeout_seconds=60,
        )
        e2e_ms = (time.perf_counter() - start) * 1000
        return result, e2e_ms

    # Launch all requests concurrently
    results = await asyncio.gather(*[single_run() for _ in range(concurrency)])

    stats = TimingStats()
    for result, e2e_ms in results:
        collect_timing(result, stats, e2e_ms)
    return stats


async def benchmark_warm_pool(
    scheduler: Scheduler,
    concurrency: int,
) -> TimingStats:
    """Benchmark warm pool with concurrent requests."""

    async def single_run() -> tuple[ExecutionResult, float]:
        start = time.perf_counter()
        result = await scheduler.run(
            code="print('ping')",
            language=Language.PYTHON,
            timeout_seconds=30,
        )
        e2e_ms = (time.perf_counter() - start) * 1000
        return result, e2e_ms

    results = await asyncio.gather(*[single_run() for _ in range(concurrency)])

    stats = TimingStats()
    for result, e2e_ms in results:
        collect_timing(result, stats, e2e_ms)
    return stats


def fmt_stats(values: list[float]) -> str:
    """Format list of values as median / p95."""
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.0f}"
    sorted_vals = sorted(values)
    median = statistics.median(sorted_vals)
    p95_idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
    p95 = sorted_vals[p95_idx]
    return f"{median:.0f} / {p95:.0f}"


def print_stats(name: str, stats: TimingStats) -> None:
    """Print timing statistics (all metrics are per-VM)."""
    if not stats.e2e:
        print(f"\n{name}: No data")
        return

    n = len(stats.e2e)
    warm_cold = f" [{stats.warm_hits} warm, {stats.cold_boots} cold]" if stats.warm_hits else ""
    print(f"\n{name} ({n} VMs concurrent){warm_cold}:")
    print("  Per-VM latency (median / p95):")
    print(f"    E2E:        {fmt_stats(stats.e2e)} ms")
    print(f"    ├─ Setup:   {fmt_stats(stats.setup)} ms")
    print(f"    ├─ Boot:    {fmt_stats(stats.boot)} ms")
    print(f"    └─ Execute: {fmt_stats(stats.execute)} ms")

    # Granular execute breakdown (if available)
    if stats.connect or stats.spawn or stats.process:
        print("  Execute breakdown:")
        if stats.connect:
            print(f"       ├─ Connect: {fmt_stats(stats.connect)} ms")
        if stats.spawn:
            print(f"       ├─ Spawn:   {fmt_stats(stats.spawn)} ms")
        if stats.process:
            print(f"       └─ Process: {fmt_stats(stats.process)} ms")

    if stats.guest_exec:
        print(f"    Guest time: {fmt_stats(stats.guest_exec)} ms")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark VM latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                 Quick benchmark (10 iterations, cold boot only)
  %(prog)s -n 20           More iterations for stable results
  %(prog)s --pool 8        Enable warm pool with 8 pre-warmed VMs
  %(prog)s -n 20 --pool 8  Full benchmark
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
    args = parser.parse_args()

    # Fixed config
    memory_mb = 192

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
    print(f"Warm pool:    {args.pool} VMs" if pool_enabled else "Warm pool:    disabled")
    print(f"Memory/VM:    {memory_mb} MB")

    # Configure scheduler - need enough slots for concurrent requests + warm pool
    max_vms = max(args.n + args.pool, 12)
    config = SchedulerConfig(
        images_dir=images_dir,
        auto_download_assets=False,
        warm_pool_size=args.pool,
        max_concurrent_vms=max_vms,
        default_memory_mb=memory_mb,
    )

    all_results: dict[str, TimingStats] = {}

    async with Scheduler(config) as scheduler:
        # Cold boot benchmark (concurrent)
        print(f"\nRunning cold boot benchmark ({args.n} concurrent)...")
        all_results["Cold Boot"] = await benchmark_cold_boot(
            scheduler, Language.PYTHON, args.n
        )

        # Warm pool benchmark (if enabled)
        if pool_enabled:
            wait_time = args.pool * 0.5 + 2
            print(f"\nWaiting {wait_time:.0f}s for warm pool to initialize...")
            await asyncio.sleep(wait_time)

            # Use pool size as concurrency to ensure all VMs come from pool
            print(f"\nRunning warm pool benchmark ({args.pool} concurrent = pool size)...")
            all_results["Warm Pool"] = await benchmark_warm_pool(scheduler, args.pool)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name, stats in all_results.items():
        print_stats(name, stats)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

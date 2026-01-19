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


def _int_list() -> list[int]:
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
    # Granular setup timing
    overlay: list[float] = field(default_factory=_float_list)  # Overlay acquisition (pool or on-demand)
    # Granular execute timing
    connect: list[float] = field(default_factory=_float_list)  # Host: channel.connect()
    spawn: list[float] = field(default_factory=_float_list)  # Guest: cmd.spawn() fork/exec
    process: list[float] = field(default_factory=_float_list)  # Guest: actual process runtime
    # Granular boot timing
    qemu_cmd_build: list[float] = field(default_factory=_float_list)
    gvproxy_start: list[float] = field(default_factory=_float_list)
    qemu_fork: list[float] = field(default_factory=_float_list)
    guest_wait: list[float] = field(default_factory=_float_list)
    # Retry tracking
    boot_retries: list[int] = field(default_factory=_int_list)
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
    # Granular setup timing
    if result.timing.overlay_ms is not None:
        stats.overlay.append(result.timing.overlay_ms)
    # Granular execute timing
    if result.timing.connect_ms is not None:
        stats.connect.append(result.timing.connect_ms)
    if result.spawn_ms is not None:
        stats.spawn.append(result.spawn_ms)
    if result.process_ms is not None:
        stats.process.append(result.process_ms)
    # Granular boot timing
    if result.timing.qemu_cmd_build_ms is not None:
        stats.qemu_cmd_build.append(result.timing.qemu_cmd_build_ms)
    if result.timing.gvproxy_start_ms is not None:
        stats.gvproxy_start.append(result.timing.gvproxy_start_ms)
    if result.timing.qemu_fork_ms is not None:
        stats.qemu_fork.append(result.timing.qemu_fork_ms)
    if result.timing.guest_wait_ms is not None:
        stats.guest_wait.append(result.timing.guest_wait_ms)
    # Retry tracking
    stats.boot_retries.append(result.timing.boot_retries)
    if result.warm_pool_hit:
        stats.warm_hits += 1
    else:
        stats.cold_boots += 1


async def benchmark_cold_boot(
    scheduler: Scheduler,
    language: Language,
    concurrency: int,
    *,
    allow_network: bool = False,
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
            allow_network=allow_network,
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
    *,
    allow_network: bool = False,
) -> TimingStats:
    """Benchmark warm pool with concurrent requests."""

    async def single_run() -> tuple[ExecutionResult, float]:
        start = time.perf_counter()
        result = await scheduler.run(
            code="print('ping')",
            language=Language.PYTHON,
            timeout_seconds=30,
            allow_network=allow_network,
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

    # Granular setup breakdown (if available - cold boots only)
    if stats.overlay:
        print("  Setup breakdown:")
        print(f"       └─ Overlay:    {fmt_stats(stats.overlay)} ms  ← pool hit <1ms")

    # Granular boot breakdown (if available - cold boots only)
    if stats.qemu_cmd_build or stats.gvproxy_start or stats.qemu_fork or stats.guest_wait:
        print("  Boot breakdown:")
        if stats.qemu_cmd_build:
            print(f"       ├─ Pre-launch: {fmt_stats(stats.qemu_cmd_build)} ms")
        if stats.gvproxy_start:
            print(f"       ├─ gvproxy:    {fmt_stats(stats.gvproxy_start)} ms")
        if stats.qemu_fork:
            print(f"       ├─ QEMU fork:  {fmt_stats(stats.qemu_fork)} ms")
        if stats.guest_wait:
            print(f"       └─ Guest wait: {fmt_stats(stats.guest_wait)} ms  ← kernel+agent")

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

    # Retry stats
    total_retries = sum(stats.boot_retries)
    retried_count = sum(1 for r in stats.boot_retries if r > 0)
    print(f"  Boot retries: {total_retries} total ({retried_count}/{len(stats.boot_retries)} VMs needed retry)")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark VM latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                   Quick benchmark (10 iterations, cold boot only)
  %(prog)s -n 20             More iterations for stable results
  %(prog)s --pool 8          Enable warm pool with 8 pre-warmed VMs
  %(prog)s --network         Benchmark with network enabled (gvproxy overhead)
  %(prog)s -n 20 --pool 8    Full benchmark
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
    args = parser.parse_args()

    # Fixed config - use lower memory for benchmarking many VMs
    memory_mb = 128

    # Determine images directory
    images_dir = Path(__file__).parent.parent / "images" / "dist"
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        print("Run 'make build-images' first.")
        return

    pool_enabled = args.pool > 0

    print("=" * 60)
    print("exec-sandbox VM Latency Benchmark")
    # Warm pool is per-language (Python + JavaScript = 2 languages)
    num_languages = 2
    total_warm_pool_vms = args.pool * num_languages if pool_enabled else 0

    print("=" * 60)
    print(f"Concurrency:  {args.n}")
    print(
        f"Warm pool:    {args.pool} VMs/lang x {num_languages} languages = {total_warm_pool_vms} VMs"
        if pool_enabled
        else "Warm pool:    disabled"
    )
    print(f"Network:      {'enabled' if args.network else 'disabled'}")
    print(f"Memory/VM:    {memory_mb} MB")

    # Configure scheduler - need enough slots for concurrent requests + warm pool (per-language)
    max_vms = max(args.n + total_warm_pool_vms, 12)
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
            scheduler, Language.PYTHON, args.n, allow_network=args.network
        )

        # Warm pool benchmark (if enabled)
        if pool_enabled:
            # Scale wait time more aggressively for larger pools
            wait_time = args.pool * 1.0 + 5
            print(f"\nWaiting {wait_time:.0f}s for warm pool to initialize...")
            await asyncio.sleep(wait_time)

            # Use pool size as concurrency to ensure all VMs come from pool
            print(f"\nRunning warm pool benchmark ({args.pool} concurrent = pool size)...")
            all_results["Warm Pool"] = await benchmark_warm_pool(scheduler, args.pool, allow_network=args.network)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name, stats in all_results.items():
        print_stats(name, stats)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

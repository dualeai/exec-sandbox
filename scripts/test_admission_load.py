#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["exec-sandbox"]
# ///
"""Load test for resource-aware admission controller.

Fires 100 concurrent VM requests against the scheduler to verify:
- Admission controller backpressure works (no OOM)
- All requests eventually complete (no deadlocks)
- No VmCapacityError leaks through (admission blocks, not rejects)

Usage:
    uv run python scripts/test_admission_load.py
"""

import asyncio
import time

from exec_sandbox import Scheduler, SchedulerConfig
from exec_sandbox.models import Language

TOTAL_VMS = 100
MAX_CONCURRENT = 10  # Overlay pool sizing
CODE = """\
import os, time
start = time.monotonic()
# CPU-bound work: compute primes to keep VM alive ~2-3s
n = 0
for i in range(2, 30_000):
    if all(i % j != 0 for j in range(2, int(i**0.5) + 1)):
        n += 1
elapsed = time.monotonic() - start
print(f'VM ok, pid={os.getpid()}, primes={n}, elapsed={elapsed:.1f}s')
"""


async def main() -> None:
    config = SchedulerConfig(
        warm_pool_size=0,
        default_memory_mb=256,
        default_timeout_seconds=60,
        # Use defaults for overcommit (1.5x memory, 4x CPU)
    )

    print(f"Launching {TOTAL_VMS} concurrent VMs")
    print("Memory: 256MB/VM, overcommit=1.5x, CPU overcommit=4.0x")
    print()

    succeeded = 0
    failed = 0
    errors: list[str] = []

    async with Scheduler(config) as scheduler:
        # Print admission snapshot
        snap = scheduler._vm_manager.admission.snapshot()
        print(f"Host: {snap.host_memory_mb:.0f}MB RAM, {snap.host_cpu_count:.1f} CPUs (source: {snap.capacity_source})")
        print(f"Budget: {snap.memory_budget_mb:.0f}MB memory, {snap.cpu_budget:.1f} CPU cores")
        if snap.available_memory_floor_mb > 0:
            avail = f"{snap.system_available_memory_mb:.0f}MB" if snap.system_available_memory_mb is not None else "unknown"
            print(f"Floor: {snap.available_memory_floor_mb}MB (current available: {avail})")
        print()

        start = time.perf_counter()

        async def run_one(i: int) -> None:
            nonlocal succeeded, failed
            try:
                result = await scheduler.run(
                    code=CODE,
                    language=Language.PYTHON,
                    timeout_seconds=120,
                )
                if result.exit_code == 0:
                    succeeded += 1
                else:
                    failed += 1
                    errors.append(f"VM {i}: exit_code={result.exit_code}, stderr={result.stderr[:100]}")
            except Exception as e:
                failed += 1
                errors.append(f"VM {i}: {type(e).__name__}: {e}")

        # Fire all 100 concurrently
        tasks = [asyncio.create_task(run_one(i)) for i in range(TOTAL_VMS)]

        # Progress tracking
        while not all(t.done() for t in tasks):
            done = sum(1 for t in tasks if t.done())
            snap = scheduler._vm_manager.admission.snapshot()
            print(
                f"\r  Progress: {done}/{TOTAL_VMS} done | "
                f"Active VMs: {snap.allocated_vm_slots} | "
                f"Memory: {snap.allocated_memory_mb:.0f}/{snap.memory_budget_mb:.0f}MB | "
                f"CPU: {snap.allocated_cpu:.1f}/{snap.cpu_budget:.1f}",
                end="",
                flush=True,
            )
            await asyncio.sleep(1)

        # Wait for all to finish
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    print(f"\r{'':80}")  # Clear progress line
    print(f"Results:")
    print(f"  Succeeded: {succeeded}/{TOTAL_VMS}")
    print(f"  Failed:    {failed}/{TOTAL_VMS}")
    print(f"  Total:     {elapsed:.1f}s")
    print(f"  Throughput: {TOTAL_VMS / elapsed:.1f} VMs/s")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    asyncio.run(main())

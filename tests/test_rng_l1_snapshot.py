"""Regression test: L1 snapshot restore must produce unique RNG output per VM.

QEMU migration replays exact CPU+RAM+device state, including the kernel CRNG.
Without the guest-agent's RNDRESEEDCRNG ioctl (called before every command
dispatch), all VMs restored from the same vmstate would produce identical
os.urandom() output — a critical security vulnerability.

This test forces L1 restore for multiple VMs and asserts their random output
diverges, validating that the reseed_crng() fix in guest-agent/src/connection.rs
works correctly.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# Code that reads urandom IMMEDIATELY — before any entropy can accumulate
# from CPU jitter / interrupts / etc.
IMMEDIATE_URANDOM = """\
import os, hashlib
# Read immediately on restore — minimal time for new entropy
data = os.urandom(1024)
print(hashlib.sha256(data).hexdigest())
"""


@pytest.fixture
async def l1_scheduler(images_dir: Path) -> AsyncGenerator[Scheduler, None]:
    """Scheduler with L1 memory snapshots enabled, warm pool disabled.

    Disabling the warm pool forces the scheduler to use cold boot → L1 cache path,
    making it deterministic which VMs come from L1 restore.
    """
    config = SchedulerConfig(
        images_dir=images_dir,
        auto_download_assets=False,
        warm_pool_size=0,  # Disable warm pool — force cold boot / L1 path
    )
    async with Scheduler(config) as sched:
        yield sched


class TestL1SnapshotRNG:
    """Verify RNG uniqueness after L1 memory snapshot restore.

    The L1 snapshot saves full CPU+RAM+device state via QEMU migration.
    On restore, the kernel CRNG resumes from the saved state. The guest-agent
    forces an immediate CRNG reseed (RNDRESEEDCRNG ioctl) before every command
    dispatch, ensuring each restored VM diverges. These tests validate that
    the reseed mechanism works — failure means identical random output across
    VMs, indicating a broken or missing reseed.
    """

    async def test_l1_restored_vms_different_random(self, l1_scheduler: Scheduler) -> None:
        """Two VMs restored from the same L1 snapshot must produce different random."""
        # Step 1: Prime the L1 cache with a cold boot
        prime_result = await l1_scheduler.run(code="print('prime')", language=Language.PYTHON)
        assert prime_result.exit_code == 0

        # Step 2: Wait for background L1 save to complete
        # The scheduler schedules a background save after first cold boot.
        # We need it to finish before the next calls can hit L1 cache.
        assert l1_scheduler._memory_snapshot_manager is not None
        for task in list(l1_scheduler._memory_snapshot_manager._in_flight_saves.values()):
            await task

        # Step 3: Run two VMs concurrently — both should restore from L1
        results = await asyncio.gather(
            l1_scheduler.run(code=IMMEDIATE_URANDOM, language=Language.PYTHON),
            l1_scheduler.run(code=IMMEDIATE_URANDOM, language=Language.PYTHON),
        )

        # Verify both came from L1 restore
        for i, r in enumerate(results):
            assert r.exit_code == 0, f"VM {i} failed: {r.stderr}"
            assert r.l1_cache_hit, f"VM {i} did NOT come from L1 restore (test is invalid without L1)"

        # Step 4: Assert RNG uniqueness — this is the bug reproduction
        hashes = [r.stdout.strip() for r in results]
        assert all(len(h) == 64 for h in hashes), f"Invalid SHA256 output: {hashes}"
        assert hashes[0] != hashes[1], (
            f"CRITICAL: L1-restored VMs produced identical random output!\n"
            f"  VM 0: {hashes[0]}\n"
            f"  VM 1: {hashes[1]}\n"
            f"Kernel CRNG state was cloned — no post-restore entropy re-seeding."
        )

    async def test_l1_restored_vms_sequential_different_random(self, l1_scheduler: Scheduler) -> None:
        """Even sequential L1 restores must produce different random — no state replay."""
        # Step 1: Prime L1 cache
        prime_result = await l1_scheduler.run(code="print('prime')", language=Language.PYTHON)
        assert prime_result.exit_code == 0

        # Step 2: Wait for L1 save
        assert l1_scheduler._memory_snapshot_manager is not None
        for task in list(l1_scheduler._memory_snapshot_manager._in_flight_saves.values()):
            await task

        # Step 3: Run 3 VMs sequentially from L1
        hashes: list[str] = []
        for i in range(3):
            result = await l1_scheduler.run(code=IMMEDIATE_URANDOM, language=Language.PYTHON)
            assert result.exit_code == 0, f"VM {i} failed: {result.stderr}"
            assert result.l1_cache_hit, f"VM {i} did NOT come from L1 restore"
            hashes.append(result.stdout.strip())

        # All three must be unique
        assert len(set(hashes)) == 3, (
            f"CRITICAL: L1-restored VMs produced duplicate random output!\n"
            f"  Hashes: {hashes}\n"
            f"Kernel CRNG state was replayed from snapshot."
        )

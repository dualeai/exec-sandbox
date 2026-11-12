"""A damaged L1 entry: the request cold boots, the entry goes, and a new one replaces it.

The L1 check reads the vmstate digest only. A RAM template that lost its blocks
(a power loss before the save flushed it) restores a guest that never answers.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once, run_timed, skip_unless_hwaccel, wait_l1_saves


@pytest.fixture
async def l1_scheduler(images_dir: Path, tmp_path: Path) -> AsyncGenerator[Scheduler]:
    """L1 on in its own folder, no warm pool: the second request takes the L1 path."""
    config = SchedulerConfig(
        images_dir=images_dir,
        auto_download_assets=False,
        warm_pool_size=0,
        default_timeout_seconds=120,
        memory_snapshot_cache_dir=tmp_path / "l1",
    )
    async with Scheduler(config) as sched:
        yield sched


@skip_unless_hwaccel
async def test_zeroed_ram_template_cold_boots_and_rebuilds_the_entry(l1_scheduler: Scheduler, tmp_path: Path) -> None:
    spec = VmSpec(language=Language.PYTHON)
    assert (await run_once(l1_scheduler, spec, "print('prime')")).exit_code == 0
    await wait_l1_saves(l1_scheduler)
    (ram,) = (tmp_path / "l1").glob("*.ram")
    size = ram.stat().st_size
    with ram.open("r+b") as f:  # every block lost, the published entry kept
        f.truncate(0)
        f.truncate(size)
    assert ram.stat().st_blocks == 0  # the damaged template holds no data

    damaged = await run_timed(l1_scheduler, spec, "print('after')")

    assert (damaged.result.exit_code, damaged.result.stdout.strip()) == (0, "after")
    assert damaged.startup.l1_cache_hit is False  # the restore failed: a cold boot ran the code
    await wait_l1_saves(l1_scheduler)  # the cold boot saves a new entry in the background
    (rebuilt,) = (tmp_path / "l1").glob("*.ram")
    assert rebuilt.stat().st_blocks > 0  # the damaged entry was removed, not kept

    again = await run_timed(l1_scheduler, spec, "print('again')")

    assert (again.result.exit_code, again.result.stdout.strip()) == (0, "again")
    assert again.startup.l1_cache_hit is True

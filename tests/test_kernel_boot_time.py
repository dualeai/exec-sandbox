"""A cold boot reaches the guest agent in well under a second of guest time.

Guards against built-in drivers that wait for hardware microvm lacks (see
tests/test_kernel_config.py): with them, an x86 KVM cold boot waits ~5 s before /init.
"""

import re
from pathlib import Path

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import skip_unless_hwaccel

_AGENT_START = re.compile(r"\[timing\] agent_start: (\d+)ms \(monotonic\)")


@skip_unless_hwaccel
async def test_cold_boot_reaches_agent_in_under_a_second(images_dir: Path, tmp_path: Path) -> None:
    """Guest kernel time from power-on to the agent, read from the agent's own log line."""
    config = SchedulerConfig(
        images_dir=images_dir,
        auto_download_assets=False,
        warm_pool_size=0,
        memory_snapshot_cache_dir=tmp_path / "l1",  # empty: a cold boot
    )
    lines: list[str] = []
    async with Scheduler(config) as scheduler:
        async with await scheduler.new(VmSpec(language=Language.RAW), on_boot_log=lines.append) as session:
            assert session.startup_timing.l1_cache_hit is False

    starts = [int(m.group(1)) for line in lines if (m := _AGENT_START.search(line))]
    assert starts, "no agent_start line in the boot log"
    assert starts[-1] < 1000

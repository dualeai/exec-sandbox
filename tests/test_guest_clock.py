"""Guest wall clock follows the host after cold boot, L1 restore and pause.

Every request carries host_time_ns; the guest steps CLOCK_REALTIME forward when
it lags by more than 100 ms (guest-agent/src/clock.rs). Without it, a VM restored
from an L1 template saved N seconds ago starts N seconds in the past.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.migration_client import MigrationClient
from exec_sandbox.models import Language
from exec_sandbox.permission_utils import get_expected_socket_uid
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once, run_timed, skip_unless_hwaccel, wait_l1_saves

# Forward steps happen above 100 ms of lag; allow scheduling noise on top.
_TOLERANCE_NS = 150_000_000
_READ_TIME = "import time; print(time.time_ns())"


def _assert_in_window(guest_ns: int, before_ns: int, after_ns: int) -> None:
    assert before_ns - _TOLERANCE_NS <= guest_ns <= after_ns + _TOLERANCE_NS, (
        f"guest clock off by {(guest_ns - before_ns) / 1e6:.0f} ms (request took {(after_ns - before_ns) / 1e6:.0f} ms)"
    )


@pytest.fixture
async def l1_scheduler(images_dir: Path) -> AsyncGenerator[Scheduler, None]:
    config = SchedulerConfig(images_dir=images_dir, auto_download_assets=False, warm_pool_size=0)
    async with Scheduler(config) as scheduler:
        yield scheduler


@skip_unless_hwaccel
class TestGuestClock:
    async def test_cold_boot(self, l1_scheduler: Scheduler) -> None:
        before = time.time_ns()
        result = await run_once(l1_scheduler, VmSpec(language=Language.PYTHON), _READ_TIME)
        after = time.time_ns()
        assert result.exit_code == 0, result.stderr
        _assert_in_window(int(result.stdout.strip()), before, after)

    async def test_l1_restore_of_old_template(self, l1_scheduler: Scheduler) -> None:
        """A template saved seconds ago restores with the host's time, not the save time."""
        assert (await run_once(l1_scheduler, VmSpec(language=Language.PYTHON), "print('prime')")).exit_code == 0
        await wait_l1_saves(l1_scheduler)
        await asyncio.sleep(3)

        before = time.time_ns()
        result, startup, _ports = await run_timed(l1_scheduler, VmSpec(language=Language.PYTHON), _READ_TIME)
        after = time.time_ns()

        assert result.exit_code == 0, result.stderr
        assert startup.l1_cache_hit, "test needs an L1 restore"
        _assert_in_window(int(result.stdout.strip()), before, after)

    async def test_pause_and_resume(self, l1_scheduler: Scheduler) -> None:
        async with await l1_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            vm = session._vm
            assert (await session.exec("x = 1")).exit_code == 0
            async with MigrationClient(vm.qmp_socket, get_expected_socket_uid(vm.use_qemu_vm_user)) as qmp:
                await qmp._execute("stop", arguments=None, timeout=5.0)
                await asyncio.sleep(3)
                await qmp._execute("cont", arguments=None, timeout=5.0)

            before = time.time_ns()
            result = await session.exec(_READ_TIME)
            after = time.time_ns()

        assert result.exit_code == 0, result.stderr
        _assert_in_window(int(result.stdout.strip()), before, after)

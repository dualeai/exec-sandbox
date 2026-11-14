"""The Python REPL's own modules stay out of the user's namespace, on a cold VM and on a
warm-pool VM alike: code that works on one works on the other. Requires VM."""

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_timed

_CODE = "print(sorted(n for n in ('multiprocessing', 'traceback', 'typing') if n in globals()))"


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm-pool"])
async def test_repl_modules_are_not_user_names(scheduler_config: SchedulerConfig, warm: bool) -> None:
    """None of the modules the REPL loads late (or the warm-up loads early) is a name in the
    user's namespace, whatever path made the VM."""
    config = SchedulerConfig(
        images_dir=scheduler_config.images_dir, warm_pool_size=1 if warm else 0, auto_download_assets=False
    )
    async with Scheduler(config) as scheduler:
        result, startup, _ports = await run_timed(scheduler, VmSpec(language=Language.PYTHON), _CODE)

    assert startup.warm_pool_hit is warm
    assert (result.exit_code, result.stdout.strip()) == (0, "[]"), result.stderr


async def test_user_names_survive_between_execs(scheduler: Scheduler) -> None:
    """Names that the REPL's own loop also uses (code, header, exit_code) and `traceback` stay
    the user's across execs, an error included."""
    async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
        await session.exec("code = 42; header = 'mine'; exit_code = 7; traceback = 'mine'")
        failed = await session.exec("1 / 0")
        result = await session.exec("print(code, header, exit_code, traceback)")

    assert failed.exit_code == 1
    assert (result.exit_code, result.stdout.strip()) == (0, "42 mine 7 mine"), result.stderr


async def test_a_replaced_sys_stderr_does_not_hide_the_end_of_an_exec(scheduler: Scheduler) -> None:
    """User code replaces sys.stderr: the REPL still ends each exec (its marker goes through
    the original stream), and the next exec runs."""
    async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
        replaced = await session.exec("import io, sys; sys.stderr = io.StringIO()", timeout_seconds=20)
        result = await session.exec("print('ok')", timeout_seconds=20)

    assert replaced.exit_code == 0
    assert (result.exit_code, result.stdout.strip()) == (0, "ok"), result.stderr

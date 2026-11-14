"""Tests for multiprocessing.Pool inside the VM.

Verifies that multiprocessing works in the Python REPL, which requires:
1. /dev/shm mounted as tmpfs (POSIX semaphores for SemLock)
2. cloudpickle patching multiprocessing's pickler (exec()'d functions, lambdas, closures)
3. fork start method (Python 3.14 defaults to forkserver, which hangs in the VM)

References:
- /dev/shm requirement: https://github.com/moby/moby/issues/1683
- Python 3.14 forkserver default: https://docs.python.org/3/whatsnew/3.14.html
- cloudpickle serialization: https://github.com/cloudpipe/cloudpickle
"""

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once


class TestMultiprocessingPool:
    """Verify multiprocessing.Pool works inside the VM via REPL."""

    async def test_pool_map_named_function(self, scheduler: Scheduler) -> None:
        """Pool.map with a named function defined in the REPL (exec() path)."""
        code = """\
from multiprocessing import Pool

def square(x):
    return x ** 2

with Pool(processes=2) as pool:
    result = pool.map(square, [1, 2, 3, 4, 5])
print(result)
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.strip() == "[1, 4, 9, 16, 25]"

    @pytest.mark.parametrize(
        "make_pool",
        [
            "from multiprocessing.pool import Pool\npool = Pool(processes=2)",
            "import multiprocessing\npool = multiprocessing.get_context('fork').Pool(processes=2)",
            "from concurrent.futures import ProcessPoolExecutor\npool = ProcessPoolExecutor(max_workers=2)",
        ],
        ids=["pool-module", "fork-context", "process-pool-executor"],
    )
    async def test_every_pool_form_takes_a_lambda(self, scheduler: Scheduler, make_pool: str) -> None:
        """A lambda reaches the workers of every pool form, as the first multiprocessing use of
        the session: each form starts processes through BaseProcess.start, which turns on
        cloudpickle."""
        code = f"""\
{make_pool}
with pool:
    print(list(pool.map(lambda x: x ** 2, [1, 2, 3])))
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.strip() == "[1, 4, 9]"

    async def test_dev_shm_is_tmpfs(self, scheduler: Scheduler) -> None:
        """/dev/shm must be mounted as tmpfs for POSIX semaphores. The last mount on a path is
        the one that processes see (the read-only bind of /dev repeats the mount)."""
        code = """\
with open('/proc/mounts') as mounts:
    print([line.split()[2] for line in mounts if line.split()[1] == '/dev/shm'][-1:])
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.strip() == "['tmpfs']"

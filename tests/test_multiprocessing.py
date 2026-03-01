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

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


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
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.strip() == "[1, 4, 9, 16, 25]"

    async def test_pool_map_lambda(self, scheduler: Scheduler) -> None:
        """Pool.map with a lambda — requires cloudpickle (standard pickle can't serialize lambdas)."""
        code = """\
from multiprocessing import Pool

with Pool(processes=2) as pool:
    result = pool.map(lambda x: x ** 2, [1, 2, 3])
print(result)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.strip() == "[1, 4, 9]"

    async def test_dev_shm_exists(self, scheduler: Scheduler) -> None:
        """/dev/shm must be mounted as tmpfs for POSIX semaphores."""
        code = """\
import os
print(os.path.isdir('/dev/shm'))
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout.strip() == "True"

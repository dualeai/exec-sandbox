"""Python bytecode: the REPL reads the image's precompiled stdlib, and L2
package layers ship their own .pyc (the layer is read-only at run time).

With PYTHONPYCACHEPREFIX set, CPython reads .pyc only under the prefix and
recompiled ~56 stdlib modules into /home/user at every REPL start.
"""

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once, skip_unless_hwaccel


@skip_unless_hwaccel
class TestPythonBytecode:
    async def test_repl_uses_image_bytecode(self, scheduler: Scheduler) -> None:
        code = "import os, sys; print(sys.pycache_prefix, os.path.exists('/home/user/.pycache'))"
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "None False"

    @pytest.mark.slow
    async def test_package_layer_ships_bytecode(self, scheduler: Scheduler) -> None:
        code = "import os, importlib.util, six; print(os.path.exists(importlib.util.cache_from_source(six.__file__)))"
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON, packages=["six==1.17.0"]), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "True"

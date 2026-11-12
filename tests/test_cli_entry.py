"""The sbx console script without the cli extra: a one-line install hint, never a traceback.

A module the extra does not provide is not hidden behind the hint: its import error stays.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_BLOCK_AND_RUN = """
import sys
from importlib.abc import MetaPathFinder

class _Missing(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == {missing!r}:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None

sys.meta_path.insert(0, _Missing())
sys.argv = ["sbx", *{args!r}]
from exec_sandbox._imports import cli_main
cli_main()
"""


def _sbx_without(missing: str, *args: str, home: Path) -> subprocess.CompletedProcess[str]:
    """`sbx ARGS` in a new interpreter where `missing` (and its submodules) cannot be imported.
    No EXEC_SANDBOX_* variable reaches it, the keyring package may only use its fail backend,
    and HOME and XDG_DATA_HOME are `home`: the user folders (the local store's default
    folder among them) resolve under it, never under the developer's home."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("EXEC_SANDBOX_")}
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    env["HOME"] = env["XDG_DATA_HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-I", "-c", _BLOCK_AND_RUN.format(missing=missing, args=args)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )


def test_missing_click_prints_the_install_hint(tmp_path: Path) -> None:
    """The package itself imports without the extra; the script names what is missing and what to install."""
    done = _sbx_without("click", "--version", home=tmp_path)
    assert (done.returncode, done.stderr) == (1, "The exec-sandbox CLI needs click: pip install 'exec-sandbox[cli]'\n")


@pytest.mark.parametrize("missing", ["cryptography", "keyring"])
def test_start_imports_neither_keyring_nor_cryptography(missing: str, tmp_path: Path) -> None:
    """Only the OS keyring step of the local store needs them: other commands start without
    their import time (about 20 ms at every start)."""
    done = _sbx_without(missing, "--version", home=tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("exec-sandbox, version ")


@pytest.mark.parametrize("missing", ["cryptography", "keyring"])
def test_os_keyring_step_names_the_missing_extra(missing: str, tmp_path: Path) -> None:
    """`sbx instance list` with the local store opens the OS keyring: the missing module gets
    the install hint as a setup error (EXTRA_MISSING, FIX_CONFIG: exit 2), before any keyring
    or data folder is used (nothing is made under the child's home)."""
    done = _sbx_without(missing, "instance", "list", home=tmp_path)
    assert done.returncode == 2, done.stderr
    assert "EXTRA_MISSING (FIX_CONFIG)" in done.stderr
    assert f"The exec-sandbox CLI needs {missing}: pip install 'exec-sandbox[cli]'" in done.stderr
    assert "Traceback" not in done.stderr
    assert list(tmp_path.iterdir()) == []


def test_other_missing_module_keeps_its_error(tmp_path: Path) -> None:
    """termios (standard library, imported by the CLI only) is not in the cli extra: no install hint."""
    done = _sbx_without("termios", "--version", home=tmp_path)
    assert done.returncode == 1
    assert "ModuleNotFoundError: No module named 'termios'" in done.stderr
    assert "exec-sandbox[cli]" not in done.stderr

"""`sbx run --instance` and `sbx instance` end to end: real VMs, a local store in tmp_path.

Each command goes through click's CliRunner, as from the console script: one
Scheduler for each command, so the state crosses a full stop and start. The key
comes from EXEC_SANDBOX_INSTANCE_KEYRING: the OS keyring is never used (that
the variable wins is tests/test_cli.py's fact).
"""

import base64
import json
import secrets
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from exec_sandbox import InstanceInfo, InstanceRef, Language, VmSpec
from exec_sandbox.cli import EXIT_CLI_ERROR, EXIT_SANDBOX_ERROR, OneShotRun, cli
from tests.conftest import skip_unless_hwaccel
from tests.keyring_support import isolate_account

pytestmark = skip_unless_hwaccel

_ID = "my-vm"


@pytest.fixture
def store_dir(tmp_path: Path, images_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The CLI's local store in tmp_path, its key in the environment, the built images.

    The OS account is isolated (its home in tmp_path, its OS keyring a dict); after the
    test, neither was used: the key comes from the variable."""
    root = tmp_path / "instances"
    monkeypatch.delenv("EXEC_SANDBOX_INSTANCE_STORE", raising=False)
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE_DIR", str(root))
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", base64.b64encode(secrets.token_bytes(32)).decode())
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_ASSET_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("EXEC_SANDBOX_IMAGES_DIR", str(images_dir))
    account = isolate_account(monkeypatch, home=tmp_path / "account", key=None)
    yield root
    account.assert_untouched()


def _sbx(*args: str) -> Result:
    return CliRunner().invoke(cli, list(args))


def _describe() -> InstanceInfo:
    described = _sbx("instance", "describe", _ID)
    assert described.exit_code == 0, described.output
    return InstanceInfo.model_validate_json(described.stdout)


def test_same_command_twice_continues_the_instance(store_dir: Path) -> None:
    """A Python variable and a /home/user file made by the first run are there in the second."""
    first = _sbx(
        "run", "--instance", _ID, "-c", "import pathlib; x = 41; pathlib.Path('/home/user/note.txt').write_text('kept')"
    )
    assert first.exit_code == 0, first.output
    assert f"Created instance {_ID}, generation 1" in first.stderr
    assert store_dir.is_dir()

    second = _sbx("run", "--instance", _ID, "--json", "-c", "x += 1; print(x, open('/home/user/note.txt').read())")
    assert second.exit_code == 0, second.output
    run = OneShotRun.model_validate_json(second.stdout)
    assert run.result.stdout == "42 kept\n"
    assert run.instance == InstanceRef(instance_id=_ID, generation=2)
    assert run.startup.instance_load_ms is not None  # reuse(), not a boot
    assert f"Saved instance {_ID}, generation 2" in second.stderr


def test_list_describe_rm(store_dir: Path) -> None:
    made = _sbx("run", "--instance", _ID, "-q", "-c", "y = 1")
    assert made.exit_code == 0, made.output

    listed = _sbx("instance", "list")
    assert listed.exit_code == 0, listed.output
    assert listed.stdout.splitlines() == [_ID]
    assert json.loads(_sbx("instance", "list", "--json").stdout) == [_ID]

    info = _describe()
    assert (info.instance_id, info.generation, info.verified) == (_ID, 1, True)
    assert info.spec == VmSpec(language=Language.PYTHON)

    removed = _sbx("instance", "rm", _ID)
    assert removed.exit_code == 0, removed.output
    assert removed.stdout == f"{_ID}\n"
    assert _sbx("instance", "list").stdout == ""

    again = _sbx("instance", "rm", _ID)  # already gone: still exit 0
    assert again.exit_code == 0, again.output
    assert f"No instance {_ID}" in again.stderr

    gone = _sbx("instance", "describe", _ID)
    assert gone.exit_code == EXIT_SANDBOX_ERROR
    assert "INSTANCE_NOT_FOUND (RECREATE)" in gone.stderr


def test_other_vm_settings_exit_2_and_save_nothing(store_dir: Path) -> None:
    """-m is fixed when the instance is made: another value fails before any VM starts; the same value runs."""
    made = _sbx("run", "--instance", _ID, "-m", "256", "-q", "-c", "z = 7")
    assert made.exit_code == 0, made.output

    conflict = _sbx("run", "--instance", _ID, "-m", "512", "-c", "print(z)")
    assert conflict.exit_code == EXIT_CLI_ERROR
    assert f"Instance {_ID} has other VM settings: --memory 512 (saved: 256)" in conflict.output
    assert _describe().generation == 1

    same = _sbx("run", "--instance", _ID, "-m", "256", "-c", "print(z)")
    assert same.exit_code == 0, same.output
    assert same.stdout == "7\n"
    assert _describe().generation == 2

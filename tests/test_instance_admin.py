"""open_instance_admin: describe, list and delete saved instances with the store and the keyring only.

Both stores: local (in tmp_path) and S3 (moto, in process). Most heads are
sealed here with the library's head format, as a first save writes them; one
test saves through the real InstanceManager (fake VM, no QEMU). The admin path
must never start a VM manager, fetch an image, check QEMU or probe the store:
`_forbid_vm_path` makes each of those fail the test, and a control test shows
that a Scheduler start does reach them.
"""

from __future__ import annotations

import asyncio
import base64
import random
import secrets
import sys
import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from click.testing import CliRunner

from exec_sandbox import Scheduler, SchedulerConfig, __version__
from exec_sandbox._imports import require_instance_admin
from exec_sandbox.cli import cli
from exec_sandbox.cli_keyring import open_local_keyring
from exec_sandbox.exceptions import ConfigError, ErrorCode, InstanceGoneError, InvalidInputError
from exec_sandbox.instance.keyring import KEY_BYTES
from exec_sandbox.instance.manager import InstanceManager, open_instance_admin
from exec_sandbox.instance.models import HostFacts, InstanceInfo
from exec_sandbox.models import Language
from exec_sandbox.settings import Settings
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.vm_spec import VmSpec
from tests.instance_support import BLOCK, PYTHON, InstanceEnv, Saver, SparseImage, random_block, seed_instance

if TYPE_CHECKING:
    from pathlib import Path


_SPEC = VmSpec(language=Language.JAVASCRIPT, memory_mb=512)
_HOST = HostFacts(
    arch="aarch64",
    accelerator="hvf",
    cpu_vendor=None,
    qemu_version="10.1.0",
    base_image="0" * 64,
    packages_image=None,
    release=__version__,
)


def _key() -> str:
    return base64.b64encode(secrets.token_bytes(KEY_BYTES)).decode()


def _forbid_vm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """From now on, an image fetch, a VM manager, a QEMU check or a store probe fails the test."""

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the admin path reached the VM path")

    monkeypatch.setattr("exec_sandbox.scheduler.ensure_assets", fail)
    monkeypatch.setattr("exec_sandbox.assets.ensure_assets", fail)
    monkeypatch.setattr(VmManager, "__init__", fail)
    monkeypatch.setattr("exec_sandbox.instance.manager.probe_qemu_version", fail)
    monkeypatch.setattr("exec_sandbox.instance.manager.probe", fail)


@pytest.fixture
def no_vm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_vm_path(monkeypatch)


@pytest.fixture(params=["local", "s3"])
def settings(request: pytest.FixtureRequest, tmp_path: Path) -> Settings:
    """Store settings of one test, with a fresh keyring: a local folder, or a fresh prefix of the moto bucket."""
    if request.param == "local":
        return Settings(instance_store="local", instance_store_dir=tmp_path / "instances", instance_keyring=_key())
    endpoint_url, _ = request.getfixturevalue("moto_s3_server")
    return Settings(
        instance_store="s3",
        s3_bucket="test-snapshots",
        s3_endpoint_url=endpoint_url,
        s3_instance_prefix=f"inst-{uuid.uuid4().hex}/",
        instance_keyring=_key(),
    )


def _keyring(settings: Settings) -> str:
    assert settings.instance_keyring is not None
    return settings.instance_keyring.get_secret_value()


# ----------------------------------------------------------------------------- describe, list, delete


@pytest.mark.usefixtures("no_vm_path")
async def test_describe_list_delete(settings: Settings, tmp_path: Path) -> None:
    async with open_instance_admin(settings) as admin:
        assert await admin.store.list_objects("") == []  # the open wrote nothing (no probe)
        await seed_instance(
            admin.store, tmp_path, instance_id="mine", keyring=_keyring(settings), spec=_SPEC, host=_HOST
        )
        await seed_instance(admin.store, tmp_path, instance_id="other", keyring=_key(), spec=_SPEC, host=_HOST)
        assert sorted([iid async for iid in admin.list_ids()]) == ["mine", "other"]

        mine = await admin.describe("mine")
        assert (mine.instance_id, mine.generation, mine.verified) == ("mine", 1, True)
        assert (mine.spec, mine.host) == (_SPEC, _HOST)
        other = await admin.describe("other")  # sealed with a key this keyring lacks: public header only
        assert (other.generation, other.verified, other.spec, other.host) == (1, False, None, None)

        assert await admin.delete("mine") is True
        assert await admin.store.list_objects("s/mine/") == []  # head and body object
        assert await admin.delete("mine") is False
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND) as exc:
            await admin.describe("mine")
        assert exc.value.instance_id == "mine"
        assert [iid async for iid in admin.list_ids()] == ["other"]


@pytest.mark.usefixtures("no_vm_path")
async def test_list_skips_names_that_are_not_instance_ids(moto_s3_server: Any, tmp_path: Path) -> None:
    """Another bucket writer's object under s/, in a folder whose name is no instance id (S3 takes
    any key; the local store refuses such keys): no id is listed for it."""
    endpoint_url, s3 = moto_s3_server
    prefix = f"inst-{uuid.uuid4().hex}/"
    settings = Settings(
        instance_store="s3",
        s3_bucket="test-snapshots",
        s3_endpoint_url=endpoint_url,
        s3_instance_prefix=prefix,
        instance_keyring=_key(),
    )
    async with open_instance_admin(settings) as admin:
        await seed_instance(
            admin.store, tmp_path, instance_id="mine", keyring=_keyring(settings), spec=_SPEC, host=_HOST
        )
        s3.put_object(Bucket="test-snapshots", Key=f"{prefix}s/NOT-AN-ID/x", Body=b"x")
        assert [iid async for iid in admin.list_ids()] == ["mine"]


@pytest.mark.usefixtures("no_vm_path")
async def test_ids_that_are_not_instance_ids_are_refused(settings: Settings, tmp_path: Path) -> None:
    """ "a/b" names a prefix inside instance "a": refused before any store call, and "a" stays whole."""
    async with open_instance_admin(settings) as admin:
        await seed_instance(admin.store, tmp_path, instance_id="a", keyring=_keyring(settings), spec=_SPEC, host=_HOST)
        before = await admin.store.list_objects("")
        for bad in ("a/b", "a/", "", "A", "../a"):
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.INSTANCE_INVALID_ID):
                await admin.delete(bad)
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.INSTANCE_INVALID_ID):
                await admin.describe(bad)
        assert await admin.store.list_objects("") == before
        assert (await admin.describe("a")).verified


async def test_admin_reads_what_the_manager_saved(instance_env: InstanceEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real save (InstanceManager, fake VM): the admin's describe equals the manager's; its delete removes it."""
    ram, home = SparseImage(size=4 * BLOCK), SparseImage(size=4 * BLOCK)
    ram.put(0, random_block(random.Random(1)))
    home.put(2, random_block(random.Random(2)))
    saver = Saver(
        instance_env, await instance_env.new_vm(ram, home, spec=PYTHON), manager=instance_env.manager, state=None
    )
    instance_id = (await saver.save()).instance_id
    proxy = instance_env.proxy
    settings = Settings(
        instance_store="s3",
        s3_bucket=proxy.bucket.name,
        s3_endpoint_url=proxy.endpoint_url,
        s3_instance_prefix=proxy.prefix,
        instance_keyring=instance_env.keyring,
    )
    _forbid_vm_path(monkeypatch)

    async with open_instance_admin(settings) as admin:
        assert [iid async for iid in admin.list_ids()] == [instance_id]
        info = await admin.describe(instance_id)
        assert info == await instance_env.manager.describe(instance_id)
        assert (info.verified, info.spec) == (True, VmSpec(language=Language.PYTHON))
        assert await admin.delete(instance_id) is True
        assert await admin.store.list_objects(f"s/{instance_id}/") == []
    with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND):
        await instance_env.manager.describe(instance_id)


# ----------------------------------------------------------------------------- open: settings and the VM path


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"instance_store": None, "instance_keyring": "KEY"}, "none is configured"),
        ({"instance_store": "local", "instance_keyring": None}, "instance_keyring is not"),
        ({"instance_store": "local", "instance_keyring": "not-base64!"}, "not base64"),
        ({"instance_store": "s3", "instance_keyring": "KEY", "s3_bucket": None}, "s3_bucket is not set"),
    ],
    ids=["no-store", "no-keyring", "invalid-keyring", "s3-without-bucket"],
)
@pytest.mark.usefixtures("no_vm_path")
async def test_bad_settings_refused_before_the_store_opens(tmp_path: Path, fields: dict[str, Any], match: str) -> None:
    """KEY is a fresh valid keyring. The local store's folder is never made."""
    root = tmp_path / "instances"
    keyring = _key() if fields["instance_keyring"] == "KEY" else fields["instance_keyring"]
    settings = Settings(**{**fields, "instance_keyring": keyring, "instance_store_dir": root})
    with pytest.raises(ConfigError, match=match, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG) as exc:
        async with open_instance_admin(settings):
            pass
    assert exc.value.action == "FIX_CONFIG"
    assert not root.exists()


async def test_the_vm_path_is_watched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Control: with the same patches, a Scheduler start, InstanceManager.open and a VmManager fail.
    So the tests above prove that the admin path skips them."""
    settings = Settings(instance_store="local", instance_store_dir=tmp_path / "instances", instance_keyring=_key())
    _forbid_vm_path(monkeypatch)
    config = SchedulerConfig(
        instance_store="local", instance_store_dir=settings.instance_store_dir, instance_keyring=_keyring(settings)
    )
    with pytest.raises(AssertionError, match="VM path"):
        async with Scheduler(config):
            pass
    with pytest.raises(AssertionError, match="VM path"):
        async with InstanceManager.open(settings, work_dir=tmp_path / "work", vm_manager=MagicMock()):
            pass
    with pytest.raises(AssertionError, match="VM path"):
        VmManager(settings)


@pytest.mark.parametrize(
    ("store", "install"),
    [("s3", "'exec-sandbox[s3]'"), ("local", "'exec-sandbox[cli]' (or 'exec-sandbox[s3]')")],
)
def test_import_guard_names_the_extra(store: str, install: str) -> None:
    with (
        patch.dict(sys.modules, {"exec_sandbox.instance.manager": None}),
        pytest.raises(ConfigError, match="cryptography", check=lambda e: e.code is ErrorCode.EXTRA_MISSING) as exc,
    ):
        require_instance_admin(store=store)  # type: ignore[arg-type]
    assert f"pip install {install}" in str(exc.value)


# ----------------------------------------------------------------------------- sbx instance, no mocks


@pytest.mark.usefixtures("no_vm_path")
def test_cli_commands_open_no_vm_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`sbx instance list | describe | rm` on a local store in tmp_path. The key comes from
    EXEC_SANDBOX_INSTANCE_KEYRING: the OS keyring fails the test if called. No image folder is made."""
    root, assets, key = tmp_path / "instances", tmp_path / "assets", _key()
    monkeypatch.delenv("EXEC_SANDBOX_INSTANCE_STORE", raising=False)
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE_DIR", str(root))
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", key)
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_ASSET_DIR", str(assets))
    os_keyring: MagicMock = create_autospec(open_local_keyring, side_effect=AssertionError("OS keyring used"))
    monkeypatch.setattr("exec_sandbox.cli_keyring.open_local_keyring", os_keyring)

    async def seed() -> None:
        settings = Settings(instance_store="local", instance_store_dir=root, instance_keyring=key)
        async with open_instance_admin(settings) as admin:
            await seed_instance(admin.store, tmp_path, instance_id="my-vm", keyring=key, spec=_SPEC, host=_HOST)

    asyncio.run(seed())
    runner = CliRunner()

    listed = runner.invoke(cli, ["instance", "list"])
    assert (listed.exit_code, listed.stdout) == (0, "my-vm\n"), listed.output
    described = runner.invoke(cli, ["instance", "describe", "my-vm"])
    assert described.exit_code == 0, described.output
    info = InstanceInfo.model_validate_json(described.stdout)
    assert (info.instance_id, info.verified, info.spec) == ("my-vm", True, _SPEC)
    removed = runner.invoke(cli, ["instance", "rm", "my-vm"])
    assert (removed.exit_code, removed.stdout) == (0, "my-vm\n"), removed.output
    assert runner.invoke(cli, ["instance", "list"]).stdout == ""

    os_keyring.assert_not_called()
    assert not assets.exists()

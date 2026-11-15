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
import hashlib
import random
import sys
import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from exec_sandbox import Scheduler, SchedulerConfig, __version__, instance_admin
from exec_sandbox._imports import require_instance_admin
from exec_sandbox.cli import cli
from exec_sandbox.exceptions import ConfigError, ErrorCode, InstanceGoneError, InvalidInputError, RouteError
from exec_sandbox.instance.head import canonical_json
from exec_sandbox.instance.keyring import Keyring
from exec_sandbox.instance.manager import InstanceManager, open_instance_admin
from exec_sandbox.instance.models import (
    BLOCK_BYTES,
    ChainEntry,
    GuestFacts,
    HeadBody,
    HostFacts,
    InstanceInfo,
    RecentCommit,
)
from exec_sandbox.instance.store import Applied
from exec_sandbox.models import Language
from exec_sandbox.settings import Settings
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.vm_spec import VmSpec
from tests.instance_support import (
    BLOCK,
    FAKE_LAYOUT,
    GOLDEN_KEY,
    GOLDEN_KEY_ID,
    NO_VM_TIMEOUT,
    PYTHON,
    InstanceEnv,
    Saver,
    SparseImage,
    new_keyring,
    random_block,
    seal_foreign_head,
    seed_instance,
)
from tests.keyring_support import isolate_account

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = NO_VM_TIMEOUT

_SPEC = VmSpec(language=Language.JAVASCRIPT, memory_mb=512)
_HOST = HostFacts(
    arch="aarch64",
    accelerator="hvf",
    cpu_vendor=None,
    cpu_model="host",
    qemu_version="10.1.0",
    machine_type="virt-10.1",
    release=__version__,
    device_layout=FAKE_LAYOUT,
)
_GUEST = GuestFacts(base_image="0" * 64, packages_image=None, kernel="1" * 64, initramfs="2" * 64)


def _forbid_vm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """From now on, an image fetch, a VM manager, a QEMU check or a store probe fails the test."""

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the admin path reached the VM path")

    monkeypatch.setattr("exec_sandbox.scheduler.ensure_assets", fail)
    monkeypatch.setattr("exec_sandbox.assets.ensure_assets", fail)
    monkeypatch.setattr(VmManager, "__init__", fail)
    monkeypatch.setattr("exec_sandbox.instance.manager.require_qemu_version", fail)
    monkeypatch.setattr("exec_sandbox.worker_facts.require_qemu_version", fail)
    monkeypatch.setattr("exec_sandbox.instance.manager.probe", fail)


@pytest.fixture
def no_vm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_vm_path(monkeypatch)


@pytest.fixture(params=["local", "s3"])
def settings(request: pytest.FixtureRequest, tmp_path: Path) -> Settings:
    """Store settings of one test, with a fresh keyring: a local folder, or a fresh prefix of the moto bucket."""
    if request.param == "local":
        return Settings(
            instance_store="local", instance_store_dir=tmp_path / "instances", instance_keyring=new_keyring()
        )
    endpoint_url, _ = request.getfixturevalue("moto_s3_server")
    return Settings(
        instance_store="s3",
        s3_bucket="test-snapshots",
        s3_endpoint_url=endpoint_url,
        s3_instance_prefix=f"inst-{uuid.uuid4().hex}/",
        instance_keyring=new_keyring(),
    )


def _keyring(settings: Settings) -> str:
    assert settings.instance_keyring is not None
    return settings.instance_keyring.get_secret_value()


# ----------------------------------------------------------------------------- describe, list, delete


@pytest.mark.parametrize(
    ("access", "writer_release", "spec", "host", "valid"),
    [
        pytest.param("full", __version__, _SPEC, _HOST, True, id="full"),
        pytest.param("full", __version__, _SPEC, None, False, id="full-without-host"),
        pytest.param("full", None, _SPEC, _HOST, False, id="full-without-release"),
        pytest.param("other_release", "9.9.9", None, None, True, id="other-release"),
        pytest.param("other_release", None, None, None, True, id="other-release-without-release"),
        pytest.param("other_release", "9.9.9", _SPEC, None, False, id="other-release-with-spec"),
        pytest.param("no_key", None, None, None, True, id="no-key"),
        pytest.param("no_key", "9.9.9", None, None, False, id="no-key-with-release"),
        pytest.param("no_key", None, None, _HOST, False, id="no-key-with-host"),
    ],
)
def test_instance_info_fields_follow_access(
    access: str, writer_release: str | None, spec: VmSpec | None, host: HostFacts | None, valid: bool
) -> None:
    """spec and host are set exactly when access is "full"; full has a writer_release (its
    host's), no_key none."""
    fields = {
        "instance_id": "vm",
        "generation": 1,
        "recent_commits": (RecentCommit(generation=1, commit_id="a" * 32),),
        "key_id": "b" * 64,
        "access": access,
        "writer_release": writer_release,
        "spec": spec,
        "host": host,
        "guest": _GUEST if host is not None else None,
        "egress_domains": () if access == "full" else None,
    }
    if valid:
        assert InstanceInfo.model_validate(fields).access == access
    else:
        with pytest.raises(ValidationError):
            InstanceInfo.model_validate(fields)


@pytest.mark.usefixtures("no_vm_path")
async def test_the_public_opener_needs_no_scheduler(tmp_path: Path) -> None:
    """exec_sandbox.instance_admin: describe, list and delete from a SchedulerConfig, with no
    VM manager, no guest image and no QEMU check (a cleanup job on any host)."""
    keyring = new_keyring()
    config = SchedulerConfig(instance_store="local", instance_store_dir=tmp_path / "store", instance_keyring=keyring)
    async with instance_admin(config) as admin:
        await seed_instance(
            admin.store, tmp_path, instance_id="mine", keyring=keyring, spec=_SPEC, host=_HOST, guest=_GUEST
        )
        assert [instance_id async for instance_id in admin.list_ids()] == ["mine"]
        assert (await admin.describe("mine")).access == "full"
        assert await admin.delete("mine") is True
    with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG, match="none is configured"):
        async with instance_admin(SchedulerConfig(instance_store=None)):
            pass


@pytest.mark.usefixtures("no_vm_path")
async def test_the_public_opener_reads_the_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A store and keyring set by EXEC_SANDBOX_* variables only open, as for a Scheduler; an
    invalid variable is SETTING_INVALID (it names the variable)."""
    keyring = new_keyring()
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE", "local")
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", keyring)
    async with instance_admin(SchedulerConfig()) as admin:
        assert [key.id for key in admin.keyring.keys] == [Keyring.parse(keyring).latest.id]
    monkeypatch.setenv("EXEC_SANDBOX_MAX_CONCURRENT_S3_UPLOADS", "0")
    with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID) as exc:
        async with instance_admin(SchedulerConfig()):
            pass
    assert "EXEC_SANDBOX_MAX_CONCURRENT_S3_UPLOADS" in str(exc.value)


@pytest.mark.usefixtures("no_vm_path")
async def test_list_and_delete_assets(settings: Settings, tmp_path: Path) -> None:
    """The guest images of the store, by digest, for a cleanup job; delete_asset says whether
    it was there, and refuses a value that is no digest before any store call."""
    image = tmp_path / "image"
    image.write_bytes(b"guest image")
    digest = hashlib.sha256(b"guest image").hexdigest()
    async with open_instance_admin(settings) as admin:
        assert await admin.list_assets() == []
        await admin.store.upload(f"assets/sha256/{digest}", src=image)
        await admin.store.upload("assets/sha256/not-a-digest", src=image)  # another writer's object
        assert await admin.list_assets() == [digest]
        assert await admin.delete_asset(digest) is True
        assert await admin.delete_asset(digest) is False
        assert await admin.list_assets() == []
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID):
            await admin.delete_asset("../s/mine/head")


@pytest.mark.usefixtures("no_vm_path")
async def test_describe_list_delete(settings: Settings, tmp_path: Path) -> None:
    async with open_instance_admin(settings) as admin:
        assert await admin.store.list_objects("") == []  # the open wrote nothing (no probe)
        await seed_instance(
            admin.store, tmp_path, instance_id="mine", keyring=_keyring(settings), spec=_SPEC, host=_HOST, guest=_GUEST
        )
        await seed_instance(
            admin.store, tmp_path, instance_id="other", keyring=new_keyring(), spec=_SPEC, host=_HOST, guest=_GUEST
        )
        assert sorted([iid async for iid in admin.list_ids()]) == ["mine", "other"]

        mine = await admin.describe("mine")
        assert (mine.instance_id, mine.generation, mine.access) == ("mine", 1, "full")
        assert (mine.spec, mine.host, mine.writer_release) == (_SPEC, _HOST, _HOST.release)
        other = await admin.describe("other")  # sealed with a key this keyring lacks: public header only
        assert (other.generation, other.access, other.writer_release, other.spec, other.host) == (
            1,
            "no_key",
            None,
            None,
            None,
        )

        assert await admin.delete("mine") is True
        assert await admin.store.list_objects("s/mine/") == []  # head and body object
        assert await admin.delete("mine") is False
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND) as exc:
            await admin.describe("mine")
        assert exc.value.instance_id == "mine"
        assert [iid async for iid in admin.list_ids()] == ["other"]


async def test_describe_refuses_a_changed_head(settings: Settings, tmp_path: Path) -> None:
    """A head whose authenticated bytes changed (generation 1 written as 9): describe raises
    INSTANCE_INTEGRITY and never gives the changed fields, not "no_key" or another release's
    summary (without a VM: the same rule as test_instance_e2e on both stores)."""
    async with open_instance_admin(settings) as admin:
        await seed_instance(
            admin.store, tmp_path, instance_id="mine", keyring=_keyring(settings), spec=_SPEC, host=_HOST, guest=_GUEST
        )
        got = await admin.store.get("s/mine/head", max_bytes=1 << 20)
        assert got is not None and b'"generation":1' in got[0]
        await admin.store.delete_many(["s/mine/head"])
        changed = got[0].replace(b'"generation":1', b'"generation":9')
        assert isinstance(await admin.store.put_if_absent("s/mine/head", data=changed), Applied)
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY) as exc:
            await admin.describe("mine")
        assert exc.value.instance_id == "mine"


@pytest.mark.usefixtures("no_vm_path")
async def test_describe_a_head_another_release_wrote(settings: Settings) -> None:
    """Another release's head (a header or body field this release does not know): access
    "other_release" with its summary and writer, never INSTANCE_INTEGRITY (a client would
    delete good data) and never ROUTE: GC and VERIFY run on any release. A header field this
    release does not know gives other_release even over a body it reads. Only when even the
    summary (generation, recent commits, key id) does not parse: ROUTE to another release
    (empty `required`), and, without the key, INSTANCE_KEY_MISSING (do not delete)."""
    foreign_body = b'{"host":{"release":"9.9.9"},"spec":{},"added_later":1}'
    entry = ChainEntry(commit_id="c" * 32, kind="full", size=1, stream_sizes=(1, 1), etag="e", block_bytes=BLOCK_BYTES)
    current_body = canonical_json(
        HeadBody(spec=_SPEC, host=_HOST, guest=_GUEST, egress_domains=(), chain=(entry,)).model_dump(mode="json")
    )
    ours = Keyring.parse(_keyring(settings)).latest
    stranger = Keyring.parse(new_keyring()).latest
    async with open_instance_admin(settings) as admin:
        for instance_id, kek, fields, body_json in (
            ("new-body", ours, {}, foreign_body),
            ("new-header", ours, {"added_later": 1}, foreign_body),
            ("new-header-current-body", ours, {"added_later": 1}, current_body),
            ("new-header-no-key", stranger, {"added_later": 1}, foreign_body),
            ("changed-generation", ours, {"generation": "1"}, foreign_body),
            ("changed-generation-no-key", stranger, {"generation": "1"}, foreign_body),
        ):
            data = seal_foreign_head(kek=kek, instance_id=instance_id, header_fields=fields, body_json=body_json)
            assert isinstance(await admin.store.put_if_absent(f"s/{instance_id}/head", data=data), Applied)

        for instance_id, writer in (
            ("new-body", "9.9.9"),
            ("new-header", "9.9.9"),
            ("new-header-current-body", _HOST.release),
        ):
            info = await admin.describe(instance_id)
            assert (info.instance_id, info.generation, info.key_id, info.access, info.writer_release) == (
                instance_id,
                1,
                ours.id,
                "other_release",
                writer,
            )
            assert (info.spec, info.host) == (None, None)
        info = await admin.describe("new-header-no-key")
        assert (info.generation, info.key_id, info.access, info.writer_release) == (1, stranger.id, "no_key", None)
        with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
            await admin.describe("changed-generation")
        assert (exc.value.required, exc.value.instance_id) == ({}, "changed-generation")
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_KEY_MISSING):
            await admin.describe("changed-generation-no-key")


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
        instance_keyring=new_keyring(),
    )
    async with open_instance_admin(settings) as admin:
        await seed_instance(
            admin.store, tmp_path, instance_id="mine", keyring=_keyring(settings), spec=_SPEC, host=_HOST, guest=_GUEST
        )
        s3.put_object(Bucket="test-snapshots", Key=f"{prefix}s/NOT-AN-ID/x", Body=b"x")
        assert [iid async for iid in admin.list_ids()] == ["mine"]


@pytest.mark.usefixtures("no_vm_path")
async def test_ids_that_are_not_instance_ids_are_refused(settings: Settings, tmp_path: Path) -> None:
    """ "a/b" names a prefix inside instance "a": refused before any store call, and "a" stays whole."""
    async with open_instance_admin(settings) as admin:
        await seed_instance(
            admin.store, tmp_path, instance_id="a", keyring=_keyring(settings), spec=_SPEC, host=_HOST, guest=_GUEST
        )
        before = await admin.store.list_objects("")
        for bad in ("a/b", "a/", "", "A", "../a"):
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.INSTANCE_INVALID_ID):
                await admin.delete(bad)
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.INSTANCE_INVALID_ID):
                await admin.describe(bad)
        assert await admin.store.list_objects("") == before
        assert (await admin.describe("a")).access == "full"


async def test_admin_reads_what_the_manager_saved(instance_env: InstanceEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real save (InstanceManager, fake VM) with a keyring of two keys: the admin's describe
    gives the facts of that save (its commit, the id of the newest key, the SHA-256 of the base
    image); its delete removes it."""
    keyring = f"{GOLDEN_KEY},{instance_env.keyring}"
    ram, home = SparseImage(size=4 * BLOCK), SparseImage(size=4 * BLOCK)
    ram.put(0, random_block(random.Random(1)))
    home.put(2, random_block(random.Random(2)))
    vm = await instance_env.new_vm(ram, home, spec=PYTHON)
    saver = Saver(instance_env, vm, manager=instance_env.manager_for(keyring), state=None)
    saved = await saver.save()
    instance_id = saved.instance_id
    base_digest = hashlib.sha256(instance_env.base_image.read_bytes()).hexdigest()
    proxy = instance_env.proxy
    settings = Settings(
        instance_store="s3",
        s3_bucket=proxy.bucket.name,
        s3_endpoint_url=proxy.endpoint_url,
        s3_instance_prefix=proxy.prefix,
        instance_keyring=keyring,
    )
    _forbid_vm_path(monkeypatch)

    async with open_instance_admin(settings) as admin:
        assert [iid async for iid in admin.list_ids()] == [instance_id]
        info = await admin.describe(instance_id)
        assert (info.instance_id, info.generation, info.access, info.spec) == (
            instance_id,
            1,
            "full",
            VmSpec(language=Language.PYTHON),
        )
        assert info.recent_commits == (RecentCommit(generation=1, commit_id=saved.header.commit_id),)
        assert info.key_id == GOLDEN_KEY_ID
        assert info.host is not None
        assert (info.guest.base_image, info.guest.packages_image, info.host.release) == (base_digest, None, __version__)
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
    keyring = new_keyring() if fields["instance_keyring"] == "KEY" else fields["instance_keyring"]
    settings = Settings(**{**fields, "instance_keyring": keyring, "instance_store_dir": root})
    with pytest.raises(ConfigError, match=match, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG) as exc:
        async with open_instance_admin(settings):
            pass
    assert exc.value.action == "FIX_CONFIG"
    assert not root.exists()


async def test_the_vm_path_is_watched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Control: with the same patches, a Scheduler start, InstanceManager.open and a VmManager fail.
    So the tests above prove that the admin path skips them."""
    settings = Settings(
        instance_store="local", instance_store_dir=tmp_path / "instances", instance_keyring=new_keyring()
    )
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
    EXEC_SANDBOX_INSTANCE_KEYRING: the OS keyring (a dict behind the OS backend class) is never
    read, and nothing is made under the account's home. No image folder is made."""
    root, assets, key = tmp_path / "instances", tmp_path / "assets", new_keyring()
    monkeypatch.delenv("EXEC_SANDBOX_INSTANCE_STORE", raising=False)
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE_DIR", str(root))
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", key)
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_ASSET_DIR", str(assets))
    account = isolate_account(monkeypatch, home=tmp_path / "account", key=new_keyring())

    async def seed() -> None:
        settings = Settings(instance_store="local", instance_store_dir=root, instance_keyring=key)
        async with open_instance_admin(settings) as admin:
            await seed_instance(
                admin.store, tmp_path, instance_id="my-vm", keyring=key, spec=_SPEC, host=_HOST, guest=_GUEST
            )

    asyncio.run(seed())
    runner = CliRunner()

    listed = runner.invoke(cli, ["instance", "list"])
    assert (listed.exit_code, listed.stdout) == (0, "my-vm\n"), listed.output
    described = runner.invoke(cli, ["instance", "describe", "my-vm"])
    assert described.exit_code == 0, described.output
    info = InstanceInfo.model_validate_json(described.stdout)
    assert (info.instance_id, info.access, info.spec) == ("my-vm", "full", _SPEC)
    removed = runner.invoke(cli, ["instance", "rm", "my-vm"])
    assert (removed.exit_code, removed.stdout) == (0, "my-vm\n"), removed.output
    assert runner.invoke(cli, ["instance", "list"]).stdout == ""

    account.assert_untouched()
    assert not assets.exists()

"""Saved instances are on when instance_store is set: store selection, keyring, start checks, import
guards, and what each worker must have (QEMU floor, host facts at load).

The "s3" store reuses the S3 config (bucket, region, endpoint) under its own prefix.
"""

import asyncio
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from exec_sandbox import Language, Scheduler, SchedulerConfig, VmSpec, __version__, constants
from exec_sandbox._imports import (
    require_aioboto3,
    require_cli_keyring,
    require_instance_admin,
    require_instance_manager,
    require_s3_instance_store,
)
from exec_sandbox.exceptions import ConfigError, ErrorCode, InstanceGoneError, RouteError
from exec_sandbox.instance.local_store import LocalInstanceStore
from exec_sandbox.instance.manager import InstanceManager
from exec_sandbox.instance.models import HostFacts
from exec_sandbox.platform_utils import HostArch, HostOS, child_env
from exec_sandbox.scheduler import settings_from_config
from exec_sandbox.settings import Settings
from exec_sandbox.vm_types import AccelType
from tests.instance_support import (
    BLOCK,
    NO_VM_TIMEOUT,
    PYTHON,
    InstanceEnv,
    Saver,
    SparseImage,
    new_keyring,
    no_packages,
    seed_instance,
)
from tests.s3_faults import Edit, Request, S3Proxy

pytestmark = NO_VM_TIMEOUT


class TestS3Settings:
    @pytest.mark.parametrize(
        ("snapshot", "instance"),
        [("snapshots/", "snapshots/inst/"), ("a/", "a/"), ("", "instances/")],
        ids=["nested", "same", "empty-contains-all"],
    )
    def test_overlapping_prefixes_refused(self, snapshot: str, instance: str) -> None:
        with pytest.raises(ValidationError, match="overlap"):
            SchedulerConfig(s3_snapshot_prefix=snapshot, s3_instance_prefix=instance)

    def test_defaults_are_distinct(self) -> None:
        config = SchedulerConfig()
        assert (config.s3_snapshot_prefix, config.s3_instance_prefix) == ("snapshots/", "instances/")

    def test_environment_and_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        key = new_keyring()
        monkeypatch.setenv("EXEC_SANDBOX_S3_BUCKET", "from-env")
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", key)
        settings = Settings()
        assert settings.s3_bucket == "from-env"
        assert settings.instance_keyring is not None
        assert settings.instance_keyring.get_secret_value() == key
        assert key not in repr(settings)
        assert key not in repr(SchedulerConfig(instance_keyring=key))
        assert Settings(s3_bucket="argument").s3_bucket == "argument"  # argument wins

    def test_keyring_never_reaches_child_processes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", new_keyring())
        assert "EXEC_SANDBOX_INSTANCE_KEYRING" not in child_env()


class TestStoreSettings:
    def test_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The local store's root defaults to the user's data folder (not the cache: it is not made again)."""
        monkeypatch.delenv("EXEC_SANDBOX_INSTANCE_STORE", raising=False)
        monkeypatch.delenv("EXEC_SANDBOX_INSTANCE_STORE_DIR", raising=False)
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setenv("XDG_DATA_HOME", "/xdg/data")
        settings = Settings()
        assert settings.instance_store is None
        assert settings.instance_store_dir == Path("/xdg/data/exec-sandbox/instances")
        assert SchedulerConfig().instance_store_dir == Path("/xdg/data/exec-sandbox/instances")

    def test_environment(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE", "local")
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE_DIR", str(tmp_path))
        settings = Settings()
        assert (settings.instance_store, settings.instance_store_dir) == ("local", tmp_path)
        assert Settings(instance_store="s3").instance_store == "s3"  # argument wins

    @pytest.mark.parametrize("value", ["S3", "nfs", ""])
    def test_unknown_store_refused(self, value: str) -> None:
        with pytest.raises(ValidationError, match="instance_store"):
            SchedulerConfig(instance_store=value)  # type: ignore[arg-type]

    def test_bridge_copies_set_fields(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A value set in code wins over the environment; an unset field keeps the environment's."""
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE", "local")
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE_DIR", str(tmp_path / "env"))
        from_env = settings_from_config(SchedulerConfig(), images_dir=tmp_path)
        assert (from_env.instance_store, from_env.instance_store_dir) == ("local", tmp_path / "env")
        config = SchedulerConfig(instance_store="s3", instance_store_dir=tmp_path / "code")
        from_code = settings_from_config(config, images_dir=tmp_path)
        assert (from_code.instance_store, from_code.instance_store_dir) == ("s3", tmp_path / "code")
        assert settings_from_config(SchedulerConfig(instance_store=None), images_dir=tmp_path).instance_store is None


class TestImportGuards:
    def test_core_import_needs_no_extra(self) -> None:
        """`import exec_sandbox` and the store contract load without cryptography and aiobotocore;
        the instance manager and the S3 store load only when instances are on."""
        code = (
            "import sys\n"
            "for name in ('cryptography', 'aiobotocore', 'botocore', 'aioboto3', 'click', 'keyring'):\n"
            "    sys.modules[name] = None\n"
            "import exec_sandbox, exec_sandbox.instance.store\n"
            "loaded = {'exec_sandbox.instance.manager', 'exec_sandbox.instance.s3_store'} & set(sys.modules)\n"
            "assert not loaded, loaded\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=60)

    @pytest.mark.parametrize("require", [require_instance_manager, require_instance_admin], ids=["manager", "admin"])
    @pytest.mark.parametrize(
        ("store", "install"),
        [("s3", "'exec-sandbox[s3]'"), ("local", "'exec-sandbox[cli]' (or 'exec-sandbox[s3]')")],
    )
    def test_manager_names_the_extra(self, require: Any, store: str, install: str) -> None:
        """The manager (Scheduler) and the admin (describe, list, delete) need the same extra."""
        with (
            patch.dict(sys.modules, {"exec_sandbox.instance.manager": None}),
            pytest.raises(ConfigError, match="cryptography", check=lambda e: e.code is ErrorCode.EXTRA_MISSING) as exc,
        ):
            require(store=store)
        assert f"pip install {install}" in str(exc.value)

    def test_s3_store_names_the_extra(self) -> None:
        with (
            patch.dict(sys.modules, {"exec_sandbox.instance.s3_store": None}),
            pytest.raises(
                ConfigError,
                match=r"pip install 'exec-sandbox\[s3\]'",
                check=lambda e: e.code is ErrorCode.EXTRA_MISSING,
            ),
        ):
            require_s3_instance_store()

    def test_l3_cache_names_the_extra(self) -> None:
        """The L3 snapshot cache (the S3 copy of the L2 disk cache) needs aioboto3 (the s3 extra)."""
        with (
            patch.dict(sys.modules, {"aioboto3": None}),
            pytest.raises(
                ConfigError,
                match=r"^The L3 snapshot cache \(s3_bucket\) needs the s3 extra\. Install with: pip install 'exec-sandbox\[s3\]'$",
                check=lambda e: e.code is ErrorCode.EXTRA_MISSING,
            ),
        ):
            require_aioboto3()

    def test_cli_keyring_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The OS keyring step of the CLI needs keyring (the cli extra; test_cli_entry runs the
        CLI without keyring, then without cryptography)."""
        monkeypatch.delitem(sys.modules, "exec_sandbox.cli_keyring", raising=False)
        with (
            patch.dict(sys.modules, {"keyring": None}),
            pytest.raises(
                ConfigError,
                match=r"^The exec-sandbox CLI needs keyring: pip install 'exec-sandbox\[cli\]'$",
                check=lambda e: e.code is ErrorCode.EXTRA_MISSING,
            ),
        ):
            require_cli_keyring()


class TestSchedulerStart:
    @staticmethod
    def _config(
        base: SchedulerConfig,
        *,
        endpoint_url: str,
        prefix: str,
        store: str | None,
        keyring: str | None,
        bucket: str | None,
    ) -> SchedulerConfig:
        """`base` plus the store, the bucket at `endpoint_url` and an instance prefix (validated, so
        fields count as set)."""
        return SchedulerConfig.model_validate(
            {
                **base.model_dump(exclude_unset=True),
                "instance_store": store,
                "s3_bucket": bucket,
                "s3_endpoint_url": endpoint_url,
                "s3_instance_prefix": prefix,
                "instance_keyring": keyring,
            }
        )

    @classmethod
    def _moto_config(
        cls,
        base: SchedulerConfig,
        moto_s3_server: tuple[str, object],
        *,
        store: str | None,
        keyring: str | None,
        bucket: str | None,
    ) -> SchedulerConfig:
        """`_config` on the moto bucket, under a fresh instance prefix."""
        endpoint_url, _ = moto_s3_server
        return cls._config(
            base,
            endpoint_url=endpoint_url,
            prefix=f"inst-{uuid.uuid4().hex}/",
            store=store,
            keyring=keyring,
            bucket=bucket,
        )

    async def test_store_opened(self, scheduler_config: SchedulerConfig, moto_s3_server) -> None:
        config = self._moto_config(
            scheduler_config, moto_s3_server, store="s3", keyring=new_keyring(), bucket="test-snapshots"
        )
        async with Scheduler(config) as s:
            assert [iid async for iid in s.list_instances()] == []  # on, and the fresh prefix is empty

    async def test_without_store_instances_are_off(self, scheduler_config: SchedulerConfig, moto_s3_server) -> None:
        """A keyring and a bucket alone do not turn saved instances on."""
        config = self._moto_config(
            scheduler_config, moto_s3_server, store=None, keyring=new_keyring(), bucket="test-snapshots"
        )
        async with Scheduler(config) as s:
            with pytest.raises(
                ConfigError, match="instance store", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
            ) as exc:
                [iid async for iid in s.list_instances()]
            assert "instance_store" in (exc.value.hint or "")

    async def test_store_without_conditional_writes_fails_the_start(
        self, scheduler_config: SchedulerConfig, s3_proxy: S3Proxy
    ) -> None:
        """The start probes the store: one that ignores If-None-Match (the fault proxy strips it,
        so the store overwrites) would lose updates."""

        def unconditional(req: Request) -> None:
            req.headers.popall("If-None-Match", None)

        s3_proxy.add("PUT_IF", "^probe/", Edit(request=unconditional), times=None)
        config = self._config(
            scheduler_config,
            endpoint_url=s3_proxy.endpoint_url,
            prefix=s3_proxy.prefix,
            store="s3",
            keyring=new_keyring(),
            bucket=s3_proxy.bucket.name,
        )
        with pytest.raises(
            ConfigError, match="conditional", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ) as exc:
            async with Scheduler(config):
                pass
        assert exc.value.action == "FIX_CONFIG"
        assert s3_proxy.count("PUT_IF", "^probe/") == 2  # the create, then the create of the existing key

    async def test_local_store_opens_in_its_folder(self, scheduler_config: SchedulerConfig, tmp_path: Path) -> None:
        """instance_store="local": no bucket needed; the folder is made private (0700) at the start."""
        root = tmp_path / "instances"
        config = SchedulerConfig.model_validate(
            {
                **scheduler_config.model_dump(exclude_unset=True),
                "instance_store": "local",
                "instance_store_dir": str(root),
                "instance_keyring": new_keyring(),
            }
        )
        async with Scheduler(config) as s:
            assert [iid async for iid in s.list_instances()] == []
        assert stat.S_IMODE(root.stat().st_mode) & 0o777 == 0o700

    async def test_unsafe_local_folder_fails_the_start(self, scheduler_config: SchedulerConfig, tmp_path: Path) -> None:
        root = tmp_path / "instances"
        root.mkdir(mode=0o777)
        root.chmod(0o777)  # the umask can remove bits at mkdir
        config = SchedulerConfig.model_validate(
            {
                **scheduler_config.model_dump(exclude_unset=True),
                "instance_store": "local",
                "instance_store_dir": str(root),
                "instance_keyring": new_keyring(),
            }
        )
        with pytest.raises(
            ConfigError, match="group or others have access", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ):
            async with Scheduler(config):
                pass

    @pytest.mark.parametrize(
        "case",
        [
            ("s3", "not-base64!", "test-snapshots", "not base64"),
            ("s3", "KEY", None, "s3_bucket is not set"),
            ("s3", None, "test-snapshots", "instance_keyring is not"),
            ("local", None, None, "instance_keyring is not"),
        ],
        ids=["invalid-keyring", "s3-without-bucket", "store-without-keyring", "local-without-keyring"],
    )
    async def test_bad_settings_fail_the_start(
        self, scheduler_config: SchedulerConfig, moto_s3_server, case: tuple[str, str | None, str | None, str]
    ) -> None:
        """(store, keyring, bucket, error text); KEY is a fresh valid keyring."""
        store, keyring, bucket, match = case
        config = self._moto_config(
            scheduler_config,
            moto_s3_server,
            store=store,
            keyring=new_keyring() if keyring == "KEY" else keyring,
            bucket=bucket,
        )
        with pytest.raises(ConfigError, match=match, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG) as exc:
            async with Scheduler(config):
                pass
        assert exc.value.action == "FIX_CONFIG"


class TestWorkerChecks:
    """What this worker must have: QEMU at or above the floor (checked by InstanceManager.open
    before the store probe), the arch, accelerator and CPU vendor of the host that saved
    (checked by each load, ROUTE before any download), and ACL support when QEMU runs as
    qemu-vm (the migration file is private to this user)."""

    @staticmethod
    def _settings(tmp_path: Path) -> Settings:
        return Settings(
            instance_store="local",
            instance_store_dir=tmp_path / "instances",
            instance_asset_dir=tmp_path / "assets",
            instance_keyring=new_keyring(),
        )

    @staticmethod
    def _qemu(monkeypatch: pytest.MonkeyPatch, version: tuple[int, int, int] | None) -> None:
        """The QEMU version probe of the manager finds `version` (None: no version found)."""

        async def probe_qemu_version() -> tuple[int, int, int] | None:
            return version

        monkeypatch.setattr("exec_sandbox.instance.manager.probe_qemu_version", probe_qemu_version)

    @pytest.mark.parametrize(
        ("version", "found"),
        [((10, 99, 99), "10.99.99"), ((10, 2, 0), "10.2.0"), (None, "unknown")],
        ids=["just-below", "below", "unknown"],
    )
    async def test_qemu_below_the_floor_fails_the_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: tuple[int, int, int] | None, found: str
    ) -> None:
        assert constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION == (11, 0, 0)  # the floor the message names
        self._qemu(monkeypatch, version)
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.MIGRATION_UNSUPPORTED) as exc:
            async with InstanceManager.open(
                self._settings(tmp_path), work_dir=tmp_path / "work", vm_manager=MagicMock()
            ):
                pass
        assert str(exc.value) == f"Saved instances need QEMU >= 11.0.0; this worker has {found}."
        assert exc.value.hint == "Upgrade QEMU on this worker, or unset instance_store."
        assert exc.value.action == "FIX_CONFIG"
        assert not (tmp_path / "work").exists()  # refused before the folders and the store probe
        assert not (tmp_path / "assets").exists()

    async def test_work_folder_in_memory_fails_the_open(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A save writes the VM's memory into the work folder: on tmpfs that is memory charged to
        the VM, which then runs out of it (OOM-killed mid-save without swap)."""
        self._qemu(monkeypatch, (11, 0, 0))
        monkeypatch.setattr(
            "exec_sandbox.instance.manager.memory_filesystem", lambda path: "tmpfs" if path == tmp_path else None
        )
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG) as exc:
            async with InstanceManager.open(
                self._settings(tmp_path), work_dir=tmp_path / "work", vm_manager=MagicMock()
            ):
                pass
        assert "is on tmpfs" in str(exc.value)
        assert exc.value.hint == "Set EXEC_SANDBOX_HOME_DISK_DIR (it holds this work folder) to a folder on disk."
        assert not (tmp_path / "work").exists()

    async def test_qemu_at_the_floor_opens(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._qemu(monkeypatch, (11, 0, 0))  # the documented floor, as a literal
        async with InstanceManager.open(self._settings(tmp_path), work_dir=tmp_path / "work", vm_manager=MagicMock()):
            assert (tmp_path / "work").is_dir()

    async def test_the_store_closes_after_the_loads_in_flight(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The open block ends (Scheduler exit closes the store) while a load still reads the
        store: the block ends only once the load ended, never under it. The store is the seam:
        its read waits until the block ends."""
        self._qemu(monkeypatch, (11, 0, 0))
        reading, block_ends = asyncio.Event(), asyncio.Event()
        get = LocalInstanceStore.get

        async def held_get(self: LocalInstanceStore, key: str, /, *, max_bytes: int) -> tuple[bytes, str] | None:
            reading.set()
            await block_ends.wait()
            return await get(self, key, max_bytes=max_bytes)

        async with InstanceManager.open(
            self._settings(tmp_path), work_dir=tmp_path / "work", vm_manager=MagicMock()
        ) as manager:
            monkeypatch.setattr(LocalInstanceStore, "get", held_get)  # after the store probe
            loading = asyncio.create_task(manager.load("gone", packages_image=no_packages))
            await reading.wait()
            asyncio.get_running_loop().call_soon(block_ends.set)  # once the block's end waits
        assert loading.done()
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND):
            await loading

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                (
                    ("aarch64", "hvf", None, "host"),
                    (HostArch.AARCH64, AccelType.TCG, None, "neoverse-v1,pauth=off"),
                    {"accelerator": "hvf", "cpu_model": "host"},
                    {"accelerator": "tcg", "cpu_model": "neoverse-v1,pauth=off"},
                ),
                id="accelerator",
            ),
            pytest.param(
                (
                    ("x86_64", "kvm", "GenuineIntel", "m,vendor=GenuineIntel"),
                    (HostArch.X86_64, AccelType.KVM, "AuthenticAMD", "m,vendor=AuthenticAMD"),
                    {"cpu_vendor": "GenuineIntel", "cpu_model": "m,vendor=GenuineIntel"},
                    {"cpu_vendor": "AuthenticAMD", "cpu_model": "m,vendor=AuthenticAMD"},
                ),
                id="cpu-vendor",
            ),
            pytest.param(
                (
                    ("x86_64", "kvm", "GenuineIntel", "m,+tsc-deadline,vendor=GenuineIntel"),
                    (HostArch.X86_64, AccelType.KVM, "GenuineIntel", "m,vendor=GenuineIntel"),
                    {"cpu_model": "m,+tsc-deadline,vendor=GenuineIntel"},
                    {"cpu_model": "m,vendor=GenuineIntel"},
                ),
                id="cpu-model-tsc-deadline",
            ),
            pytest.param(
                (
                    ("x86_64", "kvm", "GenuineIntel", "m,vendor=GenuineIntel"),
                    (HostArch.AARCH64, AccelType.KVM, None, "host"),
                    {"arch": "x86_64", "cpu_vendor": "GenuineIntel", "cpu_model": "m,vendor=GenuineIntel"},
                    {"arch": "aarch64", "cpu_vendor": "None", "cpu_model": "host"},
                ),
                id="arch",
            ),
        ],
    )
    async def test_load_on_another_host_routes_before_any_download(
        self,
        instance_env: InstanceEnv,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        case: tuple[
            tuple[str, str, str | None, str],
            tuple[HostArch, AccelType, str | None, str],
            dict[str, str],
            dict[str, str],
        ],
    ) -> None:
        """(the head's arch, accelerator, cpu_vendor and cpu_model; what this host's probes find;
        the error's `required` and `found`). cpu-model-tsc-deadline: same vendor, but the saving
        host had TSC_DEADLINE (in the guest CPU model): QEMU would refuse it after the download."""
        saved, (arch, accelerator, vendor, cpu_model), required, found = case

        async def detect_accel_type(*, force_emulation: bool) -> AccelType:
            return accelerator

        async def guest_cpu_model(_arch: HostArch, _accel: AccelType) -> str:
            return cpu_model

        monkeypatch.setattr("exec_sandbox.instance.manager.detect_host_arch", lambda: arch)
        monkeypatch.setattr("exec_sandbox.instance.manager.detect_accel_type", detect_accel_type)
        monkeypatch.setattr("exec_sandbox.instance.manager.detect_host_x86_vendor", lambda: vendor)
        monkeypatch.setattr("exec_sandbox.instance.manager.guest_cpu_model", guest_cpu_model)
        host = HostFacts(
            arch=saved[0],
            accelerator=saved[1],
            cpu_vendor=saved[2],
            cpu_model=saved[3],
            qemu_version="11.0.0",
            base_image="0" * 64,
            packages_image=None,
            release=__version__,
        )
        scratch = tmp_path / "seed"
        scratch.mkdir()
        await seed_instance(
            instance_env.store,
            scratch,
            instance_id="my-vm",
            keyring=instance_env.keyring,
            spec=VmSpec(language=Language.PYTHON),
            host=host,
        )
        mark = len(instance_env.proxy.log)
        with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
            await instance_env.load("my-vm", manager=instance_env.manager)
        assert (exc.value.action, exc.value.instance_id) == ("ROUTE", "my-vm")
        assert (exc.value.required, exc.value.found) == (required, found)
        assert instance_env.proxy.entries(since=mark) == [("GET", "s/my-vm/head", 200)]  # the head only
        assert instance_env.work_files() == set()

    async def test_load_of_another_release_routes_before_any_download(
        self, instance_env: InstanceEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An instance another exec-sandbox release saved: its guest agent runs on in the saved
        RAM, so the load is refused (ROUTE, the release in `required` and `found`) before any
        download, never migrated. The other facts match this host."""

        async def detect_accel_type(*, force_emulation: bool) -> AccelType:
            return AccelType.HVF

        async def guest_cpu_model(_arch: HostArch, _accel: AccelType) -> str:
            return "host"

        monkeypatch.setattr("exec_sandbox.instance.manager.detect_host_arch", lambda: HostArch.AARCH64)
        monkeypatch.setattr("exec_sandbox.instance.manager.detect_accel_type", detect_accel_type)
        monkeypatch.setattr("exec_sandbox.instance.manager.guest_cpu_model", guest_cpu_model)
        host = HostFacts(
            arch="aarch64",
            accelerator="hvf",
            cpu_vendor=None,
            cpu_model="host",
            qemu_version="11.0.0",
            base_image="0" * 64,
            packages_image=None,
            release="0.0.1",
        )
        scratch = tmp_path / "seed"
        scratch.mkdir()
        await seed_instance(
            instance_env.store,
            scratch,
            instance_id="my-vm",
            keyring=instance_env.keyring,
            spec=VmSpec(language=Language.PYTHON),
            host=host,
        )
        mark = len(instance_env.proxy.log)
        with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
            await instance_env.load("my-vm", manager=instance_env.manager)
        assert (exc.value.action, exc.value.instance_id) == ("ROUTE", "my-vm")
        assert (exc.value.required, exc.value.found) == ({"release": "0.0.1"}, {"release": __version__})
        assert instance_env.proxy.entries(since=mark) == [("GET", "s/my-vm/head", 200)]  # the head only
        assert instance_env.work_files() == set()

    async def test_save_without_acl_support_names_the_dependency(
        self, instance_env: InstanceEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QEMU runs as qemu-vm and this host cannot give it write access to the migration
        file (no setfacl): ConfigError HOST_DEPENDENCY (FIX_CONFIG) before QEMU is asked and
        before any store call; the save's files go."""
        monkeypatch.setattr("exec_sandbox.instance.manager.grant_qemu_vm_acl", AsyncMock(return_value=False))
        vm = await instance_env.new_vm(SparseImage(size=BLOCK), SparseImage(size=BLOCK), spec=PYTHON)
        vm.use_qemu_vm_user = True
        mark = len(instance_env.proxy.log)
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc:
            await instance_env.manager.save(vm, None, new_instance_id=None)  # type: ignore[arg-type]
        assert exc.value.message == (
            "Could not give qemu-vm write access to the migration file (needs setfacl, package acl)"
        )
        assert vm.qmp.commands == []
        assert instance_env.proxy.entries(since=mark) == []
        await instance_env.manager.drain()
        assert instance_env.work_files() == set()

    async def test_load_without_acl_support_names_the_dependency(
        self, instance_env: InstanceEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QEMU runs as qemu-vm and this host cannot give it read access to the loaded
        migration file (no setfacl): ConfigError HOST_DEPENDENCY (FIX_CONFIG), never a store
        or disk error; the loaded files go, and the instance stays intact."""
        vm = await instance_env.new_vm(SparseImage(size=BLOCK), SparseImage(size=BLOCK), spec=PYTHON)
        state = await Saver(instance_env, vm, manager=instance_env.manager, state=None).save()
        await instance_env.manager.drain()
        monkeypatch.setattr("exec_sandbox.instance.manager.probe_qemu_vm_user", AsyncMock(return_value=True))
        monkeypatch.setattr("exec_sandbox.instance.manager.grant_qemu_vm_acl", AsyncMock(return_value=False))
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc:
            await instance_env.load(state.instance_id, manager=instance_env.manager)
        assert exc.value.message == (
            "Could not give qemu-vm read access to the migration file (needs setfacl, package acl)"
        )
        assert exc.value.instance_id == state.instance_id
        await instance_env.manager.drain()
        assert instance_env.work_files() == {state.folder, state.vmstate, state.home}  # the saver's own base

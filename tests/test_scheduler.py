"""Tests for Scheduler.

Unit tests: Test validation, config, error handling (no QEMU needed).
Integration tests: Test real VM execution (requires QEMU + images).
"""

import asyncio
import contextlib
import json
import logging
import platform
import shutil
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, NamedTuple
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from exec_sandbox import package_validator
from exec_sandbox.config import SchedulerConfig
from exec_sandbox.disk_snapshot_manager import DiskSnapshotManager
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorAction,
    ErrorCode,
    InternalError,
    InvalidInputError,
    RouteError,
    TransientError,
)
from exec_sandbox.files.locks import CacheEntry, EntryLease
from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager
from exec_sandbox.models import ExecutionResult, ExposedPort, Language, PortMapping, StartupTiming, TimingBreakdown
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.scheduler import Scheduler, settings_from_config
from exec_sandbox.session import Session
from exec_sandbox.settings import Settings
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_timing import VmTiming
from exec_sandbox.vm_types import VmState
from tests.conftest import (
    create_test_qcow2,
    l1_hit,
    make_destroy_mock,
    run_once,
    run_timed,
    skip_unless_fast_virt,
    skip_unless_hwaccel,
)
from tests.instance_fixtures import primed_l1_dir_fixture  # noqa: F401 - fixture "primed_l1_dir"
from tests.test_files_locks import evictable

# ============================================================================
# Unit Tests - No QEMU needed
# ============================================================================


class TestSchedulerInit:
    """Tests for Scheduler initialization."""

    def test_init_default_config(self) -> None:
        """Without a config, the Scheduler has the default SchedulerConfig."""
        scheduler = Scheduler()
        assert scheduler.config == SchedulerConfig()

    def test_init_custom_config(self) -> None:
        """Scheduler accepts custom config."""
        config = SchedulerConfig(default_timeout_seconds=60)
        scheduler = Scheduler(config)
        assert scheduler.config.default_timeout_seconds == 60


class TestCreateSettings:
    """SchedulerConfig -> Settings bridge: code wins, else env var, else default."""

    @staticmethod
    def _settings(config: SchedulerConfig, tmp_path: Path) -> Settings:
        return settings_from_config(config, images_dir=tmp_path)

    def test_env_var_used_when_config_unset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_S3_BUCKET", "env-bucket")
        monkeypatch.setenv("EXEC_SANDBOX_MEMORY_OVERCOMMIT_RATIO", "3.5")
        settings = self._settings(SchedulerConfig(), tmp_path)
        assert settings.s3_bucket == "env-bucket"
        assert settings.memory_overcommit_ratio == 3.5

    def test_disk_overcommit_ratio_is_a_setting(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Like the memory and CPU ratios: the default, the env var, a config value that wins
        over it, and the bounds (1.0: the whole disk; 64.0)."""
        assert self._settings(SchedulerConfig(), tmp_path).disk_overcommit_ratio == 4.0
        monkeypatch.setenv("EXEC_SANDBOX_DISK_OVERCOMMIT_RATIO", "2.0")
        assert self._settings(SchedulerConfig(), tmp_path).disk_overcommit_ratio == 2.0
        assert self._settings(SchedulerConfig(disk_overcommit_ratio=8.0), tmp_path).disk_overcommit_ratio == 8.0
        for ratio in (0.5, 65.0):
            with pytest.raises(ValidationError):
                SchedulerConfig(disk_overcommit_ratio=ratio)

    def test_config_value_beats_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_S3_BUCKET", "env-bucket")
        settings = self._settings(SchedulerConfig(s3_bucket="code-bucket"), tmp_path)
        assert settings.s3_bucket == "code-bucket"

    def test_explicit_none_beats_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_S3_BUCKET", "env-bucket")
        settings = self._settings(SchedulerConfig(s3_bucket=None), tmp_path)
        assert settings.s3_bucket is None

    @pytest.mark.parametrize(
        ("name", "value", "why"),
        [
            ("EXEC_SANDBOX_HOST_MEMORY_RESERVE_RATIO", "10", "Input should be less than or equal to 0.5"),
            ("EXEC_SANDBOX_MAX_CONCURRENT_S3_UPLOADS", "0", "Input should be greater than or equal to 1"),
            ("EXEC_SANDBOX_INSTANCE_STORE", "s3cr3t-t0ken", "Input should be 's3' or 'local'"),
        ],
    )
    def test_invalid_env_var_is_setting_invalid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str, why: str
    ) -> None:
        """Env vars get the same bounds as SchedulerConfig fields. A value Settings refuses is a
        ConfigError SETTING_INVALID (a SandboxError, not pydantic's error) that names the variable
        and why, never the value (a variable can hold a secret), with no chained error that holds it."""
        monkeypatch.setenv(name, value)
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID) as exc:
            self._settings(SchedulerConfig(), tmp_path)
        assert str(exc.value) == f"Invalid EXEC_SANDBOX_* variables: {name}: {why}."
        assert exc.value.context == {"variables": [name]}
        assert exc.value.action is ErrorAction.FIX_CONFIG
        assert value not in f"{exc.value} {exc.value.hint} {exc.value.context}"
        assert exc.value.__cause__ is None
        assert exc.value.__suppress_context__

    def test_rule_across_variables_says_why(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rule across fields (overlapping S3 prefixes) has no single variable: its reason only."""
        monkeypatch.setenv("EXEC_SANDBOX_S3_INSTANCE_PREFIX", "a/")
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.SETTING_INVALID) as exc:
            self._settings(SchedulerConfig(s3_snapshot_prefix="a/b/"), tmp_path)
        assert str(exc.value).startswith("Invalid EXEC_SANDBOX_* variables: Value error, s3_snapshot_prefix ")
        assert exc.value.context == {"variables": []}

    def test_config_value_is_still_checked_by_the_config(self) -> None:
        """A bad value set in code stays the caller's own error, at construction."""
        with pytest.raises(ValidationError):
            SchedulerConfig(max_concurrent_s3_uploads=0)

    def test_defaults_without_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No env var and no config value: the documented defaults (no bucket, "snapshots/", and
        the L2 folder in the cache folder, here the one EXEC_SANDBOX_CACHE_DIR names)."""
        monkeypatch.delenv("EXEC_SANDBOX_S3_BUCKET", raising=False)
        monkeypatch.delenv("EXEC_SANDBOX_S3_SNAPSHOT_PREFIX", raising=False)
        monkeypatch.setenv("EXEC_SANDBOX_CACHE_DIR", str(tmp_path / "cache"))
        settings = self._settings(SchedulerConfig(), tmp_path)
        assert settings.s3_bucket is None
        assert settings.s3_snapshot_prefix == "snapshots/"
        assert settings.disk_snapshot_cache_dir == tmp_path / "cache" / "disk-snapshots"


class _Fakes(NamedTuple):
    """The stand-ins of a _fake_start: the scheduler gets them from its own __aenter__."""

    manager: MagicMock  # VmManager (_fake_manager)
    disk: MagicMock  # L2 cache (DiskSnapshotManager)
    memory: Any  # L1 cache: the test's, else a stand-in with no L1 key (each VM cold boots)
    warm: MagicMock  # warm pool, made when warm_pool_size > 0
    store_closed: AsyncMock  # the close of the instance store, when saved instances are on


def _fake_manager() -> MagicMock:
    """A VmManager stand-in: admission admits at once (its reservation is `manager.reservation`),
    create_vm and restore_vm do what the test sets, no VM is left at the stop, and destroy_vm
    confirms the process death."""
    manager = MagicMock(start=AsyncMock(), stop=AsyncMock(), create_vm=AsyncMock(), restore_vm=AsyncMock())
    manager.get_active_vms.return_value = {}
    manager.reservation = MagicMock()
    admission = MagicMock()
    admission.__aenter__ = AsyncMock(return_value=manager.reservation)
    admission.__aexit__ = AsyncMock(return_value=False)
    manager.reservation_context = MagicMock(return_value=admission)
    manager.destroy_vm = make_destroy_mock("confirmed")
    return manager


def _store(**methods: object) -> MagicMock:
    """An InstanceManager stand-in with `methods`; its stop (the Scheduler stop waits for it)
    ends at once."""
    return MagicMock(**{"stop": AsyncMock(), **methods})


@contextlib.contextmanager
def _fake_start(
    scheduler: Scheduler,
    tmp_path: Path,
    *,
    manager: MagicMock | None = None,
    memory: object = None,
    instances: MagicMock | None = None,
) -> Iterator[_Fakes]:
    """A start of `scheduler` (its own __aenter__) with no host work: no download, and the
    stand-ins of _Fakes in place of the classes it makes. Saved instances are off, or on with
    an instance store that opens to `instances`."""
    fakes = _Fakes(
        manager=manager if manager is not None else _fake_manager(),
        disk=MagicMock(start=AsyncMock(), stop=AsyncMock()),
        memory=memory if memory is not None else MagicMock(stop=AsyncMock(), compute_cache_key=AsyncMock()),
        warm=MagicMock(start=AsyncMock(), stop=AsyncMock(), get_vm=AsyncMock(return_value=None)),
        store_closed=AsyncMock(),
    )
    if memory is None:
        fakes.memory.compute_cache_key.return_value = None

    @contextlib.asynccontextmanager
    async def open_store(*_args: object, **_kwargs: object) -> AsyncIterator[MagicMock | None]:
        try:
            yield instances
        finally:
            await fakes.store_closed()

    with (
        patch("exec_sandbox.scheduler.ensure_assets", new_callable=AsyncMock, return_value=tmp_path),
        patch("exec_sandbox.scheduler.ensure_base_image", new_callable=AsyncMock),
        patch(
            "exec_sandbox.scheduler.settings_from_config",
            return_value=MagicMock(instance_store=None if instances is None else "local"),
        ),
        patch("exec_sandbox.scheduler.require_instance_manager", return_value=SimpleNamespace(open=open_store)),
        patch("exec_sandbox.scheduler.VmManager", return_value=fakes.manager),
        patch("exec_sandbox.scheduler.DiskSnapshotManager", return_value=fakes.disk),
        patch("exec_sandbox.scheduler.MemorySnapshotManager", return_value=fakes.memory),
        patch("exec_sandbox.scheduler.WarmVMPool", return_value=fakes.warm),
    ):
        yield fakes


@contextlib.asynccontextmanager
async def _running(
    config: SchedulerConfig,
    tmp_path: Path,
    *,
    manager: MagicMock | None = None,
    memory: object = None,
    instances: MagicMock | None = None,
) -> AsyncIterator[tuple[Scheduler, _Fakes]]:
    """A Scheduler of `config` inside its own `async with`, started by _fake_start."""
    scheduler = Scheduler(config)
    with _fake_start(scheduler, tmp_path, manager=manager, memory=memory, instances=instances) as fakes:
        async with scheduler:
            yield scheduler, fakes


class _LoadReachedError(Exception):
    """The instance store's load was called: reuse() got past its argument checks."""


class TestSchedulerContextManager:
    """Tests for Scheduler context manager."""

    async def test_double_start_raises(self, tmp_path: Path) -> None:
        """A second start of a started Scheduler raises InvalidInputError SCHEDULER_STATE."""
        scheduler = Scheduler(SchedulerConfig(images_dir=tmp_path))

        with _fake_start(scheduler, tmp_path):
            async with scheduler:
                with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE) as exc_info:
                    await scheduler.__aenter__()

        assert str(exc_info.value) == "Scheduler already started"

    async def test_run_without_start_raises(self) -> None:
        """Calling run() without starting raises InvalidInputError SCHEDULER_STATE."""
        scheduler = Scheduler()

        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE) as exc_info:
            await run_once(scheduler, VmSpec(language=Language.PYTHON), "print(1)")

        assert str(exc_info.value) == "Scheduler not started. Use: async with Scheduler() as scheduler:"

    async def test_cancelled_partial_startup_waits_for_owned_rollback(self, tmp_path: Path) -> None:
        """Repeated cancellation cannot abandon a manager started by __aenter__: a second cancel
        comes while the rollback stops the VM manager, and __aenter__ still waits for the whole
        rollback. Then new() refuses, and a next start works (it raises if the rollback was
        cancelled)."""
        scheduler = Scheduler(SchedulerConfig(warm_pool_size=1))
        manager_stop_entered = asyncio.Event()
        release_manager_stop = asyncio.Event()
        warm_start_entered = asyncio.Event()

        async def blocked_manager_stop() -> None:
            manager_stop_entered.set()
            await release_manager_stop.wait()

        async def blocked_warm_start() -> None:
            warm_start_entered.set()
            await asyncio.Event().wait()

        with _fake_start(scheduler, tmp_path) as fakes:
            fakes.manager.stop.side_effect = blocked_manager_stop
            fakes.warm.start.side_effect = blocked_warm_start
            startup = asyncio.create_task(scheduler.__aenter__())
            await warm_start_entered.wait()
            startup.cancel()
            await manager_stop_entered.wait()
            startup.cancel()
            await asyncio.sleep(0)  # one loop pass: the second cancel reaches __aenter__
            assert not startup.done()

            release_manager_stop.set()
            with pytest.raises(asyncio.CancelledError):
                await startup

        fakes.warm.stop.assert_awaited_once()
        fakes.memory.stop.assert_awaited_once()
        fakes.manager.stop.assert_awaited_once()
        await _assert_not_started(scheduler)
        with _fake_start(scheduler, tmp_path):
            async with scheduler:
                pass

    async def test_start_and_stop_during_a_start_raise(self, tmp_path: Path) -> None:
        """While a start runs, a second start and a stop raise InvalidInputError SCHEDULER_STATE
        and change nothing: the first start ends with the one VM manager it made, and its own
        stop stops it."""
        scheduler = Scheduler(SchedulerConfig(warm_pool_size=1))
        warm_start_entered = asyncio.Event()
        release_warm_start = asyncio.Event()

        async def first_warm_start_waits() -> None:
            if not warm_start_entered.is_set():
                warm_start_entered.set()
                await release_warm_start.wait()

        with _fake_start(scheduler, tmp_path) as fakes:
            fakes.warm.start.side_effect = first_warm_start_waits
            startup = asyncio.create_task(scheduler.__aenter__())
            await warm_start_entered.wait()
            with pytest.raises(
                InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE, match="still starting"
            ):
                await scheduler.__aenter__()
            with pytest.raises(
                InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE, match="still starting"
            ):
                await scheduler.__aexit__(None, None, None)
            release_warm_start.set()
            assert await startup is scheduler

        fakes.manager.start.assert_awaited_once()
        fakes.manager.stop.assert_not_awaited()
        await scheduler.__aexit__(None, None, None)
        fakes.manager.stop.assert_awaited_once()


async def test_start_lets_admission_evict_both_caches(tmp_path: Path) -> None:
    """Gate 4 evicts L2 and L1 entries before a VM waits for home disk space."""
    async with _running(SchedulerConfig(), tmp_path) as (_scheduler, fakes):
        assert fakes.manager.add_cache.call_args_list == [call(fakes.disk), call(fakes.memory)]


class TestNewInstanceId:
    """new(instance_id=...) checks the id, then that saved instances are on, before any VM starts."""

    @pytest.mark.parametrize("bad", ["My VM", "a/b", "", "a" * 65])
    async def test_invalid_id_fails_before_any_vm(self, tmp_path: Path, bad: str) -> None:
        async with _running(SchedulerConfig(), tmp_path, instances=_store()) as (scheduler, fakes):
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.INSTANCE_INVALID_ID):
                await scheduler.new(VmSpec(language=Language.PYTHON), instance_id=bad)
        fakes.manager.reservation_context.assert_not_called()

    async def test_id_without_a_store_fails_before_any_vm(self, tmp_path: Path) -> None:
        async with _running(SchedulerConfig(), tmp_path) as (scheduler, fakes):
            with pytest.raises(
                ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG, match="none is configured"
            ) as exc:
                await scheduler.new(VmSpec(language=Language.PYTHON), instance_id="my-vm")
        assert (exc.value.action, exc.value.instance_id) == (ErrorAction.FIX_CONFIG, "my-vm")
        fakes.manager.reservation_context.assert_not_called()

    async def test_valid_id_with_a_store_starts_the_vm(self, tmp_path: Path) -> None:
        """Control: with a store and a valid id, the same new() starts a VM."""
        async with _running(SchedulerConfig(), tmp_path, instances=_store()) as (scheduler, fakes):
            fakes.manager.create_vm.return_value = _fake_vm("with-an-id")
            session = await scheduler.new(VmSpec(language=Language.PYTHON), instance_id="my-vm")
            await session.close()
        assert session.vm_id == "with-an-id"
        fakes.manager.create_vm.assert_awaited_once()


class TestIdleTimeoutArgument:
    """new() and reuse(): idle_timeout_seconds has the range of
    SchedulerConfig.session_idle_timeout_seconds (10 to 3600); None takes the config's value.
    Out of range (0 included) is InvalidInputError ARGUMENT_INVALID before any VM or store call."""

    @pytest.mark.parametrize(("argument", "used"), [(None, 77), (10, 10), (3600, 3600)])
    async def test_new_session_gets_the_timeout(self, tmp_path: Path, argument: int | None, used: int) -> None:
        """The session gets the argument, else the config's value. The value is read from the
        session: the public effect (a close after that many idle seconds, 10 s or more) is
        too slow for a unit test; tests/test_session.py tests that the timer fires."""
        async with _running(SchedulerConfig(session_idle_timeout_seconds=77), tmp_path) as (scheduler, fakes):
            fakes.manager.create_vm.return_value = _fake_vm("idle-timeout")
            session = await scheduler.new(VmSpec(language=Language.PYTHON), idle_timeout_seconds=argument)
            await session.close()
        assert session._idle_timeout_seconds == used

    @pytest.mark.parametrize("argument", [0, 9, 3601, -1])
    async def test_new_out_of_range(self, tmp_path: Path, argument: int) -> None:
        async with _running(SchedulerConfig(), tmp_path) as (scheduler, fakes):
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID) as exc:
                await scheduler.new(VmSpec(language=Language.PYTHON), idle_timeout_seconds=argument)
        assert str(exc.value) == f"idle_timeout_seconds must be between 10 and 3600, got {argument}"
        fakes.manager.reservation_context.assert_not_called()

    @pytest.mark.parametrize("argument", [0, 3601])
    async def test_reuse_out_of_range_sends_no_request(self, tmp_path: Path, argument: int) -> None:
        instances = _store(load=AsyncMock())
        async with _running(SchedulerConfig(), tmp_path, instances=instances) as (scheduler, _fakes):
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID):
                await scheduler.reuse("my-vm", idle_timeout_seconds=argument)
        instances.load.assert_not_awaited()

    async def test_reuse_in_range_goes_on(self, tmp_path: Path) -> None:
        """Control: the same reuse() with a valid timeout goes on to the load."""
        instances = _store(load=AsyncMock(side_effect=_LoadReachedError))
        async with _running(SchedulerConfig(), tmp_path, instances=instances) as (scheduler, _fakes):
            with pytest.raises(_LoadReachedError):
                await scheduler.reuse("my-vm", idle_timeout_seconds=10)
        instances.load.assert_awaited_once()


class TestInstanceCallsNeedAStart:
    """Saved-instance calls on a Scheduler outside `async with` fail before any store call."""

    @pytest.mark.parametrize("call", ["describe", "delete", "list", "reuse"])
    async def test_not_started(self, call: str) -> None:
        scheduler = Scheduler()
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE) as exc:
            match call:
                case "describe":
                    await scheduler.describe_instance("my-vm")
                case "delete":
                    await scheduler.delete_instance("my-vm")
                case "list":
                    _ = [instance_id async for instance_id in scheduler.list_instances()]
                case _:
                    await scheduler.reuse("my-vm")
        assert str(exc.value) == "Scheduler not started. Use: async with Scheduler() as scheduler:"


class TestReuseRestoreFailure:
    """reuse() after its load: a VM that does not start from the saved state releases the
    loaded files; a state QEMU refuses routes to a worker with the QEMU that wrote it."""

    @staticmethod
    def _loaded() -> tuple[MagicMock, SimpleNamespace]:
        """An instance store whose load gives a saved state that QEMU 11.0.2 wrote; returns
        (the store, the state)."""
        state = SimpleNamespace(
            body=SimpleNamespace(spec=VmSpec(language=Language.PYTHON), host=SimpleNamespace(qemu_version="11.0.2")),
            header=SimpleNamespace(generation=3),
            vmstate=Path("/saved/c.vmstate"),
            base_image=Path("/saved/base.img"),
            packages_image=None,
            folder=Path("/saved"),
        )
        return _store(load=AsyncMock(return_value=state), release=AsyncMock()), state

    async def test_state_qemu_refuses_routes(self, tmp_path: Path) -> None:
        error = TransientError(
            "L1 restore failed", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None
        )
        error.__cause__ = RouteError(
            "Migration failed: incompatible stream",
            code=ErrorCode.MIGRATION_REJECTED,
            required={},
            found={},
            context=None,
            hint=None,
            instance_id=None,
        )
        instances, state = self._loaded()

        async with _running(SchedulerConfig(), tmp_path, instances=instances) as (scheduler, fakes):
            fakes.manager.restore_vm.side_effect = error
            with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
                await scheduler.reuse("my-vm")

        assert (exc.value.action, exc.value.instance_id) == (ErrorAction.ROUTE, "my-vm")
        assert (exc.value.required, exc.value.found) == (
            {"qemu_version": "11.0.2"},
            {"qemu_version": "this worker's QEMU refused the saved state"},
        )
        assert exc.value.__cause__ is error
        instances.release.assert_awaited_once_with(state)
        # Admission leaves out the loaded state's files: they stay while the request waits
        assert fakes.manager.reservation_context.call_args.kwargs["held"] == {Path("/saved")}

    async def test_other_restore_failure_is_raised_as_it_is(self, tmp_path: Path) -> None:
        error = TransientError(
            "guest did not answer", code=ErrorCode.GUEST_SILENT_AFTER_RESTORE, context=None, hint=None, instance_id=None
        )
        instances, state = self._loaded()

        async with _running(SchedulerConfig(), tmp_path, instances=instances) as (scheduler, fakes):
            fakes.manager.restore_vm.side_effect = error
            with pytest.raises(TransientError) as exc:
                await scheduler.reuse("my-vm")

        assert exc.value is error
        instances.release.assert_awaited_once_with(state)


def _fake_vm(vm_id: str) -> MagicMock:
    """A QemuVM mock with the fields new() reads for the startup timing (0, no L1 restore),
    in the READY state (the destroy mock publishes DESTROYED, or not: "unconfirmed")."""
    vm = MagicMock(spec=QemuVM)
    vm.vm_id = vm_id
    vm.l1_restored = False
    vm.state = VmState.READY
    vm.process = MagicMock()
    vm.process.returncode = None
    # `timing` is an instance attribute, invisible to spec=QemuVM.
    vm.timing = VmTiming(
        boot_ms=0, overlay_ms=0, qemu_cmd_build_ms=0, gvproxy_start_ms=0, qemu_fork_ms=0, guest_wait_ms=0
    )
    return vm


class TestOneShotTeardownConfirmation:
    """new() + exec() + close() must not claim success before process death is proven."""

    @staticmethod
    def _cold_boots(fakes: _Fakes, *, destroy: str) -> MagicMock:
        """The VM the next new() cold boots: its exec() prints "ok"; destroy_vm as `destroy`
        says (make_destroy_mock). Returns the VM."""
        vm = _fake_vm("one-shot")
        vm.execute = AsyncMock(
            return_value=ExecutionResult(
                stdout="ok\n",
                stderr="",
                exit_code=0,
                timing=TimingBreakdown(execute_ms=1, connect_ms=None),
            )
        )
        fakes.manager.create_vm.return_value = vm
        fakes.manager.destroy_vm = make_destroy_mock(destroy)
        return vm

    async def test_unconfirmed_vm_cleanup_raises_instead_of_returning_result(self, tmp_path: Path) -> None:
        async with _running(SchedulerConfig(), tmp_path) as (scheduler, fakes):
            vm = self._cold_boots(fakes, destroy="unconfirmed")
            with pytest.raises(
                InternalError,
                check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED,
                match="process cleanup was not confirmed",
            ):
                await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')")

        fakes.manager.destroy_vm.assert_awaited_once_with(vm)

    async def test_confirmed_process_death_returns_result_while_ancillary_cleanup_retries(self, tmp_path: Path) -> None:
        """Workdir/cgroup retry must not mask a result after process death."""
        async with _running(SchedulerConfig(), tmp_path) as (scheduler, fakes):
            vm = self._cold_boots(fakes, destroy="ancillary_pending")
            result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')")

        assert result.stdout == "ok\n"
        fakes.manager.destroy_vm.assert_awaited_once_with(vm)

    async def test_confirmed_vm_cleanup_returns_result(self, tmp_path: Path) -> None:
        async with _running(SchedulerConfig(), tmp_path) as (scheduler, fakes):
            vm = self._cold_boots(fakes, destroy="confirmed")
            result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')")

        assert result.stdout == "ok\n"
        fakes.manager.destroy_vm.assert_awaited_once_with(vm)

    async def test_execution_error_remains_primary_when_cleanup_is_unconfirmed(self, tmp_path: Path) -> None:
        async with _running(SchedulerConfig(), tmp_path) as (scheduler, fakes):
            vm = self._cold_boots(fakes, destroy="unconfirmed")
            vm.execute.side_effect = RuntimeError("execute failed")
            with pytest.raises(RuntimeError, match="execute failed"):
                await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')")

        fakes.manager.destroy_vm.assert_awaited_once_with(vm)

    async def test_cancellation_remains_primary_when_cleanup_is_unconfirmed(self, tmp_path: Path) -> None:
        async with _running(SchedulerConfig(), tmp_path) as (scheduler, fakes):
            vm = self._cold_boots(fakes, destroy="unconfirmed")
            vm.execute.side_effect = asyncio.CancelledError()
            with pytest.raises(asyncio.CancelledError):
                await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')")

        fakes.manager.destroy_vm.assert_awaited_once_with(vm)


class TestL1FallbackOwnership:
    """Cold-boot fallback cannot reuse a reservation retained by an L1 reaper.

    The L1 manager is real (its restore() runs) on a VmManager stand-in; its key and its
    cache lookup are stubbed to an entry in tmp_path. Each request is a new() (then a close()
    of its session).
    """

    @staticmethod
    async def _new(scheduler: Scheduler, spec: VmSpec) -> Session:
        """new(spec), then close the session (its VM id and ports stay readable)."""
        session = await scheduler.new(spec)
        await session.close()
        return session

    @staticmethod
    @contextlib.asynccontextmanager
    async def _running_l1(restore_error: Exception | None, tmp_path: Path) -> AsyncIterator[tuple[Scheduler, _Fakes]]:
        """A started Scheduler whose L1 lookup hits: the restore raises `restore_error` (None:
        it gives the VM "l1-restored"), and a cold boot gives "cold-boot-replacement"."""
        manager = _fake_manager()
        manager.restore_vm.side_effect = restore_error
        manager.restore_vm.return_value = _fake_vm("l1-restored")
        manager.create_vm.return_value = _fake_vm("cold-boot-replacement")
        memory = l1_hit(manager, tmp_path)
        memory.schedule_background_save = AsyncMock()  # type: ignore[method-assign]
        config = SchedulerConfig(auto_download_assets=False, enable_package_validation=False)
        async with _running(config, tmp_path, manager=manager, memory=memory) as running:
            yield running

    async def test_unconfirmed_restore_cleanup_propagates_without_cold_boot(self, tmp_path: Path) -> None:
        error = InternalError(
            "Restore failure cleanup was not confirmed",
            code=ErrorCode.CLEANUP_UNCONFIRMED,
            context=None,
            hint=None,
            instance_id=None,
        )
        async with self._running_l1(error, tmp_path) as (scheduler, fakes):
            with pytest.raises(
                InternalError,
                check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED,
                match="cleanup was not confirmed",
            ):
                await scheduler.new(VmSpec(language=Language.PYTHON, memory_mb=192))

        fakes.manager.create_vm.assert_not_awaited()

    async def test_template_qemu_refused_is_removed(self, tmp_path: Path) -> None:
        """A template QEMU refused is dropped (every file), then the request cold boots."""
        error = TransientError(
            "L1 restore failed", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None
        )
        error.__cause__ = RouteError(
            "Migration failed: incompatible stream",
            code=ErrorCode.MIGRATION_REJECTED,
            required={},
            found={},
            context=None,
            hint=None,
            instance_id=None,
        )
        async with self._running_l1(error, tmp_path) as (scheduler, fakes):
            session = await self._new(scheduler, VmSpec(language=Language.PYTHON, memory_mb=192))

        assert session.vm_id == "cold-boot-replacement"
        fakes.manager.create_vm.assert_awaited_once()
        assert not (tmp_path / "cached.vmstate").exists()

    async def test_l1_entry_is_leased_through_admission(self, tmp_path: Path) -> None:
        """Admission may evict cache entries for disk space: the L1 entry is leased before it
        (and named in `held`, so gate 4 does not count on its space), and the request's
        lease goes after the restore (the VM holds its own share)."""
        vmstate = tmp_path / "cached.vmstate"
        evictable_at_admission: list[bool] = []
        async with self._running_l1(None, tmp_path) as (scheduler, fakes):
            admission = fakes.manager.reservation_context.return_value.__aenter__
            admission.side_effect = lambda *_args: (
                evictable_at_admission.append(evictable(vmstate)) or admission.return_value
            )
            session = await self._new(scheduler, VmSpec(language=Language.PYTHON, memory_mb=192))

        assert session.vm_id == "l1-restored"
        assert evictable_at_admission == [False]
        assert fakes.manager.reservation_context.call_args.kwargs["held"] == {vmstate}
        assert evictable(vmstate)

    async def test_confirmed_transient_restore_failure_can_reuse_reservation(self, tmp_path: Path) -> None:
        error = TransientError(
            "migration failed", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None
        )
        async with self._running_l1(error, tmp_path) as (scheduler, fakes):
            session = await self._new(scheduler, VmSpec(language=Language.PYTHON, memory_mb=192))

        assert session.vm_id == "cold-boot-replacement"
        assert fakes.manager.create_vm.await_args.kwargs["reservation"] is fakes.manager.reservation
        assert (tmp_path / "cached.vmstate").exists()  # not refused by QEMU: kept

    async def test_exposed_ports_are_resolved_and_reach_the_vm(self, tmp_path: Path) -> None:
        spec = VmSpec(
            language=Language.PYTHON, memory_mb=192, expose_ports=[PortMapping(internal=8080, external=3000), 9000]
        )
        async with self._running_l1(None, tmp_path) as (scheduler, fakes):
            session = await self._new(scheduler, spec)
        ports = session.exposed_ports

        assert session.vm_id == "l1-restored"
        assert all(isinstance(p, ExposedPort) for p in ports)
        assert [p.internal for p in ports] == [8080, 9000]
        assert ports[0].external == 3000  # explicit: kept
        assert isinstance(ports[1].external, int)  # none given: a host port is allocated
        assert ports[1].external > 0
        assert fakes.manager.restore_vm.await_args.kwargs["expose_ports"] == ports

    async def test_post_create_snapshot_schedule_failure_destroys_unreturned_vm(self, tmp_path: Path) -> None:
        error = TransientError(
            "restore miss", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None
        )
        async with self._running_l1(error, tmp_path) as (scheduler, fakes):
            fakes.memory.schedule_background_save.side_effect = RuntimeError("schedule failed")
            with pytest.raises(RuntimeError, match="schedule failed"):
                await scheduler.new(VmSpec(language=Language.PYTHON, memory_mb=192))

        fakes.manager.destroy_vm.assert_awaited_once_with(fakes.manager.create_vm.return_value)

    @pytest.mark.parametrize(
        "restore_error",
        [
            None,
            TransientError(
                "migration failed", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None
            ),
        ],
        ids=["l1-hit", "cold"],
    )
    async def test_packages_request_uses_its_l2_layer_and_one_key(
        self, tmp_path: Path, restore_error: Exception | None
    ) -> None:
        """One L1 key per request, made with the L2 layer the request boots with; the
        restore, the cold boot and the background save get that same layer."""
        l2 = tmp_path / "l2.qcow2"
        l2.write_bytes(b"l2")
        lease = await CacheEntry(l2).lease()
        assert lease is not None
        spec = VmSpec(language=Language.PYTHON, memory_mb=192, packages=("pandas==2.2.3",))
        evictable_at_admission: list[bool] = []
        async with self._running_l1(restore_error, tmp_path) as (scheduler, fakes):
            fakes.disk.get_or_create_snapshot = AsyncMock(return_value=lease)
            admission = fakes.manager.reservation_context.return_value.__aenter__
            admission.side_effect = lambda *_args: (
                evictable_at_admission.append(evictable(l2)) or admission.return_value
            )
            session = await self._new(scheduler, spec)

        manager, memory = fakes.manager, fakes.memory
        # Leased from the lookup: admission (it may evict for disk space) cannot take the
        # layer; the request releases its lease at its end (each VM holds its own share)
        assert evictable_at_admission == [False]
        assert evictable(l2)

        fakes.disk.get_or_create_snapshot.assert_awaited_once()
        assert {
            k: v
            for k, v in fakes.disk.get_or_create_snapshot.await_args.kwargs.items()
            if k in {"language", "packages", "memory_mb"}
        } == {"language": Language.PYTHON, "packages": ["pandas==2.2.3"], "memory_mb": 192}
        memory.compute_cache_key.assert_awaited_once_with(spec, l2_path=l2)
        assert manager.restore_vm.await_args.kwargs["snapshot_drive"] == l2
        assert manager.restore_vm.await_args.kwargs["cache_leases"][1] is lease
        if restore_error is None:
            assert session.vm_id == "l1-restored"
            manager.create_vm.assert_not_awaited()
            memory.schedule_background_save.assert_not_awaited()
        else:
            assert manager.create_vm.await_args.kwargs["snapshot_drive"] == l2
            assert manager.create_vm.await_args.kwargs["cache_leases"] == (lease,)
            memory.schedule_background_save.assert_awaited_once_with(spec, key="cached", snapshot_drive=lease)


class TestPackageValidation:
    """Tests for package validation in Scheduler."""

    @staticmethod
    @contextlib.asynccontextmanager
    async def _running_build(tmp_path: Path, *, validation: bool) -> AsyncIterator[tuple[Scheduler, MagicMock]]:
        """A started scheduler whose L2 build raises RuntimeError("the build started"): a new()
        with packages ends there once the package check let it through. Yields (scheduler,
        the L2 cache)."""
        config = SchedulerConfig(enable_package_validation=validation, auto_download_assets=False)
        async with _running(config, tmp_path) as (scheduler, fakes):
            fakes.disk.get_or_create_snapshot = AsyncMock(side_effect=RuntimeError("the build started"))
            yield scheduler, fakes.disk

    @pytest.mark.parametrize("validation", [True, False], ids=["on", "off"])
    async def test_package_outside_the_allow_list(self, tmp_path: Path, validation: bool) -> None:
        """new() with a package outside the allow-list: refused before any build or VM when
        validation is on (InvalidInputError PACKAGE_NOT_ALLOWED); built when it is off."""
        spec = VmSpec(language=Language.PYTHON, packages=("surely-not-a-catalog-package==1.0",))

        async with self._running_build(tmp_path, validation=validation) as (scheduler, snapshots):
            if validation:
                with pytest.raises(
                    InvalidInputError,
                    check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED,
                    match="not in python allow-list",
                ):
                    await scheduler.new(spec)
                snapshots.get_or_create_snapshot.assert_not_awaited()
            else:
                with pytest.raises(RuntimeError, match="the build started"):
                    await scheduler.new(spec)

    async def test_catalogs_parsed_once_off_the_loop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bundled catalogs (about 400 KB of JSON, 2 ms of parse) are parsed once per
        Scheduler, in a worker thread; later requests reuse them and validate the same way."""
        parsers: list[int] = []

        def spied_loads(text: str) -> object:
            parsers.append(threading.get_ident())
            return json.loads(text)

        monkeypatch.setattr(package_validator, "json", SimpleNamespace(loads=spied_loads))
        allowed = (
            VmSpec(language=Language.PYTHON, packages=("pandas==2.2.0",)),
            VmSpec(language=Language.JAVASCRIPT, packages=("lodash@4.17.21",)),
        )
        async with self._running_build(tmp_path, validation=True) as (scheduler, _snapshots):
            for _ in range(3):
                for spec in allowed:
                    with pytest.raises(RuntimeError, match="the build started"):
                        await scheduler.new(spec)
            with pytest.raises(
                InvalidInputError,
                check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED,
                match="not in python allow-list",
            ):
                await scheduler.new(VmSpec(language=Language.PYTHON, packages=("surely-not-a-catalog-package==1.0",)))
        assert len(parsers) == 2  # the PyPI and the npm catalog, once each
        assert threading.get_ident() not in parsers


class TestSchedulerConfig:
    """Tests for Scheduler config handling."""

    def test_config_immutable(self) -> None:
        """The config of a Scheduler cannot change after it is made."""
        scheduler = Scheduler(SchedulerConfig(default_timeout_seconds=60))

        with pytest.raises(ValidationError, match="frozen"):
            scheduler.config.default_timeout_seconds = 5  # type: ignore[misc]
        assert scheduler.config.default_timeout_seconds == 60


class TestSchedulerSnapshotInit:
    """Tests for DiskSnapshotManager initialization in Scheduler."""

    async def test_snapshot_manager_with_s3_uses_the_scheduler_vm_manager(self, images_dir: Path) -> None:
        """With S3 settings, the Scheduler starts with an L2 cache that has the S3 bucket and
        whose build VMs come from the Scheduler's own VM manager (admission counts them).

        Read where the start makes them (the classes, spied): a build with S3 set would call
        S3, and the VM manager a build uses shows only in a build VM. Without S3,
        TestBaseImageFetchBeforeSnapshot builds through new()."""
        made: dict[str, list[tuple[Any, tuple[Any, ...]]]] = {"vm": [], "l2": []}

        def spy(name: str, cls: type) -> MagicMock:
            def make(*args: Any) -> object:
                made[name].append((cls(*args), args))
                return made[name][-1][0]

            return MagicMock(side_effect=make)

        config = SchedulerConfig(
            images_dir=images_dir,
            s3_bucket="test-bucket",
            s3_region="us-east-1",
            auto_download_assets=False,
        )
        with (
            patch("exec_sandbox.scheduler.VmManager", spy("vm", VmManager)),
            patch("exec_sandbox.scheduler.DiskSnapshotManager", spy("l2", DiskSnapshotManager)),
        ):
            async with Scheduler(config):
                [(vm_manager, _)], [(l2, (l2_settings, l2_vm_manager))] = made["vm"], made["l2"]
                assert (l2_settings.s3_bucket, l2_vm_manager, l2.vm_manager) == ("test-bucket", vm_manager, vm_manager)


class _ColdBootError(Exception):
    """create_vm was called: the request got to its cold boot (each request here ends there)."""


class TestBaseImageFetchBeforeSnapshot:
    """ensure_base_image runs before the L2 snapshot of a new() with packages.

    When EXEC_SANDBOX_CACHE_DIR points to a fresh directory (pip-install user),
    ensure_assets() at startup only downloads kernel/initramfs/gvproxy (no
    language param). If packages are requested, the L2 cache key needs the
    base image on disk to hash it. Without the ensure_base_image() call before
    the L2 lookup, GuestImages.base_image() finds no file: ConfigError ASSET_NOT_FOUND.

    Each request is a new() with packages (package check off) that ends at its cold boot:
    VmManager.create_vm raises _ColdBootError, and its snapshot_drive is the L2 snapshot.
    The stand-ins patch the classes (VmManager, MemorySnapshotManager, DiskSnapshotManager):
    the Scheduler makes its own instances at its start.
    """

    @pytest.fixture(autouse=True)
    def _empty_download_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty download cache: no base image of this user's cache stands in for the one
        the test expects ensure_base_image to bring (GuestImages falls back to it)."""
        monkeypatch.setenv("EXEC_SANDBOX_CACHE_DIR", str(tmp_path / "download-cache"))

    @staticmethod
    def _config(images: Path, *, auto_download: bool, **fields: Path) -> SchedulerConfig:
        return SchedulerConfig(
            images_dir=images,
            auto_download_assets=auto_download,
            enable_package_validation=False,
            default_timeout_seconds=120,
            **fields,
        )

    @staticmethod
    def _make_fresh_images_dir(images_dir: Path, tmp_path: Path) -> tuple[Path, str]:
        """Create isolated images dir with kernel/initramfs but no base images.

        Returns (fresh_images_path, arch_string).
        """
        fresh_images = tmp_path / "fresh-images"
        fresh_images.mkdir()
        arch = "aarch64" if platform.machine() == "arm64" else platform.machine()
        for name in images_dir.iterdir():
            if name.name.startswith(("vmlinuz", "vmlinux", "initramfs")):
                shutil.copy2(name, fresh_images / name.name)
        return fresh_images, arch

    @staticmethod
    @contextlib.contextmanager
    def _snapshot_lookup(tmp_path: Path, events: list[object]) -> Iterator[Path]:
        """DiskSnapshotManager.get_or_create_snapshot appends "snapshot" to `events` and
        returns a lease on a qcow2 made beforehand (yielded)."""
        snap = tmp_path / f"snap-{uuid4().hex}.qcow2"
        snap.touch()

        async def leased(**_kwargs: object) -> EntryLease | None:
            events.append("snapshot")
            return await CacheEntry(snap).lease()

        with patch.object(DiskSnapshotManager, "get_or_create_snapshot", AsyncMock(side_effect=leased)):
            yield snap

    @staticmethod
    @contextlib.contextmanager
    def _cold_boot_only() -> Iterator[AsyncMock]:
        """No L1 key (no L1 restore, no L1 save); VmManager.create_vm raises _ColdBootError.
        Yields create_vm."""
        create_vm = AsyncMock(side_effect=_ColdBootError)
        with (
            patch.object(MemorySnapshotManager, "compute_cache_key", AsyncMock(return_value=None)),
            patch.object(VmManager, "create_vm", create_vm),
        ):
            yield create_vm

    async def test_missing_base_image_fetched_before_cache_key(self, images_dir: Path, tmp_path: Path) -> None:
        """A fresh cache folder without the base image: no ConfigError ASSET_NOT_FOUND.

        The L2 cache key calls GuestImages.base_image(), which looks for the python base image.
        Without the ensure_base_image() call, it finds no file and raises
        ConfigError ASSET_NOT_FOUND. The DiskSnapshotManager is real (only its QEMU build
        is replaced: _create_snapshot boots the build VM, and no public call builds without
        one), so the key, the base image lookup and the L2 cache folder are real.
        """
        fresh_images, arch = self._make_fresh_images_dir(images_dir, tmp_path)

        cache_dir = tmp_path / "snapshot-cache"
        cache_dir.mkdir(mode=0o700)
        # ensure_base_image mock: writes a real qcow2 at the expected path
        # (simulates what the real downloader does)
        base_image_path = fresh_images / f"python-3.14-base-{arch}.qcow2"

        async def mock_fetch(language, **kwargs):  # type: ignore[no-untyped-def]
            # Write a minimal valid qcow2 so GuestImages.base_image() finds it
            await create_test_qcow2(base_image_path)
            return base_image_path

        # The QEMU build writes to the real cache_key path, so CacheEntry.saving's
        # "already exists" check stays correct.
        async def mock_create_snapshot(language, packages, *, cache_key, **kwargs):  # type: ignore[no-untyped-def]
            snap = cache_dir / f"{cache_key}.qcow2"
            await create_test_qcow2(snap)
            return snap

        config = self._config(fresh_images, auto_download=True, disk_snapshot_cache_dir=cache_dir)
        with (
            self._cold_boot_only() as create_vm,
            patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch),
            patch.object(DiskSnapshotManager, "_create_snapshot", side_effect=mock_create_snapshot),
        ):
            async with Scheduler(config) as scheduler:
                with pytest.raises(_ColdBootError):  # ConfigError ASSET_NOT_FOUND if the fetch came after the L2 key
                    await scheduler.new(VmSpec(language=Language.PYTHON, packages=("pandas==2.0.0",)))

        snapshot = create_vm.await_args.kwargs["snapshot_drive"]
        assert (snapshot.parent, snapshot.suffix) == (cache_dir, ".qcow2")
        assert snapshot.exists()

    @pytest.mark.parametrize("language", [Language.PYTHON, Language.JAVASCRIPT, Language.RAW])
    async def test_fetch_once_before_the_snapshot(self, images_dir: Path, tmp_path: Path, language: Language) -> None:
        """auto_download_assets=True + packages: ensure_base_image runs once, with the request's
        language, before the L2 lookup; the cold boot on that snapshot does not fetch again."""
        events: list[object] = []

        async def mock_fetch(lang, **kwargs):  # type: ignore[no-untyped-def]
            events.append(("fetch", lang))

        with self._cold_boot_only() as create_vm, self._snapshot_lookup(tmp_path, events) as snap:
            async with Scheduler(self._config(images_dir, auto_download=True)) as scheduler:
                with (
                    patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch),
                    pytest.raises(_ColdBootError),
                ):
                    await scheduler.new(VmSpec(language=language, packages=("some-pkg==1.0",)))

        assert events == [("fetch", language), "snapshot"]
        assert create_vm.await_args.kwargs["snapshot_drive"] == snap

    async def test_fetch_skipped_when_auto_download_disabled(self, images_dir: Path, tmp_path: Path) -> None:
        """auto_download_assets=False → ensure_base_image NOT called (images must exist locally)."""
        events: list[object] = []

        async def mock_fetch(language, **kwargs):  # type: ignore[no-untyped-def]
            events.append(("fetch", language))

        with self._cold_boot_only(), self._snapshot_lookup(tmp_path, events):
            async with Scheduler(self._config(images_dir, auto_download=False)) as scheduler:
                with (
                    patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch),
                    pytest.raises(_ColdBootError),
                ):
                    await scheduler.new(VmSpec(language=Language.PYTHON, packages=("pandas==2.0.0",)))

        assert events == ["snapshot"], "ensure_base_image should NOT be called when auto_download_assets=False"

    async def test_snapshot_path_skipped_without_packages(self, images_dir: Path, tmp_path: Path) -> None:
        """No packages: no L2 lookup, and the cold boot gets no snapshot drive."""
        with (
            self._cold_boot_only() as create_vm,
            patch.object(DiskSnapshotManager, "get_or_create_snapshot", new_callable=AsyncMock) as lookup,
        ):
            async with Scheduler(self._config(images_dir, auto_download=True)) as scheduler:
                with (
                    patch("exec_sandbox.scheduler.ensure_base_image", new_callable=AsyncMock),
                    pytest.raises(_ColdBootError),
                ):
                    await scheduler.new(VmSpec(language=Language.PYTHON))

        lookup.assert_not_awaited()
        assert create_vm.await_args.kwargs["snapshot_drive"] is None

    async def test_fetch_called_for_each_request(self, images_dir: Path, tmp_path: Path) -> None:
        """Each request with packages fetches the base image again (ensure_base_image is
        idempotent and fast when the file is there): an image removed between two requests
        comes back."""
        events: list[object] = []

        async def mock_fetch(language, **kwargs):  # type: ignore[no-untyped-def]
            events.append(("fetch", language))

        with self._cold_boot_only(), self._snapshot_lookup(tmp_path, events):
            async with Scheduler(self._config(images_dir, auto_download=True)) as scheduler:
                with patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch):
                    for _ in range(2):
                        with pytest.raises(_ColdBootError):
                            await scheduler.new(VmSpec(language=Language.PYTHON, packages=("pkg==1.0",)))

        assert events == [("fetch", Language.PYTHON), "snapshot"] * 2

    async def test_fetch_failure_propagates(self, images_dir: Path, tmp_path: Path) -> None:
        """ensure_base_image raising TransientError ASSET_DOWNLOAD propagates to caller."""

        async def mock_fetch_fail(language, **kwargs):  # type: ignore[no-untyped-def]
            raise TransientError(
                "Network unreachable", code=ErrorCode.ASSET_DOWNLOAD, context=None, hint=None, instance_id=None
            )

        with self._cold_boot_only() as create_vm:
            async with Scheduler(self._config(images_dir, auto_download=True)) as scheduler:
                with patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch_fail):
                    with pytest.raises(
                        TransientError, check=lambda e: e.code is ErrorCode.ASSET_DOWNLOAD, match="Network unreachable"
                    ):
                        await scheduler.new(VmSpec(language=Language.PYTHON, packages=("pandas==2.0.0",)))

        create_vm.assert_not_awaited()

    async def test_concurrent_fetch_same_language(self, images_dir: Path, tmp_path: Path) -> None:
        """Two concurrent requests with the same packages succeed without races.

        Concurrent ensure_base_image calls must be idempotent and CacheEntry.saving in
        get_or_create_snapshot must serialize snapshot creation so both requests
        boot on the same snapshot.
        """
        fresh_images, arch = self._make_fresh_images_dir(images_dir, tmp_path)

        # Pre-create base image so its digest (image_digest) is stable across concurrent calls
        base_image_path = fresh_images / f"python-3.14-base-{arch}.qcow2"
        await create_test_qcow2(base_image_path)

        cache_dir = tmp_path / "snapshot-cache"
        cache_dir.mkdir(mode=0o700)

        async def mock_fetch(language, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.05)
            # No-op: base image already exists (idempotent)

        async def mock_create_snapshot(language, packages, *, cache_key, **kwargs):  # type: ignore[no-untyped-def]
            snap = cache_dir / f"{cache_key}.qcow2"
            if not snap.exists():
                await create_test_qcow2(snap)
            return snap

        config = self._config(fresh_images, auto_download=True, disk_snapshot_cache_dir=cache_dir)
        spec = VmSpec(language=Language.PYTHON, packages=("pandas==2.0.0",))
        with (
            self._cold_boot_only() as create_vm,
            patch.object(DiskSnapshotManager, "_create_snapshot", side_effect=mock_create_snapshot),
        ):
            async with Scheduler(config) as scheduler:
                with patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch):
                    outcomes = await asyncio.gather(scheduler.new(spec), scheduler.new(spec), return_exceptions=True)

        assert [type(outcome) for outcome in outcomes] == [_ColdBootError, _ColdBootError]
        snapshot_1, snapshot_2 = (call.kwargs["snapshot_drive"] for call in create_vm.await_args_list)
        assert snapshot_1.exists()
        # Both should resolve to the same snapshot file (same cache key)
        assert snapshot_1 == snapshot_2

    async def test_fetch_ok_but_snapshot_creation_fails(self, images_dir: Path, tmp_path: Path) -> None:
        """ensure_base_image succeeds but _create_snapshot raises TransientError QEMU_CRASHED.

        Verifies that the error propagates to the caller and that the
        base image (written by ensure_base_image) is NOT cleaned up — only the
        snapshot file is cleaned up by _create_snapshot's own finally block.
        """
        fresh_images, arch = self._make_fresh_images_dir(images_dir, tmp_path)

        cache_dir = tmp_path / "snapshot-cache"
        cache_dir.mkdir(mode=0o700)
        base_image_path = fresh_images / f"python-3.14-base-{arch}.qcow2"

        async def mock_fetch(language, **kwargs):  # type: ignore[no-untyped-def]
            await create_test_qcow2(base_image_path)
            return base_image_path

        config = self._config(fresh_images, auto_download=True, disk_snapshot_cache_dir=cache_dir)
        build_error = TransientError(
            "VM boot failed", code=ErrorCode.QEMU_CRASHED, context=None, hint=None, instance_id=None
        )
        with (
            self._cold_boot_only() as create_vm,
            patch.object(DiskSnapshotManager, "_create_snapshot", new_callable=AsyncMock, side_effect=build_error),
        ):
            async with Scheduler(config) as scheduler:
                with patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch):
                    with pytest.raises(
                        TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED, match="VM boot failed"
                    ):
                        await scheduler.new(VmSpec(language=Language.PYTHON, packages=("pandas==2.0.0",)))

                # Base image should still exist (not cleaned up on downstream failure)
                assert base_image_path.exists(), "Base image must survive a downstream build failure"
        create_vm.assert_not_awaited()


class TestBaseImageFetchBeforeL1Restore:
    """ensure_base_image order on the path without packages.

    Companion to TestBaseImageFetchBeforeSnapshot (which covers the packages /
    L2 disk-snapshot path). On a fresh cache, ensure_assets() at startup only
    downloads kernel/initramfs/gvproxy — no base images. With no packages, the
    L1 memory-snapshot compute_cache_key() (Scheduler._launch_for) hashes the
    base image via GuestImages.base_image(). If the cold-boot ensure_base_image() ran
    AFTER that, GuestImages.base_image() would find no file and raise ConfigError
    ASSET_NOT_FOUND: the cold path fetches ahead of the L1 key.

    NOTE: these tests keep a REAL MemorySnapshotManager and its compute_cache_key (no
    _cold_boot_only, which stubs the key and would hide the bug entirely). Each request is a
    new() that ends at its cold boot (VmManager.create_vm raises _ColdBootError).
    """

    @pytest.fixture(autouse=True)
    def _empty_download_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty download cache: no base image of this user's cache stands in for the one
        the test expects ensure_base_image to bring (GuestImages falls back to it)."""
        monkeypatch.setenv("EXEC_SANDBOX_CACHE_DIR", str(tmp_path / "download-cache"))

    # Base image filename per language (mirrors the images.BASE_IMAGES mapping).
    _BASE_FILENAME: ClassVar[dict[Language, str]] = {
        Language.PYTHON: "python-3.14-base-{arch}.qcow2",
        Language.JAVASCRIPT: "node-1.4-base-{arch}.qcow2",
        Language.RAW: "raw-base-{arch}.qcow2",
    }

    @staticmethod
    def _make_fresh_images_dir(images_dir: Path, tmp_path: Path) -> tuple[Path, str]:
        """Isolated images dir with kernel/initramfs but NO base images."""
        fresh_images = tmp_path / "fresh-images"
        fresh_images.mkdir()
        arch = "aarch64" if platform.machine() == "arm64" else platform.machine()
        for name in images_dir.iterdir():
            if name.name.startswith(("vmlinuz", "vmlinux", "initramfs")):
                shutil.copy2(name, fresh_images / name.name)
        return fresh_images, arch

    @staticmethod
    def _fresh_config(fresh: Path, tmp_path: Path, *, auto_download: bool, warm_pool_size: int) -> SchedulerConfig:
        """SchedulerConfig on a fresh (base-less) images dir with isolated caches."""
        l1 = tmp_path / "l1"
        l2 = tmp_path / "l2"
        l1.mkdir(mode=0o700, exist_ok=True)
        l2.mkdir(mode=0o700, exist_ok=True)
        return SchedulerConfig(
            images_dir=fresh,
            auto_download_assets=auto_download,
            default_timeout_seconds=120,
            disk_snapshot_cache_dir=l2,
            memory_snapshot_cache_dir=l1,
            warm_pool_size=warm_pool_size,
        )

    @classmethod
    def _mock_fetch(cls, fresh: Path, arch: str, calls: list[Language]):  # type: ignore[no-untyped-def]
        """Side-effect for ensure_base_image: record the call, write a real qcow2.

        Simulates the downloader landing the base image in base_images_dir so
        the subsequent GuestImages.base_image() and image digest succeed.
        """

        async def _fetch(language: Language, **_kwargs) -> Path:  # type: ignore[no-untyped-def]
            calls.append(language)
            path = fresh / cls._BASE_FILENAME[language].format(arch=arch)
            await create_test_qcow2(path)
            return path

        return _fetch

    @pytest.mark.parametrize("language", [Language.PYTHON, Language.JAVASCRIPT])
    async def test_no_packages_fresh_cache_fetches_before_l1(
        self, images_dir: Path, tmp_path: Path, language: Language
    ) -> None:
        """No packages and a fresh cache: no ConfigError ASSET_NOT_FOUND.

        With the fetch after the L1 key, this raises ConfigError ASSET_NOT_FOUND from
        compute_cache_key -> GuestImages.base_image. It is the whole guard for this
        order.

        Parametrized over PYTHON/JAVASCRIPT (RAW omitted: the order does not depend on the
        language, and the mock writes whichever filename it is told).
        """
        fresh, arch = self._make_fresh_images_dir(images_dir, tmp_path)
        config = self._fresh_config(fresh, tmp_path, auto_download=True, warm_pool_size=0)
        fetch_calls: list[Language] = []
        with patch.object(VmManager, "create_vm", AsyncMock(side_effect=_ColdBootError)) as create_vm:
            async with Scheduler(config) as scheduler:
                with (
                    patch(
                        "exec_sandbox.scheduler.ensure_base_image",
                        side_effect=self._mock_fetch(fresh, arch, fetch_calls),
                    ),
                    pytest.raises(_ColdBootError),  # ConfigError ASSET_NOT_FOUND if the fetch came after the L1 key
                ):
                    await scheduler.new(VmSpec(language=language))

        create_vm.assert_awaited_once()
        assert fetch_calls == [language], "fetch must run exactly once, before the L1 cache-key hash"

    async def test_auto_download_disabled_missing_base_raises(self, images_dir: Path, tmp_path: Path) -> None:
        """auto_download_assets=False + no base -> ConfigError ASSET_NOT_FOUND; fetch NOT called.

        An offline or misconfigured host still gets a clear error for an image that is
        missing: the fetch does not hide it.
        """
        fresh, arch = self._make_fresh_images_dir(images_dir, tmp_path)
        config = self._fresh_config(fresh, tmp_path, auto_download=False, warm_pool_size=0)
        fetch_calls: list[Language] = []
        with (
            patch.object(VmManager, "create_vm", AsyncMock()) as create_vm,
            patch.object(MemorySnapshotManager, "schedule_background_save", AsyncMock()),
        ):
            async with Scheduler(config) as scheduler:
                with patch(
                    "exec_sandbox.scheduler.ensure_base_image",
                    side_effect=self._mock_fetch(fresh, arch, fetch_calls),
                ):
                    with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND):
                        await scheduler.new(VmSpec(language=Language.PYTHON))

        assert fetch_calls == [], "fetch must NOT be called when auto_download_assets=False"
        create_vm.assert_not_awaited()

    async def test_on_boot_log_debug_path_still_fetches(self, images_dir: Path, tmp_path: Path) -> None:
        """on_boot_log set (the --debug path) skips L1 but still fetches before cold boot.

        The debug path skips check_cache: only the fetch before the L1 key brings the base
        image there. check_cache is skipped, yet the base is fetched and create_vm proceeds.
        """
        fresh, arch = self._make_fresh_images_dir(images_dir, tmp_path)
        config = self._fresh_config(fresh, tmp_path, auto_download=True, warm_pool_size=0)
        fetch_calls: list[Language] = []
        with (
            patch.object(VmManager, "create_vm", AsyncMock(side_effect=_ColdBootError)) as create_vm,
            patch.object(
                MemorySnapshotManager, "check_cache", autospec=True, side_effect=MemorySnapshotManager.check_cache
            ) as check_spy,
        ):
            async with Scheduler(config) as scheduler:
                with (
                    patch(
                        "exec_sandbox.scheduler.ensure_base_image",
                        side_effect=self._mock_fetch(fresh, arch, fetch_calls),
                    ),
                    pytest.raises(_ColdBootError),
                ):
                    await scheduler.new(VmSpec(language=Language.PYTHON), on_boot_log=lambda _line: None)

        create_vm.assert_awaited_once()
        check_spy.assert_not_called()  # L1 skipped when on_boot_log set
        assert fetch_calls == [Language.PYTHON], "fetch must still run on the debug/no-L1 path"

    async def test_warm_pool_hit_skips_fetch(self, images_dir: Path, tmp_path: Path) -> None:
        """Warm-pool hit returns before the cold path -> no fetch, no create_vm.

        A fetch before the warm-pool lookup would make each warm hit wait for a download: the
        fast path downloads nothing. The pool is a stand-in that hands out a
        fake VM; its start fetches each pooled base image (TestWarmPoolStartupBaseFetch), so
        the count starts after the start.
        """
        fresh, _arch = self._make_fresh_images_dir(images_dir, tmp_path)
        config = self._fresh_config(fresh, tmp_path, auto_download=True, warm_pool_size=1)
        warm_vm = _fake_vm(f"vm-{uuid4().hex}")
        pool = MagicMock(start=AsyncMock(), stop=AsyncMock(), get_vm=AsyncMock(return_value=warm_vm))
        with (
            patch("exec_sandbox.scheduler.WarmVMPool", return_value=pool),
            patch("exec_sandbox.scheduler.ensure_base_image", new_callable=AsyncMock) as fetch,
            patch.object(VmManager, "create_vm", AsyncMock()) as create_vm,
            patch.object(VmManager, "destroy_vm", make_destroy_mock()),  # the fake VM's close
        ):
            async with Scheduler(config) as scheduler:
                fetch.reset_mock()
                async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
                    assert session.vm_id == warm_vm.vm_id
                    assert session.startup_timing.warm_pool_hit is True

                fetch.assert_not_awaited()  # a warm-pool hit does not download the base image
        create_vm.assert_not_awaited()


class TestWarmPoolStartupBaseFetch:
    """The base images are downloaded before the warm pool starts.

    WarmVMPool.start() pre-boots warm_pool_size VMs per language and hashes each base image
    (compute_cache_key -> GuestImages.base_image). On a fresh cache that
    would raise ConfigError ASSET_NOT_FOUND, since ensure_assets() fetched only the
    kernel/initramfs. Scheduler.__aenter__ must fetch every pooled base image
    first (when auto_download_assets is enabled). Same user experience as the
    per-request fetch in Scheduler._launch_for.
    """

    @staticmethod
    def _warm_config(fresh: Path, tmp_path: Path, *, auto_download: bool) -> SchedulerConfig:
        l1 = tmp_path / "l1"
        l2 = tmp_path / "l2"
        l1.mkdir(mode=0o700, exist_ok=True)
        l2.mkdir(mode=0o700, exist_ok=True)
        return SchedulerConfig(
            images_dir=fresh,
            auto_download_assets=auto_download,
            warm_pool_size=1,
            default_timeout_seconds=120,
            disk_snapshot_cache_dir=l2,
            memory_snapshot_cache_dir=l1,
        )

    @staticmethod
    def _mock_warm_pool(events: list[tuple[str, object]]) -> MagicMock:
        """WarmVMPool stand-in that records start() ordering, no real VM boot."""
        pool = MagicMock()

        async def _start(*_args: object, **_kwargs: object) -> None:
            events.append(("start", None))

        pool.start = AsyncMock(side_effect=_start)
        pool.stop = AsyncMock()
        return pool

    async def test_warm_pool_startup_fetches_all_bases_first(self, images_dir: Path, tmp_path: Path) -> None:
        """auto_download=True + warm_pool_size>0 + fresh cache: every base fetched before start()."""
        fresh, _arch = TestBaseImageFetchBeforeL1Restore._make_fresh_images_dir(images_dir, tmp_path)
        config = self._warm_config(fresh, tmp_path, auto_download=True)
        events: list[tuple[str, object]] = []

        async def mock_fetch(language: str, **_kwargs: object) -> None:
            events.append(("fetch", language))

        mock_pool = self._mock_warm_pool(events)

        with (
            patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch),
            patch("exec_sandbox.scheduler.WarmVMPool", return_value=mock_pool),
        ):
            async with Scheduler(config):
                pass

        fetched = {lang for tag, lang in events if tag == "fetch"}
        assert fetched == {language.value for language in Language}, "must fetch every pooled base image"
        start_idx = next(i for i, (tag, _) in enumerate(events) if tag == "start")
        assert all(i < start_idx for i, (tag, _) in enumerate(events) if tag == "fetch"), (
            "all base fetches must precede WarmVMPool.start()"
        )
        mock_pool.start.assert_awaited_once()

    async def test_warm_pool_startup_no_fetch_when_disabled(self, images_dir: Path, tmp_path: Path) -> None:
        """auto_download=False: no fetch attempted (offline images must be provided)."""
        fresh, _arch = TestBaseImageFetchBeforeL1Restore._make_fresh_images_dir(images_dir, tmp_path)
        config = self._warm_config(fresh, tmp_path, auto_download=False)
        events: list[tuple[str, object]] = []

        async def mock_fetch(language: str, **_kwargs: object) -> None:
            events.append(("fetch", language))

        mock_pool = self._mock_warm_pool(events)

        with (
            patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch),
            patch("exec_sandbox.scheduler.WarmVMPool", return_value=mock_pool),
        ):
            async with Scheduler(config):
                pass

        assert [tag for tag, _ in events if tag == "fetch"] == [], "no fetch when auto_download_assets=False"
        mock_pool.start.assert_awaited_once()

    async def test_a_failed_fetch_ends_the_other_fetches_before_the_start_raises(self, tmp_path: Path) -> None:
        """The raw base image cannot be downloaded while the others still download: the start
        raises that error as is, once the other downloads were cancelled and have ended (none
        writes into the download cache after the start returned); no pool starts."""
        scheduler = Scheduler(SchedulerConfig(warm_pool_size=1))
        error = OSError("disk full")
        ended: list[Language] = []

        async def fetch(language: Language, **_kwargs: object) -> None:
            if language is Language.RAW:  # the last one: the other downloads have started
                raise error
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0)  # removes its partial file
                ended.append(language)
                raise

        with _fake_start(scheduler, tmp_path) as fakes, patch("exec_sandbox.scheduler.ensure_base_image", fetch):
            with pytest.raises(OSError) as exc_info:
                await scheduler.__aenter__()

        assert exc_info.value is error
        assert ended == [Language.PYTHON, Language.JAVASCRIPT]
        fakes.warm.start.assert_not_awaited()
        await _assert_not_started(scheduler)


# ============================================================================
# Unit Tests - Shutdown Ordering
# ============================================================================


async def _assert_not_started(scheduler: Scheduler) -> None:
    """new() refuses at once: InvalidInputError SCHEDULER_STATE "not started"."""
    with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE, match="not started"):
        await scheduler.new(VmSpec(language=Language.PYTHON))


class TestSchedulerShutdownOrdering:
    """Tests for Scheduler.__aexit__ shutdown ordering.

    The critical invariant: WarmVMPool.stop() must run BEFORE
    MemorySnapshotManager.stop(). Background L1 save tasks create
    sacrificial VMs via vm_manager.create_vm(cache_leases=()), which blocks on
    admission.acquire(). If the warm pool holds all resource slots,
    stopping MemorySnapshotManager first deadlocks — stop() awaits
    the background task, the task waits for a slot, and slots are
    only freed when the pool drains.

    Each scheduler starts through its own __aenter__ (_fake_start) and stops through __aexit__.
    """

    @staticmethod
    async def _started(tmp_path: Path, *, warm_pool_size: int, instances: MagicMock | None) -> tuple[Scheduler, _Fakes]:
        """A Scheduler started by _fake_start, not stopped yet."""
        scheduler = Scheduler(SchedulerConfig(warm_pool_size=warm_pool_size))
        with _fake_start(scheduler, tmp_path, instances=instances) as fakes:
            await scheduler.__aenter__()
        return scheduler, fakes

    async def _make_scheduler(self, tmp_path: Path) -> tuple[Scheduler, _Fakes]:
        """A started Scheduler with a warm pool (1 VM per language) and no saved instances."""
        return await self._started(tmp_path, warm_pool_size=1, instances=None)

    # ------------------------------------------------------------------
    # Normal cases
    # ------------------------------------------------------------------

    async def test_shutdown_order_warm_pool_before_memory_snapshot(self, tmp_path: Path) -> None:
        """WarmVMPool.stop() is called before MemorySnapshotManager.stop()."""
        scheduler, fakes = await self._make_scheduler(tmp_path)
        call_order: list[str] = []

        async def record_warm_pool() -> None:
            call_order.append("warm_pool_stop")

        async def record_memory_snapshot() -> None:
            call_order.append("memory_snapshot_stop")

        fakes.warm.stop.side_effect = record_warm_pool
        fakes.memory.stop.side_effect = record_memory_snapshot

        await scheduler.__aexit__(None, None, None)

        assert call_order == ["warm_pool_stop", "memory_snapshot_stop"]

    async def test_shutdown_completes_cleanly(self, tmp_path: Path) -> None:
        """All managers stop, and the Scheduler refuses new requests."""
        scheduler, fakes = await self._make_scheduler(tmp_path)

        await scheduler.__aexit__(None, None, None)

        fakes.warm.stop.assert_awaited_once()
        fakes.memory.stop.assert_awaited_once()
        fakes.disk.stop.assert_awaited_once()
        fakes.manager.stop.assert_awaited_once()
        await _assert_not_started(scheduler)

    async def test_instance_saves_drain_before_vms_and_store_go(self, tmp_path: Path) -> None:
        """Saves in flight end (InstanceManager.stop) after the warm pool, the L1 saves and the
        L3 uploads stop, before the VM manager stops (it destroys the remaining VMs) and before
        the store clients close. The L3 uploads are awaited: a task left running is cancelled
        at loop close."""
        order: list[str] = []

        def step(name: str) -> AsyncMock:
            async def record(*_args: object) -> None:
                order.append(name)

            return AsyncMock(side_effect=record)

        scheduler, fakes = await self._started(
            tmp_path, warm_pool_size=1, instances=_store(stop=step("instance-saves"))
        )
        fakes.warm.stop = step("warm-pool")
        fakes.memory.stop = step("l1-saves")
        fakes.disk.stop = step("l3-uploads")
        fakes.manager.stop = step("vm-manager")
        fakes.store_closed.side_effect = step("store-closed")

        await scheduler.__aexit__(None, None, None)

        assert order == ["warm-pool", "l1-saves", "l3-uploads", "instance-saves", "vm-manager", "store-closed"]

    async def test_shutdown_closes_public_ingress_before_pool_drain(self, tmp_path: Path) -> None:
        scheduler, fakes = await self._make_scheduler(tmp_path)
        pool_stop_entered = asyncio.Event()
        release_pool_stop = asyncio.Event()

        async def blocked_pool_stop() -> None:
            pool_stop_entered.set()
            await release_pool_stop.wait()

        fakes.warm.stop.side_effect = blocked_pool_stop
        shutdown = asyncio.create_task(scheduler.__aexit__(None, None, None))
        await pool_stop_entered.wait()

        await _assert_not_started(scheduler)

        release_pool_stop.set()
        await asyncio.wait_for(shutdown, timeout=1)

    async def test_cancelled_shutdown_waiter_does_not_cancel_owned_shutdown(self, tmp_path: Path) -> None:
        """A cancelled context waiter cannot strand the shared shutdown sequence."""
        scheduler, fakes = await self._make_scheduler(tmp_path)
        pool_stop_entered = asyncio.Event()
        release_pool_stop = asyncio.Event()

        async def blocked_pool_stop() -> None:
            pool_stop_entered.set()
            await release_pool_stop.wait()

        fakes.warm.stop.side_effect = blocked_pool_stop
        first_waiter = asyncio.create_task(scheduler.__aexit__(None, None, None))
        await pool_stop_entered.wait()

        first_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter

        fakes.memory.stop.assert_not_awaited()
        fakes.manager.stop.assert_not_awaited()

        # The second waiter joins the same shutdown: it ends without a CancelledError, and
        # each manager stops once
        second_waiter = asyncio.create_task(scheduler.__aexit__(None, None, None))
        release_pool_stop.set()
        await asyncio.wait_for(second_waiter, timeout=1)

        fakes.warm.stop.assert_awaited_once()
        fakes.memory.stop.assert_awaited_once()
        fakes.manager.stop.assert_awaited_once()

    async def test_reentry_is_rejected_while_owned_shutdown_is_pending(self, tmp_path: Path) -> None:
        scheduler, fakes = await self._make_scheduler(tmp_path)
        pool_stop_entered = asyncio.Event()
        release_pool_stop = asyncio.Event()

        async def blocked_pool_stop() -> None:
            pool_stop_entered.set()
            await release_pool_stop.wait()

        fakes.warm.stop.side_effect = blocked_pool_stop
        shutdown = asyncio.create_task(scheduler.__aexit__(None, None, None))
        await pool_stop_entered.wait()

        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE, match="still shutting down"
        ):
            await scheduler.__aenter__()

        release_pool_stop.set()
        await shutdown

    async def test_shutdown_failure_leaves_scheduler_closed(self, tmp_path: Path) -> None:
        scheduler, fakes = await self._make_scheduler(tmp_path)
        fakes.manager.stop.side_effect = InternalError(
            "cleanup failed", code=ErrorCode.CLEANUP_UNCONFIRMED, context=None, hint=None, instance_id=None
        )

        with pytest.raises(
            InternalError, check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED, match="cleanup failed"
        ):
            await scheduler.__aexit__(None, None, None)

        await _assert_not_started(scheduler)

    async def test_shutdown_background_save_unblocks_after_pool_drain(self, tmp_path: Path) -> None:
        """Correct ordering unblocks a background save waiting on resources."""
        scheduler, fakes = await self._make_scheduler(tmp_path)
        gate = asyncio.Event()

        async def warm_pool_stop() -> None:
            gate.set()  # Simulates freeing resource slots

        async def memory_snapshot_stop() -> None:
            await asyncio.wait_for(gate.wait(), timeout=2)  # Blocked until gate set

        fakes.warm.stop.side_effect = warm_pool_stop
        fakes.memory.stop.side_effect = memory_snapshot_stop

        # Should complete without timeout because warm_pool stops first
        await asyncio.wait_for(scheduler.__aexit__(None, None, None), timeout=5)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    async def test_shutdown_no_warm_pool(self, tmp_path: Path) -> None:
        """Shutdown works when warm_pool_size is 0 (no pool was made)."""
        scheduler, fakes = await self._started(tmp_path, warm_pool_size=0, instances=None)

        await scheduler.__aexit__(None, None, None)

        fakes.warm.stop.assert_not_awaited()
        fakes.memory.stop.assert_awaited_once()
        fakes.manager.stop.assert_awaited_once()

    async def test_start_that_fails_before_the_l1_cache_stops_what_it_made(self, tmp_path: Path) -> None:
        """A start that fails at the L2 cache start (before the L1 cache and the warm pool are
        made) stops the VM manager, raises its own error and leaves the Scheduler closed."""
        scheduler = Scheduler(SchedulerConfig(warm_pool_size=1))
        with _fake_start(scheduler, tmp_path) as fakes:
            fakes.disk.start.side_effect = OSError("disk full")
            with pytest.raises(OSError, match="disk full"):
                await scheduler.__aenter__()

        fakes.manager.stop.assert_awaited_once()
        fakes.memory.stop.assert_not_awaited()
        fakes.warm.stop.assert_not_awaited()
        await _assert_not_started(scheduler)

    async def test_stop_of_a_scheduler_that_never_started(self) -> None:
        """__aexit__ without a start (no VM manager, cache or pool to stop) ends; new() still refuses."""
        scheduler = Scheduler()

        await scheduler.__aexit__(None, None, None)

        await _assert_not_started(scheduler)

    async def test_remaining_vms_go_with_the_vm_manager_stop(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """VMs of sessions not closed are named in a warning, then destroyed by the VM manager's
        stop (once, after it closes admission), never by a pass of the Scheduler before it."""
        scheduler, fakes = await self._make_scheduler(tmp_path)
        fakes.manager.get_active_vms.return_value = {"vm-1": MagicMock(), "vm-2": MagicMock()}

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.scheduler"):
            await scheduler.__aexit__(None, None, None)

        warnings = [(r.getMessage(), r.count) for r in caplog.records if r.name == "exec_sandbox.scheduler"]  # type: ignore[attr-defined]
        assert warnings == [("Destroying remaining VMs on shutdown", 2)]
        fakes.manager.destroy_vm.assert_not_awaited()
        fakes.manager.stop.assert_awaited_once()

    # ------------------------------------------------------------------
    # Error resilience
    # ------------------------------------------------------------------

    async def test_shutdown_warm_pool_error_continues(self, tmp_path: Path) -> None:
        """WarmVMPool.stop() error does not prevent other managers from stopping."""
        scheduler, fakes = await self._make_scheduler(tmp_path)
        fakes.warm.stop.side_effect = OSError("pool drain failed")

        await scheduler.__aexit__(None, None, None)

        fakes.memory.stop.assert_awaited_once()
        fakes.manager.stop.assert_awaited_once()

    async def test_shutdown_warm_pool_internal_error_continues_then_propagates(self, tmp_path: Path) -> None:
        scheduler, fakes = await self._make_scheduler(tmp_path)
        fakes.warm.stop.side_effect = InternalError(
            "pool cleanup unconfirmed", code=ErrorCode.CLEANUP_UNCONFIRMED, context=None, hint=None, instance_id=None
        )

        with pytest.raises(
            InternalError, check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED, match="pool cleanup unconfirmed"
        ):
            await scheduler.__aexit__(None, None, None)

        fakes.memory.stop.assert_awaited_once()
        fakes.manager.stop.assert_awaited_once()

    async def test_shutdown_memory_snapshot_error_continues(self, tmp_path: Path) -> None:
        """MemorySnapshotManager.stop() error does not prevent VM cleanup."""
        scheduler, fakes = await self._make_scheduler(tmp_path)
        fakes.memory.stop.side_effect = RuntimeError("snapshot save failed")

        await scheduler.__aexit__(None, None, None)

        fakes.manager.stop.assert_awaited_once()

    async def _with_instances(self, tmp_path: Path, failing: str) -> tuple[Scheduler, Exception, _Fakes, MagicMock]:
        """A started scheduler with saved instances on; the step `failing` raises. Returns
        (scheduler, the error, its stand-ins, the instance manager)."""
        error = InternalError(
            f"{failing} failed", code=ErrorCode.CLEANUP_UNCONFIRMED, context=None, hint=None, instance_id=None
        )
        instances = _store(stop=AsyncMock(side_effect=error if failing == "instance-saves" else None))
        scheduler, fakes = await self._started(tmp_path, warm_pool_size=1, instances=instances)
        if failing == "warm-pool":
            fakes.warm.stop.side_effect = error
        if failing == "store-close":
            fakes.store_closed.side_effect = error
        return scheduler, error, fakes, instances

    @pytest.mark.parametrize("failing", ["instance-saves", "store-close"])
    async def test_failed_instance_step_still_stops_the_vms_then_propagates(self, tmp_path: Path, failing: str) -> None:
        """A failed instance drain or store close: the VM cleanup still runs, then that error is raised."""
        scheduler, error, fakes, _instances = await self._with_instances(tmp_path, failing)

        with pytest.raises(InternalError) as exc:
            await scheduler.__aexit__(None, None, None)

        assert exc.value is error
        fakes.manager.stop.assert_awaited_once()

    @pytest.mark.parametrize("failing", ["warm-pool", "instance-saves"])
    async def test_failed_step_still_stops_instances_and_closes_the_store(self, tmp_path: Path, failing: str) -> None:
        """Always completes cleanup: after a failed step, instance saves still stop and the
        store clients still close."""
        scheduler, error, fakes, instances = await self._with_instances(tmp_path, failing)

        with pytest.raises(InternalError) as exc:
            await scheduler.__aexit__(None, None, None)

        assert exc.value is error
        instances.stop.assert_awaited_once()
        fakes.store_closed.assert_awaited_once()


# ============================================================================
# Integration Tests - Require QEMU + Images
# ============================================================================


class TestSchedulerIntegration:
    """Integration tests for Scheduler with real QEMU VMs.

    These tests require:
    - QEMU installed
    - VM images built (run 'make build-images')
    """

    async def test_scheduler_lifecycle(self, scheduler_config: SchedulerConfig) -> None:
        """Scheduler starts and stops cleanly: after the stop, new() refuses."""
        async with Scheduler(scheduler_config) as scheduler:
            await scheduler.wait_pool_ready()

        await _assert_not_started(scheduler)

    async def test_run_python_multiline(self, scheduler: Scheduler) -> None:
        """Run multiline Python code."""
        code = """
for i in range(3):
    print(f"line {i}")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert (result.exit_code, result.stdout) == (0, "line 0\nline 1\nline 2\n")

    async def test_run_python_exit_code(self, scheduler: Scheduler) -> None:
        """Python code with non-zero exit."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "import sys; sys.exit(42)")

        assert result.exit_code == 42

    @pytest.mark.parametrize(("code", "exit_code"), [("-1", 255), ("256", 0), ("10**10 + 7", 7)])
    async def test_run_python_exit_code_is_a_process_status(
        self, scheduler: Scheduler, code: str, exit_code: int
    ) -> None:
        """A sys.exit() code outside 0-255 is kept like a process exit status (sys.exit(-1) is 255):
        -1 stays the code timeout only."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), f"import sys; sys.exit({code})")

        assert result.exit_code == exit_code

    async def test_run_python_stderr(self, scheduler: Scheduler) -> None:
        """Python code that writes to stderr."""
        code = """
import sys
print("stdout message")
print("stderr message", file=sys.stderr)
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert (result.exit_code, result.stdout, result.stderr) == (0, "stdout message\n", "stderr message\n")

    async def test_run_with_env_vars(self, scheduler: Scheduler) -> None:
        """Run with custom environment variables."""
        code = """
import os
print(os.environ.get('MY_VAR', 'not set'))
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, env_vars={"MY_VAR": "hello"})

        assert (result.exit_code, result.stdout) == (0, "hello\n")

    async def test_run_with_streaming(self, scheduler: Scheduler) -> None:
        """Run with streaming output callbacks."""
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        code = """
import sys
print("out1")
print("err1", file=sys.stderr)
print("out2")
"""

        result = await run_once(
            scheduler,
            VmSpec(language=Language.PYTHON),
            code,
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert result.exit_code == 0
        assert ("".join(stdout_chunks), "".join(stderr_chunks)) == ("out1\nout2\n", "err1\n")

    async def test_run_timeout(self, scheduler_config: SchedulerConfig) -> None:
        """A code timeout is a result, never an error: the guest stops the code at the
        timeout (exit_code -1), and the startup timing is there. The host deadline comes 10 s
        later (EXECUTION_TIMEOUT_MARGIN_SECONDS), and a boot failure (VM_BOOT_TIMEOUT) is not a
        timeout of the code: both fail this test."""
        config = SchedulerConfig(
            images_dir=scheduler_config.images_dir,
            default_timeout_seconds=2,
        )

        code = """
import time
time.sleep(30)
"""

        async with Scheduler(config) as scheduler:
            result, startup, _ports = await run_timed(
                scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=1
            )
        assert result.exit_code == -1
        assert startup.total_ms > 0

    async def test_run_multiple_sequential(self, scheduler: Scheduler) -> None:
        """Multiple sequential runs work (VMs not reused)."""
        result1 = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('first')")
        result2 = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('second')")

        assert (result1.exit_code, result1.stdout) == (0, "first\n")
        assert (result2.exit_code, result2.stdout) == (0, "second\n")

    async def test_on_boot_log_receives_boot_output(self, scheduler: Scheduler) -> None:
        """on_boot_log gets the boot console lines of tiny-init (its init_total timing) and of the
        guest agent (its start), each line once and without its newline."""
        boot_lines: list[str] = []

        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), "print('hello')", on_boot_log=boot_lines.append
        )

        assert (result.exit_code, result.stdout) == (0, "hello\n")
        init_lines = [line for line in boot_lines if line.startswith("[INFO] [timing] init_total: ")]
        agent_lines = [line for line in boot_lines if line.startswith("[INFO] Guest agent starting ")]
        assert (len(init_lines), len(agent_lines)) == (1, 1), boot_lines
        assert not any(line.endswith("\n") for line in boot_lines), boot_lines

    @skip_unless_hwaccel
    async def test_on_boot_log_that_raises_is_dropped_and_boot_goes_on(self, scheduler: Scheduler) -> None:
        """A callback that raises gets no more lines; the VM boots and runs code."""
        lines: list[str] = []

        def broken(line: str) -> None:
            lines.append(line)
            raise RuntimeError("callback bug")

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('hello')", on_boot_log=broken)

        assert (result.exit_code, result.stdout.strip()) == (0, "hello")
        assert len(lines) == 1


# ============================================================================
# Parametrized Tests - All Image Types
# ============================================================================


# Test data for parametrized tests across all image types
SCHEDULER_IMAGE_TEST_CASES = [
    pytest.param(
        Language.PYTHON,
        "print('hello')",
        "hello",
        id="python",
    ),
    pytest.param(
        Language.JAVASCRIPT,
        "console.log('hello')",
        "hello",
        id="javascript",
    ),
    pytest.param(
        Language.RAW,
        "echo 'hello'",
        "hello",
        id="raw",
    ),
]


class TestSchedulerAllImages:
    """Parametrized tests to verify all image types work via Scheduler.

    Each image type (python, javascript, raw) must:
    1. Boot successfully (implicit via scheduler.run)
    2. Execute code and return correct output
    """

    @pytest.mark.parametrize("language,code,expected_output", SCHEDULER_IMAGE_TEST_CASES)
    async def test_scheduler_execute_all_images(
        self,
        scheduler: Scheduler,
        language: Language,
        code: str,
        expected_output: str,
    ) -> None:
        """Scheduler executes code for all image types."""
        result = await run_once(scheduler, VmSpec(language=language), code)

        assert result.exit_code == 0, f"Exit code {result.exit_code}, stderr: {result.stderr}"
        assert result.stdout == f"{expected_output}\n"


# ============================================================================
# Unit Tests - TimingBreakdown and warm_pool_hit
# ============================================================================


class TestTimingModels:
    """Execution timing (per exec) and startup timing (per VM) are separate models."""

    def test_timing_breakdown_is_execution_only(self) -> None:
        timing = TimingBreakdown(execute_ms=50, connect_ms=2)
        assert (timing.execute_ms, timing.connect_ms) == (50, 2)
        with pytest.raises(ValidationError):
            TimingBreakdown(execute_ms=50)  # type: ignore[call-arg]  # connect_ms is required (None when not measured)

    def test_startup_timing_requires_every_field(self) -> None:
        """Built only by the library: a missing phase is a bug, never a silent default."""
        with pytest.raises(ValidationError):
            StartupTiming(total_ms=1, admission_ms=0, setup_ms=1, boot_ms=0)  # type: ignore[call-arg]

    def test_execution_result_has_no_startup_fields(self) -> None:
        result = ExecutionResult(
            stdout="hello", stderr="", exit_code=0, timing=TimingBreakdown(execute_ms=1, connect_ms=None)
        )
        assert not {"warm_pool_hit", "l1_cache_hit", "exposed_ports"} & set(result.model_dump())
        assert (result.spawn_ms, result.process_ms) == (None, None)

    def test_execution_result_timing_required(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionResult(stdout="hello", stderr="", exit_code=0)  # type: ignore[call-arg]


# ============================================================================
# Integration Tests - Timing Behavior
# ============================================================================


@pytest.fixture
async def cold_scheduler(
    scheduler_config: SchedulerConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[Scheduler]:
    """Every new() cold-boots: no warm pool, an empty L1 folder of its own, and no L1 save (no second VM)."""
    config = scheduler_config.model_copy(update={"memory_snapshot_cache_dir": tmp_path / "l1"})
    async with Scheduler(config) as sched:
        memory = sched._memory_snapshot_manager  # pyright: ignore[reportPrivateUsage]  # the one seam
        assert memory is not None
        monkeypatch.setattr(memory, "schedule_background_save", AsyncMock(return_value=None))
        yield sched


@pytest.fixture
async def l1_scheduler(scheduler_config: SchedulerConfig, primed_l1_dir: Path) -> AsyncGenerator[Scheduler]:
    """Every new() of a default spec is an L1 restore: the run's primed L1 folder, no warm pool."""
    config = scheduler_config.model_copy(update={"memory_snapshot_cache_dir": primed_l1_dir})
    async with Scheduler(config) as sched:
        yield sched


async def _new_and_first_exec(scheduler: Scheduler) -> tuple[StartupTiming, ExecutionResult, float]:
    """new() of a default Python VM and its first exec(), with the wall time of both (ms).

    The wall time uses the event loop clock, the clock of total_ms and execute_ms.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    async with await scheduler.new(VmSpec(language=Language.PYTHON)) as vm:
        result = await vm.exec("print('hello')")
        wall_ms = (loop.time() - start) * 1000
        return vm.startup_timing, result, wall_ms


# total_ms and execute_ms are each rounded to 1 ms: their sum can pass the wall time by up to 1 ms.
_ROUNDING_MS = 2


class TestSchedulerTimingIntegration:
    """Integration tests for scheduler timing behavior.

    These tests verify that:
    1. timing values are consistent: new() and exec() are two separate spans
    2. warm_pool_hit and l1_cache_hit tell how the VM started

    A test whose result depends on how the VM starts chooses it (cold_scheduler,
    l1_scheduler): the default L1 folder is shared, so the `scheduler` fixture
    restores from L1 on a host that ran the tests before and cold-boots on a new
    CI runner.
    """

    async def test_startup_phases_fit_in_the_total(self, scheduler: Scheduler) -> None:
        """Phases do not overlap: their sum stays within total_ms (each is rounded to 1 ms)."""
        _result, startup, _ports = await run_timed(scheduler, VmSpec(language=Language.PYTHON), "print('hello')")

        assert startup.total_ms > 0
        assert startup.admission_ms + startup.setup_ms + startup.boot_ms <= startup.total_ms + 2

    async def test_cold_boot_new_and_first_exec_fit_in_wall_time(self, cold_scheduler: Scheduler) -> None:
        """Cold boot: new() and the first exec() are two separate spans inside the wall time.

        new() returns before the REPL starts, so the first exec() starts the
        interpreter: its execute_ms can be longer than startup.total_ms.
        """
        startup, result, wall_ms = await _new_and_first_exec(cold_scheduler)

        assert (startup.warm_pool_hit, startup.l1_cache_hit) == (False, False), startup
        assert result.exit_code == 0
        assert startup.total_ms + result.timing.execute_ms <= wall_ms + _ROUNDING_MS, (startup, result.timing, wall_ms)

    @skip_unless_hwaccel  # as every L1 test: the primed folder needs a warm-pool start
    async def test_l1_restore_new_and_first_exec_fit_in_wall_time(self, l1_scheduler: Scheduler) -> None:
        """L1 restore: new() and the first exec() are two separate spans inside the wall time."""
        startup, result, wall_ms = await _new_and_first_exec(l1_scheduler)

        assert startup.l1_cache_hit, startup
        assert startup.boot_ms == 0
        assert result.exit_code == 0
        assert startup.total_ms + result.timing.execute_ms <= wall_ms + _ROUNDING_MS, (startup, result.timing, wall_ms)

    async def test_cold_boot_has_nonzero_boot_time(self, cold_scheduler: Scheduler) -> None:
        """A cold boot (no warm pool, no L1 restore) spends real time booting (100-500 ms typical)."""
        _result, startup, _ports = await run_timed(cold_scheduler, VmSpec(language=Language.PYTHON), "print('hello')")

        assert (startup.warm_pool_hit, startup.l1_cache_hit) == (False, False), startup
        assert startup.boot_ms > 0

    async def test_timing_with_exception(self, scheduler: Scheduler) -> None:
        """Code that raises: exit code 1 and the traceback on stderr; the guest still reports
        its timing (spawn_ms, process_ms), and the startup timing is there."""
        result, startup, _ports = await run_timed(
            scheduler, VmSpec(language=Language.PYTHON), "raise ValueError('test error')"
        )

        assert result.exit_code == 1
        assert "ValueError: test error" in result.stderr
        assert (result.spawn_ms is None, result.process_ms is None) == (False, False)
        assert startup.total_ms > 0

    async def test_large_output_comes_whole_within_the_exec_time(self, scheduler: Scheduler) -> None:
        """100 KB of output comes whole; execute_ms stays within the wall time of that exec()."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as vm:
            result, wall_ms = await _timed_exec(vm, "print('x' * 100000)")

        assert (result.exit_code, result.stdout) == (0, "x" * 100000 + "\n")
        assert result.timing.execute_ms <= wall_ms + _ROUNDING_MS, (result.timing, wall_ms)

    async def test_timing_javascript(self, scheduler: Scheduler) -> None:
        """JavaScript: the output, the startup timing, and the guest's timing of the REPL
        (spawn_ms, process_ms), each within its execution_time_ms."""
        result, startup, _ports = await run_timed(
            scheduler, VmSpec(language=Language.JAVASCRIPT), "console.log('hello')"
        )

        assert (result.exit_code, result.stdout) == (0, "hello\n")
        assert startup.total_ms > 0
        assert result.execution_time_ms is not None
        assert result.spawn_ms is not None
        assert result.process_ms is not None
        assert max(result.spawn_ms, result.process_ms) <= result.execution_time_ms, result

    async def test_granular_timing_populated(self, scheduler: Scheduler) -> None:
        """connect_ms (host) within execute_ms; the guest's spawn_ms and process_ms within its
        execution_time_ms, itself within the host's execute_ms (each is rounded to 1 ms)."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('hello')")

        assert result.exit_code == 0
        assert result.timing.connect_ms is not None
        assert result.timing.connect_ms <= result.timing.execute_ms, result.timing
        assert result.execution_time_ms is not None
        assert result.spawn_ms is not None
        assert result.process_ms is not None
        assert max(result.spawn_ms, result.process_ms) <= result.execution_time_ms, result
        assert result.execution_time_ms <= result.timing.execute_ms + _ROUNDING_MS, result

    async def test_granular_timing_consistency(self, scheduler: Scheduler) -> None:
        """A 100 ms sleep is in the guest's process_ms (the guest clock, cut to whole ms)."""
        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), "import time; time.sleep(0.1); print('done')"
        )

        assert (result.exit_code, result.stdout) == (0, "done\n")
        assert result.process_ms is not None
        assert result.process_ms >= 100


async def _timed_exec(session: Session, code: str) -> tuple[ExecutionResult, float]:
    """session.exec(code), with its wall time in ms on the event loop clock (the clock of execute_ms)."""
    loop = asyncio.get_running_loop()
    start = loop.time()
    result = await session.exec(code)
    return result, (loop.time() - start) * 1000


class TestSchedulerTimingEdgeCases:
    """Edge case tests for timing behavior."""

    async def test_timing_empty_output(self, scheduler: Scheduler) -> None:
        """Code with no output: empty stdout and stderr; execute_ms within the exec() wall time."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as vm:
            result, wall_ms = await _timed_exec(vm, "x = 1")

        assert (result.exit_code, result.stdout, result.stderr) == (0, "", "")
        assert result.timing.execute_ms <= wall_ms + _ROUNDING_MS, (result.timing, wall_ms)

    async def test_each_exec_has_its_own_timing(self, scheduler: Scheduler) -> None:
        """Two exec() calls in one session: the first sleeps 500 ms and takes at least that; the
        second (no sleep) takes less, so its time does not hold the first's."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as vm:
            slow = await vm.exec("import time; time.sleep(0.5); print('slow')")
            fast = await vm.exec("print('fast')")

        assert (slow.stdout, fast.stdout) == ("slow\n", "fast\n")
        assert slow.timing.execute_ms >= 500 - _ROUNDING_MS, slow.timing
        assert fast.timing.execute_ms < 500, fast.timing


# ============================================================================
# Integration Tests - Warm Pool Timing
# ============================================================================


class TestSchedulerWarmPoolTiming:
    """Integration tests for warm pool timing behavior.

    These tests verify that warm pool hits have:
    1. warm_pool_hit=True
    2. boot_ms=0 (boot happened at startup, not request time)
    3. execute_ms is the exec() call; startup.total_ms is only the pool checkout

    Note: setup_ms is not asserted because pool checkout time varies widely
    under TCG emulation with parallel test workers.
    """

    @pytest.fixture
    async def warm_pool_scheduler(self, scheduler_config: SchedulerConfig):
        """Scheduler with warm pool enabled."""
        config = SchedulerConfig(
            images_dir=scheduler_config.images_dir,
            warm_pool_size=2,  # Enable warm pool with 2 VMs per language
            auto_download_assets=False,
        )
        async with Scheduler(config) as sched:
            yield sched

    async def test_warm_pool_timing_has_execute_time(self, warm_pool_scheduler: Scheduler) -> None:
        """Warm pool hit: execute_ms is the real exec() time; the startup has no admission wait and no boot."""
        result, startup, _ports = await run_timed(
            warm_pool_scheduler, VmSpec(language=Language.PYTHON), "import time; time.sleep(0.05); print('done')"
        )

        assert result.exit_code == 0
        assert startup.warm_pool_hit is True
        # Execute time should reflect actual code execution (at least 50ms for sleep)
        assert result.timing.execute_ms >= 40  # Allow some variance
        # The startup has no admission wait and no boot: both happened when the pool filled
        assert (startup.admission_ms, startup.boot_ms) == (0, 0)

    @skip_unless_fast_virt
    async def test_warm_pool_startup_is_near_zero(self, warm_pool_scheduler: Scheduler) -> None:
        """A warm pool hit starts in a pool checkout: new() takes well under 100 ms.

        100ms tolerance: slow or nested virtualization cannot meet it.
        """
        _result, startup, _ports = await run_timed(
            warm_pool_scheduler, VmSpec(language=Language.PYTHON), "print('hello')"
        )

        assert startup.warm_pool_hit is True
        assert startup.total_ms <= 100

    async def test_warm_pool_exhaustion_falls_back_to_cold(self, scheduler_config: SchedulerConfig) -> None:
        """When warm pool is exhausted, falls back to cold boot with full timing."""
        # Create scheduler with very small pool (1 VM)
        config = SchedulerConfig(
            images_dir=scheduler_config.images_dir,
            warm_pool_size=1,
            auto_download_assets=False,
        )

        async with Scheduler(config) as sched:
            # Run multiple concurrent executions to exhaust pool
            # First one gets warm VM, subsequent ones may cold boot
            runs = await asyncio.gather(
                run_timed(sched, VmSpec(language=Language.PYTHON), "import time; time.sleep(0.2); print('1')"),
                run_timed(sched, VmSpec(language=Language.PYTHON), "import time; time.sleep(0.2); print('2')"),
                run_timed(sched, VmSpec(language=Language.PYTHON), "import time; time.sleep(0.2); print('3')"),
            )

            # All should succeed, each with its own output
            assert [(run.result.exit_code, run.result.stdout) for run in runs] == [(0, "1\n"), (0, "2\n"), (0, "3\n")]

            # At least one should be warm, at least one should be cold (pool exhausted)
            warm_hits = sum(1 for run in runs if run.startup.warm_pool_hit)
            cold_boots = sum(1 for run in runs if not run.startup.warm_pool_hit)

            # With pool size 1, we expect 1 warm hit and 2 cold boots
            # (assuming no replenishment completes during the test)
            assert warm_hits >= 1, "Expected at least one warm pool hit"
            assert cold_boots >= 1, "Expected at least one cold boot (pool exhaustion)"

            # Cold boots should have non-zero setup/boot times
            for run in runs:
                if not run.startup.warm_pool_hit:
                    # Cold boot - should have real timing values
                    # At least one of setup_ms or boot_ms should be > 0
                    assert run.startup.setup_ms > 0 or run.startup.boot_ms > 0

    async def test_warm_pool_javascript(self, warm_pool_scheduler: Scheduler) -> None:
        """Warm pool works for JavaScript with correct timing."""
        result, startup, _ports = await run_timed(
            warm_pool_scheduler, VmSpec(language=Language.JAVASCRIPT), "console.log('hello')"
        )

        assert (result.exit_code, result.stdout) == (0, "hello\n")
        assert startup.warm_pool_hit is True
        # No admission wait and no boot: both happened when the pool filled
        assert (startup.admission_ms, startup.boot_ms) == (0, 0)


# ============================================================================
# Package Installation Integration Tests
# ============================================================================


PACKAGE_INSTALL_TEST_CASES = [
    # Python - pure Python packages
    pytest.param(
        Language.PYTHON,
        ["requests==2.31.0"],
        'import requests; print(f"requests={requests.__version__}")',
        ["requests=2.31.0"],
        None,
        id="python-requests",
    ),
    pytest.param(
        Language.PYTHON,
        ["requests==2.31.0", "flask==3.0.0", "httpx==0.27.0"],
        """
import requests
import flask
import httpx
print(f"requests={requests.__version__}")
print(f"flask={flask.__version__}")
print(f"httpx={httpx.__version__}")
""",
        ["requests=2.31.0", "flask=3.0.0", "httpx=0.27.0"],
        None,
        id="python-multi",
    ),
    # Python - C extension packages (prebuilt musllinux wheels, no source builds)
    pytest.param(
        Language.PYTHON,
        ["numpy==2.4.2"],
        """
import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.linalg.inv(a)
det = np.linalg.det(a)
print(f"numpy={np.__version__}")
print(f"det={det:.1f}")
print(f"inv_sum={b.sum():.1f}")
""",
        ["numpy=2.4.2", "det=-2.0", "inv_sum=0.0"],
        None,
        id="python-numpy",
        marks=skip_unless_hwaccel,  # pip install routinely exceeds 940s under TCG
    ),
    pytest.param(
        Language.PYTHON,
        ["pandas==3.0.1"],
        """
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
filtered = df[df["a"] > 1]
print(f"pandas={pd.__version__}")
print(f"shape={filtered.shape}")
print(f"mean={df['b'].mean():.1f}")
""",
        ["pandas=3.0.1", "shape=(2, 2)", "mean=5.0"],
        None,
        id="python-pandas",
        marks=skip_unless_hwaccel,  # pip install routinely exceeds 940s under TCG
    ),
    pytest.param(
        Language.PYTHON,
        ["pillow==12.1.1"],
        """
from PIL import Image
import PIL
img = Image.new("RGB", (100, 50), color=(255, 0, 0))
img = img.resize((50, 25))
print(f"pillow={PIL.__version__}")
print(f"size={img.size}")
print(f"pixel={img.getpixel((0, 0))}")
""",
        ["pillow=12.1.1", "size=(50, 25)", "pixel=(255, 0, 0)"],
        None,
        id="python-pillow",
        marks=skip_unless_hwaccel,  # pip install routinely exceeds 940s under TCG
    ),
    # JavaScript - pure JS packages
    pytest.param(
        Language.JAVASCRIPT,
        ["lodash@4.17.21"],
        """
const _ = require("lodash");
const arr = [3, 1, 4, 1, 5, 9];
const sorted = _.sortBy(arr);
const chunked = _.chunk(arr, 2);
console.log("lodash=" + _.VERSION);
console.log("sorted=" + JSON.stringify(sorted));
console.log("chunks=" + chunked.length);
""",
        ["lodash=4.17.21", "sorted=[1,1,3,4,5,9]", "chunks=3"],
        None,
        id="js-lodash",
    ),
    pytest.param(
        Language.JAVASCRIPT,
        ["lodash@4.17.21", "moment@2.30.1"],
        """
const _ = require("lodash");
const moment = require("moment");
const grouped = _.groupBy(["one", "two", "three"], "length");
const date = moment("2024-01-15").format("MMMM D");
console.log("lodash=" + _.VERSION);
console.log("moment=" + moment.version);
console.log("groups=" + Object.keys(grouped).length);
console.log("date=" + date);
""",
        ["lodash=4.17.21", "moment=2.30.1", "groups=2", "date=January 15"],
        None,
        id="js-multi",
    ),
    # JavaScript - native binary packages
    pytest.param(
        Language.JAVASCRIPT,
        ["esbuild@0.25.4"],
        """
const esbuild = require("esbuild");
const result = esbuild.transformSync("const x: number = 1", { loader: "ts" });
console.log("esbuild=" + esbuild.version);
console.log("output=" + result.code.trim());
""",
        ["esbuild=0.25.4", "output=const x = 1;"],
        None,
        id="js-esbuild",
        marks=skip_unless_hwaccel,  # npm install with native binary exceeds 940s under TCG
    ),
]


@pytest.mark.slow
class TestPackageInstallation:
    """Integration tests for package installation with real QEMU VMs.

    Slow under TCG: package install correctness tests — snapshot VM creation
    is functional under TCG but boot overhead makes them too slow for the
    default suite.

    These tests verify that packages are correctly installed and persisted
    in snapshots. They catch bugs like:
    - QEMU exit code detection issues (macOS HVF vs Linux KVM)
    - Filesystem sync issues with cache=unsafe
    - Snapshot corruption during package install
    """

    @pytest.mark.parametrize("language,packages,code,expected_outputs,memory_mb", PACKAGE_INSTALL_TEST_CASES)
    async def test_package_install_and_import(
        self,
        *,
        scheduler: Scheduler,
        language: Language,
        packages: list[str],
        code: str,
        expected_outputs: list[str],
        memory_mb: int | None,
    ) -> None:
        """Packages are installed and importable."""
        sizes = {} if memory_mb is None else {"memory_mb": memory_mb}  # None: the VmSpec default
        result = await run_once(scheduler, VmSpec(language=language, packages=packages, **sizes), code)

        assert result.exit_code == 0, f"Failed: {result.stderr}"
        for expected in expected_outputs:
            assert expected in result.stdout

    async def test_python_snapshot_cache_hit(self, scheduler: Scheduler) -> None:
        """Second run with same packages uses cached snapshot."""
        packages = ["requests==2.31.0"]

        # First run: creates snapshot
        r1 = await run_once(
            scheduler, VmSpec(language=Language.PYTHON, packages=packages), 'import requests; print("first")'
        )
        assert r1.exit_code == 0, f"First run failed: {r1.stderr}"

        # Second run: should use cached snapshot (much faster)
        r2 = await run_once(
            scheduler, VmSpec(language=Language.PYTHON, packages=packages), 'import requests; print("second")'
        )
        assert r2.exit_code == 0, f"Second run failed: {r2.stderr}"
        assert "second" in r2.stdout

    async def test_python_packages_work_without_network(self, scheduler: Scheduler) -> None:
        """Packages from cached snapshot work without network access.

        This is a critical test that proves:
        1. Snapshot correctly persists installed packages
        2. Packages don't require network at runtime
        3. The filesystem sync of the build VM wrote them (cache=unsafe)
        """
        packages = ["requests==2.31.0"]

        # First run: creates snapshot (needs network for pip install)
        r1 = await run_once(
            scheduler, VmSpec(language=Language.PYTHON, packages=packages), 'import requests; print("setup")'
        )
        assert r1.exit_code == 0, f"Setup run failed: {r1.stderr}"

        # Second run: uses cached snapshot WITHOUT network
        r2 = await run_once(
            scheduler,
            VmSpec(
                language=Language.PYTHON,
                packages=packages,
                allow_network=False,  # No internet!
            ),
            'import requests; print(f"offline: {requests.__version__}")',
        )
        assert r2.exit_code == 0, f"Offline run failed: {r2.stderr}"
        assert "offline: 2.31.0" in r2.stdout

    async def test_javascript_packages_work_without_network(self, scheduler: Scheduler) -> None:
        """JavaScript packages from cached snapshot work without network."""
        packages = ["lodash@4.17.21"]

        # First run: creates snapshot
        r1 = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT, packages=packages),
            'const lodash = require("lodash"); console.log("setup")',
        )
        assert r1.exit_code == 0, f"Setup run failed: {r1.stderr}"

        # Second run: uses cached snapshot WITHOUT network
        r2 = await run_once(
            scheduler,
            VmSpec(
                language=Language.JAVASCRIPT,
                packages=packages,
                allow_network=False,  # No internet!
            ),
            'const lodash = require("lodash"); console.log("offline: " + lodash.VERSION)',
        )
        assert r2.exit_code == 0, f"Offline run failed: {r2.stderr}"
        assert "offline: 4.17.21" in r2.stdout

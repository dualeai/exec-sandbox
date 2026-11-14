"""Tests for Scheduler.

Unit tests: Test validation, config, error handling (no QEMU needed).
Integration tests: Test real VM execution (requires QEMU + images).
"""

import asyncio
import contextlib
import json
import platform
import shutil
import threading
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pydantic import ValidationError

from exec_sandbox import package_validator
from exec_sandbox.config import SchedulerConfig
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
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_timing import VmTiming
from exec_sandbox.vm_types import VmState
from tests.conftest import (
    create_test_qcow2,
    l1_hit,
    make_destroy_mock,
    random_test_id,
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
        monkeypatch.delenv("EXEC_SANDBOX_S3_BUCKET", raising=False)
        config = SchedulerConfig()
        settings = self._settings(config, tmp_path)
        assert settings.s3_bucket is None
        assert settings.s3_snapshot_prefix == config.s3_snapshot_prefix
        assert settings.disk_snapshot_cache_dir == config.disk_snapshot_cache_dir


class TestSchedulerContextManager:
    """Tests for Scheduler context manager."""

    async def test_double_start_raises(self, tmp_path: Path) -> None:
        """A second start of a started Scheduler raises InvalidInputError SCHEDULER_STATE."""
        scheduler = Scheduler(SchedulerConfig(images_dir=tmp_path))

        with _fake_start(scheduler, tmp_path):
            async with scheduler:
                with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE) as exc_info:
                    await scheduler.__aenter__()

        assert "already started" in str(exc_info.value)

    async def test_run_without_start_raises(self) -> None:
        """Calling run() without starting raises InvalidInputError SCHEDULER_STATE."""
        scheduler = Scheduler()

        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE) as exc_info:
            await run_once(scheduler, VmSpec(language=Language.PYTHON), "print(1)")

        assert "not started" in str(exc_info.value)

    async def test_cancelled_partial_startup_waits_for_owned_rollback(self, tmp_path: Path) -> None:
        """Repeated cancellation cannot abandon a manager started by __aenter__.

        The rollback task is private: a race window (a second cancel while the rollback runs)
        that no public call can open on purpose."""
        scheduler = Scheduler(SchedulerConfig(images_dir=tmp_path, warm_pool_size=1))
        manager = MagicMock()
        manager.start = AsyncMock()
        manager.get_active_vms.return_value = {}
        manager_stop_entered = asyncio.Event()
        release_manager_stop = asyncio.Event()

        async def blocked_manager_stop() -> None:
            manager_stop_entered.set()
            await release_manager_stop.wait()

        manager.stop = AsyncMock(side_effect=blocked_manager_stop)
        memory = MagicMock()
        memory.stop = AsyncMock()
        warm = MagicMock()
        warm_start_entered = asyncio.Event()

        async def blocked_warm_start() -> None:
            warm_start_entered.set()
            await asyncio.Event().wait()

        warm.start = AsyncMock(side_effect=blocked_warm_start)
        warm.stop = AsyncMock()

        with (
            patch("exec_sandbox.scheduler.ensure_assets", new_callable=AsyncMock, return_value=tmp_path),
            patch("exec_sandbox.scheduler.ensure_base_image", new_callable=AsyncMock),  # the pool's base images
            # instances off
            patch("exec_sandbox.scheduler.settings_from_config", return_value=MagicMock(instance_store=None)),
            patch("exec_sandbox.scheduler.VmManager", return_value=manager),
            patch("exec_sandbox.scheduler.DiskSnapshotManager", return_value=MagicMock(start=AsyncMock())),
            patch("exec_sandbox.scheduler.MemorySnapshotManager", return_value=memory),
            patch("exec_sandbox.scheduler.WarmVMPool", return_value=warm),
        ):
            startup = asyncio.create_task(scheduler.__aenter__())
            await warm_start_entered.wait()
            startup.cancel()
            await manager_stop_entered.wait()
            startup.cancel()
            await asyncio.sleep(0)
            assert not startup.done()

            release_manager_stop.set()
            with pytest.raises(asyncio.CancelledError):
                await startup

        warm.stop.assert_awaited_once()
        memory.stop.assert_awaited_once()
        manager.stop.assert_awaited_once()
        assert scheduler._started is False
        assert scheduler._shutdown_task is not None
        assert scheduler._shutdown_task.done()
        assert not scheduler._shutdown_task.cancelled()


@contextlib.contextmanager
def _fake_start(scheduler: Scheduler, tmp_path: Path) -> Iterator[tuple[MagicMock, MagicMock, MagicMock]]:
    """A start of `scheduler` with no host work: no asset download, saved instances off, and a
    fake VM manager, L2 cache and L1 cache (yielded in that order)."""
    manager = MagicMock(start=AsyncMock(), stop=AsyncMock())
    manager.get_active_vms.return_value = {}
    disk = MagicMock(start=AsyncMock(), stop=AsyncMock())
    memory = MagicMock(stop=AsyncMock())
    with (
        patch("exec_sandbox.scheduler.ensure_assets", new_callable=AsyncMock, return_value=tmp_path),
        patch("exec_sandbox.scheduler.settings_from_config", return_value=MagicMock(instance_store=None)),
        patch("exec_sandbox.scheduler.VmManager", return_value=manager),
        patch("exec_sandbox.scheduler.DiskSnapshotManager", return_value=disk),
        patch("exec_sandbox.scheduler.MemorySnapshotManager", return_value=memory),
    ):
        yield manager, disk, memory


async def test_start_lets_admission_evict_both_caches(tmp_path: Path) -> None:
    """Gate 4 evicts L2 and L1 entries before a VM waits for home disk space."""
    scheduler = Scheduler(SchedulerConfig(images_dir=tmp_path))
    with _fake_start(scheduler, tmp_path) as (manager, disk, memory):
        async with scheduler:
            assert manager.add_cache.call_args_list == [call(disk), call(memory)]


class TestNewInstanceId:
    """new(instance_id=...) checks the id, then that saved instances are on, before any VM
    starts. `_vm_manager` is None: a new() that reached the VM path would raise InternalError
    ("resources not initialized") instead of the error each test expects."""

    @staticmethod
    def _started(*, instances: MagicMock | None) -> Scheduler:
        scheduler = Scheduler()
        scheduler._started = True
        scheduler._instances = instances
        return scheduler

    @pytest.mark.parametrize("bad", ["My VM", "a/b", "", "a" * 65])
    async def test_invalid_id_fails_before_any_vm(self, bad: str) -> None:
        scheduler = self._started(instances=MagicMock())
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.INSTANCE_INVALID_ID):
            await scheduler.new(VmSpec(language=Language.PYTHON), instance_id=bad)

    async def test_id_without_a_store_fails_before_any_vm(self) -> None:
        scheduler = self._started(instances=None)
        with pytest.raises(
            ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG, match="none is configured"
        ) as exc:
            await scheduler.new(VmSpec(language=Language.PYTHON), instance_id="my-vm")
        assert (exc.value.action, exc.value.instance_id) == (ErrorAction.FIX_CONFIG, "my-vm")

    async def test_valid_id_with_a_store_goes_on_to_the_vm_path(self) -> None:
        """Control: with a store and a valid id, the same scheduler goes on to start a VM."""
        scheduler = self._started(instances=MagicMock())
        with pytest.raises(
            InternalError, check=lambda e: e.code is ErrorCode.INTERNAL, match="resources not initialized"
        ):
            await scheduler.new(VmSpec(language=Language.PYTHON), instance_id="my-vm")


class TestIdleTimeoutArgument:
    """new() and reuse(): idle_timeout_seconds has the range of
    SchedulerConfig.session_idle_timeout_seconds (10 to 3600); None takes the config's value.
    Out of range (0 included) is InvalidInputError ARGUMENT_INVALID before any VM or store call."""

    @pytest.mark.parametrize(("argument", "used"), [(None, 77), (10, 10), (3600, 3600)])
    async def test_new_session_gets_the_timeout(self, argument: int | None, used: int) -> None:
        """The session gets the argument, else the config's value. The value is read from the
        session: the public effect (a close after that many idle seconds, 10 s or more) is
        too slow for a unit test; tests/test_session.py tests that the timer fires."""
        scheduler, _, prepared = TestOneShotTeardownConfirmation._mock_run(VmState.DESTROYED)
        scheduler.config = SchedulerConfig(session_idle_timeout_seconds=77)
        with patch.object(scheduler, "_prepare_vm", new_callable=AsyncMock, return_value=prepared):
            session = await scheduler.new(VmSpec(language=Language.PYTHON), idle_timeout_seconds=argument)
            await session.close()
        assert session._idle_timeout_seconds == used

    @pytest.mark.parametrize("argument", [0, 9, 3601, -1])
    async def test_new_out_of_range(self, argument: int) -> None:
        scheduler, _, prepared = TestOneShotTeardownConfirmation._mock_run(VmState.DESTROYED)
        with patch.object(scheduler, "_prepare_vm", new_callable=AsyncMock, return_value=prepared) as prepare:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID) as exc:
                await scheduler.new(VmSpec(language=Language.PYTHON), idle_timeout_seconds=argument)
        assert str(exc.value) == f"idle_timeout_seconds must be between 10 and 3600, got {argument}"
        prepare.assert_not_awaited()

    @pytest.mark.parametrize("argument", [0, 3601])
    async def test_reuse_out_of_range_sends_no_request(self, argument: int) -> None:
        scheduler = Scheduler()
        scheduler._started = True
        scheduler._instances = MagicMock(load=AsyncMock())
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID):
            await scheduler.reuse("my-vm", idle_timeout_seconds=argument)
        scheduler._instances.load.assert_not_awaited()

    async def test_reuse_in_range_goes_on(self) -> None:
        """Control: the same scheduler with a valid timeout goes on (to its missing VM manager)."""
        scheduler = Scheduler()
        scheduler._started = True
        scheduler._instances = MagicMock(load=AsyncMock())
        with pytest.raises(InternalError, match="resources not initialized"):
            await scheduler.reuse("my-vm", idle_timeout_seconds=10)


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
    def _started(restore_error: Exception) -> tuple[Scheduler, MagicMock, SimpleNamespace]:
        scheduler = Scheduler()
        scheduler._started = True
        state = SimpleNamespace(
            body=SimpleNamespace(spec=VmSpec(language=Language.PYTHON), host=SimpleNamespace(qemu_version="11.0.2")),
            header=SimpleNamespace(generation=3),
            vmstate=Path("/saved/c.vmstate"),
            base_image=Path("/saved/base.img"),
            packages_image=None,
            folder=Path("/saved"),
        )
        instances = MagicMock()
        instances.load = AsyncMock(return_value=state)
        instances.release = AsyncMock()
        scheduler._instances = instances
        reservation_context = MagicMock()
        reservation_context.__aenter__ = AsyncMock(return_value=MagicMock())
        reservation_context.__aexit__ = AsyncMock(return_value=False)
        manager = MagicMock()
        manager.reservation_context = MagicMock(return_value=reservation_context)
        manager.restore_vm = AsyncMock(side_effect=restore_error)
        scheduler._vm_manager = manager
        scheduler._snapshot_manager = MagicMock()
        return scheduler, instances, state

    async def test_state_qemu_refuses_routes(self) -> None:
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
        scheduler, instances, state = self._started(error)

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
        assert scheduler._vm_manager.reservation_context.call_args.kwargs["held"] == {Path("/saved")}  # type: ignore[union-attr]

    async def test_other_restore_failure_is_raised_as_it_is(self) -> None:
        error = TransientError(
            "guest did not answer", code=ErrorCode.GUEST_SILENT_AFTER_RESTORE, context=None, hint=None, instance_id=None
        )
        scheduler, instances, state = self._started(error)

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
    def _mock_run(destroyed_state: VmState) -> tuple[Scheduler, MagicMock, tuple[object, ...]]:
        scheduler = Scheduler()
        scheduler._started = True
        scheduler._vm_manager = MagicMock()
        scheduler._vm_manager.destroy_vm = make_destroy_mock(
            "confirmed" if destroyed_state is VmState.DESTROYED else "unconfirmed"
        )

        vm = _fake_vm("one-shot-unconfirmed")
        vm.execute = AsyncMock(
            return_value=ExecutionResult(
                stdout="ok\n",
                stderr="",
                exit_code=0,
                timing=TimingBreakdown(execute_ms=1, connect_ms=None),
            )
        )
        return scheduler, vm, (vm, [], True, 0)

    async def test_unconfirmed_vm_cleanup_raises_instead_of_returning_result(self) -> None:
        scheduler, vm, prepared = self._mock_run(VmState.DESTROYING)
        with patch.object(
            scheduler,
            "_prepare_vm",
            new_callable=AsyncMock,
            return_value=prepared,
        ):
            with pytest.raises(
                InternalError,
                check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED,
                match="process cleanup was not confirmed",
            ):
                await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')")

        scheduler._vm_manager.destroy_vm.assert_awaited_once_with(vm)

    async def test_confirmed_process_death_returns_result_while_ancillary_cleanup_retries(self) -> None:
        """Workdir/cgroup retry must not mask a result after process death."""
        scheduler, vm, prepared = self._mock_run(VmState.DESTROYED)

        scheduler._vm_manager.destroy_vm = make_destroy_mock("ancillary_pending")
        with patch.object(scheduler, "_prepare_vm", new_callable=AsyncMock, return_value=prepared):
            result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')")

        assert result.stdout == "ok\n"
        scheduler._vm_manager.destroy_vm.assert_awaited_once_with(vm)

    async def test_confirmed_vm_cleanup_returns_result(self) -> None:
        scheduler, vm, prepared = self._mock_run(VmState.DESTROYED)
        with patch.object(scheduler, "_prepare_vm", new_callable=AsyncMock, return_value=prepared):
            result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')")

        assert result.stdout == "ok\n"
        scheduler._vm_manager.destroy_vm.assert_awaited_once_with(vm)

    async def test_execution_error_remains_primary_when_cleanup_is_unconfirmed(self) -> None:
        scheduler, vm, prepared = self._mock_run(VmState.DESTROYING)
        vm.execute.side_effect = RuntimeError("execute failed")
        with patch.object(scheduler, "_prepare_vm", new_callable=AsyncMock, return_value=prepared):
            with pytest.raises(RuntimeError, match="execute failed"):
                await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')")

        scheduler._vm_manager.destroy_vm.assert_awaited_once_with(vm)

    async def test_cancellation_remains_primary_when_cleanup_is_unconfirmed(self) -> None:
        scheduler, vm, prepared = self._mock_run(VmState.DESTROYING)
        vm.execute.side_effect = asyncio.CancelledError()
        with patch.object(scheduler, "_prepare_vm", new_callable=AsyncMock, return_value=prepared):
            with pytest.raises(asyncio.CancelledError):
                await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('ok')")

        scheduler._vm_manager.destroy_vm.assert_awaited_once_with(vm)


class TestL1FallbackOwnership:
    """Cold-boot fallback cannot reuse a reservation retained by an L1 reaper.

    The L1 manager is real (its restore() runs) on a mocked VmManager; its key
    and its cache lookup are stubbed to an entry in tmp_path. Each request is a
    new() (then a close() of its session).
    """

    @staticmethod
    async def _new(scheduler: Scheduler, spec: VmSpec) -> Session:
        """new(spec), then close the session (its VM id and ports stay readable)."""
        session = await scheduler.new(spec)
        await session.close()
        return session

    @staticmethod
    def _mock_prepare(
        restore_error: Exception | None, tmp_path: Path
    ) -> tuple[Scheduler, MagicMock, MagicMock, MemorySnapshotManager]:
        scheduler = Scheduler(SchedulerConfig(auto_download_assets=False, enable_package_validation=False))
        scheduler._started = True
        scheduler._settings = MagicMock()
        scheduler._warm_pool = None

        reservation = MagicMock()
        reservation_context = MagicMock()
        reservation_context.__aenter__ = AsyncMock(return_value=reservation)
        reservation_context.__aexit__ = AsyncMock(return_value=False)

        manager = MagicMock()
        manager.reservation_context = MagicMock(return_value=reservation_context)
        manager.restore_vm = AsyncMock(side_effect=restore_error, return_value=_fake_vm("l1-restored"))
        manager.create_vm = AsyncMock(return_value=_fake_vm("cold-boot-replacement"))
        manager.destroy_vm = make_destroy_mock("confirmed")
        scheduler._vm_manager = manager

        memory = l1_hit(manager, tmp_path)
        memory.schedule_background_save = AsyncMock()  # type: ignore[method-assign]
        scheduler._memory_snapshot_manager = memory
        return scheduler, manager, reservation, memory

    async def test_unconfirmed_restore_cleanup_propagates_without_cold_boot(self, tmp_path: Path) -> None:
        scheduler, manager, _reservation, _memory = self._mock_prepare(
            InternalError(
                "Restore failure cleanup was not confirmed",
                code=ErrorCode.CLEANUP_UNCONFIRMED,
                context=None,
                hint=None,
                instance_id=None,
            ),
            tmp_path,
        )

        with pytest.raises(
            InternalError, check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED, match="cleanup was not confirmed"
        ):
            await scheduler.new(VmSpec(language=Language.PYTHON, memory_mb=192))

        manager.create_vm.assert_not_awaited()

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
        scheduler, manager, _reservation, _memory = self._mock_prepare(error, tmp_path)

        session = await self._new(scheduler, VmSpec(language=Language.PYTHON, memory_mb=192))

        assert session.vm_id == "cold-boot-replacement"
        manager.create_vm.assert_awaited_once()
        assert not (tmp_path / "cached.vmstate").exists()

    async def test_l1_entry_is_leased_through_admission(self, tmp_path: Path) -> None:
        """Admission may evict cache entries for disk space: the L1 entry is leased before it
        (and named in `held`, so gate 4 does not count on its space), and the request's
        lease goes after the restore (the VM holds its own share)."""
        scheduler, manager, _reservation, _memory = self._mock_prepare(None, tmp_path)
        vmstate = tmp_path / "cached.vmstate"
        evictable_at_admission: list[bool] = []
        admission = manager.reservation_context.return_value.__aenter__
        admission.side_effect = lambda *_args: (
            evictable_at_admission.append(evictable(vmstate)) or admission.return_value
        )

        session = await self._new(scheduler, VmSpec(language=Language.PYTHON, memory_mb=192))

        assert session.vm_id == "l1-restored"
        assert evictable_at_admission == [False]
        assert manager.reservation_context.call_args.kwargs["held"] == {vmstate}
        assert evictable(vmstate)

    async def test_confirmed_transient_restore_failure_can_reuse_reservation(self, tmp_path: Path) -> None:
        scheduler, manager, reservation, _memory = self._mock_prepare(
            TransientError(
                "migration failed", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None
            ),
            tmp_path,
        )

        session = await self._new(scheduler, VmSpec(language=Language.PYTHON, memory_mb=192))

        assert session.vm_id == "cold-boot-replacement"
        assert manager.create_vm.await_args.kwargs["reservation"] is reservation
        assert (tmp_path / "cached.vmstate").exists()  # not refused by QEMU: kept

    async def test_exposed_ports_are_resolved_and_reach_the_vm(self, tmp_path: Path) -> None:
        scheduler, manager, _reservation, _memory = self._mock_prepare(None, tmp_path)
        spec = VmSpec(
            language=Language.PYTHON, memory_mb=192, expose_ports=[PortMapping(internal=8080, external=3000), 9000]
        )

        session = await self._new(scheduler, spec)
        ports = session.exposed_ports

        assert session.vm_id == "l1-restored"
        assert all(isinstance(p, ExposedPort) for p in ports)
        assert [p.internal for p in ports] == [8080, 9000]
        assert ports[0].external == 3000  # explicit: kept
        assert isinstance(ports[1].external, int)  # none given: a host port is allocated
        assert ports[1].external > 0
        assert manager.restore_vm.await_args.kwargs["expose_ports"] == ports

    async def test_post_create_snapshot_schedule_failure_destroys_unreturned_vm(self, tmp_path: Path) -> None:
        scheduler, manager, _reservation, memory = self._mock_prepare(
            TransientError("restore miss", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None),
            tmp_path,
        )
        memory.schedule_background_save.side_effect = RuntimeError("schedule failed")  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="schedule failed"):
            await scheduler.new(VmSpec(language=Language.PYTHON, memory_mb=192))

        replacement = manager.create_vm.return_value
        manager.destroy_vm.assert_awaited_once_with(replacement)

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
        scheduler, manager, _reservation, memory = self._mock_prepare(restore_error, tmp_path)
        l2 = tmp_path / "l2.qcow2"
        l2.write_bytes(b"l2")
        lease = await CacheEntry(l2).lease()
        assert lease is not None
        spec = VmSpec(language=Language.PYTHON, memory_mb=192, packages=("pandas==2.2.3",))
        snapshots = MagicMock(get_or_create_snapshot=AsyncMock(return_value=lease))
        scheduler._snapshot_manager = snapshots
        evictable_at_admission: list[bool] = []
        admission = manager.reservation_context.return_value.__aenter__
        admission.side_effect = lambda *_args: evictable_at_admission.append(evictable(l2)) or admission.return_value

        session = await self._new(scheduler, spec)

        # Leased from the lookup: admission (it may evict for disk space) cannot take the
        # layer; the request releases its lease at its end (each VM holds its own share)
        assert evictable_at_admission == [False]
        assert evictable(l2)

        snapshots.get_or_create_snapshot.assert_awaited_once()
        assert {
            k: v
            for k, v in snapshots.get_or_create_snapshot.await_args.kwargs.items()
            if k in {"language", "packages", "memory_mb"}
        } == {"language": Language.PYTHON, "packages": ["pandas==2.2.3"], "memory_mb": 192}
        memory.compute_cache_key.assert_awaited_once_with(spec, l2_path=l2)  # type: ignore[attr-defined]
        assert manager.restore_vm.await_args.kwargs["snapshot_drive"] == l2
        assert manager.restore_vm.await_args.kwargs["cache_leases"][1] is lease
        if restore_error is None:
            assert session.vm_id == "l1-restored"
            manager.create_vm.assert_not_awaited()
            memory.schedule_background_save.assert_not_awaited()  # type: ignore[attr-defined]
        else:
            assert manager.create_vm.await_args.kwargs["snapshot_drive"] == l2
            assert manager.create_vm.await_args.kwargs["cache_leases"] == (lease,)
            memory.schedule_background_save.assert_awaited_once_with(  # type: ignore[attr-defined]
                spec, key="cached", snapshot_drive=lease
            )


class TestPackageValidation:
    """Tests for package validation in Scheduler."""

    @staticmethod
    def _started(*, validation: bool) -> tuple[Scheduler, MagicMock]:
        """A started scheduler whose L2 build raises RuntimeError("the build started"): a new()
        with packages ends there once the package check let it through. Returns (scheduler,
        the L2 cache)."""
        scheduler = Scheduler(SchedulerConfig(enable_package_validation=validation, auto_download_assets=False))
        scheduler._started = True
        scheduler._settings = MagicMock()
        scheduler._vm_manager = MagicMock()
        snapshots = MagicMock(get_or_create_snapshot=AsyncMock(side_effect=RuntimeError("the build started")))
        scheduler._snapshot_manager = snapshots
        return scheduler, snapshots

    @pytest.mark.parametrize("validation", [True, False], ids=["on", "off"])
    async def test_package_outside_the_allow_list(self, validation: bool) -> None:
        """new() with a package outside the allow-list: refused before any build or VM when
        validation is on (InvalidInputError PACKAGE_NOT_ALLOWED); built when it is off."""
        scheduler, snapshots = self._started(validation=validation)
        spec = VmSpec(language=Language.PYTHON, packages=("surely-not-a-catalog-package==1.0",))

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

    async def test_catalogs_parsed_once_off_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bundled catalogs (about 400 KB of JSON, 2 ms of parse) are parsed once per
        Scheduler, in a worker thread; later requests reuse them and validate the same way."""
        parsers: list[int] = []

        def spied_loads(text: str) -> object:
            parsers.append(threading.get_ident())
            return json.loads(text)

        monkeypatch.setattr(package_validator, "json", SimpleNamespace(loads=spied_loads))
        scheduler, _snapshots = self._started(validation=True)
        allowed = (
            VmSpec(language=Language.PYTHON, packages=("pandas==2.2.0",)),
            VmSpec(language=Language.JAVASCRIPT, packages=("lodash@4.17.21",)),
        )
        for _ in range(3):
            for spec in allowed:
                with pytest.raises(RuntimeError, match="the build started"):
                    await scheduler.new(spec)
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED, match="not in python allow-list"
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
        """With S3 settings, the Scheduler starts with an L2 cache whose build VMs come from its
        own VM manager (admission counts them).

        Private fields: a build with S3 set would call S3, and the VM manager a build uses
        shows only in a build VM. Without S3, TestBaseImageFetchBeforeSnapshot builds through
        new()."""
        config = SchedulerConfig(
            images_dir=images_dir,
            s3_bucket="test-bucket",
            s3_region="us-east-1",
            auto_download_assets=False,
        )
        async with Scheduler(config) as scheduler:
            assert scheduler._snapshot_manager is not None
            assert scheduler._snapshot_manager.vm_manager is scheduler._vm_manager


class _ColdBootError(Exception):
    """create_vm was called: the request got to its cold boot (each request here ends there)."""


class TestBaseImageFetchBeforeSnapshot:
    """Regression tests: ensure_base_image before the L2 snapshot of a new() with packages.

    When EXEC_SANDBOX_CACHE_DIR points to a fresh directory (pip-install user),
    ensure_assets() at startup only downloads kernel/initramfs/gvproxy (no
    language param). If packages are requested, the L2 cache key needs the
    base image on disk to hash it. Without the ensure_base_image() call before
    the L2 lookup, GuestImages.base_image() finds no file: ConfigError ASSET_NOT_FOUND.

    Each request is a new() with packages (package check off) that ends at its cold boot:
    create_vm raises _ColdBootError, and its snapshot_drive is the L2 snapshot.
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
    def _mock_snapshot_result(scheduler: Scheduler, tmp_path: Path, events: list[object]) -> Path:
        """get_or_create_snapshot appends "snapshot" to `events` and returns a lease on a
        qcow2 made beforehand."""
        snap = tmp_path / f"snap-{random_test_id()}.qcow2"

        async def leased(**_kwargs: object) -> EntryLease | None:
            events.append("snapshot")
            return await CacheEntry(snap).lease()

        scheduler._snapshot_manager.get_or_create_snapshot = AsyncMock(  # type: ignore[union-attr]
            side_effect=leased,
        )
        snap.touch()
        return snap

    @staticmethod
    def _setup_cold_boot_mocks(scheduler: Scheduler) -> AsyncMock:
        """No warm pool and no L1; create_vm raises _ColdBootError. Returns create_vm."""
        scheduler._warm_pool = None  # type: ignore[assignment]
        scheduler._memory_snapshot_manager = None  # type: ignore[assignment]
        create_vm = AsyncMock(side_effect=_ColdBootError)
        scheduler._vm_manager.create_vm = create_vm  # type: ignore[union-attr]
        return create_vm

    async def test_missing_base_image_fetched_before_cache_key(self, images_dir: Path, tmp_path: Path) -> None:
        """Regression: fresh cache dir without base image does not raise ConfigError ASSET_NOT_FOUND.

        The L2 cache key calls GuestImages.base_image(), which looks for the python base image.
        Without the ensure_base_image() call, it finds no file and raises
        ConfigError ASSET_NOT_FOUND. The DiskSnapshotManager is real (only its QEMU build
        is replaced), so the key, the base image lookup and the L2 cache folder are real.
        """
        fresh_images, arch = self._make_fresh_images_dir(images_dir, tmp_path)

        cache_dir = tmp_path / "snapshot-cache"
        cache_dir.mkdir(mode=0o700)

        config = self._config(fresh_images, auto_download=True, disk_snapshot_cache_dir=cache_dir)
        async with Scheduler(config) as scheduler:
            create_vm = self._setup_cold_boot_mocks(scheduler)
            # ensure_base_image mock: writes a real qcow2 at the expected path
            # (simulates what the real downloader does)
            base_image_path = fresh_images / f"python-3.14-base-{arch}.qcow2"

            async def mock_fetch(language, **kwargs):  # type: ignore[no-untyped-def]
                # Write a minimal valid qcow2 so GuestImages.base_image() finds it
                await create_test_qcow2(base_image_path)
                return base_image_path

            # Mock _create_snapshot to avoid QEMU boot — write to the real
            # cache_key path so CacheEntry.saving's "already exists" check stays correct.
            async def mock_create_snapshot(language, packages, *, cache_key, **kwargs):  # type: ignore[no-untyped-def]
                snap = cache_dir / f"{cache_key}.qcow2"
                await create_test_qcow2(snap)
                return snap

            with (
                patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch),
                patch.object(scheduler._snapshot_manager, "_create_snapshot", side_effect=mock_create_snapshot),
                pytest.raises(_ColdBootError),  # ConfigError ASSET_NOT_FOUND without the fix
            ):
                await scheduler.new(VmSpec(language=Language.PYTHON, packages=("pandas==2.0.0",)))

        snapshot = create_vm.await_args.kwargs["snapshot_drive"]
        assert (snapshot.parent, snapshot.suffix) == (cache_dir, ".qcow2")
        assert snapshot.exists()

    @pytest.mark.parametrize("language", [Language.PYTHON, Language.JAVASCRIPT, Language.RAW])
    async def test_fetch_once_before_the_snapshot(self, images_dir: Path, tmp_path: Path, language: Language) -> None:
        """auto_download_assets=True + packages: ensure_base_image runs once, with the request's
        language, before the L2 lookup; the cold boot on that snapshot does not fetch again."""
        async with Scheduler(self._config(images_dir, auto_download=True)) as scheduler:
            create_vm = self._setup_cold_boot_mocks(scheduler)
            events: list[object] = []

            async def mock_fetch(lang, **kwargs):  # type: ignore[no-untyped-def]
                events.append(("fetch", lang))

            snap = self._mock_snapshot_result(scheduler, tmp_path, events)
            with (
                patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch),
                pytest.raises(_ColdBootError),
            ):
                await scheduler.new(VmSpec(language=language, packages=("some-pkg==1.0",)))

        assert events == [("fetch", language), "snapshot"]
        assert create_vm.await_args.kwargs["snapshot_drive"] == snap

    async def test_fetch_skipped_when_auto_download_disabled(self, images_dir: Path, tmp_path: Path) -> None:
        """auto_download_assets=False → ensure_base_image NOT called (images must exist locally)."""
        async with Scheduler(self._config(images_dir, auto_download=False)) as scheduler:
            self._setup_cold_boot_mocks(scheduler)
            events: list[object] = []

            async def mock_fetch(language, **kwargs):  # type: ignore[no-untyped-def]
                events.append(("fetch", language))

            self._mock_snapshot_result(scheduler, tmp_path, events)
            with (
                patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch),
                pytest.raises(_ColdBootError),
            ):
                await scheduler.new(VmSpec(language=Language.PYTHON, packages=("pandas==2.0.0",)))

        assert events == ["snapshot"], "ensure_base_image should NOT be called when auto_download_assets=False"

    async def test_snapshot_path_skipped_without_packages(self, images_dir: Path, tmp_path: Path) -> None:
        """No packages: no L2 lookup, and the cold boot gets no snapshot drive."""
        async with Scheduler(self._config(images_dir, auto_download=True)) as scheduler:
            create_vm = self._setup_cold_boot_mocks(scheduler)

            with (
                patch.object(scheduler._snapshot_manager, "get_or_create_snapshot", new_callable=AsyncMock) as lookup,
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
        async with Scheduler(self._config(images_dir, auto_download=True)) as scheduler:
            self._setup_cold_boot_mocks(scheduler)
            events: list[object] = []

            async def mock_fetch(language, **kwargs):  # type: ignore[no-untyped-def]
                events.append(("fetch", language))

            self._mock_snapshot_result(scheduler, tmp_path, events)
            with patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch):
                for _ in range(2):
                    with pytest.raises(_ColdBootError):
                        await scheduler.new(VmSpec(language=Language.PYTHON, packages=("pkg==1.0",)))

        assert events == [("fetch", Language.PYTHON), "snapshot"] * 2

    async def test_fetch_failure_propagates(self, images_dir: Path, tmp_path: Path) -> None:
        """ensure_base_image raising TransientError ASSET_DOWNLOAD propagates to caller."""
        async with Scheduler(self._config(images_dir, auto_download=True)) as scheduler:
            create_vm = self._setup_cold_boot_mocks(scheduler)

            async def mock_fetch_fail(language, **kwargs):  # type: ignore[no-untyped-def]
                raise TransientError(
                    "Network unreachable", code=ErrorCode.ASSET_DOWNLOAD, context=None, hint=None, instance_id=None
                )

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

        config = self._config(fresh_images, auto_download=True, disk_snapshot_cache_dir=cache_dir)
        async with Scheduler(config) as scheduler:
            create_vm = self._setup_cold_boot_mocks(scheduler)

            async def mock_fetch(language, **kwargs):  # type: ignore[no-untyped-def]
                await asyncio.sleep(0.05)
                # No-op: base image already exists (idempotent)

            async def mock_create_snapshot(language, packages, *, cache_key, **kwargs):  # type: ignore[no-untyped-def]
                snap = cache_dir / f"{cache_key}.qcow2"
                if not snap.exists():
                    await create_test_qcow2(snap)
                return snap

            spec = VmSpec(language=Language.PYTHON, packages=("pandas==2.0.0",))
            with (
                patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch),
                patch.object(scheduler._snapshot_manager, "_create_snapshot", side_effect=mock_create_snapshot),
            ):
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

        config = self._config(fresh_images, auto_download=True, disk_snapshot_cache_dir=cache_dir)
        async with Scheduler(config) as scheduler:
            create_vm = self._setup_cold_boot_mocks(scheduler)
            base_image_path = fresh_images / f"python-3.14-base-{arch}.qcow2"

            async def mock_fetch(language, **kwargs):  # type: ignore[no-untyped-def]
                await create_test_qcow2(base_image_path)
                return base_image_path

            with (
                patch("exec_sandbox.scheduler.ensure_base_image", side_effect=mock_fetch),
                patch.object(
                    scheduler._snapshot_manager,
                    "_create_snapshot",
                    new_callable=AsyncMock,
                    side_effect=TransientError(
                        "VM boot failed", code=ErrorCode.QEMU_CRASHED, context=None, hint=None, instance_id=None
                    ),
                ),
            ):
                with pytest.raises(
                    TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED, match="VM boot failed"
                ):
                    await scheduler.new(VmSpec(language=Language.PYTHON, packages=("pandas==2.0.0",)))

            # Base image should still exist (not cleaned up on downstream failure)
            assert base_image_path.exists(), "Base image must survive a downstream build failure"
        create_vm.assert_not_awaited()


class TestBaseImageFetchBeforeL1Restore:
    """Regression tests: ensure_base_image ordering on the NO-PACKAGES path.

    Companion to TestBaseImageFetchBeforeSnapshot (which covers the packages /
    L2 disk-snapshot path). On a fresh cache, ensure_assets() at startup only
    downloads kernel/initramfs/gvproxy — no base images. With no packages, the
    L1 memory-snapshot compute_cache_key() (Scheduler._launch_for) hashes the
    base image via GuestImages.base_image(). If the cold-boot ensure_base_image() ran
    AFTER that, GuestImages.base_image() would find no file and raise ConfigError
    ASSET_NOT_FOUND: the cold path fetches ahead of the L1 key.

    NOTE: these tests keep a REAL MemorySnapshotManager (they do NOT use
    _setup_cold_boot_mocks, which nulls it and would hide the bug entirely). Each
    request is a new() that ends at its cold boot (create_vm raises _ColdBootError).
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
    def _fresh_config(fresh: Path, tmp_path: Path, *, auto_download: bool) -> SchedulerConfig:
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
        """PRIMARY regression: no packages + fresh cache does NOT raise ConfigError ASSET_NOT_FOUND.

        With the fetch after the L1 key, this raises ConfigError ASSET_NOT_FOUND from
        compute_cache_key -> GuestImages.base_image. It is the whole guard for this
        order.

        Parametrized over PYTHON/JAVASCRIPT (RAW omitted; the fix is
        language-agnostic and the mock writes whichever filename it is told).
        """
        fresh, arch = self._make_fresh_images_dir(images_dir, tmp_path)
        config = self._fresh_config(fresh, tmp_path, auto_download=True)
        async with Scheduler(config) as scheduler:
            create_vm = AsyncMock(side_effect=_ColdBootError)
            scheduler._vm_manager.create_vm = create_vm  # type: ignore[union-attr]
            fetch_calls: list[Language] = []

            with (
                patch(
                    "exec_sandbox.scheduler.ensure_base_image", side_effect=self._mock_fetch(fresh, arch, fetch_calls)
                ),
                pytest.raises(_ColdBootError),  # ConfigError ASSET_NOT_FOUND on the buggy code
            ):
                await scheduler.new(VmSpec(language=language))

        create_vm.assert_awaited_once()
        assert fetch_calls == [language], "fetch must run exactly once, before the L1 cache-key hash"

    async def test_auto_download_disabled_missing_base_raises(self, images_dir: Path, tmp_path: Path) -> None:
        """auto_download_assets=False + no base -> ConfigError ASSET_NOT_FOUND; fetch NOT called.

        Offline/misconfigured users must still get a clear failure — the fix
        must not paper over genuinely missing images.
        """
        fresh, arch = self._make_fresh_images_dir(images_dir, tmp_path)
        config = self._fresh_config(fresh, tmp_path, auto_download=False)
        async with Scheduler(config) as scheduler:
            scheduler._vm_manager.create_vm = AsyncMock()  # type: ignore[union-attr]
            scheduler._memory_snapshot_manager.schedule_background_save = AsyncMock()  # type: ignore[union-attr]
            fetch_calls: list[Language] = []

            with patch(
                "exec_sandbox.scheduler.ensure_base_image",
                side_effect=self._mock_fetch(fresh, arch, fetch_calls),
            ):
                with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND):
                    await scheduler.new(VmSpec(language=Language.PYTHON))

            assert fetch_calls == [], "fetch must NOT be called when auto_download_assets=False"
            scheduler._vm_manager.create_vm.assert_not_awaited()  # type: ignore[union-attr]

    async def test_on_boot_log_debug_path_still_fetches(self, images_dir: Path, tmp_path: Path) -> None:
        """on_boot_log set (the --debug path) skips L1 but still fetches before cold boot.

        Guards the removal of the old cold-boot fetch: the debug path (which
        skips check_cache) must now rely on the hoisted fetch. check_cache is
        skipped, yet the base is still fetched and create_vm proceeds.
        """
        fresh, arch = self._make_fresh_images_dir(images_dir, tmp_path)
        config = self._fresh_config(fresh, tmp_path, auto_download=True)
        async with Scheduler(config) as scheduler:
            create_vm = AsyncMock(side_effect=_ColdBootError)
            scheduler._vm_manager.create_vm = create_vm  # type: ignore[union-attr]
            fetch_calls: list[Language] = []
            check_spy = AsyncMock(wraps=scheduler._memory_snapshot_manager.check_cache)  # type: ignore[union-attr]

            with (
                patch(
                    "exec_sandbox.scheduler.ensure_base_image",
                    side_effect=self._mock_fetch(fresh, arch, fetch_calls),
                ),
                patch.object(scheduler._memory_snapshot_manager, "check_cache", check_spy),
                pytest.raises(_ColdBootError),
            ):
                await scheduler.new(VmSpec(language=Language.PYTHON), on_boot_log=lambda _line: None)

            create_vm.assert_awaited_once()
            check_spy.assert_not_awaited()  # L1 skipped when on_boot_log set
            assert fetch_calls == [Language.PYTHON], "fetch must still run on the debug/no-L1 path"

    async def test_warm_pool_hit_skips_fetch(self, images_dir: Path, tmp_path: Path) -> None:
        """Warm-pool hit returns before the cold path -> no fetch, no create_vm.

        Guards against a naive fix that hoists the fetch above the warm-pool
        try: the fast path must remain download-free.
        """
        fresh, _arch = self._make_fresh_images_dir(images_dir, tmp_path)
        config = self._fresh_config(fresh, tmp_path, auto_download=True)
        async with Scheduler(config) as scheduler:
            warm_vm = _fake_vm(f"vm-{random_test_id()}")
            scheduler._warm_pool = MagicMock()  # type: ignore[assignment]
            scheduler._warm_pool.get_vm = AsyncMock(return_value=warm_vm)  # type: ignore[union-attr]
            scheduler._warm_pool.stop = AsyncMock()  # awaited during __aexit__ shutdown  # type: ignore[union-attr]
            scheduler._vm_manager.create_vm = AsyncMock()  # type: ignore[union-attr]
            fetch_calls: list[Language] = []

            with (
                patch(
                    "exec_sandbox.scheduler.ensure_base_image",
                    side_effect=self._mock_fetch(fresh, "aarch64", fetch_calls),
                ),
                patch.object(scheduler._vm_manager, "destroy_vm", make_destroy_mock()),  # the fake VM's close
            ):
                async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
                    assert session.vm_id == warm_vm.vm_id
                    assert session.startup_timing.warm_pool_hit is True

            assert fetch_calls == [], "warm-pool hit must not download the base image"
            scheduler._vm_manager.create_vm.assert_not_awaited()  # type: ignore[union-attr]


class TestWarmPoolStartupBaseFetch:
    """Regression tests: base images downloaded before warm-pool startup.

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
    """

    @staticmethod
    def _make_scheduler() -> Scheduler:
        """Create a Scheduler with _started=True and mock managers."""
        scheduler = Scheduler()
        scheduler._started = True
        scheduler._warm_pool = MagicMock()
        scheduler._warm_pool.stop = AsyncMock()
        scheduler._memory_snapshot_manager = MagicMock()
        scheduler._memory_snapshot_manager.stop = AsyncMock()
        scheduler._vm_manager = MagicMock()
        scheduler._vm_manager.stop = AsyncMock()
        scheduler._vm_manager.get_active_vms = MagicMock(return_value={})
        return scheduler

    # ------------------------------------------------------------------
    # Normal cases
    # ------------------------------------------------------------------

    async def test_shutdown_order_warm_pool_before_memory_snapshot(self) -> None:
        """WarmVMPool.stop() is called before MemorySnapshotManager.stop()."""
        scheduler = self._make_scheduler()
        call_order: list[str] = []

        async def record_warm_pool() -> None:
            call_order.append("warm_pool_stop")

        async def record_memory_snapshot() -> None:
            call_order.append("memory_snapshot_stop")

        scheduler._warm_pool.stop.side_effect = record_warm_pool  # type: ignore[union-attr]
        scheduler._memory_snapshot_manager.stop.side_effect = record_memory_snapshot  # type: ignore[union-attr]

        await scheduler.__aexit__(None, None, None)

        assert call_order.index("warm_pool_stop") < call_order.index("memory_snapshot_stop")

    async def test_shutdown_completes_cleanly(self) -> None:
        """All managers stop, and the Scheduler refuses new requests."""
        scheduler = self._make_scheduler()

        await scheduler.__aexit__(None, None, None)

        scheduler._warm_pool.stop.assert_awaited_once()  # type: ignore[union-attr]
        scheduler._memory_snapshot_manager.stop.assert_awaited_once()  # type: ignore[union-attr]
        scheduler._vm_manager.stop.assert_awaited_once()  # type: ignore[union-attr]
        await _assert_not_started(scheduler)

    async def test_instance_saves_drain_before_vms_and_store_go(self) -> None:
        """Saves in flight end (InstanceManager.drain) after the warm pool and the L1 saves
        stop, before the remaining VMs are destroyed and before the store clients close."""
        scheduler = self._make_scheduler()
        order: list[str] = []

        def step(name: str) -> AsyncMock:
            async def record(*_args: object) -> None:
                order.append(name)

            return AsyncMock(side_effect=record)

        scheduler._warm_pool.stop = step("warm-pool")  # type: ignore[union-attr]
        scheduler._memory_snapshot_manager.stop = step("l1-saves")  # type: ignore[union-attr]
        instances = MagicMock()
        instances.drain = step("instance-saves")
        scheduler._instances = instances
        scheduler._vm_manager.get_active_vms = MagicMock(return_value={"vm-1": MagicMock()})  # type: ignore[union-attr]
        scheduler._vm_manager.destroy_vm = step("destroy-vm")  # type: ignore[union-attr]
        scheduler._vm_manager.stop = step("vm-manager")  # type: ignore[union-attr]
        store = contextlib.AsyncExitStack()
        store.push_async_callback(step("store-closed"))
        scheduler._instances_exit = store

        await scheduler.__aexit__(None, None, None)

        assert order == ["warm-pool", "l1-saves", "instance-saves", "destroy-vm", "vm-manager", "store-closed"]

    async def test_shutdown_closes_public_ingress_before_pool_drain(self) -> None:
        scheduler = self._make_scheduler()
        pool_stop_entered = asyncio.Event()
        release_pool_stop = asyncio.Event()

        async def blocked_pool_stop() -> None:
            pool_stop_entered.set()
            await release_pool_stop.wait()

        scheduler._warm_pool.stop.side_effect = blocked_pool_stop  # type: ignore[union-attr]
        shutdown = asyncio.create_task(scheduler.__aexit__(None, None, None))
        await pool_stop_entered.wait()

        await _assert_not_started(scheduler)

        release_pool_stop.set()
        await asyncio.wait_for(shutdown, timeout=1)

    async def test_cancelled_shutdown_waiter_does_not_cancel_owned_shutdown(self) -> None:
        """A cancelled context waiter cannot strand the shared shutdown sequence."""
        scheduler = self._make_scheduler()
        pool_stop_entered = asyncio.Event()
        release_pool_stop = asyncio.Event()

        async def blocked_pool_stop() -> None:
            pool_stop_entered.set()
            await release_pool_stop.wait()

        scheduler._warm_pool.stop.side_effect = blocked_pool_stop  # type: ignore[union-attr]
        first_waiter = asyncio.create_task(scheduler.__aexit__(None, None, None))
        await pool_stop_entered.wait()

        first_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter

        scheduler._memory_snapshot_manager.stop.assert_not_awaited()  # type: ignore[union-attr]
        scheduler._vm_manager.stop.assert_not_awaited()  # type: ignore[union-attr]

        # The second waiter joins the same shutdown: it ends without a CancelledError, and
        # each manager stops once
        second_waiter = asyncio.create_task(scheduler.__aexit__(None, None, None))
        release_pool_stop.set()
        await asyncio.wait_for(second_waiter, timeout=1)

        scheduler._warm_pool.stop.assert_awaited_once()  # type: ignore[union-attr]
        scheduler._memory_snapshot_manager.stop.assert_awaited_once()  # type: ignore[union-attr]
        scheduler._vm_manager.stop.assert_awaited_once()  # type: ignore[union-attr]

    async def test_reentry_is_rejected_while_owned_shutdown_is_pending(self) -> None:
        scheduler = self._make_scheduler()
        pool_stop_entered = asyncio.Event()
        release_pool_stop = asyncio.Event()

        async def blocked_pool_stop() -> None:
            pool_stop_entered.set()
            await release_pool_stop.wait()

        scheduler._warm_pool.stop.side_effect = blocked_pool_stop  # type: ignore[union-attr]
        shutdown = asyncio.create_task(scheduler.__aexit__(None, None, None))
        await pool_stop_entered.wait()

        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE, match="still shutting down"
        ):
            await scheduler.__aenter__()

        release_pool_stop.set()
        await shutdown

    async def test_shutdown_failure_leaves_scheduler_closed(self) -> None:
        scheduler = self._make_scheduler()
        scheduler._vm_manager.stop.side_effect = InternalError(
            "cleanup failed", code=ErrorCode.CLEANUP_UNCONFIRMED, context=None, hint=None, instance_id=None
        )  # type: ignore[union-attr]

        with pytest.raises(
            InternalError, check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED, match="cleanup failed"
        ):
            await scheduler.__aexit__(None, None, None)

        await _assert_not_started(scheduler)

    async def test_shutdown_background_save_unblocks_after_pool_drain(self) -> None:
        """Correct ordering unblocks a background save waiting on resources."""
        scheduler = self._make_scheduler()
        gate = asyncio.Event()

        async def warm_pool_stop() -> None:
            gate.set()  # Simulates freeing resource slots

        async def memory_snapshot_stop() -> None:
            await asyncio.wait_for(gate.wait(), timeout=2)  # Blocked until gate set

        scheduler._warm_pool.stop.side_effect = warm_pool_stop  # type: ignore[union-attr]
        scheduler._memory_snapshot_manager.stop.side_effect = memory_snapshot_stop  # type: ignore[union-attr]

        # Should complete without timeout because warm_pool stops first
        await asyncio.wait_for(scheduler.__aexit__(None, None, None), timeout=5)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    async def test_shutdown_no_warm_pool(self) -> None:
        """Shutdown works when _warm_pool is None."""
        scheduler = self._make_scheduler()
        scheduler._warm_pool = None

        await scheduler.__aexit__(None, None, None)

        scheduler._memory_snapshot_manager.stop.assert_awaited_once()  # type: ignore[union-attr]
        scheduler._vm_manager.stop.assert_awaited_once()  # type: ignore[union-attr]

    async def test_shutdown_no_memory_snapshot_manager(self) -> None:
        """Shutdown works when _memory_snapshot_manager is None."""
        scheduler = self._make_scheduler()
        scheduler._memory_snapshot_manager = None

        await scheduler.__aexit__(None, None, None)

        scheduler._warm_pool.stop.assert_awaited_once()  # type: ignore[union-attr]
        scheduler._vm_manager.stop.assert_awaited_once()  # type: ignore[union-attr]

    async def test_shutdown_no_vm_manager(self) -> None:
        """Shutdown works when _vm_manager is None."""
        scheduler = self._make_scheduler()
        scheduler._vm_manager = None

        await scheduler.__aexit__(None, None, None)

        scheduler._warm_pool.stop.assert_awaited_once()  # type: ignore[union-attr]
        scheduler._memory_snapshot_manager.stop.assert_awaited_once()  # type: ignore[union-attr]
        await _assert_not_started(scheduler)

    async def test_shutdown_destroys_remaining_active_vms(self) -> None:
        """Remaining active VMs are destroyed before vm_manager.stop()."""
        scheduler = self._make_scheduler()
        vm1, vm2 = MagicMock(), MagicMock()
        scheduler._vm_manager.get_active_vms.return_value = {"vm-1": vm1, "vm-2": vm2}  # type: ignore[union-attr]
        scheduler._vm_manager.destroy_vm = AsyncMock()  # type: ignore[union-attr]

        await scheduler.__aexit__(None, None, None)

        assert scheduler._vm_manager.destroy_vm.await_count == 2  # type: ignore[union-attr]
        scheduler._vm_manager.stop.assert_awaited_once()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Error resilience
    # ------------------------------------------------------------------

    async def test_shutdown_warm_pool_error_continues(self) -> None:
        """WarmVMPool.stop() error does not prevent other managers from stopping."""
        scheduler = self._make_scheduler()
        scheduler._warm_pool.stop.side_effect = OSError("pool drain failed")  # type: ignore[union-attr]

        await scheduler.__aexit__(None, None, None)

        scheduler._memory_snapshot_manager.stop.assert_awaited_once()  # type: ignore[union-attr]
        scheduler._vm_manager.stop.assert_awaited_once()  # type: ignore[union-attr]

    async def test_shutdown_warm_pool_internal_error_continues_then_propagates(self) -> None:
        scheduler = self._make_scheduler()
        scheduler._warm_pool.stop.side_effect = InternalError(
            "pool cleanup unconfirmed", code=ErrorCode.CLEANUP_UNCONFIRMED, context=None, hint=None, instance_id=None
        )  # type: ignore[union-attr]

        with pytest.raises(
            InternalError, check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED, match="pool cleanup unconfirmed"
        ):
            await scheduler.__aexit__(None, None, None)

        scheduler._memory_snapshot_manager.stop.assert_awaited_once()  # type: ignore[union-attr]
        scheduler._vm_manager.stop.assert_awaited_once()  # type: ignore[union-attr]

    async def test_shutdown_memory_snapshot_error_continues(self) -> None:
        """MemorySnapshotManager.stop() error does not prevent VM cleanup."""
        scheduler = self._make_scheduler()
        scheduler._memory_snapshot_manager.stop.side_effect = RuntimeError("snapshot save failed")  # type: ignore[union-attr]

        await scheduler.__aexit__(None, None, None)

        scheduler._vm_manager.stop.assert_awaited_once()  # type: ignore[union-attr]

    def _with_instances(self, failing: str) -> tuple[Scheduler, Exception, MagicMock, AsyncMock]:
        """A scheduler with saved instances on; the step `failing` raises. Returns
        (scheduler, the error, the instance manager, the store close)."""
        scheduler = self._make_scheduler()
        error = InternalError(
            f"{failing} failed", code=ErrorCode.CLEANUP_UNCONFIRMED, context=None, hint=None, instance_id=None
        )
        if failing == "warm-pool":
            scheduler._warm_pool.stop.side_effect = error  # type: ignore[union-attr]
        instances = MagicMock()
        instances.drain = AsyncMock(side_effect=error if failing == "instance-saves" else None)
        scheduler._instances = instances
        closed = AsyncMock(side_effect=error if failing == "store-close" else None)
        store = contextlib.AsyncExitStack()
        store.push_async_callback(closed)
        scheduler._instances_exit = store
        return scheduler, error, instances, closed

    @pytest.mark.parametrize("failing", ["instance-saves", "store-close"])
    async def test_failed_instance_step_still_stops_the_vms_then_propagates(self, failing: str) -> None:
        """A failed instance drain or store close: the VM cleanup still runs, then that error is raised."""
        scheduler, error, _instances, _closed = self._with_instances(failing)

        with pytest.raises(InternalError) as exc:
            await scheduler.__aexit__(None, None, None)

        assert exc.value is error
        scheduler._vm_manager.stop.assert_awaited_once()  # type: ignore[union-attr]

    @pytest.mark.parametrize("failing", ["warm-pool", "instance-saves"])
    async def test_failed_step_still_drains_and_closes_the_store(self, failing: str) -> None:
        """Always completes cleanup: after a failed step, instance saves still drain and the
        store clients still close."""
        scheduler, error, instances, closed = self._with_instances(failing)

        with pytest.raises(InternalError) as exc:
            await scheduler.__aexit__(None, None, None)

        assert exc.value is error
        instances.drain.assert_awaited_once()
        closed.assert_awaited_once()

    async def test_shutdown_destroy_vm_error_continues(self) -> None:
        """One VM destroy failure does not prevent others from being destroyed."""
        scheduler = self._make_scheduler()
        vm1, vm2 = MagicMock(), MagicMock()
        scheduler._vm_manager.get_active_vms.return_value = {"vm-1": vm1, "vm-2": vm2}  # type: ignore[union-attr]
        # First call raises, second succeeds
        scheduler._vm_manager.destroy_vm = AsyncMock(  # type: ignore[union-attr]
            side_effect=[OSError("destroy failed"), None]
        )

        await scheduler.__aexit__(None, None, None)

        # Both VMs should have had destroy attempted (gather with return_exceptions=True)
        assert scheduler._vm_manager.destroy_vm.await_count == 2  # type: ignore[union-attr]
        scheduler._vm_manager.stop.assert_awaited_once()  # type: ignore[union-attr]


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

        assert result.exit_code == 0
        assert "line 0" in result.stdout
        assert "line 1" in result.stdout
        assert "line 2" in result.stdout

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

        assert result.exit_code == 0
        assert "stdout message" in result.stdout
        assert "stderr message" in result.stderr

    async def test_run_with_env_vars(self, scheduler: Scheduler) -> None:
        """Run with custom environment variables."""
        code = """
import os
print(os.environ.get('MY_VAR', 'not set'))
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, env_vars={"MY_VAR": "hello"})

        assert result.exit_code == 0
        assert "hello" in result.stdout

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

        assert result1.exit_code == 0
        assert "first" in result1.stdout
        assert result2.exit_code == 0
        assert "second" in result2.stdout

    async def test_on_boot_log_receives_boot_output(self, scheduler: Scheduler) -> None:
        """on_boot_log callback receives kernel/init boot output."""
        boot_lines: list[str] = []

        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), "print('hello')", on_boot_log=boot_lines.append
        )

        assert result.exit_code == 0
        assert "hello" in result.stdout
        # Boot output should contain init or timing markers from tiny-init/guest-agent
        assert len(boot_lines) > 0, "on_boot_log callback should receive boot output"
        assert any(
            "init" in line.lower() or "timing" in line.lower() or "info" in line.lower() for line in boot_lines
        ), f"Expected boot markers in output, got: {boot_lines[:5]}"

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
        assert expected_output in result.stdout, f"Expected '{expected_output}' in stdout: {result.stdout}"


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
        """Timing is populated even when code raises exception."""
        result, startup, _ports = await run_timed(
            scheduler, VmSpec(language=Language.PYTHON), "raise ValueError('test error')"
        )

        assert result.exit_code != 0
        assert startup.total_ms > 0

    async def test_timing_with_large_output(self, scheduler: Scheduler) -> None:
        """Timing is populated for executions with large output."""
        # Generate 100KB of output
        code = "print('x' * 100000)"

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        assert len(result.stdout) > 0
        # Timing should still be populated
        assert result.timing is not None
        assert result.timing.execute_ms >= 0

    async def test_timing_javascript(self, scheduler: Scheduler) -> None:
        """Timing works for JavaScript executions."""
        result, startup, _ports = await run_timed(
            scheduler, VmSpec(language=Language.JAVASCRIPT), "console.log('hello')"
        )

        assert result.exit_code == 0
        assert startup.total_ms > 0

    async def test_granular_timing_populated(self, scheduler: Scheduler) -> None:
        """Granular timing fields (connect_ms, spawn_ms, process_ms) are populated."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('hello')")

        assert result.exit_code == 0

        # connect_ms should be populated (host-measured)
        assert result.timing.connect_ms is not None
        assert result.timing.connect_ms >= 0

        # Guest-reported timing should be populated
        assert result.spawn_ms is not None
        assert result.spawn_ms >= 0

        assert result.process_ms is not None
        assert result.process_ms >= 0

        # Guest timing should be reasonable relative to execution_time_ms
        if result.execution_time_ms is not None:
            # spawn + process should be <= execution_time_ms (with some margin for streaming)
            assert result.spawn_ms <= result.execution_time_ms
            assert result.process_ms <= result.execution_time_ms

    async def test_granular_timing_consistency(self, scheduler: Scheduler) -> None:
        """Granular timing values should be consistent with each other."""
        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), "import time; time.sleep(0.1); print('done')"
        )

        assert result.exit_code == 0

        # With a 100ms sleep, process_ms should be at least ~100ms
        if result.process_ms is not None:
            assert result.process_ms >= 50  # Allow some margin for timing variance


class TestSchedulerTimingEdgeCases:
    """Edge case tests for timing behavior."""

    async def test_timing_very_fast_execution(self, scheduler: Scheduler) -> None:
        """Timing handles very fast executions (sub-millisecond code)."""
        result = await run_once(
            scheduler,
            VmSpec(
                language=Language.PYTHON,
            ),
            "pass",  # Minimal Python code
        )

        assert result.exit_code == 0
        # Even for fast code, timing should be populated
        assert result.timing is not None
        # execute_ms might be 0 for very fast code, that's OK
        assert result.timing.execute_ms >= 0

    async def test_timing_empty_output(self, scheduler: Scheduler) -> None:
        """Timing works when code produces no output."""
        result = await run_once(
            scheduler,
            VmSpec(
                language=Language.PYTHON,
            ),
            "x = 1",  # No print
        )

        assert result.exit_code == 0
        assert result.stdout == ""
        # Timing should still be populated
        assert result.timing is not None

    async def test_multiple_runs_have_independent_timing(self, scheduler: Scheduler) -> None:
        """Each run has its own independent timing values."""
        # Fast execution
        result1 = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print('fast')")

        # Slightly slower execution
        result2 = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), "import time; time.sleep(0.1); print('slow')"
        )

        # Both should have timing
        assert result1.timing is not None
        assert result2.timing is not None

        # Second should have longer execute_ms (sleeping 100ms)
        # Allow some variance for timing precision
        assert result2.timing.execute_ms >= 50  # At least 50ms for sleep(0.1)


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

            # All should succeed
            for run in runs:
                assert run.result.exit_code == 0
                assert run.startup.total_ms >= 0

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

        assert result.exit_code == 0
        assert startup.warm_pool_hit is True
        # Boot is "free" for warm pool - it happened at startup
        assert startup.boot_ms == 0
        assert result.timing.execute_ms >= 0


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
        3. Filesystem sync worked (cache=unsafe issue fixed)
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

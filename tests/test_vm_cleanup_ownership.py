"""Who owns a VM's processes, files and admission slot after a failed attempt, and the
QEMU identity of the VM (VmManager, no QEMU started).

- Process death covers the whole VM cgroup: _force_cleanup_all_resources kills and waits
  for every process left in it before it removes files.
- A failed attempt whose processes are dead gives its reservation back to the caller
  (the boot retry, or the L1 cold-boot fallback, uses it); only files stay with the
  reaper. CLEANUP_UNCONFIRMED stays for an unconfirmed process death.
- The QEMU identity is the process decision: a grant that fails raises HOST_DEPENDENCY,
  never a silent run as this user.

The faults here (a process that does not die, files that stay, a grant that fails for the
qemu-vm user) cannot be made with a real VM on a test host: the tests call the private
cleanup steps with fake processes and read the owner of each reservation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox import constants
from exec_sandbox.cgroup import Cgroup
from exec_sandbox.exceptions import ConfigError, ErrorCode, InternalError
from exec_sandbox.models import Language
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import VmState

if TYPE_CHECKING:
    from collections.abc import Iterator

    from exec_sandbox.vm_manager import VmManager

_FILES_PENDING = {"qemu": True, "gvproxy": True, "workdir": False, "cgroup": False}
_ALL_DONE = {"qemu": True, "gvproxy": True, "workdir": True, "cgroup": True}


def _vm(manager: VmManager, reservation: object) -> MagicMock:
    """A launched VM in the registry that holds `reservation` (as _launch_qemu_vm leaves it)."""
    vm = MagicMock()
    vm.vm_id = "attempt-vm"
    vm.qemu_log_task = None
    vm.gvproxy_log_task = None
    vm.resource_reservation = reservation
    vm.state = VmState.BOOTING
    vm.begin_destroy = AsyncMock()

    async def confirm() -> None:
        vm.state = VmState.DESTROYED

    vm.confirm_destroyed = AsyncMock(side_effect=confirm)
    vm.channel.close = AsyncMock()
    vm.cleanup_lock = asyncio.Lock()
    manager._vms[vm.vm_id] = vm  # pyright: ignore[reportPrivateUsage]
    return vm


@pytest.fixture
def cleanup_results() -> Iterator[dict[str, bool]]:
    """What _force_cleanup_all_resources reports (the test sets it)."""
    yield {}


@pytest.fixture
def manager(unit_test_vm_manager: VmManager, cleanup_results: dict[str, bool]) -> Iterator[VmManager]:
    async def force_cleanup(**_kwargs: Any) -> dict[str, bool]:
        return dict(cleanup_results)

    with (
        patch.object(unit_test_vm_manager, "_force_cleanup_all_resources", side_effect=force_cleanup),
        patch.object(unit_test_vm_manager, "_schedule_failed_cleanup_retry") as vm_reaper,
        patch.object(unit_test_vm_manager, "_schedule_failed_pending_cleanup_retry") as pending_reaper,
        patch.object(unit_test_vm_manager._admission, "release", new_callable=AsyncMock) as release,  # pyright: ignore[reportPrivateUsage]
    ):
        unit_test_vm_manager.reapers = (vm_reaper, pending_reaper)  # type: ignore[attr-defined]
        unit_test_vm_manager.release = release  # type: ignore[attr-defined]
        yield unit_test_vm_manager
    unit_test_vm_manager._vms.clear()  # pyright: ignore[reportPrivateUsage]
    unit_test_vm_manager._pending_launches.clear()  # pyright: ignore[reportPrivateUsage]


class TestFailedAttemptOwnership:
    async def test_dead_vm_with_files_left_gives_the_reservation_back(
        self, manager: VmManager, cleanup_results: dict[str, bool]
    ) -> None:
        """QEMU confirmed dead, its files not removed yet: no CLEANUP_UNCONFIRMED (the
        caller raises the attempt's own error, so the boot retry runs); the VM gives up
        its slot; the reaper keeps only the files."""
        cleanup_results.update(_FILES_PENDING)
        reservation = MagicMock()
        vm = _vm(manager, reservation)

        await manager._cleanup_failed_owner(vm.vm_id, operation="boot_failure_cleanup", error=RuntimeError("boot"))  # pyright: ignore[reportPrivateUsage]

        assert vm.resource_reservation is None
        assert manager._vms[vm.vm_id] is vm  # pyright: ignore[reportPrivateUsage]  # the reaper owns the files
        manager.reapers[0].assert_called_once_with(vm)  # type: ignore[attr-defined]
        manager.release.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_reaper_then_releases_nothing(self, manager: VmManager, cleanup_results: dict[str, bool]) -> None:
        """The reaper later removes the files: no admission release (the caller's next VM
        holds that reservation now)."""
        cleanup_results.update(_FILES_PENDING)
        vm = _vm(manager, MagicMock())
        await manager._cleanup_failed_owner(vm.vm_id, operation="boot_failure_cleanup", error=RuntimeError("boot"))  # pyright: ignore[reportPrivateUsage]

        cleanup_results.update(_ALL_DONE)
        assert await manager.destroy_vm(vm, _schedule_retry=False) is True
        manager.release.assert_not_awaited()  # type: ignore[attr-defined]
        assert vm.vm_id not in manager._vms  # pyright: ignore[reportPrivateUsage]

    async def test_dead_pending_launch_with_files_left_gives_the_reservation_back(
        self, manager: VmManager, cleanup_results: dict[str, bool]
    ) -> None:
        cleanup_results.update(_FILES_PENDING)
        reservation = MagicMock()
        pending = manager._publish_pending_resources("pending-vm", reservation, MagicMock(), None)  # pyright: ignore[reportPrivateUsage]

        await manager._cleanup_failed_owner("pending-vm", operation="pre_boot_cleanup", error=RuntimeError("x"))  # pyright: ignore[reportPrivateUsage]

        assert pending.reservation is None
        manager.reapers[1].assert_called_once_with(pending)  # type: ignore[attr-defined]

        cleanup_results.update(_ALL_DONE)
        assert await manager._cleanup_pending_launch(pending, _schedule_retry=False) is True  # pyright: ignore[reportPrivateUsage]
        manager.release.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_unconfirmed_process_death_keeps_the_slot_and_raises(
        self, manager: VmManager, cleanup_results: dict[str, bool]
    ) -> None:
        cleanup_results.update({"qemu": False, "gvproxy": True, "workdir": False, "cgroup": False})
        reservation = MagicMock()
        vm = _vm(manager, reservation)
        with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED):
            await manager._cleanup_failed_owner(vm.vm_id, operation="boot_failure_cleanup", error=RuntimeError("b"))  # pyright: ignore[reportPrivateUsage]
        assert vm.resource_reservation is reservation

    async def test_normal_destroy_keeps_the_slot_until_files_are_gone(
        self, manager: VmManager, cleanup_results: dict[str, bool]
    ) -> None:
        """A plain destroy_vm (it releases the slot itself): the slot stays with the reaper."""
        cleanup_results.update(_FILES_PENDING)
        reservation = MagicMock()
        vm = _vm(manager, reservation)
        assert await manager.destroy_vm(vm) is False
        assert vm.resource_reservation is reservation


class TestEphemeralVm:
    @pytest.mark.parametrize(("state", "raises"), [(VmState.DESTROYED, False), (VmState.DESTROYING, True)])
    async def test_only_unconfirmed_process_death_fails_a_finished_body(
        self, unit_test_vm_manager: VmManager, state: VmState, raises: bool
    ) -> None:
        """destroy_vm not fully confirmed after a body that worked (an L1 save that
        published): an error only when QEMU itself is not confirmed dead."""
        vm = MagicMock()
        vm.vm_id = "ephemeral"
        vm.state = state
        with (
            patch.object(unit_test_vm_manager, "create_vm", new_callable=AsyncMock, return_value=vm),
            patch.object(unit_test_vm_manager, "destroy_vm", new_callable=AsyncMock, return_value=False) as destroy,
        ):

            async def run() -> None:
                async with unit_test_vm_manager.ephemeral_vm(
                    VmSpec(language=Language.PYTHON),
                    tenant_id="tenant",
                    task_id="task",
                    retry_profile=constants.RETRY_BACKGROUND,
                    direct_write_target=None,
                    snapshot_drive=None,
                    cache_leases=(),
                    mem_path=None,
                ):
                    pass

            if raises:
                with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED):
                    await run()
            else:
                await run()
        destroy.assert_awaited_once_with(vm)


class TestCgroupDrain:
    """_force_cleanup_all_resources: the reaped child is not the whole VM."""

    @staticmethod
    async def _cleanup(
        manager: VmManager, *, drained: bool, gvproxy_ended: bool, qemu_ended: bool
    ) -> tuple[dict[str, bool], list[str]]:
        calls: list[str] = []

        vm_cgroup = MagicMock(spec=Cgroup)
        vm_cgroup.kill = AsyncMock(side_effect=lambda: calls.append("kill") or drained)
        vm_cgroup.remove = AsyncMock(side_effect=lambda: calls.append("rmdir") or True)
        workdir = MagicMock()
        workdir.cleanup = AsyncMock(side_effect=lambda: calls.append("workdir") or True)
        with patch(
            "exec_sandbox.vm_manager.cleanup_process", new_callable=AsyncMock, side_effect=[qemu_ended, gvproxy_ended]
        ):
            results = await manager._force_cleanup_all_resources(  # pyright: ignore[reportPrivateUsage]
                vm_id="vm", qemu_proc=MagicMock(), gvproxy_proc=None, workdir=workdir, vm_cgroup=vm_cgroup
            )
        return results, calls

    async def test_cgroup_emptied_before_files_go(self, unit_test_vm_manager: VmManager) -> None:
        results, calls = await self._cleanup(unit_test_vm_manager, drained=True, gvproxy_ended=True, qemu_ended=True)
        assert results == _ALL_DONE
        assert calls[0] == "kill"
        assert sorted(calls[1:]) == ["rmdir", "workdir"]

    async def test_a_stuck_gvproxy_does_not_stop_the_cgroup_kill(self, unit_test_vm_manager: VmManager) -> None:
        """gvproxy does not die (D state): cgroup.kill still runs, as only it ends a qemu-vm QEMU
        whose sudo was killed; nothing of the VM is removed while gvproxy lives."""
        results, calls = await self._cleanup(unit_test_vm_manager, drained=True, gvproxy_ended=False, qemu_ended=True)
        assert results == {"qemu": True, "gvproxy": False, "workdir": False, "cgroup": False}
        assert calls == ["kill"]

    async def test_an_unconfirmed_qemu_cleanup_does_not_stop_the_cgroup_kill(
        self, unit_test_vm_manager: VmManager
    ) -> None:
        """QEMU's cleanup is not confirmed (a qemu-vm QEMU outlives its killed sudo): cgroup.kill
        still runs, as only it ends that QEMU; QEMU stays unconfirmed and nothing of the VM is
        removed."""
        results, calls = await self._cleanup(unit_test_vm_manager, drained=True, gvproxy_ended=True, qemu_ended=False)
        assert results == {"qemu": False, "gvproxy": True, "workdir": False, "cgroup": False}
        assert calls == ["kill"]

    async def test_cgroup_not_empty_is_unconfirmed_process_death(self, unit_test_vm_manager: VmManager) -> None:
        """A process stays after cgroup.kill: QEMU is not confirmed dead, and nothing of
        the live VM (folder, cgroup) is removed."""
        results, calls = await self._cleanup(unit_test_vm_manager, drained=False, gvproxy_ended=True, qemu_ended=True)
        assert results == {"qemu": False, "gvproxy": True, "workdir": False, "cgroup": False}
        assert calls == ["kill"]


class TestQemuIdentityOfAVm:
    async def test_this_user_needs_no_grant(self, unit_test_vm_manager: VmManager, tmp_path: Path) -> None:
        with (
            patch("exec_sandbox.vm_manager.probe_qemu_as_qemu_vm", AsyncMock(return_value=False)),
            patch("exec_sandbox.vm_manager.grant_qemu_vm_file_access", new_callable=AsyncMock) as grant,
        ):
            assert await unit_test_vm_manager._apply_overlay_permissions(tmp_path / "b", tmp_path / "w" / "o") is False  # pyright: ignore[reportPrivateUsage]
        grant.assert_not_awaited()

    @pytest.mark.parametrize("failing", ["base", "overlay", "folder"])
    async def test_failed_grant_is_host_dependency_not_a_silent_fallback(
        self, unit_test_vm_manager: VmManager, tmp_path: Path, failing: str
    ) -> None:
        """The process runs QEMU as qemu-vm: one VM that cannot get its files fails with
        HOST_DEPENDENCY; it never runs as this user next to qemu-vm VMs."""
        base, overlay = tmp_path / "base.qcow2", tmp_path / "w" / "overlay.qcow2"

        async def grant(path: Path, *, writable: bool) -> bool:
            return not ((failing == "base" and path == base) or (failing == "overlay" and path == overlay))

        with (
            patch("exec_sandbox.vm_manager.probe_qemu_as_qemu_vm", AsyncMock(return_value=True)),
            patch("exec_sandbox.vm_manager.grant_qemu_vm_file_access", side_effect=grant),
            patch("exec_sandbox.vm_manager.grant_qemu_vm_dir_access", MagicMock(return_value=failing != "folder")),
            pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY),
        ):
            await unit_test_vm_manager._apply_overlay_permissions(base, overlay)  # pyright: ignore[reportPrivateUsage]


class TestRestoreGrants:
    """A qemu-vm restore reads the saved state and RAM template by ACL ("r"), whoever saved them."""

    @staticmethod
    def _entry(tmp_path: Path) -> Path:
        vmstate = tmp_path / "l1.vmstate"
        vmstate.write_bytes(b"state")
        vmstate.with_suffix(".home").write_bytes(b"home")
        vmstate.with_suffix(".ram").write_bytes(b"ram")
        return vmstate

    @staticmethod
    async def _restore(
        manager: VmManager, vmstate: Path, *, granted: bool, use_template: bool
    ) -> tuple[list[tuple[list[Path], str]], AsyncMock, BaseException]:
        """restore_vm as a qemu-vm QEMU, up to the QEMU command build (it raises there).
        Returns (the ACL grants asked, the build, the error restore_vm raised)."""
        infra = MagicMock()
        infra.vm_id = "restore-vm"
        infra.workdir.use_qemu_vm_user = True
        grants: list[tuple[list[Path], str]] = []

        async def grant(paths: list[Path], *, perms: str) -> bool:
            grants.append((list(paths), perms))
            return granted

        build = AsyncMock(side_effect=RuntimeError("stop after the grants"))
        with (
            patch.object(manager, "_setup_vm_infra", AsyncMock(return_value=infra)),
            patch.object(manager, "_cleanup_failed_owner", new_callable=AsyncMock),
            patch("exec_sandbox.vm_manager.grant_qemu_vm_acl", side_effect=grant),
            patch("exec_sandbox.vm_manager.build_qemu_cmd", build),
            patch("exec_sandbox.vm_manager.create_unix_socket", MagicMock()),
            pytest.raises((ConfigError, RuntimeError)) as exc_info,
        ):
            await manager.restore_vm(
                VmSpec(language=Language.PYTHON),
                tenant_id="t",
                task_id="k",
                vmstate_path=vmstate,
                base_image=Path("/base.qcow2"),
                snapshot_drive=None,
                cache_leases=(),
                expose_ports=None,
                reservation=MagicMock(),
                machine_type=None,
                egress_domains=None,
                boot=None,
                use_template=use_template,
            )
        return grants, build, exc_info.value

    @pytest.mark.parametrize("use_template", [True, False])
    async def test_state_files_get_read_before_qemu_starts(
        self, unit_test_vm_manager: VmManager, tmp_path: Path, use_template: bool
    ) -> None:
        vmstate = self._entry(tmp_path)
        grants, build, error = await self._restore(
            unit_test_vm_manager, vmstate, granted=True, use_template=use_template
        )
        files = [vmstate, vmstate.with_suffix(".ram")] if use_template else [vmstate]
        assert grants == [(files, "r")]
        build.assert_awaited_once()
        assert (type(error), str(error)) == (RuntimeError, "stop after the grants")

    async def test_no_access_is_host_dependency_before_qemu_starts(
        self, unit_test_vm_manager: VmManager, tmp_path: Path
    ) -> None:
        """No QEMU that dies at `migrate-incoming` with EACCES: the restore stops before it
        (the L1 layer then cold boots and keeps the entry)."""
        vmstate = self._entry(tmp_path)
        grants, build, error = await self._restore(unit_test_vm_manager, vmstate, granted=False, use_template=True)
        assert grants == [([vmstate, vmstate.with_suffix(".ram")], "r")]
        assert isinstance(error, ConfigError)
        assert error.code is ErrorCode.HOST_DEPENDENCY
        build.assert_not_awaited()

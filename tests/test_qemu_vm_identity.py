"""QEMU as qemu-vm on a real Linux host (the CI setup: qemu-vm user, sudoers rule, the
test process in group qemu-vm through `sg qemu-vm`, acl, delegated cgroup v2).

- A VM's QEMU runs as qemu-vm, and its sockets are 0770 group qemu-vm whatever the sudo
  (sudo-rs gives umask 027 without the launch prefix).
- An L1 entry saved by a qemu-vm QEMU restores; so does one a QEMU of this user saved
  (one cache folder, two identities: the files are the host's, read by ACL).
- After destroy_vm, the VM cgroup is gone and no process of the VM is left.

Skipped where this process does not run QEMU as qemu-vm (macOS, a host without the setup).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from exec_sandbox import cgroup, constants
from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager
from exec_sandbox.models import Language
from exec_sandbox.permission_utils import _probe_cache as permission_probe_cache  # pyright: ignore[reportPrivateUsage]
from exec_sandbox.permission_utils import get_qemu_vm_uid, probe_qemu_as_qemu_vm, qemu_vm_acl_perms
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import skip_unless_linux

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.settings import Settings
    from exec_sandbox.vm_manager import VmManager

pytestmark = skip_unless_linux

_SPEC = VmSpec(language=Language.PYTHON)


@pytest.fixture(autouse=True)
async def _qemu_vm_host() -> None:
    if not await probe_qemu_as_qemu_vm():
        pytest.skip("this process does not run QEMU as qemu-vm here (needs the qemu-vm setup and its group)")


def _vm_processes(vm_id: str) -> dict[int, int]:
    """{pid: real uid} of every process whose command line names the VM (QEMU and its
    async-teardown process: -name guest=<vm_id>)."""
    found: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().split(b"\0")
            status = (entry / "status").read_text()
        except OSError:
            continue
        if cmdline and b"qemu-system" in cmdline[0] and any(vm_id.encode() in arg for arg in cmdline):
            uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
            found[int(entry.name)] = int(uid_line.split()[1])
    return found


async def _run(vm: QemuVM, code: str) -> str:
    result = await vm.execute(code, timeout_seconds=60, env_vars=None, on_stdout=None, on_stderr=None)
    assert result.exit_code == 0, result.stderr
    return result.stdout.strip()


async def test_vm_runs_as_qemu_vm_with_group_sockets(vm_manager: VmManager) -> None:
    vm = await vm_manager.create_vm(
        _SPEC,
        tenant_id="identity",
        task_id="uid",
        cache_leases=(),
        direct_write_target=None,
        expose_ports=None,
        on_boot_log=None,
        snapshot_drive=None,
        reservation=None,
        retry_profile=constants.RETRY_USER_FACING,
        mem_path=None,
        mem_path_share=False,
    )
    try:
        uids = set(_vm_processes(vm.vm_id).values())
        assert uids == {get_qemu_vm_uid()}
        sock = Path(vm.workdir.cmd_socket).stat()
        assert (stat.S_IMODE(sock.st_mode), sock.st_uid) == (0o770, get_qemu_vm_uid())
        assert sock.st_gid in os.getgroups() or sock.st_gid == os.getegid()
        assert await _run(vm, "print(6 * 7)") == "42"
    finally:
        assert await vm_manager.destroy_vm(vm) is True


async def test_destroy_leaves_no_cgroup_and_no_process(vm_manager: VmManager) -> None:
    """QEMU's async-teardown process outlives QEMU; destroy kills and waits for the whole
    cgroup, so the cgroup can go and destroy_vm confirms everything."""
    vm = await vm_manager.create_vm(
        _SPEC,
        tenant_id="identity",
        task_id="cgroup",
        cache_leases=(),
        direct_write_target=None,
        expose_ports=None,
        on_boot_log=None,
        snapshot_drive=None,
        reservation=None,
        retry_profile=constants.RETRY_USER_FACING,
        mem_path=None,
        mem_path_share=False,
    )
    if vm.cgroup is None:
        await vm_manager.destroy_vm(vm)
        pytest.skip("no delegated cgroup v2 for this process")
    cgroup_path = vm.cgroup.path
    # Every VM process was born in the VM cgroup (QEMU forked by sudo, its teardown
    # process): limits apply to them, and Cgroup.kill reaches them.
    in_cgroup = f"0::/{cgroup_path.relative_to(cgroup.CGROUP_V2_BASE_PATH)}"
    pids = _vm_processes(vm.vm_id)
    assert len(pids) >= 2  # QEMU and its async-teardown process
    assert {Path(f"/proc/{pid}/cgroup").read_text().strip() for pid in pids} == {in_cgroup}

    assert await vm_manager.destroy_vm(vm) is True

    assert not cgroup_path.exists()
    assert _vm_processes(vm.vm_id) == {}
    assert vm.resource_reservation is None


async def _save(memory: MemorySnapshotManager, key: str) -> Path:
    task = await memory.schedule_background_save(_SPEC, key=key, snapshot_drive=None)
    assert task is not None
    await task
    vmstate = await memory.check_cache(key)
    assert vmstate is not None, "the save published no entry (see the 'L1 ... save' warnings)"
    return vmstate


async def _restore(vm_manager: VmManager, memory: MemorySnapshotManager, key: str) -> QemuVM:
    entry = await memory.lease(key)  # before admission, as the scheduler does
    assert entry is not None, "no L1 entry to restore"
    async with vm_manager.reservation_context(
        _SPEC, vm_id="identity-restore", held=frozenset({entry.path})
    ) as reservation:
        vm = await memory.restore(
            _SPEC,
            entry=entry,
            tenant_id="identity",
            task_id="restore",
            snapshot_drive=None,
            expose_ports=None,
            reservation=reservation,
        )
        assert vm is not None, "the restore failed (see 'L1 restore failed, falling back to cold boot')"
        return vm


@pytest.mark.parametrize("saved_by_qemu_vm", [True, False], ids=["saved-as-qemu-vm", "saved-as-this-user"])
async def test_l1_save_then_restore_as_qemu_vm(
    vm_manager: VmManager, vm_settings: Settings, tmp_path: Path, saved_by_qemu_vm: bool
) -> None:
    """The cache folder is in pytest's tmp folder (0700): the grants make it traversable.
    Every entry file belongs to this user; qemu-vm reads the published state by ACL ("r",
    no write: no QEMU may change a shared template). One probe per process picks the QEMU
    identity and no setting changes it: the saved-as-this-user case sets the probe's cache."""
    memory = MemorySnapshotManager(
        vm_settings.model_copy(update={"memory_snapshot_cache_dir": tmp_path / "l1"}), vm_manager
    )
    key = await memory.compute_cache_key(_SPEC, l2_path=None)
    if not saved_by_qemu_vm:
        permission_probe_cache.qemu_as_qemu_vm = False  # this one save runs QEMU as this user
    try:
        vmstate = await _save(memory, key)
    finally:
        permission_probe_cache.qemu_as_qemu_vm = True
    ram = vmstate.with_suffix(".ram")
    assert {f.stat().st_uid for f in (vmstate, ram)} == {os.getuid()}
    expected_acl = 4 if saved_by_qemu_vm else None  # r; a save as this user grants nothing
    assert [qemu_vm_acl_perms(f) for f in (vmstate, ram)] == [expected_acl, expected_acl]

    vm = await _restore(vm_manager, memory, key)
    try:
        assert vm.l1_restored is True
        assert set(_vm_processes(vm.vm_id).values()) == {get_qemu_vm_uid()}
        assert [qemu_vm_acl_perms(f) for f in (vmstate, ram)] == [4, 4]  # the restore granted r if needed
        assert await _run(vm, "print('restored')") == "restored"
    finally:
        assert await vm_manager.destroy_vm(vm) is True
        await memory.stop()

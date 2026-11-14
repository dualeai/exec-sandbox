"""The guest images a saved instance names: the bytes the VM runs on.

The VM records the digest of its base and L2 images when QEMU opens them
(VmManager._launch_qemu_vm). A save names those digests, not the files at the
same paths at save time: an L2 entry evicted or rebuilt since holds other bytes.

Also: a save whose QMP command QEMU does not answer carries the VM's deadline diagnostics.

Each test saves through the real InstanceManager.save on a fake VM and its QMP peer on a real
Unix socket (tests/instance_support.py): no QEMU.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, Any

import pytest

from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from tests.instance_support import BLOCK, NO_VM_TIMEOUT, PYTHON, FakeQmp, Saver, SparseImage

if TYPE_CHECKING:
    from pathlib import Path

    from tests.instance_support import FakeVm, InstanceEnv

pytestmark = NO_VM_TIMEOUT


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _flip(path: Path) -> None:
    """Rebuild the file at the same path: other bytes, the same size."""
    path.write_bytes(bytes(b ^ 0xFF for b in path.read_bytes()))


async def _vm_on_a_layer(env: InstanceEnv, tmp_path: Path) -> FakeVm:
    """A fake VM on the env's base image and an L2 layer, both digests recorded at its start."""
    vm = await env.new_vm(SparseImage(size=BLOCK), SparseImage(size=BLOCK), spec=PYTHON)
    layer = tmp_path / "l2.img"
    layer.write_bytes(b"l2 layer".ljust(BLOCK, b"\0"))
    vm.packages_image, vm.packages_image_digest = layer, _sha256(layer.read_bytes())
    return vm


async def test_facts_name_the_bytes_the_vm_started_on(instance_env: InstanceEnv, tmp_path: Path) -> None:
    """Both images are rebuilt at the same paths between two saves: the second head still names
    the digests of the bytes the VM started on."""
    env = instance_env
    vm = await _vm_on_a_layer(env, tmp_path)
    assert vm.packages_image is not None
    started_on = (_sha256(env.base_image.read_bytes()), _sha256(vm.packages_image.read_bytes()))
    saver = Saver(env, vm, manager=env.manager, state=None)
    first = await saver.save()  # stores both images under their digests
    assert (first.body.host.base_image, first.body.host.packages_image) == started_on

    _flip(env.base_image)
    _flip(vm.packages_image)
    second = await saver.save()

    assert second.header.generation == 2
    assert (second.body.host.base_image, second.body.host.packages_image) == started_on


@pytest.mark.parametrize("unknown", ["base", "l2"])
async def test_unknown_digest_is_image_damaged(instance_env: InstanceEnv, tmp_path: Path, unknown: str) -> None:
    """An image missing at the VM's start has no digest: ConfigError IMAGE_DAMAGED, not a
    raw ValidationError of a "missing" digest, and the store gets nothing."""
    env = instance_env
    vm = await _vm_on_a_layer(env, tmp_path)
    if unknown == "base":
        vm.base_image_digest = None
    else:
        vm.packages_image_digest = None

    with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.IMAGE_DAMAGED) as exc:
        await env.manager.save(vm, None, new_instance_id=None)  # type: ignore[arg-type]

    assert exc.value.message == (
        f"VM {vm.vm_id}: the digest of a guest image it runs on is unknown (the file was missing "
        "when the VM started). It was not saved."
    )
    assert await env.store.list_objects("") == []


class _NoAnswerToStop(FakeQmp):
    """A QMP peer whose QEMU main loop is blocked: it never answers ``stop`` (until it is stopped)."""

    def __init__(self) -> None:
        super().__init__(on_migrate=None, on_run=None)
        self.blocked = asyncio.Event()

    async def _reply(self, command: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if command == "stop":
            await self.blocked.wait()
        return await super()._reply(command, arguments)

    async def stop(self) -> None:
        self.blocked.set()
        await super().stop()


@pytest.mark.parametrize("qemu_answers", [False, True], ids=["qmp-deadline", "stop-refused"])
async def test_a_save_that_qemu_does_not_answer_carries_deadline_diagnostics(
    instance_env: InstanceEnv, monkeypatch: pytest.MonkeyPatch, qemu_answers: bool
) -> None:
    """A QMP command of the save that times out (QEMU sent no answer: a stop that never
    returns) gets the VM's deadline diagnostics in the error context (run state, QEMU CPU
    time and threads, console); a save that QEMU refuses gets none. The QMP command deadline
    is 0.2 s here, not 10 s."""
    env = instance_env
    monkeypatch.setattr("exec_sandbox.migration_client._QMP_TIMEOUT", 0.2)
    vm = await env.new_vm(SparseImage(size=BLOCK), SparseImage(size=BLOCK), spec=PYTHON)
    if qemu_answers:
        vm.qmp.refused["stop"] = "the VM is busy"
    else:
        await vm.qmp.stop()
        vm.qmp = _NoAnswerToStop()
        await vm.qmp.start(vm.qmp_socket)
    asked: list[str] = []

    async def deadline_diagnostics(*, operation: str) -> dict[str, Any]:
        """QemuVM.deadline_diagnostics (its own tests: tests/test_deadline_diagnostics.py)."""
        asked.append(operation)
        return {"qemu_status": "running", "qemu_cpu_seconds": 1.5}

    vm.deadline_diagnostics = deadline_diagnostics  # type: ignore[attr-defined]

    with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.MIGRATION_FAILED) as exc:
        await env.manager.save(vm, None, new_instance_id=None)  # type: ignore[arg-type]

    if qemu_answers:
        assert asked == []
        assert "qemu_status" not in exc.value.context
    else:
        assert isinstance(exc.value.__cause__, TimeoutError)
        assert asked == ["save"]
        assert (exc.value.context["qemu_status"], exc.value.context["qemu_cpu_seconds"]) == ("running", 1.5)
    assert await env.store.list_objects("") == []

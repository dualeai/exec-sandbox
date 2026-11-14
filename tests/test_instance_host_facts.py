"""The guest images a saved instance names: the bytes the VM runs on.

The VM records the digest of its base and L2 images when QEMU opens them
(VmManager._launch_qemu_vm). A save names those digests, not the files at the
same paths at save time: an L2 entry evicted or rebuilt since holds other bytes.

Also: a save whose QMP command QEMU does not answer carries the VM's deadline diagnostics.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, cast
from unittest.mock import AsyncMock

import pytest

from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.instance.manager import InstanceManager

if TYPE_CHECKING:
    from pathlib import Path

_MACHINE: dict[str, Any] = {
    "arch": "x86_64",
    "accelerator": "kvm",
    "cpu_vendor": "intel",
    "cpu_model": "x86-model,vendor=GenuineIntel,enforce",
    "qemu_version": "11.1.3",
    "release": "1.0.0",
}
_BASE = hashlib.sha256(b"base image").hexdigest()
_LAYER = hashlib.sha256(b"l2 layer").hexdigest()


def _vm(tmp_path: Path, *, base_digest: str | None, packages_digest: str | None) -> SimpleNamespace:
    """A VM on a base image and an L2 layer; both files hold other bytes now (rebuilt)."""
    base, layer = tmp_path / "base.qcow2", tmp_path / "l2.qcow2"
    base.write_bytes(b"rebuilt base image")
    layer.write_bytes(b"rebuilt l2 layer")
    return SimpleNamespace(
        vm_id="saved-vm",
        base_image=base,
        packages_image=layer,
        base_image_digest=base_digest,
        packages_image_digest=packages_digest,
    )


async def _host_facts(vm: SimpleNamespace) -> Any:
    manager = SimpleNamespace(_machine_facts=AsyncMock(return_value=dict(_MACHINE)))
    return await InstanceManager._host_facts(cast("InstanceManager", manager), vm)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]


async def test_facts_name_the_bytes_the_vm_started_on(tmp_path: Path) -> None:
    facts = await _host_facts(_vm(tmp_path, base_digest=_BASE, packages_digest=_LAYER))
    assert (facts.base_image, facts.packages_image) == (_BASE, _LAYER)


@pytest.mark.parametrize(
    ("base_digest", "packages_digest"), [pytest.param(None, _LAYER, id="base"), pytest.param(_BASE, None, id="l2")]
)
async def test_unknown_digest_is_image_damaged(
    tmp_path: Path, base_digest: str | None, packages_digest: str | None
) -> None:
    """An image missing at the VM's start has no digest: ConfigError IMAGE_DAMAGED, not a
    raw ValidationError of a "missing0" digest, and nothing is saved."""
    with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.IMAGE_DAMAGED):
        await _host_facts(_vm(tmp_path, base_digest=base_digest, packages_digest=packages_digest))


class _SaveFails:
    """MigrationClient stand-in whose save_snapshot raises `error` (no QMP socket)."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __call__(self, *_args: Any) -> _SaveFails:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def save_snapshot(self, *_args: Any, **_kwargs: Any) -> None:
        raise self.error


@pytest.mark.parametrize("timed_out", [True, False], ids=["qmp-deadline", "other-failure"])
async def test_a_save_that_qemu_does_not_answer_carries_deadline_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timed_out: bool
) -> None:
    """A QMP command of the save that times out (QEMU sent no answer: a stop that never
    returns) gets the VM's deadline diagnostics in the error context (run state, QEMU CPU
    time and threads, console); another save failure gets none."""
    from exec_sandbox.exceptions import TransientError

    cause: BaseException = TimeoutError() if timed_out else OSError("broken pipe")
    error = TransientError(
        "QMP stop failed", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None
    )
    error.__cause__ = cause
    monkeypatch.setattr("exec_sandbox.instance.manager.MigrationClient", _SaveFails(error))
    monkeypatch.setattr("exec_sandbox.instance.manager.probe_qemu_version", AsyncMock(return_value=(11, 1, 3)))
    vm = SimpleNamespace(
        vm_id="saved-vm",
        use_qemu_vm_user=False,
        qmp_socket=tmp_path / "qmp.sock",
        cgroup=None,
        spec=SimpleNamespace(memory_mb=256),
        workdir=SimpleNamespace(home_disk=tmp_path / "home.img"),
        deadline_diagnostics=AsyncMock(return_value={"qemu_status": "running", "qemu_cpu_seconds": 1.5}),
    )

    with pytest.raises(TransientError) as exc:
        await InstanceManager._snapshot(  # pyright: ignore[reportPrivateUsage]
            cast("InstanceManager", SimpleNamespace()), vm, vmstate=tmp_path / "x.vmstate", home=tmp_path / "x.home"
        )

    assert exc.value is error
    assert ("qemu_status" in exc.value.context) is timed_out
    assert vm.deadline_diagnostics.await_count == (1 if timed_out else 0)

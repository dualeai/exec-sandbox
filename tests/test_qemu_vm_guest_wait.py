"""QemuVM.wait_for_guest: the end of its tasks when the boot deadline passes.

A fake guest channel stands for the guest sockets and a fake process for QEMU (it runs until
the test ends): no VM boots.
"""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from exec_sandbox.models import Language
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType


class _SlowToStopChannel:
    """A guest channel whose connect never ends; its cancel unwinds only at `release` (as a
    close of half-open sockets). The first connect (wait_for_guest's pre-connect) finds no
    socket yet."""

    def __init__(self) -> None:
        self.connects = 0
        self.unwinding = asyncio.Event()
        self.release = asyncio.Event()

    async def connect(self, timeout_seconds: float) -> None:
        self.connects += 1
        if self.connects == 1:
            raise FileNotFoundError("no socket yet")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.unwinding.set()
            await self.release.wait()
            raise


async def test_a_cancel_after_the_deadline_is_raised_not_lost() -> None:
    """The deadline passes, and wait_for_guest waits for its readiness check to end; a cancel
    of the caller meanwhile comes out as CancelledError, not as the deadline's TimeoutError
    (a boot retry would then go on for a cancelled caller)."""
    qemu_exits = asyncio.Event()
    process = MagicMock(returncode=None, wait=AsyncMock(side_effect=qemu_exits.wait))
    channel = _SlowToStopChannel()
    vm = QemuVM(
        "wait-vm",
        process=process,
        cgroup=None,
        workdir=MagicMock(qmp_events_socket=Path("/nowhere/qmp-events.sock"), use_qemu_vm_user=False),
        channel=channel,  # pyright: ignore[reportArgumentType]
        spec=VmSpec(language=Language.RAW),
        console_lines=deque(maxlen=10),
        base_image=Path("/test/base.qcow2"),
        packages_image=None,
        accel_type=AccelType.KVM,
        gvproxy_proc=None,
        qemu_log_task=None,
        gvproxy_log_task=None,
    )
    waiting = asyncio.create_task(vm.wait_for_guest(timeout=0.05))
    try:
        await asyncio.wait_for(channel.unwinding.wait(), timeout=5)  # the deadline passed
        waiting.cancel()
        channel.release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        qemu_exits.set()
        await vm.confirm_destroyed()

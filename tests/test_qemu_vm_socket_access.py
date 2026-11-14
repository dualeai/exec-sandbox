"""A guest socket this process may not connect to fails the boot wait at once.

QEMU as qemu-vm makes its sockets 0770 group qemu-vm: a process outside that group
gets EACCES on each connect. No retry helps, so wait_for_guest raises ConfigError
(HOST_DEPENDENCY) at once instead of retrying until the boot timeout.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.guest_channel import DualPortChannel
from exec_sandbox.models import Language
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.socket_auth import connect_and_verify
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType

pytestmark = pytest.mark.skipif(os.geteuid() == 0, reason="root connects whatever the socket mode")


def _listening_socket(path: Path) -> socket.socket:
    """A listening socket at `path` with mode 0 (no one but root may connect)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.listen(1)
    path.chmod(0)
    return sock


async def test_wait_for_guest_fails_at_once_on_socket_access_denied(
    short_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd, event = short_tmp_dir / "cmd.sock", short_tmp_dir / "event.sock"
    sockets = [_listening_socket(cmd), _listening_socket(event)]
    attempts: list[str] = []  # the socket of each connect

    async def counted_connect(*, path: str, **kwargs: Any) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        attempts.append(path)
        return await connect_and_verify(path=path, **kwargs)

    monkeypatch.setattr("exec_sandbox.guest_channel.connect_and_verify", counted_connect)
    qemu_exits = asyncio.Event()
    process = MagicMock()
    process.wait = AsyncMock(side_effect=qemu_exits.wait)  # QEMU runs until the test ends
    process.returncode = None
    vm = QemuVM(
        "socket-access-vm",
        process=process,
        cgroup=None,
        workdir=MagicMock(),
        channel=DualPortChannel(str(cmd), str(event), expected_uid=os.getuid()),
        spec=VmSpec(language=Language.RAW),
        console_lines=deque(maxlen=10),
        base_image=Path("/test/base.qcow2"),
        packages_image=None,
        accel_type=AccelType.KVM,
        release_callback=AsyncMock(side_effect=AssertionError("this test does not destroy the VM")),
    )
    try:
        with pytest.raises(ConfigError) as caught:
            await vm.wait_for_guest(timeout=5)
    finally:
        qemu_exits.set()
        for sock in sockets:
            sock.close()

    assert caught.value.code is ErrorCode.HOST_DEPENDENCY
    assert caught.value.context["socket"] in (str(cmd), str(event))
    assert caught.value.context["mode"] == "0o0"
    # The first denied connect raised it: no socket was tried twice (no retry until the 5 s timeout)
    assert caught.value.context["socket"] in attempts
    assert len(attempts) == len(set(attempts))

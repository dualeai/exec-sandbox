"""The guest agent never drops a host request.

The guest kernel drops the bytes that arrive while a virtio-serial port closes,
so a request sent at that moment gets no answer until the host deadline (120 s
for a file write). The agent keeps its ports open for its whole life, also when
no request comes for a long time.
"""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING

import pytest

from exec_sandbox import constants

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuVM

_IDLE_SECONDS = 19
"""Longer than 18 s: the idle time after which an agent that resets its ports would close them."""


@pytest.mark.slow
async def test_request_after_a_long_idle_is_answered(running_vm: QemuVM, monkeypatch: pytest.MonkeyPatch) -> None:
    await running_vm.write_file("before.txt", io.BytesIO(b"x"), make_executable=False)
    await asyncio.sleep(_IDLE_SECONDS)

    # A lost request fails in 10 s here, not at the 120 s default.
    monkeypatch.setattr(constants, "FILE_IO_TIMEOUT_SECONDS", 10)
    await running_vm.write_file("after.txt", io.BytesIO(b"y"), make_executable=False)

    # The agent did not reset its ports while idle (these lines print in quiet mode too).
    console = "\n".join(running_vm.console_lines)
    assert "reopening ports" not in console  # guest-agent main.rs logs it at each port reset

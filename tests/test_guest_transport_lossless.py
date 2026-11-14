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
from exec_sandbox.models import Language
from exec_sandbox.vm_spec import VmSpec

if TYPE_CHECKING:
    from exec_sandbox.vm_manager import VmManager

_IDLE_SECONDS = 19
"""Longer than 18 s: the idle time after which an agent that resets its ports would close them."""


@pytest.mark.slow
async def test_request_after_a_long_idle_is_answered(vm_manager: VmManager, monkeypatch: pytest.MonkeyPatch) -> None:
    vm = await vm_manager.create_vm(
        VmSpec(language=Language.RAW), tenant_id="test", task_id="idle-lossless", cache_leases=()
    )
    try:
        await vm.write_file("before.txt", io.BytesIO(b"x"))
        await asyncio.sleep(_IDLE_SECONDS)

        # A lost request fails in 10 s here, not at the 120 s default.
        monkeypatch.setattr(constants, "FILE_IO_TIMEOUT_SECONDS", 10)
        await vm.write_file("after.txt", io.BytesIO(b"y"))

        # The agent did not reset its ports while idle (these lines print in quiet mode too).
        console = "\n".join(vm.console_lines)
        assert "Read timeout" not in console
        assert "reopening ports" not in console
    finally:
        await vm_manager.destroy_vm(vm)

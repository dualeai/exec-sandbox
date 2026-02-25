"""Integration tests for guest agent behavior.

Tests real VM + guest agent interactions that can't be mocked.
"""

from __future__ import annotations

import asyncio

import pytest

from exec_sandbox.models import Language
from exec_sandbox.vm_manager import VmManager  # noqa: TC001
from exec_sandbox.vm_types import VmState

# Guest agent READ_TIMEOUT_MS is 12000ms (12 seconds)
# We wait longer than that to trigger the timeout/reconnect
GUEST_AGENT_READ_TIMEOUT_MS = 12000


@pytest.fixture
async def vm_manager(make_vm_manager):
    """Create a VmManager for testing."""
    yield await make_vm_manager()


class TestGuestAgentReconnect:
    """Test guest agent timeout and reconnect behavior.

    The guest agent uses NonBlockingFile with a 5-second read timeout.
    When no command is received within 5 seconds, it times out and reconnects.
    These tests verify that behavior works correctly.
    """

    async def test_reconnect_after_idle_timeout(self, vm_manager: VmManager) -> None:
        """Verify guest agent recovers after idle timeout.

        The guest agent has a 5-second read timeout. If no command is received,
        it times out and reopens the virtio-serial ports. This test verifies
        that execution still works after triggering that timeout.

        This validates the NonBlockingFile + AsyncFd implementation that enables
        proper timeout detection (unlike blocking I/O which ignores timeouts).
        """
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="test-reconnect",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            # First execution - establishes connection
            result1 = await vm.execute(
                code="print('before timeout')",
                timeout_seconds=30,
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )
            assert result1.exit_code == 0
            assert "before timeout" in result1.stdout

            # Wait longer than guest agent's READ_TIMEOUT_MS (5 seconds)
            # This triggers the guest to timeout and reconnect
            wait_time = (GUEST_AGENT_READ_TIMEOUT_MS / 1000) + 1  # 6 seconds
            await asyncio.sleep(wait_time)

            # Second execution - must work after guest reconnected
            # If NonBlockingFile timeout didn't work, guest would be hung
            result2 = await vm.execute(
                code="print('after timeout')",
                timeout_seconds=30,
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )
            assert result2.exit_code == 0
            assert "after timeout" in result2.stdout

        finally:
            await vm_manager.destroy_vm(vm)
            assert vm.state == VmState.DESTROYED

    async def test_multiple_reconnects(self, vm_manager: VmManager) -> None:
        """Verify guest agent handles multiple timeout/reconnect cycles.

        Tests that the reconnect mechanism is robust and can handle
        repeated idle periods.
        """
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="test-multi-reconnect",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            wait_time = (GUEST_AGENT_READ_TIMEOUT_MS / 1000) + 1  # 6 seconds

            for i in range(3):
                # Execute code
                result = await vm.execute(
                    code=f"print('iteration {i}')",
                    timeout_seconds=30,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0
                assert f"iteration {i}" in result.stdout

                # Wait for timeout (except after last iteration)
                if i < 2:
                    await asyncio.sleep(wait_time)

        finally:
            await vm_manager.destroy_vm(vm)

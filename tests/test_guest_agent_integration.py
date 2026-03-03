"""Integration tests for guest agent behavior.

Tests real VM + guest agent interactions that can't be mocked.
"""

from __future__ import annotations

import asyncio
from io import BytesIO

from exec_sandbox.models import Language
from exec_sandbox.vm_manager import VmManager  # noqa: TC001
from exec_sandbox.vm_types import VmState

# Guest agent READ_TIMEOUT_MS is 18000ms (18 seconds)
# See guest-agent/src/constants.rs:31 — must match to trigger reconnect
GUEST_AGENT_READ_TIMEOUT_MS = 18000


class TestGuestAgentReconnect:
    """Test guest agent timeout and reconnect behavior.

    The guest agent uses NonBlockingFile with an 18-second read timeout
    (see guest-agent/src/constants.rs READ_TIMEOUT_MS).
    When no command is received within that window, it times out and reconnects.
    These tests verify that behavior works correctly.
    """

    async def test_reconnect_after_idle_timeout(self, vm_manager: VmManager) -> None:
        """Verify guest agent recovers after idle timeout.

        The guest agent has an 18-second read timeout (READ_TIMEOUT_MS).
        If no command is received, it times out and reopens the virtio-serial
        ports. This test verifies that execution still works after triggering
        that timeout.

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

            # Wait longer than guest agent's READ_TIMEOUT_MS (18 seconds)
            # This triggers the guest to timeout and reconnect
            wait_time = (GUEST_AGENT_READ_TIMEOUT_MS / 1000) + 1
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
            wait_time = (GUEST_AGENT_READ_TIMEOUT_MS / 1000) + 1

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


class TestWriteFileAfterIdleTimeout:
    """Test write_file works after guest agent idle timeout.

    Reproduces the CI failure: host holds a stale connection after the
    guest agent's 18s read timeout triggers a reconnect cycle.  Without
    the is_connected() fix, connect() sees is_connected()=True, skips
    reconnection, and write_file sends into a dead socket.
    """

    async def test_write_file_after_idle_timeout(self, vm_manager: VmManager) -> None:
        """write_file succeeds after a single idle timeout cycle."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="test-write-after-idle",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            # Establish connection via execute
            result1 = await vm.execute(
                code="print('connected')",
                timeout_seconds=30,
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )
            assert result1.exit_code == 0

            # Wait for guest READ_TIMEOUT_MS + margin to trigger reconnect
            wait_time = (GUEST_AGENT_READ_TIMEOUT_MS / 1000) + 2
            await asyncio.sleep(wait_time)

            # write_file with small content — the exact failing case from CI
            await vm.write_file("test.txt", BytesIO(b"x"))

            # Verify via execute
            result2 = await vm.execute(
                code="print(open('/home/user/test.txt').read())",
                timeout_seconds=30,
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )
            assert result2.exit_code == 0
            assert "x" in result2.stdout

        finally:
            await vm_manager.destroy_vm(vm)
            assert vm.state == VmState.DESTROYED

    async def test_write_file_after_multiple_idle_timeouts(self, vm_manager: VmManager) -> None:
        """write_file survives 3 consecutive idle timeout cycles."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="test-write-multi-idle",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            wait_time = (GUEST_AGENT_READ_TIMEOUT_MS / 1000) + 2

            for i in range(3):
                # Execute to ensure connection is alive
                result = await vm.execute(
                    code=f"print('cycle {i}')",
                    timeout_seconds=30,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0

                # Wait for idle timeout
                await asyncio.sleep(wait_time)

                # write_file after timeout
                rel_path = f"file_{i}.txt"
                content = f"data_{i}".encode()
                await vm.write_file(rel_path, BytesIO(content))

                # Verify via execute (absolute path inside VM)
                verify = await vm.execute(
                    code=f"print(open('/home/user/{rel_path}').read())",
                    timeout_seconds=30,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert verify.exit_code == 0
                assert f"data_{i}" in verify.stdout

        finally:
            await vm_manager.destroy_vm(vm)
            assert vm.state == VmState.DESTROYED

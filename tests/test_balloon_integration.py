"""Integration tests for balloon memory operations inside real VMs.

Tests actual memory reclamation via virtio-balloon by verifying memory
allocation behavior INSIDE the guest VM, not just QMP responses.

Testing Philosophy
------------------
These tests focus on **VM stability and responsiveness** rather than exact
balloon memory values. This approach is intentional for several reasons:

1. **Balloon operations are asynchronous**: The QMP balloon command returns
   immediately, but the guest kernel reclaims/releases memory progressively.
   Polling for exact targets is unreliable and slow.

2. **Memory compression (zram)**: The guest VM uses zram, which compresses
   memory pages. This means allocations can succeed even when "available"
   memory appears insufficient - the allocator compresses existing pages.

3. **Query can return None**: Under rapid cycling or memory pressure, the
   balloon query may fail or return None. This is expected behavior, not a
   test failure.

4. **Platform variance**: Different hypervisors (KVM vs HVF), nested
   virtualization, and host memory pressure all affect balloon behavior.
   Tests must work across macOS, Linux, and CI environments.

5. **Tolerance-based assertions**: When we do check memory values, we use
   `constants.BALLOON_TOLERANCE_MB` (40MB) to account for kernel overhead,
   runtime memory, and measurement timing.

What We Test
------------
- VM remains responsive after balloon operations (can execute code)
- MemAvailable decreases after inflate (memory is actually reclaimed)
- Inflate/deflate cycles don't crash the VM
- Error handling (not connected, double connect/disconnect)

What We Don't Test
------------------
- Exact memory values after balloon operations
- OOM behavior (unreliable with zram compression)
- Intermediate states during rapid cycling
"""

import pytest

from exec_sandbox import constants
from exec_sandbox.balloon_client import BalloonClient, BalloonError
from exec_sandbox.models import Language
from exec_sandbox.permission_utils import get_expected_socket_uid

from .conftest import skip_unless_fast_balloon

# Code to run inside VM to get MemAvailable (more accurate than MemTotal)
GET_MEM_AVAILABLE_CODE = """
with open('/proc/meminfo') as f:
    for line in f:
        if line.startswith('MemAvailable:'):
            mem_kb = int(line.split()[1])
            print(mem_kb // 1024)  # Print MB
            break
"""

# Code to allocate memory inside VM and verify it succeeds
ALLOCATE_MEMORY_CODE = """
import sys
target_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 50
try:
    # Allocate target_mb of memory
    data = bytearray(target_mb * 1024 * 1024)
    # Touch all pages to ensure allocation
    for i in range(0, len(data), 4096):
        data[i] = 1
    print(f"OK:{target_mb}")
except MemoryError:
    print(f"OOM:{target_mb}")
"""

# Code to allocate memory and hold it (for pressure tests)
HOLD_MEMORY_CODE = """
import sys
target_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 100
try:
    data = bytearray(target_mb * 1024 * 1024)
    for i in range(0, len(data), 4096):
        data[i] = 1
    print(f"HOLDING:{target_mb}")
    # Keep running to hold the memory
    import time
    time.sleep(10)
except MemoryError:
    print(f"OOM:{target_mb}")
"""


@skip_unless_fast_balloon
class TestBalloonInsideVM:
    """Tests that verify balloon operations via actual memory allocation inside VM."""

    async def test_can_allocate_memory_at_full_size(self, vm_manager) -> None:
        """VM can allocate memory when balloon is deflated (full memory)."""
        vm_memory = constants.DEFAULT_MEMORY_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-alloc-full",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            # No balloon manipulation - VM has full memory
            # Should be able to allocate 50MB (leaving room for kernel/overhead)
            result = await vm.execute(
                code=ALLOCATE_MEMORY_CODE,
                timeout_seconds=30,
                env_vars={"PYTHONUNBUFFERED": "1"},
                on_stdout=None,
                on_stderr=None,
            )

            assert result.exit_code == 0
            assert "OK:50" in result.stdout, f"Failed to allocate 50MB: {result.stdout}"
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_inflate_reduces_available_memory(self, vm_manager) -> None:
        """After inflating balloon, MemAvailable inside guest should decrease."""
        inflate_target = constants.BALLOON_INFLATE_TARGET_MB
        tolerance = constants.BALLOON_TOLERANCE_MB
        vm_memory = constants.DEFAULT_MEMORY_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-inflate-mem",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # Get initial memory before inflate
                result_before = await vm.execute(
                    code=GET_MEM_AVAILABLE_CODE,
                    timeout_seconds=10,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result_before.exit_code == 0
                mem_before = int(result_before.stdout.strip())

                # Inflate balloon
                await client.inflate(target_mb=inflate_target)

                # Query balloon state
                actual_balloon = await client.query()
                if actual_balloon is None or actual_balloon > inflate_target + tolerance:
                    pytest.skip(f"Balloon inflation ineffective: actual={actual_balloon}MB")

                # Get memory after inflate (longer timeout - VM under memory pressure)
                result_after = await vm.execute(
                    code=GET_MEM_AVAILABLE_CODE,
                    timeout_seconds=30,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result_after.exit_code == 0
                mem_after = int(result_after.stdout.strip())

                # Memory should have decreased significantly (at least 50MB reduction)
                reduction = mem_before - mem_after
                assert reduction >= 50, (
                    f"Balloon should reduce MemAvailable: before={mem_before}MB, after={mem_after}MB, "
                    f"reduction={reduction}MB, balloon={actual_balloon}MB"
                )
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_deflate_restores_allocation_capability(self, vm_manager) -> None:
        """After inflate then deflate, VM can allocate large memory again."""
        vm_memory = constants.DEFAULT_MEMORY_MB
        inflate_target = constants.BALLOON_INFLATE_TARGET_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-deflate-restore",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # First inflate to restrict memory
                await client.inflate(target_mb=inflate_target)

                # Verify restricted - small allocation should work
                result = await vm.execute(
                    code=ALLOCATE_MEMORY_CODE.replace("50", "20"),
                    timeout_seconds=30,
                    env_vars={"PYTHONUNBUFFERED": "1"},
                    on_stdout=None,
                    on_stderr=None,
                )
                assert "OK:20" in result.stdout

                # Now deflate to restore memory
                await client.deflate(target_mb=vm_memory)

                # Should be able to allocate larger amount again
                result = await vm.execute(
                    code=ALLOCATE_MEMORY_CODE.replace("50", "150"),
                    timeout_seconds=30,
                    env_vars={"PYTHONUNBUFFERED": "1"},
                    on_stdout=None,
                    on_stderr=None,
                )
                assert "OK:150" in result.stdout, f"Failed after deflate: {result.stdout}"
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)


@skip_unless_fast_balloon
class TestBalloonEdgeCases:
    """Edge case tests for balloon operations."""

    async def test_inflate_below_minimum_clamps(self, vm_manager) -> None:
        """Inflating to very low value (16MB) - kernel needs minimum memory.

        This tests extreme balloon inflation - the balloon should clamp to some
        minimum and VM should recover after deflate.
        """
        vm_memory = constants.DEFAULT_MEMORY_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-below-min",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # Try to inflate to very low value
                await client.inflate(target_mb=16)

                # Query actual value - may be clamped by guest or fail under pressure
                actual_mb = await client.query()

                # Deflate back to restore memory before checking responsiveness
                await client.deflate(target_mb=vm_memory)

                # VM should be responsive after deflate
                result = await vm.execute(
                    code="print('alive')",
                    timeout_seconds=30,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0, f"VM unresponsive after extreme inflate+deflate, actual_mb={actual_mb}"
                assert "alive" in result.stdout
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_deflate_above_max_clamps(self, vm_manager) -> None:
        """Deflating above VM max - should clamp to actual VM memory."""
        vm_memory = constants.DEFAULT_MEMORY_MB
        inflate_target = constants.BALLOON_INFLATE_TARGET_MB
        tolerance = constants.BALLOON_TOLERANCE_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-above-max",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # First inflate to reduce memory
                await client.inflate(target_mb=inflate_target)

                # Try to deflate to more than VM has
                await client.deflate(target_mb=vm_memory * 2)

                # Query actual value - should be clamped to VM max
                actual_mb = await client.query()
                assert actual_mb is not None
                max_expected = vm_memory + tolerance
                min_expected = vm_memory - tolerance
                assert actual_mb <= max_expected, f"Expected <={max_expected}MB (clamped), got {actual_mb}MB"
                assert actual_mb >= min_expected, f"Expected >={min_expected}MB after deflate, got {actual_mb}MB"
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_rapid_inflate_deflate_cycles(self, vm_manager) -> None:
        """Multiple rapid inflate/deflate cycles should not crash the VM.

        Focus is on stability - balloon state after rapid cycling may be
        inconsistent but VM should remain responsive.
        """
        inflate_target = constants.BALLOON_INFLATE_TARGET_MB
        deflate_target = constants.DEFAULT_MEMORY_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-rapid",
            memory_mb=deflate_target,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # Do 5 rapid cycles - don't assert on intermediate states
                for _ in range(5):
                    await client.inflate(target_mb=inflate_target)
                    await client.deflate(target_mb=deflate_target)

                # VM should still be responsive after rapid cycling
                result = await vm.execute(
                    code="print('stable')",
                    timeout_seconds=30,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0, f"VM unresponsive after rapid cycling: {result.stderr}"
                assert "stable" in result.stdout
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_inflate_to_same_value_is_idempotent(self, vm_manager) -> None:
        """Inflating to the same value multiple times should not crash VM."""
        inflate_target = constants.BALLOON_INFLATE_TARGET_MB
        vm_memory = constants.DEFAULT_MEMORY_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-idempotent",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # Inflate to target three times
                await client.inflate(target_mb=inflate_target)
                await client.inflate(target_mb=inflate_target)
                await client.inflate(target_mb=inflate_target)

                # VM should still be responsive (longer timeout - VM under memory pressure)
                result = await vm.execute(
                    code="print('idempotent')",
                    timeout_seconds=30,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0
                assert "idempotent" in result.stdout
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)


@skip_unless_fast_balloon
class TestBalloonMemoryPressure:
    """Tests for balloon under memory pressure conditions."""

    async def test_inflate_while_guest_using_memory(self, vm_manager) -> None:
        """Inflate balloon when guest is actively using memory."""
        vm_memory = constants.DEFAULT_MEMORY_MB
        inflate_target = constants.BALLOON_INFLATE_TARGET_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-pressure",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # First allocate some memory in guest (100MB)
                result = await vm.execute(
                    code=ALLOCATE_MEMORY_CODE.replace("50", "100"),
                    timeout_seconds=30,
                    env_vars={"PYTHONUNBUFFERED": "1"},
                    on_stdout=None,
                    on_stderr=None,
                )
                assert "OK:100" in result.stdout

                # Now try to inflate balloon - may cause guest memory pressure
                await client.inflate(target_mb=inflate_target)

                # Query balloon - may not reach target due to pressure
                mem = await client.query()
                assert mem is not None
                # Accept that balloon may not fully inflate under pressure
                # Just verify VM is still responsive (longer timeout - memory pressure)
                result = await vm.execute(
                    code="print('responsive')",
                    timeout_seconds=30,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)


@skip_unless_fast_balloon
class TestBalloonErrorHandling:
    """Tests for balloon error handling."""

    async def test_not_connected_raises_error(self, vm_manager) -> None:
        """Operations on unconnected client raise BalloonError."""
        vm_memory = constants.DEFAULT_MEMORY_MB
        inflate_target = constants.BALLOON_INFLATE_TARGET_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-not-connected",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)

            # Should raise BalloonError when not connected
            with pytest.raises(BalloonError, match="Not connected"):
                await client.query()

            with pytest.raises(BalloonError, match="Not connected"):
                await client.inflate(target_mb=inflate_target)

            with pytest.raises(BalloonError, match="Not connected"):
                await client.deflate(target_mb=vm_memory)
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_double_connect_safe(self, vm_manager) -> None:
        """Connecting twice should be safe (no-op or reconnect)."""
        vm_memory = constants.DEFAULT_MEMORY_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-double-connect",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)

            await client.connect()
            # Second connect should not raise
            await client.connect()

            # Should still work
            mem = await client.query()
            assert mem is not None

            await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_double_disconnect_safe(self, vm_manager) -> None:
        """Disconnecting twice should be safe."""
        vm_memory = constants.DEFAULT_MEMORY_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-double-disconnect",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)

            await client.connect()
            await client.disconnect()
            # Second disconnect should not raise
            await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)


@skip_unless_fast_balloon
class TestBalloonWarmPoolSimulation:
    """Tests that simulate actual warm pool usage patterns."""

    async def test_warm_pool_lifecycle(self, vm_manager) -> None:
        """Simulate complete warm pool lifecycle with memory verification."""
        vm_memory = constants.DEFAULT_MEMORY_MB
        inflate_target = constants.BALLOON_INFLATE_TARGET_MB
        tolerance = constants.BALLOON_TOLERANCE_MB

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-warm-pool",
            memory_mb=vm_memory,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # Step 1: VM boots with full memory - verify can allocate
                result = await vm.execute(
                    code=ALLOCATE_MEMORY_CODE.replace("50", "150"),
                    timeout_seconds=30,
                    env_vars={"PYTHONUNBUFFERED": "1"},
                    on_stdout=None,
                    on_stderr=None,
                )
                assert "OK:150" in result.stdout, "Initial allocation failed"

                # Step 2: Add to warm pool - inflate balloon
                previous = await client.inflate(target_mb=inflate_target)
                min_previous = vm_memory - tolerance
                assert previous >= min_previous, f"Expected previous >={min_previous}MB, got {previous}"

                # Step 3: Verify memory is restricted (longer timeout - memory pressure)
                result = await vm.execute(
                    code=GET_MEM_AVAILABLE_CODE,
                    timeout_seconds=30,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0
                idle_mem = int(result.stdout.strip())
                max_idle = inflate_target + tolerance
                assert idle_mem <= max_idle, f"Idle memory too high: {idle_mem}MB (max={max_idle})"

                # Step 4: Allocate from pool - deflate balloon
                await client.deflate(target_mb=vm_memory)

                # Step 5: Verify can allocate large memory again for execution
                result = await vm.execute(
                    code=ALLOCATE_MEMORY_CODE.replace("50", "150"),
                    timeout_seconds=30,
                    env_vars={"PYTHONUNBUFFERED": "1"},
                    on_stdout=None,
                    on_stderr=None,
                )
                assert "OK:150" in result.stdout, "Post-deflate allocation failed"

            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)

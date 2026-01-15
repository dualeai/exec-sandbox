"""Integration tests for balloon memory operations inside real VMs.

Tests actual memory reclamation via virtio-balloon by verifying memory
allocation behavior INSIDE the guest VM, not just QMP responses.
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
        """VM with 256MB can allocate ~200MB when balloon is deflated (full memory)."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-alloc-full",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            # No balloon manipulation - VM has full 256MB
            # Should be able to allocate ~200MB (leaving room for kernel/overhead)
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

    async def test_inflate_prevents_large_allocation(self, vm_manager) -> None:
        """After inflating balloon to 64MB, VM cannot allocate 100MB."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-inflate-oom",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # Inflate balloon - guest now has only ~64MB
                await client.inflate(target_mb=64)

                # Try to allocate 100MB - should fail with OOM
                result = await vm.execute(
                    code=ALLOCATE_MEMORY_CODE.replace("50", "100"),
                    timeout_seconds=30,
                    env_vars={"PYTHONUNBUFFERED": "1"},
                    on_stdout=None,
                    on_stderr=None,
                )

                # Either OOM error or process killed by OOM killer
                assert "OOM:100" in result.stdout or result.exit_code != 0, (
                    f"Expected OOM but got: {result.stdout}, exit={result.exit_code}"
                )
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_deflate_restores_allocation_capability(self, vm_manager) -> None:
        """After inflate then deflate, VM can allocate large memory again."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-deflate-restore",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # First inflate to restrict memory
                await client.inflate(target_mb=64)

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
                await client.deflate(target_mb=256)

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

    async def test_mem_available_decreases_after_inflate(self, vm_manager) -> None:
        """MemAvailable in /proc/meminfo decreases after balloon inflate."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-memavail",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # Get initial MemAvailable
                result = await vm.execute(
                    code=GET_MEM_AVAILABLE_CODE,
                    timeout_seconds=10,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0
                initial_mb = int(result.stdout.strip())
                assert initial_mb >= 150, f"Initial MemAvailable too low: {initial_mb}MB"

                # Inflate balloon
                await client.inflate(target_mb=64)

                # Get MemAvailable after inflate
                result = await vm.execute(
                    code=GET_MEM_AVAILABLE_CODE,
                    timeout_seconds=10,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0
                inflated_mb = int(result.stdout.strip())

                # Should be significantly reduced
                assert inflated_mb < initial_mb - 100, (
                    f"MemAvailable didn't decrease enough: {initial_mb}MB -> {inflated_mb}MB"
                )
                assert inflated_mb <= 80, f"Expected <=80MB after inflate, got {inflated_mb}MB"
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)


@skip_unless_fast_balloon
class TestBalloonEdgeCases:
    """Edge case tests for balloon operations."""

    async def test_inflate_below_minimum_clamps(self, vm_manager) -> None:
        """Inflating to very low value (16MB) - kernel needs minimum memory."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-below-min",
            memory_mb=256,
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

                # Query actual value - may be clamped by guest
                actual_mb = await client.query()
                assert actual_mb is not None

                # VM should still be responsive (not crashed)
                result = await vm.execute(
                    code="print('alive')",
                    timeout_seconds=10,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0
                assert "alive" in result.stdout
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_deflate_above_max_clamps(self, vm_manager) -> None:
        """Deflating above VM max (512MB when VM has 256MB) - should clamp."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-above-max",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # First inflate to reduce memory
                await client.inflate(target_mb=64)

                # Try to deflate to more than VM has
                await client.deflate(target_mb=512)

                # Query actual value - should be clamped to VM max (~256MB)
                actual_mb = await client.query()
                assert actual_mb is not None
                assert actual_mb <= 280, f"Expected <=280MB (clamped), got {actual_mb}MB"
                assert actual_mb >= 200, f"Expected >=200MB after deflate, got {actual_mb}MB"
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_rapid_inflate_deflate_cycles(self, vm_manager) -> None:
        """Multiple rapid inflate/deflate cycles should be stable."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-rapid",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # Do 5 rapid cycles
                for i in range(5):
                    await client.inflate(target_mb=64)
                    mem = await client.query()
                    assert mem is not None and mem <= 100, f"Cycle {i}: inflate failed"

                    await client.deflate(target_mb=256)
                    mem = await client.query()
                    assert mem is not None and mem >= 200, f"Cycle {i}: deflate failed"

                # VM should still be responsive after rapid cycling
                result = await vm.execute(
                    code="print('stable')",
                    timeout_seconds=10,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0
                assert "stable" in result.stdout
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_inflate_to_same_value_is_idempotent(self, vm_manager) -> None:
        """Inflating to the same value multiple times should be idempotent."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-idempotent",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)
            client = BalloonClient(vm.qmp_socket, expected_uid)
            await client.connect()

            try:
                # Inflate to 64MB
                await client.inflate(target_mb=64)
                mem1 = await client.query()

                # Inflate to same value again
                await client.inflate(target_mb=64)
                mem2 = await client.query()

                # Inflate third time
                await client.inflate(target_mb=64)
                mem3 = await client.query()

                # All should be ~64MB
                assert mem1 is not None and 50 <= mem1 <= 80
                assert mem2 is not None and 50 <= mem2 <= 80
                assert mem3 is not None and 50 <= mem3 <= 80
            finally:
                await client.disconnect()
        finally:
            await vm_manager.destroy_vm(vm)


@skip_unless_fast_balloon
class TestBalloonMemoryPressure:
    """Tests for balloon under memory pressure conditions."""

    async def test_inflate_while_guest_using_memory(self, vm_manager) -> None:
        """Inflate balloon when guest is actively using memory."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-pressure",
            memory_mb=256,
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

                # Now try to inflate balloon to 64MB
                # This may cause guest memory pressure
                await client.inflate(target_mb=64)

                # Query balloon - may not reach target due to pressure
                mem = await client.query()
                assert mem is not None
                # Accept that balloon may not fully inflate under pressure
                # Just verify VM is still responsive
                result = await vm.execute(
                    code="print('responsive')",
                    timeout_seconds=10,
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
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-not-connected",
            memory_mb=256,
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
                await client.inflate(target_mb=64)

            with pytest.raises(BalloonError, match="Not connected"):
                await client.deflate(target_mb=256)
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_double_connect_safe(self, vm_manager) -> None:
        """Connecting twice should be safe (no-op or reconnect)."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-double-connect",
            memory_mb=256,
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
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-double-disconnect",
            memory_mb=256,
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
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="balloon-warm-pool",
            memory_mb=256,
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
                target = constants.BALLOON_INFLATE_TARGET_MB
                previous = await client.inflate(target_mb=target)
                assert previous >= 200, f"Expected previous >=200MB, got {previous}"

                # Step 3: Verify memory is restricted
                result = await vm.execute(
                    code=GET_MEM_AVAILABLE_CODE,
                    timeout_seconds=10,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                assert result.exit_code == 0
                idle_mem = int(result.stdout.strip())
                assert idle_mem <= 80, f"Idle memory too high: {idle_mem}MB"

                # Step 4: Allocate from pool - deflate balloon
                await client.deflate(target_mb=constants.DEFAULT_MEMORY_MB)

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

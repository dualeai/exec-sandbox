"""Tests for VmManager.

Unit tests: State machine, platform detection.
Integration tests: Real VM lifecycle (requires QEMU + images).
"""

import asyncio
import sys

import pytest

from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostOS, detect_host_os
from exec_sandbox.vm_manager import VALID_STATE_TRANSITIONS, VmState, _check_kvm_available

# ============================================================================
# Unit Tests - VM State Machine
# ============================================================================


class TestVmState:
    """Tests for VmState enum."""

    def test_state_values(self) -> None:
        """VmState has expected values."""
        assert VmState.CREATING.value == "creating"
        assert VmState.BOOTING.value == "booting"
        assert VmState.READY.value == "ready"
        assert VmState.EXECUTING.value == "executing"
        assert VmState.DESTROYING.value == "destroying"
        assert VmState.DESTROYED.value == "destroyed"

    def test_all_states_defined(self) -> None:
        """All 6 VM states are defined."""
        assert len(VmState) == 6


class TestStateTransitions:
    """Tests for VM state transition table."""

    def test_all_states_have_transitions(self) -> None:
        """All states have transition rules defined."""
        assert set(VmState) == set(VALID_STATE_TRANSITIONS.keys())

    def test_creating_transitions(self) -> None:
        """CREATING can transition to BOOTING or DESTROYING."""
        assert VALID_STATE_TRANSITIONS[VmState.CREATING] == {VmState.BOOTING, VmState.DESTROYING}

    def test_booting_transitions(self) -> None:
        """BOOTING can transition to READY or DESTROYING."""
        assert VALID_STATE_TRANSITIONS[VmState.BOOTING] == {VmState.READY, VmState.DESTROYING}

    def test_ready_transitions(self) -> None:
        """READY can transition to EXECUTING or DESTROYING."""
        assert VALID_STATE_TRANSITIONS[VmState.READY] == {VmState.EXECUTING, VmState.DESTROYING}

    def test_executing_transitions(self) -> None:
        """EXECUTING can transition to READY or DESTROYING."""
        assert VALID_STATE_TRANSITIONS[VmState.EXECUTING] == {VmState.READY, VmState.DESTROYING}

    def test_destroying_transitions(self) -> None:
        """DESTROYING can only transition to DESTROYED."""
        assert VALID_STATE_TRANSITIONS[VmState.DESTROYING] == {VmState.DESTROYED}

    def test_destroyed_is_terminal(self) -> None:
        """DESTROYED is terminal state (no transitions)."""
        assert VALID_STATE_TRANSITIONS[VmState.DESTROYED] == set()

    def test_every_state_can_transition_to_destroying(self) -> None:
        """All non-terminal states can transition to DESTROYING (error handling)."""
        non_terminal = [s for s in VmState if s not in (VmState.DESTROYING, VmState.DESTROYED)]
        for state in non_terminal:
            assert VmState.DESTROYING in VALID_STATE_TRANSITIONS[state], (
                f"State {state} should be able to transition to DESTROYING"
            )


# ============================================================================
# Unit Tests - Platform Detection
# ============================================================================


class TestKvmDetection:
    """Tests for KVM availability detection."""

    def test_kvm_detection_runs(self) -> None:
        """_check_kvm_available returns a boolean."""
        result = _check_kvm_available()
        assert isinstance(result, bool)

    def test_kvm_matches_platform(self) -> None:
        """KVM available on Linux, not on macOS."""
        host_os = detect_host_os()
        kvm_available = _check_kvm_available()

        if host_os == HostOS.MACOS:
            assert kvm_available is False
        # On Linux, KVM might or might not be available


class TestHostOSForVm:
    """Tests for host OS detection in VM context."""

    def test_detect_host_os_for_vm(self) -> None:
        """Host OS detection returns valid value."""
        host_os = detect_host_os()
        assert host_os in (HostOS.LINUX, HostOS.MACOS, HostOS.UNKNOWN)

    def test_current_platform(self) -> None:
        """Current platform is detected correctly."""
        host_os = detect_host_os()
        if sys.platform == "darwin":
            assert host_os == HostOS.MACOS
        elif sys.platform.startswith("linux"):
            assert host_os == HostOS.LINUX


# ============================================================================
# Integration Tests - Require QEMU + Images
# ============================================================================


# Test data for parametrized tests across all image types
IMAGE_TEST_CASES = [
    pytest.param(
        Language.PYTHON,
        "print('hello')",
        "hello",
        id="python",
    ),
    pytest.param(
        Language.JAVASCRIPT,
        "console.log('hello')",
        "hello",
        id="javascript",
    ),
    pytest.param(
        Language.RAW,
        "echo 'hello'",
        "hello",
        id="raw",
    ),
]


class TestVmManagerIntegration:
    """Integration tests for VmManager with real QEMU VMs."""

    async def test_vm_manager_init(self, vm_manager, vm_settings) -> None:
        """VmManager initializes correctly."""
        assert vm_manager.settings == vm_settings

    async def test_create_and_destroy_vm(self, vm_manager) -> None:
        """Create and destroy a VM."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="test-1",
            snapshot_path=None,
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            assert vm.vm_id is not None
            assert vm.state == VmState.READY
        finally:
            await vm_manager.destroy_vm(vm)
            assert vm.state == VmState.DESTROYED

    async def test_vm_execute_code(self, vm_manager) -> None:
        """Execute code in a VM."""
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="test-1",
            snapshot_path=None,
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            result = await vm.execute(
                code="print('hello from vm')",
                language=Language.PYTHON,
                timeout_seconds=30,
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )

            assert result.exit_code == 0
            assert "hello from vm" in result.stdout
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_multiple_vms(self, vm_manager) -> None:
        """Create multiple VMs concurrently."""
        import asyncio

        # Create 2 VMs concurrently
        create_tasks = [
            vm_manager.create_vm(
                language=Language.PYTHON,
                tenant_id="test",
                task_id=f"test-{i}",
                snapshot_path=None,
                memory_mb=256,
                allow_network=False,
                allowed_domains=None,
            )
            for i in range(2)
        ]

        vms = await asyncio.gather(*create_tasks)

        try:
            assert len(vms) == 2
            for vm in vms:
                assert vm.state == VmState.READY
        finally:
            # Destroy all VMs
            destroy_tasks = [vm_manager.destroy_vm(vm) for vm in vms]
            await asyncio.gather(*destroy_tasks)


class TestAllImageTypes:
    """Parametrized tests to verify all image types boot and execute code.

    Each image type (python, javascript, raw) must:
    1. Boot successfully (guest agent responds to ping)
    2. Execute code and return correct output
    """

    async def test_default_uses_hardware_acceleration(self, vm_manager) -> None:
        """Verify default settings (force_emulation=False) use hardware accel when available.

        On macOS: -accel hvf, -cpu host
        On Linux with KVM: -accel kvm, -cpu host
        Without hardware accel: -accel tcg (fallback)

        This test verifies that when hardware acceleration is available,
        we actually use it (not accidentally falling back to TCG).
        """
        from exec_sandbox.vm_manager import _check_hvf_available, _check_kvm_available

        vm = await vm_manager.create_vm(
            language=Language.RAW,
            tenant_id="test-hwaccel",
            task_id="verify-hwaccel",
            snapshot_path=None,
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            # Find this VM's QEMU process
            # On Linux, we read /proc/*/cmdline directly to avoid ps aux truncation
            # On macOS, we use ps aux since /proc doesn't exist
            import platform

            if platform.system() == "Linux":
                # Read /proc/*/cmdline for each process - this gives full command line
                # without truncation. The cmdline uses NUL as separator, we convert to spaces.
                proc = await asyncio.create_subprocess_exec(
                    "bash",
                    "-c",
                    f"for f in /proc/[0-9]*/cmdline; do cat \"$f\" 2>/dev/null | tr '\\0' ' '; echo; done | grep -E 'qemu.*{vm.vm_id}|{vm.vm_id}.*qemu'",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                ps_output = stdout.decode()
            else:
                # macOS: use ps aux (no truncation issues on macOS)
                proc = await asyncio.create_subprocess_exec(
                    "ps",
                    "aux",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                ps_output = stdout.decode()

            accel_found = None
            cpu_found = None
            qemu_line_found = False

            for line in ps_output.split("\n"):
                if vm.vm_id in line and "qemu" in line:
                    qemu_line_found = True
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "-accel" and i + 1 < len(parts):
                            accel_found = parts[i + 1]
                        if p == "-cpu" and i + 1 < len(parts):
                            cpu_found = parts[i + 1]
                    break

            # Verify we found the QEMU process
            assert qemu_line_found, (
                f"Could not find QEMU process for VM {vm.vm_id}\nps output sample:\n{ps_output[:2000]}"
            )
            assert accel_found is not None, f"Could not find -accel argument in QEMU command line:\n{ps_output[:2000]}"
            assert cpu_found is not None, f"Could not find -cpu argument in QEMU command line:\n{ps_output[:2000]}"

            # Check what hardware acceleration should be available
            # Note: HVF is macOS-only, KVM is Linux-only
            kvm_available = _check_kvm_available()
            hvf_available = await _check_hvf_available()

            if hvf_available:
                # macOS with HVF available should use HVF (Hypervisor.framework)
                assert accel_found == "hvf", f"Expected HVF on macOS, got: -accel {accel_found}"
                assert cpu_found == "host", f"Expected '-cpu host' with HVF, got: -cpu {cpu_found}"
            elif kvm_available:
                # Linux with KVM should use KVM
                assert accel_found == "kvm", f"Expected KVM on Linux with KVM available, got: -accel {accel_found}"
                assert cpu_found == "host", f"Expected '-cpu host' with KVM, got: -cpu {cpu_found}"
            else:
                # Fallback to TCG (this is expected in some CI environments)
                assert accel_found.startswith("tcg"), (
                    f"Expected TCG fallback without hardware accel, got: -accel {accel_found}"
                )
        finally:
            await vm_manager.destroy_vm(vm)

    @pytest.mark.parametrize("language,code,expected_output", IMAGE_TEST_CASES)
    async def test_vm_health_check_all_images(
        self,
        vm_manager,
        language: Language,
        code: str,
        expected_output: str,
    ) -> None:
        """VM boots and guest agent responds for all image types."""
        vm = await vm_manager.create_vm(
            language=language,
            tenant_id="test",
            task_id=f"health-check-{language.value}",
            snapshot_path=None,
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            # VM reaching READY state means:
            # 1. QEMU started successfully
            # 2. Kernel booted
            # 3. Guest agent started
            # 4. Guest agent responded to ping with version
            assert vm.vm_id is not None
            assert vm.state == VmState.READY
        finally:
            await vm_manager.destroy_vm(vm)
            assert vm.state == VmState.DESTROYED

    @pytest.mark.parametrize("language,code,expected_output", IMAGE_TEST_CASES)
    async def test_vm_execute_code_all_images(
        self,
        vm_manager,
        language: Language,
        code: str,
        expected_output: str,
    ) -> None:
        """VM executes code and returns correct output for all image types."""
        vm = await vm_manager.create_vm(
            language=language,
            tenant_id="test",
            task_id=f"execute-{language.value}",
            snapshot_path=None,
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            result = await vm.execute(
                code=code,
                language=language,
                timeout_seconds=30,
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )

            assert result.exit_code == 0, f"Exit code {result.exit_code}, stderr: {result.stderr}"
            assert expected_output in result.stdout, f"Expected '{expected_output}' in stdout: {result.stdout}"
        finally:
            await vm_manager.destroy_vm(vm)


class TestEmulationMode:
    """Tests with forced software emulation (TCG) to verify emulation code paths.

    These tests use force_emulation=True to bypass KVM/HVF hardware acceleration,
    ensuring the TCG emulation path works correctly. Useful for catching issues
    that only appear in CI environments without hardware virtualization support.
    """

    async def test_emulation_uses_tcg_not_hvf(self, emulation_vm_manager) -> None:
        """Verify force_emulation=True actually uses TCG, not hardware acceleration.

        This test inspects the running QEMU process to verify:
        1. -accel is set to 'tcg' (software emulation), not 'hvf' or 'kvm'
        2. -cpu is NOT 'host' (should be an emulated CPU like 'cortex-a57' or 'qemu64')
        """
        vm = await emulation_vm_manager.create_vm(
            language=Language.RAW,
            tenant_id="test-emulation",
            task_id="verify-tcg",
            snapshot_path=None,
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            # Find this VM's QEMU process
            # On Linux, we read /proc/*/cmdline directly to avoid ps aux truncation
            # On macOS, we use ps aux since /proc doesn't exist
            import platform

            if platform.system() == "Linux":
                # Read /proc/*/cmdline for each process - this gives full command line
                # without truncation. The cmdline uses NUL as separator, we convert to spaces.
                proc = await asyncio.create_subprocess_exec(
                    "bash",
                    "-c",
                    f"for f in /proc/[0-9]*/cmdline; do cat \"$f\" 2>/dev/null | tr '\\0' ' '; echo; done | grep -E 'qemu.*{vm.vm_id}|{vm.vm_id}.*qemu'",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                ps_output = stdout.decode()
            else:
                # macOS: use ps aux (no truncation issues on macOS)
                proc = await asyncio.create_subprocess_exec(
                    "ps",
                    "aux",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                ps_output = stdout.decode()

            accel_found = None
            cpu_found = None
            qemu_line_found = False

            for line in ps_output.split("\n"):
                if vm.vm_id in line and "qemu" in line:
                    qemu_line_found = True
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "-accel" and i + 1 < len(parts):
                            accel_found = parts[i + 1]
                        if p == "-cpu" and i + 1 < len(parts):
                            cpu_found = parts[i + 1]
                    break

            # Verify we found the QEMU process
            assert qemu_line_found, (
                f"Could not find QEMU process for VM {vm.vm_id}\nps output sample:\n{ps_output[:2000]}"
            )
            assert accel_found is not None, f"Could not find -accel argument in QEMU command line:\n{ps_output[:2000]}"
            assert cpu_found is not None, f"Could not find -cpu argument in QEMU command line:\n{ps_output[:2000]}"

            # With force_emulation=True, MUST use TCG, MUST NOT use HVF/KVM
            assert accel_found.startswith("tcg"), (
                f"Expected TCG emulation with force_emulation=True, got: -accel {accel_found}"
            )
            assert accel_found not in ("hvf", "kvm"), (
                f"force_emulation=True should NOT use hardware acceleration, got: -accel {accel_found}"
            )
            # CPU should be emulated (cortex-a57 for ARM, qemu64 for x86), NOT 'host'
            assert cpu_found != "host", f"Expected emulated CPU with force_emulation=True, got: -cpu {cpu_found}"
        finally:
            await emulation_vm_manager.destroy_vm(vm)

    @pytest.mark.parametrize("language,code,expected_output", IMAGE_TEST_CASES)
    async def test_vm_boot_with_emulation(
        self,
        emulation_vm_manager,
        language: Language,
        code: str,
        expected_output: str,
    ) -> None:
        """VM boots successfully with forced software emulation."""
        vm = await emulation_vm_manager.create_vm(
            language=language,
            tenant_id="test-emulation",
            task_id=f"emulation-boot-{language.value}",
            snapshot_path=None,
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            assert vm.vm_id is not None
            assert vm.state == VmState.READY
        finally:
            await emulation_vm_manager.destroy_vm(vm)
            assert vm.state == VmState.DESTROYED

    @pytest.mark.parametrize("language,code,expected_output", IMAGE_TEST_CASES)
    async def test_vm_execute_with_emulation(
        self,
        emulation_vm_manager,
        language: Language,
        code: str,
        expected_output: str,
    ) -> None:
        """VM executes code correctly with forced software emulation."""
        vm = await emulation_vm_manager.create_vm(
            language=language,
            tenant_id="test-emulation",
            task_id=f"emulation-exec-{language.value}",
            snapshot_path=None,
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            result = await vm.execute(
                code=code,
                language=language,
                timeout_seconds=120,  # Longer timeout for TCG emulation
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )

            assert result.exit_code == 0, f"Exit code {result.exit_code}, stderr: {result.stderr}"
            assert expected_output in result.stdout, f"Expected '{expected_output}' in stdout: {result.stdout}"
        finally:
            await emulation_vm_manager.destroy_vm(vm)

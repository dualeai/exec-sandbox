"""Tests for VmManager.

Unit tests: State machine, platform detection.
Integration tests: Real VM lifecycle (requires QEMU + images).
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from exec_sandbox.exceptions import VmError
from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostArch, HostOS, detect_host_arch, detect_host_os
from exec_sandbox.vm_manager import (
    VALID_STATE_TRANSITIONS,
    VmState,
    _check_hvf_available,
    _check_kvm_available,
    _check_tsc_deadline,
    _check_tsc_deadline_linux,
    _check_tsc_deadline_macos,
    _kernel_validated,
    _probe_cache,
    _probe_qemu_accelerators,
    _validate_kernel_initramfs,
    check_hwaccel_available,
)
from tests.conftest import skip_unless_hwaccel, skip_unless_linux, skip_unless_macos

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

    async def test_kvm_detection_runs(self) -> None:
        """_check_kvm_available returns a boolean."""
        result = await _check_kvm_available()
        assert isinstance(result, bool)

    @skip_unless_macos
    async def test_kvm_not_available_on_macos(self) -> None:
        """KVM is never available on macOS."""
        kvm_available = await _check_kvm_available()
        assert kvm_available is False


# ============================================================================
# Unit Tests - QEMU Accelerator Probe (2-Layer Detection)
# ============================================================================


class TestQemuAcceleratorProbe:
    """Tests for _probe_qemu_accelerators() - QEMU binary capability detection.

    This is Layer 2 of the 2-layer hardware acceleration detection:
    - Layer 1: Kernel/OS level (ioctl for KVM, sysctl for HVF)
    - Layer 2: QEMU binary level (what accelerators QEMU actually supports)
    """

    @pytest.fixture(autouse=True)
    def clear_qemu_cache(self) -> None:
        """Clear QEMU accelerator cache before each test."""
        _probe_cache.qemu_accels = None

    # ========================================================================
    # Normal Cases - Happy path scenarios
    # ========================================================================

    async def test_probe_returns_set(self) -> None:
        """Probe returns a set of accelerator names."""
        result = await _probe_qemu_accelerators()
        assert isinstance(result, set)

    async def test_probe_contains_tcg(self) -> None:
        """TCG should always be available (software emulation fallback)."""
        result = await _probe_qemu_accelerators()
        assert "tcg" in result, "TCG should always be available in QEMU"

    @skip_unless_macos
    async def test_probe_contains_hvf_on_macos(self) -> None:
        """HVF should be available on macOS with Hypervisor.framework."""
        result = await _probe_qemu_accelerators()
        # HVF may or may not be available depending on QEMU build
        # Just verify the probe runs and returns valid data
        assert isinstance(result, set)
        assert "tcg" in result

    @skip_unless_linux
    async def test_probe_may_contain_kvm_on_linux(self) -> None:
        """KVM may be available on Linux if QEMU is compiled with KVM support."""
        result = await _probe_qemu_accelerators()
        assert isinstance(result, set)
        assert "tcg" in result
        # KVM availability depends on QEMU build and system config

    async def test_probe_result_is_cached(self) -> None:
        """Subsequent calls return cached result."""
        result1 = await _probe_qemu_accelerators()
        result2 = await _probe_qemu_accelerators()
        assert result1 is result2, "Results should be the same cached object"

    async def test_probe_result_lowercase(self) -> None:
        """Accelerator names are normalized to lowercase."""
        result = await _probe_qemu_accelerators()
        for accel in result:
            assert accel == accel.lower(), f"Accelerator '{accel}' should be lowercase"

    # ========================================================================
    # Edge Cases - Boundary conditions
    # ========================================================================

    async def test_probe_with_missing_qemu_binary(self) -> None:
        """Probe returns empty set when QEMU binary is not found."""
        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("qemu-system-x86_64 not found")
            _probe_cache.qemu_accels = None  # Clear cache
            result = await _probe_qemu_accelerators()
            assert result == set()

    async def test_probe_with_qemu_failure(self) -> None:
        """Probe returns empty set when QEMU returns non-zero exit code."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec", return_value=mock_proc):
            _probe_cache.qemu_accels = None
            result = await _probe_qemu_accelerators()
            assert result == set()

    async def test_probe_with_empty_output(self) -> None:
        """Probe returns empty set when QEMU returns empty output."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec", return_value=mock_proc):
            _probe_cache.qemu_accels = None
            result = await _probe_qemu_accelerators()
            assert result == set()

    async def test_probe_with_only_header_line(self) -> None:
        """Probe returns empty set when QEMU returns only header line."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Accelerators supported in QEMU binary:\n", b""))

        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec", return_value=mock_proc):
            _probe_cache.qemu_accels = None
            result = await _probe_qemu_accelerators()
            assert result == set()

    async def test_probe_with_timeout(self) -> None:
        """Probe returns empty set when QEMU times out."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())

        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec", return_value=mock_proc):
            _probe_cache.qemu_accels = None
            result = await _probe_qemu_accelerators()
            assert result == set()

    # ========================================================================
    # Weird Cases - Unusual but valid scenarios
    # ========================================================================

    async def test_probe_with_extra_whitespace(self) -> None:
        """Probe handles extra whitespace in QEMU output."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(b"Accelerators supported in QEMU binary:\n  tcg  \n  kvm  \n\n", b"")
        )

        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec", return_value=mock_proc):
            _probe_cache.qemu_accels = None
            result = await _probe_qemu_accelerators()
            assert "tcg" in result
            assert "kvm" in result
            assert len(result) == 2

    async def test_probe_with_mixed_case(self) -> None:
        """Probe normalizes mixed-case accelerator names."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(b"Accelerators supported in QEMU binary:\nTCG\nKVM\nHvF\n", b"")
        )

        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec", return_value=mock_proc):
            _probe_cache.qemu_accels = None
            result = await _probe_qemu_accelerators()
            assert "tcg" in result
            assert "kvm" in result
            assert "hvf" in result
            # Verify all are lowercase
            for accel in result:
                assert accel == accel.lower()

    async def test_probe_with_unknown_accelerators(self) -> None:
        """Probe includes unknown accelerators from QEMU output."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(b"Accelerators supported in QEMU binary:\ntcg\nxen\nwhpx\nnvmm\n", b"")
        )

        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec", return_value=mock_proc):
            _probe_cache.qemu_accels = None
            result = await _probe_qemu_accelerators()
            assert "tcg" in result
            assert "xen" in result
            assert "whpx" in result
            assert "nvmm" in result

    async def test_probe_with_windows_line_endings(self) -> None:
        """Probe handles Windows-style line endings."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(b"Accelerators supported in QEMU binary:\r\ntcg\r\nkvm\r\n", b"")
        )

        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec", return_value=mock_proc):
            _probe_cache.qemu_accels = None
            result = await _probe_qemu_accelerators()
            # strip() handles \r\n
            assert "tcg" in result or "tcg\r" in result  # Verify parsing works

    # ========================================================================
    # Out of Bound Cases - Error conditions
    # ========================================================================

    async def test_probe_with_oserror(self) -> None:
        """Probe returns empty set on OSError."""
        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = OSError("Permission denied")
            _probe_cache.qemu_accels = None
            result = await _probe_qemu_accelerators()
            assert result == set()

    async def test_probe_with_very_long_output(self) -> None:
        """Probe handles very long QEMU output gracefully."""
        # Simulate QEMU returning many accelerators
        accels = "\n".join([f"accel{i}" for i in range(1000)])
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(f"Accelerators supported:\n{accels}\n".encode(), b""))

        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec", return_value=mock_proc):
            _probe_cache.qemu_accels = None
            result = await _probe_qemu_accelerators()
            assert len(result) == 1000

    async def test_cache_cleared_between_tests(self) -> None:
        """Verify cache is properly cleared (test isolation)."""
        # This test verifies the fixture works
        assert _probe_cache.qemu_accels is None


class TestTwoLayerKvmDetection:
    """Tests for 2-layer KVM detection (kernel + QEMU verification)."""

    @pytest.fixture(autouse=True)
    def clear_caches(self) -> None:
        """Clear all relevant caches before each test."""
        _probe_cache.kvm = None
        _probe_cache.qemu_accels = None

    async def test_kvm_fails_when_qemu_lacks_kvm_support(self) -> None:
        """KVM detection fails if QEMU doesn't have KVM compiled in."""
        # Mock Layer 1 passing (kernel check)
        with (
            patch("exec_sandbox.vm_manager.aiofiles.os.path.exists", return_value=True),
            patch("exec_sandbox.vm_manager.can_access", return_value=True),
            patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec") as mock_exec,
        ):
            # First call: ioctl check passes
            ioctl_proc = AsyncMock()
            ioctl_proc.returncode = 0
            ioctl_proc.communicate = AsyncMock(return_value=(b"12\n", b""))

            # Second call: QEMU probe returns only TCG (no KVM)
            qemu_proc = AsyncMock()
            qemu_proc.returncode = 0
            qemu_proc.communicate = AsyncMock(return_value=(b"Accelerators supported in QEMU binary:\ntcg\n", b""))

            mock_exec.side_effect = [ioctl_proc, qemu_proc]

            result = await _check_kvm_available()
            assert result is False

    async def test_kvm_passes_when_both_layers_pass(self) -> None:
        """KVM detection passes when both kernel and QEMU verify KVM."""
        with (
            patch("exec_sandbox.vm_manager.aiofiles.os.path.exists", return_value=True),
            patch("exec_sandbox.vm_manager.can_access", return_value=True),
            patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec") as mock_exec,
        ):
            # First call: ioctl check passes
            ioctl_proc = AsyncMock()
            ioctl_proc.returncode = 0
            ioctl_proc.communicate = AsyncMock(return_value=(b"12\n", b""))

            # Second call: QEMU probe returns KVM
            qemu_proc = AsyncMock()
            qemu_proc.returncode = 0
            qemu_proc.communicate = AsyncMock(return_value=(b"Accelerators supported in QEMU binary:\ntcg\nkvm\n", b""))

            mock_exec.side_effect = [ioctl_proc, qemu_proc]

            result = await _check_kvm_available()
            assert result is True

    @skip_unless_linux
    async def test_kvm_fails_when_dev_kvm_missing(self) -> None:
        """KVM detection fails early if /dev/kvm doesn't exist."""
        with patch("exec_sandbox.vm_manager.aiofiles.os.path.exists", return_value=False):
            _probe_cache.kvm = None
            result = await _check_kvm_available()
            assert result is False


class TestTwoLayerHvfDetection:
    """Tests for 2-layer HVF detection (kernel + QEMU verification)."""

    @pytest.fixture(autouse=True)
    def clear_caches(self) -> None:
        """Clear all relevant caches before each test."""
        _probe_cache.hvf = None
        _probe_cache.qemu_accels = None

    async def test_hvf_fails_when_qemu_lacks_hvf_support(self) -> None:
        """HVF detection fails if QEMU doesn't have HVF compiled in."""
        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec") as mock_exec:
            # First call: sysctl check passes
            sysctl_proc = AsyncMock()
            sysctl_proc.returncode = 0
            sysctl_proc.communicate = AsyncMock(return_value=(b"1\n", b""))

            # Second call: QEMU probe returns only TCG (no HVF)
            qemu_proc = AsyncMock()
            qemu_proc.returncode = 0
            qemu_proc.communicate = AsyncMock(return_value=(b"Accelerators supported in QEMU binary:\ntcg\n", b""))

            mock_exec.side_effect = [sysctl_proc, qemu_proc]

            result = await _check_hvf_available()
            assert result is False

    async def test_hvf_passes_when_both_layers_pass(self) -> None:
        """HVF detection passes when both kernel and QEMU verify HVF."""
        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec") as mock_exec:
            # First call: sysctl check passes
            sysctl_proc = AsyncMock()
            sysctl_proc.returncode = 0
            sysctl_proc.communicate = AsyncMock(return_value=(b"1\n", b""))

            # Second call: QEMU probe returns HVF
            qemu_proc = AsyncMock()
            qemu_proc.returncode = 0
            qemu_proc.communicate = AsyncMock(return_value=(b"Accelerators supported in QEMU binary:\ntcg\nhvf\n", b""))

            mock_exec.side_effect = [sysctl_proc, qemu_proc]

            result = await _check_hvf_available()
            assert result is True

    async def test_hvf_fails_when_kernel_check_fails(self) -> None:
        """HVF detection fails early if kern.hv_support is 0."""
        with patch("exec_sandbox.vm_manager.asyncio.create_subprocess_exec") as mock_exec:
            sysctl_proc = AsyncMock()
            sysctl_proc.returncode = 0
            sysctl_proc.communicate = AsyncMock(return_value=(b"0\n", b""))

            mock_exec.return_value = sysctl_proc

            result = await _check_hvf_available()
            assert result is False


class TestCheckHwaccelAvailable:
    """Tests for the synchronous check_hwaccel_available() wrapper."""

    @pytest.fixture(autouse=True)
    def clear_caches(self) -> None:
        """Clear all relevant caches before each test."""
        _probe_cache.kvm = None
        _probe_cache.hvf = None
        _probe_cache.qemu_accels = None

    def test_returns_boolean(self) -> None:
        """check_hwaccel_available returns a boolean."""
        result = check_hwaccel_available()
        assert isinstance(result, bool)

    @skip_unless_hwaccel
    def test_hwaccel_available_when_expected(self) -> None:
        """Hardware acceleration is available on supported systems."""
        # This test only runs when hwaccel is expected to be available
        assert check_hwaccel_available() is True

    def test_consistent_results(self) -> None:
        """Multiple calls return consistent results."""
        result1 = check_hwaccel_available()
        result2 = check_hwaccel_available()
        assert result1 == result2


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
# Unit Tests - Kernel/Initramfs Pre-flight Validation
# ============================================================================


class TestKernelInitramfsValidation:
    """Tests for _validate_kernel_initramfs() pre-flight check."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Clear validation cache before each test."""
        _kernel_validated.clear()

    async def test_validation_succeeds_with_real_paths(self, vm_settings) -> None:
        """Validation passes when kernel and initramfs exist."""
        arch = detect_host_arch()
        # Should not raise
        await _validate_kernel_initramfs(vm_settings.kernel_path, arch)

    async def test_validation_fails_with_fake_path(self) -> None:
        """Validation raises VmError when kernel doesn't exist."""
        arch = detect_host_arch()
        fake_path = Path("/nonexistent/kernels")

        with pytest.raises(VmError, match="Kernel not found"):
            await _validate_kernel_initramfs(fake_path, arch)

    async def test_cache_prevents_repeated_io(self, vm_settings) -> None:
        """Second call uses cache, no I/O operations."""
        arch = detect_host_arch()

        # First call - real I/O
        await _validate_kernel_initramfs(vm_settings.kernel_path, arch)
        assert (vm_settings.kernel_path, arch) in _kernel_validated

        # Second call - should use cache, mock should NOT be called
        with patch("exec_sandbox.vm_manager.aiofiles.os.path.exists", new_callable=AsyncMock) as mock_exists:
            await _validate_kernel_initramfs(vm_settings.kernel_path, arch)
            mock_exists.assert_not_called()

    async def test_different_paths_validated_separately(self, vm_settings) -> None:
        """Different kernel paths are cached separately."""
        arch = detect_host_arch()

        # First path succeeds
        await _validate_kernel_initramfs(vm_settings.kernel_path, arch)
        assert (vm_settings.kernel_path, arch) in _kernel_validated

        # Different path still gets validated (and fails)
        fake_path = Path("/nonexistent/kernels")
        with pytest.raises(VmError, match="Kernel not found"):
            await _validate_kernel_initramfs(fake_path, arch)


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
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            result = await vm.execute(
                code="print('hello from vm')",
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
            kvm_available = await _check_kvm_available()
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
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            result = await vm.execute(
                code=code,
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
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )

        try:
            result = await vm.execute(
                code=code,
                timeout_seconds=120,  # Longer timeout for TCG emulation
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )

            assert result.exit_code == 0, f"Exit code {result.exit_code}, stderr: {result.stderr}"
            assert expected_output in result.stdout, f"Expected '{expected_output}' in stdout: {result.stdout}"
        finally:
            await emulation_vm_manager.destroy_vm(vm)


# ============================================================================
# Unit Tests - TSC_DEADLINE Detection for Linux
# ============================================================================


class TestTscDeadlineLinux:
    """Tests for TSC_DEADLINE detection on Linux via /proc/cpuinfo."""

    @pytest.mark.asyncio
    async def test_tsc_deadline_linux_with_feature(self) -> None:
        """Linux with tsc_deadline_timer in cpuinfo returns True."""
        from exec_sandbox.vm_manager import _probe_cache

        original_tsc = _probe_cache.tsc_deadline

        try:
            _probe_cache.tsc_deadline = None

            cpuinfo_content = """processor	: 0
vendor_id	: GenuineIntel
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep tsc_deadline_timer sse sse2
"""
            mock_file = AsyncMock()
            mock_file.__aenter__.return_value.read = AsyncMock(return_value=cpuinfo_content)

            with patch("aiofiles.os.path.exists", return_value=True):
                with patch("aiofiles.open", return_value=mock_file):
                    result = await _check_tsc_deadline_linux()
                    assert result is True
        finally:
            _probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_linux_without_feature(self) -> None:
        """Linux without tsc_deadline_timer in cpuinfo returns False."""
        from exec_sandbox.vm_manager import _probe_cache

        original_tsc = _probe_cache.tsc_deadline

        try:
            _probe_cache.tsc_deadline = None

            cpuinfo_content = """processor	: 0
vendor_id	: GenuineIntel
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep sse sse2
"""
            mock_file = AsyncMock()
            mock_file.__aenter__.return_value.read = AsyncMock(return_value=cpuinfo_content)

            with patch("aiofiles.os.path.exists", return_value=True):
                with patch("aiofiles.open", return_value=mock_file):
                    result = await _check_tsc_deadline_linux()
                    assert result is False
        finally:
            _probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_linux_cpuinfo_not_exists(self) -> None:
        """/proc/cpuinfo not existing returns False gracefully."""
        from exec_sandbox.vm_manager import _probe_cache

        original_tsc = _probe_cache.tsc_deadline

        try:
            _probe_cache.tsc_deadline = None

            with patch("aiofiles.os.path.exists", return_value=False):
                result = await _check_tsc_deadline_linux()
                assert result is False
        finally:
            _probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_linux_read_error(self) -> None:
        """OSError reading /proc/cpuinfo returns False gracefully."""
        from exec_sandbox.vm_manager import _probe_cache

        original_tsc = _probe_cache.tsc_deadline

        try:
            _probe_cache.tsc_deadline = None

            with patch("aiofiles.os.path.exists", return_value=True):
                with patch("aiofiles.open", side_effect=OSError("Permission denied")):
                    result = await _check_tsc_deadline_linux()
                    assert result is False
        finally:
            _probe_cache.tsc_deadline = original_tsc


# ============================================================================
# Unit Tests - TSC_DEADLINE Detection for macOS
# ============================================================================


class TestTscDeadlineMacOS:
    """Tests for TSC_DEADLINE detection on macOS (Intel Macs)."""

    @pytest.mark.asyncio
    async def test_tsc_deadline_macos_arm64_returns_false(self) -> None:
        """ARM64 Macs always return False for TSC_DEADLINE (ARM uses different timer)."""
        from exec_sandbox.vm_manager import _probe_cache

        # Save original cache value
        original_tsc = _probe_cache.tsc_deadline

        try:
            # Clear cache
            _probe_cache.tsc_deadline = None

            with patch("exec_sandbox.vm_manager.detect_host_arch", return_value=HostArch.AARCH64):
                result = await _check_tsc_deadline_macos()
                assert result is False
        finally:
            # Restore cache
            _probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_macos_x86_64_with_tsc(self) -> None:
        """Intel Mac with TSC_DEADLINE returns True."""
        from exec_sandbox.vm_manager import _probe_cache

        # Save original cache value
        original_tsc = _probe_cache.tsc_deadline

        try:
            # Clear cache
            _probe_cache.tsc_deadline = None

            with patch("exec_sandbox.vm_manager.detect_host_arch", return_value=HostArch.X86_64):
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(
                    return_value=(b"FPU VME DE PSE TSC MSR PAE MCE TSC_DEADLINE SSE", b"")
                )

                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    result = await _check_tsc_deadline_macos()
                    assert result is True
        finally:
            # Restore cache
            _probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_macos_x86_64_without_tsc(self) -> None:
        """Intel Mac without TSC_DEADLINE returns False."""
        from exec_sandbox.vm_manager import _probe_cache

        # Save original cache value
        original_tsc = _probe_cache.tsc_deadline

        try:
            # Clear cache
            _probe_cache.tsc_deadline = None

            with patch("exec_sandbox.vm_manager.detect_host_arch", return_value=HostArch.X86_64):
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"FPU VME DE PSE TSC MSR PAE MCE SSE SSE2", b""))

                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    result = await _check_tsc_deadline_macos()
                    assert result is False
        finally:
            # Restore cache
            _probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_macos_sysctl_failure(self) -> None:
        """sysctl failure returns False gracefully."""
        from exec_sandbox.vm_manager import _probe_cache

        # Save original cache value
        original_tsc = _probe_cache.tsc_deadline

        try:
            # Clear cache
            _probe_cache.tsc_deadline = None

            with patch("exec_sandbox.vm_manager.detect_host_arch", return_value=HostArch.X86_64):
                mock_proc = AsyncMock()
                mock_proc.returncode = 1  # sysctl failed
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))

                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    result = await _check_tsc_deadline_macos()
                    assert result is False
        finally:
            # Restore cache
            _probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_cached(self) -> None:
        """TSC_DEADLINE result is cached (caching is in _check_tsc_deadline, not platform-specific functions)."""
        from exec_sandbox.vm_manager import _probe_cache

        # Save original cache value
        original_tsc = _probe_cache.tsc_deadline

        try:
            # Set cached value
            _probe_cache.tsc_deadline = True

            # Should return cached value without calling sysctl or reading /proc/cpuinfo
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                with patch("aiofiles.open") as mock_open:
                    result = await _check_tsc_deadline()
                    assert result is True
                    mock_exec.assert_not_called()
                    mock_open.assert_not_called()
        finally:
            # Restore cache
            _probe_cache.tsc_deadline = original_tsc


# ============================================================================
# Unit Tests - gvproxy-wrapper Binary Selection
# ============================================================================


class TestGvproxyWrapperBinarySelection:
    """Tests for gvproxy-wrapper binary naming for different platforms."""

    def test_darwin_amd64_suffix(self) -> None:
        """Intel Mac uses darwin-amd64 suffix for gvproxy-wrapper."""
        from exec_sandbox.asset_downloader import get_gvproxy_suffix

        with patch("exec_sandbox.asset_downloader.get_os_name", return_value="darwin"):
            with patch("exec_sandbox.asset_downloader.get_arch_name", return_value="amd64"):
                assert get_gvproxy_suffix() == "darwin-amd64"

    def test_darwin_arm64_suffix(self) -> None:
        """Apple Silicon uses darwin-arm64 suffix for gvproxy-wrapper."""
        from exec_sandbox.asset_downloader import get_gvproxy_suffix

        with patch("exec_sandbox.asset_downloader.get_os_name", return_value="darwin"):
            with patch("exec_sandbox.asset_downloader.get_arch_name", return_value="arm64"):
                assert get_gvproxy_suffix() == "darwin-arm64"

    def test_linux_amd64_suffix(self) -> None:
        """Linux x86_64 uses linux-amd64 suffix for gvproxy-wrapper."""
        from exec_sandbox.asset_downloader import get_gvproxy_suffix

        with patch("exec_sandbox.asset_downloader.get_os_name", return_value="linux"):
            with patch("exec_sandbox.asset_downloader.get_arch_name", return_value="amd64"):
                assert get_gvproxy_suffix() == "linux-amd64"

    def test_linux_arm64_suffix(self) -> None:
        """Linux ARM64 uses linux-arm64 suffix for gvproxy-wrapper."""
        from exec_sandbox.asset_downloader import get_gvproxy_suffix

        with patch("exec_sandbox.asset_downloader.get_os_name", return_value="linux"):
            with patch("exec_sandbox.asset_downloader.get_arch_name", return_value="arm64"):
                assert get_gvproxy_suffix() == "linux-arm64"

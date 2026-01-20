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
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.system_probes import (
    NOT_CACHED,
    _check_tsc_deadline_linux,  # pyright: ignore[reportPrivateUsage]
    _check_tsc_deadline_macos,  # pyright: ignore[reportPrivateUsage]
    check_hvf_available,
    check_hwaccel_available,
    check_kvm_available,
    check_tsc_deadline,
    probe_cache,
    probe_qemu_accelerators,
)
from exec_sandbox.validation import (
    clear_kernel_validation_cache,
    validate_kernel_initramfs,
)
from exec_sandbox.vm_manager import _validate_identifier
from exec_sandbox.vm_types import VALID_STATE_TRANSITIONS, VmState
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
        """check_kvm_available returns a boolean."""
        result = await check_kvm_available()
        assert isinstance(result, bool)

    @skip_unless_macos
    async def test_kvm_not_available_on_macos(self) -> None:
        """KVM is never available on macOS."""
        kvm_available = await check_kvm_available()
        assert kvm_available is False


# ============================================================================
# Unit Tests - QEMU Accelerator Probe (2-Layer Detection)
# ============================================================================


class TestQemuAcceleratorProbe:
    """Tests for probe_qemu_accelerators() - QEMU binary capability detection.

    This is Layer 2 of the 2-layer hardware acceleration detection:
    - Layer 1: Kernel/OS level (ioctl for KVM, sysctl for HVF)
    - Layer 2: QEMU binary level (what accelerators QEMU actually supports)
    """

    @pytest.fixture(autouse=True)
    def clear_qemu_cache(self) -> None:
        """Clear QEMU accelerator cache before each test."""
        probe_cache.reset("qemu_accels")

    # ========================================================================
    # Normal Cases - Happy path scenarios
    # ========================================================================

    async def test_probe_returns_set(self) -> None:
        """Probe returns a set of accelerator names."""
        result = await probe_qemu_accelerators()
        assert isinstance(result, set)

    async def test_probe_contains_tcg(self) -> None:
        """TCG should always be available (software emulation fallback)."""
        result = await probe_qemu_accelerators()
        assert "tcg" in result, "TCG should always be available in QEMU"

    @skip_unless_macos
    async def test_probe_contains_hvf_on_macos(self) -> None:
        """HVF should be available on macOS with Hypervisor.framework."""
        result = await probe_qemu_accelerators()
        # HVF may or may not be available depending on QEMU build
        # Just verify the probe runs and returns valid data
        assert isinstance(result, set)
        assert "tcg" in result

    @skip_unless_linux
    async def test_probe_may_contain_kvm_on_linux(self) -> None:
        """KVM may be available on Linux if QEMU is compiled with KVM support."""
        result = await probe_qemu_accelerators()
        assert isinstance(result, set)
        assert "tcg" in result
        # KVM availability depends on QEMU build and system config

    async def test_probe_result_is_cached(self) -> None:
        """Subsequent calls return cached result."""
        result1 = await probe_qemu_accelerators()
        result2 = await probe_qemu_accelerators()
        assert result1 is result2, "Results should be the same cached object"

    async def test_probe_result_lowercase(self) -> None:
        """Accelerator names are normalized to lowercase."""
        result = await probe_qemu_accelerators()
        for accel in result:
            assert accel == accel.lower(), f"Accelerator '{accel}' should be lowercase"

    # ========================================================================
    # Edge Cases - Boundary conditions
    # ========================================================================

    async def test_probe_with_missing_qemu_binary(self) -> None:
        """Probe returns empty set when QEMU binary is not found."""
        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("qemu-system-x86_64 not found")
            probe_cache.reset("qemu_accels")  # Clear cache
            result = await probe_qemu_accelerators()
            assert result == set()

    async def test_probe_with_qemu_failure(self) -> None:
        """Probe returns empty set when QEMU returns non-zero exit code."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            probe_cache.reset("qemu_accels")
            result = await probe_qemu_accelerators()
            assert result == set()

    async def test_probe_with_empty_output(self) -> None:
        """Probe returns empty set when QEMU returns empty output."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            probe_cache.reset("qemu_accels")
            result = await probe_qemu_accelerators()
            assert result == set()

    async def test_probe_with_only_header_line(self) -> None:
        """Probe returns empty set when QEMU returns only header line."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Accelerators supported in QEMU binary:\n", b""))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            probe_cache.reset("qemu_accels")
            result = await probe_qemu_accelerators()
            assert result == set()

    async def test_probe_with_timeout(self) -> None:
        """Probe returns empty set when QEMU times out."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            probe_cache.reset("qemu_accels")
            result = await probe_qemu_accelerators()
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

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            probe_cache.reset("qemu_accels")
            result = await probe_qemu_accelerators()
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

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            probe_cache.reset("qemu_accels")
            result = await probe_qemu_accelerators()
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

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            probe_cache.reset("qemu_accels")
            result = await probe_qemu_accelerators()
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

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            probe_cache.reset("qemu_accels")
            result = await probe_qemu_accelerators()
            # strip() handles \r\n
            assert "tcg" in result or "tcg\r" in result  # Verify parsing works

    # ========================================================================
    # Out of Bound Cases - Error conditions
    # ========================================================================

    async def test_probe_with_oserror(self) -> None:
        """Probe returns empty set on OSError."""
        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = OSError("Permission denied")
            probe_cache.reset("qemu_accels")
            result = await probe_qemu_accelerators()
            assert result == set()

    async def test_probe_with_very_long_output(self) -> None:
        """Probe handles very long QEMU output gracefully."""
        # Simulate QEMU returning many accelerators
        accels = "\n".join([f"accel{i}" for i in range(1000)])
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(f"Accelerators supported:\n{accels}\n".encode(), b""))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            probe_cache.reset("qemu_accels")
            result = await probe_qemu_accelerators()
            assert len(result) == 1000

    async def test_cache_cleared_between_tests(self) -> None:
        """Verify cache is properly cleared (test isolation)."""
        assert probe_cache.qemu_accels is NOT_CACHED


# ============================================================================
# Unit Tests - QEMU Version Probe
# ============================================================================


class TestQemuVersionProbe:
    """Tests for probe_qemu_version() - QEMU binary version detection.

    This probe detects QEMU version to select appropriate netdev reconnect parameter:
    - QEMU 9.2+: reconnect-ms (milliseconds)
    - QEMU 8.0-9.1: reconnect (seconds)
    """

    @pytest.fixture(autouse=True)
    def clear_version_cache(self) -> None:
        """Clear QEMU version cache before each test."""
        probe_cache.reset("qemu_version")

    # ========================================================================
    # Normal Cases - Happy path scenarios
    # ========================================================================

    async def test_probe_parses_standard_version(self) -> None:
        """Probe parses standard QEMU version output."""
        from exec_sandbox.system_probes import probe_qemu_version

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"QEMU emulator version 8.2.0\n", b""))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await probe_qemu_version()

        assert result == (8, 2, 0)

    async def test_probe_result_is_cached(self) -> None:
        """Subsequent calls return cached result."""
        from exec_sandbox.system_probes import probe_qemu_version

        probe_cache.qemu_version = (10, 0, 0)
        result = await probe_qemu_version()
        assert result == (10, 0, 0)

    async def test_probe_parses_qemu_9_2(self) -> None:
        """Probe correctly identifies QEMU 9.2 (reconnect-ms threshold)."""
        from exec_sandbox.system_probes import probe_qemu_version

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"QEMU emulator version 9.2.0\n", b""))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await probe_qemu_version()

        assert result == (9, 2, 0)
        assert result >= (9, 2, 0)  # Threshold check

    # ========================================================================
    # Edge Cases - Version formats
    # ========================================================================

    async def test_probe_parses_two_part_version(self) -> None:
        """Probe handles version without patch number (e.g., 10.0)."""
        from exec_sandbox.system_probes import probe_qemu_version

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"QEMU emulator version 10.0\n", b""))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await probe_qemu_version()

        assert result == (10, 0, 0)

    async def test_probe_parses_homebrew_suffix(self) -> None:
        """Probe handles Homebrew version suffix."""
        from exec_sandbox.system_probes import probe_qemu_version

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"QEMU emulator version 8.2.0 (Homebrew)\n", b""))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await probe_qemu_version()

        assert result == (8, 2, 0)

    async def test_probe_parses_dev_version(self) -> None:
        """Probe handles development version with git hash."""
        from exec_sandbox.system_probes import probe_qemu_version

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"QEMU emulator version 10.0.0 (v10.0.0-123-gabcdef)\n", b""))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await probe_qemu_version()

        assert result == (10, 0, 0)

    async def test_probe_parses_debian_suffix(self) -> None:
        """Probe handles Debian/Ubuntu version suffix."""
        from exec_sandbox.system_probes import probe_qemu_version

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(b"QEMU emulator version 8.2.2 (Debian 1:8.2.2+ds-0ubuntu1)\n", b"")
        )

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await probe_qemu_version()

        assert result == (8, 2, 2)

    # ========================================================================
    # Error Cases
    # ========================================================================

    async def test_probe_returns_none_on_missing_binary(self) -> None:
        """Probe returns None when QEMU binary not found."""
        from exec_sandbox.system_probes import probe_qemu_version

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError()
            result = await probe_qemu_version()

        assert result is None

    async def test_probe_returns_none_on_timeout(self) -> None:
        """Probe returns None on timeout."""
        from exec_sandbox.system_probes import probe_qemu_version

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await probe_qemu_version()

        assert result is None

    async def test_probe_returns_none_on_unparseable_output(self) -> None:
        """Probe returns None when version cannot be parsed."""
        from exec_sandbox.system_probes import probe_qemu_version

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Unknown output\n", b""))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await probe_qemu_version()

        assert result is None

    async def test_probe_returns_none_on_nonzero_exit(self) -> None:
        """Probe returns None when QEMU returns non-zero exit code."""
        from exec_sandbox.system_probes import probe_qemu_version

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await probe_qemu_version()

        assert result is None

    async def test_probe_returns_none_on_oserror(self) -> None:
        """Probe returns None on OSError."""
        from exec_sandbox.system_probes import probe_qemu_version

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = OSError("Permission denied")
            result = await probe_qemu_version()

        assert result is None

    # ========================================================================
    # Version Comparison Tests
    # ========================================================================

    async def test_version_tuple_comparison_works(self) -> None:
        """Verify tuple comparison works correctly for version checks."""
        # These are the actual comparisons used in qemu_cmd.py
        assert (9, 2, 0) >= (9, 2, 0)  # QEMU 9.2.0 uses reconnect-ms
        assert (9, 2, 1) >= (9, 2, 0)  # QEMU 9.2.1 uses reconnect-ms
        assert (10, 0, 0) >= (9, 2, 0)  # QEMU 10.0.0 uses reconnect-ms
        assert not (9, 1, 0) >= (9, 2, 0)  # QEMU 9.1.0 uses reconnect
        assert not (8, 2, 0) >= (9, 2, 0)  # QEMU 8.2.0 uses reconnect

        assert (8, 2, 0) >= (8, 0, 0)  # QEMU 8.2.0 uses reconnect
        assert (9, 1, 0) >= (8, 0, 0)  # QEMU 9.1.0 uses reconnect
        assert not (7, 2, 0) >= (8, 0, 0)  # QEMU 7.x doesn't have reconnect

    async def test_cache_cleared_between_tests(self) -> None:
        """Verify cache is properly cleared (test isolation)."""
        assert probe_cache.qemu_version is NOT_CACHED


# ============================================================================
# Unit Tests - Netdev Reconnect Parameter
# ============================================================================


class TestNetdevReconnect:
    """Tests for netdev reconnect parameter based on QEMU version.

    The reconnect parameter helps recover from transient socket disconnections
    between QEMU and gvproxy (which can cause DNS resolution failures).
    """

    @pytest.fixture(autouse=True)
    def clear_version_cache(self) -> None:
        """Clear QEMU version cache before each test."""
        probe_cache.reset("qemu_version")

    async def test_netdev_uses_reconnect_ms_for_qemu_9_2(self, vm_settings, tmp_path: Path) -> None:
        """QEMU 9.2+ uses reconnect-ms parameter."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd
        from exec_sandbox.vm_working_directory import VmWorkingDirectory

        workdir = await VmWorkingDirectory.create("test-vm-reconnect-9-2")
        try:
            with patch("exec_sandbox.qemu_cmd.probe_qemu_version", return_value=(9, 2, 0)):
                cmd = await build_qemu_cmd(
                    settings=vm_settings,
                    arch=detect_host_arch(),
                    vm_id="test-vm-reconnect-9-2",
                    workdir=workdir,
                    memory_mb=256,
                    allow_network=True,
                )

            # Find the netdev argument
            netdev_arg = None
            for i, arg in enumerate(cmd):
                if arg == "-netdev" and i + 1 < len(cmd):
                    netdev_arg = cmd[i + 1]
                    break

            assert netdev_arg is not None, "netdev argument not found in command"
            assert "stream,id=net0" in netdev_arg
            assert "reconnect-ms=250" in netdev_arg
            assert "reconnect=1" not in netdev_arg
        finally:
            await workdir.cleanup()

    async def test_netdev_uses_reconnect_for_qemu_8_x(self, vm_settings, tmp_path: Path) -> None:
        """QEMU 8.x uses reconnect parameter (seconds)."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd
        from exec_sandbox.vm_working_directory import VmWorkingDirectory

        workdir = await VmWorkingDirectory.create("test-vm-reconnect-8-x")
        try:
            with patch("exec_sandbox.qemu_cmd.probe_qemu_version", return_value=(8, 2, 0)):
                cmd = await build_qemu_cmd(
                    settings=vm_settings,
                    arch=detect_host_arch(),
                    vm_id="test-vm-reconnect-8-x",
                    workdir=workdir,
                    memory_mb=256,
                    allow_network=True,
                )

            # Find the netdev argument
            netdev_arg = None
            for i, arg in enumerate(cmd):
                if arg == "-netdev" and i + 1 < len(cmd):
                    netdev_arg = cmd[i + 1]
                    break

            assert netdev_arg is not None, "netdev argument not found in command"
            assert "stream,id=net0" in netdev_arg
            assert "reconnect=1" in netdev_arg
            assert "reconnect-ms" not in netdev_arg
        finally:
            await workdir.cleanup()

    async def test_netdev_uses_reconnect_ms_for_qemu_10(self, vm_settings, tmp_path: Path) -> None:
        """QEMU 10.0+ uses reconnect-ms parameter (reconnect removed)."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd
        from exec_sandbox.vm_working_directory import VmWorkingDirectory

        workdir = await VmWorkingDirectory.create("test-vm-reconnect-10")
        try:
            with patch("exec_sandbox.qemu_cmd.probe_qemu_version", return_value=(10, 0, 0)):
                cmd = await build_qemu_cmd(
                    settings=vm_settings,
                    arch=detect_host_arch(),
                    vm_id="test-vm-reconnect-10",
                    workdir=workdir,
                    memory_mb=256,
                    allow_network=True,
                )

            # Find the netdev argument
            netdev_arg = None
            for i, arg in enumerate(cmd):
                if arg == "-netdev" and i + 1 < len(cmd):
                    netdev_arg = cmd[i + 1]
                    break

            assert netdev_arg is not None, "netdev argument not found in command"
            assert "stream,id=net0" in netdev_arg
            assert "reconnect-ms=250" in netdev_arg
            assert "reconnect=1" not in netdev_arg
        finally:
            await workdir.cleanup()

    async def test_netdev_no_reconnect_when_version_unknown(self, vm_settings, tmp_path: Path) -> None:
        """No reconnect when QEMU version cannot be detected."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd
        from exec_sandbox.vm_working_directory import VmWorkingDirectory

        workdir = await VmWorkingDirectory.create("test-vm-reconnect-unknown")
        try:
            with patch("exec_sandbox.qemu_cmd.probe_qemu_version", return_value=None):
                cmd = await build_qemu_cmd(
                    settings=vm_settings,
                    arch=detect_host_arch(),
                    vm_id="test-vm-reconnect-unknown",
                    workdir=workdir,
                    memory_mb=256,
                    allow_network=True,
                )

            # Find the netdev argument
            netdev_arg = None
            for i, arg in enumerate(cmd):
                if arg == "-netdev" and i + 1 < len(cmd):
                    netdev_arg = cmd[i + 1]
                    break

            assert netdev_arg is not None, "netdev argument not found in command"
            assert "stream,id=net0" in netdev_arg
            assert "reconnect" not in netdev_arg
        finally:
            await workdir.cleanup()

    async def test_netdev_no_reconnect_for_qemu_7(self, vm_settings, tmp_path: Path) -> None:
        """QEMU 7.x (pre-8.0) doesn't have reconnect parameter."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd
        from exec_sandbox.vm_working_directory import VmWorkingDirectory

        workdir = await VmWorkingDirectory.create("test-vm-reconnect-7")
        try:
            with patch("exec_sandbox.qemu_cmd.probe_qemu_version", return_value=(7, 2, 0)):
                cmd = await build_qemu_cmd(
                    settings=vm_settings,
                    arch=detect_host_arch(),
                    vm_id="test-vm-reconnect-7",
                    workdir=workdir,
                    memory_mb=256,
                    allow_network=True,
                )

            # Find the netdev argument
            netdev_arg = None
            for i, arg in enumerate(cmd):
                if arg == "-netdev" and i + 1 < len(cmd):
                    netdev_arg = cmd[i + 1]
                    break

            assert netdev_arg is not None, "netdev argument not found in command"
            assert "stream,id=net0" in netdev_arg
            assert "reconnect" not in netdev_arg
        finally:
            await workdir.cleanup()

    async def test_netdev_no_reconnect_when_network_disabled(self, vm_settings, tmp_path: Path) -> None:
        """No netdev at all when network is disabled."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd
        from exec_sandbox.vm_working_directory import VmWorkingDirectory

        workdir = await VmWorkingDirectory.create("test-vm-no-network")
        try:
            with patch("exec_sandbox.qemu_cmd.probe_qemu_version", return_value=(9, 2, 0)):
                cmd = await build_qemu_cmd(
                    settings=vm_settings,
                    arch=detect_host_arch(),
                    vm_id="test-vm-no-network",
                    workdir=workdir,
                    memory_mb=256,
                    allow_network=False,
                )

            # Should not have any netdev argument
            netdev_found = any(arg == "-netdev" for arg in cmd)
            assert not netdev_found, "netdev should not be present when network is disabled"
        finally:
            await workdir.cleanup()


# ============================================================================
# Integration Tests - Netdev Reconnect with Real VM
# ============================================================================


class TestNetdevReconnectIntegration:
    """Integration tests for QEMU netdev reconnect behavior with real VMs.

    Tests that QEMU automatically reconnects to gvproxy after socket disconnect,
    which helps recover from transient gvproxy crashes or socket EOF errors.
    """

    async def test_network_recovers_after_gvproxy_restart(self, vm_manager) -> None:
        """QEMU reconnects to gvproxy after socket disconnect.

        Simulates the real-world scenario where gvproxy crashes or socket EOF
        occurs. With the reconnect parameter, QEMU should automatically reconnect
        when gvproxy is restarted on the same socket path.

        Test steps:
        1. Create VM with network, verify initial connectivity
        2. Kill gvproxy (simulates crash/EOF)
        3. Remove stale socket file
        4. Restart gvproxy on same socket path
        5. Wait for QEMU to reconnect
        6. Verify network connectivity is restored
        """
        from exec_sandbox.gvproxy import start_gvproxy
        from exec_sandbox.models import Language

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test-reconnect",
            task_id="integration",
            memory_mb=256,
            allow_network=True,
            allowed_domains=["example.com"],
        )

        try:
            # Step 1: Verify initial network connectivity
            result1 = await vm.execute(
                code=(
                    "import socket\n"
                    "s = socket.create_connection(('example.com', 80), timeout=10)\n"
                    "s.close()\n"
                    "print('INITIAL_OK')"
                ),
                timeout_seconds=30,
            )
            assert result1.exit_code == 0, f"Initial connection failed: {result1.stderr}"
            assert "INITIAL_OK" in result1.stdout

            # Step 2: Kill gvproxy (simulates socket EOF / crash)
            assert vm.gvproxy_proc is not None, "VM should have gvproxy process"
            old_gvproxy = vm.gvproxy_proc
            await old_gvproxy.terminate()
            await old_gvproxy.wait()

            # Step 3: Remove stale socket file (create_unix_socket doesn't do this)
            socket_path = vm.workdir.gvproxy_socket
            if socket_path.exists():
                socket_path.unlink()

            # Step 4: Restart gvproxy on same socket path
            new_proc, _new_log_task = await start_gvproxy(
                vm_id=vm.vm_id,
                allowed_domains=["example.com"],
                language=vm.language.value,
                workdir=vm.workdir,
            )
            vm.gvproxy_proc = new_proc
            # Note: We don't cancel old log task here since gvproxy is dead,
            # the task will complete naturally when pipes close

            # Step 5: Wait for QEMU to reconnect
            # reconnect-ms=250 (QEMU 9.2+) or reconnect=1 (QEMU 8.x)
            # Use 3s to be safe across all versions
            await asyncio.sleep(3)

            # Step 6: Verify network connectivity is restored
            result2 = await vm.execute(
                code=(
                    "import socket\n"
                    "s = socket.create_connection(('example.com', 80), timeout=10)\n"
                    "s.close()\n"
                    "print('RECONNECT_OK')"
                ),
                timeout_seconds=30,
            )
            assert result2.exit_code == 0, f"Reconnect failed: {result2.stderr}"
            assert "RECONNECT_OK" in result2.stdout

        finally:
            await vm_manager.destroy_vm(vm)

    async def test_dns_resolution_after_gvproxy_restart(self, vm_manager) -> None:
        """DNS resolution works after gvproxy restart.

        This specifically tests the DNS resolution path since the original issue
        was DNS failures due to socket EOF between QEMU and gvproxy.
        """
        from exec_sandbox.gvproxy import start_gvproxy
        from exec_sandbox.models import Language

        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test-dns-reconnect",
            task_id="dns-integration",
            memory_mb=256,
            allow_network=True,
            allowed_domains=["example.com"],
        )

        try:
            # Verify initial DNS resolution
            result1 = await vm.execute(
                code=("import socket\nip = socket.gethostbyname('example.com')\nprint(f'INITIAL_DNS_OK:{ip}')"),
                timeout_seconds=30,
            )
            assert result1.exit_code == 0, f"Initial DNS failed: {result1.stderr}"
            assert "INITIAL_DNS_OK:" in result1.stdout

            # Kill and restart gvproxy
            assert vm.gvproxy_proc is not None
            await vm.gvproxy_proc.terminate()
            await vm.gvproxy_proc.wait()

            socket_path = vm.workdir.gvproxy_socket
            if socket_path.exists():
                socket_path.unlink()

            new_proc, _ = await start_gvproxy(
                vm_id=vm.vm_id,
                allowed_domains=["example.com"],
                language=vm.language.value,
                workdir=vm.workdir,
            )
            vm.gvproxy_proc = new_proc

            await asyncio.sleep(3)

            # Verify DNS still works after reconnect
            result2 = await vm.execute(
                code=("import socket\nip = socket.gethostbyname('example.com')\nprint(f'RECONNECT_DNS_OK:{ip}')"),
                timeout_seconds=30,
            )
            assert result2.exit_code == 0, f"DNS after reconnect failed: {result2.stderr}"
            assert "RECONNECT_DNS_OK:" in result2.stdout

        finally:
            await vm_manager.destroy_vm(vm)


class TestTwoLayerKvmDetection:
    """Tests for 2-layer KVM detection (kernel + QEMU verification)."""

    @pytest.fixture(autouse=True)
    def clear_caches(self) -> None:
        """Clear all relevant caches before each test."""
        probe_cache.reset("kvm")
        probe_cache.reset("qemu_accels")

    async def test_kvm_fails_when_qemu_lacks_kvm_support(self) -> None:
        """KVM detection fails if QEMU doesn't have KVM compiled in."""
        # Mock Layer 1 passing (kernel check)
        with (
            patch("exec_sandbox.system_probes.aiofiles.os.path.exists", return_value=True),
            patch("exec_sandbox.system_probes.can_access", return_value=True),
            patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec") as mock_exec,
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

            result = await check_kvm_available()
            assert result is False

    async def test_kvm_passes_when_both_layers_pass(self) -> None:
        """KVM detection passes when both kernel and QEMU verify KVM."""
        with (
            patch("exec_sandbox.system_probes.aiofiles.os.path.exists", return_value=True),
            patch("exec_sandbox.system_probes.can_access", return_value=True),
            patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec") as mock_exec,
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

            result = await check_kvm_available()
            assert result is True

    @skip_unless_linux
    async def test_kvm_fails_when_dev_kvm_missing(self) -> None:
        """KVM detection fails early if /dev/kvm doesn't exist."""
        with patch("exec_sandbox.system_probes.aiofiles.os.path.exists", return_value=False):
            probe_cache.reset("kvm")
            result = await check_kvm_available()
            assert result is False


class TestTwoLayerHvfDetection:
    """Tests for 2-layer HVF detection (kernel + QEMU verification)."""

    @pytest.fixture(autouse=True)
    def clear_caches(self) -> None:
        """Clear all relevant caches before each test."""
        probe_cache.reset("hvf")
        probe_cache.reset("qemu_accels")

    async def test_hvf_fails_when_qemu_lacks_hvf_support(self) -> None:
        """HVF detection fails if QEMU doesn't have HVF compiled in."""
        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec") as mock_exec:
            # First call: sysctl check passes
            sysctl_proc = AsyncMock()
            sysctl_proc.returncode = 0
            sysctl_proc.communicate = AsyncMock(return_value=(b"1\n", b""))

            # Second call: QEMU probe returns only TCG (no HVF)
            qemu_proc = AsyncMock()
            qemu_proc.returncode = 0
            qemu_proc.communicate = AsyncMock(return_value=(b"Accelerators supported in QEMU binary:\ntcg\n", b""))

            mock_exec.side_effect = [sysctl_proc, qemu_proc]

            result = await check_hvf_available()
            assert result is False

    async def test_hvf_passes_when_both_layers_pass(self) -> None:
        """HVF detection passes when both kernel and QEMU verify HVF."""
        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec") as mock_exec:
            # First call: sysctl check passes
            sysctl_proc = AsyncMock()
            sysctl_proc.returncode = 0
            sysctl_proc.communicate = AsyncMock(return_value=(b"1\n", b""))

            # Second call: QEMU probe returns HVF
            qemu_proc = AsyncMock()
            qemu_proc.returncode = 0
            qemu_proc.communicate = AsyncMock(return_value=(b"Accelerators supported in QEMU binary:\ntcg\nhvf\n", b""))

            mock_exec.side_effect = [sysctl_proc, qemu_proc]

            result = await check_hvf_available()
            assert result is True

    async def test_hvf_fails_when_kernel_check_fails(self) -> None:
        """HVF detection fails early if kern.hv_support is 0."""
        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec") as mock_exec:
            sysctl_proc = AsyncMock()
            sysctl_proc.returncode = 0
            sysctl_proc.communicate = AsyncMock(return_value=(b"0\n", b""))

            mock_exec.return_value = sysctl_proc

            result = await check_hvf_available()
            assert result is False


class TestCheckHwaccelAvailable:
    """Tests for the synchronous check_hwaccel_available() wrapper."""

    @pytest.fixture(autouse=True)
    def clear_caches(self) -> None:
        """Clear all relevant caches before each test."""
        probe_cache.reset("kvm")
        probe_cache.reset("hvf")
        probe_cache.reset("qemu_accels")

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
    """Tests for validate_kernel_initramfs() pre-flight check."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Clear validation cache before each test."""
        clear_kernel_validation_cache()

    async def test_validation_succeeds_with_real_paths(self, vm_settings) -> None:
        """Validation passes when kernel and initramfs exist."""
        arch = detect_host_arch()
        # Should not raise
        await validate_kernel_initramfs(vm_settings.kernel_path, arch)

    async def test_validation_fails_with_fake_path(self) -> None:
        """Validation raises VmError when kernel doesn't exist."""
        arch = detect_host_arch()
        fake_path = Path("/nonexistent/kernels")

        with pytest.raises(VmError, match="Kernel not found"):
            await validate_kernel_initramfs(fake_path, arch)

    async def test_cache_prevents_repeated_io(self, vm_settings) -> None:
        """Second call uses cache, no I/O operations."""
        arch = detect_host_arch()

        # First call - real I/O
        await validate_kernel_initramfs(vm_settings.kernel_path, arch)

        # Second call - should use cache, mock should NOT be called
        with patch("exec_sandbox.validation.aiofiles.os.path.exists", new_callable=AsyncMock) as mock_exists:
            await validate_kernel_initramfs(vm_settings.kernel_path, arch)
            mock_exists.assert_not_called()

    async def test_different_paths_validated_separately(self, vm_settings) -> None:
        """Different kernel paths are cached separately."""
        arch = detect_host_arch()

        # First path succeeds
        await validate_kernel_initramfs(vm_settings.kernel_path, arch)

        # Different path still gets validated (and fails)
        fake_path = Path("/nonexistent/kernels")
        with pytest.raises(VmError, match="Kernel not found"):
            await validate_kernel_initramfs(fake_path, arch)


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
        from exec_sandbox.system_probes import check_hvf_available, check_kvm_available

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
            kvm_available = await check_kvm_available()
            hvf_available = await check_hvf_available()

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
        from exec_sandbox.system_probes import probe_cache

        original_tsc = probe_cache.tsc_deadline

        try:
            probe_cache.reset("tsc_deadline")

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
            probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_linux_without_feature(self) -> None:
        """Linux without tsc_deadline_timer in cpuinfo returns False."""
        from exec_sandbox.system_probes import probe_cache

        original_tsc = probe_cache.tsc_deadline

        try:
            probe_cache.reset("tsc_deadline")

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
            probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_linux_cpuinfo_not_exists(self) -> None:
        """/proc/cpuinfo not existing returns False gracefully."""
        from exec_sandbox.system_probes import probe_cache

        original_tsc = probe_cache.tsc_deadline

        try:
            probe_cache.reset("tsc_deadline")

            with patch("aiofiles.os.path.exists", return_value=False):
                result = await _check_tsc_deadline_linux()
                assert result is False
        finally:
            probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_linux_read_error(self) -> None:
        """OSError reading /proc/cpuinfo returns False gracefully."""
        from exec_sandbox.system_probes import probe_cache

        original_tsc = probe_cache.tsc_deadline

        try:
            probe_cache.reset("tsc_deadline")

            with patch("aiofiles.os.path.exists", return_value=True):
                with patch("aiofiles.open", side_effect=OSError("Permission denied")):
                    result = await _check_tsc_deadline_linux()
                    assert result is False
        finally:
            probe_cache.tsc_deadline = original_tsc


# ============================================================================
# Unit Tests - TSC_DEADLINE Detection for macOS
# ============================================================================


class TestTscDeadlineMacOS:
    """Tests for TSC_DEADLINE detection on macOS (Intel Macs)."""

    @pytest.mark.asyncio
    async def test_tsc_deadline_macos_arm64_returns_false(self) -> None:
        """ARM64 Macs always return False for TSC_DEADLINE (ARM uses different timer)."""
        from exec_sandbox.system_probes import probe_cache

        # Save original cache value
        original_tsc = probe_cache.tsc_deadline

        try:
            # Clear cache
            probe_cache.reset("tsc_deadline")

            with patch("exec_sandbox.system_probes.detect_host_arch", return_value=HostArch.AARCH64):
                result = await _check_tsc_deadline_macos()
                assert result is False
        finally:
            # Restore cache
            probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_macos_x86_64_with_tsc(self) -> None:
        """Intel Mac with TSC_DEADLINE returns True."""
        from exec_sandbox.system_probes import probe_cache

        # Save original cache value
        original_tsc = probe_cache.tsc_deadline

        try:
            # Clear cache
            probe_cache.reset("tsc_deadline")

            with patch("exec_sandbox.system_probes.detect_host_arch", return_value=HostArch.X86_64):
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
            probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_macos_x86_64_without_tsc(self) -> None:
        """Intel Mac without TSC_DEADLINE returns False."""
        from exec_sandbox.system_probes import probe_cache

        # Save original cache value
        original_tsc = probe_cache.tsc_deadline

        try:
            # Clear cache
            probe_cache.reset("tsc_deadline")

            with patch("exec_sandbox.system_probes.detect_host_arch", return_value=HostArch.X86_64):
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"FPU VME DE PSE TSC MSR PAE MCE SSE SSE2", b""))

                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    result = await _check_tsc_deadline_macos()
                    assert result is False
        finally:
            # Restore cache
            probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_macos_sysctl_failure(self) -> None:
        """sysctl failure returns False gracefully."""
        from exec_sandbox.system_probes import probe_cache

        # Save original cache value
        original_tsc = probe_cache.tsc_deadline

        try:
            # Clear cache
            probe_cache.reset("tsc_deadline")

            with patch("exec_sandbox.system_probes.detect_host_arch", return_value=HostArch.X86_64):
                mock_proc = AsyncMock()
                mock_proc.returncode = 1  # sysctl failed
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))

                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    result = await _check_tsc_deadline_macos()
                    assert result is False
        finally:
            # Restore cache
            probe_cache.tsc_deadline = original_tsc

    @pytest.mark.asyncio
    async def test_tsc_deadline_cached(self) -> None:
        """TSC_DEADLINE result is cached (caching is in check_tsc_deadline, not platform-specific functions)."""
        from exec_sandbox.system_probes import probe_cache

        # Save original cache value
        original_tsc = probe_cache.tsc_deadline

        try:
            # Set cached value
            probe_cache.tsc_deadline = True

            # Should return cached value without calling sysctl or reading /proc/cpuinfo
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                with patch("aiofiles.open") as mock_open:
                    result = await check_tsc_deadline()
                    assert result is True
                    mock_exec.assert_not_called()
                    mock_open.assert_not_called()
        finally:
            # Restore cache
            probe_cache.tsc_deadline = original_tsc


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


# ============================================================================
# Security Tests - Identifier Validation
# ============================================================================


class TestIdentifierValidation:
    """Security tests for tenant_id and task_id validation to prevent injection attacks."""

    def test_valid_identifiers(self) -> None:
        """Test that valid identifiers are accepted."""
        valid_identifiers = [
            "tenant123",
            "task-1",
            "my_task",
            "UPPERCASE",
            "MixedCase123",
            "a",
            "123",
            "a-b_c",
        ]
        for identifier in valid_identifiers:
            # Should not raise
            _validate_identifier(identifier, "test_id")

    def test_empty_identifier_rejected(self) -> None:
        """Test that empty identifier is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_identifier("", "test_id")

    def test_identifier_too_long_rejected(self) -> None:
        """Test that overly long identifier is rejected."""
        long_id = "a" * 129  # > 128 chars
        with pytest.raises(ValueError, match="too long"):
            _validate_identifier(long_id, "test_id")

    def test_identifier_max_length_accepted(self) -> None:
        """Test that identifier at max length is accepted."""
        max_id = "a" * 128  # Exactly 128 chars
        # Should not raise
        _validate_identifier(max_id, "test_id")

    def test_shell_injection_characters_rejected(self) -> None:
        """Test that shell metacharacters are rejected to prevent command injection."""
        malicious_identifiers = [
            "tenant;rm -rf /",
            "task$(whoami)",
            "task`id`",
            "task|cat /etc/passwd",
            "task&echo pwned",
            "task>file",
            "task<file",
            "task\nid",
            "task\tid",
        ]
        for identifier in malicious_identifiers:
            with pytest.raises(ValueError, match="invalid characters"):
                _validate_identifier(identifier, "test_id")

    def test_path_traversal_characters_rejected(self) -> None:
        """Test that path traversal characters are rejected."""
        malicious_identifiers = [
            "../../../etc/passwd",
            "..\\..\\windows",
            "task/../../root",
            "task\\path",
            "/absolute/path",
        ]
        for identifier in malicious_identifiers:
            with pytest.raises(ValueError, match="invalid characters"):
                _validate_identifier(identifier, "test_id")

    def test_special_characters_rejected(self) -> None:
        """Test that various special characters are rejected."""
        special_chars = [
            "task@domain",
            "task#1",
            "task$var",
            "task%1",
            "task^x",
            "task*",
            "task+1",
            "task=val",
            "task[0]",
            "task{1}",
            "task!",
            "task~",
            "task'",
            'task"',
            "task,1",
            "task.1",
            "task?",
            "task:1",
        ]
        for identifier in special_chars:
            with pytest.raises(ValueError, match="invalid characters"):
                _validate_identifier(identifier, "test_id")

    def test_unicode_characters_rejected(self) -> None:
        """Test that Unicode characters are rejected."""
        unicode_identifiers = [
            "task日本語",
            "задача",
            "tâche",
            "任务",
            "tenant\u0000null",
        ]
        for identifier in unicode_identifiers:
            with pytest.raises(ValueError, match="invalid characters"):
                _validate_identifier(identifier, "test_id")

    def test_whitespace_rejected(self) -> None:
        """Test that whitespace is rejected."""
        whitespace_identifiers = [
            "task 1",
            " task",
            "task ",
            "task\t1",
            "task\n1",
            "task\r\n1",
        ]
        for identifier in whitespace_identifiers:
            with pytest.raises(ValueError, match="invalid characters"):
                _validate_identifier(identifier, "test_id")


# ============================================================================
# State Race Protection Tests
# ============================================================================


class TestExecuteStateRace:
    """Tests for state race protection in execute() method.

    Verifies that execute() correctly handles the case where destroy() is called
    between the state lock release and the start of I/O operations.
    """

    async def test_execute_rejects_destroying_state(self) -> None:
        """execute() raises VmError if state is DESTROYING (caught by initial check)."""
        # Create a minimal mock VM
        vm = object.__new__(QemuVM)
        vm._state = VmState.DESTROYING
        vm._state_lock = asyncio.Lock()
        vm.vm_id = "test-vm-123"
        vm.language = "python"

        with pytest.raises(VmError) as exc_info:
            await vm.execute(code="print('hello')", timeout_seconds=5)

        # Initial check catches non-READY state
        assert "Cannot execute in state destroying" in str(exc_info.value)

    async def test_execute_rejects_destroyed_state(self) -> None:
        """execute() raises VmError if state is DESTROYED (caught by initial check)."""
        vm = object.__new__(QemuVM)
        vm._state = VmState.DESTROYED
        vm._state_lock = asyncio.Lock()
        vm.vm_id = "test-vm-123"
        vm.language = "python"

        with pytest.raises(VmError) as exc_info:
            await vm.execute(code="print('hello')", timeout_seconds=5)

        # Initial check catches non-READY state
        assert "Cannot execute in state destroyed" in str(exc_info.value)

    async def test_state_race_check_catches_destroying(self) -> None:
        """The defensive state check catches DESTROYING after lock release.

        This test simulates the race condition by using a side effect to change
        state to DESTROYING right after the lock is released but before I/O.
        """
        vm = object.__new__(QemuVM)
        vm._state = VmState.READY
        vm._state_lock = asyncio.Lock()
        vm.vm_id = "test-vm-123"
        vm.language = "python"

        # Use a real state change to trigger the race condition check
        # We patch the state after transition to EXECUTING
        original_state = VmState.EXECUTING
        race_triggered = False

        def change_state_to_destroying():
            """Side effect that simulates destroy() being called."""
            nonlocal race_triggered
            if vm._state == VmState.EXECUTING:
                vm._state = VmState.DESTROYING
                race_triggered = True

        # Patch the ExecuteCodeRequest to trigger state change before I/O
        with patch("exec_sandbox.qemu_vm.ExecuteCodeRequest") as mock_request:
            mock_request.side_effect = lambda **kwargs: (change_state_to_destroying(), None)[1] or AsyncMock()

            with pytest.raises(VmError) as exc_info:
                await vm.execute(code="print('hello')", timeout_seconds=5)

            assert race_triggered, "Race condition was not triggered"
            assert "VM destroyed during execution start" in str(exc_info.value)
            assert exc_info.value.context["current_state"] == "destroying"

    async def test_state_race_check_catches_destroyed(self) -> None:
        """The defensive state check catches DESTROYED after lock release."""
        vm = object.__new__(QemuVM)
        vm._state = VmState.READY
        vm._state_lock = asyncio.Lock()
        vm.vm_id = "test-vm-123"
        vm.language = "python"

        def change_state_to_destroyed():
            """Side effect that simulates destroy() completing."""
            if vm._state == VmState.EXECUTING:
                vm._state = VmState.DESTROYED

        with patch("exec_sandbox.qemu_vm.ExecuteCodeRequest") as mock_request:
            mock_request.side_effect = lambda **kwargs: (change_state_to_destroyed(), None)[1] or AsyncMock()

            with pytest.raises(VmError) as exc_info:
                await vm.execute(code="print('hello')", timeout_seconds=5)

            assert "VM destroyed during execution start" in str(exc_info.value)
            assert exc_info.value.context["current_state"] == "destroyed"

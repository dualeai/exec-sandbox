"""Tests for VmManager.

Unit tests: State machine, platform detection.
Integration tests: Real VM lifecycle (requires QEMU + images).
"""

import asyncio
import errno
import grp
import hashlib
import json
import os
import pwd
import re
import signal
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager, ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from exec_sandbox import cgroup, constants
from exec_sandbox.cgroup import Cgroup
from exec_sandbox.constants import GuestErrorType
from exec_sandbox.exceptions import (
    CommunicationError,
    ConfigError,
    ErrorCode,
    InternalError,
    InvalidInputError,
    OutcomeUnknownError,
    SandboxError,
    SocketAuthError,
    TransientError,
)
from exec_sandbox.files.locks import CacheEntry, EntryLease
from exec_sandbox.images import GuestImages, guest_digests
from exec_sandbox.models import ExposedPort, Language
from exec_sandbox.platform_utils import (
    HostArch,
    HostOS,
    detect_host_arch,
)
from exec_sandbox.process_registry import unregister_process
from exec_sandbox.qemu_vm import QemuDiagnostics, QemuVM
from exec_sandbox.settings import Settings
from exec_sandbox.subprocess_utils import child_shell, start_managed_process
from exec_sandbox.system_probes import (
    NOT_CACHED,
    check_hvf_available,
    check_kvm_available,
    check_tsc_deadline,
    detect_accel_type,
    probe_cache,
    probe_qemu_accelerators,
)
from exec_sandbox.validation import (
    clear_kernel_validation_cache,
    validate_kernel_initramfs,
)
from exec_sandbox.vm_manager import (
    VmManager,
    _share_all,
    _stop_qemu,
)
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import VALID_STATE_TRANSITIONS, AccelType, VmState
from exec_sandbox.vm_working_directory import VmWorkingDirectory
from tests.conftest import asyncio_with, skip_unless_linux, skip_unless_macos
from tests.test_files_locks import evictable
from tests.test_subprocess_utils import write_foreign_elf

# Home disks of test workdirs (no VM runs here, so no file is created)
_HOME_DIR = Path(tempfile.gettempdir())


@pytest.fixture
def unit_test_vm_manager(unit_test_settings: Any) -> Iterator[VmManager]:
    """VmManager for unit tests that boot no VM, with its folder made (the first step of
    start(), which these tests skip): _setup_vm_infra records each VM in it."""
    manager = VmManager(unit_test_settings)
    manager._open_run_dir()  # pyright: ignore[reportPrivateUsage]
    yield manager
    if manager._run_dir is not None:  # pyright: ignore[reportPrivateUsage]
        manager._run_dir.close()  # pyright: ignore[reportPrivateUsage]


# ============================================================================
# Unit Tests - VM State Machine
# ============================================================================


# The x86 CPU model is a cross-host contract (a state saved on one host loads on
# another only with the same model): written out here, never derived from qemu_cmd.
_X86_MODEL = (
    "qemu64-v1,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+abm,+movbe,+xsave,+xsaveopt,+cx16,+lahf_lm,"
    "+popcnt,+sse4.1,+sse4.2,+ssse3,+aes,+pclmulqdq,+rdrand,+rdseed,+hypervisor,-svm,-vmx"
)
_X86_KVM_PV = "+kvm-pv-ipi,+kvm-pv-tlb-flush,+kvm-pv-sched-yield,+kvm-poll-control,+kvm-asyncpf-int"


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


# ============================================================================
# Unit Tests - Platform Detection
# ============================================================================


class TestKvmDetection:
    """Tests for KVM availability detection. Each Linux case (faked /dev/kvm, ioctl and QEMU
    answers): TestTwoLayerKvmDetection."""

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

    # ========================================================================
    # Normal Cases - Happy path scenarios
    # ========================================================================

    async def test_probe_contains_tcg(self) -> None:
        """TCG should always be available (software emulation fallback)."""
        result = await probe_qemu_accelerators()
        assert isinstance(result, set)
        assert "tcg" in result, "TCG should always be available in QEMU"

    @skip_unless_macos
    async def test_probe_contains_hvf_on_macos(self) -> None:
        """The macOS QEMU build (Homebrew) has HVF: `-accel help` lists the accelerators
        compiled in, also on a host without Hypervisor.framework access."""
        assert "hvf" in await probe_qemu_accelerators()

    @skip_unless_linux
    async def test_probe_contains_kvm_on_linux(self) -> None:
        """The Linux QEMU builds (distribution packages, scripts/build-qemu.sh --enable-kvm)
        have KVM: `-accel help` lists it also on a host without /dev/kvm."""
        assert "kvm" in await probe_qemu_accelerators()

    async def test_probe_result_is_cached(self) -> None:
        """Subsequent calls return cached result."""
        result1 = await probe_qemu_accelerators()
        result2 = await probe_qemu_accelerators()
        assert result1 is result2, "Results should be the same cached object"

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
        mock_proc.kill = MagicMock()  # Process.kill is sync: run_probe kills the child on a timeout

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
            assert result == {"tcg", "kvm"}

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
            assert result == {"tcg", "kvm", "hvf"}

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
            assert result == {"tcg", "xen", "whpx", "nvmm"}

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
            assert result == {"tcg", "kvm"}  # no "\r" is left on a name

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
        """The second call gives the first answer and runs no QEMU."""
        from exec_sandbox.system_probes import probe_qemu_version

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"QEMU emulator version 10.0.0\n", b""))

        with patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", return_value=mock_proc) as spawn:
            assert await probe_qemu_version() == (10, 0, 0)
            assert await probe_qemu_version() == (10, 0, 0)

        assert spawn.call_count == 1

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
        mock_proc.kill = MagicMock()  # Process.kill is sync: run_probe kills the child on a timeout

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

    async def test_cache_cleared_between_tests(self) -> None:
        """Verify cache is properly cleared (test isolation)."""
        assert probe_cache.qemu_version is NOT_CACHED


# ============================================================================
# Shared base class for build_qemu_cmd() unit tests
# ============================================================================

# Functions qemu_cmd imports from the probe modules that read only their arguments, or only
# what they are given (qemu_unrunnable_error: the error of probe_qemu_runnable's answer).
_PURE_PROBE_MODULE_HELPERS = frozenset({"qemu_system_binary", "qemu_unrunnable_error"})


def _qemu_cmd_host_probes() -> list[str]:
    """Each function qemu_cmd imports from a module that reads the host, but the pure helpers.

    Derived from the module, so a probe added to build_qemu_cmd() is guarded at once.
    """
    from exec_sandbox import qemu_cmd

    probe_modules = {"exec_sandbox.permission_utils", "exec_sandbox.platform_utils", "exec_sandbox.system_probes"}
    return sorted(
        name
        for name, value in vars(qemu_cmd).items()
        if callable(value)
        and not isinstance(value, type)
        and getattr(value, "__module__", None) in probe_modules
        and name not in _PURE_PROBE_MODULE_HELPERS
    )


class _QemuCmdTestBase:
    """Base class for build_qemu_cmd() unit tests with shared fixtures.

    Provides:
    - workdir: async fixture providing a temporary VmWorkingDirectory
    - a guard: a host probe that the test did not pin (_qemu_cmd_mocks) fails
      the test, so the result never depends on the host that runs it
    """

    @pytest.fixture(autouse=True)
    def _no_real_host_probe(self) -> Iterator[None]:
        def guard(name: str) -> Any:
            def fail(*_args: object, **_kwargs: object) -> None:
                pytest.fail(f"build_qemu_cmd() ran the real host probe {name}(): pin it with _qemu_cmd_mocks()")

            return fail

        with ExitStack() as stack:
            for name in _qemu_cmd_host_probes():
                stack.enter_context(patch(f"exec_sandbox.qemu_cmd.{name}", side_effect=guard(name)))
            yield

    @pytest.fixture
    async def workdir(self):
        from exec_sandbox.vm_working_directory import VmWorkingDirectory

        wd = await VmWorkingDirectory.create("test-vm", home_dir=_HOME_DIR)
        try:
            yield wd
        finally:
            await wd.cleanup()


def _option(cmd: list[str], flag: str) -> str | None:
    """The value after the first `flag` in a QEMU command, or None without it."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# ============================================================================
# Unit Tests - Netdev Reconnect Parameter
# ============================================================================


class TestNetdevReconnect(_QemuCmdTestBase):
    """Tests for netdev reconnect parameter based on QEMU version.

    The reconnect parameter helps recover from transient socket disconnections
    between QEMU and gvproxy (which can cause DNS resolution failures).
    """

    @pytest.mark.parametrize(
        "qemu_version, present, absent",
        [
            pytest.param((10, 0, 0), "reconnect-ms=250", "reconnect=1", id="10.0-reconnect-ms"),
            pytest.param((9, 2, 0), "reconnect-ms=250", "reconnect=1", id="9.2-reconnect-ms"),
            pytest.param((9, 1, 9), "reconnect=1", "reconnect-ms", id="9.1-reconnect-seconds"),
            pytest.param((8, 2, 0), "reconnect=1", "reconnect-ms", id="8.x-reconnect-seconds"),
            pytest.param((8, 0, 0), "reconnect=1", "reconnect-ms", id="8.0-reconnect-seconds"),
            pytest.param((7, 2, 0), None, "reconnect", id="7.x-none"),
            pytest.param(None, None, "reconnect", id="unknown-none"),
        ],
    )
    async def test_netdev_reconnect_per_qemu_version(
        self,
        vm_settings,
        workdir,
        qemu_version: tuple[int, int, int] | None,
        present: str | None,
        absent: str,
    ) -> None:
        """QEMU 9.2+ uses reconnect-ms, 8.0-9.1 reconnect (seconds), older or unknown QEMU none."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(qemu_version=qemu_version):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-reconnect",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=True),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        netdev = _option(cmd, "-netdev")
        assert netdev is not None, "netdev argument not found in command"
        assert "stream,id=net0" in netdev
        assert present is None or present in netdev
        assert absent not in netdev

    async def test_netdev_no_reconnect_when_network_disabled(self, vm_settings, workdir) -> None:
        """No netdev at all when network is disabled."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-no-network",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert "-netdev" not in cmd, "netdev should not be present when network is disabled"


# ============================================================================
# Unit Tests - NIC None (Cold-Start Optimization)
# ============================================================================


class TestQemuNicNone(_QemuCmdTestBase):
    """-nic none suppresses QEMU's default NIC when no network is needed.

    Without -nic none, non-microvm machine types (ARM64 virt, x86 pc/q35)
    create a default NIC that causes the guest-agent's verify_gvproxy() to
    burn ~4s in exponential-backoff retries. Adding -nic none eliminates
    this cold-start overhead.
    """

    @pytest.mark.parametrize("arch", [HostArch.X86_64, HostArch.AARCH64], ids=["x86", "arm64"])
    @pytest.mark.parametrize(
        "allow_network, expose_ports, expect_nic_none",
        [
            pytest.param(False, None, True, id="no-network"),
            pytest.param(True, None, False, id="allow-network"),
            pytest.param(
                False,
                [ExposedPort(internal=8080, external=9090)],
                False,
                id="expose-ports-only",
            ),
            pytest.param(
                True,
                [ExposedPort(internal=8080, external=9090)],
                False,
                id="both-network-and-ports",
            ),
            pytest.param(False, [], True, id="empty-expose-ports"),
        ],
    )
    async def test_nic_none_flag(
        self,
        *,
        vm_settings,
        workdir,
        arch: HostArch,
        allow_network: bool,
        expose_ports: list[ExposedPort] | None,
        expect_nic_none: bool,
    ) -> None:
        """Verify -nic none presence/absence based on network configuration."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-nic-none",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=allow_network),
                expose_ports=expose_ports,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        has_nic_none = _option(cmd, "-nic") == "none"
        assert has_nic_none == expect_nic_none
        if expect_nic_none:
            assert "-netdev" not in cmd
            assert not any("virtio-net" in arg for arg in cmd)
        else:
            assert "-netdev" in cmd


# ============================================================================
# Unit Tests - Run-With Exit-With-Parent
# ============================================================================


class TestRunWithExitParent(_QemuCmdTestBase):
    """-run-with for QEMU 10.2+: exit-with-parent=on on every host, async-teardown=on on Linux only.

    exit-with-parent=on kills QEMU when its parent dies (PR_SET_PDEATHSIG on
    Linux, kqueue on macOS). QEMU builds async-teardown only on Linux: the
    macOS QEMU refuses the option ("Invalid parameter 'async-teardown'").
    """

    @pytest.mark.parametrize(
        "host_os, accel_type, arch, expected",
        [
            pytest.param(
                HostOS.LINUX,
                AccelType.KVM,
                HostArch.X86_64,
                "exit-with-parent=on,async-teardown=on",
                id="linux-x86",
            ),
            pytest.param(
                HostOS.LINUX,
                AccelType.TCG,
                HostArch.AARCH64,
                "exit-with-parent=on,async-teardown=on",
                id="linux-arm64",
            ),
            pytest.param(HostOS.MACOS, AccelType.HVF, HostArch.AARCH64, "exit-with-parent=on", id="macos-arm64"),
        ],
    )
    @pytest.mark.parametrize("qemu_version", [(10, 2, 0), (11, 1, 3)], ids=["10.2", "11.1"])
    async def test_run_with_for_qemu_10_2_and_later(
        self,
        *,
        vm_settings,
        workdir,
        qemu_version: tuple[int, int, int],
        host_os: HostOS,
        accel_type: AccelType,
        arch: HostArch,
        expected: str,
    ) -> None:
        """QEMU 10.2+ gets one -run-with: async-teardown=on is added on Linux only."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(qemu_version=qemu_version, host_os=host_os, accel_type=accel_type):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-exit-parent",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert cmd.count("-run-with") == 1
        assert _option(cmd, "-run-with") == expected

    @pytest.mark.parametrize("host_os", [HostOS.LINUX, HostOS.MACOS], ids=["linux", "macos"])
    @pytest.mark.parametrize("qemu_version", [(10, 1, 0), None], ids=["10.1", "unknown"])
    async def test_no_run_with_before_qemu_10_2(
        self, vm_settings, workdir, qemu_version: tuple[int, int, int] | None, host_os: HostOS
    ) -> None:
        """QEMU 10.1 or an unknown version: no -run-with (exit-with-parent is from 10.2)."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(qemu_version=qemu_version, host_os=host_os):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-exit-parent",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert "-run-with" not in cmd


# ============================================================================
# Tests - SMP / CPU cores
# ============================================================================


class TestSmpCpuCores(_QemuCmdTestBase):
    """Verify -smp flag scales with cpu_cores parameter."""

    @pytest.mark.parametrize("arch", [HostArch.X86_64, HostArch.AARCH64], ids=["x86", "arm64"])
    @pytest.mark.parametrize("cpu_cores", [1, 2, 4])
    async def test_smp_matches_cpu_cores(self, vm_settings, workdir, cpu_cores: int, arch: HostArch) -> None:
        """-smp N matches the cpu_cores parameter."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id=f"test-smp-{cpu_cores}",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=cpu_cores, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        # maxcpus capped to prevent QEMU allocating extra vCPU structs
        assert _option(cmd, "-smp") == f"{cpu_cores},maxcpus={cpu_cores}"


class TestQemuExecutableSelection(_QemuCmdTestBase):
    """argv[0] honors a configured QEMU path, else falls back to the PATH basename."""

    async def test_command_uses_configured_qemu_path(self, vm_settings, workdir, tmp_path: Path) -> None:
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        configured = tmp_path / "qemu-system-aarch64"
        configured.write_bytes(b"")
        settings = vm_settings.model_copy(update={"qemu_bin_arm": configured})
        with _qemu_cmd_mocks(host_os=HostOS.MACOS, accel_type=AccelType.HVF):
            cmd = await build_qemu_cmd(
                settings=settings,
                arch=HostArch.AARCH64,
                vm_id="exact-qemu-executable",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=192, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert cmd[0] == str(configured)

    async def test_command_falls_back_to_path_basename(self, vm_settings, workdir) -> None:
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        settings = vm_settings.model_copy(update={"qemu_bin_arm": Path("/missing/qemu-system-aarch64")})
        with _qemu_cmd_mocks(host_os=HostOS.MACOS, accel_type=AccelType.HVF):
            cmd = await build_qemu_cmd(
                settings=settings,
                arch=HostArch.AARCH64,
                vm_id="exact-qemu-executable",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=192, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert cmd[0] == "qemu-system-aarch64"


# ============================================================================
# Unit Tests - cpuidle Governor Per Accel Type
# ============================================================================


class TestCpuidleGovernor(_QemuCmdTestBase):
    """Verify cpuidle kernel cmdline is set correctly per acceleration type.

    KVM → cpuidle.governor=haltpoll (in-guest polling avoids VM-exit cost).
    HVF/TCG → cpuidle.off=1 (bypass cpuidle, call WFI directly).
    """

    @pytest.mark.parametrize(
        "accel_type, expected",
        [
            pytest.param(AccelType.KVM, "cpuidle.governor=haltpoll", id="kvm"),
            pytest.param(AccelType.HVF, "cpuidle.off=1", id="hvf"),
            pytest.param(AccelType.TCG, "cpuidle.off=1", id="tcg"),
        ],
    )
    async def test_one_cpuidle_directive_per_accel_type(
        self, vm_settings, workdir, accel_type: AccelType, expected: str
    ) -> None:
        """The kernel command line has one cpuidle directive, the one of the accel type."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(accel_type=accel_type):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-cpuidle",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        cpuidle = [token for token in _extract_kernel_cmdline(cmd).split() if "cpuidle." in token]
        assert cpuidle == [expected]


# ============================================================================
# Unit Tests - Speculative-execution posture in kernel cmdline
# ============================================================================

# Expected tokens for the guest mitigation posture:
# the VM boundary (KVM/HVF) is the isolation layer; guest-side mitigations add
# CPU overhead and per-CPU state without protecting a co-resident tenant.
# nokaslr is forbidden so the cmdline does not explicitly disable address
# randomization. The x86 base enables CONFIG_RANDOMIZE_BASE, but this
# token test does not prove runtime KASLR on the direct-vmlinux PVH path. arm64
# has CONFIG_RANDOMIZE_BASE=n and remains without KASLR/KPTI under
# mitigations=off; enabling arm64 KPTI is a separate performance decision.
# Guest-side reporting companion: test_speculative_execution.py::
# TestMitigationStatus.
REQUIRED_CMDLINE_TOKENS = ("mitigations=off",)
FORBIDDEN_CMDLINE_TOKENS = ("nokaslr",)


class TestKernelCmdlineMitigations(_QemuCmdTestBase):
    """Pin the speculative-execution posture of the built kernel cmdline.

    Host-side seam: build_qemu_cmd() is pure command construction (no VM
    boot, no image files needed). This CANNOT be asserted from inside the
    guest — guest-agent masks /proc/cmdline with a /dev/null bind-mount
    (guest-agent/src/init.rs), so in-guest reads return "" and such a test
    would pass with any cmdline.
    """

    async def test_mitigation_tokens_match_posture(self, vm_settings, workdir) -> None:
        """mitigations=off present, nokaslr absent — the density posture."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-mitigations",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        # Token-split, not substring: substring would match inside other params.
        tokens = _extract_kernel_cmdline(cmd).split()
        for token in REQUIRED_CMDLINE_TOKENS:
            assert token in tokens, (
                f"density posture requires '{token}' in kernel cmdline "
                f"(see qemu_cmd.py mitigations block); got: {' '.join(tokens)}"
            )
        for token in FORBIDDEN_CMDLINE_TOKENS:
            assert token not in tokens, (
                f"'{token}' is forbidden: do not explicitly disable address randomization; got: {' '.join(tokens)}"
            )


class TestKernelCmdlinePressureStallInformation(_QemuCmdTestBase):
    """Pin PSI activation required by the guest capped-zram guard."""

    @pytest.mark.parametrize("arch", [HostArch.X86_64, HostArch.AARCH64])
    async def test_memory_psi_is_explicitly_enabled(self, vm_settings, workdir, arch) -> None:
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-memory-psi",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        tokens = _extract_kernel_cmdline(cmd).split()
        assert tokens.count("psi=1") == 1


# ============================================================================
# Shared helpers for build_qemu_cmd() unit tests
# ============================================================================


@contextmanager
def _fake_guest_port_listeners() -> Iterator[list[socket.socket]]:
    """_guest_port_listeners with two unbound sockets (the launch tests' working folder is a
    mock). At exit, each one the launch made must be closed, on success and on failure."""
    made: list[socket.socket] = []

    def make(_workdir: object) -> list[socket.socket]:
        made.extend(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) for _ in range(2))
        return made[-2:]

    with patch("exec_sandbox.vm_manager._guest_port_listeners", side_effect=make):
        try:
            yield made
        finally:
            left_open = [sock for sock in made if sock.fileno() != -1]
            for sock in made:
                sock.close()
            assert not left_open, "the launch left a listener open"


def _qemu_cmd_mocks(
    *,
    qemu_version: tuple[int, int, int] | None = (9, 2, 0),
    host_os: HostOS = HostOS.LINUX,
    accel_type: AccelType = AccelType.KVM,
    tsc_deadline: bool = True,
    unshare: bool = False,
    io_uring: bool = False,
    direct_io: bool = False,
    qemu_sandbox: bool = True,
    qemu_vm_uid: int | None = None,
    os_getuid: int = 1000,
    x86_vendor: str = "AuthenticAMD",
):
    """Return a combined context manager that patches all build_qemu_cmd() probes.

    Also mocks get_qemu_vm_uid() and os.getuid() to prevent any real
    user-database or privilege lookups during tests.
    """
    stack = ExitStack()
    stack.enter_context(patch("exec_sandbox.qemu_cmd.detect_host_x86_vendor", return_value=x86_vendor))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.probe_qemu_version", return_value=qemu_version))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.probe_qemu_runnable", return_value=None))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.detect_host_os", return_value=host_os))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.detect_accel_type", return_value=accel_type))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.check_tsc_deadline", return_value=tsc_deadline))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.probe_unshare_support", return_value=unshare))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.probe_io_uring_support", return_value=io_uring))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.probe_drive_direct_io", return_value=direct_io))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.probe_qemu_sandbox_support", return_value=qemu_sandbox))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.get_qemu_vm_uid", return_value=qemu_vm_uid))
    stack.enter_context(patch("exec_sandbox.qemu_cmd.os.getuid", return_value=os_getuid))
    return stack


def _extract_machine_type(cmd: list[str]) -> str:
    """Extract the -M value from a QEMU command list."""
    idx = cmd.index("-M")
    return cmd[idx + 1]


def _extract_kernel_cmdline(cmd: list[str]) -> str:
    """Extract the -append value from a QEMU command list."""
    idx = cmd.index("-append")
    return cmd[idx + 1]


def _extract_cpu_model(cmd: list[str]) -> str:
    """Extract the -cpu value from a QEMU command list."""
    idx = cmd.index("-cpu")
    return cmd[idx + 1]


def _extract_accel(cmd: list[str]) -> str:
    """Extract the -accel value from a QEMU command list."""
    idx = cmd.index("-accel")
    return cmd[idx + 1]


# ============================================================================
# Unit Tests - Machine Type Selection
# ============================================================================


class TestMachineTypeSelection(_QemuCmdTestBase):
    """Verify -M flag matches arch/accel/platform combinations.

    The core of the microvm refactoring: all x86_64 paths use microvm,
    machine_type is built from parts, KVM/HVF TSC logic is unified.
    """

    @pytest.mark.parametrize(
        "arch, accel_type, tsc, host_os, expected_machine_type",
        [
            pytest.param(
                HostArch.X86_64,
                AccelType.KVM,
                True,
                HostOS.LINUX,
                "microvm,acpi=off,x-option-roms=off,pit=off,pic=off,rtc=off,isa-serial=off,mem-merge=on,dump-guest-core=off,memory-backend=mem0",
                id="x86-kvm-tsc-linux",
            ),
            pytest.param(
                HostArch.X86_64,
                AccelType.KVM,
                False,
                HostOS.LINUX,
                "microvm,acpi=off,x-option-roms=off,isa-serial=off,mem-merge=on,dump-guest-core=off,memory-backend=mem0",
                id="x86-kvm-notsc-linux",
            ),
            pytest.param(
                HostArch.X86_64,
                AccelType.HVF,
                True,
                HostOS.MACOS,
                "microvm,acpi=off,x-option-roms=off,pit=off,pic=off,rtc=off,isa-serial=off,mem-merge=off,memory-backend=mem0",
                id="x86-hvf-tsc-macos",
            ),
            pytest.param(
                HostArch.X86_64,
                AccelType.HVF,
                False,
                HostOS.MACOS,
                "microvm,acpi=off,x-option-roms=off,isa-serial=off,mem-merge=off,memory-backend=mem0",
                id="x86-hvf-notsc-macos",
            ),
            # TCG and arm64: the TSC_DEADLINE answer (True here) changes nothing
            pytest.param(
                HostArch.X86_64,
                AccelType.TCG,
                True,
                HostOS.LINUX,
                "microvm,acpi=off,x-option-roms=off,isa-serial=off,mem-merge=on,dump-guest-core=off,memory-backend=mem0",
                id="x86-tcg-linux",
            ),
            pytest.param(
                HostArch.X86_64,
                AccelType.TCG,
                True,
                HostOS.MACOS,
                "microvm,acpi=off,x-option-roms=off,isa-serial=off,mem-merge=off,memory-backend=mem0",
                id="x86-tcg-macos",
            ),
            pytest.param(
                HostArch.AARCH64,
                AccelType.KVM,
                True,
                HostOS.LINUX,
                "virt,virtualization=off,highmem=off,gic-version=3,its=off,dtb-randomness=off,mem-merge=on,dump-guest-core=off,memory-backend=mem0",
                id="arm64-kvm-linux",
            ),
            pytest.param(
                HostArch.AARCH64,
                AccelType.HVF,
                True,
                HostOS.MACOS,
                "virt,virtualization=off,highmem=off,gic-version=3,its=off,dtb-randomness=off,mem-merge=off,memory-backend=mem0",
                id="arm64-hvf-macos",
            ),
        ],
    )
    async def test_machine_type(
        self,
        *,
        vm_settings,
        workdir,
        arch: HostArch,
        accel_type: AccelType,
        tsc: bool,
        host_os: HostOS,
        expected_machine_type: str,
    ) -> None:
        """Machine type string matches expected value for arch/accel/platform. Only x86 with
        KVM or HVF reads the TSC_DEADLINE answer (pit/pic/rtc=off)."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(host_os=host_os, accel_type=accel_type, tsc_deadline=tsc):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-machine-type",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert _extract_machine_type(cmd) == expected_machine_type

    @pytest.mark.parametrize(
        "host_os, accel_type, mem_merge, dump_guest_core",
        [
            pytest.param(HostOS.LINUX, AccelType.KVM, "on", ",dump-guest-core=off", id="linux"),
            pytest.param(HostOS.MACOS, AccelType.HVF, "off", "", id="macos"),
        ],
    )
    @pytest.mark.parametrize(
        "qemu_version, transports",
        [
            pytest.param(None, "", id="unknown"),
            pytest.param((10, 2, 0), "", id="10.2"),
            pytest.param((11, 0, 0), ",virtio-mmio-transports=8", id="11.0"),
            pytest.param((11, 1, 3), ",virtio-mmio-transports=8", id="11.1"),
        ],
    )
    async def test_arm64_virtio_mmio_transports_from_qemu_11(
        self,
        *,
        vm_settings,
        workdir,
        qemu_version: tuple[int, int, int] | None,
        transports: str,
        host_os: HostOS,
        accel_type: AccelType,
        mem_merge: str,
        dump_guest_core: str,
    ) -> None:
        """QEMU 11.0+ caps the arm64 virt board at 8 virtio-mmio slots; older QEMU has no such property."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(qemu_version=qemu_version, host_os=host_os, accel_type=accel_type):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.AARCH64,
                vm_id="test-vm-machine-type",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert _extract_machine_type(cmd) == (
            "virt,virtualization=off,highmem=off,gic-version=3,its=off,dtb-randomness=off,"
            f"mem-merge={mem_merge}{transports}{dump_guest_core},memory-backend=mem0"
        )


# ============================================================================
# Unit Tests - Console Selection
# ============================================================================


class TestConsoleSelection(_QemuCmdTestBase):
    """Verify kernel console= parameter matches architecture.

    ARM64 uses PL011 UART (ttyAMA0), x86 uses virtio-console (hvc0).
    """

    @pytest.mark.parametrize(
        "arch, expected_console, absent_console",
        [
            pytest.param(HostArch.X86_64, "console=hvc0", "console=ttyAMA0", id="x86"),
            pytest.param(HostArch.AARCH64, "console=ttyAMA0", "console=hvc0", id="arm64"),
        ],
    )
    async def test_console_per_arch(
        self,
        vm_settings,
        workdir,
        arch: HostArch,
        expected_console: str,
        absent_console: str,
    ) -> None:
        """Each architecture gets its correct console device."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-console",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        cmdline = _extract_kernel_cmdline(cmd)
        assert expected_console in cmdline
        assert absent_console not in cmdline

    @pytest.mark.parametrize(
        ("arch", "consoles"),
        [
            pytest.param(HostArch.X86_64, ["console=hvc0"], id="x86"),
            pytest.param(HostArch.AARCH64, ["console=hvc0", "console=ttyAMA0"], id="arm64"),
        ],
    )
    async def test_a_debug_boot_prints_the_kernel_to_hvc0(
        self, vm_settings, workdir, arch: HostArch, consoles: list[str]
    ) -> None:
        """A debug boot gets the kernel's lines on hvc0 (QEMU's stdout): on ARM64 the PL011
        has no backend, so hvc0 comes first and ttyAMA0 stays last (/dev/console)."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-console-debug",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                debug_boot=True,
                expose_ports=None,
                direct_write=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert [p for p in _extract_kernel_cmdline(cmd).split() if p.startswith("console=")] == consoles


# ============================================================================
# Unit Tests - CPU Model Selection
# ============================================================================


class TestCpuModelSelection(_QemuCmdTestBase):
    """Verify -cpu flag matches arch/accel combinations.

    ARM64 HVF/KVM → host, ARM64 TCG → neoverse-v1,pauth=off (no PAC, no BTI),
    x86 (every accel) → the fixed AVX2-level model; KVM adds PV flags,
    hardware accel with TSC_DEADLINE adds +tsc-deadline.
    """

    @pytest.mark.parametrize(
        "arch, accel_type, expected_cpu",
        [
            pytest.param(HostArch.AARCH64, AccelType.HVF, "host", id="arm64-hvf"),
            pytest.param(HostArch.AARCH64, AccelType.KVM, "host", id="arm64-kvm"),
            pytest.param(HostArch.AARCH64, AccelType.TCG, "neoverse-v1,pauth=off", id="arm64-tcg"),
            pytest.param(
                HostArch.X86_64,
                AccelType.KVM,
                f"{_X86_MODEL},{_X86_KVM_PV},+tsc-deadline,vendor=GenuineIntel,enforce",
                id="x86-kvm",
            ),
            pytest.param(
                HostArch.X86_64,
                AccelType.HVF,
                f"{_X86_MODEL},+tsc-deadline,vendor=GenuineIntel,enforce",
                id="x86-hvf",
            ),
            pytest.param(HostArch.X86_64, AccelType.TCG, f"{_X86_MODEL},vendor=GenuineIntel,enforce", id="x86-tcg"),
        ],
    )
    async def test_cpu_model(
        self,
        vm_settings,
        workdir,
        arch: HostArch,
        accel_type: AccelType,
        expected_cpu: str,
    ) -> None:
        """CPU model string matches expected value."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(accel_type=accel_type, x86_vendor="GenuineIntel"):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-cpu-model",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert _extract_cpu_model(cmd) == expected_cpu


# ============================================================================
# Unit Tests - Accel String
# ============================================================================


class TestAccelString(_QemuCmdTestBase):
    """Verify -accel flag matches acceleration type."""

    @pytest.mark.parametrize(
        "accel_type, cpu_cores, expected_accel",
        [
            pytest.param(AccelType.HVF, 1, "hvf", id="hvf"),
            pytest.param(AccelType.KVM, 1, "kvm", id="kvm"),
            # tb-size: the 128 MB translation cache cap that the cgroup limit and admission count
            pytest.param(AccelType.TCG, 1, "tcg,thread=single,tb-size=128", id="tcg-one-vcpu"),
            pytest.param(AccelType.TCG, 2, "tcg,thread=multi,tb-size=128", id="tcg-two-vcpus-mttcg"),
        ],
    )
    async def test_accel_string(
        self,
        vm_settings,
        workdir,
        accel_type: AccelType,
        cpu_cores: int,
        expected_accel: str,
    ) -> None:
        """Accel string matches expected value; TCG runs one thread per vCPU (MTTCG) from 2 vCPUs."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(accel_type=accel_type):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-accel",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=cpu_cores, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert _extract_accel(cmd) == expected_accel

    @pytest.mark.parametrize(
        ("arch", "qemu_version", "expected_accel"),
        [
            pytest.param(HostArch.AARCH64, (11, 1, 0), "hvf,kernel-irqchip=off", id="arm64-11.1-own-gic"),
            pytest.param(HostArch.AARCH64, (11, 0, 0), "hvf", id="arm64-11.0-no-property"),
            pytest.param(HostArch.X86_64, (11, 1, 0), "hvf", id="x86-mac"),
        ],
    )
    async def test_hvf_arm64_uses_qemus_own_gic_from_11_1(
        self, vm_settings, workdir, arch: HostArch, qemu_version: tuple[int, int, int], expected_accel: str
    ) -> None:
        """ARM64 HVF on QEMU 11.1+: kernel-irqchip=off (with Apple's GIC a lost vCPU kick hung
        QMP stop); QEMU 11.0 has no such property, and an Intel Mac has no GIC choice."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(accel_type=AccelType.HVF, qemu_version=qemu_version):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-hvf-gic",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert _extract_accel(cmd) == expected_accel


# ============================================================================
# Unit Tests - ARM64 TCG Version Guard
# ============================================================================


class TestArm64TcgVersionGuard(_QemuCmdTestBase):
    """Verify QEMU version guard for ARM64 TCG (regime_is_user crash).

    QEMU < 9.0.4 has a deterministic crash on ARM64 TCG.
    """

    @pytest.mark.parametrize(
        "qemu_version, should_raise",
        [
            pytest.param((8, 2, 0), True, id="8.2.0-raises"),
            pytest.param((9, 0, 3), True, id="9.0.3-raises"),
            pytest.param((9, 0, 4), False, id="9.0.4-ok"),
            pytest.param((10, 0, 0), False, id="10.0.0-ok"),
            pytest.param(None, False, id="unknown-version-ok"),
        ],
    )
    async def test_arm64_tcg_version_guard(
        self,
        vm_settings,
        workdir,
        qemu_version: tuple[int, int, int] | None,
        should_raise: bool,
    ) -> None:
        """ARM64 TCG raises ConfigError HOST_DEPENDENCY for QEMU < 9.0.4."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(qemu_version=qemu_version, accel_type=AccelType.TCG):
            if should_raise:
                with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY):
                    await build_qemu_cmd(
                        settings=vm_settings,
                        arch=HostArch.AARCH64,
                        vm_id="test-vm-arm64-tcg",
                        workdir=workdir,
                        spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                        expose_ports=None,
                        direct_write=False,
                        debug_boot=False,
                        snapshot_drive=None,
                        defer_incoming=False,
                        qmp_fd=None,
                        guest_port_fds=None,
                        mem_path=None,
                        mem_path_share=False,
                    )
            else:
                cmd = await build_qemu_cmd(
                    settings=vm_settings,
                    arch=HostArch.AARCH64,
                    vm_id="test-vm-arm64-tcg",
                    workdir=workdir,
                    spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                    expose_ports=None,
                    direct_write=False,
                    debug_boot=False,
                    snapshot_drive=None,
                    defer_incoming=False,
                    qmp_fd=None,
                    guest_port_fds=None,
                    mem_path=None,
                    mem_path_share=False,
                )
                assert _extract_cpu_model(cmd) == "neoverse-v1,pauth=off"  # the arm64 TCG command


# ============================================================================
# Unit Tests - Platform Conditional Flags
# ============================================================================


class TestPlatformConditionalFlags(_QemuCmdTestBase):
    """Verify platform-specific flags in QEMU command line.

    Tests -sandbox (Linux-only), 8250.nr_uarts=0 (x86-only),
    tsc=reliable (x86-only), -nodefaults (microvm-only). dump-guest-core=off
    (Linux-only): TestMachineTypeSelection (the whole -M value).
    """

    @pytest.mark.parametrize(
        "host_os, accel_type, expected_object, expect_device_iothread",
        [
            pytest.param(
                HostOS.MACOS,
                AccelType.HVF,
                "main-loop,id=main-loop0,thread-pool-min=0,thread-pool-max=8",
                False,
                id="macos-hvf-main-loop",
            ),
            pytest.param(
                HostOS.MACOS,
                AccelType.TCG,
                "main-loop,id=main-loop0,thread-pool-min=0,thread-pool-max=8",
                False,
                id="macos-tcg-main-loop",
            ),
            pytest.param(
                HostOS.LINUX,
                AccelType.TCG,
                "main-loop,id=main-loop0,thread-pool-min=0,thread-pool-max=8",
                False,
                id="linux-tcg-main-loop",
            ),
            pytest.param(
                HostOS.LINUX,
                AccelType.KVM,
                "iothread,id=iothread0-test-vm-platform-io,thread-pool-min=0,thread-pool-max=8",
                True,
                id="linux-kvm-dedicated-iothread",
            ),
        ],
    )
    async def test_block_aio_context_is_platform_specific(
        self,
        *,
        vm_settings,
        workdir,
        host_os: HostOS,
        accel_type: AccelType,
        expected_object: str,
        expect_device_iothread: bool,
    ) -> None:
        """Linux KVM keeps the IOThread; without KVM (HVF, TCG on macOS or Linux) block I/O uses the
        capped main-loop pool: with the IOThread, a disk request could stall for good."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(host_os=host_os, accel_type=accel_type, io_uring=False):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.AARCH64,
                vm_id="test-vm-platform-io",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        objects = [cmd[index + 1] for index, value in enumerate(cmd[:-1]) if value == "-object"]
        devices = [cmd[index + 1] for index, value in enumerate(cmd[:-1]) if value == "-device"]
        base_device = next(device for device in devices if "serial=base" in device)
        base_drive = cmd[cmd.index("-drive") + 1]

        assert expected_object in objects
        assert ("iothread=" in base_device) == expect_device_iothread
        assert "aio=threads" in base_drive

    @pytest.mark.parametrize(
        "qemu_sandbox, expect_sandbox",
        [
            pytest.param(True, True, id="sandbox-supported"),
            pytest.param(False, False, id="sandbox-unsupported"),
        ],
    )
    async def test_sandbox_flag(self, vm_settings, workdir, qemu_sandbox: bool, expect_sandbox: bool) -> None:
        """-sandbox present when probe reports support, absent otherwise."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(qemu_sandbox=qemu_sandbox):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-platform-flags",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert ("-sandbox" in cmd) == expect_sandbox

    @pytest.mark.parametrize(
        "arch, expect_uarts",
        [
            pytest.param(HostArch.X86_64, True, id="x86-has-uarts"),
            pytest.param(HostArch.AARCH64, False, id="arm64-no-uarts"),
        ],
    )
    async def test_8250_nr_uarts(self, vm_settings, workdir, arch: HostArch, expect_uarts: bool) -> None:
        """8250.nr_uarts=0 in cmdline on x86, absent on arm64."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-platform-flags",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        cmdline = _extract_kernel_cmdline(cmd)
        assert ("8250.nr_uarts=0" in cmdline) == expect_uarts

    @pytest.mark.parametrize(
        "arch, accel, expect_tsc",
        [
            pytest.param(HostArch.X86_64, AccelType.KVM, False, id="x86-kvm-uses-kvmclock"),
            pytest.param(HostArch.X86_64, AccelType.TCG, True, id="x86-tcg-pins-tsc"),
            pytest.param(HostArch.AARCH64, AccelType.KVM, False, id="arm64-no-tsc"),
        ],
    )
    async def test_tsc_clocksource(
        self, vm_settings, workdir, arch: HostArch, accel: AccelType, expect_tsc: bool
    ) -> None:
        """x86 KVM keeps kvm-clock (it survives a load on a host with another TSC rate);
        x86 without KVM pins the TSC; arm64 never sets TSC flags."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(accel_type=accel):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-platform-flags",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        cmdline = _extract_kernel_cmdline(cmd)
        assert ("tsc=reliable" in cmdline) == expect_tsc
        assert ("clocksource=tsc" in cmdline) == expect_tsc

    @pytest.mark.parametrize(
        "arch, expect_nodefaults",
        [
            pytest.param(HostArch.X86_64, True, id="x86-microvm-has-nodefaults"),
            pytest.param(HostArch.AARCH64, False, id="arm64-virt-no-nodefaults"),
        ],
    )
    async def test_nodefaults(self, vm_settings, workdir, arch: HostArch, expect_nodefaults: bool) -> None:
        """-nodefaults -no-user-config present on x86 (microvm), absent on arm64 (virt)."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-platform-flags",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert ("-nodefaults" in cmd) == expect_nodefaults
        assert ("-no-user-config" in cmd) == expect_nodefaults


# ============================================================================
# Unit Tests - Optional Features (snapshot, defer, debug, direct_write)
# ============================================================================


class TestOptionalFeatures(_QemuCmdTestBase):
    """Verify optional build_qemu_cmd() parameters affect the command line."""

    async def test_snapshot_drive_adds_second_drive(self, vm_settings, workdir) -> None:
        """snapshot_drive adds a second virtio-blk device with serial=snap."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-optional",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                snapshot_drive="/tmp/snap.qcow2",
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        # Second drive present
        assert any("serial=snap" in arg for arg in cmd)
        # init.snap=1 in cmdline
        cmdline = _extract_kernel_cmdline(cmd)
        assert "init.snap=1" in cmdline

    @pytest.mark.parametrize(("direct_write", "expect_read_only"), [(False, True), (True, False)])
    async def test_snapshot_drive_read_only_outside_build(
        self, vm_settings, workdir, direct_write: bool, expect_read_only: bool
    ) -> None:
        """The shared L2 drive (hd1) is read-only, except while the L2 snapshot is built."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-optional",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                snapshot_drive="/tmp/snap.qcow2",
                direct_write=direct_write,
                expose_ports=None,
                debug_boot=False,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        drives = [arg for arg in cmd if arg.startswith("file=")]
        hd1 = next(d for d in drives if "id=hd1," in d)
        assert ("read-only=on" in hd1.split(",")) == expect_read_only

    @pytest.mark.parametrize(
        "direct_folders",
        [frozenset({"overlay", "home", "layer"}), frozenset({"home"}), frozenset()],
        ids=["all-folders", "home-folder-only", "no-folder"],
    )
    @pytest.mark.parametrize("direct_write", [True, False], ids=["snapshot-build", "other-vm"])
    async def test_written_drives_use_o_direct_where_their_folder_fits(
        self, vm_settings, tmp_path: Path, direct_folders: frozenset[str], direct_write: bool
    ) -> None:
        """Each drive the VM writes (overlay hd0, home hd2, a build's layer hd1) skips the host
        page cache when its own folder fits O_DIRECT (probe_drive_direct_io), so its writes
        are no dirty page cache in the VM's cgroup; the shared base under the overlay stays
        cached; the shared read-only layer stays cache=unsafe; no other drive gets a backing
        option."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd
        from exec_sandbox.vm_working_directory import VmWorkingDirectory

        layer = tmp_path / "l2" / "snap.qcow2"
        (tmp_path / "home").mkdir()
        workdir = await VmWorkingDirectory.create("test-vm-direct", home_dir=tmp_path / "home")
        folders = {
            workdir.overlay_image.parent.parent: "overlay",  # the folder of the VM folders: fixed
            workdir.home_disk.parent: "home",
            layer.parent: "layer",
        }
        assert len(folders) == 3  # three distinct folders: each drive asks for its own
        probed: list[str] = []

        async def fits(folder: Path) -> bool:
            probed.append(folders[folder])
            return folders[folder] in direct_folders

        try:
            with _qemu_cmd_mocks(), patch("exec_sandbox.qemu_cmd.probe_drive_direct_io", side_effect=fits):
                cmd = await build_qemu_cmd(
                    settings=vm_settings,
                    arch=HostArch.X86_64,
                    vm_id="test-vm-direct",
                    workdir=workdir,
                    spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                    snapshot_drive=str(layer),
                    direct_write=direct_write,
                    expose_ports=None,
                    debug_boot=False,
                    defer_incoming=False,
                    qmp_fd=None,
                    guest_port_fds=None,
                    mem_path=None,
                    mem_path_share=False,
                )
        finally:
            await workdir.cleanup()

        drives = {
            next(o for o in d.split(",") if o.startswith("id=")): set(d.split(","))
            for d in cmd
            if d.startswith("file=")
        }
        direct = {
            "id=hd0": "overlay" in direct_folders,
            "id=hd2": "home" in direct_folders,
            "id=hd1": direct_write and "layer" in direct_folders,
        }
        for drive, options in drives.items():
            if direct[drive]:
                assert {"cache.direct=on", "cache.no-flush=on"} <= options, drive
                assert not any(o.startswith("cache=") for o in options), drive
            else:
                assert "cache=unsafe" in options, drive
            if drive != "id=hd0":
                assert not any(o.startswith("backing.") for o in options), drive
        assert ("backing.cache.direct=off" in drives["id=hd0"]) == direct["id=hd0"]
        assert ("read-only=on" in drives["id=hd1"]) == (not direct_write)
        assert sorted(probed) == sorted(["overlay", "home", *(["layer"] if direct_write else [])])

    async def test_an_l1_template_save_keeps_its_drives_buffered(self, vm_settings, workdir) -> None:
        """Guest RAM in a shared file (mem_path_share): no O_DIRECT, whose reads would land in
        file-backed shared pages."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(direct_io=True):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-template",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                mem_path="/tmp/l1.ram",
                mem_path_share=True,
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
            )

        drives = [d for d in cmd if d.startswith("file=")]
        assert drives and all("cache=unsafe" in d.split(",") for d in drives), drives

    async def test_home_disk_drive(self, vm_settings, workdir) -> None:
        """Every VM has its private raw home disk (serial=home, hd2), never a shared one."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.AARCH64,
                vm_id="test-vm-home",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, disk_mb=512),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        hd2 = next(arg for arg in cmd if "id=hd2," in arg).split(",")
        assert f"file={workdir.home_disk}" in hd2
        assert "format=raw" in hd2
        assert "file.locking=off" not in hd2
        assert any("drive=hd2,serial=home" in arg for arg in cmd)

    @pytest.mark.parametrize("arch", [HostArch.X86_64, HostArch.AARCH64])
    async def test_one_vm_shape_mem0_no_balloon(self, vm_settings, workdir, arch: HostArch) -> None:
        """Every VM: RAM is the explicit backend mem0 and there is no balloon."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=arch,
                vm_id="test-vm-shape",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert "memory-backend-ram,id=mem0,size=256M" in cmd
        assert "memory-backend=mem0" in _extract_machine_type(cmd)
        assert not any("balloon" in arg for arg in cmd)

    async def test_no_snapshot_drive(self, vm_settings, workdir) -> None:
        """Without snapshot_drive, no second drive or init.snap."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-optional",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert not any("serial=snap" in arg for arg in cmd)
        cmdline = _extract_kernel_cmdline(cmd)
        assert "init.snap=1" not in cmdline

    async def test_defer_incoming(self, vm_settings, workdir) -> None:
        """defer_incoming=True adds -incoming defer."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-optional",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                defer_incoming=True,
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        idx = cmd.index("-incoming")
        assert cmd[idx + 1] == "defer"

    async def test_no_defer_incoming(self, vm_settings, workdir) -> None:
        """defer_incoming=False (default) omits -incoming."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-optional",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert "-incoming" not in cmd

    async def test_debug_boot_verbose(self, vm_settings, workdir) -> None:
        """debug_boot=True enables verbose kernel/init logging."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-optional",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                debug_boot=True,
                expose_ports=None,
                direct_write=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        cmdline = _extract_kernel_cmdline(cmd)
        assert "loglevel=7" in cmdline
        assert "printk.devkmsg=on" in cmdline
        assert "init.quiet=0" in cmdline

    async def test_debug_boot_quiet(self, vm_settings, workdir) -> None:
        """debug_boot=False (default) uses quiet logging."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-optional",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                debug_boot=False,
                expose_ports=None,
                direct_write=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        cmdline = _extract_kernel_cmdline(cmd)
        assert "quiet loglevel=0" in cmdline
        assert "printk.devkmsg=off" in cmdline
        assert "init.quiet=1" in cmdline

    async def test_direct_write(self, vm_settings, workdir) -> None:
        """direct_write=True adds init.rw=1."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-optional",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                direct_write=True,
                expose_ports=None,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        cmdline = _extract_kernel_cmdline(cmd)
        assert "init.rw=1" in cmdline

    async def test_no_direct_write(self, vm_settings, workdir) -> None:
        """direct_write=False (default) omits init.rw=1."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-optional",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                direct_write=False,
                expose_ports=None,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        cmdline = _extract_kernel_cmdline(cmd)
        assert "init.rw=1" not in cmdline


# ============================================================================
# Unit Tests - Unshare Namespaces
# ============================================================================


class TestUnshareNamespaces(_QemuCmdTestBase):
    """Verify Linux namespace isolation (unshare) prefix in QEMU command."""

    async def test_unshare_no_network(self, vm_settings, workdir) -> None:
        """Linux + unshare + no network → full namespace isolation including --net."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(host_os=HostOS.LINUX, unshare=True):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-unshare",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert cmd[0] == "unshare"
        assert "--net" in cmd
        assert "--pid" in cmd
        assert "--fork" in cmd
        # "--" separates unshare args from qemu binary
        separator_idx = cmd.index("--")
        assert "qemu-system-x86_64" in cmd[separator_idx + 1]

    async def test_unshare_with_network(self, vm_settings, workdir) -> None:
        """Linux + unshare + network → QEMU still gets its own netns (gvproxy is a Unix socket)."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(host_os=HostOS.LINUX, unshare=True):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-unshare",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=True),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert cmd[0] == "unshare"
        assert "--net" in cmd
        assert "--pid" in cmd
        assert "--fork" in cmd

    async def test_no_unshare_support(self, vm_settings, workdir) -> None:
        """Linux without unshare support → no unshare prefix."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(host_os=HostOS.LINUX, unshare=False):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-unshare",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert cmd[0] != "unshare"
        assert "qemu-system-x86_64" in cmd[0]

    async def test_macos_no_unshare(self, vm_settings, workdir) -> None:
        """macOS never uses unshare regardless of probe result."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        with _qemu_cmd_mocks(host_os=HostOS.MACOS, unshare=True):
            cmd = await build_qemu_cmd(
                settings=vm_settings,
                arch=HostArch.X86_64,
                vm_id="test-vm-unshare",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert cmd[0] != "unshare"


# ============================================================================
# Integration Tests - Netdev Reconnect with Real VM
# ============================================================================


class TestNetdevReconnectIntegration:
    """Integration tests for QEMU netdev reconnect behavior with real VMs.

    Tests that QEMU automatically reconnects to gvproxy after socket disconnect,
    which helps recover from transient gvproxy crashes or socket EOF errors.
    """

    async def test_network_recovers_after_gvproxy_restart(self, vm_manager, execute_timeout) -> None:
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
            VmSpec(language=Language.PYTHON, memory_mb=256, allow_network=True, allowed_domains=("cp.cloudflare.com",)),
            tenant_id="test-reconnect",
            task_id="integration",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
        )

        # TLS on port 443 with SNI — OutboundAllow only permits TLS
        # connections whose SNI matches a pattern (plain TCP is blocked).
        tls_connect_code = (
            "import socket, ssl\n"
            "ctx = ssl.create_default_context()\n"
            "s = socket.create_connection(('cp.cloudflare.com', 443), timeout=10)\n"
            "ss = ctx.wrap_socket(s, server_hostname='cp.cloudflare.com')\n"
            "ss.close()\n"
        )

        try:
            # Step 1: Verify initial network connectivity (retry for transient
            # TLS failures under TCG emulation where the external handshake
            # can hit SSLEOFError intermittently).
            last_err = ""
            for attempt in range(3):
                if attempt:
                    await asyncio.sleep(2)
                result1 = await vm.execute(
                    code=tls_connect_code + "print('INITIAL_OK')",
                    timeout_seconds=execute_timeout,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                if result1.exit_code == 0 and "INITIAL_OK" in result1.stdout:
                    break
                last_err = result1.stderr
            else:
                pytest.fail(f"Initial connection failed after 3 attempts: {last_err}")

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
            new_proc, _new_log_task, wait_ready = await start_gvproxy(
                vm_id=vm.vm_id,
                allowed_domains=["cp.cloudflare.com"],
                language=vm.spec.language.value,
                workdir=vm.workdir,
                images_dir=vm_manager.images.images_dir,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )
            await wait_ready()
            vm.gvproxy_proc = new_proc
            # Note: We don't cancel old log task here since gvproxy is dead,
            # the task will complete naturally when pipes close

            # Step 5+6: Wait for QEMU to reconnect and verify connectivity.
            # QEMU reconnect-ms=250 triggers automatic reconnection attempts,
            # but under CI CPU contention the full path (QEMU socket reconnect
            # → gvproxy TLS proxy ready → upstream handshake) can take longer
            # than the fixed 3s. Retry with backoff instead.
            last_err = ""
            for attempt in range(5):
                await asyncio.sleep(2 + attempt)  # 2s, 3s, 4s, 5s, 6s
                result2 = await vm.execute(
                    code=tls_connect_code + "print('RECONNECT_OK')",
                    timeout_seconds=execute_timeout,
                    env_vars=None,
                    on_stdout=None,
                    on_stderr=None,
                )
                if result2.exit_code == 0 and "RECONNECT_OK" in result2.stdout:
                    break
                last_err = result2.stderr
            else:
                pytest.fail(f"Reconnect failed after 5 attempts: {last_err}")

        finally:
            await vm_manager.destroy_vm(vm)

    async def test_dns_resolution_after_gvproxy_restart(self, vm_manager, execute_timeout) -> None:
        """DNS resolution works after gvproxy restart.

        The DNS path on its own: a socket EOF between QEMU and gvproxy shows first as
        failed DNS lookups.
        """
        from tenacity import (
            AsyncRetrying,
            retry_if_exception_type,
            stop_after_attempt,
            wait_random_exponential,
        )

        from exec_sandbox.gvproxy import start_gvproxy
        from exec_sandbox.models import Language

        vm = await vm_manager.create_vm(
            VmSpec(language=Language.PYTHON, memory_mb=256, allow_network=True, allowed_domains=("example.com",)),
            tenant_id="test-dns-reconnect",
            task_id="dns-integration",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
        )

        dns_code = "import socket\nip = socket.gethostbyname('example.com')\nprint(f'DNS_OK:{ip}')"

        try:
            # Verify initial DNS resolution (retry with jitter for TCG
            # emulation + pytest -n auto CPU contention on CI).
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5),
                wait=wait_random_exponential(min=1, max=10),
                retry=retry_if_exception_type(AssertionError),
                reraise=True,
            ):
                with attempt:
                    result1 = await vm.execute(
                        code=dns_code, timeout_seconds=execute_timeout, env_vars=None, on_stdout=None, on_stderr=None
                    )
                    assert result1.exit_code == 0 and "DNS_OK:" in result1.stdout, (
                        f"Initial DNS failed: {result1.stderr}"
                    )

            # Kill and restart gvproxy
            assert vm.gvproxy_proc is not None
            await vm.gvproxy_proc.terminate()
            await vm.gvproxy_proc.wait()

            socket_path = vm.workdir.gvproxy_socket
            if socket_path.exists():
                socket_path.unlink()

            new_proc, _, wait_ready = await start_gvproxy(
                vm_id=vm.vm_id,
                allowed_domains=["example.com"],
                language=vm.spec.language.value,
                workdir=vm.workdir,
                images_dir=vm_manager.images.images_dir,
                expose_ports=None,
                block_outbound=False,
                on_process_started=None,
            )
            await wait_ready()
            vm.gvproxy_proc = new_proc

            # Verify DNS still works after reconnect (retry with jitter
            # for CI CPU contention, same rationale as TLS reconnect test).
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5),
                wait=wait_random_exponential(min=2, max=15),
                retry=retry_if_exception_type(AssertionError),
                reraise=True,
            ):
                with attempt:
                    result2 = await vm.execute(
                        code=dns_code, timeout_seconds=execute_timeout, env_vars=None, on_stdout=None, on_stderr=None
                    )
                    assert result2.exit_code == 0 and "DNS_OK:" in result2.stdout, (
                        f"DNS after reconnect failed: {result2.stderr}"
                    )

        finally:
            await vm_manager.destroy_vm(vm)


class TestTwoLayerKvmDetection:
    """Tests for 2-layer KVM detection (kernel + QEMU verification)."""

    async def test_kvm_fails_when_qemu_lacks_kvm_support(self) -> None:
        """KVM detection fails if QEMU doesn't have KVM compiled in."""
        # Mock Layer 1 passing (kernel check)
        with (
            patch.object(Path, "exists", return_value=True),  # /dev/kvm
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
            patch.object(Path, "exists", return_value=True),  # /dev/kvm
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
        with patch.object(Path, "exists", return_value=False):  # no /dev/kvm
            probe_cache.reset("kvm")
            result = await check_kvm_available()
            assert result is False


class TestTwoLayerHvfDetection:
    """Tests for 2-layer HVF detection (kernel + QEMU verification)."""

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


class TestDetectAccelType:
    """detect_accel_type asks KVM first, then HVF on macOS only; TCG when neither answers. The
    host answers at the file and process boundary: no /dev/kvm, while sysctl and QEMU report
    HVF."""

    @pytest.mark.parametrize(
        ("host_os", "expected", "spawns"),
        [(HostOS.LINUX, AccelType.TCG, 0), (HostOS.MACOS, AccelType.HVF, 2), (HostOS.UNKNOWN, AccelType.TCG, 0)],
        ids=["linux-asks-kvm", "macos-asks-hvf", "unknown-asks-kvm-only"],
    )
    async def test_the_accelerator_of_the_host_os(self, host_os: HostOS, expected: AccelType, spawns: int) -> None:
        def answer(stdout: bytes) -> AsyncMock:
            proc = AsyncMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(stdout, b""))
            return proc

        # sysctl kern.hv_support, then QEMU's -accel help
        spawn = AsyncMock(side_effect=[answer(b"1\n"), answer(b"Accelerators supported in QEMU binary:\ntcg\nhvf\n")])
        with (
            patch("exec_sandbox.system_probes.detect_host_os", return_value=host_os),
            patch.object(Path, "exists", return_value=False),  # no /dev/kvm
            patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", spawn),
        ):
            assert await detect_accel_type() is expected
        assert spawn.call_count == spawns


# ============================================================================
# Unit Tests - Kernel/Initramfs Pre-flight Validation
# ============================================================================


class TestDestroyVmStatus:
    """Unit coverage for the QEMU-termination status propagated to sessions.

    The seam is VmManager._force_cleanup_all_resources: it stands in for the OS answer of each
    cleanup step (a QEMU that does not die, a folder that stays), which no test can make a
    real process or file system give on demand. The ephemeral_vm tests stub create_vm and
    destroy_vm for the same reason: the case is destroy_vm's answer."""

    @staticmethod
    def _mock_vm() -> MagicMock:
        vm = MagicMock()
        vm.vm_id = "destroy-status-vm"
        vm.qemu_log_task = None
        vm.gvproxy_log_task = None
        vm.process = MagicMock()
        vm.gvproxy_proc = None
        vm.workdir = None
        vm.cgroup = None
        vm.resource_reservation = None
        vm.state = VmState.DESTROYING
        vm.begin_destroy = AsyncMock()

        async def confirm() -> None:
            # Mirror the real contract: confirm_destroyed publishes DESTROYED.
            vm.state = VmState.DESTROYED

        vm.confirm_destroyed = AsyncMock(side_effect=confirm)
        vm.channel.close = AsyncMock()
        vm.cleanup_lock = asyncio.Lock()
        return vm

    async def test_returns_false_when_qemu_termination_is_unconfirmed(self, unit_test_vm_manager) -> None:
        """Registry removal must not be reported as confirmed QEMU destruction."""
        cleanup = {"qemu": False, "gvproxy": True, "workdir": True, "cgroup": True}
        vm = self._mock_vm()
        reservation = MagicMock()
        vm.resource_reservation = reservation

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager, "_schedule_failed_cleanup_retry") as schedule_retry,
            patch.object(unit_test_vm_manager._admission, "release", new_callable=AsyncMock) as release,
        ):
            assert await unit_test_vm_manager.destroy_vm(vm) is False
        # DESTROYED must never be published while a process is unconfirmed —
        # session/scheduler key retirement on this state.
        vm.confirm_destroyed.assert_not_awaited()
        assert vm.state is VmState.DESTROYING
        assert vm.channel.close.await_count == 1  # no re-close before confirmed death
        assert unit_test_vm_manager.get_active_vms()[vm.vm_id] is vm
        assert vm.resource_reservation is reservation
        release.assert_not_awaited()
        schedule_retry.assert_called_once_with(vm)

    async def test_returns_false_when_gvproxy_termination_is_unconfirmed(self, unit_test_vm_manager) -> None:
        """No process may become unowned while its termination is unconfirmed."""
        cleanup = {"qemu": True, "gvproxy": False, "workdir": True, "cgroup": True}
        vm = self._mock_vm()

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager, "_schedule_failed_cleanup_retry") as schedule_retry,
        ):
            assert await unit_test_vm_manager.destroy_vm(vm) is False
        vm.confirm_destroyed.assert_not_awaited()
        assert vm.state is VmState.DESTROYING
        assert unit_test_vm_manager.get_active_vms()[vm.vm_id] is vm
        schedule_retry.assert_called_once_with(vm)

    @pytest.mark.parametrize("failed_resource", ["workdir", "cgroup"])
    async def test_returns_false_when_one_ancillary_cleanup_is_unconfirmed(
        self,
        unit_test_vm_manager,
        failed_resource: str,
    ) -> None:
        """Dead processes do not make an unconfirmed host resource ownerless."""
        cleanup = {"qemu": True, "gvproxy": True, "workdir": True, "cgroup": True}
        cleanup[failed_resource] = False
        vm = self._mock_vm()

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager, "_schedule_failed_cleanup_retry") as schedule_retry,
        ):
            assert await unit_test_vm_manager.destroy_vm(vm) is False
        vm.confirm_destroyed.assert_awaited_once()
        assert vm.state is VmState.DESTROYED
        # Channel closed once up-front and re-closed after confirmed process
        # death (reaps write workers resurrected by a racing operation).
        assert vm.channel.close.await_count == 2
        assert unit_test_vm_manager.get_active_vms()[vm.vm_id] is vm
        schedule_retry.assert_called_once_with(vm)

    async def test_returns_true_when_all_resource_cleanup_is_confirmed(self, unit_test_vm_manager) -> None:
        cleanup = {"qemu": True, "gvproxy": True, "workdir": True, "cgroup": True}
        vm = self._mock_vm()

        with patch.object(
            unit_test_vm_manager,
            "_force_cleanup_all_resources",
            new_callable=AsyncMock,
            return_value=cleanup,
        ):
            assert await unit_test_vm_manager.destroy_vm(vm) is True
        vm.confirm_destroyed.assert_awaited_once()
        assert vm.state is VmState.DESTROYED
        assert vm.channel.close.await_count == 2
        assert vm.vm_id not in unit_test_vm_manager.get_active_vms()

    async def test_confirmed_process_death_drains_late_console_evidence(self, unit_test_vm_manager) -> None:
        """The reader must observe lines emitted during termination before unpublish."""
        cleanup = {"qemu": True, "gvproxy": True, "workdir": True, "cgroup": True}
        vm = self._mock_vm()
        process_dead = asyncio.Event()
        observed: list[str] = []

        async def drain_after_death() -> None:
            await process_dead.wait()
            observed.append("late console line")

        async def cleanup_resources(**_kwargs) -> dict[str, bool]:  # type: ignore[no-untyped-def]
            process_dead.set()
            await asyncio.sleep(0)
            return cleanup

        vm.qemu_log_task = asyncio.create_task(drain_after_death())
        with patch.object(unit_test_vm_manager, "_force_cleanup_all_resources", side_effect=cleanup_resources):
            assert await unit_test_vm_manager.destroy_vm(vm) is True

        assert observed == ["late console line"]
        assert vm.qemu_log_task.done()

    async def test_unconfirmed_process_keeps_its_log_drain_owned(self, unit_test_vm_manager) -> None:
        """A live/unconfirmed QEMU keeps both process and pipe-reader ownership."""
        cleanup = {"qemu": False, "gvproxy": True, "workdir": True, "cgroup": True}
        vm = self._mock_vm()
        never = asyncio.Event()
        vm.qemu_log_task = asyncio.create_task(never.wait())

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager, "_schedule_failed_cleanup_retry") as schedule_retry,
        ):
            assert await unit_test_vm_manager.destroy_vm(vm) is False

        assert not vm.qemu_log_task.done()
        schedule_retry.assert_called_once_with(vm)
        vm.qemu_log_task.cancel()
        await asyncio.gather(vm.qemu_log_task, return_exceptions=True)

    async def test_successful_attempt_can_return_reservation_to_boot_retry(self, unit_test_vm_manager) -> None:
        """Internal boot retry cleanup must not release its shared reservation."""
        cleanup = {"qemu": True, "gvproxy": True, "workdir": True, "cgroup": True}
        vm = self._mock_vm()
        reservation = MagicMock()
        vm.resource_reservation = reservation

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager._admission, "release", new_callable=AsyncMock) as release,
        ):
            assert await unit_test_vm_manager.destroy_vm(vm, _release_admission=False) is True

        release.assert_not_awaited()
        assert vm.resource_reservation is None

    async def test_stop_destroys_every_tracked_vm_and_retries_one_not_confirmed(self, unit_test_vm_manager) -> None:
        """VmManager.stop (the one VM destroy of a Scheduler exit) destroys each tracked VM. A VM
        whose process death is not confirmed at first does not hold back the others, and the
        stop returns once a retry confirms it, with no VM left."""
        for name in ("a", "b", "c"):
            vm = self._mock_vm()
            vm.vm_id = f"stop-{name}"
            unit_test_vm_manager._vms[vm.vm_id] = vm
        attempts: list[str] = []

        async def cleanup(*, vm_id: str, **_resources: object) -> dict[str, bool]:
            attempts.append(vm_id)
            confirmed = not (vm_id == "stop-b" and attempts.count("stop-b") == 1)
            return {"qemu": confirmed, "gvproxy": True, "workdir": True, "cgroup": True}

        with patch.object(unit_test_vm_manager, "_force_cleanup_all_resources", side_effect=cleanup):
            async with asyncio.timeout(10):
                await unit_test_vm_manager.stop()

        assert sorted(attempts[:3]) == ["stop-a", "stop-b", "stop-c"]  # b's retry does not hold a and c back
        assert attempts[3:] == ["stop-b"]
        assert unit_test_vm_manager.get_active_vms() == {}

    async def test_destroy_commit_finishes_release_before_double_cancellation(self, unit_test_vm_manager) -> None:
        cleanup = {"qemu": True, "gvproxy": True, "workdir": True, "cgroup": True}
        vm = self._mock_vm()
        reservation = MagicMock()
        vm.resource_reservation = reservation
        unit_test_vm_manager._vms[vm.vm_id] = vm
        release_entered = asyncio.Event()
        allow_release = asyncio.Event()

        async def blocked_release(_reservation) -> None:  # type: ignore[no-untyped-def]
            release_entered.set()
            await allow_release.wait()

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager._admission, "release", side_effect=blocked_release) as release,
            patch.object(unit_test_vm_manager, "_schedule_failed_cleanup_retry") as schedule_retry,
        ):
            task = asyncio.create_task(unit_test_vm_manager.destroy_vm(vm))
            await release_entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert unit_test_vm_manager.get_active_vms()[vm.vm_id] is vm

            allow_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        release.assert_awaited_once_with(reservation)
        assert vm.vm_id not in unit_test_vm_manager.get_active_vms()
        assert vm.resource_reservation is None
        schedule_retry.assert_not_called()

    async def test_destroy_vm_confirmed_releases_reservation(self, unit_test_vm_manager) -> None:
        cleanup = {"qemu": True, "gvproxy": True, "workdir": True, "cgroup": True}
        channel = MagicMock()
        channel.close = AsyncMock()
        vm = QemuVM(
            "context-owned-vm",
            process=MagicMock(),
            cgroup=None,
            workdir=MagicMock(),
            channel=channel,
            spec=VmSpec(language=Language.PYTHON),
            console_lines=deque(maxlen=10),
            base_image=Path("/test/base.qcow2"),
            packages_image=None,
            accel_type=AccelType.KVM,
            gvproxy_proc=None,
            qemu_log_task=None,
            gvproxy_log_task=None,
        )
        reservation = MagicMock()
        vm.resource_reservation = reservation
        unit_test_vm_manager._vms[vm.vm_id] = vm

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager._admission, "release", new_callable=AsyncMock) as release,
        ):
            assert await unit_test_vm_manager.destroy_vm(vm) is True

        release.assert_awaited_once_with(reservation)
        assert vm.vm_id not in unit_test_vm_manager.get_active_vms()

    async def test_unconfirmed_destroy_is_retried_until_confirmed(self, unit_test_vm_manager) -> None:
        """destroy_vm returns False while QEMU's end is unconfirmed and keeps the VM; a
        background owner tries again (after 1 s, then 2 s) until the cleanup is confirmed, and
        only then does the VM go."""
        unconfirmed = {"qemu": False, "gvproxy": True, "workdir": True, "cgroup": True}
        confirmed = {"qemu": True, "gvproxy": True, "workdir": True, "cgroup": True}
        vm = self._mock_vm()
        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                side_effect=[unconfirmed, unconfirmed, confirmed],
            ) as cleanup,
            patch("exec_sandbox.vm_manager.asyncio", asyncio_with(sleep=(sleep := AsyncMock()))),
        ):
            assert await unit_test_vm_manager.destroy_vm(vm) is False
            assert unit_test_vm_manager.get_active_vms()[vm.vm_id] is vm
            for _ in range(100):  # turns of the loop: the owner never waits (sleep is a stub)
                if vm.vm_id not in unit_test_vm_manager.get_active_vms():
                    break
                await asyncio.sleep(0)

        assert vm.vm_id not in unit_test_vm_manager.get_active_vms()
        assert vm.state is VmState.DESTROYED
        assert cleanup.await_count == 3
        assert [call.args[0] for call in sleep.await_args_list] == [1.0, 2.0]

    async def test_ephemeral_vm_rejects_unconfirmed_cleanup_on_success(self, unit_test_vm_manager) -> None:
        vm = MagicMock()
        vm.vm_id = "ephemeral-unconfirmed"
        with (
            patch.object(unit_test_vm_manager, "create_vm", new_callable=AsyncMock, return_value=vm),
            patch.object(unit_test_vm_manager, "destroy_vm", new_callable=AsyncMock, return_value=False) as destroy,
        ):
            with pytest.raises(
                InternalError,
                check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED,
                match="cleanup was not confirmed",
            ):
                async with unit_test_vm_manager.ephemeral_vm(
                    VmSpec(language=Language.PYTHON),
                    tenant_id="tenant",
                    task_id="task",
                    retry_profile=constants.RETRY_BACKGROUND,
                    direct_write_target=None,
                    snapshot_drive=None,
                    cache_leases=(),
                    mem_path=None,
                    mem_path_share=False,
                ):
                    pass

        destroy.assert_awaited_once_with(vm)

    @staticmethod
    def _ephemeral_build_vm(manager: VmManager, target: Path) -> AbstractAsyncContextManager[QemuVM]:
        return manager.ephemeral_vm(
            VmSpec(language=Language.PYTHON),
            tenant_id="tenant",
            task_id="task",
            retry_profile=constants.RETRY_BACKGROUND,
            direct_write_target=target,
            snapshot_drive=None,
            cache_leases=(),
            mem_path=None,
            mem_path_share=False,
        )

    @pytest.mark.parametrize("chown_works", [True, False], ids=["taken-back", "sudo-fails"])
    async def test_ephemeral_vm_takes_back_the_target_qemu_vm_wrote(
        self, unit_test_vm_manager, tmp_path: Path, *, chown_works: bool
    ) -> None:
        """QEMU as qemu-vm owns an L2 build target for the build (it writes it). After a clean
        exit, once destroy_vm confirmed QEMU dead, the target goes back to this user (sudo -n
        chown): owned by qemu-vm, any later QEMU as qemu-vm could change the layer that every
        later VM boots with. A chown that fails is HOST_DEPENDENCY: the layer is never published."""
        target = tmp_path / "build.qcow2.part"
        vm = MagicMock()
        vm.vm_id, vm.use_qemu_vm_user = "build", True
        steps: list[str] = []
        me = pwd.getpwuid(os.geteuid())

        async def destroy(_vm: object) -> bool:
            steps.append("destroy")
            return True

        async def chown(path: Path, user: str, group: str, *, owner_uid: int) -> bool:
            steps.append(f"chown {path.name} of {owner_uid} to {user}:{group}")
            return chown_works

        with (
            patch.object(unit_test_vm_manager, "create_vm", new_callable=AsyncMock, return_value=vm),
            patch.object(unit_test_vm_manager, "destroy_vm", destroy),
            patch("exec_sandbox.vm_manager.get_qemu_vm_uid", return_value=986),
            patch("exec_sandbox.vm_manager.chown_async", chown),
        ):
            if chown_works:
                async with self._ephemeral_build_vm(unit_test_vm_manager, target):
                    steps.append("build")
            else:
                with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="chown"):
                    async with self._ephemeral_build_vm(unit_test_vm_manager, target):
                        steps.append("build")

        group = grp.getgrgid(me.pw_gid).gr_name
        assert steps == ["build", "destroy", f"chown {target.name} of 986 to {me.pw_name}:{group}"]

    @pytest.mark.parametrize("case", ["failed-build", "qemu-as-this-user"])
    async def test_ephemeral_vm_takes_back_nothing_else(self, unit_test_vm_manager, tmp_path: Path, case: str) -> None:
        """A failed build removes its target, and a QEMU that runs as this user never had it:
        no chown (a chown error would hide the build error)."""
        vm = MagicMock()
        vm.vm_id, vm.use_qemu_vm_user = "build", case == "failed-build"
        chown = AsyncMock(return_value=False)
        with (
            patch.object(unit_test_vm_manager, "create_vm", new_callable=AsyncMock, return_value=vm),
            patch.object(unit_test_vm_manager, "destroy_vm", new_callable=AsyncMock, return_value=True) as destroy,
            patch("exec_sandbox.vm_manager.chown_async", chown),
        ):
            if case == "failed-build":
                with pytest.raises(RuntimeError, match="install failed"):
                    async with self._ephemeral_build_vm(unit_test_vm_manager, tmp_path / "build.qcow2.part"):
                        raise RuntimeError("install failed")
            else:
                async with self._ephemeral_build_vm(unit_test_vm_manager, tmp_path / "build.qcow2.part"):
                    pass

        destroy.assert_awaited_once_with(vm)
        chown.assert_not_awaited()

    async def test_stop_waits_for_cleanup_owner_instead_of_cancelling_it(self, unit_test_vm_manager) -> None:
        """Manager shutdown cannot report success while its cleanup owner is pending."""
        release_reaper = asyncio.Event()
        reaper_cancelled = False

        async def pending_reaper() -> None:
            nonlocal reaper_cancelled
            try:
                await release_reaper.wait()
            except asyncio.CancelledError:
                reaper_cancelled = True
                raise

        reaper = asyncio.create_task(pending_reaper())
        unit_test_vm_manager._failed_cleanup_tasks["pending-vm"] = reaper
        with (
            patch.object(unit_test_vm_manager._admission, "stop", new_callable=AsyncMock) as admission_stop,
            patch.object(unit_test_vm_manager._overlay_pool, "stop", new_callable=AsyncMock) as overlay_stop,
            patch(
                "exec_sandbox.vm_manager.drain_workdir_cleanup_tasks",
                new_callable=AsyncMock,
            ) as drain_workdirs,
        ):
            stop_task = asyncio.create_task(unit_test_vm_manager.stop())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not stop_task.done()
            assert not reaper.cancelled()
            assert unit_test_vm_manager._stopping is True
            admission_stop.assert_awaited_once()
            overlay_stop.assert_not_awaited()

            release_reaper.set()
            await asyncio.wait_for(stop_task, timeout=1)

        assert not reaper_cancelled
        admission_stop.assert_awaited_once()
        drain_workdirs.assert_awaited_once()
        overlay_stop.assert_awaited_once()

    async def test_cancelled_stop_waiter_does_not_cancel_shared_cleanup(self, unit_test_vm_manager) -> None:
        """A later waiter can observe the same manager-owned stop task complete."""
        release_reaper = asyncio.Event()

        async def pending_reaper() -> None:
            await release_reaper.wait()

        reaper = asyncio.create_task(pending_reaper())
        unit_test_vm_manager._failed_cleanup_tasks["pending-vm"] = reaper
        with (
            patch.object(unit_test_vm_manager._admission, "stop", new_callable=AsyncMock) as admission_stop,
            patch.object(unit_test_vm_manager._overlay_pool, "stop", new_callable=AsyncMock) as overlay_stop,
        ):
            first_waiter = asyncio.create_task(unit_test_vm_manager.stop())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            owned_stop = unit_test_vm_manager._stop_task
            assert owned_stop is not None

            first_waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_waiter
            assert not owned_stop.cancelled()
            assert not reaper.cancelled()
            overlay_stop.assert_not_awaited()

            second_waiter = asyncio.create_task(unit_test_vm_manager.stop())
            release_reaper.set()
            await asyncio.wait_for(second_waiter, timeout=1)

        assert unit_test_vm_manager._stop_task is owned_stop
        admission_stop.assert_awaited_once()
        overlay_stop.assert_awaited_once()

    async def test_stop_rejects_new_launch_and_waits_for_admitted_launch(self, unit_test_vm_manager) -> None:
        """STOPPING closes ingress without losing an operation that already entered."""
        launch_entered = asyncio.Event()
        release_launch = asyncio.Event()

        async def admitted_launch() -> None:
            async with unit_test_vm_manager._launch_operation():
                launch_entered.set()
                await release_launch.wait()

        launch_task = asyncio.create_task(admitted_launch())
        await launch_entered.wait()
        admission_stopped = asyncio.Event()

        async def stop_admission() -> None:
            admission_stopped.set()

        with (
            patch.object(unit_test_vm_manager._admission, "stop", new_callable=AsyncMock) as admission_stop,
            patch.object(unit_test_vm_manager._overlay_pool, "stop", new_callable=AsyncMock) as overlay_stop,
        ):
            admission_stop.side_effect = stop_admission
            stop_task = asyncio.create_task(unit_test_vm_manager.stop())
            await admission_stopped.wait()

            with pytest.raises(
                InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE, match="stopping"
            ):
                async with unit_test_vm_manager._launch_operation():
                    pytest.fail("launch entered after STOPPING")
            assert not stop_task.done()
            admission_stop.assert_awaited_once()
            overlay_stop.assert_not_awaited()

            release_launch.set()
            await launch_task
            await asyncio.wait_for(stop_task, timeout=1)

        overlay_stop.assert_awaited_once()


class TestStartAndStop:
    """start() and stop() of one manager: a stop before, during or after a start. The host
    probes are faked at their module (system_probes, validation); the overlay pool is faked
    to record its start and stop (it would spawn qemu-storage-daemon)."""

    @staticmethod
    @contextmanager
    def _probes(manager: VmManager, steps: list[str], *, accel: Callable[..., Awaitable[AccelType]]) -> Iterator[None]:
        async def pool_start(_images: list[Path]) -> None:
            steps.append("pool start")

        async def pool_stop() -> None:
            steps.append("pool stop")

        with (
            patch("exec_sandbox.vm_manager.detect_accel_type", side_effect=accel),
            patch("exec_sandbox.vm_manager.probe_io_uring_support", new_callable=AsyncMock, return_value=False),
            patch("exec_sandbox.vm_manager.probe_unshare_support", new_callable=AsyncMock, return_value=False),
            patch("exec_sandbox.vm_manager.probe_qemu_version", new_callable=AsyncMock, return_value=(10, 2)),
            patch("exec_sandbox.vm_manager.probe_qemu_sandbox_support", new_callable=AsyncMock, return_value=False),
            patch("exec_sandbox.vm_manager.probe_qemu_as_qemu_vm", new_callable=AsyncMock, return_value=False),
            patch("exec_sandbox.vm_manager.probe_qemu_runnable", new_callable=AsyncMock, return_value=None),
            patch("exec_sandbox.vm_manager.check_tsc_deadline", new_callable=AsyncMock),
            patch("exec_sandbox.vm_manager.validate_kernel_initramfs", new_callable=AsyncMock),
            patch.object(manager._overlay_pool, "start", side_effect=pool_start),
            patch.object(manager._overlay_pool, "stop", side_effect=pool_stop),
        ):
            yield

    async def test_a_stopped_manager_does_not_start(self, tmp_path: Path) -> None:
        """start() after stop(): SCHEDULER_STATE, and nothing starts (the stop that would end it
        ran already)."""
        manager = VmManager(Settings(base_images_dir=tmp_path, kernel_path=tmp_path, home_disk_dir=tmp_path / "home"))
        steps: list[str] = []

        async def tcg(*, force_emulation: bool) -> AccelType:
            return AccelType.TCG

        with self._probes(manager, steps, accel=tcg):
            await manager.stop()
            try:
                with pytest.raises(
                    InvalidInputError, check=lambda e: e.code is ErrorCode.SCHEDULER_STATE, match="stopped"
                ):
                    await manager.start()
            finally:
                await manager.stop()
                if manager._run_dir is not None:
                    manager._run_dir.close()
        assert steps == ["pool stop"]

    async def test_a_stop_during_the_start_stops_what_it_started(self, tmp_path: Path) -> None:
        """stop() while start() runs: the stop waits for the start, then stops the overlay pool
        the start started (a stop that went first would leave it running)."""
        manager = VmManager(Settings(base_images_dir=tmp_path, kernel_path=tmp_path, home_disk_dir=tmp_path / "home"))
        steps: list[str] = []
        probing, resume = asyncio.Event(), asyncio.Event()

        async def slow_probe(*, force_emulation: bool) -> AccelType:
            probing.set()
            await resume.wait()
            return AccelType.TCG

        with self._probes(manager, steps, accel=slow_probe):
            starting = asyncio.create_task(manager.start())
            await probing.wait()
            stopping = asyncio.create_task(manager.stop())
            await asyncio.wait({stopping}, timeout=0.2)  # the stop goes as far as it can
            resume.set()
            await asyncio.wait_for(starting, timeout=5)
            await asyncio.wait_for(stopping, timeout=5)
        assert steps == ["pool start", "pool stop"]

    async def test_a_destroy_after_stop_leaves_the_closed_folder_alone(
        self, unit_test_vm_manager: VmManager, tmp_path: Path
    ) -> None:
        """A destroy after stop() (a session closed late) must not remove records through the
        fd of the run folder that stop() closed: the OS gives that number to the next open,
        here a folder that holds files of the same names."""
        assert unit_test_vm_manager._run_dir is not None
        closed_fd = unit_test_vm_manager._run_dir._fd  # pyright: ignore[reportPrivateUsage]
        workdir = await VmWorkingDirectory.create("late-destroy", home_dir=tmp_path)
        vm = TestDestroyVmStatus._mock_vm()
        vm.vm_id, vm.workdir, vm.process = "late-destroy-vm", workdir, None
        await unit_test_vm_manager.stop()

        other = tmp_path / "other"
        other.mkdir()
        names = sorted([f"{vm.vm_id}.cgroup", f"{workdir.path.name}.folder"])
        for name in names:
            (other / name).touch()
        opened: list[int] = []
        try:
            while not opened or opened[-1] < closed_fd:  # the lowest free numbers, up to the closed one
                opened.append(os.open(other, os.O_RDONLY | os.O_DIRECTORY))
            assert opened[-1] == closed_fd
            assert await unit_test_vm_manager.destroy_vm(vm) is True
        finally:
            for fd in opened:
                os.close(fd)
        assert sorted(p.name for p in other.iterdir()) == names


class TestPendingLaunchOwnership:
    """Pre-QemuVM resources remain owned across every fallible handoff.

    Each test drives one private step of a launch (_setup_vm_infra, _start_gvproxy_for_vm,
    _launch_qemu_vm, _cleanup_pending_launch): the fault it injects (a cancel during a
    release, a cgroup attach that fails after the spawn, a QEMU that does not die) comes
    between two steps, a window that create_vm does not open on demand."""

    @staticmethod
    def _pending(unit_test_vm_manager):  # type: ignore[no-untyped-def]
        infra = MagicMock()
        infra.vm_id = "pending-launch-vm"
        infra.workdir = MagicMock()
        infra.cgroup = Cgroup(Path("/test/cgroup/pending-launch-vm"))
        infra.base_image = Path("/test/images/base.qcow2")  # absent: its digest is unknown
        reservation = MagicMock()
        pending = unit_test_vm_manager._publish_pending_resources(infra.vm_id, reservation, infra.workdir, infra.cgroup)
        infra.pending = pending
        return infra, reservation, pending

    async def test_no_owner_reservation_release_survives_repeated_cancellation(self, unit_test_vm_manager) -> None:
        reservation = MagicMock()
        release_entered = asyncio.Event()
        allow_release = asyncio.Event()
        release_finished = asyncio.Event()

        async def blocked_release(_reservation) -> None:  # type: ignore[no-untyped-def]
            release_entered.set()
            await allow_release.wait()
            release_finished.set()

        with (
            patch("exec_sandbox.vm_manager.detect_accel_type", new_callable=AsyncMock, return_value=AccelType.HVF),
            patch.object(unit_test_vm_manager._admission, "acquire", new_callable=AsyncMock, return_value=reservation),
            patch.object(unit_test_vm_manager._admission, "release", side_effect=blocked_release) as release,
        ):

            async def cancelled_reservation() -> None:
                async with unit_test_vm_manager.reservation_context(
                    VmSpec(language=Language.PYTHON), vm_id="cancelled-reservation", held=frozenset()
                ):
                    raise asyncio.CancelledError

            task = asyncio.create_task(cancelled_reservation())
            await release_entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()

            allow_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert release_finished.is_set()
        release.assert_awaited_once_with(reservation)

    async def test_unconfirmed_cleanup_retains_owner_and_reservation(self, unit_test_vm_manager) -> None:
        """Dependent resources and admission stay held while process death is unconfirmed."""
        _infra, reservation, pending = self._pending(unit_test_vm_manager)
        cleanup = {"qemu": False, "gvproxy": True, "workdir": False, "cgroup": False}

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager, "_schedule_failed_pending_cleanup_retry") as schedule_retry,
            patch.object(unit_test_vm_manager._admission, "release", new_callable=AsyncMock) as release,
        ):
            assert await unit_test_vm_manager._cleanup_pending_launch(pending) is False

        assert unit_test_vm_manager._pending_launches[pending.vm_id] is pending
        assert pending.reservation is reservation
        release.assert_not_awaited()
        schedule_retry.assert_called_once_with(pending)

    async def test_confirmed_attempt_cleanup_preserves_shared_reservation(self, unit_test_vm_manager) -> None:
        """A retryable attempt can clean resources without releasing the outer reservation."""
        _infra, reservation, pending = self._pending(unit_test_vm_manager)
        cleanup = {"qemu": True, "gvproxy": True, "workdir": True, "cgroup": True}

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager._admission, "release", new_callable=AsyncMock) as release,
        ):
            await unit_test_vm_manager._cleanup_failed_owner(
                pending.vm_id, operation="boot_failure_cleanup", error=RuntimeError("boot failed")
            )

        assert pending.vm_id not in unit_test_vm_manager._pending_launches
        assert pending.reservation is reservation
        release.assert_not_awaited()

    async def test_unconfirmed_attempt_cleanup_raises_cleanup_unconfirmed(self, unit_test_vm_manager) -> None:
        """A failed attempt whose cleanup is not confirmed: InternalError CLEANUP_UNCONFIRMED,
        caused by the attempt's failure; the owner stays published for the reaper."""
        _infra, _reservation, pending = self._pending(unit_test_vm_manager)
        cleanup = {"qemu": False, "gvproxy": True, "workdir": False, "cgroup": False}
        failure = RuntimeError("boot failed")

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager, "_schedule_failed_pending_cleanup_retry"),
        ):
            with pytest.raises(
                InternalError,
                check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED,
                match=r"^VM resource cleanup was not confirmed for pending-launch-vm \(boot_failure_cleanup\)$",
            ) as exc_info:
                await unit_test_vm_manager._cleanup_failed_owner(
                    pending.vm_id, operation="boot_failure_cleanup", error=failure
                )

        assert exc_info.value.__cause__ is failure
        assert exc_info.value.context == {"vm_id": "pending-launch-vm", "operation": "boot_failure_cleanup"}
        assert unit_test_vm_manager._pending_launches[pending.vm_id] is pending

    async def test_ancillary_cleanup_failure_retains_retry_owner(self, unit_test_vm_manager) -> None:
        """A removed process is not enough to forget leaked workdir/cgroup state."""
        _infra, _reservation, pending = self._pending(unit_test_vm_manager)
        cleanup = {"qemu": True, "gvproxy": True, "workdir": False, "cgroup": False}

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager, "_schedule_failed_pending_cleanup_retry") as schedule_retry,
            patch.object(unit_test_vm_manager._admission, "release", new_callable=AsyncMock) as release,
        ):
            assert await unit_test_vm_manager._cleanup_pending_launch(pending) is False

        assert unit_test_vm_manager._pending_launches[pending.vm_id] is pending
        release.assert_not_awaited()
        schedule_retry.assert_called_once_with(pending)

    async def test_cleanup_commit_finishes_release_before_double_cancellation(self, unit_test_vm_manager) -> None:
        _infra, _reservation, pending = self._pending(unit_test_vm_manager)
        cleanup = {"qemu": True, "gvproxy": True, "workdir": True, "cgroup": True}
        release_entered = asyncio.Event()
        allow_release = asyncio.Event()
        release_finished = asyncio.Event()

        async def blocked_release(_reservation) -> None:  # type: ignore[no-untyped-def]
            release_entered.set()
            await allow_release.wait()
            release_finished.set()

        with (
            patch.object(
                unit_test_vm_manager,
                "_force_cleanup_all_resources",
                new_callable=AsyncMock,
                return_value=cleanup,
            ),
            patch.object(unit_test_vm_manager._admission, "release", side_effect=blocked_release) as release,
            patch.object(unit_test_vm_manager, "_schedule_failed_pending_cleanup_retry") as schedule_retry,
        ):
            task = asyncio.create_task(unit_test_vm_manager._cleanup_pending_launch(pending))
            await release_entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert unit_test_vm_manager._pending_launches[pending.vm_id] is pending

            allow_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert release_finished.is_set()
        release.assert_awaited_once_with(pending.reservation)
        assert pending.vm_id not in unit_test_vm_manager._pending_launches
        schedule_retry.assert_not_called()

    async def test_infra_cgroup_failure_cleans_published_expected_path(self, unit_test_vm_manager) -> None:
        workdir = MagicMock()
        workdir.overlay_image = Path("/tmp/pending-overlay.qcow2")
        reservation = MagicMock()
        observed: Cgroup | None = None

        async def acquire_overlay(*_args, **_kwargs) -> bool:  # type: ignore[no-untyped-def]
            assert len(unit_test_vm_manager._pending_launches) == 1
            return True

        async def cleanup_owner(vm_id: str, **_kwargs: object) -> None:
            nonlocal observed
            pending = unit_test_vm_manager._pending_launches[vm_id]
            observed = pending.cgroup
            unit_test_vm_manager._pending_launches.pop(vm_id)

        with (
            patch("exec_sandbox.vm_manager.validate_kernel_initramfs", new_callable=AsyncMock),
            patch("exec_sandbox.vm_manager.VmWorkingDirectory.create", new_callable=AsyncMock, return_value=workdir),
            patch.object(GuestImages, "base_image", return_value=Path("/tmp/base.qcow2")),
            patch("exec_sandbox.vm_manager.detect_accel_type", new_callable=AsyncMock, return_value=AccelType.HVF),
            patch.object(unit_test_vm_manager._overlay_pool, "acquire", side_effect=acquire_overlay),
            patch("exec_sandbox.vm_manager.cgroup.cgroup_v2_available", return_value=True),
            patch.object(Cgroup, "create", new_callable=AsyncMock, side_effect=OSError("cgroup")),
            patch.object(unit_test_vm_manager, "_cleanup_failed_owner", side_effect=cleanup_owner) as cleanup,
        ):
            with pytest.raises(OSError, match="cgroup"):
                await unit_test_vm_manager._setup_vm_infra(
                    VmSpec(language=Language.PYTHON),
                    "tenant",
                    "task",
                    reservation,
                    base_image=Path("/fake/base.qcow2"),
                    home_template=None,
                )

        cleanup.assert_awaited_once()
        assert observed is not None
        assert observed.path.parent == Path("/sys/fs/cgroup/code-exec/tenant")
        assert observed.path.name.startswith("tenant-task-")
        assert not unit_test_vm_manager._pending_launches

    async def test_qemu_vm_without_a_cgroup_is_refused_before_any_grant(self, unit_test_vm_manager) -> None:
        """QEMU as qemu-vm with no real cgroup: this user cannot signal that QEMU, and nothing
        could confirm its end once sudo is gone (its leases would go while it runs). The VM
        is refused (HOST_DEPENDENCY) before qemu-vm gets any file."""
        workdir = MagicMock()
        workdir.overlay_image = Path("/tmp/pending-overlay.qcow2")

        async def cleanup_owner(vm_id: str, **_kwargs: object) -> None:
            unit_test_vm_manager._pending_launches.pop(vm_id)

        with (
            patch("exec_sandbox.vm_manager.validate_kernel_initramfs", new_callable=AsyncMock),
            patch("exec_sandbox.vm_manager.VmWorkingDirectory.create", new_callable=AsyncMock, return_value=workdir),
            patch("exec_sandbox.vm_manager.detect_accel_type", new_callable=AsyncMock, return_value=AccelType.KVM),
            patch.object(unit_test_vm_manager._overlay_pool, "acquire", new_callable=AsyncMock),
            patch.object(Cgroup, "create", new_callable=AsyncMock, return_value=None),  # this host gives no cgroup
            patch("exec_sandbox.vm_manager.probe_qemu_as_qemu_vm", new_callable=AsyncMock, return_value=True),
            patch.object(unit_test_vm_manager, "_apply_overlay_permissions", new_callable=AsyncMock) as grants,
            patch.object(unit_test_vm_manager, "_cleanup_failed_owner", side_effect=cleanup_owner),
            pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="no cgroup v2"),
        ):
            await unit_test_vm_manager._setup_vm_infra(
                VmSpec(language=Language.PYTHON),
                "tenant",
                "task",
                MagicMock(),
                base_image=Path("/fake/base.qcow2"),
                home_template=None,
            )
        grants.assert_not_awaited()
        assert not unit_test_vm_manager._pending_launches

    async def test_refused_cgroup_create_still_cleans_its_path(self, unit_test_vm_manager) -> None:
        """A create that the host refused (EACCES during its writes: None) can leave a folder:
        a later failure (here a qemu-vm QEMU, which needs a cgroup) still gives the cleanup
        the VM's cgroup path, not nothing."""
        workdir = MagicMock()
        workdir.overlay_image = Path("/tmp/pending-overlay.qcow2")
        observed: list[tuple[str, Cgroup | None]] = []

        async def cleanup_owner(vm_id: str, **_kwargs: object) -> None:
            observed.append((vm_id, unit_test_vm_manager._pending_launches.pop(vm_id).cgroup))

        with (
            patch("exec_sandbox.vm_manager.validate_kernel_initramfs", new_callable=AsyncMock),
            patch("exec_sandbox.vm_manager.VmWorkingDirectory.create", new_callable=AsyncMock, return_value=workdir),
            patch("exec_sandbox.vm_manager.detect_accel_type", new_callable=AsyncMock, return_value=AccelType.KVM),
            patch.object(unit_test_vm_manager._overlay_pool, "acquire", new_callable=AsyncMock),
            patch("exec_sandbox.vm_manager.cgroup.cgroup_v2_available", return_value=True),
            patch.object(Cgroup, "create", new_callable=AsyncMock, return_value=None),
            patch("exec_sandbox.vm_manager.probe_qemu_as_qemu_vm", new_callable=AsyncMock, return_value=True),
            patch.object(unit_test_vm_manager, "_cleanup_failed_owner", side_effect=cleanup_owner),
            pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="no cgroup v2"),
        ):
            await unit_test_vm_manager._setup_vm_infra(
                VmSpec(language=Language.PYTHON),
                "tenant",
                "task",
                MagicMock(),
                base_image=Path("/fake/base.qcow2"),
                home_template=None,
            )

        [(vm_id, cleaned)] = observed
        assert vm_id.startswith("tenant-task-")
        assert cleaned == Cgroup(Path("/sys/fs/cgroup/code-exec/tenant") / vm_id)
        assert not unit_test_vm_manager._pending_launches
        unit_test_vm_manager._run_dir.forget_cgroup(vm_id)  # recorded (cgroup v2 on): its owner cleanup is a stand-in

    async def test_gvproxy_is_owned_before_post_spawn_cgroup_failure(self, unit_test_vm_manager) -> None:
        """The readiness callback publishes gvproxy before any later await can fail."""
        infra, _reservation, pending = self._pending(unit_test_vm_manager)
        proc = MagicMock()
        proc.pid = 4242
        release_log = asyncio.Event()

        async def log_drain() -> None:
            await release_log.wait()

        log_task = asyncio.create_task(log_drain())

        async def start_with_callback(*_args, on_process_started=None, **_kwargs):  # type: ignore[no-untyped-def]
            assert on_process_started is not None
            on_process_started(proc, log_task)
            return proc, log_task, AsyncMock()

        try:
            with (
                patch("exec_sandbox.vm_manager.start_gvproxy", side_effect=start_with_callback),
                patch.object(Cgroup, "attach", new_callable=AsyncMock, side_effect=OSError("cgroup attach failed")),
            ):
                with pytest.raises(OSError, match="cgroup attach failed"):
                    await unit_test_vm_manager._start_gvproxy_for_vm(
                        infra,
                        pending,
                        VmSpec(language=Language.PYTHON, allow_network=True),
                        None,
                    )

            assert unit_test_vm_manager._pending_launches[pending.vm_id] is pending
            assert pending.gvproxy_proc is proc
            assert pending.gvproxy_log_task is log_task
        finally:
            release_log.set()
            await log_task
            unit_test_vm_manager._pending_launches.pop(pending.vm_id, None)

    async def test_qemu_handoff_precedes_post_spawn_cgroup_failure(self, unit_test_vm_manager) -> None:
        """QEMU transfers synchronously from pending owner to VM registry before cgroup await."""
        infra, reservation, pending = self._pending(unit_test_vm_manager)
        infra.workdir.use_qemu_vm_user = False
        infra.channel = MagicMock()
        proc = MagicMock()
        proc.pid = 4343
        release_log = asyncio.Event()

        async def blocked_drain(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            await release_log.wait()

        try:
            with (
                patch(
                    "exec_sandbox.vm_manager.start_managed_process",
                    new_callable=AsyncMock,
                    return_value=proc,
                ),
                patch("exec_sandbox.subprocess_utils.drain_subprocess_output", side_effect=blocked_drain),
                patch.object(
                    Cgroup, "attach", new_callable=AsyncMock, side_effect=OSError("qemu cgroup attach failed")
                ),
            ):
                with pytest.raises(OSError, match="qemu cgroup attach failed"):
                    await unit_test_vm_manager._launch_qemu_vm(
                        infra,
                        pending,
                        qemu_cmd=["/bin/sh"],  # exists: the launch checks it before the spawn
                        spec=VmSpec(language=Language.PYTHON),
                        packages_image=None,
                        reservation=reservation,
                        on_boot_log=None,
                        gvproxy_proc=None,
                        gvproxy_log_task=None,
                        pass_fds=(),
                        cache_leases=(),
                    )

            vm = unit_test_vm_manager._vms[pending.vm_id]
            assert vm.process is proc
            assert vm.resource_reservation is reservation
            assert pending.vm_id not in unit_test_vm_manager._pending_launches
            assert proc.drain_task is vm.qemu_log_task  # cleanup never reads the pipes the drain task reads
        finally:
            vm = unit_test_vm_manager._vms.pop(pending.vm_id, None)
            release_log.set()
            if vm is not None and vm.qemu_log_task is not None:
                await vm.qemu_log_task


class TestCacheLeasesOfAVm:
    """A VM holds its own share of each cache lease from before QEMU opens the entry until
    QEMU's death is confirmed; only then may an eviction take the entry.

    The seam is _force_cleanup_all_resources (a QEMU death confirmed or not, which a real
    process does not give on demand) and _launch_qemu_vm (a spawn that fails)."""

    @staticmethod
    async def _lease(tmp_path: Path) -> EntryLease:
        entry = tmp_path / "l2.qcow2"
        entry.write_bytes(b"l2")
        lease = await CacheEntry(entry).lease()
        assert lease is not None
        return lease

    @pytest.mark.parametrize(("qemu_dead", "evictable_after"), [(True, True), (False, False)])
    async def test_destroy_releases_the_share_once_qemu_is_dead(
        self, unit_test_vm_manager: VmManager, tmp_path: Path, qemu_dead: bool, evictable_after: bool
    ) -> None:
        lease = await self._lease(tmp_path)
        channel = MagicMock()
        channel.close = AsyncMock()
        vm = QemuVM(
            "lease-vm",
            process=MagicMock(returncode=0),
            cgroup=None,
            workdir=MagicMock(),
            channel=channel,
            spec=VmSpec(language=Language.PYTHON),
            console_lines=deque(),
            base_image=tmp_path / "base.qcow2",
            packages_image=lease.path,
            accel_type=AccelType.KVM,
            gvproxy_proc=None,
            qemu_log_task=None,
            gvproxy_log_task=None,
        )
        vm.cache_leases = (lease.share(),)
        lease.release()  # the request is over: only the VM's share holds the entry
        unit_test_vm_manager._vms[vm.vm_id] = vm
        cleanup = {"qemu": qemu_dead, "gvproxy": True, "workdir": True, "cgroup": True}
        try:
            with (
                patch.object(
                    unit_test_vm_manager, "_force_cleanup_all_resources", new_callable=AsyncMock, return_value=cleanup
                ),
                patch.object(unit_test_vm_manager, "_schedule_failed_cleanup_retry"),
            ):
                assert not evictable(lease.path)
                await unit_test_vm_manager.destroy_vm(vm)
            assert evictable(lease.path) is evictable_after
        finally:
            vm.release_cache_leases()
            unit_test_vm_manager._vms.pop(vm.vm_id, None)

    @pytest.mark.parametrize(("qemu_dead", "evictable_after"), [(True, True), (False, False)])
    async def test_failed_launch_keeps_the_share_until_qemu_is_dead(
        self, unit_test_vm_manager: VmManager, tmp_path: Path, qemu_dead: bool, evictable_after: bool
    ) -> None:
        """The spawn fails: the pending owner holds the share (taken before the spawn) until
        its cleanup confirms that no QEMU runs."""
        lease = await self._lease(tmp_path)
        infra, reservation, pending = TestPendingLaunchOwnership._pending(unit_test_vm_manager)
        infra.workdir.use_qemu_vm_user = False
        cleanup = {"qemu": qemu_dead, "gvproxy": True, "workdir": True, "cgroup": True}
        try:
            with (
                patch(
                    "exec_sandbox.vm_manager.start_managed_process",
                    new_callable=AsyncMock,
                    side_effect=OSError(errno.EAGAIN, "Resource temporarily unavailable"),
                ),
                pytest.raises(TransientError),
            ):
                await unit_test_vm_manager._launch_qemu_vm(
                    infra,
                    pending,
                    qemu_cmd=["/bin/sh"],  # exists: the launch checks it before the spawn
                    spec=VmSpec(language=Language.PYTHON),
                    packages_image=lease.path,
                    reservation=reservation,
                    on_boot_log=None,
                    gvproxy_proc=None,
                    gvproxy_log_task=None,
                    pass_fds=(),
                    cache_leases=(lease,),
                )
            lease.release()  # the caller's own lease goes; the launch's share stays
            assert not evictable(lease.path)
            with (
                patch.object(
                    unit_test_vm_manager, "_force_cleanup_all_resources", new_callable=AsyncMock, return_value=cleanup
                ),
                patch.object(unit_test_vm_manager, "_schedule_failed_pending_cleanup_retry"),
            ):
                await unit_test_vm_manager._cleanup_pending_launch(pending, _release_admission=False)
            assert evictable(lease.path) is evictable_after
        finally:
            pending.release_cache_leases()
            unit_test_vm_manager._pending_launches.pop(pending.vm_id, None)


class TestConfirmDestroyed:
    """confirm_destroyed publishes DESTROYED (and drops the VM's cache leases) even when its
    caller is cancelled while the event watch ends."""

    async def test_cancel_during_the_event_watch_end_still_publishes_destroyed(self, tmp_path: Path) -> None:
        entry = tmp_path / "l2.qcow2"
        entry.write_bytes(b"l2")
        lease = await CacheEntry(entry).lease()
        assert lease is not None
        vm = QemuVM(
            "confirm-vm",
            process=MagicMock(returncode=0),
            cgroup=None,
            workdir=MagicMock(),
            channel=MagicMock(),
            spec=VmSpec(language=Language.RAW),
            console_lines=deque(),
            base_image=tmp_path / "base.qcow2",
            packages_image=entry,
            accel_type=AccelType.KVM,
            gvproxy_proc=None,
            qemu_log_task=None,
            gvproxy_log_task=None,
        )
        vm.cache_leases = (lease,)
        closing, closed = asyncio.Event(), asyncio.Event()

        async def watch() -> None:  # as QmpEventWatch._run: its finally awaits the socket close
            try:
                await asyncio.Event().wait()
            finally:
                closing.set()
                await closed.wait()

        vm._event_watch._task = asyncio.create_task(watch())  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0)  # the watch runs
        caller = asyncio.create_task(vm.confirm_destroyed())
        await closing.wait()  # confirm_destroyed cancelled the watch; it waits for its end
        caller.cancel()
        await asyncio.sleep(0)  # the cancel reaches confirm_destroyed while the watch still ends
        closed.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert vm.state is VmState.DESTROYED
        assert evictable(entry)


class TestImageDigestsOfAVm:
    """A VM records the digests of its guest images when QEMU opens them: a saved instance
    names those bytes, also after the file at that path changes (a rebuilt L2).

    _launch_qemu_vm with a faked spawn: the file must change at the spawn, a moment that
    create_vm does not give on demand."""

    async def test_digests_are_taken_before_the_spawn(self, unit_test_vm_manager: VmManager, tmp_path: Path) -> None:
        base, layer = tmp_path / "base.qcow2", tmp_path / "l2.qcow2"
        base.write_bytes(b"base image")
        layer.write_bytes(b"l2 layer")
        infra, reservation, pending = TestPendingLaunchOwnership._pending(unit_test_vm_manager)
        infra.base_image = base
        infra.workdir.use_qemu_vm_user = False
        infra.channel = MagicMock()
        proc = MagicMock(pid=4343)
        seen_at_spawn: list[bytes] = []

        async def spawn(*_args: object, **_kwargs: object) -> MagicMock:
            seen_at_spawn.append(layer.read_bytes())
            layer.write_bytes(b"rebuilt l2 layer")  # rebuilt at the same path while the VM runs
            return proc

        try:
            with (
                patch("exec_sandbox.vm_manager.start_managed_process", side_effect=spawn),
                patch("exec_sandbox.vm_manager.start_drain", return_value=None),
                patch.object(Cgroup, "attach", new_callable=AsyncMock, return_value=True),
                patch.object(QemuVM, "start_event_watch"),
            ):
                vm = await unit_test_vm_manager._launch_qemu_vm(
                    infra,
                    pending,
                    qemu_cmd=["/bin/sh"],  # exists: the launch checks it before the spawn
                    spec=VmSpec(language=Language.PYTHON),
                    packages_image=layer,
                    reservation=reservation,
                    on_boot_log=None,
                    gvproxy_proc=None,
                    gvproxy_log_task=None,
                    pass_fds=(),
                    cache_leases=(),
                )
            assert seen_at_spawn == [b"l2 layer"]
            assert vm.base_image_digest == hashlib.sha256(b"base image").hexdigest()
            assert vm.packages_image_digest == hashlib.sha256(b"l2 layer").hexdigest()
        finally:
            unit_test_vm_manager._vms.pop(pending.vm_id, None)
            unit_test_vm_manager._pending_launches.pop(pending.vm_id, None)


class TestLaunchLeases:
    """The launch's own shares of the caller's leases, and the digests of its images."""

    async def test_shares_are_all_taken_or_none(self, tmp_path: Path) -> None:
        """A share that fails (a released lease) leaves no share of the earlier leases."""
        held, gone = tmp_path / "held.qcow2", tmp_path / "gone.qcow2"
        held.write_bytes(b"x")
        gone.write_bytes(b"x")
        first, second = await CacheEntry(held).lease(), await CacheEntry(gone).lease()
        assert first is not None
        assert second is not None
        second.release()
        with pytest.raises(ValueError, match="already released"):
            _share_all((first, second))
        first.release()
        assert evictable(held)  # no share of `first` was left behind

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file of mode 000")
    def test_unreadable_image_is_a_local_disk_error(self, tmp_path: Path) -> None:
        """A guest image this process cannot read (here: no access) fails the launch with
        LOCAL_DISK, not a raw OSError out of create_vm."""
        base = tmp_path / "base.qcow2"
        base.write_bytes(b"base")
        base.chmod(0)
        try:
            with pytest.raises(
                TransientError,
                check=lambda e: e.code is ErrorCode.LOCAL_DISK and e.context["errno"] == errno.EACCES,
                match=r"^Guest image base\.qcow2 cannot be read: Permission denied$",
            ):
                guest_digests(base, None)
        finally:
            base.chmod(0o600)


_PDEATHSIG_PROBE = "import ctypes; v = ctypes.c_int(); ctypes.CDLL(None).prctl(2, ctypes.byref(v)); print(v.value)"
"""Prints the parent-death signal of its process (2 = PR_GET_PDEATHSIG)."""


async def _launch_for_test(manager: VmManager, infra: Any, pending: Any, reservation: Any, qemu_cmd: list[str]) -> None:
    """_launch_qemu_vm of `qemu_cmd` for the `_pending` launch of TestPendingLaunchOwnership."""
    await manager._launch_qemu_vm(
        infra,
        pending,
        qemu_cmd=qemu_cmd,
        spec=VmSpec(language=Language.PYTHON),
        packages_image=None,
        reservation=reservation,
        on_boot_log=None,
        gvproxy_proc=None,
        gvproxy_log_task=None,
        pass_fds=(),
        cache_leases=(),
    )


class TestThisUserQemuEndsWithUs:
    """A QEMU that runs as this user, maybe under unshare (which outlives a killed host process
    and keeps QEMU, its child): the first program gets a parent-death signal, with no
    Python preexec_fn where setpriv works."""

    @staticmethod
    async def _run(cmd: list[str], steps: list[tuple[str, str]]) -> tuple[int | None, bytes, bytes]:
        """Run `cmd` through start_managed_process, as the launch of a QEMU of this user does:
        its exit code, stdout and stderr."""
        proc = await start_managed_process(
            cmd, steps=steps, end_with_this_process=True, pass_fds=(), on_process_started=None
        )
        try:
            out, err = await proc.communicate()
        finally:
            unregister_process(proc)
        return proc.returncode, out, err

    @skip_unless_linux
    async def test_first_program_gets_sigkill_when_this_process_ends(self) -> None:
        _, out, _ = await self._run([sys.executable, "-I", "-c", _PDEATHSIG_PROBE], [])
        assert int(out) == signal.SIGKILL

    @pytest.mark.parametrize("refused", [False, True], ids=["joined", "refused"])
    async def test_joins_its_cgroup_and_runs_also_when_refused(self, tmp_path: Path, refused: bool) -> None:
        """The child joins the VM cgroup before it execs. A refused move does not stop it (the
        attach after the spawn tries again) and prints nothing to the console."""
        procs, marker = tmp_path / "cgroup.procs", tmp_path / "ran"
        if refused:
            procs.mkdir()  # every open for write fails
        else:
            procs.write_bytes(b"")
        returncode, _, stderr = await self._run(["/bin/sh", "-c", f"touch {marker}"], [Cgroup(tmp_path).join_step()])
        assert (returncode, stderr) == (0, b"")
        assert marker.exists()
        assert refused or procs.read_bytes() == b"0\n"


class TestStopQemu:
    """_stop_qemu: a QEMU of this user gets SIGKILL first (its drives go with the VM folder: its
    clean SIGTERM exit keeps nothing); a qemu-vm QEMU gets SIGTERM only (sudo does not pass
    SIGKILL on). Real processes that ignore SIGTERM stand in for QEMU."""

    _IGNORES_SIGTERM: Final = ["/bin/sh", "-c", "trap '' TERM; echo ready; while :; do sleep 0.01; done"]
    _EXITS_3_ON_SIGTERM: Final = ["/bin/sh", "-c", "trap 'exit 3' TERM; echo ready; while :; do sleep 0.01; done"]

    @staticmethod
    async def _start(cmd: list[str]) -> Any:
        proc = await start_managed_process(
            cmd, steps=(), end_with_this_process=True, pass_fds=(), on_process_started=None
        )
        assert proc.stdout is not None
        assert await proc.stdout.readline() == b"ready\n"  # the trap is set
        return proc

    async def test_this_user_qemu_ends_at_once_with_sigkill(self) -> None:
        proc = await self._start(self._IGNORES_SIGTERM)
        try:
            async with asyncio.timeout(2):  # SIGTERM first would wait term_timeout (5 s)
                assert await _stop_qemu(proc, vm_id="vm", as_qemu_vm=False) is True
            assert proc.returncode == -signal.SIGKILL
        finally:
            unregister_process(proc)

    @pytest.mark.parametrize(
        ("as_qemu_vm", "returncode"), [(True, 3), (False, -signal.SIGKILL)], ids=["qemu-vm", "this-user"]
    )
    async def test_the_first_signal_follows_who_runs_qemu(self, as_qemu_vm: bool, returncode: int) -> None:
        """SIGTERM first for a qemu-vm QEMU (sudo does not pass SIGKILL on): its handler exits
        with 3. SIGKILL first for a QEMU of this user: no handler runs."""
        proc = await self._start(self._EXITS_3_ON_SIGTERM)
        try:
            assert await _stop_qemu(proc, vm_id="vm", as_qemu_vm=as_qemu_vm) is True
            assert proc.returncode == returncode
        finally:
            unregister_process(proc)


class TestQemuLaunchHasNoPreexec:
    """No Python preexec_fn on a QEMU launch where setpriv works: any preexec_fn makes Popen fork
    this process on the event loop's thread (about 31 ms at 2 GB of RSS), not vfork (0.3-0.4 ms)."""

    @pytest.mark.parametrize("qemu_vm", [False, True], ids=["this-user", "qemu-vm"])
    async def test_launch_spawns_with_no_preexec(self, unit_test_vm_manager: VmManager, qemu_vm: bool) -> None:
        """This user: setpriv sets the parent-death signal, a shell joins the cgroup. qemu-vm: a
        shell joins the cgroup or stops; no setpriv (sudo drops the signal, the cgroup ends QEMU).
        The OS spawn is faked (it fails with EAGAIN): the test reads the argv and the preexec_fn
        that the launch gives it."""
        infra, reservation, pending = TestPendingLaunchOwnership._pending(unit_test_vm_manager)
        infra.workdir.use_qemu_vm_user = qemu_vm
        spawn = AsyncMock(side_effect=OSError(errno.EAGAIN, os.strerror(errno.EAGAIN)))
        try:
            with (
                patch("exec_sandbox.subprocess_utils.detect_host_os", return_value=HostOS.LINUX),
                patch(
                    "exec_sandbox.subprocess_utils.probe_setpriv_pdeathsig", new_callable=AsyncMock, return_value=True
                ),
                patch("exec_sandbox.subprocess_utils.asyncio.create_subprocess_exec", spawn),
                pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED),
            ):
                await _launch_for_test(unit_test_vm_manager, infra, pending, reservation, ["/bin/sh", "-c", "exit 0"])
        finally:
            unit_test_vm_manager._pending_launches.pop(pending.vm_id, None)
        cmd, kwargs = list(spawn.call_args.args), spawn.call_args.kwargs
        assert kwargs["preexec_fn"] is None
        assert cmd[-3:] == ["/bin/sh", "-c", "exit 0"]
        assert "/test/cgroup/pending-launch-vm" in cmd  # the cgroup the shell joins
        assert (cmd[:4] == ["setpriv", "--pdeathsig", "KILL", "--"]) is not qemu_vm


class TestQemuVmNeedsItsCgroup:
    """A qemu-vm QEMU this user cannot signal: only its VM cgroup can end it and prove its
    end, so it never starts outside it. A shell joins, not a preexec_fn."""

    _NOT_JOINABLE = Path(cgroup.CGROUP_V2_BASE_PATH) / "code-exec" / "fxlease-no-such-tenant" / "vm"

    @staticmethod
    async def _run(cgroup_path: Path, cmd: list[str]) -> tuple[int, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *child_shell(cmd, steps=[Cgroup(cgroup_path).join_or_stop_step()]),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        assert proc.returncode is not None
        return proc.returncode, out

    async def test_child_outside_its_cgroup_never_runs_the_command(self, tmp_path: Path) -> None:
        """The child could not join the VM cgroup: it stops before exec (no sudo, no QEMU), and
        prints no joined line."""
        marker = tmp_path / "ran"
        returncode, out = await self._run(self._NOT_JOINABLE, ["/bin/sh", "-c", f"touch {marker}"])
        assert returncode != 0
        assert out == b""
        assert not marker.exists()

    async def test_the_move_is_the_proof(self, tmp_path: Path) -> None:
        """A write to cgroup.procs that succeeds is the join: no /proc/self/cgroup read, which
        a cgroup file system bind-mounted from a subtree (a container without its own cgroup
        namespace) would show with a longer path."""
        marker = tmp_path / "ran"
        (tmp_path / "cgroup.procs").write_bytes(b"")
        assert await self._run(tmp_path, ["/bin/sh", "-c", f"touch {marker}"]) == (0, cgroup.JOINED)
        assert marker.exists()
        assert (tmp_path / "cgroup.procs").read_bytes() == b"0\n"

    @skip_unless_linux
    async def test_a_refused_move_stops_the_child(self, tmp_path: Path) -> None:
        """The kernel refuses the move (the open works, the write fails, as for a process
        outside the delegated subtree): the child stops before exec."""
        marker = tmp_path / "ran"
        (tmp_path / "cgroup.procs").symlink_to("/dev/full")  # every write fails (ENOSPC)
        returncode, out = await self._run(tmp_path, ["/bin/sh", "-c", f"touch {marker}"])
        assert (returncode != 0, out) == (True, b"")
        assert not marker.exists()

    @skip_unless_linux
    async def test_child_in_its_cgroup_runs(self) -> None:
        """The child joined the VM cgroup (a delegated subtree, as on CI): the command runs in it."""
        made = await Cgroup.create(tenant_id="fxlease-join", vm_id="fxlease-join-vm", memory_mb=128, cpu_cores=0.5)
        if made is None:
            pytest.skip("no delegated cgroup v2 subtree on this host")
        cgroup_path = made.path
        try:
            # The cgroup's own process list, not /proc/self/cgroup: a bind-mounted subtree (no
            # cgroup namespace) shows a longer path there although the move worked.
            returncode, out = await self._run(
                cgroup_path, ["/bin/sh", "-c", 'echo $$; cat "$1/cgroup.procs"', "sh", str(cgroup_path)]
            )
            joined, pid, *members = out.decode().split()
            assert (returncode, joined) == (0, "joined")
            assert pid in members
        finally:
            assert await made.close()

    async def test_no_real_cgroup_starts_nothing(self, unit_test_vm_manager: VmManager) -> None:
        """A qemu-vm QEMU and no cgroup on this host: HOST_DEPENDENCY, and no process is spawned
        (the OS spawn is faked to record a call)."""
        infra, reservation, pending = TestPendingLaunchOwnership._pending(unit_test_vm_manager)
        infra.workdir.use_qemu_vm_user = True
        infra.cgroup = None  # this host gave the VM no cgroup
        spawn = AsyncMock()
        try:
            with (
                patch("exec_sandbox.subprocess_utils.asyncio.create_subprocess_exec", spawn),
                pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="could not join"),
            ):
                await _launch_for_test(unit_test_vm_manager, infra, pending, reservation, ["/bin/sh", "-c", "exit 0"])
        finally:
            unit_test_vm_manager._pending_launches.pop(pending.vm_id, None)
        spawn.assert_not_called()
        assert pending.qemu_proc is None

    async def test_qemu_vm_outside_its_cgroup_is_refused(self, unit_test_vm_manager: VmManager, tmp_path: Path) -> None:
        """The launch of a qemu-vm QEMU whose child could not join its cgroup: HOST_DEPENDENCY.
        The child stopped before its command (no sudo, no QEMU ran), `pending` owns it, and
        no VM is published."""
        infra, reservation, pending = TestPendingLaunchOwnership._pending(unit_test_vm_manager)
        infra.workdir.use_qemu_vm_user = True
        infra.cgroup = Cgroup(self._NOT_JOINABLE)
        infra.channel = MagicMock()
        marker = tmp_path / "ran"
        try:
            with (
                patch("exec_sandbox.vm_manager.guest_digests", return_value=(None, None)),
                pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="could not join"),
            ):
                await _launch_for_test(
                    unit_test_vm_manager, infra, pending, reservation, ["/bin/sh", "-c", f"touch {marker}"]
                )
            assert pending.qemu_proc is not None
            assert await pending.qemu_proc.wait() != 0
            assert not marker.exists()
            assert pending.vm_id not in unit_test_vm_manager._vms
        finally:
            if pending.qemu_proc is not None:
                unregister_process(pending.qemu_proc)
            unit_test_vm_manager._vms.pop(pending.vm_id, None)
            unit_test_vm_manager._pending_launches.pop(pending.vm_id, None)


class TestLaunchErrorCodes:
    """The code of each launch failure: the spawn errno decides the class, a missing
    qemu-vm user is the host's, and a boot timeout keeps the console out of the message.

    The spawn cases run _launch_qemu_vm on real binaries. The boot cases run _create_vm_impl
    with its steps stubbed (infra, gvproxy, launch, cleanup): the answer of a QEMU or guest
    (exit 126, a gvproxy that fails late, no guest answer) needs a booted VM through create_vm."""

    @staticmethod
    async def _launch(manager: VmManager, qemu_cmd: list[str]) -> None:
        infra, reservation, pending = TestPendingLaunchOwnership._pending(manager)
        infra.workdir.use_qemu_vm_user = False
        try:
            # a cgroup (_pending): no ulimit shell wraps the command, so the spawn itself fails
            await manager._launch_qemu_vm(
                infra,
                pending,
                qemu_cmd=qemu_cmd,
                spec=VmSpec(language=Language.PYTHON),
                packages_image=None,
                reservation=reservation,
                on_boot_log=None,
                gvproxy_proc=None,
                gvproxy_log_task=None,
                pass_fds=(),
                cache_leases=(),
            )
        finally:
            manager._pending_launches.pop(pending.vm_id, None)

    async def test_start_stops_on_a_qemu_this_host_cannot_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The QEMU binary is for another CPU: start raises HOST_DEPENDENCY that names
        it, before any VM launch (no overlay pool, no kernel check). Behind unshare or stdbuf,
        or under busybox ash, a launch would run it as a shell script (QEMU_CRASHED)."""
        binary = write_foreign_elf(tmp_path / "qemu-system-test")
        for variable in ("EXEC_SANDBOX_QEMU_BIN_X86", "EXEC_SANDBOX_QEMU_BIN_ARM"):
            monkeypatch.setenv(variable, str(binary))
        settings = Settings(base_images_dir=tmp_path, kernel_path=tmp_path, home_disk_dir=tmp_path / "home")
        manager = VmManager(settings)
        try:
            with (
                patch("exec_sandbox.vm_manager.detect_accel_type", new_callable=AsyncMock, return_value=AccelType.TCG),
                patch("exec_sandbox.vm_manager.probe_io_uring_support", new_callable=AsyncMock, return_value=False),
                patch("exec_sandbox.vm_manager.probe_unshare_support", new_callable=AsyncMock, return_value=False),
                patch("exec_sandbox.vm_manager.probe_qemu_sandbox_support", new_callable=AsyncMock, return_value=False),
                patch("exec_sandbox.vm_manager.probe_qemu_as_qemu_vm", new_callable=AsyncMock, return_value=False),
                patch.object(manager._overlay_pool, "start", new_callable=AsyncMock) as pool_start,
                pytest.raises(
                    ConfigError,
                    check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY,
                    match=rf"^Failed to start {re.escape(str(binary))}: ",
                ) as exc_info,
            ):
                await manager.start()
        finally:
            if manager._run_dir is not None:
                manager._run_dir.close()
        assert exc_info.value.context == {"binary": str(binary), "errno": errno.ENOEXEC}
        pool_start.assert_not_awaited()

    async def test_a_context_whose_start_fails_leaves_no_run_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No __aexit__ follows a failed __aenter__: the manager stops itself, and its run folder
        goes now, not at the next start's sweep."""
        binary = write_foreign_elf(tmp_path / "qemu-system-test")
        for variable in ("EXEC_SANDBOX_QEMU_BIN_X86", "EXEC_SANDBOX_QEMU_BIN_ARM"):
            monkeypatch.setenv(variable, str(binary))
        home = tmp_path / "home"
        manager = VmManager(Settings(base_images_dir=tmp_path, kernel_path=tmp_path, home_disk_dir=home))
        with (
            patch("exec_sandbox.vm_manager.detect_accel_type", new_callable=AsyncMock, return_value=AccelType.TCG),
            patch("exec_sandbox.vm_manager.probe_io_uring_support", new_callable=AsyncMock, return_value=False),
            patch("exec_sandbox.vm_manager.probe_unshare_support", new_callable=AsyncMock, return_value=False),
            patch("exec_sandbox.vm_manager.probe_qemu_sandbox_support", new_callable=AsyncMock, return_value=False),
            patch("exec_sandbox.vm_manager.probe_qemu_as_qemu_vm", new_callable=AsyncMock, return_value=False),
            pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY),
        ):
            async with manager:
                pytest.fail("the start must fail")
        assert [p.name for p in home.iterdir()] == []

    async def test_missing_qemu_binary_is_host_dependency(self, unit_test_vm_manager, tmp_path: Path) -> None:
        binary = str(tmp_path / "qemu-system-missing")
        with pytest.raises(
            ConfigError,
            check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY,
            match=r"^Failed to start .+/qemu-system-missing: ",
        ) as exc_info:
            await self._launch(unit_test_vm_manager, [binary])
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)
        assert exc_info.value.context["errno"] == errno.ENOENT
        assert exc_info.value.context["binary"] == binary

    async def test_qemu_binary_without_exec_permission_is_host_dependency(
        self, unit_test_vm_manager, tmp_path: Path
    ) -> None:
        binary = tmp_path / "qemu-system-test"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o644)
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc_info:
            await self._launch(unit_test_vm_manager, [str(binary)])
        assert exc_info.value.context["errno"] == errno.EACCES

    @pytest.mark.parametrize("err", [errno.EAGAIN, errno.ENOMEM], ids=["process-limit", "no-memory"])
    async def test_fork_limit_is_transient(self, unit_test_vm_manager, err: int) -> None:
        """A fork that hits a process or memory limit can pass later: RETRY (and the
        boot retry loop retries it), not FIX_CONFIG."""
        with patch(
            "exec_sandbox.vm_manager.start_managed_process",
            new_callable=AsyncMock,
            side_effect=OSError(err, os.strerror(err)),
        ):
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED) as exc_info:
                await self._launch(unit_test_vm_manager, ["/bin/sh"])  # a command that can start
        assert exc_info.value.context["errno"] == err

    async def test_missing_qemu_vm_user_is_host_dependency(self, unit_test_vm_manager, tmp_path: Path) -> None:
        """Overlay handed to qemu-vm, but no qemu-vm uid for the socket peer check."""
        workdir = MagicMock()
        workdir.overlay_image = tmp_path / "overlay.qcow2"
        workdir.cmd_socket, workdir.event_socket, workdir.qmp_socket = (tmp_path / n for n in ("c", "e", "q"))
        with (
            patch("exec_sandbox.vm_manager.validate_kernel_initramfs", new_callable=AsyncMock),
            patch("exec_sandbox.vm_manager.VmWorkingDirectory.create", new_callable=AsyncMock, return_value=workdir),
            patch("exec_sandbox.vm_manager.detect_accel_type", new_callable=AsyncMock, return_value=AccelType.HVF),
            patch.object(unit_test_vm_manager._overlay_pool, "acquire", new_callable=AsyncMock),
            # a VM with a cgroup
            patch.object(Cgroup, "create", new_callable=AsyncMock, return_value=Cgroup(Path("/test/cgroup/vm"))),
            patch.object(unit_test_vm_manager, "_apply_overlay_permissions", new_callable=AsyncMock, return_value=True),
            patch.object(unit_test_vm_manager, "_prepare_home_disk", new_callable=AsyncMock),
            patch("exec_sandbox.vm_manager.get_qemu_vm_uid", return_value=None),
            patch.object(unit_test_vm_manager, "_cleanup_failed_owner", new_callable=AsyncMock) as cleanup,
        ):
            with pytest.raises(
                ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match=r"^qemu-vm user required"
            ):
                await unit_test_vm_manager._setup_vm_infra(
                    VmSpec(language=Language.PYTHON),
                    "tenant",
                    "task",
                    MagicMock(),
                    base_image=tmp_path / "base.qcow2",
                    home_template=None,
                )
        cleanup.assert_awaited_once()
        unit_test_vm_manager._pending_launches.clear()

    @pytest.mark.parametrize("returncode", [126, 127, 1])
    async def test_qemu_launch_that_ends_with_126_or_127_is_host_dependency(
        self, unit_test_vm_manager, tmp_path: Path, returncode: int
    ) -> None:
        """QEMU ends during boot with 126 or 127: an exec in its launch failed (the launch
        shell runs it, so Popen cannot report it): ConfigError HOST_DEPENDENCY that names the
        configured QEMU binary, with the console (the shell's line) in context; no boot retry.
        Any other code stays a crash (QEMU_CRASHED, RETRY)."""
        qemu = tmp_path / "qemu-system-test"
        qemu.write_bytes(b"")
        unit_test_vm_manager.settings = unit_test_vm_manager.settings.model_copy(
            update={"qemu_bin_x86": qemu, "qemu_bin_arm": qemu}
        )
        console = "sh: 1: exec: /usr/bin/qemu-system-x86_64: Exec format error"
        infra = MagicMock()
        infra.vm_id = "exec-failed-vm"
        infra.workdir.use_qemu_vm_user = False
        vm = MagicMock()
        vm.process.returncode = returncode
        crash = TransientError(
            f"QEMU process died during boot (exit code {returncode})",
            code=ErrorCode.QEMU_CRASHED,
            context={"console_log": console, "exit_code": returncode},
            hint=None,
            instance_id=None,
        )
        vm.wait_for_guest = AsyncMock(side_effect=crash)
        with (
            _fake_guest_port_listeners(),
            patch.object(GuestImages, "base_image", return_value=Path("/tmp/base.qcow2")),
            patch.object(unit_test_vm_manager, "_setup_vm_infra", new_callable=AsyncMock, return_value=infra),
            patch("exec_sandbox.vm_manager.build_qemu_cmd", new_callable=AsyncMock, return_value=["qemu", "-m", "256"]),
            patch.object(
                unit_test_vm_manager,
                "_start_gvproxy_for_vm",
                new_callable=AsyncMock,
                return_value=(None, None, 0, None),
            ),
            patch.object(unit_test_vm_manager, "_launch_qemu_vm", new_callable=AsyncMock, return_value=vm),
            patch.object(unit_test_vm_manager, "_cleanup_failed_owner", new_callable=AsyncMock) as cleanup,
            pytest.raises((ConfigError, TransientError)) as exc_info,
        ):
            await unit_test_vm_manager._create_vm_impl(
                VmSpec(language=Language.PYTHON),
                tenant_id="t",
                task_id="boot",
                reservation=MagicMock(),
                direct_write_target=None,
                expose_ports=None,
                on_boot_log=None,
                snapshot_drive=None,
                boot_timeout_seconds=1,
                cache_leases=(),
                mem_path=None,
                mem_path_share=False,
            )
        cleanup.assert_awaited_once()
        if returncode == 1:
            assert exc_info.value is crash
            return
        error = exc_info.value
        assert error.code is ErrorCode.HOST_DEPENDENCY
        assert str(error).startswith(f"Failed to start {qemu}: exit status {returncode} (")
        assert error.context["console_log"] == console
        assert error.context["qemu_cmd"] == "qemu -m 256"
        assert error.__cause__ is crash

    async def test_a_gvproxy_that_fails_during_the_boot_fails_the_create(self, unit_test_vm_manager) -> None:
        """gvproxy-wrapper gets ready while QEMU boots: the create waits for it once the
        guest answered; its error fails the create, which cleans up."""
        failed = TransientError(
            "gvproxy-wrapper exited with code 1 before it was ready",
            code=ErrorCode.GVPROXY_FAILED,
            context={},
            hint=None,
            instance_id=None,
        )
        order: list[str] = []
        infra = MagicMock()
        infra.vm_id = "gvproxy-late-vm"
        infra.workdir.use_qemu_vm_user = False
        vm = MagicMock()
        vm.process.returncode = None
        vm.wait_for_guest = AsyncMock(side_effect=lambda **_: order.append("guest"))
        vm.transition_state = AsyncMock()

        async def gvproxy_ready() -> None:
            order.append("gvproxy")
            raise failed

        with (
            _fake_guest_port_listeners(),
            patch.object(GuestImages, "base_image", return_value=Path("/tmp/base.qcow2")),
            patch.object(unit_test_vm_manager, "_setup_vm_infra", new_callable=AsyncMock, return_value=infra),
            patch("exec_sandbox.vm_manager.build_qemu_cmd", new_callable=AsyncMock, return_value=["qemu"]),
            patch.object(
                unit_test_vm_manager,
                "_start_gvproxy_for_vm",
                new_callable=AsyncMock,
                return_value=(MagicMock(), None, 1, gvproxy_ready),
            ),
            patch.object(unit_test_vm_manager, "_launch_qemu_vm", new_callable=AsyncMock, return_value=vm),
            patch.object(unit_test_vm_manager, "_cleanup_failed_owner", new_callable=AsyncMock) as cleanup,
            pytest.raises(TransientError) as exc_info,
        ):
            await unit_test_vm_manager._create_vm_impl(
                VmSpec(language=Language.PYTHON, allow_network=True),
                tenant_id="t",
                task_id="boot",
                reservation=MagicMock(),
                direct_write_target=None,
                expose_ports=None,
                on_boot_log=None,
                snapshot_drive=None,
                boot_timeout_seconds=1,
                cache_leases=(),
                mem_path=None,
                mem_path_share=False,
            )
        assert exc_info.value is failed
        assert order == ["guest", "gvproxy"]  # the gvproxy wait overlaps the boot
        vm.transition_state.assert_not_awaited()  # never READY
        cleanup.assert_awaited_once()

    async def test_boot_timeout_is_one_line_with_the_console_in_context(self, unit_test_vm_manager) -> None:
        """The guest never answers: TransientError VM_BOOT_TIMEOUT (CLI exit 124). The
        console stays in context (str(error) is one line)."""
        console = "[    0.100] kernel line\nqemu-system: warning: a host warning"
        infra = MagicMock()
        infra.vm_id = "boot-timeout-vm"
        infra.workdir.use_qemu_vm_user = False
        infra.workdir.overlay_image = Path("/tmp/overlay.qcow2")
        vm = MagicMock()
        vm.wait_for_guest = AsyncMock(side_effect=TimeoutError("Guest agent not ready after 1s"))
        vm.collect_diagnostics = MagicMock(
            return_value=QemuDiagnostics(
                vm_id="boot-timeout-vm",
                exit_code=None,
                signal_name="",
                console_log=console,
                accel_type="hvf",
                host_os="macos",
            )
        )
        with (
            _fake_guest_port_listeners(),
            patch.object(GuestImages, "base_image", return_value=Path("/tmp/base.qcow2")),
            patch.object(unit_test_vm_manager, "_setup_vm_infra", new_callable=AsyncMock, return_value=infra),
            patch("exec_sandbox.vm_manager.build_qemu_cmd", new_callable=AsyncMock, return_value=["qemu", "-m", "256"]),
            patch.object(
                unit_test_vm_manager,
                "_start_gvproxy_for_vm",
                new_callable=AsyncMock,
                return_value=(None, None, 0, None),
            ),
            patch.object(unit_test_vm_manager, "_launch_qemu_vm", new_callable=AsyncMock, return_value=vm),
            patch.object(unit_test_vm_manager, "_cleanup_failed_owner", new_callable=AsyncMock) as cleanup,
        ):
            with pytest.raises(
                TransientError,
                check=lambda e: e.code is ErrorCode.VM_BOOT_TIMEOUT,
                match=r"^Guest agent not ready after 1s \(qemu_running=True\)$",
            ) as exc_info:
                await unit_test_vm_manager._create_vm_impl(
                    VmSpec(language=Language.PYTHON),
                    tenant_id="t",
                    task_id="boot",
                    reservation=MagicMock(),
                    direct_write_target=None,
                    expose_ports=None,
                    on_boot_log=None,
                    snapshot_drive=None,
                    boot_timeout_seconds=1,
                    cache_leases=(),
                    mem_path=None,
                    mem_path_share=False,
                )
        assert exc_info.value.context["console_log"] == console
        assert exc_info.value.context["qemu_cmd"] == "qemu -m 256"
        cleanup.assert_awaited_once()


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
        await validate_kernel_initramfs(
            GuestImages(
                images_dir=vm_settings.kernel_path, kernel_dir=vm_settings.kernel_path, download_dir=None, arch=arch
            )
        )

    async def test_validation_fails_with_fake_path(self) -> None:
        """Validation raises ConfigError ASSET_NOT_FOUND when kernel doesn't exist."""
        arch = detect_host_arch()
        fake_path = Path("/nonexistent/kernels")

        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND, match="Kernel not found"):
            await validate_kernel_initramfs(
                GuestImages(images_dir=fake_path, kernel_dir=fake_path, download_dir=None, arch=arch)
            )

    async def test_a_folder_is_checked_once(self, tmp_path: Path) -> None:
        """A folder that passed is not checked again: its files can go and the check still
        passes, until the cache is cleared."""
        (tmp_path / "vmlinuz-aarch64").write_text("fake-bzimage")
        (tmp_path / "initramfs-aarch64").write_text("fake-cpio")
        images = GuestImages(images_dir=tmp_path, kernel_dir=tmp_path, download_dir=None, arch=HostArch.AARCH64)
        await validate_kernel_initramfs(images)

        for name in ("vmlinuz-aarch64", "initramfs-aarch64"):
            (tmp_path / name).unlink()
        await validate_kernel_initramfs(images)

        clear_kernel_validation_cache()
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND, match="Kernel not found"):
            await validate_kernel_initramfs(images)

    async def test_different_paths_validated_separately(self, vm_settings) -> None:
        """Different kernel paths are cached separately."""
        arch = detect_host_arch()

        # First path succeeds
        await validate_kernel_initramfs(
            GuestImages(
                images_dir=vm_settings.kernel_path, kernel_dir=vm_settings.kernel_path, download_dir=None, arch=arch
            )
        )

        # Different path still gets validated (and fails)
        fake_path = Path("/nonexistent/kernels")
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND, match="Kernel not found"):
            await validate_kernel_initramfs(
                GuestImages(images_dir=fake_path, kernel_dir=fake_path, download_dir=None, arch=arch)
            )

    @pytest.mark.parametrize(
        ("arch", "suffix", "boots"),
        [(HostArch.X86_64, "x86_64", True), (HostArch.AARCH64, "aarch64", False)],
        ids=["x86_64", "aarch64"],
    )
    async def test_vmlinux_only_is_a_kernel_where_it_boots(
        self, tmp_path: Path, arch: HostArch, suffix: str, boots: bool
    ) -> None:
        """vmlinux (PVH direct boot) + initramfs passes on x86_64; aarch64 boots vmlinuz, so a
        lone vmlinux there is no kernel (the check follows images.kernel_names, as QEMU does)."""
        (tmp_path / f"vmlinux-{suffix}").write_text("fake-elf")
        (tmp_path / f"initramfs-{suffix}").write_text("fake-cpio")
        images = GuestImages(images_dir=tmp_path, kernel_dir=tmp_path, download_dir=None, arch=arch)
        if boots:
            await validate_kernel_initramfs(images)
        else:
            with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND):
                await validate_kernel_initramfs(images)

    async def test_validation_accepts_vmlinuz_only(self, tmp_path: Path) -> None:
        """Validation passes with vmlinuz (no vmlinux) + initramfs."""
        arch = detect_host_arch()
        arch_suffix = "aarch64" if arch == HostArch.AARCH64 else "x86_64"

        # Create vmlinuz + initramfs (no vmlinux)
        (tmp_path / f"vmlinuz-{arch_suffix}").write_text("fake-bzimage")
        (tmp_path / f"initramfs-{arch_suffix}").write_text("fake-cpio")

        # Should NOT raise
        await validate_kernel_initramfs(
            GuestImages(images_dir=tmp_path, kernel_dir=tmp_path, download_dir=None, arch=arch)
        )

    async def test_validation_fails_neither_kernel(self, tmp_path: Path) -> None:
        """Validation fails when neither vmlinux nor vmlinuz exists."""
        arch = detect_host_arch()
        arch_suffix = "aarch64" if arch == HostArch.AARCH64 else "x86_64"

        # Only initramfs, no kernel
        (tmp_path / f"initramfs-{arch_suffix}").write_text("fake-cpio")

        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND, match="Kernel not found"):
            await validate_kernel_initramfs(
                GuestImages(images_dir=tmp_path, kernel_dir=tmp_path, download_dir=None, arch=arch)
            )


class TestQemuKernelPathSelection(_QemuCmdTestBase):
    """The kernel and initramfs that the QEMU command boots: vmlinux (PVH direct boot) on x86_64
    when it exists, else vmlinuz; aarch64 always boots vmlinuz."""

    @pytest.mark.parametrize(
        ("arch", "files", "kernel", "initramfs"),
        [
            (HostArch.X86_64, ["vmlinux-x86_64", "vmlinuz-x86_64"], "vmlinux-x86_64", "initramfs-x86_64"),
            (HostArch.X86_64, ["vmlinuz-x86_64"], "vmlinuz-x86_64", "initramfs-x86_64"),
            (HostArch.AARCH64, ["vmlinux-aarch64", "vmlinuz-aarch64"], "vmlinuz-aarch64", "initramfs-aarch64"),
        ],
        ids=["x86-prefers-vmlinux", "x86-falls-back-to-vmlinuz", "arm64-always-vmlinuz"],
    )
    async def test_command_boots_the_selected_kernel(
        self, *, vm_settings, workdir, tmp_path: Path, arch: HostArch, files: list[str], kernel: str, initramfs: str
    ) -> None:
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        for name in files:
            (tmp_path / name).write_text("fake")
        with _qemu_cmd_mocks():
            cmd = await build_qemu_cmd(
                settings=vm_settings.model_copy(update={"kernel_path": tmp_path}),
                arch=arch,
                vm_id="test-vm-kernel",
                workdir=workdir,
                spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=False),
                expose_ports=None,
                direct_write=False,
                debug_boot=False,
                snapshot_drive=None,
                defer_incoming=False,
                qmp_fd=None,
                guest_port_fds=None,
                mem_path=None,
                mem_path_share=False,
            )

        assert _option(cmd, "-kernel") == str(tmp_path / kernel)
        assert _option(cmd, "-initrd") == str(tmp_path / initramfs)


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
    """Integration tests for VmManager with real QEMU VMs. One VM of each image: TestAllImageTypes."""

    async def test_multiple_vms(self, vm_manager) -> None:
        """Two VMs created at once: each READY, with its own id (its cgroup path comes from it)
        and working folder, both tracked by the manager."""
        create_tasks = [
            vm_manager.create_vm(
                VmSpec(language=Language.PYTHON, memory_mb=256, allow_network=False, allowed_domains=None),
                tenant_id="test",
                task_id=f"test-{i}",
                cache_leases=(),
                direct_write_target=None,
                expose_ports=None,
                on_boot_log=None,
                snapshot_drive=None,
                reservation=None,
                retry_profile=constants.RETRY_USER_FACING,
                mem_path=None,
                mem_path_share=False,
            )
            for i in range(2)
        ]

        vms = await asyncio.gather(*create_tasks)

        try:
            assert [vm.state for vm in vms] == [VmState.READY, VmState.READY]
            assert len({vm.vm_id for vm in vms}) == 2
            assert len({vm.workdir.path for vm in vms}) == 2
            assert vm_manager.get_active_vms() == {vm.vm_id: vm for vm in vms}
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
        """Default settings (force_emulation=False) use the hardware accelerator when the host
        has one, never TCG by mistake.

        On macOS: -accel hvf; on Linux with KVM: -accel kvm. -cpu host on arm64, the fixed
        x86 model on x86_64. Without hardware accel: -accel tcg.
        """
        from exec_sandbox.system_probes import check_hvf_available, check_kvm_available, probe_qemu_version

        vm = await vm_manager.create_vm(
            VmSpec(language=Language.RAW, memory_mb=256, allow_network=False, allowed_domains=None),
            tenant_id="test-hwaccel",
            task_id="verify-hwaccel",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
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

            if hvf_available or kvm_available:
                # macOS with HVF uses HVF (Hypervisor.framework); Linux with KVM uses KVM
                # ARM64 HVF on QEMU 11.1+ runs QEMU's own GIC (TestAccelString)
                own_gic = (
                    hvf_available and platform.machine() == "arm64" and (await probe_qemu_version() or (0,)) >= (11, 1)
                )
                expected_accel = "hvf,kernel-irqchip=off" if own_gic else ("hvf" if hvf_available else "kvm")
                assert accel_found == expected_accel, f"Expected {expected_accel}, got: -accel {accel_found}"
                if detect_host_arch() == HostArch.X86_64:
                    # The fixed AVX2-level model (the cross-host contract) and enforce; KVM adds
                    # its PV flags. The flags of each accelerator: TestCpuModelSelection.
                    assert cpu_found.startswith(f"{_X86_MODEL},"), cpu_found
                    assert cpu_found.endswith(",enforce"), cpu_found
                    assert (f",{_X86_KVM_PV}," in cpu_found) == (accel_found == "kvm"), cpu_found
                else:
                    assert cpu_found == "host", f"Expected '-cpu host' with {accel_found}, got: -cpu {cpu_found}"
            else:
                # Fallback to TCG (this is expected in some CI environments)
                assert accel_found.startswith("tcg"), (
                    f"Expected TCG fallback without hardware accel, got: -accel {accel_found}"
                )
        finally:
            await vm_manager.destroy_vm(vm)

    @pytest.mark.parametrize("language,code,expected_output", IMAGE_TEST_CASES)
    async def test_vm_execute_code_all_images(
        self,
        vm_manager,
        execute_timeout,
        language: Language,
        code: str,
        expected_output: str,
    ) -> None:
        """Each image boots to READY (the guest agent answered its ping), runs code with the
        right output, and is DESTROYED after destroy_vm. A cold boot reports each step of
        its start, the overlay and the QEMU command build too."""
        vm = await vm_manager.create_vm(
            VmSpec(language=language, memory_mb=256, allow_network=False, allowed_domains=None),
            tenant_id="test",
            task_id=f"execute-{language.value}",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
        )

        try:
            assert vm.state == VmState.READY
            t = vm.timing
            assert t.setup_ms is not None and t.overlay_ms is not None, t
            assert t.boot_ms is not None and t.guest_wait_ms is not None, t
            assert t.qemu_cmd_build_ms is not None and t.gvproxy_start_ms is not None and t.qemu_fork_ms is not None, t
            assert t.guest_wait_ms > 0, t  # a kernel boot takes far more than 1 ms
            # Each step is part of its phase: the overlay of the setup; the command build, the
            # gvproxy start, the QEMU spawn and the guest wait, one after the other, of the boot
            # (each value rounds to the ms: the sum is at most 2 ms over)
            assert t.overlay_ms <= t.setup_ms + 1, t
            assert t.qemu_cmd_build_ms + t.gvproxy_start_ms + t.qemu_fork_ms + t.guest_wait_ms <= t.boot_ms + 2, t
            result = await vm.execute(
                code=code,
                timeout_seconds=execute_timeout,
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )

            assert result.exit_code == 0, f"Exit code {result.exit_code}, stderr: {result.stderr}"
            assert expected_output in result.stdout, f"Expected '{expected_output}' in stdout: {result.stdout}"
        finally:
            await vm_manager.destroy_vm(vm)
            assert vm.state == VmState.DESTROYED


class TestEmulationMode:
    """Tests with forced software emulation (TCG) to verify emulation code paths.

    These tests use force_emulation=True to bypass KVM/HVF hardware acceleration,
    ensuring the TCG emulation path works correctly. Useful for catching issues
    that only appear in CI environments without hardware virtualization support.
    """

    async def test_emulation_uses_tcg_not_hvf(self, emulation_vm_manager) -> None:
        """force_emulation=True uses TCG, not hardware acceleration. The running QEMU process has:
        1. -accel tcg,... (software emulation), not hvf or kvm
        2. an emulated CPU, never host: neoverse-v1,pauth=off on arm64; on x86_64 the fixed
           model with no KVM PV flags and no +tsc-deadline
        """
        vm = await emulation_vm_manager.create_vm(
            VmSpec(language=Language.RAW, memory_mb=256, allow_network=False, allowed_domains=None),
            tenant_id="test-emulation",
            task_id="verify-tcg",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
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
            assert accel_found.startswith("tcg,"), (
                f"Expected TCG emulation with force_emulation=True, got: -accel {accel_found}"
            )
            if detect_host_arch() == HostArch.AARCH64:
                assert cpu_found == "neoverse-v1,pauth=off", cpu_found
            else:
                assert cpu_found.startswith(f"{_X86_MODEL},"), cpu_found
                assert cpu_found.endswith(",enforce"), cpu_found
                assert _X86_KVM_PV not in cpu_found, cpu_found
                assert "+tsc-deadline" not in cpu_found, cpu_found
        finally:
            await emulation_vm_manager.destroy_vm(vm)

    @pytest.mark.parametrize("language,code,expected_output", IMAGE_TEST_CASES)
    async def test_vm_execute_with_emulation(
        self,
        emulation_vm_manager,
        language: Language,
        code: str,
        expected_output: str,
    ) -> None:
        """Under forced software emulation, each image boots to READY, runs code with the
        right output, and is DESTROYED after destroy_vm."""
        vm = await emulation_vm_manager.create_vm(
            VmSpec(language=language, memory_mb=256, allow_network=False, allowed_domains=None),
            tenant_id="test-emulation",
            task_id=f"emulation-exec-{language.value}",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
        )

        try:
            assert vm.state == VmState.READY
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
            assert vm.state == VmState.DESTROYED

    async def test_two_vcpus_run_with_multi_threaded_tcg(self, emulation_vm_manager) -> None:
        """A 2-vCPU VM under TCG (thread=multi: one host thread per vCPU) boots, sees two
        CPUs, and runs two CPU-bound processes to their end."""
        vm = await emulation_vm_manager.create_vm(
            VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=2, allow_network=False, allowed_domains=None),
            tenant_id="test-emulation",
            task_id="emulation-mttcg",
            cache_leases=(),
            direct_write_target=None,
            expose_ports=None,
            on_boot_log=None,
            snapshot_drive=None,
            reservation=None,
            retry_profile=constants.RETRY_USER_FACING,
            mem_path=None,
            mem_path_share=False,
        )
        code = (
            "import multiprocessing as mp, os\n"
            "def work(n):\n"
            "    return sum(i * i % 7 for i in range(n))\n"
            "with mp.Pool(2) as pool:\n"
            "    print(os.cpu_count(), pool.map(work, [200_000, 200_000]))"
        )
        try:
            result = await vm.execute(code=code, timeout_seconds=120, env_vars=None, on_stdout=None, on_stderr=None)
            assert result.exit_code == 0, result.stderr
            assert result.stdout.strip() == "2 [399999, 399999]", result.stdout
        finally:
            await emulation_vm_manager.destroy_vm(vm)


# ============================================================================
# Unit Tests - TSC_DEADLINE Detection (check_tsc_deadline)
# ============================================================================


def _tsc_host(host_os: HostOS, arch: HostArch) -> ExitStack:
    """check_tsc_deadline on a host of `host_os` and `arch` (this host is either)."""
    stack = ExitStack()
    stack.enter_context(patch("exec_sandbox.system_probes.detect_host_os", return_value=host_os))
    stack.enter_context(patch("exec_sandbox.system_probes.detect_host_arch", return_value=arch))
    return stack


def _sysctl(stdout: bytes, *, returncode: int = 0) -> AsyncMock:
    """A faked spawn of sysctl that prints `stdout` and ends with `returncode`."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, b""))
    return AsyncMock(return_value=proc)


class TestTscDeadlineLinux:
    """TSC_DEADLINE on x86_64 Linux: the tsc_deadline_timer flag of /proc/cpuinfo."""

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            ("fpu vme de pse tsc msr pae mce cx8 apic sep tsc_deadline_timer sse sse2", True),
            ("fpu vme de pse tsc msr pae mce cx8 apic sep sse sse2", False),
        ],
        ids=["with-feature", "without-feature"],
    )
    async def test_cpuinfo_flag(self, flags: str, expected: bool) -> None:
        cpuinfo = f"processor\t: 0\nvendor_id\t: GenuineIntel\nflags\t\t: {flags}\n"
        with (
            _tsc_host(HostOS.LINUX, HostArch.X86_64),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=cpuinfo),
        ):
            assert await check_tsc_deadline() is expected

    async def test_cpuinfo_not_exists(self) -> None:
        """/proc/cpuinfo not existing returns False gracefully."""
        with _tsc_host(HostOS.LINUX, HostArch.X86_64), patch.object(Path, "exists", return_value=False):
            assert await check_tsc_deadline() is False

    async def test_cpuinfo_read_error(self) -> None:
        """OSError reading /proc/cpuinfo returns False gracefully."""
        with (
            _tsc_host(HostOS.LINUX, HostArch.X86_64),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", side_effect=OSError("Permission denied")),
        ):
            assert await check_tsc_deadline() is False


class TestTscDeadlineMacOS:
    """TSC_DEADLINE on macOS: sysctl machdep.cpu.features on Intel Macs; never on ARM64."""

    @pytest.mark.parametrize("host_os", [HostOS.MACOS, HostOS.LINUX], ids=["macos", "linux"])
    async def test_arm64_is_false_without_a_probe(self, host_os: HostOS) -> None:
        """ARM64 has no TSC (another timer): False, whatever the CPU features say, and no
        sysctl or /proc/cpuinfo read."""
        spawn = _sysctl(b"FPU TSC TSC_DEADLINE SSE")
        with (
            _tsc_host(host_os, HostArch.AARCH64),
            patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", spawn),
            patch.object(Path, "read_text", return_value="flags\t\t: tsc_deadline_timer\n") as read,
        ):
            assert await check_tsc_deadline() is False
        spawn.assert_not_called()
        read.assert_not_called()

    @pytest.mark.parametrize(
        ("features", "returncode", "expected"),
        [
            (b"FPU VME DE PSE TSC MSR PAE MCE TSC_DEADLINE SSE", 0, True),
            (b"FPU VME DE PSE TSC MSR PAE MCE SSE SSE2", 0, False),
            (b"", 1, False),
        ],
        ids=["with-tsc-deadline", "without-tsc-deadline", "sysctl-fails"],
    )
    async def test_intel_mac_reads_sysctl(self, features: bytes, returncode: int, expected: bool) -> None:
        spawn = _sysctl(features, returncode=returncode)
        with (
            _tsc_host(HostOS.MACOS, HostArch.X86_64),
            patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", spawn),
        ):
            assert await check_tsc_deadline() is expected
        assert spawn.call_args.args == ("/usr/sbin/sysctl", "-n", "machdep.cpu.features")

    async def test_answer_is_cached(self) -> None:
        """The second call gives the first answer and runs no sysctl."""
        spawn = _sysctl(b"FPU TSC_DEADLINE")
        with (
            _tsc_host(HostOS.MACOS, HostArch.X86_64),
            patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec", spawn),
        ):
            assert await check_tsc_deadline() is True
            assert await check_tsc_deadline() is True
        assert spawn.call_count == 1


# ============================================================================
# Security Tests - Identifier Validation
# ============================================================================


class TestIdentifierValidation:
    """Security tests for tenant_id and task_id validation to prevent injection attacks.

    Through create_vm: each identifier goes into file, socket and cgroup paths. An identifier
    that passes gets to the next check, the kernel (none in /nonexistent: ASSET_NOT_FOUND); one
    that does not raises ValueError first, before any file is made."""

    @staticmethod
    async def _create(manager: VmManager, *, tenant_id: str = "tenant", task_id: str = "task") -> None:
        with patch.object(GuestImages, "base_image", return_value=Path("/nonexistent/base.qcow2")):
            await manager.create_vm(
                VmSpec(language=Language.PYTHON),
                tenant_id=tenant_id,
                task_id=task_id,
                cache_leases=(),
                reservation=MagicMock(),
                direct_write_target=None,
                expose_ports=None,
                on_boot_log=None,
                snapshot_drive=None,
                retry_profile=constants.RETRY_USER_FACING,
                mem_path=None,
                mem_path_share=False,
            )

    async def _rejects(self, manager: VmManager, identifiers: list[str], match: str) -> None:
        for identifier in identifiers:
            with pytest.raises(ValueError, match=rf"^tenant_id {match}"):
                await self._create(manager, tenant_id=identifier)
            with pytest.raises(ValueError, match=rf"^task_id {match}"):
                await self._create(manager, task_id=identifier)
        assert [path.name for path in manager.work_dir.iterdir()] == [".lock"]  # no VM folder, no record

    async def test_valid_identifiers(self, unit_test_vm_manager: VmManager) -> None:
        """Test that valid identifiers are accepted (up to 128 characters)."""
        valid_identifiers = [
            "tenant123",
            "task-1",
            "my_task",
            "UPPERCASE",
            "MixedCase123",
            "a",
            "123",
            "a-b_c",
            "a" * 128,
        ]
        for identifier in valid_identifiers:
            for ids in ({"tenant_id": identifier}, {"task_id": identifier}):
                with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND):
                    await self._create(unit_test_vm_manager, **ids)

    async def test_empty_identifier_rejected(self, unit_test_vm_manager: VmManager) -> None:
        await self._rejects(unit_test_vm_manager, [""], "cannot be empty")

    async def test_identifier_too_long_rejected(self, unit_test_vm_manager: VmManager) -> None:
        await self._rejects(unit_test_vm_manager, ["a" * 129], "too long: 129 > 128")

    async def test_shell_injection_characters_rejected(self, unit_test_vm_manager: VmManager) -> None:
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
        await self._rejects(unit_test_vm_manager, malicious_identifiers, "contains invalid characters")

    async def test_path_traversal_characters_rejected(self, unit_test_vm_manager: VmManager) -> None:
        """Test that path traversal characters are rejected."""
        malicious_identifiers = [
            "../../../etc/passwd",
            "..\\..\\windows",
            "task/../../root",
            "task\\path",
            "/absolute/path",
        ]
        await self._rejects(unit_test_vm_manager, malicious_identifiers, "contains invalid characters")

    async def test_special_characters_rejected(self, unit_test_vm_manager: VmManager) -> None:
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
        await self._rejects(unit_test_vm_manager, special_chars, "contains invalid characters")

    async def test_unicode_characters_rejected(self, unit_test_vm_manager: VmManager) -> None:
        """Test that Unicode characters are rejected."""
        unicode_identifiers = [
            "task日本語",
            "задача",
            "tâche",
            "任务",
            "tenant\u0000null",
        ]
        await self._rejects(unit_test_vm_manager, unicode_identifiers, "contains invalid characters")

    async def test_whitespace_rejected(self, unit_test_vm_manager: VmManager) -> None:
        """Test that whitespace is rejected."""
        whitespace_identifiers = [
            "task 1",
            " task",
            "task ",
            "task\t1",
            "task\n1",
            "task\r\n1",
        ]
        await self._rejects(unit_test_vm_manager, whitespace_identifiers, "contains invalid characters")

    async def test_trailing_newline_rejected(self, unit_test_vm_manager: VmManager) -> None:
        """A newline at the end goes into the VM folder, socket and cgroup paths too."""
        await self._rejects(unit_test_vm_manager, ["task\n"], "contains invalid characters")


# ============================================================================
# Execute State Tests
# ============================================================================


class TestExecuteState:
    """execute() runs only on a READY VM."""

    @pytest.mark.parametrize("state", [VmState.DESTROYING, VmState.DESTROYED], ids=["destroying", "destroyed"])
    async def test_execute_rejects_a_vm_that_is_not_ready(self, state: VmState) -> None:
        """execute() raises InternalError INTERNAL when the VM is not READY; nothing is sent."""
        channel = _RecordingCompletionChannel()
        vm = TestExecuteExceptionWrapping._make_vm(channel, state)

        with pytest.raises(
            InternalError,
            check=lambda e: e.code is ErrorCode.INTERNAL,
            match=rf"^Cannot execute in state {state.value}, must be READY$",
        ):
            await vm.execute(code="print('hello')", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert channel.requests == []


# ============================================================================
# Execute Env Var Validation Tests
# ============================================================================


class TestExecuteEnvVarValidation:
    """Verify execute() wraps ValidationError as InvalidInputError ENV_VAR_INVALID.

    Users catching SandboxError must catch invalid env vars. Without the
    conversion, raw pydantic.ValidationError leaks through.
    """

    @pytest.fixture
    def channel(self) -> "_RecordingCompletionChannel":
        return _RecordingCompletionChannel()

    @pytest.fixture
    def vm(self, channel: "_RecordingCompletionChannel") -> Iterator[QemuVM]:
        """A READY QemuVM. An invalid request sends nothing and leaves the VM READY (the
        session can go on)."""
        vm = TestExecuteExceptionWrapping._make_vm(channel)
        yield vm
        assert channel.requests == []
        assert vm.state is VmState.READY

    # -- Control characters ------------------------------------------------

    @pytest.mark.parametrize(
        "env_vars, match",
        [
            pytest.param({"K": "v\n"}, "control character", id="newline-value"),
            pytest.param({"K": "v\x00"}, "control character", id="null-value"),
            pytest.param({"K": "\x1b[31m"}, "control character", id="esc-value"),
            pytest.param({"K": "v\r"}, "control character", id="cr-value"),
            pytest.param({"K": "v\x07"}, "control character", id="bell-value"),
            pytest.param({"K": "v\x7f"}, "control character", id="del-value"),
            pytest.param({"K\x00": "v"}, "control character", id="null-name"),
            pytest.param({"K\n": "v"}, "control character", id="newline-name"),
        ],
    )
    async def test_control_chars_raise_env_var_validation_error(self, vm, env_vars: dict[str, str], match: str) -> None:
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=match
        ) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=env_vars, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value, SandboxError)
        assert exc_info.value.__cause__ is None  # the ValidationError holds every env var value
        assert exc_info.value.context["vm_id"] == "test-exec-wrap"

    # -- Size / count limits -----------------------------------------------

    @pytest.mark.parametrize(
        "env_vars, match",
        [
            pytest.param({"": "v"}, "length", id="empty-name"),
            pytest.param({"A" * 257: "v"}, "length", id="name-too-long"),
            pytest.param({"K": "x" * 4097}, "too large", id="value-too-long"),
            pytest.param(
                {f"V{i}": "v" for i in range(101)},
                "(?i)too many",
                id="too-many-vars",
            ),
        ],
    )
    async def test_limits_raise_env_var_validation_error(self, vm, env_vars: dict[str, str], match: str) -> None:
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=match
        ) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=env_vars, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value, SandboxError)
        assert exc_info.value.__cause__ is None  # the ValidationError holds every env var value

    async def test_error_never_holds_an_env_var_value(self, vm) -> None:
        """The error names the field and the reason, never a value: the other env vars are
        often credentials, and the CLI prints the message."""
        secret = "sk-live-0123456789abcdef"  # noqa: S105 - a fake key the error must not show
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID) as exc_info:
            await vm.execute(
                code="x", timeout_seconds=5, env_vars={"API_KEY": secret, "BAD": "v\n"}, on_stdout=None, on_stderr=None
            )
        error = exc_info.value
        rendered = f"{error!s} {error!r} {error.context} {error.hint} {error.__cause__!r}"
        assert "control character" in str(error)
        for fragment in (secret, secret[:5], secret[-6:]):
            assert fragment not in rendered

    # -- Boundary values that must NOT raise --------------------------------

    @pytest.mark.parametrize(
        "env_vars",
        [
            pytest.param({"K": "v\t"}, id="tab-allowed"),
            pytest.param({"K": "Hello 世界 🌍"}, id="utf8-allowed"),
            pytest.param({"A" * 256: "v"}, id="name-at-limit"),
            pytest.param({"K": "x" * 4096}, id="value-at-limit"),
            pytest.param(
                {f"V{i}": "v" for i in range(100)},
                id="vars-at-limit",
            ),
        ],
    )
    async def test_valid_env_vars_no_error(self, env_vars: dict[str, str]) -> None:
        """Valid env vars pass validation: the request reaches the guest with them, and
        the execution returns its result."""
        channel = _RecordingCompletionChannel()
        vm = TestExecuteExceptionWrapping._make_vm(channel)
        result = await vm.execute(code="x", timeout_seconds=5, env_vars=env_vars, on_stdout=None, on_stderr=None)
        assert result.exit_code == 0
        assert [request.env_vars for request in channel.requests] == [env_vars]

    # -- Weird / adversarial ------------------------------------------------

    @pytest.mark.parametrize(
        "env_vars",
        [
            pytest.param({"K": "\n"}, id="value-only-newline"),
            pytest.param({"K": "\x00"}, id="value-only-null"),
            pytest.param({"K": "\nfoo"}, id="control-at-start"),
            pytest.param({"K": "foo\n"}, id="control-at-end"),
            pytest.param({"K": "fo\no"}, id="control-in-middle"),
            pytest.param({"K": "\n\r\x07"}, id="multiple-control-chars"),
        ],
    )
    async def test_adversarial_values_raise_env_var_validation_error(self, vm, env_vars: dict[str, str]) -> None:
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID):
            await vm.execute(code="x", timeout_seconds=5, env_vars=env_vars, on_stdout=None, on_stderr=None)


# ============================================================================
# Unit Tests - execute() exception wrapping
# ============================================================================


class _FakeChannelBase:
    """Common no-op surface so fakes survive the QEMU-death race path,
    which calls fail_pending_operations on whichever channel is wired."""

    async def fail_pending_operations(self, _reason: str) -> None:
        return


class _ConnectRaisesChannel(_FakeChannelBase):
    """connect() raises immediately; stream_messages() never called."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def connect(self, timeout_seconds: float) -> None:
        raise self._error

    async def stream_messages(
        self,
        request: object,
        timeout: int,
    ) -> AsyncGenerator:
        yield  # type: ignore[unreachable]  # make it an async generator


class _ConnectOKStreamRaisesChannel(_FakeChannelBase):
    """connect() succeeds; stream_messages() raises immediately."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def connect(self, timeout_seconds: float) -> None:
        pass

    async def stream_messages(
        self,
        request: object,
        timeout: int,
    ) -> AsyncGenerator:
        raise self._error
        yield  # type: ignore[unreachable]  # make it an async generator


class _HardDeadlineChannel:
    """Connected channel whose total response-stream deadline expired."""

    async def connect(self, timeout_seconds: float) -> None:
        pass

    async def stream_messages(
        self,
        request: object,
        timeout: int,
    ) -> AsyncGenerator:
        from exec_sandbox.guest_channel import StreamDeadlineExceededError

        raise StreamDeadlineExceededError("total stream deadline")
        yield  # type: ignore[unreachable]


class _GuestErrorChannel:
    """connect() succeeds; stream_messages() yields a StreamingErrorMessage."""

    def __init__(self, error_type: str, message: str) -> None:
        self._error_type = error_type
        self._message = message

    async def connect(self, timeout_seconds: float) -> None:
        pass

    async def stream_messages(
        self,
        request: object,
        timeout: int,
    ) -> AsyncGenerator:
        from exec_sandbox.guest_agent_protocol import StreamingErrorMessage

        yield StreamingErrorMessage(error_type=self._error_type, message=self._message)


class _GuestErrorAfterOutputChannel:
    """connect() succeeds; stream yields output chunks then a StreamingErrorMessage."""

    def __init__(self, stdout_chunks: list[str], error_type: str, message: str) -> None:
        self._stdout_chunks = stdout_chunks
        self._error_type = error_type
        self._message = message

    async def connect(self, timeout_seconds: float) -> None:
        pass

    async def stream_messages(
        self,
        request: object,
        timeout: int,
    ) -> AsyncGenerator:
        from exec_sandbox.guest_agent_protocol import OutputChunkMessage, StreamingErrorMessage

        for chunk in self._stdout_chunks:
            yield OutputChunkMessage(type="stdout", chunk=chunk)
        yield StreamingErrorMessage(error_type=self._error_type, message=self._message)


class _BlockingExecutionChannel:
    """Connected channel whose execution stream remains pending until cancelled."""

    def __init__(self) -> None:
        self.cancelled = asyncio.Event()
        self.failed = asyncio.Event()

    async def connect(self, timeout_seconds: float) -> None:
        pass

    async def stream_messages(
        self,
        request: object,
        timeout: int,
    ) -> AsyncGenerator:
        try:
            await self.failed.wait()
            raise OutcomeUnknownError(
                "event transport closed",
                code=ErrorCode.GUEST_OUTCOME_UNKNOWN,
                context=None,
                hint=None,
                instance_id=None,
            )
            yield  # type: ignore[unreachable]
        finally:
            self.cancelled.set()

    async def fail_pending_operations(self, _reason: str) -> None:
        self.failed.set()


class _ReleasedCompletionChannel:
    """Yield a valid terminal event when the process waiter releases it."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def connect(self, timeout_seconds: float) -> None:
        pass

    async def stream_messages(
        self,
        request: object,
        timeout: int,
    ) -> AsyncGenerator:
        from exec_sandbox.guest_agent_protocol import ExecutionCompleteMessage

        await self.release.wait()
        yield ExecutionCompleteMessage(exit_code=137, execution_time_ms=321)

    async def fail_pending_operations(self, _reason: str) -> None:
        self.release.set()


class _RecordingCompletionChannel:
    """Connected channel that keeps each request and answers it with exit code 0."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def connect(self, timeout_seconds: float) -> None:
        pass

    async def stream_messages(
        self,
        request: object,
        timeout: int,
    ) -> AsyncGenerator:
        from exec_sandbox.guest_agent_protocol import ExecutionCompleteMessage

        self.requests.append(request)
        yield ExecutionCompleteMessage(exit_code=0, execution_time_ms=1)

    async def fail_pending_operations(self, _reason: str) -> None:
        pass


class _ImmediateCompletionChannel:
    """Connected channel returning one successful terminal event per request."""

    async def connect(self, timeout_seconds: float) -> None:
        pass

    async def stream_messages(
        self,
        request: object,
        timeout: int,
    ) -> AsyncGenerator:
        from exec_sandbox.guest_agent_protocol import ExecutionCompleteMessage

        yield ExecutionCompleteMessage(exit_code=0, execution_time_ms=1)

    async def fail_pending_operations(self, _reason: str) -> None:
        pass


class TestExecuteExceptionWrapping:
    """Unit tests for execute()'s error handling.

    Covers:
    - Transport errors (OSError, EOF, etc.) → TransientError GUEST_UNAVAILABLE
    - TimeoutError before dispatch → TransientError GUEST_UNAVAILABLE (VM_BOOT_TIMEOUT is the boot's)
    - CancelledError propagates bare
    - Guest error classification: execution_error → TransientError GUEST_UNAVAILABLE (raised),
      timeout_error → exit_code=-1 (returned), others → raised
    """

    @staticmethod
    def _make_vm(channel: object, state: VmState = VmState.READY) -> QemuVM:
        """Build a QemuVM with a mock channel for unit testing execute()."""
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        async def _wait_forever() -> int:
            await asyncio.Event().wait()
            return 0

        mock_process.wait = AsyncMock(side_effect=_wait_forever)

        mock_workdir = MagicMock()
        mock_workdir.overlay_image = Path("/test/overlay.qcow2")

        vm = QemuVM(
            vm_id="test-exec-wrap",
            process=mock_process,
            cgroup=None,  # no cgroup stats I/O
            workdir=mock_workdir,
            channel=channel,  # type: ignore[arg-type]
            spec=VmSpec(language=Language.PYTHON),
            console_lines=deque(maxlen=100),
            base_image=Path("/test/base.qcow2"),
            packages_image=None,
            accel_type=AccelType.KVM,
            gvproxy_proc=None,
            qemu_log_task=None,
            gvproxy_log_task=None,
        )
        vm._state = state  # pyright: ignore[reportPrivateUsage]
        return vm

    # -- Group A: Transport errors → TransientError GUEST_UNAVAILABLE ----------

    async def test_oserror_from_stream(self) -> None:
        """OSError from stream_messages() wrapped as TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(_ConnectOKStreamRaisesChannel(OSError("broken pipe")))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, OSError)

    async def test_incomplete_read_error(self) -> None:
        """IncompleteReadError (EOF) wrapped as TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(_ConnectOKStreamRaisesChannel(asyncio.IncompleteReadError(b"", 100)))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, asyncio.IncompleteReadError)

    async def test_limit_overrun_error(self) -> None:
        """LimitOverrunError (oversized message) wrapped as TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(_ConnectOKStreamRaisesChannel(asyncio.LimitOverrunError("limit", 512)))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, asyncio.LimitOverrunError)

    async def test_json_decode_error(self) -> None:
        """JSONDecodeError wrapped as TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(_ConnectOKStreamRaisesChannel(json.JSONDecodeError("bad", "", 0)))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)

    async def test_validation_error(self) -> None:
        """pydantic.ValidationError wrapped as TransientError GUEST_UNAVAILABLE."""
        try:
            from exec_sandbox.guest_agent_protocol import ExecutionCompleteMessage

            ExecutionCompleteMessage.model_validate({"bad": "data"})
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            pydantic_err = e

        vm = self._make_vm(_ConnectOKStreamRaisesChannel(pydantic_err))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, ValidationError)

    async def test_runtime_error(self) -> None:
        """RuntimeError (channel not connected) wrapped as TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(_ConnectOKStreamRaisesChannel(RuntimeError("Channel not connected")))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    async def test_communication_error(self) -> None:
        """CommunicationError wrapped as TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(_ConnectRaisesChannel(CommunicationError("agent failed", context=None)))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, CommunicationError)

    async def test_socket_auth_error(self) -> None:
        """SocketAuthError (CommunicationError subclass) wrapped as TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(
            _ConnectRaisesChannel(SocketAuthError("auth failed", expected_uid=1000, actual_uid=0, context=None))
        )
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, SocketAuthError)

    # -- Group B: TimeoutError before dispatch → TransientError GUEST_UNAVAILABLE --

    async def test_timeout_from_connect(self) -> None:
        """TimeoutError from connect() of a booted VM: GUEST_UNAVAILABLE, not the boot's VM_BOOT_TIMEOUT."""
        vm = self._make_vm(_ConnectRaisesChannel(TimeoutError("connect timed out")))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, TimeoutError)

    async def test_timeout_from_stream(self) -> None:
        """A nested transport timeout is not mislabeled as the 15s host deadline."""
        vm = self._make_vm(_ConnectOKStreamRaisesChannel(TimeoutError("hard timeout")))
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, TimeoutError)
        assert "transport timed out" in str(exc_info.value)
        assert exc_info.value.context["soft_timeout_seconds"] == 5
        assert exc_info.value.context["hard_timeout_seconds"] == 15
        assert exc_info.value.context["timeout_phase"] == "execution"
        assert exc_info.value.context["timeout_source"] == "nested_transport"

    async def test_true_stream_deadline_is_outcome_unknown_after_dispatch(self) -> None:
        """The 15s total deadline cannot imply that dispatched code did not run."""
        vm = self._make_vm(_HardDeadlineChannel())
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert "15s host deadline" in str(exc_info.value)
        assert exc_info.value.context["soft_timeout_seconds"] == 5
        assert exc_info.value.context["hard_timeout_seconds"] == 15
        assert exc_info.value.context["timeout_margin_seconds"] == 10
        assert exc_info.value.context["timeout_source"] == "total_stream_deadline"

    async def test_timeout_context_has_seconds(self) -> None:
        """TransientError GUEST_UNAVAILABLE context includes timeout_seconds from caller."""
        vm = self._make_vm(_ConnectRaisesChannel(TimeoutError()))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=17, env_vars=None, on_stdout=None, on_stderr=None)
        assert exc_info.value.context["timeout_seconds"] == 17
        assert exc_info.value.context["timeout_phase"] == "connect"

    # -- Group B2: QEMU process death races execution transport ----------------

    async def test_qemu_death_wakes_blocked_execution_before_hard_timeout(self) -> None:
        """A dead QEMU publishes outcome-unknown without waiting for 15 seconds."""
        channel = _BlockingExecutionChannel()
        vm = self._make_vm(channel)
        vm.process.returncode = 17
        vm.process.wait = AsyncMock(return_value=17)

        async with asyncio.timeout(1):
            with pytest.raises(
                OutcomeUnknownError,
                check=lambda e: e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN,
                match="QEMU exited during execution",
            ) as exc_info:
                await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

        assert exc_info.value.context["exit_code"] == 17
        assert exc_info.value.context["hard_timeout_seconds"] == 15
        await asyncio.wait_for(channel.cancelled.wait(), timeout=1)

    async def test_buffered_terminal_event_wins_process_death_race(self) -> None:
        """A valid exit-137 terminal frame is preserved when QEMU powers off."""
        channel = _ReleasedCompletionChannel()
        vm = self._make_vm(channel)
        vm.process.returncode = 0

        async def _exit_and_release() -> int:
            channel.release.set()
            return 0

        vm.process.wait = AsyncMock(side_effect=_exit_and_release)

        result = await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert result.exit_code == 137
        assert result.execution_time_ms == 321
        assert vm.state == VmState.DESTROYING

    async def test_process_exit_one_turn_after_terminal_retires_vm(self) -> None:
        """A just-completed stream cannot publish a dead VM as reusable."""
        vm = self._make_vm(_ImmediateCompletionChannel())
        vm.process.returncode = 0

        async def _exit_next_turn() -> int:
            await asyncio.sleep(0)
            return 0

        vm.process.wait = AsyncMock(side_effect=_exit_next_turn)
        result = await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

        assert result.exit_code == 0
        assert vm.state == VmState.DESTROYING

    async def test_repeated_execution_reuses_one_process_exit_waiter(self) -> None:
        """Successful session reuse cannot retain one subprocess waiter per exec."""
        vm = self._make_vm(_ImmediateCompletionChannel())
        wait_started = asyncio.Event()
        release_exit = asyncio.Event()
        destroying = asyncio.Event()
        begin_destroy = vm.begin_destroy

        async def controlled_exit() -> int:
            wait_started.set()
            await release_exit.wait()
            return 0

        async def observe_destroying() -> None:
            await begin_destroy()
            destroying.set()

        vm.process.wait = AsyncMock(side_effect=controlled_exit)
        vm.begin_destroy = observe_destroying  # type: ignore[method-assign]
        for _ in range(2):
            result = await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
            assert result.exit_code == 0

        assert wait_started.is_set()
        assert vm.process.wait.await_count == 1
        release_exit.set()
        await asyncio.wait_for(destroying.wait(), timeout=1)
        assert vm.state is VmState.DESTROYING

    async def test_caller_cancellation_drains_stream_but_keeps_shared_death_waiter(self) -> None:
        """Cancelling execute cannot leak its stream task or cancel VM death ownership."""
        channel = _BlockingExecutionChannel()
        vm = self._make_vm(channel)
        wait_started = asyncio.Event()
        release_exit = asyncio.Event()
        destroying = asyncio.Event()
        begin_destroy = vm.begin_destroy

        async def controlled_exit() -> int:
            wait_started.set()
            await release_exit.wait()
            return 0

        async def observe_destroying() -> None:
            await begin_destroy()
            destroying.set()

        vm.process.wait = AsyncMock(side_effect=controlled_exit)
        vm.begin_destroy = observe_destroying  # type: ignore[method-assign]
        execution = asyncio.create_task(
            vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        )
        await wait_started.wait()
        execution.cancel()

        with pytest.raises(asyncio.CancelledError):
            await execution
        await asyncio.wait_for(channel.cancelled.wait(), timeout=1)
        release_exit.set()
        await asyncio.wait_for(destroying.wait(), timeout=1)
        assert vm.state is VmState.DESTROYING

    # -- Group C: Must NOT be wrapped — propagate bare -------------------------

    async def test_cancelled_error_propagates(self) -> None:
        """CancelledError propagates bare, not wrapped as TransientError."""
        vm = self._make_vm(_ConnectRaisesChannel(asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    async def test_code_validation_error_propagates(self) -> None:
        """InvalidInputError CODE_INVALID from guest error propagates bare."""
        vm = self._make_vm(_GuestErrorChannel("code_error", "syntax error"))
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.CODE_INVALID):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    async def test_output_limit_error_propagates(self) -> None:
        """InvalidInputError OUTPUT_LIMIT from guest error propagates bare."""
        vm = self._make_vm(_GuestErrorChannel("output_limit_error", "output too large"))
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.OUTPUT_LIMIT):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    # -- Group D: OSError subclasses -------------------------------------------

    async def test_connection_reset_wrapped(self) -> None:
        """ConnectionResetError (OSError subclass) wrapped as TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(_ConnectOKStreamRaisesChannel(ConnectionResetError("peer reset")))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, ConnectionResetError)

    async def test_broken_pipe_wrapped(self) -> None:
        """BrokenPipeError (OSError subclass) wrapped as TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(_ConnectOKStreamRaisesChannel(BrokenPipeError()))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert isinstance(exc_info.value.__cause__, BrokenPipeError)

    # -- Group E: Context and cause verification -------------------------------

    async def test_context_has_vm_id_language_error_type(self) -> None:
        """TransientError GUEST_UNAVAILABLE context includes vm_id, language, and error_type."""
        vm = self._make_vm(_ConnectRaisesChannel(OSError("refused")))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        ctx = exc_info.value.context
        assert ctx["vm_id"] == "test-exec-wrap"
        assert ctx["language"] == Language.PYTHON
        assert ctx["error_type"] == "OSError"

    async def test_cause_chain_set(self) -> None:
        """OSError from connect(): TransientError GUEST_UNAVAILABLE whose __cause__ is the
        original exception (identity, not copy)."""
        original = OSError("the original error")
        vm = self._make_vm(_ConnectRaisesChannel(original))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert exc_info.value.__cause__ is original

    # -- Group F: Guest error classification in _handle_exec_error -------------

    async def test_execution_error_raises_transient(self) -> None:
        """execution_error (REPL spawn failed) raises TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(_GuestErrorChannel("execution_error", "spawn failed"))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert "spawn failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is None

    async def test_timeout_error_returns_result(self) -> None:
        """timeout_error returns exit_code=-1 result, no exception."""
        vm = self._make_vm(_GuestErrorChannel("timeout_error", "exceeded 30s"))
        result = await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert result.exit_code == -1
        assert "[timeout_error]" in result.stderr

    async def test_request_error_raises_internal(self) -> None:
        """request_error raises InternalError GUEST_PROTOCOL."""
        vm = self._make_vm(_GuestErrorChannel("request_error", "invalid JSON"))
        with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    # -- Group G: Partial output before errors ---------------------------------

    async def test_execution_error_after_partial_stdout(self) -> None:
        """execution_error after partial stdout raises TransientError GUEST_UNAVAILABLE (output lost)."""
        vm = self._make_vm(_GuestErrorAfterOutputChannel(["partial"], "execution_error", "OOM"))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    async def test_timeout_after_partial_stdout(self) -> None:
        """timeout_error after partial stdout returns result with collected output."""
        vm = self._make_vm(_GuestErrorAfterOutputChannel(["out1", "out2"], "timeout_error", "timed out"))
        result = await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert result.exit_code == -1
        assert result.stdout == "out1out2"
        assert "timed out" in result.stderr

    # -- Group H: Errors unlikely during execute() but must be handled ---------

    async def test_protocol_error_during_execute_raises(self) -> None:
        """protocol_error raises InternalError GUEST_PROTOCOL."""
        vm = self._make_vm(_GuestErrorChannel("protocol_error", "frame corruption"))
        with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    async def test_io_error_during_execute_raises(self) -> None:
        """io_error raises InvalidInputError GUEST_IO."""
        vm = self._make_vm(_GuestErrorChannel("io_error", "disk full"))
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    async def test_path_error_during_execute_raises(self) -> None:
        """path_error raises InvalidInputError PATH_INVALID."""
        vm = self._make_vm(_GuestErrorChannel("path_error", "no such file"))
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    # -- Group I: Out-of-bounds / adversarial ----------------------------------

    async def test_request_outside_its_limits_is_guest_protocol(self) -> None:
        """A request the host itself builds wrong (timeout above the guest's 300 s: Session
        checks the range first) is a protocol error, never the caller's ARGUMENT_INVALID;
        nothing is sent."""
        channel = _RecordingCompletionChannel()
        vm = self._make_vm(channel)
        with pytest.raises(
            InternalError,
            check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL,
            match=r"^execute: \[request_error\] Invalid request: timeout: ",
        ):
            await vm.execute(code="x", timeout_seconds=301, env_vars=None, on_stdout=None, on_stderr=None)
        assert channel.requests == []

    async def test_unknown_error_type_raises_internal(self) -> None:
        """Unknown error type from newer guest raises InternalError GUEST_PROTOCOL."""
        vm = self._make_vm(_GuestErrorChannel("future_new_error", "from newer guest"))
        with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    async def test_execution_error_empty_message(self) -> None:
        """execution_error with empty message still raises TransientError GUEST_UNAVAILABLE."""
        vm = self._make_vm(_GuestErrorChannel("execution_error", ""))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    async def test_timeout_error_long_message(self) -> None:
        """timeout_error with very long message returns result without exception."""
        long_msg = "x" * 10000
        vm = self._make_vm(_GuestErrorChannel("timeout_error", long_msg))
        result = await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert result.exit_code == -1
        assert long_msg in result.stderr

    # -- Group J: VM state verification ----------------------------------------

    async def test_execution_error_does_not_restore_ready(self) -> None:
        """execution_error does NOT restore READY (only input validation errors do): the VM
        stays EXECUTING and refuses the next request."""
        vm = self._make_vm(_GuestErrorChannel("execution_error", "spawn failed"))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert vm.state is VmState.EXECUTING
        with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.INTERNAL, match=r"^Cannot execute"):
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)

    async def test_timeout_error_leaves_the_vm_ready(self) -> None:
        """timeout_error is a result of the user's code (exit_code -1), not a fault of the VM:
        the VM is READY again and runs the next request."""
        vm = self._make_vm(_GuestErrorChannel("timeout_error", "timed out"))
        result = await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        assert result.exit_code == -1
        assert vm.state is VmState.READY
        assert (
            await vm.execute(code="x", timeout_seconds=5, env_vars=None, on_stdout=None, on_stderr=None)
        ).exit_code == -1


# ============================================================================
# Unit Tests - guest_error_to_exception mapping
# ============================================================================


class TestGuestErrorToException:
    """Unit tests for guest_error_to_exception() centralized error mapping."""

    @pytest.mark.parametrize(
        "error_type, expected_cls, expected_code",
        [
            pytest.param("env_var_error", InvalidInputError, ErrorCode.ENV_VAR_INVALID, id="env_var"),
            pytest.param("code_error", InvalidInputError, ErrorCode.CODE_INVALID, id="code"),
            pytest.param("path_error", InvalidInputError, ErrorCode.PATH_INVALID, id="path"),
            pytest.param("package_error", InvalidInputError, ErrorCode.ARGUMENT_INVALID, id="package"),
            pytest.param("io_error", InvalidInputError, ErrorCode.GUEST_IO, id="io"),
            pytest.param("output_limit_error", InvalidInputError, ErrorCode.OUTPUT_LIMIT, id="output_limit"),
            pytest.param("execution_error", TransientError, ErrorCode.GUEST_UNAVAILABLE, id="execution"),
            pytest.param("timeout_error", TransientError, ErrorCode.GUEST_UNAVAILABLE, id="timeout"),
            pytest.param("request_error", InternalError, ErrorCode.GUEST_PROTOCOL, id="request"),
            pytest.param("protocol_error", InternalError, ErrorCode.GUEST_PROTOCOL, id="protocol"),
            pytest.param("unknown_future_type", InternalError, ErrorCode.GUEST_PROTOCOL, id="unknown"),
        ],
    )
    def test_error_type_mapping(self, error_type: str, expected_cls: type, expected_code: ErrorCode) -> None:
        """Each error_type maps to the correct exception class and code."""
        from exec_sandbox.guest_agent_protocol import StreamingErrorMessage
        from exec_sandbox.qemu_vm import guest_error_to_exception

        msg = StreamingErrorMessage(type="error", message="test error", error_type=error_type)
        exc = guest_error_to_exception(msg, "vm-123", operation="read_file '/foo'")

        assert type(exc) is expected_cls
        assert exc.code is expected_code
        assert error_type in str(exc)
        assert "test error" in str(exc)

    def test_operation_in_message(self) -> None:
        """Operation name appears in the exception message."""
        from exec_sandbox.guest_agent_protocol import StreamingErrorMessage
        from exec_sandbox.qemu_vm import guest_error_to_exception

        msg = StreamingErrorMessage(type="error", message="not found", error_type="io_error")
        exc = guest_error_to_exception(msg, "vm-456", operation="read_file '/data.txt'")

        assert "read_file '/data.txt'" in str(exc)

    def test_context_fields(self) -> None:
        """Context dict contains expected fields."""
        from exec_sandbox.guest_agent_protocol import StreamingErrorMessage
        from exec_sandbox.qemu_vm import guest_error_to_exception

        msg = StreamingErrorMessage(type="error", message="bad path", error_type="path_error")
        exc = guest_error_to_exception(msg, "vm-789", operation="write_file '/evil'")

        assert exc.context["vm_id"] == "vm-789"
        assert exc.context["error_type"] == "path_error"
        assert exc.context["guest_message"] == "bad path"
        assert exc.context["operation"] == "write_file '/evil'"

    @pytest.mark.parametrize(
        "error_type_enum, expected_code",
        [
            pytest.param(GuestErrorType.ENV_VAR, ErrorCode.ENV_VAR_INVALID, id="env_var"),
            pytest.param(GuestErrorType.CODE, ErrorCode.CODE_INVALID, id="code"),
            pytest.param(GuestErrorType.REQUEST, ErrorCode.GUEST_PROTOCOL, id="request"),
        ],
    )
    def test_enum_value_round_trip(self, error_type_enum: GuestErrorType, expected_code: ErrorCode) -> None:
        """GuestErrorType enum .value round-trips through StreamingErrorMessage.

        str(GuestErrorType.ENV_VAR) gives 'GuestErrorType.ENV_VAR', which matches no case
        branch: a message carries .value ('env_var_error').
        """
        from exec_sandbox.guest_agent_protocol import StreamingErrorMessage
        from exec_sandbox.qemu_vm import guest_error_to_exception

        msg = StreamingErrorMessage(type="error", message="test", error_type=error_type_enum.value)
        exc = guest_error_to_exception(msg, "vm-roundtrip", operation="execute")
        assert exc.code is expected_code


# ============================================================================
# Unit Tests - wait_for_guest task racing
# ============================================================================


class TestWaitForGuest:
    """Tests for QemuVM.wait_for_guest() death_task/guest_task racing logic.

    Verifies correct behavior when QEMU dies, guest becomes ready,
    or timeouts fire during boot wait.
    """

    @staticmethod
    def _make_vm(
        process_wait_event: asyncio.Event | None = None,
        returncode: int = 0,
        connect_side_effect: object = None,
        send_request_side_effect: object = None,
        accel_type: AccelType = AccelType.KVM,
    ) -> QemuVM:
        """Build a real QemuVM with mocked process and channel.

        Args:
            process_wait_event: If set, process.wait() blocks until event is set.
                If None, process.wait() returns immediately (QEMU exits).
            returncode: Process return code after exit.
            connect_side_effect: Side effect for channel.connect().
            send_request_side_effect: Side effect for channel.send_request().
            accel_type: Accelerator of the VM (its diagnostics name it).
        """
        from exec_sandbox.guest_agent_protocol import PongMessage

        mock_process = AsyncMock()
        if process_wait_event is not None:

            async def _blocking_wait() -> None:
                await process_wait_event.wait()

            mock_process.wait = AsyncMock(side_effect=_blocking_wait)
        mock_process.returncode = returncode
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        mock_channel = AsyncMock()
        if connect_side_effect is not None:
            mock_channel.connect = AsyncMock(side_effect=connect_side_effect)
        if send_request_side_effect is not None:
            mock_channel.send_request = AsyncMock(side_effect=send_request_side_effect)
        else:
            mock_channel.send_request = AsyncMock(return_value=PongMessage(version="1.0"))

        mock_workdir = MagicMock()
        mock_workdir.overlay_image = Path("/test/overlay.qcow2")

        return QemuVM(
            vm_id="test-wait-for-guest",
            process=mock_process,
            cgroup=None,
            workdir=mock_workdir,
            channel=mock_channel,
            spec=VmSpec(language=Language.PYTHON),
            console_lines=deque(maxlen=100),
            base_image=Path("/test/base.qcow2"),
            packages_image=None,
            accel_type=accel_type,
            gvproxy_proc=None,
            qemu_log_task=None,
            gvproxy_log_task=None,
        )

    async def _wait_with_alive_process(self, vm: QemuVM, alive: asyncio.Event, timeout: float = 5) -> None:
        """Call wait_for_guest, ensuring the mock process is unblocked on exit."""
        try:
            await vm.wait_for_guest(timeout=timeout)
        finally:
            alive.set()

    # ------------------------------------------------------------------
    # Normal: guest becomes ready
    # ------------------------------------------------------------------

    async def test_guest_ready_immediately(self) -> None:
        """Guest responds to ping on first attempt: one ping, then wait_for_guest returns."""
        alive = asyncio.Event()
        vm = self._make_vm(process_wait_event=alive)
        await self._wait_with_alive_process(vm, alive)
        assert vm.channel.send_request.await_count == 1  # type: ignore[attr-defined]

    async def test_guest_ready_after_retries(self) -> None:
        """Guest fails with retryable errors then succeeds: three pings."""
        from exec_sandbox.guest_agent_protocol import PongMessage

        alive = asyncio.Event()
        attempts = iter(
            [
                OSError("connection refused"),
                OSError("connection refused"),
                PongMessage(version="1.0"),
            ]
        )

        def side_effect(*_args, **_kwargs):
            result = next(attempts)
            if isinstance(result, Exception):
                raise result
            return result

        vm = self._make_vm(process_wait_event=alive, send_request_side_effect=side_effect)
        await self._wait_with_alive_process(vm, alive)
        assert next(attempts, None) is None  # each attempt was used: two retries, then the pong

    async def test_a_ping_that_raises_stop_async_iteration_is_not_ready(self) -> None:
        """A ping that raises StopAsyncIteration (an exhausted async iterator) fails the wait at
        once with RuntimeError. tenacity raises an attempt's exception from __anext__, where
        StopAsyncIteration ends the retry loop as if the ping had answered: the guest was
        "ready" with no pong."""
        alive = asyncio.Event()
        vm = self._make_vm(process_wait_event=alive, send_request_side_effect=StopAsyncIteration())
        with pytest.raises(
            RuntimeError,
            check=lambda e: isinstance(e.__cause__, StopAsyncIteration),
            match=r"^A retried call raised StopAsyncIteration$",
        ):
            await self._wait_with_alive_process(vm, alive)
        assert vm.channel.send_request.await_count == 1  # type: ignore[attr-defined]  # not retried

    # ------------------------------------------------------------------
    # QEMU crashes — process exits before guest is ready
    # ------------------------------------------------------------------

    @patch("exec_sandbox.qemu_vm.detect_host_os", return_value=HostOS.LINUX)
    async def test_qemu_crash_raises_qemu_crashed(self, _mock_os) -> None:
        """QEMU dies with non-zero exit code during boot: one line names the exit code; the
        console (QEMU stderr lines, kernel) is in context only."""
        # Guest must not respond instantly, otherwise it wins the race
        vm = self._make_vm(returncode=1, connect_side_effect=OSError("connection refused"))
        vm.console_lines.extend(["qemu-system: cannot allocate memory", "[    0.1] kernel line"])

        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.QEMU_CRASHED,
            match=r"^QEMU process died during boot \(exit code 1\)$",
        ) as exc_info:
            await vm.wait_for_guest(timeout=5)
        assert exc_info.value.context["console_log"] == "qemu-system: cannot allocate memory\n[    0.1] kernel line"

    async def test_qemu_crash_signal_raises_qemu_crashed(self) -> None:
        """QEMU killed by signal (negative return code)."""
        vm = self._make_vm(returncode=-9, connect_side_effect=OSError("connection refused"))

        with pytest.raises(
            TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED, match=r"\(exit code -9, SIGKILL\)$"
        ):
            await vm.wait_for_guest(timeout=5)

    @patch("exec_sandbox.qemu_vm.detect_host_os", return_value=HostOS.MACOS)
    async def test_macos_clean_exit_during_boot_raises_qemu_crashed(self, _mock_os) -> None:
        """macOS QEMU exits cleanly (code 0) during boot → must raise, not CancelledError.

        A death monitor that returned on this exit would leave the wait on a cancelled
        guest task. Verifies:
        - TransientError QEMU_CRASHED is raised (not CancelledError)
        - Error message contains "clean exit"
        - Error context includes host_os for diagnostics
        """
        vm = self._make_vm(returncode=0, connect_side_effect=OSError("connection refused"))

        with pytest.raises(
            TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED, match="clean exit"
        ) as exc_info:
            await vm.wait_for_guest(timeout=5)

        assert exc_info.value.context.get("host_os") == "macos"

    @patch("exec_sandbox.qemu_vm.detect_host_os", return_value=HostOS.LINUX)
    async def test_tcg_clean_exit_during_boot_raises_qemu_crashed(self, _mock_os) -> None:
        """TCG exits with code 0 during boot (ARM64 timing race)."""
        vm = self._make_vm(returncode=0, connect_side_effect=OSError("connection refused"), accel_type=AccelType.TCG)

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED):
            await vm.wait_for_guest(timeout=5)

    # ------------------------------------------------------------------
    # Timeout — guest never becomes ready
    # ------------------------------------------------------------------

    async def test_timeout_when_guest_never_ready(self) -> None:
        """Guest never responds, timeout fires."""
        alive = asyncio.Event()
        vm = self._make_vm(
            process_wait_event=alive,
            connect_side_effect=OSError("connection refused"),
        )

        with pytest.raises(TimeoutError):
            await self._wait_with_alive_process(vm, alive, timeout=0.5)

    # ------------------------------------------------------------------
    # Edge: non-retryable guest error
    # ------------------------------------------------------------------

    async def test_non_retryable_guest_error_propagates(self) -> None:
        """Guest check raises non-retryable error (not in tenacity retry list)."""
        alive = asyncio.Event()
        vm = self._make_vm(
            process_wait_event=alive,
            send_request_side_effect=ValueError("unexpected protocol error"),
        )

        with pytest.raises(ValueError, match="unexpected protocol error"):
            await self._wait_with_alive_process(vm, alive)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @patch("exec_sandbox.qemu_vm.detect_host_os", return_value=HostOS.LINUX)
    async def test_collect_diagnostics_signal(self, _mock_os) -> None:
        """collect_diagnostics reports SIGKILL for exit code -9."""
        vm = self._make_vm(returncode=-9)
        diag = vm.collect_diagnostics()
        assert diag.signal_name == "SIGKILL"
        assert diag.exit_code == -9
        assert diag.accel_type == "kvm"
        assert diag.host_os == "linux"

    @patch("exec_sandbox.qemu_vm.detect_host_os", return_value=HostOS.MACOS)
    async def test_collect_diagnostics_fields(self, _mock_os) -> None:
        """collect_diagnostics produces expected fields and never reads the QEMU pipes:
        the drain task is their only reader (its lines are in the console buffer)."""
        vm = self._make_vm(returncode=0, accel_type=AccelType.HVF)
        vm.console_lines.append("boot line 1")
        diag = vm.collect_diagnostics()

        vm.process.communicate.assert_not_awaited()
        # Verify fields
        assert diag.vm_id == "test-wait-for-guest"
        assert diag.exit_code == 0
        assert diag.signal_name == ""
        assert "boot line 1" in diag.console_log
        assert diag.accel_type == "hvf"
        assert diag.host_os == "macos"

    @patch("exec_sandbox.qemu_vm.detect_host_os", return_value=HostOS.MACOS)
    async def test_collect_diagnostics_names_forced_emulation(self, _mock_os) -> None:
        """A VM that runs on TCG (force_emulation) on a host with HVF reports tcg: the
        accelerator of its own QEMU command, not the one the host could use."""
        vm = self._make_vm(returncode=0, accel_type=AccelType.TCG)
        assert vm.collect_diagnostics().accel_type == "tcg"

    # ------------------------------------------------------------------
    # CancelledError from guest_task (child task cancellation)
    # ------------------------------------------------------------------

    async def test_guest_cancelled_error_retried_then_succeeds(self) -> None:
        """Guest check raises CancelledError (transient), retried, then succeeds: two pings.

        A CancelledError from a child task (guest_task) escapes asyncio.timeout(): the
        conversion to TimeoutError only applies to the current task's cancellation.
        wait_for_guest wraps it as RuntimeError, so tenacity retries.
        """
        from exec_sandbox.guest_agent_protocol import PongMessage

        alive = asyncio.Event()
        attempts = iter(
            [
                asyncio.CancelledError(),
                PongMessage(version="1.0"),
            ]
        )

        def side_effect(*_args, **_kwargs):
            result = next(attempts)
            if isinstance(result, BaseException):
                raise result
            return result

        vm = self._make_vm(process_wait_event=alive, send_request_side_effect=side_effect)
        await self._wait_with_alive_process(vm, alive)
        assert next(attempts, None) is None  # the cancelled attempt, then the pong

    async def test_guest_cancelled_error_persistent_becomes_timeout(self) -> None:
        """Guest check always raises CancelledError → eventually times out.

        When every attempt raises CancelledError, wait_for_guest converts each
        to RuntimeError (retried by tenacity). The outer asyncio.timeout()
        eventually fires and raises TimeoutError.
        """
        alive = asyncio.Event()
        vm = self._make_vm(
            process_wait_event=alive,
            send_request_side_effect=asyncio.CancelledError(),
        )

        with pytest.raises(TimeoutError):
            await self._wait_with_alive_process(vm, alive, timeout=0.5)

    async def test_guest_cancelled_error_connect_phase(self) -> None:
        """CancelledError during connect() inside guest_task is also retried.

        Pre-connect uses a very short timeout and fails with TimeoutError
        (expected). The retry loop's guest_task then hits CancelledError
        on connect(), which wait_for_guest wraps as RuntimeError for retry.
        """
        alive = asyncio.Event()
        # Pre-connect gets TimeoutError (expected/caught), first retry gets
        # CancelledError (wrapped→retried), subsequent retries succeed (None).
        responses: list[BaseException | None] = [
            TimeoutError("pre-connect timeout"),
            asyncio.CancelledError(),
        ]
        call_idx = 0

        def connect_side_effect(*_args, **_kwargs):
            nonlocal call_idx
            idx = call_idx
            call_idx += 1
            if idx < len(responses):
                result = responses[idx]
                if isinstance(result, BaseException):
                    raise result

        # The ping answers with a pong (_make_vm). Not send_request_side_effect=PongMessage(...):
        # a pydantic model is iterable, so the mock would return its fields one by one.
        vm = self._make_vm(process_wait_event=alive, connect_side_effect=connect_side_effect)
        await self._wait_with_alive_process(vm, alive)
        # pre-connect, the cancelled connect, the connect that works and its one ping
        assert call_idx == 3
        assert vm.channel.send_request.await_count == 1  # type: ignore[attr-defined]

    async def test_guest_cancelled_mixed_with_other_retryable_errors(self) -> None:
        """CancelledError interleaved with OSError — both retried, then success."""
        from exec_sandbox.guest_agent_protocol import PongMessage

        alive = asyncio.Event()
        attempts = iter(
            [
                OSError("connection refused"),
                asyncio.CancelledError(),
                OSError("connection refused"),
                asyncio.CancelledError(),
                PongMessage(version="1.0"),
            ]
        )

        def side_effect(*_args, **_kwargs):
            result = next(attempts)
            if isinstance(result, BaseException):
                raise result
            return result

        vm = self._make_vm(process_wait_event=alive, send_request_side_effect=side_effect)
        await self._wait_with_alive_process(vm, alive)
        assert next(attempts, None) is None  # four retried failures, then the pong

    async def test_external_cancellation_propagates(self) -> None:
        """Genuine external cancellation (not from guest_task) still propagates.

        When the caller cancels the task (not the timeout, not a child task),
        CancelledError must not be swallowed.
        """
        alive = asyncio.Event()
        block = asyncio.Event()
        connecting = asyncio.Event()

        async def blocking_connect(*_args, **_kwargs):
            connecting.set()
            await block.wait()  # Block forever until cancelled

        vm = self._make_vm(
            process_wait_event=alive,
            connect_side_effect=blocking_connect,
        )

        task = asyncio.create_task(self._wait_with_alive_process(vm, alive, timeout=30))
        await connecting.wait()  # the wait is in its first connect (the fake ignores its timeout)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


# ============================================================================
# Unit Tests - Guest port socket activation
# ============================================================================


class TestGuestPortSocketActivation(_QemuCmdTestBase):
    """The cmd and event ports on listening sockets that QEMU inherits, when no sudo starts it:
    QEMU sends the guest's output on the socket it accepts, which takes the listener's send
    buffer (macOS gives 8 KiB)."""

    @staticmethod
    def _chardevs(cmd: list[str]) -> list[str]:
        return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-chardev"]

    async def test_guest_port_fds_give_fd_chardevs(self, vm_settings, workdir) -> None:
        with _qemu_cmd_mocks():
            cmd = await TestQmpSocketActivation._build_cmd(vm_settings, workdir, guest_port_fds=(8, 9))

        chardevs = self._chardevs(cmd)
        assert "socket,id=cmd0,fd=8,server=on,wait=off" in chardevs
        assert "socket,id=event0,fd=9,server=on,wait=off" in chardevs
        assert not [c for c in chardevs if workdir.cmd_socket in c or workdir.event_socket in c]

    async def test_no_guest_port_fds_give_paths(self, vm_settings, workdir) -> None:
        with _qemu_cmd_mocks():
            cmd = await TestQmpSocketActivation._build_cmd(vm_settings, workdir)

        chardevs = self._chardevs(cmd)
        assert f"socket,id=cmd0,path={workdir.cmd_socket},server=on,wait=off" in chardevs
        assert f"socket,id=event0,path={workdir.event_socket},server=on,wait=off" in chardevs

    async def test_the_socket_qemu_accepts_has_the_raised_send_buffer(self, workdir) -> None:
        """Each port's listener: the socket accepted on it sends with at least
        SOCKET_SEND_BUFFER_BYTES."""
        from exec_sandbox.guest_channel import SOCKET_SEND_BUFFER_BYTES
        from exec_sandbox.vm_manager import _close_listeners, _guest_port_listeners

        listeners = _guest_port_listeners(workdir)
        try:
            assert len(listeners) == 2
            for listener, path in zip(listeners, (workdir.cmd_socket, workdir.event_socket), strict=True):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect(path)
                    accepted, _ = listener.accept()
                    with accepted:
                        assert accepted.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF) >= SOCKET_SEND_BUFFER_BYTES
        finally:
            _close_listeners(listeners)

    async def test_no_listeners_when_sudo_starts_qemu(self, workdir) -> None:
        """sudo closes inherited fds: QEMU makes the sockets at their paths."""
        from exec_sandbox.vm_manager import _guest_port_listeners

        workdir.use_qemu_vm_user = True
        try:
            with (
                patch("exec_sandbox.vm_manager.get_qemu_vm_uid", return_value=999),
                patch("exec_sandbox.vm_manager.os.getuid", return_value=1000),
            ):
                assert _guest_port_listeners(workdir) == []
            assert not Path(workdir.cmd_socket).exists()
        finally:
            workdir.use_qemu_vm_user = False  # as test_qmp_fd_not_used_with_sudo_path


# ============================================================================
# Unit Tests - QMP Socket Activation (fd-based vs path-based)
# ============================================================================


class TestQmpSocketActivation(_QemuCmdTestBase):
    """Verify QMP socket activation (fd-based) vs legacy path-based syntax.

    When qmp_fd is set, build_qemu_cmd() must emit:
        -chardev socket,id=qmp0,fd=N,server=on,wait=off
        -mon chardev=qmp0,mode=control
    When qmp_fd is None, it must emit legacy:
        -qmp unix:<path>,server=on,wait=off
    """

    @staticmethod
    async def _build_cmd(vm_settings, workdir, *, vm_id: str = "test-vm", allow_network: bool = False, **kwargs):
        """Build a QEMU command with shared test defaults for QMP tests (`kwargs` override them)."""
        from exec_sandbox.qemu_cmd import build_qemu_cmd

        return await build_qemu_cmd(
            settings=vm_settings,
            arch=HostArch.X86_64,
            vm_id=vm_id,
            workdir=workdir,
            spec=VmSpec(language=Language.PYTHON, memory_mb=256, cpu_cores=1, allow_network=allow_network),
            **{
                "expose_ports": None,
                "direct_write": False,
                "debug_boot": False,
                "snapshot_drive": None,
                "defer_incoming": False,
                "qmp_fd": None,
                "guest_port_fds": None,
                "mem_path": None,
                "mem_path_share": False,
            }
            | kwargs,
        )

    async def test_qmp_fd_generates_chardev_mon_syntax(self, vm_settings, workdir) -> None:
        """qmp_fd=7 produces -chardev/-mon pair, never -qmp."""
        with _qemu_cmd_mocks():
            cmd = await self._build_cmd(vm_settings, workdir, vm_id="test-vm-qmp-fd", qmp_fd=7)

        assert "socket,id=qmp0,fd=7,server=on,wait=off" in " ".join(cmd)
        assert "-mon" in cmd
        mon_idx = cmd.index("-mon")
        assert cmd[mon_idx + 1] == "chardev=qmp0,mode=control"
        assert "-qmp" not in cmd

    async def test_qmp_fd_none_generates_legacy_syntax(self, vm_settings, workdir) -> None:
        """qmp_fd=None (default) produces -qmp, never -chardev with qmp0."""
        with _qemu_cmd_mocks():
            cmd = await self._build_cmd(vm_settings, workdir, vm_id="test-vm-qmp-legacy")

        assert "-qmp" in cmd
        qmp_idx = cmd.index("-qmp")
        assert cmd[qmp_idx + 1].startswith("unix:")
        # Must not contain socket-activation syntax
        cmd_str = " ".join(cmd)
        assert "qmp0" not in cmd_str
        # The only -mon is the event-only monitor (QemuVM.start_event_watch)
        assert [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-mon"] == ["chardev=qmpev0,mode=control"]

    async def test_qmp_fd_zero(self, vm_settings, workdir) -> None:
        """qmp_fd=0 is valid (not falsy) and generates fd=0."""
        with _qemu_cmd_mocks():
            cmd = await self._build_cmd(vm_settings, workdir, vm_id="test-vm-qmp-fd0", qmp_fd=0)

        cmd_str = " ".join(cmd)
        assert "socket,id=qmp0,fd=0,server=on,wait=off" in cmd_str
        assert "-qmp" not in cmd

    async def test_qmp_fd_not_used_with_sudo_path(self, vm_settings, workdir) -> None:
        """use_qemu_vm_user=True + different UID: sudo wraps cmd, legacy -qmp syntax."""
        workdir.use_qemu_vm_user = True
        try:
            # qemu_vm_uid=999, os_getuid=1000 → different user → sudo wraps
            with _qemu_cmd_mocks(qemu_vm_uid=999, os_getuid=1000):
                cmd = await self._build_cmd(vm_settings, workdir, vm_id="test-vm-qmp-sudo")

            assert "-qmp" in cmd
            qmp_idx = cmd.index("-qmp")
            assert cmd[qmp_idx + 1].startswith("unix:")
            assert "server=on,wait=off" in cmd[qmp_idx + 1]
            assert "sudo" in cmd
        finally:
            # This flag is part of the command fixture, not ownership of the
            # host-created temporary directory. Do not make fixture teardown
            # invoke a real credentialed `sudo rm` after mocked UID tests.
            workdir.use_qemu_vm_user = False

    async def test_qmp_fd_skips_sudo_when_already_target_user(self, vm_settings, workdir) -> None:
        """use_qemu_vm_user=True + same UID: no sudo, socket activation works."""
        workdir.use_qemu_vm_user = True
        try:
            # qemu_vm_uid=999, os_getuid=999 → same user → no sudo needed
            with _qemu_cmd_mocks(qemu_vm_uid=999, os_getuid=999):
                cmd = await self._build_cmd(vm_settings, workdir, vm_id="test-vm-qmp-same-user", qmp_fd=7)

            # Socket activation syntax (no sudo blocking fds)
            cmd_str = " ".join(cmd)
            assert "socket,id=qmp0,fd=7,server=on,wait=off" in cmd_str
            assert "-mon" in cmd
            # No sudo or legacy -qmp
            assert "sudo" not in cmd
            assert "-qmp" not in cmd
        finally:
            workdir.use_qemu_vm_user = False

    async def test_qmp_fd_with_all_features(self, vm_settings, workdir) -> None:
        """Smoke test: qmp_fd + defer_incoming + allow_network + snapshot_drive."""
        with _qemu_cmd_mocks():
            cmd = await self._build_cmd(
                vm_settings,
                workdir,
                vm_id="test-vm-qmp-all",
                allow_network=True,
                qmp_fd=7,
                defer_incoming=True,
                snapshot_drive="/tmp/snap.qcow2",
            )

        cmd_str = " ".join(cmd)
        # Socket activation syntax present
        assert "socket,id=qmp0,fd=7,server=on,wait=off" in cmd_str
        assert "-mon" in cmd
        # Defer incoming present
        idx = cmd.index("-incoming")
        assert cmd[idx + 1] == "defer"
        # Snapshot drive present
        assert any("serial=snap" in arg for arg in cmd)
        # No legacy -qmp
        assert "-qmp" not in cmd


# ============================================================================
# Unit Tests - restore_vm failures after QEMU started
# ============================================================================


class TestRestoreFailuresAfterLaunch:
    """restore_vm once QEMU runs: a QEMU that dies, and a guest that does not answer after
    its state loaded. The infrastructure, QEMU and the migration are stubbed at their
    collaborators; restore_vm and its error mapping run as they are."""

    @staticmethod
    async def _restore(
        manager: VmManager,
        tmp_path: Path,
        *,
        returncode: int | None,
        connect_error: BaseException | None,
        host_disk_full: BaseException | None = None,
        gvproxy_ready: Callable[[], Awaitable[None]] | None = None,
        vm: MagicMock | None = None,
    ) -> QemuVM:
        """Run restore_vm with QEMU's exit code `returncode` (None: it runs) and a guest
        channel whose connect raises `connect_error` (None: it answers). `host_disk_full`
        is what the VM's raise_if_host_disk_full raises (None: the disk has room). Returns
        what restore_vm returns (the stubbed VM)."""
        vmstate = tmp_path / "entry.vmstate"
        vmstate.write_bytes(b"vmstate")
        vmstate.with_suffix(".home").write_bytes(b"home")
        infra = MagicMock()
        infra.vm_id = "restore-vm"
        infra.workdir.use_qemu_vm_user = False
        infra.channel.connect = AsyncMock(side_effect=connect_error)
        vm = MagicMock() if vm is None else vm
        vm.process.returncode = returncode
        vm.collect_diagnostics = MagicMock()
        vm.transition_state = AsyncMock()
        vm.raise_if_host_disk_full = AsyncMock(side_effect=host_disk_full)
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        launch = AsyncMock(return_value=vm)
        leases = (MagicMock(),)  # the L1 entry's lease: the launch takes the VM's share of it
        try:
            with (
                patch.object(manager, "_setup_vm_infra", new_callable=AsyncMock, return_value=infra),
                _fake_guest_port_listeners(),
                patch("exec_sandbox.vm_manager.create_unix_socket", return_value=MagicMock()),
                patch("exec_sandbox.vm_manager.build_qemu_cmd", new_callable=AsyncMock, return_value=["qemu"]),
                patch.object(
                    manager,
                    "_start_gvproxy_for_vm",
                    new_callable=AsyncMock,
                    return_value=(None, None, 0, gvproxy_ready),
                ),
                patch.object(manager, "_launch_qemu_vm", launch),
                patch("exec_sandbox.vm_manager.probe_qemu_version", new_callable=AsyncMock, return_value=(11, 1, 3)),
                patch("exec_sandbox.vm_manager.MigrationClient", return_value=client),
                patch.object(manager, "_cleanup_failed_owner", new_callable=AsyncMock, return_value=True),
                patch.object(manager, "_cleanup_reservation_owners", new_callable=AsyncMock),
            ):
                return await manager.restore_vm(
                    VmSpec(language=Language.PYTHON),
                    tenant_id="t",
                    task_id="restore",
                    vmstate_path=vmstate,
                    base_image=tmp_path / "base.img",
                    snapshot_drive=None,
                    cache_leases=leases,
                    expose_ports=None,
                    reservation=MagicMock(),
                )
        finally:
            assert launch.await_args.kwargs["cache_leases"] is leases

    @pytest.mark.parametrize("returncode", [-6, -9, 0])
    async def test_qemu_dead_after_launch_is_qemu_crashed(
        self, unit_test_vm_manager, tmp_path: Path, returncode: int
    ) -> None:
        """QEMU gone before the state loads (any exit code, also 0): the restore stops
        with TransientError QEMU_CRASHED, which names the exit code; nothing loads."""
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.QEMU_CRASHED,
            match=rf"^QEMU died during L1 restore for restore-vm \(rc={returncode}\)$",
        ) as exc_info:
            await self._restore(unit_test_vm_manager, tmp_path, returncode=returncode, connect_error=None)
        assert exc_info.value.context == {"vm_id": "restore-vm", "returncode": returncode}

    @pytest.mark.parametrize("returncode", [126, 127])
    async def test_qemu_launch_ended_with_126_or_127_is_host_dependency(
        self, unit_test_vm_manager, tmp_path: Path, returncode: int
    ) -> None:
        """An exec in QEMU's launch failed: ConfigError HOST_DEPENDENCY naming the configured
        QEMU binary, not QEMU_CRASHED; nothing loads."""
        binary = tmp_path / "qemu-system-test"
        binary.write_bytes(b"")
        unit_test_vm_manager.settings = unit_test_vm_manager.settings.model_copy(
            update={"qemu_bin_x86": binary, "qemu_bin_arm": binary}
        )
        qemu = str(binary)
        with pytest.raises(
            ConfigError,
            check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY,
            match=rf"^Failed to start {re.escape(qemu)}: exit status {returncode} \(",
        ) as exc_info:
            await self._restore(unit_test_vm_manager, tmp_path, returncode=returncode, connect_error=None)
        assert exc_info.value.context == {
            "vm_id": "restore-vm",
            "returncode": returncode,
            "qemu_cmd": "qemu",
            "binary": qemu,
            "exit_code": returncode,
        }

    @pytest.mark.parametrize(
        "connect_error",
        [pytest.param(TimeoutError(), id="no-answer"), pytest.param(ConnectionResetError(), id="port-closed")],
    )
    async def test_silent_guest_after_the_state_loaded(
        self, unit_test_vm_manager, tmp_path: Path, connect_error: BaseException
    ) -> None:
        """The state loaded, then the guest does not answer (or QEMU closed its port): the
        restore raises GUEST_SILENT_AFTER_RESTORE, which makes the L1 layer cold boot and
        drop the entry (a torn RAM template would fail every later hit the same way)."""
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_SILENT_AFTER_RESTORE) as exc_info:
            await self._restore(unit_test_vm_manager, tmp_path, returncode=None, connect_error=connect_error)
        assert exc_info.value.__cause__ is connect_error
        assert exc_info.value.context["vm_id"] == "restore-vm"
        assert exc_info.value.context["vmstate_path"] == str(tmp_path / "entry.vmstate")

    async def test_socket_auth_failure_after_the_state_loaded_keeps_the_entry(
        self, unit_test_vm_manager, tmp_path: Path
    ) -> None:
        """A failed socket peer check is not a silent guest: TransientError GUEST_UNAVAILABLE
        (the caller cold boots), not the code that drops the L1 entry."""
        auth = SocketAuthError("peer uid mismatch", expected_uid=1000, actual_uid=0, context=None)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE) as exc_info:
            await self._restore(unit_test_vm_manager, tmp_path, returncode=None, connect_error=auth)
        assert exc_info.value.__cause__ is auth

    async def test_guest_paused_on_a_full_host_disk_is_local_disk(self, unit_test_vm_manager, tmp_path: Path) -> None:
        """`cont` replays a home disk write that finds no space: QEMU pauses the VM again and
        the guest never answers. LOCAL_DISK, so the L1 layer cold boots and keeps a good
        entry (GUEST_SILENT_AFTER_RESTORE would remove it)."""
        full = TransientError(
            "VM restore-vm stopped: the host disk is full",
            code=ErrorCode.LOCAL_DISK,
            context=None,
            hint=None,
            instance_id=None,
        )
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK) as exc_info:
            await self._restore(
                unit_test_vm_manager, tmp_path, returncode=None, connect_error=TimeoutError(), host_disk_full=full
            )
        assert exc_info.value is full

    async def test_a_gvproxy_that_fails_during_the_restore_fails_it(self, unit_test_vm_manager, tmp_path: Path) -> None:
        """gvproxy-wrapper gets ready while the state loads: the restore waits for it before
        the VM is ready; its error fails the restore."""
        failed = TransientError(
            "gvproxy-wrapper did not become ready in time",
            code=ErrorCode.GVPROXY_FAILED,
            context={},
            hint=None,
            instance_id=None,
        )

        vm = MagicMock()
        vm.transition_state = AsyncMock()

        async def gvproxy_ready() -> None:
            vm.transition_state.assert_not_awaited()  # the wait comes before READY
            raise failed

        with pytest.raises(TransientError) as exc_info:
            await self._restore(
                unit_test_vm_manager, tmp_path, returncode=None, connect_error=None, gvproxy_ready=gvproxy_ready, vm=vm
            )
        assert exc_info.value is failed
        vm.transition_state.assert_not_awaited()

    async def test_guest_that_answers_is_restored(self, unit_test_vm_manager, tmp_path: Path) -> None:
        """Control: the same harness with a guest that answers gives a READY restored VM."""
        vm = await self._restore(unit_test_vm_manager, tmp_path, returncode=None, connect_error=None)
        assert vm.l1_restored is True
        vm.transition_state.assert_awaited_once_with(VmState.READY)  # type: ignore[attr-defined]


class TestRestoreWithoutHomeTemplate:
    """restore_vm of an L1 entry whose home disk template is gone."""

    async def test_home_template_gone_is_transient_before_any_setup(self, unit_test_vm_manager, tmp_path: Path) -> None:
        """The L1 entry's .home was removed after check_cache (eviction): restore_vm fails as
        transient (the caller cold boots with the same reservation) before any VM folder exists."""
        spec = VmSpec(language=Language.PYTHON)
        vmstate = tmp_path / "entry.vmstate"
        vmstate.write_bytes(b"vmstate")
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK) as exc:
            async with unit_test_vm_manager.reservation_context(
                spec, vm_id="restore-no-home", held=frozenset()
            ) as reservation:
                await unit_test_vm_manager.restore_vm(
                    spec,
                    tenant_id="t",
                    task_id="restore-no-home",
                    vmstate_path=vmstate,
                    base_image=tmp_path / "base.img",
                    snapshot_drive=None,
                    cache_leases=(),
                    expose_ports=None,
                    reservation=reservation,
                )
        assert str(exc.value) == f"L1 home disk missing (evicted?): {tmp_path / 'entry.home'}"
        assert unit_test_vm_manager.get_active_vms() == {}


class TestPrepareHomeDisk:
    """Home disk failures are transient, so an L1 restore falls back to a cold boot.

    _prepare_home_disk on its own: through restore_vm, a VM would boot after it; the clone is
    faked (clone_sparse) to choose between a copy-on-write clone and a plain copy."""

    async def test_missing_template_is_transient(self, unit_test_vm_manager, tmp_path: Path) -> None:
        workdir = await VmWorkingDirectory.create("vm-home", home_dir=tmp_path)
        try:
            with pytest.raises(
                TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK, match="Could not prepare the home disk"
            ):
                await unit_test_vm_manager._prepare_home_disk(
                    workdir, VmSpec(language=Language.PYTHON), tmp_path / "evicted.home"
                )
        finally:
            await workdir.cleanup()

    @pytest.mark.parametrize("cow", [True, False], ids=["clone", "copy"])
    async def test_only_a_clone_shares_blocks_with_the_template(
        self, unit_test_vm_manager, tmp_path: Path, cow: bool
    ) -> None:
        """A copy-on-write clone shares the template's blocks until the guest writes them: gate 4
        does not count them as written, also while the clone is made (the note comes first:
        without it, a probe in that window would count the whole clone as written). A plain
        copy (ext4) takes them from free space at once: counting them as shared would hide
        that space from gate 4."""
        template = tmp_path / "entry.home"
        template.write_bytes(b"x" * 1024 * 1024)
        workdir = await VmWorkingDirectory.create("vm-home", home_dir=tmp_path)
        shared = unit_test_vm_manager._admission._shared_disk_bytes
        noted_during_clone: list[bool] = []

        def clone(_src: Path, dst: Path, *, stop: threading.Event | None) -> bool:
            noted_during_clone.append(dst.name in shared)
            return cow

        try:
            with patch("exec_sandbox.vm_manager.clone_sparse", side_effect=clone):
                await unit_test_vm_manager._prepare_home_disk(workdir, VmSpec(language=Language.PYTHON), template)
            assert noted_during_clone == [True]
            assert (workdir.home_disk.name in shared) is cow
        finally:
            shared.pop(workdir.home_disk.name, None)
            await workdir.cleanup()

    async def test_a_failed_clone_leaves_no_shared_blocks_note(self, unit_test_vm_manager, tmp_path: Path) -> None:
        """The note made before the clone goes when the clone fails."""
        template = tmp_path / "entry.home"
        template.write_bytes(b"x" * 1024 * 1024)
        workdir = await VmWorkingDirectory.create("vm-home", home_dir=tmp_path)
        try:
            with (
                patch("exec_sandbox.vm_manager.clone_sparse", side_effect=OSError(errno.ENOSPC, "No space left")),
                pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK),
            ):
                await unit_test_vm_manager._prepare_home_disk(workdir, VmSpec(language=Language.PYTHON), template)
            assert workdir.home_disk.name not in unit_test_vm_manager._admission._shared_disk_bytes
        finally:
            await workdir.cleanup()

    async def test_a_cancel_stops_the_copy_and_comes_out_once_it_ended(
        self, unit_test_vm_manager, tmp_path: Path
    ) -> None:
        """A cancel during a copy of the data (no reflink: ext4) stops it at its next chunk and
        comes out once the copy ended and removed its partial disk: no disk appears after the
        cleanup of the launch, and no copy of GBs holds the cancel (the copy here would take
        5 s to its end)."""
        template = tmp_path / "entry.home"
        template.write_bytes(b"x")
        workdir = await VmWorkingDirectory.create("vm-home", home_dir=tmp_path)
        copying, ended = threading.Event(), threading.Event()

        def slow_copy(_src: Path, dst: Path, *, stop: threading.Event | None) -> bool:
            dst.write_bytes(b"partial")
            copying.set()
            assert stop is not None
            stopped = stop.wait(5)  # a copy of the data checks stop before each chunk
            dst.unlink()
            ended.set()
            if stopped:
                raise InterruptedError(errno.EINTR, "The copy was stopped")
            return False

        try:
            with patch("exec_sandbox.vm_manager.clone_sparse", side_effect=slow_copy):
                preparing = asyncio.create_task(
                    unit_test_vm_manager._prepare_home_disk(workdir, VmSpec(language=Language.PYTHON), template)
                )
                assert await asyncio.to_thread(copying.wait, 5)
                started = time.monotonic()
                preparing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await preparing
                assert time.monotonic() - started < 2
                assert ended.is_set()
                assert not workdir.home_disk.exists()
        finally:
            await workdir.cleanup()


class TestAddCache:
    """A cache on the home disk filesystem joins gate 4: a request short of disk space evicts it.
    The free space is faked at the OS call (shutil.disk_usage); the cache is a fake
    EvictableCache in the manager's own folder (the home disk filesystem)."""

    @pytest.mark.parametrize("added", [True, False], ids=["added", "not-added"])
    async def test_a_request_short_of_disk_space_evicts_an_added_cache(
        self, unit_test_vm_manager: VmManager, added: bool
    ) -> None:
        """No free space and one 100 GB entry: with the cache added, the request evicts the entry
        and gets its reservation; without it, LOCAL_DISK and the entry stays."""
        mib = 1024 * 1024
        free_mb = 0.0
        entries = [100_000.0]

        async def evict_next() -> int | None:
            nonlocal free_mb
            if not entries:
                return None
            size = entries.pop()
            free_mb += size  # the filesystem gets the entry's space back
            return int(size * mib)

        class OneEntryCache:
            cache_dir = unit_test_vm_manager.work_dir

            def eviction_order(self) -> SimpleNamespace:
                return SimpleNamespace(evict_next=evict_next)

            async def reclaimable_bytes(self, *, held: frozenset[Path]) -> int:
                del held  # no lease here
                return int(sum(entries) * mib)

        if added:
            unit_test_vm_manager.add_cache(OneEntryCache())
        reservation = unit_test_vm_manager.reservation_context(
            VmSpec(language=Language.PYTHON), vm_id="short-of-disk", held=frozenset()
        )
        with patch("exec_sandbox.admission.shutil.disk_usage", lambda _path: SimpleNamespace(free=int(free_mb * mib))):
            if added:
                async with reservation:
                    pass
            else:
                with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                    async with reservation:
                        pytest.fail("admitted with no free space")
        assert entries == ([] if added else [100_000.0])


_UNUSED_TENANT = Path(cgroup.CGROUP_V2_BASE_PATH) / cgroup.CGROUP_APP_NAMESPACE / "test-vm-cgroup-records"
"""A tenant cgroup no VM uses: the records of TestVmCgroupRecords point under it."""


class TestVmCgroupRecords:
    """Each VM cgroup is recorded in the manager's folder before it exists, and its working
    folder before QEMU starts; each record goes once its cgroup or folder is gone: what a
    crash of this process leaves is what the watcher and the next start kill and remove
    (PrivateRunDir.record_cgroup, record_folder).

    The private steps (_setup_vm_infra, _force_cleanup_all_resources) are the seam: the records
    exist only between steps of a launch, and a removal that fails needs a faked OS answer."""

    async def test_cgroup_and_working_folder_are_recorded_before_the_cgroup_is_made(
        self, unit_test_vm_manager, tmp_path: Path
    ) -> None:
        workdir = MagicMock()
        workdir.overlay_image = Path("/tmp/pending-overlay.qcow2")
        workdir.path = tmp_path / "vm-tenant-t-abc"
        records_at_setup: list[tuple[str, Path]] = []
        vm_ids: list[str] = []
        recorders: list[int] = []
        real_record = unit_test_vm_manager._run_dir.record_cgroup

        def record(cgroup_path: Path) -> None:  # the first record starts the watcher: never on the loop
            recorders.append(threading.get_ident())
            real_record(cgroup_path)

        async def create(*, vm_id: str, **_kwargs: object) -> Cgroup:
            vm_ids.append(vm_id)
            watcher = f"{unit_test_vm_manager.work_dir.name}.cgroup"  # the watcher's own cgroup (Linux)
            records_at_setup.extend(
                (p.name, p.readlink())
                for p in sorted(unit_test_vm_manager.work_dir.glob("*.*"))
                if p.is_symlink() and p.name != watcher
            )
            raise OSError("cgroup")

        async def cleanup_owner(vm_id: str, **_kwargs: object) -> None:
            unit_test_vm_manager._pending_launches.pop(vm_id)

        with (
            patch("exec_sandbox.vm_manager.validate_kernel_initramfs", new_callable=AsyncMock),
            patch("exec_sandbox.vm_manager.VmWorkingDirectory.create", new_callable=AsyncMock, return_value=workdir),
            patch("exec_sandbox.vm_manager.detect_accel_type", new_callable=AsyncMock, return_value=AccelType.KVM),
            patch.object(unit_test_vm_manager._overlay_pool, "acquire", new_callable=AsyncMock),
            patch("exec_sandbox.vm_manager.cgroup.cgroup_v2_available", return_value=True),
            patch.object(Cgroup, "create", side_effect=create),
            patch.object(unit_test_vm_manager._run_dir, "record_cgroup", side_effect=record),
            patch.object(unit_test_vm_manager, "_cleanup_failed_owner", side_effect=cleanup_owner),
            pytest.raises(OSError, match="cgroup"),
        ):
            await unit_test_vm_manager._setup_vm_infra(
                VmSpec(language=Language.PYTHON),
                "tenant",
                "task",
                MagicMock(),
                base_image=Path("/fake/base.qcow2"),
                home_template=None,
            )
        assert sorted(records_at_setup) == [
            (f"{vm_ids[0]}.cgroup", Path("/sys/fs/cgroup/code-exec/tenant") / vm_ids[0]),
            ("vm-tenant-t-abc.folder", workdir.path),
        ]
        assert recorders != [] and threading.get_ident() not in recorders
        unit_test_vm_manager._run_dir.forget_cgroup(vm_ids[0])  # its owner cleanup is a stand-in
        unit_test_vm_manager._run_dir.forget_folder(workdir.path.name)

    async def test_record_goes_once_the_cgroup_is_removed(self, unit_test_vm_manager) -> None:
        unit_test_vm_manager._run_dir.record_cgroup(_UNUSED_TENANT / "vm-1")
        workdir = await VmWorkingDirectory.create("vm-1", home_dir=unit_test_vm_manager.work_dir)
        try:
            results = await unit_test_vm_manager._force_cleanup_all_resources(
                vm_id="vm-1",
                qemu_proc=None,
                gvproxy_proc=None,
                workdir=workdir,
                vm_cgroup=Cgroup(_UNUSED_TENANT / "vm-1"),
            )
        finally:
            await workdir.cleanup()
        assert results["cgroup"] is True
        assert not (unit_test_vm_manager.work_dir / "vm-1.cgroup").is_symlink()

    @pytest.mark.parametrize(
        "failing", ["exec_sandbox.vm_manager.cleanup_process", "exec_sandbox.cgroup.Cgroup.remove"]
    )
    async def test_record_stays_while_the_cgroup_does(self, unit_test_vm_manager, failing: str) -> None:
        """Process death unconfirmed, or the cgroup not removed: the cgroup stays, and so does
        its record (a crash then leaves it to the watcher and the next start)."""
        unit_test_vm_manager._run_dir.record_cgroup(_UNUSED_TENANT / "vm-1")
        workdir = await VmWorkingDirectory.create("vm-1", home_dir=unit_test_vm_manager.work_dir)
        try:
            with patch(failing, new_callable=AsyncMock, return_value=False):
                results = await unit_test_vm_manager._force_cleanup_all_resources(
                    vm_id="vm-1",
                    qemu_proc=None,
                    gvproxy_proc=None,
                    workdir=workdir,
                    vm_cgroup=Cgroup(_UNUSED_TENANT / "vm-1"),
                )
        finally:
            await workdir.cleanup()
        assert results["cgroup"] is False
        assert (unit_test_vm_manager.work_dir / "vm-1.cgroup").is_symlink()
        unit_test_vm_manager._run_dir.forget_cgroup("vm-1")

    @pytest.mark.parametrize("removed", [True, False], ids=["removed", "kept"])
    async def test_working_folder_record_goes_with_the_folder(self, unit_test_vm_manager, removed: bool) -> None:
        """The record goes once the working folder is removed, and only then: a folder that
        stays (a failed removal) is the next start's to remove after a crash."""
        workdir = await VmWorkingDirectory.create("vm-1", home_dir=unit_test_vm_manager.work_dir)
        unit_test_vm_manager._run_dir.record_folder(workdir.path)
        record = unit_test_vm_manager.work_dir / f"{workdir.path.name}.folder"
        try:
            with ExitStack() as stack:
                if not removed:
                    stack.enter_context(
                        patch.object(VmWorkingDirectory, "cleanup", new_callable=AsyncMock, return_value=False)
                    )
                results = await unit_test_vm_manager._force_cleanup_all_resources(
                    vm_id="vm-1", qemu_proc=None, gvproxy_proc=None, workdir=workdir, vm_cgroup=None
                )
            assert results["workdir"] is removed
            assert record.is_symlink() is not removed
        finally:
            await workdir.cleanup()
            unit_test_vm_manager._run_dir.forget_folder(workdir.path.name)


_KILLED_OWNER = """
import asyncio, sys
from pathlib import Path
from exec_sandbox import constants
from exec_sandbox.models import Language
from exec_sandbox.settings import Settings
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.vm_spec import VmSpec

async def main() -> None:
    images, root = Path(sys.argv[1]), Path(sys.argv[2])
    kernel = images / "kernels" if (images / "kernels").exists() else images
    manager = VmManager(Settings(base_images_dir=images, kernel_path=kernel, home_disk_dir=root))
    await manager.start()
    vm = await manager.create_vm(
        VmSpec(language=Language.RAW),
        tenant_id="test-killed-owner",
        task_id="crash",
        direct_write_target=None,
        expose_ports=None,
        on_boot_log=None,
        snapshot_drive=None,
        cache_leases=(),
        reservation=None,
        retry_profile=constants.RETRY_USER_FACING,
        mem_path=None,
        mem_path_share=False,
    )
    print(vm.cgroup.path if vm.cgroup is not None else None, vm.workdir.path, flush=True)
    await asyncio.Event().wait()  # until the test kills this process (never a read of a shared stdin)

asyncio.run(main())
"""
"""A VmManager process with one VM, killed by the test (SIGKILL: no cleanup runs)."""


def _wait_until_empty(cgroup_path: Path, *, timeout: float) -> bool:
    """Whether `cgroup_path` has no process left within `timeout` s: poll(2) on cgroup.events,
    which wakes at each change. It kills nothing itself (Cgroup.kill would)."""
    import select
    import time

    fd = os.open(cgroup_path / "cgroup.events", os.O_RDONLY)
    try:
        poller = select.poll()
        poller.register(fd, select.POLLPRI | select.POLLERR)
        deadline = time.monotonic() + timeout
        while b"populated 1" in os.pread(fd, 4096, 0):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            poller.poll(remaining * 1000)
        return True
    finally:
        os.close(fd)


@skip_unless_linux
class TestKilledOwnerVmsEnd:
    """A process killed with SIGKILL runs no cleanup. A QEMU that runs as qemu-vm has sudo as
    its parent: exit-with-parent does not end it, and it would run on with its RAM. The
    watcher kills its cgroup at once; the next sweep removes the cgroup, the folder and the
    VM's working folder in the temp folder."""

    async def test_vm_of_a_killed_process_ends_at_once_and_the_sweep_removes_it(self, images_dir: Path) -> None:
        import shutil
        from contextlib import suppress

        import psutil

        from exec_sandbox.permission_utils import probe_qemu_as_qemu_vm
        from exec_sandbox.private_run_dir import remove_dead_folders

        if not await probe_qemu_as_qemu_vm():
            pytest.skip("QEMU runs as this user here: it ends with its process (TestKilledOwnerWorkingFolder)")
        root = Path(tempfile.mkdtemp(prefix="hd-"))  # short and 0700 (PrivateRunDir checks the root)
        tmp = Path(tempfile.mkdtemp(prefix="ko-"))  # the VM's working folder: short socket paths
        owner = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _KILLED_OWNER,
            str(images_dir),
            str(root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env={**os.environ, "TMPDIR": str(tmp)},
        )
        cgroup_path: Path | None = None
        try:
            assert owner.stdout is not None
            cgroup_path = Path((await owner.stdout.readline()).decode().split()[0])
            assert cgroup_path.parent.name == "test-killed-owner"
            assert (cgroup_path / "cgroup.procs").read_text().split()  # sudo and the qemu-vm QEMU
            owner.kill()
            assert await owner.wait() == -9
            start = asyncio.get_running_loop().time()
            assert await asyncio.to_thread(_wait_until_empty, cgroup_path, timeout=2.0)
            ended_s = asyncio.get_running_loop().time() - start
            assert ended_s < 1.0, ended_s  # the watcher acts in a few ms; QEMU then unmaps its RAM
            # Left for the sweep: the manager's folder, and the VM's working folder in the temp folder
            assert [folder.name.startswith("vmm-") for folder in root.iterdir()] == [True]
            assert len([folder for folder in tmp.iterdir() if folder.name.startswith("vm-")]) == 1
            await asyncio.to_thread(remove_dead_folders, root)
            assert not cgroup_path.exists()
            assert list(root.iterdir()) == []
            assert [folder for folder in tmp.iterdir() if folder.name.startswith("vm-")] == []
        finally:
            if owner.returncode is None:
                owner.kill()
                await owner.wait()
            # What a failure before the sweep leaves: the VM's and the watcher's cgroups; then
            # this test's tenant cgroup (no other test uses it)
            namespace = Path(cgroup.CGROUP_V2_BASE_PATH) / cgroup.CGROUP_APP_NAMESPACE
            left = [] if cgroup_path is None else [cgroup_path]
            left += [namespace / "exec-sandbox-watcher" / folder.name for folder in root.iterdir()]
            await asyncio.to_thread(Cgroup.end_all, [Cgroup(each) for each in left if each.exists()])
            with suppress(OSError):
                (namespace / "test-killed-owner").rmdir()
            # The killed process's overlay daemon (qemu-storage-daemon, its socket in `tmp`) ends
            # with it (subprocess_utils.end_with_this_process): this only ends one that a failure left.
            for process in psutil.process_iter(["cmdline"]):
                if any(str(tmp) in arg for arg in process.info["cmdline"] or []):
                    with suppress(psutil.Error):
                        process.kill()
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)


class TestKilledOwnerWorkingFolder:
    """Where QEMU runs as this user (macOS, Linux without qemu-vm), it ends with its process:
    exit-with-parent ties it to its parent, this process or (Linux namespaces) unshare, which
    gets SIGKILL when this process ends. Its working folder in the temp folder stays: the next
    sweep removes it. With qemu-vm: TestKilledOwnerVmsEnd."""

    async def test_working_folder_of_a_killed_process_goes_at_the_next_sweep(self, images_dir: Path) -> None:
        import shutil
        from contextlib import suppress

        import psutil

        from exec_sandbox.permission_utils import probe_qemu_as_qemu_vm
        from exec_sandbox.private_run_dir import remove_dead_folders

        if await probe_qemu_as_qemu_vm():
            pytest.skip("QEMU runs as qemu-vm here: TestKilledOwnerVmsEnd covers it")
        root = Path(tempfile.mkdtemp(prefix="hd-"))  # short and 0700 (PrivateRunDir checks the root)
        tmp = Path(tempfile.mkdtemp(prefix="ko-"))  # the VM's working folder: short socket paths
        owner = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _KILLED_OWNER,
            str(images_dir),
            str(root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env={**os.environ, "TMPDIR": str(tmp)},
        )
        try:
            assert owner.stdout is not None
            workdir = Path((await owner.stdout.readline()).decode().split()[1])
            assert workdir.parent == tmp.resolve() or workdir.parent == tmp
            owner.kill()
            assert await owner.wait() == -9
            assert workdir.exists()  # no cleanup ran
            await asyncio.to_thread(remove_dead_folders, root)
            assert not workdir.exists()
            assert list(root.iterdir()) == []
        finally:
            if owner.returncode is None:
                owner.kill()
                await owner.wait()
            # The killed process's overlay daemon: on macOS only the start-time reaper of a
            # process with the same temp folder ends it (on Linux, it ends with that process),
            # and this test removes that folder
            for process in psutil.process_iter(["cmdline"]):
                if any(str(tmp) in arg for arg in process.info["cmdline"] or []):
                    with suppress(psutil.Error):
                        process.kill()
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)

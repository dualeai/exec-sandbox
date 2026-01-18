"""QEMU microVM lifecycle management with multi-layer security.

Architecture:
- Supports Linux with KVM or TCG acceleration
- 6-layer security: KVM + unprivileged + seccomp + cgroups + namespaces + MAC
- qcow2 snapshot-based boot <400ms
- TCP host-guest communication

Performance Optimizations (QEMU 10.0+):
- CPU host passthrough (KVM): Enables all host CPU features (AVX2, AES-NI)
- Memory preallocation: Eliminates page fault latency during code execution
- virtio-blk: 4K blocks, num-queues=1, queue-size=256
- virtio-net: multiqueue off, TCP offload disabled (simpler for short VMs)
- Drive tuning: detect-zeroes=unmap, copy-on-read off, werror/rerror explicit
- Machine: mem-merge off (no KSM), dump-guest-core off
- io_uring AIO: Modern Linux async I/O (probed at startup, threads fallback)
- cache=unsafe: Safe for ephemeral VMs, major I/O performance boost
- microvm fast shutdown: -no-reboot + triple-fault for ~1-2s cleanup
"""

import asyncio
import contextlib
import json
import logging
import os
import platform
import re
import signal
import sys
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TextIO
from uuid import uuid4

import aiofiles
import aiofiles.os
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    wait_random_exponential,
)

from exec_sandbox import cgroup, constants
from exec_sandbox._logging import get_logger
from exec_sandbox.dns_filter import generate_dns_zones_json
from exec_sandbox.exceptions import VmError, VmTimeoutError
from exec_sandbox.guest_agent_protocol import (
    ExecuteCodeRequest,
    PingRequest,
    PongMessage,
)
from exec_sandbox.guest_channel import DualPortChannel, GuestChannel
from exec_sandbox.models import ExecutionResult, Language, TimingBreakdown
from exec_sandbox.overlay_pool import OverlayPool, QemuImgError
from exec_sandbox.permission_utils import (
    can_access,
    chmod_async,
    chown_to_qemu_vm,
    ensure_traversable,
    get_qemu_vm_uid,
    grant_qemu_vm_access,
    probe_sudo_as_qemu_vm,
)
from exec_sandbox.platform_utils import HostArch, HostOS, ProcessWrapper, detect_host_arch, detect_host_os
from exec_sandbox.resource_cleanup import cleanup_process, cleanup_vm_processes
from exec_sandbox.settings import Settings
from exec_sandbox.socket_auth import create_unix_socket
from exec_sandbox.subprocess_utils import drain_subprocess_output, read_log_tail
from exec_sandbox.vm_timing import VmTiming
from exec_sandbox.vm_working_directory import VmWorkingDirectory

logger = get_logger(__name__)

# KVM ioctl constants for probing
# See: linux/kvm.h - these are stable ABI
_KVM_GET_API_VERSION = 0xAE00
_KVM_API_VERSION_EXPECTED = 12  # Stable since Linux 2.6.38

# Security: Identifier validation pattern
# Only alphanumeric, underscore, and hyphen allowed to prevent:
# - Shell command injection via malicious tenant_id/task_id
# - Path traversal attacks (no '..', '/')
# - Socket path manipulation
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_IDENTIFIER_MAX_LENGTH = 128  # Reasonable limit for identifiers

# QEMU binary extraction pattern for error diagnostics
# Extracts binary name from shell wrapper commands (e.g., "qemu-system-x86_64")
_QEMU_BINARY_PATTERN = re.compile(r"(qemu-system-[^\s]+)")


def _validate_identifier(value: str, name: str) -> None:
    """Validate identifier contains only safe characters.

    Prevents shell injection and path traversal attacks by ensuring identifiers
    (tenant_id, task_id) contain only alphanumeric characters, underscores, and hyphens.

    Args:
        value: The identifier value to validate
        name: Human-readable name for error messages

    Raises:
        ValueError: If identifier contains invalid characters or is too long
    """
    if not value:
        raise ValueError(f"{name} cannot be empty")
    if len(value) > _IDENTIFIER_MAX_LENGTH:
        raise ValueError(f"{name} too long: {len(value)} > {_IDENTIFIER_MAX_LENGTH}")
    if not _IDENTIFIER_PATTERN.match(value):
        raise ValueError(f"{name} contains invalid characters (only [a-zA-Z0-9_-] allowed): {value!r}")


def _log_task_exception(task: asyncio.Task[None]) -> None:
    """Log exceptions from background tasks.

    Callback for asyncio.Task.add_done_callback() that properly logs any
    unhandled exceptions from background tasks. Prevents silent failures.

    Args:
        task: The completed asyncio task to check for exceptions
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Background task failed",
            extra={"task_name": task.get_name()},
            exc_info=exc,
        )


class VmState(Enum):
    """VM lifecycle states."""

    CREATING = "creating"
    BOOTING = "booting"
    READY = "ready"
    EXECUTING = "executing"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


class AccelType(Enum):
    """QEMU acceleration type."""

    KVM = "kvm"  # Linux hardware virtualization
    HVF = "hvf"  # macOS hardware virtualization
    TCG = "tcg"  # Software emulation (slow, but works everywhere)


# Valid state transitions for VM lifecycle
VALID_STATE_TRANSITIONS: dict[VmState, set[VmState]] = {
    VmState.CREATING: {VmState.BOOTING, VmState.DESTROYING},
    VmState.BOOTING: {VmState.READY, VmState.DESTROYING},
    VmState.READY: {VmState.EXECUTING, VmState.DESTROYING},
    VmState.EXECUTING: {VmState.READY, VmState.DESTROYING},
    VmState.DESTROYING: {VmState.DESTROYED},
    VmState.DESTROYED: set(),  # Terminal state - no transitions allowed
}

# Validate all states have transition rules defined
if set(VmState) != set(VALID_STATE_TRANSITIONS.keys()):
    _missing = set(VmState) - set(VALID_STATE_TRANSITIONS.keys())
    raise RuntimeError(f"Missing states in transition table: {_missing}")


# =============================================================================
# Cached System Probes
# =============================================================================
# These probes detect system capabilities once and cache the results.
# Async probes use a shared cache container to avoid global statements.


class _ProbeCache:
    """Container for cached system probe results.

    Uses a class to avoid global statements while maintaining module-level caching.
    Locks are lazily initialized to ensure they're created in the right event loop.

    The locks prevent cache stampede when multiple VMs start concurrently - without
    them, all VMs would run the detection subprocess simultaneously instead of
    sharing the cached result.
    """

    __slots__ = (
        "_locks",
        "hvf",
        "io_uring",
        "kvm",
        "qemu_accels",
        "tsc_deadline",
        "unshare",
    )

    def __init__(self) -> None:
        self.hvf: bool | None = None
        self.io_uring: bool | None = None
        self.kvm: bool | None = None
        self.qemu_accels: set[str] | None = None  # Accelerators available in QEMU binary
        self.tsc_deadline: bool | None = None
        self.unshare: bool | None = None
        self._locks: dict[str, asyncio.Lock] = {}

    def get_lock(self, name: str) -> asyncio.Lock:
        """Get or create a lock for the given probe (lazy initialization).

        Locks must be created lazily because asyncio.Lock requires an event loop,
        which may not exist at module import time.
        """
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]


# Module-level cache instance
_probe_cache = _ProbeCache()


async def _probe_qemu_accelerators() -> set[str]:
    """Probe QEMU binary for available accelerators (cached).

    This provides a 2nd layer of verification beyond OS-level checks (ioctl/sysctl).
    Even if /dev/kvm exists and responds to ioctl, QEMU may not have the accelerator
    compiled in, or may fail to initialize it in certain environments.

    Uses `qemu-system-xxx -accel help` to get the list of accelerators that QEMU
    actually supports. This is the same method recommended by QEMU documentation.

    References:
        - QEMU docs: "help can also be passed as an argument to another option"
        - libvirt probes QEMU binary presence + /dev/kvm (drvqemu.html)
        - GitHub Actions KVM issues: https://github.com/orgs/community/discussions/8305

    Returns:
        Set of available accelerator names (e.g., {"tcg", "kvm"} or {"tcg", "hvf"})
    """
    # Fast path: return cached result (no lock needed)
    if _probe_cache.qemu_accels is not None:
        return _probe_cache.qemu_accels

    # Slow path: acquire lock to prevent stampede, then check cache again
    async with _probe_cache.get_lock("qemu_accels"):
        # Double-check after acquiring lock (another task may have populated cache)
        if _probe_cache.qemu_accels is not None:
            return _probe_cache.qemu_accels

        # Determine QEMU binary based on host architecture
        arch = detect_host_arch()
        qemu_bin = "qemu-system-aarch64" if arch == HostArch.AARCH64 else "qemu-system-x86_64"

        try:
            proc = await asyncio.create_subprocess_exec(
                qemu_bin,
                "-accel",
                "help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)

            if proc.returncode != 0:
                logger.warning(
                    "QEMU accelerator probe failed",
                    extra={"qemu_bin": qemu_bin, "returncode": proc.returncode},
                )
                _probe_cache.qemu_accels = set()
                return _probe_cache.qemu_accels

            # Parse output: "Accelerators supported in QEMU binary:\ntcg\nkvm\n"
            # or "tcg\nhvf\n" on macOS
            output = stdout.decode().strip()
            accels: set[str] = set()
            for raw_line in output.split("\n"):
                accel_name = raw_line.strip().lower()
                # Skip header line and empty lines
                if accel_name and not accel_name.startswith("accelerator"):
                    accels.add(accel_name)

            _probe_cache.qemu_accels = accels
            logger.debug(
                "QEMU accelerator probe complete",
                extra={"qemu_bin": qemu_bin, "accelerators": sorted(accels)},
            )

        except FileNotFoundError:
            logger.warning(
                "QEMU binary not found for accelerator probe",
                extra={"qemu_bin": qemu_bin},
            )
            _probe_cache.qemu_accels = set()
        except (OSError, TimeoutError) as e:
            logger.warning(
                "QEMU accelerator probe failed",
                extra={"qemu_bin": qemu_bin, "error": str(e)},
            )
            _probe_cache.qemu_accels = set()

        return _probe_cache.qemu_accels


async def _check_kvm_available() -> bool:
    """Check if KVM acceleration is available and accessible (cached).

    Two-layer verification approach
    ===============================
    Layer 1 (Kernel): Verify /dev/kvm exists, is accessible, and responds to ioctl
    Layer 2 (QEMU):   Verify QEMU binary has KVM support via `-accel help`

    This 2-layer approach catches edge cases where:
    - /dev/kvm exists but KVM module is broken (nested VMs, containers)
    - KVM ioctl works but QEMU doesn't have KVM compiled in
    - GitHub Actions runners with inconsistent KVM availability

    References:
    - GitHub Actions KVM issues: https://github.com/orgs/community/discussions/8305
    - libvirt capability probing: https://libvirt.org/drvqemu.html

    KVM vs TCG: Virtualization modes with vastly different characteristics
    ======================================================================

    KVM (Kernel-based Virtual Machine) - Production mode:
    - Hardware-assisted virtualization (Intel VT-x / AMD-V)
    - VM boot time: <400ms (with snapshot cache)
    - CPU overhead: near-native performance (~5% penalty)
    - Security: Hardware-enforced memory isolation (EPT/NPT)
    - Requirements: Linux host + KVM kernel module + /dev/kvm device
    - Use case: Production deployments, CI/CD

    TCG (Tiny Code Generator) - Development fallback:
    - Software-based CPU emulation (no hardware virtualization)
    - VM boot time: 2-5s (5-10x slower than KVM)
    - CPU overhead: 10-50x slower (instruction-level emulation)
    - Security: Software-based isolation (weaker than hardware)
    - Requirements: Any platform (Linux, macOS, Windows)
    - Use case: Development/testing only (macOS Docker Desktop)

    Production requirement: KVM is MANDATORY for performance and security.
    TCG is acceptable ONLY for local development and testing.

    Returns:
        True if both kernel and QEMU verify KVM is available
        False otherwise (falls back to TCG software emulation)
    """
    # Fast path: return cached result (no lock needed)
    if _probe_cache.kvm is not None:
        return _probe_cache.kvm

    # Slow path: acquire lock to prevent stampede, then check cache again
    async with _probe_cache.get_lock("kvm"):
        # Double-check after acquiring lock (another task may have populated cache)
        if _probe_cache.kvm is not None:
            return _probe_cache.kvm

        kvm_path = "/dev/kvm"
        if not await aiofiles.os.path.exists(kvm_path):
            logger.debug("KVM not available: /dev/kvm does not exist")
            _probe_cache.kvm = False
            return False

        # Check if we can actually access /dev/kvm (not just that it exists)
        # This catches permission issues that would cause QEMU to fail or hang
        # See: https://github.com/actions/runner-images/issues/8542
        if not await can_access(kvm_path, os.R_OK | os.W_OK):
            logger.debug("KVM not available: permission denied on /dev/kvm")
            _probe_cache.kvm = False
            return False

        # Actually try to open /dev/kvm and check API version via subprocess
        # Some environments (nested VMs, containers) have /dev/kvm but it doesn't work
        # Uses subprocess to avoid blocking the event loop with ioctl()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                f"import fcntl; f=open('{kvm_path}','rb'); print(fcntl.ioctl(f.fileno(), {_KVM_GET_API_VERSION}))",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode != 0:
                logger.warning("KVM device accessible but ioctl failed")
                _probe_cache.kvm = False
                return False

            api_version = int(stdout.decode().strip())
            if api_version != _KVM_API_VERSION_EXPECTED:
                logger.warning(
                    "KVM available but unexpected API version",
                    extra={"api_version": api_version, "expected": _KVM_API_VERSION_EXPECTED},
                )
                _probe_cache.kvm = False
                return False

            logger.debug("KVM ioctl check passed", extra={"api_version": api_version})

        except (OSError, TimeoutError, ValueError) as e:
            logger.debug("KVM not available: failed to verify /dev/kvm", extra={"error": str(e)})
            _probe_cache.kvm = False
            return False

        # Layer 2: Verify QEMU binary has KVM support compiled in
        # Even if /dev/kvm works, QEMU may not have KVM support or may fail to initialize it
        # See: https://github.com/orgs/community/discussions/8305 (GitHub Actions KVM issues)
        qemu_accels = await _probe_qemu_accelerators()
        if "kvm" not in qemu_accels:
            logger.warning(
                "KVM not available: QEMU binary does not support KVM accelerator",
                extra={"available_accelerators": sorted(qemu_accels)},
            )
            _probe_cache.kvm = False
            return False

        logger.debug("KVM available and working (kernel + QEMU verified)")
        _probe_cache.kvm = True
        return _probe_cache.kvm


async def _check_hvf_available() -> bool:
    """Check if HVF (Hypervisor.framework) acceleration is available on macOS (cached).

    HVF requires:
    - macOS host (automatically implied by caller)
    - CPU with virtualization extensions
    - Hypervisor entitlement (usually available)
    - NOT running inside a VM without nested virtualization

    GitHub Actions macOS runners run inside VMs without nested virtualization,
    so HVF is not available there. This check detects that case.

    Returns:
        True if HVF is available and can be used
        False otherwise (falls back to TCG software emulation)
    """
    # Fast path: return cached result (no lock needed)
    if _probe_cache.hvf is not None:
        return _probe_cache.hvf

    # Slow path: acquire lock to prevent stampede, then check cache again
    async with _probe_cache.get_lock("hvf"):
        # Double-check after acquiring lock (another task may have populated cache)
        if _probe_cache.hvf is not None:
            return _probe_cache.hvf

        try:
            # sysctl kern.hv_support returns 1 if Hypervisor.framework is available
            proc = await asyncio.create_subprocess_exec(
                "/usr/sbin/sysctl",
                "-n",
                "kern.hv_support",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            hvf_kernel_support = proc.returncode == 0 and stdout.decode().strip() == "1"

            if not hvf_kernel_support:
                logger.debug("HVF not available: kern.hv_support is not enabled")
                _probe_cache.hvf = False
                return False

            logger.debug("HVF kernel support check passed")

        except (OSError, TimeoutError) as e:
            logger.debug("HVF not available: sysctl check failed", extra={"error": str(e)})
            _probe_cache.hvf = False
            return False

        # Layer 2: Verify QEMU binary has HVF support compiled in
        # Even if kern.hv_support is enabled, QEMU may not have HVF support
        qemu_accels = await _probe_qemu_accelerators()
        if "hvf" not in qemu_accels:
            logger.warning(
                "HVF not available: QEMU binary does not support HVF accelerator",
                extra={"available_accelerators": sorted(qemu_accels)},
            )
            _probe_cache.hvf = False
            return False

        logger.debug("HVF available and working (kernel + QEMU verified)")
        _probe_cache.hvf = True
        return _probe_cache.hvf


def check_hwaccel_available() -> bool:
    """Check if hardware acceleration (KVM or HVF) is available.

    Synchronous function for use in pytest skipif markers.
    TCG (software emulation) is 10-50x slower than hardware virtualization,
    making timing-sensitive tests unreliable.

    Returns:
        True if KVM (Linux) or HVF (macOS) is available
        False otherwise (will use TCG software emulation)
    """
    host_os = detect_host_os()

    if host_os == HostOS.LINUX:
        return asyncio.run(_check_kvm_available())
    if host_os == HostOS.MACOS:
        return asyncio.run(_check_hvf_available())
    return False


def check_fast_balloon_available() -> bool:
    """Check if fast balloon operations are expected (not degraded nested virtualization).

    Synchronous function for use in pytest skipif markers.
    Used for timing-sensitive tests that include balloon inflate/deflate overhead.

    Background
    ==========
    Balloon operations (memory reclaim via virtio-balloon) have vastly different
    performance characteristics depending on the virtualization environment:

    - Bare-metal KVM: Balloon operations complete in <100ms
    - Nested KVM (CI runners): Balloon operations can take 5+ seconds due to
      hypervisor overhead, often timing out after retry limits

    The Problem
    ===========
    GitHub Actions runners are VMs on Azure, creating nested virtualization when
    running QEMU. Even when /dev/kvm exists and KVM "works", balloon operations
    are significantly degraded:

    1. KVM availability is inconsistent on GitHub Actions - "/dev/kvm sometimes
       exists (and works!), and sometimes it doesn't" (GitHub community #8305)

    2. pytest-xdist workers perform independent test collection, so flaky KVM
       detection can cause tests to run on workers where KVM isn't actually fast

    3. TSC_DEADLINE timer (required for efficient APIC timer virtualization) is
       often not exposed to nested VMs, causing timer fallback to slower modes

    Solution
    ========
    Use TSC_DEADLINE availability as a proxy for "fast virtualization":

    - TSC_DEADLINE is a CPU feature, not a kernel module state - deterministic
    - When missing, QEMU enables legacy PIT/PIC timers (slower, more overhead)
    - Reliably identifies degraded nested virt vs bare-metal/L1 KVM

    References
    ==========
    - Linux kernel timekeeping: https://docs.kernel.org/virt/kvm/x86/timekeeping.html
    - QEMU Hyper-V enlightenments: https://www.qemu.org/docs/master/system/i386/hyperv.html
    - GitHub Actions KVM issues: https://github.com/actions/runner-images/issues/8542
    - pytest-xdist collection: https://pytest-xdist.readthedocs.io/en/stable/how-it-works.html

    Returns:
        True if balloon operations are expected to be fast:
          - Linux x86_64: KVM available AND TSC_DEADLINE available
          - Linux ARM64: KVM available (ARM uses different timer, less affected)
          - macOS: HVF available (nested virt not possible on macOS)
        False otherwise (balloon operations may be slow, skip timing tests)
    """
    if not check_hwaccel_available():
        return False

    host_os = detect_host_os()
    host_arch = detect_host_arch()

    # On Linux x86_64, TSC_DEADLINE absence indicates degraded nested virt
    # See: https://www.qemu.org/docs/master/system/i386/microvm.html
    if host_os == HostOS.LINUX and host_arch == HostArch.X86_64:
        return asyncio.run(_check_tsc_deadline())

    # On macOS, HVF availability implies not nested (macOS doesn't support nested virt)
    # If we got here, HVF is available, so balloon should be fast
    if host_os == HostOS.MACOS:
        return True

    # On ARM64 Linux, KVM availability is sufficient
    # ARM uses GIC timer (not TSC/APIC), less affected by nested virt overhead
    # Unknown platform: conservative assumption (return False)
    return host_os == HostOS.LINUX and host_arch == HostArch.AARCH64


async def _check_tsc_deadline() -> bool:
    """Check if TSC_DEADLINE CPU feature is available (cached).

    TSC_DEADLINE is required to disable PIT (i8254) and PIC (i8259) in microvm.
    Without TSC_DEADLINE, the APIC timer cannot use deadline mode, and the system
    needs the legacy PIT for timer interrupts.

    In nested virtualization (e.g., GitHub Actions runners), TSC_DEADLINE may not
    be exposed to the guest, causing boot hangs if PIT/PIC are disabled.

    See: https://www.qemu.org/docs/master/system/i386/microvm.html

    Returns:
        True if TSC_DEADLINE is available, False otherwise
    """
    # Fast path: return cached result (no lock needed)
    if _probe_cache.tsc_deadline is not None:
        return _probe_cache.tsc_deadline

    # Slow path: acquire lock to prevent stampede
    async with _probe_cache.get_lock("tsc_deadline"):
        # Double-check after acquiring lock
        if _probe_cache.tsc_deadline is not None:
            return _probe_cache.tsc_deadline

        # TSC_DEADLINE is x86-only
        if detect_host_arch() != HostArch.X86_64:
            _probe_cache.tsc_deadline = False
            return False

        # Dispatch to platform-specific implementation
        host_os = detect_host_os()
        if host_os == HostOS.LINUX:
            return await _check_tsc_deadline_linux()
        if host_os == HostOS.MACOS:
            return await _check_tsc_deadline_macos()

        # Unknown platform
        _probe_cache.tsc_deadline = False
        return False


async def _check_tsc_deadline_linux() -> bool:
    """Linux-specific TSC_DEADLINE check via /proc/cpuinfo.

    Note: Called from _check_tsc_deadline() which handles caching and locking.
    """
    cpuinfo_path = "/proc/cpuinfo"
    if not await aiofiles.os.path.exists(cpuinfo_path):
        _probe_cache.tsc_deadline = False
        return False

    try:
        async with aiofiles.open(cpuinfo_path) as f:
            cpuinfo = await f.read()
        # Look for tsc_deadline_timer in the flags line
        # Format: "flags : fpu vme ... tsc_deadline_timer ..."
        for line in cpuinfo.split("\n"):
            if line.startswith("flags"):
                has_tsc = "tsc_deadline_timer" in line.split()
                _probe_cache.tsc_deadline = has_tsc
                if has_tsc:
                    logger.debug("TSC_DEADLINE available (can disable PIT/PIC)")
                else:
                    logger.debug("TSC_DEADLINE not available (keeping PIT/PIC enabled)")
                return has_tsc
    except OSError as e:
        logger.warning("Failed to read /proc/cpuinfo for TSC_DEADLINE check", extra={"error": str(e)})

    _probe_cache.tsc_deadline = False
    return False


async def _check_tsc_deadline_macos() -> bool:
    """macOS-specific TSC_DEADLINE check via sysctl.

    Note: Called from _check_tsc_deadline() which handles caching and locking.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "/usr/sbin/sysctl",
            "-n",
            "machdep.cpu.features",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode == 0:
            features = stdout.decode().upper()
            has_tsc = "TSC_DEADLINE" in features or "TSCDEAD" in features
            _probe_cache.tsc_deadline = has_tsc
            if has_tsc:
                logger.debug("TSC_DEADLINE available on macOS (can disable PIT/PIC)")
            else:
                logger.debug("TSC_DEADLINE not available on macOS (keeping legacy timers)")
            return has_tsc
    except (OSError, TimeoutError):
        pass

    _probe_cache.tsc_deadline = False
    return False


async def _probe_io_uring_support() -> bool:
    """Probe for io_uring support using syscall test (cached).

    Returns:
        True if io_uring fully available, False otherwise
    """
    # Fast path: return cached result (no lock needed)
    if _probe_cache.io_uring is not None:
        return _probe_cache.io_uring

    # Slow path: acquire lock to prevent stampede
    async with _probe_cache.get_lock("io_uring"):
        # Double-check after acquiring lock
        if _probe_cache.io_uring is not None:
            return _probe_cache.io_uring

        # io_uring is Linux-only - immediately return False on other platforms
        if detect_host_os() != HostOS.LINUX:
            _probe_cache.io_uring = False
            return False

        # Check 1: Sysctl restrictions (kernel 5.12+)
        sysctl_path = "/proc/sys/kernel/io_uring_disabled"
        if await aiofiles.os.path.exists(sysctl_path):
            try:
                async with aiofiles.open(sysctl_path) as f:
                    content = await f.read()
                disabled_value = int(content.strip())
                # io_uring_disabled sysctl values: 0=enabled, 1=restricted, 2=disabled
                if disabled_value == 2:  # noqa: PLR2004
                    logger.info(
                        "io_uring disabled via sysctl",
                        extra={"sysctl_value": disabled_value},
                    )
                    _probe_cache.io_uring = False
                    return False
                if disabled_value == 1:
                    logger.debug(
                        "io_uring restricted to CAP_SYS_ADMIN",
                        extra={"sysctl_value": disabled_value},
                    )
            except (ValueError, OSError) as e:
                logger.warning("Failed to read io_uring_disabled sysctl", extra={"error": str(e)})

        # Check 2: Syscall probe via subprocess (avoids blocking event loop)
        # Uses subprocess to prevent blocking - ctypes syscall would block the event loop
        # Exit codes: 0=available (EINVAL/EFAULT), 1=not available (ENOSYS), 2=blocked (EPERM), 3=error
        try:
            probe_script = """
import ctypes
import errno
import sys

try:
    libc = ctypes.CDLL(None, use_errno=True)
    # __NR_io_uring_setup = 425
    result = libc.syscall(425, 0, None)
    if result == -1:
        err = ctypes.get_errno()
        if err == errno.ENOSYS:
            sys.exit(1)  # Not available
        if err in (errno.EINVAL, errno.EFAULT):
            sys.exit(0)  # Available (kernel recognized syscall)
        if err == errno.EPERM:
            sys.exit(2)  # Blocked by seccomp/container
        sys.exit(3)  # Unexpected error
    sys.exit(0)  # Available
except Exception:
    sys.exit(3)  # Error
"""
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                probe_script,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5)

            if proc.returncode == 0:
                logger.info(
                    "io_uring syscall available",
                    extra={"kernel": platform.release()},
                )
                _probe_cache.io_uring = True
                return True
            if proc.returncode == 1:
                logger.info(
                    "io_uring syscall not available (ENOSYS)",
                    extra={"kernel": platform.release()},
                )
                _probe_cache.io_uring = False
                return False
            if proc.returncode == 2:  # noqa: PLR2004
                logger.warning(
                    "io_uring blocked by seccomp/container policy",
                    extra={"kernel": platform.release()},
                )
                _probe_cache.io_uring = False
                return False

            logger.warning(
                "io_uring probe failed with unexpected result",
                extra={"exit_code": proc.returncode},
            )
            _probe_cache.io_uring = False
            return False

        except (OSError, TimeoutError) as e:
            logger.warning(
                "io_uring syscall probe failed",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            _probe_cache.io_uring = False
            return False


async def _probe_unshare_support() -> bool:
    """Probe for unshare (Linux namespace) support (cached).

    Tests if the current environment allows creating new namespaces via unshare.
    This requires either:
    - Root privileges
    - CAP_SYS_ADMIN capability
    - Unprivileged user namespaces enabled (/proc/sys/kernel/unprivileged_userns_clone=1)

    Returns:
        True if unshare works, False otherwise (skip namespace isolation)
    """
    # Fast path: return cached result (no lock needed)
    if _probe_cache.unshare is not None:
        return _probe_cache.unshare

    # Slow path: acquire lock to prevent stampede
    async with _probe_cache.get_lock("unshare"):
        # Double-check after acquiring lock
        if _probe_cache.unshare is not None:
            return _probe_cache.unshare

        # Skip on non-Linux - unshare is Linux-specific
        if detect_host_os() == HostOS.MACOS:
            _probe_cache.unshare = False
            return False

        try:
            # Test unshare with minimal namespaces (pid requires fork)
            proc = await asyncio.create_subprocess_exec(
                "/usr/bin/unshare",
                "--pid",
                "--fork",
                "--",
                "/usr/bin/true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)

            if proc.returncode == 0:
                logger.info("unshare available (namespace isolation enabled)")
                _probe_cache.unshare = True
            else:
                stderr_text = stderr.decode().strip() if stderr else ""
                logger.warning(
                    "unshare unavailable (namespace isolation disabled)",
                    extra={"exit_code": proc.returncode, "stderr": stderr_text[:200]},
                )
                _probe_cache.unshare = False
        except (OSError, TimeoutError) as e:
            logger.warning(
                "unshare probe failed",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            _probe_cache.unshare = False

        return _probe_cache.unshare


# Pre-flight validation cache keyed by (kernel_path, arch)
_kernel_validated: set[tuple[Path, HostArch]] = set()


async def _validate_kernel_initramfs(kernel_path: Path, arch: HostArch) -> None:
    """Pre-flight check: validate kernel and initramfs exist (cached, one-time per config).

    This is NOT a probe (optional feature) - it's a hard requirement.
    Raises VmError if files are missing.
    """
    cache_key = (kernel_path, arch)
    if cache_key in _kernel_validated:
        return

    arch_suffix = "aarch64" if arch == HostArch.AARCH64 else "x86_64"
    kernel = kernel_path / f"vmlinuz-{arch_suffix}"
    initramfs = kernel_path / f"initramfs-{arch_suffix}"

    if not await aiofiles.os.path.exists(kernel):
        raise VmError(
            f"Kernel not found: {kernel}",
            context={"kernel_path": str(kernel), "arch": arch_suffix},
        )
    if not await aiofiles.os.path.exists(initramfs):
        raise VmError(
            f"Initramfs not found: {initramfs}",
            context={"initramfs_path": str(initramfs), "arch": arch_suffix},
        )

    _kernel_validated.add(cache_key)


class QemuVM:
    """Handle to running QEMU microVM.

    Lifecycle managed by VmManager.
    Communicates via TCP.

    Security:
    - Layer 1: Hardware isolation (KVM) or TCG software emulation
    - Layer 2: Unprivileged user (qemu-vm if available, optional)
    - Layer 3: Seccomp syscall filtering
    - Layer 4: cgroup v2 resource limits
    - Layer 5: Linux namespaces (PID, net, mount, UTS, IPC)
    - Layer 6: SELinux/AppArmor (optional production hardening)

    Context Manager Usage:
        Supports async context manager protocol for automatic cleanup:

        ```python
        async with await manager.launch_vm(...) as vm:
            result = await vm.execute(code="print('hello')", timeout_seconds=30)
            # VM automatically destroyed on exit, even if exception occurs
        ```

        Manual cleanup still available via destroy() method for explicit control.

    Attributes:
        vm_id: Unique VM identifier format: {tenant_id}-{task_id}-{uuid4}
        process: QEMU subprocess handle
        cgroup_path: cgroup v2 path for resource limits
        workdir: Working directory containing all VM temp files
        overlay_image: Ephemeral qcow2 overlay (property, from workdir)
        gvproxy_proc: Optional gvproxy-wrapper process for DNS filtering
        gvproxy_socket: Optional QEMU stream socket path (property, from workdir)
        gvproxy_log_task: Optional background task draining gvproxy stdout/stderr
    """

    def __init__(
        self,
        vm_id: str,
        process: ProcessWrapper,
        cgroup_path: Path,
        workdir: VmWorkingDirectory,
        channel: GuestChannel,
        language: Language,
        gvproxy_proc: ProcessWrapper | None = None,
        qemu_log_task: asyncio.Task[None] | None = None,
        gvproxy_log_task: asyncio.Task[None] | None = None,
        console_log: TextIO | None = None,
    ):
        """Initialize VM handle.

        Args:
            vm_id: Unique VM identifier (scoped by tenant_id)
            process: Running QEMU subprocess (ProcessWrapper for PID-reuse safety)
            cgroup_path: cgroup v2 path for cleanup
            workdir: Working directory containing overlay, sockets, and logs
            channel: Communication channel for TCP guest agent
            language: Programming language for this VM
            gvproxy_proc: Optional gvproxy-wrapper process (ProcessWrapper)
            qemu_log_task: Background task draining QEMU stdout/stderr (prevents pipe deadlock)
            gvproxy_log_task: Background task draining gvproxy stdout/stderr (prevents pipe deadlock)
            console_log: Optional file handle for QEMU console log
        """
        self.vm_id = vm_id
        self.process = process
        self.cgroup_path = cgroup_path
        self.workdir = workdir
        self.channel = channel
        self.language = language
        self.gvproxy_proc = gvproxy_proc
        self.qemu_log_task = qemu_log_task
        self.gvproxy_log_task = gvproxy_log_task
        self.console_log: TextIO | None = console_log
        self._destroyed = False
        self._state = VmState.CREATING
        self._state_lock = asyncio.Lock()
        # Timing instrumentation (set by VmManager.create_vm)
        self.timing = VmTiming()

    # -------------------------------------------------------------------------
    # Timing properties (backwards-compatible accessors to VmTiming)
    # -------------------------------------------------------------------------

    @property
    def setup_ms(self) -> int | None:
        """Get resource setup time in milliseconds."""
        return self.timing.setup_ms

    @setup_ms.setter
    def setup_ms(self, value: int) -> None:
        """Set resource setup time in milliseconds."""
        self.timing.setup_ms = value

    @property
    def overlay_ms(self) -> int | None:
        """Get overlay acquisition time in milliseconds."""
        return self.timing.overlay_ms

    @overlay_ms.setter
    def overlay_ms(self, value: int) -> None:
        """Set overlay acquisition time in milliseconds."""
        self.timing.overlay_ms = value

    @property
    def boot_ms(self) -> int | None:
        """Get VM boot time in milliseconds."""
        return self.timing.boot_ms

    @boot_ms.setter
    def boot_ms(self, value: int) -> None:
        """Set VM boot time in milliseconds."""
        self.timing.boot_ms = value

    @property
    def qemu_cmd_build_ms(self) -> int | None:
        """Time for pre-launch setup (command build, socket cleanup, channel creation)."""
        return self.timing.qemu_cmd_build_ms

    @qemu_cmd_build_ms.setter
    def qemu_cmd_build_ms(self, value: int) -> None:
        self.timing.qemu_cmd_build_ms = value

    @property
    def gvproxy_start_ms(self) -> int | None:
        """Time to start gvproxy (0 if network disabled)."""
        return self.timing.gvproxy_start_ms

    @gvproxy_start_ms.setter
    def gvproxy_start_ms(self, value: int) -> None:
        self.timing.gvproxy_start_ms = value

    @property
    def qemu_fork_ms(self) -> int | None:
        """Time for QEMU process fork/exec."""
        return self.timing.qemu_fork_ms

    @qemu_fork_ms.setter
    def qemu_fork_ms(self, value: int) -> None:
        self.timing.qemu_fork_ms = value

    @property
    def guest_wait_ms(self) -> int | None:
        """Time waiting for guest agent (kernel + initramfs + agent init)."""
        return self.timing.guest_wait_ms

    @guest_wait_ms.setter
    def guest_wait_ms(self, value: int) -> None:
        self.timing.guest_wait_ms = value

    # -------------------------------------------------------------------------
    # Other VM properties
    # -------------------------------------------------------------------------

    @property
    def overlay_image(self) -> Path:
        """Path to overlay image (from workdir)."""
        return self.workdir.overlay_image

    @property
    def gvproxy_socket(self) -> Path | None:
        """Path to gvproxy socket (from workdir, None if no network)."""
        return self.workdir.gvproxy_socket if self.gvproxy_proc else None

    @property
    def use_qemu_vm_user(self) -> bool:
        """Whether QEMU runs as qemu-vm user."""
        return self.workdir.use_qemu_vm_user

    @property
    def qmp_socket(self) -> Path:
        """Path to QMP control socket (from workdir)."""
        return self.workdir.qmp_socket

    async def __aenter__(self) -> "QemuVM":
        """Enter async context manager.

        Returns:
            Self for use in async with statement
        """
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> bool:
        """Exit async context manager - ensure cleanup.

        Returns:
            False to propagate exceptions
        """
        # Cleanup VM when exiting context (always runs destroy)
        # destroy() is idempotent and state-safe, will skip if already destroying/destroyed
        await self.destroy()
        return False  # Don't suppress exceptions

    @property
    def state(self) -> VmState:
        """Current VM state."""
        return self._state

    async def transition_state(self, new_state: VmState) -> None:
        """Transition VM to new state with validation.

        Validates state transition against VALID_STATE_TRANSITIONS to prevent
        invalid state changes (e.g., DESTROYED → READY).

        Args:
            new_state: Target state to transition to

        Raises:
            VmError: If transition is invalid for current state
        """
        async with self._state_lock:
            # Validate transition is allowed from current state
            allowed_transitions = VALID_STATE_TRANSITIONS.get(self._state, set())
            if new_state not in allowed_transitions:
                raise VmError(
                    f"Invalid state transition: {self._state.value} → {new_state.value}",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                        "target_state": new_state.value,
                        "allowed_transitions": [s.value for s in allowed_transitions],
                    },
                )

            old_state = self._state
            self._state = new_state
            logger.debug(
                "VM state transition",
                extra={
                    "debug_category": "lifecycle",
                    "vm_id": self.vm_id,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                },
            )

    async def execute(  # noqa: PLR0912, PLR0915
        self,
        code: str,
        timeout_seconds: int,
        env_vars: dict[str, str] | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Execute code via TCP guest agent communication.

        Implementation:
        1. Connect to guest via TCP (127.0.0.1 + allocated port)
        2. Send execution request JSON with action, language, code, timeout, env_vars
        3. Wait for result with timeout (cgroup enforced)
        4. Parse result: stdout, stderr, exit_code, memory_mb, execution_time_ms

        Timeout Architecture (3-layer system):
        1. Init timeout (5s): Connection establishment to guest agent
        2. Soft timeout (timeout_seconds): Guest agent enforcement (sent in request)
        3. Hard timeout (timeout_seconds + 2s): Host watchdog protection

        Example: timeout_seconds=30
        - connect(5s) - Fixed init window for socket establishment
        - send_request(timeout=32s) - 30s soft + 2s margin
        - Guest enforces 30s, host kills at 32s if guest hangs

        Args:
            code: Code to execute in guest VM
            timeout_seconds: Maximum execution time (enforced by cgroup)
            env_vars: Environment variables for code execution (default: None)
            on_stdout: Optional callback for real-time stdout streaming
            on_stderr: Optional callback for real-time stderr streaming

        Returns:
            ExecutionResult with stdout, stderr, exit code, and resource usage

        Raises:
            VmError: VM not in READY state or communication failed
            VmTimeoutError: Execution exceeded timeout_seconds
        """
        # Validate VM is in READY state before execution (atomic check-and-set)
        async with self._state_lock:
            if self._state != VmState.READY:
                raise VmError(
                    f"Cannot execute in state {self._state.value}, must be READY",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                        "language": self.language,
                    },
                )

            # Validate transition to EXECUTING (inline to avoid lock re-acquisition)
            allowed_transitions = VALID_STATE_TRANSITIONS.get(self._state, set())
            if VmState.EXECUTING not in allowed_transitions:
                raise VmError(
                    f"Invalid state transition: {self._state.value} → {VmState.EXECUTING.value}",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                        "target_state": VmState.EXECUTING.value,
                        "allowed_transitions": [s.value for s in allowed_transitions],
                    },
                )

            # Transition to EXECUTING inside same lock
            old_state = self._state
            self._state = VmState.EXECUTING
            logger.debug(
                "VM state transition",
                extra={
                    "debug_category": "lifecycle",
                    "vm_id": self.vm_id,
                    "old_state": old_state.value,
                    "new_state": self._state.value,
                },
            )

        # Prepare execution request
        request = ExecuteCodeRequest(
            language=self.language,
            code=code,
            timeout=timeout_seconds,
            env_vars=env_vars or {},
        )

        try:
            # Re-check state before expensive I/O operations
            # Between lock release and here, destroy() could have been called
            # which would transition state to DESTROYING or DESTROYED
            # Note: pyright doesn't understand async race conditions, so we suppress the warning
            if self._state in (VmState.DESTROYING, VmState.DESTROYED):  # type: ignore[comparison-overlap]
                raise VmError(
                    "VM destroyed during execution start",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                    },
                )

            # Connect to guest via TCP with timing
            # Fixed init timeout (connection establishment, independent of execution timeout)
            connect_start = asyncio.get_event_loop().time()
            await self.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
            connect_ms = round((asyncio.get_event_loop().time() - connect_start) * 1000)

            # Stream execution output to console
            # Hard timeout = soft timeout (guest enforcement) + margin (host watchdog)
            hard_timeout = timeout_seconds + constants.EXECUTION_TIMEOUT_MARGIN_SECONDS

            # Stream messages and collect output
            exit_code = -1
            execution_time_ms: int | None = None
            spawn_ms: int | None = None
            process_ms: int | None = None
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            async for msg in self.channel.stream_messages(request, timeout=hard_timeout):
                # Type-safe message handling
                from exec_sandbox.guest_agent_protocol import (  # noqa: PLC0415
                    ExecutionCompleteMessage,
                    OutputChunkMessage,
                    StreamingErrorMessage,
                )

                if isinstance(msg, OutputChunkMessage):
                    # Collect chunk for return to user
                    if msg.type == "stdout":
                        stdout_chunks.append(msg.chunk)
                        # Call streaming callback if provided
                        if on_stdout:
                            on_stdout(msg.chunk)
                    else:  # stderr
                        stderr_chunks.append(msg.chunk)
                        # Call streaming callback if provided
                        if on_stderr:
                            on_stderr(msg.chunk)

                    # Also log for debugging (truncated)
                    logger.debug(
                        "VM output",
                        extra={
                            "vm_id": self.vm_id,
                            "stream": msg.type,
                            "chunk": msg.chunk[:200],
                        },
                    )
                elif isinstance(msg, ExecutionCompleteMessage):
                    # Execution complete - capture all timing fields
                    exit_code = msg.exit_code
                    execution_time_ms = msg.execution_time_ms
                    spawn_ms = msg.spawn_ms
                    process_ms = msg.process_ms
                elif isinstance(msg, StreamingErrorMessage):
                    # Streaming error from guest - include details in log message
                    logger.error(
                        f"Guest agent error: [{msg.error_type}] {msg.message}",
                        extra={
                            "vm_id": self.vm_id,
                            "error_message": msg.message,
                            "error_type": msg.error_type,
                        },
                    )
                    # Store error in stderr so callers can see what went wrong
                    stderr_chunks.append(f"[{msg.error_type}] {msg.message}")
                    exit_code = -1
                    break

            # Measure external resources from host (cgroup v2)
            external_cpu_ms, external_mem_mb = await self._read_cgroup_stats()

            # Concatenate collected chunks
            stdout_full = "".join(stdout_chunks)
            stderr_full = "".join(stderr_chunks)

            # Truncate to limits
            stdout_truncated = stdout_full[: constants.MAX_STDOUT_SIZE]
            stderr_truncated = stderr_full[: constants.MAX_STDERR_SIZE]

            # Debug log final execution output
            logger.debug(
                "Code execution complete",
                extra={
                    "vm_id": self.vm_id,
                    "exit_code": exit_code,
                    "execution_time_ms": execution_time_ms,
                    "stdout_len": len(stdout_full),
                    "stderr_len": len(stderr_full),
                    "stdout": stdout_truncated[:500],  # First 500 chars for debug
                    "stderr": stderr_truncated[:500] if stderr_truncated else None,
                },
            )

            # Parse result with both internal (guest) and external (host) measurements
            # Note: timing is a placeholder here - scheduler will populate actual values
            exec_result = ExecutionResult(
                stdout=stdout_truncated,  # Return to user
                stderr=stderr_truncated,  # Return to user
                exit_code=exit_code,
                execution_time_ms=execution_time_ms,  # Guest-reported
                external_cpu_time_ms=external_cpu_ms or None,  # Host-measured
                external_memory_peak_mb=external_mem_mb or None,  # Host-measured
                timing=TimingBreakdown(setup_ms=0, boot_ms=0, execute_ms=0, total_ms=0, connect_ms=connect_ms),
                spawn_ms=spawn_ms,  # Guest-reported granular timing
                process_ms=process_ms,  # Guest-reported granular timing
            )

            # Success - transition back to READY for reuse (if not destroyed)
            try:
                await self.transition_state(VmState.READY)
            except VmError as e:
                # VM destroyed while executing, skip transition
                logger.debug(
                    "VM destroyed during execution, skipping READY transition",
                    extra={"vm_id": self.vm_id, "error": str(e)},
                )
            return exec_result

        except asyncio.CancelledError:
            logger.warning(
                "Code execution cancelled",
                extra={"vm_id": self.vm_id, "language": self.language},
            )
            # Re-raise to propagate cancellation
            raise
        except TimeoutError as e:
            raise VmTimeoutError(
                f"VM {self.vm_id} execution exceeded {timeout_seconds}s timeout",
                context={
                    "vm_id": self.vm_id,
                    "timeout_seconds": timeout_seconds,
                    "language": self.language,
                },
            ) from e
        except (OSError, json.JSONDecodeError) as e:
            raise VmError(
                f"VM {self.vm_id} communication failed: {e}",
                context={
                    "vm_id": self.vm_id,
                    "language": self.language,
                    "error_type": type(e).__name__,
                },
            ) from e

    async def _read_cgroup_stats(self) -> tuple[int | None, int | None]:
        """Read external CPU time and peak memory from cgroup v2.

        Returns:
            Tuple of (cpu_time_ms, peak_memory_mb)
            Returns (None, None) if cgroup not available or read fails
        """
        return await cgroup.read_cgroup_stats(self.cgroup_path)

    async def destroy(self) -> None:
        """Clean up VM and resources.

        Cleanup steps:
        1. Close communication channel
        2. Terminate QEMU process (SIGTERM → SIGKILL if needed)
        3. Remove cgroup
        4. Delete ephemeral overlay image

        Called automatically by VmManager after execution or on error.
        Idempotent: safe to call multiple times.

        State Lock Strategy:
        - Lock held during state check + transition to DESTROYING
        - Released during cleanup (blocking I/O operations)
        - DESTROYING state prevents concurrent destroy() from proceeding
        """
        # Atomic state check and transition (prevent concurrent destroy)
        async with self._state_lock:
            if self._destroyed:
                logger.debug("VM already destroyed, skipping", extra={"vm_id": self.vm_id})
                return

            # Set destroyed flag immediately to prevent concurrent destroy
            self._destroyed = True

            # Validate transition to DESTROYING (inline to avoid lock re-acquisition)
            allowed_transitions = VALID_STATE_TRANSITIONS.get(self._state, set())
            if VmState.DESTROYING not in allowed_transitions:
                raise VmError(
                    f"Invalid state transition: {self._state.value} → {VmState.DESTROYING.value}",
                    context={
                        "vm_id": self.vm_id,
                        "current_state": self._state.value,
                        "target_state": VmState.DESTROYING.value,
                        "allowed_transitions": [s.value for s in allowed_transitions],
                    },
                )

            old_state = self._state
            self._state = VmState.DESTROYING
            logger.debug(
                "VM state transition",
                extra={
                    "debug_category": "lifecycle",
                    "vm_id": self.vm_id,
                    "old_state": old_state.value,
                    "new_state": self._state.value,
                },
            )

        # Cleanup operations outside lock (blocking I/O)
        # Step 1: Close communication channel
        with contextlib.suppress(OSError, RuntimeError):
            await self.channel.close()

        # Step 2: Terminate QEMU and gvproxy processes (SIGTERM → SIGKILL)
        await cleanup_vm_processes(self.process, self.gvproxy_proc, self.vm_id)

        # Step 3-4: Parallel cleanup (cgroup + workdir)
        # After QEMU terminates, cleanup tasks are independent
        # workdir.cleanup() removes overlay, sockets, and console log in one operation
        await asyncio.gather(
            cgroup.cleanup_cgroup(self.cgroup_path, self.vm_id),
            self.workdir.cleanup(),
            return_exceptions=True,
        )

        # Final state transition (acquires lock again - safe for same task)
        await self.transition_state(VmState.DESTROYED)


class VmManager:
    """QEMU microVM lifecycle manager with cross-platform support.

    Architecture:
    - Runtime detection: KVM or TCG acceleration
    - qcow2 snapshot-based boot with CoW overlays
    - 6-layer security architecture
    - TCP guest agent communication
    - cgroup v2 resource limits

    Usage:
        async with VmManager(settings) as manager:
            vm = await manager.create_vm(Language.PYTHON, "tenant-123", "task-456")
            result = await vm.execute("print('hello')", timeout_seconds=30)
            await manager.destroy_vm(vm)
    """

    def __init__(self, settings: Settings):
        """Initialize QEMU manager (sync part only).

        Args:
            settings: Service configuration (paths, limits, etc.)

        Note: Call `await start()` after construction to run async system probes.

        Note on crash recovery:
            VM registry is in-memory only. If service crashes, registry is lost
            but QEMU processes may still be running. On restart:
            - Registry initializes empty (logged below)
            - Zombie QEMU processes are orphaned (no cleanup attempted)
            - Orphaned VMs timeout naturally (max runtime: 2 min)
        """
        self.settings = settings
        self.arch = detect_host_arch()
        self._initialized = False

        self._vms: dict[str, QemuVM] = {}  # vm_id → VM object
        self._vms_lock = asyncio.Lock()  # Protect registry access

        # Overlay pool for fast VM boot (auto-manages base image discovery and pooling)
        self._overlay_pool = OverlayPool(
            max_concurrent_vms=settings.max_concurrent_vms,
            images_path=settings.base_images_dir,
        )

    async def start(self) -> None:
        """Start VmManager and run async system probes.

        This method runs all async system capability probes and caches their results
        at module level. This prevents cache stampede when multiple VMs start
        concurrently - all probes are pre-warmed here instead of racing during VM creation.

        Must be called before creating VMs.
        """
        if self._initialized:
            return

        # Run all async probes concurrently (they cache their results at module level)
        # This prevents cache stampede when multiple VMs start concurrently
        accel_type, io_uring_available, unshare_available = await asyncio.gather(
            self._detect_accel_type(),  # Pre-warms HVF/KVM + QEMU accelerator caches
            _probe_io_uring_support(),
            _probe_unshare_support(),
        )

        # Pre-warm TSC deadline (unified function handles arch/OS dispatch)
        await _check_tsc_deadline()

        # Pre-flight check: validate kernel and initramfs exist (cached)
        await _validate_kernel_initramfs(self.settings.kernel_path, self.arch)

        # Start overlay pool (discovers base images internally)
        await self._overlay_pool.start()

        self._initialized = True

        # Log registry initialization (empty on startup, even after crash)
        logger.info(
            "VM registry initialized",
            extra={
                "max_concurrent_vms": self.settings.max_concurrent_vms,
                "accel_type": accel_type.value,
                "io_uring_available": io_uring_available,
                "unshare_available": unshare_available,
                "note": "All system probes pre-warmed (stampede prevention)",
            },
        )

    def get_active_vms(self) -> dict[str, QemuVM]:
        """Get snapshot of active VMs (for debugging/metrics).

        Returns:
            Copy of VM registry (vm_id → QemuVM)
        """
        return dict(self._vms)

    async def stop(self) -> None:
        """Stop VmManager and cleanup resources (overlay pool).

        Should be called when the VmManager is no longer needed.
        """
        await self._overlay_pool.stop()

    async def __aenter__(self) -> "VmManager":
        """Enter async context manager, starting the manager."""
        await self.start()
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, stopping the manager."""
        await self.stop()

    async def create_vm(  # noqa: PLR0912, PLR0915
        self,
        language: Language,
        tenant_id: str,
        task_id: str,
        backing_image: Path | None = None,
        memory_mb: int = constants.DEFAULT_MEMORY_MB,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
        direct_write_target: Path | None = None,
    ) -> QemuVM:
        """Create and boot QEMU microVM.

        Workflow:
        1. Generate unique VM ID and CID
        2. Create ephemeral qcow2 overlay from backing image (or write directly)
        3. Set up cgroup v2 resource limits
        4. Build QEMU command (platform-specific)
        5. Launch QEMU subprocess
        6. Wait for guest agent ready

        Args:
            language: Programming language (python or javascript)
            tenant_id: Tenant identifier for isolation
            task_id: Task identifier
            backing_image: Base image for overlay (default: language base image)
            memory_mb: Memory limit in MB (128-2048, default 512)
            allow_network: Enable network access (default: False, isolated)
            allowed_domains: Whitelist of allowed domains if allow_network=True
            direct_write_target: If set, write directly to this file (no overlay).
                Used for snapshot creation. Mutually exclusive with backing_image.

        Returns:
            QemuVM handle for code execution

        Raises:
            VmError: VM creation failed
            asyncio.TimeoutError: VM boot timeout (>5s)
        """
        # Start timing
        start_time = asyncio.get_event_loop().time()

        # Security: Validate identifiers to prevent shell injection and path traversal
        # Must be done BEFORE any use of tenant_id/task_id in paths, commands, or IDs
        _validate_identifier(tenant_id, "tenant_id")
        _validate_identifier(task_id, "task_id")

        # Step 0: Validate kernel and initramfs exist (cached, one-time check)
        await _validate_kernel_initramfs(self.settings.kernel_path, self.arch)
        arch_suffix = "aarch64" if self.arch == HostArch.AARCH64 else "x86_64"
        kernel_path = self.settings.kernel_path / f"vmlinuz-{arch_suffix}"
        initramfs_path = self.settings.kernel_path / f"initramfs-{arch_suffix}"

        # Step 1: Generate VM identifiers
        vm_id = f"{tenant_id}-{task_id}-{uuid4()}"

        # Validate mutual exclusivity
        if backing_image and direct_write_target:
            raise VmError("backing_image and direct_write_target are mutually exclusive")

        # Step 1.5: Create working directory for all VM temp files
        # Uses tempfile.mkdtemp() for atomic, secure directory creation (mode 0700)
        # For direct_write_target mode, use it as the overlay path
        workdir = await VmWorkingDirectory.create(
            vm_id,
            custom_overlay_path=direct_write_target,
        )

        # Domain whitelist semantics:
        # - None or [] = no filtering (full internet access)
        # - list with domains = whitelist filtering via gvproxy
        logger.debug(
            "Network configuration",
            extra={
                "debug_category": "network",
                "vm_id": vm_id,
                "allow_network": allow_network,
                "allowed_domains": allowed_domains,
                "will_enable_filtering": bool(allowed_domains and len(allowed_domains) > 0),
            },
        )

        # Step 2: Determine virtualization mode early (needed for cgroup memory sizing)
        # TCG mode requires significantly more memory for translation block cache
        accel_type = await self._detect_accel_type()
        use_tcg = accel_type == AccelType.TCG

        # Step 3-4: Parallel resource setup (overlay + cgroup)
        # These operations are independent and can run concurrently
        # Note: gvproxy moved to boot phase to reduce contention under high concurrency
        # Resolve to absolute path - qemu-img resolves backing file relative to overlay location,
        # and VmWorkingDirectory places overlay in a temp dir, so relative paths would break
        base_image = (backing_image or self.get_base_image(language)).resolve()

        # Initialize ALL tracking variables before try block for finally cleanup
        cgroup_path: Path | None = None
        gvproxy_proc: ProcessWrapper | None = None
        gvproxy_log_task: asyncio.Task[None] | None = None
        qemu_proc: ProcessWrapper | None = None
        qemu_log_task: asyncio.Task[None] | None = None
        console_log: TextIO | None = None
        vm_created = False  # Flag to skip cleanup if VM successfully created

        # IMPORTANT: Always use gvproxy for network-enabled VMs
        # SLIRP user networking has reliability issues with containerized unprivileged execution
        enable_dns_filtering = allow_network  # Force gvproxy for all network-enabled VMs

        try:
            # Log network configuration for debugging
            logger.debug(
                "Network configuration",
                extra={
                    "debug_category": "network",
                    "vm_id": vm_id,
                    "allow_network": allow_network,
                    "allowed_domains": allowed_domains,
                    "domains_count": len(allowed_domains) if allowed_domains else 0,
                },
            )

            # Unified setup phase: overlay + cgroup (gvproxy moved to boot phase)
            # This reduces setup contention under high concurrency
            overlay_ms = 0  # Default for direct_write mode (no overlay)
            if direct_write_target:
                # Direct write mode - VM writes directly to target file (no overlay)
                # Used for L2 snapshot creation where disk changes are written directly
                workdir.use_qemu_vm_user = False  # target file owned by current user
                cgroup_path = await cgroup.setup_cgroup(vm_id, tenant_id, memory_mb, use_tcg)
            # Normal mode - create overlay backed by base image
            # This allows the backing image to remain read-only and shareable
            # Pool handles fast path (from pool) or slow path (on-demand) internally
            else:
                overlay_start = asyncio.get_event_loop().time()
                try:
                    await self._overlay_pool.acquire(base_image, workdir.overlay_image)
                except QemuImgError as e:
                    raise VmError(str(e)) from e
                overlay_ms = round((asyncio.get_event_loop().time() - overlay_start) * 1000)
                # Apply permissions in parallel with cgroup setup
                perm_result, cgroup_result = await asyncio.gather(
                    self._apply_overlay_permissions(base_image, workdir.overlay_image),
                    cgroup.setup_cgroup(vm_id, tenant_id, memory_mb, use_tcg),
                    return_exceptions=True,
                )
                if isinstance(perm_result, BaseException):
                    raise perm_result
                if isinstance(cgroup_result, BaseException):
                    raise cgroup_result
                cgroup_path = cgroup_result
                workdir.use_qemu_vm_user = perm_result
            setup_complete_time = asyncio.get_event_loop().time()

            # Step 5: Build QEMU command (always Linux in container)
            qemu_cmd_start = asyncio.get_event_loop().time()
            qemu_cmd = await self._build_linux_cmd(
                language,
                vm_id,
                workdir,
                memory_mb,
                allow_network,
                enable_dns_filtering,
            )

            # Step 6: Create dual-port Unix socket communication channel for guest agent
            # Socket paths are now in workdir (shorter, under 108-byte Unix socket limit)
            cmd_socket = workdir.cmd_socket
            event_socket = workdir.event_socket

            # Clean up any stale socket files before QEMU creates new ones
            # This ensures QEMU gets a clean state for chardev sockets
            for socket_path in [cmd_socket, event_socket, str(workdir.qmp_socket)]:
                with contextlib.suppress(OSError):
                    await aiofiles.os.remove(socket_path)

            # Determine expected UID for socket authentication (mandatory)
            # Verifies QEMU process identity before sending commands
            if workdir.use_qemu_vm_user:
                expected_uid = get_qemu_vm_uid()
                if expected_uid is None:
                    # qemu-vm user expected but doesn't exist - configuration error
                    raise VmError(
                        "qemu-vm user required for socket authentication but not found",
                        {"use_qemu_vm_user": True},
                    )
            else:
                expected_uid = os.getuid()

            channel: GuestChannel = DualPortChannel(cmd_socket, event_socket, expected_uid=expected_uid)

            # If cgroups unavailable, wrap with ulimit for host resource control
            # ulimit works on Linux, macOS, BSD (POSIX)
            if not cgroup.is_cgroup_available(cgroup_path):
                qemu_cmd = cgroup.wrap_with_ulimit(qemu_cmd, memory_mb)
            qemu_cmd_build_ms = round((asyncio.get_event_loop().time() - qemu_cmd_start) * 1000)

            # Boot phase: Start gvproxy BEFORE QEMU (if network enabled)
            gvproxy_start_time = asyncio.get_event_loop().time()
            gvproxy_start_ms = 0
            # gvproxy must create socket before QEMU connects to it
            # Moved from setup phase to boot phase to reduce contention under high concurrency
            if allow_network:
                logger.info(
                    "Starting gvproxy-wrapper in boot phase (before QEMU)",
                    extra={"vm_id": vm_id, "allowed_domains": allowed_domains},
                )
                gvproxy_proc, gvproxy_log_task = await self._start_gvproxy(vm_id, allowed_domains, language, workdir)
                # Attach gvproxy to cgroup for resource limits
                await cgroup.attach_if_available(cgroup_path, gvproxy_proc.pid)
                gvproxy_start_ms = round((asyncio.get_event_loop().time() - gvproxy_start_time) * 1000)

            # Step 7: Launch QEMU
            qemu_start_time = asyncio.get_event_loop().time()
            try:
                # Set umask 007 for qemu-vm user to create sockets with 0660 permissions
                # This is done via preexec_fn to avoid shell injection (no 'sh -c' needed)
                def _set_umask_007() -> None:
                    os.umask(0o007)

                qemu_proc = ProcessWrapper(
                    await asyncio.create_subprocess_exec(
                        *qemu_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True,  # Create new process group for proper cleanup
                        preexec_fn=_set_umask_007 if workdir.use_qemu_vm_user else None,
                    )
                )
                # Capture fork time immediately after subprocess creation
                qemu_fork_ms = round((asyncio.get_event_loop().time() - qemu_start_time) * 1000)

                # Attach process to cgroup (only if cgroups available)
                await cgroup.attach_if_available(cgroup_path, qemu_proc.pid)

                # Check if process crashed immediately BEFORE starting drain task
                # This ensures we can capture stderr/stdout on immediate crash
                await asyncio.sleep(0.02)
                if qemu_proc.returncode is not None:
                    stdout_text, stderr_text = await self._capture_qemu_output(qemu_proc)

                    # Build full command string for debugging
                    import shlex  # noqa: PLC0415

                    qemu_cmd_str = " ".join(shlex.quote(arg) for arg in qemu_cmd)

                    # Log detailed error for debugging
                    logger.error(
                        "QEMU crashed immediately",
                        extra={
                            "vm_id": vm_id,
                            "exit_code": qemu_proc.returncode,
                            "stderr": stderr_text[:2000] if stderr_text else "(empty)",
                            "stdout": stdout_text[:2000] if stdout_text else "(empty)",
                            "qemu_cmd": qemu_cmd_str[:2000],
                        },
                    )

                    max_bytes = constants.QEMU_OUTPUT_MAX_BYTES
                    raise VmError(
                        f"QEMU crashed immediately (exit code {qemu_proc.returncode}). "
                        f"stderr: {stderr_text[:max_bytes] if stderr_text else '(empty)'}, "
                        f"stdout: {stdout_text[:max_bytes] if stdout_text else '(empty)'}",
                        context={
                            "vm_id": vm_id,
                            "language": language,
                            "exit_code": qemu_proc.returncode,
                            "memory_mb": memory_mb,
                            "allow_network": allow_network,
                            "qemu_cmd": qemu_cmd_str,
                        },
                    )

                # Background task to drain QEMU output (prevent 64KB pipe deadlock)
                # Started AFTER crash check to ensure we can capture error output
                console_log_path = workdir.console_log

                # Open file asynchronously to avoid blocking the event loop
                # The file handle itself stays synchronous for simple write callbacks
                loop = asyncio.get_running_loop()
                console_log = await loop.run_in_executor(
                    None,
                    lambda: console_log_path.open("w", buffering=1),  # Line buffering
                )

                # Capture in local variable for type narrowing (console_log is definitely not None here)
                # Type assertion: we just opened the file above, so console_log is a valid TextIO
                assert console_log is not None  # noqa: S101 - type narrowing, not runtime check
                _console_log: TextIO = console_log

                def write_to_console(line: str) -> None:
                    """Write line to console log file and structured logs."""
                    try:
                        _console_log.write(f"[{vm_id}] {line}\n")
                    except OSError as e:
                        logger.error(f"Console write failed: {e}", extra={"context_id": vm_id})

                qemu_log_task = asyncio.create_task(
                    drain_subprocess_output(
                        qemu_proc,
                        process_name="QEMU",
                        context_id=vm_id,
                        stdout_handler=write_to_console,
                        stderr_handler=write_to_console,
                    )
                )
                qemu_log_task.add_done_callback(_log_task_exception)

            except (OSError, FileNotFoundError) as e:
                raise VmError(
                    f"Failed to launch QEMU: {e}",
                    context={
                        "vm_id": vm_id,
                        "language": language,
                        "memory_mb": memory_mb,
                    },
                ) from e

            # Step 8: Wait for guest agent ready
            guest_wait_start = asyncio.get_event_loop().time()
            vm = QemuVM(
                vm_id,
                qemu_proc,
                cgroup_path,
                workdir,
                channel,
                language,
                gvproxy_proc,
                qemu_log_task,
                gvproxy_log_task,
                console_log,
            )

            # Step 8a: Register VM in registry (before BOOTING to ensure tracking)
            async with self._vms_lock:
                if len(self._vms) >= self.settings.max_concurrent_vms:
                    raise VmError(f"VM pool full: {len(self._vms)}/{self.settings.max_concurrent_vms} VMs active")
                self._vms[vm.vm_id] = vm

            # Transition to BOOTING state
            await vm.transition_state(VmState.BOOTING)

            try:
                await self._wait_for_guest(vm, timeout=constants.VM_BOOT_TIMEOUT_SECONDS)
                boot_complete_time = asyncio.get_event_loop().time()
                guest_wait_ms = round((boot_complete_time - guest_wait_start) * 1000)
                # Store timing on VM for scheduler to use
                vm.setup_ms = round((setup_complete_time - start_time) * 1000)
                vm.boot_ms = round((boot_complete_time - setup_complete_time) * 1000)
                # Granular setup timing
                vm.overlay_ms = overlay_ms
                # Granular boot timing
                vm.qemu_cmd_build_ms = qemu_cmd_build_ms
                vm.gvproxy_start_ms = gvproxy_start_ms
                vm.qemu_fork_ms = qemu_fork_ms
                vm.guest_wait_ms = guest_wait_ms
                # Transition to READY state after boot completes
                await vm.transition_state(VmState.READY)
            except TimeoutError as e:
                # Capture QEMU output for debugging
                stdout_text, stderr_text = await self._capture_qemu_output(qemu_proc)

                # Flush console log before reading to ensure all buffered content is written
                if console_log:
                    with contextlib.suppress(OSError):
                        console_log.flush()

                # Read console log (last N bytes for debugging)
                console_output = await read_log_tail(str(console_log_path), constants.CONSOLE_LOG_MAX_BYTES)

                # Build command string for debugging
                import shlex  # noqa: PLC0415

                qemu_cmd_str = " ".join(shlex.quote(arg) for arg in qemu_cmd)

                # Log output if available
                logger.error(
                    "Guest agent boot timeout",
                    extra={
                        "vm_id": vm_id,
                        "stderr": stderr_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stderr_text else "(empty)",
                        "stdout": stdout_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stdout_text else "(empty)",
                        "console_log": console_output,
                        "qemu_running": qemu_proc.returncode is None,
                        "qemu_returncode": qemu_proc.returncode,
                        "qemu_cmd": qemu_cmd_str[:1000],
                        "overlay_image": str(workdir.overlay_image),
                        "kernel_path": str(kernel_path),
                        "initramfs_path": str(initramfs_path),
                    },
                )

                await vm.destroy()

                # Get QEMU binary from command - handle ulimit wrapper case
                # When wrapped with bash -c, qemu_cmd is ["bash", "-c", "ulimit ... && exec qemu-system-..."]
                qemu_binary = "(unknown)"
                if qemu_cmd:
                    if qemu_cmd[0] == "bash" and len(qemu_cmd) > 2:  # noqa: PLR2004
                        # Extract actual QEMU binary from shell command string
                        shell_cmd_str = qemu_cmd[2]
                        qemu_match = _QEMU_BINARY_PATTERN.search(shell_cmd_str)
                        qemu_binary = qemu_match.group(1) if qemu_match else f"bash -c '{shell_cmd_str[:100]}...'"
                    else:
                        qemu_binary = qemu_cmd[0]

                raise VmError(
                    f"Guest agent not ready after {constants.VM_BOOT_TIMEOUT_SECONDS}s: {e}. "
                    f"qemu_binary={qemu_binary}, qemu_running={qemu_proc.returncode is None}, "
                    f"returncode={qemu_proc.returncode}, "
                    f"stderr: {stderr_text[:200] if stderr_text else '(empty)'}, "
                    f"console: {console_output[-constants.CONSOLE_LOG_PREVIEW_BYTES :] if console_output else '(empty)'}",
                    context={
                        "vm_id": vm_id,
                        "language": language,
                        "timeout_seconds": constants.VM_BOOT_TIMEOUT_SECONDS,
                        "console_log": console_output,
                        "qemu_running": qemu_proc.returncode is None,
                        "qemu_returncode": qemu_proc.returncode,
                        "qemu_cmd": qemu_cmd_str[:1000],
                        "kernel_path": str(kernel_path),
                        "initramfs_path": str(initramfs_path),
                        "overlay_image": str(workdir.overlay_image),
                    },
                ) from e

            # Log boot time with breakdown
            total_boot_ms = round((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.info(
                "VM created",
                extra={
                    "vm_id": vm_id,
                    "language": language,
                    "setup_ms": vm.setup_ms,
                    "boot_ms": vm.boot_ms,
                    "total_boot_ms": total_boot_ms,
                    # Granular setup breakdown
                    "overlay_ms": overlay_ms,
                    # Granular boot breakdown
                    "qemu_cmd_build_ms": qemu_cmd_build_ms,
                    "gvproxy_start_ms": gvproxy_start_ms,
                    "qemu_fork_ms": qemu_fork_ms,
                    "guest_wait_ms": guest_wait_ms,
                },
            )

            # Mark VM as successfully created to skip cleanup in finally
            vm_created = True
            return vm

        finally:
            # Comprehensive cleanup on failure (vm_created flag prevents cleanup on success)
            if not vm_created:
                logger.info(
                    "VM creation failed, cleaning up resources",
                    extra={
                        "vm_id": vm_id,
                        "qemu_started": qemu_proc is not None,
                        "gvproxy_started": gvproxy_proc is not None,
                    },
                )

                # Close console log file if opened (prevent resource leak)
                if console_log is not None:
                    with contextlib.suppress(OSError):
                        console_log.close()

                # Cancel log drain task if started
                if qemu_log_task is not None and not qemu_log_task.done():
                    qemu_log_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await qemu_log_task

                # Remove from registry if it was added (defensive - always try)
                async with self._vms_lock:
                    self._vms.pop(vm_id, None)

                await self._force_cleanup_all_resources(
                    vm_id=vm_id,
                    qemu_proc=qemu_proc,
                    gvproxy_proc=gvproxy_proc,
                    workdir=workdir,
                    cgroup_path=cgroup_path,
                )

    async def _start_gvproxy(
        self,
        vm_id: str,
        allowed_domains: list[str] | None,
        language: str,
        workdir: VmWorkingDirectory,
    ) -> tuple[ProcessWrapper, asyncio.Task[None]]:
        r"""Start gvproxy-wrapper with DNS filtering for this VM.

        Architecture Decision: gvisor-tap-vsock over alternatives
        ========================================================

        Chosen: gvisor-tap-vsock
        - ✅ Built-in DNS filtering via zones (regex-based)
        - ✅ Production-ready (Podman default since 2022)
        - ✅ 10MB memory overhead per VM
        - ✅ Simple JSON zone configuration
        - ✅ Zero CVEs (vs SLIRP: CVE-2021-3592/3/4/5, CVE-2020-29129/30)

        Socket Pre-binding (systemd activation pattern)
        ==============================================
        We create and bind the Unix socket in Python BEFORE spawning gvproxy,
        then pass the file descriptor to the child process. This eliminates
        the 100-300ms polling latency that was required when gvproxy created
        the socket itself.

        Args:
            vm_id: Unique VM identifier
            allowed_domains: Whitelist of allowed domains
            language: Programming language (for default registries)
            workdir: VM working directory containing socket paths

        Returns:
            Tuple of (gvproxy_process, gvproxy_log_task)

        Raises:
            VmError: Failed to start gvproxy-wrapper
        """
        socket_path = workdir.gvproxy_socket

        # Generate DNS zones JSON configuration
        dns_zones_json = generate_dns_zones_json(allowed_domains, language)

        logger.info(
            "Starting gvproxy-wrapper with DNS filtering",
            extra={
                "vm_id": vm_id,
                "allowed_domains": allowed_domains,
                "language": language,
                "dns_zones_json": dns_zones_json,
            },
        )

        # Pre-create and bind socket in parent process (systemd socket activation pattern)
        # This eliminates polling latency - socket is ready before gvproxy starts
        try:
            parent_sock = create_unix_socket(str(socket_path))
            socket_fd = parent_sock.fileno()
        except OSError as e:
            raise VmError(
                f"Failed to create gvproxy socket: {e}",
                context={
                    "vm_id": vm_id,
                    "language": language,
                    "socket_path": str(socket_path),
                },
            ) from e

        # Start gvproxy-wrapper with pre-bound FD
        from exec_sandbox.assets import get_gvproxy_path  # noqa: PLC0415

        gvproxy_binary = await get_gvproxy_path()
        if gvproxy_binary is None:
            parent_sock.close()
            raise VmError(
                "gvproxy-wrapper binary not found. "
                "Either enable auto_download_assets=True in SchedulerConfig, "
                "or run 'make build' to build it locally."
            )
        try:
            proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    str(gvproxy_binary),
                    "-listen-fd",
                    str(socket_fd),
                    "-dns-zones",
                    dns_zones_json,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,  # Create new process group for proper cleanup
                    pass_fds=(socket_fd,),  # Pass pre-bound socket FD to child
                )
            )
        except (OSError, FileNotFoundError) as e:
            parent_sock.close()
            raise VmError(
                f"Failed to start gvproxy-wrapper: {e}",
                context={
                    "vm_id": vm_id,
                    "language": language,
                    "allowed_domains": allowed_domains,
                    "binary_path": str(gvproxy_binary),
                },
            ) from e

        # Close parent's copy of FD (child has its own via pass_fds)
        parent_sock.close()

        # Wait for gvproxy to be ready (virtualnetwork.New() must complete before QEMU connects)
        # gvproxy prints "Listening on QEMU socket" after initialization is complete
        #
        # Design note: We use stdout event detection instead of polling or kqueue/inotify.
        # - Polling (asyncio.sleep loop): Would add 5-20ms latency between socket creation and detection
        # - kqueue (macOS) / inotify (Linux): Native but adds ~50 lines of platform-specific code
        # - Event-based (current): Instant notification via stdout, simple, cross-platform
        ready_event = asyncio.Event()

        def check_ready(line: str) -> None:
            logger.debug("[gvproxy-wrapper]", extra={"vm_id": vm_id, "output": line})
            if "Listening on QEMU socket" in line:
                ready_event.set()

        # Background task to drain gvproxy output (prevent pipe deadlock)
        gvproxy_log_task = asyncio.create_task(
            drain_subprocess_output(
                proc,
                process_name="gvproxy-wrapper",
                context_id=vm_id,
                stdout_handler=check_ready,
                stderr_handler=lambda line: logger.error(
                    f"[gvproxy-wrapper error] {line}", extra={"vm_id": vm_id, "output": line}
                ),
            )
        )
        gvproxy_log_task.add_done_callback(_log_task_exception)

        # Wait for gvproxy to signal readiness (timeout after 5 seconds)
        try:
            await asyncio.wait_for(ready_event.wait(), timeout=5.0)
        except TimeoutError:
            await proc.terminate()
            await proc.wait()
            raise VmError(
                "gvproxy-wrapper did not become ready in time",
                context={
                    "vm_id": vm_id,
                    "language": language,
                    "socket_path": str(socket_path),
                },
            ) from None

        # Grant qemu-vm user access to socket via ACL (more secure than chmod 666)
        # Only needed on Linux when qemu-vm user exists; skipped on macOS
        await grant_qemu_vm_access(socket_path)

        logger.info(
            "gvproxy-wrapper started successfully",
            extra={
                "vm_id": vm_id,
                "socket": str(socket_path),
                "dns_filtering": True,
            },
        )

        return proc, gvproxy_log_task

    async def _force_cleanup_all_resources(
        self,
        vm_id: str,
        qemu_proc: ProcessWrapper | None = None,
        gvproxy_proc: ProcessWrapper | None = None,
        workdir: VmWorkingDirectory | None = None,
        cgroup_path: Path | None = None,
    ) -> dict[str, bool]:
        """Comprehensive cleanup of ALL VM resources in reverse dependency order.

        This is the MAIN cleanup method used in finally blocks.

        Best practices:
        - Cleans in reverse dependency order (processes → workdir → cgroup)
        - NEVER raises exceptions (logs errors instead)
        - Safe to call multiple times (idempotent)
        - Handles None/already-cleaned resources
        - Returns status dict for monitoring/debugging

        Cleanup order (reverse dependencies):
        1. QEMU process (depends on: workdir files, cgroup, networking)
        2. gvproxy process (QEMU networking dependency)
        3. Working directory (contains overlay, sockets, logs - single rmtree)
        4. Cgroup directory (QEMU process was in it)

        Args:
            vm_id: VM identifier for logging
            qemu_proc: QEMU subprocess (can be None)
            gvproxy_proc: gvproxy subprocess (can be None)
            workdir: VM working directory containing all temp files (can be None)
            cgroup_path: cgroup directory path (can be None)

        Returns:
            Dictionary with cleanup status for each resource
        """
        logger.info("Starting comprehensive resource cleanup", extra={"vm_id": vm_id})
        results: dict[str, bool] = {}

        # Phase 1: Kill processes in parallel (independent operations)
        process_results = await asyncio.gather(
            cleanup_process(
                proc=qemu_proc,
                name="QEMU",
                context_id=vm_id,
                term_timeout=5.0,
                kill_timeout=2.0,
            ),
            cleanup_process(
                proc=gvproxy_proc,
                name="gvproxy",
                context_id=vm_id,
                term_timeout=3.0,
                kill_timeout=2.0,
            ),
            return_exceptions=True,
        )
        results["qemu"] = process_results[0] if isinstance(process_results[0], bool) else False
        results["gvproxy"] = process_results[1] if isinstance(process_results[1], bool) else False

        # Phase 2: Cleanup workdir and cgroup in parallel (after processes dead)
        # workdir.cleanup() removes overlay, sockets, and console log in one operation
        async def cleanup_workdir() -> bool:
            if workdir is None:
                return True
            return await workdir.cleanup()

        file_results = await asyncio.gather(
            cleanup_workdir(),
            cgroup.cleanup_cgroup(
                cgroup_path=cgroup_path,
                context_id=vm_id,
            ),
            return_exceptions=True,
        )
        results["workdir"] = file_results[0] if isinstance(file_results[0], bool) else False
        results["cgroup"] = file_results[1] if isinstance(file_results[1], bool) else False

        # Log summary
        success_count = sum(results.values())
        total_count = len(results)
        if success_count == total_count:
            logger.info("Cleanup completed successfully", extra={"vm_id": vm_id, "results": results})
        else:
            logger.warning(
                "Cleanup completed with errors",
                extra={
                    "vm_id": vm_id,
                    "results": results,
                    "success": success_count,
                    "total": total_count,
                },
            )

        return results

    async def destroy_vm(self, vm: QemuVM) -> None:
        """Destroy VM and clean up resources using defensive generic cleanup.

        This method uses the comprehensive cleanup orchestrator to ensure
        all resources are properly cleaned up even if some operations fail.

        Args:
            vm: QemuVM handle to destroy
        """
        try:
            # Close console log file before cancelling tasks
            if vm.console_log:
                with contextlib.suppress(OSError):
                    vm.console_log.close()

            # Cancel output reader tasks (prevent pipe deadlock during cleanup)
            if vm.qemu_log_task and not vm.qemu_log_task.done():
                vm.qemu_log_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await vm.qemu_log_task

            if vm.gvproxy_log_task and not vm.gvproxy_log_task.done():
                vm.gvproxy_log_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await vm.gvproxy_log_task

            # Destroy VM (transitions state, closes channel)
            await vm.destroy()

            # Comprehensive cleanup using defensive generic functions
            await self._force_cleanup_all_resources(
                vm_id=vm.vm_id,
                qemu_proc=vm.process,
                gvproxy_proc=vm.gvproxy_proc,
                workdir=vm.workdir,
                cgroup_path=vm.cgroup_path,
            )
        finally:
            # ALWAYS remove from registry, even on failure
            async with self._vms_lock:
                self._vms.pop(vm.vm_id, None)

    async def _capture_qemu_output(self, process: ProcessWrapper) -> tuple[str, str]:
        """Capture stdout/stderr from QEMU process.

        Args:
            process: QEMU subprocess

        Returns:
            Tuple of (stdout, stderr) as strings, empty if process still running
        """
        if process.returncode is not None:
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=1.0)
                return (stdout.decode() if stdout else "", stderr.decode() if stderr else "")
            except TimeoutError:
                pass
        return "", ""

    def get_base_image(self, language: str) -> Path:
        """Get base image path for language via auto-discovery.

        Auto-discovers images matching patterns:
        - python: python-*-base-*.qcow2
        - javascript: node-*-base-*.qcow2
        - raw: raw-base-*.qcow2

        Args:
            language: Programming language (python, javascript, or raw)

        Returns:
            Path to base qcow2 image

        Raises:
            VmError: Base image not found
        """
        # Pattern prefixes for each language
        patterns = {
            "python": "python-*-base-*.qcow2",
            "javascript": "node-*-base-*.qcow2",
            "raw": "raw-base-*.qcow2",
        }

        pattern = patterns.get(language)
        if not pattern:
            raise VmError(f"Unknown language: {language}")

        # Find matching images
        matches = list(self.settings.base_images_dir.glob(pattern))
        if not matches:
            raise VmError(
                f"Base image not found for language: {language}. "
                f"Pattern: {pattern}, dir: {self.settings.base_images_dir}"
            )

        # Return first match (sorted for determinism)
        return sorted(matches)[0]

    async def _detect_accel_type(self) -> AccelType:
        """Detect which QEMU accelerator to use.

        This is the single source of truth for virtualization mode detection.
        Used for both cgroup memory sizing (TCG needs more) and QEMU command building.

        Returns:
            AccelType.KVM if Linux KVM available
            AccelType.HVF if macOS HVF available
            AccelType.TCG if software emulation needed (or force_emulation=True)
        """
        if self.settings.force_emulation:
            return AccelType.TCG
        if await _check_kvm_available():
            return AccelType.KVM
        if detect_host_os() == HostOS.MACOS and await _check_hvf_available():
            return AccelType.HVF
        return AccelType.TCG

    async def _apply_overlay_permissions(self, base_image: Path, overlay_image: Path) -> bool:
        """Apply permissions to overlay (chown/chmod for qemu-vm isolation).

        Args:
            base_image: Base qcow2 image (needs read permission for qemu-vm)
            overlay_image: Overlay image (will be chowned to qemu-vm if possible)

        Returns:
            True if overlay was chowned to qemu-vm (QEMU should run as qemu-vm),
            False if overlay is owned by current user (QEMU should run as current user)
        """
        # Change ownership to qemu-vm user for process isolation (optional hardening)
        # Only if: Linux + qemu-vm user exists + can run sudo -u qemu-vm
        # The stronger probe_sudo_as_qemu_vm() ensures we can actually execute as qemu-vm
        # (probe_qemu_vm_user only checks if user exists, not sudo permissions)
        # Returns whether QEMU should run as qemu-vm user (based on chown success)
        if await probe_sudo_as_qemu_vm():
            # Make base image accessible to qemu-vm user
            # qemu-vm needs: read on file + execute on all parent directories
            # This is safe because the base image is read-only (writes go to overlay)

            # Make all parent directories traversable (a+x) up to /tmp or root
            current = base_image.parent
            dirs_to_chmod: list[Path] = []
            while current != current.parent and str(current) not in ("/", "/tmp"):  # noqa: S108
                dirs_to_chmod.append(current)
                current = current.parent

            await ensure_traversable(dirs_to_chmod)

            # Make base image readable
            if not await chmod_async(base_image, "a+r"):
                logger.debug("Could not chmod base image (qemu-vm may not have access)")

            if await chown_to_qemu_vm(overlay_image):
                # Make workdir accessible to qemu-vm for socket creation
                # mkdtemp creates with mode 0700, but qemu-vm needs access to create sockets
                workdir_path = overlay_image.parent
                if not await chmod_async(workdir_path, "a+rwx"):
                    logger.debug("Could not chmod workdir for qemu-vm access")
                return True  # Overlay chowned to qemu-vm, QEMU should run as qemu-vm
            logger.debug("Could not chown overlay to qemu-vm user (optional hardening)")
            return False  # Chown failed, QEMU should run as current user

        return False  # qemu-vm not available, QEMU should run as current user

    async def _build_linux_cmd(  # noqa: PLR0912, PLR0915
        self,
        language: str,  # noqa: ARG002
        vm_id: str,
        workdir: VmWorkingDirectory,
        memory_mb: int,
        allow_network: bool,
        enable_dns_filtering: bool = False,  # noqa: ARG002
    ) -> list[str]:
        """Build QEMU command for Linux (KVM + unshare + namespaces).

        Args:
            language: Programming language
            vm_id: Unique VM identifier
            workdir: VM working directory containing overlay and socket paths
            memory_mb: Guest VM memory in MB
            allow_network: Enable network access
            enable_dns_filtering: Enable DNS filtering via gvisor-tap-vsock

        Returns:
            QEMU command as list of strings
        """
        # Determine QEMU binary, machine type, and kernel based on architecture
        is_macos = detect_host_os() == HostOS.MACOS

        # Detect hardware acceleration type (centralized in _detect_accel_type)
        accel_type = await self._detect_accel_type()
        logger.info(
            "Hardware acceleration detection",
            extra={"vm_id": vm_id, "accel_type": accel_type.value, "is_macos": is_macos},
        )

        # Build accelerator string for QEMU
        if accel_type == AccelType.HVF:
            accel = "hvf"
        elif accel_type == AccelType.KVM:
            accel = "kvm"
        else:
            # TCG software emulation fallback (12x slower than KVM/HVF)
            #
            # thread=single: Disable MTTCG to reduce thread count per VM. Without this,
            # each VM creates multiple threads for parallel translation, exhausting
            # system thread limits when running parallel tests (qemu_thread_create:
            # Resource temporarily unavailable). Single-threaded TCG is slower but
            # prevents SIGABRT crashes on CI runners without KVM.
            # See: https://www.qemu.org/docs/master/devel/multi-thread-tcg.html
            #
            # tb-size: Translation block cache size in MB. QEMU 5.0+ defaults to 1GB
            # which causes OOM on CI runners with multiple VMs. Must match
            # cgroup.TCG_TB_CACHE_SIZE_MB for correct cgroup memory limits.
            # See cgroup.py for size rationale and benchmarks.
            accel = f"tcg,thread=single,tb-size={cgroup.TCG_TB_CACHE_SIZE_MB}"
            logger.warning(
                "Using TCG software emulation (slow) - KVM/HVF not available",
                extra={"vm_id": vm_id, "accel": accel},
            )

        # Track whether to use virtio-console (hvc0) or ISA serial (ttyS0)
        # Determined per-architecture below
        use_virtio_console = False

        if self.arch == HostArch.AARCH64:
            arch_suffix = "aarch64"
            qemu_bin = "qemu-system-aarch64"
            # highmem=off: Keep all RAM below 4GB for simpler memory mapping (faster boot)
            # gic-version=3: Explicit GIC version for TCG (ITS not modeled in TCG)
            # virtualization=off: Disable nested virt emulation (not needed, faster TCG)
            machine_type = (
                "virt,virtualization=off,highmem=off,gic-version=3,mem-merge=off"
                if is_macos
                else "virt,virtualization=off,highmem=off,gic-version=3,mem-merge=off,dump-guest-core=off"
            )
            # ARM64 always uses virtio-console (no ISA serial on virt machine)
            use_virtio_console = True
        else:
            arch_suffix = "x86_64"
            qemu_bin = "qemu-system-x86_64"
            # Machine type selection based on acceleration:
            # - microvm: Optimized for KVM/HVF, requires hardware virtualization
            # - q35: Standard machine type that works with TCG (software emulation)
            # microvm is designed specifically for hardware virtualization and doesn't work correctly with TCG
            # See: https://www.qemu.org/docs/master/system/i386/microvm.html
            #
            # CRITICAL: acpi=off forces qboot instead of SeaBIOS
            # With ACPI enabled (default), microvm uses SeaBIOS which has issues with direct kernel boot
            # on QEMU 8.2. With acpi=off, it uses qboot which is specifically designed for direct kernel boot.
            # See: https://www.kraxel.org/blog/2020/10/qemu-microvm-acpi/
            if accel_type == AccelType.KVM:
                # =============================================================
                # Console Device Timing: ISA Serial vs Virtio-Console
                # =============================================================
                # ISA serial (ttyS0) is available IMMEDIATELY at boot because:
                #   - It's a simple I/O port at 0x3F8 emulated by QEMU
                #   - No driver initialization required
                #   - Kernel can write to it from first instruction
                #
                # Virtio-console (hvc0) is available LATER (~30-50ms) because:
                #   - Requires virtio-mmio bus discovery during kernel init
                #   - Requires virtio-serial driver initialization
                #   - Not available during early boot
                #
                # If kernel uses console=hvc0 but hvc0 doesn't exist yet → HANG
                # See: https://gist.github.com/mcastelino/aa118275991d4f561ee22dc915b9345f
                #
                # =============================================================
                # TSC_DEADLINE Requirement for Non-Legacy Mode
                # =============================================================
                # pit=off, pic=off, and isa-serial=off require TSC_DEADLINE CPU feature
                # See: https://www.qemu.org/docs/master/system/i386/microvm.html
                #
                # In nested VMs (e.g., GitHub Actions on Azure/Hyper-V), TSC_DEADLINE
                # may not be exposed to the guest. Without it:
                #   - PIT/PIC disabled → no timer/interrupt source → kernel hang
                #   - ISA serial disabled → must use hvc0 → early boot hang
                #
                # =============================================================
                # Nested VM Fallback: microvm with Legacy Devices Enabled
                # =============================================================
                # When TSC_DEADLINE is unavailable (nested VMs on Azure/Hyper-V),
                # we keep microvm but enable ALL legacy devices:
                #
                # QEMU microvm legacy devices (enabled by default unless disabled):
                #   - i8259 PIC: Interrupt controller for legacy interrupt routing
                #   - i8254 PIT: Timer for scheduling and interrupt generation
                #   - MC146818 RTC: Real-time clock for timekeeping
                #   - ISA serial: Console output at ttyS0 (available at T=0)
                #
                # Why NOT fall back to 'pc' machine type:
                #   - microvm with virtio-mmio is simpler and faster to boot
                #   - Maintains consistent configuration between nested/bare-metal
                #   - virtio-mmio works fine in nested VMs when legacy devices present
                #   - 'pc' would require virtio-pci which needs different initramfs
                #
                # The key insight: without TSC_DEADLINE, kvmclock timing may be
                # unreliable in nested VMs. The PIT provides fallback timer source.
                #
                # See: https://www.qemu.org/docs/master/system/i386/microvm.html
                # =============================================================
                tsc_available = await _check_tsc_deadline()
                if tsc_available:
                    # Full optimization: TSC_DEADLINE available, use non-legacy mode
                    machine_type = "microvm,acpi=off,x-option-roms=off,pit=off,pic=off,rtc=off,isa-serial=off,mem-merge=off,dump-guest-core=off"
                    use_virtio_console = True
                else:
                    # Nested VM compatibility: use microvm with timer legacy devices
                    # Without TSC_DEADLINE, we need:
                    #   - PIT (i8254) for timer interrupts
                    #   - PIC (i8259) for interrupt handling
                    #   - RTC for timekeeping (kvmclock may not work in nested VMs)
                    # We disable ISA serial to avoid conflicts with virtio-serial.
                    # Console output goes via virtio-console (hvc0) instead of ttyS0.
                    # See: https://bugs.launchpad.net/qemu/+bug/1224444 (virtio-mmio issues)
                    logger.info(
                        "TSC_DEADLINE not available, using microvm with legacy timers but virtio-console for nested VM compatibility",
                        extra={"vm_id": vm_id},
                    )
                    machine_type = "microvm,acpi=off,x-option-roms=off,isa-serial=off,mem-merge=off,dump-guest-core=off"
                    use_virtio_console = True
            elif accel_type == AccelType.HVF:
                # macOS with HVF - configuration depends on architecture
                # Note: dump-guest-core=off not included - may not be supported on macOS QEMU
                if self.arch == HostArch.X86_64:
                    # Intel Mac: check TSC_DEADLINE availability
                    tsc_available = await _check_tsc_deadline()
                    if tsc_available:
                        # Full optimization: TSC_DEADLINE available, disable legacy devices
                        machine_type = (
                            "microvm,acpi=off,x-option-roms=off,pit=off,pic=off,rtc=off,isa-serial=off,mem-merge=off"
                        )
                    else:
                        # Conservative: keep legacy timers for older Intel Macs
                        logger.info(
                            "TSC_DEADLINE not available on Intel Mac, using microvm with legacy timers",
                            extra={"vm_id": vm_id},
                        )
                        machine_type = "microvm,acpi=off,x-option-roms=off,isa-serial=off,mem-merge=off"
                else:
                    # ARM64 Mac: no x86 legacy devices needed
                    # ARM uses different timer mechanism (CNTVCT_EL0), no TSC concept
                    machine_type = (
                        "microvm,acpi=off,x-option-roms=off,pit=off,pic=off,rtc=off,isa-serial=off,mem-merge=off"
                    )
                use_virtio_console = True
            else:
                # TCG emulation: use 'pc' (i440FX) which is simpler and more proven with direct kernel boot
                # q35 uses PCIe which can have issues with PCI device enumeration on some QEMU versions
                # See: https://wiki.qemu.org/Features/Q35
                machine_type = "pc,mem-merge=off,dump-guest-core=off"
                use_virtio_console = False
                logger.info(
                    "Using pc machine type (TCG emulation, hardware virtualization not available)",
                    extra={"vm_id": vm_id, "accel": accel},
                )

        # Auto-discover kernel and initramfs based on architecture
        # Note: existence validated in create_vm() before calling this method
        kernel_path = self.settings.kernel_path / f"vmlinuz-{arch_suffix}"
        initramfs_path = self.settings.kernel_path / f"initramfs-{arch_suffix}"

        # Layer 5: Linux namespaces (optional - requires capabilities or user namespaces)
        cmd: list[str] = []
        if detect_host_os() != HostOS.MACOS and await _probe_unshare_support():
            if allow_network:
                unshare_args = ["unshare", "--pid", "--mount", "--uts", "--ipc", "--fork"]
                cmd.extend([*unshare_args, "--"])
            else:
                unshare_args = ["unshare", "--pid", "--net", "--mount", "--uts", "--ipc", "--fork"]
                cmd.extend([*unshare_args, "--"])

        # Build QEMU command arguments
        # Determine if we're using microvm (requires -nodefaults to avoid BIOS fallback)
        is_microvm = "microvm" in machine_type

        # =============================================================
        # Virtio Transport Selection: MMIO vs PCI
        # =============================================================
        # Virtio devices can use two transport mechanisms:
        #
        # virtio-mmio (suffix: -device):
        #   - Memory-mapped I/O, no PCI bus required
        #   - Simpler, smaller footprint, faster boot (~13%)
        #   - Used by: microvm (x86 - both nested and bare-metal), virt (ARM64)
        #   - Works in nested VMs when legacy devices (PIT/PIC/RTC) are enabled
        #
        # virtio-pci (suffix: -pci):
        #   - Standard PCI bus with MSI-X interrupts
        #   - Used by: pc/q35 (x86 TCG emulation)
        #   - Requires different initramfs with virtio_pci.ko
        #
        # Selection criteria:
        #   microvm (x86)        → virtio-mmio (all KVM modes, nested or bare-metal)
        #   pc (x86 TCG)         → virtio-pci (software emulation fallback)
        #   virt (ARM64)         → virtio-mmio (initramfs loads virtio_mmio.ko)
        #
        # CRITICAL: ARM64 initramfs loads virtio_mmio.ko, NOT virtio_pci.ko
        # Using PCI devices on ARM64 causes boot hang (kernel can't find root device)
        # =============================================================
        virtio_suffix = "device" if (is_microvm or self.arch == HostArch.AARCH64) else "pci"

        qemu_args = [qemu_bin]

        # Set VM name for process identification (visible in ps aux, used by hwaccel test)
        # Format: guest=vm_id - the vm_id includes tenant, task, and uuid for uniqueness
        qemu_args.extend(["-name", f"guest={vm_id}"])

        # CRITICAL: -nodefaults -no-user-config are required for microvm to avoid BIOS fallback
        # See: https://www.qemu.org/docs/master/system/i386/microvm.html
        # For q35, we don't use these flags as the machine expects standard PC components
        if is_microvm:
            qemu_args.extend(["-nodefaults", "-no-user-config"])

        # Console selection based on machine type and architecture:
        # ┌─────────────────────────┬─────────────┬────────────────────────────────┐
        # │ Configuration           │ Console     │ Reason                         │
        # ├─────────────────────────┼─────────────┼────────────────────────────────┤
        # │ x86 microvm + TSC       │ hvc0        │ Non-legacy, virtio-console     │
        # │ x86 microvm - TSC       │ ttyS0       │ Legacy mode, ISA serial        │
        # │ x86 pc (TCG only)       │ ttyS0       │ Software emulation fallback    │
        # │ ARM64 virt              │ ttyAMA0     │ PL011 UART (always available)  │
        # └─────────────────────────┴─────────────┴────────────────────────────────┘
        # ttyS0 (ISA serial) is used when we need reliable early boot console (x86)
        # ttyAMA0 (PL011 UART) is used for ARM64 virt machine
        # hvc0 (virtio-console) is NOT reliable for kernel console on ARM64 because
        # it requires virtio-serial driver initialization (not available at early boot)
        # See: https://blog.memzero.de/toying-with-virtio/
        if self.arch == HostArch.AARCH64:
            # ARM64 virt machine has PL011 UART (ttyAMA0) - reliable at early boot
            # Note: hvc0 doesn't work for console because virtio-serial isn't ready
            # when kernel tries to open /dev/console, causing init to crash
            console_params = "console=ttyAMA0 loglevel=7"
        elif use_virtio_console:
            # x86 non-legacy mode: ISA serial disabled, use virtio-console
            console_params = "console=hvc0 loglevel=7"
        else:
            # x86 legacy mode or TCG: ISA serial available at T=0, reliable boot
            console_params = "console=ttyS0 loglevel=7"

        qemu_args.extend(
            [
                "-accel",
                accel,
                "-cpu",
                # For hardware accel use host CPU, for TCG use optimized emulated CPUs
                # ARM64 TCG: cortex-a57 is 3x faster than max (no pauth overhead)
                # x86 TCG: Haswell required for AVX2 (Python/Bun built for x86_64_v3)
                # See: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1033643
                # See: https://gitlab.com/qemu-project/qemu/-/issues/844
                (
                    "host"
                    if accel_type in (AccelType.HVF, AccelType.KVM)
                    else "cortex-a57"
                    if self.arch == HostArch.AARCH64
                    else "Haswell"
                ),
                "-M",
                machine_type,
                "-no-reboot",
                "-m",
                f"{memory_mb}M",
                "-smp",
                "1",
                "-kernel",
                str(kernel_path),
                "-initrd",
                str(initramfs_path),
                "-append",
                # Boot params: console varies by machine type, minimal kernel logging
                # =============================================================
                # Boot Parameter Optimizations (validated Jan 2025):
                # - nokaslr: Skip KASLR (safe for ephemeral isolated VMs)
                # - noresume: Skip hibernate resume check (VMs don't hibernate)
                # - swiotlb=noforce: Disable software I/O TLB (virtio uses direct DMA)
                # - panic=-1: Immediate reboot on panic (boot timeout handles loops)
                # - i8042.nokbd: Skip keyboard port check (no PS/2 in VM)
                # - tsc=reliable: Trust TSC clocksource (x86_64 only, kvmclock stable)
                # See: https://github.com/firecracker-microvm/firecracker
                # See: https://www.qemu.org/docs/master/system/i386/microvm.html
                # =============================================================
                f"{console_params} root=/dev/vda rootflags=rw,noatime rootfstype=ext4 rootwait=2 fsck.mode=skip reboot=t panic=-1 preempt=none i8042.noaux i8042.nomux i8042.nopnp i8042.nokbd init=/init random.trust_cpu=on raid=noautodetect mitigations=off nokaslr noresume swiotlb=noforce"
                # init.net=1: load network modules (only when allow_network=True)
                # init.balloon=1: load balloon module (always, needed for warm pool)
                + (" init.net=1" if allow_network else "")
                + " init.balloon=1"
                # tsc=reliable only for x86_64 (TSC is x86-specific, ARM uses CNTVCT_EL0)
                + (" tsc=reliable" if self.arch == HostArch.X86_64 else ""),
            ]
        )

        # Platform-specific memory configuration
        # Note: -mem-prealloc removed for faster boot (demand-paging is fine for ephemeral VMs)
        host_os = detect_host_os()

        # Layer 3: Seccomp sandbox - Linux only
        if detect_host_os() != HostOS.MACOS:
            qemu_args.extend(
                [
                    "-sandbox",
                    "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny",
                ]
            )

        # Determine AIO mode based on cached startup probe
        io_uring_available = await _probe_io_uring_support()
        aio_mode = "io_uring" if io_uring_available else "threads"
        if not io_uring_available:
            logger.debug(
                "Using aio=threads (io_uring not available)",
                extra={"reason": "syscall_probe_failed", "vm_id": vm_id},
            )

        # IOThread configuration
        match host_os:
            case HostOS.LINUX:
                use_iothread = True
            case HostOS.MACOS | HostOS.UNKNOWN:
                use_iothread = False

        iothread_id = f"iothread0-{vm_id}" if use_iothread else None
        if use_iothread:
            qemu_args.extend(["-object", f"iothread,id={iothread_id}"])

        # Disk configuration
        # Uses overlay backed by either:
        # - snapshot_path (cached L2 qcow2) for pre-installed packages
        # - base_image for cold boot
        qemu_args.extend(
            [
                "-drive",
                f"file={workdir.overlay_image},"
                f"format=qcow2,"
                f"if=none,"
                f"id=hd0,"
                f"cache=unsafe,"
                f"aio={aio_mode},"
                f"discard=unmap,"
                f"detect-zeroes=unmap,"
                f"werror=report,"
                f"rerror=report,"
                f"copy-on-read=off,"
                f"bps={constants.DISK_BPS_LIMIT},"
                f"bps_max={constants.DISK_BPS_BURST},"
                f"iops={constants.DISK_IOPS_LIMIT},"
                f"iops_max={constants.DISK_IOPS_BURST},"
                # Disable QEMU file locking to allow concurrent VMs sharing same backing file.
                # On Linux, QEMU uses OFD (Open File Descriptor) locks which cause "Failed to
                # get shared write lock" errors when multiple VMs access the same base image.
                # macOS doesn't enforce OFD locks, so this issue only manifests on Linux/CI.
                # Safe because: (1) each VM has unique overlay, (2) base image is read-only.
                f"file.locking=off",
            ]
        )

        # Platform-specific block device
        match host_os:
            case HostOS.MACOS:
                qemu_args.extend(
                    [
                        "-device",
                        f"virtio-blk-{virtio_suffix},drive=hd0,num-queues=1,queue-size=128",
                    ]
                )
            case HostOS.LINUX | HostOS.UNKNOWN:
                qemu_args.extend(
                    [
                        "-device",
                        # NOTE: Removed logical_block_size=4096,physical_block_size=4096
                        # Small ext4 filesystems (<512MB) use 1024-byte blocks by default, so forcing
                        # 4096-byte block size causes mount failures ("Invalid argument")
                        f"virtio-blk-{virtio_suffix},drive=hd0,iothread={iothread_id},num-queues=1,queue-size=128",
                    ]
                )

        # Display/console configuration
        # -nographic: headless mode
        # -monitor none: disable QEMU monitor (it uses stdio by default with -nographic,
        #   which conflicts with our -chardev stdio in environments without a proper TTY)
        qemu_args.extend(
            [
                "-nographic",
                "-monitor",
                "none",
            ]
        )

        # virtio-serial device for guest agent communication AND kernel console (hvc0)
        # With microvm + -nodefaults, we must explicitly configure:
        # 1. virtconsole for kernel console=hvc0 (required for boot output)
        # 2. virtserialport for guest agent cmd/event channels
        qemu_args.extend(
            [
                # Chardevs for communication channels
                # server=on: QEMU creates a listening Unix socket
                # wait=off: QEMU starts VM immediately without waiting for client connection
                # Note: Socket permissions (via umask) are set in _build_linux_cmd.
                # The guest agent retries connection so timing is handled.
                "-chardev",
                f"socket,id=cmd0,path={workdir.cmd_socket},server=on,wait=off",
                "-chardev",
                f"socket,id=event0,path={workdir.event_socket},server=on,wait=off",
                # Chardev for console output - connected to virtconsole (hvc0)
                "-chardev",
                "stdio,id=virtiocon0,mux=on,signal=off",
            ]
        )

        # Serial port configuration:
        # - virtio-console mode (hvc0): Disable serial to avoid stdio conflict
        # - ISA serial mode (ttyS0): Connect serial to chardev for console output
        if use_virtio_console:
            # Disable default serial to prevent "cannot use stdio by multiple character devices"
            # ARM64 virt has a default PL011 UART, x86 microvm has ISA serial
            qemu_args.extend(["-serial", "none"])
        else:
            # x86 legacy mode: connect ISA serial to chardev for ttyS0
            qemu_args.extend(["-serial", "chardev:virtiocon0"])

        # =============================================================
        # Virtio-Serial Device Configuration
        # =============================================================
        # Virtio-serial provides guest agent communication channels (cmd/event ports).
        # Console output handling depends on use_virtio_console flag:
        #
        # NON-LEGACY MODE (use_virtio_console=True):
        #   - virtconsole device created for hvc0 (kernel console)
        #   - 3 ports: virtconsole (nr=0) + cmd (nr=1) + event (nr=2)
        #   - ISA serial disabled via isa-serial=off in machine type
        #   - Requires TSC_DEADLINE for reliable boot timing
        #
        # LEGACY MODE (use_virtio_console=False):
        #   - Still uses microvm with virtio-mmio (for nested VMs)
        #   - Or uses 'pc' with virtio-pci (for TCG emulation only)
        #   - NO virtconsole device (would conflict with ISA serial chardev)
        #   - 3 ports but only 2 used: cmd (nr=1) + event (nr=2)
        #   - Port 0 reserved for virtconsole (QEMU backward compat requirement)
        #   - ISA serial enabled, connected to stdio chardev for ttyS0
        #   - Used when TSC_DEADLINE unavailable (nested VMs) or TCG emulation
        #
        # Why not always create virtconsole?
        #   - Both virtconsole and ISA serial would use same chardev (virtiocon0)
        #   - QEMU allows mux=on sharing, but causes output interleaving issues
        #   - Cleaner to use one console device exclusively
        #
        # See: https://bugs.launchpad.net/qemu/+bug/1639791 (early virtio console lost)
        # See: https://gist.github.com/mcastelino/aa118275991d4f561ee22dc915b9345f
        # =============================================================
        if use_virtio_console:
            qemu_args.extend(
                [
                    "-device",
                    f"virtio-serial-{virtio_suffix},max_ports=3",
                    # hvc0 console device - must be nr=0 to be hvc0
                    "-device",
                    "virtconsole,chardev=virtiocon0,nr=0",
                    "-device",
                    "virtserialport,chardev=cmd0,name=org.dualeai.cmd,nr=1",
                    "-device",
                    "virtserialport,chardev=event0,name=org.dualeai.event,nr=2",
                ]
            )
        else:
            # Legacy mode: no virtconsole, ISA serial handles console output
            # Port 0 is reserved for virtconsole (backward compat), so start at nr=1
            # See: QEMU error "Port number 0 on virtio-serial devices reserved for virtconsole"
            qemu_args.extend(
                [
                    "-device",
                    f"virtio-serial-{virtio_suffix},max_ports=3",
                    "-device",
                    "virtserialport,chardev=cmd0,name=org.dualeai.cmd,nr=1",
                    "-device",
                    "virtserialport,chardev=event0,name=org.dualeai.event,nr=2",
                ]
            )

        # virtio-balloon for host memory efficiency (deflate/inflate for warm pool)
        # - deflate-on-oom: guest returns memory under OOM pressure
        # - free-page-reporting: proactive free page hints to host (QEMU 5.1+/kernel 5.7+)
        qemu_args.extend(
            [
                "-device",
                f"virtio-balloon-{virtio_suffix},deflate-on-oom=on,free-page-reporting=on",
            ]
        )

        # virtio-net configuration (optional, internet access only)
        if allow_network:
            qemu_args.extend(
                [
                    "-netdev",
                    f"stream,id=net0,addr.type=unix,addr.path={workdir.gvproxy_socket}",
                    "-device",
                    f"virtio-net-{virtio_suffix},netdev=net0,mq=off,csum=off,gso=off,host_tso4=off,host_tso6=off,mrg_rxbuf=off,ctrl_rx=off,guest_announce=off",
                ]
            )

        # QMP (QEMU Monitor Protocol) socket for VM control operations
        qemu_args.extend(
            [
                "-qmp",
                f"unix:{workdir.qmp_socket},server=on,wait=off",
            ]
        )

        # Run QEMU as unprivileged user if qemu-vm user is available (optional hardening)
        # Falls back to current user if qemu-vm doesn't exist - VM still provides isolation
        if workdir.use_qemu_vm_user:
            # SECURITY: Avoid shell injection by not using 'sh -c'.
            # Instead, we use direct exec with preexec_fn to set umask.
            # stdbuf -oL forces line-buffered stdout to ensure console output is captured
            # immediately rather than being block-buffered (which happens with piped stdout).
            # IMPORTANT: stdbuf must come AFTER sudo - sudo sanitizes LD_PRELOAD for security.
            #
            # umask 007 is set via preexec_fn at subprocess creation time.
            # Creates chardev sockets with owner+group permissions (0660).
            # Host user must be in 'qemu-vm' group to connect to sockets owned by 'qemu-vm'.
            # More secure than 0666 (world-writable). Follows libvirt group membership pattern.
            cmd.extend(["sudo", "-u", "qemu-vm", "stdbuf", "-oL", *qemu_args])
            return cmd

        cmd.extend(qemu_args)

        return cmd

    async def _wait_for_guest(self, vm: QemuVM, timeout: float) -> None:  # noqa: PLR0915
        """Wait for guest agent using event-driven racing.

        Races QEMU process death monitor against guest readiness checks with retry logic.

        Args:
            vm: QemuVM handle
            timeout: Maximum wait time in seconds

        Raises:
            VmError: QEMU process died
            asyncio.TimeoutError: Guest not ready within timeout
        """

        async def monitor_process_death() -> None:
            """Monitor QEMU process death - kernel-notified, instant."""
            await vm.process.wait()

            # macOS HVF: Clean QEMU exit (code 0) is expected with -no-reboot
            host_os = detect_host_os()
            match host_os:
                case HostOS.MACOS if vm.process.returncode == 0:
                    logger.info(
                        "QEMU process exited cleanly (expected on macOS HVF with -no-reboot)",
                        extra={"vm_id": vm.vm_id, "exit_code": 0},
                    )
                    return
                case _:
                    pass

            # Process died - capture output
            stdout_text, stderr_text = await self._capture_qemu_output(vm.process)
            signal_name = ""
            if vm.process.returncode and vm.process.returncode < 0:
                sig = -vm.process.returncode
                signal_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else f"signal {sig}"

            # Read console log (last N bytes for debugging)
            console_output = await read_log_tail(str(vm.workdir.console_log), constants.CONSOLE_LOG_MAX_BYTES)

            logger.error(
                "QEMU process exited unexpectedly",
                extra={
                    "vm_id": vm.vm_id,
                    "exit_code": vm.process.returncode,
                    "signal": signal_name,
                    "stdout": stdout_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stdout_text else "(empty)",
                    "stderr": stderr_text[: constants.QEMU_OUTPUT_MAX_BYTES] if stderr_text else "(empty)",
                    "console_log": console_output,
                },
            )
            stderr_preview = stderr_text[:200] if stderr_text else "(empty)"
            console_preview = console_output[-constants.CONSOLE_LOG_PREVIEW_BYTES :] if console_output else "(empty)"
            raise VmError(
                f"QEMU process died (exit code {vm.process.returncode}, {signal_name}). "
                f"stderr: {stderr_preview}, console: {console_preview}"
            )

        async def check_guest_ready() -> None:
            """Single guest readiness check attempt."""
            await vm.channel.connect(timeout_seconds=5)
            response = await vm.channel.send_request(PingRequest(), timeout=5)

            # Ping returns PongMessage
            if not isinstance(response, PongMessage):
                raise RuntimeError(f"Guest ping returned unexpected type: {type(response)}")

            logger.info("Guest agent ready", extra={"vm_id": vm.vm_id, "version": response.version})

        # Race with retry logic (tenacity exponential backoff with full jitter)
        death_task: asyncio.Task[None] | None = None
        guest_task: asyncio.Task[None] | None = None
        try:
            async with asyncio.timeout(timeout):
                death_task = asyncio.create_task(monitor_process_death())

                # Pre-connect to chardev sockets to trigger QEMU's poll registration.
                # Without this, QEMU may not add sockets to its poll set until after
                # guest opens virtio-serial ports, causing reads to return EOF.
                # See: https://bugs.launchpad.net/qemu/+bug/1224444 (virtio-mmio socket race)
                #
                # Timeout is short (1s vs previous 2s) because sockets are usually not ready this early.
                # The retry loop below handles actual connection with proper exponential backoff.
                try:
                    await vm.channel.connect(timeout_seconds=1)
                    logger.debug("Pre-connected to guest channel sockets", extra={"vm_id": vm.vm_id})
                except (TimeoutError, OSError) as e:
                    # Expected - sockets may not be ready yet, retry loop will handle
                    logger.debug("Pre-connect to sockets deferred", extra={"vm_id": vm.vm_id, "reason": str(e)})

                # Retry with exponential backoff + full jitter
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(
                        (TimeoutError, OSError, json.JSONDecodeError, RuntimeError, asyncio.IncompleteReadError)
                    ),
                    # Reduced min from 0.1s to 0.01s for faster guest detection (agent ready in ~200-300ms)
                    wait=wait_random_exponential(multiplier=0.05, min=0.01, max=1.0),
                    before_sleep=before_sleep_log(logger, logging.DEBUG),
                ):
                    with attempt:
                        guest_task = asyncio.create_task(check_guest_ready())

                        # Race: first one wins
                        done, _pending = await asyncio.wait(
                            {death_task, guest_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        # Check which completed
                        if death_task in done:
                            # QEMU died - cancel guest and retrieve exception
                            guest_task.cancel()
                            # Suppress ALL exceptions - we're about to re-raise VmError from death_task.
                            # Race condition: guest_task may also have completed with an exception
                            # (e.g., IncompleteReadError) which we must suppress to avoid masking VmError.
                            # Use BaseException to also catch CancelledError (not a subclass of Exception in Python 3.8+).
                            with contextlib.suppress(BaseException):
                                await guest_task
                            await death_task  # Re-raise VmError

                        # Guest task completed - check result (raises if failed, triggering retry)
                        await guest_task

        except TimeoutError:
            # Flush console log before reading to ensure all buffered content is written
            if vm.console_log:
                with contextlib.suppress(OSError):
                    vm.console_log.flush()

            console_output = await read_log_tail(str(vm.workdir.console_log), constants.CONSOLE_LOG_MAX_BYTES)

            logger.error(
                "Guest agent timeout",
                extra={
                    "vm_id": vm.vm_id,
                    "timeout": timeout,
                    "qemu_running": vm.process.returncode is None,
                    "console_output": console_output,
                    "overlay_image": str(vm.overlay_image) if vm.overlay_image else "(none)",
                },
            )

            raise TimeoutError(f"Guest agent not ready after {timeout}s") from None

        finally:
            # Always clean up tasks to prevent "Task exception was never retrieved" warnings.
            # This handles all exit paths: success, TimeoutError, VmError, and any other exception.
            # Use BaseException to catch CancelledError (which is not a subclass of Exception in Python 3.8+).
            for task in (death_task, guest_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (death_task, guest_task):
                if task is not None:
                    with contextlib.suppress(BaseException):
                        await task

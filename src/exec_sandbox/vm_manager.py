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
import ctypes
import errno
import fcntl
import hashlib
import json
import logging
import os
import platform
import signal
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TextIO
from uuid import uuid4

import aiofiles
import aiofiles.os
from tenacity import AsyncRetrying, before_sleep_log, retry_if_exception_type, wait_random_exponential

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.dns_filter import generate_dns_zones_json
from exec_sandbox.exceptions import VmError, VmTimeoutError
from exec_sandbox.guest_agent_protocol import (
    ExecuteCodeRequest,
    PingRequest,
    PongMessage,
)
from exec_sandbox.guest_channel import DualPortChannel, GuestChannel
from exec_sandbox.models import ExecutionResult, Language
from exec_sandbox.platform_utils import HostArch, HostOS, ProcessWrapper, detect_host_arch, detect_host_os
from exec_sandbox.resource_cleanup import (
    cleanup_cgroup,
    cleanup_file,
    cleanup_overlay,
    cleanup_process,
)
from exec_sandbox.settings import Settings
from exec_sandbox.subprocess_utils import drain_subprocess_output, read_log_tail

logger = get_logger(__name__)

# KVM ioctl constants for probing
# See: linux/kvm.h - these are stable ABI
_KVM_GET_API_VERSION = 0xAE00
_KVM_API_VERSION_EXPECTED = 12  # Stable since Linux 2.6.38


def _get_gvproxy_wrapper_path() -> Path:
    """Get path to gvproxy-wrapper binary for current platform.

    Detection order:
    1. Cached download directory (auto-downloaded from GitHub Releases)
    2. Repo-relative path (development mode)

    Returns:
        Path to the platform-specific gvproxy-wrapper binary

    Raises:
        VmError: Binary not found for current platform
    """
    # Detect OS
    host_os = detect_host_os()
    if host_os == HostOS.MACOS:
        os_name = "darwin"
    elif host_os == HostOS.LINUX:
        os_name = "linux"
    else:
        raise VmError("Unsupported OS for gvproxy-wrapper")

    # Detect architecture
    host_arch = detect_host_arch()
    match host_arch:
        case HostArch.AARCH64:
            arch = "arm64"
        case HostArch.X86_64:
            arch = "amd64"
        case _:
            raise VmError(f"Unsupported architecture for gvproxy-wrapper: {host_arch}")

    binary_name = f"gvproxy-wrapper-{os_name}-{arch}"

    # 1. Check cached download directory first
    from exec_sandbox.assets import get_cached_asset_path  # noqa: PLC0415

    if cached_path := get_cached_asset_path(binary_name):
        return cached_path

    # 2. Fall back to repo-relative path (dev mode)
    repo_root = Path(__file__).parent.parent.parent
    binary_path = repo_root / "gvproxy-wrapper" / "bin" / binary_name

    if not binary_path.exists():
        raise VmError(
            "gvproxy-wrapper binary not found. "
            "Either enable auto_download_assets=True in SchedulerConfig, "
            "or run 'make build' to build it locally."
        )

    return binary_path


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
    """

    __slots__ = (
        "hvf",
        "io_uring",
        "kvm",
        "qemu_vm_user",
        "tsc_deadline",
        "unshare",
    )

    def __init__(self) -> None:
        self.hvf: bool | None = None
        self.io_uring: bool | None = None
        self.kvm: bool | None = None
        self.qemu_vm_user: bool | None = None
        self.tsc_deadline: bool | None = None
        self.unshare: bool | None = None


# Module-level cache instance
_probe_cache = _ProbeCache()


def _check_kvm_available() -> bool:
    """Check if KVM acceleration is available and accessible (cached).

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
        True if /dev/kvm exists and is accessible (enables KVM mode)
        False otherwise (falls back to TCG software emulation)
    """
    # Fast path: return cached result
    if _probe_cache.kvm is not None:
        return _probe_cache.kvm

    kvm_path = Path("/dev/kvm")
    if not kvm_path.exists():
        logger.debug("KVM not available: /dev/kvm does not exist")
        _probe_cache.kvm = False
        return False

    # Check if we can actually access /dev/kvm (not just that it exists)
    # This catches permission issues that would cause QEMU to fail or hang
    # See: https://github.com/actions/runner-images/issues/8542
    try:
        # os.access with R_OK|W_OK checks actual permission for the current user
        if not os.access(kvm_path, os.R_OK | os.W_OK):
            logger.debug("KVM not available: permission denied on /dev/kvm")
            _probe_cache.kvm = False
            return False

        # Actually try to open /dev/kvm to verify it works
        # Some environments (nested VMs, containers) have /dev/kvm but it doesn't work
        with kvm_path.open("rb") as f:
            try:
                api_version = fcntl.ioctl(f.fileno(), _KVM_GET_API_VERSION)
                _probe_cache.kvm = api_version == _KVM_API_VERSION_EXPECTED
                if _probe_cache.kvm:
                    logger.debug("KVM available and working", extra={"api_version": api_version})
                else:
                    logger.warning(
                        "KVM available but unexpected API version",
                        extra={"api_version": api_version, "expected": _KVM_API_VERSION_EXPECTED},
                    )
            except OSError as e:
                # ioctl failed - KVM not actually usable
                logger.warning(
                    "KVM device accessible but ioctl failed",
                    extra={"error": str(e)},
                )
                _probe_cache.kvm = False
    except OSError as e:
        logger.debug("KVM not available: failed to open /dev/kvm", extra={"error": str(e)})
        _probe_cache.kvm = False

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
    # Fast path: return cached result
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
        _probe_cache.hvf = proc.returncode == 0 and stdout.decode().strip() == "1"
    except (OSError, TimeoutError):
        _probe_cache.hvf = False

    return _probe_cache.hvf


async def _check_tsc_deadline_available() -> bool:
    """Check if TSC_DEADLINE CPU feature is available (cached).

    TSC_DEADLINE is required to disable PIT (i8254) and PIC (i8259) in microvm.
    Without TSC_DEADLINE, the APIC timer cannot use deadline mode, and the system
    needs the legacy PIT for timer interrupts.

    In nested virtualization (e.g., GitHub Actions runners), TSC_DEADLINE may not
    be exposed to the guest, causing boot hangs if PIT/PIC are disabled.

    See: https://www.qemu.org/docs/master/system/i386/microvm.html
    "if your host's CPUs have the TSC_DEADLINE feature, you can also disable
    both the i8259 PIC and the i8254 PIT"

    Returns:
        True if TSC_DEADLINE is available, False otherwise
    """
    # Fast path: return cached result
    if _probe_cache.tsc_deadline is not None:
        return _probe_cache.tsc_deadline

    # TSC_DEADLINE is x86-only
    if detect_host_arch() != HostArch.X86_64:
        _probe_cache.tsc_deadline = False
        return False

    # Check /proc/cpuinfo for tsc_deadline_timer flag
    cpuinfo_path = "/proc/cpuinfo"
    if not await aiofiles.os.path.exists(cpuinfo_path):
        # Not on Linux or /proc not mounted
        _probe_cache.tsc_deadline = False
        return False

    try:
        async with aiofiles.open(cpuinfo_path) as f:
            cpuinfo = await f.read()
        # Look for tsc_deadline_timer in the flags line
        # Format: "flags : fpu vme ... tsc_deadline_timer ..."
        for line in cpuinfo.split("\n"):
            if line.startswith("flags"):
                has_tsc_deadline = "tsc_deadline_timer" in line.split()
                _probe_cache.tsc_deadline = has_tsc_deadline
                if has_tsc_deadline:
                    logger.debug("TSC_DEADLINE available (can disable PIT/PIC)")
                else:
                    logger.debug("TSC_DEADLINE not available (keeping PIT/PIC enabled)")
                return has_tsc_deadline
    except OSError as e:
        logger.warning("Failed to read /proc/cpuinfo for TSC_DEADLINE check", extra={"error": str(e)})

    _probe_cache.tsc_deadline = False
    return False


async def _probe_io_uring_support() -> bool:
    """Probe for io_uring support using syscall test (cached).

    Returns:
        True if io_uring fully available, False otherwise
    """
    # Fast path: return cached result
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

    # Check 2: Syscall probe (most reliable)
    try:
        libc = ctypes.CDLL(None, use_errno=True)

        # __NR_io_uring_setup syscall number: 425
        nr_io_uring_setup = 425

        # Call io_uring_setup(0, NULL) - should fail with EINVAL if supported
        result = libc.syscall(nr_io_uring_setup, 0, None)

        if result == -1:
            err = ctypes.get_errno()
            if err == errno.ENOSYS:
                logger.info(
                    "io_uring syscall not available (ENOSYS)",
                    extra={"kernel": platform.release()},
                )
                _probe_cache.io_uring = False
                return False
            if err in (errno.EINVAL, errno.EFAULT):
                # Both indicate io_uring syscall is recognized and available:
                # - EINVAL: kernel validated params and rejected entries=0
                # - EFAULT: kernel tried to access params pointer (NULL)
                logger.info(
                    "io_uring syscall available",
                    extra={"kernel": platform.release(), "probe_errno": err},
                )
                _probe_cache.io_uring = True
                return True
            if err == errno.EPERM:
                logger.warning(
                    "io_uring blocked by seccomp/container policy",
                    extra={"errno": err},
                )
                _probe_cache.io_uring = False
                return False
            logger.warning(
                "io_uring probe failed with unexpected error",
                extra={"errno": err, "error": os.strerror(err)},
            )
            _probe_cache.io_uring = False
            return False

        _probe_cache.io_uring = True
        return True

    except (OSError, ValueError, RuntimeError) as e:
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
    # Fast path: return cached result
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


async def _probe_qemu_vm_user() -> bool:
    """Check if qemu-vm user exists for process isolation (cached).

    Returns:
        True if qemu-vm user exists, False otherwise
    """
    # Fast path: return cached result
    if _probe_cache.qemu_vm_user is not None:
        return _probe_cache.qemu_vm_user

    # Skip on macOS - not applicable
    if detect_host_os() == HostOS.MACOS:
        _probe_cache.qemu_vm_user = False
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "/usr/bin/id",
            "qemu-vm",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        _probe_cache.qemu_vm_user = proc.returncode == 0
        if _probe_cache.qemu_vm_user:
            logger.info("qemu-vm user available (process isolation enabled)")
        else:
            logger.debug("qemu-vm user not found (process isolation disabled)")
    except (OSError, TimeoutError) as e:
        logger.debug(f"qemu-vm user probe failed: {e}")
        _probe_cache.qemu_vm_user = False

    return _probe_cache.qemu_vm_user


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
            result = await vm.execute(code="print('hello')", language="python", timeout_seconds=30)
            # VM automatically destroyed on exit, even if exception occurs
        ```

        Manual cleanup still available via destroy() method for explicit control.

    Attributes:
        vm_id: Unique VM identifier format: {tenant_id}-{task_id}-{uuid4}
        process: QEMU subprocess handle
        cgroup_path: cgroup v2 path for resource limits
        overlay_image: Ephemeral qcow2 overlay (deleted on destroy)
        gvproxy_proc: Optional gvproxy-wrapper process for DNS filtering
        gvproxy_socket: Optional QEMU stream socket path
        gvproxy_log_task: Optional background task draining gvproxy stdout/stderr
    """

    def __init__(
        self,
        vm_id: str,
        process: ProcessWrapper,
        cgroup_path: Path,
        overlay_image: Path,
        channel: GuestChannel,
        language: str,
        gvproxy_proc: ProcessWrapper | None = None,
        gvproxy_socket: Path | None = None,
        qmp_socket_path: Path | None = None,
        qemu_log_task: asyncio.Task[None] | None = None,
        gvproxy_log_task: asyncio.Task[None] | None = None,
        console_log: TextIO | None = None,
        use_qemu_vm_user: bool = False,
    ):
        """Initialize VM handle.

        Args:
            vm_id: Unique VM identifier (scoped by tenant_id)
            process: Running QEMU subprocess (ProcessWrapper for PID-reuse safety)
            cgroup_path: cgroup v2 path for cleanup
            overlay_image: Ephemeral qcow2 overlay for cleanup
            channel: Communication channel for TCP guest agent
            language: Programming language for this VM
            gvproxy_proc: Optional gvproxy-wrapper process (ProcessWrapper)
            gvproxy_socket: Optional QEMU stream socket path
            qmp_socket_path: Optional QMP (QEMU Monitor Protocol) socket path for memory snapshots
            qemu_log_task: Background task draining QEMU stdout/stderr (prevents pipe deadlock)
            gvproxy_log_task: Background task draining gvproxy stdout/stderr (prevents pipe deadlock)
            console_log: Optional file handle for QEMU console log
            use_qemu_vm_user: Whether QEMU ran as qemu-vm user (requires sudo rm for cleanup)
        """
        self.vm_id = vm_id
        self.process = process
        self.cgroup_path = cgroup_path
        self.overlay_image = overlay_image
        self.channel = channel
        self.language = language
        self.gvproxy_proc = gvproxy_proc
        self.gvproxy_socket = gvproxy_socket
        self.qmp_socket_path = qmp_socket_path
        self.qemu_log_task = qemu_log_task
        self.gvproxy_log_task = gvproxy_log_task
        self.console_log: TextIO | None = console_log
        self.use_qemu_vm_user = use_qemu_vm_user
        self._destroyed = False
        self._state = VmState.CREATING
        self._state_lock = asyncio.Lock()
        self._creation_time = asyncio.get_event_loop().time()
        # Timing instrumentation (set by VmManager.create_vm)
        self._setup_ms: int | None = None  # Resource setup time
        self._boot_ms: int | None = None  # VM boot time (kernel + guest-agent)

    @property
    def setup_ms(self) -> int | None:
        """Get resource setup time in milliseconds."""
        return self._setup_ms

    @setup_ms.setter
    def setup_ms(self, value: int) -> None:
        """Set resource setup time in milliseconds."""
        self._setup_ms = value

    @property
    def boot_ms(self) -> int | None:
        """Get VM boot time in milliseconds."""
        return self._boot_ms

    @boot_ms.setter
    def boot_ms(self, value: int) -> None:
        """Set VM boot time in milliseconds."""
        self._boot_ms = value

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
        language: Language,
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
            language: Programming language (python or javascript)
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
                        "language": language,
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
            language=language,
            code=code,
            timeout=timeout_seconds,
            env_vars=env_vars or {},
        )

        try:
            # Connect to guest via TCP
            # Fixed init timeout (connection establishment, independent of execution timeout)
            await self.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)

            # Stream execution output to console
            # Hard timeout = soft timeout (guest enforcement) + margin (host watchdog)
            hard_timeout = timeout_seconds + constants.EXECUTION_TIMEOUT_MARGIN_SECONDS

            # Stream messages and collect output
            exit_code = -1
            execution_time_ms = None
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
                    # Execution complete
                    exit_code = msg.exit_code
                    execution_time_ms = msg.execution_time_ms
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
            exec_result = ExecutionResult(
                stdout=stdout_truncated,  # Return to user
                stderr=stderr_truncated,  # Return to user
                exit_code=exit_code,
                execution_time_ms=execution_time_ms,  # Guest-reported
                external_cpu_time_ms=external_cpu_ms or None,  # Host-measured
                external_memory_peak_mb=external_mem_mb or None,  # Host-measured
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
                extra={"vm_id": self.vm_id, "language": language},
            )
            # Re-raise to propagate cancellation
            raise
        except TimeoutError as e:
            raise VmTimeoutError(
                f"VM {self.vm_id} execution exceeded {timeout_seconds}s timeout",
                context={
                    "vm_id": self.vm_id,
                    "timeout_seconds": timeout_seconds,
                    "language": language,
                },
            ) from e
        except (OSError, json.JSONDecodeError) as e:
            raise VmError(
                f"VM {self.vm_id} communication failed: {e}",
                context={
                    "vm_id": self.vm_id,
                    "language": language,
                    "error_type": type(e).__name__,
                },
            ) from e

    async def _read_cgroup_stats(self) -> tuple[int | None, int | None]:
        """Read external CPU time and peak memory from cgroup v2.

        Returns:
            Tuple of (cpu_time_ms, peak_memory_mb)
            Returns (None, None) if cgroup not available or read fails
        """
        if not self.cgroup_path or not await aiofiles.os.path.exists(self.cgroup_path):
            return (None, None)

        cpu_time_ms: int | None = None
        peak_memory_mb: int | None = None

        try:
            # Read cpu.stat for usage_usec (microseconds)
            cpu_stat_file = self.cgroup_path / "cpu.stat"
            if await aiofiles.os.path.exists(cpu_stat_file):
                async with aiofiles.open(cpu_stat_file) as f:
                    cpu_stat = await f.read()
                for line in cpu_stat.splitlines():
                    if line.startswith("usage_usec"):
                        usage_usec = int(line.split()[1])
                        cpu_time_ms = usage_usec // 1000  # Convert to milliseconds
                        break

            # Read memory.peak for peak memory usage (bytes)
            memory_peak_file = self.cgroup_path / "memory.peak"
            if await aiofiles.os.path.exists(memory_peak_file):
                async with aiofiles.open(memory_peak_file) as f:
                    peak_bytes = int((await f.read()).strip())
                peak_memory_mb = peak_bytes // (1024 * 1024)  # Convert to MB

        except (OSError, ValueError) as e:
            logger.debug(
                f"Failed to read cgroup stats: {e}",
                extra={"vm_id": self.vm_id, "cgroup_path": str(self.cgroup_path)},
            )

        return (cpu_time_ms, peak_memory_mb)

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

        # Step 3-4: Parallel cleanup (cgroup + overlay)
        # After QEMU terminates, cleanup tasks are independent
        async def cleanup_cgroup_fn():
            """Remove cgroup asynchronously."""
            if self.cgroup_path.exists():
                try:
                    # Move all processes to parent cgroup first
                    parent_cgroup = self.cgroup_path.parent / "cgroup.procs"
                    if parent_cgroup.exists():
                        async with aiofiles.open(self.cgroup_path / "cgroup.procs") as f:
                            pids = (await f.read()).strip().split("\n")
                        for pid in pids:
                            if pid:
                                try:
                                    async with aiofiles.open(parent_cgroup, "w") as f:
                                        await f.write(pid)
                                except (OSError, PermissionError):
                                    pass

                    # Remove cgroup directory
                    await aiofiles.os.rmdir(self.cgroup_path)
                except (OSError, PermissionError):
                    pass

        async def cleanup_overlay_fn():
            """Delete overlay image asynchronously."""
            if self.overlay_image.exists():
                with contextlib.suppress(OSError, PermissionError):
                    if self.use_qemu_vm_user:
                        # Overlay was chowned to qemu-vm, need sudo to delete
                        proc = await asyncio.create_subprocess_exec(
                            "sudo",
                            "rm",
                            "-f",
                            str(self.overlay_image),
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        await proc.wait()
                    else:
                        await aiofiles.os.remove(self.overlay_image)

        # Run cleanup tasks in parallel
        await asyncio.gather(
            cleanup_cgroup_fn(),
            cleanup_overlay_fn(),
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
        manager = VmManager(settings)
        await manager.initialize()  # Run async system probes
        vm = await manager.create_vm("python", "tenant-123", "task-456")
        result = await vm.execute("print('hello')", "python", timeout_seconds=30)
        await manager.destroy_vm(vm)
    """

    def __init__(self, settings: Settings):
        """Initialize QEMU manager (sync part only).

        Args:
            settings: Service configuration (paths, limits, etc.)

        Note: Call `await initialize()` after construction to run async system probes.

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

    async def initialize(self) -> None:
        """Run async system probes and complete initialization.

        This method runs all async system capability probes (HVF, unshare, qemu-vm user)
        and logs the initialization status. Must be called before creating VMs.

        The probes are cached at module level, so subsequent calls are fast no-ops.
        """
        if self._initialized:
            return

        # Run all async probes concurrently (they cache their results at module level)
        io_uring_available, unshare_available, qemu_vm_user_available = await asyncio.gather(
            _probe_io_uring_support(),
            _probe_unshare_support(),
            _probe_qemu_vm_user(),
        )

        # Pre-flight check: validate kernel and initramfs exist (cached)
        await _validate_kernel_initramfs(self.settings.kernel_path, self.arch)

        self._initialized = True

        # Log registry initialization (empty on startup, even after crash)
        logger.info(
            "VM registry initialized",
            extra={
                "max_concurrent_vms": self.settings.max_concurrent_vms,
                "io_uring_available": io_uring_available,
                "unshare_available": unshare_available,
                "qemu_vm_user_available": qemu_vm_user_available,
                "note": "Any zombie QEMU processes from crashes will timeout naturally (max 2min)",
            },
        )

    def get_active_vms(self) -> dict[str, QemuVM]:
        """Get snapshot of active VMs (for debugging/metrics).

        Returns:
            Copy of VM registry (vm_id → QemuVM)
        """
        return dict(self._vms)

    async def create_vm(  # noqa: PLR0912, PLR0915
        self,
        language: str,
        tenant_id: str,
        task_id: str,
        snapshot_path: Path | None = None,
        memory_mb: int = constants.DEFAULT_MEMORY_MB,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
    ) -> QemuVM:
        """Create and boot QEMU microVM from snapshot.

        Workflow:
        1. Generate unique VM ID and CID
        2. Create ephemeral qcow2 overlay from snapshot
        3. Set up cgroup v2 resource limits
        4. Build QEMU command (platform-specific)
        5. Launch QEMU subprocess
        6. Wait for guest agent ready

        Args:
            language: Programming language (python or javascript)
            tenant_id: Tenant identifier for isolation
            task_id: Task identifier
            snapshot_path: Optional snapshot base image (default: use base image)
            memory_mb: Memory limit in MB (128-2048, default 512)
            allow_network: Enable network access (default: False, isolated)
            allowed_domains: Whitelist of allowed domains if allow_network=True

        Returns:
            QemuVM handle for code execution

        Raises:
            VmError: VM creation failed
            asyncio.TimeoutError: VM boot timeout (>5s)
        """
        # Start timing
        start_time = asyncio.get_event_loop().time()

        # Step 0: Validate kernel and initramfs exist (cached, one-time check)
        await _validate_kernel_initramfs(self.settings.kernel_path, self.arch)
        arch_suffix = "aarch64" if self.arch == HostArch.AARCH64 else "x86_64"
        kernel_path = self.settings.kernel_path / f"vmlinuz-{arch_suffix}"
        initramfs_path = self.settings.kernel_path / f"initramfs-{arch_suffix}"

        # Step 1: Generate VM identifiers
        vm_id = f"{tenant_id}-{task_id}-{uuid4()}"

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

        # Step 2-4: Parallel resource setup (overlay + cgroup + gvproxy)
        # These operations are independent and can run concurrently
        overlay_image = Path(f"/tmp/qemu-{vm_id}.qcow2")  # noqa: S108
        base_image = snapshot_path or self.get_base_image(language)

        # Initialize ALL tracking variables before try block for finally cleanup
        cgroup_path: Path | None = None
        gvproxy_proc: ProcessWrapper | None = None
        gvproxy_socket: Path | None = None
        gvproxy_log_task: asyncio.Task[None] | None = None
        qemu_proc: ProcessWrapper | None = None
        use_qemu_vm_user = False  # Set by _create_overlay if qemu-vm user available
        vm_created = False  # Flag to skip cleanup if VM successfully created

        # IMPORTANT: Always use gvproxy for network-enabled VMs
        # SLIRP user networking has reliability issues with containerized unprivileged execution
        enable_dns_filtering = allow_network  # Force gvproxy for all network-enabled VMs

        try:
            # Run independent setup tasks in parallel
            # Add gvproxy startup to parallel tasks if network enabled
            logger.debug(
                "Checking gvproxy condition",
                extra={
                    "debug_category": "network",
                    "vm_id": vm_id,
                    "allow_network": allow_network,
                    "allowed_domains": allowed_domains,
                    "domains_count": len(allowed_domains) if allowed_domains else 0,
                },
            )

            if allow_network:
                # Always use gvproxy for network-enabled VMs
                # Pass allowed_domains to gvproxy - None means use language defaults
                logger.info(
                    "Adding gvproxy-wrapper to parallel tasks",
                    extra={"vm_id": vm_id, "allowed_domains": allowed_domains},
                )
                # Run all tasks in parallel including gvproxy
                overlay_result, cgroup_result, gvproxy_result = await asyncio.gather(
                    self._create_overlay(base_image, overlay_image),
                    self._setup_cgroup(vm_id, tenant_id),
                    self._start_gvproxy(vm_id, allowed_domains, language),
                    return_exceptions=True,
                )
                setup_complete_time = asyncio.get_event_loop().time()

                # Check for failures
                if isinstance(overlay_result, BaseException):
                    raise overlay_result
                if isinstance(cgroup_result, BaseException):
                    raise cgroup_result
                if isinstance(gvproxy_result, BaseException):
                    raise gvproxy_result

                # Type narrowing: after BaseException checks
                cgroup_path = cgroup_result
                use_qemu_vm_user = overlay_result  # bool from _create_overlay

                # Unpack tuple from _start_gvproxy
                gvproxy_proc, gvproxy_socket, gvproxy_log_task = gvproxy_result
            else:
                # Run overlay and cgroup setup in parallel (no gvproxy)
                overlay_result, cgroup_result = await asyncio.gather(
                    self._create_overlay(base_image, overlay_image),
                    self._setup_cgroup(vm_id, tenant_id),
                    return_exceptions=True,
                )
                setup_complete_time = asyncio.get_event_loop().time()

                # Check for failures
                if isinstance(overlay_result, BaseException):
                    raise overlay_result
                if isinstance(cgroup_result, BaseException):
                    raise cgroup_result

                # Type narrowing: after BaseException checks
                cgroup_path = cgroup_result
                use_qemu_vm_user = overlay_result  # bool from _create_overlay

            # Step 5: Build QEMU command (always Linux in container)
            qemu_cmd = await self._build_linux_cmd(
                language,
                vm_id,
                overlay_image,
                memory_mb,
                allow_network,
                enable_dns_filtering,
                gvproxy_socket,
                use_qemu_vm_user=use_qemu_vm_user,
            )

            # Step 6: Create dual-port Unix socket communication channel for guest agent
            cmd_socket = self._get_cmd_socket_path(vm_id)
            event_socket = self._get_event_socket_path(vm_id)
            qmp_socket_path = self._get_qmp_socket_path(vm_id)

            # Clean up any stale socket files before QEMU creates new ones
            # This ensures QEMU gets a clean state for chardev sockets
            for socket_path in [cmd_socket, event_socket, qmp_socket_path]:
                with contextlib.suppress(OSError):
                    await aiofiles.os.remove(socket_path)

            channel: GuestChannel = DualPortChannel(cmd_socket, event_socket)

            # If cgroups unavailable, wrap with ulimit for host resource control
            # ulimit works on Linux, macOS, BSD (POSIX)
            cgroups_available = str(cgroup_path).startswith("/sys/fs/cgroup")
            if not cgroups_available:
                qemu_cmd = self._wrap_with_ulimit(qemu_cmd, memory_mb)

            # Step 7: Launch QEMU
            try:
                qemu_proc = ProcessWrapper(
                    await asyncio.create_subprocess_exec(
                        *qemu_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True,  # Create new process group for proper cleanup
                    )
                )

                # Attach process to cgroup (only if cgroups available)
                if cgroups_available and qemu_proc.pid is not None:
                    await self._attach_to_cgroup(cgroup_path, qemu_proc.pid)

                # Check if process crashed immediately BEFORE starting drain task
                # This ensures we can capture stderr/stdout on immediate crash
                await asyncio.sleep(0.1)
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
                console_log_path = Path(f"/tmp/vm-{vm_id}-console.log")  # noqa: S108
                console_log = console_log_path.open("w", buffering=1)  # noqa: ASYNC230 - Line buffering for sync writes

                def write_to_console(line: str) -> None:
                    """Write line to console log file and structured logs."""
                    try:
                        console_log.write(f"[{vm_id}] {line}\n")
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
                qemu_log_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

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
            vm = QemuVM(
                vm_id,
                qemu_proc,
                cgroup_path,
                overlay_image,
                channel,
                language,
                gvproxy_proc,
                gvproxy_socket,
                qmp_socket_path,
                qemu_log_task,
                gvproxy_log_task,
                console_log,
                use_qemu_vm_user,
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
                # Store timing on VM for scheduler to use
                vm.setup_ms = int((setup_complete_time - start_time) * 1000)
                vm.boot_ms = int((boot_complete_time - setup_complete_time) * 1000)
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
                        "overlay_image": str(overlay_image) if overlay_image else "(none)",
                        "kernel_path": str(kernel_path),
                        "initramfs_path": str(initramfs_path),
                    },
                )

                await vm.destroy()

                # Get QEMU binary from command - handle ulimit wrapper case
                # When wrapped with sh -c, qemu_cmd is ["sh", "-c", "ulimit ... && exec qemu-system-..."]
                qemu_binary = "(unknown)"
                if qemu_cmd:
                    if qemu_cmd[0] == "sh" and len(qemu_cmd) > 2:  # noqa: PLR2004
                        # Extract actual QEMU binary from shell command string
                        import re  # noqa: PLC0415

                        shell_cmd_str = qemu_cmd[2]
                        qemu_match = re.search(r"(qemu-system-[^\s]+)", shell_cmd_str)
                        qemu_binary = qemu_match.group(1) if qemu_match else f"sh -c '{shell_cmd_str[:100]}...'"
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
                        "overlay_image": str(overlay_image) if overlay_image else None,
                    },
                ) from e

            # Log boot time with breakdown
            total_boot_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.info(
                "VM created",
                extra={
                    "vm_id": vm_id,
                    "language": language,
                    "setup_ms": vm.setup_ms,
                    "boot_ms": vm.boot_ms,
                    "total_boot_ms": total_boot_ms,
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
                # Remove from registry if it was added (defensive - always try)
                async with self._vms_lock:
                    self._vms.pop(vm_id, None)

                await self._force_cleanup_all_resources(
                    vm_id=vm_id,
                    qemu_proc=qemu_proc,
                    gvproxy_proc=gvproxy_proc,
                    gvproxy_socket=gvproxy_socket,
                    overlay_image=overlay_image,
                    cgroup_path=cgroup_path,
                    use_qemu_vm_user=use_qemu_vm_user,
                )

    async def _start_gvproxy(
        self,
        vm_id: str,
        allowed_domains: list[str] | None,
        language: str,
    ) -> tuple[ProcessWrapper, Path, asyncio.Task[None]]:
        r"""Start gvproxy-wrapper with DNS filtering for this VM.

        Architecture Decision: gvisor-tap-vsock over alternatives
        ========================================================

        Chosen: gvisor-tap-vsock
        - ✅ Built-in DNS filtering via zones (regex-based)
        - ✅ Production-ready (Podman default since 2022)
        - ✅ 10MB memory overhead per VM
        - ✅ Simple JSON zone configuration
        - ✅ Zero CVEs (vs SLIRP: CVE-2021-3592/3/4/5, CVE-2020-29129/30)

        Args:
            vm_id: Unique VM identifier
            allowed_domains: Whitelist of allowed domains
            language: Programming language (for default registries)

        Returns:
            Tuple of (gvproxy_process, socket_path, gvproxy_log_task)

        Raises:
            VmError: Failed to start gvproxy-wrapper
        """
        socket_path = self._get_socket_path(vm_id, prefix="gvproxy")

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

        # Start gvproxy-wrapper
        gvproxy_binary = _get_gvproxy_wrapper_path()
        try:
            proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    str(gvproxy_binary),
                    "-listen-qemu",
                    f"unix://{socket_path}",
                    "-dns-zones",
                    dns_zones_json,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,  # Create new process group for proper cleanup
                )
            )
        except (OSError, FileNotFoundError) as e:
            raise VmError(
                f"Failed to start gvproxy-wrapper: {e}",
                context={
                    "vm_id": vm_id,
                    "language": language,
                    "allowed_domains": allowed_domains,
                    "binary_path": str(gvproxy_binary),
                },
            ) from e

        # Background task to drain gvproxy output (prevent pipe deadlock)
        gvproxy_log_task = asyncio.create_task(
            drain_subprocess_output(
                proc,
                process_name="gvproxy-wrapper",
                context_id=vm_id,
                stdout_handler=lambda line: logger.debug("[gvproxy-wrapper]", extra={"vm_id": vm_id, "output": line}),
                stderr_handler=lambda line: logger.error(
                    "[gvproxy-wrapper error]", extra={"vm_id": vm_id, "output": line}
                ),
            )
        )
        gvproxy_log_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

        # Wait for socket creation (max 5 seconds)
        for _ in range(50):
            if socket_path.exists():
                break
            await asyncio.sleep(0.1)
        else:
            await proc.terminate()
            await proc.wait()
            raise VmError(
                f"gvproxy-wrapper socket not created: {socket_path}",
                context={
                    "vm_id": vm_id,
                    "language": language,
                    "socket_path": str(socket_path),
                    "allowed_domains": allowed_domains,
                },
            )

        # Fix socket permissions so qemu-vm user can connect
        chmod_proc = ProcessWrapper(
            await asyncio.create_subprocess_exec(
                "chmod",
                "666",
                str(socket_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        )
        await chmod_proc.communicate()

        logger.info(
            "gvproxy-wrapper started successfully",
            extra={
                "vm_id": vm_id,
                "socket": str(socket_path),
                "dns_filtering": True,
            },
        )

        return proc, socket_path, gvproxy_log_task

    async def _force_cleanup_all_resources(
        self,
        vm_id: str,
        qemu_proc: ProcessWrapper | None = None,
        gvproxy_proc: ProcessWrapper | None = None,
        gvproxy_socket: Path | None = None,
        overlay_image: Path | None = None,
        cgroup_path: Path | None = None,
        use_qemu_vm_user: bool = False,
    ) -> dict[str, bool]:
        """Comprehensive cleanup of ALL VM resources in reverse dependency order.

        This is the MAIN cleanup method used in finally blocks.

        Best practices:
        - Cleans in reverse dependency order (processes → files → directories)
        - NEVER raises exceptions (logs errors instead)
        - Safe to call multiple times (idempotent)
        - Handles None/already-cleaned resources
        - Returns status dict for monitoring/debugging
        - Uses standardized generic cleanup functions

        Cleanup order (reverse dependencies):
        1. QEMU process (depends on: overlay, cgroup, networking)
        2. gvproxy process (QEMU networking dependency)
        3. gvproxy socket (gvproxy dependency)
        4. Overlay file (QEMU disk dependency)
        5. Cgroup directory (QEMU process was in it)

        Args:
            vm_id: VM identifier for logging
            qemu_proc: QEMU subprocess (can be None)
            gvproxy_proc: gvproxy subprocess (can be None)
            gvproxy_socket: gvproxy socket path (can be None)
            overlay_image: qcow2 overlay path (can be None)
            cgroup_path: cgroup directory path (can be None)
            use_qemu_vm_user: Whether QEMU ran as qemu-vm user (requires sudo rm for overlay)

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

        # Phase 2: Cleanup files in parallel (after processes dead)
        file_results = await asyncio.gather(
            cleanup_file(
                file_path=gvproxy_socket,
                context_id=vm_id,
                description="gvproxy socket",
            ),
            cleanup_overlay(
                overlay_path=overlay_image,
                context_id=vm_id,
                use_qemu_vm_user=use_qemu_vm_user,
            ),
            cleanup_cgroup(
                cgroup_path=cgroup_path,
                context_id=vm_id,
            ),
            return_exceptions=True,
        )
        results["socket"] = file_results[0] if isinstance(file_results[0], bool) else False
        results["overlay"] = file_results[1] if isinstance(file_results[1], bool) else False
        results["cgroup"] = file_results[2] if isinstance(file_results[2], bool) else False

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
                gvproxy_socket=vm.gvproxy_socket,
                overlay_image=vm.overlay_image,
                cgroup_path=vm.cgroup_path,
                use_qemu_vm_user=vm.use_qemu_vm_user,
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

    def _get_socket_path(self, vm_id: str, prefix: str = "qemu-serial") -> Path:
        """Generate short socket path from VM ID using full SHA-256 hash.

        UNIX domain sockets have 108-byte path limit.

        Args:
            vm_id: Unique VM identifier
            prefix: Socket file prefix (default: qemu-serial)

        Returns:
            Path to socket file with hashed vm_id
        """
        hash_hex = hashlib.sha256(vm_id.encode()).hexdigest()
        return Path(f"/tmp/{prefix}-{hash_hex}.sock")  # noqa: S108

    def _get_cmd_socket_path(self, vm_id: str) -> str:
        """Get path to virtio-serial Unix socket for command channel (host → guest).

        Args:
            vm_id: Unique VM identifier

        Returns:
            /tmp/cmd-{hash}.sock
        """
        return str(self._get_socket_path(vm_id, prefix="cmd"))

    def _get_event_socket_path(self, vm_id: str) -> str:
        """Get path to virtio-serial Unix socket for event channel (guest → host).

        Args:
            vm_id: Unique VM identifier

        Returns:
            /tmp/event-{hash}.sock
        """
        return str(self._get_socket_path(vm_id, prefix="event"))

    def _get_qmp_socket_path(self, vm_id: str) -> Path:
        """Get path to QMP (QEMU Monitor Protocol) Unix socket.

        Used for memory snapshot operations (savevm/loadvm).

        Args:
            vm_id: Unique VM identifier

        Returns:
            /tmp/qmp-{hash}.sock
        """
        return self._get_socket_path(vm_id, prefix="qmp")

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
        if _check_kvm_available():
            return AccelType.KVM
        if detect_host_os() == HostOS.MACOS and await _check_hvf_available():
            return AccelType.HVF
        return AccelType.TCG

    async def _create_overlay(self, base_image: Path, overlay_image: Path) -> bool:
        """Create ephemeral qcow2 overlay with CoW backing file.

        Args:
            base_image: Base qcow2 image (backing file)
            overlay_image: Ephemeral overlay to create

        Returns:
            True if overlay was chowned to qemu-vm (QEMU should run as qemu-vm),
            False if overlay is owned by current user (QEMU should run as current user)

        Raises:
            VmError: qemu-img command failed
        """
        cmd = [
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            str(base_image),
            "-o",
            "lazy_refcounts=on,extended_l2=on,cluster_size=128k",
            str(overlay_image),
        ]

        proc = ProcessWrapper(
            await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        )
        _stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise VmError(f"qemu-img create failed: {stderr.decode()}")

        # Change ownership to qemu-vm user for process isolation (optional hardening)
        # Only if: Linux + qemu-vm user exists + have sudo access
        # Returns whether QEMU should run as qemu-vm user (based on chown success)
        if detect_host_os() != HostOS.MACOS and await _probe_qemu_vm_user():
            # Make base image accessible to qemu-vm user
            # qemu-vm needs: read on file + execute on all parent directories
            # This is safe because the base image is read-only (writes go to overlay)

            # Make all parent directories traversable (a+x) up to /tmp or root
            current = base_image.parent
            dirs_to_chmod: list[Path] = []
            while current != current.parent and str(current) not in ("/", "/tmp"):  # noqa: S108
                dirs_to_chmod.append(current)
                current = current.parent

            for dir_path in dirs_to_chmod:
                chmod_dir_proc = ProcessWrapper(
                    await asyncio.create_subprocess_exec(
                        "chmod",
                        "a+x",
                        str(dir_path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True,
                    )
                )
                await chmod_dir_proc.communicate()

            # Make base image readable
            chmod_proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    "chmod",
                    "a+r",
                    str(base_image),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            )
            await chmod_proc.communicate()
            if chmod_proc.returncode != 0:
                logger.debug("Could not chmod base image (qemu-vm may not have access)")

            chown_proc = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    "sudo",
                    "chown",
                    "qemu-vm:qemu-vm",
                    str(overlay_image),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            )
            await chown_proc.communicate()
            if chown_proc.returncode == 0:
                return True  # Overlay chowned to qemu-vm, QEMU should run as qemu-vm
            logger.debug("Could not chown overlay to qemu-vm user (optional hardening)")
            return False  # Chown failed, QEMU should run as current user

        return False  # qemu-vm not available, QEMU should run as current user

    async def _setup_cgroup(self, vm_id: str, tenant_id: str) -> Path:
        """Set up cgroup v2 resource limits.

        Limits:
        - memory.max: 512MB
        - cpu.max: 100000 (1 vCPU)
        - pids.max: 100 (fork bomb prevention)

        Args:
            vm_id: Unique VM identifier
            tenant_id: Tenant identifier

        Returns:
            Path to cgroup directory (dummy path if cgroups unavailable)

        Note:
            Gracefully degrades to no resource limits on Docker Desktop (read-only /sys/fs/cgroup)
            or environments without cgroup v2 support
        """
        tenant_cgroup = Path(f"/sys/fs/cgroup/code-exec/{tenant_id}")
        cgroup_path = tenant_cgroup / vm_id

        try:
            # Create tenant cgroup and enable controllers for nested VM cgroups
            # In cgroup v2, subtree_control only affects immediate children,
            # so we must enable controllers at each level of the hierarchy
            await aiofiles.os.makedirs(tenant_cgroup, exist_ok=True)
            async with aiofiles.open(tenant_cgroup / "cgroup.subtree_control", "w") as f:
                await f.write("+memory +cpu +pids")

            # Create VM cgroup
            await aiofiles.os.makedirs(cgroup_path, exist_ok=True)

            # Set memory limit (512MB default + overhead)
            memory_mb = 512
            async with aiofiles.open(cgroup_path / "memory.max", "w") as f:
                await f.write(str((memory_mb + constants.CGROUP_MEMORY_OVERHEAD_MB) * 1024 * 1024))

            # Set CPU limit (1 vCPU)
            async with aiofiles.open(cgroup_path / "cpu.max", "w") as f:
                await f.write("100000 100000")

            # Set PID limit (fork bomb prevention)
            async with aiofiles.open(cgroup_path / "pids.max", "w") as f:
                await f.write(str(constants.CGROUP_PIDS_LIMIT))

            # Verify cgroup.procs is writable (required for attaching processes)
            # Writing to control files (memory.max, etc.) requires different privileges
            # than writing to cgroup.procs, which needs proper systemd delegation
            async with aiofiles.open(cgroup_path / "cgroup.procs", "a") as f:
                pass  # Just test we can open for writing

        except OSError as e:
            # Gracefully degrade if cgroups unavailable (e.g., Docker Desktop, CI runners)
            # Note: PermissionError is a subclass of OSError
            if e.errno in (constants.ERRNO_READ_ONLY_FILESYSTEM, constants.ERRNO_PERMISSION_DENIED):
                logger.warning(
                    "cgroup v2 unavailable, resource limits disabled",
                    extra={"vm_id": vm_id, "path": str(cgroup_path), "errno": e.errno},
                )
                return Path(f"/tmp/cgroup-{vm_id}")  # noqa: S108
            raise VmError(f"Failed to setup cgroup: {e}") from e

        return cgroup_path

    async def _attach_to_cgroup(self, cgroup_path: Path, pid: int) -> None:
        """Attach process to cgroup.

        Args:
            cgroup_path: cgroup directory
            pid: Process ID to attach

        Raises:
            VmError: Failed to attach process
        """
        try:
            async with aiofiles.open(cgroup_path / "cgroup.procs", "w") as f:
                await f.write(str(pid))
        except (OSError, PermissionError) as e:
            raise VmError(f"Failed to attach PID {pid} to cgroup: {e}") from e

    def _wrap_with_ulimit(self, qemu_cmd: list[str], memory_mb: int) -> list[str]:
        """Wrap QEMU command with ulimit for resource control (cgroups alternative).

        Args:
            qemu_cmd: Original QEMU command
            memory_mb: Guest memory in MB

        Returns:
            Command wrapped with ulimit via sh -c
        """
        import shlex  # noqa: PLC0415

        qemu_cmd_str = " ".join(shlex.quote(arg) for arg in qemu_cmd)

        # QEMU overhead: ~1.5-2x guest memory for KVM
        virtual_mem_kb = memory_mb * 1024 * constants.ULIMIT_MEMORY_MULTIPLIER

        # Platform-specific: macOS kernel doesn't support modifying virtual memory limits
        if detect_host_os() == HostOS.MACOS:
            shell_cmd = f"exec {qemu_cmd_str}"
        else:
            shell_cmd = f"ulimit -v {virtual_mem_kb} && exec {qemu_cmd_str}"

        return ["sh", "-c", shell_cmd]

    async def _build_linux_cmd(  # noqa: PLR0912, PLR0915
        self,
        language: str,  # noqa: ARG002
        vm_id: str,
        overlay_image: Path,
        memory_mb: int,
        allow_network: bool,
        enable_dns_filtering: bool = False,  # noqa: ARG002
        gvproxy_socket: Path | None = None,
        loadvm_snapshot: str | None = None,
        use_qemu_vm_user: bool = False,
    ) -> list[str]:
        """Build QEMU command for Linux (KVM + unshare + namespaces).

        Args:
            language: Programming language
            vm_id: Unique VM identifier
            overlay_image: Ephemeral qcow2 overlay
            memory_mb: Guest VM memory in MB
            allow_network: Enable network access
            enable_dns_filtering: Enable DNS filtering via gvisor-tap-vsock
            gvproxy_socket: QEMU stream socket path for gvproxy connection
            loadvm_snapshot: Optional snapshot name to restore from (for fast VM boot)
            use_qemu_vm_user: Run QEMU as qemu-vm user (optional hardening)

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
            # MTTCG is enabled by default for supported configs, no need to force it
            # tb-size=512 increases translation block cache for better code caching
            accel = "tcg,tb-size=512"

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
                tsc_available = await _check_tsc_deadline_available()
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
                # macOS with HVF uses microvm in non-legacy mode
                # HVF always has TSC_DEADLINE equivalent, can disable pit/pic
                machine_type = "microvm,acpi=off,x-option-roms=off,pit=off,pic=off,rtc=off,isa-serial=off,mem-merge=off"
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
        # │ ARM64 virt              │ hvc0        │ No ISA serial on ARM           │
        # └─────────────────────────┴─────────────┴────────────────────────────────┘
        # ttyS0 (ISA serial) is used when we need reliable early boot console
        # hvc0 (virtio-console) is used when ISA serial is disabled
        # See: https://blog.memzero.de/toying-with-virtio/
        if self.arch == HostArch.AARCH64:
            # ARM64 virt machine has no ISA serial, must use virtio-console
            console_params = "console=hvc0 loglevel=7"
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
                # See: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1033643
                ("host" if accel_type in (AccelType.HVF, AccelType.KVM) else "cortex-a57" if self.arch == HostArch.AARCH64 else "qemu64"),
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
                f"{console_params} root=/dev/vda rootflags=rw,noatime rootfstype=ext4 rootwait=2 fsck.mode=skip reboot=t preempt=none i8042.noaux i8042.nomux i8042.nopnp init=/init random.trust_cpu=on raid=noautodetect mitigations=off",
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
        qemu_args.extend(
            [
                "-drive",
                f"file={overlay_image},"
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
                f"iops_max={constants.DISK_IOPS_BURST}",
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
        cmd_socket_path = self._get_cmd_socket_path(vm_id)
        event_socket_path = self._get_event_socket_path(vm_id)
        qemu_args.extend(
            [
                # Chardevs for communication channels
                # server=on: QEMU creates a listening Unix socket
                # wait=off: QEMU starts VM immediately without waiting for client connection
                # Note: Socket permissions (via umask 000) are the key fix for Linux,
                # NOT wait=on. The guest agent retries connection so timing is handled.
                "-chardev",
                f"socket,id=cmd0,path={cmd_socket_path},server=on,wait=off",
                "-chardev",
                f"socket,id=event0,path={event_socket_path},server=on,wait=off",
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

        # virtio-balloon for memory reclamation before snapshots
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
            if not gvproxy_socket:
                raise RuntimeError(f"gvproxy socket required for network-enabled VM but not provided (vm_id={vm_id})")
            qemu_args.extend(
                [
                    "-netdev",
                    f"stream,id=net0,addr.type=unix,addr.path={gvproxy_socket}",
                    "-device",
                    f"virtio-net-{virtio_suffix},netdev=net0,mq=off,csum=off,gso=off,host_tso4=off,host_tso6=off,mrg_rxbuf=off,ctrl_rx=off,guest_announce=off",
                ]
            )

        # QMP (QEMU Monitor Protocol) socket for memory snapshot operations
        # Used for savevm/loadvm commands to create/restore memory snapshots
        qmp_socket_path = self._get_qmp_socket_path(vm_id)
        qemu_args.extend(
            [
                "-qmp",
                f"unix:{qmp_socket_path},server=on,wait=off",
            ]
        )

        # Restore from memory snapshot if specified
        # This provides fast VM boot (~50-200ms vs ~300-400ms cold boot)
        if loadvm_snapshot:
            qemu_args.extend(["-loadvm", loadvm_snapshot])

        # Run QEMU as unprivileged user if qemu-vm user is available (optional hardening)
        # Falls back to current user if qemu-vm doesn't exist - VM still provides isolation
        if use_qemu_vm_user:
            # stdbuf -oL forces line-buffered stdout to ensure console output is captured
            # immediately rather than being block-buffered (which happens with piped stdout)
            # IMPORTANT: stdbuf must come AFTER sudo - sudo sanitizes LD_PRELOAD for security
            #
            # umask 000: Create chardev sockets with world-readable/writable permissions (0666).
            # Without this, sockets are created with restrictive permissions (e.g., 0600)
            # and the host process cannot connect to sockets owned by 'qemu-vm'.
            import shlex  # noqa: PLC0415

            qemu_cmd_str = " ".join(shlex.quote(arg) for arg in qemu_args)
            cmd.extend(["sudo", "-u", "qemu-vm", "sh", "-c", f"umask 000 && exec stdbuf -oL {qemu_cmd_str}"])
            return cmd  # Already added qemu_args via sh -c

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
            console_log_path = f"/tmp/vm-{vm.vm_id}-console.log"  # noqa: S108
            console_output = await read_log_tail(console_log_path, constants.CONSOLE_LOG_MAX_BYTES)

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
                try:
                    await vm.channel.connect(timeout_seconds=2)
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
                            with contextlib.suppress(asyncio.CancelledError, OSError, RuntimeError):
                                await guest_task
                            await death_task  # Re-raise VmError

                        # Guest task completed - check result (raises if failed, triggering retry)
                        await guest_task

                # Success - cancel death monitor
                death_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await death_task

        except TimeoutError:
            # Clean up in-flight tasks
            for task in (death_task, guest_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (death_task, guest_task):
                if task is not None:
                    with contextlib.suppress(asyncio.CancelledError, OSError, RuntimeError):
                        await task

            # Flush console log before reading to ensure all buffered content is written
            if vm.console_log:
                with contextlib.suppress(OSError):
                    vm.console_log.flush()

            console_log_path = f"/tmp/vm-{vm.vm_id}-console.log"  # noqa: S108
            console_output = await read_log_tail(console_log_path, constants.CONSOLE_LOG_MAX_BYTES)

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

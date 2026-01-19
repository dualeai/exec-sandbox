"""System capability probes for VM acceleration and feature detection.

These probes detect system capabilities once and cache the results.
Async probes use a shared cache container to avoid global statements.
"""

import asyncio
import logging
import os
import platform
import sys
from pathlib import Path

import aiofiles
import aiofiles.os

from exec_sandbox.exceptions import VmDependencyError
from exec_sandbox.permission_utils import can_access
from exec_sandbox.platform_utils import HostArch, HostOS, detect_host_arch, detect_host_os
from exec_sandbox.vm_types import AccelType

logger = logging.getLogger(__name__)

# Public exports for backward compatibility with vm_manager
__all__ = [
    "_ProbeCache",
    "_check_hvf_available",
    "_check_kvm_available",
    "_check_tsc_deadline",
    "_check_tsc_deadline_linux",
    "_check_tsc_deadline_macos",
    "_kernel_validated",
    "_probe_cache",
    "_probe_io_uring_support",
    "_probe_qemu_accelerators",
    "_probe_unshare_support",
    "_validate_kernel_initramfs",
    "check_fast_balloon_available",
    "check_hwaccel_available",
    "detect_accel_type",
]

# KVM ioctl constants for probing
# See: linux/kvm.h - these are stable ABI
_KVM_GET_API_VERSION = 0xAE00
_KVM_API_VERSION_EXPECTED = 12  # Stable since Linux 2.6.38


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
        raise VmDependencyError(
            f"Kernel not found: {kernel}",
            context={"kernel_path": str(kernel), "arch": arch_suffix},
        )
    if not await aiofiles.os.path.exists(initramfs):
        raise VmDependencyError(
            f"Initramfs not found: {initramfs}",
            context={"initramfs_path": str(initramfs), "arch": arch_suffix},
        )

    _kernel_validated.add(cache_key)


async def detect_accel_type(
    kvm_available: bool | None = None, hvf_available: bool | None = None, force_emulation: bool = False
) -> AccelType:
    """Detect which QEMU accelerator to use.

    This is the single source of truth for virtualization mode detection.
    Used for both cgroup memory sizing (TCG needs more) and QEMU command building.

    Args:
        kvm_available: Override KVM check result (for testing)
        hvf_available: Override HVF check result (for testing)
        force_emulation: Force TCG software emulation

    Returns:
        AccelType.KVM if Linux KVM available
        AccelType.HVF if macOS HVF available
        AccelType.TCG if software emulation needed (or force_emulation=True)
    """
    if force_emulation:
        return AccelType.TCG
    if kvm_available is None:
        kvm_available = await _check_kvm_available()
    if kvm_available:
        return AccelType.KVM
    if detect_host_os() == HostOS.MACOS:
        if hvf_available is None:
            hvf_available = await _check_hvf_available()
        if hvf_available:
            return AccelType.HVF
    return AccelType.TCG

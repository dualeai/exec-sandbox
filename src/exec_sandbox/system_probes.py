"""System capability probes for VM acceleration and feature detection.

These probes detect system capabilities once and cache the results.
Async probes use a shared cache container to avoid global statements.
"""

import asyncio
import logging
import os
import platform
import re
import sys
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import aiofiles
import aiofiles.os

from exec_sandbox.permission_utils import can_access
from exec_sandbox.platform_utils import HostArch, HostOS, detect_host_arch, detect_host_os
from exec_sandbox.vm_types import AccelType

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = [
    "NOT_CACHED",
    "ProbeCache",
    "check_fast_balloon_available",
    "check_hvf_available",
    "check_hwaccel_available",
    "check_kvm_available",
    "check_tsc_deadline",
    "detect_accel_type",
    "probe_cache",
    "probe_io_uring_support",
    "probe_qemu_accelerators",
    "probe_qemu_version",
    "probe_unshare_support",
]

# Sentinel value for "not yet cached" (distinguishes from None results)
_NOT_CACHED: Any = object()
NOT_CACHED = _NOT_CACHED  # Public alias for testing

# KVM ioctl constants for probing
# See: linux/kvm.h - these are stable ABI
_KVM_GET_API_VERSION = 0xAE00
_KVM_API_VERSION_EXPECTED = 12  # Stable since Linux 2.6.38


class ProbeCache:
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
        "qemu_version",
        "tsc_deadline",
        "unshare",
    )

    def __init__(self) -> None:
        self.hvf: bool | None = _NOT_CACHED
        self.io_uring: bool | None = _NOT_CACHED
        self.kvm: bool | None = _NOT_CACHED
        self.qemu_accels: set[str] | None = _NOT_CACHED  # Accelerators available in QEMU binary
        self.qemu_version: tuple[int, int, int] | None = _NOT_CACHED  # QEMU semver (major, minor, patch)
        self.tsc_deadline: bool | None = _NOT_CACHED
        self.unshare: bool | None = _NOT_CACHED
        self._locks: dict[str, asyncio.Lock] = {}

    def get_lock(self, name: str) -> asyncio.Lock:
        """Get or create a lock for the given probe (lazy initialization).

        Locks must be created lazily because asyncio.Lock requires an event loop,
        which may not exist at module import time.
        """
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    def reset(self, attr: str | None = None) -> None:
        """Reset cached probe result(s) to uncached state (for testing).

        Args:
            attr: Specific attribute to reset, or None to reset all.
        """
        if attr is not None:
            setattr(self, attr, _NOT_CACHED)
        else:
            self.hvf = _NOT_CACHED
            self.io_uring = _NOT_CACHED
            self.kvm = _NOT_CACHED
            self.qemu_accels = _NOT_CACHED
            self.qemu_version = _NOT_CACHED
            self.tsc_deadline = _NOT_CACHED
            self.unshare = _NOT_CACHED


# Module-level cache instance
probe_cache = ProbeCache()


def _async_cached_probe(
    cache_attr: str,
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Decorator for async probes with double-checked locking cache.

    Eliminates boilerplate pattern:
        if cache.attr is not _NOT_CACHED: return cache.attr
        async with cache.get_lock("attr"):
            if cache.attr is not _NOT_CACHED: return cache.attr
            result = await actual_work()
            cache.attr = result
            return result

    Uses a sentinel value (_NOT_CACHED) to distinguish "not yet cached" from
    cached None results, allowing probes that return None on failure to be cached.

    Args:
        cache_attr: Name of the attribute on probe_cache to use for caching

    Returns:
        Decorated async function with automatic caching
    """

    def decorator(func: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Fast path: return cached result (no lock needed)
            cached = getattr(probe_cache, cache_attr)
            if cached is not _NOT_CACHED:
                return cached
            # Slow path: acquire lock to prevent stampede, then check cache again
            async with probe_cache.get_lock(cache_attr):
                # Double-check after acquiring lock (another task may have populated cache)
                cached = getattr(probe_cache, cache_attr)
                if cached is not _NOT_CACHED:
                    return cached
                result = await func(*args, **kwargs)
                setattr(probe_cache, cache_attr, result)
                return result

        return wrapper

    return decorator


@_async_cached_probe("qemu_accels")
async def probe_qemu_accelerators() -> set[str]:
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
            return set()

        # Parse output: "Accelerators supported in QEMU binary:\ntcg\nkvm\n"
        # or "tcg\nhvf\n" on macOS
        output = stdout.decode().strip()
        accels: set[str] = set()
        for raw_line in output.split("\n"):
            accel_name = raw_line.strip().lower()
            # Skip header line and empty lines
            if accel_name and not accel_name.startswith("accelerator"):
                accels.add(accel_name)

        logger.debug(
            "QEMU accelerator probe complete",
            extra={"qemu_bin": qemu_bin, "accelerators": sorted(accels)},
        )
        return accels

    except FileNotFoundError:
        logger.warning(
            "QEMU binary not found for accelerator probe",
            extra={"qemu_bin": qemu_bin},
        )
        return set()
    except (OSError, TimeoutError) as e:
        logger.warning(
            "QEMU accelerator probe failed",
            extra={"qemu_bin": qemu_bin, "error": str(e)},
        )
        return set()


@_async_cached_probe("qemu_version")
async def probe_qemu_version() -> tuple[int, int, int] | None:
    """Probe QEMU binary version (cached).

    Returns:
        Semver tuple (major, minor, patch), or None if detection fails.
        Example: (8, 2, 0) for QEMU 8.2.0

    Tuples compare naturally as semver: (9, 2, 0) >= (9, 2, 0) works correctly.
    """
    # Determine QEMU binary based on host architecture
    arch = detect_host_arch()
    qemu_bin = "qemu-system-aarch64" if arch == HostArch.AARCH64 else "qemu-system-x86_64"

    try:
        proc = await asyncio.create_subprocess_exec(
            qemu_bin,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)

        if proc.returncode != 0:
            logger.warning("QEMU version probe failed", extra={"returncode": proc.returncode})
            return None

        # Parse "QEMU emulator version X.Y.Z" (handles suffixes like "(Homebrew)")
        output = stdout.decode().strip()
        match = re.search(r"QEMU emulator version (\d+)\.(\d+)(?:\.(\d+))?", output)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            patch = int(match.group(3)) if match.group(3) else 0
            logger.debug("QEMU version detected", extra={"version": f"{major}.{minor}.{patch}"})
            return (major, minor, patch)

        logger.warning("Could not parse QEMU version", extra={"output": output[:100]})
        return None

    except FileNotFoundError:
        logger.warning("QEMU binary not found", extra={"qemu_bin": qemu_bin})
        return None
    except (OSError, TimeoutError) as e:
        logger.warning("QEMU version probe failed", extra={"error": str(e)})
        return None


@_async_cached_probe("kvm")
async def check_kvm_available() -> bool:
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
    kvm_path = "/dev/kvm"
    if not await aiofiles.os.path.exists(kvm_path):
        logger.debug("KVM not available: /dev/kvm does not exist")
        return False

    # Check if we can actually access /dev/kvm (not just that it exists)
    # This catches permission issues that would cause QEMU to fail or hang
    # See: https://github.com/actions/runner-images/issues/8542
    if not await can_access(kvm_path, os.R_OK | os.W_OK):
        logger.debug("KVM not available: permission denied on /dev/kvm")
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
            return False

        api_version = int(stdout.decode().strip())
        if api_version != _KVM_API_VERSION_EXPECTED:
            logger.warning(
                "KVM available but unexpected API version",
                extra={"api_version": api_version, "expected": _KVM_API_VERSION_EXPECTED},
            )
            return False

        logger.debug("KVM ioctl check passed", extra={"api_version": api_version})

    except (OSError, TimeoutError, ValueError) as e:
        logger.debug("KVM not available: failed to verify /dev/kvm", extra={"error": str(e)})
        return False

    # Layer 2: Verify QEMU binary has KVM support compiled in
    # Even if /dev/kvm works, QEMU may not have KVM support or may fail to initialize it
    # See: https://github.com/orgs/community/discussions/8305 (GitHub Actions KVM issues)
    qemu_accels = await probe_qemu_accelerators()
    if "kvm" not in qemu_accels:
        logger.warning(
            "KVM not available: QEMU binary does not support KVM accelerator",
            extra={"available_accelerators": sorted(qemu_accels)},
        )
        return False

    logger.debug("KVM available and working (kernel + QEMU verified)")
    return True


@_async_cached_probe("hvf")
async def check_hvf_available() -> bool:
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
            return False

        logger.debug("HVF kernel support check passed")

    except (OSError, TimeoutError) as e:
        logger.debug("HVF not available: sysctl check failed", extra={"error": str(e)})
        return False

    # Layer 2: Verify QEMU binary has HVF support compiled in
    # Even if kern.hv_support is enabled, QEMU may not have HVF support
    qemu_accels = await probe_qemu_accelerators()
    if "hvf" not in qemu_accels:
        logger.warning(
            "HVF not available: QEMU binary does not support HVF accelerator",
            extra={"available_accelerators": sorted(qemu_accels)},
        )
        return False

    logger.debug("HVF available and working (kernel + QEMU verified)")
    return True


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
        return asyncio.run(check_kvm_available())
    if host_os == HostOS.MACOS:
        return asyncio.run(check_hvf_available())
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
        return asyncio.run(check_tsc_deadline())

    # On macOS, HVF availability implies not nested (macOS doesn't support nested virt)
    # If we got here, HVF is available, so balloon should be fast
    if host_os == HostOS.MACOS:
        return True

    # On ARM64 Linux, KVM availability is sufficient
    # ARM uses GIC timer (not TSC/APIC), less affected by nested virt overhead
    # Unknown platform: conservative assumption (return False)
    return host_os == HostOS.LINUX and host_arch == HostArch.AARCH64


@_async_cached_probe("tsc_deadline")
async def check_tsc_deadline() -> bool:
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
    # TSC_DEADLINE is x86-only
    if detect_host_arch() != HostArch.X86_64:
        return False

    # Dispatch to platform-specific implementation
    host_os = detect_host_os()
    if host_os == HostOS.LINUX:
        return await _check_tsc_deadline_linux()
    if host_os == HostOS.MACOS:
        return await _check_tsc_deadline_macos()

    # Unknown platform
    return False


async def _check_tsc_deadline_linux() -> bool:
    """Linux-specific TSC_DEADLINE check via /proc/cpuinfo.

    Note: Called from check_tsc_deadline() which handles caching and locking.
    """
    cpuinfo_path = "/proc/cpuinfo"
    if not await aiofiles.os.path.exists(cpuinfo_path):
        return False

    try:
        async with aiofiles.open(cpuinfo_path) as f:
            cpuinfo = await f.read()
        # Look for tsc_deadline_timer in the flags line
        # Format: "flags : fpu vme ... tsc_deadline_timer ..."
        for line in cpuinfo.split("\n"):
            if line.startswith("flags"):
                has_tsc = "tsc_deadline_timer" in line.split()
                if has_tsc:
                    logger.debug("TSC_DEADLINE available (can disable PIT/PIC)")
                else:
                    logger.debug("TSC_DEADLINE not available (keeping PIT/PIC enabled)")
                return has_tsc
    except OSError as e:
        logger.warning("Failed to read /proc/cpuinfo for TSC_DEADLINE check", extra={"error": str(e)})

    return False


async def _check_tsc_deadline_macos() -> bool:
    """macOS-specific TSC_DEADLINE check via sysctl.

    Note: Called from check_tsc_deadline() which handles caching and locking.
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
            if has_tsc:
                logger.debug("TSC_DEADLINE available on macOS (can disable PIT/PIC)")
            else:
                logger.debug("TSC_DEADLINE not available on macOS (keeping legacy timers)")
            return has_tsc
    except (OSError, TimeoutError):
        pass

    return False


@_async_cached_probe("io_uring")
async def probe_io_uring_support() -> bool:
    """Probe for io_uring support using syscall test (cached).

    Returns:
        True if io_uring fully available, False otherwise
    """
    # io_uring is Linux-only - immediately return False on other platforms
    if detect_host_os() != HostOS.LINUX:
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
            return True
        if proc.returncode == 1:
            logger.info(
                "io_uring syscall not available (ENOSYS)",
                extra={"kernel": platform.release()},
            )
            return False
        if proc.returncode == 2:  # noqa: PLR2004
            logger.warning(
                "io_uring blocked by seccomp/container policy",
                extra={"kernel": platform.release()},
            )
            return False

        logger.warning(
            "io_uring probe failed with unexpected result",
            extra={"exit_code": proc.returncode},
        )
        return False

    except (OSError, TimeoutError) as e:
        logger.warning(
            "io_uring syscall probe failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return False


@_async_cached_probe("unshare")
async def probe_unshare_support() -> bool:
    """Probe for unshare (Linux namespace) support (cached).

    Tests if the current environment allows creating new namespaces via unshare.
    This requires either:
    - Root privileges
    - CAP_SYS_ADMIN capability
    - Unprivileged user namespaces enabled (/proc/sys/kernel/unprivileged_userns_clone=1)

    Returns:
        True if unshare works, False otherwise (skip namespace isolation)
    """
    # Skip on non-Linux - unshare is Linux-specific
    if detect_host_os() == HostOS.MACOS:
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
            return True

        stderr_text = stderr.decode().strip() if stderr else ""
        logger.warning(
            "unshare unavailable (namespace isolation disabled)",
            extra={"exit_code": proc.returncode, "stderr": stderr_text[:200]},
        )
        return False
    except (OSError, TimeoutError) as e:
        logger.warning(
            "unshare probe failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return False


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
        kvm_available = await check_kvm_available()
    if kvm_available:
        return AccelType.KVM
    if detect_host_os() == HostOS.MACOS:
        if hvf_available is None:
            hvf_available = await check_hvf_available()
        if hvf_available:
            return AccelType.HVF
    return AccelType.TCG

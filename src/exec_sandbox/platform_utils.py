"""Cross-platform OS detection and configuration utilities.

Uses psutil's built-in OS detection constants for robust platform identification.
Provides PID-reuse safe process management wrappers.
"""

import asyncio
import contextlib
import platform
from enum import Enum, auto
from functools import cache
from typing import Literal

import psutil
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential


class HostOS(Enum):
    """Supported host operating systems."""

    LINUX = auto()
    """Linux (production environment with KVM/containers)."""

    MACOS = auto()
    """macOS (development environment with HVF)."""

    UNKNOWN = auto()
    """Unsupported or unrecognized OS."""


@cache
def detect_host_os() -> HostOS:
    """Detect current host operating system using psutil constants.

    Returns:
        HostOS enum indicating current platform

    Example:
        >>> from exec_sandbox.platform_utils import detect_host_os, HostOS
        >>> os_type = detect_host_os()
        >>> match os_type:
        ...     case HostOS.LINUX:
        ...         # Use KVM, iothread, mem-prealloc
        ...         pass
        ...     case HostOS.MACOS:
        ...         # Use HVF, no iothread, no mem-prealloc
        ...         pass
        ...     case HostOS.UNKNOWN:
        ...         raise RuntimeError("Unsupported OS")
    """
    if psutil.LINUX:
        return HostOS.LINUX
    if psutil.MACOS:
        return HostOS.MACOS
    return HostOS.UNKNOWN


class HostArch(Enum):
    """Supported host CPU architectures."""

    X86_64 = auto()
    """x86_64/amd64 architecture."""

    AARCH64 = auto()
    """ARM64/aarch64 architecture."""

    UNKNOWN = auto()
    """Unsupported or unrecognized architecture."""


@cache
def detect_host_arch() -> HostArch:
    """Detect current host CPU architecture.

    Returns:
        HostArch enum indicating current architecture

    Example:
        >>> from exec_sandbox.platform_utils import detect_host_arch, HostArch
        >>> arch = detect_host_arch()
        >>> match arch:
        ...     case HostArch.X86_64:
        ...         qemu_bin = "qemu-system-x86_64"
        ...     case HostArch.AARCH64:
        ...         qemu_bin = "qemu-system-aarch64"
        ...     case HostArch.UNKNOWN:
        ...         raise RuntimeError("Unsupported architecture")
    """
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return HostArch.X86_64
    if machine in ("arm64", "aarch64"):
        return HostArch.AARCH64
    return HostArch.UNKNOWN


def get_os_name() -> str:
    """Get OS name string for binary naming conventions.

    Returns:
        "darwin" for macOS, "linux" for Linux

    Raises:
        ValueError: If running on unsupported OS
    """
    match detect_host_os():
        case HostOS.MACOS:
            return "darwin"
        case HostOS.LINUX:
            return "linux"
        case _:
            raise ValueError("Unsupported OS")


def get_arch_name(convention: Literal["kernel", "go"] = "kernel") -> str:
    """Get architecture string for naming conventions.

    Args:
        convention: "kernel" for x86_64/aarch64 (Linux kernel style),
                   "go" for amd64/arm64 (Go toolchain style)

    Returns:
        Architecture string in requested convention

    Raises:
        ValueError: If running on unsupported architecture
    """
    arch = detect_host_arch()
    match (convention, arch):
        case ("go", HostArch.X86_64):
            return "amd64"
        case ("go", HostArch.AARCH64):
            return "arm64"
        case ("kernel", HostArch.X86_64):
            return "x86_64"
        case ("kernel", HostArch.AARCH64):
            return "aarch64"
        case _:
            raise ValueError(f"Unsupported architecture: {arch}")


# Transient psutil exceptions that may occur during process inspection
# Used for retry logic and error handling in ProcessWrapper
_PSUTIL_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    psutil.NoSuchProcess,
    psutil.AccessDenied,
)

# Retry configuration for psutil operations
# Exponential backoff: 1ms, 2ms, 4ms, 8ms, 16ms, 32ms, 64ms (7 attempts, ~127ms total)
_PSUTIL_RETRY_ATTEMPTS = 7
_PSUTIL_RETRY_MULTIPLIER = 0.001
_PSUTIL_RETRY_MIN = 0.001
_PSUTIL_RETRY_MAX = 0.064


class ProcessWrapper:
    """PID-reuse safe process wrapper using psutil.

    Wraps asyncio.subprocess.Process with psutil.Process for safer PID monitoring.
    Protects against PID reuse edge cases where OS recycles PIDs.
    """

    def __init__(self, async_proc: asyncio.subprocess.Process) -> None:
        """Wrap asyncio process with psutil for PID-safe monitoring.

        Args:
            async_proc: asyncio subprocess.Process instance
        """
        self.async_proc = async_proc
        self.psutil_proc: psutil.Process | None = None

        # Wrap with psutil for PID-reuse safe monitoring
        # Use exponential backoff to handle race conditions where the process
        # may not be immediately visible to psutil (especially on free-threaded
        # Python where scheduling differs). Retries: 1ms, 2ms, 4ms, 8ms, 16ms, 32ms, 64ms
        if async_proc.pid:
            try:
                for attempt in Retrying(
                    stop=stop_after_attempt(_PSUTIL_RETRY_ATTEMPTS),
                    wait=wait_exponential(
                        multiplier=_PSUTIL_RETRY_MULTIPLIER, min=_PSUTIL_RETRY_MIN, max=_PSUTIL_RETRY_MAX
                    ),
                    retry=retry_if_exception_type(_PSUTIL_TRANSIENT_ERRORS),
                    reraise=True,
                ):
                    with attempt:
                        self.psutil_proc = psutil.Process(async_proc.pid)
            except _PSUTIL_TRANSIENT_ERRORS:
                # Process not visible after retries - fallback to returncode-based checks
                pass

    async def is_running(self) -> bool:
        """Check if process exists and hasn't terminated (PID-reuse safe).

        Returns True for processes in ANY non-terminated state, including:
        - Running (normal execution)
        - Sleeping (waiting for I/O)
        - Stopped (SIGSTOP/SIGTSTP - frozen but alive)

        Use is_stopped() to specifically detect frozen processes that won't
        respond to communication.

        Process state guide:
        - Running/Sleeping: is_running()=True,  is_stopped()=False → can communicate
        - Stopped (SIGSTOP): is_running()=True,  is_stopped()=True  → cannot communicate
        - Terminated:        is_running()=False, is_stopped()=False → process gone

        Returns:
            True if process exists (even if stopped), False if terminated
        """
        if not self.psutil_proc:
            return self.async_proc.returncode is None

        def _check() -> bool:
            for attempt in Retrying(
                stop=stop_after_attempt(_PSUTIL_RETRY_ATTEMPTS),
                wait=wait_exponential(
                    multiplier=_PSUTIL_RETRY_MULTIPLIER, min=_PSUTIL_RETRY_MIN, max=_PSUTIL_RETRY_MAX
                ),
                retry=retry_if_exception_type(_PSUTIL_TRANSIENT_ERRORS),
                reraise=True,
            ):
                with attempt:
                    return self.psutil_proc.is_running()  # type: ignore[union-attr]
            return False

        try:
            return await asyncio.to_thread(_check)
        except _PSUTIL_TRANSIENT_ERRORS:
            return False

    async def is_stopped(self) -> bool:
        """Check if process is stopped/frozen (SIGSTOP/SIGTSTP).

        Stopped processes are alive but won't respond to communication.
        Use this to detect frozen VMs before attempting health checks.

        Process state guide:
        - Running/Sleeping: is_running()=True,  is_stopped()=False → can communicate
        - Stopped (SIGSTOP): is_running()=True,  is_stopped()=True  → cannot communicate
        - Terminated:        is_running()=False, is_stopped()=False → process gone

        Returns:
            True if process is in stopped state, False otherwise
        """
        if not self.psutil_proc:
            return False

        def _check() -> bool:
            for attempt in Retrying(
                stop=stop_after_attempt(_PSUTIL_RETRY_ATTEMPTS),
                wait=wait_exponential(
                    multiplier=_PSUTIL_RETRY_MULTIPLIER, min=_PSUTIL_RETRY_MIN, max=_PSUTIL_RETRY_MAX
                ),
                retry=retry_if_exception_type(_PSUTIL_TRANSIENT_ERRORS),
                reraise=True,
            ):
                with attempt:
                    return self.psutil_proc.status() == psutil.STATUS_STOPPED  # type: ignore[union-attr]
            return False

        try:
            return await asyncio.to_thread(_check)
        except _PSUTIL_TRANSIENT_ERRORS:
            return False

    @property
    def pid(self) -> int | None:
        """Process ID."""
        return self.async_proc.pid

    @property
    def returncode(self) -> int | None:
        """Process return code (None if still running)."""
        return self.async_proc.returncode

    async def wait(self) -> int:
        """Wait for process to complete.

        Returns:
            Process exit code
        """
        return await self.async_proc.wait()

    @property
    def stdout(self):
        """Process stdout stream."""
        return self.async_proc.stdout

    @property
    def stderr(self):
        """Process stderr stream."""
        return self.async_proc.stderr

    async def terminate(self) -> None:
        """Terminate process (SIGTERM) - async, non-blocking.

        Async version prevents blocking event loop on system/kernel hangs.
        Uses asyncio.to_thread() for blocking psutil operations.

        Raises:
            ProcessLookupError: If process is not running
        """
        if not await self.is_running():
            raise ProcessLookupError(f"Process {self.pid} is not running")

        if self.psutil_proc:
            # Use psutil (suppress errors for race conditions where process dies during call)
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                await asyncio.to_thread(self.psutil_proc.terminate)
        else:
            self.async_proc.terminate()

    async def kill(self) -> None:
        """Kill process (SIGKILL) - async, non-blocking.

        Async version prevents blocking event loop on system/kernel hangs.
        Uses asyncio.to_thread() for blocking psutil operations.

        Raises:
            ProcessLookupError: If process is not running
        """
        if not await self.is_running():
            raise ProcessLookupError(f"Process {self.pid} is not running")

        if self.psutil_proc:
            # Use psutil (suppress errors for race conditions where process dies during call)
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                await asyncio.to_thread(self.psutil_proc.kill)
        else:
            self.async_proc.kill()

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        """Wait for process to terminate and return stdout/stderr.

        Args:
            input: Data to send to stdin

        Returns:
            Tuple of (stdout, stderr) bytes
        """
        return await self.async_proc.communicate(input)

    async def wait_with_timeout(self, timeout: float) -> int:
        """Wait for process with timeout, handling pipe draining automatically.

        Prevents pipe buffer deadlock by draining stdout/stderr if pipes exist.
        Handles both scenarios:
        - Pipes exist with no background reader: use communicate() to drain
        - Pipes exist with background reader: use wait() (pipes drained by reader)
        - No pipes: use wait()

        Best practice pattern from Nov 2025 asyncio docs for subprocess cleanup.

        Args:
            timeout: Timeout in seconds

        Returns:
            Process exit code

        Raises:
            asyncio.TimeoutError: If process doesn't exit within timeout
        """
        has_pipes = self.stdout is not None or self.stderr is not None

        if has_pipes:
            try:
                # Drain stdout/stderr pipes to prevent deadlock
                await asyncio.wait_for(self.async_proc.communicate(), timeout=timeout)
            except RuntimeError:
                # Another coroutine already reading streams (e.g., background logger)
                # Pipes being drained by other reader, so just wait for exit
                await asyncio.wait_for(self.wait(), timeout=timeout)
        else:
            # No pipes - just wait for process exit
            await asyncio.wait_for(self.wait(), timeout=timeout)

        return self.returncode  # type: ignore[return-value]

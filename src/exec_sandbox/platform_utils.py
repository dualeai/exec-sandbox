"""Cross-platform OS detection and configuration utilities.

Uses psutil's built-in OS detection constants for robust platform identification.
Provides PID-reuse safe process management wrappers.
"""

import asyncio
import contextlib
from enum import Enum, auto
from functools import cache

import psutil


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
        if async_proc.pid:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                # Process already died or inaccessible
                self.psutil_proc = psutil.Process(async_proc.pid)

    async def is_running(self) -> bool:
        """Check if process is still running (PID-reuse safe).

        Async version prevents blocking event loop on system/kernel hangs.
        Uses asyncio.to_thread() to run blocking psutil call in thread pool.

        Returns:
            True if process is running, False otherwise
        """
        if not self.psutil_proc:
            return self.async_proc.returncode is None

        # Run blocking psutil call in thread pool (Python 3.9+)
        # Protects against system/kernel hangs blocking event loop
        try:
            return await asyncio.to_thread(self.psutil_proc.is_running)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
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
        """
        if self.psutil_proc and await self.is_running():
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                await asyncio.to_thread(self.psutil_proc.terminate)
        else:
            self.async_proc.terminate()

    async def kill(self) -> None:
        """Kill process (SIGKILL) - async, non-blocking.

        Async version prevents blocking event loop on system/kernel hangs.
        Uses asyncio.to_thread() for blocking psutil operations.
        """
        if self.psutil_proc and await self.is_running():
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

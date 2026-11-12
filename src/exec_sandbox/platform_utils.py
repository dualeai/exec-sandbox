"""Cross-platform OS detection and configuration utilities.

Uses psutil's built-in OS detection constants for robust platform identification.
Provides PID-reuse safe process management wrappers.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.util
import os
import platform
import pwd
import subprocess
from enum import Enum, StrEnum, auto
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Final

import psutil
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

_CHILD_ENV_BLACKLIST: Final = (
    "EXEC_SANDBOX", "KEY", "SECRET", "TOKEN", "PASS", "PWD", "CRED", "AUTH", "URL", "PROXY", "DSN", "CONN", "HEADER",
)  # fmt: skip


def child_env() -> dict[str, str]:
    """Library env minus every EXEC_SANDBOX_* var and every var whose name suggests a secret,
    URL or proxy; no child needs one."""
    return {k: v for k, v in os.environ.items() if not any(word in k.upper() for word in _CHILD_ENV_BLACKLIST)}


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
        ...         # Use KVM, ioeventfd, seccomp sandbox
        ...         pass
        ...     case HostOS.MACOS:
        ...         # Use HVF (no ioeventfd, no seccomp)
        ...         pass
        ...     case HostOS.UNKNOWN:
        ...         raise RuntimeError("Unsupported OS")
    """
    if psutil.LINUX:
        return HostOS.LINUX
    if psutil.MACOS:
        return HostOS.MACOS
    return HostOS.UNKNOWN


if TYPE_CHECKING:
    from collections.abc import Mapping


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


@cache
def detect_host_x86_vendor() -> str:
    """CPUID vendor of the host CPU ("GenuineIntel" or "AuthenticAMD").

    x86 guests get the host vendor (KVM runs best with matching vendors). A
    non-x86 host runs x86 guests under TCG and gets qemu64's own vendor.
    """
    if detect_host_arch() != HostArch.X86_64:
        return "AuthenticAMD"
    if detect_host_os() == HostOS.MACOS:
        return subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.vendor"],
            capture_output=True,
            text=True,
            check=True,
            env=child_env(),
        ).stdout.strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("vendor_id"):
            return line.split(":", 1)[1].strip()
    return "AuthenticAMD"


def kernel_image_path(kernel_dir: Path, arch: HostArch) -> Path:
    """Return the kernel image QEMU boots for this architecture.

    x86_64 prefers the uncompressed vmlinux (PVH direct boot, ~50ms faster) when
    present; other cases use vmlinuz. Shared by the QEMU command builder and the
    L1 cache key so the key always fingerprints the kernel actually booted.
    """
    arch_suffix = "aarch64" if arch == HostArch.AARCH64 else "x86_64"
    vmlinux_path = kernel_dir / f"vmlinux-{arch_suffix}"
    if arch == HostArch.X86_64 and vmlinux_path.exists():
        return vmlinux_path
    return kernel_dir / f"vmlinuz-{arch_suffix}"


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


def get_arch_name() -> str:
    """Get the host architecture in the Linux kernel's names.

    Returns:
        "x86_64" or "aarch64"

    Raises:
        ValueError: If running on unsupported architecture
    """
    arch = detect_host_arch()
    match arch:
        case HostArch.X86_64:
            return "x86_64"
        case HostArch.AARCH64:
            return "aarch64"
        case _:
            raise ValueError(f"Unsupported architecture: {arch}")


class UserDir(StrEnum):
    """Kind of per-user folder. The cost of lost files decides where each kind goes."""

    CACHE = "cache"
    """Files the library can make again (images, snapshots). macOS may delete files in
    ~/Library/Caches when the disk is nearly full, and users clear ~/.cache freely."""

    DATA = "data"
    """Files the library cannot make again (the "local" instance store)."""


# kind -> (override env var, folder under the macOS home, XDG env var, folder under the home elsewhere)
_USER_DIR_RULES: Final[dict[UserDir, tuple[str | None, tuple[str, ...], str, tuple[str, ...]]]] = {
    UserDir.CACHE: ("EXEC_SANDBOX_CACHE_DIR", ("Library", "Caches"), "XDG_CACHE_HOME", (".cache",)),
    UserDir.DATA: (None, ("Library", "Application Support"), "XDG_DATA_HOME", (".local", "share")),
}


def get_user_dir(*, kind: UserDir, app_name: str) -> Path:
    """Get the per-user folder of this kind for the app.

    Returns:
        - macOS: ~/Library/Caches/<app_name>/ or ~/Library/Application Support/<app_name>/
        - Linux: $XDG_CACHE_HOME/<app_name>/ or $XDG_DATA_HOME/<app_name>/ when the value is an
          absolute path; else ~/.cache/<app_name>/ or ~/.local/share/<app_name>/
        - Other: ~/.cache/<app_name>/ or ~/.local/share/<app_name>/

    "~" is Path.home() (HOME when it is set). Override CACHE with the EXEC_SANDBOX_CACHE_DIR
    environment variable. DATA has no override here: callers take theirs from settings
    (instance_store_dir). For a path that ignores the environment, use get_account_dir.
    """
    return _user_dir(kind=kind, app_name=app_name, home=Path.home(), env=os.environ)


def get_account_dir(*, kind: UserDir, app_name: str) -> Path:
    """The default per-user folder of this kind for the effective user: the home from the user
    database, and no environment override (HOME, XDG_*, EXEC_SANDBOX_CACHE_DIR are ignored).

    Every process of one account gets the same path, whatever its environment (an IDE task,
    a shell, another HOME): use it for a lock those processes must share. A process under
    `sudo` (also `sudo -E`) runs as root and gets root's folder.
    """
    return _user_dir(kind=kind, app_name=app_name, home=Path(pwd.getpwuid(os.geteuid()).pw_dir), env={})


def _user_dir(*, kind: UserDir, app_name: str, home: Path, env: Mapping[str, str]) -> Path:
    override_env, macos_parts, xdg_env, home_parts = _USER_DIR_RULES[kind]
    if override_env and (env_path := env.get(override_env)):
        return Path(env_path)

    match detect_host_os():
        case HostOS.MACOS:
            return home.joinpath(*macos_parts, app_name)
        case HostOS.LINUX:
            # XDG Base Directory spec: an empty or relative value is invalid and is ignored
            xdg_home = env.get(xdg_env, "")
            if Path(xdg_home).is_absolute():
                return Path(xdg_home) / app_name
            return home.joinpath(*home_parts, app_name)
        case _:
            # Fallback for other platforms
            return home.joinpath(*home_parts, app_name)


# ---------------------------------------------------------------------------
# Page-cache advisory prefetch
# ---------------------------------------------------------------------------
#
# Asks the kernel to asynchronously read a file into the page cache (zero-copy,
# returns near-instantly).  Used before concurrent QEMU restores so that every
# process hits RAM instead of storage.
#
# macOS: fcntl(fd, F_RDADVISE=44, struct radvisory{off_t, int})
#        — "Issue an advisory read async with no copy to user"
#        — F_RDADVISE is not exposed by Python's fcntl module (cpython#113092),
#          so we call libc directly via ctypes.
# Linux: os.posix_fadvise(fd, 0, size, POSIX_FADV_WILLNEED)
#        — Recent kernels (≥3.x) perform truly async readahead.
#
# References:
#   darwin-xnu  bsd/sys/fcntl.h  (F_RDADVISE = 44, struct radvisory)
#   PostgreSQL  commit 2024-08-27 "Add prefetching support on macOS"

# macOS ctypes plumbing (module-level so the CDLL is loaded once)
_F_RDADVISE = 44  # from darwin-xnu/bsd/sys/fcntl.h


class _Radvisory(ctypes.Structure):
    """macOS struct radvisory { off_t ra_offset; int ra_count; }."""

    _fields_ = [
        ("ra_offset", ctypes.c_int64),  # off_t
        ("ra_count", ctypes.c_int),  # int
    ]


_libc_path = ctypes.util.find_library("c")
_libc = ctypes.CDLL(_libc_path) if _libc_path else None


def advise_willneed(path: Path) -> None:
    """Ask the kernel to prefetch *path* into the page cache.

    Non-blocking, zero-copy.  The kernel schedules background I/O and the call
    returns near-instantly; actual disk reads happen asynchronously.

    Best-effort: silently ignored on unsupported platforms or on failure (the
    file is advisory, not required — QEMU will simply read from disk).
    """
    fd = -1
    try:
        fd = os.open(str(path), os.O_RDONLY)
        size = os.fstat(fd).st_size
        host = detect_host_os()
        if host is HostOS.MACOS and _libc is not None:
            advisory = _Radvisory(ra_offset=0, ra_count=min(size, 2**31 - 1))
            _libc.fcntl(fd, _F_RDADVISE, ctypes.byref(advisory))
        elif host is HostOS.LINUX:
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_WILLNEED)  # type: ignore[attr-defined]
    except OSError:
        pass  # Advisory — silently degrade
    finally:
        if fd >= 0:
            os.close(fd)


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
        # The background task that reads stdout/stderr, set by subprocess_utils.start_drain:
        # while it runs, wait_with_timeout never reads the pipes.
        self.drain_task: asyncio.Task[None] | None = None

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
    def stdout(self) -> asyncio.StreamReader | None:
        """Process stdout stream."""
        return self.async_proc.stdout

    @property
    def stderr(self) -> asyncio.StreamReader | None:
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
        Handles three cases:
        - Pipes read by a running background task (drain_task): use wait() only.
          communicate() would read the same streams; when the two readers
          interleave, one loses its wakeup, and communicate() waits for the full
          timeout although the process has exited. A drain task that ended (a
          line over the stream limit ends it) reads nothing more: communicate()
          drains then, or a child that still writes blocks on a full pipe.
        - Pipes exist with no background reader: use communicate() to drain
          (wait() if another coroutine already reads them: RuntimeError)
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

        drained = self.drain_task is not None and not self.drain_task.done()
        if has_pipes and not drained:
            try:
                # Drain stdout/stderr pipes to prevent deadlock
                await asyncio.wait_for(self.async_proc.communicate(), timeout=timeout)
            except RuntimeError:
                # Another coroutine already reading streams (e.g., background logger)
                # Pipes being drained by other reader, so just wait for exit
                await asyncio.wait_for(self.wait(), timeout=timeout)
        else:
            # No pipes, or the drain task reads them - just wait for process exit
            await asyncio.wait_for(self.wait(), timeout=timeout)

        return self.returncode  # type: ignore[return-value]

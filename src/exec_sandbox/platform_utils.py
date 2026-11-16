"""Cross-platform OS detection and configuration utilities.

- Host OS (psutil constants), CPU architecture and x86 vendor, and their names
- child_env: the environment every child process gets
- Per-user folders (get_user_dir, get_account_dir)
- ProcessWrapper: PID-reuse safe process management
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import pwd
import subprocess
from enum import Enum, StrEnum, auto
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Final

import psutil

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


def detect_host_os() -> HostOS:
    """Detect current host operating system using psutil constants.

    Returns:
        HostOS enum indicating current platform

    Example:
        >>> from exec_sandbox.platform_utils import detect_host_os, HostOS
        >>> os_type = detect_host_os()
        >>> match os_type:
        ...     case HostOS.LINUX:
        ...         # Use KVM and QEMU's seccomp sandbox
        ...         pass
        ...     case HostOS.MACOS:
        ...         # Use HVF (no seccomp)
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
    """CPUID vendor of the host CPU: its 12 characters, spaces kept ("GenuineIntel",
    "AuthenticAMD", Zhaoxin's "  Shanghai  "); QEMU refuses a vendor of another length.

    x86 guests get the host vendor (KVM runs best with matching vendors). A
    non-x86 host runs x86 guests under TCG and gets qemu64's own vendor.
    """
    if detect_host_arch() != HostArch.X86_64:
        return "AuthenticAMD"
    if detect_host_os() == HostOS.MACOS:
        return subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.vendor"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            env=child_env(),
        ).stdout.removesuffix("\n")
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("vendor_id"):
            # The kernel prints "vendor_id\t: <the 12 bytes>"
            return line.split(": ", 1)[1]
    return "AuthenticAMD"


def get_os_name() -> str:
    """The host OS as binary names spell it: "darwin" for macOS, "linux" for Linux (the import
    gate of exec_sandbox refuses any other OS)."""
    return "darwin" if detect_host_os() is HostOS.MACOS else "linux"


def get_arch_name() -> str:
    """The host architecture in the Linux kernel's names (arch_name of detect_host_arch).

    Raises:
        ValueError: If running on unsupported architecture
    """
    return arch_name(detect_host_arch())


def arch_of(name: str) -> HostArch:
    """The HostArch of a name that arch_name gives ("x86_64", "aarch64")."""
    return {"x86_64": HostArch.X86_64, "aarch64": HostArch.AARCH64}.get(name, HostArch.UNKNOWN)


def arch_name(arch: HostArch) -> str:
    """`arch` in the Linux kernel's names, as image and binary names spell it: "x86_64" or "aarch64".

    Raises:
        ValueError: UNKNOWN
    """
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


_APP_FOLDER: Final = "exec-sandbox"


def get_user_dir(*, kind: UserDir) -> Path:
    """Get the per-user folder of this kind for exec-sandbox.

    Returns:
        - macOS: ~/Library/Caches/exec-sandbox/ or ~/Library/Application Support/exec-sandbox/
        - Linux: $XDG_CACHE_HOME/exec-sandbox/ or $XDG_DATA_HOME/exec-sandbox/ when the value is
          an absolute path; else ~/.cache/exec-sandbox/ or ~/.local/share/exec-sandbox/

    "~" is Path.home() (HOME when it is set). Override CACHE with the EXEC_SANDBOX_CACHE_DIR
    environment variable. DATA has no override here: callers take theirs from settings
    (instance_store_dir). For a path that ignores the environment, use get_account_dir.
    """
    return _user_dir(kind=kind, home=Path.home(), env=os.environ)


def get_account_dir(*, kind: UserDir) -> Path:
    """The default per-user folder of this kind for the effective user: the home from the user
    database, and no environment override (HOME, XDG_*, EXEC_SANDBOX_CACHE_DIR are ignored).

    Every process of one account gets the same path, whatever its environment (an IDE task,
    a shell, another HOME): use it for a lock those processes must share. A process under
    `sudo` (also `sudo -E`) runs as root and gets root's folder.
    """
    return _user_dir(kind=kind, home=Path(pwd.getpwuid(os.geteuid()).pw_dir), env={})


def _user_dir(*, kind: UserDir, home: Path, env: Mapping[str, str]) -> Path:
    override_env, macos_parts, xdg_env, home_parts = _USER_DIR_RULES[kind]
    if override_env and (env_path := env.get(override_env)):
        return Path(env_path)

    if detect_host_os() is HostOS.MACOS:
        return home.joinpath(*macos_parts, _APP_FOLDER)
    # Linux (the import gate refuses any other OS). XDG Base Directory spec: an empty or
    # relative value is invalid and is ignored
    xdg_home = env.get(xdg_env, "")
    if Path(xdg_home).is_absolute():
        return Path(xdg_home) / _APP_FOLDER
    return home.joinpath(*home_parts, _APP_FOLDER)


# psutil errors that mean "this process is gone or not ours to inspect". None of them
# clears with time: a reaped pid never comes back (a retry could only find another
# process on a reused pid), and permissions do not change within milliseconds.
_PSUTIL_GONE_ERRORS: tuple[type[Exception], ...] = (
    psutil.NoSuchProcess,
    psutil.AccessDenied,
)


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

        # Wrap with psutil for PID-reuse safe monitoring. One attempt: the kernel lists a
        # child as soon as fork() returns. NoSuchProcess means the asyncio child watcher
        # already reaped a short process; returncode-based checks then apply.
        if async_proc.returncode is None:
            with contextlib.suppress(*_PSUTIL_GONE_ERRORS):
                self.psutil_proc = psutil.Process(async_proc.pid)

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
            True while the process exists, stopped or not (a child that exited and is not
            reaped yet too: psutil counts a zombie as running). False once it ended, and also
            when this user can no longer see or read it (psutil NoSuchProcess or
            AccessDenied, for example /proc mounted with hidepid after the child exec'd
            sudo): terminate and kill then raise ProcessLookupError, and cleanup_process
            counts that as an end.
        """
        if not self.psutil_proc:
            return self.async_proc.returncode is None

        try:
            return await asyncio.to_thread(self.psutil_proc.is_running)
        except _PSUTIL_GONE_ERRORS:
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

        proc = self.psutil_proc
        try:
            return await asyncio.to_thread(lambda: proc.status() == psutil.STATUS_STOPPED)
        except _PSUTIL_GONE_ERRORS:
            return False

    @property
    def pid(self) -> int:
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

    async def communicate(self) -> tuple[bytes, bytes]:
        """Wait for process to terminate and return stdout/stderr.

        Returns:
            Tuple of (stdout, stderr) bytes
        """
        return await self.async_proc.communicate()

    async def wait_with_timeout(self, timeout: float) -> int:
        """Wait for process with timeout, handling pipe draining automatically.

        Prevents pipe buffer deadlock by draining stdout/stderr if pipes exist.
        Handles three cases:
        - Pipes read by a running background task (drain_task): use wait() only.
          communicate() would read the same streams; when the two readers
          interleave, one loses its wakeup, and communicate() waits for the full
          timeout although the process has exited. A drain task that ended (an
          error of its handler) reads nothing more: communicate() drains then, or
          a child that still writes blocks on a full pipe.
        - Pipes exist with no background reader: use communicate() to drain
          (wait() if another coroutine already reads them: RuntimeError)
        - No pipes: use wait()

        Args:
            timeout: Timeout in seconds

        Returns:
            Process exit code

        Raises:
            TimeoutError: If process doesn't exit within timeout
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

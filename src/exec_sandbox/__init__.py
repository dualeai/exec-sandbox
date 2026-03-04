"""exec-sandbox: Secure code execution in microVMs.

A standalone Python library for executing untrusted code in isolated QEMU microVMs.

Quick Start (single execution):
    ```python
    from exec_sandbox import Scheduler

    async with Scheduler() as scheduler:
        result = await scheduler.run(
            code="print('hello')",
            language="python",
        )
        print(result.stdout)  # "hello\\n"
    ```

Session (stateful multi-step execution):
    ```python
    from exec_sandbox import Scheduler

    async with Scheduler() as scheduler:
        async with await scheduler.session(language="python") as session:
            await session.exec("x = 42")
            result = await session.exec("print(x)")
            print(result.stdout)  # "42\\n"
    ```

With Configuration:
    ```python
    from exec_sandbox import Scheduler, SchedulerConfig

    config = SchedulerConfig(
        s3_bucket="my-snapshots",  # Enable S3 cache
    )
    async with Scheduler(config) as scheduler:
        result = await scheduler.run(
            code="import pandas; print(pandas.__version__)",
            language="python",
            packages=["pandas==2.2.0"],
        )
    ```

Requirements:
    - QEMU 8.0+ with KVM (Linux) or HVF (macOS) acceleration
    - VM images from GitHub Releases
    - Python 3.12+

For S3 snapshot caching:
    pip install exec-sandbox[s3]
"""

# ---------------------------------------------------------------------------
# Platform gate — fail fast on unsupported operating systems
# ---------------------------------------------------------------------------
# exec-sandbox is deeply POSIX-specific: Unix domain sockets (AF_UNIX),
# signals (SIGKILL/SIGTERM/SIGSTOP), cgroups v2, /proc, fcntl file locking,
# termios, os.getuid(), unshare namespaces, and QEMU with KVM (Linux) or
# HVF (macOS) acceleration.  None of these are available on Windows.
#
# We detect the platform via psutil (see platform_utils.detect_host_os) and
# raise PermanentError — a SandboxError subclass — so that cross-platform
# orchestrators can handle the rejection gracefully:
#
#     from exec_sandbox.exceptions import PermanentError
#
#     try:
#         from exec_sandbox import Scheduler
#     except PermanentError:
#         # Fall back to another execution backend
#         pass
#
# The exceptions module is pure Python with no POSIX dependencies, so it
# imports successfully on any platform.
# ---------------------------------------------------------------------------
from exec_sandbox.platform_utils import HostOS, detect_host_os

if detect_host_os() == HostOS.UNKNOWN:
    import platform

    from exec_sandbox.exceptions import PermanentError

    raise PermanentError(
        "exec-sandbox requires Linux or macOS. "
        f"Detected platform '{platform.system()}' is not supported because "
        "exec-sandbox relies on QEMU with KVM/HVF acceleration, Unix domain sockets, "
        "and POSIX process management (signals, cgroups, overlayfs). "
        "Consider running inside WSL2 on Windows.",
        context={"platform": platform.system()},
    )

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("exec-sandbox")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from exec_sandbox._logging import configure_logging
from exec_sandbox.config import SchedulerConfig
from exec_sandbox.exceptions import (
    AssetChecksumError,
    AssetDownloadError,
    AssetError,
    AssetNotFoundError,
    BalloonTransientError,
    CodeValidationError,
    CommunicationError,
    EnvVarValidationError,
    GuestAgentError,
    InputValidationError,
    OutputLimitError,
    PackageNotAllowedError,
    PermanentError,
    SandboxDependencyError,
    SandboxError,
    SessionClosedError,
    SnapshotError,
    SocketAuthError,
    TransientError,
    VmBootError,
    VmBootTimeoutError,
    VmCapacityError,
    VmConfigError,
    VmDependencyError,
    VmError,
    VmGvproxyError,
    VmOverlayError,
    VmPermanentError,
    VmQemuCrashError,
    VmTimeoutError,
    VmTransientError,
)
from exec_sandbox.models import ExecutionResult, ExposedPort, FileInfo, Language, PortMapping, TimingBreakdown
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.session import Session

__all__ = [
    "AssetChecksumError",
    "AssetDownloadError",
    "AssetError",
    "AssetNotFoundError",
    "BalloonTransientError",
    "CodeValidationError",
    "CommunicationError",
    "EnvVarValidationError",
    "ExecutionResult",
    "ExposedPort",
    "FileInfo",
    "GuestAgentError",
    "InputValidationError",
    "Language",
    "OutputLimitError",
    "PackageNotAllowedError",
    "PermanentError",
    "PortMapping",
    "SandboxDependencyError",
    "SandboxError",
    "Scheduler",
    "SchedulerConfig",
    "Session",
    "SessionClosedError",
    "SnapshotError",
    "SocketAuthError",
    "TimingBreakdown",
    "TransientError",
    "VmBootError",
    "VmBootTimeoutError",
    "VmCapacityError",
    "VmConfigError",
    "VmDependencyError",
    "VmError",
    "VmGvproxyError",
    "VmOverlayError",
    "VmPermanentError",
    "VmQemuCrashError",
    "VmTimeoutError",
    "VmTransientError",
    "configure_logging",
]

"""exec-sandbox: run untrusted code in QEMU microVMs, from Python or the sbx command line.

Quick Start:
    ```python
    from exec_sandbox import Scheduler, VmSpec

    async with Scheduler() as scheduler:
        async with await scheduler.new(VmSpec(language="python")) as session:
            await session.exec("x = 42")
            result = await session.exec("print(x)")  # state stays across exec() calls
            print(result.stdout)  # "42\\n"
    ```

VmSpec says what a VM is made of (defaults included); keep a catalog of specs:
    ```python
    from exec_sandbox import Scheduler, SchedulerConfig, VmSpec

    DATA = VmSpec(language="python", memory_mb=1024, packages=("pandas==3.0.1",))
    async with Scheduler(SchedulerConfig(s3_bucket="my-snapshots")) as scheduler:
        async with await scheduler.new(DATA) as session:
            result = await session.exec("import pandas; print(pandas.__version__)")
    ```

Saved instances: Session.save(), Scheduler.reuse() (with SchedulerConfig.instance_store).
Errors: exec_sandbox.exceptions (one error class per action).

Requirements:
    - QEMU 8.0 or later; QEMU 11.0 or later for L1 memory snapshots and saved
      instances; QEMU 9.0.4 or later on ARM64 without KVM or HVF
    - KVM (Linux) or HVF (macOS) recommended: without it, QEMU uses software
      emulation (TCG), about 5-8x slower
    - VM images from GitHub Releases
    - Python 3.12 or later

For the L3 snapshot cache (S3) and the "s3" instance store:
    pip install 'exec-sandbox[s3]'
For the "local" instance store (and the sbx command line):
    pip install 'exec-sandbox[cli]'
"""

# ---------------------------------------------------------------------------
# Platform gate — fail fast on unsupported operating systems
# ---------------------------------------------------------------------------
# exec-sandbox is deeply POSIX-specific: Unix domain sockets (AF_UNIX),
# signals (SIGKILL/SIGTERM/SIGSTOP), cgroups v2, /proc, fcntl file locking,
# os.getuid(), unshare namespaces, and QEMU with KVM (Linux) or
# HVF (macOS) acceleration.  None of these are available on Windows.
#
# We detect the platform via psutil (see platform_utils.detect_host_os) and
# raise ConfigError (code PLATFORM_UNSUPPORTED), a SandboxError, so that
# cross-platform orchestrators can handle the rejection gracefully. Python runs
# this file before any exec_sandbox module, so no import of the error classes
# can come before this check: catch Exception and test its code (ErrorCode is
# a str enum).
#
#     try:
#         from exec_sandbox import Scheduler
#     except Exception as e:
#         if getattr(e, "code", None) != "PLATFORM_UNSUPPORTED":
#             raise
#         ...  # fall back to another execution backend
# ---------------------------------------------------------------------------
from exec_sandbox.platform_utils import HostOS, detect_host_os

if detect_host_os() == HostOS.UNKNOWN:
    import platform

    from exec_sandbox.exceptions import ConfigError, ErrorCode

    raise ConfigError(
        f"exec-sandbox runs on Linux or macOS, not on '{platform.system()}' (it needs QEMU with "
        "KVM or HVF, Unix sockets and POSIX processes). On Windows, run it in WSL2.",
        code=ErrorCode.PLATFORM_UNSUPPORTED,
        context={"platform": platform.system()},
        hint=None,
        instance_id=None,
    )

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("exec-sandbox")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from exec_sandbox._logging import configure_logging
from exec_sandbox.config import SchedulerConfig
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorAction,
    ErrorCode,
    InstanceGoneError,
    InstanceNotSavedWarning,
    InternalError,
    InvalidInputError,
    OutcomeUnknownError,
    RouteError,
    SandboxError,
    SessionClosedError,
    TransientError,
)
from exec_sandbox.instance.models import HostFacts, InstanceInfo, RecentCommit
from exec_sandbox.models import (
    ExecutionResult,
    ExposedPort,
    FileInfo,
    InstanceRef,
    Language,
    PortMapping,
    StartupTiming,
    TimingBreakdown,
)
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.session import Session
from exec_sandbox.vm_spec import VmSpec

__all__ = [
    "ConfigError",
    "ErrorAction",
    "ErrorCode",
    "ExecutionResult",
    "ExposedPort",
    "FileInfo",
    "HostFacts",
    "InstanceGoneError",
    "InstanceInfo",
    "InstanceNotSavedWarning",
    "InstanceRef",
    "InternalError",
    "InvalidInputError",
    "Language",
    "OutcomeUnknownError",
    "PortMapping",
    "RecentCommit",
    "RouteError",
    "SandboxError",
    "Scheduler",
    "SchedulerConfig",
    "Session",
    "SessionClosedError",
    "StartupTiming",
    "TimingBreakdown",
    "TransientError",
    "VmSpec",
    "configure_logging",
]

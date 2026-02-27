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

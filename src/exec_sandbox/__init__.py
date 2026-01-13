"""exec-sandbox: Secure code execution in microVMs.

A standalone Python library for executing untrusted code in isolated QEMU microVMs
with 6-layer security architecture.

Quick Start:
    ```python
    from exec_sandbox import Scheduler

    async with Scheduler() as scheduler:
        result = await scheduler.run(
            code="print('hello')",
            language="python",
        )
        print(result.stdout)  # "hello\\n"
    ```

With Configuration:
    ```python
    from exec_sandbox import Scheduler, SchedulerConfig

    config = SchedulerConfig(
        max_concurrent_vms=5,
        s3_bucket="my-snapshots",  # Enable S3 cache
    )
    async with Scheduler(config) as scheduler:
        result = await scheduler.run(
            code="import pandas; print(pandas.__version__)",
            language="python",
            packages=["pandas==2.2.0"],
        )
    ```

Security Architecture (6 layers):
    1. KVM/HVF hardware virtualization (CPU ring -1 isolation)
    2. Unprivileged QEMU process (no root required)
    3. Seccomp syscall filtering
    4. cgroup v2 resource limits (memory, CPU, PIDs)
    5. Linux namespaces (PID, network, mount, UTS, IPC)
    6. MAC policies (AppArmor/SELinux when available)

Requirements:
    - QEMU 8.0+ with KVM (Linux) or HVF (macOS) acceleration
    - VM images from GitHub Releases
    - Python 3.12+

For S3 snapshot caching:
    pip install exec-sandbox[s3]
"""

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.exceptions import (
    CommunicationError,
    GuestAgentError,
    PackageNotAllowedError,
    SandboxDependencyError,
    SandboxError,
    SnapshotError,
    VmBootError,
    VmError,
    VmTimeoutError,
)
from exec_sandbox.models import ExecutionResult, Language, TimingBreakdown
from exec_sandbox.scheduler import Scheduler

__all__ = [
    "CommunicationError",
    "ExecutionResult",
    "GuestAgentError",
    "Language",
    "PackageNotAllowedError",
    "SandboxDependencyError",
    "SandboxError",
    "Scheduler",
    "SchedulerConfig",
    "SnapshotError",
    "TimingBreakdown",
    "VmBootError",
    "VmError",
    "VmTimeoutError",
]

__version__ = "0.1.0"

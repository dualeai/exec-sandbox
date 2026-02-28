"""Scheduler configuration for exec-sandbox.

SchedulerConfig provides all configuration options for the Scheduler,
including VM pool settings, resource limits, paths, and S3 backup.

Example:
    ```python
    from exec_sandbox import Scheduler, SchedulerConfig

    # Default configuration
    async with Scheduler() as scheduler:
        result = await scheduler.run(code="print('hello')", language="python")

    # Custom configuration
    config = SchedulerConfig(
        default_memory_mb=512,
        s3_bucket="my-snapshots",
    )
    async with Scheduler(config) as scheduler:
        result = await scheduler.run(code="...", language="python")
    ```
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - pydantic evaluates annotations at runtime

from pydantic import BaseModel, ConfigDict, Field

from exec_sandbox import constants
from exec_sandbox.platform_utils import get_cache_dir


class SchedulerConfig(BaseModel):
    """Configuration for Scheduler.

    All fields have sensible defaults for local development.
    Production deployments should tune resource overcommit ratios based on host resources.

    Attributes:
        warm_pool_size: Number of pre-booted VMs per language (python, javascript).
            0 disables warm pool. Default: 0 (cold boot only).
        default_memory_mb: Default guest VM memory in MB. Can be overridden per-run.
            Minimum: 128. No upper bound (limited by host resources). Default: 256.
        default_timeout_seconds: Default execution timeout in seconds.
            Can be overridden per-run. Range: 1-300. Default: 30.
        images_dir: Directory containing base VM images (qcow2, kernels).
            If None, auto-detects from (in priority order):
            - EXEC_SANDBOX_IMAGES_DIR env var
            - ./images/dist/ (local build)
            - ~/.cache/exec-sandbox/ (download cache)
        disk_snapshot_cache_dir: Local directory for L2 snapshot cache.
            Default: OS-specific cache dir / disk-snapshots
            (macOS: ~/Library/Caches/exec-sandbox/, Linux: ~/.cache/exec-sandbox/)
        memory_snapshot_cache_dir: Local directory for L1 memory snapshot cache.
            Default: OS-specific cache dir / memory-snapshots
            (macOS: ~/Library/Caches/exec-sandbox/, Linux: ~/.cache/exec-sandbox/)
        s3_bucket: S3 bucket name for snapshot backup (L3 cache).
            If None, S3 backup is disabled. Requires aioboto3 optional dependency.
        s3_region: AWS region for S3 bucket. Default: us-east-1.
        s3_prefix: Prefix for S3 keys. Default: "snapshots/".
        enable_package_validation: Validate packages against allowlist.
            Disable for testing only. Default: True.
        auto_download_assets: Automatically download VM images from GitHub
            Releases if not found locally. Uses cache directory for storage.
            Default: True.
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable after creation
        extra="forbid",  # Reject unknown fields
    )

    # VM pool
    warm_pool_size: int = Field(
        default=0,
        ge=0,
        description="Pre-booted VMs per language (0 disables warm pool)",
    )

    # Defaults for run()
    default_memory_mb: int = Field(
        default=256,
        ge=128,
        description="Default guest VM memory in MB",
    )
    default_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Default execution timeout in seconds",
    )
    session_idle_timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Idle timeout for sessions (auto-destroy after inactivity)",
    )

    # Paths
    images_dir: Path | None = Field(
        default=None,
        description="Directory containing VM images (auto-detect if None)",
    )
    disk_snapshot_cache_dir: Path = Field(
        default_factory=lambda: get_cache_dir() / "disk-snapshots",
        description="Local disk snapshot cache directory (L2 cache)",
    )
    memory_snapshot_cache_dir: Path = Field(
        default_factory=lambda: get_cache_dir() / "memory-snapshots",
        description="Local memory snapshot cache directory (L1 cache)",
    )

    # S3 snapshot backup (optional)
    s3_bucket: str | None = Field(
        default=None,
        description="S3 bucket for snapshot backup (None disables S3)",
    )
    s3_region: str = Field(
        default="us-east-1",
        description="AWS region for S3 bucket",
    )
    s3_prefix: str = Field(
        default="snapshots/",
        description="Prefix for S3 keys",
    )
    max_concurrent_s3_uploads: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Max concurrent background S3 uploads",
    )

    # Resource overcommit
    memory_overcommit_ratio: float = Field(
        default=constants.DEFAULT_MEMORY_OVERCOMMIT_RATIO,
        ge=1.0,
        le=10.0,
        description="Memory overcommit ratio. Effective budget = host_total * (1 - reserve_ratio) * ratio",
    )
    cpu_overcommit_ratio: float = Field(
        default=constants.DEFAULT_CPU_OVERCOMMIT_RATIO,
        ge=1.0,
        le=20.0,
        description="CPU overcommit ratio. Effective budget = (host_cpus - reserve) * ratio",
    )
    host_memory_reserve_ratio: float = Field(
        default=constants.DEFAULT_HOST_MEMORY_RESERVE_RATIO,
        ge=0.01,
        le=0.5,
        description="Fraction of host memory reserved for OS (e.g. 0.1 = 10%)",
    )
    host_cpu_reserve_cores: float = Field(
        default=constants.DEFAULT_HOST_CPU_RESERVE_CORES,
        ge=0.0,
        le=16.0,
        description="CPU cores reserved for host processes (fixed, not a ratio)",
    )
    # Features
    enable_package_validation: bool = Field(
        default=True,
        description="Validate packages against allowlist",
    )
    auto_download_assets: bool = Field(
        default=True,
        description="Automatically download VM images from GitHub Releases if not found",
    )

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
        max_concurrent_vms=5,
        default_memory_mb=512,
        s3_bucket="my-snapshots",
    )
    async with Scheduler(config) as scheduler:
        result = await scheduler.run(code="...", language="python")
    ```
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SchedulerConfig(BaseModel):
    """Configuration for Scheduler.

    All fields have sensible defaults for local development.
    Production deployments should tune max_concurrent_vms based on host resources.

    Attributes:
        max_concurrent_vms: Maximum number of VMs that can run concurrently.
            Each VM uses ~256-512MB memory. Default: 10.
        warm_pool_size: Number of pre-booted VMs per language (python, javascript).
            0 disables warm pool. Default: 0 (cold boot only).
        default_memory_mb: Default guest VM memory in MB. Can be overridden per-run.
            Range: 128-2048. Default: 256.
        default_timeout_seconds: Default execution timeout in seconds.
            Can be overridden per-run. Range: 1-300. Default: 30.
        images_dir: Directory containing base VM images (qcow2, kernels).
            If None, auto-detects from standard locations:
            - Linux: ~/.local/share/exec-sandbox/images/
            - macOS: ~/Library/Application Support/exec-sandbox/images/
            - Env: EXEC_SANDBOX_IMAGES_DIR
        snapshot_cache_dir: Local directory for snapshot cache (L1 cache).
            Default: /tmp/exec-sandbox-cache
        s3_bucket: S3 bucket name for snapshot backup (L3 cache).
            If None, S3 backup is disabled. Requires aioboto3 optional dependency.
        s3_region: AWS region for S3 bucket. Default: us-east-1.
        s3_prefix: Prefix for S3 keys. Default: "snapshots/".
        enable_package_validation: Validate packages against allowlist.
            Disable for testing only. Default: True.
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable after creation
        extra="forbid",  # Reject unknown fields
    )

    # VM pool
    max_concurrent_vms: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent VMs (each uses ~256-512MB memory)",
    )
    warm_pool_size: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Pre-booted VMs per language (0 disables warm pool)",
    )

    # Defaults for run()
    default_memory_mb: int = Field(
        default=256,
        ge=128,
        le=2048,
        description="Default guest VM memory in MB",
    )
    default_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Default execution timeout in seconds",
    )

    # Paths
    images_dir: Path | None = Field(
        default=None,
        description="Directory containing VM images (auto-detect if None)",
    )
    snapshot_cache_dir: Path = Field(
        default=Path("/tmp/exec-sandbox-cache"),
        description="Local snapshot cache directory (L1 cache)",
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

    # Features
    enable_package_validation: bool = Field(
        default=True,
        description="Validate packages against allowlist",
    )

    def get_images_dir(self) -> Path:
        """Get images directory, auto-detecting if not configured.

        Detection order:
        1. Explicit images_dir from config
        2. EXEC_SANDBOX_IMAGES_DIR environment variable
        3. Platform-specific default:
           - Linux: ~/.local/share/exec-sandbox/images/
           - macOS: ~/Library/Application Support/exec-sandbox/images/

        Returns:
            Path to images directory

        Raises:
            FileNotFoundError: Images directory does not exist
        """
        import os
        import sys

        if self.images_dir is not None:
            path = self.images_dir
        elif env_path := os.environ.get("EXEC_SANDBOX_IMAGES_DIR"):
            path = Path(env_path)
        elif sys.platform == "darwin":
            path = Path.home() / "Library" / "Application Support" / "exec-sandbox" / "images"
        else:
            path = Path.home() / ".local" / "share" / "exec-sandbox" / "images"

        if not path.exists():
            raise FileNotFoundError(
                f"Images directory not found: {path}. "
                f"Download images from GitHub Releases or set EXEC_SANDBOX_IMAGES_DIR."
            )

        return path

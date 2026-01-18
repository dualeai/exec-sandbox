"""Runtime configuration from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from exec_sandbox import constants


class Settings(BaseSettings):
    """Runtime configuration from environment variables.

    All settings can be overridden via environment variables with EXEC_SANDBOX_ prefix.
    Example: EXEC_SANDBOX_MAX_CONCURRENT_VMS=20
    """

    model_config = SettingsConfigDict(
        env_prefix="EXEC_SANDBOX_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"

    # QEMU
    qemu_bin_x86: Path = Path("/usr/bin/qemu-system-x86_64")
    qemu_bin_arm: Path = Path("/usr/bin/qemu-system-aarch64")
    qemu_img_bin: Path = Path("/usr/bin/qemu-img")
    kernel_path: Path = Path("/images/kernels")
    base_images_dir: Path = Path("/images")  # qcow2 base images

    # Execution
    max_concurrent_vms: int = 10
    snapshot_cache_dir: Path = Path("/tmp/exec-sandbox-snapshots")  # noqa: S108

    # Snapshot cache (2-tier: L2=local disk, L3=S3)
    snapshot_cache_ttl_days: int = 14  # AWS Lambda SnapStart pattern
    snapshot_cache_max_size_gb: int = 50  # L2 cache size limit
    s3_bucket: str | None = None  # S3 enabled when set
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None  # Custom S3 endpoint (for testing/MinIO)
    max_concurrent_s3_uploads: int = 4  # Max concurrent background S3 uploads

    # Base images (Alpine Docker images)
    base_image_python: str = "python:3.14-alpine"
    base_image_node: str = "node:23-alpine"

    # Limits
    execution_timeout_max: int = constants.MAX_TIMEOUT_SECONDS
    memory_limit_max: int = constants.MAX_MEMORY_MB

    # Testing/Debug
    force_emulation: bool = False
    """Force software emulation instead of hardware virtualization (KVM/HVF).
    Useful for testing emulation code paths locally."""

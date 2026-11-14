"""Runtime configuration from environment variables."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from exec_sandbox import constants
from exec_sandbox.platform_utils import UserDir, get_user_dir

InstanceStoreKind = Literal["s3", "local"]
"""Store of saved instances: an S3-API bucket, or a folder on this host."""


class QemuBinaries(BaseSettings):
    """The qemu-system files to run (EXEC_SANDBOX_QEMU_BIN_X86, EXEC_SANDBOX_QEMU_BIN_ARM):
    used when the file exists, else the name on PATH (system_probes.qemu_system_binary).
    Its own settings class: the probes read these two variables alone, so another invalid
    variable never fails a probe."""

    model_config = SettingsConfigDict(env_prefix="EXEC_SANDBOX_", extra="ignore")

    qemu_bin_x86: Path = Path("/usr/bin/qemu-system-x86_64")
    qemu_bin_arm: Path = Path("/usr/bin/qemu-system-aarch64")


class Settings(QemuBinaries):
    """Runtime configuration from environment variables.

    All settings can be overridden via environment variables with EXEC_SANDBOX_ prefix.
    Example: EXEC_SANDBOX_FORCE_EMULATION=true
    """

    model_config = SettingsConfigDict(
        env_prefix="EXEC_SANDBOX_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # QEMU binaries: QemuBinaries
    kernel_path: Path = Path("/images/kernels")
    base_images_dir: Path = Path("/images")  # qcow2 base images
    # Where a base image that base_images_dir lacks is downloaded (the scheduler: the asset
    # download cache); None: base images come from base_images_dir only
    base_images_download_dir: Path | None = None

    # Execution — both caches under OS-specific cache directory
    disk_snapshot_cache_dir: Path = Field(default_factory=lambda: get_user_dir(kind=UserDir.CACHE) / "disk-snapshots")
    memory_snapshot_cache_dir: Path = Field(
        default_factory=lambda: get_user_dir(kind=UserDir.CACHE) / "memory-snapshots"
    )
    # Home disk files of running VMs: on host disk, not the (often tmpfs) temp dir
    home_disk_dir: Path = Field(default_factory=lambda: get_user_dir(kind=UserDir.CACHE) / "home-disks")

    # Snapshot cache (2-tier: L2=local disk, L3=S3)
    snapshot_cache_ttl_days: int = 14  # Tag ttl_days=N of each L3 upload: a bucket lifecycle rule can expire by it
    s3_bucket: str | None = None  # L3 cache on when set; also the bucket of the "s3" instance store
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None  # S3-compatible endpoint (R2, MinIO, Ceph RGW)
    s3_snapshot_prefix: str = "snapshots/"  # Key prefix of the L3 cache
    s3_instance_prefix: str = "instances/"  # Key prefix of saved instances
    max_concurrent_s3_uploads: int = Field(default=128, ge=1)  # Max concurrent background S3 uploads

    # Saved instances: on when instance_store is set; the keyring is then required
    instance_store: InstanceStoreKind | None = None
    instance_store_dir: Path = Field(
        default_factory=lambda: get_user_dir(kind=UserDir.DATA) / "instances"
    )  # root of the local store
    instance_keyring: SecretStr | None = None  # env EXEC_SANDBOX_INSTANCE_KEYRING
    instance_asset_dir: Path = Field(
        default_factory=lambda: get_user_dir(kind=UserDir.CACHE) / "instance-assets"
    )  # images by digest

    @model_validator(mode="after")
    def _check_s3_prefixes(self) -> Self:
        check_s3_prefixes(snapshot=self.s3_snapshot_prefix, instance=self.s3_instance_prefix)
        return self

    # Resource overcommit (same bounds as SchedulerConfig: env vars reach these fields directly)
    memory_overcommit_ratio: float = Field(default=constants.DEFAULT_MEMORY_OVERCOMMIT_RATIO, ge=1.0, le=10.0)
    cpu_overcommit_ratio: float = Field(default=constants.DEFAULT_CPU_OVERCOMMIT_RATIO, ge=1.0, le=20.0)
    disk_overcommit_ratio: float = Field(default=constants.DEFAULT_DISK_OVERCOMMIT_RATIO, ge=1.0, le=64.0)
    host_memory_reserve_ratio: float = Field(default=constants.DEFAULT_HOST_MEMORY_RESERVE_RATIO, ge=0.01, le=0.5)
    host_cpu_reserve_cores: float = Field(default=constants.DEFAULT_HOST_CPU_RESERVE_CORES, ge=0.0, le=16.0)
    # Host resource overrides, each on its own (None = detected: cgroup limit, else psutil).
    # Useful for testing or container deployments where psutil reports host resources
    host_memory_mb: float | None = None
    host_cpu_count: float | None = None

    # Runtime available-memory floor (Gate 3, always-on)
    # Gate 3 rejects when system_available - requested_memory < floor.
    # 0 (default) = reject only when system literally can't fit the VM.
    # >0 = additionally reserve this much headroom (e.g. 512 for OS safety).
    available_memory_floor_mb: int = 0
    disk_free_floor_mb: int = Field(default=1024, ge=0)  # Free space admission keeps on the home disk filesystem

    # Testing/Debug
    force_emulation: bool = False
    """Force software emulation instead of hardware virtualization (KVM/HVF).
    Useful for testing emulation code paths locally."""


def check_s3_prefixes(*, snapshot: str, instance: str) -> None:
    """The L3 cache and saved instances share the bucket: neither prefix may contain
    the other, or a lifecycle rule or a listing of one would reach the other."""
    if snapshot.startswith(instance) or instance.startswith(snapshot):
        raise ValueError(
            f's3_snapshot_prefix "{snapshot}" and s3_instance_prefix "{instance}" overlap; use two distinct prefixes'
        )

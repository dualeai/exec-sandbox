"""Scheduler configuration for exec-sandbox.

SchedulerConfig provides all configuration options for the Scheduler,
including VM pool settings, resource limits, paths, the S3 cache and saved instances.

Example:
    ```python
    from exec_sandbox import Scheduler, SchedulerConfig, VmSpec

    # Default configuration
    async with Scheduler() as scheduler:
        async with await scheduler.new(VmSpec(language="python")) as vm:
            result = await vm.exec("print('hello')")

    # Custom configuration (VM sizes live in VmSpec, not here)
    config = SchedulerConfig(warm_pool_size=2, s3_bucket="my-snapshots")
    async with Scheduler(config) as scheduler:
        ...
    ```
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - pydantic evaluates annotations at runtime
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from exec_sandbox import constants
from exec_sandbox.platform_utils import UserDir, get_user_dir
from exec_sandbox.settings import InstanceStoreKind, check_s3_prefixes

MIN_SESSION_IDLE_TIMEOUT_SECONDS: Final[int] = 10
MAX_SESSION_IDLE_TIMEOUT_SECONDS: Final[int] = 3600
"""Range of session_idle_timeout_seconds, and of the idle_timeout_seconds of Scheduler.new() and reuse()."""


class SchedulerConfig(BaseModel):
    """Configuration for Scheduler.

    A field that you do not set in code takes its value from EXEC_SANDBOX_<FIELD>
    (the field name in capitals); a value set in code wins. Six fields read no such
    variable: warm_pool_size, default_timeout_seconds, session_idle_timeout_seconds,
    enable_package_validation, auto_download_assets and images_dir (images_dir: see
    below). An invalid variable fails the Scheduler start (ConfigError SETTING_INVALID).

    Attributes:
        warm_pool_size: Number of warm-pool VMs per language (python, javascript, raw).
            0 turns the warm pool off. Default: 0.
        default_timeout_seconds: Default execution timeout in seconds. An exec()
            call can set its own. Range: 1-300. Default: 30.
        session_idle_timeout_seconds: Close a session after this long without a call.
            Range: 10-3600. Default: 300.
        images_dir: Folder of the VM images (qcow2, kernels).
            If None, exec-sandbox looks in this order:
            - EXEC_SANDBOX_IMAGES_DIR env var
            - ./images/dist/ (local build)
            - OS-specific cache folder / v<version>/ (download cache; macOS:
              ~/Library/Caches/exec-sandbox/, Linux: ~/.cache/exec-sandbox/)
        disk_snapshot_cache_dir: Folder of the L2 cache (disk snapshots).
            Default: OS-specific cache folder / disk-snapshots
        memory_snapshot_cache_dir: Folder of the L1 cache (memory snapshots).
            Default: OS-specific cache folder / memory-snapshots
        s3_bucket: S3 bucket for the L3 snapshot cache (under s3_snapshot_prefix)
            and the "s3" instance store (under s3_instance_prefix). None turns the
            L3 cache off. Needs the s3 extra (pip install exec-sandbox[s3]).
        s3_region: AWS region of the S3 bucket. Default: us-east-1.
        s3_endpoint_url: S3-compatible endpoint (R2, MinIO, Ceph RGW). Default: None (AWS).
        s3_snapshot_prefix: Key prefix of the L3 cache. Default: "snapshots/".
        s3_instance_prefix: Key prefix of saved instances. Default: "instances/".
        max_concurrent_s3_uploads: Max concurrent background S3 uploads. Default: 128.
        memory_overcommit_ratio, cpu_overcommit_ratio, host_memory_reserve_ratio,
            host_cpu_reserve_cores: Admission budgets (see each field's description).
        enable_package_validation: Check packages against the allow-list.
            Turn it off for tests only. Default: True.
        auto_download_assets: Download the VM images from GitHub Releases, into
            the cache folder, when they are not on this host. Default: True.
        instance_store: Where saved instances live (Session.save, Scheduler.reuse):
            "s3" (s3_bucket under s3_instance_prefix; the s3 extra) or "local" (a
            folder on this host, instance_store_dir; the cli or the s3 extra). Every
            worker needs QEMU 11.0 or later. Not set in code: EXEC_SANDBOX_INSTANCE_STORE
            decides (unset: off). instance_store=None in code turns saved instances
            off, whatever the environment holds.
        instance_store_dir: Root folder of the "local" store. Default: OS-specific
            data dir / instances (macOS: ~/Library/Application Support/exec-sandbox/,
            Linux: ~/.local/share/exec-sandbox/).
        instance_keyring: Keys that encrypt saved instances. Required when instance_store
            is set: without it, the Scheduler start fails (ConfigError INSTANCE_CONFIG).
            Also EXEC_SANDBOX_INSTANCE_KEYRING. To rotate: add the new key at the end on
            every worker, then move it to the front, then remove the old key when no
            instance uses it (describe_instance(id).key_id).
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable after creation
        extra="forbid",  # Reject unknown fields
    )

    # VM pool
    warm_pool_size: int = Field(
        default=0,
        ge=0,
        description="Warm-pool VMs per language, started before the request (0: no warm pool)",
    )

    # Defaults for Session.exec()
    default_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Default execution timeout in seconds",
    )
    session_idle_timeout_seconds: int = Field(
        default=300,
        ge=MIN_SESSION_IDLE_TIMEOUT_SECONDS,
        le=MAX_SESSION_IDLE_TIMEOUT_SECONDS,
        description="Close a session after this long without a call",
    )

    # Paths
    images_dir: Path | None = Field(
        default=None,
        description="Folder of the VM images (None: found automatically)",
    )
    disk_snapshot_cache_dir: Path = Field(
        default_factory=lambda: get_user_dir(kind=UserDir.CACHE, app_name="exec-sandbox") / "disk-snapshots",
        description="Folder of the L2 cache (disk snapshots)",
    )
    memory_snapshot_cache_dir: Path = Field(
        default_factory=lambda: get_user_dir(kind=UserDir.CACHE, app_name="exec-sandbox") / "memory-snapshots",
        description="Folder of the L1 cache (memory snapshots)",
    )
    # S3: one bucket for the L3 snapshot cache and for the "s3" instance store, each under its prefix
    s3_bucket: str | None = Field(
        default=None,
        description='S3 bucket of the L3 cache and the "s3" instance store (None disables the L3 cache)',
    )
    s3_region: str = Field(
        default="us-east-1",
        description="AWS region of the S3 bucket",
    )
    s3_endpoint_url: str | None = Field(
        default=None,
        description="Endpoint of an S3-compatible store (R2, MinIO, Ceph RGW); None for AWS",
    )
    s3_snapshot_prefix: str = Field(
        default="snapshots/",
        description="Key prefix of the L3 snapshot cache",
    )
    s3_instance_prefix: str = Field(
        default="instances/",
        description="Key prefix of saved instances "
        "(needs conditional writes: AWS S3, R2, Ceph RGW 19.2.7 or later, MinIO)",
    )
    max_concurrent_s3_uploads: int = Field(
        default=128,
        ge=1,
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
        description="Fraction of host memory reserved for the OS (for example 0.1 = 10%)",
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
        description="Check packages against the allow-list",
    )
    auto_download_assets: bool = Field(
        default=True,
        description="Download missing VM images from GitHub Releases",
    )

    # Saved instances (Session.save / Scheduler.reuse): on when instance_store is set
    instance_store: InstanceStoreKind | None = Field(
        default=None,
        description='Store of saved instances: "s3" (s3_bucket) or "local" (instance_store_dir); None turns them off',
    )
    instance_store_dir: Path = Field(
        default_factory=lambda: get_user_dir(kind=UserDir.DATA, app_name="exec-sandbox") / "instances",
        description='Root folder of the "local" instance store',
    )
    instance_keyring: SecretStr | None = Field(
        default=None,
        description="Comma-separated base64 keys of 32 random bytes, newest first: the first encrypts new saves, "
        "the others only open older ones. Parsed at Scheduler start; never logged or passed to child processes.",
    )

    @model_validator(mode="after")
    def _check_s3_prefixes(self) -> Self:
        check_s3_prefixes(snapshot=self.s3_snapshot_prefix, instance=self.s3_instance_prefix)
        return self

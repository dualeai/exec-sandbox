"""Unit tests for SchedulerConfig.

Tests configuration validation and get_images_dir() path resolution.
No mocks - uses real filesystem and environment variables.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.platform_utils import get_cache_dir

# ============================================================================
# Config Validation
# ============================================================================


class TestSchedulerConfigValidation:
    """Tests for SchedulerConfig field validation."""

    def test_defaults(self) -> None:
        """SchedulerConfig has sensible defaults."""
        config = SchedulerConfig()
        assert config.warm_pool_size == 0
        assert config.default_memory_mb == 256
        assert config.default_timeout_seconds == 30
        assert config.images_dir is None
        assert config.disk_snapshot_cache_dir == get_cache_dir() / "disk-snapshots"
        assert config.s3_bucket is None
        assert config.s3_region == "us-east-1"
        assert config.s3_prefix == "snapshots/"
        assert config.enable_package_validation is True

    def test_warm_pool_size_range(self) -> None:
        """warm_pool_size must be >= 0."""
        # Valid: 0 (disabled)
        config = SchedulerConfig(warm_pool_size=0)
        assert config.warm_pool_size == 0

        # Valid: large value (no upper bound)
        config = SchedulerConfig(warm_pool_size=100)
        assert config.warm_pool_size == 100

        # Invalid: negative
        with pytest.raises(ValidationError):
            SchedulerConfig(warm_pool_size=-1)

    def test_default_memory_mb_range(self) -> None:
        """default_memory_mb must be >= 128."""
        # Valid: min
        config = SchedulerConfig(default_memory_mb=128)
        assert config.default_memory_mb == 128

        # Valid: large value (no upper bound)
        config = SchedulerConfig(default_memory_mb=8192)
        assert config.default_memory_mb == 8192

        # Invalid: < 128
        with pytest.raises(ValidationError):
            SchedulerConfig(default_memory_mb=127)

    def test_default_timeout_seconds_range(self) -> None:
        """default_timeout_seconds must be 1-300."""
        # Valid: min
        config = SchedulerConfig(default_timeout_seconds=1)
        assert config.default_timeout_seconds == 1

        # Valid: max
        config = SchedulerConfig(default_timeout_seconds=300)
        assert config.default_timeout_seconds == 300

        # Invalid: 0
        with pytest.raises(ValidationError):
            SchedulerConfig(default_timeout_seconds=0)

        # Invalid: > 300
        with pytest.raises(ValidationError):
            SchedulerConfig(default_timeout_seconds=301)

    def test_extra_fields_forbidden(self) -> None:
        """SchedulerConfig rejects unknown fields."""
        with pytest.raises(ValidationError):
            SchedulerConfig(unknown_field="value")  # type: ignore[call-arg]


# ============================================================================
# Full Config with S3
# ============================================================================


class TestSchedulerConfigS3:
    """Tests for S3-related configuration."""

    def test_s3_config(self) -> None:
        """SchedulerConfig with S3 settings."""
        config = SchedulerConfig(
            s3_bucket="my-bucket",
            s3_region="eu-west-1",
            s3_prefix="cache/",
        )
        assert config.s3_bucket == "my-bucket"
        assert config.s3_region == "eu-west-1"
        assert config.s3_prefix == "cache/"

    def test_s3_disabled_by_default(self) -> None:
        """S3 is disabled when bucket is None."""
        config = SchedulerConfig()
        assert config.s3_bucket is None


# ============================================================================
# Cache Dir Integration
# ============================================================================


class TestSchedulerConfigCacheDir:
    """Tests for snapshot cache dir defaults using get_cache_dir()."""

    def test_env_override_affects_default(self) -> None:
        """EXEC_SANDBOX_CACHE_DIR env var changes both cache dir defaults."""
        with patch.dict("os.environ", {"EXEC_SANDBOX_CACHE_DIR": "/env/cache"}):
            config = SchedulerConfig()
            assert str(config.disk_snapshot_cache_dir).startswith("/env/cache")
            assert str(config.memory_snapshot_cache_dir).startswith("/env/cache")

    def test_explicit_override_bypasses_get_cache_dir(self) -> None:
        """Explicit path overrides get_cache_dir() factory default."""
        config = SchedulerConfig(
            disk_snapshot_cache_dir=Path("/explicit/disk"),
            memory_snapshot_cache_dir=Path("/explicit/memory"),
        )
        assert config.disk_snapshot_cache_dir == Path("/explicit/disk")
        assert config.memory_snapshot_cache_dir == Path("/explicit/memory")

    def test_snapshot_subdirs_appended_to_cache_dir(self) -> None:
        """Snapshot cache dirs end with expected subdirectory names."""
        config = SchedulerConfig()
        assert str(config.disk_snapshot_cache_dir).endswith("/disk-snapshots")
        assert str(config.memory_snapshot_cache_dir).endswith("/memory-snapshots")

    def test_default_factory_evaluates_at_instantiation(self) -> None:
        """default_factory evaluates at each SchedulerConfig() call, not frozen."""
        with patch.dict("os.environ", {"EXEC_SANDBOX_CACHE_DIR": "/first"}):
            config1 = SchedulerConfig()

        with patch.dict("os.environ", {"EXEC_SANDBOX_CACHE_DIR": "/second"}):
            config2 = SchedulerConfig()

        assert config1.disk_snapshot_cache_dir != config2.disk_snapshot_cache_dir

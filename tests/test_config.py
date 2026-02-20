"""Unit tests for SchedulerConfig.

Tests configuration validation and get_images_dir() path resolution.
No mocks - uses real filesystem and environment variables.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from exec_sandbox.config import SchedulerConfig

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
        assert config.snapshot_cache_dir == Path("/tmp/exec-sandbox-cache")
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

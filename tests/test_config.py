"""Unit tests for SchedulerConfig.

Tests configuration validation and the default cache folders.
No mocks - uses real filesystem and environment variables.
"""

import sys
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
        """SchedulerConfig has sensible defaults (the cache folders: TestSchedulerConfigCacheDir)."""
        config = SchedulerConfig()
        assert config.warm_pool_size == 0
        assert config.default_timeout_seconds == 30
        assert config.images_dir is None
        assert config.s3_bucket is None
        assert config.s3_region == "us-east-1"
        assert config.s3_snapshot_prefix == "snapshots/"
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
            s3_snapshot_prefix="cache/",
        )
        assert config.s3_bucket == "my-bucket"
        assert config.s3_region == "eu-west-1"
        assert config.s3_snapshot_prefix == "cache/"


# ============================================================================
# Cache Dir Integration
# ============================================================================


class TestSchedulerConfigCacheDir:
    """The default L1 and L2 cache folders: in EXEC_SANDBOX_CACHE_DIR, else in the OS cache
    folder of the user. The expected paths are written out."""

    def test_default_is_the_user_cache_folder(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No EXEC_SANDBOX_CACHE_DIR: macOS ~/Library/Caches/exec-sandbox, Linux
        ~/.cache/exec-sandbox (no XDG_CACHE_HOME), "~" from HOME."""
        monkeypatch.delenv("EXEC_SANDBOX_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        cache = tmp_path / ("Library/Caches" if sys.platform == "darwin" else ".cache") / "exec-sandbox"

        config = SchedulerConfig()

        assert config.disk_snapshot_cache_dir == cache / "disk-snapshots"
        assert config.memory_snapshot_cache_dir == cache / "memory-snapshots"

    def test_env_override_affects_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EXEC_SANDBOX_CACHE_DIR env var changes both cache dir defaults."""
        monkeypatch.setenv("EXEC_SANDBOX_CACHE_DIR", "/env/cache")
        config = SchedulerConfig()
        assert config.disk_snapshot_cache_dir == Path("/env/cache/disk-snapshots")
        assert config.memory_snapshot_cache_dir == Path("/env/cache/memory-snapshots")

    def test_explicit_override_bypasses_user_cache_dir(self) -> None:
        """An explicit path wins over the default."""
        config = SchedulerConfig(
            disk_snapshot_cache_dir=Path("/explicit/disk"),
            memory_snapshot_cache_dir=Path("/explicit/memory"),
        )
        assert config.disk_snapshot_cache_dir == Path("/explicit/disk")
        assert config.memory_snapshot_cache_dir == Path("/explicit/memory")

    def test_default_factory_evaluates_at_instantiation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each SchedulerConfig() reads the environment again: the default is not frozen at import."""
        monkeypatch.setenv("EXEC_SANDBOX_CACHE_DIR", "/first")
        config1 = SchedulerConfig()
        monkeypatch.setenv("EXEC_SANDBOX_CACHE_DIR", "/second")
        config2 = SchedulerConfig()

        assert config1.disk_snapshot_cache_dir == Path("/first/disk-snapshots")
        assert config2.disk_snapshot_cache_dir == Path("/second/disk-snapshots")

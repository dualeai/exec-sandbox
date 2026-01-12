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
        assert config.max_concurrent_vms == 10
        assert config.warm_pool_size == 0
        assert config.default_memory_mb == 256
        assert config.default_timeout_seconds == 30
        assert config.images_dir is None
        assert config.snapshot_cache_dir == Path("/tmp/exec-sandbox-cache")
        assert config.s3_bucket is None
        assert config.s3_region == "us-east-1"
        assert config.s3_prefix == "snapshots/"
        assert config.enable_package_validation is True

    def test_max_concurrent_vms_range(self) -> None:
        """max_concurrent_vms must be 1-100."""
        # Valid: min
        config = SchedulerConfig(max_concurrent_vms=1)
        assert config.max_concurrent_vms == 1

        # Valid: max
        config = SchedulerConfig(max_concurrent_vms=100)
        assert config.max_concurrent_vms == 100

        # Invalid: 0
        with pytest.raises(ValidationError):
            SchedulerConfig(max_concurrent_vms=0)

        # Invalid: > 100
        with pytest.raises(ValidationError):
            SchedulerConfig(max_concurrent_vms=101)

    def test_warm_pool_size_range(self) -> None:
        """warm_pool_size must be 0-10."""
        # Valid: 0 (disabled)
        config = SchedulerConfig(warm_pool_size=0)
        assert config.warm_pool_size == 0

        # Valid: 10
        config = SchedulerConfig(warm_pool_size=10)
        assert config.warm_pool_size == 10

        # Invalid: negative
        with pytest.raises(ValidationError):
            SchedulerConfig(warm_pool_size=-1)

        # Invalid: > 10
        with pytest.raises(ValidationError):
            SchedulerConfig(warm_pool_size=11)

    def test_default_memory_mb_range(self) -> None:
        """default_memory_mb must be 128-2048."""
        # Valid: min
        config = SchedulerConfig(default_memory_mb=128)
        assert config.default_memory_mb == 128

        # Valid: max
        config = SchedulerConfig(default_memory_mb=2048)
        assert config.default_memory_mb == 2048

        # Invalid: < 128
        with pytest.raises(ValidationError):
            SchedulerConfig(default_memory_mb=127)

        # Invalid: > 2048
        with pytest.raises(ValidationError):
            SchedulerConfig(default_memory_mb=2049)

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

    def test_immutable(self) -> None:
        """SchedulerConfig is frozen (immutable)."""
        config = SchedulerConfig()
        with pytest.raises(ValidationError):
            config.max_concurrent_vms = 20  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """SchedulerConfig rejects unknown fields."""
        with pytest.raises(ValidationError):
            SchedulerConfig(unknown_field="value")  # type: ignore[call-arg]


# ============================================================================
# get_images_dir() Path Resolution
# ============================================================================


class TestGetImagesDir:
    """Tests for get_images_dir() method.

    Uses real filesystem and environment variables.
    """

    def test_explicit_images_dir_exists(self, tmp_path: Path) -> None:
        """get_images_dir() returns explicit path when it exists."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        config = SchedulerConfig(images_dir=images_dir)
        result = config.get_images_dir()

        assert result == images_dir

    def test_explicit_images_dir_not_exists(self, tmp_path: Path) -> None:
        """get_images_dir() raises FileNotFoundError for missing explicit path when auto_download disabled."""
        missing_dir = tmp_path / "nonexistent"

        # With auto_download_assets=False, missing directory should raise error
        config = SchedulerConfig(images_dir=missing_dir, auto_download_assets=False)

        with pytest.raises(FileNotFoundError) as exc_info:
            config.get_images_dir()

        assert "nonexistent" in str(exc_info.value)
        assert "download images from GitHub Releases" in str(exc_info.value)

    def test_env_var_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_images_dir() uses EXEC_SANDBOX_IMAGES_DIR env var."""
        images_dir = tmp_path / "env-images"
        images_dir.mkdir()

        monkeypatch.setenv("EXEC_SANDBOX_IMAGES_DIR", str(images_dir))

        config = SchedulerConfig()  # No explicit images_dir
        result = config.get_images_dir()

        assert result == images_dir

    def test_env_var_not_exists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_images_dir() raises FileNotFoundError for missing env var path when auto_download disabled."""
        missing_dir = tmp_path / "missing-env-path"
        monkeypatch.setenv("EXEC_SANDBOX_IMAGES_DIR", str(missing_dir))

        # With auto_download_assets=False, missing directory should raise error
        config = SchedulerConfig(auto_download_assets=False)

        with pytest.raises(FileNotFoundError) as exc_info:
            config.get_images_dir()

        assert "missing-env-path" in str(exc_info.value)

    def test_explicit_overrides_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit images_dir takes precedence over env var."""
        explicit_dir = tmp_path / "explicit"
        explicit_dir.mkdir()

        env_dir = tmp_path / "env"
        env_dir.mkdir()

        monkeypatch.setenv("EXEC_SANDBOX_IMAGES_DIR", str(env_dir))

        config = SchedulerConfig(images_dir=explicit_dir)
        result = config.get_images_dir()

        assert result == explicit_dir
        assert result != env_dir

    def test_platform_default_macos_with_auto_download(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_images_dir() uses macOS cache path when auto_download_assets=True."""
        from exec_sandbox import platform_utils
        from exec_sandbox.platform_utils import HostOS

        # Clear env vars
        monkeypatch.delenv("EXEC_SANDBOX_IMAGES_DIR", raising=False)
        monkeypatch.delenv("EXEC_SANDBOX_CACHE_DIR", raising=False)

        # Simulate macOS by patching detect_host_os in the module
        monkeypatch.setattr(platform_utils, "detect_host_os", lambda: HostOS.MACOS)

        # Override Path.home() to use tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # With auto_download_assets=True (default), uses cache path
        scheduler_config = SchedulerConfig()
        result = scheduler_config.get_images_dir()

        # Should use Caches directory (not Application Support)
        expected = tmp_path / "Library" / "Caches" / "exec-sandbox"
        assert result == expected

    def test_platform_default_macos_without_auto_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_images_dir() uses macOS Application Support path when auto_download_assets=False."""
        from exec_sandbox import platform_utils
        from exec_sandbox.platform_utils import HostOS

        # Clear env vars
        monkeypatch.delenv("EXEC_SANDBOX_IMAGES_DIR", raising=False)
        monkeypatch.delenv("EXEC_SANDBOX_CACHE_DIR", raising=False)

        # Simulate macOS by patching detect_host_os in the module
        monkeypatch.setattr(platform_utils, "detect_host_os", lambda: HostOS.MACOS)

        # Create the expected default directory
        macos_default = tmp_path / "Library" / "Application Support" / "exec-sandbox" / "images"
        macos_default.mkdir(parents=True)

        # Override Path.home() to use tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        scheduler_config = SchedulerConfig(auto_download_assets=False)
        result = scheduler_config.get_images_dir()

        assert result == macos_default

    def test_platform_default_linux_with_auto_download(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_images_dir() uses Linux cache path when auto_download_assets=True."""
        from exec_sandbox import platform_utils
        from exec_sandbox.platform_utils import HostOS

        # Clear env vars
        monkeypatch.delenv("EXEC_SANDBOX_IMAGES_DIR", raising=False)
        monkeypatch.delenv("EXEC_SANDBOX_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

        # Simulate Linux by patching detect_host_os in the module
        monkeypatch.setattr(platform_utils, "detect_host_os", lambda: HostOS.LINUX)

        # Override Path.home() to use tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # With auto_download_assets=True (default), uses cache path
        scheduler_config = SchedulerConfig()
        result = scheduler_config.get_images_dir()

        # Should use .cache directory (not .local/share)
        expected = tmp_path / ".cache" / "exec-sandbox"
        assert result == expected

    def test_platform_default_linux_without_auto_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_images_dir() uses Linux .local/share path when auto_download_assets=False."""
        from exec_sandbox import platform_utils
        from exec_sandbox.platform_utils import HostOS

        # Clear env vars
        monkeypatch.delenv("EXEC_SANDBOX_IMAGES_DIR", raising=False)
        monkeypatch.delenv("EXEC_SANDBOX_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

        # Simulate Linux by patching detect_host_os in the module
        monkeypatch.setattr(platform_utils, "detect_host_os", lambda: HostOS.LINUX)

        # Create the expected default directory
        linux_default = tmp_path / ".local" / "share" / "exec-sandbox" / "images"
        linux_default.mkdir(parents=True)

        # Override Path.home() to use tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        scheduler_config = SchedulerConfig(auto_download_assets=False)
        result = scheduler_config.get_images_dir()

        assert result == linux_default

    def test_platform_default_not_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_images_dir() raises FileNotFoundError when default path missing and auto_download disabled."""
        # Clear env vars
        monkeypatch.delenv("EXEC_SANDBOX_IMAGES_DIR", raising=False)
        monkeypatch.delenv("EXEC_SANDBOX_CACHE_DIR", raising=False)

        # Use a non-existent home directory
        fake_home = Path("/nonexistent/home/user")
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # With auto_download_assets=False, missing directory should raise error
        config = SchedulerConfig(auto_download_assets=False)

        with pytest.raises(FileNotFoundError) as exc_info:
            config.get_images_dir()

        assert "exec-sandbox" in str(exc_info.value)
        assert "download images from GitHub Releases" in str(exc_info.value)


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

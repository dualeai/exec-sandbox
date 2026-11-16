"""Unit tests for SchedulerConfig.

Tests configuration validation and the default cache folders.
No mocks - uses real filesystem and environment variables.
"""

import sys
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError
from pydantic.fields import FieldInfo

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.settings import Settings


def _refused(**fields: object) -> list[tuple[tuple[int | str, ...], str]]:
    """The field and the error type of each error that SchedulerConfig(**fields) raises."""
    with pytest.raises(ValidationError) as exc:
        SchedulerConfig(**fields)
    return [(error["loc"], error["type"]) for error in exc.value.errors()]


# ============================================================================
# Config Validation
# ============================================================================


class TestSchedulerConfigValidation:
    """Tests for SchedulerConfig field validation."""

    def test_defaults(self) -> None:
        """No warm pool, a 30 s timeout, no S3, package validation on (the cache folders:
        TestSchedulerConfigCacheDir)."""
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
        assert _refused(warm_pool_size=-1) == [(("warm_pool_size",), "greater_than_equal")]

    def test_default_timeout_seconds_range(self) -> None:
        """default_timeout_seconds must be 1-300."""
        # Valid: min
        config = SchedulerConfig(default_timeout_seconds=1)
        assert config.default_timeout_seconds == 1

        # Valid: max
        config = SchedulerConfig(default_timeout_seconds=300)
        assert config.default_timeout_seconds == 300

        # Invalid: 0, and > 300
        assert _refused(default_timeout_seconds=0) == [(("default_timeout_seconds",), "greater_than_equal")]
        assert _refused(default_timeout_seconds=301) == [(("default_timeout_seconds",), "less_than_equal")]

    def test_extra_fields_forbidden(self) -> None:
        """SchedulerConfig rejects unknown fields."""
        assert _refused(unknown_field="value") == [(("unknown_field",), "extra_forbidden")]


class TestSchedulerConfigMatchesSettings:
    """The Scheduler passes each field that SchedulerConfig and Settings share, when set in code,
    to Settings, which reads the EXEC_SANDBOX_* variables for the others (settings_from_config)."""

    def test_shared_fields_have_one_type_bounds_and_default(self) -> None:
        """A value that SchedulerConfig accepts, Settings accepts too (else the Scheduler start
        would blame a variable), and a field left unset means the same in both."""
        shared = SchedulerConfig.model_fields.keys() & Settings.model_fields.keys()

        def shape(field: FieldInfo) -> tuple[object, ...]:
            return (field.annotation, field.metadata, field.get_default(call_default_factory=True))

        config_fields, settings_fields = SchedulerConfig.model_fields, Settings.model_fields
        assert {name: shape(config_fields[name]) for name in shared} == {
            name: shape(settings_fields[name]) for name in shared
        }


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


class TestInstanceKeyring:
    @pytest.mark.parametrize("value", [pytest.param("k1,k2", id="str"), pytest.param(SecretStr("k1,k2"), id="secret")])
    def test_a_str_is_kept_as_a_secret(self, value: str | SecretStr) -> None:
        """The str that keyring.get_password returns is taken as it is, and kept as a
        SecretStr: never in a repr, a log or a dump."""
        config = SchedulerConfig(instance_keyring=value)
        assert isinstance(config.instance_keyring, SecretStr)
        assert config.instance_keyring.get_secret_value() == "k1,k2"
        assert "k1" not in repr(config)
        assert "k1" not in config.model_dump_json()

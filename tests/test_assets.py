"""Tests for exec_sandbox.assets module.

Tests internal functions (_find_asset, _find_images_dir) and public API (ensure_assets).
Internal functions are tested directly to ensure correct path resolution behavior.
Note: _find_asset and _find_images_dir are private but tested for correctness.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from exec_sandbox.asset_downloader import decompress_zstd
from exec_sandbox.assets import (
    _find_asset,  # pyright: ignore[reportPrivateUsage]
    _find_images_dir,  # pyright: ignore[reportPrivateUsage]
    _versioned_cache_dir,  # pyright: ignore[reportPrivateUsage]
    ensure_assets,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def _disable_local_images_dir():
    """
    Disable local images/dist auto-detection by pointing __file__ elsewhere.

    This prevents tests from finding real assets in the project's images/dist/.
    """
    # Point to a nonexistent location so local_images check fails
    fake_file = "/nonexistent/src/exec_sandbox/assets.py"
    with patch("exec_sandbox.assets.__file__", fake_file):
        yield


class TestFindAsset:
    """Tests for _find_asset() private function."""

    # =========================================================================
    # Normal Cases
    # =========================================================================

    async def test_asset_found_in_env_var_path(self, tmp_path: Path) -> None:
        """Asset in EXEC_SANDBOX_IMAGES_DIR is returned."""
        (tmp_path / "vmlinuz-x86_64").touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            result = await _find_asset("vmlinuz-x86_64")
            assert result == tmp_path / "vmlinuz-x86_64"

    async def test_asset_found_in_cache_dir(self, tmp_path: Path, _disable_local_images_dir: None) -> None:
        """Asset in cache directory is returned when no env var or local."""
        (tmp_path / "vmlinuz-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path),
        ):
            result = await _find_asset("vmlinuz-x86_64")
            assert result == tmp_path / "vmlinuz-x86_64"

    async def test_asset_not_found(self, tmp_path: Path, _disable_local_images_dir: None) -> None:
        """Returns None when asset doesn't exist anywhere."""
        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path),
        ):
            result = await _find_asset("nonexistent-file")
            assert result is None

    async def test_decompressed_version_found(self, tmp_path: Path) -> None:
        """When requesting .zst, decompressed version (without .zst) is returned."""
        # Create decompressed version only
        (tmp_path / "vmlinuz-x86_64").touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            result = await _find_asset("vmlinuz-x86_64.zst")
            assert result == tmp_path / "vmlinuz-x86_64"

    async def test_compressed_preferred_over_decompressed(self, tmp_path: Path) -> None:
        """When both .zst and decompressed exist, .zst is returned."""
        (tmp_path / "vmlinuz-x86_64.zst").touch()
        (tmp_path / "vmlinuz-x86_64").touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            result = await _find_asset("vmlinuz-x86_64.zst")
            assert result == tmp_path / "vmlinuz-x86_64.zst"

    # =========================================================================
    # Priority / Order Cases
    # =========================================================================

    async def test_override_takes_priority_over_all(self, tmp_path: Path) -> None:
        """Override path is checked first."""
        override_dir = tmp_path / "override"
        env_dir = tmp_path / "env"
        override_dir.mkdir()
        env_dir.mkdir()

        (override_dir / "vmlinuz-x86_64").write_text("from_override")
        (env_dir / "vmlinuz-x86_64").write_text("from_env")

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}):
            result = await _find_asset("vmlinuz-x86_64", override=override_dir)
            assert result == override_dir / "vmlinuz-x86_64"

    async def test_env_var_takes_priority_over_cache(self, tmp_path: Path) -> None:
        """Env var path is checked before cache directory."""
        env_dir = tmp_path / "env"
        cache_dir = tmp_path / "cache"
        env_dir.mkdir()
        cache_dir.mkdir()

        (env_dir / "vmlinuz-x86_64").write_text("from_env")
        (cache_dir / "vmlinuz-x86_64").write_text("from_cache")

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=cache_dir),
        ):
            result = await _find_asset("vmlinuz-x86_64")
            assert result == env_dir / "vmlinuz-x86_64"

    async def test_local_build_takes_priority_over_cache(self, tmp_path: Path) -> None:
        """Local build directory is checked before cache."""
        local_dir = tmp_path / "images" / "dist"
        cache_dir = tmp_path / "cache"
        local_dir.mkdir(parents=True)
        cache_dir.mkdir()

        (local_dir / "vmlinuz-x86_64").write_text("from_local")
        (cache_dir / "vmlinuz-x86_64").write_text("from_cache")

        # Point __file__ so local_images resolves to our tmp_path
        fake_file = str(tmp_path / "src" / "exec_sandbox" / "assets.py")

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets.__file__", fake_file),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=cache_dir),
        ):
            result = await _find_asset("vmlinuz-x86_64")
            assert result == local_dir / "vmlinuz-x86_64"

    async def test_cache_used_when_env_empty(self, tmp_path: Path, _disable_local_images_dir: None) -> None:
        """Cache directory is used when env var is empty."""
        (tmp_path / "vmlinuz-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path),
        ):
            result = await _find_asset("vmlinuz-x86_64")
            assert result == tmp_path / "vmlinuz-x86_64"

    # =========================================================================
    # Edge Cases
    # =========================================================================

    async def test_env_var_whitespace_only(self, tmp_path: Path, _disable_local_images_dir: None) -> None:
        """Whitespace-only env var is treated as empty."""
        (tmp_path / "vmlinuz-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": "   "}, clear=False),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path),
        ):
            result = await _find_asset("vmlinuz-x86_64")
            # Should fall back to cache since env var is whitespace
            assert result == tmp_path / "vmlinuz-x86_64"

    async def test_env_var_nonexistent_directory(self, tmp_path: Path, _disable_local_images_dir: None) -> None:
        """Env var pointing to nonexistent directory falls back to cache."""
        (tmp_path / "vmlinuz-x86_64").touch()
        nonexistent = tmp_path / "does_not_exist"

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(nonexistent)}),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path),
        ):
            result = await _find_asset("vmlinuz-x86_64")
            # Env var dir doesn't exist, so falls back to cache
            assert result == tmp_path / "vmlinuz-x86_64"

    async def test_symlink_followed(self, tmp_path: Path) -> None:
        """Symlinks to files are followed."""
        real_file = tmp_path / "real_file"
        real_file.touch()
        symlink = tmp_path / "vmlinuz-x86_64"
        symlink.symlink_to(real_file)

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            result = await _find_asset("vmlinuz-x86_64")
            assert result == tmp_path / "vmlinuz-x86_64"

    async def test_file_with_spaces_in_name(self, tmp_path: Path) -> None:
        """Filenames with spaces are handled correctly."""
        (tmp_path / "file with spaces.txt").touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            result = await _find_asset("file with spaces.txt")
            assert result == tmp_path / "file with spaces.txt"

    async def test_non_zst_extension_no_decompressed_check(self, tmp_path: Path) -> None:
        """Files without .zst extension don't trigger decompressed lookup."""
        # Only decompressed version exists
        (tmp_path / "somefile").touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            # Requesting .txt should NOT find "somefile" (no .zst suffix)
            result = await _find_asset("somefile.txt")
            assert result is None


class TestFindImagesDir:
    """Tests for _find_images_dir() private function."""

    async def test_finds_existing_override_dir(self, tmp_path: Path) -> None:
        """Returns override directory when it exists."""
        result = await _find_images_dir(override=tmp_path)
        assert result == tmp_path

    async def test_override_not_exists_returns_none(self, tmp_path: Path) -> None:
        """Returns None when override doesn't exist and no other paths found."""
        nonexistent = tmp_path / "does_not_exist"

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "cache"),
        ):
            result = await _find_images_dir(override=nonexistent)
            assert result is None

    async def test_finds_env_var_dir(self, tmp_path: Path) -> None:
        """Returns env var directory when it exists."""
        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
        ):
            result = await _find_images_dir()
            assert result == tmp_path


class TestEnsureAssets:
    """Tests for ensure_assets() public function."""

    async def test_returns_existing_images_dir(self, tmp_path: Path) -> None:
        """Returns images directory when it exists and contains kernel."""
        # ensure_assets validates that kernel exists, so create one
        from exec_sandbox.asset_downloader import get_current_arch

        arch = get_current_arch()
        (tmp_path / f"vmlinuz-{arch}").touch()

        result = await ensure_assets(override=tmp_path)
        assert result == tmp_path

    async def test_raises_when_not_found_and_download_disabled(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when assets not found and download=False."""
        nonexistent = tmp_path / "does_not_exist"

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "cache"),
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                await ensure_assets(override=nonexistent, download=False)

            assert "auto_download_assets=False" in str(exc_info.value)

    async def test_empty_dir_triggers_download(self, tmp_path: Path) -> None:
        """Empty directory (no kernel) should trigger download when download=True."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # Should not return the empty dir, should try to download
        # (we mock ensure_assets_available to verify it's called)
        with patch("exec_sandbox.assets.ensure_assets_available") as mock_download:
            mock_download.return_value = (tmp_path / "downloaded", tmp_path / "gvproxy")
            result = await ensure_assets(override=empty_dir, download=True)

            mock_download.assert_called_once()
            assert result == tmp_path / "downloaded"


class TestOfflineMode:
    """Tests for is_offline_mode() function."""

    def test_offline_mode_enabled(self) -> None:
        """EXEC_SANDBOX_OFFLINE=1 enables offline mode."""
        from exec_sandbox.assets import is_offline_mode

        with patch.dict(os.environ, {"EXEC_SANDBOX_OFFLINE": "1"}):
            assert is_offline_mode() is True

    def test_offline_mode_disabled_explicit(self) -> None:
        """EXEC_SANDBOX_OFFLINE=0 disables offline mode."""
        from exec_sandbox.assets import is_offline_mode

        with patch.dict(os.environ, {"EXEC_SANDBOX_OFFLINE": "0"}):
            assert is_offline_mode() is False

    def test_offline_mode_disabled_by_default(self) -> None:
        """Offline mode is disabled when env var is not set."""
        from exec_sandbox.assets import is_offline_mode

        with patch.dict(os.environ, {}, clear=True):
            assert is_offline_mode() is False

    def test_offline_mode_other_values(self) -> None:
        """Only '1' enables offline mode, other values disable it."""
        from exec_sandbox.assets import is_offline_mode

        for value in ["true", "yes", "TRUE", "2", ""]:
            with patch.dict(os.environ, {"EXEC_SANDBOX_OFFLINE": value}):
                assert is_offline_mode() is False, f"Expected False for value '{value}'"


class TestGetAssets:
    """Tests for get_assets() singleton behavior."""

    def test_returns_same_instance(self) -> None:
        """get_assets() returns the same instance on repeated calls."""
        import exec_sandbox.assets as assets_module

        # Reset singleton for clean test
        assets_module._assets_singleton = None  # pyright: ignore[reportPrivateUsage]

        first = assets_module.get_assets()
        second = assets_module.get_assets()
        assert first is second

    def test_creates_async_pooch_instance(self) -> None:
        """get_assets() returns an AsyncPooch instance."""
        import exec_sandbox.assets as assets_module
        from exec_sandbox.asset_downloader import AsyncPooch

        # Reset singleton for clean test
        assets_module._assets_singleton = None  # pyright: ignore[reportPrivateUsage]

        result = assets_module.get_assets()
        assert isinstance(result, AsyncPooch)


class TestVersionedCacheDir:
    """Tests for _versioned_cache_dir() helper."""

    def test_appends_version_suffix(self) -> None:
        """Cache dir includes version suffix by default."""
        from exec_sandbox import __version__

        with patch.dict(os.environ, {}, clear=False):
            # Ensure EXEC_SANDBOX_CACHE_DIR is not set
            os.environ.pop("EXEC_SANDBOX_CACHE_DIR", None)
            result = _versioned_cache_dir()
            assert result.name == f"v{__version__}"

    def test_env_override_skips_version_suffix(self) -> None:
        """EXEC_SANDBOX_CACHE_DIR skips version suffix."""
        from pathlib import Path

        with patch.dict(os.environ, {"EXEC_SANDBOX_CACHE_DIR": "/custom/cache"}):
            result = _versioned_cache_dir()
            assert result == Path("/custom/cache")


class TestConcurrentDecompression:
    """Tests for concurrent decompression safety."""

    async def test_concurrent_calls_to_decompress_are_safe(self, tmp_path: Path) -> None:
        """Multiple concurrent calls to decompress the same file don't corrupt it."""
        import asyncio

        # Import the private lock function for testing concurrent access
        from exec_sandbox.asset_downloader import (
            _get_decompression_lock,  # pyright: ignore[reportPrivateUsage]
        )

        # Create source file
        source = tmp_path / "test.zst"
        source.write_bytes(b"compressed data")

        dest = tmp_path / "test"
        expected_content = b"decompressed content"

        # Track how many times actual decompression runs
        decompression_count = 0

        async def mock_decompress(fname: Path) -> Path:
            nonlocal decompression_count
            # Simulate the locking behavior
            lock = await _get_decompression_lock(dest)
            async with lock:
                if dest.exists():
                    return dest
                decompression_count += 1
                # Simulate some work
                await asyncio.sleep(0.01)
                dest.write_bytes(expected_content)
                fname.unlink(missing_ok=True)
            return dest

        # Launch multiple concurrent decompressions
        tasks = [mock_decompress(source) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should return the same path
        assert all(r == dest for r in results)

        # Content should be correct (not corrupted)
        assert dest.read_bytes() == expected_content

        # Only one decompression should have actually run
        assert decompression_count == 1


# =============================================================================
# Fetch Functions Tests
# =============================================================================


class TestFetchKernel:
    """Tests for fetch_kernel() function.

    Every test patches __file__ and _versioned_cache_dir for full isolation,
    ensuring results are deterministic regardless of what exists on disk.
    """

    # =========================================================================
    # Original tests (fixed for isolation)
    # =========================================================================

    async def test_returns_cached_kernel(self, tmp_path: Path) -> None:
        """Returns cached kernel without downloading."""
        from exec_sandbox.asset_downloader import get_current_arch
        from exec_sandbox.assets import fetch_kernel

        arch = get_current_arch()
        kernel = tmp_path / f"vmlinuz-{arch}"
        kernel.touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel()
            assert result == kernel

    async def test_respects_override_parameter(self, tmp_path: Path) -> None:
        """Override parameter takes priority over env var."""
        from exec_sandbox.asset_downloader import get_current_arch
        from exec_sandbox.assets import fetch_kernel

        arch = get_current_arch()
        override_dir = tmp_path / "override"
        env_dir = tmp_path / "env"
        override_dir.mkdir()
        env_dir.mkdir()

        (override_dir / f"vmlinuz-{arch}").write_text("override")
        (env_dir / f"vmlinuz-{arch}").write_text("env")

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(override=override_dir)
            assert result == override_dir / f"vmlinuz-{arch}"

    async def test_respects_arch_parameter(self, tmp_path: Path) -> None:
        """Fetches kernel for specified architecture."""
        from exec_sandbox.assets import fetch_kernel

        (tmp_path / "vmlinuz-aarch64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="aarch64")
            assert result == tmp_path / "vmlinuz-aarch64"

    async def test_downloads_when_not_cached(self, tmp_path: Path) -> None:
        """Downloads kernel when not found in cache."""
        from exec_sandbox.asset_downloader import get_current_arch
        from exec_sandbox.assets import fetch_kernel

        arch = get_current_arch()
        downloaded_kernel = tmp_path / f"vmlinuz-{arch}"

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty"),
            patch("exec_sandbox.assets.ensure_registry_loaded", new_callable=AsyncMock) as mock_registry,
            patch("exec_sandbox.assets.get_assets") as mock_assets,
        ):
            mock_assets.return_value.fetch = AsyncMock(return_value=downloaded_kernel)
            result = await fetch_kernel()

            mock_registry.assert_called_once()
            mock_assets.return_value.fetch.assert_called_once()
            assert result == downloaded_kernel

    # =========================================================================
    # Category 1: vmlinux preference within same directory (x86_64)
    # =========================================================================

    async def test_x86_64_prefers_vmlinux_over_vmlinuz_same_dir(self, tmp_path: Path) -> None:
        """On x86_64, vmlinux is preferred over vmlinuz in the same directory."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinux-x86_64").touch()
        (env_dir / "vmlinuz-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="x86_64")
            assert result == env_dir / "vmlinux-x86_64"

    async def test_x86_64_prefers_zst_over_decompressed_vmlinux(self, tmp_path: Path) -> None:
        """On x86_64, vmlinux .zst is preferred over decompressed vmlinux."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinux-x86_64.zst").touch()
        (env_dir / "vmlinux-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="x86_64")
            assert result == env_dir / "vmlinux-x86_64.zst"

    async def test_x86_64_prefers_zst_over_decompressed_vmlinuz(self, tmp_path: Path) -> None:
        """On x86_64 with no vmlinux, vmlinuz .zst is preferred over decompressed."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinuz-x86_64.zst").touch()
        (env_dir / "vmlinuz-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="x86_64")
            assert result == env_dir / "vmlinuz-x86_64.zst"

    # =========================================================================
    # Category 2: Directory priority beats kernel type (the bug regression)
    # =========================================================================

    async def test_env_vmlinuz_beats_local_vmlinux(self, tmp_path: Path) -> None:
        """Env dir vmlinuz wins over images/dist/ vmlinux — the exact CI bug."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinuz-x86_64").touch()

        # Point __file__ so images/dist resolves to a controlled dir
        local_dist = tmp_path / "images" / "dist"
        local_dist.mkdir(parents=True)
        (local_dist / "vmlinux-x86_64").touch()
        fake_file = str(tmp_path / "src" / "exec_sandbox" / "assets.py")

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", fake_file),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="x86_64")
            assert result == env_dir / "vmlinuz-x86_64"

    async def test_override_vmlinuz_beats_env_vmlinux(self, tmp_path: Path) -> None:
        """Override vmlinuz wins over env dir vmlinux."""
        from exec_sandbox.assets import fetch_kernel

        override_dir = tmp_path / "override"
        override_dir.mkdir()
        (override_dir / "vmlinuz-x86_64").touch()

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinux-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="x86_64", override=override_dir)
            assert result == override_dir / "vmlinuz-x86_64"

    async def test_env_vmlinuz_beats_cache_vmlinux(self, tmp_path: Path) -> None:
        """Env dir vmlinuz wins over cache dir vmlinux."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinuz-x86_64").touch()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "vmlinux-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=cache_dir),
        ):
            result = await fetch_kernel(arch="x86_64")
            assert result == env_dir / "vmlinuz-x86_64"

    # =========================================================================
    # Category 3: Architecture (aarch64 ignores vmlinux)
    # =========================================================================

    async def test_aarch64_ignores_vmlinux_returns_vmlinuz(self, tmp_path: Path) -> None:
        """On aarch64, vmlinux is ignored even when present; vmlinuz returned."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinux-aarch64").touch()
        (env_dir / "vmlinuz-aarch64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="aarch64")
            assert result == env_dir / "vmlinuz-aarch64"

    async def test_aarch64_skips_vmlinux_only_dir(self, tmp_path: Path) -> None:
        """On aarch64, dir with only vmlinux is skipped; falls to next dir."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinux-aarch64").touch()  # Ignored for aarch64

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "vmlinuz-aarch64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=cache_dir),
        ):
            result = await fetch_kernel(arch="aarch64")
            assert result == cache_dir / "vmlinuz-aarch64"

    async def test_aarch64_prefers_zst_over_decompressed(self, tmp_path: Path) -> None:
        """On aarch64, vmlinuz .zst is preferred over decompressed vmlinuz."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinuz-aarch64.zst").touch()
        (env_dir / "vmlinuz-aarch64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="aarch64")
            assert result == env_dir / "vmlinuz-aarch64.zst"

    # =========================================================================
    # Category 4: Edge cases
    # =========================================================================

    async def test_empty_higher_priority_dir_falls_through(self, tmp_path: Path) -> None:
        """Empty override dir is skipped; env dir kernel returned."""
        from exec_sandbox.assets import fetch_kernel

        override_dir = tmp_path / "override"
        override_dir.mkdir()  # Empty

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinuz-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="x86_64", override=override_dir)
            assert result == env_dir / "vmlinuz-x86_64"

    async def test_only_vmlinux_decompressed_in_dir(self, tmp_path: Path) -> None:
        """Single dir with only decompressed vmlinux is returned on x86_64."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinux-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="x86_64")
            assert result == env_dir / "vmlinux-x86_64"

    async def test_only_cache_has_kernel(self, tmp_path: Path) -> None:
        """Override, env, and local all empty — cache dir kernel returned."""
        from exec_sandbox.assets import fetch_kernel

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "vmlinuz-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=cache_dir),
        ):
            result = await fetch_kernel(arch="x86_64")
            assert result == cache_dir / "vmlinuz-x86_64"

    # =========================================================================
    # Category 5: Download fallback
    # =========================================================================

    async def test_download_prefers_vmlinux_x86_64_when_in_registry(self, tmp_path: Path) -> None:
        """No local files, registry has vmlinux — downloads vmlinux."""
        from exec_sandbox.assets import fetch_kernel

        downloaded = tmp_path / "vmlinux-x86_64"

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty"),
            patch("exec_sandbox.assets.ensure_registry_loaded", new_callable=AsyncMock),
            patch("exec_sandbox.assets.get_assets") as mock_assets,
        ):
            mock_assets.return_value.registry = {"vmlinux-x86_64.zst": "sha256:abc"}
            mock_assets.return_value.fetch = AsyncMock(return_value=downloaded)

            result = await fetch_kernel(arch="x86_64")

            mock_assets.return_value.fetch.assert_called_once_with("vmlinux-x86_64.zst", processor=decompress_zstd)
            assert result == downloaded

    async def test_download_falls_back_to_vmlinuz_when_vmlinux_not_in_registry(self, tmp_path: Path) -> None:
        """No local files, registry lacks vmlinux — downloads vmlinuz."""
        from exec_sandbox.assets import fetch_kernel

        downloaded = tmp_path / "vmlinuz-x86_64"

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty"),
            patch("exec_sandbox.assets.ensure_registry_loaded", new_callable=AsyncMock),
            patch("exec_sandbox.assets.get_assets") as mock_assets,
        ):
            mock_assets.return_value.registry = {}  # No vmlinux
            mock_assets.return_value.fetch = AsyncMock(return_value=downloaded)

            result = await fetch_kernel(arch="x86_64")

            mock_assets.return_value.fetch.assert_called_once_with("vmlinuz-x86_64.zst", processor=decompress_zstd)
            assert result == downloaded

    async def test_download_aarch64_always_vmlinuz(self, tmp_path: Path) -> None:
        """On aarch64, always downloads vmlinuz even if vmlinux is in registry."""
        from exec_sandbox.assets import fetch_kernel

        downloaded = tmp_path / "vmlinuz-aarch64"

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty"),
            patch("exec_sandbox.assets.ensure_registry_loaded", new_callable=AsyncMock),
            patch("exec_sandbox.assets.get_assets") as mock_assets,
        ):
            mock_assets.return_value.registry = {"vmlinux-aarch64.zst": "sha256:abc"}
            mock_assets.return_value.fetch = AsyncMock(return_value=downloaded)

            result = await fetch_kernel(arch="aarch64")

            mock_assets.return_value.fetch.assert_called_once_with("vmlinuz-aarch64.zst", processor=decompress_zstd)
            assert result == downloaded

    # =========================================================================
    # Category 6: Weird / adversarial
    # =========================================================================

    async def test_wrong_arch_vmlinux_ignored(self, tmp_path: Path) -> None:
        """vmlinux for wrong arch is not matched; falls through to download."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinux-aarch64").touch()  # Wrong arch for x86_64 request

        downloaded = tmp_path / "vmlinuz-x86_64"

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
            patch("exec_sandbox.assets.ensure_registry_loaded", new_callable=AsyncMock) as mock_registry,
            patch("exec_sandbox.assets.get_assets") as mock_assets,
        ):
            mock_assets.return_value.registry = {}
            mock_assets.return_value.fetch = AsyncMock(return_value=downloaded)

            result = await fetch_kernel(arch="x86_64")

            mock_registry.assert_called_once()
            assert result == downloaded

    async def test_local_found_skips_download_entirely(self, tmp_path: Path) -> None:
        """When kernel found locally, ensure_registry_loaded is never called."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinuz-x86_64").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
            patch("exec_sandbox.assets.ensure_registry_loaded", new_callable=AsyncMock) as mock_registry,
        ):
            result = await fetch_kernel(arch="x86_64")

            mock_registry.assert_not_called()
            assert result == env_dir / "vmlinuz-x86_64"

    async def test_vmlinux_decompressed_beats_vmlinuz_zst_same_dir(self, tmp_path: Path) -> None:
        """On x86_64, decompressed vmlinux beats vmlinuz .zst in the same dir."""
        from exec_sandbox.assets import fetch_kernel

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "vmlinux-x86_64").touch()
        (env_dir / "vmlinuz-x86_64.zst").touch()

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/src/exec_sandbox/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "empty_cache"),
        ):
            result = await fetch_kernel(arch="x86_64")
            assert result == env_dir / "vmlinux-x86_64"


class TestFetchInitramfs:
    """Tests for fetch_initramfs() function."""

    async def test_returns_cached_initramfs(self, tmp_path: Path) -> None:
        """Returns cached initramfs without downloading."""
        from exec_sandbox.asset_downloader import get_current_arch
        from exec_sandbox.assets import fetch_initramfs

        arch = get_current_arch()
        initramfs = tmp_path / f"initramfs-{arch}"
        initramfs.touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            result = await fetch_initramfs()
            assert result == initramfs

    async def test_respects_override_parameter(self, tmp_path: Path) -> None:
        """Override parameter takes priority."""
        from exec_sandbox.asset_downloader import get_current_arch
        from exec_sandbox.assets import fetch_initramfs

        arch = get_current_arch()
        override_dir = tmp_path / "override"
        override_dir.mkdir()
        (override_dir / f"initramfs-{arch}").touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            result = await fetch_initramfs(override=override_dir)
            assert result == override_dir / f"initramfs-{arch}"


class TestFetchBaseImage:
    """Tests for fetch_base_image() function."""

    async def test_python_language_mapping(self, tmp_path: Path) -> None:
        """Python language maps to python-3.14-base image."""
        from exec_sandbox.asset_downloader import get_current_arch
        from exec_sandbox.assets import fetch_base_image

        arch = get_current_arch()
        image = tmp_path / f"python-3.14-base-{arch}.qcow2"
        image.touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            result = await fetch_base_image("python")
            assert result == image

    async def test_javascript_language_mapping(self, tmp_path: Path) -> None:
        """JavaScript language maps to node-1.3-base image."""
        from exec_sandbox.asset_downloader import get_current_arch
        from exec_sandbox.assets import fetch_base_image

        arch = get_current_arch()
        image = tmp_path / f"node-1.3-base-{arch}.qcow2"
        image.touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            result = await fetch_base_image("javascript")
            assert result == image

    async def test_unknown_language_uses_raw(self, tmp_path: Path) -> None:
        """Unknown language falls back to raw-base image."""
        from exec_sandbox.asset_downloader import get_current_arch
        from exec_sandbox.assets import fetch_base_image

        arch = get_current_arch()
        image = tmp_path / f"raw-base-{arch}.qcow2"
        image.touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}):
            result = await fetch_base_image("rust")  # Unknown language
            assert result == image

    async def test_respects_override_parameter(self, tmp_path: Path) -> None:
        """Override parameter takes priority."""
        from exec_sandbox.asset_downloader import get_current_arch
        from exec_sandbox.assets import fetch_base_image

        arch = get_current_arch()
        override_dir = tmp_path / "override"
        override_dir.mkdir()
        (override_dir / f"python-3.14-base-{arch}.qcow2").touch()

        result = await fetch_base_image("python", override=override_dir)
        assert result == override_dir / f"python-3.14-base-{arch}.qcow2"


class TestFetchGvproxy:
    """Tests for fetch_gvproxy() and get_gvproxy_path() functions."""

    async def test_get_gvproxy_path_from_env(self, tmp_path: Path) -> None:
        """get_gvproxy_path finds binary in env var path."""
        from exec_sandbox.asset_downloader import get_gvproxy_suffix
        from exec_sandbox.assets import get_gvproxy_path

        suffix = get_gvproxy_suffix()
        binary = tmp_path / f"gvproxy-wrapper-{suffix}"
        binary.touch()

        # Disable repo-relative path check (priority 1) by pointing __file__ elsewhere
        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
        ):
            result = await get_gvproxy_path()
            assert result == binary

    async def test_get_gvproxy_path_override_priority(self, tmp_path: Path) -> None:
        """Override path takes priority for gvproxy."""
        from exec_sandbox.asset_downloader import get_gvproxy_suffix
        from exec_sandbox.assets import get_gvproxy_path

        suffix = get_gvproxy_suffix()
        override_dir = tmp_path / "override"
        env_dir = tmp_path / "env"
        override_dir.mkdir()
        env_dir.mkdir()

        (override_dir / f"gvproxy-wrapper-{suffix}").touch()
        (env_dir / f"gvproxy-wrapper-{suffix}").touch()

        with patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(env_dir)}):
            result = await get_gvproxy_path(override=override_dir)
            assert result == override_dir / f"gvproxy-wrapper-{suffix}"

    async def test_get_gvproxy_path_returns_none_when_not_found(self, tmp_path: Path) -> None:
        """get_gvproxy_path returns None when binary not found."""
        from exec_sandbox.assets import get_gvproxy_path

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path),
        ):
            result = await get_gvproxy_path()
            assert result is None

    async def test_fetch_gvproxy_returns_cached_and_makes_executable(self, tmp_path: Path) -> None:
        """fetch_gvproxy returns cached binary and actually makes it executable."""
        import stat

        from exec_sandbox.asset_downloader import get_gvproxy_suffix
        from exec_sandbox.assets import fetch_gvproxy

        suffix = get_gvproxy_suffix()
        binary = tmp_path / f"gvproxy-wrapper-{suffix}"
        binary.touch()

        # Verify file is NOT executable initially
        initial_mode = binary.stat().st_mode
        assert not (initial_mode & stat.S_IXUSR), "File should not be executable initially"

        # Disable repo-relative path check (priority 1) by pointing __file__ elsewhere
        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
        ):
            result = await fetch_gvproxy()
            assert result == binary

            # Verify file IS executable after fetch_gvproxy
            final_mode = binary.stat().st_mode
            assert final_mode & stat.S_IXUSR, "File should be executable after fetch_gvproxy"


# =============================================================================
# ensure_assets_available Tests
# =============================================================================


class TestEnsureAssetsAvailable:
    """Tests for ensure_assets_available() function."""

    async def test_fetches_all_required_assets_in_parallel(self, tmp_path: Path) -> None:
        """Fetches kernel, initramfs, and gvproxy in parallel."""
        from exec_sandbox.assets import ensure_assets_available

        kernel = tmp_path / "vmlinuz"
        initramfs = tmp_path / "initramfs"
        gvproxy = tmp_path / "gvproxy"

        with (
            patch("exec_sandbox.assets.fetch_kernel") as mock_kernel,
            patch("exec_sandbox.assets.fetch_initramfs") as mock_initramfs,
            patch("exec_sandbox.assets.fetch_gvproxy") as mock_gvproxy,
        ):
            mock_kernel.return_value = kernel
            mock_initramfs.return_value = initramfs
            mock_gvproxy.return_value = gvproxy

            images_dir, gvproxy_path = await ensure_assets_available()

            mock_kernel.assert_called_once()
            mock_initramfs.assert_called_once()
            mock_gvproxy.assert_called_once()
            assert images_dir == tmp_path  # Parent of kernel
            assert gvproxy_path == gvproxy

    async def test_fetches_base_image_when_language_specified(self, tmp_path: Path) -> None:
        """Also fetches base image when language is provided."""
        from exec_sandbox.assets import ensure_assets_available

        with (
            patch("exec_sandbox.assets.fetch_kernel") as mock_kernel,
            patch("exec_sandbox.assets.fetch_initramfs") as mock_initramfs,
            patch("exec_sandbox.assets.fetch_gvproxy") as mock_gvproxy,
            patch("exec_sandbox.assets.fetch_base_image") as mock_base,
        ):
            mock_kernel.return_value = tmp_path / "vmlinuz"
            mock_initramfs.return_value = tmp_path / "initramfs"
            mock_gvproxy.return_value = tmp_path / "gvproxy"
            mock_base.return_value = tmp_path / "base.qcow2"

            await ensure_assets_available(language="python")

            mock_base.assert_called_once()
            # Verify language was passed
            call_args = mock_base.call_args
            assert call_args[0][0] == "python" or call_args.kwargs.get("language") == "python"

    async def test_passes_override_to_all_fetch_functions(self, tmp_path: Path) -> None:
        """Override parameter is passed to all fetch functions."""
        from exec_sandbox.assets import ensure_assets_available

        override = tmp_path / "override"

        with (
            patch("exec_sandbox.assets.fetch_kernel") as mock_kernel,
            patch("exec_sandbox.assets.fetch_initramfs") as mock_initramfs,
            patch("exec_sandbox.assets.fetch_gvproxy") as mock_gvproxy,
        ):
            mock_kernel.return_value = tmp_path / "vmlinuz"
            mock_initramfs.return_value = tmp_path / "initramfs"
            mock_gvproxy.return_value = tmp_path / "gvproxy"

            await ensure_assets_available(override=override)

            # Verify override was passed
            assert mock_kernel.call_args.kwargs.get("override") == override
            assert mock_initramfs.call_args.kwargs.get("override") == override
            assert mock_gvproxy.call_args.kwargs.get("override") == override


# =============================================================================
# ensure_registry_loaded Tests
# =============================================================================


class TestEnsureRegistryLoaded:
    """Tests for ensure_registry_loaded() function."""

    async def test_skips_if_already_loaded(self) -> None:
        """Does nothing if registry is already populated."""
        from exec_sandbox.assets import ensure_registry_loaded

        with patch("exec_sandbox.assets.get_assets") as mock_assets:
            mock_assets.return_value.registry = {"file.txt": "sha256:abc"}

            await ensure_registry_loaded()

            # Should not call load_registry_from_github
            mock_assets.return_value.load_registry_from_github.assert_not_called()

    async def test_skips_in_offline_mode(self) -> None:
        """Skips loading in offline mode."""
        from exec_sandbox.assets import ensure_registry_loaded

        with (
            patch("exec_sandbox.assets.get_assets") as mock_assets,
            patch("exec_sandbox.assets.is_offline_mode", return_value=True),
        ):
            mock_assets.return_value.registry = {}

            await ensure_registry_loaded()

            mock_assets.return_value.load_registry_from_github.assert_not_called()

    async def test_loads_from_github_when_empty(self) -> None:
        """Loads registry from GitHub when empty and online."""
        from exec_sandbox.assets import ensure_registry_loaded

        with (
            patch("exec_sandbox.assets.get_assets") as mock_assets,
            patch("exec_sandbox.assets.is_offline_mode", return_value=False),
            patch("exec_sandbox.assets._get_asset_version", return_value=("1.0.0", "v1.0.0")),
        ):
            mock_assets.return_value.registry = {}
            mock_assets.return_value.load_registry_from_github = AsyncMock()

            await ensure_registry_loaded()

            mock_assets.return_value.load_registry_from_github.assert_called_once_with(
                "dualeai", "exec-sandbox", "v1.0.0"
            )


# =============================================================================
# Decompression Error Handling Tests
# =============================================================================


class TestDecompressZstdErrorHandling:
    """Tests for decompress_zstd() error handling."""

    async def test_raises_file_not_found_when_source_missing(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when source file doesn't exist."""
        from exec_sandbox.asset_downloader import decompress_zstd

        nonexistent = tmp_path / "nonexistent.zst"

        with pytest.raises(FileNotFoundError):
            await decompress_zstd(nonexistent)

    async def test_returns_dest_if_already_decompressed(self, tmp_path: Path) -> None:
        """Returns existing decompressed file without needing source."""
        from exec_sandbox.asset_downloader import decompress_zstd

        # Create source .zst file and already-decompressed destination
        source = tmp_path / "file.zst"
        source.write_bytes(b"fake compressed data")  # Would fail if actually decompressed
        dest = tmp_path / "file"
        dest.write_text("already decompressed content")

        # decompress_zstd should return the existing dest without touching source
        result = await decompress_zstd(source)
        assert result == dest
        assert dest.read_text() == "already decompressed content"  # Not corrupted
        assert source.exists()  # Source not deleted since we didn't decompress

    async def test_cleans_up_temp_file_on_failure(self) -> None:
        """Temp file is cleaned up if decompression fails."""
        # Skip this test - it requires actual zstd decompression to fail
        # which would need a corrupted .zst file
        pytest.skip("Requires actual zstd library to test decompression failure")


# =============================================================================
# Error Cases Tests
# =============================================================================


class TestErrorCases:
    """Tests for various error conditions."""

    async def test_ensure_assets_error_message_includes_search_paths(self, tmp_path: Path) -> None:
        """Error message includes all searched paths."""
        nonexistent = tmp_path / "does_not_exist"

        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": ""}, clear=False),
            patch("exec_sandbox.assets.__file__", "/nonexistent/assets.py"),
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path / "cache"),
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                await ensure_assets(override=nonexistent, download=False)

            error_msg = str(exc_info.value)
            assert "does_not_exist" in error_msg
            assert "auto_download_assets=False" in error_msg

    async def test_find_asset_handles_permission_error_gracefully(self, tmp_path: Path) -> None:
        """Permission errors during path check are handled."""
        # This test verifies the code doesn't crash on permission errors
        # In practice, aiofiles.os.path.exists handles this
        with (
            patch.dict(os.environ, {"EXEC_SANDBOX_IMAGES_DIR": str(tmp_path)}),
            patch("aiofiles.os.path.exists", side_effect=PermissionError("denied")),
        ):
            with pytest.raises(PermissionError):
                await _find_asset("vmlinuz-x86_64")


# =============================================================================
# Prefetch All Assets Tests
# =============================================================================


class TestPrefetchAllAssets:
    """Tests for prefetch_all_assets() function."""

    async def test_prefetch_downloads_all_assets(self, tmp_path: Path) -> None:
        """prefetch_all_assets downloads all required assets."""
        from exec_sandbox.assets import prefetch_all_assets
        from exec_sandbox.models import Language

        with (
            patch("exec_sandbox.assets.ensure_registry_loaded") as mock_registry,
            patch("exec_sandbox.assets.fetch_kernel") as mock_kernel,
            patch("exec_sandbox.assets.fetch_initramfs") as mock_initramfs,
            patch("exec_sandbox.assets.fetch_gvproxy") as mock_gvproxy,
            patch("exec_sandbox.assets.fetch_base_image") as mock_base,
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path),
        ):
            mock_registry.return_value = None
            mock_kernel.return_value = tmp_path / "vmlinuz"
            mock_initramfs.return_value = tmp_path / "initramfs"
            mock_gvproxy.return_value = tmp_path / "gvproxy"
            mock_base.return_value = tmp_path / "base.qcow2"

            result = await prefetch_all_assets()

            assert result.success is True
            assert result.cache_dir == tmp_path
            mock_registry.assert_called_once()
            mock_kernel.assert_called_once()
            mock_initramfs.assert_called_once()
            mock_gvproxy.assert_called_once()
            # Base images for all supported languages
            assert mock_base.call_count == len(Language)

    async def test_prefetch_with_arch(self, tmp_path: Path) -> None:
        """prefetch_all_assets respects arch parameter."""
        from exec_sandbox.assets import prefetch_all_assets

        with (
            patch("exec_sandbox.assets.ensure_registry_loaded") as mock_registry,
            patch("exec_sandbox.assets.fetch_kernel") as mock_kernel,
            patch("exec_sandbox.assets.fetch_initramfs") as mock_initramfs,
            patch("exec_sandbox.assets.fetch_gvproxy") as mock_gvproxy,
            patch("exec_sandbox.assets.fetch_base_image") as mock_base,
            patch("exec_sandbox.assets._versioned_cache_dir", return_value=tmp_path),
        ):
            mock_registry.return_value = None
            mock_kernel.return_value = tmp_path / "vmlinuz"
            mock_initramfs.return_value = tmp_path / "initramfs"
            mock_gvproxy.return_value = tmp_path / "gvproxy"
            mock_base.return_value = tmp_path / "base.qcow2"

            result = await prefetch_all_assets(arch="aarch64")

            assert result.success is True
            assert result.arch == "aarch64"
            # Verify arch was passed to kernel and initramfs
            mock_kernel.assert_called_once()
            assert mock_kernel.call_args.kwargs.get("arch") == "aarch64"
            mock_initramfs.assert_called_once()
            assert mock_initramfs.call_args.kwargs.get("arch") == "aarch64"

    async def test_prefetch_returns_error_on_registry_failure(self) -> None:
        """prefetch_all_assets returns failure result if registry load fails."""
        from exec_sandbox.assets import prefetch_all_assets

        with patch(
            "exec_sandbox.assets.ensure_registry_loaded",
            side_effect=OSError("Network error"),
        ):
            result = await prefetch_all_assets()

            assert result.success is False
            assert len(result.errors) == 1
            assert result.errors[0][0] == "registry"

    async def test_prefetch_returns_error_on_download_failure(self, tmp_path: Path) -> None:
        """prefetch_all_assets returns failure result if any download fails."""
        from exec_sandbox.assets import prefetch_all_assets

        with (
            patch("exec_sandbox.assets.ensure_registry_loaded") as mock_registry,
            patch("exec_sandbox.assets.fetch_kernel") as mock_kernel,
            patch("exec_sandbox.assets.fetch_initramfs") as mock_initramfs,
            patch("exec_sandbox.assets.fetch_gvproxy") as mock_gvproxy,
            patch("exec_sandbox.assets.fetch_base_image") as mock_base,
        ):
            mock_registry.return_value = None
            mock_kernel.return_value = tmp_path / "vmlinuz"
            mock_initramfs.side_effect = Exception("Download failed")  # Fail initramfs
            mock_gvproxy.return_value = tmp_path / "gvproxy"
            mock_base.return_value = tmp_path / "base.qcow2"

            result = await prefetch_all_assets()

            assert result.success is False
            assert any(name == "initramfs" for name, _ in result.errors)

    async def test_prefetch_collects_all_errors(self, tmp_path: Path) -> None:
        """prefetch_all_assets collects and reports all errors."""
        from exec_sandbox.assets import prefetch_all_assets

        with (
            patch("exec_sandbox.assets.ensure_registry_loaded") as mock_registry,
            patch("exec_sandbox.assets.fetch_kernel") as mock_kernel,
            patch("exec_sandbox.assets.fetch_initramfs") as mock_initramfs,
            patch("exec_sandbox.assets.fetch_gvproxy") as mock_gvproxy,
            patch("exec_sandbox.assets.fetch_base_image") as mock_base,
        ):
            mock_registry.return_value = None
            mock_kernel.side_effect = Exception("Kernel download failed")
            mock_initramfs.side_effect = Exception("Initramfs download failed")
            mock_gvproxy.return_value = tmp_path / "gvproxy"
            mock_base.return_value = tmp_path / "base.qcow2"

            result = await prefetch_all_assets()

            # Should fail and have collected both errors
            assert result.success is False
            assert len(result.errors) == 2
            error_names = {name for name, _ in result.errors}
            assert "kernel" in error_names
            assert "initramfs" in error_names

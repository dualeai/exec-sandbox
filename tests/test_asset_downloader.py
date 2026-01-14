"""Tests for asset_downloader module."""

from __future__ import annotations

import hashlib
import sys
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import aiohttp
import pytest
from aioresponses import aioresponses

from exec_sandbox.asset_downloader import (
    AsyncPooch,
    decompress_zstd,
    get_cache_dir,
    get_current_arch,
    get_gvproxy_suffix,
    os_cache,
    retrieve,
    untar,
)
from exec_sandbox.exceptions import AssetDownloadError, AssetNotFoundError
from exec_sandbox.platform_utils import HostOS


class TestGetCacheDir:
    """Tests for get_cache_dir function."""

    def test_returns_path(self):
        """Should return a Path object."""
        result = get_cache_dir()
        assert isinstance(result, Path)

    def test_default_app_name(self):
        """Should use exec-sandbox as default app name."""
        result = get_cache_dir()
        assert "exec-sandbox" in str(result)

    def test_custom_app_name(self):
        """Should use custom app name when provided."""
        result = get_cache_dir("custom-app")
        assert "custom-app" in str(result)

    def test_env_override(self):
        """Should respect EXEC_SANDBOX_CACHE_DIR environment variable."""
        with patch.dict("os.environ", {"EXEC_SANDBOX_CACHE_DIR": "/custom/path"}):
            result = get_cache_dir()
            assert result == Path("/custom/path")

    def test_darwin_platform(self):
        """Should use Library/Caches on macOS."""
        with patch("exec_sandbox.asset_downloader.detect_host_os", return_value=HostOS.MACOS):
            with patch.dict("os.environ", {}, clear=True):
                result = get_cache_dir()
                assert "Library/Caches" in str(result)

    def test_linux_platform(self):
        """Should use .cache on Linux."""
        with patch("exec_sandbox.asset_downloader.detect_host_os", return_value=HostOS.LINUX):
            with patch.dict("os.environ", {}, clear=True):
                result = get_cache_dir()
                assert ".cache" in str(result)

    def test_linux_xdg_cache_home(self):
        """Should respect XDG_CACHE_HOME on Linux."""
        with patch("exec_sandbox.asset_downloader.detect_host_os", return_value=HostOS.LINUX):
            with patch.dict("os.environ", {"XDG_CACHE_HOME": "/xdg/cache"}):
                result = get_cache_dir()
                assert result == Path("/xdg/cache/exec-sandbox")

    def test_unknown_platform_fallback(self):
        """Should fall back to .cache on unknown platforms."""
        with patch("exec_sandbox.asset_downloader.detect_host_os", return_value=HostOS.UNKNOWN):
            with patch.dict("os.environ", {}, clear=True):
                result = get_cache_dir()
                assert ".cache" in str(result)


class TestOsCache:
    """Tests for os_cache alias function."""

    def test_alias_for_get_cache_dir(self):
        """os_cache should be an alias for get_cache_dir."""
        assert os_cache("test") == get_cache_dir("test")


class TestGetCurrentArch:
    """Tests for get_current_arch function."""

    def test_x86_64(self):
        """Should return x86_64 for x86_64 architecture."""
        with patch("exec_sandbox.asset_downloader.get_arch_name", return_value="x86_64"):
            assert get_current_arch() == "x86_64"

    def test_aarch64(self):
        """Should return aarch64 for ARM64 architecture."""
        with patch("exec_sandbox.asset_downloader.get_arch_name", return_value="aarch64"):
            assert get_current_arch() == "aarch64"


class TestGetGvproxySuffix:
    """Tests for get_gvproxy_suffix function."""

    def test_darwin_arm64(self):
        """Should return darwin-arm64 for macOS ARM."""
        with patch("exec_sandbox.asset_downloader.get_os_name", return_value="darwin"):
            with patch("exec_sandbox.asset_downloader.get_arch_name", return_value="arm64"):
                assert get_gvproxy_suffix() == "darwin-arm64"

    def test_darwin_amd64(self):
        """Should return darwin-amd64 for macOS Intel."""
        with patch("exec_sandbox.asset_downloader.get_os_name", return_value="darwin"):
            with patch("exec_sandbox.asset_downloader.get_arch_name", return_value="amd64"):
                assert get_gvproxy_suffix() == "darwin-amd64"

    def test_linux_arm64(self):
        """Should return linux-arm64 for Linux ARM."""
        with patch("exec_sandbox.asset_downloader.get_os_name", return_value="linux"):
            with patch("exec_sandbox.asset_downloader.get_arch_name", return_value="arm64"):
                assert get_gvproxy_suffix() == "linux-arm64"

    def test_linux_amd64(self):
        """Should return linux-amd64 for Linux x86."""
        with patch("exec_sandbox.asset_downloader.get_os_name", return_value="linux"):
            with patch("exec_sandbox.asset_downloader.get_arch_name", return_value="amd64"):
                assert get_gvproxy_suffix() == "linux-amd64"


class TestRetrieve:
    """Tests for retrieve function."""

    async def test_downloads_and_caches(self, tmp_path: Path):
        """Should download file and cache it locally."""
        content = b"hello world"
        content_hash = hashlib.sha256(content).hexdigest()

        with aioresponses() as m:
            m.get(
                "https://example.com/test.txt",
                body=content,
            )

            path = await retrieve(
                url="https://example.com/test.txt",
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
            )

            assert path.exists()
            assert path.read_bytes() == content

    async def test_uses_cache_on_second_call(self, tmp_path: Path):
        """Should use cached file on second call without re-downloading."""
        content = b"cached content"
        content_hash = hashlib.sha256(content).hexdigest()

        with aioresponses() as m:
            m.get("https://example.com/cached.txt", body=content)

            # First call - downloads
            path1 = await retrieve(
                url="https://example.com/cached.txt",
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
            )

            # Second call - should use cache (no mock needed)
            path2 = await retrieve(
                url="https://example.com/cached.txt",
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
            )

            assert path1 == path2
            # aioresponses should only have been called once
            m.assert_called_once()

    async def test_checksum_verification_fails(self, tmp_path: Path):
        """Should raise AssetDownloadError (wrapping AssetChecksumError) on hash mismatch after retries."""
        with aioresponses() as m:
            # Mock all retry attempts with wrong content
            for _ in range(3):
                m.get("https://example.com/bad.txt", body=b"actual content")

            with pytest.raises(AssetDownloadError):  # Wraps AssetChecksumError after retries
                await retrieve(
                    url="https://example.com/bad.txt",
                    known_hash="sha256:wronghash",
                    path=tmp_path,
                )

    @pytest.mark.skip(reason="aioresponses doesn't properly support sequential error-then-success mocking")
    async def test_retries_on_failure(self, tmp_path: Path):
        """Should retry on network failure."""
        # Note: This test is skipped because aioresponses has limitations with
        # mocking sequential responses where the first fails and subsequent succeed.
        # The retry logic is tested implicitly by test_checksum_verification_fails.

    async def test_http_404_error(self, tmp_path: Path):
        """Should raise AssetDownloadError on HTTP 404."""
        with aioresponses() as m:
            for _ in range(3):  # All retry attempts return 404
                m.get("https://example.com/notfound.txt", status=404)

            with pytest.raises(AssetDownloadError):
                await retrieve(
                    url="https://example.com/notfound.txt",
                    known_hash="sha256:abc123",
                    path=tmp_path,
                )

    async def test_http_500_error(self, tmp_path: Path):
        """Should raise AssetDownloadError on HTTP 500."""
        with aioresponses() as m:
            for _ in range(3):  # All retry attempts return 500
                m.get("https://example.com/error.txt", status=500)

            with pytest.raises(AssetDownloadError):
                await retrieve(
                    url="https://example.com/error.txt",
                    known_hash="sha256:abc123",
                    path=tmp_path,
                )

    async def test_connection_error(self, tmp_path: Path):
        """Should raise AssetDownloadError on connection failure."""
        with aioresponses() as m:
            for _ in range(3):  # All retry attempts fail
                m.get(
                    "https://example.com/timeout.txt",
                    exception=aiohttp.ClientConnectionError("Connection refused"),
                )

            with pytest.raises(AssetDownloadError):
                await retrieve(
                    url="https://example.com/timeout.txt",
                    known_hash="sha256:abc123",
                    path=tmp_path,
                )

    async def test_empty_hash_skips_verification(self, tmp_path: Path):
        """Should skip hash verification when hash is empty."""
        content = b"any content"

        with aioresponses() as m:
            m.get("https://example.com/nohash.txt", body=content)

            # Empty hash should succeed without verification
            path = await retrieve(
                url="https://example.com/nohash.txt",
                known_hash="",
                path=tmp_path,
            )

            assert path.exists()
            assert path.read_bytes() == content

    async def test_processor_is_called(self, tmp_path: Path):
        """Should call processor function after download."""
        content = b"original content"
        content_hash = hashlib.sha256(content).hexdigest()

        # Track if processor was called
        processor_called = False
        processor_input = None

        async def mock_processor(path: Path) -> Path:
            nonlocal processor_called, processor_input
            processor_called = True
            processor_input = path
            # Create processed file
            processed_path = path.with_suffix(".processed")
            processed_path.write_bytes(b"processed")
            return processed_path

        with aioresponses() as m:
            m.get("https://example.com/process.txt", body=content)

            result_path = await retrieve(
                url="https://example.com/process.txt",
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
                processor=mock_processor,
            )

            assert processor_called
            assert processor_input == tmp_path / "process.txt"
            assert result_path == tmp_path / "process.processed"

    async def test_redownloads_on_hash_mismatch(self, tmp_path: Path):
        """Should re-download when cached file has wrong hash."""
        content = b"correct content"
        content_hash = hashlib.sha256(content).hexdigest()

        # Pre-populate cache with wrong content
        cached_file = tmp_path / "cached.txt"
        cached_file.write_bytes(b"wrong content in cache")

        with aioresponses() as m:
            m.get("https://example.com/cached.txt", body=content)

            path = await retrieve(
                url="https://example.com/cached.txt",
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
            )

            # Should have re-downloaded and now have correct content
            assert path.read_bytes() == content

    async def test_streaming_memory_usage(self, tmp_path: Path):
        """Should use bounded memory (~64KB chunks) regardless of file size.

        Tests _verify_hash directly with a large file to ensure streaming
        hash computation uses bounded memory.
        """
        from exec_sandbox.asset_downloader import _verify_hash

        # Create a 4MB file on disk
        file_size = 4 * 1024 * 1024  # 4MB
        large_file = tmp_path / "large.bin"

        # Write file in chunks to avoid memory spike during setup
        chunk = b"x" * (64 * 1024)  # 64KB chunks
        hasher = hashlib.sha256()
        with large_file.open("wb") as f:
            for _ in range(file_size // len(chunk)):
                f.write(chunk)
                hasher.update(chunk)
        expected_hash = hasher.hexdigest()

        # Measure peak memory during hash verification
        tracemalloc.start()
        result = await _verify_hash(large_file, f"sha256:{expected_hash}")
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert result is True

        # Peak memory should be bounded by:
        # - 64KB chunk buffer
        # - ~130KB Python file I/O baseline overhead
        # - hashlib internal state
        # - asyncio.to_thread overhead (~100KB)
        # - Free-threaded Python (3.14t+) has ~150KB additional overhead
        # Empirically: ~350KB for 64KB chunks, ~550KB on 3.14t
        max_allowed = 600 * 1024  # 600KB - proves streaming works for 4MB file
        assert peak_memory < max_allowed, (
            f"Peak memory {peak_memory / 1024:.1f}KB exceeded {max_allowed / 1024:.1f}KB limit "
            f"for {file_size / 1024 / 1024:.0f}MB file. Streaming may be broken."
        )


class TestAsyncPooch:
    """Tests for AsyncPooch class."""

    def test_init(self, tmp_path: Path):
        """Should initialize with provided configuration."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://example.com/v{version}",
            version="1.0.0",
            registry={"file.txt": "sha256:abc123"},
        )

        assert pooch.path == tmp_path
        assert pooch.version == "1.0.0"
        assert "file.txt" in pooch.registry

    def test_env_override_path(self, tmp_path: Path):
        """Should respect environment variable for path override."""
        with patch.dict("os.environ", {"CUSTOM_ENV": "/custom/path"}):
            pooch = AsyncPooch(
                path=tmp_path,
                base_url="https://example.com",
                version="1.0.0",
                env="CUSTOM_ENV",
            )
            assert pooch.path == Path("/custom/path")

    async def test_fetch_downloads_file(self, tmp_path: Path):
        """Should download file from registry."""
        content = b"registry content"
        content_hash = hashlib.sha256(content).hexdigest()

        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://example.com/v{version}",
            version="1.0.0",
            registry={"file.txt": f"sha256:{content_hash}"},
        )

        with aioresponses() as m:
            m.get("https://example.com/v1.0.0/file.txt", body=content)

            path = await pooch.fetch("file.txt")

            assert path.exists()
            assert path.read_bytes() == content

    async def test_fetch_not_in_registry(self, tmp_path: Path):
        """Should raise AssetNotFoundError for unknown file."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://example.com",
            version="1.0.0",
            registry={},
        )

        with pytest.raises(AssetNotFoundError):
            await pooch.fetch("unknown.txt")

    async def test_load_registry_from_file(self, tmp_path: Path):
        """Should load registry from file."""
        registry_file = tmp_path / "registry.txt"
        registry_file.write_text("file1.txt sha256:hash1\nfile2.txt sha256:hash2\n")

        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://example.com",
            version="1.0.0",
        )
        await pooch.load_registry_from_file(registry_file)

        assert pooch.registry["file1.txt"] == "sha256:hash1"
        assert pooch.registry["file2.txt"] == "sha256:hash2"

    async def test_load_registry_from_github(self, tmp_path: Path):
        """Should load registry from GitHub release API."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0",
        )

        github_response = {
            "tag_name": "v1.0.0",
            "assets": [
                {"name": "file1.txt", "digest": "sha256:hash1"},
                {"name": "file2.txt", "digest": "sha256:hash2"},
            ],
        }

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/tags/v1.0.0",
                payload=github_response,
            )

            await pooch.load_registry_from_github("owner", "repo", "v1.0.0")

            assert pooch.registry["file1.txt"] == "sha256:hash1"
            assert pooch.registry["file2.txt"] == "sha256:hash2"

    async def test_load_registry_with_comments_and_empty_lines(self, tmp_path: Path):
        """Should skip comments and empty lines in registry file."""
        registry_file = tmp_path / "registry.txt"
        registry_file.write_text(
            "# This is a comment\n\nfile1.txt sha256:hash1\n   \n# Another comment\nfile2.txt sha256:hash2\n"
        )

        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://example.com",
            version="1.0.0",
        )
        await pooch.load_registry_from_file(registry_file)

        assert len(pooch.registry) == 2
        assert pooch.registry["file1.txt"] == "sha256:hash1"
        assert pooch.registry["file2.txt"] == "sha256:hash2"

    async def test_load_registry_with_malformed_lines(self, tmp_path: Path):
        """Should skip malformed lines (less than 2 parts) in registry file."""
        registry_file = tmp_path / "registry.txt"
        registry_file.write_text("file1.txt sha256:hash1\nmalformed_line_no_hash\nfile2.txt sha256:hash2\n")

        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://example.com",
            version="1.0.0",
        )
        await pooch.load_registry_from_file(registry_file)

        assert len(pooch.registry) == 2
        assert "malformed_line_no_hash" not in pooch.registry

    async def test_github_api_404_error(self, tmp_path: Path):
        """Should raise AssetNotFoundError when GitHub release not found."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0",
        )

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/tags/v1.0.0",
                status=404,
            )

            with pytest.raises(AssetNotFoundError):
                await pooch.load_registry_from_github("owner", "repo", "v1.0.0")

    async def test_github_assets_without_digest(self, tmp_path: Path):
        """Should handle assets without digest (older releases)."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0",
        )

        github_response = {
            "tag_name": "v1.0.0",
            "assets": [
                {"name": "file1.txt", "digest": "sha256:hash1"},
                {"name": "file2.txt"},  # No digest
            ],
        }

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/tags/v1.0.0",
                payload=github_response,
            )

            await pooch.load_registry_from_github("owner", "repo", "v1.0.0")

            assert pooch.registry["file1.txt"] == "sha256:hash1"
            assert pooch.registry["file2.txt"] == ""  # Empty hash for missing digest

    async def test_github_latest_tag(self, tmp_path: Path):
        """Should use /releases/latest endpoint for 'latest' tag."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0",
        )

        github_response = {
            "tag_name": "v2.0.0",
            "assets": [{"name": "file.txt", "digest": "sha256:hash"}],
        }

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/latest",
                payload=github_response,
            )

            await pooch.load_registry_from_github("owner", "repo", "latest")

            assert pooch.registry["file.txt"] == "sha256:hash"

    async def test_fetch_fallback_to_configured_version(self, tmp_path: Path):
        """Should use configured version when _resolved_version is not set."""
        content = b"fallback content"
        content_hash = hashlib.sha256(content).hexdigest()

        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://example.com/v{version}",
            version="1.0.0",
            registry={"file.txt": f"sha256:{content_hash}"},
        )

        with aioresponses() as m:
            # Should use configured version directly
            m.get("https://example.com/v1.0.0/file.txt", body=content)

            path = await pooch.fetch("file.txt")

            assert path.exists()
            assert path.read_bytes() == content

    async def test_fetch_uses_resolved_version_from_github(self, tmp_path: Path):
        """Should use actual tag from GitHub API instead of configured version."""
        content = b"resolved version content"
        content_hash = hashlib.sha256(content).hexdigest()

        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0.dev0",  # This would be used if _resolved_version wasn't set
            registry={},
        )

        github_response = {
            "tag_name": "v2.0.0",  # Actual tag from GitHub
            "assets": [
                {"name": "file.txt", "digest": f"sha256:{content_hash}"},
            ],
        }

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/latest",
                payload=github_response,
            )
            # Should use "v2.0.0" (resolved) instead of "vlatest"
            m.get(
                "https://github.com/owner/repo/releases/download/v2.0.0/file.txt",
                body=content,
            )

            await pooch.load_registry_from_github("owner", "repo", "latest")
            path = await pooch.fetch("file.txt")

            assert path.exists()
            assert path.read_bytes() == content
            assert pooch._resolved_version == "v2.0.0"


class TestDecompressZstd:
    """Tests for decompress_zstd function."""

    async def test_decompresses_file(self, tmp_path: Path):
        """Should decompress .zst file and remove original."""
        # Use native zstd or backports.zstd
        if sys.version_info >= (3, 14):
            from compression import zstd
        else:
            from backports import zstd

        # Create compressed file
        original_content = b"Hello, World! " * 100
        compressed_data = zstd.compress(original_content)

        compressed_file = tmp_path / "test.txt.zst"
        compressed_file.write_bytes(compressed_data)

        # Decompress
        result_path = await decompress_zstd(compressed_file)

        # Verify
        assert result_path == tmp_path / "test.txt"
        assert result_path.exists()
        assert result_path.read_bytes() == original_content
        assert not compressed_file.exists()  # Original should be deleted

    async def test_corrupted_zstd_file(self, tmp_path: Path):
        """Should raise error on corrupted zstd file."""
        # Use native zstd or backports.zstd to get the error type
        if sys.version_info >= (3, 14):
            from compression.zstd import ZstdError
        else:
            from backports.zstd import ZstdError

        # Create corrupted file (not valid zstd data)
        corrupted_file = tmp_path / "corrupted.txt.zst"
        corrupted_file.write_bytes(b"this is not valid zstd data")

        with pytest.raises(ZstdError):
            await decompress_zstd(corrupted_file)

    async def test_empty_zstd_file(self, tmp_path: Path):
        """Should handle empty compressed file."""
        # Use native zstd or backports.zstd
        if sys.version_info >= (3, 14):
            from compression import zstd
        else:
            from backports import zstd

        # Create compressed empty file
        compressed_data = zstd.compress(b"")

        compressed_file = tmp_path / "empty.txt.zst"
        compressed_file.write_bytes(compressed_data)

        # Decompress
        result_path = await decompress_zstd(compressed_file)

        # Verify
        assert result_path.exists()
        assert result_path.read_bytes() == b""


class TestUntar:
    """Tests for untar function."""

    async def test_extracts_tar_archive(self, tmp_path: Path):
        """Should extract tar archive and remove original."""
        import tarfile

        # Create a tar archive
        archive_content_dir = tmp_path / "content"
        archive_content_dir.mkdir()
        (archive_content_dir / "file1.txt").write_text("content1")
        (archive_content_dir / "file2.txt").write_text("content2")

        tar_path = tmp_path / "archive.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(archive_content_dir / "file1.txt", arcname="file1.txt")
            tar.add(archive_content_dir / "file2.txt", arcname="file2.txt")

        # Extract
        result_dir = await untar(tar_path)

        # Verify
        assert result_dir.exists()
        assert (result_dir / "file1.txt").read_text() == "content1"
        assert (result_dir / "file2.txt").read_text() == "content2"
        assert not tar_path.exists()  # Original should be deleted

    async def test_extracts_tar_gz_archive(self, tmp_path: Path):
        """Should extract .tar.gz archive and remove original."""
        import tarfile

        # Create a tar.gz archive
        archive_content_dir = tmp_path / "content"
        archive_content_dir.mkdir()
        (archive_content_dir / "file.txt").write_text("gzip content")

        tar_path = tmp_path / "archive.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(archive_content_dir / "file.txt", arcname="file.txt")

        # Extract
        result_dir = await untar(tar_path)

        # Verify
        assert result_dir.exists()
        assert (result_dir / "file.txt").read_text() == "gzip content"
        assert not tar_path.exists()  # Original should be deleted

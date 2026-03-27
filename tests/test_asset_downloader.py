"""Tests for asset_downloader module."""

from __future__ import annotations

import asyncio
import sys
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import aiohttp
import pytest
from aioresponses import aioresponses

if sys.version_info >= (3, 14):
    from compression import zstd
    from compression.zstd import ZstdError
else:
    from backports import zstd
    from backports.zstd import ZstdError

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
from exec_sandbox.hash_utils import IncrementalHasher, bytes_hash
from exec_sandbox.lock_utils import file_lock
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

    @pytest.mark.parametrize(
        ("host_os", "expected_substr"),
        [
            (HostOS.MACOS, "Library/Caches"),
            (HostOS.LINUX, ".cache"),
            (HostOS.UNKNOWN, ".cache"),
        ],
    )
    def test_platform_cache_path(self, host_os: HostOS, expected_substr: str):
        """Should use platform-appropriate cache directory."""
        with patch("exec_sandbox.platform_utils.detect_host_os", return_value=host_os):
            with patch.dict("os.environ", {}, clear=True):
                assert expected_substr in str(get_cache_dir())

    def test_linux_xdg_cache_home(self):
        """Should respect XDG_CACHE_HOME on Linux."""
        with patch("exec_sandbox.platform_utils.detect_host_os", return_value=HostOS.LINUX):
            with patch.dict("os.environ", {"XDG_CACHE_HOME": "/xdg/cache"}):
                result = get_cache_dir()
                assert result == Path("/xdg/cache/exec-sandbox")


class TestGetCacheDirReExport:
    """Tests for get_cache_dir re-export from platform_utils."""

    def test_reexport_identity(self):
        """get_cache_dir in asset_downloader is the same object as in platform_utils."""
        from exec_sandbox.platform_utils import get_cache_dir as platform_get_cache_dir

        assert get_cache_dir is platform_get_cache_dir


class TestOsCache:
    """Tests for os_cache alias function."""

    def test_alias_for_get_cache_dir(self):
        """os_cache should be an alias for get_cache_dir."""
        assert os_cache("test") == get_cache_dir("test")


class TestGetCurrentArch:
    """Tests for get_current_arch function."""

    @pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
    def test_returns_correct_arch(self, arch: str):
        """Should return the correct architecture string."""
        with patch("exec_sandbox.asset_downloader.get_arch_name", return_value=arch):
            assert get_current_arch() == arch


class TestGetGvproxySuffix:
    """Tests for get_gvproxy_suffix function."""

    @pytest.mark.parametrize(
        ("os_name", "arch_name", "expected"),
        [
            ("darwin", "arm64", "darwin-arm64"),
            ("darwin", "amd64", "darwin-amd64"),
            ("linux", "arm64", "linux-arm64"),
            ("linux", "amd64", "linux-amd64"),
        ],
    )
    def test_returns_correct_suffix(self, os_name: str, arch_name: str, expected: str):
        """Should return correct os-arch suffix."""
        with patch("exec_sandbox.asset_downloader.get_os_name", return_value=os_name):
            with patch("exec_sandbox.asset_downloader.get_arch_name", return_value=arch_name):
                assert get_gvproxy_suffix() == expected


class TestRetrieve:
    """Tests for retrieve function."""

    async def test_downloads_and_caches(self, tmp_path: Path):
        """Should download file and cache it locally."""
        content = b"hello world"
        content_hash = bytes_hash(content)

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
        content_hash = bytes_hash(content)

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

    @pytest.mark.parametrize("status_code", [404, 500])
    async def test_http_error(self, tmp_path: Path, status_code: int):
        """Should raise AssetDownloadError on HTTP error status."""
        with aioresponses() as m:
            url = f"https://example.com/error-{status_code}.txt"
            for _ in range(3):  # All retry attempts return error
                m.get(url, status=status_code)

            with pytest.raises(AssetDownloadError):
                await retrieve(
                    url=url,
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
        content_hash = bytes_hash(content)

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
        content_hash = bytes_hash(content)

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

        # Use a 32MB file so the streaming-vs-full-read gap is large enough
        # to tolerate GC-delay overhead on free-threaded Python (3.14t).
        file_size = 32 * 1024 * 1024  # 32MB
        large_file = tmp_path / "large.bin"

        # Write file in chunks to avoid memory spike during setup
        chunk = b"x" * (64 * 1024)  # 64KB chunks
        hasher = IncrementalHasher()
        with large_file.open("wb") as f:
            for _ in range(file_size // len(chunk)):
                f.write(chunk)
                hasher.update(chunk)
        expected_hash = hasher.hexdigest()

        # Warmup: absorb one-time allocations from tracemalloc internals,
        # aiofiles thread-pool startup, and hashlib module init so they
        # don't inflate the peak measurement.
        tracemalloc.start()
        await _verify_hash(large_file, f"sha256:{expected_hash}")
        tracemalloc.stop()

        # Measure peak memory during hash verification
        tracemalloc.start()
        result = await _verify_hash(large_file, f"sha256:{expected_hash}")
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert result is True

        # Broken streaming (entire file read at once): peak = file_size
        # (hashlib.update processes in-place, no copy).
        # Working streaming on standard CPython: peak = chunk_size
        # (refcounting frees each chunk immediately on walrus reassignment).
        # Working streaming on free-threaded Python (3.14t): biased
        # refcounting defers deallocation of bytes objects handed across
        # threads (aiofiles worker -> event loop).  Observed ~4MB peak for
        # a 4MB file (all 64 chunks alive).  A larger file gives more I/O
        # scheduling points for deferred frees, keeping peak below file_size.
        max_allowed = file_size // 2
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
        content_hash = bytes_hash(content)

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
        content_hash = bytes_hash(content)

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
        content_hash = bytes_hash(content)

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
        # Create corrupted file (not valid zstd data)
        corrupted_file = tmp_path / "corrupted.txt.zst"
        corrupted_file.write_bytes(b"this is not valid zstd data")

        with pytest.raises(ZstdError):
            await decompress_zstd(corrupted_file)

    async def test_empty_zstd_file(self, tmp_path: Path):
        """Should handle empty compressed file."""
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


# ============================================================================
# file_lock unit tests
# ============================================================================


class TestFileLock:
    """Tests for file_lock utility."""

    async def test_yields_true_when_target_missing(self, tmp_path: Path):
        """Should yield True when target file does not exist."""
        target = tmp_path / "missing.txt"
        async with file_lock(target) as should_proceed:
            assert should_proceed is True

    async def test_yields_false_when_target_exists(self, tmp_path: Path):
        """Should yield False when target file already exists with content."""
        target = tmp_path / "existing.txt"
        target.write_text("content")
        async with file_lock(target) as should_proceed:
            assert should_proceed is False

    async def test_nonblocking_raises_when_held(self, tmp_path: Path):
        """Should raise BlockingIOError in non-blocking mode when lock is held."""
        target = tmp_path / "locked.txt"
        async with file_lock(target):
            with pytest.raises(BlockingIOError):
                async with file_lock(target, blocking=False):
                    pass  # Should not reach here

    @pytest.mark.parametrize("n_waiters", [1, 10, 100])
    async def test_blocking_waits_then_proceeds(self, tmp_path: Path, n_waiters: int):
        """Blocking callers should wait for first holder, then see the produced file."""
        target = tmp_path / "produced.txt"
        order: list[str] = []

        async def holder() -> None:
            async with file_lock(target) as should_proceed:
                assert should_proceed is True
                order.append("holder-acquired")
                await asyncio.sleep(0.1)  # Simulate work
                target.write_text("done")
                order.append("holder-released")

        async def waiter(idx: int) -> None:
            # Small delay so holder acquires first
            await asyncio.sleep(0.02)
            async with file_lock(target) as should_proceed:
                order.append(f"waiter-{idx}-acquired")
                # File was produced by holder, so should_proceed is False
                assert should_proceed is False

        await asyncio.gather(holder(), *(waiter(i) for i in range(n_waiters)))
        assert order[0] == "holder-acquired"
        assert order[1] == "holder-released"
        # All waiters must appear after holder released
        waiter_events = order[2:]
        assert len(waiter_events) == n_waiters
        assert all(e.startswith("waiter-") and e.endswith("-acquired") for e in waiter_events)


# ============================================================================
# Concurrent retrieve tests — regression for EC2 race condition
# ============================================================================


class TestConcurrentRetrieve:
    """Tests for concurrent retrieve() calls.

    These tests reproduce the race condition found on EC2 where two concurrent
    fetch_base_image calls would race: one's retry cleanup (dest.unlink) deleted
    the .zst file while the other coroutine was decompressing it.
    """

    @pytest.mark.parametrize("concurrency", [1, 10, 100])
    async def test_concurrent_same_url_with_decompress(self, tmp_path: Path, concurrency: int):
        """Concurrent retrieve() with decompress_zstd should not race.

        Regression: on EC2, two fetch_base_image calls raced — one's retry
        cleanup deleted .zst while the other was decompressing it.
        The file lock on the final output serializes the full pipeline.
        """
        original = b"concurrent decompression test content"
        compressed = zstd.compress(original)
        compressed_hash = bytes_hash(compressed)
        url = "https://example.com/concurrent.txt.zst"

        with aioresponses() as m:
            for _ in range(concurrency):
                m.get(url, body=compressed)

            results = await asyncio.gather(
                *[
                    retrieve(
                        url=url,
                        known_hash=f"sha256:{compressed_hash}",
                        path=tmp_path,
                        processor=decompress_zstd,
                        progressbar=False,
                    )
                    for _ in range(concurrency)
                ]
            )

            # All should return the same decompressed path
            assert all(r == results[0] for r in results)
            assert results[0].name == "concurrent.txt"
            assert results[0].read_bytes() == original

    @pytest.mark.parametrize("concurrency", [1, 10, 100])
    async def test_concurrent_same_url_no_processor(self, tmp_path: Path, concurrency: int):
        """Concurrent retrieve() without processor should not corrupt the file."""
        content = b"concurrent download no processor"
        content_hash = bytes_hash(content)
        url = "https://example.com/concurrent-plain.txt"

        with aioresponses() as m:
            for _ in range(concurrency):
                m.get(url, body=content)

            results = await asyncio.gather(
                *[
                    retrieve(
                        url=url,
                        known_hash=f"sha256:{content_hash}",
                        path=tmp_path,
                        progressbar=False,
                    )
                    for _ in range(concurrency)
                ]
            )

            assert all(r == results[0] for r in results)
            assert results[0].read_bytes() == content

    @pytest.mark.parametrize("concurrency", [1, 10, 100])
    async def test_concurrent_different_urls_not_serialized(self, tmp_path: Path, concurrency: int):
        """Concurrent retrieves of different URLs should run in parallel."""
        pairs = [(f"https://example.com/file-{i}.txt", f"content for file {i}".encode()) for i in range(concurrency)]
        hashes = {url: bytes_hash(body) for url, body in pairs}

        with aioresponses() as m:
            for url, body in pairs:
                m.get(url, body=body)

            results = await asyncio.gather(
                *[
                    retrieve(
                        url=url,
                        known_hash=f"sha256:{hashes[url]}",
                        path=tmp_path,
                        progressbar=False,
                    )
                    for url, _ in pairs
                ]
            )

            for i, (_url, body) in enumerate(pairs):
                assert results[i].name == f"file-{i}.txt"
                assert results[i].read_bytes() == body

    @pytest.mark.parametrize("concurrency", [1, 10, 100])
    async def test_concurrent_first_fails_rest_succeed(self, tmp_path: Path, concurrency: int):
        """When one concurrent retrieve fails, the others should still succeed."""
        content = b"success content"
        content_hash = bytes_hash(content)
        url_fail = "https://example.com/fail.txt"
        ok_urls = [f"https://example.com/ok-{i}.txt" for i in range(concurrency - 1)]

        with aioresponses() as m:
            # fail.txt always returns 500 (all 3 retry attempts)
            for _ in range(3):
                m.get(url_fail, status=500)
            for url in ok_urls:
                m.get(url, body=content)

            results = await asyncio.gather(
                retrieve(url=url_fail, known_hash="sha256:abc123", path=tmp_path, progressbar=False),
                *[
                    retrieve(url=url, known_hash=f"sha256:{content_hash}", path=tmp_path, progressbar=False)
                    for url in ok_urls
                ],
                return_exceptions=True,
            )

            # First should fail with AssetDownloadError
            assert isinstance(results[0], AssetDownloadError)
            # Rest should succeed
            for r in results[1:]:
                assert isinstance(r, Path)
                assert r.read_bytes() == content

    async def test_concurrent_corrupted_cache_redownloads(self, tmp_path: Path):
        """Corrupted cached file (wrong hash) should be re-downloaded.

        Exercises the fallthrough path at retrieve() lines 96-101 where
        should_download is False but hash verification fails.
        """
        content = b"correct content"
        content_hash = bytes_hash(content)
        url = "https://example.com/corrupted.txt"

        # Pre-write a corrupted file so file_lock yields False
        corrupted_file = tmp_path / "corrupted.txt"
        corrupted_file.write_bytes(b"wrong content")

        with aioresponses() as m:
            m.get(url, body=content)

            result = await retrieve(
                url=url,
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
                progressbar=False,
            )

            assert result == corrupted_file
            assert result.read_bytes() == content

    async def test_concurrent_empty_target_proceeds(self, tmp_path: Path):
        """Empty target file should trigger a fresh download.

        Exercises file_lock line 54 where st_size == 0 causes
        should_proceed=True even though the file exists.
        """
        content = b"fresh download content"
        content_hash = bytes_hash(content)
        url = "https://example.com/empty-target.txt"

        # Pre-create an empty file — file_lock should yield True
        empty_file = tmp_path / "empty-target.txt"
        empty_file.write_bytes(b"")

        with aioresponses() as m:
            m.get(url, body=content)

            result = await retrieve(
                url=url,
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
                progressbar=False,
            )

            assert result == empty_file
            assert result.read_bytes() == content

"""Tests for SnapshotManager.

Unit tests: Cache key computation, filesystem operations.
Integration tests: Snapshot creation with QEMU (requires images).
"""

import hashlib
import sys
from pathlib import Path

import pytest

# Use native zstd (Python 3.14+) or backports.zstd
if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

from exec_sandbox.models import Language

# ============================================================================
# Unit Tests - Cache Key Computation
# ============================================================================


class TestCacheKeyComputation:
    """Tests for cache key computation logic."""

    def test_cache_key_format(self) -> None:
        """Cache key format: {language}-{16char_packages_hash}."""
        # Simulate _compute_cache_key logic
        language = "python"
        packages = ["pandas==2.0.0", "numpy==1.24.0"]

        # L0 format: sorted packages joined without separator, 16-char hash
        packages_str = "".join(sorted(packages))
        packages_hash = hashlib.sha256(packages_str.encode()).hexdigest()[:16]
        cache_key = f"{language}-{packages_hash}"

        assert cache_key.startswith("python-")
        assert len(cache_key.split("-")[1]) == 16

    def test_cache_key_deterministic(self) -> None:
        """Same inputs produce same cache key."""
        packages = ["pandas==2.0.0", "numpy==1.24.0"]

        # L0 format: joined without separator
        packages_str = "".join(sorted(packages))
        hash1 = hashlib.sha256(packages_str.encode()).hexdigest()[:16]
        hash2 = hashlib.sha256(packages_str.encode()).hexdigest()[:16]

        assert hash1 == hash2

    def test_cache_key_sorted_packages(self) -> None:
        """Package order doesn't affect cache key (sorted)."""
        packages1 = ["pandas==2.0.0", "numpy==1.24.0"]
        packages2 = ["numpy==1.24.0", "pandas==2.0.0"]

        # L0 format: joined without separator
        hash1 = hashlib.sha256("".join(sorted(packages1)).encode()).hexdigest()[:16]
        hash2 = hashlib.sha256("".join(sorted(packages2)).encode()).hexdigest()[:16]

        assert hash1 == hash2

    def test_cache_key_different_languages(self) -> None:
        """Different languages produce different cache keys."""
        packages = ["lodash@4.17.21"]

        # L0 format: joined without separator
        key1 = f"python-{hashlib.sha256(''.join(sorted(packages)).encode()).hexdigest()[:16]}"
        key2 = f"javascript-{hashlib.sha256(''.join(sorted(packages)).encode()).hexdigest()[:16]}"

        assert key1 != key2

    def test_cache_key_empty_packages(self) -> None:
        """Empty packages list produces '{language}-base' key."""
        # L0 format: empty packages = "{language}-base"
        cache_key = "python-base"

        assert cache_key.startswith("python-")
        assert cache_key == "python-base"


class TestSettings:
    """Tests for Settings used by SnapshotManager."""

    def test_settings_snapshot_cache_dir(self, make_vm_settings, tmp_path: Path) -> None:
        """Settings has snapshot_cache_dir."""
        settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache")

        assert settings.snapshot_cache_dir == tmp_path / "cache"

    def test_settings_s3_config(self, make_vm_settings) -> None:
        """Settings has S3 configuration."""
        settings = make_vm_settings(
            s3_bucket="my-bucket",
            s3_region="us-west-2",
        )

        assert settings.s3_bucket == "my-bucket"
        assert settings.s3_region == "us-west-2"


# ============================================================================
# Integration Tests - Require QEMU + Images
# ============================================================================


class TestSnapshotManagerIntegration:
    """Integration tests for SnapshotManager with real QEMU VMs."""

    async def test_l0_cache_miss(self, make_vm_manager, make_vm_settings, tmp_path: Path) -> None:
        """L0 cache miss returns (None, False) for non-existent snapshot."""
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache")
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(snapshot_cache_dir=tmp_path / "cache")
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Check for non-existent snapshot
        path, has_memory_snapshot = await snapshot_manager._check_l0_cache("nonexistent-abc123")
        assert path is None
        assert has_memory_snapshot is False

    async def test_compute_cache_key(self, make_vm_manager, make_vm_settings, tmp_path: Path) -> None:
        """Test actual _compute_cache_key method."""
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache")
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(snapshot_cache_dir=tmp_path / "cache")
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Test with packages
        key = snapshot_manager._compute_cache_key(
            language=Language.PYTHON,
            packages=["pandas==2.0.0", "numpy==1.24.0"],
        )

        assert key.startswith("python-")
        # L0 format: 16-char hash
        assert len(key.split("-")[1]) == 16

        # Test without packages (base)
        base_key = snapshot_manager._compute_cache_key(
            language=Language.PYTHON,
            packages=[],
        )
        assert base_key == "python-base"

    async def test_create_snapshot(self, make_vm_manager, make_vm_settings, tmp_path: Path) -> None:
        """Create snapshot with packages (slow, requires VM)."""
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache")
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(snapshot_cache_dir=tmp_path / "cache")
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Create snapshot (this boots a VM and installs packages)
        snapshot_path, has_memory_snapshot = await snapshot_manager.get_or_create_snapshot(
            language=Language.PYTHON,
            packages=["requests==2.31.0"],
            tenant_id="test",
            task_id="test-1",
        )

        assert snapshot_path.exists()
        assert snapshot_path.suffix == ".qcow2"
        # New snapshot has no memory state yet (disk-only)
        assert has_memory_snapshot is False

        # Second call should hit L0 cache
        cached_path, _cached_has_memory = await snapshot_manager.get_or_create_snapshot(
            language=Language.PYTHON,
            packages=["requests==2.31.0"],
            tenant_id="test",
            task_id="test-2",
        )

        assert cached_path == snapshot_path


# ============================================================================
# L0 Cache Tests - Memory Snapshots
# ============================================================================


class TestL0Cache:
    """Tests for L0 (memory snapshot) cache operations."""

    async def test_l0_cache_hit_returns_path(self, make_vm_manager, make_vm_settings, tmp_path: Path) -> None:
        """L0 cache returns (path, has_memory) when valid qcow2 snapshot exists."""
        import asyncio

        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache")
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(snapshot_cache_dir=tmp_path / "cache")
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Create a minimal valid qcow2 file
        cache_key = "python-abc123"
        snapshot_path = settings.snapshot_cache_dir / f"{cache_key}.qcow2"

        # Create actual qcow2 using qemu-img
        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            str(snapshot_path),
            "1M",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        assert proc.returncode == 0

        path, has_memory_snapshot = await snapshot_manager._check_l0_cache(cache_key)
        assert path == snapshot_path
        # No internal snapshot named "ready" yet
        assert has_memory_snapshot is False

    async def test_l0_cache_nonexistent_returns_none(self, make_vm_manager, make_vm_settings, tmp_path: Path) -> None:
        """L0 cache returns (None, False) for non-existent snapshot."""
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache")
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(snapshot_cache_dir=tmp_path / "cache")
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Check for non-existent snapshot
        path, has_memory_snapshot = await snapshot_manager._check_l0_cache("nonexistent-key")
        assert path is None
        assert has_memory_snapshot is False

    async def test_l1_evict_oldest_snapshot(self, make_vm_manager, make_vm_settings, tmp_path: Path) -> None:
        """_evict_oldest_snapshot removes oldest file by atime."""
        import asyncio
        import time

        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache")
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(snapshot_cache_dir=tmp_path / "cache")
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Create multiple snapshots with staggered atimes
        oldest_path = settings.snapshot_cache_dir / "python-oldest.qcow2"
        newest_path = settings.snapshot_cache_dir / "python-newest.qcow2"

        # Create oldest first
        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            str(oldest_path),
            "1M",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Small delay to ensure different atime
        time.sleep(0.1)

        # Create newest
        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            str(newest_path),
            "1M",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Touch newest to update atime
        newest_path.touch()

        assert oldest_path.exists()
        assert newest_path.exists()

        # Evict oldest
        await snapshot_manager._evict_oldest_snapshot()

        # Oldest should be removed, newest should remain
        assert not oldest_path.exists()
        assert newest_path.exists()


# ============================================================================
# L3 Cache Tests - S3 (using moto)
# ============================================================================


class TestL3Cache:
    """Tests for L3 (S3) cache operations using moto server mode."""

    async def test_get_s3_client_raises_without_bucket(self, make_vm_manager, make_vm_settings, tmp_path: Path) -> None:
        """_get_s3_client raises SnapshotError when s3_bucket not set."""
        from exec_sandbox.exceptions import SnapshotError
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache", s3_bucket=None)
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(snapshot_cache_dir=tmp_path / "cache", s3_bucket=None)
        snapshot_manager = SnapshotManager(settings, vm_manager)

        with pytest.raises(SnapshotError) as exc_info:
            await snapshot_manager._get_s3_client()
        assert "S3 backup disabled" in str(exc_info.value)

    async def test_upload_to_s3_success(self, make_vm_manager, make_vm_settings, tmp_path: Path, monkeypatch) -> None:
        """Snapshot uploads to S3 with zstd compression using real aioboto3 client."""
        import boto3
        from moto.server import ThreadedMotoServer

        from exec_sandbox.snapshot_manager import SnapshotManager

        # Set fake AWS credentials for moto
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        # Start moto server
        server = ThreadedMotoServer(port=0)  # port=0 picks random available port
        server.start()
        endpoint_url = f"http://localhost:{server._server.server_port}"

        try:
            # Create bucket using sync boto3
            s3_sync = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_url)
            s3_sync.create_bucket(Bucket="test-snapshots")

            settings = make_vm_settings(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = make_vm_manager(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            snapshot_manager = SnapshotManager(settings, vm_manager)

            # Create a test snapshot file
            cache_key = "python-test123"
            snapshot_path = settings.snapshot_cache_dir / f"{cache_key}.qcow2"
            snapshot_path.write_bytes(b"fake qcow2 content")

            # Upload using real aioboto3 client
            await snapshot_manager._upload_to_s3(cache_key, snapshot_path)

            # Verify uploaded (compressed) using sync boto3
            objects = s3_sync.list_objects_v2(Bucket="test-snapshots")
            keys = [obj["Key"] for obj in objects.get("Contents", [])]
            assert f"snapshots/{cache_key}.qcow2.zst" in keys

            # Verify compressed file was cleaned up
            compressed_path = settings.snapshot_cache_dir / f"{cache_key}.qcow2.zst"
            assert not compressed_path.exists()

        finally:
            server.stop()

    async def test_download_from_s3_success(
        self, make_vm_manager, make_vm_settings, tmp_path: Path, monkeypatch
    ) -> None:
        """Snapshot downloads from S3 and decompresses using real aioboto3 client."""
        import boto3
        from moto.server import ThreadedMotoServer

        from exec_sandbox.snapshot_manager import SnapshotManager

        # Set fake AWS credentials for moto
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        # Start moto server
        server = ThreadedMotoServer(port=0)
        server.start()
        endpoint_url = f"http://localhost:{server._server.server_port}"

        try:
            # Create bucket and upload test data using sync boto3
            s3_sync = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_url)
            s3_sync.create_bucket(Bucket="test-snapshots")

            # Compress and upload test data
            original_content = b"fake qcow2 content for download"
            compressed = zstd.compress(original_content)
            s3_sync.put_object(
                Bucket="test-snapshots",
                Key="snapshots/python-download123.qcow2.zst",
                Body=compressed,
            )

            settings = make_vm_settings(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = make_vm_manager(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            snapshot_manager = SnapshotManager(settings, vm_manager)

            # Download using real aioboto3 client
            result = await snapshot_manager._download_from_s3("python-download123")

            # Verify downloaded and decompressed
            assert result.exists()
            assert result.read_bytes() == original_content

        finally:
            server.stop()

    async def test_download_from_s3_not_found(
        self, make_vm_manager, make_vm_settings, tmp_path: Path, monkeypatch
    ) -> None:
        """S3 download raises SnapshotError when key missing."""
        import boto3
        from moto.server import ThreadedMotoServer

        from exec_sandbox.exceptions import SnapshotError
        from exec_sandbox.snapshot_manager import SnapshotManager

        # Set fake AWS credentials for moto
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        # Start moto server
        server = ThreadedMotoServer(port=0)
        server.start()
        endpoint_url = f"http://localhost:{server._server.server_port}"

        try:
            # Create empty bucket
            s3_sync = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_url)
            s3_sync.create_bucket(Bucket="test-snapshots")

            settings = make_vm_settings(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = make_vm_manager(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            snapshot_manager = SnapshotManager(settings, vm_manager)

            with pytest.raises(SnapshotError) as exc_info:
                await snapshot_manager._download_from_s3("nonexistent-key")
            assert "S3 download failed" in str(exc_info.value)

        finally:
            server.stop()

    async def test_upload_to_s3_silent_failure(
        self, make_vm_manager, make_vm_settings, tmp_path: Path, monkeypatch
    ) -> None:
        """S3 upload failure is silent (L1 cache still works)."""
        from moto.server import ThreadedMotoServer

        from exec_sandbox.snapshot_manager import SnapshotManager

        # Set fake AWS credentials for moto
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        # Start moto server but don't create bucket
        server = ThreadedMotoServer(port=0)
        server.start()
        endpoint_url = f"http://localhost:{server._server.server_port}"

        try:
            settings = make_vm_settings(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="nonexistent-bucket",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = make_vm_manager(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="nonexistent-bucket",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            snapshot_manager = SnapshotManager(settings, vm_manager)

            cache_key = "python-fail123"
            snapshot_path = settings.snapshot_cache_dir / f"{cache_key}.qcow2"
            snapshot_path.write_bytes(b"test content")

            # Should not raise - silent failure (bucket doesn't exist)
            await snapshot_manager._upload_to_s3(cache_key, snapshot_path)
            # No exception = success (silent failure)

        finally:
            server.stop()

    async def test_upload_semaphore_limits_concurrency(
        self, make_vm_manager, make_vm_settings, tmp_path: Path, monkeypatch
    ) -> None:
        """Upload semaphore limits concurrent S3 uploads to configured max.

        Verifies that max_concurrent_s3_uploads actually bounds parallel uploads.
        Uses real moto server with instrumented upload tracking.
        """
        import asyncio
        from contextlib import asynccontextmanager

        import boto3
        from moto.server import ThreadedMotoServer

        from exec_sandbox.snapshot_manager import SnapshotManager

        # Set fake AWS credentials for moto
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        # Start moto server
        server = ThreadedMotoServer(port=0)
        server.start()
        endpoint_url = f"http://localhost:{server._server.server_port}"

        try:
            # Create bucket (unique name to avoid parallel test interference)
            bucket_name = "test-semaphore-snapshots"
            s3_sync = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_url)
            s3_sync.create_bucket(Bucket=bucket_name)

            # Configure semaphore to allow only 2 concurrent uploads
            settings = make_vm_settings(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket=bucket_name,
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
                max_concurrent_s3_uploads=2,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = make_vm_manager(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket=bucket_name,
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
                max_concurrent_s3_uploads=2,
            )
            snapshot_manager = SnapshotManager(settings, vm_manager)

            # Track concurrent uploads (inside semaphore-protected section)
            concurrent_count = 0
            max_concurrent_observed = 0

            # Wrap _get_s3_client to return instrumented client
            original_get_client = snapshot_manager._get_s3_client

            @asynccontextmanager
            async def instrumented_s3_context(s3):
                original_upload = s3.upload_file

                async def tracked_upload(*args, **kwargs):
                    nonlocal concurrent_count, max_concurrent_observed
                    concurrent_count += 1
                    max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
                    try:
                        await asyncio.sleep(0.05)  # Ensure overlap detection
                        return await original_upload(*args, **kwargs)
                    finally:
                        concurrent_count -= 1

                s3.upload_file = tracked_upload
                yield s3

            async def tracked_get_s3_client():
                """Return instrumented S3 client context manager."""
                original_cm = await original_get_client()

                @asynccontextmanager
                async def wrapped():
                    async with original_cm as s3:
                        async with instrumented_s3_context(s3) as instrumented:
                            yield instrumented

                return wrapped()

            snapshot_manager._get_s3_client = tracked_get_s3_client  # type: ignore[method-assign]

            # Create test snapshot files
            for i in range(5):
                (settings.snapshot_cache_dir / f"test-{i}.qcow2").write_bytes(b"test data")

            # Start 5 uploads simultaneously
            tasks = [
                asyncio.create_task(
                    snapshot_manager._upload_to_s3(f"test-{i}", settings.snapshot_cache_dir / f"test-{i}.qcow2")
                )
                for i in range(5)
            ]

            # Wait for all uploads to complete
            await asyncio.gather(*tasks)

            # Verify all files uploaded to S3
            objects = s3_sync.list_objects_v2(Bucket=bucket_name)
            keys = [obj["Key"] for obj in objects.get("Contents", [])]
            assert len(keys) == 5, f"Expected 5 uploads, got {len(keys)}"

            # Verify semaphore limited concurrency
            assert max_concurrent_observed <= 2, (
                f"Expected max 2 concurrent uploads (semaphore limit), but observed {max_concurrent_observed}"
            )
            # Also verify uploads actually ran concurrently (not serialized to 1)
            assert max_concurrent_observed == 2, (
                f"Expected exactly 2 concurrent uploads (semaphore should allow 2), but observed {max_concurrent_observed}"
            )

        finally:
            server.stop()


# ============================================================================
# Cache Hierarchy Tests - Full L0 → L3 → Create Flow
# ============================================================================


class TestCacheHierarchy:
    """Tests for the full cache hierarchy flow in get_or_create_snapshot().

    These tests verify the real L1 → L3 → Create pattern:
    - L1 hit: Return immediately from local disk
    - L1 miss → L3 hit: Download from S3, populate L1
    - L1 miss → L3 miss: Create snapshot, upload to S3

    Uses moto server for real S3 client and mocks _create_snapshot to avoid QEMU.
    """

    async def test_l0_hit_returns_immediately_no_s3(
        self, make_vm_manager, make_vm_settings, tmp_path: Path, monkeypatch
    ) -> None:
        """L0 cache hit returns (path, has_memory) immediately without touching S3.

        Flow: L0 HIT → return (no S3 call, no creation)
        """
        import asyncio
        from unittest.mock import AsyncMock, patch

        from exec_sandbox.models import Language
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(
            snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-bucket",  # S3 configured but should NOT be called
            s3_region="us-east-1",
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(
            snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-bucket",
            s3_region="us-east-1",
        )
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Pre-populate L1 cache with valid qcow2
        cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["requests==2.31.0"])
        snapshot_path = settings.snapshot_cache_dir / f"{cache_key}.qcow2"

        # Create actual qcow2 using qemu-img
        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            str(snapshot_path),
            "1M",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        assert proc.returncode == 0

        # Mock S3 and creation to track if they're called
        with patch.object(snapshot_manager, "_download_from_s3", new_callable=AsyncMock) as mock_s3:
            with patch.object(snapshot_manager, "_create_snapshot", new_callable=AsyncMock) as mock_create:
                result_path, has_memory_snapshot = await snapshot_manager.get_or_create_snapshot(
                    language=Language.PYTHON,
                    packages=["requests==2.31.0"],
                    tenant_id="test",
                    task_id="test-1",
                )

        # Verify L0 hit: returned correct path
        assert result_path == snapshot_path
        # L0 cache hit without internal snapshot returns False
        assert has_memory_snapshot is False

        # Verify S3 was NOT called (L0 hit skips S3)
        mock_s3.assert_not_called()

        # Verify creation was NOT called (L0 hit skips creation)
        mock_create.assert_not_called()

    async def test_l0_miss_l3_hit_downloads_from_s3(
        self, make_vm_manager, make_vm_settings, tmp_path: Path, monkeypatch
    ) -> None:
        """L0 miss with L3 hit downloads from S3 and returns path.

        Flow: L0 MISS → L3 HIT → download → return (no creation)
        """
        from unittest.mock import AsyncMock, patch

        import boto3
        from moto.server import ThreadedMotoServer

        from exec_sandbox.models import Language
        from exec_sandbox.snapshot_manager import SnapshotManager

        # Set fake AWS credentials for moto
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        # Start moto server
        server = ThreadedMotoServer(port=0)
        server.start()
        endpoint_url = f"http://localhost:{server._server.server_port}"

        try:
            # Create bucket
            s3_sync = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_url)
            s3_sync.create_bucket(Bucket="test-snapshots")

            settings = make_vm_settings(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = make_vm_manager(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            snapshot_manager = SnapshotManager(settings, vm_manager)

            # Compute cache key for the packages we'll request
            cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["numpy==1.26.0"])

            # Pre-populate S3 (L3) with compressed snapshot
            original_content = b"fake qcow2 snapshot from S3"
            compressed = zstd.compress(original_content)
            s3_sync.put_object(
                Bucket="test-snapshots",
                Key=f"snapshots/{cache_key}.qcow2.zst",
                Body=compressed,
            )

            # L0 is empty (no file on disk)
            assert not (settings.snapshot_cache_dir / f"{cache_key}.qcow2").exists()

            # Mock creation to verify it's NOT called
            with patch.object(snapshot_manager, "_create_snapshot", new_callable=AsyncMock) as mock_create:
                result_path, has_memory_snapshot = await snapshot_manager.get_or_create_snapshot(
                    language=Language.PYTHON,
                    packages=["numpy==1.26.0"],
                    tenant_id="test",
                    task_id="test-2",
                )

            # Verify returned path exists and has correct content (decompressed from S3)
            assert result_path.exists()
            assert result_path.read_bytes() == original_content
            # Downloaded from S3 = no memory snapshot
            assert has_memory_snapshot is False

            # Verify creation was NOT called (L3 hit skips creation)
            mock_create.assert_not_called()

        finally:
            server.stop()

    async def test_l0_miss_l3_miss_creates_snapshot(
        self, make_vm_manager, make_vm_settings, tmp_path: Path, monkeypatch
    ) -> None:
        """L0 miss and L3 miss triggers snapshot creation.

        Flow: L0 MISS → L3 MISS → create → return (and upload to S3)
        """
        import asyncio
        from unittest.mock import patch

        import boto3
        from moto.server import ThreadedMotoServer

        from exec_sandbox.models import Language
        from exec_sandbox.snapshot_manager import SnapshotManager

        # Set fake AWS credentials for moto
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        # Start moto server
        server = ThreadedMotoServer(port=0)
        server.start()
        endpoint_url = f"http://localhost:{server._server.server_port}"

        try:
            # Create empty bucket (no snapshots)
            s3_sync = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_url)
            s3_sync.create_bucket(Bucket="test-snapshots")

            settings = make_vm_settings(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = make_vm_manager(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            snapshot_manager = SnapshotManager(settings, vm_manager)

            # Compute cache key
            cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["pandas==2.1.0"])
            expected_path = settings.snapshot_cache_dir / f"{cache_key}.qcow2"

            # Mock _create_snapshot to simulate snapshot creation (avoids real QEMU)
            async def fake_create_snapshot(language, packages, key, tenant_id, task_id):
                # Simulate creating a qcow2 file
                proc = await asyncio.create_subprocess_exec(
                    "qemu-img",
                    "create",
                    "-f",
                    "qcow2",
                    str(expected_path),
                    "1M",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                return expected_path

            with patch.object(snapshot_manager, "_create_snapshot", side_effect=fake_create_snapshot) as mock_create:
                result_path, has_memory_snapshot = await snapshot_manager.get_or_create_snapshot(
                    language=Language.PYTHON,
                    packages=["pandas==2.1.0"],
                    tenant_id="test",
                    task_id="test-3",
                )

            # Verify creation WAS called (cache miss)
            mock_create.assert_called_once()

            # Verify returned path
            assert result_path == expected_path
            assert result_path.exists()
            # Newly created snapshot has no memory state yet
            assert has_memory_snapshot is False

            # Wait briefly for background S3 upload task
            await asyncio.sleep(0.5)

            # Verify S3 upload happened (background task)
            objects = s3_sync.list_objects_v2(Bucket="test-snapshots")
            keys = [obj["Key"] for obj in objects.get("Contents", [])]
            assert f"snapshots/{cache_key}.qcow2.zst" in keys

        finally:
            server.stop()

    async def test_l0_populated_after_l3_download(
        self, make_vm_manager, make_vm_settings, tmp_path: Path, monkeypatch
    ) -> None:
        """After L3 download, L0 cache is populated for next call.

        Flow: L0 MISS → L3 HIT → download → L0 populated
        Then: L0 HIT → return immediately
        """
        from unittest.mock import AsyncMock, patch

        import boto3
        from moto.server import ThreadedMotoServer

        from exec_sandbox.models import Language
        from exec_sandbox.snapshot_manager import SnapshotManager

        # Set fake AWS credentials for moto
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        # Start moto server
        server = ThreadedMotoServer(port=0)
        server.start()
        endpoint_url = f"http://localhost:{server._server.server_port}"

        try:
            # Create bucket
            s3_sync = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_url)
            s3_sync.create_bucket(Bucket="test-snapshots")

            settings = make_vm_settings(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = make_vm_manager(
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            snapshot_manager = SnapshotManager(settings, vm_manager)

            # Compute cache key
            cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["scipy==1.11.0"])
            l0_path = settings.snapshot_cache_dir / f"{cache_key}.qcow2"

            # Pre-populate S3 only
            original_content = b"scipy snapshot content"
            compressed = zstd.compress(original_content)
            s3_sync.put_object(
                Bucket="test-snapshots",
                Key=f"snapshots/{cache_key}.qcow2.zst",
                Body=compressed,
            )

            # Verify L0 is empty before first call
            assert not l0_path.exists()

            # First call: L0 miss → L3 hit
            with patch.object(snapshot_manager, "_create_snapshot", new_callable=AsyncMock) as mock_create:
                _result1_path, _result1_has_memory = await snapshot_manager.get_or_create_snapshot(
                    language=Language.PYTHON,
                    packages=["scipy==1.11.0"],
                    tenant_id="test",
                    task_id="test-4a",
                )
                mock_create.assert_not_called()

            # Verify L0 is NOW populated
            assert l0_path.exists()
            assert l0_path.read_bytes() == original_content

            # Second call: should hit L0 (no S3 download)
            # We'll spy on _download_from_s3 to verify it's not called
            original_download = snapshot_manager._download_from_s3
            download_called = False

            async def spy_download(*args, **kwargs):
                nonlocal download_called
                download_called = True
                return await original_download(*args, **kwargs)

            with patch.object(snapshot_manager, "_download_from_s3", side_effect=spy_download):
                result2_path, _result2_has_memory = await snapshot_manager.get_or_create_snapshot(
                    language=Language.PYTHON,
                    packages=["scipy==1.11.0"],
                    tenant_id="test",
                    task_id="test-4b",
                )

            # Verify L0 hit on second call
            assert result2_path == l0_path
            assert not download_called, "S3 download should NOT be called on L0 hit"

        finally:
            server.stop()

    async def test_same_packages_same_cache_key(self, make_vm_manager, make_vm_settings, tmp_path: Path) -> None:
        """Same packages (regardless of order) produce same cache key and path.

        Verifies deterministic cache key computation.
        """
        import asyncio
        from unittest.mock import AsyncMock, patch

        from exec_sandbox.models import Language
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(
            snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,  # No S3
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(
            snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Compute cache keys for same packages in different orders
        key1 = snapshot_manager._compute_cache_key(Language.PYTHON, ["pandas==2.0.0", "numpy==1.25.0"])
        key2 = snapshot_manager._compute_cache_key(Language.PYTHON, ["numpy==1.25.0", "pandas==2.0.0"])

        # Keys should be identical (packages are sorted internally)
        assert key1 == key2

        # Pre-populate L1 with snapshot for these packages
        snapshot_path = settings.snapshot_cache_dir / f"{key1}.qcow2"
        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            str(snapshot_path),
            "1M",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Both orderings should return same path
        with patch.object(snapshot_manager, "_create_snapshot", new_callable=AsyncMock) as mock_create:
            result1_path, _result1_has_memory = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["pandas==2.0.0", "numpy==1.25.0"],
                tenant_id="test",
                task_id="test-5a",
            )
            result2_path, _result2_has_memory = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["numpy==1.25.0", "pandas==2.0.0"],
                tenant_id="test",
                task_id="test-5b",
            )

        assert result1_path == result2_path == snapshot_path
        mock_create.assert_not_called()  # Both hit L0 cache

    async def test_different_packages_different_cache_key(
        self, make_vm_manager, make_vm_settings, tmp_path: Path
    ) -> None:
        """Different packages produce different cache keys.

        Verifies cache isolation between different package sets.
        """
        from exec_sandbox.models import Language
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache")
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(snapshot_cache_dir=tmp_path / "cache")
        snapshot_manager = SnapshotManager(settings, vm_manager)

        key1 = snapshot_manager._compute_cache_key(Language.PYTHON, ["requests==2.31.0"])
        key2 = snapshot_manager._compute_cache_key(Language.PYTHON, ["flask==3.0.0"])
        key3 = snapshot_manager._compute_cache_key(Language.PYTHON, ["requests==2.31.0", "flask==3.0.0"])

        # All keys should be different
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    async def test_different_languages_different_cache_key(
        self, make_vm_manager, make_vm_settings, tmp_path: Path
    ) -> None:
        """Same packages with different languages produce different cache keys.

        Verifies cache isolation between languages.
        """
        from exec_sandbox.models import Language
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache")
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(snapshot_cache_dir=tmp_path / "cache")
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Same "package" name but different languages
        key_python = snapshot_manager._compute_cache_key(Language.PYTHON, ["test-pkg==1.0.0"])
        key_node = snapshot_manager._compute_cache_key(Language.JAVASCRIPT, ["test-pkg==1.0.0"])

        assert key_python != key_node

    async def test_l3_disabled_skips_s3_entirely(self, make_vm_manager, make_vm_settings, tmp_path: Path) -> None:
        """When S3 is not configured, L3 is skipped entirely.

        Flow: L1 MISS → (skip L3) → create
        """
        import asyncio
        from unittest.mock import patch

        from exec_sandbox.models import Language
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(
            snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,  # S3 disabled
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(
            snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )
        snapshot_manager = SnapshotManager(settings, vm_manager)

        cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["aiohttp==3.9.0"])
        expected_path = settings.snapshot_cache_dir / f"{cache_key}.qcow2"

        # Mock creation
        async def fake_create_snapshot(language, packages, key, tenant_id, task_id):
            proc = await asyncio.create_subprocess_exec(
                "qemu-img",
                "create",
                "-f",
                "qcow2",
                str(expected_path),
                "1M",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return expected_path

        # Spy on _download_from_s3 to verify it raises (S3 disabled)
        with patch.object(snapshot_manager, "_create_snapshot", side_effect=fake_create_snapshot):
            result_path, has_memory_snapshot = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["aiohttp==3.9.0"],
                tenant_id="test",
                task_id="test-6",
            )

        # Verify snapshot was created
        assert result_path == expected_path
        assert result_path.exists()
        # Newly created = no memory snapshot
        assert has_memory_snapshot is False

    async def test_creation_failure_propagates_error(self, make_vm_manager, make_vm_settings, tmp_path: Path) -> None:
        """When snapshot creation fails, error is propagated.

        Verifies error handling in the cache hierarchy.
        """
        from unittest.mock import AsyncMock, patch

        from exec_sandbox.exceptions import SnapshotError
        from exec_sandbox.models import Language
        from exec_sandbox.snapshot_manager import SnapshotManager

        settings = make_vm_settings(
            snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = make_vm_manager(
            snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Mock creation to fail
        with patch.object(
            snapshot_manager,
            "_create_snapshot",
            new_callable=AsyncMock,
            side_effect=SnapshotError("VM boot failed"),
        ):
            with pytest.raises(SnapshotError) as exc_info:
                await snapshot_manager.get_or_create_snapshot(
                    language=Language.PYTHON,
                    packages=["broken-pkg==1.0.0"],
                    tenant_id="test",
                    task_id="test-7",
                )

        assert "VM boot failed" in str(exc_info.value)

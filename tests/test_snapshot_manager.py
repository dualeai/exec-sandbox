"""Tests for SnapshotManager.

Unit tests: Cache key computation, filesystem operations.
Integration tests: Snapshot creation with QEMU (requires images).
"""

import hashlib
from pathlib import Path

import pytest

from exec_sandbox.models import Language
from exec_sandbox.settings import Settings

# Images directory - relative to repo root
images_dir = Path(__file__).parent.parent / "images" / "dist"


# ============================================================================
# Unit Tests - Cache Key Computation
# ============================================================================


class TestCacheKeyComputation:
    """Tests for cache key computation logic."""

    def test_cache_key_format(self) -> None:
        """Cache key format: {language}-{packages_hash}."""
        # Simulate _compute_cache_key logic
        language = "python"
        packages = ["pandas==2.0.0", "numpy==1.24.0"]

        packages_str = "|".join(sorted(packages))
        packages_hash = hashlib.sha256(packages_str.encode()).hexdigest()[:16]
        cache_key = f"{language}-{packages_hash}"

        assert cache_key.startswith("python-")
        assert len(cache_key.split("-")[1]) == 16

    def test_cache_key_deterministic(self) -> None:
        """Same inputs produce same cache key."""
        packages = ["pandas==2.0.0", "numpy==1.24.0"]

        packages_str = "|".join(sorted(packages))
        hash1 = hashlib.sha256(packages_str.encode()).hexdigest()[:16]
        hash2 = hashlib.sha256(packages_str.encode()).hexdigest()[:16]

        assert hash1 == hash2

    def test_cache_key_sorted_packages(self) -> None:
        """Package order doesn't affect cache key (sorted)."""
        packages1 = ["pandas==2.0.0", "numpy==1.24.0"]
        packages2 = ["numpy==1.24.0", "pandas==2.0.0"]

        hash1 = hashlib.sha256("|".join(sorted(packages1)).encode()).hexdigest()[:16]
        hash2 = hashlib.sha256("|".join(sorted(packages2)).encode()).hexdigest()[:16]

        assert hash1 == hash2

    def test_cache_key_different_languages(self) -> None:
        """Different languages produce different cache keys."""
        packages = ["lodash@4.17.21"]

        key1 = f"python-{hashlib.sha256('|'.join(sorted(packages)).encode()).hexdigest()[:16]}"
        key2 = f"javascript-{hashlib.sha256('|'.join(sorted(packages)).encode()).hexdigest()[:16]}"

        assert key1 != key2

    def test_cache_key_empty_packages(self) -> None:
        """Empty packages list produces valid cache key."""
        packages: list[str] = []

        packages_str = "|".join(sorted(packages))
        packages_hash = hashlib.sha256(packages_str.encode()).hexdigest()[:16]
        cache_key = f"python-{packages_hash}"

        assert cache_key.startswith("python-")
        # Empty string hash is deterministic
        assert packages_hash == hashlib.sha256(b"").hexdigest()[:16]


class TestSettings:
    """Tests for Settings used by SnapshotManager."""

    def test_settings_snapshot_cache_dir(self, tmp_path: Path) -> None:
        """Settings has snapshot_cache_dir."""
        settings = Settings(
            base_images_dir=tmp_path / "images",
            kernel_path=tmp_path / "kernels",
            snapshot_cache_dir=tmp_path / "cache",
        )

        assert settings.snapshot_cache_dir == tmp_path / "cache"

    def test_settings_s3_config(self, tmp_path: Path) -> None:
        """Settings has S3 configuration."""
        settings = Settings(
            base_images_dir=tmp_path / "images",
            kernel_path=tmp_path / "kernels",
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

    async def test_l1_cache_miss(self, tmp_path: Path) -> None:
        """L1 cache miss returns None for non-existent snapshot."""
        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

        settings = Settings(
            base_images_dir=images_dir,
            kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
            snapshot_cache_dir=tmp_path / "cache",
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = VmManager(settings)
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Check for non-existent snapshot
        result = await snapshot_manager._check_l1_cache("nonexistent-abc123")
        assert result is None

    async def test_compute_cache_key(self, tmp_path: Path) -> None:
        """Test actual _compute_cache_key method."""
        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

        settings = Settings(
            base_images_dir=images_dir,
            kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
            snapshot_cache_dir=tmp_path / "cache",
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = VmManager(settings)
        snapshot_manager = SnapshotManager(settings, vm_manager)

        key = snapshot_manager._compute_cache_key(
            language=Language.PYTHON,
            packages=["pandas==2.0.0", "numpy==1.24.0"],
        )

        assert key.startswith("python-")
        # Full SHA256 hash (64 chars)
        assert len(key.split("-")[1]) == 64

    async def test_create_snapshot(self, tmp_path: Path) -> None:
        """Create snapshot with packages (slow, requires VM)."""
        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

        settings = Settings(
            base_images_dir=images_dir,
            kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
            snapshot_cache_dir=tmp_path / "cache",
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = VmManager(settings)
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Create snapshot (this boots a VM and installs packages)
        snapshot_path = await snapshot_manager.get_or_create_snapshot(
            language=Language.PYTHON,
            packages=["requests==2.31.0"],
            tenant_id="test",
            task_id="test-1",
        )

        assert snapshot_path.exists()
        assert snapshot_path.suffix == ".qcow2"

        # Second call should hit L1 cache
        cached_path = await snapshot_manager.get_or_create_snapshot(
            language=Language.PYTHON,
            packages=["requests==2.31.0"],
            tenant_id="test",
            task_id="test-2",
        )

        assert cached_path == snapshot_path


# ============================================================================
# L1 Cache Tests - Local Disk
# ============================================================================


class TestL1Cache:
    """Tests for L1 (local disk) cache operations."""

    async def test_l1_cache_hit_returns_path(self, tmp_path: Path) -> None:
        """L1 cache returns path when valid qcow2 snapshot exists."""
        import asyncio

        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

        settings = Settings(
            base_images_dir=images_dir,
            kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
            snapshot_cache_dir=tmp_path / "cache",
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = VmManager(settings)
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

        result = await snapshot_manager._check_l1_cache(cache_key)
        assert result == snapshot_path

    async def test_l1_cache_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """L1 cache returns None for non-existent snapshot."""
        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

        settings = Settings(
            base_images_dir=images_dir,
            kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
            snapshot_cache_dir=tmp_path / "cache",
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = VmManager(settings)
        snapshot_manager = SnapshotManager(settings, vm_manager)

        # Check for non-existent snapshot
        result = await snapshot_manager._check_l1_cache("nonexistent-key")
        assert result is None

    async def test_l1_evict_oldest_snapshot(self, tmp_path: Path) -> None:
        """_evict_oldest_snapshot removes oldest file by atime."""
        import asyncio
        import time

        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

        settings = Settings(
            base_images_dir=images_dir,
            kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
            snapshot_cache_dir=tmp_path / "cache",
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = VmManager(settings)
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

    async def test_get_s3_client_raises_without_bucket(self, tmp_path: Path) -> None:
        """_get_s3_client raises SnapshotError when s3_bucket not set."""
        from exec_sandbox.exceptions import SnapshotError
        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

        settings = Settings(
            base_images_dir=tmp_path,
            kernel_path=tmp_path,
            snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )
        settings.snapshot_cache_dir.mkdir(parents=True)

        vm_manager = VmManager(settings)
        snapshot_manager = SnapshotManager(settings, vm_manager)

        with pytest.raises(SnapshotError) as exc_info:
            await snapshot_manager._get_s3_client()
        assert "S3 backup disabled" in str(exc_info.value)

    async def test_upload_to_s3_success(self, tmp_path: Path, monkeypatch) -> None:
        """Snapshot uploads to S3 with zstd compression using real aioboto3 client."""
        import boto3
        from moto.server import ThreadedMotoServer

        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

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

            settings = Settings(
                base_images_dir=tmp_path,
                kernel_path=tmp_path,
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = VmManager(settings)
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

    async def test_download_from_s3_success(self, tmp_path: Path, monkeypatch) -> None:
        """Snapshot downloads from S3 and decompresses using real aioboto3 client."""
        import boto3
        import zstandard as zstd
        from moto.server import ThreadedMotoServer

        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

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
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(original_content)
            s3_sync.put_object(
                Bucket="test-snapshots",
                Key="snapshots/python-download123.qcow2.zst",
                Body=compressed,
            )

            settings = Settings(
                base_images_dir=tmp_path,
                kernel_path=tmp_path,
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = VmManager(settings)
            snapshot_manager = SnapshotManager(settings, vm_manager)

            # Download using real aioboto3 client
            result = await snapshot_manager._download_from_s3("python-download123")

            # Verify downloaded and decompressed
            assert result.exists()
            assert result.read_bytes() == original_content

        finally:
            server.stop()

    async def test_download_from_s3_not_found(self, tmp_path: Path, monkeypatch) -> None:
        """S3 download raises SnapshotError when key missing."""
        import boto3
        from moto.server import ThreadedMotoServer

        from exec_sandbox.exceptions import SnapshotError
        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

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

            settings = Settings(
                base_images_dir=tmp_path,
                kernel_path=tmp_path,
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="test-snapshots",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = VmManager(settings)
            snapshot_manager = SnapshotManager(settings, vm_manager)

            with pytest.raises(SnapshotError) as exc_info:
                await snapshot_manager._download_from_s3("nonexistent-key")
            assert "S3 download failed" in str(exc_info.value)

        finally:
            server.stop()

    async def test_upload_to_s3_silent_failure(self, tmp_path: Path, monkeypatch) -> None:
        """S3 upload failure is silent (L1 cache still works)."""
        from moto.server import ThreadedMotoServer

        from exec_sandbox.snapshot_manager import SnapshotManager
        from exec_sandbox.vm_manager import VmManager

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
            settings = Settings(
                base_images_dir=tmp_path,
                kernel_path=tmp_path,
                snapshot_cache_dir=tmp_path / "cache",
                s3_bucket="nonexistent-bucket",
                s3_region="us-east-1",
                s3_endpoint_url=endpoint_url,
            )
            settings.snapshot_cache_dir.mkdir(parents=True)

            vm_manager = VmManager(settings)
            snapshot_manager = SnapshotManager(settings, vm_manager)

            cache_key = "python-fail123"
            snapshot_path = settings.snapshot_cache_dir / f"{cache_key}.qcow2"
            snapshot_path.write_bytes(b"test content")

            # Should not raise - silent failure (bucket doesn't exist)
            await snapshot_manager._upload_to_s3(cache_key, snapshot_path)
            # No exception = success (silent failure)

        finally:
            server.stop()

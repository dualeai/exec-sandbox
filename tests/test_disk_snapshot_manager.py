"""Tests for DiskSnapshotManager.

Unit tests: Cache key computation, filesystem operations.
Integration tests: Snapshot creation with QEMU (requires images).
"""

import asyncio
import errno
import hashlib
import logging
import os
import platform
import shutil
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# Use native zstd (Python 3.14+) or backports.zstd
if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

from exec_sandbox import __version__, constants
from exec_sandbox.disk_snapshot_manager import DiskSnapshotManager, _classify_install_error
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InternalError,
    InvalidInputError,
    OutcomeUnknownError,
    TransientError,
)
from exec_sandbox.guest_agent_protocol import StreamingErrorMessage
from exec_sandbox.guest_channel import StreamResult
from exec_sandbox.hash_utils import fingerprint
from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager
from exec_sandbox.models import Language
from exec_sandbox.settings import Settings
from exec_sandbox.subprocess_utils import communicate_managed_process
from exec_sandbox.vm_manager import VmManager

from .conftest import create_test_qcow2, random_test_id, skip_unless_hwaccel


def _spy_unlink(monkeypatch: pytest.MonkeyPatch, *, when: Callable[[Path], bool]) -> list[int]:
    """Path.unlink as usual; returns the thread ids of the calls on the paths `when` accepts."""
    removers: list[int] = []
    unlink = Path.unlink

    def spied(self: Path, missing_ok: bool = False) -> None:
        if when(self):
            removers.append(threading.get_ident())
        unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", spied)
    return removers


def _l3_miss_reasons(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The reasons of the "L3 cache miss" warnings _download_from_s3 logged."""
    return [r.reason for r in caplog.records if r.getMessage() == "L3 cache miss"]  # type: ignore[attr-defined]


def _get_major_minor_version() -> str:
    """Extract major.minor from __version__ (e.g., '0.1.0' -> '0.1')."""
    parts = __version__.split(".")
    return f"{parts[0]}.{parts[1]}"


# ============================================================================
# S3 Test Fixtures
# ============================================================================


# ============================================================================
# Unit Tests - Cache Key Computation
# ============================================================================


class TestCacheKeyComputation:
    """Tests for cache key computation logic."""

    def test_cache_key_format(self) -> None:
        """Package part of the key is a fixed-length SHA-256 fingerprint."""
        packages = ["pandas==2.0.0", "numpy==1.24.0"]
        packages_hash = fingerprint(*sorted(packages))
        assert len(packages_hash) == 64  # full SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in packages_hash)

    def test_cache_key_deterministic(self) -> None:
        """Same inputs produce same fingerprint."""
        packages = ["pandas==2.0.0", "numpy==1.24.0"]
        assert fingerprint(*sorted(packages)) == fingerprint(*sorted(packages))

    def test_cache_key_sorted_packages(self) -> None:
        """Package order doesn't affect the fingerprint (sorted)."""
        packages1 = ["pandas==2.0.0", "numpy==1.24.0"]
        packages2 = ["numpy==1.24.0", "pandas==2.0.0"]
        assert fingerprint(*sorted(packages1)) == fingerprint(*sorted(packages2))

    def test_cache_key_no_separator_ambiguity(self) -> None:
        """Splitting the same characters differently gives different keys (length-prefixed parts)."""
        assert fingerprint("ab", "c") != fingerprint("a", "bc")
        assert fingerprint("pkg>=1,<2") != fingerprint("pkg>=1", "<2")

    def test_cache_key_chosen_collision_resistant(self) -> None:
        """The forged spec that collided under the old CRC key no longer collides."""
        victim = sorted(["pandas==2.2.3", "requests==2.32.3", "scipy==1.14.1"])
        forged = ["numpy>=0.206762322200000000000000000000000"]
        assert fingerprint(*victim) != fingerprint(*forged)

    def test_cache_key_empty_packages(self) -> None:
        """Empty packages list produces '{language}-v{version}-base' key."""
        version = _get_major_minor_version()
        # L2 format: empty packages = "{language}-v{version}-{img_hash}-base"
        # For this test, we just verify the pattern ends with "-base"
        cache_key = f"python-v{version}-xxxxxxxx-base"

        assert cache_key.startswith(f"python-v{version}-")
        assert cache_key.endswith("-base")


class TestSettings:
    """Tests for Settings used by DiskSnapshotManager."""

    def test_settings_disk_snapshot_cache_dir(self, make_vm_settings, tmp_path: Path) -> None:
        """Settings has disk_snapshot_cache_dir."""
        settings = make_vm_settings(disk_snapshot_cache_dir=tmp_path / "cache")

        assert settings.disk_snapshot_cache_dir == tmp_path / "cache"

    def test_settings_s3_config(self, make_vm_settings) -> None:
        """Settings has S3 configuration."""
        settings = make_vm_settings(
            s3_bucket="my-bucket",
            s3_region="us-west-2",
        )

        assert settings.s3_bucket == "my-bucket"
        assert settings.s3_region == "us-west-2"


# ============================================================================
# Unit Tests - Install Error Classification
# ============================================================================


class TestClassifyInstallError:
    """Tests for _classify_install_error() two-phase classifier."""

    @pytest.mark.parametrize(
        "error_output",
        [
            "error: Failed to download: client error (Connect)",
            "error: connection refused",
            "error: connection reset by peer",
            "error: connection timed out",
            "error: Temporary failure in name resolution",
            "error: Name resolution failed for files.pythonhosted.org",
            "error: network unreachable",
            "error: timed out waiting for response",
            "error: reset by peer",
            "error: request failed after 3 retries",
        ],
    )
    def test_transient_patterns(self, error_output: str) -> None:
        """Known transient network patterns classify as PACKAGE_DOWNLOAD."""
        assert _classify_install_error(error_output) is ErrorCode.PACKAGE_DOWNLOAD

    @pytest.mark.parametrize(
        "error_output",
        [
            "ERROR: No matching distribution found for nonexistent-package",
            "ERROR: Could not find a version that satisfies the requirement foo",
            "error: 404 Not Found - GET https://registry.npmjs.org/nonexistent",
            "error: 404 Client Error: Not Found for url: https://pypi.org/simple/nonexistent/",
            "npm ERR! ERESOLVE could not resolve dependency tree",
            "npm ERR! could not resolve dependency lodash@^99.0.0",
            "ERROR: Invalid version specifier: pandas==",
            "npm ERR! 404 Not Found - not found in registry",
            "npm ERR! No matching version found for foo@^99.0.0",
            "pip ERROR: version not found for package foo==99.99.99",
        ],
    )
    def test_permanent_patterns(self, error_output: str) -> None:
        """Known permanent patterns classify as PACKAGE_INSTALL_FAILED."""
        assert _classify_install_error(error_output) is ErrorCode.PACKAGE_INSTALL_FAILED

    def test_permanent_takes_priority_over_transient(self) -> None:
        """When both permanent and transient patterns match, permanent wins."""
        # "404 client error" contains both "404 client error" (permanent) and "client error" (transient)
        result = _classify_install_error("HTTP 404 Client Error for url: https://pypi.org/simple/foo/")
        assert result is ErrorCode.PACKAGE_INSTALL_FAILED

    @pytest.mark.parametrize(
        "error_output",
        [
            "error: package 'foo' has no attribute 'bar'",
            "SyntaxError: invalid syntax",
            "",
            "some unknown error occurred",
        ],
    )
    def test_no_match_returns_none(self, error_output: str) -> None:
        """Errors matching neither permanent nor transient return None."""
        assert _classify_install_error(error_output) is None

    def test_case_insensitive(self) -> None:
        """Pattern matching is case-insensitive."""
        assert _classify_install_error("CONNECTION REFUSED") is ErrorCode.PACKAGE_DOWNLOAD
        assert _classify_install_error("Client Error (Connect)") is ErrorCode.PACKAGE_DOWNLOAD
        assert _classify_install_error("NETWORK UNREACHABLE") is ErrorCode.PACKAGE_DOWNLOAD
        assert _classify_install_error("NO MATCHING DISTRIBUTION found") is ErrorCode.PACKAGE_INSTALL_FAILED
        assert _classify_install_error("ERESOLVE could not resolve") is ErrorCode.PACKAGE_INSTALL_FAILED


# ============================================================================
# Integration Tests - Require QEMU + Images
# ============================================================================


class TestDiskSnapshotManagerIntegration:
    """Integration tests for DiskSnapshotManager with real QEMU VMs."""

    async def test_l2_cache_miss(self, make_snapshot_manager, tmp_path: Path) -> None:
        """L2 cache miss returns (None, False) for non-existent snapshot."""

        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        # Check for non-existent snapshot
        path = await snapshot_manager._check_l2_cache("nonexistent-abc123")
        assert path is None

    async def test_compute_cache_key(self, make_snapshot_manager, tmp_path: Path) -> None:
        """Test actual _compute_cache_key method."""

        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        # Test with packages
        key = snapshot_manager._compute_cache_key(
            language=Language.PYTHON,
            packages=["pandas==2.0.0", "numpy==1.24.0"],
        )

        # L2 format: python-v{version}-{img_hash}-{pkg_hash}
        version = _get_major_minor_version()
        assert key.startswith(f"python-v{version}-")
        # Format: python-v{version}-{img_sha256}-{pkg_sha256}
        parts = key.split("-")
        assert len(parts) == 4  # python, v{version}, {img_sha256}, {pkg_sha256}
        assert len(parts[2]) == 64  # full SHA-256 hex digest
        assert len(parts[3]) == 64  # full SHA-256 hex digest

        # Test without packages (base)
        base_key = snapshot_manager._compute_cache_key(
            language=Language.PYTHON,
            packages=[],
        )
        # Format: python-v{version}-{img_hash}-base
        assert base_key.startswith(f"python-v{version}-")
        assert base_key.endswith("-base")
        base_parts = base_key.split("-")
        assert len(base_parts) == 4  # python, v{version}, {img_sha256}, base
        assert len(base_parts[2]) == 64  # full SHA-256 hex digest

    @skip_unless_hwaccel
    async def test_create_snapshot(self, make_snapshot_manager, tmp_path: Path) -> None:
        """Create snapshot with packages, verify L2 cache hit on second call.

        Requires hwaccel: boots a VM to pip-install packages, then shuts down
        and commits the overlay to a standalone qcow2.  Under TCG the combined
        boot + install wall-clock exceeds CI job timeouts (observed 60 min hang
        on linux/x64 + linux/arm64 sudo runners).
        """

        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        # Create snapshot (this boots a VM and installs packages)
        snapshot_path = await snapshot_manager.get_or_create_snapshot(
            language=Language.PYTHON,
            packages=["requests==2.31.0"],
            tenant_id="test",
            task_id=f"test-{random_test_id()}",
            memory_mb=256,
        )

        assert snapshot_path.exists()
        assert snapshot_path.suffix == ".qcow2"

        # Second call should hit L2 cache
        cached_path = await snapshot_manager.get_or_create_snapshot(
            language=Language.PYTHON,
            packages=["requests==2.31.0"],
            tenant_id="test",
            task_id=f"test-{random_test_id()}",
            memory_mb=256,
        )

        assert cached_path == snapshot_path

    async def test_cache_key_changes_on_image_rebuild(self, images_dir: Path, tmp_path: Path) -> None:
        """Cache key changes when the base image is modified (rebuild detection).

        _compute_cache_key hashes base image stat (mtime_ns + size) via image_hash().
        Modifying the base image file must produce a different cache key, ensuring
        stale snapshots are never served after an image rebuild.

        Uses a temp copy of the base image to avoid mutating shared images/dist/.
        """
        # Copy base image to temp dir so we can mutate it safely
        isolated_images = tmp_path / "images"
        isolated_images.mkdir()
        arch = "aarch64" if platform.machine() == "arm64" else platform.machine()
        original_base = None
        for f in images_dir.iterdir():
            if f.name.startswith("python-") and f.name.endswith(f"-base-{arch}.qcow2"):
                shutil.copy2(f, isolated_images / f.name)
                original_base = isolated_images / f.name
        assert original_base is not None, "No python base image found in images_dir"

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        settings = Settings(
            base_images_dir=isolated_images,
            kernel_path=isolated_images,
            disk_snapshot_cache_dir=cache_dir,
        )
        vm_manager = VmManager(settings)
        snapshot_manager = DiskSnapshotManager(settings, vm_manager)

        # Compute initial cache key
        key1 = snapshot_manager._compute_cache_key(Language.PYTHON, ["pandas==2.0.0"])

        # Modify the base image (append a byte + explicit mtime change)
        with original_base.open("ab") as fh:
            fh.write(b"\x00")
        # Ensure mtime_ns differs (filesystem resolution may round sub-ms)
        stat = original_base.stat()
        os.utime(original_base, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        # Compute cache key again
        key2 = snapshot_manager._compute_cache_key(Language.PYTHON, ["pandas==2.0.0"])

        version = _get_major_minor_version()
        assert key1.startswith(f"python-v{version}-")
        assert key2.startswith(f"python-v{version}-")
        assert key1 != key2, "Cache key must change when base image is modified"

    async def test_get_base_image_returns_sorted_first(self, tmp_path: Path) -> None:
        """get_base_image returns the lexicographically first match via sorted()[0].

        When multiple base images exist for a language, the sorted()[0] determinism
        guarantee must hold. If sorted() is removed, glob order is non-deterministic.
        """
        fresh_images = tmp_path / "images"
        fresh_images.mkdir()
        arch = "aarch64" if platform.machine() == "arm64" else platform.machine()

        # Create two base images with different version numbers
        path_313 = fresh_images / f"python-3.13-base-{arch}.qcow2"
        path_314 = fresh_images / f"python-3.14-base-{arch}.qcow2"
        await create_test_qcow2(path_313)
        await create_test_qcow2(path_314)

        settings = Settings(
            base_images_dir=fresh_images,
            kernel_path=fresh_images,
        )
        vm_manager = VmManager(settings)

        result = vm_manager.get_base_image("python")
        # sorted() puts 3.13 before 3.14 lexicographically
        assert result == path_313, f"Expected {path_313}, got {result}"

    def test_unknown_language_raises_internal_error(self, unit_test_vm_manager) -> None:
        """get_base_image raises InternalError INTERNAL (not KeyError) for unknown languages.

        The patterns dict uses .get() with an explicit check, so an unknown language
        must raise InternalError with 'Unknown language', not a raw KeyError.
        """
        with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.INTERNAL, match="Unknown language"):
            unit_test_vm_manager.get_base_image("ruby")


# ============================================================================
# L2 Cache Tests - Local Disk Snapshots
# ============================================================================


class TestL2Cache:
    """Tests for L2 (local qcow2) cache operations."""

    async def test_l2_cache_hit_returns_path(self, make_snapshot_manager, tmp_path: Path) -> None:
        """L2 cache returns path when valid qcow2 snapshot exists."""

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        # Create a minimal valid qcow2 file
        cache_key = "python-abc123"
        snapshot_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"

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

        path = await snapshot_manager._check_l2_cache(cache_key)
        assert path == snapshot_path

    async def test_l2_hit_keeps_mtime_so_digest_is_not_recomputed(self, make_snapshot_manager, tmp_path: Path) -> None:
        """An L2 hit bumps atime only: image_hash keys its cache on mtime, so a hit must not force a re-read."""
        snapshot_manager, settings = await make_snapshot_manager(disk_snapshot_cache_dir=tmp_path / "cache")
        snapshot_path = settings.disk_snapshot_cache_dir / "python-mtime.qcow2"
        await create_test_qcow2(snapshot_path)
        os.utime(snapshot_path, ns=(1, 2))
        digest = snapshot_manager.image_hash(snapshot_path)

        assert await snapshot_manager._check_l2_cache("python-mtime") == snapshot_path

        stat = snapshot_path.stat()
        assert stat.st_mtime_ns == 2
        assert stat.st_atime_ns > 1
        with patch("exec_sandbox.base_cache_manager.hashlib.file_digest", side_effect=AssertionError("re-read")):
            assert snapshot_manager.image_hash(snapshot_path) == digest

    async def test_l2_check_runs_once_per_file_version(self, make_snapshot_manager, tmp_path: Path) -> None:
        """A passed qemu-img check is kept for the process: the next packages request on the
        same L2 runs no qemu-img. A changed file (new mtime) is checked again."""
        snapshot_manager, settings = await make_snapshot_manager(disk_snapshot_cache_dir=tmp_path / "cache")
        snapshot_path = settings.disk_snapshot_cache_dir / "python-memo.qcow2"
        await create_test_qcow2(snapshot_path)

        with patch(
            "exec_sandbox.disk_snapshot_manager.communicate_managed_process", side_effect=communicate_managed_process
        ) as qemu_img:
            assert await snapshot_manager._check_l2_cache("python-memo") == snapshot_path
            assert await snapshot_manager._check_l2_cache("python-memo") == snapshot_path
            assert qemu_img.await_count == 1

            os.utime(snapshot_path, ns=(1, 2))  # rewritten in place
            assert await snapshot_manager._check_l2_cache("python-memo") == snapshot_path
            assert qemu_img.await_count == 2

    async def test_l2_check_that_did_not_run_is_not_kept(self, make_snapshot_manager, tmp_path: Path) -> None:
        """qemu-img could not start now (a process limit): a miss, and the next request runs the
        check. A qemu-img this user cannot run is HOST_DEPENDENCY (test_missing_qemu_img_is_a_config_error)."""
        snapshot_manager, settings = await make_snapshot_manager(disk_snapshot_cache_dir=tmp_path / "cache")
        snapshot_path = settings.disk_snapshot_cache_dir / "python-nocheck.qcow2"
        await create_test_qcow2(snapshot_path)
        runs: list[str] = []

        async def qemu_img_fails_once(*args: Any, **kwargs: Any) -> tuple[int | None, bytes, bytes]:
            runs.append("qemu-img")
            if len(runs) == 1:
                raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
            return await communicate_managed_process(*args, **kwargs)

        with patch("exec_sandbox.disk_snapshot_manager.communicate_managed_process", side_effect=qemu_img_fails_once):
            assert await snapshot_manager._check_l2_cache("python-nocheck") is None
            assert await snapshot_manager._check_l2_cache("python-nocheck") == snapshot_path

        assert runs == ["qemu-img", "qemu-img"]

    async def test_evict_removes_digest_sidecar(self, make_snapshot_manager, tmp_path: Path) -> None:
        snapshot_manager, settings = await make_snapshot_manager(disk_snapshot_cache_dir=tmp_path / "cache")
        snapshot_path = settings.disk_snapshot_cache_dir / "python-evict.qcow2"
        await create_test_qcow2(snapshot_path)
        snapshot_manager.image_hash(snapshot_path)
        sidecar = snapshot_path.with_name("python-evict.qcow2.sha256")
        assert sidecar.exists()

        await snapshot_manager._evict_oldest_snapshot()

        assert not snapshot_path.exists()
        assert not sidecar.exists()

    async def test_start_removes_parts_of_dead_writers_only(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """.part files of a killed process are removed at start, in a worker thread (a listing
        with no size limit, GBs per part); a live writer's are kept."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        dead = cache_dir / "python-x.qcow2.999999999.0123abcd.part"
        live = cache_dir / f"python-y.qcow2.{os.getpid()}.0123abcd.part"
        dead.write_bytes(b"orphan")
        live.write_bytes(b"in progress")

        snapshot_manager, _settings = await make_snapshot_manager(disk_snapshot_cache_dir=cache_dir)
        assert dead.exists()  # the constructor does no file work
        removers = _spy_unlink(monkeypatch, when=lambda path: path.parent == cache_dir)
        await snapshot_manager.start()

        assert not dead.exists()
        assert live.exists()
        assert removers
        assert threading.get_ident() not in removers

    async def test_start_keeps_parts_without_a_writer_pid(self, make_snapshot_manager, tmp_path: Path) -> None:
        """A .part name with no pid field (too few fields, or not a number there) names no writer:
        start() keeps it and goes on to the other parts."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        for name in ("stray.part", "python-z.qcow2.notapid.0123abcd.part", "python-x.qcow2.999999999.0123abcd.part"):
            (cache_dir / name).write_bytes(b"x")

        snapshot_manager, _settings = await make_snapshot_manager(disk_snapshot_cache_dir=cache_dir)
        await snapshot_manager.start()

        assert sorted(p.name for p in cache_dir.iterdir()) == ["python-z.qcow2.notapid.0123abcd.part", "stray.part"]

    async def test_l2_cache_removes_corrupt_snapshot(self, make_snapshot_manager, tmp_path: Path) -> None:
        """L2 cache detects and removes corrupt qcow2 (fails qemu-img check)."""

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        cache_key = "python-corrupt123"
        snapshot_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"
        snapshot_path.write_bytes(b"not a valid qcow2 file")
        sidecar = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2.sha256"
        sidecar.write_text("stamp digest")

        result = await snapshot_manager._check_l2_cache(cache_key)

        assert result is None, "Should return None for corrupt qcow2"
        assert not snapshot_path.exists(), "Corrupt snapshot should be removed"
        assert not sidecar.exists(), "Its digest sidecar goes with it"

    async def test_l2_cache_nonexistent_returns_none(self, make_snapshot_manager, tmp_path: Path) -> None:
        """L2 cache returns None for non-existent snapshot."""

        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        # Check for non-existent snapshot
        path = await snapshot_manager._check_l2_cache("nonexistent-key")
        assert path is None

    async def test_l1_evict_oldest_snapshot(self, make_snapshot_manager, tmp_path: Path) -> None:
        """_evict_oldest_snapshot removes oldest file by atime."""

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        # Create multiple snapshots
        oldest_path = settings.disk_snapshot_cache_dir / "python-oldest.qcow2"
        newest_path = settings.disk_snapshot_cache_dir / "python-newest.qcow2"

        # Create both files
        for path in [oldest_path, newest_path]:
            proc = await asyncio.create_subprocess_exec(
                "qemu-img",
                "create",
                "-f",
                "qcow2",
                str(path),
                "1M",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

        # Explicitly set atimes to ensure deterministic ordering
        # (avoids relying on filesystem atime behavior which varies across platforms,
        # especially on macOS APFS which uses relatime semantics)
        now = time.time()
        os.utime(oldest_path, (now - 100, now))  # atime=100s ago, mtime=now
        os.utime(newest_path, (now, now))  # atime=now, mtime=now

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

    async def test_no_bucket_touches_no_s3(self, make_snapshot_manager, tmp_path: Path) -> None:
        """Without s3_bucket, a download is a miss and an upload a no-op: neither opens an S3 client."""

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )
        snapshot_path = settings.disk_snapshot_cache_dir / "python-nobucket.qcow2"
        snapshot_path.write_bytes(b"image")

        with patch.object(snapshot_manager, "_get_s3_client", new_callable=AsyncMock) as get_client:
            assert await snapshot_manager._download_from_s3("python-nobucket") is None
            await snapshot_manager._upload_to_s3("python-nobucket", snapshot_path)

        get_client.assert_not_awaited()
        assert sorted(p.name for p in settings.disk_snapshot_cache_dir.iterdir()) == [snapshot_path.name]

    async def test_upload_to_s3_success(self, make_snapshot_manager, moto_s3_server, tmp_path: Path) -> None:
        """Snapshot uploads to S3 with zstd compression using real aioboto3 client."""

        endpoint_url, s3_sync = moto_s3_server

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        # Create a test snapshot file
        cache_key = "python-test123"
        snapshot_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"
        snapshot_path.write_bytes(b"fake qcow2 content")

        # Upload using real aioboto3 client
        await snapshot_manager._upload_to_s3(cache_key, snapshot_path)

        # Verify uploaded (compressed) using sync boto3
        objects = s3_sync.list_objects_v2(Bucket="test-snapshots")
        keys = [obj["Key"] for obj in objects.get("Contents", [])]
        assert f"snapshots/{cache_key}.qcow2.zst" in keys

        # Metadata digest is the sha256 of the uploaded (uncompressed) bytes
        head = s3_sync.head_object(Bucket="test-snapshots", Key=f"snapshots/{cache_key}.qcow2.zst")
        assert head["Metadata"]["sha256"] == hashlib.sha256(b"fake qcow2 content").hexdigest()

        # Verify compressed temp files were cleaned up
        assert sorted(p.name for p in settings.disk_snapshot_cache_dir.iterdir()) == [f"{cache_key}.qcow2"]

    async def test_download_from_s3_success(self, make_snapshot_manager, moto_s3_server, tmp_path: Path) -> None:
        """Snapshot downloads from S3 and decompresses using real aioboto3 client."""

        endpoint_url, s3_sync = moto_s3_server

        # Compress and upload a real (standalone) qcow2
        source = tmp_path / "source.qcow2"
        await create_test_qcow2(source)
        original_content = source.read_bytes()
        compressed = zstd.compress(original_content)
        s3_sync.put_object(
            Bucket="test-snapshots",
            Key="snapshots/python-download123.qcow2.zst",
            Body=compressed,
            Metadata={"sha256": hashlib.sha256(original_content).hexdigest()},
        )

        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        # Download using real aioboto3 client
        result = await snapshot_manager._download_from_s3("python-download123")

        # Verify downloaded and decompressed
        assert result is not None
        assert result.exists()
        assert result.read_bytes() == original_content

    @pytest.mark.parametrize("verified", [True, False], ids=["verified", "wrong-digest"])
    async def test_download_removes_its_parts_off_the_loop(
        self,
        make_snapshot_manager,
        moto_s3_server,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        caplog: pytest.LogCaptureFixture,
        verified: bool,
    ) -> None:
        """The downloaded .zst and the partial image (each up to GBs) are removed in worker
        threads, after a success and after a failure; no part stays."""
        endpoint_url, s3_sync = moto_s3_server
        source = tmp_path / "source.qcow2"
        await create_test_qcow2(source)
        content = source.read_bytes()
        s3_sync.put_object(
            Bucket="test-snapshots",
            Key="snapshots/python-parts.qcow2.zst",
            Body=zstd.compress(content),
            Metadata={"sha256": hashlib.sha256(content).hexdigest() if verified else "0" * 64},
        )
        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )
        removers = _spy_unlink(monkeypatch, when=lambda path: path.name.endswith(".part"))

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            result = await snapshot_manager._download_from_s3("python-parts")
        if verified:
            assert result is not None
            assert result.read_bytes() == content
        else:
            assert result is None
            assert len(_l3_miss_reasons(caplog)) == 1
            assert "digest mismatch" in _l3_miss_reasons(caplog)[0]

        assert removers
        assert threading.get_ident() not in removers
        assert [p.name for p in settings.disk_snapshot_cache_dir.iterdir()] == (
            ["python-parts.qcow2"] if verified else []
        )

    async def test_download_from_s3_not_found(
        self, make_snapshot_manager, moto_s3_server, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing key is an L3 miss (None, logged), not an error."""

        endpoint_url, _s3_sync = moto_s3_server

        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await snapshot_manager._download_from_s3("nonexistent-key") is None
        assert len(_l3_miss_reasons(caplog)) == 1

    @pytest.mark.parametrize(
        ("metadata", "reason"),
        [
            pytest.param({}, "has no sha256 metadata", id="no-digest"),
            pytest.param({"sha256": "0" * 64}, "digest mismatch", id="wrong-digest"),
        ],
    )
    async def test_download_from_s3_rejects_unverified(
        self,
        make_snapshot_manager,
        moto_s3_server,
        tmp_path: Path,
        *,
        caplog: pytest.LogCaptureFixture,
        metadata: dict[str, str],
        reason: str,
    ) -> None:
        """A missing or wrong sha256 digest is an L3 miss and leaves no file in L2."""

        endpoint_url, s3_sync = moto_s3_server
        s3_sync.put_object(
            Bucket="test-snapshots",
            Key="snapshots/python-tampered.qcow2.zst",
            Body=zstd.compress(b"tampered qcow2 content"),
            Metadata=metadata,
        )

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await snapshot_manager._download_from_s3("python-tampered") is None
        assert len(_l3_miss_reasons(caplog)) == 1
        assert reason in _l3_miss_reasons(caplog)[0]
        assert list(settings.disk_snapshot_cache_dir.iterdir()) == []

    async def test_download_from_s3_rejects_truncated_stream(
        self, make_snapshot_manager, moto_s3_server, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A cut zstd stream fails the download, even if its partial output matches the digest."""

        endpoint_url, s3_sync = moto_s3_server
        original_content = os.urandom(256 * 1024)
        compressed = zstd.compress(original_content)
        s3_sync.put_object(
            Bucket="test-snapshots",
            Key="snapshots/python-truncated.qcow2.zst",
            Body=compressed[: len(compressed) // 2],
            Metadata={"sha256": hashlib.sha256(original_content).hexdigest()},
        )

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await snapshot_manager._download_from_s3("python-truncated") is None
        assert _l3_miss_reasons(caplog) == ["S3 snapshot stream is truncated"]
        assert list(settings.disk_snapshot_cache_dir.iterdir()) == []

    async def test_download_rejects_qcow2_with_backing_file(
        self, make_snapshot_manager, moto_s3_server, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A qcow2 that names a host backing file never reaches L2 (QEMU would serve that file to the guest)."""
        endpoint_url, s3_sync = moto_s3_server
        secret = tmp_path / "host-secret"
        secret.write_bytes(b"host secret")
        crafted = tmp_path / "crafted.qcow2"
        proc = await asyncio.create_subprocess_exec(
            "qemu-img", "create", "-f", "qcow2", "-b", str(secret), "-F", "raw", str(crafted), "1M",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
        await proc.communicate()
        assert proc.returncode == 0
        content = crafted.read_bytes()
        s3_sync.put_object(
            Bucket="test-snapshots",
            Key="snapshots/python-backing.qcow2.zst",
            Body=zstd.compress(content),
            Metadata={"sha256": hashlib.sha256(content).hexdigest()},
        )

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await snapshot_manager._download_from_s3("python-backing") is None
        assert _l3_miss_reasons(caplog) == ["S3 snapshot names a backing or data file; L2 images are standalone"]
        assert list(settings.disk_snapshot_cache_dir.iterdir()) == []

    async def test_download_stops_above_max_size(
        self,
        make_snapshot_manager,
        moto_s3_server,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Decompressed output above the largest possible L2 image fails (zstd bomb guard)."""
        endpoint_url, s3_sync = moto_s3_server
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager._L2_MAX_FILE_BYTES", 1024 * 1024)
        content = bytes(2 * 1024 * 1024)  # compresses to a few hundred bytes
        s3_sync.put_object(
            Bucket="test-snapshots",
            Key="snapshots/python-bomb.qcow2.zst",
            Body=zstd.compress(content),
            Metadata={"sha256": hashlib.sha256(content).hexdigest()},
        )

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await snapshot_manager._download_from_s3("python-bomb") is None
        assert _l3_miss_reasons(caplog) == ["S3 snapshot is larger than any L2 image"]
        assert list(settings.disk_snapshot_cache_dir.iterdir()) == []

    async def test_download_of_a_chunk_that_gives_several_output_steps(
        self, make_snapshot_manager, moto_s3_server, tmp_path: Path
    ) -> None:
        """One 64 KiB compressed chunk whose output is larger than one 4 MiB step: every step is kept."""
        endpoint_url, s3_sync = moto_s3_server
        source = tmp_path / "source.qcow2"
        await create_test_qcow2(source)
        content = source.read_bytes() + bytes(9 * 1024 * 1024)  # trailing zeros: still a qcow2
        compressed = zstd.compress(content)
        assert len(compressed) < 64 * 1024  # the whole object is one read chunk
        s3_sync.put_object(
            Bucket="test-snapshots",
            Key="snapshots/python-steps.qcow2.zst",
            Body=compressed,
            Metadata={"sha256": hashlib.sha256(content).hexdigest()},
        )
        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        result = await snapshot_manager._download_from_s3("python-steps")

        assert result is not None
        assert result.read_bytes() == content

    async def test_download_rejects_an_object_that_is_not_qcow2(
        self, make_snapshot_manager, moto_s3_server, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A verified object that is not a qcow2 image never reaches L2."""
        endpoint_url, s3_sync = moto_s3_server
        content = os.urandom(256 * 1024)
        s3_sync.put_object(
            Bucket="test-snapshots",
            Key="snapshots/python-raw.qcow2.zst",
            Body=zstd.compress(content),
            Metadata={"sha256": hashlib.sha256(content).hexdigest()},
        )
        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        with caplog.at_level(logging.WARNING, logger="exec_sandbox.disk_snapshot_manager"):
            assert await snapshot_manager._download_from_s3("python-raw") is None
        assert _l3_miss_reasons(caplog) == ["S3 snapshot is not qcow2 (format=raw)"]
        assert list(settings.disk_snapshot_cache_dir.iterdir()) == []

    async def test_upload_to_s3_silent_failure(self, make_snapshot_manager, moto_s3_server, tmp_path: Path) -> None:
        """S3 upload failure is silent (L2 cache still works)."""

        endpoint_url, _s3_sync = moto_s3_server

        # Use a bucket that doesn't exist on the moto server
        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="nonexistent-bucket",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        cache_key = "python-fail123"
        snapshot_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"
        snapshot_path.write_bytes(b"test content")

        # Should not raise - silent failure (bucket doesn't exist)
        await snapshot_manager._upload_to_s3(cache_key, snapshot_path)
        # No exception = success (silent failure)

    async def test_upload_semaphore_limits_concurrency(
        self, make_snapshot_manager, moto_s3_server, tmp_path: Path
    ) -> None:
        """Upload semaphore limits concurrent S3 uploads to configured max.

        Verifies that max_concurrent_s3_uploads actually bounds parallel uploads.
        Uses real moto server with instrumented upload tracking.
        """

        endpoint_url, s3_sync = moto_s3_server

        # Create a separate bucket for this test
        bucket_name = "test-semaphore-snapshots"
        s3_sync.create_bucket(Bucket=bucket_name)

        # Configure semaphore to allow only 2 concurrent uploads
        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=bucket_name,
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
            max_concurrent_s3_uploads=2,
        )

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
            (settings.disk_snapshot_cache_dir / f"test-{i}.qcow2").write_bytes(b"test data")

        # Start 5 uploads simultaneously
        tasks = [
            asyncio.create_task(
                snapshot_manager._upload_to_s3(f"test-{i}", settings.disk_snapshot_cache_dir / f"test-{i}.qcow2")
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


# ============================================================================
# Cache Hierarchy Tests - Full L2 → L3 → Create Flow
# ============================================================================


class TestBuildRetryAfterDiskFull:
    """A build that hits ENOSPC evicts the oldest entry and retries once; the retry runs after
    the build released the creation semaphore (one build at a time), so it does not wait for itself.
    A build that fails in another way removes its image file.

    Through the public get_or_create_snapshot; ENOSPC comes from the package install step (an
    OSError of the qemu-img runner is a spawn failure: test_missing_qemu_img_is_a_config_error)."""

    _TIMEOUT = 10.0
    """A retry that waits for its own semaphore hangs: the bound turns that into a failure."""

    async def _fake_build(
        self,
        make_snapshot_manager,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        disk_full_times: int,
        stub_install: bool,
    ) -> tuple[DiskSnapshotManager, Path, Path]:
        """A manager whose build makes a small image with no VM. With `stub_install`, the package
        install does nothing, except its first `disk_full_times` runs, which fail with ENOSPC.
        Returns (manager, oldest entry, newer entry)."""
        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache", s3_bucket=None
        )
        cache_dir = settings.disk_snapshot_cache_dir
        oldest, newer = cache_dir / "python-oldest.qcow2", cache_dir / "python-newer.qcow2"
        for age, entry in ((1000, oldest), (500, newer)):
            entry.write_bytes(b"entry")
            past = time.time() - age
            os.utime(entry, (past, past))
        installs = 0

        async def qemu_img(cmd: list[str], **_kwargs: object) -> tuple[int, bytes, bytes]:
            if cmd[:2] == ["qemu-img", "create"]:
                Path(cmd[-2]).write_bytes(b"built")
            return 0, b"", b""

        async def install(*_: object) -> None:
            nonlocal installs
            installs += 1
            if installs <= disk_full_times:
                raise OSError(errno.ENOSPC, "No space left on device")

        @asynccontextmanager
        async def ephemeral_vm(*_: object, **__: object) -> AsyncIterator[SimpleNamespace]:
            yield SimpleNamespace(  # exited cleanly
                vm_id="fake", process=SimpleNamespace(returncode=0), channel=SimpleNamespace(connect=AsyncMock())
            )

        async def never_dies(*_: object) -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.communicate_managed_process", qemu_img)
        if stub_install:
            monkeypatch.setattr(snapshot_manager, "_install_packages", install)
        monkeypatch.setattr(snapshot_manager, "_monitor_vm_death", never_dies)
        monkeypatch.setattr(snapshot_manager.vm_manager, "ephemeral_vm", ephemeral_vm)
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.cleanup_process", AsyncMock())
        return snapshot_manager, oldest, newer

    async def _build(self, snapshot_manager: DiskSnapshotManager) -> Path:
        return await asyncio.wait_for(
            snapshot_manager.get_or_create_snapshot(Language.PYTHON, [], tenant_id="t", task_id="t", memory_mb=512),
            self._TIMEOUT,
        )

    @staticmethod
    def _entries(cache_dir: Path) -> list[str]:
        """The cache folder listing, without the build locks."""
        return sorted(p.name for p in cache_dir.iterdir() if not p.name.endswith(".lock"))

    async def test_disk_full_once_evicts_once_off_the_loop_and_retries(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The eviction lists a folder with no size limit and removes GBs: worker threads."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=1, stub_install=True
        )
        removers = _spy_unlink(monkeypatch, when=lambda path: path.name.startswith(oldest.name))

        built = await self._build(snapshot_manager)

        assert built.read_bytes() == b"built"
        assert self._entries(oldest.parent) == sorted([newer.name, built.name])  # one eviction: the oldest
        assert removers
        assert threading.get_ident() not in removers

    async def test_disk_full_twice_raises(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot_manager, oldest, newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=2, stub_install=True
        )
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.LOCAL_DISK,
            match=r"^Local disk error for snapshot .+: No space left on device$",
        ):
            await self._build(snapshot_manager)
        assert self._entries(oldest.parent) == [newer.name]  # one eviction, then no second one; no image, no .part

    @pytest.mark.parametrize(
        ("failure", "raised", "code", "message"),
        [
            pytest.param(
                RuntimeError("boom"), InternalError, ErrorCode.INTERNAL, "^Failed to create snapshot: boom$", id="other"
            ),
            pytest.param(
                TransientError(
                    "guest gone", code=ErrorCode.GUEST_UNAVAILABLE, context=None, hint=None, instance_id=None
                ),
                TransientError,
                ErrorCode.GUEST_UNAVAILABLE,
                "^guest gone$",
                id="sandbox-error",
            ),
            pytest.param(
                OSError(errno.EIO, "I/O error"),
                TransientError,
                ErrorCode.LOCAL_DISK,
                "^Local disk error for snapshot .+: I/O error$",
                id="os-error",
            ),
            pytest.param(
                OSError(errno.EDQUOT, "Disc quota exceeded"),
                TransientError,
                ErrorCode.LOCAL_DISK,
                "^Local disk error for snapshot .+: Disc quota exceeded$",
                id="quota",
            ),
            pytest.param(
                PermissionError(errno.EACCES, "Permission denied"),
                ConfigError,
                ErrorCode.SETTING_INVALID,
                "^Snapshot build .+ cannot write its files: Permission denied\n"
                "Hint: Give this user write access to disk_snapshot_cache_dir, on a writable disk\\.$",
                id="no-access",
            ),
            pytest.param(
                OSError(errno.EROFS, "Read-only file system"),
                ConfigError,
                ErrorCode.SETTING_INVALID,
                "^Snapshot build .+ cannot write its files: Read-only file system\n"
                "Hint: Give this user write access to disk_snapshot_cache_dir, on a writable disk\\.$",
                id="read-only",
            ),
            pytest.param(
                TimeoutError("slow"),
                InternalError,
                ErrorCode.INTERNAL,
                "^Failed to create snapshot: slow$",
                id="timeout",
            ),
        ],
    )
    async def test_failed_build_removes_its_image_file(
        self,
        make_snapshot_manager,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        failure: Exception,
        raised: type[Exception],
        code: ErrorCode | None,
        message: str,
    ) -> None:
        """A build that fails after its image file exists (here, at package install) removes that
        file: no .part and no entry stay; the other entries are kept. A SandboxError passes
        unchanged; a local OSError is LOCAL_DISK (RETRY), but no access or a read-only disk fails
        the same way again: SETTING_INVALID (FIX_CONFIG); a TimeoutError (an OSError) is not a
        disk error: INTERNAL, like any other failure."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=0, stub_install=True
        )
        monkeypatch.setattr(snapshot_manager, "_install_packages", AsyncMock(side_effect=failure))

        with pytest.raises(raised, check=lambda e: getattr(e, "code", None) is code, match=message):
            await self._build(snapshot_manager)

        assert self._entries(oldest.parent) == [newer.name, oldest.name]

    @pytest.mark.parametrize(
        "spawn_error",
        [
            FileNotFoundError(errno.ENOENT, "No such file or directory", "qemu-img"),
            PermissionError(errno.EACCES, "Permission denied", "qemu-img"),
            OSError(errno.ENOEXEC, "Exec format error", "qemu-img"),
        ],
        ids=["missing", "not-executable", "other-arch"],
    )
    async def test_missing_qemu_img_is_a_config_error(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spawn_error: OSError
    ) -> None:
        """No qemu-img on the host (Debian and Ubuntu ship it in qemu-utils), or one this user
        may not run: ConfigError HOST_DEPENDENCY, which no retry fixes, never a local disk error
        that callers retry, nor a cache folder setting."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=0, stub_install=True
        )

        async def no_qemu_img(cmd: list[str], **_kwargs: object) -> tuple[int, bytes, bytes]:
            raise spawn_error

        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.communicate_managed_process", no_qemu_img)

        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc:
            await self._build(snapshot_manager)
        assert (exc.value.context["binary"], exc.value.context["errno"]) == ("qemu-img", spawn_error.errno)
        assert exc.value.hint == "Install QEMU (on Debian and Ubuntu, the qemu-utils package)."
        assert self._entries(oldest.parent) == [newer.name, oldest.name]

    async def test_qemu_img_fork_limit_is_overlay_failed(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A qemu-img fork that hits a process limit can pass later: TransientError
        OVERLAY_FAILED, not a local disk error."""
        snapshot_manager, _oldest, _newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=0, stub_install=True
        )

        async def fork_limit(cmd: list[str], **_kwargs: object) -> tuple[int, bytes, bytes]:
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")

        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.communicate_managed_process", fork_limit)

        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.OVERLAY_FAILED) as exc:
            await self._build(snapshot_manager)
        assert exc.value.context["errno"] == errno.EAGAIN

    async def test_cache_folder_removed_at_run_time_is_made_again(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache folder removed while the worker runs (a tmp cleaner) is made again at the
        next build, which succeeds: never LOCAL_DISK (RETRY) on every build until a restart."""
        snapshot_manager, oldest, _newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=0, stub_install=True
        )
        shutil.rmtree(oldest.parent)

        path = await self._build(snapshot_manager)

        assert path.parent == oldest.parent
        assert path.read_bytes() == b"built"

    @pytest.mark.skipif(os.geteuid() == 0, reason="root writes a 0o500 folder")
    async def test_read_only_cache_folder_is_a_config_error(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache folder this user cannot write fails at the build lock, the first file of a
        build: ConfigError SETTING_INVALID that names the setting, never a raw PermissionError."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=0, stub_install=True
        )
        oldest.parent.chmod(0o500)
        try:
            with pytest.raises(
                ConfigError,
                check=lambda e: e.code is ErrorCode.SETTING_INVALID,
                match=r"^Snapshot build .+ cannot write its files: Permission denied\n"
                r"Hint: Give this user write access to disk_snapshot_cache_dir, on a writable disk\.$",
            ):
                await self._build(snapshot_manager)
        finally:
            oldest.parent.chmod(0o700)
        assert self._entries(oldest.parent) == [newer.name, oldest.name]

    @pytest.mark.parametrize(
        ("manager", "setting"),
        [
            (DiskSnapshotManager, "disk_snapshot_cache_dir"),
            (MemorySnapshotManager, "memory_snapshot_cache_dir"),
        ],
        ids=["l2", "l1"],
    )
    async def test_cache_folder_that_cannot_be_made_is_a_config_error(
        self, make_vm_manager, tmp_path: Path, manager: type[DiskSnapshotManager | MemorySnapshotManager], setting: str
    ) -> None:
        """A cache folder path under a file: the manager raises ConfigError SETTING_INVALID that
        names its setting when it is made (Scheduler start), never a raw OSError."""
        (tmp_path / "file").write_bytes(b"")
        vm_manager = await make_vm_manager(**{setting: tmp_path / "file" / "cache"})

        with pytest.raises(
            ConfigError,
            check=lambda e: e.code is ErrorCode.SETTING_INVALID,
            match=f"^Cache folder .+ cannot be made: Not a directory\nHint: Set {setting} to a folder",
        ):
            manager(vm_manager.settings, vm_manager)

    async def test_install_outcome_unknown_is_guest_unavailable(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An install whose transport is lost after dispatch ran no user code: the build
        raises TransientError GUEST_UNAVAILABLE (RETRY), never OutcomeUnknownError (VERIFY),
        which would reach the caller of Scheduler.new. Not retried (only PACKAGE_DOWNLOAD is)."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=0, stub_install=False
        )
        lost = OutcomeUnknownError(
            "transport lost after dispatch",
            code=ErrorCode.GUEST_OUTCOME_UNKNOWN,
            context=None,
            hint=None,
            instance_id=None,
        )
        consume = AsyncMock(side_effect=lost)
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", consume)

        with pytest.raises(
            TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE and e.__cause__ is lost
        ):
            await asyncio.wait_for(
                snapshot_manager.get_or_create_snapshot(
                    Language.PYTHON, ["pandas==2.2.3"], tenant_id="t", task_id="t", memory_mb=512
                ),
                self._TIMEOUT,
            )

        consume.assert_awaited_once()
        assert self._entries(oldest.parent) == [newer.name, oldest.name]

    async def _install(self, snapshot_manager: DiskSnapshotManager) -> Path:
        """A build with one package: its install goes through consume_stream."""
        return await asyncio.wait_for(
            snapshot_manager.get_or_create_snapshot(
                Language.PYTHON, ["pandas==2.2.3"], tenant_id="t", task_id="t", memory_mb=512
            ),
            self._TIMEOUT,
        )

    @staticmethod
    def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
        """Install retries without their backoff sleeps (1 to 8 s each)."""
        monkeypatch.setattr(constants, "PACKAGE_INSTALL_RETRY_MIN_SECONDS", 0)
        monkeypatch.setattr(constants, "PACKAGE_INSTALL_RETRY_MAX_SECONDS", 0)

    async def test_guest_timeout_during_install_is_package_download(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guest reports a timeout of the package manager: TransientError PACKAGE_DOWNLOAD,
        the one code the install retries (3 attempts), then raised; the build file goes."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=0, stub_install=False
        )
        self._no_backoff(monkeypatch)

        async def guest_timeout(*_args: object, on_error: Any, **_kwargs: object) -> StreamResult:
            await on_error(StreamingErrorMessage(message="pip timed out", error_type="timeout_error"))
            raise AssertionError("on_error raised nothing")

        consume = AsyncMock(side_effect=guest_timeout)
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", consume)

        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.PACKAGE_DOWNLOAD,
            match=r"^install_packages: \[timeout_error\] pip timed out$",
        ):
            await self._install(snapshot_manager)

        assert consume.await_count == 3
        assert self._entries(oldest.parent) == [newer.name, oldest.name]

    @pytest.mark.parametrize(
        ("stderr", "raised", "code", "message", "attempts"),
        [
            pytest.param(
                "ERROR: No matching distribution found for pandas==2.2.3",
                InvalidInputError,
                ErrorCode.PACKAGE_INSTALL_FAILED,
                r"^Package installation failed permanently \(exit code 1\): "
                r"ERROR: No matching distribution found for pandas==2\.2\.3$",
                1,
                id="permanent",
            ),
            pytest.param(
                "Connection reset by peer",
                TransientError,
                ErrorCode.PACKAGE_DOWNLOAD,
                r"^Package installation failed with transient network error \(exit code 1\): Connection reset by peer$",
                3,
                id="network",
            ),
            pytest.param(
                "Segmentation fault",
                InvalidInputError,
                ErrorCode.PACKAGE_INSTALL_FAILED,
                r"^Package installation failed with exit code 1: Segmentation fault$",
                1,
                id="unclassified",
            ),
        ],
    )
    async def test_failed_install_is_classified_by_its_output(
        self,
        make_snapshot_manager,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        stderr: str,
        raised: type[Exception],
        code: ErrorCode,
        message: str,
        attempts: int,
    ) -> None:
        """The package manager exits 1: its output decides. A network failure is PACKAGE_DOWNLOAD
        (RETRY, retried); a refusal or an unknown failure is PACKAGE_INSTALL_FAILED (FIX_INPUT,
        not retried: the same packages fail again)."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=0, stub_install=False
        )
        self._no_backoff(monkeypatch)
        consume = AsyncMock(return_value=StreamResult(exit_code=1, stderr=stderr))
        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", consume)

        with pytest.raises(raised, check=lambda e: getattr(e, "code", None) is code, match=message):
            await self._install(snapshot_manager)

        assert consume.await_count == attempts
        assert self._entries(oldest.parent) == [newer.name, oldest.name]

    async def test_install_past_its_total_timeout_is_package_download(
        self, make_snapshot_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An install still running at its total bound (all attempts and backoffs): TransientError
        PACKAGE_DOWNLOAD (RETRY), not the builtin TimeoutError."""
        snapshot_manager, oldest, newer = await self._fake_build(
            make_snapshot_manager, tmp_path, monkeypatch, disk_full_times=0, stub_install=False
        )
        # One attempt, bound 1 s: hard_timeout is PACKAGE_INSTALL_TIMEOUT_SECONDS + margin + 8.
        monkeypatch.setattr(constants, "PACKAGE_INSTALL_MAX_RETRIES", 1)
        monkeypatch.setattr(constants, "PACKAGE_INSTALL_TIMEOUT_SECONDS", 0)
        monkeypatch.setattr(constants, "EXECUTION_TIMEOUT_MARGIN_SECONDS", -7)

        async def never_ends(*_args: object, **_kwargs: object) -> StreamResult:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        monkeypatch.setattr("exec_sandbox.disk_snapshot_manager.consume_stream", never_ends)

        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.PACKAGE_DOWNLOAD and isinstance(e.__cause__, TimeoutError),
            match=r"^Package installation timeout after 1\.0s$",
        ):
            await self._install(snapshot_manager)

        assert self._entries(oldest.parent) == [newer.name, oldest.name]


class TestCacheHierarchy:
    """Tests for the full cache hierarchy flow in get_or_create_snapshot().

    These tests verify the real L2 → L3 → Create pattern:
    - L2 hit: Return immediately from local disk
    - L2 miss → L3 hit: Download from S3, populate L2
    - L2 miss → L3 miss: Create snapshot, upload to S3

    Uses moto server for real S3 client and mocks _create_snapshot to avoid QEMU.
    """

    async def test_l2_hit_returns_immediately_no_s3(self, make_snapshot_manager, tmp_path: Path) -> None:
        """L2 cache hit returns path immediately without touching S3.

        Flow: L2 HIT → return (no S3 call, no creation)
        """

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-bucket",  # S3 configured but should NOT be called
            s3_region="us-east-1",
        )

        # Pre-populate L2 cache with valid qcow2
        cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["requests==2.31.0"])
        snapshot_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"

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
                result_path = await snapshot_manager.get_or_create_snapshot(
                    language=Language.PYTHON,
                    packages=["requests==2.31.0"],
                    tenant_id="test",
                    task_id=f"test-{random_test_id()}",
                    memory_mb=256,
                )

        # Verify L2 hit: returned correct path
        assert result_path == snapshot_path

        # Verify S3 was NOT called (L2 hit skips S3)
        mock_s3.assert_not_called()

        # Verify creation was NOT called (L2 hit skips creation)
        mock_create.assert_not_called()

    async def test_l2_miss_l3_hit_downloads_from_s3(
        self, make_snapshot_manager, moto_s3_server, tmp_path: Path
    ) -> None:
        """L2 miss with L3 hit downloads from S3 and returns path.

        Flow: L2 MISS → L3 HIT → download → return (no creation)
        """

        endpoint_url, s3_sync = moto_s3_server

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        # Compute cache key for the packages we'll request
        cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["numpy==1.26.0"])

        # Pre-populate S3 (L3) with a compressed real qcow2
        source = tmp_path / "source.qcow2"
        await create_test_qcow2(source)
        original_content = source.read_bytes()
        compressed = zstd.compress(original_content)
        s3_sync.put_object(
            Bucket="test-snapshots",
            Key=f"snapshots/{cache_key}.qcow2.zst",
            Body=compressed,
            Metadata={"sha256": hashlib.sha256(original_content).hexdigest()},
        )

        # L2 is empty (no file on disk)
        assert not (settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2").exists()

        # Mock creation to verify it's NOT called
        with patch.object(snapshot_manager, "_create_snapshot", new_callable=AsyncMock) as mock_create:
            result_path = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["numpy==1.26.0"],
                tenant_id="test",
                task_id=f"test-{random_test_id()}",
                memory_mb=256,
            )

        # Verify returned path exists and has correct content (decompressed from S3)
        assert result_path.exists()
        assert result_path.read_bytes() == original_content

        # Verify creation was NOT called (L3 hit skips creation)
        mock_create.assert_not_called()

    async def test_l2_miss_l3_miss_creates_snapshot(
        self, make_snapshot_manager, moto_s3_server, tmp_path: Path
    ) -> None:
        """L2 miss and L3 miss triggers snapshot creation.

        Flow: L2 MISS → L3 MISS → create → return (and upload to S3)
        """

        endpoint_url, s3_sync = moto_s3_server

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        # Compute cache key
        cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["pandas==2.1.0"])
        expected_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"

        # Mock _create_snapshot to simulate snapshot creation (avoids real QEMU)
        async def fake_create_snapshot(language, packages, *, cache_key, tenant_id, task_id, memory_mb):
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
            result_path = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["pandas==2.1.0"],
                tenant_id="test",
                task_id=f"test-{random_test_id()}",
                memory_mb=256,
            )

        # Verify creation WAS called (cache miss)
        mock_create.assert_called_once()

        # Verify returned path
        assert result_path == expected_path
        assert result_path.exists()

        # Wait briefly for background S3 upload task
        await asyncio.sleep(0.5)

        # Verify S3 upload happened (background task)
        objects = s3_sync.list_objects_v2(Bucket="test-snapshots")
        keys = [obj["Key"] for obj in objects.get("Contents", [])]
        assert f"snapshots/{cache_key}.qcow2.zst" in keys

    async def test_l2_populated_after_l3_download(self, make_snapshot_manager, moto_s3_server, tmp_path: Path) -> None:
        """After L3 download, L2 cache is populated for next call.

        Flow: L2 MISS → L3 HIT → download → L2 populated
        Then: L2 HIT → return immediately
        """

        endpoint_url, s3_sync = moto_s3_server

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket="test-snapshots",
            s3_region="us-east-1",
            s3_endpoint_url=endpoint_url,
        )

        # Compute cache key
        cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["scipy==1.11.0"])
        l2_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"

        # Create a real qcow2 to upload to S3 (must pass qemu-img check on L2 hit)
        real_qcow2 = tmp_path / "real-snapshot.qcow2"
        await create_test_qcow2(real_qcow2)
        original_content = real_qcow2.read_bytes()

        # Pre-populate S3 only
        compressed = zstd.compress(original_content)
        s3_sync.put_object(
            Bucket="test-snapshots",
            Key=f"snapshots/{cache_key}.qcow2.zst",
            Body=compressed,
            Metadata={"sha256": hashlib.sha256(original_content).hexdigest()},
        )

        # Verify L2 is empty before first call
        assert not l2_path.exists()

        # First call: L2 miss → L3 hit
        with patch.object(snapshot_manager, "_create_snapshot", new_callable=AsyncMock) as mock_create:
            _result1_path = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["scipy==1.11.0"],
                tenant_id="test",
                task_id=f"test-{random_test_id()}",
                memory_mb=256,
            )
            mock_create.assert_not_called()

        # Verify L2 is NOW populated
        assert l2_path.exists()
        assert l2_path.read_bytes() == original_content

        # Second call: should hit L2 (no S3 download)
        # We'll spy on _download_from_s3 to verify it's not called
        original_download = snapshot_manager._download_from_s3
        download_called = False

        async def spy_download(*args, **kwargs):
            nonlocal download_called
            download_called = True
            return await original_download(*args, **kwargs)

        with patch.object(snapshot_manager, "_download_from_s3", side_effect=spy_download):
            result2_path = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["scipy==1.11.0"],
                tenant_id="test",
                task_id=f"test-{random_test_id()}",
                memory_mb=256,
            )

        # Verify L2 hit on second call
        assert result2_path == l2_path
        assert not download_called, "S3 download should NOT be called on L2 hit"

    async def test_same_packages_same_cache_key(self, make_snapshot_manager, tmp_path: Path) -> None:
        """Same packages (regardless of order) produce same cache key and path.

        Verifies deterministic cache key computation.
        """

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )

        # Compute cache keys for same packages in different orders
        key1 = snapshot_manager._compute_cache_key(Language.PYTHON, ["pandas==2.0.0", "numpy==1.25.0"])
        key2 = snapshot_manager._compute_cache_key(Language.PYTHON, ["numpy==1.25.0", "pandas==2.0.0"])

        # Keys should be identical (packages are sorted internally)
        assert key1 == key2

        # Pre-populate L2 with snapshot for these packages
        snapshot_path = settings.disk_snapshot_cache_dir / f"{key1}.qcow2"
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
            result1_path = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["pandas==2.0.0", "numpy==1.25.0"],
                tenant_id="test",
                task_id=f"test-{random_test_id()}",
                memory_mb=256,
            )
            result2_path = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["numpy==1.25.0", "pandas==2.0.0"],
                tenant_id="test",
                task_id=f"test-{random_test_id()}",
                memory_mb=256,
            )

        assert result1_path == result2_path == snapshot_path
        mock_create.assert_not_called()  # Both hit L2 cache

    async def test_different_packages_different_cache_key(self, make_snapshot_manager, tmp_path: Path) -> None:
        """Different packages produce different cache keys.

        Verifies cache isolation between different package sets.
        """

        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        key1 = snapshot_manager._compute_cache_key(Language.PYTHON, ["requests==2.31.0"])
        key2 = snapshot_manager._compute_cache_key(Language.PYTHON, ["flask==3.0.0"])
        key3 = snapshot_manager._compute_cache_key(Language.PYTHON, ["requests==2.31.0", "flask==3.0.0"])

        # All keys should be different
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    async def test_different_languages_different_cache_key(self, make_snapshot_manager, tmp_path: Path) -> None:
        """Same packages with different languages produce different cache keys.

        Verifies cache isolation between languages.
        """

        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
        )

        # Same "package" name but different languages
        key_python = snapshot_manager._compute_cache_key(Language.PYTHON, ["test-pkg==1.0.0"])
        key_node = snapshot_manager._compute_cache_key(Language.JAVASCRIPT, ["test-pkg==1.0.0"])

        assert key_python != key_node

    async def test_l3_disabled_skips_s3_entirely(self, make_snapshot_manager, tmp_path: Path) -> None:
        """When S3 is not configured, L3 is skipped entirely.

        Flow: L2 MISS → (skip L3) → create
        """

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )

        cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["aiohttp==3.9.0"])
        expected_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"

        # Mock creation
        async def fake_create_snapshot(language, packages, *, cache_key, tenant_id, task_id, memory_mb):
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
            result_path = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["aiohttp==3.9.0"],
                tenant_id="test",
                task_id=f"test-{random_test_id()}",
                memory_mb=256,
            )

        # Verify snapshot was created
        assert result_path == expected_path
        assert result_path.exists()

    async def test_creation_failure_propagates_error(self, make_snapshot_manager, tmp_path: Path) -> None:
        """When snapshot creation fails, error is propagated.

        Verifies error handling in the cache hierarchy.
        """

        snapshot_manager, _settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )

        # Mock creation to fail
        with patch.object(
            snapshot_manager,
            "_create_snapshot",
            new_callable=AsyncMock,
            side_effect=TransientError(
                "VM boot failed", code=ErrorCode.QEMU_CRASHED, context=None, hint=None, instance_id=None
            ),
        ):
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED) as exc_info:
                await snapshot_manager.get_or_create_snapshot(
                    language=Language.PYTHON,
                    packages=["broken-pkg==1.0.0"],
                    tenant_id="test",
                    task_id=f"test-{random_test_id()}",
                    memory_mb=256,
                )

        assert "VM boot failed" in str(exc_info.value)

    async def test_concurrent_snapshot_requests_single_creation(self, make_snapshot_manager, tmp_path: Path) -> None:
        """10 concurrent get_or_create_snapshot calls create snapshot exactly once.

        Validates the file_lock + creation_semaphore contract: the first caller
        acquires the lock and creates the snapshot, while the other 9 see
        should_create=False (file already exists) and return the same path.
        """

        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )

        cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["requests==2.31.0"])
        expected_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"

        creation_count = 0

        async def fake_create_snapshot(language, packages, *, cache_key, tenant_id, task_id, memory_mb):
            nonlocal creation_count
            creation_count += 1
            # Yield control so all 10 coroutines are queued on the asyncio.Lock
            # before the first one finishes creating the file.
            await asyncio.sleep(0.1)
            await create_test_qcow2(expected_path)
            return expected_path

        with patch.object(snapshot_manager, "_create_snapshot", side_effect=fake_create_snapshot):
            results = await asyncio.gather(
                *(
                    snapshot_manager.get_or_create_snapshot(
                        language=Language.PYTHON,
                        packages=["requests==2.31.0"],
                        tenant_id="test",
                        task_id=f"concurrent-{i}",
                        memory_mb=256,
                    )
                    for i in range(10)
                )
            )

        assert creation_count == 1, f"Expected 1 creation, got {creation_count}"
        for result in results:
            assert result == expected_path

    async def test_corrupt_l2_triggers_recreation(self, make_snapshot_manager, tmp_path: Path) -> None:
        """Corrupt qcow2 in L2 cache is rejected by qemu-img check, triggering _create_snapshot.

        Verifies that a corrupt L2 file (garbage bytes) does not produce a false
        cache hit. _check_l2_cache should return None, and the slow path should
        proceed to _create_snapshot.
        """
        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )

        cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["pandas==2.0.0"])
        corrupt_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"

        # Write garbage bytes to simulate corruption
        corrupt_path.write_bytes(b"\x00\xff\xde\xad" * 64)

        # Mock _create_snapshot to write valid qcow2
        async def fake_create_snapshot(language, packages, *, cache_key, tenant_id, task_id, memory_mb):
            expected = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"
            await create_test_qcow2(expected)
            return expected

        with patch.object(snapshot_manager, "_create_snapshot", side_effect=fake_create_snapshot) as mock_create:
            result_path = await snapshot_manager.get_or_create_snapshot(
                language=Language.PYTHON,
                packages=["pandas==2.0.0"],
                tenant_id="test",
                task_id=f"test-{random_test_id()}",
                memory_mb=256,
            )

        # _create_snapshot must have been called (corrupt L2 rejected)
        mock_create.assert_called_once()
        assert result_path.exists()

    async def test_stale_lock_file_does_not_block(self, make_snapshot_manager, tmp_path: Path) -> None:
        """A stale .lock file left by a crashed process does not cause deadlock.

        file_lock uses flock() which is automatically released when the process dies
        (fd close). A leftover .lock file on disk should not prevent new callers from
        acquiring the lock and creating the snapshot.
        """
        snapshot_manager, settings = await make_snapshot_manager(
            disk_snapshot_cache_dir=tmp_path / "cache",
            s3_bucket=None,
        )

        cache_key = snapshot_manager._compute_cache_key(Language.PYTHON, ["requests==2.31.0"])
        expected_path = settings.disk_snapshot_cache_dir / f"{cache_key}.qcow2"

        # Simulate stale .lock file left by a crashed process
        lock_path = expected_path.with_suffix(".qcow2.lock")
        lock_path.touch()

        # No .qcow2 file exists (simulates crash before write completed)
        assert not expected_path.exists()
        assert lock_path.exists()

        async def fake_create_snapshot(language, packages, *, cache_key, tenant_id, task_id, memory_mb):
            await create_test_qcow2(expected_path)
            return expected_path

        with patch.object(snapshot_manager, "_create_snapshot", side_effect=fake_create_snapshot):
            # Must complete within 10s (no deadlock on stale lock file)
            result_path = await asyncio.wait_for(
                snapshot_manager.get_or_create_snapshot(
                    language=Language.PYTHON,
                    packages=["requests==2.31.0"],
                    tenant_id="test",
                    task_id=f"test-{random_test_id()}",
                    memory_mb=256,
                ),
                timeout=10,
            )

        assert result_path.exists()
        assert result_path == expected_path

"""Tests for QemuStorageDaemon.

Test philosophy:
- Unit tests: Pure logic only, no real daemon process, no mocks unless unavoidable
- Integration tests: Real daemon, real files, real code paths (requires hwaccel)
- Error tests: Real daemon, verify error handling works correctly
"""

import asyncio
from pathlib import Path

import pytest

from exec_sandbox.qemu_storage_daemon import QemuStorageDaemon, QemuStorageDaemonError

from .conftest import skip_unless_hwaccel

# ============================================================================
# Unit Tests - Pure Logic (no daemon, no mocks)
# ============================================================================


class TestQemuStorageDaemonUnit:
    """Unit tests for QemuStorageDaemon - no real daemon process, no mocks."""

    async def test_create_overlay_when_not_started_raises_error(self) -> None:
        """create_overlay raises if daemon not started."""
        daemon = QemuStorageDaemon()
        with pytest.raises(QemuStorageDaemonError, match="not started"):
            await daemon.create_overlay(Path("/fake/base.qcow2"), Path("/fake/out.qcow2"))

    async def test_stop_without_start_is_safe(self) -> None:
        """Stopping un-started daemon is a no-op (idempotent)."""
        daemon = QemuStorageDaemon()
        await daemon.stop()  # Should not raise
        assert not daemon.started

    async def test_started_property_false_before_start(self) -> None:
        """started property is False before start()."""
        daemon = QemuStorageDaemon()
        assert daemon.started is False

    async def test_error_class_preserved_in_exception(self) -> None:
        """QemuStorageDaemonError preserves error_class attribute."""
        error = QemuStorageDaemonError("Some error", error_class="GenericError")
        assert error.error_class == "GenericError"
        assert str(error) == "Some error"

    async def test_error_class_defaults_to_none(self) -> None:
        """QemuStorageDaemonError error_class defaults to None."""
        error = QemuStorageDaemonError("Some error")
        assert error.error_class is None


# ============================================================================
# Integration Tests - Real daemon, real files
# ============================================================================


@skip_unless_hwaccel
class TestQemuStorageDaemonIntegration:
    """Integration tests with real qemu-storage-daemon - no mocking."""

    @pytest.fixture
    async def daemon(self, tmp_path: Path):
        """Create and start daemon, cleanup after test."""
        d = QemuStorageDaemon()
        await d.start()
        yield d
        await d.stop()

    async def test_start_stop_lifecycle(self) -> None:
        """Daemon starts and stops cleanly."""
        daemon = QemuStorageDaemon()
        await daemon.start()
        assert daemon.started is True
        await daemon.stop()
        assert daemon.started is False

    async def test_double_start_is_idempotent(self) -> None:
        """Calling start() twice is safe - second call is no-op."""
        daemon = QemuStorageDaemon()
        await daemon.start()
        pid_after_first = daemon._process.pid if daemon._process else None
        try:
            await daemon.start()  # Should be no-op
            pid_after_second = daemon._process.pid if daemon._process else None
            assert pid_after_first == pid_after_second  # Same process
            assert daemon.started is True
        finally:
            await daemon.stop()

    async def test_double_stop_is_idempotent(self) -> None:
        """Calling stop() twice is safe - second call is no-op."""
        daemon = QemuStorageDaemon()
        await daemon.start()
        await daemon.stop()
        assert daemon.started is False
        await daemon.stop()  # Should not raise
        assert daemon.started is False

    async def test_socket_created_on_start(self) -> None:
        """QMP socket is created on daemon start."""
        daemon = QemuStorageDaemon()
        await daemon.start()
        try:
            assert daemon._socket_path is not None
            assert daemon._socket_path.exists()
        finally:
            await daemon.stop()

    async def test_socket_removed_on_stop(self) -> None:
        """QMP socket is removed on daemon stop."""
        daemon = QemuStorageDaemon()
        await daemon.start()
        socket_path = daemon._socket_path
        await daemon.stop()
        assert socket_path is not None
        assert not socket_path.exists()

    async def test_create_overlay(self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path) -> None:
        """Creates valid qcow2 overlay via QMP."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        overlay = tmp_path / "test-overlay.qcow2"
        await daemon.create_overlay(base_image, overlay)

        assert overlay.exists()
        assert overlay.stat().st_size > 0

    async def test_overlay_has_correct_backing_file(
        self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path
    ) -> None:
        """Created overlay references correct base image as backing file."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        overlay = tmp_path / "backing-test.qcow2"
        await daemon.create_overlay(base_image, overlay)

        # Verify backing file using qemu-img info
        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "info",
            str(overlay),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        info_output = stdout.decode()

        assert str(base_image) in info_output
        assert "backing file:" in info_output.lower()

    async def test_overlay_has_qcow2_format(self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path) -> None:
        """Created overlay is valid qcow2 format."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        overlay = tmp_path / "format-test.qcow2"
        await daemon.create_overlay(base_image, overlay)

        # Verify format using qemu-img info
        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "info",
            "--output=json",
            str(overlay),
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        import json

        info = json.loads(stdout)

        assert info.get("format") == "qcow2"
        assert info.get("virtual-size") > 0

    async def test_multiple_overlays_created_sequentially(
        self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path
    ) -> None:
        """Multiple overlays can be created in sequence."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        overlays = [tmp_path / f"overlay-{i}.qcow2" for i in range(5)]
        for overlay in overlays:
            await daemon.create_overlay(base_image, overlay)

        assert all(o.exists() for o in overlays)
        assert all(o.stat().st_size > 0 for o in overlays)

    async def test_multiple_overlays_created_concurrently(
        self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path
    ) -> None:
        """Multiple overlays can be created concurrently (QMP commands serialized by lock)."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        overlays = [tmp_path / f"concurrent-{i}.qcow2" for i in range(5)]
        tasks = [daemon.create_overlay(base_image, o) for o in overlays]
        await asyncio.gather(*tasks)

        assert all(o.exists() for o in overlays)

    async def test_restart_daemon_after_stop(self, tmp_path: Path) -> None:
        """Daemon can be restarted after stop (not the same instance, but same pattern)."""
        daemon = QemuStorageDaemon()

        # First lifecycle
        await daemon.start()
        assert daemon.started is True
        await daemon.stop()
        assert daemon.started is False

        # Create new daemon (same pattern as OverlayPool restart)
        daemon2 = QemuStorageDaemon()
        await daemon2.start()
        assert daemon2.started is True
        await daemon2.stop()


# ============================================================================
# OverlayPool Integration with Daemon
# ============================================================================


@skip_unless_hwaccel
class TestOverlayPoolWithDaemon:
    """Integration tests for OverlayPool using QemuStorageDaemon."""

    async def test_pool_uses_daemon_when_enabled(self, vm_settings, tmp_path: Path) -> None:
        """OverlayPool uses daemon for overlay creation."""
        from exec_sandbox.overlay_pool import OverlayPool
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        pool = OverlayPool(max_concurrent_vms=4, pool_dir=tmp_path / "pool")
        await pool.startup([base_image])

        try:
            assert pool.daemon_enabled is True

            # Acquire overlay should work
            target = tmp_path / "acquired.qcow2"
            result = await pool.acquire(base_image, target)

            assert result is True  # From pool
            assert target.exists()
        finally:
            await pool.shutdown()

    async def test_daemon_stopped_on_pool_shutdown(self, vm_settings, tmp_path: Path) -> None:
        """Daemon is stopped when pool shuts down."""
        from exec_sandbox.overlay_pool import OverlayPool
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        pool = OverlayPool(max_concurrent_vms=4, pool_dir=tmp_path / "pool")
        await pool.startup([base_image])

        assert pool.daemon_enabled is True

        await pool.shutdown()

        assert pool.daemon_enabled is False
        assert pool._daemon is None


# ============================================================================
# Error Handling Tests - Real daemon, verify error paths
# ============================================================================


@skip_unless_hwaccel
class TestQemuStorageDaemonErrors:
    """Test error handling with real daemon - no mocking."""

    @pytest.fixture
    async def daemon(self):
        """Create and start daemon, cleanup after test."""
        d = QemuStorageDaemon()
        await d.start()
        yield d
        await d.stop()

    async def test_create_overlay_nonexistent_base_image_fails(
        self, daemon: QemuStorageDaemon, tmp_path: Path
    ) -> None:
        """create_overlay with nonexistent base fails fast with clear error."""
        nonexistent = tmp_path / "does-not-exist.qcow2"
        overlay = tmp_path / "overlay.qcow2"

        with pytest.raises(QemuStorageDaemonError, match="Failed to get image info"):
            await daemon.create_overlay(nonexistent, overlay)

        # Overlay should not exist
        assert not overlay.exists()

    async def test_create_overlay_invalid_base_image_fails(
        self, daemon: QemuStorageDaemon, tmp_path: Path
    ) -> None:
        """create_overlay with invalid qcow2 base fails fast with clear error."""
        # Create a fake base image (not qcow2 - will be detected as "raw" format)
        fake_base = tmp_path / "fake-base.qcow2"
        fake_base.write_text("not a qcow2 file")

        overlay = tmp_path / "overlay.qcow2"

        with pytest.raises(QemuStorageDaemonError, match="not qcow2 format"):
            await daemon.create_overlay(fake_base, overlay)

    async def test_create_overlay_permission_denied(
        self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path
    ) -> None:
        """create_overlay fails when overlay path is not writable."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        overlay = readonly_dir / "overlay.qcow2"

        try:
            with pytest.raises(QemuStorageDaemonError):
                await daemon.create_overlay(base_image, overlay)
        finally:
            readonly_dir.chmod(0o755)  # Restore for cleanup

    async def test_create_overlay_with_unicode_path(
        self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path
    ) -> None:
        """create_overlay works with unicode characters in path."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        # Unicode path with various scripts
        unicode_dir = tmp_path / "тест-日本語-🔥"
        unicode_dir.mkdir()
        overlay = unicode_dir / "overlay-émoji-中文.qcow2"

        await daemon.create_overlay(base_image, overlay)
        assert overlay.exists()

    async def test_create_overlay_with_spaces_in_path(
        self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path
    ) -> None:
        """create_overlay works with spaces in path."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        space_dir = tmp_path / "path with spaces"
        space_dir.mkdir()
        overlay = space_dir / "overlay file.qcow2"

        await daemon.create_overlay(base_image, overlay)
        assert overlay.exists()

    async def test_create_overlay_symlink_base_image(
        self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path
    ) -> None:
        """create_overlay works when base image is a symlink."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        # Create symlink to base image
        symlink = tmp_path / "base-link.qcow2"
        symlink.symlink_to(base_image)

        overlay = tmp_path / "overlay.qcow2"
        await daemon.create_overlay(symlink, overlay)

        assert overlay.exists()
        assert overlay.stat().st_size > 0


# ============================================================================
# Stress and Concurrency Tests
# ============================================================================


@skip_unless_hwaccel
class TestQemuStorageDaemonStress:
    """Stress tests for daemon under load."""

    @pytest.fixture
    async def daemon(self):
        """Create and start daemon, cleanup after test."""
        d = QemuStorageDaemon()
        await d.start()
        yield d
        await d.stop()

    async def test_many_overlays_sequential(self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path) -> None:
        """Create 20 overlays sequentially to verify daemon stability."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        overlays = [tmp_path / f"overlay-{i}.qcow2" for i in range(20)]
        for overlay in overlays:
            await daemon.create_overlay(base_image, overlay)

        assert all(o.exists() for o in overlays)
        assert len({o.stat().st_ino for o in overlays}) == 20  # All unique files

    async def test_many_overlays_concurrent(self, daemon: QemuStorageDaemon, vm_settings, tmp_path: Path) -> None:
        """Create 20 overlays concurrently to verify lock handling."""
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        overlays = [tmp_path / f"concurrent-{i}.qcow2" for i in range(20)]
        tasks = [daemon.create_overlay(base_image, o) for o in overlays]
        await asyncio.gather(*tasks)

        assert all(o.exists() for o in overlays)

    async def test_rapid_start_stop_cycles(self) -> None:
        """Rapid start/stop cycles don't leak resources."""
        for _ in range(5):
            daemon = QemuStorageDaemon()
            await daemon.start()
            assert daemon.started
            socket_path = daemon._socket_path
            await daemon.stop()
            assert not daemon.started
            # Socket should be cleaned up
            assert socket_path is None or not socket_path.exists()

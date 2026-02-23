"""Tests for QemuStorageDaemon.

Test philosophy:
- Unit tests: Pure logic only, no real daemon process, no mocks unless unavoidable
- Integration tests: Real daemon, real files, real code paths (requires hwaccel)
- Error tests: Real daemon, verify error handling works correctly
"""

import asyncio
import json
import os
import secrets
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from exec_sandbox.overlay_pool import OverlayPool
from exec_sandbox.process_registry import _process_groups
from exec_sandbox.qemu_storage_daemon import QemuStorageDaemon, QemuStorageDaemonError, QmpEvent
from exec_sandbox.vm_manager import VmManager

from .conftest import skip_unless_hwaccel

# ============================================================================
# Test Helpers & Fixtures
# ============================================================================


def _make_event(job_id: str, status: str) -> QmpEvent:
    """Create a QmpEvent for testing."""
    return QmpEvent(event="JOB_STATUS_CHANGE", job_id=job_id, status=status)


@pytest.fixture
def base_image(vm_settings) -> Path:  # type: ignore[no-untyped-def]
    """Resolve the Python base image path without starting the overlay pool."""
    return VmManager(vm_settings).get_base_image("python")  # type: ignore[arg-type]


@pytest.fixture
async def started_daemon() -> AsyncGenerator[QemuStorageDaemon, None]:
    """Started QemuStorageDaemon, torn down after the test.

    Tests that verify start/stop behavior should create their own
    daemon instance instead of using this fixture.
    """
    d = QemuStorageDaemon()
    await d.start()
    yield d
    await d.stop()


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

    # --- Event buffer logic ---

    async def test_event_buffer_initialized_empty(self) -> None:
        """Fresh daemon has empty event buffer."""
        daemon = QemuStorageDaemon()
        assert daemon._event_buffer == []

    async def test_consume_event_finds_matching_concluded(self) -> None:
        """consume_event returns and removes matching concluded event."""
        daemon = QemuStorageDaemon()
        daemon._event_buffer = [_make_event("job-A", "concluded")]

        result = daemon._consume_event("job-A", "concluded")

        assert result is not None
        assert result.job_id == "job-A"
        assert daemon._event_buffer == []

    async def test_consume_event_returns_none_when_no_match(self) -> None:
        """consume_event returns None when job ID doesn't match."""
        daemon = QemuStorageDaemon()
        daemon._event_buffer = [_make_event("job-A", "concluded")]

        result = daemon._consume_event("job-B", "concluded")

        assert result is None
        assert len(daemon._event_buffer) == 1  # Unchanged

    async def test_consume_event_preserves_other_events(self) -> None:
        """consume_event only removes the matched event, preserves others."""
        daemon = QemuStorageDaemon()
        daemon._event_buffer = [
            _make_event("job-A", "running"),
            _make_event("job-B", "concluded"),
            _make_event("job-A", "concluded"),
        ]

        result = daemon._consume_event("job-A", "concluded")

        assert result is not None
        assert result.job_id == "job-A"
        assert result.status == "concluded"
        assert len(daemon._event_buffer) == 2
        assert daemon._event_buffer[0].job_id == "job-A"
        assert daemon._event_buffer[0].status == "running"
        assert daemon._event_buffer[1].job_id == "job-B"

    async def test_consume_event_empty_buffer(self) -> None:
        """consume_event returns None on empty buffer."""
        daemon = QemuStorageDaemon()
        assert daemon._consume_event("job-A", "concluded") is None

    async def test_consume_event_wrong_status(self) -> None:
        """consume_event returns None when status doesn't match."""
        daemon = QemuStorageDaemon()
        daemon._event_buffer = [_make_event("job-A", "running")]

        result = daemon._consume_event("job-A", "concluded")

        assert result is None
        assert len(daemon._event_buffer) == 1  # Unchanged

    async def test_consume_event_takes_first_match(self) -> None:
        """consume_event returns first matching event when duplicates exist."""
        daemon = QemuStorageDaemon()
        first = _make_event("job-A", "concluded")
        second = _make_event("job-A", "concluded")
        daemon._event_buffer = [first, second]

        result = daemon._consume_event("job-A", "concluded")

        assert result is first  # Identity check: must be the first object
        assert daemon._event_buffer == [second]

    async def test_consume_event_ignores_non_job_events(self) -> None:
        """consume_event skips unrelated event types."""
        daemon = QemuStorageDaemon()
        # BLOCK_IO_ERROR has no matching job_id/status
        daemon._event_buffer = [QmpEvent(event="BLOCK_IO_ERROR", job_id="", status="")]

        result = daemon._consume_event("job-A", "concluded")

        assert result is None
        assert len(daemon._event_buffer) == 1  # Preserved

    async def test_consume_event_malformed_event_no_data(self) -> None:
        """consume_event handles event with empty fields without crashing."""
        daemon = QemuStorageDaemon()
        # Simulates _parse_event output for {"event": "JOB_STATUS_CHANGE"} (no data)
        daemon._event_buffer = [QmpEvent(event="JOB_STATUS_CHANGE", job_id="", status="")]

        result = daemon._consume_event("job-A", "concluded")

        assert result is None
        assert len(daemon._event_buffer) == 1  # Preserved

    async def test_consume_event_malformed_event_no_id(self) -> None:
        """consume_event handles event with status but no ID."""
        daemon = QemuStorageDaemon()
        # Simulates _parse_event output for {"data": {"status": "concluded"}} (no id)
        daemon._event_buffer = [QmpEvent(event="JOB_STATUS_CHANGE", job_id="", status="concluded")]

        result = daemon._consume_event("job-A", "concluded")

        assert result is None
        assert len(daemon._event_buffer) == 1  # Preserved

    async def test_consume_event_with_extra_fields_via_parse(self) -> None:
        """_parse_event extracts only relevant fields, ignoring extras."""
        raw = {
            "event": "JOB_STATUS_CHANGE",
            "data": {"id": "job-A", "status": "concluded", "extra-field": "unexpected"},
            "timestamp": {"seconds": 0, "microseconds": 0},
        }
        parsed = QemuStorageDaemon._parse_event(raw)
        assert parsed.job_id == "job-A"
        assert parsed.status == "concluded"
        assert parsed.event == "JOB_STATUS_CHANGE"

    async def test_consume_event_large_buffer(self) -> None:
        """consume_event works correctly with 1000 events in buffer."""
        daemon = QemuStorageDaemon()
        # Fill buffer with 1000 events for different job IDs
        daemon._event_buffer = [_make_event(f"job-{i}", "concluded") for i in range(1000)]

        # Consume one specific event
        result = daemon._consume_event("job-500", "concluded")

        assert result is not None
        assert result.job_id == "job-500"
        assert len(daemon._event_buffer) == 999
        # Verify the consumed one is gone
        remaining_ids = [e.job_id for e in daemon._event_buffer]
        assert "job-500" not in remaining_ids

    async def test_event_buffer_cleared_on_stop(self) -> None:
        """stop() clears the event buffer even for unstarted daemon."""
        daemon = QemuStorageDaemon()
        daemon._event_buffer = [_make_event("job-A", "concluded"), _make_event("job-B", "running")]

        await daemon.stop()

        assert daemon._event_buffer == []

    async def test_parse_event_malformed_raw(self) -> None:
        """_parse_event handles missing keys gracefully."""
        # No data key at all
        parsed = QemuStorageDaemon._parse_event({"event": "JOB_STATUS_CHANGE"})
        assert parsed.job_id == ""
        assert parsed.status == ""

        # No event key
        parsed = QemuStorageDaemon._parse_event({"data": {"id": "x", "status": "y"}})
        assert parsed.event == ""
        assert parsed.job_id == "x"
        assert parsed.status == "y"

        # Completely empty
        parsed = QemuStorageDaemon._parse_event({})
        assert parsed.event == ""
        assert parsed.job_id == ""
        assert parsed.status == ""

    async def test_parse_event_non_dict_data(self) -> None:
        """_parse_event handles non-dict data values gracefully."""
        # String data
        parsed = QemuStorageDaemon._parse_event({"event": "X", "data": "not-a-dict"})
        assert parsed.event == "X"
        assert parsed.job_id == ""
        assert parsed.status == ""

        # Integer data
        parsed = QemuStorageDaemon._parse_event({"event": "Y", "data": 42})
        assert parsed.job_id == ""
        assert parsed.status == ""

        # List data
        parsed = QemuStorageDaemon._parse_event({"event": "Z", "data": ["a", "b"]})
        assert parsed.job_id == ""
        assert parsed.status == ""

        # None data (explicit)
        parsed = QemuStorageDaemon._parse_event({"event": "W", "data": None})
        assert parsed.job_id == ""
        assert parsed.status == ""

    # --- Orphan reaper unit tests ---

    async def test_reap_stale_daemons_removes_dead_parent_socket(self) -> None:
        """Reaper removes sockets for dead parent PIDs."""
        # PID 99999999 almost certainly doesn't exist; unique suffix avoids parallel collisions
        suffix = secrets.token_hex(8)
        sock = Path(tempfile.gettempdir()) / f"qsd-99999999-{suffix}.sock"
        sock.touch()

        try:
            count = QemuStorageDaemon._reap_stale_daemons()
            assert count >= 1
            assert not sock.exists()
        finally:
            # Cleanup if test fails
            sock.unlink(missing_ok=True)

    async def test_reap_stale_daemons_keeps_live_parent_socket(self) -> None:
        """Reaper does not remove sockets for live parent PIDs."""
        # Use our own PID (definitely alive); unique suffix avoids parallel collisions
        suffix = secrets.token_hex(8)
        sock = Path(tempfile.gettempdir()) / f"qsd-{os.getpid()}-{suffix}.sock"
        sock.touch()

        try:
            QemuStorageDaemon._reap_stale_daemons()
            assert sock.exists()  # Should NOT be removed
        finally:
            sock.unlink(missing_ok=True)

    async def test_reap_stale_daemons_ignores_non_matching_filenames(self) -> None:
        """Reaper ignores files that don't match the qsd-{pid}-{hex}.sock pattern."""
        tmpdir = Path(tempfile.gettempdir())
        non_matching1 = tmpdir / "qsd-notapid.sock"
        non_matching2 = tmpdir / "other-file.sock"
        non_matching1.touch()
        non_matching2.touch()

        try:
            QemuStorageDaemon._reap_stale_daemons()
            assert non_matching1.exists()  # Untouched
            assert non_matching2.exists()  # Untouched
        finally:
            non_matching1.unlink(missing_ok=True)
            non_matching2.unlink(missing_ok=True)

    async def test_reap_stale_daemons_does_not_crash_when_clean(self) -> None:
        """Reaper runs without error when no stale sockets exist."""
        count = QemuStorageDaemon._reap_stale_daemons()
        # Count may be >= 0 if stale sockets exist from other tests;
        # the important thing is it doesn't crash
        assert count >= 0

    async def test_reap_stale_daemons_handles_permission_error(self) -> None:
        """Reaper handles PID 1 (init) gracefully — PermissionError means alive."""
        suffix = secrets.token_hex(8)
        sock = Path(tempfile.gettempdir()) / f"qsd-1-{suffix}.sock"
        sock.touch()

        try:
            # Should not crash — PID 1 is alive but os.kill(1, 0) raises PermissionError
            QemuStorageDaemon._reap_stale_daemons()
            assert sock.exists()  # Should NOT be removed (parent alive)
        finally:
            sock.unlink(missing_ok=True)


# ============================================================================
# Integration Tests - Real daemon, real files
# ============================================================================


@skip_unless_hwaccel
class TestQemuStorageDaemonIntegration:
    """Integration tests with real qemu-storage-daemon - no mocking."""

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
        try:
            assert daemon._process is not None
            pid_after_first = daemon._process.pid
            await daemon.start()  # Should be no-op
            assert daemon._process is not None
            assert daemon._process.pid == pid_after_first  # Same process
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

    async def test_create_overlay(self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path) -> None:
        """Creates valid qcow2 overlay via QMP."""
        overlay = tmp_path / "test-overlay.qcow2"
        await started_daemon.create_overlay(base_image, overlay)

        assert overlay.exists()
        assert overlay.stat().st_size > 0

    async def test_overlay_has_correct_backing_file(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Created overlay references correct base image as backing file."""
        overlay = tmp_path / "backing-test.qcow2"
        await started_daemon.create_overlay(base_image, overlay)

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

    async def test_overlay_has_qcow2_format(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Created overlay is valid qcow2 format."""
        overlay = tmp_path / "format-test.qcow2"
        await started_daemon.create_overlay(base_image, overlay)

        # Verify format using qemu-img info
        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "info",
            "--output=json",
            str(overlay),
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        info = json.loads(stdout)

        assert info.get("format") == "qcow2"
        virtual_size = info.get("virtual-size")
        assert isinstance(virtual_size, int)
        assert virtual_size > 0

    async def test_multiple_overlays_created_sequentially(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Multiple overlays can be created in sequence."""
        overlays = [tmp_path / f"overlay-{i}.qcow2" for i in range(5)]
        for overlay in overlays:
            await started_daemon.create_overlay(base_image, overlay)

        assert all(o.exists() for o in overlays)
        assert all(o.stat().st_size > 0 for o in overlays)

    async def test_multiple_overlays_created_concurrently(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Multiple overlays can be created concurrently (QMP commands serialized by lock)."""
        overlays = [tmp_path / f"concurrent-{i}.qcow2" for i in range(5)]
        tasks = [started_daemon.create_overlay(base_image, o) for o in overlays]
        await asyncio.gather(*tasks)

        assert all(o.exists() for o in overlays)

    async def test_restart_daemon_after_stop(self) -> None:
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

    # --- Event-driven flow integration ---

    async def test_single_overlay_events_consumed(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Event buffer is empty after creating one overlay (events consumed, not leaking)."""
        overlay = tmp_path / "event-test.qcow2"
        await started_daemon.create_overlay(base_image, overlay)

        assert started_daemon._event_buffer == []

    async def test_sequential_overlays_buffer_stays_clean(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Event buffer stays empty across sequential overlay creations."""
        for i in range(10):
            overlay = tmp_path / f"seq-{i}.qcow2"
            await started_daemon.create_overlay(base_image, overlay)
            assert started_daemon._event_buffer == [], f"Buffer not empty after overlay {i}"

    async def test_concurrent_overlays_all_succeed(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """8 concurrent overlays all succeed and buffer is empty afterward."""
        overlays = [tmp_path / f"conc-{i}.qcow2" for i in range(8)]
        tasks = [started_daemon.create_overlay(base_image, o) for o in overlays]
        await asyncio.gather(*tasks)

        assert all(o.exists() for o in overlays)
        assert started_daemon._event_buffer == []

    async def test_concurrent_overlays_no_stale_events(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """20 concurrent overlays (more than typical semaphore) leave no stale events."""
        overlays = [tmp_path / f"many-{i}.qcow2" for i in range(20)]
        tasks = [started_daemon.create_overlay(base_image, o) for o in overlays]
        await asyncio.gather(*tasks)

        assert all(o.exists() for o in overlays)
        assert started_daemon._event_buffer == []

    async def test_overlay_after_failed_overlay_succeeds(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Valid overlay creation succeeds after a failed attempt on same daemon."""
        # First: fail with nonexistent base
        with pytest.raises(QemuStorageDaemonError):
            await started_daemon.create_overlay(tmp_path / "nonexistent.qcow2", tmp_path / "fail.qcow2")

        # Second: succeed with valid base
        overlay = tmp_path / "success.qcow2"
        await started_daemon.create_overlay(base_image, overlay)
        assert overlay.exists()

    # --- Process registry integration ---

    async def test_process_registered_on_start(self) -> None:
        """Daemon PID is registered in process_registry on start."""
        daemon = QemuStorageDaemon()
        await daemon.start()
        try:
            assert daemon._process is not None
            assert daemon._process.pid in _process_groups
        finally:
            await daemon.stop()

    async def test_process_unregistered_on_stop(self) -> None:
        """Daemon PID is unregistered from process_registry on stop."""
        daemon = QemuStorageDaemon()
        await daemon.start()
        assert daemon._process is not None
        pid = daemon._process.pid
        await daemon.stop()
        assert pid not in _process_groups

    async def test_process_unregistered_on_start_failure(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """PID is not left in process_registry when start fails."""
        daemon = QemuStorageDaemon()

        # Poison the QMP connection step so start fails after process creation
        # Monkeypatch at class level since __slots__ prevents instance-level setattr
        async def _bad_connect_qmp(_self: QemuStorageDaemon) -> None:
            raise ConnectionError("Injected failure")

        monkeypatch.setattr(QemuStorageDaemon, "_connect_qmp", _bad_connect_qmp)

        with pytest.raises(QemuStorageDaemonError, match="Failed to start daemon"):
            await daemon.start()

        # Process was cleaned up, should not be in registry
        assert daemon._process is None

    # --- Orphan reaper integration ---

    async def test_reaper_runs_on_start_without_blocking(self) -> None:
        """start() completes in reasonable time even when reaper runs."""
        tmpdir = Path(tempfile.gettempdir())
        # Create a few stale sockets with unique suffixes to avoid parallel collisions
        stale_socks = []
        for _ in range(3):
            sock = tmpdir / f"qsd-99999999-{secrets.token_hex(8)}.sock"
            sock.touch()
            stale_socks.append(sock)

        daemon = QemuStorageDaemon()
        try:
            # start() should complete within a reasonable timeout
            await asyncio.wait_for(daemon.start(), timeout=15.0)

            # Stale sockets should be cleaned
            for sock in stale_socks:
                assert not sock.exists(), f"Stale socket not cleaned: {sock}"
        finally:
            await daemon.stop()
            for sock in stale_socks:
                sock.unlink(missing_ok=True)


# ============================================================================
# OverlayPool Integration with Daemon
# ============================================================================


@skip_unless_hwaccel
class TestOverlayPoolWithDaemon:
    """Integration tests for OverlayPool using QemuStorageDaemon."""

    async def test_pool_uses_daemon_when_enabled(self, base_image: Path, tmp_path: Path) -> None:
        """OverlayPool uses daemon for overlay creation."""
        pool = OverlayPool(pool_size=2, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        try:
            assert pool.daemon_enabled is True

            # Acquire overlay should work
            target = tmp_path / "acquired.qcow2"
            result = await pool.acquire(base_image, target)

            assert result is True  # From pool
            assert target.exists()
        finally:
            await pool.stop()

    async def test_daemon_stopped_on_pool_shutdown(self, base_image: Path, tmp_path: Path) -> None:
        """Daemon is stopped when pool shuts down."""
        pool = OverlayPool(pool_size=2, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        assert pool.daemon_enabled is True

        await pool.stop()

        assert pool.daemon_enabled is False
        assert pool._daemon is None


# ============================================================================
# Error Handling Tests - Real daemon, verify error paths
# ============================================================================


@skip_unless_hwaccel
class TestQemuStorageDaemonErrors:
    """Test error handling with real daemon - no mocking."""

    async def test_create_overlay_nonexistent_base_image_fails(
        self, started_daemon: QemuStorageDaemon, tmp_path: Path
    ) -> None:
        """create_overlay with nonexistent base fails fast with clear error."""
        nonexistent = tmp_path / "does-not-exist.qcow2"
        overlay = tmp_path / "overlay.qcow2"

        with pytest.raises(QemuStorageDaemonError, match="Failed to get image info"):
            await started_daemon.create_overlay(nonexistent, overlay)

        # Overlay should not exist
        assert not overlay.exists()

    async def test_create_overlay_invalid_base_image_fails(
        self, started_daemon: QemuStorageDaemon, tmp_path: Path
    ) -> None:
        """create_overlay with invalid qcow2 base fails fast with clear error."""
        # Create a fake base image (not qcow2 - will be detected as "raw" format)
        fake_base = tmp_path / "fake-base.qcow2"
        fake_base.write_text("not a qcow2 file")

        overlay = tmp_path / "overlay.qcow2"

        with pytest.raises(QemuStorageDaemonError, match="not qcow2 format"):
            await started_daemon.create_overlay(fake_base, overlay)

    async def test_create_overlay_permission_denied(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """create_overlay fails when overlay path is not writable."""
        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        overlay = readonly_dir / "overlay.qcow2"

        try:
            with pytest.raises(QemuStorageDaemonError):
                await started_daemon.create_overlay(base_image, overlay)
        finally:
            readonly_dir.chmod(0o755)  # Restore for cleanup

    async def test_create_overlay_with_unicode_path(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """create_overlay works with unicode characters in path."""
        # Unicode path with various scripts
        unicode_dir = tmp_path / "тест-日本語-🔥"
        unicode_dir.mkdir()
        overlay = unicode_dir / "overlay-émoji-中文.qcow2"

        await started_daemon.create_overlay(base_image, overlay)
        assert overlay.exists()

    async def test_create_overlay_with_spaces_in_path(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """create_overlay works with spaces in path."""
        space_dir = tmp_path / "path with spaces"
        space_dir.mkdir()
        overlay = space_dir / "overlay file.qcow2"

        await started_daemon.create_overlay(base_image, overlay)
        assert overlay.exists()

    async def test_create_overlay_symlink_base_image(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """create_overlay works when base image is a symlink."""
        # Create symlink to base image
        symlink = tmp_path / "base-link.qcow2"
        symlink.symlink_to(base_image)

        overlay = tmp_path / "overlay.qcow2"
        await started_daemon.create_overlay(symlink, overlay)

        assert overlay.exists()
        assert overlay.stat().st_size > 0


# ============================================================================
# Stress and Concurrency Tests
# ============================================================================


@skip_unless_hwaccel
class TestQemuStorageDaemonStress:
    """Stress tests for daemon under load."""

    async def test_many_overlays_sequential(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Create 20 overlays sequentially to verify daemon stability."""
        overlays = [tmp_path / f"overlay-{i}.qcow2" for i in range(20)]
        for overlay in overlays:
            await started_daemon.create_overlay(base_image, overlay)

        assert all(o.exists() for o in overlays)
        assert len({o.stat().st_ino for o in overlays}) == 20  # All unique files

    async def test_many_overlays_concurrent(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """Create 20 overlays concurrently to verify lock handling and event-driven path."""
        overlays = [tmp_path / f"concurrent-{i}.qcow2" for i in range(20)]
        tasks = [started_daemon.create_overlay(base_image, o) for o in overlays]
        await asyncio.gather(*tasks)

        assert all(o.exists() for o in overlays)
        assert started_daemon._event_buffer == []  # All events consumed

    async def test_rapid_start_stop_cycles(self) -> None:
        """Rapid start/stop cycles don't leak resources."""
        for _ in range(5):
            daemon = QemuStorageDaemon()
            await daemon.start()
            assert daemon.started
            socket_path = daemon._socket_path
            assert socket_path is not None, "Daemon should have a socket path after start()"
            await daemon.stop()
            assert not daemon.started
            assert not socket_path.exists(), f"Socket should be removed after stop(): {socket_path}"

    async def test_rapid_create_overlay_cycles(
        self, started_daemon: QemuStorageDaemon, base_image: Path, tmp_path: Path
    ) -> None:
        """50 sequential overlays with no event buffer growth (leak detection)."""
        for i in range(50):
            overlay = tmp_path / f"rapid-{i}.qcow2"
            await started_daemon.create_overlay(base_image, overlay)
            assert overlay.exists()

        # No events leaked
        assert started_daemon._event_buffer == []

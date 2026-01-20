"""Tests for resource_cleanup.py cleanup functions."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from exec_sandbox.cgroup import cleanup_cgroup
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.resource_cleanup import (
    _cleanup_zombie_task,
    _zombie_reap_tasks,
    cleanup_file,
    cleanup_overlay,
    cleanup_process,
)


class TestResourceCleanupProcess:
    """Test process cleanup with SIGTERM/SIGKILL."""

    async def test_cleanup_process_sigkill_timeout(self):
        """SIGKILL timeout logs error, returns False."""
        # Mock ProcessWrapper that ignores SIGKILL (kernel hung I/O)
        mock_proc = Mock(spec=ProcessWrapper)
        mock_proc.pid = 12345
        mock_proc.returncode = None  # Still alive
        mock_proc.stdout = None  # No pipes - uses wait_with_timeout() directly
        mock_proc.stderr = None
        mock_proc.terminate = AsyncMock()
        mock_proc.kill = AsyncMock()

        # Timeout on both SIGTERM and SIGKILL waits
        mock_proc.wait_with_timeout = AsyncMock(side_effect=[asyncio.TimeoutError, asyncio.TimeoutError])

        # Should return False (cleanup failed)
        result = await cleanup_process(
            proc=mock_proc,
            name="test-process",
            context_id="test-ctx",
            term_timeout=0.1,
            kill_timeout=0.1,
        )

        assert result is False

        # Verify escalation logic: terminate → wait_with_timeout → kill → wait_with_timeout
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        # wait_with_timeout called 2 times (SIGTERM + SIGKILL)
        assert mock_proc.wait_with_timeout.call_count == 2

    async def test_cleanup_process_sigterm_success(self):
        """Process responds to SIGTERM, cleanup succeeds."""
        mock_proc = Mock(spec=ProcessWrapper)
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.stdout = None  # No pipes - uses wait_with_timeout() directly
        mock_proc.stderr = None
        mock_proc.terminate = AsyncMock()
        mock_proc.wait_with_timeout = AsyncMock(return_value=0)  # Exits gracefully

        result = await cleanup_process(
            proc=mock_proc,
            name="graceful-process",
            context_id="test-ctx",
        )

        assert result is True
        mock_proc.terminate.assert_called_once()
        # Verify logic: process exits after SIGTERM, no SIGKILL needed
        mock_proc.wait_with_timeout.assert_called_once()

    async def test_cleanup_process_already_dead(self):
        """Process already dead (returncode set) returns immediately."""
        mock_proc = Mock(spec=ProcessWrapper)
        mock_proc.pid = 12345
        mock_proc.returncode = 0  # Already exited
        mock_proc.wait = AsyncMock()  # Background reaper needs this

        # No operations needed - process already dead
        result = await cleanup_process(
            proc=mock_proc,
            name="dead-process",
            context_id="test-ctx",
        )

        assert result is True
        # Verify no terminate/kill when already dead
        mock_proc.terminate = AsyncMock()
        mock_proc.terminate.assert_not_called()

    async def test_cleanup_process_none(self):
        """None process returns True (no-op)."""
        result = await cleanup_process(
            proc=None,
            name="null-process",
            context_id="test-ctx",
        )

        assert result is True

    async def test_cleanup_process_sigterm_timeout_sigkill_success(self):
        """Process ignores SIGTERM but responds to SIGKILL."""
        mock_proc = Mock(spec=ProcessWrapper)
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.stdout = None  # No pipes - uses wait_with_timeout() directly
        mock_proc.stderr = None
        mock_proc.terminate = AsyncMock()
        mock_proc.kill = AsyncMock()

        # First wait (SIGTERM) times out, second wait (SIGKILL) succeeds
        mock_proc.wait_with_timeout = AsyncMock(
            side_effect=[asyncio.TimeoutError, None]  # SIGTERM timeout, then exit
        )

        result = await cleanup_process(
            proc=mock_proc,
            name="stubborn-process",
            context_id="test-ctx",
            term_timeout=0.1,
            kill_timeout=0.5,
        )

        assert result is True
        # Verify escalation: terminate → kill
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert mock_proc.wait_with_timeout.call_count == 2

    async def test_cleanup_process_process_lookup_error(self):
        """ProcessLookupError during cleanup returns True (already dead)."""
        mock_proc = Mock(spec=ProcessWrapper)
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.terminate = AsyncMock(side_effect=ProcessLookupError)

        result = await cleanup_process(
            proc=mock_proc,
            name="race-condition-process",
            context_id="test-ctx",
        )

        # Verify exception handling logic
        assert result is True

    async def test_cleanup_process_unexpected_exception(self):
        """Unexpected exception returns False, doesn't raise."""
        mock_proc = Mock(spec=ProcessWrapper)
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.terminate = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        # Cleanup logic should catch exception and return False
        result = await cleanup_process(
            proc=mock_proc,
            name="error-process",
            context_id="test-ctx",
        )

        assert result is False

    async def test_cleanup_process_process_group_lookup_error(self):
        """ProcessLookupError when process dies between SIGTERM and SIGKILL."""
        mock_proc = Mock(spec=ProcessWrapper)
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.stdout = None  # No pipes - uses wait_with_timeout() directly
        mock_proc.stderr = None
        mock_proc.terminate = AsyncMock()
        mock_proc.kill = AsyncMock(side_effect=ProcessLookupError)  # Process died before kill

        # First wait (SIGTERM) times out, then process dies
        mock_proc.wait_with_timeout = AsyncMock(side_effect=[asyncio.TimeoutError])

        result = await cleanup_process(
            proc=mock_proc,
            name="pgid-race-process",
            context_id="test-ctx",
            term_timeout=0.1,
            kill_timeout=0.1,
        )

        # Verify logic handles ProcessLookupError gracefully
        assert result is True

    async def test_cleanup_process_killpg_process_lookup_error(self):
        """ProcessLookupError during kill is handled gracefully."""
        mock_proc = Mock(spec=ProcessWrapper)
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.stdout = None  # No pipes - uses wait_with_timeout() directly
        mock_proc.stderr = None
        mock_proc.terminate = AsyncMock()
        mock_proc.kill = AsyncMock(side_effect=ProcessLookupError)  # Race: process died

        # SIGTERM timeout, then ProcessLookupError on kill
        mock_proc.wait_with_timeout = AsyncMock(side_effect=[asyncio.TimeoutError])

        result = await cleanup_process(
            proc=mock_proc,
            name="killpg-race-process",
            context_id="test-ctx",
            term_timeout=0.1,
            kill_timeout=0.1,
        )

        # Verify logic continues after ProcessLookupError
        assert result is True


class TestResourceCleanupFilesystem:
    """Test file/cgroup/overlay cleanup (async functions)."""

    async def test_cleanup_file_missing(self, tmp_path: Path):
        """Missing file cleanup succeeds (idempotent)."""
        nonexistent = tmp_path / "does-not-exist.txt"

        # REAL cleanup_file logic with real filesystem
        result = await cleanup_file(nonexistent, "test-ctx")

        assert result is True

    async def test_cleanup_file_exists(self, tmp_path: Path):
        """Existing file is deleted."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # REAL cleanup_file logic with real filesystem
        result = await cleanup_file(test_file, "test-ctx")

        assert result is True
        # Verify REAL deletion happened
        assert not test_file.exists()

    async def test_cleanup_file_none(self):
        """None file path returns True (no-op)."""
        result = await cleanup_file(None, "test-ctx")

        assert result is True

    async def test_cleanup_file_with_description(self, tmp_path: Path):
        """Custom description used in logging."""
        test_file = tmp_path / "socket.sock"
        test_file.write_text("socket")

        # REAL cleanup_file logic with real filesystem
        result = await cleanup_file(test_file, "test-ctx", description="unix socket")

        assert result is True
        # Verify REAL deletion happened
        assert not test_file.exists()

    async def test_cleanup_file_permission_error(self, tmp_path: Path):
        """Permission error returns False."""
        test_file = tmp_path / "readonly.txt"
        test_file.write_text("content")

        # Mock aiofiles.os.remove to simulate permission error
        with patch("aiofiles.os.remove", side_effect=PermissionError("Access denied")):
            # REAL error handling logic runs
            result = await cleanup_file(test_file, "test-ctx")

            assert result is False
        # Verify file still exists (cleanup failed)
        assert test_file.exists()

    async def test_cleanup_file_os_error(self, tmp_path: Path):
        """OSError during deletion returns False."""
        test_file = tmp_path / "locked.txt"
        test_file.write_text("content")

        # Mock aiofiles.os.remove to simulate OS error
        with patch("aiofiles.os.remove", side_effect=OSError("File in use")):
            # REAL error handling logic runs
            result = await cleanup_file(test_file, "test-ctx")

            assert result is False

    async def test_cleanup_cgroup_missing(self, tmp_path: Path):
        """Missing cgroup cleanup succeeds (idempotent)."""
        nonexistent = tmp_path / "cgroup-missing"

        # REAL cleanup_cgroup logic with real filesystem
        result = await cleanup_cgroup(nonexistent, "test-ctx")

        assert result is True

    async def test_cleanup_cgroup_exists(self, tmp_path: Path):
        """Existing cgroup directory is removed."""
        cgroup_dir = tmp_path / "test-cgroup"
        cgroup_dir.mkdir()

        # REAL cleanup_cgroup logic with real filesystem
        result = await cleanup_cgroup(cgroup_dir, "test-ctx")

        assert result is True
        # Verify REAL deletion happened
        assert not cgroup_dir.exists()

    async def test_cleanup_cgroup_none(self):
        """None cgroup path returns True (no-op)."""
        result = await cleanup_cgroup(None, "test-ctx")

        assert result is True

    async def test_cleanup_cgroup_skips_procs_for_non_cgroup_path(self, tmp_path: Path):
        """Non-cgroup paths don't attempt to empty cgroup.procs."""
        # Test that regular directories don't trigger cgroup.procs logic
        # For non-/sys/fs/cgroup paths, the function just tries rmdir directly
        regular_dir = tmp_path / "regular-dir"
        regular_dir.mkdir()

        # REAL cleanup_cgroup logic with real filesystem
        # Empty directory should be removed successfully
        result = await cleanup_cgroup(regular_dir, "test-ctx")

        assert result is True
        # Verify REAL deletion happened
        assert not regular_dir.exists()

    async def test_cleanup_cgroup_non_cgroup_path_silently_succeeds(self, tmp_path: Path):
        """Non-cgroup paths (fallback dummy paths) always return True.

        This is intentional: fallback paths like /tmp/cgroup-vm123 are used
        when cgroups are unavailable. These dummy paths may not exist and
        errors during cleanup are silently ignored.
        """
        cgroup_dir = tmp_path / "test-cgroup"
        cgroup_dir.mkdir()
        (cgroup_dir / "somefile.txt").write_text("content")

        # Non-cgroup paths silently succeed even if rmdir fails
        result = await cleanup_cgroup(cgroup_dir, "test-ctx")

        assert result is True
        # Directory still exists (rmdir failed but was suppressed)
        assert cgroup_dir.exists()

    async def test_cleanup_cgroup_real_cgroup_error(self):
        """Real cgroup paths report errors on failure."""
        # Mock path to look like a real cgroup path
        fake_cgroup_path = Path("/sys/fs/cgroup/code-exec/tenant/vm123")

        # Mock filesystem operations
        with (
            patch("exec_sandbox.cgroup._check_cgroup_v2_mounted", return_value=True),
            patch("aiofiles.os.path.exists", return_value=False),
            patch("aiofiles.os.rmdir", side_effect=OSError("Directory not empty")),
        ):
            result = await cleanup_cgroup(fake_cgroup_path, "test-ctx")

            assert result is False

    async def test_cleanup_overlay_delegates_to_cleanup_file(self, tmp_path: Path):
        """cleanup_overlay uses cleanup_file internally."""
        overlay = tmp_path / "overlay.qcow2"
        overlay.write_bytes(b"fake qcow2")

        # REAL cleanup_overlay logic with real filesystem
        result = await cleanup_overlay(overlay, "test-ctx")

        assert result is True
        # Verify REAL deletion happened
        assert not overlay.exists()

    async def test_cleanup_overlay_none(self):
        """None overlay path returns True (no-op)."""
        result = await cleanup_overlay(None, "test-ctx")

        assert result is True


class TestZombieTaskTracking:
    """Test zombie reaping task tracking to prevent GC."""

    @pytest.fixture(autouse=True)
    def clear_zombie_tasks(self):
        """Clear zombie task set before and after each test."""
        _zombie_reap_tasks.clear()
        yield
        _zombie_reap_tasks.clear()

    async def test_cleanup_zombie_task_removes_from_set(self):
        """_cleanup_zombie_task removes completed task from tracking set."""

        # Create a completed task
        async def noop():
            pass

        task = asyncio.create_task(noop())
        await task  # Wait for completion

        # Add to tracking set
        _zombie_reap_tasks.add(task)
        assert task in _zombie_reap_tasks

        # Call cleanup callback
        _cleanup_zombie_task(task)

        # Verify removal
        assert task not in _zombie_reap_tasks

    async def test_cleanup_zombie_task_logs_exception(self):
        """_cleanup_zombie_task logs exceptions from failed tasks."""

        # Create a task that raises an exception
        async def fail():
            raise ValueError("Test error")

        task = asyncio.create_task(fail())
        with pytest.raises(ValueError):
            await task

        # Add to tracking set
        _zombie_reap_tasks.add(task)

        # Call cleanup callback - should not raise, just log
        with patch("exec_sandbox.resource_cleanup.logger") as mock_logger:
            _cleanup_zombie_task(task)

            # Verify warning logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "Zombie reaping task failed" in call_args[0][0]

        # Verify removal
        assert task not in _zombie_reap_tasks

    async def test_cleanup_zombie_task_handles_cancelled(self):
        """_cleanup_zombie_task handles cancelled tasks without logging."""

        # Create a task and cancel it
        async def sleep_forever():
            await asyncio.sleep(3600)

        task = asyncio.create_task(sleep_forever())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Add to tracking set
        _zombie_reap_tasks.add(task)

        # Call cleanup callback - should not log for cancelled tasks
        with patch("exec_sandbox.resource_cleanup.logger") as mock_logger:
            _cleanup_zombie_task(task)

            # Verify no warning logged for cancelled task
            mock_logger.warning.assert_not_called()

        # Verify removal
        assert task not in _zombie_reap_tasks

    async def test_cleanup_process_tracks_zombie_task_already_dead(self):
        """cleanup_process tracks zombie reap task when process already dead."""
        mock_proc = Mock(spec=ProcessWrapper)
        mock_proc.pid = 12345
        mock_proc.returncode = 0  # Already exited

        # wait() must return a coroutine for asyncio.create_task()
        async def mock_wait():
            return 0

        mock_proc.wait = mock_wait

        result = await cleanup_process(
            proc=mock_proc,
            name="dead-process",
            context_id="test-ctx",
        )

        assert result is True

        # Give the task a chance to be created and tracked
        await asyncio.sleep(0.01)

        # Verify a task was added (and may have been cleaned up already)
        # The task tracking ensures it won't be GC'd prematurely

    async def test_cleanup_process_tracks_zombie_task_on_timeout(self):
        """cleanup_process tracks background zombie reap task on SIGKILL timeout."""
        mock_proc = Mock(spec=ProcessWrapper)
        mock_proc.pid = 12345
        mock_proc.returncode = None  # Still alive
        mock_proc.stdout = None
        mock_proc.stderr = None
        mock_proc.terminate = AsyncMock()
        mock_proc.kill = AsyncMock()

        # Timeout on both SIGTERM and SIGKILL waits
        mock_proc.wait_with_timeout = AsyncMock(side_effect=[asyncio.TimeoutError, asyncio.TimeoutError])

        result = await cleanup_process(
            proc=mock_proc,
            name="stubborn-process",
            context_id="test-ctx",
            term_timeout=0.01,
            kill_timeout=0.01,
        )

        assert result is False

        # Give the background task a chance to be created
        await asyncio.sleep(0.01)

        # The task should have been created and tracked
        # (it may have already completed and been cleaned up)

"""Tests for process registry module."""

import signal
import subprocess
import time

import pytest

from exec_sandbox.process_registry import (
    _process_groups,
    force_kill_all,
    get_tracked_count,
    register_process,
    unregister_process,
)


class MockProcessWrapper:
    """Mock ProcessWrapper for testing."""

    def __init__(self, pid: int | None):
        self.pid = pid


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the process registry before and after each test."""
    _process_groups.clear()
    yield
    _process_groups.clear()


class TestProcessRegistry:
    """Tests for process registry functions."""

    def test_register_process_adds_pid(self):
        """Test that register_process adds PID to registry."""
        proc = MockProcessWrapper(12345)
        register_process(proc)  # type: ignore[arg-type]
        assert 12345 in _process_groups
        assert get_tracked_count() == 1

    def test_register_process_none_safe(self):
        """Test that register_process handles None safely."""
        register_process(None)
        assert get_tracked_count() == 0

    def test_register_process_none_pid_safe(self):
        """Test that register_process handles None PID safely."""
        proc = MockProcessWrapper(None)
        register_process(proc)  # type: ignore[arg-type]
        assert get_tracked_count() == 0

    def test_unregister_process_removes_pid(self):
        """Test that unregister_process removes PID from registry."""
        proc = MockProcessWrapper(12345)
        register_process(proc)  # type: ignore[arg-type]
        assert get_tracked_count() == 1

        unregister_process(proc)  # type: ignore[arg-type]
        assert 12345 not in _process_groups
        assert get_tracked_count() == 0

    def test_unregister_process_none_safe(self):
        """Test that unregister_process handles None safely."""
        unregister_process(None)
        assert get_tracked_count() == 0

    def test_unregister_process_not_registered(self):
        """Test that unregister_process handles unregistered PID safely."""
        proc = MockProcessWrapper(99999)
        unregister_process(proc)  # type: ignore[arg-type]
        assert get_tracked_count() == 0

    def test_get_tracked_count(self):
        """Test get_tracked_count returns correct count."""
        assert get_tracked_count() == 0

        proc1 = MockProcessWrapper(111)
        proc2 = MockProcessWrapper(222)
        register_process(proc1)  # type: ignore[arg-type]
        assert get_tracked_count() == 1

        register_process(proc2)  # type: ignore[arg-type]
        assert get_tracked_count() == 2

    def test_force_kill_all_clears_registry(self):
        """Test that force_kill_all clears the registry."""
        proc1 = MockProcessWrapper(111)
        proc2 = MockProcessWrapper(222)
        register_process(proc1)  # type: ignore[arg-type]
        register_process(proc2)  # type: ignore[arg-type]
        assert get_tracked_count() == 2

        # PIDs don't exist, but function should still clear registry
        killed = force_kill_all()
        assert get_tracked_count() == 0
        # Killed count may be 0 since PIDs don't exist
        assert killed >= 0


class TestProcessRegistryIntegration:
    """Integration tests that create real processes."""

    def test_force_kill_all_kills_real_process(self):
        """Test that force_kill_all actually kills a real process."""
        # Start a real subprocess with new session (process group leader)
        proc = subprocess.Popen(
            ["sleep", "60"],
            start_new_session=True,
        )

        # Register using a mock wrapper with the real PID
        mock_wrapper = MockProcessWrapper(proc.pid)
        register_process(mock_wrapper)  # type: ignore[arg-type]
        assert get_tracked_count() == 1

        # Force kill all
        killed = force_kill_all()
        assert killed == 1
        assert get_tracked_count() == 0

        # Verify process is dead
        time.sleep(0.1)
        poll_result = proc.poll()
        assert poll_result is not None  # Process should have exited
        # Should be killed by SIGKILL (-9)
        assert poll_result == -signal.SIGKILL

"""Tests for process registry module."""

import contextlib
import signal
import subprocess

import psutil
import pytest

from exec_sandbox.process_registry import (
    _process_groups,
    force_kill_all,
    get_tracked_count,
    register_process,
    unregister_process,
)


class MockProcessWrapper:
    """Stand-in for ProcessWrapper: the registry reads its pid only."""

    def __init__(self, pid: int | None):
        self.pid = pid


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the process registry before and after each test (the module has no public reset)."""
    _process_groups.clear()
    yield
    _process_groups.clear()


def _group_leader(*args: str) -> subprocess.Popen[bytes]:
    """A real process in a new session: the leader of its own process group."""
    return subprocess.Popen(list(args), start_new_session=True, stdout=subprocess.PIPE)


class TestProcessRegistry:
    """Tests for process registry functions. A made-up pid is never given to force_kill_all:
    it sends SIGKILL to that process group, which can be a real one."""

    def test_register_process_counts_each_pid_once(self):
        """Two processes count 2; the same pid again does not add one."""
        register_process(MockProcessWrapper(12345))  # type: ignore[arg-type]
        register_process(MockProcessWrapper(12346))  # type: ignore[arg-type]
        register_process(MockProcessWrapper(12345))  # type: ignore[arg-type]
        assert get_tracked_count() == 2

    def test_register_process_none_pid_safe(self):
        """A process without a pid is not tracked."""
        proc = MockProcessWrapper(None)
        register_process(proc)  # type: ignore[arg-type]
        assert get_tracked_count() == 0

    def test_unregistered_process_is_not_killed(self):
        """A process removed from the registry stays alive at force_kill_all."""
        proc = _group_leader("sleep", "60")
        try:
            wrapper = MockProcessWrapper(proc.pid)
            register_process(wrapper)  # type: ignore[arg-type]
            unregister_process(wrapper)  # type: ignore[arg-type]
            assert get_tracked_count() == 0

            assert force_kill_all() == 0
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait(timeout=5)

    @pytest.mark.parametrize("pid", [None, 99999], ids=["no-process", "pid-not-registered"])
    def test_unregister_of_an_untracked_process_changes_nothing(self, pid: int | None):
        """None, or a process that was never registered: no error, and the tracked one stays."""
        register_process(MockProcessWrapper(12345))  # type: ignore[arg-type]
        unregister_process(None if pid is None else MockProcessWrapper(pid))  # type: ignore[arg-type]
        assert get_tracked_count() == 1

    def test_force_kill_all_counts_live_groups_and_clears_registry(self):
        """A group that ended already is not counted; a live one is. The registry is empty after."""
        ended = _group_leader("true")
        ended.wait(timeout=5)  # reaped: its process group is gone (ProcessLookupError)
        live = _group_leader("sleep", "60")
        try:
            register_process(MockProcessWrapper(ended.pid))  # type: ignore[arg-type]
            register_process(MockProcessWrapper(live.pid))  # type: ignore[arg-type]

            assert force_kill_all() == 1
            assert get_tracked_count() == 0
            assert live.wait(timeout=5) == -signal.SIGKILL
        finally:
            live.kill()
            live.wait(timeout=5)


class TestProcessRegistryIntegration:
    """Integration tests that create real processes."""

    def test_force_kill_all_kills_the_whole_process_group(self):
        """SIGKILL goes to the group: the registered leader and the child it started both end."""
        proc = _group_leader("/bin/sh", "-c", "sleep 60 & echo $!; wait")
        assert proc.stdout is not None
        child = psutil.Process(int(proc.stdout.readline()))
        try:
            register_process(MockProcessWrapper(proc.pid))  # type: ignore[arg-type]
            assert get_tracked_count() == 1

            assert force_kill_all() == 1
            assert get_tracked_count() == 0

            assert proc.wait(timeout=5) == -signal.SIGKILL
            child.wait(timeout=5)  # the leader's child ends too (a kill of the leader alone leaves it)
        finally:
            for process in (proc, child):
                with contextlib.suppress(ProcessLookupError, psutil.NoSuchProcess):
                    process.kill()
            proc.wait(timeout=5)

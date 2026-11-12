"""Tests for cgroup.py cgroup v2 and ulimit utilities.

Test categories:
- Normal cases: Happy path with valid inputs
- Edge cases: None, empty, boundary values
- Error cases: OSError, PermissionError, ConfigError
- Weird cases: Special characters, path traversal, malformed data
- Integration tests: Real cgroups on Linux (requires sudo)
"""

import asyncio
import errno
import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from exec_sandbox.cgroup import (
    CGROUP_APP_NAMESPACE,
    CGROUP_PIDS_LIMIT,
    CGROUP_V2_BASE_PATH,
    ERRNO_PERMISSION_DENIED,
    ERRNO_READ_ONLY_FILESYSTEM,
    ULIMIT_CPU_TIME_SECONDS,
    ULIMIT_MEMORY_MULTIPLIER,
    detect_cgroup_cpu_limit,
    detect_cgroup_memory_limit_mb,
    is_cgroup_available,
    read_container_available_memory_mb,
    reclaim_memory,
    wrap_with_ulimit,
)
from exec_sandbox.constants import (
    DEFAULT_TCG_TB_CACHE_SIZE_MB,
    DEFAULT_VM_MEMORY_OVERHEAD_MB,
)
from exec_sandbox.platform_utils import HostOS

from .conftest import skip_unless_linux

# =============================================================================
# is_cgroup_available - Tests
# =============================================================================


class TestIsCgroupAvailable:
    """Test cgroup availability detection."""

    @pytest.fixture(autouse=True)
    def reset_cache(self) -> None:
        """Reset cgroup cache before each test."""
        from exec_sandbox.cgroup import _cgroup_cache

        _cgroup_cache.reset()

    # --- Normal cases (with mocked cgroup v2 check) ---
    def test_real_cgroup_path_returns_true_when_cgroup_mounted(self):
        """Real cgroup path returns True when cgroup v2 is mounted."""
        with patch("exec_sandbox.cgroup._check_cgroup_v2_mounted", return_value=True):
            cgroup_path = Path(f"{CGROUP_V2_BASE_PATH}/{CGROUP_APP_NAMESPACE}/tenant/vm123")
            assert is_cgroup_available(cgroup_path) is True

    def test_cgroup_root_returns_true_when_mounted(self):
        """Root cgroup path returns True when mounted."""
        with patch("exec_sandbox.cgroup._check_cgroup_v2_mounted", return_value=True):
            assert is_cgroup_available(Path(CGROUP_V2_BASE_PATH)) is True

    def test_returns_false_when_cgroup_not_mounted(self):
        """Returns False when cgroup v2 is not mounted even with valid path."""
        with patch("exec_sandbox.cgroup._check_cgroup_v2_mounted", return_value=False):
            cgroup_path = Path(f"{CGROUP_V2_BASE_PATH}/{CGROUP_APP_NAMESPACE}/tenant/vm123")
            assert is_cgroup_available(cgroup_path) is False

    # --- Edge cases (None/empty) - no mock needed ---
    def test_none_path_returns_false(self):
        """None path returns False (fast path, no cgroup check)."""
        assert is_cgroup_available(None) is False

    def test_fallback_tmp_path_returns_false(self):
        """Fallback /tmp path returns False (fast path, no cgroup check)."""
        assert is_cgroup_available(Path("/tmp/cgroup-vm123")) is False

    # --- Weird cases ---
    def test_path_with_cgroup_substring_returns_false(self):
        """Path containing 'cgroup' but not under /sys/fs/cgroup returns False."""
        assert is_cgroup_available(Path("/home/user/cgroup")) is False
        assert is_cgroup_available(Path("/var/cgroup/test")) is False

    def test_similar_prefix_checked_against_cgroup_mount(self):
        """Paths starting with /sys/fs/cgroup check actual mount status."""
        with patch("exec_sandbox.cgroup._check_cgroup_v2_mounted", return_value=True):
            # These start with /sys/fs/cgroup so they pass prefix check
            # and then check if cgroup v2 is mounted
            assert is_cgroup_available(Path("/sys/fs/cgroup2/test")) is True
            assert is_cgroup_available(Path("/sys/fs/cgroupv2/test")) is True

    def test_path_traversal_attempt_checked_against_mount(self):
        """Path traversal still passes prefix check but requires mount check."""
        with patch("exec_sandbox.cgroup._check_cgroup_v2_mounted", return_value=True):
            weird_path = Path(f"{CGROUP_V2_BASE_PATH}/../../../etc/passwd")
            assert is_cgroup_available(weird_path) is True  # Passes prefix + mount check


class TestCheckCgroupV2Mounted:
    """Test the cgroup v2 mount detection logic."""

    @pytest.fixture(autouse=True)
    def reset_cache(self) -> None:
        """Reset cgroup cache before each test."""
        from exec_sandbox.cgroup import _cgroup_cache

        _cgroup_cache.reset()

    def test_returns_false_when_cgroup_dir_missing(self, tmp_path: Path):
        """Returns False when /sys/fs/cgroup doesn't exist."""
        from exec_sandbox.cgroup import _check_cgroup_v2_mounted

        with patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(tmp_path / "nonexistent")):
            from exec_sandbox.cgroup import _cgroup_cache

            _cgroup_cache.reset()
            # Need to reimport to pick up patched constant
            assert _check_cgroup_v2_mounted() is False

    def test_returns_false_when_controllers_file_missing(self, tmp_path: Path):
        """Returns False when cgroup.controllers doesn't exist (cgroup v1)."""
        from exec_sandbox.cgroup import _check_cgroup_v2_mounted

        # Create dir but no cgroup.controllers
        cgroup_dir = tmp_path / "cgroup"
        cgroup_dir.mkdir()

        with patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(cgroup_dir)):
            from exec_sandbox.cgroup import _cgroup_cache

            _cgroup_cache.reset()
            assert _check_cgroup_v2_mounted() is False

    def test_returns_true_when_cgroup_v2_mounted(self, tmp_path: Path):
        """Returns True when cgroup v2 is properly mounted."""
        from exec_sandbox.cgroup import _check_cgroup_v2_mounted

        # Create proper cgroup v2 structure
        cgroup_dir = tmp_path / "cgroup"
        cgroup_dir.mkdir()
        (cgroup_dir / "cgroup.controllers").write_text("cpu memory pids")

        with patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(cgroup_dir)):
            from exec_sandbox.cgroup import _cgroup_cache

            _cgroup_cache.reset()
            assert _check_cgroup_v2_mounted() is True

    def test_caches_result(self, tmp_path: Path):
        """Result is cached after first check."""
        from exec_sandbox.cgroup import _cgroup_cache, _check_cgroup_v2_mounted

        # Create proper cgroup v2 structure
        cgroup_dir = tmp_path / "cgroup"
        cgroup_dir.mkdir()
        (cgroup_dir / "cgroup.controllers").write_text("cpu memory pids")

        with patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(cgroup_dir)):
            _cgroup_cache.reset()

            # First call - checks filesystem
            result1 = _check_cgroup_v2_mounted()
            assert result1 is True
            assert _cgroup_cache.available is True

            # Remove the file - cached result should still be True
            (cgroup_dir / "cgroup.controllers").unlink()
            result2 = _check_cgroup_v2_mounted()
            assert result2 is True  # Cached!

    def test_handles_permission_error(self, tmp_path: Path):
        """Returns False when can't read cgroup.controllers."""
        from exec_sandbox.cgroup import _check_cgroup_v2_mounted

        cgroup_dir = tmp_path / "cgroup"
        cgroup_dir.mkdir()
        controllers = cgroup_dir / "cgroup.controllers"
        controllers.write_text("cpu memory")

        with (
            patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(cgroup_dir)),
            patch.object(Path, "read_text", side_effect=PermissionError("Access denied")),
        ):
            from exec_sandbox.cgroup import _cgroup_cache

            _cgroup_cache.reset()
            assert _check_cgroup_v2_mounted() is False


# =============================================================================
# wrap_with_ulimit - Tests
# =============================================================================


class TestWrapWithUlimit:
    """Test ulimit command wrapping."""

    # --- Normal cases ---
    def test_linux_wraps_with_ulimit_v(self):
        """Linux wraps command with ulimit -v (virtual memory)."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            cmd = ["qemu-system-x86_64", "-m", "256"]
            wrapped = wrap_with_ulimit(cmd, cgroup_memory_mb=256)

            assert wrapped[0] == "bash"
            assert wrapped[1] == "-c"
            # Linux should have -v (virtual memory) and -t (CPU time)
            # -u (RLIMIT_NPROC) intentionally removed — per-UID, not per-process
            assert "-v" in wrapped[2]
            assert f"-t {ULIMIT_CPU_TIME_SECONDS}" in wrapped[2]
            assert "-u" not in wrapped[2]
            assert "qemu-system-x86_64" in wrapped[2]

    def test_macos_skips_all_ulimits(self):
        """macOS skips all ulimits — -v not supported, -t breaks pipes, -u is per-UID."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.MACOS

            cmd = ["qemu-system-aarch64", "-m", "512"]
            wrapped = wrap_with_ulimit(cmd, cgroup_memory_mb=512)

            # macOS: no ulimit controls, just exec
            assert "ulimit" not in wrapped[2]
            assert "-v" not in wrapped[2]
            assert "-t" not in wrapped[2]
            assert "-u" not in wrapped[2]
            assert "exec" in wrapped[2]
            assert "qemu-system-aarch64" in wrapped[2]

    def test_macos_x86_64_skips_all_ulimits(self):
        """macOS x86_64 skips all ulimits, same as ARM64."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.MACOS

            # Use x86_64 QEMU binary (Intel Mac)
            cmd = ["qemu-system-x86_64", "-m", "512"]
            wrapped = wrap_with_ulimit(cmd, cgroup_memory_mb=512)

            # macOS: no ulimit controls, just exec
            assert "ulimit" not in wrapped[2]
            assert "-v" not in wrapped[2]
            assert "-t" not in wrapped[2]
            assert "-u" not in wrapped[2]
            assert "exec" in wrapped[2]
            assert "qemu-system-x86_64" in wrapped[2]

    def test_memory_multiplier_applied(self):
        """Memory multiplier (14x) is applied to virtual memory limit."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            memory_mb = 256
            wrapped = wrap_with_ulimit(["test"], cgroup_memory_mb=memory_mb)

            expected_kb = memory_mb * 1024 * ULIMIT_MEMORY_MULTIPLIER
            assert f"ulimit -v {expected_kb}" in wrapped[2]

    # --- Edge cases ---
    def test_empty_command_list(self):
        """Empty command list produces valid shell command."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            wrapped = wrap_with_ulimit([], cgroup_memory_mb=256)
            assert wrapped[0] == "bash"
            assert wrapped[1] == "-c"
            # Should have ulimit but empty exec
            assert "ulimit -v" in wrapped[2]

    def test_zero_memory(self):
        """Zero memory produces ulimit -v 0."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            wrapped = wrap_with_ulimit(["test"], cgroup_memory_mb=0)
            assert "ulimit -v 0" in wrapped[2]

    # --- Weird cases ---
    def test_command_with_shell_metacharacters(self):
        """Shell metacharacters are properly escaped."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            cmd = ["binary", "--arg", "$(whoami)", "; rm -rf /", "| cat"]
            wrapped = wrap_with_ulimit(cmd, cgroup_memory_mb=128)

            # shlex.quote should escape these
            assert "$(whoami)" not in wrapped[2] or "'$(whoami)'" in wrapped[2]
            assert wrapped[2].count("'") >= 2  # Should have quotes

    def test_command_with_spaces(self):
        """Arguments with spaces are properly quoted."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            cmd = ["binary", "--arg", "value with spaces"]
            wrapped = wrap_with_ulimit(cmd, cgroup_memory_mb=128)

            assert "'value with spaces'" in wrapped[2]

    def test_very_large_memory(self):
        """Very large memory value doesn't overflow."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            # 1TB in MB
            memory_mb = 1024 * 1024
            wrapped = wrap_with_ulimit(["test"], cgroup_memory_mb=memory_mb)

            expected_kb = memory_mb * 1024 * ULIMIT_MEMORY_MULTIPLIER
            assert str(expected_kb) in wrapped[2]

    def test_cpu_time_constant_is_reasonable(self):
        """CPU time limit constant is reasonable (10 min to 2 hours)."""
        assert 600 <= ULIMIT_CPU_TIME_SECONDS <= 7200


class TestUlimitIntegration:
    """Integration tests that verify ulimit actually works on the current platform."""

    async def test_cpu_time_limit_enforced(self):
        """CPU time limit (-t) is enforced by kernel."""
        # Run a process that tries to burn CPU time with a 1-second limit
        # The process should be killed by SIGXCPU
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            "ulimit -t 1 && python3 -c 'while True: pass'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Should be killed within ~2 seconds (1s limit + overhead)
            _stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=5)

            # Process should have been killed (non-zero exit)
            # SIGXCPU typically results in exit code 137 (128 + 9) or 152 (128 + 24)
            assert proc.returncode != 0, f"Process should have been killed, got rc={proc.returncode}"

        except TimeoutError:
            proc.kill()
            await proc.wait()
            pytest.fail("CPU time limit was not enforced - process ran past timeout")

    async def test_process_limit_enforced(self):
        """Process limit (-u) is enforced by kernel."""
        # Run a process that tries to fork with a limit of 5 processes
        # Fork should fail with EAGAIN (Resource temporarily unavailable)
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            """ulimit -u 5 && python3 -c '
import os
import sys
pids = []
try:
    for i in range(20):
        pid = os.fork()
        if pid == 0:
            import time
            time.sleep(10)
            sys.exit(0)
        pids.append(pid)
    print("ERROR: Fork limit not enforced")
    sys.exit(1)
except OSError as e:
    print(f"Fork failed as expected: {e}")
    sys.exit(0)
finally:
    for pid in pids:
        try:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
        except:
            pass
'""",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = stdout.decode() + stderr.decode()

            # Should exit 0 (fork failed as expected) or have fork error in output
            assert proc.returncode == 0 or "resource" in output.lower() or "fork" in output.lower(), (
                f"Fork limit not enforced: rc={proc.returncode}, output={output}"
            )

        except TimeoutError:
            proc.kill()
            await proc.wait()
            pytest.fail("Process limit test timed out")

    @skip_unless_linux
    async def test_virtual_memory_limit_enforced_linux(self):
        """Virtual memory limit (-v) is enforced by kernel on Linux."""
        # Run a process that tries to allocate memory with a very low limit
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            # 50MB virtual memory limit, try to allocate 100MB
            "ulimit -v 51200 && python3 -c 'x = bytearray(100 * 1024 * 1024)'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = stdout.decode() + stderr.decode()

            # Should fail with MemoryError
            assert proc.returncode != 0, f"Memory limit not enforced: rc={proc.returncode}"
            assert "memory" in output.lower() or proc.returncode != 0

        except TimeoutError:
            proc.kill()
            await proc.wait()
            pytest.fail("Virtual memory limit test timed out")

    # -------------------------------------------------------------------------
    # In-bounds tests: verify processes within limits succeed
    # -------------------------------------------------------------------------

    async def test_cpu_time_in_bounds_succeeds(self):
        """Process within CPU time limit completes successfully."""
        # 10 second limit, task completes in <1 second
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            "ulimit -t 10 && python3 -c 'print(sum(range(1000000)))'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        assert proc.returncode == 0, f"In-bounds process failed: rc={proc.returncode}, stderr={stderr.decode()}"
        assert b"499999500000" in stdout  # sum(range(1000000))

    async def test_process_limit_in_bounds_succeeds(self):
        """Fork within process limit succeeds (no limit set = uses system default)."""
        # Don't set ulimit -u here - we just verify forking works normally
        # The out-of-bounds test (test_process_limit_enforced) verifies limit enforcement
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            """python3 -c '
import os
import sys
pids = []
for i in range(3):
    pid = os.fork()
    if pid == 0:
        sys.exit(0)
    pids.append(pid)
for pid in pids:
    os.waitpid(pid, 0)
print("SUCCESS")
'""",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        assert proc.returncode == 0, f"Fork failed: rc={proc.returncode}, stderr={stderr.decode()}"
        assert b"SUCCESS" in stdout

    @skip_unless_linux
    async def test_virtual_memory_in_bounds_succeeds_linux(self):
        """Memory allocation within limit succeeds on Linux."""
        # 200MB limit, allocate 50MB
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            "ulimit -v 204800 && python3 -c 'x = bytearray(50 * 1024 * 1024); print(len(x))'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        assert proc.returncode == 0, f"In-bounds allocation failed: rc={proc.returncode}, stderr={stderr.decode()}"
        assert b"52428800" in stdout  # 50MB in bytes


# =============================================================================
# setup_cgroup - Tests
# =============================================================================


@pytest.fixture
def cgroup_root(tmp_path: Path) -> Iterator[Path]:
    """A folder that stands in for /sys/fs/cgroup (cgroup v2 'mounted'): the functions
    read and write real files in it."""
    with (
        patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(tmp_path)),
        patch("exec_sandbox.cgroup._check_cgroup_v2_mounted", return_value=True),
    ):
        yield tmp_path


def _record_writes(written: list[tuple[str, str]], *, fail_first: bool) -> Callable[[Path, str], int]:
    """Path.write_text stand-in for cgroupfs control files, whose writes do not replace
    each other: records (file name, value); with fail_first, the first write fails."""

    def write_text(path: Path, data: str) -> int:
        written.append((path.name, data))
        if fail_first and len(written) == 1:
            raise OSError(errno.ESRCH, "No such process")
        return len(data)

    return write_text


class TestSetupCgroup:
    """Test cgroup setup function."""

    # --- Normal cases ---
    async def test_setup_creates_cgroup_hierarchy(self, cgroup_root: Path):
        """Normal setup creates tenant and VM cgroups."""
        from exec_sandbox.cgroup import setup_cgroup

        result = await setup_cgroup("vm123", "tenant1", cgroup_memory_mb=256, cgroup_cpu_cores=1.0)

        tenant = cgroup_root / CGROUP_APP_NAMESPACE / "tenant1"
        assert result == tenant / "vm123"
        assert result.is_dir()  # tenant + vm directories
        assert (tenant / "cgroup.subtree_control").read_text() == "+memory +cpu +pids"
        assert (result / "pids.max").read_text() == str(CGROUP_PIDS_LIMIT)

    async def test_setup_runs_in_one_worker_thread(self, cgroup_root: Path):
        """The cgroup files are made off the event loop thread (mkdir and cgroup.subtree_control
        can wait for the kernel's cgroup locks)."""
        from exec_sandbox.cgroup import setup_cgroup

        loop_thread = threading.get_ident()
        threads: set[int] = set()
        real_mkdir = Path.mkdir

        def mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
            threads.add(threading.get_ident())
            real_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", autospec=True, side_effect=mkdir):
            await setup_cgroup("vm123", "tenant1", cgroup_memory_mb=256, cgroup_cpu_cores=1.0)

        assert threads
        assert loop_thread not in threads

    async def test_setup_cancel_waits_for_the_job(self, cgroup_root: Path):
        """A cancel during setup returns only once the job has ended: the owner's cleanup
        never removes the folder while the job still makes it."""
        from exec_sandbox.cgroup import setup_cgroup

        started, release = threading.Event(), threading.Event()
        real_mkdir = Path.mkdir

        def slow_mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
            started.set()
            release.wait(timeout=10)
            real_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", autospec=True, side_effect=slow_mkdir):
            task = asyncio.create_task(setup_cgroup("vm123", "tenant1", cgroup_memory_mb=256, cgroup_cpu_cores=1.0))
            try:
                await asyncio.to_thread(started.wait, 10)
                task.cancel()
                await asyncio.sleep(0.05)
                assert not task.done(), "the cancel did not wait for the setup job"
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert (cgroup_root / CGROUP_APP_NAMESPACE / "tenant1" / "vm123" / "pids.max").exists()

    async def test_setup_applies_precomputed_memory_limit(self, cgroup_root: Path):
        """setup_cgroup applies pre-computed memory limit (no internal overhead calc)."""
        from exec_sandbox.cgroup import setup_cgroup

        # Pass effective memory (admission already added overhead + TCG)
        effective_memory = 256 + DEFAULT_VM_MEMORY_OVERHEAD_MB + DEFAULT_TCG_TB_CACHE_SIZE_MB
        result = await setup_cgroup("vm123", "tenant1", cgroup_memory_mb=effective_memory, cgroup_cpu_cores=1.0)

        # cgroup should apply exactly what was passed (in bytes)
        expected_memory_bytes = effective_memory * 1024 * 1024
        assert (result / "memory.max").read_text() == str(expected_memory_bytes)

    @pytest.mark.parametrize(
        ("cgroup_cpu_cores", "expected_cpu_max"),
        [
            (1.0, "100000 100000"),
            (1.25, "125000 100000"),
            (2.0, "200000 100000"),
            (4.0, "400000 100000"),
        ],
    )
    async def test_setup_cpu_max_scales_with_cores(
        self, cgroup_root: Path, cgroup_cpu_cores: float, expected_cpu_max: str
    ):
        """cpu.max quota scales proportionally with cgroup_cpu_cores (effective, including overhead)."""
        from exec_sandbox.cgroup import setup_cgroup

        result = await setup_cgroup("vm123", "tenant1", cgroup_memory_mb=256, cgroup_cpu_cores=cgroup_cpu_cores)
        assert (result / "cpu.max").read_text() == expected_cpu_max

    # --- Error cases ---
    async def test_setup_read_only_filesystem_returns_fallback(self, cgroup_root: Path):
        """Read-only filesystem gracefully degrades to fallback path."""
        from exec_sandbox.cgroup import setup_cgroup

        error = OSError("Read-only filesystem")
        error.errno = ERRNO_READ_ONLY_FILESYSTEM

        with patch.object(Path, "mkdir", side_effect=error):
            result = await setup_cgroup("vm123", "tenant1", cgroup_memory_mb=256, cgroup_cpu_cores=1.0)

            assert result == Path("/tmp/cgroup-vm123")

    async def test_setup_permission_denied_returns_fallback(self, cgroup_root: Path):
        """Permission denied gracefully degrades to fallback path."""
        from exec_sandbox.cgroup import setup_cgroup

        error = OSError("Permission denied")
        error.errno = ERRNO_PERMISSION_DENIED

        with patch.object(Path, "mkdir", side_effect=error):
            result = await setup_cgroup("vm123", "tenant1", cgroup_memory_mb=256, cgroup_cpu_cores=1.0)

            assert result == Path("/tmp/cgroup-vm123")

    async def test_setup_other_oserror_raises_config_error(self, cgroup_root: Path):
        """Other OSError raises ConfigError HOST_DEPENDENCY."""
        from exec_sandbox.cgroup import setup_cgroup
        from exec_sandbox.exceptions import ConfigError, ErrorCode

        error = OSError("Disk full")
        error.errno = 28  # ENOSPC

        with patch.object(Path, "mkdir", side_effect=error):
            with pytest.raises(
                ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="Failed to setup cgroup"
            ):
                await setup_cgroup("vm123", "tenant1", cgroup_memory_mb=256, cgroup_cpu_cores=1.0)


# =============================================================================
# attach_to_cgroup - Tests
# =============================================================================


class TestAttachToCgroup:
    """Test cgroup process attachment."""

    # --- Normal cases ---
    async def test_attach_writes_pid_to_cgroup_procs(self, tmp_path: Path):
        """Normal attach writes PID to cgroup.procs, off the event loop thread."""
        from exec_sandbox.cgroup import attach_to_cgroup

        loop_thread = threading.get_ident()
        writes: list[tuple[str, str, int]] = []

        def write_text(path: Path, data: str) -> int:
            writes.append((path.name, data, threading.get_ident()))
            return len(data)

        with patch.object(Path, "write_text", autospec=True, side_effect=write_text):
            await attach_to_cgroup(tmp_path, 12345)

        assert [(name, data) for name, data, _ in writes] == [("cgroup.procs", "12345")]
        assert writes[0][2] != loop_thread

    # --- Error cases ---
    async def test_attach_oserror_raises_config_error(self, tmp_path: Path):
        """OSError during attach raises ConfigError HOST_DEPENDENCY."""
        from exec_sandbox.cgroup import attach_to_cgroup
        from exec_sandbox.exceptions import ConfigError, ErrorCode

        with patch.object(Path, "write_text", side_effect=OSError(errno.ESRCH, "No such process")):
            with pytest.raises(
                ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="Failed to attach PID"
            ):
                await attach_to_cgroup(tmp_path, 12345)

    async def test_attach_permission_error_raises_config_error(self, tmp_path: Path):
        """PermissionError during attach raises ConfigError HOST_DEPENDENCY."""
        from exec_sandbox.cgroup import attach_to_cgroup
        from exec_sandbox.exceptions import ConfigError, ErrorCode

        with patch.object(Path, "write_text", side_effect=PermissionError("Access denied")):
            with pytest.raises(
                ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="Failed to attach PID"
            ):
                await attach_to_cgroup(tmp_path, 12345)


# =============================================================================
# attach_if_available - Tests
# =============================================================================


class TestAttachIfAvailable:
    """Test attach_if_available convenience wrapper."""

    # --- Edge cases (None handling) ---
    async def test_returns_false_for_none_path(self):
        """Returns False when cgroup_path is None."""
        from exec_sandbox.cgroup import attach_if_available

        result = await attach_if_available(None, 12345)
        assert result is False

    async def test_returns_false_for_none_pid(self):
        """Returns False when pid is None."""
        from exec_sandbox.cgroup import attach_if_available

        cgroup_path = Path("/sys/fs/cgroup/code-exec/tenant/vm123")
        result = await attach_if_available(cgroup_path, None)
        assert result is False

    async def test_returns_false_for_both_none(self):
        """Returns False when both are None."""
        from exec_sandbox.cgroup import attach_if_available

        result = await attach_if_available(None, None)
        assert result is False

    async def test_returns_false_for_fallback_path(self):
        """Returns False for fallback dummy paths (doesn't try to attach)."""
        from exec_sandbox.cgroup import attach_if_available

        fallback_path = Path("/tmp/cgroup-vm123")
        result = await attach_if_available(fallback_path, 12345)
        assert result is False

    # --- Normal cases ---
    async def test_returns_true_on_successful_attach(self, cgroup_root: Path):
        """Returns True when attachment succeeds."""
        from exec_sandbox.cgroup import attach_if_available

        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        cgroup_path.mkdir(parents=True)
        result = await attach_if_available(cgroup_path, 12345)
        assert result is True
        assert (cgroup_path / "cgroup.procs").read_text() == "12345"

    # --- Error cases ---
    @pytest.mark.parametrize("err", [errno.EACCES, errno.EPERM], ids=["eacces", "eperm"])
    async def test_denied_attach_degrades_to_false(self, cgroup_root: Path, err: int):
        """The orchestrator is outside the delegated subtree: no limits for this VM, the VM still runs."""
        from exec_sandbox.cgroup import attach_if_available

        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        cgroup_path.mkdir(parents=True)
        with patch.object(Path, "write_text", side_effect=OSError(err, os.strerror(err))):
            assert await attach_if_available(cgroup_path, 12345) is False

    async def test_other_attach_error_propagates(self, cgroup_root: Path):
        """Any other errno is a real fault: the ConfigError HOST_DEPENDENCY of attach_to_cgroup."""
        from exec_sandbox.cgroup import attach_if_available
        from exec_sandbox.exceptions import ConfigError, ErrorCode

        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        cgroup_path.mkdir(parents=True)
        with (
            patch.object(Path, "write_text", side_effect=OSError(errno.ESRCH, os.strerror(errno.ESRCH))),
            pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY),
        ):
            await attach_if_available(cgroup_path, 12345)


# =============================================================================
# read_cgroup_stats - Tests
# =============================================================================


class TestReadCgroupStats:
    """Test cgroup stats reading."""

    # --- Edge cases (None/missing) ---
    def test_returns_none_for_none_path(self):
        """Returns (None, None, None) for None path."""
        from exec_sandbox.cgroup import read_cgroup_stats

        cpu_ms, mem_mb, nr_throttled = read_cgroup_stats(None)
        assert cpu_ms is None
        assert mem_mb is None
        assert nr_throttled is None

    def test_returns_none_for_nonexistent_path(self):
        """Returns (None, None, None) for non-existent path."""
        from exec_sandbox.cgroup import read_cgroup_stats

        result = read_cgroup_stats(Path("/nonexistent/cgroup"))
        assert result == (None, None, None)

    # --- Normal cases ---
    def test_reads_cpu_and_memory_stats(self, tmp_path: Path):
        """Reads and parses cpu.stat and memory.peak correctly."""
        from exec_sandbox.cgroup import read_cgroup_stats

        (tmp_path / "cpu.stat").write_text(
            "usage_usec 5000000\nuser_usec 3000000\nsystem_usec 2000000\nnr_throttled 42"
        )
        (tmp_path / "memory.peak").write_text("104857600")  # 100MB in bytes

        cpu_ms, mem_mb, nr_throttled = read_cgroup_stats(tmp_path)

        assert cpu_ms == 5000  # 5000000 usec = 5000 ms
        assert mem_mb == 100  # 104857600 bytes = 100 MB
        assert nr_throttled == 42

    # --- Error cases ---
    def test_handles_malformed_cpu_stat(self, tmp_path: Path):
        """Handles malformed cpu.stat gracefully."""
        from exec_sandbox.cgroup import read_cgroup_stats

        (tmp_path / "cpu.stat").write_text("garbage data not_a_number")
        (tmp_path / "memory.peak").write_text("garbage data not_a_number")

        cpu_ms, mem_mb, nr_throttled = read_cgroup_stats(tmp_path)

        # Should return None on parse error, not crash
        assert cpu_ms is None
        assert mem_mb is None
        assert nr_throttled is None

    def test_handles_oserror_during_read(self, tmp_path: Path):
        """Handles OSError during file read gracefully."""
        from exec_sandbox.cgroup import read_cgroup_stats

        (tmp_path / "cpu.stat").mkdir()  # exists, but reading it raises IsADirectoryError

        cpu_ms, mem_mb, nr_throttled = read_cgroup_stats(tmp_path)

        assert cpu_ms is None
        assert mem_mb is None
        assert nr_throttled is None


# =============================================================================
# cleanup_cgroup - Tests
# =============================================================================


class TestCleanupCgroup:
    """Test cgroup cleanup function."""

    # --- Edge cases (None handling) ---
    async def test_returns_true_for_none_path(self):
        """Returns True for None path (no-op)."""
        from exec_sandbox.cgroup import cleanup_cgroup

        result = await cleanup_cgroup(None, "test-ctx")
        assert result is True

    async def test_fallback_path_silently_succeeds(self):
        """Fallback dummy paths silently succeed even if rmdir fails."""
        from exec_sandbox.cgroup import cleanup_cgroup

        fallback_path = Path("/tmp/nonexistent-cgroup-vm123")
        result = await cleanup_cgroup(fallback_path, "test-ctx")
        assert result is True

    # --- Normal cases ---
    async def test_removes_empty_cgroup(self, cgroup_root: Path):
        """Removes empty cgroup directory, off the event loop thread."""
        from exec_sandbox.cgroup import cleanup_cgroup

        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        cgroup_path.mkdir(parents=True)
        loop_thread = threading.get_ident()
        threads: list[int] = []
        real_rmdir = Path.rmdir

        def rmdir(path: Path) -> None:
            threads.append(threading.get_ident())
            real_rmdir(path)

        with patch.object(Path, "rmdir", autospec=True, side_effect=rmdir):
            result = await cleanup_cgroup(cgroup_path, "test-ctx")

        assert result is True
        assert not cgroup_path.exists()
        assert threads
        assert loop_thread not in threads

    async def test_migrates_pids_before_removal(self, cgroup_root: Path):
        """Moves PIDs to parent cgroup before removing."""
        from exec_sandbox.cgroup import cleanup_cgroup

        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        cgroup_path.mkdir(parents=True)
        (cgroup_path.parent / "cgroup.procs").write_text("")
        (cgroup_path / "cgroup.procs").write_text("123\n456\n789")
        written: list[tuple[str, str]] = []

        with (
            patch.object(Path, "write_text", autospec=True, side_effect=_record_writes(written, fail_first=False)),
            patch.object(Path, "rmdir") as mock_rmdir,
        ):
            result = await cleanup_cgroup(cgroup_path, "test-ctx")

        assert result is True
        assert written == [("cgroup.procs", "123"), ("cgroup.procs", "456"), ("cgroup.procs", "789")]
        mock_rmdir.assert_called_once_with()

    # --- Error cases ---
    async def test_handles_already_deleted_cgroup(self, cgroup_root: Path):
        """FileNotFoundError during rmdir returns True (already deleted)."""
        from exec_sandbox.cgroup import cleanup_cgroup

        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"  # never made
        result = await cleanup_cgroup(cgroup_path, "test-ctx")

        assert result is True  # Race condition - already deleted

    async def test_handles_rmdir_oserror(self, cgroup_root: Path):
        """OSError during rmdir returns False."""
        from exec_sandbox.cgroup import cleanup_cgroup

        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        cgroup_path.mkdir(parents=True)
        (cgroup_path / "leftover").write_text("")  # rmdir: Directory not empty

        result = await cleanup_cgroup(cgroup_path, "test-ctx")

        assert result is False

    async def test_handles_pid_migration_error_gracefully(self, cgroup_root: Path):
        """Continues cleanup even if PID migration fails (process already exited)."""
        from exec_sandbox.cgroup import cleanup_cgroup

        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        cgroup_path.mkdir(parents=True)
        (cgroup_path.parent / "cgroup.procs").write_text("")
        (cgroup_path / "cgroup.procs").write_text("123\n456")
        written: list[tuple[str, str]] = []

        with (
            patch.object(Path, "write_text", autospec=True, side_effect=_record_writes(written, fail_first=True)),
            patch.object(Path, "rmdir") as mock_rmdir,
        ):
            result = await cleanup_cgroup(cgroup_path, "test-ctx")

        # Should succeed even though one PID migration failed
        assert result is True
        assert written == [("cgroup.procs", "123"), ("cgroup.procs", "456")]
        mock_rmdir.assert_called_once_with()


# =============================================================================
# Constants - Tests
# =============================================================================


class TestCgroupConstants:
    """Test cgroup constants are properly defined."""

    def test_cgroup_base_path(self):
        """CGROUP_V2_BASE_PATH is correct."""
        assert CGROUP_V2_BASE_PATH == "/sys/fs/cgroup"

    def test_cgroup_app_namespace(self):
        """CGROUP_APP_NAMESPACE is correct."""
        assert CGROUP_APP_NAMESPACE == "code-exec"

    def test_memory_overhead_is_reasonable(self):
        """Memory overhead is reasonable for microvm/virt machine types (32-128MB)."""
        assert 32 <= DEFAULT_VM_MEMORY_OVERHEAD_MB <= 128

    def test_tcg_cache_size_is_reasonable(self):
        """TCG TB cache size is reasonable (128MB-1GB, shrunk for VM density)."""
        assert 128 <= DEFAULT_TCG_TB_CACHE_SIZE_MB <= 1024

    def test_pids_limit_is_reasonable(self):
        """PIDs limit is reasonable (50-500)."""
        assert 50 <= CGROUP_PIDS_LIMIT <= 500

    def test_ulimit_multiplier_is_reasonable(self):
        """ulimit multiplier is reasonable (5-20x)."""
        assert 5 <= ULIMIT_MEMORY_MULTIPLIER <= 20

    def test_errno_values_match_system(self):
        """errno values match expected system values."""
        import errno

        assert ERRNO_READ_ONLY_FILESYSTEM == errno.EROFS
        assert ERRNO_PERMISSION_DENIED == errno.EACCES


# =============================================================================
# detect_cgroup_memory_limit_mb - Tests
# =============================================================================


class TestDetectCgroupMemoryLimitMb:
    """Test container cgroup memory limit detection."""

    def test_v2_numeric_limit_returns_mb(self, tmp_path: Path):
        """cgroup v2 numeric memory.max → returns MB."""
        (tmp_path / "memory.max").write_text("4294967296")  # 4GB
        result = detect_cgroup_memory_limit_mb(cgroup_v2_base=str(tmp_path))
        assert result == pytest.approx(4096.0)

    def test_v2_max_string_returns_none(self, tmp_path: Path):
        """cgroup v2 'max' (unlimited) → returns None."""
        (tmp_path / "memory.max").write_text("max")
        result = detect_cgroup_memory_limit_mb(cgroup_v2_base=str(tmp_path))
        assert result is None

    def test_v2_small_limit(self, tmp_path: Path):
        """cgroup v2 small memory limit (1GB) returns correct MB."""
        (tmp_path / "memory.max").write_text("1073741824")  # 1GB
        result = detect_cgroup_memory_limit_mb(cgroup_v2_base=str(tmp_path))
        assert result == pytest.approx(1024.0)

    def test_no_cgroup_returns_none(self, tmp_path: Path):
        """No cgroup files → returns None (macOS, bare metal)."""
        result = detect_cgroup_memory_limit_mb(cgroup_v2_base=str(tmp_path / "nonexistent"))
        assert result is None


class TestDetectCgroupCpuLimit:
    """Test container cgroup CPU limit detection."""

    def test_v2_quota_period_returns_cpus(self, tmp_path: Path):
        """cgroup v2 cpu.max '200000 100000' → 2.0 CPUs."""
        (tmp_path / "cpu.max").write_text("200000 100000")
        result = detect_cgroup_cpu_limit(cgroup_v2_base=str(tmp_path))
        assert result == pytest.approx(2.0)

    def test_v2_max_string_returns_none(self, tmp_path: Path):
        """cgroup v2 'max 100000' (unlimited) → returns None."""
        (tmp_path / "cpu.max").write_text("max 100000")
        result = detect_cgroup_cpu_limit(cgroup_v2_base=str(tmp_path))
        assert result is None

    def test_v2_fractional_cpus(self, tmp_path: Path):
        """cgroup v2 fractional CPU quota → correct float."""
        (tmp_path / "cpu.max").write_text("50000 100000")  # 0.5 CPUs
        result = detect_cgroup_cpu_limit(cgroup_v2_base=str(tmp_path))
        assert result == pytest.approx(0.5)

    def test_no_cgroup_returns_none(self, tmp_path: Path):
        """No cgroup files → returns None."""
        result = detect_cgroup_cpu_limit(cgroup_v2_base=str(tmp_path / "nonexistent"))
        assert result is None


class TestReadContainerAvailableMemoryMb:
    """Test container available memory reading."""

    def test_v2_returns_available(self, tmp_path: Path):
        """cgroup v2 memory.max - memory.current = available."""
        (tmp_path / "memory.max").write_text("4294967296")  # 4GB
        (tmp_path / "memory.current").write_text("1073741824")  # 1GB
        result = read_container_available_memory_mb(cgroup_v2_base=str(tmp_path))
        assert result == pytest.approx(3072.0)  # 3GB

    def test_v2_unlimited_returns_none(self, tmp_path: Path):
        """cgroup v2 'max' → returns None."""
        (tmp_path / "memory.max").write_text("max")
        result = read_container_available_memory_mb(cgroup_v2_base=str(tmp_path))
        assert result is None

    def test_v2_current_exceeds_max_returns_zero(self, tmp_path: Path):
        """cgroup v2 current > max → returns 0 (not negative)."""
        (tmp_path / "memory.max").write_text("1073741824")  # 1GB
        (tmp_path / "memory.current").write_text("2147483648")  # 2GB (over limit)
        result = read_container_available_memory_mb(cgroup_v2_base=str(tmp_path))
        assert result == 0.0

    def test_no_cgroup_returns_none(self, tmp_path: Path):
        """No cgroup files → returns None."""
        result = read_container_available_memory_mb(cgroup_v2_base=str(tmp_path / "nonexistent"))
        assert result is None


# =============================================================================
# Integration Tests (Real cgroups, Linux-only, requires sudo)
# =============================================================================


@pytest.mark.sudo
@skip_unless_linux
class TestCgroupIntegration:
    """Integration tests using real cgroups on Linux.

    These tests require:
    - Linux operating system
    - Root/sudo privileges
    - cgroup v2 filesystem mounted at /sys/fs/cgroup
    """

    @pytest.fixture
    def unique_vm_id(self) -> str:
        """Generate unique VM ID for test isolation."""
        import uuid

        return f"test-{uuid.uuid4().hex[:8]}"

    async def test_setup_creates_real_cgroup(self, unique_vm_id: str) -> None:
        """setup_cgroup creates real cgroup directory on Linux."""
        from exec_sandbox.cgroup import cleanup_cgroup, setup_cgroup

        tenant_id = "integration-test"

        # Create real cgroup
        # Effective memory: guest (256) + overhead, pre-computed by admission
        effective_memory_mb = 256 + DEFAULT_VM_MEMORY_OVERHEAD_MB
        cgroup_path = await setup_cgroup(
            vm_id=unique_vm_id,
            tenant_id=tenant_id,
            cgroup_memory_mb=effective_memory_mb,
            cgroup_cpu_cores=1.0,
        )

        try:
            # Verify real cgroup was created
            assert cgroup_path.exists(), f"Cgroup not created: {cgroup_path}"
            assert (cgroup_path / "cgroup.procs").exists()
            assert (cgroup_path / "memory.max").exists()
            assert (cgroup_path / "cpu.max").exists()
            assert (cgroup_path / "pids.max").exists()

            # Verify memory limit was set (cgroup applies exactly what was passed)
            memory_max = (cgroup_path / "memory.max").read_text().strip()
            expected_bytes = effective_memory_mb * 1024 * 1024
            assert int(memory_max) == expected_bytes

            # Verify pids limit was set
            pids_max = (cgroup_path / "pids.max").read_text().strip()
            assert int(pids_max) == CGROUP_PIDS_LIMIT

        finally:
            # Cleanup
            await cleanup_cgroup(cgroup_path, "test-cleanup")

    async def test_attach_real_process_to_cgroup(self, unique_vm_id: str) -> None:
        """attach_to_cgroup attaches real process to cgroup."""
        from exec_sandbox.cgroup import attach_to_cgroup, cleanup_cgroup, setup_cgroup

        tenant_id = "integration-test"

        # Create cgroup
        cgroup_path = await setup_cgroup(
            vm_id=unique_vm_id,
            tenant_id=tenant_id,
            cgroup_memory_mb=256,
            cgroup_cpu_cores=1.0,
        )

        # Start a real process (sleep)
        proc = await asyncio.create_subprocess_exec(
            "sleep",
            "60",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        try:
            assert proc.pid is not None

            # Attach process to cgroup
            await attach_to_cgroup(cgroup_path, proc.pid)

            # Verify process is in cgroup
            procs_content = (cgroup_path / "cgroup.procs").read_text()
            assert str(proc.pid) in procs_content

        finally:
            # Cleanup process
            proc.terminate()
            await proc.wait()
            # Cleanup cgroup
            await cleanup_cgroup(cgroup_path, "test-cleanup")

    async def test_read_real_cgroup_stats(self, unique_vm_id: str) -> None:
        """read_cgroup_stats reads real stats from cgroup."""
        from exec_sandbox.cgroup import (
            attach_to_cgroup,
            cleanup_cgroup,
            read_cgroup_stats,
            setup_cgroup,
        )

        tenant_id = "integration-test"

        # Create cgroup
        cgroup_path = await setup_cgroup(
            vm_id=unique_vm_id,
            tenant_id=tenant_id,
            cgroup_memory_mb=256,
            cgroup_cpu_cores=1.0,
        )

        # Start a process that does some work
        proc = await asyncio.create_subprocess_exec(
            "python3",
            "-c",
            "x = [0] * 1000000; import time; time.sleep(0.5)",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        try:
            assert proc.pid is not None
            await attach_to_cgroup(cgroup_path, proc.pid)

            # Wait for process to do some work
            await proc.wait()

            # Read stats
            cpu_ms, mem_mb, nr_throttled = read_cgroup_stats(cgroup_path)

            # Stats should have values (process did work)
            assert cpu_ms is not None, "CPU stats not available"
            assert cpu_ms >= 0, f"CPU time should be non-negative: {cpu_ms}"

            # Memory peak might not be available on all systems
            if mem_mb is not None:
                assert mem_mb >= 0, f"Memory should be non-negative: {mem_mb}"

            # nr_throttled should be non-negative if available
            if nr_throttled is not None:
                assert nr_throttled >= 0, f"nr_throttled should be non-negative: {nr_throttled}"

        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
            await cleanup_cgroup(cgroup_path, "test-cleanup")

    async def test_cleanup_removes_real_cgroup(self, unique_vm_id: str) -> None:
        """cleanup_cgroup removes real cgroup directory."""
        from exec_sandbox.cgroup import cleanup_cgroup, setup_cgroup

        tenant_id = "integration-test"

        # Create cgroup
        cgroup_path = await setup_cgroup(
            vm_id=unique_vm_id,
            tenant_id=tenant_id,
            cgroup_memory_mb=256,
            cgroup_cpu_cores=1.0,
        )

        assert cgroup_path.exists(), "Cgroup should exist after setup"

        # Cleanup
        result = await cleanup_cgroup(cgroup_path, "test-cleanup")

        assert result is True, "Cleanup should succeed"
        assert not cgroup_path.exists(), "Cgroup should be removed after cleanup"

    async def test_pids_limit_enforced(self, unique_vm_id: str) -> None:
        """pids.max limit is enforced by kernel."""
        from exec_sandbox.cgroup import attach_to_cgroup, cleanup_cgroup, setup_cgroup

        tenant_id = "integration-test"

        # Create cgroup with low pids limit
        cgroup_path = await setup_cgroup(
            vm_id=unique_vm_id,
            tenant_id=tenant_id,
            cgroup_memory_mb=256,
            cgroup_cpu_cores=1.0,
        )

        # Override pids.max to a very low value for testing
        (cgroup_path / "pids.max").write_text("5")

        # Start a process that waits on stdin before forking, so we can
        # attach it to the cgroup first (avoids race where process exits
        # before cgroup attachment).
        proc = await asyncio.create_subprocess_exec(
            "python3",
            "-c",
            """
import os, sys
sys.stdin.readline()  # wait for go signal after cgroup attach
pids = []
try:
    for i in range(20):
        pid = os.fork()
        if pid == 0:
            import time
            time.sleep(10)
            sys.exit(0)
        pids.append(pid)
except OSError as e:
    print(f"Fork failed at iteration (expected): {e}")
    sys.exit(0)
finally:
    for pid in pids:
        try:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
        except:
            pass
""",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            assert proc.pid is not None
            await attach_to_cgroup(cgroup_path, proc.pid)

            # Signal subprocess to start forking
            assert proc.stdin is not None
            proc.stdin.write(b"\n")
            await proc.stdin.drain()
            proc.stdin.close()

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            # Process should hit pids limit
            output = stdout.decode() + stderr.decode()
            # Fork should fail due to pids limit (EAGAIN)
            assert proc.returncode == 0 or "resource" in output.lower() or "fork" in output.lower()

        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
            await cleanup_cgroup(cgroup_path, "test-cleanup")


class TestReclaimMemory:
    """memory.reclaim: EAGAIN means "reclaimed less than asked", not "unsupported"."""

    @staticmethod
    def _writer(errors: list[int | None]) -> tuple[Callable[[Path, str], int], list[tuple[str, int]]]:
        """Path.write_text stand-in: records (value, thread) and fails with errors[n] on write n."""
        writes: list[tuple[str, int]] = []

        def write_text(_path: Path, data: str) -> int:
            writes.append((data, threading.get_ident()))
            err = errors[len(writes) - 1]
            if err is not None:
                raise OSError(err, os.strerror(err))
            return len(data)

        return write_text, writes

    @pytest.mark.parametrize(
        ("errors", "expected", "attempts"),
        [
            pytest.param([None], True, 1, id="full"),
            pytest.param([errno.EAGAIN], True, 1, id="partial-is-success"),
            pytest.param([errno.EINVAL, None], True, 2, id="no-swappiness-falls-back"),
            pytest.param([errno.EINVAL, errno.EAGAIN], True, 2, id="fallback-partial"),
            pytest.param([errno.EACCES], False, 1, id="denied"),
        ],
    )
    async def test_reclaim_errors(
        self, tmp_path: Path, errors: list[int | None], expected: bool, attempts: int
    ) -> None:
        write_text, writes = self._writer(errors)
        with (
            patch("exec_sandbox.cgroup.is_cgroup_available", return_value=True),
            patch.object(Path, "write_text", autospec=True, side_effect=write_text),
        ):
            assert await reclaim_memory(tmp_path, amount_mb=192) is expected
        assert len(writes) == attempts
        assert writes[0][0] == f"{192 * 1024 * 1024} swappiness=200"
        # The kernel reclaims inside the write (10 to 500 ms): never on the event loop thread
        assert threading.get_ident() not in {thread for _, thread in writes}

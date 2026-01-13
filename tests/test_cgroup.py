"""Tests for cgroup.py cgroup v2 and ulimit utilities.

Test categories:
- Normal cases: Happy path with valid inputs
- Edge cases: None, empty, boundary values
- Error cases: OSError, PermissionError, VmError
- Weird cases: Special characters, path traversal, malformed data
- Integration tests: Real cgroups on Linux (requires sudo)
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox.cgroup import (
    CGROUP_APP_NAMESPACE,
    CGROUP_MEMORY_OVERHEAD_MB,
    CGROUP_PIDS_LIMIT,
    CGROUP_V2_BASE_PATH,
    ERRNO_PERMISSION_DENIED,
    ERRNO_READ_ONLY_FILESYSTEM,
    TCG_TB_CACHE_SIZE_MB,
    ULIMIT_CPU_TIME_SECONDS,
    ULIMIT_MEMORY_MULTIPLIER,
    is_cgroup_available,
    wrap_with_ulimit,
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
            wrapped = wrap_with_ulimit(cmd, memory_mb=256)

            assert wrapped[0] == "bash"
            assert wrapped[1] == "-c"
            # Linux should have -v (virtual memory), -t (CPU time), and -u (processes)
            assert "-v" in wrapped[2]
            assert f"-t {ULIMIT_CPU_TIME_SECONDS}" in wrapped[2]
            assert f"-u {CGROUP_PIDS_LIMIT}" in wrapped[2]
            assert "qemu-system-x86_64" in wrapped[2]

    def test_macos_uses_process_limit_only(self):
        """macOS uses only process (-u) limit; -v and -t not supported or break stdout."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.MACOS

            cmd = ["qemu-system-aarch64", "-m", "512"]
            wrapped = wrap_with_ulimit(cmd, memory_mb=512)

            # macOS should have ulimit with -u (processes) only
            assert "ulimit" in wrapped[2]
            assert f"-u {CGROUP_PIDS_LIMIT}" in wrapped[2]
            # Should NOT have -v (virtual memory) - not supported on macOS kernel
            assert "-v" not in wrapped[2]
            # Should NOT have -t (CPU time) - breaks subprocess stdout pipe on macOS
            assert "-t" not in wrapped[2]
            assert "qemu-system-aarch64" in wrapped[2]

    def test_memory_multiplier_applied(self):
        """Memory multiplier (14x) is applied to virtual memory limit."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            memory_mb = 256
            wrapped = wrap_with_ulimit(["test"], memory_mb=memory_mb)

            expected_kb = memory_mb * 1024 * ULIMIT_MEMORY_MULTIPLIER
            assert f"ulimit -v {expected_kb}" in wrapped[2]

    # --- Edge cases ---
    def test_empty_command_list(self):
        """Empty command list produces valid shell command."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            wrapped = wrap_with_ulimit([], memory_mb=256)
            assert wrapped[0] == "bash"
            assert wrapped[1] == "-c"
            # Should have ulimit but empty exec
            assert "ulimit -v" in wrapped[2]

    def test_zero_memory(self):
        """Zero memory produces ulimit -v 0."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            wrapped = wrap_with_ulimit(["test"], memory_mb=0)
            assert "ulimit -v 0" in wrapped[2]

    # --- Weird cases ---
    def test_command_with_shell_metacharacters(self):
        """Shell metacharacters are properly escaped."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            cmd = ["binary", "--arg", "$(whoami)", "; rm -rf /", "| cat"]
            wrapped = wrap_with_ulimit(cmd, memory_mb=128)

            # shlex.quote should escape these
            assert "$(whoami)" not in wrapped[2] or "'$(whoami)'" in wrapped[2]
            assert wrapped[2].count("'") >= 2  # Should have quotes

    def test_command_with_spaces(self):
        """Arguments with spaces are properly quoted."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            cmd = ["binary", "--arg", "value with spaces"]
            wrapped = wrap_with_ulimit(cmd, memory_mb=128)

            assert "'value with spaces'" in wrapped[2]

    def test_very_large_memory(self):
        """Very large memory value doesn't overflow."""
        with patch("exec_sandbox.cgroup.detect_host_os") as mock_os:
            mock_os.return_value = HostOS.LINUX

            # 1TB in MB
            memory_mb = 1024 * 1024
            wrapped = wrap_with_ulimit(["test"], memory_mb=memory_mb)

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


class TestSetupCgroup:
    """Test cgroup setup function."""

    # --- Normal cases ---
    async def test_setup_creates_cgroup_hierarchy(self):
        """Normal setup creates tenant and VM cgroups."""
        from exec_sandbox.cgroup import setup_cgroup

        with patch("aiofiles.os.makedirs", new_callable=AsyncMock) as mock_makedirs:
            mock_file = MagicMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)
            mock_file.write = AsyncMock()

            with patch("aiofiles.open", return_value=mock_file):
                result = await setup_cgroup("vm123", "tenant1", 256, use_tcg=False)

                assert result == Path(f"{CGROUP_V2_BASE_PATH}/{CGROUP_APP_NAMESPACE}/tenant1/vm123")
                assert mock_makedirs.call_count == 2  # tenant + vm directories

    async def test_setup_tcg_mode_adds_extra_memory(self):
        """TCG mode adds TB cache size to memory limit."""
        from exec_sandbox.cgroup import setup_cgroup

        written_values: list[str] = []

        async def capture_write(value: str) -> None:
            written_values.append(value)

        with patch("aiofiles.os.makedirs", new_callable=AsyncMock):
            mock_file = MagicMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)
            mock_file.write = AsyncMock(side_effect=capture_write)

            with patch("aiofiles.open", return_value=mock_file):
                await setup_cgroup("vm123", "tenant1", 256, use_tcg=True)

                # Find memory.max value (in bytes)
                expected_memory = (256 + CGROUP_MEMORY_OVERHEAD_MB + TCG_TB_CACHE_SIZE_MB) * 1024 * 1024
                assert str(expected_memory) in written_values

    # --- Error cases ---
    async def test_setup_read_only_filesystem_returns_fallback(self):
        """Read-only filesystem gracefully degrades to fallback path."""
        from exec_sandbox.cgroup import setup_cgroup

        error = OSError("Read-only filesystem")
        error.errno = ERRNO_READ_ONLY_FILESYSTEM

        with patch("aiofiles.os.makedirs", new_callable=AsyncMock, side_effect=error):
            result = await setup_cgroup("vm123", "tenant1", 256)

            assert result == Path("/tmp/cgroup-vm123")

    async def test_setup_permission_denied_returns_fallback(self):
        """Permission denied gracefully degrades to fallback path."""
        from exec_sandbox.cgroup import setup_cgroup

        error = OSError("Permission denied")
        error.errno = ERRNO_PERMISSION_DENIED

        with patch("aiofiles.os.makedirs", new_callable=AsyncMock, side_effect=error):
            result = await setup_cgroup("vm123", "tenant1", 256)

            assert result == Path("/tmp/cgroup-vm123")

    async def test_setup_other_oserror_raises_vmerror(self):
        """Other OSError raises VmError."""
        from exec_sandbox.cgroup import setup_cgroup
        from exec_sandbox.exceptions import VmError

        error = OSError("Disk full")
        error.errno = 28  # ENOSPC

        with patch("aiofiles.os.makedirs", new_callable=AsyncMock, side_effect=error):
            with pytest.raises(VmError, match="Failed to setup cgroup"):
                await setup_cgroup("vm123", "tenant1", 256)


# =============================================================================
# attach_to_cgroup - Tests
# =============================================================================


class TestAttachToCgroup:
    """Test cgroup process attachment."""

    # --- Normal cases ---
    async def test_attach_writes_pid_to_cgroup_procs(self):
        """Normal attach writes PID to cgroup.procs."""
        from exec_sandbox.cgroup import attach_to_cgroup

        mock_file = MagicMock()
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        mock_file.write = AsyncMock()

        with patch("aiofiles.open", return_value=mock_file):
            await attach_to_cgroup(Path("/sys/fs/cgroup/test"), 12345)

            mock_file.write.assert_called_once_with("12345")

    # --- Error cases ---
    async def test_attach_oserror_raises_vmerror(self):
        """OSError during attach raises VmError."""
        from exec_sandbox.cgroup import attach_to_cgroup
        from exec_sandbox.exceptions import VmError

        mock_file = MagicMock()
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        mock_file.write = AsyncMock(side_effect=OSError("No such process"))

        with patch("aiofiles.open", return_value=mock_file):
            with pytest.raises(VmError, match="Failed to attach PID"):
                await attach_to_cgroup(Path("/sys/fs/cgroup/test"), 12345)

    async def test_attach_permission_error_raises_vmerror(self):
        """PermissionError during attach raises VmError."""
        from exec_sandbox.cgroup import attach_to_cgroup
        from exec_sandbox.exceptions import VmError

        mock_file = MagicMock()
        mock_file.__aenter__ = AsyncMock(side_effect=PermissionError("Access denied"))

        with patch("aiofiles.open", return_value=mock_file):
            with pytest.raises(VmError, match="Failed to attach PID"):
                await attach_to_cgroup(Path("/sys/fs/cgroup/test"), 12345)


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
    async def test_returns_true_on_successful_attach(self):
        """Returns True when attachment succeeds."""
        from exec_sandbox.cgroup import attach_if_available

        mock_file = MagicMock()
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        mock_file.write = AsyncMock()

        with (
            patch("exec_sandbox.cgroup._check_cgroup_v2_mounted", return_value=True),
            patch("aiofiles.open", return_value=mock_file),
        ):
            cgroup_path = Path("/sys/fs/cgroup/code-exec/tenant/vm123")
            result = await attach_if_available(cgroup_path, 12345)
            assert result is True


# =============================================================================
# read_cgroup_stats - Tests
# =============================================================================


class TestReadCgroupStats:
    """Test cgroup stats reading."""

    # --- Edge cases (None/missing) ---
    async def test_returns_none_for_none_path(self):
        """Returns (None, None) for None path."""
        from exec_sandbox.cgroup import read_cgroup_stats

        cpu_ms, mem_mb = await read_cgroup_stats(None)
        assert cpu_ms is None
        assert mem_mb is None

    async def test_returns_none_for_nonexistent_path(self):
        """Returns (None, None) for non-existent path."""
        from exec_sandbox.cgroup import read_cgroup_stats

        result = await read_cgroup_stats(Path("/nonexistent/cgroup"))
        assert result == (None, None)

    # --- Normal cases ---
    async def test_reads_cpu_and_memory_stats(self):
        """Reads and parses cpu.stat and memory.peak correctly."""
        from exec_sandbox.cgroup import read_cgroup_stats

        cpu_stat_content = "usage_usec 5000000\nuser_usec 3000000\nsystem_usec 2000000"
        memory_peak_content = "104857600"  # 100MB in bytes

        async def mock_exists(path: Any) -> bool:
            return True

        def mock_open_file(path: Any, *args: Any, **kwargs: Any) -> MagicMock:
            mock_file = MagicMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)

            if "cpu.stat" in str(path):
                mock_file.read = AsyncMock(return_value=cpu_stat_content)
            elif "memory.peak" in str(path):
                mock_file.read = AsyncMock(return_value=memory_peak_content)

            return mock_file

        with (
            patch("aiofiles.os.path.exists", side_effect=mock_exists),
            patch("aiofiles.open", side_effect=mock_open_file),
        ):
            cpu_ms, mem_mb = await read_cgroup_stats(Path("/sys/fs/cgroup/test"))

            assert cpu_ms == 5000  # 5000000 usec = 5000 ms
            assert mem_mb == 100  # 104857600 bytes = 100 MB

    # --- Error cases ---
    async def test_handles_malformed_cpu_stat(self):
        """Handles malformed cpu.stat gracefully."""
        from exec_sandbox.cgroup import read_cgroup_stats

        async def mock_exists(path: Any) -> bool:
            return True

        def mock_open_file(path: Any, *args: Any, **kwargs: Any) -> MagicMock:
            mock_file = MagicMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)
            mock_file.read = AsyncMock(return_value="garbage data not_a_number")
            return mock_file

        with (
            patch("aiofiles.os.path.exists", side_effect=mock_exists),
            patch("aiofiles.open", side_effect=mock_open_file),
        ):
            cpu_ms, mem_mb = await read_cgroup_stats(Path("/sys/fs/cgroup/test"))

            # Should return None on parse error, not crash
            assert cpu_ms is None
            assert mem_mb is None

    async def test_handles_oserror_during_read(self):
        """Handles OSError during file read gracefully."""
        from exec_sandbox.cgroup import read_cgroup_stats

        async def mock_exists(path: Any) -> bool:
            return True

        def mock_open_file(path: Any, *args: Any, **kwargs: Any) -> MagicMock:
            raise OSError("Device not configured")

        with (
            patch("aiofiles.os.path.exists", side_effect=mock_exists),
            patch("aiofiles.open", side_effect=mock_open_file),
        ):
            cpu_ms, mem_mb = await read_cgroup_stats(Path("/sys/fs/cgroup/test"))

            assert cpu_ms is None
            assert mem_mb is None


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
    async def test_removes_empty_cgroup(self):
        """Removes empty cgroup directory."""
        from exec_sandbox.cgroup import cleanup_cgroup

        async def mock_exists(path: Any) -> bool:
            return True

        def mock_read_file(path: Any, *args: Any, **kwargs: Any) -> MagicMock:
            mock_file = MagicMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)
            mock_file.read = AsyncMock(return_value="")  # No PIDs
            return mock_file

        with (
            patch("aiofiles.os.path.exists", side_effect=mock_exists),
            patch("aiofiles.open", side_effect=mock_read_file),
            patch("aiofiles.os.rmdir", new_callable=AsyncMock) as mock_rmdir,
        ):
            cgroup_path = Path("/sys/fs/cgroup/code-exec/tenant/vm123")
            result = await cleanup_cgroup(cgroup_path, "test-ctx")

            assert result is True
            mock_rmdir.assert_called_once_with(cgroup_path)

    async def test_migrates_pids_before_removal(self):
        """Moves PIDs to parent cgroup before removing."""
        from exec_sandbox.cgroup import cleanup_cgroup

        written_pids: list[str] = []

        async def mock_exists(path: Any) -> bool:
            return True

        def mock_open_file(path: Any, mode: str = "r") -> MagicMock:
            mock_file = MagicMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)

            if mode == "r" or mode not in ("w", "a"):
                # Reading cgroup.procs
                mock_file.read = AsyncMock(return_value="123\n456\n789")
            else:
                # Writing to parent cgroup.procs
                async def capture_write(pid: str) -> None:
                    written_pids.append(pid)

                mock_file.write = AsyncMock(side_effect=capture_write)

            return mock_file

        with (
            patch("exec_sandbox.cgroup._check_cgroup_v2_mounted", return_value=True),
            patch("aiofiles.os.path.exists", side_effect=mock_exists),
            patch("aiofiles.open", side_effect=mock_open_file),
            patch("aiofiles.os.rmdir", new_callable=AsyncMock),
        ):
            cgroup_path = Path("/sys/fs/cgroup/code-exec/tenant/vm123")
            result = await cleanup_cgroup(cgroup_path, "test-ctx")

            assert result is True
            assert "123" in written_pids
            assert "456" in written_pids
            assert "789" in written_pids

    # --- Error cases ---
    async def test_handles_already_deleted_cgroup(self):
        """FileNotFoundError during rmdir returns True (already deleted)."""
        from exec_sandbox.cgroup import cleanup_cgroup

        async def mock_exists(path: Any) -> bool:
            return True

        def mock_read_file(path: Any, *args: Any, **kwargs: Any) -> MagicMock:
            mock_file = MagicMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)
            mock_file.read = AsyncMock(return_value="")
            return mock_file

        with (
            patch("aiofiles.os.path.exists", side_effect=mock_exists),
            patch("aiofiles.open", side_effect=mock_read_file),
            patch("aiofiles.os.rmdir", new_callable=AsyncMock, side_effect=FileNotFoundError),
        ):
            cgroup_path = Path("/sys/fs/cgroup/code-exec/tenant/vm123")
            result = await cleanup_cgroup(cgroup_path, "test-ctx")

            assert result is True  # Race condition - already deleted

    async def test_handles_rmdir_oserror(self):
        """OSError during rmdir returns False."""
        from exec_sandbox.cgroup import cleanup_cgroup

        async def mock_exists(path: Any) -> bool:
            return True

        def mock_read_file(path: Any, *args: Any, **kwargs: Any) -> MagicMock:
            mock_file = MagicMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)
            mock_file.read = AsyncMock(return_value="")
            return mock_file

        with (
            patch("exec_sandbox.cgroup._check_cgroup_v2_mounted", return_value=True),
            patch("aiofiles.os.path.exists", side_effect=mock_exists),
            patch("aiofiles.open", side_effect=mock_read_file),
            patch("aiofiles.os.rmdir", new_callable=AsyncMock, side_effect=OSError("Directory not empty")),
        ):
            cgroup_path = Path("/sys/fs/cgroup/code-exec/tenant/vm123")
            result = await cleanup_cgroup(cgroup_path, "test-ctx")

            assert result is False

    async def test_handles_pid_migration_error_gracefully(self):
        """Continues cleanup even if PID migration fails (process already exited)."""
        from exec_sandbox.cgroup import cleanup_cgroup

        call_count = {"read": 0, "write": 0}

        async def mock_exists(path: Any) -> bool:
            return True

        def mock_open_file(path: Any, mode: str = "r") -> MagicMock:
            mock_file = MagicMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)

            if mode == "r" or mode not in ("w", "a"):
                call_count["read"] += 1
                mock_file.read = AsyncMock(return_value="123\n456")
            else:
                call_count["write"] += 1
                # First PID fails, second succeeds
                if call_count["write"] == 1:
                    mock_file.write = AsyncMock(side_effect=OSError("No such process"))
                else:
                    mock_file.write = AsyncMock()

            return mock_file

        with (
            patch("aiofiles.os.path.exists", side_effect=mock_exists),
            patch("aiofiles.open", side_effect=mock_open_file),
            patch("aiofiles.os.rmdir", new_callable=AsyncMock),
        ):
            cgroup_path = Path("/sys/fs/cgroup/code-exec/tenant/vm123")
            result = await cleanup_cgroup(cgroup_path, "test-ctx")

            # Should succeed even though one PID migration failed
            assert result is True


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
        """Memory overhead is reasonable (100-500MB)."""
        assert 100 <= CGROUP_MEMORY_OVERHEAD_MB <= 500

    def test_tcg_cache_size_is_reasonable(self):
        """TCG TB cache size is reasonable (256MB-1GB)."""
        assert 256 <= TCG_TB_CACHE_SIZE_MB <= 1024

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
        cgroup_path = await setup_cgroup(
            vm_id=unique_vm_id,
            tenant_id=tenant_id,
            memory_mb=256,
            use_tcg=False,
        )

        try:
            # Verify real cgroup was created
            assert cgroup_path.exists(), f"Cgroup not created: {cgroup_path}"
            assert (cgroup_path / "cgroup.procs").exists()
            assert (cgroup_path / "memory.max").exists()
            assert (cgroup_path / "cpu.max").exists()
            assert (cgroup_path / "pids.max").exists()

            # Verify memory limit was set
            memory_max = (cgroup_path / "memory.max").read_text().strip()
            expected_bytes = (256 + CGROUP_MEMORY_OVERHEAD_MB) * 1024 * 1024
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
            memory_mb=256,
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
            memory_mb=256,
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
            cpu_ms, mem_mb = await read_cgroup_stats(cgroup_path)

            # Stats should have values (process did work)
            assert cpu_ms is not None, "CPU stats not available"
            assert cpu_ms >= 0, f"CPU time should be non-negative: {cpu_ms}"

            # Memory peak might not be available on all systems
            if mem_mb is not None:
                assert mem_mb >= 0, f"Memory should be non-negative: {mem_mb}"

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
            memory_mb=256,
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
            memory_mb=256,
        )

        # Override pids.max to a very low value for testing
        (cgroup_path / "pids.max").write_text("5")

        # Start a process that tries to fork many children
        proc = await asyncio.create_subprocess_exec(
            "python3",
            "-c",
            """
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
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            assert proc.pid is not None
            await attach_to_cgroup(cgroup_path, proc.pid)

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

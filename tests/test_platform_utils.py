"""Unit tests for platform utilities.

Tests ProcessWrapper and OS detection with real processes.
No mocks - spawns actual processes.
"""

import asyncio
import platform
import sys

import pytest

from exec_sandbox.platform_utils import (
    HostArch,
    HostOS,
    ProcessWrapper,
    detect_host_arch,
    detect_host_os,
    get_arch_name,
    get_os_name,
)

# ============================================================================
# OS Detection
# ============================================================================


class TestDetectHostOS:
    """Tests for detect_host_os function."""

    def test_detect_current_platform(self) -> None:
        """detect_host_os returns valid HostOS for current platform."""
        result = detect_host_os()
        assert isinstance(result, HostOS)
        assert result in (HostOS.LINUX, HostOS.MACOS, HostOS.UNKNOWN)

    def test_detect_matches_sys_platform(self) -> None:
        """detect_host_os matches sys.platform."""
        result = detect_host_os()
        if sys.platform == "darwin":
            assert result == HostOS.MACOS
        elif sys.platform.startswith("linux"):
            assert result == HostOS.LINUX

    def test_detect_is_cached(self) -> None:
        """detect_host_os is cached (same instance returned)."""
        result1 = detect_host_os()
        result2 = detect_host_os()
        # Same enum value (cached via @cache)
        assert result1 is result2


class TestHostOSEnum:
    """Tests for HostOS enum."""

    def test_enum_values(self) -> None:
        """HostOS has expected values."""
        assert HostOS.LINUX is not None
        assert HostOS.MACOS is not None
        assert HostOS.UNKNOWN is not None

    def test_enum_distinct(self) -> None:
        """HostOS values are distinct."""
        assert HostOS.LINUX != HostOS.MACOS
        assert HostOS.LINUX != HostOS.UNKNOWN
        assert HostOS.MACOS != HostOS.UNKNOWN


# ============================================================================
# Architecture Detection
# ============================================================================


class TestDetectHostArch:
    """Tests for detect_host_arch function."""

    def test_detect_current_architecture(self) -> None:
        """detect_host_arch returns valid HostArch for current platform."""
        result = detect_host_arch()
        assert isinstance(result, HostArch)
        assert result in (HostArch.X86_64, HostArch.AARCH64, HostArch.UNKNOWN)

    def test_detect_matches_platform_machine(self) -> None:
        """detect_host_arch matches platform.machine()."""
        result = detect_host_arch()
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            assert result == HostArch.X86_64
        elif machine in ("arm64", "aarch64"):
            assert result == HostArch.AARCH64

    def test_detect_is_cached(self) -> None:
        """detect_host_arch is cached (same instance returned)."""
        result1 = detect_host_arch()
        result2 = detect_host_arch()
        # Same enum value (cached via @cache)
        assert result1 is result2


class TestHostArchEnum:
    """Tests for HostArch enum."""

    def test_enum_values(self) -> None:
        """HostArch has expected values."""
        assert HostArch.X86_64 is not None
        assert HostArch.AARCH64 is not None
        assert HostArch.UNKNOWN is not None

    def test_enum_distinct(self) -> None:
        """HostArch values are distinct."""
        assert HostArch.X86_64 != HostArch.AARCH64
        assert HostArch.X86_64 != HostArch.UNKNOWN
        assert HostArch.AARCH64 != HostArch.UNKNOWN


class TestGetOsName:
    """Tests for get_os_name helper function."""

    def test_returns_string(self) -> None:
        """get_os_name returns a string."""
        result = get_os_name()
        assert isinstance(result, str)

    def test_returns_valid_os_name(self) -> None:
        """get_os_name returns valid OS name for current platform."""
        host_os = detect_host_os()
        if host_os == HostOS.MACOS:
            assert get_os_name() == "darwin"
        elif host_os == HostOS.LINUX:
            assert get_os_name() == "linux"


class TestGetArchName:
    """Tests for get_arch_name helper function."""

    def test_kernel_convention(self) -> None:
        """get_arch_name returns kernel-style names by default."""
        result = get_arch_name("kernel")
        host_arch = detect_host_arch()
        if host_arch == HostArch.X86_64:
            assert result == "x86_64"
        elif host_arch == HostArch.AARCH64:
            assert result == "aarch64"

    def test_go_convention(self) -> None:
        """get_arch_name returns Go-style names with convention='go'."""
        result = get_arch_name("go")
        host_arch = detect_host_arch()
        if host_arch == HostArch.X86_64:
            assert result == "amd64"
        elif host_arch == HostArch.AARCH64:
            assert result == "arm64"

    def test_default_convention_is_kernel(self) -> None:
        """get_arch_name defaults to kernel convention."""
        assert get_arch_name() == get_arch_name("kernel")


# ============================================================================
# ProcessWrapper
# ============================================================================


async def create_wrapped_process(cmd: list[str], **kwargs) -> ProcessWrapper:
    """Helper to create a wrapped subprocess."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    return ProcessWrapper(proc)


class TestProcessWrapperInit:
    """Tests for ProcessWrapper initialization."""

    async def test_wrap_simple_process(self) -> None:
        """ProcessWrapper wraps a simple process."""
        proc = await create_wrapped_process(["echo", "hello"])
        assert proc.pid is not None
        assert proc.pid > 0
        await proc.wait()

    async def test_psutil_proc_created(self) -> None:
        """ProcessWrapper creates psutil.Process for monitoring."""
        proc = await create_wrapped_process(["sleep", "0.1"])
        # psutil_proc should be set for running process
        assert proc.psutil_proc is not None
        await proc.wait()

    async def test_wrap_finished_process(self) -> None:
        """ProcessWrapper handles process that finishes quickly."""
        proc = await create_wrapped_process(["true"])
        await proc.wait()
        # Should have a return code
        assert proc.returncode == 0


class TestProcessWrapperProperties:
    """Tests for ProcessWrapper properties."""

    async def test_pid_property(self) -> None:
        """pid property returns process ID."""
        proc = await create_wrapped_process(["echo", "test"])
        assert isinstance(proc.pid, int)
        assert proc.pid > 0
        await proc.wait()

    async def test_returncode_none_while_running(self) -> None:
        """returncode is None while process is running."""
        proc = await create_wrapped_process(["sleep", "0.5"])
        # Check immediately - should still be running
        assert proc.returncode is None
        await proc.wait()

    async def test_returncode_after_exit(self) -> None:
        """returncode is set after process exits."""
        proc = await create_wrapped_process(["true"])
        await proc.wait()
        assert proc.returncode == 0

        proc = await create_wrapped_process(["false"])
        await proc.wait()
        assert proc.returncode == 1

    async def test_stdout_property(self) -> None:
        """stdout property returns stream."""
        proc = await create_wrapped_process(["echo", "test"])
        assert proc.stdout is not None
        await proc.wait()

    async def test_stderr_property(self) -> None:
        """stderr property returns stream."""
        proc = await create_wrapped_process(["bash", "-c", "echo err >&2"])
        assert proc.stderr is not None
        await proc.wait()


class TestProcessWrapperIsRunning:
    """Tests for ProcessWrapper.is_running method."""

    async def test_is_running_true(self) -> None:
        """is_running returns True for running process."""
        proc = await create_wrapped_process(["sleep", "1"])
        result = await proc.is_running()
        assert result is True
        # Clean up
        await proc.terminate()
        await proc.wait()

    async def test_is_running_false_after_exit(self) -> None:
        """is_running returns False after process exits."""
        proc = await create_wrapped_process(["true"])
        await proc.wait()
        result = await proc.is_running()
        assert result is False

    async def test_is_running_async(self) -> None:
        """is_running doesn't block the event loop."""
        proc = await create_wrapped_process(["sleep", "0.1"])

        # Run is_running concurrently with other tasks
        async def other_task() -> str:
            await asyncio.sleep(0.01)
            return "done"

        results = await asyncio.gather(
            proc.is_running(),
            other_task(),
        )

        assert results[0] is True
        assert results[1] == "done"

        await proc.wait()


class TestProcessWrapperWait:
    """Tests for ProcessWrapper.wait method."""

    async def test_wait_success(self) -> None:
        """wait returns exit code on success."""
        proc = await create_wrapped_process(["true"])
        code = await proc.wait()
        assert code == 0

    async def test_wait_failure(self) -> None:
        """wait returns exit code on failure."""
        proc = await create_wrapped_process(["false"])
        code = await proc.wait()
        assert code == 1

    async def test_wait_custom_exit_code(self) -> None:
        """wait returns custom exit code."""
        proc = await create_wrapped_process(["bash", "-c", "exit 42"])
        code = await proc.wait()
        assert code == 42


class TestProcessWrapperTerminate:
    """Tests for ProcessWrapper.terminate method."""

    async def test_terminate_running_process(self) -> None:
        """terminate stops a running process."""
        proc = await create_wrapped_process(["sleep", "10"])

        # Verify it's running
        assert await proc.is_running()

        # Terminate
        await proc.terminate()

        # Wait for it to stop
        await asyncio.sleep(0.1)
        code = await proc.wait()

        # SIGTERM typically results in -15 or 143
        assert proc.returncode is not None
        assert not await proc.is_running()

    async def test_terminate_already_finished(self) -> None:
        """terminate on finished process raises ProcessLookupError."""
        proc = await create_wrapped_process(["true"])
        await proc.wait()

        # asyncio.subprocess raises ProcessLookupError for dead process
        with pytest.raises(ProcessLookupError):
            await proc.terminate()
        assert proc.returncode == 0


class TestProcessWrapperKill:
    """Tests for ProcessWrapper.kill method."""

    async def test_kill_running_process(self) -> None:
        """kill forcefully stops a running process."""
        proc = await create_wrapped_process(["sleep", "10"])

        assert await proc.is_running()

        await proc.kill()
        await proc.wait()

        # SIGKILL results in -9 or 137
        assert proc.returncode is not None
        assert not await proc.is_running()

    async def test_kill_already_finished(self) -> None:
        """kill on finished process raises ProcessLookupError."""
        proc = await create_wrapped_process(["true"])
        await proc.wait()

        # asyncio.subprocess raises ProcessLookupError for dead process
        with pytest.raises(ProcessLookupError):
            await proc.kill()
        assert proc.returncode == 0


class TestProcessWrapperCommunicate:
    """Tests for ProcessWrapper.communicate method."""

    async def test_communicate_captures_stdout(self) -> None:
        """communicate returns stdout."""
        proc = await create_wrapped_process(["echo", "hello"])
        stdout, stderr = await proc.communicate()
        assert b"hello" in stdout
        assert stderr == b""

    async def test_communicate_captures_stderr(self) -> None:
        """communicate returns stderr."""
        proc = await create_wrapped_process(["bash", "-c", "echo error >&2"])
        stdout, stderr = await proc.communicate()
        assert stdout == b""
        assert b"error" in stderr

    async def test_communicate_captures_both(self) -> None:
        """communicate returns both stdout and stderr."""
        proc = await create_wrapped_process(["bash", "-c", "echo out; echo err >&2"])
        stdout, stderr = await proc.communicate()
        assert b"out" in stdout
        assert b"err" in stderr


class TestProcessWrapperWaitWithTimeout:
    """Tests for ProcessWrapper.wait_with_timeout method."""

    async def test_wait_with_timeout_completes(self) -> None:
        """wait_with_timeout returns when process completes."""
        proc = await create_wrapped_process(["echo", "fast"])
        code = await proc.wait_with_timeout(5.0)
        assert code == 0

    async def test_wait_with_timeout_times_out(self) -> None:
        """wait_with_timeout raises TimeoutError on timeout."""
        proc = await create_wrapped_process(["sleep", "10"])

        with pytest.raises(asyncio.TimeoutError):
            await proc.wait_with_timeout(0.1)

        # Clean up
        await proc.kill()
        await proc.wait()

    async def test_wait_with_timeout_drains_pipes(self) -> None:
        """wait_with_timeout drains stdout/stderr pipes."""
        proc = await create_wrapped_process(["bash", "-c", "echo output; sleep 0.1; echo done"])

        code = await proc.wait_with_timeout(5.0)
        assert code == 0

    async def test_wait_with_timeout_no_pipes(self) -> None:
        """wait_with_timeout works without pipes."""
        # Create process without pipes
        async_proc = await asyncio.create_subprocess_exec(
            "true",
            stdout=None,
            stderr=None,
        )
        proc = ProcessWrapper(async_proc)

        code = await proc.wait_with_timeout(5.0)
        assert code == 0

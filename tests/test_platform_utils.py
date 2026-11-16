"""Unit tests for platform utilities.

Tests OS and architecture detection, the per-user folders, and ProcessWrapper.
ProcessWrapper tests spawn real processes; a stand-in replaces psutil, the host OS or
the machine name only for a case that a real process or host cannot show.
"""

import asyncio
import os
import pwd
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

from exec_sandbox.platform_utils import (
    HostArch,
    HostOS,
    ProcessWrapper,
    UserDir,
    detect_host_arch,
    detect_host_os,
    detect_host_x86_vendor,
    get_account_dir,
    get_arch_name,
    get_os_name,
    get_user_dir,
)
from exec_sandbox.subprocess_utils import start_drain
from tests.conftest import (
    skip_unless_aarch64,
    skip_unless_linux,
    skip_unless_macos,
    skip_unless_x86_64,
)

# ============================================================================
# OS Detection
# ============================================================================


class TestDetectHostOS:
    """Tests for detect_host_os function."""

    def test_detect_matches_sys_platform(self) -> None:
        """detect_host_os (from psutil) matches sys.platform (the tests run on macOS or Linux)."""
        assert detect_host_os() is {"darwin": HostOS.MACOS, "linux": HostOS.LINUX}[sys.platform]


# ============================================================================
# Architecture Detection
# ============================================================================


class TestDetectHostArch:
    """Tests for detect_host_arch function."""

    @pytest.fixture(autouse=True)
    def fresh_cache(self) -> Iterator[None]:
        """detect_host_arch keeps its first answer: each test starts and ends with none kept."""
        detect_host_arch.cache_clear()
        yield
        detect_host_arch.cache_clear()

    @pytest.mark.parametrize(
        ("machine", "expected"),
        [
            ("x86_64", HostArch.X86_64),
            ("AMD64", HostArch.X86_64),  # Windows spelling, upper case
            ("arm64", HostArch.AARCH64),  # macOS spelling
            ("aarch64", HostArch.AARCH64),  # Linux spelling
            ("riscv64", HostArch.UNKNOWN),
            ("", HostArch.UNKNOWN),
        ],
    )
    def test_machine_names(self, monkeypatch: pytest.MonkeyPatch, machine: str, expected: HostArch) -> None:
        """Each spelling of platform.machine() gives its architecture."""
        monkeypatch.setattr("exec_sandbox.platform_utils.platform.machine", lambda: machine)
        assert detect_host_arch() is expected


class TestGetOsName:
    """Tests for get_os_name helper function."""

    @skip_unless_macos
    def test_returns_darwin_on_macos(self) -> None:
        """get_os_name returns 'darwin' on macOS."""
        assert get_os_name() == "darwin"

    @skip_unless_linux
    def test_returns_linux_on_linux(self) -> None:
        """get_os_name returns 'linux' on Linux."""
        assert get_os_name() == "linux"


class TestDetectHostX86Vendor:
    """detect_host_x86_vendor on Linux x86: the vendor_id line of /proc/cpuinfo, as the
    kernel prints it ("vendor_id\\t: %s" of the 12 CPUID bytes)."""

    @pytest.fixture(autouse=True)
    def linux_x86(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """A Linux x86 host; each test starts and ends with no answer kept."""
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_arch", lambda: HostArch.X86_64)
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: HostOS.LINUX)
        detect_host_x86_vendor.cache_clear()
        yield
        detect_host_x86_vendor.cache_clear()

    @pytest.mark.parametrize(
        "vendor",
        ["GenuineIntel", "AuthenticAMD", "HygonGenuine", "CentaurHauls", "  Shanghai  "],
        ids=["intel", "amd", "hygon", "centaur", "zhaoxin"],
    )
    def test_the_12_characters_spaces_kept(self, vendor: str) -> None:
        cpuinfo = f"processor\t: 0\nvendor_id\t: {vendor}\ncpu family\t: 7\n"
        with patch.object(Path, "read_text", return_value=cpuinfo):
            assert detect_host_x86_vendor() == vendor


class TestGetArchName:
    """Tests for get_arch_name helper function."""

    @skip_unless_x86_64
    def test_kernel_name_x86_64(self) -> None:
        """get_arch_name returns 'x86_64' on x86_64."""
        assert get_arch_name() == "x86_64"

    @skip_unless_aarch64
    def test_kernel_name_aarch64(self) -> None:
        """get_arch_name returns 'aarch64' on ARM64."""
        assert get_arch_name() == "aarch64"


# ============================================================================
# get_user_dir(kind=CACHE)
# ============================================================================


class TestUserDirCache:
    """Tests for get_user_dir(kind=UserDir.CACHE)."""

    # --- Normal cases ---

    @pytest.mark.parametrize(
        ("host_os", "expected"),
        [
            pytest.param(HostOS.MACOS, "/home/u/Library/Caches/exec-sandbox", id="macos-library-caches"),
            pytest.param(HostOS.LINUX, "/home/u/.cache/exec-sandbox", id="linux-dot-cache"),
        ],
    )
    def test_default_per_os(self, monkeypatch: pytest.MonkeyPatch, host_os: HostOS, expected: str) -> None:
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: host_os)
        monkeypatch.setenv("HOME", "/home/u")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("EXEC_SANDBOX_CACHE_DIR", raising=False)
        assert get_user_dir(kind=UserDir.CACHE) == Path(expected)

    # --- Env var overrides ---

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param("/custom", "/custom", id="absolute"),
            pytest.param("/path with spaces/cache", "/path with spaces/cache", id="spaces"),
            pytest.param("/données/caché", "/données/caché", id="unicode"),
            pytest.param("/custom/path/", "/custom/path", id="trailing-slash"),
            pytest.param("relative/path", "relative/path", id="relative-as-is"),
        ],
    )
    @pytest.mark.parametrize("host_os", [HostOS.MACOS, HostOS.LINUX])
    def test_exec_sandbox_cache_dir_override(
        self, monkeypatch: pytest.MonkeyPatch, host_os: HostOS, value: str, expected: str
    ) -> None:
        """EXEC_SANDBOX_CACHE_DIR is the cache folder on every OS, also over XDG_CACHE_HOME."""
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: host_os)
        monkeypatch.setenv("EXEC_SANDBOX_CACHE_DIR", value)
        monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
        assert get_user_dir(kind=UserDir.CACHE) == Path(expected)

    def test_xdg_cache_home_on_linux(self) -> None:
        """XDG_CACHE_HOME on Linux → /xdg/exec-sandbox."""
        with patch("exec_sandbox.platform_utils.detect_host_os", return_value=HostOS.LINUX):
            with patch.dict("os.environ", {"XDG_CACHE_HOME": "/xdg"}, clear=True):
                result = get_user_dir(kind=UserDir.CACHE)
                assert result == Path("/xdg/exec-sandbox")

    def test_xdg_cache_home_ignored_on_macos(self) -> None:
        """XDG_CACHE_HOME is ignored on macOS."""
        with patch("exec_sandbox.platform_utils.detect_host_os", return_value=HostOS.MACOS):
            with patch.dict("os.environ", {"XDG_CACHE_HOME": "/xdg", "HOME": "/home/u"}, clear=True):
                result = get_user_dir(kind=UserDir.CACHE)
                assert result == Path("/home/u/Library/Caches/exec-sandbox")

    # --- Edge cases ---

    def test_empty_xdg_cache_home_ignored(self) -> None:
        """An empty XDG_CACHE_HOME is no absolute path (XDG spec): ~/.cache."""
        with patch("exec_sandbox.platform_utils.detect_host_os", return_value=HostOS.LINUX):
            with patch.dict("os.environ", {"XDG_CACHE_HOME": "", "HOME": "/home/u"}, clear=True):
                result = get_user_dir(kind=UserDir.CACHE)
                assert result == Path("/home/u/.cache/exec-sandbox")


# ============================================================================
# get_user_dir(kind=DATA)
# ============================================================================


class TestUserDirData:
    """Tests for get_user_dir(kind=UserDir.DATA)."""

    def test_macos_application_support(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: HostOS.MACOS)
        monkeypatch.setenv("HOME", "/home/u")
        monkeypatch.setenv("XDG_DATA_HOME", "/xdg")  # not used on macOS
        assert get_user_dir(kind=UserDir.DATA) == Path("/home/u/Library/Application Support/exec-sandbox")

    def test_linux_absolute_xdg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setenv("HOME", "/home/u")
        monkeypatch.setenv("XDG_DATA_HOME", "/xdg/data")
        assert get_user_dir(kind=UserDir.DATA) == Path("/xdg/data/exec-sandbox")

    @pytest.mark.parametrize("xdg", ["relative/data", "./data", ""])
    def test_linux_relative_or_empty_xdg_ignored(self, monkeypatch: pytest.MonkeyPatch, xdg: str) -> None:
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setenv("HOME", "/home/u")
        monkeypatch.setenv("XDG_DATA_HOME", xdg)
        assert get_user_dir(kind=UserDir.DATA) == Path("/home/u/.local/share/exec-sandbox")

    def test_without_xdg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setenv("HOME", "/home/u")
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        assert get_user_dir(kind=UserDir.DATA) == Path("/home/u/.local/share/exec-sandbox")

    def test_cache_override_not_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EXEC_SANDBOX_CACHE_DIR moves caches, never data the library cannot make again."""
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: HostOS.LINUX)
        monkeypatch.setenv("HOME", "/home/u")
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("EXEC_SANDBOX_CACHE_DIR", "/cache")
        assert get_user_dir(kind=UserDir.DATA) == Path("/home/u/.local/share/exec-sandbox")
        assert get_user_dir(kind=UserDir.CACHE) == Path("/cache")

    @pytest.mark.parametrize(
        ("host_os", "data", "cache"),
        [
            (
                HostOS.MACOS,
                "/accounts/me/Library/Application Support/exec-sandbox",
                "/accounts/me/Library/Caches/exec-sandbox",
            ),
            (HostOS.LINUX, "/accounts/me/.local/share/exec-sandbox", "/accounts/me/.cache/exec-sandbox"),
        ],
        ids=["macos", "linux"],
    )
    def test_account_dir_ignores_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, host_os: HostOS, data: str, cache: str
    ) -> None:
        """get_account_dir: the home of the effective user in the user database; HOME, XDG_* and
        EXEC_SANDBOX_CACHE_DIR (absolute paths, so get_user_dir would take them) change nothing."""
        real_getpwuid = pwd.getpwuid

        def getpwuid(uid: int) -> pwd.struct_passwd:
            entry = real_getpwuid(os.getuid())
            return pwd.struct_passwd((*entry[:5], "/accounts/me" if uid == 4242 else "/accounts/other", entry.pw_shell))

        monkeypatch.setattr(pwd, "getpwuid", getpwuid)
        monkeypatch.setattr(os, "geteuid", lambda: 4242)
        monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: host_os)
        for name, value in [("HOME", "/env/home"), ("XDG_DATA_HOME", "/env/xdg-data"),
                            ("XDG_CACHE_HOME", "/env/xdg-cache"), ("EXEC_SANDBOX_CACHE_DIR", "/env/cache")]:  # fmt: skip
            monkeypatch.setenv(name, value)
        assert get_account_dir(kind=UserDir.DATA) == Path(data)
        assert get_account_dir(kind=UserDir.CACHE) == Path(cache)


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

    async def test_pid_names_the_spawned_process(self) -> None:
        """pid and psutil_proc name the process that asyncio spawned."""
        proc = await create_wrapped_process(["sleep", "10"])
        try:
            assert psutil.Process(proc.pid).name() == "sleep"
            assert proc.psutil_proc is not None
            assert proc.psutil_proc.pid == proc.pid
        finally:
            await proc.kill()
            await proc.wait()

    async def test_a_reaped_process_is_not_looked_up(self) -> None:
        """Once asyncio reaped the child, its pid may belong to another process: no psutil lookup."""
        raw = await asyncio.create_subprocess_exec("true")
        await raw.wait()

        with patch("exec_sandbox.platform_utils.psutil.Process", side_effect=AssertionError("looked up")) as lookup:
            wrapper = ProcessWrapper(raw)

        assert lookup.call_count == 0
        assert wrapper.psutil_proc is None

    async def test_a_missing_process_is_looked_up_once(self) -> None:
        """NoSuchProcess never clears with time: one lookup, no retry."""
        raw = await asyncio.create_subprocess_exec("sleep", "5")
        try:
            gone = psutil.NoSuchProcess(raw.pid)
            with patch("exec_sandbox.platform_utils.psutil.Process", side_effect=gone) as lookup:
                wrapper = ProcessWrapper(raw)

            assert lookup.call_count == 1
            assert wrapper.psutil_proc is None
        finally:
            raw.kill()
            await raw.wait()


class TestProcessWrapperProperties:
    """Tests for ProcessWrapper properties."""

    async def test_returncode_none_while_running(self) -> None:
        """returncode is None while process is running."""
        proc = await create_wrapped_process(["sleep", "10"])
        try:
            assert proc.returncode is None
        finally:
            await proc.kill()
            await proc.wait()

    @pytest.mark.parametrize(
        ("cmd", "code"),
        [(["true"], 0), (["false"], 1), (["sh", "-c", "exit 42"], 42)],
        ids=["zero", "one", "42"],
    )
    async def test_wait_returns_the_exit_code(self, cmd: list[str], code: int) -> None:
        """wait() returns the exit code, and returncode keeps it."""
        proc = await create_wrapped_process(cmd)
        assert await proc.wait() == code
        assert proc.returncode == code

    async def test_stdout_and_stderr_streams(self) -> None:
        """stdout and stderr are the child's two pipes."""
        proc = await create_wrapped_process(["sh", "-c", "echo out; echo err >&2"])
        assert proc.stdout is not None
        assert proc.stderr is not None
        assert (await proc.stdout.read(), await proc.stderr.read()) == (b"out\n", b"err\n")
        await proc.wait()


class TestProcessWrapperIsRunning:
    """Tests for ProcessWrapper.is_running method."""

    async def test_is_running_true(self) -> None:
        """is_running returns True for running process."""
        proc = await create_wrapped_process(["sleep", "10"])
        try:
            assert await proc.is_running() is True
        finally:
            await proc.kill()
            await proc.wait()

    async def test_is_running_false_after_exit(self) -> None:
        """is_running returns False after process exits."""
        proc = await create_wrapped_process(["true"])
        await proc.wait()
        assert await proc.is_running() is False


class TestProcessWrapperSignals:
    """ProcessWrapper.terminate sends SIGTERM, kill sends SIGKILL; a process that ended gets
    neither (ProcessLookupError)."""

    @pytest.mark.parametrize(("method", "signum"), [("terminate", signal.SIGTERM), ("kill", signal.SIGKILL)])
    async def test_signal_stops_a_running_process(self, method: str, signum: signal.Signals) -> None:
        proc = await create_wrapped_process(["sleep", "10"])
        await getattr(proc, method)()
        assert await proc.wait() == -signum
        assert not await proc.is_running()

    @pytest.mark.parametrize("method", ["terminate", "kill"])
    async def test_signal_to_a_finished_process_raises(self, method: str) -> None:
        proc = await create_wrapped_process(["true"])
        await proc.wait()

        with pytest.raises(ProcessLookupError):
            await getattr(proc, method)()
        assert proc.returncode == 0


class TestProcessWrapperCommunicate:
    """Tests for ProcessWrapper.communicate method."""

    async def test_communicate_captures_both(self) -> None:
        """communicate returns stdout and stderr apart."""
        proc = await create_wrapped_process(["sh", "-c", "echo out; echo err >&2"])
        assert await proc.communicate() == (b"out\n", b"err\n")


class TestProcessWrapperWaitWithTimeout:
    """Tests for ProcessWrapper.wait_with_timeout method."""

    async def test_wait_with_timeout_times_out(self) -> None:
        """wait_with_timeout raises TimeoutError on timeout."""
        proc = await create_wrapped_process(["sleep", "10"])

        with pytest.raises(asyncio.TimeoutError):
            await proc.wait_with_timeout(0.1)

        # Clean up
        await proc.kill()
        await proc.wait()

    async def test_wait_with_timeout_drains_pipes(self) -> None:
        """A child that writes more than a pipe holds (1 MB, no reader) exits: wait_with_timeout
        reads the pipes. wait() alone would wait for the full timeout."""
        proc = await create_wrapped_process(["sh", "-c", "head -c 1000000 /dev/zero; echo err >&2"])
        assert await proc.wait_with_timeout(5.0) == 0

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

    async def test_pipes_read_by_drain_task_exit_without_stall(self, tmp_path: Path) -> None:
        """A process whose pipes a drain task reads is waited without reading them.

        The two yields below order the loop so that communicate() starts its
        readers after the output reached the streams and before the woken drain
        readers run: these then drop the wakeup of communicate(), which never
        ends (the close() stall of 5 s on ~3% of VMs). wait() returns at exit.

        Flag files set up the race without timing: the child writes only once the
        drain readers wait ("go"), and the loop stays blocked until both lines are in
        the pipes ("written").
        """
        go, written = tmp_path / "go", tmp_path / "written"
        script = 'while [ ! -e "$1" ]; do sleep 0.01; done; echo out; echo err >&2; : > "$2"; sleep 0.8'
        proc = await create_wrapped_process(["sh", "-c", script, "sh", str(go), str(written)])
        lines: list[str] = []
        drain = start_drain(
            proc, process_name="test", context_id="test", stdout_handler=lines.append, stderr_handler=lines.append
        )
        await asyncio.sleep(0.05)  # both drain readers wait for data
        go.touch()
        deadline = time.monotonic() + 5.0
        while not written.exists():  # blocks the loop: the child writes both lines, nothing reads them yet
            assert time.monotonic() < deadline, "the child did not write"
            time.sleep(0.005)
        await asyncio.sleep(0)  # pass 1: this task, then the pipe reads (their data reaches the streams in pass 2)
        await asyncio.sleep(0)  # pass 2: this task, then the data: the drain readers are woken, run in pass 3

        assert await proc.wait_with_timeout(2.0) == 0
        await drain
        assert sorted(lines) == ["err", "out"]  # the drain task got every line

    async def test_ended_drain_task_leaves_the_pipes_to_communicate(self) -> None:
        """A drain task that ended (an error of its handler ends it) reads nothing more:
        wait() alone would time out on a child blocked on its full stdout pipe."""
        proc = await create_wrapped_process(["sh", "-c", "head -c 1000000 /dev/zero"])
        ended = asyncio.create_task(asyncio.sleep(0))
        await ended
        proc.drain_task = ended

        assert await proc.wait_with_timeout(5.0) == 0

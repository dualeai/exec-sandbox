"""Unit tests for subprocess utilities, with real subprocesses."""

import ast
import asyncio
import contextlib
import errno
import logging
import os
import select
import signal
import struct
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import psutil
import pytest

import exec_sandbox
from exec_sandbox import subprocess_utils
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.platform_utils import HostOS, ProcessWrapper
from exec_sandbox.process_registry import get_tracked_count, unregister_process
from exec_sandbox.subprocess_utils import (  # pyright: ignore[reportPrivateUsage]
    _launch_command,
    child_shell,
    communicate_managed_process,
    drain_subprocess_output,
    exec_errno,
    exec_failed_error,
    start_drain,
    start_managed_process,
    wait_for_socket,
)
from exec_sandbox.system_probes import probe_setpriv_pdeathsig
from tests.conftest import asyncio_with, skip_unless_linux


async def _close_writer(_r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    """Minimal handler for Unix server sockets — close immediately."""
    w.close()
    await w.wait_closed()


async def create_process(cmd: list[str]) -> ProcessWrapper:
    """Helper to create a wrapped subprocess."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return ProcessWrapper(proc)


_SPAWN_CALLS = {
    ("asyncio", "create_subprocess_exec"),
    ("asyncio", "create_subprocess_shell"),
    *(("subprocess", name) for name in ("run", "Popen", "check_output", "check_call", "call")),
}


class TestChildEnv:
    """Child processes never inherit the instance keyring or cloud credentials."""

    def test_every_spawn_site_passes_env(self) -> None:
        """AST walk over the package: each spawn call passes env= (a new site cannot forget it)."""
        missing: list[str] = []
        sites = 0
        for path in sorted(Path(exec_sandbox.__file__).parent.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if (ast.unparse(node.func.value), node.func.attr) not in _SPAWN_CALLS:
                    continue
                sites += 1
                if not any(keyword.arg == "env" for keyword in node.keywords):
                    missing.append(f"{path.name}:{node.lineno}")
        assert sites > 0
        assert missing == []

    def test_every_spawn_site_sets_stdin(self) -> None:
        """Each spawn call sets stdin= (or input=): no child gets this process's stdin, the
        terminal of the CLI (QEMU's console chardev would read it and clear its ISIG)."""
        missing: list[str] = []
        for path in sorted(Path(exec_sandbox.__file__).parent.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if (ast.unparse(node.func.value), node.func.attr) not in _SPAWN_CALLS:
                    continue
                if not any(keyword.arg in ("stdin", "input") for keyword in node.keywords):
                    missing.append(f"{path.name}:{node.lineno}")
        assert missing == []

    async def test_managed_child_does_not_see_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", "secret-keyring")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-aws")
        monkeypatch.setenv("CHILD_ENV_PROBE", "kept")

        returncode, stdout, _ = await communicate_managed_process(
            ["/usr/bin/env"], process_name="env", context_id="test-child-env"
        )

        assert returncode == 0
        assert b"CHILD_ENV_PROBE=kept" in stdout
        assert b"secret-keyring" not in stdout
        assert b"secret-aws" not in stdout


class TestStartManagedProcess:
    """Process publication is atomic with respect to caller cancellation."""

    async def test_cancelled_waiter_defers_until_spawn_is_published(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cancel that comes while the child is made: the waiter ends only after the child
        is registered and given to on_process_started, so no child is left without an owner.
        The spawn waits on a gate before the real spawn: the window is too short to hit
        with timing only."""
        spawn_entered = asyncio.Event()
        release_spawn = asyncio.Event()
        published: list[ProcessWrapper] = []
        real_spawn = asyncio.create_subprocess_exec

        async def gated_spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
            spawn_entered.set()
            await release_spawn.wait()
            return await real_spawn(*args, **kwargs)

        monkeypatch.setattr(subprocess_utils, "asyncio", asyncio_with(create_subprocess_exec=gated_spawn))
        tracked = get_tracked_count()
        waiter = asyncio.create_task(
            start_managed_process(
                ["sleep", "60"], steps=(), end_with_this_process=False, pass_fds=(), on_process_started=published.append
            )
        )
        await spawn_entered.wait()
        waiter.cancel()
        await asyncio.sleep(0)

        assert not waiter.done()
        release_spawn.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        [proc] = published
        try:
            assert proc.returncode is None  # the child runs
            assert get_tracked_count() == tracked + 1  # and force_kill_all reaches it
        finally:
            await proc.kill()
            await proc.wait()
            unregister_process(proc)

    async def test_child_stdin_is_dev_null(self) -> None:
        """Not the host's stdin: QEMU's console would read it into the guest. A pipe stands in
        for the host's stdin (pytest already puts /dev/null on fd 0)."""
        probe = "import os; print(os.fstat(0).st_rdev)"
        read_end, write_end = os.pipe()
        saved = os.dup(0)
        os.dup2(read_end, 0)
        try:
            proc = await start_managed_process(
                [sys.executable, "-I", "-c", probe],
                steps=(),
                end_with_this_process=False,
                pass_fds=(),
                on_process_started=None,
            )
        finally:
            os.dup2(saved, 0)
            for fd in (saved, read_end, write_end):
                os.close(fd)
        out, _ = await proc.communicate()
        assert int(out) == Path(os.devnull).stat().st_rdev


def write_foreign_elf(path: Path) -> Path:
    """An executable ELF file for a CPU that no kernel runs (e_machine 0xBEEF): its exec
    fails with ENOEXEC, as for a binary of another architecture, and no binfmt_misc
    handler (qemu-user) takes it."""
    header = b"\x7fELF\x02\x01\x01" + bytes(9)
    header += struct.pack("<HHIQQQIHHHHHH", 2, 0xBEEF, 1, 0x400000, 64, 0, 0, 64, 56, 0, 64, 0, 0)
    path.write_bytes(header + bytes(range(256)))
    path.chmod(0o755)
    return path


_PDEATHSIG_PROBE = "import ctypes; v = ctypes.c_int(); ctypes.CDLL(None).prctl(2, ctypes.byref(v)); print(v.value)"
"""Prints the parent-death signal of its process (2 = PR_GET_PDEATHSIG)."""

_KILLED_OWNER = """
import asyncio

from exec_sandbox.subprocess_utils import start_managed_process


async def main() -> None:
    print((await start_managed_process(["sleep", "60"], steps=(), end_with_this_process=True, pass_fds=(), on_process_started=None)).pid, flush=True)
    await asyncio.Event().wait()  # until the test kills this process


asyncio.run(main())
"""
"""A process with a child that ends with it, killed by the test (SIGKILL: no cleanup runs)."""


class TestChildShell:
    """Shell steps before a command, in place of a Python preexec_fn."""

    async def test_steps_run_in_order_and_no_value_is_parsed(self, tmp_path: Path) -> None:
        """Values and arguments are positional parameters: a value that a shell would expand
        stays as it is."""
        trap, folder = "$(touch injected) `touch injected` *", tmp_path / "sub"
        folder.mkdir()
        cmd = child_shell(
            ["/bin/sh", "-c", 'printf "%s|%s" "$0" "$1" > second', trap, "a b"],
            steps=[('cd "$1"', str(folder)), ('printf "%s" "$1" > first', trap)],
        )
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=tmp_path)
        assert await proc.wait() == 0
        assert (folder / "first").read_text() == trap
        assert (folder / "second").read_text() == f"{trap}|a b"
        assert sorted(path.name for path in tmp_path.rglob("*")) == ["first", "second", "sub"]

    def test_no_step_is_the_command_itself(self) -> None:
        assert child_shell(["qemu-system-missing", "-version"], steps=()) == ["qemu-system-missing", "-version"]

    @pytest.mark.parametrize(
        "case",
        [
            ("{tmp}/qemu-system-test", None, b"", errno.ENOENT),
            ("{tmp}/qemu-system-test", 0o644, b"#!/bin/sh\n", errno.EACCES),
            ("qemu-system-test", None, b"", errno.ENOENT),
            ("qemu-system-test", 0o644, b"#!/bin/sh\n", errno.EACCES),
            ("qemu-system-test", 0o755, b"#!/bin/sh\n", None),
            ("{tmp}/qemu-system-test", 0o755, b"hello\n", errno.ENOEXEC),
            ("{tmp}/qemu-system-test", 0o755, b"", errno.ENOEXEC),
            ("{tmp}/qemu-system-test", 0o755, b"\x7fELF\x02\x01\x01", None),
        ],
        ids=[
            "missing",
            "no-exec-bit",
            "not-on-path",
            "on-path-no-exec-bit",
            "on-path",
            "text-no-shebang",
            "empty",
            "elf",
        ],
    )
    def test_command_exec_cannot_start_raises_as_popen_would(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: tuple[str, int | None, bytes, int | None]
    ) -> None:
        """The shell, not Popen, runs the exec of the command: its error would only be an exit
        status. A command that exec would not find, may not run or would refuse as no
        program (ENOEXEC) raises first: a shell runs such a file as a shell script, and an
        empty one "runs" with exit 0. A bare name is looked up on the PATH (here: only the
        test folder)."""
        name, mode, content, err = case
        binary = tmp_path / "qemu-system-test"
        if mode is not None:
            binary.write_bytes(content)
            binary.chmod(mode)
        monkeypatch.setenv("PATH", str(tmp_path))
        cmd = [name.format(tmp=tmp_path), "-version"]
        if err is None:
            assert child_shell(cmd, steps=[(":", "x")])[-2:] == cmd
            return
        with pytest.raises(OSError, check=lambda e: e.errno == err and e.filename == cmd[0]):
            child_shell(cmd, steps=[(":", "x")])


class TestExecLookupCache:
    """child_shell looks a binary up once per name and PATH (exec_errno), not at each launch."""

    def test_one_lookup_per_name_and_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A binary found once is not looked up again for the same PATH: its removal shows at
        the exec, not here. Another PATH is a new lookup."""
        program = tmp_path / "qemu-system-test"
        program.write_bytes(b"#!/bin/sh\n")
        program.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        child_shell([program.name], steps=[(":", "x")])
        program.unlink()

        for _ in range(2):
            assert child_shell([program.name], steps=[(":", "x")])[-1] == program.name

        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{tmp_path / 'other'}")
        with pytest.raises(FileNotFoundError):
            child_shell([program.name], steps=[(":", "x")])

    def test_a_binary_installed_after_a_failure_starts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failure is not kept: the binary installed after a failed launch is found at the
        next one; each failure raises a new OSError (one raised object would grow its
        traceback at each raise)."""
        monkeypatch.setenv("PATH", str(tmp_path))
        errors = []
        for _ in range(2):
            with pytest.raises(FileNotFoundError) as info:
                child_shell(["qemu-system-late"], steps=[(":", "x")])
            errors.append(info.value)
        assert errors[0] is not errors[1]
        program = tmp_path / "qemu-system-late"
        program.write_bytes(b"#!/bin/sh\n")
        program.chmod(0o755)
        child_shell([program.name], steps=[(":", "x")])
        assert exec_errno(program.name, str(tmp_path)) is None


class TestBinfmtMisc:
    """A file of no known start that an enabled binfmt_misc entry claims: the kernel runs it
    through the entry's interpreter, so the check before the spawn takes it. A fake
    entries folder stands in for /proc/sys/fs/binfmt_misc."""

    @staticmethod
    def _entries(folder: Path, *, status: str, entries: dict[str, str]) -> Path:
        folder.mkdir()
        (folder / "status").write_text(f"{status}\n")
        (folder / "register").write_text("")  # write-only in /proc: never an entry
        for name, body in entries.items():
            (folder / name).write_text(body)
        return folder

    @staticmethod
    def _program(path: Path, content: bytes) -> str:
        path.write_bytes(content)
        path.chmod(0o755)
        return str(path)

    _PYC = "enabled\ninterpreter /usr/bin/python3.14\nflags: \noffset 0\nmagic 2b0e0d0a\n"
    _MASKED = "enabled\ninterpreter /usr/bin/qemu-test\nflags: F\noffset 2\nmagic aa00\nmask ff00\n"
    _EXE = "enabled\ninterpreter /usr/bin/wine\nflags: \nextension .exe\n"

    @pytest.mark.parametrize(
        "case",
        [
            ("enabled", {"python3.14": _PYC}, "run.pyc", b"\x2b\x0e\x0d\x0a" + bytes(12), True),
            ("enabled", {"python3.14": _PYC.replace("enabled", "disabled", 1)}, "run.pyc", b"\x2b\x0e\x0d\x0a", False),
            ("disabled", {"python3.14": _PYC}, "run.pyc", b"\x2b\x0e\x0d\x0a", False),
            ("enabled", {"python3.14": _PYC}, "run.pyc", b"\x2b\x0e\x0d\x0b", False),
            ("enabled", {"qemu-test": _MASKED}, "tool", b"xx\xaa\x55", True),
            ("enabled", {"qemu-test": _MASKED}, "tool", b"xx\xab\x55", False),
            ("enabled", {"wine": _EXE}, "tool.exe", b"MZ text", True),
            ("enabled", {"wine": _EXE}, "tool.exe.txt", b"MZ text", False),
            ("enabled", {"broken": "enabled\nmagic zz\n"}, "tool", b"zz", False),
        ],
        ids=[
            "magic",
            "entry-disabled",
            "binfmt-misc-disabled",
            "other-magic",
            "magic-under-mask",
            "masked-mismatch",
            "extension",
            "other-extension",
            "unreadable-entry",
        ],
    )
    def test_an_enabled_entry_that_claims_the_file_takes_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: tuple[str, dict[str, str], str, bytes, bool]
    ) -> None:
        status, entries, file, content, claimed = case
        folder = self._entries(tmp_path / "binfmt_misc", status=status, entries=entries)
        monkeypatch.setattr(subprocess_utils, "_BINFMT_MISC", folder)
        program = self._program(tmp_path / file, content)
        if claimed:
            assert child_shell([program], steps=[(":", "x")])[-1] == program
            return
        with pytest.raises(OSError, check=lambda e: e.errno == errno.ENOEXEC):
            child_shell([program], steps=[(":", "x")])

    @pytest.mark.parametrize(
        ("entry", "file"),
        [
            (b"enabled\ninterpreter /usr/bin/\xff-interp\nflags: \noffset 0\nmagic 2b0e0d0a\n", "run.pyc"),
            pytest.param(  # binfmt_misc is Linux only, and APFS refuses a file name that is not UTF-8
                b"enabled\ninterpreter /usr/bin/wine\nflags: \nextension .\xe9xe\n",
                "tool.\udce9xe",
                marks=skip_unless_linux,
                id="extension",
            ),
        ],
        ids=["interpreter", "extension"],
    )
    def test_an_entry_that_is_not_utf8_still_claims(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: bytes, file: str
    ) -> None:
        """Entry files hold bytes (an interpreter path or an extension need not be UTF-8): the
        kernel still runs the file through such an entry."""
        folder = self._entries(tmp_path / "binfmt_misc", status="enabled", entries={})
        (folder / "entry").write_bytes(entry)
        monkeypatch.setattr(subprocess_utils, "_BINFMT_MISC", folder)
        program = self._program(tmp_path / file, b"\x2b\x0e\x0d\x0a" + bytes(12))
        assert child_shell([program], steps=[(":", "x")])[-1] == program

    def test_magic_past_the_end_of_the_file_reads_zeros(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The kernel matches the magic in a zero-filled buffer of the file's first 256 bytes."""
        entry = "enabled\ninterpreter /usr/bin/x\nflags: \noffset 10\nmagic 0000\n"
        monkeypatch.setattr(
            subprocess_utils, "_BINFMT_MISC", self._entries(tmp_path / "b", status="enabled", entries={"z": entry})
        )
        program = self._program(tmp_path / "short", b"abc")
        assert child_shell([program], steps=[(":", "x")])[-1] == program

    def test_no_binfmt_misc_refuses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not mounted (or no Linux): no entry claims anything."""
        monkeypatch.setattr(subprocess_utils, "_BINFMT_MISC", tmp_path / "missing")
        program = self._program(tmp_path / "run.pyc", b"\x2b\x0e\x0d\x0a")
        with pytest.raises(OSError, check=lambda e: e.errno == errno.ENOEXEC):
            child_shell([program], steps=[(":", "x")])


class TestExecFailedError:
    """A launch that ends with 126 or 127: an exec in it failed."""

    @pytest.mark.parametrize(
        ("returncode", "reason"),
        [
            (126, "a program of the launch cannot run: no exec bit or another CPU architecture"),
            (127, "a program of the launch was not found, or its ELF interpreter is missing"),
        ],
    )
    def test_126_and_127_are_host_dependency(self, returncode: int, reason: str) -> None:
        """As spawn_error gives for an exec that Popen runs: no retry fixes it."""
        error = exec_failed_error(
            returncode,
            binary="/opt/gvproxy",
            detail="exec: /opt/gvproxy: not found",
            context={"vm_id": "v"},
            hint="install it",
        )
        assert error is not None
        assert error.code is ErrorCode.HOST_DEPENDENCY
        assert str(error) == (
            f"Failed to start /opt/gvproxy: exit status {returncode} ({reason}): exec: /opt/gvproxy: not found"
        )
        assert error.context == {"vm_id": "v", "binary": "/opt/gvproxy", "exit_code": returncode}

    @pytest.mark.parametrize("returncode", [None, 0, 1, 2, 125, 128, -9])
    def test_any_other_end_is_none(self, returncode: int | None) -> None:
        assert exec_failed_error(returncode, binary="b", detail=None, context={}, hint=None) is None

    @skip_unless_linux
    async def test_command_of_another_arch_ends_the_launch_with_126(self, tmp_path: Path) -> None:
        """The check before the spawn takes any ELF; the kernel refuses this one (ENOEXEC)
        and the launch shell ends with 126 (dash and bash refuse a binary file as a script)."""
        binary = write_foreign_elf(tmp_path / "gvproxy-wrapper-test")
        proc = await start_managed_process(
            [str(binary)], steps=(), end_with_this_process=True, pass_fds=(), on_process_started=None
        )
        await proc.communicate()
        unregister_process(proc)
        error = exec_failed_error(proc.returncode, binary=str(binary), detail=None, context={}, hint=None)
        assert (proc.returncode, error is not None) == (126, True)


class TestEndWithThisProcess:
    """A child that the kernel SIGKILLs when this process ends: setpriv --pdeathsig, no Python
    preexec_fn, so Popen can vfork; the preexec_fn only without setpriv. Some tests read
    the launch from _launch_command: a preexec_fn changes only how Popen makes the child (fork
    or vfork), which no public result shows."""

    @skip_unless_linux
    async def test_child_gets_sigkill_with_no_preexec(self) -> None:
        if not await probe_setpriv_pdeathsig():
            pytest.skip("no setpriv --pdeathsig on this host (util-linux before 2.33)")
        cmd, preexec_fn = await _launch_command(
            [sys.executable, "-I", "-c", _PDEATHSIG_PROBE], steps=(), end_with_this_process=True
        )
        assert preexec_fn is None
        returncode, out, _ = await communicate_managed_process(cmd, process_name="probe", context_id="test-pdeathsig")
        assert (returncode, int(out)) == (0, signal.SIGKILL)

    @skip_unless_linux
    async def test_child_of_another_parent_stops_before_its_command(self, tmp_path: Path) -> None:
        """A parent that ended before setpriv's prctl call sends no signal: the shell sees
        another parent (here: another pid stands for this process) and exits first."""
        if not await probe_setpriv_pdeathsig():
            pytest.skip("no setpriv --pdeathsig on this host (util-linux before 2.33)")
        marker = tmp_path / "ran"
        with patch.object(os, "getpid", return_value=1):
            cmd, _ = await _launch_command(["/bin/sh", "-c", f"touch {marker}"], steps=(), end_with_this_process=True)
        returncode, _, _ = await communicate_managed_process(cmd, process_name="probe", context_id="test-ppid")
        assert returncode == 1
        assert not marker.exists()

    @skip_unless_linux
    async def test_without_setpriv_the_preexec_sets_the_signal(self) -> None:
        """A host without setpriv --pdeathsig: the same signal through die_with_spawning_thread
        (Popen then forks this process)."""
        probe = [sys.executable, "-I", "-c", _PDEATHSIG_PROBE]
        with patch("exec_sandbox.subprocess_utils.probe_setpriv_pdeathsig", new_callable=AsyncMock, return_value=False):
            cmd, preexec_fn = await _launch_command(probe, steps=(), end_with_this_process=True)
            assert cmd[0] == sys.executable
            assert preexec_fn is not None
            proc = await start_managed_process(
                probe, steps=(), end_with_this_process=True, pass_fds=(), on_process_started=None
            )
        out, _ = await proc.communicate()
        unregister_process(proc)
        assert int(out) == signal.SIGKILL

    async def test_no_signal_where_the_os_has_none(self) -> None:
        """No setpriv --pdeathsig and no prctl signal (macOS): the command itself, no setpriv,
        no preexec_fn. A capability, as the probe and the fallback answer it."""
        with (
            patch("exec_sandbox.subprocess_utils.probe_setpriv_pdeathsig", new_callable=AsyncMock, return_value=False),
            patch("exec_sandbox.subprocess_utils.detect_host_os", return_value=HostOS.MACOS),
        ):
            assert await _launch_command(["qemu-storage-daemon"], steps=(), end_with_this_process=True) == (
                ["qemu-storage-daemon"],
                None,
            )

    async def test_a_child_not_ending_with_this_process_runs_its_steps_only(self) -> None:
        """Without end_with_this_process (a setuid launcher, a transient run): the steps, no
        setpriv, no preexec_fn, whatever the host."""
        assert await _launch_command(["true"], steps=[(":", "x")], end_with_this_process=False) == (
            ["/bin/sh", "-c", ':; shift; exec "$@"', "sh", "x", "true"],
            None,
        )

    @skip_unless_linux
    async def test_child_of_a_killed_process_ends_at_once(self) -> None:
        """No VM cgroup and no watcher reach such a child (gvproxy, qemu-storage-daemon, QEMU as
        this user without cgroups): its process's end is its end."""
        owner = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _KILLED_OWNER, stdout=asyncio.subprocess.PIPE
        )
        pidfd: int | None = None
        try:
            assert owner.stdout is not None
            pidfd = os.pidfd_open(int(await owner.stdout.readline()))  # readable once the child ends
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            assert poller.poll(0) == []  # alive while its process lives
            owner.kill()
            assert await owner.wait() == -signal.SIGKILL
            start = asyncio.get_running_loop().time()
            assert await asyncio.to_thread(poller.poll, 2000)  # ends within ms; the timeout only bounds a failure
            assert asyncio.get_running_loop().time() - start < 1.0
        finally:
            if owner.returncode is None:
                owner.kill()
                await owner.wait()
            if pidfd is not None:
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)  # a failed test leaves no child
                os.close(pidfd)


class TestCommunicateManagedProcess:
    """Transient children retain cleanup ownership across cancellation: a real child, and the
    real process registry. cleanup_process is wrapped only to make one attempt report a death
    it did not confirm (an OS fault a test cannot cause)."""

    @staticmethod
    async def _child(argv: list[str]) -> psutil.Process:
        """The child of this process with this command line, once it runs."""
        async with asyncio.timeout(5):
            while True:
                for child in psutil.Process().children():
                    with contextlib.suppress(psutil.Error):
                        if child.cmdline() == argv:
                            return child
                await asyncio.sleep(0.005)

    @staticmethod
    def _unconfirmed_first(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
        """cleanup_process whose first attempt reports an unconfirmed death without acting;
        the next ones run the real cleanup. Returns the result of each attempt."""
        results: list[bool] = []
        real_cleanup = subprocess_utils.cleanup_process

        async def cleanup(
            proc: ProcessWrapper,
            name: str,
            context_id: str,
            *,
            term_timeout: float,
            kill_timeout: float,
            kill_first: bool,
        ) -> bool:
            results.append(
                bool(results)
                and await real_cleanup(
                    proc,
                    name,
                    context_id,
                    term_timeout=term_timeout,
                    kill_timeout=kill_timeout,
                    kill_first=kill_first,
                )
            )
            return results[-1]

        monkeypatch.setattr(subprocess_utils, "cleanup_process", cleanup)
        return results

    async def test_cancelled_communication_kills_reaps_and_unregisters(self) -> None:
        argv = ["sleep", "61.5"]
        tracked = get_tracked_count()
        task = asyncio.create_task(communicate_managed_process(argv, process_name="sleep", context_id="cancelled"))
        child = await self._child(argv)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not psutil.pid_exists(child.pid)  # ended and reaped
        assert get_tracked_count() == tracked

    async def test_normal_completion_unregisters_after_confirmed_cleanup(self) -> None:
        tracked = get_tracked_count()
        result = await communicate_managed_process(
            ["/bin/sh", "-c", "echo out; echo err >&2"], process_name="sh", context_id="normal"
        )

        assert result == (0, b"out\n", b"err\n")
        assert get_tracked_count() == tracked

    async def test_unconfirmed_cleanup_blocks_return_and_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The first cleanup does not confirm the end: the call waits 1 s (here a stand-in
        sleep that records it), with the child still registered, then tries again."""
        tracked = get_tracked_count()
        attempts = self._unconfirmed_first(monkeypatch)
        registered_at_backoff: list[int] = []

        async def sleep(delay: float) -> None:
            registered_at_backoff.append(get_tracked_count() - tracked)
            assert delay == 1.0

        monkeypatch.setattr(subprocess_utils, "asyncio", asyncio_with(sleep=sleep))
        result = await communicate_managed_process(["/bin/sh", "-c", "echo out"], process_name="sh", context_id="retry")

        assert result == (0, b"out\n", b"")
        assert attempts == [False, True]
        assert registered_at_backoff == [1]  # unregistered only after a confirmed end
        assert get_tracked_count() == tracked

    async def test_cancellation_during_cleanup_backoff_still_confirms_death(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracked = get_tracked_count()
        attempts = self._unconfirmed_first(monkeypatch)
        backoff_entered = asyncio.Event()
        release_backoff = asyncio.Event()

        async def blocked_sleep(_delay: float) -> None:
            backoff_entered.set()
            await release_backoff.wait()

        monkeypatch.setattr(subprocess_utils, "asyncio", asyncio_with(sleep=blocked_sleep))
        task = asyncio.create_task(
            communicate_managed_process(["/bin/sh", "-c", "exit 0"], process_name="sh", context_id="cancel-backoff")
        )
        await backoff_entered.wait()
        task.cancel()
        release_backoff.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert attempts == [False, True]
        assert get_tracked_count() == tracked


class TestDrainSubprocessOutput:
    """Tests for drain_subprocess_output function.

    Uses real subprocesses - no mocking.
    """

    async def test_drain_both_stdout_stderr(self) -> None:
        """Each stream goes to its own handler, and to that handler only."""
        captured_stdout: list[str] = []
        captured_stderr: list[str] = []

        # Process that writes to both streams
        proc = await create_process(["bash", "-c", "echo 'stdout line'; echo 'stderr line' >&2"])

        await drain_subprocess_output(
            proc,
            stdout_handler=captured_stdout.append,
            stderr_handler=captured_stderr.append,
        )

        await proc.wait()

        assert captured_stdout == ["stdout line"]
        assert captured_stderr == ["stderr line"]

    async def test_drain_multiple_lines(self) -> None:
        """Each line reaches the handler once, in order, without its newline."""
        captured: list[str] = []

        proc = await create_process(["bash", "-c", "echo line1; echo line2; echo line3"])

        await drain_subprocess_output(proc, stdout_handler=captured.append, stderr_handler=[].append)

        await proc.wait()

        assert captured == ["line1", "line2", "line3"]

    async def test_drain_reads_both_pipes_at_once(self) -> None:
        """A child that writes 2 MB to stderr before it writes stdout: a drain that read stdout
        to its end first would wait for ever. asyncio buffers 128 KiB of a pipe that nobody
        reads, the pipe holds 64 KiB more, then the child blocks on its write."""
        captured_stdout: list[str] = []
        captured_stderr: list[str] = []
        script = "import sys; sys.stderr.write(''.join(f'err {i}\\n' for i in range(200_000))); print('out')"
        proc = await create_process([sys.executable, "-c", script])

        async with asyncio.timeout(10):
            await drain_subprocess_output(
                proc,
                stdout_handler=captured_stdout.append,
                stderr_handler=captured_stderr.append,
            )
            assert await proc.wait() == 0

        assert captured_stdout == ["out"]
        assert captured_stderr == [f"err {i}" for i in range(200_000)]

    async def test_drain_large_output(self) -> None:
        """Output larger than a pipe (64 KiB): every line reaches the handler, in order."""
        captured: list[str] = []

        # 3000 lines of about 32 bytes: 95 KB
        proc = await create_process(
            ["bash", "-c", 'for i in $(seq 1 3000); do echo "line $i with some padding text"; done']
        )

        await drain_subprocess_output(proc, stdout_handler=captured.append, stderr_handler=[].append)

        await proc.wait()

        assert captured == [f"line {i} with some padding text" for i in range(1, 3001)]

    @pytest.mark.parametrize(
        ("output", "lines"),
        [  # the child builds its output: one argument holds at most 128 KiB on Linux
            ("'x' * 200_000 + '\\nafter\\n' + 'y' * 300_000", ["after"]),
            ("'short\\n' + 'z' * 300_000 + '\\nlast'", ["short", "last"]),
            ("'a' * 300_000 + '\\n' + 'b' * 300_000 + '\\n'", []),
        ],
        ids=["long-then-line-then-long-at-eof", "line-long-line-at-eof", "only-long-lines"],
    )
    async def test_drain_drops_a_line_over_the_limit_whole(self, output: str, lines: list[str]) -> None:
        """A line over the reader's 64 KiB limit is dropped whole (no tail comes out as a
        line, even when the line spans several reads) and the drain reads on to EOF, so the
        pipe never fills; stderr the same."""
        for stream in ("stdout", "stderr"):
            captured: list[str] = []
            proc = await create_process(
                [sys.executable, "-c", f"import sys; sys.{stream}.write({output}); sys.{stream}.flush()"]
            )
            await drain_subprocess_output(proc, stdout_handler=captured.append, stderr_handler=captured.append)
            assert await proc.wait() == 0
            assert captured == lines

    async def test_drain_empty_output(self) -> None:
        """Drain from process with no output."""
        captured_stdout: list[str] = []
        captured_stderr: list[str] = []

        proc = await create_process(["true"])  # Does nothing, exits 0

        await drain_subprocess_output(
            proc,
            stdout_handler=captured_stdout.append,
            stderr_handler=captured_stderr.append,
        )

        await proc.wait()

        assert captured_stdout == []
        assert captured_stderr == []

    async def test_drain_binary_safe(self) -> None:
        """A line that is not UTF-8 is skipped; the drain goes on to the next line."""
        captured: list[str] = []

        proc = await create_process(["bash", "-c", r"printf '\377\376 bad\nnormal text\n'"])

        await drain_subprocess_output(proc, stdout_handler=captured.append, stderr_handler=[].append)

        await proc.wait()

        assert captured == ["normal text"]


class TestStartDrain:
    """start_drain: its task is the process's drain_task (cleanup then never reads the
    pipes), it reads both pipes, and a failure of the task is logged."""

    async def test_task_is_the_drain_task_and_reads_both_pipes(self) -> None:
        lines: list[str] = []
        proc = await create_process(["bash", "-c", "echo out; echo err >&2"])

        task = start_drain(
            proc, process_name="bash", context_id="drain-1", stdout_handler=lines.append, stderr_handler=lines.append
        )

        assert proc.drain_task is task
        await task
        await proc.wait()
        assert sorted(lines) == ["err", "out"]

    async def test_failed_task_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        proc = await create_process(["echo", "line"])

        def fail(_line: str) -> None:
            raise RuntimeError("handler failed")

        with caplog.at_level(logging.ERROR, logger="exec_sandbox.subprocess_utils"):
            task = start_drain(
                proc, process_name="echo", context_id="drain-2", stdout_handler=fail, stderr_handler=[].append
            )
            with pytest.raises(ExceptionGroup):
                await task  # the log callback was added first: it ran before this await returns
        await proc.wait()

        assert [(r.getMessage(), getattr(r, "task_name", None)) for r in caplog.records] == [
            ("Background task failed", "echo-drain-drain-2")
        ]


class TestWaitForSocket:
    """Tests for wait_for_socket function.

    Uses real Unix sockets - no mocking.
    Uses short_tmp_dir fixture (short paths) instead of tmp_path
    to avoid macOS AF_UNIX 104-byte path limit.
    """

    async def test_socket_ready_immediately(self, short_tmp_dir: Path) -> None:
        """Socket already listening before wait_for_socket is called."""
        sock_path = short_tmp_dir / "s.sock"

        server = await asyncio.start_unix_server(_close_writer, path=str(sock_path))
        try:
            result = await wait_for_socket(sock_path, timeout=2.0, abort_check=lambda: None, keep_connection=False)
            assert result is None
        finally:
            server.close()
            await server.wait_closed()

    async def test_socket_appears_after_the_wait_starts(self, short_tmp_dir: Path) -> None:
        """The socket appears while the wait polls for its file (abort_check marks a poll): the
        wait goes on, then connects once the socket listens."""
        sock_path = short_tmp_dir / "s.sock"
        polled = asyncio.Event()
        wait = asyncio.create_task(
            wait_for_socket(sock_path, timeout=2.0, abort_check=polled.set, keep_connection=False)
        )
        await polled.wait()
        assert not wait.done()  # no file yet: it polls

        server = await asyncio.start_unix_server(_close_writer, path=str(sock_path))
        try:
            assert await wait is None
        finally:
            server.close()
            await server.wait_closed()

    async def test_keep_connection_true_returns_streams(self, short_tmp_dir: Path) -> None:
        """keep_connection=True returns the (reader, writer) of its connection: the server's
        reply to "ping" comes on that reader."""
        sock_path = short_tmp_dir / "s.sock"

        async def pong(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.write(b"pong to " + await reader.readline())
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(pong, path=str(sock_path))
        try:
            result = await wait_for_socket(sock_path, timeout=2.0, keep_connection=True, abort_check=lambda: None)
            assert result is not None
            r, w = result
            w.write(b"ping\n")
            await w.drain()
            assert await r.read() == b"pong to ping\n"
            w.close()
            await w.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_timeout_no_socket(self, short_tmp_dir: Path) -> None:
        """TimeoutError when socket file never appears."""
        sock_path = short_tmp_dir / "s.sock"
        with pytest.raises(TimeoutError):
            await wait_for_socket(sock_path, timeout=0.1, abort_check=lambda: None, keep_connection=False)

    async def test_socket_file_exists_but_not_listening(self, short_tmp_dir: Path) -> None:
        """Fails when path exists as regular file, not a socket.

        On Linux: connect() retries with ConnectionRefused → TimeoutError.
        On macOS: connect() raises OSError (ENOTSOCK) immediately.
        """
        sock_path = short_tmp_dir / "s.sock"
        sock_path.write_text("not a socket")
        expected = (
            pytest.raises(OSError, check=lambda e: e.errno == errno.ENOTSOCK)
            if sys.platform == "darwin"
            else pytest.raises(TimeoutError)
        )
        with expected:
            await wait_for_socket(sock_path, timeout=0.2, abort_check=lambda: None, keep_connection=False)

    async def test_abort_check_raises(self, short_tmp_dir: Path) -> None:
        """abort_check exception propagates instead of TimeoutError."""
        sock_path = short_tmp_dir / "s.sock"
        call_count = 0

        def _abort() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("process died")

        with pytest.raises(RuntimeError, match="process died"):
            await wait_for_socket(sock_path, timeout=5.0, abort_check=_abort, keep_connection=False)
        assert call_count == 2  # the wait ended at the first raise

    async def test_zero_timeout(self, short_tmp_dir: Path) -> None:
        """Zero timeout raises TimeoutError immediately."""
        sock_path = short_tmp_dir / "s.sock"
        with pytest.raises(TimeoutError):
            await wait_for_socket(sock_path, timeout=0.0, abort_check=lambda: None, keep_connection=False)

    async def test_concurrent_wait_for_socket(self, short_tmp_dir: Path) -> None:
        """Multiple concurrent waits all succeed against the same server."""
        sock_path = short_tmp_dir / "s.sock"

        server = await asyncio.start_unix_server(_close_writer, path=str(sock_path))
        try:
            results = await asyncio.gather(
                wait_for_socket(sock_path, timeout=2.0, keep_connection=False, abort_check=lambda: None),
                wait_for_socket(sock_path, timeout=2.0, keep_connection=False, abort_check=lambda: None),
                wait_for_socket(sock_path, timeout=2.0, keep_connection=False, abort_check=lambda: None),
            )
            assert all(r is None for r in results)
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root connects to a socket of mode 0")
    async def test_socket_this_process_may_not_use_is_host_dependency(self, short_tmp_dir: Path) -> None:
        """connect() gives EACCES (a socket QEMU made as qemu-vm, and this process lacks the
        group): ConfigError HOST_DEPENDENCY at once, with the socket's owner and mode, not a
        raw PermissionError and not a wait until the timeout."""
        sock_path = short_tmp_dir / "s.sock"
        server = await asyncio.start_unix_server(_close_writer, path=str(sock_path))
        sock_path.chmod(0o000)
        try:
            with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc_info:
                await wait_for_socket(sock_path, timeout=5.0, abort_check=lambda: None, keep_connection=False)
        finally:
            server.close()
            await server.wait_closed()
        assert isinstance(exc_info.value.__cause__, PermissionError)
        assert exc_info.value.context["mode"] == "0o0"
        assert exc_info.value.context["owner_uid"] == os.getuid()

"""probe_qemu_sandbox_support: decides from QEMU's QMP greeting, not from a fixed wait.

The fake QEMU is a real process (one sh script per module: macOS checks each new
executable on its first run, ~0.1 s), so the probe runs its real subprocess and pipe
handling. The binary comes from EXEC_SANDBOX_QEMU_BIN_X86 / _ARM, the variables VMs read;
the host OS is pinned to Linux. Its behaviour comes from a mode file next to it. The bound
of the greeting wait has no setting: the tests that need it short or long patch the
private constant.
"""

import asyncio
import contextlib
import logging
import os
import shlex
import signal
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.platform_utils import HostOS
from exec_sandbox.system_probes import probe_qemu_sandbox_support, require_qemu_sandbox_support
from tests.conftest import skip_unless_linux

_GREETING = '{"QMP": {"version": {"qemu": {"micro": 3, "minor": 1, "major": 11}}, "capabilities": []}}'

_OUTSIDE = (
    f'{shlex.quote(sys.executable)} -c \'import os, sys; os.setpgid(0, 0); open(sys.argv[1], "w").close(); '
    'os.execvp("sleep", ["sleep", "3600"])\' "$here/outside_ready" & echo $! > "$here/outside_pid"; '
    'while [ ! -e "$here/outside_ready" ]; do sleep 0.01; done'
)
"""A process that leaves QEMU's group (its pid in outside_pid) and holds QEMU's pipes; QEMU
goes on once it has left (outside_ready): a kill of the group cannot reach it."""

_MODES = {
    "greets": f"echo '{_GREETING}'; exec sleep 3600",
    "no-seccomp": "echo \"qemu-system-x86_64: -sandbox on: There is no option group 'sandbox'\" >&2; exit 1",
    "filter-refused": (
        'echo "qemu-system-x86_64: -sandbox on,obsolete=deny: failed to install seccomp syscall filter in the kernel"'
        " >&2; exit 1"
    ),
    "filter-no-memory": (
        'echo "qemu-system-x86_64: -sandbox on: failed to install seccomp syscall filter in the kernel:'
        ' Cannot allocate memory" >&2; exit 1'
    ),
    "other-error": 'echo "gtk initialization failed" >&2; exit 1',
    "exits-0": "exit 0",
    "killed": "kill -9 $$",
    "sigsys": "kill -SYS $$",
    "closes-then-sigsys": "exec 1>&- 2>&-; sleep 0.3; kill -SYS $$",
    "closes-then-hangs": 'exec 1>&- 2>&-; : > "$here/closed"; exec sleep 3600',
    # QEMU greets while a process that left its group holds the pipes
    "outside-group-greets": f"{_OUTSIDE}; echo '{_GREETING}'; exec sleep 3600",
    # QEMU stays silent while a process that left its group holds the pipes
    "outside-group-silent": f"{_OUTSIDE}; exec sleep 3600",
    # QEMU exits (code 3) while a process of its group holds stderr
    "stderr-held": 'sleep 3600 >/dev/null & echo $! > "$here/child_pid"; exec 1>&-; exit 3',
    # A wrapper that does not exec QEMU: its child greets and holds the pipes, the wrapper waits
    "wrapper-greets": f'sh -c \'echo $$ > "$1"; echo "$2"; exec sleep 3600\' qemu "$here/child_pid" \'{_GREETING}\' & wait',
    "banner": "echo 'a wrapper banner'; exec sleep 3600",
    "silent": "exec sleep 3600",
    "long-line": "head -c 200000 /dev/zero | tr '\\0' x; echo; exec sleep 3600",
    "stderr-flood": f"head -c 300000 /dev/zero | tr '\\0' x >&2; echo '{_GREETING}'; exec sleep 3600",
}


@pytest.fixture(scope="module")
def fake_qemu(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("fake-qemu")
    script = folder / "qemu-system-fake"
    cases = "\n".join(f"  {mode}) {body} ;;" for mode, body in _MODES.items())
    script.write_text(
        "#!/bin/sh\n"
        'here=$(dirname "$0")\n'
        'echo run >> "$here/runs"\n'
        'echo $$ > "$here/pid"\n'
        'printf "%s\\n" "$@" > "$here/args"\n'
        f'case $(cat "$here/mode") in\n{cases}\nesac\n'
    )
    script.chmod(0o755)
    return script


async def _probe(
    binary: Path, mode: str | None = None, *, path: Path | None = None, ask: Callable[[], Awaitable[bool | None]]
) -> bool | None:
    """`ask` (probe_qemu_sandbox_support or require_qemu_sandbox_support) on Linux with
    `binary` as the QEMU of both arches; `path` replaces PATH (where the bare name is found
    when `binary` is not a file)."""
    if mode is not None:
        (binary.parent / "mode").write_text(mode)
    (binary.parent / "pid").unlink(missing_ok=True)  # _left_running reads this run's only
    env = {"EXEC_SANDBOX_QEMU_BIN_X86": str(binary), "EXEC_SANDBOX_QEMU_BIN_ARM": str(binary)}
    if path is not None:
        env["PATH"] = str(path)
    with (
        patch.dict(os.environ, env),
        patch("exec_sandbox.system_probes.detect_host_os", return_value=HostOS.LINUX),
    ):
        return await ask()


def _tasks_left() -> set[asyncio.Task[object]]:
    """The tasks still running besides the test's own: the probe leaves none (its stderr
    reader, however it is wrapped)."""
    return {task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()}


def _left_running(binary: Path) -> bool:
    """The fake QEMU of the last _probe still runs. No pid file: the probe stopped it before
    its first line (a loaded host at a short bound), so nothing runs."""
    try:
        os.kill(int((binary.parent / "pid").read_text()), 0)
    except (FileNotFoundError, ProcessLookupError):
        return False
    return True


async def test_probe_checks_the_filter_vms_install(fake_qemu: Path) -> None:
    """The probe starts QEMU with the -sandbox value of every VM, not a weaker one: on, with
    obsolete calls, privilege changes, process spawns and resource control denied."""
    assert await _probe(fake_qemu, "greets", ask=probe_qemu_sandbox_support) is True

    args = (fake_qemu.parent / "args").read_text().splitlines()
    assert args[args.index("-sandbox") + 1] == (
        "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny"
    )


async def test_greeting_means_supported_without_waiting(fake_qemu: Path) -> None:
    """QEMU greeted with -sandbox on: supported at once, though QEMU would run for an hour and
    the bound is an hour too. The probe stops QEMU."""
    with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 3600):
        async with asyncio.timeout(30):
            assert await _probe(fake_qemu, "greets", ask=probe_qemu_sandbox_support) is True
    assert not _left_running(fake_qemu)


@pytest.mark.parametrize("mode", ["no-seccomp", "filter-refused", "sigsys"])
async def test_a_seccomp_end_before_greeting_means_not_supported(fake_qemu: Path, mode: str) -> None:
    """Only the greeting proves the filter. An end before it that is about seccomp (built
    without it, a filter the kernel refuses, the filter's SIGSYS) is no sandbox."""
    assert await _probe(fake_qemu, mode, ask=probe_qemu_sandbox_support) is False


@pytest.mark.parametrize(
    ("mode", "error", "code"),
    [
        pytest.param("other-error", ConfigError, ErrorCode.HOST_DEPENDENCY, id="other-error"),
        pytest.param("exits-0", ConfigError, ErrorCode.HOST_DEPENDENCY, id="exits-0"),
        pytest.param("filter-no-memory", TransientError, ErrorCode.QEMU_CRASHED, id="filter-no-memory"),
    ],
)
async def test_another_end_before_greeting_starts_no_vm(
    fake_qemu: Path, mode: str, error: type[ConfigError | TransientError], code: ErrorCode
) -> None:
    """An end before the greeting that says nothing about seccomp (a library, memory, a limit)
    is no answer: not kept (a kept False would start every later VM without its filter), and
    no VM starts on it; the next call asks QEMU again. A limit can pass (QEMU_CRASHED); any
    other end needs a fix of the host (HOST_DEPENDENCY)."""
    runs = fake_qemu.parent / "runs"
    runs.unlink(missing_ok=True)
    with pytest.raises(error, check=lambda e: e.code is code):
        await _probe(fake_qemu, mode, ask=require_qemu_sandbox_support)
    assert await _probe(fake_qemu, "greets", ask=require_qemu_sandbox_support) is True  # asked again
    assert runs.read_text().count("run") == 2


@pytest.mark.parametrize("mode", ["banner", "silent", "long-line"])
async def test_running_qemu_without_greeting_is_bounded(fake_qemu: Path, mode: str) -> None:
    """Still running at the bound, greeting or not: -sandbox was accepted (an unknown option or a
    failed filter exits at once). The probe stops QEMU, never waits for it."""
    with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 0.2):
        assert await _probe(fake_qemu, mode, ask=probe_qemu_sandbox_support) is True
    assert not _left_running(fake_qemu)


async def test_stderr_never_blocks_the_greeting(fake_qemu: Path) -> None:
    """300 KB on stderr before the greeting: the probe drains it, so QEMU reaches its greeting."""
    with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 3600):
        async with asyncio.timeout(30):
            assert await _probe(fake_qemu, "stderr-flood", ask=probe_qemu_sandbox_support) is True


async def test_binary_that_cannot_run_is_no_answer(tmp_path: Path) -> None:
    """No file at the setting and no qemu-system on PATH, or a file without the execute bit:
    no answer (None), and no VM starts on it: ConfigError HOST_DEPENDENCY."""
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    not_executable = tmp_path / "qemu-system-plain-file"
    not_executable.write_text("#!/bin/sh\nexit 0\n")

    assert await _probe(tmp_path / "absent", path=empty_path, ask=probe_qemu_sandbox_support) is None
    assert await _probe(not_executable, ask=probe_qemu_sandbox_support) is None
    with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY):
        await _probe(not_executable, ask=require_qemu_sandbox_support)


async def test_macos_spawns_nothing(fake_qemu: Path) -> None:
    """macOS has no seccomp: not supported, and no QEMU starts (a QEMU that would greet)."""
    runs = fake_qemu.parent / "runs"
    runs.unlink(missing_ok=True)
    (fake_qemu.parent / "mode").write_text("greets")
    env = {"EXEC_SANDBOX_QEMU_BIN_X86": str(fake_qemu), "EXEC_SANDBOX_QEMU_BIN_ARM": str(fake_qemu)}
    with (
        patch.dict(os.environ, env),
        patch("exec_sandbox.system_probes.detect_host_os", return_value=HostOS.MACOS),
    ):
        assert await probe_qemu_sandbox_support() is False
    assert not runs.exists()


async def test_result_is_cached(fake_qemu: Path) -> None:
    """The first answer stays: a second probe starts no QEMU."""
    runs = fake_qemu.parent / "runs"
    runs.unlink(missing_ok=True)

    assert await _probe(fake_qemu, "greets", ask=probe_qemu_sandbox_support) is True
    assert await _probe(fake_qemu, "no-seccomp", ask=probe_qemu_sandbox_support) is True  # the first answer stays
    assert runs.read_text().count("run") == 1


@skip_unless_linux
async def test_real_qemu_greets_with_sandbox_on() -> None:
    """The QEMU of this host starts with -sandbox on (CI builds it with seccomp)."""
    with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 3600):
        async with asyncio.timeout(30):
            assert await probe_qemu_sandbox_support() is True


async def test_a_seccomp_exit_before_the_greeting_is_kept(fake_qemu: Path) -> None:
    """QEMU ran and exited before its greeting with a seccomp error: a final answer, kept, so
    the next VM starts no probe QEMU."""
    runs = fake_qemu.parent / "runs"
    runs.unlink(missing_ok=True)

    assert await _probe(fake_qemu, "no-seccomp", ask=probe_qemu_sandbox_support) is False
    assert await _probe(fake_qemu, "greets", ask=probe_qemu_sandbox_support) is False  # the first answer stays
    assert runs.read_text().count("run") == 1


async def test_a_binary_that_did_not_start_is_asked_again(tmp_path: Path) -> None:
    """A QEMU file that did not start (no execute bit: an upgrade half written) is no answer:
    the next probe runs QEMU again."""
    binary = tmp_path / "qemu-system-fake"
    binary.write_text(f"#!/bin/sh\necho '{_GREETING}'\nexec sleep 3600\n")
    binary.chmod(0o644)
    assert await _probe(binary, ask=probe_qemu_sandbox_support) is None
    binary.chmod(0o755)
    assert await _probe(binary, ask=probe_qemu_sandbox_support) is True


async def test_a_killed_probe_is_asked_again(fake_qemu: Path) -> None:
    """A probe QEMU that a signal killed (the OOM killer, a process limit) is no answer: not
    kept, since a kept False would start every later VM without seccomp. The next call asks
    QEMU again, and no VM starts on the missing answer (TransientError: the VM would run
    without its filter)."""
    runs = fake_qemu.parent / "runs"
    runs.unlink(missing_ok=True)
    with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED):
        await _probe(fake_qemu, "killed", ask=require_qemu_sandbox_support)
    assert await _probe(fake_qemu, "greets", ask=require_qemu_sandbox_support) is True  # asked again
    assert runs.read_text().count("run") == 2


async def test_a_seccomp_kill_is_kept(fake_qemu: Path) -> None:
    runs = fake_qemu.parent / "runs"
    runs.unlink(missing_ok=True)
    assert await _probe(fake_qemu, "sigsys", ask=require_qemu_sandbox_support) is False
    assert await _probe(fake_qemu, "greets", ask=require_qemu_sandbox_support) is False  # the first answer stays
    assert runs.read_text().count("run") == 1


async def test_a_qemu_that_closes_stdout_before_its_sigsys_is_waited_for(fake_qemu: Path) -> None:
    """QEMU closes stdout 0.3 s before its filter's SIGSYS ends it: the probe waits for that end
    (it kills only a QEMU still running at the bound), so the answer is the SIGSYS one, with any
    child watcher (a kill at the EOF would end QEMU with SIGKILL: no answer)."""
    assert await _probe(fake_qemu, "closes-then-sigsys", ask=probe_qemu_sandbox_support) is False


async def test_a_cancel_while_the_probe_waits_for_the_exit_stops_qemu(fake_qemu: Path) -> None:
    """QEMU closes stdout and then hangs: the probe waits for its exit (up to the bound), and a
    cancel during that wait still stops QEMU (a QEMU left running holds its memory)."""
    (fake_qemu.parent / "closed").unlink(missing_ok=True)
    task = asyncio.create_task(_probe(fake_qemu, "closes-then-hangs", ask=probe_qemu_sandbox_support))
    for _ in range(250):  # QEMU marks its closed stdout: the probe then reads the EOF and waits
        if (fake_qemu.parent / "closed").exists():
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("QEMU never marked its closed stdout")
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(100):  # the kill is sent; asyncio's child watcher reaps it in ms
        if not _left_running(fake_qemu):
            break
        await asyncio.sleep(0.02)
    assert not _left_running(fake_qemu)


async def test_a_qemu_that_closes_stdout_and_hangs_is_killed_at_the_bound(
    fake_qemu: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """QEMU closes stdout and still runs at the bound of the exit wait: the probe kills it (no
    answer: None), logs that kill, and QEMU does not keep running."""
    with (
        patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 0.5),
        caplog.at_level(logging.WARNING, logger="exec_sandbox.system_probes"),
    ):
        async with asyncio.timeout(10):
            assert await _probe(fake_qemu, "closes-then-hangs", ask=probe_qemu_sandbox_support) is None

    killed = [record for record in caplog.records if record.getMessage() == "QEMU sandbox probe killed"]
    assert [record.__dict__["returncode"] for record in killed] == [-9]
    assert not _left_running(fake_qemu)


async def test_a_member_that_holds_stderr_after_the_exit_is_stopped(fake_qemu: Path) -> None:
    """QEMU exits (code 3, no greeting) while a process of its group holds stderr: the probe
    kills the group, so it answers (None: another exit) and the member does not
    keep running. Python builds differ on wait(): where it returns at QEMU's exit (3.14.10) the
    held stderr calls the kill; where it waits for the pipes (3.12, 3.14.3) the bound of the
    exit wait does."""
    (fake_qemu.parent / "child_pid").unlink(missing_ok=True)
    with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 0.5):
        async with asyncio.timeout(10):
            assert await _probe(fake_qemu, "stderr-held", ask=probe_qemu_sandbox_support) is None

    member = int((fake_qemu.parent / "child_pid").read_text())
    for _ in range(100):  # the kill is sent; its new parent reaps it in ms
        try:
            os.kill(member, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail(f"the group member {member} still runs")


async def test_a_cancel_of_the_greeting_wait_is_raised_with_pipes_held_outside(fake_qemu: Path) -> None:
    """A cancel while the probe waits for the greeting is raised as it is, also when a process
    that left QEMU's group holds the pipes (the bounded reap would time out): the caller sees
    its cancel, never a probe that answered None."""
    for name in ("outside_pid", "outside_ready"):
        (fake_qemu.parent / name).unlink(missing_ok=True)
    # 5 s: the fake starts in time on a loaded host; the probe never waits for it (the cancel)
    with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 5.0):
        task = asyncio.create_task(_probe(fake_qemu, "outside-group-silent", ask=probe_qemu_sandbox_support))
        try:
            for _ in range(250):  # the outside process left the group: the probe waits for the greeting
                if (fake_qemu.parent / "outside_ready").exists():
                    break
                await asyncio.sleep(0.02)
            else:
                pytest.fail("the outside process never left QEMU's group")
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)  # the reader's cancel ends
            assert _tasks_left() == set()
        finally:
            with contextlib.suppress(FileNotFoundError, ValueError, ProcessLookupError):
                os.kill(int((fake_qemu.parent / "outside_pid").read_text()), signal.SIGKILL)


async def test_pipes_held_outside_the_group_bound_the_reap(fake_qemu: Path) -> None:
    """QEMU greets, and a process that left its group holds the pipes after the kill: the reap
    is bounded, so the probe answers None, never waits for pipes that do not close (the bound
    of the stderr read shows where wait() returns at QEMU's exit)."""
    for name in ("outside_pid", "outside_ready"):
        (fake_qemu.parent / name).unlink(missing_ok=True)
    try:
        with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 1.0):
            async with asyncio.timeout(10):
                assert await _probe(fake_qemu, "outside-group-greets", ask=probe_qemu_sandbox_support) is None
        await asyncio.sleep(0)  # the reader's cancel ends
        assert _tasks_left() == set()
    finally:
        with contextlib.suppress(FileNotFoundError, ValueError, ProcessLookupError):
            os.kill(int((fake_qemu.parent / "outside_pid").read_text()), signal.SIGKILL)


async def test_a_qemu_that_a_wrapper_did_not_exec_is_stopped(fake_qemu: Path) -> None:
    """A wrapper that did not exec QEMU: QEMU greets and holds the pipes. The probe kills its
    whole group, so it answers (no wait for pipes that never close), and QEMU does not keep
    running."""
    (fake_qemu.parent / "child_pid").unlink(missing_ok=True)
    async with asyncio.timeout(15):
        assert await _probe(fake_qemu, "wrapper-greets", ask=probe_qemu_sandbox_support) is True

    qemu = int((fake_qemu.parent / "child_pid").read_text())
    for _ in range(100):  # the kill is sent; its new parent reaps it in ms
        try:
            os.kill(qemu, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail(f"the wrapper's QEMU {qemu} still runs")


async def test_a_late_child_watcher_keeps_the_seccomp_kill(fake_qemu: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded host: asyncio's child watcher reaps the probe QEMU 0.3 s after its exit. The
    probe waits for that exit: its status is QEMU's (SIGSYS), never the 255 that asyncio
    reports when another waitpid took the status first (a kill that polls, as Popen.kill
    does). The watcher thread's blocking waitpid is the one delayed (macOS, Linux without
    pidfd); Popen's own poll keeps the real waitpid (bound at its definition)."""
    real_waitpid = os.waitpid

    def late_waitpid(pid: int, options: int) -> tuple[int, int]:
        if options == 0 and threading.current_thread() is not threading.main_thread():
            time.sleep(0.3)
        return real_waitpid(pid, options)

    monkeypatch.setattr(os, "waitpid", late_waitpid)
    assert await _probe(fake_qemu, "sigsys", ask=probe_qemu_sandbox_support) is False

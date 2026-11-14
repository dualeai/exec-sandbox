"""Tests for Session (persistent VM for multi-step execution).

Every execution runs in the REPL wrapper, so the whole suite also checks the REPL.
The tests here are about what a session adds:
- State persistence across exec() calls
- Session lifecycle (idle timeout, close, context manager)
- Error recovery and state preservation
- Edge cases (REPL protocol abuse, language quirks)
- Concurrency (serialization, coexistence with run())
"""

import asyncio
import re

import pytest

from exec_sandbox.exceptions import (
    ErrorCode,
    InvalidInputError,
    SessionClosedError,
    TransientError,
)
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once, skip_on_python_312_subprocess_bug, skip_unless_hwaccel
from tests.instance_support import await_until


# =============================================================================
# TestSessionNormal - Happy path (state persistence is the core value)
# =============================================================================
class TestSessionNormal:
    """Happy path tests for session state persistence and basic functionality."""

    async def test_python_variable_persistence(self, scheduler: Scheduler) -> None:
        """Python variables persist across exec calls."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("x = 42")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("print(x)")
            assert (result.exit_code, result.stdout) == (0, "42\n"), result.stderr

    async def test_repl_state_survives_a_host_reconnect(self, scheduler: Scheduler) -> None:
        """The host drops its connection to the guest agent (a restart of the host side of the
        channel); the next call reconnects, and the REPL in the guest kept its state: the agent
        keeps its ports open across a host close. No public call drops the host connection: the
        test closes the VM's channel."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("x = 42")
            assert setup.exit_code == 0, setup.stderr
            await session._vm.channel.close()  # pyright: ignore[reportPrivateUsage]

            result = await session.exec("print(x)")

            assert result.exit_code == 0, result.stderr
            assert result.stdout.strip() == "42"

    async def test_python_import_persistence(self, scheduler: Scheduler) -> None:
        """Python imports persist across exec calls."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("import json")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec('print(json.dumps({"a": 1}))')
            assert (result.exit_code, result.stdout) == (0, '{"a": 1}\n'), result.stderr

    async def test_python_function_persistence(self, scheduler: Scheduler) -> None:
        """Python function definitions persist across exec calls."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec('def greet(n): return f"hi {n}"')
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec('print(greet("alice"))')
            assert (result.exit_code, result.stdout) == (0, "hi alice\n"), result.stderr

    async def test_python_class_persistence(self, scheduler: Scheduler) -> None:
        """Python class definitions persist across exec calls."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec('class Dog:\n  sound="woof"')
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("print(Dog.sound)")
            assert (result.exit_code, result.stdout) == (0, "woof\n"), result.stderr

    async def test_python_accumulate_state(self, scheduler: Scheduler) -> None:
        """Python state accumulates across multiple exec calls."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("items = []")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            for _ in range(3):
                r = await session.exec("items.append(1)")
                assert r.exit_code == 0, (
                    f"append exec timed out or failed: exit_code={r.exit_code}, stderr={r.stderr!r}"
                )
            result = await session.exec("print(len(items))")
            assert (result.exit_code, result.stdout) == (0, "3\n"), result.stderr

    async def test_js_variable_persistence(self, scheduler: Scheduler) -> None:
        """JavaScript variables persist across exec calls."""
        async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
            setup = await session.exec("var x = 42;")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("console.log(x);")
            assert (result.exit_code, result.stdout) == (0, "42\n"), result.stderr

    async def test_js_function_persistence(self, scheduler: Scheduler) -> None:
        """JavaScript function definitions persist across exec calls."""
        async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
            setup = await session.exec("function greet(n) { return 'hi ' + n; }")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("console.log(greet('bob'));")
            assert (result.exit_code, result.stdout) == (0, "hi bob\n"), result.stderr

    async def test_ts_typed_variable_persistence(self, scheduler: Scheduler) -> None:
        """TypeScript typed variables persist across exec calls."""
        async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
            setup = await session.exec("let x: number = 10;")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("console.log(x + 1);")
            assert (result.exit_code, result.stdout) == (0, "11\n"), result.stderr

    async def test_ts_interface_reuse_across_execs(self, scheduler: Scheduler) -> None:
        """TypeScript interface defined in one exec is usable in the next."""
        async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
            s1 = await session.exec("interface Item { name: string; qty: number }")
            assert s1.exit_code == 0, (
                f"interface exec timed out or failed: exit_code={s1.exit_code}, stderr={s1.stderr!r}"
            )
            s2 = await session.exec("function describe(i: Item): string { return `${i.name}:${i.qty}`; }")
            assert s2.exit_code == 0, (
                f"function exec timed out or failed: exit_code={s2.exit_code}, stderr={s2.stderr!r}"
            )
            result = await session.exec('console.log(describe({ name: "apple", qty: 3 }));')
            assert (result.exit_code, result.stdout) == (0, "apple:3\n"), result.stderr

    async def test_shell_env_var_persistence(self, scheduler: Scheduler) -> None:
        """Shell environment variables persist across exec calls."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            setup = await session.exec("export FOO=bar")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("echo $FOO")
            assert (result.exit_code, result.stdout) == (0, "bar\n"), result.stderr

    async def test_shell_cwd_persistence(self, scheduler: Scheduler) -> None:
        """Shell working directory persists across exec calls."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            setup = await session.exec("cd /tmp")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("pwd")
            assert (result.exit_code, result.stdout) == (0, "/tmp\n"), result.stderr

    async def test_shell_function_persistence(self, scheduler: Scheduler) -> None:
        """Shell function definitions persist across exec calls."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            setup = await session.exec('greet() { echo "hi $1"; }')
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("greet alice")
            assert (result.exit_code, result.stdout) == (0, "hi alice\n"), result.stderr

    async def test_multiple_sequential_execs(self, scheduler: Scheduler) -> None:
        """Multiple sequential execs all succeed, and exec_count counts each one."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            assert session.exec_count == 0
            for i in range(10):
                result = await session.exec(f"print({i})")
                assert result.exit_code == 0, result.stderr
                assert result.stdout == f"{i}\n"
                assert session.exec_count == i + 1

    async def test_env_vars_passed(self, scheduler: Scheduler) -> None:
        """Environment variables are passed to execution."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec(
                'import os; print(os.environ["FOO"])',
                env_vars={"FOO": "bar"},
            )
            assert (result.exit_code, result.stdout) == (0, "bar\n"), result.stderr


# =============================================================================
# TestSessionEdgeCases - Error recovery + state preservation
# =============================================================================
class TestSessionEdgeCases:
    """Tests for error recovery and state preservation."""

    async def test_nonzero_exit_preserves_state(self, scheduler: Scheduler) -> None:
        """State survives across an error."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("x = 42")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("raise ValueError('boom')")
            assert result.exit_code == 1, (
                f"expected ValueError (exit_code=1), got exit_code={result.exit_code}, stderr={result.stderr!r}"
            )
            assert result.stderr.splitlines()[-1] == "ValueError: boom"
            result = await session.exec("print(x)")
            assert (result.exit_code, result.stdout) == (0, "42\n"), result.stderr

    async def test_sys_exit_preserves_state(self, scheduler: Scheduler) -> None:
        """sys.exit() is caught by wrapper, state preserved."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("x = 99")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("import sys; sys.exit(0)")
            assert result.exit_code == 0, result.stderr  # the wrapper catches SystemExit
            result = await session.exec("print(x)")
            assert (result.exit_code, result.stdout) == (0, "99\n"), result.stderr

    async def test_syntax_error_preserves_state(self, scheduler: Scheduler) -> None:
        """SyntaxError doesn't destroy state."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("x = 42")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("def broken(")
            assert result.exit_code == 1, (
                f"expected SyntaxError (exit_code=1), got exit_code={result.exit_code}, stderr={result.stderr!r}"
            )
            assert result.stderr.splitlines()[-1] == "SyntaxError: '(' was never closed"
            result = await session.exec("print(x)")
            assert (result.exit_code, result.stdout) == (0, "42\n"), result.stderr

    async def test_code_validation_error_preserves_session(self, scheduler: Scheduler) -> None:
        """InvalidInputError CODE_INVALID doesn't close the session — caller can retry."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("x = 42")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.CODE_INVALID):
                await session.exec("")  # empty code → CODE_INVALID
            # Session still alive and state preserved
            result = await session.exec("print(x)")
            assert (result.exit_code, result.stdout) == (0, "42\n"), result.stderr

    async def test_overwrite_variable(self, scheduler: Scheduler) -> None:
        """Variables can be overwritten."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("x = 1")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("x = 2; print(x)")
            assert (result.exit_code, result.stdout) == (0, "2\n"), result.stderr

    @skip_unless_hwaccel  # pip install routinely exceeds 940s under TCG
    async def test_session_with_packages(self, scheduler: Scheduler) -> None:
        """Session with pre-installed packages works.

        Requires hwaccel: pip install in a snapshot VM routinely exceeds the
        940s total timeout (3 x 308s attempts) under TCG software emulation,
        especially with concurrent load on CI runners.
        """
        async with await scheduler.new(VmSpec(language=Language.PYTHON, packages=["requests==2.32.3"])) as session:
            result = await session.exec("import requests; print(requests.__version__)")
            assert (result.exit_code, result.stdout) == (0, "2.32.3\n"), result.stderr


# =============================================================================
# TestSessionWeirdCases - REPL protocol abuse & language quirks
# =============================================================================
class TestSessionWeirdCases:
    """Tests for REPL protocol edge cases and language quirks."""

    async def test_fake_sentinel_on_stderr(self, scheduler: Scheduler) -> None:
        """Fake sentinel in user stderr doesn't confuse the real sentinel: the exec ends
        normally and the fake text stays user output (each exec has its own sentinel ID)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec('import sys; sys.stderr.write("__SENTINEL_fake_0__\\n")')
            assert (result.exit_code, result.stderr) == (0, "__SENTINEL_fake_0__\n")
            result = await session.exec('print("alive")')
            assert (result.exit_code, result.stdout) == (0, "alive\n"), result.stderr

    async def test_unicode_variable_names(self, scheduler: Scheduler) -> None:
        """Unicode variable names and multi-byte UTF-8 code work."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Use actual multi-byte UTF-8 (café has 2-byte é) to test byte-count protocol
            setup = await session.exec("café = 42")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("print(café)")
            assert (result.exit_code, result.stdout) == (0, "42\n"), result.stderr

    async def test_os_exit_kills_repl(self, scheduler: Scheduler) -> None:
        """os._exit() kills REPL, next exec works with fresh state."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("x = 42")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            # os._exit() kills the REPL process immediately
            await session.exec("import os; os._exit(1)")
            # REPL respawned, x is gone
            result = await session.exec("print('x' in globals())")
            assert (result.exit_code, result.stdout) == (0, "False\n"), result.stderr
            assert not session.closed

    async def test_background_thread_survives(self, scheduler: Scheduler) -> None:
        """Background daemon thread doesn't interfere with session."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec(
                "import threading, time\n"
                "def bg(): time.sleep(60)\n"
                "t = threading.Thread(target=bg, daemon=True)\n"
                "t.start()"
            )
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec('print("ok")')
            assert (result.exit_code, result.stdout) == (0, "ok\n"), result.stderr

    async def test_subprocess_popen_survives_across_execs(self, scheduler: Scheduler) -> None:
        """subprocess.Popen child survives across exec() calls (VM persists)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Exec 1: spawn a long-lived subprocess, capture its PID
            result = await session.exec("import subprocess\np = subprocess.Popen(['sleep', '60'])\nprint(p.pid)")
            assert result.exit_code == 0
            pid = result.stdout.strip()
            assert pid.isdigit()

            # Exec 2: verify the subprocess is still alive via /proc
            result = await session.exec(f"import os; print(os.path.exists('/proc/{pid}/stat'))")
            assert (result.exit_code, result.stdout) == (0, "True\n"), result.stderr

            # Exec 3: session still works normally
            result = await session.exec('print("ok")')
            assert (result.exit_code, result.stdout) == (0, "ok\n"), result.stderr

    @skip_on_python_312_subprocess_bug
    async def test_daemon_double_fork_survives_across_execs(self, scheduler: Scheduler) -> None:
        """Double-fork daemon survives across exec() calls, reparented to PID 1."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Exec 1: classic double-fork + setsid to create a daemon.
            # The daemon resets PR_SET_DUMPABLE so its /proc entry is visible —
            # the REPL sets dumpable=0 for ptrace protection, and forked children
            # inherit this, making /proc/{pid}/ inaccessible to same-UID processes.
            # After exec() the kernel resets dumpable to 1 automatically, but a
            # pure-fork daemon never exec()s.
            result = await session.exec(
                "import os, time, ctypes\n"
                "pid = os.fork()\n"
                "if pid == 0:\n"
                "    os.setsid()\n"
                "    pid2 = os.fork()\n"
                "    if pid2 == 0:\n"
                "        ctypes.CDLL('libc.so.6').prctl(4, 1, 0, 0, 0)  # PR_SET_DUMPABLE=1\n"
                "        with open('/tmp/daemon.pid', 'w') as f:\n"
                "            f.write(str(os.getpid()))\n"
                "        time.sleep(60)\n"
                "        os._exit(0)\n"
                "    else:\n"
                "        os._exit(0)\n"
                "else:\n"
                "    os.waitpid(pid, 0)\n"
                "    time.sleep(0.5)\n"
                "    print('spawned')"
            )
            assert (result.exit_code, result.stdout) == (0, "spawned\n"), result.stderr

            # Exec 2: verify daemon is alive and reparented to PID 1.
            # Retry briefly — reparenting to PID 1 (init reaping the
            # intermediate child) is async and may race under CI load.
            result = await session.exec(
                "import time\n"
                "with open('/tmp/daemon.pid') as f:\n"
                "    daemon_pid = f.read().strip()\n"
                "ppid = None\n"
                "for _attempt in range(20):\n"
                "    try:\n"
                "        with open(f'/proc/{daemon_pid}/stat') as f:\n"
                "            stat = f.read()\n"
                "        ppid = stat[stat.rfind(')') + 2:].split()[1]\n"
                "        if ppid == '1':\n"
                "            break\n"
                "    except FileNotFoundError:\n"
                "        pass\n"
                "    time.sleep(0.1)\n"
                "if ppid is None:\n"
                "    raise FileNotFoundError(f'/proc/{daemon_pid}/stat not found after retries')\n"
                "print(f'alive ppid={ppid}')"
            )
            assert (result.exit_code, result.stdout) == (0, "alive ppid=1\n"), result.stderr

            # Exec 3: session still works normally
            result = await session.exec('print("ok")')
            assert (result.exit_code, result.stdout) == (0, "ok\n"), result.stderr

    async def test_shell_pipe_in_session(self, scheduler: Scheduler) -> None:
        """Shell pipes work in session."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            result = await session.exec("echo hello | tr a-z A-Z")
            assert (result.exit_code, result.stdout) == (0, "HELLO\n"), result.stderr

    async def test_shell_background_job(self, scheduler: Scheduler) -> None:
        """Shell background job doesn't block session."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            assert (await session.exec("sleep 60 &")).exit_code == 0
            result = await session.exec("echo ok")
            assert (result.exit_code, result.stdout) == (0, "ok\n"), result.stderr

    async def test_shell_bash_array(self, scheduler: Scheduler) -> None:
        """Bash indexed arrays work (fails under busybox ash)."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            result = await session.exec('arr=(one two three); echo "${arr[1]}"')
            assert (result.exit_code, result.stdout) == (0, "two\n"), result.stderr

    async def test_shell_bash_associative_array(self, scheduler: Scheduler) -> None:
        """Bash associative arrays work (fails under busybox ash)."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            result = await session.exec('declare -A map; map[key]="value"; echo "${map[key]}"')
            assert (result.exit_code, result.stdout) == (0, "value\n"), result.stderr

    async def test_shell_bash_trap(self, scheduler: Scheduler) -> None:
        """Bash EXIT trap fires (silent failure under busybox ash)."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            result = await session.exec("(trap 'echo TRAP_FIRED' EXIT; echo body)")
            assert (result.exit_code, result.stdout) == (0, "body\nTRAP_FIRED\n"), result.stderr

    async def test_shell_bash_process_substitution(self, scheduler: Scheduler) -> None:
        """Bash process substitution works (syntax error under busybox ash)."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            result = await session.exec("cat <(echo hello)")
            assert (result.exit_code, result.stdout) == (0, "hello\n"), result.stderr

    async def test_shell_bash_regex(self, scheduler: Scheduler) -> None:
        """Bash [[ =~ ]] regex matching works (syntax error under busybox ash)."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            result = await session.exec('[[ "2025-01-15" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && echo MATCH')
            assert (result.exit_code, result.stdout) == (0, "MATCH\n"), result.stderr

    async def test_shell_bash_here_string(self, scheduler: Scheduler) -> None:
        """Bash here-strings work (syntax error under busybox ash)."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            result = await session.exec('read -r word <<< "hello world"; echo "$word"')
            assert (result.exit_code, result.stdout) == (0, "hello world\n"), result.stderr

    async def test_stderr_no_trailing_newline(self, scheduler: Scheduler) -> None:
        """Stderr without trailing newline doesn't prevent sentinel detection.

        When user code writes to stderr without \\n, the kernel pipe may deliver
        user text + sentinel concatenated on the same line. The guest agent must
        find the sentinel anywhere in the line (not just at the start).
        """
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Write to stderr WITHOUT trailing newline — sentinel may concatenate
            result = await session.exec('import sys; sys.stderr.write("no-newline")')
            assert (result.exit_code, result.stderr) == (0, "no-newline")
            # Session still alive
            result = await session.exec('print("alive")')
            assert (result.exit_code, result.stdout) == (0, "alive\n"), result.stderr

    async def test_env_var_key_with_special_chars(self, scheduler: Scheduler) -> None:
        """Env var keys with quotes/backslashes are escaped for Python/JS: a key like FOO'BAR
        must not give broken code such as environ['FOO'BAR'] (unmatched quote)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec(
                'import os; print(os.environ["FOO\'BAR"])',
                env_vars={"FOO'BAR": "works"},
            )
            assert (result.exit_code, result.stdout) == (0, "works\n"), result.stderr

    async def test_shell_invalid_env_var_key_skipped(self, scheduler: Scheduler) -> None:
        """Shell env vars with non-identifier keys are skipped (not crash); the valid keys
        are set.

        Shell export syntax requires POSIX identifier keys ([a-zA-Z_][a-zA-Z0-9_]*).
        Keys with spaces or special chars would produce broken syntax.
        """
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            # Key "BAD KEY" has a space — invalid for shell export
            result = await session.exec(
                'echo "$GOOD_KEY"',
                env_vars={"BAD KEY": "value", "GOOD_KEY": "hello"},
            )
            assert (result.exit_code, result.stdout) == (0, "hello\n"), result.stderr

    async def test_deep_recursion(self, scheduler: Scheduler) -> None:
        """RecursionError doesn't kill session."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec("def f(): f()\ntry:\n    f()\nexcept RecursionError:\n    pass")
            assert result.exit_code == 0, result.stderr  # RecursionError caught - session alive
            result = await session.exec('print("alive")')
            assert (result.exit_code, result.stdout) == (0, "alive\n"), result.stderr


# =============================================================================
# TestSessionOutOfBounds - Resource exhaustion & lifecycle violations
# =============================================================================
@pytest.fixture
def short_idle_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduler.new() takes an idle_timeout_seconds of 1 s (the public minimum is 10 s): the
    idle tests then wait seconds, not tens of seconds. The session arms its real timer."""
    monkeypatch.setattr("exec_sandbox.scheduler.MIN_SESSION_IDLE_TIMEOUT_SECONDS", 1)


class TestSessionOutOfBounds:
    """Tests for lifecycle violations and resource limits."""

    async def test_idle_timeout_auto_closes(self, scheduler: Scheduler, short_idle_minimum: None) -> None:
        """A session with no call for idle_timeout_seconds closes itself: closed turns True,
        and the next exec raises SessionClosedError."""
        session = await scheduler.new(VmSpec(language=Language.PYTHON), idle_timeout_seconds=1)
        try:
            result = await session.exec("1")
            assert result.exit_code == 0, result.stderr
            assert not session.closed
            await await_until(lambda: session.closed)  # the 1 s deadline
            with pytest.raises(SessionClosedError):
                await session.exec("print(1)")
        finally:
            await session.close()

    async def test_idle_timeout_reset_on_exec(self, scheduler: Scheduler, short_idle_minimum: None) -> None:
        """Each exec re-arms the idle deadline at its end: execs 1.2 s apart keep a 2 s session
        open for 3.6 s, longer than one deadline."""
        session = await scheduler.new(VmSpec(language=Language.PYTHON), idle_timeout_seconds=2)
        try:
            # The first exec starts the REPL (0.1-1 s, 3-5 s under TCG): no deadline runs during a call
            assert (await session.exec("1")).exit_code == 0
            for i in range(3):
                await asyncio.sleep(1.2)  # idle time under the deadline
                result = await session.exec(f"print({i})")
                assert (result.exit_code, result.stdout) == (0, f"{i}\n"), result.stderr
            assert not session.closed
        finally:
            await session.close()

    @pytest.mark.slow
    async def test_sigkill_closes_session(self, scheduler: Scheduler) -> None:
        """SIGKILL produces terminal exit 137 and closes the session.

        Slow under TCG: terminal-session correctness test with no timing
        assertions — works under TCG but too slow for the default suite.
        """
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            setup = await session.exec("x = 42")
            assert setup.exit_code == 0, (
                f"setup exec timed out or failed: exit_code={setup.exit_code}, stderr={setup.stderr!r}"
            )
            result = await session.exec("import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
            assert result.exit_code == 137  # 128 + 9 (SIGKILL)
            assert session.closed
            with pytest.raises(SessionClosedError):
                await session.exec("print('must_not_run')")

    @pytest.mark.slow
    async def test_user_exit_137_closes_session(self, scheduler: Scheduler) -> None:
        """A voluntary sys.exit(137) is treated as terminal, like SIGKILL.

        Pins the documented reservation: numeric 137 retires the session even
        when it did not come from a signal (guard/OOM-killer/user alike).
        """
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec("import sys; sys.exit(137)")
            assert result.exit_code == 137
            assert session.closed
            with pytest.raises(SessionClosedError):
                await session.exec("print('must_not_run')")

    @staticmethod
    def _end_repl_before_result(end: str) -> str:
        """Shell code that ends the REPL (with `end`) before the result of this exec reaches
        the host, so the next exec finds it gone. The REPL prints the result marker with
        `printf`: a function of that name ends the REPL instead, and a child of it prints
        the marker once the REPL is reaped (`kill -0` fails)."""
        return (
            "printf() { case $1 in __SENTINEL_*) "
            '(while kill -0 $$ 2>/dev/null; do :; done; builtin printf "$@") & '
            f"{end} ;; "
            '*) builtin printf "$@" ;; esac; }'
        )

    async def test_repl_killed_between_calls_closes_session(self, scheduler: Scheduler) -> None:
        """Exit 137 of the REPL between two calls retires the VM: the next exec raises
        TransientError at once, then the session is closed.

        The PID 1 reaper of the guest must not take the exit status of the REPL first:
        the guest would then start a new REPL, and the next exec would run in it."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            first = await session.exec(self._end_repl_before_result("kill -9 $$"))
            assert first.exit_code == 0

            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE, match="retired"):
                await session.exec("echo must_not_run")
            assert session.closed

    async def test_repl_exit_between_calls_resets_state(self, scheduler: Scheduler) -> None:
        """An exit of the REPL between two calls with a code other than 137: the next
        exec runs in a new REPL (the state is lost) and the session stays open."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            first = await session.exec("kept=yes; " + self._end_repl_before_result("exit 3"))
            assert first.exit_code == 0

            second = await session.exec('echo "kept=$kept"')
            assert (second.exit_code, second.stdout.strip()) == (0, "kept=")
            assert not session.closed

    @skip_unless_hwaccel
    async def test_exec_timeout_preserves_session(self, scheduler: Scheduler) -> None:
        """Execution timeout returns error result but session stays alive, with a new REPL
        (the state is lost).

        Requires hwaccel: 2s timeout — TCG (~5-8x slower) cannot complete
        REPL boot and timeout semantics within the deadline.
        """
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            assert (await session.exec("x = 1")).exit_code == 0
            # Timeout returns result with exit_code=-1 (not an exception)
            result = await session.exec("import time; time.sleep(60)", timeout_seconds=2)
            assert (result.exit_code, result.stderr) == (-1, "[timeout_error] Execution timeout after 2s")
            # Session still alive — REPL respawned, fresh state
            result = await session.exec("print('x' in globals())")
            assert (result.exit_code, result.stdout) == (0, "False\n"), result.stderr

    async def test_many_concurrent_sessions(self, scheduler: Scheduler) -> None:
        """Multiple sessions can coexist."""
        sessions = []
        try:
            for _ in range(3):
                s = await scheduler.new(VmSpec(language=Language.PYTHON))
                sessions.append(s)

            # All sessions work independently, each on its own VM
            assert len({s.vm_id for s in sessions}) == 3
            for i, s in enumerate(sessions):
                result = await s.exec(f"print({i})")
                assert (result.exit_code, result.stdout) == (0, f"{i}\n"), result.stderr
        finally:
            for s in sessions:
                await s.close()


# =============================================================================
# TestSessionConcurrency - Serialization & coexistence
# =============================================================================
class TestSessionConcurrency:
    """Tests for exec serialization and coexistence with run()."""

    async def test_concurrent_execs_serialized(self, scheduler: Scheduler) -> None:
        """Concurrent exec() calls are serialized (no interleaved output)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            results = await asyncio.gather(
                session.exec("print('a' * 100)"),
                session.exec("print('b' * 100)"),
            )
            # Both succeed, each with its own output only (serialized by exec_lock)
            assert [(r.exit_code, r.stdout) for r in results] == [(0, "a" * 100 + "\n"), (0, "b" * 100 + "\n")]

    async def test_session_plus_run_coexist(self, scheduler: Scheduler) -> None:
        """Session and run() work simultaneously (separate VMs)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Run both in parallel
            session_result, run_result = await asyncio.gather(
                session.exec('print("session")'),
                run_once(scheduler, VmSpec(language=Language.PYTHON), 'print("run")'),
            )
            assert (session_result.exit_code, session_result.stdout) == (0, "session\n"), session_result.stderr
            assert (run_result.exit_code, run_result.stdout) == (0, "run\n"), run_result.stderr


# =============================================================================
# TestWorkingDirectory - REPL cwd is /home/user
# =============================================================================
class TestWorkingDirectory:
    """Verify REPL working directory is /home/user for all languages."""

    async def test_python_cwd(self, scheduler: Scheduler) -> None:
        """Python REPL starts in /home/user."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec("import os; print(os.getcwd())")
            assert (result.exit_code, result.stdout) == (0, "/home/user\n"), result.stderr

    async def test_javascript_cwd(self, scheduler: Scheduler) -> None:
        """JavaScript REPL starts in /home/user."""
        async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
            result = await session.exec("console.log(process.cwd());")
            assert (result.exit_code, result.stdout) == (0, "/home/user\n"), result.stderr

    async def test_shell_cwd(self, scheduler: Scheduler) -> None:
        """Shell REPL starts in /home/user."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            result = await session.exec("pwd")
            assert (result.exit_code, result.stdout) == (0, "/home/user\n"), result.stderr


# =============================================================================
# TestLazyCloudpickle - Deferred cloudpickle import
# =============================================================================
@pytest.mark.slow
class TestLazyCloudpickle:
    """Tests for lazy cloudpickle import in REPL.

    The REPL imports cloudpickle at the first multiprocessing.Process.start(), not at start.

    Slow under TCG: pure correctness tests for cloudpickle/multiprocessing
    import — no timing assertions, but TCG boot overhead makes them too
    slow for the default suite.
    """

    async def test_multiprocessing_process_works(self, scheduler: Scheduler) -> None:
        """multiprocessing.Process.start() triggers lazy cloudpickle patch."""
        code = """
import multiprocessing
import os

def worker():
    print(f"worker pid={os.getpid()}")

p = multiprocessing.Process(target=worker)
p.start()
p.join(timeout=10)
print(f"exit_code={p.exitcode}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert re.fullmatch(r"worker pid=\d+\nexit_code=0\n", result.stdout), result.stdout

    async def test_multiprocessing_pool_works(self, scheduler: Scheduler) -> None:
        """multiprocessing.Pool internally calls Process.start() — triggers patch."""
        code = """
import multiprocessing

def square(x):
    return x * x

with multiprocessing.Pool(2) as pool:
    results = pool.map(square, [1, 2, 3, 4])
print(results)
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "[1, 4, 9, 16]\n"), result.stderr

    async def test_cloudpickle_not_imported_without_multiprocessing(self, scheduler: Scheduler) -> None:
        """Verify cloudpickle is NOT imported for simple scripts."""
        code = """
import sys
print('cloudpickle' in sys.modules)
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "False\n"), result.stderr

    async def test_cloudpickle_imported_after_process_start(self, scheduler: Scheduler) -> None:
        """Verify cloudpickle IS imported after Process.start()."""
        code = """
import multiprocessing, sys, os

def noop():
    pass

p = multiprocessing.Process(target=noop)
p.start()
p.join(timeout=10)
print('cloudpickle' in sys.modules)
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "True\n"), result.stderr

"""Tests for Session (persistent VM for multi-step execution).

Since ALL execution now uses the REPL wrapper, the entire existing test suite
validates the REPL mechanism implicitly. Session-specific tests focus on:
- State persistence across exec() calls
- Session lifecycle (idle timeout, close, context manager)
- Error recovery and state preservation
- Edge cases (REPL protocol abuse, language quirks)
- Concurrency (serialization, coexistence with run())
"""

import asyncio

import pytest

from exec_sandbox.exceptions import SessionClosedError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from tests.conftest import skip_unless_hwaccel


# =============================================================================
# TestSessionNormal - Happy path (state persistence is the core value)
# =============================================================================
class TestSessionNormal:
    """Happy path tests for session state persistence and basic functionality."""

    async def test_session_returns_session_object(self, scheduler: Scheduler) -> None:
        """scheduler.session() returns a Session object."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            assert session is not None
            assert not session.closed

    async def test_basic_exec(self, scheduler: Scheduler) -> None:
        """Basic exec returns output and exit code."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec('print("hi")')
            assert result.exit_code == 0
            assert "hi" in result.stdout

    async def test_python_variable_persistence(self, scheduler: Scheduler) -> None:
        """Python variables persist across exec calls."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("x = 42")
            result = await session.exec("print(x)")
            assert result.exit_code == 0
            assert "42" in result.stdout

    async def test_python_import_persistence(self, scheduler: Scheduler) -> None:
        """Python imports persist across exec calls."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("import json")
            result = await session.exec('print(json.dumps({"a": 1}))')
            assert result.exit_code == 0
            assert '{"a": 1}' in result.stdout

    async def test_python_function_persistence(self, scheduler: Scheduler) -> None:
        """Python function definitions persist across exec calls."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec('def greet(n): return f"hi {n}"')
            result = await session.exec('print(greet("alice"))')
            assert result.exit_code == 0
            assert "hi alice" in result.stdout

    async def test_python_class_persistence(self, scheduler: Scheduler) -> None:
        """Python class definitions persist across exec calls."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec('class Dog:\n  sound="woof"')
            result = await session.exec("print(Dog.sound)")
            assert result.exit_code == 0
            assert "woof" in result.stdout

    async def test_python_accumulate_state(self, scheduler: Scheduler) -> None:
        """Python state accumulates across multiple exec calls."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("items = []")
            await session.exec("items.append(1)")
            await session.exec("items.append(1)")
            await session.exec("items.append(1)")
            result = await session.exec("print(len(items))")
            assert result.exit_code == 0
            assert "3" in result.stdout

    async def test_js_variable_persistence(self, scheduler: Scheduler) -> None:
        """JavaScript variables persist across exec calls."""
        async with await scheduler.session(language=Language.JAVASCRIPT) as session:
            await session.exec("var x = 42;")
            result = await session.exec("console.log(x);")
            assert result.exit_code == 0
            assert "42" in result.stdout

    async def test_js_function_persistence(self, scheduler: Scheduler) -> None:
        """JavaScript function definitions persist across exec calls."""
        async with await scheduler.session(language=Language.JAVASCRIPT) as session:
            await session.exec("function greet(n) { return 'hi ' + n; }")
            result = await session.exec("console.log(greet('bob'));")
            assert result.exit_code == 0
            assert "hi bob" in result.stdout

    async def test_shell_env_var_persistence(self, scheduler: Scheduler) -> None:
        """Shell environment variables persist across exec calls."""
        async with await scheduler.session(language=Language.RAW) as session:
            await session.exec("export FOO=bar")
            result = await session.exec("echo $FOO")
            assert result.exit_code == 0
            assert "bar" in result.stdout

    async def test_shell_cwd_persistence(self, scheduler: Scheduler) -> None:
        """Shell working directory persists across exec calls."""
        async with await scheduler.session(language=Language.RAW) as session:
            await session.exec("cd /tmp")
            result = await session.exec("pwd")
            assert result.exit_code == 0
            assert "/tmp" in result.stdout

    async def test_shell_function_persistence(self, scheduler: Scheduler) -> None:
        """Shell function definitions persist across exec calls."""
        async with await scheduler.session(language=Language.RAW) as session:
            await session.exec('greet() { echo "hi $1"; }')
            result = await session.exec("greet alice")
            assert result.exit_code == 0
            assert "hi alice" in result.stdout

    async def test_multiple_sequential_execs(self, scheduler: Scheduler) -> None:
        """Multiple sequential execs all succeed."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            for i in range(10):
                result = await session.exec(f"print({i})")
                assert result.exit_code == 0
                assert str(i) in result.stdout
            assert session.exec_count == 10

    async def test_exec_count_increments(self, scheduler: Scheduler) -> None:
        """exec_count tracks successful executions."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            assert session.exec_count == 0
            await session.exec("print(1)")
            assert session.exec_count == 1
            await session.exec("print(2)")
            assert session.exec_count == 2
            await session.exec("print(3)")
            assert session.exec_count == 3

    async def test_streaming_callbacks(self, scheduler: Scheduler) -> None:
        """on_stdout callback receives streaming chunks."""
        chunks: list[str] = []
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec(
                'print("hello world")',
                on_stdout=chunks.append,
            )
            assert result.exit_code == 0
            assert "hello world" in "".join(chunks)

    async def test_stderr_streaming(self, scheduler: Scheduler) -> None:
        """on_stderr callback receives stderr chunks."""
        chunks: list[str] = []
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec(
                'import sys; sys.stderr.write("oops\\n")',
                on_stderr=chunks.append,
            )
            assert "oops" in "".join(chunks)

    async def test_env_vars_passed(self, scheduler: Scheduler) -> None:
        """Environment variables are passed to execution."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec(
                'import os; print(os.environ["FOO"])',
                env_vars={"FOO": "bar"},
            )
            assert result.exit_code == 0
            assert "bar" in result.stdout

    async def test_timing_populated(self, scheduler: Scheduler) -> None:
        """Timing breakdown is populated."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec("print(1)")
            assert result.timing.execute_ms > 0
            assert result.timing.total_ms > 0

    async def test_closed_starts_false(self, scheduler: Scheduler) -> None:
        """Fresh session is not closed."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            assert session.closed is False

    async def test_context_manager(self, scheduler: Scheduler) -> None:
        """Context manager closes session on exit."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            assert session.vm_id  # VM is assigned
            assert not session.closed
        assert session.closed


# =============================================================================
# TestSessionEdgeCases - Error recovery + state preservation
# =============================================================================
class TestSessionEdgeCases:
    """Tests for error recovery and state preservation."""

    async def test_nonzero_exit_preserves_session(self, scheduler: Scheduler) -> None:
        """Non-zero exit code doesn't close the session."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec("raise ValueError('test')")
            assert result.exit_code != 0
            # Session still alive
            result = await session.exec('print("alive")')
            assert result.exit_code == 0
            assert "alive" in result.stdout

    async def test_nonzero_exit_preserves_state(self, scheduler: Scheduler) -> None:
        """State survives across an error."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("x = 42")
            result = await session.exec("raise ValueError('boom')")
            assert result.exit_code != 0
            result = await session.exec("print(x)")
            assert result.exit_code == 0
            assert "42" in result.stdout

    async def test_sys_exit_preserves_state(self, scheduler: Scheduler) -> None:
        """sys.exit() is caught by wrapper, state preserved."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("x = 99")
            result = await session.exec("import sys; sys.exit(0)")
            # SystemExit caught by wrapper
            result = await session.exec("print(x)")
            assert result.exit_code == 0
            assert "99" in result.stdout

    async def test_syntax_error_preserves_state(self, scheduler: Scheduler) -> None:
        """SyntaxError doesn't destroy state."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("x = 42")
            result = await session.exec("def broken(")
            assert result.exit_code != 0
            result = await session.exec("print(x)")
            assert result.exit_code == 0
            assert "42" in result.stdout

    async def test_overwrite_variable(self, scheduler: Scheduler) -> None:
        """Variables can be overwritten."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("x = 1")
            result = await session.exec("x = 2; print(x)")
            assert result.exit_code == 0
            assert "2" in result.stdout

    async def test_close_idempotent(self, scheduler: Scheduler) -> None:
        """close() can be called multiple times."""
        session = await scheduler.session(language=Language.PYTHON)
        await session.close()
        await session.close()  # No error

    async def test_rapid_execs(self, scheduler: Scheduler) -> None:
        """Multiple rapid execs all succeed."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            for i in range(5):
                result = await session.exec(f"print({i})")
                assert result.exit_code == 0

    async def test_session_with_packages(self, scheduler: Scheduler) -> None:
        """Session with pre-installed packages works."""
        async with await scheduler.session(
            language=Language.PYTHON,
            packages=["requests==2.32.3"],
        ) as session:
            result = await session.exec("import requests; print(requests.__version__)")
            assert result.exit_code == 0
            assert "2.32" in result.stdout


# =============================================================================
# TestSessionWeirdCases - REPL protocol abuse & language quirks
# =============================================================================
class TestSessionWeirdCases:
    """Tests for REPL protocol edge cases and language quirks."""

    async def test_fake_sentinel_on_stderr(self, scheduler: Scheduler) -> None:
        """Fake sentinel in user stderr doesn't confuse the real sentinel."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec('import sys; sys.stderr.write("__SENTINEL_fake_0__\\n")')
            # Should complete normally (unique sentinel ID prevents confusion)
            result = await session.exec('print("alive")')
            assert result.exit_code == 0
            assert "alive" in result.stdout

    async def test_unicode_variable_names(self, scheduler: Scheduler) -> None:
        """Unicode variable names and multi-byte UTF-8 code work."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            # Use actual multi-byte UTF-8 (café has 2-byte é) to test byte-count protocol
            await session.exec("café = 42")
            result = await session.exec("print(café)")
            assert result.exit_code == 0
            assert "42" in result.stdout

    async def test_os_exit_kills_repl(self, scheduler: Scheduler) -> None:
        """os._exit() kills REPL, next exec works with fresh state."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("x = 42")
            # os._exit() kills the REPL process immediately
            result = await session.exec("import os; os._exit(1)")
            # REPL respawned, x is gone
            result = await session.exec('print("fresh")')
            assert result.exit_code == 0
            assert "fresh" in result.stdout

    async def test_background_thread_survives(self, scheduler: Scheduler) -> None:
        """Background daemon thread doesn't interfere with session."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec(
                "import threading, time\n"
                "def bg(): time.sleep(60)\n"
                "t = threading.Thread(target=bg, daemon=True)\n"
                "t.start()"
            )
            result = await session.exec('print("ok")')
            assert result.exit_code == 0
            assert "ok" in result.stdout

    async def test_shell_pipe_in_session(self, scheduler: Scheduler) -> None:
        """Shell pipes work in session."""
        async with await scheduler.session(language=Language.RAW) as session:
            result = await session.exec("echo hello | tr a-z A-Z")
            assert result.exit_code == 0
            assert "HELLO" in result.stdout

    async def test_shell_background_job(self, scheduler: Scheduler) -> None:
        """Shell background job doesn't block session."""
        async with await scheduler.session(language=Language.RAW) as session:
            await session.exec("sleep 60 &")
            result = await session.exec("echo ok")
            assert result.exit_code == 0
            assert "ok" in result.stdout

    async def test_stderr_no_trailing_newline(self, scheduler: Scheduler) -> None:
        """Stderr without trailing newline doesn't prevent sentinel detection.

        When user code writes to stderr without \\n, the kernel pipe may deliver
        user text + sentinel concatenated on the same line. The guest agent must
        find the sentinel anywhere in the line (not just at the start).
        """
        async with await scheduler.session(language=Language.PYTHON) as session:
            # Write to stderr WITHOUT trailing newline — sentinel may concatenate
            result = await session.exec('import sys; sys.stderr.write("no-newline")')
            assert result.exit_code == 0
            assert "no-newline" in result.stderr
            # Session still alive
            result = await session.exec('print("alive")')
            assert result.exit_code == 0
            assert "alive" in result.stdout

    async def test_env_var_key_with_special_chars(self, scheduler: Scheduler) -> None:
        """Env var keys with quotes/backslashes are escaped for Python/JS.

        Before the fix, a key like FOO'BAR would produce broken code:
        environ['FOO'BAR'] (unmatched quote).
        """
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec(
                'import os; print(os.environ["FOO\'BAR"])',
                env_vars={"FOO'BAR": "works"},
            )
            assert result.exit_code == 0
            assert "works" in result.stdout

    async def test_shell_invalid_env_var_key_skipped(self, scheduler: Scheduler) -> None:
        """Shell env vars with non-identifier keys are skipped (not crash).

        Shell export syntax requires POSIX identifier keys ([a-zA-Z_][a-zA-Z0-9_]*).
        Keys with spaces or special chars would produce broken syntax.
        """
        async with await scheduler.session(language=Language.RAW) as session:
            # Key "BAD KEY" has a space — invalid for shell export
            # Should not crash, just skip the invalid key
            result = await session.exec(
                "echo ok",
                env_vars={"BAD KEY": "value", "GOOD_KEY": "hello"},
            )
            assert result.exit_code == 0
            assert "ok" in result.stdout

    async def test_deep_recursion(self, scheduler: Scheduler) -> None:
        """RecursionError doesn't kill session."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec("def f(): f()\ntry:\n    f()\nexcept RecursionError:\n    pass")
            # RecursionError caught - session alive
            result = await session.exec('print("alive")')
            assert result.exit_code == 0
            assert "alive" in result.stdout


# =============================================================================
# TestSessionOutOfBounds - Resource exhaustion & lifecycle violations
# =============================================================================
class TestSessionOutOfBounds:
    """Tests for lifecycle violations and resource limits."""

    async def test_exec_after_close(self, scheduler: Scheduler) -> None:
        """exec() after close() raises SessionClosedError."""
        session = await scheduler.session(language=Language.PYTHON)
        await session.close()
        with pytest.raises(SessionClosedError):
            await session.exec("print(1)")

    async def test_idle_timeout_auto_closes(self, scheduler: Scheduler) -> None:
        """Session auto-closes after idle timeout."""
        session = await scheduler.session(
            language=Language.PYTHON,
            idle_timeout_seconds=10,  # Use minimum allowed
        )
        try:
            # Override the idle timeout to 1s for testing
            session._idle_timeout_seconds = 1
            session._reset_idle_timer()
            await asyncio.sleep(2)
            with pytest.raises(SessionClosedError):
                await session.exec("print(1)")
        finally:
            await session.close()

    async def test_idle_timeout_reset_on_exec(self, scheduler: Scheduler) -> None:
        """Idle timer resets on each exec."""
        session = await scheduler.session(
            language=Language.PYTHON,
            idle_timeout_seconds=10,
        )
        try:
            session._idle_timeout_seconds = 2
            session._reset_idle_timer()
            # Exec at ~1s resets the 2s timer
            await asyncio.sleep(1)
            result = await session.exec("print(1)")
            assert result.exit_code == 0
            # Should still work at ~2.5s from start (timer was reset at ~1s)
            await asyncio.sleep(0.5)
            result = await session.exec("print(2)")
            assert result.exit_code == 0
        finally:
            await session.close()

    @skip_unless_hwaccel
    async def test_exec_timeout_preserves_session(self, scheduler: Scheduler) -> None:
        """Execution timeout returns error result but session stays alive."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            # Timeout returns result with exit_code=-1 (not an exception)
            result = await session.exec("import time; time.sleep(60)", timeout_seconds=2)
            assert result.exit_code == -1
            assert "timeout" in result.stderr.lower()
            # Session still alive — REPL respawned, fresh state
            result = await session.exec('print("alive")')
            assert result.exit_code == 0
            assert "alive" in result.stdout

    async def test_many_concurrent_sessions(self, scheduler: Scheduler) -> None:
        """Multiple sessions can coexist."""
        sessions = []
        try:
            for _ in range(3):
                s = await scheduler.session(language=Language.PYTHON)
                sessions.append(s)

            # All sessions work independently
            for i, s in enumerate(sessions):
                result = await s.exec(f"print({i})")
                assert result.exit_code == 0
                assert str(i) in result.stdout
        finally:
            for s in sessions:
                await s.close()


# =============================================================================
# TestSessionReconnect - Guest agent 12s idle timeout
# =============================================================================
@skip_unless_hwaccel
class TestSessionReconnect:
    """Tests for state persistence across guest agent reconnect cycles.

    Guest agent drops connection after 12s idle (READ_TIMEOUT_MS=12000).
    REPL subprocess must survive (guest agent process stays alive).
    """

    @pytest.mark.slow
    async def test_state_survives_reconnect(self, scheduler: Scheduler) -> None:
        """State persists across guest agent reconnect (~13s idle)."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("x = 42")
            # Wait for guest agent reconnect cycle
            await asyncio.sleep(13)
            result = await session.exec("print(x)")
            assert result.exit_code == 0
            assert "42" in result.stdout

    @pytest.mark.slow
    async def test_multiple_reconnects_preserve_state(self, scheduler: Scheduler) -> None:
        """State persists across multiple reconnect cycles."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.exec("x = 1")
            await asyncio.sleep(13)
            await session.exec("x += 1")
            await asyncio.sleep(13)
            result = await session.exec("print(x)")
            assert result.exit_code == 0
            assert "2" in result.stdout


# =============================================================================
# TestSessionConcurrency - Serialization & coexistence
# =============================================================================
class TestSessionConcurrency:
    """Tests for exec serialization and coexistence with run()."""

    async def test_concurrent_execs_serialized(self, scheduler: Scheduler) -> None:
        """Concurrent exec() calls are serialized (no interleaved output)."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            results = await asyncio.gather(
                session.exec("print('a' * 100)"),
                session.exec("print('b' * 100)"),
            )
            # Both should succeed (serialized by exec_lock)
            for r in results:
                assert r.exit_code == 0

    async def test_session_plus_run_coexist(self, scheduler: Scheduler) -> None:
        """Session and run() work simultaneously (separate VMs)."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            # Run both in parallel
            session_result, run_result = await asyncio.gather(
                session.exec('print("session")'),
                scheduler.run(code='print("run")', language=Language.PYTHON),
            )
            assert session_result.exit_code == 0
            assert "session" in session_result.stdout
            assert run_result.exit_code == 0
            assert "run" in run_result.stdout


# =============================================================================
# TestWorkingDirectory - REPL cwd is /home/user
# =============================================================================
class TestWorkingDirectory:
    """Verify REPL working directory is /home/user for all languages."""

    async def test_python_cwd(self, scheduler: Scheduler) -> None:
        """Python REPL starts in /home/user."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec("import os; print(os.getcwd())")
            assert result.exit_code == 0
            assert "/home/user" in result.stdout

    async def test_javascript_cwd(self, scheduler: Scheduler) -> None:
        """JavaScript REPL starts in /home/user."""
        async with await scheduler.session(language=Language.JAVASCRIPT) as session:
            result = await session.exec("console.log(process.cwd());")
            assert result.exit_code == 0
            assert "/home/user" in result.stdout

    async def test_shell_cwd(self, scheduler: Scheduler) -> None:
        """Shell REPL starts in /home/user."""
        async with await scheduler.session(language=Language.RAW) as session:
            result = await session.exec("pwd")
            assert result.exit_code == 0
            assert "/home/user" in result.stdout

    async def test_run_python_cwd(self, scheduler: Scheduler) -> None:
        """scheduler.run() also uses /home/user as cwd."""
        result = await scheduler.run(
            code="import os; print(os.getcwd())",
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert "/home/user" in result.stdout

    async def test_run_shell_cwd(self, scheduler: Scheduler) -> None:
        """scheduler.run() shell also uses /home/user as cwd."""
        result = await scheduler.run(
            code="pwd",
            language=Language.RAW,
        )
        assert result.exit_code == 0
        assert "/home/user" in result.stdout

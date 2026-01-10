"""Edge case, weird, and out-of-bounds tests for code execution.

Tests that the execution system handles unusual inputs gracefully:
1. Edge cases: Empty code, large output, special characters
2. Weird cases: Binary data, null bytes, infinite loops
3. Out of bounds: Memory exhaustion, deep recursion
4. Error cases: Syntax errors, import errors, runtime errors
"""

from pathlib import Path

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# Images directory - relative to repo root
images_dir = Path(__file__).parent.parent / "images" / "dist"


# =============================================================================
# Edge Cases: Unusual but valid inputs
# =============================================================================
class TestEdgeCases:
    """Edge cases that should work but might break naive implementations."""

    async def test_empty_code(self) -> None:
        """Empty code string is rejected by guest-agent validation."""
        config = SchedulerConfig(images_dir=images_dir)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code="",
                language=Language.PYTHON,
            )

            # Empty code is rejected with validation error (exit_code=-1)
            assert result.exit_code == -1
            assert "Code cannot be empty" in result.stderr

    async def test_whitespace_only_code(self) -> None:
        """Whitespace-only code is rejected by guest-agent validation."""
        config = SchedulerConfig(images_dir=images_dir)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code="   \n\n\t\t\n   ",
                language=Language.PYTHON,
            )

            # Whitespace-only code is rejected (trimmed = empty)
            assert result.exit_code == -1
            assert "Code cannot be empty" in result.stderr

    async def test_comment_only_code(self) -> None:
        """Comment-only code executes without error."""
        config = SchedulerConfig(images_dir=images_dir)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code="# This is just a comment\n# Another comment",
                language=Language.PYTHON,
            )

            assert result.exit_code == 0
            assert result.stdout == ""

    async def test_large_output_1mb(self) -> None:
        """Code producing ~1MB of output."""
        config = SchedulerConfig(images_dir=images_dir)

        # Generate ~1MB of output (1000 lines of 1000 chars each)
        code = """
for i in range(1000):
    print('x' * 1000)
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=60,
            )

            assert result.exit_code == 0
            # Should have substantial output (may be truncated)
            assert len(result.stdout) > 100000  # At least 100KB

    async def test_many_lines_output(self) -> None:
        """Code producing many small lines of output."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
for i in range(10000):
    print(i)
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=30,
            )

            assert result.exit_code == 0
            # Should contain first and last numbers
            assert "0" in result.stdout
            assert "9999" in result.stdout

    async def test_very_long_single_line(self) -> None:
        """Code with a very long single line."""
        config = SchedulerConfig(images_dir=images_dir)

        # 10KB string
        long_string = "x" * 10000
        code = f"print('{long_string}')"

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code == 0
            assert "x" * 100 in result.stdout  # At least some of it

    async def test_special_characters_in_output(self) -> None:
        """Code outputting special characters."""
        config = SchedulerConfig(images_dir=images_dir)

        code = r"""
print("Tab:\there")
print("Newline in string: line1\nline2")
print("Backslash: \\")
print("Quote: \"hello\"")
print("Unicode: café ñ 你好 🎉")
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code == 0
            assert "Tab:" in result.stdout
            assert "café" in result.stdout
            assert "你好" in result.stdout

    async def test_ansi_escape_codes(self) -> None:
        """Code outputting ANSI escape codes."""
        config = SchedulerConfig(images_dir=images_dir)

        code = r"""
print("\033[31mRed text\033[0m")
print("\033[1mBold text\033[0m")
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code == 0
            # ANSI codes should pass through
            assert "Red text" in result.stdout

    async def test_rapid_stdout_stderr_interleaving(self) -> None:
        """Rapid alternating stdout/stderr output."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import sys
for i in range(100):
    print(f"out{i}")
    print(f"err{i}", file=sys.stderr)
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code == 0
            assert "out0" in result.stdout
            assert "out99" in result.stdout
            assert "err0" in result.stderr
            assert "err99" in result.stderr


# =============================================================================
# Weird Cases: Unusual behavior that should be handled gracefully
# =============================================================================
class TestWeirdCases:
    """Weird inputs that might cause problems."""

    async def test_null_bytes_in_output(self) -> None:
        """Code outputting null bytes."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import sys
sys.stdout.buffer.write(b"before\\x00after\\n")
sys.stdout.buffer.flush()
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            # Should handle null bytes without crashing
            assert result.exit_code == 0
            assert "before" in result.stdout or "after" in result.stdout

    async def test_binary_data_in_output(self) -> None:
        """Code outputting binary data."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import sys
# Write some binary data
sys.stdout.buffer.write(bytes(range(256)))
sys.stdout.buffer.write(b"\\nDONE\\n")
sys.stdout.buffer.flush()
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            # Should complete without hanging
            assert result.exit_code == 0
            assert "DONE" in result.stdout

    async def test_infinite_loop_times_out(self) -> None:
        """Infinite loop is killed by timeout."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
while True:
    pass
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=3,
            )

            # Should be killed by timeout, not hang forever
            # Exit code varies (137 for SIGKILL, or timeout-specific)
            exec_time = result.execution_time_ms or 0
            assert result.exit_code != 0 or exec_time >= 2500

    async def test_infinite_output_times_out(self) -> None:
        """Infinite output is killed by timeout."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
while True:
    print("spam")
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=3,
            )

            # Should be killed by timeout
            exec_time = result.execution_time_ms or 0
            assert result.exit_code != 0 or exec_time >= 2500
            assert "spam" in result.stdout

    async def test_sleep_respects_timeout(self) -> None:
        """Sleep is interrupted by timeout."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import time
import sys
print("starting", flush=True)
sys.stdout.flush()
time.sleep(60)
print("done", flush=True)
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=3,
            )

            # Should timeout before sleep completes
            # Note: "starting" may or may not be captured depending on streaming timing
            assert "done" not in result.stdout

    async def test_exit_immediately(self) -> None:
        """Code that exits immediately."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import sys
sys.exit(0)
print("should not print")
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code == 0
            assert "should not print" not in result.stdout

    async def test_os_exit_immediately(self) -> None:
        """Code that calls os._exit (bypasses cleanup)."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import os
os._exit(42)
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code == 42


# =============================================================================
# Out of Bounds: Resource exhaustion
# =============================================================================
class TestOutOfBounds:
    """Tests for resource limits and exhaustion."""

    async def test_memory_allocation_large(self) -> None:
        """Attempting to allocate lots of memory."""
        config = SchedulerConfig(images_dir=images_dir)

        # Try to allocate 500MB (should fail or be killed with 256MB VM)
        code = """
try:
    data = bytearray(500 * 1024 * 1024)  # 500MB
    print("ALLOCATED")
except MemoryError:
    print("MEMORY_ERROR")
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                memory_mb=256,
                timeout_seconds=30,
            )

            # Should either get MemoryError or be OOM-killed
            assert "MEMORY_ERROR" in result.stdout or result.exit_code != 0

    async def test_deep_recursion(self) -> None:
        """Deep recursion hits stack limit."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import sys
sys.setrecursionlimit(100000)

def recurse(n):
    if n <= 0:
        return 0
    return 1 + recurse(n - 1)

try:
    result = recurse(50000)
    print(f"RESULT:{result}")
except RecursionError:
    print("RECURSION_ERROR")
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=30,
            )

            # Should hit recursion limit
            assert "RECURSION_ERROR" in result.stdout or "RESULT:" in result.stdout

    async def test_many_file_descriptors(self) -> None:
        """Opening many file descriptors."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import os
fds = []
try:
    for i in range(10000):
        fd = os.open("/dev/null", os.O_RDONLY)
        fds.append(fd)
    print(f"OPENED:{len(fds)}")
except OSError as e:
    print(f"FD_LIMIT:{len(fds)}")
finally:
    for fd in fds:
        try:
            os.close(fd)
        except:
            pass
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            # Should hit fd limit or succeed
            assert "OPENED:" in result.stdout or "FD_LIMIT:" in result.stdout

    async def test_subprocess_spawning(self) -> None:
        """Spawning subprocesses from code."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import subprocess
result = subprocess.run(["echo", "hello from subprocess"], capture_output=True, text=True)
print(f"SUBPROCESS:{result.stdout.strip()}")
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            # Subprocesses may or may not be allowed
            # Should not hang or crash either way
            assert result.exit_code == 0 or "SUBPROCESS:" not in result.stdout


# =============================================================================
# Error Cases: Code that should fail gracefully
# =============================================================================
class TestErrorCases:
    """Code that produces errors should be handled gracefully."""

    async def test_syntax_error(self) -> None:
        """Python syntax error."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
def broken(
    print("missing closing paren"
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code != 0
            assert "SyntaxError" in result.stderr or "syntax" in result.stderr.lower()

    async def test_import_error(self) -> None:
        """Import non-existent module."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import nonexistent_module_xyz123
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code != 0
            assert "ModuleNotFoundError" in result.stderr or "No module" in result.stderr

    async def test_name_error(self) -> None:
        """Reference undefined variable."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
print(undefined_variable_xyz)
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code != 0
            assert "NameError" in result.stderr

    async def test_division_by_zero(self) -> None:
        """Division by zero."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
x = 1 / 0
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code != 0
            assert "ZeroDivisionError" in result.stderr

    async def test_file_not_found(self) -> None:
        """Open non-existent file."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
with open("/nonexistent/path/to/file.txt") as f:
    print(f.read())
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            assert result.exit_code != 0
            assert "FileNotFoundError" in result.stderr or "No such file" in result.stderr

    async def test_permission_denied(self) -> None:
        """Write to read-only filesystem fails even as root.

        Note: VM runs as root, so traditional permission tests on /etc/shadow
        won't work. Instead, test writing to /proc which is read-only.
        """
        config = SchedulerConfig(images_dir=images_dir)

        code = """
with open("/proc/version", "w") as f:
    f.write("test")
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
            )

            # Writing to /proc should fail even as root (may be OSError, IOError, or PermissionError)
            assert result.exit_code != 0
            assert any(
                err in result.stderr
                for err in ["PermissionError", "Read-only", "Operation not permitted", "OSError", "I/O error"]
            )

    async def test_keyboard_interrupt_handling(self) -> None:
        """Code that catches KeyboardInterrupt/signals is killed by timeout."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
import signal
import sys

def handler(signum, frame):
    print("CAUGHT_SIGNAL", flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

print("READY", flush=True)
sys.stdout.flush()
import time
time.sleep(10)
print("DONE", flush=True)
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=3,
            )

            # Should be killed by timeout - "DONE" should not appear
            assert "DONE" not in result.stdout


# =============================================================================
# JavaScript Edge Cases
# =============================================================================
class TestJavaScriptEdgeCases:
    """Edge cases specific to JavaScript/Bun."""

    async def test_js_syntax_error(self) -> None:
        """JavaScript syntax error."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
function broken( {
    console.log("missing paren");
}
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.JAVASCRIPT,
            )

            assert result.exit_code != 0

    async def test_js_undefined_variable(self) -> None:
        """JavaScript undefined variable."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
console.log(undefinedVariable);
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.JAVASCRIPT,
            )

            # Bun may print undefined or throw ReferenceError
            assert result.exit_code != 0 or "undefined" in result.stdout.lower()

    async def test_js_async_await(self) -> None:
        """JavaScript async/await."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
async function main() {
    await new Promise(resolve => setTimeout(resolve, 100));
    console.log("ASYNC_DONE");
}
main();
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.JAVASCRIPT,
            )

            assert result.exit_code == 0
            assert "ASYNC_DONE" in result.stdout

    async def test_js_promise_rejection(self) -> None:
        """JavaScript unhandled promise rejection."""
        config = SchedulerConfig(images_dir=images_dir)

        code = """
Promise.reject(new Error("test rejection"));
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.JAVASCRIPT,
            )

            # Should handle rejection gracefully
            # Exit code depends on Bun's behavior
            assert result.exit_code != 0 or "rejection" in result.stderr.lower()


# =============================================================================
# RAW/Shell Edge Cases
# =============================================================================
class TestRawEdgeCases:
    """Edge cases for RAW/shell execution."""

    async def test_raw_command_not_found(self) -> None:
        """Non-existent command."""
        config = SchedulerConfig(images_dir=images_dir)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code="nonexistent_command_xyz123",
                language=Language.RAW,
            )

            assert result.exit_code != 0
            assert "not found" in result.stderr.lower() or "command not found" in result.stderr.lower()

    async def test_raw_pipe(self) -> None:
        """Shell pipe."""
        config = SchedulerConfig(images_dir=images_dir)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code="echo 'hello world' | tr 'a-z' 'A-Z'",
                language=Language.RAW,
            )

            assert result.exit_code == 0
            assert "HELLO WORLD" in result.stdout

    async def test_raw_redirect(self) -> None:
        """Shell redirect."""
        config = SchedulerConfig(images_dir=images_dir)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code="echo 'test' > /tmp/test.txt && cat /tmp/test.txt",
                language=Language.RAW,
            )

            assert result.exit_code == 0
            assert "test" in result.stdout

    async def test_raw_environment_variable(self) -> None:
        """Shell environment variable."""
        config = SchedulerConfig(images_dir=images_dir)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code="MY_VAR='hello' && echo $MY_VAR",
                language=Language.RAW,
            )

            # Shell variable expansion
            assert result.exit_code == 0

"""Edge case, weird, and out-of-bounds tests for code execution.

Tests that the execution system handles unusual inputs gracefully:
1. Edge cases: Empty code, large output, special characters
2. Weird cases: Binary data, null bytes, infinite loops
3. Out of bounds: Memory exhaustion, deep recursion
4. Error cases: Syntax errors, import errors, runtime errors
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from tests.conftest import skip_unless_hwaccel


# =============================================================================
# Edge Cases: Unusual but valid inputs
# =============================================================================
class TestEdgeCases:
    """Edge cases that should work but might break naive implementations."""

    async def test_empty_code(self, scheduler: Scheduler) -> None:
        """Empty code string is rejected by guest-agent validation."""
        result = await scheduler.run(
            code="",
            language=Language.PYTHON,
        )

        # Empty code is rejected with validation error (exit_code=-1)
        assert result.exit_code == -1
        assert "Code cannot be empty" in result.stderr

    async def test_whitespace_only_code(self, scheduler: Scheduler) -> None:
        """Whitespace-only code is rejected by guest-agent validation."""
        result = await scheduler.run(
            code="   \n\n\t\t\n   ",
            language=Language.PYTHON,
        )

        # Whitespace-only code is rejected (trimmed = empty)
        assert result.exit_code == -1
        assert "Code cannot be empty" in result.stderr

    async def test_comment_only_code(self, scheduler: Scheduler) -> None:
        """Comment-only code executes without error."""
        result = await scheduler.run(
            code="# This is just a comment\n# Another comment",
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert result.stdout == ""

    async def test_large_output_1mb(self, scheduler: Scheduler) -> None:
        """Code producing ~1MB of output."""
        # Generate ~1MB of output (1000 lines of 1000 chars each)
        code = """
for i in range(1000):
    print('x' * 1000)
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=60,
        )

        assert result.exit_code == 0
        # Should have substantial output (may be truncated)
        assert len(result.stdout) > 100000  # At least 100KB

    async def test_many_lines_output(self, scheduler: Scheduler) -> None:
        """Code producing many small lines of output."""
        code = """
for i in range(10000):
    print(i)
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0
        # Should contain first and last numbers
        assert "0" in result.stdout
        assert "9999" in result.stdout

    async def test_very_long_single_line(self, scheduler: Scheduler) -> None:
        """Code with a very long single line."""
        # 10KB string
        long_string = "x" * 10000
        code = f"print('{long_string}')"

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert "x" * 100 in result.stdout  # At least some of it

    async def test_special_characters_in_output(self, scheduler: Scheduler) -> None:
        """Code outputting special characters."""
        code = r"""
print("Tab:\there")
print("Newline in string: line1\nline2")
print("Backslash: \\")
print("Quote: \"hello\"")
print("Unicode: café ñ 你好 🎉")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert "Tab:" in result.stdout
        assert "café" in result.stdout
        assert "你好" in result.stdout

    async def test_ansi_escape_codes(self, scheduler: Scheduler) -> None:
        """Code outputting ANSI escape codes."""
        code = r"""
print("\033[31mRed text\033[0m")
print("\033[1mBold text\033[0m")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        # ANSI codes should pass through
        assert "Red text" in result.stdout

    async def test_rapid_stdout_stderr_interleaving(self, scheduler: Scheduler) -> None:
        """Rapid alternating stdout/stderr output."""
        code = """
import sys
for i in range(100):
    print(f"out{i}")
    print(f"err{i}", file=sys.stderr)
"""

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

    async def test_null_bytes_in_output(self, scheduler: Scheduler) -> None:
        """Code outputting null bytes."""
        code = """
import sys
sys.stdout.buffer.write(b"before\\x00after\\n")
sys.stdout.buffer.flush()
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        # Should handle null bytes without crashing
        assert result.exit_code == 0
        assert "before" in result.stdout or "after" in result.stdout

    async def test_binary_data_in_output(self, scheduler: Scheduler) -> None:
        """Code outputting binary data."""
        code = """
import sys
# Write some binary data
sys.stdout.buffer.write(bytes(range(256)))
sys.stdout.buffer.write(b"\\nDONE\\n")
sys.stdout.buffer.flush()
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        # Should complete without hanging
        assert result.exit_code == 0
        assert "DONE" in result.stdout

    async def test_infinite_loop_times_out(self, scheduler: Scheduler) -> None:
        """Infinite loop is killed by timeout."""
        code = """
while True:
    pass
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=3,
        )

        # Should be killed by timeout, not hang forever
        # Exit code varies (137 for SIGKILL, or timeout-specific)
        exec_time = result.execution_time_ms or 0
        assert result.exit_code != 0 or exec_time >= 2500

    @skip_unless_hwaccel
    async def test_infinite_output_times_out(self, scheduler: Scheduler) -> None:
        """Infinite output is killed by timeout.

        Note: Requires hardware acceleration because TCG is 10-50x slower,
        making the 3s timeout elapse before the VM can produce any output.
        """
        code = """
while True:
    print("spam")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=3,
        )

        # Should be killed by timeout
        exec_time = result.execution_time_ms or 0
        assert result.exit_code != 0 or exec_time >= 2500
        assert "spam" in result.stdout

    async def test_sleep_respects_timeout(self, scheduler: Scheduler) -> None:
        """Sleep is interrupted by timeout."""
        code = """
import time
import sys
print("starting", flush=True)
sys.stdout.flush()
time.sleep(60)
print("done", flush=True)
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=3,
        )

        # Should timeout before sleep completes
        # Note: "starting" may or may not be captured depending on streaming timing
        assert "done" not in result.stdout

    async def test_exit_immediately(self, scheduler: Scheduler) -> None:
        """Code that exits immediately."""
        code = """
import sys
sys.exit(0)
print("should not print")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0
        assert "should not print" not in result.stdout

    async def test_os_exit_immediately(self, scheduler: Scheduler) -> None:
        """Code that calls os._exit (bypasses cleanup)."""
        code = """
import os
os._exit(42)
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 42

    @skip_unless_hwaccel
    async def test_sigterm_graceful_exit(self, scheduler: Scheduler) -> None:
        """Process catches SIGTERM and exits gracefully.

        Reliability: Verifies signal type via output.
        """
        code = """
import signal
import sys
import time

def handler(signum, frame):
    # Print signal name to verify SIGTERM was sent (not SIGKILL)
    sig_name = signal.Signals(signum).name
    print(f"RECEIVED_{sig_name}", flush=True)
    sys.exit(42)

signal.signal(signal.SIGTERM, handler)
print("READY", flush=True)
time.sleep(60)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=2,
        )

        # Verify SIGTERM was sent (not SIGKILL)
        assert "RECEIVED_SIGTERM" in result.stdout

    @skip_unless_hwaccel
    async def test_normal_exit_no_termination_needed(self, scheduler: Scheduler) -> None:
        """Process exits normally before timeout - no termination needed.

        Normal case: verifies baseline behavior when graceful termination
        is not triggered.
        """
        code = """
print("STARTING", flush=True)
print("DONE", flush=True)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0
        assert "STARTING" in result.stdout
        assert "DONE" in result.stdout
        # Should complete very quickly (no timeout, no termination)
        assert result.timing is not None
        assert result.timing.execute_ms < 2000  # < 2s

    @skip_unless_hwaccel
    async def test_sigterm_ignored_escalates_to_sigkill(self, scheduler: Scheduler) -> None:
        """Process ignoring SIGTERM is killed by SIGKILL after grace period.

        Weird case: verifies SIGTERM→SIGKILL escalation via timing.
        If SIGTERM worked, execution would be ~2s. Since it's ignored,
        must wait 5s grace period before SIGKILL, so execution >= 5s.

        Note: Requires hardware acceleration because TCG is 10-50x slower,
        making the 2s timeout fail before the process can even print output.
        """
        code = """
import signal
import time

# Ignore SIGTERM - process won't exit gracefully
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("IGNORING_SIGTERM", flush=True)
time.sleep(60)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=2,  # Host adds 8s margin for grace period
        )

        assert "IGNORING_SIGTERM" in result.stdout
        # Key assertion: execution time >= 5s (grace period elapsed)
        # If SIGTERM worked, would be ~2s. Since ignored, must wait 5s grace.
        assert result.timing is not None
        assert result.timing.execute_ms >= 5000  # At least 5s (grace period)
        assert result.timing.execute_ms < 12000  # Under 12s

    async def test_nested_process_tree_termination(self, scheduler: Scheduler) -> None:
        """Nested/deep process tree is terminated via process group.

        Out of bounds case: shell spawns children that spawn grandchildren.
        Process group signal should kill entire tree.
        """
        # Parent spawns child, child spawns grandchild
        # All should be killed by process group signal
        code = """
sh -c 'sh -c "sleep 60" & sleep 60' &
sh -c 'sh -c "sleep 60" & sleep 60' &
echo NESTED_SPAWNED
wait
"""
        result = await scheduler.run(
            code=code,
            language=Language.RAW,
            timeout_seconds=2,  # Host adds 8s margin for grace period
        )

        assert "NESTED_SPAWNED" in result.stdout
        # If nested processes weren't killed, would take 60s
        assert result.timing is not None
        assert result.timing.execute_ms < 12000  # < 12s (much less than 60s)

    async def test_subprocess_tree_termination(self, scheduler: Scheduler) -> None:
        """Shell subprocesses are terminated via process group.

        Reliability: If process group kill fails, `wait` would block for 60s.
        Timing proves all children were killed.
        """
        # Spawn 3 background sleeps, then wait for them
        # If process group works: killed after timeout + grace
        # If process group fails: wait blocks for 60s (test timeout)
        code = "sleep 60 & sleep 60 & sleep 60 & echo SPAWNED && wait"

        result = await scheduler.run(
            code=code,
            language=Language.RAW,
            timeout_seconds=2,  # Host adds 8s margin for grace period
        )

        assert "SPAWNED" in result.stdout
        # Key assertion: completes in reasonable time (not 60s)
        # Expected: ~2s timeout + ~5s grace = ~7s (host allows 2+8=10s)
        assert result.timing is not None
        assert result.timing.execute_ms < 12000  # < 12s (much less than 60s)

    @skip_unless_hwaccel
    async def test_python_subprocess_tree_termination(self, scheduler: Scheduler) -> None:
        """Python subprocesses are terminated via process group.

        Reliability: Same timing strategy as shell test.
        """
        code = """
import subprocess
import time

# Spawn background processes
for _ in range(3):
    subprocess.Popen(["sleep", "60"])

print("SPAWNED", flush=True)
time.sleep(60)  # Wait to be killed
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=2,  # Host adds 8s margin for grace period
        )

        assert "SPAWNED" in result.stdout
        # Should complete after timeout + grace, not after 60s
        # Expected: ~2s timeout + ~5s grace = ~7s (host allows 2+8=10s)
        assert result.timing is not None
        assert result.timing.execute_ms < 12000  # < 12s (much less than 60s)


# =============================================================================
# Out of Bounds: Resource exhaustion
# =============================================================================
class TestOutOfBounds:
    """Tests for resource limits and exhaustion."""

    async def test_memory_allocation_large(self, scheduler: Scheduler) -> None:
        """Attempting to allocate lots of memory."""
        # Try to allocate 500MB (should fail or be killed with 256MB VM)
        code = """
try:
    data = bytearray(500 * 1024 * 1024)  # 500MB
    print("ALLOCATED")
except MemoryError:
    print("MEMORY_ERROR")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            memory_mb=256,
            timeout_seconds=30,
        )

        # Should either get MemoryError or be OOM-killed
        assert "MEMORY_ERROR" in result.stdout or result.exit_code != 0

    async def test_deep_recursion(self, scheduler: Scheduler) -> None:
        """Deep recursion hits stack limit."""
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

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        # Should hit recursion limit
        assert "RECURSION_ERROR" in result.stdout or "RESULT:" in result.stdout

    async def test_many_file_descriptors(self, scheduler: Scheduler) -> None:
        """Opening many file descriptors."""
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

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        # Should hit fd limit or succeed
        assert "OPENED:" in result.stdout or "FD_LIMIT:" in result.stdout

    async def test_subprocess_spawning(self, scheduler: Scheduler) -> None:
        """Spawning subprocesses from code."""
        code = """
import subprocess
result = subprocess.run(["echo", "hello from subprocess"], capture_output=True, text=True)
print(f"SUBPROCESS:{result.stdout.strip()}")
"""

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

    async def test_syntax_error(self, scheduler: Scheduler) -> None:
        """Python syntax error."""
        code = """
def broken(
    print("missing closing paren"
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr or "syntax" in result.stderr.lower()

    async def test_import_error(self, scheduler: Scheduler) -> None:
        """Import non-existent module."""
        code = """
import nonexistent_module_xyz123
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code != 0
        assert "ModuleNotFoundError" in result.stderr or "No module" in result.stderr

    async def test_name_error(self, scheduler: Scheduler) -> None:
        """Reference undefined variable."""
        code = """
print(undefined_variable_xyz)
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code != 0
        assert "NameError" in result.stderr

    async def test_division_by_zero(self, scheduler: Scheduler) -> None:
        """Division by zero."""
        code = """
x = 1 / 0
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code != 0
        assert "ZeroDivisionError" in result.stderr

    async def test_file_not_found(self, scheduler: Scheduler) -> None:
        """Open non-existent file."""
        code = """
with open("/nonexistent/path/to/file.txt") as f:
    print(f.read())
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code != 0
        assert "FileNotFoundError" in result.stderr or "No such file" in result.stderr

    async def test_permission_denied(self, scheduler: Scheduler) -> None:
        """Write to read-only filesystem fails.

        Note: /proc is read-only regardless of UID, making it a reliable
        target for testing write-permission errors.
        """
        code = """
with open("/proc/version", "w") as f:
    f.write("test")
"""

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

    async def test_keyboard_interrupt_handling(self, scheduler: Scheduler) -> None:
        """Code that catches KeyboardInterrupt/signals is killed by timeout."""
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

    async def test_js_syntax_error(self, scheduler: Scheduler) -> None:
        """JavaScript syntax error."""
        code = """
function broken( {
    console.log("missing paren");
}
"""

        result = await scheduler.run(
            code=code,
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code != 0

    async def test_js_undefined_variable(self, scheduler: Scheduler) -> None:
        """JavaScript undefined variable."""
        code = """
console.log(undefinedVariable);
"""

        result = await scheduler.run(
            code=code,
            language=Language.JAVASCRIPT,
        )

        # Bun may print undefined or throw ReferenceError
        assert result.exit_code != 0 or "undefined" in result.stdout.lower()

    async def test_js_async_await(self, scheduler: Scheduler) -> None:
        """JavaScript async/await."""
        code = """
async function main() {
    await new Promise(resolve => setTimeout(resolve, 100));
    console.log("ASYNC_DONE");
}
main();
"""

        result = await scheduler.run(
            code=code,
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "ASYNC_DONE" in result.stdout

    async def test_js_promise_rejection(self, scheduler: Scheduler) -> None:
        """JavaScript unhandled promise rejection."""
        code = """
Promise.reject(new Error("test rejection"));
"""

        result = await scheduler.run(
            code=code,
            language=Language.JAVASCRIPT,
        )

        # Should handle rejection gracefully
        # Exit code depends on Bun's behavior
        assert result.exit_code != 0 or "rejection" in result.stderr.lower()


# =============================================================================
# TypeScript Support (loader: 'ts' in Bun REPL wrapper)
# =============================================================================
class TestTypeScriptSupport:
    """TypeScript syntax is accepted and transpiled by the Bun REPL wrapper."""

    # --- Normal cases: core TS features ---

    async def test_ts_type_annotation(self, scheduler: Scheduler) -> None:
        """Basic type annotation on variable."""
        result = await scheduler.run(
            code="const x: number = 42; console.log(x);",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "42" in result.stdout

    async def test_ts_interface(self, scheduler: Scheduler) -> None:
        """Interface definition and typed object."""
        result = await scheduler.run(
            code='interface User { name: string; age: number } const u: User = { name: "Alice", age: 30 }; console.log(u.name, u.age);',
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "Alice" in result.stdout
        assert "30" in result.stdout

    async def test_ts_generic_function(self, scheduler: Scheduler) -> None:
        """Generic function with type parameter."""
        result = await scheduler.run(
            code="function identity<T>(x: T): T { return x; } console.log(identity(7));",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "7" in result.stdout

    async def test_ts_enum(self, scheduler: Scheduler) -> None:
        """TypeScript enum."""
        result = await scheduler.run(
            code="enum Color { Red, Green, Blue } console.log(Color.Green);",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "1" in result.stdout

    async def test_ts_type_alias(self, scheduler: Scheduler) -> None:
        """Type alias."""
        result = await scheduler.run(
            code='type Pair = [number, string]; const p: Pair = [1, "hello"]; console.log(p[0], p[1]);',
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "1" in result.stdout
        assert "hello" in result.stdout

    async def test_ts_as_cast(self, scheduler: Scheduler) -> None:
        """'as' type assertion."""
        result = await scheduler.run(
            code='const x: unknown = "hello"; console.log((x as string).length);',
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "5" in result.stdout

    # --- Edge cases ---

    async def test_ts_generic_arrow_function(self, scheduler: Scheduler) -> None:
        """Generic arrow function (valid with 'ts' loader, ambiguous with 'tsx')."""
        result = await scheduler.run(
            code='const id = <T>(x: T): T => x; console.log(id("hi"));',
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "hi" in result.stdout

    async def test_ts_optional_chaining_with_types(self, scheduler: Scheduler) -> None:
        """Optional chaining with typed objects."""
        result = await scheduler.run(
            code='const o: { a?: { b?: number } } = {}; console.log(o?.a?.b ?? "none");',
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "none" in result.stdout

    async def test_ts_type_only_constructs_stripped(self, scheduler: Scheduler) -> None:
        """Type-only constructs are erased, no runtime footprint."""
        result = await scheduler.run(
            code="type Foo = { x: number }; const f: Foo = { x: 1 }; console.log(f.x);",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "1" in result.stdout

    async def test_ts_plain_js_still_works(self, scheduler: Scheduler) -> None:
        """Plain JavaScript is unaffected by the 'ts' loader (regression guard)."""
        result = await scheduler.run(
            code="var arr = [1, 2, 3]; console.log(arr.length);",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "3" in result.stdout

    async def test_ts_top_level_await_with_types(self, scheduler: Scheduler) -> None:
        """Top-level await combined with TypeScript annotations."""
        result = await scheduler.run(
            code="const r: number = await Promise.resolve(42); console.log(r);",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "42" in result.stdout

    async def test_ts_empty_object_type(self, scheduler: Scheduler) -> None:
        """Empty object type annotation."""
        result = await scheduler.run(
            code="const x: {} = {}; console.log(typeof x);",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "object" in result.stdout

    # --- Weird / out-of-bounds cases ---

    async def test_ts_deeply_nested_generics(self, scheduler: Scheduler) -> None:
        """Deeply nested generic types."""
        code = """
const m: Map<string, Map<string, Array<[number, string]>>> = new Map();
const inner = new Map<string, Array<[number, string]>>();
inner.set("key", [[1, "a"], [2, "b"]]);
m.set("outer", inner);
console.log(m.get("outer")!.get("key")![0][1]);
"""
        result = await scheduler.run(
            code=code,
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "a" in result.stdout

    async def test_ts_mixed_js_and_ts(self, scheduler: Scheduler) -> None:
        """Mixed untyped JS and typed TS in the same snippet."""
        code = """
var count = 0;
function inc(): number { return ++count; }
inc(); inc(); inc();
const result: string = `count=${count}`;
console.log(result);
"""
        result = await scheduler.run(
            code=code,
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "count=3" in result.stdout

    async def test_ts_types_are_erased_at_runtime(self, scheduler: Scheduler) -> None:
        """Types are erased — no runtime type checking (by design)."""
        # "number" annotation but assigned a string via `as any` — runs fine
        result = await scheduler.run(
            code='const x: number = "hello" as any; console.log(typeof x);',
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0
        assert "string" in result.stdout

    # --- Error cases ---

    async def test_ts_invalid_syntax(self, scheduler: Scheduler) -> None:
        """Invalid TypeScript syntax is rejected."""
        result = await scheduler.run(
            code="const x: = 42;",
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code != 0


# =============================================================================
# RAW/Shell Edge Cases
# =============================================================================
class TestRawEdgeCases:
    """Edge cases for RAW/shell execution."""

    async def test_raw_command_not_found(self, scheduler: Scheduler) -> None:
        """Non-existent command."""
        result = await scheduler.run(
            code="nonexistent_command_xyz123",
            language=Language.RAW,
        )

        assert result.exit_code != 0
        assert "not found" in result.stderr.lower() or "command not found" in result.stderr.lower()

    async def test_raw_pipe(self, scheduler: Scheduler) -> None:
        """Shell pipe."""
        result = await scheduler.run(
            code="echo 'hello world' | tr 'a-z' 'A-Z'",
            language=Language.RAW,
        )

        assert result.exit_code == 0
        assert "HELLO WORLD" in result.stdout

    async def test_raw_redirect(self, scheduler: Scheduler) -> None:
        """Shell redirect."""
        result = await scheduler.run(
            code="echo 'test' > /tmp/test.txt && cat /tmp/test.txt",
            language=Language.RAW,
        )

        assert result.exit_code == 0
        assert "test" in result.stdout

    async def test_raw_environment_variable(self, scheduler: Scheduler) -> None:
        """Shell environment variable."""
        result = await scheduler.run(
            code="MY_VAR='hello' && echo $MY_VAR",
            language=Language.RAW,
        )

        # Shell variable expansion
        assert result.exit_code == 0

    # ---- Shebang handling (temp-file dispatch) ----

    async def test_raw_shebang_awk(self, scheduler: Scheduler) -> None:
        """AWK shebang executes via kernel binfmt_script."""
        result = await scheduler.run(
            code='#!/usr/bin/awk -f\nBEGIN { print "shebang_awk_ok" }',
            language=Language.RAW,
        )

        assert result.exit_code == 0
        assert "shebang_awk_ok" in result.stdout

    async def test_raw_shebang_sh(self, scheduler: Scheduler) -> None:
        """Shell shebang via temp file."""
        result = await scheduler.run(
            code="#!/bin/sh\necho shebang_sh_ok",
            language=Language.RAW,
        )

        assert result.exit_code == 0
        assert "shebang_sh_ok" in result.stdout

    async def test_raw_shebang_bash(self, scheduler: Scheduler) -> None:
        """Bash shebang via temp file."""
        result = await scheduler.run(
            code="#!/bin/bash\necho shebang_bash_ok",
            language=Language.RAW,
        )

        assert result.exit_code == 0
        assert "shebang_bash_ok" in result.stdout

    async def test_raw_shebang_sed(self, scheduler: Scheduler) -> None:
        """Sed shebang processes input."""
        result = await scheduler.run(
            code="#!/bin/sh\necho hello | sed 's/hello/world/'",
            language=Language.RAW,
        )

        assert result.exit_code == 0
        assert "world" in result.stdout

    async def test_raw_shebang_nonzero_exit(self, scheduler: Scheduler) -> None:
        """Non-zero exit code propagates through temp-file wrapper."""
        result = await scheduler.run(
            code="#!/bin/sh\nexit 42",
            language=Language.RAW,
        )

        assert result.exit_code == 42

    async def test_raw_shebang_stderr(self, scheduler: Scheduler) -> None:
        """Stderr output from shebanged script."""
        result = await scheduler.run(
            code="#!/bin/sh\necho err_msg >&2",
            language=Language.RAW,
        )

        assert "err_msg" in result.stderr

    async def test_raw_shebang_env_vars(self, scheduler: Scheduler) -> None:
        """Exported env vars are inherited by shebanged subprocess."""
        result = await scheduler.run(
            code="#!/bin/sh\necho $MY_VAR",
            language=Language.RAW,
            env_vars={"MY_VAR": "from_env"},
        )

        assert result.exit_code == 0
        assert "from_env" in result.stdout

    async def test_raw_shebang_empty_body(self, scheduler: Scheduler) -> None:
        """Shebang with empty body is valid."""
        result = await scheduler.run(
            code="#!/bin/sh\n",
            language=Language.RAW,
        )

        assert result.exit_code == 0

    async def test_raw_shebang_special_chars(self, scheduler: Scheduler) -> None:
        """Special characters are preserved (heredoc quoting prevents expansion)."""
        result = await scheduler.run(
            code="#!/bin/sh\necho 'quotes \"double\" $dollar `backtick`'",
            language=Language.RAW,
        )

        assert result.exit_code == 0
        assert "$dollar" in result.stdout
        assert "`backtick`" in result.stdout

    async def test_raw_shebang_sentinel_in_code(self, scheduler: Scheduler) -> None:
        """Sentinel collision in code is handled."""
        result = await scheduler.run(
            code='#!/bin/sh\necho "_EXEC_SANDBOX_EOF_"',
            language=Language.RAW,
        )

        assert result.exit_code == 0
        assert "_EXEC_SANDBOX_EOF_" in result.stdout

    async def test_raw_shebang_nonexistent_interpreter(self, scheduler: Scheduler) -> None:
        """Non-existent interpreter fails with non-zero exit."""
        result = await scheduler.run(
            code="#!/usr/bin/nonexistent_interp_xyz\necho hello",
            language=Language.RAW,
        )

        assert result.exit_code != 0

    async def test_raw_shebang_no_newline(self, scheduler: Scheduler) -> None:
        """Shebang with no trailing newline and no body."""
        result = await scheduler.run(
            code="#!/bin/sh",
            language=Language.RAW,
        )

        assert result.exit_code == 0

    async def test_raw_no_shebang_hash_comment(self, scheduler: Scheduler) -> None:
        """Plain # comment is not treated as shebang."""
        result = await scheduler.run(
            code="# just a comment\necho hello",
            language=Language.RAW,
        )

        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_raw_shebang_with_leading_space(self, scheduler: Scheduler) -> None:
        """Leading space means no shebang detection; eval path used."""
        result = await scheduler.run(
            code="  #!/bin/sh\necho hello",
            language=Language.RAW,
        )

        # Leading space: bash treats #! as comment, echo runs normally
        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_raw_shebang_multiline_output(self, scheduler: Scheduler) -> None:
        """AWK shebang producing multiline output."""
        result = await scheduler.run(
            code="#!/usr/bin/awk -f\nBEGIN { for(i=1;i<=100;i++) print i }",
            language=Language.RAW,
        )

        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 100
        assert lines[0].strip() == "1"
        assert lines[99].strip() == "100"

    async def test_raw_shebang_awk_multiline_program(self, scheduler: Scheduler) -> None:
        """AWK shebang with BEGIN, pattern, and END blocks."""
        # Use a shell wrapper to pipe input into the awk script, since
        # a bare awk with an END block reads stdin (hangs without input).
        result = await scheduler.run(
            code='#!/bin/sh\nprintf "line1\\nline2\\n" | awk \'BEGIN{print "start"} /line/{print "matched:"$0} END{print "end"}\'',
            language=Language.RAW,
        )

        assert result.exit_code == 0
        assert "start" in result.stdout
        assert "matched:line1" in result.stdout
        assert "matched:line2" in result.stdout
        assert "end" in result.stdout


# =============================================================================
# Stdin EOF: user code reading stdin gets immediate EOF (not a hang)
# =============================================================================


class TestStdinEofNormal:
    """Common stdin-reading patterns return EOF immediately instead of hanging."""

    async def test_python_input_eof(self, scheduler: Scheduler) -> None:
        """input() raises EOFError when stdin is /dev/null."""
        code = """\
try:
    input("prompt> ")
except EOFError:
    print("CAUGHT_EOF")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "CAUGHT_EOF" in result.stdout

    async def test_python_stdin_read_empty(self, scheduler: Scheduler) -> None:
        """sys.stdin.read() returns empty string at EOF."""
        code = """\
import sys
data = sys.stdin.read()
print(repr(data))
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "''" in result.stdout

    async def test_python_stdin_iteration_empty(self, scheduler: Scheduler) -> None:
        """for line in sys.stdin iterates zero times at EOF."""
        code = """\
import sys
count = 0
for line in sys.stdin:
    count += 1
print(f"COUNT={count}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "COUNT=0" in result.stdout

    async def test_shell_read_eof(self, scheduler: Scheduler) -> None:
        """Shell read returns failure at EOF, allowing fallback."""
        result = await scheduler.run(
            code='read var || echo "NO_INPUT"',
            language=Language.RAW,
        )
        assert result.exit_code == 0
        assert "NO_INPUT" in result.stdout

    async def test_js_stdin_read_null(self, scheduler: Scheduler) -> None:
        """process.stdin.read() returns null at EOF."""
        code = """\
const d = process.stdin.read();
console.log(String(d));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0
        assert "null" in result.stdout


class TestStdinEofEdge:
    """Boundary conditions and sequential stdin interactions."""

    async def test_python_input_catch_continue(self, scheduler: Scheduler) -> None:
        """Code catches EOFError from input() and continues execution."""
        code = """\
try:
    input()
except EOFError:
    print("CAUGHT")
print("CONTINUED")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "CAUGHT" in result.stdout
        assert "CONTINUED" in result.stdout

    async def test_python_stdin_read_twice(self, scheduler: Scheduler) -> None:
        """sys.stdin.read() called twice both return empty string (idempotent EOF)."""
        code = """\
import sys
a = sys.stdin.read()
b = sys.stdin.read()
print(f"A={repr(a)} B={repr(b)}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "A=''" in result.stdout
        assert "B=''" in result.stdout

    async def test_python_stdin_buffer_read(self, scheduler: Scheduler) -> None:
        """sys.stdin.buffer.read() returns b'' (real file, not StringIO)."""
        code = """\
import sys
data = sys.stdin.buffer.read()
print(repr(data))
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "b''" in result.stdout

    async def test_python_stdin_readline_empty(self, scheduler: Scheduler) -> None:
        """sys.stdin.readline() returns '' at EOF (not EOFError)."""
        code = """\
import sys
line = sys.stdin.readline()
print(repr(line))
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "''" in result.stdout

    async def test_session_reuse_after_stdin_read(self, scheduler: Scheduler) -> None:
        """REPL protocol channel is unaffected after user code reads stdin."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            r1 = await session.exec("import sys; sys.stdin.read()")
            assert r1.exit_code == 0

            r2 = await session.exec('print("STILL_WORKS")')
            assert r2.exit_code == 0
            assert "STILL_WORKS" in r2.stdout


class TestStdinEofWeird:
    """Valid but unusual stdin patterns."""

    async def test_python_stdin_fileno_valid(self, scheduler: Scheduler) -> None:
        """sys.stdin.fileno() returns a valid int (real file object, not StringIO)."""
        code = """\
import sys
fd = sys.stdin.fileno()
print(f"FD={fd}")
print(f"TYPE={type(fd).__name__}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "TYPE=int" in result.stdout

    async def test_python_stdin_isatty_false(self, scheduler: Scheduler) -> None:
        """sys.stdin.isatty() returns False (/dev/null is not a TTY)."""
        code = """\
import sys
print(f"ISATTY={sys.stdin.isatty()}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "ISATTY=False" in result.stdout

    async def test_python_json_load_stdin(self, scheduler: Scheduler) -> None:
        """json.load(sys.stdin) raises JSONDecodeError on empty input (not hang)."""
        code = """\
import json, sys
try:
    json.load(sys.stdin)
except json.JSONDecodeError:
    print("JSON_DECODE_ERROR")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "JSON_DECODE_ERROR" in result.stdout

    async def test_python_select_stdin_readable(self, scheduler: Scheduler) -> None:
        """/dev/null reports as readable via select (immediate EOF)."""
        code = """\
import select, sys
r, _, _ = select.select([sys.stdin], [], [], 0)
print(f"READABLE={len(r) > 0}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "READABLE=True" in result.stdout

    async def test_shell_wc_stdin_zero(self, scheduler: Scheduler) -> None:
        """wc -l with no file arg reads stdin (/dev/null), reports 0 lines."""
        result = await scheduler.run(
            code="wc -l",
            language=Language.RAW,
        )
        assert result.exit_code == 0
        assert "0" in result.stdout


class TestStdinEofOutOfBounds:
    """Adversarial and limitation-probing stdin patterns."""

    async def test_python_os_read_fd0_blocks(self, scheduler: Scheduler) -> None:
        """os.read(0, ...) blocks because fd 0 is the protocol pipe, not /dev/null.

        Known limitation: only sys.stdin is redirected; OS-level fd 0 remains
        connected to the protocol pipe for the REPL command channel.
        """
        code = """\
import os, signal
signal.alarm(2)
try:
    os.read(0, 1024)
    print("READ_RETURNED")
except Exception as e:
    print(f"ERROR={type(e).__name__}")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=3,
        )
        assert result.exit_code == -1

    async def test_python_stdin_write_raises(self, scheduler: Scheduler) -> None:
        """Writing to stdin (opened read-only from /dev/null) raises."""
        code = """\
import sys
try:
    sys.stdin.write("x")
except Exception as e:
    print(f"ERROR={type(e).__name__}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "UnsupportedOperation" in result.stdout

    async def test_python_open_dev_stdin_blocks_or_denied(self, scheduler: Scheduler) -> None:
        """/dev/stdin reopens fd 0 (protocol pipe) or is denied by sandbox.

        Known limitation: /dev/stdin is a symlink to /proc/self/fd/0. If
        accessible, it reopens the protocol pipe and blocks. If /proc/self/fd
        is restricted, open() raises PermissionError.
        """
        code = """\
import signal
signal.alarm(2)
try:
    data = open("/dev/stdin").read()
    print(f"DATA={repr(data)}")
except PermissionError:
    print("PERMISSION_ERROR")
except Exception as e:
    print(f"ERROR={type(e).__name__}")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=3,
        )
        # Either blocks until timeout (fd 0 is protocol pipe) or permission denied
        assert result.exit_code == -1 or "PERMISSION_ERROR" in result.stdout

    async def test_python_fileinput_stdin_eof(self, scheduler: Scheduler) -> None:
        """fileinput.input('-') iterates zero times at EOF."""
        code = """\
import fileinput
count = 0
for line in fileinput.input("-"):
    count += 1
print(f"COUNT={count}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "COUNT=0" in result.stdout

"""Edge case, weird, and out-of-bounds tests for code execution.

Tests that the execution system handles unusual inputs gracefully:
1. Edge cases: Empty code, large output, special characters
2. Weird cases: Binary data, null bytes, infinite loops
3. Out of bounds: Memory exhaustion, deep recursion
4. Error cases: Syntax errors, import errors, runtime errors
"""

import io
import time

import pytest

from exec_sandbox.exceptions import ErrorCode, InvalidInputError, TransientError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once, run_timed, skip_unless_hwaccel, skip_unless_linux


# =============================================================================
# Edge Cases: Unusual but valid inputs
# =============================================================================
class TestEdgeCases:
    """Edge cases that should work but might break naive implementations."""

    async def test_empty_code(self, scheduler: Scheduler) -> None:
        """Empty code string raises InvalidInputError CODE_INVALID."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.CODE_INVALID, match="Code cannot be empty"
        ):
            await run_once(scheduler, VmSpec(language=Language.PYTHON), "")

    async def test_whitespace_only_code(self, scheduler: Scheduler) -> None:
        """Whitespace-only code raises InvalidInputError CODE_INVALID."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.CODE_INVALID, match="Code cannot be empty"
        ):
            await run_once(scheduler, VmSpec(language=Language.PYTHON), "   \n\n\t\t\n   ")

    async def test_comment_only_code(self, scheduler: Scheduler) -> None:
        """Comment-only code executes without error."""
        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), "# This is just a comment\n# Another comment"
        )

        assert result.exit_code == 0
        assert result.stdout == ""

    async def test_large_output_1mb(self, scheduler: Scheduler) -> None:
        """Code producing ~1MB of output (just under guest-agent 1MB stdout limit)."""
        # Generate output just under the 1MB limit: 999 lines x 1001 bytes = 999,999 bytes
        code = """
for i in range(999):
    print('x' * 1000)
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        # Should have substantial output (just under 1MB limit)
        assert len(result.stdout) > 100000  # At least 100KB

    async def test_many_lines_output(self, scheduler: Scheduler) -> None:
        """Code producing many small lines of output."""
        code = """
for i in range(10000):
    print(i)
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=30)

        assert result.exit_code == 0
        # Should contain first and last numbers
        assert "0" in result.stdout
        assert "9999" in result.stdout

    async def test_very_long_single_line(self, scheduler: Scheduler) -> None:
        """Code with a very long single line."""
        # 10KB string
        long_string = "x" * 10000
        code = f"print('{long_string}')"

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        assert "out0" in result.stdout
        assert "out99" in result.stdout
        assert "err0" in result.stderr
        assert "err99" in result.stderr

    async def test_bom_prefix_stripped_python(self, scheduler: Scheduler) -> None:
        """UTF-8 BOM (U+FEFF) before Python code is stripped, execution succeeds."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "\ufeffprint('hello')")

        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_bom_prefix_stripped_javascript(self, scheduler: Scheduler) -> None:
        """UTF-8 BOM (U+FEFF) before JavaScript code is stripped, execution succeeds."""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), "\ufeffconsole.log('hello')")

        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_bom_prefix_stripped_raw(self, scheduler: Scheduler) -> None:
        """UTF-8 BOM (U+FEFF) before shell code is stripped, execution succeeds."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "\ufeffecho hello")

        assert result.exit_code == 0
        assert "hello" in result.stdout


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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        # Should handle null bytes without crashing
        assert result.exit_code == 0
        assert "before" in result.stdout or "after" in result.stdout

    @pytest.mark.parametrize("language", [Language.PYTHON, Language.JAVASCRIPT, Language.RAW])
    async def test_null_bytes_rejected_all_languages(self, scheduler: Scheduler, language: Language) -> None:
        """Null bytes rejected regardless of language."""
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.CODE_INVALID, match="null bytes"):
            await run_once(scheduler, VmSpec(language=language), "echo hi\x00")

    async def test_escaped_null_repr_accepted(self, scheduler: Scheduler) -> None:
        r"""Literal '\\x00' string (no actual null byte) executes fine."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print(repr('hello\\x00world'))")
        assert result.exit_code == 0
        assert "hello\\x00world" in result.stdout

    async def test_binary_data_in_output(self, scheduler: Scheduler) -> None:
        """Code outputting binary data."""
        code = """
import sys
# Write some binary data
sys.stdout.buffer.write(bytes(range(256)))
sys.stdout.buffer.write(b"\\nDONE\\n")
sys.stdout.buffer.flush()
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        # Should complete without hanging
        assert result.exit_code == 0
        assert "DONE" in result.stdout

    async def test_infinite_loop_times_out(self, scheduler: Scheduler) -> None:
        """Infinite loop is killed by timeout."""
        code = """
while True:
    pass
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=3)

        # Should be killed by timeout, not hang forever
        # Exit code varies (137 for SIGKILL, or timeout-specific)
        exec_time = result.execution_time_ms or 0
        assert result.exit_code != 0 or exec_time >= 2500

    @skip_unless_hwaccel
    async def test_infinite_output_times_out(self, scheduler: Scheduler) -> None:
        """Infinite output is killed by timeout.

        Requires hwaccel: 3s timeout is too short for TCG (~5-8x slower) —
        the VM cannot produce any output before the deadline expires.
        """
        code = """
while True:
    print("spam")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=3)

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=3)

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        assert "should not print" not in result.stdout

    async def test_code_that_closes_its_output_and_returns_ends_the_exec(self, scheduler: Scheduler) -> None:
        """Code closes fd 1 and 2: the REPL writes the exec's end on its own copies of them,
        so the exec ends at once and the REPL serves the next one; a file opened later (it
        may get fd 1) never gets a protocol marker."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            closed = await session.exec("import os\nos.close(1)\nos.close(2)\n", timeout_seconds=30)
            after = await session.exec(
                "f = open('/home/user/f.txt', 'w'); f.write('mine'); f.close()", timeout_seconds=30
            )
            content = io.BytesIO()
            await session.read_file("f.txt", destination=content)

        assert (closed.exit_code, closed.timing.execute_ms < 5_000) == (0, True), closed.stderr
        assert after.exit_code == 0, after.stderr
        assert content.getvalue() == b"mine"

    async def test_code_that_closes_its_output_and_hangs_ends_at_its_timeout(self, scheduler: Scheduler) -> None:
        """Code closes fd 1 and 2, then keeps running: the REPL keeps the pipes open (its
        copies), so the exec ends at its timeout, as any code that runs on."""
        code = "import os, time\nos.close(1)\nos.close(2)\ntime.sleep(600)\n"

        timed = await run_timed(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=5)

        assert timed.result.exit_code != 0
        assert timed.result.timing.execute_ms < 10_000

    @skip_unless_linux
    async def test_disk_writes_do_not_fill_the_vm_cgroup(self, scheduler: Scheduler) -> None:
        """A VM writes 256 MiB to its home disk: the writes skip the host page cache, so its
        cgroup never reaches memory.max (buffered, a 1 GiB write reached it 480 times).
        Linux, with VM cgroups and a home disk folder that takes O_DIRECT."""
        from exec_sandbox.system_probes import probe_drive_direct_io

        code = (
            "import os\nwith open('/home/user/big.bin', 'wb') as f:\n    for _ in range(16):\n"
            "        f.write(os.urandom(16 << 20))\n    os.fsync(f.fileno())"
        )
        async with await scheduler.new(VmSpec(language=Language.PYTHON, disk_mb=1024)) as session:
            vm = session._vm
            if vm.cgroup is None:
                pytest.skip("no VM cgroup on this host")
            if not await probe_drive_direct_io(vm.workdir.home_disk.parent):
                pytest.skip("the home disk folder does not take O_DIRECT")
            events = vm.cgroup.path / "memory.events"
            before = dict(line.split() for line in events.read_text().splitlines())["max"]
            result = await session.exec(code, timeout_seconds=120)
            after = dict(line.split() for line in events.read_text().splitlines())["max"]

        assert result.exit_code == 0, result.stderr
        assert after == before

    async def test_os_exit_immediately(self, scheduler: Scheduler) -> None:
        """Code that calls os._exit (bypasses cleanup)."""
        code = """
import os
os._exit(42)
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 42

    async def test_exec_after_exit_137_fails_at_once(self, vm_manager: VmManager, execute_timeout: int) -> None:
        """Exit 137 retires the VM: the guest refuses the next code with a reply at once.

        Without the reply (a dropped connection), the host waits for its whole deadline
        (timeout + margin), then raises OutcomeUnknownError."""
        vm = await vm_manager.create_vm(
            VmSpec(language=Language.RAW), tenant_id="test", task_id="exit-137", cache_leases=()
        )
        try:
            killed = await vm.execute("kill -9 $$", timeout_seconds=execute_timeout)
            assert killed.exit_code == 137

            start = time.monotonic()
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE, match="retired"):
                await vm.execute("echo second", timeout_seconds=1)
            assert time.monotonic() - start < 1
        finally:
            await vm_manager.destroy_vm(vm)

    @skip_unless_hwaccel
    async def test_sigterm_graceful_exit(self, scheduler: Scheduler) -> None:
        """Process catches SIGTERM and exits gracefully.

        Requires hwaccel: 2s timeout is too short for TCG (~5-8x slower) —
        the signal handler cannot fire before the deadline.
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
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=2)

        # Verify SIGTERM was sent (not SIGKILL)
        assert "RECEIVED_SIGTERM" in result.stdout
        # Process caught SIGTERM and called sys.exit(42) — normal exit, not a signal kill
        assert result.exit_code == 42

    @pytest.mark.slow
    async def test_signal_kill_returns_128_plus_signal(self, scheduler: Scheduler) -> None:
        """Process killed by uncaught signal returns 128+signal_number.

        Normal case: SIGSEGV (signal 11) kills the REPL process.
        The guest-agent should report exit_code=128+11=139 per Unix convention.
        Slow under TCG: signal delivery + REPL teardown need the full
        scheduler default timeout (120s hwaccel / 240s TCG).
        """
        code = """
import os
import signal
os.kill(os.getpid(), signal.SIGSEGV)
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        # SIGSEGV = signal 11, exit_code = 128 + 11 = 139
        assert result.exit_code == 139

    @pytest.mark.slow
    async def test_normal_exit_no_termination_needed(self, scheduler: Scheduler) -> None:
        """Process exits normally before timeout — no termination needed.

        Slow under TCG: correctness test with generous 30s timeout,
        but TCG boot overhead (~5-8x) makes it too slow for the default suite.
        """
        code = """
print("STARTING", flush=True)
print("DONE", flush=True)
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=30)

        assert result.exit_code == 0
        assert "STARTING" in result.stdout
        assert "DONE" in result.stdout

    @skip_unless_hwaccel
    async def test_sigterm_ignored_escalates_to_sigkill(self, scheduler: Scheduler) -> None:
        """Process ignoring SIGTERM is killed by SIGKILL after grace period.

        Requires hwaccel: 2s timeout + >= 5000ms timing assertion — TCG
        (~5-8x slower) cannot meet these timing constraints.
        """
        code = """
import signal
import time

# Ignore SIGTERM - process won't exit gracefully
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("IGNORING_SIGTERM", flush=True)
time.sleep(60)
"""
        result = await run_once(
            scheduler,
            VmSpec(
                language=Language.PYTHON,
            ),
            code,
            timeout_seconds=2,  # Host adds bounded readiness/grace watchdog margin
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
        result = await run_once(
            scheduler,
            VmSpec(
                language=Language.RAW,
            ),
            code,
            timeout_seconds=2,  # Host adds bounded readiness/grace watchdog margin
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

        result = await run_once(
            scheduler,
            VmSpec(
                language=Language.RAW,
            ),
            code,
            timeout_seconds=2,  # Host adds bounded readiness/grace watchdog margin
        )

        assert "SPAWNED" in result.stdout
        # Key assertion: completes in reasonable time (not 60s)
        # Expected: ~2s timeout + ~5s grace = ~7s (host backstop is 2+10=12s)
        assert result.timing is not None
        assert result.timing.execute_ms < 12000  # < 12s (much less than 60s)

    @skip_unless_hwaccel
    async def test_python_subprocess_tree_termination(self, scheduler: Scheduler) -> None:
        """Python subprocesses are terminated via process group.

        Requires hwaccel: 15s timeout is marginal for TCG — Python boot
        under TCG takes ~8-12s, leaving little room for the actual test.
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
        result = await run_once(
            scheduler,
            VmSpec(
                language=Language.PYTHON,
            ),
            code,
            timeout_seconds=15,  # Must exceed cold Python start (~8s) + code import time
        )

        assert "SPAWNED" in result.stdout
        # Should complete after timeout + grace, not after 60s
        # Expected: ~15s timeout + ~5s grace = ~20s (host backstop is 15+10=25s)
        assert result.timing is not None
        assert result.timing.execute_ms < 25000  # < 25s (much less than 60s)


# =============================================================================
# Out of Bounds: Resource exhaustion
# =============================================================================
class TestOutOfBounds:
    """Tests for resource limits and exhaustion."""

    # slow: not for its time (about 1 s) but to run alone (-n 0): its memory pressure slows the VMs of
    # parallel tests, and their time limits fail (a QMP stop, a sudo bound: 2 of 3 runs)
    @pytest.mark.slow
    @pytest.mark.parametrize("allocation", ["bytearray(500 << 20)", "[bytearray(1 << 20) for _ in range(500)]"])
    async def test_memory_allocation_large(self, scheduler: Scheduler, allocation: str) -> None:
        """More zero-filled memory than the VM has (500 MB in 256 MB): MemoryError, or the
        guest's OOM guard ends the code (137) at once. Zero pages fill zram's logical size at
        no physical cost; the guard counts them, so the exec does not thrash until its
        timeout (120 s without that count)."""
        code = (
            f"try:\n    data = {allocation}\n    print('ALLOCATED')\nexcept MemoryError:\n    print('MEMORY_ERROR')\n"
        )

        timed = await run_timed(scheduler, VmSpec(language=Language.PYTHON, memory_mb=256), code, timeout_seconds=60)

        result = timed.result
        assert "MEMORY_ERROR" in result.stdout or result.exit_code == 137, (result.exit_code, result.stderr)
        assert result.timing.execute_ms < 20_000

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=30)

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        # Should hit fd limit or succeed
        assert "OPENED:" in result.stdout or "FD_LIMIT:" in result.stdout

    async def test_subprocess_spawning(self, scheduler: Scheduler) -> None:
        """Spawning subprocesses from code."""
        code = """
import subprocess
result = subprocess.run(["echo", "hello from subprocess"], capture_output=True, text=True)
print(f"SUBPROCESS:{result.stdout.strip()}")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, result.stderr
        assert "SUBPROCESS:hello from subprocess" in result.stdout


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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr or "syntax" in result.stderr.lower()

    async def test_import_error(self, scheduler: Scheduler) -> None:
        """Import non-existent module."""
        code = """
import nonexistent_module_xyz123
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code != 0
        assert "ModuleNotFoundError" in result.stderr or "No module" in result.stderr

    async def test_name_error(self, scheduler: Scheduler) -> None:
        """Reference undefined variable."""
        code = """
print(undefined_variable_xyz)
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code != 0
        assert "NameError" in result.stderr

    async def test_division_by_zero(self, scheduler: Scheduler) -> None:
        """Division by zero."""
        code = """
x = 1 / 0
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code != 0
        assert "ZeroDivisionError" in result.stderr

    async def test_file_not_found(self, scheduler: Scheduler) -> None:
        """Open non-existent file."""
        code = """
with open("/nonexistent/path/to/file.txt") as f:
    print(f.read())
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code != 0
        assert "FileNotFoundError" in result.stderr or "No such file" in result.stderr

    async def test_permission_denied(self, scheduler: Scheduler) -> None:
        """Write to read-only filesystem fails.

        Note: /proc/version is masked (bind-mounted to /dev/null) for
        security, so writes there silently succeed. Target /etc/passwd
        instead — it lives on the EROFS rootfs and is bind-remounted
        read-only, so writes always fail with EROFS (errno 30).
        """
        code = """
with open("/etc/passwd", "w") as f:
    f.write("test")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        # Writing to EROFS rootfs should fail even as root
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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=3)

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

        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code != 0

    async def test_js_undefined_variable(self, scheduler: Scheduler) -> None:
        """JavaScript undefined variable."""
        code = """
console.log(undefinedVariable);
"""

        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

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

        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 0
        assert "ASYNC_DONE" in result.stdout

    async def test_js_promise_rejection(self, scheduler: Scheduler) -> None:
        """JavaScript unhandled promise rejection."""
        code = """
Promise.reject(new Error("test rejection"));
"""

        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        # Should handle rejection gracefully
        # Exit code depends on Bun's behavior
        assert result.exit_code != 0 or "rejection" in result.stderr.lower()

    # --- Web API globals ---

    async def test_performance_global_available(self, scheduler: Scheduler) -> None:
        """performance Web API is available in JS sandbox context."""
        code = """\
console.log('type=' + typeof performance);
console.log('now_type=' + typeof performance.now);
const t = performance.now();
console.log('is_number=' + (typeof t === 'number' && t > 0));
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "type=object" in result.stdout
        assert "now_type=function" in result.stdout
        assert "is_number=true" in result.stdout


# =============================================================================
# JavaScript Dynamic Import (ESM import() via __import wrapper)
# =============================================================================
class TestJavaScriptDynamicImport:
    """ESM dynamic import() in the JavaScript/Bun VM context."""

    # --- Normal cases ---

    async def test_dynamic_import_builtin(self, scheduler: Scheduler) -> None:
        """Dynamic import of a built-in module."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            "const path = await import('path'); console.log(path.join('a','b'));",
        )

        assert result.exit_code == 0
        assert "a/b" in result.stdout

    async def test_dynamic_import_destructured(self, scheduler: Scheduler) -> None:
        """Destructured dynamic import."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            "const { join } = await import('path'); console.log(join('a','b'));",
        )

        assert result.exit_code == 0
        assert "a/b" in result.stdout

    async def test_dynamic_import_multiple_modules(self, scheduler: Scheduler) -> None:
        """Sequential dynamic imports of multiple modules."""
        code = """\
const path = await import('path');
const url = await import('url');
console.log(path.join('a','b'));
console.log(typeof url.parse);
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 0
        assert "a/b" in result.stdout
        assert "function" in result.stdout

    async def test_dynamic_import_inside_function(self, scheduler: Scheduler) -> None:
        """Dynamic import inside an async function."""
        code = """\
async function f() {
    const m = await import('crypto');
    return m.randomUUID();
}
console.log(await f());
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 0
        # UUID format: 8-4-4-4-12 hex chars = 36 chars total
        assert len(result.stdout.strip()) == 36

    # --- Edge cases ---

    async def test_dynamic_import_computed_specifier(self, scheduler: Scheduler) -> None:
        """Dynamic import with a computed (variable) specifier."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            "const mod = 'path'; const p = await import(mod); console.log(typeof p.join);",
        )

        assert result.exit_code == 0
        assert "function" in result.stdout

    async def test_dynamic_import_promise_all(self, scheduler: Scheduler) -> None:
        """Parallel dynamic imports via Promise.all."""
        code = """\
const [p, u] = await Promise.all([import('path'), import('url')]);
console.log(typeof p.join, typeof u.parse);
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 0
        assert result.stdout.strip().count("function") == 2

    async def test_dynamic_import_coexists_with_require(self, scheduler: Scheduler) -> None:
        """Dynamic import and require resolve the same module."""
        code = """\
const imported = await import('path');
const required = require('path');
console.log(imported.join === required.join);
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 0
        assert "true" in result.stdout

    async def test_dynamic_import_session_persistence(self, scheduler: Scheduler) -> None:
        """Imported module persists across session exec() calls."""
        async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
            r1 = await session.exec("var path = await import('path');")
            assert r1.exit_code == 0

            r2 = await session.exec("console.log(path.join('a','b'));")
            assert r2.exit_code == 0
            assert "a/b" in r2.stdout

    # --- Weird cases ---

    async def test_dynamic_import_string_literal_harmless(self, scheduler: Scheduler) -> None:
        """String containing 'import()' doesn't break a real import."""
        code = """\
const s = "import()";
const path = await import('path');
console.log(path.join('a','b'));
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 0
        assert "a/b" in result.stdout

    async def test_reimport_function_not_mangled(self, scheduler: Scheduler) -> None:
        """User function named 'reimport' is not confused with import()."""
        code = """\
function reimport(x) { return x; }
console.log(reimport(42));
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 0
        assert "42" in result.stdout

    # --- Error cases ---

    async def test_dynamic_import_nonexistent_module(self, scheduler: Scheduler) -> None:
        """Dynamic import of a package that is not installed fails at once: no auto-install
        from the registry (an auto-install waits 30 s with no network)."""
        timed = await run_timed(
            scheduler, VmSpec(language=Language.JAVASCRIPT), "await import('nonexistent_xyz_module');"
        )

        assert timed.result.exit_code != 0
        assert timed.result.timing.execute_ms < 5_000

    async def test_dynamic_import_fire_and_forget_error(self, scheduler: Scheduler) -> None:
        """Unhandled dynamic import rejection is caught by unhandledRejection handler."""
        code = """\
import('nonexistent_xyz');
await new Promise(r => setTimeout(r, 200));
console.log("after");
"""
        timed = await run_timed(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert timed.result.exit_code != 0
        assert timed.result.timing.execute_ms < 5_000


# =============================================================================
# TypeScript Support (loader: 'ts' in Bun REPL wrapper)
# =============================================================================
class TestTypeScriptSupport:
    """TypeScript syntax is accepted and transpiled by the Bun REPL wrapper."""

    # --- Normal cases: core TS features ---

    async def test_ts_type_annotation(self, scheduler: Scheduler) -> None:
        """Basic type annotation on variable."""
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), "const x: number = 42; console.log(x);"
        )

        assert result.exit_code == 0
        assert "42" in result.stdout

    async def test_ts_interface(self, scheduler: Scheduler) -> None:
        """Interface definition and typed object."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            'interface User { name: string; age: number } const u: User = { name: "Alice", age: 30 }; console.log(u.name, u.age);',
        )

        assert result.exit_code == 0
        assert "Alice" in result.stdout
        assert "30" in result.stdout

    async def test_ts_generic_function(self, scheduler: Scheduler) -> None:
        """Generic function with type parameter."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            "function identity<T>(x: T): T { return x; } console.log(identity(7));",
        )

        assert result.exit_code == 0
        assert "7" in result.stdout

    async def test_ts_enum(self, scheduler: Scheduler) -> None:
        """TypeScript enum."""
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), "enum Color { Red, Green, Blue } console.log(Color.Green);"
        )

        assert result.exit_code == 0
        assert "1" in result.stdout

    async def test_ts_type_alias(self, scheduler: Scheduler) -> None:
        """Type alias."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            'type Pair = [number, string]; const p: Pair = [1, "hello"]; console.log(p[0], p[1]);',
        )

        assert result.exit_code == 0
        assert "1" in result.stdout
        assert "hello" in result.stdout

    async def test_ts_as_cast(self, scheduler: Scheduler) -> None:
        """'as' type assertion."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            'const x: unknown = "hello"; console.log((x as string).length);',
        )

        assert result.exit_code == 0
        assert "5" in result.stdout

    # --- Edge cases ---

    async def test_ts_generic_arrow_function(self, scheduler: Scheduler) -> None:
        """Generic arrow function (valid with 'ts' loader, ambiguous with 'tsx')."""
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), 'const id = <T>(x: T): T => x; console.log(id("hi"));'
        )

        assert result.exit_code == 0
        assert "hi" in result.stdout

    async def test_ts_optional_chaining_with_types(self, scheduler: Scheduler) -> None:
        """Optional chaining with typed objects."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            'const o: { a?: { b?: number } } = {}; console.log(o?.a?.b ?? "none");',
        )

        assert result.exit_code == 0
        assert "none" in result.stdout

    async def test_ts_type_only_constructs_stripped(self, scheduler: Scheduler) -> None:
        """Type-only constructs are erased, no runtime footprint."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            "type Foo = { x: number }; const f: Foo = { x: 1 }; console.log(f.x);",
        )

        assert result.exit_code == 0
        assert "1" in result.stdout

    async def test_ts_plain_js_still_works(self, scheduler: Scheduler) -> None:
        """Plain JavaScript is unaffected by the 'ts' loader (regression guard)."""
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), "var arr = [1, 2, 3]; console.log(arr.length);"
        )

        assert result.exit_code == 0
        assert "3" in result.stdout

    async def test_ts_top_level_await_with_types(self, scheduler: Scheduler) -> None:
        """Top-level await combined with TypeScript annotations."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            "const r: number = await Promise.resolve(42); console.log(r);",
        )

        assert result.exit_code == 0
        assert "42" in result.stdout

    async def test_ts_empty_object_type(self, scheduler: Scheduler) -> None:
        """Empty object type annotation."""
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), "const x: {} = {}; console.log(typeof x);"
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
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

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
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 0
        assert "count=3" in result.stdout

    async def test_ts_types_are_erased_at_runtime(self, scheduler: Scheduler) -> None:
        """Types are erased — no runtime type checking (by design)."""
        # "number" annotation but assigned a string via `as any` — runs fine
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), 'const x: number = "hello" as any; console.log(typeof x);'
        )

        assert result.exit_code == 0
        assert "string" in result.stdout

    # --- Error cases ---

    async def test_ts_invalid_syntax(self, scheduler: Scheduler) -> None:
        """Invalid TypeScript syntax is rejected."""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), "const x: = 42;")

        assert result.exit_code != 0


# =============================================================================
# RAW/Shell Edge Cases
# =============================================================================
class TestRawEdgeCases:
    """Edge cases for RAW/shell execution."""

    async def test_raw_command_not_found(self, scheduler: Scheduler) -> None:
        """Non-existent command."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "nonexistent_command_xyz123")

        assert result.exit_code != 0
        assert "not found" in result.stderr.lower() or "command not found" in result.stderr.lower()

    async def test_raw_pipe(self, scheduler: Scheduler) -> None:
        """Shell pipe."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "echo 'hello world' | tr 'a-z' 'A-Z'")

        assert result.exit_code == 0
        assert "HELLO WORLD" in result.stdout

    async def test_raw_redirect(self, scheduler: Scheduler) -> None:
        """Shell redirect."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "echo 'test' > /tmp/test.txt && cat /tmp/test.txt"
        )

        assert result.exit_code == 0
        assert "test" in result.stdout

    async def test_raw_environment_variable(self, scheduler: Scheduler) -> None:
        """Shell environment variable."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "MY_VAR='hello' && echo $MY_VAR")

        # Shell variable expansion
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"

    # ---- Shebang handling (temp-file dispatch) ----

    async def test_raw_shebang_awk(self, scheduler: Scheduler) -> None:
        """AWK shebang executes via kernel binfmt_script."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), '#!/usr/bin/awk -f\nBEGIN { print "shebang_awk_ok" }'
        )

        assert result.exit_code == 0
        assert "shebang_awk_ok" in result.stdout

    async def test_raw_shebang_sh(self, scheduler: Scheduler) -> None:
        """Shell shebang via temp file."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\necho shebang_sh_ok")

        assert result.exit_code == 0
        assert "shebang_sh_ok" in result.stdout

    async def test_raw_shebang_bash(self, scheduler: Scheduler) -> None:
        """Bash shebang via temp file."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/bash\necho shebang_bash_ok")

        assert result.exit_code == 0
        assert "shebang_bash_ok" in result.stdout

    async def test_raw_shebang_sed(self, scheduler: Scheduler) -> None:
        """Sed shebang processes input."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\necho hello | sed 's/hello/world/'"
        )

        assert result.exit_code == 0
        assert "world" in result.stdout

    async def test_raw_shebang_nonzero_exit(self, scheduler: Scheduler) -> None:
        """Non-zero exit code propagates through temp-file wrapper."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\nexit 42")

        assert result.exit_code == 42

    async def test_raw_shebang_stderr(self, scheduler: Scheduler) -> None:
        """Stderr output from shebanged script."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\necho err_msg >&2")

        assert "err_msg" in result.stderr

    async def test_raw_shebang_env_vars(self, scheduler: Scheduler) -> None:
        """Exported env vars are inherited by shebanged subprocess."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\necho $MY_VAR", env_vars={"MY_VAR": "from_env"}
        )

        assert result.exit_code == 0
        assert "from_env" in result.stdout

    async def test_raw_shebang_empty_body(self, scheduler: Scheduler) -> None:
        """Shebang with empty body is valid."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\n")

        assert result.exit_code == 0

    async def test_raw_shebang_special_chars(self, scheduler: Scheduler) -> None:
        """Special characters are preserved (heredoc quoting prevents expansion)."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\necho 'quotes \"double\" $dollar `backtick`'"
        )

        assert result.exit_code == 0
        assert "$dollar" in result.stdout
        assert "`backtick`" in result.stdout

    async def test_raw_shebang_sentinel_in_code(self, scheduler: Scheduler) -> None:
        """Sentinel collision in code is handled."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), '#!/bin/sh\necho "_EXEC_SANDBOX_EOF_"')

        assert result.exit_code == 0
        assert "_EXEC_SANDBOX_EOF_" in result.stdout

    async def test_raw_shebang_nonexistent_interpreter(self, scheduler: Scheduler) -> None:
        """Non-existent interpreter fails with non-zero exit."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "#!/usr/bin/nonexistent_interp_xyz\necho hello"
        )

        assert result.exit_code != 0

    async def test_raw_shebang_no_newline(self, scheduler: Scheduler) -> None:
        """Shebang with no trailing newline and no body."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/sh")

        assert result.exit_code == 0

    async def test_raw_no_shebang_hash_comment(self, scheduler: Scheduler) -> None:
        """Plain # comment is not treated as shebang."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "# just a comment\necho hello")

        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_raw_shebang_with_leading_space(self, scheduler: Scheduler) -> None:
        """Leading space means no shebang detection; eval path used."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "  #!/bin/sh\necho hello")

        # Leading space: bash treats #! as comment, echo runs normally
        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_raw_shebang_multiline_output(self, scheduler: Scheduler) -> None:
        """AWK shebang producing multiline output."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "#!/usr/bin/awk -f\nBEGIN { for(i=1;i<=100;i++) print i }"
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
        result = await run_once(
            scheduler,
            VmSpec(language=Language.RAW),
            '#!/bin/sh\nprintf "line1\\nline2\\n" | awk \'BEGIN{print "start"} /line/{print "matched:"$0} END{print "end"}\'',
        )

        assert result.exit_code == 0
        assert "start" in result.stdout
        assert "matched:line1" in result.stdout
        assert "matched:line2" in result.stdout
        assert "end" in result.stdout


# =============================================================================
# HOME Environment: HOME must equal /home/user (SANDBOX_ROOT) for uid=1000
# =============================================================================


class TestHomeEnvironmentNormal:
    """HOME=/home/user for all REPL types (uid=1000 runs in /home/user)."""

    async def test_raw_home_is_sandbox_root(self, scheduler: Scheduler) -> None:
        """Raw shell $HOME equals /home/user."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "echo $HOME")
        assert result.exit_code == 0
        assert result.stdout.strip() == "/home/user"

    async def test_python_home_is_sandbox_root(self, scheduler: Scheduler) -> None:
        """Python os.environ['HOME'] equals /home/user."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "import os; print(os.environ['HOME'])")
        assert result.exit_code == 0
        assert result.stdout.strip() == "/home/user"

    async def test_javascript_home_is_sandbox_root(self, scheduler: Scheduler) -> None:
        """JavaScript process.env.HOME equals /home/user."""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), "console.log(process.env.HOME)")
        assert result.exit_code == 0
        assert result.stdout.strip() == "/home/user"

    async def test_raw_tilde_expansion(self, scheduler: Scheduler) -> None:
        """Tilde expands to /home/user in raw shell."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "echo ~")
        assert result.exit_code == 0
        assert result.stdout.strip() == "/home/user"

    async def test_raw_home_writable(self, scheduler: Scheduler) -> None:
        """Files can be created under $HOME."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "echo ok > $HOME/test_home.txt && cat $HOME/test_home.txt"
        )
        assert result.exit_code == 0
        assert "ok" in result.stdout


class TestHomeEnvironmentEdge:
    """Edge cases for HOME consistency."""

    async def test_raw_home_persists_across_executions(self, scheduler: Scheduler) -> None:
        """HOME stays /home/user on second execution in same session."""
        async with await scheduler.new(VmSpec(language=Language.RAW)) as session:
            outputs = [(await session.exec("echo $HOME")).stdout.strip() for _ in range(2)]
        assert outputs == ["/home/user", "/home/user"]

    async def test_raw_subprocess_inherits_home(self, scheduler: Scheduler) -> None:
        """Child processes inherit correct HOME."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "bash -c 'echo $HOME'")
        assert result.exit_code == 0
        assert result.stdout.strip() == "/home/user"

    async def test_raw_env_vars_coexist_with_home(self, scheduler: Scheduler) -> None:
        """Custom env_vars don't clobber HOME."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "echo $HOME $MY_VAR", env_vars={"MY_VAR": "hello"}
        )
        assert result.exit_code == 0
        assert "/home/user" in result.stdout
        assert "hello" in result.stdout

    async def test_python_pathlib_home(self, scheduler: Scheduler) -> None:
        """pathlib.Path.home() returns /home/user."""
        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), "from pathlib import Path; print(Path.home())"
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "/home/user"


class TestHomeEnvironmentWeird:
    """Weird corner cases for HOME."""

    async def test_raw_home_matches_passwd(self, scheduler: Scheduler) -> None:
        """$HOME matches the home dir in /etc/passwd for uid=1000."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.RAW),
            "echo HOME=$HOME; awk -F: '$3==1000{print \"passwd=\"$6}' /etc/passwd",
        )
        assert result.exit_code == 0
        assert "HOME=/home/user" in result.stdout
        assert "passwd=/home/user" in result.stdout

    async def test_raw_cd_tilde_pwd(self, scheduler: Scheduler) -> None:
        """cd ~ + pwd returns /home/user."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "cd ~ && pwd")
        assert result.exit_code == 0
        assert result.stdout.strip() == "/home/user"

    async def test_python_expanduser(self, scheduler: Scheduler) -> None:
        """os.path.expanduser('~') returns /home/user."""
        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), "import os; print(os.path.expanduser('~'))"
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "/home/user"


class TestHomeEnvironmentOutOfBounds:
    """Out-of-bounds / adversarial HOME scenarios."""

    async def test_raw_user_can_override_home(self, scheduler: Scheduler) -> None:
        """User can override HOME via env_vars (not blocked)."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "echo $HOME", env_vars={"HOME": "/tmp"})
        assert result.exit_code == 0
        assert result.stdout.strip() == "/tmp"

    async def test_python_user_can_override_home(self, scheduler: Scheduler) -> None:
        """User can override HOME via env_vars in Python."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.PYTHON),
            "import os; print(os.environ['HOME'])",
            env_vars={"HOME": "/tmp"},
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "/tmp"


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
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert "CAUGHT_EOF" in result.stdout

    async def test_python_stdin_iteration_empty(self, scheduler: Scheduler) -> None:
        """for line in sys.stdin iterates zero times at EOF."""
        code = """\
import sys
count = 0
for line in sys.stdin:
    count += 1
print(f"COUNT={count}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert "COUNT=0" in result.stdout

    async def test_shell_read_eof(self, scheduler: Scheduler) -> None:
        """Shell read returns failure at EOF, allowing fallback."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), 'read var || echo "NO_INPUT"')
        assert result.exit_code == 0
        assert "NO_INPUT" in result.stdout

    async def test_js_stdin_read_null(self, scheduler: Scheduler) -> None:
        """process.stdin.read() returns null at EOF."""
        code = """\
const d = process.stdin.read();
console.log(String(d));
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)
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
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
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
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
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
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert "b''" in result.stdout

    async def test_python_stdin_readline_empty(self, scheduler: Scheduler) -> None:
        """sys.stdin.readline() returns '' at EOF (not EOFError)."""
        code = """\
import sys
line = sys.stdin.readline()
print(repr(line))
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert "''" in result.stdout

    async def test_session_reuse_after_stdin_read(self, scheduler: Scheduler) -> None:
        """REPL protocol channel is unaffected after user code reads stdin."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
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
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert "TYPE=int" in result.stdout

    async def test_python_stdin_isatty_false(self, scheduler: Scheduler) -> None:
        """sys.stdin.isatty() returns False (/dev/null is not a TTY)."""
        code = """\
import sys
print(f"ISATTY={sys.stdin.isatty()}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
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
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert "JSON_DECODE_ERROR" in result.stdout

    async def test_python_select_stdin_readable(self, scheduler: Scheduler) -> None:
        """/dev/null reports as readable via select (immediate EOF)."""
        code = """\
import select, sys
r, _, _ = select.select([sys.stdin], [], [], 0)
print(f"READABLE={len(r) > 0}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert "READABLE=True" in result.stdout

    async def test_shell_wc_stdin_zero(self, scheduler: Scheduler) -> None:
        """wc -l with no file arg reads stdin (/dev/null), reports 0 lines."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "wc -l")
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
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=3)
        # SIGALRM (signal 14) kills the REPL process → exit_code = 128 + 14 = 142.
        # On slow hosts, the 3s host timeout may fire first → exit_code = -1.
        assert result.exit_code in (142, -1)

    async def test_python_stdin_write_raises(self, scheduler: Scheduler) -> None:
        """Writing to stdin (opened read-only from /dev/null) raises."""
        code = """\
import sys
try:
    sys.stdin.write("x")
except Exception as e:
    print(f"ERROR={type(e).__name__}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
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
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=3)
        # Either blocks until SIGALRM kills the REPL (128+14=142), permission denied,
        # or the 3s host timeout fires first on slow hosts (exit_code=-1).
        assert result.exit_code in (142, -1) or "PERMISSION_ERROR" in result.stdout

    async def test_python_fileinput_stdin_eof(self, scheduler: Scheduler) -> None:
        """fileinput.input('-') iterates zero times at EOF."""
        code = """\
import fileinput
count = 0
for line in fileinput.input("-"):
    count += 1
print(f"COUNT={count}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert "COUNT=0" in result.stdout

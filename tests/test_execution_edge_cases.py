"""Edge case, weird, and out-of-bounds tests for code execution.

Tests that the execution system handles unusual inputs gracefully:
1. Edge cases: Empty code, large output, special characters
2. Weird cases: Binary data, null bytes, infinite loops
3. Out of bounds: Memory exhaustion, deep recursion
4. Error cases: Syntax errors, import errors, runtime errors
"""

import io
import logging
import re
import time

import pytest

from exec_sandbox.exceptions import ErrorCode, InvalidInputError, TransientError
from exec_sandbox.models import Language
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once, skip_unless_hwaccel, skip_unless_linux


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
        """Output just under the guest's 1 MB stdout limit arrives whole."""
        # 999 lines x 1001 bytes = 999,999 bytes
        code = """
for i in range(999):
    print('x' * 1000)
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        assert result.stdout == ("x" * 1000 + "\n") * 999

    async def test_many_lines_output(self, scheduler: Scheduler) -> None:
        """10,000 short lines arrive whole and in order."""
        code = """
for i in range(10000):
    print(i)
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=30)

        assert result.exit_code == 0
        assert result.stdout == "".join(f"{i}\n" for i in range(10000))

    async def test_very_long_single_line(self, scheduler: Scheduler) -> None:
        """A 10 KB line of code runs, and its 10 KB output line arrives whole."""
        long_string = "x" * 10000
        code = f"print('{long_string}')"

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        assert result.stdout == long_string + "\n"

    async def test_special_characters_in_output(self, scheduler: Scheduler) -> None:
        """Tabs, newlines, backslashes, quotes and non-ASCII text arrive unchanged."""
        code = r"""
print("Tab:\there")
print("Newline in string: line1\nline2")
print("Backslash: \\")
print("Quote: \"hello\"")
print("Unicode: café ñ 你好 🎉")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        assert result.stdout == (
            'Tab:\there\nNewline in string: line1\nline2\nBackslash: \\\nQuote: "hello"\nUnicode: café ñ 你好 🎉\n'
        )

    async def test_long_multibyte_output_comes_out_whole(self, scheduler: Scheduler) -> None:
        """Output many pipe reads long, all 3-byte characters: a read that ends inside a
        character does not cut it (no U+FFFD), on stdout and on stderr."""
        code = "import sys; print('中' * 100_000); print('文' * 30_000, file=sys.stderr)"

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, result.stderr[:200]
        assert result.stdout == "中" * 100_000 + "\n"
        assert result.stderr == "文" * 30_000 + "\n"

    async def test_a_long_line_of_control_characters_comes_out_whole(self, scheduler: Scheduler) -> None:
        """90000 control characters on one stderr line (no newline before the end): each is
        6 bytes in JSON, so one frame of it would pass the host's line limit; the guest sends
        frames of at most 64 KB of text, and the line comes out whole."""
        code = "import sys; sys.stderr.write('\\x01' * 90_000 + '\\n')"

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        assert result.stderr == "\x01" * 90_000 + "\n"

    async def test_ansi_escape_codes(self, scheduler: Scheduler) -> None:
        """ANSI escape codes pass through unchanged."""
        code = r"""
print("\033[31mRed text\033[0m")
print("\033[1mBold text\033[0m")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        assert result.stdout == "\x1b[31mRed text\x1b[0m\n\x1b[1mBold text\x1b[0m\n"


# =============================================================================
# Weird Cases: Unusual behavior that should be handled gracefully
# =============================================================================
class TestWeirdCases:
    """Weird inputs that might cause problems."""

    @pytest.mark.parametrize("language", [Language.PYTHON, Language.JAVASCRIPT, Language.RAW])
    async def test_null_bytes_rejected_all_languages(self, scheduler: Scheduler, language: Language) -> None:
        """Null bytes rejected regardless of language."""
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.CODE_INVALID, match="null bytes"):
            await run_once(scheduler, VmSpec(language=language), "echo hi\x00")

    async def test_escaped_null_repr_accepted(self, scheduler: Scheduler) -> None:
        r"""Literal '\\x00' string (no actual null byte) executes fine."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "print(repr('hello\\x00world'))")
        assert result.exit_code == 0
        assert result.stdout == "'hello\\x00world'\n"

    async def test_binary_data_in_output(self, scheduler: Scheduler) -> None:
        """All 256 byte values on stdout: the ASCII bytes (NUL included) pass through, each
        byte that is not UTF-8 becomes one U+FFFD, and the output after them is intact."""
        code = """
import sys
sys.stdout.buffer.write(bytes(range(256)))
sys.stdout.buffer.write(b"\\nDONE\\n")
sys.stdout.buffer.flush()
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        assert result.stdout == "".join(map(chr, range(128))) + "\ufffd" * 128 + "\nDONE\n"

    async def test_infinite_loop_times_out(self, scheduler: Scheduler) -> None:
        """An infinite loop ends at its timeout: exit code -1 and a timeout message."""
        code = """
while True:
    pass
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=3)

        assert result.exit_code == -1
        assert "Execution timeout after 3s" in result.stderr

    @skip_unless_hwaccel
    async def test_infinite_output_times_out(self, scheduler: Scheduler) -> None:
        """Infinite output ends at its timeout, with the output that came before it.

        Requires hwaccel: 3s timeout is too short for TCG (~5-8x slower) —
        the VM cannot produce any output before the deadline expires.
        """
        code = """
while True:
    print("spam")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=3)

        assert result.exit_code == -1
        assert result.stdout.startswith("spam\nspam\n")

    async def test_exit_immediately(self, scheduler: Scheduler) -> None:
        """sys.exit(0) ends the code: exit code 0, and the line after it does not run."""
        code = """
import sys
sys.exit(0)
print("should not print")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0
        assert result.stdout == ""

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=5)

        assert result.exit_code == -1
        assert result.timing.execute_ms < 10_000

    @skip_unless_linux
    async def test_disk_writes_do_not_fill_the_vm_cgroup(self, scheduler: Scheduler) -> None:
        """A VM writes 256 MiB to its home disk: the writes skip the host page cache, so its
        cgroup never reaches memory.max (buffered, a 1 GiB write reached it 480 times).
        Linux, with VM cgroups and a home disk folder that takes O_DIRECT. The test reads
        session._vm: no public API gives the VM's cgroup or home disk folder."""
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

    async def test_exec_after_exit_137_fails_at_once(self, running_vm: QemuVM, execute_timeout: int) -> None:
        """Exit 137 retires the VM: the guest refuses the next code with a reply at once.

        Without the reply (a dropped connection), the host waits for its whole deadline
        (timeout + margin), then raises OutcomeUnknownError."""
        killed = await running_vm.execute(
            "import os; os.kill(os.getpid(), 9)",
            timeout_seconds=execute_timeout,
            env_vars=None,
            on_stdout=None,
            on_stderr=None,
        )
        assert killed.exit_code == 137

        start = time.monotonic()
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GUEST_UNAVAILABLE, match="retired"):
            await running_vm.execute(
                "print('second')", timeout_seconds=1, env_vars=None, on_stdout=None, on_stderr=None
            )
        assert time.monotonic() - start < 1

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
        assert result.stdout == "READY\nRECEIVED_SIGTERM\n"
        # Process caught SIGTERM and called sys.exit(42) — normal exit, not a signal kill
        assert result.exit_code == 42

    async def test_partial_stderr_line_at_timeout_is_sent_once(self, scheduler: Scheduler) -> None:
        """A stderr line with no newline, then a timeout whose SIGTERM handler exits: the
        partial line reaches the caller once. At the deadline the agent keeps the partial
        line in its line buffer, and the drain after SIGTERM continues that line: a copy
        sent at the deadline too would go out twice ("partialpartial")."""
        code = """
import signal
import sys
import time

def handler(signum, frame):
    sys.exit(42)

signal.signal(signal.SIGTERM, handler)
sys.stderr.write("partial")
sys.stderr.flush()
time.sleep(60)
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=2)

        assert (result.exit_code, result.stderr) == (42, "partial")

    async def test_lines_with_no_newline_at_timeout_with_no_handler_go_out(self, scheduler: Scheduler) -> None:
        """Progress lines with no newline on both streams, then a timeout with no SIGTERM
        handler: SIGTERM ends the REPL with no sentinel, and both lines still reach the
        caller whole, stderr's before the timeout error. The stdout line ends in "__", a
        possible start of the agent's stdout end marker."""
        code = """
import sys
import time

sys.stdout.write("Progress [__")
sys.stdout.flush()
sys.stderr.write("Processing...")
sys.stderr.flush()
time.sleep(60)
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=2)

        assert (result.exit_code, result.stdout, result.stderr) == (
            -1,
            "Progress [__",
            "Processing...[timeout_error] Execution timeout after 2s",
        )

    async def test_timeout_warning_shows_where_the_exec_waits(
        self, scheduler: Scheduler, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An exec whose child sleeps reaches its timeout: the host's timeout warning holds
        the guest's [repl-timeout] line, with each process of the exec, its state and its
        wait channel, so a stall on a CI runner shows the command and where it waited. The
        child is forked (`&`): BusyBox can run a plain `sleep` inside the shell itself."""
        with caplog.at_level(logging.WARNING, logger="exec_sandbox.qemu_vm"):
            result = await run_once(scheduler, VmSpec(language=Language.RAW), "sleep 60 & wait", timeout_seconds=2)

        assert result.exit_code == -1
        [warning] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Execution timeout")]
        assert re.search(r"\[repl-timeout\] pid=\d+ .* members=\[\d+ \S+ S wchan=\S+ stack=\[", warning), warning

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
        assert result.stdout == "STARTING\nDONE\n"

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

        assert result.exit_code == -1
        assert result.stdout == "IGNORING_SIGTERM\n"
        # SIGTERM ignored: SIGKILL comes after the 5 s grace period (an end at ~2 s means SIGTERM worked)
        assert result.timing.execute_ms >= 5000
        assert result.timing.execute_ms < 12000

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

        assert result.exit_code == -1
        assert result.stdout == "NESTED_SPAWNED\n"
        # If nested processes weren't killed, would take 60s
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

        assert result.exit_code == -1
        assert result.stdout == "SPAWNED\n"
        # Key assertion: completes in reasonable time (not 60s)
        # Expected: ~2s timeout + ~5s grace = ~7s (host backstop is 2+10=12s)
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

        assert result.exit_code == -1
        assert result.stdout == "SPAWNED\n"
        # Should complete after timeout + grace, not after 60s
        # Expected: ~15s timeout + ~5s grace = ~20s (host backstop is 15+10=25s)
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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON, memory_mb=256), code, timeout_seconds=60)

        assert "MEMORY_ERROR" in result.stdout or result.exit_code == 137, (result.exit_code, result.stderr)
        assert result.timing.execute_ms < 20_000

    async def test_deep_recursion(self, scheduler: Scheduler) -> None:
        """50,000-deep recursion runs under a raised recursion limit; past the limit, the code
        gets RecursionError and the REPL lives on."""
        code = """
import sys

def recurse(n):
    if n <= 0:
        return 0
    return 1 + recurse(n - 1)

sys.setrecursionlimit(100000)
print(f"RESULT:{recurse(50000)}")
sys.setrecursionlimit(1000)
try:
    recurse(5000)
except RecursionError:
    print("RECURSION_ERROR")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=30)

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "RESULT:50000\nRECURSION_ERROR\n"

    async def test_many_file_descriptors(self, scheduler: Scheduler) -> None:
        """Opening 10,000 file descriptors stops at the open-file limit with EMFILE, and the
        REPL lives on."""
        code = """
import errno, os
fds = []
try:
    for i in range(10000):
        fds.append(os.open("/dev/null", os.O_RDONLY))
    print(f"OPENED:{len(fds)}")
except OSError as e:
    print(f"FD_LIMIT:{errno.errorcode[e.errno]}")
finally:
    for fd in fds:
        os.close(fd)
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "FD_LIMIT:EMFILE\n"

    async def test_subprocess_spawning(self, scheduler: Scheduler) -> None:
        """Spawning subprocesses from code."""
        code = """
import subprocess
result = subprocess.run(["echo", "hello from subprocess"], capture_output=True, text=True)
print(f"SUBPROCESS:{result.stdout.strip()}")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert (result.exit_code, result.stdout) == (0, "SUBPROCESS:hello from subprocess\n"), result.stderr


# =============================================================================
# Error Cases: Code that should fail gracefully
# =============================================================================
class TestErrorCases:
    """Code that produces errors should be handled gracefully."""

    async def test_syntax_error(self, scheduler: Scheduler) -> None:
        """A Python syntax error: exit code 1, SyntaxError on stderr."""
        code = """
def broken(
    print("missing closing paren"
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 1
        assert "SyntaxError: " in result.stderr

    async def test_division_by_zero(self, scheduler: Scheduler) -> None:
        """Division by zero: exit code 1, ZeroDivisionError on stderr."""
        code = """
x = 1 / 0
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 1
        assert "ZeroDivisionError: division by zero" in result.stderr

    async def test_permission_denied(self, scheduler: Scheduler) -> None:
        """A write to a read-only file system: exit code 1, EROFS on stderr.

        /proc/version is masked (bind-mounted to /dev/null), so a write there
        succeeds. /etc/passwd is on the read-only root file system, so a write
        always fails with EROFS (errno 30).
        """
        code = """
with open("/etc/passwd", "w") as f:
    f.write("test")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 1
        assert "OSError: [Errno 30] Read-only file system: '/etc/passwd'" in result.stderr


# =============================================================================
# JavaScript Edge Cases
# =============================================================================
class TestJavaScriptEdgeCases:
    """Edge cases specific to JavaScript/Bun."""

    async def test_js_syntax_error(self, scheduler: Scheduler) -> None:
        """A JavaScript syntax error: exit code 1, a parse error on stderr."""
        code = """
function broken( {
    console.log("missing paren");
}
"""

        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 1
        assert "Parse error" in result.stderr

    async def test_js_undefined_variable(self, scheduler: Scheduler) -> None:
        """An undefined variable: exit code 1, ReferenceError on stderr."""
        code = """
console.log(undefinedVariable);
"""

        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 1
        assert "ReferenceError: undefinedVariable is not defined" in result.stderr

    async def test_js_async_await(self, scheduler: Scheduler) -> None:
        """The exec waits for a pending timer that a not-awaited async function starts."""
        code = """
async function main() {
    await new Promise(resolve => setTimeout(resolve, 100));
    console.log("ASYNC_DONE");
}
main();
"""

        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 0
        assert result.stdout == "ASYNC_DONE\n"

    async def test_js_promise_rejection(self, scheduler: Scheduler) -> None:
        """An unhandled promise rejection: exit code 1, the error on stderr."""
        code = """
Promise.reject(new Error("test rejection"));
"""

        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 1
        assert "Error: test rejection" in result.stderr

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
        assert (result.exit_code, result.stdout) == (0, "type=object\nnow_type=function\nis_number=true\n"), (
            result.stderr
        )


# =============================================================================
# JavaScript Dynamic Import (the main context's module loader)
# =============================================================================
class TestJavaScriptDynamicImport:
    """ESM dynamic import() in the JavaScript/Bun VM context."""

    # --- Normal cases ---

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
        assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\n", result.stdout)

    # --- Edge cases ---

    async def test_dynamic_import_coexists_with_require(self, scheduler: Scheduler) -> None:
        """Dynamic import and require resolve the same module."""
        code = """\
const imported = await import('path');
const required = require('path');
console.log(imported.join === required.join);
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 0
        assert result.stdout == "true\n"

    async def test_dynamic_import_session_persistence(self, scheduler: Scheduler) -> None:
        """Imported module persists across session exec() calls."""
        async with await scheduler.new(VmSpec(language=Language.JAVASCRIPT)) as session:
            r1 = await session.exec("var path = await import('path');")
            assert r1.exit_code == 0

            r2 = await session.exec("console.log(path.join('a','b'));")
            assert r2.exit_code == 0
            assert r2.stdout == "a/b\n"

    # --- Weird cases ---

    async def test_dynamic_import_nonexistent_module(self, scheduler: Scheduler) -> None:
        """Dynamic import of a package that is not installed fails at once: no auto-install
        from the registry (an auto-install waits 30 s with no network)."""
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), "await import('nonexistent_xyz_module');"
        )

        assert result.exit_code == 1
        assert "Cannot find package 'nonexistent_xyz_module'" in result.stderr
        assert result.timing.execute_ms < 5_000

    async def test_dynamic_import_fire_and_forget_error(self, scheduler: Scheduler) -> None:
        """A dynamic import that nobody awaits and that fails: the code runs on, then the
        exec ends with exit code 1 and the error on stderr, at once (no registry wait)."""
        code = """\
import('nonexistent_xyz');
await new Promise(r => setTimeout(r, 200));
console.log("after");
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert result.exit_code == 1
        assert result.stdout == "after\n"
        assert "Cannot find package 'nonexistent_xyz'" in result.stderr
        assert result.timing.execute_ms < 5_000


# =============================================================================
# TypeScript Support (loader: 'ts' in Bun REPL wrapper)
# =============================================================================
class TestTypeScriptSupport:
    """TypeScript syntax is accepted and transpiled by the Bun REPL wrapper."""

    # --- Normal cases: core TS features ---

    async def test_ts_enum(self, scheduler: Scheduler) -> None:
        """TypeScript enum."""
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), "enum Color { Red, Green, Blue } console.log(Color.Green);"
        )

        assert result.exit_code == 0
        assert result.stdout == "1\n"

    async def test_ts_generic_arrow_function(self, scheduler: Scheduler) -> None:
        """Generic arrow function (valid with 'ts' loader, ambiguous with 'tsx')."""
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), 'const id = <T>(x: T): T => x; console.log(id("hi"));'
        )

        assert result.exit_code == 0
        assert result.stdout == "hi\n"

    async def test_ts_top_level_await_with_types(self, scheduler: Scheduler) -> None:
        """Top-level await combined with TypeScript annotations."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.JAVASCRIPT),
            "const r: number = await Promise.resolve(42); console.log(r);",
        )

        assert result.exit_code == 0
        assert result.stdout == "42\n"

    async def test_ts_type_only_syntax_is_erased(self, scheduler: Scheduler) -> None:
        """Type-only syntax (interfaces, aliases, annotations, generics, casts, `!`) compiles
        away: the code runs as the JavaScript it holds, with no type check at run time."""
        code = """\
interface User { name: string; age: number }
type Pair = [number, string];
function identity<T>(x: T): T { return x; }
const u: User = { name: "Alice", age: 30 };
const p: Pair = [1, "hello"];
const m = new Map<string, Array<[number, string]>>([["key", [[1, "a"]]]]);
const o: { a?: { b?: number } } = {};
const x: number = "text" as any;
console.log(u.name, identity(p[1]), m.get("key")![0][1], o?.a?.b ?? "none", typeof x);
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)

        assert (result.exit_code, result.stdout) == (0, "Alice hello a none string\n"), result.stderr

    # --- Error cases ---

    async def test_ts_invalid_syntax(self, scheduler: Scheduler) -> None:
        """Invalid TypeScript syntax is rejected."""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), "const x: = 42;")

        assert result.exit_code == 1
        assert "Unexpected =" in result.stderr


# =============================================================================
# RAW/Shell Edge Cases
# =============================================================================
class TestRawEdgeCases:
    """Edge cases for RAW/shell execution."""

    async def test_raw_command_not_found(self, scheduler: Scheduler) -> None:
        """Non-existent command."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "nonexistent_command_xyz123")

        assert result.exit_code == 127
        assert "nonexistent_command_xyz123: command not found" in result.stderr

    async def test_raw_pipe(self, scheduler: Scheduler) -> None:
        """Shell pipe."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "echo 'hello world' | tr 'a-z' 'A-Z'")

        assert result.exit_code == 0
        assert result.stdout == "HELLO WORLD\n"

    async def test_raw_redirect(self, scheduler: Scheduler) -> None:
        """Shell redirect."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "echo 'test' > /tmp/test.txt && cat /tmp/test.txt"
        )

        assert result.exit_code == 0
        assert result.stdout == "test\n"

    async def test_raw_environment_variable(self, scheduler: Scheduler) -> None:
        """Shell environment variable."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "MY_VAR='hello' && echo $MY_VAR")

        # Shell variable expansion
        assert result.exit_code == 0
        assert result.stdout == "hello\n"

    # ---- Shebang handling (temp-file dispatch) ----

    async def test_raw_shebang_awk(self, scheduler: Scheduler) -> None:
        """AWK shebang executes via kernel binfmt_script."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), '#!/usr/bin/awk -f\nBEGIN { print "shebang_awk_ok" }'
        )

        assert result.exit_code == 0
        assert result.stdout == "shebang_awk_ok\n"

    async def test_raw_shebang_sh(self, scheduler: Scheduler) -> None:
        """Shell shebang via temp file."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\necho shebang_sh_ok")

        assert result.exit_code == 0
        assert result.stdout == "shebang_sh_ok\n"

    async def test_raw_shebang_bash(self, scheduler: Scheduler) -> None:
        """Bash shebang via temp file."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/bash\necho shebang_bash_ok")

        assert result.exit_code == 0
        assert result.stdout == "shebang_bash_ok\n"

    async def test_raw_shebang_sed(self, scheduler: Scheduler) -> None:
        """A sh shebang script runs a sed pipe."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\necho hello | sed 's/hello/world/'"
        )

        assert result.exit_code == 0
        assert result.stdout == "world\n"

    async def test_raw_shebang_nonzero_exit(self, scheduler: Scheduler) -> None:
        """Non-zero exit code propagates through temp-file wrapper."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\nexit 42")

        assert result.exit_code == 42

    async def test_raw_shebang_stderr(self, scheduler: Scheduler) -> None:
        """Stderr output from shebanged script."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\necho err_msg >&2")

        assert result.exit_code == 0
        assert result.stderr == "err_msg\n"

    async def test_raw_shebang_env_vars(self, scheduler: Scheduler) -> None:
        """Exported env vars are inherited by shebanged subprocess."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\necho $MY_VAR", env_vars={"MY_VAR": "from_env"}
        )

        assert result.exit_code == 0
        assert result.stdout == "from_env\n"

    async def test_raw_shebang_empty_body(self, scheduler: Scheduler) -> None:
        """Shebang with empty body is valid."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\n")

        assert (result.exit_code, result.stdout, result.stderr) == (0, "", "")

    async def test_raw_shebang_special_chars(self, scheduler: Scheduler) -> None:
        """Special characters are preserved (heredoc quoting prevents expansion)."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "#!/bin/sh\necho 'quotes \"double\" $dollar `backtick`'"
        )

        assert result.exit_code == 0
        assert result.stdout == 'quotes "double" $dollar `backtick`\n'

    async def test_raw_shebang_sentinel_in_code(self, scheduler: Scheduler) -> None:
        """A script line that is the heredoc sentinel alone (here a heredoc of its own): the
        sentinel grows until no line of the code is it, so the whole script runs."""
        script = "#!/bin/sh\ncat <<'X'\n_EXEC_SANDBOX_EOF_\nX\necho done"
        result = await run_once(scheduler, VmSpec(language=Language.RAW), script)

        assert result.exit_code == 0
        assert result.stdout == "_EXEC_SANDBOX_EOF_\ndone\n"

    async def test_raw_shebang_nonexistent_interpreter(self, scheduler: Scheduler) -> None:
        """Non-existent interpreter fails with non-zero exit."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "#!/usr/bin/nonexistent_interp_xyz\necho hello"
        )

        assert result.exit_code == 126
        assert "/usr/bin/nonexistent_interp_xyz: bad interpreter" in result.stderr

    async def test_raw_shebang_no_newline(self, scheduler: Scheduler) -> None:
        """Shebang with no trailing newline and no body."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "#!/bin/sh")

        assert (result.exit_code, result.stdout, result.stderr) == (0, "", "")

    async def test_raw_no_shebang_hash_comment(self, scheduler: Scheduler) -> None:
        """Plain # comment is not treated as shebang."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "# just a comment\necho hello")

        assert result.exit_code == 0
        assert result.stdout == "hello\n"

    async def test_raw_shebang_with_leading_space(self, scheduler: Scheduler) -> None:
        """Leading space means no shebang detection; eval path used."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "  #!/bin/sh\necho hello")

        # Leading space: bash treats #! as comment, echo runs normally
        assert result.exit_code == 0
        assert result.stdout == "hello\n"

    async def test_raw_shebang_multiline_output(self, scheduler: Scheduler) -> None:
        """AWK shebang producing multiline output."""
        result = await run_once(
            scheduler, VmSpec(language=Language.RAW), "#!/usr/bin/awk -f\nBEGIN { for(i=1;i<=100;i++) print i }"
        )

        assert result.exit_code == 0
        assert result.stdout == "".join(f"{i}\n" for i in range(1, 101))

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
        assert result.stdout == "start\nmatched:line1\nmatched:line2\nend\n"


# =============================================================================
# HOME Environment: HOME must equal /home/user (SANDBOX_ROOT) for uid=1000
# =============================================================================


class TestHomeEnvironmentNormal:
    """HOME=/home/user in each REPL (uid=1000 runs in /home/user). The shell's:
    TestHomeEnvironmentWeird.test_raw_home_matches_passwd."""

    async def test_python_home_is_sandbox_root(self, scheduler: Scheduler) -> None:
        """Python os.environ['HOME'] equals /home/user."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), "import os; print(os.environ['HOME'])")
        assert (result.exit_code, result.stdout) == (0, "/home/user\n"), result.stderr

    async def test_javascript_home_is_sandbox_root(self, scheduler: Scheduler) -> None:
        """JavaScript process.env.HOME equals /home/user."""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), "console.log(process.env.HOME)")
        assert (result.exit_code, result.stdout) == (0, "/home/user\n"), result.stderr


class TestHomeEnvironmentWeird:
    """Weird corner cases for HOME."""

    async def test_raw_home_matches_passwd(self, scheduler: Scheduler) -> None:
        """$HOME matches the home dir in /etc/passwd for uid=1000."""
        result = await run_once(
            scheduler,
            VmSpec(language=Language.RAW),
            "echo HOME=$HOME; awk -F: '$3==1000{print \"passwd=\"$6}' /etc/passwd",
        )
        assert (result.exit_code, result.stdout) == (0, "HOME=/home/user\npasswd=/home/user\n"), result.stderr


class TestHomeEnvironmentOutOfBounds:
    """Out-of-bounds / adversarial HOME scenarios."""

    async def test_raw_user_can_override_home(self, scheduler: Scheduler) -> None:
        """User can override HOME via env_vars (not blocked)."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "echo $HOME", env_vars={"HOME": "/tmp"})
        assert (result.exit_code, result.stdout) == (0, "/tmp\n"), result.stderr


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
        assert result.stdout == "prompt> CAUGHT_EOF\n"

    async def test_shell_read_eof(self, scheduler: Scheduler) -> None:
        """Shell read returns failure at EOF, allowing fallback."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), 'read var || echo "NO_INPUT"')
        assert result.exit_code == 0
        assert result.stdout == "NO_INPUT\n"

    async def test_js_stdin_read_null(self, scheduler: Scheduler) -> None:
        """process.stdin.read() returns null at EOF."""
        code = """\
const d = process.stdin.read();
console.log(String(d));
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)
        assert result.exit_code == 0
        assert result.stdout == "null\n"


class TestStdinEofEdge:
    """Boundary conditions and sequential stdin interactions."""

    async def test_session_reuse_after_stdin_read(self, scheduler: Scheduler) -> None:
        """REPL protocol channel is unaffected after user code reads stdin."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            r1 = await session.exec("import sys; sys.stdin.read()")
            assert r1.exit_code == 0

            r2 = await session.exec('print("STILL_WORKS")')
            assert r2.exit_code == 0
            assert r2.stdout == "STILL_WORKS\n"


class TestStdinEofWeird:
    """Valid but unusual stdin patterns."""

    async def test_python_stdin_fileno_valid(self, scheduler: Scheduler) -> None:
        """sys.stdin is a real file (fileno() works, unlike StringIO), open on /dev/null."""
        code = """\
import os, sys
print(os.readlink(f"/proc/self/fd/{sys.stdin.fileno()}"))
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "/dev/null\n"


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
        # The timeout leaves room for a cold REPL start under TCG: the alarm ends the read
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=30)
        # SIGALRM (signal 14) kills the REPL process → exit_code = 128 + 14 = 142.
        assert result.exit_code == 142

    async def test_python_open_dev_stdin_denied(self, scheduler: Scheduler) -> None:
        """/dev/stdin cannot reopen fd 0 (the protocol pipe): open() raises PermissionError.

        /dev/stdin links to /proc/self/fd/0. The REPL is not dumpable
        (PR_SET_DUMPABLE=0), so the kernel makes its /proc/self/fd root's, and
        UID 1000 cannot open it. The alarm ends a read that blocks, if the
        open ever succeeds.
        """
        code = """\
import signal
signal.alarm(2)
try:
    data = open("/dev/stdin").read()
    print(f"DATA={repr(data)}")
except PermissionError:
    print("PERMISSION_ERROR")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=3)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "PERMISSION_ERROR\n"

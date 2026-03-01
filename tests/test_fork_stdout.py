"""Tests for fork + stdout capture with the REPL PID guard.

When user code calls os.fork() inside the persistent Python REPL, the child
process inherits the entire REPL wrapper — including the while-loop and
sentinel-writing code. Without the PID guard, if the child doesn't call
os._exit() (e.g., uses sys.exit(), exit(), or falls through), the child's
copy of the REPL wrapper writes a premature sentinel to stderr. The
guest-agent treats the first sentinel as authoritative and stops capturing,
losing the parent's stdout.

The PID guard fix (in PYTHON_REPL_WRAPPER) records _repl_pid before the
while loop. After each exec(), if getpid() != _repl_pid, the wrapper flushes
stdout/stderr and calls os._exit(exit_code) — preventing the child from
writing a sentinel while preserving its buffered output.

Reliability strategy:
- waitpid() synchronization eliminates races (kernel-guaranteed ordering)
- Assert presence of markers, not ordering (POSIX pipe atomicity for <4096B)
- No time.sleep() for synchronization
- All outputs well under 64KB pipe capacity

Integration tests require QEMU + images (run 'make build-images').
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


class TestForkStdoutCapture:
    """Normal: fork + stdout capture works with the PID guard."""

    async def test_fork_child_sys_exit_parent_prints(self, scheduler: Scheduler) -> None:
        """THE reported bug: child sys.exit() no longer writes premature sentinel.

        Without PID guard, child's sys.exit() raises SystemExit caught by the
        REPL wrapper, which writes a sentinel before the parent finishes.
        """
        code = """
import os

pid = os.fork()
if pid == 0:
    # Child: print and exit via sys.exit (triggers SystemExit in REPL wrapper)
    print("CHILD_OK", flush=True)
    import sys
    sys.exit(0)

# Parent: wait for child, then print
os.waitpid(pid, 0)
print("PARENT_OK")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert "CHILD_OK" in result.stdout, f"Child output missing: {result.stdout}"
        assert "PARENT_OK" in result.stdout, f"Parent output missing: {result.stdout}"

    async def test_fork_child_falls_through_no_exit(self, scheduler: Scheduler) -> None:
        """Child prints and falls through without any exit call.

        The child returns from exec(), PID guard catches it (pid != _repl_pid),
        flushes, and calls os._exit(0).
        """
        code = """
import os

pid = os.fork()
if pid == 0:
    # Child: print and fall through (no exit call at all)
    print("CHILD_FELL_THROUGH", flush=True)
else:
    # Parent: wait for child, then print
    os.waitpid(pid, 0)
    print("PARENT_OK")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert "CHILD_FELL_THROUGH" in result.stdout, f"Child output missing: {result.stdout}"
        assert "PARENT_OK" in result.stdout, f"Parent output missing: {result.stdout}"

    async def test_fork_child_os_exit_with_explicit_flush(self, scheduler: Scheduler) -> None:
        """Baseline: child with flush=True + os._exit(0) already works.

        This pattern works without the PID guard. Proves the fix doesn't regress it.
        """
        code = """
import os

pid = os.fork()
if pid == 0:
    print("CHILD_BASELINE", flush=True)
    os._exit(0)

os.waitpid(pid, 0)
print("PARENT_BASELINE")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert "CHILD_BASELINE" in result.stdout, f"Child output missing: {result.stdout}"
        assert "PARENT_BASELINE" in result.stdout, f"Parent output missing: {result.stdout}"


class TestForkStdoutEdgeCases:
    """Edge: boundary conditions for the PID guard."""

    async def test_fork_child_os_exit_without_flush_loses_output(self, scheduler: Scheduler) -> None:
        """Child print() without flush + os._exit(0) loses child output.

        This documents a known CPython limitation (issue #61432, Won't Fix):
        os._exit() skips Python buffer flush. stdout is block-buffered on pipes,
        so unflushed output is lost. The PID guard is never reached here because
        os._exit() terminates before the REPL wrapper code.
        """
        code = """
import os

pid = os.fork()
if pid == 0:
    # No flush=True, and os._exit skips Python buffer flush
    print("CHILD_LOST")
    os._exit(0)

os.waitpid(pid, 0)
print("PARENT_PRESENT")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        # Child output is lost because os._exit() doesn't flush Python buffers
        assert "CHILD_LOST" not in result.stdout, f"Child output should be lost with os._exit: {result.stdout}"
        assert "PARENT_PRESENT" in result.stdout, f"Parent output missing: {result.stdout}"

    async def test_fork_child_nonzero_exit_code_preserved(self, scheduler: Scheduler) -> None:
        """Child sys.exit(42) propagates exit code through PID guard.

        The REPL wrapper's except SystemExit handler sets exit_code = e.code.
        The PID guard then calls os._exit(exit_code), preserving the code.
        Parent uses waitpid to verify the child's exit status.
        """
        code = """
import os

pid = os.fork()
if pid == 0:
    import sys
    sys.exit(42)

_, status = os.waitpid(pid, 0)
child_exit = os.waitstatus_to_exitcode(status)
print(f"CHILD_EXIT:{child_exit}")
print("PARENT_OK")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert "CHILD_EXIT:42" in result.stdout, f"Child exit code not preserved: {result.stdout}"
        assert "PARENT_OK" in result.stdout, f"Parent output missing: {result.stdout}"

    async def test_fork_child_sys_exit_none_is_success(self, scheduler: Scheduler) -> None:
        """Child sys.exit(None) and sys.exit() are treated as success (exit 0).

        CPython convention: sys.exit(None) is equivalent to sys.exit(0).
        The REPL wrapper must map e.code=None to exit_code=0, not 1.
        """
        code = """
import os

# Test sys.exit(None) — explicit None
pid = os.fork()
if pid == 0:
    import sys
    sys.exit(None)

_, status = os.waitpid(pid, 0)
exit_none = os.waitstatus_to_exitcode(status)

# Test sys.exit() — no argument, equivalent to sys.exit(None)
pid2 = os.fork()
if pid2 == 0:
    import sys
    sys.exit()

_, status2 = os.waitpid(pid2, 0)
exit_noarg = os.waitstatus_to_exitcode(status2)

print(f"EXIT_NONE:{exit_none}")
print(f"EXIT_NOARG:{exit_noarg}")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert "EXIT_NONE:0" in result.stdout, f"sys.exit(None) should be exit 0: {result.stdout}"
        assert "EXIT_NOARG:0" in result.stdout, f"sys.exit() should be exit 0: {result.stdout}"

    async def test_fork_child_exception_parent_continues(self, scheduler: Scheduler) -> None:
        """Child raises ValueError; REPL wrapper catches it, PID guard fires.

        The child's traceback goes to stderr (via traceback.print_exc()).
        PID guard fires with exit_code=1. Parent continues normally.
        """
        code = """
import os

pid = os.fork()
if pid == 0:
    raise ValueError("child_error_marker")

_, status = os.waitpid(pid, 0)
child_exit = os.waitstatus_to_exitcode(status)
print(f"CHILD_EXIT:{child_exit}")
print("PARENT_CONTINUES")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert "PARENT_CONTINUES" in result.stdout, f"Parent output missing: {result.stdout}"
        assert "CHILD_EXIT:1" in result.stdout, f"Child should exit with code 1: {result.stdout}"
        assert "child_error_marker" in result.stderr, f"Child traceback missing from stderr: {result.stderr}"

    async def test_fork_sequential_three_children(self, scheduler: Scheduler) -> None:
        """Three sequential fork+waitpid cycles, each child uses sys.exit(0).

        Tests that repeated PID guard invocations don't corrupt REPL state.
        """
        code = """
import os

for i in range(3):
    pid = os.fork()
    if pid == 0:
        print(f"CHILD_{i}", flush=True)
        import sys
        sys.exit(0)
    os.waitpid(pid, 0)

print("PARENT_ALL_DONE")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        for i in range(3):
            assert f"CHILD_{i}" in result.stdout, f"Child {i} output missing: {result.stdout}"
        assert "PARENT_ALL_DONE" in result.stdout, f"Parent output missing: {result.stdout}"


class TestForkStdoutWeirdCases:
    """Weird: unusual but valid fork patterns."""

    async def test_fork_nested_grandchild_stdout(self, scheduler: Scheduler) -> None:
        """Parent -> child -> grandchild. All three print markers.

        Two levels of waitpid() synchronization. Both child and grandchild
        have pid != _repl_pid, so PID guard catches both.
        """
        code = """
import os

pid = os.fork()
if pid == 0:
    # Child
    grandchild = os.fork()
    if grandchild == 0:
        # Grandchild
        print("GRANDCHILD_OK", flush=True)
        import sys
        sys.exit(0)
    # Child waits for grandchild
    os.waitpid(grandchild, 0)
    print("CHILD_OK", flush=True)
    import sys
    sys.exit(0)

# Parent waits for child
os.waitpid(pid, 0)
print("PARENT_OK")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert "GRANDCHILD_OK" in result.stdout, f"Grandchild output missing: {result.stdout}"
        assert "CHILD_OK" in result.stdout, f"Child output missing: {result.stdout}"
        assert "PARENT_OK" in result.stdout, f"Parent output missing: {result.stdout}"

    async def test_fork_child_large_output_flushed(self, scheduler: Scheduler) -> None:
        """Child generates 200 lines (~10KB) with per-line flush.

        Tests that many small writes from a forked child using sys.exit()
        are all captured correctly. Each line is individually flushed (under
        PIPE_BUF for atomic writes). Total ~10KB is well under 64KB pipe
        capacity.
        """
        code = """
import os

pid = os.fork()
if pid == 0:
    for i in range(200):
        print(f"CHILD_LINE_{i}", flush=True)
    import sys
    sys.exit(0)

os.waitpid(pid, 0)
print("PARENT_AFTER_CHILD")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        # Verify all 200 lines are present
        for i in range(200):
            assert f"CHILD_LINE_{i}" in result.stdout, f"Child line {i} missing: stdout length={len(result.stdout)}"
        assert "PARENT_AFTER_CHILD" in result.stdout, f"Parent output missing: {result.stdout}"

    async def test_fork_child_writes_stderr(self, scheduler: Scheduler) -> None:
        """Child writes to both stdout and stderr. PID guard flushes both.

        Child's stderr lines are processed by the guest-agent's line scanner
        as user stderr (no sentinel pattern match).
        """
        code = """
import os

pid = os.fork()
if pid == 0:
    import sys
    print("CHILD_STDOUT", flush=True)
    print("CHILD_STDERR", file=sys.stderr, flush=True)
    sys.exit(0)

os.waitpid(pid, 0)
print("PARENT_STDOUT")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert "CHILD_STDOUT" in result.stdout, f"Child stdout missing: {result.stdout}"
        assert "CHILD_STDERR" in result.stderr, f"Child stderr missing: {result.stderr}"
        assert "PARENT_STDOUT" in result.stdout, f"Parent stdout missing: {result.stdout}"


class TestForkStdoutOutOfBounds:
    """Out of bounds: failure modes and limits."""

    async def test_fork_result_exit_code_is_parent_not_child(self, scheduler: Scheduler) -> None:
        """Child sys.exit(99), parent completes normally. result.exit_code must be 0.

        This is the most direct test of the fix's correctness: without the PID
        guard, the child's premature sentinel reports exit_code=99. With the
        fix, only the parent writes the sentinel (exit_code=0).
        """
        code = """
import os

pid = os.fork()
if pid == 0:
    import sys
    sys.exit(99)

os.waitpid(pid, 0)
print("PARENT_DONE")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, (
            f"Expected parent exit_code 0, got {result.exit_code} "
            f"(child's code leaked via premature sentinel?): {result.stderr}"
        )
        assert "PARENT_DONE" in result.stdout, f"Parent output missing: {result.stdout}"

    async def test_fork_repl_state_survives_child_death(self, scheduler: Scheduler) -> None:
        """REPL namespace is not corrupted by the fork+PID guard cycle.

        Sets x = 42 before fork. Child sys.exit()'s. Parent prints x after
        waitpid. The child's copy of ns is destroyed with the child; the
        parent's ns is unchanged.
        """
        code = """
import os

x = 42
pid = os.fork()
if pid == 0:
    import sys
    sys.exit(0)

os.waitpid(pid, 0)
print(f"x:{x}")
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert "x:42" in result.stdout, f"REPL state corrupted: {result.stdout}"

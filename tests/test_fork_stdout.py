"""Tests for fork + stdout capture with the REPL PID guard.

When user code calls os.fork() inside the persistent Python REPL, the child
process inherits the entire REPL wrapper — including the while-loop and
sentinel-writing code. If the child does not call os._exit() (it uses
sys.exit(), exit(), or falls through), its copy of the REPL wrapper would
write a premature sentinel to stderr. The guest agent takes the first
sentinel as the end of the exec and stops capturing: the parent's stdout
would be lost.

The PID guard (PYTHON_REPL_WRAPPER, python_repl.py) records _repl_pid before
the while loop. After each exec(), if getpid() != _repl_pid, the wrapper flushes
stdout/stderr and calls os._exit(exit_code): the child writes no sentinel and
keeps its buffered output.

Reliability strategy:
- waitpid() orders the writes of the child and the parent (no time.sleep())
- Each test asserts the exact stdout: a lost or a duplicate line fails it
- All outputs well under 64KB pipe capacity

The parent's output after children that leave the REPL loop: tests/test_fork_output.py.

Integration tests require QEMU + images (run 'make build-images').
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once


class TestForkStdoutCapture:
    """Normal: fork + stdout capture works with the PID guard."""

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert result.stdout == "CHILD_FELL_THROUGH\nPARENT_OK\n"


class TestForkStdoutEdgeCases:
    """Edge: boundary conditions for the PID guard."""

    async def test_fork_child_nonzero_exit_code_preserved(self, scheduler: Scheduler) -> None:
        """Child sys.exit(42) propagates exit code through PID guard; the result is the parent's.

        The REPL wrapper's except SystemExit handler sets exit_code = e.code.
        The PID guard then calls os._exit(exit_code), preserving the code.
        Parent uses waitpid to verify the child's exit status. Only the parent
        writes the sentinel: result.exit_code is 0, not the child's 42.
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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        # Without the PID guard, the child's premature sentinel gives its own exit code (42)
        assert result.exit_code == 0, f"Parent exit_code 0 expected, got {result.exit_code}: {result.stderr}"
        assert result.stdout == "CHILD_EXIT:42\nPARENT_OK\n"

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert result.stdout == "EXIT_NONE:0\nEXIT_NOARG:0\n"

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert result.stdout == "CHILD_EXIT:1\nPARENT_CONTINUES\n"
        assert "ValueError: child_error_marker" in result.stderr, result.stderr


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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert result.stdout == "GRANDCHILD_OK\nCHILD_OK\nPARENT_OK\n"

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert result.stdout == "".join(f"CHILD_LINE_{i}\n" for i in range(200)) + "PARENT_AFTER_CHILD\n"

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

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"Expected exit_code 0, got {result.exit_code}: {result.stderr}"
        assert result.stdout == "CHILD_STDOUT\nPARENT_STDOUT\n"
        assert result.stderr == "CHILD_STDERR\n"


class TestForkStdoutOutOfBounds:
    """Out of bounds: failure modes and limits."""

    async def test_fork_repl_state_survives_child_death(self, scheduler: Scheduler) -> None:
        """The REPL and its namespace survive the fork + PID guard cycle.

        The first exec sets x = 42, forks, and the child calls sys.exit(). The next
        exec of the same session reads x: the parent REPL still runs, with its
        namespace. The child's copy of the namespace goes with the child.
        """
        code = """
import os

x = 42
pid = os.fork()
if pid == 0:
    import sys
    sys.exit(0)

os.waitpid(pid, 0)
"""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as vm:
            forked = await vm.exec(code)
            after = await vm.exec('print(f"x:{x}")')

        assert forked.exit_code == 0, f"Expected exit_code 0, got {forked.exit_code}: {forked.stderr}"
        assert (after.exit_code, after.stdout.strip()) == (0, "x:42"), f"REPL state lost: {after}"

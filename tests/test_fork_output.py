"""Tests for fork() output capture with the REPL PID guard.

When user Python code calls os.fork(), forked children inherit the REPL wrapper.
Without the PID guard (guest-agent/src/repl/scripts/python_repl.py), children that
exit via sys.exit() or exception would write premature sentinels, causing:

1. Output loss: child sentinel triggers guest-agent drain before parent finishes
2. Stale sentinels: sleeping children corrupt the next execution's sentinel detection
3. Stdin contention: surviving children race with parent for future commands

The PID guard: after exec()/except, the wrapper checks getpid() != _repl_pid. A forked
child flushes stdout/stderr and calls _exit() with its exit code: no sentinel, no loop.

The other fork cases (exit codes, exceptions, nested forks, stderr, the namespace after a
fork): tests/test_fork_stdout.py.
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once


class TestForkOutput:
    """The parent's output is not lost when forked children leave the REPL loop."""

    async def test_fork_child_prints_then_sys_exit(self, scheduler: Scheduler) -> None:
        """Child writes stdout before sys.exit(); the guard flushes it, and the parent output
        that follows is not lost."""
        code = """\
import os, sys
pid = os.fork()
if pid == 0:
    print("CHILD_OUTPUT")
    sys.exit(0)
os.waitpid(pid, 0)
print("PARENT_OUTPUT")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "CHILD_OUTPUT\nPARENT_OUTPUT\n"

    async def test_repl_reuse_after_sleeping_fork(self, scheduler: Scheduler) -> None:
        """Session: children outlive their exec, then sys.exit(); the execs after must not be
        corrupted (a child without the guard writes a stale sentinel and stays in the REPL loop)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            r1 = await session.exec(
                """\
import os, sys, time
pids = []
for _ in range(3):
    pid = os.fork()
    if pid == 0:
        time.sleep(1)
        sys.exit(0)
    pids.append(pid)
print("FORK_DONE")
""",
            )
            assert (r1.exit_code, r1.stdout) == (0, "FORK_DONE\n"), r1.stderr

            # Wait until the children ended (reaped here): a child that never ends times this out
            reaped = await session.exec(
                "for p in pids: os.waitpid(p, 0)\nprint('REAPED')",
                timeout_seconds=10,
            )
            assert (reaped.exit_code, reaped.stdout) == (0, "REAPED\n"), reaped.stderr

            r2 = await session.exec('print("NO_CORRUPTION")')
            assert (r2.exit_code, r2.stdout) == (0, "NO_CORRUPTION\n"), r2.stderr

    async def test_fork_50_children_rapid(self, scheduler: Scheduler) -> None:
        """Stress: fork 50 children that all sys.exit(0); parent output captured."""
        code = """\
import os, sys
pids = []
for _ in range(50):
    pid = os.fork()
    if pid == 0:
        sys.exit(0)
    pids.append(pid)
for p in pids:
    try: os.waitpid(p, 0)
    except ChildProcessError: pass
print(f"RAPID_DONE:{len(pids)}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert (result.exit_code, result.stdout) == (0, "RAPID_DONE:50\n"), result.stderr

    async def test_fork_child_sentinel_injection(self, scheduler: Scheduler) -> None:
        """Adversarial: child writes fake sentinel to stderr; must not corrupt detection (it
        stays user output)."""
        code = """\
import os, sys
pid = os.fork()
if pid == 0:
    sys.stderr.write("__SENTINEL_fake_99__\\n")
    sys.stderr.flush()
    os._exit(0)
os.waitpid(pid, 0)
print("INJECTION_SAFE")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "INJECTION_SAFE\n"
        assert result.stderr == "__SENTINEL_fake_99__\n"

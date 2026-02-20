"""Tests for fork() output capture with the REPL PID guard.

When user Python code calls os.fork(), forked children inherit the REPL wrapper.
Without the PID guard (main.rs lines 257-265), children that exit via sys.exit()
or exception would write premature sentinels, causing:

1. Output loss: child sentinel triggers guest-agent drain before parent finishes
2. Stale sentinels: sleeping children corrupt the next execution's sentinel detection
3. Stdin contention: surviving children race with parent for future commands

The PID guard fix: after exec()/except, check getpid() != _repl_pid. If we're a
forked child, flush stdout/stderr and _exit() immediately — no sentinel, no loop.

Test categories:
- Normal: parent stdout captured after fork (the core bug scenario)
- Edge: timing, boundary conditions, REPL reuse after forks
- Weird: nested forks, stderr, non-zero exit codes, file IPC
- Out of bounds: stress (50 children), resource limits, sentinel injection
"""

import asyncio

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


# =============================================================================
# Normal: Parent stdout captured after fork
# =============================================================================
class TestForkOutputNormal:
    """Core bug scenario: parent output must not be lost when children sys.exit()."""

    async def test_fork_sys_exit_parent_output(self, scheduler: Scheduler) -> None:
        """Fork 20 children that sys.exit(0); parent prints after all are reaped."""
        code = """\
import os, sys
pids = []
for _ in range(20):
    pid = os.fork()
    if pid == 0:
        sys.exit(0)
    pids.append(pid)
for p in pids:
    try: os.waitpid(p, 0)
    except ChildProcessError: pass
print(f"PARENT_DONE:{len(pids)}")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "PARENT_DONE:20" in result.stdout

    async def test_fork_os_exit_parent_output(self, scheduler: Scheduler) -> None:
        """Regression baseline: children use os._exit(0), which already worked."""
        code = """\
import os
pids = []
for _ in range(20):
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    pids.append(pid)
for p in pids:
    try: os.waitpid(p, 0)
    except ChildProcessError: pass
print(f"PARENT_DONE:{len(pids)}")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "PARENT_DONE:20" in result.stdout

    async def test_fork_child_work_before_exit(self, scheduler: Scheduler) -> None:
        """Child does computation then sys.exit(0); parent prints summary."""
        code = """\
import os, sys
pid = os.fork()
if pid == 0:
    _ = sum(range(1000))
    sys.exit(0)
os.waitpid(pid, 0)
print("PARENT_COMPUTED")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "PARENT_COMPUTED" in result.stdout


# =============================================================================
# Edge: Timing and boundary conditions
# =============================================================================
class TestForkOutputEdgeCases:
    """Boundary conditions: child stdout, REPL reuse, exceptions in children."""

    async def test_fork_child_prints_then_sys_exit(self, scheduler: Scheduler) -> None:
        """Child writes stdout before sys.exit(); parent output must not be lost."""
        code = """\
import os, sys
pid = os.fork()
if pid == 0:
    print("CHILD_OUTPUT")
    sys.exit(0)
os.waitpid(pid, 0)
print("PARENT_OUTPUT")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "PARENT_OUTPUT" in result.stdout

    async def test_repl_reuse_after_fork(self, scheduler: Scheduler) -> None:
        """Session: first exec forks children, second exec runs normal code."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            r1 = await session.exec(
                """\
import os, sys
for _ in range(5):
    pid = os.fork()
    if pid == 0:
        sys.exit(0)
    os.waitpid(pid, 0)
print("FORK_DONE")
""",
                timeout_seconds=30,
            )
            assert r1.exit_code == 0, f"stderr: {r1.stderr}"
            assert "FORK_DONE" in r1.stdout

            r2 = await session.exec('print("REPL_OK")')
            assert r2.exit_code == 0, f"stderr: {r2.stderr}"
            assert "REPL_OK" in r2.stdout

    async def test_fork_child_raises_exception(self, scheduler: Scheduler) -> None:
        """Child raises RuntimeError; PID guard prevents sentinel after traceback, parent OK."""
        code = """\
import os, sys
pid = os.fork()
if pid == 0:
    raise RuntimeError("child error")
os.waitpid(pid, 0)
print("PARENT_OK")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "PARENT_OK" in result.stdout

    async def test_repl_reuse_after_sleeping_fork(self, scheduler: Scheduler) -> None:
        """Session: children sleep 2s then sys.exit(); next exec must not be corrupted."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            r1 = await session.exec(
                """\
import os, sys, time
for _ in range(3):
    pid = os.fork()
    if pid == 0:
        time.sleep(2)
        sys.exit(0)
print("FORK_DONE")
""",
                timeout_seconds=30,
            )
            assert r1.exit_code == 0, f"stderr: {r1.stderr}"
            assert "FORK_DONE" in r1.stdout

            # Wait for children to wake up and (without fix) write stale sentinels
            await asyncio.sleep(3)

            r2 = await session.exec('print("NO_CORRUPTION")')
            assert r2.exit_code == 0, f"stderr: {r2.stderr}"
            assert "NO_CORRUPTION" in r2.stdout


# =============================================================================
# Weird: Unusual but valid fork patterns
# =============================================================================
class TestForkOutputWeirdCases:
    """Unusual but valid patterns: nested forks, stderr, non-zero exits, file IPC."""

    async def test_nested_fork_grandchild_sys_exit(self, scheduler: Scheduler) -> None:
        """Parent -> child -> grandchild, all using sys.exit(); guard works at every level."""
        code = """\
import os, sys
pid = os.fork()
if pid == 0:
    gpid = os.fork()
    if gpid == 0:
        sys.exit(0)
    os.waitpid(gpid, 0)
    sys.exit(0)
os.waitpid(pid, 0)
print("NESTED_OK")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "NESTED_OK" in result.stdout

    async def test_fork_child_writes_stderr(self, scheduler: Scheduler) -> None:
        """Child writes to stderr then sys.exit(); no sentinel artifacts in stderr."""
        code = """\
import os, sys
pid = os.fork()
if pid == 0:
    sys.stderr.write("CHILD_ERR\\n")
    sys.exit(0)
os.waitpid(pid, 0)
print("STDERR_OK")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "STDERR_OK" in result.stdout
        assert "__SENTINEL_" not in result.stderr

    async def test_fork_with_exit_code_nonzero(self, scheduler: Scheduler) -> None:
        """Child exits with sys.exit(42); parent exit_code is 0 (parent's exit)."""
        code = """\
import os, sys
pid = os.fork()
if pid == 0:
    sys.exit(42)
_, status = os.waitpid(pid, 0)
child_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
print(f"CHILD_EXIT:{child_code}")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "CHILD_EXIT:" in result.stdout

    async def test_fork_file_based_ipc(self, scheduler: Scheduler) -> None:
        """Child writes to file, parent reads; output capture works alongside file IPC."""
        code = """\
import os
pid = os.fork()
if pid == 0:
    with open('/tmp/fork_ipc.txt', 'w') as f:
        f.write('CHILD_WROTE')
    os._exit(0)
os.waitpid(pid, 0)
with open('/tmp/fork_ipc.txt') as f:
    data = f.read()
print(f"IPC:{data}")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "IPC:CHILD_WROTE" in result.stdout


# =============================================================================
# Out of bounds: Stress and adversarial
# =============================================================================
class TestForkOutputOutOfBounds:
    """Stress tests and adversarial patterns."""

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
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=60,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "RAPID_DONE:50" in result.stdout

    async def test_fork_resource_limit_partial(self, scheduler: Scheduler) -> None:
        """Fork until OSError or 500; parent handles gracefully and still outputs."""
        code = """\
import os
pids = []
hit_limit = False
try:
    for _ in range(500):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        pids.append(pid)
except OSError:
    hit_limit = True
finally:
    for p in pids:
        try: os.waitpid(p, 0)
        except ChildProcessError: pass
print(f"FORK_RESULT:forked={len(pids)},limited={hit_limit}")
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=60,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "FORK_RESULT:forked=" in result.stdout

    async def test_fork_child_sentinel_injection(self, scheduler: Scheduler) -> None:
        """Adversarial: child writes fake sentinel to stderr; must not corrupt detection."""
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
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "INJECTION_SAFE" in result.stdout

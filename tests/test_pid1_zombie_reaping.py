"""Tests for PID 1 zombie reaping in guest-agent.

The guest-agent runs as PID 1 inside QEMU microVMs and must reap orphan
processes that would otherwise become zombies.

Key insight: A simple os.fork() does NOT create an orphan - the parent
continues running. To create an orphan, we need a double-fork pattern:

    Main → forks → Child A → forks → Grandchild B
                          ↓
                     A exits immediately
                          ↓
                B is orphaned, reparented to PID 1

Integration tests require QEMU + images (run 'make build-images').
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once


def _orphans_code(count: int, *, grandchild_sleep_s: float) -> str:
    """Python code that makes `count` orphans by double fork, then waits until PID 1 has
    reaped each of them: kill(pid, 0) finds a zombie, and ESRCH once it is reaped. Not /proc:
    it hides these processes from the REPL user (hidepid=invisible, and a non-dumpable process
    belongs to root there), zombies too.

    Each child sends the pid of its grandchild through a pipe, then exits: the grandchild
    is an orphan (reparented to PID 1). The grandchildren hold the write end of the pipe, so
    the read ends once all of them have exited. The code then checks each pid (5 s at most)
    and prints ORPHANS:<count> NOT_REAPED:<count of pids still there>.
    """
    return f"""
import os, time

def exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True

read_fd, write_fd = os.pipe()
for _ in range({count}):
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        grandchild = os.fork()
        if grandchild == 0:
            time.sleep({grandchild_sleep_s})
            os._exit(0)
        os.write(write_fd, f"{{grandchild}}\\n".encode())
        os._exit(0)  # the grandchild becomes an orphan
    os.waitpid(pid, 0)
os.close(write_fd)
with os.fdopen(read_fd) as pipe:
    orphans = [int(line) for line in pipe]  # EOF: every grandchild has exited

deadline = time.monotonic() + 5
left = orphans
while left and time.monotonic() < deadline:
    left = [pid for pid in orphans if exists(pid)]
    if left:
        time.sleep(0.01)
print(f"ORPHANS:{{len(orphans)}} NOT_REAPED:{{len(left)}}")
"""


class TestPid1ZombieReaping:
    """Integration tests for zombie reaping when guest-agent runs as PID 1.

    These tests verify that the guest-agent properly reaps orphan processes
    to prevent zombie accumulation in the VM.
    """

    async def test_orphan_reparented_to_pid1(self, scheduler: Scheduler) -> None:
        """Verify orphan processes are actually reparented to PID 1.

        This test confirms our double-fork pattern creates real orphans.
        Without this verification, we can't be sure we're testing PID 1 reaping.
        """
        code = """
import os, time

# Double-fork: the grandchild sends its parent's pid once its first parent has exited
read_fd, write_fd = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(read_fd)
    child = os.getpid()
    grandchild = os.fork()
    if grandchild == 0:
        deadline = time.monotonic() + 5
        while os.getppid() == child and time.monotonic() < deadline:
            time.sleep(0.001)  # the child exits at once: then the orphan gets a new parent
        os.write(write_fd, f"{os.getppid()}".encode())
        os._exit(0)
    os._exit(0)

os.close(write_fd)
os.waitpid(pid, 0)
with os.fdopen(read_fd) as pipe:
    print(f"ORPHAN_PPID:{pipe.read()}")
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"Execution failed: {result.stderr}"
        # Orphan should be reparented to PID 1 (guest-agent)
        assert result.stdout == "ORPHAN_PPID:1\n", f"Orphan not reparented to PID 1, got: {result.stdout}"

    async def test_multiple_orphans_reaped(self, scheduler: Scheduler) -> None:
        """Ten orphans that are still alive when their parent exits: PID 1 reaps each one."""
        code = _orphans_code(10, grandchild_sleep_s=0.2)

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"Execution failed: {result.stderr}"
        assert result.stdout == "ORPHANS:10 NOT_REAPED:0\n", f"Expected every orphan reaped, got: {result.stdout}"

    async def test_rapid_orphan_spawning(self, scheduler: Scheduler) -> None:
        """Fifty orphans that exit at once (reap_zombies takes SIGCHLD in batches): PID 1 reaps each one."""
        code = _orphans_code(50, grandchild_sleep_s=0)

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)

        assert result.exit_code == 0, f"Execution failed: {result.stderr}"
        assert result.stdout == "ORPHANS:50 NOT_REAPED:0\n", f"Expected every orphan reaped, got: {result.stdout}"

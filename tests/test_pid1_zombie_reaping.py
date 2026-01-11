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
import os
import time

# Double-fork: create orphan and verify its parent becomes PID 1
pid = os.fork()
if pid == 0:
    grandchild = os.fork()
    if grandchild == 0:
        # Grandchild: wait for parent to exit, then check ppid
        time.sleep(0.1)
        ppid = os.getppid()
        # Write to file since stdout is complicated with forks
        with open('/tmp/orphan_ppid.txt', 'w') as f:
            f.write(f'{ppid}')
        os._exit(0)
    else:
        # Parent: exit immediately
        os._exit(0)

# Main: wait for grandchild to write its ppid
time.sleep(0.5)

with open('/tmp/orphan_ppid.txt') as f:
    orphan_ppid = f.read().strip()

print(f'ORPHAN_PPID:{orphan_ppid}')
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"Execution failed: {result.stderr}"
        # Orphan should be reparented to PID 1 (guest-agent)
        assert "ORPHAN_PPID:1" in result.stdout, f"Orphan not reparented to PID 1, got: {result.stdout}"

    async def test_single_orphan_reaped(self, scheduler: Scheduler) -> None:
        """Single orphan process is reaped and doesn't become a zombie."""
        # Double-fork to create orphan: A forks B, A exits, B is orphan
        code = """
import os
import time

# Double-fork to create orphan
pid = os.fork()
if pid == 0:
    # Child A: fork again then exit immediately
    grandchild = os.fork()
    if grandchild == 0:
        # Grandchild B: becomes orphan when A exits, reparented to PID 1
        time.sleep(0.3)
        os._exit(0)
    else:
        # A exits immediately, orphaning B
        os._exit(0)

# Main process: wait for B to exit and be reaped by guest-agent (PID 1)
time.sleep(1)

# Count zombies whose parent is PID 1 (orphans that guest-agent should reap)
# Zombies with other parents are children of processes that haven't called wait()
zombie_count = 0
for entry in os.listdir('/proc'):
    if entry.isdigit():
        try:
            with open(f'/proc/{entry}/status') as f:
                content = f.read()
                # Only count if: State is Z (zombie) AND PPid is 1 (orphan)
                is_zombie = False
                is_orphan = False
                for line in content.splitlines():
                    if line.startswith('State:') and 'Z' in line:
                        is_zombie = True
                    if line.startswith('PPid:') and line.split()[1] == '1':
                        is_orphan = True
                if is_zombie and is_orphan:
                    zombie_count += 1
        except (IOError, FileNotFoundError, PermissionError):
            pass

print(f'ZOMBIE_COUNT:{zombie_count}')
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"Execution failed: {result.stderr}"
        assert "ZOMBIE_COUNT:0" in result.stdout, f"Expected no zombies, got: {result.stdout}"

    async def test_multiple_orphans_reaped(self, scheduler: Scheduler) -> None:
        """Multiple orphan processes are all reaped."""
        code = """
import os
import time

def create_orphan():
    '''Double-fork: creates orphan that will be reparented to PID 1.'''
    grandchild = os.fork()
    if grandchild == 0:
        # Grandchild: becomes orphan when parent exits
        time.sleep(0.2)
        os._exit(0)
    else:
        # Parent: exit immediately, orphaning grandchild
        os._exit(0)

# Create 10 orphans via double-fork
for _ in range(10):
    pid = os.fork()
    if pid == 0:
        create_orphan()
        # Never reached - create_orphan always exits
        os._exit(0)

# Wait for all orphans to exit and be reaped
time.sleep(2)

# Count zombies whose parent is PID 1 (orphans that guest-agent should reap)
zombie_count = 0
for entry in os.listdir('/proc'):
    if entry.isdigit():
        try:
            with open(f'/proc/{entry}/status') as f:
                content = f.read()
                is_zombie = False
                is_orphan = False
                for line in content.splitlines():
                    if line.startswith('State:') and 'Z' in line:
                        is_zombie = True
                    if line.startswith('PPid:') and line.split()[1] == '1':
                        is_orphan = True
                if is_zombie and is_orphan:
                    zombie_count += 1
        except (IOError, FileNotFoundError, PermissionError):
            pass

print(f'ZOMBIE_COUNT:{zombie_count}')
print('ORPHANS_CREATED:10')
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"Execution failed: {result.stderr}"
        assert "ZOMBIE_COUNT:0" in result.stdout, f"Expected no zombies after 10 orphans, got: {result.stdout}"

    async def test_rapid_orphan_spawning(self, scheduler: Scheduler) -> None:
        """Rapid spawning of orphans doesn't overwhelm the reaper."""
        code = """
import os
import time

# Rapidly spawn 50 orphans via double-fork
for i in range(50):
    pid = os.fork()
    if pid == 0:
        # Child: fork grandchild then exit
        grandchild = os.fork()
        if grandchild == 0:
            # Grandchild: minimal work then exit
            _ = i * 2
            os._exit(0)
        # Child exits immediately, orphaning grandchild
        os._exit(0)

# Allow time for all orphans to exit and be reaped
# reap_zombies() uses SIGCHLD batching, so this tests that path
time.sleep(3)

# Count zombies whose parent is PID 1 (orphans that guest-agent should reap)
zombie_count = 0
for entry in os.listdir('/proc'):
    if entry.isdigit():
        try:
            with open(f'/proc/{entry}/status') as f:
                content = f.read()
                is_zombie = False
                is_orphan = False
                for line in content.splitlines():
                    if line.startswith('State:') and 'Z' in line:
                        is_zombie = True
                    if line.startswith('PPid:') and line.split()[1] == '1':
                        is_orphan = True
                if is_zombie and is_orphan:
                    zombie_count += 1
        except (IOError, FileNotFoundError, PermissionError):
            pass

print(f'ZOMBIE_COUNT:{zombie_count}')
print('RAPID_SPAWNS:50')
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=60,
        )

        assert result.exit_code == 0, f"Execution failed: {result.stderr}"
        assert "ZOMBIE_COUNT:0" in result.stdout, f"Expected no zombies after rapid spawning, got: {result.stdout}"


class TestZombieReapingShell:
    """Shell-based tests for zombie reaping."""

    async def test_shell_orphan_reaped(self, scheduler: Scheduler) -> None:
        """Shell-created orphan processes are reaped by PID 1."""
        # Shell script that creates actual orphans via double-fork pattern
        # Simple background jobs (&) are NOT orphans - they're children of the shell
        code = """
#!/bin/sh
# Double-fork in shell to create orphan:
# sh -c '...' runs a subshell that forks, parent exits, child is orphan

# Create 3 orphans
for i in 1 2 3; do
    sh -c 'sh -c "sleep 0.2" &' &
done

# Wait for orphans to exit and be reaped
sleep 1

# Count zombies whose parent is PID 1 (true orphans)
zombie_count=0
for pid_dir in /proc/[0-9]*; do
    if [ -f "$pid_dir/status" ]; then
        # Check if zombie AND parent is PID 1
        if grep -q "^State:.*Z" "$pid_dir/status" 2>/dev/null; then
            ppid=$(grep "^PPid:" "$pid_dir/status" 2>/dev/null | awk '{print $2}')
            if [ "$ppid" = "1" ]; then
                zombie_count=$((zombie_count + 1))
            fi
        fi
    fi
done

echo "ZOMBIE_COUNT:$zombie_count"
"""

        result = await scheduler.run(
            code=code,
            language=Language.RAW,
            timeout_seconds=30,
        )

        assert result.exit_code == 0, f"Execution failed: {result.stderr}"
        assert "ZOMBIE_COUNT:0" in result.stdout, f"Expected no zombies from shell orphans, got: {result.stdout}"

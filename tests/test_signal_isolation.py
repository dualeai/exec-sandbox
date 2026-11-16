"""Tests for signal isolation between UID 1000 REPL and PID 1 guest-agent.

Guest-agent runs as root (PID 1). The REPL subprocess runs as UID 1000.
The Linux kernel enforces that unprivileged processes cannot signal processes
owned by a different UID. This is the primary signal boundary.

Test categories:
- Normal: signals work correctly within the UID 1000 boundary (self + children)
- Edge: every signal type to PID 1 but SIGCONT must raise PermissionError (EPERM)
- Weird: bypass attempts via alternative syscalls (tgkill, pidfd, rt_sigqueueinfo)
- Out of bounds: the /proc/1 files and capability escalation (CVE-informed)

Writes to /proc/sysrq-trigger and /proc/sys/kernel/core_pattern: test_device_security.py.
ptrace of PID 1: test_kernel_attack_surface.py::TestPtraceRestricted.

References:
- kill(2): https://man7.org/linux/man-pages/man2/kill.2.html
- pidfd_send_signal(2): https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html
- tgkill(2): https://man7.org/linux/man-pages/man2/tkill.2.html
- CVE-2024-21626: runc /proc/self/fd container escape
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once


# =============================================================================
# Normal: Signals work correctly within the UID 1000 boundary
# =============================================================================
class TestSignalSelfAndChildren:
    """REPL can signal itself and its own children (same UID)."""

    async def test_self_signal_sigusr1(self, scheduler: Scheduler) -> None:
        """REPL sends SIGUSR1 to itself, handler fires."""
        code = """\
import os, signal

received = []

def handler(signum, frame):
    received.append(signum)

signal.signal(signal.SIGUSR1, handler)
os.kill(os.getpid(), signal.SIGUSR1)
print(f"RECEIVED:{len(received)}")
print(f"SIGNUM:{received[0]}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        # SIGUSR1 = 10 on Linux (guest), regardless of host platform
        assert (result.exit_code, result.stdout) == (0, "RECEIVED:1\nSIGNUM:10\n"), result.stderr

    async def test_self_signal_sigterm_with_handler(self, scheduler: Scheduler) -> None:
        """REPL sends SIGTERM to itself, handler catches it."""
        code = """\
import os, signal

caught = False

def handler(signum, frame):
    global caught
    caught = True

signal.signal(signal.SIGTERM, handler)
os.kill(os.getpid(), signal.SIGTERM)
print(f"CAUGHT:{caught}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "CAUGHT:True\n"), result.stderr

    async def test_signal_own_child(self, scheduler: Scheduler) -> None:
        """REPL forks a child, sends SIGKILL, waitpid confirms kill."""
        code = """\
import os, signal, time

pid = os.fork()
if pid == 0:
    # Child: sleep forever
    while True:
        time.sleep(1)
else:
    # Parent: kill child (SIGKILL ends it at any point after fork)
    os.kill(pid, signal.SIGKILL)
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        print(f"CHILD_KILLED:signal={os.WTERMSIG(status)}")
    else:
        print(f"CHILD_EXIT:{os.WEXITSTATUS(status)}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "CHILD_KILLED:signal=9\n"), result.stderr

    async def test_process_group_isolation(self, scheduler: Scheduler) -> None:
        """REPL PGID differs from PID 1 PGID (read with getpgid(1)), so kill(0, sig) cannot reach PID 1."""
        code = """\
import os
print(f"PGID_DIFFERS:{os.getpgrp() != os.getpgid(1)}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "PGID_DIFFERS:True\n"

    async def test_signal_0_self_check(self, scheduler: Scheduler) -> None:
        """Signal 0 to own PID succeeds (existence check)."""
        code = """\
import os
try:
    os.kill(os.getpid(), 0)
    print("EXISTS")
except ProcessLookupError:
    print("NOT_FOUND")
except PermissionError:
    print("NO_PERM")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "EXISTS\n"), result.stderr


# =============================================================================
# Edge: Every signal type to PID 1 must raise PermissionError
# =============================================================================
class TestSignalPid1Blocked:
    """All signals to PID 1 (root-owned guest-agent) are blocked by EPERM."""

    async def test_signal_0_pid1_blocked(self, scheduler: Scheduler) -> None:
        """Signal 0 (existence check) to PID 1 raises PermissionError."""
        code = """\
import os
try:
    os.kill(1, 0)
    print("unexpected_success")
except PermissionError:
    print("blocked")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "blocked\n"), result.stderr

    async def test_all_standard_signals_pid1_blocked(self, scheduler: Scheduler) -> None:
        """All standard signals (1-31) to PID 1 raise PermissionError (SIGKILL, SIGSTOP,
        SIGINT, SIGTERM and SIGUSR1 among them), never another error; then the guest agent
        still runs the next exec of the session.

        Exception: SIGCONT (18) is allowed across UID boundaries within the same
        session by the Linux kernel (it's the only signal with this property).
        This is harmless -- SIGCONT only resumes a stopped process.
        """
        code = """\
import os
blocked = 0
allowed = []
other = []
for sig in range(1, 32):
    try:
        os.kill(1, sig)
        allowed.append(sig)
    except PermissionError:
        blocked += 1
    except OSError as e:
        other.append((sig, e.errno))
print(f"BLOCKED:{blocked}")
print(f"ALLOWED:{allowed}")
print(f"OTHER:{other}")
"""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec(code)
            assert result.exit_code == 0
            # 30 blocked (all except SIGCONT), each with EPERM
            assert result.stdout == "BLOCKED:30\nALLOWED:[18]\nOTHER:[]\n"

            # The guest agent (PID 1) still serves the session
            alive = await session.exec('print("STILL_ALIVE")')
            assert alive.exit_code == 0
            assert alive.stdout == "STILL_ALIVE\n"

    async def test_realtime_signals_pid1_blocked(self, scheduler: Scheduler) -> None:
        """RT signals (SIGRTMIN..SIGRTMIN+4) to PID 1 raise PermissionError, never another error."""
        code = """\
import os, signal
rt_min = signal.SIGRTMIN
blocked = 0
unexpected = []
other = []
for i in range(5):
    sig = rt_min + i
    try:
        os.kill(1, sig)
        unexpected.append(sig)
    except PermissionError:
        blocked += 1
    except OSError as e:
        other.append((sig, e.errno))
print(f"RT_BLOCKED:{blocked} RT_UNEXPECTED:{unexpected} RT_OTHER:{other}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "RT_BLOCKED:5 RT_UNEXPECTED:[] RT_OTHER:[]\n"

    async def test_shell_kill_pid1_blocked(self, scheduler: Scheduler) -> None:
        """`kill -9 1` from bash fails with permission error."""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "kill -9 1 2>&1; echo exit=$?")
        assert result.exit_code == 0
        assert "kill: (1) - Operation not permitted\n" in result.stdout
        assert result.stdout.endswith("exit=1\n")


# =============================================================================
# Weird: Bypass attempts via alternative syscalls or targeting strategies
# =============================================================================
class TestSignalBypassAttempts:
    """Creative attempts to bypass the signal boundary via alternative syscalls."""

    async def test_kill_minus1_broadcast(self, scheduler: Scheduler) -> None:
        """kill(-1, SIGKILL) -- Linux skips PID 1; guest-agent survives.

        Linux kernel excludes both PID 1 (pid > 1 check) and the calling process
        (same_thread_group check) from kill(-1). os.killpg(1, sig) is the same kill(-1, sig) call.
        The call result depends on which other UID 1000 processes exist, so the test does not
        check it: it checks that the caller and PID 1 survive.
        """
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # kill(-1, SIGKILL) excludes PID 1 and the calling process itself
            r1 = await session.exec("""\
import os, signal
try:
    os.kill(-1, signal.SIGKILL)
except ProcessLookupError:
    pass  # no other process to signal
print("CALLER_ALIVE")
""")
            assert r1.exit_code == 0
            assert r1.stdout == "CALLER_ALIVE\n"

            # Guest-agent (PID 1) survived -- verify by running a follow-up command
            r2 = await session.exec('print("AGENT_ALIVE")')
            assert (r2.exit_code, r2.stdout) == (0, "AGENT_ALIVE\n"), r2.stderr

    async def test_kill_0_own_process_group(self, scheduler: Scheduler) -> None:
        """kill(0, SIGTERM) only affects REPL's group, not PID 1."""
        code = """\
import os, signal

# Install handler so we don't die from our own signal
caught = False
def handler(signum, frame):
    global caught
    caught = True

signal.signal(signal.SIGTERM, handler)
# kill(0, sig) sends to all processes in the caller's process group
os.kill(0, signal.SIGTERM)
print(f"CAUGHT_OWN:{caught}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "CAUGHT_OWN:True\n"), result.stderr

    async def test_signal_pid2_kthreadd_blocked(self, scheduler: Scheduler) -> None:
        """Signal to PID 2 (kthreadd, a root kernel thread) raises PermissionError."""
        code = """\
import os, signal
try:
    os.kill(2, signal.SIGTERM)
    print("unexpected_success")
except PermissionError:
    print("blocked:EPERM")
except ProcessLookupError:
    print("blocked:ESRCH")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "blocked:EPERM\n"

    async def test_signal_nonexistent_pid(self, scheduler: Scheduler) -> None:
        """Signal to nonexistent PID 99999 raises ProcessLookupError (ESRCH)."""
        code = """\
import os, signal
try:
    os.kill(99999, signal.SIGTERM)
    print("unexpected_success")
except ProcessLookupError:
    print("ESRCH")
except PermissionError:
    print("EPERM")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "ESRCH\n"), result.stderr

    async def test_ctypes_kill_pid1_blocked(self, scheduler: Scheduler) -> None:
        """Direct libc kill(1, 9) via ctypes returns -1 with errno=EPERM."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
ret = libc.kill(1, 9)
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "ret=-1 errno=1\n"  # EPERM = 1

    async def test_ctypes_tgkill_pid1_blocked(self, scheduler: Scheduler) -> None:
        """Direct tgkill(1, 1, 9) via ctypes syscall -- thread-level signal, same EPERM."""
        code = """\
import ctypes, platform

libc = ctypes.CDLL("libc.so.6", use_errno=True)
# tgkill syscall number (the guest runs one of these two; another arch is a KeyError)
NR_TGKILL = {"x86_64": 234, "aarch64": 131}[platform.machine()]

ret = libc.syscall(NR_TGKILL, 1, 1, 9)
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "ret=-1 errno=1\n"  # EPERM = 1

    async def test_ctypes_rt_sigqueueinfo_pid1_blocked(self, scheduler: Scheduler) -> None:
        """rt_sigqueueinfo(1, SIGTERM, &info) via ctypes -- signal+data path, EPERM."""
        code = """\
import ctypes, os, platform, struct

libc = ctypes.CDLL("libc.so.6", use_errno=True)
# rt_sigqueueinfo syscall number (the guest runs one of these two; another arch is a KeyError)
NR_RT_SIGQUEUEINFO = {"x86_64": 129, "aarch64": 138}[platform.machine()]

# siginfo_t structure: si_signo(int), si_errno(int), si_code(int), ...
# SI_QUEUE = -1, must set si_pid to our PID
# Minimum 128 bytes for siginfo_t
si = bytearray(128)
# si_signo = SIGTERM (15)
struct.pack_into("i", si, 0, 15)
# si_errno = 0
struct.pack_into("i", si, 4, 0)
# si_code = SI_QUEUE (-1)
struct.pack_into("i", si, 8, -1)
# si_pid at offset 12
struct.pack_into("i", si, 12, os.getpid())

si_buf = (ctypes.c_char * len(si)).from_buffer(si)
ret = libc.syscall(NR_RT_SIGQUEUEINFO, 1, 15, ctypes.byref(si_buf))
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "ret=-1 errno=1\n"  # EPERM = 1

    async def test_ctypes_pidfd_send_signal_pid1_blocked(self, scheduler: Scheduler) -> None:
        """pidfd_open(1) + pidfd_send_signal(fd, 9) -- fd-based signal path, EPERM.

        pidfd_open has no permission check, so the REPL gets a pidfd for PID 1: the send is the
        step the kernel refuses.
        """
        code = """\
import ctypes, os, platform

libc = ctypes.CDLL("libc.so.6", use_errno=True)
# pidfd_open and pidfd_send_signal have the same syscall numbers on x86_64 and aarch64
assert platform.machine() in ("x86_64", "aarch64"), platform.machine()
NR_PIDFD_OPEN = 434
NR_PIDFD_SEND_SIGNAL = 424

# Step 1: pidfd_open(1, 0)
pidfd = libc.syscall(NR_PIDFD_OPEN, 1, 0)
print(f"OPEN:errno={ctypes.get_errno() if pidfd < 0 else 0}")
# Step 2: pidfd_send_signal(pidfd, SIGKILL, NULL, 0)
ret = libc.syscall(NR_PIDFD_SEND_SIGNAL, pidfd, 9, 0, 0)
print(f"SEND:ret={ret} errno={ctypes.get_errno()}")
os.close(pidfd)
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "OPEN:errno=0\nSEND:ret=-1 errno=1\n"  # EPERM = 1

    async def test_javascript_kill_pid1_blocked(self, scheduler: Scheduler) -> None:
        """process.kill(1, 'SIGKILL') throws EPERM from JS runtime."""
        code = """\
try {
    process.kill(1, 'SIGKILL');
    console.log('unexpected_success');
} catch (e) {
    if (e.code === 'EPERM') {
        console.log('blocked');
    } else {
        console.log('error:' + e.code + ':' + e.message);
    }
}
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)
        assert (result.exit_code, result.stdout) == (0, "blocked\n"), result.stderr


# =============================================================================
# Out of bounds: /proc-based attacks, ptrace, capability escalation
# =============================================================================
class TestSignalOutOfBounds:
    """Non-signal vectors to influence or exfiltrate from PID 1, informed by recent CVEs."""

    async def test_proc_pid1_files_blocked(self, scheduler: Scheduler) -> None:
        """No /proc/1 file is reachable: ENOENT (hidepid hides PID 1) for each vector. One VM
        tries each:
        - listing /proc/1/fd (fd leak: CVE-2024-21626, CVE-2025-31133, runc container escapes);
        - reading /proc/1/environ (secret leak) and /proc/1/maps (ASLR leak: CVE-2023-3269,
          StackRot);
        - opening /proc/1/mem read-write (code injection: https://lwn.net/Articles/476684/);
        - writing /proc/1/oom_score_adj (OOM targeting).
        """
        code = """\
import errno, os
attempts = {
    'listdir /proc/1/fd': lambda: os.listdir('/proc/1/fd'),
    'read /proc/1/environ': lambda: open('/proc/1/environ', 'rb').read(),
    'read /proc/1/maps': lambda: open('/proc/1/maps').read(),
    'open-rdwr /proc/1/mem': lambda: os.close(os.open('/proc/1/mem', os.O_RDWR)),
    'write /proc/1/oom_score_adj': lambda: open('/proc/1/oom_score_adj', 'w').write('1000'),
}
for label, attempt in attempts.items():
    try:
        attempt()
        print(f'{label}:DONE')
    except OSError as e:
        print(f'{label}:{errno.errorcode[e.errno]}')
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.splitlines() == [
            "listdir /proc/1/fd:ENOENT",
            "read /proc/1/environ:ENOENT",
            "read /proc/1/maps:ENOENT",
            "open-rdwr /proc/1/mem:ENOENT",
            "write /proc/1/oom_score_adj:ENOENT",
        ]

    async def test_prctl_cap_kill_blocked(self, scheduler: Scheduler) -> None:
        """prctl(PR_CAP_AMBIENT, RAISE, CAP_KILL) fails -- can't grant self CAP_KILL."""
        code = """\
import ctypes

libc = ctypes.CDLL("libc.so.6", use_errno=True)
# PR_CAP_AMBIENT = 47, PR_CAP_AMBIENT_RAISE = 2, CAP_KILL = 5
ret = libc.prctl(47, 2, 5, 0, 0)
err = ctypes.get_errno()
if ret == -1:
    print(f"blocked:errno={err}")
else:
    print(f"unexpected_success:ret={ret}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "blocked:errno=1\n"  # EPERM = 1

"""Tests for signal isolation between UID 1000 REPL and PID 1 guest-agent.

Guest-agent runs as root (PID 1). The REPL subprocess runs as UID 1000.
The Linux kernel enforces that unprivileged processes cannot signal processes
owned by a different UID. This is the primary signal boundary.

Test categories:
- Normal: signals work correctly within the UID 1000 boundary (self + children)
- Edge: every signal type to PID 1 must raise PermissionError (EPERM)
- Weird: bypass attempts via alternative syscalls (tgkill, pidfd, rt_sigqueueinfo)
- Out of bounds: /proc-based attacks, ptrace, capability escalation (CVE-informed)

References:
- kill(2): https://man7.org/linux/man-pages/man2/kill.2.html
- pidfd_send_signal(2): https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html
- tgkill(2): https://man7.org/linux/man-pages/man2/tkill.2.html
- CVE-2024-21626: runc /proc/self/fd container escape
- CVE-2025-31133/52565/52881: runc /proc/sys/kernel/core_pattern escapes
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "RECEIVED:1" in result.stdout
        # SIGUSR1 = 10 on Linux (guest), regardless of host platform
        assert "SIGNUM:10" in result.stdout

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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "CAUGHT:True" in result.stdout

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
    # Parent: kill child
    time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        print(f"CHILD_KILLED:signal={os.WTERMSIG(status)}")
    else:
        print(f"CHILD_EXIT:{os.WEXITSTATUS(status)}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "CHILD_KILLED:signal=9" in result.stdout

    async def test_process_group_isolation(self, scheduler: Scheduler) -> None:
        """REPL PGID differs from PID 1 PGID."""
        code = """\
import os
my_pgid = os.getpgrp()
print(f"MY_PGID:{my_pgid}")
print(f"MY_PID:{os.getpid()}")
print(f"PGID_NOT_1:{my_pgid != 1}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "PGID_NOT_1:True" in result.stdout

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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXISTS" in result.stdout


# =============================================================================
# Edge: Every signal type to PID 1 must raise PermissionError
# =============================================================================
class TestSignalPid1Blocked:
    """All signals to PID 1 (root-owned guest-agent) are blocked by EPERM."""

    async def test_sigkill_pid1_blocked(self, scheduler: Scheduler) -> None:
        """SIGKILL to PID 1 raises PermissionError (primary gap coverage)."""
        code = """\
import os, signal
try:
    os.kill(1, signal.SIGKILL)
    print("unexpected_success")
except PermissionError:
    print("blocked")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_sigstop_pid1_blocked(self, scheduler: Scheduler) -> None:
        """SIGSTOP (uncatchable) to PID 1 raises PermissionError."""
        code = """\
import os, signal
try:
    os.kill(1, signal.SIGSTOP)
    print("unexpected_success")
except PermissionError:
    print("blocked")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_sigint_pid1_blocked(self, scheduler: Scheduler) -> None:
        """SIGINT to PID 1 raises PermissionError."""
        code = """\
import os, signal
try:
    os.kill(1, signal.SIGINT)
    print("unexpected_success")
except PermissionError:
    print("blocked")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_sigusr1_pid1_blocked(self, scheduler: Scheduler) -> None:
        """SIGUSR1 to PID 1 raises PermissionError."""
        code = """\
import os, signal
try:
    os.kill(1, signal.SIGUSR1)
    print("unexpected_success")
except PermissionError:
    print("blocked")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_all_standard_signals_pid1_blocked(self, scheduler: Scheduler) -> None:
        """All standard signals (1-31) to PID 1 raise PermissionError.

        Exception: SIGCONT (18) is allowed across UID boundaries within the same
        session by the Linux kernel (it's the only signal with this property).
        This is harmless -- SIGCONT only resumes a stopped process.
        """
        code = """\
import os, signal
blocked = 0
allowed = []
# SIGCONT (18) is allowed cross-UID within the same session -- kernel design
EXPECTED_ALLOWED = {signal.SIGCONT}
for sig in range(1, 32):
    try:
        os.kill(1, sig)
        allowed.append(sig)
    except PermissionError:
        blocked += 1
    except OSError:
        # EINVAL for invalid signal numbers on this arch -- still not a success
        blocked += 1
unexpected = [s for s in allowed if s not in EXPECTED_ALLOWED]
print(f"BLOCKED:{blocked}")
print(f"ALLOWED:{sorted(allowed)}")
if unexpected:
    print(f"UNEXPECTED:{unexpected}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # 30 blocked (all except SIGCONT)
        assert "BLOCKED:30" in result.stdout
        assert "UNEXPECTED:" not in result.stdout

    async def test_realtime_signals_pid1_blocked(self, scheduler: Scheduler) -> None:
        """RT signals (SIGRTMIN..SIGRTMIN+4) to PID 1 raise PermissionError."""
        code = """\
import os, signal
rt_min = signal.SIGRTMIN
blocked = 0
unexpected = []
for i in range(5):
    sig = rt_min + i
    try:
        os.kill(1, sig)
        unexpected.append(sig)
    except PermissionError:
        blocked += 1
    except OSError:
        blocked += 1
print(f"RT_BLOCKED:{blocked}")
if unexpected:
    print(f"RT_UNEXPECTED:{unexpected}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "RT_BLOCKED:5" in result.stdout
        assert "RT_UNEXPECTED:" not in result.stdout

    async def test_shell_kill_pid1_blocked(self, scheduler: Scheduler) -> None:
        """`kill -9 1` from bash fails with permission error."""
        result = await scheduler.run(
            code="kill -9 1 2>&1; echo exit=$?",
            language=Language.RAW,
        )
        assert result.exit_code == 0
        assert "exit=1" in result.stdout or "ermission" in result.stdout or "not permitted" in result.stdout


# =============================================================================
# Weird: Bypass attempts via alternative syscalls or targeting strategies
# =============================================================================
class TestSignalBypassAttempts:
    """Creative attempts to bypass the signal boundary via alternative syscalls."""

    async def test_kill_minus1_broadcast(self, scheduler: Scheduler) -> None:
        """kill(-1, SIGKILL) -- POSIX excludes PID 1; guest-agent survives.

        Linux kernel excludes both PID 1 (pid > 1 check) and the calling process
        (same_thread_group check) from kill(-1). So on a minimal VM, this is
        essentially a no-op. The key assertion is that PID 1 survives.
        """
        async with await scheduler.session(language=Language.PYTHON) as session:
            # kill(-1, SIGKILL) excludes PID 1 and the calling process itself
            r1 = await session.exec("""\
import os, signal
try:
    os.kill(-1, signal.SIGKILL)
    print("CALL_OK")
except (PermissionError, ProcessLookupError, OSError) as e:
    print(f"CALL_ERR:{type(e).__name__}")
""")
            assert r1.exit_code == 0
            assert "CALL_OK" in r1.stdout or "CALL_ERR:" in r1.stdout

            # Guest-agent (PID 1) survived -- verify by running a follow-up command
            r2 = await session.exec('print("AGENT_ALIVE")')
            assert r2.exit_code == 0
            assert "AGENT_ALIVE" in r2.stdout

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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "CAUGHT_OWN:True" in result.stdout

    async def test_signal_pid2_kthreadd_blocked(self, scheduler: Scheduler) -> None:
        """Signal to PID 2 (kthreadd kernel thread) raises PermissionError or ESRCH."""
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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked:" in result.stdout

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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "ESRCH" in result.stdout

    async def test_ctypes_kill_pid1_blocked(self, scheduler: Scheduler) -> None:
        """Direct libc kill(1, 9) via ctypes returns -1 with errno=EPERM."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
ret = libc.kill(1, 9)
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "ret=-1" in result.stdout
        # EPERM = 1
        assert "errno=1" in result.stdout

    async def test_ctypes_tgkill_pid1_blocked(self, scheduler: Scheduler) -> None:
        """Direct tgkill(1, 1, 9) via ctypes syscall -- thread-level signal, same EPERM."""
        code = """\
import ctypes, platform

libc = ctypes.CDLL("libc.so.6", use_errno=True)
# tgkill syscall number: x86_64=234, aarch64=131
arch = platform.machine()
if arch == "x86_64":
    NR_TGKILL = 234
elif arch == "aarch64":
    NR_TGKILL = 131
else:
    print(f"SKIP:unknown_arch={arch}")
    raise SystemExit(0)

ret = libc.syscall(NR_TGKILL, 1, 1, 9)
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        if "SKIP:" in result.stdout:
            return
        assert "ret=-1" in result.stdout
        # EPERM = 1
        assert "errno=1" in result.stdout

    async def test_ctypes_rt_sigqueueinfo_pid1_blocked(self, scheduler: Scheduler) -> None:
        """rt_sigqueueinfo(1, SIGTERM, &info) via ctypes -- signal+data path, EPERM."""
        code = """\
import ctypes, os, platform, struct

libc = ctypes.CDLL("libc.so.6", use_errno=True)
# rt_sigqueueinfo syscall: x86_64=129, aarch64=138
arch = platform.machine()
if arch == "x86_64":
    NR_RT_SIGQUEUEINFO = 129
elif arch == "aarch64":
    NR_RT_SIGQUEUEINFO = 138
else:
    print(f"SKIP:unknown_arch={arch}")
    raise SystemExit(0)

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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        if "SKIP:" in result.stdout:
            return
        assert "ret=-1" in result.stdout
        # EPERM = 1
        assert "errno=1" in result.stdout

    async def test_ctypes_pidfd_send_signal_pid1_blocked(self, scheduler: Scheduler) -> None:
        """pidfd_open(1) + pidfd_send_signal(fd, 9) -- fd-based signal path, EPERM."""
        code = """\
import ctypes, os, platform

libc = ctypes.CDLL("libc.so.6", use_errno=True)
arch = platform.machine()
# pidfd_open and pidfd_send_signal have the same syscall numbers on x86_64 and aarch64
if arch not in ("x86_64", "aarch64"):
    print(f"SKIP:unknown_arch={arch}")
    raise SystemExit(0)
NR_PIDFD_OPEN = 434
NR_PIDFD_SEND_SIGNAL = 424

# Step 1: pidfd_open(1, 0) -- may fail with EPERM itself
pidfd = libc.syscall(NR_PIDFD_OPEN, 1, 0)
err_open = ctypes.get_errno()

if pidfd < 0:
    # EPERM on pidfd_open is acceptable (kernel may block cross-UID pidfd)
    print(f"PIDFD_OPEN_FAILED:errno={err_open}")
    print("blocked")
else:
    # Step 2: pidfd_send_signal(pidfd, SIGKILL, NULL, 0)
    ret = libc.syscall(NR_PIDFD_SEND_SIGNAL, pidfd, 9, 0, 0)
    err_send = ctypes.get_errno()
    os.close(pidfd)
    print(f"SEND_RESULT:ret={ret} errno={err_send}")
    if ret == -1 and err_send == 1:  # EPERM
        print("blocked")
    else:
        print("unexpected_success")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_killpg_pid1_group_blocked(self, scheduler: Scheduler) -> None:
        """os.killpg(1, sig) maps to kill(-1, sig) which is broadcast, not PGID 1.

        kill(-1, sig) sends to all processes the caller can signal EXCEPT PID 1
        (POSIX requirement). Returns 0 if at least one process was signaled.
        This is safe: PID 1 is explicitly excluded by the kernel.
        """
        code = """\
import os, signal

# Install handler so we don't die from our own signal
caught = False
def handler(signum, frame):
    global caught
    caught = True

signal.signal(signal.SIGUSR1, handler)

# killpg(1, sig) == kill(-1, sig) == broadcast (POSIX "all signalable except PID 1")
# This succeeds because it signals our own process, but PID 1 is excluded
try:
    os.killpg(1, signal.SIGUSR1)
    print(f"BROADCAST_OK:caught_self={caught}")
except PermissionError:
    print("blocked:EPERM")
except ProcessLookupError:
    print("blocked:ESRCH")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # Either broadcast succeeded (safe, PID 1 excluded) or blocked
        assert "BROADCAST_OK:" in result.stdout or "blocked:" in result.stdout

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
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0
        assert "blocked" in result.stdout


# =============================================================================
# Out of bounds: /proc-based attacks, ptrace, capability escalation
# =============================================================================
class TestSignalOutOfBounds:
    """Non-signal vectors to influence or exfiltrate from PID 1, informed by recent CVEs."""

    async def test_proc_pid1_fd_dir_blocked(self, scheduler: Scheduler) -> None:
        """/proc/1/fd/ not listable (fd leak vector).

        References: CVE-2024-21626, CVE-2025-31133 (runc container escapes).
        """
        code = """\
import os
try:
    entries = os.listdir('/proc/1/fd')
    print(f"unexpected_success:{len(entries)} fds")
except PermissionError:
    print("blocked")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_proc_pid1_environ_read_blocked(self, scheduler: Scheduler) -> None:
        """/proc/1/environ not readable (secret leak vector)."""
        code = """\
try:
    with open('/proc/1/environ', 'rb') as f:
        data = f.read()
    print(f"unexpected_success:{len(data)} bytes")
except PermissionError:
    print("blocked")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_proc_pid1_mem_write_blocked(self, scheduler: Scheduler) -> None:
        """/proc/1/mem not writable (code injection vector).

        Reference: https://lwn.net/Articles/476684/
        """
        code = """\
import os
try:
    fd = os.open('/proc/1/mem', os.O_RDWR)
    os.close(fd)
    print("unexpected_success")
except PermissionError:
    print("blocked:EPERM")
except FileNotFoundError:
    print("blocked:ENOENT")
except OSError as e:
    print(f"blocked:errno={e.errno}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked:" in result.stdout

    async def test_proc_pid1_maps_read_blocked(self, scheduler: Scheduler) -> None:
        """/proc/1/maps not readable (ASLR leak vector).

        Reference: CVE-2023-3269 (StackRot).
        """
        code = """\
try:
    with open('/proc/1/maps') as f:
        data = f.read()
    print(f"unexpected_success:{len(data)} bytes")
except PermissionError:
    print("blocked")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_proc_pid1_oom_score_adj_blocked(self, scheduler: Scheduler) -> None:
        """/proc/1/oom_score_adj not writable (OOM targeting vector)."""
        code = """\
try:
    with open('/proc/1/oom_score_adj', 'w') as f:
        f.write('1000')
    print("unexpected_success")
except PermissionError:
    print("blocked")
except OSError as e:
    print(f"blocked:errno={e.errno}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_proc_sysrq_trigger_blocked(self, scheduler: Scheduler) -> None:
        """/proc/sysrq-trigger not writable (write 'c' = kernel crash, 'b' = reboot).

        Reference: CVE-2025-52881.
        """
        code = """\
try:
    with open('/proc/sysrq-trigger', 'w') as f:
        f.write('h')  # 'h' = help (safest), but should fail before write
    print("unexpected_success")
except PermissionError:
    print("blocked")
except FileNotFoundError:
    print("blocked:not_found")
except OSError as e:
    print(f"blocked:errno={e.errno}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_proc_core_pattern_blocked(self, scheduler: Scheduler) -> None:
        """/proc/sys/kernel/core_pattern not writable (arbitrary code execution vector).

        Reference: CVE-2025-52881.
        """
        code = """\
try:
    with open('/proc/sys/kernel/core_pattern', 'w') as f:
        f.write('|/tmp/evil %p')
    print("unexpected_success")
except PermissionError:
    print("blocked")
except FileNotFoundError:
    print("blocked:not_found")
except OSError as e:
    print(f"blocked:errno={e.errno}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_ptrace_pid1_blocked(self, scheduler: Scheduler) -> None:
        """ptrace(PTRACE_ATTACH, 1) fails -- cannot debug PID 1."""
        code = """\
import ctypes

libc = ctypes.CDLL("libc.so.6", use_errno=True)
# PTRACE_ATTACH = 16
ret = libc.ptrace(16, 1, 0, 0)
err = ctypes.get_errno()
if ret == -1:
    print(f"blocked:errno={err}")
else:
    print("unexpected_success")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked:" in result.stdout

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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked:" in result.stdout

    async def test_guest_agent_survives_signal_storm(self, scheduler: Scheduler) -> None:
        """Session: 31 signal attempts all fail (except SIGCONT), then print("STILL_ALIVE") succeeds."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            # Send all 31 standard signals to PID 1
            # SIGCONT (18) is allowed cross-UID by Linux -- harmless, only resumes stopped processes
            storm = """\
import os
failed = 0
for sig in range(1, 32):
    try:
        os.kill(1, sig)
    except (PermissionError, OSError):
        failed += 1
print(f"SIGNALS_BLOCKED:{failed}")
"""
            r1 = await session.exec(storm)
            assert r1.exit_code == 0
            # 30 blocked (all except SIGCONT which is allowed cross-UID)
            assert "SIGNALS_BLOCKED:30" in r1.stdout

            # Verify the session (and guest-agent) is still alive
            r2 = await session.exec('print("STILL_ALIVE")')
            assert r2.exit_code == 0
            assert "STILL_ALIVE" in r2.stdout

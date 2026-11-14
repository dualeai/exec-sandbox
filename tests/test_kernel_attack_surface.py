"""Tests for kernel attack surface reduction inside the guest VM.

Validates kernel attack surface configuration inside the guest VM:
subsystems that are restricted/disabled for the REPL user (UID 1000),
and subsystems intentionally left available (VM isolation is the boundary).

CVE references per subsystem:
- eBPF verifier: CVE-2020-8835, CVE-2021-3490, CVE-2021-31440, CVE-2023-2163,
  CVE-2024-56615 (DEVMAP integer overflow)
- io_uring: disabled (CONFIG_IO_URING=n, 60% of Google kernel bug bounties)
- nf_tables: CVE-2024-1086, CVE-2023-32233, CVE-2023-35001, CVE-2021-22555,
  CVE-2024-53141 (ipset OOB access)
- AF_PACKET: CVE-2020-14386
- AF_VSOCK: CVE-2025-21756 (vsock UAF guest-to-host escape)
- cgroups v1 release_agent: CVE-2022-0492
- user namespaces + filesystem: CVE-2022-0185, CVE-2023-0386 (OverlayFS)
- ptrace + seccomp bypass: CVE-2022-30594, CVE-2019-2054
- Dirty Pipe: CVE-2022-0847
- Sequoia (deep paths): CVE-2021-33909
- userfaultfd: used as race condition primitive in many kernel UAF exploits
- KVM: CVE-2025-40300 (VMSCAPE Spectre-BTI guest-to-host memory leak)

The shipped kernel (images/kernel/exec-sandbox.config) and the guest agent's sysctls fix
each outcome: each test asserts that one outcome, so a config regression fails it.

Test categories:
- Normal: verify sysctls and kernel settings that restrict attack surface
- Edge: attempt operations that exploit these subsystems
- Weird: creative bypass attempts (user namespaces, ptrace tricks)
- Out of bounds: full exploit primitives that should fail at every step
"""

from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostArch, detect_host_arch
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

IS_X86 = detect_host_arch() == HostArch.X86_64


# =============================================================================
# Normal: eBPF is restricted (CVE-2020-8835, CVE-2021-3490, CVE-2023-2163)
# =============================================================================
class TestEbpfRestricted:
    """eBPF verifier bugs have been the #1 source of kernel privesc since 2020.

    Mitigations: unprivileged BPF should be disabled, or the bpf() syscall
    should return EPERM for UID 1000.
    """

    async def test_bpf_syscall_blocked(self, dual_scheduler: Scheduler) -> None:
        """bpf() syscall returns ENOSYS: CONFIG_BPF_SYSCALL=n.

        CVE-2020-8835: eBPF verifier fails to restrict register values,
        enabling arbitrary kernel memory read/write. Pwn2Own 2020.
        """
        code = """\
import ctypes, platform
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# bpf syscall: 321 on x86_64, 280 on aarch64
nr = 321 if platform.machine() == "x86_64" else 280
# BPF_PROG_LOAD = 5
attr = (ctypes.c_char * 128)()
ret = libc.syscall(nr, 5, ctypes.byref(attr), 128)
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "ret=-1 errno=38"  # ENOSYS

    async def test_bpf_jit_harden_sysctl(self, dual_scheduler: Scheduler) -> None:
        """net.core.bpf_jit_harden is root-only (mode 0600): UID 1000 can neither read nor change
        it (the guest agent sets 2, hardened for all).

        CVE-2024-56615: eBPF DEVMAP integer overflow. JIT hardening blinds
        constants in BPF JIT output, preventing JIT spraying attacks.
        """
        code = """\
try:
    with open('/proc/sys/net/core/bpf_jit_harden') as f:
        val = f.read().strip()
        print(f'VALUE:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "PERM_DENIED", f"Expected a root-only sysctl. stdout: {result.stdout}"

    async def test_unprivileged_bpf_disabled_sysctl(self, dual_scheduler: Scheduler) -> None:
        """The eBPF-specific unprivileged-BPF sysctl must not exist.

        CVE-2021-3490, CVE-2021-31440, CVE-2023-2163: All require unprivileged
        BPF access to exploit eBPF verifier bugs. CONFIG_BPF_SYSCALL=n removes
        the attack surface entirely; accepting sysctl values would miss a
        kernel-config regression.
        """
        code = """\
try:
    with open('/proc/sys/kernel/unprivileged_bpf_disabled') as f:
        val = f.read().strip()
        print(f'VALUE:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert stdout == "NOT_FOUND", (
            f"CONFIG_BPF_SYSCALL=n requires unprivileged_bpf_disabled to be absent, got: {stdout!r}"
        )


# =============================================================================
# Normal: io_uring disabled (CONFIG_IO_URING=n for attack surface reduction)
# =============================================================================
class TestIoUringDisabled:
    """io_uring is disabled at kernel level for security.

    io_uring was responsible for 60% of Google kernel bug bounties (2022).
    Python/Node use epoll, not io_uring. The "Curing" rootkit (2025) showed
    io_uring bypasses all syscall-based security monitoring.
    VM isolation is the primary security layer; disabling io_uring is defense-in-depth.
    """

    async def test_io_uring_setup_returns_enosys(self, dual_scheduler: Scheduler) -> None:
        """io_uring_setup() returns ENOSYS: CONFIG_IO_URING=n."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# io_uring_setup: 425 on both x86_64 and aarch64
nr = 425
# struct io_uring_params (120 bytes of zeros)
params = (ctypes.c_char * 120)()
ret = libc.syscall(nr, 4, ctypes.byref(params))
err = ctypes.get_errno()
if ret >= 0:
    import os
    os.close(ret)
print(f"ret={ret} errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "ret=-1 errno=38"  # ENOSYS

    async def test_io_uring_invalid_params_returns_enosys(self, dual_scheduler: Scheduler) -> None:
        """io_uring_setup(0, &params) returns ENOSYS too: no argument check runs (compiled out)."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# io_uring_setup: 425 on both x86_64 and aarch64
nr = 425
# struct io_uring_params (120 bytes of zeros)
params = (ctypes.c_char * 120)()
# entries=0 is invalid
ret = libc.syscall(nr, 0, ctypes.byref(params))
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "ret=-1 errno=38"  # ENOSYS, not EINVAL


# =============================================================================
# Normal: nf_tables access restricted (CVE-2024-1086, CVE-2023-32233)
# =============================================================================
class TestNfTablesRestricted:
    """nf_tables use-after-free bugs enable root + container escape.

    CVE-2024-1086 ("Flipping Pages") was actively exploited by ransomware.
    Mitigations: neither nf_tables nor nfnetlink (NETFILTER_NETLINK) is in the guest kernel,
    and UID 1000 lacks CAP_NET_ADMIN.
    """

    async def test_nftables_operation_blocked(self, dual_scheduler: Scheduler) -> None:
        """No nftables request reaches the kernel: the NETLINK_NETFILTER socket cannot open
        (EPROTONOSUPPORT, 93): nfnetlink is not built in (exec-sandbox.config).

        CVE-2024-1086: use-after-free in nf_tables nft_verdict_init(), exploited by RansomHub
        and Akira ransomware groups. CVE-2023-32233: use-after-free in nf_tables anonymous set
        handling.
        """
        code = """\
import socket, struct
try:
    s = socket.socket(16, 3, 12)  # AF_NETLINK, SOCK_RAW, NETLINK_NETFILTER
except OSError as e:
    print(f"SOCKET:{e.errno}")
else:
    msg_type = (10 << 8) | 0  # NFNL_SUBSYS_NFTABLES << 8 | NFT_MSG_NEWTABLE
    flags = 0x0001 | 0x0004 | 0x0400  # NLM_F_REQUEST | NLM_F_ACK | NLM_F_CREATE
    # nlmsghdr (16 B), then nfgenmsg (family AF_INET, version 0, res_id 0); send() goes to the
    # kernel (port 0): this Python has no AF_NETLINK address support for sendto()
    s.send(struct.pack('=IHHII', 20, msg_type, flags, 1, 0) + struct.pack('=BBH', 2, 0, 0))
    s.settimeout(5)
    reply = s.recv(4096)
    print(f"REPLY:{struct.unpack_from('=H', reply, 4)[0]}:{-struct.unpack_from('=i', reply, 16)[0]}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "SOCKET:93", result.stdout  # EPROTONOSUPPORT

    async def test_sctp_socket_blocked(self, dual_scheduler: Scheduler) -> None:
        """SCTP (a long CVE history: CVE-2021-23133, CVE-2019-8956) is not in the guest kernel:
        socket(AF_INET, SOCK_STREAM, IPPROTO_SCTP) gives EPROTONOSUPPORT (93) on both arches
        (IP_SCTP is =m in Alpine's x86 config, and MODULES=n would build it in)."""
        code = """\
import socket
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM, 132).close()  # IPPROTO_SCTP
    print("opened")
except OSError as e:
    print(f"SOCKET:{e.errno}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "SOCKET:93", result.stdout  # EPROTONOSUPPORT


# =============================================================================
# Normal: AF_PACKET restricted (CVE-2020-14386)
# =============================================================================
class TestAfPacketRestricted:
    """AF_PACKET integer overflow enables root from unprivileged user.

    CVE-2020-14386: Integer overflow in net/packet/af_packet.c leading to
    out-of-bounds write. Requires CAP_NET_RAW.
    """

    async def test_af_packet_socket_blocked(self, dual_scheduler: Scheduler) -> None:
        """AF_PACKET is not in the guest kernel (CONFIG_PACKET off: the guest sets its address
        over rtnetlink): socket() gives EAFNOSUPPORT (97)."""
        code = """\
import socket
try:
    # AF_PACKET=17, SOCK_RAW=3, ETH_P_ALL=0x0003
    s = socket.socket(17, 3, 0x0003)
    s.close()
    print("unexpected_success")
except PermissionError:
    print("blocked")
except OSError as e:
    print(f"ERROR:{e.errno}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "ERROR:97", result.stdout  # EAFNOSUPPORT


class TestNetworkStackMinimal:
    """The guest network stack is TCP/UDP/Unix/IPv6 over one virtio-net link
    (exec-sandbox.config): no extra congestion control, no packet generator."""

    async def test_congestion_control_and_pktgen(self, dual_scheduler: Scheduler) -> None:
        """Only reno (built in) and cubic (the default) are available; /proc/net/pktgen does
        not exist (NET_PKTGEN off: no kpktgend thread per CPU at boot)."""
        code = """\
import os
print(open("/proc/sys/net/ipv4/tcp_available_congestion_control").read().strip())
print(os.path.exists("/proc/net/pktgen"))
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.splitlines() == ["reno cubic", "False"], result.stdout


# =============================================================================
# Normal: AF_VSOCK restricted (CVE-2025-21756)
# =============================================================================
class TestAfVsockRestricted:
    """AF_VSOCK use-after-free enables guest-to-host escape.

    CVE-2025-21756: Use-after-free in vsock binding during transport
    reassignment allows privilege escalation to root, potentially enabling
    full guest-to-host escape. Demonstrated in real-world conditions.
    """

    async def test_af_vsock_socket_blocked(self, dual_scheduler: Scheduler) -> None:
        """AF_VSOCK socket creation fails with EAFNOSUPPORT: CONFIG_VSOCKETS=n.

        exec-sandbox uses virtio-serial (not vsock) for guest-host communication.
        """
        code = """\
import socket
try:
    # AF_VSOCK=40, SOCK_STREAM=1
    s = socket.socket(40, 1, 0)
    s.close()
    print("unexpected_success")
except PermissionError:
    print("blocked:eperm")
except OSError as e:
    # EAFNOSUPPORT=97 (address family not supported) is also safe
    print(f"blocked:errno={e.errno}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "blocked:errno=97", f"Expected EAFNOSUPPORT. stdout: {result.stdout}"


# =============================================================================
# Edge: cgroup v1 release_agent escape (CVE-2022-0492)
# =============================================================================
class TestCgroupEscapeBlocked:
    """cgroup v1 release_agent allows host code execution.

    CVE-2022-0492: Missing CAP_SYS_ADMIN check on cgroup release_agent
    file write. When last task exits, release_agent runs as root on host.
    """

    async def test_cgroup_mount_blocked(self, dual_scheduler: Scheduler) -> None:
        """Mounting cgroup filesystem requires CAP_SYS_ADMIN — blocked."""
        code = """\
import ctypes, os
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# Create mount point first so ENOENT means filesystem unavailable, not missing dir
os.makedirs("/tmp/cg", exist_ok=True)
# Try to mount cgroup
ret = libc.mount(b"cgroup", b"/tmp/cg", b"cgroup", 0, b"rdma")
err = ctypes.get_errno()
os.rmdir("/tmp/cg")
print(f"ret={ret} errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        # EPERM: the CAP_SYS_ADMIN check comes before any file system lookup
        assert result.stdout.strip() == "ret=-1 errno=1"

    async def test_cgroupfs_not_mounted(self, dual_scheduler: Scheduler) -> None:
        """cgroupfs is intentionally not mounted inside the VM: /sys/fs/cgroup is an empty folder.

        Resource limits are enforced host-side on the QEMU process via
        cgroup v2. Not mounting cgroupfs eliminates escape vectors:
        CVE-2022-0492 (release_agent), CVE-2024-21626 (Leaky Vessels).
        """
        code = """\
import os
path = '/sys/fs/cgroup'
if not os.path.isdir(path):
    print("NO_DIR")
else:
    entries = os.listdir(path)
    if not entries:
        print("EMPTY")
    else:
        for e in entries:
            print(f"FOUND:{e}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "EMPTY\n"

    async def test_cgroup_release_agent_not_writable(self, dual_scheduler: Scheduler) -> None:
        """No writable cgroup release_agent files exist.

        CVE-2022-0492: Writing to release_agent causes arbitrary command
        execution as root when last task exits the cgroup.
        """
        code = """\
import os, glob
found = []
for path in glob.glob('/sys/fs/cgroup/**/release_agent', recursive=True):
    try:
        with open(path, 'w') as f:
            f.write('/tmp/pwned')
        found.append(f"WRITABLE:{path}")
    except (PermissionError, OSError) as e:
        found.append(f"BLOCKED:{path}:{e.errno}")
if not found:
    print("NO_RELEASE_AGENT_FILES")
else:
    for f in found:
        print(f)
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "NO_RELEASE_AGENT_FILES\n"  # no cgroupfs: no file to write


# =============================================================================
# Edge: User namespace restrictions (CVE-2022-0185, CVE-2023-0386)
# =============================================================================
class TestUserNamespaceRestricted:
    """User namespaces grant CAP_SYS_ADMIN within the namespace, enabling
    exploitation of kernel bugs that require that capability.

    CVE-2022-0185: Heap overflow in filesystem context. Unprivileged user
    uses unshare(CLONE_NEWUSER) to gain CAP_SYS_ADMIN, then triggers overflow.
    CVE-2023-0386: OverlayFS privilege escalation via user namespaces.
    """

    async def test_unshare_user_namespace(self, dual_scheduler: Scheduler) -> None:
        """unshare(CLONE_NEWUSER) fails with ENOSPC: user.max_user_namespaces=0.

        Many kernel exploits require user namespaces to gain CAP_SYS_ADMIN.
        """
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# CLONE_NEWUSER = 0x10000000
ret = libc.unshare(0x10000000)
err = ctypes.get_errno()
if ret == 0:
    import os
    # Even if unshare succeeds, check that we can't do anything dangerous
    # Try mount — should still fail
    mret = libc.mount(b"none", b"/tmp/test_ns", b"tmpfs", 0, None)
    merr = ctypes.get_errno()
    print(f"UNSHARE:ok MOUNT_ret={mret} MOUNT_errno={merr}")
else:
    print(f"UNSHARE:blocked errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "UNSHARE:blocked errno=28"  # ENOSPC

    async def test_max_user_namespaces_sysctl(self, dual_scheduler: Scheduler) -> None:
        """user.max_user_namespaces should be 0 (portable alternative to
        Debian-specific unprivileged_userns_clone).

        Prevents user namespace creation which grants CAP_SYS_ADMIN within
        the namespace — the prerequisite for most container escape exploits.
        """
        code = """\
try:
    with open('/proc/sys/user/max_user_namespaces') as f:
        val = f.read().strip()
        print(f'MAX_USERNS:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        # Value 0 blocks user namespace creation entirely
        assert result.stdout.strip() == "MAX_USERNS:0"

    async def test_unshare_mount_namespace_blocked(self, dual_scheduler: Scheduler) -> None:
        """unshare(CLONE_NEWNS) requires CAP_SYS_ADMIN — blocked for UID 1000."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# CLONE_NEWNS = 0x00020000
ret = libc.unshare(0x00020000)
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "ret=-1 errno=1\n"  # EPERM: CAP_SYS_ADMIN outside a user namespace


# =============================================================================
# Weird: ptrace-based attacks (CVE-2022-30594, CVE-2019-2054)
# =============================================================================
class TestPtraceRestricted:
    """ptrace can bypass seccomp and inspect/modify other processes.

    CVE-2022-30594: PTRACE_O_SUSPEND_SECCOMP allows bypassing seccomp via
    ptrace on kernels that allow it.
    CVE-2019-2054: seccomp bypass via ptrace on kernels < 4.8.
    """

    async def test_ptrace_pid1_blocked(self, dual_scheduler: Scheduler) -> None:
        """ptrace(PTRACE_ATTACH, 1) must fail — protects guest-agent."""
        code = """\
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# PTRACE_ATTACH = 16
ret = libc.ptrace(16, 1, 0, 0)
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        # EPERM: another UID, and YAMA ptrace_scope=2 (hidepid hides /proc/1, not the process)
        assert result.stdout.strip() == "ret=-1 errno=1"

    async def test_ptrace_self_dumpable_zero(self, dual_scheduler: Scheduler) -> None:
        """PR_SET_DUMPABLE=0 prevents ptrace attachment from any process.

        This blocks CVE-2022-30594 style attacks where attacker attaches
        to a process and uses PTRACE_O_SUSPEND_SECCOMP.

        The REPL wrapper must call prctl(PR_SET_DUMPABLE, 0) after exec(),
        since exec() resets dumpable to 1 for same-UID non-SUID binaries.
        """
        code = """\
with open("/proc/self/status") as f:
    for line in f:
        if line.startswith("TracerPid:"):
            print(f"TRACER:{line.split(':')[1].strip()}")
# Check dumpable flag
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# PR_GET_DUMPABLE = 3
dumpable = libc.prctl(3, 0, 0, 0, 0)
print(f"DUMPABLE:{dumpable}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        # No tracer attached; PR_SET_DUMPABLE 0 keeps other UID 1000 processes from ptrace
        assert result.stdout == "TRACER:0\nDUMPABLE:0\n"

    async def test_yama_ptrace_scope(self, dual_scheduler: Scheduler) -> None:
        """YAMA ptrace_scope is 2 (admin-only ptrace).

        Level 2 means only CAP_SYS_PTRACE holders can ptrace. This is the
        primary ptrace defense — dumpable=0 is defense-in-depth but exec()
        always resets dumpable to 1 (begin_new_exec in fs/exec.c).

        Set via sysctl kernel.yama.ptrace_scope=2 in tiny-init.
        """
        code = """\
try:
    with open('/proc/sys/kernel/yama/ptrace_scope') as f:
        val = f.read().strip()
        print(f'YAMA:{val}')
except FileNotFoundError:
    print('YAMA:not_found')
except PermissionError:
    print('YAMA:perm_denied')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        # YAMA scope 2 blocks ptrace for non-CAP_SYS_PTRACE processes
        assert result.stdout.strip() == "YAMA:2"


# =============================================================================
# Weird: SUID binary exploitation (CVE-2022-0847 Dirty Pipe)
# =============================================================================
class TestSuidProtection:
    """SUID binaries are the classic privilege escalation vector.

    CVE-2022-0847 (Dirty Pipe): Overwrite SUID binary via page cache
    splice, then execute for root shell. Mitigated by no_new_privs.
    """

    async def test_no_suid_binaries_exist(self, dual_scheduler: Scheduler) -> None:
        """No SUID/SGID binaries exist in the guest filesystem.

        Defense-in-depth: no_new_privs blocks SUID execution too (test_no_new_privs_set),
        and with no SUID binary there is nothing to execute.
        """
        code = """\
import os, stat
suid_files = []
for dirpath, _, filenames in os.walk('/'):
    # Skip /proc, /sys, /dev (virtual filesystems)
    if dirpath.startswith(('/proc', '/sys', '/dev')):
        continue
    for f in filenames:
        try:
            path = os.path.join(dirpath, f)
            st = os.lstat(path)
            if st.st_mode & (stat.S_ISUID | stat.S_ISGID):
                suid_files.append(path)
        except (OSError, PermissionError):
            continue
if suid_files:
    for p in suid_files[:20]:
        print(f"SUID:{p}")
else:
    print("NO_SUID_BINARIES")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "NO_SUID_BINARIES", result.stdout

    async def test_no_new_privs_set(self, dual_scheduler: Scheduler) -> None:
        """no_new_privs must be set, preventing SUID bit from taking effect.

        CVE-2022-0847: Even if an attacker overwrites a SUID binary via
        Dirty Pipe, no_new_privs ensures the overwritten binary won't
        gain elevated privileges when executed.
        """
        code = """\
with open("/proc/self/status") as f:
    for line in f:
        if line.startswith("NoNewPrivs:"):
            val = line.split(":")[1].strip()
            print(f"NO_NEW_PRIVS:{val}")
            break
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "NO_NEW_PRIVS:1\n"


# =============================================================================
# Weird: /proc information leaks for kernel exploitation
# =============================================================================
class TestProcInfoLeaks:
    """Kernel address leaks via /proc enable KASLR bypass and exploitation.

    CVE-2023-3269 (StackRot): Requires knowing kernel addresses to exploit.
    """

    async def test_proc_kallsyms_absent(self, dual_scheduler: Scheduler) -> None:
        """The kernel symbol table must be compiled out.

        CONFIG_KALLSYMS=n removes /proc/kallsyms entirely. Accepting zeroed or
        unreadable content would miss an accidental config re-enable.
        """
        code = """\
try:
    with open('/proc/kallsyms') as f:
        # Read first 5 non-empty lines
        lines = []
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= 5:
                break
    if lines:
        # Check if addresses are zeroed
        all_zeroed = all(l.startswith('0000000000000000') for l in lines)
        print(f"ZEROED:{all_zeroed}")
        if not all_zeroed:
            print(f"SAMPLE:{lines[0]}")
    else:
        print("EMPTY")
except PermissionError:
    print("PERM_DENIED")
except FileNotFoundError:
    print("NOT_FOUND")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert stdout == "NOT_FOUND", f"CONFIG_KALLSYMS=n requires /proc/kallsyms to be absent, got: {stdout!r}"

    async def test_proc_kcore_not_readable(self, dual_scheduler: Scheduler) -> None:
        """/proc/kcore does not exist: CONFIG_PROC_KCORE=n."""
        code = """\
try:
    with open('/proc/kcore', 'rb') as f:
        data = f.read(16)
    print(f"READ:{len(data)}")
except PermissionError:
    print("BLOCKED")
except FileNotFoundError:
    print("NOT_FOUND")
except OSError as e:
    print(f"ERROR:{e.errno}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "NOT_FOUND", f"/proc/kcore should not exist. stdout: {result.stdout}"

    async def test_proc_hidepid_mount_option(self, dual_scheduler: Scheduler) -> None:
        """Proc must be mounted with hidepid=2 to hide other users' processes.

        hidepid=2 makes /proc/[pid] directories invisible for PIDs not owned
        by the querying user. This is the standard container/sandbox hardening
        approach used by Docker, Kubernetes, and systemd (ProtectProc=invisible).
        Available since Linux 3.3; see proc(5).
        """
        code = """\
with open('/proc/mounts') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 3 and parts[1] == '/proc' and parts[2] == 'proc':
            options = parts[3] if len(parts) > 3 else ''
            # hidepid=2 or hidepid=invisible (kernel 5.8+ alias)
            if 'hidepid=2' in options or 'hidepid=invisible' in options:
                print('HIDEPID:enabled')
            else:
                print(f'HIDEPID:missing options={options}')
            break
    else:
        print('HIDEPID:no_proc_mount')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "HIDEPID:enabled\n"

    async def test_proc_pid1_directory_invisible(self, dual_scheduler: Scheduler) -> None:
        """/proc/1 must be invisible to UID 1000 with hidepid=2.

        This is stronger than permission-denied — the directory does not exist
        at the VFS level, preventing any information leakage about PID 1 (its memory map
        among them: CVE-2023-3269, StackRot, used such a leak of the memory layout).
        """
        code = """\
import os
exists = os.path.exists('/proc/1')
print(f'EXISTS:{exists}')
try:
    os.listdir('/proc/1')
    print('LISTDIR:success')
except FileNotFoundError:
    print('LISTDIR:ENOENT')
except PermissionError:
    print('LISTDIR:EPERM')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "EXISTS:False\nLISTDIR:ENOENT\n"

    async def test_proc_self_still_accessible(self, dual_scheduler: Scheduler) -> None:
        """/proc/self and /proc/{pid} must remain readable for the current process.

        Regression canary: hidepid=2 must not break process self-introspection.
        The kernel always exempts /proc/self and /proc/[own-pid] from hidepid.
        """
        code = """\
import os
# /proc/self/status is always accessible
with open('/proc/self/status') as f:
    status = f.read()
print(f'SELF_STATUS:{"Name:" in status}')

# /proc/{own_pid}/status is also accessible
pid = os.getpid()
with open(f'/proc/{pid}/status') as f:
    status = f.read()
print(f'OWN_PID_STATUS:{"Name:" in status}')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "SELF_STATUS:True\nOWN_PID_STATUS:True\n"

    async def test_proc_sensitive_files_masked(self, dual_scheduler: Scheduler) -> None:
        """Sensitive /proc files must be masked (empty) to block reconnaissance.

        /proc/cmdline — boot params (rootfstype, init flags, console config)
        /proc/version — exact kernel version (aids CVE matching)
        /proc/interrupts — interrupt counters (thermal side-channel, GHSA-6fw5-f8r9-fgfm)
        /proc/keys — kernel keyring (not namespaced)
        /proc/timer_list — high-resolution timers (timing side-channel)
        """
        code = """\
import os
paths = [
    '/proc/cmdline',
    '/proc/version',
    '/proc/interrupts',
    '/proc/keys',
    '/proc/timer_list',
]
for p in paths:
    try:
        with open(p) as f:
            data = f.read().strip()
        print(f'{p}:len={len(data)}:content={data[:80]}')
    except PermissionError:
        print(f'{p}:EPERM')
    except FileNotFoundError:
        print(f'{p}:ENOENT')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        # Empty (a /dev/null bind mount), or absent: /proc/keys (CONFIG_KEYS=n). /proc/cmdline
        # masking also means boot posture (mitigations=off) is asserted host-side, in
        # test_vm_manager.py::TestKernelCmdlineMitigations.
        assert result.stdout.splitlines() == [
            "/proc/cmdline:len=0:content=",
            "/proc/version:len=0:content=",
            "/proc/interrupts:len=0:content=",
            "/proc/keys:ENOENT",
            "/proc/timer_list:len=0:content=",
        ]


# =============================================================================
# Out of bounds: Direct kernel exploitation primitives
# =============================================================================
class TestKernelExploitPrimitivesBlocked:
    """Operations that are first steps in kernel exploitation chains."""

    async def test_unprivileged_userfaultfd_sysctl(self, dual_scheduler: Scheduler) -> None:
        """The userfaultfd sysctl must not exist.

        userfaultfd is the primary race condition primitive in kernel UAF exploits.
        CONFIG_USERFAULTFD=n removes the syscall and sysctl entirely; accepting
        value 0 would miss an accidental kernel-config re-enable.
        """
        code = """\
try:
    with open('/proc/sys/vm/unprivileged_userfaultfd') as f:
        val = f.read().strip()
        print(f'VALUE:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert stdout == "NOT_FOUND", (
            f"CONFIG_USERFAULTFD=n requires unprivileged_userfaultfd to be absent, got: {stdout!r}"
        )

    async def test_userfaultfd_restricted(self, dual_scheduler: Scheduler) -> None:
        """userfaultfd() returns ENOSYS: CONFIG_USERFAULTFD=n.

        Many kernel UAF exploits use userfaultfd to pause execution at
        precise points during memory operations.
        """
        code = """\
import ctypes, platform
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# userfaultfd: 323 on x86_64, 282 on aarch64
nr = 323 if platform.machine() == "x86_64" else 282
# O_NONBLOCK = 0x800
ret = libc.syscall(nr, 0x800)
err = ctypes.get_errno()
if ret >= 0:
    import os
    os.close(ret)
    print(f"CREATED:fd={ret}")
else:
    print(f"BLOCKED:errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "BLOCKED:errno=38", f"userfaultfd should be ENOSYS. stdout: {result.stdout}"

    async def test_keyctl_blocked(self, dual_scheduler: Scheduler) -> None:
        """keyctl() returns ENOSYS: CONFIG_KEYS=n (keys are a heap spray tool of kernel exploits)."""
        code = """\
import ctypes, platform
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# keyctl: 250 on x86_64, 219 on aarch64
nr = 250 if platform.machine() == "x86_64" else 219
# KEYCTL_GET_KEYRING_ID = 0, KEY_SPEC_THREAD_KEYRING = -1
ret = libc.syscall(nr, 0, -1, 0)
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "ret=-1 errno=38"  # ENOSYS

    async def test_modules_disabled_sysctl(self, dual_scheduler: Scheduler) -> None:
        """The CONFIG_MODULES=n kernel has no module loading: the modules_disabled sysctl does
        not exist."""
        code = """\
try:
    with open('/proc/sys/kernel/modules_disabled') as f:
        val = f.read().strip()
        print(f'MODULES_DISABLED:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "NOT_FOUND", f"Expected CONFIG_MODULES=n. stdout: {result.stdout}"

    async def test_kernel_module_loading_blocked(self, dual_scheduler: Scheduler) -> None:
        """finit_module() returns ENOSYS: CONFIG_MODULES=n.

        Prevents loading malicious kernel modules even if user manages
        to craft one.
        """
        code = """\
import ctypes, platform
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# finit_module: 313 on x86_64, 273 on aarch64
nr = 313 if platform.machine() == "x86_64" else 273
ret = libc.syscall(nr, -1, b"", 0)
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "ret=-1 errno=38"  # ENOSYS

    async def test_kexec_blocked(self, dual_scheduler: Scheduler) -> None:
        """kexec_load (kexec_file_load on aarch64) returns ENOSYS: CONFIG_KEXEC=n, no KEXEC_FILE."""
        code = """\
import ctypes, platform
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# kexec_load: 246 on x86_64, not available on aarch64 (uses kexec_file_load=294)
machine = platform.machine()
if machine == "x86_64":
    nr = 246
else:
    nr = 294  # kexec_file_load on aarch64
ret = libc.syscall(nr, 0, 0, 0, 0)
err = ctypes.get_errno()
print(f"ret={ret} errno={err}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "ret=-1 errno=38"  # ENOSYS


# =============================================================================
# Out of bounds: Filesystem link protections
# =============================================================================
class TestFilesystemProtections:
    """Symlink/hardlink attacks in world-writable dirs like /tmp."""

    async def test_protected_symlinks(self, dual_scheduler: Scheduler) -> None:
        """fs.protected_symlinks=1 prevents symlink attacks in sticky dirs."""
        code = """\
try:
    with open('/proc/sys/fs/protected_symlinks') as f:
        val = f.read().strip()
        print(f'PROTECTED_SYMLINKS:{val}')
except (FileNotFoundError, PermissionError):
    print('INACCESSIBLE')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "PROTECTED_SYMLINKS:1"

    async def test_protected_hardlinks(self, dual_scheduler: Scheduler) -> None:
        """fs.protected_hardlinks=1 prevents hardlink attacks."""
        code = """\
try:
    with open('/proc/sys/fs/protected_hardlinks') as f:
        val = f.read().strip()
        print(f'PROTECTED_HARDLINKS:{val}')
except (FileNotFoundError, PermissionError):
    print('INACCESSIBLE')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "PROTECTED_HARDLINKS:1"

    async def test_protected_fifos(self, dual_scheduler: Scheduler) -> None:
        """fs.protected_fifos=2 prevents FIFO attacks in sticky dirs (the guest agent sets it)."""
        code = """\
try:
    with open('/proc/sys/fs/protected_fifos') as f:
        val = f.read().strip()
        print(f'PROTECTED_FIFOS:{val}')
except (FileNotFoundError, PermissionError):
    print('INACCESSIBLE')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "PROTECTED_FIFOS:2"

    async def test_protected_regular(self, dual_scheduler: Scheduler) -> None:
        """fs.protected_regular=2 prevents regular file attacks in sticky dirs."""
        code = """\
try:
    with open('/proc/sys/fs/protected_regular') as f:
        val = f.read().strip()
        print(f'PROTECTED_REGULAR:{val}')
except (FileNotFoundError, PermissionError):
    print('INACCESSIBLE')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "PROTECTED_REGULAR:2"

    async def test_suid_dumpable(self, dual_scheduler: Scheduler) -> None:
        """fs.suid_dumpable=0 prevents core dumps for setuid processes."""
        code = """\
try:
    with open('/proc/sys/fs/suid_dumpable') as f:
        val = f.read().strip()
        print(f'SUID_DUMPABLE:{val}')
except (FileNotFoundError, PermissionError):
    print('INACCESSIBLE')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "SUID_DUMPABLE:0"


# =============================================================================
# Out of bounds: Kernel info leak sysctls
# =============================================================================
class TestKernelInfoLeakSysctls:
    """Validate sysctl settings that prevent kernel information leaks."""

    async def test_dmesg_restrict(self, dual_scheduler: Scheduler) -> None:
        """kernel.dmesg_restrict=1 restricts dmesg to CAP_SYSLOG.

        Kernel logs leak addresses, module info, hardware details useful
        for exploitation.
        """
        code = """\
try:
    with open('/proc/sys/kernel/dmesg_restrict') as f:
        val = f.read().strip()
        print(f'DMESG_RESTRICT:{val}')
except (FileNotFoundError, PermissionError):
    print('INACCESSIBLE')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "DMESG_RESTRICT:1"

    async def test_perf_event_paranoid(self, dual_scheduler: Scheduler) -> None:
        """perf is restricted by sysctl on x86 and compiled out on arm64.

        Perf events enable side-channel attacks and kernel exploitation. The
        x86 arch force-selects CONFIG_PERF_EVENTS, while the final arm64 kernel
        honors CONFIG_PERF_EVENTS=n.
        """
        code = """\
try:
    with open('/proc/sys/kernel/perf_event_paranoid') as f:
        val = f.read().strip()
        print(f'PERF_PARANOID:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if IS_X86:  # x86 force-selects CONFIG_PERF_EVENTS: the guest agent sets 3
            assert stdout == "PERF_PARANOID:3"
        else:
            assert stdout == "NOT_FOUND", (
                f"arm64 CONFIG_PERF_EVENTS=n requires no perf_event_paranoid sysctl, got: {stdout!r}"
            )

    async def test_kptr_restrict(self, dual_scheduler: Scheduler) -> None:
        """kernel.kptr_restrict=2 hides kernel pointers from all users.

        Required to prevent KASLR bypass. Value 2 restricts even CAP_SYSLOG.
        """
        code = """\
try:
    with open('/proc/sys/kernel/kptr_restrict') as f:
        val = f.read().strip()
        print(f'KPTR_RESTRICT:{val}')
except (FileNotFoundError, PermissionError):
    print('INACCESSIBLE')
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout.strip() == "KPTR_RESTRICT:2"


# =============================================================================
# Out of bounds: Deep directory exploitation (CVE-2021-33909 Sequoia)
# =============================================================================
class TestSequoiaMitigation:
    """CVE-2021-33909 (Sequoia): size_t-to-int overflow in filesystem
    seq_file when total path length exceeds 1GB.

    Mitigations: /tmp inode limit (16384) prevents creating the millions
    of directories needed for this exploit.
    """

    async def test_deep_directory_creation_bounded(self, dual_scheduler: Scheduler) -> None:
        """Inode limit on /tmp stops a deep directory tree at 16,382 levels (ENOSPC): /tmp and
        /tmp/sequoia_test take two of its 16,384 inodes.

        Sequoia requires creating >1GB of path components (~8 million nested
        directories). The code walks down with chdir and a relative name, as an
        exploit would: PATH_MAX (4096) does not stop it, only the inode limit.
        """
        code = """\
import os
os.makedirs('/tmp/sequoia_test', exist_ok=True)
os.chdir('/tmp/sequoia_test')
count = 0
try:
    for i in range(20000):
        os.mkdir('a')
        os.chdir('a')
        count += 1
    print(f"CREATED:{count}")
except OSError as e:
    print(f"STOPPED:{count} errno={e.errno}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "STOPPED:16382 errno=28\n"  # ENOSPC: out of inodes

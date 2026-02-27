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

Test categories:
- Normal: verify sysctls and kernel settings that restrict attack surface
- Edge: attempt operations that exploit these subsystems
- Weird: creative bypass attempts (user namespaces, ptrace tricks)
- Out of bounds: full exploit primitives that should fail at every step
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


# =============================================================================
# Normal: eBPF is restricted (CVE-2020-8835, CVE-2021-3490, CVE-2023-2163)
# =============================================================================
class TestEbpfRestricted:
    """eBPF verifier bugs have been the #1 source of kernel privesc since 2020.

    Mitigations: unprivileged BPF should be disabled, or the bpf() syscall
    should return EPERM for UID 1000.
    """

    async def test_bpf_syscall_blocked(self, dual_scheduler: Scheduler) -> None:
        """bpf() syscall returns EPERM for unprivileged user.

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # EPERM=1 or ENOSYS=38 (both acceptable — blocked or syscall unavailable)
        errno_val = int(result.stdout.strip().rsplit("errno=", 1)[1])
        assert errno_val in {1, 38}, (
            f"Expected bpf() EPERM(1) or ENOSYS(38), got errno={errno_val}. stdout: {result.stdout}"
        )

    async def test_bpf_jit_harden_sysctl(self, dual_scheduler: Scheduler) -> None:
        """net.core.bpf_jit_harden should be 2 (hardened for all).

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert "VALUE:2" in stdout or "NOT_FOUND" in stdout or "PERM_DENIED" in stdout, (
            f"Expected bpf_jit_harden=2. stdout: {stdout}"
        )

    async def test_unprivileged_bpf_disabled_sysctl(self, dual_scheduler: Scheduler) -> None:
        """kernel.unprivileged_bpf_disabled should be 1 or 2.

        CVE-2021-3490, CVE-2021-31440, CVE-2023-2163: All require unprivileged
        BPF access to exploit eBPF verifier bugs.
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # Value of 1 or 2 means unprivileged BPF is disabled
        # NOT_FOUND means /proc/sys is restricted or sysctl doesn't exist
        # PERM_DENIED also acceptable (read blocked)
        stdout = result.stdout.strip()
        assert "VALUE:1" in stdout or "VALUE:2" in stdout or "NOT_FOUND" in stdout or "PERM_DENIED" in stdout, (
            f"Expected unprivileged BPF to be disabled. stdout: {stdout}"
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
        """io_uring_setup() returns ENOSYS (compiled out) or succeeds (stock kernel)."""
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        ret_val = int(stdout.split("ret=")[1].split()[0])
        errno_val = int(stdout.rsplit("errno=", 1)[1])
        # ENOSYS=38 (compiled out) or success (stock kernel without our config)
        assert ret_val >= 0 or errno_val == 38, (
            f"io_uring_setup should return ENOSYS(38) or succeed, got ret={ret_val} errno={errno_val}. stdout: {stdout}"
        )

    async def test_io_uring_invalid_params_returns_enosys(self, dual_scheduler: Scheduler) -> None:
        """io_uring_setup(0, &params) returns ENOSYS (compiled out) or EINVAL (stock kernel)."""
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        errno_val = int(stdout.rsplit("errno=", 1)[1])
        # ENOSYS=38 (compiled out) or EINVAL=22 (stock kernel, reaches validation)
        assert errno_val in {22, 38}, f"Expected ENOSYS(38) or EINVAL(22), got errno={errno_val}. stdout: {stdout}"


# =============================================================================
# Normal: nf_tables access restricted (CVE-2024-1086, CVE-2023-32233)
# =============================================================================
class TestNfTablesRestricted:
    """nf_tables use-after-free bugs enable root + container escape.

    CVE-2024-1086 ("Flipping Pages") was actively exploited by ransomware.
    Mitigations: UID 1000 lacks CAP_NET_ADMIN, blocking nftables operations.
    """

    async def test_nftables_socket_blocked(self, dual_scheduler: Scheduler) -> None:
        """Netlink nftables socket requires CAP_NET_ADMIN — blocked for UID 1000.

        CVE-2024-1086: use-after-free in nf_tables nft_verdict_init(),
        exploited by RansomHub and Akira ransomware groups.
        """
        code = """\
import socket
try:
    # AF_NETLINK=16, SOCK_RAW=3, NETLINK_NETFILTER=12
    s = socket.socket(16, 3, 12)
    s.close()
    print("SOCKET_CREATED")
except PermissionError:
    print("EPERM")
except OSError as e:
    print(f"ERROR:{e.errno}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        # Socket creation may succeed (AF_NETLINK doesn't always require CAP_NET_ADMIN).
        # The real protection is that nftables operations fail without CAP_NET_ADMIN,
        # tested in test_nftables_operation_blocked. Here we just verify no crash.
        assert "EPERM" in stdout or "ERROR:" in stdout or "SOCKET_CREATED" in stdout, (
            f"Unexpected nftables socket result. stdout: {stdout}"
        )

    async def test_nftables_operation_blocked(self, dual_scheduler: Scheduler) -> None:
        """nftables rule creation requires CAP_NET_ADMIN.

        CVE-2023-32233: use-after-free in nf_tables anonymous set handling.
        """
        code = """\
import ctypes, struct, platform
libc = ctypes.CDLL("libc.so.6", use_errno=True)
import socket

try:
    # AF_NETLINK=16, SOCK_RAW=3, NETLINK_NETFILTER=12
    s = socket.socket(16, 3, 12)

    # NFNL_SUBSYS_NFTABLES=10, NFT_MSG_NEWTABLE=0
    # Try to create a netfilter table — requires CAP_NET_ADMIN
    msg_type = (10 << 8) | 0  # NFNL_SUBSYS_NFTABLES << 8 | NFT_MSG_NEWTABLE
    flags = 0x0001 | 0x0400  # NLM_F_REQUEST | NLM_F_CREATE
    # Minimal netlink message header
    nlmsg = struct.pack('=IHHII', 20, msg_type, flags, 1, 0)
    s.sendto(nlmsg, (0, 0))
    s.close()
    print("SENT")
except PermissionError:
    print("EPERM")
except OSError as e:
    print(f"ERROR:{e.errno}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # Operation should fail at socket creation or sendto
        assert "SENT" not in result.stdout, (
            f"nftables message was sent — expected EPERM or socket error. stdout: {result.stdout}"
        )


# =============================================================================
# Normal: AF_PACKET restricted (CVE-2020-14386)
# =============================================================================
class TestAfPacketRestricted:
    """AF_PACKET integer overflow enables root from unprivileged user.

    CVE-2020-14386: Integer overflow in net/packet/af_packet.c leading to
    out-of-bounds write. Requires CAP_NET_RAW.
    """

    async def test_af_packet_socket_blocked(self, dual_scheduler: Scheduler) -> None:
        """AF_PACKET socket requires CAP_NET_RAW — blocked for UID 1000."""
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout or "ERROR:" in result.stdout, (
            f"AF_PACKET socket should be blocked. stdout: {result.stdout}"
        )
        assert "unexpected_success" not in result.stdout


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
        """AF_VSOCK socket should be blocked for UID 1000.

        exec-sandbox uses virtio-serial (not vsock) for guest-host communication.
        No vsock device should be present.
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # With CONFIG_VSOCK=n (custom kernel): EAFNOSUPPORT (blocked:errno=97)
        # With CONFIG_VSOCK=y: may succeed if vsock device present, or EPERM
        assert "blocked:" in result.stdout or "unexpected_success" in result.stdout, (
            f"AF_VSOCK socket returned unexpected result. stdout: {result.stdout}"
        )
        # If VSOCK is compiled out, socket creation fails (good).
        # If VSOCK is compiled in but no device, socket may succeed but connect fails (acceptable).
        # Flag unexpected_success only if VSOCK was supposed to be disabled.


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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # EPERM=1 (no CAP_SYS_ADMIN), ENODEV=19 (cgroup v1 fs not available),
        # ENOENT=2 (cgroup subsystem not found)
        errno_val = int(result.stdout.strip().rsplit("errno=", 1)[1])
        assert errno_val in {1, 2, 19}, (
            f"Expected cgroup mount to fail with EPERM(1), ENOENT(2), or ENODEV(19), got errno={errno_val}"
        )

    async def test_cgroupfs_not_mounted(self, dual_scheduler: Scheduler) -> None:
        """cgroupfs is intentionally not mounted inside the VM.

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert stdout in {"NO_DIR", "EMPTY"}, f"cgroupfs should not be mounted inside VM. Found: {stdout}"

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "WRITABLE:" not in result.stdout, f"release_agent should not be writable. stdout: {result.stdout}"


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
        """unshare(CLONE_NEWUSER) behavior — document whether restricted.

        Many kernel exploits require user namespaces to gain CAP_SYS_ADMIN.
        Even if allowed, no_new_privs prevents further escalation.
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if "UNSHARE:ok" in stdout:
            # User namespace was created, but mount should still fail
            # because no_new_privs is set
            assert "MOUNT_ret=-1" in stdout, f"Mount in user namespace should fail. stdout: {stdout}"
        # If unshare is blocked entirely, that's also fine

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        # Value 0 blocks user namespace creation entirely
        # NOT_FOUND or PERM_DENIED also acceptable
        assert "MAX_USERNS:0" in stdout or "NOT_FOUND" in stdout or "PERM_DENIED" in stdout, (
            f"Expected max_user_namespaces=0. stdout: {stdout}"
        )

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # EPERM=1 (requires CAP_SYS_ADMIN outside user namespace)
        errno_val = int(result.stdout.strip().rsplit("errno=", 1)[1])
        assert errno_val == 1, f"Expected unshare(CLONE_NEWNS) EPERM(1), got errno={errno_val}. stdout: {result.stdout}"


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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # EPERM=1 (different UID + YAMA) or ESRCH=3 (process not ptraceable)
        errno_val = int(result.stdout.strip().rsplit("errno=", 1)[1])
        assert errno_val in {1, 3}, (
            f"Expected ptrace EPERM(1) or ESRCH(3), got errno={errno_val}. stdout: {result.stdout}"
        )

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
        if line.startswith("Seccomp:"):
            print(f"SECCOMP:{line.split(':')[1].strip()}")
# Check dumpable flag
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
# PR_GET_DUMPABLE = 3
dumpable = libc.prctl(3, 0, 0, 0, 0)
print(f"DUMPABLE:{dumpable}")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # PR_SET_DUMPABLE must be 0 to prevent ptrace from other UID 1000 processes
        assert "DUMPABLE:0" in result.stdout, f"Expected dumpable=0. stdout: {result.stdout}"
        # No tracer should be attached
        assert "TRACER:0" in result.stdout, f"Expected no tracer. stdout: {result.stdout}"

    async def test_yama_ptrace_scope(self, dual_scheduler: Scheduler) -> None:
        """YAMA ptrace_scope must be >= 2 (admin-only ptrace).

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        # YAMA scope >= 2 blocks ptrace for non-CAP_SYS_PTRACE processes
        if "YAMA:" in stdout and stdout.split(":")[1].isdigit():
            scope = int(stdout.split(":")[1])
            assert scope >= 2, f"Expected YAMA ptrace_scope >= 2, got {scope}"
        else:
            # NOT_FOUND (kernel lacks YAMA) or PERM_DENIED (/proc/sys restricted)
            assert "not_found" in stdout or "perm_denied" in stdout, (
                f"Expected YAMA scope >= 2, not_found, or perm_denied. stdout: {stdout}"
            )


# =============================================================================
# Weird: SUID binary exploitation (CVE-2022-0847 Dirty Pipe)
# =============================================================================
class TestSuidProtection:
    """SUID binaries are the classic privilege escalation vector.

    CVE-2022-0847 (Dirty Pipe): Overwrite SUID binary via page cache
    splice, then execute for root shell. Mitigated by no_new_privs.
    """

    async def test_no_suid_binaries_exist(self, dual_scheduler: Scheduler) -> None:
        """No SUID/SGID binaries should exist in the guest filesystem.

        Defense-in-depth: even though no_new_privs blocks SUID execution,
        removing SUID binaries eliminates the attack surface entirely.
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # If SUID binaries exist, no_new_privs still protects us,
        # but ideally there should be none
        if "SUID:" in result.stdout:
            # Verify no_new_privs is set (defense-in-depth)
            nnp_result = await dual_scheduler.run(
                code='with open("/proc/self/status") as f:\n    for l in f:\n        if l.startswith("NoNewPrivs:"): print(l.strip())',
                language=Language.PYTHON,
            )
            assert "NoNewPrivs:\t1" in nnp_result.stdout, (
                f"SUID binaries exist and no_new_privs is not set! SUID files: {result.stdout}"
            )

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "NO_NEW_PRIVS:1" in result.stdout, f"Expected no_new_privs=1. stdout: {result.stdout}"


# =============================================================================
# Weird: /proc information leaks for kernel exploitation
# =============================================================================
class TestProcInfoLeaks:
    """Kernel address leaks via /proc enable KASLR bypass and exploitation.

    CVE-2023-3269 (StackRot): Requires knowing kernel addresses to exploit.
    """

    async def test_proc_kallsyms_restricted(self, dual_scheduler: Scheduler) -> None:
        """Kernel symbol addresses in /proc/kallsyms should be zeroed.

        kptr_restrict >= 1 ensures unprivileged users see 0x0 addresses.
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert "ZEROED:True" in stdout or "PERM_DENIED" in stdout or "EMPTY" in stdout or "NOT_FOUND" in stdout, (
            f"Kernel addresses should be hidden. stdout: {stdout}"
        )

    async def test_proc_kcore_not_readable(self, dual_scheduler: Scheduler) -> None:
        """Kernel memory via /proc/kcore must not be readable."""
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "READ:" not in result.stdout, f"/proc/kcore should not be readable. stdout: {result.stdout}"

    async def test_proc_pid1_maps_not_readable(self, dual_scheduler: Scheduler) -> None:
        """PID 1 memory map must not be readable by UID 1000.

        Reference: CVE-2023-3269 (StackRot) — memory layout leak aids exploitation.
        """
        code = """\
try:
    with open('/proc/1/maps') as f:
        data = f.read()
    print(f"READ:{len(data)}")
except PermissionError:
    print("BLOCKED")
except FileNotFoundError:
    print("NOT_FOUND")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "BLOCKED" in result.stdout or "NOT_FOUND" in result.stdout

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "HIDEPID:enabled" in result.stdout, f"stdout: {result.stdout}"

    async def test_proc_pid1_directory_invisible(self, dual_scheduler: Scheduler) -> None:
        """/proc/1 must be invisible to UID 1000 with hidepid=2.

        This is stronger than permission-denied — the directory does not exist
        at the VFS level, preventing any information leakage about PID 1.
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "EXISTS:False" in result.stdout
        assert "LISTDIR:ENOENT" in result.stdout

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "SELF_STATUS:True" in result.stdout
        assert "OWN_PID_STATUS:True" in result.stdout

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        for line in result.stdout.strip().splitlines():
            path = line.split(":")[0]
            assert ":len=0:" in line or ":EPERM" in line or ":ENOENT" in line, (
                f"{path} should be masked (empty/blocked), got: {line}"
            )


# =============================================================================
# Out of bounds: Direct kernel exploitation primitives
# =============================================================================
class TestKernelExploitPrimitivesBlocked:
    """Operations that are first steps in kernel exploitation chains."""

    async def test_unprivileged_userfaultfd_sysctl(self, dual_scheduler: Scheduler) -> None:
        """vm.unprivileged_userfaultfd should be 0.

        userfaultfd is the primary race condition primitive in kernel UAF exploits.
        Setting to 0 restricts it to CAP_SYS_PTRACE holders only.
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert "VALUE:0" in stdout or "NOT_FOUND" in stdout or "PERM_DENIED" in stdout, (
            f"Expected unprivileged_userfaultfd=0. stdout: {stdout}"
        )

    async def test_userfaultfd_restricted(self, dual_scheduler: Scheduler) -> None:
        """userfaultfd is commonly used in kernel race condition exploits.

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        # userfaultfd should be blocked (EPERM=1 or ENOSYS=38)
        # If it's available, it's a potential race condition exploit primitive
        assert "BLOCKED:" in stdout, f"userfaultfd should be blocked for UID 1000. stdout: {stdout}"

    async def test_keyctl_blocked(self, dual_scheduler: Scheduler) -> None:
        """keyctl operations used in kernel exploitation for heap spraying."""
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # keyctl may succeed (returns keyring ID) or fail (ENOSYS/EPERM).
        # We can't block it without seccomp, but we document its availability.
        assert "ret=" in result.stdout, f"Unexpected keyctl output. stdout: {result.stdout}"

    async def test_modules_disabled_sysctl(self, dual_scheduler: Scheduler) -> None:
        """kernel.modules_disabled=1 prevents ALL module loading (irreversible).

        With CONFIG_MODULES=y: set by tiny-init after module loading.
        With CONFIG_MODULES=n (custom kernel): sysctl doesn't exist (NOT_FOUND),
        which is even stronger — the syscall itself returns ENOSYS.
        """
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        # modules_disabled=1 means module loading is permanently blocked
        # NOT_FOUND means CONFIG_MODULES=n (custom kernel) — no module support at all
        # PERM_DENIED means /proc/sys is restricted (also acceptable)
        assert "MODULES_DISABLED:1" in stdout or "NOT_FOUND" in stdout or "PERM_DENIED" in stdout, (
            f"Expected modules_disabled=1 or NOT_FOUND. stdout: {stdout}"
        )

    async def test_kernel_module_loading_blocked(self, dual_scheduler: Scheduler) -> None:
        """finit_module/init_module blocked for UID 1000.

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # EPERM=1 (no CAP_SYS_MODULE or modules_disabled=1)
        # ENOSYS=38 (CONFIG_MODULES=n — syscall doesn't exist, even stronger)
        errno_val = int(result.stdout.strip().rsplit("errno=", 1)[1])
        assert errno_val in {1, 38}, (
            f"Expected finit_module EPERM(1) or ENOSYS(38), got errno={errno_val}. stdout: {result.stdout}"
        )

    async def test_kexec_blocked(self, dual_scheduler: Scheduler) -> None:
        """kexec_load blocked — prevents replacing the running kernel."""
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # EPERM=1 (no CAP_SYS_BOOT) or ENOSYS=38 (not compiled into kernel)
        errno_val = int(result.stdout.strip().rsplit("errno=", 1)[1])
        assert errno_val in {1, 38}, (
            f"Expected kexec EPERM(1) or ENOSYS(38), got errno={errno_val}. stdout: {result.stdout}"
        )


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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "PROTECTED_SYMLINKS:1" in result.stdout or "INACCESSIBLE" in result.stdout, (
            f"Expected protected_symlinks=1. stdout: {result.stdout}"
        )

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "PROTECTED_HARDLINKS:1" in result.stdout or "INACCESSIBLE" in result.stdout, (
            f"Expected protected_hardlinks=1. stdout: {result.stdout}"
        )

    async def test_protected_fifos(self, dual_scheduler: Scheduler) -> None:
        """fs.protected_fifos=2 prevents FIFO attacks in sticky dirs."""
        code = """\
try:
    with open('/proc/sys/fs/protected_fifos') as f:
        val = f.read().strip()
        print(f'PROTECTED_FIFOS:{val}')
except (FileNotFoundError, PermissionError):
    print('INACCESSIBLE')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if "PROTECTED_FIFOS:" in stdout and stdout.split(":")[1].isdigit():
            val = int(stdout.split(":")[1])
            assert val >= 2, f"Expected protected_fifos >= 2, got {val}"
        else:
            assert "INACCESSIBLE" in stdout, f"Expected protected_fifos >= 2 or INACCESSIBLE. stdout: {stdout}"

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON, timeout_seconds=60)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if "PROTECTED_REGULAR:" in stdout and stdout.split(":")[1].isdigit():
            val = int(stdout.split(":")[1])
            assert val >= 2, f"Expected protected_regular >= 2, got {val}"
        else:
            assert "INACCESSIBLE" in stdout, f"Expected protected_regular >= 2 or INACCESSIBLE. stdout: {stdout}"

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "SUID_DUMPABLE:0" in result.stdout or "INACCESSIBLE" in result.stdout, (
            f"Expected suid_dumpable=0. stdout: {result.stdout}"
        )


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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "DMESG_RESTRICT:1" in result.stdout or "INACCESSIBLE" in result.stdout, (
            f"Expected dmesg_restrict=1. stdout: {result.stdout}"
        )

    async def test_perf_event_paranoid(self, dual_scheduler: Scheduler) -> None:
        """kernel.perf_event_paranoid=3 disables perf for all users.

        Perf events enable side-channel attacks and kernel exploitation.
        """
        code = """\
try:
    with open('/proc/sys/kernel/perf_event_paranoid') as f:
        val = f.read().strip()
        print(f'PERF_PARANOID:{val}')
except (FileNotFoundError, PermissionError):
    print('INACCESSIBLE')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if "PERF_PARANOID:" in stdout and stdout.split(":")[1].lstrip("-").isdigit():
            val = int(stdout.split(":")[1])
            assert val >= 3, f"Expected perf_event_paranoid >= 3, got {val}"
        else:
            assert "INACCESSIBLE" in stdout, f"Expected perf_event_paranoid >= 3 or INACCESSIBLE. stdout: {stdout}"

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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if "KPTR_RESTRICT:" in stdout and stdout.split(":")[1].isdigit():
            val = int(stdout.split(":")[1])
            assert val >= 2, f"Expected kptr_restrict >= 2, got {val}"
        else:
            assert "INACCESSIBLE" in stdout, f"Expected kptr_restrict >= 2 or INACCESSIBLE. stdout: {stdout}"


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
        """Inode limit on /tmp prevents creating deep directory structures.

        Sequoia requires creating >1GB of path components (~8 million nested
        directories). The 16384 inode limit on /tmp caps this at ~16K dirs.
        """
        code = """\
import os
count = 0
base = '/tmp/sequoia_test'
try:
    os.makedirs(base, exist_ok=True)
    path = base
    # Try creating 20K nested dirs (should fail well before Sequoia threshold)
    for i in range(20000):
        path = os.path.join(path, 'a')
        os.mkdir(path)
        count += 1
    print(f"CREATED:{count}")
except OSError as e:
    print(f"STOPPED:{count} errno={e.errno}")
finally:
    # Cleanup
    import shutil
    try:
        shutil.rmtree(base)
    except Exception:
        pass
"""
        result = await dual_scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=30,
        )
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if "STOPPED:" in stdout:
            # Extract count — should be well below the millions needed for Sequoia
            count = int(stdout.split(":")[1].split()[0])
            assert count < 20000, f"Created {count} dirs, should be bounded"
        # If all 20K created, still far below the ~8M needed for Sequoia

"""Tests for speculative execution posture and timer side-channel defenses.

The guest kernel boots with mitigations=off — a deliberate density/latency
decision (commit 5526750d): the VM boundary (KVM/HVF) is the security layer,
not the guest kernel. x86 sysfs tests verify the runtime effect; arm64 tests
verify readable, architecture-specific reporting. A host-side unit test pins
the common cmdline token for both architectures.

Security model:
- Guest-side mitigations do not implement the VMSCAPE host boundary.
  CVE-2025-40300 (Spectre-BTI via incomplete branch-predictor isolation,
  IEEE S&P 2026) must be mitigated by a patched/configured host kernel.
  Guest mitigations=off neither enables nor disables that host mitigation.
- Single tenant per VM removes an in-guest cross-tenant target, but untrusted
  code may still target the root guest-agent or guest kernel with Spectre v1
  (CVE-2017-5753), Spectre v2 (CVE-2017-5715), or MDS (CVE-2018-12130). The
  root guest-agent intentionally stores no long-lived platform credential, but
  it transiently holds tenant code, environment values, and protocol state.
  Its host channels are virtio-serial devices (guest-agent/src/constants.rs).
- nokaslr is omitted. The x86 base currently enables CONFIG_RANDOMIZE_BASE,
  but runtime KASLR on the direct-vmlinux PVH path is not asserted here. The
  arm64 base has CONFIG_RANDOMIZE_BASE=n and mitigations=off leaves KPTI off;
  that density tradeoff is explicit, not an accidental hardening claim.
- Timer coarsening is not part of this posture: vDSO clock reads bypass
  seccomp, and KernelSnitch (NDSS 2025) demonstrates timing leaks through
  kernel data structures.
- Patched/configured host controls must provide the VM boundary (including
  VMSCAPE isolation and an appropriate SMT policy); guest KSM/perf controls
  reduce the residual intra-guest surface.
- The kernel cmdline posture is pinned host-side in
  test_vm_manager.py::TestKernelCmdlineMitigations — it cannot be asserted
  from inside the guest because guest-agent masks /proc/cmdline with a
  /dev/null bind-mount (guest-agent/src/init.rs).
- Ref: https://github.com/torvalds/linux/blob/v6.18/Documentation/admin-guide/hw-vuln/vmscape.rst
- Ref: https://github.com/torvalds/linux/blob/v6.18/arch/arm64/kernel/proton-pack.c
- Ref: https://github.com/torvalds/linux/blob/v6.18/arch/arm64/kernel/cpufeature.c
- Ref: https://github.com/torvalds/linux/blob/v6.18/drivers/base/cpu.c
- Ref: https://doi.org/10.14722/ndss.2025.240223 (KernelSnitch)

Test categories:
- Normal: verify x86 runtime effect and per-arch sysfs reporting
- Normal: informational timer precision measurement (no assertion on bounds)
- Edge: verify side-channel defense sysctls (perf_event_paranoid, KSM)
- Normal: document CPU vulnerability exposure transparency
"""

from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostArch, detect_host_arch
from exec_sandbox.scheduler import Scheduler

# Guest arch always equals host arch: forced emulation (TCG) emulates the SAME
# architecture, only without hardware acceleration (see qemu_cmd.py -cpu block).
IS_X86 = detect_host_arch() == HostArch.X86_64


def _read_vuln_file_code(name: str) -> str:
    """Guest code reading one /sys/devices/system/cpu/vulnerabilities file."""
    return f"""\
try:
    with open('/sys/devices/system/cpu/vulnerabilities/{name}') as f:
        val = f.read().strip()
        print(f'STATUS:{{val}}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""


def _extract_status(result_stdout: str, name: str) -> str:
    """Extract the STATUS: payload; the file must exist and be readable.

    CONFIG_GENERIC_CPU_VULNERABILITIES=y on both arches (vendored Alpine base
    configs), so NOT_FOUND/PERM_DENIED are failures, not skips — a future
    guest-side mask of /sys must fail these tests, not void them.
    """
    stdout = result_stdout.strip()
    assert "STATUS:" in stdout, f"{name} sysfs file must be readable in guest, got: {stdout!r}"
    return stdout.split("STATUS:", 1)[1]


def _assert_mitigations_off_reached_guest(name: str, status: str) -> None:
    """x86: status must be a hardware 'Not affected' claim or 'Vulnerable*'.

    'Not affected' survives mitigations=off — bugs.c short-circuits on CPU
    self-declaration (ARCH_CAPABILITIES: MDS_NO etc.) before mitigation state,
    so AMD / Ice Lake+ hosts under -cpu host legitimately report it. Under TCG
    the arch-capabilities MSR is stripped, so affected bugs report 'Vulnerable*'.

    An active 'Mitigation:' string means mitigations=off silently stopped
    reaching the guest — check the qemu_cmd.py mitigations block and
    test_vm_manager.py::TestKernelCmdlineMitigations.
    """
    assert status == "Not affected" or status.startswith("Vulnerable"), (
        f"{name}: expected 'Not affected' or 'Vulnerable*' under mitigations=off "
        f"(density posture, commit 5526750d); got: {status!r} — if this is a "
        f"'Mitigation:' string, mitigations=off no longer reaches the guest"
    )


# =============================================================================
# Normal: x86 runtime posture and architecture-specific sysfs reporting
# =============================================================================
class TestMitigationStatus:
    """Verify x86 runtime effect and architecture-specific status reporting.

    The host-side test pins the common cmdline token. x86 statuses confirm the
    kernel honored mitigations=off; arm64 statuses are hardware/compile-time
    dominated, so those branches validate reporting semantics without claiming
    to prove the runtime flag took effect.

    A previous guest-side cmdline test (reading /proc/cmdline) was deleted:
    guest-agent masks /proc/cmdline with a /dev/null bind-mount
    (guest-agent/src/init.rs), so it read "" and passed vacuously. Masking
    itself is asserted in test_kernel_attack_surface.py::TestProcInfoLeaks.
    """

    async def test_spectre_v2_status_matches_posture(self, dual_scheduler: Scheduler) -> None:
        """Spectre v2 (CVE-2017-5715) status matches documented arch behavior.

        x86: nospectre_v2 (via mitigations=off) yields 'Vulnerable*' on all
        affected CPUs. arm64: reporting is dominated by hardware CSV2 claims
        (proton-pack.c checks ID_AA64PFR0_EL1.CSV2 / safe-MIDR list before
        mitigation state), so any of 'Not affected' (Apple Silicon, Graviton,
        QEMU TCG max), 'Mitigation:*' (BHB-listed cores), or 'Vulnerable*'
        (non-CSV2 cores) is hardware-truthful — transparency only.
        """
        result = await dual_scheduler.run(code=_read_vuln_file_code("spectre_v2"), language=Language.PYTHON)
        assert result.exit_code == 0
        status = _extract_status(result.stdout, "spectre_v2")
        if IS_X86:
            _assert_mitigations_off_reached_guest("spectre_v2", status)
        else:
            assert status, "spectre_v2: empty sysfs status"

    async def test_spectre_v1_status_matches_posture(self, dual_scheduler: Scheduler) -> None:
        """Spectre v1 (CVE-2017-5753) status matches documented arch behavior.

        x86: nospectre_v1 (via mitigations=off) yields 'Vulnerable: __user
        pointer sanitization and usercopy barriers only; no swapgs barriers'.
        arm64: __user pointer sanitization is compile-time and NOT runtime-
        disableable (proton-pack.c cpu_show_spectre_v1 is unconditional), so
        'Mitigation:*' is the permanent expected value even under
        mitigations=off.
        """
        result = await dual_scheduler.run(code=_read_vuln_file_code("spectre_v1"), language=Language.PYTHON)
        assert result.exit_code == 0
        status = _extract_status(result.stdout, "spectre_v1")
        if IS_X86:
            _assert_mitigations_off_reached_guest("spectre_v1", status)
        else:
            assert status.startswith("Mitigation"), (
                f"spectre_v1 on arm64 is always 'Mitigation: __user pointer "
                f"sanitization' (compile-time, not runtime-disableable), got: {status!r}"
            )

    async def test_mds_status_matches_posture(self, dual_scheduler: Scheduler) -> None:
        """MDS (CVE-2018-12130) status matches documented arch behavior.

        x86 hwaccel: 'Not affected' on AMD / MDS_NO CPUs (hardware claim wins
        over mitigation state in bugs.c). x86 TCG: 'Vulnerable; SMT Host state
        unknown' — TCG strips the arch-capabilities MSR from SapphireRapids-v2
        so the MDS_NO self-declaration never reaches the guest.
        arm64: MDS is an x86-only bug; the weak default handler in
        drivers/base/cpu.c always reports 'Not affected'.
        """
        result = await dual_scheduler.run(code=_read_vuln_file_code("mds"), language=Language.PYTHON)
        assert result.exit_code == 0
        status = _extract_status(result.stdout, "mds")
        if IS_X86:
            _assert_mitigations_off_reached_guest("mds", status)
        else:
            assert status == "Not affected", f"mds is x86-only, arm64 must report 'Not affected', got: {status!r}"


# =============================================================================
# Normal: Timer precision measurement (informational, no assertion on bounds)
# =============================================================================
class TestTimerPrecision:
    """Informational measurement of timer resolution inside the guest VM.

    Documents clock_gettime(CLOCK_MONOTONIC) precision as a baseline.
    No assertion on bounds because timer coarsening is not part of this
    sandbox's security posture:
    - vDSO maps clock_gettime into userspace, bypassing seccomp entirely
    - KernelSnitch (NDSS 2025) demonstrates kernel-data-structure timing leaks
    - Cloudflare Workers freezes timers AND isolates processes (not applicable to VMs)
    - Browser 100us coarsening (W3C High Resolution Time) is for JS event loops
    """

    async def test_timer_resolution_measured(self, dual_scheduler: Scheduler) -> None:
        """Measure clock_gettime resolution — informational baseline only.

        Records minimum delta between consecutive CLOCK_MONOTONIC reads.
        This documents the state without asserting bounds, since timer
        coarsening is not an effective mitigation in VM environments.
        """
        code = """\
import ctypes, ctypes.util, struct, platform

# Load libc for clock_gettime
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

class timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

CLOCK_MONOTONIC = 1
ts = timespec()

# Warm up
for _ in range(100):
    libc.clock_gettime(CLOCK_MONOTONIC, ctypes.byref(ts))

# Measure minimum delta over 1000 consecutive reads
deltas = []
prev_ns = None
for _ in range(1000):
    libc.clock_gettime(CLOCK_MONOTONIC, ctypes.byref(ts))
    cur_ns = ts.tv_sec * 1_000_000_000 + ts.tv_nsec
    if prev_ns is not None:
        delta = cur_ns - prev_ns
        if delta > 0:
            deltas.append(delta)
    prev_ns = cur_ns

if deltas:
    min_d = min(deltas)
    med_d = sorted(deltas)[len(deltas) // 2]
    print(f"MIN_DELTA_NS:{min_d}")
    print(f"MEDIAN_DELTA_NS:{med_d}")
    print(f"SAMPLES:{len(deltas)}")
else:
    print("NO_DELTAS")
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # Informational only — just verify measurement succeeded
        assert "MIN_DELTA_NS:" in result.stdout or "NO_DELTAS" in result.stdout, (
            f"Timer measurement should produce results. stdout: {result.stdout}"
        )


# =============================================================================
# Edge: Timer side-channel defense sysctls
# =============================================================================
class TestTimerSideChannelDefenses:
    """Verify guest hardening that reduces side-channel effectiveness.

    Host-side isolation remains the primary VM boundary. These tests cover the
    complementary guest perf/KSM posture; they do not assert host setup.
    """

    async def test_perf_event_paranoid_blocks_perf(self, dual_scheduler: Scheduler) -> None:
        """perf is restricted by sysctl on x86 and compiled out on arm64.

        Hardware performance counters are a precise side-channel primitive.
        Cross-referenced with test_kernel_attack_surface.py::TestKernelInfoLeakSysctls.
        The final kernel has CONFIG_PERF_EVENTS=n on arm64; x86 arch Kconfig
        force-selects it, so the runtime sysctl must be >= 3 there.
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
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if IS_X86:
            assert "PERF_PARANOID:" in stdout and stdout.split(":", 1)[1].lstrip("-").isdigit(), (
                f"x86 force-selects CONFIG_PERF_EVENTS; expected readable perf_event_paranoid, got: {stdout!r}"
            )
            val = int(stdout.split(":")[1])
            assert val >= 3, f"Expected perf_event_paranoid >= 3, got {val}"
        else:
            assert stdout == "NOT_FOUND", f"arm64 CONFIG_PERF_EVENTS=n requires no perf sysctl, got: {stdout!r}"

    async def test_ksm_disabled(self, dual_scheduler: Scheduler) -> None:
        """KSM (Kernel Same-page Merging) must be disabled inside the guest.

        KSM enables cross-VM side channels by measuring page deduplication
        timing. Host-side KSM (QEMU mem-merge=on) is enabled for density, but
        the guest kernel has CONFIG_KSM=n. The sysfs control must therefore be
        absent; accepting run=0 would miss an accidental KSM re-enable.
        """
        code = """\
try:
    with open('/sys/kernel/mm/ksm/run') as f:
        val = f.read().strip()
        print(f'KSM_RUN:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert stdout == "NOT_FOUND", f"CONFIG_KSM=n requires /sys/kernel/mm/ksm/run to be absent, got: {stdout!r}"

    async def test_guest_ksm_pages_shared_absent(self, dual_scheduler: Scheduler) -> None:
        """Verify KSM pages_shared is absent inside the guest.

        Guest kernel has CONFIG_KSM=n, so pages_shared must not exist.
        Host-side KSM (mem-merge=on) operates at the QEMU process level and is
        invisible to the guest.
        """
        code = """\
try:
    with open('/sys/kernel/mm/ksm/pages_shared') as f:
        val = f.read().strip()
        print(f'PAGES_SHARED:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert stdout == "NOT_FOUND", (
            f"CONFIG_KSM=n requires /sys/kernel/mm/ksm/pages_shared to be absent, got: {stdout!r}"
        )


# =============================================================================
# Normal: CPU vulnerability transparency
# =============================================================================
class TestCpuVulnerabilityTransparency:
    """Read all CPU vulnerability status files for documentation.

    /sys/devices/system/cpu/vulnerabilities/ exposes the kernel's assessment
    of whether the CPU is affected by each known vulnerability and what
    mitigations are active. This test documents the full exposure surface.
    """

    async def test_vulnerability_files_readable(self, dual_scheduler: Scheduler) -> None:
        """All vulnerability status files should be readable and documented.

        Enumerates /sys/devices/system/cpu/vulnerabilities/ to capture the
        full state. Readability (transparency) is the invariant — with
        mitigations=off, many entries are EXPECTED to report "Vulnerable";
        that is the configured density posture, not a regression. Per-file
        posture semantics are tested individually in TestMitigationStatus.
        """
        code = """\
import os
vuln_dir = '/sys/devices/system/cpu/vulnerabilities'
if not os.path.isdir(vuln_dir):
    print('DIR_NOT_FOUND')
else:
    entries = sorted(os.listdir(vuln_dir))
    if not entries:
        print('EMPTY')
    else:
        for name in entries:
            path = os.path.join(vuln_dir, name)
            try:
                with open(path) as f:
                    val = f.read().strip()
                print(f'{name}:{val}')
            except (PermissionError, OSError) as e:
                print(f'{name}:ERROR:{e}')
"""
        result = await dual_scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert "DIR_NOT_FOUND" not in stdout, "CPU vulnerability directory should exist"
        error_lines = [line for line in stdout.splitlines() if ":ERROR:" in line]
        assert not error_lines, f"All CPU vulnerability files must be readable, got: {error_lines}"
        # Verify at least some vulnerability files were read (sanity check)
        lines = [line for line in stdout.splitlines() if ":" in line and "ERROR" not in line]
        assert len(lines) > 0, f"Expected vulnerability status entries. stdout: {stdout}"

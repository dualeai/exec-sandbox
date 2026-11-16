"""Tests for speculative execution posture and timer side-channel defenses.

The guest kernel boots with mitigations=off — a deliberate density/latency
decision: the VM boundary (KVM/HVF) is the security layer, not the guest kernel.
x86 sysfs tests verify the runtime effect; arm64 tests verify readable,
architecture-specific reporting. A host-side unit test pins the common cmdline
token for both architectures.

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
- nokaslr is omitted. The x86 base config enables CONFIG_RANDOMIZE_BASE,
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
- Normal: the guest monotonic clock never goes back (no assertion on its resolution)
- Edge: verify the guest has no KSM (perf_event_paranoid: test_kernel_attack_surface.py)
- Normal: document CPU vulnerability exposure transparency
"""

import re

from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostArch, detect_host_arch
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

# Guest arch always equals host arch: forced emulation (TCG) emulates the SAME
# architecture, only without hardware acceleration (see qemu_cmd.py -cpu block).
IS_X86 = detect_host_arch() == HostArch.X86_64


def _read_vuln_file_code(name: str) -> str:
    """Guest code printing one /sys/devices/system/cpu/vulnerabilities file.

    CONFIG_GENERIC_CPU_VULNERABILITIES=y on both arches (vendored Alpine base
    configs), so a missing or unreadable file fails the exec (exit code 1), not a
    skip — a future guest-side mask of /sys must fail these tests, not void them.
    """
    return f"print(open('/sys/devices/system/cpu/vulnerabilities/{name}').read().strip())"


def _assert_mitigations_off_reached_guest(name: str, status: str) -> None:
    """x86: status must be a hardware 'Not affected' claim or 'Vulnerable*'.

    'Not affected' survives mitigations=off — the kernel decides whether the CPU is
    affected (vendor and model lists, ARCH_CAPABILITIES bits) before the mitigation
    state. Every x86 guest (KVM and TCG) gets the fixed CPU model of qemu_cmd.py
    (_X86_CPU_FLAGS, with the host's vendor) and no arch-capabilities flag, so the
    vendor lists decide: an AMD host can report 'Not affected', and affected bugs
    report 'Vulnerable*'.

    An active 'Mitigation:' string means mitigations=off does not reach the
    guest — check the qemu_cmd.py mitigations block and
    test_vm_manager.py::TestKernelCmdlineMitigations.
    """
    assert status == "Not affected" or status.startswith("Vulnerable"), (
        f"{name}: expected 'Not affected' or 'Vulnerable*' under mitigations=off "
        f"(density posture); got: {status!r} — a 'Mitigation:' string means "
        f"mitigations=off does not reach the guest"
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

    No guest-side cmdline test: guest-agent masks /proc/cmdline with a
    /dev/null bind-mount (guest-agent/src/init.rs), so a read gives "" and
    cannot fail. The mask itself is asserted in
    test_kernel_attack_surface.py::TestProcInfoLeaks.
    """

    async def test_spectre_v2_status_matches_posture(self, dual_scheduler: Scheduler) -> None:
        """Spectre v2 (CVE-2017-5715) status matches documented arch behavior.

        x86: nospectre_v2 (via mitigations=off) yields 'Vulnerable*' on all
        affected CPUs. arm64: reporting is dominated by hardware CSV2 and BHB claims
        (proton-pack.c checks ID_AA64PFR0_EL1.CSV2 / safe-MIDR list before
        mitigation state), so any of 'Not affected', 'Mitigation:*' (Apple Silicon
        under HVF and the neoverse-v1 TCG model: 'Mitigation: CSV2, but not BHB'),
        or 'Vulnerable*' (non-CSV2 cores) is hardware-truthful — transparency only.
        """
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), _read_vuln_file_code("spectre_v2"))
        assert result.exit_code == 0, result.stderr
        status = result.stdout.strip()
        if IS_X86:
            _assert_mitigations_off_reached_guest("spectre_v2", status)
        else:
            assert status == "Not affected" or status.startswith(("Mitigation: ", "Vulnerable")), (
                f"spectre_v2: not a kernel status string: {status!r}"
            )

    async def test_spectre_v1_status_matches_posture(self, dual_scheduler: Scheduler) -> None:
        """Spectre v1 (CVE-2017-5753) status matches documented arch behavior.

        x86: nospectre_v1 (via mitigations=off) yields 'Vulnerable: __user
        pointer sanitization and usercopy barriers only; no swapgs barriers'.
        arm64: __user pointer sanitization is compile-time and NOT runtime-
        disableable (proton-pack.c cpu_show_spectre_v1 is unconditional), so
        'Mitigation:*' is the permanent expected value even under
        mitigations=off.
        """
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), _read_vuln_file_code("spectre_v1"))
        assert result.exit_code == 0, result.stderr
        status = result.stdout.strip()
        if IS_X86:
            _assert_mitigations_off_reached_guest("spectre_v1", status)
        else:
            assert status.startswith("Mitigation"), (
                f"spectre_v1 on arm64 is always 'Mitigation: __user pointer "
                f"sanitization' (compile-time, not runtime-disableable), got: {status!r}"
            )

    async def test_mds_status_matches_posture(self, dual_scheduler: Scheduler) -> None:
        """MDS (CVE-2018-12130) status matches documented arch behavior.

        x86 (KVM and TCG): 'Not affected' on an AMD host (the kernel's vendor list
        wins over mitigation state), else 'Vulnerable; SMT Host state unknown': the
        fixed guest CPU model has no arch-capabilities flag, so no MDS_NO reaches
        the guest. arm64: MDS is an x86-only bug; the weak default handler in
        drivers/base/cpu.c always reports 'Not affected'.
        """
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), _read_vuln_file_code("mds"))
        assert result.exit_code == 0, result.stderr
        status = result.stdout.strip()
        if IS_X86:
            _assert_mitigations_off_reached_guest("mds", status)
        else:
            assert status == "Not affected", f"mds is x86-only, arm64 must report 'Not affected', got: {status!r}"


# =============================================================================
# Normal: Timer reads (no assertion on resolution)
# =============================================================================
class TestTimerPrecision:
    """clock_gettime(CLOCK_MONOTONIC) inside the guest VM.

    The test prints the resolution but does not assert it, because timer
    coarsening is not part of this sandbox's security posture:
    - vDSO maps clock_gettime into userspace, bypassing seccomp entirely
    - KernelSnitch (NDSS 2025) demonstrates kernel-data-structure timing leaks
    - Cloudflare Workers freezes timers AND isolates processes (not applicable to VMs)
    - Browser 100us coarsening (W3C High Resolution Time) is for JS event loops
    """

    async def test_timer_resolution_measured(self, dual_scheduler: Scheduler) -> None:
        """1000 back-to-back CLOCK_MONOTONIC reads never go back, and the clock advances.

        The minimum and median step go to stdout for information: timer
        coarsening is not a mitigation in VM environments, so no bound on them.
        """
        code = """\
import time

# time.monotonic_ns() is clock_gettime(CLOCK_MONOTONIC)
for _ in range(100):  # warm up
    time.monotonic_ns()

# Steps between 1000 consecutive reads
reads = [time.monotonic_ns() for _ in range(1000)]
deltas = [b - a for a, b in zip(reads, reads[1:])]

forward = sorted(d for d in deltas if d > 0)
print(f"BACKWARD:{sum(d < 0 for d in deltas)}")
print(f"FORWARD:{len(forward)}")
if forward:
    print(f"MIN_DELTA_NS:{forward[0]} MEDIAN_DELTA_NS:{forward[len(forward) // 2]}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0, result.stderr
        # A MIN_DELTA_NS line only when at least one step went forward: the clock advances
        steps = r"BACKWARD:0\nFORWARD:\d+\nMIN_DELTA_NS:\d+ MEDIAN_DELTA_NS:\d+\n"
        assert re.fullmatch(steps, result.stdout), result.stdout


# =============================================================================
# Edge: Timer side-channel defense sysctls
# =============================================================================
class TestTimerSideChannelDefenses:
    """Verify guest hardening that reduces side-channel effectiveness.

    Host-side isolation remains the primary VM boundary. These tests cover the
    complementary guest perf/KSM posture; they do not assert host setup.
    """

    async def test_ksm_disabled(self, dual_scheduler: Scheduler) -> None:
        """KSM (Kernel Same-page Merging) must be disabled inside the guest.

        KSM enables cross-VM side channels by measuring page deduplication
        timing. Host-side KSM (QEMU mem-merge=on, Linux hosts) is enabled for
        density, but the guest kernel has CONFIG_KSM=n. The sysfs control must therefore be
        absent; accepting run=0 would miss an accidental KSM re-enable.
        Host-side KSM (mem-merge=on) operates at the QEMU process level and is
        invisible to the guest. perf_event_paranoid, the other guest control:
        test_kernel_attack_surface.py::TestKernelInfoLeakSysctls::test_info_leak_sysctls.
        """
        code = "import os; print(os.path.exists('/sys/kernel/mm/ksm/run'))"
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "False\n"), result.stderr


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
import errno, os
vuln_dir = '/sys/devices/system/cpu/vulnerabilities'
for name in sorted(os.listdir(vuln_dir)):
    try:
        print(f"{name}:{open(os.path.join(vuln_dir, name)).read().strip()}")
    except OSError as e:  # the other files are still read
        print(f"{name}:ERROR:{errno.errorcode[e.errno]}")
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.PYTHON), code)
        # A missing folder fails the exec
        assert result.exit_code == 0, result.stderr
        # The files that TestMitigationStatus reads are listed, each with a status; none unreadable
        statuses = dict(line.split(":", 1) for line in result.stdout.splitlines())
        assert {"mds", "spectre_v1", "spectre_v2"} <= statuses.keys(), result.stdout
        assert all(statuses.values()), result.stdout
        assert [name for name, status in statuses.items() if status.startswith("ERROR:")] == [], result.stdout

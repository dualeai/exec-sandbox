"""Tests for speculative execution mitigations and timer side-channel posture.

Validates that the guest VM runs with kernel-default speculative execution
mitigations enabled (mitigations=auto), matching the security posture of
AWS Lambda/Firecracker. Documents timer precision as informational baseline.

CVE references:
- CVE-2025-40300 (VMSCAPE): Spectre-BTI guest-to-host memory leak via KVM/QEMU
  incomplete branch predictor isolation (IEEE S&P 2026, ETH Zurich)
- Spectre v1 (CVE-2017-5753): Bounds check bypass via branch misprediction
- Spectre v2 (CVE-2017-5715): Branch target injection
- MDS (CVE-2018-12130): Microarchitectural data sampling
- KernelSnitch (NDSS 2025): clock_gettime as side-channel primitive — shows
  timer coarsening is provably ineffective (kernel data structure timing leaks)

Security model:
- Guest mitigations at kernel defaults (mitigations=auto) — same as Firecracker
- No timer coarsening — no VM-based FaaS does it; vDSO bypasses seccomp anyway
- Host-side hardening (SMT control, KSM disabled, perf restricted) is primary defense
- Ref: https://github.com/firecracker-microvm/firecracker/blob/main/docs/prod-host-setup.md
- Ref: https://docs.kernel.org/admin-guide/hw-vuln/spectre.html

Test categories:
- Normal: verify mitigations are active (not "Vulnerable")
- Normal: informational timer precision measurement (no assertion on bounds)
- Edge: verify side-channel defense sysctls (perf_event_paranoid, KSM)
- Normal: document CPU vulnerability exposure transparency
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


# =============================================================================
# Normal: Spectre/MDS mitigations active (CVE-2025-40300, CVE-2017-5715)
# =============================================================================
class TestMitigationStatus:
    """Verify CPU speculative execution mitigations are enabled in guest.

    With mitigations=auto (kernel default), the kernel enables mitigations
    relevant to the actual CPU. This matches Firecracker/AWS Lambda posture.
    CVE-2025-40300 (VMSCAPE) demonstrates guest-to-host Spectre-BTI leaks
    when mitigations are disabled.
    """

    async def test_spectre_v2_mitigated(self, scheduler: Scheduler) -> None:
        """Spectre v2 (branch target injection) must not report 'Vulnerable'.

        CVE-2017-5715: Branch target injection enables cross-boundary
        speculative execution. CVE-2025-40300 exploits this for guest-to-host.
        """
        code = """\
try:
    with open('/sys/devices/system/cpu/vulnerabilities/spectre_v2') as f:
        val = f.read().strip()
        print(f'STATUS:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if "STATUS:" in stdout:
            status = stdout.split("STATUS:", 1)[1]
            assert "Vulnerable" not in status or "Mitigation" in status, (
                f"Spectre v2 should be mitigated, got: {status}"
            )

    async def test_spectre_v1_mitigated(self, scheduler: Scheduler) -> None:
        """Spectre v1 (bounds check bypass) must not report 'Vulnerable'.

        CVE-2017-5753: Conditional branch misprediction enables out-of-bounds
        speculative reads. Kernel mitigates with array_index_nospec barriers.
        """
        code = """\
try:
    with open('/sys/devices/system/cpu/vulnerabilities/spectre_v1') as f:
        val = f.read().strip()
        print(f'STATUS:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if "STATUS:" in stdout:
            status = stdout.split("STATUS:", 1)[1]
            assert "Vulnerable" not in status or "Mitigation" in status, (
                f"Spectre v1 should be mitigated, got: {status}"
            )

    async def test_mds_mitigated(self, scheduler: Scheduler) -> None:
        """MDS (microarchitectural data sampling) must not report 'Vulnerable'.

        CVE-2018-12130: Enables reading stale data from CPU buffers.
        Mitigation: CPU buffer clearing on context switches.
        """
        code = """\
try:
    with open('/sys/devices/system/cpu/vulnerabilities/mds') as f:
        val = f.read().strip()
        print(f'STATUS:{val}')
except FileNotFoundError:
    print('NOT_FOUND')
except PermissionError:
    print('PERM_DENIED')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if "STATUS:" in stdout:
            status = stdout.split("STATUS:", 1)[1]
            # "Not affected" is fine (ARM or newer Intel CPUs)
            assert "Vulnerable" not in status or "Mitigation" in status, (
                f"MDS should be mitigated or not affected, got: {status}"
            )

    async def test_mitigations_not_disabled_in_cmdline(self, scheduler: Scheduler) -> None:
        """Kernel cmdline must NOT contain 'mitigations=off' or 'nokaslr'.

        These were previously used for performance but disable critical
        protections against speculative execution attacks (CVE-2025-40300).
        """
        code = """\
with open('/proc/cmdline') as f:
    cmdline = f.read().strip()
print(f'CMDLINE:{cmdline}')
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        cmdline = result.stdout.strip().split("CMDLINE:", 1)[1]
        assert "mitigations=off" not in cmdline, f"mitigations=off must not be in kernel cmdline: {cmdline}"
        assert "nokaslr" not in cmdline, f"nokaslr must not be in kernel cmdline: {cmdline}"


# =============================================================================
# Normal: Timer precision measurement (informational, no assertion on bounds)
# =============================================================================
class TestTimerPrecision:
    """Informational measurement of timer resolution inside the guest VM.

    Documents clock_gettime(CLOCK_MONOTONIC) precision as a baseline.
    No assertion on bounds — no VM-based FaaS coarsens timers:
    - vDSO maps clock_gettime into userspace, bypassing seccomp entirely
    - KernelSnitch (NDSS 2025) shows timer coarsening is provably ineffective
    - Cloudflare Workers freezes timers AND isolates processes (not applicable to VMs)
    - Browser 100us coarsening (W3C High Resolution Time) is for JS event loops
    """

    async def test_timer_resolution_measured(self, scheduler: Scheduler) -> None:
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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        # Informational only — just verify measurement succeeded
        assert "MIN_DELTA_NS:" in result.stdout or "NO_DELTAS" in result.stdout, (
            f"Timer measurement should produce results. stdout: {result.stdout}"
        )


# =============================================================================
# Edge: Timer side-channel defense sysctls
# =============================================================================
class TestTimerSideChannelDefenses:
    """Verify host-side defenses that reduce side-channel effectiveness.

    The primary defense against speculative execution side channels in VMs
    is host-side hardening (Firecracker prod-host-setup model), not timer
    coarsening. These tests verify the guest-visible effects.
    """

    async def test_perf_event_paranoid_blocks_perf(self, scheduler: Scheduler) -> None:
        """perf_event_paranoid >= 3 disables perf counters for all users.

        Hardware performance counters are a precise side-channel primitive.
        Cross-referenced with test_kernel_attack_surface.py::TestKernelInfoLeakSysctls.
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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        if "PERF_PARANOID:" in stdout and stdout.split(":")[1].lstrip("-").isdigit():
            val = int(stdout.split(":")[1])
            assert val >= 3, f"Expected perf_event_paranoid >= 3, got {val}"
        else:
            assert "NOT_FOUND" in stdout or "PERM_DENIED" in stdout, (
                f"Expected perf_event_paranoid >= 3 or inaccessible. stdout: {stdout}"
            )

    async def test_ksm_disabled(self, scheduler: Scheduler) -> None:
        """KSM (Kernel Same-page Merging) must be disabled inside the guest.

        KSM enables cross-VM side channels by measuring page deduplication
        timing. QEMU mem-merge=off disables KSM for the VM's memory on the
        host side; this test verifies the guest kernel also has it off.
        Ref: Firecracker prod-host-setup recommends disabling KSM.
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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        # KSM run=0 means disabled, NOT_FOUND means not compiled in — both safe
        assert "KSM_RUN:0" in stdout or "NOT_FOUND" in stdout or "PERM_DENIED" in stdout, (
            f"Expected KSM disabled (run=0) or not available. stdout: {stdout}"
        )

    async def test_mem_merge_off_effect(self, scheduler: Scheduler) -> None:
        """Verify KSM pages_shared is 0 (effect of QEMU mem-merge=off).

        Even if KSM kernel support exists, mem-merge=off means QEMU never
        registers the VM's memory with KSM, so pages_shared should be 0.
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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert "PAGES_SHARED:0" in stdout or "NOT_FOUND" in stdout or "PERM_DENIED" in stdout, (
            f"Expected KSM pages_shared=0. stdout: {stdout}"
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

    async def test_vulnerability_files_readable(self, scheduler: Scheduler) -> None:
        """All vulnerability status files should be readable and documented.

        Enumerates /sys/devices/system/cpu/vulnerabilities/ to capture the
        full mitigation state. This is informational — some vulnerabilities
        may report "Vulnerable" when the hypervisor/CPU lacks hardware support
        for a specific mitigation (e.g., spec_store_bypass under HVF).
        The critical mitigations (Spectre v1/v2, MDS) are tested individually.
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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        stdout = result.stdout.strip()
        assert "DIR_NOT_FOUND" not in stdout, "CPU vulnerability directory should exist"
        # Verify at least some vulnerability files were read (sanity check)
        lines = [line for line in stdout.splitlines() if ":" in line and "ERROR" not in line]
        assert len(lines) > 0, f"Expected vulnerability status entries. stdout: {stdout}"

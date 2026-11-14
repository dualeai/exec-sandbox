"""QEMU command line builder for microVM execution.

Builds QEMU command arguments based on platform capabilities, acceleration type,
and VM configuration.
"""

import os
from pathlib import Path

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.images import GuestImages
from exec_sandbox.models import ExposedPort
from exec_sandbox.permission_utils import QEMU_VM_SUDO, get_qemu_vm_uid
from exec_sandbox.platform_utils import HostArch, HostOS, detect_host_os, detect_host_x86_vendor
from exec_sandbox.settings import Settings
from exec_sandbox.system_probes import (
    check_tsc_deadline,
    detect_accel_type,
    probe_drive_direct_io,
    probe_io_uring_support,
    probe_qemu_runnable,
    probe_qemu_sandbox_support,
    probe_qemu_version,
    probe_unshare_support,
    qemu_system_binary,
    qemu_unrunnable_error,
)
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType
from exec_sandbox.vm_working_directory import VmWorkingDirectory

logger = get_logger(__name__)

# x86 guests see one fixed CPU, not the host's, so a saved instance loads on
# any host that has these flags: an AVX2-level (x86-64-v3) flag list on a
# versioned base model (QEMU has no x86-64-vN model). With `enforce`, QEMU
# refuses to start on a host that lacks a flag, instead of hiding it from a
# guest that already uses it (an instance saved on a newer host). Hosts older
# than Haswell / Zen 1 cannot run it. +hypervisor lets the guest find KVM
# (kvm-clock, see the kernel command line). -svm,-vmx: no nested virtualization
# (CVE-2024-50115).
# rdrand/rdseed: entropy without a host round trip (TCG emulates both).
# aes/pclmulqdq: AES-NI and carry-less multiply (every AVX2-level host has both).
# Without them the guest runs AES-GCM in software: TLS 1.3 measured 122 -> 1466 MB/s
# on KVM, 12.8 -> 56.5 MB/s on TCG (QEMU maps both to host instructions).
# xsaveopt: every AVX2-level host has it, and under TCG a guest with xsave but
# no xsaveopt hangs before the first console line (measured, QEMU 11.1.3).
_X86_CPU_FLAGS = (
    "qemu64-v1,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+abm,+movbe,+xsave,+xsaveopt,+cx16,+lahf_lm,"
    "+popcnt,+sse4.1,+sse4.2,+ssse3,+aes,+pclmulqdq,+rdrand,+rdseed,+hypervisor,-svm,-vmx"
)
# Hypercall paravirt features that named models leave off (KVM only).
_X86_KVM_PV_FLAGS = "+kvm-pv-ipi,+kvm-pv-tlb-flush,+kvm-pv-sched-yield,+kvm-poll-control,+kvm-asyncpf-int"


def _qmp_monitor(chardev: str, *, qemu_version: tuple[int, int, int] | None) -> list[str]:
    """Arguments of a QMP monitor on `chardev`: the monitor-qmp object from QEMU 11.1, which
    deprecates `-mon` (a warning at each start), else `-mon`."""
    if qemu_version is not None and qemu_version >= (11, 1, 0):
        return ["-object", f"monitor-qmp,id=mon-{chardev},chardev={chardev}"]
    return ["-mon", f"chardev={chardev},mode=control"]


def _x86_cpu(accel_type: AccelType, *, tsc_deadline: bool) -> str:
    """-cpu value for x86 guests (the fixed model above). tsc-deadline follows the pit/pic/rtc=off machine choice."""
    parts = [_X86_CPU_FLAGS]
    if accel_type == AccelType.KVM:
        parts.append(_X86_KVM_PV_FLAGS)
    if tsc_deadline:
        parts.append("+tsc-deadline")
    parts += [f"vendor={detect_host_x86_vendor()}", "enforce"]
    return ",".join(parts)


async def guest_cpu_model(arch: HostArch, accel_type: AccelType) -> str:
    """-cpu value of a VM on this host. Part of the L1 key and of a saved instance's host facts:
    a restore needs the CPU it was saved on.

    ARM64 hardware accel: "host".
    ARM64 TCG: "neoverse-v1,pauth=off" (ARMv8.4 with SVE; the model and its pauth
    property are in QEMU 9.0.0, below the 9.0.4 floor of ARM TCG): the newest model
    without BTI (the ARMv9 models have it, and QEMU cannot turn it off), with pointer
    authentication off. These two features cost most to emulate.
    With PAC every kernel function signs and checks its return address,
    a helper call each even with the fast impdef algorithm (QARMA5 costs ~50% of TCG
    cycles). With BTI QEMU checks the target page on every indirect branch, whether the
    guest uses BTI or not: an interpreter dispatches each opcode through one. Measured
    on macOS, paired cold VMs: "max,pauth=off" against "max,pauth-impdef=on": VM start
    -10%, first exec -10% (Python and JavaScript), syscall and stat -24%, file
    create+write+unlink -38%, an object-heavy JavaScript run -40%; then "neoverse-n1"
    against "max,pauth=off" (20 samples each): Python compute -23% and its loop step
    -28% (faster in 10 of 10 rounds), start -8%, first exec and file I/O within noise;
    then "neoverse-v1,pauth=off" against "neoverse-n1" (20 samples each): start -6% (10
    of 10 rounds), JavaScript first exec -9%, Python loop step and syscall -8%, file
    create+write+unlink -11%, the rest equal. The model drops MOPS and MTE (no measured
    cost). PAC and BTI are hardenings inside the guest; TCG is no isolation boundary
    anyway (security note in build_qemu_cmd, at the TCG accel), and HVF/KVM keep them
    ("host"). A model with MOPS also has BTI (ARMv8.8 > v8.5).
    x86 (every accel): the fixed AVX2-level model, see _x86_cpu.
    NOTE (non-adoption): ARM `max,sve=off,sme=off` (individually maskable on `max`
    since QEMU 11.0) was measured with no change (boot and REPLs never use SVE/SME).
    See: https://www.qemu.org/docs/master/system/arm/cpu-features.html
    See: https://www.qemu.org/docs/master/system/i386/cpu.html
    See: https://gitlab.com/qemu-project/qemu/-/issues/844
    """
    if arch == HostArch.X86_64:
        # TCG never provides TSC_DEADLINE; KVM/HVF may lack it in nested VMs
        tsc_deadline = accel_type in (AccelType.KVM, AccelType.HVF) and await check_tsc_deadline()
        return _x86_cpu(accel_type, tsc_deadline=tsc_deadline)
    return "host" if accel_type in (AccelType.HVF, AccelType.KVM) else "neoverse-v1,pauth=off"


async def build_qemu_cmd(  # noqa: PLR0912, PLR0915
    settings: Settings,
    arch: HostArch,
    vm_id: str,
    workdir: VmWorkingDirectory,
    spec: VmSpec,
    *,
    expose_ports: list[ExposedPort] | None,
    direct_write: bool,
    debug_boot: bool,
    snapshot_drive: str | None,
    defer_incoming: bool,
    qmp_fd: int | None,
    guest_port_fds: tuple[int, int] | None,
    mem_path: str | None,
    mem_path_share: bool,
) -> list[str]:
    """Build QEMU command for microVM execution.

    Args:
        settings: Service configuration (paths, limits, etc.)
        arch: Host CPU architecture
        vm_id: Unique VM identifier
        workdir: VM working directory: overlay, socket and home disk paths
        spec: Creation settings (memory, vCPUs, network access, ...)
        expose_ports: Ports to expose from guest to host, host ports resolved.
            When set without allow_network, uses gvproxy with
            BlockAllOutbound (Mode 1). When set with allow_network, uses gvproxy
            with outbound filtering (Mode 2). In both modes gvproxy sets up the
            forwards at its start (-port-forward).
        direct_write: Mount rootfs read-write (for snapshot creation).
        debug_boot: Enable verbose kernel/init boot logging. When True,
            sets loglevel=7, printk.devkmsg=on, init.quiet=0 for full
            boot diagnostics.
        snapshot_drive: Path to ext4 qcow2 for snapshot overlay (serial=snap).
            When set, tiny-init discovers drives by serial number and mounts
            EROFS base + ext4 snapshot via overlayfs. For snapshot creation
            (direct_write=True), the ext4 drive is writable; for usage, read-only.
        defer_incoming: Start QEMU with `-incoming defer` for a restore (L1
            template or saved instance). The VM starts paused, waiting for a
            migration stream via QMP.
        qmp_fd: File descriptor of a pre-created listening socket for QMP
            (socket activation pattern). When set, uses ``-chardev socket,fd=N,...``
            + a QMP monitor on it (``-object monitor-qmp``; ``-mon`` before QEMU 11.1) instead of the ``-qmp`` shorthand
            (which does not support ``fd=``). None uses the
            path-based ``-qmp`` syntax (QEMU makes the socket).
        guest_port_fds: File descriptors of pre-created listening sockets for the
            cmd and event ports (socket activation, as qmp_fd). None:
            QEMU makes the sockets at their paths.
        mem_path: Path to file-backed RAM for VM Templating (COW sharing).
            Template save uses share=on (QEMU writes RAM to file).
            Template restore uses share=off (MAP_PRIVATE = COW from file).
            None uses anonymous memory (memory-backend-ram).
        mem_path_share: When True, memory-backend-file uses share=on (template
            save). When False, uses share=off,readonly=on,rom=off (clone restore,
            MAP_PRIVATE COW). Only meaningful when mem_path is set.

    Returns:
        QEMU command as list of strings

    Raises:
        ConfigError: HOST_DEPENDENCY: this host cannot run the QEMU binary
            (probe_qemu_runnable), or ARM64 TCG with a QEMU below 9.0.4
        TransientError: QEMU_CRASHED: the QEMU probe exec hit a process, memory or
            file-handle limit (qemu_unrunnable_error)
    """
    # Determine QEMU binary, machine type, and kernel based on architecture
    is_macos = detect_host_os() == HostOS.MACOS

    # Detect hardware acceleration type (centralized in detect_accel_type)
    accel_type = await detect_accel_type(force_emulation=settings.force_emulation)

    # QEMU version gates several flags below (machine props, netdev reconnect,
    # -run-with). Probe once up front; the result is cached in system_probes.
    qemu_version = await probe_qemu_version()
    if (unrunnable := await probe_qemu_runnable()) is not None:  # also without VmManager.start
        raise qemu_unrunnable_error(unrunnable) from unrunnable[1]
    logger.info(
        "Hardware acceleration detection",
        extra={"vm_id": vm_id, "accel_type": accel_type.value, "is_macos": is_macos},
    )

    # Build accelerator string for QEMU
    if accel_type == AccelType.HVF:
        # ARM64, QEMU 11.1+: QEMU's own GIC, not Apple's (kernel-irqchip, on by default for
        # virt-11.1). With Apple's GIC the guest's WFI waits inside hv_vcpu_run, and a vCPU
        # kick (hv_vcpus_exit) that lands while the vCPU returns for an MMIO exit is lost: the
        # HVF loop (hvf_arch_vcpu_exec) re-enters the guest without checking for a stop, and
        # pause_all_vcpus waits with no timeout. QMP stop, quit and SIGTERM then hung for 10 s
        # or more (6 of 6 trials of a stop loop; 0 in 12000 stops with QEMU's GIC, the only
        # mode before 11.1). Cost per idle VM: about 25 context switches/s against 10-16 with
        # Apple's GIC (the guest's timer interrupts exit to QEMU); no exec latency change.
        own_gic = arch == HostArch.AARCH64 and qemu_version is not None and qemu_version >= (11, 1, 0)
        accel = "hvf,kernel-irqchip=off" if own_gic else "hvf"
    elif accel_type == AccelType.KVM:
        accel = "kvm"
    else:
        # TCG software emulation fallback (~5-8x slower than KVM/HVF)
        #
        # NOTE (security envelope): QEMU's security policy grants the
        # "virtualization use case" isolation guarantee only to microvm (x86) /
        # virt (aarch64) *with* a hardware accelerator (KVM/HVF). The TCG path is
        # explicitly NOT security-supported ("must not rely on QEMU for
        # isolation"). It is a functional fallback for CI / trusted workloads
        # only; prod isolation always runs on KVM or HVF.
        # See: https://qemu-project.gitlab.io/qemu/system/security.html
        #
        # thread: multi (MTTCG: one host thread per vCPU) when the VM has more than one
        # vCPU, else single (one vCPU runs the same either way). MTTCG adds no
        # translation threads: measured 9-11 QEMU threads in both modes at 1 and 2 vCPUs
        # (the I/O worker pool varies more), and with 2 vCPUs two CPU-bound processes
        # took 35% less time (EC2 x86 TCG, 5.1-6.2 s -> 3.6-4.0 s), the start 4% less.
        # See: https://www.qemu.org/docs/master/devel/multi-thread-tcg.html
        #
        # tb-size: the translation cache cap, which the cgroup limit and the admission
        # budget count (constants.DEFAULT_TCG_TB_CACHE_SIZE_MB: rationale, measurements).
        accel = (
            f"tcg,thread={'multi' if spec.cpu_cores > 1 else 'single'},tb-size={constants.DEFAULT_TCG_TB_CACHE_SIZE_MB}"
        )
        logger.warning(
            "Using TCG software emulation (slow) - KVM/HVF not available",
            extra={"vm_id": vm_id, "accel": accel},
        )

        # ARM64 TCG: QEMU < 9.0.4 has a deterministic crash in regime_is_user()
        # (target/arm/internals.h). The E10 mmuidx (normal EL1&0 translation regime)
        # falls through to g_assert_not_reached() after `default: return false`,
        # crashing every ARM64 kernel↔userspace transition. Fixed upstream in
        # commit 1505b651fdbd (Peter Maydell). Not backported to Ubuntu 24.04's
        # QEMU 8.2.2. Retries are pointless — the crash is deterministic.
        if arch == HostArch.AARCH64 and qemu_version is not None and qemu_version < (9, 0, 4):
            version_str = ".".join(str(v) for v in qemu_version)
            raise ConfigError(
                f"QEMU {version_str} has a deterministic ARM64 TCG crash in "
                f"regime_is_user() (target/arm/internals.h). Every kernel↔userspace "
                f"transition triggers g_assert_not_reached(). "
                f"Upgrade to QEMU >= 9.0.4 to fix this. "
                f"See: https://gitlab.com/qemu-project/qemu/-/commit/1505b651fdbd",
                code=ErrorCode.HOST_DEPENDENCY,
                context=None,
                hint=None,
                instance_id=None,
            )

    tsc_available = False  # x86 only: set below from the TSC_DEADLINE probe
    if arch == HostArch.AARCH64:
        # highmem=off: Keep all RAM below 4GB for simpler memory mapping (faster boot)
        # gic-version=3: Explicit GIC version for TCG (ITS not modeled in TCG)
        # virtualization=off: Disable nested virt emulation (not needed, faster TCG)
        # its=off: Disable GICv3 ITS (Interrupt Translation Service) — we use virtio-mmio
        #   (not PCI), so MSI-X translation is unused; skips kernel ITS init (1-3ms)
        # dtb-randomness=off: Skip writing random seeds to DTB — redundant with
        #   random.trust_cpu (the kernel default since Linux 6.2; not on the command line)
        # mem-merge: KSM page dedup (Linux only; macOS HVF does not support it)
        mem_merge = "off" if is_macos else "on"
        # virtio-mmio-transports=8: cap the number of virtio-mmio transport slots
        #   (QEMU virt board default is 32). We instantiate at most 6 virtio-mmio
        #   devices (blk-base + optional blk-snap + blk-home + serial + rng +
        #   optional net); 8 leaves headroom. Fewer empty slots means the guest's
        #   virtio_mmio driver skips ~24 spurious MMIO magic/device-id probes at
        #   boot and the DTB is smaller. Keep >= the live device count or QEMU aborts.
        #   Machine property added in QEMU 11.0 — gated to avoid aborting older
        #   local installs. See docs/system/arm/virt.rst.
        mmio_transports = ",virtio-mmio-transports=8" if qemu_version is not None and qemu_version >= (11, 0, 0) else ""
        machine_type = (
            f"virt,virtualization=off,highmem=off,gic-version=3,its=off,dtb-randomness=off,mem-merge={mem_merge}{mmio_transports}"
            + ("" if is_macos else ",dump-guest-core=off")
        )
    else:
        # Machine type: microvm for all acceleration modes (KVM, HVF, TCG).
        # microvm provides direct kernel boot via qboot (acpi=off), skipping
        # SeaBIOS/iPXE. Under TCG this means ~2s boot vs 30+s with pc (i440FX).
        # See: https://www.qemu.org/docs/master/system/i386/microvm.html
        # See: https://www.kraxel.org/blog/2020/10/qemu-microvm-acpi/

        # Determine if legacy timer devices (PIT/PIC/RTC) can be disabled.
        # pit=off and pic=off require TSC_DEADLINE CPU feature (LAPIC timer
        # replaces PIT as the timer source). Without TSC_DEADLINE, PIT/PIC
        # must remain enabled or the kernel hangs (no timer/interrupt source).
        # TCG never provides TSC_DEADLINE; KVM/HVF may lack it in nested VMs.
        if accel_type in (AccelType.KVM, AccelType.HVF):
            tsc_available = await check_tsc_deadline()
            if not tsc_available:
                logger.info(
                    "TSC_DEADLINE not available, keeping legacy timer devices (PIT/PIC/RTC)",
                    extra={"vm_id": vm_id},
                )
        else:
            logger.info(
                "Using microvm with TCG emulation (hardware virtualization not available)",
                extra={"vm_id": vm_id, "accel": accel},
            )

        # Build machine_type from components
        parts = ["microvm", "acpi=off", "x-option-roms=off"]
        if tsc_available:
            parts.extend(["pit=off", "pic=off", "rtc=off"])
        # mem-merge: KSM page dedup (Linux only; macOS HVF does not support it)
        parts.extend(["isa-serial=off", f"mem-merge={'off' if is_macos else 'on'}"])
        if not is_macos:
            parts.append("dump-guest-core=off")
        machine_type = ",".join(parts)

    # The configured QEMU path when it exists (e.g. CI build-from-source), else the bare
    # name from PATH: the binary the version and accelerator probes ran.
    qemu_bin = qemu_system_binary(settings, arch=arch)

    # One VM shape, so every VM can be saved and loaded: guest RAM is always the
    # explicit backend "mem0", so the RAM block has one name in every saved state
    # and any VM's state loads into any other VM. Anonymous RAM for cold boots
    # and loads; a file for L1 templates (VM Templating, COW sharing across VMs):
    # template save share=on (QEMU writes guest RAM to the file via MAP_SHARED),
    # template restore share=off,readonly=on,rom=off (MAP_PRIVATE = COW).
    # Ref: https://www.qemu.org/docs/master/system/vm-templating.html
    if mem_path is None:
        backend_opts = f"memory-backend-ram,id=mem0,size={spec.memory_mb}M"
    elif mem_path_share:
        # Template save: share=on (QEMU writes guest RAM to file via MAP_SHARED)
        backend_opts = f"memory-backend-file,id=mem0,size={spec.memory_mb}M,mem-path={mem_path},share=on"
    else:
        # Template restore: share=off (MAP_PRIVATE = COW), readonly=on (open
        # file O_RDONLY for safety), rom=off (don't mark memory as ROM — guest
        # must be able to write).
        # Ref: https://www.qemu.org/docs/master/system/vm-templating.html
        backend_opts = (
            f"memory-backend-file,id=mem0,size={spec.memory_mb}M,mem-path={mem_path},share=off,readonly=on,rom=off"
        )
    mem_backend_args = ["-object", backend_opts]
    machine_type += ",memory-backend=mem0"

    # Auto-discover kernel and initramfs based on architecture
    # Note: existence validated by validate_kernel_initramfs (VmManager.start, and
    # VmManager._setup_vm_infra at each create and restore)
    # Prefer uncompressed vmlinux for PVH direct boot (x86_64 only, ~50ms faster)
    # QEMU auto-detects PVH ELF note and skips kernel decompression
    images = GuestImages.from_settings(settings, arch=arch)
    kernel_path, initramfs_path = images.kernel, images.initramfs

    # Layer 5: Linux namespaces (optional - requires root or CAP_SYS_ADMIN: probe_unshare_support)
    # QEMU always gets its own network namespace, also with network access: QEMU
    # needs no host network. Guest traffic goes through gvproxy over a filesystem
    # Unix socket, which a namespace does not cut. QEMU itself then reaches no
    # host network.
    cmd: list[str] = []
    if detect_host_os() != HostOS.MACOS and await probe_unshare_support():
        # --kill-child (this-user path only): QEMU, pid 1 of the new namespace (so every process
        # in it), gets SIGKILL when unshare ends; unshare ends with this process
        # (start_managed_process, end_with_this_process). On the qemu-vm path pid 1 is sudo: the exec of a
        # setuid program drops the parent-death signal, and the VM cgroup ends that QEMU.
        cmd.extend(["unshare", "--pid", "--net", "--mount", "--uts", "--ipc", "--fork", "--kill-child=SIGKILL", "--"])

    # Build QEMU command arguments
    # Determine if we're using microvm (requires -nodefaults to avoid BIOS fallback)
    is_microvm = "microvm" in machine_type

    # Virtio transport: always MMIO (the virtio-*-device models). Both microvm (x86) and
    # virt (ARM64) use virtio-mmio. The kernel has CONFIG_VIRTIO_MMIO=y built-in
    # and CONFIG_MODULES is not set — PCI devices would cause boot hang.

    qemu_args = [qemu_bin]

    # Set VM name for process identification (visible in ps aux, used by hwaccel test)
    # Format: guest=vm_id - the vm_id includes tenant, task, and uuid for uniqueness
    qemu_args.extend(["-name", f"guest={vm_id}"])

    # CRITICAL: -nodefaults -no-user-config are required for microvm to avoid BIOS fallback
    # See: https://www.qemu.org/docs/master/system/i386/microvm.html
    if is_microvm:
        qemu_args.extend(["-nodefaults", "-no-user-config"])

    # Console: ARM64 uses PL011 UART (ttyAMA0), x86 uses virtio-console (hvc0).
    # hvc0 is NOT reliable on ARM64 as /dev/console — virtio-serial isn't ready when
    # the kernel opens /dev/console, causing init to crash.
    # See: https://blog.memzero.de/toying-with-virtio/
    # The PL011 has no backend (-serial none below), so a debug boot on ARM64 also
    # names hvc0 first: the kernel prints there once hvc0 registers (about 30 ms in;
    # the earlier lines are not replayed), and the last console= (ttyAMA0) stays
    # /dev/console.
    if arch == HostArch.AARCH64:
        console_params = ("console=hvc0 " if debug_boot else "") + "console=ttyAMA0 loglevel=1"
    else:
        console_params = "console=hvc0 loglevel=1"

    qemu_args.extend(
        [
            "-accel",
            accel,
            "-cpu",
            await guest_cpu_model(arch, accel_type),
            *mem_backend_args,  # -object memory-backend-{ram,file},id=mem0
            "-M",
            machine_type,
            "-no-reboot",
            "-m",
            f"{spec.memory_mb}M",
            "-smp",
            f"{spec.cpu_cores},maxcpus={spec.cpu_cores}",  # Cap maxcpus to prevent QEMU allocating extra vCPU structs
            "-kernel",
            str(kernel_path),
            "-initrd",
            str(initramfs_path),
            "-append",
            # =============================================================
            # Kernel Command Line (runtime-only params)
            # =============================================================
            # Most boot optimizations are enforced at compile time via kernel
            # config (exec-sandbox.config). Only params with NO CONFIG equivalent
            # remain here. See exec-sandbox.config for the full CONFIG↔cmdline map.
            #
            # Not on this line (the kernel config or its defaults set them): init_on_free,
            #   scsi_mod.scan, audit, slab_nomerge, nomodule, preempt,
            #   noresume, raid, numa_balancing, i8042.*, random.trust_cpu,
            #   panic, rcupdate.rcu_expedited, edd, noautogroup, io_delay
            # =============================================================
            f"{console_params} root=/dev/vda rootflags=noatime rootfstype=erofs rootwait fsck.mode=skip reboot=t init=/init page_alloc.shuffle=0 swiotlb=0"
            # Disable page zeroing on alloc — KVM provides zero pages via shared zero
            # page, so guest-side zeroing is redundant for ephemeral VMs. Overrides
            # CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y at runtime. Saves 5-10% page faults.
            + " init_on_alloc=0"
            # THP: CONFIG_TRANSPARENT_HUGEPAGE=y enables EROFS large folios
            # (16-64KB per fault instead of 4KB), but transparent_hugepage=never
            # disables anonymous 2MB hugepages (no khugepaged, no compaction stalls).
            # File-backed large folios work regardless of this sysfs setting.
            # NOTE (non-adoption): keeping anon THP off is deliberate for density —
            # 2MB hugepages defeat free-page-reporting and KSM (both 4KB-granular)
            # and add compaction stalls; enabling them would hurt host-RAM-per-VM.
            + " transparent_hugepage=never"
            # Boot verbosity: debug_boot enables full kernel/init logging for diagnostics
            # Note: loglevel=7 overrides loglevel=1 set in console_params (kernel uses last occurrence)
            + (" loglevel=7" if debug_boot else " quiet loglevel=0")
            # Skip timer calibration — safe in virtualized env with reliable TSC (10-30ms)
            + " no_timer_check"
            # Revert to normal (lazy) RCU after boot. Expedited RCU uses IPIs and
            # per-CPU tracking for fast grace periods — useful during boot but wasteful
            # for steady-state single-execution VMs. Normal RCU has lower memory and CPU
            # overhead at the cost of ~10-30ms grace period latency (irrelevant for
            # ephemeral workloads).
            + " rcupdate.rcu_normal_after_boot=1"
            # /dev/kmsg access: on for debug, off for production
            + (" printk.devkmsg=on" if debug_boot else " printk.devkmsg=off")
            # Skip 8250 UART probing when ISA serial disabled (2-5ms, x86_64 only)
            + (" 8250.nr_uarts=0" if arch == HostArch.X86_64 else "")
            # =============================================================
            # Haltpoll cpuidle governor — KVM only
            # =============================================================
            # Polls in-guest for up to 200µs (guest_halt_poll_ns) before issuing
            # HLT, avoiding VM-exit cost on short idle windows.  Measured: 20%
            # latency reduction (sockperf), 4-71% FPS gain (Intel gaming study).
            #
            # Why KVM-only:
            #
            # 1. Governor registers unconditionally since kernel 6.x (the
            #    kvm_para_available() guard was removed from init_haltpoll() to
            #    allow bare-metal testing — LKML 2023-11 "governors/haltpoll:
            #    Drop kvm_para_available() check").  The cpuidle-haltpoll DRIVER
            #    still gates on kvm_para_available(), but the governor itself
            #    activates on any hypervisor when forced via cmdline.
            #
            # 2. On KVM the trade-off is favorable: each HLT VM-exit costs
            #    ~5-10µs, and the guest coordinates with the host via
            #    MSR_KVM_POLL_CONTROL to suppress redundant host-side halt
            #    polling.  On HVF neither mechanism exists — the 200µs
            #    busy-loop (cpu_relax → ARM64 yield, running at full speed
            #    in-guest) is pure overhead.
            #
            # 3. Measured impact on HVF (macOS ARM64, QEMU 10.2, kernel 6.18):
            #    ~65% host-CPU per idle vCPU.  Background wakeups (RCU
            #    callbacks, virtio interrupts)
            #    trigger ~3k poll cycles/sec x 200us = ~65% busy.  The
            #    adaptive algorithm keeps poll_limit_ns at max because
            #    wakeups keep landing inside the poll window.
            #
            # 4. On HVF/TCG we disable cpuidle entirely (cpuidle.off=1).
            #    ARM64 virt: no idle-states DT nodes, so only poll + WFI.
            #    x86 microvm: no ACPI, so no C-state tables, only poll + HLT.
            #    Any governor (menu, TEO) would always pick the idle instruction
            #    — the prediction algorithm is wasted overhead.  cpuidle.off=1
            #    bypasses the framework and calls the idle instruction (WFI on
            #    ARM64, HLT on x86) directly.  HVF properly blocks the vCPU
            #    thread via qemu_wait_io_event() → halt_cond, yielding <5%
            #    idle CPU.  TCG suspends TB execution on HLT.
            #
            # References:
            #   - LWN: cpuidle haltpoll driver & governor
            #     https://lwn.net/Articles/792618/
            #   - LKML: Drop kvm_para_available() from governor
            #     https://lkml.indiana.edu/hypermail/linux/kernel/2311.2/03791.html
            #   - QEMU HVF WFI idle-CPU issue
            #     https://gitlab.com/qemu-project/qemu/-/issues/959
            #   - Kernel cpuidle docs
            #     https://docs.kernel.org/admin-guide/pm/cpuidle.html
            # =============================================================
            + (" cpuidle.governor=haltpoll" if accel_type == AccelType.KVM else " cpuidle.off=1")
            # Skip deferred probe timeout (no hardware needing async probe, 0-5ms)
            + " deferred_probe_timeout=0"
            # Disable CPU speculative execution mitigations in the guest kernel.
            # VMSCAPE (CVE-2025-40300) must be mitigated by a patched/configured
            # HOST kernel; guest mitigations=off neither enables nor disables that
            # host defense (Linux does not apply VMSCAPE mitigation in guests).
            # Guest-side mitigations add CPU overhead and per-CPU state without
            # protecting a co-resident tenant (there is only one tenant per VM).
            # This accepts residual risk to the trusted guest-agent and guest
            # kernel from untrusted guest code.
            # nokaslr is deliberately omitted: the x86 base config has
            # CONFIG_RANDOMIZE_BASE=y, although this cmdline guard does not prove
            # runtime randomization on the direct-vmlinux PVH boot path. The arm64
            # base has CONFIG_RANDOMIZE_BASE=n; with mitigations=off its KPTI also
            # remains off. That arm64 density tradeoff is accepted here rather
            # than silently adding kpti=1 without performance evidence.
            # The cmdline token/omission is pinned by
            # test_vm_manager.py::TestKernelCmdlineMitigations.
            + " mitigations=off"
            # Limit possible CPUs to match -smp. Without this, QEMU machine types
            # advertise many possible CPUs (virt: up to 512, microvm: up to 256),
            # and the kernel allocates per-CPU data structures for ALL possible CPUs.
            # nr_cpus=N caps possible CPUs, saving ~50-200KB per unused CPU slot.
            + f" nr_cpus={spec.cpu_cores}"
            # Disable lockup detection — saves per-CPU watchdog threads and timers.
            # Ephemeral microVMs rely on host-side timeouts, not guest watchdogs.
            + " nowatchdog"
            # The guest's capped-zram safety guard uses a memory PSI stall
            # trigger. The merged Alpine base config builds PSI in but sets
            # CONFIG_PSI_DEFAULT_DISABLED=y (images/kernel/alpine-virt-*.config),
            # so PSI stays off unless this explicit token is present;
            # fail-closed zram setup verifies it.
            + " psi=1"
            # Disable cgroup controllers — minimal init with no systemd, no
            # resource isolation needed inside the VM. Saves slab caches and
            # per-CPU structures for each disabled controller.
            + " cgroup_disable=memory,cpu,blkio,pids"
            # Disable NUMA-aware workqueues — single-NUMA VM has no topology to
            # optimize for. Saves per-NUMA-node workqueue pool allocation.
            + " workqueue.disable_numa=1"
            + (" init.rw=1" if direct_write else "")
            + (" init.snap=1" if snapshot_drive else "")
            # Guest-agent log verbosity: init.quiet=0 un-gates log_info! macros
            + (" init.quiet=0" if debug_boot else " init.quiet=1")
            # x86 under KVM uses kvm-clock: KVM rewrites its scale after a load
            # on a host with another TSC rate; a forced TSC clocksource would run
            # at the wrong rate there. Without KVM (TCG, HVF) there is no kvm-clock:
            # pin the TSC and skip clocksource probing (5-10ms).
            + (" tsc=reliable clocksource=tsc" if arch == HostArch.X86_64 and accel_type != AccelType.KVM else ""),
        ]
    )

    # Platform-specific memory configuration
    # Note: -mem-prealloc tested for cold-start on HVF (pre-populates host pages)
    # but had NO effect — ARM VHE Stage-2 PTEs are still created lazily by hardware
    # on first guest access, regardless of host page presence.

    # Layer 3: Seccomp sandbox — requires CONFIG_SECCOMP at QEMU compile time.
    # Probe rather than assume: some runners (e.g. CodSpeed macro) may resolve
    # a system QEMU built without --enable-seccomp.
    if await probe_qemu_sandbox_support():
        qemu_args.extend(
            [
                "-sandbox",
                constants.QEMU_SANDBOX_POLICY,
            ]
        )

    # Determine AIO mode based on cached startup probes (qemu_version probed above)
    io_uring_available = await probe_io_uring_support()
    aio_mode = "io_uring" if io_uring_available else "threads"

    # The drives the VM writes (rootfs overlay, home disk, the layer of a snapshot build)
    # use O_DIRECT where it helps (probe_drive_direct_io: Linux, a disk filesystem that
    # takes it, not btrfs or ZFS). Buffered, their writes are dirty host page cache, charged to
    # the VM's cgroup on top of its guest RAM (which init_on_free fills at boot): a VM that
    # writes much, at run time or in a build, reached its cgroup limit hundreds of times
    # per GiB written, and a pandas build was OOM-killed. Flushes stay ignored
    # (no-flush), as with cache=unsafe. Elsewhere (macOS has no memory cgroup; tmpfs,
    # btrfs, ZFS, a filesystem without O_DIRECT): cache=unsafe.
    # An L1 template save (guest RAM in a shared file) keeps cache=unsafe: O_DIRECT reads
    # would land in file-backed shared pages; that VM lives briefly and writes little.
    async def written_drive_cache(folder: Path) -> str:
        direct = not mem_path_share and await probe_drive_direct_io(folder)
        return "cache.direct=on,cache.no-flush=on," if direct else "cache=unsafe,"

    # The overlay sits in the VM's own folder: the probe asks the folder of the VM folders,
    # the same filesystem, fixed for the process life (its answer is cached by folder)
    overlay_cache = await written_drive_cache(workdir.overlay_image.parent.parent)
    if overlay_cache.startswith("cache.direct=on"):
        # The shared read-only base stays in the host page cache: every VM reads it
        overlay_cache += "backing.cache.direct=off,"
    if not io_uring_available:
        logger.debug(
            "Using aio=threads (io_uring not available)",
            extra={"reason": "syscall_probe_failed", "vm_id": vm_id},
        )

    # IOThread — offloads block I/O completion (pread, qcow2 decompress) to a
    # dedicated AioContext. Keep it on Linux (KVM and TCG), where ioeventfd or
    # main-loop offload provides a measured benefit. macOS has no eventfd, and
    # with the IOThread a disk request can stall for good: QEMU sits idle (IOThread
    # in g_poll, workers waiting, vCPU halted) while the guest waits for it. HVF
    # showed post-boot read stalls; TCG stalled about 1 cold VM in 100 under load
    # (home disk mkfs/mount at boot, or a REPL page-in), 0 in 736 on the main loop,
    # at the same speed (paired rounds). Route macOS block I/O through the main
    # loop. Keep the same aio=threads pool cap in both configurations so
    # the platform conditional changes AioContext placement, not concurrency.
    use_dedicated_iothread = not is_macos
    iothread_id = f"iothread0-{vm_id}"
    # Cap I/O thread pool to 8 (QEMU default is 64).  With single-queue
    # virtio-blk on qcow2, 8 is sufficient.  When aio=io_uring is active
    # the pool is unused (no-op).  When aio=threads, this caps peak threads
    # from ~72 to ~16, reducing per-VM thread count and stack memory.
    if use_dedicated_iothread:
        qemu_args.extend(["-object", f"iothread,id={iothread_id},thread-pool-min=0,thread-pool-max=8"])
        device_iothread = f",iothread={iothread_id}"
    else:
        qemu_args.extend(["-object", "main-loop,id=main-loop0,thread-pool-min=0,thread-pool-max=8"])
        device_iothread = ""

    # Disk configuration (EROFS base drive, serial=base)
    # Uses qcow2 overlay backed by the EROFS base image
    #
    # NOTE (non-adoption): RWF_DSYNC FUA (QEMU 10.0 file-posix) is deliberately
    # unused. It only helps cache=writethrough/directsync; no-flush (cache=unsafe or
    # cache.no-flush=on) drops all flushes/FUA outright, which strictly dominates for
    # discard-on-exit overlays (a host crash loses in-flight overlay writes, but the
    # overlay is thrown away on VM exit). Switching to a flushing cache mode to gain FUA would be a large
    # regression here.
    qemu_args.extend(
        [
            "-drive",
            f"file={workdir.overlay_image},"
            f"format=qcow2,"
            f"if=none,"
            f"id=hd0,"
            f"{overlay_cache}"
            f"aio={aio_mode},"
            f"discard=unmap,"
            f"detect-zeroes=unmap,"
            f"werror=report,"
            f"rerror=report,"
            f"copy-on-read=off,"
            f"bps={constants.DISK_BPS_LIMIT},"
            f"bps_max={constants.DISK_BPS_BURST},"
            f"iops={constants.DISK_IOPS_LIMIT},"
            f"iops_max={constants.DISK_IOPS_BURST},"
            # Disable QEMU file locking to allow concurrent VMs sharing same backing file.
            # On Linux, QEMU uses OFD (Open File Descriptor) locks which cause "Failed to
            # get shared write lock" errors when multiple VMs access the same base image.
            # macOS doesn't enforce OFD locks, so this issue only manifests on Linux/CI.
            # Safe because: (1) each VM has unique overlay, (2) base image is read-only.
            f"file.locking=off",
        ]
    )

    # Block device — serial=base: stable identifier for tiny-init to find the EROFS
    # rootfs drive, regardless of /dev/vdX assignment. ARM virt (MMIO) enumerates
    # virtio devices in reverse declaration order, so /dev/vda != first -device on ARM.
    # event_idx=off reduces interrupt coalescing overhead during boot.
    #
    # NOTE (non-adoption): num-queues=1 is deliberate. virtio-blk multiqueue /
    # iothread-vq-mapping (QEMU 9.0) only pays off with many vCPUs and an
    # I/O-bound guest saturating a single iothread. Our VMs have 1-2 vCPUs, a
    # read-only page-cache-hot EROFS base and a tiny no-flush overlay — never
    # iothread-CPU-bound. macOS routes block I/O to the main loop (above) and TCG has
    # at most one thread per vCPU, so extra queues buy nothing here.
    qs = constants.VIRTIO_QUEUE_SIZE
    qemu_args.extend(
        [
            "-device",
            f"virtio-blk-device,drive=hd0,serial=base{device_iothread},num-queues=1,queue-size={qs},event_idx=off",
        ]
    )

    # Snapshot overlay drive — ext4 layer for overlayfs merge with EROFS base.
    # Presented as a second virtio-blk device; tiny-init detects it by serial=snap
    # and sets up overlayfs automatically (rw for snapshot creation, ro for usage).
    # Outside snapshot creation the shared L2 file is opened read-only, so a guest
    # can never write to a cache entry that other VMs use at the same time.
    if snapshot_drive:
        # Written only by a snapshot build (written_drive_cache); else shared and read-only
        snapshot_mode = (
            await written_drive_cache(Path(snapshot_drive).parent) if direct_write else "read-only=on,cache=unsafe,"
        )
        qemu_args.extend(
            [
                "-drive",
                f"file={snapshot_drive},"
                f"format=qcow2,"
                f"if=none,"
                f"id=hd1,"
                f"{snapshot_mode}"
                f"aio={aio_mode},"
                f"discard=unmap,"
                f"detect-zeroes=unmap,"
                f"werror=report,"
                f"rerror=report,"
                f"file.locking=off",
            ]
        )
        qemu_args.extend(
            [
                "-device",
                f"virtio-blk-device,drive=hd1,serial=snap{device_iothread},num-queues=1,queue-size={qs},event_idx=off",
            ]
        )

    # Home disk (serial=home): raw sparse file of spec.disk_mb, private to this
    # VM, formatted by tiny-init and mounted on /home/user. Every VM has one (one
    # VM shape). Same I/O limits as the base drive; default file locking (unlike
    # the shared base and L2 files, nothing else opens it).
    # werror=enospc: a write that finds the host disk full pauses the VM (other
    # write errors still go to the guest). The guest never gets the error (no
    # EIO, no ext4 error), and BLOCK_IO_ERROR carries nospace=true only for a
    # drive with this policy (with werror=report QEMU leaves it out). QemuVM's
    # event watch then fails the calls at once (TransientError LOCAL_DISK).
    qemu_args.extend(
        [
            "-drive",
            f"file={workdir.home_disk},"
            f"format=raw,"
            f"if=none,"
            f"id=hd2,"
            f"{await written_drive_cache(workdir.home_disk.parent)}"
            f"aio={aio_mode},"
            f"discard=unmap,"
            f"detect-zeroes=unmap,"
            f"werror=enospc,"
            f"rerror=report,"
            f"bps={constants.DISK_BPS_LIMIT},"
            f"bps_max={constants.DISK_BPS_BURST},"
            f"iops={constants.DISK_IOPS_LIMIT},"
            f"iops_max={constants.DISK_IOPS_BURST}",
            "-device",
            f"virtio-blk-device,drive=hd2,serial=home{device_iothread},num-queues=1,queue-size={qs},event_idx=off",
        ]
    )

    # Display/console configuration
    # -nographic: headless mode
    # -monitor none: disable QEMU monitor (it uses stdio by default with -nographic,
    #   which conflicts with our -chardev stdio in environments without a proper TTY)
    qemu_args.extend(
        [
            "-nographic",
            "-monitor",
            "none",
        ]
    )

    # The guest port sockets: inherited listeners (guest_port_fds), else QEMU makes them
    if guest_port_fds is not None:
        cmd_socket, event_socket = (f"fd={fd}" for fd in guest_port_fds)
    else:
        cmd_socket, event_socket = f"path={workdir.cmd_socket}", f"path={workdir.event_socket}"

    # Chardevs for guest agent communication AND kernel console (hvc0).
    # All machine types need explicit chardev setup:
    #   microvm: -nodefaults suppresses everything, so all devices must be declared
    #   ARM64 virt: no -nodefaults, but we still need named ports for guest-agent
    qemu_args.extend(
        [
            # Chardevs for communication channels
            # server=on: QEMU listens (on the inherited socket, else one it creates)
            # wait=off: QEMU starts VM immediately without waiting for client connection
            # Note: Socket permissions (umask 007 as qemu-vm) come from QEMU_VM_SUDO.
            # The guest agent retries connection so timing is handled.
            "-chardev",
            f"socket,id=cmd0,{cmd_socket},server=on,wait=off",
            "-chardev",
            f"socket,id=event0,{event_socket},server=on,wait=off",
            # Chardev for console output - connected to virtconsole (hvc0)
            # No mux (the -chardev default): -monitor none leaves no QEMU monitor
            # to share stdio with, so no mux input layer or per-chardev state.
            "-chardev",
            "stdio,id=virtiocon0,signal=off",
        ]
    )

    # Disable default serial to prevent "cannot use stdio by multiple character devices"
    # ARM64 virt (without -nodefaults) has a default PL011 UART that would grab stdio.
    # x86 microvm already suppresses ISA serial via isa-serial=off + -nodefaults,
    # so -serial none is redundant there but harmless.
    qemu_args.extend(["-serial", "none"])

    # =============================================================
    # Virtio-Serial Device Configuration
    # =============================================================
    # Virtio-serial provides guest agent communication channels (cmd/event ports).
    # All paths use virtio-console (hvc0) with ISA serial disabled:
    #   - virtconsole device created for hvc0 (the kernel console on x86; tiny-init
    #     writes to hvc0 first on both architectures)
    #   - 3 ports: virtconsole (nr=0) + cmd (nr=1) + event (nr=2)
    #   - ISA serial disabled via isa-serial=off in machine type
    #
    # See: https://bugs.launchpad.net/qemu/+bug/1639791 (early virtio console lost)
    # See: https://gist.github.com/mcastelino/aa118275991d4f561ee22dc915b9345f
    # =============================================================
    # NOTE (non-adoption): guest-agent transport stays on virtio-serial. vhost-vsock
    # was evaluated as an alternative; no QEMU 10.1→11.0 reliability/perf change
    # shifts the tradeoff, and virtio-serial's named ports (cmd/event) map cleanly
    # onto the existing guest-agent protocol.
    qemu_args.extend(
        [
            "-device",
            "virtio-serial-device,max_ports=3",
            # hvc0 console device - must be nr=0 to be hvc0
            "-device",
            "virtconsole,chardev=virtiocon0,nr=0",
            # An id makes QEMU send VSERPORT_CHANGE when the guest closes the port
            # (QemuVM._on_qmp_event fails the calls in flight on it).
            "-device",
            f"virtserialport,chardev=cmd0,name=org.dualeai.cmd,nr=1,id={constants.GUEST_PORT_IDS[0]}",
            "-device",
            f"virtserialport,chardev=event0,name=org.dualeai.event,nr=2,id={constants.GUEST_PORT_IDS[1]}",
        ]
    )

    # No virtio-balloon: every VM has the same device set, so every VM can
    # be saved and loaded. Idle guest memory returns to the host through
    # memory.reclaim + zswap; init_on_free zeroes freed guest pages.

    # virtio-rng: continuous host→guest entropy via /dev/urandom backend.
    # Feeds the kernel input pool so RNDRESEEDCRNG ioctl (guest-agent) has fresh
    # entropy to reseed from after L1 snapshot restore. Without this, the input
    # pool after restore contains only stale (cloned) entropy.
    # max-bytes + period: rate-limit to 256B/1000ms (0.25KB/s). Ephemeral VMs
    # only need initial entropy reseed; continuous high-rate entropy is wasteful.
    qemu_args.extend(
        [
            "-object",
            "rng-random,id=rng0,filename=/dev/urandom",
            "-device",
            "virtio-rng-device,rng=rng0,max-bytes=256,period=1000",
        ]
    )

    # =============================================================
    # Network Configuration: Three Modes + No-Network
    # =============================================================
    # All modes use gvproxy with socket networking for fast boot (~300ms).
    # SLIRP is not used: it is ~40x slower (~11s boot).
    #
    # Mode 0: No network (default)
    #   - Explicit "-nic none" suppresses QEMU's default NIC
    #   - Without this, machine types without -nodefaults (ARM64 virt)
    #     create a default NIC; the guest-agent then requires a reachable
    #     gvproxy and marks the network terminally failed, so every
    #     ExecuteCode/InstallPackages on the VM is rejected (fail-closed)
    #   - microvm already uses -nodefaults so -nic none is redundant
    #     but harmless there
    #   - See: https://www.qemu.org/docs/master/system/qemu-manpage.html
    #
    # Mode 1: Port forwarding only (expose_ports + no allow_network)
    #   - Uses gvproxy with BlockAllOutbound (no internet)
    #   - Port forwarding handled by gvproxy at startup
    #
    # Mode 2: Port forwarding with internet (expose_ports + allow_network)
    #   - Uses gvproxy with OutboundAllow filtering (DNS + TLS)
    #   - Port forwarding handled by gvproxy at startup
    #
    # Mode 3: Internet only (allow_network, no expose_ports)
    #   - Standard gvproxy configuration with OutboundAllow filtering
    #
    # =============================================================

    # NOTE (non-adoption): the in-tree passt netdev (QEMU 10.1) is not used. It
    # lacks gvproxy's outbound filtering (allowed domains, outbound block) that
    # Modes 1-3 above depend on, so it is not a drop-in replacement.
    needs_network = spec.allow_network or bool(expose_ports)
    if needs_network:
        # All modes use socket networking to gvproxy (fast ~300ms boot)
        # Build netdev options with reconnect for socket resilience
        # Helps recover from transient gvproxy disconnections (DNS failures, socket EOF)
        netdev_opts = f"stream,id=net0,addr.type=unix,addr.path={workdir.gvproxy_socket}"

        # Add reconnect parameter (version-dependent)
        # - QEMU 9.2+: reconnect-ms (milliseconds), reconnect removed in 10.0
        # - QEMU 8.0-9.1: reconnect (seconds), minimum 1s
        if qemu_version is not None and qemu_version >= (9, 2, 0):
            netdev_opts += ",reconnect-ms=250"  # 250ms - balanced recovery
        elif qemu_version is not None and qemu_version >= (8, 0, 0):
            netdev_opts += ",reconnect=1"  # 1s minimum (integer-only param)

        mode_desc = (
            "Mode 1 (port-forward only, no internet)"
            if expose_ports and not spec.allow_network
            else "Mode 2 (port-forward + internet)"
            if expose_ports and spec.allow_network
            else "Mode 3 (internet only)"
        )
        logger.info(
            f"Configuring socket networking via gvproxy ({mode_desc})",
            extra={
                "vm_id": vm_id,
                "expose_ports": [(p.internal, p.external) for p in expose_ports] if expose_ports else None,
                "allow_network": spec.allow_network,
            },
        )

        qemu_args.extend(
            [
                "-netdev",
                netdev_opts,
                "-device",
                "virtio-net-device,netdev=net0,mq=off,csum=off,gso=off,host_tso4=off,host_tso6=off,mrg_rxbuf=off,ctrl_rx=off,guest_announce=off",
            ]
        )
    else:
        # Suppress QEMU's default NIC.  Without this, machine types that don't
        # use -nodefaults (ARM64 virt) create a default virtio-net device; the
        # guest-agent then detects eth0 and requires a reachable gvproxy
        # (verify_gvproxy), which fails without one — the network state goes
        # terminal-failed and every ExecuteCode/InstallPackages on the VM is
        # rejected by the untrusted-readiness gate. A NIC that the spec did
        # not ask for fails the VM for good, not merely slows it.
        qemu_args.extend(["-nic", "none"])

    # QMP (QEMU Monitor Protocol) socket for VM control operations
    #
    # Two modes:
    #   1. Socket activation (qmp_fd != None): Parent pre-creates a listening
    #      Unix socket and passes the fd to QEMU via pass_fds. QEMU's
    #      chardev/char-socket.c accepts fd= with server=on — it skips
    #      bind()/listen() and uses the inherited fd directly. This eliminates
    #      the TOCTOU race between wait_for_socket() probe-close and
    #      MigrationClient's real connect on single-client QMP chardev.
    #      The -qmp shorthand does NOT support fd= — must use the explicit
    #      -chardev + -mon pair (same as libvirt since QEMU 2.12).
    #      Refs:
    #        - QEMU chardev fd= support: chardev/char-socket.c tcp_chr_new_server()
    #        - libvirt fd-passing: https://libvirt.org/news.html (2018, QEMU 2.12)
    #        - Kata Containers: uses fd=3 for QMP monitor
    #
    #   2. Path mode (qmp_fd == None): QEMU creates and binds the socket
    #      itself. Used by every cold boot (create_vm), and by a restore when
    #      fd inheritance is impossible (sudo wraps the command — sudo closes
    #      all fds >= 3 by default).
    if qmp_fd is not None:
        qemu_args.extend(
            [
                "-chardev",
                f"socket,id=qmp0,fd={qmp_fd},server=on,wait=off",
                *_qmp_monitor("qmp0", qemu_version=qemu_version),
            ]
        )
    else:
        qemu_args.extend(["-qmp", f"unix:{workdir.qmp_socket},server=on,wait=off"])

    # Event-only QMP monitor: QemuVM.start_event_watch keeps one connection for the
    # VM's life (BLOCK_IO_ERROR, VSERPORT_CHANGE, SHUTDOWN). The monitor above serves
    # one client at a time, and saves and diagnostics need it. QEMU makes the socket
    # (as qemu-vm through sudo).
    qemu_args.extend(
        [
            "-chardev",
            f"socket,id=qmpev0,path={workdir.qmp_events_socket},server=on,wait=off",
            *_qmp_monitor("qmpev0", qemu_version=qemu_version),
        ]
    )

    # Restore (L1 memory snapshot or saved instance): QEMU starts paused, waiting for migration stream
    if defer_incoming:
        qemu_args.extend(["-incoming", "defer"])

    # exit-with-parent=on (QEMU 10.2+): QEMU exits when its parent does (PR_SET_PDEATHSIG
    # on Linux, kqueue on macOS/FreeBSD). On the this-user path only, this ends QEMU with
    # this process: its parent is this process, or unshare when namespaces work (unshare
    # ends with this process (start_managed_process, end_with_this_process), and its --kill-child ends
    # QEMU). For a QEMU that runs as qemu-vm, the parent is sudo, which drops the
    # parent-death signal (setuid) and outlives a killed host process: that QEMU ends
    # through its VM cgroup (PrivateRunDir.record_cgroup and the per-manager watcher).
    #
    # async-teardown=on (Linux only): fork a teardown clone that unmaps guest RAM
    # so the main QEMU can exit without blocking on munmap. Trims destroy_vm()
    # tail latency under mass pool recycle. macOS QEMU rejects this sub-option
    # (only chroot/exit-with-parent/user exist there), so it is Linux-gated.
    #
    # NOTE (non-adoption): -run-with chroot=/user= are not used. Both require a
    # privileged launcher (QEMU starts root, then drops), which conflicts with our
    # unprivileged model — sudo -u qemu-vm + fd-passing (socket-activated QMP).
    # Filesystem confinement is already provided by `unshare --mount` on Linux.
    if qemu_version is not None and qemu_version >= (10, 2, 0):
        run_with_opts = "exit-with-parent=on"
        if not is_macos:
            run_with_opts += ",async-teardown=on"
        qemu_args.extend(["-run-with", run_with_opts])

    # Run QEMU as qemu-vm when this process decided so (probe_qemu_as_qemu_vm: one
    # decision per process, checked end to end), else as the current user
    #
    # TODO (security; out of scope for now, by design): investigate and decide, with
    # upstream sources. Every VM's QEMU runs as the same qemu-vm uid, so a guest that
    # escapes into its QEMU can read and write every live VM's overlay and home disk
    # (opened by path: chown / ACL u:qemu-vm), e.g. change another tenant's state just
    # before that tenant saves its instance.
    # - QEMU requires an unprivileged uid (done) and states that a guest must reach
    #   only its own disk image; it leaves the mechanism to the management tool:
    #   https://www.qemu.org/docs/master/system/security.html
    # - libvirt keeps one per-host uid and keeps guests apart with per-VM labels (sVirt):
    #   https://libvirt.org/drvqemu.html#selinux-svirt-confinement
    #   https://libvirt.org/drvqemu.html#apparmor-svirt-confinement
    # - Passing each disk as an fd (-add-fd) would remove the path access, but sudo
    #   >= 1.9.14 closes inherited fds (below), so it needs another launcher.
    # Options to evaluate: per-VM SELinux/AppArmor labels, a per-VM uid pool (jailer).
    if workdir.use_qemu_vm_user:
        # Skip sudo when already running as qemu-vm user. This enables fd
        # inheritance (socket activation) which sudo >= 1.9.14 blocks via its
        # closefrom security policy (closes all fds >= 3 before exec).
        # See: https://www.sudo.ws/docs/man/sudo.man/
        qemu_vm_uid = get_qemu_vm_uid()
        if qemu_vm_uid is None or os.getuid() != qemu_vm_uid:
            # QEMU_VM_SUDO: sudo -n -u qemu-vm, then a fixed sh script that sets umask 007
            # and execs the command ("$@": no shell parsing of any argument, no injection).
            # The chardev and QMP sockets QEMU makes are then 0770, group qemu-vm: the
            # host (in group qemu-vm, checked by the probe) can connect, other users
            # cannot. The umask is set after sudo, because sudo-rs ORs the caller's umask
            # with 022 (no group write) and classic sudo lets PAM pam_umask set it.
            # stdbuf -oL forces line-buffered stdout to ensure console output is captured
            # immediately rather than being block-buffered (which happens with piped stdout).
            # IMPORTANT: stdbuf must come AFTER sudo - sudo sanitizes LD_PRELOAD for security.
            cmd.extend([*QEMU_VM_SUDO, "stdbuf", "-oL", *qemu_args])
            return cmd
        # Already running as qemu-vm — no sudo needed, fall through to direct exec

    cmd.extend(qemu_args)

    return cmd

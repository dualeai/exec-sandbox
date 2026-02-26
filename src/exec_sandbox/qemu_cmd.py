"""QEMU command line builder for microVM execution.

Builds QEMU command arguments based on platform capabilities, acceleration type,
and VM configuration.
"""

from exec_sandbox import cgroup, constants
from exec_sandbox._logging import get_logger
from exec_sandbox.models import ExposedPort
from exec_sandbox.platform_utils import HostArch, HostOS, detect_host_os
from exec_sandbox.settings import Settings
from exec_sandbox.system_probes import (
    check_tsc_deadline,
    detect_accel_type,
    probe_io_uring_support,
    probe_qemu_version,
    probe_unshare_support,
)
from exec_sandbox.vm_types import AccelType
from exec_sandbox.vm_working_directory import VmWorkingDirectory

logger = get_logger(__name__)


async def build_qemu_cmd(  # noqa: PLR0912, PLR0915
    settings: Settings,
    arch: HostArch,
    vm_id: str,
    workdir: VmWorkingDirectory,
    memory_mb: int,
    cpu_cores: int,
    allow_network: bool,
    expose_ports: list[ExposedPort] | None = None,
    direct_write: bool = False,
    debug_boot: bool = False,
    snapshot_drive: str | None = None,
    defer_incoming: bool = False,
) -> list[str]:
    """Build QEMU command for Linux (KVM + unshare + namespaces).

    Args:
        settings: Service configuration (paths, limits, etc.)
        arch: Host CPU architecture
        vm_id: Unique VM identifier
        workdir: VM working directory containing overlay and socket paths
        memory_mb: Guest VM memory in MB
        cpu_cores: Number of vCPUs for the guest (maps to -smp)
        allow_network: Enable network access via gvproxy (outbound internet)
        expose_ports: List of ports to expose from guest to host.
            When set without allow_network, uses QEMU user-mode networking
            with hostfwd (Mode 1). When set with allow_network, port
            forwarding is handled by gvproxy API (Mode 2).
        direct_write: Mount rootfs read-write (for snapshot creation).
        debug_boot: Enable verbose kernel/init boot logging. When True,
            sets loglevel=7, printk.devkmsg=on, init.quiet=0 for full
            boot diagnostics.
        snapshot_drive: Path to ext4 qcow2 for snapshot overlay (serial=snap).
            When set, tiny-init discovers drives by serial number and mounts
            EROFS base + ext4 snapshot via overlayfs. For snapshot creation
            (direct_write=True), the ext4 drive is writable; for usage, read-only.
        defer_incoming: Start QEMU with `-incoming defer` for L1 memory snapshot
            restore. The VM starts paused, waiting for a migration stream via QMP.

    Returns:
        QEMU command as list of strings
    """
    # Determine QEMU binary, machine type, and kernel based on architecture
    is_macos = detect_host_os() == HostOS.MACOS

    # Detect hardware acceleration type (centralized in detect_accel_type)
    accel_type = await detect_accel_type(force_emulation=settings.force_emulation)
    logger.info(
        "Hardware acceleration detection",
        extra={"vm_id": vm_id, "accel_type": accel_type.value, "is_macos": is_macos},
    )

    # Build accelerator string for QEMU
    if accel_type == AccelType.HVF:
        accel = "hvf"
    elif accel_type == AccelType.KVM:
        accel = "kvm"
    else:
        # TCG software emulation fallback (12x slower than KVM/HVF)
        #
        # thread=single: Disable MTTCG to reduce thread count per VM. Without this,
        # each VM creates multiple threads for parallel translation, exhausting
        # system thread limits when running parallel tests (qemu_thread_create:
        # Resource temporarily unavailable). Single-threaded TCG is slower but
        # prevents SIGABRT crashes on CI runners without KVM.
        # See: https://www.qemu.org/docs/master/devel/multi-thread-tcg.html
        #
        # tb-size: Translation block cache size in MB. QEMU 5.0+ defaults to 1GB
        # which causes OOM on CI runners with multiple VMs. Must match
        # cgroup.TCG_TB_CACHE_SIZE_MB for correct cgroup memory limits.
        # See cgroup.py for size rationale and benchmarks.
        accel = f"tcg,thread=single,tb-size={cgroup.TCG_TB_CACHE_SIZE_MB}"
        logger.warning(
            "Using TCG software emulation (slow) - KVM/HVF not available",
            extra={"vm_id": vm_id, "accel": accel},
        )

    # Track whether to use virtio-console (hvc0) or ISA serial (ttyS0)
    # Determined per-architecture below
    use_virtio_console = False

    if arch == HostArch.AARCH64:
        arch_suffix = "aarch64"
        qemu_bin = "qemu-system-aarch64"
        # highmem=off: Keep all RAM below 4GB for simpler memory mapping (faster boot)
        # gic-version=3: Explicit GIC version for TCG (ITS not modeled in TCG)
        # virtualization=off: Disable nested virt emulation (not needed, faster TCG)
        # its=off: Disable GICv3 ITS (Interrupt Translation Service) — we use virtio-mmio
        #   (not PCI), so MSI-X translation is unused; skips kernel ITS init (1-3ms)
        # dtb-randomness=off: Skip writing random seeds to DTB — redundant with
        #   random.trust_cpu=on in kernel cmdline
        machine_type = (
            "virt,virtualization=off,highmem=off,gic-version=3,its=off,dtb-randomness=off,mem-merge=off"
            if is_macos
            else "virt,virtualization=off,highmem=off,gic-version=3,its=off,dtb-randomness=off,mem-merge=off,dump-guest-core=off"
        )
        # ARM64 always uses virtio-console (no ISA serial on virt machine)
        use_virtio_console = True
    else:
        arch_suffix = "x86_64"
        qemu_bin = "qemu-system-x86_64"
        # Machine type selection based on acceleration:
        # - microvm: Optimized for KVM/HVF, requires hardware virtualization
        # - q35: Standard machine type that works with TCG (software emulation)
        # microvm is designed specifically for hardware virtualization and doesn't work correctly with TCG
        # See: https://www.qemu.org/docs/master/system/i386/microvm.html
        #
        # CRITICAL: acpi=off forces qboot instead of SeaBIOS
        # With ACPI enabled (default), microvm uses SeaBIOS which has issues with direct kernel boot
        # on QEMU 8.2. With acpi=off, it uses qboot which is specifically designed for direct kernel boot.
        # See: https://www.kraxel.org/blog/2020/10/qemu-microvm-acpi/
        if accel_type == AccelType.KVM:
            # =============================================================
            # Console Device Timing: ISA Serial vs Virtio-Console
            # =============================================================
            # ISA serial (ttyS0) is available IMMEDIATELY at boot because:
            #   - It's a simple I/O port at 0x3F8 emulated by QEMU
            #   - No driver initialization required
            #   - Kernel can write to it from first instruction
            #
            # Virtio-console (hvc0) is available LATER (~30-50ms) because:
            #   - Requires virtio-mmio bus discovery during kernel init
            #   - Requires virtio-serial driver initialization
            #   - Not available during early boot
            #
            # If kernel uses console=hvc0 but hvc0 doesn't exist yet -> HANG
            # See: https://gist.github.com/mcastelino/aa118275991d4f561ee22dc915b9345f
            #
            # =============================================================
            # TSC_DEADLINE Requirement for Non-Legacy Mode
            # =============================================================
            # pit=off, pic=off, and isa-serial=off require TSC_DEADLINE CPU feature
            # See: https://www.qemu.org/docs/master/system/i386/microvm.html
            #
            # In nested VMs (e.g., GitHub Actions on Azure/Hyper-V), TSC_DEADLINE
            # may not be exposed to the guest. Without it:
            #   - PIT/PIC disabled -> no timer/interrupt source -> kernel hang
            #   - ISA serial disabled -> must use hvc0 -> early boot hang
            #
            # =============================================================
            # Nested VM Fallback: microvm with Legacy Devices Enabled
            # =============================================================
            # When TSC_DEADLINE is unavailable (nested VMs on Azure/Hyper-V),
            # we keep microvm but enable ALL legacy devices:
            #
            # QEMU microvm legacy devices (enabled by default unless disabled):
            #   - i8259 PIC: Interrupt controller for legacy interrupt routing
            #   - i8254 PIT: Timer for scheduling and interrupt generation
            #   - MC146818 RTC: Real-time clock for timekeeping
            #   - ISA serial: Console output at ttyS0 (available at T=0)
            #
            # Why NOT fall back to 'pc' machine type:
            #   - microvm with virtio-mmio is simpler and faster to boot
            #   - Maintains consistent configuration between nested/bare-metal
            #   - virtio-mmio works fine in nested VMs when legacy devices present
            #   - 'pc' would require virtio-pci which needs different initramfs
            #
            # The key insight: without TSC_DEADLINE, kvmclock timing may be
            # unreliable in nested VMs. The PIT provides fallback timer source.
            #
            # See: https://www.qemu.org/docs/master/system/i386/microvm.html
            # =============================================================
            tsc_available = await check_tsc_deadline()
            if tsc_available:
                # Full optimization: TSC_DEADLINE available, use non-legacy mode
                machine_type = "microvm,acpi=off,x-option-roms=off,pit=off,pic=off,rtc=off,isa-serial=off,mem-merge=off,dump-guest-core=off"
                use_virtio_console = True
            else:
                # Nested VM compatibility: use microvm with timer legacy devices
                # Without TSC_DEADLINE, we need:
                #   - PIT (i8254) for timer interrupts
                #   - PIC (i8259) for interrupt handling
                #   - RTC for timekeeping (kvmclock may not work in nested VMs)
                # We disable ISA serial to avoid conflicts with virtio-serial.
                # Console output goes via virtio-console (hvc0) instead of ttyS0.
                # See: https://bugs.launchpad.net/qemu/+bug/1224444 (virtio-mmio issues)
                logger.info(
                    "TSC_DEADLINE not available, using microvm with legacy timers but virtio-console for nested VM compatibility",
                    extra={"vm_id": vm_id},
                )
                machine_type = "microvm,acpi=off,x-option-roms=off,isa-serial=off,mem-merge=off,dump-guest-core=off"
                use_virtio_console = True
        elif accel_type == AccelType.HVF:
            # macOS with HVF - configuration depends on architecture
            # Note: dump-guest-core=off not included - may not be supported on macOS QEMU
            if arch == HostArch.X86_64:
                # Intel Mac: check TSC_DEADLINE availability
                tsc_available = await check_tsc_deadline()
                if tsc_available:
                    # Full optimization: TSC_DEADLINE available, disable legacy devices
                    machine_type = (
                        "microvm,acpi=off,x-option-roms=off,pit=off,pic=off,rtc=off,isa-serial=off,mem-merge=off"
                    )
                else:
                    # Conservative: keep legacy timers for older Intel Macs
                    logger.info(
                        "TSC_DEADLINE not available on Intel Mac, using microvm with legacy timers",
                        extra={"vm_id": vm_id},
                    )
                    machine_type = "microvm,acpi=off,x-option-roms=off,isa-serial=off,mem-merge=off"
            else:
                # ARM64 Mac: no x86 legacy devices needed
                # ARM uses different timer mechanism (CNTVCT_EL0), no TSC concept
                machine_type = "microvm,acpi=off,x-option-roms=off,pit=off,pic=off,rtc=off,isa-serial=off,mem-merge=off"
            use_virtio_console = True
        else:
            # TCG emulation: use 'pc' (i440FX) which is simpler and more proven with direct kernel boot
            # q35 uses PCIe which can have issues with PCI device enumeration on some QEMU versions
            # See: https://wiki.qemu.org/Features/Q35
            machine_type = "pc,mem-merge=off,dump-guest-core=off"
            use_virtio_console = False
            logger.info(
                "Using pc machine type (TCG emulation, hardware virtualization not available)",
                extra={"vm_id": vm_id, "accel": accel},
            )

    # Auto-discover kernel and initramfs based on architecture
    # Note: existence validated in create_vm() before calling this method
    # Prefer uncompressed vmlinux for PVH direct boot (x86_64 only, ~50ms faster)
    # QEMU auto-detects PVH ELF note and skips kernel decompression
    vmlinux_path = settings.kernel_path / f"vmlinux-{arch_suffix}"
    vmlinuz_path = settings.kernel_path / f"vmlinuz-{arch_suffix}"
    kernel_path = vmlinux_path if arch == HostArch.X86_64 and vmlinux_path.exists() else vmlinuz_path
    initramfs_path = settings.kernel_path / f"initramfs-{arch_suffix}"

    # Layer 5: Linux namespaces (optional - requires capabilities or user namespaces)
    cmd: list[str] = []
    if detect_host_os() != HostOS.MACOS and await probe_unshare_support():
        if allow_network:
            unshare_args = ["unshare", "--pid", "--mount", "--uts", "--ipc", "--fork"]
            cmd.extend([*unshare_args, "--"])
        else:
            unshare_args = ["unshare", "--pid", "--net", "--mount", "--uts", "--ipc", "--fork"]
            cmd.extend([*unshare_args, "--"])

    # Build QEMU command arguments
    # Determine if we're using microvm (requires -nodefaults to avoid BIOS fallback)
    is_microvm = "microvm" in machine_type

    # =============================================================
    # Virtio Transport Selection: MMIO vs PCI
    # =============================================================
    # Virtio devices can use two transport mechanisms:
    #
    # virtio-mmio (suffix: -device):
    #   - Memory-mapped I/O, no PCI bus required
    #   - Simpler, smaller footprint, faster boot (~13%)
    #   - Used by: microvm (x86 - both nested and bare-metal), virt (ARM64)
    #   - Works in nested VMs when legacy devices (PIT/PIC/RTC) are enabled
    #
    # virtio-pci (suffix: -pci):
    #   - Standard PCI bus with MSI-X interrupts
    #   - Used by: pc/q35 (x86 TCG emulation)
    #   - Requires different initramfs with virtio_pci.ko
    #
    # Selection criteria:
    #   microvm (x86)        -> virtio-mmio (all KVM modes, nested or bare-metal)
    #   pc (x86 TCG)         -> virtio-pci (software emulation fallback)
    #   virt (ARM64)         -> virtio-mmio (initramfs loads virtio_mmio.ko)
    #
    # CRITICAL: ARM64 initramfs loads virtio_mmio.ko, NOT virtio_pci.ko
    # Using PCI devices on ARM64 causes boot hang (kernel can't find root device)
    # =============================================================
    virtio_suffix = "device" if (is_microvm or arch == HostArch.AARCH64) else "pci"

    qemu_args = [qemu_bin]

    # Set VM name for process identification (visible in ps aux, used by hwaccel test)
    # Format: guest=vm_id - the vm_id includes tenant, task, and uuid for uniqueness
    qemu_args.extend(["-name", f"guest={vm_id}"])

    # CRITICAL: -nodefaults -no-user-config are required for microvm to avoid BIOS fallback
    # See: https://www.qemu.org/docs/master/system/i386/microvm.html
    # For q35, we don't use these flags as the machine expects standard PC components
    if is_microvm:
        qemu_args.extend(["-nodefaults", "-no-user-config"])

    # Console selection based on machine type and architecture:
    # +--------------------------+-------------+--------------------------------+
    # | Configuration            | Console     | Reason                         |
    # +--------------------------+-------------+--------------------------------+
    # | x86 microvm + TSC        | hvc0        | Non-legacy, virtio-console     |
    # | x86 microvm - TSC        | hvc0        | ISA serial off, virtio-console |
    # | x86 pc (TCG only)        | ttyS0       | Software emulation fallback    |
    # | ARM64 virt               | ttyAMA0     | PL011 UART (always available)  |
    # +--------------------------+-------------+--------------------------------+
    # ttyS0 (ISA serial) is used when we need reliable early boot console (x86)
    # ttyAMA0 (PL011 UART) is used for ARM64 virt machine
    # hvc0 (virtio-console) is NOT reliable for kernel console on ARM64 because
    # it requires virtio-serial driver initialization (not available at early boot)
    # See: https://blog.memzero.de/toying-with-virtio/
    if arch == HostArch.AARCH64:
        # ARM64 virt machine has PL011 UART (ttyAMA0) - reliable at early boot
        # Note: hvc0 doesn't work for console because virtio-serial isn't ready
        # when kernel tries to open /dev/console, causing init to crash
        # loglevel=1 (KERN_ALERT): suppress kernel printk below ALERT during boot.
        # DEBUG (7) emits hundreds of messages, each requiring MMIO writes through
        # PL011 UART. Panics still print (console_flush_on_panic). tiny-init
        # messages are unaffected (direct libc::write to fd 2, not printk).
        # Ref: https://www.kernel.org/doc/html/latest/admin-guide/kernel-parameters.html
        console_params = "console=ttyAMA0 loglevel=1"
    elif use_virtio_console:
        # x86 non-legacy mode: ISA serial disabled, use virtio-console
        # loglevel=1: suppress kernel printk noise. Each message triggers a
        # virtqueue notification (MMIO doorbell + vCPU exit) on virtio-console.
        # Ref: https://www.kernel.org/doc/html/latest/admin-guide/kernel-parameters.html
        console_params = "console=hvc0 loglevel=1"
    else:
        # x86 legacy mode or TCG: ISA serial available at T=0, reliable boot
        # loglevel=1: suppress kernel printk noise. ISA serial (ttyS0) transmits
        # byte-by-byte through I/O port 0x3F8 -- extremely expensive under TCG
        # software emulation. Biggest win for TCG boot latency.
        # Ref: https://www.kernel.org/doc/html/latest/admin-guide/kernel-parameters.html
        console_params = "console=ttyS0 loglevel=1"

    qemu_args.extend(
        [
            "-accel",
            accel,
            "-cpu",
            # For hardware accel use host CPU with nested virt flags masked;
            # for TCG use optimized emulated CPUs.
            # -svm,-vmx: Hide AMD SVM and Intel VMX flags from guest to prevent
            # nested virtualization attacks (CVE-2024-50115 KVM nSVM nCR3 bug).
            # These flags are x86-specific — ARM64 HVF uses plain "host".
            # ARM64 TCG: cortex-a57 is 3x faster than max (no pauth overhead)
            # x86 TCG: Haswell required for AVX2 (Python/Bun built for x86_64_v3)
            # See: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1033643
            # See: https://gitlab.com/qemu-project/qemu/-/issues/844
            (
                "host"
                if accel_type in (AccelType.HVF, AccelType.KVM) and arch == HostArch.AARCH64
                else "host,-svm,-vmx"
                if accel_type in (AccelType.HVF, AccelType.KVM)
                else "cortex-a57"
                if arch == HostArch.AARCH64
                else "Haswell"
            ),
            "-M",
            machine_type,
            "-no-reboot",
            "-m",
            f"{memory_mb}M",
            "-smp",
            str(cpu_cores),
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
            # Removed (enforced by CONFIG): init_on_alloc, init_on_free,
            #   scsi_mod.scan, audit, slab_nomerge, nomodule, preempt,
            #   noresume, raid, numa_balancing, i8042.*, random.trust_cpu,
            #   panic, rcupdate.rcu_expedited, edd, noautogroup, io_delay
            # =============================================================
            f"{console_params} root=/dev/vda rootflags=noatime rootfstype=erofs rootwait fsck.mode=skip reboot=t init=/init page_alloc.shuffle=1 swiotlb=noforce"
            # THP: CONFIG_TRANSPARENT_HUGEPAGE=y enables EROFS large folios
            # (16-64KB per fault instead of 4KB), but transparent_hugepage=never
            # disables anonymous 2MB hugepages (no khugepaged, no compaction stalls).
            # File-backed large folios work regardless of this sysfs setting.
            + " transparent_hugepage=never"
            # Boot verbosity: debug_boot enables full kernel/init logging for diagnostics
            # Note: loglevel=7 overrides loglevel=1 set in console_params (kernel uses last occurrence)
            + (" loglevel=7" if debug_boot else " quiet loglevel=0")
            # Skip timer calibration — safe in virtualized env with reliable TSC (10-30ms)
            + " no_timer_check"
            # Keep expedited RCU after boot (built-in boot expediting covers boot phase)
            + " rcupdate.rcu_normal_after_boot=0"
            # /dev/kmsg access: on for debug, off for production
            + (" printk.devkmsg=on" if debug_boot else " printk.devkmsg=off")
            # Skip 8250 UART probing when ISA serial disabled (2-5ms, x86_64 only)
            + (" 8250.nr_uarts=0" if use_virtio_console and arch == HostArch.X86_64 else "")
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
            #    callbacks from rcu_normal_after_boot=0, virtio interrupts)
            #    trigger ~3k poll cycles/sec x 200us = ~65% busy.  The
            #    adaptive algorithm keeps poll_limit_ns at max because
            #    wakeups keep landing inside the poll window.
            #
            # 4. On HVF/TCG we disable cpuidle entirely (cpuidle.off=1).
            #    QEMU's virt machine generates no idle-states DT nodes, so the
            #    guest has only two cpuidle states: poll (busy-loop) and WFI.
            #    Any governor (menu, TEO) would always pick WFI — the prediction
            #    algorithm is wasted overhead.  cpuidle.off=1 bypasses the
            #    framework and calls cpu_do_idle() (WFI) directly.  HVF properly
            #    blocks the vCPU thread on WFI via qemu_wait_io_event() →
            #    halt_cond, yielding <5% idle CPU.
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
            + (" init.rw=1" if direct_write else "")
            + (" init.snap=1" if snapshot_drive else "")
            # Guest-agent log verbosity: init.quiet=0 un-gates log_info! macros
            + (" init.quiet=0" if debug_boot else " init.quiet=1")
            # Explicit TSC clocksource selection, skip probing (5-10ms, x86_64 only)
            + (" tsc=reliable clocksource=tsc" if arch == HostArch.X86_64 else ""),
        ]
    )

    # Platform-specific memory configuration
    # Note: -mem-prealloc tested for cold-start on HVF (pre-populates host pages)
    # but had NO effect — ARM VHE Stage-2 PTEs are still created lazily by hardware
    # on first guest access, regardless of host page presence.

    # Layer 3: Seccomp sandbox - Linux only
    if detect_host_os() != HostOS.MACOS:
        qemu_args.extend(
            [
                "-sandbox",
                "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny",
            ]
        )

    # Determine AIO mode and QEMU version based on cached startup probes
    io_uring_available = await probe_io_uring_support()
    qemu_version = await probe_qemu_version()
    aio_mode = "io_uring" if io_uring_available else "threads"
    if not io_uring_available:
        logger.debug(
            "Using aio=threads (io_uring not available)",
            extra={"reason": "syscall_probe_failed", "vm_id": vm_id},
        )

    # IOThread — offloads block I/O completion (pread, qcow2 decompress) to a
    # dedicated thread, separate from the vCPU thread. Safe and functional on
    # all accelerators (KVM, HVF, TCG):
    #   KVM:  ioeventfd enables zero-exit I/O kicks — clear performance win.
    #   HVF:  no ioeventfd (no eventfd on macOS), so no cold-start benefit
    #         (tested session 4: 9,798ms identical with/without). Bottleneck
    #         is Mach kernel hv_trap (~360µs/exit), not QEMU I/O processing.
    #   TCG:  software-emulated ioeventfd, moderate benefit (I/O off main loop).
    iothread_id = f"iothread0-{vm_id}"
    qemu_args.extend(["-object", f"iothread,id={iothread_id}"])

    # Disk configuration (EROFS base drive, serial=base)
    # Uses qcow2 overlay backed by the EROFS base image
    qemu_args.extend(
        [
            "-drive",
            f"file={workdir.overlay_image},"
            f"format=qcow2,"
            f"if=none,"
            f"id=hd0,"
            f"cache=unsafe,"
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
    # D2: event_idx=off reduces interrupt coalescing overhead during boot.
    # queue-size=256: tested at 128 vs 256 during cold-start investigation (session 4);
    # no measurable wall-time difference on HVF (bottleneck is Mach kernel, not queue
    # depth). Kept at 256 as a reasonable setting for throughput workloads.
    qemu_args.extend(
        [
            "-device",
            f"virtio-blk-{virtio_suffix},drive=hd0,serial=base,iothread={iothread_id},num-queues=1,queue-size=256,event_idx=off",
        ]
    )

    # Snapshot overlay drive — ext4 layer for overlayfs merge with EROFS base.
    # Presented as a second virtio-blk device; tiny-init detects it by serial=snap
    # and sets up overlayfs automatically (rw for snapshot creation, ro for usage).
    if snapshot_drive:
        qemu_args.extend(
            [
                "-drive",
                f"file={snapshot_drive},"
                f"format=qcow2,"
                f"if=none,"
                f"id=hd1,"
                f"cache=unsafe,"
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
                f"virtio-blk-{virtio_suffix},drive=hd1,serial=snap,iothread={iothread_id},num-queues=1,queue-size=256,event_idx=off",
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

    # virtio-serial device for guest agent communication AND kernel console (hvc0)
    # With microvm + -nodefaults, we must explicitly configure:
    # 1. virtconsole for kernel console=hvc0 (required for boot output)
    # 2. virtserialport for guest agent cmd/event channels
    qemu_args.extend(
        [
            # Chardevs for communication channels
            # server=on: QEMU creates a listening Unix socket
            # wait=off: QEMU starts VM immediately without waiting for client connection
            # Note: Socket permissions (via umask) are set in _build_linux_cmd.
            # The guest agent retries connection so timing is handled.
            "-chardev",
            f"socket,id=cmd0,path={workdir.cmd_socket},server=on,wait=off",
            "-chardev",
            f"socket,id=event0,path={workdir.event_socket},server=on,wait=off",
            # Chardev for console output - connected to virtconsole (hvc0)
            "-chardev",
            "stdio,id=virtiocon0,mux=on,signal=off",
        ]
    )

    # Serial port configuration:
    # - virtio-console mode (hvc0): Disable serial to avoid stdio conflict
    # - ISA serial mode (ttyS0): Connect serial to chardev for console output
    if use_virtio_console:
        # Disable default serial to prevent "cannot use stdio by multiple character devices"
        # ARM64 virt has a default PL011 UART, x86 microvm has ISA serial
        qemu_args.extend(["-serial", "none"])
    else:
        # x86 legacy mode: connect ISA serial to chardev for ttyS0
        qemu_args.extend(["-serial", "chardev:virtiocon0"])

    # =============================================================
    # Virtio-Serial Device Configuration
    # =============================================================
    # Virtio-serial provides guest agent communication channels (cmd/event ports).
    # Console output handling depends on use_virtio_console flag:
    #
    # NON-LEGACY MODE (use_virtio_console=True):
    #   - virtconsole device created for hvc0 (kernel console)
    #   - 3 ports: virtconsole (nr=0) + cmd (nr=1) + event (nr=2)
    #   - ISA serial disabled via isa-serial=off in machine type
    #   - Requires TSC_DEADLINE for reliable boot timing
    #
    # LEGACY MODE (use_virtio_console=False):
    #   - Still uses microvm with virtio-mmio (for nested VMs)
    #   - Or uses 'pc' with virtio-pci (for TCG emulation only)
    #   - NO virtconsole device (would conflict with ISA serial chardev)
    #   - 3 ports but only 2 used: cmd (nr=1) + event (nr=2)
    #   - Port 0 reserved for virtconsole (QEMU backward compat requirement)
    #   - ISA serial enabled, connected to stdio chardev for ttyS0
    #   - Used when TSC_DEADLINE unavailable (nested VMs) or TCG emulation
    #
    # Why not always create virtconsole?
    #   - Both virtconsole and ISA serial would use same chardev (virtiocon0)
    #   - QEMU allows mux=on sharing, but causes output interleaving issues
    #   - Cleaner to use one console device exclusively
    #
    # See: https://bugs.launchpad.net/qemu/+bug/1639791 (early virtio console lost)
    # See: https://gist.github.com/mcastelino/aa118275991d4f561ee22dc915b9345f
    # =============================================================
    if use_virtio_console:
        qemu_args.extend(
            [
                "-device",
                f"virtio-serial-{virtio_suffix},max_ports=3",
                # hvc0 console device - must be nr=0 to be hvc0
                "-device",
                "virtconsole,chardev=virtiocon0,nr=0",
                "-device",
                "virtserialport,chardev=cmd0,name=org.dualeai.cmd,nr=1",
                "-device",
                "virtserialport,chardev=event0,name=org.dualeai.event,nr=2",
            ]
        )
    else:
        # Legacy mode: no virtconsole, ISA serial handles console output
        # Port 0 is reserved for virtconsole (backward compat), so start at nr=1
        # See: QEMU error "Port number 0 on virtio-serial devices reserved for virtconsole"
        qemu_args.extend(
            [
                "-device",
                f"virtio-serial-{virtio_suffix},max_ports=3",
                "-device",
                "virtserialport,chardev=cmd0,name=org.dualeai.cmd,nr=1",
                "-device",
                "virtserialport,chardev=event0,name=org.dualeai.event,nr=2",
            ]
        )

    # virtio-balloon for host memory efficiency (deflate/inflate for warm pool)
    # - deflate-on-oom: guest returns memory under OOM pressure
    # - free-page-reporting: proactive free page hints to host (QEMU 5.1+/kernel 5.7+)
    #   Disabled permanently to avoid kernel page scanning overhead (10-20ms)
    qemu_args.extend(
        [
            "-device",
            f"virtio-balloon-{virtio_suffix},deflate-on-oom=on,free-page-reporting=off",
        ]
    )

    # virtio-rng: continuous host→guest entropy via /dev/urandom backend.
    # Feeds the kernel input pool so RNDRESEEDCRNG ioctl (guest-agent) has fresh
    # entropy to reseed from after L1 snapshot restore. Without this, the input
    # pool after restore contains only stale (cloned) entropy.
    # max-bytes + period: rate-limit to 1024B/100ms (10KB/s) to avoid host exhaustion.
    qemu_args.extend(
        [
            "-object",
            "rng-random,id=rng0,filename=/dev/urandom",
            "-device",
            f"virtio-rng-{virtio_suffix},rng=rng0,max-bytes=1024,period=100",
        ]
    )

    # =============================================================
    # Network Configuration: Three Modes + No-Network
    # =============================================================
    # All modes use gvproxy with socket networking for fast boot (~300ms).
    # SLIRP was removed because it's ~40x slower (~11s boot).
    #
    # Mode 0: No network (default)
    #   - Explicit "-nic none" suppresses QEMU's default NIC
    #   - Without this, machine types without -nodefaults (ARM64 virt,
    #     x86 pc/q35) create a default NIC, causing the guest-agent's
    #     verify_gvproxy() to burn ~4s in exponential-backoff retries
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

    needs_network = allow_network or bool(expose_ports)
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
            if expose_ports and not allow_network
            else "Mode 2 (port-forward + internet)"
            if expose_ports and allow_network
            else "Mode 3 (internet only)"
        )
        logger.info(
            f"Configuring socket networking via gvproxy ({mode_desc})",
            extra={
                "vm_id": vm_id,
                "expose_ports": [(p.internal, p.external) for p in expose_ports] if expose_ports else None,
                "allow_network": allow_network,
            },
        )

        qemu_args.extend(
            [
                "-netdev",
                netdev_opts,
                "-device",
                f"virtio-net-{virtio_suffix},netdev=net0,mq=off,csum=off,gso=off,host_tso4=off,host_tso6=off,mrg_rxbuf=off,ctrl_rx=off,guest_announce=off",
            ]
        )
    else:
        # Suppress QEMU's default NIC.  Without this, machine types that don't
        # use -nodefaults (ARM64 virt, x86 pc/q35) create a default virtio-net
        # device, causing the guest-agent to detect eth0 and run verify_gvproxy()
        # with exponential-backoff retries (~4s) before marking NETWORK_READY.
        # ExecuteCode/InstallPackages gate on NETWORK_READY, so the default NIC
        # adds ~4s to every cold-start execution even when no network is needed.
        qemu_args.extend(["-nic", "none"])

    # QMP (QEMU Monitor Protocol) socket for VM control operations
    qemu_args.extend(
        [
            "-qmp",
            f"unix:{workdir.qmp_socket},server=on,wait=off",
        ]
    )

    # L1 memory snapshot restore: QEMU starts paused, waiting for migration stream
    if defer_incoming:
        qemu_args.extend(["-incoming", "defer"])

    # Orphan protection: kill QEMU if parent process dies (QEMU 10.2+)
    # Uses PR_SET_PDEATHSIG on Linux, kqueue on macOS/FreeBSD
    if qemu_version is not None and qemu_version >= (10, 2, 0):
        qemu_args.extend(["-run-with", "exit-with-parent=on"])

    # Run QEMU as unprivileged user if qemu-vm user is available (optional hardening)
    # Falls back to current user if qemu-vm doesn't exist - VM still provides isolation
    if workdir.use_qemu_vm_user:
        # SECURITY: Avoid shell injection by not using 'sh -c'.
        # Instead, we use direct exec with preexec_fn to set umask.
        # stdbuf -oL forces line-buffered stdout to ensure console output is captured
        # immediately rather than being block-buffered (which happens with piped stdout).
        # IMPORTANT: stdbuf must come AFTER sudo - sudo sanitizes LD_PRELOAD for security.
        #
        # umask 007 is set via preexec_fn at subprocess creation time.
        # Creates chardev sockets with owner+group permissions (0660).
        # Host user must be in 'qemu-vm' group to connect to sockets owned by 'qemu-vm'.
        # More secure than 0666 (world-writable). Follows libvirt group membership pattern.
        cmd.extend(["sudo", "-u", "qemu-vm", "stdbuf", "-oL", *qemu_args])
        return cmd

    cmd.extend(qemu_args)

    return cmd

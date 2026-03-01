"""QEMU command line builder for microVM execution.

Builds QEMU command arguments based on platform capabilities, acceleration type,
and VM configuration.
"""

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import VmDependencyError
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
    memory_mb: float,
    cpu_cores: float,
    allow_network: bool,
    expose_ports: list[ExposedPort] | None = None,
    direct_write: bool = False,
    debug_boot: bool = False,
    snapshot_drive: str | None = None,
    defer_incoming: bool = False,
) -> list[str]:
    """Build QEMU command for microVM execution.

    Args:
        settings: Service configuration (paths, limits, etc.)
        arch: Host CPU architecture
        vm_id: Unique VM identifier
        workdir: VM working directory containing overlay and socket paths
        memory_mb: Guest VM memory in MB
        cpu_cores: Number of vCPUs for the guest (maps to -smp)
        allow_network: Enable network access via gvproxy (outbound internet)
        expose_ports: List of ports to expose from guest to host.
            When set without allow_network, uses gvproxy with
            BlockAllOutbound (Mode 1). When set with allow_network, port
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
        # TCG software emulation fallback (~5-8x slower than KVM/HVF)
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
        # constants.DEFAULT_TCG_TB_CACHE_SIZE_MB for correct cgroup memory limits.
        # See constants.py for size rationale and benchmarks.
        accel = f"tcg,thread=single,tb-size={constants.DEFAULT_TCG_TB_CACHE_SIZE_MB}"
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
        if arch == HostArch.AARCH64:
            qemu_version_check = await probe_qemu_version()
            if qemu_version_check is not None and qemu_version_check < (9, 0, 4):
                version_str = ".".join(str(v) for v in qemu_version_check)
                raise VmDependencyError(
                    f"QEMU {version_str} has a deterministic ARM64 TCG crash in "
                    f"regime_is_user() (target/arm/internals.h). Every kernel↔userspace "
                    f"transition triggers g_assert_not_reached(). "
                    f"Upgrade to QEMU >= 9.0.4 to fix this. "
                    f"See: https://gitlab.com/qemu-project/qemu/-/commit/1505b651fdbd"
                )

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
    else:
        arch_suffix = "x86_64"
        qemu_bin = "qemu-system-x86_64"
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
        tsc_available = False
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
        parts.extend(["isa-serial=off", "mem-merge=off"])
        if not is_macos:
            parts.append("dump-guest-core=off")
        machine_type = ",".join(parts)

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

    # Virtio transport: always MMIO (-device suffix). Both microvm (x86) and
    # virt (ARM64) use virtio-mmio. The kernel has CONFIG_VIRTIO_MMIO=y built-in
    # and CONFIG_MODULES is not set — PCI devices would cause boot hang.
    virtio_suffix = "device"

    qemu_args = [qemu_bin]

    # Set VM name for process identification (visible in ps aux, used by hwaccel test)
    # Format: guest=vm_id - the vm_id includes tenant, task, and uuid for uniqueness
    qemu_args.extend(["-name", f"guest={vm_id}"])

    # CRITICAL: -nodefaults -no-user-config are required for microvm to avoid BIOS fallback
    # See: https://www.qemu.org/docs/master/system/i386/microvm.html
    if is_microvm:
        qemu_args.extend(["-nodefaults", "-no-user-config"])

    # Console: ARM64 uses PL011 UART (ttyAMA0), x86 uses virtio-console (hvc0).
    # hvc0 is NOT reliable on ARM64 — virtio-serial isn't ready when the kernel
    # opens /dev/console, causing init to crash.
    # See: https://blog.memzero.de/toying-with-virtio/
    console_params = "console=ttyAMA0 loglevel=1" if arch == HostArch.AARCH64 else "console=hvc0 loglevel=1"

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
            # ARM64 TCG: "max" exposes every feature the current QEMU TCG
            # version supports, adapting automatically — on QEMU 8.2 (Ubuntu
            # 24.04 CI) it enables fewer ARMv9 features than on 10.x, but only
            # ones that work. A named model like neoverse-n2 demands features
            # (SVE2, MTE, full PAC) that QEMU 8.x TCG cannot emulate correctly,
            # causing guest panics during early boot. "max" is what QEMU's own
            # TCG test suite uses. pauth-impdef=on forces the fast impdef PAC
            # algorithm (QEMU 10.0+ defaults to this for virt >= 10.0, but
            # older versions use QARMA5 which costs ~50% of TCG cycles).
            # x86 TCG: SapphireRapids-v2 provides full Spectre/MDS mitigation
            # flags (spec-ctrl, stibp, ssbd, arch-capabilities, md-clear,
            # mds-no, taa-no, gds-no, rfds-no). TCG silently strips AVX-512
            # and AMX (not emulated). On ARM64 hosts the aarch64 TCG backend
            # also lacks 256-bit vector ops (TCG_TARGET_HAS_v256=0), so AVX2
            # is unavailable — effective ceiling is SSE4.2. On x86 hosts the
            # effective ceiling is x86_64-v3 (AVX2/FMA).
            # See: https://www.qemu.org/docs/master/system/i386/cpu.html
            # See: https://gitlab.com/qemu-project/qemu/-/issues/844
            (
                "host"
                if accel_type in (AccelType.HVF, AccelType.KVM) and arch == HostArch.AARCH64
                else "host,-svm,-vmx"
                if accel_type in (AccelType.HVF, AccelType.KVM)
                else "max,pauth-impdef=on"
                if arch == HostArch.AARCH64
                else "SapphireRapids-v2"
            ),
            "-M",
            machine_type,
            "-no-reboot",
            "-m",
            f"{int(memory_mb)}M",  # QEMU -m requires integer MB
            "-smp",
            str(int(cpu_cores)),  # QEMU -smp requires integer vCPU count
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
            #    callbacks from rcu_normal_after_boot=0, virtio interrupts)
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
    # Cap I/O thread pool to 16 (QEMU default is 64).  With single-queue
    # virtio-blk on qcow2, 16 is more than sufficient.  When aio=io_uring
    # is active the pool is unused (no-op).  When aio=threads, this caps
    # peak threads from ~72 to ~24, preventing pids.max exhaustion.
    qemu_args.extend(["-object", f"iothread,id={iothread_id},thread-pool-min=0,thread-pool-max=16"])

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

    # Chardevs for guest agent communication AND kernel console (hvc0).
    # All machine types need explicit chardev setup:
    #   microvm: -nodefaults suppresses everything, so all devices must be declared
    #   ARM64 virt: no -nodefaults, but we still need named ports for guest-agent
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
    #   - virtconsole device created for hvc0 (kernel console)
    #   - 3 ports: virtconsole (nr=0) + cmd (nr=1) + event (nr=2)
    #   - ISA serial disabled via isa-serial=off in machine type
    #
    # See: https://bugs.launchpad.net/qemu/+bug/1639791 (early virtio console lost)
    # See: https://gist.github.com/mcastelino/aa118275991d4f561ee22dc915b9345f
    # =============================================================
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
    #   - Without this, machine types without -nodefaults (ARM64 virt)
    #     create a default NIC, causing the guest-agent's
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
        # use -nodefaults (ARM64 virt) create a default virtio-net
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

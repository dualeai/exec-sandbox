"""QEMU command line builder for microVM execution.

Builds QEMU command arguments based on platform capabilities, acceleration type,
and VM configuration.
"""

import logging

from exec_sandbox import cgroup, constants
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

logger = logging.getLogger(__name__)


async def build_qemu_cmd(  # noqa: PLR0912, PLR0915
    settings: Settings,
    arch: HostArch,
    vm_id: str,
    workdir: VmWorkingDirectory,
    memory_mb: int,
    cpu_cores: int,
    allow_network: bool,
    expose_ports: list[ExposedPort] | None = None,
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
        machine_type = (
            "virt,virtualization=off,highmem=off,gic-version=3,mem-merge=off"
            if is_macos
            else "virt,virtualization=off,highmem=off,gic-version=3,mem-merge=off,dump-guest-core=off"
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
    kernel_path = settings.kernel_path / f"vmlinuz-{arch_suffix}"
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
    # | x86 microvm - TSC        | ttyS0       | Legacy mode, ISA serial        |
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
        console_params = "console=ttyAMA0 loglevel=7"
    elif use_virtio_console:
        # x86 non-legacy mode: ISA serial disabled, use virtio-console
        console_params = "console=hvc0 loglevel=7"
    else:
        # x86 legacy mode or TCG: ISA serial available at T=0, reliable boot
        console_params = "console=ttyS0 loglevel=7"

    qemu_args.extend(
        [
            "-accel",
            accel,
            "-cpu",
            # For hardware accel use host CPU, for TCG use optimized emulated CPUs
            # ARM64 TCG: cortex-a57 is 3x faster than max (no pauth overhead)
            # x86 TCG: Haswell required for AVX2 (Python/Bun built for x86_64_v3)
            # See: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1033643
            # See: https://gitlab.com/qemu-project/qemu/-/issues/844
            (
                "host"
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
            # Boot params: console varies by machine type, minimal kernel logging
            # =============================================================
            # Boot Parameter Optimizations (validated Jan 2025):
            # - nokaslr: Skip KASLR (safe for ephemeral isolated VMs)
            # - noresume: Skip hibernate resume check (VMs don't hibernate)
            # - swiotlb=noforce: Disable software I/O TLB (virtio uses direct DMA)
            # - panic=-1: Immediate reboot on panic (boot timeout handles loops)
            # - i8042.nokbd: Skip keyboard port check (no PS/2 in VM)
            # - tsc=reliable: Trust TSC clocksource (x86_64 only, kvmclock stable)
            # See: https://github.com/firecracker-microvm/firecracker
            # See: https://www.qemu.org/docs/master/system/i386/microvm.html
            # =============================================================
            f"{console_params} root=/dev/vda rootflags=rw,noatime rootfstype=ext4 rootwait=2 fsck.mode=skip reboot=t panic=-1 preempt=none i8042.noaux i8042.nomux i8042.nopnp i8042.nokbd init=/init random.trust_cpu=on raid=noautodetect mitigations=off nokaslr noresume swiotlb=noforce"
            # init.net=1: load network modules when networking is needed
            # Required for: allow_network=True (gvproxy) OR expose_ports (QEMU hostfwd)
            # init.balloon=1: load balloon module (always, needed for warm pool)
            + (" init.net=1" if (allow_network or expose_ports) else "")
            + " init.balloon=1"
            # tsc=reliable only for x86_64 (TSC is x86-specific, ARM uses CNTVCT_EL0)
            + (" tsc=reliable" if arch == HostArch.X86_64 else ""),
        ]
    )

    # Platform-specific memory configuration
    # Note: -mem-prealloc removed for faster boot (demand-paging is fine for ephemeral VMs)
    host_os = detect_host_os()

    # Layer 3: Seccomp sandbox - Linux only
    if detect_host_os() != HostOS.MACOS:
        qemu_args.extend(
            [
                "-sandbox",
                "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny",
            ]
        )

    # Determine AIO mode based on cached startup probe
    io_uring_available = await probe_io_uring_support()
    aio_mode = "io_uring" if io_uring_available else "threads"
    if not io_uring_available:
        logger.debug(
            "Using aio=threads (io_uring not available)",
            extra={"reason": "syscall_probe_failed", "vm_id": vm_id},
        )

    # IOThread configuration
    match host_os:
        case HostOS.LINUX:
            use_iothread = True
        case HostOS.MACOS | HostOS.UNKNOWN:
            use_iothread = False

    iothread_id = f"iothread0-{vm_id}" if use_iothread else None
    if use_iothread:
        qemu_args.extend(["-object", f"iothread,id={iothread_id}"])

    # Disk configuration
    # Uses overlay backed by either:
    # - snapshot_path (cached L2 qcow2) for pre-installed packages
    # - base_image for cold boot
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

    # Platform-specific block device
    match host_os:
        case HostOS.MACOS:
            qemu_args.extend(
                [
                    "-device",
                    f"virtio-blk-{virtio_suffix},drive=hd0,num-queues=1,queue-size=128",
                ]
            )
        case HostOS.LINUX | HostOS.UNKNOWN:
            qemu_args.extend(
                [
                    "-device",
                    # NOTE: Removed logical_block_size=4096,physical_block_size=4096
                    # Small ext4 filesystems (<512MB) use 1024-byte blocks by default, so forcing
                    # 4096-byte block size causes mount failures ("Invalid argument")
                    f"virtio-blk-{virtio_suffix},drive=hd0,iothread={iothread_id},num-queues=1,queue-size=128",
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
    qemu_args.extend(
        [
            "-device",
            f"virtio-balloon-{virtio_suffix},deflate-on-oom=on,free-page-reporting=on",
        ]
    )

    # =============================================================
    # Network Configuration: Three Modes (all via gvproxy)
    # =============================================================
    # All modes use gvproxy with socket networking for fast boot (~300ms).
    # SLIRP was removed because it's ~40x slower (~11s boot).
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
        qemu_version = await probe_qemu_version()
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

    # QMP (QEMU Monitor Protocol) socket for VM control operations
    qemu_args.extend(
        [
            "-qmp",
            f"unix:{workdir.qmp_socket},server=on,wait=off",
        ]
    )

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

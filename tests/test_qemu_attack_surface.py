"""Tests for QEMU/KVM attack surface minimization.

Validates that the guest VM exposes only the minimal set of devices and
interfaces needed for operation. Every emulated device is a potential
attack surface for VM escape via QEMU bugs.

CVE references:
- CVE-2019-14378: QEMU SLiRP heap overflow via IPv4 fragmentation
- CVE-2020-14364: QEMU USB emulation out-of-bounds read/write
- CVE-2019-14835: vhost-net buffer overflow during live migration
- CVE-2021-3947: QEMU NVME stack buffer overflow
- CVE-2023-6693: Virtio-net stack buffer overflow in flush_tx (mrg_rxbuf)
- CVE-2023-20869/70: VMware Bluetooth device sharing escape (Pwn2Own 2023)
- CVE-2024-3446: Virtio DMA reentrancy double free (CVSS 8.2, affects virtio-serial)
- CVE-2024-3567: SCTP checksum assertion crash in network TX path
- CVE-2024-6505: Virtio-net RSS heap overflow
- CVE-2024-7730: Virtio-snd heap buffer overflow
- CVE-2024-8354: USB endpoint assertion crash
- CVE-2024-24474: SCSI ESP buffer overflow
- CVE-2025-12464: e1000 network device stack buffer overflow
- CVE-2025-22224/25/26: VMware ESXi TOCTOU VM escape chain (in-the-wild)
- CVE-2024-22267: Pwn2Own 2024 VMware Bluetooth UAF VM escape
- CVE-2025-41236/37/38/39: Pwn2Own 2025 VMware escape chain

Principle: No unnecessary emulated devices = No unnecessary attack surface.

Test categories:
- Normal: verify minimal device set (only virtio-blk, virtio-net, virtio-serial, virtio-rng)
- Edge: no USB, sound, display, graphics, floppy, parallel, SCSI, legacy NIC devices
- Weird: the network is gvproxy's, not SLiRP's
- Out of bounds: only known device nodes, a virtio-net without the risky features, no nested
  virtualization
"""

from exec_sandbox.models import Language, PortMapping
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

_NETWORK_DEVICE = VmSpec(language=Language.PYTHON, expose_ports=[PortMapping(internal=8080)])
"""A VM with the virtio-net device and no outbound access (an exposed port, mode 1)."""


# =============================================================================
# Normal: Only essential virtio devices present
# =============================================================================
class TestMinimalDeviceSet:
    """Guest should expose only virtio-blk, virtio-net, virtio-serial and
    virtio-rng (no balloon). No legacy emulated hardware."""

    async def test_only_virtio_devices(self, scheduler: Scheduler) -> None:
        """The guest has exactly the virtio devices of a VM without a network: entropy (rng),
        console (virtio-serial), and two block devices (the root image and the home disk)."""
        code = """\
import glob
print(sorted(open(f'{d}/device').read().strip() for d in glob.glob('/sys/bus/virtio/devices/*')))
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        # virtio IDs: 0x0002 block, 0x0003 console (virtio-serial), 0x0004 entropy (rng).
        # Any other is extra attack surface: balloon 0x0005, virtio-snd 0x0019 (CVE-2024-7730).
        assert (result.exit_code, result.stdout) == (0, "['0x0002', '0x0002', '0x0003', '0x0004']\n"), result.stderr

    async def test_no_pci_non_virtio_devices(self, scheduler: Scheduler) -> None:
        """No legacy PCI devices (IDE, AHCI, e1000, rtl8139, AC97, etc.).

        CVE-2020-14364: USB emulation OOB R/W — example of legacy device bugs.
        """
        code = """\
import glob, os
drivers = [os.path.basename(os.readlink(link)) for link in glob.glob('/sys/bus/pci/devices/*/driver')]
print([d for d in drivers if 'virtio' not in d])  # virtio-pci is expected, everything else is suspect
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        # No PCI device bound to a driver other than virtio
        assert (result.exit_code, result.stdout) == (0, "[]\n"), result.stderr


# =============================================================================
# Edge: No USB subsystem (CVE-2020-14364)
# =============================================================================
class TestNoUsbDevices:
    """USB emulation is a major attack surface in QEMU.

    CVE-2020-14364: Out-of-bounds read/write in hw/usb/core.c via
    USBDevice setup_len exceeding data_buf[4096]. No USB = no risk.
    """

    async def test_no_usb_bus_or_devices(self, scheduler: Scheduler) -> None:
        """No USB bus in the guest, and no /dev/bus/usb or /dev/usb device nodes."""
        code = "import os; print([p for p in ('/sys/bus/usb', '/dev/bus/usb', '/dev/usb') if os.path.lexists(p)])"
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "[]\n"), result.stderr


# =============================================================================
# Edge: No sound, display, or graphics devices
# =============================================================================
class TestNoMultimediaDevices:
    """Sound and display devices add attack surface without value for
    code execution sandboxes."""

    async def test_no_sound_or_graphics_devices(self, scheduler: Scheduler) -> None:
        """No ALSA/sound device, framebuffer or DRM graphics device: no /dev/snd, /dev/dsp,
        /dev/fb0 or /dev/dri node, and the sound, graphics and drm sysfs classes are empty or
        absent.

        CVE-2023-20869: VMware Bluetooth device sharing stack overflow
        demonstrates that peripheral device emulation is dangerous.
        """
        code = """\
import os
print([p for p in ('/dev/snd', '/dev/dsp', '/dev/fb0', '/dev/dri') if os.path.lexists(p)])
print([c for c in ('sound', 'graphics', 'drm') if os.path.isdir(f'/sys/class/{c}') and os.listdir(f'/sys/class/{c}')])
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "[]\n[]\n"), result.stderr


# =============================================================================
# Edge: No legacy I/O devices
# =============================================================================
class TestNoLegacyDevices:
    """Legacy devices (floppy, parallel, PS/2) add attack surface."""

    async def test_no_floppy_or_parallel_port(self, scheduler: Scheduler) -> None:
        """No floppy disk (/dev/fd0, /dev/fd1) or parallel port (/dev/parport*, /dev/lp*) device
        nodes."""
        code = "import glob; print(glob.glob('/dev/fd[01]') + glob.glob('/dev/parport*') + glob.glob('/dev/lp*'))"
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "[]\n"), result.stderr


# =============================================================================
# Edge: No SCSI devices (CVE-2024-24474)
# =============================================================================
class TestNoScsiDevices:
    """SCSI emulation has had buffer overflow CVEs.

    CVE-2024-24474: SCSI ESP buffer overflow in hw/scsi/esp.c.
    exec-sandbox uses virtio-blk for disk, not SCSI. No SCSI = no risk.
    """

    async def test_no_scsi_bus(self, scheduler: Scheduler) -> None:
        """No SCSI bus or SCSI host adapters should exist."""
        code = """\
import os, glob
scsi_hosts = glob.glob('/sys/class/scsi_host/host*')
scsi_devices = glob.glob('/sys/bus/scsi/devices/*')
print(f"SCSI_HOSTS:{len(scsi_hosts)}")
print(f"SCSI_DEVICES:{len(scsi_devices)}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "SCSI_HOSTS:0\nSCSI_DEVICES:0\n"


# =============================================================================
# Edge: No legacy network devices (CVE-2025-12464)
# =============================================================================
class TestNoLegacyNicDevices:
    """Legacy emulated NICs (e1000, rtl8139) have had multiple CVEs.

    CVE-2025-12464: e1000 stack buffer overflow in loopback mode.
    exec-sandbox uses virtio-net only. No legacy NIC = no risk.
    """

    async def test_no_legacy_nic_drivers(self, scheduler: Scheduler) -> None:
        """A VM with a network device: each network interface with a device uses virtio_net."""
        code = """\
import glob, os
print([os.path.basename(os.readlink(link)) for link in glob.glob('/sys/class/net/*/device/driver')])
"""
        result = await run_once(scheduler, _NETWORK_DEVICE, code)
        assert (result.exit_code, result.stdout) == (0, "['virtio_net']\n"), result.stderr


# =============================================================================
# Weird: Networking uses gvproxy, not SLiRP (CVE-2019-14378)
# =============================================================================
class TestNetworkingSecure:
    """QEMU's SLiRP user-mode networking has had multiple heap overflow CVEs.

    CVE-2019-14378: Heap buffer overflow in SLiRP during IPv4 fragmented
    packet reassembly. exec-sandbox uses gvproxy instead.
    """

    async def test_network_is_gvproxy_not_slirp(self, scheduler: Scheduler) -> None:
        """A VM with a network device: its address is 192.168.127.2 and its default gateway
        192.168.127.1 (gvproxy), not 10.0.2.15 and 10.0.2.2 (SLiRP)."""
        code = """\
import socket, struct
with open('/proc/net/route') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 3 and parts[1] == '00000000':
            # Gateway in /proc/net/route is a hex u32 in host byte order
            # (little-endian on x86_64 and aarch64 Linux).
            # inet_ntoa expects network byte order (big-endian), so use '<I'
            # to read as little-endian and let inet_ntoa handle the rest.
            gw_ip = socket.inet_ntoa(struct.pack('<I', int(parts[2], 16)))
            print(f"GATEWAY:{gw_ip}")
            break
    else:
        print("NO_DEFAULT_ROUTE")
# A UDP connect sends nothing: it only picks the source address of the route
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
    probe.connect(('192.168.127.1', 9))
    print(f"ADDRESS:{probe.getsockname()[0]}")
"""
        result = await run_once(scheduler, _NETWORK_DEVICE, code)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "GATEWAY:192.168.127.1\nADDRESS:192.168.127.2\n"


# =============================================================================
# Out of bounds: QEMU process-level protections
# =============================================================================
class TestQemuSandboxFlags:
    """What the guest can reach of QEMU: device nodes, virtio-net features, and CPU
    virtualization."""

    async def test_dev_holds_only_known_device_nodes(self, scheduler: Scheduler) -> None:
        """Each node in /dev is a known one: the guest gets no node of another emulated device
        (a QMP channel among them).

        No allowance for nodes the guest cannot have: the kernel has no GPIOLIB, NVMe, VMCI,
        NVRAM, IOMMUFD or VT (images/kernel/exec-sandbox.config), the guest agent removes
        /dev/zram0 before user code runs, and nothing mounts /dev/pts or /dev/mqueue.
        """
        code = """\
import glob, os
# Known safe devices
safe = {'null', 'zero', 'full', 'random', 'urandom', 'tty',
        'console', 'ptmx', 'shm', 'fd',
        'stdin', 'stdout', 'stderr', 'kmsg', 'cpu_dma_latency',
        'virtio-ports',
        'hwrng'}    # Hardware RNG via virtio-rng (feeds /dev/random entropy pool)
# ttyS*/ttyAMA0: serial ports; hvc*: virtio consoles; vport*: virtio-serial ports;
# rtc*: the ARM64 virt PL031 clock. A dangling symlink is no node.
for dev in glob.glob('/dev/*'):
    name = os.path.basename(dev)
    if name not in safe and not name.startswith(('tty', 'hvc', 'vport', 'rtc')) and os.path.exists(dev):
        print(dev)
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, ""), result.stderr  # no unknown device node

    async def test_virtio_net_minimal_features(self, scheduler: Scheduler) -> None:
        """The virtio-net device of a VM with a network device has the features QEMU turns off,
        off: the guest reads them from its features bits (bit N of the virtio spec at index N).

        CVE-2023-6693: Stack buffer overflow in flush_tx when MRG_RXBUF is
        enabled. Our config sets mrg_rxbuf=off to prevent this.
        CVE-2024-6505: RSS heap overflow. Our config does not enable RSS.

        Reducing features reduces QEMU code paths reachable from guest.
        """
        code = """\
import glob
for d in glob.glob('/sys/bus/virtio/devices/*'):
    if open(f'{d}/device').read().strip() == '0x0001':  # virtio-net
        print(open(f'{d}/features').read().strip())
"""
        result = await run_once(scheduler, _NETWORK_DEVICE, code)
        assert result.exit_code == 0, result.stderr
        (features,) = result.stdout.splitlines()
        assert len(features) >= 64, features  # one char per bit: 64, or 128 on kernels with more feature bits
        # virtio-net feature bits: CSUM 0, GSO 6, HOST_TSO4 11, HOST_TSO6 12, MRG_RXBUF 15,
        # CTRL_RX 18, GUEST_ANNOUNCE 21, MQ 22, RSS 60
        bits = {"CSUM": 0, "GSO": 6, "HOST_TSO4": 11, "HOST_TSO6": 12, "MRG_RXBUF": 15}
        bits |= {"CTRL_RX": 18, "GUEST_ANNOUNCE": 21, "MQ": 22, "RSS": 60}
        assert {name: features[bit] for name, bit in bits.items()} == dict.fromkeys(bits, "0")

    async def test_no_nested_virtualization(self, scheduler: Scheduler) -> None:
        """Nested virtualization must be disabled.

        CVE-2024-50115: KVM nSVM nCR3 register handling bug in nested AMD SVM.
        The guest CPU has no VMX or SVM: -svm,-vmx on x86 (qemu_cmd.py _X86_CPU_FLAGS),
        virtualization=off on the ARM64 virt machine.
        """
        code = """\
import os
# Check for KVM device inside guest
kvm_exists = os.path.exists('/dev/kvm')
print(f"KVM_IN_GUEST:{kvm_exists}")
# Check for VMX/SVM CPU flags
with open('/proc/cpuinfo') as f:
    cpuinfo = f.read()
has_vmx = 'vmx' in cpuinfo
has_svm = 'svm' in cpuinfo
print(f"VMX:{has_vmx}")
print(f"SVM:{has_svm}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "KVM_IN_GUEST:False\nVMX:False\nSVM:False\n"

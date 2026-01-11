#!/bin/bash
# Test custom initramfs with full QEMU args matching vm_manager.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGES_DIR="$REPO_ROOT/images/dist"

ARCH="aarch64"
KERNEL="$IMAGES_DIR/vmlinuz-$ARCH"
INITRAMFS="$IMAGES_DIR/initramfs-$ARCH"
DISK="$IMAGES_DIR/python-3.14-base-$ARCH.qcow2"

# Build custom initramfs
echo "Building custom initramfs..."
"$SCRIPT_DIR/build-initramfs.sh" "$ARCH" "$IMAGES_DIR"

# Create temp disk copy
TEMP_DISK=$(mktemp)
cp "$DISK" "$TEMP_DISK"
trap "rm -f $TEMP_DISK" EXIT

# Create temp socket paths
CMD_SOCK=$(mktemp -u)
EVENT_SOCK=$(mktemp -u)

echo ""
echo "=== Test Configuration ==="
echo "Kernel:    $KERNEL"
echo "Initramfs: $INITRAMFS ($(du -h "$INITRAMFS" | cut -f1))"
echo "Disk:      $DISK"
echo "CMD sock:  $CMD_SOCK"
echo "EVT sock:  $EVENT_SOCK"
echo ""
echo "Starting QEMU with verbose output..."
echo "Press Ctrl+A then X to exit QEMU"
echo "=========================="
echo ""

# Run QEMU with args matching vm_manager.py for aarch64
qemu-system-aarch64 \
    -M virt,mem-merge=off \
    -accel hvf \
    -cpu host \
    -m 256m \
    -smp 1 \
    -kernel "$KERNEL" \
    -initrd "$INITRAMFS" \
    -drive "file=$TEMP_DISK,format=qcow2,if=virtio,cache=unsafe" \
    -append "console=ttyAMA0 loglevel=7 root=/dev/vda rootflags=rw,noatime rootfstype=ext4 rootwait=2 fsck.mode=skip reboot=t preempt=none init=/init" \
    -chardev "socket,id=cmd0,path=$CMD_SOCK,server=on,wait=off" \
    -chardev "socket,id=event0,path=$EVENT_SOCK,server=on,wait=off" \
    -device "virtio-serial-device,max_ports=4" \
    -device "virtserialport,chardev=cmd0,name=org.dualeai.cmd,nr=1" \
    -device "virtserialport,chardev=event0,name=org.dualeai.event,nr=2" \
    -nographic \
    -no-reboot

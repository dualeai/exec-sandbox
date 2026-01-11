#!/bin/bash
# Test custom initramfs boot with visible console output
# This helps debug boot issues by showing all kernel and init messages

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGES_DIR="$REPO_ROOT/images/dist"

# Detect architecture
case "$(uname -m)" in
    x86_64|amd64) ARCH="x86_64"; QEMU_ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64"; QEMU_ARCH="aarch64" ;;
    *) echo "Unsupported arch"; exit 1 ;;
esac

# Build custom initramfs
echo "Building custom initramfs for $ARCH..."
"$SCRIPT_DIR/build-initramfs.sh" "$ARCH" "$IMAGES_DIR"

# Check files exist
KERNEL="$IMAGES_DIR/vmlinuz-$ARCH"
INITRAMFS="$IMAGES_DIR/initramfs-$ARCH"
DISK="$IMAGES_DIR/python-3.14-base-$ARCH.qcow2"

if [ ! -f "$KERNEL" ]; then
    echo "Kernel not found: $KERNEL"
    exit 1
fi
if [ ! -f "$INITRAMFS" ]; then
    echo "Initramfs not found: $INITRAMFS"
    exit 1
fi
if [ ! -f "$DISK" ]; then
    echo "Disk not found: $DISK (trying raw-base)"
    DISK="$IMAGES_DIR/raw-base-$ARCH.qcow2"
fi

echo ""
echo "=== Test Configuration ==="
echo "Kernel:    $KERNEL ($(du -h "$KERNEL" | cut -f1))"
echo "Initramfs: $INITRAMFS ($(du -h "$INITRAMFS" | cut -f1))"
echo "Disk:      $DISK"
echo ""
echo "Starting QEMU with verbose output..."
echo "Press Ctrl+A then X to exit QEMU"
echo "=========================="
echo ""

# Create a copy of the disk to avoid corrupting the original
TEMP_DISK=$(mktemp)
cp "$DISK" "$TEMP_DISK"
trap "rm -f $TEMP_DISK" EXIT

# Build QEMU command based on architecture
if [ "$ARCH" = "aarch64" ]; then
    # macOS on Apple Silicon
    qemu-system-aarch64 \
        -M virt,accel=hvf \
        -cpu host \
        -m 256m \
        -kernel "$KERNEL" \
        -initrd "$INITRAMFS" \
        -drive "file=$TEMP_DISK,format=qcow2,if=virtio,cache=unsafe" \
        -append "console=ttyAMA0 loglevel=7 root=/dev/vda rootflags=rw rootfstype=ext4 init=/init" \
        -nographic \
        -no-reboot
else
    # x86_64
    qemu-system-x86_64 \
        -M microvm,x-option-roms=off \
        -enable-kvm \
        -cpu host \
        -m 256m \
        -kernel "$KERNEL" \
        -initrd "$INITRAMFS" \
        -drive "file=$TEMP_DISK,format=qcow2,if=virtio,cache=unsafe" \
        -append "console=ttyS0 loglevel=7 root=/dev/vda rootflags=rw rootfstype=ext4 init=/init" \
        -nographic \
        -no-reboot
fi

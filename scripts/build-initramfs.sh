#!/bin/bash
set -euo pipefail

# Build minimal initramfs for QEMU microVM
#
# With CONFIG_MODULES=n (custom kernel), all drivers are built-in.
# Initramfs contains only: tiny-init binary + device nodes.
# No kernel modules needed — saves ~10s build time + ~400KB.
#
# Features:
# - tiny-init: single static Rust binary (~50-100KB vs 1MB busybox)
# - zstd compression (~30% smaller than LZ4)
# - Size: ~80-100KB (down from ~500KB with modules)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_DIR="$SCRIPT_DIR/../images"
DEFAULT_OUTPUT_DIR="$IMAGES_DIR/dist"

ARCH="${1:-x86_64}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# Convert OUTPUT_DIR to absolute path
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# Map architecture names
case "$ARCH" in
    x86_64|amd64)
        ARCH_NAME="x86_64"
        ;;
    aarch64|arm64)
        ARCH_NAME="aarch64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo "Building minimal initramfs for $ARCH_NAME..."

# Create temp directory for initramfs
INITRAMFS_DIR=$(mktemp -d)
trap 'rm -rf "$INITRAMFS_DIR"' EXIT

# Create directory structure (no lib/modules — CONFIG_MODULES=n, all drivers built-in)
mkdir -p "$INITRAMFS_DIR"/{bin,dev,proc,sys,tmp,mnt}

# Copy tiny-init binary (pre-built by build-tiny-init.sh)
TINY_INIT="$OUTPUT_DIR/tiny-init-$ARCH_NAME"
if [ ! -f "$TINY_INIT" ]; then
    echo "ERROR: tiny-init binary not found: $TINY_INIT"
    echo "Run: ./scripts/build-tiny-init.sh $ARCH_NAME"
    exit 1
fi
cp "$TINY_INIT" "$INITRAMFS_DIR/init"
chmod 755 "$INITRAMFS_DIR/init"

# Create essential device nodes
# These are created before devtmpfs is mounted
mknod -m 622 "$INITRAMFS_DIR/dev/console" c 5 1 2>/dev/null || true
mknod -m 666 "$INITRAMFS_DIR/dev/null" c 1 3 2>/dev/null || true
# ttyS0 (COM1) for early serial output - major 4, minor 64
mknod -m 666 "$INITRAMFS_DIR/dev/ttyS0" c 4 64 2>/dev/null || true
# hvc0 (virtio console) for microvm console=hvc0 - major 229, minor 0
mknod -m 666 "$INITRAMFS_DIR/dev/hvc0" c 229 0 2>/dev/null || true
# ttyAMA0 (PL011 UART) for ARM64 virt machine - major 204, minor 64
mknod -m 666 "$INITRAMFS_DIR/dev/ttyAMA0" c 204 64 2>/dev/null || true

# Custom kernel (CONFIG_MODULES=n): all drivers are built-in.
# No kernel modules to extract — initramfs is just tiny-init + device nodes.

# Create cpio archive with zstd compression (~30% smaller than LZ4)
# Alpine's linux-virt kernel has CONFIG_RD_ZSTD=y built-in
cd "$INITRAMFS_DIR"
find . | cpio -o -H newc --quiet 2>/dev/null | zstd -19 > "$OUTPUT_DIR/initramfs-$ARCH_NAME"

# Verify zstd format (magic: 28 b5 2f fd)
MAGIC=$(od -A n -t x1 -N 4 "$OUTPUT_DIR/initramfs-$ARCH_NAME" | tr -d ' ')
if [ "$MAGIC" != "28b52ffd" ]; then
    echo "ERROR: Invalid zstd format (got $MAGIC, expected 28b52ffd)"
    exit 1
fi

# Report size (use du for portable human-readable size)
NEW_SIZE=$(du -h "$OUTPUT_DIR/initramfs-$ARCH_NAME" | cut -f1)
echo "Built minimal initramfs: $OUTPUT_DIR/initramfs-$ARCH_NAME ($NEW_SIZE)"

# Show size comparison if old initramfs exists
if [ -f "$OUTPUT_DIR/initramfs-$ARCH_NAME.alpine-backup" ]; then
    OLD_SIZE=$(du -h "$OUTPUT_DIR/initramfs-$ARCH_NAME.alpine-backup" | cut -f1)
    echo "  (was $OLD_SIZE with Alpine's stock initramfs)"
fi

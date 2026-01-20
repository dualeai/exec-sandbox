#!/bin/bash
set -euo pipefail

# Build minimal initramfs for QEMU microVM
#
# Features:
# - tiny-init: single static Rust binary (~50-100KB vs 1MB busybox)
# - zstd compression (~30% smaller than LZ4)
# - Size: ~500KB vs Alpine's 9.3 MB
#
# Expected boot time savings: 100-150ms

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
        DOCKER_PLATFORM="linux/amd64"
        ARCH_NAME="x86_64"
        ;;
    aarch64|arm64)
        DOCKER_PLATFORM="linux/arm64"
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

# Create directory structure
mkdir -p "$INITRAMFS_DIR"/{bin,dev,proc,sys,tmp,mnt,lib/modules}

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

# Extract essential kernel modules from Alpine
# Modules needed:
# - virtio_blk: for virtio block device (/dev/vda)
# - virtio_mmio: for virtio-serial on virt machine type (aarch64)
# - virtio_net: for network device (gvproxy networking)
# - virtio_balloon: for memory balloon (snapshot size optimization)
# - ext4 + dependencies (jbd2, mbcache, crc16, crc32c): for ext4 filesystem
# - zram + lz4: for compressed swap (memory optimization)
#
# Modules are extracted UNCOMPRESSED and STRIPPED for faster boot:
# - Outer zstd compression handles size reduction
# - Skips userspace gzip decompression during boot
# - Strip removes debug symbols (~8% reduction)
echo "Extracting kernel modules..."
docker run --rm --platform "$DOCKER_PLATFORM" alpine:3.21 sh -c "
    apk add --no-cache linux-virt binutils >/dev/null 2>&1
    # Get kernel version
    KVER=\$(ls /lib/modules/)
    # Create temp dir for processing
    TMPDIR=\$(mktemp -d)
    cd /lib/modules/\$KVER
    # Copy required modules
    for mod in \
        kernel/drivers/block/virtio_blk.ko.gz \
        kernel/drivers/virtio/virtio_mmio.ko.gz \
        kernel/drivers/virtio/virtio_balloon.ko.gz \
        kernel/net/core/failover.ko.gz \
        kernel/drivers/net/net_failover.ko.gz \
        kernel/drivers/net/virtio_net.ko.gz \
        kernel/fs/ext4/ext4.ko.gz \
        kernel/fs/jbd2/jbd2.ko.gz \
        kernel/fs/mbcache.ko.gz \
        kernel/lib/crc16.ko.gz \
        kernel/lib/libcrc32c.ko.gz \
        kernel/crypto/crc32c_generic.ko.gz \
        kernel/drivers/block/zram/zram.ko.gz \
        kernel/lib/lz4/lz4_compress.ko.gz \
        kernel/crypto/lz4.ko.gz
    do
        mkdir -p \$TMPDIR/\$(dirname \$mod)
        cp \$mod \$TMPDIR/\$mod 2>/dev/null || true
    done
    # Decompress modules
    find \$TMPDIR -name '*.ko.gz' -exec gunzip -f {} \;
    # Strip debug symbols (~8% reduction)
    find \$TMPDIR -name '*.ko' -exec strip --strip-unneeded {} \;
    # Create tar archive
    cd \$TMPDIR && tar -cf - .
" | tar -xf - -C "$INITRAMFS_DIR/lib/modules/" 2>/dev/null || true

# Create the kernel version directory structure
KVER=$(docker run --rm --platform "$DOCKER_PLATFORM" alpine:3.21 sh -c "apk add --no-cache linux-virt >/dev/null 2>&1; ls /lib/modules/")
if [ -d "$INITRAMFS_DIR/lib/modules/kernel" ]; then
    mkdir -p "$INITRAMFS_DIR/lib/modules/$KVER"
    mv "$INITRAMFS_DIR/lib/modules/kernel" "$INITRAMFS_DIR/lib/modules/$KVER/"
    mv "$INITRAMFS_DIR/lib/modules/modules."* "$INITRAMFS_DIR/lib/modules/$KVER/" 2>/dev/null || true
fi

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

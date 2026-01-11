#!/bin/bash
set -euo pipefail

# Build minimal initramfs for QEMU microVM
#
# Features:
# - Static busybox (for mount, switch_root)
# - Minimal init script (~15 lines)
# - LZ4 compression (5x faster than gzip)
# - Size: ~1-2 MB vs Alpine's 9.3 MB
#
# Expected boot time savings: 50-100ms

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
trap "rm -rf $INITRAMFS_DIR" EXIT

# Create directory structure
mkdir -p "$INITRAMFS_DIR"/{bin,dev,proc,sys,tmp,mnt,lib/modules}

# Get static busybox from Alpine
# Install busybox-static package which provides a fully static binary
docker run --rm --platform "$DOCKER_PLATFORM" alpine:3.21 sh -c "
    apk add --no-cache busybox-static >/dev/null 2>&1
    cat /bin/busybox.static
" > "$INITRAMFS_DIR/bin/busybox"
chmod 755 "$INITRAMFS_DIR/bin/busybox"

# Create busybox symlinks for commands used in init script
# Note: mkswap and swapon added for zram swap setup
for cmd in sh mount umount switch_root sleep usleep insmod modprobe gzip uname rm ls cat mkdir basename ln head mkswap swapon awk; do
    ln -s busybox "$INITRAMFS_DIR/bin/$cmd"
done

# Copy minimal init script
cp "$IMAGES_DIR/minimal-init.sh" "$INITRAMFS_DIR/init"
chmod 755 "$INITRAMFS_DIR/init"

# Create essential device nodes
# These are created before devtmpfs is mounted
mknod -m 622 "$INITRAMFS_DIR/dev/console" c 5 1 2>/dev/null || true
mknod -m 666 "$INITRAMFS_DIR/dev/null" c 1 3 2>/dev/null || true

# Extract essential kernel modules from Alpine
# Modules needed:
# - virtio_blk: for virtio block device (/dev/vda)
# - virtio_mmio: for virtio-serial on virt machine type (aarch64)
# - virtio_balloon: for memory balloon (snapshot size optimization)
# - ext4 + dependencies (jbd2, mbcache, crc16, crc32c): for ext4 filesystem
# - zram + lz4: for compressed swap (memory optimization)
echo "Extracting kernel modules..."
docker run --rm --platform "$DOCKER_PLATFORM" alpine:3.21 sh -c "
    apk add --no-cache linux-virt >/dev/null 2>&1
    # Get kernel version
    KVER=\$(ls /lib/modules/)
    # Create a tar of the required modules
    cd /lib/modules/\$KVER
    tar -cf - \
        kernel/drivers/block/virtio_blk.ko.gz \
        kernel/drivers/virtio/virtio_mmio.ko.gz \
        kernel/drivers/virtio/virtio_balloon.ko.gz \
        kernel/fs/ext4/ext4.ko.gz \
        kernel/fs/jbd2/jbd2.ko.gz \
        kernel/fs/mbcache.ko.gz \
        kernel/lib/crc16.ko.gz \
        kernel/lib/libcrc32c.ko.gz \
        kernel/crypto/crc32c_generic.ko.gz \
        kernel/drivers/block/zram/zram.ko.gz \
        kernel/lib/lz4/lz4_compress.ko.gz \
        kernel/lib/lz4/lz4_decompress.ko.gz \
        kernel/lib/lz4/lz4hc_compress.ko.gz \
        kernel/crypto/lz4.ko.gz \
        modules.dep modules.alias modules.symbols 2>/dev/null || true
" | tar -xf - -C "$INITRAMFS_DIR/lib/modules/" 2>/dev/null || true

# Create the kernel version directory structure
KVER=$(docker run --rm --platform "$DOCKER_PLATFORM" alpine:3.21 sh -c "apk add --no-cache linux-virt >/dev/null 2>&1; ls /lib/modules/")
if [ -d "$INITRAMFS_DIR/lib/modules/kernel" ]; then
    mkdir -p "$INITRAMFS_DIR/lib/modules/$KVER"
    mv "$INITRAMFS_DIR/lib/modules/kernel" "$INITRAMFS_DIR/lib/modules/$KVER/"
    mv "$INITRAMFS_DIR/lib/modules/modules."* "$INITRAMFS_DIR/lib/modules/$KVER/" 2>/dev/null || true
fi

# Create cpio archive with LZ4 compression (5x faster decompression than gzip)
# Alpine's linux-virt kernel has CONFIG_RD_LZ4=y built-in
# -l flag creates legacy LZ4 format required by Linux kernel
cd "$INITRAMFS_DIR"
find . | cpio -o -H newc --quiet 2>/dev/null | lz4 -9 -l > "$OUTPUT_DIR/initramfs-$ARCH_NAME"

# Report size comparison
NEW_SIZE=$(ls -lh "$OUTPUT_DIR/initramfs-$ARCH_NAME" | awk '{print $5}')
echo "Built minimal initramfs: $OUTPUT_DIR/initramfs-$ARCH_NAME ($NEW_SIZE)"

# Show size comparison if old initramfs exists
if [ -f "$OUTPUT_DIR/initramfs-$ARCH_NAME.alpine-backup" ]; then
    OLD_SIZE=$(ls -lh "$OUTPUT_DIR/initramfs-$ARCH_NAME.alpine-backup" | awk '{print $5}')
    echo "  (was $OLD_SIZE with Alpine's stock initramfs)"
fi

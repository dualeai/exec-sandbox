#!/bin/busybox sh
# Minimal init for QEMU microVM
# Replaces Alpine's full initramfs init with a minimal script
# Expected boot time savings: 50-100ms

# Mount essential virtual filesystems
/bin/busybox mount -t devtmpfs devtmpfs /dev
/bin/busybox mount -t proc proc /proc
/bin/busybox mount -t sysfs sysfs /sys
/bin/busybox mount -t tmpfs -o size=128M tmpfs /tmp

# Load kernel modules
KVER=$(/bin/busybox uname -r)

# Helper function to load a gzipped module
load_module() {
    local mod_path=$1
    local mod_name=$2
    if [ -f "$mod_path" ]; then
        /bin/busybox gzip -d -c "$mod_path" > "/tmp/$mod_name.ko"
        /bin/busybox insmod "/tmp/$mod_name.ko" 2>/dev/null
        /bin/busybox rm "/tmp/$mod_name.ko"
    fi
}

# Load virtio drivers
# virtio_mmio is needed for virtio-serial on virt machine type (aarch64)
load_module "/lib/modules/$KVER/kernel/drivers/virtio/virtio_mmio.ko.gz" "virtio_mmio"
# virtio_blk is needed for /dev/vda
load_module "/lib/modules/$KVER/kernel/drivers/block/virtio_blk.ko.gz" "virtio_blk"
# virtio_balloon is needed for memory reclamation before snapshots
load_module "/lib/modules/$KVER/kernel/drivers/virtio/virtio_balloon.ko.gz" "virtio_balloon"

# Load ext4 and its dependencies (order matters)
load_module "/lib/modules/$KVER/kernel/lib/crc16.ko.gz" "crc16"
load_module "/lib/modules/$KVER/kernel/crypto/crc32c_generic.ko.gz" "crc32c_generic"
load_module "/lib/modules/$KVER/kernel/lib/libcrc32c.ko.gz" "libcrc32c"
load_module "/lib/modules/$KVER/kernel/fs/mbcache.ko.gz" "mbcache"
load_module "/lib/modules/$KVER/kernel/fs/jbd2/jbd2.ko.gz" "jbd2"
load_module "/lib/modules/$KVER/kernel/fs/ext4/ext4.ko.gz" "ext4"

# Load zram and lz4 compression for compressed swap
# This effectively extends available memory by 2-3x with minimal CPU overhead
load_module "/lib/modules/$KVER/kernel/lib/lz4/lz4_compress.ko.gz" "lz4_compress"
load_module "/lib/modules/$KVER/kernel/lib/lz4/lz4_decompress.ko.gz" "lz4_decompress"
load_module "/lib/modules/$KVER/kernel/crypto/lz4.ko.gz" "lz4"
load_module "/lib/modules/$KVER/kernel/drivers/block/zram/zram.ko.gz" "zram"

# Configure zram compressed swap if module loaded successfully
if [ -e /sys/block/zram0 ]; then
    # Use lz4 for low-latency (2M IOPS vs 820K for zstd)
    echo "lz4" > /sys/block/zram0/comp_algorithm 2>/dev/null || \
    echo "lzo" > /sys/block/zram0/comp_algorithm 2>/dev/null || true

    # Size: 50% of RAM (with lz4's 2.6x ratio, effectively 1.3x more memory)
    MEM_KB=$(/bin/busybox awk '/MemTotal/{print $2}' /proc/meminfo)
    ZRAM_SIZE=$((MEM_KB * 512))  # 50% in bytes (KB * 1024 / 2)
    echo "$ZRAM_SIZE" > /sys/block/zram0/disksize 2>/dev/null || true

    # Create and enable swap with high priority
    /bin/busybox mkswap /dev/zram0 >/dev/null 2>&1
    /bin/busybox swapon -p 100 /dev/zram0 2>/dev/null || true

    # Optimize VM settings for compressed swap
    echo 0 > /proc/sys/vm/page-cluster 2>/dev/null || true    # Disable readahead
    echo 100 > /proc/sys/vm/swappiness 2>/dev/null || true    # Prefer swap over cache
fi

# Wait for virtio block device to appear (up to 400ms with 20ms intervals)
# Devices typically appear within 10-30ms with virtio-mmio
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -b /dev/vda ] && break
    /bin/busybox usleep 20000
done

# Wait for virtio-serial ports to appear (up to 400ms with 20ms intervals)
# These appear after virtio_mmio is loaded
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -d /sys/class/virtio-ports ] && [ "$(ls /sys/class/virtio-ports/ 2>/dev/null)" ] && break
    /bin/busybox usleep 20000
done

# Create virtio-ports symlinks (normally done by udev/mdev)
# These map vportXpY -> /dev/virtio-ports/<name> based on sysfs
/bin/busybox mkdir -p /dev/virtio-ports
for vport in /sys/class/virtio-ports/vport*; do
    if [ -d "$vport" ]; then
        dev_name=$(/bin/busybox basename "$vport")
        port_name=$(/bin/busybox cat "$vport/name" 2>/dev/null)
        if [ -n "$port_name" ]; then
            /bin/busybox ln -sf "../$dev_name" "/dev/virtio-ports/$port_name"
        fi
    fi
done

# Mount root filesystem
/bin/busybox mount -t ext4 -o rw,noatime /dev/vda /mnt

# Pivot to real root and exec guest-agent
cd /mnt
/bin/busybox mount --move /dev dev
/bin/busybox mount --move /proc proc
/bin/busybox mount --move /sys sys
/bin/busybox mount --move /tmp tmp
exec /bin/busybox switch_root /mnt /usr/local/bin/guest-agent

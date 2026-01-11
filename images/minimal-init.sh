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

# Load ext4 and its dependencies (order matters)
load_module "/lib/modules/$KVER/kernel/lib/crc16.ko.gz" "crc16"
load_module "/lib/modules/$KVER/kernel/crypto/crc32c_generic.ko.gz" "crc32c_generic"
load_module "/lib/modules/$KVER/kernel/lib/libcrc32c.ko.gz" "libcrc32c"
load_module "/lib/modules/$KVER/kernel/fs/mbcache.ko.gz" "mbcache"
load_module "/lib/modules/$KVER/kernel/fs/jbd2/jbd2.ko.gz" "jbd2"
load_module "/lib/modules/$KVER/kernel/fs/ext4/ext4.ko.gz" "ext4"

# Wait for virtio block device to appear (up to 1 second)
for i in 1 2 3 4 5 6 7 8 9 10; do
    [ -b /dev/vda ] && break
    /bin/busybox sleep 0.1
done

# Wait for virtio-serial ports to appear (up to 2 seconds)
# These appear after virtio_mmio is loaded
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -d /sys/class/virtio-ports ] && [ "$(ls /sys/class/virtio-ports/ 2>/dev/null)" ] && break
    /bin/busybox sleep 0.1
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

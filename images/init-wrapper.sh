#!/bin/sh
# Minimal init wrapper for QEMU microVMs
# Alpine container images lack init system and essential filesystem mounts
# This script provides minimal initialization before starting guest-agent

# Mount essential virtual filesystems
# These are required for the guest-agent to function properly
# || true allows idempotent remounts (already mounted = success)
mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sys /sys 2>/dev/null || true
mount -t devtmpfs dev /dev 2>/dev/null || true
# Tmpfs size matches constants.TMPFS_SIZE_MB (128MB default)
mount -t tmpfs -o size=128M tmpfs /tmp 2>/dev/null || true

# Load virtio kernel modules
# virtio_console: for virtio-serial support (TCG mode)
# virtio_net: for network device (TCP guest agent communication)
# virtio_pci: PCI bus support for virtio devices
# Silent failure if modules already built-in or not available
modprobe virtio_pci 2>/dev/null || true
modprobe virtio_net 2>/dev/null || true
modprobe virtio_console 2>/dev/null || true

# Set PATH with standard directories
# uv binary at /usr/local/bin/uv
# Python executables at /usr/local/bin (via UV_PYTHON_BIN_DIR in Dockerfile)
# bun at /usr/local/bin/bun
export PATH="/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Disable uv cache for ephemeral VMs (rootfs is read-only)
export UV_NO_CACHE=1

# Wait for network interface (up to 2 seconds)
# QEMU user-mode networking auto-configures via DHCP
for i in 1 2 3 4 5 6 7 8 9 10; do
    if [ -d /sys/class/net/eth0 ]; then
        # Give DHCP a moment to complete
        sleep 0.2
        break
    fi
    sleep 0.2
done

# Run network configuration script (for gvproxy mode)
/etc/local.d/network.start

# Execute guest-agent as PID 1
# Using exec replaces this shell process, ensuring guest-agent is PID 1
exec /usr/local/bin/guest-agent

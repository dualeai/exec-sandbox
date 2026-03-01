#!/usr/bin/env bash
# Build and install QEMU from source for CI.
#
# Ubuntu 24.04 ships QEMU 8.2.2 which has a deterministic ARM64 TCG crash
# in regime_is_user() (target/arm/internals.h). Fixed upstream in QEMU 9.0.4
# (commit 1505b651fdbd, Peter Maydell). Not backported to Ubuntu Noble.
# We build from source on both architectures for consistency.
#
# Supports ccache for incremental compilation caching. QEMU's configure wrapper
# breaks meson's ccache auto-detection, so we pass --cc="ccache gcc" explicitly
# when ccache is in PATH. In CI, ~/.ccache is persisted via actions/cache.
#
# Usage:
#   QEMU_VERSION=10.2.1 ./scripts/build-qemu.sh          # auto-detect arch
#   QEMU_VERSION=10.2.1 ./scripts/build-qemu.sh x86_64   # explicit arch
#   QEMU_VERSION=10.2.1 ./scripts/build-qemu.sh aarch64  # explicit arch
#
# Environment:
#   QEMU_VERSION   Required. Version to build (e.g. 10.2.1).
#   QEMU_PREFIX    Build output prefix. Default: ~/qemu-build

set -euo pipefail

QEMU_VERSION="${QEMU_VERSION:?QEMU_VERSION must be set}"
QEMU_PREFIX="${QEMU_PREFIX:-$HOME/qemu-build}"

# ============================================================================
# Architecture detection
# ============================================================================

detect_arch() {
    case "$(uname -m)" in
        x86_64)  echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
    esac
}

ARCH="${1:-$(detect_arch)}"

case "$ARCH" in
    x86_64)  TARGET_LIST="x86_64-softmmu" ;;
    aarch64) TARGET_LIST="aarch64-softmmu" ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

# ============================================================================
# ccache detection
# ============================================================================
# QEMU's configure wrapper breaks meson's built-in ccache auto-detection
# (see https://www.qemu.org/docs/master/devel/build-environment.html).
# When ccache is available, we pass it explicitly via --cc to configure.

CCACHE_ARGS=()
if command -v ccache &>/dev/null; then
    # Configure ccache: compress objects, cap cache size to avoid unbounded growth
    export CCACHE_COMPRESS=1
    export CCACHE_MAXSIZE=500M
    CCACHE_ARGS=(--cc="ccache gcc")
    ccache --zero-stats 2>/dev/null || true
    echo "ccache detected, enabling compilation caching (max ${CCACHE_MAXSIZE})"
fi

# ============================================================================
# Build from source
# ============================================================================

echo "Building QEMU $QEMU_VERSION for $ARCH (target-list=$TARGET_LIST)..."

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

cd "$WORK_DIR"
curl -fsSL "https://download.qemu.org/qemu-${QEMU_VERSION}.tar.xz" | tar xJ
cd "qemu-${QEMU_VERSION}"

# --target-list: build only the system emulator for the current host arch.
#   QEMU supports ~50 targets (x86_64, aarch64, riscv64, mips, ...); building
#   all is slow and unnecessary. We only need the native-arch emulator for TCG.
# --without-default-features: all auto-detected features default to disabled,
#   so we must explicitly enable everything we need.
# --enable-fdt: device tree support, required for aarch64-softmmu (virt machine),
#   harmless for x86_64. Needs libfdt-dev.
# --enable-kvm: KVM acceleration support.
# --enable-linux-io-uring: io_uring async I/O backend for block drives (aio=io_uring).
#   Needs liburing-dev (>= 0.3). Falls back to aio=threads if unavailable at runtime.
# --enable-seccomp: seccomp sandbox for QEMU process (-sandbox on).
#   Needs libseccomp-dev (>= 2.3.0).
# --enable-slirp: user-mode networking backend. Needs libslirp-dev.
# --enable-tools: qemu-img, qemu-storage-daemon, qemu-nbd.
#   We use qemu-img for overlay/snapshot ops and qemu-storage-daemon for fast
#   overlay creation via QMP. Building from source ensures version parity.
# --enable-vhost-kernel: vhost kernel backend (Linux /dev/vhost-net).
#   Required by vhost-net — without it, --enable-vhost-net is a silent no-op
#   because vhost_net needs (vhost_kernel OR vhost_user OR vhost_vdpa).
# --enable-vhost-net: vhost-net kernel acceleration for virtio-net.
# --disable-docs: skip building documentation (needs sphinx).
# --disable-user: skip linux-user/bsd-user emulators (we only need system emulation).
# See: https://wiki.qemu.org/Hosts/Linux
./configure \
    --prefix="$QEMU_PREFIX" \
    --target-list="$TARGET_LIST" \
    --without-default-features \
    "${CCACHE_ARGS[@]}" \
    --enable-fdt \
    --enable-kvm \
    --enable-linux-io-uring \
    --enable-seccomp \
    --enable-slirp \
    --enable-tools \
    --enable-vhost-kernel \
    --enable-vhost-net \
    --disable-docs \
    --disable-user

make -j"$(nproc)"
make install

# Show ccache hit/miss stats for CI debugging
if command -v ccache &>/dev/null; then
    echo ""
    echo "=== ccache stats ==="
    ccache --show-stats 2>/dev/null || true
fi

echo "QEMU $QEMU_VERSION installed to $QEMU_PREFIX"

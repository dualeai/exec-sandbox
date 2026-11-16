#!/bin/bash
# Build VM images (kernel + qcow2) using Docker
#
# Orchestrates the build of all VM image components:
#   1. guest-agent + tiny-init (Rust binaries, in parallel)
#   2. kernel + initramfs + vmlinux (extract-kernel.sh orchestrates all three)
#   3. qcow2 disk images (python, node, raw variants)
#
# Artifact dependency graph:
#
#   tiny-init src → tiny-init binary → initramfs (QEMU -initrd, NOT inside qcow2)
#   guest-agent src → guest-agent binary → embedded in EROFS qcow2
#   kernel config fragment → custom vmlinuz (CONFIG_MODULES=n, all drivers built-in)
#   rootfs packages (Alpine, Python, Node) → EROFS qcow2 (full rebuild only)
#
# Caching: each artifact except vmlinux has a .hash sidecar file (content-addressable
# input key, NOT an output checksum). When inputs match, the build step is skipped.
# EROFS is read-only, so any change triggers a full qcow2 rebuild.
#
# Compilers, mkfs.erofs and qemu-img run inside Linux containers, so builds are
# the same on macOS and Linux hosts. initramfs packing (cpio, zstd) and vmlinux
# extraction run on the host.
#
# Usage:
#   ./scripts/build-images.sh              # Build for all architectures
#   ./scripts/build-images.sh x86_64       # Build for specific arch
#   ./scripts/build-images.sh aarch64      # Build for specific arch
#   ./scripts/build-images.sh aarch64 python  # Build specific variant for arch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/images/dist"


build_for_arch() {
    local arch=$1
    local variant=$2

    echo "=== Building for $arch (variant: $variant) ==="

    # Step 1: Build guest-agent and tiny-init in parallel
    echo "[$arch] Building guest-agent + tiny-init..."
    "$SCRIPT_DIR/build-guest-agent.sh" "$arch" &
    local pid_agent=$!
    "$SCRIPT_DIR/build-tiny-init.sh" "$arch" &
    local pid_init=$!

    # Wait for both before failing: a build left running would outlive this script
    local failed=0
    wait $pid_agent || { echo "[$arch] Guest-agent build failed" >&2; failed=1; }
    wait $pid_init || { echo "[$arch] tiny-init build failed" >&2; failed=1; }
    [ $failed -eq 0 ] || return 1

    # Step 2: Build kernel + initramfs + vmlinux (extract-kernel.sh orchestrates)
    # Kernel build (build-kernel.sh) runs first, then initramfs + vmlinux extraction.
    echo "[$arch] Building kernel + initramfs..."
    "$SCRIPT_DIR/extract-kernel.sh" "$arch" || { echo "[$arch] Kernel build failed" >&2; return 1; }

    # Step 3: Build qcow2 images (parallelized inside build-qcow2.sh)
    echo "[$arch] Building qcow2 images..."
    "$SCRIPT_DIR/build-qcow2.sh" "$variant" "$arch"
}

main() {
    local target="${1:-all}"
    local variant="${2:-all}"

    # Check Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker is required. Install from https://docker.com" >&2
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    echo "Preparing build environment..."

    if [ "$target" = "all" ]; then
        # Build both architectures in parallel
        build_for_arch "x86_64" "$variant" &
        local pid_x86=$!
        build_for_arch "aarch64" "$variant" &
        local pid_arm=$!

        # Wait for both to complete
        local failed=0
        wait $pid_x86 || failed=1
        wait $pid_arm || failed=1

        if [ $failed -ne 0 ]; then
            echo "Build failed" >&2
            exit 1
        fi
    else
        build_for_arch "$target" "$variant"
    fi

    echo ""
    echo "=== Build Complete ==="
    ls -lh "$OUTPUT_DIR/"
}

main "$@"

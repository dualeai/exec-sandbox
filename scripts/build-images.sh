#!/bin/bash
# Build VM images (kernel + qcow2) using Docker
#
# Orchestrates the build of all VM image components:
#   1. guest-agent (Rust binary)
#   2. kernel + initramfs (from Alpine)
#   3. qcow2 disk images (python, node, raw variants)
#
# All build commands run inside Linux containers with the repo mounted.
# This ensures consistent builds across macOS and Linux hosts.
#
# Usage:
#   ./scripts/build-images.sh              # Build for all architectures
#   ./scripts/build-images.sh x86_64       # Build for specific arch
#   ./scripts/build-images.sh aarch64      # Build for specific arch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/images/dist"

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *) echo "Unsupported arch: $(uname -m)" >&2; exit 1 ;;
    esac
}

build_for_arch() {
    local arch=$1

    echo "=== Building for $arch ==="

    # Step 1+2: Build guest-agent and extract kernel in parallel
    echo "[$arch] Building guest-agent + extracting kernel..."
    "$SCRIPT_DIR/build-guest-agent.sh" "$arch" &
    local pid_agent=$!
    "$SCRIPT_DIR/extract-kernel.sh" "$arch" &
    local pid_kernel=$!

    wait $pid_agent || { echo "[$arch] Guest-agent build failed" >&2; return 1; }
    wait $pid_kernel || { echo "[$arch] Kernel extraction failed" >&2; return 1; }

    # Step 3: Build qcow2 images (parallelized inside build-qcow2.sh)
    echo "[$arch] Building qcow2 images..."
    "$SCRIPT_DIR/build-qcow2.sh" all "$arch"
}

main() {
    local target="${1:-all}"

    # Check Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker is required. Install from https://docker.com" >&2
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    echo "Preparing build environment..."

    if [ "$target" = "all" ]; then
        # Build both architectures in parallel
        build_for_arch "x86_64" &
        local pid_x86=$!
        build_for_arch "aarch64" &
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
        build_for_arch "$target"
    fi

    echo ""
    echo "=== Build Complete ==="
    ls -lh "$OUTPUT_DIR/"
}

main "$@"

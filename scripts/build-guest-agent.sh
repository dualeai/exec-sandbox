#!/bin/bash
# Build guest-agent Rust binary using Docker
#
# Uses Docker to ensure consistent builds across macOS and Linux hosts.
# Produces a statically-linked musl binary.
#
# Usage:
#   ./scripts/build-guest-agent.sh              # Build for current arch
#   ./scripts/build-guest-agent.sh x86_64       # Build for x86_64
#   ./scripts/build-guest-agent.sh aarch64      # Build for aarch64
#   ./scripts/build-guest-agent.sh all          # Build for both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/images/dist"
RUST_VERSION="${RUST_VERSION:-1.83}"

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *) echo "Unsupported arch: $(uname -m)" >&2; exit 1 ;;
    esac
}

# =============================================================================
# Cache helpers - content-addressable build caching via .hash sidecar files
# =============================================================================

# Compute hash for guest-agent inputs
compute_hash() {
    local arch=$1
    (
        echo "arch=$arch"
        echo "rust=$RUST_VERSION"
        cat "$REPO_ROOT/guest-agent/Cargo.lock" 2>/dev/null || true
        cat "$REPO_ROOT/guest-agent/Cargo.toml" 2>/dev/null || true
        find "$REPO_ROOT/guest-agent/src" -type f -name "*.rs" -print0 2>/dev/null | \
            sort -z | xargs -0 cat 2>/dev/null || true
    ) | sha256sum | cut -d' ' -f1
}

# Check if output is up-to-date (hash matches)
cache_hit() {
    local output_file=$1
    local current_hash=$2
    local hash_file="${output_file}.hash"

    if [ -f "$output_file" ] && [ -f "$hash_file" ]; then
        local cached_hash
        cached_hash=$(cat "$hash_file" 2>/dev/null || echo "")
        [ "$cached_hash" = "$current_hash" ]
    else
        return 1
    fi
}

# Save hash after successful build
save_hash() {
    local output_file=$1
    local hash=$2
    echo "$hash" > "${output_file}.hash"
}

# =============================================================================
# Build function
# =============================================================================

build_for_arch() {
    local arch=$1
    local rust_target="${arch}-unknown-linux-musl"
    local output_file="$OUTPUT_DIR/guest-agent-linux-$arch"

    # Check cache
    local current_hash
    current_hash=$(compute_hash "$arch")

    if cache_hit "$output_file" "$current_hash"; then
        echo "Guest-agent up-to-date: $output_file (cache hit)"
        return 0
    fi

    echo "Building guest-agent for $arch (Rust $RUST_VERSION)..."

    mkdir -p "$OUTPUT_DIR"

    docker run --rm \
        -v "$REPO_ROOT:/workspace" \
        -w /workspace/guest-agent \
        --platform linux/$([ "$arch" = "aarch64" ] && echo "arm64" || echo "amd64") \
        rust:$RUST_VERSION-slim \
        bash -c "
            rustup target add $rust_target
            cargo build --release --target $rust_target
            cp target/$rust_target/release/guest-agent /workspace/images/dist/guest-agent-linux-$arch
        "

    save_hash "$output_file" "$current_hash"

    local size
    size=$(du -h "$output_file" | cut -f1)
    echo "Built: guest-agent-linux-$arch ($size)"
}

main() {
    local target="${1:-$(detect_arch)}"

    # Check Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker is required. Install from https://docker.com" >&2
        exit 1
    fi

    if [ "$target" = "all" ]; then
        build_for_arch "x86_64" &
        local pid_x86=$!
        build_for_arch "aarch64" &
        local pid_arm=$!

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
}

main "$@"

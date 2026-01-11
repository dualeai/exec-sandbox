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

# Buildx cache configuration (for CI)
# Set BUILDX_CACHE_FROM and BUILDX_CACHE_TO to enable external caching
# Example: BUILDX_CACHE_FROM="type=gha" BUILDX_CACHE_TO="type=gha,mode=max"
BUILDX_CACHE_FROM="${BUILDX_CACHE_FROM:-}"
BUILDX_CACHE_TO="${BUILDX_CACHE_TO:-}"

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

    local docker_platform="linux/$([ "$arch" = "aarch64" ] && echo "arm64" || echo "amd64")"
    # Scope includes arch and Rust version to avoid cache collisions
    local cache_scope="guest-agent-rust${RUST_VERSION}-${arch}"
    local cache_args=()
    [ -n "$BUILDX_CACHE_FROM" ] && cache_args+=(--cache-from "$BUILDX_CACHE_FROM,scope=$cache_scope")
    [ -n "$BUILDX_CACHE_TO" ] && cache_args+=(--cache-to "$BUILDX_CACHE_TO,scope=$cache_scope")

    # Use buildx with cache mounts for cargo registry and target
    DOCKER_BUILDKIT=1 docker buildx build \
        --platform "$docker_platform" \
        --output "type=local,dest=$OUTPUT_DIR" \
        --build-arg RUST_VERSION="$RUST_VERSION" \
        --build-arg RUST_TARGET="$rust_target" \
        --build-arg ARCH="$arch" \
        ${cache_args[@]+"${cache_args[@]}"} \
        -f - "$REPO_ROOT" <<'DOCKERFILE'
# syntax=docker/dockerfile:1.4
ARG RUST_VERSION
FROM rust:${RUST_VERSION}-slim
ARG RUST_TARGET
ARG ARCH
WORKDIR /workspace
COPY guest-agent/ ./guest-agent/
RUN rustup target add ${RUST_TARGET}
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/workspace/guest-agent/target,sharing=locked \
    cd guest-agent && \
    cargo build --release --target ${RUST_TARGET} && \
    cp target/${RUST_TARGET}/release/guest-agent /guest-agent-linux-${ARCH}
FROM scratch
ARG ARCH
COPY --from=0 /guest-agent-linux-* .
DOCKERFILE

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

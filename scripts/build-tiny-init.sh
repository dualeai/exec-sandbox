#!/bin/bash
# Build tiny-init Rust binary using Docker
#
# Uses Docker with Rust cross-compilation (no QEMU emulation needed).
# Produces statically-linked musl binaries optimized for size.
#
# Usage:
#   ./scripts/build-tiny-init.sh              # Build for current arch
#   ./scripts/build-tiny-init.sh x86_64       # Build for x86_64
#   ./scripts/build-tiny-init.sh aarch64      # Build for aarch64
#   ./scripts/build-tiny-init.sh all          # Build for both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/images/dist"
RUST_VERSION="${RUST_VERSION:?RUST_VERSION must be set (exported by root Makefile)}"

# NOTE: Check periodically if -Zbuild-std and panic_immediate_abort have been stabilized.
# When stable, we can add: RUSTFLAGS="-Cpanic=immediate-abort" cargo build -Zbuild-std=std,panic_abort
# This would reduce binary size by ~30-50%. Track: https://github.com/rust-lang/rust/issues/115022

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
# Cache helpers
# =============================================================================

compute_hash() {
    local arch=$1
    (
        echo "arch=$arch"
        echo "rust=$RUST_VERSION"
        cat "$REPO_ROOT/tiny-init/Cargo.lock" 2>/dev/null || true
        cat "$REPO_ROOT/tiny-init/Cargo.toml" 2>/dev/null || true
        find "$REPO_ROOT/tiny-init/src" -type f -name "*.rs" -print0 2>/dev/null | \
            sort -z | xargs -0 cat 2>/dev/null || true
    ) | sha256sum | cut -d' ' -f1
}

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
    local output_file="$OUTPUT_DIR/tiny-init-$arch"

    local current_hash
    current_hash=$(compute_hash "$arch")

    if cache_hit "$output_file" "$current_hash"; then
        echo "tiny-init up-to-date: $output_file (cache hit)"
        return 0
    fi

    echo "Building tiny-init for $arch (Rust $RUST_VERSION, cross-compile)..."

    mkdir -p "$OUTPUT_DIR"

    # Scope includes arch and Rust version to avoid cache collisions
    local cache_scope="tiny-init-rust${RUST_VERSION}-${arch}"
    local cache_args=()
    [ -n "$BUILDX_CACHE_FROM" ] && cache_args+=(--cache-from "$BUILDX_CACHE_FROM,scope=$cache_scope")
    [ -n "$BUILDX_CACHE_TO" ] && cache_args+=(--cache-to "$BUILDX_CACHE_TO,scope=$cache_scope")

    # Build using buildx with cross-compilation (NO --platform flag to avoid QEMU)
    # The Dockerfile downloads musl.cc cross-toolchains for cross-arch builds
    DOCKER_BUILDKIT=1 docker buildx build \
        --output "type=local,dest=$OUTPUT_DIR" \
        --build-arg RUST_VERSION="$RUST_VERSION" \
        --build-arg RUST_TARGET="$rust_target" \
        --build-arg ARCH="$arch" \
        ${cache_args[@]+"${cache_args[@]}"} \
        -f - "$REPO_ROOT" <<'DOCKERFILE'
# syntax=docker/dockerfile:1.4
ARG RUST_VERSION
FROM rust:${RUST_VERSION}-slim AS builder
ARG RUST_TARGET
ARG ARCH
WORKDIR /workspace

# Install wget for downloading cross-toolchain
RUN apt-get update -qq && apt-get install -qq -y wget >/dev/null 2>&1

# Download and setup cross-compiler if needed (cached layer)
RUN --mount=type=cache,target=/tmp/toolchain-cache,sharing=locked \
    set -e && \
    HOST_ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ] && [ "$HOST_ARCH" != "x86_64" ]; then \
        if [ ! -f /tmp/toolchain-cache/x86_64-linux-musl-cross.tgz ]; then \
            wget -q https://musl.cc/x86_64-linux-musl-cross.tgz -O /tmp/toolchain-cache/x86_64-linux-musl-cross.tgz; \
        fi && \
        tar -xzf /tmp/toolchain-cache/x86_64-linux-musl-cross.tgz -C /usr/local; \
    elif [ "$ARCH" = "aarch64" ] && [ "$HOST_ARCH" != "aarch64" ]; then \
        if [ ! -f /tmp/toolchain-cache/aarch64-linux-musl-cross.tgz ]; then \
            wget -q https://musl.cc/aarch64-linux-musl-cross.tgz -O /tmp/toolchain-cache/aarch64-linux-musl-cross.tgz; \
        fi && \
        tar -xzf /tmp/toolchain-cache/aarch64-linux-musl-cross.tgz -C /usr/local; \
    fi

# Add Rust target
RUN rustup target add ${RUST_TARGET}

# Copy source
COPY tiny-init/ ./tiny-init/

# Build with cross-compilation
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/workspace/tiny-init/target,sharing=locked \
    set -e && \
    HOST_ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ] && [ "$HOST_ARCH" != "x86_64" ]; then \
        export PATH="/usr/local/x86_64-linux-musl-cross/bin:$PATH" && \
        export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=x86_64-linux-musl-gcc && \
        export CC_x86_64_unknown_linux_musl=x86_64-linux-musl-gcc; \
    elif [ "$ARCH" = "aarch64" ] && [ "$HOST_ARCH" != "aarch64" ]; then \
        export PATH="/usr/local/aarch64-linux-musl-cross/bin:$PATH" && \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=aarch64-linux-musl-gcc && \
        export CC_aarch64_unknown_linux_musl=aarch64-linux-musl-gcc; \
    fi && \
    cd tiny-init && \
    cargo build --release --target ${RUST_TARGET} && \
    cp target/${RUST_TARGET}/release/tiny-init /tiny-init-${ARCH}

FROM scratch
ARG ARCH
COPY --from=builder /tiny-init-* .
DOCKERFILE

    save_hash "$output_file" "$current_hash"

    local size
    size=$(du -h "$output_file" | cut -f1)
    echo "Built: tiny-init-$arch ($size)"
}

main() {
    local target="${1:-$(detect_arch)}"

    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker is required. Install from https://docker.com" >&2
        exit 1
    fi

    if [ "$target" = "all" ]; then
        build_for_arch "x86_64"
        build_for_arch "aarch64"
    else
        build_for_arch "$target"
    fi
}

main "$@"

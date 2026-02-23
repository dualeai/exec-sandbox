#!/bin/bash
# Build custom Linux kernel for QEMU microVM using Docker BuildKit
#
# Uses Alpine's linux-virt config as base, merges our exec-sandbox.config
# fragment on top using the kernel's scripts/kconfig/merge_config.sh.
# Zero-maintenance: when Alpine upgrades the kernel, merge auto-inherits
# new defaults — we only maintain what we intentionally changed.
#
# Cross-compilation via musl.cc toolchains (same as build-guest-agent.sh).
# BuildKit cache mounts for kernel source, ccache, and toolchains.
#
# Usage:
#   ./scripts/build-kernel.sh              # Build for current arch
#   ./scripts/build-kernel.sh x86_64       # Build for x86_64
#   ./scripts/build-kernel.sh aarch64      # Build for aarch64
#   ./scripts/build-kernel.sh all          # Build for both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/images/dist"
ALPINE_VERSION="${ALPINE_VERSION:?ALPINE_VERSION must be set (exported by root Makefile)}"

# Buildx cache configuration (for CI)
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

# Get kernel package version from Alpine's APKINDEX (reused from extract-kernel.sh)
get_kernel_version() {
    local arch=$1
    curl -sf "https://dl-cdn.alpinelinux.org/alpine/v${ALPINE_VERSION}/main/${arch}/APKINDEX.tar.gz" 2>/dev/null \
        | tar -xzO APKINDEX 2>/dev/null \
        | grep -A1 "^P:linux-virt$" \
        | grep "^V:" \
        | cut -d: -f2 \
        || echo "unknown"
}

# Get upstream kernel version (strip Alpine revision suffix like -r0)
get_upstream_kernel_version() {
    local alpine_ver=$1
    # Alpine version format: 6.12.67-r0 -> upstream: 6.12.67
    echo "${alpine_ver%%-*}"
}

compute_hash() {
    local arch=$1
    local kernel_ver
    kernel_ver=$(get_kernel_version "$arch")
    (
        echo "alpine=$ALPINE_VERSION arch=$arch kernel=$kernel_ver"
        cat "$REPO_ROOT/images/kernel/exec-sandbox.config"
        cat "$SCRIPT_DIR/build-kernel.sh"
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
    local output_file="$OUTPUT_DIR/vmlinuz-$arch"

    local current_hash
    current_hash=$(compute_hash "$arch")

    if cache_hit "$output_file" "$current_hash"; then
        echo "Kernel up-to-date: $output_file (cache hit)"
        return 0
    fi

    local alpine_kernel_ver
    alpine_kernel_ver=$(get_kernel_version "$arch")
    local kernel_ver
    kernel_ver=$(get_upstream_kernel_version "$alpine_kernel_ver")

    if [ "$kernel_ver" = "unknown" ]; then
        echo "ERROR: Could not determine kernel version from Alpine APKINDEX" >&2
        return 1
    fi

    echo "Building custom kernel $kernel_ver for $arch (Alpine $ALPINE_VERSION)..."

    mkdir -p "$OUTPUT_DIR"

    # Map arch to kernel ARCH, image target, and Docker platform
    local kernel_arch kernel_image docker_platform
    case "$arch" in
        x86_64)
            kernel_arch="x86"
            kernel_image="bzImage"
            docker_platform="linux/amd64"
            ;;
        aarch64)
            kernel_arch="arm64"
            kernel_image="Image"
            docker_platform="linux/arm64"
            ;;
    esac

    # Scope includes arch and kernel version to avoid cache collisions
    local cache_scope="kernel-${kernel_ver}-${arch}"
    local cache_args=()
    [ -n "$BUILDX_CACHE_FROM" ] && cache_args+=(--cache-from "$BUILDX_CACHE_FROM,scope=$cache_scope")
    [ -n "$BUILDX_CACHE_TO" ] && cache_args+=(--cache-to "$BUILDX_CACHE_TO,scope=$cache_scope")

    # Build using buildx with cross-compilation (NO --platform flag on builder to avoid QEMU)
    DOCKER_BUILDKIT=1 docker buildx build \
        --output "type=local,dest=$OUTPUT_DIR" \
        --build-arg ALPINE_VERSION="$ALPINE_VERSION" \
        --build-arg DOCKER_PLATFORM="$docker_platform" \
        --build-arg ARCH="$arch" \
        --build-arg KERNEL_ARCH="$kernel_arch" \
        --build-arg KERNEL_IMAGE="$kernel_image" \
        --build-arg KERNEL_VERSION="$kernel_ver" \
        ${cache_args[@]+"${cache_args[@]}"} \
        -f - "$REPO_ROOT" <<'DOCKERFILE'
# syntax=docker/dockerfile:1.4

# Stage 1: Extract Alpine's linux-virt config for target arch only
ARG ALPINE_VERSION
ARG DOCKER_PLATFORM
FROM --platform=${DOCKER_PLATFORM} alpine:${ALPINE_VERSION} AS config-extractor
RUN apk add --no-cache linux-virt >/dev/null 2>&1 && cp /boot/config-*-virt /alpine-virt.config

# Stage 2: Build kernel natively with cross-compilation
FROM debian:bookworm-slim AS builder
ARG ARCH
ARG KERNEL_ARCH
ARG KERNEL_IMAGE
ARG KERNEL_VERSION

# Install kernel build dependencies
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    build-essential flex bison bc libelf-dev libssl-dev perl \
    wget ca-certificates cpio zstd lz4 ccache python3 xz-utils >/dev/null 2>&1

# Download cross-compiler if needed (cached)
RUN --mount=type=cache,target=/tmp/toolchain-cache,sharing=locked \
    set -e && \
    HOST_ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] && [ "$HOST_ARCH" != "aarch64" ]; then \
        if [ ! -f /tmp/toolchain-cache/aarch64-linux-musl-cross.tgz ]; then \
            wget -q https://musl.cc/aarch64-linux-musl-cross.tgz \
                -O /tmp/toolchain-cache/aarch64-linux-musl-cross.tgz; \
        fi && \
        tar -xzf /tmp/toolchain-cache/aarch64-linux-musl-cross.tgz -C /usr/local; \
    elif [ "$ARCH" = "x86_64" ] && [ "$HOST_ARCH" != "x86_64" ]; then \
        if [ ! -f /tmp/toolchain-cache/x86_64-linux-musl-cross.tgz ]; then \
            wget -q https://musl.cc/x86_64-linux-musl-cross.tgz \
                -O /tmp/toolchain-cache/x86_64-linux-musl-cross.tgz; \
        fi && \
        tar -xzf /tmp/toolchain-cache/x86_64-linux-musl-cross.tgz -C /usr/local; \
    fi

WORKDIR /build

# Download kernel source (cached)
RUN --mount=type=cache,target=/tmp/kernel-cache,sharing=locked \
    set -e && \
    MAJOR=$(echo $KERNEL_VERSION | cut -d. -f1) && \
    TARBALL="linux-${KERNEL_VERSION}.tar.xz" && \
    if [ ! -f "/tmp/kernel-cache/$TARBALL" ] || [ ! -s "/tmp/kernel-cache/$TARBALL" ]; then \
        rm -f "/tmp/kernel-cache/$TARBALL" && \
        wget -q "https://cdn.kernel.org/pub/linux/kernel/v${MAJOR}.x/$TARBALL" \
            -O "/tmp/kernel-cache/$TARBALL"; \
    fi && \
    tar -xf "/tmp/kernel-cache/$TARBALL" -C /build/ && \
    ln -s "/build/linux-${KERNEL_VERSION}" /build/linux

# Copy Alpine's base config (single arch, from config-extractor stage)
COPY --from=config-extractor /alpine-virt.config /tmp/alpine-virt.config

# Copy our config fragment
COPY images/kernel/exec-sandbox.config /tmp/exec-sandbox.config

# Merge configs + build kernel with persistent build tree for incremental builds.
# The cache mount preserves .o files and .cmd dependency tracking between builds,
# so `make` only recompiles files affected by config changes.
RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/build/kernel-obj,id=kernel-obj-${ARCH} \
    set -e && \
    # Populate build tree from source (skip if already matches kernel version)
    if [ ! -f /build/kernel-obj/Makefile ] || \
       [ "$(cat /build/kernel-obj/.kernel-version 2>/dev/null)" != "${KERNEL_VERSION}" ]; then \
        echo "Populating build tree (kernel ${KERNEL_VERSION})..." && \
        rm -rf /build/kernel-obj/* /build/kernel-obj/.* 2>/dev/null || true && \
        cp -a /build/linux/. /build/kernel-obj/ && \
        echo "${KERNEL_VERSION}" > /build/kernel-obj/.kernel-version; \
    fi && \
    cd /build/kernel-obj && \
    # Merge configs: Alpine base + our fragment -> .config
    cp /tmp/alpine-virt.config /tmp/base.config && \
    scripts/kconfig/merge_config.sh -m /tmp/base.config /tmp/exec-sandbox.config && \
    # Resolve dependencies + build
    HOST_ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ] && [ "$HOST_ARCH" != "x86_64" ]; then \
        export CROSS_COMPILE=x86_64-linux-musl- && \
        export PATH="/usr/local/x86_64-linux-musl-cross/bin:$PATH"; \
    elif [ "$ARCH" = "aarch64" ] && [ "$HOST_ARCH" != "aarch64" ]; then \
        export CROSS_COMPILE=aarch64-linux-musl- && \
        export PATH="/usr/local/aarch64-linux-musl-cross/bin:$PATH"; \
    fi && \
    make ARCH=${KERNEL_ARCH} ${CROSS_COMPILE:+CROSS_COMPILE=$CROSS_COMPILE} olddefconfig && \
    export KBUILD_BUILD_TIMESTAMP='' && \
    export KBUILD_BUILD_USER='exec-sandbox' && \
    export KBUILD_BUILD_HOST='buildkit' && \
    export PATH="/usr/lib/ccache:$PATH" && \
    make -j$(nproc) ARCH=${KERNEL_ARCH} ${CROSS_COMPILE:+CROSS_COMPILE=$CROSS_COMPILE} ${KERNEL_IMAGE} && \
    if [ "$KERNEL_ARCH" = "x86" ]; then \
        cp arch/x86/boot/bzImage /vmlinuz-${ARCH}; \
    else \
        cp arch/${KERNEL_ARCH}/boot/${KERNEL_IMAGE} /vmlinuz-${ARCH}; \
    fi

# Output stage
FROM scratch
ARG ARCH
COPY --from=builder /vmlinuz-* .
DOCKERFILE

    save_hash "$output_file" "$current_hash"

    local size
    size=$(du -h "$output_file" | cut -f1)
    echo "Built: vmlinuz-$arch ($size)"
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

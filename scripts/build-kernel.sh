#!/bin/bash
# Build custom Linux kernel for QEMU microVM using Docker BuildKit
#
# Kernel version + tarball sha256 come from versions.lock (managed by
# `make upgrade`). The base config is Alpine's linux-virt config, vendored
# into images/kernel/alpine-virt-<arch>.config at upgrade time; our
# exec-sandbox.config fragment is merged on top using the kernel's
# scripts/kconfig/merge_config.sh. Kernel bumps land as reviewable git
# diffs (lock + vendored configs) — builds hash only git-tracked bytes,
# so cached builds need no network.
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
LOCK_FILE="$REPO_ROOT/versions.lock"

if [ ! -f "$LOCK_FILE" ]; then
    echo "ERROR: versions.lock not found — run './scripts/upgrade-versions.sh' (or restore it from git)" >&2
    exit 1
fi
# Fail-closed: versions.lock is the single source of truth. Parsed with grep
# (never sourced — lock values must not be executable). `|| true` keeps a
# missing key from silently killing the script before the -z diagnostics.
lock_get() {
    grep -m1 "^$1=" "$LOCK_FILE" | cut -d= -f2- || true
}
KERNEL_VERSION=$(lock_get KERNEL_VERSION)
KERNEL_TARBALL_VERSION=$(lock_get KERNEL_TARBALL_VERSION)
KERNEL_SHA256=$(lock_get KERNEL_SHA256)
if [ -z "$KERNEL_VERSION" ] || [ -z "$KERNEL_TARBALL_VERSION" ] || [ -z "$KERNEL_SHA256" ]; then
    echo "ERROR: versions.lock lacks KERNEL_VERSION/KERNEL_TARBALL_VERSION/KERNEL_SHA256 — run 'make upgrade'" >&2
    exit 1
fi

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

# All hash inputs are git-tracked bytes — cached builds need no network,
# and cache keys change exactly at reviewed commits (lock or config edits).
# Lock KEY=VALUE lines only: comment lines (distribution dates) are
# documentation and must never invalidate the kernel cache.
# Keep byte-identical to compute_kernel_hash in extract-kernel.sh.
compute_hash() {
    local arch=$1
    (
        echo "arch=$arch"
        grep '^[A-Z]' "$LOCK_FILE"
        cat "$REPO_ROOT/images/kernel/exec-sandbox.config"
        cat "$REPO_ROOT/images/kernel/alpine-virt-${arch}.config"
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

    echo "Building custom kernel $KERNEL_VERSION for $arch (from versions.lock)..."

    mkdir -p "$OUTPUT_DIR"

    # Map arch to kernel ARCH and image target
    local kernel_arch kernel_image
    case "$arch" in
        x86_64)
            kernel_arch="x86"
            kernel_image="bzImage"
            ;;
        aarch64)
            kernel_arch="arm64"
            kernel_image="Image"
            ;;
    esac

    # Scope includes arch and kernel version to avoid cache collisions
    local cache_scope="kernel-${KERNEL_VERSION}-${arch}"
    local cache_args=()
    [ -n "$BUILDX_CACHE_FROM" ] && cache_args+=(--cache-from "$BUILDX_CACHE_FROM,scope=$cache_scope")
    [ -n "$BUILDX_CACHE_TO" ] && cache_args+=(--cache-to "$BUILDX_CACHE_TO,scope=$cache_scope")

    # Build using buildx with cross-compilation (NO --platform flag on builder to avoid QEMU)
    DOCKER_BUILDKIT=1 docker buildx build \
        --output "type=local,dest=$OUTPUT_DIR" \
        --build-arg ARCH="$arch" \
        --build-arg KERNEL_ARCH="$kernel_arch" \
        --build-arg KERNEL_IMAGE="$kernel_image" \
        --build-arg KERNEL_VERSION="$KERNEL_VERSION" \
        --build-arg KERNEL_TARBALL_VERSION="$KERNEL_TARBALL_VERSION" \
        --build-arg KERNEL_SHA256="$KERNEL_SHA256" \
        ${cache_args[@]+"${cache_args[@]}"} \
        -f - "$REPO_ROOT" <<'DOCKERFILE'
# syntax=docker/dockerfile:1.4

# Build kernel natively with cross-compilation
FROM debian:bookworm-slim AS builder
ARG ARCH
ARG KERNEL_ARCH
ARG KERNEL_IMAGE
ARG KERNEL_VERSION
ARG KERNEL_TARBALL_VERSION
ARG KERNEL_SHA256

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

# Download kernel source (cached), pinned by sha256 from versions.lock.
# Both the cached copy and a fresh download are verified — a corrupt or
# tampered tarball fails the build instead of being compiled.
# KERNEL_TARBALL_VERSION comes from the lock (kernel.org names first-in-series
# tarballs without .0: linux-6.19.tar.xz for Alpine's 6.19.0).
RUN --mount=type=cache,target=/tmp/kernel-cache,sharing=locked \
    set -e && \
    MAJOR=$(echo $KERNEL_VERSION | cut -d. -f1) && \
    TARBALL="linux-${KERNEL_TARBALL_VERSION}.tar.xz" && \
    if ! echo "${KERNEL_SHA256}  /tmp/kernel-cache/$TARBALL" | sha256sum -c - >/dev/null 2>&1; then \
        rm -f "/tmp/kernel-cache/$TARBALL" && \
        wget -q "https://cdn.kernel.org/pub/linux/kernel/v${MAJOR}.x/$TARBALL" \
            -O "/tmp/kernel-cache/$TARBALL" && \
        echo "${KERNEL_SHA256}  /tmp/kernel-cache/$TARBALL" | sha256sum -c -; \
    fi && \
    tar -xf "/tmp/kernel-cache/$TARBALL" -C /build/ && \
    ln -s "/build/linux-${KERNEL_TARBALL_VERSION}" /build/linux

# Copy Alpine's base config (vendored into git by `make upgrade`)
COPY images/kernel/alpine-virt-${ARCH}.config /tmp/alpine-virt.config

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
    # Fail-closed: verify the fragment survived olddefconfig. Kconfig `select`
    # statements silently re-enable disabled options (e.g. legacy NTFS_FS
    # selecting NTFS3_FS), and merge_config.sh -m never verifies the final
    # config. Arch-forced exceptions are documented in exec-sandbox.config.
    if [ "$KERNEL_ARCH" = "x86" ]; then \
        ALLOW="CONFIG_PERF_EVENTS CONFIG_NLS CONFIG_HOTPLUG_CPU"; \
    else \
        ALLOW=""; \
    fi && \
    sed -n 's/^# \(CONFIG_[A-Z0-9_]*\) is not set$/\1/p' /tmp/exec-sandbox.config > /tmp/frag-unset.list && \
    grep -E '^CONFIG_[A-Z0-9_]+=' /tmp/exec-sandbox.config > /tmp/frag-set.list && \
    : > /tmp/config-violations && \
    while IFS= read -r opt; do \
        case " $ALLOW " in *" $opt "*) continue ;; esac; \
        v=$(grep -E "^${opt}=" .config) && \
            echo "REVERTED: fragment disables ${opt}, final config has ${v}" >> /tmp/config-violations || true; \
    done < /tmp/frag-unset.list && \
    # Wanted values must match when the symbol exists at all in the final
    # config. Absent = arch-inapplicable or renamed upstream: tolerated but
    # logged, so a typo'd directive is visible instead of silently inert.
    while IFS= read -r line; do \
        opt="${line%%=*}" && \
        if grep -qE "^${opt}=|^# ${opt} is not set" .config; then \
            grep -qxF "$line" .config || \
                echo "DROPPED: fragment wants '${line}', final config disagrees" >> /tmp/config-violations; \
        else \
            echo "NOTE: fragment sets '${line}' but symbol is absent on this arch" >&2; \
        fi; \
    done < /tmp/frag-set.list && \
    if [ -s /tmp/config-violations ]; then \
        echo "ERROR: exec-sandbox.config did not survive merge_config.sh + olddefconfig:" >&2 && \
        cat /tmp/config-violations >&2 && \
        exit 1; \
    fi && \
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
COPY --from=builder /vmlinuz-* .
DOCKERFILE

    # Sidecar (.hash) is written by extract-kernel.sh only — the orchestrator
    # owns it so a standalone build here leaves a hash miss, forcing
    # extract-kernel.sh to refresh derived artifacts (vmlinux PVH extraction)
    # instead of serving a stale one.

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

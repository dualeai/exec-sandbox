#!/bin/bash
# Build qcow2 images from Alpine base with language runtimes
#
# Variants:
#   python - Alpine + Python + gcc/musl-dev (for C extensions)
#   node   - Alpine + bun
#   raw    - Alpine only (no runtime)
#
# All images include common tools for AI agent workflows:
#   git, jq, bash, coreutils, tar, gzip, unzip, file, curl
#
# Requires: Docker
#
# Usage:
#   ./scripts/build-qcow2.sh                  # Build all for current arch
#   ./scripts/build-qcow2.sh python x86_64    # Build specific image
#   ./scripts/build-qcow2.sh all              # Build all variants for current arch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/images/dist"

PYTHON_VERSION="${PYTHON_VERSION:-3.14.2}"
PYTHON_BUILD_DATE="${PYTHON_BUILD_DATE:-20251217}"  # From astral-sh/python-build-standalone
UV_VERSION="${UV_VERSION:-0.9.24}"  # From astral-sh/uv
CLOUDPICKLE_VERSION="${CLOUDPICKLE_VERSION:-3.1.2}"  # From https://github.com/cloudpipe/cloudpickle/releases
BUN_VERSION="${BUN_VERSION:-1.3.5}"
ALPINE_VERSION="${ALPINE_VERSION:-3.21}"

# Buildx cache configuration (for CI)
# Set BUILDX_CACHE_FROM and BUILDX_CACHE_TO to enable external caching
# Example: BUILDX_CACHE_FROM="type=gha" BUILDX_CACHE_TO="type=gha,mode=max"
BUILDX_CACHE_FROM="${BUILDX_CACHE_FROM:-}"
BUILDX_CACHE_TO="${BUILDX_CACHE_TO:-}"

# Package lists for each variant
# Common: essential tools for AI agent workflows
# iputils: provides ping for guest-agent gvproxy connectivity check at boot
COMMON_PKGS="ca-certificates curl git jq bash coreutils tar gzip unzip file iputils"

# Python: add build tools for C extensions (numpy, pandas, etc.)
PYTHON_PKGS="$COMMON_PKGS gcc musl-dev libffi-dev"

# Node: bun needs libgcc/libstdc++
NODE_PKGS="$COMMON_PKGS libgcc libstdc++"

# Raw: minimal but useful
RAW_PKGS="$COMMON_PKGS"

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *) echo "Unsupported arch: $(uname -m)" >&2; exit 1 ;;
    esac
}

check_deps() {
    local missing=()
    command -v docker >/dev/null 2>&1 || missing+=("docker")

    if [ ${#missing[@]} -ne 0 ]; then
        echo "Missing: ${missing[*]}" >&2
        exit 1
    fi
}

# =============================================================================
# APK cache helpers - BuildKit cache mounts for faster package installs
# =============================================================================

# Create Alpine rootfs with packages using BuildKit cache mounts
# Replaces separate docker create + docker run apk add steps
# Usage: create_alpine_rootfs <rootfs_dir> <docker_platform> <packages...>
create_alpine_rootfs() {
    local rootfs_dir=$1
    local docker_platform=$2
    shift 2
    local packages="$*"

    # Use docker buildx with BuildKit cache mounts for APK
    # This caches both APK index and downloaded packages across builds
    # Scope includes arch and Alpine version to avoid cache collisions
    local platform_arch="${docker_platform#linux/}"  # Extract arch from linux/arm64 -> arm64
    local cache_scope="alpine-${ALPINE_VERSION}-${platform_arch}"
    local cache_args=()
    [ -n "$BUILDX_CACHE_FROM" ] && cache_args+=(--cache-from "$BUILDX_CACHE_FROM,scope=$cache_scope")
    [ -n "$BUILDX_CACHE_TO" ] && cache_args+=(--cache-to "$BUILDX_CACHE_TO,scope=$cache_scope")

    DOCKER_BUILDKIT=1 docker buildx build \
        --platform "$docker_platform" \
        --output "type=local,dest=$rootfs_dir" \
        --build-arg PACKAGES="$packages" \
        --build-arg ALPINE_VERSION="$ALPINE_VERSION" \
        "${cache_args[@]+"${cache_args[@]}"}" \
        --quiet \
        -f - . <<'DOCKERFILE'
# syntax=docker/dockerfile:1.4
ARG ALPINE_VERSION
FROM alpine:${ALPINE_VERSION}
ARG PACKAGES
# Use BuildKit cache mount for APK (persists across builds)
RUN --mount=type=cache,target=/var/cache/apk,sharing=locked \
    apk update && apk add --no-progress $PACKAGES
DOCKERFILE
}

# =============================================================================
# Cache helpers - content-addressable build caching via .hash sidecar files
# =============================================================================

# Compute hash for qcow2 inputs
compute_qcow2_hash() {
    local variant=$1
    local target_arch=$2
    local guest_agent=$3

    (
        # Version info
        echo "variant=$variant"
        echo "arch=$target_arch"
        echo "alpine=$ALPINE_VERSION"

        # Variant-specific versions
        case "$variant" in
            python)
                echo "python=$PYTHON_VERSION"
                echo "python_build=$PYTHON_BUILD_DATE"
                echo "uv=$UV_VERSION"
                echo "cloudpickle=$CLOUDPICKLE_VERSION"
                echo "pkgs=$PYTHON_PKGS"
                ;;
            node)
                echo "bun=$BUN_VERSION"
                echo "pkgs=$NODE_PKGS"
                ;;
            raw)
                echo "pkgs=$RAW_PKGS"
                ;;
        esac

        # Guest agent binary hash (includes init logic)
        sha256sum "$guest_agent" 2>/dev/null | cut -d' ' -f1 || echo "no-agent"

        # Build script itself (invalidate cache when build logic changes)
        cat "$SCRIPT_DIR/build-qcow2.sh" 2>/dev/null || true
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
# Build functions
# =============================================================================

# Find and verify guest-agent binary for target architecture.
#
# IMPORTANT: This function verifies the binary's ELF architecture matches the
# requested target. This prevents a subtle bug where a cached binary of the
# wrong architecture gets embedded in the qcow2 image, causing execv() to fail
# with ENOEXEC (errno=8) at boot time.
#
# Args:
#   $1 - target_arch: "x86_64" or "aarch64"
#
# Returns:
#   Prints path to verified binary on stdout
#   Exits with error if no valid binary found
#
# Search order:
#   1. images/dist/guest-agent-linux-{arch} (CI build output)
#   2. guest-agent/target/{arch}-unknown-linux-musl/release/guest-agent (local cross-compile)
#
# Note: We intentionally do NOT fall back to target/release/guest-agent (without
# arch qualifier) as that binary's architecture depends on how it was built and
# could silently be wrong.
find_guest_agent() {
    local target_arch=$1
    local rust_target="${target_arch}-unknown-linux-musl"

    # Map target_arch to ELF architecture string used by `file` command
    local elf_arch
    case "$target_arch" in
        x86_64)  elf_arch="x86-64" ;;
        aarch64) elf_arch="ARM aarch64" ;;
        *) echo "Unknown architecture: $target_arch" >&2; exit 1 ;;
    esac

    # Search paths - only arch-qualified paths to prevent wrong-arch bugs
    local paths=(
        "$OUTPUT_DIR/guest-agent-linux-$target_arch"
        "$REPO_ROOT/guest-agent/target/$rust_target/release/guest-agent"
    )

    for p in "${paths[@]}"; do
        if [ -f "$p" ]; then
            # Verify binary architecture matches target
            if ! file "$p" | grep -q "$elf_arch"; then
                echo "WARNING: $p exists but is wrong architecture (expected $elf_arch)" >&2
                echo "  Actual: $(file "$p")" >&2
                continue
            fi
            echo "$p"
            return 0
        fi
    done

    echo "Guest agent not found for $target_arch" >&2
    echo "Build with: ./scripts/build-guest-agent.sh $target_arch" >&2
    exit 1
}

# Create rootfs with Python runtime from python-build-standalone
create_python_rootfs() {
    local rootfs_dir=$1
    local target_arch=$2
    local docker_platform
    local python_arch

    case "$target_arch" in
        x86_64)
            docker_platform="linux/amd64"
            python_arch="x86_64_v3"  # Modern x86-64 (AVX2)
            ;;
        aarch64)
            docker_platform="linux/arm64"
            python_arch="aarch64"
            ;;
    esac

    # Create Alpine rootfs with packages (BuildKit cached)
    echo "  Creating Alpine base + installing packages: $PYTHON_PKGS"
    # shellcheck disable=SC2086  # Word splitting is intentional for package list
    create_alpine_rootfs "$rootfs_dir" "$docker_platform" $PYTHON_PKGS

    echo "  Downloading Python $PYTHON_VERSION + uv $UV_VERSION in parallel..."

    # Download python-build-standalone and uv in parallel
    # Format: cpython-{version}+{date}-{arch}-unknown-linux-musl-install_only_stripped.tar.gz
    # Using musl variant for Alpine Linux compatibility
    local python_url="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_BUILD_DATE}/cpython-${PYTHON_VERSION}%2B${PYTHON_BUILD_DATE}-${python_arch}-unknown-linux-musl-install_only_stripped.tar.gz"

    local uv_arch
    case "$target_arch" in
        x86_64)  uv_arch="x86_64" ;;
        aarch64) uv_arch="aarch64" ;;
    esac
    local uv_url="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${uv_arch}-unknown-linux-musl.tar.gz"

    # Create temp files for parallel downloads
    local tmp_python="$rootfs_dir/../python.tar.gz"
    local tmp_uv="$rootfs_dir/../uv.tar.gz"

    # Download both in parallel
    curl -sfL "$python_url" -o "$tmp_python" &
    local pid_python=$!
    curl -sfL "$uv_url" -o "$tmp_uv" &
    local pid_uv=$!

    # Wait for downloads and check results
    local failed=0
    if ! wait $pid_python; then
        echo "Failed to download Python. Check PYTHON_VERSION ($PYTHON_VERSION) and PYTHON_BUILD_DATE ($PYTHON_BUILD_DATE)" >&2
        echo "URL: $python_url" >&2
        failed=1
    fi
    if ! wait $pid_uv; then
        echo "Failed to download uv from: $uv_url" >&2
        failed=1
    fi
    [ $failed -ne 0 ] && exit 1

    # Extract both
    mkdir -p "$rootfs_dir/opt" "$rootfs_dir/usr/local/bin"
    tar -xzf "$tmp_python" -C "$rootfs_dir/opt"
    tar -xzf "$tmp_uv" -C "$rootfs_dir/usr/local/bin" --strip-components=1
    rm -f "$tmp_python" "$tmp_uv"
    chmod 755 "$rootfs_dir/usr/local/bin/uv" "$rootfs_dir/usr/local/bin/uvx"

    # Create symlinks for Python
    # Note: install_only_stripped doesn't include pip - use 'uv pip' instead
    ln -sf /opt/python/bin/python3 "$rootfs_dir/usr/local/bin/python3"
    ln -sf /opt/python/bin/python3 "$rootfs_dir/usr/local/bin/python"

    # Pre-install cloudpickle for multiprocessing support in REPL
    # cloudpickle serializes functions defined in exec() (lambdas, closures, dynamic functions)
    # by extracting only referenced globals via bytecode analysis — the industry standard
    # approach used by PySpark, Ray, Dask, and joblib/loky.
    # Uses Docker because uv/python in rootfs are Linux binaries (can't run on macOS host).
    # See: https://github.com/cloudpipe/cloudpickle
    mkdir -p "$rootfs_dir/home/user/site-packages"
    docker run --rm \
        -v "$rootfs_dir:/rootfs" \
        --platform "$docker_platform" \
        "alpine:${ALPINE_VERSION}" \
        /rootfs/usr/local/bin/uv pip install \
            --python /rootfs/opt/python/bin/python3 \
            --target /rootfs/home/user/site-packages \
            "cloudpickle==$CLOUDPICKLE_VERSION"
}

# Create rootfs with Node/bun runtime using Docker
create_node_rootfs() {
    local rootfs_dir=$1
    local target_arch=$2
    local docker_platform

    case "$target_arch" in
        x86_64)  docker_platform="linux/amd64" ;;
        aarch64) docker_platform="linux/arm64" ;;
    esac

    # Create Alpine rootfs with packages (BuildKit cached)
    echo "  Creating Alpine base + installing packages: $NODE_PKGS"
    # shellcheck disable=SC2086  # Word splitting is intentional for package list
    create_alpine_rootfs "$rootfs_dir" "$docker_platform" $NODE_PKGS

    # Copy bun from official image
    echo "  Installing bun runtime..."
    docker run --rm \
        -v "$rootfs_dir:/rootfs" \
        --platform "$docker_platform" \
        "oven/bun:1.3-alpine" \
        sh -c "cp /usr/local/bin/bun /rootfs/usr/local/bin/bun"
}

# Create raw Alpine rootfs (no runtime)
create_raw_rootfs() {
    local rootfs_dir=$1
    local target_arch=$2
    local docker_platform

    case "$target_arch" in
        x86_64)  docker_platform="linux/amd64" ;;
        aarch64) docker_platform="linux/arm64" ;;
    esac

    # Create Alpine rootfs with packages (BuildKit cached)
    echo "  Creating Alpine base + installing packages: $RAW_PKGS"
    # shellcheck disable=SC2086  # Word splitting is intentional for package list
    create_alpine_rootfs "$rootfs_dir" "$docker_platform" $RAW_PKGS
}

build_qcow2() {
    local variant=$1
    local target_arch=$2
    local output_name

    # Extract major.minor version
    local python_minor="${PYTHON_VERSION%.*}"  # 3.14.2 -> 3.14
    local bun_minor="${BUN_VERSION%.*}"        # 1.3.5 -> 1.3

    case "$variant" in
        python) output_name="python-${python_minor}-base-$target_arch" ;;
        node)   output_name="node-${bun_minor}-base-$target_arch" ;;
        raw)    output_name="raw-base-$target_arch" ;;
        *) echo "Unknown variant: $variant (use: python, node, raw)" >&2; exit 1 ;;
    esac

    local guest_agent
    guest_agent=$(find_guest_agent "$target_arch")

    # Check cache
    local qcow2_img="$OUTPUT_DIR/$output_name.qcow2"
    local current_hash
    current_hash=$(compute_qcow2_hash "$variant" "$target_arch" "$guest_agent")

    if cache_hit "$qcow2_img" "$current_hash"; then
        echo "qcow2 up-to-date: $output_name (cache hit)"
        return 0
    fi

    echo "Building $output_name..."
    echo "  Using guest-agent: $guest_agent"

    local tmp_dir
    tmp_dir=$(mktemp -d)
    local rootfs_dir="$tmp_dir/rootfs"
    mkdir -p "$rootfs_dir"

    # Create rootfs with appropriate runtime
    case "$variant" in
        python) create_python_rootfs "$rootfs_dir" "$target_arch" ;;
        node)   create_node_rootfs "$rootfs_dir" "$target_arch" ;;
        raw)    create_raw_rootfs "$rootfs_dir" "$target_arch" ;;
    esac

    # Copy guest-agent
    mkdir -p "$rootfs_dir/usr/local/bin"
    cp "$guest_agent" "$rootfs_dir/usr/local/bin/guest-agent"
    chmod 755 "$rootfs_dir/usr/local/bin/guest-agent"

    # Configure DNS (gvproxy gateway) - duplicate for Alpine musl quirk
    # Note: init-wrapper.sh and network-init.start removed - logic moved to guest-agent
    echo "nameserver 192.168.127.1" > "$rootfs_dir/etc/resolv.conf"
    echo "nameserver 192.168.127.1" >> "$rootfs_dir/etc/resolv.conf"

    # Create directories
    mkdir -p "$rootfs_dir/tmp" "$rootfs_dir/home/user"
    chmod 1777 "$rootfs_dir/tmp"

    # Create sandbox user (UID 1000) for REPL execution
    # Guest-agent runs as root (PID 1), but REPL subprocess drops to this user.
    # This blocks mount(2), ptrace, module loading, etc. without needing seccomp.
    # Note: passwd/group are text files (no permission issues on macOS), but chown
    # to UID 1000 requires root — so it's done inside Docker below.
    echo "user:x:1000:1000:sandbox:/home/user:/sbin/nologin" >> "$rootfs_dir/etc/passwd"
    echo "user:x:1000:" >> "$rootfs_dir/etc/group"

    # Create qcow2 using Docker (virt-make-fs requires Linux)
    echo "  Creating qcow2..."
    mkdir -p "$OUTPUT_DIR"
    local qcow2_img="$OUTPUT_DIR/$output_name.qcow2"

    local rootfs_size
    rootfs_size=$(du -sm "$rootfs_dir" | cut -f1)
    local img_size=$((rootfs_size + 100))

    # Build guestfs image with BuildKit cache (caches apt-get install across builds)
    local guestfs_image="exec-sandbox-guestfs:latest"
    # Handle both macOS (arm64) and Linux (aarch64)
    local host_arch
    case "$(uname -m)" in
        arm64|aarch64) host_arch="arm64" ;;
        *) host_arch="amd64" ;;
    esac
    local host_platform="linux/${host_arch}"

    # Scope includes host arch to avoid cache collisions in multi-arch CI
    local cache_scope="guestfs-${host_arch}"
    local cache_args=()
    [ -n "$BUILDX_CACHE_FROM" ] && cache_args+=(--cache-from "$BUILDX_CACHE_FROM,scope=$cache_scope")
    [ -n "$BUILDX_CACHE_TO" ] && cache_args+=(--cache-to "$BUILDX_CACHE_TO,scope=$cache_scope")

    DOCKER_BUILDKIT=1 docker buildx build \
        --platform "$host_platform" \
        --load \
        --tag "$guestfs_image" \
        "${cache_args[@]+"${cache_args[@]}"}" \
        --quiet \
        -f - . <<'DOCKERFILE'
# syntax=docker/dockerfile:1.4
FROM debian:sid-slim
# Use BuildKit cache mount for apt (persists across builds)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    apt-get update -qq && \
    apt-get install -y -qq guestfs-tools qemu-utils >/dev/null 2>&1
DOCKERFILE

    # Run virt-make-fs + qemu-img using cached image
    # Note: cleanup inside Docker because files created by Docker (root) and chowned
    # to UID 1000 cannot be deleted by the CI runner (UID 1001) on the host due to
    # /tmp sticky bit. Removing inside the container (as root) avoids this.
    docker run --rm \
        -v "$tmp_dir:/build" \
        -v "$OUTPUT_DIR:/output" \
        --platform "$host_platform" \
        "$guestfs_image" \
        bash -c "
            chown -R 1000:1000 /build/rootfs/home/user
            virt-make-fs --format=raw --type=ext4 --size=+${img_size}M /build/rootfs /build/rootfs.raw
            qemu-img convert -f raw -O qcow2 -c -m 8 -W /build/rootfs.raw /output/$output_name.qcow2
            rm -rf /build/rootfs /build/rootfs.raw
        "

    rm -rf "$tmp_dir"
    save_hash "$qcow2_img" "$current_hash"
    echo "Built: $qcow2_img"
}

main() {
    local target="${1:-all}"
    local arch="${2:-$(detect_arch)}"

    check_deps

    if [ "$target" = "all" ]; then
        # Build all variants in parallel
        build_qcow2 "python" "$arch" &
        local pid_python=$!
        build_qcow2 "node" "$arch" &
        local pid_node=$!
        build_qcow2 "raw" "$arch" &
        local pid_raw=$!

        local failed=0
        wait $pid_python || failed=1
        wait $pid_node || failed=1
        wait $pid_raw || failed=1

        if [ $failed -ne 0 ]; then
            echo "Build failed" >&2
            exit 1
        fi
    else
        build_qcow2 "$target" "$arch"
    fi
}

main "$@"

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
IMAGES_DIR="$REPO_ROOT/images"

PYTHON_VERSION="${PYTHON_VERSION:-3.14.2}"
PYTHON_BUILD_DATE="${PYTHON_BUILD_DATE:-20251217}"  # From astral-sh/python-build-standalone
UV_VERSION="${UV_VERSION:-0.9.24}"  # From astral-sh/uv
BUN_VERSION="${BUN_VERSION:-1.3.5}"
ALPINE_VERSION="${ALPINE_VERSION:-3.21}"

# Package lists for each variant
# Common: essential tools for AI agent workflows
COMMON_PKGS="ca-certificates curl git jq bash coreutils tar gzip unzip file"

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

        # Guest agent binary hash
        sha256sum "$guest_agent" 2>/dev/null | cut -d' ' -f1 || echo "no-agent"

        # Init scripts hash
        cat "$IMAGES_DIR/init-wrapper.sh" 2>/dev/null || true
        cat "$IMAGES_DIR/network-init.start" 2>/dev/null || true
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

# Find guest-agent binary for target arch
find_guest_agent() {
    local target_arch=$1
    local rust_target="${target_arch}-unknown-linux-musl"

    local paths=(
        "$OUTPUT_DIR/guest-agent-linux-$target_arch"
        "$REPO_ROOT/guest-agent/target/$rust_target/release/guest-agent"
        "$REPO_ROOT/guest-agent/target/release/guest-agent"
    )

    for p in "${paths[@]}"; do
        if [ -f "$p" ]; then
            echo "$p"
            return 0
        fi
    done

    echo "Guest agent not found for $target_arch" >&2
    echo "Build with: cd guest-agent && cargo build --release --target $rust_target" >&2
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

    echo "  Creating Alpine base..."

    # Start with Alpine base
    local container_id
    container_id=$(docker create --platform "$docker_platform" \
        "alpine:$ALPINE_VERSION" \
        /bin/true)
    docker export "$container_id" | tar -xf - -C "$rootfs_dir"
    docker rm "$container_id" >/dev/null

    # Install packages
    echo "  Installing packages: $PYTHON_PKGS"
    docker run --rm \
        -v "$rootfs_dir:/rootfs" \
        --platform "$docker_platform" \
        "alpine:$ALPINE_VERSION" \
        sh -c "apk add --no-cache --root /rootfs $PYTHON_PKGS >/dev/null 2>&1"

    echo "  Downloading Python $PYTHON_VERSION from python-build-standalone..."

    # Download and extract python-build-standalone
    # Format: cpython-{version}+{date}-{arch}-unknown-linux-musl-install_only_stripped.tar.gz
    # Using musl variant for Alpine Linux compatibility
    local python_url="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_BUILD_DATE}/cpython-${PYTHON_VERSION}%2B${PYTHON_BUILD_DATE}-${python_arch}-unknown-linux-musl-install_only_stripped.tar.gz"

    mkdir -p "$rootfs_dir/opt"
    if ! curl -sfL "$python_url" | tar -xzf - -C "$rootfs_dir/opt"; then
        echo "Failed to download Python. Check PYTHON_VERSION ($PYTHON_VERSION) and PYTHON_BUILD_DATE ($PYTHON_BUILD_DATE)" >&2
        echo "URL: $python_url" >&2
        exit 1
    fi

    echo "  Installing uv $UV_VERSION..."

    # Download uv standalone binary from astral-sh
    local uv_arch
    case "$target_arch" in
        x86_64)  uv_arch="x86_64" ;;
        aarch64) uv_arch="aarch64" ;;
    esac
    local uv_url="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${uv_arch}-unknown-linux-musl.tar.gz"

    mkdir -p "$rootfs_dir/usr/local/bin"
    if ! curl -sfL "$uv_url" | tar -xzf - -C "$rootfs_dir/usr/local/bin" --strip-components=1; then
        echo "Failed to download uv from: $uv_url" >&2
        exit 1
    fi
    chmod 755 "$rootfs_dir/usr/local/bin/uv" "$rootfs_dir/usr/local/bin/uvx"

    # Create symlinks for Python
    # Note: install_only_stripped doesn't include pip - use 'uv pip' instead
    ln -sf /opt/python/bin/python3 "$rootfs_dir/usr/local/bin/python3"
    ln -sf /opt/python/bin/python3 "$rootfs_dir/usr/local/bin/python"
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

    echo "  Installing bun runtime..."

    # Start with Alpine base
    local container_id
    container_id=$(docker create --platform "$docker_platform" \
        "alpine:$ALPINE_VERSION" \
        /bin/true)
    docker export "$container_id" | tar -xf - -C "$rootfs_dir"
    docker rm "$container_id" >/dev/null

    # Copy bun from official image
    docker run --rm \
        -v "$rootfs_dir:/rootfs" \
        --platform "$docker_platform" \
        "oven/bun:1.3-alpine" \
        sh -c "cp /usr/local/bin/bun /rootfs/usr/local/bin/bun"

    # Install packages
    echo "  Installing packages: $NODE_PKGS"
    docker run --rm \
        -v "$rootfs_dir:/rootfs" \
        --platform "$docker_platform" \
        "alpine:$ALPINE_VERSION" \
        sh -c "apk add --no-cache --root /rootfs $NODE_PKGS >/dev/null 2>&1"
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

    echo "  Creating raw Alpine rootfs..."

    local container_id
    container_id=$(docker create --platform "$docker_platform" \
        "alpine:$ALPINE_VERSION" \
        /bin/true)
    docker export "$container_id" | tar -xf - -C "$rootfs_dir"
    docker rm "$container_id" >/dev/null

    # Install packages
    echo "  Installing packages: $RAW_PKGS"
    docker run --rm \
        -v "$rootfs_dir:/rootfs" \
        --platform "$docker_platform" \
        "alpine:$ALPINE_VERSION" \
        sh -c "apk add --no-cache --root /rootfs $RAW_PKGS >/dev/null 2>&1"
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

    # Copy init scripts
    cp "$IMAGES_DIR/init-wrapper.sh" "$rootfs_dir/init"
    chmod 755 "$rootfs_dir/init"

    mkdir -p "$rootfs_dir/etc/local.d"
    cp "$IMAGES_DIR/network-init.start" "$rootfs_dir/etc/local.d/network.start"
    chmod 755 "$rootfs_dir/etc/local.d/network.start"

    # Configure DNS (gvproxy gateway) - duplicate for Alpine musl quirk
    echo "nameserver 192.168.127.1" > "$rootfs_dir/etc/resolv.conf"
    echo "nameserver 192.168.127.1" >> "$rootfs_dir/etc/resolv.conf"

    # Create directories
    mkdir -p "$rootfs_dir/tmp" "$rootfs_dir/home/user"
    chmod 1777 "$rootfs_dir/tmp"

    # Create qcow2 using Docker (virt-make-fs requires Linux)
    echo "  Creating qcow2..."
    mkdir -p "$OUTPUT_DIR"
    local qcow2_img="$OUTPUT_DIR/$output_name.qcow2"

    local rootfs_size
    rootfs_size=$(du -sm "$rootfs_dir" | cut -f1)
    local img_size=$((rootfs_size + 100))

    # Run virt-make-fs + qemu-img in Docker (debian:sid has working guestfs-tools)
    docker run --rm \
        -v "$tmp_dir:/build" \
        -v "$OUTPUT_DIR:/output" \
        --platform linux/$([ "$(uname -m)" = "arm64" ] && echo "arm64" || echo "amd64") \
        debian:sid-slim \
        bash -c "
            rm -rf /var/lib/apt/lists/*
            apt-get update -qq
            apt-get install -y -qq guestfs-tools qemu-utils >/dev/null 2>&1
            virt-make-fs --format=raw --type=ext4 --size=+${img_size}M /build/rootfs /build/rootfs.raw
            qemu-img convert -f raw -O qcow2 -c /build/rootfs.raw /output/$output_name.qcow2
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

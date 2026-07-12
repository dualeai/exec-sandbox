#!/bin/bash
# Refresh system version pins in versions.lock (called by `make upgrade`)
#
# Resolves the latest upstream versions and content hashes for:
#   ALPINE_VERSION      latest stable branch from alpinelinux.org/releases.json
#   ALPINE_IMAGE_DIGEST manifest-list digest of alpine:<ver> (multi-arch)
#   QEMU_VERSION        latest stable tag from the QEMU GitLab repository
#   QEMU_SHA256         sha256 of qemu-<ver>.tar.xz (computed once; QEMU
#                       publishes only GPG .sig files, no sha256 lists)
#   RUST_VERSION        latest stable channel (major.minor, matches rust:X.Y-slim tags)
#   RUST_IMAGE_DIGEST   manifest-list digest of rust:<ver>-slim (multi-arch)
#   LINUX_VIRT_VERSION  Alpine linux-virt package version (kernel source of truth)
#   KERNEL_VERSION      upstream kernel version (LINUX_VIRT_VERSION minus -rN)
#   KERNEL_TARBALL_VERSION  kernel.org tarball name component (KERNEL_VERSION
#                       minus a trailing .0 — first-in-series tarballs are
#                       named linux-6.19.tar.xz, never linux-6.19.0.tar.xz)
#   KERNEL_SHA256       sha256 of linux-<tarball_ver>.tar.xz from kernel.org sha256sums.asc
#
# Also vendors Alpine's linux-virt base kernel configs into
# images/kernel/alpine-virt-{x86_64,aarch64}.config so kernel builds hash
# only git-tracked bytes (no live APKINDEX dependency at build time).
#
# Alpine dl-cdn deletes superseded packages within days of a bump, so the
# extracted config is verified against the locked version instead of pinned.
#
# Usage:
#   ./scripts/upgrade-versions.sh    # resolve latest everything, rewrite the lock

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCK_FILE="$REPO_ROOT/versions.lock"
KERNEL_CONFIG_DIR="$REPO_ROOT/images/kernel"

# =============================================================================
# Helpers
# =============================================================================

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Read a KEY=VALUE entry from the existing lock file (empty if absent)
lock_get() {
    local key=$1
    [ -f "$LOCK_FILE" ] || return 0
    grep -m1 "^${key}=" "$LOCK_FILE" 2>/dev/null | cut -d= -f2- || true
}

# Lock values are consumed by Make, shell, and Dockerfiles — reject anything
# outside a safe charset so hostile upstream metadata (APKINDEX, registries)
# can never smuggle shell/Make syntax into the committed lock.
validate_lock_value() {
    local key=$1 value=$2
    case "$value" in
        "" | *[!A-Za-z0-9.:_-]*)
            die "Refusing to write $key='$value' to versions.lock (empty or invalid characters)"
            ;;
    esac
}

# Print "KEY: old -> new" when the value changed vs the existing lock
summary_line() {
    local key=$1 new=$2
    local old
    old=$(lock_get "$key")
    if [ -z "$old" ]; then
        echo "  $key: (new) -> $new"
    elif [ "$old" != "$new" ]; then
        echo "  $key: $old -> $new"
    fi
}

# =============================================================================
# Version resolution
# =============================================================================

# Latest stable Alpine branch (e.g. "3.24") from the official releases feed.
# Uses the feed's authoritative latest_stable marker (branch sort would rank
# a just-cut, not-yet-released branch highest).
resolve_alpine() {
    curl -sf --retry 3 "https://alpinelinux.org/releases.json" | python3 -c '
import json, sys
data = json.load(sys.stdin)
latest = data["latest_stable"]
if not latest.startswith("v"):
    raise SystemExit(1)
print(latest[1:])
' || die "Could not resolve latest Alpine branch from alpinelinux.org/releases.json"
}

# Latest stable QEMU version (e.g. "11.0.2") from GitLab tags (excludes -rc).
# order_by=version puts the true max on page 1 (default updated-desc ordering
# could displace it via re-tagged old releases); max() still applied because
# GitLab version-sorts -rc tags above their releases.
resolve_qemu() {
    curl -sf --retry 3 "https://gitlab.com/api/v4/projects/qemu-project%2Fqemu/repository/tags?per_page=50&order_by=version&sort=desc" | python3 -c '
import json, re, sys
tags = json.load(sys.stdin)
versions = []
for tag in tags:
    m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag["name"])
    if m:
        versions.append(tuple(int(g) for g in m.groups()))
print(".".join(str(p) for p in max(versions)))
' || die "Could not resolve latest QEMU version from GitLab tags"
}

# Latest stable Rust as major.minor (e.g. "1.97") — matches rust:X.Y-slim tags.
# The channel manifest lists sections alphabetically: [pkg.cargo] carries a
# version line (0.x) BEFORE [pkg.rust] — parse must be section-scoped. The
# ~850KB body is fetched fully first: a mid-pipeline early-exit (grep -m1)
# would SIGPIPE curl under pipefail.
resolve_rust() {
    local body
    body=$(curl -sf --retry 3 "https://static.rust-lang.org/dist/channel-rust-stable.toml") \
        || die "Could not fetch channel-rust-stable.toml"
    local full
    # awk reads to EOF (f=0 after print, no early exit) — SIGPIPE-safe
    full=$(printf '%s\n' "$body" \
        | awk '/^\[pkg\.rust\]$/ {f=1} f && /^version = /{print; f=0}' \
        | cut -d'"' -f2 | cut -d' ' -f1)
    [ -n "$full" ] || die "Could not parse [pkg.rust] version from channel-rust-stable.toml"
    echo "$full" | cut -d. -f1,2
}

# linux-virt package version for one arch from the Alpine APKINDEX.
apkindex_linux_virt() {
    local alpine_ver=$1 arch=$2
    curl -sf --retry 3 "https://dl-cdn.alpinelinux.org/alpine/v${alpine_ver}/main/${arch}/APKINDEX.tar.gz" \
        | tar -xzO APKINDEX 2>/dev/null \
        | grep -A1 "^P:linux-virt$" \
        | grep "^V:" \
        | cut -d: -f2
}

# linux-virt version, asserted identical across both architectures.
# APKINDEX skew between arches is transient (aports CI lag) — fail and retry later.
resolve_linux_virt() {
    local alpine_ver=$1
    local ver_x86 ver_arm
    ver_x86=$(apkindex_linux_virt "$alpine_ver" x86_64) \
        || die "Could not read linux-virt version from Alpine v${alpine_ver} x86_64 APKINDEX"
    ver_arm=$(apkindex_linux_virt "$alpine_ver" aarch64) \
        || die "Could not read linux-virt version from Alpine v${alpine_ver} aarch64 APKINDEX"
    if [ "$ver_x86" != "$ver_arm" ]; then
        die "linux-virt version skew between arches (x86_64=$ver_x86 aarch64=$ver_arm) — transient APKINDEX lag, retry later"
    fi
    echo "$ver_x86"
}

# =============================================================================
# Hash resolution
# =============================================================================

# kernel.org tarball name component: strip a trailing .0 (mirrors Alpine's own
# ${pkgver%.0} in the aports linux-lts APKBUILD). 6.19.0 -> 6.19; 6.18.38 unchanged.
kernel_tarball_version() {
    echo "${1%.0}"
}

# sha256 of the kernel source tarball from kernel.org's sha256sums.asc
# (fetched over HTTPS; the file is PGP-clearsigned but the signature is not
# verified here — the pin's integrity rests on TLS + the reviewed lock diff).
resolve_kernel_sha256() {
    local kernel_ver=$1
    local tarball_ver
    tarball_ver=$(kernel_tarball_version "$kernel_ver")
    local major="${kernel_ver%%.*}"
    local sha
    sha=$(curl -sf --retry 3 "https://cdn.kernel.org/pub/linux/kernel/v${major}.x/sha256sums.asc" \
        | grep -E "^[0-9a-f]{64}  linux-${tarball_ver}\.tar\.xz$" | cut -d' ' -f1) || true
    [ -n "$sha" ] || die "linux-${tarball_ver}.tar.xz not found in kernel.org sha256sums.asc"
    echo "$sha"
}

# sha256 of qemu-<ver>.tar.xz. QEMU publishes no sha256 lists (GPG .sig only),
# so the hash is computed from a one-time download and pinned (trust on first
# use — subsequent builds verify against this pin). Reuses the existing lock
# entry when the version is unchanged to avoid a ~130MB download per run.
resolve_qemu_sha256() {
    local qemu_ver=$1
    local old_ver old_sha
    old_ver=$(lock_get QEMU_VERSION)
    old_sha=$(lock_get QEMU_SHA256)
    if [ "$old_ver" = "$qemu_ver" ] && [ -n "$old_sha" ]; then
        echo "$old_sha"
        return 0
    fi
    local url="https://download.qemu.org/qemu-${qemu_ver}.tar.xz"
    local tmp
    tmp=$(mktemp)
    echo "Downloading $url for sha256 pinning..." >&2
    curl -sfL --retry 3 -o "$tmp" "$url" || { rm -f "$tmp"; die "Could not download $url"; }
    local sha
    sha=$(sha256sum "$tmp" | cut -d' ' -f1) || { rm -f "$tmp"; die "sha256sum failed on $url"; }
    rm -f "$tmp"
    echo "$sha"
}

# Manifest-list digest of a multi-arch Docker image (no pull needed).
resolve_image_digest() {
    local image=$1
    local digest
    digest=$(docker buildx imagetools inspect "$image" 2>/dev/null \
        | awk '/^Digest:/ {print $2; exit}') || true
    case "$digest" in
        sha256:*) echo "$digest" ;;
        *) die "Could not resolve manifest digest for $image (is Docker running?)" ;;
    esac
}

# =============================================================================
# Vendored kernel base configs
# =============================================================================

# Extract Alpine's linux-virt kernel config for one arch and verify it matches
# the locked version (guards the race where Alpine bumps linux-virt between
# APKINDEX resolution and extraction — superseded apks vanish from dl-cdn,
# so pinning is impossible; verification is the only option).
vendor_config() {
    local arch=$1 alpine_ver=$2 alpine_digest=$3 expected_kver=$4
    local platform
    case "$arch" in
        x86_64)  platform="linux/amd64" ;;
        aarch64) platform="linux/arm64" ;;
    esac

    local out_file="$KERNEL_CONFIG_DIR/alpine-virt-${arch}.config"
    local raw
    raw=$(docker run --rm --platform "$platform" "alpine:${alpine_ver}@${alpine_digest}" sh -c \
        'apk add --no-cache linux-virt >/dev/null 2>&1 && basename /boot/config-*-virt && cat /boot/config-*-virt') \
        || die "Config extraction failed for $arch (alpine:${alpine_ver})"

    # First line = config filename (parameter expansion — `| head -1` would
    # SIGPIPE under pipefail on the ~120KB payload)
    local config_name="${raw%%$'\n'*}"
    case "$config_name" in
        config-"${expected_kver}"-*-virt) ;;
        *) die "Extracted config '$config_name' does not match locked kernel $expected_kver for $arch — Alpine bumped mid-run, retry" ;;
    esac

    printf '%s\n' "$raw" | tail -n +2 > "$out_file"
    [ -s "$out_file" ] || die "Extracted config for $arch is empty"
    grep -q '^CONFIG_VIRTIO=' "$out_file" || die "Extracted config for $arch lacks CONFIG_VIRTIO — not a linux-virt config"
}

# =============================================================================
# Main
# =============================================================================

main() {
    command -v docker >/dev/null 2>&1 || die "Docker is required (image digests + config extraction)"
    command -v python3 >/dev/null 2>&1 || die "python3 is required"
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required (brew install coreutils on macOS)"

    # Resolve versions
    local alpine_new qemu_new rust_new
    echo "Resolving latest versions..."
    alpine_new=$(resolve_alpine)
    qemu_new=$(resolve_qemu)
    rust_new=$(resolve_rust)

    # Kernel derives from the (possibly new) Alpine branch
    local linux_virt_new kernel_new kernel_tarball_new
    linux_virt_new=$(resolve_linux_virt "$alpine_new")
    kernel_new="${linux_virt_new%%-*}"
    kernel_tarball_new=$(kernel_tarball_version "$kernel_new")

    echo "Resolved: alpine=$alpine_new qemu=$qemu_new rust=$rust_new linux-virt=$linux_virt_new"

    # Resolve hashes
    local kernel_sha_new qemu_sha_new alpine_digest_new rust_digest_new
    kernel_sha_new=$(resolve_kernel_sha256 "$kernel_new")
    qemu_sha_new=$(resolve_qemu_sha256 "$qemu_new")
    alpine_digest_new=$(resolve_image_digest "alpine:${alpine_new}")
    rust_digest_new=$(resolve_image_digest "rust:${rust_new}-slim")

    # Sanity: QEMU tarball must exist where build-qemu.sh downloads it
    curl -sfI "https://download.qemu.org/qemu-${qemu_new}.tar.xz" >/dev/null \
        || die "qemu-${qemu_new}.tar.xz not found on download.qemu.org"

    # Vendor kernel base configs (verified against the locked version)
    mkdir -p "$KERNEL_CONFIG_DIR"
    local arch
    for arch in x86_64 aarch64; do
        echo "Vendoring Alpine linux-virt config for $arch..."
        vendor_config "$arch" "$alpine_new" "$alpine_digest_new" "$kernel_new"
    done

    # Summary of changes (before writing)
    echo "Changes:"
    summary_line ALPINE_VERSION "$alpine_new"
    summary_line ALPINE_IMAGE_DIGEST "$alpine_digest_new"
    summary_line QEMU_VERSION "$qemu_new"
    summary_line QEMU_SHA256 "$qemu_sha_new"
    summary_line RUST_VERSION "$rust_new"
    summary_line RUST_IMAGE_DIGEST "$rust_digest_new"
    summary_line LINUX_VIRT_VERSION "$linux_virt_new"
    summary_line KERNEL_VERSION "$kernel_new"
    summary_line KERNEL_TARBALL_VERSION "$kernel_tarball_new"
    summary_line KERNEL_SHA256 "$kernel_sha_new"

    # Validate then write the lock atomically
    validate_lock_value ALPINE_VERSION "$alpine_new"
    validate_lock_value ALPINE_IMAGE_DIGEST "$alpine_digest_new"
    validate_lock_value QEMU_VERSION "$qemu_new"
    validate_lock_value QEMU_SHA256 "$qemu_sha_new"
    validate_lock_value RUST_VERSION "$rust_new"
    validate_lock_value RUST_IMAGE_DIGEST "$rust_digest_new"
    validate_lock_value LINUX_VIRT_VERSION "$linux_virt_new"
    validate_lock_value KERNEL_VERSION "$kernel_new"
    validate_lock_value KERNEL_TARBALL_VERSION "$kernel_tarball_new"
    validate_lock_value KERNEL_SHA256 "$kernel_sha_new"

    # mktemp inside the repo: same filesystem as the destination, so mv is an
    # atomic rename() (cross-device mv is copy+unlink — torn lock on crash).
    local tmp_lock
    tmp_lock=$(mktemp "$REPO_ROOT/versions.lock.XXXXXX")
    chmod 644 "$tmp_lock"
    cat > "$tmp_lock" <<EOF
# Managed by \`make upgrade\` (scripts/upgrade-versions.sh). Do not edit manually.
# KEY=VALUE, no spaces: consumed by Make (include) and parsed by shell via grep
# (never sourced — lock values must not be executable).
ALPINE_VERSION=$alpine_new
ALPINE_IMAGE_DIGEST=$alpine_digest_new
QEMU_VERSION=$qemu_new
QEMU_SHA256=$qemu_sha_new
RUST_VERSION=$rust_new
RUST_IMAGE_DIGEST=$rust_digest_new
LINUX_VIRT_VERSION=$linux_virt_new
KERNEL_VERSION=$kernel_new
KERNEL_TARBALL_VERSION=$kernel_tarball_new
KERNEL_SHA256=$kernel_sha_new
EOF
    mv "$tmp_lock" "$LOCK_FILE"

    echo "Wrote $LOCK_FILE"
    git -C "$REPO_ROOT" --no-pager diff --stat -- versions.lock images/kernel/ 2>/dev/null || true
}

main "$@"

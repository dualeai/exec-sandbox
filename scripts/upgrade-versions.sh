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
# Supply-chain quarantine (mirrors uv's exclude-newer, commit c6b54393):
# reject upstream releases distributed after the cutoff (now minus
# QUARANTINE_DAYS; on/before the cutoff passes), so every adopted release has
# survived a public quarantine window.
#   - Alpine: the branch is the atomic unit — a too-young branch is skipped for
#     the newest branch GA'd on/before the cutoff, and everything Alpine-rooted
#     (image digest, linux-virt/kernel pins, vendored configs) follows the
#     chosen branch. Kernel patch bumps WITHIN a branch are adopted
#     immediately: the series never changes inside a stable branch, Alpine
#     bumps linux-virt roughly weekly (a 7-day gate would freeze pins nearly
#     forever), and kernel point releases carry unannounced security fixes.
#   - QEMU: newest release tag whose tarball was published (HTTP Last-Modified
#     on download.qemu.org — server-set, unlike backdatable git tag dates)
#     on/before the cutoff.
#   - Rust: channel manifests carry their release date; the walk descends
#     minor by minor (each minor's latest patch) until one clears the window.
#     Only when the walk dead-ends does the previous RUST_VERSION +
#     RUST_IMAGE_DIGEST pair freeze from the lock (pair kept atomically — in
#     tag@digest references the tag is decorative, so a new version with an
#     old digest would silently pull the old toolchain).
#   - Image digests are NOT quarantined: a same-tag re-push is a patch or
#     security refresh of an already-cleared version, and pinning superseded
#     untagged manifests risks registry garbage collection.
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

# Quarantine window: reject releases distributed less than this many days ago.
# All date math is day-resolution UTC via python3 (BSD/GNU `date -d/-v/-r`
# flags are mutually incompatible — never use them here).
QUARANTINE_DAYS=7
NOW_EPOCH=$(date +%s)
CUTOFF_EPOCH=$(( NOW_EPOCH - QUARANTINE_DAYS * 86400 ))

# =============================================================================
# Helpers
# =============================================================================

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# ISO date (UTC) from a unix epoch
iso_date() {
    python3 -c "import datetime, sys; print(datetime.datetime.fromtimestamp(int(sys.argv[1]), datetime.timezone.utc).strftime('%Y-%m-%d'))" "$1"
}

# ISO date + N days (for "rerun after <date>" hints)
iso_date_plus_days() {
    python3 -c "import datetime, sys; print((datetime.date.fromisoformat(sys.argv[1]) + datetime.timedelta(days=int(sys.argv[2]))).isoformat())" "$1" "$2"
}

# Read a KEY=VALUE entry from the existing lock file (empty if absent)
lock_get() {
    local key=$1
    [ -f "$LOCK_FILE" ] || return 0
    grep -m1 "^${key}=" "$LOCK_FILE" 2>/dev/null | cut -d= -f2- || true
}

# Read a key's recorded distribution date from the existing lock's comment
# line. `sed -n…p` is required: plain sed echoes a NON-matching line whole,
# which would nest garbage into the next comment. Only a strict ISO date can
# come back; anything else (including "unknown") reads as empty.
lock_get_distributed() {
    local key=$1
    [ -f "$LOCK_FILE" ] || return 0
    grep -m1 "^# ${key}: distributed " "$LOCK_FILE" 2>/dev/null \
        | sed -nE "s/^# ${key}: distributed ([0-9]{4}-[0-9]{2}-[0-9]{2}).*/\1/p" || true
}

# Dates are written into lock comment lines which bypass validate_lock_value
# — a hostile upstream string containing a newline would split into a live
# KEY line (consumed by grep -m1 and make include), and a trailing backslash
# makes GNU make swallow the next line as a comment continuation. Only a
# strict ISO date or the literal "unknown" may pass.
validate_date() {
    local what=$1 value=$2
    case "$value" in
        unknown) ;;
        [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
        *) die "Refusing to write $what date '$value' into versions.lock (not YYYY-MM-DD or 'unknown')" ;;
    esac
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

# Distribution date to record for a key: when the pinned VALUE is unchanged,
# carry the previously recorded date forward verbatim — the date describes
# the pinned artifact, not today's observation. This makes comment bytes a
# pure function of the values (no cache churn, byte-stable no-change runs).
final_date() {
    local key=$1 value=$2 fresh_date=$3
    local old_value old_date
    old_value=$(lock_get "$key")
    if [ "$old_value" = "$value" ]; then
        old_date=$(lock_get_distributed "$key")
        if [ -n "$old_date" ]; then
            echo "$old_date"
            return 0
        fi
    fi
    echo "${fresh_date:-unknown}"
}

# HTTP Last-Modified of a URL as an ISO date (UTC). Empty output on any
# failure — the CALLER must treat empty as fatal where the date gates
# quarantine (a missing header must never read as "old enough").
http_last_modified_date() {
    local url=$1
    curl -sfI --retry 3 "$url" | python3 -c "
import email.utils, sys
for line in sys.stdin:
    name, _, value = line.partition(':')
    if name.strip().lower() == 'last-modified':
        print(email.utils.parsedate_to_datetime(value.strip()).strftime('%Y-%m-%d'))
        break
" || true
}

# =============================================================================
# Version resolution (quarantine-aware)
# =============================================================================

# Newest stable Alpine branch GA'd on/before the cutoff, capped at the feed's
# latest_stable marker (a just-cut, pre-GA branch sorts highest). branch_date
# is the x.y.0 release date (verified equal for every branch). Output:
# "<version> <ga-date>".
resolve_alpine() {
    curl -sf --retry 3 "https://alpinelinux.org/releases.json" | python3 -c "
import datetime, json, sys
cutoff = datetime.date.fromisoformat(sys.argv[1])
data = json.load(sys.stdin)
latest = data['latest_stable']
if not latest.startswith('v'):
    raise SystemExit(1)
ceiling = tuple(int(x) for x in latest[1:].split('.'))
candidates = []
for b in data['release_branches']:
    name = b['rel_branch']
    if not b.get('branch_date') or not name.startswith('v'):
        continue  # edge has a null branch_date
    version = tuple(int(x) for x in name[1:].split('.'))
    if version > ceiling:
        continue
    ga = datetime.date.fromisoformat(b['branch_date'])
    if ga <= cutoff:
        candidates.append((version, name[1:], b['branch_date']))
    elif version == ceiling:
        print(f'QUARANTINED: alpine {name[1:]} GA {b[\"branch_date\"]} — selecting previous branch', file=sys.stderr)
if not candidates:
    raise SystemExit(1)
_, version, ga = max(candidates)
print(version, ga)
" "$1" || die "Could not resolve a quarantine-cleared Alpine branch from alpinelinux.org/releases.json"
}

# Newest stable QEMU release whose TARBALL was published on/before the
# cutoff. The GitLab tags API (order_by=version) is used for enumeration
# only — git tag/commit dates are author-set and backdatable; the clock is
# the server-set Last-Modified of the tarball on download.qemu.org (the
# exact artifact build-qemu.sh downloads). At most the 10 newest release
# tags are probed before giving up. Output: "<version> <lm-date>".
resolve_qemu() {
    local cutoff_date=$1
    local versions
    versions=$(curl -sf --retry 3 "https://gitlab.com/api/v4/projects/qemu-project%2Fqemu/repository/tags?per_page=50&order_by=version&sort=desc" | python3 -c "
import json, re, sys
tags = json.load(sys.stdin)
versions = set()
for tag in tags:
    m = re.fullmatch(r'v(\d+)\.(\d+)\.(\d+)', tag['name'])
    if m:
        versions.add(tuple(int(g) for g in m.groups()))
for v in sorted(versions, reverse=True):
    print('.'.join(str(p) for p in v))
") || die "Could not enumerate QEMU versions from GitLab tags"
    [ -n "$versions" ] || die "No QEMU release tags found"

    local ver lm_date checked=0
    while read -r ver; do
        checked=$((checked + 1))
        [ "$checked" -le 10 ] || break
        lm_date=$(http_last_modified_date "https://download.qemu.org/qemu-${ver}.tar.xz")
        # Tarball absent (tagged but not yet distributed) -> try older;
        # tarball present but header missing -> fatal, never "old enough".
        if [ -z "$lm_date" ]; then
            if curl -sfI --retry 3 "https://download.qemu.org/qemu-${ver}.tar.xz" >/dev/null 2>&1; then
                die "qemu-${ver}.tar.xz has no Last-Modified header — cannot quarantine-gate"
            fi
            continue
        fi
        if [ "$lm_date" \< "$cutoff_date" ] || [ "$lm_date" = "$cutoff_date" ]; then
            echo "$ver $lm_date"
            return 0
        fi
        echo "QUARANTINED: qemu $ver tarball published $lm_date — trying previous release" >&2
    done <<< "$versions"
    die "No quarantine-cleared QEMU release found in the newest 10 tags"
}

# Newest stable Rust whose channel date clears the cutoff, walking minor by
# minor down from current stable: latest minor's latest patch first, then
# minor-1's latest patch, and so on (each versioned manifest
# channel-rust-X.Y.toml tracks that minor's newest patch — the exact artifact
# the rust:X.Y-slim image tag ships, so its date gates what is consumed).
# Output: "<major.minor> <channel-date>". When the walk exhausts (cap, major
# rollover, or missing manifest) the newest too-young pair is returned and
# the CALLER's gate freezes to the previous lock pin.
#
# Section-scoped parse: manifests list sections alphabetically and
# [pkg.cargo] carries a version line (0.x) BEFORE [pkg.rust]. Each ~850KB
# body is fetched fully first, and both extractions read to EOF (a
# mid-pipeline early-exit like grep -m1 would SIGPIPE the producer under
# pipefail).
resolve_rust() {
    local cutoff_date=$1
    local url="https://static.rust-lang.org/dist/channel-rust-stable.toml"
    local steps=0 body channel_date full major minor
    while :; do
        body=$(curl -sf --retry 3 "$url") || die "Could not fetch $url"
        channel_date=$(printf '%s\n' "$body" | awk -F'"' '/^date = /{d=$2} END{print d}')
        [ -n "$channel_date" ] || die "Could not parse channel date from $url"
        full=$(printf '%s\n' "$body" \
            | awk '/^\[pkg\.rust\]$/ {f=1} f && /^version = /{print; f=0}' \
            | cut -d'"' -f2 | cut -d' ' -f1)
        [ -n "$full" ] || die "Could not parse [pkg.rust] version from $url"
        major=$(echo "$full" | cut -d. -f1)
        minor=$(echo "$full" | cut -d. -f2)
        if [ "$channel_date" \< "$cutoff_date" ] || [ "$channel_date" = "$cutoff_date" ]; then
            echo "${major}.${minor} $channel_date"
            return 0
        fi
        steps=$((steps + 1))
        [ "$steps" -le 8 ] || break
        # Major rollover: the prior major's last minor is not derivable
        [ "$minor" -gt 0 ] || break
        echo "QUARANTINED: rust ${major}.${minor} distributed $channel_date — trying ${major}.$((minor - 1))" >&2
        url="https://static.rust-lang.org/dist/channel-rust-${major}.$((minor - 1)).toml"
        # A missing versioned manifest ends the walk (freeze fallback)
        curl -sfI --retry 3 "$url" >/dev/null 2>&1 || break
    done
    echo "${major}.${minor} $channel_date"
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

kernel_tarball_url() {
    local kernel_ver=$1
    local tarball_ver major="${kernel_ver%%.*}"
    tarball_ver=$(kernel_tarball_version "$kernel_ver")
    echo "https://cdn.kernel.org/pub/linux/kernel/v${major}.x/linux-${tarball_ver}.tar.xz"
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

# Build date of a multi-arch image: max OCI `created` across platforms
# (approximates push time; per-platform builds can straggle by days).
# Outputs "unknown" on any parse trouble — the date is documentation, not a
# gate, so it degrades gracefully.
image_created_date() {
    local image=$1
    docker buildx imagetools inspect "$image" --format '{{json .Image}}' 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'created' in data:
        objs = [data]
    else:
        objs = [v for v in data.values() if isinstance(v, dict)]
    dates = [o['created'][:10] for o in objs if 'created' in o]
    print(max(dates) if dates else 'unknown')
except Exception:
    print('unknown')
" || true
    # No `|| echo unknown` here: python already prints 'unknown' on parse
    # trouble, and under pipefail a docker failure would OTHERWISE stack a
    # second line onto the substitution ("unknown\nunknown" -> validate_date die).
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

    local cutoff_date
    cutoff_date=$(iso_date "$CUTOFF_EPOCH")
    echo "Resolving latest versions (quarantine: distributed on/before $cutoff_date)..."

    # --- Alpine branch: the atomic quarantine unit -------------------------
    local out alpine_new alpine_date
    out=$(resolve_alpine "$cutoff_date") || die "Alpine resolution failed"
    read -r alpine_new alpine_date <<< "$out"
    [ -n "$alpine_new" ] || die "Alpine resolution returned no version"
    [ -n "$alpine_date" ] || die "Alpine resolution returned no date"

    # --- QEMU: newest release with a quarantine-cleared tarball ------------
    local qemu_new qemu_date
    out=$(resolve_qemu "$cutoff_date") || die "QEMU resolution failed"
    read -r qemu_new qemu_date <<< "$out"
    [ -n "$qemu_new" ] || die "QEMU resolution returned no version"
    [ -n "$qemu_date" ] || die "QEMU resolution returned no date"

    # --- Rust: freeze the version+digest PAIR while the channel is young ---
    local rust_new rust_date rust_frozen=false
    out=$(resolve_rust "$cutoff_date") || die "Rust resolution failed"
    read -r rust_new rust_date <<< "$out"
    [ -n "$rust_new" ] || die "Rust resolution returned no version"
    [ -n "$rust_date" ] || die "Rust resolution returned no date"
    if [ "$rust_date" \> "$cutoff_date" ]; then
        local kept_rust
        kept_rust=$(lock_get RUST_VERSION)
        [ -n "$kept_rust" ] || die "rust $rust_new distributed $rust_date is inside the ${QUARANTINE_DAYS}-day quarantine and no previous RUST_VERSION exists in the lock — retry after $(iso_date_plus_days "$rust_date" "$QUARANTINE_DAYS")"
        echo "QUARANTINED: rust $rust_new distributed $rust_date — keeping $kept_rust; rerun after $(iso_date_plus_days "$rust_date" "$QUARANTINE_DAYS")" >&2
        rust_new="$kept_rust"
        rust_date=$(lock_get_distributed RUST_VERSION)
        if [ -z "$rust_date" ]; then
            # Heal a missing date for the kept pin from its versioned channel
            # manifest (real release date; documentation only — degrades to
            # "unknown" on failure). awk reads to EOF: SIGPIPE-safe.
            rust_date=$(curl -sf --retry 3 "https://static.rust-lang.org/dist/channel-rust-${kept_rust}.toml" \
                | awk -F'"' '/^date = /{d=$2} END{print d}' || true)
        fi
        rust_date="${rust_date:-unknown}"
        rust_frozen=true
    fi

    # --- Kernel: follows the chosen Alpine branch (no separate gate) -------
    local linux_virt_new kernel_new kernel_tarball_new kernel_sha_new kernel_date
    linux_virt_new=$(resolve_linux_virt "$alpine_new")
    kernel_new="${linux_virt_new%%-*}"
    kernel_tarball_new=$(kernel_tarball_version "$kernel_new")
    kernel_sha_new=$(resolve_kernel_sha256 "$kernel_new")
    kernel_date=$(lock_get_distributed LINUX_VIRT_VERSION)
    if [ "$(lock_get LINUX_VIRT_VERSION)" != "$linux_virt_new" ] || [ -z "$kernel_date" ]; then
        kernel_date=$(http_last_modified_date "$(kernel_tarball_url "$kernel_new")")
        [ -n "$kernel_date" ] || die "No Last-Modified for the kernel tarball — cannot record distribution date"
    fi

    echo "Resolved: alpine=$alpine_new qemu=$qemu_new rust=$rust_new linux-virt=$linux_virt_new"

    # --- Hashes and digests (digests are not quarantined; the rust pair ----
    # --- freezes together to never mix a new tag with an old digest) -------
    local qemu_sha_new alpine_digest_new rust_digest_new
    qemu_sha_new=$(resolve_qemu_sha256 "$qemu_new")
    alpine_digest_new=$(resolve_image_digest "alpine:${alpine_new}")
    if [ "$rust_frozen" = true ]; then
        rust_digest_new=$(lock_get RUST_IMAGE_DIGEST)
        [ -n "$rust_digest_new" ] || die "Rust is quarantine-frozen but the lock has no previous RUST_IMAGE_DIGEST"
    else
        rust_digest_new=$(resolve_image_digest "rust:${rust_new}-slim")
    fi

    # Distribution dates for digest pins (documentation only, never a gate);
    # final_date carries the old date forward when the value is unchanged, so
    # imagetools is only consulted for new digests or missing recorded dates.
    local alpine_digest_date rust_digest_date
    alpine_digest_date=$(lock_get_distributed ALPINE_IMAGE_DIGEST)
    if [ "$(lock_get ALPINE_IMAGE_DIGEST)" != "$alpine_digest_new" ] || [ -z "$alpine_digest_date" ]; then
        alpine_digest_date=$(image_created_date "alpine:${alpine_new}")
    fi
    rust_digest_date=$(lock_get_distributed RUST_IMAGE_DIGEST)
    if [ "$(lock_get RUST_IMAGE_DIGEST)" != "$rust_digest_new" ] || [ -z "$rust_digest_date" ]; then
        rust_digest_date=$(image_created_date "rust:${rust_new}-slim")
    fi

    # Sanity: QEMU tarball must exist where build-qemu.sh downloads it
    curl -sfI "https://download.qemu.org/qemu-${qemu_new}.tar.xz" >/dev/null \
        || die "qemu-${qemu_new}.tar.xz not found on download.qemu.org"

    # --- Vendor kernel base configs (always from the chosen branch) --------
    mkdir -p "$KERNEL_CONFIG_DIR"
    local arch
    for arch in x86_64 aarch64; do
        echo "Vendoring Alpine linux-virt config for $arch..."
        vendor_config "$arch" "$alpine_new" "$alpine_digest_new" "$kernel_new"
    done

    # --- Distribution dates: carry forward when the value is unchanged -----
    local d_alpine d_alpine_digest d_qemu d_rust d_rust_digest d_kernel
    d_alpine=$(final_date ALPINE_VERSION "$alpine_new" "$alpine_date")
    d_alpine_digest=$(final_date ALPINE_IMAGE_DIGEST "$alpine_digest_new" "$alpine_digest_date")
    d_qemu=$(final_date QEMU_VERSION "$qemu_new" "$qemu_date")
    d_rust=$(final_date RUST_VERSION "$rust_new" "$rust_date")
    d_rust_digest=$(final_date RUST_IMAGE_DIGEST "$rust_digest_new" "$rust_digest_date")
    d_kernel=$(final_date LINUX_VIRT_VERSION "$linux_virt_new" "$kernel_date")
    local d
    for d in "$d_alpine" "$d_alpine_digest" "$d_qemu" "$d_rust" "$d_rust_digest" "$d_kernel"; do
        validate_date "distribution" "$d"
    done

    # --- Summary of changes (before writing) -------------------------------
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

    # --- Validate then write the lock atomically ---------------------------
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
    local generated_date
    generated_date=$(iso_date "$NOW_EPOCH")
    local tmp_lock
    tmp_lock=$(mktemp "$REPO_ROOT/versions.lock.XXXXXX")
    chmod 644 "$tmp_lock"
    cat > "$tmp_lock" <<EOF
# Managed by \`make upgrade\` (scripts/upgrade-versions.sh). Do not edit manually.
# KEY=VALUE, no spaces: consumed by Make (include) and parsed by shell via grep
# (never sourced — lock values must not be executable).
# Quarantine: releases distributed less than ${QUARANTINE_DAYS} days before generation are
# rejected (uv equivalent: commit c6b54393).
# Generated: ${generated_date}
# ALPINE_VERSION: distributed ${d_alpine} (branch GA)
ALPINE_VERSION=$alpine_new
# ALPINE_IMAGE_DIGEST: distributed ${d_alpine_digest} (image build)
ALPINE_IMAGE_DIGEST=$alpine_digest_new
# QEMU_VERSION: distributed ${d_qemu} (tarball publication)
QEMU_VERSION=$qemu_new
# QEMU_SHA256: distributed ${d_qemu} (tarball publication)
QEMU_SHA256=$qemu_sha_new
# RUST_VERSION: distributed ${d_rust} (stable channel release)
RUST_VERSION=$rust_new
# RUST_IMAGE_DIGEST: distributed ${d_rust_digest} (image build)
RUST_IMAGE_DIGEST=$rust_digest_new
# LINUX_VIRT_VERSION: distributed ${d_kernel} (kernel.org tarball publication)
LINUX_VIRT_VERSION=$linux_virt_new
# KERNEL_VERSION: distributed ${d_kernel} (ships in linux-virt above)
KERNEL_VERSION=$kernel_new
# KERNEL_TARBALL_VERSION: distributed ${d_kernel} (derived from KERNEL_VERSION)
KERNEL_TARBALL_VERSION=$kernel_tarball_new
# KERNEL_SHA256: distributed ${d_kernel} (kernel.org tarball publication)
KERNEL_SHA256=$kernel_sha_new
EOF

    # Idempotence: compare everything except the Generated timestamp. The
    # carry-forward rule makes comment dates stable once recorded (they only
    # change when a value changes or a missing date is healed), so a
    # no-change run leaves the committed lock byte-for-byte untouched.
    if [ -f "$LOCK_FILE" ] \
        && diff <(grep -v '^# Generated:' "$LOCK_FILE") <(grep -v '^# Generated:' "$tmp_lock") >/dev/null 2>&1; then
        rm -f "$tmp_lock"
        echo "versions.lock unchanged"
    else
        mv "$tmp_lock" "$LOCK_FILE"
        echo "Wrote $LOCK_FILE"
    fi
    git -C "$REPO_ROOT" --no-pager diff --stat -- versions.lock images/kernel/ 2>/dev/null || true
}

main "$@"

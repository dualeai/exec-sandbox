#!/bin/bash
# Extract uncompressed vmlinux ELF from a bzImage/vmlinuz.
# Adapted from linux v6.12 scripts/extract-vmlinux.
#
# Uses tr-based binary matching (no PCRE) for portability across GNU grep
# versions. grep -P with octal escapes on binary data is unreliable on
# PCRE2-backed grep (Ubuntu 24.04+).
#
# Usage: ./scripts/extract-vmlinux.sh <bzImage> > vmlinux

set -euo pipefail

if [ $# -lt 1 ] || [ ! -f "$1" ]; then
    echo "Usage: $0 <bzImage>" >&2
    exit 1
fi

img="$1"
tmp=$(mktemp /tmp/vmlinux-XXXXXX)
trap 'rm -f "$tmp"' EXIT

# Try decompressing from each offset where magic bytes match.
#
# Uses the kernel's tr-based binary search trick (no PCRE, no grep -P):
#   1. tr maps: magic1[0]->'\n', magic1[1..]->magic2, '\n'->'=', magic2->'='
#      For gzip (\037\213\010, key=xy): \037->'\n', \213->'x', \010->'y'
#   2. grep -abo "^xy" finds 0-indexed offset of 'x' (originally \213)
#   3. tail -c+<offset> starts 1-indexed, so lands on \037 (one byte before)
#      This is the key trick: grep's 0-indexed offset == tail's 1-indexed - 1
#
# The key (magic2) length MUST be len(magic1) - 1 to map all remaining bytes.
# Values match linux/scripts/extract-vmlinux: xy for 3-byte, xxx for 4-byte,
# abcde for 6-byte magic.
#
# || true guards: required because set -euo pipefail is active. Without them:
#   - grep returning 1 (no match) + pipefail kills the script
#   - Decompressor exit 2 (e.g. gunzip trailing garbage) + pipefail kills it
# The original kernel script avoids this by using #!/bin/sh with no set -e.
#
# Decompresses to a temp file and validates non-empty output (-s), so wrong
# offsets and decompressor failures are silently skipped. Tries ALL matching
# offsets per format (not just the first).
#
# Args: $1=magic_bytes $2=transform_key $3...=decompressor command
try_decompress() {
    local magic1="$1" magic2="$2"
    shift 2
    for pos in $(tr "$magic1\n$magic2" "\n$magic2=" < "$img" | grep -abo "^$magic2" || true); do
        pos=${pos%%:*}
        tail -c+"$pos" "$img" | "$@" > "$tmp" 2>/dev/null || true
        if [ -s "$tmp" ]; then
            cat "$tmp"
            return 0
        fi
    done
    return 1
}

# Try gzip (most common for Alpine, CONFIG_KERNEL_GZIP=y)
try_decompress '\037\213\010' xy    gunzip      && exit 0
# Try xz
try_decompress '\3757zXZ\000' abcde unxz        && exit 0
# Try bzip2
try_decompress 'BZh'          xy    bunzip2      && exit 0
# Try LZMA
try_decompress '\135\0\0\0'   xxx   unlzma       && exit 0
# Try lzop
try_decompress '\211\114\132' xy    lzop -d      && exit 0
# Try LZ4
try_decompress '\002!L\030'   xxx   lz4 -d       && exit 0
# Try zstd
try_decompress '(\265/\375'   xxx   unzstd       && exit 0

echo "ERROR: Could not extract vmlinux from $img" >&2
exit 1

#!/bin/bash
# Extract uncompressed vmlinux ELF from a bzImage/vmlinuz.
# Adapted from linux/scripts/extract-vmlinux.
#
# Usage: ./scripts/extract-vmlinux.sh <bzImage> > vmlinux

set -euo pipefail

if [ $# -lt 1 ] || [ ! -f "$1" ]; then
    echo "Usage: $0 <bzImage>" >&2
    exit 1
fi

BZIMAGE="$1"

# Try each decompressor, searching for its magic bytes in the bzImage.
# The kernel stores compressed vmlinux at a known offset after the setup header.
try_decompress() {
    local magic="$1"
    shift
    # Find offset of magic bytes and try decompression from first match
    local offset
    offset=$(grep -aboP "$magic" "$BZIMAGE" 2>/dev/null | head -1 | cut -d: -f1) || true
    if [ -n "$offset" ]; then
        dd if="$BZIMAGE" bs=1 skip="$offset" 2>/dev/null | "$@" 2>/dev/null
        return 0
    fi
    return 1
}

# Try gzip (most common for Alpine, CONFIG_KERNEL_GZIP=y)
try_decompress '\037\213\010' gunzip && exit 0

# Try xz
try_decompress '\3757zXZ\000' unxz && exit 0

# Try bzip2
try_decompress 'BZh' bunzip2 && exit 0

# Try LZMA
try_decompress '\135\0\0\0' unlzma && exit 0

# Try lzop
try_decompress '\211\114\132' lzop -d && exit 0

# Try LZ4
try_decompress '\002!L\030' lz4 -d - /dev/stdout && exit 0

# Try zstd
try_decompress '(\265/\375' unzstd && exit 0

echo "ERROR: Could not extract vmlinux from $BZIMAGE" >&2
exit 1

"""Hash utilities for fingerprinting and file integrity.

Two categories:
- CRC32/64: Fast non-cryptographic fingerprints for cache keys
- SHA256: Cryptographic hashes for file integrity verification
"""

from __future__ import annotations

import binascii
import hashlib
from typing import TYPE_CHECKING

import aiofiles

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# Default chunk size for file hashing (64KB)
FILE_HASH_CHUNK_SIZE = 64 * 1024


# =============================================================================
# Non-cryptographic fingerprints (fast, for cache keys)
# =============================================================================


def crc32(data: str) -> str:
    """Compute CRC32 hash as 8-character hex string (32 bits).

    Use for: file fingerprints, cache keys where 32 bits is sufficient.

    Args:
        data: String to hash

    Returns:
        8-character lowercase hex string
    """
    crc = binascii.crc32(data.encode()) & 0xFFFFFFFF
    return f"{crc:08x}"


def crc64(data: str) -> str:
    """Compute 64-bit hash as 16-character hex string using chained CRC32.

    Uses two CRC32 passes: first on the data, second seeded with the first result.
    This provides 64 bits of output for better collision resistance.

    Use for: package lists, cache keys where more uniqueness is needed.

    Args:
        data: String to hash

    Returns:
        16-character lowercase hex string
    """
    data_bytes = data.encode()
    crc1 = binascii.crc32(data_bytes) & 0xFFFFFFFF
    crc2 = binascii.crc32(data_bytes, crc1) & 0xFFFFFFFF
    return f"{crc1:08x}{crc2:08x}"


# =============================================================================
# Cryptographic hashes (for file integrity verification)
# =============================================================================


class IncrementalHasher:
    """Incremental hasher for streaming hash computation.

    Use when you need to compute hash while processing data (e.g., during download).

    Example:
        hasher = IncrementalHasher("sha256")
        for chunk in stream:
            hasher.update(chunk)
            process(chunk)
        digest = hasher.hexdigest()
    """

    def __init__(self, algorithm: str = "sha256") -> None:
        """Initialize incremental hasher.

        Args:
            algorithm: Hash algorithm (default: sha256)
        """
        self._hasher = hashlib.new(algorithm)

    def update(self, data: bytes) -> None:
        """Update hash with data chunk.

        Args:
            data: Bytes to add to hash
        """
        self._hasher.update(data)

    def hexdigest(self) -> str:
        """Return hex digest of all data added so far.

        Returns:
            Hex digest string
        """
        return self._hasher.hexdigest()


async def file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a file asynchronously.

    Reads file in chunks to minimize memory usage.

    Args:
        path: Path to file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex digest string
    """
    hasher = IncrementalHasher(algorithm)
    async with aiofiles.open(path, "rb") as f:
        while chunk := await f.read(FILE_HASH_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_hash_iter(chunks: Iterator[bytes], algorithm: str = "sha256") -> str:
    """Compute cryptographic hash from an iterator of chunks.

    Use for streaming hash computation during downloads.

    Args:
        chunks: Iterator yielding bytes chunks
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex digest string
    """
    hasher = IncrementalHasher(algorithm)
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


def bytes_hash(data: bytes, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of bytes.

    Args:
        data: Bytes to hash
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex digest string
    """
    hasher = IncrementalHasher(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def parse_hash_spec(hash_spec: str) -> tuple[str, str]:
    """Parse hash specification in format 'algorithm:digest'.

    Args:
        hash_spec: Hash spec like 'sha256:abc123...'

    Returns:
        Tuple of (algorithm, expected_digest)

    Example:
        >>> parse_hash_spec("sha256:abc123")
        ("sha256", "abc123")
    """
    if ":" not in hash_spec:
        raise ValueError(f"Invalid hash spec (missing ':'): {hash_spec}")
    return tuple(hash_spec.split(":", 1))  # type: ignore[return-value]

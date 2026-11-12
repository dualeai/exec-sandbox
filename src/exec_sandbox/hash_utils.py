"""Hash utilities for fingerprinting and file integrity.

Two categories:
- fingerprint(): SHA-256 cache keys with a canonical, separator-free encoding
- SHA256: Cryptographic hashes for file integrity verification
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Cache-key fingerprints (SHA-256, canonical encoding)
# =============================================================================


def fingerprint(*parts: str) -> str:
    """Compute a SHA-256 fingerprint of an ordered list of strings.

    Each part is prefixed with its byte length, so two different lists never
    encode to the same bytes, whatever the parts contain. Inputs may be
    tenant-chosen (package specs) and cache entries can be shared across
    tenants, so the hash must resist chosen inputs: SHA-256, never CRC.

    Args:
        *parts: Ordered strings to fingerprint

    Returns:
        Full SHA-256 hex digest
    """
    hasher = hashlib.sha256()
    for part in parts:
        data = part.encode()
        hasher.update(f"{len(data)}:".encode())
        hasher.update(data)
    return hasher.hexdigest()


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
    return await asyncio.to_thread(_file_digest, path, algorithm)


def _file_digest(path: Path, algorithm: str) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


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

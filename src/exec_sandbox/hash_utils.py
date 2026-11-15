"""Hash utilities for fingerprinting and file integrity.

Two categories:
- fingerprint(): SHA-256 cache keys with a canonical, separator-free encoding
- file_hash, parse_hash_spec: cryptographic hashes that check the integrity of a
  file (asset downloads)
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


async def file_hash(path: Path, algorithm: str) -> str:
    """Compute cryptographic hash of a file asynchronously.

    One worker-thread job (hashlib.file_digest): neither the reads nor the
    hashing run on the event loop.

    Args:
        path: Path to file
        algorithm: hashlib algorithm name (for example "sha256")

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

    Raises:
        ValueError: `hash_spec` has no ':'

    Example:
        >>> parse_hash_spec("sha256:abc123")
        ("sha256", "abc123")
    """
    algorithm, sep, digest = hash_spec.partition(":")
    if not sep:
        raise ValueError(f"Invalid hash spec (missing ':'): {hash_spec}")
    return algorithm, digest

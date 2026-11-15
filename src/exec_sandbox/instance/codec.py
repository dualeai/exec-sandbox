"""Block records: one 64 KiB block of a stream, compressed then sealed.

record = nonce (12 bytes, random) | AES-256-GCM(data key, zstd(block), aad)

The AAD binds the record to its instance (the SHA-256 of its id), its commit,
its stream (0 = QEMU migration file, 1 = home disk) and its byte offset, so a
record copied to another place, commit or instance fails authentication. AEAD
is the integrity check: there is no separate hash.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import TYPE_CHECKING

from cryptography.exceptions import InvalidTag

from exec_sandbox.exceptions import RouteError, damaged_instance_data, host_mismatch
from exec_sandbox.instance.ids import is_commit_id, is_instance_id
from exec_sandbox.instance.models import BLOCK_BYTES

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

NONCE_BYTES = 12
ZSTD_LEVEL = 3

TAG_BYTES = 16
"""AES-GCM tag: with the nonce, the smallest possible record."""

_AAD_MAGIC = b"exs1"
_KIND_BLOCK = b"b"
_KIND_INDEX = b"i"


def block_aads(*, instance_id: str, commit_id: str, stream: int) -> Callable[[int], bytes]:
    """The AAD of each block offset of one stream, fixed width (no separator ambiguity):
    magic | kind | ids | stream | offset. The ids are checked and hashed once."""
    prefix = _AAD_MAGIC + _KIND_BLOCK + _ids(instance_id, commit_id) + stream.to_bytes(1, "big")
    return lambda offset: prefix + offset.to_bytes(8, "big")


def object_index_aad(*, instance_id: str, commit_id: str) -> bytes:
    """AAD of a body object's index: its own kind byte, so it never opens as a block."""
    return _AAD_MAGIC + _KIND_INDEX + _ids(instance_id, commit_id)


def seal_block(cipher: AESGCM, block: bytes, aad: bytes) -> bytes:
    """Compress and seal one block with a fresh random nonce. `cipher`: AES-GCM of the
    data key; pack.py makes one per codec task, so two threads never share one."""
    return seal_block_with_nonce(cipher, os.urandom(NONCE_BYTES), block, aad)


def seal_block_with_nonce(cipher: AESGCM, nonce: bytes, block: bytes, aad: bytes) -> bytes:
    """Deterministic form of seal_block (golden vectors only: never reuse a nonce)."""
    return nonce + cipher.encrypt(nonce, zstd.compress(block, level=ZSTD_LEVEL), aad)


def open_block(cipher: AESGCM, record: bytes, aad: bytes, *, instance_id: str) -> bytes:
    """Authenticate and decompress one record (`cipher` as in seal_block); any change
    raises InstanceGoneError (INSTANCE_INTEGRITY). An authenticated record that does not
    decompress to one block raises RouteError (INSTANCE_HOST_MISMATCH: other_release)."""
    failed = f"Instance {instance_id}: a stored block failed authentication."
    if len(record) < NONCE_BYTES + TAG_BYTES:
        raise damaged_instance_data(failed, instance_id=instance_id)
    try:
        compressed = cipher.decrypt(record[:NONCE_BYTES], record[NONCE_BYTES:], aad)
    except InvalidTag:
        raise damaged_instance_data(failed, instance_id=instance_id) from None
    # Authenticated, so a release of this library made it; still bound the output (one block).
    decompressor = zstd.ZstdDecompressor()
    try:
        block = decompressor.decompress(compressed, max_length=BLOCK_BYTES + 1)
    except zstd.ZstdError:
        raise other_release(instance_id, what="block") from None
    if not decompressor.eof or len(block) > BLOCK_BYTES:
        raise other_release(instance_id, what="block")
    return block


def other_release(instance_id: str, *, what: str) -> RouteError:
    """Authenticated data this release cannot read: ROUTE to another release.

    The data key authenticated it, so a release of this library wrote it and the
    data is good: never RECREATE, a client would delete it. No format number is
    needed here: authentication already tells "ours, other release" from "damaged".
    No fact names the release that reads it: `required` is empty.
    """
    return host_mismatch(instance_id=instance_id, required={}, found={}, reason=f"this release cannot read its {what}")


def _ids(instance_id: str, commit_id: str) -> bytes:
    """48 fixed bytes: the full SHA-256 of the instance id (any length up to 64) and the commit id."""
    if not is_instance_id(instance_id) or not is_commit_id(commit_id):
        raise ValueError("not a valid instance id or commit id")
    return hashlib.sha256(instance_id.encode()).digest() + bytes.fromhex(commit_id)

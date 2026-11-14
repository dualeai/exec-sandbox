"""Head object: the one small object that names an instance's current state.

bytes = MAGIC (8) | header length (u32) | header | sealed body
header = canonical JSON of `HeadHeader`: the public part, readable without
         keys (instance id, generation, commit ids, key id, wrapped data key)
body   = nonce (12) | AES-256-GCM(data key, canonical JSON, AAD = everything before it)

No format number: one layout exists. A body that authenticates but does not
parse was written by this library, so it is not "damaged" (ROUTE, never
RECREATE). Integers big endian.

The header is authenticated as the body's AAD: a reader trusts nothing in it
until the body opens. Read order: parse the header, find the key by its id,
unwrap the data key, then open the body (which proves the header). The header
names its instance, so a head copied to another instance's object key (path) fails.
"""

from __future__ import annotations

import json
import os
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ValidationError

from exec_sandbox.exceptions import damaged_instance_data
from exec_sandbox.instance.codec import NONCE_BYTES, other_release
from exec_sandbox.instance.keyring import Keyring, unwrap_data_key
from exec_sandbox.instance.models import HeadBody, HeadHeader

MAGIC = b"EXSIHEAD"
MAX_HEAD_BYTES = 1024 * 1024
"""Largest head a reader accepts (store reads stop above it)."""

_LENGTH_BYTES = 4
_PREFIX_BYTES = len(MAGIC) + _LENGTH_BYTES
_MAX_HEADER_BYTES = 64 * 1024


def canonical_json(value: dict[str, Any]) -> bytes:
    """One byte form per value: sorted keys, no spaces, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def seal_head(*, header: HeadHeader, body: BaseModel, data_key: bytes) -> bytes:
    """Build head bytes with a fresh random nonce."""
    return seal_head_with_nonce(header=header, body=body, data_key=data_key, nonce=os.urandom(NONCE_BYTES))


def seal_head_with_nonce(*, header: HeadHeader, body: BaseModel, data_key: bytes, nonce: bytes) -> bytes:
    """Deterministic form of seal_head (golden vectors only: never reuse a nonce)."""
    header_bytes = canonical_json(header.model_dump(mode="json"))
    prefix = MAGIC + len(header_bytes).to_bytes(_LENGTH_BYTES, "big") + header_bytes
    data = prefix + nonce + AESGCM(data_key).encrypt(nonce, canonical_json(body.model_dump(mode="json")), prefix)
    if len(data) > MAX_HEAD_BYTES:  # no reader would accept it: never commit it
        raise ValueError(f"head of {len(data)} bytes exceeds {MAX_HEAD_BYTES}")
    return data


def read_header(data: bytes, *, instance_id: str) -> HeadHeader:
    """Parse the public header. NOT authenticated yet: use only to find the key."""
    header, _ = _split(data, instance_id=instance_id)
    return header


def open_head(data: bytes, keyring: Keyring, *, instance_id: str) -> tuple[HeadHeader, HeadBody, bytes]:
    """Authenticate the whole head with the data key its header names: (header, body, data key)."""
    header, plain, data_key = _open(data, keyring, instance_id=instance_id)
    try:
        body = HeadBody.model_validate_json(plain)
    except ValidationError:
        # Authenticated, so a release of this library wrote it: one that reads this
        # body can load the instance. Never RECREATE (a client would delete good data).
        raise other_release(instance_id, what="head body") from None
    return header, body, data_key


def authenticate(data: bytes, keyring: Keyring, *, instance_id: str) -> HeadHeader:
    """Head bytes authenticated with the data key their own header names (the body is
    not parsed): a head read back after a save may come from another saver, with
    another data key."""
    return _open(data, keyring, instance_id=instance_id)[0]


def _open(data: bytes, keyring: Keyring, *, instance_id: str) -> tuple[HeadHeader, bytes, bytes]:
    """The header, the plain body and the data key. The keyring key the header names
    (ConfigError INSTANCE_KEY_MISSING: this keyring lacks it) unwraps the data key, which opens the body."""
    header, prefix_len = _split(data, instance_id=instance_id)
    kek = keyring.get(header.key_id, instance_id=instance_id)
    data_key = unwrap_data_key(kek, bytes.fromhex(header.wrapped_data_key), instance_id=instance_id)
    nonce = data[prefix_len : prefix_len + NONCE_BYTES]
    try:
        plain = AESGCM(data_key).decrypt(nonce, data[prefix_len + NONCE_BYTES :], data[:prefix_len])
    except InvalidTag:
        raise damaged_instance_data(
            f"Instance {instance_id}: head failed authentication.", instance_id=instance_id
        ) from None
    return header, plain, data_key


def _split(data: bytes, *, instance_id: str) -> tuple[HeadHeader, int]:
    """Parse MAGIC and header; return the header and the length of the authenticated prefix."""
    if len(data) < _PREFIX_BYTES or data[: len(MAGIC)] != MAGIC:
        raise damaged_instance_data(f"Instance {instance_id}: head has no exec-sandbox magic.", instance_id=instance_id)
    start = _PREFIX_BYTES
    header_len = int.from_bytes(data[start - _LENGTH_BYTES : start], "big")
    if header_len > _MAX_HEADER_BYTES or start + header_len + NONCE_BYTES > len(data):
        raise damaged_instance_data(f"Instance {instance_id}: head is truncated.", instance_id=instance_id)
    try:
        header = HeadHeader.model_validate_json(data[start : start + header_len])
    except ValidationError:
        raise damaged_instance_data(
            f"Instance {instance_id}: head header is not valid.", instance_id=instance_id
        ) from None
    if header.instance_id != instance_id:
        raise damaged_instance_data(
            f"Instance {instance_id}: head belongs to another instance.", instance_id=instance_id
        )
    return header, start + header_len

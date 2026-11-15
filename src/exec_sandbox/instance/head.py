"""Head object: the one small object that names an instance's current state.

bytes = MAGIC (8) | header length (u32) | header | sealed body
header = canonical JSON of `HeadHeader`: the public part, readable without
         keys (instance id, generation, commit ids, key id, wrapped data key)
body   = nonce (12) | AES-256-GCM(data key, canonical JSON, AAD = everything before it)

No format number: one layout exists. A header or body that authenticates but
does not parse was written by this library, so it is not "damaged" (ROUTE,
never RECREATE). Integers big endian.

The header is authenticated as the body's AAD: a reader trusts nothing in it
until the body opens. Read order: read the envelope (instance id, key id,
wrapped data key: HeadEnvelope), find the key by its id, unwrap the data key,
open the body (which proves the header), then parse the header and the body.
The header names its instance, so a head copied to another instance's object
key (path) fails.
"""

from __future__ import annotations

import json
import os
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from exec_sandbox.exceptions import damaged_instance_data
from exec_sandbox.instance.codec import NONCE_BYTES, other_release
from exec_sandbox.instance.keyring import Keyring, unwrap_data_key
from exec_sandbox.instance.models import HeadBody, HeadEnvelope, HeadHeader, HeadSummary

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


def open_head(data: bytes, keyring: Keyring, *, instance_id: str) -> tuple[HeadHeader, HeadBody, bytes]:
    """Authenticate the whole head with the data key its header names: (header, body, data
    key). A body that authenticates but does not parse was written by another release of
    this library: RouteError, never RECREATE (a client would delete good data)."""
    header, plain, data_key = open_header(data, keyring, instance_id=instance_id)
    try:
        body = HeadBody.model_validate_json(plain)
    except ValidationError:
        raise other_release(instance_id, what="head body") from None
    return header, body, data_key


def authenticate(data: bytes, keyring: Keyring, *, instance_id: str) -> HeadHeader:
    """Head bytes authenticated with the data key their own header names (the body is
    not parsed): a head read back after a save may come from another saver, with
    another data key."""
    return open_header(data, keyring, instance_id=instance_id)[0]


def open_header(data: bytes, keyring: Keyring, *, instance_id: str) -> tuple[HeadHeader, bytes, bytes]:
    """The header, the plain body and the data key. Only the envelope (HeadEnvelope) is
    read before the head authenticates: the keyring key it names (ConfigError
    INSTANCE_KEY_MISSING: this keyring lacks it) unwraps the data key, which opens the
    body. The rest of the header is parsed after: a header that authenticates but does
    not parse was written by another release (RouteError, never RECREATE)."""
    header_bytes, plain, data_key = _authenticated(data, keyring, instance_id=instance_id)
    try:
        header = HeadHeader.model_validate_json(header_bytes)
    except ValidationError:
        raise other_release(instance_id, what="head header") from None
    return header, plain, data_key


def describe_head(
    data: bytes, keyring: Keyring, *, instance_id: str
) -> tuple[HeadSummary, HeadBody | None, str | None]:
    """An authenticated head as describe shows it: its summary, its body (None when this
    release cannot read the header or the body: another release wrote them) and the
    release that wrote it. Raises as open_header, but RouteError only when even the
    summary does not parse."""
    header_bytes, plain, _ = _authenticated(data, keyring, instance_id=instance_id)
    try:
        summary = HeadSummary.model_validate_json(header_bytes)
    except ValidationError:
        raise other_release(instance_id, what="head header") from None
    try:
        HeadHeader.model_validate_json(header_bytes)
        body = HeadBody.model_validate_json(plain)
    except ValidationError:
        return summary, None, writer_release(plain)
    return summary, body, body.host.release


def read_summary(data: bytes, *, instance_id: str) -> HeadSummary:
    """The summary of the public header, NOT authenticated (describe without the key)."""
    _, header_bytes, _ = _envelope(data, instance_id=instance_id)
    try:
        return HeadSummary.model_validate_json(header_bytes)
    except ValidationError:
        raise damaged_instance_data(
            f"Instance {instance_id}: head header is not valid.", instance_id=instance_id
        ) from None


def _authenticated(data: bytes, keyring: Keyring, *, instance_id: str) -> tuple[bytes, bytes, bytes]:
    """The header bytes, the plain body and the data key of a head that authenticates."""
    envelope, header_bytes, prefix_len = _envelope(data, instance_id=instance_id)
    kek = keyring.get(envelope.key_id, instance_id=instance_id)
    data_key = unwrap_data_key(kek, bytes.fromhex(envelope.wrapped_data_key), instance_id=instance_id)
    nonce = data[prefix_len : prefix_len + NONCE_BYTES]
    try:
        plain = AESGCM(data_key).decrypt(nonce, data[prefix_len + NONCE_BYTES :], data[:prefix_len])
    except InvalidTag:
        raise damaged_instance_data(
            f"Instance {instance_id}: head failed authentication.", instance_id=instance_id
        ) from None
    return header_bytes, plain, data_key


def writer_release(plain: bytes) -> str | None:
    """The exec-sandbox release that wrote an authenticated head body, read without the
    body's model (host.release): the release to route to when this one cannot read the
    head. None when the body names none."""
    try:
        return _WriterView.model_validate_json(plain).host.release
    except ValidationError:
        return None


class _ReleaseView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    release: str = Field(min_length=1)


class _WriterView(BaseModel):
    """The one body field writer_release reads; every other field is ignored."""

    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    host: _ReleaseView


def _envelope(data: bytes, *, instance_id: str) -> tuple[HeadEnvelope, bytes, int]:
    """MAGIC, the header length and the envelope fields: the envelope, the header bytes
    and the length of the authenticated prefix. These checks run before any key, so
    they never change: a failure here is damage (INSTANCE_INTEGRITY)."""
    if len(data) < _PREFIX_BYTES or data[: len(MAGIC)] != MAGIC:
        raise damaged_instance_data(f"Instance {instance_id}: head has no exec-sandbox magic.", instance_id=instance_id)
    start = _PREFIX_BYTES
    header_len = int.from_bytes(data[start - _LENGTH_BYTES : start], "big")
    if header_len > _MAX_HEADER_BYTES or start + header_len + NONCE_BYTES > len(data):
        raise damaged_instance_data(f"Instance {instance_id}: head is truncated.", instance_id=instance_id)
    header_bytes = data[start : start + header_len]
    try:
        envelope = HeadEnvelope.model_validate_json(header_bytes)
    except ValidationError:
        raise damaged_instance_data(
            f"Instance {instance_id}: head header is not valid.", instance_id=instance_id
        ) from None
    if envelope.instance_id != instance_id:
        raise damaged_instance_data(
            f"Instance {instance_id}: head belongs to another instance.", instance_id=instance_id
        )
    return envelope, header_bytes, start + header_len

"""Keyring: the key-encryption keys (KEKs) that wrap each chain's data key.

Text format (env var and argument alike): ``"<newest key>,<older key>,..."``,
each key the standard base64 of 32 random bytes. The first key wraps the data
key of every new chain; the others only read. Rotation: add the new key at the
end on all workers, then move it to the front, then retire the old one once no
stored head names its id (describe_instance().key_id). An instance moves to the
newest key only at its next save.

A key id is derived from the key, never chosen: HMAC-SHA256(key, label), full
digest. Two different keys never share an id, so a missing key is always named
exactly. The id is safe to show: it is in every stored head.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.keywrap import InvalidUnwrap, aes_key_unwrap, aes_key_wrap

from exec_sandbox.exceptions import ConfigError, ErrorCode, damaged_instance_data

KEY_BYTES = 32
"""AES-256 key length (KEKs and data keys)."""

_KEY_ID_LABEL = b"exec-sandbox/v1/key-id"
_MIN_DISTINCT_BYTES = 16
"""A random 32-byte key has fewer distinct bytes with probability < 1e-20."""
_PRINTABLE = range(0x20, 0x7F)
"""A random 32-byte key is all printable ASCII with probability ~2e-14: such a key is a password."""

_GENERATE_HINT = (
    'Generate a key with: python -c "import base64, secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())"'
)


def key_id(secret: bytes) -> str:
    """Id of a key: HMAC-SHA256(key, label), full hex digest."""
    return hmac.new(secret, _KEY_ID_LABEL, hashlib.sha256).hexdigest()


@dataclass(frozen=True, slots=True)
class Key:
    """One key-encryption key."""

    id: str
    secret: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class Keyring:
    """Ordered keys, newest first. Parse with `Keyring.parse`."""

    keys: tuple[Key, ...]

    @classmethod
    def parse(cls, text: str) -> Keyring:
        """Parse and validate the text format. Messages never contain key material."""
        keys: list[Key] = []
        for n, raw in enumerate(text.split(","), start=1):
            entry = raw.strip()
            if not entry:
                raise _bad_entry(n, "is empty. Separate keys with one comma, newest first", hint=None)
            try:
                secret = base64.b64decode(entry, validate=True)
            except ValueError:  # binascii.Error, and non-ASCII text
                raise _bad_entry(n, "is not base64", hint=_GENERATE_HINT) from None
            if len(secret) != KEY_BYTES:
                raise _bad_entry(
                    n, f"decodes to {len(secret)} bytes; a key is {KEY_BYTES} random bytes", hint=_GENERATE_HINT
                )
            if len(set(secret)) < _MIN_DISTINCT_BYTES:
                raise _bad_entry(n, f"is not random ({len(set(secret))} different byte values)", hint=_GENERATE_HINT)
            if all(b in _PRINTABLE for b in secret):
                raise _bad_entry(n, "is text, not random bytes", hint=_GENERATE_HINT)
            key = Key(id=key_id(secret), secret=secret)
            for m, other in enumerate(keys, start=1):
                if other.id == key.id:
                    raise _bad_entry(n, f"is the same key as entry {m}", hint=None)
            keys.append(key)
        return cls(keys=tuple(keys))

    @property
    def latest(self) -> Key:
        """The key that wraps new data keys."""
        return self.keys[0]

    def get(self, wanted_id: str, *, instance_id: str) -> Key:
        """The key with this id, or ConfigError (INSTANCE_KEY_MISSING) naming the missing key."""
        for key in self.keys:
            if key.id == wanted_id:
                return key
        ids = ", ".join(key.id for key in self.keys)
        raise ConfigError(
            f'Instance {instance_id} uses key id "{wanted_id}", which is not in this worker\'s keyring '
            f"(keyring has: {ids}).",
            code=ErrorCode.INSTANCE_KEY_MISSING,
            instance_id=instance_id,
            hint=f'All workers must share one keyring. Add key "{wanted_id}" back, or send the call to a worker '
            "that has it. Do not delete the instance: unless its head was changed in the store, the data is intact.",
            context=None,
        )


def wrap_data_key(kek: Key, data_key: bytes) -> bytes:
    """AES-KW (RFC 3394) wrap of a chain's data key."""
    return aes_key_wrap(kek.secret, data_key)


def unwrap_data_key(kek: Key, wrapped: bytes, *, instance_id: str) -> bytes:
    """Unwrap a data key; a failure with the right key id means the head was changed."""
    try:
        return aes_key_unwrap(kek.secret, wrapped)
    except InvalidUnwrap:
        raise damaged_instance_data(
            f"Instance {instance_id}: the wrapped data key failed authentication.", instance_id=instance_id
        ) from None


def _bad_entry(n: int, problem: str, *, hint: str | None) -> ConfigError:
    """Entry `n` (from 1) of the keyring text is not a key."""
    return ConfigError(
        f"Instance keyring: entry {n} {problem}.",
        code=ErrorCode.INSTANCE_CONFIG,
        instance_id=None,
        hint=hint,
        context=None,
    )

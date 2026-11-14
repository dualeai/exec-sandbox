"""Instance and commit ids.

Commit ids are 32 lowercase hex characters (128 random bits), made by the library.
An instance id is made the same way, unless the caller chooses one for a new
instance (Scheduler.new(instance_id=...)): 1 to 64 lowercase letters, digits,
".", "_" or "-", starting with a letter or digit. A guessable id is safe: a head
is authenticated with the keyring, so nobody without the keyring can make a head
that the caller's keyring opens. An id that is already taken makes the first
save fail with OutcomeUnknownError INSTANCE_CONFLICT (nothing is overwritten).
"""

from __future__ import annotations

import re
import secrets

from exec_sandbox.exceptions import ErrorCode, InvalidInputError

ID_BYTES = 16
INSTANCE_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
COMMIT_ID_PATTERN = r"^[0-9a-f]{32}$"
_INSTANCE_ID_RE = re.compile(INSTANCE_ID_PATTERN)
_COMMIT_ID_RE = re.compile(COMMIT_ID_PATTERN)


def new_id() -> str:
    """A new random id (instance or commit)."""
    return secrets.token_hex(ID_BYTES)


def is_instance_id(value: str) -> bool:
    """Whether `value` is a valid instance id (INSTANCE_ID_PATTERN)."""
    return _INSTANCE_ID_RE.fullmatch(value) is not None


def is_commit_id(value: str) -> bool:
    """Whether `value` is a valid commit id (COMMIT_ID_PATTERN)."""
    return _COMMIT_ID_RE.fullmatch(value) is not None


def check_instance_id(value: str) -> str:
    """Return the id unchanged, or raise InvalidInputError (INSTANCE_INVALID_ID)."""
    if not is_instance_id(value):
        raise InvalidInputError(
            "An instance id is 1 to 64 lowercase letters, digits, '.', '_' or '-', starting with a letter or digit.",
            code=ErrorCode.INSTANCE_INVALID_ID,
            instance_id=None,
            hint="Pass the id from InstanceRef.instance_id, or choose a valid one with Scheduler.new(instance_id=...).",
            context=None,
        )
    return value

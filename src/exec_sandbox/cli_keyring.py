"""Instance key of the CLI's local store, in the OS keyring.

This module is for the `sbx` CLI only; the library core never imports it. The
library settings read EXEC_SANDBOX_INSTANCE_KEYRING first: use this module only
when that variable is not set. The key is then in the OS keyring
(macOS Keychain; Secret Service or KWallet on Linux) through the `keyring`
package, in one entry for each OS user: service KEYRING_SERVICE, username
KEYRING_USERNAME. The first run makes the key. Losing the entry loses every
instance saved with its key. There is no fallback to a plain file: on a host
without an OS keyring, set the variable.

`keyring` takes its backend from PYTHON_KEYRING_BACKEND, its keyringrc.cfg, or
the installed backend with the highest priority: that can be a plain file
(keyrings.alt), and a change of backend reads another store, where the first
run makes a second key. Only the OS backends (_OS_BACKENDS: macOS Keychain,
Secret Service, libsecret, KWallet), or a chainer of only those, are accepted.
"""

from __future__ import annotations

import base64
import fcntl
import os
import secrets
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import keyring
from keyring.backends import SecretService, chainer, kwallet, libsecret, macOS
from keyring.errors import KeyringError

from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.files.disk import make_private_dir
from exec_sandbox.instance.keyring import KEY_BYTES, Keyring

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from keyring.backend import KeyringBackend

KEYRING_SERVICE = "exec-sandbox"
KEYRING_USERNAME = "instance-keyring"
LOCK_FILE = "instance-keyring.lock"
"""File in the account's data folder (`data_dir`). Its flock lets one run at a time read, then make, the key."""

_ENTRY = f'"{KEYRING_SERVICE}" / "{KEYRING_USERNAME}"'
"""Service / username of the entry, for messages."""
_OS_BACKENDS: frozenset[type[KeyringBackend]] = frozenset(
    {
        macOS.Keyring,  # macOS Keychain
        SecretService.Keyring,  # Secret Service (GNOME Keyring, KeePassXC...) through SecretStorage
        libsecret.Keyring,  # the same Secret Service, through libsecret
        kwallet.DBusKeyring,  # KWallet 5 and later
        kwallet.DBusKeyringKWallet4,
    }
)
"""Backends that keep the key in the OS keyring. Exact classes: a subclass can keep it anywhere."""
_NO_KEYRING_HINT = (
    "Set EXEC_SANDBOX_INSTANCE_KEYRING (for example on a headless server or in a container), "
    "or install or unlock an OS keyring (macOS Keychain; Secret Service or KWallet on Linux)."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalKeyring:
    """The instance keyring of the local store, from the OS keyring."""

    text: str = field(repr=False)
    """Keyring text for Keyring.parse. Never log or print it."""
    created: bool
    """True when this call made the key: tell the user to keep a copy."""


def open_local_keyring(*, data_dir: Path) -> LocalKeyring:
    """Read the instance keyring from the OS keyring; make it first when the entry does not exist.

    A new entry holds one key (32 random bytes, base64). The keyring API cannot
    make an entry only when it is absent, so an exclusive flock on
    `<data_dir>/LOCK_FILE` (mode 0600) holds the read and the make: when two
    first runs start at the same time, one makes the key and both read it. The
    OS keyring has one entry per account, so `data_dir` must be the same for
    every process of the account, whatever its environment: pass
    get_account_dir(kind=UserDir.DATA, ...), never a folder from HOME or XDG_*.
    Refused with ConfigError (INSTANCE_CONFIG): a lock file it cannot make or lock, a
    backend that is not an OS keyring (before any read), an OS keyring error, an
    OS keyring that does not keep the new key, and an entry that Keyring.parse
    refuses (never replaced). Messages never hold key material.
    """
    with _creation_lock(data_dir=data_dir):
        backend = _os_backend()
        try:
            text = backend.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
            created = text is None
            if text is None:
                new = base64.b64encode(secrets.token_bytes(KEY_BYTES)).decode("ascii")
                backend.set_password(KEYRING_SERVICE, KEYRING_USERNAME, new)  # pyright: ignore[reportUnknownMemberType]  # keyring wraps it untyped
                text = backend.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
                if text is None or text != new:  # a backend that drops it
                    raise ConfigError(
                        f"The OS keyring ({_name(type(backend))}) did not keep the new instance key (entry {_ENTRY}).",
                        code=ErrorCode.INSTANCE_CONFIG,
                        instance_id=None,
                        hint=_NO_KEYRING_HINT,
                        context=None,
                    )
        except KeyringError as exc:
            raise ConfigError(
                f"The OS keyring failed on entry {_ENTRY}: {type(exc).__name__}: {exc}",
                code=ErrorCode.INSTANCE_CONFIG,
                instance_id=None,
                hint=_NO_KEYRING_HINT,
                context=None,
            ) from exc
    try:
        Keyring.parse(text)
    except ConfigError as exc:
        raise ConfigError(
            f"OS keyring entry {_ENTRY} is not a valid instance keyring: {exc.message}",
            code=ErrorCode.INSTANCE_CONFIG,
            instance_id=None,
            hint=f"Put back the key from your copy with: keyring set {KEYRING_SERVICE} {KEYRING_USERNAME}. "
            "Instances saved with a lost key cannot be opened again.",
            context=None,
        ) from None
    return LocalKeyring(text=text, created=created)


def _os_backend() -> KeyringBackend:
    """The backend that `keyring` chose, when it is an OS keyring (exact class). The
    chainer of keyring passes only when each backend it chains is an OS keyring: it
    reads from each one in turn, so one plain file in the chain can give the key."""
    backend = keyring.get_keyring()
    chained: list[KeyringBackend] = list(backend.backends) if type(backend) is chainer.ChainerBackend else [backend]
    refused = (
        [b for b in chained if type(b) not in _OS_BACKENDS] if chained else [backend]
    )  # an empty chain keeps nothing
    if refused:
        raise ConfigError(
            f"Keyring backend {', '.join(_name(type(b)) for b in refused)} is not an OS keyring "
            f"(macOS Keychain, Secret Service or KWallet): it can keep the instance key (entry {_ENTRY}) "
            "in a plain file, or lose it.",
            code=ErrorCode.INSTANCE_CONFIG,
            instance_id=None,
            hint=f"{_NO_KEYRING_HINT} The keyring package takes its backend from PYTHON_KEYRING_BACKEND, "
            "its keyringrc.cfg, or else all installed backends (a chain when there are more than one). "
            "To use only the OS keyring, set PYTHON_KEYRING_BACKEND, for example to keyring.backends.macOS.Keyring.",
            context=None,
        )
    return backend


def _name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


@contextmanager
def _creation_lock(*, data_dir: Path) -> Generator[None]:
    """Hold an exclusive flock on `<data_dir>/LOCK_FILE`. Make the folder (mode 0700) and the file when missing."""
    path = data_dir / LOCK_FILE
    fd = -1
    try:
        make_private_dir(data_dir)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as exc:
        if fd >= 0:
            os.close(fd)
        raise ConfigError(
            f"Cannot lock {path}: {exc.strerror or exc}.",
            code=ErrorCode.INSTANCE_CONFIG,
            context={"path": str(path)},
            instance_id=None,
            hint=f"Make {data_dir} a folder that you own and can write to (a run with sudo can make it owned by root).",
        ) from exc
    try:
        yield
    finally:
        os.close(fd)  # releases the flock

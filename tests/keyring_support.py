"""Keyring test backends and an isolated OS account for the CLI tests: the real OS keyring and
the developer's account folders are never read or changed."""

import os
import pwd
from dataclasses import dataclass
from pathlib import Path

import keyring.core
import pytest
from keyring.backend import KeyringBackend
from keyring.backends import macOS

KEYRING_ENTRY = ("exec-sandbox", "instance-keyring")
"""(service, username) of the OS keyring entry, written out: a renamed entry loses every user's key."""


class MemoryBackend(KeyringBackend):
    """Entries in a dict; `calls` counts the reads and writes."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: dict[tuple[str, str], str] = {}
        self.calls = 0

    def get_password(self, service: str, username: str) -> str | None:
        self.calls += 1
        return self.entries.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.calls += 1
        self.entries[service, username] = password


def as_os_backend(cls: type[KeyringBackend], store: KeyringBackend) -> KeyringBackend:
    """A new instance of the OS backend class `cls` whose reads and writes go to `store`
    (instance attributes): the OS keyring behind `cls` is never called."""
    backend = cls()
    vars(backend).update(get_password=store.get_password, set_password=store.set_password)
    return backend


@dataclass(frozen=True)
class Account:
    """The OS account the CLI runs as: its home in the user database, and its OS keyring."""

    home: Path
    keyring: MemoryBackend

    def linux_data_dir(self) -> Path:
        """The account's data folder on Linux, written out: the CLI's keyring lock goes there."""
        return self.home / ".local" / "share" / "exec-sandbox"

    def assert_untouched(self) -> None:
        """The OS keyring was not read, and its lock was not made (nothing under the home)."""
        assert self.keyring.calls == 0
        assert not self.home.exists()


def isolate_account(monkeypatch: pytest.MonkeyPatch, *, home: Path, key: str | None) -> Account:
    """The user database gives this account `home` (not made); the OS keyring is a dict behind
    the macOS Keychain class (the CLI accepts OS backend classes only), holding `key` if given."""
    real_getpwuid = pwd.getpwuid

    def getpwuid(uid: int) -> pwd.struct_passwd:
        entry = real_getpwuid(uid)
        if uid != os.geteuid():
            return entry
        return pwd.struct_passwd((*entry[:5], str(home), entry.pw_shell))

    monkeypatch.setattr(pwd, "getpwuid", getpwuid)
    memory = MemoryBackend()
    if key is not None:
        memory.entries[KEYRING_ENTRY] = key
    monkeypatch.setattr(keyring.core, "_keyring_backend", as_os_backend(macOS.Keyring, memory))
    return Account(home=home, keyring=memory)

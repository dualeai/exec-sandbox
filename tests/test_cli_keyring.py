"""The CLI's local instance key in the OS keyring.

Every test uses a test backend: the real OS keyring is never read or changed. A
test backend that stands for an OS keyring is an instance of a real OS backend
class whose reads and writes go to the test backend (as_os_backend): the
backend check runs as in use, on the classes it accepts.
"""

import base64
import fcntl
import multiprocessing
import os
import re
import stat
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import keyring
import keyring.core
import pytest
from keyring.backend import KeyringBackend
from keyring.backends import SecretService, chainer, fail, kwallet, libsecret, macOS, null
from keyring.errors import KeyringError, KeyringLocked, PasswordSetError

from exec_sandbox.cli_keyring import open_local_keyring
from exec_sandbox.exceptions import ConfigError, ErrorAction, ErrorCode
from tests.instance_support import new_keyring
from tests.keyring_support import KEYRING_ENTRY, MemoryBackend, as_os_backend

_SPAWN = multiprocessing.get_context("spawn")
_CHILD_TIMEOUT = 60
_LOCK_FILE = "instance-keyring.lock"


def _key_bytes(text: str) -> bytes:
    """The bytes of one key in base64 (an independent read: not the parser of the code)."""
    return base64.b64decode(text, validate=True)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _all_text(error: BaseException) -> str:
    return " ".join([str(error), *getattr(error, "__notes__", [])])


# ============================================================================
# Test backends (no priority: keyring's own detection never picks them)
# ============================================================================


class _LockProbeBackend(MemoryBackend):
    """Checks on each call that another holder of the lock file would wait."""

    def __init__(self, lock_path: Path) -> None:
        super().__init__()
        self.lock_path = lock_path

    def _assert_locked(self) -> None:
        fd = os.open(self.lock_path, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    def get_password(self, service: str, username: str) -> str | None:
        self._assert_locked()
        return super().get_password(service, username)

    def set_password(self, service: str, username: str, password: str) -> None:
        self._assert_locked()
        super().set_password(service, username, password)


class _FailingBackend(MemoryBackend):
    """Raises `on_get` on get, or `on_set` after set (which keeps what it was given)."""

    def __init__(self, *, on_get: KeyringError | None, on_set: KeyringError | None) -> None:
        super().__init__()
        self.on_get, self.on_set = on_get, on_set

    def get_password(self, service: str, username: str) -> str | None:
        if self.on_get is not None:
            raise self.on_get
        return super().get_password(service, username)

    def set_password(self, service: str, username: str, password: str) -> None:
        super().set_password(service, username, password)
        if self.on_set is not None:
            raise self.on_set


class _DroppingBackend(null.Keyring):
    """Accepts the key, keeps nothing (as the null backend), and records what it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.given: list[str] = []

    def set_password(self, service: str, username: str, password: str) -> None:
        self.given.append(password)


class _FileBackend(KeyringBackend):
    """Entries as files in a folder that spawned processes share. Slow writes widen the race the lock must close."""

    def __init__(self, folder: Path) -> None:
        super().__init__()
        self.folder = folder

    def _path(self, service: str, username: str) -> Path:
        return self.folder / f"{service}={username}"

    def get_password(self, service: str, username: str) -> str | None:
        try:
            return self._path(service, username).read_text()
        except FileNotFoundError:
            return None

    def set_password(self, service: str, username: str, password: str) -> None:
        time.sleep(0.2)
        temp = self.folder / f".{os.getpid()}.tmp"
        temp.write_text(password)
        temp.replace(self._path(service, username))


class _PlainFileBackend(KeyringBackend):
    """Entries in a plain file, as keyrings.alt.file.PlaintextKeyring."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.calls = 0

    def get_password(self, service: str, username: str) -> str | None:
        self.calls += 1
        return self.path.read_text() if self.path.exists() else None

    def set_password(self, service: str, username: str, password: str) -> None:
        self.calls += 1
        self.path.write_text(password)


class _KeychainSubclass(macOS.Keyring):
    """A subclass of an OS backend: it can keep the key anywhere. keyring registers each
    backend class for good: not viable, so its own detection never picks this one."""

    viable = False


_OS_CLASSES = [
    macOS.Keyring,
    SecretService.Keyring,
    libsecret.Keyring,
    kwallet.DBusKeyring,
    kwallet.DBusKeyringKWallet4,
]
"""The OS backends, written out: macOS Keychain, Secret Service (two clients), KWallet (5+, 4)."""


@pytest.fixture(autouse=True)
def _no_real_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test on the fail backend; the backend from before comes back after the test."""
    monkeypatch.setattr(keyring.core, "_keyring_backend", fail.Keyring())


@pytest.fixture
def backend() -> MemoryBackend:
    """The OS keyring: a dict behind the macOS Keychain class."""
    memory = MemoryBackend()
    keyring.set_keyring(as_os_backend(macOS.Keyring, memory))
    return memory


# ============================================================================
# Get or make the key
# ============================================================================


class TestLocalKeyring:
    def test_made_once_then_read(self, tmp_path: Path, backend: MemoryBackend) -> None:
        data_dir = tmp_path / "missing" / "data"
        first = open_local_keyring(data_dir=data_dir)
        assert first.created is True
        assert backend.entries == {KEYRING_ENTRY: first.text}
        assert len(_key_bytes(first.text)) == 32  # one key: 32 random bytes, base64
        assert _mode(data_dir) == 0o700
        assert _mode(data_dir / _LOCK_FILE) == 0o600
        second = open_local_keyring(data_dir=data_dir)
        assert second.created is False
        assert second.text == first.text
        assert sorted(p.name for p in data_dir.iterdir()) == [_LOCK_FILE]  # the key is not in a file

    def test_keeps_an_entry_with_several_keys(self, tmp_path: Path, backend: MemoryBackend) -> None:
        text = f"{new_keyring()},{new_keyring()}"
        backend.entries[KEYRING_ENTRY] = text
        opened = open_local_keyring(data_dir=tmp_path)
        assert (opened.text, opened.created) == (text, False)
        assert backend.entries == {KEYRING_ENTRY: text}

    def test_key_never_in_repr(self, tmp_path: Path, backend: MemoryBackend) -> None:
        opened = open_local_keyring(data_dir=tmp_path)
        assert opened.text not in repr(opened)

    def test_lock_held_while_reading_and_making(self, tmp_path: Path) -> None:
        probe = _LockProbeBackend(tmp_path / _LOCK_FILE)
        keyring.set_keyring(as_os_backend(macOS.Keyring, probe))
        assert open_local_keyring(data_dir=tmp_path).created is True
        assert probe.calls == 3  # read, make, read back

    def test_processes_racing_make_one_key(self, tmp_path: Path) -> None:
        count = 6
        folder = tmp_path / "backend"
        folder.mkdir()
        start, results = _SPAWN.Barrier(count), _SPAWN.Queue()
        args = (str(tmp_path / "data"), str(folder), start, results)
        children = [_SPAWN.Process(target=_open_in_child, args=args) for _ in range(count)]
        for child in children:
            child.start()
        try:
            outcomes = [results.get(timeout=_CHILD_TIMEOUT) for _ in range(count)]
            for child in children:
                child.join(_CHILD_TIMEOUT)
                assert child.exitcode == 0
        finally:
            for child in children:
                if child.is_alive():
                    child.kill()
        assert [created for _, created in outcomes].count(True) == 1, outcomes
        texts = {text for text, _ in outcomes}
        assert texts == {_FileBackend(folder).get_password(*KEYRING_ENTRY)}
        assert len(_key_bytes(texts.pop())) == 32


def _open_in_child(data_dir: str, folder: str, start: Any, results: Any) -> None:
    """Child process: wait for the others, then open (or make) the key at the same time."""
    keyring.set_keyring(as_os_backend(macOS.Keyring, _FileBackend(Path(folder))))
    start.wait(_CHILD_TIMEOUT)
    try:
        opened = open_local_keyring(data_dir=Path(data_dir))
    except ConfigError as exc:
        results.put((f"error: {exc}", None))
    else:
        results.put((opened.text, opened.created))


# ============================================================================
# Refusals: never a plain file, never key material in messages
# ============================================================================


class TestRefused:
    @pytest.mark.parametrize(
        ("cls", "name"),
        [(fail.Keyring, "keyring.backends.fail.Keyring"), (null.Keyring, "keyring.backends.null.Keyring")],
        ids=["fail", "null"],
    )
    def test_no_backend(self, tmp_path: Path, cls: type[KeyringBackend], name: str) -> None:
        """Not OS keyrings: the fail backend (keyring's choice when the host has none) and the
        null backend (keyring's way to turn itself off: it keeps nothing, so each run would
        make a new key)."""
        keyring.set_keyring(cls())
        with pytest.raises(
            ConfigError,
            match=re.escape(f"{name} is not an OS keyring"),
            check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG,
        ) as info:
            open_local_keyring(data_dir=tmp_path)
        assert info.value.action is ErrorAction.FIX_CONFIG
        assert "EXEC_SANDBOX_INSTANCE_KEYRING" in _all_text(info.value)
        assert "keyrings.alt" not in _all_text(info.value)  # keyring's own advice can lead to a plain file
        assert sorted(p.name for p in tmp_path.iterdir()) == [_LOCK_FILE]

    @pytest.mark.parametrize(
        ("on_get", "on_set", "why"),
        [
            (KeyringLocked("locked"), None, "KeyringLocked: locked"),
            (None, PasswordSetError("denied"), "PasswordSetError: denied"),
        ],
        ids=["get", "set"],
    )
    def test_backend_error(
        self, tmp_path: Path, on_get: KeyringError | None, on_set: KeyringError | None, why: str
    ) -> None:
        """The message names the entry and the backend's error (its class and text); the hint
        names the variable that replaces the OS keyring; no key in either."""
        failing = _FailingBackend(on_get=on_get, on_set=on_set)
        keyring.set_keyring(as_os_backend(macOS.Keyring, failing))
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG) as info:
            open_local_keyring(data_dir=tmp_path)
        assert str(info.value) == f'The OS keyring failed on entry "exec-sandbox" / "instance-keyring": {why}'
        assert "EXEC_SANDBOX_INSTANCE_KEYRING" in _all_text(info.value)
        for given in failing.entries.values():
            assert given not in _all_text(info.value)

    def test_backend_that_keeps_nothing(self, tmp_path: Path) -> None:
        """An OS keyring whose set does nothing (as the null backend). Each run would make a new
        key and lose the last one."""
        dropping = _DroppingBackend()
        keyring.set_keyring(as_os_backend(macOS.Keyring, dropping))
        with pytest.raises(
            ConfigError, match="did not keep the new instance key", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ) as info:
            open_local_keyring(data_dir=tmp_path)
        assert "EXEC_SANDBOX_INSTANCE_KEYRING" in _all_text(info.value)
        assert len(dropping.given) == 1
        assert dropping.given[0] not in _all_text(info.value)

    @pytest.mark.parametrize("content", ["", "not base64!", "c2hvcnQ=", "{k},{k}", "{k}, "])
    def test_invalid_entry_named_and_kept(self, tmp_path: Path, backend: MemoryBackend, content: str) -> None:
        key = new_keyring()
        stored = content.format(k=key)
        backend.entries[KEYRING_ENTRY] = stored
        with pytest.raises(
            ConfigError, match="is not a valid instance keyring", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ) as info:
            open_local_keyring(data_dir=tmp_path)
        text = _all_text(info.value)
        assert '"exec-sandbox" / "instance-keyring"' in str(info.value)
        assert "keyring set exec-sandbox instance-keyring" in text
        assert key not in text
        if stored.strip():
            assert stored not in text
        assert backend.entries == {KEYRING_ENTRY: stored}  # never replaced

    def test_lock_file_symlink(self, tmp_path: Path, backend: MemoryBackend) -> None:
        target = tmp_path / "elsewhere"
        target.write_text("kept")
        (tmp_path / _LOCK_FILE).symlink_to(target)
        with pytest.raises(
            ConfigError, match="Cannot lock", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ) as info:
            open_local_keyring(data_dir=tmp_path)
        assert info.value.action is ErrorAction.FIX_CONFIG
        assert target.read_text() == "kept"
        assert backend.entries == {}

    def test_data_dir_not_a_folder(self, tmp_path: Path, backend: MemoryBackend) -> None:
        (tmp_path / "data").write_text("")
        with pytest.raises(
            ConfigError, match="Cannot lock", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ) as info:
            open_local_keyring(data_dir=tmp_path / "data")
        assert str(tmp_path / "data") in _all_text(info.value)
        assert backend.entries == {}


# ============================================================================
# Backend check: OS keyrings only
# ============================================================================


class TestBackendCheck:
    def test_plain_file_backend_refused(self, tmp_path: Path) -> None:
        """PYTHON_KEYRING_BACKEND, keyringrc.cfg or keyrings.alt by priority can choose a
        plain file: refused before any read or write, so no key lands there."""
        plain = _PlainFileBackend(tmp_path / "keyring_pass.cfg")
        keyring.set_keyring(plain)
        with pytest.raises(
            ConfigError,
            match="_PlainFileBackend is not an OS keyring",
            check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG,
        ) as info:
            open_local_keyring(data_dir=tmp_path / "data")
        assert info.value.action is ErrorAction.FIX_CONFIG
        assert "EXEC_SANDBOX_INSTANCE_KEYRING" in _all_text(info.value)
        assert "PYTHON_KEYRING_BACKEND" in _all_text(info.value)
        assert plain.calls == 0
        assert not plain.path.exists()

    def test_subclass_of_an_os_backend_refused(self, tmp_path: Path) -> None:
        """Exact classes: a subclass can keep the key anywhere."""
        keyring.set_keyring(_KeychainSubclass())
        with pytest.raises(
            ConfigError,
            match="_KeychainSubclass is not an OS keyring",
            check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG,
        ):
            open_local_keyring(data_dir=tmp_path)

    @pytest.mark.parametrize("cls", _OS_CLASSES, ids=lambda cls: f"{cls.__module__}.{cls.__qualname__}")
    def test_os_backend_accepted(self, tmp_path: Path, cls: type[KeyringBackend]) -> None:
        memory = MemoryBackend()
        keyring.set_keyring(as_os_backend(cls, memory))
        opened = open_local_keyring(data_dir=tmp_path)
        assert opened.created is True
        assert memory.entries == {KEYRING_ENTRY: opened.text}

    def test_chain_of_os_backends_accepted(self, tmp_path: Path) -> None:
        """keyring chains the installed backends when there are more than one (Secret
        Service and KWallet on one host): it reads from each and writes to the first."""
        first, second = MemoryBackend(), MemoryBackend()
        second.entries[KEYRING_ENTRY] = text = new_keyring()
        chained = [
            as_os_backend(SecretService.Keyring, first),
            as_os_backend(kwallet.DBusKeyring, second),
        ]
        keyring.set_keyring(chainer.ChainerBackend())
        with patch.object(chainer.ChainerBackend, "backends", chained):  # its own value: never the real backends
            opened = open_local_keyring(data_dir=tmp_path)
        assert (opened.text, opened.created) == (text, False)
        assert first.entries == {}

    @pytest.mark.parametrize("plain_first", [True, False], ids=["plain-first", "plain-last"])
    def test_chain_with_a_plain_file_refused(self, tmp_path: Path, plain_first: bool) -> None:
        """A chain reads from each backend in turn: one plain file in it can give the key
        (one planted there, or one that a backend change left). Refused before any read."""
        memory = MemoryBackend()
        plain = _PlainFileBackend(tmp_path / "keyring_pass.cfg")
        os_backend = as_os_backend(macOS.Keyring, memory)
        keyring.set_keyring(chainer.ChainerBackend())
        with (
            patch.object(
                chainer.ChainerBackend, "backends", [plain, os_backend] if plain_first else [os_backend, plain]
            ),
            pytest.raises(
                ConfigError,
                match="_PlainFileBackend is not an OS keyring",
                check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG,
            ) as info,
        ):
            open_local_keyring(data_dir=tmp_path)
        assert "macOS" not in str(info.value).split("is not")[0]  # only the refused backend is named
        assert (plain.calls, memory.calls) == (0, 0)  # neither backend read

    def test_empty_chain_refused(self, tmp_path: Path) -> None:
        """A chain of nothing keeps nothing: each run would make a new key."""
        keyring.set_keyring(chainer.ChainerBackend())
        with (
            patch.object(chainer.ChainerBackend, "backends", []),
            pytest.raises(
                ConfigError,
                match="ChainerBackend is not an OS keyring",
                check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG,
            ),
        ):
            open_local_keyring(data_dir=tmp_path)

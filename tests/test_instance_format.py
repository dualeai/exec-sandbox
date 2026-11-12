"""Saved-instance format core: keyring, ids, block records, sealed head.

Golden vectors (tests/fixtures/instance/) are fixed bytes: every release
must open them and re-seal them byte for byte, or stored instances break.
"""

import base64
import contextlib
import hashlib
import json
import pickle
import re
import secrets
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict

from exec_sandbox import __version__
from exec_sandbox.exceptions import ConfigError, ErrorCode, InstanceGoneError, InvalidInputError, RouteError
from exec_sandbox.instance import codec, head, ids, keyring, models, pack
from tests.instance_support import head_header

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

_VECTORS = json.loads((Path(__file__).parent / "fixtures" / "instance" / "vectors.json").read_text())
_ID = "0123456789abcdef0123456789abcdef"
_COMMIT = "fedcba9876543210fedcba9876543210"
_STRICT = ConfigDict(frozen=True, extra="forbid", strict=True)
_BODY = models.HeadBody.model_validate_json(json.dumps(_VECTORS["body"]))
"""A head body (the golden one): heads in TestHead are sealed with it."""


def _prefix_len(data: bytes) -> int:
    """Where the sealed body starts: after "EXSIHEAD" (8) | header length (u32, big endian) | header."""
    return 12 + int.from_bytes(data[8:12], "big")


def _key_text() -> str:
    return base64.b64encode(secrets.token_bytes(32)).decode()


def _header(*, generation: int = 2, instance_id: str = _ID) -> models.HeadHeader:
    """A header whose fields are well formed; its key id and wrapped key name no key."""
    commit = ids.new_id()
    return models.HeadHeader(
        instance_id=instance_id,
        generation=generation,
        commit_id=commit,
        recent_commits=(models.RecentCommit(generation=generation, commit_id=commit),),
        key_id="a" * 64,
        wrapped_data_key="b" * 80,
    )


def _sealed(
    ring: keyring.Keyring, *, generation: int = 2, data_key: bytes, body: BaseModel = _BODY
) -> tuple[bytes, models.HeadHeader]:
    """Head bytes as a saver writes them (the data key wrapped by the newest key of `ring`),
    and the header they were sealed with."""
    header = head_header(
        instance_id=_ID, generation=generation, commit_id=ids.new_id(), older=(), kek=ring.latest, data_key=data_key
    )
    return head.seal_head(header=header, body=body, data_key=data_key), header


class TestKeyring:
    def test_newest_first_and_ids_derived(self) -> None:
        new, old = _key_text(), _key_text()
        ring = keyring.Keyring.parse(f" {new} ,\n{old}\n")
        assert [k.secret for k in ring.keys] == [base64.b64decode(new), base64.b64decode(old)]  # newest first
        assert all(re.fullmatch(r"[0-9a-f]{64}", k.id) for k in ring.keys)  # full HMAC-SHA256 (derivation: golden)
        assert ring.keys[0].id != ring.keys[1].id

    def test_secret_never_in_repr(self) -> None:
        text = _key_text()
        ring = keyring.Keyring.parse(text)
        assert text not in repr(ring)
        assert base64.b64decode(text).hex() not in repr(ring)

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            pytest.param("", "entry 1 is empty", id="empty"),
            pytest.param("{k},,{k2}", "entry 2 is empty", id="double-comma"),
            pytest.param("not base64!", "entry 1 is not base64", id="not-base64"),
            pytest.param(base64.b64encode(b"short").decode(), "decodes to 5 bytes", id="short"),
            pytest.param(base64.b64encode(bytes(32)).decode(), "is not random", id="zero-key"),
            pytest.param("é" * 44, "entry 1 is not base64", id="non-ascii"),
            pytest.param(base64.b64encode(b"correct horse battery staple 123").decode(), "is text", id="password"),
            pytest.param("{k},{k}", "entry 2 is the same key as entry 1", id="duplicate"),
        ],
    )
    def test_invalid_keyring(self, text: str, match: str) -> None:
        text = text.format(k=_key_text(), k2=_key_text()) if "{" in text else text
        with pytest.raises(ConfigError, match=match, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG) as exc:
            keyring.Keyring.parse(text)
        assert exc.value.code == "INSTANCE_CONFIG"
        assert exc.value.action == "FIX_CONFIG"

    def test_missing_key_names_it(self) -> None:
        ring = keyring.Keyring.parse(_key_text())
        with pytest.raises(
            ConfigError, match='uses key id "abc"', check=lambda e: e.code is ErrorCode.INSTANCE_KEY_MISSING
        ) as exc:
            ring.get("abc", instance_id=_ID)
        assert exc.value.instance_id == _ID
        assert ring.latest.id in str(exc.value)

    def test_wrap_round_trip_and_tamper(self) -> None:
        kek = keyring.Keyring.parse(_key_text()).latest
        data_key = secrets.token_bytes(32)
        wrapped = keyring.wrap_data_key(kek, data_key)
        assert keyring.unwrap_data_key(kek, wrapped, instance_id=_ID) == data_key
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
            keyring.unwrap_data_key(kek, bytes([wrapped[0] ^ 1]) + wrapped[1:], instance_id=_ID)


class TestIds:
    def test_new_ids_are_32_hex_and_unique(self) -> None:
        made = {ids.new_id() for _ in range(100)}
        assert len(made) == 100
        assert all(re.fullmatch(r"[0-9a-f]{32}", i) for i in made)
        assert all(ids.check_instance_id(i) == i for i in made)

    @pytest.mark.parametrize("good", ["a", "my-vm", "agent_7.v2", "0" * 64, "a" * 31, "a" * 33])
    def test_accepts_caller_ids(self, good: str) -> None:
        """Scheduler.new(instance_id=...) forces readable ids: they share one rule with library ids."""
        assert ids.check_instance_id(good) == good

    @pytest.mark.parametrize("bad", ["", "My-VM", ".hidden", "-x", "_x", "a/b", "../etc", "a b", "a" * 65, "é", "a\nb"])
    def test_rejects_other_strings(self, bad: str) -> None:
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.INSTANCE_INVALID_ID):
            ids.check_instance_id(bad)


class TestBlockCodec:
    def _aad(self, *, offset: int = 0, stream: int = 0, commit: str = _COMMIT, instance: str = _ID) -> bytes:
        return codec.block_aads(instance_id=instance, commit_id=commit, stream=stream)(offset)

    def test_round_trip_and_compression(self) -> None:
        cipher = AESGCM(secrets.token_bytes(32))
        block = bytes(codec.BLOCK_BYTES)
        record = codec.seal_block(cipher, block, self._aad())
        assert len(record) < 1024  # zstd before AES: a zero block costs almost nothing
        assert codec.open_block(cipher, record, self._aad(), instance_id=_ID) == block

    def test_every_byte_flip_fails(self) -> None:
        cipher = AESGCM(secrets.token_bytes(32))
        record = codec.seal_block(cipher, b"payload" * 100, self._aad())
        for i in range(len(record)):
            changed = record[:i] + bytes([record[i] ^ 0x01]) + record[i + 1 :]
            with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
                codec.open_block(cipher, changed, self._aad(), instance_id=_ID)

    @pytest.mark.parametrize(
        "moved",
        [
            pytest.param({"offset": codec.BLOCK_BYTES}, id="other-offset"),
            pytest.param({"stream": 1}, id="other-stream"),
            pytest.param({"commit": "f" * 32}, id="other-commit"),
            pytest.param({"instance": "e" * 32}, id="other-instance"),
        ],
    )
    def test_record_bound_to_its_place(self, moved: dict[str, object]) -> None:
        cipher = AESGCM(secrets.token_bytes(32))
        record = codec.seal_block(cipher, b"payload", self._aad())
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
            codec.open_block(cipher, record, self._aad(**moved), instance_id=_ID)  # type: ignore[arg-type]

    def test_truncated_record_fails(self) -> None:
        cipher = AESGCM(secrets.token_bytes(32))
        record = codec.seal_block(cipher, b"payload", self._aad())
        for cut in (1, len(record) - 1, len(record)):  # down to an empty record
            with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
                codec.open_block(cipher, record[:-cut], self._aad(), instance_id=_ID)

    @pytest.mark.parametrize(
        ("instance", "commit"),
        [
            pytest.param(_ID.upper(), _COMMIT, id="uppercase-instance"),
            pytest.param(_ID, "my-commit", id="commit-not-hex"),
            pytest.param(_ID, _COMMIT[:-2], id="commit-too-short"),
        ],
    )
    def test_aad_refuses_invalid_ids(self, instance: str, commit: str) -> None:
        with pytest.raises(ValueError, match="valid instance id or commit id"):
            codec.block_aads(instance_id=instance, commit_id=commit, stream=0)

    @pytest.mark.parametrize(
        "plain",
        [
            pytest.param(b"x", id="not-zstd"),
            pytest.param(zstd.compress(bytes(codec.BLOCK_BYTES + 1)), id="above-one-block"),
            pytest.param(zstd.compress(b"payload" * 100)[:-4], id="frame-not-ended"),
        ],
    )
    def test_authenticated_record_this_release_cannot_read_routes(self, plain: bytes) -> None:
        """A record that authenticates (the data key sealed it: a release of this library wrote
        it) but is not one ended zstd frame of at most one block: ROUTE, never RECREATE."""
        cipher = AESGCM(secrets.token_bytes(32))
        nonce = secrets.token_bytes(codec.NONCE_BYTES)
        record = nonce + cipher.encrypt(nonce, plain, self._aad())
        with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
            codec.open_block(cipher, record, self._aad(), instance_id=_ID)
        assert exc.value.action == "ROUTE"
        assert (exc.value.instance_id, exc.value.required, exc.value.found) == (
            _ID,
            {"release": "one that reads this block"},
            {"release": __version__},
        )

    def test_aad_has_one_width_for_every_instance_id(self) -> None:
        """Instance ids of any length bind through their SHA-256: no separator, no boundary shift.
        62 bytes: magic (4) | kind (1) | SHA-256 of the instance id (32) | commit id (16) | stream (1) | offset (8)."""
        aads = {self._aad(instance=i) for i in ("a", "ab", "a" * 64, _ID)}
        assert len(aads) == 4
        assert {len(a) for a in aads} == {62}


class TestHead:
    _RING = keyring.Keyring.parse(_key_text())

    def test_round_trip(self) -> None:
        data_key = secrets.token_bytes(32)
        data, header = _sealed(self._RING, data_key=data_key)
        assert head.read_header(data, instance_id=_ID) == header
        assert head.open_head(data, self._RING, instance_id=_ID) == (header, _BODY, data_key)
        assert head.authenticate(data, self._RING, instance_id=_ID) == header

    def test_every_byte_flip_fails(self) -> None:
        """Header bytes are the body's AAD: changing any byte (public or sealed) fails. A
        flip in the key id can name a key this keyring lacks (INSTANCE_KEY_MISSING)."""
        data, _ = _sealed(self._RING, data_key=secrets.token_bytes(32))
        for i in range(len(data)):
            changed = data[:i] + bytes([data[i] ^ 0x01]) + data[i + 1 :]
            with pytest.raises(  # never ROUTE: damage is not another release
                (InstanceGoneError, ConfigError),
                check=lambda e: e.code in {ErrorCode.INSTANCE_INTEGRITY, ErrorCode.INSTANCE_KEY_MISSING},
            ):
                head.open_head(changed, self._RING, instance_id=_ID)

    def test_body_of_another_head_fails(self) -> None:
        """A sealed body spliced under another public header fails."""
        data_key = secrets.token_bytes(32)
        a, _ = _sealed(self._RING, generation=2, data_key=data_key)
        b, _ = _sealed(self._RING, generation=3, data_key=data_key)
        prefix_b, body_a = b[: _prefix_len(b)], a[_prefix_len(a) :]
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
            head.open_head(prefix_b + body_a, self._RING, instance_id=_ID)

    def test_head_of_another_instance_fails(self) -> None:
        """A head copied under another instance's key names its own instance: refused unopened."""
        data = head.seal_head(header=_header(instance_id="e" * 32), body=_BODY, data_key=secrets.token_bytes(32))
        with pytest.raises(
            InstanceGoneError,
            match="belongs to another instance",
            check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY,
        ):
            head.read_header(data, instance_id=_ID)

    def test_authenticated_body_this_release_cannot_read_routes(self) -> None:
        """A body that authenticates but does not parse: written by this library, ROUTE, not RECREATE."""

        class _NewerBody(BaseModel):
            model_config = _STRICT
            x: int
            added_later: str

        data, _ = _sealed(self._RING, data_key=secrets.token_bytes(32), body=_NewerBody(x=1, added_later="y"))
        with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
            head.open_head(data, self._RING, instance_id=_ID)
        assert exc.value.action == "ROUTE"
        head.authenticate(data, self._RING, instance_id=_ID)  # the read-back never parses the body

    def test_authenticate_with_the_named_key_only(self) -> None:
        """The body must open with the data key the header wraps, and the keyring must hold
        the key that wraps it."""
        header = head_header(
            instance_id=_ID,
            generation=2,
            commit_id=ids.new_id(),
            older=(),
            kek=self._RING.latest,
            data_key=secrets.token_bytes(32),
        )
        other_key = head.seal_head(header=header, body=_BODY, data_key=secrets.token_bytes(32))
        with pytest.raises(
            InstanceGoneError,
            match="head failed authentication",
            check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY,
        ):
            head.authenticate(other_key, self._RING, instance_id=_ID)
        stranger = keyring.Keyring.parse(_key_text())
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_KEY_MISSING):
            head.authenticate(_sealed(self._RING, data_key=secrets.token_bytes(32))[0], stranger, instance_id=_ID)

    @pytest.mark.parametrize(
        "fields",
        [
            pytest.param({"key_id": "abc"}, id="short-key-id"),
            pytest.param({"key_id": "A" * 64}, id="uppercase-key-id"),
            pytest.param({"wrapped_data_key": "b" * 78}, id="short-wrapped-key"),
            pytest.param({"generation": 0}, id="generation-0"),
            pytest.param({"recent_commits": []}, id="no-recent"),
            pytest.param({"unknown": 1}, id="extra-field"),
            pytest.param({"generation": "2"}, id="string-generation"),
        ],
    )
    def test_invalid_header_is_integrity_error(self, fields: dict[str, object]) -> None:
        raw = {**_header().model_dump(mode="json"), **fields}
        header_bytes = head.canonical_json(raw)
        data = (
            head.MAGIC
            + len(header_bytes).to_bytes(4, "big")
            + header_bytes
            + bytes(codec.NONCE_BYTES + codec.TAG_BYTES)
        )
        with pytest.raises(
            InstanceGoneError, match="header is not valid", check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY
        ):
            head.read_header(data, instance_id=_ID)

    @pytest.mark.parametrize(
        "recent",
        [
            pytest.param([(2, "other")], id="first-is-not-mine"),
            pytest.param([(2, "mine"), (2, "older")], id="not-decreasing"),
        ],
    )
    def test_recent_commits_rules(self, recent: list[tuple[int, str]]) -> None:
        names = {"mine": _COMMIT, "other": "1" * 32, "older": "2" * 32}
        with pytest.raises(ValueError, match="recent_commits"):
            models.HeadHeader(
                instance_id=_ID,
                generation=2,
                commit_id=_COMMIT,
                recent_commits=tuple(models.RecentCommit(generation=g, commit_id=names[c]) for g, c in recent),
                key_id="a" * 64,
                wrapped_data_key="b" * 80,
            )

    @pytest.mark.parametrize(
        "kinds",
        [("pack",), ("pack", "full"), ("full", "full"), ("full", "pack", "full")],
        ids=["pack-first", "full-after-pack", "two-full", "full-at-the-end"],
    )
    def test_chain_is_one_full_copy_then_packs(self, kinds: tuple[str, ...]) -> None:
        raw = _BODY.model_dump(mode="json")
        raw["chain"] = [{**raw["chain"][0], "kind": kind} for kind in kinds]
        with pytest.raises(ValueError, match="a chain is one full copy followed by packs"):
            models.HeadBody.model_validate_json(json.dumps(raw))

    def test_head_above_the_reader_limit_is_never_sealed(self) -> None:
        """No reader accepts a head above MAX_HEAD_BYTES: sealing one fails before any write."""

        class _LargeBody(BaseModel):
            model_config = _STRICT
            blob: str

        with pytest.raises(ValueError, match=r"^head of \d+ bytes exceeds 1048576$"):
            head.seal_head(
                header=_header(), body=_LargeBody(blob="x" * head.MAX_HEAD_BYTES), data_key=secrets.token_bytes(32)
            )

    def test_deep_json_header_is_integrity_error(self) -> None:
        header_bytes = b"[" * 60000
        data = head.MAGIC + len(header_bytes).to_bytes(4, "big") + header_bytes
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
            head.read_header(data + bytes(codec.NONCE_BYTES), instance_id=_ID)

    @pytest.mark.parametrize("data", [b"", b"NOTAHEAD" + bytes(20), head.MAGIC + (2**31).to_bytes(4, "big")])
    def test_garbage_fails(self, data: bytes) -> None:
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
            head.open_head(data, self._RING, instance_id=_ID)


class TestGoldenVectors:
    """Format permanence: stored bytes open, and sealing the same inputs gives the same bytes.
    A failure here means stored instances no longer load: change the layout on purpose
    (MAGIC, keys) and regenerate the vectors, never by accident."""

    def test_block_record(self) -> None:
        v = _VECTORS
        data_key = bytes.fromhex(v["data_key"])
        aad = codec.block_aads(instance_id=v["instance_id"], commit_id=v["commit_id"], stream=v["block_stream"])(
            v["block_offset"]
        )
        block = (v["block_text"].encode() * 3000)[: 64 * 1024]  # one full block
        cipher = AESGCM(data_key)
        assert codec.open_block(cipher, bytes.fromhex(v["record"]), aad, instance_id=v["instance_id"]) == block
        assert codec.seal_block_with_nonce(cipher, bytes(12), block, aad).hex() == v["record"]

    def test_head(self) -> None:
        v = _VECTORS
        kek = keyring.Key(id=v["kek_id"], secret=bytes.fromhex(v["kek_secret"]))
        assert keyring.key_id(kek.secret) == v["kek_id"]
        data = bytes.fromhex(v["head"])
        header = head.read_header(data, instance_id=v["instance_id"])
        assert header == models.HeadHeader.model_validate_json(json.dumps(v["header"]))
        opened_header, body, data_key = head.open_head(data, keyring.Keyring(keys=(kek,)), instance_id=v["instance_id"])
        assert opened_header == header
        assert data_key == bytes.fromhex(v["data_key"])
        assert body.model_dump(mode="json") == v["body"]
        resealed = head.seal_head_with_nonce(header=header, body=body, data_key=data_key, nonce=bytes(range(12)))
        assert resealed == data

    def test_body_objects_of_the_head_chain(self, tmp_path: Path) -> None:
        """The chain the golden head names rebuilds known streams (change, discard, shrink)."""
        v = _VECTORS
        data_key = bytes.fromhex(v["data_key"])
        body = models.HeadBody.model_validate_json(json.dumps(v["body"]))
        targets = {models.STREAM_VMSTATE: tmp_path / "vmstate", models.STREAM_HOME: tmp_path / "home"}
        for target in targets.values():
            pack.new_stream_file(target)
        with contextlib.ExitStack() as files:
            fds = {s: files.enter_context(target.open("r+b")).fileno() for s, target in targets.items()}
            for entry in body.chain:
                obj = tmp_path / entry.commit_id
                obj.write_bytes(bytes.fromhex(v["objects"][entry.commit_id]))
                pack.apply_object(
                    obj,
                    data_key=data_key,
                    instance_id=v["instance_id"],
                    commit_id=entry.commit_id,
                    targets=fds,
                    sizes=dict(enumerate(entry.stream_sizes)),
                )
                digests = [hashlib.sha256(targets[s].read_bytes()).hexdigest() for s in sorted(targets)]
                assert digests == v["streams_after"][entry.commit_id], entry.kind


def test_instance_errors_pickle_with_hint() -> None:
    """Instance errors keep instance id, hint note, code and action across processes."""
    error = ConfigError(
        "missing",
        code=ErrorCode.INSTANCE_KEY_MISSING,
        context={"key_id": "a" * 64},
        hint="add the key",
        instance_id=_ID,
    )
    restored = pickle.loads(pickle.dumps(error))  # noqa: S301 - bytes produced by this test
    assert type(restored) is ConfigError
    assert restored.instance_id == _ID
    assert restored.context == {"key_id": "a" * 64}
    assert restored.code == "INSTANCE_KEY_MISSING"
    assert restored.action == "FIX_CONFIG"
    assert restored.__notes__ == ["Hint: add the key"]

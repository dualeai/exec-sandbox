"""Body objects: diff against a baseline, seal, and rebuild streams from a chain."""

import contextlib
import errno
import json
import os
import random
import secrets
import signal
import time
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from exec_sandbox.exceptions import ErrorCode, InstanceGoneError, RouteError
from exec_sandbox.instance import codec, ids
from exec_sandbox.instance.codec import BLOCK_BYTES
from exec_sandbox.instance.pack import PACK_MAGIC, apply_object, new_stream_file, write_object

_TRAILER = 8 + 4 + len(PACK_MAGIC)

_KEY = secrets.token_bytes(32)
_ID = ids.new_id()
_SIZE = 64 * BLOCK_BYTES


def _stream(path: Path, writes: dict[int, bytes], *, size: int = _SIZE) -> Path:
    """A sparse stream file with `writes` (block number -> bytes) at their blocks."""
    with path.open("wb") as f:
        f.truncate(size)
        for block_no, data in writes.items():
            f.seek(block_no * BLOCK_BYTES)
            f.write(data)
    return path


def _save(tmp: Path, name: str, streams: Mapping[int, tuple[Path, Path | None]]) -> tuple[Path, str]:
    """Write the object of (new file, baseline or None) per stream; it returns its own size."""
    commit = ids.new_id()
    obj = tmp / f"{name}.obj"
    assert write_object(obj, data_key=_KEY, instance_id=_ID, commit_id=commit, streams=streams) == obj.stat().st_size
    return obj, commit


def _index(obj: Path, commit: str) -> dict[str, list[list[int]]]:
    """The object's index, opened: {stream: [[block number, record length], ...]} in object order."""
    data = obj.read_bytes()
    trailer = data[-_TRAILER:]
    index_offset, index_len = int.from_bytes(trailer[:8], "big"), int.from_bytes(trailer[8:12], "big")
    sealed = data[index_offset : index_offset + index_len]
    aad = codec.object_index_aad(instance_id=_ID, commit_id=commit)
    return json.loads(AESGCM(_KEY).decrypt(sealed[:12], sealed[12:], aad))


def _stored_blocks(tmp: Path, streams: Mapping[int, tuple[Path, Path | None]]) -> list[int]:
    """Block numbers the object of stream 0 stores."""
    obj, commit = _save(tmp, "obj", streams)
    return [block_no for block_no, _ in _index(obj, commit)["0"]]


def _load(tmp: Path, chain: list[tuple[Path, str, dict[int, int]]]) -> dict[int, Path]:
    """Apply (object, commit, stream sizes at that commit) in order to empty files, open once."""
    targets = {s: tmp / f"loaded-{s}" for s in chain[0][2]}
    for target in targets.values():
        new_stream_file(target)
    with contextlib.ExitStack() as files:
        fds = {s: files.enter_context(target.open("r+b")).fileno() for s, target in targets.items()}
        for obj, commit, sizes in chain:
            apply_object(obj, data_key=_KEY, instance_id=_ID, commit_id=commit, targets=fds, sizes=sizes)
    return targets


class TestStoredBlocks:
    def test_full_copy_skips_holes_and_zero_blocks(self, tmp_path: Path) -> None:
        new = _stream(tmp_path / "new", {3: b"x" * 10, 5: bytes(BLOCK_BYTES), 9: b"y"})
        assert _stored_blocks(tmp_path, {0: (new, None)}) == [3, 9]

    def test_pack_lists_changes_including_discarded_blocks(self, tmp_path: Path) -> None:
        # 8 is discarded: a hole in `new`, data in the baseline only. 64 MiB streams: APFS
        # reports the holes of a large sparse file only (a 4 MiB one reads as all data).
        size = 1024 * BLOCK_BYTES
        base = _stream(tmp_path / "base", {1: b"a", 2: b"b", 8: b"c"}, size=size)
        new = _stream(tmp_path / "new", {1: b"a", 2: b"B", 7: b"d"}, size=size)
        assert _stored_blocks(tmp_path, {0: (new, base)}) == [2, 7, 8]

    def test_blocks_past_the_new_size_are_not_stored(self, tmp_path: Path) -> None:
        base = _stream(tmp_path / "base", {40: b"z"})
        new = _stream(tmp_path / "new", {}, size=10 * BLOCK_BYTES)
        assert _stored_blocks(tmp_path, {0: (new, base)}) == []

    def test_last_partial_block_is_sealed_as_it_is(self, tmp_path: Path) -> None:
        """A stream whose size is not a multiple of the block: its last record holds only
        the bytes up to the end (never zero padding), the bytes every release wrote."""
        new = _stream(tmp_path / "new", {2: b"tail"}, size=2 * BLOCK_BYTES + 4)
        obj, commit = _save(tmp_path, "obj", {0: (new, None)})
        ((block_no, length),) = _index(obj, commit)["0"]
        aad = codec.block_aads(instance_id=_ID, commit_id=commit, stream=0)(block_no * BLOCK_BYTES)
        assert codec.open_block(AESGCM(_KEY), obj.read_bytes()[:length], aad, instance_id=_ID) == b"tail"

    def test_without_seek_data_every_block_is_read(self, tmp_path: Path) -> None:
        """A filesystem without SEEK_DATA (EINVAL): every block counts as data, none is lost."""
        path = _stream(tmp_path / "s", {0: b"a", 3: b"b"})
        real_lseek = os.lseek

        def no_seek_data(fd: int, pos: int, how: int) -> int:
            if how == os.SEEK_DATA:
                raise OSError(errno.EINVAL, "Invalid argument")
            return real_lseek(fd, pos, how)

        with patch.object(os, "lseek", no_seek_data):
            assert _stored_blocks(tmp_path, {0: (path, None)}) == [0, 3]


class TestChain:
    def test_full_copy_then_packs_rebuild_every_state(self, tmp_path: Path) -> None:
        ram1 = _stream(tmp_path / "ram1", {0: b"r0", 4: os.urandom(BLOCK_BYTES)})
        home1 = _stream(tmp_path / "home1", {10: b"h10"})
        full = _save(tmp_path, "full", {0: (ram1, None), 1: (home1, None)})

        ram2 = _stream(tmp_path / "ram2", {0: b"r0-v2", 4: ram1.read_bytes()[4 * BLOCK_BYTES : 5 * BLOCK_BYTES]})
        home2 = _stream(tmp_path / "home2", {11: b"h11"})  # block 10 discarded
        pack = _save(tmp_path, "pack", {0: (ram2, ram1), 1: (home2, home1)})

        (tmp_path / "a").mkdir()
        sizes = {0: _SIZE, 1: _SIZE}
        loaded = _load(tmp_path / "a", [(*full, sizes)])
        assert loaded[0].read_bytes() == ram1.read_bytes()
        assert loaded[1].read_bytes() == home1.read_bytes()
        (tmp_path / "b").mkdir()
        loaded = _load(tmp_path / "b", [(*full, sizes), (*pack, sizes)])
        assert loaded[0].read_bytes() == ram2.read_bytes()
        assert loaded[1].read_bytes() == home2.read_bytes()

    def test_stream_shrinks_then_grows(self, tmp_path: Path) -> None:
        """A block past a shorter commit's end must not come back when the stream grows again."""
        big = _stream(tmp_path / "big", {2: b"early", 40: b"past the cut"})
        small = _stream(tmp_path / "small", {2: b"early"}, size=10 * BLOCK_BYTES)
        regrown = _stream(tmp_path / "regrown", {2: b"early", 50: b"new tail"})
        chain = [
            (*_save(tmp_path, "full", {0: (big, None)}), {0: _SIZE}),
            (*_save(tmp_path, "pack1", {0: (small, big)}), {0: 10 * BLOCK_BYTES}),
            (*_save(tmp_path, "pack2", {0: (regrown, small)}), {0: _SIZE}),
        ]
        for n in range(1, 4):
            (tmp_path / str(n)).mkdir()
            loaded = _load(tmp_path / str(n), chain[:n])
            assert loaded[0].read_bytes() == [big, small, regrown][n - 1].read_bytes(), n

    def test_streams_of_many_codec_tasks(self, tmp_path: Path) -> None:
        """Streams of hundreds of blocks: many codec tasks run at once (the pool), yet the
        records are in block order and the chain rebuilds both states byte for byte. The pack
        holds exactly the changed and discarded blocks, also in a run where most blocks are equal."""
        rng = random.Random(21)
        size = 700 * BLOCK_BYTES + 100  # the last block is partial
        base_blocks = {b: rng.randbytes(BLOCK_BYTES) for b in range(0, 700, 2)}
        base_blocks[700] = rng.randbytes(100)
        changed = {b: rng.randbytes(BLOCK_BYTES) for b in (10, 64, 66, 300, 302, 698)}
        new_blocks = {b: data for b, data in base_blocks.items() if b not in (40, 500)} | changed | {700: b"t" * 100}
        base = _stream(tmp_path / "base", base_blocks, size=size)
        new = _stream(tmp_path / "new", new_blocks, size=size)
        full = _save(tmp_path, "full", {0: (base, None)})
        pack = _save(tmp_path, "pack", {0: (new, base)})
        assert [b for b, _ in _index(*full)["0"]] == [*range(0, 700, 2), 700]
        assert [b for b, _ in _index(*pack)["0"]] == [10, 40, 64, 66, 300, 302, 500, 698, 700]
        (tmp_path / "a").mkdir()
        loaded = _load(tmp_path / "a", [(*full, {0: size}), (*pack, {0: size})])
        assert loaded[0].read_bytes() == new.read_bytes()

    def test_forked_child_saves_with_its_own_codec_threads(self, tmp_path: Path) -> None:
        """A process forked after a save (multiprocessing's fork start method, a pre-fork
        server) has none of the parent's codec threads: its own save must still run."""
        ram = _stream(tmp_path / "ram", {b: os.urandom(BLOCK_BYTES) for b in range(40)})
        _save(tmp_path, "parent", {0: (ram, None)})  # the parent's codec threads run now
        with warnings.catch_warnings():  # fork of a process with threads: the case under test
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:  # the child never returns into pytest
            status = 1
            try:
                _save(tmp_path, "child", {0: (ram, None)})
                status = 0
            finally:
                os._exit(status)
        deadline = time.monotonic() + 10
        while (done := os.waitpid(pid, os.WNOHANG)) == (0, 0):
            if time.monotonic() > deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                pytest.fail("the forked child's save hung on codec threads it does not have")
            time.sleep(0.05)
        assert os.waitstatus_to_exitcode(done[1]) == 0

    def test_pack_is_small_for_a_small_change(self, tmp_path: Path) -> None:
        base = _stream(tmp_path / "base", {b: os.urandom(BLOCK_BYTES) for b in range(32)})
        new = _stream(
            tmp_path / "new",
            {**{b: base.read_bytes()[b * BLOCK_BYTES : (b + 1) * BLOCK_BYTES] for b in range(32)}, 5: b"changed"},
        )
        full, _ = _save(tmp_path, "full", {0: (new, None)})
        pack, _ = _save(tmp_path, "pack", {0: (new, base)})
        assert pack.stat().st_size < full.stat().st_size / 16


class TestWriteFailure:
    def test_failed_write_leaves_no_object(self, tmp_path: Path) -> None:
        """ENOSPC or a vanished stream mid-write: no partial object stays behind."""
        ram = _stream(tmp_path / "ram", {0: b"x"})
        streams = {0: (ram, None), 1: (tmp_path / "gone", None)}
        with pytest.raises(FileNotFoundError):
            write_object(tmp_path / "obj", data_key=_KEY, instance_id=_ID, commit_id=ids.new_id(), streams=streams)
        assert not (tmp_path / "obj").exists()


class TestTamper:
    @staticmethod
    def _object(tmp_path: Path) -> tuple[Path, str]:
        ram = _stream(tmp_path / "ram", {0: b"zero", 1: b"one", 2: b"two"})
        return _save(tmp_path, "full", {0: (ram, None)})

    def _apply(self, tmp_path: Path, obj: Path, commit: str) -> None:
        target = tmp_path / "target"
        target.unlink(missing_ok=True)
        new_stream_file(target)
        with target.open("r+b") as out:
            apply_object(
                obj, data_key=_KEY, instance_id=_ID, commit_id=commit, targets={0: out.fileno()}, sizes={0: _SIZE}
            )

    def test_every_byte_flip_fails(self, tmp_path: Path) -> None:
        obj, commit = self._object(tmp_path)
        data = obj.read_bytes()
        for i in range(len(data)):
            obj.write_bytes(data[:i] + bytes([data[i] ^ 1]) + data[i + 1 :])
            with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
                self._apply(tmp_path, obj, commit)

    @staticmethod
    def _split(obj: Path, commit: str) -> tuple[list[bytes], bytes]:
        """An attacker's view: the records (lengths from the index, which the attacker
        may read: it holds no secret) and the sealed index."""
        data = obj.read_bytes()
        trailer = data[-_TRAILER:]
        index_offset, index_len = int.from_bytes(trailer[:8], "big"), int.from_bytes(trailer[8:12], "big")
        sealed = data[index_offset : index_offset + index_len]
        records, position = [], 0
        for _, length in _index(obj, commit)["0"]:
            records.append(data[position : position + length])
            position += length
        return records, sealed

    @staticmethod
    def _join(records: list[bytes], sealed: bytes) -> bytes:
        """Records, the index unchanged, and a trailer with the offsets fixed."""
        body = b"".join(records)
        return body + sealed + len(body).to_bytes(8, "big") + len(sealed).to_bytes(4, "big") + PACK_MAGIC

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(lambda r: r[1:], id="dropped"),
            pytest.param(lambda r: [r[0], *r], id="extra"),
            pytest.param(lambda r: [r[0], r[2], r[1]], id="swapped"),
        ],
    )
    def test_reordered_records_fail(self, tmp_path: Path, change: Callable[[list[bytes]], list[bytes]]) -> None:
        """The index is authenticated and each record's AAD binds its offset: dropping,
        adding or swapping records (offsets fixed, so the object is well formed) fails."""
        obj, commit = self._object(tmp_path)
        records, sealed = self._split(obj, commit)
        obj.write_bytes(self._join(change(records), sealed))
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY) as exc:
            self._apply(tmp_path, obj, commit)
        assert "out of bounds" not in str(exc.value)

    def test_object_of_another_commit_fails(self, tmp_path: Path) -> None:
        obj, _ = self._object(tmp_path)
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
            self._apply(tmp_path, obj, ids.new_id())

    @staticmethod
    def _seal_index(plain: bytes, commit: str) -> bytes:
        """An index sealed with the data key, as write_object seals it: it authenticates."""
        nonce = secrets.token_bytes(codec.NONCE_BYTES)
        return nonce + AESGCM(_KEY).encrypt(nonce, plain, codec.object_index_aad(instance_id=_ID, commit_id=commit))

    def test_authenticated_index_this_release_cannot_read_routes(self, tmp_path: Path) -> None:
        """An index that authenticates but does not parse: a release of this library wrote it,
        so ROUTE, never RECREATE."""
        commit = ids.new_id()
        obj = tmp_path / "obj"
        obj.write_bytes(self._join([], self._seal_index(b'{"x": 1}', commit)))
        with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
            self._apply(tmp_path, obj, commit)
        assert exc.value.action == "ROUTE"
        assert exc.value.required == {"release": "one that reads this object index"}

    @pytest.mark.parametrize(
        ("delta", "message"),
        [
            pytest.param(1, "object records overrun the index", id="longer"),
            pytest.param(-1, "object has records the index does not list", id="shorter"),
        ],
    )
    def test_authenticated_index_that_disagrees_with_the_records_fails(
        self, tmp_path: Path, delta: int, message: str
    ) -> None:
        """An index that authenticates, whose record lengths do not add up to its offset (records
        unchanged): refused before any block is written, with the reason."""
        obj, commit = self._object(tmp_path)
        records, _ = self._split(obj, commit)
        index = _index(obj, commit)
        index["0"][-1][1] += delta
        obj.write_bytes(self._join(records, self._seal_index(json.dumps(index).encode(), commit)))
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY) as exc:
            self._apply(tmp_path, obj, commit)
        assert str(exc.value) == f"Instance {_ID}: {message}."
        assert (tmp_path / "target").stat().st_size == 0

    def test_unknown_stream_routes(self, tmp_path: Path) -> None:
        """The index authenticates, so this library wrote it: ROUTE, never RECREATE."""
        ram = _stream(tmp_path / "ram", {0: b"x"})
        obj, commit = _save(tmp_path, "full", {7: (ram, None)})
        with pytest.raises(
            RouteError, match="streams", check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH
        ) as exc:
            self._apply(tmp_path, obj, commit)
        assert exc.value.action == "ROUTE"

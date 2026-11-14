"""The store contract (store.py) on every store: S3 against an in-process moto server, local on a tmp folder.

ETags are compared only inside their group (store.py): `get` and conditional
writes on one side, `upload` and `list_objects` on the other.
"""

import asyncio
import os
import stat
import threading
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from exec_sandbox.exceptions import ErrorCode, InstanceGoneError
from exec_sandbox.instance import local_store
from exec_sandbox.instance.local_store import LocalInstanceStore
from exec_sandbox.instance.s3_store import S3InstanceStore
from exec_sandbox.instance.store import Applied, InstanceStore, ListedObject, NotApplied, probe
from tests.instance_support import NO_VM_TIMEOUT
from tests.s3_faults import BUCKET

pytestmark = NO_VM_TIMEOUT

_MAX = 1024 * 1024
_TWO_PARTS_AND_ONE = 2 * 8 * 1024 * 1024 + 1
"""Two parts of the S3 store's multipart size (8 MiB) and one byte."""


@pytest.fixture(params=["s3", "local"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncGenerator[InstanceStore]:
    if request.param == "s3":
        endpoint_url, _ = request.getfixturevalue("moto_s3_server")
        async with S3InstanceStore.open(
            bucket=BUCKET,
            prefix=f"contract-{uuid.uuid4().hex}/",
            endpoint_url=endpoint_url,
            region="us-east-1",
        ) as s3:
            yield s3
    else:
        async with LocalInstanceStore.open(root=tmp_path / "store") as local:
            yield local


async def _children(store: InstanceStore, prefix: str) -> list[str]:
    return sorted([name async for name in store.list_children(prefix)])


async def _upload(store: InstanceStore, tmp_path: Path, key: str, data: bytes) -> str:
    src = tmp_path / f"src-{uuid.uuid4().hex}"
    src.write_bytes(data)
    try:
        return await store.upload(key, src=src)
    finally:
        src.unlink()


class TestSmallObjects:
    async def test_get_size_exists_delete(self, store: InstanceStore) -> None:
        written = await store.put_if_absent("s/a/head", data=b"body")
        assert isinstance(written, Applied)
        assert await store.get("s/a/head", max_bytes=_MAX) == (b"body", written.etag)
        assert await store.size("s/a/head") == 4
        assert await store.exists("s/a/head")
        await store.delete_many(["s/a/head", "s/never/head"])
        assert await store.get("s/a/head", max_bytes=_MAX) is None
        assert await store.size("s/a/head") is None
        assert not await store.exists("s/a/head")

    async def test_missing_key(self, store: InstanceStore) -> None:
        assert await store.get("s/none/head", max_bytes=_MAX) is None
        assert await store.size("s/none/head") is None

    async def test_get_refuses_object_above_limit(self, store: InstanceStore) -> None:
        await store.put_if_absent("k", data=bytes(100))
        with pytest.raises(
            InstanceGoneError, match="100 bytes; at most 99", check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY
        ):
            await store.get("k", max_bytes=99)

    async def test_put_if_absent(self, store: InstanceStore) -> None:
        first = await store.put_if_absent("s/a/head", data=b"first")
        assert isinstance(first, Applied)
        assert await store.put_if_absent("s/a/head", data=b"second") == NotApplied(maybe_applied=False)
        assert await store.get("s/a/head", max_bytes=_MAX) == (b"first", first.etag)

    async def test_put_if_match(self, store: InstanceStore) -> None:
        first = await store.put_if_absent("s/a/head", data=b"v1")
        assert isinstance(first, Applied)
        second = await store.put_if_match("s/a/head", data=b"v2", etag=first.etag)
        assert isinstance(second, Applied)
        assert second.etag != first.etag
        assert await store.get("s/a/head", max_bytes=_MAX) == (b"v2", second.etag)  # read after write
        stale = await store.put_if_match("s/a/head", data=b"v3", etag=first.etag)
        assert stale == NotApplied(maybe_applied=False)
        assert await store.get("s/a/head", max_bytes=_MAX) == (b"v2", second.etag)

    async def test_put_if_match_with_the_etag_of_get(self, store: InstanceStore) -> None:
        await store.put_if_absent("s/a/head", data=b"v1")
        got = await store.get("s/a/head", max_bytes=_MAX)
        assert got is not None
        assert isinstance(await store.put_if_match("s/a/head", data=b"v2", etag=got[1]), Applied)

    async def test_put_if_match_on_missing_key_never_applies(self, store: InstanceStore) -> None:
        other = await store.put_if_absent("s/other/head", data=b"x")
        assert isinstance(other, Applied)
        result = await store.put_if_match("s/missing/head", data=b"x", etag=other.etag)
        assert result == NotApplied(maybe_applied=False)
        assert await store.get("s/missing/head", max_bytes=_MAX) is None

    async def test_prefix_of_other_keys_is_no_object(self, store: InstanceStore, tmp_path: Path) -> None:
        """Key "a" while "a/b" is an object: no object (the local store has a folder there)."""
        await store.put_if_absent("a/b", data=b"x")
        assert await store.get("a", max_bytes=_MAX) is None
        assert await store.size("a") is None
        assert not await store.download("a", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None)
        assert not (tmp_path / "dest").exists()

    async def test_key_under_an_object_is_missing(self, store: InstanceStore, tmp_path: Path) -> None:
        """Key "a/b" while "a" is an object: a missing key (the local store has a file where a folder would be)."""
        await store.put_if_absent("a", data=b"x")
        assert await store.get("a/b", max_bytes=_MAX) is None
        assert await store.size("a/b") is None
        assert not await store.download("a/b", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None)
        assert not (tmp_path / "dest").exists()
        assert await store.list_objects("a/") == []

    async def test_cancelled_write_is_whole_or_absent(
        self, store: InstanceStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancel while the write runs: the call raises CancelledError, and the object is
        whole or absent. The local write (a thread job, which can end before its await on
        3.13+) is held in its write of the staging file until the cancel is sent; an S3 write
        cannot end in one loop turn."""
        data = os.urandom(_MAX)
        started, go_on = threading.Event(), threading.Event()
        if isinstance(store, LocalInstanceStore):
            write = local_store.pwrite_all

            def held_write(fd: int, chunk: bytes, offset: int) -> None:
                started.set()
                go_on.wait(10)
                write(fd, chunk, offset)

            monkeypatch.setattr(local_store, "pwrite_all", held_write)
        task = asyncio.create_task(store.put_if_absent("s/a/head", data=data))
        if isinstance(store, LocalInstanceStore):
            assert await asyncio.to_thread(started.wait, 10)
        else:
            await asyncio.sleep(0)  # the request is under way
        task.cancel()
        go_on.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        got = await store.get("s/a/head", max_bytes=_MAX)
        assert got is None or got[0] == data


class TestStreams:
    @pytest.mark.parametrize(
        "size",
        [
            pytest.param(0, id="empty"),
            pytest.param(1000, id="small"),
            pytest.param(_TWO_PARTS_AND_ONE, id="two-parts-and-one"),
        ],
    )
    async def test_round_trip(self, store: InstanceStore, tmp_path: Path, size: int) -> None:
        data = os.urandom(size)
        etag = await _upload(store, tmp_path, "s/a/b/c.p", data)
        assert etag
        assert await store.size("s/a/b/c.p") == size
        assert await store.list_objects("s/a/b/") == [ListedObject(key="s/a/b/c.p", size=size, etag=etag)]
        dest = tmp_path / "dest"
        assert await store.download("s/a/b/c.p", dest=dest, max_bytes=size, expected_size=size)
        assert dest.read_bytes() == data
        assert stat.S_IMODE(dest.stat().st_mode) == 0o600

    async def test_upload_again_gives_new_etag(self, store: InstanceStore, tmp_path: Path) -> None:
        first = await _upload(store, tmp_path, "assets/sha256/x", b"one")
        second = await _upload(store, tmp_path, "assets/sha256/x", b"other")
        assert second != first
        assert await store.list_objects("assets/") == [ListedObject(key="assets/sha256/x", size=5, etag=second)]

    async def test_download_missing_leaves_no_file(self, store: InstanceStore, tmp_path: Path) -> None:
        assert not await store.download("s/a/b/none.p", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None)
        assert not (tmp_path / "dest").exists()

    @pytest.mark.parametrize("expected_size", [None, 100], ids=["size-unknown", "size-known"])
    async def test_download_refuses_object_above_limit(
        self, store: InstanceStore, tmp_path: Path, expected_size: int | None
    ) -> None:
        await _upload(store, tmp_path, "k", bytes(100))
        with pytest.raises(
            InstanceGoneError, match="100 bytes; at most 99", check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY
        ):
            await store.download("k", dest=tmp_path / "dest", max_bytes=99, expected_size=expected_size)
        assert not (tmp_path / "dest").exists()

    async def test_download_refuses_another_size(self, store: InstanceStore, tmp_path: Path) -> None:
        await _upload(store, tmp_path, "k", bytes(1000))
        with pytest.raises(
            InstanceGoneError, match="1000 bytes; 999 expected", check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY
        ):
            await store.download("k", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=999)
        assert not (tmp_path / "dest").exists()

    async def test_download_never_overwrites(self, store: InstanceStore, tmp_path: Path) -> None:
        await _upload(store, tmp_path, "k", b"x")
        (tmp_path / "dest").write_bytes(b"mine")
        with pytest.raises(FileExistsError):
            await store.download("k", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None)
        assert (tmp_path / "dest").read_bytes() == b"mine"

    async def test_missing_source_is_a_local_error(self, store: InstanceStore, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await store.upload("k", src=tmp_path / "none")
        assert not await store.exists("k")


class TestListing:
    async def test_keys_sizes_and_etags(self, store: InstanceStore, tmp_path: Path) -> None:
        c1 = await _upload(store, tmp_path, "s/i/b/c1.f", b"full")
        c2 = await _upload(store, tmp_path, "s/i/b/c2.p", b"pack!")
        await _upload(store, tmp_path, "s/ib/b/c3.p", b"neighbour")  # same start, another instance
        await store.put_if_absent("s/i/head", data=b"head")
        assert await store.list_objects("s/i/b/") == [
            ListedObject(key="s/i/b/c1.f", size=4, etag=c1),
            ListedObject(key="s/i/b/c2.p", size=5, etag=c2),
        ]
        under_i = {o.key: o.size for o in await store.list_objects("s/i/")}
        assert under_i == {"s/i/b/c1.f": 4, "s/i/b/c2.p": 5, "s/i/head": 4}  # never s/ib/
        assert {o.key for o in await store.list_objects("")} == {"s/i/b/c1.f", "s/i/b/c2.p", "s/ib/b/c3.p", "s/i/head"}
        assert await store.list_objects("s/none/") == []

    async def test_deleted_keys_are_not_listed(self, store: InstanceStore, tmp_path: Path) -> None:
        await _upload(store, tmp_path, "s/i/b/c1.f", b"1")
        await _upload(store, tmp_path, "s/i/b/c2.f", b"2")
        await store.delete_many(["s/i/b/c1.f"])
        assert [o.key for o in await store.list_objects("s/i/")] == ["s/i/b/c2.f"]

    async def test_list_children(self, store: InstanceStore, tmp_path: Path) -> None:
        await store.put_if_absent("s/a/head", data=b"a")
        await _upload(store, tmp_path, "s/a/b/c.p", b"body")
        await _upload(store, tmp_path, "s/b/b/c.p", b"body")
        await store.put_if_absent("s/d/head", data=b"d")
        await store.put_if_absent("s/file", data=b"an object, not a folder")
        assert await _children(store, "s/") == ["a", "b", "d"]
        assert await _children(store, "s/a/") == ["b"]
        assert await _children(store, "") == ["s"]
        assert await _children(store, "none/") == []
        await store.delete_many(["s/d/head"])
        assert await _children(store, "s/") == ["a", "b"]  # a folder without objects is not listed

    async def test_delete_many_is_idempotent(self, store: InstanceStore, tmp_path: Path) -> None:
        await _upload(store, tmp_path, "s/i/b/c.p", b"body")
        await store.delete_many(["s/i/b/c.p", "s/i/head"])
        await store.delete_many(["s/i/b/c.p", "s/i/head"])
        await store.delete_many([])
        assert await store.list_objects("") == []
        assert await _children(store, "") == []


class TestProbe:
    async def test_probe_passes_and_cleans_up(self, store: InstanceStore, tmp_path: Path) -> None:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        await probe(store, scratch=scratch)
        assert await store.list_objects("probe/") == []
        assert await _children(store, "") == []
        assert not list(scratch.iterdir())

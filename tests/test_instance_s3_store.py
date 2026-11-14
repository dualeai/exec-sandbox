"""S3 instance store (aiobotocore) against a moto server: conditional writes, streaming, the probe."""

import asyncio
import collections
import os
import re
import stat
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from botocore.loaders import Loader
from botocore.retries.standard import ExponentialBackoff

from exec_sandbox import s3_parts
from exec_sandbox.exceptions import ConfigError, ErrorCode, InstanceGoneError, SandboxError, TransientError
from exec_sandbox.instance.s3_store import S3InstanceStore
from exec_sandbox.instance.store import Applied, InstanceStore, NotApplied, probe
from tests.instance_support import NO_VM_TIMEOUT, await_until, descriptors_of
from tests.s3_faults import BUCKET, After, Before, DropAfter, DropBefore, Edit, Fail, Hold, Request, Response, S3Proxy

pytestmark = NO_VM_TIMEOUT

_MAX = 1024 * 1024
_PART = s3_parts.PART_BYTES
"""Multipart part and ranged-GET size of the S3 store, read from the code (a tuning value: no setting
gives it). A larger object is a multipart upload and a download of several ranges (the round trip checks it)."""


@pytest.fixture
def prefix() -> str:
    """moto keeps one in-process backend for every server: give each test its own keys."""
    return f"inst-{uuid.uuid4().hex}/"


@pytest.fixture
async def store(moto_s3_server, prefix: str) -> AsyncGenerator[InstanceStore]:
    endpoint_url, _ = moto_s3_server
    async with S3InstanceStore.open(bucket=BUCKET, prefix=prefix, endpoint_url=endpoint_url, region="us-east-1") as s:
        yield s


@pytest.fixture
async def proxied(s3_proxy: S3Proxy) -> AsyncGenerator[tuple[InstanceStore, S3Proxy]]:
    """A store behind the fault proxy (tests/s3_faults.py): faults come from the wire, not from patches."""
    async with S3InstanceStore.open(
        bucket=BUCKET, prefix=s3_proxy.prefix, endpoint_url=s3_proxy.endpoint_url, region="us-east-1"
    ) as s:
        yield s, s3_proxy


class TestOpen:
    async def test_opens_share_api_models_not_credentials(
        self, s3_proxy: S3Proxy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An open makes no botocore loader of its own: a new loader lists every service folder
        and parses the S3 model again (40 to 300 ms for each open). Each open still reads the
        credentials again: a key set after the first open signs the requests of the next one."""
        signed_by: list[str] = []

        async def record_key(req: Request) -> None:
            found = re.search(r"Credential=([^/]+)/", req.headers["Authorization"])
            signed_by.append(found.group(1) if found else "")

        s3_proxy.add("PUT_IF", "", Before(record_key), times=None)
        loaders = 0
        new_loader = Loader.__init__

        def counted(self: Loader, *args: object, **kwargs: object) -> None:
            nonlocal loaders
            loaders += 1
            new_loader(self, *args, **kwargs)  # pyright: ignore[reportArgumentType]

        async def put_signed_by(access_key: str) -> None:
            monkeypatch.setenv("AWS_ACCESS_KEY_ID", access_key)
            async with S3InstanceStore.open(
                bucket=BUCKET, prefix=s3_proxy.prefix, endpoint_url=s3_proxy.endpoint_url, region="us-east-1"
            ) as store:
                assert isinstance(await store.put_if_absent(access_key, data=b"x"), Applied)

        await put_signed_by("first-key")  # makes the loader of the process, if no earlier test did
        monkeypatch.setattr(Loader, "__init__", counted)
        await put_signed_by("second-key")
        assert signed_by == ["first-key", "second-key"]
        assert loaders == 0


class TestSmallObjects:
    """Store contract rules live in test_instance_store_contract.py (both stores); here, S3 wire facts."""

    async def test_keys_live_under_the_store_prefix(self, store: InstanceStore, moto_s3_server, prefix: str) -> None:
        _, s3 = moto_s3_server
        await store.put_if_absent("s/a/head", data=b"body")
        assert s3.get_object(Bucket=BUCKET, Key=f"{prefix}s/a/head")["Body"].read() == b"body"
        assert s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)["KeyCount"] == 1

    @pytest.mark.parametrize(
        ("status", "code", "maybe_applied"),
        [
            pytest.param(404, "NoSuchKey", False, id="404"),
            pytest.param(409, "ConditionalRequestConflict", False, id="409"),
            pytest.param(412, "PreconditionFailed", False, id="412"),
            pytest.param(400, "InvalidRequest", True, id="400"),
            pytest.param(503, "ServiceUnavailable", True, id="503"),
        ],
    )
    async def test_only_refusal_statuses_prove_not_applied(
        self, proxied: tuple[InstanceStore, S3Proxy], status: int, code: str, maybe_applied: bool
    ) -> None:
        """404, 409 and 412 prove the write did not apply; any other answer may hide one that did."""
        store, proxy = proxied
        proxy.add("PUT_IF", "^k$", Fail(status, code))
        assert await store.put_if_absent("k", data=b"x") == NotApplied(maybe_applied=maybe_applied)


class TestStreaming:
    @pytest.mark.parametrize(
        ("size", "wire"),
        [
            pytest.param(0, (1, 0, 0), id="empty"),
            pytest.param(1000, (1, 0, 1), id="one-put"),
            pytest.param(2 * _PART + 12345, (0, 3, 3), id="multipart"),
        ],
    )
    async def test_round_trip(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path, size: int, wire: tuple[int, int, int]
    ) -> None:
        """The bytes come back; the wire shows one PUT or a multipart upload, and one ranged GET
        per part. `wire`: (single PUTs, multipart parts, ranged GETs). The ETag the upload gives
        is the one a LIST entry reports (the lineage check of a save compares the two)."""
        store, proxy = proxied
        puts, parts, ranges = wire
        src, dest = tmp_path / "src", tmp_path / "dest"
        src.write_bytes(os.urandom(size))
        etag = await store.upload("s/a/b/c.p", src=src)
        assert [(o.key, o.size, o.etag) for o in await store.list_objects("s/a/b/")] == [("s/a/b/c.p", size, etag)]
        assert re.fullmatch(r"[0-9a-f]{32}" + ("-3" if parts else ""), etag)  # MD5, or "<md5>-<parts>" (multipart)
        assert (proxy.count("PUT"), proxy.count("MPU_PART"), proxy.count("MPU_COMPLETE")) == (
            puts,
            parts,
            min(parts, 1),
        )
        assert await store.download("s/a/b/c.p", dest=dest, max_bytes=size, expected_size=None)
        assert proxy.count("GET_RANGE") == ranges
        assert dest.read_bytes() == src.read_bytes()
        assert stat.S_IMODE(dest.stat().st_mode) == 0o600

    @pytest.mark.parametrize(
        ("size", "wire"),
        [
            pytest.param(0, {"GET": 1}, id="empty"),
            pytest.param(_PART, {"GET": 1}, id="one-part"),
            pytest.param(_PART + 1, {"HEAD": 1, "GET_RANGE": 2}, id="two-parts"),
        ],
    )
    async def test_known_size_of_one_part_is_one_get(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path, size: int, wire: dict[str, int]
    ) -> None:
        """A body object (its size known) of at most one part: one plain GET, which reads one
        version. A larger one: a HEAD for the ETag, then one ranged GET per part."""
        store, proxy = proxied
        src, dest = tmp_path / "src", tmp_path / "dest"
        src.write_bytes(os.urandom(size))
        await store.upload("k", src=src)
        mark = len(proxy.log)
        assert await store.download("k", dest=dest, max_bytes=size, expected_size=size)
        assert collections.Counter(e.op for e in proxy.entries(since=mark)) == wire
        assert dest.read_bytes() == src.read_bytes()
        assert stat.S_IMODE(dest.stat().st_mode) == 0o600

    @pytest.mark.parametrize("change", ["replace", "delete"])
    async def test_object_changed_during_download(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path, change: str
    ) -> None:
        """Only a 404 reads as gone (False). A replaced object (412 on the pinned ETag)
        is a broken store: body objects are never replaced."""
        store, proxy = proxied
        (tmp_path / "src").write_bytes(b"old")
        await store.upload("k", src=tmp_path / "src")

        async def change_after_head(req: Request, resp: Response) -> None:
            if change == "replace":
                await asyncio.to_thread(proxy.bucket.write, "k", b"new")
            else:
                await asyncio.to_thread(proxy.bucket.delete, "k")

        proxy.add("HEAD", "^k$", After(change_after_head))
        if change == "replace":
            with pytest.raises(
                TransientError,
                match=r"PreconditionFailed|412",
                check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE,
            ):
                await store.download("k", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None)
        else:
            assert not await store.download("k", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None)
        assert not (tmp_path / "dest").exists()

    async def test_server_header_echo_is_bounded(self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path) -> None:
        store, proxy = proxied
        size = 2 * _PART + 1
        (tmp_path / "src").write_bytes(os.urandom(size))
        await store.upload("k", src=tmp_path / "src")

        def long_header(resp: Response) -> None:
            resp.headers["Content-Range"] = "x" * 8000  # below aiohttp's header limit (8190)

        proxy.add("GET_RANGE", "^k$", Edit(response=long_header), times=None)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await store.download("k", dest=tmp_path / "dest", max_bytes=size, expected_size=size)
        assert "xxxx" in str(exc.value)  # the header was read
        assert "x" * 65 not in str(exc.value)

    @pytest.mark.parametrize(
        ("fault", "text"),
        [
            # Which part fails first decides the text: a shortened last range of one byte is
            # invalid, so the store sends the whole object for it (longer than its range).
            pytest.param("ignore-range", r"range|shorter", id="ignore-range"),
            pytest.param("short-body", r"range|shorter", id="short-body"),
            pytest.param("long-body", r"^Instance store GET k: body longer than range ", id="long-body"),
        ],
    )
    async def test_bad_part_answer_is_unavailable(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path, fault: str, text: str
    ) -> None:
        """An endpoint that ignores Range, or cuts or pads a body, is never read as "object gone"."""
        store, proxy = proxied
        size = 2 * _PART + 1
        (tmp_path / "src").write_bytes(os.urandom(size))
        await store.upload("k", src=tmp_path / "src")

        def ignore_range(req: Request) -> None:
            req.headers.popall("Range")

        def one_byte_short(req: Request) -> None:
            first, last = req.headers["Range"].removeprefix("bytes=").split("-")
            req.headers["Range"] = f"bytes={first}-{int(last) - 1}"

        async def claim_full_range(req: Request, resp: Response) -> None:
            first, last = req.headers["Range"].removeprefix("bytes=").split("-")
            resp.headers["Content-Range"] = f"bytes {first}-{int(last) + 1}/{size}"

        def one_byte_more(resp: Response) -> None:
            resp.body += b"\0"  # Content-Range still names the asked range

        if fault == "ignore-range":
            proxy.add("GET_RANGE", "^k$", Edit(request=ignore_range), times=None)
        elif fault == "short-body":
            proxy.add("GET_RANGE", "^k$", Edit(request=one_byte_short), times=None)
            proxy.add("GET_RANGE", "^k$", After(claim_full_range), times=None)
        else:
            proxy.add("GET_RANGE", "^k$", Edit(response=one_byte_more), times=None)
        with pytest.raises(TransientError, match=text, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE):
            await store.download("k", dest=tmp_path / "dest", max_bytes=size, expected_size=None)
        assert not (tmp_path / "dest").exists()

    async def test_failed_part_aborts_the_upload(self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path) -> None:
        """A part the store does not take: the multipart upload is aborted (no stored parts stay)
        and the call fails as the part did (RETRY)."""
        store, proxy = proxied
        (tmp_path / "src").write_bytes(os.urandom(2 * _PART + 1))
        proxy.add("MPU_PART", "^k$", Fail(500, "InternalError"), times=None)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await store.upload("k", src=tmp_path / "src")
        assert str(exc.value).startswith("Instance store PUT k failed: InternalError.")
        assert exc.value.action == "RETRY"
        assert (proxy.count("MPU_ABORT", "^k$"), proxy.count("MPU_COMPLETE")) == (1, 0)
        assert not proxy.bucket.exists("k")

    async def test_cancelled_multipart_upload_is_aborted(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path
    ) -> None:
        store, proxy = proxied
        (tmp_path / "src").write_bytes(os.urandom(2 * _PART + 1))
        held = proxy.add("MPU_PART", "^k$", Hold(), times=None)
        task = asyncio.create_task(store.upload("k", src=tmp_path / "src"))
        await held.reached.wait()  # the upload is created; its parts wait at the proxy
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (proxy.count("MPU_ABORT", "^k$"), proxy.count("MPU_COMPLETE")) == (1, 0)
        held.remove()
        assert not proxy.bucket.exists("k")

    async def test_failed_head_before_ranged_reads_is_unavailable(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path
    ) -> None:
        """Only a 404 reads as gone: another failed HEAD is RETRY, and no file stays."""
        store, proxy = proxied
        (tmp_path / "src").write_bytes(os.urandom(1000))
        await store.upload("k", src=tmp_path / "src")
        proxy.add("HEAD", "^k$", Fail(500, "InternalError"))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await store.download("k", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None)
        assert str(exc.value).startswith("Instance store HEAD k failed:")
        assert exc.value.action == "RETRY"
        assert not (tmp_path / "dest").exists()

    async def test_head_answer_without_etag_is_unavailable(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path
    ) -> None:
        """The ranged reads are pinned to the HEAD's ETag: without one, nothing is read."""
        store, proxy = proxied
        (tmp_path / "src").write_bytes(os.urandom(1000))
        await store.upload("k", src=tmp_path / "src")

        def stripped(resp: Response) -> None:
            resp.headers.popall("ETag")

        proxy.add("HEAD", "^k$", Edit(response=stripped))
        mark = len(proxy.log)
        with pytest.raises(
            TransientError,
            match=r"^Instance store HEAD k: answer without its size or ETag",
            check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE,
        ):
            await store.download("k", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=None)
        assert proxy.count("GET_RANGE", since=mark) == 0
        assert not (tmp_path / "dest").exists()


class TestSizes:
    """A size other than the authenticated one: damaged only when the store says it twice.
    The first answer is the GET of an object of one part, or the HEAD before ranged GETs;
    the size is asked again (HEAD) before any body is read."""

    _PATHS = (
        pytest.param(1000, {"HEAD": 1, "GET": 1, "GET_RANGE": 0}, id="one-part"),
        pytest.param(_PART + 1000, {"HEAD": 2, "GET": 0, "GET_RANGE": 0}, id="ranged"),
    )

    @staticmethod
    async def _object(store: InstanceStore, tmp_path: Path, size: int) -> None:
        (tmp_path / "src").write_bytes(os.urandom(size))
        await store.upload("k", src=tmp_path / "src")

    @staticmethod
    def _asked(proxy: S3Proxy, mark: int) -> dict[str, int]:
        return {op: proxy.count(op, "^k$", since=mark) for op in ("HEAD", "GET", "GET_RANGE")}

    @pytest.mark.parametrize(("size", "asked"), _PATHS)
    async def test_one_wrong_size_answer_is_unavailable(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path, size: int, asked: dict[str, int]
    ) -> None:
        store, proxy = proxied
        await self._object(store, tmp_path, size)
        first = "GET" if size <= _PART else "HEAD"

        def zero(resp: Response) -> None:  # a broken proxy, once
            if first == "HEAD":
                resp.headers["Content-Length"] = "0"
            else:
                resp.body = b""

        proxy.add(first, "^k$", Edit(response=zero))
        mark = len(proxy.log)
        with pytest.raises(
            TransientError, match="0 bytes", check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE
        ):
            await store.download("k", dest=tmp_path / "dest", max_bytes=size, expected_size=size)
        assert self._asked(proxy, mark) == asked
        assert not (tmp_path / "dest").exists()

    @pytest.mark.parametrize(("size", "asked"), _PATHS)
    async def test_failed_second_size_answer_is_unavailable(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path, size: int, asked: dict[str, int]
    ) -> None:
        """The size is asked again (HEAD) before it counts as damaged: no answer to that is RETRY."""
        store, proxy = proxied
        await self._object(store, tmp_path, size)
        proxy.add("HEAD", "^k$", Fail(503, "ServiceUnavailable"), skip=asked["HEAD"] - 1)
        mark = len(proxy.log)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await store.download("k", dest=tmp_path / "dest", max_bytes=size, expected_size=size - 1)
        assert str(exc.value).startswith("Instance store HEAD k failed:")
        assert self._asked(proxy, mark) == asked
        assert not (tmp_path / "dest").exists()

    @pytest.mark.parametrize(("size", "asked"), _PATHS)
    async def test_same_wrong_size_twice_is_damaged(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path, size: int, asked: dict[str, int]
    ) -> None:
        store, proxy = proxied
        await self._object(store, tmp_path, size)
        mark = len(proxy.log)
        with pytest.raises(
            InstanceGoneError,
            match=f"{size} bytes; {size - 1} expected",
            check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY,
        ):
            await store.download("k", dest=tmp_path / "dest", max_bytes=size, expected_size=size - 1)
        assert self._asked(proxy, mark) == asked
        assert not (tmp_path / "dest").exists()


class TestListing:
    async def test_children_of_every_page(self, proxied: tuple[InstanceStore, S3Proxy]) -> None:
        """Children come from every page of the listing; an empty folder name (key "s//x") is no child."""
        store, proxy = proxied
        for key in ("s/a/head", "s/b/head", "s/c/head", "s//x"):
            proxy.bucket.write(key, b"x")

        def one_per_page(req: Request) -> None:
            req.query["max-keys"] = "1"

        proxy.add("LIST", "^s/$", Edit(request=one_per_page), times=None)
        assert [name async for name in store.list_children("s/")] == ["a", "b", "c"]
        assert proxy.count("LIST", "^s/$") == 4

    async def test_keys_outside_the_asked_prefix_are_refused(self, proxied: tuple[InstanceStore, S3Proxy]) -> None:
        """A store that ignores Prefix would hand delete_instance() every key."""
        store, proxy = proxied
        proxy.bucket.write("s/other/head", b"x")

        def ignores_prefix(req: Request) -> None:
            req.query["prefix"] = proxy.prefix

        proxy.add("LIST", "^s/mine/$", Edit(request=ignores_prefix))
        with pytest.raises(
            TransientError,
            match="outside the asked prefix",
            check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE,
        ):
            await store.list_objects("s/mine/")


class TestFailures:
    @pytest.mark.usefixtures("fake_aws_env")
    async def test_unreachable_store(self) -> None:
        """No answer: conditional writes may have applied (unknown); other calls raise RETRY."""
        async with S3InstanceStore.open(
            bucket=BUCKET, prefix="", endpoint_url="http://127.0.0.1:9", region="us-east-1"
        ) as unreachable:
            assert await unreachable.put_if_absent("k", data=b"x") == NotApplied(maybe_applied=True)
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
                await unreachable.get("k", max_bytes=_MAX)
            assert exc.value.action == "RETRY"

    async def test_missing_bucket_is_config_error(self, moto_s3_server) -> None:
        endpoint_url, _ = moto_s3_server
        async with S3InstanceStore.open(
            bucket=f"no-{uuid.uuid4().hex}", prefix="", endpoint_url=endpoint_url, region="us-east-1"
        ) as wrong:
            for call in (wrong.get("k", max_bytes=_MAX), wrong.put_if_absent("k", data=b"x")):
                with pytest.raises(
                    ConfigError, match="NoSuchBucket", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
                ) as exc:
                    await call
                assert exc.value.action == "FIX_CONFIG"

    @pytest.mark.parametrize(
        ("code", "error"),
        [
            pytest.param("AccessDenied", (ConfigError, ErrorCode.INSTANCE_CONFIG), id="access-denied"),
            pytest.param("InternalError", (TransientError, ErrorCode.INSTANCE_STORE_UNAVAILABLE), id="internal-error"),
        ],
    )
    async def test_delete_reports_kept_keys(
        self, proxied: tuple[InstanceStore, S3Proxy], code: str, error: tuple[type[SandboxError], ErrorCode]
    ) -> None:
        """DeleteObjects answers 200 with per-key errors: a kept key is never reported as deleted."""
        store, proxy = proxied
        proxy.bucket.write("k", b"x")

        def kept(resp: Response) -> None:
            resp.body = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<DeleteResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"<Error><Key>{proxy.prefix}k</Key><Code>{code}</Code><Message>kept</Message></Error>"
                "</DeleteResult>"
            ).encode()

        proxy.add("DELETE_MANY", "^k$", Edit(response=kept))
        with pytest.raises(error[0], match=code, check=lambda e: e.code is error[1]):
            await store.delete_many(["k"])


class TestStoreAnswers:
    """What a broken or hostile endpoint sends never escapes as raw text or the wrong action."""

    @pytest.mark.parametrize(
        ("status", "code", "error", "text"),
        [
            pytest.param(
                500,
                "AccessDenied",
                (TransientError, ErrorCode.INSTANCE_STORE_UNAVAILABLE),
                "AccessDenied",
                id="5xx-is-never-config",
            ),
            pytest.param(
                403, "AccessDenied", (ConfigError, ErrorCode.INSTANCE_CONFIG), "AccessDenied", id="403-is-config"
            ),
            pytest.param(
                400,
                "Bad\nInjected: line",
                (TransientError, ErrorCode.INSTANCE_STORE_UNAVAILABLE),
                "unrecognized error code",
                id="code-cleaned",
            ),
        ],
    )
    async def test_failure_classification(
        self,
        proxied: tuple[InstanceStore, S3Proxy],
        status: int,
        code: str,
        error: tuple[type[SandboxError], ErrorCode],
        text: str,
    ) -> None:
        store, proxy = proxied
        proxy.add("GET", "^k$", Fail(status, code))
        with pytest.raises(error[0], check=lambda e: e.code is error[1]) as exc:
            await store.get("k", max_bytes=_MAX)
        assert text in str(exc.value)
        assert "\n" not in str(exc.value)

    async def test_archived_object_names_the_storage_class(self, proxied: tuple[InstanceStore, S3Proxy]) -> None:
        """A GET of an object a lifecycle rule moved to an archive class (403 InvalidObjectState):
        FIX_CONFIG with its own message and hint, not the permission advice of other 403s."""
        store, proxy = proxied
        proxy.add("GET", "^k$", Fail(403, "InvalidObjectState"))
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG) as exc:
            await store.get("k", max_bytes=_MAX)
        assert "the object is in an archive storage class" in str(exc.value)
        assert exc.value.hint is not None
        assert exc.value.hint.startswith(
            "Lifecycle rules must not move objects under the instance prefix to an archive storage class"
        )
        assert "s3:GetObject" not in exc.value.hint

    async def test_get_answer_without_etag(self, proxied: tuple[InstanceStore, S3Proxy]) -> None:
        store, proxy = proxied
        await store.put_if_absent("k", data=b"x")

        def stripped(resp: Response) -> None:
            resp.headers.popall("ETag")

        proxy.add("GET", "^k$", Edit(response=stripped))
        with pytest.raises(
            TransientError, match="ETag", check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE
        ):
            await store.get("k", max_bytes=_MAX)

    @pytest.mark.parametrize("etag", ['""', '"' + "a" * 2048 + '"'], ids=["empty", "oversized"])
    async def test_unusable_etags(self, proxied: tuple[InstanceStore, S3Proxy], etag: str) -> None:
        """A GET with an unusable ETag: RETRY. A conditional PUT answered with one may have
        applied, so the commit protocol must read the head back (maybe_applied)."""
        store, proxy = proxied
        await store.put_if_absent("k", data=b"x")

        def swap(resp: Response) -> None:
            resp.headers["ETag"] = etag

        proxy.add("GET", "^k$", Edit(response=swap))
        with pytest.raises(
            TransientError, match="ETag", check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE
        ):
            await store.get("k", max_bytes=_MAX)
        proxy.add("PUT_IF", "^k2$", Edit(response=swap))
        assert await store.put_if_absent("k2", data=b"y") == NotApplied(maybe_applied=True)

    async def test_cancelled_download_leaves_no_file_and_no_fd(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path
    ) -> None:
        """Cancelled while one part is written and the other ranged GETs wait: no file stays,
        and its descriptor closes once the worker threads end (the close is shielded)."""
        store, proxy = proxied
        src, dest = tmp_path / "src", tmp_path / "dest"
        src.write_bytes(os.urandom(3 * _PART))
        await store.upload("big", src=src)
        held = proxy.add("GET_RANGE", "^big$", Hold(), times=None, skip=1)
        task = asyncio.create_task(store.download("big", dest=dest, max_bytes=4 * _PART, expected_size=None))
        await held.reached.wait()  # the file is open: a part waits at the proxy
        ((fd, inode),) = descriptors_of(dest)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        held.remove()

        def still_open() -> bool:
            try:
                return os.fstat(fd).st_ino == inode  # the number may be reused by another file since
            except OSError:
                return False

        await await_until(lambda: not still_open())  # the shielded close ends on its own, then closes the fd
        assert not dest.exists()


class TestHeadWriteIsSentOnce:
    """S3InstanceStore.open: no SDK retry and no aiohttp re-send for conditional head writes (a
    re-send that meets the write's own effect reads as refused). SDK retries are on here (the
    fake_aws_env fixture turns them off for every other client): each test counts the PUTs that
    reach the proxy for one call."""

    @pytest.fixture
    async def retrying(
        self, s3_proxy: S3Proxy, monkeypatch: pytest.MonkeyPatch
    ) -> AsyncGenerator[tuple[InstanceStore, S3Proxy]]:
        """A store whose clients may retry once (standard mode), without the backoff wait
        before the retry (a random 0 to 1 s)."""
        monkeypatch.setenv("AWS_MAX_ATTEMPTS", "2")
        monkeypatch.setenv("AWS_RETRY_MODE", "standard")
        monkeypatch.setattr(ExponentialBackoff, "delay_amount", lambda _self, _context: 0.0)
        async with S3InstanceStore.open(
            bucket=BUCKET, prefix=s3_proxy.prefix, endpoint_url=s3_proxy.endpoint_url, region="us-east-1"
        ) as s:
            yield s, s3_proxy

    async def test_reads_are_retried_by_the_sdk(self, retrying: tuple[InstanceStore, S3Proxy]) -> None:
        """Control: the same failure on a read is retried, so the counts below come from the head client."""
        store, proxy = retrying
        proxy.bucket.write("k", b"x")
        proxy.add("GET", "^k$", Fail(503, "ServiceUnavailable"))  # once
        got = await store.get("k", max_bytes=_MAX)
        assert got is not None
        assert got[0] == b"x"
        assert [e.status for e in proxy.entries("GET", "^k$")] == [503, 200]

    async def test_failed_head_put_is_not_retried_by_the_sdk(self, retrying: tuple[InstanceStore, S3Proxy]) -> None:
        store, proxy = retrying
        proxy.add("PUT_IF", "^head$", Fail(503, "ServiceUnavailable"), times=None)
        assert await store.put_if_absent("head", data=b"x") == NotApplied(maybe_applied=True)
        assert proxy.count("PUT_IF") == 1

    async def test_unanswered_head_put_is_not_sent_again(self, retrying: tuple[InstanceStore, S3Proxy]) -> None:
        store, proxy = retrying
        proxy.add("PUT_IF", "^head$", DropAfter(), times=None)
        assert await store.put_if_absent("head", data=b"x") == NotApplied(maybe_applied=True)
        assert proxy.count("PUT_IF") == 1

    async def test_dropped_connection_is_not_sent_again(self, retrying: tuple[InstanceStore, S3Proxy]) -> None:
        """aiohttp sends an idempotent PUT again after a dropped connection: the head client does
        not, even when the store never saw the first send (the commit protocol reads back)."""
        store, proxy = retrying
        proxy.add("PUT_IF", "^head$", DropBefore())  # once: a re-send would pass and apply
        assert await store.put_if_absent("head", data=b"x") == NotApplied(maybe_applied=True)
        assert proxy.entries("PUT_IF") == [("PUT_IF", "head", 0)]
        assert not proxy.bucket.exists("head")


class TestProbe:
    @pytest.mark.parametrize("status", [0, 429, 503])
    async def test_no_answer_is_unavailable_not_config(
        self, proxied: tuple[InstanceStore, S3Proxy], status: int, tmp_path: Path
    ) -> None:
        """One 503 or timeout must not tell the operator the store lacks conditional writes."""
        store, proxy = proxied
        fault = DropBefore() if status == 0 else Fail(status, "SlowDown" if status == 429 else "ServiceUnavailable")
        proxy.add("PUT_IF", "^probe/", fault, times=None)
        with pytest.raises(
            TransientError, match="no usable answer", check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE
        ):
            await probe(store, scratch=tmp_path)
        assert {e.status for e in proxy.entries("PUT_IF")} == {status}

    async def test_store_ignoring_conditions_is_refused(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path
    ) -> None:
        """A store that applies If-None-Match PUTs unconditionally fails the probe by name."""
        store, proxy = proxied

        def unconditional(req: Request) -> None:
            req.headers.popall("If-None-Match", None)

        proxy.add("PUT_IF", "^probe/", Edit(request=unconditional), times=None)
        with pytest.raises(
            ConfigError,
            match="create-only write of an existing key failed",
            check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG,
        ):
            await probe(store, scratch=tmp_path)

    @pytest.mark.parametrize(
        ("fault", "error", "message"),
        [
            pytest.param(
                "new-key-refused",
                (ConfigError, ErrorCode.INSTANCE_CONFIG),
                "Instance store does not support conditional writes: create-only write of a new key failed.",
                id="new-key-refused",
            ),
            pytest.param(
                "existing-key-unanswered",
                (TransientError, ErrorCode.INSTANCE_STORE_UNAVAILABLE),
                "Instance store probe: create-only write of an existing key got no usable answer.",
                id="existing-key-unanswered",
            ),
            pytest.param(
                "read-missing",
                (ConfigError, ErrorCode.INSTANCE_CONFIG),
                "Instance store does not support conditional writes: read after a refused create-only write failed.",
                id="read-missing",
            ),
            pytest.param(
                "read-other-bytes",
                (ConfigError, ErrorCode.INSTANCE_CONFIG),
                "Instance store does not support conditional writes: read after a refused create-only write failed.",
                id="read-other-bytes",
            ),
            pytest.param(
                "etag-unchanged",
                (ConfigError, ErrorCode.INSTANCE_CONFIG),
                "Instance store does not support conditional writes: ETag change after an If-Match write failed.",
                id="etag-unchanged",
            ),
            pytest.param(
                "read-after-if-match",
                (ConfigError, ErrorCode.INSTANCE_CONFIG),
                "Instance store does not support conditional writes: read after an If-Match write failed.",
                id="read-after-if-match",
            ),
        ],
    )
    async def test_each_step_names_its_failure(
        self,
        proxied: tuple[InstanceStore, S3Proxy],
        tmp_path: Path,
        fault: str,
        error: tuple[type[SandboxError], ErrorCode],
        message: str,
    ) -> None:
        """The probe on k1: create-only (applied), create-only again (refused), read ("first"),
        If-Match with that ETag (applied, a new ETag), read ("third"). A wrong answer to one step
        fails the start by its name (FIX_CONFIG); no answer, RETRY. The scratch keys are removed."""
        store, proxy = proxied
        k1 = "^probe/[0-9a-f]+/k1$"
        etags: list[str] = []

        async def record_etag(req: Request, resp: Response) -> None:
            etags.append(resp.headers["ETag"])

        def other_bytes(resp: Response) -> None:
            resp.body = b"other"

        def old_etag(resp: Response) -> None:
            resp.headers["ETag"] = etags[0]

        match fault:
            case "new-key-refused":
                proxy.add("PUT_IF", k1, Fail(412, "PreconditionFailed"))
            case "existing-key-unanswered":
                proxy.add("PUT_IF", k1, DropBefore(), skip=1, times=None)
            case "read-missing":
                proxy.add("GET", k1, Fail(404, "NoSuchKey"))
            case "read-other-bytes":
                proxy.add("GET", k1, Edit(response=other_bytes))
            case "etag-unchanged":
                proxy.add("GET", k1, After(record_etag))
                proxy.add("PUT_IF", k1, Edit(response=old_etag), when=lambda req: "If-Match" in req.headers)
            case "read-after-if-match":
                proxy.add("GET", k1, Edit(response=other_bytes), skip=1)
        with pytest.raises(error[0], check=lambda e: e.code is error[1]) as exc:
            await probe(store, scratch=tmp_path)
        assert str(exc.value) == message
        assert exc.value.action == ("RETRY" if error[1] is ErrorCode.INSTANCE_STORE_UNAVAILABLE else "FIX_CONFIG")
        assert proxy.bucket.listing("probe/") == []

    async def test_listing_with_other_etags_is_refused(
        self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path
    ) -> None:
        """A store whose LIST reports other ETags than its uploads would make every save a full
        copy (the lineage check fails): the probe refuses it at start (FIX_CONFIG)."""
        store, proxy = proxied

        def other_etags(resp: Response) -> None:
            resp.body = re.sub(rb"<ETag>.*?</ETag>", b"<ETag>&#34;" + b"0" * 32 + b"&#34;</ETag>", resp.body)

        proxy.add("LIST", "^probe/", Edit(response=other_etags))
        with pytest.raises(
            ConfigError,
            match="listing does not report the size and ETag",
            check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG,
        ) as exc:
            await probe(store, scratch=tmp_path)
        assert exc.value.action == "FIX_CONFIG"
        assert proxy.bucket.listing("probe/") == []  # scratch keys removed


class TestUnsupportedConditionalWrites:
    async def test_501_on_a_conditional_put_is_fix_config(self, proxied: tuple[InstanceStore, S3Proxy]) -> None:
        """S3 answers 501 to a header it does not support: a store that can never commit a
        head is a configuration error (FIX_CONFIG), not a store to retry."""
        store, proxy = proxied
        proxy.add("PUT_IF", "", Fail(501, "NotImplemented"), times=None)
        with pytest.raises(
            ConfigError,
            match="does not support conditional writes",
            check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG,
        ) as exc:
            await store.put_if_absent("k", data=b"x")
        assert exc.value.action == "FIX_CONFIG"

    async def test_probe_reports_it_at_start(self, proxied: tuple[InstanceStore, S3Proxy], tmp_path: Path) -> None:
        store, proxy = proxied
        proxy.add("PUT_IF", "", Fail(501, "NotImplemented"), times=None)
        with pytest.raises(
            ConfigError, match="conditional writes", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ):
            await probe(store, scratch=tmp_path)

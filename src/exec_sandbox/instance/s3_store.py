"""S3 instance store: an S3-API bucket with conditional writes (aiobotocore, the s3 extra).

Supported stores honour, on PutObject, ``If-None-Match: *`` (create only) and
``If-Match: <etag>`` (replace only that version): AWS S3, Cloudflare R2, Ceph
RGW >= 19.2.7, MinIO. `probe` checks this on the real store at start. A store
that answers a conditional PUT with HTTP 501 has no conditional writes:
ConfigError INSTANCE_CONFIG (FIX_CONFIG), never a retry.

Keys are the store's keys (store.py) under the store prefix. Bodies are
streamed with at most _PART_BYTES x _PARTS_IN_FLIGHT bytes per transfer in
memory. ETags of `get` and of conditional writes stay as the server sent them
(If-Match sends them back exactly); `upload` and `list_objects` give them
without their quotes, the form a PUT answer and a LIST entry share.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard, cast, final

import aiohttp
from aiobotocore.config import AioConfig
from aiobotocore.httpsession import AIOHTTPSession
from botocore.exceptions import BotoCoreError, ClientError

from exec_sandbox._imports import aws_session
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.files.disk import pwrite_all
from exec_sandbox.instance.store import Applied, InstanceStore, ListedObject, NotApplied, wrong_size

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Sequence
    from pathlib import Path

    from aiobotocore.response import StreamingBody
    from aiobotocore.session import AioSession, ClientCreatorContext
    from types_aiobotocore_s3 import S3Client
    from types_aiobotocore_s3.type_defs import CompletedPartTypeDef, DeleteTypeDef, ListObjectsV2OutputTypeDef

    from exec_sandbox.instance.store import CondResult

_DELETE_BATCH = 1000  # DeleteObjects limit
_ACCESS_DENIED = 403
_SERVER_ERROR = 500
_SAFE_CODE = re.compile(r"[A-Za-z0-9]{1,64}")
_REFUSED = frozenset({404, 409, 412})
"""Statuses of a conditional PUT that the store refused (the head is unchanged by it)."""
_NOT_FOUND = 404
_NOT_IMPLEMENTED = 501
_ECHO_CHARS = 64
"""At most this much of a server header goes into an error message."""
_MAX_ETAG_CHARS = 1024
"""Longest ETag kept in a chain entry (S3 ETags are 34-40 characters)."""
_ARCHIVED = "InvalidObjectState"
"""GET of an object in an archive storage class (Glacier Flexible Retrieval, Deep Archive,
Intelligent-Tiering archive tiers): S3 answers 403 with this code until it is restored."""
_CONFIG_CODES = frozenset(
    {"AccessDenied", "NoSuchBucket", "PermanentRedirect", "AuthorizationHeaderMalformed", "InvalidBucketName"}
)
_TRANSPORT_ERRORS = (BotoCoreError, aiohttp.ClientError, TimeoutError)
"""No answer, or the connection broke mid-body (aiobotocore lets aiohttp payload errors through)."""
_ERRORS = (ClientError, *_TRANSPORT_ERRORS)
_PART_BYTES = 8 * 1024 * 1024
"""Multipart part and ranged-GET size (S3: parts >= 5 MiB except the last, <= 10,000 parts)."""
_PARTS_IN_FLIGHT = 4
_READ_CHUNK = 1024 * 1024


class _SendOnceSession(AIOHTTPSession):
    """HTTP session of the head client: every request goes out once.

    aiohttp sends an idempotent request (PUT included) again after a dropped
    connection. A conditional head write must not: a re-send that meets the
    write's own effect gets 412 and reads as "refused" although the first send
    landed. The commit protocol reads the head back and decides instead.
    """

    async def _get_session(self, proxy_url: str | None) -> aiohttp.ClientSession:  # pyright: ignore[reportIncompatibleMethodOverride]
        session = await super()._get_session(proxy_url)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        # No public switch: aiohttp's own test client sets this flag the same way.
        session._retry_connection = False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        return session  # pyright: ignore[reportUnknownVariableType]


class _VanishedError(Exception):
    """The object is missing, or was deleted during the download (the store answered 404)."""


@final
class S3InstanceStore(InstanceStore):
    """One bucket and prefix. Open with `S3InstanceStore.open`."""

    def __init__(self, *, client: S3Client, head_client: S3Client, bucket: str, prefix: str) -> None:
        self._client = client
        self._head_client = head_client
        self._bucket = bucket
        self._prefix = prefix

    @classmethod
    @contextlib.asynccontextmanager
    async def open(
        cls, *, bucket: str, prefix: str, endpoint_url: str | None, region: str | None
    ) -> AsyncGenerator[S3InstanceStore]:
        """Two clients on one session: SDK retries on for reads and bodies, off for
        conditional head writes. A head write goes out once (no SDK retry, no
        aiohttp re-send): a re-send that meets the write's own effect reads as
        refused. The commit protocol reads the head back instead. No request is
        sent until the first call. Each open has its own session (credentials and
        settings are read again) on the API models of the process: an open costs a
        few ms."""
        session = aws_session()
        async with (
            _s3_client(session, endpoint_url=endpoint_url, region=region, config=None) as client,
            _s3_client(
                session,
                endpoint_url=endpoint_url,
                region=region,
                # One attempt: botocore's max_attempts counts retries, total_max_attempts counts sends.
                config=AioConfig(
                    retries={"total_max_attempts": 1, "mode": "standard"}, http_session_cls=_SendOnceSession
                ),
            ) as head_client,
        ):
            yield cls(client=client, head_client=head_client, bucket=bucket, prefix=prefix)

    def _full(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str, /, *, max_bytes: int) -> tuple[bytes, str] | None:
        try:
            response = await self._client.get_object(Bucket=self._bucket, Key=self._full(key))
            async with response["Body"]:
                size, etag = (
                    cast("int | None", response.get("ContentLength")),
                    response.get("ETag"),
                )  # hostile stores omit fields
                if size is None:
                    raise _malformed("GET", key, what="answer without its size")
                etag = _etag_of(etag, "GET", key)
                if size > max_bytes:
                    raise wrong_size(key, size, expected=f"at most {max_bytes}")
                body = await response["Body"].read()
        except _ERRORS as e:
            if _is_missing(e):
                return None
            raise _failure("GET", key, e) from e
        return body, etag

    async def size(self, key: str, /) -> int | None:
        """One HEAD."""
        try:
            meta = await self._client.head_object(Bucket=self._bucket, Key=self._full(key))
        except _ERRORS as e:
            if _is_missing(e):
                return None
            raise _failure("HEAD", key, e) from e
        size = cast("int | None", meta.get("ContentLength"))
        if size is None:
            raise _malformed("HEAD", key, what="answer without its size")
        return size

    async def upload(self, key: str, /, *, src: Path) -> str:
        """One PUT up to _PART_BYTES, else multipart."""
        # Opened here, not in a thread: a cancelled to_thread(os.open) leaks the fd.
        jobs = _FileJobs(os.open(src, os.O_RDONLY))
        try:
            size = jobs.size()
            if size <= _PART_BYTES:
                data = await jobs.pread(size, 0)
                try:
                    response = await self._client.put_object(Bucket=self._bucket, Key=self._full(key), Body=data)
                except _ERRORS as e:
                    raise _failure("PUT", key, e) from e
                return _unquoted(_etag_of(response.get("ETag"), "PUT", key))
            return await self._upload_parts(key, jobs, size)
        finally:
            await jobs.close()

    async def _upload_parts(self, key: str, jobs: _FileJobs, size: int) -> str:
        full = self._full(key)
        try:
            created = await self._client.create_multipart_upload(Bucket=self._bucket, Key=full)
        except _ERRORS as e:
            raise _failure("PUT", key, e) from e
        upload_id = created["UploadId"]
        gate = asyncio.Semaphore(_PARTS_IN_FLIGHT)

        async def send(number: int, offset: int) -> CompletedPartTypeDef:
            async with gate:
                data = await jobs.pread(min(_PART_BYTES, size - offset), offset)
                response = await self._client.upload_part(
                    Bucket=self._bucket, Key=full, UploadId=upload_id, PartNumber=number, Body=data
                )
            return {"PartNumber": number, "ETag": _etag_of(response.get("ETag"), "PUT", key)}

        try:
            parts = await _gather_or_cancel(
                [send(number, offset) for number, offset in enumerate(range(0, size, _PART_BYTES), start=1)]
            )
            response = await self._client.complete_multipart_upload(
                Bucket=self._bucket, Key=full, UploadId=upload_id, MultipartUpload={"Parts": parts}
            )
        except BaseException as e:
            # Best effort; the bucket's AbortIncompleteMultipartUpload rule removes what stays.
            with contextlib.suppress(*_ERRORS):
                await self._client.abort_multipart_upload(Bucket=self._bucket, Key=full, UploadId=upload_id)
            if isinstance(e, _ERRORS):
                raise _failure("PUT", key, e) from e
            raise
        return _unquoted(_etag_of(response.get("ETag"), "PUT", key))

    async def download(self, key: str, /, *, dest: Path, max_bytes: int, expected_size: int | None) -> bool:
        """An object of at most one part whose size is known (`expected_size`: a body
        object): one plain GET, which reads one version. Else a HEAD, then parallel
        ranged GETs, each pinned to the ETag the HEAD saw: a changed ETag (412) is
        INSTANCE_STORE_UNAVAILABLE (body objects are never replaced). A body longer
        or shorter than its answer says, or another range, is
        INSTANCE_STORE_UNAVAILABLE. A size other than `expected_size` is asked again
        (HEAD) before any read: the same size twice is INSTANCE_INTEGRITY (a
        changed object)."""
        # Opened here, not in a thread: a cancelled to_thread(os.open) leaks the fd and the file.
        jobs = _FileJobs(os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        try:
            try:
                if expected_size is not None and expected_size <= _PART_BYTES:
                    await self._get_whole(key, jobs, max_bytes=max_bytes, expected_size=expected_size)
                else:
                    await self._get_ranges(key, jobs, max_bytes=max_bytes, expected_size=expected_size)
            finally:
                await jobs.close()
        except BaseException as e:
            dest.unlink(missing_ok=True)
            if isinstance(e, _VanishedError):
                return False
            raise
        return True

    async def _get_whole(self, key: str, jobs: _FileJobs, *, max_bytes: int, expected_size: int) -> None:
        """One GET without Range or ETag pin: no HEAD round trip."""
        try:
            response = await self._client.get_object(Bucket=self._bucket, Key=self._full(key))
            size = cast("int | None", response.get("ContentLength"))
            body = response["Body"]
            async with body:  # releases the connection; read on the StreamingBody (sized reads)
                if size is None:
                    raise _malformed("GET", key, what="answer without its size")
                if size == expected_size:
                    if size > max_bytes:
                        raise wrong_size(key, size, expected=f"at most {max_bytes}")
                    await _receive(body, jobs, key=key, start=0, end=size)
                    return
        except _ERRORS as e:
            if _is_missing(e):
                raise _VanishedError from e
            raise _failure("GET", key, e) from e
        await self._confirm_wrong_size(key, size=size, expected=expected_size)  # body closed, unread

    async def _get_ranges(self, key: str, jobs: _FileJobs, *, max_bytes: int, expected_size: int | None) -> None:
        full = self._full(key)
        try:
            meta = await self._client.head_object(Bucket=self._bucket, Key=full)
        except _ERRORS as e:
            if _is_missing(e):
                raise _VanishedError from e
            raise _failure("HEAD", key, e) from e
        size, etag = cast("int | None", meta.get("ContentLength")), meta.get("ETag")
        if size is None or not etag:
            raise _malformed("HEAD", key, what="answer without its size or ETag")
        if expected_size is not None and size != expected_size:
            await self._confirm_wrong_size(key, size=size, expected=expected_size)
        if size > max_bytes:
            raise wrong_size(key, size, expected=f"at most {max_bytes}")
        gate = asyncio.Semaphore(_PARTS_IN_FLIGHT)

        async def fetch(start: int, end: int) -> None:
            async with gate:
                try:
                    response = await self._client.get_object(
                        Bucket=self._bucket, Key=full, Range=f"bytes={start}-{end - 1}", IfMatch=etag
                    )
                    body = response["Body"]
                    async with body:  # releases the connection; read on the StreamingBody (sized reads)
                        _check_range(response.get("ContentRange"), start=start, end=end, size=size, key=key)
                        await _receive(body, jobs, key=key, start=start, end=end)
                except _ERRORS as e:
                    if _is_missing(e):
                        raise _VanishedError from e
                    raise _failure("GET", key, e) from e

        jobs.truncate(size)
        await _gather_or_cancel([fetch(start, min(start + _PART_BYTES, size)) for start in range(0, size, _PART_BYTES)])

    async def _confirm_wrong_size(self, key: str, *, size: int, expected: int) -> NoReturn:
        try:
            again = await self._client.head_object(Bucket=self._bucket, Key=self._full(key))
        except _ERRORS as e:
            raise _failure("HEAD", key, e) from e
        if cast("int | None", again.get("ContentLength")) == size:
            raise wrong_size(key, size, expected=str(expected))
        raise _malformed("HEAD", key, what=f"size {size} bytes, {expected} expected")

    async def put_if_absent(self, key: str, /, *, data: bytes) -> CondResult:
        return await self._conditional_put(key, data, IfNoneMatch="*")

    async def put_if_match(self, key: str, /, *, data: bytes, etag: str) -> CondResult:
        """`etag` is sent exactly as received; never ``*``."""
        return await self._conditional_put(key, data, IfMatch=etag)

    async def _conditional_put(self, key: str, data: bytes, **condition: str) -> CondResult:
        try:
            response = await self._head_client.put_object(
                Bucket=self._bucket, Key=self._full(key), Body=data, **cast("dict[str, Any]", condition)
            )
        except ClientError as e:
            if _status(e) == _NOT_IMPLEMENTED:
                # S3 answers 501 to a header it does not support: this store can never commit a head.
                raise ConfigError(
                    f"Instance store PUT {key}: the store does not support conditional writes (HTTP 501).",
                    code=ErrorCode.INSTANCE_CONFIG,
                    instance_id=None,
                    hint="Use a store whose PutObject honours If-None-Match and If-Match: "
                    "AWS S3, Cloudflare R2, Ceph RGW >= 19.2.7 or MinIO.",
                    context=None,
                ) from e
            failure = _failure("PUT", key, e)
            if isinstance(failure, ConfigError):
                raise failure from e  # refused before any condition: never applied
            return NotApplied(maybe_applied=_status(e) not in _REFUSED)
        except _TRANSPORT_ERRORS:
            return NotApplied(maybe_applied=True)  # no answer: the outcome is unknown
        etag = response.get("ETag")
        if _usable_etag(etag):
            return Applied(etag=etag)
        # Applied, but without a usable ETag (missing, empty or oversized) for the next save: read it back.
        return NotApplied(maybe_applied=True)

    def _relative(self, full_key: str, *, asked: str) -> str:
        """A listed key relative to the store prefix. A store that ignores `Prefix`
        would hand other keys to delete: refuse them."""
        if not full_key.startswith(self._full(asked)):
            raise _malformed("LIST", asked, what="a key outside the asked prefix")
        return full_key[len(self._prefix) :]

    async def list_objects(self, prefix: str, /) -> list[ListedObject]:
        """Needs s3:ListBucket."""
        return [
            ListedObject(
                key=self._relative(obj.get("Key", ""), asked=prefix),
                size=obj.get("Size", -1),
                etag=_unquoted(obj.get("ETag", "")),
            )
            async for page in self._pages(prefix)
            for obj in page.get("Contents", [])
        ]

    async def list_children(self, prefix: str, /) -> AsyncGenerator[str]:
        """S3 common prefixes. Needs s3:ListBucket."""
        async for page in self._pages(prefix, Delimiter="/"):
            for common in page.get("CommonPrefixes", []):
                name = self._relative(common.get("Prefix", ""), asked=prefix)[len(prefix) :].rstrip("/")
                if name:
                    yield name

    async def _pages(self, prefix: str, **extra: str) -> AsyncGenerator[ListObjectsV2OutputTypeDef]:
        """Every page of the listing of `prefix`, until the store says it is the last."""
        token: str | None = None
        while True:
            more = extra if token is None else {**extra, "ContinuationToken": token}
            try:
                page = await self._client.list_objects_v2(
                    Bucket=self._bucket, Prefix=self._full(prefix), **cast("dict[str, Any]", more)
                )
            except _ERRORS as e:
                raise _failure("LIST", prefix, e) from e
            yield page
            token = page.get("NextContinuationToken")
            if not page.get("IsTruncated") or not token:
                return

    async def delete_many(self, keys: Sequence[str], /) -> None:
        for start in range(0, len(keys), _DELETE_BATCH):
            delete: DeleteTypeDef = {
                "Objects": [{"Key": self._full(k)} for k in keys[start : start + _DELETE_BATCH]],
                "Quiet": True,
            }
            try:
                response = await self._client.delete_objects(Bucket=self._bucket, Delete=delete)
            except _ERRORS as e:
                raise _failure("DELETE", keys[start], e) from e
            # HTTP 200 can still list keys that were not deleted.
            if errors := response.get("Errors"):
                raise _failure_of("DELETE", keys[start], status=0, code=_safe_code(errors[0].get("Code", "")))


def _s3_client(
    session: AioSession, *, endpoint_url: str | None, region: str | None, config: AioConfig | None
) -> ClientCreatorContext[S3Client]:
    # Only the S3 stubs are installed: the other services' overloads are unknown to pyright.
    return session.create_client(  # pyright: ignore[reportUnknownMemberType]
        "s3", endpoint_url=endpoint_url, region_name=region, config=config
    )


async def _receive(body: StreamingBody, jobs: _FileJobs, *, key: str, start: int, end: int) -> None:
    """Write the answer's body at its place in the file: exactly bytes start to end - 1."""
    offset = start
    while chunk := await body.read(_READ_CHUNK):
        if offset + len(chunk) > end:
            raise _malformed("GET", key, what=f"body longer than range {start}-{end - 1}")
        await jobs.pwrite(chunk, offset)
        offset += len(chunk)
    if offset != end:
        raise _malformed("GET", key, what=f"body shorter than range {start}-{end - 1}")


async def _gather_or_cancel[T](coros: Sequence[Coroutine[Any, Any, T]]) -> list[T]:
    """Run all; on the first failure cancel the rest and raise that failure unwrapped."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class _FileJobs:
    """Blocking reads and writes on one fd, in worker threads.

    The fd closes only once every job has finished (a job a thread has started
    cannot be cancelled): a closed fd number that the process reuses must never
    get another file's bytes, or have them uploaded.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._pool = ThreadPoolExecutor(max_workers=_PARTS_IN_FLIGHT, thread_name_prefix="instance-store")

    def size(self) -> int:
        return os.fstat(self._fd).st_size

    def truncate(self, size: int) -> None:
        os.ftruncate(self._fd, size)

    async def pread(self, length: int, offset: int) -> bytes:
        return await asyncio.get_running_loop().run_in_executor(self._pool, _read_exactly, self._fd, length, offset)

    async def pwrite(self, data: bytes, offset: int) -> None:
        await asyncio.get_running_loop().run_in_executor(self._pool, pwrite_all, self._fd, data, offset)

    async def close(self) -> None:
        """Wait for running jobs (queued ones are dropped), then close the fd, even if cancelled."""
        loop = asyncio.get_running_loop()
        done = loop.run_in_executor(None, functools.partial(self._pool.shutdown, wait=True, cancel_futures=True))
        done.add_done_callback(lambda _: os.close(self._fd))
        await asyncio.shield(done)


def _read_exactly(fd: int, length: int, offset: int) -> bytes:
    data = os.pread(fd, length, offset)
    if len(data) != length:
        raise OSError(f"file changed during upload: read {len(data)} of {length} bytes at {offset}")
    return data


def _status(error: ClientError) -> int:
    return int(error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))


def _code(error: ClientError) -> str:
    return _safe_code(error.response.get("Error", {}).get("Code", ""))


def _safe_code(raw: object) -> str:
    """The server's error code, if it looks like one (it goes into one-line messages)."""
    return raw if isinstance(raw, str) and _SAFE_CODE.fullmatch(raw) else "unrecognized error code"


def _is_missing(error: Exception) -> bool:
    """404 for the key itself; a missing bucket is a configuration error."""
    return isinstance(error, ClientError) and _status(error) == _NOT_FOUND and _code(error) != "NoSuchBucket"


def _failure(operation: str, key: str, error: Exception) -> ConfigError | TransientError:
    if isinstance(error, ClientError):
        return _failure_of(operation, key, status=_status(error), code=_code(error))
    return _failure_of(operation, key, status=0, code=type(error).__name__)


def _failure_of(operation: str, key: str, *, status: int, code: str) -> ConfigError | TransientError:
    """Refusals the same call can never pass are configuration errors; the rest can succeed later.
    A 5xx is never a configuration error here, whatever code comes with it (a 501 on a
    conditional PUT is: _conditional_put checks it first)."""
    if status < _SERVER_ERROR and code == _ARCHIVED:
        return ConfigError(
            f"Instance store {operation} {key} failed: {code}: the object is in an archive storage class "
            "(Glacier Flexible Retrieval, Glacier Deep Archive or an Intelligent-Tiering archive tier), "
            "which a load cannot read.",
            code=ErrorCode.INSTANCE_CONFIG,
            instance_id=None,
            hint="Lifecycle rules must not move objects under the instance prefix to an archive storage class: "
            "remove that transition, then restore the object (S3 RestoreObject) or remove the instance.",
            context=None,
        )
    if status < _SERVER_ERROR and (status == _ACCESS_DENIED or code in _CONFIG_CODES):
        return ConfigError(
            f"Instance store {operation} {key} failed: {code}.",
            code=ErrorCode.INSTANCE_CONFIG,
            instance_id=None,
            hint="Check the store bucket, region and endpoint, and give this worker s3:GetObject, s3:PutObject, "
            "s3:DeleteObject and s3:ListBucket on the store prefix (without ListBucket, S3 answers a missing "
            "key with 403).",
            context=None,
        )
    return TransientError(
        f"Instance store {operation} {key} failed: {code}.",
        code=ErrorCode.INSTANCE_STORE_UNAVAILABLE,
        instance_id=None,
        hint="The store failed or did not answer. The same call can succeed later.",
        context=None,
    )


def _malformed(operation: str, key: str, *, what: str) -> TransientError:
    return TransientError(
        f"Instance store {operation} {key}: {what}.",
        code=ErrorCode.INSTANCE_STORE_UNAVAILABLE,
        instance_id=None,
        hint="The store answered badly; the same call can succeed later.",
        context=None,
    )


def _check_range(content_range: str | None, *, start: int, end: int, size: int, key: str) -> None:
    """The part is the asked range. A store may leave out Content-Range only when
    the range is the whole object (it may answer 200 with the full body)."""
    if content_range == f"bytes {start}-{end - 1}/{size}" or (content_range is None and (start, end) == (0, size)):
        return
    shown = None if content_range is None else content_range[:_ECHO_CHARS]
    raise _malformed("GET", key, what=f"answer for range {shown!r}, asked bytes {start}-{end - 1}/{size}")


def _unquoted(etag: str) -> str:
    """An ETag without its quotes: PUT answers and LIST entries may differ only there."""
    return etag.strip('"')


def _etag_of(etag: str | None, operation: str, key: str) -> str:
    """The ETag of a GET, PUT or part upload answer; an unusable one is INSTANCE_STORE_UNAVAILABLE."""
    if _usable_etag(etag):
        return etag
    raise _malformed(operation, key, what="answer without a usable ETag")


def _usable_etag(etag: str | None) -> TypeGuard[str]:
    """Present, not empty without its quotes, bounded."""
    return etag is not None and bool(_unquoted(etag)) and len(etag) <= _MAX_ETAG_CHARS

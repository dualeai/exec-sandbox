"""S3 object transfers in parts, shared by the instance store (instance.s3_store) and the
L3 layer cache (disk_snapshot_manager).

No botocore import (the s3 extra): the callers give their client, and they map its errors.
A multipart upload that fails or is cancelled is aborted, so no incomplete upload stays
in the bucket (best effort: an AbortIncompleteMultipartUpload rule of the bucket removes
what an abort that fails leaves; the README sets one on each prefix).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from exec_sandbox.aio_utils import await_cancellation_safe, gather_or_cancel
from exec_sandbox.files.disk import pwrite_all

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

PART_BYTES = 8 * 1024 * 1024
"""Size of one part (the last can be shorter); an object up to this size is one PUT."""

PARTS_IN_FLIGHT = 4
"""Parts sent or received at once: at most PART_BYTES x PARTS_IN_FLIGHT bytes in memory per transfer."""

ABORT_SECONDS = 5.0
"""Longest wait for the abort of a multipart upload that failed or was cancelled: one
request, tens of ms on a store that answers. A store that does not answer (botocore
retries: minutes) then holds the error or the cancel of the caller no longer than this."""

logger = logging.getLogger(__name__)


class FileJobs:
    """Blocking reads and writes on one fd, in worker threads.

    The fd closes only once every job has finished (a job a thread has started
    cannot be cancelled): a closed fd number that the process reuses must never
    get another file's bytes, or have them uploaded.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._pool = ThreadPoolExecutor(max_workers=PARTS_IN_FLIGHT, thread_name_prefix="s3-parts")

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


async def upload_object(
    client: Any,
    *,
    bucket: str,
    key: str,
    src: Path,
    extra: Mapping[str, Any],
    part_etag: Callable[[str | None], str],
    abort_errors: tuple[type[BaseException], ...],
) -> Mapping[str, Any]:
    """Upload the file `src` as the object `key`: one PUT up to PART_BYTES, else a multipart
    upload, PARTS_IN_FLIGHT parts at a time. Returns the answer of the PUT or of the
    completion (its ETag).

    `extra`: more arguments of PutObject and CreateMultipartUpload (Tagging, Metadata).
    `part_etag`: the ETag of a part answer, or the caller's error for an unusable one.
    A multipart upload that fails or is cancelled is aborted before the error goes on:
    within ABORT_SECONDS, and to its end also on a second cancel. An abort that fails with
    an error in `abort_errors`, or takes longer, is logged, not raised. The client's errors
    are raised as they are.
    """
    # Opened here, not in a thread: a cancelled to_thread(os.open) leaks the fd.
    jobs = FileJobs(os.open(src, os.O_RDONLY))
    try:
        size = jobs.size()
        if size <= PART_BYTES:
            data = await jobs.pread(size, 0)
            return await client.put_object(Bucket=bucket, Key=key, Body=data, **extra)
        return await _upload_parts(
            client,
            bucket=bucket,
            key=key,
            jobs=jobs,
            size=size,
            extra=extra,
            part_etag=part_etag,
            abort_errors=abort_errors,
        )
    finally:
        await jobs.close()


async def _upload_parts(
    client: Any,
    *,
    bucket: str,
    key: str,
    jobs: FileJobs,
    size: int,
    extra: Mapping[str, Any],
    part_etag: Callable[[str | None], str],
    abort_errors: tuple[type[BaseException], ...],
) -> Mapping[str, Any]:
    created = await client.create_multipart_upload(Bucket=bucket, Key=key, **extra)
    upload_id = created["UploadId"]
    gate = asyncio.Semaphore(PARTS_IN_FLIGHT)

    async def send(number: int, offset: int) -> dict[str, Any]:
        async with gate:
            data = await jobs.pread(min(PART_BYTES, size - offset), offset)
            response = await client.upload_part(
                Bucket=bucket, Key=key, UploadId=upload_id, PartNumber=number, Body=data
            )
        return {"PartNumber": number, "ETag": part_etag(response.get("ETag"))}

    try:
        parts = await gather_or_cancel(
            [send(number, offset) for number, offset in enumerate(range(0, size, PART_BYTES), start=1)]
        )
        return await client.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts}
        )
    except BaseException:
        # Also on a cancel (a CancelledError is no Exception): no incomplete upload stays
        await _abort(client, bucket=bucket, key=key, upload_id=upload_id, abort_errors=abort_errors)
        raise


async def _abort(
    client: Any, *, bucket: str, key: str, upload_id: str, abort_errors: tuple[type[BaseException], ...]
) -> None:
    """Abort the multipart upload `upload_id` within ABORT_SECONDS. A cancel of the caller
    waits for the abort, then goes on: a second cancel (a bound on the caller's exit) does
    not leave the parts in the bucket."""
    abort = client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
    try:
        await await_cancellation_safe(asyncio.wait_for(abort, ABORT_SECONDS))
    except (TimeoutError, *abort_errors) as e:
        logger.warning(
            "S3 multipart upload not aborted: its parts stay until a bucket rule removes them",
            extra={"key": key, "error": repr(e)},
        )

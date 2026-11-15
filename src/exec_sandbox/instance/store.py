"""Instance store contract: what saved instances need from an object store.

Each store subclasses `InstanceStore` and has its own `open(*, ...)` async
context manager; the manager opens the one `instance_store` names:
    "s3"     S3InstanceStore (s3_store.py, the s3 extra): an S3-API bucket
    "local"  LocalInstanceStore (local_store.py): a folder on this host

Two kinds of objects, two APIs:
    head    a few KB, written only by conditional writes, read whole (`get`, capped)
    bodies  one object per save (full copy or pack, both streams) and guest
            images, 100s of MB to GBs: streamed between a local file and the
            store (`upload`, `download`) in bounded memory

Keys (relative, "/"-separated, made only by the helpers below):
    s/<instance_id>/head                 the head object (only conditional writes)
    s/<instance_id>/b/<commit_id>.<f|p>  body objects (full copy / pack), unique per save attempt
    assets/sha256/<digest>               guest images (base, L2 packages) by SHA-256, shared by all instances
    probe/<random>/...                   startup probe scratch

ETags are opaque strings of 1 to 1,024 characters, in two groups that are
never compared with each other: the ETags of `get` and of the conditional
writes (the If-Match condition), and the ETags of `upload` and `list_objects`
(the lineage check before each pack). Other bytes at a key always give
another ETag; the same bytes may give the same one.

Errors. Stores raise with instance_id=None (the manager names the instance):
    - a missing object gives None or False, only when the store says "no such key"
    - a refusal that the same call can never pass: ConfigError INSTANCE_CONFIG (FIX_CONFIG),
      for example access denied, a missing bucket, or no conditional writes (S3 HTTP 501)
    - any other store failure: TransientError INSTANCE_STORE_UNAVAILABLE (RETRY), never OSError
    - an object above `max_bytes`: InstanceGoneError INSTANCE_INTEGRITY, before any read
    - a local file failure (`src`, `dest`): OSError (a save or a load reports it as
      TransientError LOCAL_DISK)
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InstanceGoneError,
    TransientError,
    damaged_instance_data,
)
from exec_sandbox.instance.ids import new_id

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

INSTANCES_PREFIX = "s/"
"""Parent of every instance's keys: `list_children` of it gives the instance ids."""
_PROBE_MAX_BYTES = 64
_PROBE_BODY = b"probe body"


@dataclass(frozen=True, slots=True)
class Applied:
    """A conditional write took effect."""

    etag: str


@dataclass(frozen=True, slots=True, kw_only=True)
class NotApplied:
    """A conditional write did not take effect, or its outcome is unknown.

    maybe_applied False: the store proved that this attempt did not apply (the
    head is unchanged by it). True: no answer, a failure, or an answer without
    the ETag the next save needs; the write may have applied, so read the head back.
    """

    maybe_applied: bool


type CondResult = Applied | NotApplied


@dataclass(frozen=True, slots=True, kw_only=True)
class ListedObject:
    """One object of a listing: key relative to the store, size, and the ETag its upload returned."""

    key: str
    size: int
    etag: str


def wrong_size(key: str, size: int, *, expected: str) -> InstanceGoneError:
    """The error every store raises, before any read, for an object above `max_bytes`
    (`expected` "at most N") or of another size than `expected_size` ("N")."""
    return damaged_instance_data(f"Instance store object {key} is {size} bytes; {expected} expected.", instance_id=None)


def instance_prefix(instance_id: str) -> str:
    return f"{INSTANCES_PREFIX}{instance_id}/"


def body_prefix(instance_id: str) -> str:
    return f"{instance_prefix(instance_id)}b/"


def head_key(instance_id: str) -> str:
    return f"{instance_prefix(instance_id)}head"


def body_key(instance_id: str, commit_id: str, kind: str) -> str:
    return f"{body_prefix(instance_id)}{commit_id}.{kind}"


def asset_key(digest: str) -> str:
    return f"assets/sha256/{digest}"


class InstanceStore(ABC):
    """Objects under keys, with conditional writes. Open a store with its own `open`.

    The commit protocol (commit.py) needs all six conditional-write rules:
    1. Compare and set is atomic per key, across all processes and hosts.
    2. NotApplied(maybe_applied=False) only when the store proved that this
       attempt did not apply; else maybe_applied=True.
    3. One attempt per call, no re-send inside the store: a re-send that meets
       the write's own effect is refused, and reads as "not applied".
    4. `put_if_match` on a missing key: NotApplied(maybe_applied=False).
    5. Read after write: after Applied, `get` returns those bytes and that
       ETag; after NotApplied, `get` returns the current head.
    6. A cancelled write applies whole or not at all.
    A conditional write raises only when this attempt did not apply:
    ConfigError INSTANCE_CONFIG (refused before any condition, for example S3 HTTP 501
    for a store without conditional writes) or TransientError INSTANCE_STORE_UNAVAILABLE
    (failed before any change).

    Listing: every object under the prefix (all pages), keys relative to the
    store, exact sizes, the ETag the upload returned; never a key outside the
    prefix (delete_instance deletes what it lists), never an object deleted
    before the call began. A prefix is "" or segments that end with "/".

    Streams: an object appears whole or not at all; a failed upload leaves the
    key unchanged; memory per transfer is bounded. Read after write: once
    `upload` returns, `exists`, `size`, `download` and listings see the object
    (after its head lands, a first save checks with one `exists` that its body
    is still there; delete_instance checks the head again after it removes the
    objects).
    """

    @property
    @abstractmethod
    def location(self) -> str:
        """Where the store is, for messages: s3://bucket/prefix, or the folder."""

    @abstractmethod
    async def get(self, key: str, /, *, max_bytes: int) -> tuple[bytes, str] | None:
        """Bytes and ETag of a small object, or None when it does not exist.

        An object above `max_bytes` is not read: InstanceGoneError INSTANCE_INTEGRITY.
        """

    @abstractmethod
    async def size(self, key: str, /) -> int | None:
        """Bytes of the object at `key`, or None when it does not exist."""

    async def exists(self, key: str, /) -> bool:
        """True when an object is at `key` (one `size` call)."""
        return await self.size(key) is not None

    @abstractmethod
    async def upload(self, key: str, /, *, src: Path) -> str:
        """Stream the local file `src` to `key` (unconditional: body keys are unique
        per attempt, asset keys name their content) and return the ETag a listing
        reports for it."""

    @abstractmethod
    async def download(self, key: str, /, *, dest: Path, max_bytes: int, expected_size: int | None) -> bool:
        """Stream `key` into the new file `dest` (O_EXCL, mode 0600). True when done.

        False, and no file, only when the object does not exist or was deleted
        during the transfer. The transfer reads one version: a change during it
        is TransientError INSTANCE_STORE_UNAVAILABLE (body objects are never replaced),
        never proof that the object is gone. An object above `max_bytes` is not read:
        InstanceGoneError INSTANCE_INTEGRITY. `expected_size` (from authenticated data,
        None when unknown): another size is INSTANCE_INTEGRITY only when the store
        confirms it, else INSTANCE_STORE_UNAVAILABLE. On any error no file stays.
        """

    @abstractmethod
    async def put_if_absent(self, key: str, /, *, data: bytes) -> CondResult:
        """Create `key` only if it does not exist."""

    @abstractmethod
    async def put_if_match(self, key: str, /, *, data: bytes, etag: str) -> CondResult:
        """Replace only the version with this ETag (from `get` or Applied, as received)."""

    @abstractmethod
    async def list_objects(self, prefix: str, /) -> list[ListedObject]:
        """Objects under `prefix` with their size and ETag."""

    @abstractmethod
    def list_children(self, prefix: str, /) -> AsyncIterator[str]:
        """Names of the "folders" directly under `prefix`, each once, in any order."""

    @abstractmethod
    async def delete_many(self, keys: Sequence[str], /) -> None:
        """Delete objects; missing ones are not an error (idempotent). Raises if any key stays."""


async def probe(store: InstanceStore, *, scratch: Path) -> None:
    """Check, at each Scheduler start, that the store keeps the conditional-write
    and listing rules (it cannot test atomicity under races). `scratch`: a
    private folder for one small local file.

    Raises ConfigError (INSTANCE_CONFIG) naming the first step the store got wrong,
    or the store's refusal (access denied, missing bucket, no conditional writes)
    as the store reported it; TransientError (INSTANCE_STORE_UNAVAILABLE) when a
    step got no usable answer.
    """
    root = f"probe/{new_id()}/"
    k1, k2, k3 = f"{root}k1", f"{root}k2", f"{root}k3"
    src = scratch / f"probe-{new_id()}"
    try:
        _applied(await store.put_if_absent(k1, data=b"first"), "create-only write of a new key")
        _refused(await store.put_if_absent(k1, data=b"second"), "create-only write of an existing key")
        got = await store.get(k1, max_bytes=_PROBE_MAX_BYTES)
        if got is None or got[0] != b"first":
            _fail("read after a refused create-only write")
        etag = got[1]
        changed = _applied(
            await store.put_if_match(k1, data=b"third", etag=etag), "If-Match write with the current ETag"
        )
        if changed.etag == etag:
            _fail("ETag change after an If-Match write")
        if await store.get(k1, max_bytes=_PROBE_MAX_BYTES) != (b"third", changed.etag):
            _fail("read after an If-Match write")
        _refused(await store.put_if_match(k1, data=b"fourth", etag=etag), "If-Match write with a stale ETag")
        _refused(await store.put_if_match(k2, data=b"x", etag=etag), "If-Match write of a missing key")
        await asyncio.to_thread(src.write_bytes, _PROBE_BODY)
        uploaded = await store.upload(k3, src=src)
        if ListedObject(key=k3, size=len(_PROBE_BODY), etag=uploaded) not in await store.list_objects(root):
            # The lineage check before each pack compares them: else every save is a full copy.
            raise ConfigError(
                "Instance store listing does not report the size and ETag of an upload: "
                "every save would be a full copy.",
                code=ErrorCode.INSTANCE_CONFIG,
                instance_id=None,
                hint="Use AWS S3, Cloudflare R2, Ceph RGW >= 19.2.7 or MinIO for the S3 store.",
                context=None,
            )
    finally:
        src.unlink(missing_ok=True)
        with contextlib.suppress(TransientError, ConfigError):
            await store.delete_many([k1, k2, k3])


def _applied(result: CondResult, step: str) -> Applied:
    if isinstance(result, Applied):
        return result
    if not result.maybe_applied:
        _fail(step)
    raise _probe_unanswered(step)


def _refused(result: CondResult, step: str) -> None:
    if isinstance(result, Applied):
        _fail(step)
    if result.maybe_applied:
        raise _probe_unanswered(step)


def _probe_unanswered(step: str) -> TransientError:
    return TransientError(
        f"Instance store probe: {step} got no usable answer.",
        code=ErrorCode.INSTANCE_STORE_UNAVAILABLE,
        instance_id=None,
        hint="The store failed or did not answer; start the worker again later.",
        context=None,
    )


def _fail(step: str) -> NoReturn:
    raise ConfigError(
        f"Instance store does not support conditional writes: {step} failed.",
        code=ErrorCode.INSTANCE_CONFIG,
        instance_id=None,
        hint="Use a store that honours create-only and If-Match writes: for the S3 store, "
        "AWS S3, Cloudflare R2, Ceph RGW >= 19.2.7 or MinIO.",
        context=None,
    )

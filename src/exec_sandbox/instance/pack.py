"""Body objects: the blocks of one save, sealed, with an authenticated index.

A saved instance has two streams, each a local file: the migration file (RAM
at fixed offsets, then device state) and the home disk. A save writes ONE body
object: either a full copy (every non-zero block of each stream) or a pack
(only the blocks that differ from the baseline, the state the session loaded
or last saved). A load applies the chain in order (full copy, then packs) to
empty files, and sets each stream to its recorded size after each object.

object = record* | sealed index | trailer
record = codec.seal_block(block), AAD binds instance, commit, stream, offset
index  = nonce | AES-GCM(data key, JSON {stream: [[block_no, record_len], ...]}, AAD = exs1|i|ids)
trailer = index offset (u64) | index length (u32) | PACK_MAGIC (8)

The index is authenticated, so a record cannot be dropped, moved or added
without the load failing; record order in the object is index order.
Integers big endian.

Codec work runs on one pool of threads per process (zstd and AES-GCM release
the GIL), in tasks of _TASK_BLOCKS blocks. write_object: the calling thread
writes the records in index order, so the layout does not depend on the number
of threads. apply_object: each task writes its own blocks (each block has its
own offset).
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import functools
import itertools
import os
from typing import TYPE_CHECKING

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import NonNegativeInt, PositiveInt, TypeAdapter, ValidationError

from exec_sandbox.exceptions import damaged_instance_data
from exec_sandbox.files.disk import data_ranges, pwrite_all
from exec_sandbox.instance import codec
from exec_sandbox.instance.codec import BLOCK_BYTES, NONCE_BYTES, TAG_BYTES

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping
    from pathlib import Path
    from typing import BinaryIO

PACK_MAGIC = b"EXSIPACK"
_TRAILER = 8 + 4 + len(PACK_MAGIC)
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_ZERO_BLOCK = bytes(BLOCK_BYTES)
_INDEX = TypeAdapter(dict[NonNegativeInt, list[tuple[NonNegativeInt, PositiveInt]]], config={"strict": False})
"""Stream -> [(block number, record length)] in object order. JSON object keys are
strings, so the stream key is coerced; every value is checked."""
_TASK_BLOCKS = 32
"""Blocks per codec task (2 MiB): one read of their span per file."""
_WORKERS = min(4, os.cpu_count() or 1)
"""More threads are slower on CPython 3.12 (measured: 4 is the knee)."""
_IN_FLIGHT = 2 * _WORKERS
"""Tasks queued or running per call: bounds the memory of a write or an apply."""

type _Records = tuple[int, Callable[[int], bytes], list[tuple[int, int, int]]]
"""A task of apply_object: target fd, AAD of each offset, (block number, record offset, record length)."""


def write_object(
    dest: Path,
    *,
    data_key: bytes,
    instance_id: str,
    commit_id: str,
    streams: Mapping[int, tuple[Path, Path | None]],
) -> int:
    """Seal the blocks to store of each stream (new file, baseline or None) into the
    new file `dest` (0600) and return its size. On failure (e.g. ENOSPC) `dest` is removed.

    With a baseline (a pack): the blocks that differ from it. Without (a full
    copy): every block that is not all zeros. Only the 32-block windows that hold
    data (SEEK_DATA) are read, each as one span, holes inside it included; a block
    that holds data in either file is compared, so a block the guest discarded (a
    hole now) still counts. Blocks past the end of the new file are not stored:
    the recorded size cuts them. Each block is read once.
    """
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as out:
            index: dict[int, list[tuple[int, int]]] = {}
            for stream in sorted(streams):
                aads = codec.block_aads(instance_id=instance_id, commit_id=commit_id, stream=stream)
                index[stream] = _write_stream(out, *streams[stream], data_key=data_key, aads=aads)
            index_offset = out.tell()
            nonce = os.urandom(NONCE_BYTES)
            plain = _INDEX.dump_json(index)
            index_aad = codec.object_index_aad(instance_id=instance_id, commit_id=commit_id)
            sealed = nonce + AESGCM(data_key).encrypt(nonce, plain, index_aad)
            out.write(sealed)
            out.write(index_offset.to_bytes(8, "big") + len(sealed).to_bytes(4, "big") + PACK_MAGIC)
            return out.tell()
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def apply_object(
    src: Path,
    *,
    data_key: bytes,
    instance_id: str,
    commit_id: str,
    targets: Mapping[int, int],
    sizes: Mapping[int, int],
) -> None:
    """Authenticate the object, write its blocks into the stream files open for
    writing at the fds `targets`, then cut or extend each target to its size at
    this commit (`sizes`).

    Every record is checked before its bytes are written. A changed, missing or
    extra record raises InstanceGoneError (INSTANCE_INTEGRITY); an index or a block this release
    cannot read, or a stream it does not know, raises RouteError (INSTANCE_HOST_MISMATCH)
    (another release wrote it). After an error the targets hold partial data:
    the caller discards them. A shorter stream drops what older objects wrote
    past its end; a longer one reads as zeros there.
    """
    with src.open("rb") as f:
        size = f.seek(0, os.SEEK_END)
        if size < _TRAILER:
            raise damaged_instance_data(f"Instance {instance_id}: object is truncated.", instance_id=instance_id)
        f.seek(size - _TRAILER)
        trailer = f.read(_TRAILER)
        if trailer[12:] != PACK_MAGIC:
            raise damaged_instance_data(f"Instance {instance_id}: object has no pack magic.", instance_id=instance_id)
        index_offset = int.from_bytes(trailer[:8], "big")
        index_len = int.from_bytes(trailer[8:12], "big")
        if (
            index_len > _MAX_INDEX_BYTES
            or index_offset + index_len + _TRAILER != size
            or index_len < NONCE_BYTES + TAG_BYTES
        ):
            raise damaged_instance_data(
                f"Instance {instance_id}: object index is out of bounds.", instance_id=instance_id
            )
        f.seek(index_offset)
        sealed = f.read(index_len)
        index_aad = codec.object_index_aad(instance_id=instance_id, commit_id=commit_id)
        try:
            plain = AESGCM(data_key).decrypt(sealed[:NONCE_BYTES], sealed[NONCE_BYTES:], index_aad)
        except InvalidTag:
            raise damaged_instance_data(
                f"Instance {instance_id}: object index failed authentication.", instance_id=instance_id
            ) from None
        try:
            index = _INDEX.validate_json(plain)
        except ValidationError:
            raise codec.other_release(instance_id, what="object index", writer=None) from None
        if set(index) - set(targets):
            raise codec.other_release(instance_id, what="object's streams", writer=None)
        tasks: list[_Records] = []
        position = 0
        for stream in sorted(index):
            aads = codec.block_aads(instance_id=instance_id, commit_id=commit_id, stream=stream)
            records: list[tuple[int, int, int]] = []
            for block_no, record_len in index[stream]:
                records.append((block_no, position, record_len))
                position += record_len
                if position > index_offset:
                    raise damaged_instance_data(
                        f"Instance {instance_id}: object records overrun the index.", instance_id=instance_id
                    )
            tasks.extend(
                (targets[stream], aads, records[i : i + _TASK_BLOCKS]) for i in range(0, len(records), _TASK_BLOCKS)
            )
        if position != index_offset:
            raise damaged_instance_data(
                f"Instance {instance_id}: object has records the index does not list.", instance_id=instance_id
            )
        src_fd = f.fileno()

        def open_records(task: _Records) -> None:
            out_fd, aads, records = task
            first = records[0][1]
            data = os.pread(src_fd, records[-1][1] + records[-1][2] - first, first)
            cipher = AESGCM(data_key)
            for block_no, start, length in records:
                record = data[start - first : start - first + length]
                block = codec.open_block(cipher, record, aads(block_no * BLOCK_BYTES), instance_id=instance_id)
                pwrite_all(out_fd, block, block_no * BLOCK_BYTES)

        # The authenticated index comes from write_object, which lists each block once:
        # the tasks write to distinct offsets, so their order does not change the result.
        with contextlib.closing(_in_order(open_records, tasks)) as opened:
            for _ in opened:
                pass
    for stream, fd in targets.items():
        os.ftruncate(fd, sizes[stream])


def new_stream_file(path: Path) -> None:
    """A new empty stream file (0600): QEMU's migration target, or the start of a reconstruct."""
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))


def _write_stream(
    out: BinaryIO, new: Path, baseline: Path | None, *, data_key: bytes, aads: Callable[[int], bytes]
) -> list[tuple[int, int]]:
    """Seal the blocks of `new` to store (see write_object) into `out`, in block
    order; returns (block number, record length) of each. The last block of a file
    whose size is not a multiple of BLOCK_BYTES is sealed as it is, shorter: it is
    compared zero-padded."""
    entries: list[tuple[int, int]] = []
    with contextlib.ExitStack() as files:
        new_fd = files.enter_context(new.open("rb")).fileno()
        base_fd = None if baseline is None else files.enter_context(baseline.open("rb")).fileno()
        size = os.fstat(new_fd).st_size
        candidates = _data_blocks(new_fd)
        if base_fd is not None:
            candidates |= _data_blocks(base_fd)
        blocks = sorted(block_no for block_no in candidates if block_no * BLOCK_BYTES < size)
        windows = (list(window) for _, window in itertools.groupby(blocks, lambda block_no: block_no // _TASK_BLOCKS))

        def seal(window: list[int]) -> list[tuple[int, bytes]]:
            cipher = AESGCM(data_key)
            return [
                (block_no, codec.seal_block(cipher, block, aads(block_no * BLOCK_BYTES)))
                for block_no, block in _changed_blocks(new_fd, base_fd, window)
            ]

        # Closed before the files: no task reads them after this block.
        with contextlib.closing(_in_order(seal, windows)) as sealed:
            for records in sealed:
                for block_no, record in records:
                    out.write(record)
                    entries.append((block_no, len(record)))
    return entries


def _changed_blocks(new_fd: int, base_fd: int | None, window: list[int]) -> list[tuple[int, bytes]]:
    """(number, bytes) of each block of `window` (sorted, in one aligned run of
    _TASK_BLOCKS) that differs from the baseline, or from zeros without one. One
    read of the window's span per file: a span equal in both files (most of a
    pack) is compared at once."""
    first = window[0]
    span = (window[-1] + 1 - first) * BLOCK_BYTES
    new = os.pread(new_fd, span, first * BLOCK_BYTES)
    old = None if base_fd is None else os.pread(base_fd, span, first * BLOCK_BYTES).ljust(span, b"\0")
    if old is not None and new.ljust(span, b"\0") == old:
        return []
    changed: list[tuple[int, bytes]] = []
    for block_no in window:
        at = (block_no - first) * BLOCK_BYTES
        block = new[at : at + BLOCK_BYTES]
        if block.ljust(BLOCK_BYTES, b"\0") != (_ZERO_BLOCK if old is None else old[at : at + BLOCK_BYTES]):
            changed.append((block_no, block))
    return changed


@functools.cache
def _pool() -> concurrent.futures.ThreadPoolExecutor:
    """Codec threads of every save and load of this process; they start on first use.
    A task never submits to this pool, so it cannot wait on itself."""
    return concurrent.futures.ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="instance-codec")


# A forked child has none of the parent's threads: a pool copied from the parent never runs a task.
os.register_at_fork(after_in_child=_pool.cache_clear)


def _in_order[T, R](fn: Callable[[T], R], items: Iterable[T]) -> Generator[R]:
    """fn(item) of each item on the codec pool, results in item order, at most
    _IN_FLIGHT tasks at once. On an error, or when closed early, the tasks not
    started are cancelled and the running ones waited for."""
    pending: collections.deque[concurrent.futures.Future[R]] = collections.deque()
    try:
        for item in items:
            pending.append(_pool().submit(fn, item))
            if len(pending) >= _IN_FLIGHT:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()
        concurrent.futures.wait(pending)


def _data_blocks(fd: int) -> set[int]:
    """Numbers of the blocks that hold data (files.disk.data_ranges). When the search
    fails (an OSError: EINVAL without SEEK_DATA), every block after the ranges found so
    far counts as data: none is lost."""
    size = os.fstat(fd).st_size
    blocks: set[int] = set()
    offset = 0
    try:
        for start, end in data_ranges(fd, size, grain=BLOCK_BYTES):
            blocks.update(range(start // BLOCK_BYTES, (end - 1) // BLOCK_BYTES + 1))
            offset = end
    except OSError:
        blocks.update(range(offset // BLOCK_BYTES, (size - 1) // BLOCK_BYTES + 1))
    return blocks

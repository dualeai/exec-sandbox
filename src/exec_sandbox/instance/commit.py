"""Commit protocol: replace an instance's head only if nobody else did first.

A save writes its body objects under unique keys, then the head: create-only
for the first save, If-Match on the ETag it loaded otherwise. When the store
refuses or gives no answer, the saver reads the head back and authenticates it
first, with the key the read-back head names (another saver may have sealed it
with another data key): a store writer cannot list this commit in a forged head.
A head this keyring cannot open (forged, damaged, a key this keyring lacks,
another release) is taken as not this save's own head (only a store that damages
this save's own head after it landed breaks this): conflict when the store
refused every write and its ETag is not the base's, else outcome unknown. A head that
opens is judged on its content, also after a "refused" answer (a proxy that
re-sends a PUT that landed gets 412):
    the head is its own commit     -> committed
    its own commit is listed below -> committed, then superseded (returns None)
    head unchanged                 -> re-send the same bytes once, same condition
    no head yet (first save)       -> re-send the same bytes once, same condition
    another commit holds the slot  -> conflict when the store refused every write,
                                      or when the head still lists generations down
                                      to this one (nothing of this save is in the
                                      head); else outcome unknown
    no head, after a refused write -> gone (a later save: deleted during the save)
    otherwise                      -> outcome unknown
A write without an answer may have landed: after one, a missing head proves
nothing (unknown, never gone). A first save checks for a delete during the save
itself, after the commit (manager.py).
The re-send is the same bytes under the same condition: it cannot overwrite a
newer commit. Each head is sealed with a random nonce, so no two saves have the
same bytes or ETag (a re-send repeats its own bytes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InstanceGoneError,
    OutcomeUnknownError,
    RouteError,
    TransientError,
)
from exec_sandbox.instance.head import MAX_HEAD_BYTES, authenticate
from exec_sandbox.instance.store import Applied, head_key

if TYPE_CHECKING:
    from exec_sandbox.instance.keyring import Keyring
    from exec_sandbox.instance.models import HeadHeader
    from exec_sandbox.instance.store import CondResult, InstanceStore


@dataclass(frozen=True, slots=True, kw_only=True)
class Commit:
    """What one save writes as the head: its header, the sealed head bytes, and the
    keyring that authenticates a head read back from the store."""

    header: HeadHeader
    data: bytes
    keyring: Keyring


@dataclass(frozen=True, slots=True, kw_only=True)
class Base:
    """The head this save started from; None fields for the first save."""

    etag: str | None
    commit_id: str | None


async def commit_head(store: InstanceStore, commit: Commit, base: Base) -> str | None:
    """Write the head; return its new ETag, or raise OutcomeUnknownError
    (INSTANCE_CONFLICT or INSTANCE_SAVE_OUTCOME_UNKNOWN) or InstanceGoneError
    (INSTANCE_NOT_FOUND: a later save whose head was deleted).

    ConfigError or TransientError (INSTANCE_STORE_UNAVAILABLE): no write of this save applied
    (the store refused this worker before any condition, or failed before any change).
    None: committed, but another commit is already on top. The saver must not use
    the current ETag as its next base: that would overwrite the newer commit.
    """
    key = head_key(commit.header.instance_id)
    result = await _write(store, key, commit, base)
    if isinstance(result, Applied):
        return result.etag
    return await _resolve(store, key, commit, base, maybe_applied=result.maybe_applied)


async def _write(store: InstanceStore, key: str, commit: Commit, base: Base) -> CondResult:
    if base.etag is None:
        return await store.put_if_absent(key, data=commit.data)
    return await store.put_if_match(key, data=commit.data, etag=base.etag)


async def _resolve(store: InstanceStore, key: str, commit: Commit, base: Base, *, maybe_applied: bool) -> str | None:
    mine = commit.header
    repaired = False
    while True:
        try:
            current = await store.get(key, max_bytes=MAX_HEAD_BYTES)
        except (TransientError, ConfigError, InstanceGoneError) as e:
            raise _unknown(mine) from e  # cannot read back (or an oversized answer): no proof either way
        if current is None:
            _check_absent(mine, base, repaired=repaired, maybe_applied=maybe_applied)
        else:
            data, etag = current
            try:
                # Authenticated: else a store writer could list my commit in a forged head.
                header = authenticate(data, commit.keyring, instance_id=mine.instance_id)
            except (InstanceGoneError, RouteError, ConfigError) as e:
                # Forged, damaged, a key this worker lacks or another release: not this save's
                # head (its own always opens). Conflict when every write was refused, e.g. a
                # forced id another keyring holds; else outcome unknown.
                raise (_conflict(mine) if not maybe_applied and etag != base.etag else _unknown(mine)) from e
            # A head that opens is judged on its content, even after a "refused" answer: a
            # proxy that re-sends an applied PUT gets 412 for a write that landed.
            if header.commit_id == mine.commit_id:
                return etag
            if any(c.commit_id == mine.commit_id for c in header.recent_commits):
                return None  # committed, then superseded before the read-back
            if header.commit_id != base.commit_id:
                # Every write refused: nothing of this save landed. Else, newest first: the
                # last entry is the oldest generation the head still lists.
                if not maybe_applied or header.recent_commits[-1].generation <= mine.generation:
                    raise _conflict(mine)
                raise _unknown(mine)
            if repaired:
                raise _unknown(mine)
        repaired = True
        result = await _resend(store, key, commit, base, maybe_applied=maybe_applied)
        if isinstance(result, Applied):
            return result.etag
        maybe_applied = maybe_applied or result.maybe_applied


async def _resend(store: InstanceStore, key: str, commit: Commit, base: Base, *, maybe_applied: bool) -> CondResult:
    """The same write once more. A store error means this write did not apply; after
    an earlier write that may have applied, the outcome is then unknown."""
    try:
        return await _write(store, key, commit, base)
    except (TransientError, ConfigError) as e:
        if maybe_applied:
            raise _unknown(commit.header) from e
        raise


def _check_absent(mine: HeadHeader, base: Base, *, repaired: bool, maybe_applied: bool) -> None:
    """No head now. Raise, or return to write the head again (a first save that did not land)."""
    if base.etag is None:
        if repaired:
            raise _unknown(mine)
        return
    if maybe_applied:
        raise _unknown(mine)  # it may have landed, then the instance was deleted (or a false 404)
    raise InstanceGoneError(
        f"Instance {mine.instance_id} was deleted during this save. Nothing was saved.",
        code=ErrorCode.INSTANCE_NOT_FOUND,
        instance_id=mine.instance_id,
        hint=GONE_DURING_SAVE,
        context=None,
    )


# Hint parts of the conflict, gone and unknown-outcome errors of a save (here and manager.py)
LANDED_FIRST = (
    "If an earlier save of this session had an unknown outcome, look for its commit id in "
    "describe_instance(id).recent_commits first: it may be the save that is stored. "
)
SAVE_ELSEWHERE = (
    "Else save this session under another id to keep its work, or load the instance again (it has the other "
    "save) before you redo the work."
)
GONE_DURING_SAVE = (
    "The session stays open: save it under another id to keep its work, or start again with a new instance."
)
RETRY_SAME_SAVE = "A retry of the same save (the same instance_id) never stores a second copy."


def save_context(*, generation: int, commit_id: str) -> dict[str, object]:
    """Context of a save's INSTANCE_CONFLICT or INSTANCE_SAVE_OUTCOME_UNKNOWN: the generation
    and commit id it would make, to look for in describe_instance().recent_commits."""
    return {"generation": generation, "commit_id": commit_id}


def _conflict(mine: HeadHeader) -> OutcomeUnknownError:
    first = "an instance with this id already exists" if mine.generation == 1 else "another save committed first"
    return OutcomeUnknownError(
        f"Instance {mine.instance_id}: {first}. This save was not stored.",
        code=ErrorCode.INSTANCE_CONFLICT,
        instance_id=mine.instance_id,
        hint=LANDED_FIRST
        + ("Else this id is taken: save under another id." if mine.generation == 1 else SAVE_ELSEWHERE),
        context=save_context(generation=mine.generation, commit_id=mine.commit_id),
    )


def _unknown(mine: HeadHeader) -> OutcomeUnknownError:
    return OutcomeUnknownError(
        f"Instance {mine.instance_id}: it is unknown whether generation {mine.generation} "
        f"(commit {mine.commit_id}) was saved.",
        code=ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN,
        instance_id=mine.instance_id,
        hint='It was stored if describe_instance(id).recent_commits holds error.context["commit_id"]. '
        + RETRY_SAME_SAVE,
        context=save_context(generation=mine.generation, commit_id=mine.commit_id),
    )

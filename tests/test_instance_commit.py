"""Commit protocol against a moto server, with fixed interleavings (no timing).

The store talks to moto through the fault proxy (tests/s3_faults.py): lost
requests, lost answers and other writers come from the wire, not from patches.
"""

import asyncio
import secrets
from collections.abc import AsyncGenerator, Callable

import pytest
from pydantic import BaseModel, ConfigDict

from exec_sandbox.exceptions import ConfigError, ErrorCode, InstanceGoneError, OutcomeUnknownError
from exec_sandbox.instance import head, ids, keyring
from exec_sandbox.instance.commit import Base, Commit, commit_head
from exec_sandbox.instance.models import RECENT_COMMITS, RecentCommit
from exec_sandbox.instance.s3_store import S3InstanceStore
from exec_sandbox.instance.store import InstanceStore
from tests.instance_support import NO_VM_TIMEOUT, head_header, head_object, new_keyring
from tests.s3_faults import After, Before, DropAfter, DropBefore, Edit, Fail, Request, Response, S3Proxy

pytestmark = NO_VM_TIMEOUT

_RING = keyring.Keyring.parse(f"{new_keyring()},{new_keyring()}")
"""This worker's keyring: the newest key and an older one."""
_STRANGER = keyring.Key(id="f" * 64, secret=secrets.token_bytes(32))
"""A key-encryption key this worker does not have."""


class _Empty(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


@pytest.fixture
async def store(s3_proxy: S3Proxy) -> AsyncGenerator[InstanceStore]:
    async with S3InstanceStore.open(
        bucket=s3_proxy.bucket.name, prefix=s3_proxy.prefix, endpoint_url=s3_proxy.endpoint_url, region="us-east-1"
    ) as s:
        yield s


def _head(chain: "_Chain") -> str:
    """Proxy key regex of the chain's head object."""
    return f"^{head_object(chain.instance_id)}$"


def _body_is(commit: Commit) -> Callable[[Request], bool]:
    """Rule predicate: the request carries this commit's head bytes."""
    return lambda req: req.body == commit.data


def _lost_until_read_back(proxy: S3Proxy, chain: "_Chain") -> None:
    """Every send of the head PUT is dropped before the store until the saver reads the head
    back: the first write never reached the store, the re-send after the read-back does."""
    lost = proxy.add("PUT_IF", _head(chain), DropBefore(), times=None)

    async def reach_the_store_again(req: Request) -> None:
        lost.remove()

    proxy.add("GET", _head(chain), Before(reach_the_store_again))


def _resent_once_after_read_back(proxy: S3Proxy, chain: "_Chain", *, since: int) -> None:
    """The log shows lost head PUTs, one read-back, then the one re-send that landed."""
    ops = [(e.op, e.status) for e in proxy.entries(("PUT_IF", "GET"), _head(chain), since=since)]
    read_back = ops.index(("GET", 404 if chain.recent == () else 200))
    assert ops[:read_back] and set(ops[:read_back]) == {("PUT_IF", 0)}
    assert ops[read_back + 1 :] == [("PUT_IF", 200)]


class _Chain:
    """Builds successive heads of one instance (what savers would write)."""

    def __init__(self) -> None:
        self.instance_id = ids.new_id()
        self.recent: tuple[RecentCommit, ...] = ()
        self.data_key = secrets.token_bytes(32)

    def next(self, generation: int) -> Commit:
        return self.next_sealed(generation, kek=_RING.latest, data_key=self.data_key)

    def next_sealed(self, generation: int, *, kek: keyring.Key, data_key: bytes) -> Commit:
        """A commit as a saver writes it: its data key wrapped by `kek`, read back
        with this worker's keyring (the real check, head.open_header)."""
        header = head_header(
            instance_id=self.instance_id,
            generation=generation,
            commit_id=ids.new_id(),
            older=self.recent,
            kek=kek,
            data_key=data_key,
        )
        return Commit(
            header=header,
            data=head.seal_head(header=header, body=_Empty(), data_key=data_key),
            keyring=_RING,
        )

    def landed(self, commit: Commit) -> None:
        self.recent = commit.header.recent_commits


async def _save(store: InstanceStore, chain: _Chain, commit: Commit, base: Base) -> Base:
    etag = await commit_head(store, commit, base)
    chain.landed(commit)
    return Base(etag=etag, commit_id=commit.header.commit_id)


class TestCommitHead:
    async def test_first_then_next_save(self, store: InstanceStore) -> None:
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        second = chain.next(2)
        base = await _save(store, chain, second, base)
        assert await store.get(head_object(chain.instance_id), max_bytes=head.MAX_HEAD_BYTES) == (
            second.data,
            base.etag,
        )

    async def test_two_savers_from_one_base_second_conflicts(self, store: InstanceStore) -> None:
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        a, b = chain.next(2), chain.next(2)
        etag = await commit_head(store, a, base)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT) as exc:
            await commit_head(store, b, base)
        assert exc.value.action == "VERIFY"
        # What a caller compares with describe_instance().recent_commits
        assert exc.value.context == {"generation": 2, "commit_id": b.header.commit_id}
        assert await store.get(head_object(chain.instance_id), max_bytes=head.MAX_HEAD_BYTES) == (a.data, etag)

    async def test_lost_reply_resolves_to_success(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        """The store applied the PUT but the answer was lost: the saver finds its own commit."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        commit = chain.next(2)
        s3_proxy.add("PUT_IF", _head(chain), DropAfter(), times=None)
        etag = await commit_head(store, commit, base)
        got = await store.get(head_object(chain.instance_id), max_bytes=head.MAX_HEAD_BYTES)
        assert got == (commit.data, etag)
        assert {e.status for e in s3_proxy.entries("PUT_IF", _head(chain))[1:]} == {0}  # no answer reached it

    async def test_lost_request_is_resent_once(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        """The PUT never reached the store: the head is unchanged, the same bytes go again."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        commit = chain.next(2)
        mark = len(s3_proxy.log)
        _lost_until_read_back(s3_proxy, chain)
        etag = await commit_head(store, commit, base)
        _resent_once_after_read_back(s3_proxy, chain, since=mark)
        assert await store.get(head_object(chain.instance_id), max_bytes=head.MAX_HEAD_BYTES) == (commit.data, etag)

    async def test_unchanged_head_after_resend_is_unknown(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        s3_proxy.add("PUT_IF", _head(chain), DropBefore(), times=None)
        commit = chain.next(2)
        with pytest.raises(
            OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN
        ) as exc:
            await commit_head(store, commit, base)
        assert s3_proxy.count("GET", _head(chain)) == 2  # read back after the write and after the re-send
        assert exc.value.context == {"generation": 2, "commit_id": commit.header.commit_id}

    async def test_other_saver_with_another_data_key_is_a_conflict(
        self, store: InstanceStore, s3_proxy: S3Proxy
    ) -> None:
        """B committed a full copy under a new data key. A's write gets no answer (it may have
        landed), so A reads the head back: it opens it with the key that head names and learns
        "conflict", not "unknown"."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        a = chain.next(2)
        b = chain.next_sealed(2, kek=_RING.keys[1], data_key=secrets.token_bytes(32))  # B's full copy: new data key
        await commit_head(store, b, base)
        s3_proxy.add("PUT_IF", _head(chain), DropAfter(), times=None, when=_body_is(a))
        mark = len(s3_proxy.log)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT):
            await commit_head(store, a, base)
        assert [(e.op, e.status) for e in s3_proxy.entries(("PUT_IF", "GET"), _head(chain), since=mark)] == [
            ("PUT_IF", 0),
            ("GET", 200),
        ]

    async def test_applied_write_answered_refused_is_a_success(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        """A proxy that re-sends an applied PUT gets 412 for a write that landed: the read-back
        opens the head, finds this commit, and the save succeeds (never "conflict", which would
        delete the body the live head names)."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        mine = chain.next(2)

        def refused(resp: Response) -> None:
            resp.status = 412
            resp.body = b"<Error><Code>PreconditionFailed</Code></Error>"

        s3_proxy.add("PUT_IF", _head(chain), Edit(response=refused))
        etag = await commit_head(store, mine, base)
        assert await store.get(head_object(chain.instance_id), max_bytes=head.MAX_HEAD_BYTES) == (mine.data, etag)

    async def test_refused_write_under_a_foreign_head_is_a_conflict(self, store: InstanceStore) -> None:
        """Every write was refused and the head is another one: nothing of this save landed, so
        conflict, even when this keyring cannot open that head (e.g. a forced id another keyring uses)."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        a = chain.next(2)
        await commit_head(store, chain.next_sealed(2, kek=_STRANGER, data_key=secrets.token_bytes(32)), base)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT):
            await commit_head(store, a, base)

    async def test_unanswered_write_then_missing_head_is_unknown(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        """The write got no answer and may have landed: a missing head after it proves nothing."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))

        async def applied_then_deleted(req: Request, resp: Response) -> None:
            assert resp.status == 200
            await asyncio.to_thread(s3_proxy.bucket.delete, req.key)

        s3_proxy.add("PUT_IF", _head(chain), After(applied_then_deleted))
        s3_proxy.add("PUT_IF", _head(chain), DropAfter(), times=None)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN):
            await commit_head(store, chain.next(2), base)

    async def test_oversized_read_back_is_unknown(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        """A head no reader accepts, read back after an unanswered write: unknown, never "damaged"."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))

        async def oversized(req: Request, resp: Response) -> None:
            await asyncio.to_thread(s3_proxy.bucket.write, req.key, bytes(head.MAX_HEAD_BYTES + 1))

        s3_proxy.add("PUT_IF", _head(chain), After(oversized))
        s3_proxy.add("PUT_IF", _head(chain), DropAfter(), times=None)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN):
            await commit_head(store, chain.next(2), base)

    async def test_fork_far_behind_is_a_conflict(self, store: InstanceStore) -> None:
        """A saver more generations behind than a head lists: its write was refused (412),
        so nothing of it landed, although the head no longer lists its base."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        stale = chain.next(2)
        newer = base
        for generation in range(2, 3 + RECENT_COMMITS):
            newer = await _save(store, chain, chain.next(generation), newer)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT):
            await commit_head(store, stale, base)

    async def test_deleted_during_save(self, store: InstanceStore) -> None:
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        await store.delete_many([head_object(chain.instance_id)])
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND) as exc:
            await commit_head(store, chain.next(2), base)
        assert exc.value.action == "RECREATE"

    async def test_first_save_lost_reply(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        chain = _Chain()
        commit = chain.next(1)
        s3_proxy.add("PUT_IF", _head(chain), DropAfter(), times=None)
        etag = await commit_head(store, commit, Base(etag=None, commit_id=None))
        assert await store.get(head_object(chain.instance_id), max_bytes=head.MAX_HEAD_BYTES) == (commit.data, etag)

    async def test_first_save_lost_request_is_resent(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        chain = _Chain()
        commit = chain.next(1)
        _lost_until_read_back(s3_proxy, chain)
        etag = await commit_head(store, commit, Base(etag=None, commit_id=None))
        _resent_once_after_read_back(s3_proxy, chain, since=0)
        assert await store.get(head_object(chain.instance_id), max_bytes=head.MAX_HEAD_BYTES) == (commit.data, etag)

    async def test_first_save_still_absent_after_the_resend_is_unknown(
        self, store: InstanceStore, s3_proxy: S3Proxy
    ) -> None:
        """No answer to the first save's write nor to its one re-send, and no head: unknown
        (each write may land later), never a third send."""
        chain = _Chain()
        commit = chain.next(1)
        s3_proxy.add("PUT_IF", _head(chain), DropBefore(), times=2)
        with pytest.raises(
            OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN
        ) as exc:
            await commit_head(store, commit, Base(etag=None, commit_id=None))
        assert exc.value.instance_id == chain.instance_id
        assert [(e.op, e.status) for e in s3_proxy.entries(("PUT_IF", "GET"), _head(chain))] == [
            ("PUT_IF", 0),
            ("GET", 404),
            ("PUT_IF", 0),
            ("GET", 404),
        ]
        assert not s3_proxy.bucket.exists(head_object(chain.instance_id))

    async def test_store_error_on_the_resend_after_an_unanswered_write_is_unknown(
        self, store: InstanceStore, s3_proxy: S3Proxy
    ) -> None:
        """The first write got no answer (it may land later), the read-back finds the base head,
        and the store refuses the re-send: unknown, never the store's error alone (it would say
        that nothing of this save applied)."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        before = s3_proxy.bucket.read(head_object(chain.instance_id))
        s3_proxy.add("PUT_IF", _head(chain), DropBefore())
        s3_proxy.add("PUT_IF", _head(chain), Fail(403, "AccessDenied"), skip=1)
        with pytest.raises(
            OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN
        ) as exc:
            await commit_head(store, chain.next(2), base)
        cause = exc.value.__cause__
        assert isinstance(cause, ConfigError)
        assert cause.code is ErrorCode.INSTANCE_CONFIG
        assert s3_proxy.bucket.read(head_object(chain.instance_id)) == before

    async def test_store_error_on_the_resend_after_refused_writes_is_raised(
        self, store: InstanceStore, s3_proxy: S3Proxy
    ) -> None:
        """Every write was refused (412) and the head is the base: the re-send's store error
        is raised as it is (no write of this save applied), never "unknown"."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        before = s3_proxy.bucket.read(head_object(chain.instance_id))
        s3_proxy.add("PUT_IF", _head(chain), Fail(412, "PreconditionFailed"))
        s3_proxy.add("PUT_IF", _head(chain), Fail(403, "AccessDenied"), skip=1)
        with pytest.raises(
            ConfigError, match="AccessDenied", check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG
        ) as exc:
            await commit_head(store, chain.next(2), base)
        assert exc.value.action == "FIX_CONFIG"
        assert s3_proxy.bucket.read(head_object(chain.instance_id)) == before

    async def test_own_commit_superseded_before_read_back_returns_none(
        self, store: InstanceStore, s3_proxy: S3Proxy
    ) -> None:
        """Mine landed, then another saver committed on top: saved, but this session's
        base is gone (None), so its next save must not use the newer head's ETag."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        mine = chain.next(2)

        async def superseded(req: Request, resp: Response) -> None:
            assert resp.status == 200
            chain.landed(mine)
            await _save(store, chain, chain.next(3), Base(etag=resp.headers["ETag"], commit_id=mine.header.commit_id))

        s3_proxy.add("PUT_IF", _head(chain), After(superseded), when=_body_is(mine))
        s3_proxy.add("PUT_IF", _head(chain), DropAfter(), times=None, when=_body_is(mine))
        assert await commit_head(store, mine, base) is None

    async def test_own_commit_pushed_out_of_window_is_unknown(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        """Mine landed, then RECENT_COMMITS newer commits pushed it out of the head's list: no proof either way."""
        assert RECENT_COMMITS == 16  # InstanceInfo.recent_commits: "at most 16"
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        mine = chain.next(2)

        async def pushed_out(req: Request, resp: Response) -> None:
            assert resp.status == 200
            chain.landed(mine)
            newer = Base(etag=resp.headers["ETag"], commit_id=mine.header.commit_id)
            for generation in range(3, 3 + RECENT_COMMITS):
                newer = await _save(store, chain, chain.next(generation), newer)

        s3_proxy.add("PUT_IF", _head(chain), After(pushed_out), when=_body_is(mine))
        s3_proxy.add("PUT_IF", _head(chain), DropAfter(), times=None, when=_body_is(mine))
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN):
            await commit_head(store, mine, base)

    async def test_forged_head_listing_my_commit_is_unknown(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        """A bucket writer can learn my commit id (body keys) and write a head that lists it,
        sealed with its own key: the saver authenticates the read-back, so no false success."""
        chain = _Chain()
        base = await _save(store, chain, chain.next(1), Base(etag=None, commit_id=None))
        mine = chain.next(2)
        forged = head.seal_head(header=mine.header, body=_Empty(), data_key=secrets.token_bytes(32))
        await asyncio.to_thread(s3_proxy.bucket.write, head_object(chain.instance_id), forged)
        # My write gets no answer (it may have landed), so only the read-back can decide.
        s3_proxy.add("PUT_IF", _head(chain), DropAfter(), times=None)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN):
            await commit_head(store, mine, base)
        assert s3_proxy.bucket.read(head_object(chain.instance_id)) == forged

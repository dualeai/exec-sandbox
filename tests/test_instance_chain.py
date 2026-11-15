"""Instance chains in the store: rebuild, rollover, damage, lineage and S3 faults.

Moto behind the fault proxy (tests/s3_faults.py), a fake VM (tests/instance_support.py)
and the real InstanceManager, store, commit and pack code; no QEMU. The oracle is
byte-exact: each save records the two streams the fake guest had at the stopped
instant, and a load must rebuild exactly those bytes (or fail closed).
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import http.client
import os
import random
import re
import socket
from typing import TYPE_CHECKING

import pytest

from exec_sandbox import __version__, s3_parts
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InstanceGoneError,
    OutcomeUnknownError,
    RouteError,
    RouteKey,
    SandboxError,
    TransientError,
)
from exec_sandbox.images import image_digest
from exec_sandbox.instance import ids
from exec_sandbox.instance.head import MAX_HEAD_BYTES, open_head, seal_head
from exec_sandbox.instance.keyring import Keyring
from exec_sandbox.instance.manager import MAX_CHAIN_PACKS
from exec_sandbox.instance.models import BLOCK_BYTES, ChainEntry, HeadBody
from exec_sandbox.instance.pack import write_object
from exec_sandbox.instance.s3_store import S3InstanceStore
from exec_sandbox.instance.store import Applied, InstanceStore, NotApplied
from exec_sandbox.models import InstanceRef
from tests.instance_support import (
    BLOCK,
    BUDGET_KEYS,
    GOLDEN_KEY,
    GOLDEN_KEY_ID,
    NO_VM_TIMEOUT,
    PYTHON,
    FakeVm,
    InstanceEnv,
    Saver,
    SparseImage,
    await_until,
    chain_keys,
    constant_block,
    descriptors_of,
    digest,
    disk_full_at_download,
    fake_session,
    file_digest,
    head_header,
    head_object,
    kinds,
    random_block,
    worker_without_base_image,
)
from tests.s3_faults import (
    After,
    Before,
    DropAfter,
    DropBefore,
    Edit,
    Fail,
    Hold,
    Request,
    Response,
    S3Proxy,
    Truncate,
    moto_server,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from exec_sandbox.instance.manager import InstanceManager, InstanceState
    from exec_sandbox.session import Session

pytestmark = NO_VM_TIMEOUT

_MAX = 1024 * 1024
_REGION = "us-east-1"
_PART = s3_parts.PART_BYTES
"""S3 part and ranged-GET size, read from the code (a tuning value: no setting gives it). A body
object of at most one part loads with one GET; a larger one with a HEAD, then ranged GETs pinned to its ETag."""
_POSITION = {"full": 0, "first": 1, "middle": 2, "last": 3}
"""Objects of a chain of one full copy and three packs."""


@pytest.fixture
def env(instance_env: InstanceEnv) -> InstanceEnv:
    return instance_env


@pytest.fixture
async def store(s3_proxy: S3Proxy) -> AsyncGenerator[InstanceStore]:
    async with S3InstanceStore.open(
        bucket=s3_proxy.bucket.name, prefix=s3_proxy.prefix, endpoint_url=s3_proxy.endpoint_url, region=_REGION
    ) as s:
        yield s


def _exact(key: str) -> str:
    return f"^{re.escape(key)}$"


def _layout(seed: int) -> tuple[SparseImage, SparseImage]:
    """A guest whose full copy is four incompressible records (RAM blocks 0, 2; home blocks 0, 3)."""
    rng = random.Random(seed)
    ram, home = SparseImage(size=4 * BLOCK), SparseImage(size=8 * BLOCK)
    ram.put(0, random_block(rng))
    ram.put(2, random_block(rng))
    home.put(0, random_block(rng))
    home.put(3, random_block(rng))
    return ram, home


def _small_change(vm: FakeVm, n: int) -> None:
    """A pack of two constant-byte records: the same object size for every n."""
    vm.ram.put(1, constant_block(1 + n % 255))
    vm.home.put(1, constant_block(1 + (n + 128) % 255))


def _rewrite(vm: FakeVm, seed: int) -> None:
    """Six new incompressible blocks: a pack larger than a four-record full copy (the budget)."""
    rng = random.Random(f"rewrite-{seed}")  # never the layout's bytes
    for block_no in (0, 2, 3):
        vm.ram.put(block_no, random_block(rng))
    for block_no in (0, 3, 5):
        vm.home.put(block_no, random_block(rng))


async def _chain(env: InstanceEnv, *, packs: int, seed: int, manager: InstanceManager) -> Saver:
    """An instance saved 1 + `packs` times by `manager`: one full copy, then packs of the same size."""
    ram, home = _layout(seed)
    saver = Saver(env, await env.new_vm(ram, home, spec=PYTHON), manager=manager, state=None)
    await saver.save()
    for n in range(packs):
        _small_change(saver.vm, n + 2)
        await saver.save()
    assert saver.state is not None
    assert kinds(saver.state) == ["full"] + ["pack"] * packs
    return saver


async def _reuse(env: InstanceEnv, instance_id: str, *, manager: InstanceManager) -> Saver:
    """load() by `manager` and a VM restored from it: a session from reuse()."""
    state = await env.load(instance_id, manager=manager)
    return Saver(env, await env.vm_from(state), manager=manager, state=state)


# ----------------------------------------------------------------------------- the proxy itself


def test_moto_server_binds_with_no_name_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The moto server starts and answers while every name lookup fails: a lookup of its
    address at bind (http.server's getfqdn) blocked over 10 s on macOS runners."""

    def no_lookup(*_args: object) -> None:
        raise AssertionError("name lookup")

    monkeypatch.setattr(socket, "getfqdn", no_lookup)
    monkeypatch.setattr(socket, "gethostbyaddr", no_lookup)
    with moto_server() as server:
        assert server.server_name == "127.0.0.1"
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("POST", "/moto-api/reset")
            assert connection.getresponse().status == 200
        finally:
            connection.close()


class TestProxy:
    """The fault proxy itself: each action does what the chain tests rely on."""

    async def test_forwards_and_logs(self, store: InstanceStore, s3_proxy: S3Proxy, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.write_bytes(os.urandom(1000))
        etag = await store.upload("s/i/b/c.p", src=src)
        assert await store.size("s/i/b/c.p") == 1000
        assert '"' not in etag
        assert await store.download("s/i/b/c.p", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=1000)
        assert (tmp_path / "dest").read_bytes() == src.read_bytes()
        assert isinstance(await store.put_if_absent("s/i/head", data=b"h"), Applied)
        assert [o.key for o in await store.list_objects("s/i/")] == ["s/i/b/c.p", "s/i/head"]
        await store.delete_many(["s/i/head"])
        assert [(e.op, e.key, e.status) for e in s3_proxy.log] == [
            ("PUT", "s/i/b/c.p", 200),
            ("HEAD", "s/i/b/c.p", 200),
            ("GET", "s/i/b/c.p", 200),  # a known size of one part: one GET, no HEAD
            ("PUT_IF", "s/i/head", 200),
            ("LIST", "s/i/", 200),
            ("DELETE_MANY", "s/i/head", 200),
        ]
        assert s3_proxy.bucket.listing() == ["s/i/b/c.p"]

    async def test_fail_reaches_the_client_once_per_call(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        """SDK retries are off: one 503 is one failed call (RETRY), and the next call passes."""
        s3_proxy.bucket.write("k", b"x")
        rule = s3_proxy.add("GET", "^k$", Fail(503, "ServiceUnavailable"))
        with pytest.raises(
            TransientError, match="ServiceUnavailable", check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE
        ):
            await store.get("k", max_bytes=_MAX)
        assert (rule.hits, s3_proxy.count("GET", "^k$")) == (1, 1)
        got = await store.get("k", max_bytes=_MAX)
        assert got is not None
        assert got[0] == b"x"

    async def test_drop_before_never_applies_drop_after_does(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        """aiohttp sends an idempotent request (PUT too) once more after a dropped connection:
        a rule that must leave a call without an answer drops both sends (times=None)."""
        s3_proxy.add("PUT_IF", "^a$", DropBefore(), times=None)
        assert await store.put_if_absent("a", data=b"x") == NotApplied(maybe_applied=True)
        assert not s3_proxy.bucket.exists("a")
        s3_proxy.add("PUT_IF", "^b$", DropAfter(), times=None)
        assert await store.put_if_absent("b", data=b"x") == NotApplied(maybe_applied=True)
        assert s3_proxy.bucket.read("b") == b"x"
        assert {e.key for e in s3_proxy.entries("PUT_IF")} == {"a", "b"}
        assert {e.status for e in s3_proxy.entries("PUT_IF")} == {0}

    async def test_hold_waits_for_release(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        rule = s3_proxy.add("PUT_IF", "^k$", Hold())
        task = asyncio.create_task(store.put_if_absent("k", data=b"x"))
        await rule.reached.wait()
        assert not s3_proxy.bucket.exists("k")  # held before moto
        assert not task.done()
        rule.release()
        assert isinstance(await task, Applied)
        assert s3_proxy.bucket.read("k") == b"x"

    async def test_truncated_body_is_a_transport_failure(
        self, store: InstanceStore, s3_proxy: S3Proxy, tmp_path: Path
    ) -> None:
        (tmp_path / "src").write_bytes(os.urandom(5000))
        await store.upload("k", src=tmp_path / "src")
        s3_proxy.add("GET", "^k$", Truncate(100))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE):
            await store.download("k", dest=tmp_path / "dest", max_bytes=_MAX, expected_size=5000)
        assert not (tmp_path / "dest").exists()

    async def test_edit_request_and_response(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        """Strip a PUT's condition (moto then overwrites), and change an answer's header."""
        s3_proxy.bucket.write("k", b"old")

        def unconditional(req: Request) -> None:
            req.headers.popall("If-None-Match")

        def no_etag(resp: Response) -> None:
            resp.headers.popall("ETag")

        s3_proxy.add("PUT_IF", "^k$", Edit(request=unconditional))
        assert isinstance(await store.put_if_absent("k", data=b"new"), Applied)
        assert s3_proxy.bucket.read("k") == b"new"
        s3_proxy.add("GET", "^k$", Edit(response=no_etag))
        with pytest.raises(
            TransientError, match="ETag", check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE
        ):
            await store.get("k", max_bytes=_MAX)

    async def test_head_answer_size_can_be_edited(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        s3_proxy.bucket.write("k", b"12345")

        def lies(resp: Response) -> None:
            resp.headers["Content-Length"] = "7"

        s3_proxy.add("HEAD", "^k$", Edit(response=lies))
        assert await store.size("k") == 7
        assert await store.size("k") == 5

    async def test_before_and_after_hooks_see_moto(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        bucket = s3_proxy.bucket
        bucket.write("k", b"x")

        async def delete_first(req: Request) -> None:
            await asyncio.to_thread(bucket.delete, req.key)

        seen: list[int] = []

        async def record(req: Request, resp: Response) -> None:
            seen.append(resp.status)
            await asyncio.to_thread(bucket.write, "k2", b"y")

        s3_proxy.add("GET", "^k$", Before(delete_first))
        s3_proxy.add("GET", "^k$", After(record))
        assert await store.get("k", max_bytes=_MAX) is None
        assert seen == [404]
        assert bucket.read("k2") == b"y"

    async def test_skip_and_times(self, store: InstanceStore, s3_proxy: S3Proxy) -> None:
        s3_proxy.bucket.write("k", b"x")
        rule = s3_proxy.add("HEAD", "^k$", Fail(404, "NoSuchKey"), times=2, skip=1)
        sizes = [await store.size("k") for _ in range(5)]
        assert sizes == [1, None, None, 1, 1]
        assert (rule.matched, rule.hits) == (5, 2)

    @pytest.mark.parametrize("writers", [2, 8])
    async def test_concurrent_if_match_puts_on_one_key_one_wins(
        self, store: InstanceStore, s3_proxy: S3Proxy, writers: int
    ) -> None:
        """Conditional PUTs on one key are serialized: of N If-Match PUTs on one ETag, exactly one applies."""
        first = await store.put_if_absent("head", data=b"base")
        assert isinstance(first, Applied)
        gate = s3_proxy.add("PUT_IF", "^head$", Hold(), times=None)
        tasks = [
            asyncio.create_task(store.put_if_match("head", data=f"v{n}".encode(), etag=first.etag))
            for n in range(writers)
        ]
        await await_until(lambda: gate.hits == writers)  # every writer waits at the gate, each on its own connection
        gate.release()
        results = await asyncio.gather(*tasks)
        applied = [r for r in results if isinstance(r, Applied)]
        assert len(applied) == 1, results
        assert [r for r in results if isinstance(r, NotApplied)] == [NotApplied(maybe_applied=False)] * (writers - 1)
        current = await store.get("head", max_bytes=_MAX)
        assert current is not None
        assert current[1] == applied[0].etag


# ----------------------------------------------------------------------------- chain shape


class TestChainShape:
    async def test_chain_rebuilds_every_generation(self, env: InstanceEnv) -> None:
        """A full copy and ten packs (blocks written, grown, cut, discarded, nothing changed):
        after every save a fresh load rebuilds both streams byte for byte, and the store
        holds only the chain, the head and the guest image."""
        rng = random.Random(1)
        steps = [
            lambda vm: _small_change(vm, 11),
            lambda vm: (vm.home.resize(10 * BLOCK + 777), vm.home.put(9, constant_block(13))),
            lambda vm: vm.home.discard(3),
            lambda vm: (vm.ram.resize(6 * BLOCK), vm.ram.put(5, constant_block(14))),
            lambda vm: vm.home.resize(5 * BLOCK + 100),
            lambda vm: vm.home.put(2, random_block(rng)),
            lambda vm: vm.ram.discard(0),
            lambda vm: vm.ram.resize(3 * BLOCK),
            lambda vm: (vm.home.put(4, constant_block(15)), vm.ram.put(1, constant_block(16))),
            lambda vm: None,  # nothing changed: a pack without records
        ]
        ram, home = _layout(1)
        saver = Saver(env, await env.new_vm(ram, home, spec=PYTHON), manager=env.manager, state=None)
        state = await saver.save()
        await env.check_load(state.instance_id, generation=1, manager=env.manager, want=None)
        for generation, step in enumerate(steps, start=2):
            step(saver.vm)
            state = await saver.save()
            assert kinds(state) == ["full"] + ["pack"] * (generation - 1)
            await env.check_load(state.instance_id, generation=generation, manager=env.manager, want=None)
        asset = f"assets/sha256/{hashlib.sha256(env.base_image.read_bytes()).hexdigest()}"
        assert env.bucket.listing() == sorted([head_object(state.instance_id), *chain_keys(state), asset])

    async def test_byte_budget_rollover(self, env: InstanceEnv) -> None:
        """Packs that would outgrow the full copy start a new chain: new data key, old chain deleted."""
        ram, home = _layout(2)
        saver = Saver(env, await env.new_vm(ram, home, spec=PYTHON), manager=env.manager, state=None)
        rng = random.Random(2)
        previous: InstanceState | None = None
        for generation, kind in enumerate(["full", "pack", "full", "pack", "full"], start=1):
            if generation > 1:  # three of the four incompressible blocks change
                saver.vm.ram.put(0, random_block(rng))
                saver.vm.home.put(0, random_block(rng))
                saver.vm.home.put(3, random_block(rng))
            state = await saver.save()
            assert kinds(state)[-1] == kind, generation
            assert env.bucket.body_keys(state.instance_id) == sorted(chain_keys(state))
            if previous is not None:
                same_key = state.data_key == previous.data_key
                assert same_key == (kind == "pack")
            await env.check_load(state.instance_id, generation=generation, manager=env.manager, want=None)
            previous = state

    async def test_recent_commits_are_the_last_16_saves(self, env: InstanceEnv) -> None:
        """After 18 saves the head lists the last 16 (generation and commit id), newest first,
        as the manager wrote them (describe), not as a test helper builds a head."""
        ram, home = SparseImage(size=BLOCK), SparseImage(size=BLOCK)
        saver = Saver(env, await env.new_vm(ram, home, spec=PYTHON), manager=env.manager, state=None)
        commit_of: dict[int, str] = {}
        for generation in range(1, 19):
            saver.vm.ram.put(0, constant_block(generation))
            state = await saver.save()
            commit_of[generation] = state.header.commit_id
        info = await env.manager.describe(saver.instance_id)
        assert [(c.generation, c.commit_id) for c in info.recent_commits] == [
            (generation, commit_of[generation]) for generation in range(18, 2, -1)
        ]

    async def test_pack_cap_starts_a_full_copy(self, env: InstanceEnv) -> None:
        """MAX_CHAIN_PACKS packs, then a full copy; the longest chain loads with one GET per object."""
        assert MAX_CHAIN_PACKS == 64  # README: a full copy "after 64 packs"
        rng = random.Random(3)
        ram, home = SparseImage(size=2 * BLOCK), SparseImage(size=2 * BLOCK)
        ram.put(0, random_block(rng))  # the budget: one incompressible record; each pack is ~150 bytes
        saver = Saver(env, await env.new_vm(ram, home, spec=PYTHON), manager=env.manager, state=None)
        state = await saver.save()
        for generation in range(2, MAX_CHAIN_PACKS + 2):
            saver.vm.ram.put(1, constant_block(generation))
            mark = len(env.proxy.log)
            state = await saver.save()
            assert kinds(state)[-1] == "pack", generation
            assert env.proxy.count("LIST", since=mark) == 1, generation
        assert len(env.bucket.body_keys(state.instance_id)) == MAX_CHAIN_PACKS + 1
        assert env.bucket.size(head_object(state.instance_id)) < 64 * 1024
        mark = len(env.proxy.log)
        await env.check_load(state.instance_id, generation=MAX_CHAIN_PACKS + 1, manager=env.manager, want=None)
        assert env.proxy.count("GET", "/b/", since=mark) == MAX_CHAIN_PACKS + 1
        assert env.proxy.count(("HEAD", "GET_RANGE"), "/b/", since=mark) == 0

        saver.vm.ram.put(1, constant_block(200))
        state = await saver.save()
        assert kinds(state) == ["full"]
        assert env.bucket.body_keys(state.instance_id) == chain_keys(state)
        await env.check_load(state.instance_id, generation=MAX_CHAIN_PACKS + 2, manager=env.manager, want=None)


# ----------------------------------------------------------------------------- rollover


class TestRollover:
    async def test_key_rotation_rollover(self, env: InstanceEnv) -> None:
        """A new newest key: the next save is a full copy under it, the old chain goes, and the old key
        alone can no longer read the instance (FIX_CONFIG, not damage)."""
        saver = await _chain(env, packs=1, seed=4, manager=env.manager)
        new_key = GOLDEN_KEY
        rotated = env.manager_for(f"{new_key},{env.keyring}")
        worker = await _reuse(env, saver.instance_id, manager=rotated)
        _small_change(worker.vm, 9)
        state = await worker.save()
        assert kinds(state) == ["full"]
        assert state.header.key_id == GOLDEN_KEY_ID
        assert env.bucket.body_keys(state.instance_id) == chain_keys(state)
        await env.check_load(state.instance_id, generation=3, manager=env.manager_for(new_key), want=None)
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_KEY_MISSING) as exc:
            await env.load(state.instance_id, manager=env.manager_for(env.keyring))
        assert (exc.value.action, exc.value.instance_id) == ("FIX_CONFIG", state.instance_id)

    async def test_failed_old_chain_delete_leaves_harmless_leftovers(self, env: InstanceEnv) -> None:
        """The deletes after a rollover fail: the save still succeeds, and damaged leftovers of the
        old chain change neither the next load nor the next pack."""
        saver = await _chain(env, packs=1, seed=5, manager=env.manager)
        old = chain_keys(saver.state)  # type: ignore[arg-type]
        env.proxy.add("DELETE_MANY", "", Fail(503, "ServiceUnavailable"), times=None)
        _rewrite(saver.vm, 5)
        state = await saver.save()
        assert kinds(state) == ["full"]
        assert [e.status for e in env.proxy.entries("DELETE_MANY")] == [503]
        assert set(env.bucket.body_keys(state.instance_id)) == {*old, *chain_keys(state)}
        env.bucket.flip(old[0], 100)
        env.bucket.delete(old[1])
        await env.check_load(state.instance_id, generation=3, manager=env.manager, want=None)
        _small_change(saver.vm, 77)
        state = await saver.save()
        assert kinds(state) == ["full", "pack"]
        await env.check_load(state.instance_id, generation=4, manager=env.manager, want=None)

    async def test_load_during_rollover_follows_new_chain(self, env: InstanceEnv) -> None:
        """A load reading the old chain when a rollover deletes it gets a 404, reads the head again,
        and loads the new generation."""
        saver = await _chain(env, packs=1, seed=6, manager=env.manager)
        old_pack = chain_keys(saver.state)[1]  # type: ignore[arg-type]
        gate = env.proxy.add("GET", _exact(old_pack), Hold())
        mark = len(env.proxy.log)
        loading = asyncio.create_task(env.load(saver.instance_id, manager=env.manager))
        await gate.reached.wait()
        _rewrite(saver.vm, 6)
        state = await saver.save()
        assert kinds(state) == ["full"]
        assert old_pack not in env.bucket.listing()
        gate.release()
        loaded = await loading
        try:
            assert loaded.header.commit_id == state.header.commit_id
            want = env.saved[state.header.commit_id]
            assert (file_digest(loaded.vmstate), file_digest(loaded.home)) == (digest(want[0]), digest(want[1]))
        finally:
            await env.manager.release(loaded, wait=True)
        assert env.proxy.entries("GET", _exact(old_pack), since=mark)[-1].status == 404
        assert env.proxy.count("GET", _exact(head_object(state.instance_id)), since=mark) == 2

    async def test_rollover_unknown_outcome_keeps_old_chain(self, env: InstanceEnv) -> None:
        """No answer to the head write and no read-back: unknown outcome (VERIFY), and nothing is
        deleted, since the new head (which landed) needs the new full copy and the old head the old chain."""
        saver = await _chain(env, packs=1, seed=7, manager=env.manager)
        old = chain_keys(saver.state)  # type: ignore[arg-type]
        head = _exact(head_object(saver.instance_id))
        env.proxy.add("PUT_IF", head, DropAfter(), times=None)
        env.proxy.add("GET", head, Fail(503, "ServiceUnavailable"), times=None)
        _rewrite(saver.vm, 7)
        want = saver.vm.expected()
        with pytest.raises(
            OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN
        ) as exc:
            await saver.save()
        assert (exc.value.action, exc.value.instance_id) == ("VERIFY", saver.instance_id)
        assert env.proxy.count("DELETE_MANY") == 0
        (new_full,) = set(env.bucket.body_keys(saver.instance_id)) - set(old)
        assert new_full.endswith(".f")
        env.proxy.clear()
        loaded = await env.check_load(saver.instance_id, generation=3, want=want, manager=env.manager)
        assert chain_keys(loaded) == [new_full]

    async def test_stale_save_after_rollover_conflicts(self, env: InstanceEnv) -> None:
        """A loaded generation n; B rolled over to n+1. A's LIST misses the old chain, so A seals a
        full copy; its head write is refused: conflict (VERIFY), A's object deleted, B's chain intact."""
        saver = await _chain(env, packs=1, seed=8, manager=env.manager)
        a = await _reuse(env, saver.instance_id, manager=env.manager)
        b = await _reuse(env, saver.instance_id, manager=env.manager)
        _rewrite(b.vm, 8)
        b_state = await b.save()
        assert kinds(b_state) == ["full"]
        _small_change(a.vm, 50)
        mark = len(env.proxy.log)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT) as exc:
            await a.save()
        assert (exc.value.action, exc.value.instance_id) == ("VERIFY", saver.instance_id)
        assert env.proxy.count(("PUT", "MPU_CREATE"), r"\.f$", since=mark) == 1
        assert env.proxy.count(("PUT", "MPU_CREATE"), r"\.p$", since=mark) == 0
        assert env.bucket.body_keys(saver.instance_id) == chain_keys(b_state)
        await env.check_load(saver.instance_id, generation=3, manager=env.manager, want=None)


# ----------------------------------------------------------------------------- commit outcome


class TestCommitOutcome:
    @pytest.mark.parametrize("outcome", ["unknown", "conflict"])
    async def test_uploaded_object_is_kept_only_when_the_head_may_name_it(self, env: InstanceEnv, outcome: str) -> None:
        """Unknown outcome and conflict share one class (OutcomeUnknownError, VERIFY); only the
        code tells them apart. Unknown (no answer to the head write, no read-back): the head may
        name the pack this save uploaded, so it stays. Conflict (another save won): the pack is
        this save's alone, so it is deleted."""
        saver = await _chain(env, packs=0, seed=71, manager=env.manager)
        a = await _reuse(env, saver.instance_id, manager=env.manager)
        head = _exact(head_object(saver.instance_id))
        if outcome == "conflict":
            b = await _reuse(env, saver.instance_id, manager=env.manager)
            _small_change(b.vm, 71)
            assert kinds(await b.save()) == ["full", "pack"]
            code = ErrorCode.INSTANCE_CONFLICT
        else:
            env.proxy.add("PUT_IF", head, DropAfter(), times=None)
            env.proxy.add("GET", head, Fail(503, "ServiceUnavailable"), times=None)
            code = ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN
        _small_change(a.vm, 72)
        mark = len(env.proxy.log)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is code) as exc:
            await a.save()
        assert (exc.value.action, exc.value.instance_id) == ("VERIFY", saver.instance_id)
        (uploaded,) = {e.key for e in env.proxy.entries(("PUT", "MPU_CREATE"), r"/b/.*\.p$", since=mark)}
        assert env.bucket.exists(uploaded) == (outcome == "unknown")


# ----------------------------------------------------------------------------- damage at load


_IN_CHAIN_DAMAGE = [
    "delete",
    "flip-record",
    "flip-index",
    "flip-magic",
    "flip-index-offset",
    "cut-1",
    "cut-all",
    "append-1",
    "other-commit-same-size",
    "other-commit-other-size",
    "other-instance-same-size",
]
_SIZE_DAMAGE = frozenset({"cut-1", "cut-all", "append-1", "other-commit-other-size"})


async def _damage(env: InstanceEnv, saver: Saver, *, target: str, damage: str) -> str:
    """Damage one object of a full + 3 packs chain in the store; returns its key."""
    keys = chain_keys(saver.state)  # type: ignore[arg-type]
    key = keys[_POSITION[target]]
    bucket = env.bucket
    match damage:
        case "delete":
            bucket.delete(key)
        case "flip-record":
            bucket.flip(key, bucket.index_offset(key) // 2)
        case "flip-index":
            bucket.flip(key, bucket.index_offset(key) + 20)
        case "flip-magic":
            bucket.flip(key, -1)
        case "flip-index-offset":
            bucket.flip(key, -20)
        case "cut-1":
            bucket.truncate(key, bucket.size(key) - 1)
        case "cut-all":
            bucket.truncate(key, 0)
        case "append-1":
            bucket.append(key, b"\0")
        case "other-commit-same-size":
            other = keys[2] if target == "first" else keys[1]
            assert bucket.size(other) == bucket.size(key)
            bucket.swap(other, key)
        case "other-commit-other-size":
            other = keys[1] if target == "full" else keys[0]
            assert bucket.size(other) != bucket.size(key)
            bucket.swap(other, key)
        case "other-instance-same-size":
            twin = await _chain(env, packs=3, seed=999, manager=env.manager)
            other = chain_keys(twin.state)[_POSITION[target]]  # type: ignore[arg-type]
            assert bucket.size(other) == bucket.size(key)
            bucket.swap(other, key)
        case _:
            raise AssertionError(damage)
    return key


class TestDamageAtLoad:
    @pytest.mark.parametrize(
        ("target", "damage"),
        [
            pytest.param(t, d, id=f"{t}-{d}")
            for t in _POSITION
            for d in _IN_CHAIN_DAMAGE
            if not (t == "full" and d == "other-commit-same-size")  # the chain has one full copy
        ],
    )
    async def test_in_chain_damage_fails_closed(self, env: InstanceEnv, target: str, damage: str) -> None:
        """Any change to a chain object fails the load (RECREATE), names the instance, and leaves no
        file. A missing object is asked for twice (GET, then head re-read and HEAD) before "expired";
        a wrong size is asked twice (GET, HEAD) before "damaged", and its body is not read."""
        saver = await _chain(env, packs=3, seed=9, manager=env.manager)
        key = await _damage(env, saver, target=target, damage=damage)
        await env.manager.drain()  # the last save's old files go in a worker thread
        files = env.work_files()
        mark = len(env.proxy.log)
        no_body = (
            env.proxy.add("GET", _exact(key), Truncate(0))  # the headers, no body bytes: a body read fails short
            if damage in _SIZE_DAMAGE
            else None
        )
        expected = ErrorCode.INSTANCE_EXPIRED if damage == "delete" else ErrorCode.INSTANCE_INTEGRITY
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is expected) as exc:
            await env.load(saver.instance_id, manager=env.manager)
        assert (exc.value.action, exc.value.instance_id) == ("RECREATE", saver.instance_id)
        if expected is ErrorCode.INSTANCE_INTEGRITY:  # the object and the commit that wrote it
            assert key.endswith(exc.value.context["key"])
            name = exc.value.context["key"].split("/")[-1]
            assert name in {f"{exc.value.context['commit_id']}.f", f"{exc.value.context['commit_id']}.p"}
            assert len(exc.value.context["commit_id"]) == 32
        await env.manager.drain()  # the failed load's folder goes in a worker thread
        assert env.work_files() == files
        if damage == "delete":
            assert env.proxy.count("GET", _exact(head_object(saver.instance_id)), since=mark) == 2
        if damage == "delete" or damage in _SIZE_DAMAGE:
            asked = {op: env.proxy.count(op, _exact(key), since=mark) for op in ("GET", "HEAD", "GET_RANGE")}
            assert asked == {"GET": 1, "HEAD": 1, "GET_RANGE": 0}
        if no_body is not None:
            assert no_body.hits == 1

    @pytest.mark.parametrize("damage", ["delete", "flip", "truncate", "oversized", "other-instance"])
    async def test_head_damage(self, env: InstanceEnv, damage: str) -> None:
        """A missing head is "not found"; a changed, cut, oversized or foreign head is damaged; no body is read."""
        saver = await _chain(env, packs=1, seed=10, manager=env.manager)
        head = head_object(saver.instance_id)
        match damage:
            case "delete":
                env.bucket.delete(head)
            case "flip":
                env.bucket.flip(head, -1)
            case "truncate":
                env.bucket.truncate(head, env.bucket.size(head) // 2)
            case "oversized":
                env.bucket.append(head, bytes(MAX_HEAD_BYTES))
            case "other-instance":
                twin = await _chain(env, packs=0, seed=1010, manager=env.manager)
                env.bucket.swap(head_object(twin.instance_id), head)
        mark = len(env.proxy.log)
        expected = ErrorCode.INSTANCE_NOT_FOUND if damage == "delete" else ErrorCode.INSTANCE_INTEGRITY
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is expected) as exc:
            await env.load(saver.instance_id, manager=env.manager)
        assert (exc.value.action, exc.value.instance_id) == ("RECREATE", saver.instance_id)
        assert env.proxy.count(("HEAD", "GET", "GET_RANGE"), "/b/", since=mark) == 0
        # Where it looked: a wrong bucket, prefix or folder gives the same codes
        named = {"store": env.store.location} if damage == "delete" else {}
        assert {k: exc.value.context[k] for k in ("key", *named)} == {"key": head_object(saver.instance_id), **named}
        if damage == "delete":
            assert f"in the store {env.store.location} " in str(exc.value)

    async def test_older_head_with_its_chain_loads_the_older_generation(self, env: InstanceEnv) -> None:
        """Accepted limit: an authentic older head put back while its chain is stored loads that generation."""
        saver = await _chain(env, packs=0, seed=101, manager=env.manager)
        head = head_object(saver.instance_id)
        older = env.bucket.read(head)
        _small_change(saver.vm, 3)
        await saver.save()
        env.bucket.write(head, older)
        await env.check_load(saver.instance_id, generation=1, manager=env.manager, want=None)

    async def test_older_head_after_rollover_is_expired(self, env: InstanceEnv) -> None:
        saver = await _chain(env, packs=0, seed=102, manager=env.manager)
        head = head_object(saver.instance_id)
        older = env.bucket.read(head)
        _rewrite(saver.vm, 102)
        assert kinds(await saver.save()) == ["full"]  # the generation-1 chain is deleted
        env.bucket.write(head, older)
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_EXPIRED) as exc:
            await env.load(saver.instance_id, manager=env.manager)
        assert exc.value.instance_id == saver.instance_id

    @pytest.mark.parametrize("damage", ["loser-orphan", "leftovers", "garbage-keys", "other-instance"])
    async def test_outside_chain_damage_does_not_matter(self, env: InstanceEnv, damage: str) -> None:
        """Objects the head does not name (a conflict loser's upload, an old chain, junk, another
        instance's objects) change neither the load nor the next save, which stays a pack."""
        saver = await _chain(env, packs=1, seed=11, manager=env.manager)
        iid = saver.instance_id
        match damage:
            case "loser-orphan":
                env.bucket.write(f"s/{iid}/b/{ids.new_id()}.p", os.urandom(500))
            case "leftovers":
                env.bucket.write(f"s/{iid}/b/{ids.new_id()}.f", os.urandom(5000))
                env.bucket.write(f"s/{iid}/b/{ids.new_id()}.p", os.urandom(500))
            case "garbage-keys":
                for name in ("garbage", "x.y.z", "sub/dir/k", f"{ids.new_id()}.q"):
                    env.bucket.write(f"s/{iid}/b/{name}", b"junk")
            case "other-instance":
                other = ids.new_id()
                env.bucket.write(head_object(other), b"not a head")
                env.bucket.write(f"s/{other}/b/{ids.new_id()}.f", os.urandom(5000))
        await env.check_load(iid, generation=2, manager=env.manager, want=None)
        _small_change(saver.vm, 12)
        state = await saver.save()
        assert kinds(state) == ["full", "pack", "pack"]
        await env.check_load(iid, generation=3, manager=env.manager, want=None)

    @pytest.mark.parametrize("damage", ["intact", "delete", "flip", "truncate"])
    @pytest.mark.parametrize("on_worker", [True, False], ids=["image-on-worker", "no-image"])
    async def test_asset_damage(self, env: InstanceEnv, damage: str, on_worker: bool) -> None:
        """A worker with the base image never reads the store's copy; a worker without it loads
        an intact copy by digest, and ROUTEs on a missing or changed one."""
        saver = await _chain(env, packs=1, seed=12, manager=env.manager)
        asset = f"assets/sha256/{saver.state.body.guest.base_image}"  # type: ignore[union-attr]
        match damage:
            case "delete":
                env.bucket.delete(asset)
            case "flip":
                env.bucket.flip(asset, 1000)
            case "truncate":
                env.bucket.truncate(asset, 1000)
        env.vm_manager.base_image = env.base_image if on_worker else None
        if on_worker or damage == "intact":
            await env.check_load(saver.instance_id, generation=2, manager=env.manager, want=None)
            return
        await env.manager.drain()  # the last save's old files go in a worker thread
        files = env.work_files()
        with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
            await env.load(saver.instance_id, manager=env.manager)
        assert (exc.value.action, exc.value.instance_id) == ("ROUTE", saver.instance_id)
        guest = saver.state.body.guest  # type: ignore[union-attr]
        host = saver.state.body.host  # type: ignore[union-attr]
        # The whole target: the checked facts (this worker has them) and the image it lacks.
        assert exc.value.required[RouteKey.BASE_IMAGE] == guest.base_image
        assert exc.value.required[RouteKey.MACHINE_TYPE] == host.machine_type
        assert exc.value.required[RouteKey.MEMORY_MB] == "272"  # the budget checked before (192 MB + 80 MB)
        assert exc.value.found == {**exc.value.required, RouteKey.BASE_IMAGE: None, **dict.fromkeys(BUDGET_KEYS)}
        # The message names only the value that does not hold (the budget and the host pass)
        assert str(exc.value).startswith(
            f'Instance {saver.instance_id} needs base_image="{guest.base_image}"; this worker has base_image=null: '
        )
        assert exc.value.context["reason"] == (
            "the image is not on this worker and not in the store"
            if damage == "delete"
            else "the store's copy of the image does not match its digest"
        )
        await env.manager.drain()  # the failed load's asset .part goes in a worker thread
        assert env.work_files() == files

    async def test_damaged_local_image_is_never_stored(self, env: InstanceEnv) -> None:
        """An image changed with its inode, mtime and size kept keeps its old digest (the
        digest cache is keyed by that stamp): the first save to a store must not put those
        bytes under that digest. The save fails and the store holds nothing."""
        digest = hashlib.sha256(env.base_image.read_bytes()).hexdigest()  # the oracle: the good bytes
        assert image_digest(env.base_image) == digest
        before = env.base_image.stat()
        damaged = bytearray(env.base_image.read_bytes())
        damaged[100] ^= 0x01
        env.base_image.write_bytes(bytes(damaged))
        os.utime(env.base_image, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert image_digest(env.base_image) == digest  # the stamp still says "good"
        ram, home = _layout(14)
        saver = Saver(env, await env.new_vm(ram, home, spec=PYTHON), manager=env.manager, state=None)
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.IMAGE_DAMAGED) as exc:
            await saver.save()
        assert exc.value.context == {"path": str(env.base_image), "digest": digest}
        assert exc.value.hint == "Download or rebuild the guest images."
        assert env.bucket.listing() == []

    @pytest.mark.parametrize("release", ["9.9.9", __version__], ids=["other-release", "this-release"])
    async def test_authenticated_unreadable_object_routes(self, env: InstanceEnv, tmp_path: Path, release: str) -> None:
        """A pack that authenticates but holds a stream this release does not know: ROUTE to another
        release (empty `required`), never RECREATE (which would delete good data), whatever
        release the head names: the release is no load condition and no route key."""
        saver = await _chain(env, packs=0, seed=13, manager=env.manager)
        state = saver.state
        assert state is not None
        iid, commit_id = state.instance_id, ids.new_id()
        block = tmp_path / "block"
        block.write_bytes(constant_block(7))
        obj = tmp_path / "pack"
        write_object(
            obj,
            data_key=state.data_key,
            instance_id=iid,
            commit_id=commit_id,
            streams={0: (block, None), 2: (block, None)},
        )
        etag = env.bucket.write(f"s/{iid}/b/{commit_id}.p", obj.read_bytes())
        entry = ChainEntry(
            commit_id=commit_id,
            kind="pack",
            size=obj.stat().st_size,
            stream_sizes=state.body.chain[0].stream_sizes,
            etag=etag,
            block_bytes=BLOCK_BYTES,
        )
        header = head_header(
            instance_id=iid,
            generation=2,
            commit_id=commit_id,
            older=state.header.recent_commits,
            kek=Keyring.parse(env.keyring).latest,
            data_key=state.data_key,
        )
        host = state.body.host.model_copy(update={"release": release})
        body = HeadBody(
            spec=state.body.spec,
            host=host,
            guest=state.body.guest,
            egress_domains=state.body.egress_domains,
            chain=(*state.body.chain, entry),
        )
        env.bucket.write(head_object(iid), seal_head(header=header, body=body, data_key=state.data_key))
        with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
            await env.load(iid, manager=env.manager)
        assert (exc.value.action, exc.value.instance_id, exc.value.required) == ("ROUTE", iid, {})

    async def test_object_deleted_during_load_is_expired(self, env: InstanceEnv) -> None:
        saver = await _chain(env, packs=1, seed=14, manager=env.manager)
        full = chain_keys(saver.state)[0]  # type: ignore[arg-type]

        async def delete(req: Request) -> None:
            await asyncio.to_thread(env.bucket.delete, req.key)

        env.proxy.add("GET", _exact(full), Before(delete))
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_EXPIRED) as exc:
            await env.load(saver.instance_id, manager=env.manager)
        assert (exc.value.action, exc.value.instance_id) == ("RECREATE", saver.instance_id)

    async def test_false_no_such_key_is_retry_not_expired(self, env: InstanceEnv) -> None:
        """The store answers "no such key" for a chain object it has; read again, the head is the
        same: one HEAD finds the object, so the store failed (RETRY), and the instance is never
        reported expired (a client would delete good data). The next load reads it."""
        saver = await _chain(env, packs=1, seed=144, manager=env.manager)
        iid = saver.instance_id
        full = chain_keys(saver.state)[0]  # type: ignore[arg-type]
        rule = env.proxy.add("GET", _exact(full), Fail(404, "NoSuchKey"))
        mark = len(env.proxy.log)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await env.load(iid, manager=env.manager)
        assert exc.value.message == f"Instance {iid}: the store answered 'no such key' for {full}, which it has."
        assert (exc.value.action, exc.value.instance_id) == ("RETRY", iid)
        assert (rule.hits, env.proxy.count("HEAD", _exact(full), since=mark)) == (1, 1)
        await env.check_load(iid, generation=2, manager=env.manager, want=None)

    async def test_head_that_keeps_changing_during_load_is_retry(self, env: InstanceEnv) -> None:
        """Each load attempt finds its object gone and, read again, a new head (another ETag):
        after three attempts the store is busy (RETRY), and the instance is not reported expired."""
        saver = await _chain(env, packs=0, seed=143, manager=env.manager)
        iid = saver.instance_id
        (full,) = chain_keys(saver.state)  # type: ignore[arg-type]
        head = head_object(iid)
        ring = Keyring.parse(env.keyring)

        def gone_and_new_head() -> None:
            env.bucket.delete(full)
            header, body, data_key = open_head(env.bucket.read(head), ring, instance_id=iid)
            env.bucket.write(head, seal_head(header=header, body=body, data_key=data_key))  # new nonce: new ETag

        async def before_body_read(req: Request) -> None:
            await asyncio.to_thread(gone_and_new_head)

        env.proxy.add("GET", _exact(full), Before(before_body_read), times=None)
        mark = len(env.proxy.log)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await env.load(iid, manager=env.manager)
        assert str(exc.value).startswith(f"Instance {iid}: the head kept changing while it was loaded.")
        assert (exc.value.action, exc.value.instance_id) == ("RETRY", iid)
        assert (env.proxy.count("GET", _exact(head), since=mark), env.proxy.count("GET", _exact(full), since=mark)) == (
            3,
            3,
        )

    async def test_object_replaced_during_load_is_retry_then_damaged(self, env: InstanceEnv) -> None:
        """An object larger than one part, replaced between the HEAD and the ranged GETs: the
        pinned ETag fails (412), which is a broken store (RETRY), never "gone"; the next load
        reads the new bytes and finds them damaged."""
        rng = random.Random(141)
        ram = SparseImage(size=160 * BLOCK)
        for block_no in range(140):  # 8.75 MiB of incompressible blocks: a full copy of two parts
            ram.put(block_no, random_block(rng))
        saver = Saver(env, await env.new_vm(ram, SparseImage(size=BLOCK), spec=PYTHON), manager=env.manager, state=None)
        (full,) = chain_keys(await saver.save())
        assert env.bucket.size(full) > _PART

        async def replace(req: Request) -> None:
            await asyncio.to_thread(env.bucket.flip, full, 100)  # same size, another ETag

        env.proxy.add("GET_RANGE", _exact(full), Before(replace))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await env.load(saver.instance_id, manager=env.manager)
        assert (exc.value.action, exc.value.instance_id) == ("RETRY", saver.instance_id)
        assert 412 in {e.status for e in env.proxy.entries("GET_RANGE", _exact(full))}
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
            await env.load(saver.instance_id, manager=env.manager)

    async def test_one_wrong_size_answer_is_retry_then_loads(self, env: InstanceEnv) -> None:
        saver = await _chain(env, packs=1, seed=142, manager=env.manager)
        pack = chain_keys(saver.state)[1]  # type: ignore[arg-type]

        def one_more(resp: Response) -> None:
            resp.body += b"\0"  # its Content-Length says one byte more

        rule = env.proxy.add("GET", _exact(pack), Edit(response=one_more))
        mark = len(env.proxy.log)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await env.load(saver.instance_id, manager=env.manager)
        assert (exc.value.action, exc.value.instance_id) == ("RETRY", saver.instance_id)
        asked = (env.proxy.count("GET", _exact(pack), since=mark), env.proxy.count("HEAD", _exact(pack), since=mark))
        assert (rule.hits, *asked) == (1, 1, 1)
        await env.check_load(saver.instance_id, generation=2, manager=env.manager, want=None)


# ----------------------------------------------------------------------------- load: objects ahead


class TestDownloadAhead:
    async def test_downloads_run_ahead_within_the_window(self, env: InstanceEnv) -> None:
        """While the full copy's GET waits, the load fetches the next 7 objects: 8 at a time, so
        at most 8 object files on local disk; the rest follow as objects are applied. One GET
        per object, no HEAD, and the load is byte-exact."""
        saver = await _chain(env, packs=10, seed=15, manager=env.manager)
        keys = chain_keys(saver.state)  # type: ignore[arg-type]
        await env.manager.drain()  # the last save's old files go in a worker thread
        files = env.work_files()
        held = env.proxy.add("GET", _exact(keys[0]), Hold())
        mark = len(env.proxy.log)
        loading = asyncio.create_task(env.check_load(saver.instance_id, generation=11, manager=env.manager, want=None))
        try:
            await held.reached.wait()
            await await_until(lambda: env.proxy.count("GET", "/b/", since=mark) >= 7)
            assert env.proxy.count("GET", "/b/", since=mark) == 7
            assert {e.key for e in env.proxy.entries("GET", "/b/", since=mark)} == set(keys[1:8])
            assert len([path for path in env.work_files() - files if path.suffix in {".f", ".p"}]) == 8
        finally:
            held.release()
        await loading
        assert env.proxy.count("GET", "/b/", since=mark) == len(keys)
        assert env.proxy.count(("HEAD", "GET_RANGE"), "/b/", since=mark) == 0
        assert env.work_files() == files

    async def test_first_missing_object_in_chain_order_decides(self, env: InstanceEnv) -> None:
        """Two objects gone; the later one's 404 comes first (the earlier GET is held). The load
        still follows chain order, as a load of one object at a time would: it asks the store
        about the earlier object (HEAD), then says "expired"."""
        saver = await _chain(env, packs=3, seed=16, manager=env.manager)
        keys = chain_keys(saver.state)  # type: ignore[arg-type]
        env.bucket.delete(keys[1])
        env.bucket.delete(keys[3])
        held = env.proxy.add("GET", _exact(keys[1]), Hold())
        mark = len(env.proxy.log)
        loading = asyncio.create_task(env.load(saver.instance_id, manager=env.manager))
        try:
            await held.reached.wait()
            await await_until(lambda: env.proxy.count("GET", _exact(keys[3]), since=mark) == 1)
        finally:
            held.release()
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_EXPIRED) as exc:
            await loading
        assert exc.value.instance_id == saver.instance_id
        assert env.proxy.count("HEAD", _exact(keys[1]), since=mark) == 1
        assert env.proxy.count("HEAD", _exact(keys[3]), since=mark) == 0

    async def test_cancelled_load_leaves_no_file_and_no_fd(self, env: InstanceEnv) -> None:
        """Cancelled while a pack's GET waits: the other downloads stop, no work file stays, and
        the descriptors of the stream files (opened once per load) are closed."""
        saver = await _chain(env, packs=3, seed=17, manager=env.manager)
        keys = chain_keys(saver.state)  # type: ignore[arg-type]
        await env.manager.drain()  # the last save's old files go in a worker thread
        files = env.work_files()
        held = env.proxy.add("GET", _exact(keys[2]), Hold())
        loading = asyncio.create_task(env.load(saver.instance_id, manager=env.manager))
        try:
            await held.reached.wait()
            (vmstate,) = [path for path in env.work_files() - files if path.suffix == ".vmstate"]
            ((fd, inode),) = descriptors_of(vmstate)
            loading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await loading
        finally:
            held.remove()

        def still_open() -> bool:
            try:
                return os.fstat(fd).st_ino == inode  # the number may be reused by another file since
            except OSError:
                return False

        await await_until(lambda: not still_open())
        await env.manager.drain()  # the cancelled load's folder goes in a worker thread
        assert env.work_files() == files


# ----------------------------------------------------------------------------- lineage at save


_REPAIRABLE = ["delete", "flip-record", "cut-1", "other-commit-same-size", "other-instance-same-size"]


class TestLineage:
    @pytest.mark.parametrize("origin", ["loaded", "own-save"])
    @pytest.mark.parametrize(
        ("target", "damage"),
        [
            pytest.param(t, d, id=f"{t}-{d}")
            for t in _POSITION
            for d in _REPAIRABLE
            if not (t == "full" and d == "other-commit-same-size")
        ],
    )
    async def test_damage_between_load_and_save_repairs(
        self, env: InstanceEnv, target: str, damage: str, origin: str
    ) -> None:
        """One LIST before the pack finds the changed object: the save writes one full copy under a
        new data key instead (no pack upload), the old chain goes, and the new chain alone loads."""
        saver = await _chain(env, packs=3, seed=15, manager=env.manager)
        session = await _reuse(env, saver.instance_id, manager=env.manager) if origin == "loaded" else saver
        before = session.state
        assert before is not None
        await _damage(env, saver, target=target, damage=damage)
        _small_change(session.vm, 99)
        mark = len(env.proxy.log)
        state = await session.save()
        assert kinds(state) == ["full"]
        assert state.data_key != before.data_key
        assert env.proxy.count("LIST", since=mark) == 1
        assert env.proxy.count(("PUT", "MPU_CREATE", "MPU_PART"), r"\.p$", since=mark) == 0
        assert env.bucket.body_keys(state.instance_id) == chain_keys(state)
        await env.check_load(state.instance_id, generation=5, manager=env.manager, want=None)
        _small_change(session.vm, 100)
        assert kinds(await session.save()) == ["full", "pack"]
        await env.check_load(state.instance_id, generation=6, manager=env.manager, want=None)

    @pytest.mark.parametrize("damage", ["delete", "flip-record"])
    @pytest.mark.parametrize("target", ["full", "first"])
    async def test_damage_after_list_fails_closed_then_repairs(
        self, env: InstanceEnv, target: str, damage: str
    ) -> None:
        """Damage right after the LIST: that save commits a pack on a broken chain. The next load
        fails closed (never wrong bytes); the same session's next save sees the damage and repairs."""
        saver = await _chain(env, packs=1, seed=16, manager=env.manager)
        key = chain_keys(saver.state)[_POSITION[target]]  # type: ignore[arg-type]

        def damage_now() -> None:
            if damage == "delete":
                env.bucket.delete(key)
            else:
                env.bucket.flip(key, env.bucket.index_offset(key) // 2)

        async def after_list(req: Request, resp: Response) -> None:
            await asyncio.to_thread(damage_now)

        env.proxy.add("LIST", "", After(after_list))
        _small_change(saver.vm, 30)
        assert kinds(await saver.save()) == ["full", "pack", "pack"]
        expected = ErrorCode.INSTANCE_EXPIRED if damage == "delete" else ErrorCode.INSTANCE_INTEGRITY
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is expected) as exc:
            await env.load(saver.instance_id, manager=env.manager)
        assert exc.value.action == "RECREATE"
        _small_change(saver.vm, 31)
        state = await saver.save()
        assert kinds(state) == ["full"]
        assert env.bucket.body_keys(state.instance_id) == chain_keys(state)
        await env.check_load(state.instance_id, generation=4, manager=env.manager, want=None)

    @pytest.mark.parametrize("edit", ["drop-entry", "other-size", "other-etag", "no-etag"])
    async def test_lineage_false_alarm_costs_one_full_copy(self, env: InstanceEnv, edit: str) -> None:
        """A LIST that misreports an intact chain object costs one full copy, never a broken chain."""
        saver = await _chain(env, packs=1, seed=17, manager=env.manager)
        listed_key = env.proxy.prefix + chain_keys(saver.state)[1]  # type: ignore[arg-type]

        def misreport(block: str) -> str:
            if f"<Key>{listed_key}</Key>" not in block:
                return block
            match edit:
                case "drop-entry":
                    return ""
                case "other-size":
                    return re.sub(r"<Size>(\d+)</Size>", lambda m: f"<Size>{int(m[1]) + 1}</Size>", block)
                case "other-etag":
                    return re.sub(r"<ETag>.*?</ETag>", "<ETag>&#34;" + "0" * 32 + "&#34;</ETag>", block)
                case _:
                    return re.sub(r"<ETag>.*?</ETag>", "", block)

        def lie(resp: Response) -> None:
            text = resp.body.decode()
            edited = re.sub(r"<Contents>.*?</Contents>", lambda m: misreport(m[0]), text, flags=re.S)
            assert edited != text
            resp.body = edited.encode()

        rule = env.proxy.add("LIST", "", Edit(response=lie))
        _small_change(saver.vm, 40)
        state = await saver.save()
        assert rule.hits == 1
        assert kinds(state) == ["full"]
        assert env.bucket.body_keys(state.instance_id) == chain_keys(state)
        await env.check_load(state.instance_id, generation=3, manager=env.manager, want=None)

    async def test_unquoted_put_etag_is_no_false_alarm(self, env: InstanceEnv) -> None:
        def unquote(resp: Response) -> None:
            resp.headers["ETag"] = resp.headers["ETag"].strip('"')

        rule = env.proxy.add("PUT", "/b/", Edit(response=unquote), times=None)
        saver = await _chain(env, packs=2, seed=181, manager=env.manager)
        assert rule.hits == 3
        await env.check_load(saver.instance_id, generation=3, manager=env.manager, want=None)

    async def test_multipart_etag_is_no_false_alarm(self, env: InstanceEnv) -> None:
        """A full copy above the part size is a multipart upload ("<md5>-<parts>" ETag); LIST reports the same."""
        rng = random.Random(182)
        ram, home = SparseImage(size=136 * BLOCK), SparseImage(size=2 * BLOCK)
        for block_no in range(136):
            ram.put(block_no, random_block(rng))
        saver = Saver(env, await env.new_vm(ram, home, spec=PYTHON), manager=env.manager, state=None)
        first = await saver.save()
        assert env.proxy.count("MPU_COMPLETE", r"\.f$") == 1
        assert "-" in first.body.chain[0].etag
        _small_change(saver.vm, 2)
        assert kinds(await saver.save()) == ["full", "pack"]
        await env.check_load(first.instance_id, generation=2, manager=env.manager, want=None)

    async def test_paginated_list_is_no_false_alarm(self, env: InstanceEnv) -> None:
        saver = await _chain(env, packs=1, seed=183, manager=env.manager)
        for _ in range(2):  # more objects under b/: four pages of one
            env.bucket.write(f"s/{saver.instance_id}/b/{ids.new_id()}.p", b"orphan")

        def one_per_page(req: Request) -> None:
            req.query["max-keys"] = "1"

        env.proxy.add("LIST", "", Edit(request=one_per_page), times=None)
        mark = len(env.proxy.log)
        _small_change(saver.vm, 3)
        assert kinds(await saver.save()) == ["full", "pack", "pack"]
        assert env.proxy.count("LIST", since=mark) == 4
        await env.check_load(saver.instance_id, generation=3, manager=env.manager, want=None)

    async def test_same_bytes_put_again_is_no_false_alarm(self, env: InstanceEnv) -> None:
        saver = await _chain(env, packs=1, seed=184, manager=env.manager)
        pack = chain_keys(saver.state)[1]  # type: ignore[arg-type]
        etag = env.bucket.etag(pack)
        env.bucket.write(pack, env.bucket.read(pack))
        assert env.bucket.etag(pack) == etag
        _small_change(saver.vm, 3)
        assert kinds(await saver.save()) == ["full", "pack", "pack"]


# ----------------------------------------------------------------------------- store answers at save


class TestStoreAnswersAtSave:
    """Through a real Session over the fake VM: which failures keep the session open."""

    async def _session(self, env: InstanceEnv, seed: int) -> tuple[FakeVm, Session]:
        ram, home = _layout(seed)
        vm = await env.new_vm(ram, home, spec=PYTHON)
        session = fake_session(
            vm=vm, vm_manager=env.vm_manager, instances=env.manager, state=None, new_instance_id=None
        )
        assert (await session.save()).generation == 1
        return vm, session

    @pytest.mark.parametrize(
        ("rule", "error", "action"),
        [
            pytest.param(
                Fail(503, "ServiceUnavailable"),
                (TransientError, ErrorCode.INSTANCE_STORE_UNAVAILABLE),
                "RETRY",
                id="list-503",
            ),
            pytest.param(
                Fail(403, "AccessDenied"), (ConfigError, ErrorCode.INSTANCE_CONFIG), "FIX_CONFIG", id="list-403"
            ),
            pytest.param(
                "ignores-prefix",
                (TransientError, ErrorCode.INSTANCE_STORE_UNAVAILABLE),
                "RETRY",
                id="list-ignores-prefix",
            ),
        ],
    )
    async def test_list_failures(
        self, env: InstanceEnv, rule: object, error: tuple[type[SandboxError], ErrorCode], action: str
    ) -> None:
        """No LIST answer, no permission, or keys outside the asked prefix: nothing uploaded, the
        session stays open, and the next save is a pack."""
        vm, session = await self._session(env, seed=19)
        if rule == "ignores-prefix":

            def whole_bucket(req: Request) -> None:
                req.query["prefix"] = env.proxy.prefix

            env.proxy.add("LIST", "", Edit(request=whole_bucket), times=None)
        else:
            env.proxy.add("LIST", "", rule, times=None)  # type: ignore[arg-type]
        _small_change(vm, 5)
        mark = len(env.proxy.log)
        with pytest.raises(error[0], check=lambda e: e.code is error[1]) as exc:
            await session.save()
        assert exc.value.action == action
        assert env.proxy.count(("PUT", "MPU_CREATE", "PUT_IF"), "^s/", since=mark) == 0
        assert not session.closed
        env.proxy.clear()
        want = vm.expected()
        ref = await session.save()
        assert ref.generation == 2
        state = await env.check_load(ref.instance_id, generation=2, want=want, manager=env.manager)
        assert kinds(state) == ["full", "pack"]
        await session.close()

    @pytest.mark.parametrize("etag", ['""', None, '"' + "a" * 2048 + '"'], ids=["empty", "missing", "2-KiB"])
    async def test_hostile_put_etag_is_retry(self, env: InstanceEnv, etag: str | None) -> None:
        """A body PUT answer without a usable ETag: RETRY, the session stays open, the head is unchanged."""
        vm, session = await self._session(env, seed=20)

        def hostile(resp: Response) -> None:
            if etag is None:
                resp.headers.popall("ETag")
            else:
                resp.headers["ETag"] = etag

        env.proxy.add("PUT", r"/b/.*\.p$", Edit(response=hostile))
        _small_change(vm, 6)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await session.save()
        assert exc.value.action == "RETRY"
        assert not session.closed
        assert env.proxy.count("PUT_IF") == 1  # the first save's head only
        want = vm.expected()
        ref = await session.save()
        assert ref.generation == 2
        await env.check_load(ref.instance_id, generation=2, want=want, manager=env.manager)
        await session.close()


class TestSupersededSave:
    async def test_next_save_after_a_superseded_commit_conflicts(self, env: InstanceEnv) -> None:
        """A's head write lands but its answer is lost, and B commits on top before A reads the
        head back: A's save is stored (generation 2), but A has no base left. A's next save is
        refused before any upload (VERIFY), A stays open, and B's head stays."""
        first = await _chain(env, packs=0, seed=191, manager=env.manager)
        iid = first.instance_id
        a_state = await env.load(iid, manager=env.manager)
        a_vm = await env.vm_from(a_state)
        a = fake_session(vm=a_vm, vm_manager=env.vm_manager, instances=env.manager, state=a_state, new_instance_id=None)
        head = _exact(head_object(iid))

        async def b_commits_on_top(req: Request, resp: Response) -> None:
            assert resp.status == 200  # A's head landed
            b = await _reuse(env, iid, manager=env.manager)
            _small_change(b.vm, 9)
            assert (await b.save()).header.generation == 3

        # Both rules are taken by A's write alone (times=1): B's write inside the hook passes.
        env.proxy.add("PUT_IF", head, After(b_commits_on_top))
        env.proxy.add("PUT_IF", head, DropAfter())
        _small_change(a_vm, 10)
        assert (await a.save()).generation == 2
        b_head = env.bucket.read(head_object(iid))
        await env.manager.drain()
        assert not a_state.folder.exists()  # a plain save keeps its folder: superseded at once, it goes

        _small_change(a_vm, 11)
        mark = len(env.proxy.log)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT) as exc:
            await a.save()
        assert str(exc.value).startswith(
            f"Instance {iid}: another save committed after this session's last save. This save was not stored."
        )
        assert (exc.value.action, exc.value.instance_id) == ("VERIFY", iid)
        assert exc.value.context is not None
        assert exc.value.context["generation"] == 3  # the generation this save would have made
        assert ids.is_commit_id(exc.value.context["commit_id"])
        assert not a.closed
        assert env.proxy.count(("PUT", "MPU_CREATE", "PUT_IF"), since=mark) == 0
        assert env.bucket.read(head_object(iid)) == b_head
        await a.close()

    async def test_a_save_under_a_new_id_superseded_at_once_releases_its_files(self, env: InstanceEnv) -> None:
        """save(instance_id=Y): Y's head lands but its answer is lost, and B commits Y@2 on top
        before A reads the head back. A moves to Y (Y@1, with no base left), and A's local files of
        Y and of X go at once (no later save reads them)."""
        first = await _chain(env, packs=0, seed=193, manager=env.manager)
        a_state = await env.load(first.instance_id, manager=env.manager)
        a_vm = await env.vm_from(a_state)
        a = fake_session(vm=a_vm, vm_manager=env.vm_manager, instances=env.manager, state=a_state, new_instance_id=None)
        head = _exact(head_object("y-193"))
        others: list[Saver] = []

        async def b_commits_on_top(req: Request, resp: Response) -> None:
            assert resp.status == 200  # A's head of Y landed
            b = await _reuse(env, "y-193", manager=env.manager)
            others.append(b)
            _small_change(b.vm, 9)
            assert (await b.save()).header.generation == 2

        env.proxy.add("PUT_IF", head, After(b_commits_on_top))
        env.proxy.add("PUT_IF", head, DropAfter())
        _small_change(a_vm, 10)
        assert (await a.save(instance_id="y-193")).generation == 1
        assert a.instance == InstanceRef(instance_id="y-193", generation=1)
        (b,) = others
        for saver in (b, first):  # the files that the other session and the chain's own saver keep
            assert saver.state is not None
            await env.manager.release(saver.state, wait=True)
        await env.manager.drain()
        assert env.work_files() == set()  # A's files of Y and of X are gone
        _small_change(a_vm, 11)
        mark = len(env.proxy.log)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT):
            await a.save()  # no base left: B's Y@2 is on top of A's Y@1
        assert env.proxy.count(("PUT", "MPU_CREATE", "PUT_IF"), since=mark) == 0
        await a.close()


class TestErrorsNameTheInstance:
    """Store errors do not know which instance they served: save, load, describe and delete name it."""

    async def test_first_save_names_its_new_random_id(self, env: InstanceEnv) -> None:
        ram, home = _layout(41)
        saver = Saver(env, await env.new_vm(ram, home, spec=PYTHON), manager=env.manager, state=None)
        for op in ("PUT", "MPU_CREATE"):
            env.proxy.add(op, "^s/", Fail(503, "ServiceUnavailable"), times=None)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await saver.save()
        (refused_key,) = {e.key for e in env.proxy.entries(("PUT", "MPU_CREATE"), "^s/")}
        instance_id = refused_key.split("/")[1]  # s/<instance>/b/<commit>.f
        assert re.fullmatch(r"[0-9a-f]{32}", instance_id)
        assert exc.value.instance_id == instance_id

    @pytest.mark.parametrize(("call", "op"), [("describe", "GET"), ("delete", "HEAD")])
    async def test_admin_calls_name_the_id(self, env: InstanceEnv, call: str, op: str) -> None:
        saver = await _chain(env, packs=0, seed=43, manager=env.manager)
        env.proxy.add(op, "/head$", Fail(503, "ServiceUnavailable"), times=None)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await getattr(env.manager, call)(saver.instance_id)
        assert exc.value.instance_id == saver.instance_id


class TestLoadLocalFiles:
    """Every local-file failure of a load is TransientError LOCAL_DISK (RETRY): nothing was loaded, no
    partial file stays, and the same load succeeds once the cause is gone."""

    @pytest.mark.parametrize("step", ["asset", "body", "folder"])
    async def test_local_failure_at_load_is_retry(
        self, env: InstanceEnv, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, step: str
    ) -> None:
        """A worker without the base image. "asset": its disk is full while it downloads the
        image. "body": its disk is full while it downloads a chain object, after the image is in
        its cache. "folder": its work folder is gone (shutdown) after the image is in its cache."""
        saver = await _chain(env, packs=1, seed=53, manager=env.manager)
        assert saver.state is not None
        worker = worker_without_base_image(env, root=tmp_path / "worker-b")
        cached = worker.asset_dir / f"{saver.state.body.guest.base_image}.img"
        match step:
            case "asset":
                disk_full_at_download(monkeypatch, env.store, under=worker.asset_dir)
            case "body":
                disk_full_at_download(monkeypatch, env.store, under=worker.work_dir)
            case _:
                worker.work_dir.rmdir()
        with pytest.raises(
            TransientError, match="loading needs local disk space", check=lambda e: e.code is ErrorCode.LOCAL_DISK
        ) as exc:
            await env.load(saver.instance_id, manager=worker)
        failed = errno.ENOENT if step == "folder" else errno.ENOSPC
        assert exc.value.context == {"instance_id": saver.instance_id, "errno": failed}
        await worker.drain()  # the partial files go in a worker thread
        # The partial files are removed; an image whose digest was checked stays in the cache
        # (the lock files of its entry stay: files.locks never deletes them).
        assert {path for path in worker.asset_dir.iterdir() if path.suffix != ".lock"} == (
            set() if step == "asset" else {cached}
        )
        if step == "folder":
            worker.work_dir.mkdir(mode=0o700)
        assert list(worker.work_dir.iterdir()) == []

        mark = len(env.proxy.log)
        await env.check_load(saver.instance_id, generation=2, manager=worker, want=None)
        assert {path for path in worker.asset_dir.iterdir() if path.suffix != ".lock"} == {cached}
        assert bool(env.proxy.count("GET_RANGE", "^assets/", since=mark)) == (step == "asset")
        assert list(worker.work_dir.iterdir()) == []
        # No load holds the image any more (the failed one released its lease): eviction takes it
        assert (await worker.assets.eviction_order().evict_next() or 0) > 0
        assert not cached.exists()

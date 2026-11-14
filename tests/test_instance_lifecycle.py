"""Session lifecycle of saved instances: generations, ordering, close, cancel, drain, disk full, forks.

M level: moto behind the fault proxy (tests/s3_faults.py), a fake VM and QMP
peer (tests/instance_support.py), and the real Session, InstanceManager, commit,
pack and store code; no QEMU. The fake guest runs a tiny op language: each
exec() changes one block (or the size) of the RAM or home-disk stream. Each
save records the two streams the guest had at the stopped instant, and a load
must give back exactly those bytes.

Order comes only from gates (proxy holds, the QMP migration hold) and the
session's FIFO lock; bounded polls (await_until) only wait for effects that
end in the background.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import random
import re
import shutil
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from exec_sandbox.exceptions import (
    ErrorCode,
    InstanceGoneError,
    InstanceNotSavedWarning,
    InvalidInputError,
    OutcomeUnknownError,
    SessionClosedError,
    TransientError,
)
from exec_sandbox.instance.store import head_key
from exec_sandbox.models import InstanceRef, StartupTiming
from exec_sandbox.session import Session
from tests.instance_support import (
    BLOCK,
    NO_VM_TIMEOUT,
    PYTHON,
    FakeVm,
    InstanceEnv,
    SparseImage,
    await_until,
    chain_keys,
    constant_block,
    digest,
    disk_full_once,
    fake_session,
    file_digest,
    head_object,
    new_keyring,
    random_block,
)
from tests.s3_faults import After, Before, Fail, Hold, Request, Response

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from exec_sandbox.instance.manager import InstanceState

pytestmark = NO_VM_TIMEOUT

type Streams = tuple[bytes, bytes]
"""(vmstate, home) bytes."""

_BLOCKS = 8
"""Blocks per stream at the start (RAM and home disk)."""
_WAIT = 10.0
"""Bound for a gate to be reached: correct code reaches it in milliseconds."""
_TIMING = StartupTiming.model_validate(dict.fromkeys(StartupTiming.model_fields, 0))
"""How a session started, all zeros: the idle timeout test reads none of it."""
_HYPOTHESIS = settings(
    max_examples=25,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


def _exact(key: str) -> str:
    return f"^{re.escape(key)}$"


def _body(instance_id: str) -> str:
    return f"^{re.escape(f's/{instance_id}/b/')}"


async def _within[T](aw: Awaitable[T]) -> T:
    return await asyncio.wait_for(aw, _WAIT)


# ----------------------------------------------------------------------------- the fake guest


def _op(op: str, stream: str, *, block: int, value: int, tag: int | None) -> str:
    """Code for the fake guest: one change to one stream.

    put: block <- 64 KiB of byte `value`; rand: block <- 64 KiB from Random(value);
    discard: block becomes a hole; resize: the stream is `value` bytes.
    """
    return json.dumps({"op": op, "s": stream, "b": block, "v": value, "tag": tag})


def _apply(vm: FakeVm, op: dict[str, Any]) -> None:
    image = vm.ram if op["s"] == "ram" else vm.home
    match op["op"]:
        case "put":
            image.put(op["b"], constant_block(op["v"]))
        case "rand":
            image.put(op["b"], random_block(random.Random(op["v"])))
        case "discard":
            image.discard(op["b"])
        case "resize":
            image.resize(op["v"])
        case other:
            raise AssertionError(other)


def _change(vm: FakeVm, code: str) -> None:
    """Change the guest without exec(): no InstanceNotSavedWarning at close."""
    _apply(vm, json.loads(code))


def _random_op(vm: FakeVm, rng: random.Random) -> str:
    """One valid change for the guest as it is now: mostly small constant blocks, sometimes an
    incompressible block (packs then outgrow the full copy: rollover), holes and resizes."""
    stream = rng.choice(("ram", "home"))
    image = vm.ram if stream == "ram" else vm.home
    kind = rng.choices(("put", "rand", "discard", "resize"), weights=(10, 2, 3, 2))[0]
    if kind == "discard" and image.blocks:
        return _op("discard", stream, block=rng.choice(sorted(image.blocks)), value=0, tag=None)
    if kind == "resize":
        tail = rng.choice((0, rng.randrange(1, BLOCK)))
        return _op("resize", stream, block=0, value=rng.randint(1, _BLOCKS) * BLOCK - tail, tag=None)
    block = rng.randrange(-(-image.size // BLOCK))
    if kind == "rand":
        return _op("rand", stream, block=block, value=rng.getrandbits(32), tag=None)
    return _op("put", stream, block=block, value=rng.randint(1, 255), tag=None)


def _layout(seed: int) -> tuple[SparseImage, SparseImage]:
    """Four incompressible blocks: the full copy is large enough for many small packs."""
    rng = random.Random(seed)
    ram, home = SparseImage(size=_BLOCKS * BLOCK), SparseImage(size=_BLOCKS * BLOCK)
    ram.put(0, random_block(rng))
    ram.put(2, random_block(rng))
    home.put(0, random_block(rng))
    home.put(3, random_block(rng))
    return ram, home


class Guest:
    """A FakeVm whose execute() runs one `_op` on its streams, and whose saves record the bytes they captured."""

    def __init__(self, vm: FakeVm, log: list[tuple[object, ...]]) -> None:
        self.vm = vm
        self.log = log
        self.snapshots: list[Streams] = []
        """Streams at each ``migrate`` of this VM, in order."""
        self.yields = 0
        """Loop turns an exec() gives away before and after its change (a window for an overlap to show)."""
        vm.on_execute = self._execute
        migrate = vm.qmp.on_migrate
        assert migrate is not None

        def on_migrate(path: Path) -> None:  # runs in a thread while the VM is stopped
            self.snapshots.append(vm.expected())
            self.log.append(("migrate", vm.vm_id))
            migrate(path)

        vm.qmp.on_migrate = on_migrate

    async def _execute(self, code: str) -> str:
        op = json.loads(code)
        self.log.append(("exec-start", op["tag"]))
        for _ in range(self.yields):
            await asyncio.sleep(0)
        _apply(self.vm, op)
        for _ in range(self.yields):
            await asyncio.sleep(0)
        self.log.append(("exec-end", op["tag"]))
        return ""


@dataclass(slots=True)
class Fork:
    """One session over a fake VM: what Scheduler.new() or reuse() returns."""

    session: Session
    guest: Guest
    state: InstanceState | None
    """The state reuse() loaded (its folder is the session's), None for new()."""

    @property
    def vm(self) -> FakeVm:
        return self.guest.vm


class Lab:
    """Sessions on one InstanceEnv, and the bytes each committed generation saved."""

    def __init__(self, env: InstanceEnv) -> None:
        self.env = env
        self.want: dict[tuple[str, int], Streams] = {}
        self.log: list[tuple[object, ...]] = []

    async def new(self, seed: int, *, instance_id: str | None) -> Fork:
        """A new session; its first save gives the instance `instance_id` (None: a random id)."""
        ram, home = _layout(seed)
        return self._fork(await self.env.new_vm(ram, home, spec=PYTHON), state=None, new_instance_id=instance_id)

    async def reuse(self, instance_id: str) -> Fork:
        """load() and a VM restored from the loaded files, which must be the bytes of the generation loaded."""
        state = await self.env.load(instance_id, manager=self.env.manager)
        try:
            want = self.want[(instance_id, state.header.generation)]
            assert (file_digest(state.vmstate), file_digest(state.home)) == (digest(want[0]), digest(want[1]))
            vm = await self.env.vm_from(state)
        except BaseException:
            await self.env.manager.release(state)
            raise
        return self._fork(vm, state=state, new_instance_id=None)

    def _fork(self, vm: FakeVm, *, state: InstanceState | None, new_instance_id: str | None) -> Fork:
        session = fake_session(
            vm=vm,
            vm_manager=self.env.vm_manager,
            instances=self.env.manager,
            state=state,
            new_instance_id=new_instance_id,
        )
        return Fork(session=session, guest=Guest(vm, self.log), state=state)

    async def save(self, fork: Fork) -> InstanceRef:
        """session.save(); records the streams of its snapshot under the generation it committed."""
        ref = await fork.session.save()
        self.want[(ref.instance_id, ref.generation)] = fork.guest.snapshots[-1]
        return ref

    async def base(self, seed: int) -> str:
        """A new instance at generation 1 (its session closed)."""
        fork = await self.new(seed, instance_id=None)
        ref = await self.save(fork)
        await fork.session.close()
        return ref.instance_id

    async def check(self, instance_id: str, generation: int) -> InstanceState:
        """A load by another worker (same keyring) gives back the bytes this generation saved."""
        return await self.env.check_load(
            instance_id,
            generation=generation,
            manager=self.env.manager_for(self.env.keyring),
            want=self.want[(instance_id, generation)],
        )

    async def generation(self, instance_id: str) -> int:
        return (await self.env.manager.describe(instance_id)).generation


@pytest.fixture
def lab(instance_env: InstanceEnv) -> Lab:
    return Lab(instance_env)


async def _close_quietly(forks: list[Fork]) -> None:
    """Close sessions whose unsaved changes do not matter to the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InstanceNotSavedWarning)
        for fork in forks:
            await fork.session.close()


# ----------------------------------------------------------------------------- generations


class TestGenerations:
    @pytest.mark.timeout(60)  # 3 examples of 100 generations: about 300 saves, loads and reads of the head
    @settings(_HYPOTHESIS, max_examples=3)
    @given(seed=st.integers(min_value=0, max_value=2**32 - 1))
    async def test_generations_fake(self, lab: Lab, seed: int) -> None:
        """100 generations of seeded changes (blocks, holes, resizes, incompressible blocks that force
        rollovers). After each save the stored generation is the session's; at seeded points an
        unsaved change is dropped (close warns) and reuse() gives back exactly the last save."""
        env = lab.env
        rng = random.Random(seed)
        fork = await lab.new(seed, instance_id=None)
        instance_id = ""
        reuses = 0
        for generation in range(1, 101):
            for _ in range(rng.randint(0, 3)):  # 0: a save with nothing changed
                await fork.session.exec(_random_op(fork.vm, rng))
            ref = await lab.save(fork)
            instance_id = ref.instance_id
            assert ref.generation == generation
            assert fork.session.instance == ref
            assert await lab.generation(instance_id) == generation
            if generation < 100 and rng.random() < 0.15:
                reuses += 1
                lost = _op("resize", "home", block=0, value=fork.vm.home.size + BLOCK, tag=None)  # never what was saved
                await fork.session.exec(lost)
                assert fork.vm.expected() != lab.want[(instance_id, generation)]
                with pytest.warns(InstanceNotSavedWarning):
                    await fork.session.close()
                fork = await lab.reuse(instance_id)  # checks the bytes of the last save
                assert fork.session.instance == ref
        assert reuses > 0
        await fork.session.close()
        head = await lab.check(instance_id, 100)
        assert env.bucket.body_keys(instance_id) == sorted(chain_keys(head))
        assert env.work_files() == set()


# ----------------------------------------------------------------------------- one session, FIFO


class TestOneSession:
    async def test_fifo_exec_save(self, lab: Lab) -> None:
        """gather(e1, s1, e2, s2, e3): the lock runs them one at a time in call order; s1 holds e1
        only, s2 e1 and e2; close() warns about e3."""
        env = lab.env
        fork = await lab.new(seed=5, instance_id=None)
        fork.guest.yields = 10
        ram0 = fork.vm.ram.copy()
        home = fork.vm.home.content()

        async def head_written(req: Request, resp: Response) -> None:
            lab.log.append(("head", resp.status))

        env.proxy.add("PUT_IF", "/head$", After(head_written), times=None)
        e1, s1, e2, s2, e3 = await asyncio.gather(
            fork.session.exec(_op("put", "ram", block=1, value=11, tag=1)),
            lab.save(fork),
            fork.session.exec(_op("put", "ram", block=2, value=12, tag=2)),
            lab.save(fork),
            fork.session.exec(_op("put", "ram", block=3, value=13, tag=3)),
        )
        vm_id = fork.vm.vm_id
        assert lab.log == [
            ("exec-start", 1),
            ("exec-end", 1),
            ("migrate", vm_id),
            ("head", 200),
            ("exec-start", 2),
            ("exec-end", 2),
            ("migrate", vm_id),
            ("head", 200),
            ("exec-start", 3),
            ("exec-end", 3),
        ]
        assert [r.exit_code for r in (e1, e2, e3)] == [0, 0, 0]
        assert (s1.generation, s2.generation, s2.instance_id) == (1, 2, s1.instance_id)
        assert fork.session.exec_count == 3

        def model(execs: int) -> Streams:
            ram = ram0.copy()
            for n in range(1, execs + 1):
                ram.put(n, constant_block(10 + n))
            return ram.content(), home

        assert lab.want[(s1.instance_id, 1)] == model(1)
        assert lab.want[(s1.instance_id, 2)] == model(2)
        assert fork.vm.expected() == model(3)
        with pytest.warns(InstanceNotSavedWarning):
            await fork.session.close()
        await lab.check(s1.instance_id, 2)  # a load reads the newest head: generation 1 is checked by its snapshot
        assert await lab.generation(s1.instance_id) == 2
        assert env.work_files() == set()

    async def test_exec_then_save_closes_without_warning(self, lab: Lab) -> None:
        fork = await lab.new(seed=51, instance_id=None)
        _, ref = await asyncio.gather(fork.session.exec(_op("put", "home", block=1, value=51, tag=1)), lab.save(fork))
        assert ref.generation == 1
        assert lab.want[(ref.instance_id, 1)] == fork.vm.expected()  # the save holds the exec before it
        with warnings.catch_warnings():
            warnings.simplefilter("error", InstanceNotSavedWarning)
            await fork.session.close()
        await lab.check(ref.instance_id, 1)

    async def test_concurrent_saves_one_session(self, lab: Lab) -> None:
        """gather(s, s, s) on one session: three commits in a row, g+1..g+3, none refused."""
        env = lab.env
        instance_id = await lab.base(seed=6)
        fork = await lab.reuse(instance_id)
        _change(fork.vm, _op("put", "ram", block=1, value=6, tag=None))
        mark = len(env.proxy.log)
        refs = await asyncio.gather(lab.save(fork), lab.save(fork), lab.save(fork))
        assert [(r.instance_id, r.generation) for r in refs] == [(instance_id, g) for g in (2, 3, 4)]
        assert fork.session.instance == refs[-1]
        assert [e.status for e in env.proxy.entries("PUT_IF", since=mark)] == [200, 200, 200]
        assert len(fork.guest.snapshots) == 3
        await fork.session.close()
        head = await lab.check(instance_id, 4)
        assert env.bucket.body_keys(instance_id) == sorted(chain_keys(head))
        assert env.work_files() == set()


# ----------------------------------------------------------------------------- close, cancel, drain


_PHASES = ["migrate", "body-put", "head-put"]


@dataclass(frozen=True, slots=True)
class Gate:
    """A save held at one phase: the migration (QMP query-migrate) or a store PUT (proxy)."""

    reached: Callable[[], Awaitable[object]]
    release: Callable[[], None]


def _gate(env: InstanceEnv, fork: Fork, instance_id: str, phase: str) -> Gate:
    match phase:
        case "migrate":
            hold = asyncio.Event()
            fork.vm.qmp.hold_migration = hold
            return Gate(reached=fork.vm.qmp.migration_waiting.wait, release=hold.set)
        case "body-put":
            rule = env.proxy.add(("PUT", "MPU_CREATE"), _body(instance_id), Hold())
        case "head-put":
            rule = env.proxy.add("PUT_IF", _exact(head_key(instance_id)), Hold())
        case _:
            raise AssertionError(phase)
    return Gate(reached=rule.reached.wait, release=rule.release)


async def _held_save(lab: Lab, seed: int, phase: str) -> tuple[str, Fork, Gate, asyncio.Task[InstanceRef]]:
    """A session of generation 1 with a change, saving, held at `phase`."""
    instance_id = await lab.base(seed)
    fork = await lab.reuse(instance_id)
    _change(fork.vm, _op("put", "ram", block=1, value=seed % 255 + 1, tag=None))
    gate = _gate(lab.env, fork, instance_id, phase)
    saving = asyncio.create_task(lab.save(fork))
    await _within(gate.reached())
    return instance_id, fork, gate, saving


class TestCloseDuringSave:
    @pytest.mark.parametrize("phase", _PHASES)
    async def test_close_during_save(self, lab: Lab, phase: str) -> None:
        """close() while the save is held. Migration: close() kills the VM, the save fails
        (SessionClosedError), the head stays. Body or head PUT: the snapshot was taken, the save
        lands (g+1). Either way the session is closed and no local file stays."""
        env = lab.env
        instance_id, fork, gate, saving = await _held_save(lab, seed=7, phase=phase)
        before = env.bucket.body_keys(instance_id)
        mark = len(env.proxy.log)
        try:
            await _within(fork.session.close())
            assert fork.session.closed
        finally:
            gate.release()
        if phase == "migrate":
            with pytest.raises(SessionClosedError):
                await _within(saving)
            assert await lab.generation(instance_id) == 1
            assert env.bucket.body_keys(instance_id) == before
            assert env.proxy.count(("PUT", "MPU_CREATE", "PUT_IF"), since=mark) == 0
        else:
            ref = await _within(saving)
            assert (ref.instance_id, ref.generation) == (instance_id, 2)
            assert fork.session.instance == ref
            assert await lab.generation(instance_id) == 2
            await lab.check(instance_id, 2)
        assert env.work_files() == set()
        with pytest.raises(SessionClosedError):
            await fork.session.exec(_op("put", "ram", block=1, value=1, tag=None))
        with pytest.raises(SessionClosedError):
            await fork.session.save()

    @pytest.mark.parametrize("phase", _PHASES)
    async def test_cancel_during_save(self, lab: Lab, phase: str) -> None:
        """A cancelled save closes the session and leaves no local file. Migration or body PUT: the
        head stays (an orphan body object may land; delete_instance removes it). Head PUT: the head
        write runs on and lands g+1 once released."""
        env = lab.env
        instance_id, fork, gate, saving = await _held_save(lab, seed=8, phase=phase)
        before = env.bucket.body_keys(instance_id)
        mark = len(env.proxy.log)
        try:
            saving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await _within(saving)
            assert fork.session.closed
            assert env.work_files() == set()
            assert await lab.generation(instance_id) == 1
        finally:
            gate.release()
        match phase:
            case "migrate":
                assert env.proxy.count(("PUT", "MPU_CREATE", "PUT_IF"), since=mark) == 0
                assert await lab.generation(instance_id) == 1
            case "body-put":
                await await_until(lambda: env.proxy.count(("PUT", "MPU_CREATE"), _body(instance_id), since=mark) == 1)
                assert env.proxy.count("PUT_IF", since=mark) == 0
                assert await lab.generation(instance_id) == 1
                assert len(set(env.bucket.body_keys(instance_id)) - set(before)) == 1  # the orphan
                assert await env.manager.delete(instance_id)
                assert env.bucket.listing(f"s/{instance_id}/") == []
            case _:
                await _within(env.manager.drain())
                assert await lab.generation(instance_id) == 2
                lab.want[(instance_id, 2)] = fork.guest.snapshots[-1]
                head = await lab.check(instance_id, 2)
                assert env.bucket.body_keys(instance_id) == sorted(chain_keys(head))
        assert env.work_files() == set()

    async def test_drain_waits_for_cancelled_commit(self, lab: Lab) -> None:
        """A save cancelled during its head write: drain() (what Scheduler exit awaits before the
        store clients go) returns only after that write landed. The store records, when it
        applies the write, whether drain() had already returned."""
        env = lab.env
        instance_id = await lab.base(seed=9)
        fork = await lab.reuse(instance_id)
        _change(fork.vm, _op("put", "ram", block=1, value=10, tag=None))
        draining: list[asyncio.Task[None]] = []
        drained_before_write: list[bool] = []

        async def head_applied(req: Request, resp: Response) -> None:
            drained_before_write.append(draining[0].done())

        head = _exact(head_key(instance_id))
        gate = env.proxy.add("PUT_IF", head, Hold())
        env.proxy.add("PUT_IF", head, After(head_applied))
        saving = asyncio.create_task(lab.save(fork))
        await _within(gate.reached.wait())
        try:
            saving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await _within(saving)
            assert fork.session.closed
            draining.append(asyncio.create_task(env.manager.drain()))
            assert await lab.generation(instance_id) == 1  # the head write is still held
        finally:
            gate.release()
        await _within(draining[0])
        assert drained_before_write == [False]  # the write ran on after the cancel, and drain() waited for it
        assert await lab.generation(instance_id) == 2  # no poll: drain returned after the write
        lab.want[(instance_id, 2)] = fork.guest.snapshots[-1]
        await lab.check(instance_id, 2)
        assert env.work_files() == set()


class TestWorkFileRemoval:
    async def test_drain_waits_for_a_load_in_flight(self, lab: Lab) -> None:
        """A load downloads when drain() starts (a reuse() while the Scheduler stops): drain()
        returns only after the load ended, so the store clients and the work folder never go
        under it."""
        env = lab.env
        instance_id = await lab.base(seed=17)
        (key,) = env.bucket.body_keys(instance_id)
        gate = env.proxy.add("GET", _exact(key), Hold())
        loading = asyncio.create_task(env.load(instance_id, manager=env.manager))
        await _within(gate.reached.wait())
        try:
            draining = asyncio.create_task(env.manager.drain())
        finally:
            gate.release()
        await _within(draining)
        assert loading.done()
        await env.manager.release(loading.result())
        assert env.work_files() == set()

    async def test_stop_refuses_a_save_that_starts_after_it(self, lab: Lab) -> None:
        """stop() (what Scheduler exit awaits) refuses a save that starts while it waits for a
        load in flight: sessions that keep saving cannot hold the exit (drain() alone waits for
        every call that starts). The load in flight still ends before stop() returns."""
        env = lab.env
        instance_id = await lab.base(seed=17)
        fork = await lab.new(18, instance_id=None)
        (key,) = env.bucket.body_keys(instance_id)
        gate = env.proxy.add("GET", _exact(key), Hold())
        loading = asyncio.create_task(env.load(instance_id, manager=env.manager))
        await _within(gate.reached.wait())
        stopping = asyncio.create_task(env.manager.stop())
        try:
            await asyncio.sleep(0)  # stop() began: it waits for the load
            with pytest.raises(
                InvalidInputError,
                check=lambda e: e.code is ErrorCode.SCHEDULER_STATE,
                match=r"^The scheduler is stopping: no save or load starts now$",
            ):
                await fork.session.save()
        finally:
            gate.release()
        await _within(stopping)
        assert loading.done()
        await env.manager.release(loading.result())
        await fork.session.close()

    async def test_drain_returns_when_only_ended_jobs_remain(self, lab: Lab) -> None:
        """A job that ended but whose discard callback has not run yet (a window a loaded
        host hits): drain() must return, not spin. White-box: no public call can hold
        the set in that state. A drain() that loops without a yield never returns here (the test
        times out)."""
        manager = lab.env.manager
        ended = asyncio.ensure_future(asyncio.sleep(0))
        await ended
        manager._background.add(ended)  # pyright: ignore[reportPrivateUsage]  # ended, still listed
        await manager.drain()

    async def test_save_returns_before_its_files_are_removed(self, lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
        """A save removes its large work files (its body object, the old base's streams) in a
        worker thread, off the event loop: the save returns without waiting for them, and
        drain() (what Scheduler exit awaits before the work folder goes) waits for them."""
        env = lab.env
        instance_id = await lab.base(seed=14)
        fork = await lab.reuse(instance_id)
        assert fork.state is not None
        old, folder = fork.state, fork.state.folder
        loop_thread, removers = threading.get_ident(), list[int]()
        released = threading.Event()
        unlink = Path.unlink

        def held(self: Path, missing_ok: bool = False) -> None:
            if self.parent == folder:
                removers.append(threading.get_ident())
                if removers[-1] != loop_thread:  # a removal on the loop must not block the test
                    released.wait(_WAIT)
            unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", held)
        await fork.session.exec(_op("put", "ram", block=1, value=14, tag=1))
        try:
            assert (await _within(lab.save(fork))).generation == 2
            draining = asyncio.create_task(env.manager.drain())
            done, _ = await asyncio.wait({draining}, timeout=0.2)
            assert not done, "drain() returned while the save's files were being removed"
            assert old.vmstate.exists()
            assert old.home.exists()
        finally:
            released.set()
        await _within(draining)
        assert removers
        assert loop_thread not in removers
        assert not old.vmstate.exists()
        assert not old.home.exists()
        assert sorted(path.suffix for path in folder.iterdir()) == [".home", ".vmstate"]  # the new base only
        await fork.session.close()
        assert env.work_files() == set()

    async def test_failed_save_removes_its_files_off_the_loop(self, lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
        """A save that fails (local disk full while sealing) raises its own error at once; its
        new files go in a worker thread, off the event loop, and drain() waits for them."""
        env = lab.env
        instance_id = await lab.base(seed=15)
        fork = await lab.reuse(instance_id)
        assert fork.state is not None
        base, folder = {fork.state.vmstate, fork.state.home}, fork.state.folder
        loop_thread, removers = threading.get_ident(), list[int]()
        released = threading.Event()
        unlink = Path.unlink

        def held(self: Path, missing_ok: bool = False) -> None:
            if self.parent == folder and self not in base:
                removers.append(threading.get_ident())
                if removers[-1] != loop_thread:  # a removal on the loop must not block the test
                    released.wait(_WAIT)
            unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", held)
        disk_full_once(monkeypatch, step="seal")
        await fork.session.exec(_op("put", "ram", block=1, value=15, tag=1))
        try:
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK) as exc:
                await _within(fork.session.save())
            assert exc.value.context["errno"] == errno.ENOSPC
            draining = asyncio.create_task(env.manager.drain())
            done, _ = await asyncio.wait({draining}, timeout=0.2)
            assert not done, "drain() returned while the failed save's files were being removed"
        finally:
            released.set()
        await _within(draining)
        assert removers
        assert loop_thread not in removers
        assert sorted(path.suffix for path in folder.iterdir()) == [".home", ".vmstate"]  # the base only
        with pytest.warns(InstanceNotSavedWarning):  # the exec's change was never saved
            await fork.session.close()
        assert env.work_files() == set()

    async def test_failed_load_removes_its_folder_off_the_loop(self, lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
        """A load that fails (a changed body object) raises its own error at once; its work
        folder goes in a worker thread, off the event loop, and drain() waits for it."""
        env = lab.env
        instance_id = await lab.base(seed=16)
        (key,) = env.bucket.body_keys(instance_id)
        env.bucket.flip(key, env.bucket.size(key) // 2)
        await env.manager.drain()
        files = env.work_files()
        loop_thread, removers = threading.get_ident(), list[int]()
        released = threading.Event()
        rmtree = shutil.rmtree

        def held(path: Path, *args: Any, **kwargs: Any) -> None:
            if Path(path).name.startswith(f"{instance_id}-"):
                removers.append(threading.get_ident())
                if removers[-1] != loop_thread:  # a removal on the loop must not block the test
                    released.wait(_WAIT)
            rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", held)
        try:
            with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY) as exc:
                await _within(env.load(instance_id, manager=env.manager))
            assert (exc.value.action, exc.value.instance_id) == ("RECREATE", instance_id)
            draining = asyncio.create_task(env.manager.drain())
            done, _ = await asyncio.wait({draining}, timeout=0.2)
            assert not done, "drain() returned while the failed load's folder was being removed"
        finally:
            released.set()
        await _within(draining)
        assert removers
        assert loop_thread not in removers
        assert env.work_files() == files


# ----------------------------------------------------------------------------- disk full


class TestDiskFull:
    @pytest.mark.parametrize("where", ["clone", "seal", "qemu"])
    async def test_disk_full(self, lab: Lab, monkeypatch: pytest.MonkeyPatch, where: str) -> None:
        """Local disk full (home copy or sealing: TransientError LOCAL_DISK) or QEMU's disk full
        (migration failed, VM resumed: TransientError MIGRATION_FAILED, vm_resumed): nothing saved,
        no new file, the session stays open, and the next exec and save succeed."""
        env = lab.env
        instance_id = await lab.base(seed=10)
        fork = await lab.reuse(instance_id)
        await fork.session.exec(_op("put", "ram", block=1, value=10, tag=1))
        files = env.work_files()
        keys = env.bucket.body_keys(instance_id)
        mark = len(env.proxy.log)
        if where == "qemu":
            fork.vm.qmp.migration_error = "No space left on device"
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.MIGRATION_FAILED) as exc:
                await fork.session.save()
            assert exc.value.context["vm_resumed"] is True
            assert "No space left on device" in str(exc.value)
            assert fork.vm.qmp.commands[-2:] == ["migrate_cancel", "cont"]
            fork.vm.qmp.migration_error = None
        else:
            disk_full_once(monkeypatch, step="clone" if where == "clone" else "seal")
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK) as exc:
                await fork.session.save()
            assert exc.value.context["errno"] == errno.ENOSPC
            if where == "clone":  # failed while the VM was stopped: it was resumed
                assert fork.vm.qmp.commands[-2:] == ["migrate_cancel", "cont"]
        assert not fork.session.closed
        assert fork.session.instance is not None
        assert fork.session.instance.generation == 1
        assert await lab.generation(instance_id) == 1
        assert env.bucket.body_keys(instance_id) == keys
        assert env.proxy.count(("PUT", "MPU_CREATE", "PUT_IF"), f"^s/{instance_id}/", since=mark) == 0
        await env.manager.drain()  # the failed save's files go in a worker thread
        assert env.work_files() == files
        await fork.session.exec(_op("put", "home", block=2, value=20, tag=2))
        ref = await lab.save(fork)
        assert ref.generation == 2
        await fork.session.close()
        await lab.check(instance_id, 2)
        assert env.work_files() == set()

    async def test_failed_save_that_leaves_the_vm_paused_closes_the_session(self, lab: Lab) -> None:
        """QEMU did not write the state and refused `cont`: the VM may stay paused, so no later
        exec could run. The session closes (TransientError MIGRATION_FAILED, vm_resumed False);
        nothing was saved."""
        env = lab.env
        instance_id = await lab.base(seed=11)
        fork = await lab.reuse(instance_id)
        await fork.session.exec(_op("put", "ram", block=1, value=11, tag=1))
        fork.vm.qmp.migration_error = "No space left on device"
        fork.vm.qmp.refused["cont"] = "cont refused"
        mark = len(env.proxy.log)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.MIGRATION_FAILED) as exc:
            await fork.session.save()
        assert exc.value.context["vm_resumed"] is False
        assert str(exc.value).startswith("Save failed and the VM did not resume:")
        assert fork.session.closed
        assert env.proxy.count(("PUT", "MPU_CREATE", "PUT_IF"), since=mark) == 0
        assert await lab.generation(instance_id) == 1
        await env.manager.drain()
        assert env.work_files() == set()


# ----------------------------------------------------------------------------- idle timeout


class TestIdleTimeout:
    @pytest.mark.parametrize("changed", [True, False], ids=["changes", "no-changes"])
    async def test_idle_close_warns_about_unsaved_changes(
        self, lab: Lab, caplog: pytest.LogCaptureFixture, changed: bool
    ) -> None:
        """The idle timeout closes a reused session; a warning names the instance and its last
        saved generation only when code ran since that save. The session is made as reuse()
        makes it, with an idle timeout of 0 s (the public minimum, 10 s, is too long for a test):
        the timer fires at the first loop turn after the last call."""
        env = lab.env
        instance_id = await lab.base(seed=12)
        state = await env.load(instance_id, manager=env.manager)
        vm = await env.vm_from(state)
        with caplog.at_level(logging.WARNING, logger="exec_sandbox.session"):
            session = Session(
                vm=vm,  # type: ignore[arg-type]
                vm_manager=env.vm_manager,  # type: ignore[arg-type]
                spec=vm.spec,
                startup_timing=_TIMING,
                instances=env.manager,
                instance=state,
                new_instance_id=None,
                idle_timeout_seconds=0,
                default_timeout_seconds=60,
            )
            if changed:  # no await before the exec: it stops the timer before the timer can fire
                await session.exec(_op("put", "ram", block=1, value=12, tag=1))
            await await_until(lambda: session.closed)
            await session.close()  # the idle close is done (it never warns: no InstanceNotSavedWarning)
        warned = [
            (record.instance_id, record.generation)  # type: ignore[attr-defined]
            for record in caplog.records
            if record.getMessage() == "Idle timeout closes a session with changes since the last save"
        ]
        assert warned == ([(instance_id, 1)] if changed else [])
        assert session.exec_count == (1 if changed else 0)
        assert env.work_files() == set()


# ----------------------------------------------------------------------------- work files that will not go


def _unlink_fails(monkeypatch: pytest.MonkeyPatch, *, when: Callable[[Path], bool]) -> list[Path]:
    """Path.unlink of a path `when` accepts fails with EIO (rmtree is not touched); returns the
    paths it refused."""
    refused: list[Path] = []
    unlink = Path.unlink

    def failing(self: Path, missing_ok: bool = False) -> None:
        if when(self):
            refused.append(self)
            raise OSError(errno.EIO, "Input/output error", str(self))
        unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing)
    return refused


class TestCleanupFailure:
    async def test_committed_save_keeps_its_result(self, lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
        """The old base files of a committed save cannot be removed (EIO): the save still returns
        its generation and becomes the session's base, so the next save is not a conflict;
        close removes the work folder."""
        env = lab.env
        instance_id = await lab.base(seed=11)
        fork = await lab.reuse(instance_id)
        assert fork.state is not None
        old = {fork.state.vmstate, fork.state.home}
        refused = _unlink_fails(monkeypatch, when=old.__contains__)
        await fork.session.exec(_op("put", "ram", block=1, value=11, tag=1))
        assert (await lab.save(fork)).generation == 2
        await _within(env.manager.drain())  # the old base's files go in a worker thread after the save
        assert set(refused) == old
        assert all(path.exists() for path in old)
        assert not fork.session.closed
        await fork.session.exec(_op("put", "home", block=2, value=22, tag=2))
        assert (await lab.save(fork)).generation == 3
        await fork.session.close()
        assert env.work_files() == set()
        await lab.check(instance_id, 3)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root removes files from a folder it may not write")
    async def test_close_logs_work_files_that_will_not_go(self, lab: Lab, caplog: pytest.LogCaptureFixture) -> None:
        """The work folder of a closing session cannot be emptied (no write access to it):
        close still returns, and each file that stays is logged, never dropped in silence."""
        instance_id = await lab.base(seed=13)
        fork = await lab.reuse(instance_id)
        assert fork.state is not None
        folder, kept = fork.state.folder, {fork.state.vmstate, fork.state.home}
        folder.chmod(0o500)
        try:
            # On the library logger: a CLI test (configure_logging, -q) may have left it at ERROR
            with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
                await fork.session.close()
            logged = {
                record.path  # type: ignore[attr-defined]
                for record in caplog.records
                if record.getMessage() == "Instance work file not removed"
            }
            assert logged == {str(path) for path in (*kept, folder)}  # each file, then the folder
            assert all(path.exists() for path in kept)
        finally:
            folder.chmod(0o700)
            shutil.rmtree(folder)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root looks into a folder it may not search")
    async def test_close_logs_a_work_folder_it_cannot_look_at(self, lab: Lab, caplog: pytest.LogCaptureFixture) -> None:
        """The work folder of a closing session cannot even be looked at (EACCES: no search
        access to the folder above it): close still returns, and the folder is logged."""
        env = lab.env
        instance_id = await lab.base(seed=18)
        fork = await lab.reuse(instance_id)
        assert fork.state is not None
        folder = fork.state.folder
        await _within(env.manager.drain())
        env.work_dir.chmod(0o600)
        try:
            with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
                await fork.session.close()
        finally:
            env.work_dir.chmod(0o700)
        logged = [
            record.path  # type: ignore[attr-defined]
            for record in caplog.records
            if record.getMessage() == "Instance work file not removed"
        ]
        assert logged == [str(folder)]
        shutil.rmtree(folder)

    async def test_refused_save_keeps_its_error(self, lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
        """The new files of a refused save cannot be removed (EIO): the caller still gets the
        conflict, not the OSError of the cleanup."""
        instance_id = await lab.base(seed=12)
        a, b = await lab.reuse(instance_id), await lab.reuse(instance_id)
        await a.session.exec(_op("put", "ram", block=1, value=12, tag=1))
        assert (await lab.save(a)).generation == 2
        assert b.state is not None
        old, folder = {b.state.vmstate, b.state.home}, b.state.folder
        refused = _unlink_fails(
            monkeypatch, when=lambda path: path.parent == folder and path.suffix in {".vmstate", ".home"}
        )
        await b.session.exec(_op("put", "home", block=1, value=120, tag=2))
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT) as exc:
            await b.session.save()
        _assert_loser(b, exc.value, instance_id)
        await _within(lab.env.manager.drain())  # the refused save's files go in a worker thread
        assert sorted(path.suffix for path in refused) == [".home", ".vmstate"]
        assert not old & set(refused)
        await a.session.close()
        await lab.check(instance_id, 2)


# ----------------------------------------------------------------------------- forks and delete


def _assert_loser(fork: Fork, error: BaseException, instance_id: str) -> None:
    assert isinstance(error, OutcomeUnknownError), repr(error)
    assert (error.code, error.action, error.instance_id) == ("INSTANCE_CONFLICT", "VERIFY", instance_id)
    assert fork.session.closed
    assert fork.state is not None
    assert not fork.state.folder.exists()


class TestForks:
    @pytest.mark.parametrize("k", [1, 5, 20])
    async def test_fork_sequential(self, lab: Lab, k: int) -> None:
        """A and B load g; A saves k times; B's save is refused (412): conflict (VERIFY), B closed,
        its body object deleted, its folder gone, later calls SessionClosedError; A's chain stays."""
        env = lab.env
        instance_id = await lab.base(seed=13)
        a, b = await lab.reuse(instance_id), await lab.reuse(instance_id)
        for n in range(k):
            await a.session.exec(_op("put", "ram", block=1, value=1 + n, tag=n))
            assert (await lab.save(a)).generation == 2 + n
        await b.session.exec(_op("put", "home", block=1, value=200, tag=0))
        mark = len(env.proxy.log)
        with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT) as exc:
            await b.session.save()
        _assert_loser(b, exc.value, instance_id)
        assert [e.status for e in env.proxy.entries("PUT_IF", since=mark)] == [412]
        (upload,) = [e.key for e in env.proxy.entries(("PUT", "MPU_CREATE"), _body(instance_id), since=mark)]
        assert upload not in env.bucket.listing()
        with pytest.raises(SessionClosedError):
            await b.session.exec(_op("put", "home", block=1, value=201, tag=None))
        with pytest.raises(SessionClosedError):
            await b.session.save()
        await b.session.close()
        await a.session.close()
        head = await lab.check(instance_id, 1 + k)
        assert env.bucket.body_keys(instance_id) == sorted(chain_keys(head))
        assert env.work_files() == set()

    @pytest.mark.parametrize("n", [2, 8])
    async def test_fork_concurrent_saves(self, lab: Lab, n: int) -> None:
        """N forks of g save at once (head writes held until all wait): exactly one commits g+1,
        N-1 conflict and close; the head is the winner's bytes; the store holds its chain only."""
        env = lab.env
        instance_id = await lab.base(seed=14)
        forks = [await lab.reuse(instance_id) for _ in range(n)]
        for i, fork in enumerate(forks):
            await fork.session.exec(_op("put", "ram", block=1, value=1 + i, tag=i))
        gate = env.proxy.add("PUT_IF", _exact(head_key(instance_id)), Hold(), times=None)
        mark = len(env.proxy.log)
        tasks = [asyncio.create_task(lab.save(fork)) for fork in forks]
        try:
            await await_until(lambda: gate.hits == n)  # every save waits at the gate
        finally:
            gate.release()
        results = await _within(asyncio.gather(*tasks, return_exceptions=True))
        winners = [i for i, r in enumerate(results) if isinstance(r, InstanceRef)]
        assert len(winners) == 1, results
        (winner,) = winners
        assert results[winner] == InstanceRef(instance_id=instance_id, generation=2)
        assert not forks[winner].session.closed
        for i, result in enumerate(results):
            if i != winner:
                assert isinstance(result, BaseException)
                _assert_loser(forks[i], result, instance_id)
        assert sorted(e.status for e in env.proxy.entries("PUT_IF", since=mark)) == [200] + [412] * (n - 1)
        assert env.proxy.count(("PUT", "MPU_CREATE"), _body(instance_id), since=mark) == n
        await _close_quietly(forks)
        head = await lab.check(instance_id, 2)
        assert env.bucket.body_keys(instance_id) == sorted(chain_keys(head))
        assert env.work_files() == set()

    @pytest.mark.timeout(30)  # 25 examples, each of up to 3 forks and 15 steps (exec or save)
    @_HYPOTHESIS
    @given(
        scripts=st.lists(st.lists(st.sampled_from(["exec", "save"]), min_size=1, max_size=5), min_size=2, max_size=3)
    )
    async def test_fork_interleaved(self, lab: Lab, scripts: list[list[str]]) -> None:
        """Forks of one generation run exec/save scripts at once. The first commit wins: only that
        fork ever commits, its generations follow each other, every other fork's first save
        conflicts and from then on gets only SessionClosedError; the head is the last commit."""
        env = lab.env
        instance_id = await lab.base(seed=15)
        forks = [await lab.reuse(instance_id) for _ in scripts]
        outcomes: list[list[tuple[str, int]]] = [[] for _ in scripts]

        async def run(i: int) -> None:
            fork = forks[i]
            for j, step in enumerate(scripts[i]):
                try:
                    if step == "exec":
                        await fork.session.exec(_op("put", "ram", block=1, value=1 + 8 * i + j, tag=j))
                        outcomes[i].append(("exec", 0))
                    else:
                        outcomes[i].append(("saved", (await lab.save(fork)).generation))
                except OutcomeUnknownError as e:
                    assert (e.code, e.action, e.instance_id) == ("INSTANCE_CONFLICT", "VERIFY", instance_id)
                    outcomes[i].append(("conflict", 0))
                except SessionClosedError:
                    outcomes[i].append(("closed", 0))

        await _within(asyncio.gather(*(run(i) for i in range(len(scripts)))))

        committed = [g for steps in outcomes for kind, g in steps if kind == "saved"]
        assert sorted(committed) == list(range(2, 2 + len(committed))), outcomes
        assert sum(any(kind == "saved" for kind, _ in steps) for steps in outcomes) <= 1, outcomes
        for i, steps in enumerate(outcomes):
            kinds = [kind for kind, _ in steps]
            first = kinds.index("conflict") if "conflict" in kinds else len(kinds)
            assert "closed" not in kinds[:first], outcomes
            assert all(kind == "closed" for kind in kinds[first + 1 :]), outcomes
            assert forks[i].session.closed == (first < len(kinds)), outcomes
        generation = max(committed, default=1)
        assert await lab.generation(instance_id) == generation
        await _close_quietly(forks)
        head = await lab.check(instance_id, generation)
        assert env.bucket.body_keys(instance_id) == sorted(chain_keys(head))
        assert env.work_files() == set()

    async def test_fork_loaded_after_save(self, lab: Lab) -> None:
        """A commits g+1; C, loaded after that, commits g+2 (no false conflict); A and B, now stale, conflict."""
        env = lab.env
        instance_id = await lab.base(seed=16)
        a, b = await lab.reuse(instance_id), await lab.reuse(instance_id)
        await a.session.exec(_op("put", "ram", block=1, value=161, tag=1))
        assert (await lab.save(a)).generation == 2
        c = await lab.reuse(instance_id)  # loads generation 2 (checked byte for byte)
        await c.session.exec(_op("put", "home", block=1, value=162, tag=1))
        assert (await lab.save(c)).generation == 3
        for stale in (b, a):
            await stale.session.exec(_op("put", "ram", block=2, value=163, tag=2))
            with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT) as exc:
                await stale.session.save()
            assert (exc.value.action, stale.session.closed) == ("VERIFY", True)
        await c.session.close()
        head = await lab.check(instance_id, 3)
        assert env.bucket.body_keys(instance_id) == sorted(chain_keys(head))
        assert env.work_files() == set()

    async def test_save_after_delete(self, lab: Lab) -> None:
        """delete_instance during a session: its save is InstanceGoneError INSTANCE_NOT_FOUND (RECREATE), the
        session closes, its upload is deleted, nothing of the instance stays (store or disk)."""
        env = lab.env
        instance_id = await lab.base(seed=17)
        fork = await lab.reuse(instance_id)
        await fork.session.exec(_op("put", "ram", block=1, value=17, tag=1))
        assert await env.manager.delete(instance_id)
        mark = len(env.proxy.log)
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND) as exc:
            await fork.session.save()
        assert (exc.value.action, exc.value.instance_id) == ("RECREATE", instance_id)
        assert fork.session.closed
        assert env.proxy.count(("PUT", "MPU_CREATE"), _body(instance_id), since=mark) == 1
        assert env.bucket.listing(f"s/{instance_id}/") == []
        assert env.work_files() == set()
        with pytest.raises(SessionClosedError):
            await fork.session.exec(_op("put", "ram", block=1, value=18, tag=None))
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND):
            await env.manager.describe(instance_id)


# ----------------------------------------------------------------------------- Scheduler.new(instance_id=...)


class TestForcedId:
    """A session from Scheduler.new(instance_id=...): its first save commits under that id, and
    never over an instance that already has it (the first head write is create-only)."""

    async def test_first_save_uses_the_forced_id(self, lab: Lab) -> None:
        fork = await lab.new(seed=21, instance_id="my-vm")
        ref = await lab.save(fork)
        assert ref == InstanceRef(instance_id="my-vm", generation=1)
        await fork.session.close()
        assert head_object("my-vm") in lab.env.bucket.listing()
        await lab.check("my-vm", 1)

    @pytest.mark.parametrize("owner", ["same-keyring", "other-keyring"])
    async def test_taken_id_conflicts_and_keeps_the_instance(self, lab: Lab, owner: str) -> None:
        """The id is taken (by this keyring, or by one this worker lacks, whose head it cannot
        open): the create-only head write is refused, so the save conflicts (VERIFY), the session
        closes, its upload is deleted, and the first instance stays as it was."""
        env = lab.env
        first_owner = env.manager if owner == "same-keyring" else env.manager_for(new_keyring())
        first_vm = await env.new_vm(*_layout(22), spec=PYTHON)
        want = first_vm.expected()
        await first_owner.release(await first_owner.save(first_vm, None, new_instance_id="taken"))  # type: ignore[arg-type]
        head, objects = env.bucket.read(head_object("taken")), env.bucket.listing("s/taken/")

        fork = await lab.new(seed=23, instance_id="taken")
        mark = len(env.proxy.log)
        with pytest.raises(
            OutcomeUnknownError, match="already exists", check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT
        ) as exc:
            await fork.session.save()
        assert (exc.value.action, exc.value.instance_id) == ("VERIFY", "taken")
        assert fork.session.closed
        assert [e.status for e in env.proxy.entries("PUT_IF", since=mark)] == [412]
        assert env.bucket.read(head_object("taken")) == head
        assert env.bucket.listing("s/taken/") == objects  # the loser's upload is deleted
        await env.manager.drain()  # the loser's folder goes in a worker thread
        assert env.work_files() == set()
        await env.check_load("taken", generation=1, manager=first_owner, want=want)


# ----------------------------------------------------------------------------- delete during a first save


class TestDeleteDuringFirstSave:
    """delete_instance during a first save finds no head and removes the save's object: the
    save must not leave a head that names nothing."""

    async def test_delete_before_the_head_write(self, lab: Lab) -> None:
        """Another worker deletes the instance between the upload and the create-only head
        write. The head lands, the save finds its object gone, removes the head and raises
        InstanceGoneError INSTANCE_NOT_FOUND (RECREATE); the session closes, nothing of the instance stays."""
        env = lab.env
        fork = await lab.new(seed=19, instance_id=None)
        remover = env.manager_for(env.keyring)
        deleted: list[bool] = []

        async def delete(req: Request) -> None:
            deleted.append(await remover.delete(req.key.split("/")[1]))  # s/<id>/head

        env.proxy.add("PUT_IF", "/head$", Before(delete))
        with pytest.raises(
            InstanceGoneError, match="deleted during this save", check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND
        ) as exc:
            await fork.session.save()
        instance_id = exc.value.instance_id
        assert instance_id is not None
        assert (exc.value.action, deleted) == ("RECREATE", [False])
        assert [e.status for e in env.proxy.entries("PUT_IF")] == [200]  # the head landed
        assert fork.session.closed
        assert env.bucket.listing(f"s/{instance_id}/") == []
        await env.manager.drain()  # the failed save's folder goes in a worker thread
        assert env.work_files() == set()
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND):
            await env.manager.describe(instance_id)

    async def test_delete_that_listed_the_object_before_the_head_write(self, lab: Lab) -> None:
        """Another worker's delete finds no head and lists the save's object; the head lands and
        the save's check still finds the object, so the save succeeds; then the delete removes
        the object. The delete also removes the head that appeared (as if it ran after the
        save) and returns True: no head names a missing object."""
        env = lab.env
        fork = await lab.new(seed=21, instance_id=None)
        remover = env.manager_for(env.keyring)
        held = env.proxy.add("DELETE_MANY", r"^s/[^/]+/b/", Hold())  # the delete's removal of the listed object
        deletes: list[asyncio.Task[bool]] = []

        async def start_delete(req: Request) -> None:
            deletes.append(asyncio.create_task(remover.delete(req.key.split("/")[1])))  # s/<id>/head
            await _within(held.reached.wait())  # the delete listed the object and waits to remove it

        env.proxy.add("PUT_IF", "/head$", Before(start_delete))
        ref = await lab.save(fork)
        assert ref.generation == 1
        held.release()
        assert await _within(deletes[0]) is True
        await fork.session.close()
        assert env.bucket.listing(f"s/{ref.instance_id}/") == []
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND):
            await env.manager.describe(ref.instance_id)
        assert env.work_files() == set()

    async def test_check_without_answer_is_unknown(self, lab: Lab) -> None:
        """The store does not answer the check after the head write: the outcome is unknown
        (VERIFY), the session closes, and the head it wrote stays and loads."""
        env = lab.env
        fork = await lab.new(seed=20, instance_id=None)
        env.proxy.add("HEAD", r"^s/[^/]+/b/", Fail(503, "ServiceUnavailable"), times=None)
        with pytest.raises(
            OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN
        ) as exc:
            await fork.session.save()
        instance_id = exc.value.instance_id
        assert instance_id is not None
        assert exc.value.action == "VERIFY"
        assert env.proxy.count("HEAD", r"^s/[^/]+/b/") == 1
        assert fork.session.closed
        env.proxy.clear()
        await env.check_load(instance_id, generation=1, want=fork.guest.snapshots[-1], manager=env.manager)
        # The error's context names the save, as recent_commits lists it: it landed
        info = await env.manager.describe(instance_id)
        assert exc.value.context == {"generation": 1, "commit_id": info.recent_commits[0].commit_id}
        await env.manager.drain()  # the failed save's folder goes in a worker thread
        assert env.work_files() == set()

"""Saved instances on real VMs under concurrency: parallel sessions, many saves at
once, and two sessions of one saved state that both save (sequentially).

Concurrent saves of one instance (forks that race on the head) are not here: bare
moto does not make a conditional PUT atomic, so two writers can both win.
"""

import asyncio
import hashlib
import io
import logging
from pathlib import Path

import pytest

from exec_sandbox import (
    ErrorCode,
    InstanceRef,
    Language,
    OutcomeUnknownError,
    Scheduler,
    Session,
    SessionClosedError,
    VmSpec,
)
from tests.conftest import skip_unless_hwaccel
from tests.instance_fixtures import (
    GuestModel,
    InstanceWorld,
    assert_origin,
    assert_restored,
    close_saved,
    close_unsaved,
    instance_world_fixture,  # noqa: F401 - fixture "instance_world"
    new_model,
    play_round,
    primed_l1_dir_fixture,  # noqa: F401 - fixture "primed_l1_dir"
    reuse_and_check,
    start_guest,
)

pytestmark = skip_unless_hwaccel

logger = logging.getLogger(__name__)

_PY = VmSpec(language=Language.PYTHON)


def _busy_work() -> tuple[dict[Language, str], dict[Language, str]]:
    """Code that keeps a VM busy for a moment, and its exact output."""
    digest = b""
    for _ in range(20000):
        digest = hashlib.sha256(digest).digest()
    code = {
        Language.PYTHON: "import hashlib\nh = b''\nfor _ in range(20000):\n    h = hashlib.sha256(h).digest()\nprint(h.hex())",
        Language.RAW: "i=0; while [ $i -lt 20000 ]; do i=$((i+1)); done; echo $i",
    }
    return code, {Language.PYTHON: digest.hex(), Language.RAW: "20000"}


_BUSY_CODE, _BUSY_OUTPUT = _busy_work()


async def _first_done(task: asyncio.Task[object], other: asyncio.Task[object]) -> None:
    """Wait until `task` ends; fail if `other` ends first."""
    done, _ = await asyncio.wait({task, other}, return_when=asyncio.FIRST_COMPLETED)
    if task not in done:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        pytest.fail(f"{other!r} ended first")
    assert not other.done()


@pytest.mark.timeout(400)  # above the 250 s exec: a session that waits for another fails, not times out
async def test_parallel_sessions(instance_world: InstanceWorld, primed_l1_dir: Path) -> None:
    """While A runs a long exec, B runs exec() and save() to the end: sessions do not wait for each other."""
    async with instance_world.scheduler(
        memory_snapshot_cache_dir=primed_l1_dir, warm_pool_size=0, disk_snapshot_cache_dir=None
    ) as scheduler:
        a, b = await asyncio.gather(scheduler.new(_PY), scheduler.new(_PY))
        started = asyncio.Event()

        def on_stdout(chunk: str) -> None:
            if "started" in chunk:
                started.set()

        long_exec = asyncio.create_task(
            a.exec(
                "import time\nprint('started', flush=True)\ntime.sleep(250)", timeout_seconds=280, on_stdout=on_stdout
            )
        )
        try:
            assert_origin(a, "l1")
            assert_origin(b, "l1")
            await _first_done(asyncio.create_task(started.wait()), long_exec)  # A's code runs in the guest

            model = new_model(Language.PYTHON, seed="parallel")

            async def b_work() -> InstanceRef:
                await start_guest(b, model, ticker=True)
                await play_round(b, model)
                return await b.save()

            b_task = asyncio.create_task(b_work())
            await _first_done(b_task, long_exec)
            ref = b_task.result()
            assert ref.generation == 1

            long_exec.cancel()
            with pytest.raises(asyncio.CancelledError):
                await long_exec
            assert a.closed  # a cancelled exec retires its VM
            await close_saved(b)
        finally:
            long_exec.cancel()
            await asyncio.gather(long_exec, return_exceptions=True)
            await a.close()
            await b.close()
        await reuse_and_check(scheduler, ref, model)
        assert instance_world.instance_folders() == []


async def _saves_while_busy(sessions: list[Session], busy: list[Session]) -> list[InstanceRef]:
    """Every session saves at once while the busy sessions run exec() in a loop."""
    saves_done = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def timed_save(session: Session) -> InstanceRef:
        start = loop.time()
        ref = await session.save()
        logger.info("save of %s (%s) took %.2f s", session.vm_id, session.spec.language.value, loop.time() - start)
        return ref

    async def keep_busy(session: Session) -> int:
        runs = 0
        while not saves_done.is_set():
            result = await session.exec(_BUSY_CODE[session.spec.language])
            assert (result.exit_code, result.stdout.strip()) == (0, _BUSY_OUTPUT[session.spec.language]), result.stderr
            runs += 1
        return runs

    busy_tasks = [asyncio.create_task(keep_busy(s)) for s in busy]
    try:
        refs = await asyncio.gather(*(timed_save(s) for s in sessions))
    finally:
        saves_done.set()
        runs = await asyncio.gather(*busy_tasks)
    logger.info("busy execs during the saves: %s", runs)
    return refs


@pytest.mark.timeout(300)
@pytest.mark.parametrize("count", [4, pytest.param(12, marks=pytest.mark.slow)], ids=["4", "12"])
async def test_many_instances_save_at_once(instance_world: InstanceWorld, primed_l1_dir: Path, count: int) -> None:
    """`count` sessions (mixed languages) make their first saves at once, then their packs at
    once, while other sessions run code. Each instance keeps its own state and store objects."""
    world = instance_world
    languages = [list(Language)[i % len(Language)] for i in range(count)]
    async with world.scheduler(
        memory_snapshot_cache_dir=primed_l1_dir, warm_pool_size=0, disk_snapshot_cache_dir=None
    ) as scheduler:
        sessions = list(await asyncio.gather(*(scheduler.new(VmSpec(language=lang)) for lang in languages)))
        busy = list(await asyncio.gather(*(scheduler.new(VmSpec(language=lang)) for lang in _BUSY_CODE)))
        try:
            for session in [*sessions, *busy]:
                assert_origin(session, "l1")
            models = [new_model(lang, seed=f"many/{i}") for i, lang in enumerate(languages)]
            await asyncio.gather(*(start_guest(s, m, ticker=True) for s, m in zip(sessions, models, strict=True)))

            await asyncio.gather(*(play_round(s, m) for s, m in zip(sessions, models, strict=True)))
            first = await _saves_while_busy(sessions, busy)  # full copies
            ids = [ref.instance_id for ref in first]
            assert len(set(ids)) == count
            assert [ref.generation for ref in first] == [1] * count

            await asyncio.gather(*(play_round(s, m) for s, m in zip(sessions, models, strict=True)))
            second = await _saves_while_busy(sessions, busy)  # packs
            assert second == [InstanceRef(instance_id=iid, generation=2) for iid in ids]
            for session in sessions:
                await close_saved(session)
        finally:
            for session in [*sessions, *busy]:
                await session.close()
        assert world.instance_folders() == []

        assert sorted([iid async for iid in scheduler.list_instances()]) == sorted(ids)
        assert {key.split("/")[1] for key in world.keys("s/")} == set(ids)  # nothing under another id
        for iid in ids:
            info = await scheduler.describe_instance(iid)
            commits = {c.generation: c.commit_id for c in info.recent_commits}
            assert world.instance_keys(iid) == {"head", f"b/{commits[1]}.f", f"b/{commits[2]}.p"}

        await _reuse_all(scheduler, second, models)
        assert world.instance_folders() == []


async def _reuse_all(scheduler: Scheduler, refs: list[InstanceRef], models: list[GuestModel]) -> None:
    """Reuse every instance at once; each equals its own model. Closed one by one (warnings are global state)."""
    reused = list(await asyncio.gather(*(scheduler.reuse(ref.instance_id) for ref in refs)))
    try:
        for vm, ref in zip(reused, refs, strict=True):
            assert_origin(vm, "reuse")
            assert vm.instance == ref
        await asyncio.gather(*(assert_restored(vm, m) for vm, m in zip(reused, models, strict=True)))
        for vm in reused:
            await close_unsaved(vm)
    finally:
        for vm in reused:
            await vm.close()


@pytest.mark.timeout(300)
async def test_fork_sequential(instance_world: InstanceWorld, primed_l1_dir: Path) -> None:
    """A and B load generation 1; A saves generation 2; B's save then conflicts: B is closed,
    its uploaded object and its local folder are gone, and the instance is A's state."""
    world = instance_world
    async with world.scheduler(
        memory_snapshot_cache_dir=primed_l1_dir, warm_pool_size=0, disk_snapshot_cache_dir=None
    ) as scheduler:
        async with await scheduler.new(_PY) as first:
            assert_origin(first, "l1")
            model = new_model(Language.PYTHON, seed="fork")
            await start_guest(first, model, ticker=True)
            await play_round(first, model)
            ref = await first.save()
            await close_saved(first)

        a = await scheduler.reuse(ref.instance_id)
        b = await scheduler.reuse(ref.instance_id)
        try:
            for vm in (a, b):
                assert_origin(vm, "reuse")
                assert vm.instance == ref
            assert len(world.instance_folders(ref.instance_id)) == 2
            model_a, model_b = model, model.fork("fork/b")
            await play_round(a, model_a)
            await play_round(b, model_b)

            saved = await a.save()
            assert saved == InstanceRef(instance_id=ref.instance_id, generation=2)
            with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT) as exc:
                await b.save()
            assert exc.value.action == "VERIFY"
            assert exc.value.instance_id == ref.instance_id
            assert b.closed
            assert b.instance == ref  # B never saved
            with pytest.raises(SessionClosedError):
                await b.exec("print(1)")
            with pytest.raises(SessionClosedError):
                await b.save()
            with pytest.raises(SessionClosedError):
                await b.write_file("state/after.bin", b"after")
            with pytest.raises(SessionClosedError):
                await b.read_file("state/note1.txt", destination=io.BytesIO())
            with pytest.raises(SessionClosedError):
                await b.list_files("state")

            info = await scheduler.describe_instance(ref.instance_id)
            assert info.generation == 2
            commits = {c.generation: c.commit_id for c in info.recent_commits}
            # B's uploaded pack is deleted: only the full copy and A's pack stay
            assert world.instance_keys(ref.instance_id) == {"head", f"b/{commits[1]}.f", f"b/{commits[2]}.p"}
            assert len(world.instance_folders(ref.instance_id)) == 1  # B's folder is gone, A's stays
            await close_saved(a)
            assert world.instance_folders(ref.instance_id) == []
        finally:
            await a.close()
            await b.close()
        await reuse_and_check(scheduler, saved, model_a)

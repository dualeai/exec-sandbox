"""Saved instances on real VMs, per language x VM origin x image (base or an L2 packages layer).

Each session saves twice, then makes changes it never saves and closes. reuse()
must bring back the second save exactly: variables, the files in /home/user,
and a ticker process that runs on under its pid with its tick lines still 1..n
(RAM and the home disk come from one paused instant). Then a third save, and
reuse again.
"""

import asyncio
import hashlib
from pathlib import Path

import pytest

from exec_sandbox import InstanceRef, Language, Scheduler, Session, VmSpec
from tests.conftest import skip_unless_hwaccel
from tests.instance_fixtures import (
    L2_PACKAGES,
    GuestModel,
    InstanceWorld,
    Origin,
    assert_origin,
    assert_packages_import,
    assert_restored,
    close_saved,
    close_unsaved,
    instance_world_fixture,  # noqa: F401 - fixture "instance_world"
    lose_changes,
    new_model,
    new_session,
    play_round,
    primed_l1_dir_fixture,  # noqa: F401 - fixture "primed_l1_dir"
    reuse_and_check,
    start_guest,
)

pytestmark = skip_unless_hwaccel


async def _walk(scheduler: Scheduler, world: InstanceWorld, session: Session, model: GuestModel) -> InstanceRef:
    """round -> save (gen 1) -> round -> save (gen 2) -> lost changes -> close (warns) -> reuse == gen 2
    -> round -> save (gen 3). Returns the gen-3 ref; `model` ends at gen 3."""
    await start_guest(session, model)
    await play_round(session, model)
    first = await session.save()
    assert first.generation == 1
    assert session.instance == first
    await play_round(session, model)
    second = await session.save()
    assert second == InstanceRef(instance_id=first.instance_id, generation=2)
    assert session.instance == second

    await lose_changes(session, model)
    await close_unsaved(session)
    assert world.instance_folders(first.instance_id) == []

    async with await scheduler.reuse(first.instance_id) as vm:
        assert_origin(vm, "reuse")
        assert vm.instance == second
        await assert_restored(vm, model)
        await play_round(vm, model)
        third = await vm.save()
        assert third == InstanceRef(instance_id=first.instance_id, generation=3)
        await close_saved(vm)
    return third


def _file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("language", "packages"),
    [
        pytest.param(Language.PYTHON, (), id="python-base"),
        pytest.param(Language.JAVASCRIPT, (), id="javascript-base"),
        pytest.param(Language.RAW, (), id="raw-base"),
        pytest.param(Language.PYTHON, L2_PACKAGES[Language.PYTHON], id="python-l2"),
        pytest.param(Language.JAVASCRIPT, L2_PACKAGES[Language.JAVASCRIPT], id="javascript-l2"),
    ],
)
async def test_origin_walk(instance_world: InstanceWorld, language: Language, packages: tuple[str, ...]) -> None:
    """Cold boot, then L1 restore (a second Scheduler on the same private L1 dir: the first
    one's exit awaited its background L1 save). With an L2: the packages import, and a worker
    with an empty L2 cache loads the L2 by digest from the store."""
    world = instance_world
    spec = VmSpec(language=language, packages=packages)
    l1_dir = world.private_dir("l1")
    last: tuple[InstanceRef, GuestModel] | None = None
    origins: tuple[Origin, ...] = ("cold", "l1")
    for origin in origins:
        async with world.scheduler(
            memory_snapshot_cache_dir=l1_dir, warm_pool_size=0, disk_snapshot_cache_dir=None
        ) as scheduler:
            session = await new_session(scheduler, spec)
            assert_origin(session, origin)
            if packages:
                await assert_packages_import(session)
            model = new_model(language, seed=f"walk/{origin}")
            ref = await _walk(scheduler, world, session, model)
            await reuse_and_check(scheduler, ref, model)
            assert world.instance_folders() == []
        last = (ref, model)

    if not packages:
        return
    assert last is not None
    ref, model = last
    # Another worker: the same store, no L2 in its disk-snapshot cache
    async with world.scheduler(
        memory_snapshot_cache_dir=l1_dir, warm_pool_size=0, disk_snapshot_cache_dir=world.private_dir("l2")
    ) as scheduler:
        info = await scheduler.describe_instance(ref.instance_id)
        assert info.host is not None
        digest = info.guest.packages_image
        assert digest is not None
        downloaded = world.asset_dir / f"{digest}.img"
        assert not downloaded.exists()  # every earlier load used the local L2 (same digest)
        async with await scheduler.reuse(ref.instance_id) as vm:
            assert_origin(vm, "reuse")
            await assert_packages_import(vm)
            await assert_restored(vm, model)
            await close_unsaved(vm)
        assert await asyncio.to_thread(_file_sha256, downloaded) == digest


@pytest.mark.timeout(300)
@pytest.mark.parametrize("language", list(Language), ids=[lang.value for lang in Language])
async def test_origin_pool(instance_world: InstanceWorld, primed_l1_dir: Path, language: Language) -> None:
    """The same walk from a warm-pool VM (itself restored from L1)."""
    world = instance_world
    async with world.scheduler(
        memory_snapshot_cache_dir=primed_l1_dir, warm_pool_size=1, disk_snapshot_cache_dir=None
    ) as scheduler:
        await scheduler.wait_pool_ready()  # returns silently on timeout: the origin check below decides
        session = await scheduler.new(VmSpec(language=language))
        assert_origin(session, "pool")
        model = new_model(language, seed="pool")
        ref = await _walk(scheduler, world, session, model)
        await reuse_and_check(scheduler, ref, model)
        assert world.instance_folders() == []

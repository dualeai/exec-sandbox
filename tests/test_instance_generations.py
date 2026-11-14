"""Many generations of one saved instance on real VMs: each save is exact.

From an L1-restored VM, saves alternate between a save in the running session
and a save after reuse(). After each reuse the guest equals the model of the
last save, and the store's head names the same generation as the session.
"""

from pathlib import Path

import pytest

from exec_sandbox import InstanceRef, Language, VmSpec
from tests.conftest import skip_unless_hwaccel
from tests.instance_fixtures import (
    InstanceWorld,
    assert_origin,
    assert_restored,
    close_saved,
    instance_world_fixture,  # noqa: F401 - fixture "instance_world"
    new_model,
    play_round,
    primed_l1_dir_fixture,  # noqa: F401 - fixture "primed_l1_dir"
    reuse_and_check,
    start_guest,
)

pytestmark = skip_unless_hwaccel


@pytest.mark.timeout(300)
@pytest.mark.parametrize("generations", [6, pytest.param(20, marks=pytest.mark.slow)], ids=["6", "20"])
@pytest.mark.parametrize("language", list(Language), ids=[lang.value for lang in Language])
async def test_generations_exact(
    instance_world: InstanceWorld, primed_l1_dir: Path, language: Language, generations: int
) -> None:
    """Odd generations save in the running session, even ones reuse() first."""
    world = instance_world
    async with world.scheduler(
        memory_snapshot_cache_dir=primed_l1_dir, warm_pool_size=0, disk_snapshot_cache_dir=None
    ) as scheduler:
        session = await scheduler.new(VmSpec(language=language))
        assert_origin(session, "l1")
        model = new_model(language, seed="generations")
        await start_guest(session, model, ticker=True)
        ref: InstanceRef | None = None
        try:
            for generation in range(1, generations + 1):
                if generation % 2 == 0:
                    assert ref is not None
                    await close_saved(session)
                    session = await scheduler.reuse(ref.instance_id)
                    assert_origin(session, "reuse")
                    assert session.instance == ref
                    info = await scheduler.describe_instance(ref.instance_id)
                    assert info.generation == ref.generation
                    await assert_restored(session, model)
                await play_round(session, model)
                saved = await session.save()
                instance_id = ref.instance_id if ref is not None else saved.instance_id
                assert saved == InstanceRef(instance_id=instance_id, generation=generation)
                assert session.instance == saved
                ref = saved
            await close_saved(session)
        finally:
            await session.close()
        assert ref is not None
        info = await scheduler.describe_instance(ref.instance_id)
        assert (info.generation, info.verified) == (generations, True)
        await reuse_and_check(scheduler, ref, model)
        assert world.instance_folders() == []

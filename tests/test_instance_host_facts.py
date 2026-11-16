"""The guest images a saved instance names: the bytes the VM runs on.

The VM records the digest of its base and L2 images when QEMU opens them
(VmManager._launch_qemu_vm). A save names those digests, not the files at the
same paths at save time: an L2 entry evicted or rebuilt since holds other bytes.

Also: a save whose QMP command QEMU does not answer carries the VM's deadline diagnostics.

Each test saves through the real InstanceManager.save on a fake VM and its QMP peer on a real
Unix socket (tests/instance_support.py): no QEMU.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
from typing import TYPE_CHECKING, Any

import pytest

from exec_sandbox import constants, system_probes
from exec_sandbox.admission import ResourceAdmissionController
from exec_sandbox.exceptions import ConfigError, ErrorCode, RouteError, RouteKey, TransientError
from exec_sandbox.qemu_cmd import CounterRate
from tests.instance_support import (
    BLOCK,
    BUDGET_KEYS,
    FAKE_COUNTER_HZ,
    FAKE_LAYOUT,
    NO_VM_TIMEOUT,
    PYTHON,
    FakeQmp,
    Saver,
    SparseImage,
    no_packages,
    qemu_file,
    worker_without_base_image,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.instance_support import FakeVm, InstanceEnv

pytestmark = NO_VM_TIMEOUT


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _flip(path: Path) -> None:
    """Rebuild the file at the same path: other bytes, the same size."""
    path.write_bytes(bytes(b ^ 0xFF for b in path.read_bytes()))


async def _vm_on_a_layer(env: InstanceEnv, tmp_path: Path) -> FakeVm:
    """A fake VM on the env's base image and an L2 layer, both digests recorded at its start."""
    vm = await env.new_vm(SparseImage(size=BLOCK), SparseImage(size=BLOCK), spec=PYTHON)
    layer = tmp_path / "l2.img"
    layer.write_bytes(b"l2 layer".ljust(BLOCK, b"\0"))
    vm.packages_image, vm.packages_image_digest = layer, _sha256(layer.read_bytes())
    return vm


async def test_facts_name_the_bytes_the_vm_started_on(instance_env: InstanceEnv, tmp_path: Path) -> None:
    """Both images are rebuilt at the same paths between two saves: the second head still names
    the digests of the bytes the VM started on."""
    env = instance_env
    vm = await _vm_on_a_layer(env, tmp_path)
    assert vm.packages_image is not None
    started_on = (_sha256(env.base_image.read_bytes()), _sha256(vm.packages_image.read_bytes()))
    saver = Saver(env, vm, manager=env.manager, state=None)
    first = await saver.save()  # stores both images under their digests
    assert (first.body.guest.base_image, first.body.guest.packages_image) == started_on

    _flip(env.base_image)
    _flip(vm.packages_image)
    second = await saver.save()

    assert second.header.generation == 2
    assert (second.body.guest.base_image, second.body.guest.packages_image) == started_on


@pytest.mark.parametrize("unknown", ["base", "l2"])
async def test_unknown_digest_is_image_damaged(instance_env: InstanceEnv, tmp_path: Path, unknown: str) -> None:
    """An image missing at the VM's start has no digest: ConfigError IMAGE_DAMAGED, not a
    raw ValidationError of a "missing" digest, and the store gets nothing."""
    env = instance_env
    vm = await _vm_on_a_layer(env, tmp_path)
    if unknown == "base":
        vm.base_image_digest = None
    else:
        vm.packages_image_digest = None

    with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.IMAGE_DAMAGED) as exc:
        await env.manager.save(vm, None, new_instance_id=None, resumed=lambda: None)  # type: ignore[arg-type]

    assert exc.value.message == (
        f"VM {vm.vm_id}: the digest of a guest file it runs on is unknown (the file was missing "
        "when the VM started). It was not saved."
    )
    assert await env.store.list_objects("") == []


async def test_unknown_machine_type_is_not_saved(instance_env: InstanceEnv, tmp_path: Path) -> None:
    """A VM whose machine type the probe could not resolve: ConfigError HOST_DEPENDENCY, and the
    store gets nothing (a load starts QEMU with that type)."""
    vm = await _vm_on_a_layer(instance_env, tmp_path)
    vm.machine_type = None
    with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc:
        await instance_env.manager.save(vm, None, new_instance_id=None, resumed=lambda: None)  # type: ignore[arg-type]
    assert exc.value.message == (
        f"VM {vm.vm_id}: the QEMU machine type it runs is unknown (this QEMU does not list the "
        "type it started with). It was not saved."
    )
    assert await instance_env.store.list_objects("") == []


async def test_unreadable_qemu_version_is_not_saved(
    instance_env: InstanceEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker whose QEMU version cannot be read: ConfigError HOST_DEPENDENCY, and the store
    gets nothing (a load compares its QEMU with the saved version)."""
    vm = await _vm_on_a_layer(instance_env, tmp_path)
    qemu_file(tmp_path, monkeypatch)

    async def no_version() -> None:
        return None

    monkeypatch.setattr("exec_sandbox.system_probes.probe_qemu_version", no_version)
    monkeypatch.setattr("exec_sandbox.instance.manager.require_qemu_version", system_probes.require_qemu_version)
    monkeypatch.setattr("exec_sandbox.worker_facts.require_qemu_version", system_probes.require_qemu_version)
    with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="QEMU version"):
        await instance_env.manager.save(vm, None, new_instance_id=None, resumed=lambda: None)  # type: ignore[arg-type]
    assert await instance_env.store.list_objects("") == []


async def test_a_load_without_the_l2_image_routes_to_both_images(instance_env: InstanceEnv, tmp_path: Path) -> None:
    """No L2 image on this worker and none in the store: ROUTE to a worker that holds the L2
    image and the base image it lies on (`found` gives this worker's base image)."""
    env = instance_env
    saver = Saver(env, await _vm_on_a_layer(env, tmp_path), manager=env.manager, state=None)
    guest = (await saver.save()).body.guest
    env.bucket.delete(f"assets/sha256/{guest.packages_image}")
    await env.manager.drain()  # the save's old files go in a worker thread

    with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
        await env.load(saver.instance_id, manager=env.manager)

    required = exc.value.required
    assert (required[RouteKey.BASE_IMAGE], required[RouteKey.PACKAGES_IMAGE]) == (
        guest.base_image,
        guest.packages_image,
    )
    assert exc.value.found == {**required, RouteKey.PACKAGES_IMAGE: None, **dict.fromkeys(BUDGET_KEYS)}
    assert exc.value.context["reason"] == "the image is not on this worker and not in the store"


async def test_a_vm_above_the_whole_budget_routes_before_the_download(
    instance_env: InstanceEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker whose whole budget can never hold the saved VM: ROUTE (CAPACITY_NEVER_FITS)
    after the host check and before any download, with the checked facts and the budget.
    `found` gives this worker's own values: its counter_hz "any" (TSC scaling), not the
    saved rate it accepted."""
    env = instance_env
    saver = Saver(env, await _vm_on_a_layer(env, tmp_path), manager=env.manager, state=None)
    host = (await saver.save()).body.host
    await env.manager.drain()  # the save's old files go in a worker thread
    env.vm_manager.budget = ResourceAdmissionController(  # 100 MB * 0.9: below any VM
        disk_dir=tmp_path,
        disk_free_floor_mb=0,
        memory_overcommit_ratio=1.0,
        cpu_overcommit_ratio=1.0,
        disk_overcommit_ratio=constants.DEFAULT_DISK_OVERCOMMIT_RATIO,
        host_memory_reserve_ratio=0.1,
        host_memory_mb=100.0,
        host_cpu_count=8.0,
        available_memory_floor_mb=0,
        host_cpu_reserve_cores=0.0,
    )
    files = env.work_files()

    async def scaling(_arch: object, _accel: object) -> CounterRate:
        return CounterRate(hz=FAKE_COUNTER_HZ, any_rate=True)

    monkeypatch.setattr("exec_sandbox.worker_facts.counter_rate", scaling)
    with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.CAPACITY_NEVER_FITS) as exc:
        await env.load(saver.instance_id, manager=env.manager)

    assert exc.value.instance_id == saver.instance_id
    assert exc.value.required[RouteKey.MACHINE_TYPE] == host.machine_type
    assert exc.value.required[RouteKey.MEMORY_MB] == "272"  # 192 MB and 80 MB of overhead
    assert exc.value.found[RouteKey.MEMORY_MB] == "90"
    assert (exc.value.required[RouteKey.COUNTER_HZ], exc.value.found[RouteKey.COUNTER_HZ]) == (
        str(FAKE_COUNTER_HZ),
        "any",
    )
    assert env.work_files() == files  # nothing was downloaded


class _NoAnswerToStop(FakeQmp):
    """A QMP peer whose QEMU main loop is blocked: it never answers ``stop`` (until it is stopped)."""

    def __init__(self) -> None:
        super().__init__(on_migrate=None, on_run=None)
        self.blocked = asyncio.Event()

    async def _reply(self, command: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if command == "stop":
            await self.blocked.wait()
        return await super()._reply(command, arguments)

    async def stop(self) -> None:
        self.blocked.set()
        await super().stop()


@pytest.mark.parametrize("qemu_answers", [False, True], ids=["qmp-deadline", "stop-refused"])
async def test_a_save_that_qemu_does_not_answer_carries_deadline_diagnostics(
    instance_env: InstanceEnv, monkeypatch: pytest.MonkeyPatch, qemu_answers: bool
) -> None:
    """A QMP command of the save that times out (QEMU sent no answer: a stop that never
    returns) gets the VM's deadline diagnostics in the error context (run state, QEMU CPU
    time and threads, console); a save that QEMU refuses gets none. The QMP command deadline
    is 0.2 s here, not 10 s."""
    env = instance_env
    monkeypatch.setattr("exec_sandbox.migration_client._QMP_TIMEOUT", 0.2)
    vm = await env.new_vm(SparseImage(size=BLOCK), SparseImage(size=BLOCK), spec=PYTHON)
    if qemu_answers:
        vm.qmp.refused["stop"] = "the VM is busy"
    else:
        await vm.qmp.stop()
        vm.qmp = _NoAnswerToStop()
        await vm.qmp.start(vm.qmp_socket)
    asked: list[str] = []

    async def deadline_diagnostics(*, operation: str) -> dict[str, Any]:
        """QemuVM.deadline_diagnostics (its own tests: tests/test_deadline_diagnostics.py)."""
        asked.append(operation)
        return {"qemu_status": "running", "qemu_cpu_seconds": 1.5}

    vm.deadline_diagnostics = deadline_diagnostics  # type: ignore[attr-defined]

    with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.MIGRATION_FAILED) as exc:
        await env.manager.save(vm, None, new_instance_id=None, resumed=lambda: None)  # type: ignore[arg-type]

    if qemu_answers:
        assert asked == []
        assert "qemu_status" not in exc.value.context
    else:
        assert isinstance(exc.value.__cause__, TimeoutError)
        assert asked == ["save"]
        assert (exc.value.context["qemu_status"], exc.value.context["qemu_cpu_seconds"]) == ("running", 1.5)
    assert await env.store.list_objects("") == []


@pytest.mark.parametrize("worker", ["has-the-image", "store-has-it", "nowhere"])
async def test_a_check_downloads_nothing(instance_env: InstanceEnv, tmp_path: Path, worker: str) -> None:
    """InstanceManager.check (Scheduler.check_reuse): what a load checks before its download,
    with no body or image download: passes when the image is on this worker or in the store,
    else ROUTE as the load would."""
    env = instance_env
    saver = Saver(env, await _vm_on_a_layer(env, tmp_path), manager=env.manager, state=None)
    guest = (await saver.save()).body.guest
    await env.manager.drain()  # the save's old files go in a worker thread
    if worker != "has-the-image":
        env.vm_manager.base_image = None
    if worker == "nowhere":
        env.bucket.delete(f"assets/sha256/{guest.base_image}")
    files = env.work_files()
    mark = len(env.proxy.log)

    if worker == "nowhere":
        with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
            await env.manager.check(saver.instance_id, packages_image=no_packages)
        assert exc.value.required[RouteKey.BASE_IMAGE] == guest.base_image
        assert exc.value.instance_id == saver.instance_id
    else:
        await env.manager.check(saver.instance_id, packages_image=no_packages)

    assert env.proxy.count(("GET", "GET_RANGE"), "/b/", since=mark) == 0  # no body object
    assert env.proxy.count(("GET", "GET_RANGE"), "^assets/", since=mark) == 0  # no image
    assert env.work_files() == files


async def test_a_loaded_image_is_never_evicted(instance_env: InstanceEnv, tmp_path: Path) -> None:
    """The images a load fetched stay in the asset cache under admission's eviction while the
    state (then the VM) holds them; after release, an eviction can take them."""
    env = instance_env
    saver = Saver(env, await _vm_on_a_layer(env, tmp_path), manager=env.manager, state=None)
    guest = (await saver.save()).body.guest
    await env.manager.drain()
    worker = worker_without_base_image(env, root=tmp_path / "worker-b")

    state = await env.load(saver.instance_id, manager=worker)
    assert guest.packages_image is not None
    cached = {worker.assets.cache_dir / f"{digest}.img" for digest in (guest.base_image, guest.packages_image)}
    assert {state.base_image, state.packages_image} == cached
    assert {lease.path for lease in state.asset_leases} == cached
    assert await worker.assets.eviction_order().evict_next() is None  # in use: skipped
    assert all(path.exists() for path in cached)

    await worker.release(state, wait=True)
    evictions = worker.assets.eviction_order()
    assert [(await evictions.evict_next() or 0) > 0 for _ in cached] == [True, True]  # their bytes
    assert not any(path.exists() for path in cached)


async def test_the_save_keeps_the_egress_policy_of_the_vm(instance_env: InstanceEnv, tmp_path: Path) -> None:
    """A save stores the domains the VM could reach (its gvproxy's list) beside the spec: a
    load and describe give that list, whatever the default of the release that loads it; the
    spec stays as the user gave it."""
    env = instance_env
    vm = await _vm_on_a_layer(env, tmp_path)
    vm.egress_domains = ("a.example",)
    saver = Saver(env, vm, manager=env.manager, state=None)
    saved = await saver.save()
    assert (saved.body.egress_domains, saved.body.spec) == (("a.example",), vm.spec)

    loaded = await env.load(saver.instance_id, manager=env.manager)
    assert loaded.body.egress_domains == ("a.example",)
    await env.manager.release(loaded, wait=True)
    assert (await env.manager.describe(saver.instance_id)).egress_domains == ("a.example",)


async def test_a_check_with_a_disk_error_is_local_disk(
    instance_env: InstanceEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asset cache's lock file cannot be opened (no file handle left): the check raises
    TransientError LOCAL_DISK, as a load does, not a raw OSError."""
    env = instance_env
    saver = Saver(env, await _vm_on_a_layer(env, tmp_path), manager=env.manager, state=None)
    await saver.save()
    env.vm_manager.base_image = None  # the image would come from the asset cache

    async def no_handle(_self: object) -> None:
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr("exec_sandbox.instance.manager.CacheEntry.lease", no_handle)
    with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK) as exc:
        await env.manager.check(saver.instance_id, packages_image=no_packages)
    assert exc.value.context == {"instance_id": saver.instance_id, "errno": errno.EMFILE}


async def test_a_failed_load_releases_the_images_it_leased(instance_env: InstanceEnv, tmp_path: Path) -> None:
    """The base image comes from the asset cache, the L2 image is nowhere: the load ROUTEs and
    releases its lease on the base image, which an eviction can then take."""
    env = instance_env
    saver = Saver(env, await _vm_on_a_layer(env, tmp_path), manager=env.manager, state=None)
    guest = (await saver.save()).body.guest
    await env.manager.drain()
    worker = worker_without_base_image(env, root=tmp_path / "worker-b")
    env.bucket.delete(f"assets/sha256/{guest.packages_image}")

    with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH):
        await env.load(saver.instance_id, manager=worker)
    cached = worker.assets.cache_dir / f"{guest.base_image}.img"
    assert cached.exists()
    assert (await worker.assets.eviction_order().evict_next() or 0) > 0
    assert not cached.exists()


async def test_a_check_releases_its_lease(instance_env: InstanceEnv, tmp_path: Path) -> None:
    """A check that finds the image in the asset cache leases it while it looks, then releases
    it: an eviction can take it after."""
    env = instance_env
    saver = Saver(env, await _vm_on_a_layer(env, tmp_path), manager=env.manager, state=None)
    guest = (await saver.save()).body.guest
    await env.manager.drain()
    worker = worker_without_base_image(env, root=tmp_path / "worker-b")
    await worker.release(await env.load(saver.instance_id, manager=worker), wait=True)  # fills the asset cache

    await worker.check(saver.instance_id, packages_image=no_packages)
    evictions = worker.assets.eviction_order()
    assert [(await evictions.evict_next() or 0) > 0 for _ in range(2)] == [True, True]  # base and L2
    assert not (worker.assets.cache_dir / f"{guest.base_image}.img").exists()


async def test_two_loads_of_one_image_download_it_once(instance_env: InstanceEnv, tmp_path: Path) -> None:
    """Two loads at once on a worker without the images: the second waits for the first's
    download (the entry's save lock), then leases the same file: the store serves the images
    once, as for one load."""
    env = instance_env
    saver = Saver(env, await _vm_on_a_layer(env, tmp_path), manager=env.manager, state=None)
    await saver.save()
    await env.manager.drain()

    def downloads(since: int) -> int:
        return env.proxy.count(("GET", "GET_RANGE"), "^assets/", since=since)

    one = worker_without_base_image(env, root=tmp_path / "worker-one")
    mark = len(env.proxy.log)
    await one.release(await env.load(saver.instance_id, manager=one), wait=True)
    single = downloads(mark)
    assert single > 0

    two = worker_without_base_image(env, root=tmp_path / "worker-two")
    mark = len(env.proxy.log)
    states = await asyncio.gather(*(env.load(saver.instance_id, manager=two) for _ in range(2)))
    assert downloads(mark) == single
    assert len({state.base_image for state in states}) == 1
    for state in states:
        await two.release(state, wait=True)


async def test_another_device_layout_routes_before_the_download(instance_env: InstanceEnv, tmp_path: Path) -> None:
    """The save stores the VM's device layout; a load whose command would have another layout
    (a release that moved a disk to another slot) ROUTEs to another release (empty
    `required`) before any download, and names the first line that differs."""
    env = instance_env
    vm = await _vm_on_a_layer(env, tmp_path)
    saver = Saver(env, vm, manager=env.manager, state=None)
    host = (await saver.save()).body.host
    assert host.device_layout == vm.device_layout
    await env.manager.drain()
    env.vm_manager.layout = FAKE_LAYOUT.replace("serial=base", "serial=home")
    files = env.work_files()
    mark = len(env.proxy.log)

    with pytest.raises(RouteError, check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH) as exc:
        await env.load(saver.instance_id, manager=env.manager)

    assert (exc.value.required, exc.value.instance_id) == ({}, saver.instance_id)
    # The command of this restore: the saved spec and machine type, with the saved L2 layer
    assert env.vm_manager.layout_calls == [(vm.spec, host.machine_type, True)]
    assert str(exc.value) == (
        f"Instance {saver.instance_id}: this release runs another device layout for it (saved "
        "'-device virtio-blk-device drive=hd0 serial=base', here '-device virtio-blk-device drive=hd0 serial=home')."
    )
    assert env.proxy.count(("GET", "GET_RANGE"), "/b/", since=mark) == 0  # no body object
    assert env.work_files() == files

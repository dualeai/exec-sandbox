"""Saved instances end to end: real VMs, a moto S3 store.

new -> exec -> save -> close -> reuse: the Python state and /home/user come back.
"""

import asyncio
import errno
import hashlib
import json
import random
import shutil
import tempfile
import threading
import time
import uuid
import warnings
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import pytest

from exec_sandbox import InstanceNotSavedWarning, Scheduler, SchedulerConfig, VmSpec
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InstanceGoneError,
    InvalidInputError,
    OutcomeUnknownError,
    RouteError,
    TransientError,
)
from exec_sandbox.instance import manager as instance_manager
from exec_sandbox.instance.keyring import Keyring
from exec_sandbox.models import InstanceRef, Language
from exec_sandbox.platform_utils import detect_host_arch
from exec_sandbox.qemu_cmd import resolve_machine_type
from tests.conftest import skip_unless_hwaccel
from tests.instance_support import await_until, disk_full_once, new_keyring
from tests.s3_faults import BUCKET, After, Bucket, DropAfter, Edit, Fail, Hold, Request, Response, S3Proxy

pytestmark = skip_unless_hwaccel

_PY = VmSpec(language=Language.PYTHON)
_HEAD = r"^s/[0-9a-f]{32}/head$"
_BODY = r"^s/[0-9a-f]{32}/b/"


def _config(base: SchedulerConfig, endpoint_url: str, prefix: str, keyring: str) -> SchedulerConfig:
    """`base` plus the S3 store on the moto bucket, an instance prefix and a keyring (validated, so
    fields count as set)."""
    return SchedulerConfig.model_validate(
        {
            **base.model_dump(exclude_unset=True),
            "instance_store": "s3",
            "s3_bucket": BUCKET,
            "s3_endpoint_url": endpoint_url,
            "s3_instance_prefix": prefix,
            "instance_keyring": keyring,
        }
    )


@pytest.fixture
def home_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """EXEC_SANDBOX_HOME_DISK_DIR of this test: each VmManager's folder, with its instances/ work
    folder, is under it."""
    root = Path(tempfile.mkdtemp(prefix="hd-"))  # short and 0700 (PrivateRunDir checks the root)
    monkeypatch.setenv("EXEC_SANDBOX_HOME_DISK_DIR", str(root))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def asset_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """EXEC_SANDBOX_INSTANCE_ASSET_DIR of this test: guest images a worker downloaded by digest."""
    folder = tmp_path / "assets"
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_ASSET_DIR", str(folder))
    return folder


@pytest.fixture
async def store(
    scheduler_config: SchedulerConfig, s3_proxy: S3Proxy, home_root: Path, asset_dir: Path
) -> AsyncGenerator[tuple[Scheduler, str, object]]:
    """A Scheduler with saved instances on (moto bucket behind the fault proxy, fresh prefix); yields
    (scheduler, prefix, s3 client). A test that injects store faults also takes `s3_proxy` (the same
    proxy); one that checks local files takes `home_root` or `asset_dir`."""
    prefix = s3_proxy.prefix
    async with Scheduler(_config(scheduler_config, s3_proxy.endpoint_url, prefix, new_keyring())) as scheduler:
        yield scheduler, prefix, s3_proxy.bucket.s3


@pytest.fixture
def images_without_python_base(images_dir: Path, tmp_path: Path) -> Path:
    """The built images, except the Python base images: the images folder of a worker that lacks them."""
    folder = tmp_path / "images-without-python-base"
    folder.mkdir()
    for image in images_dir.iterdir():
        if not image.name.startswith("python-"):
            (folder / image.name).symlink_to(image)
    return folder


def _worker_config(
    scheduler: Scheduler, scheduler_config: SchedulerConfig, s3_proxy: S3Proxy, *, images_dir: Path
) -> SchedulerConfig:
    """Another worker on the same store and keyring as `scheduler`, with its own images folder."""
    keyring = scheduler.config.instance_keyring
    assert keyring is not None
    base = SchedulerConfig.model_validate({**scheduler_config.model_dump(exclude_unset=True), "images_dir": images_dir})
    return _config(base, s3_proxy.endpoint_url, s3_proxy.prefix, keyring.get_secret_value())


def _instance_folders(home_root: Path, instance_id: str) -> list[Path]:
    """Local work folders of one instance, in every VmManager of this test."""
    return sorted(home_root.glob(f"vmm-*/instances/{instance_id}-*"))


def _objects(s3: object, prefix: str, instance_id: str) -> list[str]:
    listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{prefix}s/{instance_id}/b/")  # type: ignore[attr-defined]
    return sorted(o["Key"].rsplit("/", 1)[1] for o in listing.get("Contents", []))


class TestSaveAndReuse:
    async def test_state_and_home_come_back(self, store: tuple[Scheduler, str, object]) -> None:
        scheduler, prefix, s3 = store
        async with await scheduler.new(_PY) as vm:
            assert vm.instance is None
            await vm.exec("x = 41")
            await vm.exec("open('/home/user/note.txt', 'w').write('kept')")
            first = await vm.save()
            assert first.generation == 1
            assert vm.instance == first
            await vm.exec("x = 'after the save, not saved'")

        async with await scheduler.reuse(first.instance_id) as vm:
            assert vm.instance == first
            assert vm.startup_timing.instance_load_ms is not None
            assert vm.startup_timing.instance_load_ms > 0  # the download and rebuild of the saved state
            result = await vm.exec("x += 1; print(x, open('/home/user/note.txt').read())")
            assert result.stdout.strip() == "42 kept", result.stderr
            second = await vm.save()
            assert (second.instance_id, second.generation) == (first.instance_id, 2)

        kinds = [name.rsplit(".", 1)[1] for name in _objects(s3, prefix, first.instance_id)]
        assert sorted(kinds) == ["f", "p"], kinds  # the second save is a pack on the full copy

        async with await scheduler.reuse(first.instance_id) as vm:
            result = await vm.exec("print(x)")
            assert result.stdout.strip() == "42"
            assert vm.instance == second


class TestErrors:
    async def test_unknown_and_invalid_ids(self, store: tuple[Scheduler, str, object]) -> None:
        """A valid id that was never saved (library-made or caller-chosen) is not found; an id
        outside the pattern is refused before any request."""
        scheduler, _, _ = store
        for unknown in (uuid.uuid4().hex, "never-saved"):
            with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND) as exc:
                await scheduler.reuse(unknown)
            assert exc.value.action == "RECREATE"
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.INSTANCE_INVALID_ID):
            await scheduler.reuse("Not An Id")

    async def test_save_without_store(self, scheduler: Scheduler) -> None:
        async with await scheduler.new(_PY) as vm:
            with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
                await vm.save()
            assert not vm.closed  # nothing happened: the session stays usable


class TestBudget:
    async def test_large_change_writes_a_new_full_copy_and_drops_the_old_chain(
        self, store: tuple[Scheduler, str, object]
    ) -> None:
        """A pack bigger than the full copy (the budget) is replaced by a new full copy. The change is
        sized from the stored full copy, so the test holds whatever that copy's size is."""
        scheduler, prefix, s3 = store
        bucket = Bucket(s3, prefix)
        async with await scheduler.new(_PY) as vm:
            await vm.exec("x = 1")
            ref = await vm.save()
            (first,) = bucket.body_keys(ref.instance_id)
            # Random bytes do not compress: more of them than the full copy's size outgrow the budget.
            mib = bucket.size(first) // (1024 * 1024) + 8
            write = f"import os\nwith open('/home/user/big.bin', 'wb') as f:\n    for _ in range({mib}): f.write(os.urandom(1 << 20))"
            assert (await vm.exec(write)).exit_code == 0
            assert (await vm.save()).generation == 2
        now = bucket.body_keys(ref.instance_id)
        assert [n.rsplit(".", 1)[1] for n in now] == ["f"], now
        assert now != [first]  # the old full copy was deleted after the new one committed
        async with await scheduler.reuse(ref.instance_id) as vm:
            result = await vm.exec("import os; print(x, os.path.getsize('/home/user/big.bin'))")
            assert result.stdout.split() == ["1", str(mib * 1024 * 1024)]


class TestSaveFailures:
    async def test_store_down_keeps_the_session_open(
        self, store: tuple[Scheduler, str, object], s3_proxy: S3Proxy
    ) -> None:
        """Nothing was committed: the VM still runs, and a later save can succeed."""
        scheduler, _, _ = store
        async with await scheduler.new(_PY) as vm:
            await vm.exec("x = 1")
            down = s3_proxy.add(("PUT", "MPU_CREATE"), _BODY, Fail(503, "ServiceUnavailable"), times=None)
            mark = len(s3_proxy.log)
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
                await vm.save()
            assert exc.value.action == "RETRY"
            assert down.hits >= 1
            assert s3_proxy.count("PUT_IF", _HEAD, since=mark) == 0  # no head write
            down.remove()
            assert not vm.closed
            assert vm.instance is None
            assert (await vm.exec("x += 1; print(x)")).stdout.strip() == "2"
            ref = await vm.save()
        assert ref.generation == 1
        async with await scheduler.reuse(ref.instance_id) as vm:
            assert (await vm.exec("print(x)")).stdout.strip() == "2"

    async def test_unknown_outcome_keeps_the_session_open(
        self, store: tuple[Scheduler, str, object], s3_proxy: S3Proxy
    ) -> None:
        """No answer to the head write and no read-back: unknown outcome; the VM runs on and the
        session stays open."""
        scheduler, _, _ = store
        async with await scheduler.new(_PY) as vm:
            s3_proxy.add("PUT_IF", _HEAD, DropAfter(), times=None)
            s3_proxy.add("GET", _HEAD, Fail(503, "ServiceUnavailable"), times=None)
            with pytest.raises(
                OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN
            ) as exc:
                await vm.save()
            assert exc.value.action == "VERIFY"
            assert not vm.closed
            assert (await vm.exec("print(1)")).stdout == "1\n"

    async def test_local_disk_full_keeps_the_session_open(
        self, store: tuple[Scheduler, str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ENOSPC while sealing: a typed, retryable error; nothing saved; the VM runs on."""
        scheduler, _, _ = store
        async with await scheduler.new(_PY) as vm:
            await vm.exec("x = 1")
            disk_full_once(monkeypatch, step="seal")
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                await vm.save()
            assert not vm.closed
            assert vm.instance is None
            assert (await vm.save()).generation == 1

    async def test_cancelled_save_stops_the_home_copy_first(
        self, store: tuple[Scheduler, str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A save cancelled while the VM is stopped and its home disk is copied: the copy stops
        (at its next chunk) and the cancel comes out once it ended. A copy still running would
        read the disk that the resumed guest writes, into the folder that the save removes.
        The private seam of disk_full_once: the manager module's name for files.disk.clone_sparse."""
        scheduler, _, _ = store
        copying, ended = threading.Event(), threading.Event()

        def stoppable_copy(_source: Path, target: Path, *, stop: threading.Event | None) -> bool:
            target.write_bytes(b"partial")
            copying.set()
            assert stop is not None
            stop.wait(timeout=30)
            target.unlink()
            ended.set()
            raise InterruptedError(errno.EINTR, "The copy was stopped")

        monkeypatch.setattr(instance_manager, "clone_sparse", stoppable_copy)
        async with await scheduler.new(_PY) as vm:
            await vm.exec("x = 1")
            saving = asyncio.create_task(vm.save())
            assert await asyncio.to_thread(copying.wait, 30)
            started = time.monotonic()
            saving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await saving
            assert ended.is_set()  # the copy ended before the cancel came out
            assert time.monotonic() - started < 10  # the copy stopped: a waited copy takes 30 s

    async def test_scheduler_exit_waits_for_a_save_in_flight(
        self, store: tuple[Scheduler, str, object], scheduler_config: SchedulerConfig, s3_proxy: S3Proxy
    ) -> None:
        """The store records, when it applies the head write, whether the exit had already ended."""
        _, prefix, _ = store
        endpoint_url = s3_proxy.endpoint_url
        keyring = new_keyring()
        scheduler = Scheduler(_config(scheduler_config, endpoint_url, prefix, keyring))
        await scheduler.__aenter__()
        vm = await scheduler.new(_PY)
        await vm.exec("x = 'saved during shutdown'")
        closing: list[asyncio.Task[None]] = []
        exited_before_commit: list[bool] = []

        async def head_applied(req: Request, resp: Response) -> None:
            exited_before_commit.append(closing[0].done())

        uploading = s3_proxy.add(("PUT", "MPU_CREATE"), _BODY, Hold())
        s3_proxy.add("PUT_IF", _HEAD, After(head_applied))
        saving = asyncio.create_task(vm.save())
        await uploading.reached.wait()
        closing.append(asyncio.create_task(scheduler.__aexit__(None, None, None)))
        uploading.release()
        ref = await saving
        await closing[0]
        assert exited_before_commit == [False]
        async with (
            Scheduler(_config(scheduler_config, endpoint_url, prefix, keyring)) as again,
            await again.reuse(ref.instance_id) as vm,
        ):
            assert (await vm.exec("print(x)")).stdout.strip() == "saved during shutdown"

    async def test_close_during_save_leaves_no_files(
        self, store: tuple[Scheduler, str, object], s3_proxy: S3Proxy, home_root: Path
    ) -> None:
        """close() while the upload runs: the snapshot was taken, the save still lands,
        and the save (not close) removes the instance folder."""
        scheduler, _, _ = store
        vm = await scheduler.new(_PY)
        first = await vm.save()  # the second save shares this save's folder
        uploading = s3_proxy.add(("PUT", "MPU_CREATE"), _BODY, Hold())
        saving = asyncio.create_task(vm.save())
        await uploading.reached.wait()
        await vm.close()
        uploading.release()
        second = await saving
        assert (second.instance_id, second.generation) == (first.instance_id, 2)
        assert _instance_folders(home_root, first.instance_id) == []

    async def test_cancelled_save_still_removes_its_object_on_conflict(
        self, store: tuple[Scheduler, str, object], s3_proxy: S3Proxy
    ) -> None:
        """The commit runs on after the caller is cancelled, and so does its cleanup: another
        session commits while this save's head write is held, so the held write conflicts."""
        scheduler, prefix, s3 = store
        vm = await scheduler.new(_PY)
        ref = await vm.save()
        other = await scheduler.reuse(ref.instance_id)
        try:
            before = _objects(s3, prefix, ref.instance_id)
            committing = s3_proxy.add("PUT_IF", _HEAD, Hold())
            saving = asyncio.create_task(vm.save())
            await committing.reached.wait()
            (mine,) = set(_objects(s3, prefix, ref.instance_id)) - set(before)  # the new body is uploaded
            saving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await saving
            assert (await other.save()).generation == 2  # lands while this save's head write waits
            committing.release()
            # The shielded commit and its cleanup end on their own.
            await await_until(lambda: mine not in _objects(s3, prefix, ref.instance_id))
            assert len(_objects(s3, prefix, ref.instance_id)) == 2  # the full copy and the other's pack
        finally:
            await vm.close()
            await other.close()
        assert s3_proxy.entries("PUT_IF", _HEAD)[-1].status == 412


class TestLoadFailures:
    async def test_worker_without_the_key_is_fix_config(
        self, store: tuple[Scheduler, str, object], scheduler_config: SchedulerConfig, moto_s3_server
    ) -> None:
        scheduler, prefix, _ = store
        async with await scheduler.new(_PY) as vm:
            ref = await vm.save()
        endpoint_url, _ = moto_s3_server
        async with Scheduler(_config(scheduler_config, endpoint_url, prefix, new_keyring())) as stranger:
            with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_KEY_MISSING) as exc:
                await stranger.reuse(ref.instance_id)
        assert exc.value.action == "FIX_CONFIG"

    async def test_one_false_404_is_retry_not_expired(
        self, store: tuple[Scheduler, str, object], s3_proxy: S3Proxy
    ) -> None:
        scheduler, _, _ = store
        async with await scheduler.new(_PY) as vm:
            ref = await vm.save()
        lie = s3_proxy.add("HEAD", _BODY, Fail(404, "NoSuchKey"))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.INSTANCE_STORE_UNAVAILABLE) as exc:
            await scheduler.reuse(ref.instance_id)
        assert exc.value.action == "RETRY"
        assert lie.hits == 1
        async with await scheduler.reuse(ref.instance_id) as vm:  # the store answers well again
            assert vm.instance == ref

    async def test_missing_full_copy_is_expired(self, store: tuple[Scheduler, str, object]) -> None:
        scheduler, prefix, s3 = store
        async with await scheduler.new(_PY) as vm:
            ref = await vm.save()
        (full,) = _objects(s3, prefix, ref.instance_id)
        s3.delete_object(Bucket=BUCKET, Key=f"{prefix}s/{ref.instance_id}/b/{full}")  # type: ignore[attr-defined]
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_EXPIRED) as exc:
            await scheduler.reuse(ref.instance_id)
        assert exc.value.action == "RECREATE"

    async def test_describe_reports_only_what_it_authenticated(
        self, store: tuple[Scheduler, str, object], s3_proxy: S3Proxy
    ) -> None:
        """A forged head on one read (generation 1 changed to 9): describe() raises
        INSTANCE_INTEGRITY and never gives the forged fields as authenticated."""
        scheduler, _, _ = store
        async with await scheduler.new(_PY) as vm:
            ref = await vm.save()

        def forged(resp: Response) -> None:
            assert b'"generation":1' in resp.body
            resp.body = resp.body.replace(b'"generation":1', b'"generation":9')

        rule = s3_proxy.add("GET", _HEAD, Edit(response=forged))
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY):
            await scheduler.describe_instance(ref.instance_id)
        assert rule.hits == 1


class TestChain:
    @pytest.mark.parametrize("fault", ["deleted", "one-bit-changed"])
    async def test_broken_lineage_at_save_writes_a_full_copy(
        self, store: tuple[Scheduler, str, object], fault: str
    ) -> None:
        """One LIST before each pack: the middle pack of a full copy and three packs, deleted or
        with one bit changed (same size, another ETag), makes the next save a new full copy, so
        the chain the head names always loads, with that save's state."""
        scheduler, prefix, s3 = store
        bucket = Bucket(s3, prefix)
        order: list[str] = []  # body keys, oldest first
        async with await scheduler.new(_PY) as vm:
            for n in range(4):
                await vm.exec(f"x = {n}")
                ref = await vm.save()
                (new,) = set(bucket.body_keys(ref.instance_id)) - set(order)
                order.append(new)
            assert [k.rsplit(".", 1)[1] for k in order] == ["f", "p", "p", "p"]
            if fault == "deleted":
                bucket.delete(order[2])
            else:
                bucket.flip(order[2], bucket.size(order[2]) // 2)
            await vm.exec("x = 'after the repair'")
            state = await vm.save()
        (full,) = bucket.body_keys(ref.instance_id)
        assert full.endswith(".f")
        assert full not in order
        async with await scheduler.reuse(ref.instance_id) as vm:
            assert vm.instance == state
            assert (await vm.exec("print(x)")).stdout.strip() == "after the repair"

    async def test_full_copy_gone_at_save_writes_a_new_one(self, store: tuple[Scheduler, str, object]) -> None:
        """The expiry rule removed the full copy: the next save writes a new full copy
        (not a pack on a chain that can no longer load) and the instance loads."""
        scheduler, prefix, s3 = store
        async with await scheduler.new(_PY) as vm:
            await vm.exec("x = 'kept'")
            ref = await vm.save()
            (full,) = _objects(s3, prefix, ref.instance_id)
            s3.delete_object(Bucket=BUCKET, Key=f"{prefix}s/{ref.instance_id}/b/{full}")  # type: ignore[attr-defined]
            await vm.save()
        objects = _objects(s3, prefix, ref.instance_id)
        assert [n.rsplit(".", 1)[1] for n in objects] == ["f"], objects
        assert objects != [full]
        async with await scheduler.reuse(ref.instance_id) as vm:
            assert (await vm.exec("print(x)")).stdout.strip() == "kept"

    async def test_new_newest_key_rewrites_the_chain(
        self, store: tuple[Scheduler, str, object], scheduler_config: SchedulerConfig, moto_s3_server
    ) -> None:
        """After a rotation the next save is a full copy under a new data key: the old key can be dropped."""
        scheduler, prefix, s3 = store
        old_key = scheduler.config.instance_keyring
        assert old_key is not None
        async with await scheduler.new(_PY) as vm:
            await vm.exec("x = 'kept across the rotation'")
            ref = await vm.save()
        old_objects = _objects(s3, prefix, ref.instance_id)

        endpoint_url, _ = moto_s3_server
        new_key = new_keyring()
        rotated = _config(scheduler_config, endpoint_url, prefix, f"{new_key},{old_key.get_secret_value()}")
        async with Scheduler(rotated) as worker, await worker.reuse(ref.instance_id) as vm:
            await vm.save()
        objects = _objects(s3, prefix, ref.instance_id)
        assert [n.rsplit(".", 1)[1] for n in objects] == ["f"]
        assert objects != old_objects

        async with Scheduler(_config(scheduler_config, endpoint_url, prefix, new_key)) as worker:
            info = await worker.describe_instance(ref.instance_id)
            assert (info.key_id, info.access) == (Keyring.parse(new_key).latest.id, "full")
            async with await worker.reuse(ref.instance_id) as vm:
                assert (await vm.exec("print(x)")).stdout.strip() == "kept across the rotation"

    async def test_load_fails_closed_real_vm(self, store: tuple[Scheduler, str, object], home_root: Path) -> None:
        """One changed bit in a pack (same size: only authentication finds it): reuse() fails with
        InstanceGoneError INSTANCE_INTEGRITY (RECREATE) before a VM starts, and leaves no instance
        files on the worker."""
        scheduler, prefix, s3 = store
        bucket = Bucket(s3, prefix)
        async with await scheduler.new(_PY) as vm:
            await vm.exec("x = 0")
            await vm.save()
            await vm.exec("x = 1")
            ref = await vm.save()
        (pack,) = [k for k in bucket.body_keys(ref.instance_id) if k.endswith(".p")]
        bucket.flip(pack, bucket.size(pack) // 2)
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_INTEGRITY) as exc:
            await scheduler.reuse(ref.instance_id)
        assert (exc.value.action, exc.value.instance_id) == ("RECREATE", ref.instance_id)
        # The failed load's folder goes in a worker thread (Scheduler exit waits for it)
        await await_until(lambda: _instance_folders(home_root, ref.instance_id) == [])


class TestAdmin:
    async def test_describe_list_delete(self, store: tuple[Scheduler, str, object]) -> None:
        scheduler, prefix, s3 = store
        async with await scheduler.new(_PY) as vm:
            ref = await vm.save()
        info = await scheduler.describe_instance(ref.instance_id)
        assert (info.instance_id, info.generation, info.access) == (ref.instance_id, 1, "full")
        assert info.spec == _PY
        assert info.host is not None and info.guest is not None
        assert info.guest.packages_image is None  # no packages: only the base image
        # The kernel and initramfs this worker booted the VM with, by digest (provenance): the
        # SHA-256 of their bytes, read here, not through the digest cache the VM manager uses
        images = scheduler._vm_manager.images  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage]
        assert (info.guest.kernel, info.guest.initramfs) == (
            hashlib.sha256(Path(images.kernel).read_bytes()).hexdigest(),
            hashlib.sha256(Path(images.initramfs).read_bytes()).hexdigest(),
        )
        # The machine type this QEMU resolved for the new VM (virt-11.1 for QEMU 11.1 on arm64)
        assert info.host.machine_type == await resolve_machine_type(detect_host_arch(), None)
        assert info.host.machine_type not in {None, "virt"}
        assert [iid async for iid in scheduler.list_instances()] == [ref.instance_id]

        assert await scheduler.delete_instance(ref.instance_id) is True
        assert _objects(s3, prefix, ref.instance_id) == []
        with pytest.raises(InstanceGoneError, check=lambda e: e.code is ErrorCode.INSTANCE_NOT_FOUND):
            await scheduler.reuse(ref.instance_id)
        assert await scheduler.delete_instance(ref.instance_id) is False  # idempotent
        assert [iid async for iid in scheduler.list_instances()] == []

    async def test_describe_without_the_key_has_no_access(
        self, store: tuple[Scheduler, str, object], scheduler_config: SchedulerConfig, moto_s3_server
    ) -> None:
        scheduler, prefix, _ = store
        async with await scheduler.new(_PY) as vm:
            ref = await vm.save()
        endpoint_url, _ = moto_s3_server
        async with Scheduler(_config(scheduler_config, endpoint_url, prefix, new_keyring())) as stranger:
            info = await stranger.describe_instance(ref.instance_id)
        assert (info.generation, info.access, info.writer_release, info.spec, info.host) == (
            1,
            "no_key",
            None,
            None,
            None,
        )


class TestNotSavedWarning:
    async def test_warns_only_for_unsaved_changes_of_an_instance(self, store: tuple[Scheduler, str, object]) -> None:
        scheduler, _, _ = store
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            async with await scheduler.new(_PY) as vm:
                await vm.exec("x = 1")  # never saved: a plain one-shot, no warning
            async with await scheduler.new(_PY) as vm:
                await vm.exec("x = 1")
                ref = await vm.save()  # saved after the last exec: no warning
            assert not [w for w in caught if issubclass(w.category, InstanceNotSavedWarning)]
            async with await scheduler.reuse(ref.instance_id) as vm:
                await vm.exec("x = 2")
        assert [str(w.message) for w in caught if issubclass(w.category, InstanceNotSavedWarning)] == [
            f"Session closed with changes since the last save of instance {ref.instance_id} (generation 1); "
            "call save(), or save(instance_id=...) under another id, to keep them."
        ]


class TestAssets:
    """Guest images travel with the instance, by digest: another worker can load it."""

    async def test_worker_without_the_image_loads_and_saves_by_digest(
        self,
        store: tuple[Scheduler, str, object],
        scheduler_config: SchedulerConfig,
        s3_proxy: S3Proxy,
        asset_dir: Path,
        images_without_python_base: Path,
    ) -> None:
        """A worker without the Python base image: reuse() fetches it by digest, and the next
        save records the image the VM runs on, not this worker's (missing) own."""
        scheduler, prefix, s3 = store
        async with await scheduler.new(_PY) as vm:
            await vm.exec("x = 'from another worker'")
            ref = await vm.save()
        saved = await scheduler.describe_instance(ref.instance_id)
        assert saved.host is not None
        key = f"{prefix}assets/sha256/{saved.guest.base_image}"
        assert s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"] > 0  # type: ignore[attr-defined]
        downloaded = asset_dir / f"{saved.guest.base_image}.img"
        assert not downloaded.exists()

        other = _worker_config(scheduler, scheduler_config, s3_proxy, images_dir=images_without_python_base)
        async with Scheduler(other) as worker:
            async with await worker.reuse(ref.instance_id) as vm:
                assert (await vm.exec("print(x)")).stdout.strip() == "from another worker"
                await vm.save()
            assert downloaded.exists()
            again = await worker.describe_instance(ref.instance_id)
        assert again.host is not None
        assert again.guest.base_image == saved.guest.base_image

    async def test_image_deleted_from_the_store_comes_back_at_the_next_save(
        self, store: tuple[Scheduler, str, object]
    ) -> None:
        scheduler, prefix, s3 = store
        async with await scheduler.new(_PY) as vm:
            ref = await vm.save()
            info = await scheduler.describe_instance(ref.instance_id)
            assert info.host is not None
            key = f"{prefix}assets/sha256/{info.guest.base_image}"
            s3.delete_object(Bucket=BUCKET, Key=key)  # type: ignore[attr-defined]
            await vm.save()
        assert s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"] > 0  # type: ignore[attr-defined]

    async def test_damaged_store_copy_routes(
        self,
        store: tuple[Scheduler, str, object],
        scheduler_config: SchedulerConfig,
        s3_proxy: S3Proxy,
        images_without_python_base: Path,
    ) -> None:
        scheduler, prefix, s3 = store
        async with await scheduler.new(_PY) as vm:
            ref = await vm.save()
        info = await scheduler.describe_instance(ref.instance_id)
        assert info.host is not None
        s3.put_object(Bucket=BUCKET, Key=f"{prefix}assets/sha256/{info.guest.base_image}", Body=b"x")  # type: ignore[attr-defined]
        other = _worker_config(scheduler, scheduler_config, s3_proxy, images_dir=images_without_python_base)
        async with Scheduler(other) as worker:
            with pytest.raises(
                RouteError,
                match="does not match its digest",
                check=lambda e: e.code is ErrorCode.INSTANCE_HOST_MISMATCH,
            ) as exc:
                await worker.reuse(ref.instance_id)
        assert exc.value.action == "ROUTE"


_PY_USE = "import os, random, ssl\n_ = (ssl.RAND_bytes(16), random.random())\nprint('ok')"
_PY_DRAW = "print(ssl.RAND_bytes(16).hex(), random.random(), os.urandom(16).hex())"
_JS_USE = (
    "globalThis.c = require('crypto'); c.randomBytes(16); c.randomUUID(); "
    "crypto.getRandomValues(new Uint8Array(16)); Math.random(); c.generateKeyPairSync('ed25519'); console.log('ok')"
)
_JS_DRAW = (
    "console.log(JSON.stringify([c.randomBytes(16).toString('hex'), c.randomUUID(), "
    "Buffer.from(crypto.getRandomValues(new Uint8Array(16))).toString('hex'), "
    "Buffer.from(crypto.getRandomValues(new Uint8Array(4096))).toString('hex').slice(0, 64), Math.random(), "
    "c.generateKeyPairSync('ed25519').publicKey.export({format: 'der', type: 'spki'}).toString('hex'), "
    "require('node:vm').runInNewContext('Math.random()')]))"
)


async def _two_copies(scheduler: Scheduler, spec: VmSpec, *, use: str, draw: str) -> tuple[str, str]:
    """Save a VM after `use` ran, load the save twice, run `draw` in each copy: the stdout of each copy."""
    async with await scheduler.new(spec) as vm:
        assert (await vm.exec(use)).exit_code == 0
        ref = await vm.save()
    copies = [await scheduler.reuse(ref.instance_id) for _ in range(2)]
    try:
        results = [await copy.exec(draw) for copy in copies]
    finally:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InstanceNotSavedWarning)
            for copy in copies:
                await copy.close()
    for result in results:
        assert result.exit_code == 0, result.stderr
    return results[0].stdout, results[1].stdout


class TestRandomInCopies:
    """Two VMs loaded from one save get new random values from their first call."""

    async def test_python(self, store: tuple[Scheduler, str, object]) -> None:
        scheduler, _, _ = store
        first, second = await _two_copies(scheduler, _PY, use=_PY_USE, draw=_PY_DRAW)
        for a, b in zip(first.split(), second.split(), strict=True):  # ssl.RAND_bytes, random, os.urandom
            assert a != b

    async def test_python_random_gets_a_new_seed_at_each_load(self, store: tuple[Scheduler, str, object]) -> None:
        """random.seed(42) before a save does not continue after a load."""
        scheduler, _, _ = store
        seeded = random.Random(42)
        _ = [seeded.random() for _ in range(3)]
        continued = repr(seeded.random())
        with pytest.warns(InstanceNotSavedWarning):  # the exec after the save is not saved
            async with await scheduler.new(_PY) as vm:
                assert (
                    await vm.exec("import random\nrandom.seed(42)\n_ = [random.random() for _ in range(3)]")
                ).exit_code == 0
                ref = await vm.save()
                assert (await vm.exec("print(repr(random.random()))")).stdout == f"{continued}\n"  # no load: continues
        with pytest.warns(InstanceNotSavedWarning):  # the exec after the load is not saved
            async with await scheduler.reuse(ref.instance_id) as vm:
                loaded = (await vm.exec("print(repr(random.random()))")).stdout
        assert loaded != f"{continued}\n"

    async def test_javascript(self, store: tuple[Scheduler, str, object]) -> None:
        """Each value differs in the two copies: the crypto wrappers, Math.random (its pool of the
        save is dropped before each exec), the ed25519 key of generateKeyPairSync (BoringSSL's
        generator of the REPL thread, reseeded at the first exec) and Math.random of a node:vm
        context made in the copy."""
        scheduler, _, _ = store
        first, second = await _two_copies(scheduler, VmSpec(language=Language.JAVASCRIPT), use=_JS_USE, draw=_JS_DRAW)
        for a, b in zip(json.loads(first), json.loads(second), strict=True):
            assert a != b

    async def test_shell(self, store: tuple[Scheduler, str, object]) -> None:
        scheduler, _, _ = store
        first, second = await _two_copies(
            scheduler,
            VmSpec(language=Language.RAW),
            use="echo $RANDOM",
            draw="echo $RANDOM $RANDOM $RANDOM",
        )
        assert first != second


class TestFanOut:
    async def test_children_of_one_save_are_saved_under_their_own_ids(
        self, store: tuple[Scheduler, str, object]
    ) -> None:
        """Two sessions load X; each saves under a new id and continues it; X, C1 and C2 each
        load their own state."""
        scheduler, _, _ = store
        async with await scheduler.new(_PY) as vm:
            await vm.exec("x = 1")
            source = await vm.save()
        c1, c2 = await scheduler.reuse(source.instance_id), await scheduler.reuse(source.instance_id)
        try:
            await c1.exec("x = 2")
            await c2.exec("x = 3")
            assert await c1.save(instance_id="child-1") == InstanceRef(instance_id="child-1", generation=1)
            assert await c2.save(instance_id="child-2") == InstanceRef(instance_id="child-2", generation=1)
            assert c1.instance == InstanceRef(instance_id="child-1", generation=1)
        finally:
            await c1.close()
            await c2.close()
        for instance_id, value in ((source.instance_id, "1"), ("child-1", "2"), ("child-2", "3")):
            async with await scheduler.reuse(instance_id) as vm:
                assert (await vm.exec("print(x)")).stdout == f"{value}\n"


class TestForcedId:
    """Scheduler.new(instance_id=...): one id rule for the SDK and the CLI."""

    async def test_new_with_a_forced_id_saves_and_reuses_under_it(self, store: tuple[Scheduler, str, object]) -> None:
        scheduler, _, _ = store
        async with await scheduler.new(_PY, instance_id="my-vm") as vm:
            await vm.exec("x = 'named'")
            ref = await vm.save()
        assert (ref.instance_id, ref.generation) == ("my-vm", 1)
        async with await scheduler.reuse("my-vm") as vm:
            assert (await vm.exec("print(x)")).stdout.strip() == "named"

    async def test_forced_id_that_exists_is_a_conflict(self, store: tuple[Scheduler, str, object]) -> None:
        """A second new VM with the same id never overwrites the first: its first save conflicts,
        the session stays open and its VM runs on, and the body object it uploaded is deleted."""
        scheduler, prefix, s3 = store
        async with await scheduler.new(_PY, instance_id="taken") as vm:
            await vm.exec("x = 'first'")
            await vm.save()
        (first,) = _objects(s3, prefix, "taken")
        async with await scheduler.new(_PY, instance_id="taken") as vm:
            await vm.exec("x = 'second'")
            with pytest.raises(OutcomeUnknownError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFLICT) as exc:
                await vm.save()
            assert exc.value.action == "VERIFY"
            assert not vm.closed
            assert (await vm.exec("print(x)")).stdout == "second\n"  # the VM runs on
        assert _objects(s3, prefix, "taken") == [first]  # the loser's body is gone
        async with await scheduler.reuse("taken") as vm:
            assert (await vm.exec("print(x)")).stdout.strip() == "first"
            assert vm.instance is not None
            assert vm.instance.generation == 1

    async def test_invalid_forced_id_fails_before_any_vm(self, store: tuple[Scheduler, str, object]) -> None:
        scheduler, _, _ = store
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.INSTANCE_INVALID_ID):
            await scheduler.new(_PY, instance_id="My VM")

    async def test_forced_id_without_a_store_fails_before_any_vm(self, scheduler: Scheduler) -> None:
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.INSTANCE_CONFIG):
            await scheduler.new(_PY, instance_id="my-vm")

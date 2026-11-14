"""Saved-instance tests without QEMU: a QMP peer, a VM, a VM manager, and a store harness.

`FakeQmp` is a QMP peer on a real Unix socket. It answers a save the way QEMU
does (stop, migrate to a ``file:`` URI, query-migrate, cont), keeps QEMU's run
state (``stop`` pauses the vCPUs, ``cont`` runs them), and calls back on
``migrate`` to write the migration file and whenever the guest runs on.
`FakeVm` duck-types the QemuVM fields that InstanceManager and Session read;
its guest is two `SparseImage` streams (RAM and home disk) that the test
changes block by block, so a test knows the exact bytes each save must bring
back. A running fake guest writes to its home disk file, as a real one does: a
home copy taken while the VM runs gets bytes no paused instant had. `FakeVmManager`
gives the local base image (or none) and retires a VM by stopping its QMP peer.
`InstanceEnv` (fixture `instance_env`) puts the real InstanceManager on a store
behind the fault proxy and records the bytes each commit saved: a load is
checked byte for byte against them.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import hashlib
import json
import os
import secrets
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import pytest

import exec_sandbox.instance.manager as instance_manager
from exec_sandbox import constants
from exec_sandbox.images import image_digest
from exec_sandbox.instance.codec import BLOCK_BYTES
from exec_sandbox.instance.head import seal_head
from exec_sandbox.instance.ids import new_id
from exec_sandbox.instance.keyring import KEY_BYTES, Key, Keyring, wrap_data_key
from exec_sandbox.instance.manager import InstanceManager
from exec_sandbox.instance.models import RECENT_COMMITS, ChainEntry, HeadBody, HeadHeader, HostFacts, RecentCommit
from exec_sandbox.instance.s3_store import S3InstanceStore
from exec_sandbox.instance.store import Applied
from exec_sandbox.models import ExecutionResult, Language, StartupTiming, TimingBreakdown
from exec_sandbox.session import Session
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType, VmState

if TYPE_CHECKING:
    import random
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from exec_sandbox.instance.manager import InstanceState
    from exec_sandbox.instance.store import InstanceStore
    from tests.s3_faults import Bucket, S3Proxy

BLOCK = BLOCK_BYTES
"""Block size of instance streams (64 KiB): the unit a pack stores."""
PYTHON = VmSpec(language=Language.PYTHON)
"""The spec of a fake VM when the test does not care which."""
NO_VM_TIMEOUT = pytest.mark.timeout(10)
"""Time budget of each saved-instance test that runs no VM (setup, call and teardown). Such a
test takes tens of ms. A host stall (a lookup or connect that waits for its timeout: 35 s on
macOS CI) then fails the test that meets it, by name, instead of slowing the job until the
step timeout."""


# ----------------------------------------------------------------------------- polling


async def await_until(predicate: Callable[[], bool]) -> None:
    """Poll `predicate` every 50 ms until it is true; AssertionError after 10 s.

    Only for effects that finish in the background (a cleanup delete, a task
    the code does not expose): ordering between steps uses gates and events.
    """
    try:
        async with asyncio.timeout(10):
            while not predicate():  # noqa: ASYNC110 - no event signals these effects
                await asyncio.sleep(0.05)
    except TimeoutError:
        raise AssertionError("condition still false after 10 s") from None


def descriptors_of(path: Path) -> list[tuple[int, int]]:
    """(fd, inode) of each descriptor this process has open on the file at `path` (from /dev/fd)."""
    target = path.stat()
    found: list[tuple[int, int]] = []
    for entry in Path("/dev/fd").iterdir():
        fd = int(entry.name)
        try:
            opened = os.fstat(fd)
        except OSError:  # the descriptor of the listing itself, closed by now
            continue
        if (opened.st_dev, opened.st_ino) == (target.st_dev, target.st_ino):
            found.append((fd, opened.st_ino))
    return found


# ----------------------------------------------------------------------------- guest streams


def random_block(rng: random.Random) -> bytes:
    """64 KiB that zstd cannot compress: its record has a fixed size."""
    return rng.randbytes(BLOCK)


def constant_block(value: int) -> bytes:
    """64 KiB of one byte value: compresses to a record of a few dozen bytes, the same size for every value."""
    return bytes([value]) * BLOCK


def digest(data: bytes) -> tuple[int, str]:
    """(size, sha256): a short, exact fingerprint for byte compares in assertion messages."""
    return len(data), hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> tuple[int, str]:
    return digest(path.read_bytes())


@dataclass(slots=True)
class SparseImage:
    """A stream the test controls: data blocks by number, holes elsewhere, and a byte size."""

    size: int
    blocks: dict[int, bytes] = field(default_factory=dict)

    def put(self, block_no: int, data: bytes) -> None:
        if len(data) != BLOCK or block_no * BLOCK >= self.size:
            raise ValueError(f"block {block_no} ({len(data)} bytes) does not fit a stream of {self.size} bytes")
        self.blocks[block_no] = data

    def discard(self, block_no: int) -> None:
        """The guest drops the block: a hole now (reads as zeros)."""
        self.blocks.pop(block_no)

    def resize(self, size: int) -> None:
        """Grow (zeros) or shrink (blocks past the end are gone)."""
        self.size = size
        self.blocks = {b: d for b, d in self.blocks.items() if b * BLOCK < size}

    def content(self) -> bytes:
        """The bytes a reader of the stream sees."""
        out = bytearray(self.size)
        for block_no, data in self.blocks.items():
            start = block_no * BLOCK
            end = min(start + len(data), self.size)
            out[start:end] = data[: end - start]
        return bytes(out)

    def write_to(self, path: Path) -> None:
        """Write the stream as a sparse file (holes are never written)."""
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            for block_no, data in sorted(self.blocks.items()):
                start = block_no * BLOCK
                os.pwrite(fd, data[: self.size - start], start)
            os.ftruncate(fd, self.size)
        finally:
            os.close(fd)

    def copy(self) -> SparseImage:
        return SparseImage(size=self.size, blocks=dict(self.blocks))

    @classmethod
    def from_file(cls, path: Path) -> SparseImage:
        """A stream with the file's bytes (blocks of zeros become holes)."""
        data = path.read_bytes()
        image = cls(size=len(data))
        for start in range(0, len(data), BLOCK):
            block = data[start : start + BLOCK]
            if block.count(0) != len(block):
                image.blocks[start // BLOCK] = block.ljust(BLOCK, b"\0")
        return image


# ----------------------------------------------------------------------------- QMP peer


class FakeQmp:
    """A QMP peer: greeting, qmp_capabilities, then a reply per command.

    Each command gets QEMU's answer to a good save: ``stop`` and ``cont`` pause and
    run the vCPUs, ``migrate`` calls `on_migrate` with the file path of its
    ``file:`` URI, ``query-migrate`` answers "completed" (or "failed" with
    `migration_error`), any other command {}. `on_run` is called whenever the
    guest runs on: at ``cont``, and after a ``migrate`` sent while the vCPUs run
    (a live migration: the guest goes on writing). `hold_migration`, when set,
    makes query-migrate wait until it is set free. `refused` maps a command to the
    error QEMU gives it instead (GenericError with that description).
    """

    def __init__(self, *, on_migrate: Callable[[Path], None] | None, on_run: Callable[[], None] | None) -> None:
        self.on_migrate = on_migrate
        self.on_run = on_run
        self.running = True
        """QEMU's run state: the vCPUs run until ``stop``, and again after ``cont``."""
        self.commands: list[str] = []
        self.hung_up = asyncio.Event()
        self.migration_error: str | None = None
        self.hold_migration: asyncio.Event | None = None
        """Set to an unset Event to hold query-migrate; set it to let the migration end."""
        self.migration_waiting = asyncio.Event()
        """Set when a query-migrate waits on `hold_migration`."""
        self.refused: dict[str, str] = {}
        """Command -> error description: QEMU refuses that command (and does nothing)."""
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.add(writer)
        try:
            writer.write(json.dumps({"QMP": {"version": {}, "capabilities": []}}).encode() + b"\n")
            await writer.drain()
            while line := await reader.readline():
                request = json.loads(line)
                command = request["execute"]
                self.commands.append(command)
                reply = await self._reply(command, request.get("arguments") or {})
                writer.write(
                    json.dumps({**reply, **({"id": request["id"]} if "id" in request else {})}).encode() + b"\n"
                )
                await writer.drain()
        except ConnectionError:
            pass
        finally:
            self._writers.discard(writer)
            self.hung_up.set()  # the client closed its end (or the peer was stopped)
            writer.close()

    async def _reply(self, command: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if command in self.refused:
            return {"error": {"class": "GenericError", "desc": self.refused[command]}}
        if command == "qmp_capabilities":
            return {"return": {}}
        if command == "stop":
            self.running = False
        if command == "cont":
            self.running = True
            await self._guest_runs()
        if command == "migrate" and self.on_migrate is not None:
            uri: str = arguments["uri"]
            await asyncio.to_thread(self.on_migrate, Path(uri.removeprefix("file:")))
            if self.running:
                await self._guest_runs()
        if command == "query-migrate":
            if self.hold_migration is not None:
                self.migration_waiting.set()
                await self.hold_migration.wait()
            if self.migration_error is not None:
                return {"return": {"status": "failed", "error-desc": self.migration_error}}
            return {"return": {"status": "completed"}}
        return {"return": {}}

    async def _guest_runs(self) -> None:
        if self.on_run is not None:
            await asyncio.to_thread(self.on_run)

    async def start(self, path: Path) -> None:
        self._server = await asyncio.start_unix_server(self.serve, path=str(path))

    async def stop(self) -> None:
        """QEMU is gone: stop listening and cut every connection."""
        if self._server is not None:
            self._server.close()
            for writer in list(self._writers):
                writer.close()
            await self._server.wait_closed()
            self._server = None


# ----------------------------------------------------------------------------- VM and VM manager


class FakeVm:
    """Duck-typed QemuVM for InstanceManager.save and Session, with a guest the test writes.

    The guest is `ram` (what QEMU writes to the migration file) and `home` (the
    home disk). At ``migrate`` (the VM is stopped), FakeQmp writes `ram` to the
    migration file and `home` to `workdir.home_disk`, so the saved streams are
    `expected()` at that instant. While the vCPUs run, the guest writes to the
    home disk file (see `_run`): only a copy taken while the VM is stopped holds
    `expected()`.
    """

    def __init__(self, *, spec: VmSpec, base_image: Path, root: Path, ram: SparseImage, home: SparseImage) -> None:
        root.mkdir(mode=0o700, exist_ok=True)
        self.vm_id = f"fake-{uuid.uuid4().hex[:12]}"
        self.spec = spec
        self.base_image = base_image
        self.packages_image: Path | None = None
        # The digests VmManager records when QEMU opens the images (InstanceManager._host_facts)
        self.base_image_digest: str | None = image_digest(base_image)
        self.packages_image_digest: str | None = None
        self.use_qemu_vm_user = False
        self.cgroup = None  # no memory cgroup (QemuVM.cgroup)
        self.workdir = SimpleNamespace(home_disk=root / "home.img")
        self.qmp_socket = root / "qmp.sock"
        self.state = VmState.READY
        self.process = SimpleNamespace(returncode=None)
        self.exposed_ports: list[Any] = []
        self.ram = ram
        self.home = home
        self.home.write_to(self.workdir.home_disk)
        self.qmp = FakeQmp(on_migrate=self._migrate, on_run=self._run)
        self.on_execute: Callable[[str], Awaitable[str]] | None = None
        """Runs for each execute(code) and returns its stdout; default: empty output."""

    def _migrate(self, path: Path) -> None:
        self.ram.write_to(path)
        self.home.write_to(self.workdir.home_disk)

    def _run(self) -> None:
        """The guest runs: it writes to its home disk file (the first byte changes). A save
        that copies the disk after ``cont`` gets bytes no paused instant had, and its load
        check fails. The next ``migrate`` writes the file again from `home`."""
        with self.workdir.home_disk.open("r+b") as f:
            first = f.read(1)
            if first:
                f.seek(0)
                f.write(bytes([first[0] ^ 0xFF]))

    def expected(self) -> tuple[bytes, bytes]:
        """(vmstate, home) bytes a save taken now writes, and a load of it must give back."""
        return self.ram.content(), self.home.content()

    async def start(self) -> FakeVm:
        await self.qmp.start(self.qmp_socket)
        return self

    async def execute(
        self,
        *,
        code: str,
        timeout_seconds: int,
        env_vars: dict[str, str] | None,
        on_stdout: Callable[[str], None] | None,
        on_stderr: Callable[[str], None] | None,
    ) -> ExecutionResult:
        stdout = await self.on_execute(code) if self.on_execute is not None else ""
        return ExecutionResult(
            stdout=stdout, stderr="", exit_code=0, timing=TimingBreakdown(execute_ms=0, connect_ms=None)
        )


class FakeVmManager:
    """What InstanceManager and Session use of VmManager: settings, the local base image, destroy_vm."""

    def __init__(self, *, base_image: Path | None) -> None:
        self.settings = SimpleNamespace(force_emulation=False)
        self.base_image = base_image
        """The worker's own base image; None: this worker lacks it (a load fetches it by digest)."""
        self.images = SimpleNamespace(base_image=self._base_image)  # VmManager.images (GuestImages)

    def _base_image(self, language: object) -> Path:
        if self.base_image is None:
            raise FileNotFoundError(f"no base image for {language} on this worker")
        return self.base_image

    async def destroy_vm(self, vm: FakeVm) -> bool:
        await vm.qmp.stop()
        vm.state = VmState.DESTROYED
        return True


def fake_session(
    *,
    vm: FakeVm,
    vm_manager: FakeVmManager,
    instances: InstanceManager,
    state: InstanceState | None,
    new_instance_id: str | None,
) -> Session:
    """A real Session over a FakeVm (what Scheduler.new() or reuse() return, without QEMU).
    `new_instance_id`: the id Scheduler.new(instance_id=...) gives the first save (None: random)."""
    return Session(
        vm=vm,  # type: ignore[arg-type]
        vm_manager=vm_manager,  # type: ignore[arg-type]
        spec=vm.spec,
        startup_timing=StartupTiming(
            total_ms=0,
            admission_ms=0,
            setup_ms=0,
            boot_ms=0,
            warm_pool_hit=False,
            l1_cache_hit=False,
            l1_restore_ms=None,
            overlay_ms=0,
            qemu_cmd_build_ms=0,
            gvproxy_start_ms=0,
            qemu_fork_ms=0,
            guest_wait_ms=0,
            boot_retries=0,
            instance_load_ms=None if state is None else 0,
        ),
        instances=instances,
        instance=state,
        new_instance_id=new_instance_id,
        idle_timeout_seconds=3600,
        default_timeout_seconds=60,
    )


async def no_packages(_spec: VmSpec) -> Path | None:
    """`packages_image` for InstanceManager.load: this worker has no L2 image."""
    return None


async def qemu_floor() -> tuple[int, int, int]:
    """Stand-in for the QEMU version probe: the fake VM needs no QEMU binary."""
    return constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION


def accelerator_probe(host: AccelType) -> Callable[..., Awaitable[AccelType]]:
    """Stand-in for the accelerator probe of InstanceManager: `host` (the worker probed it
    once), or TCG when emulation is forced, as the real probe answers."""

    async def detect_accel_type(*, force_emulation: bool) -> AccelType:
        return AccelType.TCG if force_emulation else host

    return detect_accel_type


def new_keyring() -> str:
    """One fresh keyring key (the text format)."""
    return base64.b64encode(secrets.token_bytes(32)).decode()


def head_header(
    *, instance_id: str, generation: int, commit_id: str, older: tuple[RecentCommit, ...], kek: Key, data_key: bytes
) -> HeadHeader:
    """The public header of a head that a test seals itself: its commit first in the recent
    list (at most RECENT_COMMITS, newest first), its data key wrapped by `kek`."""
    return HeadHeader(
        instance_id=instance_id,
        generation=generation,
        commit_id=commit_id,
        recent_commits=(RecentCommit(generation=generation, commit_id=commit_id), *older)[:RECENT_COMMITS],
        key_id=kek.id,
        wrapped_data_key=wrap_data_key(kek, data_key).hex(),
    )


async def seed_instance(
    store: InstanceStore, scratch: Path, *, instance_id: str, keyring: str, spec: VmSpec, host: HostFacts
) -> None:
    """What a first save stores: one full copy and a head sealed with the newest key of `keyring`.
    The key layout is written out here, not taken from store.py."""
    data_key = os.urandom(KEY_BYTES)
    commit_id = new_id()
    body_object = scratch / f"{instance_id}-{commit_id}.f"
    body_object.write_bytes(os.urandom(1000))
    etag = await store.upload(f"s/{instance_id}/b/{commit_id}.f", src=body_object)
    header = head_header(
        instance_id=instance_id,
        generation=1,
        commit_id=commit_id,
        older=(),
        kek=Keyring.parse(keyring).latest,
        data_key=data_key,
    )
    body = HeadBody(
        spec=spec,
        host=host,
        chain=(ChainEntry(commit_id=commit_id, kind="full", size=1000, stream_sizes=(0, 0), etag=etag),),
    )
    head = seal_head(header=header, body=body, data_key=data_key)
    assert isinstance(await store.put_if_absent(f"s/{instance_id}/head", data=head), Applied)


_DISK_FULL_SEAMS = {"clone": "clone_sparse", "seal": "write_object"}


def disk_full_once(monkeypatch: pytest.MonkeyPatch, *, step: Literal["clone", "seal"]) -> None:
    """The next local write of a save in `step` fails with ENOSPC, once; later calls run as usual.

    The one private seam of the save path (no public hook makes a local write fail):
    the manager module's name for a dependency is replaced, never the manager's own
    code. "clone": files.disk.clone_sparse (the home disk copy, while the VM is
    stopped); "seal": pack.write_object (the body object).
    """
    name = _DISK_FULL_SEAMS[step]
    real = getattr(instance_manager, name)
    calls = 0

    def fail_first(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(*args, **kwargs)

    monkeypatch.setattr(instance_manager, name, fail_first)


# ----------------------------------------------------------------------------- store harness


_OBJECT_EXTENSION = {"full": "f", "pack": "p"}
"""Key suffix of each body object kind (store.py key layout)."""


def chain_keys(state: InstanceState) -> list[str]:
    """Store keys of the chain an authenticated head names, oldest first. The key layout
    (s/<instance>/b/<commit>.<f|p>) is written out here, not taken from store.py."""
    return [f"s/{state.instance_id}/b/{e.commit_id}.{_OBJECT_EXTENSION[e.kind]}" for e in state.body.chain]


def head_object(instance_id: str) -> str:
    """Store key of an instance's head (store.py key layout, written out)."""
    return f"s/{instance_id}/head"


def kinds(state: InstanceState) -> list[str]:
    """Kinds of the chain objects, oldest first ("full", then "pack"s)."""
    return [e.kind for e in state.body.chain]


class InstanceEnv:
    """One test's store (behind the fault proxy), managers, fake VMs, and the bytes each commit saved."""

    def __init__(self, *, proxy: S3Proxy, store: InstanceStore, root: Path, tmp: Path, base_image: Path) -> None:
        self.proxy = proxy
        self.bucket: Bucket = proxy.bucket
        self.store = store
        self.root = root
        """Short folder for QMP sockets (macOS: 104-byte socket paths)."""
        self.base_image = base_image
        self.work_dir = tmp / "work"
        self.asset_dir = tmp / "assets"
        self.work_dir.mkdir(mode=0o700)
        self.asset_dir.mkdir(mode=0o700)
        self.vm_manager = FakeVmManager(base_image=base_image)
        self.keyring = new_keyring()
        self.manager = self.manager_for(self.keyring)
        self.vms: list[FakeVm] = []
        self.saved: dict[str, tuple[bytes, bytes]] = {}
        """commit id -> (vmstate, home) bytes that commit saved."""

    @classmethod
    @contextlib.asynccontextmanager
    async def open(cls, *, proxy: S3Proxy, root: Path, tmp: Path) -> AsyncGenerator[InstanceEnv]:
        base_image = tmp / "base.img"
        base_image.write_bytes(os.urandom(3 * BLOCK + 123))
        async with S3InstanceStore.open(
            bucket=proxy.bucket.name, prefix=proxy.prefix, endpoint_url=proxy.endpoint_url, region="us-east-1"
        ) as store:
            env = cls(proxy=proxy, store=store, root=root, tmp=tmp, base_image=base_image)
            try:
                yield env
            finally:
                proxy.clear()
                for vm in env.vms:
                    with contextlib.suppress(Exception):
                        await vm.qmp.stop()

    def manager_for(self, keyring: str) -> InstanceManager:
        """Another worker on the same store and folders, with this keyring."""
        return InstanceManager(
            store=self.store,
            keyring=Keyring.parse(keyring),
            work_dir=self.work_dir,
            asset_dir=self.asset_dir,
            vm_manager=self.vm_manager,  # type: ignore[arg-type]
        )

    async def new_vm(self, ram: SparseImage, home: SparseImage, *, spec: VmSpec) -> FakeVm:
        vm = FakeVm(
            spec=spec,
            base_image=self.base_image,
            root=self.root / f"v{len(self.vms)}",
            ram=ram,
            home=home,
        )
        self.vms.append(vm)
        return await vm.start()

    async def vm_from(self, state: InstanceState) -> FakeVm:
        """The VM a reuse() of `state` runs: the loaded streams are its RAM and home disk."""
        return await self.new_vm(
            SparseImage.from_file(state.vmstate), SparseImage.from_file(state.home), spec=state.body.spec
        )

    async def load(self, instance_id: str, *, manager: InstanceManager) -> InstanceState:
        return await manager.load(instance_id, packages_image=no_packages)

    async def check_load(
        self,
        instance_id: str,
        *,
        generation: int,
        manager: InstanceManager,
        want: tuple[bytes, bytes] | None,
    ) -> InstanceState:
        """Load with `manager`, check both streams are `want` (None: the bytes that commit
        saved, as Saver recorded them), release the files."""
        state = await self.load(instance_id, manager=manager)
        try:
            assert state.header.generation == generation
            expected = want if want is not None else self.saved[state.header.commit_id]
            got = (file_digest(state.vmstate), file_digest(state.home))
            assert got == (digest(expected[0]), digest(expected[1]))
        finally:
            await manager.release(state)
        return state

    def work_files(self) -> set[Path]:
        """Every file and folder in the managers' work and asset folders."""
        return set(self.work_dir.rglob("*")) | set(self.asset_dir.rglob("*"))


class Saver:
    """One session: a VM, the state of its last load or save, and the manager it saves with."""

    def __init__(self, env: InstanceEnv, vm: FakeVm, *, manager: InstanceManager, state: InstanceState | None) -> None:
        self.env = env
        self.vm = vm
        self.manager = manager
        self.state = state

    @property
    def instance_id(self) -> str:
        assert self.state is not None
        return self.state.instance_id

    async def save(self) -> InstanceState:
        """manager.save(); records the streams the guest had for the commit's byte check."""
        expected = self.vm.expected()
        self.state = await self.manager.save(self.vm, self.state, new_instance_id=None)  # type: ignore[arg-type]
        self.env.saved[self.state.header.commit_id] = expected
        return self.state


# ----------------------------------------------------------------------------- another worker


def worker_without_base_image(env: InstanceEnv, *, root: Path) -> InstanceManager:
    """Another worker on the same store and keyring, with its own work and asset folders
    under `root`, and no local base image: its load downloads the image by digest."""
    work_dir, asset_dir = root / "work", root / "assets"
    work_dir.mkdir(mode=0o700, parents=True)
    asset_dir.mkdir(mode=0o700)
    return InstanceManager(
        store=env.store,
        keyring=Keyring.parse(env.keyring),
        work_dir=work_dir,
        asset_dir=asset_dir,
        vm_manager=FakeVmManager(base_image=None),  # type: ignore[arg-type]
    )


def disk_full_at_download(monkeypatch: pytest.MonkeyPatch, store: InstanceStore, *, under: Path) -> None:
    """The store's next download to a file under the folder `under` fails with ENOSPC, once,
    after it wrote part of the file and left it there (the caller must remove it); other
    downloads run as usual. The store is the boundary: no manager code is replaced."""
    real = store.download
    failed = False

    async def download(key: str, /, *, dest: Path, max_bytes: int, expected_size: int | None) -> bool:
        nonlocal failed
        if failed or under not in dest.parents:
            return await real(key, dest=dest, max_bytes=max_bytes, expected_size=expected_size)
        failed = True
        dest.write_bytes(b"\1" * BLOCK)  # the part written before the disk was full
        raise OSError(errno.ENOSPC, "No space left on device", str(dest))

    monkeypatch.setattr(store, "download", download)

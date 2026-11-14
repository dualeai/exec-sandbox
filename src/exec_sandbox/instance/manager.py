"""Saved instances: the store and the keyring; for save and load, also a local
work folder and the VM manager of one Scheduler.

Save: stop the VM, write the migration file (RAM + devices) and copy the home
disk while it is stopped, let it run again; then diff both streams against the
session's baseline (the state it loaded or last saved), seal a pack or a full
copy, upload it, and commit the head (create-only for the first save, If-Match
otherwise). After the head of a first save commits, the save checks that its
body object is still in the store (a delete_instance can run between the two);
a delete that removes the body after that check finds the new head and removes it too.
A full copy, with a new data key, is written for the first save,
when the newest keyring key does not wrap the chain's data key, when the chain
has MAX_CHAIN_PACKS packs, when the packs since the last full copy would outgrow
it, or when an object of the chain is missing, resized or replaced in the store
(an expiry rule, a manual delete, a store writer: one listing before each pack
compares every key, size and ETag). A pack on a broken chain would commit a
state no load can rebuild; the full copy repairs it.

Load: read and open the head, check this host can run it (ROUTE before any
download), download the chain into empty sparse files; the caller restores the
VM from them. A body object that disappears mid-load means the head moved on
(re-read it) or it was removed from the store (InstanceGoneError INSTANCE_EXPIRED).

A local file error (OSError) of a save or a load, in the work folder or the
asset folder, becomes TransientError LOCAL_DISK: nothing was saved or loaded.

Admin (describe, delete, list): `InstanceAdmin`, the store and the keyring only.
`open_instance_admin` opens it with no VM, no guest image, no QEMU and no store
probe; `InstanceManager` (save, load) is an InstanceAdmin too.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from exec_sandbox import __version__, constants
from exec_sandbox._imports import require_s3_instance_store
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InstanceGoneError,
    OutcomeUnknownError,
    SandboxError,
    TransientError,
    host_mismatch,
    no_instance_store,
)
from exec_sandbox.files.disk import clone_sparse, make_private_dir, memory_filesystem
from exec_sandbox.hash_utils import file_hash
from exec_sandbox.images import image_digest
from exec_sandbox.instance import ids
from exec_sandbox.instance.commit import Base, Commit, commit_head, save_context
from exec_sandbox.instance.head import MAX_HEAD_BYTES, open_head, read_header, seal_head
from exec_sandbox.instance.keyring import KEY_BYTES, Keyring, wrap_data_key
from exec_sandbox.instance.local_store import LocalInstanceStore
from exec_sandbox.instance.models import (
    RECENT_COMMITS,
    STREAM_HOME,
    STREAM_VMSTATE,
    ChainEntry,
    HeadBody,
    HeadHeader,
    HostFacts,
    InstanceInfo,
    RecentCommit,
)
from exec_sandbox.instance.pack import apply_object, new_stream_file, write_object
from exec_sandbox.instance.store import (
    INSTANCES_PREFIX,
    InstanceStore,
    asset_key,
    body_key,
    body_prefix,
    head_key,
    instance_prefix,
    probe,
)
from exec_sandbox.migration_client import MigrationClient, migration_timeout
from exec_sandbox.permission_utils import (
    get_expected_socket_uid,
    grant_qemu_vm_private_file_access,
    probe_qemu_vm_user,
)
from exec_sandbox.platform_utils import HostArch, detect_host_arch, detect_host_x86_vendor
from exec_sandbox.qemu_cmd import guest_cpu_model
from exec_sandbox.system_probes import detect_accel_type, probe_qemu_version

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Coroutine, Generator, Mapping
    from typing import Any

    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.settings import Settings
    from exec_sandbox.vm_manager import VmManager
    from exec_sandbox.vm_spec import VmSpec

logger = get_logger(__name__)

_LOAD_ATTEMPTS = 3
"""Head reads per load while body objects vanish because newer saves replaced them."""
_OBJECT_KIND = {"full": "f", "pack": "p"}
_CHECKED_FACTS = ("arch", "accelerator", "cpu_vendor", "cpu_model", "release")
"""Host facts a load must match. cpu_model: QEMU refuses a state saved with other CPU
features, but only at restore, after the download. Images are fetched by digest.
qemu_version is not checked: QEMU accepts or refuses the saved state at restore (then
ROUTE). release: the saved guest agent runs on in RAM and speaks its own release's
protocol, so a load on another exec-sandbox release is refused (ROUTE), never migrated."""
MAX_CHAIN_PACKS = 64
"""Packs after a full copy before the next full copy: bounds the head (one entry
per pack) and the load (one download per object)."""
_MAX_ASSET_BYTES = 16 * 1024 * 1024 * 1024
"""Largest guest image the store may serve (base images and L2 layers are far smaller)."""
_OBJECTS_AHEAD = 8
"""Chain objects a load has at once: the one it applies and the next ones it downloads
(at most this many object files on local disk). An object above one S3 part sends a HEAD,
then its ranged GETs, 4 at a time, so a load can ask for more than the body client's 10
connections (botocore's default): the extra requests wait for a free connection."""


class _VanishedError(Exception):
    """A body object the head lists is gone (the store answered "no such key")."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceState:
    """What a session knows about the instance it continues: the head it loaded
    or committed last, and that state as local files (the baseline of the next pack)."""

    header: HeadHeader
    body: HeadBody
    etag: str | None
    """ETag of this session's head, the base of its next save. None: this session's
    save committed, but another save is already on top (the next save conflicts)."""
    data_key: bytes = field(repr=False)
    folder: Path
    """Private folder of this session's instance files (removed when the session closes)."""
    vmstate: Path
    home: Path
    base_image: Path
    """Local base image with the saved digest (the overlay's backing file)."""
    packages_image: Path | None
    """Local L2 packages image with the saved digest, None without packages."""

    @property
    def instance_id(self) -> str:
        return self.header.instance_id


@contextlib.asynccontextmanager
async def _open_store(settings: Settings) -> AsyncGenerator[tuple[InstanceStore, Keyring]]:
    """Keyring, then the store `instance_store` names, open: the first checks of both
    openers. S3: no request is sent yet. Local: the folder is made and checked now
    (LocalInstanceStore.open). Raises ConfigError INSTANCE_CONFIG (FIX_CONFIG) when
    either is not set or not valid (local: its folder is not safe), TransientError
    INSTANCE_STORE_UNAVAILABLE for a local disk error, and ConfigError EXTRA_MISSING
    when the store's extra is not installed."""
    if settings.instance_store is None:
        raise no_instance_store(instance_id=None)
    if settings.instance_keyring is None:
        raise ConfigError(
            "instance_store is set, but instance_keyring is not: saved instances are sealed with its keys.",
            code=ErrorCode.INSTANCE_CONFIG,
            instance_id=None,
            hint="Set instance_keyring (EXEC_SANDBOX_INSTANCE_KEYRING), or unset instance_store.",
            context=None,
        )
    keyring = Keyring.parse(settings.instance_keyring.get_secret_value())
    match settings.instance_store:
        case "s3":
            if settings.s3_bucket is None:
                raise ConfigError(
                    'instance_store is "s3", but s3_bucket is not set.',
                    code=ErrorCode.INSTANCE_CONFIG,
                    instance_id=None,
                    hint="Set s3_bucket (EXEC_SANDBOX_S3_BUCKET), or use another instance_store.",
                    context=None,
                )
            async with require_s3_instance_store().open(
                bucket=settings.s3_bucket,
                prefix=settings.s3_instance_prefix,
                endpoint_url=settings.s3_endpoint_url,
                region=settings.s3_region,
            ) as store:
                yield store, keyring
        case "local":
            async with LocalInstanceStore.open(root=settings.instance_store_dir) as store:
                yield store, keyring


@contextlib.asynccontextmanager
async def open_instance_admin(settings: Settings) -> AsyncGenerator[InstanceAdmin]:
    """Saved instances of the store `instance_store` names, to describe, list and
    delete them. Checks the keyring and the store settings only: no VM manager, no
    guest image, no QEMU check, no store probe (no test object is written).

    Raises ConfigError INSTANCE_CONFIG (FIX_CONFIG) when the store or the keyring
    is not set or not valid (the local store also makes and checks its folder), and
    ConfigError EXTRA_MISSING when the store's extra is not installed. An S3 store
    that does not answer fails at the first call (TransientError
    INSTANCE_STORE_UNAVAILABLE); the local store raises it at open for a disk error
    (full, I/O).
    """
    async with _open_store(settings) as (store, keyring):
        yield InstanceAdmin(store=store, keyring=keyring)


class InstanceAdmin:
    """Describe, delete and list: the store and the keyring only. Open with
    `open_instance_admin`."""

    def __init__(self, *, store: InstanceStore, keyring: Keyring) -> None:
        self.store = store
        self.keyring = keyring

    async def describe(self, instance_id: str) -> InstanceInfo:
        """Public header always; the sealed part when this keyring has the key (verified).

        Raises InvalidInputError INSTANCE_INVALID_ID (before any store call),
        InstanceGoneError INSTANCE_NOT_FOUND when there is no head, InstanceGoneError
        INSTANCE_INTEGRITY or RouteError INSTANCE_HOST_MISMATCH when the head does not
        open with the key it names, and the store's errors (TransientError
        INSTANCE_STORE_UNAVAILABLE, ConfigError INSTANCE_CONFIG).
        """
        ids.check_instance_id(instance_id)
        with _named(instance_id):
            got = await self.store.get(head_key(instance_id), max_bytes=MAX_HEAD_BYTES)
            if got is None:
                raise _not_found(instance_id)
            # One read: `verified` speaks for the very bytes the other fields come from.
            try:
                header, body, _ = open_head(got[0], self.keyring, instance_id=instance_id)
            except ConfigError:  # INSTANCE_KEY_MISSING: the only ConfigError of open_head
                header, body = read_header(got[0], instance_id=instance_id), None
            return InstanceInfo(
                instance_id=instance_id,
                generation=header.generation,
                recent_commits=header.recent_commits,
                key_id=header.key_id,
                verified=body is not None,
                spec=None if body is None else body.spec,
                host=None if body is None else body.host,
            )

    async def delete(self, instance_id: str) -> bool:
        """Head first (the instance is gone and unreadable at once), then every body
        object under its prefix (orphans included). False: there was no head (the
        other objects are still deleted). A first save that runs at the same time
        either sees its object gone and fails, or commits first: then this call
        finds the new head after the objects go, removes it too and returns True.
        InvalidInputError INSTANCE_INVALID_ID before any store call: an id such as
        "a/b" names a prefix inside instance "a". The store's errors (TransientError
        INSTANCE_STORE_UNAVAILABLE, ConfigError INSTANCE_CONFIG) can come after the
        head is gone: a new call returns False and removes what is left."""
        ids.check_instance_id(instance_id)
        with _named(instance_id):
            existed = await self.store.exists(head_key(instance_id))
            if existed:
                await self.store.delete_many([head_key(instance_id)])
            await self._delete_objects(instance_id)
            if not existed and await self.store.exists(head_key(instance_id)):
                # A first save wrote its head while this call removed the objects it
                # listed, and the save's own check could come first: remove that
                # instance too, as if this call ran after the save.
                existed = True
                await self.store.delete_many([head_key(instance_id)])
                await self._delete_objects(instance_id)
        logger.info("Instance deleted", extra={"instance_id": instance_id, "existed": existed})
        return existed

    async def _delete_objects(self, instance_id: str) -> None:
        """Every object under the instance's prefix."""
        await self.store.delete_many([o.key for o in await self.store.list_objects(instance_prefix(instance_id))])

    async def list_ids(self) -> AsyncGenerator[str]:
        """Ids that have any object in the store (with or without a head), in store
        order; names that are not ids are skipped."""
        async for name in self.store.list_children(INSTANCES_PREFIX):
            if ids.is_instance_id(name):
                yield name


class InstanceManager(InstanceAdmin):
    """An InstanceAdmin that also saves and loads (a local work folder, the VM manager).
    Open with `InstanceManager.open` (checks the keyring, QEMU and the store)."""

    def __init__(
        self, *, store: InstanceStore, keyring: Keyring, work_dir: Path, asset_dir: Path, vm_manager: VmManager
    ) -> None:
        super().__init__(store=store, keyring=keyring)
        self.work_dir = work_dir
        self.asset_dir = asset_dir
        self._vm_manager = vm_manager
        self._saves = 0
        self._no_saves = asyncio.Event()
        self._no_saves.set()
        self._background: set[asyncio.Task[Any]] = set()
        """Head writes, which run on when their save is cancelled, and the work file
        removals of saves and loads, which run on after their call returns."""

    @classmethod
    @contextlib.asynccontextmanager
    async def open(
        cls, settings: Settings, *, work_dir: Path, vm_manager: VmManager
    ) -> AsyncGenerator[InstanceManager]:
        """Saved instances live in the store `instance_store` names. Fail fast, in
        this order: keyring, store settings, QEMU floor, work folder (not on a
        memory filesystem), store probe. Each raises ConfigError INSTANCE_CONFIG
        (FIX_CONFIG; MIGRATION_UNSUPPORTED for a QEMU below the floor), or
        TransientError INSTANCE_STORE_UNAVAILABLE when the store does not answer. A
        store whose extra is not installed raises ConfigError EXTRA_MISSING."""
        async with _open_store(settings) as (store, keyring):
            qemu = await probe_qemu_version()
            if qemu is None or qemu < constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION:
                floor = ".".join(map(str, constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION))
                found = ".".join(map(str, qemu)) if qemu else "unknown"
                raise ConfigError(
                    f"Saved instances need QEMU >= {floor}; this worker has {found}.",
                    code=ErrorCode.MIGRATION_UNSUPPORTED,
                    instance_id=None,
                    hint="Upgrade QEMU on this worker, or unset instance_store.",
                    context=None,
                )
            if (filesystem := await asyncio.to_thread(memory_filesystem, work_dir.parent)) is not None:
                raise ConfigError(
                    f"The work folder of saved instances, {work_dir}, is on {filesystem}: a save writes the "
                    "VM's memory there, into memory charged to the VM, which then runs out of it.",
                    code=ErrorCode.INSTANCE_CONFIG,
                    instance_id=None,
                    hint="Set EXEC_SANDBOX_HOME_DISK_DIR (it holds this work folder) to a folder on disk.",
                    context={"work_dir": str(work_dir), "filesystem": filesystem},
                )
            work_dir.mkdir(mode=0o700, exist_ok=True)
            make_private_dir(settings.instance_asset_dir)
            await probe(store, scratch=work_dir)
            logger.info(
                "Instance store ready",
                extra={"store": settings.instance_store, "key_ids": [key.id for key in keyring.keys]},
            )
            yield cls(
                store=store,
                keyring=keyring,
                work_dir=work_dir,
                asset_dir=settings.instance_asset_dir,
                vm_manager=vm_manager,
            )

    # ------------------------------------------------------------------ save

    async def save(self, vm: QemuVM, state: InstanceState | None, *, new_instance_id: str | None) -> InstanceState:
        """Save the running VM as the next generation (the first when `state` is None).

        A first save uses `new_instance_id`, or a new random id when it is None.
        The VM keeps running. Raises OutcomeUnknownError (INSTANCE_CONFLICT: another
        save won, or a first save's id is taken; INSTANCE_SAVE_OUTCOME_UNKNOWN), or
        InstanceGoneError INSTANCE_NOT_FOUND (deleted meanwhile): the caller closes the
        session on those. TransientError LOCAL_DISK: local disk full or files gone;
        nothing was saved. ConfigError IMAGE_DAMAGED: a guest image does not match its
        digest; nothing was saved.
        Any other failure before the head write leaves the head unchanged (uploaded
        objects can stay: delete_instance or an expiry rule of the store removes
        them). If the caller is cancelled during the head write, the write finishes
        in the background and can land.

        A first save checks, after its head write, that its uploaded object is
        still in the store: a delete_instance during the save finds no head and
        removes the object. Then the save removes its head (best effort) and
        raises InstanceGoneError INSTANCE_NOT_FOUND; when the store does not answer
        this check, OutcomeUnknownError INSTANCE_SAVE_OUTCOME_UNKNOWN. A later save
        needs no check: delete removes the head first, so its If-Match write fails.
        """
        instance_id = state.instance_id if state is not None else new_instance_id or ids.new_id()
        self._saves += 1
        self._no_saves.clear()
        try:
            with _named(instance_id):
                return await self._save(vm, state, instance_id=instance_id)
        finally:
            self._saves -= 1
            if not self._saves:
                self._no_saves.set()

    async def drain(self) -> None:
        """Wait until no save and no background job runs: a head write runs on after
        its save is cancelled, and the work files of a save or a load (folders, body
        objects, an asset's .part file) are removed in a worker thread after it
        returns. Shutdown calls it before the VMs, the work folder and the store
        clients go."""
        await self._no_saves.wait()
        # Only jobs that have not ended: a gather of ended jobs completes without a yield,
        # so their discard callbacks never run and a `while self._background` loop spins.
        while running := [job for job in self._background if not job.done()]:
            await asyncio.gather(*running, return_exceptions=True)

    def _in_background[T](self, job: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Run `job` on its own: it ends even when its caller is cancelled or returns
        first. drain() waits for it."""
        task = asyncio.ensure_future(job)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def _remove_in_background(self, *paths: Path) -> None:
        """_remove_quietly (files and folders) in a worker thread: a large file can
        take tens of ms to remove, too long for the event loop. Names are unique
        (commit ids, mkdtemp): a removal that runs late never hits a newer file."""
        self._in_background(asyncio.to_thread(_remove_quietly, *paths))

    async def _save(self, vm: QemuVM, state: InstanceState | None, *, instance_id: str) -> InstanceState:
        """save() itself (save() counts it as in flight)."""
        commit_id = ids.new_id()
        if state is not None and state.etag is None:
            raise OutcomeUnknownError(
                f"Instance {state.instance_id}: another save committed after this session's last save. "
                "This save was not stored.",
                code=ErrorCode.INSTANCE_CONFLICT,
                instance_id=state.instance_id,
                hint="Load the instance again before you redo the work.",
                context=save_context(generation=state.header.generation + 1, commit_id=commit_id),
            )
        obj: Path | None = None
        with _local_files(instance_id, step="saving"):
            folder = state.folder if state is not None else self._new_folder(instance_id)
            vmstate, home = folder / f"{commit_id}.vmstate", folder / f"{commit_id}.home"
            try:
                await self._snapshot(vm, vmstate=vmstate, home=home)
                facts = await self._host_facts(vm)
                await self._ensure_vm_assets(vm, facts=facts)
                kind, obj, size, data_key = await self._write_body(
                    instance_id=instance_id, commit_id=commit_id, state=state, vmstate=vmstate, home=home
                )
                stream_sizes = (vmstate.stat().st_size, home.stat().st_size)
                object_key = body_key(instance_id, commit_id, _OBJECT_KIND[kind])
                object_etag = await self.store.upload(object_key, src=obj)
            except BaseException:
                self._drop_new_files(state, folder, *(path for path in (vmstate, home, obj) if path is not None))
                raise
        self._remove_in_background(obj)  # uploaded: the store holds it now

        entry = ChainEntry(commit_id=commit_id, kind=kind, size=size, stream_sizes=stream_sizes, etag=object_etag)
        chain = (entry,) if kind == "full" or state is None else (*state.body.chain, entry)
        body = HeadBody(spec=vm.spec, host=facts, chain=chain)
        header = self._next_header(state, instance_id=instance_id, commit_id=commit_id, data_key=data_key)
        generation = header.generation
        commit = Commit(
            header=header, data=seal_head(header=header, body=body, data_key=data_key), keyring=self.keyring
        )
        base = Base(
            etag=state.etag if state is not None else None,
            commit_id=state.header.commit_id if state is not None else None,
        )
        superseded = (
            [body_key(instance_id, e.commit_id, _OBJECT_KIND[e.kind]) for e in state.body.chain]
            if state is not None and kind == "full"
            else []
        )
        try:
            # Shielded with its store cleanup: a cancelled caller must neither cut a
            # head write off half-known nor skip the deletes its outcome allows.
            commit_task = self._in_background(self._commit(commit, base, uploaded=object_key, superseded=superseded))
            etag = await asyncio.shield(commit_task)
        except BaseException:
            self._drop_new_files(state, folder, vmstate, home)
            raise

        if state is not None:  # committed: not awaited, so neither a file error nor a cancel turns it into an error
            self._remove_in_background(state.vmstate, state.home)
        logger.info(
            "Instance saved",
            extra={"instance_id": instance_id, "generation": generation, "kind": kind, "object_bytes": size},
        )
        return InstanceState(
            header=header,
            body=body,
            etag=etag,
            data_key=data_key,
            folder=folder,
            vmstate=vmstate,
            home=home,
            base_image=vm.base_image,
            packages_image=vm.packages_image,
        )

    async def _commit(self, commit: Commit, base: Base, *, uploaded: str, superseded: list[str]) -> str | None:
        """commit_head, then the store deletes its outcome allows. An unknown outcome
        keeps the uploaded object: the head may name it. A first save then checks
        that its object is still in the store (see save())."""
        instance_id = commit.header.instance_id
        try:
            etag = await commit_head(self.store, commit, base)
        except (OutcomeUnknownError, InstanceGoneError) as e:
            # By code: INSTANCE_CONFLICT and INSTANCE_SAVE_OUTCOME_UNKNOWN share a class, and
            # only a conflict (or a gone instance) is definitely not committed: ours to remove.
            if isinstance(e, InstanceGoneError) or e.code is ErrorCode.INSTANCE_CONFLICT:
                await self._delete_quietly(instance_id, [uploaded])
            raise
        if base.etag is None and etag is not None and not await self._still_stored(commit.header, uploaded):
            # A delete_instance found no head yet and removed the object: the new head names nothing.
            await self._delete_quietly(instance_id, [head_key(instance_id)])
            raise InstanceGoneError(
                f"Instance {instance_id} was deleted during this save (its uploaded state is gone from the store). "
                "Nothing was saved.",
                code=ErrorCode.INSTANCE_NOT_FOUND,
                instance_id=instance_id,
                hint="Start again with a new instance.",
                context=None,
            )
        if superseded:
            # The new chain needs none of the old objects; a load still reading
            # them finds them gone, re-reads the head and follows the new chain.
            await self._delete_quietly(instance_id, superseded)
        return etag

    async def _delete_quietly(self, instance_id: str, keys: list[str]) -> None:
        """Best effort: what stays is removed by delete_instance or an expiry rule of the store."""
        try:
            await self.store.delete_many(keys)
        except (TransientError, ConfigError) as e:
            logger.warning(
                "Instance objects not deleted (delete_instance or an expiry rule removes them)",
                extra={"instance_id": instance_id, "count": len(keys), "error": str(e)},
            )

    async def _still_stored(self, header: HeadHeader, key: str) -> bool:
        """`key` is in the store. The head is written already: a store that does not
        answer leaves the outcome unknown (OutcomeUnknownError INSTANCE_SAVE_OUTCOME_UNKNOWN)."""
        try:
            return await self.store.exists(key)
        except (TransientError, ConfigError) as e:
            raise OutcomeUnknownError(
                f"Instance {header.instance_id}: generation {header.generation} (commit {header.commit_id}) "
                "was written, but the store did not tell if its saved state is still there.",
                code=ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN,
                instance_id=header.instance_id,
                hint="Load the instance before you redo the work.",
                context=save_context(generation=header.generation, commit_id=header.commit_id),
            ) from e

    def _drop_new_files(self, state: InstanceState | None, folder: Path, *files: Path) -> None:
        """A save that did not (or may not have) become this session's base: drop its
        files in a worker thread (drain() waits for it)."""
        if state is None:  # first save: the folder holds nothing else
            self._remove_in_background(folder)
        else:
            self._remove_in_background(*files)

    def _next_header(
        self, state: InstanceState | None, *, instance_id: str, commit_id: str, data_key: bytes
    ) -> HeadHeader:
        """Header of the next generation. Rotation on use: every save wraps the data key with the newest key."""
        kek = self.keyring.latest
        generation = state.header.generation + 1 if state is not None else 1
        older = state.header.recent_commits if state is not None else ()
        return HeadHeader(
            instance_id=instance_id,
            generation=generation,
            commit_id=commit_id,
            recent_commits=(RecentCommit(generation=generation, commit_id=commit_id), *older)[:RECENT_COMMITS],
            key_id=kek.id,
            wrapped_data_key=wrap_data_key(kek, data_key).hex(),
        )

    async def _snapshot(self, vm: QemuVM, *, vmstate: Path, home: Path) -> None:
        """Migration file and a copy of the home disk, both of one stopped instant."""
        qemu_version = await probe_qemu_version()
        new_stream_file(vmstate)
        if vm.use_qemu_vm_user and not await grant_qemu_vm_private_file_access(vmstate):
            raise ConfigError(
                "Could not give qemu-vm write access to the migration file (needs setfacl, package acl)",
                code=ErrorCode.HOST_DEPENDENCY,
                context={"vm_id": vm.vm_id},
                hint=None,
                instance_id=None,
            )

        async def copy_home() -> None:  # the VM is stopped: the copy matches the saved RAM
            await asyncio.to_thread(clone_sparse, vm.workdir.home_disk, home)

        async with MigrationClient(vm.qmp_socket, get_expected_socket_uid(vm.use_qemu_vm_user)) as client:
            try:
                await client.save_snapshot(
                    vmstate,
                    qemu_version=qemu_version,
                    timeout=migration_timeout(vm.spec.memory_mb),
                    use_template=False,
                    keep_running=True,
                    while_paused=copy_home,
                    memory_limited=vm.cgroup is not None,
                )
            except TransientError as e:
                # A QMP deadline (QEMU sent no answer in time): the run state, CPU time and
                # threads of QEMU and the console tell a blocked main loop from a slow one
                if isinstance(e.__cause__, TimeoutError):
                    e.context.update(await vm.deadline_diagnostics(operation="save"))
                raise

    async def _chain_intact(self, instance_id: str, chain: tuple[ChainEntry, ...]) -> bool:
        """One listing: every object of the chain is in the store with the size and
        the ETag of its upload. A missing or changed object makes this save a full copy."""
        listed = {o.key: o for o in await self.store.list_objects(body_prefix(instance_id))}
        for entry in chain:
            key = body_key(instance_id, entry.commit_id, _OBJECT_KIND[entry.kind])
            found = listed.get(key)
            if found is None or (found.size, found.etag) != (entry.size, entry.etag):
                logger.warning(
                    "Instance chain object missing or changed in the store; this save writes a full copy",
                    extra={"instance_id": instance_id, "key": key, "found": found is not None},
                )
                return False
        return True

    async def _write_body(
        self,
        *,
        instance_id: str,
        commit_id: str,
        state: InstanceState | None,
        vmstate: Path,
        home: Path,
    ) -> tuple[Literal["full", "pack"], Path, int, bytes]:
        """The body object of this save, its size and the data key that seals it.

        A pack (sealed with the chain's data key) when there is a chain, the
        newest keyring key wraps its data key, the chain has room
        (MAX_CHAIN_PACKS), one listing finds every object of the chain unchanged
        in the store (key, size, ETag), and the packs stay within the budget (the
        full copy's size). Else a full copy with a new data key: a retired key
        never reads a chain started after its retirement.
        """
        folder = vmstate.parent
        if (
            state is not None
            and state.header.key_id == self.keyring.latest.id
            and len(state.body.chain) - 1 < MAX_CHAIN_PACKS
            and await self._chain_intact(instance_id, state.body.chain)
        ):
            pack = folder / f"{commit_id}.p"
            size = await asyncio.to_thread(
                write_object,
                pack,
                data_key=state.data_key,
                instance_id=instance_id,
                commit_id=commit_id,
                streams={STREAM_VMSTATE: (vmstate, state.vmstate), STREAM_HOME: (home, state.home)},
            )
            if sum(entry.size for entry in state.body.chain[1:]) + size <= state.body.chain[0].size:
                return "pack", pack, size, state.data_key
            self._remove_in_background(pack)
        data_key = os.urandom(KEY_BYTES)
        full = folder / f"{commit_id}.f"
        size = await asyncio.to_thread(
            write_object,
            full,
            data_key=data_key,
            instance_id=instance_id,
            commit_id=commit_id,
            streams={STREAM_VMSTATE: (vmstate, None), STREAM_HOME: (home, None)},
        )
        return "full", full, size, data_key

    # ------------------------------------------------------------------ load

    async def load(
        self, instance_id: str, *, packages_image: Callable[[VmSpec], Awaitable[Path | None]]
    ) -> InstanceState:
        """Head, host check (ROUTE before any download), then the chain as local files.

        `packages_image` finds the local L2 image for the stored spec (None: none here).
        Does not check the id (callers do). Raises InstanceGoneError
        INSTANCE_NOT_FOUND when there is no head, ConfigError INSTANCE_KEY_MISSING when
        this keyring lacks its key, InstanceGoneError INSTANCE_INTEGRITY or RouteError
        INSTANCE_HOST_MISMATCH (another release) when it does not open, and the store's
        errors; RouteError INSTANCE_HOST_MISMATCH when the host facts differ (arch,
        accelerator, CPU vendor, guest CPU model, or the exec-sandbox release that saved
        it) or a guest image is neither on this worker nor in the store (or the store's
        copy does not match its digest); InstanceGoneError INSTANCE_EXPIRED when a body object stays
        gone; TransientError INSTANCE_STORE_UNAVAILABLE when the head keeps changing;
        TransientError LOCAL_DISK for a local file error (nothing was loaded).
        """
        previous_etag: str | None = None
        missing: str | None = None
        with _named(instance_id):
            for _ in range(_LOAD_ATTEMPTS):
                got = await self.store.get(head_key(instance_id), max_bytes=MAX_HEAD_BYTES)
                if got is None:
                    raise _not_found(instance_id)
                data, etag = got
                header, body, data_key = open_head(data, self.keyring, instance_id=instance_id)
                if etag == previous_etag and missing is not None:
                    if await self.store.exists(missing):  # one false "no such key" never makes an instance expire
                        raise TransientError(
                            f"Instance {instance_id}: the store answered 'no such key' for {missing}, which it has.",
                            code=ErrorCode.INSTANCE_STORE_UNAVAILABLE,
                            instance_id=instance_id,
                            hint="The store failed; the same call can succeed later.",
                            context=None,
                        )
                    raise InstanceGoneError(
                        f"Instance {instance_id}: part of its saved state is gone from the store "
                        "(an expiry rule or a manual delete removed it).",
                        code=ErrorCode.INSTANCE_EXPIRED,
                        instance_id=instance_id,
                        hint="Start again with Scheduler.new().",
                        context=None,
                    )
                await self._check_host(instance_id, body)
                with _local_files(instance_id, step="loading"):
                    base_image, packages = await self._resolve_images(instance_id, body, packages_image)
                    folder = self._new_folder(instance_id)
                    try:
                        vmstate, home = await self._download(header, body, data_key, folder)
                    except _VanishedError as e:
                        self._remove_in_background(folder)
                        previous_etag, missing = etag, e.key
                        continue
                    except BaseException:
                        self._remove_in_background(folder)
                        raise
                    return InstanceState(
                        header=header,
                        body=body,
                        etag=etag,
                        data_key=data_key,
                        folder=folder,
                        vmstate=vmstate,
                        home=home,
                        base_image=base_image,
                        packages_image=packages,
                    )
        raise TransientError(
            f"Instance {instance_id}: the head kept changing while it was loaded.",
            code=ErrorCode.INSTANCE_STORE_UNAVAILABLE,
            instance_id=instance_id,
            hint="Other workers save this instance right now; the same call can succeed later.",
            context=None,
        )

    async def _check_host(self, instance_id: str, body: HeadBody) -> None:
        local = await self._machine_facts()
        saved = body.host.model_dump()
        differ = [name for name in _CHECKED_FACTS if saved[name] != local[name]]
        if differ:
            raise host_mismatch(
                instance_id=instance_id,
                required={name: str(saved[name]) for name in differ},
                found={name: str(local[name]) for name in differ},
            )

    async def _download(self, header: HeadHeader, body: HeadBody, data_key: bytes, folder: Path) -> tuple[Path, Path]:
        """The chain into two new stream files, opened once. At most _OBJECTS_AHEAD
        objects are downloading or on local disk at once, the applied one included;
        objects are applied in chain order, one at a time, each in a worker thread
        that then removes it. The first missing object in chain order raises
        _VanishedError. On an error or a cancel, the downloads stop; load() then
        starts the removal of the folder with every file in it (drain() waits for it)."""
        instance_id = header.instance_id
        vmstate, home = folder / f"{header.commit_id}.vmstate", folder / f"{header.commit_id}.home"
        new_stream_file(vmstate)
        new_stream_file(home)
        objects = [folder / f"{entry.commit_id}.{_OBJECT_KIND[entry.kind]}" for entry in body.chain]

        async def fetch(n: int) -> None:
            entry = body.chain[n]
            key = body_key(instance_id, entry.commit_id, _OBJECT_KIND[entry.kind])
            if not await self.store.download(key, dest=objects[n], max_bytes=entry.size, expected_size=entry.size):
                raise _VanishedError(key)

        def apply(n: int, targets: Mapping[int, int]) -> None:
            try:
                apply_object(
                    objects[n],
                    data_key=data_key,
                    instance_id=instance_id,
                    commit_id=body.chain[n].commit_id,
                    targets=targets,
                    sizes=dict(enumerate(body.chain[n].stream_sizes)),
                )
            finally:
                _remove_quietly(objects[n])

        fetches: list[asyncio.Task[None]] = []
        applying: asyncio.Future[None] | None = None
        targets: dict[int, int] = {}
        try:
            for stream, path in ((STREAM_VMSTATE, vmstate), (STREAM_HOME, home)):
                targets[stream] = os.open(path, os.O_WRONLY)
            for n in range(len(objects)):
                # A slot frees once its object is applied and removed.
                while len(fetches) < min(n + _OBJECTS_AHEAD, len(objects)):
                    fetches.append(asyncio.ensure_future(fetch(len(fetches))))
                await fetches[n]
                # An executor future, not a task: loop shutdown (asyncio.run) cancels every
                # task, and a cancelled task reads as done while its thread still writes.
                applying = asyncio.get_running_loop().run_in_executor(None, apply, n, targets)
                await asyncio.shield(applying)
        except BaseException:
            for fetching in fetches:
                fetching.cancel()
            await asyncio.gather(*fetches, return_exceptions=True)
            raise
        finally:
            _close_after(applying, list(targets.values()))
        if await probe_qemu_vm_user() and not await grant_qemu_vm_private_file_access(vmstate):
            raise ConfigError(
                "Could not give qemu-vm read access to the migration file (needs setfacl, package acl)",
                code=ErrorCode.HOST_DEPENDENCY,
                context={"instance_id": instance_id},
                hint=None,
                instance_id=None,
            )
        return vmstate, home

    # ------------------------------------------------------------------ assets

    async def _ensure_vm_assets(self, vm: QemuVM, *, facts: HostFacts) -> None:
        """Store the VM's guest images by digest before the commit (at commit time,
        the store holds every image the head names)."""
        await self._ensure_asset(vm.base_image, facts.base_image)
        if vm.packages_image is not None and facts.packages_image is not None:
            await self._ensure_asset(vm.packages_image, facts.packages_image)

    async def _ensure_asset(self, path: Path, digest: str) -> None:
        """Upload a guest image under its digest unless the store has it at its size.
        Checked at every save (one HEAD): an image deleted or cut in the store comes
        back. A same-size change is not seen here; a load checks the digest (ROUTE).

        Before an upload, the image is hashed again: its recorded digest comes from
        image_digest at the VM's start, which trusts a sidecar keyed by the file's
        inode, mtime and size, so a damaged image with the same stamp keeps its old
        digest. ConfigError IMAGE_DAMAGED then: nothing is stored under a digest the
        bytes do not have."""
        key = asset_key(digest)
        if await self.store.size(key) != path.stat().st_size:  # missing or cut: (re)upload
            if await file_hash(path) != digest:
                raise ConfigError(
                    f"Guest image {path} does not match its recorded digest {digest} (changed or damaged). "
                    "It was not stored.",
                    code=ErrorCode.IMAGE_DAMAGED,
                    context={"path": str(path), "digest": digest},
                    hint="Download or rebuild the guest images.",
                    instance_id=None,
                )
            await self.store.upload(key, src=path)
            logger.info("Instance asset uploaded", extra={"digest": digest, "bytes": path.stat().st_size})

    async def _resolve_images(
        self, instance_id: str, body: HeadBody, packages_image: Callable[[VmSpec], Awaitable[Path | None]]
    ) -> tuple[Path, Path | None]:
        base = await self._resolve_image(
            instance_id, role="base_image", digest=body.host.base_image, local=self._local_base_image(body.spec)
        )
        if body.host.packages_image is None:
            return base, None
        packages = await self._resolve_image(
            instance_id, role="packages_image", digest=body.host.packages_image, local=await packages_image(body.spec)
        )
        return base, packages

    async def _resolve_image(self, instance_id: str, *, role: str, digest: str, local: Path | None) -> Path:
        """A local image with `digest`: the worker's own, a cached download, or a new
        download from the store (checked against the digest). ROUTE when none."""
        if local is not None:
            try:
                same = await asyncio.to_thread(image_digest, local) == digest
            except OSError as e:  # unreadable here (no access, an I/O error): no usable local copy
                logger.warning(
                    "Local guest image unreadable", extra={"image": str(local), "role": role, "error": str(e)}
                )
                same = False
            if same:
                return local
        cached = self.asset_dir / f"{digest}.img"
        if cached.exists():  # only ever renamed into place after its digest was checked
            return cached
        part = self.asset_dir / f"{digest}.{ids.new_id()}.part"
        try:
            if not await self.store.download(
                asset_key(digest), dest=part, max_bytes=_MAX_ASSET_BYTES, expected_size=None
            ):
                problem = "not on this worker, not in the store"
            elif await file_hash(part) != digest:
                problem = "the store's copy does not match its digest"
            else:
                part.replace(cached)
                logger.info("Instance asset downloaded", extra={"digest": digest, "role": role})
                return cached
        finally:
            self._remove_in_background(part)  # up to GBs; gone already after the rename
        # The instance is intact: a worker that has the image can load it.
        raise host_mismatch(instance_id=instance_id, required={role: digest}, found={role: problem})

    def _local_base_image(self, spec: VmSpec) -> Path | None:
        try:
            return self._vm_manager.images.base_image(spec.language)
        except (ConfigError, FileNotFoundError):
            return None

    # ------------------------------------------------------------------ helpers

    async def release(self, state: InstanceState) -> None:
        """Remove a closed session's instance files (what will not go is logged)."""
        await asyncio.to_thread(_remove_quietly, state.folder)

    def _new_folder(self, instance_id: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{instance_id}-", dir=self.work_dir))

    async def _machine_facts(self) -> dict[str, str | None]:
        """This host: arch, accelerator, CPU vendor (x86 only), guest CPU model, QEMU version and
        exec-sandbox release."""
        arch = detect_host_arch()
        accel = await detect_accel_type(force_emulation=self._vm_manager.settings.force_emulation)
        qemu = await probe_qemu_version()
        return {
            "arch": arch.name.lower(),
            "accelerator": str(accel.value),
            "cpu_vendor": detect_host_x86_vendor() if arch == HostArch.X86_64 else None,
            "cpu_model": await guest_cpu_model(arch, accel),
            "qemu_version": ".".join(map(str, qemu)) if qemu else "unknown",
            "release": __version__,
        }

    async def _host_facts(self, vm: QemuVM) -> HostFacts:
        """The saved VM's facts: this host, and the digests of the images the VM runs on,
        taken when QEMU opened them (after reuse() these may differ from this worker's own
        images; a file evicted or rebuilt at the same path since does not change them).

        Raises:
            ConfigError: IMAGE_DAMAGED: a digest is unknown (the image was missing at the
                VM's start); nothing is saved
        """
        base_digest = vm.base_image_digest
        packages_digest = vm.packages_image_digest
        if base_digest is None or (vm.packages_image is not None and packages_digest is None):
            raise ConfigError(
                f"VM {vm.vm_id}: the digest of a guest image it runs on is unknown (the file was missing "
                "when the VM started). It was not saved.",
                code=ErrorCode.IMAGE_DAMAGED,
                context={"base_image": str(vm.base_image), "packages_image": str(vm.packages_image)},
                hint="Download or rebuild the guest images.",
                instance_id=None,
            )
        return HostFacts.model_validate(
            {**await self._machine_facts(), "base_image": base_digest, "packages_image": packages_digest}
        )


@contextlib.contextmanager
def _local_files(instance_id: str, *, step: Literal["saving", "loading"]) -> Generator[None]:
    """A save or a load on local files (work folder, asset folder, a store's `src` or
    `dest`): an OSError (local disk full, a folder gone at shutdown) becomes
    TransientError LOCAL_DISK. The block starts the removal of its own partial files (drain()
    waits for it) before the error leaves it."""
    try:
        yield
    except OSError as e:
        raise TransientError(
            f"Instance {instance_id}: {step} needs local disk space and files ({e.strerror or e}). "
            f"Nothing was {'saved' if step == 'saving' else 'loaded'}.",
            code=ErrorCode.LOCAL_DISK,
            context={"instance_id": instance_id, "errno": e.errno},
            hint=None,
            instance_id=instance_id,
        ) from e


def _remove_quietly(*paths: Path) -> None:
    """Remove work files and folders without hiding the outcome they follow (a
    committed save, the error being raised): what will not go is logged. release()
    removes the rest of a work folder; an asset download's `.part` file stays in the
    asset folder until it is removed by hand. Blocking: async code uses
    InstanceManager._remove_in_background."""

    def not_removed(_function: object, path: object, error: BaseException) -> None:
        if not isinstance(error, FileNotFoundError):  # gone already: removed meanwhile
            logger.warning("Instance work file not removed", extra={"path": str(path), "error": str(error)})

    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, onexc=not_removed)
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            not_removed(None, path, e)


def _close_after(job: asyncio.Future[None] | None, fds: Collection[int]) -> None:
    """Close `fds` once `job` (a worker thread that writes to them) has ended, now
    when it has: a closed fd number the process reuses must never get its bytes."""

    def close(_: object) -> None:
        for fd in fds:
            os.close(fd)

    if job is None or job.done():
        close(None)
    else:
        job.add_done_callback(close)


def _not_found(instance_id: str) -> InstanceGoneError:
    return InstanceGoneError(
        f"No instance {instance_id} in the store (never saved, deleted or expired).",
        code=ErrorCode.INSTANCE_NOT_FOUND,
        instance_id=instance_id,
        hint="Start again with Scheduler.new().",
        context=None,
    )


@contextlib.contextmanager
def _named(instance_id: str) -> Generator[None]:
    """Store errors do not know which instance they served: name it for the caller."""
    try:
        yield
    except SandboxError as e:
        if e.instance_id is None:
            e.instance_id = instance_id
        raise

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
import errno
import itertools
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from exec_sandbox import constants
from exec_sandbox._imports import require_s3_instance_store
from exec_sandbox._version import __version__
from exec_sandbox.aio_utils import run_stoppable
from exec_sandbox.base_cache_manager import BaseCacheManager
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InstanceGoneError,
    InternalError,
    InvalidInputError,
    OutcomeUnknownError,
    RouteError,
    RouteKey,
    SandboxError,
    TransientError,
    host_mismatch,
    named_instance,
    no_instance_store,
)
from exec_sandbox.files.disk import clone_sparse, memory_filesystem
from exec_sandbox.files.locks import CacheEntry, EntryLease, sharing
from exec_sandbox.hash_utils import file_hash
from exec_sandbox.images import image_digest
from exec_sandbox.instance import ids
from exec_sandbox.instance.commit import (
    GONE_DURING_SAVE,
    LANDED_FIRST,
    RETRY_SAME_SAVE,
    SAVE_ELSEWHERE,
    Base,
    Commit,
    commit_head,
    save_context,
)
from exec_sandbox.instance.head import MAX_HEAD_BYTES, describe_head, open_head, read_summary, seal_head
from exec_sandbox.instance.keyring import KEY_BYTES, Keyring, wrap_data_key
from exec_sandbox.instance.local_store import LocalInstanceStore
from exec_sandbox.instance.models import (
    BLOCK_BYTES,
    RECENT_COMMITS,
    STREAM_HOME,
    STREAM_VMSTATE,
    ChainEntry,
    GuestFacts,
    HeadBody,
    HeadHeader,
    HeadSummary,
    HostFacts,
    InstanceAccess,
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
from exec_sandbox.memory_snapshot_files import home_path_from_vmstate
from exec_sandbox.migration_client import MigrationClient, migration_timeout
from exec_sandbox.permission_utils import (
    get_expected_socket_uid,
    grant_qemu_vm_acl,
    probe_qemu_vm_user,
)
from exec_sandbox.settings import kept_secret
from exec_sandbox.system_probes import require_qemu_machine_types, require_qemu_version
from exec_sandbox.worker_facts import counter_fact, machine_facts, qemu_version_fact

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Coroutine, Generator, Mapping
    from typing import Any

    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.settings import Settings
    from exec_sandbox.vm_manager import VmManager
    from exec_sandbox.vm_spec import VmSpec

logger = logging.getLogger(__name__)

_LOAD_ATTEMPTS = 3
"""Head reads per load while body objects vanish because newer saves replaced them."""
_OBJECT_KIND = {"full": "f", "pack": "p"}
_CHECKED_FACTS = (RouteKey.ARCH, RouteKey.ACCELERATOR, RouteKey.CPU_MODEL, RouteKey.CPU_ID)
"""Host facts a load must match (with the machine type, rule CONTAINS, and the guest's
counter rate, rule EQUAL_OR_ANY). cpu_model and cpu_id: QEMU refuses a state saved with
other CPU features, but only at restore, after the download (HVF may load it silently).
Images are fetched by digest. qemu_version is not checked: QEMU accepts or refuses the
saved state at restore. The exec-sandbox release is not checked: a release loads the
instances of another one; data it cannot read raises RouteError (another release)."""
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


class AssetCache(BaseCacheManager):
    """The guest images that loads fetched from the store, by digest (`<digest>.img`): under
    the disk-pressure eviction of admission (VmManager.add_cache), oldest first, never one
    a VM runs on or a load holds (EntryLease)."""

    _cache_ext: ClassVar[str] = ".img"


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceState:
    """What a session knows about the instance it continues: the head it loaded
    or committed last, and that state as local files (the baseline of the next pack)."""

    header: HeadHeader
    body: HeadBody
    etag: str | None
    """ETag of this session's head, the base of its next save. None: another save is on
    top of this session's head (it committed and was superseded, or it lost a save race),
    so the next save conflicts before it sends anything."""
    data_key: bytes = field(repr=False)
    folder: Path
    """Private folder of this session's instance files (removed when the session closes,
    when it saves under another id, or once another save is on top of this head: no later
    save reads them)."""
    vmstate: Path
    home: Path
    base_image: Path
    """Local base image with the saved digest (the overlay's backing file)."""
    packages_image: Path | None
    """Local L2 packages image with the saved digest, None without packages."""
    asset_leases: tuple[EntryLease, ...]
    """Leases on the images of the asset cache this state runs on (eviction skips them):
    the restore's VM takes its own shares; release() closes these."""

    @property
    def instance_id(self) -> str:
        return self.header.instance_id


def instance_keyring(settings: Settings) -> Keyring:
    """The keyring of the instance store, parsed (no store call). Raises ConfigError
    INSTANCE_CONFIG when it is not set or not valid (Keyring.parse)."""
    if (secret := kept_secret(settings.instance_keyring)) is None:  # a SecretStr already: the type narrows
        raise ConfigError(
            "instance_store is set, but instance_keyring is not: saved instances are sealed with its keys.",
            code=ErrorCode.INSTANCE_CONFIG,
            instance_id=None,
            hint="Set instance_keyring (EXEC_SANDBOX_INSTANCE_KEYRING), or unset instance_store.",
            context=None,
        )
    return Keyring.parse(secret.get_secret_value())


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
    keyring = instance_keyring(settings)
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
        """Public header always; the sealed part when this keyring has the key and this
        release reads the body (InstanceInfo.access).

        Raises InvalidInputError INSTANCE_INVALID_ID (before any store call),
        InstanceGoneError INSTANCE_NOT_FOUND when there is no head, InstanceGoneError
        INSTANCE_INTEGRITY when the head fails authentication, ConfigError
        INSTANCE_KEY_MISSING when this keyring lacks the key and this release cannot
        read even the summary of the public header (do not delete: another release
        may have written it), RouteError INSTANCE_HOST_MISMATCH when the head
        authenticates but even its summary (generation, recent commits, key id) does
        not parse, and the store's errors (TransientError INSTANCE_STORE_UNAVAILABLE,
        ConfigError INSTANCE_CONFIG).
        """
        ids.check_instance_id(instance_id)
        with named_instance(instance_id):
            with _object_named(head_key(instance_id), None):  # an oversized head is INSTANCE_INTEGRITY
                got = await self.store.get(head_key(instance_id), max_bytes=MAX_HEAD_BYTES)
            if got is None:
                raise _not_found(instance_id, store=self.store)
            # One read: `access` speaks for the very bytes the other fields come from.
            try:
                with _object_named(head_key(instance_id), None):
                    summary, body, writer = describe_head(got[0], self.keyring, instance_id=instance_id)
            except ConfigError as missing:  # INSTANCE_KEY_MISSING: the only ConfigError of describe_head
                try:
                    summary = read_summary(got[0], instance_id=instance_id)
                except InstanceGoneError:  # unauthenticated and unreadable here: maybe another release's
                    raise missing from None
                return _info(summary, access="no_key", writer_release=None, body=None)
            if body is None:  # authenticated, written by another release: the data is intact
                return _info(summary, access="other_release", writer_release=writer, body=None)
            return _info(summary, access="full", writer_release=writer, body=body)

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
        with named_instance(instance_id):
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

    async def list_assets(self) -> list[str]:
        """Digests (full SHA-256) of the guest images in the store. A save stores its images
        before its head: a cleanup that deletes the images no head names must keep those
        newer than its longest save."""
        listed = await self.store.list_objects(asset_key(""))
        names = (entry.key.removeprefix(asset_key("")) for entry in listed)
        return [name for name in names if re.fullmatch(r"[0-9a-f]{64}", name)]

    async def delete_asset(self, digest: str) -> bool:
        """Delete the guest image `digest` from the store; False: it was not there. An
        instance whose body names it then loads only on a worker that has the image (else
        RouteError). Raises InvalidInputError ARGUMENT_INVALID for a digest that is not
        64 lowercase hex characters, before any store call."""
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise InvalidInputError(
                "An image digest is 64 lowercase hex characters (SHA-256).",
                code=ErrorCode.ARGUMENT_INVALID,
                context=None,
                hint=None,
                instance_id=None,
            )
        existed = await self.store.exists(asset_key(digest))
        await self.store.delete_many([asset_key(digest)])
        return existed

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
        self, *, store: InstanceStore, keyring: Keyring, work_dir: Path, assets: AssetCache, vm_manager: VmManager
    ) -> None:
        super().__init__(store=store, keyring=keyring)
        self.work_dir = work_dir
        self.assets = assets
        """Images from the store; the Scheduler gives it to admission's eviction."""
        self._vm_manager = vm_manager
        self._calls = 0
        """Saves and loads in flight (they use the store, the VM and the work folder)."""
        self._idle = asyncio.Event()
        self._idle.set()
        self._stopping = False
        """stop() began: no save or load starts."""
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
        memory filesystem), asset cache folder, store probe. Each raises ConfigError
        INSTANCE_CONFIG (FIX_CONFIG; MIGRATION_UNSUPPORTED for a QEMU below the floor), or
        TransientError INSTANCE_STORE_UNAVAILABLE when the store does not answer. The
        QEMU version probe raises ConfigError HOST_DEPENDENCY or TransientError
        QEMU_CRASHED (require_qemu_version). A store whose extra is not installed raises
        ConfigError EXTRA_MISSING. The asset cache folder (instance_asset_dir) raises
        ConfigError SETTING_INVALID when it cannot be made or is not private,
        TransientError LOCAL_DISK for a disk error. A file error of the work folder (its
        mkdir, the probe's scratch file) is an OSError. When the
        block ends, the saves, loads and background jobs still in flight end (drain)
        before the store closes."""
        async with _open_store(settings) as (store, keyring):
            qemu = await require_qemu_version()
            if qemu < constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION:
                floor = ".".join(map(str, constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION))
                found = ".".join(map(str, qemu))
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
            assets = AssetCache(settings.instance_asset_dir, setting="instance_asset_dir")
            await probe(store, scratch=work_dir)
            key_ids = [key.id for key in keyring.keys]
            logger.info(
                f"Instance store ready ({settings.instance_store}), key ids newest first: {', '.join(key_ids)}.",
                extra={"store": settings.instance_store, "key_ids": key_ids},
            )
            manager = cls(
                store=store,
                keyring=keyring,
                work_dir=work_dir,
                assets=assets,
                vm_manager=vm_manager,
            )
            try:
                yield manager
            finally:
                await manager.stop()  # the store closes once no call or background job uses it

    # ------------------------------------------------------------------ save

    async def save(
        self, vm: QemuVM, state: InstanceState | None, *, new_instance_id: str | None, resumed: Callable[[], None]
    ) -> InstanceState:
        """Save the running VM as the next generation (the first when `state` is None).

        A first save uses `new_instance_id`, or a new random id when it is None.
        The VM keeps running: `resumed()` runs at most once, when the VM runs again after its
        pause and its state is in local files (nothing after it reads the guest's disk or
        RAM); it does not run when the pause or the write fails. A MIGRATION_FAILED error has
        context["vm_resumed"] True when the VM runs on (also when the QMP connection fails
        before the pause). Raises OutcomeUnknownError (INSTANCE_CONFLICT: another
        save won, or a first save's id is taken; INSTANCE_SAVE_OUTCOME_UNKNOWN), or
        InstanceGoneError INSTANCE_NOT_FOUND (deleted meanwhile): the VM runs on, and the
        caller's session stays open on those. TransientError LOCAL_DISK: local disk full or files gone;
        nothing was saved. ConfigError IMAGE_DAMAGED: a guest image does not match its
        digest, or its digest is unknown (the file was missing at the VM's start); nothing
        was saved. ConfigError HOST_DEPENDENCY: QEMU does not list the
        VM's machine type, or qemu-vm cannot get write access to the migration file;
        ConfigError MIGRATION_UNSUPPORTED or TransientError MIGRATION_FAILED: the migration failed
        (MigrationClient.save_snapshot); InvalidInputError SCHEDULER_STATE: stop() began;
        nothing was saved. The store's errors (TransientError INSTANCE_STORE_UNAVAILABLE,
        ConfigError INSTANCE_CONFIG) of an upload or of the head write: no head write of
        this save applied.
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
        with self._in_flight(), named_instance(instance_id):
            return await self._save(vm, state, instance_id=instance_id, resumed=resumed)

    async def stop(self) -> None:
        """Refuse new saves and loads (InvalidInputError SCHEDULER_STATE), then wait until
        the calls and background jobs in flight end (drain). Shutdown calls it before the VMs,
        the work folder and the store clients go: drain() alone waits on for as long as
        sessions start new saves."""
        self._stopping = True
        await self.drain()

    async def drain(self) -> None:
        """Wait until no save, no load and no background job runs: a head write runs on
        after its save is cancelled, and the work files of a save or a load (folders, body
        objects, an asset's .part file) are removed in a worker thread after it returns.
        A call that starts meanwhile is waited for too. stop() calls it once it refuses
        new saves and loads."""
        while not (self._idle.is_set() and all(job.done() for job in self._background)):
            await self._idle.wait()
            # Only jobs that have not ended: a gather of ended jobs completes without a yield,
            # so their discard callbacks never run and a loop on `self._background` spins.
            await asyncio.gather(*[job for job in self._background if not job.done()], return_exceptions=True)

    @contextlib.contextmanager
    def _in_flight(self) -> Generator[None]:
        """A save or a load: drain() waits for it. Refused once stop() began."""
        if self._stopping:
            raise InvalidInputError(
                "The scheduler is stopping: no save or load starts now",
                code=ErrorCode.SCHEDULER_STATE,
                context=None,
                hint=None,
                instance_id=None,
            )
        self._calls += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._calls -= 1
            if not self._calls:
                self._idle.set()

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

    async def _save(
        self, vm: QemuVM, state: InstanceState | None, *, instance_id: str, resumed: Callable[[], None]
    ) -> InstanceState:
        """save() itself (save() counts it as in flight)."""
        commit_id = ids.new_id()
        if state is not None and state.etag is None:
            raise OutcomeUnknownError(
                f"Instance {state.instance_id}: another save committed after this session's last save. "
                "This save was not stored.",
                code=ErrorCode.INSTANCE_CONFLICT,
                instance_id=state.instance_id,
                hint=LANDED_FIRST + SAVE_ELSEWHERE,
                context=save_context(generation=state.header.generation + 1, commit_id=commit_id),
            )
        obj: Path | None = None
        # Shares of the VM's cache leases: a close() after resumed() may end the VM's own
        # while this save still reads its images (_ensure_vm_assets)
        with sharing(vm.cache_leases), _local_files(instance_id, step="saving"):
            folder = state.folder if state is not None else self._new_folder(instance_id)
            vmstate = folder / f"{commit_id}.vmstate"
            home = home_path_from_vmstate(vmstate)  # where VmManager.restore_vm looks for it
            try:
                await self._snapshot(vm, vmstate=vmstate, home=home)
                # The VM runs again and its state is in local files: the caller may let guest calls
                # run (nothing below reads the guest's disk or RAM)
                resumed()
                facts = await self._host_facts(vm)
                guest = self._guest_facts(vm)
                await self._ensure_vm_assets(vm, guest=guest)
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

        entry = ChainEntry(
            block_bytes=BLOCK_BYTES,
            commit_id=commit_id,
            kind=kind,
            size=size,
            stream_sizes=stream_sizes,
            etag=object_etag,
        )
        chain = (entry,) if kind == "full" or state is None else (*state.body.chain, entry)
        body = HeadBody(spec=vm.spec, host=facts, guest=guest, egress_domains=vm.egress_domains, chain=chain)
        header = self._next_header(state, instance_id=instance_id, commit_id=commit_id, data_key=data_key)
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
            extra={"instance_id": instance_id, "generation": header.generation, "kind": kind, "object_bytes": size},
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
            asset_leases=(),  # the VM holds its own shares of the images it runs on
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
                hint=GONE_DURING_SAVE,
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
                hint="Its head is written, but its saved state may be gone from the store: reuse(id) tells "
                "(InstanceGoneError INSTANCE_EXPIRED when it is gone). " + RETRY_SAME_SAVE,
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
        qemu_version = await require_qemu_version()
        new_stream_file(vmstate)
        if vm.use_qemu_vm_user and not await grant_qemu_vm_acl([vmstate], perms="rw"):
            raise ConfigError(
                "Could not give qemu-vm write access to the migration file (needs setfacl, package acl)",
                code=ErrorCode.HOST_DEPENDENCY,
                context={"vm_id": vm.vm_id},
                hint=None,
                instance_id=None,
            )

        async def copy_home() -> None:  # the VM is stopped: the copy matches the saved RAM
            # A cancel resumes the VM (save_snapshot): the copy stops first (a copy of the data
            # at its next chunk), so none runs on to read a disk the guest writes, into a folder
            # the save removes
            await run_stoppable(lambda stop: clone_sparse(vm.workdir.home_disk, home, stop=stop), undo=None)

        connected = False
        try:
            async with MigrationClient(vm.qmp_socket, get_expected_socket_uid(vm.use_qemu_vm_user)) as client:
                connected = True
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
        except TransientError as e:
            if not connected and vm.process.returncode is None:  # QEMU runs and never paused the VM
                e.context["vm_resumed"] = True
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
        """Head, host and budget checks (ROUTE before any download), then the chain as local files.

        `packages_image` finds the local L2 image for the stored spec (None: none here).
        Does not check the id (callers do). Raises InvalidInputError SCHEDULER_STATE once
        stop() began; InstanceGoneError INSTANCE_NOT_FOUND when there is no head,
        ConfigError INSTANCE_KEY_MISSING when this keyring lacks its key, InstanceGoneError
        INSTANCE_INTEGRITY or RouteError INSTANCE_HOST_MISMATCH (another release) when it
        does not open, and the store's errors; RouteError INSTANCE_HOST_MISMATCH when the
        host facts differ (arch, accelerator, guest CPU model, host CPU identity, the guest's
        counter rate by its rule, the machine type this QEMU has), when this release runs
        another device layout for the VM, or
        when a guest image is neither on this worker nor in the store (or the store's copy
        does not match its digest); RouteError CAPACITY_NEVER_FITS when the VM is larger
        than this worker's whole budget; ConfigError HOST_DEPENDENCY or TransientError
        QEMU_CRASHED when a QEMU probe or the layout build fails
        (VmManager.restore_layout); ConfigError HOST_DEPENDENCY when qemu-vm cannot get
        read access to the migration file; InstanceGoneError INSTANCE_EXPIRED when a body
        object stays gone; TransientError INSTANCE_STORE_UNAVAILABLE when the head keeps
        changing; TransientError LOCAL_DISK for a local file error (nothing was loaded).
        """
        previous_etag: str | None = None
        missing: str | None = None
        with self._in_flight(), named_instance(instance_id):
            for _ in range(_LOAD_ATTEMPTS):
                with _object_named(head_key(instance_id), None):  # an oversized head is INSTANCE_INTEGRITY
                    got = await self.store.get(head_key(instance_id), max_bytes=MAX_HEAD_BYTES)
                    if got is None:
                        raise _not_found(instance_id, store=self.store)
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
                checked = await self._check_host(instance_id, body)
                with _local_files(instance_id, step="loading"):
                    base_image, packages, leases = await self._resolve_images(
                        instance_id, body, packages_image, checked=checked
                    )
                    try:
                        folder = self._new_folder(instance_id)
                    except BaseException:
                        _release(leases)
                        raise
                    try:
                        vmstate, home = await self._download(header, body, data_key, folder)
                    except _VanishedError as e:
                        _release(leases)
                        self._remove_in_background(folder)
                        previous_etag, missing = etag, e.key
                        continue
                    except BaseException:
                        _release(leases)
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
                        asset_leases=leases,
                    )
        raise TransientError(
            f"Instance {instance_id}: the head kept changing while it was loaded.",
            code=ErrorCode.INSTANCE_STORE_UNAVAILABLE,
            instance_id=instance_id,
            hint="Other workers save this instance right now; the same call can succeed later.",
            context=None,
        )

    async def check(self, instance_id: str, *, packages_image: Callable[[VmSpec], Awaitable[Path | None]]) -> None:
        """What load checks before its download, with no download: the head opens with this
        keyring and this release, this host and its budget pass (_check_host), and each guest
        image is on this worker or in the store (a load checks the digest of a store copy).
        `packages_image` as for load. Does not check the id (callers do). Raises what load
        raises before its download."""
        with self._in_flight(), named_instance(instance_id):
            with _object_named(head_key(instance_id), None):
                got = await self.store.get(head_key(instance_id), max_bytes=MAX_HEAD_BYTES)
                if got is None:
                    raise _not_found(instance_id, store=self.store)
                _header, body, _data_key = open_head(got[0], self.keyring, instance_id=instance_id)
            checked = await self._check_host(instance_id, body)
            images = [(RouteKey.BASE_IMAGE, body.guest.base_image, self._local_base_image(body.spec), checked)]
            if body.guest.packages_image is not None:
                local = await packages_image(body.spec)
                with_base = checked.plus(RouteKey.BASE_IMAGE, body.guest.base_image)
                images.append((RouteKey.PACKAGES_IMAGE, body.guest.packages_image, local, with_base))
            for role, digest, local, at in images:
                with _local_files(instance_id, step="loading"):  # a lease's lock file error: LOCAL_DISK
                    here = await self._local_image(role=role, digest=digest, local=local)
                if here is not None and here[1] is not None:
                    here[1].release()  # nothing runs on it
                if here is None and not await self.store.exists(asset_key(digest)):
                    raise _image_missing(instance_id, role=role, digest=digest, checked=at, problem=_IMAGE_NOWHERE)

    async def _check_host(self, instance_id: str, body: HeadBody) -> _Checked:
        """The checked facts of the saved host and the budget the VM needs, before any
        download: ROUTE when this host's facts differ (INSTANCE_HOST_MISMATCH) or the VM can
        never fit its whole budget (CAPACITY_NEVER_FITS)."""
        target = _target(body)
        found = await self._found(target)
        if any(not key.holds(found[key], need) for key, need in target.items()):
            raise host_mismatch(instance_id=instance_id, required=target, found=found, reason=None)
        here = await self._vm_manager.restore_layout(
            body.spec, machine_type=body.host.machine_type, packages=body.guest.packages_image is not None
        )
        if here != body.host.device_layout:
            raise _layout_differs(instance_id, saved=body.host.device_layout, here=here)
        need, have = await self._vm_manager.check_fits(body.spec, vm_id=instance_id, required=target, found=found)
        return _Checked(required={**target, **need}, found={**found, **have})

    async def refused_by_qemu(self, state: InstanceState, *, qemu_error: str) -> SandboxError:
        """The error of a loaded state that cannot run on this worker (migration_client.refusal_cause:
        QEMU refused it or aborted on it, or its guest panicked right after it). QEMU loads a
        state that the same or an older QEMU saved: on an older QEMU, ROUTE to a worker with
        the saver's facts, the budget the VM needs, its guest images and a QEMU of the saved
        version or newer (qemu_version, rule AT_LEAST); on the same or a newer QEMU,
        InternalError (REPORT): QEMU broke that rule, or the device model of this release
        differs. `qemu_error`: QEMU's reason, in the message."""
        host = state.body.host
        local_qemu = await qemu_version_fact()
        if RouteKey.QEMU_VERSION.holds(local_qemu, host.qemu_version):
            return InternalError(
                f"Instance {state.header.instance_id}: QEMU {local_qemu} cannot run a state that QEMU "
                f"{host.qemu_version} saved: {qemu_error}.",
                code=ErrorCode.INTERNAL,
                context=None,
                hint=None,
                instance_id=state.header.instance_id,
            )
        checked = await self._check_host(state.header.instance_id, state.body)  # the load passed them
        guest = state.body.guest
        images = {RouteKey.BASE_IMAGE: guest.base_image}  # this worker has them: the load resolved them
        if guest.packages_image is not None:
            images[RouteKey.PACKAGES_IMAGE] = guest.packages_image
        return host_mismatch(
            instance_id=state.header.instance_id,
            required={**checked.required, **images, RouteKey.QEMU_VERSION: host.qemu_version},
            found={**checked.found, **images, RouteKey.QEMU_VERSION: local_qemu},
            reason=f"the saved state cannot run on this worker's QEMU ({qemu_error})",
        )

    async def _download(self, header: HeadHeader, body: HeadBody, data_key: bytes, folder: Path) -> tuple[Path, Path]:
        """The chain into two new stream files, opened once. At most _OBJECTS_AHEAD
        objects are downloading or on local disk at once, the applied one included;
        objects are applied in chain order, one at a time, each in a worker thread
        that then removes it. The first missing object in chain order raises
        _VanishedError. On an error or a cancel, the downloads stop; load() then
        starts the removal of the folder with every file in it (drain() waits for it)."""
        instance_id = header.instance_id
        vmstate = folder / f"{header.commit_id}.vmstate"
        home = home_path_from_vmstate(vmstate)  # where VmManager.restore_vm looks for it
        new_stream_file(vmstate)
        new_stream_file(home)
        objects = [folder / f"{entry.commit_id}.{_OBJECT_KIND[entry.kind]}" for entry in body.chain]

        async def fetch(n: int) -> None:
            entry = body.chain[n]
            key = body_key(instance_id, entry.commit_id, _OBJECT_KIND[entry.kind])
            with _object_named(key, entry.commit_id):  # a wrong size is INSTANCE_INTEGRITY
                if not await self.store.download(key, dest=objects[n], max_bytes=entry.size, expected_size=entry.size):
                    raise _VanishedError(key)

        def apply(n: int, targets: Mapping[int, int]) -> None:
            entry = body.chain[n]
            try:
                with _object_named(body_key(instance_id, entry.commit_id, _OBJECT_KIND[entry.kind]), entry.commit_id):
                    apply_object(
                        objects[n],
                        data_key=data_key,
                        instance_id=instance_id,
                        commit_id=entry.commit_id,
                        targets=targets,
                        sizes=dict(enumerate(entry.stream_sizes)),
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
        if await probe_qemu_vm_user() and not await grant_qemu_vm_acl([vmstate], perms="rw"):
            raise ConfigError(
                "Could not give qemu-vm read access to the migration file (needs setfacl, package acl)",
                code=ErrorCode.HOST_DEPENDENCY,
                context={"instance_id": instance_id},
                hint=None,
                instance_id=None,
            )
        return vmstate, home

    # ------------------------------------------------------------------ assets

    async def _ensure_vm_assets(self, vm: QemuVM, *, guest: GuestFacts) -> None:
        """Store the VM's guest images by digest before the commit (at commit time,
        the store holds every image the head names)."""
        await self._ensure_asset(vm.base_image, guest.base_image)
        if vm.packages_image is not None and guest.packages_image is not None:
            await self._ensure_asset(vm.packages_image, guest.packages_image)

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
            if await file_hash(path, "sha256") != digest:
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
        self,
        instance_id: str,
        body: HeadBody,
        packages_image: Callable[[VmSpec], Awaitable[Path | None]],
        *,
        checked: _Checked,
    ) -> tuple[Path, Path | None, tuple[EntryLease, ...]]:
        """The base and the packages image (None without packages), and the leases on the
        ones of the asset cache (released here on an error)."""
        leases: list[EntryLease] = []
        try:
            base, lease = await self._resolve_image(
                instance_id,
                role=RouteKey.BASE_IMAGE,
                digest=body.guest.base_image,
                local=self._local_base_image(body.spec),
                checked=checked,
            )
            leases.extend([lease] if lease is not None else [])
            if body.guest.packages_image is None:
                return base, None, tuple(leases)
            packages, lease = await self._resolve_image(
                instance_id,
                role=RouteKey.PACKAGES_IMAGE,
                digest=body.guest.packages_image,
                local=await packages_image(body.spec),
                checked=checked.plus(RouteKey.BASE_IMAGE, body.guest.base_image),
            )
            leases.extend([lease] if lease is not None else [])
        except BaseException:
            _release(leases)
            raise
        return base, packages, tuple(leases)

    async def _resolve_image(
        self, instance_id: str, *, role: RouteKey, digest: str, local: Path | None, checked: _Checked
    ) -> tuple[Path, EntryLease | None]:
        """A local image with `digest`: the worker's own, one of the asset cache, or a new
        download from the store (checked against the digest) into the asset cache; with the
        lease on an image of the asset cache (None: the worker's own). ROUTE when none, with
        the facts checked before (`checked`, this worker passed them) and the image."""
        if (found := await self._local_image(role=role, digest=digest, local=local)) is not None:
            return found
        cached = self.assets.cache_dir / f"{digest}.img"
        async with CacheEntry(cached).saving() as missing:  # eviction needs this lock too
            if not missing:  # another load put it there meanwhile
                return cached, await _asset_lease(cached)
            part = self.assets.cache_dir / f"{digest}.{ids.new_id()}.part"
            try:
                if not await self.store.download(
                    asset_key(digest), dest=part, max_bytes=_MAX_ASSET_BYTES, expected_size=None
                ):
                    problem = _IMAGE_NOWHERE
                elif await file_hash(part, "sha256") != digest:
                    problem = "the store's copy of the image does not match its digest"
                else:
                    part.replace(cached)
                    logger.info("Instance asset downloaded", extra={"digest": digest, "role": role})
                    return cached, await _asset_lease(cached)
            finally:
                self._remove_in_background(part)  # up to GBs; gone already after the rename
        raise _image_missing(instance_id, role=role, digest=digest, checked=checked, problem=problem)

    async def _local_image(
        self, *, role: RouteKey, digest: str, local: Path | None
    ) -> tuple[Path, EntryLease | None] | None:
        """The worker's own image when it has `digest` (no lease), else the asset cache's,
        leased (eviction skips it); None: neither is here."""
        if local is not None:
            try:
                same = await asyncio.to_thread(image_digest, local) == digest
            except OSError as e:  # unreadable here (no access, an I/O error): no usable local copy
                logger.warning(
                    "Local guest image unreadable", extra={"image": str(local), "role": role, "error": str(e)}
                )
                same = False
            if same:
                return local, None
        cached = self.assets.cache_dir / f"{digest}.img"  # only ever renamed into place after its digest was checked
        lease = await CacheEntry(cached).lease()  # None: not here (never fetched, or evicted)
        return None if lease is None else (cached, lease)

    def _local_base_image(self, spec: VmSpec) -> Path | None:
        try:
            return self._vm_manager.images.base_image(spec.language)
        except ConfigError:  # ASSET_NOT_FOUND: in no folder of this worker
            return None

    # ------------------------------------------------------------------ helpers

    async def release(self, state: InstanceState, *, wait: bool) -> None:
        """Release the state's leases on the asset cache and remove its instance files (what
        will not go is logged). wait=False: the call never suspends; the removal runs in a worker
        thread after it returns (drain() waits for it)."""
        _release(state.asset_leases)
        if wait:
            await asyncio.to_thread(_remove_quietly, state.folder)
        else:  # no await before this line: drain() must see the job (a Scheduler exit can come next)
            self._remove_in_background(state.folder)

    def _new_folder(self, instance_id: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{instance_id}-", dir=self.work_dir))

    def asset_digests(self) -> frozenset[str]:
        """Digests of the guest images this worker downloaded from the store: each one is
        renamed to <digest>.img in the asset folder only after its digest check."""
        try:
            names = [path.name for path in self.assets.cache_dir.iterdir()]
        except OSError:  # the folder is gone (a cleanup between runs): none here
            return frozenset()
        return frozenset(
            name.removesuffix(".img")
            for name in names
            if name.endswith(".img") and re.fullmatch(r"[0-9a-f]{64}", name.removesuffix(".img"))
        )

    async def _found(self, target: dict[RouteKey, str]) -> dict[RouteKey, str | None]:
        """This worker's value for each key of a host target (a CONTAINS key: the target's
        value when this worker has it, else None)."""
        force_emulation = self._vm_manager.settings.force_emulation
        local = {
            **await machine_facts(force_emulation=force_emulation),
            RouteKey.COUNTER_HZ: await counter_fact(force_emulation=force_emulation),
        }
        machine_types = await require_qemu_machine_types()
        return {
            key: (value if value in machine_types else None) if key is RouteKey.MACHINE_TYPE else local[key]
            for key, value in target.items()
        }

    def _guest_facts(self, vm: QemuVM) -> GuestFacts:
        """What the saved guest got at its cold boot: the digests of its images, taken when
        QEMU opened them (after reuse() these may differ from this worker's own images; a file
        evicted or rebuilt at the same path since does not change them), of its kernel and
        initramfs, and the rate of its clock's counter (a reused VM keeps those of its first
        boot).

        Raises:
            ConfigError: IMAGE_DAMAGED: a digest is unknown (the file was missing at the VM's
                start); nothing is saved
        """
        base, packages, kernel, initramfs = (
            vm.base_image_digest,
            vm.packages_image_digest,
            vm.kernel_digest,
            vm.initramfs_digest,
        )
        if base is None or kernel is None or initramfs is None or (vm.packages_image is not None and packages is None):
            raise ConfigError(
                f"VM {vm.vm_id}: the digest of a guest file it runs on is unknown (the file was missing "
                "when the VM started). It was not saved.",
                code=ErrorCode.IMAGE_DAMAGED,
                context={"base_image": str(vm.base_image), "packages_image": str(vm.packages_image)},
                hint="Download or rebuild the guest images.",
                instance_id=None,
            )
        return GuestFacts(
            base_image=base, packages_image=packages, kernel=kernel, initramfs=initramfs, counter_hz=vm.counter_hz
        )

    async def _host_facts(self, vm: QemuVM) -> HostFacts:
        """The saving host's facts (recomputed at each save: HostFacts).

        Raises:
            ConfigError: HOST_DEPENDENCY: the QEMU machine type of the VM is unknown
        """
        if vm.machine_type is None:  # a load starts QEMU with it: never store a VM without it
            raise ConfigError(
                f"VM {vm.vm_id}: the QEMU machine type it runs is unknown (this QEMU does not list the "
                "type it started with). It was not saved.",
                code=ErrorCode.HOST_DEPENDENCY,
                context=None,
                hint="Restart the worker after a QEMU change, then save a new session.",
                instance_id=None,
            )
        facts = await machine_facts(force_emulation=self._vm_manager.settings.force_emulation)
        qemu_version = await qemu_version_fact()  # a refusal compares it with the loader's
        return HostFacts.model_validate(
            {
                **{key.value: value for key, value in facts.items()},
                "qemu_version": qemu_version,
                "release": __version__,  # a diagnostic: the release of this save
                "machine_type": vm.machine_type,
                "device_layout": vm.device_layout,
            }
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
        if os.path.isdir(path):  # noqa: PTH112 - never raises; Path.is_dir of Python 3.12 raises on EACCES or EIO
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


def _target(body: HeadBody) -> dict[RouteKey, str]:
    """The checked facts of a saved instance, as route values: its host's (a fact that does
    not apply, such as cpu_id on x86, is absent), its machine type, and the counter rate its
    guest booted with (absent when the boot host had no exact rate)."""
    saved = body.host.model_dump()
    target = {key: saved[key] for key in _CHECKED_FACTS if saved[key] is not None}
    target[RouteKey.MACHINE_TYPE] = body.host.machine_type
    if body.guest.counter_hz is not None:
        target[RouteKey.COUNTER_HZ] = str(body.guest.counter_hz)
    return target


@dataclass(frozen=True, slots=True, kw_only=True)
class _Checked:
    """The route facts a load checked so far, as a RouteError gives them: what the instance
    needs (`required`) and this worker's values (`found`, which pass)."""

    required: dict[RouteKey, str]
    found: dict[RouteKey, str | None]

    def plus(self, key: RouteKey, value: str) -> _Checked:
        """These facts and `key`, which this worker has (`value`)."""
        return _Checked(required={**self.required, key: value}, found={**self.found, key: value})


def _info(
    header: HeadSummary,
    *,
    access: InstanceAccess,
    writer_release: str | None,
    body: HeadBody | None,
) -> InstanceInfo:
    return InstanceInfo(
        instance_id=header.instance_id,
        generation=header.generation,
        recent_commits=header.recent_commits,
        key_id=header.key_id,
        access=access,
        writer_release=writer_release,
        spec=None if body is None else body.spec,
        host=None if body is None else body.host,
        guest=None if body is None else body.guest,
        egress_domains=None if body is None else body.egress_domains,
    )


_IMAGE_NOWHERE = "the image is not on this worker and not in the store"


def _release(leases: Collection[EntryLease]) -> None:
    for lease in leases:
        lease.release()


async def _asset_lease(path: Path) -> EntryLease:
    """A lease on an image of the asset cache, taken under its save lock (eviction needs that
    lock too, so the image is there); a file error is LOCAL_DISK (_local_files)."""
    if (lease := await CacheEntry(path).lease()) is None:
        raise FileNotFoundError(errno.ENOENT, "guest image removed under its save lock", str(path))
    return lease


def _layout_differs(instance_id: str, *, saved: str, here: str) -> RouteError:
    """ROUTE to another release: this release's QEMU command for the saved VM has another
    device layout; the message names the first line that differs."""
    pairs = itertools.zip_longest(saved.splitlines(), here.splitlines(), fillvalue="nothing")
    old, new = next((pair for pair in pairs if pair[0] != pair[1]), ("?", "?"))
    reason = f"this release runs another device layout for it (saved {old!r}, here {new!r})"
    return host_mismatch(instance_id=instance_id, required={}, found={}, reason=reason)


def _image_missing(instance_id: str, *, role: RouteKey, digest: str, checked: _Checked, problem: str) -> RouteError:
    """ROUTE for a guest image this worker cannot get: the instance is intact, and a worker
    that has the image (and the facts checked before) can load it."""
    return host_mismatch(
        instance_id=instance_id,
        required={**checked.required, role: digest},
        found={**checked.found, role: None},
        reason=problem,
    )


@contextlib.contextmanager
def _object_named(key: str, commit_id: str | None) -> Generator[None]:
    """Name the store object in each INSTANCE_INTEGRITY error raised in the block (its key,
    and the commit that wrote it for a body object): an operator tells a damaged copy of
    one object from another."""
    try:
        yield
    except InstanceGoneError as e:
        if e.code is ErrorCode.INSTANCE_INTEGRITY:
            e.context.update({"key": key} if commit_id is None else {"key": key, "commit_id": commit_id})
        raise


def _not_found(instance_id: str, *, store: InstanceStore) -> InstanceGoneError:
    """No head: name where it looked (a wrong bucket, prefix or folder gives this error too)."""
    return InstanceGoneError(
        f"No instance {instance_id} in the store {store.location} (never saved, deleted or expired).",
        code=ErrorCode.INSTANCE_NOT_FOUND,
        instance_id=instance_id,
        hint="Check that this is the store that saved it, else start again with Scheduler.new().",
        context={"store": store.location, "key": head_key(instance_id)},
    )

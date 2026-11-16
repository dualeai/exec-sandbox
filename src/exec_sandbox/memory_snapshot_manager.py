"""L1 memory snapshot cache manager.

Captures full VM state (CPU + RAM + device state) after REPL warm-up and
package import. On restore, the VM resumes at the exact instruction — ~50-60 ms
regardless of packages.

Cache structure (the names next to the vmstate: memory_snapshot_files):
    {cache_dir}/{key}.vmstate       # QEMU migration stream: device state only (x-ignore-shared, ~100 KB-1 MB); the RAM is {key}.ram
    {cache_dir}/{key}.vmstate.meta  # JSON metadata sidecar
    {cache_dir}/{key}.vmstate.lock  # flock for cross-process deduplication of saves
    {cache_dir}/{key}.vmstate.use.lock  # record lock (fcntl) each restored VM holds shared (no eviction)
    {cache_dir}/{key}.ram           # guest RAM of a template (memory-backend-file; see open_live_ram_dir)
    {cache_dir}/{key}.home          # home disk at the save point (cloned per restore)

Cache key includes ALL parameters that affect VM state:
    sha256 fingerprint(language, *sorted_packages, exec_sandbox_version, img_hash,
          qemu_version, the machine facts (arch, accelerator, cpu_model, cpu_id: as a save
          records them), counter_hz, memory_mb, cpu_cores, disk_mb,
          network_device, device_layout (with the migration capabilities), kernel_hash, initramfs_hash
          [, l2_image_hash when there are packages])

Format: 'l1-{language}-v{major.minor}-{sha256}'
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import functools
import hashlib
import os
import time
import weakref
from typing import TYPE_CHECKING, ClassVar, Final

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox._version import __version__
from exec_sandbox.aio_utils import run_stoppable, run_to_end
from exec_sandbox.base_cache_manager import BaseCacheManager
from exec_sandbox.exceptions import ConfigError, ErrorCode, InternalError, SandboxError
from exec_sandbox.files.disk import (
    clone_sparse,
    create_file,
    create_sparse,
    data_ranges,
    file_stamp,
    flush_to_disk,
    pwrite_all,
)
from exec_sandbox.files.locks import CacheEntry, EntryLease
from exec_sandbox.guest_agent_protocol import WarmReplAckMessage, WarmReplRequest
from exec_sandbox.hash_utils import fingerprint
from exec_sandbox.images import key_part
from exec_sandbox.memory_snapshot_files import (
    SnapshotMetadata,
    home_path_from_vmstate,
    meta_path_from_vmstate,
    open_live_ram_dir,
    ram_path_from_vmstate,
)
from exec_sandbox.migration_client import MigrationClient, migration_timeout, refusal_cause
from exec_sandbox.permission_utils import get_expected_socket_uid, grant_qemu_vm_acl, probe_qemu_as_qemu_vm
from exec_sandbox.system_probes import require_qemu_version
from exec_sandbox.worker_facts import machine_facts

if TYPE_CHECKING:
    import threading
    from collections.abc import Iterator
    from pathlib import Path

    from exec_sandbox.admission import ResourceReservation
    from exec_sandbox.models import ExposedPort
    from exec_sandbox.private_run_dir import PrivateRunDir
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.settings import Settings
    from exec_sandbox.vm_manager import VmManager
    from exec_sandbox.vm_spec import VmSpec

logger = get_logger(__name__)


_BLOCK = 64 * 1024
_ZERO_BLOCK = bytes(_BLOCK)

_PUBLISH_FLUSH_BYTES: Final = 32 * 1024 * 1024
"""_publish_ram writes the RAM through to the disk after each this many bytes."""


def _nonzero_blocks(fd: int, size: int) -> Iterator[tuple[int, bytes]]:
    """(offset, block) of each 64 KiB block below `size` of the file open at `fd` that
    holds a non-zero byte, in order. Holes are skipped (SEEK_DATA), not read; without
    SEEK_DATA every block is read. Sync."""
    try:
        ranges = list(data_ranges(fd, size, grain=_BLOCK))
    except OSError:
        ranges = [(0, size)]
    for start, end in ranges:
        for offset in range(start, end, _BLOCK):
            block = os.pread(fd, _BLOCK, offset)
            if block != _ZERO_BLOCK[: len(block)]:
                yield offset, block


def _vmstate_digest(path: Path) -> str:
    """SHA-256 hex digest of the file's size, then of each non-zero 64 KiB block
    with its offset (the size fixes each block's length).

    A mapped-ram vmstate is sparse (a 140 MB file can hold 3 MB of data): holes
    are skipped (SEEK_DATA), not read. A hole and a zero block hash alike, so a
    copy that changes the hole layout keeps the digest. Without SEEK_DATA every
    block is read. Sync: callers use asyncio.to_thread.
    """
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        fd = f.fileno()
        size = os.fstat(fd).st_size
        sha256.update(size.to_bytes(8, "big"))
        for offset, block in _nonzero_blocks(fd, size):
            sha256.update(offset.to_bytes(8, "big") + block)
    return sha256.hexdigest()


def _flush_files(paths: list[Path]) -> None:
    """flush_to_disk each file (this process made each one). Sync: callers use asyncio.to_thread."""
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            flush_to_disk(fd)
        finally:
            os.close(fd)


def _new_entry_file(path: Path, *, size: int) -> None:
    """A new private (0600) file of this process for QEMU to write (a stale one goes:
    unlinked, never truncated, as a restored VM may still map it). Sync."""
    path.unlink(missing_ok=True)
    create_sparse(path, size)


class _LiveRam:
    """The RAM file of a template save VM (_new_live_ram): `path`, in `folder` in memory,
    with `fd` open on it for _publish_ram; else {key}.ram itself (no folder, no fd).
    systemd-logind (RemoveIPC=yes, its default) removes the /dev/shm files of a user that is
    not a system user when its last login session ends, also during a save; an open file
    keeps its pages. `close` frees the memory (the fd, then the folder), also when an owner
    is lost (garbage collection): up to a whole guest RAM stays in tmpfs while the fd is open."""

    def __init__(self, *, folder: PrivateRunDir | None, path: Path, fd: int | None) -> None:
        self.path = path
        self.fd = fd
        self._finalizer = weakref.finalize(self, _close_live_ram, folder, live_fd=fd)

    def close(self) -> None:
        """Idempotent. Blocking (the folder's removal): async callers use a worker thread."""
        self._finalizer()


def _new_live_ram(ram_path: Path, *, size: int, owner: list[_LiveRam]) -> _LiveRam:
    """A new RAM file of `size` bytes for a template save VM (_new_entry_file), in the folder
    from open_live_ram_dir, else at `ram_path`. Put in `owner` as soon as it is made, so a
    caller whose await was cancelled (run_to_end waits for this job) still closes it. Sync."""
    live_dir = open_live_ram_dir(size=size)
    live_ram = ram_path if live_dir is None else live_dir.path / ram_path.name
    try:
        _new_entry_file(live_ram, size=size)
        live_fd = None if live_dir is None else os.open(live_ram, os.O_RDONLY | os.O_NOFOLLOW)
    except BaseException:
        if live_dir is not None:
            live_dir.close()
        raise
    owner.append(_LiveRam(folder=live_dir, path=live_ram, fd=live_fd))
    return owner[-1]


def _close_live_ram(live_dir: PrivateRunDir | None, *, live_fd: int | None) -> None:
    """_LiveRam's cleanup: the fd, then the folder in memory, the RAM with it, the folder
    even when the fd's close fails. Blocking."""
    try:
        if live_fd is not None:
            os.close(live_fd)
    finally:
        if live_dir is not None:
            live_dir.close()


def _publish_ram(stop: threading.Event, *, src_fd: int, dst: Path) -> None:
    """Copy the RAM file open at `src_fd` of a template save VM to a new file `dst`: only
    its non-zero 64 KiB blocks; the others stay holes, which read as zeros (a Python template
    has 68 of 192 MiB non-zero). Writes through to the disk after each 32 MiB, so the
    dirty page cache of this process stays small whatever the RAM size. A stale `dst`
    goes as in _new_entry_file. Ends at the next block once `stop` is set (a cancel: the
    caller removes the partial `dst`). Blocking: a job of aio_utils.run_stoppable, which
    waits for its end before the caller closes `src_fd`."""
    size = os.fstat(src_fd).st_size
    _new_entry_file(dst, size=size)
    fd = os.open(dst, os.O_WRONLY)
    try:
        unflushed = 0
        for offset, block in _nonzero_blocks(src_fd, size):
            if stop.is_set():
                return
            pwrite_all(fd, block, offset)
            unflushed += len(block)
            if unflushed >= _PUBLISH_FLUSH_BYTES:
                flush_to_disk(fd)
                unflushed = 0
    finally:
        os.close(fd)


async def _grant_entry_files(paths: list[Path], *, perms: str, key: str) -> None:
    """qemu-vm gets `perms` on L1 entry files by ACL: "rw" while a save VM writes them,
    "r" once published (no QEMU may change a shared template).

    Raises:
        ConfigError: HOST_DEPENDENCY: setfacl failed, or a folder above them could not be
            made traversable (a+x)
    """
    if not await grant_qemu_vm_acl(paths, perms=perms):
        raise ConfigError(
            f"Could not give qemu-vm {perms} access to the L1 entry files (needs setfacl, package acl, "
            "and a+x on the folders above the cache folder)",
            code=ErrorCode.HOST_DEPENDENCY,
            context={"key": key, "paths": [str(path) for path in paths]},
            hint=None,
            instance_id=None,
        )


class MemorySnapshotManager(BaseCacheManager):
    """L1 memory snapshot cache. Manages vmstate files + metadata sidecars.

    Uses file-based flock for cross-process deduplication of saves.
    Composable API — compute_cache_key() once per request, then check_cache(),
    lease() (its lease goes to restore()) or schedule_background_save() with that key.
    """

    _cache_ext: ClassVar[str] = ".vmstate"
    _sidecar_exts: ClassVar[tuple[str, ...]] = (".vmstate.meta", ".ram", ".home")

    def __init__(self, settings: Settings, vm_manager: VmManager):
        """Make the cache folder (memory_snapshot_cache_dir).

        Raises:
            ConfigError: SETTING_INVALID: the cache folder cannot be made, or is not private
            TransientError: LOCAL_DISK: a host error on the cache folder (disk full, I/O error)
        """
        super().__init__(settings.memory_snapshot_cache_dir, setting="memory_snapshot_cache_dir")
        self.settings = settings
        self.vm_manager = vm_manager
        # In-process deduplication of background saves: one task per key, so a second
        # call for a key in flight returns at once, without a task (the save lock
        # stops a save of another process).
        self._in_flight_saves: dict[str, asyncio.Task[None]] = {}
        # Keys whose restored guest could not reseed its kernel random source: the key fixes the
        # guest (image, kernel, initramfs), so each later save of it would fail at its first
        # restore and be removed
        self._no_save_keys: set[str] = set()

    def _cache_paths(self, key: str) -> tuple[Path, Path]:
        """Return (vmstate_path, meta_path) for a given cache key."""
        vmstate_path = self._cache_path(key)
        return (vmstate_path, meta_path_from_vmstate(vmstate_path))

    async def compute_cache_key(self, spec: VmSpec, *, l2_path: Path | None) -> str:
        """Compute L1 cache key. Includes ALL parameters that affect VM state.

        memory_mb IS in L1 key (unlike L2) because the migration stream
        encodes physical address layout. qemu_version and accel_type are included
        because different QEMU versions produce incompatible migration streams and
        different accelerators (KVM/HVF/TCG) have different CPU state; the -cpu value
        too (a restore onto other CPU features fails).
        The network device (network access or exposed ports) is included because
        it changes device topology (-nic none vs -device virtio-net).
        `l2_path` is the L2 layer the VM boots with (None without packages).

        Raises:
            ConfigError: ASSET_NOT_FOUND: the base image of `spec` is missing; HOST_DEPENDENCY:
                this host cannot run the QEMU binary, or its version or accelerators cannot be
                read (the QEMU probes), or ARM64 TCG with a QEMU below 9.0.4 (restore_layout)
            TransientError: LOCAL_DISK: an image of the key cannot be read (images.key_part);
                QEMU_CRASHED: a QEMU probe hit a process, memory or file-handle limit, the
                binary was being written, a signal ended it, or it gave no answer in time; or
                the QEMU binary came or went during the build (VmManager.restore_layout)
        """
        version_parts = __version__.split(".")
        version = f"{version_parts[0]}.{version_parts[1]}"

        # Collect all state-affecting parameters
        # Base image, then the kernel + initramfs QEMU actually boots
        guest = self.vm_manager.images
        images = [guest.base_image(spec.language), guest.kernel, guest.initramfs]
        # Pin the L2 layer by content: an L1 template resumes a guest whose page
        # cache refers to the exact L2 bytes it was saved with; a rebuilt L2
        # (different bytes for the same package specs) must not reuse it.
        if l2_path is not None:
            images.append(l2_path)
        # One worker-thread job for every digest (a stat each once known, a full read the first time)
        img_hash, *boot_and_l2_hashes = await asyncio.to_thread(lambda: [key_part(path) for path in images])
        qemu_version = await require_qemu_version()
        # The facts a save records (worker_facts.machine_facts), and the counter rate the
        # restored guest keeps: a restore from an entry passes no boot, so it gets the rate of
        # this VmManager's boots, as the template's boot did
        facts = await machine_facts(force_emulation=self.settings.force_emulation)
        counter_hz = self.vm_manager.counter_hz

        parts = [
            str(spec.language.value),
            *spec.packages,
            str(__version__),
            img_hash,
            ".".join(str(v) for v in qemu_version),
            *(f"{key}={value}" for key, value in facts.items()),
            f"counter_hz={counter_hz}",
            str(spec.memory_mb),
            str(spec.cpu_cores),
            f"disk{spec.disk_mb}",
            "net" if spec.has_network_device else "nonet",
            # The devices of the restore's command and the migration capabilities
            # (device_layout): a release that changes either never restores an entry saved
            # with the old ones
            await self.vm_manager.restore_layout(spec, machine_type=None, packages=l2_path is not None),
            *boot_and_l2_hashes,
        ]

        return f"l1-{spec.language.value}-v{version}-{fingerprint(*parts)}"

    async def check_cache(self, key: str) -> Path | None:
        """Check L1 cache for a key from compute_cache_key(). Returns vmstate path or None.

        Validates: file exists, non-zero size, metadata sidecar parses, the vmstate digest,
        the RAM template and the home disk. Updates the access time (eviction removes the
        entry with the oldest one first). A broken entry is removed as an eviction removes
        one (BaseCacheManager._remove_broken); the miss returns after it (a cancel leaves
        the job running).
        """
        checked = await self._check_entry(key, lease=None)
        return checked[0] if checked is not None else None

    async def _check_entry(self, key: str, *, lease: EntryLease | None) -> tuple[Path, str] | None:
        """check_cache itself: the vmstate path and the version of it checked (file_stamp),
        or None. `lease`, the caller's own lease of the entry, goes before a broken entry is
        removed (another lease keeps it: its VM runs on it)."""
        vmstate_path, meta_path = self._cache_paths(key)
        try:
            stat = vmstate_path.stat()
        except OSError:
            return None
        if not meta_path.exists():
            return None
        stamp = file_stamp(stat)
        problem = await self._entry_problem(vmstate_path, meta_path, size=stat.st_size)
        if problem is not None:
            logger.warning(f"{problem}, removing", extra={"key": key})
            await self._remove_broken(vmstate_path, stamp=stamp, lease=lease)
            return None
        # Access time for eviction (best effort). The mtime stays: the stamp of this version
        # does not change. utime never makes a file (Path.touch would, after an eviction).
        with contextlib.suppress(OSError):
            os.utime(vmstate_path, ns=(time.time_ns(), stat.st_mtime_ns))
        logger.debug("L1 cache hit", extra={"key": key, "size_mb": stat.st_size // (1024 * 1024)})
        return vmstate_path, stamp

    async def _entry_problem(self, vmstate_path: Path, meta_path: Path, *, size: int) -> str | None:
        """What makes the entry of `vmstate_path` (of `size` bytes) broken, or None."""
        if size == 0:
            return "L1 cache entry has zero size"
        try:
            meta = SnapshotMetadata.from_json(meta_path.read_text())
        except OSError:
            meta = None
        if meta is None:
            return "L1 cache metadata invalid"
        try:
            if await asyncio.to_thread(_vmstate_digest, vmstate_path) != meta.vmstate_sha256:
                return "L1 cache vmstate integrity check failed"
        except OSError:
            return "L1 cache vmstate unreadable during integrity check"
        if not ram_path_from_vmstate(vmstate_path).exists():
            return "L1 cache RAM template missing"
        if not home_path_from_vmstate(vmstate_path).exists():
            return "L1 cache home disk missing"
        return None

    async def _do_save(
        self,
        vm: QemuVM,
        key: str,
        vmstate_path: Path,
        meta_path: Path,
        spec: VmSpec,
        *,
        live_ram: Path,
        live_ram_fd: int | None,
    ) -> Path | None:
        """Raw save of the template VM `vm`, whose guest RAM is the file `live_ram`
        (memory-backend-file, share=on), open at `live_ram_fd` when it is in memory
        (_new_live_ram) — caller must hold the file lock.

        The host makes and owns every file of the entry; QEMU only writes into them
        (as qemu-vm by ACL), so any QEMU identity of this host can restore the entry.

        Sequence:
        1. Make the tmp file, then save the device state via MigrationClient into it
           (x-ignore-shared: no RAM; terminates QEMU).
        2. The RAM template: `live_ram` when it is {key}.ram, else (in memory) written
           to {key}.ram by _publish_ram.
        3. Clone the home disk to {key}.home (QEMU has quit, so the disk matches
           the saved RAM).
        4. Digest the vmstate, then flush the RAM template and the home copy to
           the disk (the digest covers the vmstate only).
        5. qemu-vm keeps read access only (when it wrote them).
        6. Write JSON metadata sidecar.
        7. Atomic rename tmp → final (commit point).
        On ENOSPC: clean up, evict the oldest entry, return None.
        On any error (including CancelledError): clean up partial files (one
        worker-thread job, waited for even on a cancel).
        """
        tmp_path = vmstate_path.parent / f"{vmstate_path.name}.tmp"

        def discard() -> None:
            """A failed save's files (up to the RAM size) in one worker-thread job. Its
            caller waits for it even when cancelled: the files go before the save lock."""
            tmp_path.unlink(missing_ok=True)
            self._remove_entry(vmstate_path)

        try:
            qemu_version = await require_qemu_version()
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)

            # QEMU opens the file the host made (O_CREAT without O_EXCL, then
            # ftruncate): the host owns it, qemu-vm writes it by ACL.
            await asyncio.to_thread(_new_entry_file, tmp_path, size=0)
            if vm.use_qemu_vm_user:
                await _grant_entry_files([tmp_path], perms="rw", key=key)

            # VM Templating: with x-ignore-shared, migration saves only device
            # state (~340KB). RAM stays in the memory-backend-file (share=on).
            async with MigrationClient(vm.qmp_socket, expected_uid) as client:
                await client.save_snapshot(
                    tmp_path,
                    qemu_version=qemu_version,
                    timeout=migration_timeout(spec.memory_mb),
                    use_template=True,
                    keep_running=False,
                    while_paused=None,  # QEMU quits: the home disk is cloned after
                    memory_limited=vm.cgroup is not None,
                )

            # The vmstate has no RAM (x-ignore-shared): without its template it is no entry. In
            # memory, its pages are there through the fd even when the file is gone (logind).
            ram_path = ram_path_from_vmstate(vmstate_path)
            if live_ram_fd is not None:
                # A cancel stops the copy at its next block and waits for its end: the caller
                # then closes the fd, whose number the next open takes (a job still running
                # would read that other file)
                await run_stoppable(functools.partial(_publish_ram, src_fd=live_ram_fd, dst=ram_path), undo=None)
            elif not live_ram.exists():
                raise FileNotFoundError(errno.ENOENT, "RAM template removed during the save", str(live_ram))

            # Home disk at the save point (the VM is stopped): a restored VM's
            # page cache refers to these blocks.
            home_path = home_path_from_vmstate(vmstate_path)
            home_path.unlink(missing_ok=True)
            await run_stoppable(lambda stop: clone_sparse(vm.workdir.home_disk, home_path, stop=stop), undo=None)

            # Digest of the vmstate (SHA-256 of its size and non-zero blocks); check_cache
            # verifies it before each restore.
            vmstate_sha256 = await asyncio.to_thread(_vmstate_digest, tmp_path)
            # The digest covers the vmstate only; restores trust the RAM template and
            # the home disk as they are. Write both through to the disk before the
            # entry is published, or a power loss can leave them torn under it.
            await asyncio.to_thread(_flush_files, [ram_path, home_path])
            if vm.use_qemu_vm_user:
                await _grant_entry_files([tmp_path, ram_path], perms="r", key=key)

            # Write typed metadata sidecar before atomic rename
            meta = SnapshotMetadata(vmstate_sha256=vmstate_sha256)
            meta_path.unlink(missing_ok=True)  # left by a failed save
            with create_file(meta_path, mode=0o600, dir_fd=None) as f:  # QEMU never reads it
                f.write(meta.to_json().encode())

            # Atomic commit: rename is atomic on POSIX
            tmp_path.rename(vmstate_path)

            size_mb = vmstate_path.stat().st_size / (1024 * 1024)
            logger.info(
                "L1 snapshot saved",
                extra={"key": key, "size_mb": f"{size_mb:.1f}", "language": spec.language.value},
            )
            return vmstate_path

        except OSError as e:
            await run_to_end(discard)
            if e.errno == errno.ENOSPC:
                # Disk full — evict oldest; the next request saves again
                await self.evict_oldest()
                logger.info("L1 evicted oldest on ENOSPC, will retry on next request", extra={"key": key})
                return None
            logger.warning(
                "L1 snapshot save failed (OS error)",
                extra={"key": key, "error": str(e)},
                exc_info=True,
            )
            return None

        except BaseException as e:
            # Catch BaseException (not just Exception) to handle CancelledError too.
            # Prevents orphaned partial files on SIGTERM / task cancellation.
            if not isinstance(e, Exception):
                logger.info("L1 snapshot save cancelled, cleaning up", extra={"key": key})
            else:
                logger.warning(
                    "L1 snapshot save failed",
                    extra={"key": key, "error": str(e)},
                    exc_info=True,
                )
            await run_to_end(discard)
            if isinstance(e, Exception):
                return None
            raise

    async def can_save(self) -> bool:
        """QEMU on this host can save L1 templates (at or above the migration floor).

        Checked before booting a sacrificial VM: below the floor every save fails.
        """
        qemu_version = await require_qemu_version()
        return qemu_version >= constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION

    async def lease(self, key: str) -> EntryLease | None:
        """A lease on the L1 entry `key` (compute_cache_key), or None: no entry, or its use
        lock cannot be opened (logged; the caller cold boots). Take it before admission,
        which may evict cache entries for disk space: never a leased one. restore() takes
        it over."""
        try:
            return await CacheEntry(self._cache_path(key)).lease()
        except OSError as e:
            logger.warning("L1 entry not leased", extra={"key": key, "error": str(e)})
            return None

    async def restore(
        self,
        spec: VmSpec,
        *,
        entry: EntryLease,
        tenant_id: str,
        task_id: str,
        snapshot_drive: EntryLease | None,
        expose_ports: list[ExposedPort] | None,
        reservation: ResourceReservation,
    ) -> QemuVM | None:
        """Restore a VM from the L1 entry `entry` leases (lease() of the compute_cache_key of
        `spec` and of the L2 layer `snapshot_drive` leases). Releases `entry` before it
        returns: the VM, if any, holds its own share.

        None on a miss (a broken entry) or on any failed restore: the caller cold boots
        with the same reservation. An entry QEMU refused (incompatible or corrupt:
        MIGRATION_REJECTED or MIGRATION_UNSUPPORTED in the cause chain), or whose guest
        never answered after its state loaded or could not reseed its kernel random source
        (GUEST_SILENT_AFTER_RESTORE), is removed with all its files, as an eviction removes
        one (BaseCacheManager._remove_broken: not while another lease holds it); otherwise
        every later restore of it fails again. A host problem (no access to a file or a
        socket) keeps the entry. A guest that could not reseed also stops the L1 saves of
        the key in this process (schedule_background_save).

        Raises:
            InternalError: CLEANUP_UNCONFIRMED: the failed restore's QEMU may still run
                and keeps the reservation, so no cold boot may use it
        """
        checked: tuple[Path, str] | None = None
        key = entry.path.name.removesuffix(self._cache_ext)
        try:
            checked = await self._check_entry(key, lease=entry)
            if checked is None:
                return None
            return await self.vm_manager.restore_vm(
                spec,
                tenant_id=tenant_id,
                task_id=task_id,
                vmstate_path=checked[0],
                base_image=self.vm_manager.images.base_image(spec.language),
                snapshot_drive=snapshot_drive.path if snapshot_drive is not None else None,
                cache_leases=(entry,) if snapshot_drive is None else (entry, snapshot_drive),
                expose_ports=expose_ports,
                reservation=reservation,
                machine_type=None,  # the L1 key holds the QEMU version
                egress_domains=None,  # the spec's policy
                boot=None,  # this worker's kernel and initramfs (the L1 key holds their digests)
                use_template=True,
            )
        except InternalError as e:
            if e.code is ErrorCode.CLEANUP_UNCONFIRMED:
                raise
            failure: Exception = e
        except Exception as e:  # noqa: BLE001 - every failed restore cold boots
            failure = e
        finally:
            entry.release()  # the VM, if any, holds its own share
        logger.warning(
            "L1 restore failed, falling back to cold boot",
            extra={
                "language": spec.language,
                "task_id": task_id,
                "error": f"{type(failure).__name__}: {failure}",
                "context": failure.context if isinstance(failure, SandboxError) else None,  # QEMU's console
            },
        )
        logger.debug("L1 restore traceback", exc_info=failure)
        refused = refusal_cause(failure) is not None or (
            isinstance(failure, SandboxError) and failure.code is ErrorCode.GUEST_SILENT_AFTER_RESTORE
        )
        if refused and checked is not None:
            await self._remove_broken(checked[0], stamp=checked[1], lease=None)
            logger.info(
                "L1 entry refused (by QEMU, or by its guest after the load), removed unless in use",
                extra={"vmstate_path": str(checked[0])},
            )
        # The guest answered the entropy ping with an error (guest_channel): no new saves
        if (
            isinstance(failure, SandboxError)
            and failure.code is ErrorCode.GUEST_SILENT_AFTER_RESTORE
            and "guest_error_type" in failure.context
        ):
            self._no_save_keys.add(key)
            logger.warning(
                "L1 saves stopped: the restored guest could not reseed its kernel random source", extra={"key": key}
            )
        return None

    async def schedule_background_save(
        self,
        spec: VmSpec,
        *,
        key: str,
        snapshot_drive: EntryLease | None,
    ) -> asyncio.Task[None] | None:
        """Schedule background L1 save: boot sacrificial VM, warm REPL, save, kill.

        `key` is compute_cache_key of `spec` and of the L2 layer `snapshot_drive` leases.
        The task holds its own share of that lease until it ends: no eviction removes the
        layer before its VM opens it.
        Returns the background task handle so callers can optionally await or
        monitor completion. Returns None (no-op) when QEMU is below the migration
        floor (can_save), the entry exists, a save of the key runs in this process, or a
        restore of the key in this process found a guest that could not reseed (restore).

        Used by the scheduler after a cold boot, and by the warm pool at start.
        The save lock prevents duplicate work: while another process saves the
        same key, the task skips (BlockingIOError, logged at debug level).
        """
        if key in self._no_save_keys or not await self.can_save():
            return None
        if await self.check_cache(key):
            return None

        # The template VM runs no user code: same device set as the spec (a
        # network device if it has one), without its ports or domain policy.
        template_spec = spec.model_copy(
            update={"allow_network": spec.has_network_device, "allowed_domains": None, "expose_ports": ()}
        )

        # In-process dedup: skip if a task for this key is already running
        if key in self._in_flight_saves:
            logger.debug("L1 background save already in-flight (in-process)", extra={"key": key})
            return None

        async def _background_save() -> None:
            start_time = asyncio.get_running_loop().time()
            try:
                # Acquire file lock for entire save lifecycle (VM boot + warm + save).
                # Prevents duplicate work: concurrent callers skip on BlockingIOError.
                # False: another process saved the entry meanwhile. The folder is made
                # again first (a tmp cleaner may have removed it).
                self._make_cache_dir()
                with CacheEntry(self._cache_path(key)).try_saving() as should_save:
                    if not should_save:
                        logger.debug("L1 background save skipped (already exists)", extra={"key": key})
                        return

                    # RAM file of the template save. The sacrificial VM uses
                    # memory-backend-file,share=on so QEMU writes guest RAM to this file.
                    # After migration with x-ignore-shared, the vmstate contains only
                    # device state; the RAM file is the template. The file is in memory
                    # when the host can (open_live_ram_dir), else {key}.ram itself. The
                    # host makes it (sparse, the RAM size: QEMU opens it O_RDWR and keeps
                    # its size), so qemu-vm needs no write access to its folder.
                    vmstate_path, meta_path = self._cache_paths(key)
                    result: Path | None = None
                    live_rams: list[_LiveRam] = []  # filled in its job, before a cancel reaches this task
                    try:
                        live_ram = await run_to_end(
                            functools.partial(
                                _new_live_ram,
                                ram_path_from_vmstate(vmstate_path),
                                size=template_spec.memory_mb * 1024 * 1024,
                                owner=live_rams,
                            )
                        )
                        if await probe_qemu_as_qemu_vm():  # the identity of the save VM's QEMU
                            await _grant_entry_files([live_ram.path], perms="rw", key=key)

                        # Boot sacrificial VM — ephemeral_vm handles lifecycle
                        async with self.vm_manager.ephemeral_vm(
                            template_spec,
                            tenant_id=constants.L1_SAVE_TENANT_ID,
                            task_id=f"l1-save-{spec.language.value}",
                            direct_write_target=None,
                            snapshot_drive=layer.path if layer is not None else None,
                            cache_leases=(layer,) if layer is not None else (),
                            retry_profile=constants.RETRY_BACKGROUND,
                            mem_path=str(live_ram.path),
                        ) as vm:
                            # Warm REPL (the interpreter start: 0.2-0.7 s with KVM or HVF, 3-5 s
                            # under TCG; it has its own timeout)
                            response = await vm.channel.send_request(
                                WarmReplRequest(language=spec.language),
                                timeout=constants.WARM_REPL_TIMEOUT_SECONDS,
                            )
                            if isinstance(response, WarmReplAckMessage) and response.status == "ok":
                                # Save L1 (terminates VM) — call _do_save directly (we already hold the lock)
                                # After _do_save, QEMU process is dead; ephemeral_vm's destroy_vm cleans up.
                                result = await self._do_save(
                                    vm,
                                    key,
                                    vmstate_path,
                                    meta_path,
                                    spec,
                                    live_ram=live_ram.path,
                                    live_ram_fd=live_ram.fd,
                                )
                            else:
                                logger.warning(
                                    "L1 background save: REPL warm failed", extra={"response": str(response)}
                                )
                    finally:  # one job, waited for on a cancel; its error never replaces the first one (a cancel)
                        try:
                            await run_to_end(
                                functools.partial(
                                    self._end_save, vmstate_path, published=result is not None, live_rams=live_rams
                                )
                            )
                        except Exception:
                            logger.warning("L1 background save: its files could not be removed", exc_info=True)

                    elapsed_ms = round((asyncio.get_running_loop().time() - start_time) * 1000)
                    if result:
                        logger.info(
                            "L1 background save completed",
                            extra={"key": key, "language": spec.language.value, "elapsed_ms": elapsed_ms},
                        )
                    else:
                        logger.warning(
                            "L1 background save produced no result",
                            extra={"key": key, "language": spec.language.value, "elapsed_ms": elapsed_ms},
                        )

            except BlockingIOError:
                logger.debug("L1 background save already in-flight", extra={"key": key})
            except Exception as e:
                elapsed_ms = round((asyncio.get_running_loop().time() - start_time) * 1000)
                logger.warning(
                    "L1 background save failed",
                    extra={"language": spec.language.value, "error": str(e), "elapsed_ms": elapsed_ms},
                    exc_info=True,
                )

        layer = snapshot_drive.share() if snapshot_drive is not None else None
        task = asyncio.create_task(_background_save())
        if layer is not None:  # also when the task is cancelled before it starts
            task.add_done_callback(lambda _t: layer.release())
        self._in_flight_saves[key] = task
        task.add_done_callback(lambda _t: self._in_flight_saves.pop(key, None))
        self._track_task(task, name=f"l1-save-{spec.language.value}-{key}")
        return task

    def _end_save(self, vmstate_path: Path, *, published: bool, live_rams: list[_LiveRam]) -> None:
        """A save's end (blocking): when nothing was published, the entry's files (the RAM
        file is the RAM size); then the RAM in memory, published or not, also when the
        removal fails (a read-only or failing disk)."""
        try:
            if not published:
                self._remove_entry(vmstate_path)
        finally:
            for live_ram in live_rams:
                live_ram.close()

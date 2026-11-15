"""qcow2 snapshot management for disk caching.

Implements two-tier snapshot architecture:
- L2 Cache: Local qcow2 disk snapshots (cold boot with cached packages)
- L3 Cache: S3 with zstd compression (cross-host sharing)

qcow2 optimizations:
- lazy_refcounts=on: Postpone metadata updates
- extended_l2=on: Faster CoW with subclusters
- cluster_size=128k: Balance between metadata and allocation

Snapshot structure:
- {cache_key}.qcow2: Standalone ext4 image (overlaid on EROFS base at boot)
- {cache_key}.qcow2.sha256: its content digest, cached (images.image_digest)
- {cache_key}.qcow2.lock: flock of its build (one build per key, across processes)
- {cache_key}.qcow2.use.lock: POSIX record lock each VM on the image holds shared
  (eviction skips the image meanwhile; files.locks)
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

import psutil

# Use native zstd module (Python 3.14+) or backports.zstd
if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

import logging

from tenacity import (
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from exec_sandbox import constants
from exec_sandbox._imports import aioboto3_session
from exec_sandbox._logging import get_logger
from exec_sandbox._version import __version__
from exec_sandbox.aio_utils import AsyncRetrying, run_stoppable, run_to_end
from exec_sandbox.base_cache_manager import BaseCacheManager
from exec_sandbox.dns_filter import resolved_domains
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InternalError,
    InvalidInputError,
    SandboxError,
    TransientError,
    spawn_error,
)
from exec_sandbox.files.disk import create_file, file_stamp
from exec_sandbox.files.locks import CacheEntry, EntryLease
from exec_sandbox.guest_agent_protocol import (
    InstallPackagesRequest,
    StreamingErrorMessage,
)
from exec_sandbox.guest_channel import consume_stream
from exec_sandbox.hash_utils import fingerprint
from exec_sandbox.images import key_part
from exec_sandbox.migration_client import MigrationClient
from exec_sandbox.permission_utils import get_expected_socket_uid
from exec_sandbox.qemu_vm import guest_error_to_exception
from exec_sandbox.resource_cleanup import cleanup_process
from exec_sandbox.s3_parts import upload_object
from exec_sandbox.settings import Settings  # noqa: TC001 - Used at runtime
from exec_sandbox.subprocess_utils import communicate_managed_process
from exec_sandbox.vm_spec import VmSpec

if TYPE_CHECKING:
    import threading
    from pathlib import Path

    from exec_sandbox.models import Language
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.vm_manager import VmManager

logger = get_logger(__name__)

_L2_VIRTUAL_SIZE_MB: Final = 512
"""Virtual size of an L2 snapshot image (qemu-img create)."""

_L2_MAX_FILE_BYTES: Final = (_L2_VIRTUAL_SIZE_MB + 64) * 1024 * 1024
"""Largest L2 qcow2 file: full virtual size plus qcow2 metadata headroom."""

_DECOMPRESS_STEP_BYTES: Final = 4 * 1024 * 1024
"""Max output of one zstd decompress step (bounds memory per step)."""

_L2_MODE: Final = 0o644
"""Mode of an L2 layer in the cache: QEMU as qemu-vm reads it, none may write it (every later
VM boots with it). open() would give 0664, group qemu-vm, under `sg qemu-vm` and umask 002."""

_SETUP_ERRNOS: Final = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})
"""File errors of a build that the same call gets again: no access, a read-only disk."""

_QEMU_IMG_CHECK_VALID: Final = frozenset({0, 3})
"""`qemu-img check` exit codes of a valid image: 0 no error, 3 leaked clusters only (space
lost inside the file after an interrupted write; the data is intact)."""

# In-process memo of passed `qemu-img check` runs: path -> file_stamp of the L2 file
# when it passed. One entry per path, like the images.image_digest cache. A rebuilt or
# replaced file gets a new stamp and is checked again.
_L2_CHECK_PASSED: dict[str, str] = {}


def _file_error(e: OSError, *, cache_key: str, language: Language, packages: list[str]) -> SandboxError:
    """Error of a build's file step. No access or a read-only disk: ConfigError SETTING_INVALID
    (a retry fails the same way: fix the folder). Any other file error: TransientError LOCAL_DISK."""
    reason = e.strerror or type(e).__name__
    if e.errno in _SETUP_ERRNOS:
        return ConfigError(
            f"Snapshot build {cache_key} cannot write its files: {reason}",
            code=ErrorCode.SETTING_INVALID,
            context={"cache_key": cache_key, "errno": e.errno},
            hint="Give this user write access to disk_snapshot_cache_dir, on a writable disk.",
            instance_id=None,
        )
    return TransientError(
        f"Local disk error for snapshot {cache_key}: {reason}",
        code=ErrorCode.LOCAL_DISK,
        context={"cache_key": cache_key, "language": language, "packages": packages},
        hint=None,
        instance_id=None,
    )


def _part_path(path: Path) -> Path:
    """Private temp name next to *path*; the pid lets _remove_stale_parts find orphans."""
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part")


def _remove_parts(*parts: Path) -> None:
    """Blocking: async callers use asyncio.to_thread."""
    for part in parts:
        part.unlink(missing_ok=True)


async def _qemu_img(args: list[str], *, process_name: str, cache_key: str) -> tuple[int, bytes, bytes]:
    """Run qemu-img with `args`. A missing binary, or one this user may not run, is
    ConfigError HOST_DEPENDENCY: a setup error that no retry fixes, never a local disk error;
    a fork that hits a process, memory or file-handle limit is TransientError OVERLAY_FAILED."""
    try:
        return await communicate_managed_process(["qemu-img", *args], process_name=process_name, context_id=cache_key)
    except OSError as e:
        raise spawn_error(
            e,
            binary="qemu-img",
            code=ErrorCode.OVERLAY_FAILED,
            context={"cache_key": cache_key},
            hint="Install QEMU (on Debian and Ubuntu, the qemu-utils package).",
        ) from e


async def _standalone_qcow2_problem(path: Path, cache_key: str) -> str | None:
    """Why a downloaded image is not a standalone qcow2, or None when it is.

    L2 images are standalone. A downloaded image that names a backing file or an
    external data file would make QEMU serve that host file to the guest.
    """
    returncode, stdout, stderr = await _qemu_img(
        ["info", "--output=json", "-U", str(path)], process_name="qemu-img-info", cache_key=cache_key
    )
    if returncode != 0:
        return f"qemu-img info failed: {stderr.decode().strip()}"
    info = json.loads(stdout)
    if info.get("format") != "qcow2":
        return f"S3 snapshot is not qcow2 (format={info.get('format')})"
    if "backing-filename" in info or "data-file" in info.get("format-specific", {}).get("data", {}):
        return "S3 snapshot names a backing or data file; L2 images are standalone"
    return None


_PERMANENT_PACKAGE_PATTERNS: Final[tuple[str, ...]] = (
    "no matching distribution",
    "could not find a version",
    "404 not found",
    "404 client error",
    "403 forbidden",
    "eresolve",
    "could not resolve dependency",
    "invalid version",
    "not found in registry",
    "no matching version",
    "version not found",
)
"""Patterns that indicate a permanent package install failure (won't succeed on retry)."""

_TRANSIENT_NETWORK_PATTERNS: Final[tuple[str, ...]] = (
    "client error",
    "connection refused",
    "connection reset",
    "connection timed out",
    "etimedout",
    "name resolution",
    "network unreachable",
    "server error",
    "ssl",
    "temporary failure in name resolution",
    "timed out",
    "reset by peer",
    "request failed after",
)
"""Patterns that indicate a transient network error (may succeed on retry)."""


def _classify_install_error(error_output: str) -> ErrorCode | None:
    """Two-phase error classification for package install failures.

    Phase 1: Check permanent patterns first (no matching distribution, 404):
    PACKAGE_INSTALL_FAILED.
    Phase 2: Check transient patterns (connection reset, timed out): PACKAGE_DOWNLOAD.
    Returns None if neither matches (the caller raises PACKAGE_INSTALL_FAILED).
    """
    lower = error_output.lower()
    # Phase 1: Permanent errors take priority — stop retrying immediately
    if any(pattern in lower for pattern in _PERMANENT_PACKAGE_PATTERNS):
        return ErrorCode.PACKAGE_INSTALL_FAILED
    # Phase 2: Transient errors — retry with backoff
    if any(pattern in lower for pattern in _TRANSIENT_NETWORK_PATTERNS):
        return ErrorCode.PACKAGE_DOWNLOAD
    return None


def _is_package_download(error: BaseException) -> bool:
    """The install retry predicate: a network failure of the package manager."""
    return isinstance(error, SandboxError) and error.code is ErrorCode.PACKAGE_DOWNLOAD


def _part_etag(etag: str | None) -> str:
    """The ETag of a part answer of an L3 upload (the upload fails without one)."""
    if not etag:
        raise ValueError("S3 part upload answer without an ETag")
    return etag


class DiskSnapshotManager(BaseCacheManager):
    """Manages qcow2 snapshot cache for disk caching.

    Architecture (2-tier):
    - L2 cache: Local qcow2 disk snapshots (cold boot with cached packages)
    - L3 cache: S3 with zstd compression (cross-host sharing)

    Cache key format: "{language}-v{major.minor}-{img_sha256}-{pkg_sha256}" (a request with
    no packages has no L2 layer: the scheduler boots it on the base image)

    Files only: one qcow2 file per snapshot, no metadata store (the cache key
    holds what the snapshot is). Eviction takes the entry with the oldest access
    time (st_atime), only when a build hits ENOSPC or admission needs home disk
    space.
    """

    _cache_ext: ClassVar[str] = ".qcow2"
    _sidecar_exts: ClassVar[tuple[str, ...]] = (".qcow2.sha256",)

    def __init__(self, settings: Settings, vm_manager: VmManager):
        """Initialize qcow2 snapshot manager.

        Args:
            settings: Application settings with cache configuration
            vm_manager: VmManager for VM operations

        Raises:
            ConfigError: SETTING_INVALID: the cache folder cannot be made, or is not private
            TransientError: LOCAL_DISK: a host error on the cache folder (disk full, I/O error)
        """
        super().__init__(settings.disk_snapshot_cache_dir, setting="disk_snapshot_cache_dir")
        self.settings = settings
        self.vm_manager = vm_manager

        # L3 client (lazy init)
        self._s3_session = None

        # Concurrency control: Limit concurrent snapshot creation to prevent resource exhaustion
        # Max 1 concurrent snapshot creation (heavy operations: VM boot + package install)
        self._creation_semaphore = asyncio.Semaphore(1)

        # Limit concurrent S3 uploads to prevent network saturation and memory exhaustion
        # S3 PutObject is atomic - aborted uploads leave no partial blobs
        self._upload_semaphore = asyncio.Semaphore(settings.max_concurrent_s3_uploads)
        self._stopping = False
        """stop() began: no upload starts."""

    async def start(self) -> None:
        """Delete .part files left by a killed process. One worker-thread job: the
        listing has no size limit and each part can be GBs."""
        await asyncio.to_thread(self._remove_stale_parts)

    async def stop(self) -> None:
        """Wait for the L3 uploads in flight at most L3_UPLOAD_STOP_GRACE_SECONDS, then cancel
        the ones left and wait until each ended; no upload starts once stop() began. An upload
        is a cache fill: a store that does not answer (minutes of botocore retries) does not
        hold the exit (the abort of a cancelled multipart upload waits at most
        s3_parts.ABORT_SECONDS). An interrupted upload leaves no object (S3 writes an object
        whole) and its compressed file is removed."""
        self._stopping = True
        if self._background_tasks:  # only uploads: _schedule_upload is the one tracker
            await asyncio.wait(list(self._background_tasks), timeout=constants.L3_UPLOAD_STOP_GRACE_SECONDS)
        for task in self._background_tasks:
            task.cancel()
        await super().stop()

    def _remove_stale_parts(self) -> None:
        """Delete .part files left by a killed process (name holds the writer's pid)."""
        for part in self.cache_dir.glob("*.part"):
            try:
                pid = int(part.name.split(".")[-3])
            except (IndexError, ValueError):
                continue
            if not psutil.pid_exists(pid):
                part.unlink(missing_ok=True)

    async def check_cache(
        self,
        language: Language,
        packages: list[str],
    ) -> EntryLease | None:
        """Check L2 cache without creating snapshot (Scheduler.reuse: a saved instance's
        L2 layer, used when its digest matches).

        Args:
            language: Programming language
            packages: Package list with versions

        Returns:
            A lease on the cached qcow2 snapshot (lease.path), or None on a miss. The
            caller passes it to the VM launch (cache_leases) and releases it after.

        Raises:
            OSError: the use lock cannot be opened (out of file handles, a disk error); without
                write access to the cache folder the lease holds no lock (EntryLease)
            TransientError: LOCAL_DISK: the base image cannot be read (images.key_part)
            ConfigError: ASSET_NOT_FOUND (the base image is missing), HOST_DEPENDENCY
                (qemu-img missing or not executable)
        """
        cache_key = await asyncio.to_thread(self._compute_cache_key, language, packages)
        return await self._lease_l2_cache(cache_key)

    async def _lease_l2_cache(self, cache_key: str) -> EntryLease | None:
        """A lease on the valid L2 entry `cache_key`, or None on a miss. Leased before it
        is checked: no eviction removes it from then on. An entry evicted first is a miss.

        Raises:
            OSError: the use lock cannot be opened (out of file handles, a disk error)
            ConfigError: HOST_DEPENDENCY: qemu-img missing or not executable
        """
        lease = await CacheEntry(self._cache_path(cache_key)).lease()
        if lease is None:
            return None
        try:
            checked = await self._check_l2_cache(cache_key, lease=lease)
        except BaseException:
            lease.release()
            raise
        if checked is None:  # a miss: broken (removed), gone, or qemu-img could not start now
            lease.release()  # idempotent: a removal released it already
            return None
        return lease

    async def get_or_create_snapshot(
        self,
        language: Language,
        packages: list[str],
        tenant_id: str,
        task_id: str,
        memory_mb: int,
    ) -> EntryLease:
        """Get cached snapshot or create new one.

        Cache hierarchy:
        1. Check L2 (local qcow2) → cold boot with cached disk
        2. Check L3 (S3 download) → download + cold boot
        3. Create new snapshot → package install + upload L3

        The snapshot is leased before any later step can evict it (a hit before its
        check, a download or a build before the build lock goes): the caller passes the
        lease to the VM launch (cache_leases) and releases it after.

        Args:
            language: Programming language
            packages: Package list with versions (e.g., ["pandas==2.1.0"])
            tenant_id: Tenant identifier
            task_id: Task identifier
            memory_mb: VM memory in MB (used for snapshot creation, not cache key)

        Returns:
            A lease on the snapshot qcow2 file (lease.path).

        Raises:
            TransientError: QEMU_CRASHED, LOCAL_DISK, OVERLAY_FAILED, PACKAGE_DOWNLOAD,
                GUEST_UNAVAILABLE: snapshot creation failed, can succeed later
            InvalidInputError: ARGUMENT_INVALID (the guest refused the package list),
                PACKAGE_INSTALL_FAILED
            ConfigError: HOST_DEPENDENCY (qemu-img missing or not executable),
                SETTING_INVALID (no write access to the build files, a read-only disk, a cache
                folder that cannot be made again), ASSET_NOT_FOUND (the base image is missing)
            InternalError: INTERNAL: unclassified creation failure; GUEST_PROTOCOL: the guest
                broke the protocol during the install
            SandboxError: an error of the build VM's start (VmManager.create_vm: VM_BOOT_TIMEOUT,
                GVPROXY_FAILED, CAPACITY, CAPACITY_NEVER_FITS, ...), raised as it is
        """
        cache_key = await asyncio.to_thread(self._compute_cache_key, language, packages)

        # Fast path: Check L2 cache without the build lock (read-only, safe for concurrent access)
        try:
            lease = await self._lease_l2_cache(cache_key)
        except OSError as e:
            raise _file_error(e, cache_key=cache_key, language=language, packages=packages) from e
        if lease is not None:
            logger.debug("L2 cache hit", extra={"cache_key": cache_key})
            return lease

        # Slow path: Need to create or wait for creation
        # File lock serializes across coroutines and processes; yields False if
        # the target .qcow2 was created by another caller while we waited.
        target_path = self._cache_path(cache_key)
        async with contextlib.AsyncExitStack() as stack:
            self._make_cache_dir()
            try:  # the lock file is the first file of a build in the cache folder
                should_create = await stack.enter_async_context(CacheEntry(target_path).saving())
            except OSError as e:
                raise _file_error(e, cache_key=cache_key, language=language, packages=packages) from e
            if not should_create:
                logger.debug("L2 cache hit (after lock)", extra={"cache_key": cache_key})
                return await self._lease_built(target_path, cache_key=cache_key, language=language, packages=packages)

            # L3 cache check (S3)
            if (snapshot_path := await self._download_from_s3(cache_key)) is not None:
                logger.debug("L3 cache hit", extra={"cache_key": cache_key})
                return await self._lease_built(snapshot_path, cache_key=cache_key, language=language, packages=packages)

            # Cache miss: Create new snapshot
            logger.debug("Cache miss, creating snapshot", extra={"cache_key": cache_key})
            snapshot_path = await self._create_snapshot(
                language,
                packages,
                cache_key=cache_key,
                tenant_id=tenant_id,
                task_id=task_id,
                memory_mb=memory_mb,
                enospc_retried=False,
            )
            lease = await self._lease_built(snapshot_path, cache_key=cache_key, language=language, packages=packages)

            # Upload to S3 (async, fire-and-forget) when a bucket is set
            if bucket := self.settings.s3_bucket:
                self._schedule_upload(cache_key, lease, bucket=bucket)

            return lease

    async def _lease_built(
        self, snapshot_path: Path, *, cache_key: str, language: Language, packages: list[str]
    ) -> EntryLease:
        """A lease on an image published under the build lock (eviction needs that lock
        too, so the image is there).

        Raises:
            TransientError: LOCAL_DISK: the image is gone all the same (removed by hand), or
                the use lock cannot be opened (out of file handles, a disk error: _file_error)
        """
        try:
            lease = await CacheEntry(snapshot_path).lease()
        except OSError as e:
            raise _file_error(e, cache_key=cache_key, language=language, packages=packages) from e
        if lease is None:
            raise _file_error(
                FileNotFoundError(errno.ENOENT, "snapshot removed under its build lock", str(snapshot_path)),
                cache_key=cache_key,
                language=language,
                packages=packages,
            )
        return lease

    def _compute_cache_key(
        self,
        language: Language,
        packages: list[str],
    ) -> str:
        """Compute L2 cache key for snapshot.

        Includes:
        - Library major.minor version (invalidates on lib upgrade)
        - Base image hash (invalidates when images are rebuilt)
        - Package hash (different packages = different cache entry)

        memory_mb is NOT in the cache key because disk-only snapshots work
        with any memory allocation.

        Note: allow_network is NOT in the cache key because:
        - Snapshots are always created with network (for pip/npm install)
        - User's allow_network setting only controls gvproxy at execution time

        Format: "{language}-v{major.minor}-{img_sha256}-{pkg_sha256}"

        Args:
            language: Programming language
            packages: Package list with versions (in any order: the key sorts it)

        Returns:
            Cache key string

        Raises:
            TransientError: LOCAL_DISK: the base image cannot be read (images.key_part)
            ConfigError: ASSET_NOT_FOUND: the base image is missing
        """
        # Extract major.minor from __version__ (e.g., "0.1.0" -> "0.1")
        version_parts = __version__.split(".")
        version = f"{version_parts[0]}.{version_parts[1]}"

        # Base image content digest: a rebuilt image gets a new key
        img_hash = key_part(self.vm_manager.images.base_image(language))

        return f"{language.value}-v{version}-{img_hash}-{fingerprint(*sorted(packages))}"

    async def _check_l2_cache(self, cache_key: str, *, lease: EntryLease) -> Path | None:
        """Check L2 local cache for qcow2 snapshot.

        Validates:
        1. Snapshot file exists
        2. Valid qcow2 format (via qemu-img check: _QEMU_IMG_CHECK_VALID), once per process
           for each version of the file (_L2_CHECK_PASSED); a failed check is not kept

        Args:
            cache_key: Snapshot cache key.
            lease: The caller's lease of the entry. A broken entry is removed as an eviction
                removes one (BaseCacheManager._remove_broken): this lease goes first, and
                another lease keeps the entry (its VM runs on it).

        Returns:
            Path to qcow2 snapshot if valid cache hit, None otherwise.
        """
        snapshot_path = self._cache_path(cache_key)

        try:
            stat = snapshot_path.stat()
        except OSError:
            return None

        # Verify qcow2 format (standalone ext4 snapshot, no backing file expected)
        stamp = file_stamp(stat)
        if _L2_CHECK_PASSED.get(str(snapshot_path)) != stamp:
            try:
                returncode, _, _ = await _qemu_img(
                    ["check", str(snapshot_path)], process_name="qemu-img-check", cache_key=cache_key
                )

                if returncode not in _QEMU_IMG_CHECK_VALID:
                    logger.warning(
                        "Invalid qcow2 snapshot, removing", extra={"cache_key": cache_key, "returncode": returncode}
                    )
                    await self._remove_broken(snapshot_path, stamp=stamp, lease=lease)  # with its digest sidecar
                    return None

            except (OSError, TransientError):  # qemu-img could not start now (spawn_error): a miss, not kept
                return None
            _L2_CHECK_PASSED[str(snapshot_path)] = stamp

        # Update atime for LRU tracking. Keep mtime: image_digest and _L2_CHECK_PASSED key on it.
        with contextlib.suppress(OSError):
            os.utime(snapshot_path, ns=(time.time_ns(), snapshot_path.stat().st_mtime_ns))

        return snapshot_path

    async def _create_snapshot(  # noqa: PLR0912, PLR0915
        self,
        language: Language,
        packages: list[str],
        *,
        cache_key: str,
        tenant_id: str,
        task_id: str,
        memory_mb: int,
        enospc_retried: bool,
    ) -> Path:
        """Create new qcow2 snapshot with packages installed.

        Uses asyncio.wait racing for instant crash detection.

        Workflow:
        1. Create a standalone qcow2 (no backing file) under a private .part name
        2. Boot VM with snapshot image (ephemeral_vm handles lifecycle)
        3. Install packages via guest agent (with death monitoring)
        4. Shutdown VM (writes committed to snapshot)
        5. Rename the .part file to the snapshot path, and return that path
        On ENOSPC: evict the oldest entry, then build once more.

        Args:
            language: Programming language
            packages: Package list with versions
            cache_key: Snapshot cache key
            tenant_id: Tenant identifier
            task_id: Task identifier
            memory_mb: VM memory in MB

        Returns:
            Path to created qcow2 snapshot

        Raises:
            TransientError: QEMU_CRASHED (VM crashed during snapshot creation),
                LOCAL_DISK (disk full after eviction retry, other local disk error),
                OVERLAY_FAILED, PACKAGE_DOWNLOAD, GUEST_UNAVAILABLE
            InvalidInputError: ARGUMENT_INVALID (the guest refused the package list),
                PACKAGE_INSTALL_FAILED
            ConfigError: HOST_DEPENDENCY (qemu-img missing or not executable),
                SETTING_INVALID (no write access to the build files, a read-only disk)
            InternalError: INTERNAL: any other failure; GUEST_PROTOCOL: the guest broke the
                protocol during the install
            SandboxError: an error of the build VM's start (VmManager.create_vm: VM_BOOT_TIMEOUT,
                GVPROXY_FAILED, CAPACITY, CAPACITY_NEVER_FITS, ...), raised as it is
        """
        start_time = asyncio.get_running_loop().time()
        snapshot_path = self._cache_path(cache_key)
        # Build under a private name: lock-free readers (_check_l2_cache) never see
        # a half-built image, and the final name appears only after a clean exit.
        build_path = _part_path(snapshot_path)
        success = False
        retry = False

        def build_failed(error: BaseException) -> InternalError:
            """A build failure this code does not expect (a bug, not the guest or the disk)."""
            return InternalError(
                f"Failed to create snapshot: {error}",
                code=ErrorCode.INTERNAL,
                context={"cache_key": cache_key, "language": language, "packages": packages, "tenant_id": tenant_id},
                hint=None,
                instance_id=None,
            )

        # Ensure snapshot VMs have enough memory for package managers (uv/bun).
        # Under TCG, uv cold-start JIT + TLS crypto pushes RSS above 192MB,
        # triggering zram compression in emulated code (catastrophically slow).
        effective_memory_mb = max(memory_mb, constants.SNAPSHOT_CREATION_MIN_MEMORY_MB)
        if effective_memory_mb != memory_mb:
            logger.debug(
                "Snapshot VM memory raised to minimum",
                extra={"requested_mb": memory_mb, "effective_mb": effective_memory_mb},
            )

        # Acquire semaphore to limit concurrent snapshot creation
        async with self._creation_semaphore:
            try:
                # Step 1: Create standalone ext4 qcow2 for overlay (vdb)
                await self._create_snapshot_image(build_path, cache_key, language, packages, tenant_id)

                # Steps 2-5: Boot VM, install packages, shutdown — ephemeral_vm handles
                # lifecycle. The network is on (pip/npm needs it), limited to the
                # language's package registries
                build_spec = VmSpec(
                    language=language,
                    memory_mb=effective_memory_mb,
                    allow_network=True,
                    allowed_domains=resolved_domains(None, language),
                )
                async with self.vm_manager.ephemeral_vm(
                    build_spec,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    direct_write_target=build_path,
                    snapshot_drive=None,
                    cache_leases=(),
                    retry_profile=constants.RETRY_BACKGROUND,
                    mem_path=None,
                ) as vm:
                    # Install packages with death monitoring (asyncio.wait)
                    death_task = asyncio.create_task(self._monitor_vm_death(vm, cache_key))
                    install_task = asyncio.create_task(self._install_packages(vm, language, packages))
                    try:
                        await asyncio.wait((death_task, install_task), return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        # Neither task outlives the build, also on a cancel: both cancelled (no
                        # effect on one that ended), then waited for
                        death_task.cancel()
                        install_task.cancel()
                        await asyncio.wait((death_task, install_task))
                    # The VM's death when it ended (an install error is then its effect), else
                    # the install's outcome. Both errors are read: asyncio logs an unread one.
                    install_error = None if install_task.cancelled() else install_task.exception()
                    if not death_task.cancelled():
                        death_task.result()  # raises: the watch ends only on the VM's death
                    if install_error is not None:
                        raise install_error

                    # Shutdown QEMU process cleanly: QMP quit first, so QEMU closes the layer and
                    # exits 0 whatever wraps it. A signal alone does not do it: an `unshare --fork`
                    # parent ignores SIGTERM, and the escalation to SIGKILL ends QEMU unclean.
                    # When quit fails (no QMP), cleanup_process sends the signals.
                    we_initiated_shutdown = vm.process.returncode is None
                    if we_initiated_shutdown:
                        # 2 s for all of it: else connect, handshake and quit could take ~12 s
                        with contextlib.suppress(TransientError, TimeoutError):
                            async with (
                                asyncio.timeout(2.0),
                                MigrationClient(vm.qmp_socket, get_expected_socket_uid(vm.use_qemu_vm_user)) as client,
                            ):
                                await client.quit()
                    await cleanup_process(
                        vm.process, "QEMU", cache_key, term_timeout=5.0, kill_timeout=2.0, kill_first=False
                    )

                    # Verify QEMU exited cleanly. Only a clean exit closes the layer: SIGKILL (-9),
                    # from the escalation after term_timeout or from a destroy that a Scheduler
                    # stop runs meanwhile, can leave a qcow2 that `qemu-img check` refuses
                    if we_initiated_shutdown:
                        if vm.process.returncode not in {0, -15}:
                            raise TransientError(
                                f"QEMU exited unexpectedly after its shutdown (exit code {vm.process.returncode})",
                                code=ErrorCode.QEMU_CRASHED,
                                context={
                                    "cache_key": cache_key,
                                    "exit_code": vm.process.returncode,
                                    "language": language,
                                    "packages": packages,
                                },
                                hint=None,
                                instance_id=None,
                            )
                    elif vm.process.returncode != 0:
                        raise TransientError(
                            f"QEMU died unexpectedly during snapshot creation (exit code {vm.process.returncode})",
                            code=ErrorCode.QEMU_CRASHED,
                            context={
                                "cache_key": cache_key,
                                "exit_code": vm.process.returncode,
                                "language": language,
                                "packages": packages,
                            },
                            hint=None,
                            instance_id=None,
                        )
                    else:
                        logger.info(
                            "QEMU exited cleanly (code 0) after install completed",
                            extra={"cache_key": cache_key, "language": language},
                        )

                # ephemeral_vm's finally calls destroy_vm (idempotent), and gives this user back
                # a target that QEMU wrote as qemu-vm. qemu-img makes it 0644 minus the umask;
                # qemu-vm owned it during the build and may have changed its mode.
                build_path.chmod(_L2_MODE)
                build_path.replace(snapshot_path)
                success = True

            # Handle disk full (lazy eviction)
            except OSError as e:
                if isinstance(e, TimeoutError):  # an OSError in Python, not a disk error: unexpected here
                    raise build_failed(e) from e
                if e.errno != errno.ENOSPC or enospc_retried:
                    raise _file_error(e, cache_key=cache_key, language=language, packages=packages) from e
                try:
                    await self.evict_oldest()
                except OSError as evict_error:  # the cache folder cannot be listed
                    raise _file_error(
                        evict_error, cache_key=cache_key, language=language, packages=packages
                    ) from evict_error
                retry = True  # after the semaphore is released: the retry takes it again

            except (asyncio.CancelledError, SandboxError):
                raise

            except Exception as e:
                raise build_failed(e) from e

            finally:
                # Cleanup the private build file on failure
                if not success and build_path.exists():
                    try:
                        await asyncio.to_thread(build_path.unlink)  # up to GBs
                        logger.debug("Snapshot build file cleaned up on failure", extra={"cache_key": cache_key})
                    except OSError as e:
                        logger.warning(
                            "Failed to cleanup snapshot build file",
                            extra={"cache_key": cache_key, "error": str(e)},
                        )

        if retry:
            return await self._create_snapshot(
                language,
                packages,
                cache_key=cache_key,
                tenant_id=tenant_id,
                task_id=task_id,
                memory_mb=memory_mb,
                enospc_retried=True,
            )

        # Record snapshot creation duration
        duration_ms = round((asyncio.get_running_loop().time() - start_time) * 1000)
        logger.info(
            "Snapshot created",
            extra={
                "cache_key": cache_key,
                "language": language,
                "package_count": len(packages),
                "duration_ms": duration_ms,
            },
        )

        return snapshot_path

    async def _create_snapshot_image(
        self,
        snapshot_path: Path,
        cache_key: str,
        language: str,
        packages: list[str],
        tenant_id: str,
    ) -> None:
        """Create standalone ext4 qcow2 for snapshot overlay (vdb).

        With EROFS rootfs, snapshots are standalone ext4 qcow2 images that serve
        as the upper layer in overlayfs (merged with the read-only EROFS base on vda).
        tiny-init formats and mounts this as ext4 writable during snapshot creation.

        Args:
            snapshot_path: Path to snapshot to create
            cache_key: Snapshot cache key
            language: Programming language
            packages: Package list
            tenant_id: Tenant identifier

        Raises:
            TransientError: OVERLAY_FAILED: qemu-img command failed, or its fork hit a
                process, memory or file-handle limit
            ConfigError: HOST_DEPENDENCY: qemu-img missing or not executable
        """
        returncode, _, stderr = await _qemu_img(
            [
                "create",
                "-f",
                "qcow2",
                "-o",
                "cluster_size=128k,lazy_refcounts=on,extended_l2=on",
                str(snapshot_path),
                f"{_L2_VIRTUAL_SIZE_MB}M",
            ],
            process_name="qemu-img-create-snapshot",
            cache_key=cache_key,
        )
        if returncode != 0:
            stderr_text = stderr.decode().strip()
            raise TransientError(
                f"qemu-img create failed: {stderr_text}",
                code=ErrorCode.OVERLAY_FAILED,
                context={
                    "cache_key": cache_key,
                    "language": language,
                    "packages": packages,
                    "tenant_id": tenant_id,
                    "stderr": stderr_text,
                },
                hint=None,
                instance_id=None,
            )

    async def _monitor_vm_death(self, vm: QemuVM, cache_key: str) -> None:
        """Monitor VM process for unexpected death during snapshot creation."""
        returncode = await vm.process.wait()
        diag = vm.collect_diagnostics()
        logger.error(
            "VM died during snapshot creation",
            extra={**asdict(diag), "cache_key": cache_key},
        )
        raise TransientError(
            f"VM process died during snapshot creation (exit code {returncode})",
            code=ErrorCode.QEMU_CRASHED,
            context=asdict(diag),
            hint=None,
            instance_id=None,
        )

    async def _install_packages(
        self,
        vm: QemuVM,
        language: Language,
        packages: list[str],
    ) -> None:
        """Install packages in VM via guest agent.

        Retries transient network errors (DNS, connection refused, timeouts)
        using exponential backoff. The VM stays running between retries — only
        the install request is re-sent.

        Args:
            vm: QemuVM handle
            language: Programming language
            packages: Package list with versions

        Raises:
            TransientError: PACKAGE_DOWNLOAD (network error after all retries, or
                install timeout), GUEST_UNAVAILABLE (communication error), LOCAL_DISK
                (QEMU stopped the VM because the host disk is full)
            InvalidInputError: PACKAGE_INSTALL_FAILED (permanent package error or
                unclassified failure, no retry), ARGUMENT_INVALID (the guest refused
                the package list: a language without packages, empty, too long, a bad name)
            InternalError: GUEST_PROTOCOL
        """
        request = InstallPackagesRequest(
            language=language,
            packages=packages,
            timeout=constants.PACKAGE_INSTALL_TIMEOUT_SECONDS,
        )

        # +8s: the install is the first guest op on a fresh network-enabled VM,
        # so the guest's network readiness gate (worker timeout + 1) can consume
        # up to 8s before the install's own timeout starts counting.
        hard_timeout = constants.PACKAGE_INSTALL_TIMEOUT_SECONDS + constants.EXECUTION_TIMEOUT_MARGIN_SECONDS + 8
        total_timeout = (
            hard_timeout * constants.PACKAGE_INSTALL_MAX_RETRIES
            + constants.PACKAGE_INSTALL_RETRY_MAX_SECONDS * (constants.PACKAGE_INSTALL_MAX_RETRIES - 1)
        )

        async def _handle_install_error(msg: StreamingErrorMessage) -> None:
            logger.error(
                "Guest agent install error",
                extra={"vm_id": vm.vm_id, "error": msg.message, "error_type": msg.error_type},
            )
            exc = guest_error_to_exception(msg, vm.vm_id, operation="install_packages")
            # Guest timeouts during install are transient — wrap so tenacity retries
            if isinstance(exc, TransientError):
                raise TransientError(
                    str(exc), code=ErrorCode.PACKAGE_DOWNLOAD, context={"vm_id": vm.vm_id}, hint=None, instance_id=None
                ) from exc
            raise exc

        started = time.monotonic()
        try:
            async with asyncio.timeout(total_timeout):
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(constants.PACKAGE_INSTALL_MAX_RETRIES),
                    wait=wait_random_exponential(
                        min=constants.PACKAGE_INSTALL_RETRY_MIN_SECONDS,
                        max=constants.PACKAGE_INSTALL_RETRY_MAX_SECONDS,
                    ),
                    retry=retry_if_exception(_is_package_download),
                    before_sleep=before_sleep_log(logger, logging.WARNING),
                    reraise=True,
                ):
                    with attempt:
                        await vm.channel.connect(timeout_seconds=constants.GUEST_CONNECT_TIMEOUT_SECONDS)

                        result = await consume_stream(
                            vm.channel,
                            request,
                            timeout=hard_timeout,
                            vm_id=vm.vm_id,
                            on_stdout=None,
                            on_stderr=None,
                            on_error=_handle_install_error,
                        )

                        if result.exit_code != 0:
                            error_output = result.stderr or "Unknown error"
                            code = _classify_install_error(error_output)
                            context = {"exit_code": result.exit_code, "stderr": error_output[:500]}
                            if code is ErrorCode.PACKAGE_DOWNLOAD:
                                raise TransientError(
                                    f"Package installation failed with transient network error (exit code {result.exit_code}): {error_output[:500]}",
                                    code=ErrorCode.PACKAGE_DOWNLOAD,
                                    context=context,
                                    hint=None,
                                    instance_id=None,
                                )
                            raise InvalidInputError(
                                f"Package installation failed permanently (exit code {result.exit_code}): {error_output[:500]}"
                                if code is ErrorCode.PACKAGE_INSTALL_FAILED
                                else f"Package installation failed with exit code {result.exit_code}: {error_output[:500]}",
                                code=ErrorCode.PACKAGE_INSTALL_FAILED,
                                context=context,
                                hint=None,
                                instance_id=None,
                            )

        except TimeoutError as e:
            # A host deadline: the guest connection or the stream of one attempt (not
            # retried: only PACKAGE_DOWNLOAD is), or the bound of all attempts. A guest
            # that QEMU paused on a full host disk never answers: LOCAL_DISK.
            await vm.raise_if_host_disk_full()
            elapsed = time.monotonic() - started
            raise TransientError(
                f"Package installation timeout after {elapsed:.1f}s",
                code=ErrorCode.PACKAGE_DOWNLOAD,
                context={
                    "vm_id": vm.vm_id,
                    "language": language,
                    "packages": packages,
                    "hard_timeout_seconds": hard_timeout,
                    "total_timeout_seconds": total_timeout,
                    **await vm.deadline_diagnostics(operation="install_packages"),
                },
                hint=None,
                instance_id=None,
            ) from e

        # An explicit list, not SandboxError: GUEST_OUTCOME_UNKNOWN (VERIFY) must not
        # escape Scheduler.new for an install that ran no user code.
        except (InvalidInputError, TransientError, InternalError):
            raise

        except Exception as e:
            # A full host disk paused the VM: the stop failed the install (OutcomeUnknownError
            # in flight, a closed channel after): LOCAL_DISK, not a guest problem
            await vm.raise_if_host_disk_full()
            raise TransientError(
                f"Package installation failed (communication error): {e}",
                code=ErrorCode.GUEST_UNAVAILABLE,
                context={"vm_id": vm.vm_id, "language": language, "packages": packages},
                hint=None,
                instance_id=None,
            ) from e

    def _s3_key(self, name: str) -> str:
        """Object key for an L3 entry, under the configured prefix."""
        return f"{self.settings.s3_snapshot_prefix}{name}"

    async def _download_from_s3(self, cache_key: str) -> Path | None:
        """Download, decompress and verify a snapshot from S3 into the L2 cache.

        The object's ``sha256`` metadata (written at upload) must match the
        decompressed bytes; a truncated zstd stream, a missing digest or a
        mismatch is a cache miss. The final file appears atomically (rename),
        so a concurrent reader never sees a partial snapshot.

        Args:
            cache_key: Snapshot cache key

        Returns:
            Path to downloaded qcow2 snapshot, or None on a miss (no bucket, no
            object, a missing digest, a mismatch, a truncated or oversized stream,
            an image that is not a standalone qcow2, any failure): it is logged.
        """
        if not self.settings.s3_bucket:
            return None
        snapshot_path = self._cache_path(cache_key)
        compressed_name = snapshot_path.with_suffix(".qcow2.zst").name
        compressed_path = _part_path(snapshot_path.with_name(compressed_name))
        partial_path = _part_path(snapshot_path)

        try:
            s3_key = self._s3_key(compressed_name)
            # download_file opens the file to write (truncate): it keeps the mode made here
            create_file(compressed_path, mode=0o600, dir_fd=None).close()
            async with await self._get_s3_client() as s3:  # type: ignore[union-attr]
                head = cast(
                    "dict[str, Any]",
                    await s3.head_object(Bucket=self.settings.s3_bucket, Key=s3_key),  # type: ignore[union-attr]
                )
                metadata = cast("dict[str, str]", head.get("Metadata", {}))
                expected_sha256 = metadata.get("sha256", "")
                if not expected_sha256:
                    raise ValueError(f"S3 object {s3_key} has no sha256 metadata")
                await s3.download_file(  # type: ignore[union-attr]
                    self.settings.s3_bucket,
                    s3_key,
                    str(compressed_path),
                )

            chunk_size = 64 * 1024  # 64KB chunks for streaming

            def _decompress_and_hash() -> str:
                # Bounded: max_length caps each output step, and the total stops
                # at the largest qcow2 an L2 build can produce (zstd bomb guard).
                decompressor = zstd.ZstdDecompressor()
                hasher = hashlib.sha256()
                written = 0
                with compressed_path.open("rb") as src, create_file(partial_path, mode=_L2_MODE, dir_fd=None) as dst:
                    while chunk := src.read(chunk_size):
                        while True:
                            out = decompressor.decompress(chunk, max_length=_DECOMPRESS_STEP_BYTES)
                            chunk = b""
                            written += len(out)
                            if written > _L2_MAX_FILE_BYTES:
                                raise ValueError("S3 snapshot is larger than any L2 image")
                            hasher.update(out)
                            dst.write(out)
                            if decompressor.needs_input or decompressor.eof:
                                break
                if not decompressor.eof:
                    raise ValueError("S3 snapshot stream is truncated")
                compressed_path.unlink()  # read whole: removed here, in this worker thread (up to GBs)
                return hasher.hexdigest()

            actual_sha256 = await asyncio.to_thread(_decompress_and_hash)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"S3 snapshot digest mismatch for {s3_key}: expected {expected_sha256}, got {actual_sha256}"
                )
            if problem := await _standalone_qcow2_problem(partial_path, cache_key):
                raise ValueError(problem)
            partial_path.replace(snapshot_path)

        except Exception as e:  # noqa: BLE001 - every failure (S3, a check above) is a miss
            logger.warning("L3 cache miss", extra={"cache_key": cache_key, "reason": str(e)})
            return None
        finally:
            # After a failure each can be GBs (after a success both are gone): one worker-thread job
            await asyncio.to_thread(_remove_parts, compressed_path, partial_path)

        return snapshot_path

    def _schedule_upload(self, cache_key: str, lease: EntryLease, *, bucket: str) -> None:
        """Upload the snapshot `lease` holds to S3 in a background task. The task holds its own
        share of the lease until it ends (also when cancelled before it starts): no eviction
        removes the image while the task waits for its turn or reads it."""
        if self._stopping:
            logger.info("S3 upload skipped: the manager is stopping", extra={"cache_key": cache_key})
            return
        share = lease.share()
        task = asyncio.create_task(self._upload_to_s3(cache_key, share.path, bucket=bucket))
        task.add_done_callback(lambda _t: share.release())
        self._track_task(task, name=f"s3-upload-{cache_key}")

    async def _upload_to_s3(self, cache_key: str, snapshot_path: Path, *, bucket: str) -> None:
        """Upload compressed snapshot to S3 (async, fire-and-forget). The caller keeps the
        image leased until the upload ends (_schedule_upload).

        Bounded by upload_semaphore to prevent:
        - Network saturation
        - Memory exhaustion from compression buffers
        - S3 rate limiting (unlikely but possible)

        Args:
            cache_key: Snapshot cache key
            snapshot_path: Local qcow2 snapshot path
            bucket: The L3 bucket (s3_bucket, set)
        """
        compressed_path = _part_path(snapshot_path.with_suffix(".qcow2.zst"))

        # Acquire semaphore to limit concurrent uploads
        async with self._upload_semaphore:
            try:
                # Compress with zstd (level 3 for speed, run in thread pool to avoid blocking)
                chunk_size = 64 * 1024  # 64KB chunks for streaming

                def _compress(stop: threading.Event) -> str:
                    # Digest from the same read as the compressed bytes: the metadata always
                    # matches the object. A stop (the upload is cancelled) ends the job at its
                    # next chunk and removes the partial file; its "" is never used.
                    compressor = zstd.ZstdCompressor(level=3)
                    hasher = hashlib.sha256()
                    with (
                        snapshot_path.open("rb") as src,
                        create_file(compressed_path, mode=0o600, dir_fd=None) as dst,
                    ):
                        while not stop.is_set():
                            chunk = src.read(chunk_size)
                            if not chunk:
                                break
                            hasher.update(chunk)
                            compressed = compressor.compress(chunk)
                            if compressed:
                                dst.write(compressed)
                        if not stop.is_set():
                            # Flush remaining data
                            final = compressor.flush()
                            if final:
                                dst.write(final)
                    if stop.is_set():
                        compressed_path.unlink(missing_ok=True)
                        return ""
                    return hasher.hexdigest()

                snapshot_sha256 = await run_stoppable(
                    _compress, undo=lambda _digest: compressed_path.unlink(missing_ok=True)
                )

                async with await self._get_s3_client() as s3:  # type: ignore[union-attr]
                    # Upload compressed qcow2; the digest of the uncompressed
                    # bytes lets downloads verify the content end to end.
                    s3_key = self._s3_key(snapshot_path.with_suffix(".qcow2.zst").name)
                    # Not upload_file: its multipart upload is not aborted on a cancel (stop()
                    # after the grace), and an incomplete upload would stay in the bucket
                    await upload_object(
                        s3,
                        bucket=bucket,
                        key=s3_key,
                        src=compressed_path,
                        extra={
                            "Tagging": f"ttl_days={self.settings.snapshot_cache_ttl_days}",
                            "Metadata": {"sha256": snapshot_sha256},
                        },
                        part_etag=_part_etag,
                        abort_errors=(Exception,),  # a fire-and-forget cache fill: any abort error
                    )

                # Cleanup compressed file (up to GBs)
                await asyncio.to_thread(compressed_path.unlink)

            except asyncio.CancelledError:  # stop(): the compressed file goes, also on a second cancel
                await run_to_end(lambda: compressed_path.unlink(missing_ok=True))
                raise
            except Exception as e:  # noqa: BLE001 - Fire-and-forget S3 upload
                # Silent failure (L2 cache still works)
                # Catch all exceptions including botocore.exceptions.ClientError
                logger.warning("S3 upload failed silently", extra={"cache_key": cache_key, "error": str(e)})
                await asyncio.to_thread(compressed_path.unlink, missing_ok=True)

    async def _get_s3_client(self):  # type: ignore[no-untyped-def]
        """Get S3 client (lazy init). The callers check that s3_bucket is set.

        Raises:
            ConfigError: EXTRA_MISSING if aioboto3 is not installed

        Returns:
            S3 client context manager from aioboto3 (untyped library)
        """
        if self._s3_session is None:
            # On the API models of the process: a session of its own parses them again (~0.2 s)
            self._s3_session = aioboto3_session()

        return self._s3_session.client(  # type: ignore[no-any-return]
            "s3",
            region_name=self.settings.s3_region,
            endpoint_url=self.settings.s3_endpoint_url,
        )

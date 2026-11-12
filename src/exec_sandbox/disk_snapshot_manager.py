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
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

import psutil

# Use native zstd module (Python 3.14+) or backports.zstd
if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

import logging

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from exec_sandbox import __version__, constants
from exec_sandbox._imports import require_aioboto3
from exec_sandbox._logging import get_logger
from exec_sandbox.base_cache_manager import BaseCacheManager, file_stamp
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InternalError,
    InvalidInputError,
    SandboxError,
    TransientError,
    spawn_error,
)
from exec_sandbox.guest_agent_protocol import (
    InstallPackagesRequest,
    StreamingErrorMessage,
)
from exec_sandbox.guest_channel import consume_stream
from exec_sandbox.hash_utils import fingerprint
from exec_sandbox.lock_utils import file_lock
from exec_sandbox.models import Language
from exec_sandbox.qemu_vm import guest_error_to_exception
from exec_sandbox.resource_cleanup import cleanup_process
from exec_sandbox.settings import Settings  # noqa: TC001 - Used at runtime
from exec_sandbox.subprocess_utils import communicate_managed_process
from exec_sandbox.vm_spec import VmSpec

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.vm_manager import VmManager

logger = get_logger(__name__)

_L2_VIRTUAL_SIZE_MB: Final = 512
"""Virtual size of an L2 snapshot image (qemu-img create)."""

_L2_MAX_FILE_BYTES: Final = (_L2_VIRTUAL_SIZE_MB + 64) * 1024 * 1024
"""Largest L2 qcow2 file: full virtual size plus qcow2 metadata headroom."""

_DECOMPRESS_STEP_BYTES: Final = 4 * 1024 * 1024
"""Max output of one zstd decompress step (bounds memory per step)."""

_SETUP_ERRNOS: Final = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})
"""File errors of a build that the same call gets again: no access, a read-only disk."""

# In-process memo of passed `qemu-img check` runs: path -> file_stamp of the L2 file
# when it passed. One entry per path, like the image_hash digest cache. A rebuilt or
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


async def _qemu_img(args: list[str], *, process_name: str, cache_key: str) -> tuple[int | None, bytes, bytes]:
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


class DiskSnapshotManager(BaseCacheManager):
    """Manages qcow2 snapshot cache for disk caching.

    Architecture (2-tier):
    - L2 cache: Local qcow2 disk snapshots (cold boot with cached packages)
    - L3 cache: S3 with zstd compression (cross-host sharing)

    Cache key format:
    - "{language}-v{major.minor}-base" for base images (no packages)
    - "{language}-v{major.minor}-{16char_hash}" for packages

    Simplifications:
    - ❌ No Redis (never implemented)
    - ❌ No metadata tracking (parse from cache_key)
    - ❌ No proactive eviction (lazy on disk full)
    - ✅ Pure filesystem (atime tracking only)
    - ✅ Single qcow2 file per snapshot
    """

    _cache_ext: ClassVar[str] = ".qcow2"
    _sidecar_exts: ClassVar[tuple[str, ...]] = (".qcow2.sha256",)

    def __init__(self, settings: Settings, vm_manager: VmManager):
        """Initialize qcow2 snapshot manager.

        Args:
            settings: Application settings with cache configuration
            vm_manager: VmManager for VM operations
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

    async def start(self) -> None:
        """Delete .part files left by a killed process. One worker-thread job: the
        listing has no size limit and each part can be GBs."""
        await asyncio.to_thread(self._remove_stale_parts)

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
    ) -> Path | None:
        """Check L2 cache without creating snapshot.

        Use this for warm pool: creating L2 cache for base images (no packages)
        is pointless - would boot VM just to shut it down. Check-only instead.
        Returns cached snapshot if available, None if cache miss.

        Args:
            language: Programming language
            packages: Package list (empty for base image)

        Returns:
            Path to cached qcow2 snapshot, or None if cache miss.
        """
        cache_key = await asyncio.to_thread(self._compute_cache_key, language, packages)
        return await self._check_l2_cache(cache_key)

    async def get_or_create_snapshot(
        self,
        language: Language,
        packages: list[str],
        tenant_id: str,
        task_id: str,
        memory_mb: int,
    ) -> Path:
        """Get cached snapshot or create new one.

        Cache hierarchy:
        1. Check L2 (local qcow2) → cold boot with cached disk
        2. Check L3 (S3 download) → download + cold boot
        3. Create new snapshot → package install + upload L3

        Args:
            language: Programming language
            packages: Package list with versions (e.g., ["pandas==2.1.0"])
            tenant_id: Tenant identifier
            task_id: Task identifier
            memory_mb: VM memory in MB (used for snapshot creation, not cache key)

        Returns:
            Path to snapshot qcow2 file.

        Raises:
            TransientError: QEMU_CRASHED, LOCAL_DISK, OVERLAY_FAILED, PACKAGE_DOWNLOAD,
                GUEST_UNAVAILABLE: snapshot creation failed, can succeed later
            InvalidInputError: ARGUMENT_INVALID (the guest refused the package list),
                PACKAGE_INSTALL_FAILED
            ConfigError: HOST_DEPENDENCY (qemu-img missing or not executable),
                SETTING_INVALID (no write access to the build files, a read-only disk, a cache
                folder that cannot be made again)
            InternalError: INTERNAL: unclassified creation failure
        """
        cache_key = await asyncio.to_thread(self._compute_cache_key, language, packages)

        # Fast path: Check L2 cache without lock (read-only, safe for concurrent access)
        snapshot_path = await self._check_l2_cache(cache_key)
        if snapshot_path:
            logger.debug("L2 cache hit", extra={"cache_key": cache_key})
            return snapshot_path

        # Slow path: Need to create or wait for creation
        # File lock serializes across coroutines and processes; yields False if
        # the target .qcow2 was created by another caller while we waited.
        target_path = self._cache_path(cache_key)
        async with contextlib.AsyncExitStack() as stack:
            self._make_cache_dir()
            try:  # the lock file is the first file of a build in the cache folder
                should_create = await stack.enter_async_context(file_lock(target_path))
            except OSError as e:
                raise _file_error(e, cache_key=cache_key, language=language, packages=packages) from e
            if not should_create:
                logger.debug("L2 cache hit (after lock)", extra={"cache_key": cache_key})
                return target_path

            # L3 cache check (S3) - only for images with packages
            # Base images are already distributed via asset downloads, no need for S3
            if packages and (snapshot_path := await self._download_from_s3(cache_key)) is not None:
                logger.debug("L3 cache hit", extra={"cache_key": cache_key})
                return snapshot_path

            # Cache miss: Create new snapshot
            logger.debug("Cache miss, creating snapshot", extra={"cache_key": cache_key})
            snapshot_path = await self._create_snapshot(
                language, packages, cache_key=cache_key, tenant_id=tenant_id, task_id=task_id, memory_mb=memory_mb
            )

            # Upload to S3 (async, fire-and-forget) - only for images with packages
            # Base images don't need S3 - they're already globally distributed
            if packages:
                upload_task: asyncio.Task[None] = asyncio.create_task(self._upload_to_s3(cache_key, snapshot_path))
                self._track_task(upload_task, name=f"s3-upload-{cache_key}")

            return snapshot_path

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

        Format:
        - "{language}-v{major.minor}-{img_hash}-base" for base (no packages)
        - "{language}-v{major.minor}-{img_hash}-{16char_pkg_hash}" for packages

        Args:
            language: Programming language
            packages: Sorted package list with versions

        Returns:
            Cache key string
        """
        # Extract major.minor from __version__ (e.g., "0.1.0" -> "0.1")
        version_parts = __version__.split(".")
        version = f"{version_parts[0]}.{version_parts[1]}"

        # Include base image hash (first 8 chars) to invalidate cache on image rebuild
        base_image = self.vm_manager.get_base_image(language)
        img_hash = self.image_hash(base_image)

        base = f"{language.value}-v{version}-{img_hash}"

        if not packages:
            return f"{base}-base"
        return f"{base}-{fingerprint(*sorted(packages))}"

    async def _check_l2_cache(self, cache_key: str) -> Path | None:
        """Check L2 local cache for qcow2 snapshot.

        Validates:
        1. Snapshot file exists
        2. Valid qcow2 format (via qemu-img check), once per process for each
           version of the file (_L2_CHECK_PASSED); a failed check is not kept

        Args:
            cache_key: Snapshot cache key.

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

                if returncode != 0:
                    logger.debug("Invalid qcow2 snapshot, removing", extra={"cache_key": cache_key})
                    await asyncio.to_thread(self._remove_entry, snapshot_path)  # with its digest sidecar; up to GBs
                    return None

            except (OSError, TransientError):  # qemu-img could not start now (spawn_error): a miss, not kept
                return None
            _L2_CHECK_PASSED[str(snapshot_path)] = stamp

        # Update atime for LRU tracking. Keep mtime: image_hash and _L2_CHECK_PASSED key on it.
        with contextlib.suppress(OSError):
            os.utime(snapshot_path, ns=(time.time_ns(), snapshot_path.stat().st_mtime_ns))

        return snapshot_path

    async def _create_snapshot(  # noqa: PLR0912, PLR0915
        self,
        language: Language,
        packages: list[str],
        cache_key: str,
        tenant_id: str,
        task_id: str,
        memory_mb: int,
        *,
        _enospc_retried: bool = False,
    ) -> Path:
        """Create new qcow2 snapshot with packages installed.

        Uses asyncio.wait racing for instant crash detection.

        Workflow:
        1. Create qcow2 with backing file (base image)
        2. Boot VM with snapshot image (ephemeral_vm handles lifecycle)
        3. Install packages via guest agent (with death monitoring)
        4. Shutdown VM (writes committed to snapshot)
        5. Return snapshot path

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
            InternalError: INTERNAL: any other failure
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

                # Step 2: Determine network configuration for snapshot creation
                # ALWAYS enable network during snapshot creation (pip/npm needs it)
                # Restrict to package registries only for security
                if packages:
                    if language == "python":
                        package_domains = list(constants.PYTHON_PACKAGE_DOMAINS)
                    elif language == "javascript":
                        package_domains = list(constants.NPM_PACKAGE_DOMAINS)
                    else:
                        package_domains = []
                else:
                    package_domains = None

                # Step 3-5: Boot VM, install packages, shutdown — ephemeral_vm handles lifecycle
                build_spec = VmSpec(
                    language=Language(language),
                    memory_mb=effective_memory_mb,
                    allow_network=True,
                    allowed_domains=tuple(package_domains) if package_domains is not None else None,
                )
                async with self.vm_manager.ephemeral_vm(
                    build_spec,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    direct_write_target=build_path,
                    snapshot_drive=None,
                    retry_profile=constants.RETRY_BACKGROUND,
                    mem_path=None,
                    mem_path_share=False,
                ) as vm:
                    # Install packages with death monitoring (asyncio.wait)
                    death_task = asyncio.create_task(self._monitor_vm_death(vm, cache_key))
                    install_task = asyncio.create_task(self._install_packages(vm, Language(language), packages))

                    try:
                        done, pending = await asyncio.wait(
                            {death_task, install_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        for task in pending:
                            task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task

                        completed_task = done.pop()
                        if completed_task == death_task:
                            await completed_task  # Propagate VM death exception
                        else:
                            await completed_task  # Propagate install exception if any

                    except Exception:
                        for task in [death_task, install_task]:
                            if not task.done():
                                task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await task
                        raise

                    # Shutdown QEMU process cleanly
                    we_initiated_shutdown = vm.process.returncode is None
                    await cleanup_process(vm.process, "QEMU", cache_key, term_timeout=5.0)

                    # Verify QEMU exited cleanly
                    if we_initiated_shutdown:
                        if vm.process.returncode not in {0, -9, -15}:
                            raise TransientError(
                                f"QEMU exited unexpectedly after terminate (exit code {vm.process.returncode})",
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

                # ephemeral_vm's finally calls destroy_vm (idempotent)
                build_path.replace(snapshot_path)
                success = True

            # Handle disk full (lazy eviction)
            except OSError as e:
                if isinstance(e, TimeoutError):  # an OSError in Python, not a disk error: unexpected here
                    raise build_failed(e) from e
                if e.errno == errno.ENOSPC and not _enospc_retried:
                    await self._evict_oldest_snapshot()
                    retry = True  # after the semaphore is released: the retry takes it again
                else:
                    raise _file_error(e, cache_key=cache_key, language=language, packages=packages) from e

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
                _enospc_retried=True,
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
            TransientError: OVERLAY_FAILED: qemu-img command failed
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
        diag = await vm.collect_diagnostics()
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
            SnapshotError: Package installation failed
            GuestAgentError: Guest agent returned error (permanent)
            PackageInstallTransientError: Transient network error (after all retries exhausted)
            PackageInstallPermanentError: Permanent package error (no retry)
        """
        if not packages:
            return

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
            raise TransientError(
                f"Package installation timeout after {total_timeout}s",
                code=ErrorCode.PACKAGE_DOWNLOAD,
                context={"vm_id": vm.vm_id, "language": language, "packages": packages},
                hint=None,
                instance_id=None,
            ) from e

        # An explicit list, not SandboxError: GUEST_OUTCOME_UNKNOWN (VERIFY) must not
        # escape Scheduler.new for an install that ran no user code.
        except (InvalidInputError, TransientError, InternalError):
            raise

        except Exception as e:
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
                with compressed_path.open("rb") as src, partial_path.open("wb") as dst:
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

    async def _upload_to_s3(self, cache_key: str, snapshot_path: Path) -> None:
        """Upload compressed snapshot to S3 (async, fire-and-forget).

        Bounded by upload_semaphore to prevent:
        - Network saturation
        - Memory exhaustion from compression buffers
        - S3 rate limiting (unlikely but possible)

        Args:
            cache_key: Snapshot cache key
            snapshot_path: Local qcow2 snapshot path
        """
        if not self.settings.s3_bucket:
            return  # S3 backup disabled
        compressed_path = _part_path(snapshot_path.with_suffix(".qcow2.zst"))

        # Acquire semaphore to limit concurrent uploads
        async with self._upload_semaphore:
            try:
                # Compress with zstd (level 3 for speed, run in thread pool to avoid blocking)
                chunk_size = 64 * 1024  # 64KB chunks for streaming

                def _compress() -> str:
                    # Digest from the same read as the compressed bytes, so the
                    # metadata always matches the object even if L2 is evicted.
                    compressor = zstd.ZstdCompressor(level=3)
                    hasher = hashlib.sha256()
                    with Path(snapshot_path).open("rb") as src, Path(compressed_path).open("wb") as dst:
                        while True:
                            chunk = src.read(chunk_size)
                            if not chunk:
                                break
                            hasher.update(chunk)
                            compressed = compressor.compress(chunk)
                            if compressed:
                                dst.write(compressed)
                        # Flush remaining data
                        final = compressor.flush()
                        if final:
                            dst.write(final)
                    return hasher.hexdigest()

                snapshot_sha256 = await asyncio.to_thread(_compress)

                async with await self._get_s3_client() as s3:  # type: ignore[union-attr]
                    # Upload compressed qcow2; the digest of the uncompressed
                    # bytes lets downloads verify the content end to end.
                    s3_key = self._s3_key(snapshot_path.with_suffix(".qcow2.zst").name)
                    await s3.upload_file(  # type: ignore[union-attr]
                        str(compressed_path),
                        self.settings.s3_bucket,
                        s3_key,
                        ExtraArgs={
                            "Tagging": f"ttl_days={self.settings.snapshot_cache_ttl_days}",
                            "Metadata": {"sha256": snapshot_sha256},
                        },
                    )

                # Cleanup compressed file (up to GBs)
                await asyncio.to_thread(compressed_path.unlink)

            except (OSError, RuntimeError, ConnectionError, Exception) as e:  # noqa: BLE001 - Fire-and-forget S3 upload
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
            aioboto3 = require_aioboto3()
            self._s3_session = aioboto3.Session()

        return self._s3_session.client(  # type: ignore[no-any-return]
            "s3",
            region_name=self.settings.s3_region,
            endpoint_url=self.settings.s3_endpoint_url,
        )

    async def stop(self) -> None:
        """Stop DiskSnapshotManager and wait for background upload tasks to complete."""
        await super().stop()

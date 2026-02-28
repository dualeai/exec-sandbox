"""L1 memory snapshot cache manager.

Captures full VM state (CPU + RAM + device state) after REPL warm-up and
package import. On restore, the VM resumes at the exact instruction — ~100ms
regardless of packages.

Cache structure:
    {cache_dir}/{key}.vmstate       # QEMU migration stream (sparse, ~50-140MB on disk)
    {cache_dir}/{key}.vmstate.meta  # JSON metadata sidecar
    {cache_dir}/{key}.vmstate.lock  # flock for cross-process deduplication

Cache key includes ALL parameters that affect VM state:
    crc64(language, sorted_packages, exec_sandbox_version, img_hash,
          qemu_version, arch, accel_type, memory_mb, cpu_cores,
          allow_network, format_version, kernel_hash, initramfs_hash)

Format: 'l1-{language}-v{major.minor}-{16char_hash}'
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
from typing import TYPE_CHECKING, ClassVar

from exec_sandbox import __version__, constants
from exec_sandbox._logging import get_logger
from exec_sandbox.base_cache_manager import BaseCacheManager
from exec_sandbox.guest_agent_protocol import WarmReplAckMessage, WarmReplRequest
from exec_sandbox.hash_utils import crc64
from exec_sandbox.migration_client import MigrationClient
from exec_sandbox.permission_utils import get_expected_socket_uid, grant_qemu_vm_file_access
from exec_sandbox.platform_utils import HostArch, detect_host_arch
from exec_sandbox.system_probes import detect_accel_type, probe_qemu_version

if TYPE_CHECKING:
    from pathlib import Path

    from exec_sandbox.disk_snapshot_manager import DiskSnapshotManager
    from exec_sandbox.models import Language
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.settings import Settings
    from exec_sandbox.vm_manager import VmManager

logger = get_logger(__name__)


class MemorySnapshotManager(BaseCacheManager):
    """L1 memory snapshot cache. Manages vmstate files + metadata sidecars.

    Uses file-based flock for cross-process deduplication of saves.
    Composable API — separate check_cache() + save_snapshot() +
    schedule_background_save().
    """

    _cache_ext: ClassVar[str] = ".vmstate"
    _sidecar_exts: ClassVar[tuple[str, ...]] = (".vmstate.meta",)

    def __init__(
        self,
        settings: Settings,
        vm_manager: VmManager,
        snapshot_manager: DiskSnapshotManager,
    ):
        super().__init__(settings.memory_snapshot_cache_dir)
        self.settings = settings
        self.vm_manager = vm_manager
        self.snapshot_manager = snapshot_manager
        # In-process deduplication for background saves.
        # flock works cross-process but not intra-process on macOS (BSD semantics).
        self._in_flight_saves: dict[str, asyncio.Task[None]] = {}

    def _cache_paths(self, key: str) -> tuple[Path, Path]:
        """Return (vmstate_path, meta_path) for a given cache key."""
        return (
            self.cache_dir / f"{key}.vmstate",
            self.cache_dir / f"{key}.vmstate.meta",
        )

    async def compute_cache_key(
        self,
        language: Language,
        packages: list[str],
        memory_mb: int,
        cpu_cores: int = constants.DEFAULT_VM_CPU_CORES,
        allow_network: bool = False,
    ) -> str:
        """Compute L1 cache key. Includes ALL parameters that affect VM state.

        memory_mb IS in L1 key (unlike L2) because the migration stream
        encodes physical address layout. qemu_version and accel_type are included
        because different QEMU versions produce incompatible migration streams and
        different accelerators (KVM/HVF/TCG) have different CPU state.
        allow_network is included because it changes device topology (-nic none
        vs -device virtio-net).
        """
        version_parts = __version__.split(".")
        version = f"{version_parts[0]}.{version_parts[1]}"

        # Collect all state-affecting parameters
        arch = detect_host_arch()
        base_image = self.vm_manager.get_base_image(language)
        img_hash = self.image_hash(base_image)
        qemu_version = await probe_qemu_version()
        accel_type = await detect_accel_type(force_emulation=self.settings.force_emulation)

        parts = [
            str(language.value),
            ",".join(sorted(packages)),
            str(__version__),
            img_hash,
            ".".join(str(v) for v in qemu_version) if qemu_version else "unknown",
            arch.name,
            accel_type.value,
            str(memory_mb),
            str(cpu_cores),
            "net" if allow_network else "nonet",
            f"fmt{constants.MEMORY_SNAPSHOT_FORMAT_VERSION}",
        ]

        # Add kernel + initramfs hashes
        arch_suffix = "aarch64" if arch == HostArch.AARCH64 else "x86_64"
        kernel_path = self.settings.kernel_path / f"vmlinuz-{arch_suffix}"
        initramfs_path = self.settings.kernel_path / f"initramfs-{arch_suffix}"
        parts.append(self.image_hash(kernel_path))
        parts.append(self.image_hash(initramfs_path))

        fingerprint = "|".join(parts)
        key_hash = crc64(fingerprint)
        return f"l1-{language.value}-v{version}-{key_hash}"

    async def check_cache(
        self,
        language: Language,
        packages: list[str],
        memory_mb: int,
        cpu_cores: int = constants.DEFAULT_VM_CPU_CORES,
        allow_network: bool = False,
    ) -> Path | None:
        """Check L1 cache. Returns vmstate path or None.

        Validates: file exists, non-zero size, metadata sidecar parses.
        Updates atime for LRU tracking.
        """
        key = await self.compute_cache_key(language, packages, memory_mb, cpu_cores, allow_network)
        vmstate_path, meta_path = self._cache_paths(key)

        if not vmstate_path.exists() or not meta_path.exists():
            return None

        # Validate non-zero size
        try:
            stat = vmstate_path.stat()
            if stat.st_size == 0:
                logger.warning("L1 cache entry has zero size, removing", extra={"key": key})
                vmstate_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None
        except OSError:
            return None

        # Validate metadata sidecar parses and verify vmstate integrity
        try:
            meta_text = meta_path.read_text()
            meta = json.loads(meta_text)
        except (OSError, json.JSONDecodeError):
            logger.warning("L1 cache metadata invalid, removing", extra={"key": key})
            vmstate_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

        # Verify vmstate SHA-256 integrity (if hash present in metadata)
        expected_sha256: str | None = None
        if isinstance(meta, dict):
            val = meta.get("vmstate_sha256")  # type: ignore[union-attr]
            if isinstance(val, str):
                expected_sha256 = val
        if expected_sha256:
            sha256 = hashlib.sha256()
            try:
                with vmstate_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        sha256.update(chunk)
                if sha256.hexdigest() != expected_sha256:
                    logger.warning("L1 cache vmstate integrity check failed, removing", extra={"key": key})
                    vmstate_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    return None
            except OSError:
                logger.warning("L1 cache vmstate unreadable during integrity check, removing", extra={"key": key})
                vmstate_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None

        # Update atime for LRU tracking
        vmstate_path.touch(exist_ok=True)

        logger.debug("L1 cache hit", extra={"key": key, "size_mb": stat.st_size // (1024 * 1024)})
        return vmstate_path

    async def save_snapshot(
        self,
        vm: QemuVM,
        language: Language,
        packages: list[str],
        memory_mb: int,
        cpu_cores: int = constants.DEFAULT_VM_CPU_CORES,
        allow_network: bool = False,
    ) -> Path | None:
        """Save L1 from a running warmed VM. TERMINATES the VM.

        Acquires file lock, then delegates to _do_save().
        Returns vmstate path, or None on failure / lock contention.
        After this call, the VM is dead (QEMU process exited).
        Caller is responsible for cleaning up the dead VM.
        """
        key = await self.compute_cache_key(language, packages, memory_mb, cpu_cores, allow_network)
        vmstate_path, meta_path = self._cache_paths(key)

        try:
            async with self._save_lock(key) as should_save:
                if not should_save:
                    logger.debug("L1 cache already exists (race avoided)", extra={"key": key})
                    return vmstate_path
                return await self._do_save(
                    vm,
                    key,
                    vmstate_path,
                    meta_path,
                    language,
                    packages,
                    memory_mb,
                    cpu_cores,
                    allow_network=allow_network,
                )
        except BlockingIOError:
            logger.debug("L1 save already in progress by another process", extra={"key": key})
            return None

    async def _do_save(
        self,
        vm: QemuVM,
        key: str,
        vmstate_path: Path,
        meta_path: Path,
        language: Language,
        packages: list[str],
        memory_mb: int,
        cpu_cores: int,
        *,
        allow_network: bool = False,
    ) -> Path | None:
        """Raw save — caller must hold the file lock.

        Sequence:
        1. Save VM state via MigrationClient to tmp file (terminates QEMU).
        2. Write JSON metadata sidecar.
        3. Atomic rename tmp → final (commit point).
        On ENOSPC: evict oldest entry, clean up, return None.
        On any error (including CancelledError): clean up partial files.
        """
        tmp_path = vmstate_path.parent / f"{vmstate_path.name}.tmp"
        try:
            qemu_version = await probe_qemu_version()
            expected_uid = get_expected_socket_uid(vm.use_qemu_vm_user)

            # If QEMU runs as qemu-vm, pre-create tmp file and grant write
            # access so the migrate command can write the vmstate.
            if vm.use_qemu_vm_user:
                tmp_path.touch()
                await grant_qemu_vm_file_access(tmp_path, writable=True)

            async with MigrationClient(vm.qmp_socket, expected_uid) as client:
                await client.save_snapshot(tmp_path, qemu_version=qemu_version)

            # Compute SHA-256 of vmstate for integrity verification on restore.
            # Runs at memory bandwidth speed (~GB/s), negligible vs migration I/O.
            sha256 = hashlib.sha256()
            with tmp_path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):  # 1MB chunks
                    sha256.update(chunk)
            vmstate_sha256 = sha256.hexdigest()

            # Write metadata sidecar before atomic rename
            arch = detect_host_arch()
            accel_type = await detect_accel_type(force_emulation=self.settings.force_emulation)
            meta = {
                "qemu_version": list(qemu_version) if qemu_version else None,
                "arch": arch.name,
                "accel": accel_type.value,
                "memory_mb": memory_mb,
                "cpu_cores": cpu_cores,
                "language": language.value,
                "packages": sorted(packages),
                "allow_network": allow_network,
                "exec_sandbox_version": __version__,
                "vmstate_sha256": vmstate_sha256,
                "vmstate_size": tmp_path.stat().st_size,
            }
            meta_path.write_text(json.dumps(meta))

            # Atomic commit: rename is atomic on POSIX
            tmp_path.rename(vmstate_path)

            size_mb = vmstate_path.stat().st_size / (1024 * 1024)
            logger.info(
                "L1 snapshot saved",
                extra={"key": key, "size_mb": f"{size_mb:.1f}", "language": language.value},
            )
            return vmstate_path

        except OSError as e:
            if e.errno == errno.ENOSPC:
                # Disk full — evict oldest and retry once
                tmp_path.unlink(missing_ok=True)
                vmstate_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                await self._evict_oldest_snapshot()
                logger.info("L1 evicted oldest on ENOSPC, will retry on next request", extra={"key": key})
                return None
            logger.warning(
                "L1 snapshot save failed (OS error)",
                extra={"key": key, "error": str(e)},
                exc_info=True,
            )
            tmp_path.unlink(missing_ok=True)
            vmstate_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
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
            tmp_path.unlink(missing_ok=True)
            vmstate_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            if isinstance(e, Exception):
                return None
            raise

    async def schedule_background_save(
        self,
        language: Language,
        packages: list[str],
        memory_mb: int,
        cpu_cores: int = constants.DEFAULT_VM_CPU_CORES,
        snapshot_drive: Path | None = None,
        allow_network: bool = False,
    ) -> asyncio.Task[None] | None:
        """Schedule background L1 save: boot sacrificial VM, warm REPL, save, kill.

        Returns the background task handle so callers can optionally await or
        monitor completion. Returns None if cache already exists (no-op).

        Used by scheduler after first cold boot with packages.
        File lock prevents duplicate work — concurrent callers that try to save
        the same key will get BlockingIOError and skip.
        """
        existing = await self.check_cache(
            language, packages, memory_mb, cpu_cores=cpu_cores, allow_network=allow_network
        )
        if existing:
            return None

        key = await self.compute_cache_key(
            language, packages, memory_mb, cpu_cores=cpu_cores, allow_network=allow_network
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
                async with self._save_lock(key) as should_save:
                    if not should_save:
                        logger.debug("L1 background save skipped (already exists)", extra={"key": key})
                        return

                    # Boot sacrificial VM — ephemeral_vm handles lifecycle
                    async with self.vm_manager.ephemeral_vm(
                        language=language,
                        tenant_id=constants.L1_SAVE_TENANT_ID,
                        task_id=f"l1-save-{language.value}",
                        memory_mb=memory_mb,
                        allow_network=allow_network,
                        snapshot_drive=snapshot_drive,
                        retry_profile=constants.RETRY_BACKGROUND,
                    ) as vm:
                        # Warm REPL (Python takes 10-15s on HVF, needs dedicated timeout)
                        response = await vm.channel.send_request(
                            WarmReplRequest(language=language),
                            timeout=constants.WARM_REPL_TIMEOUT_SECONDS,
                        )
                        if not (isinstance(response, WarmReplAckMessage) and response.status == "ok"):
                            logger.warning("L1 background save: REPL warm failed", extra={"response": str(response)})
                            return

                        # Save L1 (terminates VM) — call _do_save directly (we already hold the lock)
                        # After _do_save, QEMU process is dead; ephemeral_vm's destroy_vm cleans up resources.
                        vmstate_path, meta_path = self._cache_paths(key)
                        result = await self._do_save(
                            vm,
                            key,
                            vmstate_path,
                            meta_path,
                            language,
                            packages,
                            memory_mb,
                            cpu_cores,
                            allow_network=allow_network,
                        )

                    elapsed_ms = round((asyncio.get_running_loop().time() - start_time) * 1000)
                    if result:
                        logger.info(
                            "L1 background save completed",
                            extra={"key": key, "language": language.value, "elapsed_ms": elapsed_ms},
                        )
                    else:
                        logger.warning(
                            "L1 background save produced no result",
                            extra={"key": key, "language": language.value, "elapsed_ms": elapsed_ms},
                        )

            except BlockingIOError:
                logger.debug("L1 background save already in-flight", extra={"key": key})
            except Exception as e:  # noqa: BLE001 - background task, must not propagate
                elapsed_ms = round((asyncio.get_running_loop().time() - start_time) * 1000)
                logger.warning(
                    "L1 background save failed",
                    extra={"language": language.value, "error": str(e), "elapsed_ms": elapsed_ms},
                    exc_info=True,
                )

        task = asyncio.create_task(_background_save())
        self._in_flight_saves[key] = task
        task.add_done_callback(lambda _t: self._in_flight_saves.pop(key, None))
        self._track_task(task, name=f"l1-save-{language.value}-{key}")
        return task

    async def get_l2_for_l1(self, language: Language, packages: list[str]) -> Path | None:
        """Get the L2 disk snapshot path for an L1 restore.

        For packages=[], returns None (base image only).
        For packages=["pandas"], returns L2 qcow2 path via snapshot_manager.
        """
        if not packages:
            return None

        return await self.snapshot_manager.check_cache(
            language=language,
            packages=packages,
        )

    async def stop(self) -> None:
        """Wait for background save tasks."""
        await super().stop()

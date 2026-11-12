"""Tests for MemorySnapshotManager (L1 memory snapshot cache).

De-mocked: uses real Settings + real VmManager (not started) instead of
MagicMock. Only DiskSnapshotManager (L2 boundary) and system probes
(QEMU binary / CPU features) remain mocked.

Includes:
- Normal case tests for all public APIs
- Edge cases (corruption, race conditions, boundary values)
- Out-of-bound/stress scenarios (concurrent access, error recovery)
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InternalError,
    RouteError,
    SandboxError,
    TransientError,
)
from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager, _vmstate_digest
from exec_sandbox.models import Language
from exec_sandbox.vm_spec import VmSpec


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Fresh temporary cache directory."""
    return tmp_path / "l1-cache"


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    """Dummy image files so real VmManager.get_base_image() glob works."""
    d = tmp_path / "images"
    d.mkdir()
    (d / "python-3.14-base-aarch64.qcow2").touch()
    (d / "node-23-base-aarch64.qcow2").touch()
    (d / "raw-base-aarch64.qcow2").touch()
    kernels = d / "kernels"
    kernels.mkdir()
    for arch in ("aarch64", "x86_64"):
        (kernels / f"vmlinuz-{arch}").write_bytes(b"fake-kernel")
        (kernels / f"initramfs-{arch}").write_bytes(b"fake-initramfs")
    return d


@pytest.fixture
def manager(cache_dir: Path, images_dir: Path) -> MemorySnapshotManager:
    """MemorySnapshotManager with real Settings + real VmManager (not started)."""
    from exec_sandbox.settings import Settings
    from exec_sandbox.vm_manager import VmManager

    settings = Settings(
        memory_snapshot_cache_dir=cache_dir,
        base_images_dir=images_dir,
        kernel_path=images_dir / "kernels",
    )
    vm_manager = VmManager(settings)  # Not started — safe for unit tests
    return MemorySnapshotManager(settings, vm_manager)


async def _check_spec(manager: MemorySnapshotManager, spec: VmSpec) -> Path | None:
    """check_cache of the key of `spec` without an L2 layer."""
    return await manager.check_cache(await manager.compute_cache_key(spec, l2_path=None))


async def _schedule_save(manager: MemorySnapshotManager, spec: VmSpec) -> asyncio.Task[None] | None:
    """schedule_background_save of `spec` without an L2 layer."""
    key = await manager.compute_cache_key(spec, l2_path=None)
    return await manager.schedule_background_save(spec, key=key, snapshot_drive=None)


@pytest.fixture
def mock_system_probes() -> Generator[dict[str, Any], None, None]:
    """Patch system probes (arch, QEMU version, accel) with standard defaults.

    Returns dict with mock handles so tests can override return values if needed.
    """
    from exec_sandbox.platform_utils import HostArch
    from exec_sandbox.system_probes import AccelType

    with (
        patch("exec_sandbox.memory_snapshot_manager.detect_host_arch") as mock_arch,
        patch("exec_sandbox.memory_snapshot_manager.probe_qemu_version", new_callable=AsyncMock) as mock_qemu,
        patch("exec_sandbox.memory_snapshot_manager.detect_accel_type", new_callable=AsyncMock) as mock_accel,
    ):
        mock_arch.return_value = HostArch.AARCH64
        mock_qemu.return_value = (11, 1, 3)  # at or above the migration floor: saves allowed
        mock_accel.return_value = AccelType.HVF
        yield {"arch": mock_arch, "qemu_version": mock_qemu, "accel": mock_accel}


class TestComputeCacheKey:
    """Tests for cache key computation."""

    async def _compute_key(
        self,
        manager: MemorySnapshotManager,
        *,
        language: Language = Language.PYTHON,
        packages: list[str] | None = None,
        memory_mb: int = 512,
        qemu_version: tuple[int, int, int] | None = (9, 2, 0),
        accel: str = "HVF",
        allow_network: bool = False,
        l2_path: Path | None = None,
    ) -> str:
        """Helper to compute cache key with standard mocks."""
        from exec_sandbox.platform_utils import HostArch
        from exec_sandbox.system_probes import AccelType

        accel_map = {"HVF": AccelType.HVF, "KVM": AccelType.KVM, "TCG": AccelType.TCG}

        with (
            patch("exec_sandbox.memory_snapshot_manager.detect_host_arch") as mock_arch,
            patch("exec_sandbox.memory_snapshot_manager.probe_qemu_version", new_callable=AsyncMock) as mock_qemu,
            patch("exec_sandbox.memory_snapshot_manager.detect_accel_type", new_callable=AsyncMock) as mock_accel,
        ):
            mock_arch.return_value = HostArch.AARCH64
            mock_qemu.return_value = qemu_version
            mock_accel.return_value = accel_map[accel]

            return await manager.compute_cache_key(
                VmSpec(
                    language=language, packages=tuple(packages or []), memory_mb=memory_mb, allow_network=allow_network
                ),
                l2_path=l2_path,
            )

    async def test_l2_content_pins_key(self, manager: MemorySnapshotManager, tmp_path: Path) -> None:
        """Same package specs, different L2 bytes (a rebuild) -> different L1 key."""
        l2_a = tmp_path / "l2-a.qcow2"
        l2_a.write_bytes(b"layer-a")
        l2_b = tmp_path / "l2-b.qcow2"
        l2_b.write_bytes(b"layer-b")
        key_a = await self._compute_key(manager, packages=["pandas==2.2.3"], l2_path=l2_a)
        key_b = await self._compute_key(manager, packages=["pandas==2.2.3"], l2_path=l2_b)
        assert key_a != key_b

    async def test_image_digests_in_one_worker_thread_job(
        self, manager: MemorySnapshotManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The base image, kernel, initramfs and L2 digests take one worker-thread job, not
        one hop each (a hop costs 30 times a stat of a known digest)."""
        l2 = tmp_path / "l2.qcow2"
        l2.write_bytes(b"layer")
        jobs: list[str] = []
        to_thread = asyncio.to_thread

        async def counted(func: Any, /, *args: Any, **kwargs: Any) -> Any:
            jobs.append(getattr(func, "__name__", repr(func)))
            return await to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", counted)
        await self._compute_key(manager, packages=["pandas==2.2.3"], l2_path=l2)
        assert len(jobs) == 1, jobs

    async def test_same_inputs_same_key(self, manager: MemorySnapshotManager) -> None:
        """Identical inputs produce identical cache keys."""
        key1 = await self._compute_key(manager)
        key2 = await self._compute_key(manager)
        assert key1 == key2

    async def test_different_language_different_key(self, manager: MemorySnapshotManager) -> None:
        """Different languages produce different keys."""
        key_py = await self._compute_key(manager, language=Language.PYTHON)
        key_js = await self._compute_key(manager, language=Language.JAVASCRIPT)
        assert key_py != key_js

    async def test_different_packages_different_key(self, manager: MemorySnapshotManager) -> None:
        """Different packages produce different keys."""
        key_empty = await self._compute_key(manager, packages=[])
        key_pandas = await self._compute_key(manager, packages=["pandas"])
        assert key_empty != key_pandas

    async def test_different_memory_different_key(self, manager: MemorySnapshotManager) -> None:
        """Different memory_mb produces different keys."""
        key_512 = await self._compute_key(manager, memory_mb=512)
        key_256 = await self._compute_key(manager, memory_mb=256)
        assert key_512 != key_256

    async def test_different_qemu_version_different_key(self, manager: MemorySnapshotManager) -> None:
        """Different QEMU versions produce different keys."""
        key_9 = await self._compute_key(manager, qemu_version=(9, 2, 0))
        key_10 = await self._compute_key(manager, qemu_version=(10, 0, 0))
        assert key_9 != key_10

    async def test_different_accel_different_key(self, manager: MemorySnapshotManager) -> None:
        """Different accelerator types produce different keys."""
        key_hvf = await self._compute_key(manager, accel="HVF")
        key_tcg = await self._compute_key(manager, accel="TCG")
        assert key_hvf != key_tcg

    async def test_different_network_different_key(self, manager: MemorySnapshotManager) -> None:
        """Different allow_network values produce different keys (topology differs)."""
        key_nonet = await self._compute_key(manager, allow_network=False)
        key_net = await self._compute_key(manager, allow_network=True)
        assert key_nonet != key_net

    async def test_key_format(self, manager: MemorySnapshotManager) -> None:
        """Key format: l1-{language}-v{major.minor}-{hash}."""
        key = await self._compute_key(manager, language=Language.PYTHON)
        assert key.startswith("l1-python-v")
        # Hash part is a fixed-length SHA-256 fingerprint
        parts = key.split("-", 3)
        assert len(parts) == 4  # l1, python, v0.0, hash
        assert len(parts[3]) == 64  # full SHA-256 hex digest

    async def test_package_order_insensitive(self, manager: MemorySnapshotManager) -> None:
        """Package order should not affect cache key (sorted internally)."""
        key_ab = await self._compute_key(manager, packages=["a", "b"])
        key_ba = await self._compute_key(manager, packages=["b", "a"])
        assert key_ab == key_ba

    async def test_none_qemu_version(self, manager: MemorySnapshotManager) -> None:
        """None QEMU version should produce stable key with 'unknown'."""
        key = await self._compute_key(manager, qemu_version=None)
        assert key.startswith("l1-python-v")


class TestVmstateDigest:
    """The L1 integrity digest: every non-zero byte, its offset and the size count; holes are not read."""

    _SIZE = 1024 * 1024 * 1024  # APFS reports the holes of a large sparse file only
    _LAYOUT_SIZE = 64 * 1024 * 1024  # APFS reports its holes; cheap to read whole

    @staticmethod
    def _sparse(path: Path, writes: dict[int, bytes], *, size: int) -> Path:
        with path.open("wb") as f:
            f.truncate(size)
            for offset, data in writes.items():
                f.seek(offset)
                f.write(data)
        return path

    @staticmethod
    def _data_ranges(path: Path) -> list[tuple[int, int]]:
        """The (start, end) data ranges the filesystem reports (SEEK_DATA / SEEK_HOLE)."""
        ranges: list[tuple[int, int]] = []
        fd = os.open(path, os.O_RDONLY)
        try:
            size, pos = os.fstat(fd).st_size, 0
            while pos < size:
                try:
                    start = os.lseek(fd, pos, os.SEEK_DATA)
                except OSError:  # ENXIO: no data after pos
                    break
                pos = os.lseek(fd, start, os.SEEK_HOLE)
                ranges.append((start, pos))
        finally:
            os.close(fd)
        return ranges

    def test_hole_layout_does_not_change_it(self, tmp_path: Path) -> None:
        """The same bytes in two hole layouts (zeros written into a hole of one): same digest."""
        sparse = self._sparse(tmp_path / "sparse", {70_000: b"data"}, size=self._LAYOUT_SIZE)
        zeros_written = self._sparse(
            tmp_path / "zeros_written", {70_000: b"data", 32 * 1024 * 1024: bytes(64 * 1024)}, size=self._LAYOUT_SIZE
        )
        # Precondition: without holes both files have one layout and the test proves nothing
        assert self._data_ranges(sparse) != self._data_ranges(zeros_written)
        assert _vmstate_digest(sparse) == _vmstate_digest(zeros_written)

    def test_each_change_changes_it(self, tmp_path: Path) -> None:
        base = {70_000: b"data", 500_000_000: b"tail"}
        digest = _vmstate_digest(self._sparse(tmp_path / "base", base, size=self._SIZE))
        changed = {
            "data byte": ({70_000: b"dAta", 500_000_000: b"tail"}, self._SIZE),
            "byte in a hole": ({**base, 900_000_000: b"x"}, self._SIZE),
            "block moved": ({70_000 + 64 * 1024: b"data", 500_000_000: b"tail"}, self._SIZE),
            "zeros appended": (base, self._SIZE + 1),
        }
        for name, (writes, size) in changed.items():
            assert _vmstate_digest(self._sparse(tmp_path / name, writes, size=size)) != digest, name

    def test_without_seek_data_every_block_is_read_and_the_digest_is_the_same(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A filesystem without SEEK_DATA: the whole file is read, and holes still hash as zero blocks."""
        path = self._sparse(tmp_path / "vmstate", {70_000: b"data"}, size=self._LAYOUT_SIZE)
        # Precondition: the file has holes, so the two paths read different blocks
        assert self._data_ranges(path) != [(0, self._LAYOUT_SIZE)]
        with_ranges = _vmstate_digest(path)

        def no_seek_data(*_args: object, **_kwargs: object) -> list[tuple[int, int]]:
            raise OSError(errno.EINVAL, "SEEK_DATA not supported")

        monkeypatch.setattr("exec_sandbox.memory_snapshot_manager.data_ranges", no_seek_data)
        assert _vmstate_digest(path) == with_ranges

    def test_holes_are_not_read(self, tmp_path: Path) -> None:
        path = self._sparse(tmp_path / "vmstate", {0: b"a", 600_000_000: b"b"}, size=self._SIZE)
        real_pread = os.pread
        read: list[int] = []

        def counting_pread(fd: int, n: int, offset: int) -> bytes:
            data = real_pread(fd, n, offset)
            read.append(len(data))
            return data

        with patch.object(os, "pread", counting_pread):
            _vmstate_digest(path)
        assert sum(read) <= 64 * 64 * 1024  # 2 data blocks, plus what the filesystem rounds into data


class TestCheckCache:
    """Tests for check_cache (L1 cache lookups)."""

    async def _populate_cache(
        self,
        manager: MemorySnapshotManager,
        language: Language = Language.PYTHON,
        packages: list[str] | None = None,
        memory_mb: int = 512,
        data: bytes = b"fake vmstate " * 100,
    ) -> Path:
        """Helper to populate cache with fake data (requires mock_system_probes fixture)."""
        key = await manager.compute_cache_key(
            VmSpec(language=language, packages=tuple(packages or []), memory_mb=memory_mb), l2_path=None
        )
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(data)
        vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
        # Include SHA-256 so integrity check passes
        vmstate_sha256 = _vmstate_digest(vmstate) if data else None
        meta_obj: dict[str, Any] = {"test": True}
        if vmstate_sha256:
            meta_obj["vmstate_sha256"] = vmstate_sha256
        meta.write_text(json.dumps(meta_obj))
        return vmstate

    async def test_cache_miss_no_files(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Cache miss when no files exist."""
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None

    async def test_cache_hit(self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]) -> None:
        """Cache hit returns vmstate path."""
        vmstate = await self._populate_cache(manager)
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is not None
        assert result == vmstate

    async def test_zero_size_cleaned_up(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Zero-size vmstate is cleaned up and returns miss."""
        vmstate = await self._populate_cache(manager, data=b"")
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None
        assert not vmstate.exists()

    async def test_invalid_metadata_cleaned_up(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Invalid metadata causes cleanup and returns miss."""
        vmstate = await self._populate_cache(manager)
        # Corrupt metadata
        meta = vmstate.parent / f"{vmstate.name[: -len('.vmstate')]}.vmstate.meta"
        meta.write_text("not json {{{")

        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None
        assert not vmstate.exists()
        assert not meta.exists()

    async def test_missing_metadata_returns_miss(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Missing metadata file returns miss."""
        vmstate = await self._populate_cache(manager)
        # Delete metadata only
        meta = vmstate.parent / f"{vmstate.name[: -len('.vmstate')]}.vmstate.meta"
        meta.unlink()

        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None

    async def test_sha256_mismatch_cleaned_up(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """vmstate with wrong SHA-256 is cleaned up and returns miss."""
        vmstate = await self._populate_cache(manager)
        # Corrupt vmstate content (metadata still has old SHA-256)
        vmstate.write_bytes(b"corrupted data")

        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None
        assert not vmstate.exists()

    async def test_no_sha256_in_metadata_still_hits(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Legacy metadata without SHA-256 still returns hit (graceful upgrade)."""
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(b"fake vmstate data")
        vmstate.with_suffix(".home").write_bytes(b"home disk")
        meta.write_text(json.dumps({"test": True}))  # No vmstate_sha256 field

        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is not None
        assert result == vmstate


class TestHomeDiskEntry:
    """An L1 entry is vmstate + metadata + RAM template + home disk: all or nothing."""

    async def test_missing_home_disk_is_a_miss_and_entry_removed(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        spec = VmSpec(language=Language.PYTHON, memory_mb=512)
        key = await manager.compute_cache_key(spec, l2_path=None)
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(b"vmstate")
        meta.write_text(json.dumps({"vmstate_sha256": _vmstate_digest(vmstate)}))

        assert await manager.check_cache(key) is None
        assert not vmstate.exists()
        assert not meta.exists()

    async def test_failed_check_removes_every_file_off_the_loop_and_the_vmstate_last(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """An entry that fails its integrity check takes its large sidecars with it (eviction
        lists only *.vmstate), in a worker thread (the .ram file is the RAM size); the miss
        returns once they are gone. No save of a key starts while its vmstate exists
        (_save_lock): the vmstate goes last, so a removal never deletes the sidecars of a
        new save of that key."""
        key = await manager.compute_cache_key(VmSpec(language=Language.PYTHON, memory_mb=512), l2_path=None)
        files = [manager.cache_dir / f"{key}{ext}" for ext in (".vmstate", ".vmstate.meta", ".ram", ".home")]
        for path in files:
            path.write_bytes(b"x")
        files[1].write_text(json.dumps({"vmstate_sha256": "0" * 64}))  # fails the check: removed
        unlinked: list[tuple[str, int]] = []
        real_unlink = Path.unlink

        def spied_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
            if self in files:
                unlinked.append((self.name, threading.get_ident()))
            real_unlink(self, *args, **kwargs)

        with patch.object(Path, "unlink", spied_unlink):
            assert await manager.check_cache(key) is None

        assert not any(path.exists() for path in files)
        assert sorted(name for name, _ in unlinked) == sorted(path.name for path in files)
        assert threading.get_ident() not in {thread for _, thread in unlinked}
        assert unlinked[-1][0] == files[0].name

    async def test_template_entry_without_its_ram_file_is_a_miss_and_entry_removed(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """A template entry's vmstate holds no RAM: without its .ram file it cannot restore."""
        spec = VmSpec(language=Language.PYTHON, memory_mb=512)
        key = await manager.compute_cache_key(spec, l2_path=None)
        vmstate = manager.cache_dir / f"{key}.vmstate"
        vmstate.write_bytes(b"device state")
        vmstate.with_suffix(".home").write_bytes(b"home disk")
        (manager.cache_dir / f"{key}.vmstate.meta").write_text(
            json.dumps({"vmstate_sha256": _vmstate_digest(vmstate), "use_template": True})
        )

        assert await manager.check_cache(key) is None
        assert sorted(p.name for p in manager.cache_dir.iterdir() if p.name.startswith(key)) == []

    async def test_disk_size_in_key(self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]) -> None:
        """A template formats its home disk: another disk size is another template."""
        small = await manager.compute_cache_key(VmSpec(language=Language.PYTHON, disk_mb=256), l2_path=None)
        large = await manager.compute_cache_key(VmSpec(language=Language.PYTHON, disk_mb=1024), l2_path=None)
        assert small != large


@pytest.mark.usefixtures("mock_system_probes")
class TestRestore:
    """restore(): an L1 hit becomes a VM; a miss or a TransientError gives None (the caller cold boots)."""

    _SPEC = VmSpec(language=Language.PYTHON, memory_mb=512)

    @staticmethod
    async def _entry(manager: MemorySnapshotManager) -> tuple[str, list[Path]]:
        """A valid L1 entry of _SPEC (no L2 layer): its key and every file of it."""
        key = await manager.compute_cache_key(TestRestore._SPEC, l2_path=None)
        files = [manager.cache_dir / f"{key}{ext}" for ext in (".vmstate", ".vmstate.meta", ".ram", ".home")]
        files[0].write_bytes(b"vmstate")
        files[1].write_text(json.dumps({"vmstate_sha256": _vmstate_digest(files[0])}))
        files[2].write_bytes(b"ram")
        files[3].write_bytes(b"home")
        return key, files

    @staticmethod
    async def _restore(manager: MemorySnapshotManager, key: str, reservation: object) -> object:
        return await manager.restore(
            TestRestore._SPEC,
            key=key,
            tenant_id="tenant",
            task_id="task",
            snapshot_drive=Path("/l2.qcow2"),
            expose_ports=None,
            reservation=reservation,  # type: ignore[arg-type]
        )

    async def test_hit_restores_the_entry(self, manager: MemorySnapshotManager) -> None:
        key, files = await self._entry(manager)
        vm, reservation = MagicMock(), MagicMock()
        with patch.object(manager.vm_manager, "restore_vm", new=AsyncMock(return_value=vm)) as restore_vm:
            assert await self._restore(manager, key, reservation) is vm
        restore_vm.assert_awaited_once_with(
            self._SPEC,
            tenant_id="tenant",
            task_id="task",
            vmstate_path=files[0],
            base_image=manager.vm_manager.get_base_image("python"),
            snapshot_drive=Path("/l2.qcow2"),
            expose_ports=None,
            reservation=reservation,
        )

    async def test_miss_starts_nothing(self, manager: MemorySnapshotManager) -> None:
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        with patch.object(manager.vm_manager, "restore_vm", new=AsyncMock()) as restore_vm:
            assert await self._restore(manager, key, MagicMock()) is None
        restore_vm.assert_not_awaited()

    @pytest.mark.parametrize(
        ("cause", "evicted"),
        [
            pytest.param(
                RouteError(
                    "QEMU refused",
                    code=ErrorCode.MIGRATION_REJECTED,
                    required={},
                    found={},
                    context=None,
                    hint=None,
                    instance_id=None,
                ),
                True,
                id="rejected",
            ),
            pytest.param(
                ConfigError(
                    "QEMU cannot load", code=ErrorCode.MIGRATION_UNSUPPORTED, context=None, hint=None, instance_id=None
                ),
                True,
                id="unsupported",
            ),
            pytest.param(
                ConfigError(
                    "qemu-vm user missing", code=ErrorCode.HOST_DEPENDENCY, context=None, hint=None, instance_id=None
                ),
                False,
                id="setup",
            ),
            pytest.param(TimeoutError(), False, id="timeout"),
        ],
    )
    async def test_transient_failure_removes_only_what_qemu_refused(
        self, manager: MemorySnapshotManager, cause: BaseException, evicted: bool
    ) -> None:
        """A restore QEMU refused (MIGRATION_REJECTED or MIGRATION_UNSUPPORTED in the cause
        chain) removes the entry; a setup failure or a timeout keeps it (a good entry)."""
        key, files = await self._entry(manager)
        error = TransientError(
            "restore failed", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None
        )
        error.__cause__ = cause
        with patch.object(manager.vm_manager, "restore_vm", new=AsyncMock(side_effect=error)):
            assert await self._restore(manager, key, MagicMock()) is None
        assert all(not f.exists() for f in files) if evicted else all(f.exists() for f in files)

    async def test_silent_guest_removes_the_entry(self, manager: MemorySnapshotManager) -> None:
        """A guest that never answers after its state loaded: the request cold boots and
        the entry goes (else every later hit waits for the same silence)."""
        key, files = await self._entry(manager)
        error = TransientError(
            "guest did not answer", code=ErrorCode.GUEST_SILENT_AFTER_RESTORE, context=None, hint=None, instance_id=None
        )
        with patch.object(manager.vm_manager, "restore_vm", new=AsyncMock(side_effect=error)):
            assert await self._restore(manager, key, MagicMock()) is None
        assert not any(f.exists() for f in files)

    async def test_rejection_found_through_context_chain(self, manager: MemorySnapshotManager) -> None:
        """QEMU died while refusing the stream: the error is raised without `from`."""
        key, files = await self._entry(manager)
        try:
            try:
                raise RouteError(
                    "QEMU refused",
                    code=ErrorCode.MIGRATION_REJECTED,
                    required={},
                    found={},
                    context=None,
                    hint=None,
                    instance_id=None,
                )
            except RouteError:
                raise TransientError(  # noqa: B904 - as the code under test: no `from`
                    "QEMU process died", code=ErrorCode.QEMU_CRASHED, context=None, hint=None, instance_id=None
                )
        except SandboxError as caught:
            error = caught
        with patch.object(manager.vm_manager, "restore_vm", new=AsyncMock(side_effect=error)):
            assert await self._restore(manager, key, MagicMock()) is None
        assert not any(f.exists() for f in files)

    async def test_unconfirmed_cleanup_raises_cleanup_unconfirmed_and_keeps_the_entry(
        self, manager: MemorySnapshotManager
    ) -> None:
        """Cleanup of the failed restore not confirmed: no cold boot may reuse the reservation."""
        key, files = await self._entry(manager)
        error = InternalError(
            "Restore failure cleanup was not confirmed",
            code=ErrorCode.CLEANUP_UNCONFIRMED,
            context=None,
            hint=None,
            instance_id=None,
        )
        with (
            patch.object(manager.vm_manager, "restore_vm", new=AsyncMock(side_effect=error)),
            pytest.raises(
                InternalError, check=lambda e: e.code is ErrorCode.CLEANUP_UNCONFIRMED, match="not confirmed"
            ),
        ):
            await self._restore(manager, key, MagicMock())
        assert all(f.exists() for f in files)


class TestSaveLock:
    """Tests for _save_lock (file-based flock deduplication)."""

    async def test_lock_yields_true_when_no_file(self, manager: MemorySnapshotManager) -> None:
        """Lock yields True when vmstate file does not exist."""
        async with manager._save_lock("test-key") as should_save:
            assert should_save is True

    async def test_lock_yields_false_when_file_exists(self, manager: MemorySnapshotManager) -> None:
        """Lock yields False when vmstate file already exists."""
        vmstate = manager.cache_dir / "test-key.vmstate"
        vmstate.write_bytes(b"data")
        async with manager._save_lock("test-key") as should_save:
            assert should_save is False

    async def test_lock_yields_true_when_file_empty(self, manager: MemorySnapshotManager) -> None:
        """Lock yields True when vmstate file exists but is empty (partial write)."""
        vmstate = manager.cache_dir / "test-key.vmstate"
        vmstate.write_bytes(b"")
        async with manager._save_lock("test-key") as should_save:
            assert should_save is True

    async def test_concurrent_lock_raises_blocking(self, manager: MemorySnapshotManager) -> None:
        """Second concurrent lock on same key raises BlockingIOError."""
        async with manager._save_lock("test-key"):
            with pytest.raises(BlockingIOError):
                async with manager._save_lock("test-key"):
                    pass  # Should not reach here

    async def test_lock_makes_a_removed_cache_folder_again(self, manager: MemorySnapshotManager) -> None:
        """A cache folder removed while the worker runs (a tmp cleaner) is made again: the
        save goes on, instead of a FileNotFoundError on every save until a restart."""
        shutil.rmtree(manager.cache_dir)
        async with manager._save_lock("test-key") as should_save:
            assert should_save is True
        assert manager.cache_dir.is_dir()

    async def test_lock_released_after_context(self, manager: MemorySnapshotManager) -> None:
        """Lock is released when context manager exits — re-acquire succeeds."""
        async with manager._save_lock("test-key"):
            pass
        # Should succeed (lock released)
        async with manager._save_lock("test-key") as should_save:
            assert should_save is True


class TestEvictOldestSnapshot:
    """Tests for _evict_oldest_snapshot (LRU eviction)."""

    async def test_evict_removes_oldest(self, manager: MemorySnapshotManager) -> None:
        """Eviction removes the oldest snapshot by atime."""
        import os
        import time

        # Create 3 snapshots with different atimes
        for i, name in enumerate(["old", "mid", "new"]):
            vmstate = manager.cache_dir / f"{name}.vmstate"
            meta = manager.cache_dir / f"{name}.vmstate.meta"
            vmstate.write_bytes(b"data")
            meta.write_text("{}")
            # Set atime to different times (older = smaller)
            atime = time.time() - (100 - i * 10)
            os.utime(vmstate, (atime,) * 2)  # mtime too: relatime resets an atime older than mtime

        await manager._evict_oldest_snapshot()

        assert not (manager.cache_dir / "old.vmstate").exists()
        assert not (manager.cache_dir / "old.vmstate.meta").exists()
        assert (manager.cache_dir / "mid.vmstate").exists()
        assert (manager.cache_dir / "new.vmstate").exists()

    async def test_evict_no_files(self, manager: MemorySnapshotManager) -> None:
        """Eviction with no files is a no-op."""
        await manager._evict_oldest_snapshot()  # Should not raise

    async def test_evict_skips_locked_entry(self, manager: MemorySnapshotManager) -> None:
        """Eviction skips entries with an active flock and evicts the next oldest."""
        import fcntl
        import os
        import time

        # Create 2 snapshots: "locked" (oldest) and "unlocked" (newer)
        for i, name in enumerate(["locked", "unlocked"]):
            vmstate = manager.cache_dir / f"{name}.vmstate"
            meta = manager.cache_dir / f"{name}.vmstate.meta"
            vmstate.write_bytes(b"data")
            meta.write_text("{}")
            atime = time.time() - (100 - i * 10)
            os.utime(vmstate, (atime,) * 2)  # mtime too: relatime resets an atime older than mtime

        # Hold flock on "locked" entry
        lock_path = manager.cache_dir / "locked.vmstate.lock"
        fd = lock_path.open("w")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            await manager._evict_oldest_snapshot()

            # "locked" should be skipped, "unlocked" evicted instead
            assert (manager.cache_dir / "locked.vmstate").exists()
            assert not (manager.cache_dir / "unlocked.vmstate").exists()
        finally:
            fd.close()


class TestStop:
    """Tests for stop (background task cleanup)."""

    async def test_stop_waits_for_tasks(self, manager: MemorySnapshotManager) -> None:
        """Stop should wait for all background tasks."""
        completed = []

        async def slow_task() -> None:
            await asyncio.sleep(0.01)
            completed.append(True)

        task = asyncio.create_task(slow_task())
        manager._track_task(task, name="test-slow-task")

        await manager.stop()
        assert len(completed) == 1

    async def test_stop_no_tasks(self, manager: MemorySnapshotManager) -> None:
        """Stop with no tasks is a no-op."""
        await manager.stop()  # Should not raise

    async def test_context_manager(self, manager: MemorySnapshotManager) -> None:
        """Async context manager calls stop on exit."""
        async with manager:
            pass  # Should not raise


class TestScheduleBackgroundSave:
    """Tests for schedule_background_save (fire-and-forget L1 creation)."""

    async def test_skips_if_cache_exists(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Should be no-op if L1 cache already exists."""
        # Populate cache
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        data = b"fake " * 100
        vmstate.write_bytes(data)
        vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
        meta.write_text(json.dumps({"test": True, "vmstate_sha256": _vmstate_digest(vmstate)}))

        # Should not schedule anything
        await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))

        assert len(manager._background_tasks) == 0

    @pytest.mark.skipif(
        __import__("sys").platform == "darwin",
        reason="macOS flock is per-process (not per-fd) — concurrent tasks in same process don't block each other",
    )
    async def test_flock_deduplicates_concurrent_saves(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Concurrent background saves for same key — flock ensures only 1 VM boots."""
        # Mock create_vm to block until we release it
        gate = asyncio.Event()
        call_count = 0

        async def slow_create_vm(*_args: object, **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            await gate.wait()
            return MagicMock()

        with patch.object(manager.vm_manager, "create_vm", new=AsyncMock(side_effect=slow_create_vm)):
            # Schedule 5 concurrent saves for same key
            await asyncio.gather(
                _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)),
                _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)),
                _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)),
                _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)),
                _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)),
            )

            # Give background tasks a tick to hit the flock
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            # Only 1 create_vm should be called (others blocked by flock → skip)
            assert call_count == 1

            # Clean up
            gate.set()
            for task in list(manager._background_tasks):
                task.cancel()
            await asyncio.gather(*manager._background_tasks, return_exceptions=True)
            manager._background_tasks.clear()

    @pytest.mark.parametrize(
        ("qemu_version", "scheduled"),
        [
            pytest.param(None, False, id="unknown"),
            pytest.param((10, 99, 99), False, id="just-below"),
            pytest.param((11, 0, 0), True, id="floor"),
        ],
    )
    async def test_qemu_below_the_migration_floor_boots_no_vm(
        self,
        manager: MemorySnapshotManager,
        mock_system_probes: dict[str, Any],
        qemu_version: tuple[int, int, int] | None,
        scheduled: bool,
    ) -> None:
        """Below QEMU 11.0 every save fails: no sacrificial VM is booted for it."""
        mock_system_probes["qemu_version"].return_value = qemu_version
        create_vm = AsyncMock(side_effect=RuntimeError("boot failed"))
        with patch.object(manager.vm_manager, "create_vm", new=create_vm):
            task = await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

        assert (task is not None, create_vm.await_count) == (scheduled, int(scheduled))

    async def test_lock_released_after_failure(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """After background save fails, flock is released — retry can proceed."""
        # First attempt: create_vm fails
        with patch.object(manager.vm_manager, "create_vm", new=AsyncMock(side_effect=RuntimeError("boot failed"))):
            await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
            await asyncio.gather(*manager._background_tasks, return_exceptions=True)
            manager._background_tasks.clear()

        # Lock should be released — second attempt can acquire it
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        async with manager._save_lock(key) as should_save:
            assert should_save is True  # Lock free, no vmstate file


# ============================================================================
# Torture Tests — Cache Key Edge Cases
# ============================================================================


class TestComputeCacheKeyEdgeCases:
    """Edge cases and boundary values for cache key computation."""

    async def _compute_key(
        self,
        manager: MemorySnapshotManager,
        *,
        language: Language = Language.PYTHON,
        packages: list[str] | None = None,
        memory_mb: int = 512,
        cpu_cores: int = 1,
        qemu_version: tuple[int, int, int] | None = (9, 2, 0),
        accel: str = "HVF",
        allow_network: bool = False,
    ) -> str:
        from exec_sandbox.platform_utils import HostArch
        from exec_sandbox.system_probes import AccelType

        accel_map = {"HVF": AccelType.HVF, "KVM": AccelType.KVM, "TCG": AccelType.TCG}
        with (
            patch("exec_sandbox.memory_snapshot_manager.detect_host_arch") as mock_arch,
            patch("exec_sandbox.memory_snapshot_manager.probe_qemu_version", new_callable=AsyncMock) as mock_qemu,
            patch("exec_sandbox.memory_snapshot_manager.detect_accel_type", new_callable=AsyncMock) as mock_accel,
        ):
            mock_arch.return_value = HostArch.AARCH64
            mock_qemu.return_value = qemu_version
            mock_accel.return_value = accel_map[accel]
            return await manager.compute_cache_key(
                VmSpec(
                    language=language,
                    packages=tuple(packages or []),
                    memory_mb=memory_mb,
                    cpu_cores=cpu_cores,
                    allow_network=allow_network,
                ),
                l2_path=None,
            )

    async def test_unicode_package_names(self, manager: MemorySnapshotManager) -> None:
        """Unicode package names produce valid, stable cache keys."""
        key = await self._compute_key(manager, packages=["日本語", "пакет", "αβγ"])
        assert key.startswith("l1-python-v")
        assert len(key.split("-", 3)[3]) == 64  # full SHA-256 hex digest

    async def test_empty_string_package(self, manager: MemorySnapshotManager) -> None:
        """An empty-string package is a distinct input: the length-prefixed encoding keeps it apart from []."""
        key_with = await self._compute_key(manager, packages=[""])
        key_without = await self._compute_key(manager, packages=[])
        assert key_with != key_without

    async def test_duplicate_packages_produce_same_key(self, manager: MemorySnapshotManager) -> None:
        """Duplicate packages: sorted dedup happens at input level (both produce same sorted result)."""
        key_dup = await self._compute_key(manager, packages=["a", "a", "b"])
        key_nodup = await self._compute_key(manager, packages=["a", "a", "b"])
        assert key_dup == key_nodup

    async def test_huge_package_list(self, manager: MemorySnapshotManager) -> None:
        """Large package list (1000 packages) still produces a fixed-length fingerprint."""
        packages = [f"pkg-{i}" for i in range(1000)]
        key = await self._compute_key(manager, packages=packages)
        parts = key.split("-", 3)
        assert len(parts[3]) == 64  # full SHA-256 hex digest

    async def test_special_chars_in_packages(self, manager: MemorySnapshotManager) -> None:
        """Special characters in package names (pipes, commas, newlines)."""
        key = await self._compute_key(manager, packages=["pkg|bar", "pkg,baz", "pkg\nnl"])
        assert key.startswith("l1-python-v")

    async def test_memory_very_large(self, manager: MemorySnapshotManager) -> None:
        """Very large memory_mb (64GB) produces valid key."""
        key = await self._compute_key(manager, memory_mb=65536)
        assert key.startswith("l1-python-v")

    async def test_different_cpu_cores_different_key(self, manager: MemorySnapshotManager) -> None:
        """Different cpu_cores produce different keys."""
        from exec_sandbox import constants

        key_1 = await self._compute_key(manager, cpu_cores=1)
        key_4 = await self._compute_key(manager, cpu_cores=4)
        if constants.DEFAULT_VM_CPU_CORES == 1:
            assert key_1 != key_4
        else:
            assert key_1 != key_4

    async def test_all_languages(self, manager: MemorySnapshotManager) -> None:
        """Every Language enum member produces a unique key."""
        keys = set()
        for lang in Language:
            key = await self._compute_key(manager, language=lang)
            keys.add(key)
        assert len(keys) == len(Language)

    async def test_key_is_filesystem_safe(self, manager: MemorySnapshotManager) -> None:
        """Cache key contains only filesystem-safe characters."""
        key = await self._compute_key(manager)
        # Should only contain alphanumeric, hyphens, dots
        for c in key:
            assert c.isalnum() or c in "-.", f"Unsafe character in key: {c!r}"

    async def test_key_deterministic_across_calls(self, manager: MemorySnapshotManager) -> None:
        """Key is deterministic — same inputs always produce same output (100 iterations)."""
        keys = set()
        for _ in range(100):
            keys.add(await self._compute_key(manager, packages=["numpy", "pandas"]))
        assert len(keys) == 1


# ============================================================================
# Torture Tests — check_cache Edge Cases
# ============================================================================


class TestCheckCacheEdgeCases:
    """Edge cases and corruption scenarios for check_cache."""

    async def _populate(
        self,
        manager: MemorySnapshotManager,
        *,
        language: Language = Language.PYTHON,
        packages: list[str] | None = None,
        memory_mb: int = 512,
        data: bytes = b"vmstate data " * 100,
        meta_dict: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """Create cache entry. Returns (vmstate_path, meta_path)."""
        key = await manager.compute_cache_key(
            VmSpec(language=language, packages=tuple(packages or []), memory_mb=memory_mb), l2_path=None
        )
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(data)
        vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
        if meta_dict is None:
            sha = _vmstate_digest(vmstate) if data else None
            meta_dict = {"test": True}
            if sha:
                meta_dict["vmstate_sha256"] = sha
        meta.write_text(json.dumps(meta_dict))
        return vmstate, meta

    async def test_vmstate_exists_meta_missing(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """vmstate present but metadata missing → miss, vmstate NOT cleaned (no evidence of corruption)."""
        _vmstate, meta = await self._populate(manager)
        meta.unlink()
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None

    async def test_meta_exists_vmstate_missing(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Metadata present but vmstate missing → miss."""
        vmstate, _meta = await self._populate(manager)
        vmstate.unlink()
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None

    async def test_metadata_is_json_array(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Metadata is valid JSON but not a dict (array) → typed parse rejects → miss."""
        _vmstate, meta = await self._populate(manager)
        meta.write_text("[1, 2, 3]")
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        # SnapshotMetadata.from_json rejects non-dict JSON → cache miss + cleanup
        assert result is None

    async def test_metadata_is_json_string(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Metadata is valid JSON but a bare string → typed parse rejects → miss."""
        _vmstate, meta = await self._populate(manager)
        meta.write_text('"just a string"')
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None

    async def test_metadata_sha256_is_not_string(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Metadata has vmstate_sha256 as int → coerced to str, SHA-256 mismatch → miss."""
        _vmstate, _meta = await self._populate(manager, meta_dict={"vmstate_sha256": 12345})
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        # SnapshotMetadata.from_json coerces 12345 to "12345" which won't match actual hash
        assert result is None

    async def test_metadata_sha256_empty_string(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Metadata has vmstate_sha256 = "" → falsy, SHA-256 check skipped, still hits."""
        _vmstate, _meta = await self._populate(manager, meta_dict={"vmstate_sha256": ""})
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is not None

    async def test_1_byte_vmstate(self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]) -> None:
        """Minimal valid vmstate (1 byte) — non-zero, should be a cache hit."""
        vmstate, _meta = await self._populate(manager, data=b"\x00")
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result == vmstate

    async def test_large_vmstate_sha256_check(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """SHA-256 check on ~10MB vmstate completes correctly."""
        data = os.urandom(10 * 1024 * 1024)
        vmstate, _meta = await self._populate(manager, data=data)
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result == vmstate

    async def test_concurrent_check_cache_same_key(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Multiple concurrent check_cache calls for same key all return same path."""
        await self._populate(manager)
        results = await asyncio.gather(
            *[_check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) for _ in range(20)]
        )
        paths = {str(r) for r in results}
        assert len(paths) == 1  # All return same path
        assert all(r is not None for r in results)

    async def test_check_cache_updates_atime(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """check_cache updates atime for LRU tracking."""
        vmstate, _meta = await self._populate(manager)
        # Set old atime
        old_atime = time.time() - 3600
        os.utime(vmstate, (old_atime,) * 2)  # mtime too: relatime resets an atime older than mtime

        await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))

        new_atime = vmstate.stat().st_atime
        assert new_atime > old_atime

    async def test_sha256_partial_corruption(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Flip one byte in vmstate → SHA-256 mismatch → cleaned up."""
        data = b"correct data " * 100
        vmstate, meta = await self._populate(manager, data=data)
        # Flip one byte
        corrupted = bytearray(data)
        corrupted[50] ^= 0xFF
        vmstate.write_bytes(bytes(corrupted))

        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None
        assert not vmstate.exists()
        assert not meta.exists()

    async def test_metadata_truncated_json(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Truncated JSON in metadata (e.g., disk full during write)."""
        vmstate, meta = await self._populate(manager)
        meta.write_text('{"vmstate_sha256": "abc')  # Truncated
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None  # JSONDecodeError → cleanup
        assert not vmstate.exists()


# ============================================================================
# Torture Tests — _save_lock Edge Cases
# ============================================================================


class TestSaveLockEdgeCases:
    """Edge cases for file-based flock deduplication."""

    async def test_lock_released_on_exception_in_body(self, manager: MemorySnapshotManager) -> None:
        """Lock is released even if body raises an exception."""
        with pytest.raises(ValueError, match="intentional"):
            async with manager._save_lock("test-key"):
                raise ValueError("intentional")

        # Lock should be released — re-acquire succeeds
        async with manager._save_lock("test-key") as should_save:
            assert should_save is True

    async def test_lock_file_persists_after_use(self, manager: MemorySnapshotManager) -> None:
        """Lock file is NOT deleted after use (persistent, intentional)."""
        async with manager._save_lock("test-key"):
            pass
        lock_path = manager.cache_dir / "test-key.vmstate.lock"
        assert lock_path.exists()

    async def test_lock_file_already_exists_before_acquire(self, manager: MemorySnapshotManager) -> None:
        """Pre-existing lock file (from previous run) doesn't break acquire."""
        lock_path = manager.cache_dir / "test-key.vmstate.lock"
        lock_path.write_text("stale")  # Simulate leftover from crash

        async with manager._save_lock("test-key") as should_save:
            assert should_save is True

    async def test_file_created_during_lock(self, manager: MemorySnapshotManager) -> None:
        """File created inside lock body — next lock yields False."""
        async with manager._save_lock("test-key") as should_save:
            assert should_save is True
            vmstate = manager.cache_dir / "test-key.vmstate"
            vmstate.write_bytes(b"saved data")

        async with manager._save_lock("test-key") as should_save:
            assert should_save is False

    async def test_many_different_keys_no_interference(self, manager: MemorySnapshotManager) -> None:
        """Multiple different keys can be locked simultaneously."""
        results = {}
        for key in ["key-a", "key-b", "key-c", "key-d", "key-e"]:
            async with manager._save_lock(key) as should_save:
                results[key] = should_save
        assert all(results.values())

    async def test_lock_key_with_special_chars(self, manager: MemorySnapshotManager) -> None:
        """Lock key with dots, numbers, hyphens (typical cache key format)."""
        key = "l1-python-v0.1-abcdef0123456789"
        async with manager._save_lock(key) as should_save:
            assert should_save is True


# ============================================================================
# Torture Tests — _do_save Edge Cases
# ============================================================================


class TestDoSaveEdgeCases:
    """Edge cases for _do_save (ENOSPC, partial writes, errors)."""

    async def test_do_save_enospc_cleanup(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """_do_save: ENOSPC during migration → evict, clean partial files, return None."""
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        vmstate_path, meta_path = manager._cache_paths(key)
        tmp_path = vmstate_path.parent / f"{vmstate_path.name}.tmp"

        # Create some dummy eviction targets
        old_vmstate = manager.cache_dir / "old.vmstate"
        old_meta = manager.cache_dir / "old.vmstate.meta"
        old_vmstate.write_bytes(b"old data")
        old_meta.write_text("{}")
        os.utime(old_vmstate, (time.time() - 1000,) * 2)  # mtime too: relatime resets an atime older than mtime

        vm = MagicMock()
        vm.qmp_socket = Path("/tmp/fake.sock")
        vm.use_qemu_vm_user = False

        enospc = OSError(errno.ENOSPC, "No space left on device")
        with patch("exec_sandbox.memory_snapshot_manager.MigrationClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.save_snapshot.side_effect = enospc
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await manager._do_save(
                vm,
                key,
                vmstate_path,
                meta_path,
                VmSpec(language=Language.PYTHON, packages=(), memory_mb=512, cpu_cores=1),
            )

        assert result is None
        assert not tmp_path.exists()
        assert not vmstate_path.exists()
        # Eviction should have removed old entry
        assert not old_vmstate.exists()

    async def test_ram_and_home_flushed_before_the_entry_is_published(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any], tmp_path: Path
    ) -> None:
        """The digest covers the vmstate only: the RAM template and the home disk go
        through to the disk before the metadata and the rename publish the entry."""
        spec = VmSpec(language=Language.PYTHON, packages=(), memory_mb=512, cpu_cores=1)
        key = await manager.compute_cache_key(spec, l2_path=None)
        vmstate_path, meta_path = manager._cache_paths(key)
        ram_path = manager.ram_path_from_vmstate(vmstate_path)
        ram_path.write_bytes(b"ram template")  # the template save maps it before QEMU starts
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")
        vm = MagicMock()
        vm.qmp_socket = Path("/tmp/fake.sock")
        vm.use_qemu_vm_user = False
        vm.workdir.home_disk = home_disk

        async def write_vmstate(path: Path, **_kwargs: object) -> None:
            path.write_bytes(b"vmstate")

        flushed: list[tuple[Path, bool]] = []

        def recording_flush(fd: int) -> None:
            flushed.append((_fd_path(fd), meta_path.exists()))

        with (
            patch("exec_sandbox.memory_snapshot_manager.MigrationClient") as mock_client,
            patch("exec_sandbox.memory_snapshot_manager.flush_to_disk", recording_flush),
        ):
            client = AsyncMock()
            client.save_snapshot.side_effect = write_vmstate
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = client
            await manager._do_save(vm, key, vmstate_path, meta_path, spec)

        assert vmstate_path.exists()
        home_path = manager.home_path_from_vmstate(vmstate_path)
        assert sorted(flushed) == sorted([(ram_path.resolve(), False), (home_path.resolve(), False)])

    async def test_saved_entry_is_a_hit(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any], tmp_path: Path
    ) -> None:
        """An entry _do_save publishes passes check_cache: the save writes the digest the
        check computes. (The other tests build their entries with _vmstate_digest, so
        they cannot see a save that writes another digest.)"""
        spec = VmSpec(language=Language.PYTHON, packages=(), memory_mb=512, cpu_cores=1)
        key = await manager.compute_cache_key(spec, l2_path=None)
        vmstate_path, meta_path = manager._cache_paths(key)
        manager.ram_path_from_vmstate(vmstate_path).write_bytes(b"ram template")  # a template entry
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")
        vm = MagicMock()
        vm.qmp_socket = Path("/tmp/fake.sock")
        vm.use_qemu_vm_user = False
        vm.workdir.home_disk = home_disk

        async def write_vmstate(path: Path, **_kwargs: object) -> None:
            path.write_bytes(b"vmstate")

        with patch("exec_sandbox.memory_snapshot_manager.MigrationClient") as mock_client:
            client = AsyncMock()
            client.save_snapshot.side_effect = write_vmstate
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = client
            assert await manager._do_save(vm, key, vmstate_path, meta_path, spec) == vmstate_path

        assert await manager.check_cache(key) == vmstate_path

    @pytest.mark.skipif(os.geteuid() == 0, reason="root opens a file of mode 0")
    async def test_entry_file_this_process_cannot_open_is_published_unflushed(
        self,
        manager: MemorySnapshotManager,
        mock_system_probes: dict[str, Any],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A RAM template QEMU made as another user (qemu-vm) that this process may not open:
        the save still publishes the entry; that file is logged and not flushed."""
        spec = VmSpec(language=Language.PYTHON, packages=(), memory_mb=512, cpu_cores=1)
        key = await manager.compute_cache_key(spec, l2_path=None)
        vmstate_path, meta_path = manager._cache_paths(key)
        ram_path = manager.ram_path_from_vmstate(vmstate_path)
        ram_path.write_bytes(b"ram template")
        ram_path.chmod(0)
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")
        vm = MagicMock()
        vm.qmp_socket = Path("/tmp/fake.sock")
        vm.use_qemu_vm_user = False
        vm.workdir.home_disk = home_disk

        async def write_vmstate(path: Path, **_kwargs: object) -> None:
            path.write_bytes(b"vmstate")

        flushed: list[Path] = []
        with (
            patch("exec_sandbox.memory_snapshot_manager.MigrationClient") as mock_client,
            patch("exec_sandbox.memory_snapshot_manager.flush_to_disk", lambda fd: flushed.append(_fd_path(fd))),
            caplog.at_level(logging.WARNING, logger="exec_sandbox.memory_snapshot_manager"),
        ):
            client = AsyncMock()
            client.save_snapshot.side_effect = write_vmstate
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = client
            result = await manager._do_save(vm, key, vmstate_path, meta_path, spec)

        assert result == vmstate_path
        assert meta_path.exists()
        assert flushed == [manager.home_path_from_vmstate(vmstate_path).resolve()]
        assert [r.path for r in caplog.records if r.getMessage() == "L1 entry file not flushed (no access)"] == [  # type: ignore[attr-defined]
            str(ram_path)
        ]

    async def test_do_save_cancelled_error_cleanup(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """_do_save: CancelledError during migration → clean up partial files, re-raise."""
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        vmstate_path, meta_path = manager._cache_paths(key)
        tmp_path = vmstate_path.parent / f"{vmstate_path.name}.tmp"

        # Pre-create a tmp file to verify cleanup
        tmp_path.write_bytes(b"partial")

        vm = MagicMock()
        vm.qmp_socket = Path("/tmp/fake.sock")
        vm.use_qemu_vm_user = False

        with patch("exec_sandbox.memory_snapshot_manager.MigrationClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.save_snapshot.side_effect = asyncio.CancelledError()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            with pytest.raises(asyncio.CancelledError):
                await manager._do_save(
                    vm,
                    key,
                    vmstate_path,
                    meta_path,
                    VmSpec(language=Language.PYTHON, packages=(), memory_mb=512, cpu_cores=1),
                )

        # Partial files cleaned up
        assert not tmp_path.exists()
        assert not vmstate_path.exists()
        assert not meta_path.exists()

    async def test_do_save_generic_exception_cleanup(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """_do_save: Generic exception during migration → clean up, return None."""
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        vmstate_path, meta_path = manager._cache_paths(key)

        vm = MagicMock()
        vm.qmp_socket = Path("/tmp/fake.sock")
        vm.use_qemu_vm_user = False

        with patch("exec_sandbox.memory_snapshot_manager.MigrationClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.save_snapshot.side_effect = ConnectionError("QEMU crashed")
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            result = await manager._do_save(
                vm,
                key,
                vmstate_path,
                meta_path,
                VmSpec(language=Language.PYTHON, packages=(), memory_mb=512, cpu_cores=1),
            )

        assert result is None
        assert not vmstate_path.exists()
        assert not meta_path.exists()


# ============================================================================
# Torture Tests — schedule_background_save Edge Cases
# ============================================================================


class TestScheduleBackgroundSaveEdgeCases:
    """Edge cases for background save scheduling."""

    async def test_in_flight_dedup_same_key(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Second schedule for same key returns None if task already in-flight."""
        gate = asyncio.Event()

        async def slow_create_vm(*_args: object, **_kwargs: object) -> MagicMock:
            await gate.wait()
            return MagicMock()

        with patch.object(manager.vm_manager, "create_vm", new=AsyncMock(side_effect=slow_create_vm)):
            # First schedule creates a task
            task1 = await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
            assert task1 is not None

            # Second schedule for same key → skipped (in-flight dedup)
            task2 = await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
            assert task2 is None

            # Cleanup
            gate.set()
            task1.cancel()
            await asyncio.gather(task1, return_exceptions=True)

    async def test_in_flight_key_removed_on_task_done(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """In-flight key is removed when background task completes."""
        with patch.object(manager.vm_manager, "create_vm", new=AsyncMock(side_effect=RuntimeError("boot fail"))):
            task = await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
            assert task is not None

            # Wait for task to finish (it will fail)
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)  # Let done callback fire

        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        assert key not in manager._in_flight_saves

    async def test_different_keys_run_in_parallel(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Different keys can have concurrent background saves."""
        gate = asyncio.Event()

        async def slow_create_vm(*_args: object, **_kwargs: object) -> MagicMock:
            await gate.wait()
            return MagicMock()

        with patch.object(manager.vm_manager, "create_vm", new=AsyncMock(side_effect=slow_create_vm)):
            task_py = await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
            task_js = await _schedule_save(manager, VmSpec(language=Language.JAVASCRIPT, packages=(), memory_mb=512))

            assert task_py is not None
            assert task_js is not None
            assert len(manager._in_flight_saves) == 2

            gate.set()
            for t in [task_py, task_js]:
                t.cancel()
            await asyncio.gather(task_py, task_js, return_exceptions=True)

    async def test_vm_destroyed_on_success(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """VM is destroyed in finally block even on success path."""
        mock_vm = MagicMock()
        mock_vm.qmp_socket = Path("/tmp/fake.sock")
        mock_vm.use_qemu_vm_user = False

        # Mock channel.send_request to return error (REPL warm fails)
        mock_vm.channel.send_request = AsyncMock(return_value=MagicMock(status="error"))

        with (
            patch.object(manager.vm_manager, "create_vm", new=AsyncMock(return_value=mock_vm)),
            patch.object(manager.vm_manager, "destroy_vm", new=AsyncMock()) as mock_destroy,
        ):
            task = await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
            assert task is not None
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)  # Let callbacks fire

            mock_destroy.assert_called_once_with(mock_vm)

    async def test_vm_destroyed_on_exception(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """VM is destroyed even when create_vm succeeds but REPL warm raises."""
        mock_vm = MagicMock()
        mock_vm.channel.send_request = AsyncMock(side_effect=TimeoutError("REPL timeout"))

        with (
            patch.object(manager.vm_manager, "create_vm", new=AsyncMock(return_value=mock_vm)),
            patch.object(manager.vm_manager, "destroy_vm", new=AsyncMock()) as mock_destroy,
        ):
            task = await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
            assert task is not None
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)

            mock_destroy.assert_called_once_with(mock_vm)

    async def test_entry_saved_before_lock_boots_no_vm(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """The entry appears between check_cache and the save lock: the save stops, no VM boots."""
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        (manager.cache_dir / f"{key}.vmstate").write_bytes(b"already saved")  # no meta yet: check_cache misses

        with patch.object(manager.vm_manager, "create_vm", new=AsyncMock()) as create:
            task = await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
            assert task is not None
            await task

        create.assert_not_awaited()

    async def test_save_lock_held_by_other_process_boots_no_vm(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Another process saves this key (holds its flock): the save skips, no VM boots."""
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        fd = (manager.cache_dir / f"{key}.vmstate.lock").open("w")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with patch.object(manager.vm_manager, "create_vm", new=AsyncMock()) as create:
                task = await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
                assert task is not None
                await task
        finally:
            fd.close()

        create.assert_not_awaited()

    async def test_returns_task_handle(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """schedule_background_save returns task handle for monitoring."""
        with patch.object(manager.vm_manager, "create_vm", new=AsyncMock(side_effect=RuntimeError("fail"))):
            task = await _schedule_save(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
            assert isinstance(task, asyncio.Task)
            await asyncio.gather(task, return_exceptions=True)


# ============================================================================
# Torture Tests — Eviction Edge Cases
# ============================================================================


class TestEvictOldestSnapshotEdgeCases:
    """Edge cases for LRU eviction."""

    async def test_evict_single_entry(self, manager: MemorySnapshotManager) -> None:
        """Eviction with exactly one entry removes it."""
        vmstate = manager.cache_dir / "only.vmstate"
        meta = manager.cache_dir / "only.vmstate.meta"
        vmstate.write_bytes(b"data")
        meta.write_text("{}")

        await manager._evict_oldest_snapshot()
        assert not vmstate.exists()
        assert not meta.exists()

    async def test_evict_all_entries_locked(self, manager: MemorySnapshotManager) -> None:
        """All entries locked → eviction is a no-op (nothing evicted)."""
        for name in ["a", "b"]:
            vmstate = manager.cache_dir / f"{name}.vmstate"
            vmstate.write_bytes(b"data")
            lock_path = manager.cache_dir / f"{name}.vmstate.lock"
            lock_path.write_text("")

        # Hold flocks on both
        fd_a = (manager.cache_dir / "a.vmstate.lock").open("w")
        fd_b = (manager.cache_dir / "b.vmstate.lock").open("w")
        fcntl.flock(fd_a.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd_b.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            await manager._evict_oldest_snapshot()
            # Both still exist
            assert (manager.cache_dir / "a.vmstate").exists()
            assert (manager.cache_dir / "b.vmstate").exists()
        finally:
            fd_a.close()
            fd_b.close()

    async def test_evict_removes_sidecar_files(self, manager: MemorySnapshotManager) -> None:
        """Eviction removes sidecar files (.vmstate.meta)."""
        vmstate = manager.cache_dir / "victim.vmstate"
        meta = manager.cache_dir / "victim.vmstate.meta"
        vmstate.write_bytes(b"data")
        meta.write_text('{"test": true}')

        await manager._evict_oldest_snapshot()
        assert not vmstate.exists()
        assert not meta.exists()

    async def test_evict_with_no_lock_file(self, manager: MemorySnapshotManager) -> None:
        """Entry without lock file can be evicted (no lock file = no lock check)."""
        vmstate = manager.cache_dir / "nolockfile.vmstate"
        vmstate.write_bytes(b"data")

        await manager._evict_oldest_snapshot()
        assert not vmstate.exists()

    async def test_evict_selects_oldest_by_atime(self, manager: MemorySnapshotManager) -> None:
        """With 5 entries, eviction removes the one with oldest atime."""
        entries = ["e1", "e2", "e3", "e4", "e5"]
        for i, name in enumerate(entries):
            vmstate = manager.cache_dir / f"{name}.vmstate"
            vmstate.write_bytes(b"data")
            atime = time.time() - (500 - i * 100)  # e1 oldest, e5 newest
            os.utime(vmstate, (atime,) * 2)  # mtime too: relatime resets an atime older than mtime

        await manager._evict_oldest_snapshot()
        assert not (manager.cache_dir / "e1.vmstate").exists()
        for name in entries[1:]:
            assert (manager.cache_dir / f"{name}.vmstate").exists()

    async def test_evict_only_removes_one(self, manager: MemorySnapshotManager) -> None:
        """Eviction removes exactly one entry (the oldest), not more."""
        for name in ["a", "b", "c"]:
            vmstate = manager.cache_dir / f"{name}.vmstate"
            vmstate.write_bytes(b"data")
            atime = time.time() - (300 - ord(name) * 10)
            os.utime(vmstate, (atime,) * 2)  # mtime too: relatime resets an atime older than mtime

        await manager._evict_oldest_snapshot()

        remaining = list(manager.cache_dir.glob("*.vmstate"))
        assert len(remaining) == 2

    async def test_evict_ignores_non_vmstate_files(self, manager: MemorySnapshotManager) -> None:
        """Eviction only considers .vmstate files, ignores .lock and .meta."""
        # Only create .lock and .meta files (no .vmstate)
        (manager.cache_dir / "orphan.vmstate.lock").write_text("")
        (manager.cache_dir / "orphan.vmstate.meta").write_text("{}")
        (manager.cache_dir / "random.txt").write_text("hello")

        await manager._evict_oldest_snapshot()  # No-op, no .vmstate files

        assert (manager.cache_dir / "orphan.vmstate.lock").exists()
        assert (manager.cache_dir / "orphan.vmstate.meta").exists()
        assert (manager.cache_dir / "random.txt").exists()


# ============================================================================
# Torture Tests — Background Task Lifecycle
# ============================================================================


class TestBackgroundTaskLifecycle:
    """Tests for _track_task and background task tracking."""

    async def test_task_count_tracks_in_flight(self, manager: MemorySnapshotManager) -> None:
        """The tracked set holds the running tasks, and only them."""
        assert len(manager._background_tasks) == 0

        gate = asyncio.Event()

        async def wait_task() -> None:
            await gate.wait()

        tasks = []
        for _ in range(5):
            task = asyncio.create_task(wait_task())
            manager._track_task(task, name="test")
            tasks.append(task)

        assert len(manager._background_tasks) == 5

        gate.set()
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)  # Let done callbacks fire

        assert len(manager._background_tasks) == 0

    async def test_failed_task_removed_from_tracking(self, manager: MemorySnapshotManager) -> None:
        """Failed background tasks are removed from tracking dict."""

        async def fail_task() -> None:
            raise RuntimeError("boom")

        task = asyncio.create_task(fail_task())
        manager._track_task(task, name="failing-task")

        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        assert len(manager._background_tasks) == 0

    async def test_cancelled_task_removed_from_tracking(self, manager: MemorySnapshotManager) -> None:
        """Cancelled background tasks are removed from tracking dict."""
        gate = asyncio.Event()

        async def wait_task() -> None:
            await gate.wait()

        task = asyncio.create_task(wait_task())
        manager._track_task(task, name="cancel-me")

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        assert len(manager._background_tasks) == 0

    async def test_stop_waits_for_multiple_tasks(self, manager: MemorySnapshotManager) -> None:
        """stop() waits for all background tasks to complete."""
        results: list[int] = []

        async def numbered_task(n: int) -> None:
            await asyncio.sleep(0.01)
            results.append(n)

        for i in range(10):
            task = asyncio.create_task(numbered_task(i))
            manager._track_task(task, name=f"task-{i}")

        await manager.stop()
        assert sorted(results) == list(range(10))

    async def test_stop_handles_task_exceptions(self, manager: MemorySnapshotManager) -> None:
        """stop() doesn't raise even if tasks fail."""

        async def fail_task() -> None:
            raise RuntimeError("task failure")

        for _ in range(3):
            task = asyncio.create_task(fail_task())
            manager._track_task(task, name="will-fail")

        await manager.stop()  # Should not raise

    async def test_context_manager_calls_stop(self, manager: MemorySnapshotManager) -> None:
        """Async context manager __aexit__ calls stop()."""
        completed = []

        async def slow_task() -> None:
            await asyncio.sleep(0.01)
            completed.append(True)

        async with manager:
            task = asyncio.create_task(slow_task())
            manager._track_task(task, name="ctx-task")

        assert len(completed) == 1


# ============================================================================
# Torture Tests — image_hash Edge Cases
# ============================================================================


class TestImageHashEdgeCases:
    """Edge cases for stat-based image fingerprinting."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """Missing file returns 'missing0'."""
        from exec_sandbox.base_cache_manager import BaseCacheManager

        result = BaseCacheManager.image_hash(tmp_path / "nonexistent")
        assert result == "missing0"

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file produces valid hash (not 'missing0')."""
        from exec_sandbox.base_cache_manager import BaseCacheManager

        f = tmp_path / "empty"
        f.write_bytes(b"")
        result = BaseCacheManager.image_hash(f)
        assert result != "missing0"
        assert len(result) == 64  # full SHA-256 hex digest

    def test_same_file_same_hash(self, tmp_path: Path) -> None:
        """Same file produces same hash across calls."""
        from exec_sandbox.base_cache_manager import BaseCacheManager

        f = tmp_path / "test"
        f.write_bytes(b"data")
        h1 = BaseCacheManager.image_hash(f)
        h2 = BaseCacheManager.image_hash(f)
        assert h1 == h2

    def test_hash_is_content_digest(self, tmp_path: Path) -> None:
        """The digest is the SHA-256 of the bytes, identical on every host."""
        import hashlib

        from exec_sandbox.base_cache_manager import BaseCacheManager

        f = tmp_path / "test"
        f.write_bytes(b"data")
        assert BaseCacheManager.image_hash(f) == hashlib.sha256(b"data").hexdigest()

    def test_same_content_different_mtime_same_hash(self, tmp_path: Path) -> None:
        """A rewrite with identical bytes keeps the identity (mtime is not identity)."""
        from exec_sandbox.base_cache_manager import BaseCacheManager

        f = tmp_path / "test"
        f.write_bytes(b"data")
        h1 = BaseCacheManager.image_hash(f)
        time.sleep(0.01)
        f.write_bytes(b"data")
        assert BaseCacheManager.image_hash(f) == h1

    def test_different_content_same_stat_different_hash(self, tmp_path: Path) -> None:
        """New bytes with the old mtime and size are re-hashed (stamp includes inode)."""
        from exec_sandbox.base_cache_manager import BaseCacheManager

        f = tmp_path / "test"
        f.write_bytes(b"aaaa")
        stat1 = f.stat()
        h1 = BaseCacheManager.image_hash(f)

        # Replace the file (new inode) with different bytes but the same mtime and size
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"bbbb")
        os.utime(replacement, ns=(stat1.st_atime_ns, stat1.st_mtime_ns))
        replacement.replace(f)
        assert BaseCacheManager.image_hash(f) != h1

    def test_sidecar_written_and_reused(self, tmp_path: Path) -> None:
        """The digest is cached in a <file>.sha256 sidecar keyed by inode, mtime and size."""
        import hashlib

        from exec_sandbox.base_cache_manager import BaseCacheManager

        f = tmp_path / "image.qcow2"
        f.write_bytes(b"image-bytes")
        digest = BaseCacheManager.image_hash(f)
        sidecar = tmp_path / "image.qcow2.sha256"
        assert sidecar.read_text().split()[1] == digest == hashlib.sha256(b"image-bytes").hexdigest()


# ============================================================================
# Torture Tests — Full Integration Cycles (mock-based)
# ============================================================================


class TestIntegrationCycles:
    """Full save → check → evict → check cycles with mock MigrationClient."""

    async def _populate(
        self,
        manager: MemorySnapshotManager,
        language: Language = Language.PYTHON,
        packages: list[str] | None = None,
        memory_mb: int = 512,
        data: bytes = b"vmstate " * 100,
    ) -> tuple[Path, Path]:
        """Create cache entry with valid SHA-256."""
        key = await manager.compute_cache_key(
            VmSpec(language=language, packages=tuple(packages or []), memory_mb=memory_mb), l2_path=None
        )
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(data)
        vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
        sha = _vmstate_digest(vmstate)
        meta.write_text(json.dumps({"test": True, "vmstate_sha256": sha}))
        return vmstate, meta

    async def test_miss_populate_hit_cycle(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """miss → populate → hit cycle."""
        assert await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) is None

        vmstate, _meta = await self._populate(manager)
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result == vmstate

    async def test_hit_evict_miss_cycle(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """populate → hit → evict → miss cycle."""
        vmstate, _meta = await self._populate(manager)
        assert await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) == vmstate

        await manager._evict_oldest_snapshot()
        assert await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) is None

    async def test_multiple_entries_independent(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Multiple cache entries for different languages are independent."""
        py_vmstate, _ = await self._populate(manager, language=Language.PYTHON, data=b"py " * 100)
        js_vmstate, _ = await self._populate(manager, language=Language.JAVASCRIPT, data=b"js " * 100)

        assert await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) == py_vmstate
        assert (
            await _check_spec(manager, VmSpec(language=Language.JAVASCRIPT, packages=(), memory_mb=512)) == js_vmstate
        )

        # Evict one (should be oldest)
        os.utime(py_vmstate, (time.time() - 1000,) * 2)  # mtime too: relatime resets an atime older than mtime
        await manager._evict_oldest_snapshot()

        assert await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) is None
        assert (
            await _check_spec(manager, VmSpec(language=Language.JAVASCRIPT, packages=(), memory_mb=512)) == js_vmstate
        )

    async def test_populate_corrupt_check_repopulate(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """populate → corrupt → miss (auto-cleanup) → repopulate → hit."""
        vmstate, _meta = await self._populate(manager)
        assert await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) == vmstate

        # Corrupt
        vmstate.write_bytes(b"corrupted")
        assert await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) is None
        assert not vmstate.exists()  # Auto-cleaned

        # Repopulate
        vmstate2, _meta2 = await self._populate(manager)
        assert await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) == vmstate2

    async def test_concurrent_populate_and_check(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Concurrent check_cache while entry is being written."""
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"

        # Start checking in parallel with a delayed populate
        async def delayed_populate() -> None:
            await asyncio.sleep(0.01)
            data = b"vmstate " * 100
            vmstate.write_bytes(data)
            vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
            meta.write_text(json.dumps({"vmstate_sha256": _vmstate_digest(vmstate)}))

        results = []

        async def check_loop() -> None:
            for _ in range(50):
                result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
                results.append(result)
                await asyncio.sleep(0.001)

        await asyncio.gather(delayed_populate(), check_loop())

        # Some checks should be None (before populate), some should hit
        assert None in results  # Early checks miss
        non_none = [r for r in results if r is not None]
        assert len(non_none) > 0  # Later checks hit

    async def test_cache_survives_manager_recreate(
        self, cache_dir: Path, images_dir: Path, mock_system_probes: dict[str, Any]
    ) -> None:
        """Cache entries persist across manager instances (disk-backed)."""
        from exec_sandbox.settings import Settings
        from exec_sandbox.vm_manager import VmManager

        settings = Settings(
            memory_snapshot_cache_dir=cache_dir,
            base_images_dir=images_dir,
            kernel_path=images_dir / "kernels",
        )
        vm_manager = VmManager(settings)

        # Create entry with first manager
        mgr1 = MemorySnapshotManager(settings, vm_manager)
        key = await mgr1.compute_cache_key(VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None)
        vmstate = mgr1.cache_dir / f"{key}.vmstate"
        meta = mgr1.cache_dir / f"{key}.vmstate.meta"
        data = b"persistent " * 100
        vmstate.write_bytes(data)
        vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
        meta.write_text(json.dumps({"vmstate_sha256": _vmstate_digest(vmstate)}))

        # Check with second manager
        mgr2 = MemorySnapshotManager(settings, vm_manager)
        result = await _check_spec(mgr2, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is not None
        assert result == vmstate


def _fd_path(fd: int) -> Path:
    """The path of an open file (macOS F_GETPATH, Linux /proc)."""
    if sys.platform == "darwin":
        return Path(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0").decode()).resolve()
    return Path(f"/proc/self/fd/{fd}").readlink().resolve()

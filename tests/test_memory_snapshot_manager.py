"""Tests for MemorySnapshotManager (L1 memory snapshot cache).

The manager runs with real Settings and a real VmManager (not started). Stand-ins replace
the host probes (QEMU version, accelerator, CPU), the save VM (VmManager.ephemeral_vm),
the migration (MigrationClient) and setfacl (grant_qemu_vm_acl): no QEMU starts.

Includes:
- Normal case tests for all public APIs
- Edge cases (corruption, race conditions, boundary values)
- Out-of-bound/stress scenarios (concurrent access, error recovery)
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import gc
import json
import logging
import os
import re
import shutil
import stat
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Generator

import pytest

from exec_sandbox import memory_snapshot_manager as msm
from exec_sandbox.exceptions import (
    ConfigError,
    ErrorCode,
    InternalError,
    RouteError,
    SandboxError,
    TransientError,
)
from exec_sandbox.files.disk import pwrite_all
from exec_sandbox.files.locks import CacheEntry, EntryLease
from exec_sandbox.guest_agent_protocol import WarmReplAckMessage
from exec_sandbox.memory_snapshot_files import open_live_ram_dir, remove_dead_live_ram_dirs
from exec_sandbox.memory_snapshot_manager import MemorySnapshotManager, _publish_ram, _vmstate_digest
from exec_sandbox.models import Language
from exec_sandbox.platform_utils import get_arch_name
from exec_sandbox.private_run_dir import PrivateRunDir, new_folder_name
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import asyncio_with
from tests.test_files_locks import evictable, leased_here, leased_in_child

_BLOCK = 64 * 1024

_VMSTATE = b"vmstate"
"""The vmstate of the entries the tests make by hand (7 bytes)."""

_VMSTATE_DIGEST = "daf7866fe199125e85d29fcb1e7186ea601aa195d28570c8f6d0c11dceeef5c9"
"""The L1 digest of _VMSTATE, worked out from its definition: SHA-256 of the size
(7, 8 bytes big-endian), then of its one non-zero 64 KiB block: offset 0 (8 bytes
big-endian) and the 7 bytes. A check that computes another digest misses these entries."""

_KEY = re.compile(r"l1-python-v\d+\.\d+-[0-9a-f]{64}")
"""An L1 key of a Python spec: l1-{language}-v{major.minor}-{full SHA-256 hex digest}."""


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Fresh temporary cache directory."""
    return tmp_path / "l1-cache"


@pytest.fixture(autouse=True)
def live_ram_root(tmp_path: Path) -> Generator[Path]:
    """The folder a save VM's RAM would go to in memory (/dev/shm): a test folder, not made
    unless the test asks (`in_memory`): with no such folder the RAM stays in the cache
    folder. No test writes to the host's /dev/shm, and every test sees the same host."""
    root = tmp_path / "shm"
    with patch("exec_sandbox.memory_snapshot_files._LIVE_RAM_ROOT", root):
        yield root


@pytest.fixture
def in_memory(live_ram_root: Path) -> Generator[Path]:
    """`live_ram_root` is there (Linux /dev/shm). Its folders stay until closed: the save
    must remove its folder itself, not leave it to garbage collection."""
    live_ram_root.mkdir()
    live_ram_root.chmod(0o1777)  # as /dev/shm: sticky, open to every user
    kept: list[PrivateRunDir] = []
    real_open = PrivateRunDir.open

    def open_and_keep(root: Path, *, name: str) -> PrivateRunDir:
        kept.append(real_open(root, name=name))
        return kept[-1]

    with patch("exec_sandbox.memory_snapshot_files.PrivateRunDir.open", open_and_keep):
        yield live_ram_root
    for folder in kept:
        folder.close()


def _live_folders(root: Path) -> list[Path]:
    """The save folders (PrivateRunDir) under the memory root, of every process of this user."""
    user_root = root / f"exec-sandbox-{os.geteuid()}"
    return sorted(user_root.iterdir()) if user_root.exists() else []


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    """Dummy image files so the real VmManager.images (GuestImages) finds them."""
    d = tmp_path / "images"
    d.mkdir()
    arch = get_arch_name()
    (d / f"python-3.14-base-{arch}.qcow2").touch()
    (d / f"node-1.4-base-{arch}.qcow2").touch()
    (d / f"raw-base-{arch}.qcow2").touch()
    kernels = d / "kernels"
    kernels.mkdir()
    for arch in ("aarch64", "x86_64"):
        (kernels / f"vmlinuz-{arch}").write_bytes(b"fake-kernel")
        (kernels / f"initramfs-{arch}").write_bytes(b"fake-initramfs")
    return d


def test_cache_folder_is_private_whatever_the_umask(tmp_path: Path, images_dir: Path) -> None:
    """Made 0700 under umask 002 (the Ubuntu default) and any group (sg qemu-vm): the L1 and
    L2 folders (BaseCacheManager._make_cache_dir)."""
    from exec_sandbox.settings import Settings
    from exec_sandbox.vm_manager import VmManager

    cache = tmp_path / "cache" / "memory-snapshots"
    settings = Settings(memory_snapshot_cache_dir=cache, base_images_dir=images_dir, kernel_path=images_dir / "kernels")
    old = os.umask(0o002)
    try:
        MemorySnapshotManager(settings, VmManager(settings))
    finally:
        os.umask(old)
    assert [stat.S_IMODE(p.stat().st_mode) & 0o777 for p in (cache.parent, cache)] == [0o700, 0o700]


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


async def _save(manager: MemorySnapshotManager, spec: VmSpec) -> None:
    """One whole background save of `spec` (schedule_background_save, then its task)."""
    task = await _schedule_save(manager, spec)
    assert task is not None
    await task


def _files_of(manager: MemorySnapshotManager, key: str) -> list[str]:
    """The suffixes of the files of `key` in the cache folder, lock files aside (they stay:
    files.locks)."""
    return sorted(
        path.name.removeprefix(key)
        for path in manager.cache_dir.iterdir()
        if path.name.startswith(key) and not path.name.endswith(".lock")
    )


def _save_vm(*, home_disk: Path, qemu_vm: bool) -> MagicMock:
    """A save VM whose guest answers the REPL warm-up. No QEMU starts."""
    vm = MagicMock()
    vm.qmp_socket = Path("/tmp/fake.sock")
    vm.use_qemu_vm_user = qemu_vm
    vm.workdir.home_disk = home_disk
    vm.channel.send_request = AsyncMock(return_value=WarmReplAckMessage(language="python", status="ok"))
    return vm


@contextlib.contextmanager
def _template_save(
    manager: MemorySnapshotManager, *, vm: MagicMock, migrate: Callable[..., Awaitable[None]]
) -> Generator[list[Path]]:
    """schedule_background_save runs its whole save: `vm` is the save VM, and `migrate`
    is the migration that writes the device state to the file it gets. Yields the RAM file
    of each save VM."""
    mapped: list[Path] = []

    @contextlib.asynccontextmanager
    async def ephemeral_vm(*_args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
        mapped.append(Path(cast("str", kwargs["mem_path"])))
        yield vm

    client = AsyncMock()
    client.save_snapshot.side_effect = migrate
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with (
        patch(
            "exec_sandbox.memory_snapshot_manager.probe_qemu_as_qemu_vm",
            AsyncMock(return_value=vm.use_qemu_vm_user),
        ),
        patch.object(manager.vm_manager, "ephemeral_vm", ephemeral_vm),
        patch("exec_sandbox.memory_snapshot_manager.MigrationClient", return_value=client),
    ):
        yield mapped


async def _write_vmstate(path: Path, **_kwargs: object) -> None:
    """The migration: QEMU writes the device state to the file the host made."""
    with path.open("r+b") as f:  # QEMU opens the existing file: O_CREAT without O_EXCL
        f.write(b"device state")


class _SaveVms:
    """Save VMs that never reach a save: each boot records its task_id and RAM file, waits
    for `gate`, then fails (`fail`) or yields a VM whose REPL warm-up fails. The save then
    ends without an entry. No QEMU starts."""

    def __init__(self, *, gate: asyncio.Event, fail: bool) -> None:
        self.task_ids: list[str] = []
        self.mem_paths: list[Path] = []
        self._gate = gate
        self._fail = fail
        self._booted = asyncio.Condition()

    @contextlib.asynccontextmanager
    async def ephemeral_vm(
        self, *_args: object, task_id: str, mem_path: str, **_kwargs: object
    ) -> AsyncIterator[MagicMock]:
        async with self._booted:
            self.task_ids.append(task_id)
            self.mem_paths.append(Path(mem_path))
            self._booted.notify_all()
        await self._gate.wait()
        if self._fail:
            raise TransientError("boot failed", code=ErrorCode.QEMU_CRASHED, context=None, hint=None, instance_id=None)
        vm = MagicMock()
        vm.channel.send_request = AsyncMock(return_value=MagicMock(status="error"))
        yield vm

    async def booted(self, count: int) -> None:
        """Wait until `count` save VMs have booted (a hang guard of 5 s: TimeoutError when
        the saves do not boot)."""
        async with asyncio.timeout(5), self._booted:
            await self._booted.wait_for(lambda: len(self.task_ids) >= count)

    @contextlib.contextmanager
    def of(self, manager: MemorySnapshotManager) -> Generator[None]:
        """The save VMs of `manager` are these."""
        with patch.object(manager.vm_manager, "ephemeral_vm", self.ephemeral_vm):
            yield


def _open_gate() -> asyncio.Event:
    gate = asyncio.Event()
    gate.set()
    return gate


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
        # QEMU as this user (no sudo probe on a Linux host with the qemu-vm setup)
        patch("exec_sandbox.memory_snapshot_manager.probe_qemu_as_qemu_vm", new_callable=AsyncMock, return_value=False),
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

        monkeypatch.setattr("exec_sandbox.memory_snapshot_manager.asyncio", asyncio_with(to_thread=counted))
        await self._compute_key(manager, packages=["pandas==2.2.3"], l2_path=l2)
        assert len(jobs) == 1, jobs

    async def test_same_inputs_same_key(self, manager: MemorySnapshotManager) -> None:
        """Identical inputs produce identical cache keys."""
        key1 = await self._compute_key(manager, packages=["numpy", "pandas"])
        key2 = await self._compute_key(manager, packages=["numpy", "pandas"])
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

    async def test_different_cpu_model_different_key(self, manager: MemorySnapshotManager) -> None:
        """Another -cpu value (same accelerator) gives another key: a restore needs the CPU it
        was saved on (an ARM TCG entry saved with pointer auth cannot resume without it)."""
        with patch("exec_sandbox.memory_snapshot_manager.guest_cpu_model", new_callable=AsyncMock) as cpu:
            cpu.return_value = "max,pauth-impdef=on"
            key_max = await self._compute_key(manager, accel="TCG")
            cpu.return_value = "neoverse-v1,pauth=off"
            key_v1 = await self._compute_key(manager, accel="TCG")
        assert key_max != key_v1

    async def test_x86_tsc_deadline_changes_the_key(self, manager: MemorySnapshotManager) -> None:
        """On x86 the guest CPU model holds +tsc-deadline when the host has it (KVM/HVF): a
        host with it and one without make different keys, through the real guest_cpu_model."""
        from exec_sandbox.platform_utils import HostArch
        from exec_sandbox.system_probes import AccelType

        keys = []
        for tsc_deadline in (True, False):
            with (
                patch("exec_sandbox.memory_snapshot_manager.detect_host_arch", return_value=HostArch.X86_64),
                patch("exec_sandbox.memory_snapshot_manager.probe_qemu_version", new_callable=AsyncMock) as qemu,
                patch("exec_sandbox.memory_snapshot_manager.detect_accel_type", new_callable=AsyncMock) as accel,
                patch("exec_sandbox.qemu_cmd.check_tsc_deadline", new_callable=AsyncMock, return_value=tsc_deadline),
                patch("exec_sandbox.qemu_cmd.detect_host_x86_vendor", return_value="GenuineIntel"),
            ):
                qemu.return_value = (11, 1, 3)
                accel.return_value = AccelType.KVM
                keys.append(await manager.compute_cache_key(VmSpec(language=Language.PYTHON), l2_path=None))
        assert keys[0] != keys[1]

    async def test_different_network_different_key(self, manager: MemorySnapshotManager) -> None:
        """Different allow_network values produce different keys (topology differs)."""
        key_nonet = await self._compute_key(manager, allow_network=False)
        key_net = await self._compute_key(manager, allow_network=True)
        assert key_nonet != key_net

    async def test_key_format(self, manager: MemorySnapshotManager) -> None:
        """Key format: l1-{language}-v{major.minor}-{full SHA-256 hex digest}."""
        key = await self._compute_key(manager, language=Language.PYTHON)
        assert _KEY.fullmatch(key), key

    async def test_package_order_insensitive(self, manager: MemorySnapshotManager) -> None:
        """Package order should not affect cache key (sorted internally)."""
        key_ab = await self._compute_key(manager, packages=["a", "b"])
        key_ba = await self._compute_key(manager, packages=["b", "a"])
        assert key_ab == key_ba

    async def test_none_qemu_version(self, manager: MemorySnapshotManager) -> None:
        """An unknown QEMU version gives a stable key of its own."""
        key = await self._compute_key(manager, qemu_version=None)
        assert _KEY.fullmatch(key), key
        assert key == await self._compute_key(manager, qemu_version=None)
        assert key != await self._compute_key(manager, qemu_version=(9, 2, 0))


class TestVmstateDigest:
    """The L1 integrity digest: every non-zero byte, its offset and the size count; holes are
    not read. Private level: check_cache only compares two digests, so these properties of
    the digest are not visible through it."""

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

    @pytest.mark.parametrize(
        ("writes", "size", "digest"),
        [
            # SHA-256 of the size (7), then of offset 0 and the 7 bytes: _VMSTATE_DIGEST
            pytest.param({0: _VMSTATE}, 7, _VMSTATE_DIGEST, id="one-block"),
            # SHA-256 of the size (1) only: a zero block is not hashed
            pytest.param(
                {0: b"\x00"}, 1, "cd2662154e6d76b2b2b92e70c0cac3ccf534f9b74eb5b89819ec509083d00a50", id="zero-block"
            ),
            # SHA-256 of the size (4 blocks), then of offset 65536 and that whole block ("x" at 4464)
            pytest.param(
                {70_000: b"x"},
                4 * 64 * 1024,
                "1dc0d2fecb7719f54f91c162c0923575148ea635f0cc5d112081a69b928ea7f9",
                id="sparse",
            ),
        ],
    )
    def test_golden_vectors(self, tmp_path: Path, writes: dict[int, bytes], size: int, digest: str) -> None:
        """The digest of a file, worked out from its definition (sizes and offsets as 8 bytes
        big-endian)."""
        assert _vmstate_digest(self._sparse(tmp_path / "vmstate", writes, size=size)) == digest

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

    async def _populate_cache(self, manager: MemorySnapshotManager, *, empty: bool) -> Path:
        """An entry of the Python spec (requires mock_system_probes fixture): _VMSTATE with
        its digest, or an empty vmstate with no digest."""
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(b"" if empty else _VMSTATE)
        vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
        meta.write_text(json.dumps({"test": True} if empty else {"test": True, "vmstate_sha256": _VMSTATE_DIGEST}))
        return vmstate

    async def test_zero_size_cleaned_up(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Zero-size vmstate is cleaned up and returns miss."""
        vmstate = await self._populate_cache(manager, empty=True)
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None
        assert not vmstate.exists()

    async def test_invalid_metadata_cleaned_up(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Invalid metadata causes cleanup and returns miss."""
        vmstate = await self._populate_cache(manager, empty=False)
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
        """A vmstate without its metadata is a miss, and the check does not remove it."""
        vmstate = await self._populate_cache(manager, empty=False)
        # Delete metadata only
        meta = vmstate.parent / f"{vmstate.name[: -len('.vmstate')]}.vmstate.meta"
        meta.unlink()

        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None
        assert vmstate.exists()

    async def test_no_sha256_in_metadata_still_hits(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Metadata without vmstate_sha256: no digest check, and the entry is a hit."""
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(b"fake vmstate data")
        vmstate.with_suffix(".home").write_bytes(b"home disk")
        meta.write_text(json.dumps({"test": True}))  # No vmstate_sha256 field

        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result == vmstate


@pytest.mark.usefixtures("mock_system_probes")
class TestCheckCacheAtime:
    async def test_atime_update_refused_is_still_a_hit(self, manager: MemorySnapshotManager) -> None:
        """The atime update of an entry this process may not write (another user's file)
        raises PermissionError: the hit stands (the atime only orders eviction)."""
        key, files = await TestRestore._entry(manager)
        with patch(
            "exec_sandbox.memory_snapshot_manager.os.utime", side_effect=PermissionError(13, "Permission denied")
        ):
            assert await manager.check_cache(key) == files[0]

    async def test_atime_update_sets_the_access_time_and_keeps_the_stamp(self, manager: MemorySnapshotManager) -> None:
        """A hit moves the access time (eviction removes the oldest first); the mtime stays,
        so the version a restore checked (its stamp) is still the file a refusal removes."""
        key, files = await TestRestore._entry(manager)
        os.utime(files[0], (1_000_000, 2_000_000))
        assert await manager.check_cache(key) == files[0]
        stat = files[0].stat()
        assert stat.st_atime > 1_000_000
        assert stat.st_mtime == 2_000_000

    async def test_entry_evicted_after_its_check_is_not_made_again(self, manager: MemorySnapshotManager) -> None:
        """An eviction between the check and the atime update: no empty vmstate appears in
        its place (Path.touch would make one: a zero-size entry until the next check). The
        window has no public hook: the test replaces the private check step to evict there."""
        key, files = await TestRestore._entry(manager)

        async def checked_then_evicted(vmstate: Path, *_args: object, **_kwargs: object) -> None:
            for path in files:
                path.unlink()

        with patch.object(manager, "_entry_problem", new=checked_then_evicted):
            await manager.check_cache(key)
        assert not files[0].exists()


class TestHomeDiskEntry:
    """An L1 entry is vmstate + metadata + RAM template + home disk: all or nothing."""

    async def test_missing_home_disk_is_a_miss_and_entry_removed(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        spec = VmSpec(language=Language.PYTHON, memory_mb=512)
        key = await manager.compute_cache_key(spec, l2_path=None)
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(_VMSTATE)
        meta.write_text(json.dumps({"vmstate_sha256": _VMSTATE_DIGEST}))

        assert await manager.check_cache(key) is None
        assert not vmstate.exists()
        assert not meta.exists()

    async def test_failed_check_removes_every_file_off_the_loop_and_the_vmstate_last(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """An entry that fails its integrity check takes its large sidecars with it (eviction
        lists only *.vmstate), in a worker thread (the .ram file is the RAM size); the miss
        returns once they are gone. No save of a key starts while its vmstate exists
        (CacheEntry.try_saving): the vmstate goes last, so a removal never deletes the sidecars of a
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
        vmstate.write_bytes(_VMSTATE)
        vmstate.with_suffix(".home").write_bytes(b"home disk")
        (manager.cache_dir / f"{key}.vmstate.meta").write_text(
            json.dumps({"vmstate_sha256": _VMSTATE_DIGEST, "use_template": True})
        )

        assert await manager.check_cache(key) is None
        # Only the lock files of the removal stay (lock files are never deleted: files.locks)
        left = sorted(p.name for p in manager.cache_dir.iterdir() if p.name.startswith(key))
        assert left == [f"{key}.vmstate.lock", f"{key}.vmstate.use.lock"]

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
        files[0].write_bytes(_VMSTATE)
        files[1].write_text(json.dumps({"vmstate_sha256": _VMSTATE_DIGEST}))
        files[2].write_bytes(b"ram")
        files[3].write_bytes(b"home")
        return key, files

    @staticmethod
    def _layer(manager: MemorySnapshotManager) -> Path:
        """The L2 layer the restored VM runs on (an entry of another cache)."""
        return manager.cache_dir.parent / "l2.qcow2"

    @staticmethod
    async def _restore(manager: MemorySnapshotManager, key: str, reservation: object) -> object:
        """restore() as the scheduler calls it: with its lease on the L2 layer, and the L1
        entry leased before (None: a miss, no restore)."""
        layer = TestRestore._layer(manager)
        layer.write_bytes(b"l2")
        lease = await CacheEntry(layer).lease()
        assert lease is not None
        entry = await manager.lease(key)
        if entry is None:
            lease.release()
            return None
        try:
            return await manager.restore(
                TestRestore._SPEC,
                entry=entry,
                tenant_id="tenant",
                task_id="task",
                snapshot_drive=lease,
                expose_ports=None,
                reservation=reservation,  # type: ignore[arg-type]
            )
        finally:
            lease.release()

    async def test_hit_restores_the_entry(self, manager: MemorySnapshotManager, images_dir: Path) -> None:
        key, files = await self._entry(manager)
        vm, reservation = MagicMock(), MagicMock()
        with patch.object(manager.vm_manager, "restore_vm", new=AsyncMock(return_value=vm)) as restore_vm:
            assert await self._restore(manager, key, reservation) is vm
        restore_vm.assert_awaited_once_with(
            self._SPEC,
            tenant_id="tenant",
            task_id="task",
            vmstate_path=files[0],
            base_image=images_dir / f"python-3.14-base-{get_arch_name()}.qcow2",
            snapshot_drive=self._layer(manager),
            cache_leases=ANY,
            expose_ports=None,
            reservation=reservation,
        )
        leases = restore_vm.await_args.kwargs["cache_leases"]
        assert [lease.path for lease in leases] == [files[0], self._layer(manager)]

    async def test_entry_is_leased_until_the_vm_has_its_own_share(self, manager: MemorySnapshotManager) -> None:
        """No eviction removes the entry from its check to the VM launch (the VM takes its
        own share of the lease there); restore() releases its own lease after."""
        key, files = await self._entry(manager)
        evictable_at_launch: list[bool] = []

        async def restore_vm(*_args: object, **_kwargs: object) -> MagicMock:
            evictable_at_launch.append(evictable(files[0]))
            return MagicMock()

        with patch.object(manager.vm_manager, "restore_vm", new=restore_vm):
            await self._restore(manager, key, MagicMock())
        assert evictable_at_launch == [False]
        assert evictable(files[0])
        assert not leased_here(files[0])  # released, not only closed by the garbage collector

    async def test_no_lease_without_an_entry(self, manager: MemorySnapshotManager) -> None:
        """No entry, or one evicted between the lookup and the lease: no lease (the caller
        cold boots), and no lock file is made for a key that was never saved."""
        never_saved = await manager.compute_cache_key(VmSpec(language=Language.JAVASCRIPT, memory_mb=512), l2_path=None)
        assert await manager.lease(never_saved) is None
        assert list(manager.cache_dir.iterdir()) == []

        key, files = await self._entry(manager)
        assert await manager.check_cache(key) == files[0]
        assert await manager.evict_oldest() is not None
        assert await manager.lease(key) is None

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

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(PermissionError(13, "Permission denied"), id="raw-eacces"),
            pytest.param(
                ConfigError(
                    "no qemu-vm access", code=ErrorCode.HOST_DEPENDENCY, context=None, hint=None, instance_id=None
                ),
                id="host-dependency",
            ),
            pytest.param(
                InternalError("bug", code=ErrorCode.INTERNAL, context=None, hint=None, instance_id=None), id="internal"
            ),
        ],
    )
    async def test_any_failure_cold_boots_and_keeps_the_entry(
        self, manager: MemorySnapshotManager, error: Exception
    ) -> None:
        """Not only a TransientError: every failed restore gives None (the caller cold
        boots) and keeps a good entry (a host problem is no refusal of the saved state)."""
        key, files = await self._entry(manager)
        with patch.object(manager.vm_manager, "restore_vm", new=AsyncMock(side_effect=error)):
            assert await self._restore(manager, key, MagicMock()) is None
        assert all(f.exists() for f in files)

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

    async def test_refused_entry_another_process_runs_on_stays(self, manager: MemorySnapshotManager) -> None:
        """QEMU refused the entry here, but a VM of another process runs on it (its lease):
        the removal waits for a later check, as an eviction would."""
        key, files = await self._entry(manager)
        error = TransientError(
            "guest did not answer", code=ErrorCode.GUEST_SILENT_AFTER_RESTORE, context=None, hint=None, instance_id=None
        )
        with (
            leased_in_child(files[0]),
            patch.object(manager.vm_manager, "restore_vm", new=AsyncMock(side_effect=error)),
        ):
            assert await self._restore(manager, key, MagicMock()) is None
            assert all(f.exists() for f in files)
            assert not leased_here(files[0])

    async def test_broken_entry_another_process_runs_on_stays(self, manager: MemorySnapshotManager) -> None:
        """check_cache finds the entry broken while a VM of another process runs on it: a
        miss, and the entry stays until no lease holds it."""
        key, files = await self._entry(manager)
        files[1].write_text(json.dumps({"vmstate_sha256": "0" * 64}))  # fails the check
        with leased_in_child(files[0]):
            assert await manager.check_cache(key) is None
            assert all(f.exists() for f in files)
        assert await manager.check_cache(key) is None
        assert not files[0].exists()

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


@pytest.mark.usefixtures("mock_system_probes")
class TestHostOwnsEntryFiles:
    """The host makes every L1 file QEMU writes (the RAM template before the save VM boots,
    the tmp vmstate before the migration) and gives qemu-vm access by ACL: "rw" while the
    save VM writes, "r" once published. QEMU never makes a file in the cache folder, so a
    qemu-vm save works and any QEMU identity of the host can restore the entry."""

    _SPEC = VmSpec(language=Language.PYTHON, packages=(), memory_mb=512, cpu_cores=1)

    @pytest.mark.parametrize("memory", [False, True], ids=["cache", "memory"])
    async def test_ram_template_made_and_granted_before_the_save_vm_boots(
        self, manager: MemorySnapshotManager, live_ram_root: Path, request: pytest.FixtureRequest, memory: bool
    ) -> None:
        """The RAM file exists (0600, the host's, the RAM size) and qemu-vm has rw before
        the VM starts: in a new folder in memory when the host has a memory file system,
        else at {key}.ram. A save that publishes nothing removes it (and its folder)."""
        if memory:
            request.getfixturevalue("in_memory")
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        ram_path = manager.cache_dir / f"{key}.ram"
        stale = b"stale template of a crashed save"
        ram_path.write_bytes(stale)
        seen: dict[str, object] = {}
        grants: list[tuple[list[Path], str]] = []

        async def grant(paths: list[Path], *, perms: str) -> bool:
            grants.append((list(paths), perms))
            return True

        @contextlib.asynccontextmanager
        async def ephemeral_vm(*_args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
            live_ram = Path(cast("str", kwargs["mem_path"]))
            st = live_ram.stat()
            with live_ram.open("rb") as ram:
                head = ram.read(len(stale))
            seen.update(
                live_ram=live_ram,
                mode=stat.S_IMODE(st.st_mode),
                uid=st.st_uid,
                size=st.st_size,
                fresh=head == bytes(len(stale)),  # a new sparse file (ext4 can reuse the old inode number)
                grants=list(grants),
            )
            vm = MagicMock()
            vm.channel.send_request = AsyncMock(return_value=MagicMock(status="error"))  # REPL warm fails
            yield vm

        with (
            patch("exec_sandbox.memory_snapshot_manager.probe_qemu_as_qemu_vm", AsyncMock(return_value=True)),
            patch("exec_sandbox.memory_snapshot_manager.grant_qemu_vm_acl", grant),
            patch.object(manager.vm_manager, "ephemeral_vm", ephemeral_vm),
        ):
            task = await manager.schedule_background_save(self._SPEC, key=key, snapshot_drive=None)
            assert task is not None
            await task

        live_ram = cast("Path", seen.pop("live_ram"))
        assert seen == {
            "mode": 0o600,
            "uid": os.getuid(),
            "size": 512 * 1024 * 1024,
            "fresh": True,
            "grants": [([live_ram], "rw")],
        }
        if memory:  # a folder of its own, under the memory root
            assert (live_ram.name, live_ram.parent.parent) == (
                ram_path.name,
                live_ram_root / f"exec-sandbox-{os.geteuid()}",
            )
        else:
            assert live_ram == ram_path
        assert not ram_path.exists()
        assert not live_ram.exists()
        assert _live_folders(live_ram_root) == []

    async def test_save_keeps_the_l2_layer_leased_after_the_caller_lets_go(
        self, manager: MemorySnapshotManager, tmp_path: Path
    ) -> None:
        """The scheduler releases its lease on the L2 layer when its request ends; the
        save task boots its VM on the layer later. The task holds its own share until it
        ends (the VM gets it in cache_leases), so no eviction removes the layer first."""
        layer = tmp_path / "l2.qcow2"
        layer.write_bytes(b"l2")
        key = await manager.compute_cache_key(self._SPEC, l2_path=layer)
        boot = asyncio.Event()
        seen: dict[str, object] = {}

        @contextlib.asynccontextmanager
        async def ephemeral_vm(*_args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
            leases = cast("tuple[EntryLease, ...]", kwargs["cache_leases"])
            seen.update(drive=kwargs["snapshot_drive"], leases=[lease.path for lease in leases])
            vm = MagicMock()
            vm.channel.send_request = AsyncMock(return_value=MagicMock(status="error"))  # REPL warm fails
            yield vm

        async def boot_later() -> bool:
            await boot.wait()
            return False

        lease = await CacheEntry(layer).lease()
        assert lease is not None
        with (
            patch("exec_sandbox.memory_snapshot_manager.probe_qemu_as_qemu_vm", boot_later),
            patch.object(manager.vm_manager, "ephemeral_vm", ephemeral_vm),
        ):
            task = await manager.schedule_background_save(self._SPEC, key=key, snapshot_drive=lease)
            assert task is not None
            lease.release()  # the request is over
            assert not evictable(layer)  # the task's share
            boot.set()
            await task
        assert seen == {"drive": layer, "leases": [layer]}
        assert evictable(layer)

    async def test_grant_failure_boots_no_vm_and_leaves_no_file(self, manager: MemorySnapshotManager) -> None:
        """qemu-vm cannot get the RAM file (no setfacl, a folder it cannot traverse): the save
        fails before any VM starts (no QEMU exit 1 at boot), and the file goes."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        ram_path = manager.cache_dir / f"{key}.ram"
        ephemeral_vm = MagicMock()
        with (
            patch("exec_sandbox.memory_snapshot_manager.probe_qemu_as_qemu_vm", AsyncMock(return_value=True)),
            patch("exec_sandbox.memory_snapshot_manager.grant_qemu_vm_acl", AsyncMock(return_value=False)),
            patch.object(manager.vm_manager, "ephemeral_vm", ephemeral_vm),
        ):
            task = await manager.schedule_background_save(self._SPEC, key=key, snapshot_drive=None)
            assert task is not None
            await task

        ephemeral_vm.assert_not_called()
        assert not ram_path.exists()

    async def test_qemu_vm_save_writes_host_files_and_publishes_read_only(
        self, manager: MemorySnapshotManager, tmp_path: Path
    ) -> None:
        """As qemu-vm: the migration target exists before `migrate` (0600, the host's) with
        rw for qemu-vm; at publish qemu-vm keeps r on the vmstate and the RAM template."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        ram_path = manager.cache_dir / f"{key}.ram"
        tmp_vmstate = manager.cache_dir / f"{key}.vmstate.tmp"
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")
        grants: list[tuple[list[Path], str]] = []
        at_migrate: dict[str, object] = {}

        async def grant(paths: list[Path], *, perms: str) -> bool:
            grants.append((list(paths), perms))
            return True

        async def write_vmstate(path: Path, **kwargs: object) -> None:
            st = path.stat()
            at_migrate.update(path=path, mode=stat.S_IMODE(st.st_mode), uid=st.st_uid, grants=list(grants))
            await _write_vmstate(path, **kwargs)

        with (
            patch("exec_sandbox.memory_snapshot_manager.grant_qemu_vm_acl", grant),
            _template_save(manager, vm=_save_vm(home_disk=home_disk, qemu_vm=True), migrate=write_vmstate),
        ):
            await _save(manager, self._SPEC)

        assert at_migrate == {
            "path": tmp_vmstate,
            "mode": 0o600,
            "uid": os.getuid(),
            "grants": [([ram_path], "rw"), ([tmp_vmstate], "rw")],
        }
        assert grants == [([ram_path], "rw"), ([tmp_vmstate], "rw"), ([tmp_vmstate, ram_path], "r")]
        assert await manager.check_cache(key) == manager.cache_dir / f"{key}.vmstate"

    async def test_publish_grant_failure_publishes_nothing(
        self, manager: MemorySnapshotManager, tmp_path: Path
    ) -> None:
        """qemu-vm keeps write access when "r" cannot be set: the entry is not published
        (any QEMU could change a shared template), and no file of it stays."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")

        async def grant(_paths: list[Path], *, perms: str) -> bool:
            return perms == "rw"

        with (
            patch("exec_sandbox.memory_snapshot_manager.grant_qemu_vm_acl", grant),
            _template_save(manager, vm=_save_vm(home_disk=home_disk, qemu_vm=True), migrate=_write_vmstate),
        ):
            await _save(manager, self._SPEC)

        assert _files_of(manager, key) == []

    async def test_metadata_has_no_group_or_other_access(self, manager: MemorySnapshotManager, tmp_path: Path) -> None:
        """Under `sg qemu-vm` and umask 002, open() made the metadata 0664 of group qemu-vm: a
        QEMU as qemu-vm could change the digest each restore checks the vmstate against. It is
        0600, also where a failed save left one (here 0666)."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        vmstate_path = manager.cache_dir / f"{key}.vmstate"
        meta_path = manager.cache_dir / f"{key}.vmstate.meta"
        meta_path.write_text("left by a failed save")
        meta_path.chmod(0o666)
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")

        old = os.umask(0o002)
        try:
            with _template_save(manager, vm=_save_vm(home_disk=home_disk, qemu_vm=False), migrate=_write_vmstate):
                await _save(manager, self._SPEC)
        finally:
            os.umask(old)

        assert await manager.check_cache(key) == vmstate_path  # the new metadata, with the vmstate's digest
        assert {p.name: stat.S_IMODE(p.stat().st_mode) for p in (vmstate_path, meta_path)} == {
            vmstate_path.name: 0o600,
            meta_path.name: 0o600,
        }


@pytest.mark.usefixtures("mock_system_probes")
class TestTemplateRamInMemory:
    """With a memory file system (Linux /dev/shm), the save VM's RAM is a file there, not
    in the cache folder: on a disk, writeback of the RAM the guest keeps writing stalls the
    VM. The save writes the RAM to the cache once, then frees it."""

    _SPEC = VmSpec(language=Language.PYTHON, packages=(), memory_mb=512, cpu_cores=1)

    @pytest.mark.usefixtures("in_memory")
    async def test_ram_in_memory_is_published_to_the_cache_and_freed(
        self, manager: MemorySnapshotManager, live_ram_root: Path, tmp_path: Path
    ) -> None:
        """The VM maps a file under the memory root; the published entry has the same RAM
        in {key}.ram (a host file, "r" for qemu-vm); the folder in memory is gone at the end."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        vmstate_path = manager.cache_dir / f"{key}.vmstate"
        ram_path = manager.cache_dir / f"{key}.ram"
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")
        grants: list[tuple[list[Path], str]] = []
        seen: dict[str, Path] = {}

        async def grant(paths: list[Path], *, perms: str) -> bool:
            grants.append((list(paths), perms))
            return True

        @contextlib.asynccontextmanager
        async def ephemeral_vm(*_args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
            seen["live_ram"] = live_ram = Path(cast("str", kwargs["mem_path"]))
            with live_ram.open("r+b") as ram:  # the guest writes its RAM
                ram.write(b"\x01" * _BLOCK)
                ram.seek(5 * _BLOCK + 7)
                ram.write(b"guest page")
            vm = MagicMock()
            vm.qmp_socket = Path("/tmp/fake.sock")
            vm.use_qemu_vm_user = True
            vm.workdir.home_disk = home_disk
            vm.channel.send_request = AsyncMock(return_value=WarmReplAckMessage(language="python", status="ok"))
            yield vm

        async def write_vmstate(path: Path, **_kwargs: object) -> None:
            path.write_bytes(b"device state")

        with (
            patch("exec_sandbox.memory_snapshot_manager.probe_qemu_as_qemu_vm", AsyncMock(return_value=True)),
            patch("exec_sandbox.memory_snapshot_manager.grant_qemu_vm_acl", grant),
            patch.object(manager.vm_manager, "ephemeral_vm", ephemeral_vm),
            patch("exec_sandbox.memory_snapshot_manager.MigrationClient") as mock_client,
        ):
            client = AsyncMock()
            client.save_snapshot.side_effect = write_vmstate
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = client
            task = await manager.schedule_background_save(self._SPEC, key=key, snapshot_drive=None)
            assert task is not None
            await task

        live_ram = seen["live_ram"]
        assert live_ram.is_relative_to(live_ram_root)
        assert await manager.check_cache(key) == vmstate_path
        st = ram_path.stat()
        assert (st.st_size, stat.S_IMODE(st.st_mode)) == (512 * 1024 * 1024, 0o600)
        with ram_path.open("rb") as ram:  # zeros elsewhere (test_publish_writes_the_non_zero_blocks_only)
            assert ram.read(_BLOCK) == b"\x01" * _BLOCK
            ram.seek(5 * _BLOCK)
            assert ram.read(_BLOCK) == bytes(7) + b"guest page" + bytes(_BLOCK - 17)
        tmp_vmstate = vmstate_path.parent / f"{vmstate_path.name}.tmp"
        assert grants == [([live_ram], "rw"), ([tmp_vmstate], "rw"), ([tmp_vmstate, ram_path], "r")]
        assert _live_folders(live_ram_root) == []

    @pytest.mark.usefixtures("in_memory")
    async def test_ram_removed_from_memory_during_the_save_is_still_published(
        self, manager: MemorySnapshotManager, live_ram_root: Path, tmp_path: Path
    ) -> None:
        """systemd-logind (RemoveIPC=yes, its default) removes every /dev/shm file of a user
        that is not a system user when its last login session ends, here during the save (as
        on EC2 at 04:21:12, user@1000 stopped). The fd the save holds keeps the RAM pages: the
        entry is still published, with the guest's RAM."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        vmstate_path = manager.cache_dir / f"{key}.vmstate"
        ram_path = manager.cache_dir / f"{key}.ram"
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")
        user_root = live_ram_root / f"exec-sandbox-{os.geteuid()}"

        @contextlib.asynccontextmanager
        async def ephemeral_vm(*_args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
            with Path(cast("str", kwargs["mem_path"])).open("r+b") as ram:  # the guest writes its RAM
                ram.seek(3 * _BLOCK)
                ram.write(b"guest page")
            vm = MagicMock()
            vm.qmp_socket = Path("/tmp/fake.sock")
            vm.use_qemu_vm_user = False
            vm.workdir.home_disk = home_disk
            vm.channel.send_request = AsyncMock(return_value=WarmReplAckMessage(language="python", status="ok"))
            yield vm

        async def write_vmstate_then_logout(path: Path, **_kwargs: object) -> None:
            path.write_bytes(b"device state")
            shutil.rmtree(user_root)  # logind's RemoveIPC at the user's last logout

        with (
            patch.object(manager.vm_manager, "ephemeral_vm", ephemeral_vm),
            patch("exec_sandbox.memory_snapshot_manager.MigrationClient") as mock_client,
        ):
            client = AsyncMock()
            client.save_snapshot.side_effect = write_vmstate_then_logout
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = client
            task = await manager.schedule_background_save(self._SPEC, key=key, snapshot_drive=None)
            assert task is not None
            await task

        assert not user_root.exists()
        assert await manager.check_cache(key) == vmstate_path
        with ram_path.open("rb") as ram:
            ram.seek(3 * _BLOCK)
            assert ram.read(len(b"guest page")) == b"guest page"

    @pytest.mark.usefixtures("in_memory")
    @pytest.mark.parametrize("where", ["making-it", "removing-the-entry"])
    async def test_ram_in_memory_goes_when_the_save_is_cancelled(
        self, manager: MemorySnapshotManager, live_ram_root: Path, where: str
    ) -> None:
        """A cancel while the RAM file is made, or while a failed save's entry is removed:
        the fd on the RAM file and its folder in memory are gone once the task has ended (an
        open fd keeps up to the guest RAM in tmpfs until the process ends). The cancel must
        come while a worker-thread job runs: the test holds the private step of that job until
        the task has the cancel."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        loop = asyncio.get_running_loop()
        in_job = asyncio.Event()
        release = threading.Event()
        made: list[int] = []
        real_new_live_ram = msm._new_live_ram
        real_open_live_ram_dir = msm.open_live_ram_dir
        real_remove_entry = manager._remove_entry

        def open_live_ram_dir(*, size: int) -> Any:
            live_dir = real_open_live_ram_dir(size=size)
            if where == "making-it":  # the cancel comes before the RAM file and its fd are made
                loop.call_soon_threadsafe(in_job.set)
                release.wait(5)
            return live_dir

        def new_live_ram(*args: Any, **kwargs: Any) -> Any:
            live_ram = real_new_live_ram(*args, **kwargs)
            made.append(cast("int", live_ram.fd))
            return live_ram

        def remove_entry(path: Path) -> None:
            if where == "removing-the-entry":
                loop.call_soon_threadsafe(in_job.set)
                release.wait(5)
            real_remove_entry(path)

        @contextlib.asynccontextmanager
        async def boot_fails(*_args: object, **_kwargs: object) -> AsyncIterator[MagicMock]:
            raise TransientError("boot", code=ErrorCode.QEMU_CRASHED, context=None, hint=None, instance_id=None)
            yield MagicMock()  # pragma: no cover

        with (
            patch.object(msm, "open_live_ram_dir", open_live_ram_dir),
            patch.object(msm, "_new_live_ram", new_live_ram),
            patch.object(manager, "_remove_entry", remove_entry),
            patch.object(manager.vm_manager, "ephemeral_vm", boot_fails),
        ):
            task = await manager.schedule_background_save(self._SPEC, key=key, snapshot_drive=None)
            assert task is not None
            await in_job.wait()
            task.cancel()
            await asyncio.sleep(0)  # the task takes the cancel while the job still runs
            release.set()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert task.cancelled()
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(made[0])
        assert _live_folders(live_ram_root) == []

    @pytest.mark.usefixtures("in_memory")
    async def test_cancel_while_publishing_waits_for_the_copy(
        self, manager: MemorySnapshotManager, live_ram_root: Path, tmp_path: Path
    ) -> None:
        """A cancel while the RAM is copied to the cache (a stop, the end of asyncio.run): the
        copy ends before the fd is closed. Else the job read on through a closed fd, whose
        number the next open takes: another file's data in the template. The test holds the
        private copy step until the cancel came."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")
        loop = asyncio.get_running_loop()
        in_job = asyncio.Event()
        release = threading.Event()
        same_file: list[bool] = []
        real_publish_ram = msm._publish_ram

        def publish_ram(stop: threading.Event, *, src_fd: int, dst: Path) -> None:
            inode = os.fstat(src_fd).st_ino
            loop.call_soon_threadsafe(in_job.set)
            release.wait(5)  # the cancel comes while the copy runs
            try:
                same_file.append(os.fstat(src_fd).st_ino == inode)
            except OSError:
                same_file.append(False)
            real_publish_ram(stop, src_fd=src_fd, dst=dst)

        @contextlib.asynccontextmanager
        async def ephemeral_vm(*_args: object, **_kwargs: object) -> AsyncIterator[MagicMock]:
            vm = MagicMock()
            vm.qmp_socket = Path("/tmp/fake.sock")
            vm.use_qemu_vm_user = False
            vm.workdir.home_disk = home_disk
            vm.channel.send_request = AsyncMock(return_value=WarmReplAckMessage(language="python", status="ok"))
            yield vm

        async def write_vmstate(path: Path, **_kwargs: object) -> None:
            path.write_bytes(b"device state")

        with (
            patch.object(msm, "_publish_ram", publish_ram),
            patch.object(manager.vm_manager, "ephemeral_vm", ephemeral_vm),
            patch("exec_sandbox.memory_snapshot_manager.MigrationClient") as mock_client,
        ):
            client = AsyncMock()
            client.save_snapshot.side_effect = write_vmstate
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = client
            task = await manager.schedule_background_save(self._SPEC, key=key, snapshot_drive=None)
            assert task is not None
            await in_job.wait()
            task.cancel()
            done, _ = await asyncio.wait([task], timeout=0.1)  # a bound: the task cannot end first
            assert not done  # the cancelled save waits for the copy
            release.set()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert same_file == [True]
        assert task.cancelled()
        assert _live_folders(live_ram_root) == []

    @pytest.mark.usefixtures("in_memory")
    async def test_cancel_stops_a_long_copy_at_its_next_block(
        self, manager: MemorySnapshotManager, live_ram_root: Path, tmp_path: Path
    ) -> None:
        """A cancel during the copy of a large RAM (GBs: seconds) ends it at its next 64 KiB
        block, not at its end; the fd closes only after the copy has stopped. Here 200
        blocks: the write of the first one waits until the cancel has set the job's stop
        event. The test wraps the private copy step to get that event and to see the fd at
        its end."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")
        loop = asyncio.get_running_loop()
        copying = asyncio.Event()
        stops: list[threading.Event] = []
        written: list[int] = []
        fd_open_at_end: list[bool] = []
        real_publish_ram = msm._publish_ram

        def held_pwrite_all(fd: int, data: bytes, offset: int) -> None:
            written.append(offset)
            if len(written) == 1:  # the cancel comes while the first block is written
                loop.call_soon_threadsafe(copying.set)
                stops[0].wait(5)
            pwrite_all(fd, data, offset)

        def publish_ram(stop: threading.Event, *, src_fd: int, dst: Path) -> None:
            stops.append(stop)
            real_publish_ram(stop, src_fd=src_fd, dst=dst)
            fd_open_at_end.append(os.fstat(src_fd).st_size > 0)

        @contextlib.asynccontextmanager
        async def ephemeral_vm(*_args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
            with Path(cast("str", kwargs["mem_path"])).open("r+b") as ram:  # 200 non-zero blocks
                for index in range(200):
                    ram.seek(index * _BLOCK)
                    ram.write(b"\x01")
            vm = MagicMock()
            vm.qmp_socket = Path("/tmp/fake.sock")
            vm.use_qemu_vm_user = False
            vm.workdir.home_disk = home_disk
            vm.channel.send_request = AsyncMock(return_value=WarmReplAckMessage(language="python", status="ok"))
            yield vm

        async def write_vmstate(path: Path, **_kwargs: object) -> None:
            path.write_bytes(b"device state")

        with (
            patch.object(msm, "pwrite_all", held_pwrite_all),
            patch.object(msm, "_publish_ram", publish_ram),
            patch.object(manager.vm_manager, "ephemeral_vm", ephemeral_vm),
            patch("exec_sandbox.memory_snapshot_manager.MigrationClient") as mock_client,
        ):
            client = AsyncMock()
            client.save_snapshot.side_effect = write_vmstate
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = client
            task = await manager.schedule_background_save(self._SPEC, key=key, snapshot_drive=None)
            assert task is not None
            await copying.wait()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert task.cancelled()
        assert written == [0]  # stopped after the block in progress, not after 200
        assert fd_open_at_end == [True]  # the copy's end came before the fd's close
        assert _live_folders(live_ram_root) == []
        assert await manager.check_cache(key) is None  # the partial copy is gone

    @pytest.mark.usefixtures("in_memory")
    async def test_cancel_stays_a_cancel_when_the_files_cannot_be_removed(
        self, manager: MemorySnapshotManager, live_ram_root: Path
    ) -> None:
        """A cancelled save whose removal fails (EIO): the error is logged, the cancel goes on
        (the task ends cancelled, as a stop expects), and the RAM in memory goes. The disk
        fault is made by the private removal step."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        booting = asyncio.Event()

        @contextlib.asynccontextmanager
        async def boot_hangs(*_args: object, **_kwargs: object) -> AsyncIterator[MagicMock]:
            booting.set()
            await asyncio.Event().wait()
            yield MagicMock()  # pragma: no cover

        with (
            patch.object(manager, "_remove_entry", side_effect=OSError(errno.EIO, "Input/output error")),
            patch.object(manager.vm_manager, "ephemeral_vm", boot_hangs),
        ):
            task = await manager.schedule_background_save(self._SPEC, key=key, snapshot_drive=None)
            assert task is not None
            await booting.wait()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert task.cancelled()
        assert _live_folders(live_ram_root) == []

    @pytest.mark.usefixtures("in_memory")
    async def test_ram_in_memory_goes_when_the_entry_cannot_be_removed(
        self, manager: MemorySnapshotManager, live_ram_root: Path
    ) -> None:
        """A failed save whose entry cannot be removed (EIO, a read-only disk): the RAM in
        memory goes all the same. The disk fault is made by the private removal step, and the
        test gets the RAM fd from the private step that opens it."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        made: list[int] = []
        real_new_live_ram = msm._new_live_ram

        def new_live_ram(*args: Any, **kwargs: Any) -> Any:
            live_ram = real_new_live_ram(*args, **kwargs)
            made.append(cast("int", live_ram.fd))
            return live_ram

        @contextlib.asynccontextmanager
        async def boot_fails(*_args: object, **_kwargs: object) -> AsyncIterator[MagicMock]:
            raise TransientError("boot", code=ErrorCode.QEMU_CRASHED, context=None, hint=None, instance_id=None)
            yield MagicMock()  # pragma: no cover

        with (
            patch.object(msm, "_new_live_ram", new_live_ram),
            patch.object(manager, "_remove_entry", side_effect=OSError(errno.EIO, "Input/output error")),
            patch.object(manager.vm_manager, "ephemeral_vm", boot_fails),
        ):
            task = await manager.schedule_background_save(self._SPEC, key=key, snapshot_drive=None)
            assert task is not None
            await task  # the save logs its error
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(made[0])
        assert _live_folders(live_ram_root) == []

    async def test_publish_writes_the_non_zero_blocks_only(self, tmp_path: Path) -> None:
        """The guest's zero pages stay holes in the cache (a Python template: 68 of 192 MiB
        are not zero). A stale {key}.ram is replaced, never written in place: a restored VM
        may still map it. Private level: whether a file system reports a hole depends on it,
        so the test records the blocks the copy writes."""
        src = tmp_path / "live.ram"
        size = 6 * _BLOCK + 100
        with src.open("wb") as f:
            f.write(b"\x02" * _BLOCK)  # block 0: data
            f.write(bytes(2 * _BLOCK))  # blocks 1-2: written zeros (the guest freed them)
            f.seek(4 * _BLOCK + 3)
            f.write(b"x")  # block 4: one byte; block 3: a hole
            f.truncate(size)  # block 5 and the 100-byte tail: a hole
        dst = tmp_path / "entry.ram"
        dst.write_bytes(b"stale")
        written: list[tuple[int, int]] = []

        def recording_pwrite_all(fd: int, data: bytes, offset: int) -> None:
            written.append((offset, len(data)))
            pwrite_all(fd, data, offset)

        with (
            dst.open("rb") as mapped,  # a restored VM's view of the old template
            patch("exec_sandbox.memory_snapshot_manager.pwrite_all", recording_pwrite_all),
        ):
            with src.open("rb") as live:
                _publish_ram(threading.Event(), src_fd=live.fileno(), dst=dst)
            assert mapped.read() == b"stale"

        assert written == [(0, _BLOCK), (4 * _BLOCK, _BLOCK)]
        assert dst.read_bytes() == src.read_bytes()
        assert stat.S_IMODE(dst.stat().st_mode) == 0o600

    async def test_publish_writes_through_as_it_goes(self, tmp_path: Path) -> None:
        """The copy flushes each 32 MiB (here: each block), so the page cache it dirties
        stays small whatever the RAM size. Private level: dirty page cache is not visible to
        a test, so the test records the flushes."""
        src = tmp_path / "live.ram"
        with src.open("wb") as f:
            for value in (1, 2, 3):
                f.write(bytes([value]) * _BLOCK)
        flushed: list[int] = []
        with (
            patch("exec_sandbox.memory_snapshot_manager._PUBLISH_FLUSH_BYTES", _BLOCK),
            patch(
                "exec_sandbox.memory_snapshot_manager.flush_to_disk", lambda fd: flushed.append(os.fstat(fd).st_size)
            ),
        ):
            with src.open("rb") as live:
                _publish_ram(threading.Event(), src_fd=live.fileno(), dst=tmp_path / "entry.ram")
        assert len(flushed) == 3

    @pytest.mark.usefixtures("in_memory")
    async def test_save_removes_the_folder_of_a_dead_process_and_keeps_a_live_one(
        self, manager: MemorySnapshotManager, live_ram_root: Path
    ) -> None:
        """A process that died during a save leaves its RAM file in memory: the next save
        removes its folder (lock not held). Another live save's folder stays."""
        user_root = live_ram_root / f"exec-sandbox-{os.geteuid()}"
        user_root.mkdir(mode=0o700)
        dead = user_root / "vmm-dead"
        dead.mkdir()
        (dead / ".lock").touch()
        (dead / "l1-python.ram").write_bytes(b"guest RAM")
        live = PrivateRunDir.open(user_root, name=new_folder_name())
        vms = _SaveVms(gate=_open_gate(), fail=False)
        try:
            with vms.of(manager):
                await _save(manager, self._SPEC)
            assert vms.mem_paths[0].parent.parent == user_root  # the save VM's RAM was in memory
            assert _live_folders(live_ram_root) == [live.path]
        finally:
            live.close()

    @pytest.mark.usefixtures("in_memory")
    def test_start_sweep_removes_dead_folders_without_a_save(self, live_ram_root: Path) -> None:
        """VmManager.start runs remove_dead_live_ram_dirs: the RAM file of a killed save goes
        at the next start, not only at the next save (it holds memory until then)."""
        user_root = live_ram_root / f"exec-sandbox-{os.geteuid()}"
        user_root.mkdir(mode=0o700)
        dead = user_root / "vmm-dead"
        dead.mkdir()
        (dead / ".lock").touch()
        (dead / "l1-python.ram").write_bytes(b"guest RAM")
        live = PrivateRunDir.open(user_root, name=new_folder_name())
        try:
            remove_dead_live_ram_dirs()
            assert _live_folders(live_ram_root) == [live.path]
        finally:
            live.close()

    @pytest.mark.parametrize("memory", [False, True], ids=["no-shm-folder", "no-folder-yet"])
    def test_start_sweep_makes_nothing(
        self, live_ram_root: Path, request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture, memory: bool
    ) -> None:
        """No /dev/shm folder (macOS), or no save ever ran: nothing to remove, nothing made,
        and nothing to warn about."""
        if memory:
            request.getfixturevalue("in_memory")
        with caplog.at_level("WARNING", logger="exec_sandbox"):
            remove_dead_live_ram_dirs()
        assert not live_ram_root.exists() or list(live_ram_root.iterdir()) == []
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.usefixtures("in_memory")
    def test_start_sweep_leaves_a_root_others_can_write(
        self, live_ram_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """/dev/shm is open to every user: anyone can make exec-sandbox-<uid> first and put
        folders in it. Nothing in it is removed; a warning says why."""
        user_root = live_ram_root / f"exec-sandbox-{os.geteuid()}"
        user_root.mkdir()
        user_root.chmod(0o777)
        dead = user_root / "vmm-dead"
        dead.mkdir()
        (dead / ".lock").touch()
        with caplog.at_level("WARNING", logger="exec_sandbox.memory_snapshot_files"):
            remove_dead_live_ram_dirs()
        assert dead.exists()
        assert [r.getMessage() for r in caplog.records] == ["RAM folders of stopped L1 saves stay in /dev/shm"]

    @pytest.mark.usefixtures("in_memory")
    async def test_folder_goes_when_the_ram_file_cannot_be_made(
        self, manager: MemorySnapshotManager, live_ram_root: Path
    ) -> None:
        """No RAM file (here ENOSPC in memory), no folder: the save boots no VM and leaves
        nothing in memory or in the cache."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        vms = _SaveVms(gate=_open_gate(), fail=False)
        with (
            patch(
                "exec_sandbox.memory_snapshot_manager.create_sparse",
                side_effect=OSError(errno.ENOSPC, "No space left on device"),
            ),
            vms.of(manager),
        ):
            await _save(manager, self._SPEC)
        assert vms.task_ids == []
        assert _live_folders(live_ram_root) == []
        assert _files_of(manager, key) == []

    @pytest.mark.usefixtures("in_memory")
    async def test_ram_stays_in_the_cache_without_room_in_memory(
        self, live_ram_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Less free space in memory than the RAM (a container's 64 MB /dev/shm): None, and
        nothing is made in memory. Function level: a VmSpec cannot ask for more RAM than the
        free space of the test folder."""
        assert open_live_ram_dir(size=1 << 62) is None  # more than any free space
        assert _live_folders(live_ram_root) == []
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == [
            "L1 template RAM goes to the cache folder: /dev/shm is too small, the save VM runs slower"
        ]

    @pytest.mark.usefixtures("in_memory")
    async def test_ram_stays_in_the_cache_when_memory_cannot_be_used(
        self, manager: MemorySnapshotManager, live_ram_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """/dev/shm cannot be used (its sweep gets an OSError: no access, an I/O error): the save
        goes on with its RAM in the cache folder, and leaves no folder in memory."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)

        def sweep_fails(_root: Path) -> None:
            raise OSError(errno.EACCES, os.strerror(errno.EACCES))

        vms = _SaveVms(gate=_open_gate(), fail=False)
        with vms.of(manager), patch("exec_sandbox.memory_snapshot_files.remove_dead_folders", sweep_fails):
            await _save(manager, self._SPEC)
        assert vms.mem_paths == [manager.cache_dir / f"{key}.ram"]
        assert _live_folders(live_ram_root) == []
        assert "L1 template RAM goes to the cache folder" in caplog.text

    @pytest.mark.usefixtures("in_memory")
    async def test_ram_stays_in_the_cache_when_another_user_can_change_the_root(
        self, manager: MemorySnapshotManager, live_ram_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Anyone can make /dev/shm/exec-sandbox-<uid> first: a root open to others (or of
        another user) is neither used nor swept; the save goes on in the cache folder."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        user_root = live_ram_root / f"exec-sandbox-{os.geteuid()}"
        user_root.mkdir()
        user_root.chmod(0o777)
        planted = user_root / "vmm-planted"
        planted.mkdir()
        (planted / ".lock").touch()
        vms = _SaveVms(gate=_open_gate(), fail=False)
        with vms.of(manager):
            await _save(manager, self._SPEC)
        assert vms.mem_paths == [manager.cache_dir / f"{key}.ram"]
        assert list(user_root.iterdir()) == [planted]
        assert any("has mode 0o777" in getattr(record, "error", "") for record in caplog.records)


@pytest.mark.usefixtures("mock_system_probes")
class TestSaveLock:
    """The save lock of a key ({key}.vmstate.lock): one save of a key at a time, in every
    process. TestScheduleBackgroundSaveEdgeCases has the saves that skip the key."""

    _SPEC = VmSpec(language=Language.PYTHON, memory_mb=512)

    async def test_empty_vmstate_is_saved_again(self, manager: MemorySnapshotManager) -> None:
        """An empty {key}.vmstate (a miss: it has no metadata) is no entry: the save boots its VM."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        (manager.cache_dir / f"{key}.vmstate").write_bytes(b"")
        vms = _SaveVms(gate=_open_gate(), fail=False)
        with vms.of(manager):
            await _save(manager, self._SPEC)
        assert vms.task_ids == ["l1-save-python"]

    async def test_removed_cache_folder_is_made_again(self, manager: MemorySnapshotManager) -> None:
        """A cache folder removed while the worker runs (a tmp cleaner) is made again: the
        save goes on, instead of a FileNotFoundError on every save until a restart."""
        shutil.rmtree(manager.cache_dir)
        vms = _SaveVms(gate=_open_gate(), fail=False)
        with vms.of(manager):
            await _save(manager, self._SPEC)
        assert vms.task_ids == ["l1-save-python"]
        assert manager.cache_dir.is_dir()

    async def test_stale_lock_file_does_not_stop_a_save_and_stays(self, manager: MemorySnapshotManager) -> None:
        """A lock file of an earlier process (no lock held on it) does not stop a save, and
        the lock file stays after the save (files.locks never deletes one)."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        lock = manager.cache_dir / f"{key}.vmstate.lock"
        lock.write_text("stale")
        vms = _SaveVms(gate=_open_gate(), fail=False)
        with vms.of(manager):
            await _save(manager, self._SPEC)
        assert vms.task_ids == ["l1-save-python"]
        assert lock.exists()

    @pytest.mark.parametrize("fail", [True, False], ids=["boot-fails", "warm-up-fails"])
    async def test_ended_save_frees_its_key(self, manager: MemorySnapshotManager, fail: bool) -> None:
        """A save that ended without an entry leaves its key free (no lock held, no save in
        flight): the next save of the key boots its VM again."""
        vms = _SaveVms(gate=_open_gate(), fail=fail)
        with vms.of(manager):
            await _save(manager, self._SPEC)
            await _save(manager, self._SPEC)
        assert vms.task_ids == ["l1-save-python", "l1-save-python"]


class TestEvictOldestSnapshot:
    """Tests for evict_oldest (LRU eviction)."""

    async def test_evict_removes_oldest(self, manager: MemorySnapshotManager) -> None:
        """Eviction removes the oldest snapshot by atime, and only it."""
        # Create 3 snapshots with different atimes
        for i, name in enumerate(["old", "mid", "new"]):
            vmstate = manager.cache_dir / f"{name}.vmstate"
            meta = manager.cache_dir / f"{name}.vmstate.meta"
            vmstate.write_bytes(b"data")
            meta.write_text("{}")
            # Set atime to different times (older = smaller)
            atime = time.time() - (100 - i * 10)
            os.utime(vmstate, (atime,) * 2)  # mtime too: relatime resets an atime older than mtime

        await manager.evict_oldest()

        assert not (manager.cache_dir / "old.vmstate").exists()
        assert not (manager.cache_dir / "old.vmstate.meta").exists()
        assert (manager.cache_dir / "mid.vmstate").exists()
        assert (manager.cache_dir / "new.vmstate").exists()

    async def test_evict_no_files(self, manager: MemorySnapshotManager) -> None:
        """Eviction with no entry removes nothing and returns None."""
        assert await manager.evict_oldest() is None

    async def test_evict_skips_locked_entry(self, manager: MemorySnapshotManager) -> None:
        """Eviction skips entries with an active flock and evicts the next oldest."""
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
            await manager.evict_oldest()

            # "locked" should be skipped, "unlocked" evicted instead
            assert (manager.cache_dir / "locked.vmstate").exists()
            assert not (manager.cache_dir / "unlocked.vmstate").exists()
        finally:
            fd.close()


@pytest.mark.usefixtures("mock_system_probes")
class TestStop:
    """stop() waits for the saves in flight."""

    _SPEC = VmSpec(language=Language.PYTHON, memory_mb=512)

    async def test_stop_waits_for_every_save_in_flight(self, manager: MemorySnapshotManager) -> None:
        gate = asyncio.Event()
        vms = _SaveVms(gate=gate, fail=False)
        with vms.of(manager):
            tasks = [
                await _schedule_save(manager, VmSpec(language=language, memory_mb=512))
                for language in (Language.PYTHON, Language.JAVASCRIPT)
            ]
            await vms.booted(2)
            stop = asyncio.create_task(manager.stop())
            await asyncio.sleep(0)
            assert not stop.done()  # the saves wait for the gate
            gate.set()
            await stop
        assert [task is not None and task.done() for task in tasks] == [True, True]
        await manager.stop()  # nothing in flight

    async def test_stop_waits_for_a_save_scheduled_while_it_waits(self, manager: MemorySnapshotManager) -> None:
        """A save scheduled while stop() waits (the cold boot of a new() that the Scheduler's
        stop did not stop) is waited for too: stop() returns once no save is in flight."""
        gates = {language: asyncio.Event() for language in (Language.PYTHON, Language.JAVASCRIPT)}
        booted = {language: asyncio.Event() for language in gates}

        @contextlib.asynccontextmanager
        async def ephemeral_vm(spec: VmSpec, **_kwargs: object) -> AsyncIterator[MagicMock]:
            booted[spec.language].set()
            await gates[spec.language].wait()
            vm = MagicMock()
            vm.channel.send_request = AsyncMock(return_value=MagicMock(status="error"))  # no entry
            yield vm

        with patch.object(manager.vm_manager, "ephemeral_vm", ephemeral_vm):
            first = await _schedule_save(manager, VmSpec(language=Language.PYTHON, memory_mb=512))
            await booted[Language.PYTHON].wait()
            stop = asyncio.create_task(manager.stop())
            await asyncio.sleep(0)  # stop() waits for the first save
            second = await _schedule_save(manager, VmSpec(language=Language.JAVASCRIPT, memory_mb=512))
            await booted[Language.JAVASCRIPT].wait()
            gates[Language.PYTHON].set()
            assert first is not None
            await first
            gates[Language.JAVASCRIPT].set()
            await stop

        assert second is not None
        assert second.done()

    async def test_stop_returns_after_a_cancelled_save(self, manager: MemorySnapshotManager) -> None:
        """A save task cancelled in flight: stop() returns, and does not raise its CancelledError."""
        vms = _SaveVms(gate=asyncio.Event(), fail=False)
        with vms.of(manager):
            task = await _schedule_save(manager, self._SPEC)
            assert task is not None
            await vms.booted(1)
            task.cancel()
            await manager.stop()
        assert task.cancelled()

    @pytest.mark.parametrize("end", ["no-entry", "boot-failed", "cancelled"])
    async def test_ended_save_is_not_kept(self, manager: MemorySnapshotManager, end: str) -> None:
        """The manager keeps no reference to a save task once it has ended, whatever its end:
        the ended tasks of a long-running process do not pile up."""
        gate = asyncio.Event()
        if end != "cancelled":
            gate.set()
        vms = _SaveVms(gate=gate, fail=end == "boot-failed")
        with vms.of(manager):
            task = await _schedule_save(manager, self._SPEC)
            assert task is not None
            if end == "cancelled":
                await vms.booted(1)
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await asyncio.sleep(0)  # the loop's handle that woke this test holds the task until it returns
        ended = weakref.ref(task)
        del task
        gc.collect()
        assert ended() is None


@pytest.mark.usefixtures("mock_system_probes")
class TestScheduleBackgroundSave:
    """Tests for schedule_background_save (fire-and-forget L1 creation)."""

    _SPEC = VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)

    async def test_skips_if_cache_exists(self, manager: MemorySnapshotManager) -> None:
        """A valid entry of the key: no save task, no VM."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(_VMSTATE)
        vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
        meta.write_text(json.dumps({"test": True, "vmstate_sha256": _VMSTATE_DIGEST}))
        vms = _SaveVms(gate=_open_gate(), fail=False)

        with vms.of(manager):
            assert await _schedule_save(manager, self._SPEC) is None

        assert vms.task_ids == []

    async def test_concurrent_saves_of_one_key_boot_one_vm(self, manager: MemorySnapshotManager) -> None:
        """Five concurrent saves of one key in one process: one task and one VM boot (the
        other calls find the save of the key in flight)."""
        gate = asyncio.Event()
        vms = _SaveVms(gate=gate, fail=False)
        with vms.of(manager):
            tasks = await asyncio.gather(*(_schedule_save(manager, self._SPEC) for _ in range(5)))
            started = [task for task in tasks if task is not None]
            assert len(started) == 1
            await vms.booted(1)
            gate.set()
            await started[0]
        assert vms.task_ids == ["l1-save-python"]

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
        vms = _SaveVms(gate=_open_gate(), fail=True)
        with vms.of(manager):
            task = await _schedule_save(manager, self._SPEC)
            if task is not None:
                await task

        assert (task is not None, len(vms.task_ids)) == (scheduled, int(scheduled))


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
        assert _KEY.fullmatch(key), key
        assert key == await self._compute_key(manager, packages=["日本語", "пакет", "αβγ"])

    async def test_empty_string_package(self, manager: MemorySnapshotManager) -> None:
        """An empty-string package is a distinct input: the length-prefixed encoding keeps it apart from []."""
        key_with = await self._compute_key(manager, packages=[""])
        key_without = await self._compute_key(manager, packages=[])
        assert key_with != key_without

    async def test_huge_package_list(self, manager: MemorySnapshotManager) -> None:
        """Large package list (1000 packages) still produces a fixed-length fingerprint."""
        packages = [f"pkg-{i}" for i in range(1000)]
        key = await self._compute_key(manager, packages=packages)
        assert _KEY.fullmatch(key), key

    @pytest.mark.parametrize("separator", ["|", ",", "\n"], ids=["pipe", "comma", "newline"])
    async def test_special_chars_in_packages(self, manager: MemorySnapshotManager, separator: str) -> None:
        """A pipe, a comma or a newline in a package name is part of the name, never a
        separator: the key is valid, and not the key of the name split there."""
        joined = await self._compute_key(manager, packages=[f"bar{separator}pkg"])
        assert _KEY.fullmatch(joined), joined
        assert joined != await self._compute_key(manager, packages=["bar", "pkg"])

    async def test_memory_very_large(self, manager: MemorySnapshotManager) -> None:
        """Very large memory_mb (64GB) produces valid key."""
        key = await self._compute_key(manager, memory_mb=65536)
        assert _KEY.fullmatch(key), key

    async def test_different_cpu_cores_different_key(self, manager: MemorySnapshotManager) -> None:
        """Different cpu_cores produce different keys."""
        key_1 = await self._compute_key(manager, cpu_cores=1)
        key_4 = await self._compute_key(manager, cpu_cores=4)
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


# ============================================================================
# Torture Tests — check_cache Edge Cases
# ============================================================================


class TestCheckCacheEdgeCases:
    """Edge cases and corruption scenarios for check_cache."""

    async def _populate(
        self, manager: MemorySnapshotManager, *, data: bytes = _VMSTATE, meta_dict: dict[str, Any] | None = None
    ) -> tuple[Path, Path]:
        """Create an entry of the Python spec: the vmstate `data`, and `meta_dict` as its
        metadata (default: the digest of _VMSTATE). Returns (vmstate_path, meta_path)."""
        key = await manager.compute_cache_key(
            VmSpec(language=Language.PYTHON, packages=(), memory_mb=512), l2_path=None
        )
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(data)
        vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
        meta.write_text(
            json.dumps({"test": True, "vmstate_sha256": _VMSTATE_DIGEST} if meta_dict is None else meta_dict)
        )
        return vmstate, meta

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
        """Metadata is valid JSON but not a dict (array) → typed parse rejects → miss, entry removed."""
        vmstate, meta = await self._populate(manager)
        meta.write_text("[1, 2, 3]")
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None
        assert not vmstate.exists()
        assert not meta.exists()

    async def test_metadata_is_json_string(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Metadata is valid JSON but a bare string → typed parse rejects → miss, entry removed."""
        vmstate, meta = await self._populate(manager)
        meta.write_text('"just a string"')
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None
        assert not vmstate.exists()

    async def test_metadata_sha256_is_not_string(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Metadata has vmstate_sha256 as int → coerced to str, SHA-256 mismatch → miss, entry removed."""
        vmstate, _meta = await self._populate(manager, meta_dict={"vmstate_sha256": 12345})
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None
        assert not vmstate.exists()

    async def test_metadata_sha256_empty_string(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Metadata has vmstate_sha256 = "" → falsy, SHA-256 check skipped, still hits."""
        vmstate, _meta = await self._populate(manager, data=b"not the digest's bytes", meta_dict={"vmstate_sha256": ""})
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result == vmstate

    async def test_1_byte_vmstate(self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]) -> None:
        """Minimal valid vmstate (1 byte, a zero byte): its size is not zero, so it is a cache hit.
        Its digest is SHA-256 of the size alone (1, 8 bytes big-endian): no non-zero block."""
        vmstate, _meta = await self._populate(
            manager,
            data=b"\x00",
            meta_dict={"vmstate_sha256": "cd2662154e6d76b2b2b92e70c0cac3ccf534f9b74eb5b89819ec509083d00a50"},
        )
        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result == vmstate

    async def test_large_vmstate_sha256_check(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any], tmp_path: Path
    ) -> None:
        """A ~10 MB vmstate (160 blocks) that a save published is a hit: the save and the check
        compute one digest. A byte changed in its last block is a miss, and the entry goes."""
        spec = VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")
        data = bytes(range(1, 256)) * (10 * 1024 * 1024 // 255)  # no zero byte

        async def write_large_vmstate(path: Path, **_kwargs: object) -> None:
            path.write_bytes(data)

        with _template_save(manager, vm=_save_vm(home_disk=home_disk, qemu_vm=False), migrate=write_large_vmstate):
            await _save(manager, spec)
        vmstate = manager.cache_dir / f"{await manager.compute_cache_key(spec, l2_path=None)}.vmstate"
        assert await _check_spec(manager, spec) == vmstate

        with vmstate.open("r+b") as f:
            f.seek(-1, os.SEEK_END)
            f.write(b"\x00")  # was 0xff
        assert await _check_spec(manager, spec) is None
        assert not vmstate.exists()

    async def test_concurrent_check_cache_same_key(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Multiple concurrent check_cache calls for same key all return same path."""
        vmstate, _meta = await self._populate(manager)
        results = await asyncio.gather(
            *[_check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) for _ in range(20)]
        )
        assert results == [vmstate] * 20

    async def test_sha256_partial_corruption(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Flip one byte in vmstate → SHA-256 mismatch → cleaned up."""
        vmstate, meta = await self._populate(manager)
        # Flip one byte
        corrupted = bytearray(_VMSTATE)
        corrupted[3] ^= 0xFF
        vmstate.write_bytes(bytes(corrupted))

        result = await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result is None
        assert not vmstate.exists()
        assert not meta.exists()


# ============================================================================
# Torture Tests — Save Results
# ============================================================================


@pytest.mark.usefixtures("mock_system_probes")
class TestSaveResults:
    """A save publishes an entry that check_cache accepts. A save that fails at any step
    publishes nothing, and no file of it stays (also not its {key}.vmstate.tmp)."""

    _SPEC = VmSpec(language=Language.PYTHON, packages=(), memory_mb=512, cpu_cores=1)

    @pytest.fixture
    def save_vm(self, tmp_path: Path) -> MagicMock:
        """The save VM (as this user), with its home disk."""
        home_disk = tmp_path / "home.img"
        home_disk.write_bytes(b"home")
        return _save_vm(home_disk=home_disk, qemu_vm=False)

    async def test_saved_entry_is_a_hit(self, manager: MemorySnapshotManager, save_vm: MagicMock) -> None:
        """The entry the save publishes passes check_cache: the save writes the digest the
        check computes. The entry has its home disk."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        with _template_save(manager, vm=save_vm, migrate=_write_vmstate):
            await _save(manager, self._SPEC)

        assert _files_of(manager, key) == [".home", ".ram", ".vmstate", ".vmstate.meta"]
        assert await manager.check_cache(key) == manager.cache_dir / f"{key}.vmstate"
        assert (manager.cache_dir / f"{key}.home").read_bytes() == b"home"

    async def test_ram_and_home_flushed_before_the_entry_is_published(
        self, manager: MemorySnapshotManager, save_vm: MagicMock
    ) -> None:
        """The digest covers the vmstate only: the RAM template and the home disk go
        through to the disk before the metadata and the rename publish the entry. A test
        cannot cut the power: it records the flushes."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        meta_path = manager.cache_dir / f"{key}.vmstate.meta"
        flushed: list[tuple[Path, bool]] = []

        def recording_flush(fd: int) -> None:
            flushed.append((_fd_path(fd), meta_path.exists()))

        with (
            patch("exec_sandbox.memory_snapshot_manager.flush_to_disk", recording_flush),
            _template_save(manager, vm=save_vm, migrate=_write_vmstate),
        ):
            await _save(manager, self._SPEC)

        assert (manager.cache_dir / f"{key}.vmstate").exists()
        assert sorted(flushed) == sorted(
            [
                ((manager.cache_dir / f"{key}.ram").resolve(), False),
                ((manager.cache_dir / f"{key}.home").resolve(), False),
            ]
        )

    async def test_enospc_evicts_the_oldest_entry_and_leaves_no_file(
        self, manager: MemorySnapshotManager, save_vm: MagicMock
    ) -> None:
        """ENOSPC during the migration: the oldest entry is evicted (the next request saves
        again), and no file of the failed save stays."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        old_vmstate = manager.cache_dir / "old.vmstate"
        old_meta = manager.cache_dir / "old.vmstate.meta"
        old_vmstate.write_bytes(b"old data")
        old_meta.write_text("{}")
        os.utime(old_vmstate, (time.time() - 1000,) * 2)  # mtime too: relatime resets an atime older than mtime

        async def disk_full(_path: Path, **_kwargs: object) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        with _template_save(manager, vm=save_vm, migrate=disk_full):
            await _save(manager, self._SPEC)

        assert not old_vmstate.exists()
        assert not old_meta.exists()
        assert _files_of(manager, key) == []

    @pytest.mark.skipif(os.geteuid() == 0, reason="root opens a file of mode 0")
    async def test_entry_file_this_process_cannot_open_fails_the_save(
        self, manager: MemorySnapshotManager, save_vm: MagicMock
    ) -> None:
        """The host makes every file of an entry, so it can always flush them: a file it
        cannot open (here the RAM template, mode 0 after the migration) is a fault, and the
        save publishes nothing (no unflushed template)."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)

        async def migrate_then_lock_the_ram(path: Path, **kwargs: object) -> None:
            await _write_vmstate(path, **kwargs)
            (manager.cache_dir / f"{key}.ram").chmod(0)

        with _template_save(manager, vm=save_vm, migrate=migrate_then_lock_the_ram):
            await _save(manager, self._SPEC)

        assert _files_of(manager, key) == []

    async def test_cancelled_save_leaves_no_file(self, manager: MemorySnapshotManager, save_vm: MagicMock) -> None:
        """A cancel during the migration (a stop): the task ends cancelled, and no file of
        the save stays."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        migrating = asyncio.Event()

        async def migrate_until_cancelled(_path: Path, **_kwargs: object) -> None:
            migrating.set()
            await asyncio.Event().wait()

        with _template_save(manager, vm=save_vm, migrate=migrate_until_cancelled):
            task = await _schedule_save(manager, self._SPEC)
            assert task is not None
            await migrating.wait()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert task.cancelled()
        assert _files_of(manager, key) == []

    async def test_failed_migration_leaves_no_file(self, manager: MemorySnapshotManager, save_vm: MagicMock) -> None:
        """The migration fails (QEMU crashed): no entry, and no file of the save stays."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)

        async def crashed(_path: Path, **_kwargs: object) -> None:
            raise ConnectionError("QEMU crashed")

        with _template_save(manager, vm=save_vm, migrate=crashed):
            await _save(manager, self._SPEC)

        assert _files_of(manager, key) == []
        assert await manager.check_cache(key) is None


# ============================================================================
# Torture Tests — schedule_background_save Edge Cases
# ============================================================================


@pytest.mark.usefixtures("mock_system_probes")
class TestScheduleBackgroundSaveEdgeCases:
    """Edge cases for background save scheduling."""

    _SPEC = VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)

    async def test_different_keys_run_in_parallel(self, manager: MemorySnapshotManager) -> None:
        """Saves of two keys do not wait for each other: both save VMs boot at once."""
        gate = asyncio.Event()
        vms = _SaveVms(gate=gate, fail=False)
        with vms.of(manager):
            tasks = [
                await _schedule_save(manager, VmSpec(language=language, memory_mb=512))
                for language in (Language.PYTHON, Language.JAVASCRIPT)
            ]
            await vms.booted(2)  # both boot before either ends
            gate.set()
            await asyncio.gather(*(task for task in tasks if task is not None))
        assert sorted(vms.task_ids) == ["l1-save-javascript", "l1-save-python"]

    async def test_vm_destroyed_when_the_warm_up_fails(self, manager: MemorySnapshotManager) -> None:
        """The REPL warm-up answers with an error: no save, and the save VM is destroyed."""
        mock_vm = MagicMock()
        mock_vm.channel.send_request = AsyncMock(return_value=MagicMock(status="error"))

        with (
            patch.object(manager.vm_manager, "create_vm", new=AsyncMock(return_value=mock_vm)),
            patch.object(manager.vm_manager, "destroy_vm", new=AsyncMock()) as mock_destroy,
        ):
            task = await _schedule_save(manager, self._SPEC)
            assert task is not None
            await task

        mock_destroy.assert_awaited_once_with(mock_vm)

    async def test_vm_destroyed_on_exception(self, manager: MemorySnapshotManager) -> None:
        """The REPL warm-up raises: the save VM is destroyed all the same."""
        mock_vm = MagicMock()
        mock_vm.channel.send_request = AsyncMock(side_effect=TimeoutError("REPL timeout"))

        with (
            patch.object(manager.vm_manager, "create_vm", new=AsyncMock(return_value=mock_vm)),
            patch.object(manager.vm_manager, "destroy_vm", new=AsyncMock()) as mock_destroy,
        ):
            task = await _schedule_save(manager, self._SPEC)
            assert task is not None
            await task

        mock_destroy.assert_awaited_once_with(mock_vm)

    async def test_entry_saved_before_lock_boots_no_vm(self, manager: MemorySnapshotManager) -> None:
        """The entry appears between check_cache and the save lock: the save stops, no VM boots."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        (manager.cache_dir / f"{key}.vmstate").write_bytes(b"already saved")  # no meta yet: check_cache misses
        vms = _SaveVms(gate=_open_gate(), fail=False)

        with vms.of(manager):
            await _save(manager, self._SPEC)

        assert vms.task_ids == []
        assert (manager.cache_dir / f"{key}.vmstate").read_bytes() == b"already saved"

    async def test_save_lock_held_by_other_process_boots_no_vm(self, manager: MemorySnapshotManager) -> None:
        """Another process saves this key (holds its flock): the save skips, no VM boots."""
        key = await manager.compute_cache_key(self._SPEC, l2_path=None)
        vms = _SaveVms(gate=_open_gate(), fail=False)
        fd = (manager.cache_dir / f"{key}.vmstate.lock").open("w")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with vms.of(manager):
                await _save(manager, self._SPEC)
        finally:
            fd.close()

        assert vms.task_ids == []


# ============================================================================
# Torture Tests — Eviction Edge Cases
# ============================================================================


class TestEvictOldestSnapshotEdgeCases:
    """Edge cases for LRU eviction."""

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
            assert await manager.evict_oldest() is None
            # Both still exist
            assert (manager.cache_dir / "a.vmstate").exists()
            assert (manager.cache_dir / "b.vmstate").exists()
        finally:
            fd_a.close()
            fd_b.close()

    async def test_evict_removes_sidecar_files(self, manager: MemorySnapshotManager) -> None:
        """Eviction of an entry (no lock file yet) removes its sidecars too: the metadata,
        the RAM template and the home disk (eviction lists only *.vmstate)."""
        files = [manager.cache_dir / f"victim{ext}" for ext in (".vmstate", ".vmstate.meta", ".ram", ".home")]
        for path in files:
            path.write_bytes(b"data")

        assert await manager.evict_oldest() is not None
        assert [path.name for path in files if path.exists()] == []

    async def test_evict_ignores_non_vmstate_files(self, manager: MemorySnapshotManager) -> None:
        """Eviction only considers .vmstate files, ignores .lock and .meta."""
        # Only create .lock and .meta files (no .vmstate)
        (manager.cache_dir / "orphan.vmstate.lock").write_text("")
        (manager.cache_dir / "orphan.vmstate.meta").write_text("{}")
        (manager.cache_dir / "random.txt").write_text("hello")

        assert await manager.evict_oldest() is None  # no .vmstate file

        assert (manager.cache_dir / "orphan.vmstate.lock").exists()
        assert (manager.cache_dir / "orphan.vmstate.meta").exists()
        assert (manager.cache_dir / "random.txt").exists()


# ============================================================================
# Torture Tests — Full Integration Cycles (mock-based)
# ============================================================================


class TestIntegrationCycles:
    """Full save → check → evict → check cycles with mock MigrationClient."""

    async def _populate(
        self, manager: MemorySnapshotManager, language: Language = Language.PYTHON
    ) -> tuple[Path, Path]:
        """Create cache entry with valid SHA-256 (_VMSTATE and its digest)."""
        key = await manager.compute_cache_key(VmSpec(language=language, packages=(), memory_mb=512), l2_path=None)
        vmstate = manager.cache_dir / f"{key}.vmstate"
        meta = manager.cache_dir / f"{key}.vmstate.meta"
        vmstate.write_bytes(_VMSTATE)
        vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
        meta.write_text(json.dumps({"test": True, "vmstate_sha256": _VMSTATE_DIGEST}))
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

        await manager.evict_oldest()
        assert await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) is None

    async def test_multiple_entries_independent(
        self, manager: MemorySnapshotManager, mock_system_probes: dict[str, Any]
    ) -> None:
        """Multiple cache entries for different languages are independent."""
        py_vmstate, _ = await self._populate(manager, language=Language.PYTHON)
        js_vmstate, _ = await self._populate(manager, language=Language.JAVASCRIPT)

        assert await _check_spec(manager, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512)) == py_vmstate
        assert (
            await _check_spec(manager, VmSpec(language=Language.JAVASCRIPT, packages=(), memory_mb=512)) == js_vmstate
        )

        # Evict one (should be oldest)
        os.utime(py_vmstate, (time.time() - 1000,) * 2)  # mtime too: relatime resets an atime older than mtime
        await manager.evict_oldest()

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
        vmstate.write_bytes(_VMSTATE)
        vmstate.with_suffix(".home").write_bytes(b"home disk")  # an L1 entry carries its home disk
        meta.write_text(json.dumps({"vmstate_sha256": _VMSTATE_DIGEST}))

        # Check with second manager
        mgr2 = MemorySnapshotManager(settings, vm_manager)
        result = await _check_spec(mgr2, VmSpec(language=Language.PYTHON, packages=(), memory_mb=512))
        assert result == vmstate


def _fd_path(fd: int) -> Path:
    """The path of an open file (macOS F_GETPATH, Linux /proc)."""
    if sys.platform == "darwin":
        return Path(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0").decode()).resolve()
    return Path(f"/proc/self/fd/{fd}").readlink().resolve()

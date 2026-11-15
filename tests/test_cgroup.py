"""Tests for cgroup.py cgroup v2 and ulimit utilities.

Test categories:
- Normal cases: Happy path with valid inputs
- Edge cases: None, empty, boundary values
- Error cases: OSError, PermissionError, ConfigError
- Weird cases: Special characters, path traversal, malformed data
- Integration tests: Real cgroups on Linux (requires sudo)
"""

import asyncio
import contextlib
import errno
import itertools
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from exec_sandbox.cgroup import (
    CGROUP_APP_NAMESPACE,
    NO_CGROUP_STATS,
    Cgroup,
    CgroupStats,
    detect_cgroup_cpu_limit,
    detect_cgroup_memory_limit_mb,
    read_container_available_memory_mb,
    wrap_with_ulimit,
)
from exec_sandbox.platform_utils import HostOS
from tests.conftest import time_with

from .conftest import skip_unless_linux

# =============================================================================
# cgroup_v2_available - Tests
# =============================================================================


class TestCgroupV2Available:
    """cgroup_v2_available: the cgroup v2 mount detection."""

    @pytest.fixture(autouse=True)
    def reset_cache(self) -> Generator[None]:
        """Reset the cgroup cache before each test, and after it: a result of a fake cgroup
        tree must not reach the later tests of this worker (their VMs would run without a
        cgroup, and VMs of qemu-vm would be refused)."""
        from exec_sandbox.cgroup import cgroup_v2_available

        cgroup_v2_available.cache_clear()
        yield
        cgroup_v2_available.cache_clear()

    def test_returns_false_when_cgroup_dir_missing(self, tmp_path: Path):
        """Returns False when /sys/fs/cgroup doesn't exist."""
        from exec_sandbox.cgroup import cgroup_v2_available

        with patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(tmp_path / "nonexistent")):
            assert cgroup_v2_available() is False

    def test_returns_false_when_controllers_file_missing(self, tmp_path: Path):
        """Returns False when cgroup.controllers doesn't exist (cgroup v1)."""
        from exec_sandbox.cgroup import cgroup_v2_available

        # Create dir but no cgroup.controllers
        cgroup_dir = tmp_path / "cgroup"
        cgroup_dir.mkdir()

        with patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(cgroup_dir)):
            assert cgroup_v2_available() is False

    def test_returns_true_when_cgroup_v2_mounted(self, tmp_path: Path):
        """Returns True when cgroup v2 is properly mounted."""
        from exec_sandbox.cgroup import cgroup_v2_available

        # Create proper cgroup v2 structure
        cgroup_dir = tmp_path / "cgroup"
        cgroup_dir.mkdir()
        (cgroup_dir / "cgroup.controllers").write_text("cpu memory pids")

        with patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(cgroup_dir)):
            assert cgroup_v2_available() is True

    def test_caches_result(self, tmp_path: Path):
        """The first answer stays: a later change of the files does not change it."""
        from exec_sandbox.cgroup import cgroup_v2_available

        cgroup_dir = tmp_path / "cgroup"
        cgroup_dir.mkdir()
        (cgroup_dir / "cgroup.controllers").write_text("cpu memory pids")

        with patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(cgroup_dir)):
            assert cgroup_v2_available() is True
            (cgroup_dir / "cgroup.controllers").unlink()
            assert cgroup_v2_available() is True

    def test_handles_permission_error(self, tmp_path: Path):
        """Returns False when can't read cgroup.controllers."""
        from exec_sandbox.cgroup import cgroup_v2_available

        cgroup_dir = tmp_path / "cgroup"
        cgroup_dir.mkdir()
        controllers = cgroup_dir / "cgroup.controllers"
        controllers.write_text("cpu memory")

        with (
            patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(cgroup_dir)),
            patch.object(Path, "read_text", side_effect=PermissionError("Access denied")),
        ):
            assert cgroup_v2_available() is False


# =============================================================================
# wrap_with_ulimit - Tests
# =============================================================================


class TestWrapWithUlimit:
    """wrap_with_ulimit: the shell command that runs a VM with no cgroup."""

    @pytest.mark.parametrize(
        ("memory_mb", "limits"),
        [
            pytest.param(256, "ulimit -v 3670016 && ulimit -t 3600", id="256-mb"),
            pytest.param(0, "ulimit -v 0 && ulimit -t 3600", id="zero"),
            pytest.param(1024 * 1024, "ulimit -v 15032385536 && ulimit -t 3600", id="1-tb-no-overflow"),
        ],
    )
    def test_linux_sets_virtual_memory_and_cpu_time(self, memory_mb: int, limits: str):
        """Linux: virtual memory 14 times the cgroup memory (in KB) and 1 hour of CPU time. No
        -u: RLIMIT_NPROC counts every process of the user, not the processes of one VM."""
        with patch("exec_sandbox.cgroup.detect_host_os", return_value=HostOS.LINUX):
            wrapped = wrap_with_ulimit(["qemu-system-x86_64", "-m", "256"], cgroup_memory_mb=memory_mb)
        assert wrapped == ["bash", "-c", f"{limits} && exec qemu-system-x86_64 -m 256"]

    def test_macos_sets_no_limit(self):
        """macOS: no -v (not supported), no -t (it breaks the QEMU output pipes), no -u (it
        counts every process of the user): the command runs as is, with no shell."""
        with patch("exec_sandbox.cgroup.detect_host_os", return_value=HostOS.MACOS):
            wrapped = wrap_with_ulimit(["qemu-system-aarch64", "-m", "512"], cgroup_memory_mb=512)
        assert wrapped == ["qemu-system-aarch64", "-m", "512"]

    @pytest.mark.parametrize("arg", ["$(whoami)", "; rm -rf /", "| cat", "value with spaces", "it's", ""])
    def test_each_argument_reaches_the_command_unchanged(self, arg: str):
        """The shell runs the command with each argument as given: no expansion, no split. The
        Linux wrap (its quoting) on any host: bash runs its command part, without the limits
        (macOS refuses ulimit -v)."""
        with patch("exec_sandbox.cgroup.detect_host_os", return_value=HostOS.LINUX):
            wrapped = wrap_with_ulimit(["printf", "%s|", arg, "end"], cgroup_memory_mb=256)
        assert wrapped[:2] == ["bash", "-c"]
        command = wrapped[2].split(" && exec ", 1)[1]
        run = subprocess.run(["bash", "-c", f"exec {command}"], capture_output=True, text=True, timeout=10, check=True)
        assert run.stdout == f"{arg}|end|"

    @skip_unless_linux
    def test_linux_command_runs_with_the_limits(self):
        """The kernel holds the limits for the command: RLIMIT_AS is 14 times the cgroup
        memory, RLIMIT_CPU is 1 hour."""
        probe = "import resource as r; print(r.getrlimit(r.RLIMIT_AS)[0], r.getrlimit(r.RLIMIT_CPU)[0])"
        wrapped = wrap_with_ulimit([sys.executable, "-c", probe], cgroup_memory_mb=256)
        run = subprocess.run(wrapped, capture_output=True, text=True, timeout=30, check=True)
        assert run.stdout.split() == ["3758096384", "3600"]


# =============================================================================
# Cgroup.create - Tests
# =============================================================================


@pytest.fixture
def cgroup_root(tmp_path: Path) -> Iterator[Path]:
    """A folder that stands in for /sys/fs/cgroup (cgroup v2 'mounted'): the functions
    read and write real files in it."""
    with (
        patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(tmp_path)),
        patch("exec_sandbox.cgroup.cgroup_v2_available", return_value=True),
    ):
        yield tmp_path


class TestCanContainProcesses:
    """The qemu-vm probe's cgroup check: its probe cgroup goes on every end, and any failure is False."""

    @staticmethod
    def _ends(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
        """The probe cgroups killed then removed (the fake root is a plain folder)."""
        ended: list[Path] = []

        async def kill(_self: Cgroup) -> bool:
            return True

        async def remove(self: Cgroup) -> bool:
            ended.append(self.path)
            return True

        monkeypatch.setattr(Cgroup, "kill", kill)
        monkeypatch.setattr(Cgroup, "remove", remove)
        return ended

    @staticmethod
    def _with_cgroup_kill(monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe cgroup has cgroup.kill, as on Linux 5.14+ (the fake root makes plain files)."""
        real_create = Cgroup.create

        async def create_with_kill(**kwargs: Any) -> Cgroup | None:
            made = await real_create(**kwargs)
            assert made is not None
            (made.path / "cgroup.kill").touch()
            return made

        monkeypatch.setattr(Cgroup, "create", create_with_kill)

    async def test_a_refused_spawn_is_false_and_ends_the_probe(
        self, cgroup_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from exec_sandbox.cgroup import can_contain_processes

        ended = self._ends(monkeypatch)
        monkeypatch.setattr(
            "exec_sandbox.cgroup.asyncio.create_subprocess_exec", AsyncMock(side_effect=BlockingIOError(11, "EAGAIN"))
        )
        self._with_cgroup_kill(monkeypatch)  # so the probe reaches the spawn

        assert await can_contain_processes() is False
        assert [path.parent.name for path in ended] == ["exec-sandbox-probe"]

    async def test_a_cancel_during_the_spawn_ends_the_probe(
        self, cgroup_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from exec_sandbox.cgroup import can_contain_processes

        ended = self._ends(monkeypatch)
        spawning = asyncio.Event()

        async def hang(*_args: Any, **_kwargs: Any) -> None:
            spawning.set()
            await asyncio.Event().wait()

        monkeypatch.setattr("exec_sandbox.cgroup.asyncio.create_subprocess_exec", hang)
        self._with_cgroup_kill(monkeypatch)
        probe = asyncio.create_task(can_contain_processes())
        await spawning.wait()
        probe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await probe

        assert len(ended) == 1

    async def test_probe_cgroups_of_dead_processes_go_first(
        self, cgroup_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing records a probe cgroup: the next probe removes those of dead processes
        (a SIGKILL mid-probe, a failed rmdir), and keeps this process's own and other folders."""
        import subprocess

        ended = self._ends(monkeypatch)
        tenant = cgroup_root / CGROUP_APP_NAMESPACE / "exec-sandbox-probe"
        dead = subprocess.Popen(["true"])
        dead.wait()
        for name in (f"probe-{dead.pid}-1", f"probe-{os.getpid()}-1", "not-a-probe"):
            (tenant / name).mkdir(parents=True)

        from exec_sandbox.cgroup import can_contain_processes

        assert await can_contain_processes() is False  # the fake root has no cgroup.kill
        assert ended[0] == tenant / f"probe-{dead.pid}-1"
        assert len(ended) == 2  # then this probe's own cgroup
        assert ended[1].name.startswith(f"probe-{os.getpid()}-")
        assert ended[1].name != f"probe-{os.getpid()}-1"

    async def test_no_cgroup_kill_is_false(self, cgroup_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Linux before 5.14 has cgroup v2 without cgroup.kill: nothing could end a qemu-vm QEMU."""
        from exec_sandbox.cgroup import can_contain_processes

        ended = self._ends(monkeypatch)
        monkeypatch.setattr(Cgroup, "attach", AsyncMock(return_value=True))

        assert await can_contain_processes() is False  # the fake root has no cgroup.kill
        assert len(ended) == 1

    @pytest.mark.parametrize("refused", [False, True], ids=["moved", "move-refused"])
    async def test_the_answer_is_the_move_of_the_child(
        self, cgroup_root: Path, monkeypatch: pytest.MonkeyPatch, *, refused: bool
    ) -> None:
        """True only when the child's move into the probe cgroup lands (its pid in cgroup.procs,
        with the real Cgroup.attach); a refused move (EACCES: outside the delegated subtree) is
        False. The probe ends either way, and the child is gone."""
        if refused and os.geteuid() == 0:
            pytest.skip("root writes a read-only file")
        from exec_sandbox.cgroup import can_contain_processes

        ended = self._ends(monkeypatch)
        self._with_cgroup_kill(monkeypatch)
        with_kill = Cgroup.create
        made: list[Path] = []

        async def create(**kwargs: Any) -> Cgroup | None:
            cgroup = await with_kill(**kwargs)
            assert cgroup is not None
            made.append(cgroup.path)
            if refused:
                (cgroup.path / "cgroup.procs").touch()
                (cgroup.path / "cgroup.procs").chmod(0o444)  # the kernel's EACCES
            return cgroup

        monkeypatch.setattr(Cgroup, "create", create)

        assert await can_contain_processes() is (not refused)
        assert len(ended) == 1
        moved = (made[0] / "cgroup.procs").read_text()
        if refused:
            assert moved == ""
        else:
            with pytest.raises(ProcessLookupError):
                os.kill(int(moved), 0)  # the child that moved is gone


async def _created(*, memory_mb: int, cpu_cores: float) -> Path:
    """Cgroup.create of VM vm123 of tenant1; the path of the cgroup it made."""
    made = await Cgroup.create(tenant_id="tenant1", vm_id="vm123", memory_mb=memory_mb, cpu_cores=cpu_cores)
    assert made is not None
    return made.path


class TestCgroupCreate:
    """Cgroup.create: the hierarchy, its modes and limits, its worker thread, and where it gives None."""

    # --- Normal cases ---
    async def test_create_makes_the_cgroup_hierarchy(self, cgroup_root: Path):
        """Cgroup.create makes the tenant and VM cgroups."""
        result = await _created(memory_mb=256, cpu_cores=1.0)

        tenant = cgroup_root / CGROUP_APP_NAMESPACE / "tenant1"
        assert result == tenant / "vm123"
        assert result.is_dir()  # tenant + vm directories
        assert (tenant / "cgroup.subtree_control").read_text() == "+memory +cpu +pids"
        assert (result / "pids.max").read_text() == "256"

    @pytest.mark.parametrize("tenant_made", [False, True], ids=["new-tenant", "tenant-made-0775"])
    async def test_cgroups_get_no_group_write(self, cgroup_root: Path, tenant_made: bool) -> None:
        """Under `sg qemu-vm` and umask 002 a new folder gets group qemu-vm with group write:
        qemu-vm could make child cgroups (the VM cgroup's rmdir then fails) or remove an empty
        VM cgroup and make its own. A tenant cgroup made so before loses that bit."""
        tenant = cgroup_root / CGROUP_APP_NAMESPACE / "tenant1"
        old_umask = os.umask(0o002)
        try:
            if tenant_made:
                tenant.parent.mkdir(mode=0o755)
                tenant.mkdir(mode=0o775)
            result = await _created(memory_mb=256, cpu_cores=1.0)
        finally:
            os.umask(old_umask)
        assert [oct(stat.S_IMODE(path.stat().st_mode) & 0o777) for path in (tenant.parent, tenant, result)] == [
            "0o755",
            "0o755",
            "0o755",
        ]

    @pytest.mark.parametrize(
        "case",
        [
            (2 * 1024**3, "60", None, True),
            (2 * 1024**3, "60", ("max", ""), True),
            # A container without swap: its cgroup namespace root is the mount root as it sees it
            (2 * 1024**3, "60", ("0", ""), False),
            (2 * 1024**3, "60", ("0", CGROUP_APP_NAMESPACE), False),
            (2 * 1024**3, "0", None, False),  # cgroup v2 reclaim is file-only at swappiness 0
            (0, "60", None, False),
            (OSError("no /proc/meminfo"), "60", None, False),
        ],
        ids=[
            "swap",
            "swap-allowed-at-root",
            "swap-forbidden-at-root",
            "swap-forbidden-below",
            "swappiness-0",
            "no-swap",
            "swap-unreadable",
        ],
    )
    async def test_soft_memory_limit_only_where_pages_can_leave(
        self,
        cgroup_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        case: tuple[int | OSError, str, tuple[str, str] | None, bool],
    ):
        """memory.high (85% of max) lets the kernel move idle guest pages to zswap/swap. Where
        they cannot leave (no swap, or a cgroup above forbids it) the guest RAM stays resident
        and a soft limit below it only throttles the VM. zswap writeback is never turned off:
        a page the zswap pool refuses would have nowhere to go. The host swap comes from
        psutil.swap_memory (SwapTotal)."""
        from exec_sandbox import cgroup

        swap_total, swappiness, swap_max_at, soft_limit = case
        (cgroup_root / CGROUP_APP_NAMESPACE).mkdir()
        if swap_max_at is not None:
            value, folder = swap_max_at
            (cgroup_root / folder / "memory.swap.max").write_text(f"{value}\n")
        sysctl = tmp_path / "swappiness"
        sysctl.write_text(f"{swappiness}\n")
        monkeypatch.setattr("exec_sandbox.cgroup._SWAPPINESS", sysctl)

        def swap_memory() -> SimpleNamespace:
            if isinstance(swap_total, OSError):
                raise swap_total
            return SimpleNamespace(total=swap_total)

        monkeypatch.setattr(cgroup, "psutil", SimpleNamespace(swap_memory=swap_memory))
        result = await _created(memory_mb=256, cpu_cores=1.0)

        assert (result / "memory.max").read_text() == "268435456"  # 256 MiB
        if soft_limit:
            assert (result / "memory.high").read_text() == "228170137"  # 85% of it, rounded down
        else:
            assert not (result / "memory.high").exists()
        assert not (result / "memory.zswap.writeback").exists()

    async def test_create_runs_in_one_worker_thread(self, cgroup_root: Path):
        """The cgroup files are made off the event loop thread (mkdir and cgroup.subtree_control
        can wait for the kernel's cgroup locks)."""
        loop_thread = threading.get_ident()
        threads: set[int] = set()
        real_mkdir = Path.mkdir

        def mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
            threads.add(threading.get_ident())
            real_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", autospec=True, side_effect=mkdir):
            await _created(memory_mb=256, cpu_cores=1.0)

        assert len(threads) == 1  # one job: every mkdir in the same worker thread
        assert loop_thread not in threads

    async def test_create_cancel_waits_for_the_job(self, cgroup_root: Path):
        """A cancel during Cgroup.create returns only once the job has ended: the owner's cleanup
        never removes the folder while the job still makes it."""
        started, release = threading.Event(), threading.Event()
        real_mkdir = Path.mkdir

        def slow_mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
            started.set()
            release.wait(timeout=10)
            real_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", autospec=True, side_effect=slow_mkdir):
            task = asyncio.create_task(_created(memory_mb=256, cpu_cores=1.0))
            try:
                await asyncio.to_thread(started.wait, 10)
                task.cancel()
                for _ in range(3):  # the cancel reaches the task at its next step: one loop turn
                    await asyncio.sleep(0)
                assert not task.done(), "the cancel did not wait for the setup job"
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert (cgroup_root / CGROUP_APP_NAMESPACE / "tenant1" / "vm123" / "pids.max").exists()

    async def test_create_applies_precomputed_memory_limit(self, cgroup_root: Path):
        """memory.max is the memory that admission computed (overheads included there), in
        bytes: Cgroup.create adds nothing."""
        result = await _created(memory_mb=300, cpu_cores=1.0)
        assert (result / "memory.max").read_text() == "314572800"  # 300 MiB

    @pytest.mark.parametrize(
        ("cgroup_cpu_cores", "expected_cpu_max"),
        [
            (1.0, "100000 100000"),
            (1.25, "125000 100000"),
            (2.0, "200000 100000"),
            (4.0, "400000 100000"),
        ],
    )
    async def test_create_cpu_max_scales_with_cores(
        self, cgroup_root: Path, cgroup_cpu_cores: float, expected_cpu_max: str
    ):
        """cpu.max quota scales proportionally with cgroup_cpu_cores (effective, including overhead)."""
        result = await _created(memory_mb=256, cpu_cores=cgroup_cpu_cores)
        assert (result / "cpu.max").read_text() == expected_cpu_max

    # --- Error cases ---
    @pytest.mark.parametrize("err", [errno.EROFS, errno.EACCES], ids=["read-only", "no-delegation"])
    async def test_a_refused_cgroup_is_none(self, cgroup_root: Path, err: int):
        """A read-only cgroupfs (Docker Desktop) or no delegated subtree: no cgroup (the VM
        runs with ulimits only), not an error."""
        with patch.object(Path, "mkdir", side_effect=OSError(err, os.strerror(err))):
            assert await Cgroup.create(tenant_id="tenant1", vm_id="vm123", memory_mb=256, cpu_cores=1.0) is None

    async def test_a_create_refused_after_its_mkdir_removes_the_folder(self, cgroup_root: Path):
        """The VM folder is made, then a limit write is refused (EACCES): None, and the folder
        goes (once the VM runs without a cgroup, nothing else would remove it)."""
        real_write = Path.write_text

        def refuse_limits(path: Path, data: str) -> int:
            if path.name == "memory.max":
                raise OSError(errno.EACCES, os.strerror(errno.EACCES))
            return real_write(path, data)

        with patch.object(Path, "write_text", autospec=True, side_effect=refuse_limits):
            assert await Cgroup.create(tenant_id="tenant1", vm_id="vm123", memory_mb=256, cpu_cores=1.0) is None
        assert not (cgroup_root / CGROUP_APP_NAMESPACE / "tenant1" / "vm123").exists()

    async def test_no_cgroup_v2_is_none(self, cgroup_root: Path):
        """No cgroup v2 on this host (macOS, cgroup v1): None, and no folder is made."""
        with patch("exec_sandbox.cgroup.cgroup_v2_available", return_value=False):
            assert await Cgroup.create(tenant_id="tenant1", vm_id="vm123", memory_mb=256, cpu_cores=1.0) is None
        assert list(cgroup_root.iterdir()) == []

    async def test_create_other_oserror_raises_config_error(self, cgroup_root: Path):
        """Other OSError raises ConfigError HOST_DEPENDENCY."""
        from exec_sandbox.exceptions import ConfigError, ErrorCode

        error = OSError("Disk full")
        error.errno = 28  # ENOSPC

        with patch.object(Path, "mkdir", side_effect=error):
            with pytest.raises(
                ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="Failed to setup cgroup"
            ):
                await Cgroup.create(tenant_id="tenant1", vm_id="vm123", memory_mb=256, cpu_cores=1.0)


# =============================================================================
# Cgroup.attach - Tests
# =============================================================================


class TestAttach:
    """Cgroup.attach: the pid to cgroup.procs off the event loop; a refusal degrades to False."""

    async def test_attach_writes_pid_to_cgroup_procs(self, tmp_path: Path):
        """The pid goes to cgroup.procs from a worker thread (the kernel's move can wait for
        an RCU grace period)."""
        loop_thread = threading.get_ident()
        threads: list[int] = []
        real_write = Path.write_text

        def write_text(path: Path, data: str) -> int:
            threads.append(threading.get_ident())
            return real_write(path, data)

        with patch.object(Path, "write_text", autospec=True, side_effect=write_text):
            assert await Cgroup(tmp_path).attach(12345) is True

        assert (tmp_path / "cgroup.procs").read_text() == "12345"
        assert len(threads) == 1
        assert threads[0] != loop_thread

    @pytest.mark.parametrize("err", [errno.EACCES, errno.EPERM, errno.ESRCH], ids=["eacces", "eperm", "gone"])
    async def test_denied_attach_degrades_to_false(self, tmp_path: Path, err: int):
        """The orchestrator is outside the delegated subtree (EACCES, EPERM): no limits for this
        process, the VM still runs. A process that exited meanwhile (ESRCH) is no host fault:
        its exit shows where it is waited for."""
        with patch.object(Path, "write_text", side_effect=OSError(err, os.strerror(err))):
            assert await Cgroup(tmp_path).attach(12345) is False

    async def test_other_attach_error_raises_config_error(self, tmp_path: Path):
        """Any other errno is a real fault: ConfigError HOST_DEPENDENCY."""
        from exec_sandbox.exceptions import ConfigError, ErrorCode

        with (
            patch.object(Path, "write_text", side_effect=OSError(errno.ENOENT, "No such file or directory")),
            pytest.raises(
                ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY, match="Failed to attach PID"
            ),
        ):
            await Cgroup(tmp_path).attach(12345)


# =============================================================================
# Cgroup.stats - Tests
# =============================================================================


class TestStats:
    """Cgroup.stats: cpu.stat and memory.peak, each None when it cannot be read."""

    def test_missing_cgroup_has_no_stats(self, tmp_path: Path):
        assert Cgroup(tmp_path / "gone").stats() == NO_CGROUP_STATS

    def test_reads_cpu_and_memory_stats(self, tmp_path: Path):
        (tmp_path / "cpu.stat").write_text(
            "usage_usec 5000000\nuser_usec 3000000\nsystem_usec 2000000\nnr_throttled 42"
        )
        (tmp_path / "memory.peak").write_text("104857600")  # 100MB in bytes
        assert Cgroup(tmp_path).stats() == CgroupStats(cpu_time_ms=5000, peak_memory_mb=100, nr_throttled=42)

    def test_cpu_stats_without_memory_peak(self, tmp_path: Path):
        """memory.peak came with Linux 5.19: the CPU stats are still read before it."""
        (tmp_path / "cpu.stat").write_text("usage_usec 5000000\nnr_throttled 0")
        assert Cgroup(tmp_path).stats() == CgroupStats(cpu_time_ms=5000, peak_memory_mb=None, nr_throttled=0)

    def test_handles_malformed_files(self, tmp_path: Path):
        """A value that is no number (its line is read, int() fails): no such stat, no error; the
        other stats of the file are still read."""
        (tmp_path / "cpu.stat").write_text("usage_usec abc\nnr_throttled 0")
        (tmp_path / "memory.peak").write_text("not_a_number")
        assert Cgroup(tmp_path).stats() == CgroupStats(cpu_time_ms=None, peak_memory_mb=None, nr_throttled=0)

    def test_handles_oserror_during_read(self, tmp_path: Path):
        (tmp_path / "cpu.stat").mkdir()  # exists, but reading it raises IsADirectoryError
        assert Cgroup(tmp_path).stats() == NO_CGROUP_STATS


# =============================================================================
# Cgroup.remove - Tests
# =============================================================================


class TestCgroupRemove:
    """Cgroup.remove: rmdir off the event loop; a missing cgroup is removed; a failure stays (False)."""

    async def test_removes_empty_cgroup(self, cgroup_root: Path):
        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        cgroup_path.mkdir(parents=True)
        loop_thread = threading.get_ident()
        threads: list[int] = []
        real_rmdir = Path.rmdir

        def rmdir(path: Path) -> None:
            threads.append(threading.get_ident())
            real_rmdir(path)

        with patch.object(Path, "rmdir", autospec=True, side_effect=rmdir):
            assert await Cgroup(cgroup_path).remove() is True

        assert not cgroup_path.exists()
        assert len(threads) == 1
        assert threads[0] != loop_thread

    async def test_handles_already_deleted_cgroup(self, cgroup_root: Path):
        """FileNotFoundError during rmdir: True (gone)."""
        assert await Cgroup(cgroup_root / "code-exec" / "tenant" / "vm123").remove() is True

    async def test_removes_child_folders_first(self, cgroup_root: Path):
        """A child cgroup a QEMU made in the VM cgroup goes first: rmdir of the VM cgroup alone
        would fail."""
        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        (cgroup_path / "child").mkdir(parents=True)
        assert await Cgroup(cgroup_path).remove() is True
        assert not cgroup_path.exists()

    async def test_handles_rmdir_oserror(self, cgroup_root: Path):
        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        cgroup_path.mkdir(parents=True)
        (cgroup_path / "leftover").write_text("")  # rmdir: Directory not empty
        assert await Cgroup(cgroup_path).remove() is False

    async def test_rmdir_error_names_the_errno_in_the_message(
        self, cgroup_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The log formats drop `extra`: the errno and its name are in the message itself."""
        cgroup_path = cgroup_root / "code-exec" / "tenant" / "vm123"
        cgroup_path.mkdir(parents=True)
        with (
            patch.object(Path, "rmdir", side_effect=OSError(errno.EBUSY, "Device or resource busy")),
            caplog.at_level("ERROR", logger="exec_sandbox.cgroup"),
        ):
            assert await Cgroup(cgroup_path).remove() is False
        assert [r.getMessage() for r in caplog.records] == [
            f"cgroup removal error: [Errno {errno.EBUSY}] EBUSY: Device or resource busy"
        ]

    async def test_no_pid_moves_to_the_parent(self, cgroup_root: Path) -> None:
        """No pid goes to a cgroup.procs above: the parent enables controllers, so cgroup v2
        refuses a process there (no internal process rule); Cgroup.kill empties the cgroup
        instead. On cgroupfs the control files go with the folder: the rmdir stand-in
        removes them."""
        tenant = cgroup_root / "code-exec" / "tenant"
        cgroup_path = tenant / "vm123"
        cgroup_path.mkdir(parents=True)
        (tenant / "cgroup.procs").write_text("")
        (cgroup_path / "cgroup.procs").write_text("123\n")
        with patch.object(Path, "rmdir", autospec=True, side_effect=shutil.rmtree):
            assert await Cgroup(cgroup_path).remove() is True
        assert not cgroup_path.exists()
        assert [path.relative_to(cgroup_root) for path in cgroup_root.rglob("cgroup.procs")] == [
            Path("code-exec/tenant/cgroup.procs")
        ]
        assert (tenant / "cgroup.procs").read_text() == ""

    async def test_close_kills_then_removes(self, cgroup_root: Path) -> None:
        """close = kill, then remove only once empty; `async with` closes at the block's end. A
        folder stands in for the cgroup (as in TestKillCgroup): a process that does not leave
        keeps it; once cgroup.events says populated 0, the rmdir stand-in removes it with its
        control files, as cgroupfs does."""
        path = cgroup_root / "code-exec" / "tenant" / "vm123"
        path.mkdir(parents=True)
        (path / "cgroup.events").write_text("populated 1\nfrozen 0\n")
        (path / "cgroup.procs").write_text("4242\n")
        (path / "cgroup.kill").write_text("")
        clock = itertools.count(100.0, 3.0)  # 3 s later at each read: the wait ends at its first check

        with patch.object(Path, "rmdir", autospec=True, side_effect=shutil.rmtree):
            with patch("exec_sandbox.cgroup.time", time_with(monotonic=lambda: next(clock))):
                assert await Cgroup(path).close() is False
            assert (path / "cgroup.kill").read_text() == "1"
            assert path.exists()  # no rmdir: a process is still there

            (path / "cgroup.events").write_text("populated 0\nfrozen 0\n")
            async with Cgroup(path):
                assert path.exists()
        assert not path.exists()


class TestKillCgroup:
    """Cgroup.kill: SIGKILL to every process left in a VM cgroup (cgroup.kill), then a wait
    until cgroup.events says "populated 0". A folder stands in for the cgroup; the real
    kernel behaviour is in TestCgroupIntegration."""

    @staticmethod
    def _cgroup(cgroup_root: Path, *, populated: int, procs: str, kill_file: bool) -> Path:
        path = cgroup_root / "code-exec" / "tenant" / "vm123"
        path.mkdir(parents=True)
        (path / "cgroup.events").write_text(f"populated {populated}\nfrozen 0\n")
        (path / "cgroup.procs").write_text(procs)
        if kill_file:  # cgroupfs has it from Linux 5.14
            (path / "cgroup.kill").write_text("")
        return path

    async def test_empty_cgroup_is_not_killed(self, cgroup_root: Path) -> None:
        path = self._cgroup(cgroup_root, populated=0, procs="", kill_file=True)
        assert await Cgroup(path).kill() is True
        assert (path / "cgroup.kill").read_text() == ""

    async def test_gone_cgroup_is_empty(self, cgroup_root: Path) -> None:
        assert await Cgroup(cgroup_root / "code-exec" / "tenant" / "never-made").kill() is True

    async def test_populated_cgroup_is_killed_then_read_again(self, cgroup_root: Path) -> None:
        """A process left (QEMU's teardown process): "1" goes to cgroup.kill, and the wait
        ends when the events file says populated 0 (here the kill write empties it)."""
        path = self._cgroup(cgroup_root, populated=1, procs="4242\n", kill_file=True)
        kills: list[Path] = []

        def send_kill(target: Path) -> None:  # the kernel: the kill empties the cgroup
            kills.append(target)
            (target / "cgroup.events").write_text("populated 0\nfrozen 0\n")

        with patch("exec_sandbox.cgroup._send_kill", side_effect=send_kill):
            assert await Cgroup(path).kill() is True
        assert kills == [path]

    async def test_the_wait_starts_when_the_job_runs(self, cgroup_root: Path) -> None:
        """A busy executor runs the job seconds after Cgroup.kill queues it: the wait still
        gets its whole CGROUP_DRAIN_TIMEOUT_SECONDS (a kill that works is never "unconfirmed").
        A fake clock: the job starts 3 s after the call. The private _drain_cgroup stands in for
        the wait to read the deadline it gets: the real one with a fake clock never reaches
        poll(2), and its result does not show the deadline."""
        from exec_sandbox import cgroup as cgroup_module

        path = self._cgroup(cgroup_root, populated=1, procs="4242\n", kill_file=True)
        now = [100.0]
        deadlines: list[float] = []
        real_run_to_end = cgroup_module.run_to_end

        async def queued(call: Callable[[], bool]) -> bool:
            now[0] += 3.0
            return await real_run_to_end(call)

        def drain(cgroup_path: Path, *, deadline: float) -> bool:
            deadlines.append(deadline)
            return True

        with (
            patch("exec_sandbox.cgroup.time", time_with(monotonic=lambda: now[0])),
            patch("exec_sandbox.cgroup.run_to_end", queued),
            patch("exec_sandbox.cgroup._drain_cgroup", side_effect=drain),
        ):
            assert await Cgroup(path).kill() is True
        assert deadlines == [105.0]  # queued at 100 s, run at 103 s, then 2 s of wait

    @staticmethod
    def _late_clock() -> SimpleNamespace:
        """The module's `time` with a monotonic clock that is 3 s later at each read: the
        wait (CGROUP_DRAIN_TIMEOUT_SECONDS, 2 s) ends at its first check."""
        clock = itertools.count(100.0, 3.0)
        return time_with(monotonic=lambda: next(clock))

    async def test_still_populated_at_the_deadline_is_false(
        self, cgroup_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A process that does not leave (D state): "1" goes to cgroup.kill, False at the end
        of the wait, and the message names the pids still there."""
        path = self._cgroup(cgroup_root, populated=1, procs="4242\n", kill_file=True)
        with (
            patch("exec_sandbox.cgroup.time", self._late_clock()),
            caplog.at_level("ERROR", logger="exec_sandbox.cgroup"),
        ):
            assert await Cgroup(path).kill() is False
        assert (path / "cgroup.kill").read_text() == "1"
        assert [r.getMessage() for r in caplog.records] == [
            "cgroup still has processes at the end of the wait after cgroup.kill: pids 4242"
        ]

    def test_wait_ends_at_its_deadline(self, cgroup_root: Path) -> None:
        """The real poll(2) wait ends at its deadline. A private call: through Cgroup.kill
        the wait lasts CGROUP_DRAIN_TIMEOUT_SECONDS (2 s), and a fake clock never reaches
        poll(2)."""
        from exec_sandbox.cgroup import _drain_cgroup  # pyright: ignore[reportPrivateUsage]

        path = self._cgroup(cgroup_root, populated=1, procs="4242\n", kill_file=True)
        start = time.monotonic()
        assert _drain_cgroup(path, deadline=start + 0.05) is False
        assert 0.05 <= time.monotonic() - start < 0.5  # not before the deadline, not long after

    async def test_no_kill_file_is_a_wait_not_an_error(
        self, cgroup_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Linux < 5.14 has no cgroup.kill: the processes may still leave by themselves, so the
        drain waits. The file is never made (on cgroupfs, an open with O_CREAT of a missing
        file fails with EACCES: a drain error, False at once)."""
        path = self._cgroup(cgroup_root, populated=1, procs="4242\n", kill_file=False)
        with (
            patch("exec_sandbox.cgroup.time", self._late_clock()),
            caplog.at_level("ERROR", logger="exec_sandbox.cgroup"),
        ):
            assert await Cgroup(path).kill() is False
        assert not (path / "cgroup.kill").exists()
        assert [r.getMessage() for r in caplog.records] == [
            "cgroup still has processes at the end of the wait after cgroup.kill: pids 4242"
        ]


class TestCgroupEndAll:
    """Cgroup.end_all: kill then remove for the VM cgroups of a dead process (the
    sweep of its folder). Folders stand in for the cgroups; the kernel side is in
    TestCgroupIntegration. Where a test needs the order or the deadline of the waits, the
    private _drain_cgroup stands in for the wait (poll(2) on cgroup.events): a plain file
    cannot show a process that leaves, and the real wait does not show its deadline."""

    def test_every_cgroup_is_killed_before_any_wait_and_only_empty_ones_go(self, cgroup_root: Path) -> None:
        """The waits overlap (each VM unmaps its RAM meanwhile); a cgroup that keeps a process
        stays, and the result says so."""
        paths = []
        for name in ("vm-1", "vm-2", "vm-3"):
            path = cgroup_root / "code-exec" / "tenant" / name
            path.mkdir(parents=True)
            (path / "cgroup.kill").write_text("")  # cgroupfs has it; end_all never creates it
            paths.append(path)
        kills_at_first_wait: list[str] = []

        def drain(path: Path, *, deadline: float) -> bool:
            if not kills_at_first_wait:
                kills_at_first_wait.extend((p / "cgroup.kill").read_text() for p in paths)
            return path.name != "vm-2"  # vm-2 keeps a process

        with (
            patch("exec_sandbox.cgroup._drain_cgroup", side_effect=drain),
            patch.object(Path, "rmdir", autospec=True, side_effect=shutil.rmtree),  # cgroupfs: the control files go too
        ):
            assert Cgroup.end_all([Cgroup(path) for path in paths]) == [Cgroup(paths[1])]
        assert kills_at_first_wait == ["1", "1", "1"]
        assert [path.exists() for path in paths] == [False, True, False]

    def test_a_cgroup_that_cannot_be_removed_is_left(self, cgroup_root: Path) -> None:
        """Empty but its rmdir fails (a plain folder with files: ENOTEMPTY): it is left, so the
        sweep keeps its record and tries again (a folder with no record would stay for good)."""
        path = cgroup_root / "code-exec" / "tenant" / "vm-1"
        path.mkdir(parents=True)
        (path / "cgroup.events").write_text("populated 0\nfrozen 0\n")
        (path / "cgroup.kill").write_text("")
        assert Cgroup.end_all([Cgroup(path)]) == [Cgroup(path)]
        assert path.exists()

    def test_one_deadline_for_every_cgroup(self, cgroup_root: Path) -> None:
        """Cgroups whose processes do not leave (D state) cost CGROUP_DRAIN_TIMEOUT_SECONDS
        together, not that much each: every wait ends at the same deadline. A fake clock: each
        stuck wait lasts until its deadline."""
        paths = [cgroup_root / "code-exec" / "tenant" / f"vm-{index}" for index in range(3)]
        now = [100.0]
        deadlines: list[float] = []

        def stuck(path: Path, *, deadline: float) -> bool:
            deadlines.append(deadline)
            now[0] = max(now[0], deadline)
            return False

        with (
            patch("exec_sandbox.cgroup.time", time_with(monotonic=lambda: now[0])),
            patch("exec_sandbox.cgroup._drain_cgroup", side_effect=stuck),
        ):
            assert Cgroup.end_all([Cgroup(path) for path in paths]) == [Cgroup(path) for path in paths]
        assert deadlines == [102.0] * 3  # one deadline of 2 s for all
        assert now[0] == 102.0

    def test_gone_cgroups_are_ended(self, cgroup_root: Path) -> None:
        gone = cgroup_root / "code-exec" / "tenant" / "never-made"
        assert Cgroup.end_all([Cgroup(gone)]) == []
        assert not gone.exists()


# =============================================================================
# Constants - Tests
# =============================================================================


class TestCgroupConstants:
    """Where VM cgroups go."""

    def test_path_of_a_vm(self):
        """A VM's cgroup is /sys/fs/cgroup/code-exec/<tenant_id>/<vm_id>."""
        assert Cgroup.path_of(tenant_id="tenant1", vm_id="vm123") == Path("/sys/fs/cgroup/code-exec/tenant1/vm123")


# =============================================================================
# detect_cgroup_memory_limit_mb - Tests
# =============================================================================


def _read_at(base: Path, read: Callable[[], float | None]) -> float | None:
    """`read` with the cgroup v2 mount at `base` and the v1 mount at `base`/v1 (the host's own
    cgroup files are never read)."""
    with (
        patch("exec_sandbox.cgroup.CGROUP_V2_BASE_PATH", str(base)),
        patch("exec_sandbox.cgroup.CGROUP_V1_BASE_PATH", str(base / "v1")),
    ):
        return read()


def _v1_file(base: Path, controller: str, name: str, value: str) -> None:
    """A cgroup v1 file at the v1 mount of _read_at (no cgroup v2 file: v1 decides)."""
    (base / "v1" / controller).mkdir(parents=True, exist_ok=True)
    (base / "v1" / controller / name).write_text(value)


class TestDetectCgroupMemoryLimitMb:
    """Test container cgroup memory limit detection."""

    def test_v2_numeric_limit_returns_mb(self, tmp_path: Path):
        """cgroup v2 numeric memory.max → returns MB."""
        (tmp_path / "memory.max").write_text("4294967296")  # 4GB
        result = _read_at(tmp_path, detect_cgroup_memory_limit_mb)
        assert result == pytest.approx(4096.0)

    def test_v2_max_string_returns_none(self, tmp_path: Path):
        """cgroup v2 'max' (unlimited) → returns None."""
        (tmp_path / "memory.max").write_text("max")
        result = _read_at(tmp_path, detect_cgroup_memory_limit_mb)
        assert result is None

    def test_v2_small_limit(self, tmp_path: Path):
        """cgroup v2 small memory limit (1GB) returns correct MB."""
        (tmp_path / "memory.max").write_text("1073741824")  # 1GB
        result = _read_at(tmp_path, detect_cgroup_memory_limit_mb)
        assert result == pytest.approx(1024.0)

    def test_no_cgroup_returns_none(self, tmp_path: Path):
        """No cgroup files → returns None (macOS, bare metal)."""
        result = _read_at(tmp_path / "nonexistent", detect_cgroup_memory_limit_mb)
        assert result is None

    @pytest.mark.parametrize(
        ("limit", "mb"), [("2147483648", 2048.0), ("9223372036854771712", None)], ids=["limit", "unlimited"]
    )
    def test_v1_limit(self, tmp_path: Path, limit: str, mb: float | None):
        """cgroup v1 memory.limit_in_bytes; the kernel's unlimited value (page-rounded 2**63) is None."""
        _v1_file(tmp_path, "memory", "memory.limit_in_bytes", limit)
        assert _read_at(tmp_path, detect_cgroup_memory_limit_mb) == (None if mb is None else pytest.approx(mb))


class TestDetectCgroupCpuLimit:
    """Test container cgroup CPU limit detection."""

    def test_v2_quota_period_returns_cpus(self, tmp_path: Path):
        """cgroup v2 cpu.max '200000 100000' → 2.0 CPUs."""
        (tmp_path / "cpu.max").write_text("200000 100000")
        result = _read_at(tmp_path, detect_cgroup_cpu_limit)
        assert result == pytest.approx(2.0)

    def test_v2_max_string_returns_none(self, tmp_path: Path):
        """cgroup v2 'max 100000' (unlimited) → returns None."""
        (tmp_path / "cpu.max").write_text("max 100000")
        result = _read_at(tmp_path, detect_cgroup_cpu_limit)
        assert result is None

    def test_v2_fractional_cpus(self, tmp_path: Path):
        """cgroup v2 fractional CPU quota → correct float."""
        (tmp_path / "cpu.max").write_text("50000 100000")  # 0.5 CPUs
        result = _read_at(tmp_path, detect_cgroup_cpu_limit)
        assert result == pytest.approx(0.5)

    def test_no_cgroup_returns_none(self, tmp_path: Path):
        """No cgroup files → returns None."""
        result = _read_at(tmp_path / "nonexistent", detect_cgroup_cpu_limit)
        assert result is None

    @pytest.mark.parametrize(("quota", "cpus"), [("150000", 1.5), ("-1", None)], ids=["quota", "unlimited"])
    def test_v1_quota(self, tmp_path: Path, quota: str, cpus: float | None):
        """cgroup v1 cpu.cfs_quota_us / cpu.cfs_period_us; a quota of -1 is no limit."""
        _v1_file(tmp_path, "cpu", "cpu.cfs_quota_us", quota)
        _v1_file(tmp_path, "cpu", "cpu.cfs_period_us", "100000")
        assert _read_at(tmp_path, detect_cgroup_cpu_limit) == (None if cpus is None else pytest.approx(cpus))


class TestReadContainerAvailableMemoryMb:
    """Test container available memory reading."""

    def test_v2_returns_available(self, tmp_path: Path):
        """cgroup v2 memory.max - memory.current = available."""
        (tmp_path / "memory.max").write_text("4294967296")  # 4GB
        (tmp_path / "memory.current").write_text("1073741824")  # 1GB
        result = _read_at(tmp_path, read_container_available_memory_mb)
        assert result == pytest.approx(3072.0)  # 3GB

    def test_v2_unlimited_returns_none(self, tmp_path: Path):
        """cgroup v2 'max' → returns None."""
        (tmp_path / "memory.max").write_text("max")
        result = _read_at(tmp_path, read_container_available_memory_mb)
        assert result is None

    def test_v2_current_exceeds_max_returns_zero(self, tmp_path: Path):
        """cgroup v2 current > max → returns 0 (not negative)."""
        (tmp_path / "memory.max").write_text("1073741824")  # 1GB
        (tmp_path / "memory.current").write_text("2147483648")  # 2GB (over limit)
        result = _read_at(tmp_path, read_container_available_memory_mb)
        assert result == 0.0

    def test_no_cgroup_returns_none(self, tmp_path: Path):
        """No cgroup files → returns None."""
        result = _read_at(tmp_path / "nonexistent", read_container_available_memory_mb)
        assert result is None

    @pytest.mark.parametrize(
        ("usage", "available"), [("1073741824", 3072.0), ("8589934592", 0.0)], ids=["room-left", "over-the-limit"]
    )
    def test_v1_available(self, tmp_path: Path, usage: str, available: float):
        """cgroup v1 memory.limit_in_bytes - memory.usage_in_bytes, never below 0."""
        _v1_file(tmp_path, "memory", "memory.limit_in_bytes", "4294967296")  # 4 GB
        _v1_file(tmp_path, "memory", "memory.usage_in_bytes", usage)
        assert _read_at(tmp_path, read_container_available_memory_mb) == pytest.approx(available)


# =============================================================================
# Integration Tests (Real cgroups, Linux-only, requires sudo)
# =============================================================================


@pytest.mark.sudo
@skip_unless_linux
class TestCgroupIntegration:
    """Integration tests using real cgroups on Linux.

    These tests require:
    - Linux operating system
    - Root/sudo privileges
    - cgroup v2 filesystem mounted at /sys/fs/cgroup
    """

    @staticmethod
    async def _create(vm_id: str, *, memory_mb: int) -> Cgroup:
        """A real VM cgroup of tenant integration-test (cgroup v2 delegated to this process)."""
        made = await Cgroup.create(tenant_id="integration-test", vm_id=vm_id, memory_mb=memory_mb, cpu_cores=1.0)
        assert made is not None, "no delegated cgroup v2 subtree for this process"
        return made

    @pytest.fixture
    def unique_vm_id(self) -> str:
        """Generate unique VM ID for test isolation."""
        return f"test-{uuid4().hex}"

    async def test_create_makes_a_real_cgroup(self, unique_vm_id: str) -> None:
        """Cgroup.create makes a real cgroup on Linux, and the kernel holds its limits."""
        cgroup_path = (await self._create(unique_vm_id, memory_mb=300)).path

        try:
            assert (cgroup_path / "cgroup.procs").exists()
            assert (cgroup_path / "memory.max").read_text().strip() == "314572800"  # 300 MiB
            assert (cgroup_path / "cpu.max").read_text().strip() == "100000 100000"
            assert (cgroup_path / "pids.max").read_text().strip() == "256"
        finally:
            await Cgroup(cgroup_path).remove()

    async def test_attach_real_process_to_cgroup(self, unique_vm_id: str) -> None:
        """Cgroup.attach moves a real process into the cgroup."""

        # Create cgroup
        cgroup_path = (await self._create(unique_vm_id, memory_mb=256)).path

        # Start a real process (sleep)
        proc = await asyncio.create_subprocess_exec(
            "sleep",
            "60",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        try:
            assert proc.pid is not None

            # Attach process to cgroup
            assert await Cgroup(cgroup_path).attach(proc.pid)

            # Verify process is in cgroup
            assert (cgroup_path / "cgroup.procs").read_text().split() == [str(proc.pid)]

        finally:
            # Cleanup process
            proc.terminate()
            await proc.wait()
            # Cleanup cgroup
            await Cgroup(cgroup_path).remove()

    async def test_stats_of_a_real_cgroup(self, unique_vm_id: str) -> None:
        """Cgroup.stats reads real stats."""

        # Create cgroup
        cgroup_path = (await self._create(unique_vm_id, memory_mb=256)).path

        # A process that writes 16 MiB only once it is in the cgroup (it waits for a line on
        # stdin): cgroup v2 does not move what a process charged before its move
        proc = await asyncio.create_subprocess_exec(
            "python3",
            "-c",
            "import sys; sys.stdin.readline(); x = b'x' * (16 * 1024 * 1024); import time; time.sleep(0.5)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        try:
            assert proc.pid is not None
            assert await Cgroup(cgroup_path).attach(proc.pid)
            await proc.communicate(b"go\n")  # the work starts now, in the cgroup

            # Read stats
            stats = Cgroup(cgroup_path).stats()
            cpu_ms, mem_mb, nr_throttled = stats.cpu_time_ms, stats.peak_memory_mb, stats.nr_throttled

            # The process did work: a Python start takes CPU time, and its bytes alone are 16 MiB
            assert cpu_ms is not None, "CPU stats not available"
            assert cpu_ms > 0, f"CPU time of the process not counted: {cpu_ms}"
            if mem_mb is not None:  # memory.peak: Linux 5.19+
                assert mem_mb >= 16, f"Peak memory below the 16 MiB written: {mem_mb}"
            assert nr_throttled is not None, "cpu.stat has nr_throttled with the cpu controller on"

        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
            await Cgroup(cgroup_path).remove()

    async def test_remove_removes_a_real_cgroup(self, unique_vm_id: str) -> None:
        """Cgroup.remove removes a real cgroup."""

        # Create cgroup
        cgroup_path = (await self._create(unique_vm_id, memory_mb=256)).path

        assert cgroup_path.exists(), "Cgroup should exist after setup"

        # Cleanup
        result = await Cgroup(cgroup_path).remove()

        assert result is True, "Cleanup should succeed"
        assert not cgroup_path.exists(), "Cgroup should be removed after cleanup"

    async def test_kill_then_remove_removes_a_cgroup_with_a_process(self, unique_vm_id: str) -> None:
        """A process that outlives the VM's spawned process (as QEMU's teardown process does):
        rmdir alone fails (EBUSY); Cgroup.kill kills it and waits, then rmdir works."""

        cgroup_path = (await self._create(unique_vm_id, memory_mb=256)).path
        proc = await asyncio.create_subprocess_exec(
            "sleep", "60", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        try:
            assert await Cgroup(cgroup_path).attach(proc.pid)
            assert await Cgroup(cgroup_path).remove() is False  # EBUSY: a live member

            start = asyncio.get_running_loop().time()
            assert await Cgroup(cgroup_path).kill() is True
            assert asyncio.get_running_loop().time() - start < 0.5
            assert await proc.wait() == -9  # SIGKILL from cgroup.kill
            assert await Cgroup(cgroup_path).remove() is True
            assert not cgroup_path.exists()
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            await Cgroup(cgroup_path).remove()

    async def test_end_all_kills_waits_for_and_removes_each(self, unique_vm_id: str) -> None:
        """The sweep of a dead process's folder: each process left is killed (SIGKILL from
        cgroup.kill), each cgroup removed once empty."""

        paths: list[Path] = []
        procs: list[asyncio.subprocess.Process] = []
        try:
            for index in range(3):
                path = (await self._create(f"{unique_vm_id}-{index}", memory_mb=256)).path
                paths.append(path)
                proc = await asyncio.create_subprocess_exec(
                    "sleep", "60", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
                )
                procs.append(proc)
                assert await Cgroup(path).attach(proc.pid)

            start = asyncio.get_running_loop().time()
            assert await asyncio.to_thread(Cgroup.end_all, [Cgroup(path) for path in paths]) == []
            assert asyncio.get_running_loop().time() - start < 0.5
            assert [await proc.wait() for proc in procs] == [-9, -9, -9]
            assert [path.exists() for path in paths] == [False, False, False]
        finally:
            for proc in procs:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
            for path in paths:
                await Cgroup(path).kill()
                await Cgroup(path).remove()

    async def test_remove_removes_child_cgroups_first(self, unique_vm_id: str) -> None:
        """A child cgroup in a VM cgroup (made where a QEMU could write): rmdir of the VM
        cgroup alone fails (EBUSY); Cgroup.remove removes the empty children first."""

        cgroup_path = (await self._create(unique_vm_id, memory_mb=256)).path
        try:
            (cgroup_path / "child" / "grandchild").mkdir(parents=True)
            assert await Cgroup(cgroup_path).kill() is True
            assert await Cgroup(cgroup_path).remove() is True
            assert not cgroup_path.exists()
        finally:
            for each in (cgroup_path / "child" / "grandchild", cgroup_path / "child", cgroup_path):
                with contextlib.suppress(OSError):
                    each.rmdir()

    async def test_pids_limit_enforced(self, unique_vm_id: str) -> None:
        """The kernel holds pids.max: a fork in the cgroup fails at the limit."""

        # Create cgroup with low pids limit
        cgroup_path = (await self._create(unique_vm_id, memory_mb=256)).path

        # Override pids.max to a very low value for testing
        (cgroup_path / "pids.max").write_text("5")

        # Start a process that waits on stdin before forking, so we can
        # attach it to the cgroup first (avoids race where process exits
        # before cgroup attachment).
        proc = await asyncio.create_subprocess_exec(
            "python3",
            "-c",
            """
import os, sys
sys.stdin.readline()  # wait for go signal after cgroup attach
pids = []
try:
    for i in range(20):
        pid = os.fork()
        if pid == 0:
            import time
            time.sleep(10)
            sys.exit(0)
        pids.append(pid)
    print("NOT LIMITED")
    sys.exit(1)
except OSError as e:
    print(f"LIMITED: {e}")
    sys.exit(0)
finally:
    for pid in pids:
        try:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
        except:
            pass
""",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            assert proc.pid is not None
            assert await Cgroup(cgroup_path).attach(proc.pid)

            # Signal subprocess to start forking
            assert proc.stdin is not None
            proc.stdin.write(b"\n")
            await proc.stdin.drain()
            proc.stdin.close()

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            # A fork fails (EAGAIN) before the 20th: pids.max holds
            assert proc.returncode == 0, stdout.decode() + stderr.decode()
            assert stdout.decode().startswith("LIMITED: ")

        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
            await Cgroup(cgroup_path).remove()


class TestCgroupReclaim:
    """memory.reclaim: EAGAIN means "reclaimed less than asked", not "unsupported"."""

    @staticmethod
    def _writer(errors: list[int | None]) -> tuple[Callable[[Path, str], int], list[tuple[str, int]]]:
        """Path.write_text stand-in: records (value, thread) and fails with errors[n] on write n."""
        writes: list[tuple[str, int]] = []

        def write_text(_path: Path, data: str) -> int:
            writes.append((data, threading.get_ident()))
            err = errors[len(writes) - 1]
            if err is not None:
                raise OSError(err, os.strerror(err))
            return len(data)

        return write_text, writes

    @pytest.mark.parametrize(
        ("errors", "expected", "attempts"),
        [
            pytest.param([None], True, 1, id="full"),
            pytest.param([errno.EAGAIN], True, 1, id="partial-is-success"),
            pytest.param([errno.EINVAL, None], True, 2, id="no-swappiness-falls-back"),
            pytest.param([errno.EINVAL, errno.EAGAIN], True, 2, id="fallback-partial"),
            pytest.param([errno.EACCES], False, 1, id="denied"),
        ],
    )
    async def test_reclaim_errors(
        self, tmp_path: Path, errors: list[int | None], expected: bool, attempts: int
    ) -> None:
        """192 MiB is 201326592 bytes: first with swappiness=200 (Linux 6.9+), then plain."""
        write_text, writes = self._writer(errors)
        with patch.object(Path, "write_text", autospec=True, side_effect=write_text):
            assert await Cgroup(tmp_path).reclaim(amount_mb=192) is expected
        assert [data for data, _ in writes] == ["201326592 swappiness=200", "201326592"][:attempts]
        # The kernel reclaims inside the write (10 to 500 ms): never on the event loop thread
        assert threading.get_ident() not in {thread for _, thread in writes}

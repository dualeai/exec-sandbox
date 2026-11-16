"""system_probes: what a host without access to /dev/kvm is told, the /dev/kvm check with no
answer, the setpriv probe, the probe commands (no process left behind), the O_DIRECT probes,
the POSIX ACL read, the probe cache, the TSC deadline on macOS, the host counter rate and CPU
identity, the guest CPU check, KVM TSC scaling, and the cgroup pids.max warning."""

import asyncio
import ctypes
import errno
import logging
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from exec_sandbox import system_probes
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.platform_utils import HostArch, HostOS
from exec_sandbox.system_probes import check_kvm_available, probe_setpriv_pdeathsig
from tests.conftest import skip_unless_aarch64, skip_unless_hwaccel, skip_unless_linux, skip_unless_macos_arm64


class TestKvmAccess:
    @pytest.mark.parametrize("accessible", [False, True], ids=["no-access", "access"])
    async def test_dev_kvm_without_access_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, accessible: bool
    ) -> None:
        """/dev/kvm exists but this user cannot read and write it: one WARNING that names the
        fix (the kvm group), however many times the probe is asked; then TCG. With access,
        no such warning (the probe goes on to its next check)."""
        real_exists = Path.exists
        monkeypatch.setattr(Path, "exists", lambda self: str(self) == "/dev/kvm" or real_exists(self))
        monkeypatch.setattr(system_probes, "can_access", lambda _path, _mode: accessible)
        spawns = AsyncMock(side_effect=_no_subprocess)
        monkeypatch.setattr(system_probes.asyncio, "create_subprocess_exec", spawns)

        with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
            first = await check_kvm_available()
            second = await check_kvm_available()

        # With access, it goes on to its ioctl check: here that check cannot run, which is no
        # answer (None), never kept, so each call asks again; without access a kept False
        assert first is second is (None if accessible else False)
        warnings = [r for r in caplog.records if "kvm group" in r.getMessage()]
        assert spawns.await_count == (2 if accessible else 0)
        if accessible:
            assert warnings == []
        else:
            (record,) = warnings
            assert record.levelno == logging.WARNING
            assert record.getMessage() == (
                "KVM not available: this user cannot read and write /dev/kvm, so VMs use software emulation "
                "(TCG, about 5-8 times slower). Add the user to the kvm group (sudo usermod -aG kvm $USER), "
                "then log in again."
            )


async def _no_subprocess(*_args: object, **_kwargs: object) -> object:
    """The ioctl check after the access check: it fails here (no real /dev/kvm needed)."""
    raise OSError("no subprocess in this test")


class TestSetprivProbe:
    """setpriv --pdeathsig (util-linux 2.33+): children that end with this process need no
    Python preexec_fn where it works (subprocess_utils.start_managed_process)."""

    @pytest.mark.parametrize("works", [True, False], ids=["util-linux-2.33", "older"])
    async def test_probe_runs_the_launch_form_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, works: bool
    ) -> None:
        """A setpriv that knows --pdeathsig runs the rest; an older one exits 1 ("unrecognized
        option"): False, and one warning that names the version, however many times asked."""
        calls = tmp_path / "calls"
        setpriv = tmp_path / "setpriv"
        body = '[ "$1 $2 $3" = "--pdeathsig KILL --" ] && shift 3 && exec "$@"' if works else "exit 1"
        setpriv.write_text(f'#!/bin/sh\necho "$@" >> {calls}\n{body}\n')
        setpriv.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(system_probes, "detect_host_os", lambda: HostOS.LINUX)

        with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
            results = [await probe_setpriv_pdeathsig() for _ in range(3)]

        assert results == [works] * 3
        assert calls.read_text() == "--pdeathsig KILL -- /bin/sh -c :\n"  # cached: one run
        warnings = [r.getMessage() for r in caplog.records if "setpriv" in r.getMessage()]
        assert warnings == (
            []
            if works
            else [
                "setpriv --pdeathsig unavailable (util-linux 2.33+): each child that ends with this process "
                "(QEMU as this user, gvproxy, qemu-storage-daemon) starts through a fork of this process"
            ]
        )

    async def test_missing_setpriv_is_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", str(tmp_path))  # no setpriv there
        monkeypatch.setattr(system_probes, "detect_host_os", lambda: HostOS.LINUX)
        assert await probe_setpriv_pdeathsig() is False

    async def test_false_off_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_probes, "detect_host_os", lambda: HostOS.MACOS)
        assert await probe_setpriv_pdeathsig() is False


class TestProbeCommands:
    """A probe command (run_probe) that does not end, or whose probe is cancelled, is killed
    and reaped before the probe ends: no probe leaves a process behind."""

    @pytest.mark.parametrize("end", ["timeout", "cancel"])
    async def test_a_probe_leaves_no_process(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, end: str) -> None:
        """A QEMU that never answers --version. Its pid comes through a FIFO: the test reads it
        once the stand-in runs. The version probe gives up at its bound (None), or is
        cancelled; either way the stand-in is gone when the probe ends."""
        pid_pipe = tmp_path / "pid"
        os.mkfifo(pid_pipe)
        qemu = tmp_path / "qemu-system"
        qemu.write_text(f"#!/bin/sh\necho $$ > {pid_pipe}\nexec sleep 30\n")
        qemu.chmod(0o755)
        for name in ("EXEC_SANDBOX_QEMU_BIN_X86", "EXEC_SANDBOX_QEMU_BIN_ARM"):
            monkeypatch.setenv(name, str(qemu))
        monkeypatch.setattr(system_probes, "_PROBE_TIMEOUT_SECONDS", 0.5)

        probe = asyncio.create_task(system_probes.probe_qemu_version())
        pid = int(await asyncio.to_thread(pid_pipe.read_text))
        if end == "cancel":
            probe.cancel()
            with pytest.raises(asyncio.CancelledError):
                await probe
        else:
            assert await probe is None
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


class TestDirectIoProbe:
    """system_probes.probe_direct_io: the filesystem of the folder answers, cached per device,
    whatever the OS; False where the OS has no O_DIRECT."""

    _LINUX_O_DIRECT = 0o40000

    @pytest.fixture(autouse=True)
    def _fresh_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_probes.probe_cache, "direct_io", {})

    @staticmethod
    def _refusing_open(monkeypatch: pytest.MonkeyPatch, error: int) -> list[int]:
        """os.open as Linux answers a refused O_DIRECT probe: the file is made first, then the
        error. Other opens (pytest's own) go to the real call. Returns the probe's flags."""
        flags_seen: list[int] = []
        real_open = os.open

        def refuse(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            if ".direct-io-probe-" not in str(path):
                return real_open(path, flags, *args, **kwargs)
            flags_seen.append(flags)
            os.close(real_open(path, os.O_WRONLY | os.O_CREAT, 0o600))
            raise OSError(error, os.strerror(error))

        monkeypatch.setattr(system_probes, "os", SimpleNamespace(**{**vars(os), "open": refuse}))
        monkeypatch.setattr(system_probes, "_O_DIRECT", TestDirectIoProbe._LINUX_O_DIRECT)
        return flags_seen

    async def test_a_folder_that_refuses_o_direct_is_probed_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        flags_seen = self._refusing_open(monkeypatch, errno.EINVAL)
        with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
            assert await system_probes.probe_direct_io(tmp_path) is False
            assert await system_probes.probe_direct_io(tmp_path) is False

        assert len(flags_seen) == 1  # cached per device
        assert flags_seen[0] & TestDirectIoProbe._LINUX_O_DIRECT == TestDirectIoProbe._LINUX_O_DIRECT
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
            "Folder does not take O_DIRECT: writes there are buffered"
        ]
        assert list(tmp_path.iterdir()) == []  # the file the kernel made before it refused goes too

    async def test_another_open_error_is_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No space, a quota, file handles: the device may take O_DIRECT; the next save probes again."""
        flags_seen = self._refusing_open(monkeypatch, errno.ENOSPC)
        with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
            assert await system_probes.probe_direct_io(tmp_path) is False
            assert await system_probes.probe_direct_io(tmp_path) is False

        assert len(flags_seen) == 2  # not cached
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
            "O_DIRECT probe of the folder failed: writes there are buffered this time"
        ] * 2  # one per call
        assert list(tmp_path.iterdir()) == []

    async def test_an_os_with_no_o_direct_answers_no(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No O_DIRECT open flag on this OS (macOS): no probe file, False."""
        monkeypatch.setattr(system_probes, "_O_DIRECT", None)
        assert await system_probes.probe_direct_io(tmp_path) is False
        assert list(tmp_path.iterdir()) == []

    async def test_a_disk_folder_takes_o_direct(self, tmp_path: Path) -> None:
        """The real probe: a disk folder of a host with O_DIRECT takes it (macOS has none)."""
        if not hasattr(os, "O_DIRECT"):
            pytest.skip("this OS has no O_DIRECT")
        assert await system_probes.probe_direct_io(tmp_path) is True
        assert list(tmp_path.iterdir()) == []  # the probe file is gone


class TestDriveDirectIoProbe:
    """system_probes.probe_drive_direct_io: O_DIRECT for the drive files of a VM only on a
    Linux disk filesystem that takes it: not memory (tmpfs, ramfs), not checksummed (btrfs,
    ZFS), an overlay judged by its upper folder; cached per folder, a failed read not. The
    filesystem types are set through statfs_type: the test hosts have none of these."""

    _EXT4, _TMPFS, _RAMFS, _BTRFS, _ZFS, _OVERLAY = 0xEF53, 0x01021994, 0x858458F6, 0x9123683E, 0x2FC12FC1, 0x794C7630

    @pytest.fixture(autouse=True)
    def _linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_probes.sys, "platform", "linux")

    @staticmethod
    def _filesystem(
        monkeypatch: pytest.MonkeyPatch,
        *,
        f_type: int | None,
        upper_f_type: int | None = None,
        definite: bool = True,
    ) -> AsyncMock:
        """`f_type`: the folder's (None: statfs fails); `upper_f_type`: an overlay's upper
        folder (None: not visible); `definite`: the O_DIRECT probe got the filesystem's answer
        (it caches it per device), else another error it does not cache."""
        upper = Path("/upper")

        async def probe(folder: Path) -> bool:
            if definite:
                system_probes.probe_cache.direct_io[folder.stat().st_dev] = True
            return definite

        mock = AsyncMock(side_effect=probe)
        monkeypatch.setattr(system_probes, "statfs_type", lambda path: upper_f_type if path == upper else f_type)
        monkeypatch.setattr(system_probes, "overlay_upper", lambda _path: None if upper_f_type is None else upper)
        monkeypatch.setattr(system_probes, "probe_direct_io", mock)
        return mock

    @pytest.mark.parametrize(
        ("f_type", "upper_f_type", "expected"),
        [
            (_EXT4, None, True),
            (_TMPFS, None, False),
            (_RAMFS, None, False),
            (_BTRFS, None, False),
            (_ZFS, None, False),
            (_OVERLAY, _EXT4, True),
            (_OVERLAY, _BTRFS, False),
            (_OVERLAY, _TMPFS, False),
            (_OVERLAY, None, False),
        ],
        ids=[
            "ext4",
            "tmpfs",
            "ramfs",
            "btrfs",
            "zfs",
            "overlay-on-ext4",
            "overlay-on-btrfs",
            "overlay-on-tmpfs",
            "overlay-unseen",
        ],
    )
    async def test_the_filesystem_decides(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        f_type: int,
        upper_f_type: int | None,
        expected: bool,
    ) -> None:
        probe = self._filesystem(monkeypatch, f_type=f_type, upper_f_type=upper_f_type)
        assert await system_probes.probe_drive_direct_io(tmp_path) is expected
        assert probe.await_count == (1 if expected else 0)  # the O_DIRECT probe only where it can help

    async def test_a_folder_is_probed_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        probe = self._filesystem(monkeypatch, f_type=self._EXT4)
        assert [await system_probes.probe_drive_direct_io(tmp_path) for _ in range(3)] == [True] * 3
        probe.assert_awaited_once()

    @pytest.mark.parametrize("failure", ["statfs", "o-direct-probe"])
    async def test_a_failed_probe_is_asked_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        """A folder that cannot be read yet, or an O_DIRECT probe that failed for another
        reason (no space, file handles): no answer is kept, the next VM start asks again."""
        statfs_calls: list[Path] = []
        probe = self._filesystem(
            monkeypatch, f_type=None if failure == "statfs" else self._EXT4, definite=failure != "o-direct-probe"
        )
        real = system_probes.statfs_type
        monkeypatch.setattr(system_probes, "statfs_type", lambda path: statfs_calls.append(path) or real(path))
        assert [await system_probes.probe_drive_direct_io(tmp_path) for _ in range(2)] == [False, False]
        assert len(statfs_calls) == 2
        assert probe.await_count == (0 if failure == "statfs" else 2)

    async def test_off_linux_is_no(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        probe = self._filesystem(monkeypatch, f_type=self._EXT4)
        monkeypatch.setattr(system_probes.sys, "platform", "darwin")
        assert await system_probes.probe_drive_direct_io(tmp_path) is False
        probe.assert_not_awaited()


class TestPosixAclXattr:
    """system_probes.posix_acl_xattr: the OS and the filesystem say whether POSIX ACLs are
    there, not the name of the OS."""

    @staticmethod
    def _getxattr(monkeypatch: pytest.MonkeyPatch, answer: bytes | int) -> list[str]:
        """An xattr call that returns `answer`, or raises OSError(answer); returns the names read."""
        names: list[str] = []

        def getxattr(_path: Path, name: str) -> bytes:
            names.append(name)
            if isinstance(answer, int):
                raise OSError(answer, os.strerror(answer))
            return answer

        monkeypatch.setattr(system_probes, "_GETXATTR", getxattr)
        return names

    def test_an_os_with_no_xattr_call_has_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_probes, "_GETXATTR", None)
        assert system_probes.posix_acl_xattr(tmp_path) is None

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(errno.EOPNOTSUPP, id="no-acl-support"),  # no ACL support, mounted noacl
            pytest.param(errno.ENODATA, id="no-acl-beyond-mode"),
        ],
    )
    def test_no_acl_is_none(self, error: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        names = self._getxattr(monkeypatch, error)
        assert system_probes.posix_acl_xattr(tmp_path) is None
        assert names == [system_probes.POSIX_ACL_ACCESS_XATTR]

    def test_a_file_with_an_acl_gives_it(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._getxattr(monkeypatch, b"\x02\x00\x00\x00")
        assert system_probes.posix_acl_xattr(tmp_path) == b"\x02\x00\x00\x00"

    def test_the_real_call(self, tmp_path: Path) -> None:
        """A new file has no ACL beyond its mode, wherever this runs: None."""
        (tmp_path / "f").write_text("x")
        assert system_probes.posix_acl_xattr(tmp_path / "f") is None


class TestProbeCache:
    """The cache of probe results: its locks and its reset, as the probes and the tests use them."""

    async def test_reset_of_a_per_device_cache_keeps_it_a_dict(self, tmp_path: Path) -> None:
        """reset of a per-device cache gives an empty dict, not the sentinel: the next probe of
        that kind works."""
        cache = system_probes.ProbeCache()
        cache.direct_io[1] = True
        cache.reset("direct_io")
        assert cache.direct_io == {}
        system_probes.probe_cache.reset("direct_io")
        # a disk folder takes O_DIRECT where the OS has it (macOS has none)
        assert await system_probes.probe_direct_io(tmp_path) is hasattr(os, "O_DIRECT")

    def test_a_contended_lock_works_in_a_later_event_loop(self) -> None:
        """Two probes contend for a lock in one asyncio.run, then again in a second one: an
        asyncio.Lock binds to its first loop, so each loop gets its own."""
        cache = system_probes.ProbeCache()

        async def contend() -> None:
            async def hold() -> None:
                async with cache.get_lock("kvm"):
                    await asyncio.sleep(0)

            await asyncio.gather(hold(), hold())

        asyncio.run(contend())
        asyncio.run(contend())  # RuntimeError "bound to a different event loop" with one lock for all loops


class TestTscDeadlineOnMacos:
    """An Intel Mac: HVF never passes the TSC deadline timer to the guest, whatever the host
    CPU lists (TSCTMR in sysctl machdep.cpu.features): no TSC deadline, and no CPU flags are
    read. With +tsc-deadline and enforce, QEMU would refuse every VM."""

    async def test_hvf_never_has_the_deadline_timer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A CPU that lists the flag (the /proc/cpuinfo stand-in): still False on macOS."""
        reads: list[Path] = []

        def read_text(self: Path) -> str:
            reads.append(self)
            return "flags\t\t: fpu tsc tsc_deadline_timer\n"

        monkeypatch.setattr(Path, "read_text", read_text)
        monkeypatch.setattr(system_probes, "detect_host_os", lambda: HostOS.MACOS)
        monkeypatch.setattr(system_probes, "detect_host_arch", lambda: HostArch.X86_64)
        assert await system_probes.check_tsc_deadline() is False
        assert reads == []


class TestKvmCheckWithNoAnswer:
    """The /dev/kvm ioctl check that could not run (a fork limit) is no answer: no VM starts on
    a guess (detect_accel_type raises TransientError, a kept False would pin every later VM to
    TCG), and the next call checks again."""

    async def test_a_fork_limit_is_transient_and_asked_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_exists = Path.exists
        monkeypatch.setattr(Path, "exists", lambda self: str(self) == "/dev/kvm" or real_exists(self))
        monkeypatch.setattr(system_probes, "can_access", lambda _path, _mode: True)
        monkeypatch.setattr(system_probes, "detect_host_os", lambda: HostOS.LINUX)
        answers: list[object] = [OSError(errno.EAGAIN, "Resource temporarily unavailable"), (1, b"", b"")]

        async def probe(_cmd: object, **_kwargs: object) -> tuple[int, bytes, bytes]:
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer  # type: ignore[return-value]

        monkeypatch.setattr(system_probes, "run_probe", probe)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED):
            await system_probes.detect_accel_type(force_emulation=False)
        assert await system_probes.check_kvm_available() is False  # asked again: the ioctl failed, a final answer
        assert answers == []


class TestHostCounterRate:
    """probe_host_counter_hz: the rate of the counter a guest's clock runs on, from an exact
    source only."""

    @pytest.mark.parametrize(
        ("flags", "bogomips", "hz"),
        [
            pytest.param("fpu tsc constant_tsc tsc_known_freq", "5400.00", 2_700_000_000, id="stated-rate"),
            pytest.param("fpu tsc constant_tsc tsc_known_freq", "4599.99", 2_299_995_000, id="5-khz-steps"),
            pytest.param("fpu tsc constant_tsc", "5400.00", None, id="measured-at-boot"),
        ],
    )
    def test_x86_rate_only_when_the_cpu_states_it(self, flags: str, bogomips: str, hz: int | None) -> None:
        """BogoMIPS is tsc_khz / 500 when the CPU or the hypervisor states the TSC rate
        (tsc_known_freq); without that flag the kernel measured the rate at its boot."""
        cpuinfo = f"processor\t: 0\nvendor_id\t: GenuineIntel\nflags\t\t: {flags}\nbogomips\t: {bogomips}\n"
        assert system_probes._bogomips_hz(cpuinfo) == hz  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.parametrize(("error", "answer"), [(errno.EACCES, None), (errno.EPERM, None), (errno.EINVAL, OSError)])
    def test_a_refused_executable_page_is_no_source(
        self, monkeypatch: pytest.MonkeyPatch, error: int, answer: type[OSError] | None
    ) -> None:
        """A host that refuses executable memory (SELinux execmem, PaX): no exact source, an
        answer (None), not an error; the page closes with no buffer export left (BufferError
        before). Another mprotect error raises OSError. The code page never runs here."""

        def refused(*_args: object) -> int:
            ctypes.set_errno(error)
            return -1

        monkeypatch.setattr(system_probes.ctypes, "CDLL", lambda *_a, **_k: SimpleNamespace(mprotect=refused))
        if answer is None:
            assert system_probes._read_cntfrq_el0() is None  # pyright: ignore[reportPrivateUsage]
        else:
            with pytest.raises(OSError, check=lambda e: e.errno == error):
                system_probes._read_cntfrq_el0()  # pyright: ignore[reportPrivateUsage]

    @skip_unless_aarch64
    @skip_unless_linux
    async def test_arm_linux_reads_cntfrq_el0(self) -> None:
        """The generic timer rate, as the kernel computes BogoMIPS from it (rate / 500000)."""
        hz = await system_probes.probe_host_counter_hz()
        bogomips = next(line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("BogoMIPS"))
        assert hz is not None
        assert f"{hz / 500_000:.2f}" == bogomips.split(":")[1].strip()

    @skip_unless_macos_arm64
    @skip_unless_hwaccel
    async def test_hvf_rate_is_the_timebase(self) -> None:
        """Apple chips: HVF guests read a 24 MHz counter (hw.tbfrequency)."""
        assert await system_probes.probe_host_counter_hz() == 24_000_000


class TestHostCpuId:
    """probe_host_cpu_id: the host CPU an ARM64 -cpu host guest sees."""

    def test_linux_distinct_parts_sorted(self, tmp_path: Path) -> None:
        """One entry per distinct (MIDR, REVIDR) pair, sorted: big.LITTLE hosts list both kinds."""
        for cpu, midr in enumerate(["0x00000000410fd4f1", "0x00000000410fd4f1", "0x00000000410fd050"]):
            folder = tmp_path / f"cpu{cpu}" / "regs" / "identification"
            folder.mkdir(parents=True)
            (folder / "midr_el1").write_text(f"{midr}\n")
            (folder / "revidr_el1").write_text("0x0000000000000017\n")
        (tmp_path / "cpufreq").mkdir()  # not a CPU
        assert system_probes._linux_arm_cpu_id(tmp_path) == (  # pyright: ignore[reportPrivateUsage]
            "midr_el1=0x00000000410fd050,revidr_el1=0x0000000000000017;"
            "midr_el1=0x00000000410fd4f1,revidr_el1=0x0000000000000017"
        )

    def test_linux_without_the_registers(self, tmp_path: Path) -> None:
        assert system_probes._linux_arm_cpu_id(tmp_path) is None  # pyright: ignore[reportPrivateUsage]

    async def test_x86_has_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """x86 guests run a fixed CPU model: no host identity."""
        monkeypatch.setattr(system_probes, "detect_host_arch", lambda: HostArch.X86_64)
        assert await system_probes.probe_host_cpu_id() is None

    @skip_unless_macos_arm64
    @skip_unless_hwaccel
    async def test_hvf_feature_registers(self) -> None:
        """The registers QEMU's -cpu host takes under HVF, read with no entitlement."""
        cpu_id = await system_probes.probe_host_cpu_id()
        assert cpu_id is not None
        names = [part.split("=")[0] for part in cpu_id.split(",")]
        assert names[:9] == ["dfr0", "dfr1", "isar0", "isar1", "mmfr0", "mmfr1", "mmfr2", "pfr0", "pfr1"]
        assert all(re.fullmatch(r"[a-z0-9]+=0x[0-9a-f]{16}", part) for part in cpu_id.split(","))


class TestGuestCpu:
    """require_guest_cpu: the x86 guest CPU on a paused real QEMU (qemu-system-x86_64 under
    TCG runs on every host that has it)."""

    @pytest.fixture(autouse=True)
    def x86_qemu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        binary = shutil.which("qemu-system-x86_64")
        if binary is None:
            pytest.skip("no qemu-system-x86_64")
        monkeypatch.setattr(system_probes, "detect_host_arch", lambda: HostArch.X86_64)
        monkeypatch.setenv("EXEC_SANDBOX_QEMU_BIN_X86", binary)

    @pytest.mark.parametrize(
        "cpu",
        [
            "qemu64-v1,+avx2,+fma,vendor=GenuineIntel,family=6,model=60,stepping=4,enforce",
            "qemu64-v1,vendor=  Shanghai  ,enforce",
        ],
        ids=["intel-signature", "zhaoxin-vendor"],
    )
    async def test_a_cpu_tcg_runs(self, cpu: str) -> None:
        """QEMU starts it; TCG gives no TSC rate (the guest's TSC is the host's)."""
        assert await system_probes.require_guest_cpu(cpu, accel="tcg") is None

    async def test_a_cpu_the_host_cannot_run_names_qemu_reasons(self) -> None:
        """enforce and a flag TCG lacks (AVX-512): HOST_DEPENDENCY with QEMU's lines, before
        any VM, and the refusal is not kept (a fixed host runs it on the next start)."""
        cpu = "qemu64-v1,+avx512f,enforce"
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc:
            await system_probes.require_guest_cpu(cpu, accel="tcg")
        assert "avx512f" in exc.value.message
        assert exc.value.context == {"binary": shutil.which("qemu-system-x86_64"), "accel": "tcg", "cpu": cpu}
        assert system_probes.probe_cache.guest_cpu == {}

    async def test_the_answer_is_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cpu = "qemu64-v1,enforce"
        assert await system_probes.require_guest_cpu(cpu, accel="tcg") is None
        monkeypatch.setattr(system_probes, "run_probe", AsyncMock(side_effect=AssertionError("ran again")))
        assert await system_probes.require_guest_cpu(cpu, accel="tcg") is None


class TestKvmTscScaling:
    """probe_kvm_tsc_scaling: KVM_CAP_TSC_CONTROL on /dev/kvm; no /dev/kvm is an answer (kept),
    a limit is not (asked again)."""

    @pytest.fixture
    def kvm(self, monkeypatch: pytest.MonkeyPatch) -> list[object]:
        """Answers of the ioctl on a stand-in /dev/kvm (an open of /dev/null), one per call."""
        answers: list[object] = []
        real_open = Path.open
        monkeypatch.setattr(
            Path,
            "open",
            lambda self, *a, **k: real_open(Path(os.devnull) if str(self) == "/dev/kvm" else self, *a, **k),
        )

        def ioctl(_fd: int, request: int, arg: int) -> int:
            assert (request, arg) == (0xAE03, 60)  # KVM_CHECK_EXTENSION, KVM_CAP_TSC_CONTROL
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return cast("int", answer)

        monkeypatch.setattr(system_probes.fcntl, "ioctl", ioctl)
        return answers

    @pytest.mark.parametrize(("answer", "scaling"), [(1, True), (0, False)], ids=["scaling", "no-scaling"])
    async def test_the_capability(self, kvm: list[object], answer: int, scaling: bool) -> None:
        kvm.append(answer)
        assert await system_probes.probe_kvm_tsc_scaling() is scaling
        assert await system_probes.probe_kvm_tsc_scaling() is scaling  # kept: no second ioctl

    async def test_no_kvm_is_kept(self, kvm: list[object]) -> None:
        kvm.append(OSError(errno.ENOENT, "No such file or directory"))
        assert await system_probes.probe_kvm_tsc_scaling() is False
        assert await system_probes.probe_kvm_tsc_scaling() is False

    async def test_a_limit_is_asked_again(self, kvm: list[object]) -> None:
        """EMFILE (a file-handle limit) is no answer: TransientError LOCAL_DISK now (a False
        would refuse loads this worker can serve), and the next call asks KVM."""
        kvm.extend([OSError(errno.EMFILE, "Too many open files"), 1])
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
            await system_probes.probe_kvm_tsc_scaling()
        assert await system_probes.probe_kvm_tsc_scaling() is True


class TestGuestCpuFailures:
    """require_guest_cpu with a stand-in QEMU (a shell script): only QEMU's CPU feature
    refusal is HOST_DEPENDENCY; other failures can pass (QEMU_CRASHED) and nothing is kept."""

    @pytest.fixture
    def qemu(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(system_probes, "detect_host_arch", lambda: HostArch.X86_64)
        binary = tmp_path / "qemu-system-x86_64"
        monkeypatch.setenv("EXEC_SANDBOX_QEMU_BIN_X86", str(binary))
        return binary

    @staticmethod
    def _script(binary: Path, body: str) -> None:
        binary.write_text(f"#!/bin/sh\n{body}\n")
        binary.chmod(0o755)

    async def test_a_feature_refusal_is_the_host(self, qemu: Path) -> None:
        self._script(
            qemu,
            "echo 'qemu-system-x86_64: warning: host doesn'\"'\"'t support requested feature: CPUID[eax=07h,ecx=00h].EBX.avx2 [bit 5]' >&2\n"
            "echo 'qemu-system-x86_64: Host doesn'\"'\"'t support requested features' >&2\nexit 1",
        )
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc:
            await system_probes.require_guest_cpu("m,+avx2,enforce", accel="kvm")
        assert exc.value.message == (
            f"This host cannot run the guest CPU ({qemu} -accel kvm -cpu m,+avx2,enforce): qemu-system-x86_64: "
            "warning: host doesn't support requested feature: CPUID[eax=07h,ecx=00h].EBX.avx2 [bit 5] "
            "qemu-system-x86_64: Host doesn't support requested features"
        )
        assert exc.value.hint == (
            "x86 guests need an x86-64-v3 host CPU (Intel Haswell or AMD Zen 1, or newer) "
            "that passes these flags to its VMs."
        )
        assert exc.value.context == {"binary": str(qemu), "accel": "kvm", "cpu": "m,+avx2,enforce"}
        assert system_probes.probe_cache.guest_cpu == {}

    async def test_another_qemu_error_can_pass(self, qemu: Path) -> None:
        """KVM out of memory: QEMU_CRASHED with QEMU's line; not kept, the next start probes."""
        self._script(
            qemu, "echo 'qemu-system-x86_64: ioctl(KVM_CREATE_VM) failed: 12 Cannot allocate memory' >&2\nexit 1"
        )
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED) as exc:
            await system_probes.require_guest_cpu("m,enforce", accel="kvm")
        assert exc.value.message == (
            f"QEMU did not start the guest CPU ({qemu} -accel kvm -cpu m,enforce, exit code 1): "
            "qemu-system-x86_64: ioctl(KVM_CREATE_VM) failed: 12 Cannot allocate memory"
        )
        assert system_probes.probe_cache.guest_cpu == {}

    async def test_a_signal_can_pass(self, qemu: Path) -> None:
        self._script(qemu, "kill -KILL $$")
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED):
            await system_probes.require_guest_cpu("m,enforce", accel="kvm")
        assert system_probes.probe_cache.guest_cpu == {}

    async def test_no_answer_in_time_can_pass(self, qemu: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_probes, "_PROBE_TIMEOUT_SECONDS", 0.2)
        self._script(qemu, "exec sleep 30")
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED):
            await system_probes.require_guest_cpu("m,enforce", accel="kvm")

    async def test_a_binary_this_user_cannot_run(self, qemu: Path) -> None:
        """The spawn fails (EACCES): HOST_DEPENDENCY naming the binary, through the probe errors
        of every QEMU probe (_qemu_probe_failure)."""
        qemu.write_text("#!/bin/sh\nexit 0\n")
        qemu.chmod(0o644)
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc:
            await system_probes.require_guest_cpu("m,enforce", accel="kvm")
        assert exc.value.context == {"binary": str(qemu)}

    async def test_the_kvm_rate(self, qemu: Path) -> None:
        """Under KVM the vCPU's TSC rate is the qom-get answer (an int); quit's {} is not one."""
        self._script(
            qemu,
            "cat >/dev/null\n"
            'echo \'{"QMP": {"version": {}}}\'\necho \'{"return": {}}\'\necho \'{"return": 2700000000}\'\necho \'{"return": {}}\'',
        )
        assert await system_probes.require_guest_cpu("m,enforce", accel="kvm") == 2_700_000_000

    @pytest.mark.parametrize(
        "bad_line",
        ["echo '{not json'", "yes '{\"a\":' | head -n 200000 | tr -d '\\n'; echo"],
        ids=["not-json", "object-nested-too-deep"],
    )
    async def test_a_line_qemu_should_not_send_is_skipped(self, qemu: Path, bad_line: str) -> None:
        """A reply line that starts with { but is no JSON, or an object nested too deep for the
        parser (a broken QEMU), is skipped, never a raw JSONDecodeError or RecursionError: the
        rate comes from the lines that parse. 200,000 levels: the parser of every Python build
        measured refuses them (RecursionError; 3.14 builds reach about 52,000 to 116,000 levels)."""
        self._script(
            qemu,
            "cat >/dev/null\n"
            'echo \'{"QMP": {"version": {}}}\'\necho \'{"return": {}}\'\n'
            f"{bad_line}\n"
            "echo '{\"return\": 2700000000}'\necho '{\"return\": {}}'",
        )
        assert await system_probes.require_guest_cpu("m,enforce", accel="kvm") == 2_700_000_000


class TestHostReadLimits:
    """The counter rate and CPU identity reads: a host limit is TransientError LOCAL_DISK and
    is not kept (the next call reads again); another read error is no source (None, kept)."""

    @pytest.fixture(autouse=True)
    def linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_probes, "detect_host_os", lambda: HostOS.LINUX)

    async def test_counter_read_at_a_limit_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_probes, "detect_host_arch", lambda: HostArch.X86_64)
        cpuinfo = "flags\t\t: fpu tsc tsc_known_freq\nbogomips\t: 5400.00\n"
        answers: list[object] = [OSError(errno.EMFILE, "Too many open files"), cpuinfo]

        def read_text(_self: Path) -> str:
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return cast("str", answer)

        monkeypatch.setattr(Path, "read_text", read_text)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
            await system_probes.probe_host_counter_hz()
        assert await system_probes.probe_host_counter_hz() == 2_700_000_000

    @pytest.mark.parametrize(
        "error",
        [
            OSError(errno.EAGAIN, "Resource temporarily unavailable"),
            OSError(errno.ENOMEM, "Cannot allocate memory"),
            OSError(errno.EMFILE, "Too many open files"),
            OSError(errno.ENFILE, "Too many open files in system"),
            TimeoutError(),
        ],
        ids=["eagain", "enomem", "emfile", "enfile", "timeout"],
    )
    async def test_each_limit_can_pass(self, monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
        """macOS reads the rate with sysctl: a limit at its spawn, or no answer in time."""
        monkeypatch.setattr(system_probes, "detect_host_os", lambda: HostOS.MACOS)
        monkeypatch.setattr(system_probes, "detect_host_arch", lambda: HostArch.AARCH64)
        monkeypatch.setattr(system_probes, "run_probe", AsyncMock(side_effect=error))
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
            await system_probes.probe_host_counter_hz()

    async def test_counter_unreadable_is_no_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_probes, "detect_host_arch", lambda: HostArch.X86_64)
        reads: list[Path] = []

        def read_text(self: Path) -> str:
            reads.append(self)
            raise PermissionError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(Path, "read_text", read_text)
        assert await system_probes.probe_host_counter_hz() is None
        assert await system_probes.probe_host_counter_hz() is None
        assert reads == [Path("/proc/cpuinfo")]  # kept

    @pytest.mark.parametrize(
        ("error", "limit"),
        [(OSError(errno.ENFILE, "Too many open files in system"), True), (PermissionError(errno.EACCES, "no"), False)],
        ids=["limit", "unreadable"],
    )
    async def test_cpu_id_read(self, monkeypatch: pytest.MonkeyPatch, error: OSError, limit: bool) -> None:
        monkeypatch.setattr(system_probes, "detect_host_arch", lambda: HostArch.AARCH64)

        def fails(_cpus: Path) -> str | None:
            raise error

        monkeypatch.setattr(system_probes, "_linux_arm_cpu_id", fails)
        if limit:
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                await system_probes.probe_host_cpu_id()
        else:
            assert await system_probes.probe_host_cpu_id() is None  # guest_cpu_id: HOST_DEPENDENCY


class TestCgroupPidsLimitWarning:
    """warn_cgroup_pids_limit: the tightest pids.max from this process's cgroup up to the mount
    root (the innermost of equal ones), warned below 8192; nothing above the mount root is read."""

    @pytest.fixture
    def mount(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A cgroup v2 mount at tmp_path/mount, with this process in a/b/c."""
        root = tmp_path / "mount"
        (root / "a" / "b" / "c").mkdir(parents=True)
        (tmp_path / "pids.max").write_text("1\n")  # above the mount root: never read
        monkeypatch.setattr(system_probes, "CGROUP_V2_BASE_PATH", str(root))
        monkeypatch.setattr(system_probes, "cgroup_v2_available", lambda: True)
        monkeypatch.setattr(system_probes, "resolve_self_cgroup_v2", lambda: "/a/b/c")
        return root

    @pytest.mark.parametrize(
        ("limits", "warned"),
        [
            ({".": "max\n", "a": "6000\n", "a/b": "max\n", "a/b/c": "4096\n"}, (4096, "a/b/c")),
            ({"a": "4096\n", "a/b/c": "4096\n"}, (4096, "a/b/c")),
            ({".": "100\n", "a/b": "9000\n"}, (100, ".")),
            ({"a": "8192\n", "a/b": "max\n"}, None),
            ({}, None),
        ],
        ids=["tightest", "equal-innermost", "mount-root", "high-enough", "no-limit"],
    )
    def test_the_tightest_limit_is_warned(
        self,
        mount: Path,
        caplog: pytest.LogCaptureFixture,
        limits: dict[str, str],
        warned: tuple[int, str] | None,
    ) -> None:
        for folder, value in limits.items():
            (mount / folder / "pids.max").write_text(value)
        with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
            system_probes.warn_cgroup_pids_limit()
        records = [r for r in caplog.records if r.getMessage().startswith("cgroup pids.max is low")]
        assert [(getattr(r, "current", None), getattr(r, "cgroup", None)) for r in records] == (
            [] if warned is None else [warned]
        )

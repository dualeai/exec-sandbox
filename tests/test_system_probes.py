"""system_probes: what a host without access to /dev/kvm is told, the setpriv probe, the O_DIRECT
probes, the POSIX ACL read, and the probe cache."""

import asyncio
import errno
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from exec_sandbox import system_probes
from exec_sandbox.platform_utils import HostOS
from exec_sandbox.system_probes import check_kvm_available, probe_setpriv_pdeathsig


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
        monkeypatch.setattr(system_probes.asyncio, "create_subprocess_exec", _no_subprocess)

        with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
            first = await check_kvm_available()
            second = await check_kvm_available()

        assert first is second is False
        warnings = [r for r in caplog.records if "kvm group" in r.getMessage()]
        if accessible:
            assert warnings == []
        else:
            (record,) = warnings
            assert record.levelno == logging.WARNING
            assert "sudo usermod -aG kvm $USER" in record.getMessage()


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
        assert len(warnings) == (0 if works else 1)
        assert works or "util-linux 2.33" in warnings[0]

    async def test_missing_setpriv_is_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", str(tmp_path))  # no setpriv there
        monkeypatch.setattr(system_probes, "detect_host_os", lambda: HostOS.LINUX)
        assert await probe_setpriv_pdeathsig() is False

    async def test_false_off_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_probes, "detect_host_os", lambda: HostOS.MACOS)
        assert await probe_setpriv_pdeathsig() is False


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

    async def test_another_probe_error_is_this_saves_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No space, a quota, file handles: the device may take O_DIRECT; the next save probes again."""
        flags_seen = self._refusing_open(monkeypatch, errno.ENOSPC)
        with caplog.at_level(logging.WARNING, logger="exec_sandbox"):
            assert await system_probes.probe_direct_io(tmp_path) is False
            assert await system_probes.probe_direct_io(tmp_path) is False

        assert len(flags_seen) == 2  # not cached
        assert all(r.getMessage().startswith("O_DIRECT probe of the folder failed") for r in caplog.records)
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
    there (cached per device), not the name of the OS."""

    @pytest.fixture(autouse=True)
    def _fresh_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_probes.probe_cache, "posix_acls", {})

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

    def test_a_filesystem_without_posix_acls_is_asked_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EOPNOTSUPP (no ACL support, mounted noacl): None, and the device is not asked again."""
        names = self._getxattr(monkeypatch, errno.EOPNOTSUPP)
        assert system_probes.posix_acl_xattr(tmp_path) is None
        assert system_probes.posix_acl_xattr(tmp_path) is None
        assert names == [system_probes.POSIX_ACL_ACCESS_XATTR]

    def test_a_file_with_no_acl_is_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENODATA: the filesystem keeps POSIX ACLs, this file has none beyond its mode."""
        names = self._getxattr(monkeypatch, errno.ENODATA)
        assert system_probes.posix_acl_xattr(tmp_path) is None
        assert system_probes.posix_acl_xattr(tmp_path) is None
        assert len(names) == 2  # each file is read: the device is not marked as without ACLs

    def test_a_file_with_an_acl_gives_it(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._getxattr(monkeypatch, b"\x02\x00\x00\x00")
        assert system_probes.posix_acl_xattr(tmp_path) == b"\x02\x00\x00\x00"

    def test_the_real_call(self, tmp_path: Path) -> None:
        """A new file has no ACL beyond its mode, wherever this runs: None."""
        (tmp_path / "f").write_text("x")
        assert system_probes.posix_acl_xattr(tmp_path / "f") is None


class TestProbeCache:
    """The cache of probe results: its locks and its reset, as the probes and the tests use them."""

    @pytest.mark.parametrize("attr", ["direct_io", "posix_acls"])
    async def test_reset_of_a_per_device_cache_keeps_it_a_dict(self, attr: str, tmp_path: Path) -> None:
        """reset(attr) of a per-device cache gives an empty dict, not the sentinel: the next
        probe of that kind works."""
        cache = system_probes.ProbeCache()
        getattr(cache, attr)[1] = True
        cache.reset(attr)
        assert getattr(cache, attr) == {}
        system_probes.probe_cache.reset(attr)
        if attr == "direct_io":
            assert await system_probes.probe_direct_io(tmp_path) in (True, False)
        else:
            assert system_probes.posix_acl_xattr(tmp_path) is None  # a new folder: no ACL beyond its mode

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

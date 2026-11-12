"""system_probes: what a host without access to /dev/kvm is told."""

import logging
from pathlib import Path

import pytest

from exec_sandbox import system_probes
from exec_sandbox.system_probes import check_kvm_available


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

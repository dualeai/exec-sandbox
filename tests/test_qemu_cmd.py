"""The QEMU binary a VM runs is the one the version and accelerator probes ran
(qemu_cmd.build_qemu_cmd and system_probes share system_probes.qemu_system_binary).

A fake layout in tmp_path: a configured file (EXEC_SANDBOX_QEMU_BIN_X86 / _ARM) and
another qemu-system on PATH, each a script that answers with its own version.
"""

import shutil
import stat
from collections.abc import AsyncIterator
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostArch, HostOS, detect_host_arch
from exec_sandbox.qemu_cmd import build_qemu_cmd
from exec_sandbox.settings import Settings
from exec_sandbox.system_probes import probe_qemu_accelerators, probe_qemu_version
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType
from exec_sandbox.vm_working_directory import VmWorkingDirectory

_NAME = "qemu-system-aarch64" if detect_host_arch() == HostArch.AARCH64 else "qemu-system-x86_64"
_VARIABLE = "EXEC_SANDBOX_QEMU_BIN_ARM" if detect_host_arch() == HostArch.AARCH64 else "EXEC_SANDBOX_QEMU_BIN_X86"


def _fake_qemu(path: Path, *, version: str, accelerator: str) -> Path:
    """A qemu-system stand-in: `--version` and `-accel help` answer as that QEMU would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "--version" ]; then echo "QEMU emulator version {version}"; exit 0; fi\n'
        f'if [ "$1" = "-accel" ]; then printf "Accelerators supported in QEMU binary:\\ntcg\\n{accelerator}\\n"; exit 0; fi\n'
        "exit 1\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
async def workdir(tmp_path: Path) -> AsyncIterator[VmWorkingDirectory]:
    wd = await VmWorkingDirectory.create("qemu-binary", home_dir=tmp_path / "home")
    try:
        yield wd
    finally:
        await wd.cleanup()


async def _launched_binary(workdir: VmWorkingDirectory, tmp_path: Path) -> str:
    """argv[0] of the VM command, from the Settings a Scheduler builds (the variables);
    every probe but the QEMU version is a fixed answer."""
    settings = Settings(base_images_dir=tmp_path, kernel_path=tmp_path)
    with ExitStack() as stack:
        for name, value in [
            ("detect_host_os", HostOS.LINUX),
            ("check_tsc_deadline", True),
            ("probe_unshare_support", False),
            ("probe_io_uring_support", False),
            ("probe_qemu_sandbox_support", False),
            ("get_qemu_vm_uid", None),
        ]:
            stack.enter_context(patch(f"exec_sandbox.qemu_cmd.{name}", return_value=value))
        stack.enter_context(patch("exec_sandbox.qemu_cmd.detect_accel_type", AsyncMock(return_value=AccelType.TCG)))
        cmd = await build_qemu_cmd(
            settings, detect_host_arch(), "qemu-binary", workdir, VmSpec(language=Language.PYTHON)
        )
    return cmd[0]


class TestOneQemuBinary:
    async def test_configured_file_wins_for_probes_and_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory
    ) -> None:
        """The configured file exists: the probes ask it, and the VM runs it, not the QEMU on PATH."""
        configured = _fake_qemu(tmp_path / "configured" / _NAME, version="11.0.0", accelerator="kvm")
        _fake_qemu(tmp_path / "on-path" / _NAME, version="9.2.0", accelerator="hvf")
        monkeypatch.setenv("PATH", f"{tmp_path / 'on-path'}:/usr/bin:/bin")
        monkeypatch.setenv(_VARIABLE, str(configured))

        assert await probe_qemu_version() == (11, 0, 0)
        assert await probe_qemu_accelerators() == {"tcg", "kvm"}
        assert await _launched_binary(workdir, tmp_path) == str(configured)

    async def test_missing_file_falls_back_to_path_for_probes_and_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory
    ) -> None:
        """The configured file does not exist: the probes and the VM both run the name on PATH."""
        on_path = _fake_qemu(tmp_path / "on-path" / _NAME, version="9.2.0", accelerator="hvf")
        path = f"{tmp_path / 'on-path'}:/usr/bin:/bin"
        monkeypatch.setenv("PATH", path)
        monkeypatch.setenv(_VARIABLE, str(tmp_path / "missing" / _NAME))

        assert await probe_qemu_version() == (9, 2, 0)
        assert await probe_qemu_accelerators() == {"tcg", "hvf"}
        launched = await _launched_binary(workdir, tmp_path)
        assert launched == _NAME
        assert shutil.which(launched, path=path) == str(on_path)

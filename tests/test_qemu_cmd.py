"""The QEMU binary a VM runs is the one the version and accelerator probes ran
(qemu_cmd.build_qemu_cmd and system_probes share system_probes.qemu_system_binary).

A fake layout in tmp_path: a configured file (EXEC_SANDBOX_QEMU_BIN_X86 / _ARM) and
another qemu-system on PATH, each a script that answers with its own version.
"""

import errno
import re
import shutil
import stat
from collections.abc import AsyncIterator
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from exec_sandbox import constants
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostArch, HostOS, detect_host_arch
from exec_sandbox.qemu_cmd import build_qemu_cmd
from exec_sandbox.settings import Settings
from exec_sandbox.system_probes import probe_qemu_accelerators, probe_qemu_version
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType
from exec_sandbox.vm_working_directory import VmWorkingDirectory
from tests.test_subprocess_utils import write_foreign_elf

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
    """argv[0] of the VM command, from the Settings a Scheduler builds (the variables)."""
    return (await _command(workdir, tmp_path, sandbox=False, unshare=False))[0]


async def _command(workdir: VmWorkingDirectory, tmp_path: Path, *, sandbox: bool, unshare: bool) -> list[str]:
    """The VM command; every probe but the QEMU version is a fixed answer (the seccomp probe:
    `sandbox`; the namespace probe: `unshare`)."""
    settings = Settings(base_images_dir=tmp_path, kernel_path=tmp_path)
    with ExitStack() as stack:
        for name, value in [
            ("detect_host_os", HostOS.LINUX),
            ("check_tsc_deadline", True),
            ("probe_unshare_support", unshare),
            ("probe_io_uring_support", False),
            ("probe_qemu_sandbox_support", sandbox),
            ("get_qemu_vm_uid", None),
            ("detect_host_x86_vendor", "AuthenticAMD"),
        ]:
            stack.enter_context(patch(f"exec_sandbox.qemu_cmd.{name}", return_value=value))
        stack.enter_context(patch("exec_sandbox.qemu_cmd.detect_accel_type", AsyncMock(return_value=AccelType.TCG)))
        return await build_qemu_cmd(
            settings, detect_host_arch(), "qemu-binary", workdir, VmSpec(language=Language.PYTHON)
        )


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


class TestQemuThisHostCannotRun:
    """A QEMU binary that this host cannot exec: behind unshare or stdbuf (glibc execvp),
    or under busybox ash, it would run as a shell script and end as QEMU_CRASHED after the
    boot retries. The version probe execs it through Popen, which reports ENOEXEC: the VM
    command is never built, and VmManager.start raises it too."""

    @pytest.mark.parametrize("kind", ["other-arch", "text-no-shebang"])
    async def test_no_vm_command_for_a_binary_this_host_cannot_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory, kind: str
    ) -> None:
        binary = tmp_path / "configured" / _NAME
        binary.parent.mkdir()
        if kind == "other-arch":
            write_foreign_elf(binary)
        else:
            binary.write_text("echo QEMU emulator version 11.1.3\n")  # a shell would run it
            binary.chmod(0o755)
        monkeypatch.setenv(_VARIABLE, str(binary))

        assert await probe_qemu_version() is None
        with pytest.raises(
            ConfigError,
            check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY,
            match=rf"^Failed to start {re.escape(str(binary))}: ",
        ) as exc_info:
            await _command(workdir, tmp_path, sandbox=False, unshare=True)
        assert exc_info.value.context == {"binary": str(binary), "errno": errno.ENOEXEC}

    @pytest.mark.parametrize("kind", ["unparsable-version", "no-exec-bit"])
    async def test_other_probe_failures_build_the_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory, kind: str
    ) -> None:
        """An unparsable version, or a binary this user may not run (EACCES: qemu-vm may), stays
        a warning: only a binary that this host cannot exec at all stops."""
        binary = _fake_qemu(tmp_path / "configured" / _NAME, version="x", accelerator="kvm")
        if kind == "no-exec-bit":
            binary.chmod(0o644)
        monkeypatch.setenv(_VARIABLE, str(binary))
        assert await probe_qemu_version() is None
        assert (await _command(workdir, tmp_path, sandbox=False, unshare=False))[0] == str(binary)


class TestHostDiskFull:
    """A write that finds the host disk full pauses the VM and reaches QemuVM's event watch."""

    async def test_home_drive_stops_on_no_space(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory
    ) -> None:
        """werror=enospc: QEMU sets nospace=true in BLOCK_IO_ERROR only under this policy, and the
        guest never sees the error (with werror=report its ext4 goes read-only)."""
        monkeypatch.setenv(_VARIABLE, str(_fake_qemu(tmp_path / "qemu" / _NAME, version="11.1.3", accelerator="kvm")))
        cmd = await _command(workdir, tmp_path, sandbox=False, unshare=False)

        home = [arg for arg in cmd if arg.startswith(f"file={workdir.home_disk},")]
        assert len(home) == 1
        assert "werror=enospc" in home[0].split(",")

    async def test_events_have_their_own_monitor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory
    ) -> None:
        """The control monitor serves one client at a time (saves, diagnostics): events get a second one."""
        monkeypatch.setenv(_VARIABLE, str(_fake_qemu(tmp_path / "qemu" / _NAME, version="11.1.3", accelerator="kvm")))
        cmd = await _command(workdir, tmp_path, sandbox=False, unshare=False)

        chardev = f"socket,id=qmpev0,path={workdir.qmp_events_socket},server=on,wait=off"
        assert cmd[cmd.index(chardev) - 1] == "-chardev"
        assert [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-mon"] == ["chardev=qmpev0,mode=control"]


class TestGuestPorts:
    async def test_agent_ports_have_the_ids_of_the_event_watch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory
    ) -> None:
        """QEMU sends VSERPORT_CHANGE only for a port with an id: QemuVM's event watch knows
        the agent's ports by these ids (a close fails the calls in flight)."""
        monkeypatch.setenv(_VARIABLE, str(_fake_qemu(tmp_path / "qemu" / _NAME, version="11.1.3", accelerator="kvm")))
        cmd = await _command(workdir, tmp_path, sandbox=False, unshare=False)

        ports = [arg.split(",") for arg in cmd if arg.startswith("virtserialport,")]
        ids = {
            next(o for o in port if o.startswith("name=")): next(o for o in port if o.startswith("id="))
            for port in ports
        }
        assert ids == {
            "name=org.dualeai.cmd": f"id={constants.GUEST_PORT_IDS[0]}",
            "name=org.dualeai.event": f"id={constants.GUEST_PORT_IDS[1]}",
        }


class TestSandbox:
    async def test_vms_install_the_probed_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory
    ) -> None:
        """A host whose QEMU accepts the filter: every VM gets the full filter (each deny
        option stops one kind of host access)."""
        monkeypatch.setenv(_VARIABLE, str(_fake_qemu(tmp_path / "qemu" / _NAME, version="11.1.3", accelerator="kvm")))
        cmd = await _command(workdir, tmp_path, sandbox=True, unshare=False)

        assert cmd[cmd.index("-sandbox") + 1] == (
            "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny"
        )


class TestNamespaces:
    async def test_qemu_in_namespaces_ends_with_unshare(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory
    ) -> None:
        """exit-with-parent ties QEMU to unshare, which outlives a killed host process unless it
        has a parent-death signal (start_managed_process, end_with_this_process): --kill-child then ends QEMU."""
        monkeypatch.setenv(_VARIABLE, str(_fake_qemu(tmp_path / "qemu" / _NAME, version="11.1.3", accelerator="kvm")))
        cmd = await _command(workdir, tmp_path, sandbox=False, unshare=True)

        assert cmd[0] == "unshare"
        assert cmd[cmd.index("--") - 1] == "--kill-child=SIGKILL"

"""The QEMU binary a VM runs is the one the version and accelerator probes ran
(qemu_cmd.build_qemu_cmd and system_probes share system_probes.qemu_system_binary).

A fake layout in tmp_path: a configured file (EXEC_SANDBOX_QEMU_BIN_X86 / _ARM) and
another qemu-system on PATH, each a script that answers with its own version.
"""

import errno
import fcntl
import re
import shutil
import stat
from collections.abc import AsyncIterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from exec_sandbox import constants, system_probes
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.models import Language
from exec_sandbox.platform_utils import HostArch, HostOS, detect_host_arch
from exec_sandbox.qemu_cmd import CounterRate, build_qemu_cmd, counter_rate, guest_cpu_id, resolve_machine_type
from exec_sandbox.settings import Settings
from exec_sandbox.system_probes import (
    probe_qemu_accelerators,
    probe_qemu_machine_types,
    probe_qemu_version,
    require_qemu_machine_types,
    require_qemu_version,
)
from exec_sandbox.vm_spec import VmSpec
from exec_sandbox.vm_types import AccelType
from exec_sandbox.vm_working_directory import VmWorkingDirectory
from tests.conftest import skip_unless_hwaccel, skip_unless_linux, skip_unless_x86_64
from tests.test_subprocess_utils import write_foreign_elf

_NAME = "qemu-system-aarch64" if detect_host_arch() == HostArch.AARCH64 else "qemu-system-x86_64"
_VARIABLE = "EXEC_SANDBOX_QEMU_BIN_ARM" if detect_host_arch() == HostArch.AARCH64 else "EXEC_SANDBOX_QEMU_BIN_X86"


def _fake_qemu(path: Path, *, version: str, accelerator: str) -> Path:
    """A qemu-system stand-in: `--version`, `-accel help` and `-machine help` answer as that QEMU
    would (the machine list in QEMU's format: an alias line, then the versioned type of that
    QEMU, virt-MAJOR.MINOR)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    virt = "virt-" + ".".join(version.split(".")[:2])
    path.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "--version" ]; then echo "QEMU emulator version {version}"; exit 0; fi\n'
        f'if [ "$1" = "-accel" ]; then printf "Accelerators supported in QEMU binary:\\ntcg\\n{accelerator}\\n"; exit 0; fi\n'
        'if [ "$1" = "-machine" ]; then printf "Supported machines are:\\n'
        f"virt                 QEMU ARM Virtual Machine (alias of {virt})\\n"
        f"{virt}            QEMU ARM Virtual Machine\\n"
        'none                 empty machine\\n"; exit 0; fi\n'
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
            ("require_qemu_sandbox_support", sandbox),
            ("detect_host_x86_vendor", "AuthenticAMD"),
        ]:
            stack.enter_context(patch(f"exec_sandbox.qemu_cmd.{name}", return_value=value))
        stack.enter_context(patch("exec_sandbox.qemu_cmd.detect_accel_type", AsyncMock(return_value=AccelType.TCG)))
        return await build_qemu_cmd(
            settings,
            detect_host_arch(),
            "qemu-binary",
            workdir,
            VmSpec(language=Language.PYTHON),
            direct_write=False,
            debug_boot=False,
            snapshot_drive=None,
            defer_incoming=False,
            qmp_fd=None,
            guest_port_fds=None,
            mem_path=None,
            mem_path_share=False,
            machine_type=None,
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
        assert await probe_qemu_machine_types() == {"virt": "virt-11.0", "virt-11.0": "virt-11.0", "none": "none"}
        assert await _launched_binary(workdir, tmp_path) == str(configured)

    async def test_missing_file_falls_back_to_path_for_probes_and_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory
    ) -> None:
        """The configured file does not exist: the probes and the VM both run the file of the
        name on PATH, by its full path (sudo's PATH may differ)."""
        on_path = _fake_qemu(tmp_path / "on-path" / _NAME, version="9.2.0", accelerator="hvf")
        path = f"{tmp_path / 'on-path'}:/usr/bin:/bin"
        monkeypatch.setenv("PATH", path)
        monkeypatch.setenv(_VARIABLE, str(tmp_path / "missing" / _NAME))

        assert await probe_qemu_version() == (9, 2, 0)
        assert await probe_qemu_accelerators() == {"tcg", "hvf"}
        assert await _launched_binary(workdir, tmp_path) == str(on_path) == shutil.which(_NAME, path=path)


class TestQemuMachineTypesUnreadable:
    @pytest.mark.parametrize("kind", ["exit-1", "exec-error"])
    async def test_no_machine_type_without_the_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
    ) -> None:
        """`-machine help` fails (an exit code, or an exec error): no machine type, so no VM
        launch (its -M type is kept for a save): HOST_DEPENDENCY names the binary."""
        binary = _fake_qemu(tmp_path / "qemu" / _NAME, version="11.1.3", accelerator="kvm")
        binary.write_text(binary.read_text().replace('if [ "$1" = "-machine" ]', 'if [ "$1" = "-machine-x" ]'))
        monkeypatch.setenv(_VARIABLE, str(binary))
        if kind == "exec-error":
            monkeypatch.setattr(
                "exec_sandbox.system_probes.run_probe", AsyncMock(side_effect=PermissionError(13, "no"))
            )
        assert await probe_qemu_machine_types() == {}
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc_info:
            await resolve_machine_type(detect_host_arch(), None)
        assert str(exc_info.value) == f"The QEMU machine types could not be read ({binary} -machine help)."
        assert exc_info.value.context == {"binary": str(binary)}


class TestFailedQemuProbe:
    async def test_a_failed_probe_runs_again(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A QEMU probe that fails (a timeout, a fork error, a binary half written by an
        upgrade) is not cached: the next call asks QEMU again, and a real answer is cached."""
        monkeypatch.setenv(_VARIABLE, str(_fake_qemu(tmp_path / "qemu" / _NAME, version="11.1.3", accelerator="kvm")))
        answers: list[object] = [TimeoutError(), (0, b"QEMU emulator version 11.1.3\n", b"")]
        probe = AsyncMock(side_effect=answers)
        monkeypatch.setattr("exec_sandbox.system_probes.run_probe", probe)

        assert await probe_qemu_version() is None
        assert await probe_qemu_version() == (11, 1, 3)
        assert await probe_qemu_version() == (11, 1, 3)  # cached
        assert probe.await_count == 2


class TestQemuProbeErrors:
    @pytest.mark.parametrize(
        ("failure", "error"),
        [
            pytest.param(TimeoutError(), TransientError, id="no-answer-in-time"),
            pytest.param(OSError(errno.EAGAIN, "Resource temporarily unavailable"), TransientError, id="fork-limit"),
            pytest.param(PermissionError(errno.EACCES, "Permission denied"), ConfigError, id="no-access"),
            pytest.param(OSError(errno.ETXTBSY, "Text file busy"), TransientError, id="binary-being-written"),
            pytest.param((-9, b"", b""), TransientError, id="killed-by-a-signal"),
            pytest.param((1, b"", b""), ConfigError, id="exits-with-an-error"),
        ],
    )
    @pytest.mark.parametrize("probe", ["version", "machine-types", "resolve-machine-type"])
    async def test_a_probe_failure_that_can_pass_is_transient(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure: OSError | tuple[int, bytes, bytes],
        error: type[Exception],
        probe: str,
    ) -> None:
        """No version or machine types because the probe hit a fork limit, no answer in time, a
        binary being written, or a signal that killed QEMU: TransientError QEMU_CRASHED (the next
        call probes again), not FIX_CONFIG; a binary this user cannot run, or QEMU's own error
        exit, stays ConfigError HOST_DEPENDENCY."""
        monkeypatch.setenv(_VARIABLE, str(_fake_qemu(tmp_path / "qemu" / _NAME, version="11.1.3", accelerator="kvm")))
        answer = {"side_effect": failure} if isinstance(failure, BaseException) else {"return_value": failure}
        monkeypatch.setattr("exec_sandbox.system_probes.run_probe", AsyncMock(**answer))
        ask = {
            "version": require_qemu_version,
            "machine-types": require_qemu_machine_types,
            "resolve-machine-type": lambda: resolve_machine_type(HostArch.X86_64, None),
        }[probe]
        with pytest.raises(error) as exc_info:
            await ask()
        code = ErrorCode.QEMU_CRASHED if error is TransientError else ErrorCode.HOST_DEPENDENCY
        assert getattr(exc_info.value, "code", None) is code

    async def test_no_answer_in_time_says_so(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A probe timeout (a TimeoutError with no text) names the missing answer."""
        binary = _fake_qemu(tmp_path / "qemu" / _NAME, version="11.1.3", accelerator="kvm")
        monkeypatch.setenv(_VARIABLE, str(binary))
        monkeypatch.setattr("exec_sandbox.system_probes.run_probe", AsyncMock(side_effect=TimeoutError()))
        with pytest.raises(TransientError, match=r" --version \(no answer in time\)\.$"):
            await require_qemu_version()

    async def test_no_accelerator_answer_is_no_hvf_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The kernel supports HVF but the accelerator probe got no answer in time: the HVF
        check raises it (TransientError), so no VM is sized for one accelerator and launched
        under another; the next check asks QEMU again."""
        monkeypatch.setenv(_VARIABLE, str(_fake_qemu(tmp_path / "qemu" / _NAME, version="11.1.3", accelerator="hvf")))

        answers: list[object] = [TimeoutError(), (0, b"Accelerators supported in QEMU binary:\ntcg\nhvf\n", b"")]

        async def probe(cmd: tuple[str, ...], **_kwargs: object) -> tuple[int, bytes, bytes]:
            if cmd[0] == "/usr/sbin/sysctl":
                return 0, b"1\n", b""  # kern.hv_support
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer  # type: ignore[return-value]

        monkeypatch.setattr("exec_sandbox.system_probes.run_probe", probe)
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.QEMU_CRASHED):
            await system_probes.check_hvf_available()
        assert await system_probes.check_hvf_available() is True


class TestNoQemuBinary:
    async def test_no_binary_names_the_variable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No QEMU file at the configured path and none on PATH: the error says so and names
        the variable that sets the path (not "could not be read")."""
        monkeypatch.setenv(_VARIABLE, str(tmp_path / "missing" / _NAME))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc_info:
            await require_qemu_version()
        assert str(exc_info.value) == f"No QEMU binary {_NAME} on this host."
        assert exc_info.value.hint == f"Install QEMU, or set {_VARIABLE} to its path."

    async def test_a_file_this_user_cannot_run_is_not_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A QEMU file on PATH without the exec bit: "could not be read", not "No QEMU binary"."""
        binary = _fake_qemu(tmp_path / "on-path" / _NAME, version="11.1.3", accelerator="kvm")
        binary.chmod(0o644)
        monkeypatch.setenv(_VARIABLE, str(tmp_path / "missing" / _NAME))
        monkeypatch.setenv("PATH", str(binary.parent))
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc_info:
            await require_qemu_version()
        assert str(exc_info.value) == f"The QEMU version could not be read ({binary} --version)."


class TestQemuOnPath:
    async def test_a_qemu_moved_on_path_is_found_again(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """QEMU moves to another folder of PATH: the probes find and ask its new file (here a
        newer QEMU), not the gone one ("No QEMU binary")."""
        first = _fake_qemu(tmp_path / "a" / _NAME, version="11.0.0", accelerator="kvm")
        monkeypatch.setenv(_VARIABLE, str(tmp_path / "missing" / _NAME))
        monkeypatch.setenv("PATH", f"{tmp_path / 'a'}:{tmp_path / 'b'}")
        assert await require_qemu_version() == (11, 0, 0)
        first.unlink()
        _fake_qemu(tmp_path / "b" / _NAME, version="11.0.5", accelerator="kvm")
        assert await require_qemu_version() == (11, 0, 5)

    async def test_a_lower_case_variable_is_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The QEMU path variable in lower case: the probes run that file, as the VMs do."""
        binary = _fake_qemu(tmp_path / "qemu" / _NAME, version="11.0.7", accelerator="kvm")  # no real QEMU's
        monkeypatch.delenv(_VARIABLE, raising=False)
        monkeypatch.setenv(_VARIABLE.lower(), str(binary))
        assert await require_qemu_version() == (11, 0, 7)


class TestQemuUpgradeUnderAWorker:
    async def test_a_new_binary_at_the_same_path_is_probed_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A package upgrade writes a new QEMU binary at the same path while the worker runs:
        the next QEMU probes ask the new binary, so saves and command gates never use the old
        version. An unchanged binary is not run again."""
        binary = _fake_qemu(tmp_path / "qemu" / _NAME, version="11.0.0", accelerator="kvm")
        monkeypatch.setenv(_VARIABLE, str(binary))
        assert await probe_qemu_version() == (11, 0, 0)
        assert await probe_qemu_accelerators() == {"tcg", "kvm"}
        with patch("exec_sandbox.system_probes.run_probe", side_effect=AssertionError("the QEMU probe ran again")):
            assert await probe_qemu_version() == (11, 0, 0)
            assert await probe_qemu_accelerators() == {"tcg", "kvm"}

        _fake_qemu(tmp_path / "new" / _NAME, version="11.1.3", accelerator="hvf").replace(binary)  # a new file

        assert await probe_qemu_version() == (11, 1, 3)
        assert await probe_qemu_accelerators() == {"tcg", "hvf"}
        assert (await probe_qemu_machine_types())["virt"] == "virt-11.1"

    async def test_an_upgrade_during_the_probe_is_probed_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The binary is replaced while its probe runs: the answer is kept with the stamp of
        the binary it came from (taken before the probe), so the next call asks the new one."""
        binary = _fake_qemu(tmp_path / "qemu" / _NAME, version="11.0.0", accelerator="kvm")
        monkeypatch.setenv(_VARIABLE, str(binary))
        real = system_probes.run_probe

        async def upgraded_meanwhile(*args: Any, **kwargs: Any) -> Any:
            answer = await real(*args, **kwargs)  # the old binary answers
            if not (tmp_path / "new").exists():
                _fake_qemu(tmp_path / "new" / _NAME, version="11.1.3", accelerator="kvm").replace(binary)
            return answer

        monkeypatch.setattr("exec_sandbox.system_probes.run_probe", upgraded_meanwhile)
        assert await probe_qemu_version() == (11, 0, 0)
        assert await probe_qemu_version() == (11, 1, 3)


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
    async def test_no_vm_command_without_a_qemu_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workdir: VmWorkingDirectory, kind: str
    ) -> None:
        """An unparsable version, or a binary this user may not run (EACCES): every VM needs the
        version (command line, migration, saved facts), so no command is built."""
        binary = _fake_qemu(tmp_path / "configured" / _NAME, version="x", accelerator="kvm")
        if kind == "no-exec-bit":
            binary.chmod(0o644)
        monkeypatch.setenv(_VARIABLE, str(binary))
        assert await probe_qemu_version() is None
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY) as exc_info:
            await _command(workdir, tmp_path, sandbox=False, unshare=False)
        assert str(exc_info.value) == f"The QEMU version could not be read ({binary} --version)."
        assert exc_info.value.context == {"binary": str(binary)}


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

    @pytest.mark.parametrize(
        ("version", "monitor"),
        [
            ("11.1.3", ["-object", "monitor-qmp,id=mon-qmpev0,chardev=qmpev0"]),  # 11.1 deprecates -mon
            ("11.0.0", ["-mon", "chardev=qmpev0,mode=control"]),
        ],
    )
    async def test_events_have_their_own_monitor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        workdir: VmWorkingDirectory,
        version: str,
        monitor: list[str],
    ) -> None:
        """The control monitor serves one client at a time (saves, diagnostics): events get a
        second one, a monitor-qmp object from QEMU 11.1 (which warns on -mon), -mon before."""
        monkeypatch.setenv(_VARIABLE, str(_fake_qemu(tmp_path / "qemu" / _NAME, version=version, accelerator="kvm")))
        cmd = await _command(workdir, tmp_path, sandbox=False, unshare=False)

        chardev = f"socket,id=qmpev0,path={workdir.qmp_events_socket},server=on,wait=off"
        at = cmd.index(chardev)
        assert cmd[at - 1] == "-chardev"
        assert cmd[at + 1 : at + 3] == monitor
        assert "-mon" not in cmd[at + 3 :] and "monitor-qmp" not in " ".join(cmd[at + 3 :])


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
        assert ids == {"name=org.dualeai.cmd": "id=guest-cmd", "name=org.dualeai.event": "id=guest-event"}
        assert set(constants.GUEST_PORT_IDS) == {"guest-cmd", "guest-event"}  # the ids the event watch knows


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


class TestResolveMachineType:
    @pytest.mark.parametrize(
        ("arch", "saved", "resolved"),
        [
            pytest.param(HostArch.AARCH64, None, "virt-11.1", id="new-arm64-vm-gets-the-alias-target"),
            pytest.param(HostArch.AARCH64, "virt-11.0", "virt-11.0", id="saved-type-this-qemu-has"),
            pytest.param(HostArch.AARCH64, "virt-12.0", None, id="saved-type-this-qemu-lacks"),
            pytest.param(HostArch.X86_64, None, "microvm", id="x86-not-versioned"),
        ],
    )
    async def test_resolution(
        self, monkeypatch: pytest.MonkeyPatch, arch: HostArch, saved: str | None, resolved: str | None
    ) -> None:
        """The type a VM runs, from the cached machine type probe: the saved type, or the target of
        the release's alias (virt on arm64; microvm, which has no alias, on x86)."""

        async def machine_types() -> dict[str, str]:
            return {"virt": "virt-11.1", "virt-11.1": "virt-11.1", "virt-11.0": "virt-11.0", "microvm": "microvm"}

        monkeypatch.setattr("exec_sandbox.qemu_cmd.require_qemu_machine_types", machine_types)
        assert await resolve_machine_type(arch, saved) == resolved


class TestGuestCpuId:
    """guest_cpu_id: the host CPU a -cpu host guest sees, or no fact where the CPU is a model."""

    @pytest.mark.parametrize(
        ("arch", "accel"),
        [(HostArch.X86_64, AccelType.KVM), (HostArch.AARCH64, AccelType.TCG)],
        ids=["x86-fixed-model", "arm-tcg-fixed-model"],
    )
    async def test_a_fixed_model_has_none(
        self, monkeypatch: pytest.MonkeyPatch, arch: HostArch, accel: AccelType
    ) -> None:
        monkeypatch.setattr("exec_sandbox.qemu_cmd.probe_host_cpu_id", AsyncMock(side_effect=AssertionError("read")))
        assert await guest_cpu_id(arch, accel) is None

    @pytest.mark.parametrize("accel", [AccelType.KVM, AccelType.HVF])
    async def test_an_unread_identity_is_no_start(self, monkeypatch: pytest.MonkeyPatch, accel: AccelType) -> None:
        """ARM64 KVM or HVF with no identity: HOST_DEPENDENCY, never None (None means a fixed
        model: a save would load on any CPU)."""
        monkeypatch.setattr("exec_sandbox.qemu_cmd.probe_host_cpu_id", AsyncMock(return_value=None))
        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.HOST_DEPENDENCY):
            await guest_cpu_id(HostArch.AARCH64, accel)

    async def test_arm_kvm_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cpu_id = "midr_el1=0x00000000410fd4f1,revidr_el1=0x0000000000000017"
        monkeypatch.setattr("exec_sandbox.qemu_cmd.probe_host_cpu_id", AsyncMock(return_value=cpu_id))
        assert await guest_cpu_id(HostArch.AARCH64, AccelType.KVM) == cpu_id


class TestCounterRate:
    """qemu_cmd.counter_rate: each row with stand-in probes, the ARM TCG pin, and the x86 KVM
    row on this host's real probes (run on a KVM host)."""

    @pytest.mark.parametrize(
        ("host_hz", "scaling", "expected"),
        [
            (None, False, CounterRate(hz=None, any_rate=False)),
            (None, True, CounterRate(hz=None, any_rate=True)),
            (2_700_000_000, False, CounterRate(hz=2_700_001_000, any_rate=False)),
            (2_700_000_000, True, CounterRate(hz=2_700_001_000, any_rate=True)),
        ],
        ids=["measured-rate", "measured-rate-scaling", "stated-rate", "stated-rate-scaling"],
    )
    async def test_x86_kvm_rows(
        self, monkeypatch: pytest.MonkeyPatch, host_hz: int | None, scaling: bool, expected: CounterRate
    ) -> None:
        """x86 KVM: the vCPU's rate from QEMU (2_700_001_000 here, not the host probe's value)
        only when the host states its rate; TSC scaling as KVM reports it."""
        monkeypatch.setattr("exec_sandbox.qemu_cmd.probe_host_counter_hz", AsyncMock(return_value=host_hz))
        monkeypatch.setattr("exec_sandbox.qemu_cmd.probe_kvm_tsc_scaling", AsyncMock(return_value=scaling))
        monkeypatch.setattr("exec_sandbox.qemu_cmd.require_guest_cpu", AsyncMock(return_value=2_700_001_000))
        monkeypatch.setattr("exec_sandbox.qemu_cmd.check_tsc_deadline", AsyncMock(return_value=True))
        assert await counter_rate(HostArch.X86_64, AccelType.KVM) == expected

    @pytest.mark.parametrize(
        ("arch", "accel"),
        [(HostArch.X86_64, AccelType.TCG), (HostArch.AARCH64, AccelType.HVF), (HostArch.AARCH64, AccelType.KVM)],
        ids=["x86-tcg", "arm-hvf", "arm-kvm"],
    )
    async def test_other_rows_are_the_hosts_counter(
        self, monkeypatch: pytest.MonkeyPatch, arch: HostArch, accel: AccelType
    ) -> None:
        """A KVM or HVF guest reads the host counter, an x86 TCG guest's TSC is the host's: the
        host probe's rate, no QEMU probe, no scaling."""
        monkeypatch.setattr("exec_sandbox.qemu_cmd.probe_host_counter_hz", AsyncMock(return_value=24_000_000))
        guest_cpu = AsyncMock(side_effect=AssertionError("no QEMU probe"))
        monkeypatch.setattr("exec_sandbox.qemu_cmd.require_guest_cpu", guest_cpu)
        assert await counter_rate(arch, accel) == CounterRate(hz=24_000_000, any_rate=False)

    async def test_arm_tcg_is_the_pinned_rate(self) -> None:
        """The guest CPU pins it (cntfrq): the same on every host, no probe."""
        assert await counter_rate(HostArch.AARCH64, AccelType.TCG) == CounterRate(hz=62_500_000, any_rate=False)

    @skip_unless_linux
    @skip_unless_x86_64
    @skip_unless_hwaccel
    async def test_x86_kvm_rate_is_the_hosts(self) -> None:
        """The rate KVM gives a new vCPU (QEMU's tsc-frequency, from a paused real QEMU), kept
        only when the host states its TSC rate: then the kernel's tsc_khz, which BogoMIPS
        gives in 5 kHz steps (an independent source); else None."""
        rate = await counter_rate(HostArch.X86_64, AccelType.KVM)
        stated = system_probes._bogomips_hz(Path("/proc/cpuinfo").read_text())  # pyright: ignore[reportPrivateUsage]
        if stated is None:
            assert rate.hz is None
        else:
            assert rate.hz is not None
            assert 0 <= rate.hz - stated < 5000
        with Path("/dev/kvm").open("rb") as kvm:  # KVM_CHECK_EXTENSION(KVM_CAP_TSC_CONTROL), read here
            assert rate.any_rate is (fcntl.ioctl(kvm.fileno(), 0xAE03, 60) > 0)

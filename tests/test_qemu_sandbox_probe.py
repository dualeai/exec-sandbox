"""probe_qemu_sandbox_support: decides from QEMU's QMP greeting, not from a fixed wait.

The fake QEMU is a real process (one sh script per module: macOS checks each new
executable on its first run, ~0.1 s), so the probe runs its real subprocess and pipe
handling. The binary comes from EXEC_SANDBOX_QEMU_BIN_X86 / _ARM, the variables VMs read;
the host OS is pinned to Linux. Its behaviour comes from a mode file next to it. The bound
of the greeting wait has no setting: the tests that need it short or long patch the
private constant.
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from exec_sandbox.platform_utils import HostOS
from exec_sandbox.system_probes import probe_qemu_sandbox_support
from tests.conftest import skip_unless_linux

_GREETING = '{"QMP": {"version": {"qemu": {"micro": 3, "minor": 1, "major": 11}}, "capabilities": []}}'

_MODES = {
    "greets": f"echo '{_GREETING}'; exec sleep 3600",
    "no-seccomp": "echo \"qemu-system-x86_64: -sandbox on: There is no option group 'sandbox'\" >&2; exit 1",
    "other-error": 'echo "gtk initialization failed" >&2; exit 1',
    "exits-0": "exit 0",
    "killed": "kill -9 $$",
    "banner": "echo 'a wrapper banner'; exec sleep 3600",
    "silent": "exec sleep 3600",
    "long-line": "head -c 200000 /dev/zero | tr '\\0' x; echo; exec sleep 3600",
    "stderr-flood": f"head -c 300000 /dev/zero | tr '\\0' x >&2; echo '{_GREETING}'; exec sleep 3600",
}


@pytest.fixture(scope="module")
def fake_qemu(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("fake-qemu")
    script = folder / "qemu-system-fake"
    cases = "\n".join(f"  {mode}) {body} ;;" for mode, body in _MODES.items())
    script.write_text(
        "#!/bin/sh\n"
        'here=$(dirname "$0")\n'
        'echo run >> "$here/runs"\n'
        'echo $$ > "$here/pid"\n'
        'printf "%s\\n" "$@" > "$here/args"\n'
        f'case $(cat "$here/mode") in\n{cases}\nesac\n'
    )
    script.chmod(0o755)
    return script


async def _probe(binary: Path, mode: str | None = None, *, path: Path | None = None) -> bool:
    """probe_qemu_sandbox_support on Linux with `binary` as the QEMU of both arches; `path`
    replaces PATH (where the bare name is found when `binary` is not a file)."""
    if mode is not None:
        (binary.parent / "mode").write_text(mode)
    env = {"EXEC_SANDBOX_QEMU_BIN_X86": str(binary), "EXEC_SANDBOX_QEMU_BIN_ARM": str(binary)}
    if path is not None:
        env["PATH"] = str(path)
    with (
        patch.dict(os.environ, env),
        patch("exec_sandbox.system_probes.detect_host_os", return_value=HostOS.LINUX),
    ):
        return await probe_qemu_sandbox_support()


def _left_running(binary: Path) -> bool:
    try:
        os.kill(int((binary.parent / "pid").read_text()), 0)
    except ProcessLookupError:
        return False
    return True


async def test_probe_checks_the_filter_vms_install(fake_qemu: Path) -> None:
    """The probe starts QEMU with the -sandbox value of every VM, not a weaker one: on, with
    obsolete calls, privilege changes, process spawns and resource control denied."""
    assert await _probe(fake_qemu, "greets") is True

    args = (fake_qemu.parent / "args").read_text().splitlines()
    assert args[args.index("-sandbox") + 1] == (
        "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny"
    )


async def test_greeting_means_supported_without_waiting(fake_qemu: Path) -> None:
    """QEMU greeted with -sandbox on: supported at once, though QEMU would run for an hour and
    the bound is an hour too. The probe stops QEMU."""
    with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 3600):
        async with asyncio.timeout(30):
            assert await _probe(fake_qemu, "greets") is True
    assert not _left_running(fake_qemu)


@pytest.mark.parametrize("mode", ["no-seccomp", "other-error", "exits-0", "killed"])
async def test_end_before_greeting_means_not_supported(fake_qemu: Path, mode: str) -> None:
    """Only the greeting proves the filter: any end before it is no sandbox."""
    assert await _probe(fake_qemu, mode) is False


@pytest.mark.parametrize("mode", ["banner", "silent", "long-line"])
async def test_running_qemu_without_greeting_is_bounded(fake_qemu: Path, mode: str) -> None:
    """Still running at the bound, greeting or not: -sandbox was accepted (an unknown option or a
    failed filter exits at once). The probe stops QEMU, never waits for it."""
    with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 0.2):
        assert await _probe(fake_qemu, mode) is True
    assert not _left_running(fake_qemu)


async def test_stderr_never_blocks_the_greeting(fake_qemu: Path) -> None:
    """300 KB on stderr before the greeting: the probe drains it, so QEMU reaches its greeting."""
    with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 3600):
        async with asyncio.timeout(30):
            assert await _probe(fake_qemu, "stderr-flood") is True


async def test_binary_that_cannot_run_is_not_supported(tmp_path: Path) -> None:
    """No file at the setting and no qemu-system on PATH, or a file without the execute bit:
    not supported."""
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    not_executable = tmp_path / "qemu-system-plain-file"
    not_executable.write_text("#!/bin/sh\nexit 0\n")

    assert await _probe(tmp_path / "absent", path=empty_path) is False
    assert await _probe(not_executable) is False


async def test_macos_spawns_nothing() -> None:
    """macOS has no seccomp: not supported, and no QEMU starts."""
    with (
        patch("exec_sandbox.system_probes.detect_host_os", return_value=HostOS.MACOS),
        patch("exec_sandbox.system_probes.asyncio.create_subprocess_exec") as spawn,
    ):
        assert await probe_qemu_sandbox_support() is False
    spawn.assert_not_called()


async def test_result_is_cached(fake_qemu: Path) -> None:
    """The first answer stays: a second probe starts no QEMU."""
    runs = fake_qemu.parent / "runs"
    runs.unlink(missing_ok=True)

    assert await _probe(fake_qemu, "greets") is True
    assert await _probe(fake_qemu, "no-seccomp") is True  # the first answer stays
    assert runs.read_text().count("run") == 1


@skip_unless_linux
async def test_real_qemu_greets_with_sandbox_on() -> None:
    """The QEMU of this host starts with -sandbox on (CI builds it with seccomp)."""
    with patch("exec_sandbox.system_probes._SANDBOX_PROBE_TIMEOUT_SECONDS", 3600):
        async with asyncio.timeout(30):
            assert await probe_qemu_sandbox_support() is True

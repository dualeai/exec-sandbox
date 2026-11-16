"""Unit tests for the exec-sandbox CLI.

Tests CLI argument parsing, help output, and error handling.
Uses Click's CliRunner for isolated testing without running actual VMs.
"""

import asyncio
import json
import pwd
import re
import signal
import stat
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import click
import keyring.core
import pytest
from click.testing import CliRunner, Result
from keyring.backends import fail

from exec_sandbox import (
    ConfigError,
    ErrorAction,
    ErrorCode,
    ExecutionResult,
    InstanceGoneError,
    InstanceInfo,
    InstanceRef,
    InternalError,
    InvalidInputError,
    Language,
    OutcomeUnknownError,
    PortMapping,
    RecentCommit,
    Scheduler,
    SchedulerConfig,
    SessionClosedError,
    StartupTiming,
    TimingBreakdown,
    TransientError,
    VmSpec,
    WorkerFacts,
    __version__,
)
from exec_sandbox._imports import require_instance_admin
from exec_sandbox.assets import PrefetchResult
from exec_sandbox.cli import (
    MultiSourceResult,
    OneShotRun,
    _run_with_interrupt_handling,
    cli,
    detect_language,
    format_error,
    main,
    parse_download_spec,
    parse_env_vars,
    parse_port_mappings,
    parse_upload_spec,
    prefetch_command,
    run_command,
    truncate_source,
)
from exec_sandbox.exceptions import RouteKey, SandboxError, host_mismatch
from exec_sandbox.instance.keyring import Keyring
from exec_sandbox.instance.local_store import LocalInstanceStore
from exec_sandbox.instance.manager import InstanceAdmin, open_instance_admin
from exec_sandbox.instance.models import GuestFacts, HostFacts
from exec_sandbox.instance.store import body_key
from exec_sandbox.platform_utils import HostOS
from exec_sandbox.scheduler import settings_from_config
from exec_sandbox.session import Session
from tests.instance_support import FAKE_LAYOUT, new_keyring, seed_instance
from tests.keyring_support import KEYRING_ENTRY, Account, isolate_account

_SEEDED_SPEC = VmSpec(language=Language.JAVASCRIPT, memory_mb=512)
_SEEDED_HOST = HostFacts(
    arch="aarch64",
    accelerator="hvf",
    cpu_model="host",
    cpu_id="pfr0=0x1101000010110011",
    qemu_version="10.1.0",
    machine_type="virt-10.1",
    release=__version__,
    device_layout=FAKE_LAYOUT,
)
_SEEDED_GUEST = GuestFacts(
    base_image="0" * 64, packages_image=None, kernel="1" * 64, initramfs="2" * 64, counter_hz=24_000_000
)
"""An instance a test seals into a store itself (seed_instance)."""
_LOCAL_KEY = new_keyring()
"""Key in the fake OS keyring at the start of each test (see account)."""
_ENV_KEY = new_keyring()
"""Key a test puts in EXEC_SANDBOX_INSTANCE_KEYRING."""
_INSTANCE_ENV = ("EXEC_SANDBOX_INSTANCE_STORE", "EXEC_SANDBOX_INSTANCE_KEYRING", "EXEC_SANDBOX_INSTANCE_STORE_DIR")


# ============================================================================
# Test Fixtures
# ============================================================================
@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture(autouse=True)
def account(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Account:
    """No test reads the developer's instance settings, home or OS keyring: the instance
    variables are cleared, and the account is isolated (its OS keyring holds _LOCAL_KEY)."""
    for name in _INSTANCE_ENV:
        monkeypatch.delenv(name, raising=False)
    return isolate_account(monkeypatch, home=tmp_path / "account", key=_LOCAL_KEY)


def _on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host is Linux, for the folder layout."""
    monkeypatch.setattr("exec_sandbox.platform_utils.detect_host_os", lambda: HostOS.LINUX)


def _no_os_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """keyring finds no OS keyring on this host (its fail backend)."""
    monkeypatch.setattr(keyring.core, "_keyring_backend", fail.Keyring())


def _user_home(monkeypatch: pytest.MonkeyPatch, *, user: str, home: Path) -> None:
    """The user database has `user` with the home `home`; other names are not in it."""

    def getpwnam(name: str) -> pwd.struct_passwd:
        if name != user:
            raise KeyError(name)
        return pwd.struct_passwd((user, "*", 4242, 4242, "", str(home), "/bin/sh"))

    monkeypatch.setattr(pwd, "getpwnam", getpwnam)


@pytest.fixture
def mock_result() -> ExecutionResult:
    """Create a mock execution result for testing."""
    return ExecutionResult(
        stdout="hello\n",
        stderr="",
        exit_code=0,
        execution_time_ms=42,
        timing=TimingBreakdown(execute_ms=70, connect_ms=None),
    )


_STARTUP = StartupTiming(
    total_ms=320,
    admission_ms=0,
    setup_ms=10,
    boot_ms=300,
    warm_pool_hit=False,
    l1_cache_hit=False,
    l1_restore_ms=None,
    overlay_ms=0,
    qemu_cmd_build_ms=0,
    gvproxy_start_ms=0,
    qemu_fork_ms=0,
    guest_wait_ms=0,
    boot_retries=0,
    instance_load_ms=None,
)


def _mock_scheduler(scheduler_cls: MagicMock) -> AsyncMock:
    """A Scheduler mock whose new() and reuse() return the same VM mock (see _vm) on every call.

    Specced on the real classes: a CLI that calls a removed method or passes an
    unknown keyword to new() or exec() fails here, as it would in use. The VM is
    open and continues no saved instance (closed False, instance None)."""
    vm = create_autospec(Session, instance=True)
    vm.startup_timing = _STARTUP
    vm.exposed_ports = []
    vm.closed = False
    vm.instance = None
    vm.__aenter__.return_value = vm
    vm.__aexit__.return_value = None
    scheduler = create_autospec(Scheduler, instance=True)
    scheduler.new.return_value = vm
    scheduler.reuse.return_value = vm
    scheduler.__aenter__.return_value = scheduler
    scheduler.__aexit__.return_value = None
    scheduler_cls.return_value = scheduler
    return scheduler


def _vm(scheduler: AsyncMock) -> AsyncMock:
    return scheduler.new.return_value


def _calls(scheduler: AsyncMock) -> list[dict[str, Any]]:
    """Each new() + exec() pair as one dict: the VmSpec fields, the code and the exec options."""
    pairs = zip(scheduler.new.call_args_list, _vm(scheduler).exec.call_args_list, strict=True)
    return [{**new.args[0].model_dump(), "code": exec_.args[0], **exec_.kwargs} for new, exec_ in pairs]


def _codes(scheduler: AsyncMock) -> list[str]:
    """The code of every exec() call, sorted (multi-source runs are concurrent)."""
    return sorted(call.args[0] for call in _vm(scheduler).exec.call_args_list)


def _call(scheduler: AsyncMock) -> dict[str, Any]:
    (call,) = _calls(scheduler)
    return call


# ============================================================================
# Unit Tests - Helper Functions
# ============================================================================
class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_python_extension(self) -> None:
        """Detects Python from .py extension."""
        assert detect_language("script.py") == "python"
        assert detect_language("/path/to/script.py") == "python"

    def test_javascript_extensions(self) -> None:
        """Detects JavaScript from .js and .mjs extensions."""
        assert detect_language("app.js") == "javascript"
        assert detect_language("module.mjs") == "javascript"

    def test_shell_extension(self) -> None:
        """Detects raw/shell from .sh extension."""
        assert detect_language("run.sh") == "raw"

    def test_unknown_extension(self) -> None:
        """Returns None for unknown extensions."""
        assert detect_language("file.txt") is None
        assert detect_language("file.go") is None
        assert detect_language("file.rs") is None

    def test_no_extension(self) -> None:
        """Returns None for files without extension."""
        assert detect_language("Makefile") is None
        assert detect_language("README") is None

    def test_stdin_marker(self) -> None:
        """Returns None for stdin marker."""
        assert detect_language("-") is None


class TestParseEnvVars:
    """Tests for parse_env_vars function."""

    def test_single_var(self) -> None:
        """Parses single environment variable."""
        result = parse_env_vars(("KEY=value",))
        assert result == {"KEY": "value"}

    def test_multiple_vars(self) -> None:
        """Parses multiple environment variables."""
        result = parse_env_vars(("KEY1=value1", "KEY2=value2"))
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_value_with_equals(self) -> None:
        """Handles values containing equals sign."""
        result = parse_env_vars(("KEY=a=b=c",))
        assert result == {"KEY": "a=b=c"}

    def test_empty_value(self) -> None:
        """Handles empty values."""
        result = parse_env_vars(("KEY=",))
        assert result == {"KEY": ""}

    def test_empty_tuple(self) -> None:
        """Handles empty input."""
        result = parse_env_vars(())
        assert result == {}

    def test_invalid_format_no_equals(self) -> None:
        """Raises error for invalid format."""
        with pytest.raises(click.BadParameter, match="Invalid format"):
            parse_env_vars(("INVALID",))

    def test_invalid_format_empty_key(self) -> None:
        """Raises error for empty key."""
        with pytest.raises(click.BadParameter, match="Empty key"):
            parse_env_vars(("=value",))


class TestParsePortMappings:
    """Tests for parse_port_mappings function."""

    def test_forms(self) -> None:
        """INTERNAL, INTERNAL:EXTERNAL and a /tcp or /udp suffix."""
        assert parse_port_mappings(("8080", "8080:3000", "53:5353/udp")) == [
            PortMapping(internal=8080),
            PortMapping(internal=8080, external=3000),
            PortMapping(internal=53, external=5353, protocol="udp"),
        ]

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ("80/sctp", "Invalid protocol: 'sctp'"),
            ("1:2:3", "Invalid port format: '1:2:3'"),
            ("http", "Invalid port number in: 'http'. Ports must be integers."),
            ("80:", "Invalid port number in: '80:'. Ports must be integers."),
            ("0", "Invalid port mapping: '0'. 1 validation error for PortMapping"),
            ("80:80", "Invalid port mapping: '80:80'. 1 validation error for PortMapping"),
        ],
        ids=["protocol", "three-parts", "not-a-number", "empty-external", "port-zero", "external-below-1024"],
    )
    def test_invalid(self, spec: str, message: str) -> None:
        """Each invalid spec is a BadParameter on --expose: a number that is not an int, or a
        value PortMapping refuses (its pydantic message)."""
        with pytest.raises(click.BadParameter) as exc:
            parse_port_mappings((spec,))
        assert exc.value.message.startswith(message)
        assert exc.value.param_hint == "'--expose'"


class TestFormatError:
    """Tests for format_error function."""

    def test_basic_error(self) -> None:
        """The header in bold red, an empty line, then the explanation indented by two spaces."""
        result = format_error("Test error", "This is the explanation", suggestions=[])
        assert result == "\x1b[31m\x1b[1mError: Test error\x1b[0m\n\n  This is the explanation"

    def test_error_with_suggestions(self) -> None:
        """After the explanation: an empty line, "Suggestions:", then one bullet line per suggestion."""
        result = format_error(
            "Test error",
            "This is why",
            suggestions=["Try this", "Or try that"],
        )
        assert click.unstyle(result) == (
            "Error: Test error\n\n  This is why\n\n  Suggestions:\n    • Try this\n    • Or try that"
        )


# ============================================================================
# CLI Command Tests
# ============================================================================
class TestCliHelp:
    """Tests for CLI help output."""

    def test_short_help_flag(self, runner: CliRunner) -> None:
        """Shows help with -h flag."""
        result = runner.invoke(run_command, ["-h"])
        assert result.exit_code == 0
        assert "Run code in an isolated VM." in result.output


class TestCliArgumentParsing:
    """Tests for CLI argument parsing (without running actual code)."""

    def test_no_args_shows_error(self, runner: CliRunner) -> None:
        """Shows error when no arguments provided."""
        result = runner.invoke(run_command, [])
        assert result.exit_code == 2
        assert result.stderr.splitlines()[-1] == "Error: No code given. Give SOURCES, or use the -c option."

    def test_empty_code_shows_error(self, runner: CliRunner) -> None:
        """Shows error for empty code."""
        # Empty string argument passes through but results in empty code
        result = runner.invoke(run_command, [""])
        assert result.exit_code == 2
        assert result.stderr.splitlines()[-1] == "Error: No code in source: "

    def test_whitespace_only_code_shows_error(self, runner: CliRunner) -> None:
        """Shows error for whitespace-only code."""
        result = runner.invoke(run_command, ["   "])
        assert result.exit_code == 2
        assert result.stderr.splitlines()[-1] == "Error: No code in source:    "


class TestCliCodeExecution:
    """Tests for CLI code execution with mocked scheduler."""

    def test_inline_code(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes inline code."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["print('hello')"])

            assert result.exit_code == 0
            mock_scheduler.new.assert_called_once()
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["code"] == "print('hello')"
            assert call_kwargs["language"] == Language.PYTHON

    @pytest.mark.parametrize("code", ["x = 1\n" * 200, "print('a\x00b')"], ids=["longer-than-a-path", "nul-byte"])
    def test_inline_code_that_cannot_be_a_path_runs(
        self, runner: CliRunner, mock_result: ExecutionResult, code: str
    ) -> None:
        """Inline code longer than a file name, or with a NUL byte, runs as code: it is no
        path, and the file check does not raise."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, [code])

            assert result.exit_code == 0, result.output
            assert _call(mock_scheduler)["code"] == code

    def test_explicit_language(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Uses explicitly specified language."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-l", "javascript", "console.log('hi')"])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["language"] == Language.JAVASCRIPT

    def test_code_flag(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes code from -c flag."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-c", "print(1)"])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["code"] == "print(1)"

    def test_packages(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Passes packages to scheduler."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(
                run_command,
                ["-p", "requests", "-p", "pandas==2.2.0", "import requests"],
            )

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["packages"] == ("pandas==2.2.0", "requests")  # VmSpec sorts them

    def test_env_vars(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Passes environment variables to scheduler."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(
                run_command,
                ["-e", "KEY=value", "-e", "DEBUG=1", "print(1)"],
            )

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["env_vars"] == {"KEY": "value", "DEBUG": "1"}

    def test_timeout_and_memory(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """-t is the scheduler's default_timeout_seconds (the exec takes it), -m the VM memory."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-t", "60", "-m", "512", "print(1)"])

            assert result.exit_code == 0
            assert scheduler_cls.call_args.args[0].default_timeout_seconds == 60
            assert _call(mock_scheduler)["memory_mb"] == 512

    def test_network_options(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Passes network options to scheduler."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(
                run_command,
                [
                    "--network",
                    "--allow-domain",
                    "api.example.com",
                    "--allow-domain",
                    "cdn.example.com",
                    "print(1)",
                ],
            )

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["allow_network"] is True
            assert call_kwargs["allowed_domains"] == ("api.example.com", "cdn.example.com")

    @pytest.mark.parametrize(
        ("domain", "why"),
        [
            ("bad..domain", "Domain contains empty label: 'bad..domain'"),
            ("-lead.example.com", "Invalid domain label format: '-lead' in '-lead.example.com'"),
            ("x" * 300, "Domain too long: 300 > 253"),
        ],
        ids=["empty-label", "bad-label", "too-long"],
    )
    def test_invalid_domain_is_a_usage_error(self, runner: CliRunner, domain: str, why: str) -> None:
        """A domain that breaks the VmSpec rule exits 2 with the option named, before any VM starts."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            result = runner.invoke(run_command, ["--network", "--allow-domain", domain, "print(1)"])

        assert result.exit_code == 2
        assert result.stderr.splitlines()[-1] == f"Error: Invalid value for '--allow-domain': '{domain}': {why}"
        scheduler_cls.assert_not_called()

    def test_json_output(self, runner: CliRunner) -> None:
        """--json prints one object on stdout: the result as exec() gave it (every field, the
        cgroup memory peak too), the session's startup timing and exposed ports, no instance,
        and the total time that the CLI measured."""
        executed = ExecutionResult(
            stdout="hello\n",
            stderr="",
            exit_code=0,
            execution_time_ms=42,
            external_memory_peak_mb=128,
            timing=TimingBreakdown(execute_ms=70, connect_ms=None),
        )
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _vm(_mock_scheduler(scheduler_cls)).exec.return_value = executed

            result = runner.invoke(run_command, ["--json", "print(1)"])

        assert result.exit_code == 0
        printed = json.loads(result.stdout)
        assert isinstance(printed["total_ms"], int)
        assert printed == {
            "result": {  # written out: a field the JSON drops fails here
                "stdout": "hello\n",
                "stderr": "",
                "exit_code": 0,
                "execution_time_ms": 42,
                "external_cpu_time_ms": None,
                "external_memory_peak_mb": 128,
                "external_cpu_nr_throttled": None,
                "timing": {"execute_ms": 70, "connect_ms": None},
                "spawn_ms": None,
                "process_ms": None,
            },
            "startup": _STARTUP.model_dump(mode="json"),
            "exposed_ports": [],
            "instance": None,
            "total_ms": printed["total_ms"],
        }


class TestCliFileInput:
    """Tests for CLI file input."""

    def test_file_input(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Reads code from file."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('from file')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, [str(test_file)])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["code"] == "print('from file')"
            # Language should be auto-detected from .py extension
            assert call_kwargs["language"] == Language.PYTHON

    def test_file_language_detection(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Auto-detects language from file extension."""
        # Create a JS file
        test_file = tmp_path / "app.js"
        test_file.write_text("console.log('hello')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, [str(test_file)])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["language"] == Language.JAVASCRIPT


class TestCliStdinInput:
    """Tests for CLI stdin input."""

    def test_stdin_input(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Reads code from stdin."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-"], input="print('from stdin')")

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["code"] == "print('from stdin')"


class TestCliErrorHandling:
    """Tests for CLI error handling."""

    def test_package_not_allowed_error(self, runner: CliRunner) -> None:
        """PACKAGE_NOT_ALLOWED (FIX_INPUT): exit 2; the allow-list advice, then the action's."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            mock_scheduler.new.side_effect = InvalidInputError(
                "Package 'fake-package' not in python allow-list.",
                code=ErrorCode.PACKAGE_NOT_ALLOWED,
                context=None,
                hint=None,
                instance_id=None,
            )

            result = runner.invoke(run_command, ["import fake"])

        assert result.exit_code == 2
        assert result.stderr == (
            "Error: PACKAGE_NOT_ALLOWED (FIX_INPUT)\n"
            "\n"
            "  Package 'fake-package' not in python allow-list.\n"
            "\n"
            "  Suggestions:\n"
            "    • Only the top 10,000 packages of PyPI or npm are allowed.\n"
            "    • Check the spelling of the package name.\n"
            "    • To skip this check, use --no-validation.\n"
            "    • Correct the command, then run again.\n"
        )

    def test_timeout_error(self, runner: CliRunner) -> None:
        """VM_BOOT_TIMEOUT (RETRY): exit 124, like the timeout command; the boot advice, then the action's."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            mock_scheduler.new.side_effect = TransientError(
                "Guest agent not ready after 30s",
                code=ErrorCode.VM_BOOT_TIMEOUT,
                context=None,
                hint=None,
                instance_id=None,
            )

            result = runner.invoke(run_command, ["import time; time.sleep(100)"])

        assert result.exit_code == 124
        assert result.stderr == (
            "Error: VM_BOOT_TIMEOUT (RETRY)\n"
            "\n"
            "  Guest agent not ready after 30s\n"
            "\n"
            "  Suggestions:\n"
            "    • The VM did not start in time, so the code did not run. "
            "A boot timeout often comes from CPU contention and passes.\n"
            "    • Check the system load and the free resources.\n"
            "    • Run the same command again later.\n"
        )

    @pytest.mark.parametrize(
        ("error", "header", "exit_code", "advice"),
        [
            (
                ConfigError(
                    "Failed to start qemu-img: [Errno 2] No such file or directory",
                    code=ErrorCode.HOST_DEPENDENCY,
                    context=None,
                    hint=None,
                    instance_id=None,
                ),
                "Error: HOST_DEPENDENCY (FIX_CONFIG)",
                2,
                ["Install or fix the host dependency the message names", "Correct the settings, then run again."],
            ),
            (
                TransientError(
                    "No space left on device", code=ErrorCode.LOCAL_DISK, context=None, hint=None, instance_id=None
                ),
                "Error: LOCAL_DISK (RETRY)",
                125,
                ["Run the same command again later."],
            ),
            (
                ConfigError(
                    "Install with: pip install 'exec-sandbox[s3]'",
                    code=ErrorCode.EXTRA_MISSING,
                    context=None,
                    hint=None,
                    instance_id=None,
                ),
                "Error: EXTRA_MISSING (FIX_CONFIG)",
                2,
                ["Correct the settings, then run again."],
            ),
            (
                ConfigError(
                    "Release not found: v0.0.0",
                    code=ErrorCode.ASSET_NOT_FOUND,
                    context=None,
                    hint=None,
                    instance_id=None,
                ),
                "Error: ASSET_NOT_FOUND (FIX_CONFIG)",
                2,
                ["download the VM images with 'sbx prefetch'", "Correct the settings, then run again."],
            ),
            (
                TransientError(
                    "Failed to download https://example.invalid/vmlinuz after 3 attempts",
                    code=ErrorCode.ASSET_DOWNLOAD,
                    context=None,
                    hint=None,
                    instance_id=None,
                ),
                "Error: ASSET_DOWNLOAD (RETRY)",
                125,
                ["download the VM images with 'sbx prefetch'", "Run the same command again later."],
            ),
            (
                InvalidInputError(
                    "pip install failed: no matching distribution",
                    code=ErrorCode.PACKAGE_INSTALL_FAILED,
                    context=None,
                    hint=None,
                    instance_id=None,
                ),
                "Error: PACKAGE_INSTALL_FAILED (FIX_INPUT)",
                2,
                ["Correct the command, then run again."],
            ),
            (
                InternalError(
                    "Unexpected state transition", code=ErrorCode.INTERNAL, context=None, hint=None, instance_id=None
                ),
                "Error: INTERNAL (REPORT)",
                125,
                ["report it with the output of a run with EXEC_SANDBOX_LOG_LEVEL=DEBUG"],
            ),
            (
                SessionClosedError(
                    "Session is closed", code=ErrorCode.SESSION_CLOSED, context=None, hint=None, instance_id=None
                ),
                "Error: SESSION_CLOSED (RESTART)",
                125,
                ["Run the command again."],
            ),
        ],
        ids=[
            "missing-qemu-or-image",
            "transient",
            "missing-extra",
            "asset-not-found",
            "asset-download",
            "package-install-failed",
            "internal",
            "restart",
        ],
    )
    def test_sandbox_error(
        self, runner: CliRunner, error: SandboxError, header: str, exit_code: int, advice: list[str]
    ) -> None:
        """Header CODE (ACTION) and the message; exit 2 for FIX_INPUT and FIX_CONFIG, else 125;
        install advice only for a missing QEMU, kernel or image, prefetch advice for images
        that could not be found or fetched, then the advice of the action; no advice that
        does not apply."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            mock_scheduler.new.side_effect = error

            result = runner.invoke(run_command, ["print(1)"])

        assert result.exit_code == exit_code
        assert result.stderr.startswith(f"{header}\n\n  {error.message}\n")
        for line in advice:
            assert line in result.stderr
        assert result.stderr.count("•") == len(advice)  # one bullet per advice line: nothing else
        if error.code is not ErrorCode.HOST_DEPENDENCY:
            assert "QEMU" not in result.stderr


class TestCliNoValidation:
    """Tests for --no-validation flag."""

    def test_no_validation_flag(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Disables package validation with --no-validation."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["--no-validation", "print(1)"])

            assert result.exit_code == 0
            # Check that config was created with enable_package_validation=False
            config = scheduler_cls.call_args[0][0]
            assert config.enable_package_validation is False


# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================
class TestCliLanguageEdgeCases:
    """Tests for language option edge cases."""

    def test_language_case_insensitive(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Language option is case insensitive."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-l", "PYTHON", "print(1)"])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["language"] == Language.PYTHON

    def test_invalid_language(self, runner: CliRunner) -> None:
        """An unknown language is a usage error that names the value and the languages it accepts."""
        result = runner.invoke(run_command, ["-l", "ruby", "puts 'hello'"])
        assert result.exit_code == 2
        assert result.stderr.splitlines()[-1] == (
            "Error: Invalid value for '-l' / '--language': 'ruby' is not one of 'python', 'javascript', 'raw'."
        )

    def test_raw_language(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes with raw language."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-l", "raw", "echo hello"])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["language"] == Language.RAW


class TestCliFileEdgeCases:
    """Tests for file input edge cases."""

    def test_nonexistent_file_treated_as_code(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Non-existent file path is treated as inline code."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            # This looks like a file but doesn't exist, so treated as code
            result = runner.invoke(run_command, ["/nonexistent/script.py"])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            # The path string is treated as code
            assert call_kwargs["code"] == "/nonexistent/script.py"

    def test_directory_treated_as_code(self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path) -> None:
        """Directory path is treated as inline code (not a file)."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, [str(tmp_path)])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            # Directory path is treated as code string
            assert call_kwargs["code"] == str(tmp_path)

    def test_file_with_multiple_extensions(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """File with multiple extensions uses last extension."""
        test_file = tmp_path / "test.backup.py"
        test_file.write_text("print('multi ext')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, [str(test_file)])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["language"] == Language.PYTHON

    def test_hidden_file(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Hidden file is read correctly."""
        test_file = tmp_path / ".hidden.py"
        test_file.write_text("print('hidden')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, [str(test_file)])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["code"] == "print('hidden')"

    def test_file_with_spaces_in_path(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """File with spaces in path is handled correctly."""
        subdir = tmp_path / "my scripts"
        subdir.mkdir()
        test_file = subdir / "my script.py"
        test_file.write_text("print('spaces')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, [str(test_file)])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["code"] == "print('spaces')"


class TestCliCodeEdgeCases:
    """Tests for code content edge cases."""

    @pytest.mark.parametrize(
        "code",
        ["print(\"hello 'world'\")", "x = 1\nprint(x)", "print('Hello 世界 🌍')"],
        ids=["quotes", "newlines", "unicode"],
    )
    def test_code_is_passed_unchanged(self, runner: CliRunner, mock_result: ExecutionResult, code: str) -> None:
        """Quotes, new lines and non-ASCII characters in inline code get to exec() unchanged."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, [code])

            assert result.exit_code == 0
            assert _call(mock_scheduler)["code"] == code


class TestCliStdinEdgeCases:
    """Tests for stdin input edge cases."""

    def test_stdin_empty_input(self, runner: CliRunner) -> None:
        """Empty stdin shows error."""
        result = runner.invoke(run_command, ["-"], input="")
        assert result.exit_code == 2
        assert result.stderr.splitlines()[-1] == "Error: No code in source: <stdin>"

    def test_stdin_whitespace_only(self, runner: CliRunner) -> None:
        """Whitespace-only stdin shows error."""
        result = runner.invoke(run_command, ["-"], input="   \n\t\n   ")
        assert result.exit_code == 2
        assert result.stderr.splitlines()[-1] == "Error: No code in source: <stdin>"

    def test_stdin_with_explicit_language(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Stdin with explicit language works."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-l", "javascript", "-"], input="console.log('stdin')")

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["language"] == Language.JAVASCRIPT


class TestCliEnvVarEdgeCases:
    """Tests for environment variable edge cases."""

    def test_env_var_special_characters_in_value(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Env var with special characters in value."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-e", "KEY=val!@#$%^&*()", "print(1)"])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["env_vars"]["KEY"] == "val!@#$%^&*()"

    def test_env_var_with_quotes_in_value(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Env var with quotes in value."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-e", 'KEY="quoted value"', "print(1)"])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["env_vars"]["KEY"] == '"quoted value"'

    def test_env_var_numeric_key(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Env var with numeric key."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-e", "123=value", "print(1)"])

            assert result.exit_code == 0
            call_kwargs = _call(mock_scheduler)
            assert call_kwargs["env_vars"]["123"] == "value"


class TestCliTimeoutMemoryEdgeCases:
    """Tests for timeout and memory edge cases."""

    @pytest.mark.parametrize(("flag", "name"), [("-t", "--timeout"), ("-m", "--memory")])
    def test_value_that_is_not_an_integer(self, runner: CliRunner, flag: str, name: str) -> None:
        """A timeout or a memory size that is not an integer is a usage error that names the option."""
        result = runner.invoke(run_command, [flag, "abc", "print(1)"])
        assert result.exit_code == 2
        assert result.stderr.splitlines()[-1] == (
            f"Error: Invalid value for '{flag}' / '{name}': 'abc' is not a valid integer."
        )


class TestCliExitCodePassthrough:
    """Tests for exit code passthrough from executed code."""

    @pytest.mark.parametrize("json_output", [False, True], ids=["text", "json"])
    def test_nonzero_exit_code(self, runner: CliRunner, json_output: bool) -> None:
        """The code's own exit code is the CLI's, with --json too."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _vm(_mock_scheduler(scheduler_cls)).exec.return_value = _result(42)

            result = runner.invoke(run_command, [*(["--json"] if json_output else []), "exit(42)"])

        assert result.exit_code == 42

    @pytest.mark.parametrize("json_output", [False, True], ids=["text", "json"])
    def test_code_timeout_exits_124(self, runner: CliRunner, json_output: bool) -> None:
        """A code timeout (result exit code -1) exits 124, like the timeout command (not 255);
        the JSON keeps the result's own -1."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _vm(_mock_scheduler(scheduler_cls)).exec.return_value = _result(-1)

            result = runner.invoke(run_command, [*(["--json"] if json_output else []), "while True: pass"])

        assert result.exit_code == 124
        if json_output:
            assert json.loads(result.stdout)["result"]["exit_code"] == -1

    def test_code_exit_255_is_not_a_timeout(self, runner: CliRunner) -> None:
        """Code that exits 255 (sys.exit(-1) in the guest) exits 255, not 124."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _vm(_mock_scheduler(scheduler_cls)).exec.return_value = _result(255)

            result = runner.invoke(run_command, ["import sys; sys.exit(-1)"])

        assert result.exit_code == 255


class TestCliQuietMode:
    """Tests for quiet mode."""

    @pytest.mark.parametrize("quiet", [False, True], ids=["tty", "tty-quiet"])
    def test_tty_footer(self, runner: CliRunner, mock_result: ExecutionResult, quiet: bool) -> None:
        """On a terminal, a run ends with a new line on stdout and "✓ Done in <ms>ms" on stderr; -q hides both."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls, patch("exec_sandbox.cli.is_tty", return_value=True):
            _vm(_mock_scheduler(scheduler_cls)).exec.return_value = mock_result

            result = runner.invoke(run_command, [*(["-q"] if quiet else []), "print(1)"])

        assert result.exit_code == 0
        if quiet:
            assert (result.stdout, result.stderr) == ("", "")
        else:
            assert result.stdout == "\n"
            assert re.fullmatch(r"✓ Done in \d+ms\n", result.stderr), result.stderr


class TestDetectLanguageEdgeCases:
    """Additional edge cases for detect_language."""

    def test_extension_case_is_ignored(self) -> None:
        """Upper case and mixed case extensions give the same language as lower case ones."""
        assert detect_language("SCRIPT.PY") == "python"
        assert detect_language("APP.JS") == "javascript"
        assert detect_language("script.Py") == "python"
        assert detect_language("app.Js") == "javascript"

    def test_double_extension(self) -> None:
        """Uses last extension for double extensions."""
        assert detect_language("file.tar.py") == "python"
        assert detect_language("file.py.bak") is None  # .bak is unknown

    def test_dotfiles_have_no_extension(self) -> None:
        """Dotfiles (like .py, .js) are hidden files, not extension-only files."""
        # .py is a hidden file named ".py", not a file with extension .py
        assert detect_language(".py") is None
        assert detect_language(".js") is None
        # But .hidden.py IS a hidden file with .py extension
        assert detect_language(".hidden.py") == "python"

    def test_path_with_dots_in_directory(self) -> None:
        """Handles paths with dots in directory names."""
        assert detect_language("/path.to.dir/script.py") == "python"
        assert detect_language("./my.project/app.js") == "javascript"


class TestParseEnvVarsEdgeCases:
    """Additional edge cases for parse_env_vars."""

    def test_unicode_in_key_and_value(self) -> None:
        """Handles unicode in keys and values."""
        result = parse_env_vars(("キー=値",))
        assert result == {"キー": "値"}

    def test_very_long_value(self) -> None:
        """Handles very long values."""
        long_value = "x" * 10000
        result = parse_env_vars((f"KEY={long_value}",))
        assert result == {"KEY": long_value}

    def test_value_with_newlines(self) -> None:
        """Handles values with newlines."""
        result = parse_env_vars(("KEY=line1\nline2",))
        assert result == {"KEY": "line1\nline2"}

    def test_duplicate_keys(self) -> None:
        """Later duplicate key overwrites earlier."""
        result = parse_env_vars(("KEY=first", "KEY=second"))
        assert result == {"KEY": "second"}


# ============================================================================
# Multi-Input Tests (V2)
# ============================================================================
class TestParseUploadSpec:
    """Tests for parse_upload_spec function."""

    # --- Normal cases ---

    def test_basic_format(self, tmp_path: Path) -> None:
        """Parses LOCAL:GUEST format."""
        local_file = tmp_path / "data.csv"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:data.csv")
        assert local_path == str(local_file)
        assert guest_path == "data.csv"

    def test_guest_with_subdirectory(self, tmp_path: Path) -> None:
        """Parses guest path with subdirectory."""
        local_file = tmp_path / "input.txt"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:subdir/input.txt")
        assert local_path == str(local_file)
        assert guest_path == "subdir/input.txt"

    def test_relative_local_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parses relative local path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rel.csv").write_text("content")
        local_path, guest_path = parse_upload_spec("rel.csv:data.csv")
        assert local_path == "rel.csv"
        assert guest_path == "data.csv"

    def test_symlink_local_path(self, tmp_path: Path) -> None:
        """Follows symlink for local path (exists check passes)."""
        real = tmp_path / "real.txt"
        real.write_text("content")
        link = tmp_path / "link.txt"
        link.symlink_to(real)
        local_path, guest_path = parse_upload_spec(f"{link}:data.txt")
        assert local_path == str(link)
        assert guest_path == "data.txt"

    # --- Edge cases ---

    def test_multiple_colons_splits_on_first(self, tmp_path: Path) -> None:
        """Splits on first colon — guest path may contain colons."""
        local_file = tmp_path / "f.txt"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:a:b:c")
        assert local_path == str(local_file)
        assert guest_path == "a:b:c"

    def test_spaces_in_local_path(self, tmp_path: Path) -> None:
        """Handles spaces in local path."""
        local_file = tmp_path / "my file.csv"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:data.csv")
        assert local_path == str(local_file)
        assert guest_path == "data.csv"

    def test_spaces_in_guest_path(self, tmp_path: Path) -> None:
        """Handles spaces in guest path."""
        local_file = tmp_path / "f.txt"
        local_file.write_text("content")
        _, guest_path = parse_upload_spec(f"{local_file}:my data.csv")
        assert guest_path == "my data.csv"

    def test_unicode_in_paths(self, tmp_path: Path) -> None:
        """Handles unicode characters in paths."""
        local_file = tmp_path / "données.csv"
        local_file.write_text("content")
        local_path, guest_path = parse_upload_spec(f"{local_file}:数据.csv")
        assert local_path == str(local_file)
        assert guest_path == "数据.csv"

    def test_dotfile_local(self, tmp_path: Path) -> None:
        """Handles dotfile as local path."""
        local_file = tmp_path / ".hidden"
        local_file.write_text("secret")
        local_path, _ = parse_upload_spec(f"{local_file}:config")
        assert local_path == str(local_file)

    # --- Tilde expansion ---

    def test_tilde_expanded_in_local_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """~ in the local path is the HOME folder: the file there is found."""
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "data.csv").write_text("content")
        local_path, guest_path = parse_upload_spec("~/data.csv:data.csv")
        assert local_path == str(tmp_path / "data.csv")
        assert guest_path == "data.csv"

    def test_tilde_prefix_user_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """~user in the local path is the home of that user in the user database."""
        _user_home(monkeypatch, user="someone", home=tmp_path)
        (tmp_path / "data.csv").write_text("content")
        local_path, _ = parse_upload_spec("~someone/data.csv:data.csv")
        assert local_path == str(tmp_path / "data.csv")

    # --- Error cases ---

    def test_missing_colon(self) -> None:
        """Rejects spec without colon."""
        with pytest.raises(click.BadParameter, match="Invalid upload format"):
            parse_upload_spec("data.csv")

    def test_empty_guest_path(self, tmp_path: Path) -> None:
        """Rejects empty guest path (trailing colon)."""
        local_file = tmp_path / "data.csv"
        local_file.write_text("content")
        with pytest.raises(click.BadParameter, match="Invalid upload format"):
            parse_upload_spec(f"{local_file}:")

    def test_empty_local_path(self) -> None:
        """Rejects empty local path (leading colon)."""
        with pytest.raises(click.BadParameter, match="Empty local path"):
            parse_upload_spec(":data.csv")

    def test_nonexistent_local_file(self) -> None:
        """Rejects non-existent local file."""
        with pytest.raises(click.BadParameter, match="Local file not found"):
            parse_upload_spec("/nonexistent/file.csv:data.csv")

    def test_just_colon(self) -> None:
        """A bare colon has no local path and no guest path: the format error."""
        with pytest.raises(click.BadParameter, match=re.escape("Invalid upload format: ':'. Use LOCAL:GUEST")):
            parse_upload_spec(":")

    def test_double_colon(self) -> None:
        """Rejects double colon (empty local)."""
        with pytest.raises(click.BadParameter, match="Empty local path"):
            parse_upload_spec("::")

    def test_local_is_directory(self, tmp_path: Path) -> None:
        """Accepts directory as local path (exists returns True for dirs).

        The session.write_file will fail later when trying to read content,
        but parse_upload_spec only checks existence.
        """
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        # Does not raise — existence check passes for directories
        local_path, _ = parse_upload_spec(f"{subdir}:data")
        assert local_path == str(subdir)

    def test_broken_symlink(self, tmp_path: Path) -> None:
        """Rejects broken symlink (target doesn't exist)."""
        link = tmp_path / "broken"
        link.symlink_to(tmp_path / "nonexistent_target")
        with pytest.raises(click.BadParameter, match="Local file not found"):
            parse_upload_spec(f"{link}:data.txt")


class TestParseDownloadSpec:
    """Tests for parse_download_spec function."""

    # --- Normal cases ---

    def test_full_format(self) -> None:
        """Parses GUEST:LOCAL format."""
        guest_path, local_path = parse_download_spec("output.csv:./results/out.csv")
        assert guest_path == "output.csv"
        assert local_path == "./results/out.csv"

    def test_shorthand_guest_only(self) -> None:
        """Parses GUEST shorthand (no colon) -> local = basename."""
        guest_path, local_path = parse_download_spec("output.csv")
        assert guest_path == "output.csv"
        assert local_path == "output.csv"

    def test_shorthand_guest_with_subdir(self) -> None:
        """Parses GUEST with subdir -> local = basename only."""
        guest_path, local_path = parse_download_spec("subdir/output.csv")
        assert guest_path == "subdir/output.csv"
        assert local_path == "output.csv"

    def test_local_absolute_path(self) -> None:
        """Parses absolute local path."""
        guest_path, local_path = parse_download_spec("data.csv:/tmp/out.csv")
        assert guest_path == "data.csv"
        assert local_path == "/tmp/out.csv"

    def test_local_with_nested_dirs(self) -> None:
        """Parses local path with nested directories."""
        guest_path, local_path = parse_download_spec("result.bin:./deep/nested/dir/out.bin")
        assert guest_path == "result.bin"
        assert local_path == "./deep/nested/dir/out.bin"

    # --- Edge cases ---

    def test_multiple_colons_splits_on_first(self) -> None:
        """Splits on first colon — local path may contain colons."""
        guest_path, local_path = parse_download_spec("data.csv:a:b:c")
        assert guest_path == "data.csv"
        assert local_path == "a:b:c"

    def test_spaces_in_guest_path(self) -> None:
        """Handles spaces in guest path."""
        guest_path, local_path = parse_download_spec("my file.csv:out.csv")
        assert guest_path == "my file.csv"
        assert local_path == "out.csv"

    def test_spaces_in_local_path(self) -> None:
        """Handles spaces in local path."""
        guest_path, local_path = parse_download_spec("data.csv:./my output.csv")
        assert guest_path == "data.csv"
        assert local_path == "./my output.csv"

    def test_unicode_in_paths(self) -> None:
        """Handles unicode characters in paths."""
        guest_path, local_path = parse_download_spec("数据.csv:./données.csv")
        assert guest_path == "数据.csv"
        assert local_path == "./données.csv"

    def test_shorthand_deeply_nested_guest(self) -> None:
        """Shorthand with deep nesting extracts basename."""
        guest_path, local_path = parse_download_spec("a/b/c/d/result.txt")
        assert guest_path == "a/b/c/d/result.txt"
        assert local_path == "result.txt"

    def test_shorthand_dotfile(self) -> None:
        """Shorthand preserves dotfile basename."""
        guest_path, local_path = parse_download_spec(".hidden")
        assert guest_path == ".hidden"
        assert local_path == ".hidden"

    def test_guest_with_trailing_slash_shorthand(self) -> None:
        """Guest path with trailing slash — Path strips it, basename is 'subdir'."""
        guest_path, local_path = parse_download_spec("subdir/")
        assert guest_path == "subdir/"
        assert local_path == "subdir"

    # --- Tilde expansion ---

    def test_tilde_expanded_in_local_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """~ in the local path is the HOME folder."""
        monkeypatch.setenv("HOME", str(tmp_path))
        guest_path, local_path = parse_download_spec("output.csv:~/result.csv")
        assert guest_path == "output.csv"
        assert local_path == str(tmp_path / "result.csv")

    def test_tilde_prefix_user_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """~user in the local path is the home of that user in the user database; an unknown
        user stays as written."""
        _user_home(monkeypatch, user="someone", home=tmp_path)
        _, local_path = parse_download_spec("output.csv:~someone/result.csv")
        assert local_path == str(tmp_path / "result.csv")
        _, local_path = parse_download_spec("output.csv:~nobody-here/result.csv")
        assert local_path == "~nobody-here/result.csv"

    # --- Boundary cases ---

    def test_root_path_shorthand(self) -> None:
        """Guest path '/' — basename is empty, rejected."""
        with pytest.raises(click.BadParameter, match="Empty local path"):
            parse_download_spec("/")

    def test_dot_shorthand(self) -> None:
        """Guest path '.' — Path('.').name is empty string, caught by validation."""
        with pytest.raises(click.BadParameter, match="Empty local path"):
            parse_download_spec(".")

    # --- Error cases ---

    def test_empty_spec(self) -> None:
        """Rejects empty spec."""
        with pytest.raises(click.BadParameter, match="Empty download path"):
            parse_download_spec("")

    def test_empty_guest_path(self) -> None:
        """Rejects empty guest path (leading colon)."""
        with pytest.raises(click.BadParameter, match="Empty guest path"):
            parse_download_spec(":./out.csv")

    def test_empty_local_path(self) -> None:
        """Rejects empty local path (trailing colon)."""
        with pytest.raises(click.BadParameter, match="Empty local path"):
            parse_download_spec("output.csv:")

    def test_just_colon(self) -> None:
        """Rejects bare colon (both sides empty)."""
        with pytest.raises(click.BadParameter, match="Empty guest path"):
            parse_download_spec(":")

    def test_double_colon(self) -> None:
        """Rejects double colon (empty guest)."""
        with pytest.raises(click.BadParameter, match="Empty guest path"):
            parse_download_spec("::")

    def test_dotdot_shorthand(self) -> None:
        """Guest path '..' — basename is '..', rejected because '..' is a directory."""
        with pytest.raises(click.BadParameter, match="folder"):
            parse_download_spec("..")

    def test_tilde_alone_as_local_rejected(self) -> None:
        """Bare ~ as local path (full format) is rejected as directory."""
        with pytest.raises(click.BadParameter, match="folder"):
            parse_download_spec("output.csv:~")

    def test_tilde_alone_shorthand_rejected(self) -> None:
        """Shorthand ~ is rejected as directory."""
        with pytest.raises(click.BadParameter, match="folder"):
            parse_download_spec("~")

    def test_existing_directory_as_local_rejected(self, tmp_path: Path) -> None:
        """Existing directory as local path is rejected."""
        with pytest.raises(click.BadParameter, match="folder"):
            parse_download_spec(f"output.csv:{tmp_path}")


class TestCliFileIoIntegration:
    """Tests for --upload/--download argument flow through CLI to run_code."""

    def test_upload_parsed_and_passed(self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path) -> None:
        """Upload spec is parsed and tuple flows to session.write_file."""
        local_file = tmp_path / "data.csv"
        local_file.write_text("content")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_session = _vm(_mock_scheduler(scheduler_cls))
            mock_session.exec.return_value = mock_result

            result = runner.invoke(
                run_command,
                ["--upload", f"{local_file}:input.csv", "-c", "print(1)"],
            )

            assert result.exit_code == 0
            mock_session.write_file.assert_awaited_once_with("input.csv", local_file)

    def test_download_full_parsed_and_passed(
        self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Download GUEST:LOCAL spec flows correctly to session.read_file."""
        monkeypatch.chdir(tmp_path)
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_session = _vm(_mock_scheduler(scheduler_cls))
            mock_session.exec.return_value = mock_result

            async def _fake_read(path: str, *, destination: Path) -> None:
                destination.write_bytes(b"downloaded")

            mock_session.read_file.side_effect = _fake_read

            result = runner.invoke(run_command, ["--download", "out.csv:./result.csv", "-c", "print(1)"])

        assert result.exit_code == 0
        mock_session.read_file.assert_awaited_once_with("out.csv", destination=Path("result.csv"))
        assert (tmp_path / "result.csv").read_bytes() == b"downloaded"

    def test_download_destination_the_caller_can_fix_exits_2(
        self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path
    ) -> None:
        """read_file refuses a --download destination the caller can fix (ARGUMENT_INVALID,
        FIX_INPUT): exit 2, never 125 (RETRY), with the message of the error. Which destinations
        read_file refuses: tests/test_session_guard_and_io.py
        (test_destination_the_caller_can_fix_is_argument_invalid)."""
        destination = tmp_path / "file" / "sub" / "out.csv"
        message = "VM vm-1 read_file could not write the local destination (a file is in its path)"

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _vm(_mock_scheduler(scheduler_cls))
            vm.exec.return_value = mock_result
            vm.read_file.side_effect = InvalidInputError(
                message, code=ErrorCode.ARGUMENT_INVALID, context=None, hint=None, instance_id=None
            )

            result = runner.invoke(run_command, ["--download", f"out.csv:{destination}", "-c", "print(1)"])

        assert result.exit_code == 2
        assert "Error: ARGUMENT_INVALID (FIX_INPUT)" in result.stderr
        assert message in result.stderr
        vm.read_file.assert_awaited_once_with("out.csv", destination=destination)

    def test_download_shorthand_writes_to_basename(
        self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Download GUEST shorthand writes to ./basename in cwd."""
        monkeypatch.chdir(tmp_path)
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_session = _vm(_mock_scheduler(scheduler_cls))
            mock_session.exec.return_value = mock_result

            async def _fake_read(path: str, *, destination: Path) -> None:
                destination.write_bytes(b"data")

            mock_session.read_file.side_effect = _fake_read

            result = runner.invoke(run_command, ["--download", "subdir/output.csv", "-c", "print(1)"])

        assert result.exit_code == 0
        mock_session.read_file.assert_awaited_once_with("subdir/output.csv", destination=Path("output.csv"))
        assert (tmp_path / "output.csv").read_bytes() == b"data"

    @pytest.mark.parametrize("quiet", [False, True], ids=["line", "quiet"])
    def test_download_line_on_stderr(
        self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path, quiet: bool
    ) -> None:
        """Each download prints one line on stderr with its size; -q downloads the same file and prints nothing."""
        local = tmp_path / "result.csv"
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _vm(_mock_scheduler(scheduler_cls))
            vm.exec.return_value = mock_result

            async def fake_read(path: str, *, destination: Path) -> None:
                destination.write_bytes(b"downloaded")

            vm.read_file.side_effect = fake_read
            result = runner.invoke(
                run_command, [*(["-q"] if quiet else []), "--download", f"out.csv:{local}", "-c", "print(1)"]
            )

        assert result.exit_code == 0
        assert local.read_bytes() == b"downloaded"
        assert result.stderr == ("" if quiet else f"  Downloaded: out.csv -> {local} (10 bytes)\n")

    def test_upload_and_download_together(
        self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Upload and download in one command, on one session: the upload goes in before the code
        runs, the download comes out after it."""
        monkeypatch.chdir(tmp_path)
        local_file = tmp_path / "input.txt"
        local_file.write_text("hello")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_session = _vm(_mock_scheduler(scheduler_cls))
            mock_session.exec.return_value = mock_result

            async def _fake_read(path: str, *, destination: Path) -> None:
                destination.write_bytes(b"result")

            mock_session.read_file.side_effect = _fake_read

            result = runner.invoke(
                run_command,
                ["--upload", f"{local_file}:input.txt", "--download", "output.txt:result.txt", "-c", "print(1)"],
            )

        assert result.exit_code == 0
        mock_session.write_file.assert_awaited_once_with("input.txt", local_file)
        mock_session.read_file.assert_awaited_once_with("output.txt", destination=Path("result.txt"))
        steps = [name for name, _, _ in mock_session.mock_calls if name in {"write_file", "exec", "read_file"}]
        assert steps == ["write_file", "exec", "read_file"]
        assert (tmp_path / "result.txt").read_bytes() == b"result"

    def test_bad_upload_spec_shows_usage_error(self, runner: CliRunner) -> None:
        """Invalid upload spec surfaces as CLI usage error."""
        result = runner.invoke(run_command, ["--upload", "nocolon", "-c", "print(1)"])
        assert result.exit_code == 2
        assert "Invalid upload format" in result.output

    def test_bad_download_spec_shows_usage_error(self, runner: CliRunner) -> None:
        """Invalid download spec surfaces as CLI usage error."""
        result = runner.invoke(run_command, ["--download", ":", "-c", "print(1)"])
        assert result.exit_code == 2
        assert "Empty guest path" in result.output


class TestTruncateSource:
    """Tests for truncate_source helper."""

    def test_short_source(self) -> None:
        """Short sources are not truncated."""
        assert truncate_source("print(1)") == "print(1)"
        assert truncate_source("short") == "short"

    def test_long_source(self) -> None:
        """A source longer than 40 characters keeps its first 37, then "..."."""
        assert truncate_source("a" * 37 + "bcdef") == "a" * 37 + "..."

    def test_exact_length(self) -> None:
        """Exact length sources are not truncated."""
        src = "x" * 40
        assert truncate_source(src) == src


class TestMultiSourceResult:
    """Tests for MultiSourceResult dataclass."""

    @pytest.mark.parametrize(
        ("code", "action", "exit_code"),
        [
            (ErrorCode.CODE_INVALID, ErrorAction.FIX_INPUT, 2),
            (ErrorCode.HOST_DEPENDENCY, ErrorAction.FIX_CONFIG, 2),
            (ErrorCode.VM_BOOT_TIMEOUT, ErrorAction.RETRY, 124),
            (ErrorCode.QEMU_CRASHED, ErrorAction.RETRY, 125),
            (ErrorCode.GUEST_OUTCOME_UNKNOWN, ErrorAction.VERIFY, 125),
        ],
        ids=["fix-input", "fix-config", "boot-timeout", "retry", "verify"],
    )
    def test_error_result_exit_code(self, code: ErrorCode, action: ErrorAction, exit_code: int) -> None:
        """The exit code of a source with no run is the one a single source gets for that error."""
        result = MultiSourceResult(index=1, source="bad.py", run=None, error="It failed.", code=code, action=action)
        assert result.exit_code == exit_code


class TestCliMultiInput:
    """Tests for CLI multi-input functionality."""

    def test_multiple_inline_codes(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes multiple inline codes."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["print(1)", "print(2)"])

            assert result.exit_code == 0
            assert _codes(mock_scheduler) == ["print(1)", "print(2)"]

    def test_multiple_c_flags(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes multiple -c flag codes."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-c", "print(1)", "-c", "print(2)"])

            assert result.exit_code == 0
            assert _codes(mock_scheduler) == ["print(1)", "print(2)"]

    def test_mixed_c_and_positional(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """Executes mixed -c flags and positional sources."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-c", "print(1)", "print(2)"])

            assert result.exit_code == 0
            assert _codes(mock_scheduler) == ["print(1)", "print(2)"]

    def test_multiple_files(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Executes multiple file sources."""
        # Create test files
        file1 = tmp_path / "test1.py"
        file1.write_text("print('file1')")
        file2 = tmp_path / "test2.py"
        file2.write_text("print('file2')")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, [str(file1), str(file2)])

            assert result.exit_code == 0
            assert _codes(mock_scheduler) == ["print('file1')", "print('file2')"]

    def test_json_output_multi(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """--json with several sources: one array, an entry per source in the order given, each
        with its index, its source and its run."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["--json", "print(1)", "print(2)"])

        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert [(item["index"], item["source"], item["run"]["result"]["stdout"]) for item in parsed] == [
            (0, "print(1)", "hello\n"),
            (1, "print(2)", "hello\n"),
        ]

    def test_output_without_a_final_new_line_gets_one(self, runner: CliRunner) -> None:
        """Each source's block: a new line, its header, its stdout; its stderr on stderr. Output
        that does not end with a new line gets one, so the next block starts on its own line.
        Blocks come in the order the sources end (here, either order)."""
        results = {
            "x = 1": ExecutionResult(
                stdout="1", stderr="w1", exit_code=0, timing=TimingBreakdown(execute_ms=5, connect_ms=None)
            ),
            "y = 2": ExecutionResult(
                stdout="2\n", stderr="w2\n", exit_code=0, timing=TimingBreakdown(execute_ms=5, connect_ms=None)
            ),
        }
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _vm(_mock_scheduler(scheduler_cls)).exec.side_effect = lambda code, **_: results[code]

            result = runner.invoke(run_command, ["-q", "x = 1", "y = 2"])

        assert result.exit_code == 0
        first = f"\n━━━ [1/2] x = 1 {'━' * 45}\n1\n"
        second = f"\n━━━ [2/2] y = 2 {'━' * 45}\n2\n"
        assert (result.stdout, result.stderr) in {(first + second, "w1\nw2\n"), (second + first, "w2\nw1\n")}

    @pytest.mark.parametrize(
        ("exit_codes", "quiet", "summary"),
        [
            pytest.param((0, 0), False, r"✓ 2/2 passed \(\d+ms total\)", id="passed"),
            pytest.param((0, 3), False, r"⚠ 1/2 passed, 1 failed \(\d+ms total\)", id="failed"),
            pytest.param((0, 3), True, None, id="quiet"),
        ],
    )
    def test_tty_summary(
        self, runner: CliRunner, exit_codes: tuple[int, int], quiet: bool, summary: str | None
    ) -> None:
        """On a terminal, a run of several sources ends with a rule and the count of sources
        that passed and failed (exit code 0 or not) on stdout; -q hides it."""
        results = {
            f"{name} = {code}": ExecutionResult(
                stdout="", stderr="", exit_code=code, timing=TimingBreakdown(execute_ms=5, connect_ms=None)
            )
            for name, code in zip("ab", exit_codes, strict=True)
        }
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls, patch("exec_sandbox.cli.is_tty", return_value=True):
            _vm(_mock_scheduler(scheduler_cls)).exec.side_effect = lambda code, **_: results[code]

            result = runner.invoke(run_command, [*(["-q"] if quiet else []), *results])

        assert result.exit_code == max(exit_codes)
        if summary is None:
            assert "─" not in result.stdout
        else:
            assert re.search(rf"\n\n{'─' * 60}\n{summary}\n\Z", result.stdout), result.stdout

    def test_stdin_only_once(self, runner: CliRunner) -> None:
        """Stdin marker can only be used once."""
        result = runner.invoke(run_command, ["-", "-"], input="print(1)")
        assert result.exit_code == 2
        assert result.stderr.splitlines()[-1] == "Error: Stdin marker '-' can only be used once."

    def test_single_source_streams_its_output(self, runner: CliRunner) -> None:
        """One source: the chunks exec() gives to on_stdout and on_stderr go to stdout and stderr
        as they come (in that order on the terminal), and the output in the result is not
        printed again."""

        async def exec_(code: str, *, on_stdout: Any, on_stderr: Any, **_: Any) -> ExecutionResult:
            on_stdout("out-1 ")
            on_stderr("err-1 ")
            on_stdout("out-2")
            return ExecutionResult(
                stdout="out-1 out-2",
                stderr="err-1 ",
                exit_code=0,
                timing=TimingBreakdown(execute_ms=5, connect_ms=None),
            )

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _vm(_mock_scheduler(scheduler_cls)).exec.side_effect = exec_

            result = runner.invoke(run_command, ["-q", "print('hello')"])

        assert result.exit_code == 0
        assert (result.stdout, result.stderr) == ("out-1 out-2", "err-1 ")
        assert result.output == "out-1 err-1 out-2"  # stdout and stderr as the terminal shows them


class TestCliMultiInputLanguageDetection:
    """Tests for language auto-detection with multiple sources."""

    def test_mixed_language_auto_detect(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Each file gets the language of its own extension: the VM of the .py file runs python,
        the VM of the .js file runs javascript."""
        py_file = tmp_path / "test.py"
        py_file.write_text("print(1)")
        js_file = tmp_path / "test.js"
        js_file.write_text("console.log(1)")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, [str(py_file), str(js_file)])

        assert result.exit_code == 0
        languages = {call["code"]: call["language"] for call in _calls(mock_scheduler)}
        assert languages == {"print(1)": Language.PYTHON, "console.log(1)": Language.JAVASCRIPT}

    def test_global_language_override(
        self,
        runner: CliRunner,
        mock_result: ExecutionResult,
        tmp_path: Path,
    ) -> None:
        """Global -l flag overrides auto-detection for all sources."""
        py_file = tmp_path / "test.py"
        py_file.write_text("print(1)")
        js_file = tmp_path / "test.js"
        js_file.write_text("console.log(1)")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(run_command, ["-l", "raw", str(py_file), str(js_file)])

            assert result.exit_code == 0
            calls = _calls(mock_scheduler)
            # Both should be raw (forced by -l)
            assert calls[0]["language"] == Language.RAW
            assert calls[1]["language"] == Language.RAW


def _crashed() -> SandboxError:
    return TransientError("QEMU died.", code=ErrorCode.QEMU_CRASHED, context=None, hint=None, instance_id=None)


def _boot_timeout() -> SandboxError:
    return TransientError(
        "Guest agent not ready", code=ErrorCode.VM_BOOT_TIMEOUT, context=None, hint=None, instance_id=None
    )


def _output_limit() -> SandboxError:
    return InvalidInputError(
        "Output limit exceeded", code=ErrorCode.OUTPUT_LIMIT, context=None, hint=None, instance_id=None
    )


def _no_qemu() -> SandboxError:
    return ConfigError("qemu-img not found", code=ErrorCode.HOST_DEPENDENCY, context=None, hint=None, instance_id=None)


class TestCliMultiInputExitCodes:
    """Tests for exit code handling with multiple sources."""

    @pytest.mark.parametrize(
        ("outcomes", "exit_code"),
        [
            pytest.param((0, 0), 0, id="all-passed"),
            pytest.param((0, 5), 5, id="highest-exit-code"),
            pytest.param((255, 0), 255, id="exit-255"),
            pytest.param((-2, 0), 0, id="only-minus-one-is-a-timeout"),
            pytest.param((0, _crashed), 125, id="sandbox-error"),
            pytest.param((2, _crashed), 125, id="code-exit-2-is-not-fix"),
            pytest.param((124, _crashed), 125, id="code-exit-124-is-not-timeout"),
            pytest.param((130, _boot_timeout), 124, id="boot-timeout-over-exit-code"),
            pytest.param((130, -1, _crashed), 124, id="code-timeout-over-error"),
            pytest.param((130, _boot_timeout, _crashed, _output_limit), 2, id="fix-input-first"),
            pytest.param((130, _boot_timeout, _crashed, _no_qemu), 2, id="fix-config-first"),
            pytest.param((-1, _output_limit), 2, id="fix-input-over-code-timeout"),
        ],
    )
    def test_exit_code(self, runner: CliRunner, outcomes: tuple[Any, ...], exit_code: int) -> None:
        """Each source exits with a code or fails with an error. The run exits 2 when an error
        is FIX_INPUT or FIX_CONFIG, else 124 when a VM did not boot in time or the code of a
        source timed out (-1), else 125 when a source failed with another error, else the
        highest exit code. Code that exits 2 or 124 itself is an exit code, not an error."""

        def exec_(code: str, **_: Any) -> ExecutionResult:
            outcome = outcomes[int(code.removeprefix("source_"))]
            if isinstance(outcome, int):
                return _result(outcome)
            raise outcome()

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _vm(_mock_scheduler(scheduler_cls)).exec.side_effect = exec_

            result = runner.invoke(run_command, ["-q", *(f"source_{i}" for i in range(len(outcomes)))])

        assert result.exit_code == exit_code, result.output

    def test_package_not_allowed_exits_2_like_one_source(self, runner: CliRunner) -> None:
        """Every source has the same packages: each source shows the single-source error
        (header and advice), and the run exits 2."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            mock_scheduler.new.side_effect = InvalidInputError(
                "Package 'nope' not in python allow-list.",
                code=ErrorCode.PACKAGE_NOT_ALLOWED,
                context=None,
                hint=None,
                instance_id=None,
            )

            result = runner.invoke(run_command, ["-p", "nope==1.0", "print(1)", "print(2)"])

        assert result.exit_code == 2
        assert result.stderr.count("Error: PACKAGE_NOT_ALLOWED (FIX_INPUT)") == 2
        assert result.stderr.count("To skip this check, use --no-validation.") == 2

    def test_code_timeout_of_one_source_exits_124(self, runner: CliRunner) -> None:
        """One source's code times out (-1), the other exits 0: 124, not 0; its JSON entry
        has exit_code 124."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _vm(_mock_scheduler(scheduler_cls)).exec.side_effect = [_result(0), _result(-1)]

            result = runner.invoke(run_command, ["--json", "pass", "while True: pass"])

        assert result.exit_code == 124
        assert sorted(entry["exit_code"] for entry in json.loads(result.stdout)) == [0, 124]

    def test_failed_source_keeps_the_other_results(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """A SandboxError of one source becomes its result's error, code and action (exit 125 for
        VERIFY), printed after the JSON with its advice; the other source's run stays."""

        def exec_(code: str, **_: Any) -> ExecutionResult:
            if code == "boom()":
                raise OutcomeUnknownError(
                    "guest agent gone", code=ErrorCode.GUEST_OUTCOME_UNKNOWN, context=None, hint=None, instance_id=None
                )
            return mock_result

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _vm(_mock_scheduler(scheduler_cls)).exec.side_effect = exec_

            result = runner.invoke(run_command, ["--json", "print(1)", "boom()"])

        assert result.exit_code == 125
        ok, failed = json.loads(result.stdout)
        assert (ok["index"], ok["run"]["result"]["stdout"], ok["error"], ok["exit_code"]) == (0, "hello\n", None, 0)
        assert failed == {
            "index": 1,
            "source": "boom()",
            "run": None,
            "error": "guest agent gone",
            "code": "GUEST_OUTCOME_UNKNOWN",
            "action": "VERIFY",
            "exit_code": 125,
        }
        assert result.stderr.startswith("Error: GUEST_OUTCOME_UNKNOWN (VERIFY)\n\n  guest agent gone\n")
        assert "Check before you run again" in result.stderr

    def test_fix_config_of_one_source_exits_2_with_its_advice(
        self, runner: CliRunner, mock_result: ExecutionResult, tmp_path: Path
    ) -> None:
        """A FIX_CONFIG error of one source: under that source, the header, the message and the
        advice of a single source; exit 2, also when the other source's code exits higher."""
        py_file = tmp_path / "ok.py"
        py_file.write_text("raise SystemExit(5)")
        js_file = tmp_path / "broken.js"
        js_file.write_text("console.log(1)")
        exit_5 = mock_result.model_copy(update={"exit_code": 5})

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            vm = _vm(scheduler)
            vm.exec.return_value = exit_5

            async def new(spec: VmSpec, **_: Any) -> Session:
                if spec.language is Language.JAVASCRIPT:
                    raise ConfigError(
                        "node binary not found",
                        code=ErrorCode.HOST_DEPENDENCY,
                        context=None,
                        hint=None,
                        instance_id=None,
                    )
                return vm

            scheduler.new.side_effect = new

            result = runner.invoke(run_command, [str(py_file), str(js_file)])

        assert result.exit_code == 2
        assert "Error: HOST_DEPENDENCY (FIX_CONFIG)\n\n  node binary not found\n" in result.stderr
        assert "Correct the settings, then run again." in result.stderr
        assert "✗ node binary not found" not in result.stderr

    def test_fix_input_wins_over_a_boot_timeout_in_json(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """--json: each failed source has its code, action and exit code in the JSON, then its
        error on stderr; a FIX_INPUT error (2) wins over a boot timeout (124)."""

        def exec_(code: str, **_: Any) -> ExecutionResult:
            if code == "big()":
                raise InvalidInputError(
                    "Output limit exceeded", code=ErrorCode.OUTPUT_LIMIT, context=None, hint=None, instance_id=None
                )
            if code == "slow()":
                raise TransientError(
                    "Guest agent not ready", code=ErrorCode.VM_BOOT_TIMEOUT, context=None, hint=None, instance_id=None
                )
            return mock_result

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _vm(_mock_scheduler(scheduler_cls)).exec.side_effect = exec_

            result = runner.invoke(run_command, ["--json", "print(1)", "big()", "slow()"])

        assert result.exit_code == 2
        rows = [(r["code"], r["action"], r["exit_code"]) for r in json.loads(result.stdout)]
        assert rows == [(None, None, 0), ("OUTPUT_LIMIT", "FIX_INPUT", 2), ("VM_BOOT_TIMEOUT", "RETRY", 124)]
        assert result.stderr.index("Error: OUTPUT_LIMIT (FIX_INPUT)") < result.stderr.index(
            "Error: VM_BOOT_TIMEOUT (RETRY)"
        )
        assert "Correct the command, then run again." in result.stderr

    def test_scheduler_start_failure_prints_one_error(self, runner: CliRunner) -> None:
        """A failure before any source runs: one error with its advice, no result; a setup
        error (FIX_CONFIG) exits 2."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            scheduler.__aenter__.side_effect = ConfigError(
                "Failed to start qemu-img: [Errno 2] No such file or directory",
                code=ErrorCode.HOST_DEPENDENCY,
                context=None,
                hint=None,
                instance_id=None,
            )

            result = runner.invoke(run_command, ["print(1)", "print(2)"])

        assert result.exit_code == 2
        assert result.stdout == ""
        assert result.stderr == (
            "Error: HOST_DEPENDENCY (FIX_CONFIG)\n"
            "\n"
            "  Failed to start qemu-img: [Errno 2] No such file or directory\n"
            "\n"
            "  Suggestions:\n"
            "    • Install or fix the host dependency the message names "
            "(QEMU: brew install qemu on macOS, the distribution's qemu packages on Linux).\n"
            "    • Correct the settings, then run again.\n"
        )
        scheduler.new.assert_not_called()


# ============================================================================
# Ctrl+C and the loop of a command
# ============================================================================
class TestCliInterrupt:
    """Ctrl+C (SIGINT) during a run, and the end of the loop each command runs in."""

    def test_ctrl_c_closes_each_source_before_the_scheduler_stops(self, runner: CliRunner) -> None:
        """Ctrl+C while two sources run: each closes its session, then the scheduler stops (its
        stop destroys every VM, so no source may still use one); exit 130."""
        events: list[str] = []
        running = 0

        async def exec_(_code: str, **_: Any) -> ExecutionResult:
            nonlocal running
            running += 1
            if running == 2:
                signal.raise_signal(signal.SIGINT)  # Ctrl+C once both sources run
            await asyncio.Event().wait()
            raise AssertionError("only a cancel ends the run")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            _vm(scheduler).exec.side_effect = exec_
            _vm(scheduler).__aexit__.side_effect = lambda *_: events.append("session closed")
            scheduler.__aexit__.side_effect = lambda *_: events.append("scheduler stopped")

            result = runner.invoke(run_command, ["a = 1", "b = 2"])

        assert result.exit_code == 130
        assert events == ["session closed", "session closed", "scheduler stopped"]

    def test_async_generator_left_open_is_closed(self) -> None:
        """An async generator that a command leaves open runs its cleanup before the command
        returns, as with asyncio.run (a reference to it outlives the command)."""
        events: list[str] = []
        kept: list[AsyncIterator[int]] = []

        async def numbers() -> AsyncIterator[int]:
            try:
                yield 1
                yield 2
            finally:
                events.append("closed")

        async def command() -> int:
            kept.append(numbers())
            await anext(kept[0])
            return 0

        assert _run_with_interrupt_handling(command()) == 0
        assert events == ["closed"]

    def test_worker_thread_job_ends_before_the_command_returns(self) -> None:
        """A job that a command starts in the default executor and does not await has ended
        when the command returns, as with asyncio.run."""
        ended = threading.Event()

        def job() -> None:
            time.sleep(0.05)
            ended.set()

        async def command() -> int:
            asyncio.get_running_loop().run_in_executor(None, job)
            return 0

        assert _run_with_interrupt_handling(command()) == 0
        assert ended.is_set()


# ============================================================================
# Prefetch Command Tests
# ============================================================================
class TestCliPrefetch:
    """Tests for CLI prefetch command."""

    def test_prefetch_help(self, runner: CliRunner) -> None:
        """Shows help for prefetch command."""
        result = runner.invoke(cli, ["prefetch", "--help"])
        assert result.exit_code == 0
        assert "Download the VM assets before use" in result.output
        assert "--arch" in result.output
        assert "--quiet" in result.output

    def test_prefetch_downloads_all(self, runner: CliRunner, tmp_path: Path) -> None:
        """Prefetch downloads all assets and prints each one, then the cache folder."""
        with patch("exec_sandbox.cli.prefetch_all_assets") as mock_prefetch:
            mock_prefetch.return_value = PrefetchResult(
                success=True,
                arch="aarch64",
                downloaded=["kernel", "initramfs"],
                cache_dir=tmp_path,
            )

            result = runner.invoke(prefetch_command, [])

            assert result.exit_code == 0
            mock_prefetch.assert_called_once_with(arch=None)
            assert result.stdout == (
                f"Prefetching exec-sandbox assets for aarch64...\n  kernel: done\n  initramfs: done\n\nDone. Cache: {tmp_path}\n"
            )

    def test_prefetch_with_arch(self, runner: CliRunner, tmp_path: Path) -> None:
        """Prefetch respects --arch flag."""
        with patch("exec_sandbox.cli.prefetch_all_assets") as mock_prefetch:
            mock_prefetch.return_value = PrefetchResult(
                success=True,
                arch="aarch64",
                downloaded=["kernel", "initramfs"],
                cache_dir=tmp_path,
            )

            result = runner.invoke(prefetch_command, ["--arch", "aarch64"])

            assert result.exit_code == 0
            mock_prefetch.assert_called_once_with(arch="aarch64")

    def test_prefetch_quiet(self, runner: CliRunner, tmp_path: Path) -> None:
        """Prefetch respects -q flag (suppresses output on success)."""
        with patch("exec_sandbox.cli.prefetch_all_assets") as mock_prefetch:
            mock_prefetch.return_value = PrefetchResult(
                success=True,
                arch="aarch64",
                downloaded=["kernel", "initramfs"],
                cache_dir=tmp_path,
            )

            result = runner.invoke(prefetch_command, ["-q"])

            assert result.exit_code == 0
            # Quiet mode should produce no output on success
            assert result.output == ""

    def test_prefetch_returns_error_on_failure(self, runner: CliRunner) -> None:
        """A failed download: exit 1, the asset and its error on stderr, then the network advice."""
        with patch("exec_sandbox.cli.prefetch_all_assets") as mock_prefetch:
            mock_prefetch.return_value = PrefetchResult(
                success=False,
                arch="aarch64",
                errors=[("kernel", "Network error")],
            )

            result = runner.invoke(prefetch_command, [])

        assert result.exit_code == 1
        assert result.stdout == "Prefetching exec-sandbox assets for aarch64...\n"
        assert result.stderr == (
            "  kernel: failed\n"
            "\n"
            "Error: Download failed\n"
            "\n"
            "  Failed to download kernel: Network error\n"
            "\n"
            "  Suggestions:\n"
            "    - Check your network connection\n"
            "    - Set GITHUB_TOKEN for higher rate limits\n"
        )


# ============================================================================
# CLI Group Tests (Top-level help and version)
# ============================================================================
class TestCliGroup:
    """Tests for CLI group (top-level commands)."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """The group help ends with its four commands and their one-line help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert result.output.endswith(
            "Commands:\n"
            "  run       Run code in an isolated VM.\n"
            "  instance  List, describe and remove saved instances.\n"
            "  worker    Show this worker's facts for a broker.\n"
            "  prefetch  Download the VM assets before use, for offline hosts.\n"
        )

    @pytest.mark.parametrize("flag", ["--version", "-V"])
    def test_cli_version(self, runner: CliRunner, flag: str) -> None:
        """--version and -V print the program name and the package version, then exit 0."""
        result = runner.invoke(cli, [flag])
        assert result.exit_code == 0
        assert result.output == f"exec-sandbox, version {__version__}\n"

    def test_run_subcommand_help(self, runner: CliRunner) -> None:
        """Shows help for run subcommand."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run code in an isolated VM." in result.output
        assert "--language" in result.output
        assert "--timeout" in result.output


# ============================================================================
# Subcommands: named, or 'run' when no command name is given
# ============================================================================
class TestCliSubcommands:
    """`sbx run` and `sbx prefetch` by name; `sbx CODE` is `sbx run CODE` (main inserts 'run')."""

    def test_explicit_run(self, runner: CliRunner, mock_result: ExecutionResult) -> None:
        """`sbx run CODE` runs CODE in a python VM."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            result = runner.invoke(cli, ["run", "print(1)"])

        assert result.exit_code == 0
        assert (_call(mock_scheduler)["code"], _call(mock_scheduler)["language"]) == ("print(1)", Language.PYTHON)

    def test_explicit_prefetch(self, runner: CliRunner, tmp_path: Path) -> None:
        """`sbx prefetch` downloads the assets of this host's architecture."""
        with patch("exec_sandbox.cli.prefetch_all_assets") as mock_prefetch:
            mock_prefetch.return_value = PrefetchResult(
                success=True,
                arch="aarch64",
                downloaded=["kernel"],
                cache_dir=tmp_path,
            )

            result = runner.invoke(cli, ["prefetch"])

        assert result.exit_code == 0
        mock_prefetch.assert_called_once_with(arch=None)

    def test_main_inserts_run_before_code(self, monkeypatch: pytest.MonkeyPatch, mock_result: ExecutionResult) -> None:
        """`sbx 'print(1)'`: the first argument is no command name, so main() runs it as `sbx run 'print(1)'`."""
        monkeypatch.setattr(sys, "argv", ["sbx", "print(1)"])
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            mock_scheduler = _mock_scheduler(scheduler_cls)
            _vm(mock_scheduler).exec.return_value = mock_result

            with pytest.raises(SystemExit) as exited:
                main()

        assert exited.value.code == 0
        assert sys.argv == ["sbx", "run", "print(1)"]
        assert _call(mock_scheduler)["code"] == "print(1)"


# ============================================================================
# Saved instances: sbx run --instance
# ============================================================================
_ID = "my-vm"


def _result(exit_code: int) -> ExecutionResult:
    return ExecutionResult(
        stdout="",
        stderr="",
        exit_code=exit_code,
        execution_time_ms=1,
        timing=TimingBreakdown(execute_ms=1, connect_ms=None),
    )


def _info(*, spec: VmSpec | None, generation: int) -> InstanceInfo:
    """What describe_instance gives for _ID; spec None: this keyring lacks the key (access no_key)."""
    return InstanceInfo(
        instance_id=_ID,
        generation=generation,
        recent_commits=(RecentCommit(generation=generation, commit_id="a" * 32),),
        key_id="b" * 64,
        access="full" if spec is not None else "no_key",
        writer_release=None if spec is None else "1.0.1",
        spec=spec,
        host=None if spec is None else _SEEDED_HOST,
        guest=None if spec is None else _SEEDED_GUEST,
        egress_domains=None if spec is None else (),
    )


def _new_instance(scheduler: AsyncMock, *, exit_code: int) -> AsyncMock:
    """_ID does not exist; the code exits with `exit_code`; a save makes generation 1. Returns the VM."""
    scheduler.describe_instance.side_effect = InstanceGoneError(
        "No instance.", code=ErrorCode.INSTANCE_NOT_FOUND, context=None, hint=None, instance_id=_ID
    )
    vm = _vm(scheduler)
    vm.exec.return_value = _result(exit_code)
    vm.save.return_value = InstanceRef(instance_id=_ID, generation=1)
    return vm


def _saved_instance(scheduler: AsyncMock, *, spec: VmSpec | None, exit_code: int) -> AsyncMock:
    """_ID exists at generation 3 (made from `spec`); a save makes generation 4. Returns the VM."""
    scheduler.describe_instance.return_value = _info(spec=spec, generation=3)
    vm = _vm(scheduler)
    vm.instance = InstanceRef(instance_id=_ID, generation=3)
    vm.exec.return_value = _result(exit_code)
    vm.save.return_value = InstanceRef(instance_id=_ID, generation=4)
    return vm


def _config(scheduler_cls: MagicMock) -> SchedulerConfig:
    return scheduler_cls.call_args.args[0]


class TestCliInstanceRun:
    """`sbx run --instance ID`: reuse or make, then save."""

    def test_absent_instance_is_made_with_the_id(self, runner: CliRunner) -> None:
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            vm = _new_instance(scheduler, exit_code=0)

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "x = 1"])

        assert result.exit_code == 0, result.output
        scheduler.describe_instance.assert_awaited_once_with(_ID)
        scheduler.new.assert_awaited_once_with(VmSpec(language=Language.PYTHON), on_boot_log=None, instance_id=_ID)
        scheduler.reuse.assert_not_called()
        vm.save.assert_awaited_once_with()
        assert f"Created instance {_ID}, generation 1" in result.stderr

    def test_existing_instance_is_reused(self, runner: CliRunner) -> None:
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            vm = _saved_instance(scheduler, spec=VmSpec(language=Language.PYTHON), exit_code=0)

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "print(x)"])

        assert result.exit_code == 0, result.output
        scheduler.reuse.assert_awaited_once_with(_ID)
        scheduler.new.assert_not_called()
        vm.exec.assert_awaited_once()
        vm.save.assert_awaited_once_with()
        assert f"Saved instance {_ID}, generation 4" in result.stderr

    @pytest.mark.parametrize("exit_code", [1, 42])
    def test_failed_code_is_saved_and_its_exit_code_kept(self, runner: CliRunner, exit_code: int) -> None:
        """A non-zero exit leaves the session and its state: the run saves it."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _saved_instance(_mock_scheduler(scheduler_cls), spec=None, exit_code=exit_code)

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "raise SystemExit(1)"])

        assert result.exit_code == exit_code
        vm.save.assert_awaited_once_with()

    @pytest.mark.parametrize(
        ("existing", "kept"),
        [(False, "no instance was made"), (True, "it stays at generation 3")],
        ids=["new", "reused"],
    )
    def test_timeout_saves_nothing(self, runner: CliRunner, existing: bool, kept: str) -> None:
        """Exit -1: the guest stopped the REPL, so the stored generation keeps the state."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            if existing:
                vm = _saved_instance(scheduler, spec=None, exit_code=-1)
            else:
                vm = _new_instance(scheduler, exit_code=-1)

            result = runner.invoke(run_command, ["--instance", _ID, "-q", "-c", "while True: pass"])

        assert result.exit_code == 124
        vm.save.assert_not_called()
        assert f"Instance {_ID} not saved: the code timed out; {kept}." in result.stderr

    def test_stopped_vm_saves_nothing(self, runner: CliRunner) -> None:
        """Exit 137 closes the session: there is no VM left to save."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _saved_instance(_mock_scheduler(scheduler_cls), spec=None, exit_code=137)
            vm.closed = True

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "import os; os.kill(1, 9)"])

        assert result.exit_code == 137
        vm.save.assert_not_called()
        assert f"Instance {_ID} not saved: the VM stopped; it stays at generation 3." in result.stderr

    def test_order_is_uploads_exec_downloads_save(self, runner: CliRunner, tmp_path: Path) -> None:
        """Downloads come before the save: a save that fails can close the session."""
        upload = tmp_path / "in.txt"
        upload.write_text("in")
        download = tmp_path / "out.txt"

        async def fake_read(path: str, *, destination: Path) -> None:
            destination.write_bytes(b"out")

        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _new_instance(_mock_scheduler(scheduler_cls), exit_code=0)
            vm.read_file.side_effect = fake_read

            result = runner.invoke(
                run_command,
                ["--instance", _ID, "--upload", f"{upload}:in.txt", "--download", f"out.txt:{download}", "-c", "pass"],
            )

        assert result.exit_code == 0, result.output
        steps = [name for name, _, _ in vm.mock_calls if name in {"write_file", "exec", "read_file", "save"}]
        assert steps == ["write_file", "exec", "read_file", "save"]

    @pytest.mark.parametrize("json_output", [False, True], ids=["text", "json"])
    def test_failed_download_still_saves(self, runner: CliRunner, tmp_path: Path, json_output: bool) -> None:
        """The code ran and the session is open: the run saves and prints the save line, then the
        download error (GUEST_IO, FIX_INPUT: exit 2). The JSON names the saved generation."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _new_instance(_mock_scheduler(scheduler_cls), exit_code=0)
            vm.read_file.side_effect = InvalidInputError(
                "read_file: [io_error] File not found: out.txt",
                code=ErrorCode.GUEST_IO,
                context=None,
                hint=None,
                instance_id=None,
            )

            result = runner.invoke(
                run_command,
                [
                    "--instance",
                    _ID,
                    *(["--json"] if json_output else []),
                    "--download",
                    f"out.txt:{tmp_path / 'out.txt'}",
                    "-c",
                    "pass",
                ],
            )

        assert result.exit_code == 2
        vm.save.assert_awaited_once_with()
        assert result.stderr.index(f"Created instance {_ID}, generation 1") < result.stderr.index(
            "Error: GUEST_IO (FIX_INPUT)"
        )
        if json_output:
            assert OneShotRun.model_validate_json(result.stdout).instance == InstanceRef(instance_id=_ID, generation=1)

    @pytest.mark.parametrize("instance", [True, False], ids=["saved", "no-instance"])
    def test_transient_download_error_advice(self, runner: CliRunner, tmp_path: Path, instance: bool) -> None:
        """A transient download error: retry advice without an instance; after a run that saved,
        the advice names the saved generation instead (a new run runs the code again on it)."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _new_instance(_mock_scheduler(scheduler_cls), exit_code=0)
            vm.read_file.side_effect = TransientError(
                "read_file: [timeout_error] Timed out reading out.txt",
                code=ErrorCode.GUEST_UNAVAILABLE,
                context=None,
                hint=None,
                instance_id=None,
            )

            instance_args = ["--instance", _ID] if instance else []
            download = f"out.txt:{tmp_path / 'out.txt'}"
            result = runner.invoke(run_command, [*instance_args, "-q", "--download", download, "-c", "pass"])

        assert result.exit_code == 125
        assert "Error: GUEST_UNAVAILABLE (RETRY)" in result.stderr
        assert "Timed out reading out.txt" in result.stderr
        if instance:
            assert f"This run saved instance {_ID} at generation 1." in result.stderr
            assert "again later" not in result.stderr
        else:
            assert "Run the same command again later." in result.stderr
            assert "saved" not in result.stderr

    def test_quiet_hides_the_save_line(self, runner: CliRunner) -> None:
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _new_instance(_mock_scheduler(scheduler_cls), exit_code=0)

            result = runner.invoke(run_command, ["--instance", _ID, "-q", "-c", "pass"])

        assert result.exit_code == 0, result.output
        vm.save.assert_awaited_once_with()
        assert result.stderr == ""

    @pytest.mark.parametrize(
        ("args", "exit_code", "expected"),
        [
            (["--instance", _ID], 0, {"instance_id": _ID, "generation": 1}),
            (["--instance", _ID], -1, None),
            ([], 0, None),
        ],
        ids=["saved", "timeout", "no-instance"],
    )
    def test_json_instance_field(
        self, runner: CliRunner, args: list[str], exit_code: int, expected: dict[str, Any] | None
    ) -> None:
        """OneShotRun.instance is what this run saved, else null."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _new_instance(_mock_scheduler(scheduler_cls), exit_code=exit_code)

            result = runner.invoke(run_command, [*args, "--json", "-c", "pass"])

        assert json.loads(result.stdout)["instance"] == expected

    @pytest.mark.parametrize("bad", ["My-VM", "-vm", "a" * 65, "a/b", ""])
    def test_invalid_id_exits_2_before_any_start(self, runner: CliRunner, account: Account, bad: str) -> None:
        """The library's id rule, checked before the keyring, the store or a VM opens."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            result = runner.invoke(run_command, ["--instance", bad, "-c", "pass"])

        assert result.exit_code == 2
        assert "An instance id is 1 to 64 lowercase letters" in result.output
        scheduler_cls.assert_not_called()
        account.assert_untouched()

    def test_several_sources_are_refused(self, runner: CliRunner) -> None:
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            result = runner.invoke(run_command, ["--instance", _ID, "-c", "a = 1", "-c", "b = 2"])

        assert result.exit_code == 2
        assert "--instance requires exactly one source" in result.output
        scheduler_cls.assert_not_called()


_JS = VmSpec(language=Language.JAVASCRIPT)
_BIG = VmSpec(language=Language.PYTHON, memory_mb=512)
_SIX = VmSpec(language=Language.PYTHON, packages=("six==1.17.0",))
_DOMAINS = VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=("a.example.com",))
_PORTS = VmSpec(language=Language.PYTHON, expose_ports=(PortMapping(internal=8080),))


class TestCliInstanceSpecFlags:
    """VM-shaping options on reuse: absent takes the saved value, equal is accepted, different exits 2."""

    @pytest.mark.parametrize(
        ("saved", "args"),
        [
            (_BIG, []),
            (_BIG, ["-m", "512"]),
            (_JS, []),  # inline code without -l takes the saved language
            (_JS, ["-l", "javascript"]),
            (_SIX, ["-p", "six==1.17.0"]),
            (_DOMAINS, ["--network"]),
            (_DOMAINS, ["--network", "--allow-domain", "a.example.com"]),
            (_PORTS, ["--expose", "8080"]),
        ],
        ids=["memory-absent", "memory-equal", "language-absent", "language-equal", "packages-equal",
             "network-equal", "domains-equal", "ports-equal"],
    )  # fmt: skip
    def test_absent_or_equal_option_reuses(self, runner: CliRunner, saved: VmSpec, args: list[str]) -> None:
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            _saved_instance(scheduler, spec=saved, exit_code=0)

            result = runner.invoke(run_command, ["--instance", _ID, *args, "-c", "pass"])

        assert result.exit_code == 0, result.output
        scheduler.reuse.assert_awaited_once_with(_ID)

    @pytest.mark.parametrize(
        ("saved", "args", "conflict"),
        [
            (_BIG, ["-m", "256"], "--memory 256 (saved: 512)"),
            (_JS, ["-l", "python"], '--language "python" (saved: "javascript")'),
            (_SIX, ["-p", "six==1.16.0"], '--package ["six==1.16.0"] (saved: ["six==1.17.0"])'),
            (_JS, ["--network"], "--network true (saved: false)"),
            (
                _DOMAINS,
                ["--network", "--allow-domain", "b.example.com"],
                '--allow-domain ["b.example.com"] (saved: ["a.example.com"])',
            ),
            (
                _PORTS,
                ["--expose", "9090"],
                '--expose [{"internal":9090,"external":null,"protocol":"tcp"}] '
                '(saved: [{"internal":8080,"external":null,"protocol":"tcp"}])',
            ),
        ],
        ids=["memory", "language", "packages", "network", "domains", "ports"],
    )
    def test_different_option_exits_2_before_boot(
        self, runner: CliRunner, saved: VmSpec, args: list[str], conflict: str
    ) -> None:
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            _saved_instance(scheduler, spec=saved, exit_code=0)

            result = runner.invoke(run_command, ["--instance", _ID, *args, "-c", "pass"])

        assert result.exit_code == 2
        assert f"Instance {_ID} has other VM settings: {conflict}" in result.output
        scheduler.reuse.assert_not_called()
        scheduler.new.assert_not_called()

    def test_file_extension_sets_the_language(self, runner: CliRunner, tmp_path: Path) -> None:
        """A .py file asks for python: a javascript instance refuses it."""
        script = tmp_path / "step.py"
        script.write_text("print(1)")
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            _saved_instance(scheduler, spec=_JS, exit_code=0)

            result = runner.invoke(run_command, ["--instance", _ID, str(script)])

        assert result.exit_code == 2
        assert '--language "python" (saved: "javascript")' in result.output
        scheduler.reuse.assert_not_called()

    def test_unknown_spec_reuses(self, runner: CliRunner) -> None:
        """Without the instance's key, describe gives no spec: reuse() then says what is wrong."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            _saved_instance(scheduler, spec=None, exit_code=0)

            result = runner.invoke(run_command, ["--instance", _ID, "-m", "999", "-c", "pass"])

        assert result.exit_code == 0, result.output
        scheduler.reuse.assert_awaited_once_with(_ID)


class TestCliInstanceStore:
    """Which store and keyring the CLI gives the Scheduler."""

    def test_local_store_and_os_keyring_by_default(
        self, runner: CliRunner, account: Account, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The keyring lock is in the account's data folder (home from the user database), never
        under HOME or XDG_*: one keyring entry per account, so every process of the account must
        share the lock."""
        _on_linux(monkeypatch)
        monkeypatch.setenv("HOME", str(tmp_path / "other-home"))
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "other-xdg"))
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _new_instance(_mock_scheduler(scheduler_cls), exit_code=0)

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "pass"])

        assert result.exit_code == 0, result.output
        assert (account.linux_data_dir() / "instance-keyring.lock").is_file()
        assert not (tmp_path / "other-home").exists()
        assert not (tmp_path / "other-xdg").exists()
        config = _config(scheduler_cls)
        assert config.instance_store == "local"
        assert config.instance_keyring is not None
        assert config.instance_keyring.get_secret_value() == _LOCAL_KEY
        assert "keyring" not in result.stderr.lower()  # an existing key: no notice

    def test_keyring_variable_wins_over_the_os_keyring(
        self, runner: CliRunner, account: Account, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", _ENV_KEY)
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _new_instance(_mock_scheduler(scheduler_cls), exit_code=0)

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "pass"])

        assert result.exit_code == 0, result.output
        account.assert_untouched()
        config = _config(scheduler_cls)
        assert config.instance_store == "local"
        assert config.instance_keyring is not None
        assert config.instance_keyring.get_secret_value() == _ENV_KEY

    def test_s3_store_never_uses_the_os_keyring(
        self, runner: CliRunner, account: Account, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """S3 needs EXEC_SANDBOX_INSTANCE_KEYRING; without it the Scheduler start refuses (exit 2)."""
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE", "s3")
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            scheduler.__aenter__.side_effect = ConfigError(
                "instance_keyring is not set.",
                code=ErrorCode.INSTANCE_CONFIG,
                context=None,
                hint="Set EXEC_SANDBOX_INSTANCE_KEYRING.",
                instance_id=None,
            )

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "pass"])

        assert result.exit_code == 2
        account.assert_untouched()
        config = _config(scheduler_cls)
        assert (config.instance_store, config.instance_keyring) == ("s3", None)
        assert "Set EXEC_SANDBOX_INSTANCE_KEYRING." in result.stderr

    @pytest.mark.parametrize("sources", [["-c", "pass"], ["-c", "a = 1", "-c", "b = 2"]], ids=["one", "several"])
    def test_without_instance_no_store_opens(
        self,
        runner: CliRunner,
        account: Account,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sources: list[str],
    ) -> None:
        """instance_store=None wins over EXEC_SANDBOX_INSTANCE_STORE: no store, no keyring, no save."""
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE", "local")
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            _vm(scheduler).exec.return_value = _result(0)

            result = runner.invoke(run_command, sources)

        assert result.exit_code == 0, result.output
        # The Settings the Scheduler builds from this config: the variable does not turn the store on
        assert settings_from_config(_config(scheduler_cls), images_dir=tmp_path).instance_store is None
        account.assert_untouched()
        scheduler.describe_instance.assert_not_called()
        _vm(scheduler).save.assert_not_called()

    def test_new_key_notice_names_the_entry_not_the_key(self, runner: CliRunner, account: Account) -> None:
        """The first run makes the key in the OS keyring, gives it to the Scheduler and says where it is."""
        account.keyring.entries.clear()
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _new_instance(_mock_scheduler(scheduler_cls), exit_code=0)

            result = runner.invoke(run_command, ["--instance", _ID, "-q", "--json", "-c", "pass"])

        assert result.exit_code == 0, result.output
        new_key = account.keyring.entries[KEYRING_ENTRY]
        config_keyring = _config(scheduler_cls).instance_keyring
        assert config_keyring is not None
        assert config_keyring.get_secret_value() == new_key
        assert '"exec-sandbox" / "instance-keyring"' in result.stderr
        assert "keyring get exec-sandbox instance-keyring" in result.stderr
        assert "Keep a copy" in result.stderr
        assert new_key not in result.output
        json.loads(result.stdout)  # stdout stays pure JSON

    def test_keyring_failure_exits_2_with_its_hint(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_os_keyring(monkeypatch)
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            result = runner.invoke(run_command, ["--instance", _ID, "-c", "pass"])

        assert result.exit_code == 2
        assert "(FIX_CONFIG)" in result.stderr
        assert "is not an OS keyring" in result.stderr
        assert "Set EXEC_SANDBOX_INSTANCE_KEYRING" in result.stderr
        scheduler_cls.assert_not_called()


def _key_missing() -> SandboxError:
    return ConfigError(
        "The key is not in this keyring.",
        code=ErrorCode.INSTANCE_KEY_MISSING,
        context=None,
        hint="Add the key.",
        instance_id=_ID,
    )


def _host_mismatch() -> SandboxError:
    return host_mismatch(
        instance_id=_ID, required={RouteKey.ARCH: "x86_64"}, found={RouteKey.ARCH: "aarch64"}, reason=None
    )


def _store_down() -> SandboxError:
    return TransientError(
        "The store did not answer.",
        code=ErrorCode.INSTANCE_STORE_UNAVAILABLE,
        context=None,
        hint=None,
        instance_id=_ID,
    )


def _expired() -> SandboxError:
    return InstanceGoneError(
        "Part of the state is gone.", code=ErrorCode.INSTANCE_EXPIRED, context=None, hint=None, instance_id=_ID
    )


def _conflict() -> SandboxError:
    """An INSTANCE_CONFLICT as the store raises it: with the generation and commit id of the save
    that was not stored (save_context)."""
    return OutcomeUnknownError(
        "Another save won.",
        code=ErrorCode.INSTANCE_CONFLICT,
        context={"generation": 4, "commit_id": "c" * 32},
        hint=None,
        instance_id=_ID,
    )


def _refused() -> SandboxError:
    return ConfigError(
        "The store refused this worker.", code=ErrorCode.INSTANCE_CONFIG, context=None, hint=None, instance_id=_ID
    )


class TestCliInstanceErrors:
    """Exit 2 for FIX_INPUT and FIX_CONFIG, 124 for VM_BOOT_TIMEOUT, 125 for the other errors; the
    message names the code, the action and what to do."""

    @pytest.mark.parametrize("code", [ErrorCode.INSTANCE_INTEGRITY, ErrorCode.INSTANCE_EXPIRED])
    def test_damaged_instance_is_not_replaced(self, runner: CliRunner, code: ErrorCode) -> None:
        """describe finds the instance but cannot use it: the RECREATE advice, and no new VM
        takes its id (only INSTANCE_NOT_FOUND makes a new instance)."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            scheduler.describe_instance.side_effect = InstanceGoneError(
                "The head is damaged.", code=code, context=None, hint=None, instance_id=_ID
            )

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "pass"])

        assert result.exit_code == 125
        assert f"Error: {code} (RECREATE)" in result.stderr
        assert f"Remove it with 'sbx instance rm {_ID}'" in result.stderr
        scheduler.new.assert_not_called()
        scheduler.reuse.assert_not_called()

    @pytest.mark.parametrize(
        ("error", "exit_code", "words"),
        [
            (_key_missing, 2, ["INSTANCE_KEY_MISSING (FIX_CONFIG)", "Add the key."]),
            (_host_mismatch, 125, ["INSTANCE_HOST_MISMATCH (ROUTE)", 'arch="x86_64"', "Do not remove"]),
            (_store_down, 125, ["INSTANCE_STORE_UNAVAILABLE (RETRY)", "again later"]),
            (_expired, 125, ["INSTANCE_EXPIRED (RECREATE)", f"sbx instance rm {_ID}"]),
        ],
        ids=["key", "host", "store", "expired"],
    )
    def test_reuse_errors(self, runner: CliRunner, error: Any, exit_code: int, words: list[str]) -> None:
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            _saved_instance(scheduler, spec=None, exit_code=0)
            scheduler.reuse.side_effect = error()

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "pass"])

        assert result.exit_code == exit_code
        for word in words:
            assert word in result.stderr

    def test_save_conflict_exits_125_and_says_verify(self, runner: CliRunner) -> None:
        """Another run made or saved the instance first: the run's output stays, its state does not."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _new_instance(_mock_scheduler(scheduler_cls), exit_code=0)
            vm.save.side_effect = _conflict()

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "pass"])

        assert result.exit_code == 125
        assert "INSTANCE_CONFLICT (VERIFY)" in result.stderr
        assert "lists commit" not in result.stderr  # a conflict stored nothing
        assert "Created instance" not in result.stderr

    def test_save_unknown_outcome_names_the_commit_to_look_for(self, runner: CliRunner) -> None:
        """The store did not confirm the save: the advice names the command and the commit id."""
        unknown = OutcomeUnknownError(
            "No answer.",
            code=ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN,
            context={"generation": 2, "commit_id": "c" * 32},
            hint=None,
            instance_id=_ID,
        )
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _new_instance(_mock_scheduler(scheduler_cls), exit_code=0)
            vm.save.side_effect = unknown

            result = runner.invoke(run_command, ["--instance", _ID, "-c", "pass"])

        assert result.exit_code == 125
        assert f"The save is stored if 'sbx instance describe {_ID}' lists commit {'c' * 32}." in result.stderr

    @pytest.mark.parametrize(
        ("error", "exit_code", "words"),
        [(_conflict, 125, "INSTANCE_CONFLICT (VERIFY)"), (_refused, 2, "(FIX_CONFIG)")],
        ids=["verify", "fix-config"],
    )
    def test_failed_save_keeps_the_json_result(self, runner: CliRunner, error: Any, exit_code: int, words: str) -> None:
        """--json: the result of the code that ran goes to stdout (instance null), the save error
        to stderr, and the exit code is the instance error's."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            vm = _new_instance(_mock_scheduler(scheduler_cls), exit_code=3)
            vm.save.side_effect = error()

            result = runner.invoke(run_command, ["--instance", _ID, "--json", "-c", "raise SystemExit(3)"])

        assert result.exit_code == exit_code
        run = OneShotRun.model_validate_json(result.stdout)
        assert (run.result.exit_code, run.instance) == (3, None)
        assert words in result.stderr


# ============================================================================
# Saved instances: sbx instance list | describe | rm
# ============================================================================
@dataclass(frozen=True)
class _AdminMocks:
    opener: MagicMock
    """open_instance_admin, specced on the real one."""
    admin: AsyncMock
    """The InstanceAdmin the opener yields."""


@pytest.fixture
def admin_mocks(monkeypatch: pytest.MonkeyPatch) -> Iterator[_AdminMocks]:
    """The InstanceAdmin of the `sbx instance` commands, specced on the real class and opener:
    a CLI that calls a removed method or passes other arguments fails here, as it would in use.
    The test fails if the CLI starts a Scheduler."""
    admin = create_autospec(InstanceAdmin, instance=True)
    opener = create_autospec(open_instance_admin)
    opener.return_value.__aenter__.return_value = admin
    monkeypatch.setattr(
        "exec_sandbox.cli.require_instance_admin", create_autospec(require_instance_admin, return_value=opener)
    )
    scheduler_cls = create_autospec(Scheduler)
    monkeypatch.setattr("exec_sandbox.cli.Scheduler", scheduler_cls)
    yield _AdminMocks(opener=opener, admin=admin)
    scheduler_cls.assert_not_called()


@pytest.fixture
def local_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The variables of a real local store in tmp_path (not made yet), with its key: the OS keyring is not used."""
    root = tmp_path / "store"
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE_DIR", str(root))
    monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", _ENV_KEY)
    return root


class _FakeCheckScheduler:
    """A Scheduler for `sbx instance check`: its ids, and check_reuse raising `errors[id]`;
    its start raises `start_error` when set."""

    def __init__(self, ids: list[str], errors: dict[str, Exception]) -> None:
        self.ids, self.errors, self.checked = ids, errors, list[str]()
        self.start_error: Exception | None = None

    def __call__(self, _config: object) -> Self:
        return self

    async def __aenter__(self) -> Self:
        if self.start_error is not None:
            raise self.start_error
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def list_instances(self) -> AsyncIterator[str]:
        for instance_id in self.ids:
            yield instance_id

    async def check_reuse(self, instance_id: str) -> None:
        self.checked.append(instance_id)
        if (error := self.errors.get(instance_id)) is not None:
            raise error


class TestCliInstanceCheck:
    """`sbx instance check`: a line for each instance, exit 1 when one cannot continue here,
    125 when the store or this host fails (no answer)."""

    @staticmethod
    def _run(runner: CliRunner, monkeypatch: pytest.MonkeyPatch, fake: _FakeCheckScheduler, *args: str) -> Result:
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", _ENV_KEY)
        monkeypatch.setattr("exec_sandbox.cli.Scheduler", fake)
        return runner.invoke(cli, ["instance", "check", *args])

    def test_a_line_for_each_and_exit_1(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        gone = InstanceGoneError(
            "No instance b1.", code=ErrorCode.INSTANCE_NOT_FOUND, context=None, hint=None, instance_id="b1"
        )
        fake = _FakeCheckScheduler(["a1", "b1"], {"b1": gone})
        result = self._run(runner, monkeypatch, fake)
        assert result.exit_code == 1, result.output
        assert result.stdout.splitlines() == ["a1 ok", "b1 INSTANCE_NOT_FOUND (RECREATE): No instance b1."]

    def test_json_lines(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._run(runner, monkeypatch, _FakeCheckScheduler(["a1"], {}), "--json")
        assert result.exit_code == 0, result.output
        assert [json.loads(line) for line in result.stdout.splitlines()] == [{"instance_id": "a1", "ok": True}]

    def test_a_store_failure_stops_with_125(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        """A RETRY error is no answer about the instance: the check stops, exit 125."""
        down = TransientError(
            "store down", code=ErrorCode.INSTANCE_STORE_UNAVAILABLE, context=None, hint=None, instance_id=None
        )
        fake = _FakeCheckScheduler(["a1", "b1"], {"a1": down})
        result = self._run(runner, monkeypatch, fake)
        assert result.exit_code == 125, result.output
        assert fake.checked == ["a1"]

    def test_a_host_that_cannot_run_vms_stops_with_2(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        """A FIX_CONFIG error at the start (this host cannot run its QEMU): exit 2, and no
        instance is checked (no line says that an instance failed)."""
        fake = _FakeCheckScheduler(["a1"], {})
        fake.start_error = ConfigError(
            "QEMU cannot run.", code=ErrorCode.HOST_DEPENDENCY, context=None, hint=None, instance_id=None
        )
        result = self._run(runner, monkeypatch, fake)
        assert result.exit_code == 2, result.output
        assert (fake.checked, result.stdout) == ([], "")
        assert "HOST_DEPENDENCY (FIX_CONFIG)" in result.stderr

    def test_sample_picks_from_the_store(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        """K distinct instances of the store."""
        fake = _FakeCheckScheduler([f"id{n}" for n in range(10)], {})
        result = self._run(runner, monkeypatch, fake, "--sample", "3")
        assert result.exit_code == 0, result.output
        assert len(set(fake.checked)) == len(fake.checked) == 3 and set(fake.checked) <= set(fake.ids)

    def test_sample_with_ids_is_a_usage_error(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeCheckScheduler([], {})
        result = self._run(runner, monkeypatch, fake, "--sample", "3", _ID)
        assert result.exit_code == 2
        assert "--sample picks instances of the store" in result.output
        assert fake.checked == []


class TestCliCheckAndWorkerConfig:
    """`sbx instance check` and `sbx worker describe` on the specced Scheduler mock: the config
    each one starts (the local store, the keyring variable, no warm pool) and its output."""

    def test_the_check_starts_on_the_instance_store_with_no_pool(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", _ENV_KEY)
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            scheduler = _mock_scheduler(scheduler_cls)
            result = runner.invoke(cli, ["instance", "check", _ID])

        assert result.exit_code == 0, result.output
        assert result.stdout == f"{_ID} ok\n"
        scheduler.check_reuse.assert_awaited_once_with(_ID)
        config = _config(scheduler_cls)
        assert (config.instance_store, config.warm_pool_size) == ("local", 0)
        assert config.instance_keyring is not None
        assert config.instance_keyring.get_secret_value() == _ENV_KEY

    def test_worker_describe_prints_the_facts(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", _ENV_KEY)
        facts = WorkerFacts(values={RouteKey.ARCH: "aarch64", RouteKey.CPU: "1.26"}, key_ids=("1" * 64,))
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _mock_scheduler(scheduler_cls).describe_worker.return_value = facts
            result = runner.invoke(cli, ["worker", "describe"])

        assert result.exit_code == 0, result.output
        assert WorkerFacts.model_validate_json(result.stdout) == facts
        config = _config(scheduler_cls)
        assert (config.instance_store, config.warm_pool_size) == ("local", 0)
        assert config.instance_keyring is not None
        assert config.instance_keyring.get_secret_value() == _ENV_KEY

    def test_worker_describe_on_a_host_that_cannot_run_vms_exits_2(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", _ENV_KEY)
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            _mock_scheduler(scheduler_cls).__aenter__.side_effect = ConfigError(
                "QEMU cannot run.", code=ErrorCode.HOST_DEPENDENCY, context=None, hint=None, instance_id=None
            )
            result = runner.invoke(cli, ["worker", "describe"])

        assert result.exit_code == 2
        assert result.stdout == ""
        assert "HOST_DEPENDENCY (FIX_CONFIG)" in result.stderr


class TestCliInstanceCommands:
    """`sbx instance list | describe | rm` open the store and the keyring only: no Scheduler, no VM."""

    def test_list_prints_one_id_per_line(self, runner: CliRunner, admin_mocks: _AdminMocks) -> None:
        admin_mocks.admin.list_ids.return_value.__aiter__.return_value = ["a1", _ID]

        result = runner.invoke(cli, ["instance", "list"])

        assert result.exit_code == 0, result.output
        assert result.stdout == f"a1\n{_ID}\n"

    @pytest.mark.parametrize("ids", [[], ["a1", _ID]], ids=["empty", "two"])
    def test_list_json(self, runner: CliRunner, admin_mocks: _AdminMocks, ids: list[str]) -> None:
        admin_mocks.admin.list_ids.return_value.__aiter__.return_value = ids

        result = runner.invoke(cli, ["instance", "list", "--json"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == ids

    def test_keys_prints_the_id_of_each_key(
        self, runner: CliRunner, local_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The id of each key of the keyring, newest first (the key_id that describe shows):
        one on each line, or a JSON array. The store is not opened: its folder is not made."""
        newer = new_keyring()
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", f"{newer},{_ENV_KEY}")
        expected = [Keyring.parse(newer).latest.id, Keyring.parse(_ENV_KEY).latest.id]

        result = runner.invoke(cli, ["instance", "keys"])
        assert result.exit_code == 0, result.output
        assert result.stdout.splitlines() == expected
        as_json = runner.invoke(cli, ["instance", "keys", "--json"])
        assert json.loads(as_json.stdout) == expected
        assert not local_store.exists()

    @pytest.mark.parametrize("as_array", [False, True], ids=["lines", "json"])
    def test_list_long_describes_each_instance(
        self, runner: CliRunner, admin_mocks: _AdminMocks, as_array: bool
    ) -> None:
        """--long: what describe shows for each id, one JSON object per line (--json: an
        array); an id it cannot describe gives its error as data, and the listing goes on."""
        info = _info(spec=_SIX, generation=3)
        admin_mocks.admin.list_ids.return_value.__aiter__.return_value = ["a1", _ID]
        gone = InstanceGoneError(
            "No instance a1.", code=ErrorCode.INSTANCE_NOT_FOUND, context=None, hint=None, instance_id="a1"
        )

        async def describe(instance_id: str) -> InstanceInfo:
            if instance_id == "a1":
                raise gone
            return info

        admin_mocks.admin.describe.side_effect = describe
        result = runner.invoke(cli, ["instance", "list", "--long", *(["--json"] if as_array else [])])

        assert result.exit_code == 0, result.output
        objects = json.loads(result.stdout) if as_array else [json.loads(line) for line in result.stdout.splitlines()]
        error = {"code": "INSTANCE_NOT_FOUND", "action": "RECREATE", "message": "No instance a1."}
        assert objects[0] == {"instance_id": "a1", "error": error}
        assert InstanceInfo.model_validate(objects[1]) == info

    def test_describe_prints_the_info_as_json(self, runner: CliRunner, admin_mocks: _AdminMocks) -> None:
        info = _info(spec=_SIX, generation=3)
        admin_mocks.admin.describe.return_value = info

        result = runner.invoke(cli, ["instance", "describe", _ID])

        assert result.exit_code == 0, result.output
        admin_mocks.admin.describe.assert_awaited_once_with(_ID)
        assert InstanceInfo.model_validate_json(result.stdout) == info

    def test_describe_of_an_absent_instance_exits_125(self, runner: CliRunner, local_store: Path) -> None:
        """The real admin, on an empty local store."""
        result = runner.invoke(cli, ["instance", "describe", _ID])

        assert result.exit_code == 125
        assert "INSTANCE_NOT_FOUND (RECREATE)" in result.stderr
        assert result.stdout == ""

    def test_rm_removes_each_id_and_exits_0_when_gone(self, runner: CliRunner, admin_mocks: _AdminMocks) -> None:
        admin_mocks.admin.delete.side_effect = [True, False]

        result = runner.invoke(cli, ["instance", "rm", _ID, "gone"])

        assert result.exit_code == 0, result.output
        assert [call.args for call in admin_mocks.admin.delete.await_args_list] == [(_ID,), ("gone",)]
        assert result.stdout == f"{_ID}\n"
        assert result.stderr == "No instance gone.\n"

    @pytest.mark.parametrize(
        ("args", "error"),
        [
            (["rm"], "Missing argument 'ID...'"),
            (["rm", _ID, "Bad"], "An instance id is 1 to 64 lowercase letters"),
            (["describe", "Bad"], "An instance id is 1 to 64 lowercase letters"),
        ],
        ids=["none", "rm-bad", "describe-bad"],
    )
    def test_bad_ids_exit_2_before_the_store_opens(
        self, runner: CliRunner, account: Account, admin_mocks: _AdminMocks, args: list[str], error: str
    ) -> None:
        result = runner.invoke(cli, ["instance", *args])

        assert result.exit_code == 2
        assert error in result.stderr
        admin_mocks.opener.assert_not_called()
        account.assert_untouched()

    def test_store_error_exit_code(self, runner: CliRunner, admin_mocks: _AdminMocks) -> None:
        admin_mocks.admin.delete.side_effect = _store_down()

        result = runner.invoke(cli, ["instance", "rm", _ID])

        assert result.exit_code == 125
        assert "INSTANCE_STORE_UNAVAILABLE (RETRY)" in result.stderr

    def test_main_does_not_insert_run(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], admin_mocks: _AdminMocks
    ) -> None:
        """`sbx instance list` is a subcommand, not code for an implicit `sbx run` (no Scheduler starts)."""
        monkeypatch.setattr(sys, "argv", ["sbx", "instance", "list"])
        admin_mocks.admin.list_ids.return_value.__aiter__.return_value = [_ID]

        with pytest.raises(SystemExit) as exited:
            main()

        assert exited.value.code == 0
        assert sys.argv == ["sbx", "instance", "list"]
        assert capsys.readouterr().out == f"{_ID}\n"


class TestCliInstanceCommandsStore:
    """Which store and keyring the `sbx instance` commands open (the rules of `sbx run --instance`),
    through the real admin and local store: no Scheduler, no VM."""

    def test_local_store_and_os_keyring_by_default(
        self, runner: CliRunner, account: Account, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No variable: the store in the user's data folder (under HOME), opened with a key the first
        run makes in the OS keyring; the keyring lock in the account's data folder."""
        _on_linux(monkeypatch)
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        account.keyring.entries.clear()

        result = runner.invoke(cli, ["instance", "list", "--json"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == []
        store = home / ".local" / "share" / "exec-sandbox" / "instances"
        assert stat.S_IMODE(store.stat().st_mode) & 0o777 == 0o700
        assert (account.linux_data_dir() / "instance-keyring.lock").is_file()
        assert "Keep a copy" in result.stderr  # the new-key notice, on stderr only
        key = account.keyring.entries[KEYRING_ENTRY]
        assert key not in result.output

        # The store opens with that key: a head sealed with it reads with access full.
        async def seed() -> None:
            async with LocalInstanceStore.open(root=store) as opened:
                await seed_instance(
                    opened,
                    tmp_path,
                    instance_id=_ID,
                    keyring=key,
                    spec=_SEEDED_SPEC,
                    host=_SEEDED_HOST,
                    guest=_SEEDED_GUEST,
                )

        asyncio.run(seed())
        result = runner.invoke(cli, ["instance", "describe", _ID])
        assert result.exit_code == 0, result.output
        info = InstanceInfo.model_validate_json(result.stdout)
        assert (info.access, info.spec) == ("full", _SEEDED_SPEC)

    def test_variables_win_over_the_defaults(
        self,
        runner: CliRunner,
        account: Account,
        local_store: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _on_linux(monkeypatch)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))

        result = runner.invoke(cli, ["instance", "list"])

        assert result.exit_code == 0, result.output
        assert result.stdout == ""
        assert local_store.is_dir()
        assert not (tmp_path / "home").exists()  # not the default folder
        account.assert_untouched()

    def test_s3_store_never_uses_the_os_keyring(
        self, runner: CliRunner, account: Account, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """S3 needs EXEC_SANDBOX_INSTANCE_KEYRING; without it the store does not open (exit 2)."""
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE", "s3")

        result = runner.invoke(cli, ["instance", "rm", _ID])

        assert result.exit_code == 2
        assert "(FIX_CONFIG)" in result.stderr
        assert "EXEC_SANDBOX_INSTANCE_KEYRING" in result.stderr
        assert result.stdout == ""
        account.assert_untouched()

    def test_s3_store_without_the_s3_extra_names_the_extra(
        self, runner: CliRunner, local_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The S3 store module cannot be imported: EXTRA_MISSING (FIX_CONFIG), the install line,
        exit 2, no QEMU advice."""
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE", "s3")
        monkeypatch.setenv("EXEC_SANDBOX_S3_BUCKET", "bucket")
        monkeypatch.setitem(sys.modules, "exec_sandbox.instance.s3_store", None)

        result = runner.invoke(cli, ["instance", "list"])

        assert result.exit_code == 2
        assert "Error: EXTRA_MISSING (FIX_CONFIG)" in result.stderr
        assert "pip install 'exec-sandbox[s3]'" in result.stderr
        assert "QEMU" not in result.stderr
        assert result.stdout == ""

    def test_rm_of_objects_without_a_head(self, runner: CliRunner, local_store: Path, tmp_path: Path) -> None:
        """Objects under an id with no head (a first save cut off): rm removes them and says there
        was no instance (exit 0); a second rm says the same."""
        src = tmp_path / "object"
        src.write_bytes(b"left over")

        async def leave_object() -> None:
            async with LocalInstanceStore.open(root=local_store) as store:
                await store.upload(body_key(_ID, "0" * 32, "f"), src=src)

        async def objects() -> list[str]:
            async with LocalInstanceStore.open(root=local_store) as store:
                return [o.key for o in await store.list_objects("")]

        asyncio.run(leave_object())
        assert asyncio.run(objects()) != []

        for _ in range(2):
            result = runner.invoke(cli, ["instance", "rm", _ID])
            assert result.exit_code == 0, result.output
            assert result.stdout == ""
            assert result.stderr == f"No instance {_ID}.\n"
        assert asyncio.run(objects()) == []

    def test_keyring_failure_exits_2_before_the_store_opens(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _no_os_keyring(monkeypatch)
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE_DIR", str(tmp_path / "store"))

        result = runner.invoke(cli, ["instance", "list"])

        assert result.exit_code == 2
        assert "is not an OS keyring" in result.stderr
        assert "Set EXEC_SANDBOX_INSTANCE_KEYRING" in result.stderr
        assert not (tmp_path / "store").exists()


# ============================================================================
# Settings errors: EXEC_SANDBOX_* variables and options
# ============================================================================
class TestCliSettingsErrors:
    """Invalid settings exit 2 before any start: one line per field, never a value, no traceback."""

    @pytest.mark.parametrize(
        "args",
        [
            ["instance", "list"],
            ["instance", "describe", _ID],
            ["instance", "rm", _ID],
            ["run", "-c", "pass"],
            ["run", "--instance", _ID, "-c", "pass"],
            ["run", "-c", "a = 1", "-c", "b = 2"],
        ],
        ids=["list", "describe", "rm", "run", "run-instance", "run-several"],
    )
    def test_invalid_variables(
        self, runner: CliRunner, account: Account, monkeypatch: pytest.MonkeyPatch, args: list[str]
    ) -> None:
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_STORE", "nfs")
        monkeypatch.setenv("EXEC_SANDBOX_MEMORY_OVERCOMMIT_RATIO", "secret-ratio")
        monkeypatch.setenv("EXEC_SANDBOX_INSTANCE_KEYRING", _ENV_KEY)
        with (
            patch("exec_sandbox.cli.Scheduler") as scheduler_cls,
            patch("exec_sandbox.cli.require_instance_admin") as require,
        ):
            result = runner.invoke(cli, args)

        assert result.exit_code == 2
        assert isinstance(result.exception, SystemExit), result.exception  # an exit, not a crash
        assert "Error: Invalid settings" in result.stderr
        fields = [line.strip() for line in result.stderr.splitlines() if line.startswith("    ")]
        assert fields == [
            "instance_store: Input should be 's3' or 'local' (got str)",
            "memory_overcommit_ratio: Input should be a valid number, unable to parse string as a number (got str)",
        ]
        for value in ("nfs", "secret-ratio", _ENV_KEY):
            assert value not in result.output
        scheduler_cls.assert_not_called()
        require.assert_not_called()
        account.assert_untouched()

    def test_rule_across_fields_names_the_settings(
        self, runner: CliRunner, admin_mocks: _AdminMocks, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model rule (overlapping S3 prefixes) has no field: its line names the settings."""
        monkeypatch.setenv("EXEC_SANDBOX_S3_SNAPSHOT_PREFIX", "a/")
        monkeypatch.setenv("EXEC_SANDBOX_S3_INSTANCE_PREFIX", "a/")

        result = runner.invoke(cli, ["instance", "list"])

        assert result.exit_code == 2
        (line,) = [line.strip() for line in result.stderr.splitlines() if line.startswith("    ")]
        assert line.startswith("Settings: Value error, ")
        assert "overlap" in line
        admin_mocks.opener.assert_not_called()

    @pytest.mark.parametrize("timeout", ["0", "-1"])
    def test_option_out_of_range(self, runner: CliRunner, timeout: str) -> None:
        """-t builds SchedulerConfig.default_timeout_seconds (1 to 300)."""
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            result = runner.invoke(run_command, ["-t", timeout, "-c", "pass"])

        assert result.exit_code == 2
        assert isinstance(result.exception, SystemExit), result.exception
        assert "    default_timeout_seconds: Input should be greater than or equal to 1 (got int)" in (
            result.stderr.splitlines()
        )
        scheduler_cls.assert_not_called()


class TestCliSpecOptionErrors:
    """VM options the VmSpec refuses, and --allow-domain without --network: a usage error (exit 2)
    with one line that names the option, before the Scheduler starts, for one source or several."""

    @pytest.mark.parametrize(
        ("args", "line"),
        [
            (["-m", "100"], "Error: Invalid VM options: '--memory': Input should be greater than or equal to 128."),
            (
                [arg for port in range(8000, 8011) for arg in ("--expose", str(port))],
                "Error: Invalid VM options: '--expose': Tuple should have at most 10 items after validation, not 11.",
            ),
            (["--allow-domain", "a.example.com"], "Error: --allow-domain requires --network."),
        ],
        ids=["memory", "expose", "domain-without-network"],
    )
    @pytest.mark.parametrize("sources", [["-c", "pass"], ["-c", "a = 1", "-c", "b = 2"]], ids=["one", "several"])
    def test_usage_error_before_any_start(
        self, runner: CliRunner, args: list[str], line: str, sources: list[str]
    ) -> None:
        with patch("exec_sandbox.cli.Scheduler") as scheduler_cls:
            result = runner.invoke(run_command, [*args, *sources])

        assert result.exit_code == 2
        assert isinstance(result.exception, SystemExit), result.exception  # an exit, not a traceback
        assert result.stderr.splitlines()[-1] == line
        scheduler_cls.assert_not_called()

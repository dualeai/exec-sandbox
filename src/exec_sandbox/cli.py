"""Command-line interface for exec-sandbox.

Usage:
    sbx 'print("hello")'           # Run inline code
    sbx script.py                  # Run file
    echo "print(1)" | sbx -        # Run from stdin
    sbx -l python -p pandas==3.0.1 'import pandas; print(pandas.__version__)'

Multi-input (each source in its own VM, all at the same time):
    sbx 'print(1)' 'print(2)'      # Multiple inline codes
    sbx script1.py script2.py      # Multiple files

Saved instances:
    sbx run --instance my-vm -c 'x = 41'          # Makes instance my-vm, saves it
    sbx run --instance my-vm -c 'print(x + 1)'    # Continues it: prints 42, saves it
    sbx instance list | describe ID | rm ID...
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Generator

    from exec_sandbox import Session
    from exec_sandbox.instance.manager import InstanceAdmin
    from exec_sandbox.settings import InstanceStoreKind

import click
from click.core import ParameterSource
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError, computed_field
from pydantic_core import to_json

from exec_sandbox import (
    ErrorAction,
    ErrorCode,
    ExecutionResult,
    ExposedPort,
    InstanceGoneError,
    InstanceNotSavedWarning,
    InstanceRef,
    InvalidInputError,
    Language,
    PortMapping,
    SandboxError,
    Scheduler,
    SchedulerConfig,
    StartupTiming,
    VmSpec,
    __version__,
    constants,
)
from exec_sandbox._imports import require_cli_keyring, require_instance_admin
from exec_sandbox._logging import configure_logging
from exec_sandbox.assets import PrefetchResult, prefetch_all_assets
from exec_sandbox.dns_filter import validate_domain
from exec_sandbox.instance.ids import check_instance_id
from exec_sandbox.platform_utils import UserDir, get_account_dir
from exec_sandbox.process_registry import force_kill_all, get_tracked_count
from exec_sandbox.settings import Settings


def _run_with_interrupt_handling(coro: Coroutine[Any, Any, int]) -> int:
    """Run coroutine with proper Ctrl+C handling.

    An asyncio.Runner (the loop of asyncio.run(): at the end, the tasks left are cancelled
    and awaited, the async generators closed and the default executor's jobs awaited) with
    the SIGINT handler below: a Runner adds its own only over the default handler. No child
    holds the terminal: every spawn of the package sets stdin (/dev/null, or a pipe of its
    own; test_every_spawn_site_sets_stdin), so no child changes its modes and Ctrl+C stays SIGINT.

    - First Ctrl+C: Cancels main task, cleanup (__aexit__) runs
    - Second Ctrl+C: Force kill all child process groups immediately
    """
    interrupts = 0
    main_task: asyncio.Task[Any] | None = None
    runner = asyncio.Runner()

    def handle_sigint(_signum: int, _frame: Any) -> None:
        nonlocal interrupts
        interrupts += 1
        if interrupts > 1:
            count = get_tracked_count()
            if count > 0:
                click.echo(f"\nForce killing {count} VM process(es)...", err=True)
                force_kill_all()
            sys.exit(130)
        click.echo("\nInterrupted. Stopping the VMs (press Ctrl+C again to stop them at once)...", err=True)
        if main_task is not None and not main_task.done():
            main_task.cancel()
            runner.get_loop().call_soon_threadsafe(lambda: None)  # wake a select() with a long timeout

    async def main() -> int:
        nonlocal main_task
        main_task = asyncio.current_task()
        return await coro

    original_handler = signal.signal(signal.SIGINT, handle_sigint)
    try:
        with runner:
            return runner.run(main())
    except (asyncio.CancelledError, KeyboardInterrupt):
        sys.exit(130)
    finally:
        signal.signal(signal.SIGINT, original_handler)


# Exit codes following Unix conventions
EXIT_SUCCESS = 0
EXIT_CLI_ERROR = 2
EXIT_TIMEOUT = 124  # Matches `timeout` command
EXIT_SANDBOX_ERROR = 125

EXTENSION_MAP: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".sh": Language.RAW,
}


@dataclass
class SourceInput:
    """Represents a single source input for execution."""

    code: str
    label: str  # Display label (file path or truncated code)
    language: Language


def detect_language(source: str) -> Language | None:
    """The language of a source from its file extension (EXTENSION_MAP, any case), or None:
    no extension (the stdin marker "-" has none) or another one."""
    return EXTENSION_MAP.get(Path(source).suffix.lower())


def parse_env_vars(env_vars: tuple[str, ...]) -> dict[str, str]:
    """Parse KEY=VALUE environment variable strings.

    Args:
        env_vars: Tuple of "KEY=VALUE" strings

    Returns:
        Dictionary of environment variables

    Raises:
        click.BadParameter: If format is invalid
    """
    result: dict[str, str] = {}
    for env_var in env_vars:
        if "=" not in env_var:
            raise click.BadParameter(
                f"Invalid format: '{env_var}'. Use KEY=VALUE format.",
                param_hint="'-e' / '--env'",
            )
        key, value = env_var.split("=", 1)
        if not key:
            raise click.BadParameter(
                f"Empty key in: '{env_var}'. Use KEY=VALUE format.",
                param_hint="'-e' / '--env'",
            )
        result[key] = value
    return result


def parse_port_mappings(port_specs: tuple[str, ...]) -> list[PortMapping]:
    """Parse port mapping specifications.

    Formats:
        8080           - Internal port only (external auto-assigned)
        8080:3000      - Internal:external
        8080:3000/udp  - With protocol

    Args:
        port_specs: Tuple of port specification strings

    Returns:
        List of PortMapping objects

    Raises:
        click.BadParameter: If format is invalid
    """
    result: list[PortMapping] = []
    for spec in port_specs:
        port_spec = spec  # Avoid overwriting loop variable
        try:
            protocol = "tcp"
            if "/" in port_spec:
                port_spec, protocol = port_spec.rsplit("/", 1)
                if protocol not in ("tcp", "udp"):
                    raise click.BadParameter(
                        f"Invalid protocol: '{protocol}'. Use 'tcp' or 'udp'.",
                        param_hint="'--expose'",
                    )

            if ":" in port_spec:
                parts = port_spec.split(":")
                if len(parts) != 2:  # noqa: PLR2004
                    raise click.BadParameter(
                        f"Invalid port format: '{spec}'. Use INTERNAL or INTERNAL:EXTERNAL.",
                        param_hint="'--expose'",
                    )
                internal, external = int(parts[0]), int(parts[1])
                result.append(
                    PortMapping(internal=internal, external=external, protocol=protocol)  # type: ignore[arg-type]
                )
            else:
                internal = int(port_spec)
                result.append(PortMapping(internal=internal, protocol=protocol))  # type: ignore[arg-type]

        except ValidationError as e:  # a ValueError: caught before the int() one
            raise click.BadParameter(
                f"Invalid port mapping: '{spec}'. {e}",
                param_hint="'--expose'",
            ) from e
        except ValueError as e:
            raise click.BadParameter(
                f"Invalid port number in: '{spec}'. Ports must be integers.",
                param_hint="'--expose'",
            ) from e

    return result


def parse_upload_spec(spec: str) -> tuple[str, str]:
    """Parse an upload specification in LOCAL:GUEST format.

    Args:
        spec: Upload spec string, e.g. "./data.csv:data.csv"

    Returns:
        Tuple of (local_path, guest_path)

    Raises:
        click.BadParameter: If format is invalid
    """
    local, sep, guest = spec.partition(":")
    if not sep or not guest:
        raise click.BadParameter(
            f"Invalid upload format: '{spec}'. Use LOCAL:GUEST (for example ./data.csv:data.csv).",
            param_hint="'--upload'",
        )
    if not local:
        raise click.BadParameter(
            f"Empty local path in: '{spec}'. Use LOCAL:GUEST format.",
            param_hint="'--upload'",
        )
    # Expand ~ since shell can't do it inside colon-separated args
    # Uses os.path.expanduser (not Path.expanduser) to avoid normalizing ./relative paths
    local = os.path.expanduser(local)  # noqa: PTH111
    if not Path(local).exists():
        raise click.BadParameter(
            f"Local file not found: '{local}'.",
            param_hint="'--upload'",
        )
    return (local, guest)


def parse_download_spec(spec: str) -> tuple[str, str]:
    """Parse a download specification in GUEST:LOCAL or GUEST format.

    Args:
        spec: Download spec string, e.g. "output.csv:./results/out.csv" or "output.csv"

    Returns:
        Tuple of (guest_path, local_path)

    Raises:
        click.BadParameter: If format is invalid
    """
    if not spec:
        raise click.BadParameter(
            "Empty download path.",
            param_hint="'--download'",
        )
    guest, sep, local = spec.partition(":")
    if not guest:
        raise click.BadParameter(
            f"Empty guest path in: '{spec}'. Use GUEST:LOCAL or GUEST format.",
            param_hint="'--download'",
        )
    if not sep:
        # Shorthand: GUEST only → download to ./basename(GUEST)
        local = Path(guest).name
    if not local:
        raise click.BadParameter(
            f"Empty local path in: '{spec}'. Use GUEST:LOCAL (for example output.csv:./out.csv).",
            param_hint="'--download'",
        )
    # Expand ~ since shell can't do it inside colon-separated args
    # Uses os.path.expanduser (not Path.expanduser) to avoid normalizing ./relative paths
    local = os.path.expanduser(local)  # noqa: PTH111
    if Path(local).is_dir():
        raise click.BadParameter(
            f"Local path is a folder, not a file: '{local}'. Give a file path (for example ~/results/output.csv).",
            param_hint="'--download'",
        )
    return (guest, local)


def format_error(title: str, message: str, *, suggestions: list[str]) -> str:
    """Format an error message following What → Why → Fix pattern.

    Args:
        title: Short error title
        message: Detailed explanation
        suggestions: How to fix the issue (empty: no suggestions block)

    Returns:
        Formatted error string
    """
    lines = [
        click.style(f"Error: {title}", fg="red", bold=True),
        "",
        f"  {message}",
    ]

    if suggestions:
        lines.extend(["", "  Suggestions:"])
        lines.extend(f"    • {suggestion}" for suggestion in suggestions)

    return "\n".join(lines)


class OneShotRun(BaseModel):
    """One VM, one execution: what `--json` prints for a source."""

    model_config = ConfigDict(frozen=True)

    result: ExecutionResult
    startup: StartupTiming
    exposed_ports: list[ExposedPort]
    instance: InstanceRef | None = Field(
        description="With --instance: the generation this run saved. None when the run saved nothing"
    )
    total_ms: int = Field(
        description="From the start of the session (with --instance: describe, then new() or reuse()) "
        "to its close: uploads, exec, downloads and save included, measured by the CLI"
    )


def _error_exit_code(*, action: ErrorAction | None, code: ErrorCode | None) -> int:
    """The exit code of an error: EXIT_CLI_ERROR (2) for FIX_INPUT and FIX_CONFIG,
    EXIT_TIMEOUT (124) for VM_BOOT_TIMEOUT, else EXIT_SANDBOX_ERROR (125)."""
    if action in (ErrorAction.FIX_INPUT, ErrorAction.FIX_CONFIG):
        return EXIT_CLI_ERROR
    if code is ErrorCode.VM_BOOT_TIMEOUT:
        return EXIT_TIMEOUT
    return EXIT_SANDBOX_ERROR


def _timed_out(result: ExecutionResult) -> bool:
    """The code ran past its timeout: the guest stopped it, and the result's exit code is -1.
    The guest keeps the code's own exit code in 0-255, like a process (sys.exit(-1) is 255)."""
    return result.exit_code == -1


def _run_exit_code(result: ExecutionResult) -> int:
    """The exit code of a run: the code's own, or EXIT_TIMEOUT (124) when the code timed out."""
    return EXIT_TIMEOUT if _timed_out(result) else result.exit_code


class MultiSourceResult(BaseModel):
    """Result from a single source in multi-input execution."""

    model_config = ConfigDict(frozen=True)

    index: int
    source: str = Field(description="Label for display")
    run: OneShotRun | None
    error: str | None = Field(description="Why the source has no run: the message of its error")
    code: ErrorCode | None = Field(description="The code of that error; None when the source ran")
    action: ErrorAction | None = Field(description="The action of that error; None when the source ran")

    @computed_field
    @property
    def exit_code(self) -> int:
        """The run's exit code (see _run_exit_code: 124 for a code timeout); for a source with
        no run, its error's (see _error_exit_code)."""
        return _run_exit_code(self.run.result) if self.run else _error_exit_code(action=self.action, code=self.code)


def format_multi_result_json(results: list[MultiSourceResult]) -> str:
    """Format multiple execution results as a JSON array."""
    return _MULTI_RESULTS.dump_json(results, indent=2).decode()


_MULTI_RESULTS = TypeAdapter(list[MultiSourceResult])


_LABEL_CHARS = 40
"""Longest source label shown: a longer one keeps its first 37 characters, then "..."."""


def truncate_source(source: str) -> str:
    """Truncate source label for display."""
    if len(source) <= _LABEL_CHARS:
        return source
    return source[: _LABEL_CHARS - 3] + "..."


def _resolve_source(src: str, language_override: str | None) -> SourceInput:
    """Resolve a single source string to SourceInput.

    Args:
        src: Source string (file path, inline code, or "-" for stdin)
        language_override: Optional language to force (from -l flag)

    Returns:
        SourceInput with resolved code, label, and language

    Raises:
        click.UsageError: If source is empty or invalid
    """
    if src == "-":
        # Read from stdin
        if sys.stdin.isatty():
            raise click.UsageError("No code on stdin. Pipe code to stdin, or use the -c option.")
        code = sys.stdin.read()
        label = "<stdin>"
    # A file when one is at that path, else inline code. os.path.isfile is False for
    # text that cannot be a path (too long, a NUL byte); Path.exists raises
    elif os.path.isfile(src):  # noqa: PTH113
        code = Path(src).read_text()
        label = src
    else:
        code = src
        label = truncate_source(src)

    if not code.strip():
        raise click.UsageError(f"No code in source: {label}")

    # Override (a -l choice, lowercase from click) > file extension > python
    language = Language(language_override or detect_language(src) or Language.PYTHON)
    return SourceInput(code=code, label=label, language=language)


def _display_multi_summary(total_passed: int, total_failed: int, total_sources: int, total_time_ms: int) -> None:
    """Display summary line for multi-source execution."""
    click.echo()
    click.echo("─" * 60)
    if total_failed == 0:
        click.echo(
            click.style(f"✓ {total_passed}/{total_sources} passed", fg="green")
            + click.style(f" ({total_time_ms}ms total)", dim=True)
        )
    else:
        click.echo(
            click.style(f"⚠ {total_passed}/{total_sources} passed, ", fg="yellow")
            + click.style(f"{total_failed} failed", fg="red")
            + click.style(f" ({total_time_ms}ms total)", dim=True)
        )


def _compute_multi_exit_code(results: list[MultiSourceResult]) -> int:
    """Exit code of a multi-source run, from the error of each source with no run (see
    _error_exit_code) and the code timeouts: EXIT_CLI_ERROR when an error is FIX_INPUT or
    FIX_CONFIG (the command or the setup needs a change), else EXIT_TIMEOUT when an error is
    VM_BOOT_TIMEOUT or the code of a source timed out, else the highest exit code of the
    sources. Code that exits 2 or 124 itself does not count here."""
    flagged = {r.exit_code for r in results if r.run is None or _timed_out(r.run.result)}
    for exit_code in (EXIT_CLI_ERROR, EXIT_TIMEOUT):
        if exit_code in flagged:
            return exit_code
    return max((r.exit_code for r in results), default=EXIT_SUCCESS)


def is_tty() -> bool:
    """Check if stdout is connected to a terminal."""
    return sys.stdout.isatty()


def _make_boot_log_callback(debug: bool) -> Callable[[str], None] | None:
    """Create boot log callback for --debug mode, or None if disabled.

    Uses click.echo() directly (not logging) because boot lines are raw
    kernel/init console output with their own timestamps — wrapping them
    in the logging formatter would double-prefix.  click.echo(err=True)
    also handles ANSI stripping on non-TTY automatically.
    """
    if not debug:
        return None

    def _on_boot_log(line: str) -> None:
        click.echo(click.style(f"[boot] {line}", dim=True), err=True)

    return _on_boot_log


def _make_spec(
    *,
    language: Language,
    memory: int,
    packages: list[str],
    network: bool,
    allowed_domains: list[str],
    expose_ports: list[PortMapping] | None,
) -> VmSpec:
    """VmSpec from the CLI flags (empty flag lists mean "none"). A value the VmSpec refuses
    (-m below its minimum, more --expose than its maximum) raises click.UsageError (exit 2)
    with one line that names the option: callers make the spec before any VM starts."""
    try:
        return VmSpec(
            language=language,
            memory_mb=memory,
            packages=tuple(packages),
            allow_network=network,
            allowed_domains=tuple(allowed_domains) if allowed_domains else None,
            expose_ports=tuple(expose_ports or ()),
        )
    except ValidationError as e:
        options = dict(_SPEC_OPTIONS.values())  # VmSpec field -> option name
        problems = [
            f"'{options[err['loc'][0]]}': {err['msg']}" if err["loc"] and err["loc"][0] in options else err["msg"]
            for err in e.errors(include_url=False)
        ]
        raise click.UsageError(f"Invalid VM options: {'; '.join(problems)}.") from None


# ============================================================================
# Settings (EXEC_SANDBOX_* variables)
# ============================================================================


@contextlib.contextmanager
def _valid_settings() -> Generator[None]:
    """Exit 2, with no traceback, when the Settings or SchedulerConfig built inside are not valid.

    One line for each invalid field: its name, why, and the type of the value.
    Never the value: instance_keyring is a secret.
    """
    try:
        yield
    except ValidationError as e:
        fields = [
            f"    {'.'.join(map(str, err['loc'])) or e.title}: {err['msg']} (got {type(err['input']).__name__})"
            for err in e.errors(include_url=False)
        ]
        message = format_error(
            "Invalid settings", "Correct these EXEC_SANDBOX_* variables or options, then run again:", suggestions=[]
        )
        click.echo("\n".join([message, *fields]), err=True)
        raise click.exceptions.Exit(EXIT_CLI_ERROR) from None


def _env_settings() -> Settings:
    """The EXEC_SANDBOX_* settings, checked before any VM or store opens (see _valid_settings)."""
    with _valid_settings():
        return Settings()


# ============================================================================
# Saved instances (sbx run --instance, sbx instance)
# ============================================================================

_SPEC_OPTIONS: dict[str, tuple[str, str]] = {
    "language": ("language", "--language"),
    "packages": ("packages", "--package"),
    "memory": ("memory_mb", "--memory"),
    "network": ("allow_network", "--network"),
    "allowed_domains": ("allowed_domains", "--allow-domain"),
    "expose_ports": ("expose_ports", "--expose"),
}
"""Options of `sbx run` that shape the VM: click parameter name -> (VmSpec field, option name)."""

_ACTION_ADVICE: dict[ErrorAction, str] = {
    ErrorAction.RETRY: "Run the same command again later.",
    ErrorAction.VERIFY: "Check before you run again: the code may have run, or the save may be "
    "stored. For an instance, read the generation with 'sbx instance describe {id}'.",
    ErrorAction.RESTART: "Run the command again.",
    ErrorAction.FIX_INPUT: "Correct the command, then run again.",
    ErrorAction.FIX_CONFIG: "Correct the settings, then run again.",
    ErrorAction.ROUTE: "Run it on a host with the needed values "
    "(CPU, accelerator, QEMU, guest images, exec-sandbox release, memory and CPU budget). "
    "Do not remove a saved instance for this error: its data is good.",
    ErrorAction.RECREATE: "This instance cannot be used again. "
    "Remove it with 'sbx instance rm {id}', then run again to make a new one.",
    ErrorAction.REPORT: "This is a bug or a broken host: report it with the output of a run with "
    "EXEC_SANDBOX_LOG_LEVEL=DEBUG.",
}
"""What the user does next, for each action."""

_PREFETCH_ADVICE = "Check the network, then download the VM images with 'sbx prefetch' and run again."

_CODE_ADVICE: dict[ErrorCode, list[str]] = {
    ErrorCode.HOST_DEPENDENCY: [
        "Install or fix the host dependency the message names "
        "(QEMU: brew install qemu on macOS, the distribution's qemu packages on Linux)."
    ],
    ErrorCode.ASSET_DOWNLOAD: [_PREFETCH_ADVICE],
    ErrorCode.ASSET_NOT_FOUND: [_PREFETCH_ADVICE],
    ErrorCode.PACKAGE_NOT_ALLOWED: [
        "Only the top 10,000 packages of PyPI or npm are allowed.",
        "Check the spelling of the package name.",
        "To skip this check, use --no-validation.",
    ],
    ErrorCode.VM_BOOT_TIMEOUT: [
        "The VM did not start in time, so the code did not run. A boot timeout often comes from CPU contention and passes.",
        "Check the system load and the free resources.",
    ],
}
"""Extra advice for some codes, printed before the action's advice."""

_INSTANCE_IDS = TypeAdapter(list[str])


def _valid_instance_id(value: str) -> str:
    """The id unchanged; an id that breaks the library's id rule is a usage error (exit 2)."""
    try:
        return check_instance_id(value)
    except InvalidInputError as e:
        raise click.BadParameter(f"'{value}': {e.message}") from e


def _instance_id_param(_ctx: click.Context, _param: click.Parameter, value: str | None) -> str | None:
    """Click callback: check an instance id before any VM or store opens."""
    return None if value is None else _valid_instance_id(value)


def _domains_param(_ctx: click.Context, _param: click.Parameter, value: tuple[str, ...]) -> tuple[str, ...]:
    """Click callback: the VmSpec domain rule as a usage error (exit 2), before any VM starts."""
    for domain in value:
        try:
            validate_domain(domain)
        except ValueError as e:
            raise click.BadParameter(f"{domain!r}: {e}") from None
    return value


def _instance_ids_param(_ctx: click.Context, _param: click.Parameter, value: tuple[str, ...]) -> tuple[str, ...]:
    """Click callback: check each instance id before the store opens."""
    return tuple(_valid_instance_id(item) for item in value)


def _instance_store(settings: Settings) -> tuple[InstanceStoreKind, SecretStr | None]:
    """Store and keyring of the CLI's saved instances.

    The store is EXEC_SANDBOX_INSTANCE_STORE, else "local" (EXEC_SANDBOX_INSTANCE_STORE_DIR,
    else the user's data folder). The keyring is EXEC_SANDBOX_INSTANCE_KEYRING; else, for the
    local store only, the key in the OS keyring (made at the first use; the lock of that step
    is in the account's data folder, get_account_dir). The s3 store needs the variable:
    without it, opening the store fails with ConfigError INSTANCE_CONFIG. Raises ConfigError
    INSTANCE_CONFIG when the OS keyring step fails (cli_keyring.open_local_keyring), ConfigError
    EXTRA_MISSING when the cli extra lacks keyring or cryptography.
    """
    store: InstanceStoreKind = settings.instance_store or "local"
    if store != "local" or settings.instance_keyring is not None:
        return store, settings.instance_keyring
    require_cli_keyring()
    from exec_sandbox import cli_keyring  # noqa: PLC0415 - the cli extra, checked by require_cli_keyring

    service, username = cli_keyring.KEYRING_SERVICE, cli_keyring.KEYRING_USERNAME

    # The lock must be shared by every process of this account (one keyring entry per account)
    local = cli_keyring.open_local_keyring(data_dir=get_account_dir(kind=UserDir.DATA))
    if local.created:
        click.echo(
            click.style(
                f'Made a new instance key in the OS keyring (entry "{service}" / "{username}").\n'
                "  Keep a copy in a safe place: without this key, no saved instance opens again.\n"
                f"  Show it with: keyring get {service} {username}\n"
                "  On another host or in a container, set EXEC_SANDBOX_INSTANCE_KEYRING to it.",
                fg="yellow",
            ),
            err=True,
        )
    return store, SecretStr(local.text)


def _scheduler_config(*, timeout: int, no_validation: bool, instances: bool) -> SchedulerConfig:
    """SchedulerConfig of `sbx run`.

    The instance store opens only for --instance: without it, instance_store=None wins over
    EXEC_SANDBOX_INSTANCE_STORE. Invalid EXEC_SANDBOX_* variables or options exit 2 here,
    before any VM starts (the Scheduler reads the variables again).
    """
    settings = _env_settings()
    store, keyring = _instance_store(settings) if instances else (None, None)
    with _valid_settings():
        return SchedulerConfig(
            default_timeout_seconds=timeout,
            enable_package_validation=not no_validation,
            instance_store=store,
            instance_keyring=keyring,
        )


def _spec_flags(ctx: click.Context, *, source: str) -> frozenset[str]:
    """VmSpec fields that the command line sets: the options given, and the language of a file extension."""
    fields = {
        field
        for param, (field, _) in _SPEC_OPTIONS.items()
        if ctx.get_parameter_source(param) is ParameterSource.COMMANDLINE
    }
    if detect_language(source) is not None:
        fields.add("language")
    return frozenset(fields)


def _check_saved_spec(*, instance_id: str, saved: VmSpec, wanted: VmSpec, flag_fields: frozenset[str]) -> None:
    """Refuse options that ask for another VM than the saved instance (its VmSpec is fixed when it is made).

    Only the fields in `flag_fields` count: an option not given takes the saved value.
    """
    conflicts = [
        f"{option} {to_json(getattr(wanted, field)).decode()} (saved: {to_json(getattr(saved, field)).decode()})"
        for field, option in _SPEC_OPTIONS.values()
        if field in flag_fields and getattr(wanted, field) != getattr(saved, field)
    ]
    if conflicts:
        raise click.UsageError(
            f"Instance {instance_id} has other VM settings: {'; '.join(conflicts)}. "
            "Remove these options to use the saved values, or use another --instance id."
        )


async def _open_session(
    scheduler: Scheduler,
    *,
    spec: VmSpec,
    instance_id: str | None,
    flag_fields: frozenset[str],
    on_boot_log: Callable[[str], None] | None,
) -> Session:
    """A new VM from `spec`. With an instance id: describe_instance first; the saved instance
    (reuse) if it exists, else a new VM whose first save makes the instance with that id.
    Options that ask for another VM than the saved one raise click.UsageError (exit 2)
    before any VM starts."""
    if instance_id is None:
        return await scheduler.new(spec, on_boot_log=on_boot_log)
    try:
        info = await scheduler.describe_instance(instance_id)
    except InstanceGoneError as e:
        if e.code is not ErrorCode.INSTANCE_NOT_FOUND:  # never make a new one over a damaged instance
            raise
        return await scheduler.new(spec, on_boot_log=on_boot_log, instance_id=instance_id)
    if info.spec is not None:  # None: this keyring lacks the instance's key, and reuse() says so
        _check_saved_spec(instance_id=instance_id, saved=info.spec, wanted=spec, flag_fields=flag_fields)
    return await scheduler.reuse(instance_id)


async def _download_and_save(
    vm: Session,
    *,
    download_files: tuple[tuple[str, str], ...],
    instance_id: str | None,
    result: ExecutionResult,
    quiet: bool,
) -> tuple[InstanceRef | None, list[SandboxError]]:
    """After the code ran: the downloads, then (with an instance id) the save.

    A failed download stops the downloads after it, not the save, and a failed save
    does not hide the result: both errors come back, for the caller to print after
    the result. The save runs after any exit code >= 0, and stderr tells what
    happened to it. No save after a timeout (exit -1: the guest stopped the REPL) or
    when the VM stopped (the session is closed).
    """
    failures: list[SandboxError] = []
    try:
        # Stream directly to disk — zero memory overhead
        for guest_path, local_path in download_files:
            local = Path(local_path)
            await vm.read_file(guest_path, destination=local)
            if not quiet:
                file_size = local.stat().st_size
                click.echo(
                    click.style(f"  Downloaded: {guest_path} -> {local_path} ({file_size} bytes)", dim=True), err=True
                )
    except SandboxError as e:
        failures.append(e)
    if instance_id is None:
        return None, failures
    loaded = vm.instance
    kept = f"it stays at generation {loaded.generation}" if loaded else "no instance was made"
    if _timed_out(result):
        click.echo(click.style(f"Instance {instance_id} not saved: the code timed out; {kept}.", fg="yellow"), err=True)
        return None, failures
    if vm.closed:
        click.echo(click.style(f"Instance {instance_id} not saved: the VM stopped; {kept}.", fg="yellow"), err=True)
        return None, failures
    try:
        saved = await vm.save()
    except SandboxError as e:
        failures.append(e)
        return None, failures
    if not quiet:
        verb = "Saved" if loaded else "Created"
        click.echo(
            click.style(f"{verb} instance {saved.instance_id}, generation {saved.generation}", dim=True), err=True
        )
    return saved, failures


def _report_error(error: SandboxError, *, saved: InstanceRef | None) -> int:
    """Print an error with its advice (never key material); return its exit code.

    One path for every error: its code and action, its message, its hint, the
    extra advice of its code (_CODE_ADVICE: install, prefetch, package allow-list,
    boot timeout), then the advice of its action. After a run that saved (`saved`,
    for a failed download) the action's advice gives way to a note: a new run
    starts from the saved state and runs the code again on it.

    Exit code: _error_exit_code.
    """
    suggestions = [error.hint] if error.hint else []
    suggestions += _CODE_ADVICE.get(error.code, [])
    if saved is not None:
        suggestions.append(
            f"This run saved instance {saved.instance_id} at generation {saved.generation}. "
            "A new run starts from that state: it runs the code again on it."
        )
    else:
        suggestions.append(_ACTION_ADVICE[error.action].format(id=error.instance_id or "ID"))
    click.echo(format_error(f"{error.code} ({error.action})", error.message, suggestions=suggestions), err=True)
    return _error_exit_code(action=error.action, code=error.code)


async def run_code(
    *,
    code: str,
    language: Language,
    packages: list[str],
    timeout: int,
    memory: int,
    env_vars: dict[str, str],
    network: bool,
    allowed_domains: list[str],
    expose_ports: list[PortMapping] | None,
    json_output: bool,
    quiet: bool,
    no_validation: bool,
    upload_files: tuple[tuple[str, str], ...],
    download_files: tuple[tuple[str, str], ...],
    debug: bool,
    instance_id: str | None,
    flag_fields: frozenset[str],
) -> int:
    """Execute code in sandbox and return exit code.

    One VM for the whole call: uploads, the execution, downloads, then (with an
    instance id) the save. Downloads come before the save: a save that fails can
    close the session. After the code ran, a failed download does not stop the
    save, and the result (the JSON too) is printed before the errors of the
    downloads and the save; the first of these errors sets the exit code (see
    _report_error: 2 for FIX_INPUT and FIX_CONFIG, 124 for VM_BOOT_TIMEOUT, else 125).
    Without an error, the code's exit code, or 124 when the code timed out (_run_exit_code).
    VM options the VmSpec refuses raise click.UsageError (exit 2) before the scheduler starts.

    Args:
        code: Code to execute
        language: Programming language
        packages: Packages to install
        timeout: Timeout in seconds
        memory: Memory in MB
        env_vars: Environment variables
        network: Enable network
        allowed_domains: Allowed domains for network
        expose_ports: Ports to expose from guest to host
        json_output: Output as JSON
        quiet: Suppress progress output
        no_validation: Skip package validation
        upload_files: Files to upload before execution (local_path, guest_path)
        download_files: Files to download after execution (guest_path, local_path)
        debug: Enable boot log streaming to stderr
        instance_id: Saved instance to continue (reuse), or to make when it does not exist
        flag_fields: VmSpec fields that command-line options set (compared with a saved instance)

    Returns:
        Exit code to return from CLI
    """
    spec = _make_spec(
        language=language, memory=memory, packages=packages, network=network,
        allowed_domains=allowed_domains, expose_ports=expose_ports,
    )  # fmt: skip

    # Streaming callbacks for non-JSON output
    def on_stdout(chunk: str) -> None:
        if not json_output:
            click.echo(chunk, nl=False)

    def on_stderr(chunk: str) -> None:
        if not json_output:
            click.echo(chunk, nl=False, err=True)

    on_boot_log = _make_boot_log_callback(debug)

    try:
        config = _scheduler_config(timeout=timeout, no_validation=no_validation, instances=instance_id is not None)
        async with Scheduler(config) as scheduler:
            start = time.perf_counter()
            vm = await _open_session(
                scheduler, spec=spec, instance_id=instance_id, flag_fields=flag_fields, on_boot_log=on_boot_log
            )
            with warnings.catch_warnings():
                # _download_and_save tells the user when a run saves nothing
                warnings.simplefilter("ignore", InstanceNotSavedWarning)
                async with vm:
                    for local_path, guest_path in upload_files:
                        await vm.write_file(guest_path, Path(local_path))
                        if not quiet:
                            click.echo(click.style(f"  Uploaded: {local_path} -> {guest_path}", dim=True), err=True)

                    result = await vm.exec(
                        code,
                        timeout_seconds=timeout,
                        env_vars=env_vars if env_vars else None,
                        on_stdout=on_stdout,
                        on_stderr=on_stderr,
                    )

                    saved, failures = await _download_and_save(
                        vm, download_files=download_files, instance_id=instance_id, result=result, quiet=quiet
                    )
                    startup, exposed_ports = vm.startup_timing, vm.exposed_ports
            run = OneShotRun(
                result=result,
                startup=startup,
                exposed_ports=exposed_ports,
                instance=saved,
                total_ms=round((time.perf_counter() - start) * 1000),
            )

        if json_output:
            click.echo(run.model_dump_json(indent=2))
        codes = [_report_error(e, saved=saved) for e in failures]
        if codes:
            return codes[0]
        if json_output:
            return _run_exit_code(result)

        # TTY mode: show timing footer
        if is_tty() and not quiet:
            click.echo()  # Ensure newline after output
            click.echo(click.style(f"✓ Done in {run.total_ms}ms", fg="green", dim=True), err=True)

        return _run_exit_code(result)

    except SandboxError as e:
        return _report_error(e, saved=None)


async def run_multiple(
    *,
    sources: list[SourceInput],
    packages: list[str],
    timeout: int,
    memory: int,
    env_vars: dict[str, str],
    network: bool,
    allowed_domains: list[str],
    expose_ports: list[PortMapping] | None,
    json_output: bool,
    quiet: bool,
    no_validation: bool,
    debug: bool,
) -> int:
    """Execute multiple sources concurrently; return the exit code of _compute_multi_exit_code.

    The error of a source with no run is printed like the error of a single source
    (_report_error): under its source, or after the JSON with --json.

    Args:
        sources: List of SourceInput objects to execute
        packages: Packages to install (applied to all sources)
        timeout: Timeout in seconds per source
        memory: Memory in MB per VM
        env_vars: Environment variables
        network: Enable network
        allowed_domains: Allowed domains for network
        expose_ports: Ports to expose from guest to host
        json_output: Output as JSON
        quiet: Suppress progress output
        no_validation: Skip package validation
        debug: Enable boot log streaming to stderr

    Returns:
        Exit code to return from CLI (the _report_error one when the scheduler fails)
    """
    # Every spec before the scheduler starts: options the VmSpec refuses exit 2 (click.UsageError)
    specs = [
        _make_spec(
            language=src.language,
            memory=memory,
            packages=packages,
            network=network,
            allowed_domains=allowed_domains,
            expose_ports=expose_ports,
        )
        for src in sources
    ]
    config = _scheduler_config(timeout=timeout, no_validation=no_validation, instances=False)
    on_boot_log = _make_boot_log_callback(debug)

    results: list[MultiSourceResult] = []
    errors: dict[int, SandboxError] = {}  # source index -> why the source has no run
    total_passed = 0
    total_failed = 0
    start_time = time.perf_counter()

    try:
        async with Scheduler(config) as scheduler:
            # One VM per source; an error of a source is its result, the others keep running
            async def run_one(idx: int, src: SourceInput) -> MultiSourceResult:
                try:
                    start = time.perf_counter()
                    async with await scheduler.new(specs[idx], on_boot_log=on_boot_log) as vm:
                        result = await vm.exec(
                            src.code, timeout_seconds=timeout, env_vars=env_vars if env_vars else None
                        )
                        startup, exposed_ports = vm.startup_timing, vm.exposed_ports
                    run = OneShotRun(
                        result=result,
                        startup=startup,
                        exposed_ports=exposed_ports,
                        instance=None,
                        total_ms=round((time.perf_counter() - start) * 1000),
                    )
                    return MultiSourceResult(index=idx, source=src.label, run=run, error=None, code=None, action=None)
                except SandboxError as e:
                    errors[idx] = e
                    return MultiSourceResult(
                        index=idx, source=src.label, run=None, error=e.message, code=e.code, action=e.action
                    )

            tasks = [asyncio.create_task(run_one(i, src)) for i, src in enumerate(sources)]
            try:
                # Stream results as they complete
                for next_done in asyncio.as_completed(tasks):
                    result = await next_done
                    results.append(result)

                    if result.run and result.run.result.exit_code == 0:
                        total_passed += 1
                    else:
                        total_failed += 1

                    if not json_output:
                        _display_multi_result(result, len(sources), quiet, error=errors.get(result.index))
            finally:
                # Ctrl+C, or an error that is not a SandboxError: each source closes its VM before the scheduler stops
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        total_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Sort results by index for consistent output
        results.sort(key=lambda r: r.index)

        if json_output:
            click.echo(format_multi_result_json(results))
            for result in results:
                if result.index in errors:
                    _report_error(errors[result.index], saved=None)
        elif is_tty() and not quiet:
            _display_multi_summary(total_passed, total_failed, len(sources), total_time_ms)

        return _compute_multi_exit_code(results)

    except SandboxError as e:
        return _report_error(e, saved=None)


def _display_multi_result(result: MultiSourceResult, total: int, quiet: bool, *, error: SandboxError | None) -> None:
    """Display a single result from multi-source execution; `error`: why it has no run."""
    label = truncate_source(result.source)

    click.echo()
    click.echo(click.style(f"━━━ [{result.index + 1}/{total}] {label} ", bold=True) + "━" * max(0, 50 - len(label)))

    if result.run:
        run = result.run
        if run.result.stdout:
            click.echo(run.result.stdout, nl=False)
            if not run.result.stdout.endswith("\n"):
                click.echo()
        if run.result.stderr:
            click.echo(run.result.stderr, nl=False, err=True)
            if not run.result.stderr.endswith("\n"):
                click.echo(err=True)

        if not quiet:
            if run.result.exit_code == 0:
                click.echo(
                    click.style(f"✓ {run.total_ms}ms", fg="green", dim=True),
                    err=True,
                )
            else:
                click.echo(
                    click.style(
                        f"✗ {run.total_ms}ms (exit {run.result.exit_code})",
                        fg="red",
                        dim=True,
                    ),
                    err=True,
                )
    elif error is not None:
        _report_error(error, saved=None)  # its exit code is result.exit_code


# ============================================================================
# CLI Group
# ============================================================================


class _OrderedGroup(click.Group):
    """A command group whose help lists the commands in the order they are defined, not sorted."""

    def list_commands(self, ctx: click.Context) -> list[str]:  # noqa: ARG002 - click's signature
        return list(self.commands)


@click.group(cls=_OrderedGroup, invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="exec-sandbox")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """exec-sandbox: run untrusted code in isolated QEMU microVMs.

    'sbx ARGS...' without a command name is 'sbx run ARGS...'.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ============================================================================
# Run Command
# ============================================================================


# Printed after the options of 'sbx run --help': rules to look up, not to read first.
_RUN_EPILOG = """\
Saved instances (--instance ID, one source only):

\b
  - The first run makes the instance; each later run continues it:
    memory, processes and the files in /home/user.
  - Each run saves after the uploads, the code and the downloads,
    also when the code exits with an error or a download fails. A
    timeout, a stopped VM, or an error of an upload or of the run
    (for example OUTPUT_LIMIT) saves nothing.
  - -l, -p, -m, --network, --allow-domain and --expose are fixed when
    the instance is made. Leave them out to use the saved values; another
    value exits with code 2 before a VM starts.
  - Store: a folder on this host (EXEC_SANDBOX_INSTANCE_STORE_DIR,
    else the user's data folder). Key: EXEC_SANDBOX_INSTANCE_KEYRING,
    else the OS keyring. EXEC_SANDBOX_INSTANCE_STORE=s3 uses the S3
    settings and needs EXEC_SANDBOX_INSTANCE_KEYRING.

Exit codes: the code's own exit code (0-255); 2 for a usage error and for
FIX_INPUT or FIX_CONFIG errors; 124 when the code times out or the VM
does not boot in time; 125 for other errors; 130 after Ctrl+C.
"""


# Options in groups: code, run, VM, network, files, saved instance, output.
@cli.command(name="run", epilog=_RUN_EPILOG, context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("sources", nargs=-1)
@click.option(
    "-c",
    "--code",
    "inline_codes",
    multiple=True,
    help="Code to run (repeatable, in place of SOURCES)",
)
@click.option(
    "-l",
    "--language",
    type=click.Choice(["python", "javascript", "raw"], case_sensitive=False),
    help="Language of the code (default: from the file extension, else python)",
)
@click.option(
    "-t",
    "--timeout",
    default=constants.DEFAULT_TIMEOUT_SECONDS,
    show_default=True,
    help=f"Timeout in seconds (1-{constants.MAX_TIMEOUT_SECONDS})",
)
@click.option("-e", "--env", "env_vars", multiple=True, help="Environment variable KEY=VALUE (repeatable)")
@click.option(
    "-m", "--memory", default=constants.DEFAULT_MEMORY_MB, show_default=True, help="Memory in MB (128 or more)"
)
@click.option("-p", "--package", "packages", multiple=True, help="Package to install (repeatable)")
@click.option("--no-validation", is_flag=True, help="Skip the package allow-list check")
@click.option(
    "--network",
    is_flag=True,
    help="Turn on outbound network. Without --allow-domain: python reaches PyPI only, javascript npm only, "
    "raw any host",
)
@click.option(
    "--allow-domain",
    "allowed_domains",
    multiple=True,
    callback=_domains_param,
    help="Allowed domain and its subdomains (repeatable; needs --network)",
)
@click.option(
    "--expose",
    "expose_ports",
    multiple=True,
    help="Expose port: INTERNAL[:EXTERNAL][/tcp|udp] (repeatable)",
)
@click.option(
    "--upload",
    "upload_files",
    multiple=True,
    help="Upload file LOCAL:GUEST (repeatable)",
)
@click.option(
    "--download",
    "download_files",
    multiple=True,
    help="Download file GUEST:LOCAL, or GUEST to ./<file name> (repeatable)",
)
@click.option(
    "--instance",
    "instance_id",
    metavar="ID",
    callback=_instance_id_param,
    help="Saved instance to continue or make, then save (see below). ID: 1 to 64 of a-z 0-9 . _ - (first a-z or 0-9)",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output as JSON: one object (result, startup, exposed_ports, instance, total_ms), "
    "or an array for several sources",
)
@click.option("-q", "--quiet", is_flag=True, help="Hide progress output")
@click.option(
    "--debug",
    is_flag=True,
    help="Turn on DEBUG logging, and show the kernel and init boot logs on stderr (skips the warm pool and L1)",
)
@click.pass_context
def run_command(
    ctx: click.Context,
    *,
    sources: tuple[str, ...],
    language: str | None,
    inline_codes: tuple[str, ...],
    packages: tuple[str, ...],
    timeout: int,
    memory: int,
    env_vars: tuple[str, ...],
    network: bool,
    allowed_domains: tuple[str, ...],
    expose_ports: tuple[str, ...],
    json_output: bool,
    quiet: bool,
    no_validation: bool,
    upload_files: tuple[str, ...],
    download_files: tuple[str, ...],
    debug: bool,
    instance_id: str | None,
) -> NoReturn:
    """Run code in an isolated VM.

    SOURCES is one or more of these (several sources run at the same time, each in its own VM):

    \b
      - Inline code:  sbx run 'print("hello")'
      - File path:    sbx run script.py
      - Stdin:        echo 'print(1)' | sbx run -

    The file extension (.py, .js, .mjs, .sh) gives the language, else Python.
    -l sets the language of all sources.

    Examples:

    \b
      sbx run -l javascript 'console.log("hi")'             # Set the language
      sbx run -p requests==2.32.5 'import requests; ...'    # With a package
      sbx run --network --allow-domain api.example.com script.py
      sbx run --json 'print("test")' | jq .                 # JSON output
      sbx run -c 'print(1)' -c 'print(2)'                   # Several -c sources
      sbx run --upload ./data.csv:data.csv -c "print(open('data.csv').read())"
      sbx run --instance my-vm -c 'x = 41'                  # Make and save my-vm
      sbx run --instance my-vm -c 'print(x + 1)'            # Continue it: prints 42
    """
    configure_logging(quiet=quiet, level=logging.DEBUG if debug else None)

    all_sources: list[str] = list(inline_codes) + list(sources)

    stdin_count = all_sources.count("-")
    if stdin_count > 1:
        raise click.UsageError("Stdin marker '-' can only be used once.")

    if not all_sources:
        raise click.UsageError("No code given. Give SOURCES, or use the -c option.")

    # Without network access, the domain list has nothing to allow
    if allowed_domains and not network:
        raise click.UsageError("--allow-domain requires --network.")

    # Parse environment variables, port mappings and file I/O specs
    try:
        parsed_env_vars = parse_env_vars(env_vars)
        parsed_ports = parse_port_mappings(expose_ports) if expose_ports else None
        parsed_uploads = tuple(parse_upload_spec(spec) for spec in upload_files)
        parsed_downloads = tuple(parse_download_spec(spec) for spec in download_files)
    except click.BadParameter as exc:
        raise click.UsageError(str(exc)) from exc

    resolved_sources = [_resolve_source(src, language) for src in all_sources]

    # File I/O requires exactly one source
    if (parsed_uploads or parsed_downloads) and len(resolved_sources) != 1:
        raise click.UsageError("--upload/--download requires exactly one source.")
    # Sources run in parallel VMs: their saves of one instance would conflict
    if instance_id is not None and len(resolved_sources) != 1:
        raise click.UsageError("--instance requires exactly one source.")

    if len(resolved_sources) == 1:
        src_input = resolved_sources[0]
        exit_code = _run_with_interrupt_handling(
            run_code(
                code=src_input.code,
                language=src_input.language,
                packages=list(packages),
                timeout=timeout,
                memory=memory,
                env_vars=parsed_env_vars,
                network=network,
                allowed_domains=list(allowed_domains),
                expose_ports=parsed_ports,
                json_output=json_output,
                quiet=quiet,
                no_validation=no_validation,
                upload_files=parsed_uploads,
                download_files=parsed_downloads,
                debug=debug,
                instance_id=instance_id,
                flag_fields=_spec_flags(ctx, source=all_sources[0]) if instance_id is not None else frozenset(),
            )
        )
    else:
        # Multiple sources: use concurrent execution
        exit_code = _run_with_interrupt_handling(
            run_multiple(
                sources=resolved_sources,
                packages=list(packages),
                timeout=timeout,
                memory=memory,
                env_vars=parsed_env_vars,
                network=network,
                allowed_domains=list(allowed_domains),
                expose_ports=parsed_ports,
                json_output=json_output,
                quiet=quiet,
                no_validation=no_validation,
                debug=debug,
            )
        )

    sys.exit(exit_code)


# ============================================================================
# Instance Commands
# ============================================================================


async def _instance_admin(action: Callable[[InstanceAdmin], Awaitable[int]]) -> int:
    """Run `action` on the CLI's instance store; print errors; return the exit code.

    Only the store and the keyring open: no Scheduler, no VM, no image download.
    """
    try:
        settings = _env_settings()
        store, keyring = _instance_store(settings)
        opened = settings.model_copy(update={"instance_store": store, "instance_keyring": keyring})
        async with require_instance_admin(store=store)(opened) as admin:
            return await action(admin)
    except SandboxError as e:
        return _report_error(e, saved=None)


@cli.group(name="instance", cls=_OrderedGroup, context_settings={"help_option_names": ["-h", "--help"]})
def instance_group() -> None:
    """List, describe and remove saved instances.

    These commands start no VM. They use the store of 'sbx run --instance ID':
    a folder on this host (EXEC_SANDBOX_INSTANCE_STORE_DIR, else the user's
    data folder) with its key in EXEC_SANDBOX_INSTANCE_KEYRING, else in the
    OS keyring; or S3 with EXEC_SANDBOX_INSTANCE_STORE=s3 and
    EXEC_SANDBOX_INSTANCE_KEYRING.

    Exit codes: 0 on success; 2 for invalid settings, a bad id, or a FIX_INPUT
    or FIX_CONFIG error; 125 for the other errors; 130 after Ctrl+C.
    """


@instance_group.command(name="list", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "json_output", is_flag=True, help="Output as a JSON array")
def instance_list_command(*, json_output: bool) -> NoReturn:
    """Show the id of each saved instance, one on each line.

    An id can have no current state (a first save that did not finish, or a
    removal that did not finish): 'sbx instance describe' then fails, and
    'sbx instance rm' removes what is left.
    """
    configure_logging()

    async def show(admin: InstanceAdmin) -> int:
        ids = [instance_id async for instance_id in admin.list_ids()]
        if json_output:
            click.echo(_INSTANCE_IDS.dump_json(ids, indent=2).decode())
        else:
            for instance_id in ids:
                click.echo(instance_id)
        return EXIT_SUCCESS

    sys.exit(_run_with_interrupt_handling(_instance_admin(show)))


@instance_group.command(name="describe", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("instance_id", metavar="ID", callback=_instance_id_param)
def instance_describe_command(*, instance_id: str) -> NoReturn:
    """Show what the store holds for one saved instance, as JSON.

    Generation, recent commits and key id; also the VM settings (spec) and the
    host it was saved on when this keyring has the instance's key (verified).
    Without the key, "verified" is false and nothing checks the other fields.
    """
    configure_logging()

    async def show(admin: InstanceAdmin) -> int:
        info = await admin.describe(instance_id)
        click.echo(info.model_dump_json(indent=2))
        return EXIT_SUCCESS

    sys.exit(_run_with_interrupt_handling(_instance_admin(show)))


@instance_group.command(name="rm", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("instance_ids", metavar="ID...", nargs=-1, required=True, callback=_instance_ids_param)
def instance_rm_command(*, instance_ids: tuple[str, ...]) -> NoReturn:
    """Remove saved instances.

    Removes the ids in order and prints each id it removed; the first error
    stops the command. An id with no instance is not an error (the objects
    left under that id are removed too): the exit code is 0 when all the
    instances are gone.
    """
    configure_logging()

    async def remove(admin: InstanceAdmin) -> int:
        for instance_id in instance_ids:
            if await admin.delete(instance_id):
                click.echo(instance_id)
            else:
                click.echo(f"No instance {instance_id}.", err=True)
        return EXIT_SUCCESS

    sys.exit(_run_with_interrupt_handling(_instance_admin(remove)))


# ============================================================================
# Prefetch Command
# ============================================================================


@cli.command(name="prefetch", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--arch",
    type=click.Choice(["x86_64", "aarch64"]),
    default=None,
    help="Target architecture (default: this machine)",
)
@click.option("-q", "--quiet", is_flag=True, help="Hide progress output")
def prefetch_command(arch: str | None, quiet: bool) -> NoReturn:
    """Download the VM assets before use, for offline hosts.

    Downloads the kernel, the initramfs and the base images of one
    architecture to the download cache, except the files already there.
    gvproxy-wrapper (for this OS) goes there only when no images folder
    (EXEC_SANDBOX_IMAGES_DIR, a local build) holds it. Exit code 1 when a
    download fails.

    Dockerfile example:

    \b
      RUN sbx prefetch -q
      ENV EXEC_SANDBOX_OFFLINE=1
    """
    configure_logging(quiet=quiet)

    result: PrefetchResult = asyncio.run(prefetch_all_assets(arch=arch))
    _display_prefetch_result(result, quiet=quiet)
    sys.exit(EXIT_SUCCESS if result.success else 1)


def _display_prefetch_result(result: PrefetchResult, quiet: bool) -> None:
    """Display prefetch result to the user."""
    if quiet and result.success:
        return

    if not quiet:
        click.echo(f"Prefetching exec-sandbox assets for {result.arch}...")
        for name in result.downloaded:
            click.echo(f"  {name}: done")
        for name, _ in result.errors:
            click.echo(f"  {name}: failed", err=True)

    if result.errors:
        click.echo("\nError: Download failed\n", err=True)
        for name, error in result.errors:
            click.echo(f"  Failed to download {name}: {error}", err=True)
        click.echo(
            "\n  Suggestions:\n    - Check your network connection\n    - Set GITHUB_TOKEN for higher rate limits",
            err=True,
        )
    elif not quiet and result.cache_dir:
        click.echo(f"\nDone. Cache: {result.cache_dir}")


# ============================================================================
# Entry Point
# ============================================================================

# Subcommands and flags that should NOT trigger implicit 'run'
_KNOWN_SUBCOMMANDS = {*cli.commands, "--help", "-h", "--version", "-V"}


def main() -> None:
    """Entry point of `sbx`. It exits the process (click's standalone mode), also
    after the help of a bare `sbx`.

    When the first argument is not a known subcommand or flag, 'run' goes
    before it:

        sbx 'print(1)'  ->  sbx run 'print(1)'
    """
    if len(sys.argv) > 1 and sys.argv[1] not in _KNOWN_SUBCOMMANDS:
        sys.argv.insert(1, "run")
    cli()

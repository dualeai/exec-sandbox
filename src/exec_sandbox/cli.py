"""Command-line interface for exec-sandbox.

Usage:
    sbx 'print("hello")'           # Run inline code
    sbx script.py                  # Run file
    echo "print(1)" | sbx -        # Run from stdin
    sbx -l python -p pandas==3.0.1 'import pandas; print(pandas.__version__)'

Multi-input (V2):
    sbx 'print(1)' 'print(2)'      # Multiple inline codes
    sbx script1.py script2.py      # Multiple files
    sbx -j 5 *.py                  # Concurrent with limit
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import termios
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable

import click

from exec_sandbox import (
    ExecutionResult,
    Language,
    PackageNotAllowedError,
    PortMapping,
    SandboxError,
    Scheduler,
    SchedulerConfig,
    VmTimeoutError,
    __version__,
    constants,
)
from exec_sandbox._logging import configure_logging
from exec_sandbox.assets import PrefetchResult, prefetch_all_assets
from exec_sandbox.process_registry import force_kill_all, get_tracked_count


def _create_terminal_monitor(stop_event: threading.Event) -> None:
    """Background thread to keep ISIG enabled so Ctrl+C generates SIGINT."""
    try:
        tty_fd = sys.stdin.fileno()
        while not stop_event.is_set():
            try:
                attrs = termios.tcgetattr(tty_fd)
                lflag = attrs[3]
                if not (lflag & termios.ISIG):
                    # ISIG was disabled - re-enable it
                    attrs[3] = lflag | termios.ISIG
                    termios.tcsetattr(tty_fd, termios.TCSANOW, attrs)
            except (OSError, termios.error):
                pass  # Terminal not available or error reading settings
            stop_event.wait(0.1)  # Check every 100ms
    except (OSError, ValueError):
        pass  # Thread cleanup, stdin closed, or invalid fd


def _run_with_interrupt_handling(coro: Any) -> Any:
    """Run coroutine with proper Ctrl+C handling.

    Does NOT use asyncio.run() because it overrides signal handlers.
    Spawns a background thread to keep terminal ISIG enabled.

    - First Ctrl+C: Cancels main task, cleanup (__aexit__) runs
    - Second Ctrl+C: Force kill all child process groups immediately
    """
    interrupt_count = [0]
    main_task: asyncio.Task[Any] | None = None
    loop: asyncio.AbstractEventLoop | None = None
    stop_monitor = threading.Event()

    def handle_sigint(_signum: int, _frame: Any) -> None:
        interrupt_count[0] += 1
        if interrupt_count[0] > 1:
            count = get_tracked_count()
            if count > 0:
                click.echo(f"\nForce killing {count} VM process(es)...", err=True)
                force_kill_all()
            sys.exit(130)
        click.echo("\nInterrupted. Shutting down (Ctrl+C again to force)...", err=True)
        if main_task is not None and not main_task.done():
            main_task.cancel()
        if loop is not None:
            loop.call_soon_threadsafe(lambda: None)

    original_handler = signal.signal(signal.SIGINT, handle_sigint)
    monitor_thread = threading.Thread(target=_create_terminal_monitor, args=(stop_monitor,), daemon=True)
    monitor_thread.start()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            main_task = loop.create_task(coro)
            return loop.run_until_complete(main_task)
        except asyncio.CancelledError:
            sys.exit(130)
        finally:
            _cleanup_event_loop(loop)
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        stop_monitor.set()
        signal.signal(signal.SIGINT, original_handler)


def _cleanup_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Clean up asyncio event loop, cancelling pending tasks."""
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        loop.close()


# Exit codes following Unix conventions
EXIT_SUCCESS = 0
EXIT_CLI_ERROR = 2
EXIT_TIMEOUT = 124  # Matches `timeout` command
EXIT_SANDBOX_ERROR = 125

# Concurrency limits

# File extension to language mapping
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".sh": "raw",
}


@dataclass
class SourceInput:
    """Represents a single source input for execution."""

    code: str
    label: str  # Display label (file path or truncated code)
    language: Language


def detect_language(source: str | None) -> str | None:
    """Auto-detect language from file extension.

    Args:
        source: File path or stdin marker ("-") or inline code

    Returns:
        Detected language name or None if cannot detect
    """
    if not source or source == "-":
        return None

    # Check if it's a file path
    path = Path(source)
    if path.suffix:
        return EXTENSION_MAP.get(path.suffix.lower())

    return None


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
            # Check for protocol suffix
            protocol = "tcp"
            if "/" in port_spec:
                port_spec, protocol = port_spec.rsplit("/", 1)
                if protocol not in ("tcp", "udp"):
                    raise click.BadParameter(
                        f"Invalid protocol: '{protocol}'. Use 'tcp' or 'udp'.",
                        param_hint="'--expose'",
                    )

            # Parse internal:external or just internal
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

        except ValueError as e:
            if "int()" in str(e) or "invalid literal" in str(e):
                raise click.BadParameter(
                    f"Invalid port number in: '{spec}'. Ports must be integers.",
                    param_hint="'--expose'",
                ) from e
            # Re-raise pydantic validation errors
            raise click.BadParameter(
                f"Invalid port mapping: '{spec}'. {e}",
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
            f"Invalid upload format: '{spec}'. Use LOCAL:GUEST format (e.g. ./data.csv:data.csv).",
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
            f"Empty local path in: '{spec}'. Use GUEST:LOCAL format (e.g. output.csv:./out.csv).",
            param_hint="'--download'",
        )
    # Expand ~ since shell can't do it inside colon-separated args
    # Uses os.path.expanduser (not Path.expanduser) to avoid normalizing ./relative paths
    local = os.path.expanduser(local)  # noqa: PTH111
    if Path(local).is_dir():
        raise click.BadParameter(
            f"Local path is a directory, not a file: '{local}'. Specify a full file path (e.g. ~/results/output.csv).",
            param_hint="'--download'",
        )
    return (guest, local)


def format_error(title: str, message: str, suggestions: list[str] | None = None) -> str:
    """Format an error message following What → Why → Fix pattern.

    Args:
        title: Short error title
        message: Detailed explanation
        suggestions: Optional list of suggestions to fix the issue

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


def format_result_json(run: OneShotRun) -> str:
    """Format a one-shot run as JSON."""
    return run.model_dump_json(indent=2)


def format_multi_result_json(results: list[MultiSourceResult]) -> str:
    """Format multiple execution results as a JSON array."""
    return _MULTI_RESULTS.dump_json(results, indent=2).decode()


_MULTI_RESULTS = TypeAdapter(list[MultiSourceResult])


def truncate_source(source: str, max_len: int = 40) -> str:
    """Truncate source label for display."""
    if len(source) <= max_len:
        return source
    return source[: max_len - 3] + "..."


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
            raise click.UsageError("No input provided. Pipe code to stdin or use -c flag.")
        code = sys.stdin.read()
        label = "<stdin>"
    else:
        # Check if it's a file, otherwise treat as inline code
        path = Path(src)
        if path.exists() and path.is_file():
            code = path.read_text()
            label = src
        else:
            code = src
            label = truncate_source(src)

    if not code.strip():
        raise click.UsageError(f"Empty code provided for source: {label}")

    # Resolve language (override > auto-detect > python default)
    resolved_language = language_override.lower() if language_override else detect_language(src) or "python"

    # Convert to Language enum
    try:
        lang_enum = Language(resolved_language)
    except ValueError as exc:
        raise click.UsageError(f"Unknown language: {resolved_language}") from exc

    return SourceInput(code=code, label=label, language=lang_enum)


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


async def run_code(
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
    upload_files: tuple[tuple[str, str], ...] = (),
    download_files: tuple[tuple[str, str], ...] = (),
    debug: bool = False,
) -> int:
    """Execute code in sandbox and return exit code.

    When upload_files or download_files are provided, uses a session to keep the
    VM alive for file I/O. Otherwise, uses scheduler.run() for full timing breakdown.

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

    Returns:
        Exit code to return from CLI
    """
    config = SchedulerConfig(
        default_timeout_seconds=timeout,
        default_memory_mb=memory,
        enable_package_validation=not no_validation,
    )

    # Streaming callbacks for non-JSON output
    def on_stdout(chunk: str) -> None:
        if not json_output:
            click.echo(chunk, nl=False)

    def on_stderr(chunk: str) -> None:
        if not json_output:
            click.echo(chunk, nl=False, err=True)

    on_boot_log = _make_boot_log_callback(debug)

    try:
        async with Scheduler(config) as scheduler:
            if upload_files or download_files:
                # Session path: keeps VM alive for file I/O
                async with await scheduler.session(
                    language=language,
                    packages=list(packages) if packages else None,
                    memory_mb=memory,
                    allow_network=network,
                    allowed_domains=list(allowed_domains) if allowed_domains else None,
                    expose_ports=expose_ports,  # type: ignore[arg-type]
                    on_boot_log=on_boot_log,
                ) as session:
                    # Upload files
                    for local_path, guest_path in upload_files:
                        await session.write_file(guest_path, Path(local_path))
                        if not quiet:
                            click.echo(
                                click.style(f"  Uploaded: {local_path} -> {guest_path}", dim=True),
                                err=True,
                            )

                    # Execute code
                    result = await session.exec(
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

        # JSON output mode
        if json_output:
            click.echo(format_result_json(run))
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

    Returns:
        Maximum exit code from all sources (or EXIT_SANDBOX_ERROR on infrastructure failure)
    """
    config = SchedulerConfig(
        default_timeout_seconds=timeout,
        default_memory_mb=memory,
        enable_package_validation=not no_validation,
    )

    on_boot_log = _make_boot_log_callback(debug)

    results: list[MultiSourceResult] = []
    total_passed = 0
    total_failed = 0
    start_time = time.perf_counter()

    try:
        async with Scheduler(config) as scheduler:
            # Create tasks for all sources
            async def run_one(idx: int, src: SourceInput) -> MultiSourceResult:
                try:
                    result = await scheduler.run(
                        code=src.code,
                        language=src.language,
                        packages=list(packages) if packages else None,
                        timeout_seconds=timeout,
                        memory_mb=memory,
                        allow_network=network,
                        allowed_domains=list(allowed_domains) if allowed_domains else None,
                        expose_ports=expose_ports,  # type: ignore[arg-type]  # list invariance
                        env_vars=env_vars if env_vars else None,
                        on_boot_log=on_boot_log,
                    )
                    return MultiSourceResult(
                        index=idx,
                        source=src.label,
                        result=result,
                        error=None,
                    )
                except VmTimeoutError:
                    return MultiSourceResult(
                        index=idx,
                        source=src.label,
                        result=None,
                        error="Execution timed out",
                    )
                except SandboxError as e:
                    return MultiSourceResult(
                        index=idx,
                        source=src.label,
                        result=None,
                        error=str(e.message),
                    )

            tasks = [run_one(i, src) for i, src in enumerate(sources)]

            # Stream results as they complete
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)

                # Update counters
                if result.run and result.run.result.exit_code == 0:
                    total_passed += 1
                else:
                    total_failed += 1

                # Display result if not JSON mode
                if not json_output:
                    _display_multi_result(result, len(sources), quiet, error=errors.get(result.index))

        total_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Sort results by index for consistent output
        results.sort(key=lambda r: r.index)

        # Final output
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

    # Header
    click.echo()
    click.echo(click.style(f"━━━ [{result.index + 1}/{total}] {label} ", bold=True) + "━" * max(0, 50 - len(label)))

    if result.run:
        run = result.run
        # Show stdout/stderr
        if run.result.stdout:
            click.echo(run.result.stdout, nl=False)
            if not run.result.stdout.endswith("\n"):
                click.echo()
        if run.result.stderr:
            click.echo(run.result.stderr, nl=False, err=True)
            if not run.result.stderr.endswith("\n"):
                click.echo(err=True)

        # Show status
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


@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="exec-sandbox")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """exec-sandbox: Execute code in isolated VM sandboxes.

    Run 'sbx run --help' or 'sbx prefetch --help' for subcommand details.
    """
    # If no subcommand provided and no arguments, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ============================================================================
# Run Command
# ============================================================================


@cli.command(name="run", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("sources", nargs=-1)
@click.option(
    "-l",
    "--language",
    type=click.Choice(["python", "javascript", "raw"], case_sensitive=False),
    help="Programming language (auto-detected from file extension)",
)
@click.option(
    "-c",
    "--code",
    "inline_codes",
    multiple=True,
    help="Code to execute (repeatable, alternative to SOURCES)",
)
@click.option("-p", "--package", "packages", multiple=True, help="Package to install (repeatable)")
@click.option("-t", "--timeout", default=30, show_default=True, help="Timeout in seconds")
@click.option("-m", "--memory", default=constants.DEFAULT_MEMORY_MB, show_default=True, help="Memory in MB")
@click.option("-e", "--env", "env_vars", multiple=True, help="Environment variable KEY=VALUE (repeatable)")
@click.option("--network", is_flag=True, help="Enable network access")
@click.option("--allow-domain", "allowed_domains", multiple=True, help="Allowed domain (repeatable)")
@click.option(
    "--expose",
    "expose_ports",
    multiple=True,
    help="Expose port: INTERNAL or INTERNAL:EXTERNAL or INTERNAL:EXTERNAL/PROTOCOL (repeatable)",
)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output")
@click.option("--no-validation", is_flag=True, help="Skip package validation")
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
    help="Download file GUEST:LOCAL or GUEST (repeatable)",
)
@click.option("--debug", is_flag=True, help="Enable DEBUG logging and stream kernel/init boot logs to stderr")
def run_command(
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
) -> NoReturn:
    """Execute code in an isolated VM sandbox.

    SOURCES can be one or more of:

    \b
      - Inline code:  sbx run 'print("hello")'
      - File path:    sbx run script.py
      - Stdin:        echo 'print(1)' | sbx run -

    Multiple sources run concurrently:

    \b
      sbx run 'print(1)' 'print(2)' script.py    # Run 3 sources

    Language is auto-detected from file extension (.py, .js, .sh)
    or defaults to Python. Use -l to override for all sources.

    File I/O uses colon-separated paths (LOCAL:GUEST / GUEST:LOCAL):

    \b
      --upload LOCAL:GUEST      Upload local file to sandbox
      --download GUEST:LOCAL    Download sandbox file to local path
      --download GUEST          Download to ./basename(GUEST)

    Examples:

    \b
      sbx run 'print("hello")'                              # Simple Python
      sbx run -l javascript 'console.log("hi")'             # Explicit language
      sbx run script.py                                     # Run file
      sbx run -p requests==2.32.5 'import requests; ...'    # With package
      sbx run --network --allow-domain api.example.com script.py
      echo 'print(42)' | sbx run -                          # From stdin
      sbx run --json 'print("test")' | jq .                 # JSON output
      sbx run -c 'print(1)' -c 'print(2)'                   # Multiple via -c flag
      sbx run --upload ./data.csv:data.csv -c "print(open('data.csv').read())"
      sbx run --download output.csv:./out.csv -c "open('output.csv','w').write('data')"
      sbx run --download output.csv -c "open('output.csv','w').write('data')"
    """
    configure_logging(quiet=quiet, level=logging.DEBUG if debug else None)

    # Merge inline codes with positional sources
    all_sources: list[str] = list(inline_codes) + list(sources)

    # Handle stdin marker
    stdin_count = all_sources.count("-")
    if stdin_count > 1:
        raise click.UsageError("Stdin marker '-' can only be used once.")

    if not all_sources:
        raise click.UsageError("No code provided. Provide SOURCES argument or use -c flag.")

    # Parse environment variables
    try:
        parsed_env_vars = parse_env_vars(env_vars)
    except click.BadParameter as exc:
        raise click.UsageError(str(exc)) from exc

    # Parse port mappings
    try:
        parsed_ports = parse_port_mappings(expose_ports) if expose_ports else None
    except click.BadParameter as exc:
        raise click.UsageError(str(exc)) from exc

    # Parse file I/O specs
    try:
        parsed_uploads = tuple(parse_upload_spec(spec) for spec in upload_files)
    except click.BadParameter as exc:
        raise click.UsageError(str(exc)) from exc
    try:
        parsed_downloads = tuple(parse_download_spec(spec) for spec in download_files)
    except click.BadParameter as exc:
        raise click.UsageError(str(exc)) from exc

    # Build list of SourceInput objects
    resolved_sources = [_resolve_source(src, language) for src in all_sources]

    # File I/O requires exactly one source
    if (parsed_uploads or parsed_downloads) and len(resolved_sources) != 1:
        raise click.UsageError("--upload/--download requires exactly one source.")

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
# Prefetch Command
# ============================================================================


@cli.command(name="prefetch", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--arch",
    type=click.Choice(["x86_64", "aarch64"]),
    default=None,
    help="Target architecture (default: current machine)",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output")
def prefetch_command(arch: str | None, quiet: bool) -> NoReturn:
    """Pre-download VM assets for offline use.

    Downloads all required assets (kernel, initramfs, gvproxy, base images)
    to the cache directory. Useful for container builds before running with
    EXEC_SANDBOX_OFFLINE=1.

    Examples:

    \b
      sbx prefetch                    # Download all assets for current arch
      sbx prefetch --arch aarch64     # Cross-arch prefetch for multi-arch Docker
      sbx prefetch -q                 # Quiet mode (CI/Docker)

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
# Entry Point with Backward Compatibility
# ============================================================================

# Known subcommands and flags that should NOT trigger implicit 'run'
_KNOWN_SUBCOMMANDS = {"prefetch", "run", "--help", "-h", "--version", "-V"}


def main() -> NoReturn:
    """Entry point with backward compatibility.

    For backward compatibility, if the first argument is not a known subcommand
    or flag, we automatically insert 'run' so that:

        sbx 'print(1)'  ->  sbx run 'print(1)'

    This preserves the original CLI behavior while supporting new subcommands.
    """
    if len(sys.argv) > 1 and sys.argv[1] not in _KNOWN_SUBCOMMANDS:
        # Insert 'run' as the subcommand for backward compatibility
        sys.argv.insert(1, "run")
    cli()
    # cli() will sys.exit() via subcommands, but for bare 'sbx' (help display),
    # we need to exit here
    sys.exit(0)


if __name__ == "__main__":
    main()

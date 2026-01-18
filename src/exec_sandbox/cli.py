"""Command-line interface for exec-sandbox.

Usage:
    sbx 'print("hello")'           # Run inline code
    sbx script.py                  # Run file
    echo "print(1)" | sbx -        # Run from stdin
    sbx -l python -p pandas 'import pandas; print(pandas.__version__)'

Multi-input (V2):
    sbx 'print(1)' 'print(2)'      # Multiple inline codes
    sbx script1.py script2.py      # Multiple files
    sbx -j 5 *.py                  # Concurrent with limit
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import click

from exec_sandbox import (
    ExecutionResult,
    Language,
    PackageNotAllowedError,
    SandboxError,
    Scheduler,
    SchedulerConfig,
    VmTimeoutError,
    __version__,
)

# Exit codes following Unix conventions
EXIT_SUCCESS = 0
EXIT_CLI_ERROR = 2
EXIT_TIMEOUT = 124  # Matches `timeout` command
EXIT_SANDBOX_ERROR = 125

# Concurrency limits
DEFAULT_CONCURRENCY = 10
MAX_CONCURRENCY = 100

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


def format_result_json(result: ExecutionResult) -> str:
    """Format execution result as JSON.

    Args:
        result: Execution result from scheduler

    Returns:
        JSON string
    """
    return json.dumps(_result_to_dict(result), indent=2)


def _result_to_dict(result: ExecutionResult) -> dict[str, Any]:
    """Convert ExecutionResult to dictionary for JSON serialization."""
    output: dict[str, Any] = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "execution_time_ms": result.execution_time_ms,
        "timing": {
            "total_ms": result.timing.total_ms,
            "boot_ms": result.timing.boot_ms,
            "execute_ms": result.timing.execute_ms,
            "setup_ms": result.timing.setup_ms,
        },
        "warm_pool_hit": result.warm_pool_hit,
    }

    if result.external_memory_peak_mb is not None:
        output["memory_peak_mb"] = result.external_memory_peak_mb

    return output


@dataclass
class MultiSourceResult:
    """Result from a single source in multi-input execution."""

    index: int
    source: str  # Label for display
    result: ExecutionResult | None
    error: str | None  # Error message if execution failed


def format_multi_result_json(results: list[MultiSourceResult]) -> str:
    """Format multiple execution results as JSON array.

    Args:
        results: List of multi-source results

    Returns:
        JSON string with array of results
    """
    output: list[dict[str, Any]] = []
    for r in results:
        if r.result:
            item: dict[str, Any] = {
                "index": r.index,
                "source": r.source,
                **_result_to_dict(r.result),
            }
        else:
            item = {
                "index": r.index,
                "source": r.source,
                "error": r.error,
                "exit_code": EXIT_SANDBOX_ERROR,
            }
        output.append(item)
    return json.dumps(output, indent=2)


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
    """Compute the exit code for multi-source execution.

    Returns max exit code, or EXIT_TIMEOUT if any source timed out.
    """
    # Check for timeouts first
    if any(r.error == "Execution timed out" for r in results):
        return EXIT_TIMEOUT

    # Return max exit code from all results
    exit_codes = [r.result.exit_code if r.result else EXIT_SANDBOX_ERROR for r in results]
    return max(exit_codes) if exit_codes else EXIT_SUCCESS


def is_tty() -> bool:
    """Check if stdout is connected to a terminal."""
    return sys.stdout.isatty()


async def run_code(
    code: str,
    language: Language,
    packages: list[str],
    timeout: int,
    memory: int,
    env_vars: dict[str, str],
    network: bool,
    allowed_domains: list[str],
    json_output: bool,
    quiet: bool,
    no_validation: bool,
) -> int:
    """Execute code in sandbox and return exit code.

    Args:
        code: Code to execute
        language: Programming language
        packages: Packages to install
        timeout: Timeout in seconds
        memory: Memory in MB
        env_vars: Environment variables
        network: Enable network
        allowed_domains: Allowed domains for network
        json_output: Output as JSON
        quiet: Suppress progress output
        no_validation: Skip package validation

    Returns:
        Exit code to return from CLI
    """
    config = SchedulerConfig(
        default_timeout_seconds=timeout,
        default_memory_mb=memory,
        enable_package_validation=not no_validation,
        max_concurrent_vms=1,  # CLI runs single VM
    )

    # Streaming callbacks for non-JSON output
    def on_stdout(chunk: str) -> None:
        if not json_output:
            click.echo(chunk, nl=False)

    def on_stderr(chunk: str) -> None:
        if not json_output:
            click.echo(chunk, nl=False, err=True)

    try:
        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=language,
                packages=list(packages) if packages else None,
                timeout_seconds=timeout,
                memory_mb=memory,
                allow_network=network,
                allowed_domains=list(allowed_domains) if allowed_domains else None,
                env_vars=env_vars if env_vars else None,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )

        # JSON output mode
        if json_output:
            click.echo(format_result_json(result))
            return result.exit_code

        # TTY mode: show timing footer
        if is_tty() and not quiet:
            click.echo()  # Ensure newline after output
            click.echo(
                click.style(f"✓ Done in {result.timing.total_ms}ms", fg="green", dim=True),
                err=True,
            )

        return result.exit_code

    except PackageNotAllowedError as e:
        error_msg = format_error(
            f"Package not allowed: {e.message}",
            "Only packages from the top 10,000 PyPI/npm packages are allowed for security reasons.",
            [
                "Check spelling of package name",
                "Use --no-validation to bypass (not recommended)",
            ],
        )
        click.echo(error_msg, err=True)
        return EXIT_CLI_ERROR

    except VmTimeoutError:
        error_msg = format_error(
            "Execution timed out",
            f"The code did not complete within {timeout} seconds.",
            [
                "Increase timeout with -t/--timeout",
                "Check for infinite loops in your code",
            ],
        )
        click.echo(error_msg, err=True)
        return EXIT_TIMEOUT

    except SandboxError as e:
        error_msg = format_error(
            "Sandbox error",
            str(e.message),
            [
                "Check that QEMU is installed: brew install qemu",
                "Ensure VM images are available",
            ],
        )
        click.echo(error_msg, err=True)
        return EXIT_SANDBOX_ERROR


async def run_multiple(
    sources: list[SourceInput],
    packages: list[str],
    timeout: int,
    memory: int,
    env_vars: dict[str, str],
    network: bool,
    allowed_domains: list[str],
    json_output: bool,
    quiet: bool,
    no_validation: bool,
    concurrency: int,
) -> int:
    """Execute multiple sources concurrently and return max exit code.

    Args:
        sources: List of SourceInput objects to execute
        packages: Packages to install (applied to all sources)
        timeout: Timeout in seconds per source
        memory: Memory in MB per VM
        env_vars: Environment variables
        network: Enable network
        allowed_domains: Allowed domains for network
        json_output: Output as JSON
        quiet: Suppress progress output
        no_validation: Skip package validation
        concurrency: Maximum concurrent VMs

    Returns:
        Maximum exit code from all sources (or EXIT_SANDBOX_ERROR on infrastructure failure)
    """
    config = SchedulerConfig(
        default_timeout_seconds=timeout,
        default_memory_mb=memory,
        enable_package_validation=not no_validation,
        max_concurrent_vms=min(len(sources), concurrency),
    )

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
                        env_vars=env_vars if env_vars else None,
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
                if result.result and result.result.exit_code == 0:
                    total_passed += 1
                else:
                    total_failed += 1

                # Display result if not JSON mode
                if not json_output:
                    _display_multi_result(result, len(sources), quiet)

        total_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Sort results by index for consistent output
        results.sort(key=lambda r: r.index)

        # Final output
        if json_output:
            click.echo(format_multi_result_json(results))
        elif is_tty() and not quiet:
            _display_multi_summary(total_passed, total_failed, len(sources), total_time_ms)

        return _compute_multi_exit_code(results)

    except PackageNotAllowedError as e:
        error_msg = format_error(
            f"Package not allowed: {e.message}",
            "Only packages from the top 10,000 PyPI/npm packages are allowed for security reasons.",
            [
                "Check spelling of package name",
                "Use --no-validation to bypass (not recommended)",
            ],
        )
        click.echo(error_msg, err=True)
        return EXIT_CLI_ERROR

    except SandboxError as e:
        error_msg = format_error(
            "Sandbox error",
            str(e.message),
            [
                "Check that QEMU is installed: brew install qemu",
                "Ensure VM images are available",
            ],
        )
        click.echo(error_msg, err=True)
        return EXIT_SANDBOX_ERROR


def _display_multi_result(result: MultiSourceResult, total: int, quiet: bool) -> None:
    """Display a single result from multi-source execution."""
    label = truncate_source(result.source)

    # Header
    click.echo()
    click.echo(click.style(f"━━━ [{result.index + 1}/{total}] {label} ", bold=True) + "━" * max(0, 50 - len(label)))

    if result.result:
        # Show stdout/stderr
        if result.result.stdout:
            click.echo(result.result.stdout, nl=False)
            if not result.result.stdout.endswith("\n"):
                click.echo()
        if result.result.stderr:
            click.echo(result.result.stderr, nl=False, err=True)
            if not result.result.stderr.endswith("\n"):
                click.echo(err=True)

        # Show status
        if not quiet:
            if result.result.exit_code == 0:
                click.echo(
                    click.style(f"✓ {result.result.timing.total_ms}ms", fg="green", dim=True),
                    err=True,
                )
            else:
                click.echo(
                    click.style(
                        f"✗ {result.result.timing.total_ms}ms (exit {result.result.exit_code})",
                        fg="red",
                        dim=True,
                    ),
                    err=True,
                )
    else:
        # Show error
        click.echo(click.style(f"✗ {result.error}", fg="red"), err=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
@click.option("-m", "--memory", default=256, show_default=True, help="Memory in MB")
@click.option("-e", "--env", "env_vars", multiple=True, help="Environment variable KEY=VALUE (repeatable)")
@click.option("--network", is_flag=True, help="Enable network access")
@click.option("--allow-domain", "allowed_domains", multiple=True, help="Allowed domain (repeatable)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output")
@click.option("--no-validation", is_flag=True, help="Skip package validation")
@click.option(
    "-j",
    "--concurrency",
    default=DEFAULT_CONCURRENCY,
    type=click.IntRange(1, MAX_CONCURRENCY),
    show_default=True,
    help="Maximum concurrent VMs for multi-input",
)
@click.version_option(__version__, "-V", "--version", prog_name="exec-sandbox")
def main(
    sources: tuple[str, ...],
    language: str | None,
    inline_codes: tuple[str, ...],
    packages: tuple[str, ...],
    timeout: int,
    memory: int,
    env_vars: tuple[str, ...],
    network: bool,
    allowed_domains: tuple[str, ...],
    json_output: bool,
    quiet: bool,
    no_validation: bool,
    concurrency: int,
) -> NoReturn:
    """Execute code in an isolated VM sandbox.

    SOURCES can be one or more of:

    \b
      - Inline code:  sbx 'print("hello")'
      - File path:    sbx script.py
      - Stdin:        echo 'print(1)' | sbx -

    Multiple sources run concurrently (use -j to limit concurrency):

    \b
      sbx 'print(1)' 'print(2)' script.py    # Run 3 sources
      sbx -j 5 *.py                          # Max 5 concurrent VMs

    Language is auto-detected from file extension (.py, .js, .sh)
    or defaults to Python. Use -l to override for all sources.

    Examples:

    \b
      sbx 'print("hello")'                    # Simple Python
      sbx -l javascript 'console.log("hi")'   # Explicit language
      sbx script.py                           # Run file
      sbx -p requests 'import requests; ...'  # With package
      sbx --network --allow-domain api.example.com script.py
      echo 'print(42)' | sbx -                # From stdin
      sbx --json 'print("test")' | jq .       # JSON output
      sbx -c 'print(1)' -c 'print(2)'         # Multiple via -c flag
    """
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

    # Build list of SourceInput objects
    resolved_sources = [_resolve_source(src, language) for src in all_sources]

    # Single source: use original streaming behavior
    if len(resolved_sources) == 1:
        src_input = resolved_sources[0]
        exit_code = asyncio.run(
            run_code(
                code=src_input.code,
                language=src_input.language,
                packages=list(packages),
                timeout=timeout,
                memory=memory,
                env_vars=parsed_env_vars,
                network=network,
                allowed_domains=list(allowed_domains),
                json_output=json_output,
                quiet=quiet,
                no_validation=no_validation,
            )
        )
    else:
        # Multiple sources: use concurrent execution
        exit_code = asyncio.run(
            run_multiple(
                sources=resolved_sources,
                packages=list(packages),
                timeout=timeout,
                memory=memory,
                env_vars=parsed_env_vars,
                network=network,
                allowed_domains=list(allowed_domains),
                json_output=json_output,
                quiet=quiet,
                no_validation=no_validation,
                concurrency=concurrency,
            )
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

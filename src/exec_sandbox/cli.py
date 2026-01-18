"""Command-line interface for exec-sandbox.

Usage:
    sbx 'print("hello")'           # Run inline code
    sbx script.py                  # Run file
    echo "print(1)" | sbx -        # Run from stdin
    sbx -l python -p pandas 'import pandas; print(pandas.__version__)'
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import NoReturn

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

# File extension to language mapping
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".sh": "raw",
}


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
    output = {
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

    return json.dumps(output, indent=2)


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
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def on_stdout(chunk: str) -> None:
        stdout_chunks.append(chunk)
        if not json_output:
            click.echo(chunk, nl=False)

    def on_stderr(chunk: str) -> None:
        stderr_chunks.append(chunk)
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("source", required=False)
@click.option(
    "-l",
    "--language",
    type=click.Choice(["python", "javascript", "raw"], case_sensitive=False),
    help="Programming language (auto-detected from file extension)",
)
@click.option("-c", "--code", "inline_code", help="Code to execute (alternative to SOURCE)")
@click.option("-p", "--package", "packages", multiple=True, help="Package to install (repeatable)")
@click.option("-t", "--timeout", default=30, show_default=True, help="Timeout in seconds")
@click.option("-m", "--memory", default=256, show_default=True, help="Memory in MB")
@click.option("-e", "--env", "env_vars", multiple=True, help="Environment variable (KEY=VALUE)")
@click.option("--network", is_flag=True, help="Enable network access")
@click.option("--allow-domain", "allowed_domains", multiple=True, help="Allowed domain (repeatable)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output")
@click.option("--no-validation", is_flag=True, help="Skip package validation")
@click.version_option(__version__, "-V", "--version", prog_name="exec-sandbox")
def main(
    source: str | None,
    language: str | None,
    inline_code: str | None,
    packages: tuple[str, ...],
    timeout: int,
    memory: int,
    env_vars: tuple[str, ...],
    network: bool,
    allowed_domains: tuple[str, ...],
    json_output: bool,
    quiet: bool,
    no_validation: bool,
) -> NoReturn:
    """Execute code in an isolated VM sandbox.

    SOURCE can be:

    \b
      - Inline code:  sbx 'print("hello")'
      - File path:    sbx script.py
      - Stdin:        echo 'print(1)' | sbx -

    Language is auto-detected from file extension (.py, .js, .sh)
    or defaults to Python for inline code.

    Examples:

    \b
      sbx 'print("hello")'                    # Simple Python
      sbx -l javascript 'console.log("hi")'   # Explicit language
      sbx script.py                           # Run file
      sbx -p requests 'import requests; ...'  # With package
      sbx --network --allow-domain api.example.com script.py
      echo 'print(42)' | sbx -                # From stdin
      sbx --json 'print("test")' | jq .       # JSON output
    """
    # Resolve code source
    code: str

    if inline_code:
        # -c/--code takes precedence
        code = inline_code
    elif source == "-":
        # Read from stdin
        if sys.stdin.isatty():
            raise click.UsageError("No input provided. Pipe code to stdin or use -c flag.")
        code = sys.stdin.read()
    elif source:
        # Check if it's a file, otherwise treat as inline code
        path = Path(source)
        code = path.read_text() if path.exists() and path.is_file() else source
    else:
        raise click.UsageError("No code provided. Provide SOURCE argument or use -c flag.")

    if not code.strip():
        raise click.UsageError("Empty code provided.")

    # Resolve language (auto-detect from extension, default to python)
    resolved_language = language.lower() if language else (detect_language(source) or "python")

    # Convert to Language enum
    try:
        lang_enum = Language(resolved_language)
    except ValueError as exc:
        raise click.UsageError(f"Unknown language: {resolved_language}") from exc

    # Parse environment variables
    try:
        parsed_env_vars = parse_env_vars(env_vars)
    except click.BadParameter as exc:
        raise click.UsageError(str(exc)) from exc

    # Run async code
    exit_code = asyncio.run(
        run_code(
            code=code,
            language=lang_enum,
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

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

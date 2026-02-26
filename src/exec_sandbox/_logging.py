"""Centralized logging for exec-sandbox.

Library logging best practices (Python docs, PEP 282):
- Attach NullHandler to library root logger
- Never add other handlers -- that's the application's job
- Support EXEC_SANDBOX_LOG_LEVEL env var for level control
- Provide configure_logging() for CLI entry points

CLI output format (httpx convention):
    WARNING [2026-02-25 10:02:54] exec_sandbox.cgroup - message

References:
- https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
- https://www.python-httpx.org/logging/
"""

import logging
import os

import click

LIBRARY_LOGGER_NAME: str = "exec_sandbox"

# NullHandler per Python library best practice -- prevents
# "No handler found" warnings when consumers don't configure logging
logging.getLogger(LIBRARY_LOGGER_NAME).addHandler(logging.NullHandler())

# Honor EXEC_SANDBOX_LOG_LEVEL env var (e.g. "DEBUG", "WARNING", "ERROR")
_env_level = os.environ.get("EXEC_SANDBOX_LOG_LEVEL", "").strip().upper()
_env_level_value = logging.getLevelNamesMapping().get(_env_level)
if _env_level_value:  # excludes NOTSET (0) and missing keys (None)
    logging.getLogger(LIBRARY_LOGGER_NAME).setLevel(_env_level_value)

_FMT = "%(levelname)s [%(asctime)s] %(name)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _ClickHandler(logging.Handler):
    """Handler that emits via click.echo(err=True) with dim styling.

    Using click.echo() instead of StreamHandler.write() ensures ANSI codes
    are automatically stripped when stderr is not a TTY (e.g. redirected to
    a file or consumed by a parent process).
    """

    def __init__(self) -> None:
        super().__init__()
        self.formatter = logging.Formatter(fmt=_FMT, datefmt=_DATEFMT)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            click.echo(click.style(msg, dim=True), err=True)
        except Exception:  # noqa: BLE001
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    All exec_sandbox modules should use this instead of logging.getLogger()
    directly for consistent logger hierarchy.
    """
    return logging.getLogger(name)


def configure_logging(
    *,
    level: int | str | None = None,
    quiet: bool = False,
) -> None:
    """Configure library logging for CLI / application entry points.

    Adds a _ClickHandler if none exists (idempotent), then sets the
    log level.  Library consumers who configure their own handlers are
    unaffected -- the guard ensures we never stack duplicate handlers.

    Args:
        level: Log level (e.g. logging.DEBUG, "WARNING"). Overrides env var.
        quiet: If True, set level to ERROR (suppress WARNING/INFO).
               Takes precedence over level.
    """
    lib_logger = logging.getLogger(LIBRARY_LOGGER_NAME)

    # Add a click handler if the only handler is the library NullHandler.
    # This makes EXEC_SANDBOX_LOG_LEVEL work out of the box for CLI users
    # while staying invisible to library consumers who set up their own handlers.
    if not any(isinstance(h, _ClickHandler) for h in lib_logger.handlers):
        lib_logger.addHandler(_ClickHandler())

    if quiet:
        lib_logger.setLevel(logging.ERROR)
    elif level is not None:
        # setLevel() accepts both int and str; for str it validates
        # internally and raises ValueError on unknown names
        lib_logger.setLevel(level)

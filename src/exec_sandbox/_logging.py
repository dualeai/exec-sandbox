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

LIBRARY_LOGGER_NAME: str = "exec_sandbox"

# NullHandler per Python library best practice -- prevents
# "No handler found" warnings when consumers don't configure logging
logging.getLogger(LIBRARY_LOGGER_NAME).addHandler(logging.NullHandler())

# Honor EXEC_SANDBOX_LOG_LEVEL env var (e.g. "DEBUG", "WARNING", "ERROR")
_env_level = os.environ.get("EXEC_SANDBOX_LOG_LEVEL", "").strip().upper()
_env_level_value = logging.getLevelNamesMapping().get(_env_level)
if _env_level_value:  # excludes NOTSET (0) and missing keys (None)
    logging.getLogger(LIBRARY_LOGGER_NAME).setLevel(_env_level_value)


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

    Adds a stderr StreamHandler if none exists (idempotent), then sets the
    log level.  Library consumers who configure their own handlers are
    unaffected -- the guard ensures we never stack duplicate handlers.

    Args:
        level: Log level (e.g. logging.DEBUG, "WARNING"). Overrides env var.
        quiet: If True, set level to ERROR (suppress WARNING/INFO).
               Takes precedence over level.
    """
    lib_logger = logging.getLogger(LIBRARY_LOGGER_NAME)

    # Add a stderr handler if the only handler is the library NullHandler.
    # This makes EXEC_SANDBOX_LOG_LEVEL work out of the box for CLI users
    # while staying invisible to library consumers who set up their own handlers.
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler) for h in lib_logger.handlers
    ):
        handler = logging.StreamHandler()  # stderr by default
        handler.setFormatter(
            logging.Formatter(
                fmt="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        lib_logger.addHandler(handler)

    if quiet:
        lib_logger.setLevel(logging.ERROR)
    elif level is not None:
        # setLevel() accepts both int and str; for str it validates
        # internally and raises ValueError on unknown names
        lib_logger.setLevel(level)

"""Centralized logging for exec-sandbox.

Library logging best practices (Python docs, PEP 282):
- Attach NullHandler to library root logger
- Never add other handlers -- that's the application's job
- Support EXEC_SANDBOX_LOG_LEVEL env var for level control
- Provide configure_logging() for CLI entry points

CLI output format (httpx convention):
    WARNING [2026-02-25 10:02:54] exec_sandbox.cgroup - message

Non-blocking logging:
    Uses QueueHandler + QueueListener (stdlib) to decouple log emission
    from stderr I/O.  A bounded FIFO queue absorbs bursts; a daemon
    thread drains records to click.echo(err=True).  When the queue is
    full or stderr saturated, records are silently dropped instead of
    raising BlockingIOError on the caller's coroutine.

References:
- https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
- https://docs.python.org/3/library/logging.handlers.html#queuehandler
- https://www.python-httpx.org/logging/
"""

import contextlib
import logging
import logging.handlers
import os
import queue

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

# Bounded queue capacity -- large enough to absorb bursts from
# concurrent VMs, small enough to bound memory under sustained load.
_QUEUE_CAPACITY = 4096


class _ClickHandler(logging.Handler):
    """Target handler: writes to stderr via click.echo with dim styling.

    Runs on the QueueListener's daemon thread, never on the caller's
    thread/coroutine.  Using click.echo() ensures ANSI codes are
    automatically stripped when stderr is not a TTY.
    """

    def __init__(self) -> None:
        super().__init__()
        self.formatter = logging.Formatter(fmt=_FMT, datefmt=_DATEFMT)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            click.echo(click.style(msg, dim=True), err=True)
        except BlockingIOError:
            pass  # Stderr buffer full -- silently drop
        except Exception:  # noqa: BLE001
            self.handleError(record)


class _NonBlockingHandler(logging.handlers.QueueHandler):
    """Queue-backed handler that never blocks the caller.

    Records are enqueued via put_nowait() into a bounded FIFO.  A
    QueueListener daemon thread drains them to _ClickHandler.  When
    the queue is full, records are silently dropped (backpressure).
    """

    def __init__(self) -> None:
        q: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=_QUEUE_CAPACITY)
        super().__init__(q)
        self._listener = logging.handlers.QueueListener(q, _ClickHandler(), respect_handler_level=False)
        self._listener.start()

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """Skip serialization -- same-process queue, no pickle needed."""
        return record

    def enqueue(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(queue.Full):
            self.queue.put_nowait(record)

    def close(self) -> None:
        self._listener.stop()
        super().close()


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

    Adds a _NonBlockingHandler if none exists (idempotent), then sets the
    log level.  Library consumers who configure their own handlers are
    unaffected -- the guard ensures we never stack duplicate handlers.

    Args:
        level: Log level (e.g. logging.DEBUG, "WARNING"). Overrides env var.
        quiet: If True, set level to ERROR (suppress WARNING/INFO).
               Takes precedence over level.
    """
    lib_logger = logging.getLogger(LIBRARY_LOGGER_NAME)

    # Add a non-blocking handler if none exists yet.
    # This makes EXEC_SANDBOX_LOG_LEVEL work out of the box for CLI users
    # while staying invisible to library consumers who set up their own handlers.
    if not any(isinstance(h, _NonBlockingHandler) for h in lib_logger.handlers):
        lib_logger.addHandler(_NonBlockingHandler())

    if quiet:
        lib_logger.setLevel(logging.ERROR)
    elif level is not None:
        # setLevel() accepts both int and str; for str it validates
        # internally and raises ValueError on unknown names
        lib_logger.setLevel(level)

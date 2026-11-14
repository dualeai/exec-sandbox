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
    thread drains records to stderr.  When the queue is
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
import sys

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
_DIM, _RESET = "\x1b[2m", "\x1b[0m"


class _StderrHandler(logging.Handler):
    """Target handler: writes to stderr, dim when stderr is a terminal.

    Runs on the QueueListener's daemon thread, never on the caller's
    thread/coroutine.  No ANSI codes when stderr is not a TTY (files, pipes).
    """

    def __init__(self) -> None:
        super().__init__()
        self.formatter = logging.Formatter(fmt=_FMT, datefmt=_DATEFMT)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = sys.stderr
            stream.write(f"{_DIM}{msg}{_RESET}\n" if stream.isatty() else f"{msg}\n")
            stream.flush()
        except BlockingIOError:
            pass  # Stderr buffer full -- silently drop
        except Exception:  # noqa: BLE001
            self.handleError(record)


class _NonBlockingHandler(logging.handlers.QueueHandler):
    """Queue-backed handler that never blocks the caller.

    Records are enqueued via put_nowait() into a bounded FIFO.  A
    QueueListener daemon thread drains them to _StderrHandler.  When
    the queue is full, records are silently dropped (backpressure).
    close() stops the thread once (logging.shutdown at exit closes the
    handler again); after it, records are dropped.
    """

    def __init__(self) -> None:
        q: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=_QUEUE_CAPACITY)
        super().__init__(q)
        self._listener: logging.handlers.QueueListener | None = logging.handlers.QueueListener(
            q, _StderrHandler(), respect_handler_level=False
        )
        self._listener.start()

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """Skip serialization -- same-process queue, no pickle needed."""
        return record

    def enqueue(self, record: logging.LogRecord) -> None:
        if self._listener is None:  # closed: no thread drains the queue
            return
        with contextlib.suppress(queue.Full):
            self.queue.put_nowait(record)

    def close(self) -> None:
        # QueueListener.stop() fails on a second call before Python 3.13
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()
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
    """Add a stderr handler to the exec_sandbox logger, then set its level.

    Call it from a command line tool or an application entry point. A second
    call adds no second handler. Without this call, the library adds no handler.
    The handler never blocks: when its queue is full, it drops records.

    Args:
        level: Log level (for example logging.DEBUG, "WARNING"). It wins over
            EXEC_SANDBOX_LOG_LEVEL.
        quiet: If True, set the level to ERROR (hide WARNING and INFO). It wins
            over level.
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

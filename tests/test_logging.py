"""configure_logging: the stderr handler it adds to the library logger, after close; the
ExtrasFormatter of its lines and of the test logs."""

import logging
import logging.handlers
import sys

import pytest

from exec_sandbox import configure_logging
from exec_sandbox._logging import LIBRARY_LOGGER_NAME, ExtrasFormatter


def _configured() -> logging.handlers.QueueHandler:
    """The handler that configure_logging adds (tests/conftest.py removes it after the test)."""
    lib_logger = logging.getLogger(LIBRARY_LOGGER_NAME)
    before = set(lib_logger.handlers)
    configure_logging()
    (handler,) = set(lib_logger.handlers) - before
    assert isinstance(handler, logging.handlers.QueueHandler)
    return handler


def test_second_close_raises_nothing() -> None:
    """An application that closes the handler (its own logging.shutdown call) gets a second
    close from logging.shutdown at exit: that close does nothing."""
    handler = _configured()
    handler.close()
    handler.close()


def test_closed_handler_keeps_no_record() -> None:
    """After close, no thread reads the queue: a record is dropped, not kept in it."""
    handler = _configured()
    handler.close()
    logging.getLogger(LIBRARY_LOGGER_NAME).error("after close")
    assert handler.queue.qsize() == 0


def test_the_listener_formats_a_copy_of_the_record() -> None:
    """The listener thread formats its own copy of a record: the record that the caller's
    handlers format gets no field from it (asctime, which pytest's default log format does not
    use). On one shared record, a field that the listener adds while ExtrasFormatter iterates
    the fields in the caller's thread raises RuntimeError (free-threaded Python)."""
    handler = _configured()
    record = _record(extra={"vm_id": "vm-1"}, exc_info=False)
    handler.handle(record)
    handler.close()  # the listener formats every queued record before it stops
    assert "asctime" not in vars(record)


def _record(*, extra: dict[str, object], exc_info: bool) -> logging.LogRecord:
    """A WARNING record of the library logger with `extra`, as logger.warning(..., extra=) makes it."""
    try:
        raise ValueError("boom")
    except ValueError:
        info = sys.exc_info() if exc_info else None
    return logging.getLogger(LIBRARY_LOGGER_NAME).makeRecord(
        LIBRARY_LOGGER_NAME, logging.WARNING, "f.py", 1, "L1 restore failed", None, info, extra=extra
    )


class TestExtrasFormatter:
    """The library logs its details as extra fields: a log line shows them after the message."""

    def test_extra_fields_follow_the_message(self) -> None:
        record = _record(extra={"error": "TransientError: no answer", "vm_id": "vm-1"}, exc_info=False)
        line = ExtrasFormatter(fmt="%(levelname)s %(message)s").format(record)
        assert line == "WARNING L1 restore failed error='TransientError: no answer' vm_id='vm-1'"

    def test_no_extra_field_is_the_plain_line(self) -> None:
        record = _record(extra={}, exc_info=False)
        assert ExtrasFormatter(fmt="%(levelname)s %(message)s").format(record) == "WARNING L1 restore failed"

    def test_the_traceback_comes_after_the_fields(self) -> None:
        record = _record(extra={"vm_id": "vm-1"}, exc_info=True)
        first, *rest = ExtrasFormatter(fmt="%(message)s").format(record).splitlines()
        assert first == "L1 restore failed vm_id='vm-1'"
        assert rest[-1] == "ValueError: boom"


def test_captured_logs_show_the_extra_fields(caplog: pytest.LogCaptureFixture) -> None:
    """tests/conftest.py gives pytest's log handlers the ExtrasFormatter: a failed test's
    captured log (and caplog.text) names the error that an extra field holds."""
    with caplog.at_level(logging.WARNING, logger=LIBRARY_LOGGER_NAME):
        logging.getLogger(LIBRARY_LOGGER_NAME).warning("L1 restore failed", extra={"error": "MIGRATION_FAILED"})
    assert "L1 restore failed error='MIGRATION_FAILED'" in caplog.text

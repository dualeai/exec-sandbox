"""configure_logging: the stderr handler it adds to the library logger, after close."""

import logging
import logging.handlers

from exec_sandbox import configure_logging
from exec_sandbox._logging import LIBRARY_LOGGER_NAME


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

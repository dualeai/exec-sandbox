"""Logging wrapper using stdlib logging."""

import logging
from functools import cache


@cache
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    Uses caching to return the same logger for repeated calls with the same name.
    """
    return logging.getLogger(name)

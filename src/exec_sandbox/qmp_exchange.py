"""Shared id-matched QMP request/response exchange.

QMP echoes the request's ``id`` on every command response (per spec), and
interleaves asynchronous events (BALLOON_CHANGE, MIGRATION_PASS, STOP, ...) on
the same stream at any time.  A positional "first non-event line wins" match
desynchronizes the moment a command times out with its connection left open:
the late reply of command N is then consumed as the reply to command N+1.

This helper writes one identified command and consumes *its* matching response,
discarding events and any stale reply carrying a different ``id``.  The whole
read loop is bounded by a single ``asyncio.timeout`` (no manual deadline
arithmetic, no per-readline overshoot).  It returns the full response dict (with
``return``/``error`` keys) so callers keep their existing shape checks.

The bespoke positional loop in qemu_storage_daemon.py is intentionally NOT
routed through here — it carries a job/event-buffer layer that must stay pinned.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from exec_sandbox._logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

# Safety rail only — the asyncio.timeout below is the primary guard. Bounds a
# stream that floods events/stale replies faster than the deadline can fire.
_MAX_SKIPPED_MESSAGES = 64


async def qmp_id_exchange(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    request_id: str,
    command: str,
    arguments: dict[str, Any] | None,
    timeout: float,
    error_factory: Callable[[str], Exception],
) -> dict[str, Any]:
    """Write one ``id``-tagged QMP command and return its matching response.

    Args:
        reader: Connected QMP stream reader.
        writer: Connected QMP stream writer.
        request_id: Unique id echoed back by QEMU on the command's response.
        command: QMP command name.
        arguments: Optional command arguments.
        timeout: Total deadline for write + response (seconds).
        error_factory: Builds the client-specific exception from a message.

    Returns:
        The full QMP response dict (contains a ``return`` or ``error`` key).

    Raises:
        Exception: ``error_factory(...)`` on EOF or too many skipped messages.
    """
    msg: dict[str, Any] = {"execute": command, "id": request_id}
    if arguments:
        msg["arguments"] = arguments

    writer.write(json.dumps(msg).encode() + b"\n")
    await writer.drain()

    async with asyncio.timeout(timeout):
        for _ in range(_MAX_SKIPPED_MESSAGES):
            line = await reader.readline()
            if not line:
                raise error_factory("QMP connection closed unexpectedly")
            data: dict[str, Any] = json.loads(line)

            if "event" in data:
                logger.debug("QMP event (skipped)", extra={"event": data.get("event")})
                continue

            response_id = data.get("id")
            if response_id != request_id:
                # A stale reply to a prior, timed-out command. Discarding it
                # (rather than returning it) keeps the stream aligned.
                logger.warning(
                    "Discarding unmatched QMP response",
                    extra={"expected_id": request_id, "response_id": response_id},
                )
                continue

            return data
    raise error_factory("Too many QMP messages without a matching command response")

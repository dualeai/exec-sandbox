"""QMP client: one monitor connection of a QEMU process, with its own life cycle.

QmpClient wraps a connected stream (start: the greeting, then qmp_capabilities; execute:
id-tagged commands; close). QmpClient.connect opens one (peer check, start) for the length of
a `with` block: the run-state diagnostics and the tests. The migration client and the event
watch (qmp_events) hold one each.

QMP echoes the request's ``id`` on every command response (per spec), and interleaves
asynchronous events (MIGRATION_PASS, STOP, RESUME, ...) on the same stream at any time. A
positional "first non-event line wins" match desynchronizes the moment a command times out
with its connection left open: the late reply of command N is then consumed as the reply to
command N+1. execute writes one identified command and consumes *its* matching response,
giving the events to the caller or skipping them, and discarding any stale reply carrying a
different ``id``. The whole read loop is bounded by a single ``asyncio.timeout``.

qemu_storage_daemon.py keeps its own id-matched loop: it skips the job events, and its
_wait_for_job polls query-jobs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any, Self, cast

from exec_sandbox._logging import get_logger
from exec_sandbox.socket_auth import connect_and_verify

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from pathlib import Path

logger = get_logger(__name__)

# Bounds a stream that floods skipped messages (stale replies, and events with no
# on_event) faster than the deadline can fire, and the event watch's exchange, which
# has no deadline. Events given to on_event are acted on, not skipped: they do not
# count (QEMU sends one BLOCK_IO_ERROR per failed request, with no rate limit).
_MAX_SKIPPED_MESSAGES = 64

QMP_BUFFER_LIMIT = 16 * 1024 * 1024
"""StreamReader buffer limit of a QMP connection: readuntil() fails on a longer line."""


class QmpClient:
    """One QMP monitor connection. `error_factory` builds the error of a protocol failure
    (EOF, no greeting, a refused capabilities negotiation, too many skipped messages), so each
    owner raises its own class; `id_prefix` names its commands (<prefix>-1, <prefix>-2, ...).
    One user at a time: the owner serializes its calls."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        error_factory: Callable[[str], Exception],
        id_prefix: str,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._error_factory = error_factory
        self._id_prefix = id_prefix
        self._sequence = 0

    @classmethod
    @contextlib.asynccontextmanager
    async def connect(
        cls,
        path: Path,
        *,
        expected_uid: int,
        timeout: float,
        error_factory: Callable[[str], Exception],
        id_prefix: str,
    ) -> AsyncGenerator[Self]:
        """A started client on the socket at `path` (its peer is checked: connect_and_verify),
        closed when the block ends. `timeout` bounds the connect, then each step of start.

        Raises:
            What connect_and_verify raises (SocketAuthError, TimeoutError, OSError), then
            what start raises.
        """
        reader, writer = await connect_and_verify(
            path=str(path), expected_uid=expected_uid, timeout=timeout, buffer_limit=QMP_BUFFER_LIMIT
        )
        client = cls(reader, writer, error_factory=error_factory, id_prefix=id_prefix)
        try:
            await client.start(timeout=timeout)
            yield client
        finally:
            await client.close()

    @property
    def reader(self) -> asyncio.StreamReader:
        """The stream of the event lines that come after start (qmp_events reads it)."""
        return self._reader

    async def start(self, *, timeout: float) -> None:
        """Read QEMU's greeting, then leave capabilities negotiation (``qmp_capabilities``).

        Commands work only after it, and QEMU sends events to this connection only from then
        on: an event QEMU emitted before it never reaches this connection (a client that must
        not miss one queries the state right after).

        Raises:
            Exception: ``error_factory(...)`` on EOF, a first line that is no greeting, or a
                capabilities error.
            TimeoutError: No greeting, or no capabilities return, within `timeout`.
            ValueError: A line that is not JSON.
            OSError: The connection failed (reset, broken pipe).
        """
        async with asyncio.timeout(timeout):
            line = await self._reader.readline()
        if not line:
            raise self._error_factory("QMP connection closed before the greeting")
        greeting = json.loads(line)
        if not isinstance(greeting, dict) or "QMP" not in greeting:
            raise self._error_factory(f"QMP greeting expected, got {line[:200]!r}")
        response = await self._exchange(
            "qmp-capabilities", "qmp_capabilities", arguments=None, timeout=timeout, on_event=None
        )
        if "error" in response:
            raise self._error_factory(f"QMP capabilities failed: {response['error']}")

    async def execute(
        self,
        command: str,
        *,
        arguments: dict[str, Any] | None,
        timeout: float,
        on_event: Callable[[bytes], Awaitable[None]] | None,
    ) -> dict[str, Any]:
        """Write one ``id``-tagged command and return its matching response (with its
        ``return`` or ``error`` key). `on_event` gets each event line that comes before it
        (QEMU sends events as soon as capabilities are negotiated), or None skips them.
        `timeout` bounds the write and the response.

        Raises:
            Exception: ``error_factory(...)`` on EOF or too many skipped messages.
            TimeoutError: no response within `timeout`.
            ValueError: A line that is not JSON.
            OSError: The connection failed (reset, broken pipe).
        """
        self._sequence += 1
        return await self._exchange(
            f"{self._id_prefix}-{self._sequence}", command, arguments=arguments, timeout=timeout, on_event=on_event
        )

    async def _exchange(
        self,
        request_id: str,
        command: str,
        *,
        arguments: dict[str, Any] | None,
        timeout: float,
        on_event: Callable[[bytes], Awaitable[None]] | None,
    ) -> dict[str, Any]:
        """execute with `request_id`."""
        message: dict[str, Any] = {"execute": command, "id": request_id}
        if arguments:
            message["arguments"] = arguments
        skipped = 0
        async with asyncio.timeout(timeout):
            self._writer.write(json.dumps(message).encode() + b"\n")
            await self._writer.drain()  # waits only while QEMU does not read: inside the deadline
            while skipped < _MAX_SKIPPED_MESSAGES:
                line = await self._reader.readline()
                if not line:
                    raise self._error_factory("QMP connection closed unexpectedly")
                parsed = json.loads(line)
                if not isinstance(parsed, dict):  # no QMP message (QEMU sends objects only): skipped
                    skipped += 1
                    logger.warning("QMP line that is not an object, skipped", extra={"line": line[:200]})
                    continue
                data = cast("dict[str, Any]", parsed)
                if "event" in data and on_event is not None:
                    await on_event(line)
                    continue
                skipped += 1
                if "event" in data:
                    logger.debug("QMP event (skipped)", extra={"event": data.get("event")})
                    continue
                if data.get("id") != request_id:
                    # A stale reply to a prior, timed-out command. Discarding it (rather
                    # than returning it) keeps the stream aligned.
                    logger.warning(
                        "Discarding unmatched QMP response",
                        extra={"expected_id": request_id, "response_id": data.get("id")},
                    )
                    continue
                return data
        raise self._error_factory("Too many QMP messages without a matching command response")

    async def close(self) -> None:
        """Close the connection (at most 1 s for its end; an error of the close is no error)."""
        self._writer.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._writer.wait_closed(), timeout=1.0)

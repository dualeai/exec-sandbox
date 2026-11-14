"""QmpClient: the start of a connection, id-matched commands, events before a response, the
limit on skipped messages, and the close of a connection that fails to start."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from typing import TYPE_CHECKING, Any

import pytest

from exec_sandbox.qmp_client import QmpClient

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

_EVENT = b'{"event": "BLOCK_IO_ERROR", "data": {"nospace": false}, "timestamp": {"seconds": 0, "microseconds": 0}}\n'
_GREETING = b'{"QMP": {"version": {}, "capabilities": []}}\n'


async def _run(qemu_sends: bytes, act: Callable[[QmpClient], Awaitable[Any]]) -> tuple[Any, list[dict[str, Any]]]:
    """Run `act` on a client whose peer has sent `qemu_sends`, then ended its stream. Returns
    what `act` returned (or the error it raised) and the commands the client wrote."""
    ours, qemu = socket.socketpair()
    reader, writer = await asyncio.open_connection(sock=ours)
    client = QmpClient(reader, writer, error_factory=ConnectionResetError, id_prefix="q")

    def send() -> None:  # then the end of the stream, as at QEMU's exit
        qemu.sendall(qemu_sends)
        qemu.shutdown(socket.SHUT_WR)

    # a thread: the burst can be larger than a socket pair's buffer (8 KiB on macOS)
    sender = asyncio.create_task(asyncio.to_thread(send))
    try:
        result: Any = await act(client)
    except ConnectionResetError as e:
        result = e
    finally:
        await client.close()  # a client that stopped early leaves the sender blocked: closing ends it
        await asyncio.gather(sender, return_exceptions=True)
        written = b""
        while chunk := await asyncio.to_thread(qemu.recv, 65536):
            written += chunk
        qemu.close()
    return result, [json.loads(line) for line in written.splitlines()]


class TestStart:
    """start, the start of every monitor connection (migration client, run-state diagnostics,
    event watch): the greeting, then qmp_capabilities matched by its id."""

    async def test_greeting_then_capabilities_by_id(self) -> None:
        sends = _GREETING + b'{"return": {}, "id": "an-older-command"}\n{"return": {}, "id": "qmp-capabilities"}\n'
        result, written = await _run(sends, lambda client: client.start(timeout=5))
        assert result is None  # the stale reply is skipped, not taken as ours
        assert written == [{"execute": "qmp_capabilities", "id": "qmp-capabilities"}]

    @pytest.mark.parametrize(
        ("sends", "message"),
        [
            pytest.param(b"", "closed before the greeting", id="eof"),
            pytest.param(b'{"return": {}, "id": "qmp-capabilities"}\n', "greeting expected", id="no-greeting"),
            pytest.param(
                b'{"QMP": {}}\n{"error": {"class": "GenericError"}, "id": "qmp-capabilities"}\n',
                "capabilities failed",
                id="capabilities-error",
            ),
        ],
    )
    async def test_failures_use_the_error_factory(self, sends: bytes, message: str) -> None:
        result, _ = await _run(sends, lambda client: client.start(timeout=5))
        assert isinstance(result, ConnectionResetError)
        assert message in str(result)


class TestExecute:
    async def test_commands_are_numbered_under_the_prefix(self) -> None:
        sends = b'{"return": {"status": "running"}, "id": "q-1"}\n{"return": [], "id": "q-2"}\n'

        async def two(client: QmpClient) -> list[dict[str, Any]]:
            first = await client.execute("query-status", arguments=None, timeout=5, on_event=None)
            return [first, await client.execute("query-block", arguments={"flat": True}, timeout=5, on_event=None)]

        result, written = await _run(sends, two)
        assert result == [{"return": {"status": "running"}, "id": "q-1"}, {"return": [], "id": "q-2"}]
        assert written == [
            {"execute": "query-status", "id": "q-1"},
            {"execute": "query-block", "id": "q-2", "arguments": {"flat": True}},
        ]

    async def test_events_given_to_on_event_do_not_count_as_skipped(self) -> None:
        """QEMU sends one BLOCK_IO_ERROR per failed request (no rate limit): a burst before the
        response is acted on, and the response still comes (the event watch's query has no
        deadline: if these events counted, the limit would drop its connection)."""
        acted: list[bytes] = []

        async def on_event(line: bytes) -> None:
            acted.append(line)

        result, _ = await _run(
            _EVENT * 100 + b'{"return": {}, "id": "q-1"}\n',
            lambda client: client.execute("query-block", arguments=None, timeout=5, on_event=on_event),
        )
        assert result == {"return": {}, "id": "q-1"}
        assert len(acted) == 100

    async def test_a_line_that_is_no_object_is_skipped(self) -> None:
        """A JSON line that is not an object is skipped (counted), not a TypeError that would
        end the event watch while QEMU runs."""
        result, _ = await _run(
            b'[1, 2]\n"text"\n{"return": {}, "id": "q-1"}\n',
            lambda client: client.execute("query-status", arguments=None, timeout=5, on_event=None),
        )
        assert result == {"return": {}, "id": "q-1"}

    async def test_skipped_events_still_hit_the_limit(self) -> None:
        result, _ = await _run(
            _EVENT * 100 + b'{"return": {}, "id": "q-1"}\n',
            lambda client: client.execute("query-block", arguments=None, timeout=5, on_event=None),
        )
        assert isinstance(result, ConnectionResetError)
        assert "Too many QMP messages" in str(result)


async def test_connect_closes_a_connection_that_fails_to_start(short_tmp_dir: Path) -> None:
    """QmpClient.connect checks the peer, starts, and closes the socket when the start fails:
    the peer (QEMU) sees the end of the connection."""
    path = short_tmp_dir / "qmp.sock"
    peer_saw_eof = asyncio.Event()

    async def qemu(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_GREETING)
        await reader.readline()  # qmp_capabilities
        writer.write(b'{"error": {"class": "GenericError"}, "id": "qmp-capabilities"}\n')
        await writer.drain()
        if await reader.read() == b"":
            peer_saw_eof.set()
        writer.close()

    server = await asyncio.start_unix_server(qemu, path=str(path))
    try:
        with pytest.raises(OSError, match="capabilities failed") as caught:
            async with QmpClient.connect(
                path, expected_uid=os.getuid(), timeout=5, error_factory=OSError, id_prefix="t"
            ):
                pytest.fail("a failed start yields no client")
        # `caught` keeps the failed client alive (its traceback holds connect's frame): only
        # its close ends the stream, not the garbage collector
        async with asyncio.timeout(5):
            await peer_saw_eof.wait()
        assert caught.value.__traceback__ is not None
    finally:
        server.close()  # no wait_closed: it waits for a client that a broken close leaves open

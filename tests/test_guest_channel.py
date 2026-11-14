"""The end of a UnixSocketChannel connection: close() and connect() after a write failed, a
close() to a guest that reads nothing or whose caller is cancelled, and the frames that no
connection sent.

The guest side is a Unix socket server in this process. A write fails for real: the guest
closes its end, then the channel writes (EPIPE).
"""

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exec_sandbox import guest_channel
from exec_sandbox.guest_channel import DualPortChannel, UnixSocketChannel
from tests.instance_support import await_until


@dataclass
class _Guest:
    """The guest side of a channel's socket: each connection it accepted, in order.

    It keeps each writer it hands out until the test ends: since Python 3.13 a StreamWriter
    that is garbage-collected closes its socket, which would end the connection of a guest
    that reads nothing in the middle of the test."""

    path: Path
    accepted: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]]
    handed_out: list[asyncio.StreamWriter] = field(default_factory=list[asyncio.StreamWriter])

    async def next_connection(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.wait_for(self.accepted.get(), timeout=1)
        self.handed_out.append(writer)
        return reader, writer


@pytest.fixture
async def guest(short_tmp_dir: Path) -> AsyncIterator[_Guest]:
    accepted: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = asyncio.Queue()

    async def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await accepted.put((reader, writer))

    path = short_tmp_dir / "cmd.sock"
    server = await asyncio.start_unix_server(on_connect, path=str(path))
    guest = _Guest(path, accepted)
    try:
        yield guest
    finally:
        while not accepted.empty():
            accepted.get_nowait()[1].close()
        for writer in guest.handed_out:
            writer.close()
        server.close()


async def _broken(guest: _Guest) -> UnixSocketChannel:
    """A channel whose write failed: the guest closed its end, then a frame was written."""
    channel = UnixSocketChannel(str(guest.path), expected_uid=os.getuid())
    await channel.connect(timeout_seconds=1)
    _, writer = await guest.next_connection()
    writer.close()
    await writer.wait_closed()
    await channel.enqueue_write(b"lost\n", timeout=1)
    await await_until(lambda: not channel.is_connected())  # the write worker ended on EPIPE
    return channel


async def test_close_after_a_failed_write_ends_at_once(guest: _Guest) -> None:
    """A frame queued after the write failure is never sent: close() does not wait for it (a
    destroy then takes no 5 s), and raises no error of the lost connection."""
    channel = await _broken(guest)
    await channel.enqueue_write(b"queued\n", timeout=1)

    async with asyncio.timeout(1):
        await channel.close()

    assert not channel.is_connected()


async def test_connect_after_a_failed_write_opens_a_new_connection(guest: _Guest) -> None:
    """connect() replaces a connection whose write failed (it does not keep the dead socket):
    the guest accepts a new connection, and the next frame reaches it."""
    channel = await _broken(guest)

    await channel.connect(timeout_seconds=1)
    reader, _ = await guest.next_connection()
    await channel.enqueue_write(b"fresh\n", timeout=1)

    assert await asyncio.wait_for(reader.readline(), timeout=1) == b"fresh\n"
    await channel.close()


async def test_frames_that_no_connection_sent_never_reach_the_next_one(guest: _Guest) -> None:
    """A frame queued while no connection sends it (no worker) is dropped at close(): the
    next connection gets only the frames queued on it."""
    channel = UnixSocketChannel(str(guest.path), expected_uid=os.getuid())
    await channel.enqueue_write(b"stale\n", timeout=1)
    await channel.close()

    await channel.connect(timeout_seconds=1)
    reader, _ = await guest.next_connection()
    await channel.enqueue_write(b"fresh\n", timeout=1)

    assert await asyncio.wait_for(reader.readline(), timeout=1) == b"fresh\n"
    await channel.close()


_FRAME = b"x" * 65535 + b"\n"


async def _flooded(guest: _Guest) -> tuple[UnixSocketChannel, asyncio.StreamReader]:
    """A channel with more frames queued than the socket buffers hold, to a guest that reads
    nothing (stuck, or paused on a full host disk): the write worker waits in drain()."""
    channel = UnixSocketChannel(str(guest.path), expected_uid=os.getuid())
    await channel.connect(timeout_seconds=1)
    reader, _ = await guest.next_connection()
    for _ in range(64):
        await channel.enqueue_write(_FRAME, timeout=1)
    return channel, reader


async def test_close_ends_when_the_guest_reads_nothing(guest: _Guest, monkeypatch: pytest.MonkeyPatch) -> None:
    """A guest that reads nothing gets CLOSE_SEND_TIMEOUT_SECONDS (cut to 50 ms) for the frames
    still queued; then the socket closes at once (a destroy goes on to end QEMU), and the
    guest reads what its socket got, then the end of the stream."""
    monkeypatch.setattr(guest_channel, "CLOSE_SEND_TIMEOUT_SECONDS", 0.05)
    channel, reader = await _flooded(guest)

    async with asyncio.timeout(1):
        await channel.close()

    assert not channel.is_connected()
    async with asyncio.timeout(5):
        received = await reader.read()  # until the end of the stream
    assert len(received) < 64 * len(_FRAME)  # the frames still queued were dropped


async def test_a_cancelled_close_still_ends_the_connection(guest: _Guest) -> None:
    """A guest that reads nothing holds close() in its wait for the queued frames; a cancel
    of its caller then still stops the write worker and closes the socket: the guest reads
    what its socket got so far (the kernel buffers: no fixed size), then the end of the
    stream."""
    channel, reader = await _flooded(guest)

    closing = asyncio.create_task(channel.close())
    await asyncio.sleep(0.05)  # close() waits for the queue
    assert not closing.done()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert not channel.is_connected()
    async with asyncio.timeout(5):
        received = await reader.read()  # until the end of the stream
    assert len(received) < 64 * len(_FRAME)  # the frames still queued were dropped


def _two_ports(tmp: Path) -> DualPortChannel:
    """A DualPortChannel whose two ports the test replaces: no portable socket makes one
    port's connect slow while the other fails, so each port's connect/close stands in."""
    return DualPortChannel(str(tmp / "cmd.sock"), str(tmp / "event.sock"), expected_uid=os.getuid())


async def test_a_port_that_fails_to_connect_ends_the_other_connect(short_tmp_dir: Path) -> None:
    """The command port is refused while the event port still connects: connect() raises the
    refusal once the event port's connect has ended, so no connect runs on after the call."""
    channel = _two_ports(short_tmp_dir)
    event_connect_ended = asyncio.Event()

    async def refused(_timeout: float) -> None:
        await asyncio.sleep(0)  # the event port's connect is under way
        raise ConnectionRefusedError("refused")

    async def slow(_timeout: float) -> None:
        try:
            await asyncio.sleep(30)
        finally:
            event_connect_ended.set()

    channel._cmd_channel.connect = refused  # type: ignore[method-assign]
    channel._event_channel.connect = slow  # type: ignore[method-assign]
    with pytest.raises(ConnectionRefusedError, match=r"^refused$"):
        await channel.connect(timeout_seconds=5.0)
    assert event_connect_ended.is_set()


async def test_a_port_that_fails_to_close_still_closes_the_other(short_tmp_dir: Path) -> None:
    """The command port's close raises: the event port's close still runs to its end before
    close() raises that error (no close left running after the call)."""
    channel = _two_ports(short_tmp_dir)
    event_closed = asyncio.Event()

    async def failing() -> None:
        raise OSError("close failed")

    async def slow_close() -> None:
        await asyncio.sleep(0.05)
        event_closed.set()

    channel._cmd_channel.close = failing  # type: ignore[method-assign]
    channel._event_channel.close = slow_close  # type: ignore[method-assign]
    with pytest.raises(OSError, match=r"^close failed$"):
        await channel.close()
    assert event_closed.is_set()

"""Tests for the write path of UnixSocketChannel: frames to the guest, the batched drain, the
end of the connection, and the errors of its write worker.

The public behavior runs on a real Unix socket (the guest side is a server in this process).
The batch size and the errors of the write worker show only in its calls to the stream
writer: those tests drive _write_worker on a mocked writer.
"""

import asyncio
import contextlib
import logging
import os
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox.guest_channel import SOCKET_SEND_BUFFER_BYTES, FileOpDispatcher, UnixSocketChannel


@dataclass
class _Guest:
    """The guest end of a channel's socket."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter


@pytest.fixture
async def connected(short_tmp_dir: Path) -> AsyncIterator[tuple[UnixSocketChannel, _Guest]]:
    """A UnixSocketChannel connected to a Unix socket server in this process, and the server's
    end of the connection."""
    accepted: asyncio.Future[_Guest] = asyncio.get_running_loop().create_future()

    async def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        accepted.set_result(_Guest(reader, writer))

    path = short_tmp_dir / "cmd.sock"
    server = await asyncio.start_unix_server(on_connect, path=str(path))
    channel = UnixSocketChannel(str(path), expected_uid=os.getuid())
    try:
        await channel.connect(timeout_seconds=1)
        guest = await asyncio.wait_for(accepted, timeout=1)
        yield channel, guest
        await channel.close()
        guest.writer.close()
    finally:
        server.close()


# ============================================================================
# TestChannelWrites - enqueue_write, is_connected and close on a real socket
# ============================================================================


class TestChannelWrites:
    async def test_frames_reach_the_guest_in_order(self, connected: tuple[UnixSocketChannel, _Guest]) -> None:
        """More frames than one batch (16) reach the guest, all of them, in order."""
        channel, guest = connected
        frames = [f"msg{i}\n".encode() for i in range(20)]

        for frame in frames:
            await channel.enqueue_write(frame)

        received = await asyncio.wait_for(guest.reader.readexactly(sum(map(len, frames))), timeout=1)
        assert received == b"".join(frames)

    async def test_idle_channel_stays_connected_and_writes_later(
        self, connected: tuple[UnixSocketChannel, _Guest]
    ) -> None:
        """With nothing to write, the write worker waits (no poll, no exit): the channel stays
        connected, and a later frame reaches the guest."""
        channel, guest = connected
        for _ in range(10):  # loop turns in which a worker that does not wait would end
            await asyncio.sleep(0)

        assert channel.is_connected()
        await channel.enqueue_write(b"later\n")
        assert await asyncio.wait_for(guest.reader.readline(), timeout=1) == b"later\n"

    async def test_close_sends_queued_frames_then_ends_the_connection(
        self, connected: tuple[UnixSocketChannel, _Guest]
    ) -> None:
        """close() writes the frames still in the queue, then ends the connection: the guest
        reads them, then the end of the stream; the channel is no longer connected."""
        channel, guest = connected
        for i in range(3):
            await channel.enqueue_write(f"last{i}\n".encode())

        await channel.close()

        assert await asyncio.wait_for(guest.reader.read(), timeout=1) == b"last0\nlast1\nlast2\n"
        assert not channel.is_connected()

    async def test_send_buffer_holds_a_file_chunk(self, connected: tuple[UnixSocketChannel, _Guest]) -> None:
        """connect() raises the send buffer of the socket (macOS gives a Unix socket 8KB) so
        that one send writes the largest frame, a file chunk."""
        channel, _ = connected
        assert channel._writer is not None
        sock = channel._writer.get_extra_info("socket")
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF) >= SOCKET_SEND_BUFFER_BYTES

    async def test_not_connected_before_connect(self) -> None:
        """A channel that never connected is not connected."""
        assert not UnixSocketChannel("/fake/sock", expected_uid=0).is_connected()

    async def test_queue_full_backpressure(self) -> None:
        """64 frames fill the queue of a channel whose worker does not drain it (not connected:
        no worker): the next enqueue_write raises RuntimeError at its timeout."""
        channel = UnixSocketChannel("/fake/sock", expected_uid=0)
        for i in range(64):
            await channel.enqueue_write(f"msg{i}".encode(), timeout=0.1)

        with pytest.raises(RuntimeError, match="Write queue full"):
            await channel.enqueue_write(b"overflow", timeout=0.01)


# ============================================================================
# The write worker on a mocked writer
# ============================================================================


@pytest.fixture
async def channel() -> UnixSocketChannel:
    """UnixSocketChannel with mocked reader/writer (no real socket)."""
    ch = UnixSocketChannel("/fake/sock", expected_uid=0)
    ch._reader = asyncio.StreamReader()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    ch._writer = writer
    return ch


async def _start_worker(ch: UnixSocketChannel) -> asyncio.Task[None]:
    """Start the write worker as a background task."""
    return asyncio.create_task(ch._write_worker())  # pyright: ignore[reportPrivateUsage]


async def _stop_worker(ch: UnixSocketChannel, task: asyncio.Task[None]) -> None:
    """Cleanly stop the write worker by cancelling it (the authoritative stop)."""
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)


class TestWriteWorkerBatch:
    async def test_batches_up_to_16_items(self, channel: UnixSocketChannel) -> None:
        """20 queued frames: a batch of 16 (1 get + 15 get_nowait) and a batch of 4, one drain
        each. The batch shows only in the drain calls: the bytes on the wire are the same."""
        for i in range(20):
            await channel._write_queue.put(f"msg{i}\n".encode())

        task = await _start_worker(channel)
        await asyncio.wait_for(channel._write_queue.join(), timeout=1)
        for _ in range(3):  # the drain after the last batch
            await asyncio.sleep(0)

        assert channel._writer is not None
        assert channel._writer.write.call_count == 20
        assert channel._writer.drain.await_count == 2

        await _stop_worker(channel, task)

    async def test_cancel_with_queued_data_stops_the_worker(self, channel: UnixSocketChannel) -> None:
        """A cancel in the same loop turn as new data stops the worker (no error of its own)."""
        task = await _start_worker(channel)

        await channel._write_queue.put(b"last\n")
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)

    async def test_closing_transport_is_not_connected(self, channel: UnixSocketChannel) -> None:
        """A transport that is closing (its connection was lost while the worker waits) is not
        connected, before any write fails. A mocked writer: on a real socket the loss shows
        only at the next write."""
        assert channel._writer is not None
        channel._writer.is_closing = MagicMock(return_value=True)
        channel._write_task = await _start_worker(channel)
        try:
            assert not channel.is_connected()
        finally:
            await _stop_worker(channel, channel._write_task)


# ============================================================================
# TestWriteWorkerErrorHandling - the bug fix + edge cases
# ============================================================================

# Parametrized: RuntimeError raised by queue.get()
_GET_ERROR_CASES = [
    pytest.param("Event loop is closed", False, id="event-loop-closed"),
    pytest.param("something else", True, id="other-runtime-error"),
]

# Parametrized: errors raised by writer.write() or writer.drain()
_WRITER_ERROR_CASES = [
    pytest.param("drain", AsyncMock, RuntimeError("Event loop is closed"), False, id="event-loop-closed-drain"),
    pytest.param("drain", AsyncMock, OSError("broken pipe"), True, id="oserror-drain"),
    pytest.param("write", MagicMock, BrokenPipeError("broken pipe"), True, id="broken-pipe-write"),
]


class TestWriteWorkerErrorHandling:
    """A write error ends the worker (the channel then reports it is not connected), logged
    except "Event loop is closed" (the interpreter shutdown). The errors are raised by a mocked
    writer: a real socket cannot raise each of them on demand."""

    @pytest.mark.parametrize("error_msg,expect_log", _GET_ERROR_CASES)
    async def test_get_raises_runtime_error(
        self, channel: UnixSocketChannel, caplog: pytest.LogCaptureFixture, error_msg: str, expect_log: bool
    ) -> None:
        """RuntimeError from queue.get() — silent for 'Event loop is closed', logged otherwise."""
        with patch.object(
            channel._write_queue,
            "get",
            side_effect=RuntimeError(error_msg),
        ):
            with caplog.at_level(logging.ERROR, logger="exec_sandbox.guest_channel"):
                task = await _start_worker(channel)
                await asyncio.wait_for(task, timeout=5.0)

        assert task.exception() is None
        if expect_log:
            assert "write worker error" in caplog.text.lower()
        else:
            assert "write worker error" not in caplog.text.lower()

    @pytest.mark.parametrize("mock_attr,mock_cls,exc,expect_log", _WRITER_ERROR_CASES)
    async def test_writer_method_error(
        self,
        *,
        channel: UnixSocketChannel,
        caplog: pytest.LogCaptureFixture,
        mock_attr: str,
        mock_cls: type,
        exc: Exception,
        expect_log: bool,
    ) -> None:
        """Errors from writer.write()/drain() end the worker — silent for 'Event loop is
        closed', logged otherwise — and the channel is no longer connected."""
        assert channel._writer is not None
        setattr(channel._writer, mock_attr, mock_cls(side_effect=exc))
        channel._writer.is_closing = MagicMock(return_value=False)

        with caplog.at_level(logging.ERROR, logger="exec_sandbox.guest_channel"):
            channel._write_task = await _start_worker(channel)
            assert channel.is_connected()
            await channel._write_queue.put(b"data\n")
            await asyncio.wait_for(channel._write_task, timeout=5.0)

        assert channel._write_task.exception() is None
        assert not channel.is_connected()
        if expect_log:
            assert "write worker error" in caplog.text.lower()
        else:
            assert "write worker error" not in caplog.text.lower()


# ============================================================================
# TestFileOpDispatcherIsAlive - dispatch loop lifecycle
# ============================================================================


class TestFileOpDispatcherIsAlive:
    async def test_alive_then_dead_on_eof(self) -> None:
        """Dispatcher is_alive() returns True while running, False after EOF."""
        reader = asyncio.StreamReader()
        dispatcher = FileOpDispatcher(reader)

        assert not dispatcher.is_alive(), "Not alive before start"

        dispatcher.start()
        await asyncio.sleep(0.05)
        assert dispatcher.is_alive(), "Alive after start"

        # Feed EOF to make dispatch loop exit
        reader.feed_eof()
        await asyncio.sleep(0.1)
        assert not dispatcher.is_alive(), "Dead after EOF"

        await dispatcher.stop()

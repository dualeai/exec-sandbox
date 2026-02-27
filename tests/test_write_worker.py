"""Tests for UnixSocketChannel._write_worker error handling."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox.guest_channel import UnixSocketChannel


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
    ch._shutdown_event.clear()  # pyright: ignore[reportPrivateUsage]
    return asyncio.create_task(ch._write_worker())  # pyright: ignore[reportPrivateUsage]


async def _stop_worker(ch: UnixSocketChannel, task: asyncio.Task[None]) -> None:
    """Cleanly stop the write worker."""
    ch._shutdown_event.set()
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except asyncio.CancelledError:
        pass


# ============================================================================
# TestWriteWorkerShutdown - normal lifecycle
# ============================================================================


class TestWriteWorkerShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_event_stops_worker(self, channel: UnixSocketChannel):
        """Set _shutdown_event, verify worker exits cleanly."""
        task = await _start_worker(channel)

        channel._shutdown_event.set()
        await asyncio.wait_for(task, timeout=5.0)

        assert task.done()
        assert task.exception() is None

    @pytest.mark.asyncio
    async def test_timeout_cycles_until_shutdown(self, channel: UnixSocketChannel):
        """Worker stays alive with empty queue, exits on shutdown."""
        task = await _start_worker(channel)

        # Let the worker cycle through at least one timeout
        await asyncio.sleep(1.5)
        assert not task.done(), "Worker should still be running"

        await _stop_worker(channel, task)
        assert task.done()


# ============================================================================
# TestWriteWorkerWrites - data processing
# ============================================================================


class TestWriteWorkerWrites:
    @pytest.mark.asyncio
    async def test_single_write_and_drain(self, channel: UnixSocketChannel):
        """Put 1 item in queue, verify write() + drain() called."""
        task = await _start_worker(channel)

        await channel._write_queue.put(b"hello\n")
        # Give worker time to process
        await asyncio.sleep(0.1)

        assert channel._writer is not None
        channel._writer.write.assert_called_once_with(b"hello\n")
        channel._writer.drain.assert_awaited_once()

        await _stop_worker(channel, task)

    @pytest.mark.asyncio
    async def test_batches_up_to_16_items(self, channel: UnixSocketChannel):
        """Pre-fill 20 items, verify first batch is 16 (1 get + 15 get_nowait)."""
        # Pre-fill queue before starting worker so all items are available
        for i in range(20):
            await channel._write_queue.put(f"msg{i}\n".encode())

        task = await _start_worker(channel)
        # Give worker time to process
        await asyncio.sleep(0.2)

        assert channel._writer is not None
        # First batch: 16 writes (1 from get + 15 from get_nowait), then 1 drain
        # Second batch: 4 remaining writes, then 1 drain
        assert channel._writer.write.call_count == 20
        # drain is called once per batch: first batch of 16, then batch of 4
        assert channel._writer.drain.await_count == 2

        await _stop_worker(channel, task)

    @pytest.mark.asyncio
    async def test_writer_none_skips_write(self, channel: UnixSocketChannel):
        """With _writer=None, data is dequeued but no write/drain calls."""
        channel._writer = None
        task = await _start_worker(channel)

        await channel._write_queue.put(b"ignored\n")
        await asyncio.sleep(0.1)

        # Data was consumed (queue empty)
        assert channel._write_queue.empty()

        await _stop_worker(channel, task)


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
    @pytest.mark.parametrize("error_msg,expect_log", _GET_ERROR_CASES)
    @pytest.mark.asyncio
    async def test_get_raises_runtime_error(
        self, channel: UnixSocketChannel, caplog: pytest.LogCaptureFixture, error_msg: str, expect_log: bool
    ):
        """RuntimeError from queue.get() — silent for 'Event loop is closed', logged otherwise."""
        with patch.object(
            channel._write_queue,
            "get",
            side_effect=RuntimeError(error_msg),
        ):
            with caplog.at_level(logging.ERROR, logger="exec_sandbox.guest_channel"):
                task = await _start_worker(channel)
                await asyncio.wait_for(task, timeout=5.0)

        assert task.done()
        assert task.exception() is None
        if expect_log:
            assert "write worker error" in caplog.text.lower()
        else:
            assert "write worker error" not in caplog.text.lower()

    @pytest.mark.parametrize("mock_attr,mock_cls,exc,expect_log", _WRITER_ERROR_CASES)
    @pytest.mark.asyncio
    async def test_writer_method_error(
        self,
        channel: UnixSocketChannel,
        caplog: pytest.LogCaptureFixture,
        mock_attr: str,
        mock_cls: type,
        exc: Exception,
        expect_log: bool,
    ):
        """Errors from writer.write()/drain() — silent for 'Event loop is closed', logged otherwise."""
        assert channel._writer is not None
        setattr(channel._writer, mock_attr, mock_cls(side_effect=exc))

        with caplog.at_level(logging.ERROR, logger="exec_sandbox.guest_channel"):
            task = await _start_worker(channel)
            await channel._write_queue.put(b"data\n")
            await asyncio.wait_for(task, timeout=5.0)

        assert task.done()
        assert task.exception() is None
        if expect_log:
            assert "write worker error" in caplog.text.lower()
        else:
            assert "write worker error" not in caplog.text.lower()


# ============================================================================
# TestWriteWorkerEdgeCases - weird / out-of-bound
# ============================================================================


class TestWriteWorkerEdgeCases:
    @pytest.mark.asyncio
    async def test_queue_full_backpressure(self, channel: UnixSocketChannel):
        """Fill queue to maxsize (64), verify enqueue_write raises RuntimeError."""
        # Fill the queue without starting the worker (nothing draining)
        for i in range(64):
            await channel._write_queue.put(f"msg{i}".encode())

        with pytest.raises(RuntimeError, match="Write queue full"):
            await channel.enqueue_write(b"overflow", timeout=0.1)

    @pytest.mark.asyncio
    async def test_concurrent_shutdown_and_data(self, channel: UnixSocketChannel):
        """Put data AND set shutdown simultaneously — no crash."""
        task = await _start_worker(channel)

        # Race: enqueue data and signal shutdown at the same time
        await channel._write_queue.put(b"last\n")
        channel._shutdown_event.set()

        await asyncio.wait_for(task, timeout=5.0)
        assert task.done()
        # Worker should not raise
        assert task.exception() is None

    @pytest.mark.asyncio
    async def test_worker_exits_on_cancelled_error(self, channel: UnixSocketChannel):
        """Cancelling the worker task raises CancelledError."""
        task = await _start_worker(channel)

        # Let the worker enter its loop
        await asyncio.sleep(0.05)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

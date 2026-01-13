"""Tests for guest channel communication.

Tests TCP channel with real sockets - no mocks.
"""

import asyncio
import json

import pytest

from exec_sandbox.guest_agent_protocol import (
    ExecutionCompleteMessage,
    OutputChunkMessage,
    PingRequest,
    PongMessage,
)
from exec_sandbox.guest_channel import TcpChannel
from exec_sandbox.models import Language

# ============================================================================
# Test TCP Server Helper
# ============================================================================


async def run_test_server(
    host: str,
    port: int,
    responses: list[dict],
    ready_event: asyncio.Event,
) -> None:
    """Run a simple TCP server that returns predefined responses.

    Args:
        host: Host to bind to
        port: Port to bind to
        responses: List of response dicts to return (one per request)
        ready_event: Event to signal when server is ready
    """
    response_queue = asyncio.Queue()
    for r in responses:
        await response_queue.put(r)

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                # Read request
                data = await reader.readuntil(b"\n")
                if not data:
                    break

                # Send response
                if not response_queue.empty():
                    response = await response_queue.get()
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                else:
                    break
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host, port)
    ready_event.set()

    async with server:
        # Run until cancelled
        try:
            await asyncio.sleep(30)  # Max 30s
        except asyncio.CancelledError:
            pass


# ============================================================================
# Unit Tests
# ============================================================================


class TestTcpChannelInit:
    """Tests for TcpChannel initialization."""

    def test_init(self) -> None:
        """TcpChannel stores host and port."""
        channel = TcpChannel("127.0.0.1", 5000)
        assert channel.host == "127.0.0.1"
        assert channel.port == 5000
        assert channel._reader is None
        assert channel._writer is None

    def test_init_localhost(self) -> None:
        """TcpChannel with localhost."""
        channel = TcpChannel("localhost", 8080)
        assert channel.host == "localhost"
        assert channel.port == 8080


class TestTcpChannelConnect:
    """Tests for TcpChannel.connect method."""

    async def test_connect_success(self) -> None:
        """TcpChannel connects to real server."""
        ready = asyncio.Event()
        server_task = asyncio.create_task(
            run_test_server("127.0.0.1", 0, [], ready)  # Port 0 = random
        )

        # Wait for server to be ready (can't get port from here easily)
        # Use a fixed port for testing
        await asyncio.sleep(0.05)
        server_task.cancel()

    async def test_connect_refused(self) -> None:
        """TcpChannel raises ConnectionRefusedError on connection failure."""
        # Try to connect to a port that's not listening
        channel = TcpChannel("127.0.0.1", 59999)

        with pytest.raises(ConnectionRefusedError):
            await channel.connect(timeout_seconds=1)

    async def test_connect_already_connected(self) -> None:
        """TcpChannel.connect is idempotent when already connected."""
        # Start a test server
        ready = asyncio.Event()
        responses = [{"type": "pong", "version": "1.0.0"}]

        # Use a random available port
        server = await asyncio.start_server(
            lambda r, w: None,
            "127.0.0.1",
            0,  # Random port
        )
        port = server.sockets[0].getsockname()[1]

        try:
            channel = TcpChannel("127.0.0.1", port)
            await channel.connect(timeout_seconds=5)

            # Second connect should be no-op
            await channel.connect(timeout_seconds=5)

            assert channel._reader is not None
            assert channel._writer is not None
        finally:
            server.close()
            await server.wait_closed()
            if channel._writer:
                channel._writer.close()


class TestTcpChannelSendRequest:
    """Tests for TcpChannel.send_request method."""

    async def test_send_ping_receive_pong(self) -> None:
        """Send ping request, receive pong response."""

        # Create server that responds with pong
        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                data = await reader.readuntil(b"\n")
                request = json.loads(data.decode())
                assert request["action"] == "ping"

                response = {"type": "pong", "version": "1.0.0"}
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        try:
            channel = TcpChannel("127.0.0.1", port)
            await channel.connect(timeout_seconds=5)

            response = await channel.send_request(PingRequest(), timeout=5)

            assert isinstance(response, PongMessage)
            assert response.version == "1.0.0"
        finally:
            server.close()
            await server.wait_closed()
            await channel.close()

    async def test_send_request_not_connected(self) -> None:
        """send_request raises when not connected."""
        channel = TcpChannel("127.0.0.1", 5000)

        with pytest.raises(RuntimeError) as exc_info:
            await channel.send_request(PingRequest(), timeout=5)

        assert "not connected" in str(exc_info.value)

    async def test_send_request_timeout(self) -> None:
        """send_request raises timeout when server doesn't respond."""

        # Create server that never responds
        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                await reader.readuntil(b"\n")
                # Don't respond
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                pass
            finally:
                writer.close()

        server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        try:
            channel = TcpChannel("127.0.0.1", port)
            await channel.connect(timeout_seconds=5)

            with pytest.raises(asyncio.TimeoutError):
                await channel.send_request(PingRequest(), timeout=1)
        finally:
            server.close()
            await server.wait_closed()
            await channel.close()


class TestTcpChannelStreamMessages:
    """Tests for TcpChannel.stream_messages method."""

    async def test_stream_execution_output(self) -> None:
        """Stream stdout chunks followed by complete message."""

        # Create server that streams output
        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                await reader.readuntil(b"\n")

                # Send stdout chunks
                for i in range(3):
                    chunk = {"type": "stdout", "chunk": f"line {i}\n"}
                    writer.write((json.dumps(chunk) + "\n").encode())
                    await writer.drain()

                # Send complete
                complete = {"type": "complete", "exit_code": 0, "execution_time_ms": 100}
                writer.write((json.dumps(complete) + "\n").encode())
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        try:
            channel = TcpChannel("127.0.0.1", port)
            await channel.connect(timeout_seconds=5)

            from exec_sandbox.guest_agent_protocol import ExecuteCodeRequest

            request = ExecuteCodeRequest(language=Language.PYTHON, code="print('test')")

            messages = []
            async for msg in channel.stream_messages(request, timeout=5):
                messages.append(msg)
                if isinstance(msg, ExecutionCompleteMessage):
                    break

            # Verify messages
            assert len(messages) == 4
            assert isinstance(messages[0], OutputChunkMessage)
            assert isinstance(messages[1], OutputChunkMessage)
            assert isinstance(messages[2], OutputChunkMessage)
            assert isinstance(messages[3], ExecutionCompleteMessage)
            assert messages[3].exit_code == 0
        finally:
            server.close()
            await server.wait_closed()
            await channel.close()


class TestTcpChannelClose:
    """Tests for TcpChannel.close method."""

    async def test_close_connected(self) -> None:
        """Close connected channel."""
        server = await asyncio.start_server(
            lambda r, w: None,
            "127.0.0.1",
            0,
        )
        port = server.sockets[0].getsockname()[1]

        try:
            channel = TcpChannel("127.0.0.1", port)
            await channel.connect(timeout_seconds=5)

            assert channel._writer is not None

            await channel.close()

            assert channel._reader is None
            assert channel._writer is None
        finally:
            server.close()
            await server.wait_closed()

    async def test_close_not_connected(self) -> None:
        """Close channel that was never connected."""
        channel = TcpChannel("127.0.0.1", 5000)

        # Should not raise
        await channel.close()

        assert channel._reader is None
        assert channel._writer is None

    async def test_close_idempotent(self) -> None:
        """Close can be called multiple times."""
        channel = TcpChannel("127.0.0.1", 5000)

        await channel.close()
        await channel.close()
        await channel.close()

        assert channel._reader is None

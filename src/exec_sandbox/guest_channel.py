"""
Guest communication channel abstraction.

Supports multiple transports using JSON newline-delimited protocol:
- Unix socket over virtio-serial (preferred, lower latency)
- TCP over virtio-net (fallback, universal compatibility)
"""

import asyncio
import contextlib
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Protocol, runtime_checkable

from pydantic import TypeAdapter

from exec_sandbox._logging import get_logger
from exec_sandbox.guest_agent_protocol import (
    ExecutionCompleteMessage,
    GuestAgentRequest,
    StreamingErrorMessage,
    StreamingMessage,
)

logger = get_logger(__name__)

# Buffer limit for asyncio readuntil() - must exceed max JSON message size
# Default asyncio limit is 64KB, but our protocol can send large output chunks
# Using 16MB to handle large outputs with JSON overhead
STREAM_BUFFER_LIMIT = 16 * 1024 * 1024  # 16MB


@runtime_checkable
class GuestChannel(Protocol):
    """Protocol for guest-host communication.

    Supports TCP transport via port forwarding.
    Uses structural typing (Protocol) instead of inheritance.
    """

    async def connect(self, timeout_seconds: int) -> None:
        """Establish connection to guest agent."""
        ...

    async def send_request(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> StreamingMessage:
        """Send JSON request, receive JSON response.

        Args:
            request: Pydantic request model (PingRequest, InstallPackagesRequest)
            timeout: Response timeout in seconds (required, no default)

        Returns:
            StreamingMessage (PongMessage for ping, ExecutionCompleteMessage for install_packages)
        """
        ...

    def stream_messages(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        """Send request, stream multiple response messages.

        For code execution, yields:
        - OutputChunkMessage: stdout/stderr chunks (batched 64KB or 50ms)
        - ExecutionCompleteMessage: final completion with exit code
        - StreamingErrorMessage: error during streaming (validation, timeout)

        Args:
            request: Pydantic request model
            timeout: Total timeout in seconds

        Yields:
            Streaming messages from guest agent
        """
        ...

    async def close(self) -> None:
        """Close the communication channel."""
        ...


class TcpChannel:
    """
    TCP-based guest communication (user-mode networking).

    Works on:
    - macOS HVF (no virtio_console kernel module needed)
    - Linux KVM (portable alternative to vsock)
    - Windows WHPX (future compatibility)

    Uses QEMU user-mode networking with port forwarding:
    Host connects to 127.0.0.1:{host_port} -> QEMU forwards to guest:5000

    Security: Host port bound to localhost only (127.0.0.1).
    """

    def __init__(self, host: str, port: int):
        """
        Args:
            host: Host IP (127.0.0.1 for security - localhost only)
            port: Host port (forwarded to guest 5000 via QEMU user-mode networking)
        """
        self.host = host
        self.port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self, timeout_seconds: int) -> None:
        """Connect to guest TCP server via port forward.

        Single connection attempt with timeout (no retry).
        Caller handles retry logic (e.g., _wait_for_guest exponential backoff).
        """
        if self._reader and self._writer:
            return

        # Connect to host port (QEMU forwards to guest:5000)
        # Use larger buffer limit to handle 100KB JSON messages from guest agent
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port, limit=STREAM_BUFFER_LIMIT),
            timeout=float(timeout_seconds),
        )

        # Enable TCP_NODELAY for low latency (disable Nagle's algorithm)
        sock = self._writer.transport.get_extra_info("socket")  # type: ignore[union-attr]
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    async def send_request(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> StreamingMessage:
        """Send JSON + newline, receive JSON + newline (same protocol as vsock/serial).

        No retry at this level - caller handles retry logic.
        """
        if not self._reader or not self._writer:
            raise RuntimeError("Channel not connected")

        try:
            # Serialize Pydantic model to JSON
            request_json = request.model_dump_json(by_alias=False, exclude_none=True) + "\n"
            self._writer.write(request_json.encode())
            await self._writer.drain()

            # Receive response (read until newline)
            response_data = await asyncio.wait_for(self._reader.readuntil(b"\n"), timeout=float(timeout))

            # Deserialize JSON to StreamingMessage
            streaming_adapter: TypeAdapter[StreamingMessage] = TypeAdapter(StreamingMessage)
            return streaming_adapter.validate_json(response_data.decode().strip())

        except TimeoutError:
            # TimeoutError means no data received in time, but connection is still valid.
            # DO NOT reset - keep connection open for retry.
            raise
        except (asyncio.IncompleteReadError, OSError, BrokenPipeError, ConnectionError):
            # Connection actually broken - reset state so caller reconnects on next attempt
            self._reader = None
            self._writer = None
            raise

    async def stream_messages(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        """Stream multiple JSON messages for code execution."""
        if not self._reader or not self._writer:
            raise RuntimeError("Channel not connected")

        # Send request
        request_json = request.model_dump_json(by_alias=False, exclude_none=True) + "\n"
        self._writer.write(request_json.encode())
        await self._writer.drain()

        # Type adapter for discriminated union
        streaming_adapter: TypeAdapter[StreamingMessage] = TypeAdapter(StreamingMessage)

        # Read and yield messages until completion or error
        while True:
            # Read next JSON line
            response_data = await asyncio.wait_for(self._reader.readuntil(b"\n"), timeout=float(timeout))
            raw_json = response_data.decode().strip()

            # Parse as streaming message
            message = streaming_adapter.validate_json(raw_json)
            yield message

            # Stop if complete or error
            if isinstance(message, (ExecutionCompleteMessage, StreamingErrorMessage)):
                break

    async def close(self) -> None:
        """Close TCP connection."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None


class UnixSocketChannel:
    """
    Unix socket-based guest communication (virtio-serial).

    Works on:
    - Linux KVM (virtio-serial device)
    - macOS HVF (virtio-serial device)
    - All platforms with virtio-serial support

    Uses QEMU virtio-serial Unix socket:
    Host connects to /tmp/serial-{hash}.sock -> QEMU forwards to guest virtio-serial
    (Path uses SHA-256 hash to stay under 108-byte UNIX socket limit)

    Benefits:
    - 20-30% lower latency than TCP
    - No network namespace issues
    - Works with containerized unprivileged execution
    - Industry standard (libvirt, qemu-ga)

    Queueing:
    - Decoupled read/write paths prevent deadlocks
    - Bounded queue (100 items) provides backpressure
    - Background worker handles QEMU throttling
    - Fail-fast (5s) if queue full
    """

    def __init__(self, socket_path: str):
        """
        Args:
            socket_path: Unix socket path (e.g., /tmp/serial-{hash}.sock)
        """
        self.socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._write_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._write_task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()

    async def connect(self, timeout_seconds: int) -> None:
        """Connect to guest via Unix socket.

        Single connection attempt with timeout (no retry).
        Caller handles retry logic (e.g., _wait_for_guest exponential backoff).
        """
        if self._reader and self._writer:
            return

        # Connect to Unix socket (QEMU forwards to guest virtio-serial)
        # Use larger buffer limit to handle 100KB JSON messages from guest agent
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_unix_connection(self.socket_path, limit=STREAM_BUFFER_LIMIT),
            timeout=float(timeout_seconds),
        )

        # Start background write worker
        self._shutdown_event.clear()
        self._write_task = asyncio.create_task(self._write_worker())

    async def _write_worker(self) -> None:
        """Background worker drains write queue with backpressure handling.

        Decouples writing from reading to prevent deadlocks when virtio-serial
        buffers are full (128 descriptors per port).
        """
        while not self._shutdown_event.is_set():
            try:
                # Wait for data or shutdown (1s timeout prevents hang)
                data = await asyncio.wait_for(self._write_queue.get(), timeout=1.0)

                # Write data and wait for buffer space (handles QEMU throttling)
                if self._writer:
                    self._writer.write(data)
                    await self._writer.drain()

                self._write_queue.task_done()

            except TimeoutError:
                continue  # Check shutdown flag
            except Exception as e:
                # Log error with full traceback (2024 best practice)
                logger.error(
                    "UnixSocketChannel write worker error - connection broken",
                    extra={"socket_path": self.socket_path, "error": str(e), "error_type": type(e).__name__},
                    exc_info=True,
                )
                # Break loop to signal connection failure (clean shutdown)
                break

    async def send_request(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> StreamingMessage:
        """Send JSON + newline, receive JSON + newline (same protocol as TCP).

        No retry at this level - caller handles retry logic.
        Queues write to prevent blocking when virtio-serial buffer full.
        """
        if not self._reader or not self._writer:
            raise RuntimeError("Channel not connected")

        # Validate write worker is alive (fail-fast if crashed)
        if self._write_task and self._write_task.done():
            # Worker crashed - get exception for diagnostics
            try:
                self._write_task.result()  # Re-raises exception if any
            except Exception as e:
                raise RuntimeError(f"Write worker crashed: {type(e).__name__}: {e}") from e
            # Worker exited cleanly (shouldn't happen unless shutdown)
            raise RuntimeError("Write worker exited unexpectedly")

        try:
            # Serialize Pydantic model to JSON
            request_json = request.model_dump_json(by_alias=False, exclude_none=True) + "\n"

            # Queue write instead of blocking (fail-fast if queue full)
            try:
                await asyncio.wait_for(self._write_queue.put(request_json.encode()), timeout=5.0)
            except TimeoutError as e:
                raise RuntimeError("Write queue full - guest agent not draining") from e

            # Receive response (read until newline)
            response_data = await asyncio.wait_for(self._reader.readuntil(b"\n"), timeout=float(timeout))

            # Deserialize JSON to StreamingMessage
            streaming_adapter: TypeAdapter[StreamingMessage] = TypeAdapter(StreamingMessage)
            return streaming_adapter.validate_json(response_data.decode().strip())

        except TimeoutError:
            # TimeoutError means no data received in time, but connection is still valid.
            # DO NOT reset - closing the socket causes QEMU to signal "host disconnected"
            # which makes the guest read EOF. Keep connection open for retry.
            # See: nested KVM timing issues where guest boots before host sends data.
            raise
        except (
            asyncio.IncompleteReadError,
            OSError,
            BrokenPipeError,
            ConnectionError,
        ):
            # Connection actually broken - reset state so caller reconnects on next attempt
            self._reader = None
            self._writer = None
            raise

    async def stream_messages(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        """Stream multiple JSON messages for code execution.

        Queues write to prevent blocking when virtio-serial buffer full.
        """
        if not self._reader or not self._writer:
            raise RuntimeError("Channel not connected")

        # Validate write worker is alive (fail-fast if crashed)
        if self._write_task and self._write_task.done():
            try:
                self._write_task.result()
            except Exception as e:
                raise RuntimeError(f"Write worker crashed: {type(e).__name__}: {e}") from e
            raise RuntimeError("Write worker exited unexpectedly")

        # Send request (queued to prevent blocking)
        request_json = request.model_dump_json(by_alias=False, exclude_none=True) + "\n"

        try:
            await asyncio.wait_for(self._write_queue.put(request_json.encode()), timeout=5.0)
        except TimeoutError as e:
            raise RuntimeError("Write queue full - guest agent not draining") from e

        # Type adapter for discriminated union
        streaming_adapter: TypeAdapter[StreamingMessage] = TypeAdapter(StreamingMessage)

        # Read and yield messages until completion or error
        while True:
            # Read next JSON line
            response_data = await asyncio.wait_for(self._reader.readuntil(b"\n"), timeout=float(timeout))
            raw_json = response_data.decode().strip()

            # Parse as streaming message
            message = streaming_adapter.validate_json(raw_json)
            yield message

            # Stop if complete or error
            if isinstance(message, (ExecutionCompleteMessage, StreamingErrorMessage)):
                break

    async def close(self) -> None:
        """Close Unix socket connection with graceful queue drain."""
        # Signal shutdown to write worker
        self._shutdown_event.set()

        # Wait for queue to drain (max 5s)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._write_queue.join(), timeout=5.0)

        # Cancel write worker
        if self._write_task and not self._write_task.done():
            self._write_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._write_task

        # Close connection
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None

    async def enqueue_write(self, data: bytes, timeout: float = 5.0) -> None:
        """Enqueue data for writing with timeout.

        Args:
            data: Bytes to write
            timeout: Queue timeout in seconds

        Raises:
            RuntimeError: If queue is full and timeout expires
        """
        try:
            await asyncio.wait_for(self._write_queue.put(data), timeout=timeout)
        except TimeoutError as e:
            raise RuntimeError("Write queue full - guest agent not draining") from e

    def is_connected(self) -> bool:
        """Check if channel is connected."""
        return self._reader is not None and self._writer is not None

    def get_reader(self) -> asyncio.StreamReader | None:
        """Get the stream reader (for direct access when needed)."""
        return self._reader


class DualPortChannel:
    """
    Dual virtio-serial port communication (command + event ports).

    Architecture:
    - Command port (host → guest): Send commands (ping, execute, cancel)
    - Event port (guest → host): Stream events (output, completion)

    Benefits:
    - Concurrent read/write: Can send commands during execution
    - Simpler protocol: No multiplexing needed
    - Independent flow control: Per-port buffers
    - Unix-like: Separate read/write channels (stdin/stdout pattern)

    Works on:
    - Linux KVM (virtio-serial device)
    - macOS HVF (virtio-serial device)
    - All platforms with virtio-serial support

    Uses QEMU virtio-serial Unix sockets:
    - Host connects to /tmp/cmd-{hash}.sock -> QEMU forwards to guest virtio-serial port 0
    - Host connects to /tmp/event-{hash}.sock -> QEMU forwards to guest virtio-serial port 1
    (Paths use SHA-256 hash to stay under 108-byte UNIX socket limit)

    Usage:
        channel = DualPortChannel(
            cmd_socket="/tmp/cmd-{hash}.sock",
            event_socket="/tmp/event-{hash}.sock"
        )

        await channel.connect(timeout_seconds=5)

        # Send command
        await channel.send_command(ExecuteCodeRequest(...))

        # Stream events (concurrent with commands)
        async for event in channel.stream_events():
            if isinstance(event, OutputChunkMessage):
                print(event.chunk)

        # Can send ping DURING execution!
        await channel.send_command(PingRequest())

        await channel.close()
    """

    def __init__(self, cmd_socket: str, event_socket: str):
        """
        Args:
            cmd_socket: Unix socket path for command port (e.g., /tmp/cmd-{hash}.sock)
            event_socket: Unix socket path for event port (e.g., /tmp/event-{hash}.sock)
        """
        self.cmd_socket = cmd_socket
        self.event_socket = event_socket
        self._cmd_channel: UnixSocketChannel = UnixSocketChannel(cmd_socket)
        self._event_channel: UnixSocketChannel = UnixSocketChannel(event_socket)

    async def connect(self, timeout_seconds: int) -> None:
        """Connect both command and event ports.

        Connects in parallel for speed. Single connection attempt with timeout (no retry).
        Caller handles retry logic (e.g., _wait_for_guest exponential backoff).
        """
        # Connect both ports in parallel for speed
        await asyncio.gather(
            self._cmd_channel.connect(timeout_seconds),
            self._event_channel.connect(timeout_seconds),
        )

    async def send_command(
        self,
        request: GuestAgentRequest,
        timeout: int = 5,
    ) -> None:
        """Send command on command port (non-blocking).

        Commands are sent via UnixSocketChannel's send_request but we don't wait for response.
        Responses come via event port.

        Args:
            request: Pydantic request model (PingRequest, ExecuteCodeRequest, etc)
            timeout: Write timeout in seconds (default: 5)

        Raises:
            RuntimeError: If channel not connected or write queue full
        """
        # Use UnixSocketChannel's queued write mechanism
        # Serialize and enqueue the command
        request_json = request.model_dump_json(by_alias=False, exclude_none=True) + "\n"
        await self._cmd_channel.enqueue_write(request_json.encode(), timeout=float(timeout))

    async def stream_events(
        self,
        timeout: int = 300,
    ) -> AsyncGenerator[StreamingMessage]:
        """Stream events from event port.

        Reads messages from event port until ExecutionCompleteMessage or StreamingErrorMessage.
        Can be called concurrently with send_command() on command port.

        Args:
            timeout: Total timeout in seconds (default: 300s = 5min)

        Yields:
            StreamingMessage: OutputChunkMessage, ExecutionCompleteMessage, StreamingErrorMessage

        Raises:
            RuntimeError: If event channel not connected
            asyncio.TimeoutError: If no message received within timeout
        """
        # Validate event channel is connected
        if not self._event_channel.is_connected():
            raise RuntimeError("Event channel not connected")

        # Type adapter for discriminated union
        streaming_adapter: TypeAdapter[StreamingMessage] = TypeAdapter(StreamingMessage)

        reader = self._event_channel.get_reader()
        if not reader:
            raise RuntimeError("Event channel reader not available")

        # Read and yield messages until completion or error
        while True:
            # Read next JSON line from event channel
            response_data = await asyncio.wait_for(reader.readuntil(b"\n"), timeout=float(timeout))
            raw_json = response_data.decode().strip()

            # Parse as streaming message
            message = streaming_adapter.validate_json(raw_json)
            yield message

            # Stop if complete or error
            if isinstance(message, (ExecutionCompleteMessage, StreamingErrorMessage)):
                break

    async def send_request(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> StreamingMessage:
        """Send command and receive single response (compatibility method).

        For simple request-response (e.g., ping), sends on command port
        and reads first response from event port.

        Args:
            request: Pydantic request model
            timeout: Total timeout in seconds

        Returns:
            First StreamingMessage from event port

        Raises:
            RuntimeError: If channels not connected
            asyncio.TimeoutError: If no response within timeout
        """
        # Send command
        await self.send_command(request, timeout=5)

        # Read first event
        async for message in self.stream_events(timeout=timeout):
            return message

        # Should never reach here (stream_events always yields at least one message)
        raise RuntimeError("No response from event port")

    def stream_messages(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        """Send request and stream multiple response messages (compatibility method).

        For code execution, sends command and yields all events until completion.

        Args:
            request: Pydantic request model
            timeout: Total timeout in seconds

        Yields:
            StreamingMessage from event port

        Raises:
            RuntimeError: If channels not connected
        """
        return self._stream_messages_impl(request, timeout)

    async def _stream_messages_impl(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        """Implementation of stream_messages (async generator)."""
        # Send command
        await self.send_command(request, timeout=5)

        # Stream all events
        async for message in self.stream_events(timeout=timeout):
            yield message

    async def close(self) -> None:
        """Close both command and event ports.

        Closes both channels in parallel with graceful queue drain.
        """
        # Close both ports in parallel
        await asyncio.gather(
            self._cmd_channel.close(),
            self._event_channel.close(),
        )


@asynccontextmanager
async def reconnecting_channel(
    channel: GuestChannel,
    connect_timeout: int = 5,
) -> AsyncGenerator[GuestChannel]:
    """Async context manager for QEMU GA standard reconnect pattern.

    Pattern: close → connect → use → (auto-cleanup on exit)
    - Ensures clean connection state (no stale data from previous sessions)
    - Guest agent expects disconnect after each command
    - Industry standard (libvirt, QEMU GA reference implementation)

    Use for:
    - Health checks (ping)
    - One-off commands
    - Any single request-response

    Don't use for:
    - Code execution (needs persistent connection for streaming)
    - Package installation (needs persistent connection for streaming)

    Usage:
        # GOOD - one-off command
        async with reconnecting_channel(vm.channel) as ch:
            response = await ch.send_request(PingRequest(), timeout=5)

        # BAD - streaming operation
        async with reconnecting_channel(vm.channel) as ch:
            async for msg in ch.stream_messages(...):  # Will break!
                pass

        # GOOD - streaming operation
        await vm.channel.connect()
        async for msg in vm.channel.stream_messages(...):
            yield msg
        await vm.channel.close()

    Args:
        channel: Guest channel to manage
        connect_timeout: Connection timeout in seconds (default: 5)

    Yields:
        Connected channel ready for use

    Reference:
        - QEMU GA docs: connect-send-disconnect per command
        - libvirt: guest-sync before every command
    """
    # Close existing connection (if any)
    await channel.close()

    # Establish fresh connection
    await channel.connect(timeout_seconds=connect_timeout)

    try:
        # Yield connected channel to caller
        yield channel
    finally:
        # Cleanup on exit (success or exception)
        # Note: Not closing here to allow persistent connections for streaming
        # Caller decides when to close based on use case
        pass

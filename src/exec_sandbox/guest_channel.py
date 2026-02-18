"""
Guest communication channel abstraction.

Uses JSON newline-delimited protocol over dual virtio-serial Unix sockets
(DualPortChannel: command + event ports). UnixSocketChannel is the
single-port building block composed by DualPortChannel.
"""

import asyncio
import contextlib
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
from exec_sandbox.socket_auth import connect_and_verify

logger = get_logger(__name__)

# Buffer limit for asyncio readuntil() - must exceed max JSON message size
# Default asyncio limit is 64KB, but our protocol can send large output chunks
# StreamReader buffer limit — controls when asyncio pauses the transport
# (TCP backpressure).  Must exceed the largest single message:
#   - File chunks: ~200KB (128KB compressed → base64 → JSON)
#   - REPL stdout: ~70KB (64KB buffer + JSON overhead)
#   - Package output: ~55KB (50KB limit + JSON overhead)
# 512KB provides >2x headroom for the largest message while keeping
# memory bounded during file transfers (prevents O(file_size) buffering).
STREAM_BUFFER_LIMIT = 512 * 1024  # 512KB

# Cached TypeAdapter for StreamingMessage discriminated union
# Performance: Avoids rebuilding validators on every message (1000s of allocations per execution)
# Pydantic TypeAdapter is expensive to construct - caching eliminates this overhead in hot paths
_STREAMING_MESSAGE_ADAPTER: TypeAdapter[StreamingMessage] = TypeAdapter(StreamingMessage)


class FileOpDispatcher:
    """Routes event messages by op_id to waiting coroutines.

    Shared multiplexer that reads from the event channel and routes file transfer
    messages by op_id to per-operation queues. Non-file messages (execution output,
    pong, errors without op_id) go to a default queue consumed by stream_events().
    """

    def __init__(self, reader: asyncio.StreamReader):
        self._reader = reader
        self._op_queues: dict[str, asyncio.Queue[StreamingMessage]] = {}
        self._default_queue: asyncio.Queue[StreamingMessage] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()  # protects _op_queues registration

    def start(self) -> None:
        """Start the dispatch loop as a background task."""
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        """Stop the dispatch loop."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def register_op(self, op_id: str) -> asyncio.Queue[StreamingMessage]:
        """Register an op_id and return its dedicated bounded queue.

        Bounded to 4 items (~800KB) to provide backpressure through the
        dispatch loop → StreamReader → TCP transport → guest agent chain.
        Prevents O(file_size) memory accumulation during reads.
        """
        async with self._lock:
            q: asyncio.Queue[StreamingMessage] = asyncio.Queue(maxsize=4)
            self._op_queues[op_id] = q
            return q

    async def unregister_op(self, op_id: str) -> None:
        """Unregister an op_id (cleanup after operation completes)."""
        async with self._lock:
            self._op_queues.pop(op_id, None)

    async def get_default_message(self, timeout: float) -> StreamingMessage:
        """Get a message from the default queue (non-file-transfer messages)."""
        return await asyncio.wait_for(self._default_queue.get(), timeout=timeout)

    async def _dispatch_loop(self) -> None:
        """Read messages from the event channel and route by op_id."""
        while True:
            try:
                data = await self._reader.readuntil(b"\n")
            except asyncio.IncompleteReadError:
                break
            msg = _STREAMING_MESSAGE_ADAPTER.validate_json(data.rstrip(b"\n"))
            op_id = getattr(msg, "op_id", None)
            if op_id:
                # Route to registered op queue, or discard if orphaned.
                # Late messages after unregister_op() must NOT pollute the
                # default queue (which feeds stream_events / execution output).
                queue = self._op_queues.get(op_id)
                if queue:
                    await queue.put(msg)
                else:
                    logger.debug(
                        "Discarding message for unregistered op_id",
                        extra={"op_id": op_id, "msg_type": type(msg).__name__},
                    )
            else:
                await self._default_queue.put(msg)


@runtime_checkable
class GuestChannel(Protocol):
    """Protocol for guest-host communication.

    Supports Unix socket and dual-port transports.
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
            request: Pydantic request model (e.g., PingRequest, ReadFileRequest, ListFilesRequest)
            timeout: Response timeout in seconds (required, no default)

        Returns:
            StreamingMessage (e.g., PongMessage, FileListMessage)
        """
        ...

    async def send_raw_request(
        self,
        data: bytes,
        timeout: int,
    ) -> StreamingMessage:
        """Send pre-serialized JSON bytes and receive single response.

        Bypasses Pydantic serialization for performance-critical paths
        (e.g., large file writes where we build the JSON frame manually).

        Args:
            data: Pre-serialized JSON bytes (must end with newline)
            timeout: Response timeout in seconds

        Returns:
            StreamingMessage parsed from response
        """
        ...

    async def enqueue_raw(self, data: bytes) -> None:
        """Send raw bytes without waiting for response (for multi-message protocols).

        Used for streaming file transfers where multiple messages are sent
        before a response is expected (header, chunks, end).
        """
        ...

    async def register_op(self, op_id: str) -> asyncio.Queue[StreamingMessage]:
        """Register an op_id for file transfer multiplexing.

        Returns a queue that will receive messages matching this op_id.
        Must call unregister_op() when done.
        """
        ...

    async def unregister_op(self, op_id: str) -> None:
        """Unregister an op_id after operation completes."""
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
    - No network namespace issues
    - Works with containerized unprivileged execution
    - Industry standard (libvirt, qemu-ga)

    Queueing:
    - Decoupled read/write paths prevent deadlocks
    - Bounded queue (4 items, ~800KB) provides backpressure for O(chunk_size) memory
    - Background worker handles QEMU throttling
    - Fail-fast (5s) if queue full
    """

    def __init__(self, socket_path: str, expected_uid: int):
        """
        Args:
            socket_path: Unix socket path (e.g., /tmp/serial-{hash}.sock)
            expected_uid: Expected UID of QEMU process for peer verification (required)
        """
        self.socket_path = socket_path
        self.expected_uid = expected_uid
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._write_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=4)
        self._write_task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()

    async def connect(self, timeout_seconds: int) -> None:
        """Connect to guest via Unix socket with mandatory peer verification.

        Single connection attempt with timeout (no retry).
        Caller handles retry logic (e.g., _wait_for_guest exponential backoff).

        Verifies the socket server (QEMU) is running as the expected user
        via SO_PEERCRED/LOCAL_PEERCRED before allowing communication.
        """
        if self._reader and self._writer:
            return

        # Connect to Unix socket with mandatory peer credential verification
        # Buffer sized for streaming zstd-compressed file chunks (see STREAM_BUFFER_LIMIT)
        self._reader, self._writer = await connect_and_verify(
            path=self.socket_path,
            expected_uid=self.expected_uid,
            timeout=float(timeout_seconds),
            buffer_limit=STREAM_BUFFER_LIMIT,
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
                # Log error with full traceback
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
        """Send JSON + newline, receive JSON + newline.

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

            # Deserialize JSON to StreamingMessage (direct bytes, no decode/strip allocation)
            return _STREAMING_MESSAGE_ADAPTER.validate_json(response_data.rstrip(b"\n"))

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

    async def send_raw_request(
        self,
        data: bytes,
        timeout: int,
    ) -> StreamingMessage:
        """Send pre-serialized JSON bytes and receive single response.

        Bypasses Pydantic serialization for performance-critical paths.
        Queues write to prevent blocking when virtio-serial buffer full.
        """
        if not self._reader or not self._writer:
            raise RuntimeError("Channel not connected")
        if not data.endswith(b"\n"):
            raise ValueError("send_raw_request data must end with newline")

        # Validate write worker is alive (fail-fast if crashed)
        if self._write_task and self._write_task.done():
            try:
                self._write_task.result()
            except Exception as e:
                raise RuntimeError(f"Write worker crashed: {type(e).__name__}: {e}") from e
            raise RuntimeError("Write worker exited unexpectedly")

        try:
            # Queue write instead of blocking (fail-fast if queue full)
            try:
                await asyncio.wait_for(self._write_queue.put(data), timeout=5.0)
            except TimeoutError as e:
                raise RuntimeError("Write queue full - guest agent not draining") from e

            response_data = await asyncio.wait_for(self._reader.readuntil(b"\n"), timeout=float(timeout))
            return _STREAMING_MESSAGE_ADAPTER.validate_json(response_data.rstrip(b"\n"))

        except TimeoutError:
            raise
        except (
            asyncio.IncompleteReadError,
            OSError,
            BrokenPipeError,
            ConnectionError,
        ):
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

        # Read and yield messages until completion or error
        while True:
            # Read next JSON line
            response_data = await asyncio.wait_for(self._reader.readuntil(b"\n"), timeout=float(timeout))

            # Parse as streaming message (direct bytes, no decode/strip allocation)
            message = _STREAMING_MESSAGE_ADAPTER.validate_json(response_data.rstrip(b"\n"))
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

    async def __aenter__(self) -> "UnixSocketChannel":
        """Enter async context manager, connecting to guest."""
        await self.connect(timeout_seconds=5)
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, closing connection."""
        await self.close()


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

    def __init__(self, cmd_socket: str, event_socket: str, expected_uid: int):
        """
        Args:
            cmd_socket: Unix socket path for command port (e.g., /tmp/cmd-{hash}.sock)
            event_socket: Unix socket path for event port (e.g., /tmp/event-{hash}.sock)
            expected_uid: Expected UID of QEMU process for peer verification (required)
        """
        self.cmd_socket = cmd_socket
        self.event_socket = event_socket
        self._cmd_channel: UnixSocketChannel = UnixSocketChannel(cmd_socket, expected_uid)
        self._event_channel: UnixSocketChannel = UnixSocketChannel(event_socket, expected_uid)
        self._dispatcher: FileOpDispatcher | None = None

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

        # Start the file operation dispatcher on the event channel (once)
        if not self._dispatcher:
            reader = self._event_channel.get_reader()
            if reader:
                self._dispatcher = FileOpDispatcher(reader)
                self._dispatcher.start()

    async def send_command(
        self,
        request: GuestAgentRequest,
        timeout: int = 5,
    ) -> None:
        """Send command on command port (non-blocking).

        Commands are serialized and queued on the command port's write worker.
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

    async def enqueue_raw(self, data: bytes) -> None:
        """Send raw bytes on command port without waiting for response."""
        await self._cmd_channel.enqueue_write(data, timeout=5.0)

    async def register_op(self, op_id: str) -> asyncio.Queue[StreamingMessage]:
        """Register an op_id for file transfer multiplexing."""
        if not self._dispatcher:
            raise RuntimeError("Dispatcher not initialized - channel not connected")
        return await self._dispatcher.register_op(op_id)

    async def unregister_op(self, op_id: str) -> None:
        """Unregister an op_id after operation completes."""
        if self._dispatcher:
            await self._dispatcher.unregister_op(op_id)

    async def stream_events(
        self,
        timeout: int = 300,
    ) -> AsyncGenerator[StreamingMessage]:
        """Stream events from event port via dispatcher.

        Reads messages from the default queue (non-file-transfer messages).
        Can be called concurrently with send_command() on command port.

        Args:
            timeout: Total timeout in seconds (default: 300s = 5min)

        Yields:
            StreamingMessage: OutputChunkMessage, ExecutionCompleteMessage, StreamingErrorMessage

        Raises:
            RuntimeError: If event channel not connected
            asyncio.TimeoutError: If no message received within timeout
        """
        if not self._dispatcher:
            raise RuntimeError("Dispatcher not initialized - channel not connected")

        # Read and yield messages until completion or error
        while True:
            message = await self._dispatcher.get_default_message(timeout=float(timeout))
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

    async def send_raw_request(
        self,
        data: bytes,
        timeout: int,
    ) -> StreamingMessage:
        """Send pre-serialized JSON bytes and receive single response.

        Enqueues raw data on command port and reads first event from event port.
        """
        if not data.endswith(b"\n"):
            raise ValueError("send_raw_request data must end with newline")

        # Enqueue raw bytes on command channel
        await self._cmd_channel.enqueue_write(data, timeout=5.0)

        # Read first event
        async for message in self.stream_events(timeout=timeout):
            return message

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
        # Stop dispatcher first
        if self._dispatcher:
            await self._dispatcher.stop()
            self._dispatcher = None

        # Close both ports in parallel
        await asyncio.gather(
            self._cmd_channel.close(),
            self._event_channel.close(),
        )

    async def __aenter__(self) -> "DualPortChannel":
        """Enter async context manager, connecting to guest."""
        await self.connect(timeout_seconds=5)
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, closing connection."""
        await self.close()


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
        # Not closing channel here — caller owns the lifecycle
        pass

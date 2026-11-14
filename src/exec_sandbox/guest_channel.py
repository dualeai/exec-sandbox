"""
Guest communication channel abstraction.

Uses JSON newline-delimited protocol over dual virtio-serial Unix sockets
(DualPortChannel: command + event ports). UnixSocketChannel is the
single-port building block composed by DualPortChannel.
"""

import asyncio
import contextlib
import socket
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Never
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import gather_or_cancel
from exec_sandbox.exceptions import CommunicationError, ErrorCode, OutcomeUnknownError
from exec_sandbox.guest_agent_protocol import (
    ExecutionCompleteMessage,
    GuestAgentRequest,
    OutputChunkMessage,
    PingRequest,
    PongMessage,
    StreamingErrorMessage,
    StreamingMessage,
)
from exec_sandbox.socket_auth import connect_and_verify
from exec_sandbox.subprocess_utils import socket_access_error

logger = get_logger(__name__)

# StreamReader buffer limit of each port socket: readuntil() fails on a line
# longer than it (the asyncio default is 64KB), and asyncio pauses the
# transport when the buffer holds more (backpressure). Must exceed the largest
# single message:
#   - File chunks: ~175KB (a 128KB chunk that does not compress → base64 → JSON)
#   - REPL stdout: ~70KB (64KB buffer + JSON overhead)
#   - Package output: ~55KB (50KB limit + JSON overhead)
# 512KB gives more than 2x headroom for the largest message and keeps memory
# bounded during file transfers (no O(file_size) buffering).
STREAM_BUFFER_LIMIT = 512 * 1024  # 512KB

# Smallest kernel send buffer of each port socket: it holds the largest frame
# (~175KB file chunk), so one send() writes it. macOS gives a Unix socket 8KB:
# a file chunk then takes about 22 event loop turns (about 7x the host CPU).
# Linux gives about 208KB: no change there.
SOCKET_SEND_BUFFER_BYTES = 192 * 1024

# Per-operation queue depth for op_id-routed message dispatch.
# Must be deep enough that slow consumers (e.g., disk I/O during file reads)
# don't stall the dispatch loop (which blocks ALL concurrent operations).
# Matches the host-side outbound write queue depth for balanced flow control.
OP_QUEUE_DEPTH = 64

# Once QEMU has exited, its chardev must close promptly.  Give the event
# dispatcher one bounded turn to drain frames that were already accepted by
# the host kernel before publishing transport failure out of band.
EVENT_TRANSPORT_DRAIN_TIMEOUT_SECONDS = 1.0

# How long UnixSocketChannel.close() lets the guest read the frames still queued (a stuck
# or paused guest reads nothing): then the socket closes at once and drops them.
CLOSE_SEND_TIMEOUT_SECONDS = 5.0

# Cached TypeAdapter for StreamingMessage discriminated union
# Performance: Avoids rebuilding validators on every message (1000s of allocations per execution)
# Pydantic TypeAdapter is expensive to construct - caching eliminates this overhead in hot paths
# pydantic >= 2.14 types TypeAdapter's argument as TypeForm[T]; pyright does not
# yet infer TypeForm from an Annotated alias, so the declared type carries T.
_STREAMING_MESSAGE_ADAPTER: TypeAdapter[StreamingMessage] = TypeAdapter(StreamingMessage)  # pyright: ignore[reportArgumentType, reportUnknownVariableType]


class StreamDeadlineExceededError(TimeoutError):
    """The channel's total streaming deadline expired. Always after the command was handed to
    the transport (stream_messages has no await before it): its outcome is unknown."""


class OperationInbox:
    """Bounded per-operation data queue with an out-of-band failure signal."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[StreamingMessage] = asyncio.Queue(maxsize=OP_QUEUE_DEPTH)
        # One-shot future carrying the failure reason as its result. It is the
        # out-of-band signal *and* the reason store, and is awaitable directly
        # by the route race (no per-call wrapper task to create and cancel).
        # All inboxes are built inside register_op under a running loop.
        self._failed: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        # One future per get() that waits: route() and fail() set them. A get() takes data
        # only with get_nowait, so a cancelled get() loses no message, and a wait makes no
        # task (a task and asyncio.wait per message cost about 25 µs of event loop time).
        self._waiters: list[asyncio.Future[None]] = []
        # time.monotonic() of the first send of the command (no wall-clock step moves
        # it); None while nothing was sent.
        self._sent_at: float | None = None

    @property
    def command_may_have_been_sent(self) -> bool:
        """Whether transport failure can no longer prove non-execution."""
        return self._sent_at is not None

    @property
    def command_sent_at(self) -> float | None:
        """time.monotonic() of the first send of the command, or None."""
        return self._sent_at

    def mark_command_sent(self) -> None:
        """Any later transport loss makes the command outcome unknown."""
        if self._sent_at is None:
            self._sent_at = time.monotonic()

    def ensure_open(self) -> None:
        """Fail before dispatch when registration raced a dead generation."""
        if self._failed.done():
            raise CommunicationError(self._failed.result(), context=None)

    async def route(self, message: StreamingMessage) -> bool:
        """Route with bounded, cancellation-safe backpressure.

        A dispatcher can arrive here just before the consumer unregisters an
        operation.  Racing the bounded put against the out-of-band failure
        future prevents that stale, full inbox from wedging every later op.
        """
        if self._failed.done():
            return False
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            pass
        else:
            self._wake_getters()
            return True

        put_task = asyncio.create_task(self._queue.put(message))
        try:
            done, _pending = await asyncio.wait(
                {put_task, self._failed},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._failed in done:
                return False
            await put_task
            self._wake_getters()
            return True
        finally:
            # Only the per-call put task is owned here; the shared failure
            # future must survive for later ops that race the same inbox.
            if not put_task.done():
                put_task.cancel()
            await asyncio.gather(put_task, return_exceptions=True)

    def fail(self, reason: str, *, discard_buffer: bool = False) -> None:
        """Publish failure without consuming a slot in the bounded data queue."""
        if self._failed.done():
            return
        if discard_buffer:
            while not self._queue.empty():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
        self._failed.set_result(reason or "event transport is closed")
        self._wake_getters()

    def _wake_getters(self) -> None:
        """Wake each waiting get(): it reads the queue again, then the failure."""
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)

    def _raise_failure(self) -> Never:
        reason = self._failed.result()
        if self._sent_at is not None:
            raise OutcomeUnknownError(
                reason, code=ErrorCode.GUEST_OUTCOME_UNKNOWN, context=None, hint=None, instance_id=None
            )
        raise CommunicationError(reason, context=None)

    async def get(self) -> StreamingMessage:
        """Receive ordered data, then raise any out-of-band transport failure."""
        while True:
            # Data written before EOF stays ahead of the failure: the queue is read first.
            try:
                return self._queue.get_nowait()
            except asyncio.QueueEmpty:
                if self._failed.done():
                    self._raise_failure()
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
            try:
                await waiter
            finally:
                self._waiters.remove(waiter)


class FileOpDispatcher:
    """Routes event messages by op_id to waiting coroutines.

    Shared multiplexer that reads from the event channel and routes messages
    by op_id to per-operation queues.  Messages without op_id (pre-parse errors)
    are logged and discarded.
    """

    def __init__(self, reader: asyncio.StreamReader):
        self._reader = reader
        self._op_queues: dict[str, OperationInbox] = {}
        self._task: asyncio.Task[None] | None = None
        # No lock: registration/unregistration/fail are await-free, so on a
        # single event loop they cannot interleave with each other or the
        # dispatch loop's _op_queues read.
        self._failure_reason: str | None = None

    def start(self) -> None:
        """Start the dispatch loop as a background task."""
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        """Stop the dispatch loop and fail the registered ops. A cancel of the caller that
        comes while the loop ends is raised once both are done, never hidden. A cancel
        older than this call (stop() in a cancel handler, as destroy_vm runs it) is not raised
        again: only a rise of the caller's cancel count counts."""
        current = asyncio.current_task()
        cancels_before = current.cancelling() if current is not None else 0
        caller_cancel: asyncio.CancelledError | None = None
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as error:
                # The loop's end from the cancel above, unless the caller was cancelled meanwhile
                if current is not None and current.cancelling() > cancels_before:
                    caller_cancel = error
            except Exception:
                logger.error("Event dispatcher stopped after an unexpected failure", exc_info=True)
            self._task = None
        self.fail_registered_ops("event dispatcher stopped")
        if caller_cancel is not None:
            raise caller_cancel

    def is_alive(self) -> bool:
        """Check if the dispatch loop task is still running."""
        return self._task is not None and not self._task.done()

    def register_op(self, op_id: str) -> OperationInbox:
        """Register an op_id and return its dedicated bounded queue."""
        inbox = OperationInbox()
        self._op_queues[op_id] = inbox
        if self._failure_reason is not None:
            inbox.fail(self._failure_reason)
        return inbox

    def unregister_op(self, op_id: str) -> None:
        """Unregister an op_id and release any dispatcher blocked on it."""
        inbox = self._op_queues.pop(op_id, None)
        if inbox is not None:
            inbox.fail("operation unregistered", discard_buffer=True)

    async def _dispatch_loop(self) -> None:
        """Read messages from the event channel and route by op_id.

        Per-operation queues preserve bounded backpressure for large file and
        output streams. Unregistration and transport failure wake a blocked
        route without consuming a queue slot.
        """
        failure_reason = "event dispatcher exited unexpectedly"
        try:
            while True:
                data = await self._reader.readuntil(b"\n")
                try:
                    msg = _STREAMING_MESSAGE_ADAPTER.validate_json(data.rstrip(b"\n"))
                except ValidationError:
                    logger.warning(
                        "Dispatch loop: failed to parse message, discarding",
                        extra={"raw": data[:200]},
                        exc_info=True,
                    )
                    continue
                op_id = getattr(msg, "op_id", None)
                if op_id is not None:
                    # Route to registered op queue, or discard if orphaned.
                    inbox = self._op_queues.get(op_id)
                    if inbox is not None:
                        await inbox.route(msg)
                    else:
                        logger.debug(
                            "Discarding message for unregistered op_id",
                            extra={"op_id": op_id, "msg_type": type(msg).__name__},
                        )
                else:
                    # Messages without op_id are errors the guest sent before it
                    # parsed the request (no op_id to echo): no consumer, so log
                    # and discard them.
                    logger.debug(
                        "Discarding message without op_id",
                        extra={"msg_type": type(msg).__name__},
                    )
        except asyncio.IncompleteReadError as error:
            failure_reason = f"event channel reached EOF with {len(error.partial)} trailing bytes"
        except asyncio.LimitOverrunError as error:
            failure_reason = f"event channel frame exceeded the reader limit: {error}"
        except OSError as error:
            detail = str(error) or type(error).__name__
            failure_reason = f"event channel failed: {detail}"
        except asyncio.CancelledError:
            failure_reason = "event dispatcher stopped"
            raise
        except Exception as error:
            failure_reason = f"event dispatcher failed: {type(error).__name__}: {error}"
            logger.error("Unexpected event dispatcher failure", exc_info=True)
        finally:
            self.fail_registered_ops(failure_reason)

    def fail_registered_ops(self, reason: str) -> None:
        """Wake every active operation exactly once when the transport dies.

        Buffered protocol messages remain ordered ahead of transport failure.
        The failure event is out-of-band and therefore cannot consume queue
        capacity or delete a valid terminal frame.
        """
        if self._failure_reason is not None:
            return
        self._failure_reason = reason
        for inbox in self._op_queues.values():  # fail() is synchronous: the dict does not change
            inbox.fail(reason)

    def fail_ops_sent_before(self, reason: str, *, sent_before: float) -> None:
        """Wake, once, each operation whose command was first sent before `sent_before`
        (time.monotonic()). The others, and later ones, are not failed."""
        for inbox in self._op_queues.values():
            sent_at = inbox.command_sent_at
            if sent_at is not None and sent_at < sent_before:
                inbox.fail(reason)

    async def wait_for_transport_termination(self, reason: str) -> None:
        """Let real socket EOF order accepted frames before peer failure.

        QEMU process exit is not itself the event-stream ordering boundary:
        the reader may already contain a complete terminal frame.  The socket
        EOF is that boundary, so normally the dispatch task drains to EOF and
        fails registered inboxes from its ``finally`` block.  A bounded
        fallback preserves liveness if a broken transport never reports EOF.
        """
        task = self._task
        if task is None:
            self.fail_registered_ops(reason)
            return
        # asyncio.wait observes the task without cancelling it on timeout (no
        # shield needed) and treats a cancelled dispatch task as done — so a
        # racing stop() cannot surface here as a spurious caller cancellation.
        done, _ = await asyncio.wait({task}, timeout=EVENT_TRANSPORT_DRAIN_TIMEOUT_SECONDS)
        if not done:
            self.fail_registered_ops(
                f"{reason}; event transport did not terminate within {EVENT_TRANSPORT_DRAIN_TIMEOUT_SECONDS:g}s"
            )


class UnixSocketChannel:
    """
    Unix socket-based guest communication (virtio-serial).

    Works on:
    - Linux KVM (virtio-serial device)
    - macOS HVF (virtio-serial device)
    - All platforms with virtio-serial support

    Uses the QEMU Unix socket of one virtio-serial port:
    Host connects to <VM working folder>/cmd.sock (or event.sock) -> QEMU forwards
    to the guest port (VmWorkingDirectory keeps the path under the socket path limit:
    108 bytes on Linux, 104 on macOS)

    Benefits:
    - No network namespace issues
    - Works with containerized unprivileged execution
    - Industry standard (libvirt, qemu-ga)

    Queueing:
    - Decoupled read/write paths prevent deadlocks
    - Bounded queue (HOST_WRITE_QUEUE_DEPTH frames) plus one worker batch in the
      transport: about 13.5 MiB in flight at most
    - Background worker batches writes before drain for throughput
    - Fail-fast (5s) if queue full
    """

    def __init__(self, socket_path: str, expected_uid: int):
        """
        Args:
            socket_path: Unix socket path (e.g., <VM working folder>/cmd.sock)
            expected_uid: Expected UID of QEMU process for peer verification (required)
        """
        self.socket_path = socket_path
        self.expected_uid = expected_uid
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        # Bounded write queue for host→guest messages. Larger values let the
        # host pipeline more chunks before backpressure stalls producers
        # (e.g. file upload read+compress), improving throughput. Memory cost:
        # HOST_WRITE_QUEUE_DEPTH slots x ~175 KB ≈ 11 MB in the queue, plus up
        # to HOST_WRITE_BATCH_FRAMES + 1 frames (~3 MB) in the transport. Must
        # stay bounded to prevent unbounded growth on slow guest drains. Paired
        # with the guest-side WRITE_QUEUE_SIZE (128) to keep both ends balanced.
        self._write_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=constants.HOST_WRITE_QUEUE_DEPTH)
        self._write_task: asyncio.Task[None] | None = None

    async def connect(self, timeout_seconds: float) -> None:
        """Connect to guest via Unix socket with mandatory peer verification.

        Single connection attempt with timeout (no retry).
        Caller handles retry logic (e.g., wait_for_guest exponential backoff).

        Verifies the socket server (QEMU) is running as the expected user
        via SO_PEERCRED/LOCAL_PEERCRED before allowing communication.

        Raises:
            ConfigError: HOST_DEPENDENCY when this process may not connect (EACCES,
                EPERM): no retry helps, the socket mode or this process's groups must change.
            SocketAuthError: the socket's peer does not run as expected_uid
            TimeoutError: no connection within timeout_seconds
            OSError: no listener yet (no socket file, connection refused)
        """
        if self.is_connected():
            return
        if self._writer is not None:  # a connection that failed (a write, its transport) goes first
            await self.close()

        # Connect to Unix socket with mandatory peer credential verification
        # Buffer sized for streaming zstd-compressed file chunks (see STREAM_BUFFER_LIMIT)
        try:
            self._reader, self._writer = await connect_and_verify(
                path=self.socket_path,
                expected_uid=self.expected_uid,
                timeout=float(timeout_seconds),
                buffer_limit=STREAM_BUFFER_LIMIT,
            )
        except PermissionError as e:
            raise socket_access_error(Path(self.socket_path), e) from e
        raise_send_buffer(self._writer.get_extra_info("socket"))

        # Start background write worker
        self._write_task = asyncio.create_task(self._write_worker())

    async def _write_worker(self) -> None:
        """Background worker drains write queue with batched drain for throughput.

        Decouples writing from reading to prevent deadlocks when virtio-serial
        buffers are full (128 descriptors per port). Batches up to
        HOST_WRITE_BATCH_FRAMES writes before a single drain() call to amortize
        backpressure overhead. Blocks
        on the queue until data arrives; close() stops it by cancelling the
        task (after draining the queue), so no wakeup poll or stop flag is
        needed. A write error is logged and breaks the loop.
        """
        while True:
            try:
                # Block until the next frame; cancellation is the stop signal.
                data = await self._write_queue.get()

                if self._writer:
                    self._writer.write(data)
                self._write_queue.task_done()

                # Greedily batch up to HOST_WRITE_BATCH_FRAMES - 1 more queued items before draining
                for _ in range(constants.HOST_WRITE_BATCH_FRAMES - 1):
                    try:
                        data = self._write_queue.get_nowait()
                        if self._writer:
                            self._writer.write(data)
                        self._write_queue.task_done()
                    except asyncio.QueueEmpty:
                        break

                # Single drain for the whole batch
                if self._writer:
                    await self._writer.drain()

            except Exception as e:
                # The end of the loop (a RuntimeError "event loop is closed") is no error
                if not (isinstance(e, RuntimeError) and "event loop is closed" in str(e).lower()):
                    logger.error(
                        "UnixSocketChannel write worker error - connection broken",
                        extra={"socket_path": self.socket_path, "error": str(e), "error_type": type(e).__name__},
                        exc_info=True,
                    )
                break  # the connection failed: is_connected() is then False

    async def close(self) -> None:
        """Close the connection. A live connection first sends the frames still queued, and
        the socket what it holds, while the guest reads them (CLOSE_SEND_TIMEOUT_SECONDS);
        else the socket closes at once (abort). A frame not sent then is dropped: it never
        goes out on a later connection. The worker stops and the socket closes on every
        path, a cancel of the caller included; the error of a lost connection is no error of
        the close."""
        writer, worker = self._writer, self._write_task
        sent = False
        try:
            if writer is not None and worker is not None and self.is_connected():
                with contextlib.suppress(TimeoutError, OSError):
                    async with asyncio.timeout(CLOSE_SEND_TIMEOUT_SECONDS):
                        await self._write_queue.join()  # the worker gave every frame to the socket
                        worker.cancel()  # cancellation is its only stop: it waits for a next frame
                        writer.transport.set_write_buffer_limits(high=0)
                        await writer.drain()  # until the socket holds nothing more to send
                        sent = True
        finally:
            self._write_task = None
            self._writer = None
            self._reader = None
            if worker is not None:
                worker.cancel()
            while not self._write_queue.empty():
                self._write_queue.get_nowait()
                self._write_queue.task_done()
            if writer is not None:
                if sent:
                    writer.close()
                else:
                    writer.transport.abort()  # drops what the socket still holds
        if worker is not None:
            await asyncio.wait({worker})  # its cancel ends it in one loop turn
        if writer is not None:
            with contextlib.suppress(OSError):
                await writer.wait_closed()

    async def enqueue_write(self, data: bytes, *, timeout: float) -> None:
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
        """Check if channel is connected and the write path is still alive.

        Without these checks, the host keeps a dead connection (a closed
        writer or a crashed write worker): connect() sees is_connected()=True,
        skips reconnection, and the next request goes into a dead socket.
        """
        if self._reader is None or self._writer is None:
            return False
        if self._writer.is_closing():
            return False
        return not (self._write_task is not None and self._write_task.done())

    def get_reader(self) -> asyncio.StreamReader | None:
        """Get the stream reader (for direct access when needed)."""
        return self._reader


def raise_send_buffer(sock: socket.socket | None) -> None:
    """Raise the socket's send buffer to SOCKET_SEND_BUFFER_BYTES when it is smaller (macOS
    gives a Unix socket 8 KiB, Linux about 208 KiB). Speed only: a socket that refuses keeps
    its size."""
    if sock is None:
        return
    with contextlib.suppress(OSError):
        if sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF) < SOCKET_SEND_BUFFER_BYTES:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_SEND_BUFFER_BYTES)


class DualPortChannel:
    """
    Dual virtio-serial port communication (command + event ports).

    Architecture:
    - Command port (host → guest): Send commands (ping, warm REPL, execute, install
      packages, file write/read/list)
    - Event port (guest → host): Stream events (replies, output, file chunks, completion),
      routed to each operation by op_id (FileOpDispatcher)

    Benefits:
    - Concurrent read/write: Can send commands during execution
    - Independent flow control: Per-port buffers
    - Unix-like: Separate read/write channels (stdin/stdout pattern)

    Works on:
    - Linux KVM (virtio-serial device)
    - macOS HVF (virtio-serial device)
    - All platforms with virtio-serial support

    Uses QEMU virtio-serial Unix sockets in the VM working folder
    (VmWorkingDirectory keeps the paths under the socket path limit):
    - Host connects to cmd.sock -> QEMU forwards to the guest port org.dualeai.cmd
    - Host connects to event.sock -> QEMU forwards to the guest port org.dualeai.event

    Usage:
        channel = DualPortChannel(
            cmd_socket=workdir.cmd_socket,
            event_socket=workdir.event_socket,
            expected_uid=expected_uid,
        )

        await channel.connect(timeout_seconds=5)

        # Request-response (auto-assigns op_id, routes reply)
        response = await channel.send_request(PingRequest(), timeout=5)

        # Stream execution output (auto op_id routing)
        async for msg in channel.stream_messages(ExecuteCodeRequest(...), timeout=30):
            if isinstance(msg, OutputChunkMessage):
                print(msg.chunk)

        await channel.close()
    """

    def __init__(self, cmd_socket: str, event_socket: str, expected_uid: int):
        """
        Args:
            cmd_socket: Unix socket path for command port (VmWorkingDirectory.cmd_socket)
            event_socket: Unix socket path for event port (VmWorkingDirectory.event_socket)
            expected_uid: Expected UID of QEMU process for peer verification (required)
        """
        self._cmd_channel: UnixSocketChannel = UnixSocketChannel(cmd_socket, expected_uid)
        self._event_channel: UnixSocketChannel = UnixSocketChannel(event_socket, expected_uid)
        self._dispatcher: FileOpDispatcher | None = None
        self._has_been_connected: bool = False

    async def connect(self, timeout_seconds: float) -> None:
        """Connect both command and event ports.

        Connects in parallel for speed. Single connection attempt with timeout (no retry).
        Caller handles retry logic (e.g., wait_for_guest exponential backoff).

        On reconnection (after a previous close(), or an L1 restore), probes the
        guest agent with a PingRequest to confirm that the guest runs and reads
        its port before the caller sends real work.

        Raises:
            What UnixSocketChannel.connect raises; on a reconnection, TimeoutError when no
            pong came after every probe attempt (GUEST_RECONNECT_PROBE_TIMEOUTS)
        """
        # Already connected and dispatcher healthy — nothing to do.
        # Dispatcher EOF is the earliest signal that the guest has disconnected.
        if (
            self._cmd_channel.is_connected()
            and self._event_channel.is_connected()
            and (self._dispatcher is None or self._dispatcher.is_alive())
        ):
            return

        await self._raw_connect(timeout_seconds)

        if self._has_been_connected:
            await self._probe_guest_ready(timeout_seconds)

        self._has_been_connected = True

    def mark_restored(self) -> None:
        """The guest of this channel was restored from a save (L1) whose ports were
        connected: its first connect() probes the guest, as a reconnection does."""
        self._has_been_connected = True

    async def _raw_connect(self, timeout_seconds: float) -> None:
        """Low-level connect: open sockets and start dispatcher."""
        # Close stale sockets: without this, connect() is a no-op (objects still exist)
        # and the new dispatcher gets a reader that immediately EOFs.
        if self._dispatcher and not self._dispatcher.is_alive():
            await self.close()

        # Both ports in parallel; a port that fails ends the other's connect
        await gather_or_cancel(
            [self._cmd_channel.connect(timeout_seconds), self._event_channel.connect(timeout_seconds)]
        )

        # Start the file operation dispatcher on the event channel
        if not self._dispatcher:
            reader = self._event_channel.get_reader()
            if reader:
                self._dispatcher = FileOpDispatcher(reader)
                self._dispatcher.start()

    async def _probe_guest_ready(self, caller_timeout: float) -> None:
        """Send PingRequests until a PongMessage confirms that the guest listens.

        The guest agent keeps its ports open, so QEMU and the guest kernel hold
        a ping until the agent reads it: a ping with no answer yet is slow, not
        lost. Each attempt sends a new ping on the same connection and waits
        longer (GUEST_RECONNECT_PROBE_TIMEOUTS); the late pong of an earlier
        ping has another op_id and is dropped. A timeout never closes the
        sockets: each close shows the guest a host disconnect. Only a
        connection that failed is closed and opened again.
        """
        attempts = len(constants.GUEST_RECONNECT_PROBE_TIMEOUTS)
        for attempt, probe_timeout in enumerate(constants.GUEST_RECONNECT_PROBE_TIMEOUTS, start=1):
            try:
                response = await asyncio.wait_for(
                    self.send_request(PingRequest(), timeout=constants.GUEST_REQUEST_TIMEOUT_SECONDS),
                    timeout=probe_timeout,
                )
            except TimeoutError:
                logger.debug(
                    "Reconnection probe got no pong yet, sending another ping",
                    extra={"attempt": attempt, "max": attempts, "timeout_seconds": probe_timeout},
                )
                continue
            except (
                OSError,
                asyncio.IncompleteReadError,
                ConnectionError,
                RuntimeError,
                CommunicationError,
                OutcomeUnknownError,
            ) as error:
                if attempt == attempts:
                    break  # no attempt left: no reconnect to pay for
                # The connection failed: only a new one can carry the next ping.
                logger.debug(
                    "Reconnection probe connection failed, reconnecting",
                    extra={"attempt": attempt, "max": attempts, "error_type": type(error).__name__},
                )
                await self.close()
                await asyncio.sleep(constants.GUEST_RECONNECT_PROBE_DELAY)
                await self._raw_connect(caller_timeout)
                continue
            if isinstance(response, PongMessage):
                logger.debug("Reconnection probe succeeded", extra={"attempt": attempt})
                return

        raise TimeoutError(f"Guest agent did not respond to ping after {attempts} attempts")

    async def enqueue_raw(self, data: bytes) -> None:
        """Send raw bytes on command port without waiting for response."""
        await self._cmd_channel.enqueue_write(data, timeout=5.0)

    async def enqueue_registered(self, inbox: OperationInbox, data: bytes) -> None:
        """Dispatch for an inbox with no await gap in outcome classification."""
        inbox.ensure_open()
        inbox.mark_command_sent()
        await self.enqueue_raw(data)

    def register_op(self, op_id: str) -> OperationInbox:
        """Register an op_id for file transfer multiplexing."""
        if not self._dispatcher:
            raise RuntimeError("Dispatcher not initialized - channel not connected")
        return self._dispatcher.register_op(op_id)

    def unregister_op(self, op_id: str) -> None:
        """Unregister an op_id after operation completes."""
        if self._dispatcher:
            self._dispatcher.unregister_op(op_id)

    async def fail_pending_operations(self, reason: str) -> None:
        """Drain accepted event frames to socket EOF after QEMU death."""
        if self._dispatcher:
            await self._dispatcher.wait_for_transport_termination(reason)

    def abort_operations(self, reason: str) -> None:
        """Fail every operation in flight, and every later one, at once: the guest can
        never answer again (QEMU paused it for good). No EOF wait: the sockets stay open.
        A dispatched operation raises OutcomeUnknownError with `reason`."""
        if self._dispatcher:
            self._dispatcher.fail_registered_ops(reason)

    def fail_operations_sent_before(self, reason: str, *, sent_before: float) -> None:
        """Fail at once every operation in flight that was sent before `sent_before`
        (time.monotonic()): the guest closed its port then. The guest drops what it read on
        that connection (and the bytes that its kernel held), and an operation is marked sent
        when it is queued: its bytes may still wait on the host side (QEMU reads nothing from
        a closed port) and run after the reopen. Its outcome is unknown: each raises
        OutcomeUnknownError with `reason`. Operations sent at or after
        `sent_before` are not failed: QEMU keeps their bytes in the socket until the guest
        opens its port again (a caller that cannot place the close passes the time it
        learned of it: the operations sent between fail too)."""
        if self._dispatcher:
            self._dispatcher.fail_ops_sent_before(reason, sent_before=sent_before)

    async def send_request(
        self,
        request: GuestAgentRequest,
        *,
        timeout: int,
    ) -> StreamingMessage:
        """Send command and receive single response via op_id-routed queue.

        Args:
            request: Pydantic request model
            timeout: Total timeout in seconds

        Returns:
            First StreamingMessage matching the op_id

        Raises:
            RuntimeError: If channels not connected, or the write queue stays full
            TimeoutError: If no response within timeout
            CommunicationError: The event transport failed before the request was queued
            OutcomeUnknownError: GUEST_OUTCOME_UNKNOWN when the event transport failed
                after the request was queued
        """
        op_id = uuid4().hex
        tagged_request = request.model_copy(update={"op_id": op_id})
        op_queue = self.register_op(op_id)  # RuntimeError when not connected
        try:
            request_json = tagged_request.model_dump_json(by_alias=False, exclude_none=True) + "\n"
            await self.enqueue_registered(op_queue, request_json.encode())
            return await asyncio.wait_for(op_queue.get(), timeout=float(timeout))
        finally:
            self.unregister_op(op_id)

    async def stream_messages(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        """Send request and stream multiple response messages.

        For code execution, sends command and yields all events until completion.
        Timeout wraps the entire stream (total wall-clock), not per-message.

        Args:
            request: Pydantic request model
            timeout: Total timeout in seconds

        Yields:
            StreamingMessage from event port

        Raises (while iterating, not at the call):
            RuntimeError: If channels not connected
            StreamDeadlineExceededError: The total deadline expired
            CommunicationError, OutcomeUnknownError: The event transport failed (as in send_request)
        """
        timeout_context = asyncio.timeout(timeout)
        try:
            # Enter the timeout first so registration and command enqueue are
            # part of the advertised total stream deadline.
            async with timeout_context:
                op_id = uuid4().hex
                tagged_request = request.model_copy(update={"op_id": op_id})
                op_queue = self.register_op(op_id)  # RuntimeError when not connected
                try:
                    request_json = tagged_request.model_dump_json(by_alias=False, exclude_none=True) + "\n"
                    await self.enqueue_registered(op_queue, request_json.encode())
                    while True:
                        message = await op_queue.get()
                        yield message
                        if isinstance(message, (ExecutionCompleteMessage, StreamingErrorMessage)):
                            break
                finally:
                    self.unregister_op(op_id)
        except TimeoutError as error:
            if timeout_context.expired():
                raise StreamDeadlineExceededError(f"stream exceeded {timeout}s total deadline") from error
            raise

    async def close(self) -> None:
        """Close both command and event ports.

        Stops the dispatcher first, then closes both channels in parallel with graceful queue
        drain. The ports close also when the stop raises (a cancel of the caller).
        """
        try:
            if self._dispatcher:
                try:
                    await self._dispatcher.stop()
                finally:
                    self._dispatcher = None
        finally:
            # Each close runs to its end, also when the other fails; then the first error
            closed = await asyncio.gather(
                self._cmd_channel.close(), self._event_channel.close(), return_exceptions=True
            )
            for result in closed:
                if isinstance(result, BaseException):
                    raise result


@dataclass
class StreamResult:
    """Collected output from a guest agent streaming execution.

    Returned by consume_stream() after processing all messages from
    stream_messages(). Collects stdout/stderr chunks, timing data,
    and final exit code.
    """

    exit_code: int
    stdout: str
    stderr: str
    execution_time_ms: int | None
    spawn_ms: int | None
    process_ms: int | None


async def consume_stream(
    channel: DualPortChannel,
    request: GuestAgentRequest,
    *,
    timeout: int,
    vm_id: str,
    on_stdout: Callable[[str], None] | None,
    on_stderr: Callable[[str], None] | None,
    on_error: Callable[[StreamingErrorMessage], Awaitable[None]] | None,
) -> StreamResult:
    """Send request via channel, consume the streaming response, return collected result.

    Handles OutputChunkMessage, ExecutionCompleteMessage, and StreamingErrorMessage.
    Callbacks (on_stdout, on_stderr) are defensively wrapped — exceptions disable them.
    on_error is awaited for StreamingErrorMessage before the default handling (which
    appends the error to stderr and sets exit_code=-1). If on_error raises, the
    exception propagates immediately.

    Args:
        channel: Channel to stream from.
        request: Request to send.
        timeout: Hard timeout for the entire stream (seconds).
        vm_id: VM identifier for logging.
        on_stdout: Optional callback for stdout chunks.
        on_stderr: Optional callback for stderr chunks.
        on_error: Optional async callback for StreamingErrorMessage.
            Called before default handling. If it raises, the exception propagates.

    Returns:
        StreamResult with collected output, exit code, and timing.
    """
    exit_code = -1
    execution_time_ms: int | None = None
    spawn_ms: int | None = None
    process_ms: int | None = None
    chunks: dict[str, list[str]] = {"stdout": [], "stderr": []}
    callbacks = {"stdout": on_stdout, "stderr": on_stderr}

    async with contextlib.aclosing(channel.stream_messages(request, timeout=timeout)) as stream:
        async for msg in stream:
            if isinstance(msg, OutputChunkMessage):
                chunks[msg.type].append(msg.chunk)
                if (callback := callbacks[msg.type]) is not None:
                    try:
                        callback(msg.chunk)
                    except Exception:
                        logger.warning(
                            f"on_{msg.type} callback raised, disabling",
                            extra={"vm_id": vm_id},
                            exc_info=True,
                        )
                        callbacks[msg.type] = None

                logger.debug(
                    "VM output",
                    extra={"vm_id": vm_id, "stream": msg.type, "chunk": msg.chunk[:200]},
                )

            elif isinstance(msg, ExecutionCompleteMessage):
                exit_code = msg.exit_code
                execution_time_ms = msg.execution_time_ms
                spawn_ms = msg.spawn_ms
                process_ms = msg.process_ms
                break

            elif isinstance(msg, StreamingErrorMessage):
                # Let caller handle error first (may raise)
                if on_error is not None:
                    await on_error(msg)
                # Default handling: append error to stderr, exit_code stays -1
                chunks["stderr"].append(f"[{msg.error_type}] {msg.message}")
                break

    return StreamResult(
        exit_code=exit_code,
        stdout="".join(chunks["stdout"]),
        stderr="".join(chunks["stderr"]),
        execution_time_ms=execution_time_ms,
        spawn_ms=spawn_ms,
        process_ms=process_ms,
    )

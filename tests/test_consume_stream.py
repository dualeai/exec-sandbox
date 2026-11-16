"""Tests for consume_stream() in guest_channel.py.

Unit tests using a mock channel that yields predefined message sequences.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

import pytest

from exec_sandbox.exceptions import ErrorCode, InternalError, OutcomeUnknownError
from exec_sandbox.guest_agent_protocol import (
    ExecutionCompleteMessage,
    GuestAgentRequest,
    OutputChunkMessage,
    PingRequest,
    StreamingErrorMessage,
    StreamingMessage,
)
from exec_sandbox.guest_channel import StreamResult, consume_stream

# ---------------------------------------------------------------------------
# Mock channels
# ---------------------------------------------------------------------------


class _MockChannel:
    """Minimal mock implementing DualPortChannel.stream_messages for testing: yields each
    message in turn, and raises an exception that stands in the list in its place."""

    def __init__(self, messages: list[StreamingMessage | BaseException]) -> None:
        self._messages = messages

    async def stream_messages(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        for msg in self._messages:
            if isinstance(msg, BaseException):
                raise msg
            yield msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _complete(
    exit_code: int = 0,
    execution_time_ms: int = 100,
    spawn_ms: int | None = None,
    process_ms: int | None = None,
) -> ExecutionCompleteMessage:
    """Helper to build ExecutionCompleteMessage with required fields."""
    return ExecutionCompleteMessage(
        exit_code=exit_code, execution_time_ms=execution_time_ms, spawn_ms=spawn_ms, process_ms=process_ms
    )


async def _default_handling(_msg: StreamingErrorMessage) -> None:
    """An on_error that returns: the default handling of the error runs."""


async def _consume(messages: list[StreamingMessage | BaseException], **callbacks: Any) -> StreamResult:
    """Shorthand: build _MockChannel, call consume_stream with test defaults (no output
    callback, _default_handling, but those in `callbacks`)."""
    return await consume_stream(
        _MockChannel(messages),  # type: ignore[arg-type]
        PingRequest(host_entropy=None),
        timeout=10,
        vm_id="test-vm",
        **({"on_stdout": None, "on_stderr": None, "on_error": _default_handling} | callbacks),
    )


# ---------------------------------------------------------------------------
# Normal execution flow
# ---------------------------------------------------------------------------


class TestConsumeStreamNormal:
    """Normal execution flow tests."""

    async def test_stdout_and_stderr_collected(self) -> None:
        """stdout and stderr chunks are collected and joined."""
        result = await _consume(
            [
                OutputChunkMessage(type="stdout", chunk="hello "),
                OutputChunkMessage(type="stdout", chunk="world"),
                OutputChunkMessage(type="stderr", chunk="warn: something"),
                _complete(exit_code=0),
            ]
        )

        assert result.exit_code == 0
        assert result.stdout == "hello world"
        assert result.stderr == "warn: something"

    async def test_timing_fields_captured(self) -> None:
        """ExecutionCompleteMessage timing fields propagate to StreamResult."""
        result = await _consume(
            [
                _complete(exit_code=42, execution_time_ms=1234, spawn_ms=10, process_ms=1200),
            ]
        )

        assert result.exit_code == 42
        assert result.execution_time_ms == 1234
        assert result.spawn_ms == 10
        assert result.process_ms == 1200

    async def test_empty_stream_returns_defaults(self) -> None:
        """Empty stream returns exit_code=-1 and empty strings."""
        result = await _consume([])

        assert result.exit_code == -1
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.execution_time_ms is None

    async def test_callbacks_invoked(self) -> None:
        """on_stdout and on_stderr callbacks are called with chunks."""
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        result = await _consume(
            [
                OutputChunkMessage(type="stdout", chunk="out1"),
                OutputChunkMessage(type="stderr", chunk="err1"),
                _complete(exit_code=0),
            ],
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert stdout_chunks == ["out1"]
        assert stderr_chunks == ["err1"]
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Callback failure handling
# ---------------------------------------------------------------------------


class TestConsumeStreamOutputCap:
    """The guest caps stdout at 1,000,000 bytes and stderr at 100,000 (then OUTPUT_LIMIT): it
    sends at most its stdout cap, and its stderr cap and the 8 KiB read that crosses it. A
    character takes at least one byte. More characters is a guest that breaks its caps: the
    host keeps no more of it."""

    @pytest.mark.parametrize(("stream", "cap"), [("stdout", 1_008_192), ("stderr", 108_192)])
    async def test_the_cap_is_kept_whole(self, stream: str, cap: int) -> None:
        """Output up to the cap and one read, in characters of 2 bytes (é) and 64 KB chunks:
        all of it kept."""
        text = "é" * cap
        chunks = [text[i : i + 65_536] for i in range(0, cap, 65_536)]
        result = await _consume([*(OutputChunkMessage(type=stream, chunk=c) for c in chunks), _complete(exit_code=0)])

        assert getattr(result, stream) == text

    async def test_both_streams_at_their_caps_are_kept(self) -> None:
        """One exec with both streams at their caps and one read, interleaved: each stream has
        its own count, so all of it is kept."""
        stdout, stderr = "o" * 1_008_192, "e" * 108_192
        messages: list[StreamingMessage] = []
        for i in range(0, 1_008_192, 65_536):
            messages.append(OutputChunkMessage(type="stdout", chunk=stdout[i : i + 65_536]))
            if i < 108_192:
                messages.append(OutputChunkMessage(type="stderr", chunk=stderr[i : i + 65_536]))
        result = await _consume([*messages, _complete(exit_code=0)])

        assert (result.stdout, result.stderr) == (stdout, stderr)

    @pytest.mark.parametrize(
        ("stream", "cap", "message"),
        [
            ("stdout", 1_008_192, "1008192 characters of stdout (its cap: 1000000 bytes and one 8192-byte read)"),
            ("stderr", 108_192, "108192 characters of stderr (its cap: 100000 bytes and one 8192-byte read)"),
        ],
    )
    async def test_one_character_more_is_a_protocol_error(self, stream: str, cap: int, message: str) -> None:
        """One character past the cap raises GUEST_PROTOCOL before the host keeps or passes on
        that chunk."""
        seen: list[str] = []
        messages = [OutputChunkMessage(type=stream, chunk="a" * cap), OutputChunkMessage(type=stream, chunk="b")]
        with pytest.raises(InternalError, check=lambda e: e.code is ErrorCode.GUEST_PROTOCOL) as exc_info:
            await _consume([*messages, _complete(exit_code=0)], **{f"on_{stream}": seen.append})

        assert str(exc_info.value) == f"The guest sent more than {message}"
        assert seen == ["a" * cap]


class TestConsumeStreamCallbackFailures:
    """Callback failure handling tests."""

    async def test_stdout_callback_exception_disables_it(self) -> None:
        """on_stdout that raises is disabled; streaming continues."""
        call_count = 0

        def bad_callback(chunk: str) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("callback bug")

        result = await _consume(
            [
                OutputChunkMessage(type="stdout", chunk="first"),
                OutputChunkMessage(type="stdout", chunk="second"),
                _complete(exit_code=0),
            ],
            on_stdout=bad_callback,
        )

        # Callback called once (then disabled), but both chunks collected
        assert call_count == 1
        assert result.stdout == "firstsecond"
        assert result.exit_code == 0

    async def test_stderr_callback_exception_disables_it(self) -> None:
        """on_stderr that raises is disabled; streaming continues."""
        call_count = 0

        def bad_callback(chunk: str) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("callback bug")

        result = await _consume(
            [
                OutputChunkMessage(type="stderr", chunk="err1"),
                OutputChunkMessage(type="stderr", chunk="err2"),
                _complete(exit_code=0),
            ],
            on_stderr=bad_callback,
        )

        assert call_count == 1
        assert result.stderr == "err1err2"


# ---------------------------------------------------------------------------
# StreamingErrorMessage handling
# ---------------------------------------------------------------------------


class TestConsumeStreamErrors:
    """StreamingErrorMessage handling tests."""

    async def test_error_callback_that_raises_propagates(self) -> None:
        """on_error that raises, also after some output: its exception propagates."""

        async def raising_callback(msg: StreamingErrorMessage) -> None:
            raise RuntimeError(f"fatal: {msg.message}")

        with pytest.raises(RuntimeError, match="fatal: bad frame"):
            await _consume(
                [
                    OutputChunkMessage(type="stdout", chunk="partial"),
                    StreamingErrorMessage(error_type="protocol", message="bad frame"),
                ],
                on_error=raising_callback,
            )

    async def test_error_callback_that_returns_uses_default(self) -> None:
        """on_error gets the error message; when it returns, the default handling runs: the
        error is appended to stderr as "[type] message", and exit_code stays -1."""
        error = StreamingErrorMessage(error_type="execution", message="process crashed")
        received: list[StreamingErrorMessage] = []

        async def recording_callback(msg: StreamingErrorMessage) -> None:
            received.append(msg)

        result = await _consume(
            [
                OutputChunkMessage(type="stdout", chunk="partial"),
                OutputChunkMessage(type="stderr", chunk="warn\n"),
                error,
            ],
            on_error=recording_callback,
        )

        assert received == [error]
        assert result.exit_code == -1
        assert result.stdout == "partial"
        assert result.stderr == "warn\n[execution] process crashed"

    async def test_break_after_execution_complete(self) -> None:
        """Messages after ExecutionCompleteMessage are not processed."""
        result = await _consume(
            [
                _complete(exit_code=0),
                # These should never be reached due to break
                OutputChunkMessage(type="stdout", chunk="stale"),
                StreamingErrorMessage(error_type="protocol", message="stale error"),
            ]
        )

        assert result.exit_code == 0
        assert result.stdout == ""
        assert result.stderr == ""


# ---------------------------------------------------------------------------
# Transport / protocol errors — must propagate (not be swallowed)
#
# consume_stream() catches no error of the stream: each one reaches its caller
# (QemuVM.execute) as it is, and the caller maps it to the error of the call.
# Swallowed here, it would become a misleading exit_code=-1 result for a dead VM.
# ---------------------------------------------------------------------------


def _unreadable_frame() -> OutcomeUnknownError:
    """What the channel raises when the guest answers an exec with a frame this release cannot
    read (OperationInbox.fail_unreadable on a request that changes state)."""
    return OutcomeUnknownError(
        "The guest agent sent a reply this release cannot read ('bad': extra_forbidden).",
        code=ErrorCode.GUEST_OUTCOME_UNKNOWN,
        context=None,
        hint=None,
        instance_id=None,
    )


class TestConsumeStreamTransportErrors:
    """Transport and protocol errors reach the caller as they are.

    When QEMU dies, the socket breaks, or the guest sends a frame this release cannot read,
    the exception propagates through consume_stream(): QemuVM.execute() maps it to the
    error of the call (its Raises section). TimeoutError and CancelledError propagate too:
    execute() turns a timeout into a TransientError or an OutcomeUnknownError, and a cancel
    must reach the task.
    """

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(lambda: asyncio.IncompleteReadError(b"leftover", 1024), id="socket-eof"),
            pytest.param(lambda: ConnectionResetError("Connection reset by peer"), id="qemu-sigkill"),
            pytest.param(lambda: BrokenPipeError("Broken pipe"), id="write-to-dead-socket"),
            pytest.param(lambda: OSError(22, "Invalid argument"), id="os-error"),
            pytest.param(lambda: asyncio.LimitOverrunError("Separator not found, buffer full", 524288), id="oversized"),
            pytest.param(_unreadable_frame, id="unreadable-frame"),
            pytest.param(lambda: TimeoutError("hard timeout"), id="timeout"),
            pytest.param(asyncio.CancelledError, id="cancelled"),
        ],
    )
    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([], id="at-once"),
            pytest.param(
                [
                    OutputChunkMessage(type="stdout", chunk="partial out"),
                    OutputChunkMessage(type="stderr", chunk="partial err"),
                ],
                id="after-output",
            ),
        ],
    )
    async def test_error_propagates(self, error: Callable[[], BaseException], messages: list[StreamingMessage]) -> None:
        """The error of the stream reaches the caller as it is, with or without output before it."""
        raised = error()
        with pytest.raises(type(raised)) as caught:
            await _consume([*messages, raised])
        assert caught.value is raised

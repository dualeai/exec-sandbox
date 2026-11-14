"""Tests for consume_stream() in guest_channel.py.

Unit tests using a mock channel that yields predefined message sequences.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

import pytest
from pydantic import ValidationError

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
    """Minimal mock implementing DualPortChannel.stream_messages for testing."""

    def __init__(self, messages: list[StreamingMessage]) -> None:
        self._messages = messages

    async def stream_messages(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        for msg in self._messages:
            yield msg


class _ErrorAfterMessagesChannel:
    """Yields predefined messages then raises a given exception."""

    def __init__(self, messages: list[StreamingMessage], error: BaseException) -> None:
        self._messages = messages
        self._error = error

    async def stream_messages(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        for msg in self._messages:
            yield msg
        raise self._error


class _ImmediateErrorChannel:
    """Raises immediately without yielding any messages."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def stream_messages(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        raise self._error
        yield  # type: ignore[unreachable]  # make it an async generator


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


async def _consume(messages: list[StreamingMessage], **callbacks: Any) -> StreamResult:
    """Shorthand: build _MockChannel, call consume_stream with test defaults (no callback
    but those in `callbacks`)."""
    return await consume_stream(
        _MockChannel(messages),  # type: ignore[arg-type]
        PingRequest(),
        timeout=10,
        vm_id="test-vm",
        **({"on_stdout": None, "on_stderr": None, "on_error": None} | callbacks),
    )


async def _consume_error(
    error: BaseException,
    messages: list[StreamingMessage] | None = None,
) -> StreamResult:
    """Shorthand: build error channel, call consume_stream with test defaults."""
    if messages:
        channel: _ErrorAfterMessagesChannel | _ImmediateErrorChannel = _ErrorAfterMessagesChannel(messages, error)
    else:
        channel = _ImmediateErrorChannel(error)
    return await consume_stream(
        channel,  # type: ignore[arg-type]
        PingRequest(),
        timeout=10,
        vm_id="test-vm",
        on_stdout=None,
        on_stderr=None,
        on_error=None,
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

    async def test_error_without_callback_uses_default(self) -> None:
        """StreamingErrorMessage with no on_error: appends to stderr, exit_code=-1."""
        result = await _consume(
            [
                OutputChunkMessage(type="stdout", chunk="partial"),
                StreamingErrorMessage(error_type="execution", message="process crashed"),
            ]
        )

        assert result.exit_code == -1
        assert result.stdout == "partial"
        assert result.stderr == "[execution] process crashed"

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
        """on_error that returns normally: default handling runs."""
        callback_called = False

        async def logging_callback(msg: StreamingErrorMessage) -> None:
            nonlocal callback_called
            callback_called = True

        result = await _consume(
            [
                StreamingErrorMessage(error_type="execution", message="soft error"),
            ],
            on_error=logging_callback,
        )

        assert callback_called
        assert result.exit_code == -1
        assert result.stderr == "[execution] soft error"

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
# These exceptions propagate through consume_stream() to execute(), which
# wraps them as TransientError GUEST_UNAVAILABLE. The centralized retry logic then handles
# retries. Swallowing them here would prevent retry and return a misleading
# exit_code=-1 result for a dead VM.
# ---------------------------------------------------------------------------


def _validation_error() -> ValidationError:
    """A real ValidationError: the guest sent JSON that is not a message."""
    try:
        ExecutionCompleteMessage.model_validate({"bad": "data"})
    except ValidationError as error:
        return error
    raise AssertionError("the bad message validated")


class TestConsumeStreamTransportErrors:
    """Transport and protocol errors propagate for retry by the caller.

    When QEMU dies, the socket breaks, or the guest sends garbage,
    the exception must propagate through consume_stream() so execute()
    can wrap it as TransientError GUEST_UNAVAILABLE and the retry infrastructure can act.
    TimeoutError and CancelledError propagate too: execute() turns a timeout into a
    TransientError or an OutcomeUnknownError, and a cancel must reach the task.
    """

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(lambda: asyncio.IncompleteReadError(b"leftover", 1024), id="socket-eof"),
            pytest.param(lambda: ConnectionResetError("Connection reset by peer"), id="qemu-sigkill"),
            pytest.param(lambda: BrokenPipeError("Broken pipe"), id="write-to-dead-socket"),
            pytest.param(lambda: OSError(22, "Invalid argument"), id="os-error"),
            pytest.param(lambda: asyncio.LimitOverrunError("Separator not found, buffer full", 524288), id="oversized"),
            pytest.param(_validation_error, id="garbage-json"),
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
            await _consume_error(raised, messages=messages)
        assert caught.value is raised

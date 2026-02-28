"""Tests for consume_stream() in guest_channel.py.

Unit tests using a mock GuestChannel that yields predefined message sequences.
"""

import asyncio
from collections.abc import AsyncGenerator
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
    """Minimal mock implementing GuestChannel.stream_messages for testing."""

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


async def _consume(messages: list[StreamingMessage], **kwargs: Any) -> StreamResult:
    """Shorthand: build _MockChannel, call consume_stream with test defaults."""
    return await consume_stream(
        _MockChannel(messages),  # type: ignore[arg-type]
        PingRequest(),
        timeout=10,
        vm_id="test-vm",
        **kwargs,
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
        assert "[execution] process crashed" in result.stderr

    async def test_error_callback_that_raises_propagates(self) -> None:
        """on_error that raises: exception propagates, default handling skipped."""

        async def raising_callback(msg: StreamingErrorMessage) -> None:
            raise RuntimeError(f"fatal: {msg.message}")

        with pytest.raises(RuntimeError, match="fatal: bad frame"):
            await _consume(
                [
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
        assert "[execution] soft error" in result.stderr

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
# wraps them as VmTransientError. The centralized retry logic then handles
# retries. Swallowing them here would prevent retry and return a misleading
# exit_code=-1 result for a dead VM.
# ---------------------------------------------------------------------------


class TestConsumeStreamTransportErrors:
    """Transport and protocol errors propagate for retry by the caller.

    When QEMU dies, the socket breaks, or the guest sends garbage,
    the exception must propagate through consume_stream() so execute()
    can wrap it as VmTransientError and the retry infrastructure can act.
    """

    # -- Socket / connection errors (readuntil raises) -----------------------

    async def test_incomplete_read_error_propagates(self) -> None:
        """IncompleteReadError (socket EOF) propagates — not swallowed."""
        with pytest.raises(asyncio.IncompleteReadError):
            await _consume_error(asyncio.IncompleteReadError(b"leftover", 1024))

    async def test_incomplete_read_error_after_partial_output_propagates(self) -> None:
        """IncompleteReadError after partial output still propagates."""
        with pytest.raises(asyncio.IncompleteReadError):
            await _consume_error(
                asyncio.IncompleteReadError(b"", 1024),
                messages=[
                    OutputChunkMessage(type="stdout", chunk="partial out"),
                    OutputChunkMessage(type="stderr", chunk="partial err"),
                ],
            )

    async def test_connection_reset_error_propagates(self) -> None:
        """ConnectionResetError (QEMU SIGKILL) propagates."""
        with pytest.raises(ConnectionResetError):
            await _consume_error(ConnectionResetError("Connection reset by peer"))

    async def test_broken_pipe_error_propagates(self) -> None:
        """BrokenPipeError (write to dead socket) propagates."""
        with pytest.raises(BrokenPipeError):
            await _consume_error(BrokenPipeError("Broken pipe"))

    async def test_generic_os_error_propagates(self) -> None:
        """Plain OSError propagates."""
        with pytest.raises(OSError):
            await _consume_error(OSError(22, "Invalid argument"))

    # -- Protocol corruption errors ------------------------------------------

    async def test_limit_overrun_error_propagates(self) -> None:
        """LimitOverrunError (guest sends oversized message) propagates."""
        with pytest.raises(asyncio.LimitOverrunError):
            await _consume_error(
                asyncio.LimitOverrunError("Separator not found, buffer full", 524288),
            )

    async def test_validation_error_propagates(self) -> None:
        """ValidationError (guest sends garbage JSON) propagates."""
        try:
            ExecutionCompleteMessage.model_validate({"bad": "data"})
            pytest.fail("Expected ValidationError")
        except ValidationError as real_exc:
            validation_err = real_exc

        with pytest.raises(ValidationError):
            await _consume_error(validation_err)

    # -- Edge cases: errors after partial output still propagate -------------

    async def test_connection_reset_after_many_chunks_propagates(self) -> None:
        """ConnectionResetError after many interleaved chunks propagates."""
        messages: list[StreamingMessage] = []
        for i in range(20):
            messages.append(OutputChunkMessage(type="stdout", chunk=f"out{i} "))
            messages.append(OutputChunkMessage(type="stderr", chunk=f"err{i} "))

        with pytest.raises(ConnectionResetError):
            await _consume_error(ConnectionResetError("reset"), messages=messages)

    async def test_limit_overrun_after_partial_output_propagates(self) -> None:
        """LimitOverrunError after partial output propagates."""
        with pytest.raises(asyncio.LimitOverrunError):
            await _consume_error(
                asyncio.LimitOverrunError("buffer full", 524288),
                messages=[OutputChunkMessage(type="stdout", chunk="before overrun")],
            )

    # -- Boundary: errors that must NOT be caught ----------------------------

    async def test_timeout_error_propagates(self) -> None:
        """TimeoutError propagates — execute() converts to VmBootTimeoutError."""
        with pytest.raises(TimeoutError):
            await _consume_error(TimeoutError("hard timeout"))

    async def test_timeout_error_after_partial_output_propagates(self) -> None:
        """TimeoutError after partial output still propagates."""
        with pytest.raises(TimeoutError):
            await _consume_error(
                TimeoutError("hard timeout"),
                messages=[OutputChunkMessage(type="stdout", chunk="partial")],
            )

    async def test_cancelled_error_propagates(self) -> None:
        """CancelledError propagates for task cancellation."""
        with pytest.raises(asyncio.CancelledError):
            await _consume_error(asyncio.CancelledError())

    async def test_cancelled_error_after_partial_output_propagates(self) -> None:
        """CancelledError after partial output still propagates."""
        with pytest.raises(asyncio.CancelledError):
            await _consume_error(
                asyncio.CancelledError(),
                messages=[OutputChunkMessage(type="stdout", chunk="partial")],
            )

    # -- Edge: empty / degenerate error arguments ----------------------------

    async def test_incomplete_read_error_empty_partial(self) -> None:
        """IncompleteReadError with b'' partial and expected=0 propagates."""
        with pytest.raises(asyncio.IncompleteReadError):
            await _consume_error(asyncio.IncompleteReadError(b"", 0))

    async def test_connection_error_no_args(self) -> None:
        """ConnectionError() with no args propagates."""
        with pytest.raises(ConnectionError):
            await _consume_error(ConnectionError())

    # -- on_error callback is unrelated to transport errors ------------------

    async def test_on_error_callback_exception_not_affected(self) -> None:
        """on_error exception propagates normally (transport handler not involved)."""

        async def raising_on_error(msg: StreamingErrorMessage) -> None:
            raise RuntimeError(f"on_error: {msg.message}")

        with pytest.raises(RuntimeError, match="on_error: crash"):
            await _consume(
                [
                    OutputChunkMessage(type="stdout", chunk="partial"),
                    StreamingErrorMessage(error_type="execution", message="crash"),
                ],
                on_error=raising_on_error,
            )

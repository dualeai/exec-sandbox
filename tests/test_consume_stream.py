"""Tests for consume_stream() in guest_channel.py.

Unit tests using a mock GuestChannel that yields predefined message sequences.
"""

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from exec_sandbox.guest_agent_protocol import (
    ExecutionCompleteMessage,
    GuestAgentRequest,
    OutputChunkMessage,
    PingRequest,
    StreamingErrorMessage,
    StreamingMessage,
)
from exec_sandbox.guest_channel import StreamResult, consume_stream


class _MockChannel:
    """Minimal mock implementing GuestChannel.stream_messages for testing."""

    def __init__(self, messages: list[StreamingMessage]) -> None:
        self._messages = messages

    async def connect(self, timeout_seconds: float) -> None:
        pass

    async def stream_messages(
        self,
        request: GuestAgentRequest,
        timeout: int,
    ) -> AsyncGenerator[StreamingMessage]:
        for msg in self._messages:
            yield msg


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

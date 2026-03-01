"""Tests for streaming/chunking behavior of stdout/stderr callbacks.

Verifies the streaming contract:
- Callbacks fire in real-time as chunks arrive from the guest agent
- "".join(chunks) == result.stdout/stderr (within guest-enforced limits)
- Output exceeding limits raises OutputLimitError (guest-enforced)
- Buffer boundaries (64KB flush), timing (50ms flush interval), and edge cases
"""

import time

import pytest

from exec_sandbox.constants import DEFAULT_TIMEOUT_SECONDS
from exec_sandbox.exceptions import OutputLimitError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from tests.conftest import skip_unless_hwaccel


# =============================================================================
# Streaming Contract: Normal cases
# =============================================================================
class TestStreamingContract:
    """Verify the fundamental streaming contract between callbacks and results."""

    async def test_callbacks_match_result(self, scheduler: Scheduler) -> None:
        """Joined stdout chunks must equal result.stdout for output under limits."""
        stdout_chunks: list[str] = []

        code = "\n".join(f'print("line {i}")' for i in range(100))
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        assert joined == result.stdout

    async def test_stderr_callbacks_match_result(self, scheduler: Scheduler) -> None:
        """Joined stderr chunks must equal result.stderr for output under limits."""
        stderr_chunks: list[str] = []

        code = """
import sys
for i in range(100):
    print(f"err {i}", file=sys.stderr)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stderr=stderr_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stderr_chunks)
        assert joined == result.stderr

    @skip_unless_hwaccel
    async def test_multi_chunk_via_sleep(self, scheduler: Scheduler) -> None:
        """Sleeps > 50ms flush interval force multiple chunks arriving at different times.

        Requires hwaccel: chunk count assertion depends on 50ms flush interval
        vs wall-clock timing — TCG (~5-8x slower) compresses the sleep gaps
        and batches chunks unpredictably.
        """
        chunk_times: list[float] = []

        def on_stdout(_chunk: str) -> None:
            chunk_times.append(time.monotonic())

        # 5 prints with 100ms sleep between (2x the 50ms flush interval)
        code = """
import time
for i in range(5):
    print(f"chunk {i}", flush=True)
    time.sleep(0.1)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=on_stdout,
        )

        assert result.exit_code == 0
        # Multiple chunks arrived
        assert len(chunk_times) >= 2
        # Chunks arrived at different wall-clock times (not all batched at the end)
        time_spread = chunk_times[-1] - chunk_times[0]
        assert time_spread > 0.1, f"Chunks should span >100ms, got {time_spread:.3f}s"

    async def test_both_streams_simultaneously(self, scheduler: Scheduler) -> None:
        """Both on_stdout and on_stderr fire for interleaved writes."""
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        code = """
import sys
for i in range(10):
    print(f"out {i}")
    print(f"err {i}", file=sys.stderr)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert result.exit_code == 0
        assert len(stdout_chunks) >= 1
        assert len(stderr_chunks) >= 1
        assert "".join(stdout_chunks) == result.stdout
        assert "".join(stderr_chunks) == result.stderr
        # Verify all lines present
        for i in range(10):
            assert f"out {i}" in result.stdout
            assert f"err {i}" in result.stderr

    @skip_unless_hwaccel
    async def test_streaming_delivers_before_completion(self, scheduler: Scheduler) -> None:
        """First callback fires well before execution completes (not batched at end).

        Requires hwaccel: first-chunk-before-completion gap assertion (>300ms)
        is unreliable under TCG — the 1s in-guest sleep stretches to ~5-8s,
        compressing the measurable gap below threshold.
        """
        first_chunk_time: list[float] = []

        def on_stdout(_chunk: str) -> None:
            if not first_chunk_time:
                first_chunk_time.append(time.monotonic())

        # Print immediately, then sleep 1s — first chunk must arrive during the sleep
        code = """
import time
print("early", flush=True)
time.sleep(1.0)
print("late")
"""
        start = time.monotonic()
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=on_stdout,
        )
        end = time.monotonic()

        assert result.exit_code == 0
        assert len(first_chunk_time) == 1
        first_chunk_elapsed = first_chunk_time[0] - start
        total_elapsed = end - start
        # First chunk must arrive at least 300ms before run() returns
        # (code sleeps 1s after first print, so there's a large window)
        assert first_chunk_elapsed < total_elapsed - 0.3, (
            f"First chunk at {first_chunk_elapsed:.3f}s, run finished at {total_elapsed:.3f}s — "
            f"only {total_elapsed - first_chunk_elapsed:.3f}s gap, expected >300ms"
        )


# =============================================================================
# Streaming Boundaries: Edge cases around buffer sizes and output limits
# =============================================================================
class TestStreamingBoundaries:
    """Test buffer boundary conditions and output limit enforcement."""

    async def test_output_just_under_64kb(self, scheduler: Scheduler) -> None:
        """65535 bytes fits in a single buffer flush."""
        stdout_chunks: list[str] = []

        code = """
import sys
sys.stdout.write("A" * 65535)
sys.stdout.flush()
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        assert len(joined) == 65535

    async def test_output_exactly_64kb(self, scheduler: Scheduler) -> None:
        """65536 bytes hits the exact buffer boundary."""
        stdout_chunks: list[str] = []

        code = """
import sys
sys.stdout.write("A" * 65536)
sys.stdout.flush()
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        assert len(joined) == 65536

    async def test_output_just_over_64kb_splits(self, scheduler: Scheduler) -> None:
        """65537 bytes exceeds buffer, should split into >= 2 chunks."""
        stdout_chunks: list[str] = []

        code = """
import sys
sys.stdout.write("A" * 65537)
sys.stdout.flush()
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        assert len(joined) == 65537
        assert len(stdout_chunks) >= 2

    async def test_stdout_exceeds_max_raises_output_limit(self, scheduler: Scheduler) -> None:
        """1.2MB stdout raises OutputLimitError; partial output delivered via callbacks."""
        stdout_chunks: list[str] = []
        target_size = 1_200_000  # 1.2MB — over 1MB limit

        code = f"""
import sys
sys.stdout.write("X" * {target_size})
sys.stdout.flush()
"""
        with pytest.raises(OutputLimitError, match="output_limit_error"):
            await scheduler.run(
                code=code,
                language=Language.PYTHON,
                on_stdout=stdout_chunks.append,
            )

        # Callbacks received partial output up to the limit
        joined = "".join(stdout_chunks)
        assert len(joined) <= 1_000_000 + 8192  # limit + one read buffer margin

    async def test_stderr_exceeds_max_raises_output_limit(self, scheduler: Scheduler) -> None:
        """120KB stderr raises OutputLimitError; partial output delivered via callbacks."""
        stderr_chunks: list[str] = []
        target_size = 120_000  # 120KB — over 100KB limit

        code = f"""
import sys
sys.stderr.write("E" * {target_size})
sys.stderr.flush()
"""
        with pytest.raises(OutputLimitError, match="output_limit_error"):
            await scheduler.run(
                code=code,
                language=Language.PYTHON,
                on_stderr=stderr_chunks.append,
            )

        # Callbacks received partial output up to the limit
        joined = "".join(stderr_chunks)
        assert len(joined) <= 100_000 + 8192  # limit + one read buffer margin

    async def test_no_output_callbacks_never_fire(self, scheduler: Scheduler) -> None:
        """Silent code produces no callback invocations."""
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        result = await scheduler.run(
            code="x = 42",
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert result.exit_code == 0
        assert len(stdout_chunks) == 0
        assert len(stderr_chunks) == 0

    async def test_only_stderr_no_stdout(self, scheduler: Scheduler) -> None:
        """When only stderr is written, stdout callbacks don't fire."""
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        code = """
import sys
print("error line", file=sys.stderr)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert result.exit_code == 0
        assert len(stdout_chunks) == 0
        assert len(stderr_chunks) >= 1
        assert "error line" in "".join(stderr_chunks)

    async def test_only_stdout_no_stderr(self, scheduler: Scheduler) -> None:
        """When only stdout is written, stderr callbacks don't fire."""
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        result = await scheduler.run(
            code='print("hello")',
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert result.exit_code == 0
        assert len(stdout_chunks) >= 1
        assert len(stderr_chunks) == 0

    @skip_unless_hwaccel
    async def test_stderr_without_trailing_newline(self, scheduler: Scheduler) -> None:
        """Residual stderr_line_buf is flushed even without trailing newline.

        Requires hwaccel: the code itself is trivial (~5ms), but under TCG
        the full VM lifecycle (L1 restore failure → cold boot fallback →
        REPL warm-up) can consume most of the 120s test timeout, leaving
        the guest-agent starved and causing spurious execution timeouts.
        """
        stderr_chunks: list[str] = []

        code = """
import sys
sys.stderr.write("no-newline")
sys.stderr.flush()
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stderr=stderr_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stderr_chunks)
        assert "no-newline" in joined


# =============================================================================
# Streaming Weird Cases: Unusual inputs that stress the protocol
# =============================================================================
class TestStreamingWeirdCases:
    """Test unusual inputs that might confuse the streaming protocol."""

    async def test_callback_exception_disabled_stdout_still_completes(self, scheduler: Scheduler) -> None:
        """on_stdout that raises is disabled; output still collected in result."""

        def exploding_callback(_chunk: str) -> None:
            raise RuntimeError("callback boom")

        result = await scheduler.run(
            code='print("hello from stdout")',
            language=Language.PYTHON,
            on_stdout=exploding_callback,
        )
        assert result.exit_code == 0
        assert "hello from stdout" in result.stdout

    async def test_callback_exception_disabled_stderr_still_completes(self, scheduler: Scheduler) -> None:
        """on_stderr that raises is disabled; output still collected in result."""

        def exploding_callback(_chunk: str) -> None:
            raise RuntimeError("callback boom")

        result = await scheduler.run(
            code='import sys; print("hello from stderr", file=sys.stderr)',
            language=Language.PYTHON,
            on_stderr=exploding_callback,
        )
        assert result.exit_code == 0
        assert "hello from stderr" in result.stderr

    async def test_callback_exception_on_later_chunk(self, scheduler: Scheduler) -> None:
        """Callback raises on 2nd chunk; all output still collected."""
        call_count = 0

        def explode_on_second(chunk: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("boom on second chunk")

        code = """
import sys
for i in range(5):
    print(f"line-{i}", flush=True)
"""

        def collecting_callback(chunk: str) -> None:
            stdout_chunks.append(chunk)
            explode_on_second(chunk)

        stdout_chunks: list[str] = []
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=collecting_callback,
        )
        assert result.exit_code == 0
        # All lines collected in result regardless of callback failure
        for i in range(5):
            assert f"line-{i}" in result.stdout

    async def test_extremely_long_single_line(self, scheduler: Scheduler) -> None:
        """128KB with no newlines splits into >= 2 chunks."""
        stdout_chunks: list[str] = []
        target = 128 * 1024

        code = f"""
import sys
sys.stdout.write("A" * {target})
sys.stdout.flush()
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        assert len(joined) == target
        assert len(stdout_chunks) >= 2

    async def test_output_containing_json_protocol_strings(self, scheduler: Scheduler) -> None:
        """Fake JSON protocol messages in output don't confuse parser."""
        stdout_chunks: list[str] = []

        code = r"""
import json
# Mimic guest-agent protocol messages
fake_msgs = [
    {"type": "stdout", "chunk": "fake"},
    {"type": "execution_complete", "exit_code": 0},
    {"type": "streaming_error", "message": "fake error"},
]
for msg in fake_msgs:
    print(json.dumps(msg))
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        # All fake messages should appear as plain text in stdout
        assert '"type": "stdout"' in joined
        assert '"type": "execution_complete"' in joined
        assert '"type": "streaming_error"' in joined

    async def test_various_newline_forms(self, scheduler: Scheduler) -> None:
        r"""Different newline forms (\r\n, \r, \n) are preserved verbatim."""
        stdout_chunks: list[str] = []

        code = r"""
import sys
sys.stdout.write("crlf\r\n")
sys.stdout.write("cr\r")
sys.stdout.write("lf\n")
sys.stdout.write("mixed\r\n\n\r")
sys.stdout.flush()
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        assert "crlf\r\n" in joined
        assert "lf\n" in joined
        assert "mixed\r\n\n\r" in joined

    async def test_output_with_null_bytes(self, scheduler: Scheduler) -> None:
        r"""Null bytes (\x00) in output are handled without data loss."""
        stdout_chunks: list[str] = []

        code = r"""
import sys
sys.stdout.write("before\x00after\n")
sys.stdout.flush()
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        # Null bytes should be present or replaced, but both "before" and "after" must appear
        assert "before" in joined
        assert "after" in joined

    async def test_unicode_and_emoji_output(self, scheduler: Scheduler) -> None:
        """Multi-byte UTF-8 (emoji, CJK, combining chars) not corrupted."""
        stdout_chunks: list[str] = []

        code = """
print("Hello 🌍🚀")
print("中文测试")
print("café résumé naïve")
print("a\\u0301")  # a + combining accent
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        assert joined == result.stdout
        assert "🌍🚀" in joined
        assert "中文测试" in joined
        assert "café" in joined

    async def test_alternating_small_and_large_outputs(self, scheduler: Scheduler) -> None:
        """Alternating 1B and 32KB writes maintain total integrity."""
        stdout_chunks: list[str] = []

        code = """
import sys
for i in range(10):
    sys.stdout.write("X")
    sys.stdout.write("Y" * 32768)
sys.stdout.flush()
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        expected_len = 10 * (1 + 32768)
        assert len(joined) == expected_len
        assert joined == result.stdout


# =============================================================================
# Streaming Out of Bounds: Timeouts, crashes, and isolation
# =============================================================================
class TestStreamingOutOfBounds:
    """Test streaming under abnormal execution conditions."""

    async def test_timeout_captures_partial_output(self, scheduler: Scheduler) -> None:
        """Output before timeout is captured; rest is absent."""
        stdout_chunks: list[str] = []

        timeout = DEFAULT_TIMEOUT_SECONDS // 2
        code = f"""
import time
print("before-timeout", flush=True)
time.sleep({timeout * 4})
print("after-timeout")
"""
        await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=timeout,
            on_stdout=stdout_chunks.append,
        )

        joined = "".join(stdout_chunks)
        assert "before-timeout" in joined
        assert "after-timeout" not in joined

    async def test_very_fast_code_delivers_data(self, scheduler: Scheduler) -> None:
        """Sub-5ms code still delivers output via final flush."""
        stdout_chunks: list[str] = []

        result = await scheduler.run(
            code='print("fast")',
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        assert "fast" in joined
        assert len(stdout_chunks) >= 1

    async def test_crash_after_output_captures_partial(self, scheduler: Scheduler) -> None:
        """os._exit(42) after print — pre-crash output is captured."""
        stdout_chunks: list[str] = []

        code = """
import os
print("before-crash", flush=True)
os._exit(42)
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 42
        joined = "".join(stdout_chunks)
        assert "before-crash" in joined

    async def test_high_output_rate_no_data_loss(self, scheduler: Scheduler) -> None:
        """500KB burst with no sleeps — all data received."""
        stdout_chunks: list[str] = []
        target_size = 500 * 1024

        code = f"""
import sys
sys.stdout.write("D" * {target_size})
sys.stdout.flush()
"""
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            on_stdout=stdout_chunks.append,
        )

        assert result.exit_code == 0
        joined = "".join(stdout_chunks)
        assert len(joined) == target_size
        assert joined == result.stdout

    async def test_session_streaming_isolation(self, scheduler: Scheduler) -> None:
        """Session exec 1 chunks don't leak into exec 2."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            # First execution
            chunks_1: list[str] = []
            result_1 = await session.exec(
                'print("exec-one")',
                on_stdout=chunks_1.append,
            )
            assert result_1.exit_code == 0
            joined_1 = "".join(chunks_1)

            # Second execution with fresh chunk list
            chunks_2: list[str] = []
            result_2 = await session.exec(
                'print("exec-two")',
                on_stdout=chunks_2.append,
            )
            assert result_2.exit_code == 0
            joined_2 = "".join(chunks_2)

            # No cross-contamination
            assert "exec-one" in joined_1
            assert "exec-two" not in joined_1
            assert "exec-two" in joined_2
            assert "exec-one" not in joined_2

    async def test_mixed_stdout_over_stderr_under_raises_output_limit(self, scheduler: Scheduler) -> None:
        """stdout over limit raises OutputLimitError; stderr under limit is fine."""
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        stdout_size = 1_200_000  # 1.2MB — over 1MB limit
        stderr_size = 50_000  # 50KB — well under 100KB limit

        code = f"""
import sys
sys.stdout.write("O" * {stdout_size})
sys.stdout.flush()
sys.stderr.write("E" * {stderr_size})
sys.stderr.flush()
"""
        with pytest.raises(OutputLimitError, match="output_limit_error"):
            await scheduler.run(
                code=code,
                language=Language.PYTHON,
                on_stdout=stdout_chunks.append,
                on_stderr=stderr_chunks.append,
            )

    async def test_session_reuse_after_output_limit_error(self, scheduler: Scheduler) -> None:
        """REPL session survives OutputLimitError — next exec works normally."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            # First execution: exceed stdout limit
            with pytest.raises(OutputLimitError):
                await session.exec(
                    "import sys; sys.stdout.write('X' * 1_200_000); sys.stdout.flush()",
                )

            # Second execution: normal output works (REPL preserved)
            result = await session.exec('print("after-limit")')
            assert result.exit_code == 0
            assert "after-limit" in result.stdout

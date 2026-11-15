"""Payload integrity tests - verify data authenticity from small to very large.

Tests that data transmitted through the VM execution pipeline is not:
1. Corrupted (bit flips, encoding issues)
2. Truncated (missing data)
3. Modified (extra data, reordering)

Strategy: Generate deterministic output in VM, compare hash with pre-computed expected hash.
Uses dynamic fixtures to test various payload sizes.
"""

import hashlib
from collections.abc import Callable

import pytest

from exec_sandbox.exceptions import ErrorCode, InvalidInputError
from exec_sandbox.models import ExecutionResult, Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

from .conftest import skip_unless_hwaccel

# =============================================================================
# Payload size fixtures - from tiny to max stdout limit
# =============================================================================
# IMPORTANT: stdout is limited to 1,000,000 bytes (1MB decimal), also with streaming
# callbacks: a larger output fails with InvalidInputError OUTPUT_LIMIT.

PAYLOAD_SIZES = {
    "tiny": 1,  # 1 byte
    "small": 100,  # 100 bytes
    "1kb": 1_000,  # 1 KB (decimal)
    "10kb": 10_000,  # 10 KB
    "100kb": 100_000,  # 100 KB
    "500kb": 500_000,  # 500 KB
    "900kb": 900_000,  # 900 KB (safely under limit)
    "1mb": 1_000_000,  # 1 MB (exactly at guest-agent limit, not exceeding)
}

# An output 500 times the guest-agent stdout limit (InvalidInputError OUTPUT_LIMIT expected)
OVER_THE_LIMIT_SIZE = 500_000_000  # 500 MB

# The bound of the over-the-limit run: the guest would need it to generate the whole 500MB
EXTRA_LARGE_TIMEOUT_SECONDS = 300

# Throughput thresholds (MiB/s) - CI runners have variable I/O performance,
# so we use low thresholds to catch severe regressions only
MIN_THROUGHPUT_DEVZERO_MIBPS = 1  # /dev/zero, second exec of a VM (macOS HVF: 60-72)
MIN_THROUGHPUT_URANDOM_MIBPS = 0.5  # /dev/urandom, second exec of a VM (macOS HVF: 48-97)


@pytest.fixture(params=list(PAYLOAD_SIZES.keys()))
def payload_size(request: pytest.FixtureRequest) -> tuple[str, int]:
    """Fixture providing payload sizes within stdout limit."""
    name = request.param
    return name, PAYLOAD_SIZES[name]


@pytest.fixture(params=["tiny", "small", "1kb", "10kb", "100kb"])
def small_payload_size(request: pytest.FixtureRequest) -> tuple[str, int]:
    """Fixture for smaller payloads (fast tests)."""
    name = request.param
    return name, PAYLOAD_SIZES[name]


@pytest.fixture(params=["100kb", "500kb", "900kb"])
def medium_payload_size(request: pytest.FixtureRequest) -> tuple[str, int]:
    """Fixture for medium payloads (under stdout limit)."""
    name = request.param
    return name, PAYLOAD_SIZES[name]


# =============================================================================
# Helper functions
# =============================================================================


def generate_ascii_pattern(size: int) -> bytes:
    """Generate deterministic ASCII pattern."""
    chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return bytes([chars[i % len(chars)] for i in range(size)])


def generate_sequential_bytes(size: int) -> bytes:
    """Generate sequential bytes: 0x00, 0x01, ..., 0xFF, 0x00, ..."""
    return bytes([i % 256 for i in range(size)])


def python_code_for_ascii_pattern(size: int) -> str:
    """Generate Python code that outputs ASCII pattern of given size.

    Uses chunked string multiplication to stay within VM memory limits (256MB).
    Each chunk is generated independently using the same deterministic pattern.
    """
    if size == 1:
        return 'print("A", end="")'

    # Chunked approach - generate 1MB at a time to stay within VM memory
    # Pattern must match host's generate_ascii_pattern() exactly
    return f"""
import sys
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
size = {size}
chars_len = len(chars)
chunk_size = 1024 * 1024  # 1MB chunks

written = 0
while written < size:
    # Calculate this chunk's size
    remaining = size - written
    this_chunk = min(chunk_size, remaining)

    # Build chunk using string multiplication (fast)
    # Offset ensures pattern continues correctly across chunks
    offset = written % chars_len
    rotated = chars[offset:] + chars[:offset]
    repeats = (this_chunk // chars_len) + 2
    chunk = (rotated * repeats)[:this_chunk]

    sys.stdout.write(chunk)
    written += this_chunk

sys.stdout.flush()
"""


def javascript_code_for_ascii_pattern(size: int) -> str:
    """Generate JavaScript code that outputs ASCII pattern of given size."""
    if size > 100 * 1024:
        # Chunked for large sizes
        return f"""
const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const size = {size};
const chunkSize = 64 * 1024;
for (let start = 0; start < size; start += chunkSize) {{
    const end = Math.min(start + chunkSize, size);
    let chunk = "";
    for (let i = start; i < end; i++) {{
        chunk += chars[i % chars.length];
    }}
    process.stdout.write(chunk);
}}
"""
    return f"""
const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const size = {size};
let output = "";
for (let i = 0; i < size; i++) {{
    output += chars[i % chars.length];
}}
process.stdout.write(output);
"""


# Default timeout for all payload tests
TIMEOUT_SECONDS = 60


# =============================================================================
# Python payload integrity tests
# =============================================================================


class TestPythonPayloadIntegrity:
    """Python payload integrity tests across all sizes."""

    async def test_payload_integrity(self, scheduler: Scheduler, payload_size: tuple[str, int]) -> None:
        """Verify payload integrity for given size."""
        size_name, size_bytes = payload_size

        # Pre-compute expected
        expected_data = generate_ascii_pattern(size_bytes)
        expected_hash = hashlib.sha256(expected_data).hexdigest()

        # Generate code
        code = python_code_for_ascii_pattern(size_bytes)

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=TIMEOUT_SECONDS)

        assert result.exit_code == 0, f"[{size_name}] Execution failed: {result.stderr}"

        actual_data = result.stdout.encode("utf-8")  # the code writes no newline: none comes
        actual_hash = hashlib.sha256(actual_data).hexdigest()

        assert len(actual_data) == size_bytes, f"[{size_name}] Size mismatch: {len(actual_data)} vs {size_bytes}"

        assert actual_hash == expected_hash, (
            f"[{size_name}] HASH MISMATCH - DATA CORRUPTION!\nExpected: {expected_hash}\nActual: {actual_hash}"
        )


class TestPythonStreamingIntegrity:
    """Test streaming callback integrity."""

    async def test_streaming_integrity(self, scheduler: Scheduler, medium_payload_size: tuple[str, int]) -> None:
        """Verify streaming receives uncorrupted data."""
        size_name, size_bytes = medium_payload_size

        expected_hash = hashlib.sha256(generate_ascii_pattern(size_bytes)).hexdigest()
        code = python_code_for_ascii_pattern(size_bytes)

        streamed_chunks: list[str] = []

        def on_stdout(chunk: str) -> None:
            streamed_chunks.append(chunk)

        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=TIMEOUT_SECONDS, on_stdout=on_stdout
        )

        assert result.exit_code == 0

        streamed_data = "".join(streamed_chunks).encode("utf-8")
        streamed_hash = hashlib.sha256(streamed_data).hexdigest()

        assert len(streamed_data) == size_bytes, f"[{size_name}] Streamed size: {len(streamed_data)} vs {size_bytes}"

        assert streamed_hash == expected_hash, (
            f"[{size_name}] Streamed data corrupted!\nExpected: {expected_hash}\nStreamed: {streamed_hash}"
        )

        # Verify final result matches streamed
        final_hash = hashlib.sha256(result.stdout.encode("utf-8")).hexdigest()
        assert final_hash == streamed_hash, f"[{size_name}] Final vs streamed mismatch!"


# =============================================================================
# JavaScript payload integrity tests
# =============================================================================


class TestJavaScriptPayloadIntegrity:
    """JavaScript payload integrity tests."""

    async def test_payload_integrity(self, scheduler: Scheduler, small_payload_size: tuple[str, int]) -> None:
        """Verify JavaScript payload integrity."""
        size_name, size_bytes = small_payload_size

        expected_hash = hashlib.sha256(generate_ascii_pattern(size_bytes)).hexdigest()
        code = javascript_code_for_ascii_pattern(size_bytes)

        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code, timeout_seconds=TIMEOUT_SECONDS)

        assert result.exit_code == 0, f"[JS-{size_name}] Execution failed: {result.stderr}"

        stdout = result.stdout  # the code writes no newline: none comes
        actual_hash = hashlib.sha256(stdout.encode("utf-8")).hexdigest()

        assert len(stdout) == size_bytes, f"[JS-{size_name}] Size: {len(stdout)} vs {size_bytes}"

        assert actual_hash == expected_hash, f"[JS-{size_name}] Hash mismatch - corruption!"


# =============================================================================
# Binary data integrity (via base64)
# =============================================================================


class TestBinaryPayloadIntegrity:
    """Binary data integrity tests using base64 encoding."""

    async def test_binary_integrity(self, scheduler: Scheduler, small_payload_size: tuple[str, int]) -> None:
        """Verify binary data integrity via base64."""
        size_name, size_bytes = small_payload_size

        expected_binary = generate_sequential_bytes(size_bytes)
        expected_hash = hashlib.sha256(expected_binary).hexdigest()

        code = f"""
import base64
import sys

size = {size_bytes}
data = bytes([i % 256 for i in range(size)])
encoded = base64.b64encode(data).decode("ascii")
sys.stdout.write(encoded)
sys.stdout.flush()
"""

        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=TIMEOUT_SECONDS)

        assert result.exit_code == 0

        import base64

        actual_binary = base64.b64decode(result.stdout)
        actual_hash = hashlib.sha256(actual_binary).hexdigest()

        assert len(actual_binary) == size_bytes, f"[Binary-{size_name}] Size: {len(actual_binary)} vs {size_bytes}"

        assert actual_hash == expected_hash, f"[Binary-{size_name}] Binary data corrupted!"


# =============================================================================
# RAW/Shell payload integrity
# =============================================================================


class TestRawPayloadIntegrity:
    """RAW/shell payload integrity tests."""

    @pytest.mark.parametrize(
        "repeat_char,count",
        [
            ("A", 100),
            ("A", 1024),
            ("A", 10 * 1024),
            ("X", 50 * 1024),
        ],
    )
    async def test_raw_repeated_char(self, scheduler: Scheduler, repeat_char: str, count: int) -> None:
        """Verify RAW shell can output repeated characters correctly."""
        expected_data = (repeat_char * count).encode("utf-8")
        expected_hash = hashlib.sha256(expected_data).hexdigest()

        # Use head -c for exact byte count
        code = f"yes '{repeat_char}' | tr -d '\\n' | head -c {count}"

        result = await run_once(scheduler, VmSpec(language=Language.RAW), code)

        assert result.exit_code == 0

        stdout = result.stdout  # head -c ends the output with no newline
        actual_hash = hashlib.sha256(stdout.encode("utf-8")).hexdigest()

        assert len(stdout) == count, f"[RAW-{count}] Size: {len(stdout)} vs {count}"

        assert actual_hash == expected_hash, f"[RAW-{count}] Shell output corrupted!"


# =============================================================================
# Output over the stdout limit
# =============================================================================


class TestOutputOverTheLimit:
    """An output far over the guest-agent's 1MB stdout limit (MAX_STDOUT_BYTES) fails the exec
    with InvalidInputError OUTPUT_LIMIT. An output just over it, and its callbacks:
    test_streaming.py::TestStreamingBoundaries::test_stdout_exceeds_max_raises_output_limit.
    """

    async def test_output_over_the_limit_is_output_limit(self, scheduler: Scheduler) -> None:
        """A 500MB output fails the exec with InvalidInputError OUTPUT_LIMIT (the guest stops it
        at the limit: the size of the output does not matter)."""
        code = python_code_for_ascii_pattern(OVER_THE_LIMIT_SIZE)

        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.OUTPUT_LIMIT):
            await run_once(
                scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=EXTRA_LARGE_TIMEOUT_SECONDS
            )


# =============================================================================
# Throughput Benchmark Tests
# =============================================================================
# Measure raw streaming throughput by minimizing generation overhead


async def _second_run(scheduler: Scheduler, code: str, on_stdout: Callable[[str], None]) -> ExecutionResult:
    """Run code twice in one raw VM; return the second result, whose stdout goes to on_stdout.

    The first exec of a VM pays one-time costs that are not streaming; on a restored VM, the
    first fork and the first runs of dd and base64. With them, the 100 KB urandom exec takes
    13 ms on macOS HVF and took 197 ms on a slow CI runner; its second exec takes 2 ms on
    macOS HVF.
    """
    async with await scheduler.new(VmSpec(language=Language.RAW)) as vm:
        await vm.exec(code)
        return await vm.exec(code, on_stdout=on_stdout)


@pytest.mark.slow  # MiB/s thresholds are sensitive to CI runner I/O load
@skip_unless_hwaccel
class TestStreamingThroughput:
    """Benchmark tests to measure raw streaming throughput.

    Requires hwaccel: explicit MiB/s threshold assertions — TCG (~5-8x
    slower) cannot meet the minimum throughput benchmarks.

    Uses sizes under the guest-agent's 1MB stdout limit (MAX_STDOUT_BYTES).
    Tests optimized data generation (/dev/zero, /dev/urandom) to isolate
    streaming performance from Python string manipulation overhead.
    """

    @pytest.mark.parametrize("size_kb", [500, 900])
    async def test_raw_throughput_devzero(self, scheduler: Scheduler, size_kb: int) -> None:
        """Measure throughput using /dev/zero (fastest possible generation)."""
        expected_bytes = size_kb * 1000

        # Use dd from /dev/zero - zero generation overhead
        code = f"dd if=/dev/zero bs=1000 count={size_kb} 2>/dev/null"

        streamed_bytes = 0

        def on_stdout(chunk: str) -> None:
            nonlocal streamed_bytes
            streamed_bytes += len(chunk.encode("utf-8"))

        result = await _second_run(scheduler, code, on_stdout)

        # execute_ms (host-measured) of the second exec: no VM preparation, no one-time costs
        elapsed_s = result.timing.execute_ms / 1000
        throughput_mibps = (streamed_bytes / (1024 * 1024)) / elapsed_s

        assert result.exit_code == 0, f"dd failed: {result.stderr}"
        assert streamed_bytes == expected_bytes, f"Size mismatch: {streamed_bytes} vs {expected_bytes}"

        assert throughput_mibps > MIN_THROUGHPUT_DEVZERO_MIBPS, f"Throughput too low: {throughput_mibps:.1f} MiB/s"

    @pytest.mark.parametrize("size_kb", [100, 500])
    async def test_raw_throughput_urandom(self, scheduler: Scheduler, size_kb: int) -> None:
        """Measure throughput with random data (tests full pipeline)."""
        # Use dd from /dev/urandom - realistic random data
        # base64 expands by ~33%, so keep input small enough that output < 1MB
        code = f"dd if=/dev/urandom bs=1000 count={size_kb} 2>/dev/null | base64"

        streamed_bytes = 0

        def on_stdout(chunk: str) -> None:
            nonlocal streamed_bytes
            streamed_bytes += len(chunk)

        result = await _second_run(scheduler, code, on_stdout)

        # execute_ms (host-measured) of the second exec: no VM preparation, no one-time costs
        elapsed_s = result.timing.execute_ms / 1000
        # base64 expands by ~33%, so effective raw bytes is 3/4 of output
        effective_mib = (streamed_bytes * 3 // 4) / (1024 * 1024)
        throughput_mibps = effective_mib / elapsed_s

        assert result.exit_code == 0, f"dd failed: {result.stderr}"
        assert throughput_mibps > MIN_THROUGHPUT_URANDOM_MIBPS, f"Throughput too low: {throughput_mibps:.1f} MiB/s"


# =============================================================================
# Stress Tests - Rapid Sequential Payloads
# =============================================================================


class TestStreamingStress:
    """Stress tests for streaming reliability under load."""

    async def test_rapid_small_payloads(self, scheduler: Scheduler) -> None:
        """Send many small payloads rapidly to test connection stability."""
        num_iterations = 20
        payload_size = 100_000  # 100KB each

        expected_hash = hashlib.sha256(generate_ascii_pattern(payload_size)).hexdigest()
        code = python_code_for_ascii_pattern(payload_size)

        for i in range(num_iterations):
            chunks: list[str] = []

            def make_callback(chunk_list: list[str]) -> Callable[[str], None]:
                def on_stdout(chunk: str) -> None:
                    chunk_list.append(chunk)

                return on_stdout

            result = await run_once(
                scheduler, VmSpec(language=Language.PYTHON), code, timeout_seconds=30, on_stdout=make_callback(chunks)
            )

            assert result.exit_code == 0, f"Iteration {i} failed: {result.stderr}"

            streamed_data = "".join(chunks).encode("utf-8")
            actual_hash = hashlib.sha256(streamed_data).hexdigest()

            assert actual_hash == expected_hash, f"Iteration {i} corrupted!"

    async def test_interleaved_stdout_stderr(self, scheduler: Scheduler) -> None:
        """Rapid interleaving of stdout and stderr: the callbacks and the result get each line
        of each stream once, in order."""
        # Generate alternating stdout/stderr output
        code = """
import sys
for i in range(1000):
    sys.stdout.write(f"OUT{i:04d}\\n")
    sys.stdout.flush()
    sys.stderr.write(f"ERR{i:04d}\\n")
    sys.stderr.flush()
"""

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def on_stdout(chunk: str) -> None:
            stdout_chunks.append(chunk)

        def on_stderr(chunk: str) -> None:
            stderr_chunks.append(chunk)

        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON), code, on_stdout=on_stdout, on_stderr=on_stderr
        )

        assert result.exit_code == 0, f"Interleave test failed: {result.stderr}"

        # Each stream keeps all its lines, in order (the two streams interleave in any order)
        stdout_full = "".join(stdout_chunks)
        stderr_full = "".join(stderr_chunks)
        assert stdout_full == "".join(f"OUT{i:04d}\n" for i in range(1000))
        assert stderr_full == "".join(f"ERR{i:04d}\n" for i in range(1000))
        assert (result.stdout, result.stderr) == (stdout_full, stderr_full)

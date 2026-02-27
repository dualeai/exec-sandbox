"""Payload integrity tests - verify data authenticity from small to very large.

Tests that data transmitted through the VM execution pipeline is not:
1. Corrupted (bit flips, encoding issues)
2. Truncated (missing data)
3. Modified (extra data, reordering)

Strategy: Generate deterministic output in VM, compare hash with pre-computed expected hash.
Uses dynamic fixtures to test various payload sizes.
"""

from collections.abc import Callable

import pytest

from exec_sandbox.exceptions import OutputLimitError
from exec_sandbox.hash_utils import bytes_hash
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

from .conftest import skip_unless_hwaccel

# =============================================================================
# Payload size fixtures - from tiny to max stdout limit
# =============================================================================
# IMPORTANT: stdout is limited to 1,000,000 bytes (1MB decimal).
# Tests for larger payloads must use streaming callbacks.

MAX_STDOUT_BYTES = 1_000_000  # System limit from constants.py

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

# Large sizes that exceed the guest-agent stdout limit (OutputLimitError expected)
STREAMING_ONLY_SIZES = {
    "2mb": 2_000_000,  # 2 MB
    "5mb": 5_000_000,  # 5 MB
    "10mb": 10_000_000,  # 10 MB
    "25mb": 25_000_000,  # 25 MB
    "50mb": 50_000_000,  # 50 MB
    "100mb": 100_000_000,  # 100 MB
    "200mb": 200_000_000,  # 200 MB
    "500mb": 500_000_000,  # 500 MB
}

# Extra large sizes need longer timeout (generation + transfer time)
# 500MB needs significant time for generation
EXTRA_LARGE_TIMEOUT_SECONDS = 300

# Throughput thresholds (MiB/s) - CI runners have variable I/O performance,
# so we use low thresholds to catch severe regressions only
MIN_THROUGHPUT_DEVZERO_MIBPS = 1  # /dev/zero throughput (local: ~20+, CI: ~2)
MIN_THROUGHPUT_URANDOM_MIBPS = 1  # /dev/urandom throughput (local: ~15+, CI: ~3)


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


@pytest.fixture(params=list(STREAMING_ONLY_SIZES.keys()))
def streaming_payload_size(request: pytest.FixtureRequest) -> tuple[str, int]:
    """Fixture for large payloads (streaming only - exceed stdout limit)."""
    name = request.param
    return name, STREAMING_ONLY_SIZES[name]


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
        expected_hash = bytes_hash(expected_data)

        # Generate code
        code = python_code_for_ascii_pattern(size_bytes)

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=TIMEOUT_SECONDS,
        )

        assert result.exit_code == 0, f"[{size_name}] Execution failed: {result.stderr}"

        # Strip trailing newline (runtime may add one)
        stdout = result.stdout.rstrip("\n")
        actual_data = stdout.encode("utf-8")
        actual_hash = bytes_hash(actual_data)

        assert len(actual_data) == size_bytes, f"[{size_name}] Size mismatch: {len(actual_data)} vs {size_bytes}"

        assert actual_hash == expected_hash, (
            f"[{size_name}] HASH MISMATCH - DATA CORRUPTION!\nExpected: {expected_hash}\nActual: {actual_hash}"
        )


class TestPythonStreamingIntegrity:
    """Test streaming callback integrity."""

    async def test_streaming_integrity(self, scheduler: Scheduler, medium_payload_size: tuple[str, int]) -> None:
        """Verify streaming receives uncorrupted data."""
        size_name, size_bytes = medium_payload_size

        expected_hash = bytes_hash(generate_ascii_pattern(size_bytes))
        code = python_code_for_ascii_pattern(size_bytes)

        streamed_chunks: list[str] = []

        def on_stdout(chunk: str) -> None:
            streamed_chunks.append(chunk)

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=TIMEOUT_SECONDS,
            on_stdout=on_stdout,
        )

        assert result.exit_code == 0

        # Verify streamed data (strip trailing newline)
        streamed_data = "".join(streamed_chunks).rstrip("\n").encode("utf-8")
        streamed_hash = bytes_hash(streamed_data)

        assert len(streamed_data) == size_bytes, f"[{size_name}] Streamed size: {len(streamed_data)} vs {size_bytes}"

        assert streamed_hash == expected_hash, (
            f"[{size_name}] Streamed data corrupted!\nExpected: {expected_hash}\nStreamed: {streamed_hash}"
        )

        # Verify final result matches streamed
        final_hash = bytes_hash(result.stdout.rstrip("\n").encode("utf-8"))
        assert final_hash == streamed_hash, f"[{size_name}] Final vs streamed mismatch!"


# =============================================================================
# JavaScript payload integrity tests
# =============================================================================


class TestJavaScriptPayloadIntegrity:
    """JavaScript payload integrity tests."""

    async def test_payload_integrity(self, scheduler: Scheduler, small_payload_size: tuple[str, int]) -> None:
        """Verify JavaScript payload integrity."""
        size_name, size_bytes = small_payload_size

        expected_hash = bytes_hash(generate_ascii_pattern(size_bytes))
        code = javascript_code_for_ascii_pattern(size_bytes)

        result = await scheduler.run(
            code=code,
            language=Language.JAVASCRIPT,
            timeout_seconds=TIMEOUT_SECONDS,
        )

        assert result.exit_code == 0, f"[JS-{size_name}] Execution failed: {result.stderr}"

        # Strip trailing newline (runtime may add one)
        stdout = result.stdout.rstrip("\n")
        actual_hash = bytes_hash(stdout.encode("utf-8"))

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
        expected_hash = bytes_hash(expected_binary)

        code = f"""
import base64
import sys

size = {size_bytes}
data = bytes([i % 256 for i in range(size)])
encoded = base64.b64encode(data).decode("ascii")
sys.stdout.write(encoded)
sys.stdout.flush()
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=TIMEOUT_SECONDS,
        )

        assert result.exit_code == 0

        import base64

        actual_binary = base64.b64decode(result.stdout)
        actual_hash = bytes_hash(actual_binary)

        assert len(actual_binary) == size_bytes, f"[Binary-{size_name}] Size: {len(actual_binary)} vs {size_bytes}"

        assert actual_hash == expected_hash, f"[Binary-{size_name}] Binary data corrupted!"


# =============================================================================
# VM-computed hash verification
# =============================================================================


class TestVMHashVerification:
    """Verify VM can compute matching hashes."""

    async def test_vm_hash_matches_host(self, scheduler: Scheduler, small_payload_size: tuple[str, int]) -> None:
        """VM computes same hash as host for same data."""
        size_name, size_bytes = small_payload_size

        expected_data = generate_ascii_pattern(size_bytes)
        expected_hash = bytes_hash(expected_data)

        code = f"""
import hashlib
import sys

chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
size = {size_bytes}
data = "".join(chars[i % len(chars)] for i in range(size))

vm_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
print(f"VMHASH:{{vm_hash}}")
print(f"SIZE:{{len(data)}}")
sys.stdout.write(data)
sys.stdout.flush()
"""

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=TIMEOUT_SECONDS,
        )

        assert result.exit_code == 0

        lines = result.stdout.split("\n", 2)
        assert len(lines) >= 3

        vm_hash = lines[0].split(":")[1]
        # Strip trailing newline from data portion
        received_data = lines[2].rstrip("\n")

        # VM computed same hash as us?
        assert vm_hash == expected_hash, (
            f"[{size_name}] VM hash differs from expected!\nVM: {vm_hash}\nExpected: {expected_hash}"
        )

        # Data we received matches?
        host_hash = bytes_hash(received_data.encode("utf-8"))
        assert host_hash == expected_hash, f"[{size_name}] Received data corrupted in transit!"


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
        expected_hash = bytes_hash(expected_data)

        # Use head -c for exact byte count
        code = f"yes '{repeat_char}' | tr -d '\\n' | head -c {count}"

        result = await scheduler.run(
            code=code,
            language=Language.RAW,
            timeout_seconds=60,
        )

        assert result.exit_code == 0

        # Strip trailing newline (shell may add one)
        stdout = result.stdout.rstrip("\n")
        actual_hash = bytes_hash(stdout.encode("utf-8"))

        assert len(stdout) == count, f"[RAW-{count}] Size: {len(stdout)} vs {count}"

        assert actual_hash == expected_hash, f"[RAW-{count}] Shell output corrupted!"


# =============================================================================
# Large Payload Streaming Tests (1MB - 500MB)
# =============================================================================
# NOTE: Payloads >= 1MB exceed stdout limit, so we use streaming callbacks.


class TestLargePayloadStreaming:
    """Large payload streaming tests (1MB - 500MB).

    All sizes exceed the guest-agent's 1MB stdout limit (MAX_STDOUT_BYTES),
    so these verify that OutputLimitError is raised correctly.
    """

    async def test_streaming_integrity(self, scheduler: Scheduler, streaming_payload_size: tuple[str, int]) -> None:
        """Verify OutputLimitError is raised for payloads exceeding 1MB stdout limit."""
        _size_name, size_bytes = streaming_payload_size

        code = python_code_for_ascii_pattern(size_bytes)

        # Use longer timeout for extra large payloads (25MB+)
        timeout = EXTRA_LARGE_TIMEOUT_SECONDS if size_bytes >= 25_000_000 else TIMEOUT_SECONDS

        with pytest.raises(OutputLimitError):
            await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=timeout,
            )


# =============================================================================
# Throughput Benchmark Tests
# =============================================================================
# Measure raw streaming throughput by minimizing generation overhead


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

    @pytest.mark.parametrize("size_kb", [100, 500, 900])
    async def test_raw_throughput_devzero(self, scheduler: Scheduler, size_kb: int) -> None:
        """Measure throughput using /dev/zero (fastest possible generation)."""
        expected_bytes = size_kb * 1000

        # Use dd from /dev/zero - zero generation overhead
        code = f"dd if=/dev/zero bs=1000 count={size_kb} 2>/dev/null"

        streamed_bytes = 0

        def on_stdout(chunk: str) -> None:
            nonlocal streamed_bytes
            streamed_bytes += len(chunk.encode("utf-8"))

        result = await scheduler.run(
            code=code,
            language=Language.RAW,
            timeout_seconds=120,
            on_stdout=on_stdout,
        )

        # Use execute_ms (host-measured execution phase) — excludes VM
        # preparation (warm pool lookup, L1 restore, cold boot) which is
        # not part of streaming throughput.
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

        result = await scheduler.run(
            code=code,
            language=Language.RAW,
            timeout_seconds=120,
            on_stdout=on_stdout,
        )

        # Use execute_ms (host-measured execution phase) — excludes VM
        # preparation (warm pool lookup, L1 restore, cold boot) which is
        # not part of streaming throughput.
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

        expected_hash = bytes_hash(generate_ascii_pattern(payload_size))
        code = python_code_for_ascii_pattern(payload_size)

        for i in range(num_iterations):
            chunks: list[str] = []

            def make_callback(chunk_list: list[str]) -> Callable[[str], None]:
                def on_stdout(chunk: str) -> None:
                    chunk_list.append(chunk)

                return on_stdout

            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=30,
                on_stdout=make_callback(chunks),
            )

            assert result.exit_code == 0, f"Iteration {i} failed: {result.stderr}"

            streamed_data = "".join(chunks).rstrip("\n").encode("utf-8")
            actual_hash = bytes_hash(streamed_data)

            assert actual_hash == expected_hash, f"Iteration {i} corrupted!"

    async def test_interleaved_stdout_stderr(self, scheduler: Scheduler) -> None:
        """Test rapid interleaving of stdout and stderr streams."""
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

        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            timeout_seconds=60,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )

        assert result.exit_code == 0, f"Interleave test failed: {result.stderr}"

        # Verify all lines present (order may vary due to buffering)
        stdout_full = "".join(stdout_chunks)
        stderr_full = "".join(stderr_chunks)

        stdout_lines = [line for line in stdout_full.strip().split("\n") if line]
        stderr_lines = [line for line in stderr_full.strip().split("\n") if line]

        assert len(stdout_lines) == 1000, f"Missing stdout lines: {len(stdout_lines)}"
        assert len(stderr_lines) == 1000, f"Missing stderr lines: {len(stderr_lines)}"

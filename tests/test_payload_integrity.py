"""Payload integrity tests - verify data authenticity from small to very large.

Tests that data transmitted through the VM execution pipeline is not:
1. Corrupted (bit flips, encoding issues)
2. Truncated (missing data)
3. Modified (extra data, reordering)

Strategy: Generate deterministic output in VM, compare hash with pre-computed expected hash.
Uses dynamic fixtures to test various payload sizes.
"""

import hashlib
from pathlib import Path

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# Images directory - relative to repo root
images_dir = Path(__file__).parent.parent / "images" / "dist"


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
}

# Large sizes for streaming tests only (exceed stdout limit)
STREAMING_ONLY_SIZES = {
    "1mb": 1_000_000,  # 1 MB (at limit)
    "2mb": 2_000_000,  # 2 MB
    "5mb": 5_000_000,  # 5 MB
    "10mb": 10_000_000,  # 10 MB
}


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


def compute_hash(data: bytes) -> str:
    """Compute SHA256 hash."""
    return hashlib.sha256(data).hexdigest()


def generate_ascii_pattern(size: int) -> bytes:
    """Generate deterministic ASCII pattern."""
    chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return bytes([chars[i % len(chars)] for i in range(size)])


def generate_sequential_bytes(size: int) -> bytes:
    """Generate sequential bytes: 0x00, 0x01, ..., 0xFF, 0x00, ..."""
    return bytes([i % 256 for i in range(size)])


def python_code_for_ascii_pattern(size: int) -> str:
    """Generate Python code that outputs ASCII pattern of given size."""
    if size == 1:
        return 'print("A", end="")'

    # For larger sizes, write in chunks to avoid memory issues
    if size > 100 * 1024:
        return f"""
import sys
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
size = {size}
chunk_size = 64 * 1024
for start in range(0, size, chunk_size):
    end = min(start + chunk_size, size)
    chunk = "".join(chars[i % len(chars)] for i in range(start, end))
    sys.stdout.write(chunk)
sys.stdout.flush()
"""
    return f"""
import sys
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
size = {size}
output = "".join(chars[i % len(chars)] for i in range(size))
sys.stdout.write(output)
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

    async def test_payload_integrity(self, payload_size: tuple[str, int]) -> None:
        """Verify payload integrity for given size."""
        size_name, size_bytes = payload_size
        config = SchedulerConfig(images_dir=images_dir)

        # Pre-compute expected
        expected_data = generate_ascii_pattern(size_bytes)
        expected_hash = compute_hash(expected_data)

        # Generate code
        code = python_code_for_ascii_pattern(size_bytes)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=TIMEOUT_SECONDS,
            )

            assert result.exit_code == 0, f"[{size_name}] Execution failed: {result.stderr}"

            # Strip trailing newline (runtime may add one)
            stdout = result.stdout.rstrip("\n")
            actual_data = stdout.encode("utf-8")
            actual_hash = compute_hash(actual_data)

            assert len(actual_data) == size_bytes, f"[{size_name}] Size mismatch: {len(actual_data)} vs {size_bytes}"

            assert actual_hash == expected_hash, (
                f"[{size_name}] HASH MISMATCH - DATA CORRUPTION!\nExpected: {expected_hash}\nActual: {actual_hash}"
            )


class TestPythonStreamingIntegrity:
    """Test streaming callback integrity."""

    async def test_streaming_integrity(self, medium_payload_size: tuple[str, int]) -> None:
        """Verify streaming receives uncorrupted data."""
        size_name, size_bytes = medium_payload_size
        config = SchedulerConfig(images_dir=images_dir)

        expected_hash = compute_hash(generate_ascii_pattern(size_bytes))
        code = python_code_for_ascii_pattern(size_bytes)

        streamed_chunks: list[str] = []

        def on_stdout(chunk: str) -> None:
            streamed_chunks.append(chunk)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=TIMEOUT_SECONDS,
                on_stdout=on_stdout,
            )

            assert result.exit_code == 0

            # Verify streamed data (strip trailing newline)
            streamed_data = "".join(streamed_chunks).rstrip("\n").encode("utf-8")
            streamed_hash = compute_hash(streamed_data)

            assert len(streamed_data) == size_bytes, (
                f"[{size_name}] Streamed size: {len(streamed_data)} vs {size_bytes}"
            )

            assert streamed_hash == expected_hash, (
                f"[{size_name}] Streamed data corrupted!\nExpected: {expected_hash}\nStreamed: {streamed_hash}"
            )

            # Verify final result matches streamed
            final_hash = compute_hash(result.stdout.rstrip("\n").encode("utf-8"))
            assert final_hash == streamed_hash, f"[{size_name}] Final vs streamed mismatch!"


# =============================================================================
# JavaScript payload integrity tests
# =============================================================================


class TestJavaScriptPayloadIntegrity:
    """JavaScript payload integrity tests."""

    async def test_payload_integrity(self, small_payload_size: tuple[str, int]) -> None:
        """Verify JavaScript payload integrity."""
        size_name, size_bytes = small_payload_size
        config = SchedulerConfig(images_dir=images_dir)

        expected_hash = compute_hash(generate_ascii_pattern(size_bytes))
        code = javascript_code_for_ascii_pattern(size_bytes)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.JAVASCRIPT,
                timeout_seconds=TIMEOUT_SECONDS,
            )

            assert result.exit_code == 0, f"[JS-{size_name}] Execution failed: {result.stderr}"

            # Strip trailing newline (runtime may add one)
            stdout = result.stdout.rstrip("\n")
            actual_hash = compute_hash(stdout.encode("utf-8"))

            assert len(stdout) == size_bytes, f"[JS-{size_name}] Size: {len(stdout)} vs {size_bytes}"

            assert actual_hash == expected_hash, f"[JS-{size_name}] Hash mismatch - corruption!"


# =============================================================================
# Binary data integrity (via base64)
# =============================================================================


class TestBinaryPayloadIntegrity:
    """Binary data integrity tests using base64 encoding."""

    async def test_binary_integrity(self, small_payload_size: tuple[str, int]) -> None:
        """Verify binary data integrity via base64."""
        size_name, size_bytes = small_payload_size
        config = SchedulerConfig(images_dir=images_dir)

        expected_binary = generate_sequential_bytes(size_bytes)
        expected_hash = compute_hash(expected_binary)

        code = f"""
import base64
import sys

size = {size_bytes}
data = bytes([i % 256 for i in range(size)])
encoded = base64.b64encode(data).decode("ascii")
sys.stdout.write(encoded)
sys.stdout.flush()
"""

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=TIMEOUT_SECONDS,
            )

            assert result.exit_code == 0

            import base64

            actual_binary = base64.b64decode(result.stdout)
            actual_hash = compute_hash(actual_binary)

            assert len(actual_binary) == size_bytes, f"[Binary-{size_name}] Size: {len(actual_binary)} vs {size_bytes}"

            assert actual_hash == expected_hash, f"[Binary-{size_name}] Binary data corrupted!"


# =============================================================================
# VM-computed hash verification
# =============================================================================


class TestVMHashVerification:
    """Verify VM can compute matching hashes."""

    async def test_vm_hash_matches_host(self, small_payload_size: tuple[str, int]) -> None:
        """VM computes same hash as host for same data."""
        size_name, size_bytes = small_payload_size
        config = SchedulerConfig(images_dir=images_dir)

        expected_data = generate_ascii_pattern(size_bytes)
        expected_hash = compute_hash(expected_data)

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

        async with Scheduler(config) as scheduler:
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
            host_hash = compute_hash(received_data.encode("utf-8"))
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
    async def test_raw_repeated_char(self, repeat_char: str, count: int) -> None:
        """Verify RAW shell can output repeated characters correctly."""
        config = SchedulerConfig(images_dir=images_dir)

        expected_data = (repeat_char * count).encode("utf-8")
        expected_hash = compute_hash(expected_data)

        # Use head -c for exact byte count
        code = f"yes '{repeat_char}' | tr -d '\\n' | head -c {count}"

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.RAW,
                timeout_seconds=60,
            )

            assert result.exit_code == 0

            # Strip trailing newline (shell may add one)
            stdout = result.stdout.rstrip("\n")
            actual_hash = compute_hash(stdout.encode("utf-8"))

            assert len(stdout) == count, f"[RAW-{count}] Size: {len(stdout)} vs {count}"

            assert actual_hash == expected_hash, f"[RAW-{count}] Shell output corrupted!"


# =============================================================================
# Large Payload Streaming Tests (1MB - 10MB)
# =============================================================================
# NOTE: Payloads >= 1MB exceed stdout limit, so we use streaming callbacks.


class TestLargePayloadStreaming:
    """Large payload streaming tests (1MB - 10MB).

    These tests use streaming callbacks because stdout is limited to 1MB.
    """

    async def test_streaming_integrity(self, streaming_payload_size: tuple[str, int]) -> None:
        """Verify streaming receives all data for large payloads."""
        size_name, size_bytes = streaming_payload_size
        config = SchedulerConfig(images_dir=images_dir)

        expected_hash = compute_hash(generate_ascii_pattern(size_bytes))
        code = python_code_for_ascii_pattern(size_bytes)

        streamed_chunks: list[str] = []

        def on_stdout(chunk: str) -> None:
            streamed_chunks.append(chunk)

        async with Scheduler(config) as scheduler:
            result = await scheduler.run(
                code=code,
                language=Language.PYTHON,
                timeout_seconds=TIMEOUT_SECONDS,
                on_stdout=on_stdout,
            )

            assert result.exit_code == 0, f"[{size_name}] Execution failed: {result.stderr}"

            # Strip trailing newline (runtime may add one)
            streamed_data = "".join(streamed_chunks).rstrip("\n").encode("utf-8")
            streamed_hash = compute_hash(streamed_data)

            assert len(streamed_data) == size_bytes, (
                f"[{size_name}] Streamed size mismatch!\n"
                f"Expected: {size_bytes:,} bytes\n"
                f"Streamed: {len(streamed_data):,} bytes"
            )

            assert streamed_hash == expected_hash, (
                f"[{size_name}] Streamed data corrupted!\nExpected: {expected_hash}\nActual: {streamed_hash}"
            )

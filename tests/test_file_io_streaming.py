"""Integration tests for streaming file transfers under memory pressure and concurrency.

Validates that the zstd-compressed chunked file transfer protocol:
1. Handles large files (20-150 MB) in VMs without OOM
2. Supports concurrent parallel file transfers via asyncio.gather
3. Keeps guest memory usage low during transfers (streaming, not buffering)
4. Handles interleaved success/failure correctly (dispatcher multiplexing)
5. Measures upload/download throughput (write_file/read_file MiB/s)

Note: /home/user is the VM's home disk (ext4, ~973 MB usable by default), so large
files consume host disk, not guest RAM: the memory tests use the smallest VM (128 MB).

All tests require real VMs via the scheduler fixture.
"""

import asyncio
import hashlib
import os
import textwrap
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from exec_sandbox.exceptions import ErrorCode, InvalidInputError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.session import Session
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import skip_unless_hwaccel

# --- Guest memory threshold (percentage of MemTotal that must remain free) ---
# Catches O(file_size) buffering while it tolerates OS page cache variance and emulation
# overhead. In a 128 MB VM, a transfer that holds the whole 40 MB file in guest memory
# leaves 4-13% free (HVF); streaming leaves 38-48% on arm64 HVF and 37-57% on arm64 TCG
# (lowest sample of the read and the write tests, measured), and passes on x86_64 KVM.
_GUEST_MEM_AVAILABLE_MIN_PCT = 15

_SMALL_VM_MB = 128
"""The smallest VM (constants.MIN_MEMORY_MB): a transfer that buffers the file shows in its memory."""

_MEMORY_MONITOR = textwrap.dedent("""\
    import os, time
    pid = os.fork()
    if pid == 0:
        with open('/home/user/memlog.txt', 'w') as log:
            while True:
                try:
                    with open('/proc/meminfo') as f:
                        for line in f:
                            if line.startswith('MemAvailable:'):
                                log.write(line.split()[1] + '\\n')
                                log.flush()
                                break
                except Exception:
                    break
                time.sleep(0.05)
        os._exit(0)
    print(f'pid={pid}')
""")
"""Guest code: a child of the REPL writes MemAvailable (KB) to memlog.txt every 50 ms."""


async def _min_available_pct(session: Session, transfer: Callable[[], Awaitable[None]]) -> tuple[float, int]:
    """Run `transfer` while the guest samples MemAvailable; return (lowest sample as % of
    MemTotal, number of samples). The samples go on until the transfer has ended."""
    start = await session.exec(_MEMORY_MONITOR)
    assert start.exit_code == 0, f"Monitor start failed: {start.stderr}"
    monitor_pid = int(start.stdout.strip().split("=")[1])

    await transfer()

    stop = await session.exec(
        f"import os, signal\nos.kill({monitor_pid}, signal.SIGTERM)\nos.waitpid({monitor_pid}, 0)\n"
        "print(open('/home/user/memlog.txt').read())\n"
        "print('total', next(line.split()[1] for line in open('/proc/meminfo') if line.startswith('MemTotal:')))"
    )
    assert stop.exit_code == 0, f"Monitor stop failed: {stop.stderr}"
    *log, total = stop.stdout.strip().split("\n")
    samples = [int(s) for s in log if s.strip().isdigit()]
    assert len(samples) >= 2, f"Too few memory samples: {len(samples)}"
    return min(samples) / int(total.split()[1]) * 100, len(samples)


# =============================================================================
# TestStreamingLargeFiles - Large file transfers in low-memory VMs
# =============================================================================


@pytest.mark.slow
class TestStreamingLargeFiles:
    """A file larger than the guest's memory goes in and out of the VM whole.

    Slow under TCG: data integrity (SHA256 round-trip) correctness tests —
    work under TCG but large file transfers are too slow for the default suite.
    """

    async def test_write_read_file_larger_than_vm_memory(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write a 150 MB file to a 128 MB VM and read it back with SHA256 verification.

        The home disk holds it: the guest keeps at most 16 chunks of a write in memory.
        """
        size = 150 * 1024 * 1024
        content = os.urandom(size)
        expected_hash = hashlib.sha256(content).hexdigest()
        dest = tmp_path / "large_150mb.bin"

        async with await scheduler.new(VmSpec(language=Language.PYTHON, memory_mb=_SMALL_VM_MB)) as session:
            await session.write_file("large_150mb.bin", content)
            await session.read_file("large_150mb.bin", destination=dest)

        result = dest.read_bytes()
        assert len(result) == size, f"Size mismatch: got {len(result)}, expected {size}"
        actual_hash = hashlib.sha256(result).hexdigest()
        assert actual_hash == expected_hash, (
            f"SHA256 mismatch - data corruption!\nExpected: {expected_hash}\nActual:   {actual_hash}"
        )


# =============================================================================
# TestConcurrentFileTransfers - Parallel file I/O via asyncio.gather
# =============================================================================


@pytest.mark.slow
class TestConcurrentFileTransfers:
    """Prove multiple concurrent file transfers work correctly.

    Slow under TCG: concurrent SHA256 correctness tests — work under TCG
    but concurrent file I/O overhead makes them too slow for the default suite.
    """

    async def test_concurrent_write_5_files(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write 5 different 5 MB files concurrently, verify all round-trip."""
        size = 5 * 1024 * 1024
        files = {f"concurrent_{i}.bin": os.urandom(size) for i in range(5)}
        expected_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Write all 5 concurrently
            await asyncio.gather(*(session.write_file(name, data) for name, data in files.items()))

            # Read all back and verify
            for name, expected_hash in expected_hashes.items():
                dest = tmp_path / name
                await session.read_file(name, destination=dest)
                result = dest.read_bytes()
                assert len(result) == size, f"[{name}] Size mismatch: got {len(result)}, expected {size}"
                actual_hash = hashlib.sha256(result).hexdigest()
                assert actual_hash == expected_hash, f"[{name}] SHA256 mismatch"

    async def test_concurrent_read_5_files(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write 5 files sequentially, then read all 5 concurrently."""
        size = 5 * 1024 * 1024
        files = {f"read_concurrent_{i}.bin": os.urandom(size) for i in range(5)}
        expected_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Write sequentially
            for name, data in files.items():
                await session.write_file(name, data)

            # Read all 5 concurrently
            dests = {name: tmp_path / name for name in files}
            await asyncio.gather(*(session.read_file(name, destination=dest) for name, dest in dests.items()))

            # Verify all match
            for name, expected_hash in expected_hashes.items():
                result = dests[name].read_bytes()
                assert len(result) == size, f"[{name}] Size mismatch: got {len(result)}, expected {size}"
                actual_hash = hashlib.sha256(result).hexdigest()
                assert actual_hash == expected_hash, f"[{name}] SHA256 mismatch"

    async def test_concurrent_mixed_read_write(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Mix of concurrent reads and writes (3 writes + 2 reads) via asyncio.gather."""
        size = 5 * 1024 * 1024

        # Pre-write 2 files for reading
        read_files = {f"pre_written_{i}.bin": os.urandom(size) for i in range(2)}
        read_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in read_files.items()}

        # Prepare 3 files for writing
        write_files = {f"new_write_{i}.bin": os.urandom(size) for i in range(3)}
        write_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in write_files.items()}

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Pre-write files for the read phase
            for name, data in read_files.items():
                await session.write_file(name, data)

            # Mix: 3 writes + 2 reads concurrently
            read_dests = {name: tmp_path / name for name in read_files}
            await asyncio.gather(
                asyncio.gather(*(session.write_file(name, data) for name, data in write_files.items())),
                asyncio.gather(*(session.read_file(name, destination=dest) for name, dest in read_dests.items())),
            )

            # Verify reads match
            for name, expected_hash in read_hashes.items():
                result = read_dests[name].read_bytes()
                assert len(result) == size, f"[{name}] Size mismatch"
                assert hashlib.sha256(result).hexdigest() == expected_hash, f"[{name}] SHA256 mismatch"

            # Verify written files
            for name, expected_hash in write_hashes.items():
                dest = tmp_path / f"verify_{name}"
                await session.read_file(name, destination=dest)
                result = dest.read_bytes()
                assert len(result) == size, f"[{name}] Size mismatch"
                assert hashlib.sha256(result).hexdigest() == expected_hash, f"[{name}] SHA256 mismatch"


# =============================================================================
# TestStreamingMemoryEfficiency - Guest memory stays low during transfers
# =============================================================================


@pytest.mark.slow
class TestStreamingMemoryEfficiency:
    """A transfer does not take guest memory in proportion to the file size.

    Slow under TCG: the threshold holds there (_GUEST_MEM_AVAILABLE_MIN_PCT), but the
    transfers are too slow for the default suite.
    """

    async def test_multiple_large_files_dont_oom_small_vm(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write 3 x 20 MB files one after the other in a 128 MB VM without OOM.

        After each write, the guest still runs code and the file reads back whole.
        """
        size = 20 * 1024 * 1024

        async with await scheduler.new(VmSpec(language=Language.PYTHON, memory_mb=_SMALL_VM_MB)) as session:
            for i in range(3):
                content = os.urandom(size)
                expected_hash = hashlib.sha256(content).hexdigest()

                await session.write_file(f"multi_{i}.bin", content)

                # Verify guest is still responsive after each write
                result = await session.exec("print('alive')")
                assert result.exit_code == 0, f"Guest unresponsive after writing file {i}: {result.stderr}"
                assert result.stdout == "alive\n"

                # Verify file integrity
                dest = tmp_path / f"multi_{i}.bin"
                await session.read_file(f"multi_{i}.bin", destination=dest)
                assert hashlib.sha256(dest.read_bytes()).hexdigest() == expected_hash, (
                    f"File {i} SHA256 mismatch after write"
                )

    async def test_guest_peak_memory_during_write_transfer(self, scheduler: Scheduler) -> None:
        """Sample guest MemAvailable during a 40 MB write in a 128 MB VM, and after it.

        A child of the REPL samples /proc/meminfo every 50 ms. The lowest sample must stay
        at or above _GUEST_MEM_AVAILABLE_MIN_PCT of MemTotal: a write that holds the whole
        file in guest memory leaves 4-13%.
        """
        content = os.urandom(40 * 1024 * 1024)

        async with await scheduler.new(VmSpec(language=Language.PYTHON, memory_mb=_SMALL_VM_MB)) as session:
            min_pct, count = await _min_available_pct(
                session, lambda: session.write_file("peak_test_40mb.bin", content)
            )

        assert min_pct >= _GUEST_MEM_AVAILABLE_MIN_PCT, (
            f"Guest memory trough too low during 40 MB write: {min_pct:.1f}% (from {count} samples)"
        )

    async def test_guest_peak_memory_during_read_transfer(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Sample guest MemAvailable during a 40 MB read in a 128 MB VM, and after it; the
        file reads back whole. Same monitor and threshold as the write test."""
        content = os.urandom(40 * 1024 * 1024)
        dest = tmp_path / "peak_read_40mb.bin"

        async with await scheduler.new(VmSpec(language=Language.PYTHON, memory_mb=_SMALL_VM_MB)) as session:
            await session.write_file("peak_read_40mb.bin", content)
            min_pct, count = await _min_available_pct(
                session, lambda: session.read_file("peak_read_40mb.bin", destination=dest)
            )

        assert min_pct >= _GUEST_MEM_AVAILABLE_MIN_PCT, (
            f"Guest memory trough too low during 40 MB read: {min_pct:.1f}% (from {count} samples)"
        )
        assert dest.read_bytes() == content


# =============================================================================
# TestInterleavedSuccessFailure - Mixed outcomes in concurrent operations
# =============================================================================


@pytest.mark.slow
class TestInterleavedSuccessFailure:
    """Concurrent file ops where some succeed and some fail.

    Slow under TCG: FileOpDispatcher multiplexing correctness tests with
    no timing assertions — work under TCG but too slow for the default suite.

    Stress-tests the FileOpDispatcher multiplexing: each op gets its own
    op_id queue, and errors for one op must not corrupt other ops.
    """

    async def test_concurrent_writes_valid_and_invalid_paths(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Gather 3 valid writes + 2 invalid writes (path traversal).

        Valid writes must succeed with correct data. Invalid writes must raise
        InvalidInputError PATH_INVALID. No cross-contamination between ops.
        """
        size = 1 * 1024 * 1024  # 1 MB each
        valid_files = {f"valid_{i}.bin": os.urandom(size) for i in range(3)}
        valid_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in valid_files.items()}

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Fire all 5 concurrently: 3 valid + 2 invalid
            results = await asyncio.gather(
                *(session.write_file(name, data) for name, data in valid_files.items()),
                session.write_file("../escape1.bin", os.urandom(64)),
                session.write_file("../escape2.bin", os.urandom(64)),
                return_exceptions=True,
            )

            # First 3 should succeed (None return)
            for i, name in enumerate(valid_files):
                assert results[i] is None, f"[{name}] Expected success, got {results[i]}"

            # Last 2 should be InvalidInputError PATH_INVALID
            for i in range(3, 5):
                error = results[i]
                assert isinstance(error, InvalidInputError), (
                    f"Expected InvalidInputError for invalid path, got {error!r}"
                )
                assert error.code is ErrorCode.PATH_INVALID

            # Verify valid files are intact (read back + SHA256)
            for name, expected_hash in valid_hashes.items():
                dest = tmp_path / name
                await session.read_file(name, destination=dest)
                actual_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
                assert actual_hash == expected_hash, f"[{name}] SHA256 mismatch after mixed gather"

    async def test_concurrent_reads_existent_and_nonexistent(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Gather 2 valid reads + 2 reads of nonexistent files.

        Valid reads return correct data. Nonexistent reads raise InvalidInputError GUEST_IO.
        """
        size = 1 * 1024 * 1024
        files = {f"exists_{i}.bin": os.urandom(size) for i in range(2)}
        expected_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Pre-write the valid files
            for name, data in files.items():
                await session.write_file(name, data)

            # Fire 4 concurrent reads: 2 valid + 2 nonexistent
            valid_dests = {name: tmp_path / name for name in files}
            results = await asyncio.gather(
                *(session.read_file(name, destination=dest) for name, dest in valid_dests.items()),
                session.read_file("ghost_a.bin", destination=tmp_path / "ghost_a.bin"),
                session.read_file("ghost_b.bin", destination=tmp_path / "ghost_b.bin"),
                return_exceptions=True,
            )

            # First 2 should succeed
            for i, name in enumerate(files):
                assert results[i] is None, f"[{name}] Expected success, got {results[i]}"

            # Last 2 should be InvalidInputError GUEST_IO
            for i in range(2, 4):
                error = results[i]
                assert isinstance(error, InvalidInputError), (
                    f"Expected InvalidInputError for nonexistent file, got {error!r}"
                )
                assert error.code is ErrorCode.GUEST_IO

            # Verify valid files are intact
            for name, expected_hash in expected_hashes.items():
                actual_hash = hashlib.sha256(valid_dests[name].read_bytes()).hexdigest()
                assert actual_hash == expected_hash, f"[{name}] SHA256 mismatch"

    async def test_mixed_concurrent_write_read_with_failures(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Gather: valid write + invalid write + valid read + invalid read.

        Each op gets correct outcome. Dispatcher routes errors to the right op_id.
        """
        size = 1 * 1024 * 1024
        write_data = os.urandom(size)
        write_hash = hashlib.sha256(write_data).hexdigest()
        read_data = os.urandom(size)
        read_hash = hashlib.sha256(read_data).hexdigest()

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Pre-write a file for reading
            await session.write_file("for_reading.bin", read_data)

            read_dest = tmp_path / "for_reading.bin"
            results = await asyncio.gather(
                session.write_file("good_write.bin", write_data),  # success
                session.write_file("../bad_write.bin", b"x"),  # fail: traversal
                session.read_file("for_reading.bin", destination=read_dest),  # success
                session.read_file("nonexistent.bin", destination=tmp_path / "nope.bin"),  # fail: not found
                return_exceptions=True,
            )

            assert results[0] is None, f"Valid write failed: {results[0]}"
            assert isinstance(results[1], InvalidInputError), f"Invalid write should fail: {results[1]!r}"
            assert results[1].code is ErrorCode.PATH_INVALID
            assert results[2] is None, f"Valid read failed: {results[2]}"
            assert isinstance(results[3], InvalidInputError), f"Invalid read should fail: {results[3]!r}"
            assert results[3].code is ErrorCode.GUEST_IO

            # Verify successful write is intact
            verify_dest = tmp_path / "good_write.bin"
            await session.read_file("good_write.bin", destination=verify_dest)
            assert hashlib.sha256(verify_dest.read_bytes()).hexdigest() == write_hash

            # Verify successful read is intact
            assert hashlib.sha256(read_dest.read_bytes()).hexdigest() == read_hash

    async def test_session_usable_after_failed_operation(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """After a failed file op, the session remains usable for subsequent ops.

        Verifies no leaked state from the failed operation corrupts the channel.
        """
        size = 512 * 1024
        content = os.urandom(size)
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Fail: read nonexistent file
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
                await session.read_file("ghost.bin", destination=tmp_path / "ghost.bin")

            # Fail: write to traversal path
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                await session.write_file("../escape.bin", b"x")

            # Success: write + read should still work
            await session.write_file("after_failure.bin", content)
            dest = tmp_path / "after_failure.bin"
            await session.read_file("after_failure.bin", destination=dest)
            assert hashlib.sha256(dest.read_bytes()).hexdigest() == expected_hash

            # Success: exec should still work
            result = await session.exec("print('still alive')")
            assert (result.exit_code, result.stdout) == (0, "still alive\n"), result.stderr

    async def test_concurrent_list_with_writes(self, scheduler: Scheduler) -> None:
        """list_files during 2 writes: it sees the files written before, and only files that
        exist; a list after the writes sees all of them."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Write some initial files
            for i in range(3):
                await session.write_file(f"file_{i}.txt", f"content_{i}".encode())

            # Concurrent: 2 more writes + list
            results = await asyncio.gather(
                session.write_file("file_3.txt", b"content_3"),
                session.write_file("file_4.txt", b"content_4"),
                session.list_files(),
                return_exceptions=True,
            )

            # Writes should succeed
            assert results[0] is None, f"Write 3 failed: {results[0]}"
            assert results[1] is None, f"Write 4 failed: {results[1]}"

            # The list runs between the writes: it has the 3 files written before, maybe the new ones
            file_list = results[2]
            assert isinstance(file_list, list), f"list_files should return list, got {type(file_list)}"
            names = {f.name for f in file_list if f.name.startswith("file_")}
            assert {f"file_{i}.txt" for i in range(3)} <= names <= {f"file_{i}.txt" for i in range(5)}

            # After the writes, a list has all 5
            after = {f.name for f in await session.list_files() if f.name.startswith("file_")}
            assert after == {f"file_{i}.txt" for i in range(5)}


# =============================================================================
# TestFileTransferThroughput - Upload/download MiB/s benchmarks
# =============================================================================


@skip_unless_hwaccel
class TestFileTransferThroughput:
    """Measure upload (write_file) and download (read_file) throughput.

    Prints the MiB/s (no threshold) and checks the data. Requires hwaccel: under TCG
    (~5-8x slower) the large transfers take too long.

    Run with -s to see throughput output:
        uv run pytest tests/test_file_io_streaming.py::TestFileTransferThroughput -v -s -n 0
    """

    @pytest.mark.parametrize("size_mib", [10, 50])
    async def test_upload_throughput(self, scheduler: Scheduler, size_mib: int) -> None:
        """Measure host->guest write_file throughput in MiB/s."""
        size = size_mib * 1024 * 1024
        content = os.urandom(size)
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.new(VmSpec(language=Language.PYTHON, memory_mb=256)) as session:
            start = time.perf_counter()
            await session.write_file("bench.bin", content)
            elapsed = time.perf_counter() - start

            throughput = size_mib / elapsed
            # Verify integrity
            result = await session.exec(
                "import hashlib; print(hashlib.sha256(open('bench.bin','rb').read()).hexdigest())"
            )
            assert result.exit_code == 0, f"Hash check failed: {result.stderr}"
            assert result.stdout == expected_hash + "\n", "Upload data corrupted"

        print(f"\n  UPLOAD {size_mib} MiB: {elapsed:.1f}s, {throughput:.1f} MiB/s")  # noqa: T201

    @pytest.mark.parametrize("size_mib", [10, 50])
    async def test_download_throughput(self, scheduler: Scheduler, tmp_path: Path, size_mib: int) -> None:
        """Measure guest->host read_file throughput in MiB/s."""
        size = size_mib * 1024 * 1024
        expected_content = bytes(range(256)) * (size // 256)
        expected_hash = hashlib.sha256(expected_content).hexdigest()
        dest = tmp_path / "bench_dl.bin"

        async with await scheduler.new(VmSpec(language=Language.PYTHON, memory_mb=256)) as session:
            # Generate deterministic file in guest
            await session.exec(f"open('bench.bin','wb').write(bytes(range(256)) * {size // 256})")
            start = time.perf_counter()
            await session.read_file("bench.bin", destination=dest)
            elapsed = time.perf_counter() - start

        throughput = size_mib / elapsed
        assert dest.stat().st_size == size, "Download size mismatch"
        actual_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, "Download data corrupted"
        print(f"\n  DOWNLOAD {size_mib} MiB: {elapsed:.1f}s, {throughput:.1f} MiB/s")  # noqa: T201

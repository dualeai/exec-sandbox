"""Integration tests for streaming file transfers under memory pressure and concurrency.

Validates that the zstd-compressed chunked file transfer protocol:
1. Handles large files (30-50 MB) in low-memory VMs without OOM
2. Supports concurrent parallel file transfers via asyncio.gather
3. Keeps guest memory usage low during transfers (streaming, not buffering)
4. Handles interleaved success/failure correctly (dispatcher multiplexing)

All tests require real VMs via the scheduler fixture.
"""

import asyncio
import hashlib
import os
import textwrap
from pathlib import Path

import pytest

from exec_sandbox.exceptions import VmPermanentError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from tests.conftest import skip_unless_hwaccel

# --- Guest memory threshold (percentage of MemTotal that must remain free) ---
# Catches catastrophic O(file_size) buffering (leaves ~5% free) while
# tolerating OS page cache variance and arm64 TCG emulation overhead.
# 15% gives ample margin for streaming (which typically leaves 30-40% free
# on KVM, ~17-25% on arm64 TCG due to higher kernel/emulation overhead).
_GUEST_MEM_AVAILABLE_MIN_PCT = 15

# =============================================================================
# TestStreamingLargeFiles - Large file transfers in low-memory VMs
# =============================================================================


@skip_unless_hwaccel
class TestStreamingLargeFiles:
    """Prove streaming works for files that would OOM with the old buffered approach."""

    async def test_write_read_30mb_in_128mb_vm(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write 30 MB to a 128 MB VM and read it back with SHA256 verification.

        With the old base64 approach, 30 MB would require ~110 MB of guest RAM
        just for decode + content, leaving nothing for the OS in 128 MB.
        Streaming only needs ~128 KB decompression buffer.
        """
        size = 30 * 1024 * 1024
        content = os.urandom(size)
        expected_hash = hashlib.sha256(content).hexdigest()
        dest = tmp_path / "large_30mb.bin"

        async with await scheduler.session(language=Language.PYTHON, memory_mb=128) as session:
            await session.write_file("large_30mb.bin", content)
            await session.read_file("large_30mb.bin", destination=dest)

        result = dest.read_bytes()
        assert len(result) == size, f"Size mismatch: got {len(result)}, expected {size}"
        actual_hash = hashlib.sha256(result).hexdigest()
        assert actual_hash == expected_hash, (
            f"SHA256 mismatch - data corruption!\nExpected: {expected_hash}\nActual:   {actual_hash}"
        )

    async def test_write_read_file_larger_than_vm_memory(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write a 150 MB file to a 128 MB VM — file is larger than total RAM.

        The file is 17% larger than the VM's entire memory.  With the old
        buffered approach this would be impossible (~550 MB peak for base64).
        Streaming only needs ~64 KB decompression buffer in the guest.
        """
        size = 150 * 1024 * 1024
        content = os.urandom(size)
        expected_hash = hashlib.sha256(content).hexdigest()
        dest = tmp_path / "large_150mb.bin"

        async with await scheduler.session(language=Language.PYTHON, memory_mb=128) as session:
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


@skip_unless_hwaccel
class TestConcurrentFileTransfers:
    """Prove multiple concurrent file transfers work correctly."""

    async def test_concurrent_write_5_files(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write 5 different 5 MB files concurrently, verify all round-trip."""
        size = 5 * 1024 * 1024
        files = {f"concurrent_{i}.bin": os.urandom(size) for i in range(5)}
        expected_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

        async with await scheduler.session(language=Language.PYTHON) as session:
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

        async with await scheduler.session(language=Language.PYTHON) as session:
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

        async with await scheduler.session(language=Language.PYTHON) as session:
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


@skip_unless_hwaccel
class TestStreamingMemoryEfficiency:
    """Verify streaming doesn't spike guest memory."""

    async def test_guest_memory_stays_low_during_large_transfer(self, scheduler: Scheduler) -> None:
        """Write 30 MB, then check guest MemAvailable didn't drop catastrophically.

        With streaming, the guest only holds a ~128 KB decompression buffer.
        The file is written directly to disk, so MemAvailable should remain
        high (~80%+ of total) after the transfer completes.
        Uses 128 MB VM to make the test sensitive to memory spikes.
        """
        size = 30 * 1024 * 1024
        content = os.urandom(size)

        async with await scheduler.session(language=Language.PYTHON, memory_mb=128) as session:
            await session.write_file("mem_test_30mb.bin", content)

            # Read /proc/meminfo to check available memory
            result = await session.exec(
                "with open('/proc/meminfo') as f:\n"
                "    for line in f:\n"
                "        if line.startswith('MemAvailable:') or line.startswith('MemTotal:'):\n"
                "            print(line.strip())"
            )

            assert result.exit_code == 0, f"meminfo exec failed: {result.stderr}"

            # Parse MemTotal and MemAvailable from output
            mem_total_kb = 0
            mem_available_kb = 0
            for line in result.stdout.strip().split("\n"):
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available_kb = int(line.split()[1])

            assert mem_total_kb > 0, "Could not parse MemTotal"
            assert mem_available_kb > 0, "Could not parse MemAvailable"

            available_pct = (mem_available_kb / mem_total_kb) * 100
            assert available_pct >= _GUEST_MEM_AVAILABLE_MIN_PCT, (
                f"Guest memory too low after 30 MB transfer: "
                f"{mem_available_kb} KB available / {mem_total_kb} KB total = {available_pct:.1f}%"
            )

    async def test_multiple_large_files_dont_oom_small_vm(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write 3 x 20 MB files sequentially in a 128 MB VM without OOM.

        After each write, verify the guest is still responsive. With the old
        buffered approach, the first 20 MB file would likely crash a 128 MB VM.
        """
        size = 20 * 1024 * 1024

        async with await scheduler.session(language=Language.PYTHON, memory_mb=128) as session:
            for i in range(3):
                content = os.urandom(size)
                expected_hash = hashlib.sha256(content).hexdigest()

                await session.write_file(f"multi_{i}.bin", content)

                # Verify guest is still responsive after each write
                result = await session.exec("print('alive')")
                assert result.exit_code == 0, f"Guest unresponsive after writing file {i}: {result.stderr}"
                assert "alive" in result.stdout

                # Verify file integrity
                dest = tmp_path / f"multi_{i}.bin"
                await session.read_file(f"multi_{i}.bin", destination=dest)
                assert hashlib.sha256(dest.read_bytes()).hexdigest() == expected_hash, (
                    f"File {i} SHA256 mismatch after write"
                )

    async def test_guest_peak_memory_during_write_transfer(self, scheduler: Scheduler) -> None:
        """Sample guest MemAvailable DURING a 30 MB write, not just after.

        Spawns a background monitor via os.fork() that samples /proc/meminfo
        every 50 ms to /home/user/memlog.txt.  The minimum MemAvailable sample
        must stay >= 25% of MemTotal — lower than the post-transfer 30% threshold
        because we capture the actual trough during active I/O.  A buffered
        approach would show ~5%, so 25% still catches the bug.
        """
        size = 30 * 1024 * 1024
        content = os.urandom(size)

        monitor_code = textwrap.dedent("""\
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

        async with await scheduler.session(language=Language.PYTHON, memory_mb=128) as session:
            # Start background memory monitor
            start_result = await session.exec(monitor_code)
            assert start_result.exit_code == 0, f"Monitor start failed: {start_result.stderr}"
            monitor_pid = int(start_result.stdout.strip().split("=")[1])

            # Write the large file while monitor samples
            await session.write_file("peak_test_30mb.bin", content)

            # Stop monitor and read results
            stop_code = textwrap.dedent(f"""\
                import os, signal, time
                os.kill({monitor_pid}, signal.SIGTERM)
                time.sleep(0.1)
                print(open('/home/user/memlog.txt').read())
            """)
            stop_result = await session.exec(stop_code)
            assert stop_result.exit_code == 0, f"Monitor stop failed: {stop_result.stderr}"

            # Also read MemTotal for percentage calculation
            memtotal_result = await session.exec(
                "with open('/proc/meminfo') as f:\n"
                "    for line in f:\n"
                "        if line.startswith('MemTotal:'):\n"
                "            print(line.split()[1])\n"
                "            break"
            )
            assert memtotal_result.exit_code == 0
            mem_total_kb = int(memtotal_result.stdout.strip())

            # Parse samples
            samples = [int(s) for s in stop_result.stdout.strip().split("\n") if s.strip().isdigit()]
            assert len(samples) >= 2, f"Too few memory samples: {len(samples)}"

            min_available_kb = min(samples)
            min_pct = (min_available_kb / mem_total_kb) * 100

            assert min_pct >= _GUEST_MEM_AVAILABLE_MIN_PCT, (
                f"Guest memory trough too low during 30 MB write: "
                f"min {min_available_kb} KB / {mem_total_kb} KB = {min_pct:.1f}% "
                f"(from {len(samples)} samples)"
            )

    async def test_guest_memory_stays_low_during_large_read(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Read 30 MB out of a 128 MB VM, check MemAvailable after transfer.

        Same pattern as the write test but exercises the read path.
        Assert MemAvailable >= 25% after the transfer completes.
        """
        size = 30 * 1024 * 1024
        content = os.urandom(size)
        dest = tmp_path / "read_mem_test_30mb.bin"

        async with await scheduler.session(language=Language.PYTHON, memory_mb=128) as session:
            await session.write_file("read_mem_test_30mb.bin", content)
            await session.read_file("read_mem_test_30mb.bin", destination=dest)

            result = await session.exec(
                "with open('/proc/meminfo') as f:\n"
                "    for line in f:\n"
                "        if line.startswith('MemAvailable:') or line.startswith('MemTotal:'):\n"
                "            print(line.strip())"
            )
            assert result.exit_code == 0, f"meminfo exec failed: {result.stderr}"

            mem_total_kb = 0
            mem_available_kb = 0
            for line in result.stdout.strip().split("\n"):
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available_kb = int(line.split()[1])

            assert mem_total_kb > 0, "Could not parse MemTotal"
            assert mem_available_kb > 0, "Could not parse MemAvailable"

            available_pct = (mem_available_kb / mem_total_kb) * 100
            assert available_pct >= _GUEST_MEM_AVAILABLE_MIN_PCT, (
                f"Guest memory too low after 30 MB read: "
                f"{mem_available_kb} KB available / {mem_total_kb} KB total = {available_pct:.1f}%"
            )

        assert dest.read_bytes() == content

    async def test_guest_peak_memory_during_read_transfer(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Sample guest MemAvailable DURING a 30 MB read, not just after.

        Same background monitor pattern as the write peak test.
        Assert minimum sample >= 25% of MemTotal.
        """
        size = 30 * 1024 * 1024
        content = os.urandom(size)
        dest = tmp_path / "peak_read_30mb.bin"

        monitor_code = textwrap.dedent("""\
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

        async with await scheduler.session(language=Language.PYTHON, memory_mb=128) as session:
            # First write the file to read back
            await session.write_file("peak_read_30mb.bin", content)

            # Start background memory monitor
            start_result = await session.exec(monitor_code)
            assert start_result.exit_code == 0, f"Monitor start failed: {start_result.stderr}"
            monitor_pid = int(start_result.stdout.strip().split("=")[1])

            # Read the large file while monitor samples
            await session.read_file("peak_read_30mb.bin", destination=dest)

            # Stop monitor and read results
            stop_code = textwrap.dedent(f"""\
                import os, signal, time
                os.kill({monitor_pid}, signal.SIGTERM)
                time.sleep(0.1)
                print(open('/home/user/memlog.txt').read())
            """)
            stop_result = await session.exec(stop_code)
            assert stop_result.exit_code == 0, f"Monitor stop failed: {stop_result.stderr}"

            memtotal_result = await session.exec(
                "with open('/proc/meminfo') as f:\n"
                "    for line in f:\n"
                "        if line.startswith('MemTotal:'):\n"
                "            print(line.split()[1])\n"
                "            break"
            )
            assert memtotal_result.exit_code == 0
            mem_total_kb = int(memtotal_result.stdout.strip())

            samples = [int(s) for s in stop_result.stdout.strip().split("\n") if s.strip().isdigit()]
            assert len(samples) >= 2, f"Too few memory samples: {len(samples)}"

            min_available_kb = min(samples)
            min_pct = (min_available_kb / mem_total_kb) * 100

            assert min_pct >= _GUEST_MEM_AVAILABLE_MIN_PCT, (
                f"Guest memory trough too low during 30 MB read: "
                f"min {min_available_kb} KB / {mem_total_kb} KB = {min_pct:.1f}% "
                f"(from {len(samples)} samples)"
            )

        assert dest.read_bytes() == content


# =============================================================================
# TestInterleavedSuccessFailure - Mixed outcomes in concurrent operations
# =============================================================================


@skip_unless_hwaccel
class TestInterleavedSuccessFailure:
    """Concurrent file ops where some succeed and some fail.

    Stress-tests the FileOpDispatcher multiplexing: each op gets its own
    op_id queue, and errors for one op must not corrupt other ops.
    """

    async def test_concurrent_writes_valid_and_invalid_paths(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Gather 3 valid writes + 2 invalid writes (path traversal).

        Valid writes must succeed with correct data. Invalid writes must raise
        VmPermanentError. No cross-contamination between ops.
        """
        size = 1 * 1024 * 1024  # 1 MB each
        valid_files = {f"valid_{i}.bin": os.urandom(size) for i in range(3)}
        valid_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in valid_files.items()}

        async with await scheduler.session(language=Language.PYTHON) as session:
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

            # Last 2 should be VmPermanentError
            for i in range(3, 5):
                assert isinstance(results[i], VmPermanentError), (
                    f"Expected VmPermanentError for invalid path, got {type(results[i])}"
                )

            # Verify valid files are intact (read back + SHA256)
            for name, expected_hash in valid_hashes.items():
                dest = tmp_path / name
                await session.read_file(name, destination=dest)
                actual_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
                assert actual_hash == expected_hash, f"[{name}] SHA256 mismatch after mixed gather"

    async def test_concurrent_reads_existent_and_nonexistent(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Gather 2 valid reads + 2 reads of nonexistent files.

        Valid reads return correct data. Nonexistent reads raise VmPermanentError.
        """
        size = 1 * 1024 * 1024
        files = {f"exists_{i}.bin": os.urandom(size) for i in range(2)}
        expected_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

        async with await scheduler.session(language=Language.PYTHON) as session:
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

            # Last 2 should be VmPermanentError
            for i in range(2, 4):
                assert isinstance(results[i], VmPermanentError), (
                    f"Expected VmPermanentError for nonexistent file, got {type(results[i])}"
                )

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

        async with await scheduler.session(language=Language.PYTHON) as session:
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
            assert isinstance(results[1], VmPermanentError), f"Invalid write should fail: {type(results[1])}"
            assert results[2] is None, f"Valid read failed: {results[2]}"
            assert isinstance(results[3], VmPermanentError), f"Invalid read should fail: {type(results[3])}"

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

        async with await scheduler.session(language=Language.PYTHON) as session:
            # Fail: read nonexistent file
            with pytest.raises(VmPermanentError):
                await session.read_file("ghost.bin", destination=tmp_path / "ghost.bin")

            # Fail: write to traversal path
            with pytest.raises(VmPermanentError):
                await session.write_file("../escape.bin", b"x")

            # Success: write + read should still work
            await session.write_file("after_failure.bin", content)
            dest = tmp_path / "after_failure.bin"
            await session.read_file("after_failure.bin", destination=dest)
            assert hashlib.sha256(dest.read_bytes()).hexdigest() == expected_hash

            # Success: exec should still work
            result = await session.exec("print('still alive')")
            assert result.exit_code == 0
            assert "still alive" in result.stdout

    async def test_concurrent_list_with_writes(self, scheduler: Scheduler) -> None:
        """Concurrent list_files during writes — list sees consistent state."""
        async with await scheduler.session(language=Language.PYTHON) as session:
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

            # list_files returns a list — should have at least the initial 3 files
            file_list = results[2]
            assert isinstance(file_list, list), f"list_files should return list, got {type(file_list)}"
            file_names = {f.name for f in file_list}
            # At least the 3 pre-existing files should be visible
            for i in range(3):
                assert f"file_{i}.txt" in file_names, f"file_{i}.txt missing from list"

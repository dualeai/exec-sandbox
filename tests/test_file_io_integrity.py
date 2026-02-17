"""Integration tests for file I/O feature (write_file, read_file, list_files).

Tests that the file I/O APIs correctly handle:
1. Roundtrip integrity across various sizes (SHA256 verification)
2. Content types (ASCII, binary, UTF-8, null bytes, all byte values)
3. Cross-operation verification (file I/O <-> exec interoperability)
4. Basic operations (list, subdirectory, overwrite, executable permissions)

All tests require a real VM via the scheduler fixture.
"""

import hashlib
import os

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# =============================================================================
# TestFileIoRoundtrip - Write -> Read with SHA256 verification
# =============================================================================

# Parametrize sizes: 0B, 1B, 100B, 1KB, 10KB, 100KB, 1MB
ROUNDTRIP_SIZES = [
    (0, "0B"),
    (1, "1B"),
    (100, "100B"),
    (1024, "1KB"),
    (10 * 1024, "10KB"),
    (100 * 1024, "100KB"),
    (1024 * 1024, "1MB"),
]


class TestFileIoRoundtrip:
    """Write -> Read roundtrip with SHA256 verification across sizes."""

    @pytest.mark.parametrize(
        ("size", "label"),
        ROUNDTRIP_SIZES,
        ids=[label for _, label in ROUNDTRIP_SIZES],
    )
    async def test_roundtrip_integrity(self, scheduler: Scheduler, size: int, label: str) -> None:
        """Write random bytes, read back, verify SHA256 hash matches."""
        content = os.urandom(size) if size > 0 else b""
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file(f"roundtrip_{label}.bin", content)
            result = await session.read_file(f"roundtrip_{label}.bin")

        actual_hash = hashlib.sha256(result).hexdigest()

        assert len(result) == size, f"[{label}] Size mismatch: got {len(result)}, expected {size}"
        assert actual_hash == expected_hash, (
            f"[{label}] SHA256 mismatch - data corruption!\nExpected: {expected_hash}\nActual:   {actual_hash}"
        )


# =============================================================================
# TestFileContentTypes - Various content encodings and byte patterns
# =============================================================================


class TestFileContentTypes:
    """Tests for various content types surviving the write/read roundtrip."""

    async def test_ascii_roundtrip(self, scheduler: Scheduler) -> None:
        """Plain ASCII text survives roundtrip."""
        content = b"Hello, World! The quick brown fox jumps over the lazy dog. 0123456789"
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("ascii.txt", content)
            result = await session.read_file("ascii.txt")

        assert result == content
        assert hashlib.sha256(result).hexdigest() == expected_hash

    async def test_binary_roundtrip(self, scheduler: Scheduler) -> None:
        """Random binary data survives roundtrip."""
        content = os.urandom(4096)
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("binary.bin", content)
            result = await session.read_file("binary.bin")

        assert hashlib.sha256(result).hexdigest() == expected_hash

    async def test_utf8_emoji_and_cjk(self, scheduler: Scheduler) -> None:
        """UTF-8 content with emoji and CJK characters survives roundtrip."""
        text = "Hello World cafe\u0301 \u00f1 \u4f60\u597d \U0001f389\U0001f680\U0001f30d \u3053\u3093\u306b\u3061\u306f \ud55c\uad6d\uc5b4"
        content = text.encode("utf-8")
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("utf8.txt", content)
            result = await session.read_file("utf8.txt")

        assert result == content
        assert hashlib.sha256(result).hexdigest() == expected_hash

    async def test_null_bytes_in_content(self, scheduler: Scheduler) -> None:
        """Content containing null bytes survives roundtrip."""
        content = b"before\x00middle\x00\x00after\x00"
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("nulls.bin", content)
            result = await session.read_file("nulls.bin")

        assert result == content
        assert hashlib.sha256(result).hexdigest() == expected_hash

    async def test_all_256_byte_values(self, scheduler: Scheduler) -> None:
        """All 256 possible byte values survive roundtrip."""
        content = bytes(range(256))
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("all_bytes.bin", content)
            result = await session.read_file("all_bytes.bin")

        assert result == content
        assert len(result) == 256
        assert hashlib.sha256(result).hexdigest() == expected_hash

    async def test_newline_variants_survive(self, scheduler: Scheduler) -> None:
        r"""Various newline styles (\n, \r\n, \r) survive roundtrip unchanged."""
        content = b"unix\nwindows\r\nold-mac\rtrailing\n"
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("newlines.txt", content)
            result = await session.read_file("newlines.txt")

        assert result == content
        assert hashlib.sha256(result).hexdigest() == expected_hash
        # Verify no newline translation occurred
        assert b"\n" in result
        assert b"\r\n" in result
        assert b"\rtrailing" in result


# =============================================================================
# TestCrossOperationVerification - file I/O <-> exec interoperability
# =============================================================================


class TestCrossOperationVerification:
    """Verify files written by file I/O APIs are accessible to exec and vice versa."""

    async def test_write_file_then_exec_reads(self, scheduler: Scheduler) -> None:
        """File written via write_file is readable by session.exec."""
        content = b"Hello from write_file API"

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("message.txt", content)

            result = await session.exec(
                'with open("/home/user/message.txt", "rb") as f:\n    data = f.read()\nprint(data)'
            )

            assert result.exit_code == 0
            assert "Hello from write_file API" in result.stdout

    async def test_exec_writes_then_read_file(self, scheduler: Scheduler) -> None:
        """File written by session.exec is readable via read_file."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            result = await session.exec(
                'with open("/home/user/from_exec.txt", "wb") as f:\n    f.write(b"Hello from exec")'
            )
            assert result.exit_code == 0

            data = await session.read_file("from_exec.txt")

        assert data == b"Hello from exec"

    async def test_write_file_exec_modifies_read_file(self, scheduler: Scheduler) -> None:
        """Write via API, modify via exec, read back via API."""
        original = b"original content"

        async with await scheduler.session(language=Language.PYTHON) as session:
            # Write original via API
            await session.write_file("modify_me.txt", original)

            # Modify via exec
            result = await session.exec(
                'with open("/home/user/modify_me.txt", "rb") as f:\n'
                "    data = f.read()\n"
                'with open("/home/user/modify_me.txt", "wb") as f:\n'
                '    f.write(data + b" plus exec")'
            )
            assert result.exit_code == 0

            # Read back via API
            data = await session.read_file("modify_me.txt")

        assert data == b"original content plus exec"

    async def test_write_binary_exec_verifies_hash(self, scheduler: Scheduler) -> None:
        """Write binary via API, exec computes SHA256, verify it matches."""
        content = os.urandom(2048)
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("check.bin", content)

            result = await session.exec(
                "import hashlib\n"
                'with open("/home/user/check.bin", "rb") as f:\n'
                "    data = f.read()\n"
                "print(hashlib.sha256(data).hexdigest())"
            )

            assert result.exit_code == 0
            assert expected_hash in result.stdout


# =============================================================================
# TestFileIoBasicOperations - list, subdirectory, overwrite, executable
# =============================================================================


class TestFileIoBasicOperations:
    """Basic file I/O operations: list, subdirectory, overwrite, executable."""

    async def test_write_and_list(self, scheduler: Scheduler) -> None:
        """Written file appears in list_files output."""
        content = b"listed content"

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("listed.txt", content)
            entries = await session.list_files()

        names = [e.name for e in entries]
        assert "listed.txt" in names

        # Verify the entry metadata
        entry = next(e for e in entries if e.name == "listed.txt")
        assert entry.is_dir is False
        assert entry.size == len(content)

    async def test_write_to_subdirectory(self, scheduler: Scheduler) -> None:
        """Writing to a subdirectory path creates intermediate directories."""
        content = b"nested content"

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("subdir/nested/file.txt", content)
            result = await session.read_file("subdir/nested/file.txt")

        assert result == content

    async def test_list_subdirectory(self, scheduler: Scheduler) -> None:
        """list_files on a subdirectory returns its contents."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("mydir/a.txt", b"aaa")
            await session.write_file("mydir/b.txt", b"bb")
            entries = await session.list_files("mydir")

        names = sorted(e.name for e in entries)
        assert "a.txt" in names
        assert "b.txt" in names

    async def test_overwrite_file(self, scheduler: Scheduler) -> None:
        """Overwriting a file replaces its content entirely."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("overwrite.txt", b"first version")
            await session.write_file("overwrite.txt", b"second version")
            result = await session.read_file("overwrite.txt")

        assert result == b"second version"

    async def test_make_executable(self, scheduler: Scheduler) -> None:
        """File written with make_executable=True can be executed."""
        script = b"#!/bin/sh\necho EXEC_OK\n"

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("run.sh", script, make_executable=True)

            result = await session.exec("import subprocess; print(subprocess.check_output('/home/user/run.sh').decode(), end='')")

            assert result.exit_code == 0
            assert "EXEC_OK" in result.stdout

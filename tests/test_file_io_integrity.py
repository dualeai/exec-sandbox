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
from pathlib import Path

import pytest

from exec_sandbox.constants import FILE_TRANSFER_CHUNK_SIZE
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec

# =============================================================================
# TestFileIoRoundtrip - Write -> Read with SHA256 verification
# =============================================================================

_CHUNK = FILE_TRANSFER_CHUNK_SIZE

# Parametrize sizes: 0B, 1B, 100B, 1KB, 10KB, 100KB, chunk boundaries, 1MB
ROUNDTRIP_SIZES = [
    (0, "0B"),
    (1, "1B"),
    (100, "100B"),
    (1024, "1KB"),
    (10 * 1024, "10KB"),
    (100 * 1024, "100KB"),
    (_CHUNK - 1, "chunk-1"),
    (_CHUNK, "chunk"),
    (_CHUNK + 1, "chunk+1"),
    (2 * _CHUNK, "2chunks"),
    (2 * _CHUNK + 1, "2chunks+1"),
    (1024 * 1024, "1MB"),
]


class TestFileIoRoundtrip:
    """Write -> Read roundtrip with SHA256 verification across sizes."""

    @pytest.mark.parametrize(
        ("size", "label"),
        ROUNDTRIP_SIZES,
        ids=[label for _, label in ROUNDTRIP_SIZES],
    )
    async def test_roundtrip_integrity(self, scheduler: Scheduler, size: int, label: str, tmp_path: Path) -> None:
        """Write random bytes, read back, verify SHA256 hash matches."""
        content = os.urandom(size) if size > 0 else b""
        expected_hash = hashlib.sha256(content).hexdigest()
        dest = tmp_path / f"roundtrip_{label}.bin"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file(f"roundtrip_{label}.bin", content)
            await session.read_file(f"roundtrip_{label}.bin", destination=dest)

        result = dest.read_bytes()
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

    async def test_ascii_roundtrip(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Plain ASCII text survives roundtrip."""
        content = b"Hello, World! The quick brown fox jumps over the lazy dog. 0123456789"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("ascii.txt", content)
            await session.read_file("ascii.txt", destination=tmp_path / "ascii.txt")

        assert (tmp_path / "ascii.txt").read_bytes() == content

    async def test_utf8_emoji_and_cjk(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """UTF-8 content with emoji and CJK characters survives roundtrip."""
        text = "Hello World cafe\u0301 \u00f1 \u4f60\u597d \U0001f389\U0001f680\U0001f30d \u3053\u3093\u306b\u3061\u306f \ud55c\uad6d\uc5b4"
        content = text.encode("utf-8")

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("utf8.txt", content)
            await session.read_file("utf8.txt", destination=tmp_path / "utf8.txt")

        assert (tmp_path / "utf8.txt").read_bytes() == content

    async def test_null_bytes_in_content(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Content containing null bytes survives roundtrip."""
        content = b"before\x00middle\x00\x00after\x00"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("nulls.bin", content)
            await session.read_file("nulls.bin", destination=tmp_path / "nulls.bin")

        assert (tmp_path / "nulls.bin").read_bytes() == content

    async def test_all_256_byte_values(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """All 256 possible byte values survive roundtrip."""
        content = bytes(range(256))

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("all_bytes.bin", content)
            await session.read_file("all_bytes.bin", destination=tmp_path / "all_bytes.bin")

        assert (tmp_path / "all_bytes.bin").read_bytes() == content

    async def test_newline_variants_survive(self, scheduler: Scheduler, tmp_path: Path) -> None:
        r"""Various newline styles (\n, \r\n, \r) survive roundtrip unchanged."""
        content = b"unix\nwindows\r\nold-mac\rtrailing\n"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("newlines.txt", content)
            await session.read_file("newlines.txt", destination=tmp_path / "newlines.txt")

        # Byte-equal: no newline translation
        assert (tmp_path / "newlines.txt").read_bytes() == content


# =============================================================================
# TestCrossOperationVerification - file I/O <-> exec interoperability
# =============================================================================


class TestCrossOperationVerification:
    """Verify files written by file I/O APIs are accessible to exec and vice versa."""

    async def test_write_file_exec_modifies_read_file(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Write via API, modify via exec, read back via API."""
        original = b"original content"
        dest = tmp_path / "modify_me.txt"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
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
            await session.read_file("modify_me.txt", destination=dest)

        assert dest.read_bytes() == b"original content plus exec"

    async def test_write_binary_exec_verifies_hash(self, scheduler: Scheduler) -> None:
        """Write binary via API, exec computes SHA256, verify it matches."""
        content = os.urandom(2048)
        expected_hash = hashlib.sha256(content).hexdigest()

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("check.bin", content)

            result = await session.exec(
                "import hashlib\n"
                'with open("/home/user/check.bin", "rb") as f:\n'
                "    data = f.read()\n"
                "print(hashlib.sha256(data).hexdigest())"
            )

            assert result.exit_code == 0
            assert result.stdout == expected_hash + "\n"


# =============================================================================
# TestFileIoBasicOperations - list, subdirectory, overwrite, executable
# =============================================================================


class TestFileIoBasicOperations:
    """Basic file I/O operations: list, subdirectory, overwrite, executable."""

    async def test_write_to_subdirectory(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Writing to a subdirectory path creates intermediate directories."""
        content = b"nested content"
        dest = tmp_path / "nested_file.txt"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("subdir/nested/file.txt", content)
            await session.read_file("subdir/nested/file.txt", destination=dest)

        assert dest.read_bytes() == content

    async def test_list_subdirectory(self, scheduler: Scheduler) -> None:
        """list_files on a subdirectory returns its contents."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("mydir/a.txt", b"aaa")
            await session.write_file("mydir/b.txt", b"bb")
            entries = await session.list_files("mydir")

        assert sorted((e.name, e.is_dir, e.size) for e in entries) == [("a.txt", False, 3), ("b.txt", False, 2)]

    async def test_overwrite_file(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Overwriting a file replaces its content entirely."""
        dest = tmp_path / "overwrite.txt"
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("overwrite.txt", b"first version")
            await session.write_file("overwrite.txt", b"second version")
            await session.read_file("overwrite.txt", destination=dest)

        assert dest.read_bytes() == b"second version"

    async def test_make_executable(self, scheduler: Scheduler) -> None:
        """File written with make_executable=True can be executed."""
        script = b"#!/bin/sh\necho EXEC_OK\n"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("run.sh", script, make_executable=True)

            result = await session.exec(
                "import subprocess; print(subprocess.check_output('/home/user/run.sh').decode(), end='')"
            )

            assert result.exit_code == 0, result.stderr
            assert result.stdout == "EXEC_OK\n"

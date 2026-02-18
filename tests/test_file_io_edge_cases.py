"""Edge case tests for file I/O operations.

Tests path edge cases, content edge cases, and input type dispatch.
Requires VM (integration tests).
"""

from pathlib import Path

import pytest

from exec_sandbox.constants import MAX_FILE_SIZE_BYTES
from exec_sandbox.exceptions import SessionClosedError, VmPermanentError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# ============================================================================
# Path Edge Cases
# ============================================================================


class TestPathEdgeCases:
    """Edge cases for file paths."""

    async def test_dotdot_path_rejected(self, scheduler: Scheduler) -> None:
        """Pure '..' path is rejected (traversal)."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises(VmPermanentError):
                await session.write_file("..", b"data")

    async def test_spaces_in_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Spaces in filename work."""
        dest = tmp_path / "my_file.txt"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("my file.txt", b"hello")
            await session.read_file("my file.txt", destination=dest)
            assert dest.read_bytes() == b"hello"

    async def test_unicode_in_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Unicode characters in filename work."""
        dest = tmp_path / "unicode.txt"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("日本語.txt", b"konnichiwa")
            await session.read_file("日本語.txt", destination=dest)
            assert dest.read_bytes() == b"konnichiwa"

    async def test_deeply_nested_mkdir_p(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Deeply nested path creates intermediate directories."""
        dest = tmp_path / "deep.txt"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("a/b/c/d/deep.txt", b"deep")
            await session.read_file("a/b/c/d/deep.txt", destination=dest)
            assert dest.read_bytes() == b"deep"

    async def test_hidden_file(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Hidden files (dot prefix) work."""
        dest = tmp_path / "hidden.bin"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file(".hidden", b"secret")
            await session.read_file(".hidden", destination=dest)
            assert dest.read_bytes() == b"secret"

    async def test_dots_in_filename(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Multiple dots in filename work."""
        dest = tmp_path / "file.tar.gz"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("file.tar.gz", b"compressed")
            await session.read_file("file.tar.gz", destination=dest)
            assert dest.read_bytes() == b"compressed"

    async def test_255_char_filename(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Exactly 255-char filename works (POSIX max)."""
        name = "a" * 251 + ".txt"  # 255 chars total
        dest = tmp_path / "long_name.bin"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file(name, b"long name")
            await session.read_file(name, destination=dest)
            assert dest.read_bytes() == b"long name"

    async def test_256_char_filename_rejected(self, scheduler: Scheduler) -> None:
        """256-char filename is rejected."""
        name = "a" * 256
        async with await scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises((VmPermanentError, ValueError)):
                await session.write_file(name, b"too long")


# ============================================================================
# Content Edge Cases
# ============================================================================


class TestContentEdgeCases:
    """Edge cases for file content."""

    async def test_empty_file(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Empty file write/read roundtrip."""
        dest = tmp_path / "empty.txt"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("empty.txt", b"")
            await session.read_file("empty.txt", destination=dest)
            assert dest.read_bytes() == b""

    async def test_single_byte(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Single byte write/read roundtrip."""
        dest = tmp_path / "one.bin"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("one.bin", b"\x42")
            await session.read_file("one.bin", destination=dest)
            assert dest.read_bytes() == b"\x42"

    async def test_overwrite_file(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Second write overwrites first content."""
        dest = tmp_path / "overwrite.txt"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("overwrite.txt", b"first")
            await session.write_file("overwrite.txt", b"second")
            await session.read_file("overwrite.txt", destination=dest)
            assert dest.read_bytes() == b"second"

    async def test_read_nonexistent_file(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Reading nonexistent file raises error."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises(VmPermanentError):
                await session.read_file("does_not_exist.txt", destination=tmp_path / "out.bin")

    async def test_make_executable(self, scheduler: Scheduler) -> None:
        """make_executable=True allows execution of the file."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            script = b"#!/bin/sh\necho 'script ran'"
            await session.write_file("run.sh", script, make_executable=True)
            result = await session.exec(
                "import subprocess; print(subprocess.check_output(['/home/user/run.sh']).decode())"
            )
            assert result.exit_code == 0
            assert "script ran" in result.stdout

    async def test_oversized_content_rejected(self, scheduler: Scheduler) -> None:
        """Content exceeding MAX_FILE_SIZE_BYTES is rejected at the Session level."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            oversized = b"x" * (MAX_FILE_SIZE_BYTES + 1)
            with pytest.raises(ValueError, match="exceeds"):
                await session.write_file("big.bin", oversized)


# ============================================================================
# Input Type Dispatch (bytes vs Path)
# ============================================================================


class TestWriteFileInputTypes:
    """Test bytes vs Path dispatch in Session.write_file."""

    async def test_write_from_bytes(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """write_file accepts bytes directly."""
        dest = tmp_path / "from_bytes.txt"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("from_bytes.txt", b"byte data")
            await session.read_file("from_bytes.txt", destination=dest)
            assert dest.read_bytes() == b"byte data"

    async def test_write_from_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """write_file accepts a local Path."""
        local_file = tmp_path / "local.txt"
        local_file.write_bytes(b"local content")
        dest = tmp_path / "from_path.txt"

        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("from_path.txt", local_file)
            await session.read_file("from_path.txt", destination=dest)
            assert dest.read_bytes() == b"local content"

    async def test_write_from_nonexistent_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """write_file raises FileNotFoundError for nonexistent Path."""
        missing = tmp_path / "does_not_exist.txt"

        async with await scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises(FileNotFoundError):
                await session.write_file("target.txt", missing)

    async def test_write_from_oversized_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """write_file raises ValueError for oversized Path."""
        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))

        async with await scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises(ValueError, match="exceeds"):
                await session.write_file("target.bin", big_file)


# ============================================================================
# Session Lifecycle with File I/O
# ============================================================================


class TestFileIoSessionLifecycle:
    """File I/O operations respect session lifecycle."""

    async def test_file_ops_after_close_raises(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """File operations after session.close() raise SessionClosedError."""
        session = await scheduler.session(language=Language.PYTHON)
        await session.close()

        with pytest.raises(SessionClosedError):
            await session.write_file("test.txt", b"data")

        with pytest.raises(SessionClosedError):
            await session.read_file("test.txt", destination=tmp_path / "test.bin")

        with pytest.raises(SessionClosedError):
            await session.list_files()

    async def test_write_file_before_exec(self, scheduler: Scheduler) -> None:
        """write_file works immediately after session creation (before any exec)."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("early.txt", b"early bird")
            result = await session.exec("print(open('early.txt').read())")
            assert "early bird" in result.stdout

    async def test_file_persists_across_execs(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Files written persist across multiple exec calls."""
        dest = tmp_path / "persist.txt"
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("persist.txt", b"persistent data")
            await session.exec("x = 1")  # Unrelated exec
            await session.exec("y = 2")  # Another unrelated exec
            await session.read_file("persist.txt", destination=dest)
            assert dest.read_bytes() == b"persistent data"

    async def test_files_and_state_coexist(self, scheduler: Scheduler) -> None:
        """File I/O and REPL state coexist."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("data.txt", b"42")
            await session.exec("x = 100")
            result = await session.exec("print(x + int(open('data.txt').read()))")
            assert result.exit_code == 0
            assert "142" in result.stdout


# ============================================================================
# list_files Tests
# ============================================================================


class TestListFiles:
    """Tests for list_files operation."""

    async def test_list_empty_root(self, scheduler: Scheduler) -> None:
        """list_files on fresh sandbox root returns a list (may have pre-existing files)."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            files = await session.list_files()
            assert isinstance(files, list)

    async def test_list_after_writes(self, scheduler: Scheduler) -> None:
        """list_files shows written files."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("a.txt", b"a")
            await session.write_file("b.txt", b"bb")
            files = await session.list_files()
            names = {f.name for f in files}
            assert "a.txt" in names
            assert "b.txt" in names

    async def test_list_subdirectory(self, scheduler: Scheduler) -> None:
        """list_files on a subdirectory."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("subdir/x.txt", b"x")
            await session.write_file("subdir/y.txt", b"yy")
            files = await session.list_files("subdir")
            names = {f.name for f in files}
            assert "x.txt" in names
            assert "y.txt" in names

    async def test_list_nonexistent_directory(self, scheduler: Scheduler) -> None:
        """list_files on nonexistent directory raises error."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises(VmPermanentError):
                await session.list_files("does_not_exist")

    async def test_file_info_fields(self, scheduler: Scheduler) -> None:
        """FileInfo has correct name, is_dir, and size fields."""
        async with await scheduler.session(language=Language.PYTHON) as session:
            await session.write_file("info_test.txt", b"hello")
            await session.exec("import os; os.makedirs('/home/user/info_dir', exist_ok=True)")
            files = await session.list_files()
            file_map = {f.name: f for f in files}

            assert "info_test.txt" in file_map
            assert file_map["info_test.txt"].is_dir is False
            assert file_map["info_test.txt"].size == 5

            assert "info_dir" in file_map
            assert file_map["info_dir"].is_dir is True

"""Edge case tests for file I/O operations.

Tests path edge cases, content edge cases, input type dispatch,
and write abort/error recovery (no partial file at the final path).
Requires VM (integration tests).
"""

import asyncio
import errno
import io
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from exec_sandbox.constants import FILE_TRANSFER_CHUNK_SIZE
from exec_sandbox.exceptions import ErrorCode, InvalidInputError, SessionClosedError, TransientError
from exec_sandbox.guest_agent_protocol import StreamingErrorMessage
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec

_FILE_SIZE_LIMIT = 500 * 1024 * 1024
"""The largest file write_file takes (500 MiB), written out: a change of the limit shows here."""

# ============================================================================
# Path Edge Cases
# ============================================================================


class TestPathEdgeCases:
    """Edge cases for file paths."""

    async def test_dotdot_path_rejected(self, scheduler: Scheduler) -> None:
        """Pure '..' path is rejected (traversal)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                await session.write_file("..", b"data")

    async def test_spaces_in_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Spaces in filename work."""
        dest = tmp_path / "my_file.txt"
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("my file.txt", b"hello")
            await session.read_file("my file.txt", destination=dest)
            assert dest.read_bytes() == b"hello"

    async def test_unicode_in_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Unicode characters in filename work."""
        dest = tmp_path / "unicode.txt"
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("日本語.txt", b"konnichiwa")
            await session.read_file("日本語.txt", destination=dest)
            assert dest.read_bytes() == b"konnichiwa"

    async def test_guest_code_can_add_files_to_folders_a_write_made(self, scheduler: Scheduler) -> None:
        """write_file("out/sub/a.txt") makes out/ and out/sub/ for the sandbox user, as the file
        itself: guest code can then add files to them (they are not root's)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("out/sub/a.txt", b"a")
            result = await session.exec(
                "open('/home/user/out/b.txt', 'w').write('b'); open('/home/user/out/sub/c.txt', 'w').write('c');"
                " print('ok')"
            )

        assert (result.exit_code, result.stdout) == (0, "ok\n"), result.stderr

    async def test_hidden_file(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Hidden files (dot prefix) work."""
        dest = tmp_path / "hidden.bin"
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file(".hidden", b"secret")
            await session.read_file(".hidden", destination=dest)
            assert dest.read_bytes() == b"secret"

    async def test_dots_in_filename(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Multiple dots in filename work."""
        dest = tmp_path / "file.tar.gz"
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("file.tar.gz", b"compressed")
            await session.read_file("file.tar.gz", destination=dest)
            assert dest.read_bytes() == b"compressed"

    async def test_255_char_filename(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Exactly 255-char filename works (POSIX max)."""
        name = "a" * 251 + ".txt"  # 255 chars total
        dest = tmp_path / "long_name.bin"
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file(name, b"long name")
            await session.read_file(name, destination=dest)
            assert dest.read_bytes() == b"long name"

    async def test_256_char_filename_rejected(self, scheduler: Scheduler) -> None:
        """256-char filename is rejected."""
        name = "a" * 256
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                await session.write_file(name, b"too long")


# ============================================================================
# Content Edge Cases
# ============================================================================


class TestContentEdgeCases:
    """Edge cases for file content."""

    async def test_read_nonexistent_file(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Reading nonexistent file raises error."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
                await session.read_file("does_not_exist.txt", destination=tmp_path / "out.bin")

    async def test_oversized_content_rejected(self, scheduler: Scheduler) -> None:
        """Content over the 500 MiB file size limit is rejected at the Session level (ARGUMENT_INVALID)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            oversized = b"x" * (_FILE_SIZE_LIMIT + 1)
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID) as exc:
                await session.write_file("big.bin", oversized)
            assert exc.value.message == "Content is 524288001 bytes, exceeds 524288000"


# ============================================================================
# Input Type Dispatch (bytes vs Path)
# ============================================================================


class TestWriteFileInputTypes:
    """Test bytes vs Path dispatch in Session.write_file."""

    async def test_write_from_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """write_file accepts a local Path."""
        local_file = tmp_path / "local.txt"
        local_file.write_bytes(b"local content")
        dest = tmp_path / "from_path.txt"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("from_path.txt", local_file)
            await session.read_file("from_path.txt", destination=dest)
            assert dest.read_bytes() == b"local content"

    async def test_write_from_nonexistent_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """write_file raises InvalidInputError ARGUMENT_INVALID for nonexistent Path."""
        missing = tmp_path / "does_not_exist.txt"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(
                InvalidInputError,
                check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
                match=r"^Source file cannot be opened \(No such file or directory\): ",
            ):
                await session.write_file("target.txt", missing)

    async def test_write_from_oversized_path(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """write_file raises InvalidInputError ARGUMENT_INVALID for oversized Path."""
        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * (_FILE_SIZE_LIMIT + 1))

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID) as exc:
                await session.write_file("target.bin", big_file)
            assert exc.value.message == "File big.bin is 524288001 bytes, exceeds 524288000"


# ============================================================================
# Session Lifecycle with File I/O
# ============================================================================


class TestFileIoSessionLifecycle:
    """File I/O operations respect session lifecycle."""

    async def test_file_ops_after_close_raises(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """File operations after session.close() raise SessionClosedError."""
        session = await scheduler.new(VmSpec(language=Language.PYTHON))
        await session.close()

        with pytest.raises(SessionClosedError):
            await session.write_file("test.txt", b"data")

        with pytest.raises(SessionClosedError):
            await session.read_file("test.txt", destination=tmp_path / "test.bin")

        with pytest.raises(SessionClosedError):
            await session.list_files()

    async def test_write_file_before_exec(self, scheduler: Scheduler) -> None:
        """write_file works immediately after session creation (before any exec)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("early.txt", b"early bird")
            result = await session.exec("print(open('early.txt').read())")
            assert (result.exit_code, result.stdout) == (0, "early bird\n"), result.stderr

    async def test_file_persists_across_execs(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """Files written persist across multiple exec calls."""
        dest = tmp_path / "persist.txt"
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("persist.txt", b"persistent data")
            await session.exec("x = 1")  # Unrelated exec
            await session.exec("y = 2")  # Another unrelated exec
            await session.read_file("persist.txt", destination=dest)
            assert dest.read_bytes() == b"persistent data"

    async def test_files_and_state_coexist(self, scheduler: Scheduler) -> None:
        """File I/O and REPL state coexist."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("data.txt", b"42")
            await session.exec("x = 100")
            result = await session.exec("print(x + int(open('data.txt').read()))")
            assert (result.exit_code, result.stdout) == (0, "142\n"), result.stderr


# ============================================================================
# list_files Tests
# ============================================================================


class TestListFiles:
    """Tests for list_files operation."""

    async def test_list_empty_root(self, scheduler: Scheduler) -> None:
        """list_files on a fresh sandbox root gives the entries that os.listdir gives in the
        guest, sorted by name."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            files = await session.list_files()
            result = await session.exec("import json, os; print(json.dumps(sorted(os.listdir('/home/user'))))")
            assert result.exit_code == 0, result.stderr
            assert [f.name for f in files] == json.loads(result.stdout)

    async def test_list_nonexistent_directory(self, scheduler: Scheduler) -> None:
        """list_files on nonexistent directory raises error."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
                await session.list_files("does_not_exist")

    async def test_file_info_fields(self, scheduler: Scheduler) -> None:
        """FileInfo has the name, is_dir and size of each entry: a written file of 5 bytes, and a
        folder that guest code made (size 0)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("info_test.txt", b"hello")
            await session.exec("import os; os.makedirs('/home/user/info_dir', exist_ok=True)")
            files = await session.list_files()
            file_map = {f.name: (f.is_dir, f.size) for f in files}

            assert (file_map["info_test.txt"], file_map["info_dir"]) == ((False, 5), (True, 0))


# ============================================================================
# list_files Sparse File Defense
# ============================================================================


class TestListFilesSparseDefense:
    """list_files reports actual disk usage, not apparent size.

    Sparse files (created via ftruncate/truncate) have large apparent size
    (metadata.len()) but zero allocated blocks. list_files uses
    min(len, blocks*512) to deflate sparse files while preserving correct
    reporting for real files.
    """

    # --- Normal cases: real files report correct size ---

    async def test_binary_file_size(self, scheduler: Scheduler) -> None:
        """Binary file reports exact byte count."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            content = os.urandom(4097)  # Just over one page
            await session.write_file("binary.bin", content)
            files = await session.list_files()
            file_map = {f.name: f for f in files}
            assert file_map["binary.bin"].size == 4097

    async def test_empty_file_size_zero(self, scheduler: Scheduler) -> None:
        """Empty file reports size 0."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("empty.txt", b"")
            files = await session.list_files()
            file_map = {f.name: f for f in files}
            assert file_map["empty.txt"].size == 0

    # --- Edge cases: sparse files are deflated ---

    async def test_partially_written_sparse_file(self, scheduler: Scheduler) -> None:
        """Partially written sparse file reports actual data size, not apparent size.

        Write 4KB at offset 0, then truncate to 1MB: one 4 KiB block of the home disk (ext4),
        so min(len, blocks * 512) = min(1 MiB, 4096) = 4096.
        """
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            code = """\
import os
with open('/home/user/partial', 'wb') as f:
    f.write(b'x' * 4096)
os.truncate('/home/user/partial', 1024 * 1024)
# Verify apparent size vs actual blocks
s = os.stat('/home/user/partial')
print(f'APPARENT:{s.st_size}')
print(f'BLOCKS:{s.st_blocks}')
print(f'ACTUAL:{s.st_blocks * 512}')
"""
            r = await session.exec(code)
            assert (r.exit_code, r.stdout) == (0, "APPARENT:1048576\nBLOCKS:8\nACTUAL:4096\n"), r.stderr

            files = await session.list_files()
            file_map = {f.name: f for f in files}
            # The data (one 4 KiB ext4 block), not the apparent 1 MiB
            assert file_map["partial"].size == 4096

    async def test_one_byte_file_reports_one(self, scheduler: Scheduler) -> None:
        """1-byte file reports size 1 (not inflated to block size).

        min(1, 4096) = 1 — the len() side wins for small real files.
        """
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("tiny.txt", b"x")
            files = await session.list_files()
            file_map = {f.name: f for f in files}
            assert file_map["tiny.txt"].size == 1

    async def test_symlink_in_listing(self, scheduler: Scheduler) -> None:
        """Symlinks appear in listing (size 0: ext4 keeps a short link target in the inode)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("target.txt", b"data")
            await session.exec("import os; os.symlink('/home/user/target.txt', '/home/user/link.txt')")
            files = await session.list_files()
            file_map = {f.name: f for f in files}
            assert "link.txt" in file_map
            # A short symlink on the ext4 home disk has blocks=0 (inline in the inode), so min(len, 0)=0
            assert file_map["link.txt"].size == 0

    # --- Weird cases: unusual patterns ---

    async def test_multiple_sparse_files_all_deflated(self, scheduler: Scheduler) -> None:
        """Creating many sparse files — all report 0 in listing."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            code = """\
import os
for i in range(10):
    path = f'/home/user/sparse_{i}'
    open(path, 'w').close()
    os.truncate(path, (i + 1) * 1024 * 1024)  # 1MB to 10MB apparent
"""
            await session.exec(code)
            files = await session.list_files()
            file_map = {f.name: f for f in files}
            for i in range(10):
                name = f"sparse_{i}"
                assert name in file_map, f"{name} missing from listing"
                assert file_map[name].size == 0, f"{name} should report 0, got {file_map[name].size}"

    async def test_sparse_then_fill_reports_real_size(self, scheduler: Scheduler) -> None:
        """A sparse file opened again with 'wb' (truncated to 0) and filled with 50,000 bytes
        reports exactly 50,000: all of its blocks hold data."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            code = """\
import os
# Create sparse
open('/home/user/fill_me', 'w').close()
os.truncate('/home/user/fill_me', 1024 * 1024)
# Now write real data
with open('/home/user/fill_me', 'wb') as f:
    f.write(b'A' * 50000)
"""
            await session.exec(code)
            files = await session.list_files()
            file_map = {f.name: f for f in files}
            assert file_map["fill_me"].size == 50000

    # --- Out of bounds: extreme sparse files ---

    async def test_huge_sparse_file_reports_zero(self, scheduler: Scheduler) -> None:
        """100GB sparse file reports size 0 — the attack vector this defense blocks.

        os.truncate("huge", 100*1024**3) creates 100GB apparent size with 0
        actual bytes on the home disk. Without min(len, blocks*512), list_files
        would report 107,374,182,400 bytes.
        """
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            code = """\
import os
open('/home/user/huge', 'w').close()
os.truncate('/home/user/huge', 100 * 1024**3)
s = os.stat('/home/user/huge')
print(f'APPARENT:{s.st_size}')
print(f'BLOCKS:{s.st_blocks}')
"""
            r = await session.exec(code)
            assert (r.exit_code, r.stdout) == (0, "APPARENT:107374182400\nBLOCKS:0\n"), r.stderr

            files = await session.list_files()
            file_map = {f.name: f for f in files}
            assert file_map["huge"].size == 0

    async def test_sparse_file_still_readable(self, scheduler: Scheduler) -> None:
        """A sparse file that list_files reports as size 0 reads back with read_file at its
        apparent size, as zeroes: the listing deflation does not change read_file.
        """
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            code = """\
import os
# Create small sparse file (256 bytes apparent)
open('/home/user/readable_sparse', 'w').close()
os.truncate('/home/user/readable_sparse', 256)
"""
            await session.exec(code)

            # list_files should show 0
            files = await session.list_files()
            file_map = {f.name: f for f in files}
            assert file_map["readable_sparse"].size == 0

            read_back = io.BytesIO()
            await session.read_file("readable_sparse", destination=read_back)
            assert read_back.getvalue() == bytes(256)


# ============================================================================
# Write Abort / Error Recovery (no partial file at the final path)
# ============================================================================


class TestWriteAbortRecovery:
    """A write that does not end with file_end leaves no file at its final path.

    The guest writes into a temporary file and renames it into place only at file_end
    (an empty chunk to its writer task is the end marker). A write that stops before it
    (a disconnect, a frame the guest cannot decode) removes the temporary file.
    A disconnect in a write is tested through write_file (a source that fails);
    a bad frame is not, as write_file sends only valid frames.
    """

    async def test_invalid_chunk_no_partial_file(self, scheduler: Scheduler) -> None:
        """Invalid base64 chunk must NOT leave a partial file at the final path.

        Sends a write_file header followed by a chunk with invalid base64 data.
        The guest agent should return an error and clean up the tmp file.
        The frames go on the VM channel: write_file never sends a bad chunk.
        """
        target = "bad_chunk.bin"

        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            channel = session._vm.channel
            await channel.connect(5)
            op_id = uuid4().hex

            # Register op so we can receive the error response
            op_queue = channel.register_op(op_id, changes_state=True)

            try:
                for frame in (
                    # The header has every field: the guest refuses a header without one
                    {"action": "write_file", "op_id": op_id, "path": target, "make_executable": False},
                    {"action": "file_chunk", "op_id": op_id, "data": "!!!NOT-VALID-BASE64!!!"},  # not decodeable
                    {"action": "file_end", "op_id": op_id},  # so the guest agent doesn't hang
                ):
                    await channel.enqueue_raw(json.dumps(frame).encode() + b"\n")

                # Wait for response — should be an error
                response = await asyncio.wait_for(op_queue.get(), timeout=10)
                assert isinstance(response, StreamingErrorMessage), (
                    f"Expected error response, got {type(response).__name__}"
                )
            finally:
                channel.unregister_op(op_id)

            # Verify no file at the final path
            result = await session.exec("import os; print(os.path.exists('/home/user/bad_chunk.bin'))")
            assert (result.exit_code, result.stdout) == (0, "False\n"), "a partial file stayed at the final path"

    async def test_source_read_failure_writes_nothing_and_keeps_the_session(self, scheduler: Scheduler) -> None:
        """The local source fails after the header and the first chunks: write_file closes the
        channel (no file_end), so the guest drops the write. RETRY (LOCAL_DISK) holds: no file
        at the target, the session stays open, and the same call with a working source
        succeeds. Two full chunks are read before the failure: the compressor sends data to
        the guest before the disconnect."""

        class FailingSource(io.BytesIO):
            def read(self, size: int | None = -1) -> bytes:  # fails once two chunks are read
                if self.tell() >= 2 * FILE_TRANSFER_CHUNK_SIZE:
                    raise OSError(errno.EIO, os.strerror(errno.EIO))
                return super().read(size)

        content = os.urandom(3 * FILE_TRANSFER_CHUNK_SIZE)
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            source = FailingSource(content)
            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.LOCAL_DISK):
                await session.write_file("partial.bin", source)
            assert not session.closed
            assert source.tell() == 0  # set back: the same call can run again

            result = await session.exec("import os; print(os.path.exists('/home/user/partial.bin'))")
            assert (result.exit_code, result.stdout) == (0, "False\n"), (
                "the guest committed a write that never got file_end"
            )

            await session.write_file("partial.bin", io.BytesIO(content))
            read_back = io.BytesIO()
            await session.read_file("partial.bin", destination=read_back)
            assert read_back.getvalue() == content

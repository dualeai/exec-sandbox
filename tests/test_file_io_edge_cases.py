"""Edge case tests for file I/O operations.

Tests path edge cases, content edge cases, input type dispatch, write abort/error
recovery (no partial file at the final path), and the free space of the home disk
(refusal, reservation).
Requires VM (integration tests).
"""

import ast
import asyncio
import errno
import hashlib
import io
import json
import os
import re
import threading
from pathlib import Path
from typing import IO, Any, cast
from uuid import uuid4

import pytest

from exec_sandbox import constants
from exec_sandbox.constants import (
    FILE_TRANSFER_CHUNK_SIZE,
)
from exec_sandbox.exceptions import ErrorCode, InvalidInputError, SessionClosedError, TransientError
from exec_sandbox.guest_agent_protocol import StreamingErrorMessage, WriteFileRequest
from exec_sandbox.models import Language
from exec_sandbox.qemu_vm import QemuVM
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.session import Session
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
        guest, sorted by name. The exec comes first: the first REPL start writes the REPL's own
        files there, and a VM that boots cold (no memory snapshot, no warm pool) starts its REPL
        only at its first exec."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec("import json, os; print(json.dumps(sorted(os.listdir('/home/user'))))")
            assert result.exit_code == 0, result.stderr
            files = await session.list_files()
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
                    # The header has every field that the guest requires (size is optional): the guest
                    # refuses a header without one of them
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


# ============================================================================
# Free space of the home disk
# ============================================================================


class VirtualSource(io.RawIOBase):
    """A write_file source of `size` random bytes, made on read (no host memory or disk), that
    counts and hashes the bytes read. `end` is its SEEK_END position: when it is not `size`,
    the host measures a size that the reads do not give; None: not seekable (the host sends
    no size). With `hold`, the first read sets `held` (an event of the loop that made the
    source), then waits until release() (a read of 0 bytes is never held)."""

    def __init__(self, size: int, *, end: int | None, hold: bool) -> None:
        super().__init__()
        self.size = size
        self.end = end
        self.read_bytes = 0
        self.digest = hashlib.sha256()
        self.held = asyncio.Event()
        self._position = 0
        self._hold = hold
        self._loop = asyncio.get_running_loop()
        self._released = threading.Event()

    def release(self) -> None:
        """Let the held read go on (also before it is held: it is then not held)."""
        self._released.set()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return self.end is not None

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if self.end is None:
            raise io.UnsupportedOperation("seek")
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self._position, os.SEEK_END: self.end}[whence]
        self._position = base + offset
        return self._position

    def tell(self) -> int:
        return self._position

    def readinto(self, buffer: Any) -> int:
        if not len(buffer):  # Session's zero-byte probe, on the event loop: never held
            return 0
        if self._hold:
            self._hold = False
            self._loop.call_soon_threadsafe(self.held.set)
            self._released.wait()
        n = max(0, min(len(buffer), self.size - self._position))
        data = os.urandom(n)
        buffer[:n] = data
        self.digest.update(data)
        self._position += n
        self.read_bytes += n
        return n


async def _until_held(source: VirtualSource, task: asyncio.Task[None]) -> None:
    """Wait until the write `task` is held in its first read of `source` (60 s at most); a
    write that ends first raises its own error."""
    waiter = asyncio.ensure_future(source.held.wait())
    try:
        await asyncio.wait({waiter, task}, timeout=60, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
    if not source.held.is_set():
        if task.done():
            task.result()
            raise AssertionError("the write ended before its first read")
        raise TimeoutError("the write was not held in 60 s")


async def _free_bytes(session: Session) -> int:
    """The home disk's free bytes, as guest code sees them (f_bavail)."""
    result = await session.exec("import os; s = os.statvfs('/home/user'); print(s.f_bavail * s.f_frsize)")
    assert result.exit_code == 0, result.stderr
    return int(result.stdout)


async def _fill(session: Session, *, keep: int) -> int:
    """Guest code takes the free space of the home disk but `keep` bytes (a reservation, no
    data written); returns the free bytes left."""
    result = await session.exec(
        "import os\n"
        "s = os.statvfs('/home/user')\n"
        "fd = os.open('/home/user/.fill', os.O_WRONLY | os.O_CREAT, 0o600)\n"
        f"os.posix_fallocate(fd, 0, s.f_bavail * s.f_frsize - {keep})\n"
        "os.close(fd)\n"
    )
    assert result.exit_code == 0, result.stderr
    return await _free_bytes(session)


async def _guest_names(session: Session) -> list[str]:
    """The names in /home/user, sorted, as guest code sees them."""
    result = await session.exec("import os; print(sorted(os.listdir('/home/user')))")
    assert result.exit_code == 0, result.stderr
    return ast.literal_eval(result.stdout)


class TestWriteFileSpace:
    """A write_file whose size is known (bytes, a Path, a seekable stream that can seek to its
    end) is refused before the guest writes any of it when it does not fit in the free space
    of the home disk; the guest reserves the space of a write that fits until its commit."""

    async def test_write_larger_than_the_free_space_is_refused_at_the_header(self, scheduler: Scheduler) -> None:
        """A source larger than the free space is refused at the header with the needed and free
        bytes: no folder, no temporary file and no space taken; the stream is back at its
        position, and the session goes on."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            free = await _fill(session, keep=64 * 1024 * 1024)
            source = VirtualSource(_FILE_SIZE_LIMIT, end=_FILE_SIZE_LIMIT, hold=False)
            source.seek(7)
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO) as exc_info:
                await session.write_file("sub/big.bin", cast("IO[bytes]", source))
            match = re.search(r"No space for 'sub/big\.bin': needs (\d+) bytes, (\d+) bytes free", str(exc_info.value))
            assert match is not None, str(exc_info.value)
            assert int(match[1]) == _FILE_SIZE_LIMIT - 7
            assert int(match[2]) == free
            assert source.read_bytes < source.size  # the stream stopped at the refusal
            assert source.tell() == 7  # back where the call found it
            names = await _guest_names(session)
            assert "sub" not in names, names  # refused before any folder is made
            assert not [name for name in names if name.startswith(".wr.")], names
            assert await _free_bytes(session) == free
            await session.write_file("small.bin", b"x" * 4096)  # the session goes on

    async def test_header_path_refusal_stops_a_large_source(self, scheduler: Scheduler) -> None:
        """A path the guest refuses at the header (a file in its parent path) stops the stream
        too: the refusal does not wait for the whole source. The 500 MiB fit in the default disk,
        so the free-space check passes first."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            await session.write_file("blocker", b"a file, not a folder")
            source = VirtualSource(_FILE_SIZE_LIMIT, end=_FILE_SIZE_LIMIT, hold=False)
            with pytest.raises(
                InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO, match="Not a directory"
            ):
                await session.write_file("blocker/big.bin", cast("IO[bytes]", source))
            assert source.read_bytes < source.size
            assert source.tell() == 0
            assert not session.closed

    async def test_space_check_boundary_and_reservation_refusal(self, scheduler: Scheduler, tmp_path: Path) -> None:
        """On a new 512 MB disk: a file of the free space plus one byte is
        refused by the check at the header; a file of exactly the free space passes the check
        and is refused by the reservation (ext4 needs one more block for its extent tree)."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON, disk_mb=512)) as session:
            free = await _free_bytes(session)
            # The guest's own fallocate of exactly the free space fails on this layout, else
            # the reservation step below cannot be checked
            calibration = await session.exec(
                "import os\n"
                "fd = os.open('/home/user/.cal', os.O_WRONLY | os.O_CREAT, 0o600)\n"
                "try:\n"
                f"    os.posix_fallocate(fd, 0, {free})\n"
                "    print('fits')\n"
                "except OSError as e:\n"
                "    print(e.errno)\n"
                "os.close(fd); os.unlink('/home/user/.cal'); os.sync()\n"
            )
            assert calibration.stdout.strip() == str(errno.ENOSPC), f"layout changed, recalibrate: {calibration}"
            free = await _free_bytes(session)

            over = tmp_path / "over.bin"
            with over.open("wb") as sparse:  # no host disk used
                sparse.truncate(free + 1)
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO) as over_info:
                await session.write_file("deep2/fit.bin", over)
            assert f"No space for 'deep2/fit.bin': needs {free + 1} bytes, {free} bytes free" in str(over_info.value)
            assert "deep2" not in await _guest_names(session)

            exact = tmp_path / "exact.bin"
            with exact.open("wb") as sparse:
                sparse.truncate(free)
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO) as exact_info:
                await session.write_file("deep/fit.bin", exact)
            assert f"reserve {free} bytes: No space left on device" in str(exact_info.value)
            # Its temporary file and its reservation are gone; the folder the write made stays
            result = await session.exec(
                "import os; print(sorted(os.listdir('/home/user/deep'))); os.rmdir('/home/user/deep')"
            )
            assert result.stdout.strip() == "[]", result
            assert await _free_bytes(session) == free

    async def test_reserved_space_is_kept_from_guest_code(self, running_vm: QemuVM, execute_timeout: int) -> None:
        """Guest code that fills the disk once the guest reserved the write's space (after the
        header, before the first chunk) cannot make the write fail. The write waits in its
        first source read; outside a session, guest code runs meanwhile."""
        size = 8 * 1024 * 1024
        source = VirtualSource(size, end=size, hold=True)
        task = asyncio.create_task(running_vm.write_file("res.bin", cast("IO[bytes]", source), make_executable=False))
        try:
            await _until_held(source, task)
            filler = await running_vm.execute(
                "import glob, os, time\n"
                "for _ in range(60000):\n"
                "    tmp = glob.glob('/home/user/.wr.*.tmp')\n"
                f"    if tmp and os.stat(tmp[0]).st_blocks * 512 >= {size}:\n"
                "        break\n"
                "    time.sleep(0.001)\n"
                "else:\n"
                "    raise SystemExit('TIMEOUT: no reservation')\n"
                "fd = os.open('/home/user/.fill', os.O_WRONLY | os.O_CREAT, 0o600)\n"
                "step = 1 << 30\n"
                "while step >= 4096:\n"
                "    try:\n"
                "        os.posix_fallocate(fd, os.fstat(fd).st_size, step)\n"
                "    except OSError:\n"
                "        step //= 2\n"
                "os.close(fd)\n"
                "s = os.statvfs('/home/user')\n"
                "print('reserved', s.f_bavail * s.f_frsize)\n",
                timeout_seconds=execute_timeout,
                env_vars=None,
                on_stdout=None,
                on_stderr=None,
            )
        finally:
            source.release()
            await task
        assert filler.exit_code == 0 and filler.stdout.startswith("reserved"), filler
        assert int(filler.stdout.split()[1]) < 64 * 1024, filler  # the disk is full
        read_back = io.BytesIO()
        await running_vm.read_file("res.bin", destination=read_back)
        assert hashlib.sha256(read_back.getvalue()).digest() == source.digest.digest()

    @pytest.mark.parametrize(
        ("length", "end", "allocated"),
        [(10, 4 * 1024 * 1024, 4096), (1024 * 1024, 4096, 1024 * 1024), (1024 * 1024, None, 1024 * 1024)],
        ids=["reserve-more-than-sent", "size-is-a-hint", "host-sends-no-size"],
    )
    async def test_write_keeps_the_blocks_of_its_bytes(
        self, scheduler: Scheduler, length: int, end: int | None, allocated: int
    ) -> None:
        """The file gets the bytes that come before file_end, whatever size the header gives (a
        source whose end is not where its reads stop); the reserved blocks it did not use are
        freed at the commit. A source that cannot seek sends no size (as a host of an earlier
        release) and still writes."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            source = VirtualSource(length, end=end, hold=False)
            await session.write_file("w.bin", cast("IO[bytes]", source))
            stat = await session.exec(
                "import os; os.sync(); s = os.stat('/home/user/w.bin'); print(s.st_size, s.st_blocks * 512)"
            )
            assert stat.stdout.split() == [str(length), str(allocated)], stat

    async def test_header_size_over_the_limit_is_refused(self, running_vm: QemuVM, execute_timeout: int) -> None:
        """A size over the file limit (a host of another release; this host never sends it)
        is an io_error at the header, and leaves no temporary file."""
        await running_vm.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
        # model_construct: the host model refuses this size; request_line sends it as it is
        request = WriteFileRequest.model_construct(path="big.bin", make_executable=False, size=_FILE_SIZE_LIMIT + 1)
        reply = await running_vm.channel.send_request(request, timeout=30)
        assert isinstance(reply, StreamingErrorMessage), reply
        assert (reply.error_type, reply.message) == (
            "io_error",
            f"File 'big.bin' is {_FILE_SIZE_LIMIT + 1} bytes, exceeds {_FILE_SIZE_LIMIT}",
        )
        names = await running_vm.execute(
            "import os; print([n for n in os.listdir('/home/user') if n.startswith('.wr.')])",
            timeout_seconds=execute_timeout,
            env_vars=None,
            on_stdout=None,
            on_stderr=None,
        )
        assert names.stdout == "[]\n", names

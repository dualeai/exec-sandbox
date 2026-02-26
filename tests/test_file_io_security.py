"""Security tests for file I/O operations.

Tests path traversal attacks, symlink escape, and null byte injection.
Requires VM (integration tests).
"""

from pathlib import Path

import pytest

from exec_sandbox.exceptions import VmPermanentError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# Real path traversal vectors — these use actual ".." or absolute paths
# that the path validator rejects at the component level.
FILESYSTEM_TRAVERSAL_VECTORS = [
    "../etc/passwd",
    "../../etc/shadow",
    "../../../root/.ssh/id_rsa",
    "subdir/../../etc/passwd",
    "subdir/../subdir/../../etc/passwd",
    "/etc/passwd",
    "/home/user/../../../etc/passwd",
]

# Web-encoding vectors (from fuzzdb) — these use URL encoding, backslash
# substitution, or dot-stuffing to bypass web-layer path filters. On a Unix
# filesystem they are treated as LITERAL filenames (no server-side decode),
# so write_file succeeds harmlessly but read_file fails (file not found).
WEB_ENCODING_VECTORS = [
    "....//....//etc/passwd",  # 4 dots = normal filename, not ".."
    "..%2f..%2fetc/passwd",  # "%2f" is literal, not "/"
    "..\\..\\etc\\passwd",  # backslash is valid filename char on Unix
]


class TestPathTraversalWrite:
    """Path traversal attacks on write_file."""

    @pytest.mark.parametrize("path", FILESYSTEM_TRAVERSAL_VECTORS)
    async def test_write_traversal_rejected(self, dual_scheduler: Scheduler, path: str) -> None:
        """write_file rejects filesystem-level path traversal attempts."""
        async with await dual_scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises((VmPermanentError, ValueError)):
                await session.write_file(path, b"pwned")

    @pytest.mark.parametrize("path", WEB_ENCODING_VECTORS)
    async def test_write_web_encoding_harmless(self, dual_scheduler: Scheduler, path: str) -> None:
        """Web-encoding tricks create harmless literal filenames (no traversal)."""
        async with await dual_scheduler.session(language=Language.PYTHON) as session:
            # These succeed because %2f, \\, and .... are literal on Unix
            await session.write_file(path, b"data")


class TestPathTraversalRead:
    """Path traversal attacks on read_file."""

    @pytest.mark.parametrize("path", FILESYSTEM_TRAVERSAL_VECTORS)
    async def test_read_traversal_rejected(self, dual_scheduler: Scheduler, tmp_path: Path, path: str) -> None:
        """read_file rejects filesystem-level path traversal attempts."""
        async with await dual_scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises(VmPermanentError):
                await session.read_file(path, destination=tmp_path / "out.bin")

    @pytest.mark.parametrize("path", WEB_ENCODING_VECTORS)
    async def test_read_web_encoding_not_found(self, dual_scheduler: Scheduler, tmp_path: Path, path: str) -> None:
        """Web-encoding tricks fail on read because literal filename doesn't exist."""
        async with await dual_scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises(VmPermanentError):
                await session.read_file(path, destination=tmp_path / "out.bin")


class TestPathTraversalList:
    """Path traversal attacks on list_files."""

    @pytest.mark.parametrize(
        "path",
        [
            "../etc",
            "../../",
            "/etc",
            "subdir/../../etc",
        ],
    )
    async def test_list_traversal_rejected(self, dual_scheduler: Scheduler, path: str) -> None:
        """list_files rejects path traversal attempts."""
        async with await dual_scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises(VmPermanentError):
                await session.list_files(path)


class TestSymlinkEscape:
    """Symlink escape attacks on read_file."""

    async def test_symlink_to_etc_passwd(self, dual_scheduler: Scheduler, tmp_path: Path) -> None:
        """read_file follows symlink but canonicalize catches escape."""
        async with await dual_scheduler.session(language=Language.PYTHON) as session:
            # Create a symlink pointing outside sandbox
            result = await session.exec("import os; os.symlink('/etc/passwd', '/home/user/evil_link')")
            assert result.exit_code == 0

            # read_file should reject the symlink target
            with pytest.raises(VmPermanentError):
                await session.read_file("evil_link", destination=tmp_path / "evil.bin")

    async def test_multi_hop_symlink_escape(self, dual_scheduler: Scheduler, tmp_path: Path) -> None:
        """Multi-hop symlink chain that escapes sandbox is rejected."""
        async with await dual_scheduler.session(language=Language.PYTHON) as session:
            # Create chain: link1 -> link2 -> /etc/passwd
            await session.exec("import os; os.symlink('/etc', '/home/user/link_to_etc')")
            await session.exec("import os; os.symlink('/home/user/link_to_etc/passwd', '/home/user/chain_link')")

            with pytest.raises(VmPermanentError):
                await session.read_file("chain_link", destination=tmp_path / "chain.bin")


class TestNullByteInjection:
    """Null byte injection in file paths."""

    async def test_null_byte_in_path_middle(self, dual_scheduler: Scheduler) -> None:
        """Null byte in middle of path is rejected."""
        async with await dual_scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises((VmPermanentError, ValueError)):
                await session.write_file("file\x00.txt", b"data")

    async def test_null_byte_in_path_end(self, dual_scheduler: Scheduler) -> None:
        """Null byte at end of path is rejected."""
        async with await dual_scheduler.session(language=Language.PYTHON) as session:
            with pytest.raises((VmPermanentError, ValueError)):
                await session.write_file("file.txt\x00", b"data")

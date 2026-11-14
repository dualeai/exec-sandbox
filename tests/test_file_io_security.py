"""Security tests for file I/O operations.

Tests path traversal attacks, symlink escape, and null byte injection.
Requires VM (integration tests).
"""

from pathlib import Path

import pytest

from exec_sandbox.exceptions import ErrorCode, InvalidInputError
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec

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


@pytest.mark.slow
class TestPathTraversalWrite:
    """Path traversal attacks on write_file.

    Slow under TCG: guest agent path validation round-trip can exceed
    FILE_IO_TIMEOUT_SECONDS under CI CPU contention.
    """

    @pytest.mark.parametrize("path", FILESYSTEM_TRAVERSAL_VECTORS)
    async def test_write_traversal_rejected(self, dual_scheduler: Scheduler, path: str) -> None:
        """write_file rejects filesystem-level path traversal attempts."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                await session.write_file(path, b"pwned")

    @pytest.mark.parametrize("path", WEB_ENCODING_VECTORS)
    async def test_write_web_encoding_harmless(self, dual_scheduler: Scheduler, path: str) -> None:
        """Web-encoding tricks create harmless literal filenames (no traversal)."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # These succeed because %2f, \\, and .... are literal on Unix
            await session.write_file(path, b"data")


class TestPathTraversalRead:
    """Path traversal attacks on read_file."""

    @pytest.mark.parametrize("path", FILESYSTEM_TRAVERSAL_VECTORS)
    async def test_read_traversal_rejected(self, dual_scheduler: Scheduler, tmp_path: Path, path: str) -> None:
        """read_file rejects filesystem-level path traversal attempts."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                await session.read_file(path, destination=tmp_path / "out.bin")

    @pytest.mark.parametrize("path", WEB_ENCODING_VECTORS)
    async def test_read_web_encoding_not_found(self, dual_scheduler: Scheduler, tmp_path: Path, path: str) -> None:
        """Web-encoding tricks fail on read because literal filename doesn't exist."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.GUEST_IO):
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
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                await session.list_files(path)


class TestSymlinkEscape:
    """Symlink escape attacks on read_file, write_file and list_files."""

    async def test_symlink_to_etc_passwd(self, dual_scheduler: Scheduler, tmp_path: Path) -> None:
        """A link to a file out of /home/user: read_file is refused (a path error)."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Create a symlink pointing outside sandbox
            result = await session.exec("import os; os.symlink('/etc/passwd', '/home/user/evil_link')")
            assert result.exit_code == 0

            # read_file should reject the symlink target
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                await session.read_file("evil_link", destination=tmp_path / "evil.bin")

    async def test_multi_hop_symlink_escape(self, dual_scheduler: Scheduler, tmp_path: Path) -> None:
        """Multi-hop symlink chain that escapes sandbox is rejected."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            # Create chain: link1 -> link2 -> /etc/passwd
            await session.exec("import os; os.symlink('/etc', '/home/user/link_to_etc')")
            await session.exec("import os; os.symlink('/home/user/link_to_etc/passwd', '/home/user/chain_link')")

            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                await session.read_file("chain_link", destination=tmp_path / "chain.bin")

    async def test_folder_links_out_of_home_refuse_write_read_and_list(
        self, dual_scheduler: Scheduler, tmp_path: Path
    ) -> None:
        """Guest code plants folder links out of /home/user (absolute, and relative with ".."):
        the agent writes as root, so a write through one must not make a file outside; a
        read and a listing through one are refused the same way (path error)."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec(
                "import os; os.symlink('/tmp', '/home/user/out'); os.symlink('../../tmp', '/home/user/rel_out');"
                " open('/tmp/secret', 'w').write('host must not read this')"
            )
            assert result.exit_code == 0, result.stderr

            for link in ("out", "rel_out"):
                with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                    await session.write_file(f"{link}/pwned", b"pwned")
                with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                    await session.write_file(f"{link}/new_dir/pwned", b"pwned")
                with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                    await session.read_file(f"{link}/secret", destination=tmp_path / f"{link}.bin")
                with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                    await session.list_files(link)

            result = await session.exec(
                "import os; print(sorted(n for n in os.listdir('/tmp') if n.startswith(('pwned', 'new_dir'))))"
            )
            assert result.stdout.strip() == "[]"

    async def test_a_link_that_stays_in_home_still_works(self, dual_scheduler: Scheduler, tmp_path: Path) -> None:
        """Folder links that resolve inside /home/user, relative or absolute (`ln -s
        "$PWD/run" latest` makes absolute ones), are followed for write, read and list: the
        check refuses only what leaves the root."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec(
                "import os; os.mkdir('/home/user/real');"
                " os.symlink('real', '/home/user/alias'); os.symlink('/home/user/real', '/home/user/abs_alias')"
            )
            assert result.exit_code == 0, result.stderr

            for link in ("alias", "abs_alias"):
                await session.write_file(f"{link}/{link}.txt", link.encode())
                await session.read_file(f"{link}/{link}.txt", destination=tmp_path / f"{link}.txt")
            listed = await session.list_files("abs_alias")

        for link in ("alias", "abs_alias"):
            assert (tmp_path / f"{link}.txt").read_bytes() == link.encode()
        assert [(entry.name, entry.is_dir) for entry in listed] == [("abs_alias.txt", False), ("alias.txt", False)]


class TestNullByteInjection:
    """Null byte injection in file paths."""

    async def test_null_byte_in_path_middle(self, dual_scheduler: Scheduler) -> None:
        """Null byte in middle of path is rejected."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                await session.write_file("file\x00.txt", b"data")

    async def test_null_byte_in_path_end(self, dual_scheduler: Scheduler) -> None:
        """Null byte at end of path is rejected."""
        async with await dual_scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PATH_INVALID):
                await session.write_file("file.txt\x00", b"data")

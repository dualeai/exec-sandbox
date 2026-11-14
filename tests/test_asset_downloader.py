"""Tests for asset_downloader module."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import stat
import subprocess
import sys
import threading
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from typing import IO, TYPE_CHECKING
from unittest.mock import patch

import aiohttp
import aioresponses.core as aioresponses_core
import pytest
from aioresponses import aioresponses
from yarl import URL

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

if sys.version_info >= (3, 14):
    from compression import zstd
    from compression.zstd import ZstdError
else:
    from backports import zstd
    from backports.zstd import ZstdError

from exec_sandbox import asset_downloader
from exec_sandbox.asset_downloader import (
    AsyncPooch,
    decompress_zstd,
    get_current_arch,
    get_gvproxy_suffix,
    retrieve,
)
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError
from exec_sandbox.files.disk import create_file
from exec_sandbox.files.locks import CacheEntry
from exec_sandbox.hash_utils import file_hash
from tests.conftest import asyncio_with, skip_unless_aarch64, skip_unless_x86_64


class _CompatClientResponse(aiohttp.ClientResponse):
    """aioresponses 0.7.9 builds ClientResponse without the ``stream_writer``
    argument that aiohttp >= 3.14 requires (it reads ``stream_writer.output_size``
    when ``writer`` is None). Supply a writer whose output size is 0.
    """

    def __init__(self, method, url, **kwargs):  # type: ignore[no-untyped-def]  # mirrors aiohttp's signature
        kwargs.setdefault("stream_writer", SimpleNamespace(output_size=0))
        super().__init__(method, url, **kwargs)


aioresponses_core.ClientResponse = _CompatClientResponse  # type: ignore[misc]  # default response_class


@pytest.fixture
def retry_waits() -> Iterator[list[float]]:
    """The waits between download attempts, recorded and not slept (1 s, then 2 s): a
    download that fails each time ends in milliseconds."""
    waits: list[float] = []

    async def sleep(delay: float) -> None:
        waits.append(delay)
        await asyncio.sleep(0)

    with patch("exec_sandbox.asset_downloader.asyncio", asyncio_with(sleep=sleep)):
        yield waits


def attempts(m: aioresponses, url: str) -> int:
    """How many GET requests went to `url`."""
    return len(m.requests.get(("GET", URL(url)), []))


@contextlib.asynccontextmanager
async def _stalled_server(*, first_bytes: bytes):
    """A local HTTP server that reads each request, sends `first_bytes`, then stalls (it
    never closes the connection). Yields its base URL."""
    stall = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(first_bytes)
        await writer.drain()
        await stall.wait()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        stall.set()  # each handler then closes its connection
        server.close()
        await server.wait_closed()


class TestStalledServer:
    """A server that stalls ends a request at its timeout, not after aiohttp's defaults
    (30 s to connect, 300 s in all)."""

    async def test_a_stalled_download_ends_at_the_read_timeout(self, tmp_path: Path) -> None:
        """Headers and 10 of 1000 bytes, then nothing: each attempt ends after the read
        timeout, and the download fails with ASSET_DOWNLOAD."""
        head = b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n" + b"x" * 10
        with (
            patch.object(asset_downloader, "_DOWNLOAD_READ_TIMEOUT_SECONDS", 0.2),
            patch.object(asset_downloader, "RETRY_DELAY", 0.01),
        ):
            async with _stalled_server(first_bytes=head) as base:
                with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.ASSET_DOWNLOAD):
                    await asyncio.wait_for(
                        retrieve(f"{base}/asset.bin", f"sha256:{'0' * 64}", path=tmp_path, decompress=False),
                        timeout=5,
                    )

    async def test_a_metadata_request_with_no_answer_ends_at_its_timeout(self, tmp_path: Path) -> None:
        """The release metadata request gets no answer: it fails with ASSET_DOWNLOAD at its
        total timeout."""
        pooch = AsyncPooch(path=tmp_path, base_url="http://127.0.0.1/unused/v{version}", version="1.0.0")
        with patch.object(asset_downloader, "_METADATA_TIMEOUT_SECONDS", 0.2):
            async with _stalled_server(first_bytes=b"") as base:
                with (
                    patch.object(asset_downloader, "_GITHUB_API", base),
                    pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.ASSET_DOWNLOAD),
                ):
                    await asyncio.wait_for(pooch.load_registry_from_github("owner", "repo", "v1.0.0"), timeout=5)


class TestGetCurrentArch:
    """get_current_arch: this host's architecture in the kernel's names."""

    @skip_unless_x86_64
    def test_x86_64_host(self):
        assert get_current_arch() == "x86_64"

    @skip_unless_aarch64
    def test_aarch64_host(self):
        """ARM64 hosts are "aarch64", also on macOS (where the machine name is "arm64")."""
        assert get_current_arch() == "aarch64"


class TestGetGvproxySuffix:
    """Tests for get_gvproxy_suffix function."""

    @pytest.mark.parametrize(
        ("os_name", "arch", "expected"),
        [
            ("darwin", "aarch64", "darwin-arm64"),
            ("darwin", "x86_64", "darwin-amd64"),
            ("linux", "aarch64", "linux-arm64"),
            ("linux", "x86_64", "linux-amd64"),
        ],
    )
    def test_returns_correct_suffix(self, os_name: str, arch: str, expected: str):
        """The os-arch suffix of the asked arch (Go names), not of this host's arch."""
        with patch("exec_sandbox.asset_downloader.get_os_name", return_value=os_name):
            assert get_gvproxy_suffix(arch) == expected


class TestImportCost:
    def test_package_import_loads_no_aiohttp(self) -> None:
        """aiohttp (~50 ms to import) loads at the first download, not at `import exec_sandbox`."""
        code = "import sys, exec_sandbox\nassert 'aiohttp' not in sys.modules, 'aiohttp imported'\n"
        subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


class TestRetrieve:
    """Tests for retrieve function."""

    async def test_downloads_and_caches(self, tmp_path: Path):
        """Should download file and cache it locally."""
        content = b"hello world"
        content_hash = hashlib.sha256(content).hexdigest()

        with aioresponses() as m:
            m.get(
                "https://example.com/test.txt",
                body=content,
            )

            path = await retrieve(
                url="https://example.com/test.txt",
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
                decompress=False,
            )

            assert path.exists()
            assert path.read_bytes() == content

    async def test_cache_folder_is_private_whatever_the_umask(self, tmp_path: Path):
        """Made 0700 under umask 002 (the Ubuntu default) and any group (sg qemu-vm): the user
        cache folder also holds the home disk folder, so only this user may write it."""
        content = b"hello world"
        cache = tmp_path / "cache" / "exec-sandbox"
        old = os.umask(0o002)
        try:
            with aioresponses() as m:
                m.get("https://example.com/test.txt", body=content)
                await retrieve(
                    url="https://example.com/test.txt",
                    known_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
                    path=cache,
                    decompress=False,
                )
        finally:
            os.umask(old)
        assert [stat.S_IMODE(p.stat().st_mode) & 0o777 for p in (cache.parent, cache)] == [0o700, 0o700]

    async def test_assets_have_no_group_write_whatever_the_umask(self, tmp_path: Path):
        """Under `sg qemu-vm` and umask 002, open() makes files 0664 of group qemu-vm: a QEMU
        that runs as qemu-vm could change a kernel or a base image that every later VM boots.
        A downloaded asset is 0644 (QEMU as qemu-vm reads it), a decompressed one too, and
        their lock files are 0600."""
        raw, compressed = b"gvproxy bytes", zstd.compress(b"kernel bytes")
        old = os.umask(0o002)
        try:
            with aioresponses() as m:
                m.get("https://example.com/gvproxy", body=raw)
                m.get("https://example.com/vmlinuz.zst", body=compressed)
                await retrieve(
                    url="https://example.com/gvproxy",
                    known_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
                    path=tmp_path,
                    decompress=False,
                )
                await retrieve(
                    url="https://example.com/vmlinuz.zst",
                    known_hash=f"sha256:{hashlib.sha256(compressed).hexdigest()}",
                    path=tmp_path,
                    decompress=True,
                )
        finally:
            os.umask(old)
        assert {p.name: stat.S_IMODE(p.stat().st_mode) for p in tmp_path.iterdir()} == {
            "gvproxy": 0o644,
            "gvproxy.lock": 0o600,
            "vmlinuz": 0o644,
            "vmlinuz.lock": 0o600,
        }
        assert (tmp_path / "vmlinuz").read_bytes() == b"kernel bytes"

    async def test_stale_temp_files_are_replaced(self, tmp_path: Path):
        """A killed process leaves its temp files (here 0666): the next download (gvproxy.tmp)
        and unpack (vmlinuz.tmp) make new ones, never write into the old ones. The assets get
        mode 0644, not the 0666 of the old files, and no temp file stays."""
        raw, compressed = b"gvproxy bytes", zstd.compress(b"kernel bytes")
        for stale in ("gvproxy.tmp", "vmlinuz.tmp"):
            (tmp_path / stale).write_bytes(b"old")
            (tmp_path / stale).chmod(0o666)
        with aioresponses() as m:
            m.get("https://example.com/gvproxy", body=raw)
            m.get("https://example.com/vmlinuz.zst", body=compressed)
            gvproxy = await retrieve(
                url="https://example.com/gvproxy",
                known_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
                path=tmp_path,
                decompress=False,
            )
            kernel = await retrieve(
                url="https://example.com/vmlinuz.zst",
                known_hash=f"sha256:{hashlib.sha256(compressed).hexdigest()}",
                path=tmp_path,
                decompress=True,
            )
        assert (gvproxy.read_bytes(), kernel.read_bytes()) == (raw, b"kernel bytes")
        assert {p.name: stat.S_IMODE(p.stat().st_mode) for p in tmp_path.iterdir()} == {
            "gvproxy": 0o644,
            "gvproxy.lock": 0o600,
            "vmlinuz": 0o644,
            "vmlinuz.lock": 0o600,
        }

    async def test_download_hashes_writes_and_closes_off_the_event_loop(self, tmp_path: Path):
        """Each downloaded chunk is hashed and written in a worker thread (one job per
        chunk): a page-cache write can wait for writeback, never on the event loop thread.
        The temp file also closes in a worker thread: on NFS, close writes back every
        dirty page of the image just downloaded."""
        content = os.urandom(64 * 1024 + 4096)  # more than one 64 KiB block
        known_hash = f"sha256:{hashlib.sha256(content).hexdigest()}"
        loop_thread = threading.get_ident()
        threads: dict[str, list[int]] = {"hash": [], "write": [], "close": []}
        real_hash_and_write = asset_downloader._hash_and_write

        def hash_and_write(chunk: bytes, *, update: Callable[[bytes], object], sink: IO[bytes]) -> None:
            def spied_update(data: bytes) -> object:
                threads["hash"].append(threading.get_ident())
                return update(data)

            real_hash_and_write(chunk, update=spied_update, sink=sink)

        def spied_create(path: Path, *, mode: int, dir_fd: int | None) -> IO[bytes]:
            stream = create_file(path, mode=mode, dir_fd=dir_fd)
            real_write, real_close = stream.write, stream.close

            def write(data: bytes) -> int:
                threads["write"].append(threading.get_ident())
                return real_write(data)

            def close() -> None:
                threads["close"].append(threading.get_ident())
                real_close()

            stream.write = write  # type: ignore[method-assign]
            stream.close = close  # type: ignore[method-assign]
            return stream

        with (
            aioresponses() as m,
            patch.object(asset_downloader, "_hash_and_write", hash_and_write),
            patch("exec_sandbox.asset_downloader.create_file", spied_create),
        ):
            m.get("https://example.com/big.bin", body=content)
            path = await retrieve(
                url="https://example.com/big.bin", known_hash=known_hash, path=tmp_path, decompress=False
            )

        assert path.read_bytes() == content
        for step, step_threads in threads.items():
            assert step_threads, step
            assert loop_thread not in step_threads, step

    async def test_uses_cache_on_second_call(self, tmp_path: Path):
        """Should use cached file on second call without re-downloading."""
        content = b"cached content"
        content_hash = hashlib.sha256(content).hexdigest()

        with aioresponses() as m:
            m.get("https://example.com/cached.txt", body=content)

            # First call - downloads
            path1 = await retrieve(
                url="https://example.com/cached.txt",
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
                decompress=False,
            )

            # Second call - should use cache (no mock needed)
            path2 = await retrieve(
                url="https://example.com/cached.txt",
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
                decompress=False,
            )

            assert path1 == path2
            # aioresponses should only have been called once
            m.assert_called_once()

    async def test_checksum_verification_fails(self, tmp_path: Path, retry_waits: list[float]):
        """A digest that never matches: three attempts, then TransientError ASSET_DOWNLOAD,
        and no file stays."""
        url = "https://example.com/bad.txt"
        with aioresponses() as m:
            for _ in range(3):
                m.get(url, body=b"actual content")

            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.ASSET_DOWNLOAD):
                await retrieve(url=url, known_hash="sha256:wronghash", path=tmp_path, decompress=False)
            assert attempts(m, url) == 3
        assert retry_waits == [1.0, 2.0]
        assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.txt.lock"]

    async def test_retries_on_failure(self, tmp_path: Path, retry_waits: list[float]):
        """A failed attempt (no connection, then a server error) is tried again after a wait
        that doubles (1 s, 2 s): the third attempt gets the file."""
        content = b"third time"
        url = "https://example.com/flaky.txt"
        with aioresponses() as m:
            m.get(url, exception=aiohttp.ClientConnectionError("Connection refused"))
            m.get(url, status=503)
            m.get(url, body=content)

            path = await retrieve(
                url=url, known_hash=f"sha256:{hashlib.sha256(content).hexdigest()}", path=tmp_path, decompress=False
            )
            assert attempts(m, url) == 3
        assert path.read_bytes() == content
        assert retry_waits == [1.0, 2.0]

    @pytest.mark.parametrize("status_code", [404, 500])
    async def test_http_error(self, tmp_path: Path, status_code: int, retry_waits: list[float]):
        """An HTTP error status at each of the three attempts: TransientError ASSET_DOWNLOAD."""
        url = f"https://example.com/error-{status_code}.txt"
        with aioresponses() as m:
            for _ in range(3):
                m.get(url, status=status_code)

            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.ASSET_DOWNLOAD):
                await retrieve(url=url, known_hash="sha256:abc123", path=tmp_path, decompress=False)
            assert attempts(m, url) == 3
        assert retry_waits == [1.0, 2.0]

    async def test_connection_error(self, tmp_path: Path, retry_waits: list[float]):
        """No connection at each of the three attempts: TransientError ASSET_DOWNLOAD."""
        url = "https://example.com/timeout.txt"
        with aioresponses() as m:
            for _ in range(3):
                m.get(url, exception=aiohttp.ClientConnectionError("Connection refused"))

            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.ASSET_DOWNLOAD):
                await retrieve(url=url, known_hash="sha256:abc123", path=tmp_path, decompress=False)
            assert attempts(m, url) == 3
        assert retry_waits == [1.0, 2.0]

    async def test_empty_hash_skips_verification(self, tmp_path: Path):
        """Should skip hash verification when hash is empty."""
        content = b"any content"

        with aioresponses() as m:
            m.get("https://example.com/nohash.txt", body=content)

            # Empty hash should succeed without verification
            path = await retrieve(
                url="https://example.com/nohash.txt",
                known_hash="",
                path=tmp_path,
                decompress=False,
            )

            assert path.exists()
            assert path.read_bytes() == content

    async def test_redownloads_on_hash_mismatch(self, tmp_path: Path):
        """Should re-download when cached file has wrong hash."""
        content = b"correct content"
        content_hash = hashlib.sha256(content).hexdigest()

        # Pre-populate cache with wrong content
        cached_file = tmp_path / "cached.txt"
        cached_file.write_bytes(b"wrong content in cache")

        with aioresponses() as m:
            m.get("https://example.com/cached.txt", body=content)

            path = await retrieve(
                url="https://example.com/cached.txt",
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
                decompress=False,
            )

            # Should have re-downloaded and now have correct content
            assert path == cached_file
            assert path.read_bytes() == content

    async def test_streaming_memory_usage(self, tmp_path: Path):
        """The digest check of a cached file reads it in blocks: the peak memory of a
        retrieve that finds a 32 MiB file in the cache stays far below the file size."""
        # Use a 32MB file so the streaming-vs-full-read gap is large enough
        # to tolerate GC-delay overhead on free-threaded Python (3.14t).
        file_size = 32 * 1024 * 1024  # 32MB
        large_file = tmp_path / "large.bin"

        # Write file in chunks to avoid memory spike during setup
        chunk = b"x" * (64 * 1024)  # 64KB chunks
        digest = hashlib.sha256()
        with large_file.open("wb") as f:
            for _ in range(file_size // len(chunk)):
                f.write(chunk)
                digest.update(chunk)
        known_hash = f"sha256:{digest.hexdigest()}"
        url = "https://example.com/large.bin"

        with aioresponses() as m:  # no route: a download would fail
            # Warmup: absorb one-time allocations from tracemalloc internals,
            # worker thread startup (asyncio.to_thread), and hashlib module init
            # so they don't inflate the peak measurement.
            tracemalloc.start()
            await retrieve(url=url, known_hash=known_hash, path=tmp_path, decompress=False)
            tracemalloc.stop()

            # Measure peak memory during hash verification
            tracemalloc.start()
            result = await retrieve(url=url, known_hash=known_hash, path=tmp_path, decompress=False)
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert not m.requests  # the digest matched: the cached file, no download

        assert result == large_file

        # Broken streaming (entire file read at once): peak = file_size
        # (hashlib.update processes in-place, no copy).
        # Working streaming: peak = one read buffer (hashlib.file_digest
        # reuses a 256 KiB buffer in its worker thread). Half the file size
        # leaves room for deferred frees on free-threaded Python (3.14t).
        max_allowed = file_size // 2
        assert peak_memory < max_allowed, (
            f"Peak memory {peak_memory / 1024:.1f}KB exceeded {max_allowed / 1024:.1f}KB limit "
            f"for {file_size / 1024 / 1024:.0f}MB file. Streaming may be broken."
        )


class TestAsyncPooch:
    """Tests for AsyncPooch class."""

    async def test_fetch_downloads_file(self, tmp_path: Path):
        """A registry file downloads from base_url with the configured version, into path."""
        content = b"registry content"
        content_hash = hashlib.sha256(content).hexdigest()

        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://example.com/v{version}",
            version="1.0.0",
            registry={"file.txt": f"sha256:{content_hash}"},
        )

        with aioresponses() as m:
            m.get("https://example.com/v1.0.0/file.txt", body=content)

            path = await pooch.fetch("file.txt", decompress=False)

            assert path == tmp_path / "file.txt"
            assert path.read_bytes() == content

    async def test_fetch_not_in_registry(self, tmp_path: Path):
        """Should raise ConfigError ASSET_NOT_FOUND for unknown file."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://example.com",
            version="1.0.0",
            registry={},
        )

        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND):
            await pooch.fetch("unknown.txt", decompress=False)

    async def test_load_registry_from_github(self, tmp_path: Path):
        """Should load registry from GitHub release API."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0",
        )

        github_response = {
            "tag_name": "v1.0.0",
            "assets": [
                {"name": "file1.txt", "digest": "sha256:hash1"},
                {"name": "file2.txt", "digest": "sha256:hash2"},
            ],
        }

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/tags/v1.0.0",
                payload=github_response,
            )

            await pooch.load_registry_from_github("owner", "repo", "v1.0.0")

            assert pooch.registry["file1.txt"] == "sha256:hash1"
            assert pooch.registry["file2.txt"] == "sha256:hash2"

    async def test_github_api_404_error(self, tmp_path: Path):
        """Should raise ConfigError ASSET_NOT_FOUND when GitHub release not found."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0",
        )

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/tags/v1.0.0",
                status=404,
            )

            with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.ASSET_NOT_FOUND):
                await pooch.load_registry_from_github("owner", "repo", "v1.0.0")

    @pytest.mark.parametrize("token", ["test-token", None], ids=["token", "no-token"])
    async def test_github_token_authenticates_the_metadata_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: str | None
    ):
        """GITHUB_TOKEN, when set, goes in the Authorization header of the release request (a
        higher rate limit); without it the request is anonymous."""
        if token is None:
            monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        else:
            monkeypatch.setenv("GITHUB_TOKEN", token)
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0",
        )

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/tags/v1.0.0",
                payload={"tag_name": "v1.0.0", "assets": []},
            )
            await pooch.load_registry_from_github("owner", "repo", "v1.0.0")
            ((call,),) = m.requests.values()

        expected = {"Accept": "application/vnd.github+json"}
        if token is not None:
            expected["Authorization"] = "Bearer test-token"
        assert call.kwargs["headers"] == expected

    @pytest.mark.parametrize(
        "answer",
        [
            pytest.param({"status": 403}, id="rate-limited"),
            pytest.param({"status": 502}, id="server-error"),
            pytest.param({"exception": aiohttp.ClientConnectionError("no route")}, id="no-network"),
            pytest.param({"status": 200, "body": "not json", "content_type": "application/json"}, id="not-json"),
        ],
    )
    async def test_unreadable_release_metadata_is_retry(self, tmp_path: Path, answer: dict[str, object]):
        """Metadata that cannot be read now (no network, a rate limit, a server error, a bad
        body) is a TransientError ASSET_DOWNLOAD: the same call can succeed later."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0",
        )

        with aioresponses() as m:
            m.get("https://api.github.com/repos/owner/repo/releases/tags/v1.0.0", **answer)  # type: ignore[arg-type]

            with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.ASSET_DOWNLOAD) as exc:
                await pooch.load_registry_from_github("owner", "repo", "v1.0.0")
        assert exc.value.context == {"owner": "owner", "repo": "repo", "tag": "v1.0.0"}

    async def test_github_assets_without_digest(self, tmp_path: Path):
        """Should handle assets without digest (older releases)."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0",
        )

        github_response = {
            "tag_name": "v1.0.0",
            "assets": [
                {"name": "file1.txt", "digest": "sha256:hash1"},
                {"name": "file2.txt"},  # No digest
            ],
        }

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/tags/v1.0.0",
                payload=github_response,
            )

            await pooch.load_registry_from_github("owner", "repo", "v1.0.0")

            assert pooch.registry["file1.txt"] == "sha256:hash1"
            assert pooch.registry["file2.txt"] == ""  # Empty hash for missing digest

    async def test_github_latest_tag(self, tmp_path: Path):
        """Should use /releases/latest endpoint for 'latest' tag."""
        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0",
        )

        github_response = {
            "tag_name": "v2.0.0",
            "assets": [{"name": "file.txt", "digest": "sha256:hash"}],
        }

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/latest",
                payload=github_response,
            )

            await pooch.load_registry_from_github("owner", "repo", "latest")

            assert pooch.registry["file.txt"] == "sha256:hash"

    async def test_fetch_uses_resolved_version_from_github(self, tmp_path: Path):
        """Should use actual tag from GitHub API instead of configured version."""
        content = b"resolved version content"
        content_hash = hashlib.sha256(content).hexdigest()

        pooch = AsyncPooch(
            path=tmp_path,
            base_url="https://github.com/owner/repo/releases/download/v{version}",
            version="1.0.0.dev0",  # the URL had v1.0.0.dev0 without the GitHub tag
            registry={},
        )

        github_response = {
            "tag_name": "v2.0.0",  # Actual tag from GitHub
            "assets": [
                {"name": "file.txt", "digest": f"sha256:{content_hash}"},
            ],
        }

        with aioresponses() as m:
            m.get(
                "https://api.github.com/repos/owner/repo/releases/latest",
                payload=github_response,
            )
            # Should use "v2.0.0" (resolved) instead of "vlatest"
            m.get(
                "https://github.com/owner/repo/releases/download/v2.0.0/file.txt",
                body=content,
            )

            await pooch.load_registry_from_github("owner", "repo", "latest")
            path = await pooch.fetch("file.txt", decompress=False)

            assert path.read_bytes() == content


class TestDecompressZstd:
    """Tests for decompress_zstd function."""

    async def test_decompresses_file(self, tmp_path: Path):
        """Should decompress .zst file and remove original."""
        # Create compressed file
        original_content = b"Hello, World! " * 100
        compressed_data = zstd.compress(original_content)

        compressed_file = tmp_path / "test.txt.zst"
        compressed_file.write_bytes(compressed_data)

        # Decompress
        result_path = await decompress_zstd(compressed_file)

        # Verify
        assert result_path == tmp_path / "test.txt"
        assert result_path.exists()
        assert result_path.read_bytes() == original_content
        assert not compressed_file.exists()  # Original should be deleted

    async def test_corrupted_zstd_file(self, tmp_path: Path):
        """Data that is not zstd: ZstdError, the temp file is removed and the source stays."""
        # Create corrupted file (not valid zstd data)
        corrupted_file = tmp_path / "corrupted.txt.zst"
        corrupted_file.write_bytes(b"this is not valid zstd data")

        with pytest.raises(ZstdError):
            await decompress_zstd(corrupted_file)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["corrupted.txt.zst"]

    async def test_missing_source_raises_file_not_found(self, tmp_path: Path):
        """No source file: FileNotFoundError, and no temp file stays."""
        with pytest.raises(FileNotFoundError):
            await decompress_zstd(tmp_path / "nonexistent.zst")
        assert list(tmp_path.iterdir()) == []

    async def test_empty_zstd_file(self, tmp_path: Path):
        """Should handle empty compressed file."""
        # Create compressed empty file
        compressed_data = zstd.compress(b"")

        compressed_file = tmp_path / "empty.txt.zst"
        compressed_file.write_bytes(compressed_data)

        # Decompress
        result_path = await decompress_zstd(compressed_file)

        # Verify
        assert result_path.exists()
        assert result_path.read_bytes() == b""


# ============================================================================
# CacheEntry.saving unit tests
# ============================================================================


class TestFileLock:
    """Tests for CacheEntry.saving and try_saving."""

    async def test_yields_true_when_target_missing(self, tmp_path: Path):
        """Should yield True when target file does not exist."""
        target = tmp_path / "missing.txt"
        async with CacheEntry(target).saving() as should_proceed:
            assert should_proceed is True

    async def test_yields_false_when_target_exists(self, tmp_path: Path):
        """Should yield False when target file already exists with content."""
        target = tmp_path / "existing.txt"
        target.write_text("content")
        async with CacheEntry(target).saving() as should_proceed:
            assert should_proceed is False

    async def test_try_lock_raises_when_held(self, tmp_path: Path):
        """try_saving raises BlockingIOError at once while saving holds the lock."""
        target = tmp_path / "locked.txt"
        async with CacheEntry(target).saving():
            with pytest.raises(BlockingIOError), CacheEntry(target).try_saving():
                pass  # Should not reach here

    @pytest.mark.parametrize("wait", [True, False], ids=["saving", "try_saving"])
    async def test_lock_file_is_not_truncated(self, tmp_path: Path, wait: bool):
        """Taking the lock opens the existing lock file without O_TRUNC (open("w") costs
        about 0.4 ms on APFS): bytes already in it stay, the lock is the same."""
        target = tmp_path / "kept.txt"
        lock_file = tmp_path / "kept.txt.lock"
        lock_file.write_bytes(b"kept")

        async with contextlib.AsyncExitStack() as stack:
            if wait:
                should_proceed = await stack.enter_async_context(CacheEntry(target).saving())
            else:
                should_proceed = stack.enter_context(CacheEntry(target).try_saving())
            assert should_proceed is True
            assert lock_file.read_bytes() == b"kept"
            with (
                pytest.raises(BlockingIOError),
                CacheEntry(target).try_saving(),
            ):  # still an exclusive lock on the same file
                pass

        assert lock_file.read_bytes() == b"kept"

    @pytest.mark.parametrize("n_waiters", [1, 10, 100])
    async def test_blocking_waits_then_proceeds(self, tmp_path: Path, n_waiters: int):
        """Blocking callers should wait for first holder, then see the produced file."""
        target = tmp_path / "produced.txt"
        order: list[str] = []

        async def holder() -> None:
            async with CacheEntry(target).saving() as should_proceed:
                assert should_proceed is True
                order.append("holder-acquired")
                await asyncio.sleep(0.1)  # Simulate work
                target.write_text("done")
                order.append("holder-released")

        async def waiter(idx: int) -> None:
            # Small delay so holder acquires first
            await asyncio.sleep(0.02)
            async with CacheEntry(target).saving() as should_proceed:
                order.append(f"waiter-{idx}-acquired")
                # File was produced by holder, so should_proceed is False
                assert should_proceed is False

        await asyncio.gather(holder(), *(waiter(i) for i in range(n_waiters)))
        assert order[0] == "holder-acquired"
        assert order[1] == "holder-released"
        # All waiters must appear after holder released
        waiter_events = order[2:]
        assert len(waiter_events) == n_waiters
        assert all(e.startswith("waiter-") and e.endswith("-acquired") for e in waiter_events)


# ============================================================================
# file_hash unit tests
# ============================================================================


class TestFileHash:
    """Tests for hash_utils.file_hash."""

    async def test_hashes_off_the_event_loop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """file_hash returns the file's digest, and no hashing (nor reading) runs on the
        event loop thread: one worker-thread job for the whole file."""
        data = os.urandom(3 * 256 * 1024 + 5)
        path = tmp_path / "blob"
        path.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        loop_thread = threading.get_ident()
        update_threads: list[int] = []
        real_new = hashlib.new

        class RecordingHash:
            """hashlib.new stand-in: records the thread of each update()."""

            def __init__(self, name: str) -> None:
                self._hash = real_new(name)

            def update(self, chunk: bytes) -> None:
                update_threads.append(threading.get_ident())
                self._hash.update(chunk)

            def hexdigest(self) -> str:
                return self._hash.hexdigest()

        monkeypatch.setattr(hashlib, "new", RecordingHash)

        assert await file_hash(path, "sha256") == expected
        assert update_threads
        assert loop_thread not in update_threads


# ============================================================================
# Concurrent retrieve tests — regression for EC2 race condition
# ============================================================================


class TestConcurrentRetrieve:
    """Tests for concurrent retrieve() calls.

    These tests reproduce the race condition found on EC2 where two concurrent
    base image downloads would race: one's retry cleanup (dest.unlink) deleted
    the .zst file while the other coroutine was decompressing it.
    """

    @pytest.mark.parametrize("concurrency", [1, 10, 100])
    async def test_concurrent_same_url_with_decompress(self, tmp_path: Path, concurrency: int):
        """Concurrent retrieve() with decompress_zstd should not race.

        Regression: on EC2, two base image downloads raced — one's retry
        cleanup deleted .zst while the other was decompressing it.
        The file lock on the final output serializes the full pipeline.
        """
        original = b"concurrent decompression test content"
        compressed = zstd.compress(original)
        compressed_hash = hashlib.sha256(compressed).hexdigest()
        url = "https://example.com/concurrent.txt.zst"

        with aioresponses() as m:
            for _ in range(concurrency):
                m.get(url, body=compressed)

            results = await asyncio.gather(
                *[
                    retrieve(
                        url=url,
                        known_hash=f"sha256:{compressed_hash}",
                        path=tmp_path,
                        decompress=True,
                    )
                    for _ in range(concurrency)
                ]
            )

            # All should return the same decompressed path
            assert all(r == results[0] for r in results)
            assert results[0].name == "concurrent.txt"
            assert results[0].read_bytes() == original

    @pytest.mark.parametrize("concurrency", [1, 10, 100])
    async def test_concurrent_same_url_no_decompress(self, tmp_path: Path, concurrency: int):
        """Concurrent retrieve() without decompress should not corrupt the file."""
        content = b"concurrent download no decompress"
        content_hash = hashlib.sha256(content).hexdigest()
        url = "https://example.com/concurrent-plain.txt"

        with aioresponses() as m:
            for _ in range(concurrency):
                m.get(url, body=content)

            results = await asyncio.gather(
                *[
                    retrieve(
                        url=url,
                        known_hash=f"sha256:{content_hash}",
                        path=tmp_path,
                        decompress=False,
                    )
                    for _ in range(concurrency)
                ]
            )

            assert all(r == results[0] for r in results)
            assert results[0].read_bytes() == content

    @pytest.mark.parametrize("concurrency", [1, 10, 100])
    async def test_concurrent_different_urls_not_serialized(self, tmp_path: Path, concurrency: int):
        """Concurrent retrieves of different URLs should run in parallel."""
        pairs = [(f"https://example.com/file-{i}.txt", f"content for file {i}".encode()) for i in range(concurrency)]
        hashes = {url: hashlib.sha256(body).hexdigest() for url, body in pairs}

        with aioresponses() as m:
            for url, body in pairs:
                m.get(url, body=body)

            results = await asyncio.gather(
                *[
                    retrieve(
                        url=url,
                        known_hash=f"sha256:{hashes[url]}",
                        path=tmp_path,
                        decompress=False,
                    )
                    for url, _ in pairs
                ]
            )

            for i, (_url, body) in enumerate(pairs):
                assert results[i].name == f"file-{i}.txt"
                assert results[i].read_bytes() == body

    @pytest.mark.parametrize("concurrency", [1, 10, 100])
    async def test_concurrent_first_fails_rest_succeed(
        self, tmp_path: Path, concurrency: int, retry_waits: list[float]
    ):
        """When one concurrent retrieve fails, the others should still succeed."""
        content = b"success content"
        content_hash = hashlib.sha256(content).hexdigest()
        url_fail = "https://example.com/fail.txt"
        ok_urls = [f"https://example.com/ok-{i}.txt" for i in range(concurrency - 1)]

        with aioresponses() as m:
            # fail.txt always returns 500 (all 3 retry attempts)
            for _ in range(3):
                m.get(url_fail, status=500)
            for url in ok_urls:
                m.get(url, body=content)

            results = await asyncio.gather(
                retrieve(url=url_fail, known_hash="sha256:abc123", path=tmp_path, decompress=False),
                *[
                    retrieve(url=url, known_hash=f"sha256:{content_hash}", path=tmp_path, decompress=False)
                    for url in ok_urls
                ],
                return_exceptions=True,
            )

            # First should fail with TransientError ASSET_DOWNLOAD
            assert isinstance(results[0], TransientError)
            assert results[0].code is ErrorCode.ASSET_DOWNLOAD
            # Rest should succeed
            for r in results[1:]:
                assert isinstance(r, Path)
                assert r.read_bytes() == content

    async def test_concurrent_empty_target_proceeds(self, tmp_path: Path):
        """Empty target file should trigger a fresh download.

        Exercises CacheEntry.saving where st_size == 0 causes
        should_proceed=True even though the file exists.
        """
        content = b"fresh download content"
        content_hash = hashlib.sha256(content).hexdigest()
        url = "https://example.com/empty-target.txt"

        # Pre-create an empty file — saving should yield True
        empty_file = tmp_path / "empty-target.txt"
        empty_file.write_bytes(b"")

        with aioresponses() as m:
            m.get(url, body=content)

            result = await retrieve(
                url=url,
                known_hash=f"sha256:{content_hash}",
                path=tmp_path,
                decompress=False,
            )

            assert result == empty_file
            assert result.read_bytes() == content

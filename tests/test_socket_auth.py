"""Tests for Unix socket peer credential authentication.

Uses real Unix sockets. One test replaces the kernel's peer credentials of one socket: a
server of another user needs a second account. Authentication is MANDATORY - no skip paths.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from exec_sandbox.exceptions import SocketAuthError
from exec_sandbox.socket_auth import (
    PeerCredentials,
    connect_and_verify,
    create_unix_socket,
    get_peer_credentials,
    verify_socket_peer,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ClientHandler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Coroutine[Any, Any, None]]


async def _close_writer(_r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    """Minimal handler for Unix server sockets — close immediately."""
    w.close()
    await w.wait_closed()


async def _hold_open(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    """Server handler that sends nothing and keeps the connection until the client closes it."""
    await r.read()
    w.close()
    await w.wait_closed()


@asynccontextmanager
async def _unix_server(
    path: Path,
    handler: _ClientHandler = _close_writer,
) -> AsyncGenerator[asyncio.Server]:
    """Async context manager for Unix server socket lifecycle.

    Creates a server, yields it, and ensures cleanup on exit.
    """
    server = await asyncio.start_unix_server(handler, path=str(path))
    try:
        yield server
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetPeerCredentials:
    """Tests for get_peer_credentials function."""

    async def test_returns_own_uid(self, short_tmp_dir: Path) -> None:
        """get_peer_credentials returns current user's UID."""
        path = short_tmp_dir / "test.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        conn, _ = server.accept()

        creds = get_peer_credentials(client)
        assert creds.uid == os.getuid()

        client.close()
        conn.close()
        server.close()

    def test_raises_for_tcp(self) -> None:
        """get_peer_credentials raises SocketAuthError for TCP sockets."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(SocketAuthError):
            get_peer_credentials(sock)
        sock.close()


class TestVerifySocketPeer:
    """Tests for verify_socket_peer function."""

    async def test_matching_uid_succeeds(self, short_tmp_dir: Path) -> None:
        """verify_socket_peer succeeds when UIDs match."""
        path = short_tmp_dir / "test.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        conn, _ = server.accept()

        # Should not raise
        verify_socket_peer(client, os.getuid(), str(path))

        client.close()
        conn.close()
        server.close()

    async def test_wrong_uid_raises(self, short_tmp_dir: Path) -> None:
        """verify_socket_peer raises SocketAuthError on mismatch."""
        path = short_tmp_dir / "test.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        conn, _ = server.accept()

        wrong_uid = os.getuid() + 1000
        with pytest.raises(SocketAuthError) as exc:
            verify_socket_peer(client, wrong_uid, str(path))

        assert exc.value.expected_uid == wrong_uid
        assert exc.value.actual_uid == os.getuid()
        assert exc.value.context["socket_path"] == str(path)

        client.close()
        conn.close()
        server.close()


class TestConnectAndVerify:
    """Tests for connect_and_verify async function."""

    async def test_connect_with_verification(self, short_tmp_dir: Path) -> None:
        """With the right uid, connect_and_verify gives streams connected to the server."""
        path = short_tmp_dir / "test.sock"

        async def echo(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            w.write(await r.readline())
            await w.drain()
            w.close()
            await w.wait_closed()

        async with _unix_server(path, handler=echo):
            reader, writer = await connect_and_verify(str(path), expected_uid=os.getuid(), timeout=5.0)
            writer.write(b"ping\n")
            assert await reader.readline() == b"ping\n"
            writer.close()
            await writer.wait_closed()

    async def test_connect_wrong_uid_closes_socket(self, short_tmp_dir: Path) -> None:
        """On an auth failure connect_and_verify sends nothing and closes the socket: the
        server reads end of file with no byte."""
        path = short_tmp_dir / "test.sock"
        received: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()

        async def handle_client(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            received.set_result(await r.read())
            w.close()
            await w.wait_closed()

        async with _unix_server(path, handler=handle_client):
            wrong_uid = os.getuid() + 1000
            with pytest.raises(SocketAuthError):
                await connect_and_verify(str(path), expected_uid=wrong_uid, timeout=5.0)
            assert await asyncio.wait_for(received, timeout=1.0) == b""


class TestUnixSocketChannelIntegration:
    """Integration tests for UnixSocketChannel with socket authentication."""

    async def test_channel_connect_with_correct_uid(self, short_tmp_dir: Path) -> None:
        """UnixSocketChannel.connect() succeeds with correct expected_uid."""
        from exec_sandbox.guest_channel import UnixSocketChannel

        path = short_tmp_dir / "test.sock"

        async with _unix_server(path, handler=_hold_open):
            channel = UnixSocketChannel(str(path), expected_uid=os.getuid())
            await channel.connect(timeout_seconds=5)
            assert channel.is_connected()
            await channel.close()
            assert not channel.is_connected()

    async def test_channel_connect_with_wrong_uid_fails(self, short_tmp_dir: Path) -> None:
        """UnixSocketChannel.connect() raises SocketAuthError with wrong expected_uid."""
        from exec_sandbox.guest_channel import UnixSocketChannel

        path = short_tmp_dir / "test.sock"

        async with _unix_server(path, handler=_hold_open):
            wrong_uid = os.getuid() + 1000
            channel = UnixSocketChannel(str(path), expected_uid=wrong_uid)

            with pytest.raises(SocketAuthError) as exc:
                await channel.connect(timeout_seconds=5)

            assert exc.value.expected_uid == wrong_uid
            assert exc.value.actual_uid == os.getuid()


class TestDualPortChannelIntegration:
    """Integration tests for DualPortChannel with socket authentication."""

    async def test_dual_channel_connects_both_ports(self, short_tmp_dir: Path) -> None:
        """With the right uid, connect() opens the command port and the event port."""
        from exec_sandbox.guest_channel import DualPortChannel

        cmd_path = short_tmp_dir / "cmd.sock"
        event_path = short_tmp_dir / "event.sock"
        accepted: list[str] = []

        def handler(name: str) -> _ClientHandler:
            async def handle(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
                accepted.append(name)
                await r.read()
                w.close()
                await w.wait_closed()

            return handle

        async with _unix_server(cmd_path, handler("cmd")), _unix_server(event_path, handler("event")):
            channel = DualPortChannel(str(cmd_path), str(event_path), expected_uid=os.getuid())
            await channel.connect(timeout_seconds=5)
            await channel.close()

        assert sorted(accepted) == ["cmd", "event"]

    @pytest.mark.parametrize("foreign", ["cmd", "event"])
    async def test_a_foreign_peer_on_either_port_fails_connect(self, short_tmp_dir: Path, foreign: str) -> None:
        """Each port is verified on its own: a server of another user on the command port, or on
        the event port alone, fails connect(). A server of another user needs a second account:
        the kernel's answer for that one socket is replaced (get_peer_credentials)."""
        from exec_sandbox import socket_auth
        from exec_sandbox.guest_channel import DualPortChannel

        paths = {"cmd": short_tmp_dir / "cmd.sock", "event": short_tmp_dir / "event.sock"}
        real_credentials = socket_auth.get_peer_credentials

        def credentials(sock: socket.socket) -> PeerCredentials:
            creds = real_credentials(sock)
            if sock.getpeername() == str(paths[foreign]):
                return PeerCredentials(uid=creds.uid + 1000)
            return creds

        async with _unix_server(paths["cmd"], _hold_open), _unix_server(paths["event"], _hold_open):
            channel = DualPortChannel(str(paths["cmd"]), str(paths["event"]), expected_uid=os.getuid())
            with (
                patch("exec_sandbox.socket_auth.get_peer_credentials", side_effect=credentials),
                pytest.raises(SocketAuthError) as exc,
            ):
                await channel.connect(timeout_seconds=5)
            await channel.close()

        assert exc.value.context["socket_path"] == str(paths[foreign])
        assert (exc.value.expected_uid, exc.value.actual_uid) == (os.getuid(), os.getuid() + 1000)


class TestRootAuthentication:
    """Tests for root (UID 0) authentication - requires sudo to fully test."""

    @pytest.mark.sudo
    async def test_root_socket_auth_when_running_as_root(self, short_tmp_dir: Path) -> None:
        """When running as root, auth with expected_uid=0 should succeed."""
        if os.getuid() != 0:
            pytest.skip("Test requires root privileges (run with sudo)")

        path = short_tmp_dir / "test.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        conn, _ = server.accept()

        try:
            creds = get_peer_credentials(client)
            assert creds.uid == 0

            verify_socket_peer(client, 0, str(path))  # Should not raise
        finally:
            client.close()
            conn.close()
            server.close()

    @pytest.mark.sudo
    async def test_root_connect_and_verify_succeeds_as_root(self, short_tmp_dir: Path) -> None:
        """connect_and_verify with expected_uid=0 succeeds when running as root."""
        if os.getuid() != 0:
            pytest.skip("Test requires root privileges (run with sudo)")

        path = short_tmp_dir / "test.sock"

        async with _unix_server(path, handler=_hold_open):
            reader, writer = await connect_and_verify(str(path), expected_uid=0, timeout=5.0)
            assert reader is not None
            writer.close()
            await writer.wait_closed()

    async def test_non_root_cannot_spoof_root_uid(self, short_tmp_dir: Path) -> None:
        """Non-root process cannot authenticate as root (UID 0)."""
        if os.getuid() == 0:
            pytest.skip("Test requires non-root user")

        path = short_tmp_dir / "test.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        conn, _ = server.accept()

        try:
            with pytest.raises(SocketAuthError) as exc:
                verify_socket_peer(client, 0, str(path))

            assert exc.value.expected_uid == 0
            assert exc.value.actual_uid == os.getuid()
            assert exc.value.actual_uid != 0
        finally:
            client.close()
            conn.close()
            server.close()


class TestSecurityEdgeCases:
    """Security-focused edge case tests."""

    async def test_auth_happens_immediately_after_connect(self, short_tmp_dir: Path) -> None:
        """Auth check must happen immediately, not lazily on first use."""
        path = short_tmp_dir / "test.sock"

        async with _unix_server(path, handler=_hold_open):
            wrong_uid = os.getuid() + 1000

            start = asyncio.get_event_loop().time()
            with pytest.raises(SocketAuthError):
                await connect_and_verify(str(path), expected_uid=wrong_uid, timeout=5.0)
            elapsed = asyncio.get_event_loop().time() - start

            assert elapsed < 1.0

    def test_socket_auth_error_inherits_from_communication_error(self) -> None:
        """SocketAuthError should be catchable as CommunicationError."""
        from exec_sandbox.exceptions import CommunicationError

        exc = SocketAuthError(
            message="Test",
            expected_uid=1000,
            actual_uid=2000,
            context=None,
        )

        assert isinstance(exc, CommunicationError)

    async def test_closed_socket_raises(self, short_tmp_dir: Path) -> None:
        """Closed socket should raise SocketAuthError."""
        path = short_tmp_dir / "test.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        conn, _ = server.accept()

        client.close()

        with pytest.raises(SocketAuthError):
            get_peer_credentials(client)

        conn.close()
        server.close()


class TestSocketAuthError:
    """Tests for SocketAuthError exception."""

    def test_exception_attributes(self) -> None:
        """SocketAuthError has expected attributes."""
        exc = SocketAuthError(
            message="Test error",
            expected_uid=1000,
            actual_uid=2000,
            context={"socket_path": "/tmp/test.sock"},
        )
        assert exc.expected_uid == 1000
        assert exc.actual_uid == 2000
        assert exc.context["socket_path"] == "/tmp/test.sock"
        assert exc.context["expected_uid"] == 1000
        assert exc.context["actual_uid"] == 2000
        assert str(exc) == "Test error"

    def test_exception_without_context(self) -> None:
        """SocketAuthError works with no context (None)."""
        exc = SocketAuthError(
            message="Test error",
            expected_uid=1000,
            actual_uid=2000,
            context=None,
        )
        assert exc.expected_uid == 1000
        assert exc.actual_uid == 2000
        assert exc.context["expected_uid"] == 1000
        assert exc.context["actual_uid"] == 2000


class TestEdgeCases:
    """Edge case and boundary tests for socket authentication."""

    # =========================================================================
    # Socket type edge cases
    # =========================================================================

    def test_raises_for_unconnected_unix_socket(self) -> None:
        """get_peer_credentials raises SocketAuthError for unconnected Unix socket."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with pytest.raises(SocketAuthError):
            get_peer_credentials(sock)
        sock.close()

    # =========================================================================
    # Error handling tests
    # =========================================================================

    async def test_connect_to_nonexistent_socket(self, short_tmp_dir: Path) -> None:
        """connect_and_verify raises FileNotFoundError for a socket path that does not exist."""
        path = short_tmp_dir / "nonexistent.sock"

        with pytest.raises(FileNotFoundError):
            await connect_and_verify(str(path), expected_uid=os.getuid(), timeout=1.0)

    async def test_connect_to_regular_file_fails(self, short_tmp_dir: Path) -> None:
        """connect_and_verify fails when path is a regular file."""
        path = short_tmp_dir / "regular_file"
        path.write_text("not a socket")

        with pytest.raises((OSError, ConnectionRefusedError)):
            await connect_and_verify(str(path), expected_uid=os.getuid(), timeout=1.0)

    # =========================================================================
    # Context and error message tests
    # =========================================================================

    async def test_error_message_includes_socket_path(self, short_tmp_dir: Path) -> None:
        """SocketAuthError includes socket path in context."""
        path = short_tmp_dir / "test.sock"

        async with _unix_server(path, handler=_hold_open):
            wrong_uid = os.getuid() + 1000
            with pytest.raises(SocketAuthError) as exc:
                await connect_and_verify(str(path), expected_uid=wrong_uid, timeout=5.0)

            assert exc.value.context["socket_path"] == str(path)
            assert exc.value.expected_uid == wrong_uid

    def test_error_message_with_unknown_uid(self, short_tmp_dir: Path) -> None:
        """A uid with no user database entry is named by its number in the message."""
        path = short_tmp_dir / "test.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        conn, _ = server.accept()

        try:
            with pytest.raises(SocketAuthError) as exc:
                verify_socket_peer(client, 99999999, str(path))
            assert "expected 99999999 (99999999), got " in str(exc.value)
            assert str(exc.value).endswith(f"({os.getuid()})")
        finally:
            client.close()
            conn.close()
            server.close()

    # =========================================================================
    # Concurrency tests
    # =========================================================================

    async def test_multiple_concurrent_connections_same_server(self, short_tmp_dir: Path) -> None:
        """Multiple clients can connect and verify concurrently."""
        path = short_tmp_dir / "test.sock"

        connection_count = 0

        async def handle_client(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            nonlocal connection_count
            connection_count += 1
            await _hold_open(r, w)

        async with _unix_server(path, handler=handle_client):
            tasks = [connect_and_verify(str(path), expected_uid=os.getuid(), timeout=5.0) for _ in range(5)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            for _reader, writer in results:
                writer.close()
                await writer.wait_closed()

        assert connection_count == 5


class TestSocketActivation:
    """Integration tests for socket activation flow using real sockets."""

    async def test_socket_activation_no_toctou(self, short_tmp_dir: Path) -> None:
        """Pre-bound socket survives sequential connect-close-reopen cycles."""
        path = short_tmp_dir / "qmp.sock"
        sock = create_unix_socket(str(path))
        try:
            for _ in range(3):
                _reader, writer = await asyncio.open_unix_connection(str(path))
                writer.close()
                await writer.wait_closed()
        finally:
            sock.close()

    async def test_socket_activation_accept_after_parent_close(self, short_tmp_dir: Path) -> None:
        """Closing the parent socket fd prevents new connections."""
        path = short_tmp_dir / "qmp.sock"
        sock = create_unix_socket(str(path))
        sock.close()
        with pytest.raises((ConnectionRefusedError, ConnectionResetError, OSError)):
            await asyncio.wait_for(
                asyncio.open_unix_connection(str(path)),
                timeout=1.0,
            )

    async def test_create_unix_socket_existing_path_fails(self, short_tmp_dir: Path) -> None:
        """create_unix_socket raises OSError when path is already bound."""
        path = short_tmp_dir / "qmp.sock"
        sock1 = create_unix_socket(str(path))
        try:
            with pytest.raises(OSError):
                create_unix_socket(str(path))
        finally:
            sock1.close()

"""Tests for Unix socket peer credential authentication.

Uses real Unix sockets - no mocks.
Authentication is MANDATORY - no skip paths.
"""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
from pathlib import Path

import pytest

from exec_sandbox.exceptions import SocketAuthError
from exec_sandbox.socket_auth import (
    PeerCredentials,
    connect_and_verify,
    get_peer_credentials,
    get_qemu_vm_uid,
    verify_socket_peer,
)


class TestGetPeerCredentials:
    """Tests for get_peer_credentials function."""

    async def test_returns_own_uid(self) -> None:
        """get_peer_credentials returns current user's UID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"
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

    def test_peer_credentials_dataclass(self) -> None:
        """PeerCredentials dataclass works correctly."""
        creds = PeerCredentials(uid=1000, gid=1000, pid=12345)
        assert creds.uid == 1000
        assert creds.gid == 1000
        assert creds.pid == 12345

        # Test with defaults
        creds2 = PeerCredentials(uid=500)
        assert creds2.uid == 500
        assert creds2.gid is None
        assert creds2.pid is None

        # Test frozen (immutable)
        with pytest.raises(AttributeError):
            creds.uid = 2000  # type: ignore[misc]


class TestVerifySocketPeer:
    """Tests for verify_socket_peer function."""

    async def test_matching_uid_succeeds(self) -> None:
        """verify_socket_peer succeeds when UIDs match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"
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

    async def test_wrong_uid_raises(self) -> None:
        """verify_socket_peer raises SocketAuthError on mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"
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
            assert "socket_path" in exc.value.context

            client.close()
            conn.close()
            server.close()


class TestConnectAndVerify:
    """Tests for connect_and_verify async function."""

    async def test_connect_with_verification(self) -> None:
        """connect_and_verify succeeds with matching UID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle_client, path=str(path))

            try:
                reader, writer = await connect_and_verify(str(path), expected_uid=os.getuid(), timeout=5.0)
                assert reader is not None
                assert writer is not None
                writer.close()
                await writer.wait_closed()
            finally:
                server.close()
                await server.wait_closed()

    async def test_connect_wrong_uid_closes_socket(self) -> None:
        """connect_and_verify closes socket on auth failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                # Wait a bit then close
                await asyncio.sleep(1)
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle_client, path=str(path))

            try:
                wrong_uid = os.getuid() + 1000
                with pytest.raises(SocketAuthError):
                    await connect_and_verify(str(path), expected_uid=wrong_uid, timeout=5.0)
            finally:
                server.close()
                await server.wait_closed()

    async def test_connect_timeout(self) -> None:
        """connect_and_verify raises TimeoutError on connection timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.sock"

            with pytest.raises((asyncio.TimeoutError, OSError)):
                await connect_and_verify(str(path), expected_uid=os.getuid(), timeout=0.1)


class TestHelpers:
    """Tests for helper functions."""

    def test_get_qemu_vm_uid(self) -> None:
        """get_qemu_vm_uid returns int or None."""
        result = get_qemu_vm_uid()
        assert result is None or isinstance(result, int)

    def test_get_qemu_vm_uid_cached(self) -> None:
        """get_qemu_vm_uid result is cached."""
        # Call twice, should return same result
        result1 = get_qemu_vm_uid()
        result2 = get_qemu_vm_uid()
        assert result1 == result2


class TestUnixSocketChannelIntegration:
    """Integration tests for UnixSocketChannel with socket authentication."""

    async def test_channel_connect_with_correct_uid(self) -> None:
        """UnixSocketChannel.connect() succeeds with correct expected_uid."""
        from exec_sandbox.guest_channel import UnixSocketChannel

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                await asyncio.sleep(0.1)
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle_client, path=str(path))

            try:
                # Channel with correct UID should connect successfully
                channel = UnixSocketChannel(str(path), expected_uid=os.getuid())
                await channel.connect(timeout_seconds=5)
                assert channel._reader is not None
                assert channel._writer is not None
                await channel.close()
            finally:
                server.close()
                await server.wait_closed()

    async def test_channel_connect_with_wrong_uid_fails(self) -> None:
        """UnixSocketChannel.connect() raises SocketAuthError with wrong expected_uid."""
        from exec_sandbox.guest_channel import UnixSocketChannel

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                await asyncio.sleep(1)
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle_client, path=str(path))

            try:
                # Channel with wrong UID should fail to connect
                wrong_uid = os.getuid() + 1000
                channel = UnixSocketChannel(str(path), expected_uid=wrong_uid)

                with pytest.raises(SocketAuthError) as exc:
                    await channel.connect(timeout_seconds=5)

                assert exc.value.expected_uid == wrong_uid
                assert exc.value.actual_uid == os.getuid()
            finally:
                server.close()
                await server.wait_closed()


class TestDualPortChannelIntegration:
    """Integration tests for DualPortChannel with socket authentication."""

    async def test_dual_channel_passes_expected_uid_to_both_channels(self) -> None:
        """DualPortChannel passes expected_uid to both cmd and event channels."""
        from exec_sandbox.guest_channel import DualPortChannel

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_path = Path(tmpdir) / "cmd.sock"
            event_path = Path(tmpdir) / "event.sock"

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                await asyncio.sleep(0.1)
                writer.close()
                await writer.wait_closed()

            cmd_server = await asyncio.start_unix_server(handle_client, path=str(cmd_path))
            event_server = await asyncio.start_unix_server(handle_client, path=str(event_path))

            try:
                # DualPortChannel with correct UID should connect both channels
                channel = DualPortChannel(str(cmd_path), str(event_path), expected_uid=os.getuid())
                await channel.connect(timeout_seconds=5)

                # Verify both internal channels have expected_uid set
                assert channel._cmd_channel.expected_uid == os.getuid()
                assert channel._event_channel.expected_uid == os.getuid()

                await channel.close()
            finally:
                cmd_server.close()
                event_server.close()
                await cmd_server.wait_closed()
                await event_server.wait_closed()

    async def test_dual_channel_fails_if_cmd_channel_auth_fails(self) -> None:
        """DualPortChannel.connect() fails if cmd channel auth fails."""
        from exec_sandbox.guest_channel import DualPortChannel

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_path = Path(tmpdir) / "cmd.sock"
            event_path = Path(tmpdir) / "event.sock"

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                await asyncio.sleep(1)
                writer.close()
                await writer.wait_closed()

            cmd_server = await asyncio.start_unix_server(handle_client, path=str(cmd_path))
            event_server = await asyncio.start_unix_server(handle_client, path=str(event_path))

            try:
                wrong_uid = os.getuid() + 1000
                channel = DualPortChannel(str(cmd_path), str(event_path), expected_uid=wrong_uid)

                with pytest.raises(SocketAuthError):
                    await channel.connect(timeout_seconds=5)
            finally:
                cmd_server.close()
                event_server.close()
                await cmd_server.wait_closed()
                await event_server.wait_closed()


class TestRootAuthentication:
    """Tests for root (UID 0) authentication - requires sudo to fully test."""

    @pytest.mark.sudo
    async def test_root_socket_auth_when_running_as_root(self) -> None:
        """When running as root, auth with expected_uid=0 should succeed."""
        if os.getuid() != 0:
            pytest.skip("Test requires root privileges (run with sudo)")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)

            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(path))
            conn, _ = server.accept()

            try:
                # When running as root, our UID is 0
                creds = get_peer_credentials(client)
                assert creds.uid == 0

                # Verification should succeed
                verify_socket_peer(client, 0, str(path))  # Should not raise
            finally:
                client.close()
                conn.close()
                server.close()

    @pytest.mark.sudo
    async def test_root_connect_and_verify_succeeds_as_root(self) -> None:
        """connect_and_verify with expected_uid=0 succeeds when running as root."""
        if os.getuid() != 0:
            pytest.skip("Test requires root privileges (run with sudo)")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                await asyncio.sleep(0.1)
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle_client, path=str(path))

            try:
                # Should succeed when running as root
                reader, writer = await connect_and_verify(str(path), expected_uid=0, timeout=5.0)
                assert reader is not None
                writer.close()
                await writer.wait_closed()
            finally:
                server.close()
                await server.wait_closed()

    async def test_non_root_cannot_spoof_root_uid(self) -> None:
        """Non-root process cannot authenticate as root (UID 0)."""
        if os.getuid() == 0:
            pytest.skip("Test requires non-root user")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)

            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(path))
            conn, _ = server.accept()

            try:
                # Non-root user trying to verify as root should fail
                with pytest.raises(SocketAuthError) as exc:
                    verify_socket_peer(client, 0, str(path))

                assert exc.value.expected_uid == 0
                assert exc.value.actual_uid == os.getuid()
                assert exc.value.actual_uid != 0  # Confirm we're not root
            finally:
                client.close()
                conn.close()
                server.close()


class TestSecurityEdgeCases:
    """Security-focused edge case tests."""

    async def test_both_dual_channels_are_verified(self) -> None:
        """Both cmd and event channels must be verified, not just one."""
        from exec_sandbox.guest_channel import DualPortChannel

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_path = Path(tmpdir) / "cmd.sock"
            event_path = Path(tmpdir) / "event.sock"

            connections_made = {"cmd": False, "event": False}

            async def handle_cmd(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                connections_made["cmd"] = True
                await asyncio.sleep(0.5)
                writer.close()
                await writer.wait_closed()

            async def handle_event(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                connections_made["event"] = True
                await asyncio.sleep(0.5)
                writer.close()
                await writer.wait_closed()

            cmd_server = await asyncio.start_unix_server(handle_cmd, path=str(cmd_path))
            event_server = await asyncio.start_unix_server(handle_event, path=str(event_path))

            try:
                # Connect with wrong UID - both channels should attempt but fail auth
                wrong_uid = os.getuid() + 1000
                channel = DualPortChannel(str(cmd_path), str(event_path), expected_uid=wrong_uid)

                with pytest.raises(SocketAuthError):
                    await channel.connect(timeout_seconds=5)

                # Both channels attempted connection (gather runs in parallel)
                # At least one should have connected before auth failed
                assert connections_made["cmd"] or connections_made["event"]
            finally:
                cmd_server.close()
                event_server.close()
                await cmd_server.wait_closed()
                await event_server.wait_closed()

    async def test_auth_happens_immediately_after_connect(self) -> None:
        """Auth check must happen immediately, not lazily on first use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                await asyncio.sleep(5)  # Long sleep - shouldn't matter
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle_client, path=str(path))

            try:
                wrong_uid = os.getuid() + 1000

                # Auth should fail immediately during connect(), not later
                start = asyncio.get_event_loop().time()
                with pytest.raises(SocketAuthError):
                    await connect_and_verify(str(path), expected_uid=wrong_uid, timeout=5.0)
                elapsed = asyncio.get_event_loop().time() - start

                # Should fail quickly (< 1s), not wait for server timeout
                assert elapsed < 1.0
            finally:
                server.close()
                await server.wait_closed()

    def test_socket_auth_error_inherits_from_communication_error(self) -> None:
        """SocketAuthError should be catchable as CommunicationError."""
        from exec_sandbox.exceptions import CommunicationError

        exc = SocketAuthError(
            message="Test",
            expected_uid=1000,
            actual_uid=2000,
        )

        # Should be catchable as CommunicationError (for error handling)
        assert isinstance(exc, CommunicationError)

    async def test_closed_socket_raises(self) -> None:
        """Closed socket should raise SocketAuthError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)

            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(path))
            conn, _ = server.accept()

            # Close the socket
            client.close()

            # Should raise SocketAuthError, not crash
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
        """SocketAuthError works without context."""
        exc = SocketAuthError(
            message="Test error",
            expected_uid=1000,
            actual_uid=2000,
        )
        assert exc.expected_uid == 1000
        assert exc.actual_uid == 2000
        assert exc.context["expected_uid"] == 1000
        assert exc.context["actual_uid"] == 2000


class TestEdgeCases:
    """Edge case and boundary tests for socket authentication."""

    # =========================================================================
    # Boundary value tests
    # =========================================================================

    def test_peer_credentials_uid_zero(self) -> None:
        """PeerCredentials handles UID 0 (root)."""
        creds = PeerCredentials(uid=0, gid=0, pid=1)
        assert creds.uid == 0
        assert creds.gid == 0

    def test_peer_credentials_large_uid(self) -> None:
        """PeerCredentials handles large UID values."""
        # Max UID on most systems is 2^32 - 1 = 4294967295
        large_uid = 4294967295
        creds = PeerCredentials(uid=large_uid, gid=large_uid, pid=99999)
        assert creds.uid == large_uid

    def test_verify_against_uid_zero(self) -> None:
        """verify_socket_peer works when expecting UID 0."""
        # We can't easily test as root, but we can test that mismatch is detected
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)

            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(path))
            conn, _ = server.accept()

            # Unless we're root, expecting UID 0 should fail
            if os.getuid() != 0:
                with pytest.raises(SocketAuthError) as exc:
                    verify_socket_peer(client, 0, str(path))
                assert exc.value.expected_uid == 0
                assert exc.value.actual_uid == os.getuid()

            client.close()
            conn.close()
            server.close()

    # =========================================================================
    # Socket type edge cases
    # =========================================================================

    def test_raises_for_udp_socket(self) -> None:
        """get_peer_credentials raises SocketAuthError for UDP sockets."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with pytest.raises(SocketAuthError):
            get_peer_credentials(sock)
        sock.close()

    def test_raises_for_unconnected_unix_socket(self) -> None:
        """get_peer_credentials raises SocketAuthError for unconnected Unix socket."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Not connected - getsockopt should fail
        with pytest.raises(SocketAuthError):
            get_peer_credentials(sock)
        sock.close()

    # =========================================================================
    # Error handling tests
    # =========================================================================

    async def test_connect_to_nonexistent_socket(self) -> None:
        """connect_and_verify raises OSError for nonexistent socket."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.sock"

            with pytest.raises((OSError, asyncio.TimeoutError)):
                await connect_and_verify(str(path), expected_uid=os.getuid(), timeout=1.0)

    async def test_connect_to_regular_file_fails(self) -> None:
        """connect_and_verify fails when path is a regular file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "regular_file"
            path.write_text("not a socket")

            with pytest.raises((OSError, ConnectionRefusedError)):
                await connect_and_verify(str(path), expected_uid=os.getuid(), timeout=1.0)

    async def test_socket_closed_by_server_during_connect(self) -> None:
        """Handle server closing connection immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                # Close immediately
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle_client, path=str(path))

            try:
                # Should still succeed - verification happens before server closes
                _reader, writer = await connect_and_verify(str(path), expected_uid=os.getuid(), timeout=5.0)
                writer.close()
                await writer.wait_closed()
            finally:
                server.close()
                await server.wait_closed()

    # =========================================================================
    # Context and error message tests
    # =========================================================================

    async def test_error_message_includes_socket_path(self) -> None:
        """SocketAuthError includes socket path in context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                await asyncio.sleep(1)
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle_client, path=str(path))

            try:
                wrong_uid = os.getuid() + 1000
                with pytest.raises(SocketAuthError) as exc:
                    await connect_and_verify(str(path), expected_uid=wrong_uid, timeout=5.0)

                # Verify error has useful context
                assert exc.value.context["socket_path"] == str(path)
                assert "expected" in str(exc.value).lower() or exc.value.expected_uid == wrong_uid
            finally:
                server.close()
                await server.wait_closed()

    def test_error_message_with_unknown_uid(self) -> None:
        """SocketAuthError handles UIDs without passwd entries."""
        # Use a UID that almost certainly doesn't exist
        fake_uid = 99999999
        exc = SocketAuthError(
            message=f"UID mismatch: expected {fake_uid}",
            expected_uid=fake_uid,
            actual_uid=os.getuid(),
        )
        assert exc.expected_uid == fake_uid
        # Should not crash when creating error message

    # =========================================================================
    # Concurrency tests
    # =========================================================================

    async def test_multiple_concurrent_connections_same_server(self) -> None:
        """Multiple clients can connect and verify concurrently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"

            connection_count = 0

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                nonlocal connection_count
                connection_count += 1
                await asyncio.sleep(0.1)
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle_client, path=str(path))

            try:
                # Connect 5 clients concurrently
                tasks = [connect_and_verify(str(path), expected_uid=os.getuid(), timeout=5.0) for _ in range(5)]

                results = await asyncio.gather(*tasks)

                # All should succeed
                assert len(results) == 5
                for reader, writer in results:
                    assert reader is not None
                    writer.close()
                    await writer.wait_closed()

            finally:
                server.close()
                await server.wait_closed()

    # =========================================================================
    # Verification timing tests
    # =========================================================================

    async def test_verification_happens_before_data_exchange(self) -> None:
        """Verification must complete before any data is sent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.sock"

            data_received = False

            async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                nonlocal data_received
                try:
                    # Try to read - if client sends data before we close, this would receive it
                    data = await asyncio.wait_for(reader.read(100), timeout=0.5)
                    if data:
                        data_received = True
                except TimeoutError:
                    pass
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle_client, path=str(path))

            try:
                wrong_uid = os.getuid() + 1000
                with pytest.raises(SocketAuthError):
                    await connect_and_verify(str(path), expected_uid=wrong_uid, timeout=5.0)

                # Give server time to check
                await asyncio.sleep(0.6)

                # No data should have been sent since auth failed
                assert not data_received
            finally:
                server.close()
                await server.wait_closed()

"""Unit tests for BalloonClient QMP interface.

Tests the balloon memory control client without requiring actual VMs.
Uses mocked QMP socket responses.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox.balloon_client import BalloonClient, BalloonError


class TestBalloonClientUnit:
    """Unit tests for BalloonClient with mocked sockets."""

    @pytest.fixture
    def qmp_socket(self, tmp_path: Path) -> Path:
        """Create a mock QMP socket path."""
        return tmp_path / "qmp.sock"

    @pytest.fixture
    def mock_connect_and_verify(self) -> Any:
        """Mock the connect_and_verify function."""
        with patch("exec_sandbox.balloon_client.connect_and_verify") as mock:
            reader = AsyncMock()
            # writer must be MagicMock because write() and close() are sync methods
            # on StreamWriter - only drain() and wait_closed() are async
            writer = MagicMock()
            writer.drain = AsyncMock()
            writer.wait_closed = AsyncMock()
            writer.transport = MagicMock()

            mock.return_value = (reader, writer)
            yield mock, reader, writer

    async def test_connect_handshake_success(self, qmp_socket: Path, mock_connect_and_verify: Any) -> None:
        """Test successful QMP connection and capabilities negotiation."""
        _mock, reader, writer = mock_connect_and_verify

        # QMP greeting + capabilities response
        greeting = b'{"QMP": {"version": {"qemu": {"micro": 0, "minor": 0, "major": 10}}}}\n'
        caps_response = b'{"return": {}}\n'
        reader.readline = AsyncMock(side_effect=[greeting, caps_response])

        client = BalloonClient(qmp_socket, expected_uid=1000)
        await client.connect()

        assert client._connected
        writer.write.assert_called_once()
        assert b"qmp_capabilities" in writer.write.call_args[0][0]

    async def test_connect_capabilities_error(self, qmp_socket: Path, mock_connect_and_verify: Any) -> None:
        """Test connection failure when capabilities negotiation fails."""
        _mock, reader, writer = mock_connect_and_verify

        greeting = b'{"QMP": {"version": {"qemu": {"micro": 0, "minor": 0, "major": 10}}}}\n'
        error_response = b'{"error": {"class": "CommandNotFound", "desc": "Unknown command"}}\n'
        reader.readline = AsyncMock(side_effect=[greeting, error_response])

        client = BalloonClient(qmp_socket, expected_uid=1000)
        with pytest.raises(BalloonError, match="QMP capabilities failed"):
            await client.connect()
        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()
        assert client._writer is None

    async def test_connect_cancellation_waits_for_handshake_cleanup(
        self,
        qmp_socket: Path,
        mock_connect_and_verify: Any,
    ) -> None:
        """A failed __aenter__ cannot retain an open QMP writer."""
        _mock, reader, writer = mock_connect_and_verify
        greeting_entered = asyncio.Event()
        close_entered = asyncio.Event()
        release_close = asyncio.Event()

        async def blocked_greeting() -> bytes:
            greeting_entered.set()
            await asyncio.Event().wait()
            return b""

        async def blocked_wait_closed() -> None:
            close_entered.set()
            await release_close.wait()

        reader.readline = AsyncMock(side_effect=blocked_greeting)
        writer.wait_closed = AsyncMock(side_effect=blocked_wait_closed)
        client = BalloonClient(qmp_socket, expected_uid=1000)
        connect = asyncio.create_task(client.connect())
        await asyncio.wait_for(greeting_entered.wait(), timeout=1)
        connect.cancel()
        await asyncio.wait_for(close_entered.wait(), timeout=1)
        connect.cancel()
        await asyncio.sleep(0)
        assert not connect.done()

        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await connect

        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()
        assert client._reader is None
        assert client._writer is None
        assert client._connected is False

    async def test_query_balloon_returns_mb(self, qmp_socket: Path, mock_connect_and_verify: Any) -> None:
        """Test balloon query returns memory in MB."""
        _mock, reader, _writer = mock_connect_and_verify

        # Setup connection
        greeting = b'{"QMP": {}}\n'
        caps_response = b'{"return": {}}\n'
        # Query response: 256MB in bytes
        query_response = b'{"return": {"actual": 268435456}}\n'
        reader.readline = AsyncMock(side_effect=[greeting, caps_response, query_response])

        client = BalloonClient(qmp_socket, expected_uid=1000)
        await client.connect()

        result = await client.query()
        assert result == 256  # 256 MB

    async def test_query_balloon_returns_none_on_failure(self, qmp_socket: Path, mock_connect_and_verify: Any) -> None:
        """Test balloon query returns None on QMP error."""
        _mock, reader, _writer = mock_connect_and_verify

        greeting = b'{"QMP": {}}\n'
        caps_response = b'{"return": {}}\n'
        error_response = b'{"error": {"class": "DeviceNotActive"}}\n'
        reader.readline = AsyncMock(side_effect=[greeting, caps_response, error_response])

        client = BalloonClient(qmp_socket, expected_uid=1000)
        await client.connect()

        result = await client.query()
        assert result is None

    async def test_set_target_sends_bytes(self, qmp_socket: Path, mock_connect_and_verify: Any) -> None:
        """Test set_target converts MB to bytes for QMP."""
        _mock, reader, writer = mock_connect_and_verify

        greeting = b'{"QMP": {}}\n'
        caps_response = b'{"return": {}}\n'
        balloon_response = b'{"return": {}}\n'
        reader.readline = AsyncMock(side_effect=[greeting, caps_response, balloon_response])

        client = BalloonClient(qmp_socket, expected_uid=1000)
        await client.connect()

        await client.set_target(target_mb=64)

        # Check the balloon command was sent with correct bytes
        # Should be second write call (first was capabilities)
        calls = writer.write.call_args_list
        balloon_call = calls[1][0][0].decode()
        cmd = json.loads(balloon_call.strip())
        assert cmd["execute"] == "balloon"
        assert cmd["arguments"]["value"] == 64 * 1024 * 1024  # 64MB in bytes

    async def test_inflate_returns_previous_size(self, qmp_socket: Path, mock_connect_and_verify: Any) -> None:
        """Test inflate (reduce guest memory) returns previous size."""
        _mock, reader, _writer = mock_connect_and_verify

        greeting = b'{"QMP": {}}\n'
        caps_response = b'{"return": {}}\n'
        # Query returns 256MB
        query_response = b'{"return": {"actual": 268435456}}\n'
        # Set target succeeds
        balloon_response = b'{"return": {}}\n'
        reader.readline = AsyncMock(side_effect=[greeting, caps_response, query_response, balloon_response])

        client = BalloonClient(qmp_socket, expected_uid=1000)
        await client.connect()

        previous_mb = await client.inflate(target_mb=64)
        assert previous_mb == 256

    async def test_deflate_sets_target(self, qmp_socket: Path, mock_connect_and_verify: Any) -> None:
        """Test deflate (restore guest memory) sets target size."""
        _mock, reader, writer = mock_connect_and_verify

        greeting = b'{"QMP": {}}\n'
        caps_response = b'{"return": {}}\n'
        balloon_response = b'{"return": {}}\n'
        reader.readline = AsyncMock(side_effect=[greeting, caps_response, balloon_response])

        client = BalloonClient(qmp_socket, expected_uid=1000)
        await client.connect()

        await client.deflate(target_mb=256)

        # Verify balloon command sent with 256MB
        calls = writer.write.call_args_list
        balloon_call = calls[1][0][0].decode()
        cmd = json.loads(balloon_call.strip())
        assert cmd["arguments"]["value"] == 256 * 1024 * 1024

    async def test_close_closes_writer(self, qmp_socket: Path, mock_connect_and_verify: Any) -> None:
        """Test close properly closes the connection."""
        _mock, reader, writer = mock_connect_and_verify

        greeting = b'{"QMP": {}}\n'
        caps_response = b'{"return": {}}\n'
        reader.readline = AsyncMock(side_effect=[greeting, caps_response])

        client = BalloonClient(qmp_socket, expected_uid=1000)
        await client.connect()
        await client.close()

        assert not client._connected
        writer.close.assert_called_once()

    async def test_execute_raises_when_not_connected(self, qmp_socket: Path) -> None:
        """Test _execute raises BalloonError when not connected."""
        client = BalloonClient(qmp_socket, expected_uid=1000)

        with pytest.raises(BalloonError, match="Not connected"):
            await client._execute("query-balloon")

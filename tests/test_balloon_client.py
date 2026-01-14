"""Unit tests for BalloonClient QMP interface.

Tests the balloon memory control client without requiring actual VMs.
Uses mocked QMP socket responses.
"""

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
            writer = AsyncMock()
            # Configure writer.transport.get_extra_info
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
        _mock, reader, _writer = mock_connect_and_verify

        greeting = b'{"QMP": {"version": {"qemu": {"micro": 0, "minor": 0, "major": 10}}}}\n'
        error_response = b'{"error": {"class": "CommandNotFound", "desc": "Unknown command"}}\n'
        reader.readline = AsyncMock(side_effect=[greeting, error_response])

        client = BalloonClient(qmp_socket, expected_uid=1000)
        with pytest.raises(BalloonError, match="QMP capabilities failed"):
            await client.connect()

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

    async def test_disconnect_closes_writer(self, qmp_socket: Path, mock_connect_and_verify: Any) -> None:
        """Test disconnect properly closes the connection."""
        _mock, reader, writer = mock_connect_and_verify

        greeting = b'{"QMP": {}}\n'
        caps_response = b'{"return": {}}\n'
        reader.readline = AsyncMock(side_effect=[greeting, caps_response])

        client = BalloonClient(qmp_socket, expected_uid=1000)
        await client.connect()
        await client.disconnect()

        assert not client._connected
        writer.close.assert_called_once()

    async def test_execute_raises_when_not_connected(self, qmp_socket: Path) -> None:
        """Test _execute raises BalloonError when not connected."""
        client = BalloonClient(qmp_socket, expected_uid=1000)

        with pytest.raises(BalloonError, match="Not connected"):
            await client._execute("query-balloon")


class TestBalloonClientConstants:
    """Tests for balloon-related constants."""

    def test_balloon_inflate_target_reasonable(self) -> None:
        """BALLOON_INFLATE_TARGET_MB should be reasonable for idle VMs."""
        from exec_sandbox import constants

        # Should be at least 32MB for kernel overhead
        assert constants.BALLOON_INFLATE_TARGET_MB >= 32
        # Should be less than default memory (otherwise no benefit)
        assert constants.BALLOON_INFLATE_TARGET_MB < constants.DEFAULT_MEMORY_MB

    def test_balloon_timeouts_reasonable(self) -> None:
        """Balloon timeouts should be reasonable."""
        from exec_sandbox import constants

        # At least 1 second
        assert constants.BALLOON_INFLATE_TIMEOUT_SECONDS >= 1.0
        assert constants.BALLOON_DEFLATE_TIMEOUT_SECONDS >= 1.0
        # Not too long (would block warm pool operations)
        assert constants.BALLOON_INFLATE_TIMEOUT_SECONDS <= 30.0
        assert constants.BALLOON_DEFLATE_TIMEOUT_SECONDS <= 30.0

    def test_package_version_format(self) -> None:
        """__version__ should be valid semver format (used for cache key)."""
        from exec_sandbox import __version__

        parts = __version__.split(".")
        assert len(parts) >= 2, f"Expected at least major.minor, got {__version__}"
        # Major and minor should be numeric
        assert parts[0].isdigit(), f"Invalid major version '{parts[0]}' in {__version__}"
        assert parts[1].isdigit(), f"Invalid minor version '{parts[1]}' in {__version__}"

"""Tests for MigrationClient (QMP migration protocol).

De-mocked: uses real asyncio.StreamReader + FakeWriter instead of MagicMock.
Tests exercise the full _execute protocol stack (JSON serialize → write →
readline → event skipping → JSON parse).  Only connect/close remain mocked
(socket boundary).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from exec_sandbox import constants
from exec_sandbox.exceptions import MigrationTransientError
from exec_sandbox.migration_client import MigrationClient

# ============================================================================
# Test Helpers
# ============================================================================


class FakeWriter:
    """Drop-in for asyncio.StreamWriter.  Captures written data, rejects undefined methods."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass

    @property
    def commands(self) -> list[dict[str, Any]]:
        """Parse all QMP commands written."""
        return [json.loads(line) for line in self.data.decode().strip().split("\n") if line]


def _make_connected(
    responses: list[dict[str, Any]] | None = None,
    *,
    feed_eof: bool = False,
) -> tuple[MigrationClient, asyncio.StreamReader, FakeWriter]:
    """Create a MigrationClient with real StreamReader + FakeWriter already connected."""
    client = MigrationClient(Path("/tmp/fake.sock"), 501)
    reader = asyncio.StreamReader()
    writer = FakeWriter()
    if responses:
        for resp in responses:
            reader.feed_data(json.dumps(resp).encode() + b"\n")
    if feed_eof:
        reader.feed_eof()
    object.__setattr__(client, "_connected", True)
    object.__setattr__(client, "_reader", reader)
    object.__setattr__(client, "_writer", writer)
    return client, reader, writer


# ============================================================================
# Init
# ============================================================================


class TestMigrationClientInit:
    """Tests for MigrationClient initialization."""

    def test_slots_defined(self) -> None:
        """MigrationClient uses __slots__ for memory efficiency."""
        assert hasattr(MigrationClient, "__slots__")
        assert "_connected" in MigrationClient.__slots__
        assert "_lock" in MigrationClient.__slots__

    def test_init_not_connected(self) -> None:
        """New client is not connected."""
        client = MigrationClient(Path("/tmp/fake.sock"), 501)
        assert client._connected is False
        assert client._reader is None
        assert client._writer is None


# ============================================================================
# _execute — real protocol stack
# ============================================================================


class TestMigrationClientExecute:
    """Tests for _execute (QMP command sending with event skipping)."""

    async def test_not_connected_raises(self) -> None:
        """Should raise when not connected."""
        client = MigrationClient(Path("/tmp/fake.sock"), 501)
        with pytest.raises(MigrationTransientError, match="Not connected"):
            await client._execute("test-command")

    async def test_skips_events_to_find_response(self) -> None:
        """Should skip QMP event messages and return the command response."""
        client, _reader, _writer = _make_connected(
            [
                {"event": "MIGRATION", "data": {}},
                {"event": "STOP", "data": {}},
                {"return": {"status": "completed"}},
            ]
        )

        result = await client._execute("query-migrate")
        assert result == {"return": {"status": "completed"}}

    async def test_connection_closed_raises(self) -> None:
        """Should raise when connection is closed (empty readline)."""
        client, _reader, _writer = _make_connected(feed_eof=True)

        with pytest.raises(MigrationTransientError, match="closed unexpectedly"):
            await client._execute("test-command")

    async def test_execute_with_arguments(self) -> None:
        """Execute sends arguments in the JSON command."""
        client, _reader, writer = _make_connected([{"return": {}}])

        await client._execute("migrate", arguments={"uri": "file:/tmp/test.vmstate"})

        cmds = writer.commands
        assert len(cmds) == 1
        assert cmds[0]["execute"] == "migrate"
        assert cmds[0]["arguments"]["uri"] == "file:/tmp/test.vmstate"

    async def test_execute_without_arguments(self) -> None:
        """Execute without arguments doesn't include 'arguments' key."""
        client, _reader, writer = _make_connected([{"return": {}}])

        await client._execute("stop")

        cmds = writer.commands
        assert cmds[0] == {"execute": "stop"}
        assert "arguments" not in cmds[0]

    async def test_error_response_returned(self) -> None:
        """Error response from QEMU is returned (not raised) — caller decides."""
        error_resp = {"error": {"class": "GenericError", "desc": "device not found"}}
        client, _reader, _writer = _make_connected([error_resp])

        result = await client._execute("invalid-command")
        assert "error" in result
        assert result["error"]["desc"] == "device not found"


# ============================================================================
# _execute — edge cases
# ============================================================================


class TestMigrationClientExecuteEdgeCases:
    """Edge cases for _execute (malformed JSON, event floods, etc.)."""

    async def test_malformed_json_response_raises(self) -> None:
        """Malformed JSON from QEMU should raise."""
        client, reader, _writer = _make_connected()
        reader.feed_data(b"not valid json\n")

        with pytest.raises(json.JSONDecodeError):
            await client._execute("test-command")

    async def test_event_flood_hits_safety_cap(self) -> None:
        """32+ consecutive events without a response should raise."""
        events = [{"event": f"EVENT_{i}", "data": {}} for i in range(33)]
        client, _reader, _writer = _make_connected(events)

        with pytest.raises(MigrationTransientError, match="Too many QMP events"):
            await client._execute("query-migrate")

    async def test_exactly_32_events_then_response(self) -> None:
        """31 events + 1 response (within 32 iterations) should succeed."""
        responses: list[dict[str, Any]] = [{"event": f"EVENT_{i}", "data": {}} for i in range(31)]
        responses.append({"return": {"status": "ok"}})
        client, _reader, _writer = _make_connected(responses)

        result = await client._execute("query-migrate")
        assert result == {"return": {"status": "ok"}}


# ============================================================================
# _set_capabilities — real _execute with pre-loaded responses
# ============================================================================


class TestMigrationClientSetCapabilities:
    """Tests for _set_capabilities (mapped-ram + multifd on QEMU >= 9.0)."""

    async def test_enables_mapped_ram_for_qemu_9(self) -> None:
        """QEMU >= 9.0 should enable mapped-ram + multifd."""
        client, _reader, writer = _make_connected([{"return": {}}])

        await client._set_capabilities((9, 2, 0))

        cmds = writer.commands
        assert len(cmds) == 1
        assert cmds[0]["execute"] == "migrate-set-capabilities"
        cap_names = {c["capability"] for c in cmds[0]["arguments"]["capabilities"]}
        assert "multifd" in cap_names
        assert "mapped-ram" in cap_names

    async def test_no_capabilities_for_old_qemu(self) -> None:
        """QEMU < 9.0 should not set capabilities."""
        client, _reader, writer = _make_connected()

        await client._set_capabilities((8, 2, 0))

        assert len(writer.commands) == 0

    async def test_no_capabilities_for_none_version(self) -> None:
        """None version should not set capabilities."""
        client, _reader, writer = _make_connected()

        await client._set_capabilities(None)

        assert len(writer.commands) == 0

    async def test_capability_error_is_non_fatal(self) -> None:
        """Capability setting failure should not raise (falls back to streaming)."""
        client, _reader, _writer = _make_connected(
            [
                {"error": {"class": "GenericError", "desc": "mapped-ram not supported"}},
            ]
        )

        # Should not raise
        await client._set_capabilities((9, 2, 0))


class TestMigrationClientSetCapabilitiesEdgeCases:
    """Edge cases for capability negotiation."""

    async def test_boundary_qemu_version_9_0_0(self) -> None:
        """Exactly QEMU 9.0.0 should enable capabilities (>= check)."""
        client, _reader, writer = _make_connected([{"return": {}}])

        await client._set_capabilities((9, 0, 0))

        assert any(c["execute"] == "migrate-set-capabilities" for c in writer.commands)

    async def test_boundary_qemu_version_8_99_99(self) -> None:
        """QEMU 8.99.99 should NOT enable capabilities (< 9.0.0)."""
        client, _reader, writer = _make_connected()

        await client._set_capabilities((8, 99, 99))

        assert len(writer.commands) == 0

    async def test_very_high_qemu_version(self) -> None:
        """Future QEMU version (20.0.0) should enable capabilities."""
        client, _reader, writer = _make_connected([{"return": {}}])

        await client._set_capabilities((20, 0, 0))

        assert any(c["execute"] == "migrate-set-capabilities" for c in writer.commands)


# ============================================================================
# _poll_migration — real _execute with pre-loaded responses
# ============================================================================


class TestMigrationClientPollMigration:
    """Tests for _poll_migration (query-migrate polling loop)."""

    async def test_poll_until_completed(self) -> None:
        """Poll should return when status is 'completed'."""
        client, _reader, _writer = _make_connected(
            [
                {"return": {"status": "active"}},
                {"return": {"status": "active"}},
                {"return": {"status": "completed"}},
            ]
        )

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            await client._poll_migration(timeout=5.0)

    async def test_poll_raises_on_failure(self) -> None:
        """Poll should raise MigrationTransientError on 'failed' status."""
        client, _reader, _writer = _make_connected(
            [
                {"return": {"status": "active"}},
                {"return": {"status": "failed", "error-desc": "disk full"}},
            ]
        )

        with (
            patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001),
            pytest.raises(MigrationTransientError, match="disk full"),
        ):
            await client._poll_migration(timeout=5.0)

    async def test_poll_raises_on_cancelled(self) -> None:
        """Poll should raise MigrationTransientError on 'cancelled' status."""
        client, _reader, _writer = _make_connected(
            [
                {"return": {"status": "cancelled"}},
            ]
        )

        with (
            patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001),
            pytest.raises(MigrationTransientError, match="cancelled"),
        ):
            await client._poll_migration(timeout=5.0)

    async def test_poll_raises_on_query_error(self) -> None:
        """Poll should raise MigrationTransientError on query-migrate error response."""
        client, _reader, _writer = _make_connected(
            [
                {"error": {"class": "GenericError", "desc": "query failed"}},
            ]
        )

        with (
            patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001),
            pytest.raises(MigrationTransientError, match="query-migrate failed"),
        ):
            await client._poll_migration(timeout=5.0)

    async def test_poll_timeout(self) -> None:
        """Poll should raise TimeoutError when migration hangs."""
        # Feed many "active" responses so the poll loop doesn't exhaust them
        active_responses = [{"return": {"status": "active"}} for _ in range(500)]
        client, _reader, _writer = _make_connected(active_responses)

        with (
            patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001),
            pytest.raises(TimeoutError),
        ):
            await client._poll_migration(timeout=0.05)


class TestMigrationClientPollEdgeCases:
    """Edge cases for _poll_migration."""

    async def test_immediate_completion(self) -> None:
        """First poll returns completed → immediate return."""
        client, _reader, _writer = _make_connected(
            [
                {"return": {"status": "completed"}},
            ]
        )

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            await client._poll_migration(timeout=5.0)

    async def test_unknown_status_keeps_polling(self) -> None:
        """Unknown migration status keeps polling (until completed/failed/timeout)."""
        client, _reader, _writer = _make_connected(
            [
                {"return": {"status": "setup"}},
                {"return": {"status": "postcopy-active"}},
                {"return": {"status": "completed"}},
            ]
        )

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            await client._poll_migration(timeout=5.0)

    async def test_missing_status_field_keeps_polling(self) -> None:
        """Response missing 'status' field → empty string → keeps polling."""
        client, _reader, _writer = _make_connected(
            [
                {"return": {}},  # No status field
                {"return": {"status": "completed"}},
            ]
        )

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            await client._poll_migration(timeout=5.0)


# ============================================================================
# save_snapshot — full protocol through real _execute
# ============================================================================


class TestMigrationClientSaveSnapshot:
    """Tests for save_snapshot (stop → migrate → poll → quit sequence)."""

    async def test_save_command_sequence(self) -> None:
        """Save should: set-capabilities → stop → migrate → poll → quit."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # quit
            ]
        )

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            await client.save_snapshot(Path("/tmp/test.vmstate"), qemu_version=(9, 2, 0))

        cmds = [c["execute"] for c in writer.commands]
        assert cmds == ["migrate-set-capabilities", "stop", "migrate", "query-migrate", "quit"]

    async def test_save_command_sequence_old_qemu(self) -> None:
        """Save with QEMU < 9.0 should: stop → migrate → poll → quit (no capabilities)."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # quit
            ]
        )

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            await client.save_snapshot(Path("/tmp/test.vmstate"), qemu_version=(8, 2, 0))

        cmds = [c["execute"] for c in writer.commands]
        assert cmds == ["stop", "migrate", "query-migrate", "quit"]

    async def test_save_migrate_error(self) -> None:
        """Save should raise on migrate command error."""
        client, _reader, _writer = _make_connected(
            [
                {"return": {}},  # stop
                {"error": {"class": "GenericError", "desc": "cannot migrate"}},  # migrate
            ]
        )

        with pytest.raises(MigrationTransientError, match="migrate command failed"):
            await client.save_snapshot(Path("/tmp/test.vmstate"), qemu_version=(8, 0, 0))

    async def test_save_migrate_uri_format(self) -> None:
        """migrate command uses file: URI format."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # quit
            ]
        )

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            await client.save_snapshot(Path("/tmp/my-snapshot.vmstate"), qemu_version=(8, 0, 0))

        migrate_cmd = next(c for c in writer.commands if c["execute"] == "migrate")
        assert migrate_cmd["arguments"]["uri"] == "file:/tmp/my-snapshot.vmstate"


class TestMigrationClientSaveEdgeCases:
    """Edge cases for save_snapshot."""

    async def test_save_stop_error_raises(self) -> None:
        """stop command error should raise MigrationTransientError."""
        client, _reader, _writer = _make_connected(
            [
                {"error": {"class": "GenericError", "desc": "already stopped"}},  # stop
            ]
        )

        with pytest.raises(MigrationTransientError, match="stop command failed"):
            await client.save_snapshot(Path("/tmp/test.vmstate"), qemu_version=(8, 0, 0))

    async def test_save_qemu_exits_early_with_vmstate(self, tmp_path: Path) -> None:
        """QEMU exits during poll but vmstate exists → treat as success."""
        vmstate_path = tmp_path / "test.vmstate"
        vmstate_path.write_bytes(b"migration stream data")

        # stop + migrate succeed, then EOF on query-migrate (QEMU exited)
        client, reader, _writer = _make_connected(
            [
                {"return": {}},  # stop
                {"return": {}},  # migrate
            ]
        )
        reader.feed_eof()  # QEMU exits — readline returns b""

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            # Should NOT raise — vmstate file exists
            await client.save_snapshot(vmstate_path, qemu_version=(8, 0, 0))

    async def test_save_qemu_exits_early_no_vmstate_raises(self, tmp_path: Path) -> None:
        """QEMU exits during poll and no vmstate → raise error."""
        vmstate_path = tmp_path / "missing.vmstate"
        # File does NOT exist

        # stop + migrate succeed, then EOF on query-migrate (QEMU exited)
        client, reader, _writer = _make_connected(
            [
                {"return": {}},  # stop
                {"return": {}},  # migrate
            ]
        )
        reader.feed_eof()  # QEMU exits

        with (
            patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001),
            pytest.raises(MigrationTransientError, match="Connection lost during save"),
        ):
            await client.save_snapshot(vmstate_path, qemu_version=(8, 0, 0))

    async def test_save_quit_error_suppressed(self) -> None:
        """quit command error after save is suppressed (VM already exiting)."""
        # stop + migrate + query-migrate succeed, then EOF on quit (QEMU already gone)
        client, reader, writer = _make_connected(
            [
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
            ]
        )
        reader.feed_eof()  # QEMU exits before quit response

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            # Should NOT raise — quit error is suppressed
            await client.save_snapshot(Path("/tmp/test.vmstate"), qemu_version=(8, 0, 0))

        cmds = [c["execute"] for c in writer.commands]
        assert "quit" in cmds


# ============================================================================
# restore_snapshot — full protocol through real _execute
# ============================================================================


class TestMigrationClientRestoreSnapshot:
    """Tests for restore_snapshot (migrate-incoming → poll → cont sequence)."""

    async def test_restore_command_sequence(self) -> None:
        """Restore should: set-capabilities → migrate-incoming → poll → cont."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-incoming
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # cont
            ]
        )

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            await client.restore_snapshot(Path("/tmp/test.vmstate"), qemu_version=(9, 2, 0))

        cmds = [c["execute"] for c in writer.commands]
        assert cmds == ["migrate-set-capabilities", "migrate-incoming", "query-migrate", "cont"]

    async def test_restore_does_not_quit(self) -> None:
        """Restore should NOT send 'quit' (save sends quit, restore doesn't)."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-incoming
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # cont
            ]
        )

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            await client.restore_snapshot(Path("/tmp/test.vmstate"), qemu_version=(9, 2, 0))

        cmds = [c["execute"] for c in writer.commands]
        assert "quit" not in cmds

    async def test_restore_cont_error(self) -> None:
        """Restore should raise on 'cont' command error."""
        client, _reader, _writer = _make_connected(
            [
                {"return": {}},  # migrate-incoming
                {"return": {"status": "completed"}},  # query-migrate
                {"error": {"class": "GenericError", "desc": "cannot resume"}},  # cont
            ]
        )

        with (
            patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001),
            pytest.raises(MigrationTransientError, match="cont command failed"),
        ):
            await client.restore_snapshot(Path("/tmp/test.vmstate"), qemu_version=(8, 0, 0))


class TestMigrationClientRestoreEdgeCases:
    """Edge cases for restore_snapshot."""

    async def test_restore_migrate_incoming_error(self) -> None:
        """migrate-incoming error should raise."""
        client, _reader, _writer = _make_connected(
            [
                {"error": {"class": "GenericError", "desc": "file not found"}},  # migrate-incoming
            ]
        )

        with pytest.raises(MigrationTransientError, match="migrate-incoming command failed"):
            await client.restore_snapshot(Path("/tmp/test.vmstate"), qemu_version=(8, 0, 0))

    async def test_restore_migration_fails_during_poll(self) -> None:
        """Migration failure during restore poll should raise with error-desc."""
        client, _reader, _writer = _make_connected(
            [
                {"return": {}},  # migrate-incoming
                {"return": {"status": "active"}},  # query-migrate (1)
                {"return": {"status": "failed", "error-desc": "incompatible migration stream"}},
            ]
        )

        with (
            patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001),
            pytest.raises(MigrationTransientError, match="incompatible migration stream"),
        ):
            await client.restore_snapshot(Path("/tmp/test.vmstate"), qemu_version=(8, 0, 0))

    async def test_restore_uri_format(self) -> None:
        """migrate-incoming command uses file: URI format."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-incoming
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # cont
            ]
        )

        with patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001):
            await client.restore_snapshot(Path("/tmp/snapshot.vmstate"), qemu_version=(8, 0, 0))

        incoming_cmd = next(c for c in writer.commands if c["execute"] == "migrate-incoming")
        assert incoming_cmd["arguments"]["uri"] == "file:/tmp/snapshot.vmstate"


# ============================================================================
# Context Manager
# ============================================================================


class TestMigrationClientContextManager:
    """Tests for async context manager protocol."""

    async def test_context_manager_connects_and_closes(self) -> None:
        """__aenter__ calls connect(), __aexit__ calls close()."""
        connect_called = False
        close_called = False

        async def mock_connect(self: MigrationClient, timeout: float = 5.0) -> None:
            nonlocal connect_called
            connect_called = True

        async def mock_close(self: MigrationClient) -> None:
            nonlocal close_called
            close_called = True

        with (
            patch.object(MigrationClient, "connect", mock_connect),
            patch.object(MigrationClient, "close", mock_close),
        ):
            client = MigrationClient(Path("/tmp/fake.sock"), 501)
            async with client:
                assert connect_called

        assert close_called

    async def test_close_called_on_exception(self) -> None:
        """__aexit__ calls close() even when body raises."""
        close_called = False

        async def mock_connect(self: MigrationClient, timeout: float = 5.0) -> None:
            pass

        async def mock_close(self: MigrationClient) -> None:
            nonlocal close_called
            close_called = True

        with (
            patch.object(MigrationClient, "connect", mock_connect),
            patch.object(MigrationClient, "close", mock_close),
        ):
            client = MigrationClient(Path("/tmp/fake.sock"), 501)
            with pytest.raises(RuntimeError, match="body error"):
                async with client:
                    raise RuntimeError("body error")

        assert close_called

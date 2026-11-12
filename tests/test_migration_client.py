"""Tests for MigrationClient (QMP migration protocol).

De-mocked: uses real asyncio.StreamReader + FakeWriter instead of MagicMock.
Tests exercise the full _execute protocol stack (JSON serialize → write →
readline → event skipping → JSON parse).  Only connect/close remain mocked
(socket boundary).
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox import constants
from exec_sandbox.exceptions import ConfigError, ErrorCode, RouteError, SandboxError, TransientError
from exec_sandbox.migration_client import MigrationClient, migration_timeout, rejected_by_qemu
from exec_sandbox.qmp_exchange import _MAX_SKIPPED_MESSAGES
from tests.instance_support import FakeQmp

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

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
        # QMP echoes the request id on every command response. _execute tags
        # commands migration-1, migration-2, ... in order, so mirror that here:
        # each non-event response carries the next sequential id (events don't
        # consume an id). Explicitly-set ids are preserved.
        seq = 0
        for resp in responses:
            tagged = dict(resp)
            if "event" not in tagged and "id" not in tagged:
                seq += 1
                tagged["id"] = f"migration-{seq}"
            reader.feed_data(json.dumps(tagged).encode() + b"\n")
    if feed_eof:
        reader.feed_eof()
    object.__setattr__(client, "_connected", True)
    object.__setattr__(client, "_reader", reader)
    object.__setattr__(client, "_writer", writer)
    return client, reader, writer


_QEMU = (11, 2, 0)
_QMP_TIMEOUT = constants.MEMORY_SNAPSHOT_QMP_TIMEOUT_SECONDS


def _fast_poll():
    return patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001)


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


class TestMigrationClientConnectOwnership:
    """Failed QMP handshakes close their published stream owner."""

    @staticmethod
    def _streams() -> tuple[AsyncMock, MagicMock]:
        reader = AsyncMock()
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    async def test_capabilities_error_closes_writer(self) -> None:
        reader, writer = self._streams()
        reader.readline = AsyncMock(
            side_effect=[
                b'{"QMP": {}}\n',
                b'{"error": {"class": "CommandNotFound"}}\n',
            ]
        )
        client = MigrationClient(Path("/tmp/fake.sock"), 501)

        with patch(
            "exec_sandbox.migration_client.connect_and_verify",
            new_callable=AsyncMock,
            return_value=(reader, writer),
        ):
            with pytest.raises(
                TransientError, match="QMP capabilities failed", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
            ):
                await client.connect(timeout=5.0)

        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()
        assert client._writer is None

    async def test_cancellation_waits_for_handshake_cleanup(self) -> None:
        reader, writer = self._streams()
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
        client = MigrationClient(Path("/tmp/fake.sock"), 501)

        with patch(
            "exec_sandbox.migration_client.connect_and_verify",
            new_callable=AsyncMock,
            return_value=(reader, writer),
        ):
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


# ============================================================================
# _execute — real protocol stack
# ============================================================================


class TestMigrationClientExecute:
    """Tests for _execute (QMP command sending with event skipping)."""

    async def test_not_connected_raises(self) -> None:
        """Should raise when not connected."""
        client = MigrationClient(Path("/tmp/fake.sock"), 501)
        with pytest.raises(TransientError, match="Not connected", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED):
            await client._execute("test-command", arguments=None, timeout=_QMP_TIMEOUT)

    async def test_skips_events_to_find_response(self) -> None:
        """Should skip QMP event messages and return the command response."""
        client, _reader, _writer = _make_connected(
            [
                {"event": "MIGRATION", "data": {}},
                {"event": "STOP", "data": {}},
                {"return": {"status": "completed"}},
            ]
        )

        result = await client._execute("query-migrate", arguments=None, timeout=_QMP_TIMEOUT)
        assert result["return"] == {"status": "completed"}

    async def test_connection_closed_raises(self) -> None:
        """Should raise when connection is closed (empty readline)."""
        client, _reader, _writer = _make_connected(feed_eof=True)

        with pytest.raises(
            TransientError, match="closed unexpectedly", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
        ):
            await client._execute("test-command", arguments=None, timeout=_QMP_TIMEOUT)

    async def test_execute_with_arguments(self) -> None:
        """Execute sends arguments in the JSON command."""
        client, _reader, writer = _make_connected([{"return": {}}])

        await client._execute("migrate", arguments={"uri": "file:/tmp/test.vmstate"}, timeout=_QMP_TIMEOUT)

        cmds = writer.commands
        assert len(cmds) == 1
        assert cmds[0]["execute"] == "migrate"
        assert cmds[0]["arguments"]["uri"] == "file:/tmp/test.vmstate"

    async def test_execute_without_arguments(self) -> None:
        """Execute without arguments doesn't include 'arguments' key."""
        client, _reader, writer = _make_connected([{"return": {}}])

        await client._execute("stop")

        cmds = writer.commands
        assert cmds[0]["execute"] == "stop"
        assert cmds[0]["id"] == "migration-1"
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
        """More skipped messages than the safety cap should raise."""
        events = [{"event": f"EVENT_{i}", "data": {}} for i in range(_MAX_SKIPPED_MESSAGES + 1)]
        client, _reader, _writer = _make_connected(events)

        with pytest.raises(MigrationTransientError, match="Too many QMP messages"):
            await client._execute("query-migrate")

    async def test_events_just_under_cap_then_response(self) -> None:
        """cap-1 events + 1 response (within the cap) should succeed."""
        responses: list[dict[str, Any]] = [
            {"event": f"EVENT_{i}", "data": {}} for i in range(_MAX_SKIPPED_MESSAGES - 1)
        ]
        responses.append({"return": {"status": "ok"}})
        client, _reader, _writer = _make_connected(responses)

        result = await client._execute("query-migrate", arguments=None, timeout=_QMP_TIMEOUT)
        assert result["return"] == {"status": "ok"}


# ============================================================================
# Capabilities and progress, through save_snapshot and restore_snapshot
# ============================================================================

_OK: dict[str, Any] = {"return": {}}
_FAILED_QUERY: dict[str, Any] = {"error": {"class": "GenericError", "desc": "query failed"}}


def _query(status: str | None) -> dict[str, Any]:
    """A query-migrate answer (None: an answer without a status)."""
    return {"return": {"status": status}} if status is not None else {"return": {}}


async def _run(
    client: MigrationClient,
    operation: str,
    *,
    qemu_version: tuple[int, int, int] | None,
    use_template: bool,
    timeout: float,
) -> None:
    """A save (QEMU quits after it) or a restore of /tmp/t.vmstate."""
    if operation == "save":
        await client.save_snapshot(
            Path("/tmp/t.vmstate"),
            qemu_version=qemu_version,
            timeout=timeout,
            use_template=use_template,
            keep_running=False,
            while_paused=None,
        )
    else:
        await client.restore_snapshot(
            Path("/tmp/t.vmstate"), qemu_version=qemu_version, timeout=timeout, use_template=use_template
        )


def _replies(operation: str, *, queries: list[dict[str, Any]], after_failure: bool) -> list[dict[str, Any]]:
    """What QEMU answers: capabilities, the start of the migration, the query-migrate answers,
    then a save's quit or a restore's cont, or (`after_failure`) a save's migrate_cancel and cont."""
    start = [_OK, _OK, _OK] if operation == "save" else [_OK, _OK]  # save: stop, migrate
    if after_failure:
        return [*start, *queries, *([_OK, _OK] if operation == "save" else [])]
    return [*start, *queries, _OK]


class TestMigrationCapabilities:
    """Both directions set the same explicit capabilities first; no legacy fallback."""

    @pytest.mark.parametrize("operation", ["save", "restore"])
    @pytest.mark.parametrize("use_template", [False, True])
    @pytest.mark.parametrize("version", [(11, 0, 0), (20, 0, 0)], ids=["floor", "newer"])
    async def test_explicit_capability_list(
        self, operation: str, use_template: bool, version: tuple[int, int, int]
    ) -> None:
        """mapped-ram + multifd on, x-ignore-shared always sent with an explicit state."""
        client, _reader, writer = _make_connected(
            _replies(operation, queries=[_query("completed")], after_failure=False)
        )

        with _fast_poll():
            await _run(client, operation, qemu_version=version, use_template=use_template, timeout=5.0)

        first = writer.commands[0]
        assert first["execute"] == "migrate-set-capabilities"
        states = {c["capability"]: c["state"] for c in first["arguments"]["capabilities"]}
        assert states == {"multifd": True, "mapped-ram": True, "x-ignore-shared": use_template}

    @pytest.mark.parametrize("operation", ["save", "restore"])
    @pytest.mark.parametrize("version", [None, (10, 99, 99), (8, 2, 0)], ids=["unknown", "just-below", "old"])
    async def test_below_floor_sends_nothing(self, operation: str, version: tuple[int, int, int] | None) -> None:
        client, _reader, writer = _make_connected()

        with pytest.raises(
            ConfigError, match="below the migration floor", check=lambda e: e.code is ErrorCode.MIGRATION_UNSUPPORTED
        ) as exc:
            await _run(client, operation, qemu_version=version, use_template=False, timeout=5.0)

        assert writer.commands == []
        assert rejected_by_qemu(exc.value)  # an L1 entry this QEMU cannot load is evicted

    @pytest.mark.parametrize("operation", ["save", "restore"])
    async def test_refused_capability_is_rejected(self, operation: str) -> None:
        """A refused capability fails the operation (no silent legacy stream); a save had not
        stopped the VM yet, so it runs on."""
        client, _reader, writer = _make_connected(
            [{"error": {"class": "GenericError", "desc": "mapped-ram not supported"}}]
        )

        with pytest.raises(
            ConfigError, match="mapped-ram not supported", check=lambda e: e.code is ErrorCode.MIGRATION_UNSUPPORTED
        ) as exc:
            await _run(client, operation, qemu_version=_QEMU, use_template=False, timeout=5.0)

        assert [c["execute"] for c in writer.commands] == ["migrate-set-capabilities"]
        if operation == "save":
            assert exc.value.context["vm_resumed"] is True


class TestMigrationProgress:
    """query-migrate polling, seen through save_snapshot and restore_snapshot."""

    @pytest.mark.parametrize(
        "statuses",
        [
            pytest.param(["active", "active", "completed"], id="after-active"),
            pytest.param(["completed"], id="immediate"),
            pytest.param(["setup", "postcopy-active", "completed"], id="unknown-status"),
            pytest.param([None, "completed"], id="missing-status"),
        ],
    )
    async def test_polls_until_completed(self, statuses: list[str | None]) -> None:
        client, _reader, writer = _make_connected(
            _replies("restore", queries=[_query(st) for st in statuses], after_failure=False)
        )

        with _fast_poll():
            await _run(client, "restore", qemu_version=_QEMU, use_template=False, timeout=5.0)

        assert [c["execute"] for c in writer.commands] == [
            "migrate-set-capabilities",
            "migrate-incoming",
            *["query-migrate"] * len(statuses),  # polled until "completed", no further
            "cont",
        ]

    @pytest.mark.parametrize("status", ["failed", "cancelled", "failing"])
    @pytest.mark.parametrize(
        ("operation", "error"),
        [
            ("save", (TransientError, ErrorCode.MIGRATION_FAILED)),
            ("restore", (RouteError, ErrorCode.MIGRATION_REJECTED)),
        ],
    )
    async def test_ended_migration(
        self, status: str, operation: str, error: tuple[type[SandboxError], ErrorCode]
    ) -> None:
        """QEMU ends a save: transient, the VM resumes and can be kept. QEMU ends a restore: it
        refused the saved state (ROUTE), and the VM is never resumed. QEMU's reason stays."""
        queries = [_query("active"), {"return": {"status": status, "error-desc": "disk full"}}]
        client, _reader, writer = _make_connected(_replies(operation, queries=queries, after_failure=True))

        with (
            _fast_poll(),
            pytest.raises(error[0], match=f"{status}: disk full", check=lambda e: e.code is error[1]) as exc,
        ):
            await _run(client, operation, qemu_version=_QEMU, use_template=False, timeout=5.0)

        sent = [c["execute"] for c in writer.commands]
        if operation == "save":
            assert sent[-2:] == ["migrate_cancel", "cont"]
            assert exc.value.context["vm_resumed"] is True
            assert not rejected_by_qemu(exc.value)
        else:
            assert sent[-1] == "query-migrate"
            assert isinstance(exc.value, RouteError)
            assert exc.value.required == {"qemu_version": "a QEMU that accepts this saved state"}
            assert exc.value.found == {"qemu_version": "11.2.0"}
            assert rejected_by_qemu(exc.value)

    @pytest.mark.parametrize("operation", ["save", "restore"])
    async def test_failed_query_is_transient(self, operation: str) -> None:
        """query-migrate itself failing says nothing about the saved state: transient, also for a restore."""
        client, _reader, _writer = _make_connected(_replies(operation, queries=[_FAILED_QUERY], after_failure=True))

        with (
            _fast_poll(),
            pytest.raises(
                TransientError, match="query-migrate failed", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
            ) as exc,
        ):
            await _run(client, operation, qemu_version=_QEMU, use_template=False, timeout=5.0)
        assert not rejected_by_qemu(exc.value)  # a good L1 entry is kept

    @pytest.mark.parametrize("operation", ["save", "restore"])
    async def test_hung_migration_is_transient(self, operation: str) -> None:
        """A migration that never ends becomes TransientError MIGRATION_FAILED, not a bare TimeoutError."""
        client, _reader, _writer = _make_connected(
            _replies(operation, queries=[_query("active")] * 500, after_failure=True)
        )

        with (
            _fast_poll(),
            pytest.raises(
                TransientError, match="did not complete", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
            ),
        ):
            await _run(client, operation, qemu_version=_QEMU, use_template=False, timeout=0.05)


class TestRejectedByQemu:
    """rejected_by_qemu decides the L1 eviction and the reuse() ROUTE rule. It matches by code
    (MIGRATION_REJECTED, MIGRATION_UNSUPPORTED), never by class, anywhere in the cause or
    context chain: a restore whose setup fails (HOST_DEPENDENCY, also a ConfigError) or whose
    QEMU fails otherwise must not evict a good entry."""

    @staticmethod
    def _rejected() -> SandboxError:
        return RouteError(
            "migrate-incoming command failed",
            code=ErrorCode.MIGRATION_REJECTED,
            required={"qemu_version": "a QEMU that accepts this saved state"},
            found={"qemu_version": "11.2.0"},
            context=None,
            hint=None,
            instance_id=None,
        )

    @staticmethod
    def _unsupported() -> SandboxError:
        return ConfigError(
            "QEMU 10.2.0 is below the migration floor 11.0.0",
            code=ErrorCode.MIGRATION_UNSUPPORTED,
            context=None,
            hint=None,
            instance_id=None,
        )

    @staticmethod
    def _host_dependency() -> SandboxError:
        return ConfigError(
            "qemu-vm user not found", code=ErrorCode.HOST_DEPENDENCY, context=None, hint=None, instance_id=None
        )

    @staticmethod
    def _failed() -> SandboxError:
        return TransientError(
            "query-migrate failed", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None
        )

    @staticmethod
    def _crashed() -> SandboxError:
        return TransientError("QEMU exited", code=ErrorCode.QEMU_CRASHED, context=None, hint=None, instance_id=None)

    @staticmethod
    def _chained(inner: BaseException, *, link: str) -> BaseException:
        """`inner` as raised, or under the restore's TransientError wrap: `raise ... from` (cause)
        or a raise inside the handler without `from` (context only, as _check_qemu_alive does)."""
        if link == "self":
            return inner
        outer = TransientError(
            "L1 restore failed", code=ErrorCode.MIGRATION_FAILED, context=None, hint=None, instance_id=None
        )
        try:
            try:
                raise inner
            except BaseException as e:
                if link == "cause":
                    raise outer from e
                raise outer  # noqa: B904 - the context-only link is the case under test
        except TransientError as raised:
            return raised

    @pytest.mark.parametrize("link", ["self", "cause", "context"])
    @pytest.mark.parametrize(
        ("make", "refused"),
        [
            ("_rejected", True),
            ("_unsupported", True),
            ("_host_dependency", False),
            ("_failed", False),
            ("_crashed", False),
        ],
    )
    def test_matches_only_refusal_codes(self, make: str, refused: bool, link: str) -> None:
        error = self._chained(getattr(self, make)(), link=link)
        assert rejected_by_qemu(error) is refused

    def test_refusal_below_a_plain_error_is_found(self) -> None:
        """The restore wrap, then an OSError (context), then QEMU's refusal (its cause)."""
        os_error = OSError("QMP socket closed")
        assert rejected_by_qemu(self._chained(os_error, link="context")) is False
        os_error.__cause__ = self._rejected()
        assert rejected_by_qemu(self._chained(os_error, link="context")) is True

    def test_cycle_in_the_chain_ends(self) -> None:
        a, b = self._failed(), self._crashed()
        a.__context__, b.__context__ = b, a
        assert rejected_by_qemu(a) is False


class TestMigrationTimeout:
    def test_scales_with_ram(self) -> None:
        base = constants.MIGRATION_TIMEOUT_BASE_SECONDS
        assert migration_timeout(0) == base
        assert migration_timeout(4096) > migration_timeout(256) > base


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

        with _fast_poll():
            await client.save_snapshot(
                Path("/tmp/my-snapshot.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=None,
                use_template=False,
                keep_running=False,
            )

        cmds = writer.commands
        assert [c["execute"] for c in cmds] == ["migrate-set-capabilities", "stop", "migrate", "query-migrate", "quit"]
        assert cmds[2]["arguments"]["uri"] == "file:/tmp/my-snapshot.vmstate"

    async def test_keep_running_resumes_instead_of_quit(self) -> None:
        """keep_running=True: cont after completed, never quit (VM can be saved again)."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # cont
            ]
        )

        with _fast_poll():
            await client.save_snapshot(
                Path("/tmp/t.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                keep_running=True,
                while_paused=None,
                use_template=False,
            )

        assert [c["execute"] for c in writer.commands][-2:] == ["query-migrate", "cont"]

    async def test_keep_running_cont_error_raises(self) -> None:
        client, _reader, _writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"error": {"class": "GenericError", "desc": "cannot resume"}},  # cont
            ]
        )

        with (
            _fast_poll(),
            pytest.raises(
                TransientError, match="cont after save failed", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
            ),
        ):
            await client.save_snapshot(
                Path("/tmp/t.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                keep_running=True,
                while_paused=None,
                use_template=False,
            )

    async def test_migrate_error_resumes_vm(self) -> None:
        """A failed save cancels the migration and resumes the stopped VM."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # stop
                {"error": {"class": "GenericError", "desc": "cannot migrate"}},  # migrate
                {"return": {}},  # migrate_cancel
                {"return": {}},  # cont
            ]
        )

        with pytest.raises(
            TransientError, match="migrate command failed", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
        ) as exc:
            await client.save_snapshot(
                Path("/tmp/t.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=None,
                use_template=False,
                keep_running=False,
            )

        assert [c["execute"] for c in writer.commands][-2:] == ["migrate_cancel", "cont"]
        assert exc.value.context["vm_resumed"] is True  # the caller may keep the VM

    async def test_refused_cont_after_failed_save_keeps_original_error(self) -> None:
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "failing", "error-desc": "write error"}},  # query-migrate
                {"return": {}},  # migrate_cancel
                {"error": {"class": "GenericError", "desc": "Migration is not finalized yet"}},  # cont
            ]
        )

        with (
            _fast_poll(),
            pytest.raises(
                TransientError, match="write error", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
            ) as exc,
        ):
            await client.save_snapshot(
                Path("/tmp/t.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=None,
                use_template=False,
                keep_running=False,
            )

        assert [c["execute"] for c in writer.commands][-1] == "cont"
        assert exc.value.context["vm_resumed"] is False  # the VM can stay paused: the caller replaces it

    @pytest.mark.parametrize("cont", ["accepted", "refused"])
    async def test_while_paused_failure_resumes_or_reports_a_paused_vm(self, cont: str) -> None:
        """A hook failure (e.g. the home copy hits ENOSPC): with the VM running again the
        hook's error stays; with cont refused, a typed error says the VM may stay paused."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # migrate_cancel
                {"return": {}} if cont == "accepted" else {"error": {"class": "GenericError", "desc": "busy"}},
            ]
        )

        async def copy_home() -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        expected = OSError if cont == "accepted" else TransientError
        with _fast_poll(), pytest.raises(expected) as exc:
            await client.save_snapshot(
                Path("/tmp/t.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=copy_home,
                use_template=False,
                keep_running=True,
            )
        assert [c["execute"] for c in writer.commands][-2:] == ["migrate_cancel", "cont"]  # resume tried
        if cont == "accepted":
            assert exc.value.errno == errno.ENOSPC  # the hook's own error
        else:
            assert isinstance(exc.value, TransientError)
            assert exc.value.code is ErrorCode.MIGRATION_FAILED
            assert exc.value.context["vm_resumed"] is False

    async def test_refused_stop_leaves_the_vm_running(self) -> None:
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"error": {"class": "GenericError", "desc": "cannot stop"}},  # stop
            ]
        )
        with pytest.raises(
            TransientError, match="stop command failed", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
        ) as exc:
            await client.save_snapshot(
                Path("/tmp/t.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=None,
                use_template=False,
                keep_running=True,
            )
        assert exc.value.context["vm_resumed"] is True
        assert "cont" not in [c["execute"] for c in writer.commands]  # nothing was stopped

    async def test_lost_connection_is_failure_even_with_file(self, tmp_path: Path) -> None:
        """QEMU gone before "completed" is a failure, even if the file has bytes
        (mapped-ram pre-sizes the file, so its size proves nothing)."""
        vmstate_path = tmp_path / "test.vmstate"
        vmstate_path.write_bytes(b"partial migration stream")
        client, reader, _writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # stop
                {"return": {}},  # migrate
            ]
        )
        reader.feed_eof()  # QEMU exits — readline returns b""

        with _fast_poll(), pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.MIGRATION_FAILED):
            await client.save_snapshot(
                vmstate_path, qemu_version=_QEMU, timeout=5.0, while_paused=None, use_template=False, keep_running=False
            )

    async def test_quit_error_suppressed(self) -> None:
        """quit after a completed save may lose the socket: not an error."""
        client, reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
            ]
        )
        reader.feed_eof()  # QEMU exits before quit response

        with _fast_poll():
            await client.save_snapshot(
                Path("/tmp/t.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=None,
                use_template=False,
                keep_running=False,
            )

        assert writer.commands[-1]["execute"] == "quit"


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

        with _fast_poll():
            await client.restore_snapshot(
                Path("/tmp/snapshot.vmstate"), qemu_version=_QEMU, timeout=5.0, use_template=False
            )

        cmds = writer.commands
        assert [c["execute"] for c in cmds] == ["migrate-set-capabilities", "migrate-incoming", "query-migrate", "cont"]
        # exit-on-error=false: QEMU stays up on a bad stream so the error can be read
        assert cmds[1]["arguments"] == {"uri": "file:/tmp/snapshot.vmstate", "exit-on-error": False}

    async def test_restore_cont_error(self) -> None:
        client, _reader, _writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
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

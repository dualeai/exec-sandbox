"""Tests for MigrationClient (QMP migration protocol).

De-mocked: uses real asyncio.StreamReader + FakeWriter instead of MagicMock.
Tests exercise the full protocol stack through save_snapshot and restore_snapshot
(JSON serialize → write → readline → event skipping → JSON parse). Only
connect/close remain mocked (socket boundary): _make_connected gives the client
its streams.
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
from exec_sandbox.migration_client import MigrationClient, _failed, migration_timeout, rejected_by_qemu
from exec_sandbox.qmp_client import QmpClient
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
    qmp = QmpClient(reader, writer, error_factory=lambda message: _failed(message, context=None), id_prefix="migration")  # type: ignore[arg-type]
    object.__setattr__(client, "_connected", True)
    object.__setattr__(client, "_qmp", qmp)
    return client, reader, writer


_QEMU = (11, 2, 0)


def _fast_poll():
    return patch.object(constants, "MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS", 0.001)


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
            memory_limited=True,
        )
    else:
        await client.restore_snapshot(
            Path("/tmp/t.vmstate"), qemu_version=qemu_version, timeout=timeout, use_template=use_template
        )


def _replies(operation: str, *, queries: list[dict[str, Any]], after_failure: bool) -> list[dict[str, Any]]:
    """What QEMU answers: capabilities, the start of the migration, the query-migrate answers,
    then a save's quit or a restore's cont, or (`after_failure`) a save's migrate_cancel and cont."""
    start = [_OK, _OK, _OK, _OK] if operation == "save" else [_OK, _OK]  # save: parameters, stop, migrate
    if after_failure:
        return [*start, *queries, *([_OK, _OK] if operation == "save" else [])]
    return [*start, *queries, _OK]


# ============================================================================
# Connection state and the QMP exchange, through restore_snapshot
# ============================================================================


async def _restore(client: MigrationClient) -> None:
    """A restore of /tmp/t.vmstate (QEMU 11.2.0, no template)."""
    await client.restore_snapshot(Path("/tmp/t.vmstate"), qemu_version=_QEMU, timeout=5.0, use_template=False)


async def _assert_not_connected(client: MigrationClient) -> None:
    """A command on `client` fails at once: TransientError MIGRATION_FAILED "Not connected"."""
    with pytest.raises(TransientError, match="Not connected", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED):
        await _restore(client)


class TestMigrationClientInit:
    """Tests for MigrationClient initialization."""

    async def test_new_client_is_not_connected(self) -> None:
        """A new client sends no command before connect(): MIGRATION_FAILED "Not connected"."""
        await _assert_not_connected(MigrationClient(Path("/tmp/fake.sock"), 501))


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
                b'{"error": {"class": "CommandNotFound"}, "id": "qmp-capabilities"}\n',
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
        await _assert_not_connected(client)

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
            connect = asyncio.create_task(client.connect(timeout=5.0))
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
        await _assert_not_connected(client)


_EVENT: dict[str, Any] = {"event": "MIGRATION", "data": {}}


class TestMigrationClientExchange:
    """The QMP exchange of the client (commands, events, failures of the stream), through a
    restore. The limit on skipped events and the id match belong to QmpClient
    (tests/test_qmp_client.py); these tests check what the client makes of them."""

    async def test_events_are_skipped_and_commands_written_as_qmp(self) -> None:
        """Events before an answer are skipped. Each command has its own id; a command with
        arguments sends them, one without has no "arguments" key."""
        replies = _replies("restore", queries=[_query("completed")], after_failure=False)
        client, _reader, writer = _make_connected(
            [_EVENT, {"event": "STOP", "data": {}}, *replies[:1], _EVENT, *replies[1:]]
        )

        with _fast_poll():
            await _restore(client)

        assert writer.commands[1:] == [
            {
                "execute": "migrate-incoming",
                "id": "migration-2",
                "arguments": {"uri": "file:/tmp/t.vmstate", "exit-on-error": False},
            },
            {"execute": "query-migrate", "id": "migration-3"},
            {"execute": "cont", "id": "migration-4"},
        ]

    @pytest.mark.parametrize(("events", "completes"), [(63, True), (65, False)], ids=["under-the-limit", "over"])
    async def test_skipped_events_limit(self, events: int, completes: bool) -> None:
        """Up to 63 skipped messages before an answer: 64 or more is MIGRATION_FAILED (a QEMU
        that floods its monitor), not an endless read."""
        replies = _replies("restore", queries=[_query("completed")], after_failure=False)
        client, _reader, _writer = _make_connected([*[_EVENT] * events, *replies])

        with _fast_poll():
            if completes:
                await _restore(client)
            else:
                with pytest.raises(
                    TransientError, match="Too many QMP messages", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
                ):
                    await _restore(client)

    async def test_connection_closed_raises(self) -> None:
        """The end of the stream (QEMU exited): TransientError MIGRATION_FAILED."""
        client, _reader, _writer = _make_connected(feed_eof=True)

        with pytest.raises(
            TransientError, match="closed unexpectedly", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
        ):
            await _restore(client)

    async def test_malformed_json_response_raises(self) -> None:
        """Malformed JSON from QEMU raises TransientError MIGRATION_FAILED (no bare JSONDecodeError)."""
        client, reader, _writer = _make_connected()
        reader.feed_data(b"not valid json\n")

        with pytest.raises(
            TransientError,
            match="QMP migrate-set-capabilities failed",
            check=lambda e: e.code is ErrorCode.MIGRATION_FAILED,
        ):
            await _restore(client)

    async def test_command_timeout_is_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A QMP command without an answer raises TransientError MIGRATION_FAILED, not a bare
        TimeoutError (the command timeout is cut to 50 ms)."""
        monkeypatch.setattr("exec_sandbox.migration_client._QMP_TIMEOUT", 0.05)
        client, _reader, _writer = _make_connected()

        with pytest.raises(
            TransientError,
            match="QMP migrate-set-capabilities failed",
            check=lambda e: e.code is ErrorCode.MIGRATION_FAILED,
        ):
            await _restore(client)


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
        """30 s, plus the time to move the guest RAM at 100 MB/s."""
        assert migration_timeout(0) == 30.0
        assert migration_timeout(256) == pytest.approx(32.56)
        assert migration_timeout(4096) == pytest.approx(70.96)


# ============================================================================
# save_snapshot — full protocol through real _execute
# ============================================================================


async def _takes(answer: bool, probed: list[Path] | None = None) -> Any:
    async def probe(folder: Path) -> bool:
        if probed is not None:
            probed.append(folder)
        return answer

    return probe


@pytest.fixture(autouse=True)
async def _direct_io_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every scripted save folder takes O_DIRECT, whatever this host's /tmp
    (tests/test_system_probes.py checks system_probes.probe_direct_io itself)."""
    monkeypatch.setattr("exec_sandbox.migration_client.probe_direct_io", await _takes(True))


class TestMigrationClientSaveSnapshot:
    """save_snapshot: set-capabilities → stop → migrate → poll → quit | cont."""

    async def test_save_command_sequence(self) -> None:
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
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
                memory_limited=True,
            )

        cmds = writer.commands
        assert [c["execute"] for c in cmds] == [
            "migrate-set-capabilities",
            "migrate-set-parameters",
            "stop",
            "migrate",
            "query-migrate",
            "quit",
        ]
        assert cmds[3]["arguments"]["uri"] == "file:/tmp/my-snapshot.vmstate"

    @pytest.mark.parametrize(
        ("direct_io", "memory_limited", "parameters"),
        [
            (True, True, {"max-bandwidth": 0, "direct-io": True}),
            (True, False, {"max-bandwidth": 0, "direct-io": True}),
            (False, False, {"max-bandwidth": 0}),
        ],
        ids=["direct-io", "direct-io-no-cgroup", "no-direct-io-no-cgroup"],
    )
    async def test_save_parameters_before_the_stop(
        self, monkeypatch: pytest.MonkeyPatch, direct_io: bool, memory_limited: bool, parameters: dict[str, Any]
    ) -> None:
        """No rate limit (QEMU's default, 128 MiB/s, paces the whole save); O_DIRECT wherever
        the folder takes it (system_probes.probe_direct_io, whatever the OS), so the save makes
        no dirty page cache in the VM's memory cgroup (no OOM kill mid-save)."""
        monkeypatch.setattr("exec_sandbox.migration_client.probe_direct_io", await _takes(direct_io))
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # quit
            ]
        )

        with _fast_poll():
            await client.save_snapshot(
                Path("/tmp/p.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=None,
                use_template=False,
                keep_running=False,
                memory_limited=memory_limited,
            )

        set_parameters = writer.commands[1]
        assert set_parameters["execute"] == "migrate-set-parameters"
        assert set_parameters["arguments"] == parameters

    async def test_the_save_probes_the_folder_of_its_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probed: list[Path] = []
        monkeypatch.setattr("exec_sandbox.migration_client.probe_direct_io", await _takes(True, probed))
        client, _reader, _writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # quit
            ]
        )

        with _fast_poll():
            await client.save_snapshot(
                Path("/saves/vm/p.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=None,
                use_template=False,
                keep_running=False,
                memory_limited=True,
            )

        assert probed == [Path("/saves/vm")]

    async def test_a_memory_limited_vm_without_o_direct_keeps_the_paced_buffered_save(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No direct-io there, and no unpaced buffered save of a VM in a memory cgroup (its dirty
        pages OOM the VM): QEMU's defaults, so no migrate-set-parameters at all."""
        monkeypatch.setattr("exec_sandbox.migration_client.probe_direct_io", await _takes(False))
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
                Path("/tmp/p.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=None,
                use_template=False,
                keep_running=False,
                memory_limited=True,
            )

        assert [c["execute"] for c in writer.commands][:2] == ["migrate-set-capabilities", "stop"]

    async def test_refused_save_parameters_leave_the_vm_running(self) -> None:
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"error": {"class": "GenericError", "desc": "direct-io not supported"}},  # migrate-set-parameters
            ]
        )

        with pytest.raises(ConfigError, check=lambda e: e.code is ErrorCode.MIGRATION_UNSUPPORTED) as caught:
            await client.save_snapshot(
                Path("/tmp/p.vmstate"),
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=None,
                use_template=False,
                keep_running=True,
                memory_limited=True,
            )

        assert caught.value.context["vm_resumed"] is True
        assert "stop" not in [c["execute"] for c in writer.commands]

    async def test_keep_running_resumes_instead_of_quit(self) -> None:
        """keep_running=True: cont after completed, never quit (VM can be saved again)."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
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
                memory_limited=True,
            )

        assert [c["execute"] for c in writer.commands][-2:] == ["query-migrate", "cont"]

    async def test_keep_running_cont_error_raises(self) -> None:
        client, _reader, _writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
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
                memory_limited=True,
            )

    async def test_migrate_error_resumes_vm(self) -> None:
        """A failed save cancels the migration and resumes the stopped VM."""
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
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
                memory_limited=True,
            )

        assert [c["execute"] for c in writer.commands][-2:] == ["migrate_cancel", "cont"]
        assert exc.value.context["vm_resumed"] is True  # the caller may keep the VM

    async def test_refused_cont_after_failed_save_keeps_original_error(self) -> None:
        client, _reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
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
                memory_limited=True,
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
                {"return": {}},  # migrate-set-parameters
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
                memory_limited=True,
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
                {"return": {}},  # migrate-set-parameters
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
                memory_limited=True,
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
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
            ]
        )
        reader.feed_eof()  # QEMU exits — readline returns b""

        with _fast_poll(), pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.MIGRATION_FAILED):
            await client.save_snapshot(
                vmstate_path,
                qemu_version=_QEMU,
                timeout=5.0,
                while_paused=None,
                use_template=False,
                keep_running=False,
                memory_limited=True,
            )

    async def test_quit_error_suppressed(self) -> None:
        """quit after a completed save may lose the socket: not an error."""
        client, reader, writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
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
                memory_limited=True,
            )

        assert writer.commands[-1]["execute"] == "quit"


# ============================================================================
# restore_snapshot — full protocol through real _execute
# ============================================================================


class TestMigrationClientRestoreSnapshot:
    """restore_snapshot: set-capabilities → migrate-incoming → poll → cont."""

    async def test_restore_command_sequence(self) -> None:
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
            _fast_poll(),
            pytest.raises(
                TransientError, match="cont command failed", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
            ),
        ):
            await client.restore_snapshot(Path("/tmp/t.vmstate"), qemu_version=_QEMU, timeout=5.0, use_template=False)

    @pytest.mark.parametrize(
        "desc",
        [
            "Could not open '/cache/l1.vmstate': Permission denied",
            "Could not open '/cache/l1.vmstate': No such file or directory",
        ],
    )
    async def test_restore_migrate_incoming_error_is_not_a_refusal(self, desc: str) -> None:
        """The command fails before QEMU reads the stream (it could not open the file): a
        TransientError MIGRATION_FAILED, not MIGRATION_REJECTED, so no good L1 entry is evicted."""
        client, _reader, _writer = _make_connected(
            [
                {"return": {}},  # migrate-set-capabilities
                {"error": {"class": "GenericError", "desc": desc}},  # migrate-incoming
            ]
        )

        with pytest.raises(
            TransientError,
            match="migrate-incoming command failed",
            check=lambda e: e.code is ErrorCode.MIGRATION_FAILED,
        ) as exc_info:
            await client.restore_snapshot(Path("/tmp/t.vmstate"), qemu_version=_QEMU, timeout=5.0, use_template=False)
        assert not rejected_by_qemu(exc_info.value)


# ============================================================================
# Context manager, against a QMP peer on a real Unix socket
# ============================================================================


@pytest.fixture
async def fake_qmp(short_tmp_dir: Path) -> AsyncGenerator[tuple[FakeQmp, Path]]:
    fake = FakeQmp(on_migrate=None, on_run=None)
    path = short_tmp_dir / "qmp.sock"
    server = await asyncio.start_unix_server(fake.serve, path=str(path))
    yield fake, path
    server.close()
    await server.wait_closed()


class TestMigrationClientContextManager:
    """async with: the handshake runs on enter, the connection closes on exit (also on error)."""

    async def test_no_qmp_listener_is_migration_failed(self, short_tmp_dir: Path) -> None:
        """No QEMU behind the QMP socket path: TransientError MIGRATION_FAILED (a save or
        load QEMU did not finish), never a builtin OSError."""
        with pytest.raises(
            TransientError,
            check=lambda e: e.code is ErrorCode.MIGRATION_FAILED,
            match=r"^QMP connection failed: ",
        ) as exc_info:
            async with MigrationClient(short_tmp_dir / "no-qemu.sock", os.getuid()):
                pass
        assert isinstance(exc_info.value.__cause__, OSError)

    async def test_quit_that_qemu_answers_by_closing_is_no_error(self, short_tmp_dir: Path) -> None:
        """QEMU may close the QMP socket before it answers quit: quit() returns."""

        async def qemu(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.write(json.dumps({"QMP": {"version": {}, "capabilities": []}}).encode() + b"\n")
            await writer.drain()
            while line := await reader.readline():
                request = json.loads(line)
                if request["execute"] == "quit":
                    writer.close()
                    return
                writer.write(json.dumps({"return": {}, "id": request["id"]}).encode() + b"\n")
                await writer.drain()

        path = short_tmp_dir / "quit.sock"
        server = await asyncio.start_unix_server(qemu, path=str(path))
        try:
            async with MigrationClient(path, os.getuid()) as client:
                await asyncio.wait_for(client.quit(), 5.0)
        finally:
            server.close()
            await server.wait_closed()

    async def test_enter_handshakes_and_exit_hangs_up(self, fake_qmp: tuple[FakeQmp, Path]) -> None:
        fake, path = fake_qmp
        async with MigrationClient(path, os.getuid()):
            assert fake.commands == ["qmp_capabilities"]
            assert not fake.hung_up.is_set()
        await asyncio.wait_for(fake.hung_up.wait(), 5.0)

    async def test_exit_hangs_up_when_the_body_raises(self, fake_qmp: tuple[FakeQmp, Path]) -> None:
        fake, path = fake_qmp
        with pytest.raises(RuntimeError, match="body failed"):
            async with MigrationClient(path, os.getuid()):
                raise RuntimeError("body failed")
        await asyncio.wait_for(fake.hung_up.wait(), 5.0)

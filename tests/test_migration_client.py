"""Tests for MigrationClient (QMP migration protocol).

The client connects (connect(), async with) to a scripted QMP peer on a real Unix socket: the
peer gives QEMU's answers in the order of the script, echoes the id of each command, and
records the commands. The tests check the commands sent, the errors raised, and what the
client does after a failure.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exec_sandbox import constants
from exec_sandbox.exceptions import ConfigError, ErrorCode, RouteError, RouteKey, SandboxError, TransientError
from exec_sandbox.migration_client import MigrationClient, migration_timeout, refusal_cause
from exec_sandbox.qmp_client import QmpClient
from tests.instance_support import FakeQmp

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

# ============================================================================
# Test Helpers
# ============================================================================


class _ScriptedQemu:
    """A QMP peer: the greeting, a return to qmp_capabilities, then the next item of `script`
    for each command, with the command's id. An event in the script goes out before the
    answer after it; bytes go out as they are (a line that is not JSON). When the script has
    no answer left, the peer closes the connection (`at_end` "close": QEMU exited) or reads
    on without an answer ("silent": QEMU does not answer)."""

    def __init__(self, script: list[dict[str, Any] | bytes], *, at_end: Literal["close", "silent"]) -> None:
        self._script = list(script)
        self._at_end = at_end
        self._commands: list[dict[str, Any]] = []
        self._hung_up = asyncio.Event()
        self._writer: asyncio.StreamWriter | None = None

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        try:
            writer.write(json.dumps({"QMP": {"version": {}, "capabilities": []}}).encode() + b"\n")
            handshake = json.loads(await reader.readline())
            writer.write(json.dumps({"return": {}, "id": handshake["id"]}).encode() + b"\n")
            while line := await reader.readline():
                request = json.loads(line)
                self._commands.append(request)
                while self._script and isinstance(self._script[0], dict) and "event" in self._script[0]:
                    writer.write(json.dumps(self._script.pop(0)).encode() + b"\n")
                if not self._script:
                    if self._at_end == "close":
                        break
                    continue
                reply = self._script.pop(0)
                writer.write(
                    reply if isinstance(reply, bytes) else json.dumps({**reply, "id": request["id"]}).encode() + b"\n"
                )
                await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()
            self._hung_up.set()

    async def sent(self, client: MigrationClient) -> list[dict[str, Any]]:
        """The commands the client sent after the handshake: it hangs up first, so the list
        is complete."""
        await client.close()
        await asyncio.wait_for(self._hung_up.wait(), timeout=5)
        return self._commands

    def cut(self) -> None:
        """QEMU is gone: the peer closes its end of the connection."""
        if self._writer is not None:
            self._writer.close()


class _Connect(Protocol):
    async def __call__(
        self, script: list[dict[str, Any] | bytes], *, at_end: Literal["close", "silent"]
    ) -> tuple[MigrationClient, _ScriptedQemu]: ...


@pytest.fixture
async def qemu(short_tmp_dir: Path) -> AsyncIterator[_Connect]:
    """Start a scripted QMP peer and connect a MigrationClient to it (connect(): greeting and
    qmp_capabilities). Closes them at the end of the test."""
    servers: list[asyncio.Server] = []
    peers: list[_ScriptedQemu] = []
    clients: list[MigrationClient] = []

    async def connect(
        script: list[dict[str, Any] | bytes], *, at_end: Literal["close", "silent"]
    ) -> tuple[MigrationClient, _ScriptedQemu]:
        peer = _ScriptedQemu(script, at_end=at_end)
        peers.append(peer)
        path = short_tmp_dir / f"qmp{len(servers)}.sock"
        servers.append(await asyncio.start_unix_server(peer.serve, path=str(path)))
        client = MigrationClient(path, os.getuid())
        clients.append(client)
        await client.connect(timeout=5.0)
        return client, peer

    yield connect
    for client in clients:
        await client.close()
    for peer in peers:
        peer.cut()  # a client that did not hang up must not hold the server open
    for server in servers:
        server.close()
        await server.wait_closed()


def _executed(commands: list[dict[str, Any]]) -> list[str]:
    return [c["execute"] for c in commands]


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
    qemu_version: tuple[int, int, int],
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

    async def test_capabilities_error_hangs_up(self, fake_qmp: tuple[FakeQmp, Path]) -> None:
        """QEMU refuses qmp_capabilities: connect() raises MIGRATION_FAILED, the client hangs up,
        and it stays not connected."""
        fake, path = fake_qmp
        fake.refused["qmp_capabilities"] = "Capabilities negotiation is already complete"
        client = MigrationClient(path, os.getuid())

        with pytest.raises(
            TransientError, match="QMP capabilities failed", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
        ):
            await client.connect(timeout=5.0)

        await asyncio.wait_for(fake.hung_up.wait(), 5.0)
        await _assert_not_connected(client)

    async def test_cancellation_waits_for_handshake_cleanup(self) -> None:
        """A cancel during the handshake closes the streams, and a second cancel during that
        close does not end connect() before the close has ended. The streams are mocks behind
        connect_and_verify (the socket boundary): a real socket cannot hold wait_closed()."""
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

    async def test_events_are_skipped_and_commands_written_as_qmp(self, qemu: _Connect) -> None:
        """Events before an answer are skipped. Each command has its own id; a command with
        arguments sends them, one without has no "arguments" key."""
        replies = _replies("restore", queries=[_query("completed")], after_failure=False)
        client, peer = await qemu(
            [_EVENT, {"event": "STOP", "data": {}}, *replies[:1], _EVENT, *replies[1:]], at_end="close"
        )

        with _fast_poll():
            await _restore(client)

        assert (await peer.sent(client))[1:] == [
            {
                "execute": "migrate-incoming",
                "id": "migration-2",
                "arguments": {"uri": "file:/tmp/t.vmstate", "exit-on-error": False},
            },
            {"execute": "query-migrate", "id": "migration-3"},
            {"execute": "cont", "id": "migration-4"},
        ]

    @pytest.mark.parametrize(
        ("events", "completes"), [(63, True), (64, False)], ids=["under-the-limit", "at-the-limit"]
    )
    async def test_skipped_events_limit(self, qemu: _Connect, events: int, completes: bool) -> None:
        """Up to 63 skipped messages before an answer: 64 or more is MIGRATION_FAILED (a QEMU
        that floods its monitor), not an endless read."""
        replies = _replies("restore", queries=[_query("completed")], after_failure=False)
        client, _peer = await qemu([*[_EVENT] * events, *replies], at_end="close")

        with _fast_poll():
            if completes:
                await _restore(client)
            else:
                with pytest.raises(
                    TransientError, match="Too many QMP messages", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
                ):
                    await _restore(client)

    async def test_connection_closed_raises(self, qemu: _Connect) -> None:
        """The end of the stream (QEMU exited): TransientError MIGRATION_FAILED."""
        client, _peer = await qemu([], at_end="close")

        with pytest.raises(
            TransientError, match="closed unexpectedly", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
        ):
            await _restore(client)

    async def test_malformed_json_response_raises(self, qemu: _Connect) -> None:
        """Malformed JSON from QEMU raises TransientError MIGRATION_FAILED (no bare JSONDecodeError)."""
        client, _peer = await qemu([b"not valid json\n"], at_end="silent")

        with pytest.raises(
            TransientError,
            match="QMP migrate-set-capabilities failed",
            check=lambda e: e.code is ErrorCode.MIGRATION_FAILED,
        ):
            await _restore(client)

    async def test_command_timeout_is_transient(self, qemu: _Connect, monkeypatch: pytest.MonkeyPatch) -> None:
        """A QMP command without an answer raises TransientError MIGRATION_FAILED, not a bare
        TimeoutError (the command timeout is cut to 50 ms)."""
        monkeypatch.setattr("exec_sandbox.migration_client._QMP_TIMEOUT", 0.05)
        client, _peer = await qemu([], at_end="silent")

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
        self, qemu: _Connect, operation: str, use_template: bool, version: tuple[int, int, int]
    ) -> None:
        """mapped-ram + multifd on, x-ignore-shared always sent with an explicit state."""
        client, peer = await qemu(
            _replies(operation, queries=[_query("completed")], after_failure=False), at_end="close"
        )

        with _fast_poll():
            await _run(client, operation, qemu_version=version, use_template=use_template, timeout=5.0)

        first = (await peer.sent(client))[0]
        assert first["execute"] == "migrate-set-capabilities"
        states = {c["capability"]: c["state"] for c in first["arguments"]["capabilities"]}
        assert states == {"multifd": True, "mapped-ram": True, "x-ignore-shared": use_template}

    @pytest.mark.parametrize("operation", ["save", "restore"])
    @pytest.mark.parametrize("version", [(10, 99, 99), (8, 2, 0)], ids=["just-below", "old"])
    async def test_below_floor_sends_nothing(
        self, qemu: _Connect, operation: str, version: tuple[int, int, int]
    ) -> None:
        client, peer = await qemu([], at_end="silent")

        with pytest.raises(
            ConfigError, match="below the migration floor", check=lambda e: e.code is ErrorCode.MIGRATION_UNSUPPORTED
        ) as exc:
            await _run(client, operation, qemu_version=version, use_template=False, timeout=5.0)

        assert await peer.sent(client) == []
        assert refusal_cause(exc.value) is not None  # an L1 entry this QEMU cannot load is evicted

    @pytest.mark.parametrize("operation", ["save", "restore"])
    async def test_refused_capability_is_rejected(self, qemu: _Connect, operation: str) -> None:
        """A refused capability fails the operation (no silent legacy stream); a save had not
        stopped the VM yet, so it runs on."""
        client, peer = await qemu(
            [{"error": {"class": "GenericError", "desc": "mapped-ram not supported"}}], at_end="close"
        )

        with pytest.raises(
            ConfigError, match="mapped-ram not supported", check=lambda e: e.code is ErrorCode.MIGRATION_UNSUPPORTED
        ) as exc:
            await _run(client, operation, qemu_version=_QEMU, use_template=False, timeout=5.0)

        assert _executed(await peer.sent(client)) == ["migrate-set-capabilities"]
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
    async def test_polls_until_completed(self, qemu: _Connect, statuses: list[str | None]) -> None:
        client, peer = await qemu(
            _replies("restore", queries=[_query(st) for st in statuses], after_failure=False), at_end="close"
        )

        with _fast_poll():
            await _run(client, "restore", qemu_version=_QEMU, use_template=False, timeout=5.0)

        assert _executed(await peer.sent(client)) == [
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
        self, qemu: _Connect, status: str, operation: str, error: tuple[type[SandboxError], ErrorCode]
    ) -> None:
        """QEMU ends a save: transient, the VM resumes and can be kept. QEMU ends a restore: it
        refused the saved state (ROUTE), and the VM is never resumed. QEMU's reason stays."""
        queries = [_query("active"), {"return": {"status": status, "error-desc": "disk full"}}]
        client, peer = await qemu(_replies(operation, queries=queries, after_failure=True), at_end="close")

        with (
            _fast_poll(),
            pytest.raises(error[0], match=f"{status}: disk full", check=lambda e: e.code is error[1]) as exc,
        ):
            await _run(client, operation, qemu_version=_QEMU, use_template=False, timeout=5.0)

        sent = _executed(await peer.sent(client))
        if operation == "save":
            assert sent[-2:] == ["migrate_cancel", "cont"]
            assert exc.value.context["vm_resumed"] is True
            assert refusal_cause(exc.value) is None
        else:
            assert sent[-1] == "query-migrate"
            assert isinstance(exc.value, RouteError)
            assert exc.value.required == {}  # the caller names the target: it knows who saved the state
            assert exc.value.found == {"qemu_version": "11.2.0"}
            assert refusal_cause(exc.value) is not None

    @pytest.mark.parametrize("operation", ["save", "restore"])
    async def test_failed_query_is_transient(self, qemu: _Connect, operation: str) -> None:
        """query-migrate itself failing says nothing about the saved state: transient, also for a restore."""
        client, _peer = await qemu(_replies(operation, queries=[_FAILED_QUERY], after_failure=True), at_end="close")

        with (
            _fast_poll(),
            pytest.raises(
                TransientError, match="query-migrate failed", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
            ) as exc,
        ):
            await _run(client, operation, qemu_version=_QEMU, use_template=False, timeout=5.0)
        assert refusal_cause(exc.value) is None  # a good L1 entry is kept

    @pytest.mark.parametrize("operation", ["save", "restore"])
    async def test_hung_migration_is_transient(self, qemu: _Connect, operation: str) -> None:
        """A migration that never ends becomes TransientError MIGRATION_FAILED, not a bare TimeoutError."""
        client, _peer = await qemu(
            _replies(operation, queries=[_query("active")] * 500, after_failure=True), at_end="close"
        )

        with (
            _fast_poll(),
            pytest.raises(
                TransientError, match="did not complete", check=lambda e: e.code is ErrorCode.MIGRATION_FAILED
            ),
        ):
            await _run(client, operation, qemu_version=_QEMU, use_template=False, timeout=0.05)


class TestRejectedByQemu:
    """refusal_cause decides the L1 eviction and the reuse() ROUTE rule, from the evidence of
    one failure. It matches by code (MIGRATION_REJECTED, MIGRATION_UNSUPPORTED; QEMU_CRASHED
    with SIGABRT; GUEST_SILENT_AFTER_RESTORE with QEMU's exit code 0), never by class, anywhere
    in the cause or context chain: a restore whose setup fails (HOST_DEPENDENCY, also a
    ConfigError) or whose QEMU fails otherwise must not evict a good entry."""

    @staticmethod
    def _rejected() -> SandboxError:
        return RouteError(
            "migrate-incoming command failed",
            code=ErrorCode.MIGRATION_REJECTED,
            required={},
            found={RouteKey.QEMU_VERSION: "11.2.0"},
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
    def _died(returncode: int, console: str, *, loading: bool) -> SandboxError:
        return TransientError(
            f"QEMU died during the restore of vm (rc={returncode}).",
            code=ErrorCode.QEMU_CRASHED,
            context={"vm_id": "vm", "returncode": returncode, "loading": loading, "console_log": console},
            hint=None,
            instance_id=None,
        )

    _MSR_ABORT = (
        "qemu-system-x86_64: error: failed to set MSR 0x48f to 0x7fffff00036dfb\n"
        "qemu-system-x86_64: ../target/i386/kvm/kvm.c:4116: kvm_buf_set_msrs: Assertion `ret == cpu->kvm_msr_buf->nmsrs' failed."
    )

    def _aborted(self) -> SandboxError:
        """QEMU killed by SIGABRT while it loaded the state (an x86 MSR KVM refuses asserts)."""
        return self._died(-signal.SIGABRT, self._MSR_ABORT, loading=True)

    def _aborted_before_the_load(self) -> SandboxError:
        """The same abort before the load started (QEMU waits with -incoming defer): not the state."""
        return self._died(-signal.SIGABRT, self._MSR_ABORT, loading=False)

    def _aborted_on_a_limit(self) -> SandboxError:
        """QEMU could not make a thread (a host limit: error_exit aborts): it can pass."""
        return self._died(-signal.SIGABRT, "qemu: qemu_thread_create: Resource temporarily unavailable", loading=True)

    @pytest.mark.parametrize(
        "line",
        [
            "qemu: qemu_thread_create: Invalid argument",
            "qemu-system-x86_64: kvm_init_vcpu: Resource temporarily unavailable",
            "qemu-system-x86_64: cannot set up guest memory 'mem0': Cannot allocate memory",
            "(process:4242): GLib-ERROR **: ../glib/gmem.c:136: failed to allocate 1073741824 bytes",
        ],
        ids=["thread-create", "eagain", "enomem", "glib-allocation"],
    )
    def test_an_abort_on_a_host_limit_can_pass(self, line: str) -> None:
        """Each host-limit line makes a SIGABRT during the load no refusal (_HOST_LIMIT_LINES)."""
        assert refusal_cause(self._died(-signal.SIGABRT, f"[    1.2] guest line\n{line}", loading=True)) is None

    def _registers_after_init(self) -> SandboxError:
        """KVM refused the loaded vCPU registers: QEMU ends with 1 (accel/kvm/kvm-all.c)."""
        return self._died(1, "qemu-system-aarch64: Failed to put registers after init: Invalid argument", loading=True)

    def _exit_1_other(self) -> SandboxError:
        return self._died(
            1, "qemu-system-aarch64: cannot set up guest memory 'mem0': Cannot allocate memory", loading=True
        )

    def _killed(self) -> SandboxError:
        """QEMU killed by SIGKILL (the OOM killer): it can pass."""
        return self._died(-signal.SIGKILL, "(empty)", loading=True)

    @staticmethod
    def _silent(exit_code: int | None) -> SandboxError:
        console = "[    0.000000] Linux version 6.18.63\n[    2.413000] Kernel panic - not syncing: Attempted to kill init! exitcode=0x0000000b"
        return TransientError(
            "VM vm: the guest did not answer after its saved state loaded.",
            code=ErrorCode.GUEST_SILENT_AFTER_RESTORE,
            context={"vm_id": "vm", "vmstate_path": "/x", "exit_code": exit_code, "console_log": console},
            hint=None,
            instance_id=None,
        )

    def _panicked(self) -> SandboxError:
        """The guest panicked right after the load: the reboot ends QEMU with 0 (-no-reboot)."""
        return self._silent(0)

    def _terminated(self) -> SandboxError:
        """QEMU ended with 0 on a SIGTERM from outside: not a guest panic."""
        error = self._silent(0)
        error.context["console_log"] = "qemu-system-x86_64: terminating on signal 15 from pid 4242 (<unknown process>)"
        return error

    def _silent_running(self) -> SandboxError:
        """The guest did not answer and QEMU still runs (host load, or a hang): it can pass."""
        return self._silent(None)

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
            ("_aborted", True),
            ("_aborted_before_the_load", False),
            ("_aborted_on_a_limit", False),
            ("_registers_after_init", True),
            ("_exit_1_other", False),
            ("_killed", False),
            ("_panicked", True),
            ("_terminated", False),
            ("_silent_running", False),
        ],
    )
    def test_matches_only_refusal_codes(self, make: str, refused: bool, link: str) -> None:
        error = self._chained(getattr(self, make)(), link=link)
        assert (refusal_cause(error) is not None) is refused

    @pytest.mark.parametrize(
        ("make", "cause"),
        [
            ("_rejected", "migrate-incoming command failed"),
            (
                "_aborted",
                "qemu-system-x86_64: ../target/i386/kvm/kvm.c:4116: kvm_buf_set_msrs: "
                "Assertion `ret == cpu->kvm_msr_buf->nmsrs' failed.",
            ),
            ("_panicked", "Kernel panic - not syncing: Attempted to kill init! exitcode=0x0000000b"),
            ("_registers_after_init", "qemu-system-aarch64: Failed to put registers after init: Invalid argument"),
        ],
    )
    def test_the_cause_is_qemus_or_the_guests_line(self, make: str, cause: str) -> None:
        """The reason a reuse() error names: QEMU's message, its last line, the panic line."""
        assert refusal_cause(self._chained(getattr(self, make)(), link="cause")) == cause

    def test_refusal_below_a_plain_error_is_found(self) -> None:
        """The restore wrap, then an OSError (context), then QEMU's refusal (its cause)."""
        os_error = OSError("QMP socket closed")
        assert refusal_cause(self._chained(os_error, link="context")) is None
        os_error.__cause__ = self._rejected()
        assert refusal_cause(self._chained(os_error, link="context")) is not None

    def test_cycle_in_the_chain_ends(self) -> None:
        a, b = self._failed(), self._crashed()
        a.__context__, b.__context__ = b, a
        assert refusal_cause(a) is None


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

    async def test_save_command_sequence(self, qemu: _Connect) -> None:
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # quit
            ],
            at_end="close",
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

        cmds = await peer.sent(client)
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
        self,
        qemu: _Connect,
        monkeypatch: pytest.MonkeyPatch,
        direct_io: bool,
        memory_limited: bool,
        parameters: dict[str, Any],
    ) -> None:
        """No rate limit (QEMU's default, 128 MiB/s, paces the whole save); O_DIRECT wherever
        the folder takes it (system_probes.probe_direct_io, whatever the OS), so the save makes
        no dirty page cache in the VM's memory cgroup (no OOM kill mid-save)."""
        monkeypatch.setattr("exec_sandbox.migration_client.probe_direct_io", await _takes(direct_io))
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # quit
            ],
            at_end="close",
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

        set_parameters = (await peer.sent(client))[1]
        assert set_parameters["execute"] == "migrate-set-parameters"
        assert set_parameters["arguments"] == parameters

    async def test_the_save_probes_the_folder_of_its_file(
        self, qemu: _Connect, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probed: list[Path] = []
        monkeypatch.setattr("exec_sandbox.migration_client.probe_direct_io", await _takes(True, probed))
        client, _peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # quit
            ],
            at_end="close",
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
        self, qemu: _Connect, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No direct-io there, and no unpaced buffered save of a VM in a memory cgroup (its dirty
        pages OOM the VM): QEMU's defaults, so no migrate-set-parameters at all."""
        monkeypatch.setattr("exec_sandbox.migration_client.probe_direct_io", await _takes(False))
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # quit
            ],
            at_end="close",
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

        assert _executed(await peer.sent(client))[:2] == ["migrate-set-capabilities", "stop"]

    async def test_refused_save_parameters_leave_the_vm_running(self, qemu: _Connect) -> None:
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"error": {"class": "GenericError", "desc": "direct-io not supported"}},  # migrate-set-parameters
            ],
            at_end="close",
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
        assert "stop" not in _executed(await peer.sent(client))

    async def test_keep_running_resumes_instead_of_quit(self, qemu: _Connect) -> None:
        """keep_running=True: cont after completed, never quit (VM can be saved again)."""
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # cont
            ],
            at_end="close",
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

        assert _executed(await peer.sent(client))[-2:] == ["query-migrate", "cont"]

    async def test_keep_running_cont_error_raises(self, qemu: _Connect) -> None:
        client, _peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"error": {"class": "GenericError", "desc": "cannot resume"}},  # cont
            ],
            at_end="close",
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

    async def test_migrate_error_resumes_vm(self, qemu: _Connect) -> None:
        """A failed save cancels the migration and resumes the stopped VM."""
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"error": {"class": "GenericError", "desc": "cannot migrate"}},  # migrate
                {"return": {}},  # migrate_cancel
                {"return": {}},  # cont
            ],
            at_end="close",
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

        assert _executed(await peer.sent(client))[-2:] == ["migrate_cancel", "cont"]
        assert exc.value.context["vm_resumed"] is True  # the caller may keep the VM

    async def test_refused_cont_after_failed_save_keeps_original_error(self, qemu: _Connect) -> None:
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "failing", "error-desc": "write error"}},  # query-migrate
                {"return": {}},  # migrate_cancel
                {"error": {"class": "GenericError", "desc": "Migration is not finalized yet"}},  # cont
            ],
            at_end="close",
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

        assert _executed(await peer.sent(client))[-1] == "cont"
        assert exc.value.context["vm_resumed"] is False  # the VM can stay paused: the caller replaces it

    @pytest.mark.parametrize("cont", ["accepted", "refused"])
    async def test_while_paused_failure_resumes_or_reports_a_paused_vm(self, qemu: _Connect, cont: str) -> None:
        """A hook failure (e.g. the home copy hits ENOSPC): with the VM running again the
        hook's error stays; with cont refused, a typed error says the VM may stay paused."""
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # migrate_cancel
                {"return": {}} if cont == "accepted" else {"error": {"class": "GenericError", "desc": "busy"}},
            ],
            at_end="close",
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
        assert _executed(await peer.sent(client))[-2:] == ["migrate_cancel", "cont"]  # resume tried
        if cont == "accepted":
            assert exc.value.errno == errno.ENOSPC  # the hook's own error
        else:
            assert isinstance(exc.value, TransientError)
            assert exc.value.code is ErrorCode.MIGRATION_FAILED
            assert exc.value.context["vm_resumed"] is False

    async def test_a_cancel_while_paused_resumes_the_vm(self, qemu: _Connect) -> None:
        """A cancel of the caller after the stop (here in the hook) resumes the VM before it
        goes on: a session whose save was cancelled stays open, and a paused VM would run
        none of its later calls."""
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # migrate_cancel
                {"return": {}},  # cont
            ],
            at_end="close",
        )
        paused = asyncio.Event()

        async def copy_home() -> None:
            paused.set()
            await asyncio.Event().wait()

        with _fast_poll():
            save = asyncio.create_task(
                client.save_snapshot(
                    Path("/tmp/t.vmstate"),
                    qemu_version=_QEMU,
                    timeout=5.0,
                    while_paused=copy_home,
                    use_template=False,
                    keep_running=True,
                    memory_limited=True,
                )
            )
            await asyncio.wait_for(paused.wait(), timeout=5)
            save.cancel()
            with pytest.raises(asyncio.CancelledError):
                await save

        assert _executed(await peer.sent(client))[-2:] == ["migrate_cancel", "cont"]

    async def test_refused_stop_leaves_the_vm_running(self, qemu: _Connect) -> None:
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"error": {"class": "GenericError", "desc": "cannot stop"}},  # stop
            ],
            at_end="close",
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
        assert "cont" not in _executed(await peer.sent(client))  # nothing was stopped

    async def test_lost_connection_is_failure_even_with_file(self, qemu: _Connect, tmp_path: Path) -> None:
        """QEMU gone before "completed" is a failure, even if the file has bytes
        (mapped-ram pre-sizes the file, so its size proves nothing)."""
        vmstate_path = tmp_path / "test.vmstate"
        vmstate_path.write_bytes(b"partial migration stream")
        client, _peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
            ],
            at_end="close",  # QEMU exits — readline returns b""
        )

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

    async def test_quit_error_suppressed(self, qemu: _Connect) -> None:
        """quit after a completed save may lose the socket: not an error."""
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-set-parameters
                {"return": {}},  # stop
                {"return": {}},  # migrate
                {"return": {"status": "completed"}},  # query-migrate
            ],
            at_end="close",  # QEMU exits before quit response
        )

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

        assert (await peer.sent(client))[-1]["execute"] == "quit"


# ============================================================================
# restore_snapshot — full protocol through real _execute
# ============================================================================


class TestMigrationClientRestoreSnapshot:
    """restore_snapshot: set-capabilities → migrate-incoming → poll → cont."""

    async def test_restore_command_sequence(self, qemu: _Connect) -> None:
        client, peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-incoming
                {"return": {"status": "completed"}},  # query-migrate
                {"return": {}},  # cont
            ],
            at_end="close",
        )

        with _fast_poll():
            await client.restore_snapshot(
                Path("/tmp/snapshot.vmstate"), qemu_version=_QEMU, timeout=5.0, use_template=False
            )

        cmds = await peer.sent(client)
        assert [c["execute"] for c in cmds] == ["migrate-set-capabilities", "migrate-incoming", "query-migrate", "cont"]
        # exit-on-error=false: QEMU stays up on a bad stream so the error can be read
        assert cmds[1]["arguments"] == {"uri": "file:/tmp/snapshot.vmstate", "exit-on-error": False}

    async def test_restore_cont_error(self, qemu: _Connect) -> None:
        client, _peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-incoming
                {"return": {"status": "completed"}},  # query-migrate
                {"error": {"class": "GenericError", "desc": "cannot resume"}},  # cont
            ],
            at_end="close",
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
    async def test_restore_migrate_incoming_error_is_not_a_refusal(self, qemu: _Connect, desc: str) -> None:
        """The command fails before QEMU reads the stream (it could not open the file): a
        TransientError MIGRATION_FAILED, not MIGRATION_REJECTED, so no good L1 entry is evicted."""
        client, _peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"error": {"class": "GenericError", "desc": desc}},  # migrate-incoming
            ],
            at_end="close",
        )

        with pytest.raises(
            TransientError,
            match="migrate-incoming command failed",
            check=lambda e: e.code is ErrorCode.MIGRATION_FAILED,
        ) as exc_info:
            await client.restore_snapshot(Path("/tmp/t.vmstate"), qemu_version=_QEMU, timeout=5.0, use_template=False)
        assert refusal_cause(exc_info.value) is None


class TestRestoreEvidence:
    """What restore_vm reads after a failed load: whether migrate-incoming was sent (a QEMU
    death counts as the state's only then) and whether QEMU closed the connection (it ends:
    wait for its exit code; one that answered still runs)."""

    async def test_no_load_before_migrate_incoming(self, qemu: _Connect) -> None:
        client, _peer = await qemu([{"error": {"class": "GenericError", "desc": "no mapped-ram"}}], at_end="silent")
        with pytest.raises(ConfigError):
            await client.restore_snapshot(Path("/tmp/t.vmstate"), qemu_version=_QEMU, timeout=5.0, use_template=False)
        assert client.load_started is False

    async def test_qemu_gone_before_it_took_the_load(self, qemu: _Connect) -> None:
        """QEMU ended before it answered migrate-incoming: it never started the load (QEMU
        reads the stream only after its reply), so a death then is not the state's."""
        client, _peer = await qemu([{"return": {}}], at_end="close")  # set-capabilities only
        with pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.MIGRATION_FAILED):
            await client.restore_snapshot(Path("/tmp/t.vmstate"), qemu_version=_QEMU, timeout=5.0, use_template=False)
        await client.close()
        assert (client.load_started, client.qemu_closed) == (False, True)

    async def test_a_reset_is_a_closed_connection(self) -> None:
        """Linux: QEMU that exits with unread bytes on the socket resets it: the reader holds the
        error and is not at EOF, and QEMU closed the connection all the same."""
        client = MigrationClient(Path("/tmp/qmp.sock"), expected_uid=0)
        reader = asyncio.StreamReader()
        reader.set_exception(ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"))
        client._qmp = cast("Any", SimpleNamespace(reader=reader, close=AsyncMock()))  # pyright: ignore[reportPrivateUsage]
        await client.close()
        assert client.qemu_closed is True

    async def test_qemu_ended_during_the_load(self, qemu: _Connect) -> None:
        """migrate-incoming sent, then EOF (QEMU exited while it loaded)."""
        client, _peer = await qemu([{"return": {}}, {"return": {}}], at_end="close")
        with _fast_poll(), pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.MIGRATION_FAILED):
            await client.restore_snapshot(Path("/tmp/t.vmstate"), qemu_version=_QEMU, timeout=5.0, use_template=False)
        await client.close()
        assert (client.load_started, client.qemu_closed) == (True, True)

    async def test_qemu_that_answered_still_runs(self, qemu: _Connect) -> None:
        """QEMU refused the stream and answered: the connection stays up (QEMU runs)."""
        client, _peer = await qemu(
            [
                {"return": {}},  # migrate-set-capabilities
                {"return": {}},  # migrate-incoming
                {"return": {"status": "failed", "error-desc": "Unknown ramblock"}},  # query-migrate
            ],
            at_end="silent",
        )
        with _fast_poll(), pytest.raises(RouteError):
            await client.restore_snapshot(Path("/tmp/t.vmstate"), qemu_version=_QEMU, timeout=5.0, use_template=False)
        await client.close()
        assert (client.load_started, client.qemu_closed) == (True, False)


# ============================================================================
# Context manager, against a QMP peer on a real Unix socket
# ============================================================================


@pytest.fixture
async def fake_qmp(short_tmp_dir: Path) -> AsyncGenerator[tuple[FakeQmp, Path]]:
    fake = FakeQmp(on_migrate=None, on_run=None)
    path = short_tmp_dir / "qmp.sock"
    await fake.start(path)
    yield fake, path
    await fake.stop()


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

    @pytest.mark.parametrize("error", [ConnectionResetError, BrokenPipeError])
    async def test_a_connection_that_fails_mid_command_is_migration_failed(
        self, fake_qmp: tuple[FakeQmp, Path], tmp_path: Path, error: type[OSError]
    ) -> None:
        """The QMP connection fails under a command (a reset or broken pipe before `stop`):
        TransientError MIGRATION_FAILED with the VM still running, never a builtin OSError
        out of save_snapshot. The transport seam (QmpClient.execute) fails after the handshake."""
        _fake, path = fake_qmp
        async with MigrationClient(path, os.getuid()) as client:
            with (
                patch.object(QmpClient, "execute", side_effect=error("QMP socket")),
                pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.MIGRATION_FAILED) as exc_info,
            ):
                await client.save_snapshot(
                    tmp_path / "vmstate",
                    qemu_version=(11, 1, 3),
                    timeout=5.0,
                    use_template=True,
                    keep_running=False,
                    while_paused=None,
                    memory_limited=False,
                )
        assert isinstance(exc_info.value.__cause__, error)
        assert exc_info.value.context["vm_resumed"] is True

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

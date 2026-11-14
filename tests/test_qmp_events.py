"""QmpEventWatch: event delivery, the connection life cycle (wait, reconnect, back-off), and the
placing of event stamps on the monotonic clock.

Tests that wait for the end of a watch await _run, the body of the task that start makes:
the watch has no public wait for its end (stop cancels it)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from exec_sandbox import qmp_events
from exec_sandbox.qmp_events import QmpEvent, QmpEventWatch
from tests.conftest import asyncio_with, time_with

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_GREETING = b'{"QMP": {"version": {"qemu": {"micro": 3, "minor": 1, "major": 11}}, "capabilities": []}}\n'
_NOWHERE = Path("/nowhere/qmp-events.sock")


async def _no_event(_event: QmpEvent) -> None:
    pass


def _watch(
    exit_code: list[int | None],
    *,
    path: Path,
    on_event: Callable[[QmpEvent], Awaitable[None]],
    on_connect: Callable[[QmpEventWatch], Awaitable[None]],
) -> QmpEventWatch:
    """A watch of a QEMU whose exit code is `exit_code[0]` (None while it runs)."""
    return QmpEventWatch(
        path=path,
        expected_uid=os.getuid(),
        vm_id="watch-vm",
        qemu_exit_code=lambda: exit_code[0],
        on_event=on_event,
        on_connect=on_connect,
    )


async def test_events_go_to_the_owner_with_their_time(short_tmp_dir: Path) -> None:
    """After the start, on_connect queries; the events before its answer and after it reach
    on_event with their name, data and time; a line that is not JSON is skipped. EOF (QEMU's
    exit) ends the watch."""
    path = short_tmp_dir / "qmp-events.sock"
    stamp = time.time() - 0.5

    def event(name: str, **data: Any) -> bytes:
        timestamp = {"seconds": int(stamp), "microseconds": int(stamp % 1 * 1_000_000)}
        return json.dumps({"event": name, "data": data, "timestamp": timestamp}).encode() + b"\n"

    async def qemu(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_GREETING)
        for _ in range(2):  # qmp_capabilities, then the owner's query
            request = json.loads(await reader.readline())
            if request["execute"] == "query-block":
                writer.write(event("BLOCK_IO_ERROR", nospace=True))
            writer.write(json.dumps({"return": ["drive"], "id": request["id"]}).encode() + b"\n")
        writer.write(b"not json\n" + event("SHUTDOWN"))
        await writer.drain()
        writer.close()

    events: list[QmpEvent] = []
    answers: list[dict[str, Any]] = []

    async def on_event(event: QmpEvent) -> None:
        events.append(event)

    async def on_connect(watch: QmpEventWatch) -> None:
        answers.append(await watch.query("query-block"))

    server = await asyncio.start_unix_server(qemu, path=str(path))
    try:
        async with asyncio.timeout(5):
            await _watch([None], path=path, on_event=on_event, on_connect=on_connect)._run()  # pyright: ignore[reportPrivateUsage]
    finally:
        server.close()
    assert [answer["return"] for answer in answers] == [["drive"]]
    assert [(e.name, e.data) for e in events] == [("BLOCK_IO_ERROR", {"nospace": True}), ("SHUTDOWN", {})]
    now = time.monotonic()
    assert all(now - 1.5 < e.at <= now for e in events)  # made about 0.5 s ago, not at the read


async def test_connect_waits_as_long_as_qemu_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """No deadline on the socket wait (a slow QEMU start under load must not end the watch):
    only QEMU's exit ends it. A deadline shows only after it passes: the test reads the
    timeout that the watch gives wait_for_socket."""
    waits: list[dict[str, Any]] = []
    exit_code: list[int | None] = [None]

    async def wait_for_socket(_path: Path, **kwargs: Any) -> None:
        waits.append(kwargs)
        exit_code[0] = 0
        kwargs["abort_check"]()  # QEMU exited: the check ends the wait

    monkeypatch.setattr("exec_sandbox.qmp_events.wait_for_socket", wait_for_socket)
    await _watch(exit_code, path=_NOWHERE, on_event=_no_event, on_connect=AsyncMock())._run()  # pyright: ignore[reportPrivateUsage]
    assert len(waits) == 1
    assert waits[0]["timeout"] == math.inf


async def test_repeated_failures_back_off(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A connection that keeps failing while QEMU runs never spins: the pauses double from 1 ms
    to 1 s, and only the first failure in a row is a warning."""
    exit_code: list[int | None] = [None]
    pauses: list[float] = []

    async def wait_for_socket(_path: Path, **_kwargs: Any) -> None:
        raise ConnectionRefusedError("refused")

    async def sleep(seconds: float) -> None:
        pauses.append(seconds)
        if len(pauses) == 12:
            exit_code[0] = 0  # QEMU exits: the watch ends

    monkeypatch.setattr("exec_sandbox.qmp_events.wait_for_socket", wait_for_socket)
    monkeypatch.setattr("exec_sandbox.qmp_events.asyncio", asyncio_with(sleep=sleep))
    with caplog.at_level(logging.DEBUG, logger="exec_sandbox"):
        await _watch(exit_code, path=_NOWHERE, on_event=_no_event, on_connect=AsyncMock())._run()  # pyright: ignore[reportPrivateUsage]

    assert pauses == [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.0, 1.0]
    lost = [r for r in caplog.records if r.getMessage() == "QEMU event connection lost, connecting again"]
    assert [r.levelno for r in lost] == [logging.WARNING] + [logging.DEBUG] * 11


def _started_connections(monkeypatch: pytest.MonkeyPatch, sleep: Callable[[float], Awaitable[None]]) -> None:
    """Connections that start at once (no socket, no peer check, no handshake), and `sleep`
    as the watch's pause."""

    async def connect(_path: Path, **_kwargs: Any) -> tuple[MagicMock, MagicMock]:
        return MagicMock(), MagicMock(wait_closed=AsyncMock())

    monkeypatch.setattr("exec_sandbox.qmp_events.wait_for_socket", connect)
    monkeypatch.setattr("exec_sandbox.qmp_events.verify_socket_peer", lambda *_a: None)
    monkeypatch.setattr(qmp_events.QmpClient, "start", AsyncMock())
    monkeypatch.setattr("exec_sandbox.qmp_events.asyncio", asyncio_with(sleep=sleep))


async def test_failures_right_after_the_start_back_off_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection that fails right after its start (an event flood, a line over the limit)
    is no good connection: the pauses keep doubling."""
    exit_code: list[int | None] = [None]
    pauses: list[float] = []

    async def sleep(seconds: float) -> None:
        pauses.append(seconds)
        if len(pauses) == 6:
            exit_code[0] = 0

    _started_connections(monkeypatch, sleep)
    flooded = AsyncMock(side_effect=ConnectionResetError("flood"))
    await _watch(exit_code, path=_NOWHERE, on_event=_no_event, on_connect=flooded)._run()  # pyright: ignore[reportPrivateUsage]

    assert pauses == [0.001, 0.002, 0.004, 0.008, 0.016, 0.032]


async def test_a_connection_that_lasted_starts_the_backoff_over(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """After a connection that lasted (>= the longest pause), a new failure is a new event: the
    pauses start again at 1 ms, with a warning again."""
    exit_code: list[int | None] = [None]
    pauses: list[float] = []
    clock = [1000.0]
    connections = 0

    async def lost(_watch: QmpEventWatch) -> None:
        nonlocal connections
        connections += 1
        if connections == 4:
            clock[0] += 1.5  # the 4th connection lasts 1.5 s before it fails
        raise ConnectionResetError("lost")

    async def sleep(seconds: float) -> None:
        pauses.append(seconds)
        if len(pauses) == 6:
            exit_code[0] = 0

    _started_connections(monkeypatch, sleep)
    # The module's own time name only: asyncio's loop clock keeps the real time.monotonic.
    monkeypatch.setattr("exec_sandbox.qmp_events.time", time_with(monotonic=lambda: clock[0]))
    with caplog.at_level(logging.DEBUG, logger="exec_sandbox"):
        await _watch(exit_code, path=_NOWHERE, on_event=_no_event, on_connect=lost)._run()  # pyright: ignore[reportPrivateUsage]

    assert pauses == [0.001, 0.002, 0.004, 0.001, 0.002, 0.004]
    warnings = [r for r in caplog.records if r.getMessage() == "QEMU event connection lost, connecting again"]
    assert [r.levelno for r in warnings] == [logging.WARNING, logging.DEBUG, logging.DEBUG] * 2


async def test_stop_ends_the_watch_and_returns_a_caller_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """stop cancels the watch and waits for its end; a cancel of the caller meanwhile is
    returned, so the owner finishes its own end first."""
    waiting = asyncio.Event()
    closed: list[bool] = []

    async def wait_for_socket(_path: Path, **_kwargs: Any) -> None:
        waiting.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.05)  # the socket close takes a moment
            closed.append(True)

    monkeypatch.setattr("exec_sandbox.qmp_events.wait_for_socket", wait_for_socket)
    watch = _watch([None], path=_NOWHERE, on_event=_no_event, on_connect=AsyncMock())
    watch.start()
    await waiting.wait()
    stopping = asyncio.create_task(watch.stop())
    await asyncio.sleep(0.01)
    stopping.cancel()  # the caller is cancelled while the watch closes
    cancellation = await stopping
    assert isinstance(cancellation, asyncio.CancelledError)
    assert closed == [True]  # stop returned after the watch closed its socket


class TestEventMonotonic:
    """A QMP event's wall-clock stamp placed on the monotonic clock (_event_monotonic): samples
    are (monotonic, wall - monotonic), oldest first, the last taken at the read (103). A test
    cannot step the host's wall clock: the tests give the private function the samples."""

    @pytest.mark.parametrize(
        ("stamp", "samples", "expected"),
        [
            (1000.0, [(90.0, 900.0), (103.0, 900.0)], 100.0),  # no step: exact, 3 s before the read
            (1000.0, [(90.0, 900.0), (103.0, 905.0)], 103.0),  # forward step: made now
            (1000.0, [(90.0, 900.0), (103.0, -2700.0)], 103.0),  # back step of 1 h: made now
            (1000.0, [(90.0, 900.0), (103.0, 900.05)], pytest.approx(99.95)),  # 50 ms: no step (the last offset)
            # stamp, back step, then a sample (a reconnect's handshake) before the read: the
            # older sample shows the step
            (999.0, [(90.0, 900.0), (101.0, 899.5), (103.0, 899.5)], 103.0),
            (996.5, [(97.0, 900.0), (103.0, 900.0)], 96.5),  # held by QEMU just before the first sample
            (1000.0, [(-161897.0, 900.0), (101.0, 900.1), (103.0, 900.3)], 103.0),  # a step in the recent pair
            (990.0, [(97.0, 900.0), (103.0, 900.0)], 103.0),  # older than the samples and a hold
            (None, [(90.0, 900.0), (103.0, 900.0)], 103.0),  # no stamp
            (1010.0, [(90.0, 900.0), (103.0, 900.0)], 103.0),  # a stamp after now: never later than now
        ],
        ids=[
            "late-read",
            "step-forward",
            "step-back",
            "drift",
            "step-before-the-last-sample",
            "held-before-the-first-sample",
            "step-in-the-recent-pair",
            "older-than-the-samples",
            "no-stamp",
            "future-stamp",
        ],
    )
    def test_place(self, stamp: float | None, samples: list[tuple[float, float]], expected: object) -> None:
        assert qmp_events._event_monotonic(stamp, now=103.0, samples=samples) == expected  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.parametrize(
        ("drift", "expected"), [(1e-5, pytest.approx(102.0)), (0.0, 103.0)], ids=["macos", "linux"]
    )
    def test_an_old_reference_allows_drift_only_where_the_clocks_drift(
        self, monkeypatch: pytest.MonkeyPatch, drift: float, expected: object
    ) -> None:
        """A reference sample 45 h old and 0.1004 s of change since: macOS drift (placed), or
        on Linux, where the offset never drifts, a step (made now)."""
        monkeypatch.setattr(qmp_events, "_CLOCK_DRIFT_PER_SECOND", drift)
        samples = [(-161897.0, 900.0), (101.0, 900.1004), (103.0, 900.1004)]
        assert qmp_events._event_monotonic(900.1004 + 102.0, now=103.0, samples=samples) == expected  # pyright: ignore[reportPrivateUsage]

    def test_this_host_allows_drift_only_on_macos(self) -> None:
        """The case above with the drift allowance of this host: placed on macOS, made now on Linux."""
        samples = [(-161897.0, 900.0), (101.0, 900.1004), (103.0, 900.1004)]
        expected = pytest.approx(102.0) if sys.platform == "darwin" else 103.0
        assert qmp_events._event_monotonic(900.1004 + 102.0, now=103.0, samples=samples) == expected  # pyright: ignore[reportPrivateUsage]

    def test_samples_keep_the_newest_older_than_the_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The samples of the last _OFFSET_HISTORY_SECONDS stay, and the newest older one (the
        reference before a held stamp); older ones go. The kept samples bound the memory of
        the watch, and change event times only across a wall-clock step: the test reads them."""
        clock = [0.0]
        monkeypatch.setattr(qmp_events, "time", time_with(monotonic=lambda: clock[0], time=lambda: clock[0] + 900.0))
        watch = _watch([None], path=_NOWHERE, on_event=_no_event, on_connect=AsyncMock())
        for t in (0.0, 1.0, 2.0, 6.5, 7.0):
            clock[0] = t
            watch._sample()  # pyright: ignore[reportPrivateUsage]
        assert [t for t, _ in watch._samples] == [2.0, 6.5, 7.0]  # pyright: ignore[reportPrivateUsage]

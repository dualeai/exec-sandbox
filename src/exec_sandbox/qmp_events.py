"""QMP events: the event monitor of one QEMU process, with its own client and life cycle.

QmpEventWatch connects to the event-only monitor (VmWorkingDirectory.qmp_events_socket) as
long as QEMU runs, starts its QmpClient, lets its owner query the state at each connection
(QEMU sends no event of the time before it), and gives each event line, as a QmpEvent with
the time it was made, to its owner. A lost connection is made again with a back-off. The
owner (QemuVM) keeps only its reactions: the protocol, the reconnection and the clock are
here.

Stamps: QEMU stamps each event on the wall clock (g_get_real_time), and holds a
rate-limited event (VSERPORT_CHANGE: 1 per second per port) with its stamp. The watch places
a stamp on the monotonic clock against its samples of time.time() - time.monotonic()
(_event_monotonic), so a late read (a busy loop) keeps the true time and a clock step is seen.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import CommunicationError, ConfigError, SandboxError
from exec_sandbox.qmp_client import QmpClient
from exec_sandbox.socket_auth import verify_socket_peer
from exec_sandbox.subprocess_utils import wait_for_socket

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

logger = get_logger(__name__)


_CLOCK_STEP_SECONDS: Final = 0.1
"""A change of time.time() - time.monotonic() larger than this (plus the drift allowance)
between two neighbouring samples of the event watch is a step of the wall clock: an event
stamp across it cannot be placed."""

_CLOCK_DRIFT_PER_SECOND: Final = 1e-5 if sys.platform == "darwin" else 0.0
"""How fast time.time() - time.monotonic() may change with no step, per second between two
samples: macOS time.monotonic drifts about 0.6 us/s against the wall clock (the kept
reference sample can be hours old); Linux slews CLOCK_MONOTONIC with CLOCK_REALTIME, so
there any change is a step."""

_HELD_EVENT_SECONDS: Final = 2.0
"""QEMU holds a rate-limited event (with its stamp) for at most 1 s and then sends it to
every monitor, a new connection too: a stamp up to this much older than the watch's first
sample is placed with that sample (a step inside this window is not seen)."""

_OFFSET_HISTORY_SECONDS: Final = 5.0
"""How long the event watch keeps its clock samples (and the newest older one): more than
QEMU's 1 s rate limit, which holds an event with its stamp, plus a late read."""


def _stamp_of(message: dict[str, Any]) -> float | None:
    """The wall-clock timestamp of a QMP event (QEMU's g_get_real_time at the event's making,
    kept by its rate limit), or None."""
    stamp = message.get("timestamp")
    if isinstance(stamp, dict):
        stamp = cast("dict[str, Any]", stamp)
        seconds, microseconds = stamp.get("seconds"), stamp.get("microseconds")
        if isinstance(seconds, int) and isinstance(microseconds, int):
            return seconds + microseconds / 1_000_000
    return None


def _event_monotonic(stamp: float | None, *, now: float, samples: Sequence[tuple[float, float]]) -> float:
    """time.monotonic() of an event stamped `stamp` on the wall clock, read at `now`
    (monotonic). `samples` are the event watch's (time.monotonic(), time.time() -
    time.monotonic()) pairs, oldest first, the last taken at this read. With no step among
    them, a stamp after the first sample (or held by QEMU's rate limit just before it,
    _HELD_EVENT_SECONDS) converts exactly, however late the read (a busy loop); with a step
    among them, an older stamp, or no stamp, the event is taken as made now. Never later
    than now."""
    if stamp is None or not samples:
        return now
    if any(
        abs(later_offset - offset) > _CLOCK_STEP_SECONDS + _CLOCK_DRIFT_PER_SECOND * (later - at_sample)
        for (at_sample, offset), (later, later_offset) in itertools.pairwise(samples)
    ):
        return now
    at = stamp - samples[-1][1]
    return now if at < samples[0][0] - _HELD_EVENT_SECONDS else min(now, at)


_WATCH_RETRY_FIRST_SECONDS: Final = 0.001
"""First pause before the event watch connects again (doubling to _WATCH_RETRY_MAX_SECONDS)."""
_WATCH_RETRY_MAX_SECONDS: Final = 1.0


@dataclass(frozen=True)
class QmpEvent:
    """One event of the monitor: its name, its data, and when QEMU made it (time.monotonic())."""

    name: str
    data: dict[str, Any]
    at: float


class QmpEventWatch:
    """The event monitor of one QEMU: `on_event` gets each event; `on_connect` runs at each
    new connection (query: the state of the time before it). Ends when QEMU exits (EOF, or
    `qemu_exit_code` set), on an access error (a socket of another user, a wrong peer), or at
    stop. Never raises."""

    def __init__(
        self,
        *,
        path: Path,
        expected_uid: int,
        vm_id: str,
        qemu_exit_code: Callable[[], int | None],
        on_event: Callable[[QmpEvent], Awaitable[None]],
        on_connect: Callable[[QmpEventWatch], Awaitable[None]],
    ) -> None:
        self._path = path
        self._expected_uid = expected_uid
        self._vm_id = vm_id
        self._qemu_exit_code = qemu_exit_code
        self._on_event = on_event
        self._on_connect = on_connect
        self._client: QmpClient | None = None
        self._task: asyncio.Task[None] | None = None
        # (time.monotonic(), time.time() - time.monotonic()) samples, across connections (_sample)
        self._samples: deque[tuple[float, float]] = deque()

    def start(self) -> None:
        """Start the watch, once."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"qemu-events-{self._vm_id}")

    async def stop(self) -> asyncio.CancelledError | None:
        """Cancel the watch and wait for its end (it closes its socket). A cancel of the
        caller meanwhile is returned, not raised, so the owner finishes its own end first."""
        cancellation: asyncio.CancelledError | None = None
        if self._task is not None:
            self._task.cancel()
            # asyncio.wait, not await_settled: the watch's own cancel is no caller cancel
            while not self._task.done():
                try:
                    await asyncio.wait({self._task})
                except asyncio.CancelledError as error:
                    cancellation = cancellation or error
        return cancellation

    async def query(self, command: str) -> dict[str, Any]:
        """Run `command` on the current connection (on_connect): the events that come before
        its answer go to on_event, as any other."""
        if self._client is None:
            raise ConnectionResetError("QMP event watch not connected")
        return await self._client.execute(command, arguments=None, timeout=math.inf, on_event=self._deliver)

    async def _run(self) -> None:
        """Read events until QEMU exits. The connect waits for the socket as long as QEMU runs
        (a slow start under load must not end the watch). A line that is not JSON is logged
        and skipped; a connection that fails while QEMU runs (not at its exit: EOF) is made
        again, after 1 ms doubling to 1 s; only an access error ends the watch while QEMU runs."""

        def qemu_exited() -> None:
            if (code := self._qemu_exit_code()) is not None:
                raise ConnectionResetError(f"QEMU exited with code {code}")

        delay, failures = _WATCH_RETRY_FIRST_SECONDS, 0
        connected_at: float | None = None
        while self._qemu_exit_code() is None:
            try:
                # No deadline: QEMU's exit ends the wait (abort_check), and no reply of a
                # running QEMU is late for a watch that only reads
                reader, writer = await wait_for_socket(
                    self._path, timeout=math.inf, abort_check=qemu_exited, keep_connection=True
                )
                client = QmpClient(reader, writer, error_factory=ConnectionResetError, id_prefix="events")
                try:
                    verify_socket_peer(writer.get_extra_info("socket"), self._expected_uid, str(self._path))
                    await client.start(timeout=math.inf)
                    self._sample()  # the reference of this connection's first event
                    connected_at = time.monotonic()
                    self._client = client
                    await self._on_connect(self)
                    while line := await client.reader.readline():
                        await self._deliver(line)
                    return  # EOF: QEMU closes its monitor only when it exits
                finally:
                    self._client = None
                    await client.close()
            except (ConfigError, CommunicationError) as error:  # no access, a wrong peer: no retry helps
                logger.warning(
                    "QEMU event watch stopped",
                    extra={"vm_id": self._vm_id, "error": str(error), "error_type": type(error).__name__},
                )
                return
            except (OSError, ValueError, SandboxError) as error:
                if self._qemu_exit_code() is not None:
                    return
                if connected_at is not None and time.monotonic() - connected_at >= _WATCH_RETRY_MAX_SECONDS:
                    delay, failures = _WATCH_RETRY_FIRST_SECONDS, 0  # it lasted: a new failure, start over
                connected_at = None
                failures += 1
                (logger.warning if failures == 1 else logger.debug)(
                    "QEMU event connection lost, connecting again",
                    extra={
                        "vm_id": self._vm_id,
                        "error": str(error),
                        "error_type": type(error).__name__,
                        "failures": failures,
                    },
                )
                # A failure that repeats while QEMU runs must not spin: 1 ms, doubling to 1 s
                await asyncio.sleep(delay)
                delay = min(delay * 2, _WATCH_RETRY_MAX_SECONDS)

    async def _deliver(self, line: bytes) -> None:
        """One line of the monitor to on_event, with its time; a line that is not a JSON
        object with an event name is logged (unreadable) or skipped."""
        try:
            message = json.loads(line)
        except ValueError:
            logger.warning("QEMU event line unreadable, skipped", extra={"vm_id": self._vm_id, "line": line[:200]})
            return
        if not isinstance(message, dict):
            return
        message = cast("dict[str, Any]", message)
        name, data = message.get("event"), message.get("data")
        if not isinstance(name, str):
            return
        await self._on_event(
            QmpEvent(
                name=name,
                data=cast("dict[str, Any]", data) if isinstance(data, dict) else {},
                at=self._event_time(message),
            )
        )

    def _sample(self) -> float:
        """Keep (time.monotonic(), time.time() - time.monotonic()) now, across connections;
        drop the samples older than _OFFSET_HISTORY_SECONDS but the newest of them (the
        reference before a held stamp). Returns the time.monotonic() of the sample."""
        now = time.monotonic()
        self._samples.append((now, time.time() - now))
        while len(self._samples) > 1 and self._samples[1][0] <= now - _OFFSET_HISTORY_SECONDS:
            self._samples.popleft()
        return now

    def _event_time(self, message: dict[str, Any]) -> float:
        """time.monotonic() of a QMP event (_event_monotonic against the kept samples)."""
        now = self._sample()
        return _event_monotonic(_stamp_of(message), now=now, samples=self._samples)

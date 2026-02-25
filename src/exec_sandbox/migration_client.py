"""QMP client for QEMU migration (save/restore memory snapshots).

Provides L1 memory snapshot operations via QEMU's migration subsystem.
Save captures full VM state (CPU + RAM + device) to file; restore resumes
from the exact instruction.

Architecture:
- Connects to QEMU's QMP (QEMU Monitor Protocol) Unix socket
- Uses socket peer credential authentication (same as balloon_client)
- Save: `stop` (pause vCPU, avoid HVF ARM64 ISV race) →
        `migrate file:/path` → poll until completed → `quit`
- Restore: QEMU started with `-incoming defer` → `migrate-incoming file:/path` → `cont`

Usage:
    async with MigrationClient(qmp_socket_path, expected_uid) as client:
        await client.save_snapshot(vmstate_path)  # VM is dead after this

    # For restore, connect to a QEMU started with -incoming defer:
    async with MigrationClient(qmp_socket_path, expected_uid) as client:
        await client.restore_snapshot(vmstate_path)  # VM resumes
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any, Self

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import MigrationTransientError
from exec_sandbox.socket_auth import connect_and_verify

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


class MigrationClient:
    """QMP client for QEMU migration (save/restore memory snapshots).

    Connection lifecycle is protected by an asyncio lock (connect/close).
    _execute() is NOT concurrency-safe: callers must not issue commands
    concurrently on the same client (the QMP protocol is sequential).
    Follows balloon_client.py pattern: connect_and_verify(), QMP handshake,
    _execute() with event-skipping loop.

    Attributes:
        qmp_socket: Path to QMP Unix socket
        expected_uid: Expected UID of QEMU process (for socket auth)
    """

    __slots__ = (
        "_connected",
        "_expected_uid",
        "_lock",
        "_qmp_socket",
        "_reader",
        "_writer",
    )

    def __init__(self, qmp_socket: Path, expected_uid: int) -> None:
        self._qmp_socket = qmp_socket
        self._expected_uid = expected_uid
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self, timeout: float = 5.0) -> None:
        """Connect to QMP socket and complete handshake.

        Raises:
            MigrationTransientError: Connection or handshake failed
        """
        async with self._lock:
            if self._connected:
                return

            try:
                self._reader, self._writer = await connect_and_verify(
                    str(self._qmp_socket),
                    self._expected_uid,
                    timeout=timeout,
                )

                # QMP handshake: Read greeting, send capabilities
                greeting = await asyncio.wait_for(
                    self._reader.readline(),
                    timeout=timeout,
                )
                logger.debug(
                    "QMP greeting received (migration)",
                    extra={"greeting": greeting.decode().strip()},
                )

                # Complete capabilities negotiation
                self._writer.write(b'{"execute": "qmp_capabilities"}\n')
                await self._writer.drain()

                response = await asyncio.wait_for(
                    self._reader.readline(),
                    timeout=timeout,
                )
                resp_data = json.loads(response)
                if "error" in resp_data:
                    raise MigrationTransientError(f"QMP capabilities failed: {resp_data['error']}")

                self._connected = True
                logger.debug("QMP connection established (migration)")

            except (OSError, TimeoutError, json.JSONDecodeError) as e:
                await self._cleanup()
                raise MigrationTransientError(f"QMP connection failed: {e}") from e

    async def close(self) -> None:
        """Close the QMP connection."""
        async with self._lock:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Internal cleanup (must hold lock)."""
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(TimeoutError, OSError):
                await asyncio.wait_for(self._writer.wait_closed(), timeout=1.0)
        self._reader = None
        self._writer = None
        self._connected = False

    async def _execute(
        self,
        command: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = constants.MEMORY_SNAPSHOT_QMP_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Execute QMP command, skipping async events.

        Raises:
            MigrationTransientError: Command failed or not connected
        """
        if not self._connected or self._writer is None or self._reader is None:
            raise MigrationTransientError("Not connected to QMP")

        cmd: dict[str, Any] = {"execute": command}
        if arguments:
            cmd["arguments"] = arguments

        logger.debug("QMP command: %s args=%s", command, arguments)
        self._writer.write(json.dumps(cmd).encode() + b"\n")
        await self._writer.drain()

        # Skip async events (MIGRATION_PASS, STOP, RESUME, etc.) until command response.
        # Cap at 32 iterations as safety rail; the outer asyncio.timeout is the primary guard.
        async with asyncio.timeout(timeout):
            for _ in range(32):
                response = await self._reader.readline()
                if not response:
                    raise MigrationTransientError("QMP connection closed unexpectedly")
                data: dict[str, Any] = json.loads(response)
                if "event" not in data:
                    logger.debug("QMP response: %s -> %s", command, data)
                    return data
                logger.debug("QMP event: %s data=%s", data.get("event"), data.get("data"))
        raise MigrationTransientError("Too many QMP events without command response")

    async def _set_capabilities(self, qemu_version: tuple[int, int, int] | None) -> None:
        """Configure migration capabilities for file: transport.

        On QEMU >= 9.0, enables mapped-ram + multifd for parallel page I/O.
        mapped-ram maps RAM pages to fixed offsets in the file (seekable),
        multifd enables multi-threaded transfer. Together they significantly
        speed up save/restore for file:-based migration.

        Both sender (save) and receiver (restore) MUST set the same
        capabilities, otherwise QEMU rejects with "received capability is off".
        """
        if qemu_version is None or qemu_version < constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION:
            return

        capabilities = [
            {"capability": "multifd", "state": True},
            {"capability": "mapped-ram", "state": True},
        ]
        resp = await self._execute(
            "migrate-set-capabilities",
            {"capabilities": capabilities},
        )
        if "error" in resp:
            # Non-fatal: QEMU will use legacy precopy streaming format (no mapped-ram).
            # MEMORY_SNAPSHOT_FORMAT_VERSION in the cache key prevents stale entries.
            logger.warning(
                "Failed to set migration capabilities, falling back to legacy streaming",
                extra={"error": resp["error"], "qemu_version": qemu_version},
            )

    async def save_snapshot(
        self,
        vmstate_path: Path,
        qemu_version: tuple[int, int, int] | None = None,
        timeout: float = constants.MEMORY_SNAPSHOT_SAVE_TIMEOUT_SECONDS,
    ) -> None:
        """Save VM state to file. VM pauses and then quit is sent.

        After this call, the QEMU process will exit.

        With mapped-ram + multifd (QEMU >= 9.0), QEMU can finish writing the
        vmstate file and exit before our first query-migrate poll. If the connection
        is lost during polling but the vmstate file exists with non-zero size,
        treat it as success (QEMU exited after completing the save).

        Raises:
            MigrationTransientError: Save failed
        """
        await self._set_capabilities(qemu_version)

        # Stop vCPU before migration to avoid HVF assertion crash.
        # On ARM64 HVF, if the vCPU is running when migration starts, the pause
        # can race with a guest data abort that has ISV=0 (Instruction Syndrome
        # Valid = false). QEMU's hvf_handle_exception asserts isv==true, causing
        # SIGABRT. Stopping the vCPU first eliminates the race.
        # See: https://github.com/qemu/qemu/blob/master/target/arm/hvf/hvf.c
        resp = await self._execute("stop")
        if "error" in resp:
            raise MigrationTransientError(f"stop command failed: {resp['error']}")

        # Start migration to file
        uri = f"file:{vmstate_path}"
        resp = await self._execute("migrate", {"uri": uri})
        if "error" in resp:
            raise MigrationTransientError(f"migrate command failed: {resp['error']}")

        # Poll until migration completes.
        # With mapped-ram + multifd, QEMU can finish saving and exit before our
        # first query-migrate poll. This causes ConnectionResetError (socket level)
        # or MigrationTransientError("QMP connection closed unexpectedly") from _execute().
        # If the vmstate file exists with non-zero size, treat as success.
        try:
            await self._poll_migration(timeout)
        except (MigrationTransientError, ConnectionResetError, BrokenPipeError, OSError, TimeoutError) as e:
            if vmstate_path.exists() and vmstate_path.stat().st_size > 0:
                logger.info(
                    "QEMU exited during migration poll (save likely completed)",
                    extra={"vmstate_path": str(vmstate_path), "error": str(e)},
                )
            else:
                raise MigrationTransientError(f"Connection lost during save and no vmstate file: {e}") from e

        # Send quit to terminate QEMU (VM state is saved)
        with contextlib.suppress(MigrationTransientError, OSError):
            await self._execute("quit", timeout=2.0)

        logger.info("VM state saved", extra={"vmstate_path": str(vmstate_path)})

    async def restore_snapshot(
        self,
        vmstate_path: Path,
        qemu_version: tuple[int, int, int] | None = None,
        timeout: float = constants.MEMORY_SNAPSHOT_RESTORE_TIMEOUT_SECONDS,
    ) -> None:
        """Restore VM from file. Precondition: QEMU started with -incoming defer.

        After this call, the VM is running (cont sent).

        Raises:
            MigrationTransientError: Restore failed
        """
        await self._set_capabilities(qemu_version)

        # Load migration stream
        uri = f"file:{vmstate_path}"
        resp = await self._execute("migrate-incoming", {"uri": uri})
        if "error" in resp:
            raise MigrationTransientError(f"migrate-incoming command failed: {resp['error']}")

        # Poll until migration completes
        await self._poll_migration(timeout)

        # Resume the VM
        resp = await self._execute("cont")
        if "error" in resp:
            raise MigrationTransientError(f"cont command failed: {resp['error']}")

        logger.info("VM state restored", extra={"vmstate_path": str(vmstate_path)})

    async def _poll_migration(self, timeout: float) -> None:
        """Poll query-migrate at 50ms intervals until completed/failed.

        Raises:
            MigrationTransientError: Migration failed or timed out
        """
        async with asyncio.timeout(timeout):
            while True:
                resp = await self._execute("query-migrate", timeout=5.0)
                if "error" in resp:
                    raise MigrationTransientError(f"query-migrate failed: {resp['error']}")

                status = resp.get("return", {}).get("status", "")
                if status == "completed":
                    return
                if status in ("failed", "cancelled"):
                    error_desc = resp.get("return", {}).get("error-desc", "unknown")
                    raise MigrationTransientError(f"Migration {status}: {error_desc}")

                await asyncio.sleep(constants.MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS)

    async def __aenter__(self) -> Self:
        """Enter async context manager, connecting to QMP."""
        await self.connect()
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, closing the QMP connection."""
        await self.close()

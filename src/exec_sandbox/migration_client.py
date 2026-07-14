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
from exec_sandbox.aio_utils import await_cancellation_safe
from exec_sandbox.exceptions import MigrationTransientError
from exec_sandbox.qmp_exchange import qmp_id_exchange
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
        "_request_seq",
        "_writer",
    )

    def __init__(self, qmp_socket: Path, expected_uid: int) -> None:
        self._qmp_socket = qmp_socket
        self._expected_uid = expected_uid
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._request_seq = 0
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

            except BaseException as error:  # Every failed handshake owns socket cleanup.
                await self._cleanup_after_connect_failure()
                if isinstance(error, asyncio.CancelledError | MigrationTransientError):
                    raise
                if isinstance(error, OSError | TimeoutError | json.JSONDecodeError):
                    raise MigrationTransientError(f"QMP connection failed: {error}") from error
                raise

    async def close(self) -> None:
        """Close the QMP connection."""
        async with self._lock:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Internal cleanup (must hold lock)."""
        writer = self._writer
        self._reader = None
        self._writer = None
        self._connected = False
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)

    async def _cleanup_after_connect_failure(self) -> None:
        """Finish handshake cleanup despite repeated caller cancellation.

        Runs cleanup to completion and surfaces its own error, but does not
        drop a caller cancellation observed mid-cleanup: await_cancellation_safe
        re-raises it so a Ctrl-C during teardown still wins over the original
        handshake error.
        """
        await await_cancellation_safe(self._cleanup())

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

        # Tag each command with a unique id and match it on the response. The
        # save path interleaves stop/migrate/query-migrate where a shifted
        # response is corrupting, not benign: id-matching discards async events
        # AND any stale reply from a prior timed-out command.
        logger.debug("QMP command: %s args=%s", command, arguments)
        self._request_seq += 1
        return await qmp_id_exchange(
            self._reader,
            self._writer,
            f"migration-{self._request_seq}",
            command,
            arguments,
            timeout,
            MigrationTransientError,
        )

    async def _set_capabilities(
        self,
        qemu_version: tuple[int, int, int] | None,
        use_template: bool = False,
    ) -> None:
        """Configure migration capabilities for file: transport.

        On QEMU >= 9.0, enables mapped-ram + multifd for parallel page I/O.
        mapped-ram maps RAM pages to fixed offsets in the file (seekable),
        multifd enables multi-threaded transfer.

        When use_template=True, also enables x-ignore-shared: QEMU skips RAM
        pages backed by shared memory during migration, saving only device state.
        This is the key mechanism for VM Templating (COW memory sharing).
        Ref: https://www.qemu.org/docs/master/system/vm-templating.html

        Both sender (save) and receiver (restore) MUST set the same
        capabilities, otherwise QEMU rejects with "received capability is off".

        NOTE (design constraints, verified QEMU 11.0):
        - mapped-ram is INCOMPATIBLE with multifd-compression (seekable-fd
          requirement) AND with migration TLS. Any snapshot compression or
          encryption for a future S3 freeze/thaw path must therefore be done
          EXTERNALLY (client-side zstd + AEAD), not via QEMU capabilities. The
          sanctioned in-QEMU perf recovery is the `direct-io` migration parameter
          (O_DIRECT), gated to the full-RAM path when that lands.
        - CPR (cpr-reboot/transfer/exec) is NOT a resume-anywhere primitive: all
          modes are same-host only (fd/RAM handoff to a peer QEMU), so they cannot
          back S3 freeze/thaw. Their only use here would be in-place QEMU upgrades.
        - The migration stream format is QEMU-version-specific; a QEMU bump
          invalidates saved snapshots (see MEMORY_SNAPSHOT_FORMAT_VERSION).
        """
        if qemu_version is None or qemu_version < constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION:
            return

        # Unified capability list (QEMU >= 11.0). Base is mapped-ram + multifd for
        # seekable, parallel page I/O. Template mode adds x-ignore-shared to skip
        # RAM pages backed by the shared memory-backend-file (device-state-only
        # vmstate). mapped-ram + x-ignore-shared coexist safely only since 11.0
        # (earlier releases null-deref on ignored blocks — the reason the floor is
        # 11.0), so both paths share one branch-free list.
        # Ref: QEMU commit "migration/mapped-ram: Fix x-ignore-shared snapshots"
        #      (Pawel Zmarzly, Nov 2025; reviewed by Peter Xu, Dec 2025)
        capabilities = [
            {"capability": "multifd", "state": True},
            {"capability": "mapped-ram", "state": True},
        ]
        if use_template:
            capabilities.append({"capability": "x-ignore-shared", "state": True})
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
        use_template: bool = False,
    ) -> None:
        """Save VM state to file. VM pauses and then quit is sent.

        After this call, the QEMU process will exit.

        When use_template=True, x-ignore-shared is enabled: QEMU saves only
        device state (CPU registers, virtio rings, etc.), skipping RAM pages
        that are backed by shared memory-backend-file. The resulting vmstate
        is small (~100KB-1MB) instead of the full guest memory dump.

        With mapped-ram + multifd (QEMU >= 9.0), QEMU can finish writing the
        vmstate file and exit before our first query-migrate poll. If the connection
        is lost during polling but the vmstate file exists with non-zero size,
        treat it as success (QEMU exited after completing the save).

        Raises:
            MigrationTransientError: Save failed
        """
        await self._set_capabilities(qemu_version, use_template=use_template)

        # Stop vCPU before migration to avoid HVF assertion crash.
        # On ARM64 HVF, if the vCPU is running when migration starts, the pause
        # can race with a guest data abort that has ISV=0 (Instruction Syndrome
        # Valid = false). QEMU's hvf_handle_exception asserts isv==true, causing
        # SIGABRT. Stopping the vCPU first eliminates the race.
        # See: https://github.com/qemu/qemu/blob/master/target/arm/hvf/hvf.c
        #
        # NOTE (non-adoption): the background-snapshot capability (live snapshot
        # without pausing the vCPU) is not used. It is KVM-only (needs userfaultfd
        # write-protect) so excluded on HVF, and we snapshot idle sessions where a
        # brief stop is cheap and full-fidelity capture wants a quiescent VM anyway.
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
        use_template: bool = False,
    ) -> None:
        """Restore VM from file. Precondition: QEMU started with -incoming defer.

        After this call, the VM is running (cont sent).

        When use_template=True, x-ignore-shared is enabled: QEMU loads only
        device state from the vmstate file, skipping RAM (already present via
        file-backed memory-backend-file with MAP_PRIVATE COW mapping).

        Raises:
            MigrationTransientError: Restore failed
        """
        await self._set_capabilities(qemu_version, use_template=use_template)

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
        """Poll query-migrate at 5ms intervals until completed/failed.

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
                # "failing" (QEMU 11.0+): an error occurred and cleanup is underway;
                # the status will transition to "failed" shortly. Treat it as terminal
                # here so a mid-stream failure surfaces immediately instead of burning
                # another poll interval waiting for the "failed" transition.
                if status in ("failed", "cancelled", "failing"):
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

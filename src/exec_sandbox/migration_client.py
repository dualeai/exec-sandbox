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
from exec_sandbox.exceptions import (
    CommunicationError,
    ConfigError,
    ErrorCode,
    RouteError,
    SandboxError,
    TransientError,
)
from exec_sandbox.qmp_exchange import qmp_id_exchange
from exec_sandbox.socket_auth import connect_and_verify

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

logger = get_logger(__name__)

_CONNECT_TIMEOUT_SECONDS = 5.0
_QMP_TIMEOUT = constants.MEMORY_SNAPSHOT_QMP_TIMEOUT_SECONDS


def rejected_by_qemu(error: BaseException) -> bool:
    """A MIGRATION_REJECTED or MIGRATION_UNSUPPORTED error anywhere in the cause/context
    chain (a QEMU that died while refusing the stream is reported by an error raised
    without `from`). By code: a setup failure of the restore (HOST_DEPENDENCY, for
    example) must not count as a refusal of the saved state."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, SandboxError) and current.code in _REFUSED:
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


_REFUSED = frozenset({ErrorCode.MIGRATION_REJECTED, ErrorCode.MIGRATION_UNSUPPORTED})


def _failed(message: str, *, context: dict[str, Any] | None) -> TransientError:
    """The TransientError (MIGRATION_FAILED) of a save or load that QEMU did not finish."""
    return TransientError(message, code=ErrorCode.MIGRATION_FAILED, context=context, hint=None, instance_id=None)


def _unsupported(message: str) -> ConfigError:
    """The ConfigError (MIGRATION_UNSUPPORTED) of a QEMU that cannot save or load VM state."""
    return ConfigError(message, code=ErrorCode.MIGRATION_UNSUPPORTED, context=None, hint=None, instance_id=None)


def _rejected(message: str, *, qemu_version: tuple[int, int, int] | None) -> RouteError:
    """The RouteError (MIGRATION_REJECTED) of a saved state this QEMU refuses."""
    found = ".".join(map(str, qemu_version)) if qemu_version else "unknown"
    return RouteError(
        message,
        code=ErrorCode.MIGRATION_REJECTED,
        required={"qemu_version": "a QEMU that accepts this saved state"},
        found={"qemu_version": found},
        context=None,
        hint=None,
        instance_id=None,
    )


def migration_timeout(memory_mb: float) -> float:
    """Save/restore timeout: a fixed base plus the time to move the whole guest RAM at a slow disk rate."""
    return constants.MIGRATION_TIMEOUT_BASE_SECONDS + memory_mb / constants.MIGRATION_MIN_THROUGHPUT_MB_PER_SECOND


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

    async def connect(self, *, timeout: float) -> None:
        """Connect to QMP socket and complete handshake.

        Raises:
            TransientError: MIGRATION_FAILED: connection, peer authentication or handshake failed
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
                    raise _failed(f"QMP capabilities failed: {resp_data['error']}", context=None)

                self._connected = True
                logger.debug("QMP connection established (migration)")

            except BaseException as error:  # Every failed handshake owns socket cleanup.
                await self._cleanup_after_connect_failure()
                if isinstance(error, asyncio.CancelledError | SandboxError):
                    raise
                # CommunicationError: SocketAuthError, internal; the guest channel wraps it the same way.
                if isinstance(error, OSError | TimeoutError | json.JSONDecodeError | CommunicationError):
                    raise _failed(f"QMP connection failed: {error}", context=None) from error
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
        *,
        arguments: dict[str, Any] | None,
        timeout: float,
    ) -> dict[str, Any]:
        """Execute QMP command, skipping async events.

        Raises:
            TransientError: MIGRATION_FAILED: command failed or not connected
        """
        if not self._connected or self._writer is None or self._reader is None:
            raise _failed("Not connected to QMP", context=None)

        # Tag each command with a unique id and match it on the response. The
        # save path interleaves stop/migrate/query-migrate where a shifted
        # response is corrupting, not benign: id-matching discards async events
        # AND any stale reply from a prior timed-out command.
        logger.debug("QMP command: %s args=%s", command, arguments)
        self._request_seq += 1
        try:
            return await qmp_id_exchange(
                self._reader,
                self._writer,
                request_id=f"migration-{self._request_seq}",
                command=command,
                arguments=arguments,
                timeout=timeout,
                error_factory=lambda message: _failed(message, context=None),
            )
        except (TimeoutError, json.JSONDecodeError) as e:
            raise _failed(f"QMP {command} failed: {e!r}", context=None) from e

    async def _set_capabilities(
        self,
        qemu_version: tuple[int, int, int] | None,
        *,
        use_template: bool,
    ) -> None:
        """Configure migration capabilities for file: transport.

        mapped-ram maps RAM pages to fixed offsets in the file (seekable),
        multifd enables multi-threaded transfer. x-ignore-shared is always sent
        with an explicit state: on for templates (QEMU skips RAM pages backed by
        the shared memory-backend-file and saves only device state), off
        otherwise, so a capability left on by an earlier command cannot leak in.
        Ref: https://www.qemu.org/docs/master/system/vm-templating.html

        Both sender (save) and receiver (restore) MUST set the same
        capabilities, otherwise QEMU rejects with "received capability is off".
        There is no legacy fallback: a QEMU below the floor, or one that refuses
        a capability, raises ConfigError MIGRATION_UNSUPPORTED.

        NOTE (design constraints, verified QEMU 11.0):
        - mapped-ram does not work with multifd-compression (seekable-fd
          requirement) or with migration TLS. Saved instances are compressed and
          encrypted outside QEMU (zstd + AEAD in exec_sandbox.instance), not via
          QEMU capabilities.
        - CPR (cpr-reboot/transfer/exec) is not a resume-anywhere primitive: all
          modes are same-host only (fd/RAM handoff to a peer QEMU).
        - The migration stream can change between QEMU versions. The L1 cache key
          includes the QEMU version (see MEMORY_SNAPSHOT_FORMAT_VERSION). A saved
          instance has no such check: QEMU accepts or refuses its state at
          restore (then RouteError INSTANCE_HOST_MISMATCH).

        Raises:
            ConfigError: MIGRATION_UNSUPPORTED: QEMU below the floor or capability refused
        """
        if qemu_version is None or qemu_version < constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION:
            floor = ".".join(map(str, constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION))
            found = ".".join(map(str, qemu_version)) if qemu_version else "unknown"
            raise _unsupported(f"QEMU {found} is below the migration floor {floor}")

        # mapped-ram + x-ignore-shared coexist safely only since 11.0 (earlier
        # releases null-deref on ignored blocks — the reason the floor is 11.0).
        # Ref: QEMU commit "migration/mapped-ram: Fix x-ignore-shared snapshots"
        #      (Pawel Zmarzly, Nov 2025; reviewed by Peter Xu, Dec 2025)
        capabilities = [
            {"capability": "multifd", "state": True},
            {"capability": "mapped-ram", "state": True},
            {"capability": "x-ignore-shared", "state": use_template},
        ]
        resp = await self._execute(
            "migrate-set-capabilities",
            arguments={"capabilities": capabilities},
            timeout=_QMP_TIMEOUT,
        )
        if "error" in resp:
            raise _unsupported(f"migrate-set-capabilities failed: {resp['error']}")

    async def save_snapshot(
        self,
        vmstate_path: Path,
        *,
        qemu_version: tuple[int, int, int] | None,
        timeout: float,
        use_template: bool,
        keep_running: bool,
        while_paused: Callable[[], Awaitable[None]] | None,
    ) -> None:
        """Save VM state to file.

        The vCPUs stop, QEMU writes the state, and success is reported only when
        query-migrate shows "completed". `while_paused` then runs with the VM
        still stopped (e.g. copy a disk that must match the saved RAM). Then QEMU
        quits, or, with keep_running=True, the VM resumes (cont) and can be saved
        again later.
        On a failure after `stop`, the VM resumes, best effort: QEMU can refuse
        `cont` (for example while the migration finalizes), so the VM can stay
        paused. A SandboxError raised there gets context["vm_resumed"] (True when
        `cont` succeeded). Keep the VM only then; replace it after any other
        failed save.

        When use_template=True, x-ignore-shared is enabled: QEMU saves only
        device state (CPU registers, virtio rings, etc.), skipping RAM pages
        that are backed by shared memory-backend-file. The resulting vmstate
        is small (~100KB-1MB) instead of the full guest memory dump.

        Raises:
            ConfigError: MIGRATION_UNSUPPORTED: QEMU below the floor or capability refused
            TransientError: MIGRATION_FAILED: save failed
            Any error that `while_paused` raises (the VM resumes first, best effort).
        """
        try:
            await self._set_capabilities(qemu_version, use_template=use_template)
        except SandboxError as e:
            e.context["vm_resumed"] = True  # nothing stopped it: the VM runs on
            raise

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
        resp = await self._execute("stop", arguments=None, timeout=_QMP_TIMEOUT)
        if "error" in resp:  # QEMU refused to stop: the VM runs on
            raise _failed(f"stop command failed: {resp['error']}", context={"vm_resumed": True})

        try:
            resp = await self._execute("migrate", arguments={"uri": f"file:{vmstate_path}"}, timeout=_QMP_TIMEOUT)
            if "error" in resp:
                raise _failed(f"migrate command failed: {resp['error']}", context=None)
            await self._poll_migration(timeout, lambda message: _failed(message, context=None))
            if while_paused is not None:
                await while_paused()
        except Exception as e:
            resumed = await self._resume_after_failed_save()
            if not resumed:  # the VM may stay paused, whatever failed: the caller must replace it
                raise _failed(f"Save failed and the VM did not resume: {e}", context={"vm_resumed": False}) from e
            if isinstance(e, SandboxError):
                e.context["vm_resumed"] = True  # the VM runs on: a caller may keep it
            raise

        if keep_running:
            resp = await self._execute("cont", arguments=None, timeout=_QMP_TIMEOUT)
            if "error" in resp:
                raise _failed(f"cont after save failed: {resp['error']}", context=None)
        else:
            # QEMU may close the socket before it answers quit.
            with contextlib.suppress(TransientError, OSError):
                await self._execute("quit", timeout=2.0, arguments=None)

        logger.info("VM state saved", extra={"vmstate_path": str(vmstate_path), "keep_running": keep_running})

    async def _resume_after_failed_save(self) -> bool:
        """Leave the VM running: cancel a migration that may still be active
        (timeout), then resume the vCPUs that save_snapshot stopped. True when
        QEMU accepted `cont`; False: the VM can stay paused (e.g. cont refused
        while the migration finalizes), so the caller replaces it."""
        resumed = False
        for command in ("migrate_cancel", "cont"):
            with contextlib.suppress(TransientError, OSError):
                resp = await self._execute(command, timeout=2.0, arguments=None)
                if "error" in resp:
                    logger.warning(
                        "QMP command after failed save refused", extra={"command": command, "error": resp["error"]}
                    )
                elif command == "cont":
                    resumed = True
        return resumed

    async def restore_snapshot(
        self,
        vmstate_path: Path,
        *,
        qemu_version: tuple[int, int, int] | None,
        timeout: float,
        use_template: bool,
    ) -> None:
        """Restore VM from file. Precondition: QEMU started with -incoming defer.

        After this call, the VM is running (cont sent). exit-on-error=false keeps
        QEMU alive when it rejects the stream, so the error text is read from
        query-migrate and the caller owns the teardown.

        When use_template=True, x-ignore-shared is enabled: QEMU loads only
        device state from the vmstate file, skipping RAM (already present via
        file-backed memory-backend-file with MAP_PRIVATE COW mapping).

        Raises:
            RouteError: MIGRATION_REJECTED: QEMU refused the saved state
            ConfigError: MIGRATION_UNSUPPORTED: QEMU below the floor or capability refused
            TransientError: MIGRATION_FAILED: QMP failure or timeout
        """
        await self._set_capabilities(qemu_version, use_template=use_template)

        resp = await self._execute(
            "migrate-incoming",
            arguments={"uri": f"file:{vmstate_path}", "exit-on-error": False},
            timeout=_QMP_TIMEOUT,
        )
        if "error" in resp:
            raise _rejected(f"migrate-incoming command failed: {resp['error']}", qemu_version=qemu_version)

        await self._poll_migration(timeout, lambda message: _rejected(message, qemu_version=qemu_version))

        # Resume the VM
        resp = await self._execute("cont", arguments=None, timeout=_QMP_TIMEOUT)
        if "error" in resp:
            raise _failed(f"cont command failed: {resp['error']}", context=None)

        logger.info("VM state restored", extra={"vmstate_path": str(vmstate_path)})

    async def _poll_migration(
        self,
        timeout: float,
        failure: Callable[[str], SandboxError],
    ) -> None:
        """Poll query-migrate at 5ms intervals until completed/failed.

        Raises:
            SandboxError: failure(message) when the migration reached
                failed/cancelled/failing (with QEMU's error-desc)
            TransientError: MIGRATION_FAILED: query-migrate failed or timed out
        """
        try:
            async with asyncio.timeout(timeout):
                while True:
                    resp = await self._execute("query-migrate", timeout=5.0, arguments=None)
                    if "error" in resp:
                        raise _failed(f"query-migrate failed: {resp['error']}", context=None)

                    status = resp.get("return", {}).get("status", "")
                    if status == "completed":
                        return
                    # "failing" (QEMU 11.0+): an error occurred and cleanup is underway;
                    # the status will transition to "failed" shortly. Treat it as terminal
                    # here so a mid-stream failure surfaces immediately instead of burning
                    # another poll interval waiting for the "failed" transition.
                    if status in ("failed", "cancelled", "failing"):
                        error_desc = resp.get("return", {}).get("error-desc", "unknown")
                        raise failure(f"Migration {status}: {error_desc}")

                    await asyncio.sleep(constants.MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS)
        except TimeoutError as e:
            # Only the deadline: _execute turns its own timeouts into MIGRATION_FAILED
            raise _failed(f"Migration did not complete within {timeout:.0f}s", context=None) from e

    async def __aenter__(self) -> Self:
        """Enter async context manager, connecting to QMP."""
        await self.connect(timeout=_CONNECT_TIMEOUT_SECONDS)
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, closing the QMP connection."""
        await self.close()

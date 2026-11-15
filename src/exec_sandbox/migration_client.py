"""QMP client for QEMU migration (save/restore memory snapshots).

Provides memory snapshot save and restore, via QEMU's migration subsystem, for
the L1 cache and for saved instances (Session.save, Scheduler.reuse).
Save captures full VM state (CPU + RAM + device) to file; restore resumes
from the exact instruction.

Architecture:
- Connects to QEMU's QMP (QEMU Monitor Protocol) Unix socket
- Uses socket peer credential authentication (socket_auth.connect_and_verify)
- Save: `stop` (pause vCPU, avoid HVF ARM64 ISV race) →
        `migrate file:/path` → poll until completed → `while_paused` hook (VM still
        stopped) → `quit` (or `cont` to keep running)
- Restore: QEMU started with `-incoming defer` →
        `migrate-incoming file:/path` (exit-on-error=false) → poll → `cont`
- Strict: success only on an observed "completed"; no fallback to another format.

Usage:
    async with MigrationClient(qmp_socket_path, expected_uid) as client:
        await client.save_snapshot(
            vmstate_path, qemu_version=v, timeout=migration_timeout(mb), while_paused=None,
            use_template=False, keep_running=False, memory_limited=True,
        )

    # For restore, connect to a QEMU started with -incoming defer:
    async with MigrationClient(qmp_socket_path, expected_uid) as client:
        await client.restore_snapshot(vmstate_path, qemu_version=v, timeout=migration_timeout(mb), use_template=False)
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
    RouteKey,
    SandboxError,
    TransientError,
)
from exec_sandbox.qmp_client import QMP_BUFFER_LIMIT, QmpClient
from exec_sandbox.socket_auth import connect_and_verify
from exec_sandbox.system_probes import probe_direct_io

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

logger = get_logger(__name__)

_CONNECT_TIMEOUT_SECONDS = 5.0
_QMP_TIMEOUT = constants.MEMORY_SNAPSHOT_QMP_TIMEOUT_SECONDS
_POLL_FIRST_SECONDS = 0.001  # query-migrate poll: 1 ms, doubling up to MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS


def qemu_refusal(error: BaseException) -> SandboxError | None:
    """The MIGRATION_REJECTED or MIGRATION_UNSUPPORTED error anywhere in the cause/context
    chain (a QEMU that died while refusing the stream is reported by an error raised
    without `from`): its message holds QEMU's reason. None without one. By code: a setup
    failure of the restore (HOST_DEPENDENCY, for example) must not count as a refusal of
    the saved state."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, SandboxError) and current.code in _REFUSED:
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


_REFUSED = frozenset({ErrorCode.MIGRATION_REJECTED, ErrorCode.MIGRATION_UNSUPPORTED})


def _failed(message: str, *, context: dict[str, Any] | None) -> TransientError:
    """The TransientError (MIGRATION_FAILED) of a save or load that QEMU did not finish."""
    return TransientError(message, code=ErrorCode.MIGRATION_FAILED, context=context, hint=None, instance_id=None)


def _unsupported(message: str) -> ConfigError:
    """The ConfigError (MIGRATION_UNSUPPORTED) of a QEMU that cannot save or load VM state."""
    return ConfigError(message, code=ErrorCode.MIGRATION_UNSUPPORTED, context=None, hint=None, instance_id=None)


def _rejected(message: str, *, qemu_version: tuple[int, int, int]) -> RouteError:
    """The RouteError (MIGRATION_REJECTED) of a saved state this QEMU refuses."""
    return RouteError(
        message,
        code=ErrorCode.MIGRATION_REJECTED,
        required={},  # the caller names the target: it knows who saved the state
        found={RouteKey.QEMU_VERSION: ".".join(map(str, qemu_version))},
        context=None,
        hint=None,
        instance_id=None,
    )


def migration_timeout(memory_mb: float) -> float:
    """Save/restore timeout: a fixed base plus the time to move the whole guest RAM at a slow disk rate."""
    return constants.MIGRATION_TIMEOUT_BASE_SECONDS + memory_mb / constants.MIGRATION_MIN_THROUGHPUT_MB_PER_SECOND


class MigrationClient:
    """QMP client for QEMU migration (save/restore memory snapshots).

    One user at a time (an `async with` block): callers must not issue commands
    concurrently on the same client (the QMP protocol is sequential).
    Pattern: connect_and_verify(), then a QmpClient (start: QMP handshake;
    execute: id-matched commands, events skipped).

    Args:
        qmp_socket: Path to QMP Unix socket
        expected_uid: Expected UID of QEMU process (for socket auth)
    """

    __slots__ = (
        "_expected_uid",
        "_qmp",
        "_qmp_socket",
    )

    def __init__(self, qmp_socket: Path, expected_uid: int) -> None:
        self._qmp_socket = qmp_socket
        self._expected_uid = expected_uid
        self._qmp: QmpClient | None = None  # set once the handshake ends

    async def connect(self, *, timeout: float) -> None:
        """Connect to QMP socket and complete handshake.

        Raises:
            TransientError: MIGRATION_FAILED: connection, peer authentication or handshake failed
        """
        qmp: QmpClient | None = None
        try:
            reader, writer = await connect_and_verify(
                str(self._qmp_socket), self._expected_uid, timeout=timeout, buffer_limit=QMP_BUFFER_LIMIT
            )
            qmp = QmpClient(
                reader, writer, error_factory=lambda message: _failed(message, context=None), id_prefix="migration"
            )
            await qmp.start(timeout=timeout)
        except BaseException as error:  # Every failed handshake owns socket cleanup.
            # Runs to its end; a cancel that comes meanwhile is raised after it (it wins
            # over the handshake error)
            if qmp is not None:
                await await_cancellation_safe(qmp.close())
            if isinstance(error, asyncio.CancelledError | SandboxError):
                raise
            # CommunicationError: SocketAuthError, internal; the guest channel wraps it the same way.
            if isinstance(error, OSError | TimeoutError | json.JSONDecodeError | CommunicationError):
                raise _failed(f"QMP connection failed: {error}", context=None) from error
            raise
        self._qmp = qmp
        logger.debug("QMP connection established (migration)")

    async def close(self) -> None:
        """Close the QMP connection."""
        qmp, self._qmp = self._qmp, None
        if qmp is not None:
            await qmp.close()

    async def _execute(
        self,
        command: str,
        *,
        arguments: dict[str, Any] | None,
        timeout: float,
    ) -> dict[str, Any]:
        """Execute QMP command, skipping async events. Returns the response: a command
        that QEMU refuses comes back with its ``error`` key, not raised.

        Raises:
            TransientError: MIGRATION_FAILED: not connected, the connection failed (reset,
                broken pipe), no response within `timeout`, a line that is not JSON, EOF, or
                too many skipped messages
        """
        if self._qmp is None:
            raise _failed("Not connected to QMP", context=None)

        # Tag each command with a unique id and match it on the response. The
        # save path interleaves stop/migrate/query-migrate where a shifted
        # response is corrupting, not benign: id-matching discards async events
        # AND any stale reply from a prior timed-out command.
        logger.debug("QMP command: %s args=%s", command, arguments)
        try:
            return await self._qmp.execute(command, arguments=arguments, timeout=timeout, on_event=None)
        except (OSError, json.JSONDecodeError) as e:  # OSError: TimeoutError, a reset or broken pipe
            raise _failed(f"QMP {command} failed: {e!r}", context=None) from e

    async def _set_capabilities(
        self,
        qemu_version: tuple[int, int, int],
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
        There is no fallback to another format: a QEMU below the floor, or one that
        refuses a capability, raises ConfigError MIGRATION_UNSUPPORTED.

        NOTE (design constraints, verified QEMU 11.0):
        - mapped-ram does not work with multifd-compression (seekable-fd
          requirement) or with migration TLS. Saved instances are compressed and
          encrypted outside QEMU (zstd + AEAD in exec_sandbox.instance), not via
          QEMU capabilities.
        - CPR (cpr-reboot/transfer/exec) is not a resume-anywhere primitive: all
          modes are same-host only (fd/RAM handoff to a peer QEMU).
        - The migration stream can change between QEMU versions. The L1 cache key
          includes the QEMU version and the device layout of the restore, which holds
          MIGRATION_CAPABILITIES (qemu_cmd.device_layout). A load of a saved instance
          compares its machine type and device layout before the download; QEMU accepts
          or refuses the rest of its state at restore (qemu_refusal: on a QEMU below the
          saved one, RouteError with the saved QEMU version).

        Raises:
            ConfigError: MIGRATION_UNSUPPORTED: QEMU below the floor or capability refused
        """
        if qemu_version < constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION:
            floor = ".".join(map(str, constants.MEMORY_SNAPSHOT_MIN_QEMU_VERSION))
            found = ".".join(map(str, qemu_version))
            raise _unsupported(f"QEMU {found} is below the migration floor {floor}")

        # mapped-ram + x-ignore-shared coexist safely only since 11.0 (earlier
        # releases null-deref on ignored blocks — the reason the floor is 11.0).
        # Ref: QEMU commit "migration/mapped-ram: Fix x-ignore-shared snapshots"
        #      (Pawel Zmarzly, Nov 2025; reviewed by Peter Xu, Dec 2025)
        capabilities = [
            *({"capability": name, "state": True} for name in constants.MIGRATION_CAPABILITIES),
            {"capability": "x-ignore-shared", "state": use_template},
        ]
        resp = await self._execute(
            "migrate-set-capabilities",
            arguments={"capabilities": capabilities},
            timeout=_QMP_TIMEOUT,
        )
        if "error" in resp:
            raise _unsupported(f"migrate-set-capabilities failed: {resp['error']}")

    async def _set_save_parameters(self, vmstate_path: Path, *, memory_limited: bool) -> None:
        """Parameters of a save (QEMU keeps them for the process; a restore runs in a new QEMU
        with the defaults, buffered: its clean page cache can be reclaimed).

        A save folder that takes O_DIRECT (system_probes.probe_direct_io, whatever the OS):
        direct-io (mapped-ram + multifd write guest RAM with O_DIRECT) and max-bandwidth 0 (no
        rate limit). A buffered save makes dirty page cache charged to the VM's memory cgroup,
        which cgroup reclaim cannot free in time: the VM was OOM-killed mid-save under disk load
        (measured on nested KVM, ext4: file_dirty 67 MB at a 272 MB limit). With direct-io the
        VM stays at guest + its normal overhead, and a slow disk fails the save on its deadline,
        the VM resumed.

        A folder without O_DIRECT (tmpfs before 6.6, ecryptfs, vboxsf, some FUSE, an OS with
        no O_DIRECT), for a VM that a memory cgroup limits (`memory_limited`): QEMU's defaults,
        buffered and paced at 128 MiB/s (unpaced buffered saves were OOM-killed). With no
        memory cgroup: max-bandwidth 0 only (QEMU's default paces every save: ~1.5 s for
        200 MB, 55 ms without it).

        macOS: QEMU has no direct-io for migration there ("No build-time support for
        direct-io", QEMU 11.1), and F_NOCACHE on a file this process would pass to QEMU
        (add-fd) gains nothing: measured on APFS (Apple M5 Pro), 1 GiB in 4 MiB writes took
        78 ms either way, kept 0 MB of page cache instead of 1024 MB, and its read back (a
        restore) took 99 ms instead of 42 ms. Without memory cgroups that cache is plain
        reclaimable memory, so saves there stay buffered.

        Raises:
            ConfigError: MIGRATION_UNSUPPORTED if QEMU refuses a parameter
        """
        if await probe_direct_io(vmstate_path.parent):
            parameters: dict[str, Any] = {"max-bandwidth": 0, "direct-io": True}
        elif memory_limited:
            return  # QEMU's defaults: buffered, paced
        else:
            parameters = {"max-bandwidth": 0}
        resp = await self._execute("migrate-set-parameters", arguments=parameters, timeout=_QMP_TIMEOUT)
        if "error" in resp:
            raise _unsupported(f"migrate-set-parameters failed: {resp['error']}")

    async def save_snapshot(
        self,
        vmstate_path: Path,
        *,
        qemu_version: tuple[int, int, int],
        timeout: float,
        use_template: bool,
        keep_running: bool,
        while_paused: Callable[[], Awaitable[None]] | None,
        memory_limited: bool,
    ) -> None:
        """Save VM state to file. `memory_limited`: a memory cgroup limits the VM
        (a VM with a Cgroup): without O_DIRECT its save stays paced.

        The vCPUs stop, QEMU writes the state, and success is reported only when
        query-migrate shows "completed". `while_paused` then runs with the VM
        still stopped (e.g. copy a disk that must match the saved RAM). Then QEMU
        quits, or, with keep_running=True, the VM resumes (cont) and can be saved
        again later.
        On a failure after `stop` (a cancel of the caller too), the VM resumes,
        best effort: QEMU can refuse `cont` (for example while the migration
        finalizes), so the VM can stay paused. A SandboxError raised there gets context["vm_resumed"] (True when
        `cont` succeeded). Keep the VM only then; replace it after any other
        failed save.

        When use_template=True, x-ignore-shared is enabled: QEMU saves only
        device state (CPU registers, virtio rings, etc.), skipping RAM pages
        that are backed by shared memory-backend-file. The resulting vmstate
        is small (~100KB-1MB) instead of the full guest memory dump.

        Raises:
            ConfigError: MIGRATION_UNSUPPORTED: QEMU below the floor or capability refused
            TransientError: MIGRATION_FAILED: save failed (a QMP connection that fails
                before `stop` too: the VM runs on, context vm_resumed), or `cont` failed after
                a save with keep_running=True
            Any error that `while_paused` raises (the VM resumes first, best effort; when
                it does not resume, TransientError MIGRATION_FAILED replaces the error).
        """
        try:
            await self._set_capabilities(qemu_version, use_template=use_template)
            await self._set_save_parameters(vmstate_path, memory_limited=memory_limited)
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
        try:
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
        except asyncio.CancelledError:
            # A cancel of the caller, the stop sent or not: the VM resumes first (best effort,
            # as after a failure), as a caller that keeps it (a session) runs later calls on it
            await await_cancellation_safe(self._resume_after_failed_save())
            raise

        if keep_running:
            resp = await self._execute("cont", arguments=None, timeout=_QMP_TIMEOUT)
            if "error" in resp:
                raise _failed(f"cont after save failed: {resp['error']}", context=None)
        else:
            await self.quit()

        logger.info("VM state saved", extra={"vmstate_path": str(vmstate_path), "keep_running": keep_running})

    async def quit(self) -> None:
        """Ask QEMU to quit: it closes its drives and exits 0, whatever process wraps it.
        No error when QEMU closes the socket before it answers."""
        with contextlib.suppress(TransientError, OSError):
            await self._execute("quit", timeout=2.0, arguments=None)

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
        qemu_version: tuple[int, int, int],
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
            TransientError: MIGRATION_FAILED: QMP failure (a reset or broken connection
                too), timeout, or QEMU could not open the file (migrate-incoming refused)
        """
        await self._set_capabilities(qemu_version, use_template=use_template)

        resp = await self._execute(
            "migrate-incoming",
            arguments={"uri": f"file:{vmstate_path}", "exit-on-error": False},
            timeout=_QMP_TIMEOUT,
        )
        if "error" in resp:
            # The command fails before QEMU reads any of the stream: it could not open the
            # file (EACCES, ENOENT) or start the migration. Not a refusal of the saved state,
            # so not MIGRATION_REJECTED: an L1 entry stays (QEMU migration/migration.c
            # qmp_migrate_incoming, migration/file.c file_connect_incoming).
            raise _failed(f"migrate-incoming command failed: {resp['error']}", context=None)

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
        """Poll query-migrate until completed/failed, with a pause that doubles from
        _POLL_FIRST_SECONDS to MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS: an L1 restore loads
        its device state in a few ms, and a fixed 5 ms pause would add up to 5 ms to it.

        Raises:
            SandboxError: failure(message) when the migration reached
                failed/cancelled/failing (with QEMU's error-desc)
            TransientError: MIGRATION_FAILED: query-migrate failed or timed out
        """
        delay = _POLL_FIRST_SECONDS
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

                    await asyncio.sleep(delay)
                    delay = min(delay * 2, constants.MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS)
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

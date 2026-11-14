"""QEMU Storage Daemon for fast overlay creation.

Uses qemu-storage-daemon (QSD) instead of forking qemu-img processes, as
recommended by the QEMU project for storage operations outside a VM:
- https://www.qemu.org/docs/master/tools/qemu-storage-daemon.html
- https://lwn.net/Articles/911281/ (LWN: "Accessing QEMU storage features without a VM")

No process starts per overlay: about 6 ms per overlay through QMP, against
about 39 ms for a qemu-img process (its fork and exec included). The first overlay of
each base image version is also kept as a template: each later overlay of that version
is a clone of it (the same bytes): 0.3 ms, and 1.4 ms for 10 at once, where QMP takes
31 ms for 10 at once.

Architecture:
- Single persistent daemon process per OverlayPool
- QMP socket: <tempfile.gettempdir()>/qsd-{pid namespace}-{pid}-{random}.sock
- Templates: <tempfile.gettempdir()>/qsd-{pid namespace}-{pid}-{random}.templates/ (0700)
- Job completion: _wait_for_job() polls query-jobs (the job's status and error) from
  1 ms, doubling up to 50 ms; an exchange skips the JOB_STATUS_CHANGE events it reads.
- Parent death: on Linux the kernel kills the daemon when the thread that
  spawned it (the event loop's) ends, SIGKILL of this process included
  (PR_SET_PDEATHSIG, subprocess_utils.start_managed_process end_with_this_process).
- Orphan reaper: On startup, scans the temp folder for stale QSD sockets from dead
  parent processes and kills the owning daemons (and removes their template folders). Handles crashes, OOM kills,
  and kill -9 that bypass normal cleanup (the only way on macOS).
- Process registry: Daemon PID is registered with process_registry so
  force_kill_all() (2nd Ctrl+C) reaches QSD processes.

Usage:
    daemon = QemuStorageDaemon()
    await daemon.start()
    await daemon.create_overlay(base_image, overlay_path)
    await daemon.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import psutil

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import await_settled
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError, spawn_error
from exec_sandbox.files.disk import clone_sparse
from exec_sandbox.process_registry import unregister_process
from exec_sandbox.resource_cleanup import cleanup_process
from exec_sandbox.subprocess_utils import (
    exec_failed_error,
    start_managed_process,
    wait_for_socket,
)

if TYPE_CHECKING:
    from exec_sandbox.platform_utils import ProcessWrapper

logger = get_logger(__name__)

# Cleanup must outlive a caller that abandons a partially-started daemon: the process
# cleanups and the template folder removals (stop).
_cleanup_tasks: set[asyncio.Task[None]] = set()

_BINARY: Final = "qemu-storage-daemon"
_PID_MAX: Final = 2**31 - 1
"""The largest pid any kernel gives (a C int): a socket name with a larger one names no owner."""


def _pid_namespace() -> int:
    """This process's pid namespace: the inode of /proc/self/ns/pid (Linux), else 0 (macOS has
    none). A pid in a socket name means something only in that namespace."""
    try:
        return Path("/proc/self/ns/pid").stat().st_ino
    except OSError:
        return 0


_INSTALL_HINT: Final = "Install QEMU (it ships qemu-storage-daemon)."

# qcow2 options of every overlay (qemu-img create -o lazy_refcounts=on,extended_l2=on,cluster_size=128k):
# lazy refcounts for faster writes (after a host crash QEMU rebuilds the refcounts at the next open), extended L2
# entries for sub-cluster allocation of small writes, 128 KiB clusters to balance metadata size
# and I/O.
_OVERLAY_QCOW2_OPTIONS: Final[dict[str, Any]] = {"cluster-size": 131072, "lazy-refcounts": True, "extended-l2": True}


def _overlay_failed(message: str, *, context: dict[str, Any] | None) -> TransientError:
    """The TransientError (OVERLAY_FAILED) of a daemon, QMP, image-info or job failure."""
    return TransientError(message, code=ErrorCode.OVERLAY_FAILED, context=context, hint=None, instance_id=None)


_QCOW2_MAGIC: Final = b"QFI\xfb"
_QCOW2_HEADER_BYTES: Final = 32  # magic(4) version(4) backing offset(8) size(4) cluster bits(4) virtual size(8)
_QCOW2_VERSIONS: Final = (2, 3)  # version 1 ("qcow") has the same magic; QEMU's qcow2 driver refuses it
_SECTOR_BYTES: Final = 512
_JOB_POLL_FIRST_SECONDS: Final = 0.001  # job poll: 1 ms, doubling up to 50 ms
_JOB_POLL_MAX_SECONDS: Final = 0.05


def _file_identity(image_path: Path) -> tuple[Path, int, int, int, int]:
    """The path and the version of the file at it (device, inode, mtime, size): one stat.

    Raises:
        TransientError: OVERLAY_FAILED if the stat fails (missing file, no access)
    """
    try:
        st = image_path.stat()
    except OSError as e:
        raise _overlay_failed(f"Failed to stat {image_path}: {e.strerror or e}", context=None) from e
    return image_path, st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size


def _clone_overlay(src: Path, dst: Path) -> bool:
    """Make `dst` a copy of the overlay `src` with its mode (clone_sparse gives 0600 where it
    is no APFS clone; the daemon makes 0666 minus the umask). False when it cannot: a
    partial `dst` is removed, one that was there is left (blocking)."""
    try:
        clone_sparse(src, dst, stop=None)  # an overlay: a reflink clone, or a copy of KBs
    except FileExistsError:
        return False
    except OSError:
        with contextlib.suppress(OSError):
            dst.unlink()
        return False
    try:
        dst.chmod(stat.S_IMODE(src.stat().st_mode))
    except OSError:
        with contextlib.suppress(OSError):
            dst.unlink()
        return False
    return True


def _qcow2_virtual_size(image_path: Path) -> int:
    """Virtual size of a qcow2 base image, read from its header (magic at 0, version at 4,
    size as a big-endian u64 at 24): 32 bytes of a file, no qemu-img process. Rounded down
    to whole 512-byte sectors, as QEMU's qcow2 driver reports it.

    Raises:
        TransientError: OVERLAY_FAILED if the file cannot be read or is not qcow2 v2/v3
    """
    try:
        with image_path.open("rb") as f:
            header = f.read(_QCOW2_HEADER_BYTES)
    except OSError as e:
        raise _overlay_failed(f"Failed to read the header of {image_path}: {e.strerror or e}", context=None) from e
    if (
        len(header) < _QCOW2_HEADER_BYTES
        or header[:4] != _QCOW2_MAGIC
        or int.from_bytes(header[4:8], "big") not in _QCOW2_VERSIONS
    ):
        raise _overlay_failed(f"Base image {image_path} is not qcow2 format (version 2 or 3)", context=None)
    size = int.from_bytes(header[24:_QCOW2_HEADER_BYTES], "big")
    return size - size % _SECTOR_BYTES


class QemuStorageDaemon:
    """Manages qemu-storage-daemon lifecycle and QMP communication.

    The daemon process handles qcow2 overlay creation via QMP blockdev-create
    commands, avoiding fork/exec overhead of spawning qemu-img processes.

    Safe for concurrent coroutines of one event loop: an asyncio.Lock serializes QMP commands.

    Attributes:
        _socket_path: Full path to QMP socket
        _process: Daemon process wrapper
        _started: Whether daemon is running
    """

    __slots__ = (
        "_cleanup_task",
        "_lifecycle_lock",
        "_lock",
        "_process",
        "_reader",
        "_request_seq",
        "_socket_path",
        "_started",
        "_template_dir",
        "_template_writes",
        "_templates",
        "_writer",
    )

    def __init__(self) -> None:
        """Initialize daemon (does not start it)."""
        self._socket_path: Path | None = None
        self._process: ProcessWrapper | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._started = False
        self._request_seq = 0
        self._lock = asyncio.Lock()  # Serialize QMP commands
        # start() and stop() one at a time: a stop() during a start() then stops what that
        # start() made, instead of finding nothing to stop
        self._lifecycle_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        # The template of each base image version (keyed by _file_identity), made by start():
        # None when the folder could not be made (each overlay through QMP). The future gives
        # the template once its first overlay is made, or None (the next creation tries again).
        self._template_dir: Path | None = None
        self._templates: dict[tuple[Path, int, int, int, int], asyncio.Future[Path | None]] = {}
        # Template clones still being written: stop() waits for them before it removes the
        # folder, which they would fill again
        self._template_writes: set[asyncio.Future[bool]] = set()

    @property
    def started(self) -> bool:
        """Whether daemon is running."""
        return self._started

    async def start(self) -> None:
        """Start daemon and establish QMP connection.

        Idempotent: Does nothing if already started.

        Raises:
            TransientError: OVERLAY_FAILED if daemon fails to start or connect (also a fork that
                hits a process, memory or file-handle limit: spawn_error)
            ConfigError: HOST_DEPENDENCY if the qemu-storage-daemon binary is missing or cannot
                run, or this process may not connect to its socket
        """
        async with self._lifecycle_lock:
            await self._start()

    async def _start(self) -> None:
        """start() under the lifecycle lock."""
        if self._started:
            return
        if self._process is not None:
            # A prior failed/cancelled start still owns a daemon. Confirm its
            # death before assigning another socket/process handle.
            await self._stop()

        # Clean up orphaned daemons from previous crashes (non-critical)
        try:
            count = await asyncio.to_thread(self._reap_stale_daemons)
            if count:
                logger.info("Reaped stale QSD daemons", extra={"count": count})
        except (OSError, psutil.Error):
            logger.debug("Orphan reaper failed (non-critical)", exc_info=True)

        # The temp folder: a short path, under the Unix socket path limit (104-108 bytes)
        socket_path = Path(tempfile.gettempdir()) / f"qsd-{_pid_namespace()}-{os.getpid()}-{secrets.token_hex(8)}.sock"
        self._socket_path = socket_path

        # Start daemon process
        cmd = [
            _BINARY,
            "--chardev",
            f"socket,path={socket_path},server=on,wait=off,id=qmp0",
            "--monitor",
            "chardev=qmp0",
        ]

        spawned = False
        try:
            self._process = await start_managed_process(
                cmd,
                steps=(),
                end_with_this_process=True,
                pass_fds=(),
                on_process_started=lambda proc: setattr(self, "_process", proc),
            )
            spawned = True

            # Wait for socket and connect (single step to avoid TOCTOU)
            await self._wait_and_connect_qmp(socket_path)
            self._started = True

            logger.info(
                "QEMU storage daemon started",
                extra={"socket": str(socket_path), "pid": self._process.pid},
            )

        except BaseException as e:
            # Publish process cleanup before another await; connection cleanup
            # may itself be interrupted by repeated caller cancellation (the
            # globally-retained cleanup owner then keeps retrying).
            cleanup_task = self._ensure_cleanup_task()
            await self._cleanup_connection()
            await asyncio.shield(cleanup_task)
            if isinstance(e, asyncio.CancelledError | ConfigError):  # ConfigError: exec_failed_error
                raise
            if isinstance(e, OSError) and not spawned:
                raise spawn_error(
                    e, binary=cmd[0], code=ErrorCode.OVERLAY_FAILED, context={}, hint=_INSTALL_HINT
                ) from e
            raise _overlay_failed(f"Failed to start daemon: {e}", context=None) from e

        # The templates' folder, pid-scoped as the socket (the reaper removes it after a
        # crash). Without it, each overlay goes through QMP.
        # Named and recorded before the mkdir: a cancel during it leaves no folder that stop()
        # does not know
        template_dir = (
            Path(tempfile.gettempdir()) / f"qsd-{_pid_namespace()}-{os.getpid()}-{secrets.token_hex(8)}.templates"
        )
        self._template_dir = template_dir
        # A cancel waits for the mkdir thread: else the folder could come after the removal
        # of a stop() that runs at once
        made = asyncio.create_task(asyncio.to_thread(template_dir.mkdir, mode=0o700))
        try:
            await asyncio.shield(made)
        except asyncio.CancelledError:
            await await_settled(made)
            if not made.cancelled():
                made.exception()  # retrieved: the cancel wins
            raise
        except OSError:
            self._template_dir = None
            logger.debug("No overlay template folder: each overlay is made through QMP", exc_info=True)
            return

    async def stop(self) -> None:
        """Stop daemon gracefully.

        Idempotent: returns at once if not started. A start() that runs: stop() waits for it,
        then stops what it made.
        Cancellation-safe: the stop runs in a retained task, which a cancel of this call does
        not stop, also while it waits for a start().
        """
        stopping = asyncio.create_task(self._stop_locked())
        _cleanup_tasks.add(stopping)
        stopping.add_done_callback(_cleanup_tasks.discard)
        # Retrieved here: a cancelled caller no longer waits for it (no unretrieved error)
        stopping.add_done_callback(lambda task: task.cancelled() or task.exception())
        await asyncio.shield(stopping)

    async def _stop_locked(self) -> None:
        """Take the lifecycle lock, then _stop."""
        async with self._lifecycle_lock:
            await self._stop()

    async def _stop(self) -> None:
        """stop() itself; the caller holds the lifecycle lock."""
        if not self._started and self._process is None:
            return

        # Set early so idempotency guard works even if we're cancelled mid-cleanup
        was_started = self._started
        self._started = False
        templates_removed = self._remove_templates()

        try:
            # Send quit command (ignore errors - daemon may already be dead)
            if was_started:
                with contextlib.suppress(Exception):
                    await self._execute("quit")
        finally:
            # Publish cleanup ownership before another await. If this stop
            # waiter is cancelled, the global task continues retrying.
            cleanup_task = self._ensure_cleanup_task()
            await self._cleanup_connection()
            await asyncio.shield(cleanup_task)

        if templates_removed is not None:
            await asyncio.shield(templates_removed)

        logger.info("QEMU storage daemon stopped")

    def _remove_templates(self) -> asyncio.Task[None] | None:
        """Forget the templates and remove their folder in a retained task, once the clones in
        flight end: a clone during the removal would leave a file, so the folder would stay
        (create_overlay starts no clone once the folder is forgotten). It cancels no clone
        that a create_overlay awaits (asyncio.wait does not cancel what it waits for)."""
        template_dir, self._template_dir = self._template_dir, None
        self._templates.clear()
        if template_dir is None:
            return None
        writes = set(self._template_writes)

        async def remove() -> None:
            if writes:  # each about 0.3 ms
                await asyncio.wait(writes)
            try:
                await asyncio.to_thread(shutil.rmtree, template_dir)
            except FileNotFoundError:
                pass
            except OSError as e:  # a few small files; the folder of a dead owner goes at a later start
                logger.warning(
                    "Failed to remove the overlay template folder", extra={"folder": str(template_dir), "error": str(e)}
                )

        task = asyncio.create_task(remove())
        _cleanup_tasks.add(task)
        task.add_done_callback(_cleanup_tasks.discard)
        return task

    async def create_overlay(self, base_image: Path, overlay_path: Path) -> None:
        """Create qcow2 overlay with backing file: the first one of each base image version
        via QMP (_create_with_qmp), kept as its template; each later one is a clone of that
        template (the same bytes), via QMP when the clone fails.

        Raises:
            TransientError: OVERLAY_FAILED if daemon not started or creation fails
        """
        if not self._started:
            raise _overlay_failed("Daemon not started", context=None)
        # The backing path is stored in the overlay as given and read relative to the
        # overlay's folder: an absolute path keeps it valid after the pool moves the file
        base_image = base_image.absolute()
        identity = await asyncio.to_thread(_file_identity, base_image)
        template_dir = self._template_dir
        if template_dir is None:
            await self._create_with_qmp(base_image, overlay_path)
            return

        made = self._templates.get(identity)
        if made is not None:
            template = await asyncio.shield(made)  # its first overlay may still be in the making
            if template is None or not await asyncio.to_thread(_clone_overlay, template, overlay_path):
                await self._create_with_qmp(base_image, overlay_path)
            return

        made = asyncio.get_running_loop().create_future()
        self._templates[identity] = made
        template = None
        try:
            await self._create_with_qmp(base_image, overlay_path)
            # stop() began meanwhile: its removal of the folder must not race a clone
            if self._template_dir is not template_dir:
                return
            candidate = template_dir / f"{secrets.token_hex(8)}.qcow2"
            write = asyncio.ensure_future(asyncio.to_thread(_clone_overlay, overlay_path, candidate))
            self._template_writes.add(write)
            write.add_done_callback(self._template_writes.discard)
            if await asyncio.shield(write):
                template = candidate
        finally:  # also on a failure or a cancel: the waiters then use QMP
            made.set_result(template)
            if template is None and self._templates.get(identity) is made:
                del self._templates[identity]

    async def _create_with_qmp(self, base_image: Path, overlay_path: Path) -> None:
        """Create qcow2 overlay with backing file via QMP (`base_image` absolute; its virtual
        size read in a worker thread: images can be on a network disk).

        Uses 3-step approach required by qemu-storage-daemon:
        1. blockdev-create with file driver (creates empty file)
        2. blockdev-add to open the file as a node
        3. blockdev-create with qcow2 driver (writes qcow2 structure)

        Equivalent to:
            qemu-img create -f qcow2 -F qcow2 -b base_image \\
                -o lazy_refcounts=on,extended_l2=on,cluster_size=128k overlay_path

        Args:
            base_image: Base qcow2 image (backing file, read-only)
            overlay_path: Path for new overlay file (will be created)

        Raises:
            TransientError: OVERLAY_FAILED if creation fails
        """
        uid = secrets.token_hex(8)
        file_job_id = f"file-{uid}"
        node_name = f"node-{uid}"
        qcow2_job_id = f"qcow2-{uid}"

        virtual_size = await asyncio.to_thread(_qcow2_virtual_size, base_image)

        try:
            # Step 1: Create empty file
            await self._execute(
                "blockdev-create",
                {
                    "job-id": file_job_id,
                    "options": {
                        "driver": "file",
                        "filename": str(overlay_path),
                        "size": 0,
                    },
                },
            )
            await self._wait_for_job(file_job_id)

            # Step 2: Open file as blockdev node
            await self._execute(
                "blockdev-add",
                {
                    "driver": "file",
                    "node-name": node_name,
                    "filename": str(overlay_path),
                },
            )

            # Step 3: Create qcow2 on top of file node
            await self._execute(
                "blockdev-create",
                {
                    "job-id": qcow2_job_id,
                    "options": {
                        "driver": "qcow2",
                        "file": node_name,
                        "size": virtual_size,
                        "backing-file": str(base_image),
                        "backing-fmt": "qcow2",
                        **_OVERLAY_QCOW2_OPTIONS,
                    },
                },
            )
            await self._wait_for_job(qcow2_job_id)

        finally:
            # Always clean up blockdev node
            with contextlib.suppress(TransientError):
                await self._execute("blockdev-del", {"node-name": node_name})

    async def _wait_and_connect_qmp(self, socket_path: Path) -> None:
        """Wait for daemon socket and establish QMP connection in one step.

        Uses keep_connection=True to reuse the socket from the readiness probe: no
        gap between the probe's close and a second connect, which a single-client
        QMP chardev socket can refuse (ConnectionRefusedError, seen under CI load).

        Raises:
            TransientError: OVERLAY_FAILED if socket doesn't appear, the daemon exits, or
                qmp_capabilities fails
            ConfigError: HOST_DEPENDENCY: the launch exited 126 or 127 (exec_failed_error), or
                this process may not connect to the socket (wait_for_socket)
            TimeoutError, ValueError: no greeting in time, or one that is not JSON (start()
                raises them as OVERLAY_FAILED)
        """

        def _check_process_alive() -> None:
            if self._process and (returncode := self._process.returncode) is not None:
                # 126 or 127: an exec of its launch failed (ConfigError); else it ended
                raise exec_failed_error(
                    returncode, binary=_BINARY, detail=None, context={}, hint=_INSTALL_HINT
                ) or _overlay_failed(f"Daemon process exited with code {returncode}", context=None)

        startup_timeout = constants.QEMU_STORAGE_DAEMON_STARTUP_TIMEOUT_SECONDS
        socket_timeout = constants.QEMU_STORAGE_DAEMON_SOCKET_TIMEOUT_SECONDS

        try:
            self._reader, self._writer = await wait_for_socket(
                socket_path,
                timeout=startup_timeout,
                abort_check=_check_process_alive,
                keep_connection=True,
            )
        except TimeoutError as e:
            raise _overlay_failed(f"Daemon socket not ready after {startup_timeout}s", context=None) from e

        # Read greeting
        greeting = await asyncio.wait_for(self._reader.readline(), timeout=socket_timeout)
        greeting_data = json.loads(greeting)
        if "QMP" not in greeting_data:
            raise _overlay_failed(f"Invalid QMP greeting: {greeting.decode()}", context=None)

        # Send capabilities
        await self._execute("qmp_capabilities")

    async def _execute(self, command: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute QMP command and return result, within QEMU_STORAGE_DAEMON_SOCKET_TIMEOUT_SECONDS.

        Handles QMP's async event model - reads lines until we get the actual
        command response (containing "return" or "error"), skipping events.

        Args:
            command: QMP command name
            arguments: Optional command arguments

        Returns:
            QMP response dict (the 'return' field contents)

        Raises:
            TransientError: OVERLAY_FAILED if command fails or not connected
        """
        self._request_seq += 1
        request_id = f"qsd-{self._request_seq}"

        async def exchange() -> dict[str, Any]:
            async with self._lock:
                return await self._exchange(request_id, command, arguments)

        # Once a command is written, consume its identified response before
        # propagating caller cancellation. This preserves stream alignment.
        exchange_task = asyncio.create_task(exchange())
        try:
            return await asyncio.shield(exchange_task)
        except asyncio.CancelledError:
            await await_settled(exchange_task)
            if not exchange_task.cancelled():
                exchange_task.exception()
            raise

    async def _exchange(self, request_id: str, command: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Write one identified QMP command and consume its matching response.

        Raises:
            TransientError: OVERLAY_FAILED: a QMP error, no answer by the timeout, a daemon
                that died (end of the stream, a broken socket) or sent a line that is not a
                JSON object or is over the reader's limit (never a raw TimeoutError,
                OSError, ValueError, AttributeError or TypeError)
        """
        if self._writer is None or self._reader is None:
            raise _overlay_failed("Not connected to QMP", context=None)

        msg: dict[str, Any] = {"execute": command, "id": request_id}
        if arguments:
            msg["arguments"] = arguments

        try:
            self._writer.write(json.dumps(msg).encode() + b"\n")
            await self._writer.drain()
        except OSError as e:
            raise _overlay_failed(f"QMP {command}: the daemon's socket failed: {e}", context=None) from e

        loop = asyncio.get_running_loop()
        deadline = loop.time() + constants.QEMU_STORAGE_DAEMON_SOCKET_TIMEOUT_SECONDS
        while loop.time() < deadline:
            remaining = deadline - loop.time()
            result = await self._read_message(self._reader, command, timeout=max(0.1, remaining))

            if "event" in result:  # events (JOB_STATUS_CHANGE, ...): _wait_for_job polls query-jobs
                continue

            if result.get("id") != request_id:
                logger.warning(
                    "Discarding unmatched QMP response",
                    extra={"expected_id": request_id, "response_id": result.get("id")},
                )
                continue

            if "error" in result:
                error = cast("dict[str, Any]", result["error"]) if isinstance(result["error"], dict) else {}
                raise _overlay_failed(
                    error.get("desc", "Unknown QMP error"),
                    context={"error_class": error.get("class")},
                )

            if "return" in result:
                return result["return"]

        raise _overlay_failed(f"Timeout waiting for response to {command}", context=None)

    @staticmethod
    async def _read_message(reader: asyncio.StreamReader, command: str, *, timeout: float) -> dict[str, Any]:
        """The next QMP message of `command`'s exchange.

        Raises:
            TransientError: OVERLAY_FAILED: none by `timeout`, a socket that failed or ended
                (the daemon exited), a line over the reader's limit, or a line that is not a
                JSON object
        """
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        except TimeoutError as e:
            raise _overlay_failed(f"Timeout waiting for response to {command}", context=None) from e
        except OSError as e:
            raise _overlay_failed(f"QMP {command}: the daemon's socket failed: {e}", context=None) from e
        except ValueError as e:  # StreamReader.readline: a line over the limit
            raise _overlay_failed(f"QMP {command}: a line over the reader's limit: {e}", context=None) from e
        if not line:
            raise _overlay_failed(f"QMP {command}: the daemon closed its socket (it exited)", context=None)
        try:
            message = json.loads(line)
        except (ValueError, RecursionError) as e:  # RecursionError: a line nested too deep
            raise _overlay_failed(f"QMP {command}: a line that is not JSON: {line[:200]!r}", context=None) from e
        if not isinstance(message, dict):
            raise _overlay_failed(f"QMP {command}: a line that is not a JSON object: {line[:200]!r}", context=None)
        return cast("dict[str, Any]", message)

    async def _wait_for_job(self, job_id: str) -> None:
        """Wait for an async job to end: poll query-jobs (its status and error), with a pause
        that doubles from _JOB_POLL_FIRST_SECONDS to _JOB_POLL_MAX_SECONDS. A concluded job is
        dismissed; a job no longer listed concluded and was auto-dismissed.

        Args:
            job_id: Job ID to wait for (at most QEMU_STORAGE_DAEMON_JOB_TIMEOUT_SECONDS)

        Raises:
            TransientError: OVERLAY_FAILED if job times out or fails
        """
        timeout = constants.QEMU_STORAGE_DAEMON_JOB_TIMEOUT_SECONDS
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        delay = _JOB_POLL_FIRST_SECONDS
        while loop.time() < deadline:
            # query-jobs tells the job's end and its error (a JOB_STATUS_CHANGE event carries
            # no error): a job it lists as running is polled again, never taken as done
            result = await self._execute("query-jobs")
            raw_jobs = cast("list[dict[str, Any]]", result) if isinstance(result, list) else []
            job = next((raw for raw in raw_jobs if raw.get("id") == job_id), None)

            if job is None:
                return  # concluded and auto-dismissed

            if job.get("status") == "concluded":
                if error := job.get("error"):
                    with contextlib.suppress(TransientError):
                        await self._execute("job-dismiss", {"id": job_id})
                    raise _overlay_failed(f"Job {job_id} failed: {error}", context=None)
                await self._execute("job-dismiss", {"id": job_id})
                return

            # This poll is the wait: a job takes a few ms, a fixed 50 ms sleep cost ~50 ms
            await asyncio.sleep(delay)
            delay = min(delay * 2, _JOB_POLL_MAX_SECONDS)

        raise _overlay_failed(f"Job {job_id} timed out after {timeout}s", context=None)

    @staticmethod
    def _reap_stale_daemons() -> int:
        """Remove stale QSD sockets from dead parent processes and kill orphaned daemons.

        Socket filenames encode the parent's pid namespace and pid:
        qsd-{namespace}-{pid}-{random}.sock. When the parent dies (crash, OOM, kill -9), the
        daemon becomes an orphan. This method finds such orphans and cleans them up.

        Only sockets of this pid namespace count: a pid means something only in its own
        namespace, and the temp folder can be shared with other ones (a container's /tmp), so
        another namespace's live daemon is never taken for an orphan. A pid out of the C int
        range (no kernel gives one; psutil raises OverflowError) is no owner of ours.

        One scan of the processes, only when a stale socket exists: a process is the orphan of a
        socket when one of its arguments is the daemon's own `--chardev socket,path=<socket>,...`
        value. No name or executable filter: a daemon whose binary was replaced (an upgrade) or
        that runs through a wrapper still matches, and a process that only names the path does
        not. A process that ends or cannot be read during the scan is skipped: psutil
        raises psutil.Error, or on macOS SystemError for a process that exits while its
        arguments are read.

        The template folders of dead owners (qsd-{namespace}-{pid}-{random}.templates, folders
        of this user only: the temp folder is shared) are removed too.

        Returns:
            Number of stale sockets cleaned up.
        """
        namespace = _pid_namespace()
        pattern = re.compile(rf"^qsd-{namespace}-(\d+)-[0-9a-f]+\.sock$")
        template_pattern = re.compile(rf"^qsd-{namespace}-(\d+)-[a-z0-9_]+\.templates$")

        def is_stale(name: str, name_pattern: re.Pattern[str]) -> bool:
            match = name_pattern.match(name)
            if match is None:
                return False
            pid = int(match.group(1))
            return pid <= _PID_MAX and not psutil.pid_exists(pid)

        for folder in Path(tempfile.gettempdir()).glob("qsd-*.templates"):
            if not is_stale(folder.name, template_pattern):
                continue
            with contextlib.suppress(OSError):
                st = folder.lstat()
                if stat.S_ISDIR(st.st_mode) and st.st_uid == os.geteuid():
                    shutil.rmtree(folder)
                    logger.debug("Removed stale QSD template folder", extra={"folder": str(folder)})

        stale = [path for path in Path(tempfile.gettempdir()).glob("qsd-*.sock") if is_stale(path.name, pattern)]
        if not stale:
            return 0

        chardevs = tuple(f"socket,path={path}," for path in stale)  # as start() builds it
        for proc in psutil.process_iter():
            try:
                arguments = proc.cmdline()
            except (psutil.Error, SystemError):
                continue
            if any(argument.startswith(chardevs) for argument in arguments):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
                    logger.debug("Killed orphaned QSD process", extra={"pid": proc.pid})

        cleaned = 0
        for path in stale:
            with contextlib.suppress(OSError):
                path.unlink()
                cleaned += 1
                logger.debug("Removed stale QSD socket", extra={"socket": str(path)})
        return cleaned

    async def _cleanup_connection(self) -> None:
        """Close QMP connection."""
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._writer.wait_closed(), timeout=1.0)
        self._reader = None
        self._writer = None

    async def _cleanup_process(self) -> bool:
        """Terminate the daemon and clear ownership only after confirmed death."""
        if self._process is None:
            return True

        process = self._process
        destroyed = await cleanup_process(
            process,
            "qemu-storage-daemon",
            str(self._socket_path or "qsd-startup"),
            term_timeout=2.0,
            kill_timeout=2.0,
            kill_first=False,
        )
        if destroyed:
            unregister_process(process)
            self._process = None
        return destroyed

    def _ensure_cleanup_task(self) -> asyncio.Task[None]:
        """Return one globally-retained daemon cleanup owner."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return self._cleanup_task

        self._cleanup_task = asyncio.create_task(self._cleanup_process_until_confirmed())
        _cleanup_tasks.add(self._cleanup_task)
        self._cleanup_task.add_done_callback(_cleanup_tasks.discard)
        return self._cleanup_task

    async def _cleanup_process_until_confirmed(self) -> None:
        """Retry bounded termination attempts until process death is confirmed."""
        delay = 1.0
        while self._process is not None:
            try:
                if await self._cleanup_process():
                    break
            except Exception:
                logger.exception("QEMU storage daemon cleanup attempt failed")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

        if self._socket_path is not None:
            with contextlib.suppress(OSError):
                self._socket_path.unlink()

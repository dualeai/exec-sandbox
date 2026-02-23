"""QEMU Storage Daemon for fast overlay creation.

Uses qemu-storage-daemon (QSD) instead of forking qemu-img processes, as
recommended by the QEMU project for storage operations outside a VM:
- https://www.qemu.org/docs/master/tools/qemu-storage-daemon.html
- https://lwn.net/Articles/911281/ (LWN: "Accessing QEMU storage features without a VM")

This eliminates ~70ms fork/exec overhead per overlay creation, achieving
~4-5ms per overlay via QMP commands vs ~39ms via qemu-img subprocess.

Architecture:
- Single persistent daemon process per OverlayPool
- QMP socket: /tmp/qsd-{pid}-{random}.sock
- Event-driven job completion: JOB_STATUS_CHANGE events are stashed by
  _execute() and consumed by _wait_for_job(), eliminating polling storms.
  Fallback polling at 50ms interval as safety net.
- Orphan reaper: On startup, scans /tmp for stale QSD sockets from dead
  parent processes and kills the owning daemons. Handles crashes, OOM kills,
  and kill -9 that bypass normal cleanup.
- Process registry: Daemon PID is registered with process_registry so
  force_kill_all() (2nd Ctrl+C) reaches QSD processes.

Usage:
    async with QemuStorageDaemon() as daemon:
        await daemon.create_overlay(base_image, overlay_path)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import secrets
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import psutil

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import VmOverlayError as QemuStorageDaemonError
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.process_registry import register_process, unregister_process

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QmpJob:
    """QMP job status from query-jobs response."""

    job_id: str
    status: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class QmpEvent:
    """Parsed QMP event (e.g. JOB_STATUS_CHANGE).

    Only the fields relevant to job completion tracking are extracted.
    Unknown/malformed events parse to empty strings, allowing safe
    matching without KeyError.
    """

    event: str
    job_id: str
    status: str


class QemuStorageDaemon:
    """Manages qemu-storage-daemon lifecycle and QMP communication.

    The daemon process handles qcow2 overlay creation via QMP blockdev-create
    commands, avoiding fork/exec overhead of spawning qemu-img processes.

    Thread-safe: Uses asyncio lock for QMP command serialization.

    Attributes:
        _socket_path: Full path to QMP socket
        _process: Daemon process wrapper
        _started: Whether daemon is running
    """

    __slots__ = (
        "_event_buffer",
        "_lock",
        "_process",
        "_reader",
        "_socket_path",
        "_started",
        "_virtual_size_cache",
        "_writer",
    )

    def __init__(self) -> None:
        """Initialize daemon (does not start it)."""
        self._socket_path: Path | None = None
        self._process: ProcessWrapper | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._started = False
        self._lock = asyncio.Lock()  # Serialize QMP commands
        self._virtual_size_cache: dict[str, int] = {}  # Cache base image sizes
        self._event_buffer: list[QmpEvent] = []  # Events stashed by _execute(), consumed by _wait_for_job()

    @property
    def started(self) -> bool:
        """Whether daemon is running."""
        return self._started

    async def start(self) -> None:
        """Start daemon and establish QMP connection.

        Idempotent: Does nothing if already started.

        Raises:
            QemuStorageDaemonError: If daemon fails to start or connect
        """
        if self._started:
            return

        # Clean up orphaned daemons from previous crashes (non-critical)
        try:
            count = await asyncio.to_thread(self._reap_stale_daemons)
            if count:
                logger.info("Reaped stale QSD daemons", extra={"count": count})
        except (OSError, psutil.Error):
            logger.debug("Orphan reaper failed (non-critical)", exc_info=True)

        # Use /tmp for socket to avoid Unix socket path length limit (104-108 chars)
        self._socket_path = Path(tempfile.gettempdir()) / f"qsd-{os.getpid()}-{secrets.token_hex(8)}.sock"

        # Start daemon process
        cmd = [
            "qemu-storage-daemon",
            "--chardev",
            f"socket,path={self._socket_path},server=on,wait=off,id=qmp0",
            "--monitor",
            "chardev=qmp0",
        ]

        try:
            self._process = ProcessWrapper(
                await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            )
            register_process(self._process)

            # Wait for socket and connect
            await self._wait_for_socket()
            await self._connect_qmp()
            self._started = True

            logger.info(
                "QEMU storage daemon started",
                extra={"socket": str(self._socket_path), "pid": self._process.pid},
            )

        except Exception as e:
            # Cleanup on failure
            unregister_process(self._process)
            await self._cleanup_process()
            raise QemuStorageDaemonError(f"Failed to start daemon: {e}") from e

    async def stop(self) -> None:
        """Stop daemon gracefully.

        Idempotent: Clears event buffer and returns if not started.
        Cancellation-safe: _started is set False before any await.
        """
        if not self._started:
            self._event_buffer.clear()
            return

        # Set early so idempotency guard works even if we're cancelled mid-cleanup
        self._started = False

        # Send quit command (ignore errors - daemon may already be dead)
        with contextlib.suppress(Exception):
            await self._execute("quit")

        # Unregister before _cleanup_process() sets self._process = None
        unregister_process(self._process)

        try:
            # Close connection
            await self._cleanup_connection()
            await self._cleanup_process()
        finally:
            # Cleanup socket and buffer even if cancelled
            if self._socket_path and self._socket_path.exists():
                with contextlib.suppress(OSError):
                    self._socket_path.unlink()
            self._event_buffer.clear()

        logger.info("QEMU storage daemon stopped")

    async def create_overlay(
        self,
        base_image: Path,
        overlay_path: Path,
        *,
        cluster_size: int = 131072,  # 128k
        lazy_refcounts: bool = True,
        extended_l2: bool = True,
    ) -> None:
        """Create qcow2 overlay with backing file via QMP.

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
            cluster_size: qcow2 cluster size in bytes (default: 128k)
            lazy_refcounts: Enable lazy refcounts for faster writes
            extended_l2: Enable extended L2 entries for sub-cluster allocation

        Raises:
            QemuStorageDaemonError: If daemon not started or creation fails
        """
        if not self._started:
            raise QemuStorageDaemonError("Daemon not started")

        uid = secrets.token_hex(8)
        file_job_id = f"file-{uid}"
        node_name = f"node-{uid}"
        qcow2_job_id = f"qcow2-{uid}"

        # Get base image virtual size for overlay
        virtual_size = await self._get_image_virtual_size(base_image)

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
                        "cluster-size": cluster_size,
                        "lazy-refcounts": lazy_refcounts,
                        "extended-l2": extended_l2,
                    },
                },
            )
            await self._wait_for_job(qcow2_job_id)

        finally:
            # Always clean up blockdev node
            with contextlib.suppress(QemuStorageDaemonError):
                await self._execute("blockdev-del", {"node-name": node_name})

    async def _wait_for_socket(
        self,
        timeout: float = constants.QEMU_STORAGE_DAEMON_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        """Wait for daemon socket to become available.

        Args:
            timeout: Maximum time to wait for socket

        Raises:
            QemuStorageDaemonError: If socket doesn't appear within timeout
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self._socket_path and self._socket_path.exists():
                return
            # Check if process died
            if self._process and self._process.returncode is not None:
                raise QemuStorageDaemonError(f"Daemon process exited with code {self._process.returncode}")
            await asyncio.sleep(0.05)
        raise QemuStorageDaemonError(
            f"Daemon socket not ready after {timeout}s",
        )

    async def _connect_qmp(self) -> None:
        """Establish QMP connection and complete handshake.

        Raises:
            QemuStorageDaemonError: If connection or handshake fails
        """
        if not self._socket_path:
            raise QemuStorageDaemonError("Socket path not set")

        timeout = constants.QEMU_STORAGE_DAEMON_SOCKET_TIMEOUT_SECONDS

        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(self._socket_path)),
            timeout=timeout,
        )

        # Read greeting
        greeting = await asyncio.wait_for(self._reader.readline(), timeout=timeout)
        greeting_data = json.loads(greeting)
        if "QMP" not in greeting_data:
            raise QemuStorageDaemonError(f"Invalid QMP greeting: {greeting.decode()}")

        # Send capabilities
        await self._execute("qmp_capabilities")

    async def _execute(
        self,
        command: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = constants.QEMU_STORAGE_DAEMON_SOCKET_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Execute QMP command and return result.

        Handles QMP's async event model - reads lines until we get the actual
        command response (containing "return" or "error"), skipping events.

        Args:
            command: QMP command name
            arguments: Optional command arguments
            timeout: Command timeout in seconds

        Returns:
            QMP response dict (the 'return' field contents)

        Raises:
            QemuStorageDaemonError: If command fails or not connected
        """
        if self._writer is None or self._reader is None:
            raise QemuStorageDaemonError("Not connected to QMP")

        async with self._lock:
            msg: dict[str, Any] = {"execute": command}
            if arguments:
                msg["arguments"] = arguments

            self._writer.write(json.dumps(msg).encode() + b"\n")
            await self._writer.drain()

            # QMP sends async events alongside responses. Read lines until we
            # get an actual response (has "return" or "error" key), skip events.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while loop.time() < deadline:
                remaining = deadline - loop.time()
                response = await asyncio.wait_for(self._reader.readline(), timeout=max(0.1, remaining))
                result = json.loads(response)

                # Stash concluded events for _wait_for_job; discard
                # intermediate states (created, running, waiting, pending, null)
                if "event" in result:
                    parsed = self._parse_event(result)
                    if parsed.status == "concluded":
                        self._event_buffer.append(parsed)
                    continue

                if "error" in result:
                    raise QemuStorageDaemonError(
                        result["error"].get("desc", "Unknown QMP error"),
                        result["error"].get("class"),
                    )

                if "return" in result:
                    return result["return"]

            raise QemuStorageDaemonError(f"Timeout waiting for response to {command}")

    async def _get_image_virtual_size(self, image_path: Path) -> int:
        """Get image virtual size using qemu-img info (cached).

        Note: qemu-storage-daemon doesn't have query-image-info command,
        so we use qemu-img for this metadata query. Results are cached
        per image path since base image sizes don't change.

        Args:
            image_path: Path to qcow2 image

        Returns:
            Virtual size in bytes

        Raises:
            QemuStorageDaemonError: If image doesn't exist, is invalid, or query fails
        """
        cache_key = str(image_path)
        if cache_key in self._virtual_size_cache:
            return self._virtual_size_cache[cache_key]

        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "info",
            "--output=json",
            str(image_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            raise QemuStorageDaemonError(f"Failed to get image info for {image_path}: {error_msg}")

        try:
            info = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise QemuStorageDaemonError(f"Invalid JSON from qemu-img info for {image_path}: {e}") from e

        # Verify it's a qcow2 image (not raw or other format)
        fmt = info.get("format")
        if fmt != "qcow2":
            raise QemuStorageDaemonError(f"Base image {image_path} is not qcow2 format (got: {fmt})")

        size = info.get("virtual-size")
        if size is None:
            raise QemuStorageDaemonError(f"No virtual-size in qemu-img info for {image_path}")

        self._virtual_size_cache[cache_key] = size
        return size

    @staticmethod
    def _parse_job(data: dict[str, Any]) -> QmpJob:
        """Parse raw QMP job dict into QmpJob dataclass."""
        return QmpJob(
            job_id=data.get("id", ""),
            status=data.get("status", ""),
            error=data.get("error"),
        )

    @staticmethod
    def _parse_event(raw: dict[str, Any]) -> QmpEvent:
        """Parse raw QMP event dict into QmpEvent dataclass.

        Handles malformed events gracefully — missing keys or non-dict data
        default to empty strings, so matching always works without errors.
        """
        data_raw = raw.get("data")
        data = cast("dict[str, Any]", data_raw) if isinstance(data_raw, dict) else {}
        return QmpEvent(
            event=raw.get("event", ""),
            job_id=data.get("id", ""),
            status=data.get("status", ""),
        )

    def _consume_event(self, job_id: str, status: str) -> QmpEvent | None:
        """Consume and remove first matching event from buffer.

        Scans the event buffer for a QMP event matching the given job ID and
        status. If found, removes it from the buffer and returns it.

        Args:
            job_id: Job ID to match in event data
            status: Status to match (e.g. "concluded")

        Returns:
            Matching QmpEvent, or None if no match found.
        """
        for i, event in enumerate(self._event_buffer):
            if event.job_id == job_id and event.status == status:
                self._event_buffer.pop(i)
                return event
        return None

    def _purge_job_events(self, job_id: str) -> None:
        """Remove all buffered events for the given job ID.

        Called after a job completes (by any path) to prevent stale events
        from accumulating. In concurrent scenarios, concluded events for one
        job can be buffered by _execute() calls from other coroutines after
        _wait_for_job already observed completion via query-jobs.
        """
        self._event_buffer = [e for e in self._event_buffer if e.job_id != job_id]

    async def _wait_for_job(
        self,
        job_id: str,
        timeout: float = constants.QEMU_STORAGE_DAEMON_JOB_TIMEOUT_SECONDS,
    ) -> None:
        """Wait for async job to complete using event-driven approach.

        Primary path: Checks _event_buffer for a JOB_STATUS_CHANGE(concluded)
        event that was stashed by _execute() while reading QMP responses.

        Fallback path: If no event found (e.g. job concluded and auto-dismissed
        before any event was captured), queries jobs directly.

        On completion (success or failure), purges all remaining events for
        this job_id from the buffer to prevent stale event accumulation.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait

        Raises:
            QemuStorageDaemonError: If job times out or fails
        """
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while loop.time() < deadline:
                # Primary path: check event buffer for concluded event
                concluded_event = self._consume_event(job_id, "concluded")
                if concluded_event:
                    # Event found — query jobs once to get error details and dismiss.
                    # Events don't carry error messages, only status changes.
                    result = await self._execute("query-jobs")
                    raw_jobs = cast("list[dict[str, Any]]", result) if isinstance(result, list) else []

                    for raw in raw_jobs:
                        if raw.get("id") == job_id:
                            job = self._parse_job(raw)
                            if job.status == "concluded":
                                if job.error:
                                    with contextlib.suppress(QemuStorageDaemonError):
                                        await self._execute("job-dismiss", {"id": job_id})
                                    raise QemuStorageDaemonError(f"Job {job_id} failed: {job.error}")
                                # Dismiss completed job
                                await self._execute("job-dismiss", {"id": job_id})
                                return
                            # Event said concluded but query-jobs disagrees; fall through
                            break

                    # Job already auto-dismissed (not in query-jobs list)
                    return

                # Fallback: query jobs directly (handles auto-dismiss race)
                result = await self._execute("query-jobs")
                raw_jobs = cast("list[dict[str, Any]]", result) if isinstance(result, list) else []

                job: QmpJob | None = None
                for raw in raw_jobs:
                    if raw.get("id") == job_id:
                        job = self._parse_job(raw)
                        break

                if not job:
                    # Job completed and was auto-removed
                    return

                if job.status == "concluded":
                    if job.error:
                        with contextlib.suppress(QemuStorageDaemonError):
                            await self._execute("job-dismiss", {"id": job_id})
                        raise QemuStorageDaemonError(f"Job {job_id} failed: {job.error}")
                    await self._execute("job-dismiss", {"id": job_id})
                    return

                # Safety net: polling at 50ms (5x slower than before — events are primary)
                await asyncio.sleep(0.05)

            raise QemuStorageDaemonError(f"Job {job_id} timed out after {timeout}s")
        finally:
            # Purge any stale events for this job (e.g. concluded events that
            # arrived after we already observed completion via query-jobs)
            self._purge_job_events(job_id)

    @staticmethod
    def _reap_stale_daemons() -> int:
        """Remove stale QSD sockets from dead parent processes and kill orphaned daemons.

        Socket filenames encode the parent PID: qsd-{pid}-{random}.sock.
        When the parent dies (crash, OOM, kill -9), the daemon becomes an orphan.
        This method finds such orphans and cleans them up.

        Returns:
            Number of stale sockets cleaned up.
        """
        tmpdir = Path(tempfile.gettempdir())
        pattern = re.compile(r"^qsd-(\d+)-[0-9a-f]+\.sock$")
        cleaned = 0

        for sock_path in tmpdir.glob("qsd-*.sock"):
            match = pattern.match(sock_path.name)
            if not match:
                continue

            parent_pid = int(match.group(1))

            # Check if parent PID is alive
            try:
                os.kill(parent_pid, 0)
                # Parent alive, skip
                continue
            except PermissionError:
                # Parent alive but not ours, skip
                continue
            except ProcessLookupError:
                pass  # Parent dead, proceed to cleanup

            # Find and kill the QSD process whose cmdline contains this socket path
            sock_str = str(sock_path)
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):  # type: ignore[reportUnknownMemberType]
                try:
                    cmdline: list[str] = proc.info.get("cmdline") or []  # type: ignore[union-attr]
                    if sock_str in " ".join(cmdline):
                        os.killpg(proc.pid, signal.SIGKILL)
                        logger.debug(
                            "Killed orphaned QSD process",
                            extra={"pid": proc.pid, "socket": sock_str},
                        )
                except (ProcessLookupError, PermissionError, psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Remove stale socket (regardless of whether QSD process was found)
            with contextlib.suppress(OSError):
                sock_path.unlink()
                cleaned += 1
                logger.debug("Removed stale QSD socket", extra={"socket": sock_str})

        return cleaned

    async def _cleanup_connection(self) -> None:
        """Close QMP connection."""
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._writer.wait_closed(), timeout=1.0)
        self._reader = None
        self._writer = None

    async def _cleanup_process(self) -> None:
        """Terminate and cleanup daemon process."""
        if self._process is None:
            return

        # Wait for process to exit
        try:
            await asyncio.wait_for(self._process.wait(), timeout=2.0)
        except TimeoutError:
            # Force kill
            with contextlib.suppress(ProcessLookupError):
                await self._process.terminate()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=2.0)

        self._process = None

    async def __aenter__(self) -> Self:
        """Enter async context manager, starting the daemon."""
        await self.start()
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Exit async context manager, stopping the daemon."""
        await self.stop()

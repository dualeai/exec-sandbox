"""VM working directory management for isolated temp file handling.

Provides atomic directory creation with secure cleanup for all VM-related
temporary files (overlay, sockets).
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Self

from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import await_settled
from exec_sandbox.exceptions import ErrorCode, TransientError
from exec_sandbox.permission_utils import sudo_rm

logger = get_logger(__name__)

_WORKDIR_CLEANUP_TASKS: set[asyncio.Task[None]] = set()


class VmWorkingDirectory:
    """Manages a dedicated temporary directory for a single VM's files.

    All VM-related temporary files (overlay image, sockets) are stored in a
    single directory created atomically via tempfile.mkdtemp() in the temp
    folder. This simplifies cleanup (one rmtree, plus the home disk, which lives
    in `home_dir`) and keeps socket paths under the Unix domain socket limit
    (108 bytes on Linux, 104 on macOS).

    Usage (VmManager: each VM keeps its directory until destroy_vm):
        workdir = await VmWorkingDirectory.create(vm_id, home_dir=home_dir)
        try:
            overlay = workdir.overlay_image
            # VM runs for extended period...
        finally:
            await workdir.cleanup()

    Attributes:
        vm_id: VM identifier for logging
        use_qemu_vm_user: QEMU runs as qemu-vm (set by VmManager): build_qemu_cmd starts it
            through sudo, and cleanup() removes its files with sudo
        path: Root directory path (created by mkdtemp with mode 0700). When QEMU
            runs as qemu-vm, it gets group qemu-vm and mode 2770
            (VmManager._apply_overlay_permissions): other users get no access.
        overlay_image: Path to qcow2 overlay file
        home_disk: Path to the raw home disk file (in `home_dir`, outside `path`)
        cmd_socket: Path to command channel Unix socket (str for QEMU args)
        event_socket: Path to event channel Unix socket (str for QEMU args)
        qmp_socket: Path to QMP control socket
        qmp_events_socket: Path to the QMP socket that carries QEMU events only
        gvproxy_socket: Path to gvproxy network socket
    """

    __slots__ = (
        "_cleaned_up",
        "_cleanup_task",
        "_home_dir",
        "_path",
        "_retry_task",
        "use_qemu_vm_user",
        "vm_id",
    )

    def __init__(self, vm_id: str, path: Path, *, home_dir: Path) -> None:
        """Initialize working directory (private - use create() classmethod).

        Args:
            vm_id: VM identifier for logging
            path: Directory path (create() makes it; VmManager.restore_layout passes one it
                never makes, for the paths only)
            home_dir: Directory for the home disk file: host disk, not the
                (often tmpfs) temp directory that holds `path`
        """
        self.vm_id = vm_id
        self._path = path
        # QEMU runs as qemu-vm (set by VmManager): it owns the files, so cleanup needs sudo
        self.use_qemu_vm_user = False
        self._cleaned_up = False
        self._cleanup_task: asyncio.Task[bool] | None = None
        self._retry_task: asyncio.Task[None] | None = None
        self._home_dir = home_dir

    @classmethod
    async def create(cls, vm_id: str, *, home_dir: Path) -> Self:
        """Create a new working directory atomically.

        Uses tempfile.mkdtemp() for secure, atomic directory creation with
        mode 0700 (owner-only access).

        Args:
            vm_id: VM identifier for logging and directory naming
            home_dir: Directory for the home disk file (see __init__)

        Returns:
            VmWorkingDirectory instance with directory created

        Raises:
            TransientError: LOCAL_DISK, mkdtemp failed (a full disk, no inodes or file
                handles: this host's state, it can pass)
        """
        # Create directory in thread pool (blocking I/O)
        # mkdtemp creates with mode 0700 by default (secure)
        creation_task = asyncio.create_task(
            asyncio.to_thread(
                tempfile.mkdtemp,
                prefix=f"vm-{vm_id[:8]}-",  # Include truncated vm_id for debugging
            )
        )
        try:
            path = await asyncio.shield(creation_task)
        except asyncio.CancelledError:
            # A worker thread cannot be cancelled. Wait for mkdtemp to either
            # fail or publish its path, then remove any directory the caller
            # can no longer receive.
            await await_settled(creation_task)
            if not creation_task.cancelled() and creation_task.exception() is None:
                abandoned = cls(vm_id, Path(creation_task.result()), home_dir=home_dir)
                cleanup_task = asyncio.create_task(abandoned.cleanup())
                await await_settled(cleanup_task)
                cleanup_confirmed = False
                if not cleanup_task.cancelled():
                    try:
                        cleanup_confirmed = cleanup_task.result()
                    except Exception:
                        logger.exception(
                            "Failed to clean abandoned VM working directory",
                            extra={"vm_id": vm_id, "path": str(abandoned.path)},
                        )
                if not cleanup_confirmed:
                    abandoned._retain_cleanup()
            elif not creation_task.cancelled():
                creation_task.exception()
            raise
        except OSError as e:
            raise TransientError(
                f"Could not make the VM working folder: {e.strerror or type(e).__name__}",
                code=ErrorCode.LOCAL_DISK,
                context={"vm_id": vm_id, "folder": tempfile.gettempdir()},
                hint=None,
                instance_id=None,
            ) from e

        logger.debug(
            "Created VM working directory",
            extra={"vm_id": vm_id, "path": path},
        )

        return cls(vm_id, Path(path), home_dir=home_dir)

    @property
    def path(self) -> Path:
        """Root directory path."""
        return self._path

    @property
    def overlay_image(self) -> Path:
        """Path to qcow2 overlay image."""
        return self._path / "overlay.qcow2"

    @property
    def home_disk(self) -> Path:
        """Path to the raw sparse home disk (serial=home, mounted on /home/user).

        Kept out of the working directory: that directory is often a host tmpfs
        (/tmp), where the disk would take host RAM.
        """
        return self._home_dir / f"{self._path.name}.img"

    @property
    def cmd_socket(self) -> str:
        """Path to command channel Unix socket (as string for QEMU args)."""
        return str(self._path / "cmd.sock")

    @property
    def event_socket(self) -> str:
        """Path to event channel Unix socket (as string for QEMU args)."""
        return str(self._path / "event.sock")

    @property
    def qmp_socket(self) -> Path:
        """Path to QMP control socket."""
        return self._path / "qmp.sock"

    @property
    def qmp_events_socket(self) -> Path:
        """Path to the QMP socket that only carries QEMU events (QemuVM.start_event_watch)."""
        return self._path / "qmp-events.sock"

    @property
    def gvproxy_socket(self) -> Path:
        """Path to gvproxy network socket."""
        return self._path / "gvproxy.sock"

    async def cleanup(self) -> bool:
        """Remove the home disk, then the working directory and all its contents.

        Safe to call multiple times (idempotent); concurrent calls share one attempt.
        Uses sudo rm -rf if files are owned by qemu-vm user. A cancelled caller
        gets CancelledError once the attempt has ended; an attempt that failed then
        goes on as a retry task of this module (drain_workdir_cleanup_tasks).

        Returns:
            True if cleanup succeeded, False if errors occurred
        """
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_once())
        cleanup_task = self._cleanup_task
        # The filesystem operation may be running in a worker thread. Do not
        # detach it; establish its result before propagating cancellation.
        cancellation = await await_settled(cleanup_task)

        try:
            cleaned = cleanup_task.result()
        except Exception as error:
            logger.error(
                "VM working directory cleanup task failed",
                extra={"vm_id": self.vm_id, "path": str(self._path)},
                exc_info=error,
            )
            cleaned = False
        if self._cleanup_task is cleanup_task:
            self._cleanup_task = None

        if cancellation is not None:
            if not cleaned:
                self._retain_cleanup()
            raise asyncio.CancelledError
        return cleaned

    def _retain_cleanup(self) -> None:
        """Give retry cleanup strong module ownership.

        This module-level retry can coexist with VmManager's cleanup reaper
        for the same directory (cancellation mid-destroy). That is safe:
        cleanup() is single-flight via _cleanup_task, so concurrent owners
        serialize on one removal attempt instead of racing rm -rf.
        """
        if self._retry_task is not None and not self._retry_task.done():
            return
        task = asyncio.create_task(_retry_workdir_cleanup(self))
        self._retry_task = task
        _WORKDIR_CLEANUP_TASKS.add(task)

        def cleanup_done(completed: asyncio.Task[None]) -> None:
            _WORKDIR_CLEANUP_TASKS.discard(completed)
            if self._retry_task is completed:
                self._retry_task = None
            if completed.cancelled():
                return
            if error := completed.exception():
                logger.error(
                    "VM working directory cleanup task failed",
                    extra={"vm_id": self.vm_id, "path": str(self.path)},
                    exc_info=error,
                )

        task.add_done_callback(cleanup_done)

    async def _cleanup_once(self) -> bool:
        """Perform one retryable cleanup attempt."""
        if self._cleaned_up:
            logger.debug(
                "VM working directory already cleaned up",
                extra={"vm_id": self.vm_id, "path": str(self._path)},
            )
            return True

        # The home disk lives outside the working directory (the host owns it)
        await asyncio.to_thread(self.home_disk.unlink, missing_ok=True)

        if not self._path.exists():
            self._cleaned_up = True
            logger.debug(
                "VM working directory already removed",
                extra={"vm_id": self.vm_id, "path": str(self._path)},
            )
            return True

        try:
            if self.use_qemu_vm_user:
                # Files owned by qemu-vm user, need sudo
                if not await sudo_rm(self._path):
                    logger.error(
                        "VM working directory sudo rm failed",
                        extra={
                            "vm_id": self.vm_id,
                            "path": str(self._path),
                        },
                    )
                    return False
            else:
                # Normal cleanup via shutil.rmtree (async via thread pool)
                await asyncio.to_thread(shutil.rmtree, self._path)

            if self._path.exists():
                logger.error(
                    "VM working directory still exists after cleanup",
                    extra={"vm_id": self.vm_id, "path": str(self._path)},
                )
                return False

            self._cleaned_up = True
            logger.debug(
                "VM working directory cleaned up",
                extra={"vm_id": self.vm_id, "path": str(self._path)},
            )
            return True

        except Exception as e:
            logger.error(
                "VM working directory cleanup error",
                extra={
                    "vm_id": self.vm_id,
                    "path": str(self._path),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return False


async def _retry_workdir_cleanup(workdir: VmWorkingDirectory) -> None:
    """Retain a directory owner until removal is confirmed."""
    delay = 0.05
    while not await workdir.cleanup():
        logger.error(
            "Retrying abandoned VM working directory cleanup",
            extra={"vm_id": workdir.vm_id, "path": str(workdir.path), "retry_delay_seconds": delay},
        )
        await asyncio.sleep(delay)
        delay = min(delay * 2, 5.0)


async def drain_workdir_cleanup_tasks() -> None:
    """Wait until every module-owned abandoned-directory retry is finished (also the ones
    that start meanwhile)."""
    while tasks := tuple(_WORKDIR_CLEANUP_TASKS):
        await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
        _WORKDIR_CLEANUP_TASKS.difference_update(tasks)

"""VM working directory management for isolated temp file handling.

Provides atomic directory creation with secure cleanup for all VM-related
temporary files (overlay, sockets).
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Self

from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import await_settled
from exec_sandbox.permission_utils import sudo_rm

if TYPE_CHECKING:
    from types import TracebackType

logger = get_logger(__name__)

_WORKDIR_CLEANUP_TASKS: set[asyncio.Task[None]] = set()


class VmWorkingDirectory:
    """Manages a dedicated temporary directory for a single VM's files.

    All VM-related temporary files (overlay image, sockets) are stored in a
    single directory created atomically via tempfile.mkdtemp().
    This simplifies cleanup (single rmtree) and ensures socket paths stay
    under the 108-byte Unix domain socket limit.

    Usage as context manager (preferred for short-lived VMs):
        async with await VmWorkingDirectory.create(vm_id) as workdir:
            overlay = workdir.overlay_image
            # VM runs...
        # Directory automatically cleaned up

    Usage with manual lifecycle (for warm pool VMs):
        workdir = await VmWorkingDirectory.create(vm_id)
        try:
            overlay = workdir.overlay_image
            # VM runs for extended period...
        finally:
            await workdir.cleanup()

    Attributes:
        vm_id: VM identifier for logging
        path: Root directory path (created by mkdtemp with mode 0700)
        overlay_image: Path to qcow2 overlay file
        cmd_socket: Path to command channel Unix socket (str for QEMU args)
        event_socket: Path to event channel Unix socket (str for QEMU args)
        qmp_socket: Path to QMP control socket
        gvproxy_socket: Path to gvproxy network socket
    """

    __slots__ = (
        "_cleaned_up",
        "_cleanup_task",
        "_custom_overlay_path",
        "_path",
        "_retry_task",
        "_use_qemu_vm_user",
        "vm_id",
    )

    def __init__(
        self,
        vm_id: str,
        path: Path,
        use_qemu_vm_user: bool = False,
        custom_overlay_path: Path | None = None,
    ) -> None:
        """Initialize working directory (private - use create() classmethod).

        Args:
            vm_id: VM identifier for logging
            path: Directory path (must already exist)
            use_qemu_vm_user: Whether files will be owned by qemu-vm user
            custom_overlay_path: Override overlay path (for skip_overlay mode)
        """
        self.vm_id = vm_id
        self._path = path
        self._use_qemu_vm_user = use_qemu_vm_user
        self._cleaned_up = False
        self._cleanup_task: asyncio.Task[bool] | None = None
        self._retry_task: asyncio.Task[None] | None = None
        self._custom_overlay_path = custom_overlay_path

    @classmethod
    async def create(cls, vm_id: str, custom_overlay_path: Path | None = None) -> Self:
        """Create a new working directory atomically.

        Uses tempfile.mkdtemp() for secure, atomic directory creation with
        mode 0700 (owner-only access).

        Args:
            vm_id: VM identifier for logging and directory naming
            custom_overlay_path: Override overlay path (for skip_overlay mode)

        Returns:
            VmWorkingDirectory instance with directory created
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
                abandoned = cls(vm_id, Path(creation_task.result()), custom_overlay_path=custom_overlay_path)
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

        logger.debug(
            "Created VM working directory",
            extra={"vm_id": vm_id, "path": path},
        )

        return cls(vm_id, Path(path), custom_overlay_path=custom_overlay_path)

    @property
    def path(self) -> Path:
        """Root directory path."""
        return self._path

    @property
    def overlay_image(self) -> Path:
        """Path to qcow2 overlay image.

        Returns custom path if set via constructor (for skip_overlay mode), otherwise default.
        """
        if self._custom_overlay_path is not None:
            return self._custom_overlay_path
        return self._path / "overlay.qcow2"

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
    def gvproxy_socket(self) -> Path:
        """Path to gvproxy network socket."""
        return self._path / "gvproxy.sock"

    @property
    def use_qemu_vm_user(self) -> bool:
        """Whether files are owned by qemu-vm user (requires sudo for cleanup)."""
        return self._use_qemu_vm_user

    @use_qemu_vm_user.setter
    def use_qemu_vm_user(self, value: bool) -> None:
        """Set qemu-vm user ownership flag."""
        self._use_qemu_vm_user = value

    async def cleanup(self) -> bool:
        """Remove working directory and all contents.

        Safe to call multiple times (idempotent).
        Uses sudo rm -rf if files are owned by qemu-vm user.

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
        except Exception as error:  # noqa: BLE001 - preserve a retry owner for every failure
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

        if not self._path.exists():
            self._cleaned_up = True
            logger.debug(
                "VM working directory already removed",
                extra={"vm_id": self.vm_id, "path": str(self._path)},
            )
            return True

        try:
            if self._use_qemu_vm_user:
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

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> bool:
        """Exit async context manager - cleanup directory."""
        if not await self.cleanup():
            self._retain_cleanup()
        return False  # Don't suppress exceptions


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
    """Wait until every module-owned abandoned-directory retry is finished."""
    while True:
        tasks = tuple(task for task in _WORKDIR_CLEANUP_TASKS if not task.done())
        if not tasks:
            _WORKDIR_CLEANUP_TASKS.difference_update(task for task in _WORKDIR_CLEANUP_TASKS if task.done())
            if not _WORKDIR_CLEANUP_TASKS:
                return
            await asyncio.sleep(0)
            continue
        await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

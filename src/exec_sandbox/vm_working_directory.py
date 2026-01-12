"""VM working directory management for isolated temp file handling.

Provides atomic directory creation with secure cleanup for all VM-related
temporary files (overlay, sockets, logs).
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Self

from exec_sandbox._logging import get_logger

if TYPE_CHECKING:
    from types import TracebackType

logger = get_logger(__name__)


class VmWorkingDirectory:
    """Manages a dedicated temporary directory for a single VM's files.

    All VM-related temporary files (overlay image, sockets, console log) are
    stored in a single directory created atomically via tempfile.mkdtemp().
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
        console_log: Path to console log file
        cmd_socket: Path to command channel Unix socket (str for QEMU args)
        event_socket: Path to event channel Unix socket (str for QEMU args)
        qmp_socket: Path to QMP control socket
        gvproxy_socket: Path to gvproxy network socket
    """

    __slots__ = (
        "_cleaned_up",
        "_path",
        "_use_qemu_vm_user",
        "vm_id",
    )

    def __init__(self, vm_id: str, path: Path, use_qemu_vm_user: bool = False) -> None:
        """Initialize working directory (private - use create() classmethod).

        Args:
            vm_id: VM identifier for logging
            path: Directory path (must already exist)
            use_qemu_vm_user: Whether files will be owned by qemu-vm user
        """
        self.vm_id = vm_id
        self._path = path
        self._use_qemu_vm_user = use_qemu_vm_user
        self._cleaned_up = False

    @classmethod
    async def create(cls, vm_id: str) -> Self:
        """Create a new working directory atomically.

        Uses tempfile.mkdtemp() for secure, atomic directory creation with
        mode 0700 (owner-only access).

        Args:
            vm_id: VM identifier for logging and directory naming

        Returns:
            VmWorkingDirectory instance with directory created
        """
        # Create directory in thread pool (blocking I/O)
        # mkdtemp creates with mode 0700 by default (secure)
        path = await asyncio.to_thread(
            tempfile.mkdtemp,
            prefix=f"vm-{vm_id[:8]}-",  # Include truncated vm_id for debugging
        )

        logger.debug(
            "Created VM working directory",
            extra={"vm_id": vm_id, "path": path},
        )

        return cls(vm_id, Path(path))

    @property
    def path(self) -> Path:
        """Root directory path."""
        return self._path

    @property
    def overlay_image(self) -> Path:
        """Path to qcow2 overlay image."""
        return self._path / "overlay.qcow2"

    @property
    def console_log(self) -> Path:
        """Path to console log file."""
        return self._path / "console.log"

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
        if self._cleaned_up:
            logger.debug(
                "VM working directory already cleaned up",
                extra={"vm_id": self.vm_id, "path": str(self._path)},
            )
            return True

        self._cleaned_up = True

        if not self._path.exists():
            logger.debug(
                "VM working directory already removed",
                extra={"vm_id": self.vm_id, "path": str(self._path)},
            )
            return True

        try:
            if self._use_qemu_vm_user:
                # Files owned by qemu-vm user, need sudo
                proc = await asyncio.create_subprocess_exec(
                    "sudo",
                    "rm",
                    "-rf",
                    str(self._path),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
                if proc.returncode != 0:
                    logger.error(
                        "VM working directory sudo rm failed",
                        extra={
                            "vm_id": self.vm_id,
                            "path": str(self._path),
                            "returncode": proc.returncode,
                        },
                    )
                    return False
            else:
                # Normal cleanup via shutil.rmtree (async via thread pool)
                await asyncio.to_thread(shutil.rmtree, self._path, ignore_errors=True)

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
        await self.cleanup()
        return False  # Don't suppress exceptions

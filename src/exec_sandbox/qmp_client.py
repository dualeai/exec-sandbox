"""QMP (QEMU Monitor Protocol) client for memory snapshots.

This module provides an async wrapper around the qemu.qmp library for
controlling QEMU instances via the QMP protocol. Primary use case is
creating and restoring memory snapshots for fast VM restore.
"""

import asyncio
import types
from pathlib import Path
from typing import Any

from qemu.qmp import QMPClient  # type: ignore[import-untyped]

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import SnapshotError

# Minimum parts needed to identify a snapshot entry in info snapshots output
_MIN_SNAPSHOT_PARTS = 2

_logger = get_logger(__name__)


class QMPClientWrapper:
    """Async QMP client for QEMU control.

    Provides a high-level interface for QMP operations, specifically
    focused on memory snapshot management via savevm/loadvm commands.

    Usage:
        async with QMPClientWrapper(socket_path) as qmp:
            await qmp.save_snapshot("ready")
    """

    def __init__(self, socket_path: str | Path):
        """Initialize QMP client wrapper.

        Args:
            socket_path: Path to QEMU QMP Unix socket.
        """
        self._socket_path = str(socket_path)
        self._client: QMPClient | None = None

    async def __aenter__(self) -> "QMPClientWrapper":
        """Context manager entry - connect to QMP socket."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit - disconnect from QMP socket."""
        await self.disconnect()

    async def connect(self, timeout: float = 5.0) -> None:
        """Connect to QMP socket.

        Args:
            timeout: Connection timeout in seconds.

        Raises:
            SnapshotError: If connection fails or times out.
        """
        self._client = QMPClient("exec-sandbox")
        try:
            await asyncio.wait_for(
                self._client.connect(self._socket_path),
                timeout=timeout,
            )
            _logger.debug("Connected to QMP socket: %s", self._socket_path)
        except TimeoutError as e:
            await self._cleanup_client()
            msg = f"QMP connection timed out after {timeout}s"
            raise SnapshotError(msg, {"socket": self._socket_path}) from e
        except OSError as e:
            await self._cleanup_client()
            msg = f"QMP connection failed: {e}"
            raise SnapshotError(msg, {"socket": self._socket_path}) from e

    async def disconnect(self) -> None:
        """Disconnect from QMP socket.

        Safe to call multiple times or if never connected.
        """
        await self._cleanup_client()

    async def _cleanup_client(self) -> None:
        """Internal cleanup of QMP client."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:  # noqa: BLE001 - Best effort cleanup
                _logger.debug("QMP disconnect error (ignored)", exc_info=True)
            finally:
                self._client = None

    async def save_snapshot(self, name: str, timeout: float = 30.0) -> None:
        """Save VM memory snapshot.

        Creates an internal snapshot in the qcow2 disk image containing:
        - CPU register state
        - RAM contents
        - Device state (virtio-serial, virtio-blk, etc.)

        Args:
            name: Snapshot name (e.g., "ready", "golden").
            timeout: Operation timeout in seconds.

        Raises:
            SnapshotError: If snapshot creation fails or times out.
        """
        if self._client is None:
            msg = "QMP client not connected"
            raise SnapshotError(msg)

        _logger.debug("Creating memory snapshot: %s", name)

        try:
            # Use human-monitor-command since savevm is not a native QMP command
            # First sync disk to ensure consistency
            await asyncio.wait_for(
                self._client.execute(
                    "human-monitor-command",
                    {"command-line": "sync"},
                ),
                timeout=timeout / 3,  # Use 1/3 of timeout for sync
            )

            # Now save the snapshot
            result = await asyncio.wait_for(
                self._client.execute(
                    "human-monitor-command",
                    {"command-line": f"savevm {name}"},
                ),
                timeout=timeout * 2 / 3,  # Use 2/3 of timeout for savevm
            )

            # Check for errors in response
            if result and isinstance(result, str) and "error" in result.lower():
                msg = f"savevm failed: {result}"
                raise SnapshotError(msg, {"snapshot": name, "response": result})

            _logger.info("Memory snapshot created: %s", name)

        except TimeoutError as e:
            msg = f"savevm timed out after {timeout}s"
            raise SnapshotError(msg, {"snapshot": name}) from e

    async def delete_snapshot(self, name: str, timeout: float = 10.0) -> None:
        """Delete VM memory snapshot.

        Args:
            name: Snapshot name to delete.
            timeout: Operation timeout in seconds.

        Raises:
            SnapshotError: If deletion fails or times out.
        """
        if self._client is None:
            msg = "QMP client not connected"
            raise SnapshotError(msg)

        _logger.debug("Deleting memory snapshot: %s", name)

        try:
            result = await asyncio.wait_for(
                self._client.execute(
                    "human-monitor-command",
                    {"command-line": f"delvm {name}"},
                ),
                timeout=timeout,
            )

            if result and isinstance(result, str) and "error" in result.lower():
                msg = f"delvm failed: {result}"
                raise SnapshotError(msg, {"snapshot": name, "response": result})

            _logger.debug("Memory snapshot deleted: %s", name)

        except TimeoutError as e:
            msg = f"delvm timed out after {timeout}s"
            raise SnapshotError(msg, {"snapshot": name}) from e

    async def query_snapshots(self, timeout: float = 10.0) -> list[dict[str, Any]]:
        """List all snapshots in the qcow2 image.

        Returns:
            List of snapshot info dicts with keys: id, name, vm-state-size, etc.

        Raises:
            SnapshotError: If query fails or times out.
        """
        if self._client is None:
            msg = "QMP client not connected"
            raise SnapshotError(msg)

        try:
            result = await asyncio.wait_for(
                self._client.execute(
                    "human-monitor-command",
                    {"command-line": "info snapshots"},
                ),
                timeout=timeout,
            )

            # Parse the text output (info snapshots returns text, not JSON)
            # Format: "ID  TAG                 VM SIZE  DATE          VM CLOCK"
            snapshots: list[dict[str, Any]] = []
            if result and isinstance(result, str):
                lines = result.strip().split("\n")
                for line in lines[1:]:  # Skip header
                    parts = line.split()
                    if len(parts) >= _MIN_SNAPSHOT_PARTS:
                        snapshots.append({"id": parts[0], "name": parts[1]})

            return snapshots

        except TimeoutError as e:
            msg = f"info snapshots timed out after {timeout}s"
            raise SnapshotError(msg) from e

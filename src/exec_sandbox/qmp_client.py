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
from exec_sandbox.socket_auth import SocketAuthError, verify_socket_peer

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

    def __init__(self, socket_path: str | Path, expected_uid: int):
        """Initialize QMP client wrapper.

        Args:
            socket_path: Path to QEMU QMP Unix socket.
            expected_uid: Expected UID of QEMU process for peer verification (required).
        """
        self._socket_path = str(socket_path)
        self._expected_uid = expected_uid
        self._client: QMPClient | None = None

    async def __aenter__(self) -> "QMPClientWrapper":
        """Context manager entry - connect to QMP socket."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit - disconnect from QMP socket."""
        await self.disconnect()

    async def connect(self, timeout: float = 5.0) -> None:
        """Connect to QMP socket with mandatory peer verification.

        Args:
            timeout: Connection timeout in seconds.

        Raises:
            SnapshotError: If connection fails, times out, or peer verification fails.
        """
        self._client = QMPClient("exec-sandbox")
        try:
            await asyncio.wait_for(
                self._client.connect(self._socket_path),
                timeout=timeout,
            )

            # Verify peer credentials after connection (mandatory)
            # Access underlying socket from QMPClient's internal asyncio protocol.
            # Uses getattr() because:
            # 1. qemu.qmp is untyped (see import with type: ignore)
            # 2. _protocol/_transport are private attrs that may change between versions
            # 3. Defensive fallback to None avoids AttributeError if internals change
            protocol = getattr(self._client, "_protocol", None)
            transport = getattr(protocol, "_transport", None) if protocol else None
            sock = transport.get_extra_info("socket") if transport else None

            if sock is not None:
                verify_socket_peer(sock, self._expected_uid, self._socket_path)
            else:
                # Cannot access socket - fail authentication
                # This could indicate qemu.qmp library internals changed
                await self._cleanup_client()
                raise SocketAuthError(
                    "Cannot verify QMP socket peer: unable to access underlying socket",
                    expected_uid=self._expected_uid,
                    actual_uid=0,
                    context={
                        "socket": self._socket_path,
                        "has_protocol": protocol is not None,
                        "has_transport": transport is not None,
                    },
                )

            _logger.debug("Connected to QMP socket: %s", self._socket_path)
        except TimeoutError as e:
            await self._cleanup_client()
            msg = f"QMP connection timed out after {timeout}s"
            raise SnapshotError(msg, {"socket": self._socket_path}) from e
        except SocketAuthError as e:
            await self._cleanup_client()
            msg = f"QMP socket authentication failed: {e}"
            raise SnapshotError(msg, {"socket": self._socket_path, **e.context}) from e
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

    # =========================================================================
    # Balloon operations for memory reclamation
    # =========================================================================

    async def balloon_set_target(self, target_mb: int, timeout: float = 5.0) -> None:
        """Set balloon target size (actual guest memory).

        Inflates/deflates the balloon to adjust guest-visible memory.
        Guest will see reduced memory when balloon is inflated.

        Args:
            target_mb: Target guest memory in MB.
            timeout: Operation timeout in seconds.

        Raises:
            SnapshotError: If balloon operation fails or times out.
        """
        if self._client is None:
            msg = "QMP client not connected"
            raise SnapshotError(msg)

        target_bytes = target_mb * 1024 * 1024
        try:
            await asyncio.wait_for(
                self._client.execute("balloon", {"value": target_bytes}),
                timeout=timeout,
            )
            _logger.debug("Balloon target set to %dMB", target_mb)
        except TimeoutError as e:
            msg = f"balloon timed out after {timeout}s"
            raise SnapshotError(msg, {"target_mb": target_mb}) from e
        except Exception as e:
            # Balloon device may not be present - graceful failure
            msg = f"balloon failed: {e}"
            raise SnapshotError(msg, {"target_mb": target_mb}) from e

    async def balloon_query(self, timeout: float = 5.0) -> dict[str, Any] | None:
        """Query current balloon status.

        Returns:
            Dict with 'actual' (current guest memory in bytes), or None if
            balloon device is not available.

        Raises:
            SnapshotError: If query fails or times out (except for missing device).
        """
        if self._client is None:
            msg = "QMP client not connected"
            raise SnapshotError(msg)

        try:
            result = await asyncio.wait_for(
                self._client.execute("query-balloon"),
                timeout=timeout,
            )
            # QMP returns dict with "actual" key containing bytes
            if isinstance(result, dict):
                return dict(result)  # type: ignore[arg-type]
            return None
        except TimeoutError as e:
            msg = f"query-balloon timed out after {timeout}s"
            raise SnapshotError(msg) from e
        except Exception as e:  # noqa: BLE001 - Balloon device may not be present
            _logger.debug("query-balloon failed (balloon device may be missing): %s", e)
            return None

    async def balloon_deflate_for_snapshot(
        self,
        original_mb: int,
        min_mb: int = 64,
        timeout: float = 10.0,
    ) -> int:
        """Deflate balloon to reclaim free pages before snapshot.

        Reduces guest memory to minimum viable size, waits for stabilization,
        then returns the achieved size. This reduces snapshot size by excluding
        unused guest memory pages.

        Args:
            original_mb: Original guest memory allocation in MB.
            min_mb: Minimum target memory in MB (default 64MB floor).
            timeout: Maximum time to wait for deflation.

        Returns:
            Actual guest memory in MB after deflation, or original_mb if
            balloon is unavailable.
        """
        # Check if balloon is available
        status = await self.balloon_query(timeout / 3)
        if status is None:
            _logger.debug("Balloon not available, skipping deflation")
            return original_mb

        try:
            # Set aggressive deflation target
            await self.balloon_set_target(min_mb, timeout / 3)

            # Wait for deflation to stabilize (guest needs time to return pages)
            await asyncio.sleep(0.5)

            # Query actual result
            status = await self.balloon_query(timeout / 3)
            if status is None:
                return original_mb

            actual_bytes = status.get("actual", original_mb * 1024 * 1024)
            actual_mb = actual_bytes // (1024 * 1024)

            saved_mb = original_mb - actual_mb
            if saved_mb > 0:
                _logger.info(
                    "Balloon deflated: %dMB -> %dMB (saved %dMB)",
                    original_mb,
                    actual_mb,
                    saved_mb,
                )
            else:
                _logger.debug("Balloon deflation: no memory reclaimed")

            return actual_mb

        except SnapshotError as e:
            # Log but continue - balloon is optional optimization
            _logger.warning("Balloon deflation failed: %s", e)
            return original_mb

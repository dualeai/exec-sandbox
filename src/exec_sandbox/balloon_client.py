"""QMP client for balloon memory control.

Provides memory deflation/inflation for warm pool VMs to reduce host memory
pressure when VMs are idle.

Architecture:
- Connects to QEMU's QMP (QEMU Monitor Protocol) Unix socket
- Uses socket peer credential authentication (same as guest channel)
- Balloon device reclaims memory by inflating a memory balloon inside the guest
- Deflating the balloon returns memory to the host

Usage:
    client = BalloonClient(qmp_socket_path, expected_uid)
    await client.connect()
    try:
        await client.deflate(target_mb=64)  # Reduce guest memory to 64MB
        # ... VM waits in pool ...
        await client.inflate(target_mb=256)  # Restore memory before execution
    finally:
        await client.disconnect()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_result,
    stop_after_attempt,
    wait_fixed,
)

from exec_sandbox import constants

if TYPE_CHECKING:
    from pathlib import Path
from exec_sandbox._logging import get_logger
from exec_sandbox.socket_auth import connect_and_verify

logger = get_logger(__name__)


class BalloonError(Exception):
    """Balloon operation failed."""


class BalloonClient:
    """QMP client for balloon memory control.

    Thread-safe: Uses asyncio lock for connection state.

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
        "_writer",
    )

    def __init__(self, qmp_socket: Path, expected_uid: int) -> None:
        """Initialize balloon client.

        Args:
            qmp_socket: Path to QMP Unix socket
            expected_uid: Expected UID of QEMU process (for authentication)
        """
        self._qmp_socket = qmp_socket
        self._expected_uid = expected_uid
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self, timeout: float = 5.0) -> None:
        """Connect to QMP socket and complete handshake.

        Args:
            timeout: Connection timeout in seconds

        Raises:
            BalloonError: Connection or handshake failed
        """
        async with self._lock:
            if self._connected:
                return

            try:
                # Connect with socket authentication
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
                    "QMP greeting received",
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
                    raise BalloonError(f"QMP capabilities failed: {resp_data['error']}")

                self._connected = True
                logger.debug("QMP connection established")

            except (OSError, TimeoutError, json.JSONDecodeError) as e:
                await self._cleanup()
                raise BalloonError(f"QMP connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from QMP socket."""
        async with self._lock:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Internal cleanup (must hold lock)."""
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._writer.wait_closed(), timeout=1.0)
        self._reader = None
        self._writer = None
        self._connected = False

    async def _execute(
        self,
        command: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Execute QMP command.

        Args:
            command: QMP command name
            arguments: Optional command arguments
            timeout: Command timeout in seconds

        Returns:
            QMP response dict

        Raises:
            BalloonError: Command failed or not connected
        """
        if not self._connected or self._writer is None or self._reader is None:
            raise BalloonError("Not connected to QMP")

        cmd: dict[str, Any] = {"execute": command}
        if arguments:
            cmd["arguments"] = arguments

        self._writer.write(json.dumps(cmd).encode() + b"\n")
        await self._writer.drain()

        response = await asyncio.wait_for(
            self._reader.readline(),
            timeout=timeout,
        )
        return json.loads(response)

    async def query(self, timeout: float = 5.0) -> int | None:
        """Query current balloon memory.

        Args:
            timeout: Query timeout in seconds

        Returns:
            Current balloon target in MB, or None if balloon not available
        """
        try:
            resp = await self._execute("query-balloon", timeout=timeout)
            if "error" in resp:
                # Device error (e.g., balloon not available)
                logger.warning("Balloon query failed", extra={"error": resp["error"]})
                return None
            if "return" in resp:
                # QMP returns bytes, convert to MB
                actual_bytes = resp["return"].get("actual")
                if actual_bytes is not None:
                    return actual_bytes // (1024 * 1024)
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Balloon query failed", extra={"error": str(e)})
            return None

    async def set_target(self, target_mb: int, timeout: float = 5.0) -> None:
        """Set balloon memory target.

        Args:
            target_mb: Target memory in MB
            timeout: Command timeout in seconds

        Raises:
            BalloonError: Command failed
        """
        # QMP expects bytes
        target_bytes = target_mb * 1024 * 1024

        resp = await self._execute(
            "balloon",
            {"value": target_bytes},
            timeout=timeout,
        )

        if "error" in resp:
            raise BalloonError(f"Balloon set failed: {resp['error']}")

        logger.debug(
            "Balloon target set",
            extra={"target_mb": target_mb},
        )

    async def inflate(
        self,
        target_mb: int = constants.BALLOON_INFLATE_TARGET_MB,
        timeout: float = constants.BALLOON_INFLATE_TIMEOUT_SECONDS,
        wait_for_target: bool = True,
    ) -> int:
        """Inflate balloon to reduce guest memory for idle pool VMs.

        Inflating the balloon takes memory FROM the guest, allowing the host
        to reclaim it. Used when adding VM to warm pool.

        Note: Balloon operations are asynchronous - the guest needs time to
        respond to balloon requests. This method polls until the target is
        reached, following QEMU's own test patterns (10 retries @ 0.5s).
        See: https://www.mail-archive.com/qemu-devel@nongnu.org/msg1102693.html

        Args:
            target_mb: Target guest memory in MB (default: BALLOON_INFLATE_TARGET_MB)
            timeout: Operation timeout in seconds
            wait_for_target: If True, poll until balloon reaches target

        Returns:
            Previous guest memory in MB (for deflate restore)

        Raises:
            BalloonError: Inflation failed
        """
        # Query current size for restore
        current_mb = await self.query(timeout=timeout)
        if current_mb is None:
            logger.warning("Could not query balloon, assuming default memory")
            current_mb = constants.DEFAULT_MEMORY_MB

        await self.set_target(target_mb, timeout=timeout)

        # Balloon operations are asynchronous - guest needs time to respond.
        # Poll until target is reached, following QEMU test patterns.
        if wait_for_target:

            async def _check_balloon() -> int | None:
                return await self.query(timeout=1.0)

            def _not_at_target(result: int | None) -> bool:
                """Retry while balloon hasn't reached target."""
                return result is None or result > target_mb + constants.BALLOON_TOLERANCE_MB

            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(10),
                    wait=wait_fixed(0.5),
                    retry=retry_if_result(_not_at_target),
                    before_sleep=before_sleep_log(logger, logging.DEBUG),
                    reraise=True,
                ):
                    with attempt:
                        actual_mb = await _check_balloon()
                        if not _not_at_target(actual_mb):
                            logger.debug(
                                "Balloon reached target",
                                extra={"actual_mb": actual_mb, "target_mb": target_mb},
                            )
            except RetryError:
                # Log warning but don't fail - balloon may still be inflating
                actual_mb = await self.query(timeout=1.0)
                logger.warning(
                    "Balloon did not reach target within retry limit",
                    extra={"actual_mb": actual_mb, "target_mb": target_mb},
                )

        logger.info(
            "Balloon inflated",
            extra={"from_mb": current_mb, "to_mb": target_mb},
        )

        return current_mb

    async def deflate(
        self,
        target_mb: int,
        timeout: float = constants.BALLOON_DEFLATE_TIMEOUT_SECONDS,
        wait_for_target: bool = True,
    ) -> None:
        """Deflate balloon to restore guest memory before code execution.

        Deflating the balloon returns memory TO the guest. Used before
        executing user code.

        Note: Like inflate, balloon operations are asynchronous - the guest
        needs time to reclaim memory from the balloon. This method polls
        until the target is reached.

        Args:
            target_mb: Target guest memory in MB
            timeout: Operation timeout in seconds
            wait_for_target: If True, poll until balloon reaches target

        Raises:
            BalloonError: Deflation failed
        """
        await self.set_target(target_mb, timeout=timeout)

        # Balloon operations are asynchronous - guest needs time to reclaim memory.
        # Poll until target is reached, following QEMU test patterns.
        if wait_for_target:

            async def _check_balloon() -> int | None:
                return await self.query(timeout=1.0)

            def _not_at_target(result: int | None) -> bool:
                """Retry while balloon hasn't reached target (memory still too low)."""
                return result is None or result < target_mb - constants.BALLOON_TOLERANCE_MB

            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(10),
                    wait=wait_fixed(0.5),
                    retry=retry_if_result(_not_at_target),
                    before_sleep=before_sleep_log(logger, logging.DEBUG),
                    reraise=True,
                ):
                    with attempt:
                        actual_mb = await _check_balloon()
                        if not _not_at_target(actual_mb):
                            logger.debug(
                                "Balloon reached deflate target",
                                extra={"actual_mb": actual_mb, "target_mb": target_mb},
                            )
            except RetryError:
                # Log warning but don't fail - balloon may still be deflating
                actual_mb = await self.query(timeout=1.0)
                logger.warning(
                    "Balloon did not reach deflate target within retry limit",
                    extra={"actual_mb": actual_mb, "target_mb": target_mb},
                )

        logger.info(
            "Balloon deflated",
            extra={"target_mb": target_mb},
        )

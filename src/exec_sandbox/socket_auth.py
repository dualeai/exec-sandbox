"""Unix socket peer credential authentication.

Uses SO_PEERCRED on Linux and LOCAL_PEERCRED on macOS to verify
socket server identity before sending commands.

Security Model:
- Host (Python) connects to socket created by QEMU
- Before sending sensitive commands, verify QEMU process UID matches expected user
- Prevents connecting to attacker-controlled sockets at same path
- Credentials captured at connect() time - no TOCTOU race condition
- Authentication is MANDATORY - never skipped
"""

from __future__ import annotations

import asyncio
import pwd
import socket
import struct
from dataclasses import dataclass

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import SocketAuthError
from exec_sandbox.permission_utils import get_qemu_vm_uid
from exec_sandbox.platform_utils import HostOS, detect_host_os

logger = get_logger(__name__)

# Platform constants for SO_PEERCRED (Linux)
_SO_PEERCRED = 17  # Linux socket option
_UCRED_SIZE = 12  # struct ucred: pid(i), uid(I), gid(I)

# Platform constants for LOCAL_PEERCRED (macOS)
_LOCAL_PEERCRED = 0x001  # macOS SOL_LOCAL value
_XUCRED_SIZE = 76  # struct xucred size

# Re-export SocketAuthError for convenience
__all__ = [
    "PeerCredentials",
    "SocketAuthError",
    "connect_and_verify",
    "create_unix_socket",
    "get_peer_credentials",
    "get_qemu_vm_uid",
    "verify_socket_peer",
]


@dataclass(frozen=True)
class PeerCredentials:
    """Peer process credentials from Unix socket.

    Attributes:
        uid: User ID of peer process
        gid: Group ID of peer process (Linux only, None on macOS)
        pid: Process ID of peer process (Linux only, None on macOS)
    """

    uid: int
    gid: int | None = None
    pid: int | None = None


def get_peer_credentials(sock: socket.socket) -> PeerCredentials:
    """Get peer credentials from connected Unix socket.

    Args:
        sock: Connected Unix domain socket

    Returns:
        PeerCredentials from the socket peer

    Raises:
        SocketAuthError: If credentials cannot be retrieved (unsupported platform,
            invalid socket type, or socket error)
    """
    # Validate socket family - only Unix domain sockets support peer credentials
    if sock.family != socket.AF_UNIX:
        raise SocketAuthError(
            f"Socket authentication requires Unix domain socket, got socket family {sock.family}",
            expected_uid=0,
            actual_uid=0,
            context={"socket_family": str(sock.family)},
        )

    host_os = detect_host_os()
    if host_os == HostOS.LINUX:
        return _get_peer_credentials_linux(sock)
    if host_os == HostOS.MACOS:
        return _get_peer_credentials_macos(sock)

    raise SocketAuthError(
        f"Socket authentication not supported on platform: {host_os}",
        expected_uid=0,
        actual_uid=0,
        context={"platform": str(host_os)},
    )


def _get_peer_credentials_linux(sock: socket.socket) -> PeerCredentials:
    """Get peer credentials on Linux via SO_PEERCRED.

    Returns:
        PeerCredentials with pid, uid, gid

    Raises:
        SocketAuthError: If credentials cannot be retrieved or socket is not connected
    """
    try:
        data = sock.getsockopt(socket.SOL_SOCKET, _SO_PEERCRED, _UCRED_SIZE)
        pid, uid, gid = struct.unpack("iII", data)

        # PID 0 indicates unconnected socket or invalid credentials
        # (PID 0 is the kernel swapper process, never a valid peer)
        if pid == 0:
            raise SocketAuthError(
                "Socket is not connected or has invalid peer credentials (pid=0)",
                expected_uid=0,
                actual_uid=uid,
                context={"pid": pid, "uid": uid, "gid": gid, "platform": "linux"},
            )

        return PeerCredentials(uid=uid, gid=gid, pid=pid)
    except (OSError, struct.error) as e:
        raise SocketAuthError(
            f"Failed to get peer credentials: {e}",
            expected_uid=0,
            actual_uid=0,
            context={"error": str(e), "platform": "linux"},
        ) from e


def _get_peer_credentials_macos(sock: socket.socket) -> PeerCredentials:
    """Get peer credentials on macOS via LOCAL_PEERCRED.

    Note: macOS doesn't provide PID via LOCAL_PEERCRED, so pid=None.

    Returns:
        PeerCredentials with uid (gid and pid are None)

    Raises:
        SocketAuthError: If credentials cannot be retrieved
    """
    try:
        # LOCAL_PEERCRED returns struct xucred:
        # uint32_t cr_version (offset 0)
        # uid_t cr_uid (offset 4)
        # short cr_ngroups (offset 8)
        # gid_t cr_groups[NGROUPS] (offset 10)
        data = sock.getsockopt(0, _LOCAL_PEERCRED, _XUCRED_SIZE)  # 0 = SOL_LOCAL
        _, uid = struct.unpack_from("Ii", data, 0)  # First field is cr_version (unused)
        return PeerCredentials(uid=uid, gid=None, pid=None)
    except (OSError, struct.error) as e:
        raise SocketAuthError(
            f"Failed to get peer credentials: {e}",
            expected_uid=0,
            actual_uid=0,
            context={"error": str(e), "platform": "macos"},
        ) from e


def verify_socket_peer(
    sock: socket.socket,
    expected_uid: int,
    socket_path: str | None = None,
) -> None:
    """Verify socket peer is running as expected user.

    Args:
        sock: Connected Unix domain socket
        expected_uid: Expected UID of peer process
        socket_path: Socket path for error context (optional)

    Raises:
        SocketAuthError: Peer UID doesn't match expected, or credentials
            cannot be retrieved
    """
    creds = get_peer_credentials(sock)

    if creds.uid != expected_uid:
        # Get usernames for better error message
        try:
            expected_user = pwd.getpwuid(expected_uid).pw_name
        except KeyError:
            expected_user = str(expected_uid)
        try:
            actual_user = pwd.getpwuid(creds.uid).pw_name
        except KeyError:
            actual_user = str(creds.uid)

        raise SocketAuthError(
            f"Socket peer UID mismatch: expected {expected_user} ({expected_uid}), got {actual_user} ({creds.uid})",
            expected_uid=expected_uid,
            actual_uid=creds.uid,
            context={"socket_path": socket_path, "peer_pid": creds.pid},
        )

    logger.debug(
        "Socket peer credentials verified",
        extra={"uid": creds.uid, "gid": creds.gid, "pid": creds.pid},
    )


async def connect_and_verify(
    path: str,
    expected_uid: int,
    timeout: float = 5.0,
    buffer_limit: int = 16 * 1024 * 1024,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to Unix socket and verify peer credentials.

    This is the primary entry point for authenticated socket connections.
    Authentication is MANDATORY and cannot be skipped.

    Args:
        path: Unix socket path
        expected_uid: Expected UID of peer process (required)
        timeout: Connection timeout in seconds
        buffer_limit: asyncio stream buffer limit

    Returns:
        Tuple of (reader, writer) streams

    Raises:
        SocketAuthError: Peer verification failed or credentials unavailable
        asyncio.TimeoutError: Connection timed out
        OSError: Connection failed
    """
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(path, limit=buffer_limit),
        timeout=timeout,
    )

    sock = writer.transport.get_extra_info("socket")
    if sock is None:
        writer.close()
        await writer.wait_closed()
        raise SocketAuthError(
            "Cannot verify socket peer: transport has no socket",
            expected_uid=expected_uid,
            actual_uid=0,
            context={"socket_path": path},
        )

    try:
        verify_socket_peer(sock, expected_uid, path)
    except SocketAuthError:
        writer.close()
        await writer.wait_closed()
        raise

    return reader, writer


def create_unix_socket(path: str, backlog: int = 128) -> socket.socket:
    """Create and bind a Unix domain socket (socket activation pattern).

    Creates a listening socket that can be passed to a child process via
    file descriptor inheritance. This eliminates polling latency - the
    socket is ready before the child process starts.

    Args:
        path: Unix socket path
        backlog: Listen backlog (default 128)

    Returns:
        Bound and listening socket. Caller is responsible for closing it
        after the child process has inherited the FD.

    Raises:
        OSError: Socket creation, bind, or listen failed

    Example:
        sock = create_unix_socket("/tmp/my.sock")
        fd = sock.fileno()
        proc = subprocess.Popen(..., pass_fds=(fd,))
        sock.close()  # Close parent's copy after child inherits
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
        sock.listen(backlog)
        return sock
    except OSError:
        sock.close()
        raise

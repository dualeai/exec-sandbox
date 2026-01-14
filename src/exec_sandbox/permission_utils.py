"""Cross-platform permission utilities for qemu-vm user isolation.

Centralizes chmod, chown, setfacl, sudo operations with:
- Platform guards (Linux vs macOS)
- Consistent error handling
- Async subprocess wrappers
- Verification functions

Security:
- Uses ACLs (setfacl) instead of world-writable permissions (0666)
- Uses 0660 (owner+group) instead of 0666 for sockets
- Follows libvirt group membership pattern
"""

import asyncio
import grp
import logging
import os
import pwd
from functools import lru_cache
from pathlib import Path

import aiofiles.os

from exec_sandbox.platform_utils import HostOS, ProcessWrapper, detect_host_os

# Minimum number of parts in an ACL entry (type:name:perms)
_ACL_ENTRY_MIN_PARTS = 3

logger = logging.getLogger(__name__)

# =============================================================================
# Cache for probes
# =============================================================================


class _ProbeCache:
    """Container for probe results to avoid global statements."""

    qemu_vm_user: bool | None = None
    sudo_as_qemu_vm: bool | None = None  # Can run commands as qemu-vm via sudo
    setfacl_available: bool | None = None
    getfacl_available: bool | None = None


_probe_cache = _ProbeCache()


# =============================================================================
# qemu-vm User State
# =============================================================================


@lru_cache(maxsize=1)
def get_qemu_vm_uid() -> int | None:
    """Get UID for qemu-vm user (cached).

    Returns:
        UID of qemu-vm user, or None if user doesn't exist
    """
    try:
        return pwd.getpwnam("qemu-vm").pw_uid
    except KeyError:
        return None


def get_expected_socket_uid(use_qemu_vm_user: bool) -> int:
    """Get expected UID for socket authentication.

    Used to verify QEMU process identity before sending commands.
    Falls back to current user if qemu-vm user not available.

    Args:
        use_qemu_vm_user: Whether QEMU is running as qemu-vm user

    Returns:
        Expected UID of QEMU process for socket peer verification
    """
    if use_qemu_vm_user:
        uid = get_qemu_vm_uid()
        if uid is not None:
            return uid
    return os.getuid()


async def probe_qemu_vm_user() -> bool:
    """Check if qemu-vm user exists for process isolation (cached).

    Returns False on macOS since qemu-vm user is Linux-only.

    Returns:
        True if qemu-vm user exists, False otherwise
    """
    # Fast path: return cached result
    if _probe_cache.qemu_vm_user is not None:
        return _probe_cache.qemu_vm_user

    # Skip on non-Linux - not applicable
    if detect_host_os() != HostOS.LINUX:
        _probe_cache.qemu_vm_user = False
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "/usr/bin/id",
            "qemu-vm",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        _probe_cache.qemu_vm_user = proc.returncode == 0
        if _probe_cache.qemu_vm_user:
            logger.info("qemu-vm user available (process isolation enabled)")
        else:
            logger.debug("qemu-vm user not found (process isolation disabled)")
    except (OSError, TimeoutError) as e:
        logger.debug(f"qemu-vm user probe failed: {e}")
        _probe_cache.qemu_vm_user = False

    return _probe_cache.qemu_vm_user


async def probe_sudo_as_qemu_vm() -> bool:
    """Check if we can run commands as qemu-vm user via sudo (cached).

    This is a stronger check than probe_qemu_vm_user() - it verifies not just
    that the user exists, but that we have sudo permission to run commands as
    that user.

    Required for QEMU process isolation: the QEMU process runs via
    'sudo -u qemu-vm <qemu-cmd>'.

    Returns False on macOS since qemu-vm user is Linux-only.

    Returns:
        True if sudo -u qemu-vm works, False otherwise
    """
    # Fast path: return cached result
    if _probe_cache.sudo_as_qemu_vm is not None:
        return _probe_cache.sudo_as_qemu_vm

    # Skip on non-Linux - not applicable
    if detect_host_os() != HostOS.LINUX:
        _probe_cache.sudo_as_qemu_vm = False
        return False

    # Must have qemu-vm user first
    if not await probe_qemu_vm_user():
        _probe_cache.sudo_as_qemu_vm = False
        return False

    try:
        # Use -n (non-interactive) to prevent sudo from prompting for password
        # Use /bin/true as a simple command that always succeeds
        proc = await asyncio.create_subprocess_exec(
            "sudo",
            "-n",  # Non-interactive - fail if password required
            "-u",
            "qemu-vm",
            "/bin/true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        _probe_cache.sudo_as_qemu_vm = proc.returncode == 0
        if _probe_cache.sudo_as_qemu_vm:
            logger.info("sudo -u qemu-vm available (QEMU will run as qemu-vm)")
        else:
            logger.warning(
                "sudo -u qemu-vm not available (QEMU will run as current user). "
                "Add to sudoers: 'username ALL=(qemu-vm) NOPASSWD: ALL'"
            )
    except (OSError, TimeoutError) as e:
        logger.debug(f"sudo -u qemu-vm probe failed: {e}")
        _probe_cache.sudo_as_qemu_vm = False

    return _probe_cache.sudo_as_qemu_vm


# =============================================================================
# ACL Operations (Linux only, no-op on macOS)
# =============================================================================


async def _probe_setfacl() -> bool:
    """Check if setfacl is available (cached)."""
    if _probe_cache.setfacl_available is not None:
        return _probe_cache.setfacl_available

    if detect_host_os() != HostOS.LINUX:
        _probe_cache.setfacl_available = False
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "setfacl",
            "--version",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        _probe_cache.setfacl_available = proc.returncode == 0
    except (OSError, FileNotFoundError, TimeoutError):
        _probe_cache.setfacl_available = False

    return _probe_cache.setfacl_available


async def _probe_getfacl() -> bool:
    """Check if getfacl is available (cached)."""
    if _probe_cache.getfacl_available is not None:
        return _probe_cache.getfacl_available

    if detect_host_os() != HostOS.LINUX:
        _probe_cache.getfacl_available = False
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "getfacl",
            "--version",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        _probe_cache.getfacl_available = proc.returncode == 0
    except (OSError, FileNotFoundError, TimeoutError):
        _probe_cache.getfacl_available = False

    return _probe_cache.getfacl_available


async def set_acl_user(path: Path, user: str, perms: str = "rw") -> bool:
    """Set ACL for a specific user on a file.

    Equivalent to: setfacl -m u:{user}:{perms} {path}

    Args:
        path: File or directory path
        user: Username to grant access
        perms: Permission string (e.g., "rw", "rwx", "r")

    Returns:
        True if successful, False otherwise (including on macOS)
    """
    if not await _probe_setfacl():
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "setfacl",
            "-m",
            f"u:{user}:{perms}",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.debug(f"setfacl failed: {stderr.decode().strip()}")
            return False
        logger.debug(f"Set ACL u:{user}:{perms} on {path}")
        return True
    except (OSError, FileNotFoundError) as e:
        logger.debug(f"setfacl failed: {e}")
        return False


async def remove_acl_user(path: Path, user: str) -> bool:
    """Remove ACL entry for a specific user.

    Equivalent to: setfacl -x u:{user} {path}

    Args:
        path: File or directory path
        user: Username to remove from ACL

    Returns:
        True if successful, False otherwise (including on macOS)
    """
    if not await _probe_setfacl():
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "setfacl",
            "-x",
            f"u:{user}",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    except (OSError, FileNotFoundError):
        return False


async def get_acl(path: Path) -> dict[str, str] | None:
    """Get ACL entries for a file.

    Parses getfacl output into a dictionary.

    Args:
        path: File or directory path

    Returns:
        Dictionary mapping ACL entries to permissions, e.g.:
        {'user::': 'rw-', 'user:qemu-vm': 'rw-', 'group::': 'r--', ...}
        Returns None on error or macOS.
    """
    if not await _probe_getfacl():
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            "getfacl",
            "-p",  # Don't strip leading directory
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None

        # Parse getfacl output
        acl: dict[str, str] = {}
        for raw_line in stdout.decode().splitlines():
            stripped = raw_line.strip()
            # Skip comments and empty lines
            if not stripped or stripped.startswith("#"):
                continue
            # Parse entries like "user:qemu-vm:rw-" or "user::rw-"
            if ":" in stripped:
                parts = stripped.split(":")
                if len(parts) >= _ACL_ENTRY_MIN_PARTS:
                    entry_type = parts[0]  # user, group, mask, other
                    entry_name = parts[1]  # username/groupname or empty
                    entry_perms = parts[2]  # rwx permissions
                    key = f"{entry_type}:{entry_name}" if entry_name else f"{entry_type}::"
                    acl[key] = entry_perms
        return acl
    except (OSError, FileNotFoundError):
        return None


# =============================================================================
# Mode Operations (chmod)
# =============================================================================


async def chmod_async(path: Path, mode: str) -> bool:
    """Change file mode bits asynchronously.

    Args:
        path: File or directory path
        mode: Mode string - octal ('755') or symbolic ('a+x', 'u+rw')

    Returns:
        True if successful, False otherwise
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "chmod",
            mode,
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.debug(f"chmod {mode} {path} failed: {stderr.decode().strip()}")
            return False
        return True
    except (OSError, FileNotFoundError) as e:
        logger.debug(f"chmod failed: {e}")
        return False


async def chmod_executable(path: Path) -> None:
    """Set file as executable (0o755).

    Uses Path.chmod() in a thread to avoid blocking.

    Args:
        path: File path to make executable
    """
    await asyncio.to_thread(path.chmod, 0o755)


async def get_mode(path: Path) -> int | None:
    """Get file mode bits.

    Args:
        path: File or directory path

    Returns:
        Mode bits as integer, or None if file doesn't exist
    """
    try:
        stat_result = await aiofiles.os.stat(path)
        return stat_result.st_mode
    except (OSError, FileNotFoundError):
        return None


# =============================================================================
# Ownership Operations (chown)
# =============================================================================


async def chown_async(path: Path, user: str, group: str) -> bool:
    """Change file ownership asynchronously.

    Requires appropriate privileges (usually sudo).

    Args:
        path: File or directory path
        user: New owner username
        group: New owner group

    Returns:
        True if successful, False otherwise
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo",
            "chown",
            f"{user}:{group}",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.debug(f"chown {user}:{group} {path} failed: {stderr.decode().strip()}")
            return False
        return True
    except (OSError, FileNotFoundError) as e:
        logger.debug(f"chown failed: {e}")
        return False


async def chown_to_qemu_vm(path: Path) -> bool:
    """Change file ownership to qemu-vm:qemu-vm.

    Shorthand for chown_async(path, 'qemu-vm', 'qemu-vm').

    Args:
        path: File or directory path

    Returns:
        True if successful, False otherwise
    """
    return await chown_async(path, "qemu-vm", "qemu-vm")


async def get_owner(path: Path) -> tuple[str, str] | None:
    """Get file owner (user, group).

    Args:
        path: File or directory path

    Returns:
        Tuple of (username, groupname), or None if file doesn't exist
    """
    try:
        stat_result = await aiofiles.os.stat(path)
        uid = stat_result.st_uid
        gid = stat_result.st_gid

        # Convert UID/GID to names
        try:
            username = pwd.getpwuid(uid).pw_name
        except KeyError:
            username = str(uid)

        try:
            groupname = grp.getgrgid(gid).gr_name
        except KeyError:
            groupname = str(gid)

        return (username, groupname)
    except (OSError, FileNotFoundError):
        return None


# =============================================================================
# Privileged Execution (sudo)
# =============================================================================


async def sudo_exec(
    args: list[str],
    start_new_session: bool = True,
    stdout: int | None = asyncio.subprocess.PIPE,
    stderr: int | None = asyncio.subprocess.PIPE,
) -> ProcessWrapper:
    """Run command with sudo.

    Args:
        args: Command and arguments (without 'sudo' prefix)
        start_new_session: Whether to start a new session (default True)
        stdout: stdout handling (default PIPE)
        stderr: stderr handling (default PIPE)

    Returns:
        ProcessWrapper for the sudo process
    """
    return ProcessWrapper(
        await asyncio.create_subprocess_exec(
            "sudo",
            *args,
            start_new_session=start_new_session,
            stdout=stdout,
            stderr=stderr,
        )
    )


async def sudo_rm(path: Path) -> bool:
    """Remove file or directory with sudo.

    Args:
        path: Path to remove

    Returns:
        True if successful, False otherwise
    """
    try:
        proc = await sudo_exec(["rm", "-rf", str(path)])
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.debug(f"sudo rm -rf {path} failed: {stderr.decode().strip()}")
            return False
        return True
    except (OSError, FileNotFoundError) as e:
        logger.debug(f"sudo rm failed: {e}")
        return False


# =============================================================================
# Verification
# =============================================================================


def can_access(path: Path, mode: int = os.R_OK | os.W_OK) -> bool:
    """Check if current user can access path with specified mode.

    Wrapper around os.access().

    Args:
        path: File or directory path
        mode: Access mode (os.R_OK, os.W_OK, os.X_OK, or combination)

    Returns:
        True if access is allowed, False otherwise
    """
    return os.access(path, mode)


async def verify_user_access(path: Path, user: str, perms: str = "rw") -> bool:
    """Verify that a user has specified permissions on a file.

    Checks ACL entries to verify access.

    Args:
        path: File or directory path
        user: Username to check
        perms: Required permissions (e.g., "rw", "r", "rwx")

    Returns:
        True if user has at least the specified permissions, False otherwise
    """
    acl = await get_acl(path)
    if acl is None:
        return False

    # Check user-specific ACL entry
    user_key = f"user:{user}"
    if user_key in acl:
        user_perms = acl[user_key]
        # Check each required permission
        return all(p in user_perms for p in perms)

    return False


# =============================================================================
# High-level Helpers
# =============================================================================


async def grant_qemu_vm_access(path: Path) -> bool:
    """Grant qemu-vm user read/write access to a file via ACL.

    This is more secure than chmod 666 as it only grants access to
    the specific user that needs it.

    No-op if qemu-vm user doesn't exist or on macOS.

    Args:
        path: File path (typically a socket)

    Returns:
        True if ACL was set, False if skipped or failed
    """
    if not await probe_qemu_vm_user():
        logger.debug("qemu-vm user not available, skipping ACL")
        return False

    return await set_acl_user(path, "qemu-vm", "rw")


async def ensure_traversable(dirs: list[Path]) -> bool:
    """Ensure directories are traversable (a+x) for qemu-vm user.

    Args:
        dirs: List of directory paths

    Returns:
        True if all directories were made traversable, False if any failed
    """
    success = True
    for dir_path in dirs:
        if not await chmod_async(dir_path, "a+x"):
            success = False
    return success

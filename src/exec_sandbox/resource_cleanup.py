"""Resource cleanup utilities for VM lifecycle management.

Defensive cleanup operations that log errors but don't fail.
Used by VmManager for VM lifecycle management.
"""

import asyncio
from pathlib import Path

import aiofiles
import aiofiles.os

from exec_sandbox._logging import get_logger
from exec_sandbox.platform_utils import ProcessWrapper

logger = get_logger(__name__)


async def cleanup_process(
    proc: ProcessWrapper | None,
    name: str,
    context_id: str,
    term_timeout: float = 3.0,
    kill_timeout: float = 2.0,
) -> bool:
    """Force cleanup of subprocess (SIGTERM → SIGKILL).

    Best practices (2024-2025 research):
    - Always await process.wait() after terminate/kill to prevent zombies
    - Use asyncio.create_task(proc.wait()) to reap in background if needed
    - SIGTERM (graceful) → wait with timeout → SIGKILL (force)
    - Handle ProcessLookupError for already-dead processes
    - Never raise exceptions in cleanup (log instead)
    - Check returncode before trying to kill
    - Thread-safe: uses asyncio primitives only
    - Uses ProcessWrapper for PID-reuse safety and async-only operations

    Args:
        proc: ProcessWrapper to kill (None safe - returns immediately)
        name: Process name for logging (e.g., "gvproxy", "qemu", "unshare")
        context_id: Context for logging (e.g., vm_id, tenant_id)
        term_timeout: Seconds to wait after SIGTERM before SIGKILL
        kill_timeout: Seconds to wait after SIGKILL before giving up

    Returns:
        True if process cleaned successfully, False if issues occurred
    """
    if proc is None:
        return True

    try:
        # Check if already dead (prevents unnecessary operations)
        if proc.returncode is not None:
            logger.debug(
                f"{name} already terminated",
                extra={"context_id": context_id, "returncode": proc.returncode},
            )
            # Ensure zombie is reaped
            _ = asyncio.create_task(proc.wait())  # noqa: RUF006
            return True

        # Phase 1: SIGTERM (graceful shutdown) - async, non-blocking
        logger.debug(f"Sending SIGTERM to {name}", extra={"context_id": context_id})
        await proc.terminate()

        try:
            await proc.wait_with_timeout(timeout=term_timeout)
            logger.debug(
                f"{name} stopped gracefully (SIGTERM)",
                extra={"context_id": context_id, "returncode": proc.returncode},
            )
            return True
        except TimeoutError:
            logger.warning(
                f"{name} didn't respond to SIGTERM, force killing",
                extra={"context_id": context_id, "term_timeout": term_timeout},
            )

        # Phase 2: SIGKILL (force kill) - async, non-blocking
        logger.debug(f"Sending SIGKILL to {name}", extra={"context_id": context_id})
        await proc.kill()

        try:
            await proc.wait_with_timeout(timeout=kill_timeout)
            logger.warning(
                f"{name} force killed (SIGKILL)",
                extra={"context_id": context_id, "returncode": proc.returncode},
            )
            return True
        except TimeoutError:
            logger.error(
                f"{name} didn't respond to SIGKILL within timeout",
                extra={"context_id": context_id, "kill_timeout": kill_timeout, "pid": proc.pid},
            )

            # Still try to reap in background to prevent zombie
            async def drain_and_wait() -> None:
                try:
                    await proc.wait_with_timeout(timeout=999999)  # Infinite-like timeout
                except Exception:
                    pass

            _ = asyncio.create_task(drain_and_wait())  # noqa: RUF006
            return False

    except ProcessLookupError:
        # Process already dead (race condition between check and kill)
        logger.debug(f"{name} already dead (ProcessLookupError)", extra={"context_id": context_id})
        return True

    except Exception as e:
        # Never raise - log and return failure
        logger.error(
            f"{name} cleanup error",
            extra={"context_id": context_id, "error": str(e), "error_type": type(e).__name__},
            exc_info=True,
        )
        return False


async def cleanup_cgroup(
    cgroup_path: Path | None,
    context_id: str,
) -> bool:
    """Remove cgroup directory.

    Silently succeeds if cgroup doesn't exist.

    Best practices:
    - Empty directory before removal (for cgroups: write "" to cgroup.procs)
    - Handle directory already removed
    - Handle permission errors
    - Never raise exceptions (log instead)

    Args:
        cgroup_path: Path to cgroup to remove (None safe - returns immediately)
        context_id: Context for logging

    Returns:
        True if cgroup cleaned successfully, False if issues occurred
    """
    if cgroup_path is None:
        return True

    try:
        # For cgroups: empty processes first
        if str(cgroup_path).startswith("/sys/fs/cgroup"):
            try:
                procs_file = cgroup_path / "cgroup.procs"
                if await aiofiles.os.path.exists(procs_file):
                    async with aiofiles.open(procs_file, "w") as f:
                        await f.write("")
            except Exception as e:
                logger.warning(
                    "Failed to empty cgroup",
                    extra={"context_id": context_id, "error": str(e)},
                )

        # Remove directory
        await aiofiles.os.rmdir(cgroup_path)
        logger.debug(
            "cgroup removed",
            extra={"context_id": context_id, "path": str(cgroup_path)},
        )
        return True

    except FileNotFoundError:
        # Already deleted (race condition) - success
        return True

    except OSError as e:
        # Directory not empty, permission denied, etc.
        logger.error(
            "cgroup removal error",
            extra={"context_id": context_id, "path": str(cgroup_path), "error": str(e), "error_type": type(e).__name__},
        )
        return False

    except Exception as e:
        logger.error(
            "cgroup cleanup error",
            extra={
                "context_id": context_id,
                "path": str(cgroup_path),
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        return False


async def cleanup_file(
    file_path: Path | None,
    context_id: str,
    description: str = "file",
) -> bool:
    """Delete file.

    Silently succeeds if file doesn't exist.

    Best practices (2024-2025 research):
    - Use aiofiles.os.remove() for async file deletion
    - Handle FileNotFoundError manually (aiofiles lacks missing_ok)
    - Try-except is thread-safe (unlike if-exists check)
    - Handle FileNotFoundError, PermissionError, OSError
    - Never raise exceptions in cleanup (log instead)

    Args:
        file_path: Path to file to delete (None safe - returns immediately)
        context_id: Context for logging (e.g., vm_id)
        description: Description for logging (e.g., "overlay image", "socket")

    Returns:
        True if file cleaned successfully, False if issues occurred
    """
    if file_path is None:
        return True

    try:
        await aiofiles.os.remove(file_path)
        logger.debug(
            f"{description} deleted",
            extra={"context_id": context_id, "path": str(file_path)},
        )
        return True

    except FileNotFoundError:
        # Already deleted (race condition) - success (equivalent to missing_ok=True)
        logger.debug(
            f"{description} already deleted",
            extra={"context_id": context_id, "path": str(file_path)},
        )
        return True

    except PermissionError as e:
        logger.error(
            f"{description} permission denied",
            extra={"context_id": context_id, "path": str(file_path), "error": str(e)},
        )
        return False

    except OSError as e:
        # File in use, read-only filesystem, etc.
        logger.error(
            f"{description} OS error during deletion",
            extra={"context_id": context_id, "path": str(file_path), "error": str(e), "error_type": type(e).__name__},
        )
        return False

    except Exception as e:
        # Never raise - log and return failure
        logger.error(
            f"{description} cleanup error",
            extra={
                "context_id": context_id,
                "path": str(file_path),
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        return False


async def cleanup_overlay(
    overlay_path: Path | None,
    context_id: str,
) -> bool:
    """Delete overlay qcow2 file.

    Silently succeeds if overlay doesn't exist.

    Convenience wrapper around cleanup_file for overlay images.

    Args:
        overlay_path: Path to overlay image to delete (None safe - returns immediately)
        context_id: Context for logging (e.g., vm_id)

    Returns:
        True if overlay cleaned successfully, False if issues occurred
    """
    return await cleanup_file(
        file_path=overlay_path,
        context_id=context_id,
        description="overlay image",
    )

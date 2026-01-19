"""gvproxy-wrapper lifecycle management for DNS filtering.

Provides the start_gvproxy function which handles starting and waiting for
the gvproxy-wrapper process that provides network connectivity with DNS filtering.
"""

import asyncio
import logging

from exec_sandbox.dns_filter import generate_dns_zones_json
from exec_sandbox.exceptions import VmDependencyError, VmGvproxyError
from exec_sandbox.permission_utils import grant_qemu_vm_access
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.socket_auth import create_unix_socket
from exec_sandbox.subprocess_utils import drain_subprocess_output, log_task_exception
from exec_sandbox.vm_working_directory import VmWorkingDirectory

logger = logging.getLogger(__name__)


async def start_gvproxy(
    vm_id: str,
    allowed_domains: list[str] | None,
    language: str,
    workdir: VmWorkingDirectory,
) -> tuple[ProcessWrapper, asyncio.Task[None]]:
    r"""Start gvproxy-wrapper with DNS filtering for this VM.

    Architecture Decision: gvisor-tap-vsock over alternatives
    ========================================================

    Chosen: gvisor-tap-vsock
    - Built-in DNS filtering via zones (regex-based)
    - Production-ready (Podman default since 2022)
    - 10MB memory overhead per VM
    - Simple JSON zone configuration
    - Zero CVEs (vs SLIRP: CVE-2021-3592/3/4/5, CVE-2020-29129/30)

    Socket Pre-binding (systemd activation pattern)
    ==============================================
    We create and bind the Unix socket in Python BEFORE spawning gvproxy,
    then pass the file descriptor to the child process. This eliminates
    the 100-300ms polling latency that was required when gvproxy created
    the socket itself.

    Args:
        vm_id: Unique VM identifier
        allowed_domains: Whitelist of allowed domains
        language: Programming language (for default registries)
        workdir: VM working directory containing socket paths

    Returns:
        Tuple of (gvproxy_process, gvproxy_log_task)

    Raises:
        VmGvproxyError: Failed to create socket or start gvproxy-wrapper
        VmDependencyError: gvproxy-wrapper binary not found
    """
    socket_path = workdir.gvproxy_socket

    # Generate DNS zones JSON configuration
    dns_zones_json = generate_dns_zones_json(allowed_domains, language)

    logger.info(
        "Starting gvproxy-wrapper with DNS filtering",
        extra={
            "vm_id": vm_id,
            "allowed_domains": allowed_domains,
            "language": language,
            "dns_zones_json": dns_zones_json,
        },
    )

    # Pre-create and bind socket in parent process (systemd socket activation pattern)
    # This eliminates polling latency - socket is ready before gvproxy starts
    try:
        parent_sock = create_unix_socket(str(socket_path))
        socket_fd = parent_sock.fileno()
    except OSError as e:
        raise VmGvproxyError(
            f"Failed to create gvproxy socket: {e}",
            context={
                "vm_id": vm_id,
                "language": language,
                "socket_path": str(socket_path),
            },
        ) from e

    # Start gvproxy-wrapper with pre-bound FD
    from exec_sandbox.assets import get_gvproxy_path  # noqa: PLC0415

    gvproxy_binary = await get_gvproxy_path()
    if gvproxy_binary is None:
        parent_sock.close()
        raise VmDependencyError(
            "gvproxy-wrapper binary not found. "
            "Either enable auto_download_assets=True in SchedulerConfig, "
            "or run 'make build' to build it locally."
        )
    try:
        proc = ProcessWrapper(
            await asyncio.create_subprocess_exec(
                str(gvproxy_binary),
                "-listen-fd",
                str(socket_fd),
                "-dns-zones",
                dns_zones_json,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # Create new process group for proper cleanup
                pass_fds=(socket_fd,),  # Pass pre-bound socket FD to child
            )
        )
    except (OSError, FileNotFoundError) as e:
        parent_sock.close()
        raise VmGvproxyError(
            f"Failed to start gvproxy-wrapper: {e}",
            context={
                "vm_id": vm_id,
                "language": language,
                "allowed_domains": allowed_domains,
                "binary_path": str(gvproxy_binary),
            },
        ) from e

    # Wait for gvproxy to be ready (virtualnetwork.New() must complete before QEMU connects)
    # gvproxy prints "Listening on QEMU socket" after initialization is complete
    #
    # IMPORTANT: Keep parent_sock open until gvproxy signals readiness.
    # On macOS HVF, closing the parent's socket reference before gvproxy calls
    # net.FileListener() can cause the FD to be in an inconsistent state,
    # leading to "cannot read size from socket: EOF" errors when QEMU connects.
    #
    # Design note: We use stdout event detection instead of polling or kqueue/inotify.
    # - Polling (asyncio.sleep loop): Would add 5-20ms latency between socket creation and detection
    # - kqueue (macOS) / inotify (Linux): Native but adds ~50 lines of platform-specific code
    # - Event-based (current): Instant notification via stdout, simple, cross-platform
    ready_event = asyncio.Event()

    def check_ready(line: str) -> None:
        logger.debug("[gvproxy-wrapper]", extra={"vm_id": vm_id, "output": line})
        if "Listening on QEMU socket" in line:
            ready_event.set()

    # Background task to drain gvproxy output (prevent pipe deadlock)
    gvproxy_log_task = asyncio.create_task(
        drain_subprocess_output(
            proc,
            process_name="gvproxy-wrapper",
            context_id=vm_id,
            stdout_handler=check_ready,
            stderr_handler=lambda line: logger.error(
                f"[gvproxy-wrapper error] {line}", extra={"vm_id": vm_id, "output": line}
            ),
        )
    )
    gvproxy_log_task.add_done_callback(log_task_exception)

    # Wait for gvproxy to signal readiness (timeout after 5 seconds)
    try:
        await asyncio.wait_for(ready_event.wait(), timeout=5.0)
    except TimeoutError:
        parent_sock.close()
        await proc.terminate()
        await proc.wait()
        raise VmGvproxyError(
            "gvproxy-wrapper did not become ready in time",
            context={
                "vm_id": vm_id,
                "language": language,
                "socket_path": str(socket_path),
            },
        ) from None

    # Close parent's copy of FD now that gvproxy is fully initialized
    # (child has its own via pass_fds, socket stays alive)
    parent_sock.close()

    # Grant qemu-vm user access to socket via ACL (more secure than chmod 666)
    # Only needed on Linux when qemu-vm user exists; skipped on macOS
    await grant_qemu_vm_access(socket_path)

    logger.info(
        "gvproxy-wrapper started successfully",
        extra={
            "vm_id": vm_id,
            "socket": str(socket_path),
            "dns_filtering": True,
        },
    )

    return proc, gvproxy_log_task

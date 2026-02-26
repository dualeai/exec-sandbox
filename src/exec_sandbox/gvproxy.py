"""gvproxy-wrapper lifecycle management for outbound filtering.

Provides the start_gvproxy function which handles starting and waiting for
the gvproxy-wrapper process that provides network connectivity with outbound filtering.
"""

import asyncio
import contextlib
import json

from exec_sandbox._logging import get_logger
from exec_sandbox.dns_filter import generate_outbound_allow_json
from exec_sandbox.exceptions import VmDependencyError, VmGvproxyError
from exec_sandbox.models import ExposedPort
from exec_sandbox.permission_utils import grant_qemu_vm_access
from exec_sandbox.platform_utils import ProcessWrapper
from exec_sandbox.socket_auth import create_unix_socket
from exec_sandbox.subprocess_utils import drain_subprocess_output, log_task_exception
from exec_sandbox.vm_working_directory import VmWorkingDirectory

logger = get_logger(__name__)


async def start_gvproxy(  # noqa: PLR0915
    vm_id: str,
    allowed_domains: list[str] | None,
    language: str,
    workdir: VmWorkingDirectory,
    expose_ports: list[ExposedPort] | None = None,
    block_outbound: bool = False,
) -> tuple[ProcessWrapper, asyncio.Task[None]]:
    r"""Start gvproxy-wrapper with outbound filtering for this VM.

    Architecture Decision: gvisor-tap-vsock over alternatives
    ========================================================

    Chosen: gvisor-tap-vsock
    - Built-in outbound filtering via OutboundAllow (DNS + TLS, regex-based)
    - Production-ready (Podman default since 2022)
    - 10MB memory overhead per VM
    - Simple JSON pattern configuration
    - Zero CVEs (vs SLIRP: CVE-2021-3592/3/4/5, CVE-2020-29129/30)

    Socket Pre-binding (systemd activation pattern)
    ==============================================
    We create and bind the Unix socket in Python BEFORE spawning gvproxy,
    then pass the file descriptor to the child process. This eliminates
    the 100-300ms polling latency that was required when gvproxy created
    the socket itself.

    Port Forwarding (Mode 1 & 2)
    ============================
    When expose_ports is provided, gvproxy-wrapper is started with
    port forwards configured via the -port-forward flag. The port
    forwards are set up at gvproxy startup time via the Forwards
    configuration field in gvisor-tap-vsock.

    Outbound Blocking (Mode 1)
    ==========================
    When block_outbound is True, all guest-initiated outbound TCP/UDP
    connections are blocked at the gvproxy level. This provides true
    network isolation while still allowing host-to-guest port forwarding.

    Outbound Filtering (Mode 2)
    ===========================
    When allowed_domains is non-empty, OutboundAllow patterns are set.
    Both DNS resolution and TLS connections are filtered — only domains
    matching at least one pattern can be resolved and connected to.

    Args:
        vm_id: Unique VM identifier
        allowed_domains: Whitelist of allowed domains
        language: Programming language (for default registries)
        workdir: VM working directory containing socket paths
        expose_ports: List of ports to expose (Mode 1/2 port forwarding)
        block_outbound: Block all guest-initiated outbound connections (Mode 1)

    Returns:
        Tuple of (gvproxy_process, gvproxy_log_task)

    Raises:
        VmGvproxyError: Failed to create socket or start gvproxy-wrapper
        VmDependencyError: gvproxy-wrapper binary not found
    """
    socket_path = workdir.gvproxy_socket

    # Generate OutboundAllow patterns JSON
    outbound_allow_json = generate_outbound_allow_json(allowed_domains, language)

    logger.info(
        "Starting gvproxy-wrapper with outbound filtering",
        extra={
            "vm_id": vm_id,
            "allowed_domains": allowed_domains,
            "language": language,
            "outbound_allow_json": outbound_allow_json,
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

    # Build gvproxy-wrapper command arguments
    gvproxy_args = [
        str(gvproxy_binary),
        "-listen-fd",
        str(socket_fd),
    ]

    # Add OutboundAllow patterns for outbound filtering (Mode 2)
    # Only pass when there are patterns (non-empty JSON array)
    if outbound_allow_json != "[]":
        gvproxy_args.extend(["-outbound-allow", outbound_allow_json])

    # Add outbound blocking for Mode 1 (port-forward only, no internet)
    # This blocks all guest-initiated TCP/UDP connections at the gvproxy level
    if block_outbound:
        gvproxy_args.append("-block-outbound")
        logger.info(
            "Enabling outbound blocking (Mode 1: port-forward only)",
            extra={"vm_id": vm_id},
        )

    # Add port forwarding configuration if needed (Mode 1/2)
    # Port forwards are configured at gvproxy startup via the Forwards config field
    # Format: {"local_addr:port": "guest_ip:port", ...}
    # Guest IP in gvproxy is always 192.168.127.2
    if expose_ports:
        forwards_dict = {f"{p.host}:{p.external}": f"192.168.127.2:{p.internal}" for p in expose_ports}
        port_forward_json = json.dumps(forwards_dict)
        gvproxy_args.extend(["-port-forward", port_forward_json])
        logger.info(
            "Configuring gvproxy port forwarding (Mode 2)",
            extra={
                "vm_id": vm_id,
                "expose_ports": [(p.internal, p.external) for p in expose_ports],
                "forwards": forwards_dict,
            },
        )

    try:
        proc = ProcessWrapper(
            await asyncio.create_subprocess_exec(
                *gvproxy_args,
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
    except BaseException:
        # CancelledError or other unexpected exception — clean up socket and process
        parent_sock.close()
        with contextlib.suppress(ProcessLookupError):
            await proc.terminate()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
        raise
    else:
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
            "outbound_filtering": outbound_allow_json != "[]",
            "port_forwarding": expose_ports is not None and len(expose_ports) > 0,
            "block_outbound": block_outbound,
        },
    )

    return proc, gvproxy_log_task

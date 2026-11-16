"""gvproxy-wrapper lifecycle management for outbound filtering.

Provides the start_gvproxy function which handles starting and waiting for
the gvproxy-wrapper process that provides network connectivity with outbound filtering.
"""

import asyncio
import json
import logging
import re
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Final

from exec_sandbox import constants
from exec_sandbox.assets import get_gvproxy_path
from exec_sandbox.dns_filter import generate_outbound_allow_json
from exec_sandbox.exceptions import ConfigError, ErrorCode, TransientError, spawn_error
from exec_sandbox.models import ExposedPort
from exec_sandbox.permission_utils import probe_qemu_as_qemu_vm, set_acl_user
from exec_sandbox.platform_utils import ProcessWrapper, get_arch_name
from exec_sandbox.socket_auth import create_unix_socket
from exec_sandbox.subprocess_utils import exec_failed_error, start_drain, start_managed_process
from exec_sandbox.vm_working_directory import VmWorkingDirectory

logger = logging.getLogger(__name__)


_QEMU_SOCKET_CLOSED: Final = "cannot read size from socket: EOF"
"""What gvisor-tap-vsock logs (logrus, level=error, on stderr) when QEMU's end of the network
socket closes: the normal end of a VM once QEMU has connected."""


def _stderr_level(line: str, *, qemu_connected: bool) -> int:
    """The log level of a gvproxy stderr line: ERROR, but DEBUG for the close of QEMU's socket
    after QEMU connected (each VM end). Before QEMU connected, that close is a fault (a
    listener handed over in a bad state) and stays ERROR."""
    if qemu_connected and _QEMU_SOCKET_CLOSED in line:
        return logging.DEBUG
    return logging.ERROR


_OUTPUT_TAIL_LINES: Final = 5
"""Last output lines of gvproxy-wrapper kept for the error of an end before it is ready."""

_OUTPUT_LINE_CHARS: Final = 300
"""The part of the quoted output line that the message of that error quotes."""

_GO_CRASH_STARTS: Final = ("panic:", "fatal error:", "runtime:", "runtime/cgo:")
"""How the Go runtime starts the report of a crash on stderr: its cause comes first, then
goroutine stacks and registers (the last lines say nothing then)."""

_GO_SIGNAL_LINE: Final = re.compile(r"SIG[A-Z0-9]+: ")
"""How the Go runtime names a signal it dies of (SIGABRT: abort, SIGSEGV: segmentation
violation) when no panic or fatal error came first: its traceback follows, whose `runtime:`
lines (unknown pc, unexpected return pc) are not the cause."""


async def _wait_ready(
    proc: ProcessWrapper,
    ready: asyncio.Event,
    *,
    drain: asyncio.Task[None],
    output: deque[str],
    crash: list[str],
    binary: str,
    context: dict[str, Any],
) -> None:
    """Wait until gvproxy-wrapper is ready, at most 5 s. When it ends first (an exec in its
    launch failed, or a crash), raise at once with its last output lines: Go's log, with
    log.Fatal, writes to stdout (gvproxy-wrapper/main.go), which the log has at DEBUG only;
    a Go runtime crash puts its cause first (`crash`: the first such stderr line), and the
    message quotes that line when there is one.
    The pipes are at their end once the process wait returns, so `drain` (which fills
    `output`) ends at once.

    Raises:
        ConfigError: HOST_DEPENDENCY: its launch ended with 126 or 127 (exec_failed_error)
        TransientError: GVPROXY_FAILED: it ended with another status
        TimeoutError: neither within 5 s
    """
    ready_task = asyncio.create_task(ready.wait())
    end_task = asyncio.create_task(proc.wait())
    try:
        done, _ = await asyncio.wait((ready_task, end_task), timeout=5.0, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (ready_task, end_task):
            task.cancel()
        await asyncio.gather(ready_task, end_task, return_exceptions=True)
    if ready_task in done:
        return
    if end_task not in done:
        raise TimeoutError
    await asyncio.wait((drain,))  # its last lines (a failure of the drain is logged there)
    exit_code = end_task.result()
    last = (crash[0] if crash else output[-1] if output else "")[:_OUTPUT_LINE_CHARS] or None
    context = {**context, "exit_code": exit_code, "output": list(output), "crash": crash[0] if crash else None}
    raise exec_failed_error(exit_code, binary=binary, detail=last, context=context, hint=None) or TransientError(
        f"gvproxy-wrapper exited with code {exit_code} before it was ready" + (f": {last}" if last else ""),
        code=ErrorCode.GVPROXY_FAILED,
        context=context,
        hint=None,
        instance_id=None,
    )


async def start_gvproxy(  # noqa: PLR0915
    vm_id: str,
    *,
    allowed_domains: Sequence[str],
    workdir: VmWorkingDirectory,
    images_dir: Path,
    expose_ports: list[ExposedPort] | None,
    block_outbound: bool,
    on_process_started: Callable[[ProcessWrapper, asyncio.Task[None] | None], None],
) -> tuple[ProcessWrapper, asyncio.Task[None], Callable[[], Awaitable[None]]]:
    r"""Start gvproxy-wrapper with outbound filtering for this VM.

    Architecture Decision: gvisor-tap-vsock over alternatives
    ========================================================

    Chosen: gvisor-tap-vsock
    - Built-in outbound filtering via OutboundAllow (DNS + TLS, regex-based)
    - Production-ready (Podman default since 2022)
    - ~15-30 MB of memory per VM (measured: constants.DEFAULT_VM_MEMORY_OVERHEAD_MB)
    - Simple JSON pattern configuration
    - Zero CVEs (vs SLIRP: CVE-2021-3592/3/4/5, CVE-2020-29129/30)

    Socket Pre-binding (systemd activation pattern)
    ==============================================
    We create and bind the Unix socket in Python BEFORE spawning gvproxy,
    then pass the file descriptor to the child process: the socket exists
    before gvproxy starts, so no poll waits for it.

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

    Outbound Filtering (Modes 2 and 3)
    ==================================
    OutboundAllow patterns are set for a non-empty allowed_domains (the caller resolves
    a spec's None to the language's package registries: dns_filter.resolved_domains).
    Both DNS resolution and TLS connections are filtered — only domains matching at
    least one pattern can be resolved and connected to.

    Args:
        vm_id: Unique VM identifier
        allowed_domains: Domains the guest may reach, and their subdomains
            (dns_filter.resolved_domains); empty: no filter
        workdir: VM working directory containing socket paths
        images_dir: The images folder (Settings.base_images_dir): gvproxy-wrapper is searched
            there first, as the scheduler start does when it makes sure the binary exists
            (assets.ensure_assets)
        expose_ports: List of ports to expose (Mode 1/2 port forwarding)
        block_outbound: Block all guest-initiated outbound connections (Mode 1)
        on_process_started: Called with the process once it is spawned (no drain task
            yet: None), then again with its drain task; the caller then owns both on every
            path, and start_gvproxy never stops them

    Returns:
        Tuple of (gvproxy_process, gvproxy_log_task, wait_ready). It returns once the process
        runs, before it is ready: QEMU can start meanwhile, as its connect waits in the
        socket's listen queue (about 5 ms saved per network VM). The caller awaits
        `wait_ready()` before the VM is ready; on its error, the caller stops the process.
        wait_ready raises TransientError GVPROXY_FAILED (it ended, or was not ready within
        5 s) or ConfigError HOST_DEPENDENCY (its launch ended with 126 or 127).

    Raises:
        TransientError: GVPROXY_FAILED: failed to create socket, or the fork of
            gvproxy-wrapper hit a process, memory or file-handle limit
        ConfigError: HOST_DEPENDENCY: gvproxy-wrapper binary not found or cannot run, or
            qemu-vm (when QEMU runs as qemu-vm) cannot get access to the socket (setfacl)
    """
    socket_path = workdir.gvproxy_socket

    # Generate OutboundAllow patterns JSON
    outbound_allow_json = generate_outbound_allow_json(allowed_domains)

    logger.info(
        "Starting gvproxy-wrapper with outbound filtering",
        extra={
            "vm_id": vm_id,
            "allowed_domains": allowed_domains,
            "outbound_allow_json": outbound_allow_json,
        },
    )

    # gvproxy-wrapper prints "Listening on QEMU socket" once virtualnetwork.New() is done
    # and it adopted the inherited listener. Event-based: the drain reads its stdout, no
    # polling and no platform file-watch code.
    ready_event = asyncio.Event()
    qemu_connected = False

    output: deque[str] = deque(maxlen=_OUTPUT_TAIL_LINES)  # for the error of an end before ready
    crash: list[str] = []  # the cause of a Go runtime crash: its first crash or SIGxxx: line

    def check_ready(line: str) -> None:
        nonlocal qemu_connected
        output.append(line)
        logger.debug("[gvproxy-wrapper]", extra={"vm_id": vm_id, "output": line})
        if "Listening on QEMU socket" in line:
            ready_event.set()
        elif "QEMU connected from" in line:
            qemu_connected = True

    def check_error(line: str) -> None:
        output.append(line)
        if not crash and (line.startswith(_GO_CRASH_STARTS) or _GO_SIGNAL_LINE.match(line)):
            crash.append(line)  # the cause comes first in each Go crash report
        level = _stderr_level(line, qemu_connected=qemu_connected)
        logger.log(level, f"[gvproxy-wrapper error] {line}", extra={"vm_id": vm_id, "output": line})

    # Pre-create and bind socket in parent process (systemd socket activation pattern)
    # This eliminates polling latency - socket is ready before gvproxy starts
    try:
        parent_sock = create_unix_socket(str(socket_path))
    except OSError as e:
        raise TransientError(
            f"Failed to create gvproxy socket: {e}",
            code=ErrorCode.GVPROXY_FAILED,
            context={"vm_id": vm_id, "socket_path": str(socket_path)},
            hint=None,
            instance_id=None,
        ) from e

    try:
        # Start gvproxy-wrapper with pre-bound FD
        gvproxy_binary = await get_gvproxy_path(override=images_dir, arch=get_arch_name())
        if gvproxy_binary is None:
            raise ConfigError(
                f"gvproxy-wrapper binary not found in {images_dir}, gvproxy-wrapper/bin, "
                "EXEC_SANDBOX_IMAGES_DIR, images/dist or the download cache. Run `sbx prefetch` with "
                "network access, or turn on auto_download_assets with network access, or build it "
                "(make build).",
                code=ErrorCode.HOST_DEPENDENCY,
                context=None,
                hint=None,
                instance_id=None,
            )

        # Build gvproxy-wrapper command arguments
        socket_fd = parent_sock.fileno()
        gvproxy_args = [
            str(gvproxy_binary),
            "-listen-fd",
            str(socket_fd),
            # The network constants (one source: constants.py): required flags of the wrapper
            "-subnet",
            constants.GUEST_NETWORK_SUBNET,
            "-gateway-ip",
            constants.GUEST_GATEWAY_IP,
            "-gateway-mac",
            constants.GUEST_GATEWAY_MAC,
            "-mtu",
            str(constants.GUEST_MTU),
        ]

        # Add OutboundAllow patterns for outbound filtering (Modes 2 and 3)
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
        # Format: {"local_addr:port": "guest_ip:port", ...}; a UDP forward's key has the "udp:"
        # prefix (gvisor-tap-vsock forwardHostVM: a key without it is TCP)
        if expose_ports:
            forwards_dict = {
                f"{'udp:' if p.protocol == 'udp' else ''}{p.host}:{p.external}": f"{constants.GUEST_IP}:{p.internal}"
                for p in expose_ports
            }
            port_forward_json = json.dumps(forwards_dict)
            gvproxy_args.extend(["-port-forward", port_forward_json])
            logger.info(
                "Configuring gvproxy port forwarding (Modes 1 and 2)",
                extra={
                    "vm_id": vm_id,
                    "expose_ports": [(p.internal, p.external) for p in expose_ports],
                    "forwards": forwards_dict,
                },
            )

        try:
            # Ends with this process (Linux): with no VM cgroup, nothing else ends it
            proc = await start_managed_process(
                gvproxy_args,
                steps=(),
                end_with_this_process=True,
                pass_fds=(socket_fd,),
                on_process_started=lambda proc: on_process_started(proc, None),
            )
        except OSError as e:
            raise spawn_error(
                e,
                binary=str(gvproxy_binary),
                code=ErrorCode.GVPROXY_FAILED,
                context={"vm_id": vm_id, "allowed_domains": allowed_domains},
                hint=None,
            ) from e

        # Background task to drain gvproxy output (prevent pipe deadlock)
        gvproxy_log_task = start_drain(
            proc,
            process_name="gvproxy-wrapper",
            context_id=vm_id,
            stdout_handler=check_ready,
            stderr_handler=check_error,
        )
        on_process_started(proc, gvproxy_log_task)

        # qemu-vm (QEMU through sudo) needs access to the socket before QEMU connects: without
        # it, the VM would fail later at its boot
        if await probe_qemu_as_qemu_vm() and not await set_acl_user([socket_path], user="qemu-vm", perms="rw"):
            raise ConfigError(
                "Could not give qemu-vm access to the gvproxy socket (needs setfacl, package acl)",
                code=ErrorCode.HOST_DEPENDENCY,
                context={"vm_id": vm_id, "socket": str(socket_path)},
                hint="Install setfacl (package acl) on this host.",
                instance_id=None,
            )
    finally:
        parent_sock.close()  # gvproxy-wrapper has its own copy since its exec (pass_fds)

    async def wait_ready() -> None:
        """Wait until gvproxy-wrapper is ready, at most 5 s, or raise how it ended."""
        try:
            await _wait_ready(
                proc,
                ready_event,
                drain=gvproxy_log_task,
                output=output,
                crash=crash,
                binary=str(gvproxy_binary),
                context={"vm_id": vm_id},
            )
        except TimeoutError:
            raise TransientError(
                "gvproxy-wrapper did not become ready in time",
                code=ErrorCode.GVPROXY_FAILED,
                context={"vm_id": vm_id, "socket_path": str(socket_path)},
                hint=None,
                instance_id=None,
            ) from None
        logger.info(
            "gvproxy-wrapper started successfully",
            extra={
                "vm_id": vm_id,
                "socket": str(socket_path),
                "outbound_filtering": outbound_allow_json != "[]",
                "port_forwarding": bool(expose_ports),
                "block_outbound": block_outbound,
            },
        )

    return proc, gvproxy_log_task, wait_ready

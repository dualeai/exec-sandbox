"""Port forwarding utilities for exposing guest VM ports to the host.

This module provides port allocation and mapping resolution functions for
exposing guest VM ports to the host.

Architecture (gvproxy -port-forward forwards the ports in both modes):
- Mode 1: expose_ports without allow_network -> gvproxy -block-outbound (no
  guest-initiated connections)
- Mode 2: expose_ports with allow_network -> outbound as allowed_domains permits
"""

import socket
from collections.abc import Sequence
from typing import Literal

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import ErrorCode, TransientError
from exec_sandbox.models import ExposedPort, PortMapping

logger = get_logger(__name__)


def allocate_ephemeral_ports(count: int, *, protocol: Literal["tcp", "udp"], exclude: frozenset[int]) -> list[int]:
    """Allocate unique ephemeral ports on PORT_FORWARD_BIND_HOST by binding and releasing.

    Holds all sockets open until every port is assigned, preventing the
    OS from recycling a recently-closed port on the next bind-to-0 call
    (observed on Linux with ``SO_REUSEADDR``).

    Args:
        count: Number of ports to allocate.
        protocol: The socket type that probes each port: a port free for TCP can be in use
            for UDP.
        exclude: Ports never returned (the fixed host ports of the same VM: nothing binds
            them before gvproxy, so the OS can pick one). A picked one stays held, so the
            next bind gets another: at most count + len(exclude) binds.

    Returns:
        List of unique available ephemeral port numbers (>= 1024).

    Raises:
        OSError: A socket could not be made or bound (out of file handles, no free port).

    Note:
        There's a small race window where another process could claim
        a port between release and gvproxy binding. This is acceptable
        for ephemeral VM ports - the user can retry if needed.
    """
    host = constants.PORT_FORWARD_BIND_HOST
    held: list[socket.socket] = []
    ports: list[int] = []
    try:
        while len(ports) < count:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM if protocol == "tcp" else socket.SOCK_DGRAM)
            held.append(s)
            if protocol == "tcp":  # for UDP, SO_REUSEADDR lets two sockets share a port
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, 0))
            port = s.getsockname()[1]
            if port not in exclude:
                ports.append(port)
    finally:
        for s in held:
            s.close()
    logger.debug("Allocated ephemeral ports", extra={"ports": ports, "host": host})
    return ports


def resolve_port_mappings(mappings: Sequence[PortMapping]) -> list[ExposedPort]:
    """Allocate a host port (on PORT_FORWARD_BIND_HOST) for each mapping with no external port.

    Args:
        mappings: VmSpec.expose_ports (VmSpec turns an int into a PortMapping and enforces
            MAX_EXPOSED_PORTS)

    Returns:
        List of ExposedPort with allocated external ports

    Raises:
        TransientError: GVPROXY_FAILED: a host port could not be allocated (out of file
            handles, no free port: allocate_ephemeral_ports); it can pass later
    """
    host = constants.PORT_FORWARD_BIND_HOST
    try:
        allocated = {
            protocol: iter(
                allocate_ephemeral_ports(
                    sum(1 for m in mappings if m.external is None and m.protocol == protocol),
                    protocol=protocol,
                    exclude=frozenset(
                        m.external for m in mappings if m.external is not None and m.protocol == protocol
                    ),
                )
            )
            for protocol in ("tcp", "udp")
        }
    except OSError as e:
        raise TransientError(
            f"Could not allocate host ports for the port forwards: {e.strerror or type(e).__name__}",
            code=ErrorCode.GVPROXY_FAILED,
            context={"host": host, "errno": e.errno},
            hint=None,
            instance_id=None,
        ) from e
    result: list[ExposedPort] = []

    for mapping in mappings:
        external = mapping.external if mapping.external is not None else next(allocated[mapping.protocol])
        result.append(
            ExposedPort(
                internal=mapping.internal,
                external=external,
                host=host,
                protocol=mapping.protocol,
            )
        )

    logger.info(
        "Resolved port mappings",
        extra={
            "count": len(result),
            "ports": [(p.internal, p.external) for p in result],
        },
    )

    return result

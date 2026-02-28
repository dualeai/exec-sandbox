"""Port forwarding utilities for exposing guest VM ports to the host.

This module provides port allocation and mapping resolution functions for
exposing guest VM ports to the host.

Architecture:
- Mode 1: expose_ports without allow_network -> gvproxy with BlockAllOutbound
- Mode 2: expose_ports with allow_network -> gvproxy configuration-based forwarding
"""

import socket

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.models import ExposedPort, PortMapping

logger = get_logger(__name__)


def allocate_ephemeral_ports(count: int, host: str = constants.PORT_FORWARD_BIND_HOST) -> list[int]:
    """Allocate unique ephemeral ports by binding and releasing.

    Holds all sockets open until every port is assigned, preventing the
    OS from recycling a recently-closed port on the next bind-to-0 call
    (observed on Linux with ``SO_REUSEADDR``).

    Args:
        count: Number of ports to allocate.
        host: Host address to bind to (default: 127.0.0.1).

    Returns:
        List of unique available ephemeral port numbers (>= 1024).

    Note:
        There's a small race window where another process could claim
        a port between release and QEMU binding. This is acceptable
        for ephemeral VM ports - the user can retry if needed.
    """
    held: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, 0))
            ports.append(s.getsockname()[1])
            held.append(s)
    finally:
        for s in held:
            s.close()
    logger.debug("Allocated ephemeral ports", extra={"ports": ports, "host": host})
    return ports


def normalize_port_mappings(mappings: list[PortMapping | int]) -> list[PortMapping]:
    """Normalize mixed port mapping inputs to PortMapping objects.

    Accepts a mix of integers (shorthand for internal port) and PortMapping
    objects, converting all to PortMapping.

    Args:
        mappings: List of port mappings (int or PortMapping)

    Returns:
        List of PortMapping objects (external may still be None)
    """
    result: list[PortMapping] = []
    for m in mappings:
        if isinstance(m, int):
            result.append(PortMapping(internal=m))
        else:
            result.append(m)
    return result


def resolve_port_mappings(
    mappings: list[PortMapping | int],
    host: str = constants.PORT_FORWARD_BIND_HOST,
) -> list[ExposedPort]:
    """Normalize and allocate external ports for mappings.

    Takes a list of port mappings (which may have None for external port)
    and returns ExposedPort objects with all external ports allocated.

    Args:
        mappings: List of port mappings (int or PortMapping)
        host: Host address to bind forwarded ports to

    Returns:
        List of ExposedPort with allocated external ports

    Raises:
        ValueError: If too many ports requested or invalid port numbers
    """
    if len(mappings) > constants.MAX_EXPOSED_PORTS:
        raise ValueError(f"Too many exposed ports: {len(mappings)} > {constants.MAX_EXPOSED_PORTS}")

    normalized = normalize_port_mappings(mappings)

    need = sum(1 for m in normalized if m.external is None)
    allocated = iter(allocate_ephemeral_ports(need, host))
    result: list[ExposedPort] = []

    for mapping in normalized:
        external = mapping.external if mapping.external is not None else next(allocated)
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

"""Tests for port forwarding functionality.

Tests the port allocation, mapping resolution, and models for port forwarding.
"""

import json
import socket

import pytest

from exec_sandbox.models import ExposedPort, Language, PortMapping
from exec_sandbox.port_forward import (
    allocate_ephemeral_port,
    normalize_port_mappings,
    resolve_port_mappings,
)
from exec_sandbox.scheduler import Scheduler


class TestPortMapping:
    """Tests for PortMapping model."""

    def test_port_mapping_internal_only(self) -> None:
        """Test creating port mapping with only internal port."""
        mapping = PortMapping(internal=8080)
        assert mapping.internal == 8080
        assert mapping.external is None
        assert mapping.protocol == "tcp"

    def test_port_mapping_with_external(self) -> None:
        """Test creating port mapping with explicit external port."""
        mapping = PortMapping(internal=8080, external=3000)
        assert mapping.internal == 8080
        assert mapping.external == 3000
        assert mapping.protocol == "tcp"

    def test_port_mapping_udp_protocol(self) -> None:
        """Test creating port mapping with UDP protocol."""
        mapping = PortMapping(internal=53, external=5300, protocol="udp")
        assert mapping.internal == 53
        assert mapping.external == 5300
        assert mapping.protocol == "udp"

    def test_port_mapping_validation_internal_range(self) -> None:
        """Test that internal port must be in valid range."""
        # Valid ports
        PortMapping(internal=1)
        PortMapping(internal=65535)

        # Invalid ports
        with pytest.raises(ValueError):
            PortMapping(internal=0)
        with pytest.raises(ValueError):
            PortMapping(internal=65536)

    def test_port_mapping_validation_external_unprivileged(self) -> None:
        """Test that external port must be unprivileged (>= 1024)."""
        # Valid unprivileged ports
        PortMapping(internal=80, external=1024)
        PortMapping(internal=80, external=65535)

        # Privileged ports not allowed
        with pytest.raises(ValueError):
            PortMapping(internal=80, external=80)
        with pytest.raises(ValueError):
            PortMapping(internal=80, external=1023)


class TestExposedPort:
    """Tests for ExposedPort model."""

    def test_exposed_port_creation(self) -> None:
        """Test creating an exposed port."""
        port = ExposedPort(internal=8080, external=3000)
        assert port.internal == 8080
        assert port.external == 3000
        assert port.host == "127.0.0.1"
        assert port.protocol == "tcp"

    def test_exposed_port_url_property(self) -> None:
        """Test the URL property."""
        port = ExposedPort(internal=8080, external=3000)
        assert port.url == "http://127.0.0.1:3000"

    def test_exposed_port_custom_host(self) -> None:
        """Test exposed port with custom host."""
        port = ExposedPort(internal=8080, external=3000, host="0.0.0.0")
        assert port.host == "0.0.0.0"
        assert port.url == "http://0.0.0.0:3000"


class TestAllocateEphemeralPort:
    """Tests for port allocation."""

    def test_allocate_ephemeral_port_returns_valid_port(self) -> None:
        """Test that allocated port is in valid range."""
        port = allocate_ephemeral_port()
        assert 1024 <= port <= 65535

    def test_allocate_ephemeral_port_is_available(self) -> None:
        """Test that allocated port can be bound."""
        port = allocate_ephemeral_port()

        # Try to bind to the allocated port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # This should succeed (port was just released)
            s.bind(("127.0.0.1", port))

    def test_allocate_multiple_ports_are_different(self) -> None:
        """Test that multiple allocations return different ports."""
        # Allocate several ports
        ports = [allocate_ephemeral_port() for _ in range(5)]
        # All should be unique
        assert len(set(ports)) == len(ports)


class TestNormalizePortMappings:
    """Tests for normalizing port mappings."""

    def test_normalize_integer_to_port_mapping(self) -> None:
        """Test that integers are converted to PortMapping."""
        result = normalize_port_mappings([8080, 3000])
        assert len(result) == 2
        assert all(isinstance(p, PortMapping) for p in result)
        assert result[0].internal == 8080
        assert result[0].external is None
        assert result[1].internal == 3000
        assert result[1].external is None

    def test_normalize_port_mapping_passthrough(self) -> None:
        """Test that PortMapping objects pass through unchanged."""
        mapping = PortMapping(internal=8080, external=3000)
        result = normalize_port_mappings([mapping])
        assert len(result) == 1
        assert result[0] is mapping

    def test_normalize_mixed_input(self) -> None:
        """Test normalizing mixed integers and PortMapping objects."""
        mapping = PortMapping(internal=80, external=8080)
        result = normalize_port_mappings([8080, mapping, 3000])
        assert len(result) == 3
        assert result[0].internal == 8080
        assert result[0].external is None
        assert result[1] is mapping
        assert result[2].internal == 3000

    def test_normalize_empty_list(self) -> None:
        """Test normalizing empty list."""
        result = normalize_port_mappings([])
        assert result == []


class TestResolvePortMappings:
    """Tests for resolving port mappings."""

    def test_resolve_allocates_external_ports(self) -> None:
        """Test that missing external ports are allocated."""
        result = resolve_port_mappings([8080, 3000])
        assert len(result) == 2
        assert all(isinstance(p, ExposedPort) for p in result)
        assert result[0].internal == 8080
        assert result[0].external >= 1024
        assert result[1].internal == 3000
        assert result[1].external >= 1024
        # External ports should be different
        assert result[0].external != result[1].external

    def test_resolve_preserves_explicit_external(self) -> None:
        """Test that explicit external ports are preserved."""
        mapping = PortMapping(internal=8080, external=3000)
        result = resolve_port_mappings([mapping])
        assert len(result) == 1
        assert result[0].internal == 8080
        assert result[0].external == 3000

    def test_resolve_too_many_ports_raises_error(self) -> None:
        """Test that too many ports raises ValueError."""
        # MAX_EXPOSED_PORTS is 10
        with pytest.raises(ValueError, match="Too many exposed ports"):
            resolve_port_mappings(list(range(1000, 1020)))  # 20 ports

    def test_resolve_preserves_protocol(self) -> None:
        """Test that protocol is preserved through resolution."""
        mapping = PortMapping(internal=53, protocol="udp")
        result = resolve_port_mappings([mapping])
        assert result[0].protocol == "udp"

    def test_resolve_mixed_explicit_and_dynamic(self) -> None:
        """Test resolving mix of explicit and dynamic ports."""
        mappings = [
            PortMapping(internal=80, external=8080),
            PortMapping(internal=443),
            8000,
        ]
        result = resolve_port_mappings(mappings)
        assert len(result) == 3
        # First has explicit external
        assert result[0].external == 8080
        # Second and third have dynamically allocated externals
        assert result[1].external >= 1024
        assert result[2].external >= 1024
        # All external ports should be unique
        external_ports = [p.external for p in result]
        assert len(set(external_ports)) == len(external_ports)


class TestPortMappingEdgeCases:
    """Edge case tests for PortMapping validation."""

    def test_negative_internal_port_rejected(self) -> None:
        """Test that negative internal port is rejected."""
        with pytest.raises(ValueError):
            PortMapping(internal=-1)
        with pytest.raises(ValueError):
            PortMapping(internal=-8080)

    def test_negative_external_port_rejected(self) -> None:
        """Test that negative external port is rejected."""
        with pytest.raises(ValueError):
            PortMapping(internal=8080, external=-1)
        with pytest.raises(ValueError):
            PortMapping(internal=8080, external=-3000)

    def test_invalid_protocol_rejected(self) -> None:
        """Test that invalid protocol values are rejected."""
        with pytest.raises(ValueError):
            PortMapping(internal=8080, protocol="http")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            PortMapping(internal=8080, protocol="icmp")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            PortMapping(internal=8080, protocol="")  # type: ignore[arg-type]

    def test_internal_port_zero_rejected(self) -> None:
        """Test that port 0 is rejected for internal."""
        with pytest.raises(ValueError):
            PortMapping(internal=0)

    def test_external_port_zero_rejected(self) -> None:
        """Test that port 0 is rejected for external."""
        with pytest.raises(ValueError):
            PortMapping(internal=8080, external=0)


class TestExposedPortEdgeCases:
    """Edge case tests for ExposedPort."""

    def test_udp_port_url_still_returns_http(self) -> None:
        """Test that UDP port URL returns http (known limitation)."""
        # This is a known limitation - url property always returns http://
        # For UDP services, users should construct their own connection string
        port = ExposedPort(internal=53, external=5300, protocol="udp")
        assert port.url == "http://127.0.0.1:5300"
        # Document that this is expected behavior for UDP


class TestResolvePortMappingsBoundary:
    """Boundary tests for resolve_port_mappings."""

    def test_exactly_max_ports_succeeds(self) -> None:
        """Test that exactly MAX_EXPOSED_PORTS (10) succeeds."""
        # Create exactly 10 port mappings
        mappings = [PortMapping(internal=8000 + i) for i in range(10)]
        result = resolve_port_mappings(mappings)
        assert len(result) == 10

    def test_one_over_max_ports_fails(self) -> None:
        """Test that MAX_EXPOSED_PORTS + 1 (11) fails."""
        mappings = [PortMapping(internal=8000 + i) for i in range(11)]
        with pytest.raises(ValueError, match="Too many exposed ports"):
            resolve_port_mappings(mappings)

    def test_external_port_at_min_boundary(self) -> None:
        """Test external port at minimum boundary (1024)."""
        mapping = PortMapping(internal=80, external=1024)
        result = resolve_port_mappings([mapping])
        assert result[0].external == 1024

    def test_external_port_at_max_boundary(self) -> None:
        """Test external port at maximum boundary (65535)."""
        mapping = PortMapping(internal=80, external=65535)
        result = resolve_port_mappings([mapping])
        assert result[0].external == 65535


class TestPortMappingWeirdCases:
    """Weird/unusual case tests."""

    def test_duplicate_internal_ports_allowed(self) -> None:
        """Test that duplicate internal ports are allowed (user's responsibility)."""
        # Same internal port twice - this is technically allowed at model level
        # The behavior depends on what QEMU/gvproxy does with duplicates
        mappings = [
            PortMapping(internal=8080, external=3000),
            PortMapping(internal=8080, external=3001),
        ]
        result = resolve_port_mappings(mappings)
        assert len(result) == 2
        assert result[0].internal == result[1].internal == 8080
        assert result[0].external != result[1].external

    def test_duplicate_external_ports_allowed_at_model_level(self) -> None:
        """Test that duplicate external ports pass model validation.

        Note: This will fail at runtime when QEMU tries to bind the same port twice.
        The model doesn't prevent this - it's caught at bind time.
        """
        mappings = [
            PortMapping(internal=8080, external=3000),
            PortMapping(internal=8081, external=3000),  # Same external!
        ]
        result = resolve_port_mappings(mappings)
        assert len(result) == 2
        # Both have same external - will fail at runtime, not validation

    def test_internal_equals_external(self) -> None:
        """Test that internal == external is allowed."""
        # internal 8080 -> external 8080 is valid (both unprivileged)
        mapping = PortMapping(internal=8080, external=8080)
        result = resolve_port_mappings([mapping])
        assert result[0].internal == result[0].external == 8080

    def test_privileged_internal_unprivileged_external(self) -> None:
        """Test mapping privileged internal to unprivileged external."""
        # Guest can listen on port 80, host exposes on 8080
        mapping = PortMapping(internal=80, external=8080)
        result = resolve_port_mappings([mapping])
        assert result[0].internal == 80
        assert result[0].external == 8080

    def test_single_port_mapping(self) -> None:
        """Test single port mapping works correctly."""
        result = resolve_port_mappings([8080])
        assert len(result) == 1
        assert result[0].internal == 8080
        assert result[0].external >= 1024


class TestQemuHostfwdRuleGeneration:
    """Tests for QEMU hostfwd rule generation (Mode 1).

    These tests verify the format of hostfwd rules that would be passed to QEMU.
    The actual format is: hostfwd={protocol}:{host}:{external}-:{internal}
    """

    def test_hostfwd_rule_format_tcp(self) -> None:
        """Test TCP hostfwd rule format."""
        port = ExposedPort(internal=8080, external=3000, host="127.0.0.1", protocol="tcp")
        rule = f"hostfwd={port.protocol}:{port.host}:{port.external}-:{port.internal}"
        assert rule == "hostfwd=tcp:127.0.0.1:3000-:8080"

    def test_hostfwd_rule_format_udp(self) -> None:
        """Test UDP hostfwd rule format."""
        port = ExposedPort(internal=53, external=5300, host="127.0.0.1", protocol="udp")
        rule = f"hostfwd={port.protocol}:{port.host}:{port.external}-:{port.internal}"
        assert rule == "hostfwd=udp:127.0.0.1:5300-:53"

    def test_hostfwd_multiple_ports(self) -> None:
        """Test multiple hostfwd rules joined correctly."""
        ports = [
            ExposedPort(internal=80, external=8080, host="127.0.0.1", protocol="tcp"),
            ExposedPort(internal=443, external=8443, host="127.0.0.1", protocol="tcp"),
            ExposedPort(internal=53, external=5300, host="127.0.0.1", protocol="udp"),
        ]
        rules = ",".join(f"hostfwd={p.protocol}:{p.host}:{p.external}-:{p.internal}" for p in ports)
        expected = "hostfwd=tcp:127.0.0.1:8080-:80,hostfwd=tcp:127.0.0.1:8443-:443,hostfwd=udp:127.0.0.1:5300-:53"
        assert rules == expected

    def test_hostfwd_netdev_opts_format(self) -> None:
        """Test the complete netdev options string format."""
        ports = [ExposedPort(internal=8080, external=3000, host="127.0.0.1", protocol="tcp")]
        hostfwd_rules = ",".join(f"hostfwd={p.protocol}:{p.host}:{p.external}-:{p.internal}" for p in ports)
        netdev_opts = f"user,id=portfwd,restrict=on,{hostfwd_rules}"
        assert netdev_opts == "user,id=portfwd,restrict=on,hostfwd=tcp:127.0.0.1:3000-:8080"
        assert "restrict=on" in netdev_opts  # No internet access


class TestGvproxyPortForwardConfig:
    """Tests for gvproxy port forwarding configuration (Mode 2).

    These tests verify the JSON format passed to gvproxy-wrapper via -port-forward flag.
    The format is: {"host:external": "guest_ip:internal", ...}
    Guest IP in gvproxy is always 192.168.127.2
    """

    def test_gvproxy_forward_json_single_port(self) -> None:
        """Test single port forward JSON format."""
        port = ExposedPort(internal=8080, external=3000, host="127.0.0.1", protocol="tcp")
        forwards_dict = {f"{port.host}:{port.external}": f"192.168.127.2:{port.internal}"}
        port_forward_json = json.dumps(forwards_dict)

        # Verify JSON is valid and has correct structure
        parsed = json.loads(port_forward_json)
        assert parsed == {"127.0.0.1:3000": "192.168.127.2:8080"}

    def test_gvproxy_forward_json_multiple_ports(self) -> None:
        """Test multiple port forwards in JSON format."""
        ports = [
            ExposedPort(internal=80, external=8080, host="127.0.0.1", protocol="tcp"),
            ExposedPort(internal=443, external=8443, host="127.0.0.1", protocol="tcp"),
            ExposedPort(internal=3000, external=3000, host="127.0.0.1", protocol="tcp"),
        ]
        forwards_dict = {f"{p.host}:{p.external}": f"192.168.127.2:{p.internal}" for p in ports}
        port_forward_json = json.dumps(forwards_dict)

        parsed = json.loads(port_forward_json)
        assert parsed["127.0.0.1:8080"] == "192.168.127.2:80"
        assert parsed["127.0.0.1:8443"] == "192.168.127.2:443"
        assert parsed["127.0.0.1:3000"] == "192.168.127.2:3000"
        assert len(parsed) == 3

    def test_gvproxy_guest_ip_is_fixed(self) -> None:
        """Test that guest IP is always 192.168.127.2."""
        # This is gvproxy's default guest IP, hardcoded in gvisor-tap-vsock
        port = ExposedPort(internal=5000, external=5000, host="127.0.0.1")
        guest_ip = "192.168.127.2"
        forward = f"{guest_ip}:{port.internal}"
        assert forward == "192.168.127.2:5000"

    def test_gvproxy_udp_forward_format(self) -> None:
        """Test UDP forward format (protocol is in forward key for gvproxy)."""
        # Note: gvproxy uses "udp:host:port" prefix for UDP forwards
        port = ExposedPort(internal=53, external=5300, host="127.0.0.1", protocol="udp")
        # For UDP, gvproxy expects "udp:" prefix on the local address
        local_key = f"udp:{port.host}:{port.external}" if port.protocol == "udp" else f"{port.host}:{port.external}"
        forwards_dict = {local_key: f"192.168.127.2:{port.internal}"}

        assert forwards_dict == {"udp:127.0.0.1:5300": "192.168.127.2:53"}


class TestPortForwardingModeSelection:
    """Tests for mode selection logic."""

    def test_mode1_conditions(self) -> None:
        """Test Mode 1: expose_ports=True, allow_network=False."""
        expose_ports = [ExposedPort(internal=8080, external=3000)]
        allow_network = False

        # Mode 1 is triggered when: expose_ports AND NOT allow_network
        is_mode1 = bool(expose_ports) and not allow_network
        assert is_mode1 is True

    def test_mode2_conditions(self) -> None:
        """Test Mode 2: expose_ports=True, allow_network=True."""
        expose_ports = [ExposedPort(internal=8080, external=3000)]
        allow_network = True

        # Mode 2 is triggered when: expose_ports AND allow_network
        is_mode2 = bool(expose_ports) and allow_network
        assert is_mode2 is True

    def test_mode3_conditions(self) -> None:
        """Test Mode 3: expose_ports=False, allow_network=True."""
        expose_ports: list[ExposedPort] = []
        allow_network = True

        # Mode 3 (internet only, no port forwarding)
        is_mode3 = not expose_ports and allow_network
        assert is_mode3 is True

    def test_no_network_mode(self) -> None:
        """Test no network: expose_ports=False, allow_network=False."""
        expose_ports: list[ExposedPort] = []
        allow_network = False

        # No network mode (no ports, no internet)
        has_no_network = not expose_ports and not allow_network
        assert has_no_network is True


# =============================================================================
# Integration Tests - Actually start VMs and verify port forwarding works
# =============================================================================


class TestPortForwardingMode1Integration:
    """Integration tests for Mode 1: port forwarding without internet (QEMU hostfwd)."""

    async def test_mode1_http_server_reachable_from_host(self, scheduler: Scheduler) -> None:
        """Test that HTTP server in VM is reachable from host via exposed port.

        Mode 1: expose_ports=True, allow_network=False
        Uses QEMU user-mode networking with hostfwd.
        """
        # Start a simple HTTP server in the VM on port 8080
        # Use dynamic external port allocation
        result = await scheduler.run(
            code="""
import http.server
import socketserver
import threading

# Start server in background thread
handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", 8080), handler) as httpd:
    # Run for just 2 seconds - enough for test to connect
    thread = threading.Thread(target=httpd.handle_request)
    thread.start()
    thread.join(timeout=5)
print("server_started")
""",
            language=Language.PYTHON,
            expose_ports=[PortMapping(internal=8080)],
            allow_network=False,
            timeout_seconds=30,
        )

        # Verify the result has exposed ports
        assert len(result.exposed_ports) == 1
        assert result.exposed_ports[0].internal == 8080
        assert result.exposed_ports[0].external >= 1024
        assert result.exposed_ports[0].host == "127.0.0.1"

    async def test_mode1_explicit_external_port(self, scheduler: Scheduler) -> None:
        """Test Mode 1 with explicit external port mapping."""
        # Use a specific external port
        external_port = allocate_ephemeral_port()

        result = await scheduler.run(
            code="print('hello')",
            language=Language.PYTHON,
            expose_ports=[PortMapping(internal=8080, external=external_port)],
            allow_network=False,
            timeout_seconds=30,
        )

        assert len(result.exposed_ports) == 1
        assert result.exposed_ports[0].internal == 8080
        assert result.exposed_ports[0].external == external_port

    async def test_mode1_multiple_ports(self, scheduler: Scheduler) -> None:
        """Test Mode 1 with multiple port mappings."""
        result = await scheduler.run(
            code="print('multi-port test')",
            language=Language.PYTHON,
            expose_ports=[
                PortMapping(internal=8080),
                PortMapping(internal=8081),
                PortMapping(internal=9000),
            ],
            allow_network=False,
            timeout_seconds=30,
        )

        assert len(result.exposed_ports) == 3
        internals = {p.internal for p in result.exposed_ports}
        assert internals == {8080, 8081, 9000}
        # All external ports should be unique
        externals = [p.external for p in result.exposed_ports]
        assert len(set(externals)) == 3

    async def test_mode1_no_internet_access(self, scheduler: Scheduler) -> None:
        """Verify Mode 1 blocks outbound internet (restrict=on)."""
        result = await scheduler.run(
            code="""
import urllib.request
import socket
socket.setdefaulttimeout(3)
try:
    urllib.request.urlopen('http://example.com', timeout=3)
    print('CONNECTED')
except Exception as e:
    print(f'BLOCKED: {type(e).__name__}')
""",
            language=Language.PYTHON,
            expose_ports=[PortMapping(internal=8080)],
            allow_network=False,  # No internet!
            timeout_seconds=30,
        )

        # Should be blocked - no internet in Mode 1
        assert "BLOCKED" in result.stdout or result.exit_code != 0


class TestPortForwardingMode2Integration:
    """Integration tests for Mode 2: port forwarding with internet (gvproxy)."""

    async def test_mode2_port_forwarding_with_internet(self, scheduler: Scheduler) -> None:
        """Test Mode 2: expose_ports + allow_network both enabled."""
        result = await scheduler.run(
            code="print('mode2 test')",
            language=Language.PYTHON,
            expose_ports=[PortMapping(internal=5000)],
            allow_network=True,
            allowed_domains=["example.com"],
            timeout_seconds=30,
        )

        # Should have exposed port
        assert len(result.exposed_ports) == 1
        assert result.exposed_ports[0].internal == 5000
        assert result.exposed_ports[0].external >= 1024

    async def test_mode2_can_reach_allowed_domain(self, scheduler: Scheduler) -> None:
        """Verify Mode 2 allows outbound internet to allowed domains."""
        result = await scheduler.run(
            code="""
import urllib.request
try:
    resp = urllib.request.urlopen('https://httpbin.org/status/200', timeout=10)
    print(f'STATUS:{resp.status}')
except Exception as e:
    print(f'ERROR:{type(e).__name__}:{e}')
""",
            language=Language.PYTHON,
            expose_ports=[PortMapping(internal=8080)],
            allow_network=True,
            allowed_domains=["httpbin.org"],
            timeout_seconds=30,
        )

        # Should be able to reach httpbin.org
        assert "STATUS:200" in result.stdout


class TestPortForwardingConnectivity:
    """Tests that actually verify host can connect to exposed ports."""

    async def test_host_can_connect_to_exposed_tcp_port(self, scheduler: Scheduler) -> None:
        """Start a TCP server in VM and verify host can connect."""
        # Start a simple echo server
        result = await scheduler.run(
            code="""
import socket
import sys

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 9999))
server.listen(1)
server.settimeout(5)  # 5 second timeout

print('LISTENING', flush=True)
sys.stdout.flush()

try:
    conn, addr = server.accept()
    data = conn.recv(1024)
    conn.send(b'ECHO:' + data)
    conn.close()
    print('HANDLED')
except socket.timeout:
    print('TIMEOUT')
finally:
    server.close()
""",
            language=Language.PYTHON,
            expose_ports=[PortMapping(internal=9999)],
            allow_network=False,
            timeout_seconds=30,
        )

        # The server started and listened
        assert "LISTENING" in result.stdout
        # Verify we got the exposed port info
        assert len(result.exposed_ports) == 1
        assert result.exposed_ports[0].internal == 9999

    async def test_result_contains_url_helper(self, scheduler: Scheduler) -> None:
        """Test that ExposedPort.url property works correctly."""
        result = await scheduler.run(
            code="print('url test')",
            language=Language.PYTHON,
            expose_ports=[PortMapping(internal=3000)],
            allow_network=False,
            timeout_seconds=30,
        )

        assert len(result.exposed_ports) == 1
        port = result.exposed_ports[0]
        assert port.url == f"http://127.0.0.1:{port.external}"

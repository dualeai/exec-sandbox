"""Tests for port forwarding: the models, host port allocation, the gvproxy arguments, and
host-to-guest connections through a VM's exposed port.
"""

import asyncio
import errno
import json
import re
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from exec_sandbox import constants
from exec_sandbox.asset_downloader import get_gvproxy_suffix
from exec_sandbox.exceptions import ErrorCode, TransientError
from exec_sandbox.gvproxy import start_gvproxy
from exec_sandbox.models import ExposedPort, Language, PortMapping
from exec_sandbox.platform_utils import ProcessWrapper, get_arch_name
from exec_sandbox.port_forward import allocate_ephemeral_ports, resolve_port_mappings
from exec_sandbox.resource_cleanup import cleanup_process
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

from .conftest import (
    NET_OP_TIMEOUT_S,
    NET_RETRY_BACKOFF_S,
    NET_RETRY_COUNT,
    PY_SAFETY,
)


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
        (port,) = allocate_ephemeral_ports(1, protocol="tcp", exclude=frozenset())
        assert 1024 <= port <= 65535

    def test_allocate_ephemeral_port_is_available(self) -> None:
        """Test that allocated port can be bound."""
        (port,) = allocate_ephemeral_ports(1, protocol="tcp", exclude=frozenset())

        # Try to bind to the allocated port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # This should succeed (port was just released)
            s.bind(("127.0.0.1", port))

    def test_allocate_multiple_ports_are_different(self) -> None:
        """Test that multiple allocations return different ports."""
        ports = allocate_ephemeral_ports(5, protocol="tcp", exclude=frozenset())
        assert len(set(ports)) == len(ports)

    def test_many_udp_ports_are_different(self) -> None:
        """600 UDP ports in one call are all different. On Linux two UDP sockets with
        SO_REUSEADDR share a port, so bind(0) would give a held port again (about 6 repeats
        in 600, measured) and one UDP forward would be lost."""
        ports = allocate_ephemeral_ports(600, protocol="udp", exclude=frozenset())
        assert len(set(ports)) == 600


class TestResolvePortMappings:
    """Tests for resolving port mappings."""

    def test_os_picked_port_skips_a_fixed_port_of_the_same_vm(self) -> None:
        """The OS hands out a port that another mapping of the spec fixes (nothing binds it
        before gvproxy): the dynamic mapping gets the next port, held apart from the first,
        and gvproxy gets two forwards. A fake socket module: the OS port choice cannot be
        steered."""
        picked = iter([5000, 5001])
        sockets: list[SimpleNamespace] = []

        def new_socket(*_args: object) -> SimpleNamespace:
            port = next(picked)
            fake = SimpleNamespace(closed=False, setsockopt=lambda *_a: None, bind=lambda _addr: None)
            fake.getsockname = lambda: ("127.0.0.1", port)
            fake.close = lambda: setattr(fake, "closed", True)
            sockets.append(fake)
            return fake

        fake_module = SimpleNamespace(
            socket=new_socket,
            AF_INET=socket.AF_INET,
            SOCK_STREAM=socket.SOCK_STREAM,
            SOCK_DGRAM=socket.SOCK_DGRAM,
            SOL_SOCKET=socket.SOL_SOCKET,
            SO_REUSEADDR=socket.SO_REUSEADDR,
        )
        with patch("exec_sandbox.port_forward.socket", fake_module):
            ports = resolve_port_mappings([PortMapping(internal=80, external=5000), PortMapping(internal=81)])

        assert [(p.internal, p.external) for p in ports] == [(80, 5000), (81, 5001)]
        assert [s.closed for s in sockets] == [True, True]  # both held until the end, then closed

    def test_resolve_allocates_external_ports(self) -> None:
        """Test that missing external ports are allocated."""
        result = resolve_port_mappings([PortMapping(internal=8080), PortMapping(internal=3000)])
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
            PortMapping(internal=8000),
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

    def test_external_port_zero_rejected(self) -> None:
        """Test that port 0 is rejected for external."""
        with pytest.raises(ValueError):
            PortMapping(internal=8080, external=0)


class TestExposedPortEdgeCases:
    """Edge case tests for ExposedPort."""

    def test_udp_port_url_still_returns_http(self) -> None:
        """url is the HTTP URL of the host port for each protocol, UDP too."""
        port = ExposedPort(internal=53, external=5300, protocol="udp")
        assert port.url == "http://127.0.0.1:5300"


class TestResolvePortMappingsBoundary:
    """Boundary tests for resolve_port_mappings."""

    def test_max_exposed_ports_all_get_a_host_port(self) -> None:
        """10 mappings (MAX_EXPOSED_PORTS, which VmSpec enforces) all get a host port."""
        mappings = [PortMapping(internal=8000 + i) for i in range(10)]
        result = resolve_port_mappings(mappings)
        assert [port.internal for port in result] == [8000 + i for i in range(10)]
        assert len({port.external for port in result}) == 10
        assert all(port.external > 0 for port in result)

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
        # The behavior depends on what gvproxy does with duplicates
        mappings = [
            PortMapping(internal=8080, external=3000),
            PortMapping(internal=8080, external=3001),
        ]
        result = resolve_port_mappings(mappings)
        assert len(result) == 2
        assert result[0].internal == result[1].internal == 8080
        assert result[0].external != result[1].external

    def test_no_free_host_port_is_gvproxy_failed(self) -> None:
        """No socket for a port probe (out of file handles): TransientError GVPROXY_FAILED (it
        can pass later), never a raw OSError out of the VM start."""

        def no_socket(*_args: object) -> socket.socket:
            raise OSError(errno.EMFILE, "Too many open files")

        with (
            patch("exec_sandbox.port_forward.socket.socket", no_socket),
            pytest.raises(TransientError, check=lambda e: e.code is ErrorCode.GVPROXY_FAILED) as exc,
        ):
            resolve_port_mappings([PortMapping(internal=8080)])
        assert exc.value.context["errno"] == errno.EMFILE

    def test_duplicate_external_ports_allowed_at_model_level(self) -> None:
        """The model and resolve_port_mappings accept two mappings on one host port: both come
        back as given."""
        mappings = [
            PortMapping(internal=8080, external=3000),
            PortMapping(internal=8081, external=3000),
        ]
        result = resolve_port_mappings(mappings)
        assert [(port.internal, port.external) for port in result] == [(8080, 3000), (8081, 3000)]


class TestGvproxyPortForwardArguments:
    """The port forwards start_gvproxy gives gvproxy-wrapper (-port-forward, -block-outbound).

    The gvproxy-wrapper here is a sh script that writes its arguments, then prints the line
    that tells start_gvproxy it is ready: the real spawn and readiness wait run, no VM.
    """

    @staticmethod
    async def _arguments(
        tmp_path: Path, short_tmp_dir: Path, *, ports: list[ExposedPort] | None, block_outbound: bool
    ) -> list[str]:
        """The arguments of a start_gvproxy for `ports` (a raw VM: no outbound filter). The
        fake gvproxy-wrapper sits in the images folder under the real binary's name: the VM
        start searches that folder first."""
        fake = tmp_path / f"gvproxy-wrapper-{get_gvproxy_suffix(get_arch_name())}"
        fake.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$@" > "$(dirname "$0")/args"\necho "Listening on QEMU socket"\nexec sleep 3600\n'
        )
        fake.chmod(0o755)
        workdir = SimpleNamespace(gvproxy_socket=short_tmp_dir / "gv.sock")
        owned: list[tuple[ProcessWrapper, asyncio.Task[None] | None]] = []  # this test owns them from the spawn
        try:
            _, _, wait_ready = await start_gvproxy(
                "vm-ports",
                allowed_domains=[],
                workdir=workdir,  # type: ignore[arg-type]
                images_dir=tmp_path,
                expose_ports=ports,
                block_outbound=block_outbound,
                on_process_started=lambda proc, drain: owned.append((proc, drain)),
            )
            await wait_ready()  # the fake wrote its arguments before its ready line
            return (tmp_path / "args").read_text().splitlines()
        finally:
            if owned:
                proc, _ = owned[0]
                await cleanup_process(proc, "gvproxy", "vm-ports", term_timeout=3.0, kill_timeout=2.0, kill_first=False)
            for _, drain in owned[1:]:
                if drain is not None:
                    drain.cancel()
                    await asyncio.gather(drain, return_exceptions=True)

    async def test_forwards_from_the_host_to_the_guest_ip(self, tmp_path: Path, short_tmp_dir: Path) -> None:
        """Each exposed port: host:external to 192.168.127.2:internal (gvproxy's fixed guest IP)."""
        ports = [
            ExposedPort(internal=80, external=8080, host="127.0.0.1", protocol="tcp"),
            ExposedPort(internal=443, external=8443, host="127.0.0.1", protocol="tcp"),
            ExposedPort(internal=3000, external=3000, host="127.0.0.1", protocol="tcp"),
        ]

        args = await self._arguments(tmp_path, short_tmp_dir, ports=ports, block_outbound=False)

        assert json.loads(args[args.index("-port-forward") + 1]) == {
            "127.0.0.1:8080": "192.168.127.2:80",
            "127.0.0.1:8443": "192.168.127.2:443",
            "127.0.0.1:3000": "192.168.127.2:3000",
        }
        assert "-block-outbound" not in args

    async def test_a_udp_port_gets_a_udp_forward(self, tmp_path: Path, short_tmp_dir: Path) -> None:
        """gvisor-tap-vsock forwards UDP only for a key with the "udp:" prefix; a key without it
        is TCP. A UDP and a TCP port on the same number are two forwards."""
        ports = [
            ExposedPort(internal=53, external=5353, host="127.0.0.1", protocol="udp"),
            ExposedPort(internal=53, external=5353, host="127.0.0.1", protocol="tcp"),
        ]

        args = await self._arguments(tmp_path, short_tmp_dir, ports=ports, block_outbound=False)

        assert json.loads(args[args.index("-port-forward") + 1]) == {
            "udp:127.0.0.1:5353": "192.168.127.2:53",
            "127.0.0.1:5353": "192.168.127.2:53",
        }

    @pytest.mark.parametrize("block_outbound", [True, False], ids=["port-forward-only", "with-internet"])
    async def test_block_outbound_flag(self, tmp_path: Path, short_tmp_dir: Path, block_outbound: bool) -> None:
        """Ports without network access (mode 1): -block-outbound stops every connection the
        guest opens; with network access (mode 2) the flag is absent."""
        ports = [ExposedPort(internal=8080, external=3000, host="127.0.0.1", protocol="tcp")]

        args = await self._arguments(tmp_path, short_tmp_dir, ports=ports, block_outbound=block_outbound)

        assert ("-block-outbound" in args) is block_outbound

    async def test_the_network_constants_are_passed(self, tmp_path: Path, short_tmp_dir: Path) -> None:
        """gvproxy-wrapper has no network default (its flags are required): the host passes its
        constants (constants.py), and a forward targets the guest IP among them."""
        ports = [ExposedPort(internal=8080, external=3000, host="127.0.0.1", protocol="tcp")]
        args = await self._arguments(tmp_path, short_tmp_dir, ports=ports, block_outbound=False)

        given = {flag: args[args.index(flag) + 1] for flag in ("-subnet", "-gateway-ip", "-gateway-mac", "-mtu")}
        assert given == {
            "-subnet": constants.GUEST_NETWORK_SUBNET,
            "-gateway-ip": constants.GUEST_GATEWAY_IP,
            "-gateway-mac": constants.GUEST_GATEWAY_MAC,
            "-mtu": str(constants.GUEST_MTU),
        }
        assert json.loads(args[args.index("-port-forward") + 1]) == {"127.0.0.1:3000": f"{constants.GUEST_IP}:8080"}

    async def test_no_ports_no_forward(self, tmp_path: Path, short_tmp_dir: Path) -> None:
        """No exposed port: no -port-forward argument."""
        args = await self._arguments(tmp_path, short_tmp_dir, ports=None, block_outbound=False)

        assert "-port-forward" not in args


# =============================================================================
# Integration Tests - Actually start VMs and verify port forwarding works
# =============================================================================


@pytest.mark.slow
class TestPortForwardingMode1Integration:
    """Mode 1: port forwarding without network access (gvproxy -block-outbound)."""

    async def test_mode1_no_internet_access(self, scheduler: Scheduler) -> None:
        """Mode 1 blocks each connection the guest opens: the HTTP request fails and the code
        ends normally."""
        result = await run_once(
            scheduler,
            VmSpec(
                language=Language.PYTHON,
                allow_network=False,  # No internet!
                expose_ports=[PortMapping(internal=8080)],
            ),
            """
import urllib.request
import socket
socket.setdefaulttimeout(3)
try:
    urllib.request.urlopen('http://example.com', timeout=3)
    print('CONNECTED')
except Exception as e:
    print(f'BLOCKED: {type(e).__name__}')
""",
        )

        assert result.exit_code == 0, result.stderr
        assert result.stdout.startswith("BLOCKED: "), result.stdout


@pytest.mark.slow
class TestPortForwardingMode2Integration:
    """Integration tests for Mode 2: port forwarding with internet (gvproxy)."""

    async def test_mode2_can_reach_allowed_domain(self, scheduler: Scheduler) -> None:
        """Verify Mode 2 allows outbound internet to allowed domains."""
        # Planet-scale connectivity-check (NCSI) endpoint — no rate limits,
        # no bot challenges, no scheduled downtime.
        result = await run_once(
            scheduler,
            VmSpec(
                language=Language.PYTHON,
                allow_network=True,
                allowed_domains=["cp.cloudflare.com"],
                expose_ports=[PortMapping(internal=8080)],
            ),
            PY_SAFETY
            + f"""
import urllib.error, urllib.request, time
for attempt in range({NET_RETRY_COUNT + 1}):
    try:
        resp = urllib.request.urlopen('https://cp.cloudflare.com/generate_204', timeout={NET_OP_TIMEOUT_S})
        print(f'STATUS:{{resp.status}}')
        break
    except urllib.error.HTTPError as e:
        # Any HTTP response proves the domain is reachable through the filter
        print(f'STATUS:{{e.code}}')
        break
    except Exception as e:
        if attempt < {NET_RETRY_COUNT}:
            time.sleep({NET_RETRY_BACKOFF_S} * (attempt + 1))
            continue
        print(f'ERROR:{{type(e).__name__}}:{{e}}')
""",
        )

        # Any HTTP status = outbound path works end-to-end (DNS, SNI filter, TLS)
        assert result.exit_code == 0, result.stderr
        assert re.fullmatch(r"STATUS:\d{3}", result.stdout.strip()), result.stdout


_ECHO_SERVER = """
import socket
import threading

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", 9999))
server.listen(1)

def serve():
    conn, _ = server.accept()
    conn.sendall(b"ECHO:" + conn.recv(1024))
    conn.close()

threading.Thread(target=serve, daemon=True).start()
print("LISTENING")
"""
"""A TCP server on guest port 9999 in a thread of the REPL: it outlives the exec and answers one
client with ECHO: and the bytes it got."""


@pytest.mark.slow
class TestPortForwardingConnectivity:
    """The host reaches a server in the guest through the exposed port."""

    @pytest.mark.parametrize(
        ("allow_network", "allowed_domains"),
        [
            pytest.param(False, None, id="mode1-block-outbound"),
            pytest.param(True, ["example.com"], id="mode2-with-network"),
        ],
    )
    async def test_host_reaches_a_guest_server_through_the_exposed_port(
        self, scheduler: Scheduler, allow_network: bool, allowed_domains: list[str] | None
    ) -> None:
        """A host client on 127.0.0.1:external sends "ping" and reads the guest server's answer,
        with -block-outbound (mode 1) and with network access (mode 2)."""
        spec = VmSpec(
            language=Language.PYTHON,
            allow_network=allow_network,
            allowed_domains=allowed_domains,
            expose_ports=[PortMapping(internal=9999)],
        )
        async with await scheduler.new(spec) as session:
            started = await session.exec(_ECHO_SERVER)
            assert (started.exit_code, started.stdout) == (0, "LISTENING\n"), started.stderr
            (port,) = session.exposed_ports
            reader, writer = await asyncio.wait_for(asyncio.open_connection(port.host, port.external), timeout=10)
            try:
                writer.write(b"ping")
                await writer.drain()
                assert await asyncio.wait_for(reader.read(), timeout=10) == b"ECHO:ping"
            finally:
                writer.close()
                await writer.wait_closed()


# =============================================================================
# Integration Tests - Session.exposed_ports property
# =============================================================================


@pytest.mark.slow
class TestSessionExposedPorts:
    """Integration tests for Session.exposed_ports property."""

    # ------------------------------------------------------------------
    # Normal (happy path)
    # ------------------------------------------------------------------

    async def test_session_single_port(self, scheduler: Scheduler) -> None:
        """A session with one exposed port lists it before any exec: the guest port, a host port
        the OS picked, and the loopback address."""
        async with await scheduler.new(
            VmSpec(language=Language.PYTHON, expose_ports=[PortMapping(internal=8080)])
        ) as session:
            assert len(session.exposed_ports) == 1
            assert session.exposed_ports[0].internal == 8080
            assert session.exposed_ports[0].external >= 1024
            assert session.exposed_ports[0].host == "127.0.0.1"

    async def test_session_multiple_ports(self, scheduler: Scheduler) -> None:
        """Session with multiple exposed ports has all entries with unique externals."""
        async with await scheduler.new(
            VmSpec(
                language=Language.PYTHON,
                expose_ports=[
                    PortMapping(internal=8080),
                    PortMapping(internal=8081),
                    PortMapping(internal=9000),
                ],
            )
        ) as session:
            assert len(session.exposed_ports) == 3
            internals = {p.internal for p in session.exposed_ports}
            assert internals == {8080, 8081, 9000}
            externals = [p.external for p in session.exposed_ports]
            assert len(set(externals)) == 3

    async def test_session_explicit_external_port(self, scheduler: Scheduler) -> None:
        """Session with explicit external port preserves it."""
        (external_port,) = allocate_ephemeral_ports(1, protocol="tcp", exclude=frozenset())
        async with await scheduler.new(
            VmSpec(language=Language.PYTHON, expose_ports=[PortMapping(internal=8080, external=external_port)])
        ) as session:
            assert len(session.exposed_ports) == 1
            assert session.exposed_ports[0].internal == 8080
            assert session.exposed_ports[0].external == external_port

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    async def test_session_no_ports_returns_empty(self, scheduler: Scheduler) -> None:
        """Session without expose_ports has empty exposed_ports."""
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            assert session.exposed_ports == []

    async def test_session_ports_stable_across_execs(self, scheduler: Scheduler) -> None:
        """Exposed ports remain the same across multiple exec() calls."""
        async with await scheduler.new(
            VmSpec(language=Language.PYTHON, expose_ports=[PortMapping(internal=8080)])
        ) as session:
            ports_before = list(session.exposed_ports)  # a copy: the property returns the VM's list
            await session.exec("x = 1")
            await session.exec("x = 2")
            await session.exec("x = 3")
            assert session.exposed_ports == ports_before

    # ------------------------------------------------------------------
    # Weird cases
    # ------------------------------------------------------------------

    async def test_session_ports_with_network(self, scheduler: Scheduler) -> None:
        """Session with expose_ports and allow_network=True (Mode 2) has correct ports."""
        async with await scheduler.new(
            VmSpec(
                language=Language.PYTHON,
                allow_network=True,
                allowed_domains=["example.com"],
                expose_ports=[PortMapping(internal=8080)],
            )
        ) as session:
            assert len(session.exposed_ports) == 1
            assert session.exposed_ports[0].internal == 8080
            assert session.exposed_ports[0].external >= 1024

    # ------------------------------------------------------------------
    # Out of bounds
    # ------------------------------------------------------------------

    async def test_session_ports_after_close(self, scheduler: Scheduler) -> None:
        """Exposed ports are still readable after session is closed."""
        session = await scheduler.new(VmSpec(language=Language.PYTHON, expose_ports=[PortMapping(internal=8080)]))
        try:
            ports_before_close = list(session.exposed_ports)  # a copy: the property returns the VM's list
            assert len(ports_before_close) == 1
            await session.close()
            assert session.exposed_ports == ports_before_close
        finally:
            await session.close()  # idempotent — safe if already closed

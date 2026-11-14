"""Defense-in-depth tests for DNS resolv.conf hijack prevention.

Verifies that even if /etc/resolv.conf were modified, gvproxy intercepts
DNS at the network layer — the sole egress route (192.168.127.1 gateway)
ensures all traffic flows through gvproxy regardless of resolv.conf content.

Two independent defense layers are tested:
1. Network layer: gvproxy owns the gateway; direct DNS to external resolvers
   (e.g. 8.8.8.8) is blocked at the network level.
2. Filesystem layer: /etc/resolv.conf is bind-mounted read-only (EROFS),
   preventing modification even by root.
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

# =============================================================================
# Network layer: gvproxy blocks direct DNS to external resolvers
# =============================================================================


class TestNetworkLayerDNSBlocking:
    """Prove gvproxy intercepts DNS at the network layer, independent of resolv.conf."""

    async def test_direct_tcp_to_external_dns_blocked(self, scheduler: Scheduler) -> None:
        """Raw TCP to 8.8.8.8:53 is refused by gvproxy when outbound filtering is active.

        Even if resolv.conf pointed at 8.8.8.8, the TCP connection cannot
        reach it: gvproxy refuses non-TLS traffic to arbitrary IPs (ECONNREFUSED, not a time-out).
        """
        code = """\
import socket
try:
    sock = socket.create_connection(("8.8.8.8", 53), timeout=5)
    sock.sendall(b"\\x00\\x1e\\x01\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00"
                 b"\\x07example\\x03com\\x00\\x00\\x01\\x00\\x01")
    data = sock.recv(1024)
    sock.close()
    print(f"TCP_CONNECTED:{len(data)}")
except Exception as e:
    print(f"BLOCKED:{type(e).__name__}")
"""
        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=["cp.cloudflare.com"]), code
        )
        assert result.stdout == "BLOCKED:ConnectionRefusedError\n", (
            f"Expected gvproxy to refuse raw TCP to 8.8.8.8:53.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    async def test_udp_dns_to_external_resolver_blocked(self, scheduler: Scheduler) -> None:
        """Raw UDP DNS query to 8.8.8.8:53 for a non-allowed domain gets no reply (the 3 s socket time-out).

        Constructs a minimal DNS query for example.com and sends it via UDP
        directly to Google's public DNS. gvproxy does not forward it.
        """
        code = """\
import socket
import struct

# Minimal DNS query for example.com A record
query = (
    struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)  # header
    + b"\\x07example\\x03com\\x00"  # QNAME
    + struct.pack("!HH", 1, 1)  # QTYPE=A, QCLASS=IN
)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(3)
try:
    sock.sendto(query, ("8.8.8.8", 53))
    data, _ = sock.recvfrom(512)
    print(f"UDP_RESPONSE:{len(data)}")
except Exception as e:
    print(f"BLOCKED:{type(e).__name__}")
finally:
    sock.close()
"""
        result = await run_once(
            scheduler, VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=["cp.cloudflare.com"]), code
        )
        assert result.stdout == "BLOCKED:TimeoutError\n", (
            f"Expected UDP DNS to 8.8.8.8:53 to get no reply.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    async def test_only_route_is_through_gvproxy_gateway(self, scheduler: Scheduler) -> None:
        """The routes are the default one via 192.168.127.1 (the gvproxy gateway) and the link
        route of eth0: no alternative route could bypass gvproxy filtering. (BusyBox ip prints
        every route for `show default`.)
        """
        result = await run_once(
            scheduler,
            VmSpec(language=Language.RAW, allow_network=True, allowed_domains=["cp.cloudflare.com"]),
            "ip route show default",
        )
        assert (result.exit_code, result.stdout) == (
            0,
            "default via 192.168.127.1 dev eth0 \n192.168.127.0/24 dev eth0 scope link  src 192.168.127.2 \n",
        ), result.stderr


# =============================================================================
# Filesystem layer: /etc/resolv.conf is read-only bind mount
# =============================================================================


class TestResolvConfReadOnlyMount:
    """Verify /etc/resolv.conf is protected by read-only bind mount (EROFS)."""

    async def test_resolv_conf_is_readonly_mount(self, scheduler: Scheduler) -> None:
        """/etc/resolv.conf is on a read-only mount: os.statvfs() gives the ST_RDONLY flag.

        The open(2) for write that this mount refuses (EROFS, before the Unix permissions):
        test_etc_hardening.py::TestEtcWeirdCases::test_ctypes_open_wronly.
        """
        code = """\
import os
print(f"statvfs_rdonly={bool(os.statvfs('/etc/resolv.conf').f_flag & os.ST_RDONLY)}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "statvfs_rdonly=True\n"), result.stderr

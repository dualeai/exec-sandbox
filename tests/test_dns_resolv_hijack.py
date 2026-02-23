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

# =============================================================================
# Network layer: gvproxy blocks direct DNS to external resolvers
# =============================================================================


class TestNetworkLayerDNSBlocking:
    """Prove gvproxy intercepts DNS at the network layer, independent of resolv.conf."""

    async def test_direct_tcp_to_external_dns_blocked(self, scheduler: Scheduler) -> None:
        """Raw TCP to 8.8.8.8:53 is blocked by gvproxy when outbound filtering is active.

        Even if resolv.conf pointed at 8.8.8.8, the TCP connection cannot
        reach it — gvproxy does not forward non-TLS traffic to arbitrary IPs.
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
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            allow_network=True,
            allowed_domains=["httpbin.org"],
        )
        assert "BLOCKED:" in result.stdout, (
            f"Expected raw TCP to 8.8.8.8:53 to be blocked by gvproxy.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    async def test_udp_dns_to_external_resolver_blocked(self, scheduler: Scheduler) -> None:
        """Raw UDP DNS query to 8.8.8.8:53 for non-allowed domain times out.

        Constructs a minimal DNS query for example.com and sends it via UDP
        directly to Google's public DNS. gvproxy should not forward this.
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
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
            allow_network=True,
            allowed_domains=["httpbin.org"],
        )
        assert "BLOCKED:" in result.stdout, (
            f"Expected UDP DNS to 8.8.8.8:53 to be blocked/timeout.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    async def test_only_route_is_through_gvproxy_gateway(self, scheduler: Scheduler) -> None:
        """ip route shows sole default route via 192.168.127.1 (gvproxy gateway).

        No alternative route exists that could bypass gvproxy filtering.
        """
        result = await scheduler.run(
            code="ip route show default",
            language=Language.RAW,
            allow_network=True,
            allowed_domains=["httpbin.org"],
        )
        assert "192.168.127.1" in result.stdout, (
            f"Expected default route via gvproxy gateway 192.168.127.1.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # Should be the only default route (no bypass paths)
        default_lines = [line for line in result.stdout.strip().splitlines() if line.startswith("default")]
        assert len(default_lines) == 1, (
            f"Expected exactly one default route, found {len(default_lines)}.\nroutes: {default_lines}"
        )


# =============================================================================
# Filesystem layer: /etc/resolv.conf is read-only bind mount
# =============================================================================


class TestResolvConfReadOnlyMount:
    """Verify /etc/resolv.conf is protected by read-only bind mount (EROFS)."""

    async def test_resolv_conf_is_readonly_mount(self, scheduler: Scheduler) -> None:
        """Write to /etc/resolv.conf returns EROFS (Read-only file system).

        Uses os.statvfs() to check the ST_RDONLY flag on the mount point,
        confirming the bind-mount is active — distinct from EACCES (Unix perms).
        Also verifies via ctypes open(2) that the kernel returns EROFS (errno 30).
        """
        code = """\
import os, ctypes, errno

# Check 1: statvfs ST_RDONLY flag
st = os.statvfs('/etc/resolv.conf')
is_rdonly = bool(st.f_flag & os.ST_RDONLY)
print(f"statvfs_rdonly={is_rdonly}")

# Check 2: open(2) returns EROFS (errno 30) for root-level write attempt
# Note: UID 1000 may get EACCES first (DAC check), but the mount flag
# is still detectable via statvfs above.
libc = ctypes.CDLL("libc.so.6", use_errno=True)
fd = libc.open(b"/etc/resolv.conf", os.O_WRONLY)
err = ctypes.get_errno()
if fd == -1:
    if err == errno.EROFS:
        print("open_blocked=erofs")
    elif err == errno.EACCES:
        print("open_blocked=eacces")
    else:
        print(f"open_blocked=errno_{err}")
else:
    os.close(fd)
    print("open_blocked=none")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert "statvfs_rdonly=True" in result.stdout, (
            f"Expected /etc/resolv.conf to be on a read-only mount.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "open_blocked=" in result.stdout
        assert "open_blocked=none" not in result.stdout, (
            f"Expected open(2) to fail on /etc/resolv.conf.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

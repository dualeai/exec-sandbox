"""Tests for loopback interface (lo) in the guest VM.

The guest-agent brings up `lo` over rtnetlink in setup_network() (guest-agent/src/init.rs).
Without this, all localhost/127.0.0.1 connections fail inside the microvm — breaking
http.createServer + fetch('http://localhost:...') in Bun, Python socket servers, etc.

Test categories:
- Normal: loopback interface is UP, TCP round-trip works on all backends
- Edge: loopback works without network, address variants, port reuse
- Weird: unusual but valid patterns (UDP, multiple servers, Bun-native API)
- Out of bounds: privileged ports rejected, ephemeral port allocation
"""

import re

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once


def _py_tcp_roundtrip_code(*, bind: str) -> str:
    """Self-contained asyncio TCP server on `bind` that answers PING with PONG: the client
    connects to 127.0.0.1 and prints what it sent and got."""
    return f"""\
import asyncio

async def main():
    async def handle(reader, writer):
        await reader.read(1024)
        writer.write(b'PONG')
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, '{bind}', 0)
    port = server.sockets[0].getsockname()[1]

    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    writer.write(b'PING')
    await writer.drain()
    resp = await reader.read(1024)
    writer.close()
    await writer.wait_closed()
    server.close()
    await server.wait_closed()
    print(f"sent:PING recv:{{resp.decode()}}")

asyncio.run(main())
"""


def _js_http_fetch_code(*, fetch_host: str) -> str:
    """Self-contained node:http server on 127.0.0.1 + fetch of `fetch_host` that prints the
    status and the body."""
    return f"""\
const http = require('http');
const server = http.createServer((req, res) => {{
  res.writeHead(200);
  res.end('HELLO_FROM_SERVER');
}});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;
const res = await fetch('http://{fetch_host}:' + port);
const body = await res.text();
server.close();
console.log('status:' + res.status + ' body:' + body);
"""


# =============================================================================
# Normal: Loopback interface exists and is UP
# =============================================================================
class TestLoopbackInterfaceUp:
    """Loopback interface is configured and operational."""

    async def test_loopback_up_shell(self, scheduler: Scheduler) -> None:
        """ip addr show lo reports the UP and LOWER_UP flags and the addresses 127.0.0.1/8 and ::1.

        A bind to 127.0.0.1 from Python and JavaScript: test_python_ephemeral_port_bind and
        test_javascript_http_server_fetch.
        """
        result = await run_once(scheduler, VmSpec(language=Language.RAW), "ip addr show lo")
        assert result.exit_code == 0
        assert result.stdout == (
            "1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN qlen 1000\n"
            "    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00\n"
            "    inet 127.0.0.1/8 scope host lo\n"
            "       valid_lft forever preferred_lft forever\n"
            "    inet6 ::1/128 scope host \n"
            "       valid_lft forever preferred_lft forever\n"
        )


# =============================================================================
# Normal: TCP round-trip on loopback (bind + connect + send/recv)
# =============================================================================
class TestLoopbackTcpRoundtrip:
    """Bind, connect, and exchange data over loopback. The default VmSpec has no network
    access (allow_network=False: no gvproxy, no eth0): loopback works without it."""

    async def test_python_tcp_loopback(self, scheduler: Scheduler) -> None:
        """Python TCP server + client on 127.0.0.1 exchange PING/PONG."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), _py_tcp_roundtrip_code(bind="127.0.0.1"))
        assert result.exit_code == 0
        assert result.stdout == "sent:PING recv:PONG\n"

    async def test_javascript_http_server_fetch(self, scheduler: Scheduler) -> None:
        """node:http createServer + fetch on 127.0.0.1: the case that fails when lo is down."""
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), _js_http_fetch_code(fetch_host="127.0.0.1")
        )
        assert result.exit_code == 0
        assert result.stdout == "status:200 body:HELLO_FROM_SERVER\n"

    async def test_shell_tcp_loopback(self, scheduler: Scheduler) -> None:
        """Shell nc listener + sender exchange data on loopback."""
        code = """\
# Start listener in background (BusyBox nc: -l -p PORT)
nc -l -p 9999 > /tmp/nc_out &
LISTENER_PID=$!
# Retry sending data until listener is ready (avoids fixed sleep race)
for i in $(seq 1 20); do
  echo "SHELL_PING" | nc -w 1 127.0.0.1 9999 2>/dev/null && break
  sleep 0.1
done
wait $LISTENER_PID 2>/dev/null || true
cat /tmp/nc_out
"""
        result = await run_once(scheduler, VmSpec(language=Language.RAW), code)
        assert result.exit_code == 0
        assert result.stdout == "SHELL_PING\n"


# =============================================================================
# Edge: Address variants and binding patterns
# =============================================================================
class TestLoopbackEdgeCases:
    """Address variants and common real-world binding patterns."""

    async def test_python_bind_0000_connect_localhost(self, scheduler: Scheduler) -> None:
        """Bind 0.0.0.0, connect via 127.0.0.1 — common server pattern."""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), _py_tcp_roundtrip_code(bind="0.0.0.0"))
        assert result.exit_code == 0
        assert result.stdout == "sent:PING recv:PONG\n"

    async def test_python_localhost_dns_resolves(self, scheduler: Scheduler) -> None:
        """socket.getaddrinfo('localhost', ...) resolves to 127.0.0.1 only (AF_INET)."""
        code = """\
import socket
results = socket.getaddrinfo('localhost', 80, socket.AF_INET, socket.SOCK_STREAM)
addrs = [r[4][0] for r in results]
print('resolved:' + ','.join(addrs))
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "resolved:127.0.0.1\n"

    async def test_javascript_localhost_string_resolves(self, scheduler: Scheduler) -> None:
        """fetch('http://localhost:...') works — validates /etc/hosts config."""
        result = await run_once(
            scheduler, VmSpec(language=Language.JAVASCRIPT), _js_http_fetch_code(fetch_host="localhost")
        )
        assert result.exit_code == 0
        assert result.stdout == "status:200 body:HELLO_FROM_SERVER\n"

    async def test_python_sequential_bind_reuse(self, scheduler: Scheduler) -> None:
        """Bind, close, re-bind same port with SO_REUSEADDR — port reuse on loopback."""
        code = """\
import socket

s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s1.bind(('127.0.0.1', 0))
port = s1.getsockname()[1]
s1.close()

s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s2.bind(('127.0.0.1', port))
s2.close()
print("reuse_ok")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "reuse_ok\n"


# =============================================================================
# Weird: Unusual but valid loopback patterns
# =============================================================================
class TestLoopbackWeirdCases:
    """Unusual but valid networking patterns on loopback."""

    async def test_python_udp_loopback(self, scheduler: Scheduler) -> None:
        """UDP send/recv on 127.0.0.1 works (not just TCP)."""
        code = """\
import socket

recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.settimeout(5)
recv_sock.bind(('127.0.0.1', 0))
port = recv_sock.getsockname()[1]

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_sock.sendto(b'UDP_PING', ('127.0.0.1', port))

data, addr = recv_sock.recvfrom(1024)
recv_sock.close()
send_sock.close()
print(f"udp:{data.decode()} from:{addr[0]}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "udp:UDP_PING from:127.0.0.1\n"

    async def test_python_multiple_servers_different_ports(self, scheduler: Scheduler) -> None:
        """Two servers on loopback, different ports, both reachable."""
        code = """\
import asyncio

async def main():
    def replying(reply):
        async def handle(reader, writer):
            writer.write(reply)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        return handle

    servers = [await asyncio.start_server(replying(reply), '127.0.0.1', 0) for reply in (b'SERVER_A', b'SERVER_B')]
    replies = []
    for server in servers:
        reader, writer = await asyncio.open_connection('127.0.0.1', server.sockets[0].getsockname()[1])
        replies.append((await reader.read(1024)).decode())
        writer.close()
        await writer.wait_closed()
    for server in servers:
        server.close()
        await server.wait_closed()
    print(','.join(replies))

asyncio.run(main())
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        assert result.stdout == "SERVER_A,SERVER_B\n"

    async def test_javascript_bun_tcp_loopback(self, scheduler: Scheduler) -> None:
        """Bun.listen + Bun.connect TCP round-trip on 127.0.0.1 (Bun-native API)."""
        code = """\
const server = Bun.listen({
  hostname: '127.0.0.1',
  port: 0,
  socket: {
    data(socket, data) {
      socket.write('BUN_TCP_PONG');
      socket.end();
    },
  },
});
const port = server.port;
const resp = await Promise.race([
  new Promise((resolve, reject) => {
    Bun.connect({
      hostname: '127.0.0.1',
      port,
      socket: {
        data(socket, data) {
          resolve(new TextDecoder().decode(data));
          socket.end();
        },
        open(socket) { socket.write('PING'); },
        error(socket, err) { reject(err); },
        close() {},
      },
    });
  }),
  new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 5000)),
]);
server.stop();
console.log('bun_tcp:' + resp);
"""
        result = await run_once(scheduler, VmSpec(language=Language.JAVASCRIPT), code)
        assert result.exit_code == 0
        assert result.stdout == "bun_tcp:BUN_TCP_PONG\n"


# =============================================================================
# Out of bounds: Privilege and resource limits on loopback
# =============================================================================
class TestLoopbackOutOfBounds:
    """Privilege and resource constraints apply to loopback too."""

    async def test_python_privileged_port_rejected(self, scheduler: Scheduler) -> None:
        """UID 1000 cannot bind to port 80 on loopback: EACCES (no CAP_NET_BIND_SERVICE)."""
        code = """\
import socket, errno
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 80))
    s.close()
    print("unexpected_success")
except OSError as e:
    print(f"blocked:{errno.errorcode[e.errno]}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert (result.exit_code, result.stdout) == (0, "blocked:EACCES\n"), result.stderr

    async def test_python_ephemeral_port_bind(self, scheduler: Scheduler) -> None:
        """Python binds 127.0.0.1 port 0 and gets a port of the kernel's ephemeral range
        (net.ipv4.ip_local_port_range, default 32768-60999)."""
        code = """\
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 0))
port = s.getsockname()[1]
s.close()
print(f"ephemeral:{port}")
"""
        result = await run_once(scheduler, VmSpec(language=Language.PYTHON), code)
        assert result.exit_code == 0
        port = re.fullmatch(r"ephemeral:(\d+)\n", result.stdout)
        assert port, result.stdout
        assert 32768 <= int(port[1]) <= 60999, result.stdout

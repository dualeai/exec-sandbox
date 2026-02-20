"""Tests for loopback interface (lo) in the guest VM.

The guest-agent brings up `lo` via `ip link set lo up` in setup_init_environment().
Without this, all localhost/127.0.0.1 connections fail inside the microvm — breaking
http.createServer + fetch('http://localhost:...') in Bun, Python socket servers, etc.

Test categories:
- Normal: loopback interface is UP, TCP round-trip works on all backends
- Edge: loopback works without network, address variants, port reuse
- Weird: unusual but valid patterns (UDP, multiple servers, Bun-native API)
- Out of bounds: privileged ports rejected, ephemeral port allocation
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


def _py_tcp_server_code(response: str, bind: str = "127.0.0.1") -> str:
    """Self-contained asyncio TCP echo-server that prints the response payload."""
    return f"""\
import asyncio

async def main():
    async def handle(reader, writer):
        writer.write(b'{response}')
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, '{bind}', 0)
    port = server.sockets[0].getsockname()[1]

    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    resp = await reader.read(1024)
    writer.close()
    await writer.wait_closed()
    server.close()
    await server.wait_closed()
    print(resp.decode())

asyncio.run(main())
"""


def _js_http_fetch_code(response: str, fetch_host: str = "127.0.0.1") -> str:
    """Self-contained node:http server + fetch that prints the response body."""
    return f"""\
const http = require('http');
const server = http.createServer((req, res) => {{
  res.writeHead(200);
  res.end('{response}');
}});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;
const res = await fetch('http://{fetch_host}:' + port);
const body = await res.text();
server.close();
console.log(body);
"""


# =============================================================================
# Normal: Loopback interface exists and is UP
# =============================================================================
class TestLoopbackInterfaceUp:
    """Loopback interface is configured and operational."""

    async def test_loopback_up_shell(self, scheduler: Scheduler) -> None:
        """ip addr show lo reports state UP and 127.0.0.1."""
        result = await scheduler.run(
            code="ip addr show lo",
            language=Language.RAW,
        )
        assert result.exit_code == 0
        assert "UP" in result.stdout
        assert "127.0.0.1" in result.stdout

    async def test_loopback_up_python(self, scheduler: Scheduler) -> None:
        """Python can bind a socket to 127.0.0.1."""
        code = """\
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 0))
addr = s.getsockname()
s.close()
print(f"bound:{addr[0]}:{addr[1]}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "bound:127.0.0.1:" in result.stdout

    async def test_loopback_up_javascript(self, scheduler: Scheduler) -> None:
        """node:http server binds to 127.0.0.1 successfully."""
        code = """\
const http = require('http');
const server = http.createServer((req, res) => { res.end('ok'); });
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const addr = server.address();
console.log('bound:' + addr.address + ':' + addr.port);
server.close();
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0
        assert "bound:127.0.0.1:" in result.stdout


# =============================================================================
# Normal: TCP round-trip on loopback (bind + connect + send/recv)
# =============================================================================
class TestLoopbackTcpRoundtrip:
    """Bind, connect, and exchange data over loopback."""

    async def test_python_tcp_loopback(self, scheduler: Scheduler) -> None:
        """Python TCP server + client on 127.0.0.1 exchange PING/PONG."""
        code = """\
import asyncio

async def main():
    async def handle(reader, writer):
        await reader.read(1024)
        writer.write(b'PONG')
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]

    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    writer.write(b'PING')
    await writer.drain()
    resp = await reader.read(1024)
    writer.close()
    await writer.wait_closed()
    server.close()
    await server.wait_closed()
    print(f"sent:PING recv:{resp.decode()}")

asyncio.run(main())
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "sent:PING recv:PONG" in result.stdout

    async def test_javascript_http_server_fetch(self, scheduler: Scheduler) -> None:
        """node:http createServer + fetch on 127.0.0.1 — the original bug scenario."""
        code = """\
const http = require('http');
const server = http.createServer((req, res) => {
  res.writeHead(200);
  res.end('HELLO_FROM_SERVER');
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;
const res = await fetch('http://127.0.0.1:' + port);
const body = await res.text();
server.close();
console.log('status:' + res.status + ' body:' + body);
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0
        assert "status:200" in result.stdout
        assert "body:HELLO_FROM_SERVER" in result.stdout

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
        result = await scheduler.run(code=code, language=Language.RAW)
        assert result.exit_code == 0
        assert "SHELL_PING" in result.stdout


# =============================================================================
# Edge: Loopback works even with allow_network=False
# =============================================================================
class TestLoopbackWithoutNetwork:
    """Loopback operates independently of external networking."""

    async def test_python_loopback_no_network(self, scheduler: Scheduler) -> None:
        """TCP loopback works with allow_network=False (no gvproxy, no eth0)."""
        result = await scheduler.run(
            code=_py_tcp_server_code("NO_NET_OK"),
            language=Language.PYTHON,
            allow_network=False,
        )
        assert result.exit_code == 0
        assert "NO_NET_OK" in result.stdout

    async def test_javascript_loopback_no_network(self, scheduler: Scheduler) -> None:
        """fetch('http://127.0.0.1:...') works with allow_network=False."""
        result = await scheduler.run(
            code=_js_http_fetch_code("NO_NET_JS"),
            language=Language.JAVASCRIPT,
            allow_network=False,
        )
        assert result.exit_code == 0
        assert "NO_NET_JS" in result.stdout


# =============================================================================
# Edge: Address variants and binding patterns
# =============================================================================
class TestLoopbackEdgeCases:
    """Address variants and common real-world binding patterns."""

    async def test_python_bind_0000_connect_localhost(self, scheduler: Scheduler) -> None:
        """Bind 0.0.0.0, connect via 127.0.0.1 — common server pattern."""
        result = await scheduler.run(
            code=_py_tcp_server_code("WILDCARD_OK", bind="0.0.0.0"),
            language=Language.PYTHON,
        )
        assert result.exit_code == 0
        assert "WILDCARD_OK" in result.stdout

    async def test_python_localhost_dns_resolves(self, scheduler: Scheduler) -> None:
        """socket.getaddrinfo('localhost', ...) resolves to 127.0.0.1."""
        code = """\
import socket
results = socket.getaddrinfo('localhost', 80, socket.AF_INET, socket.SOCK_STREAM)
addrs = [r[4][0] for r in results]
print('resolved:' + ','.join(addrs))
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "127.0.0.1" in result.stdout

    async def test_javascript_localhost_string_resolves(self, scheduler: Scheduler) -> None:
        """fetch('http://localhost:...') works — validates /etc/hosts config."""
        result = await scheduler.run(
            code=_js_http_fetch_code("LOCALHOST_STR", fetch_host="localhost"),
            language=Language.JAVASCRIPT,
        )
        assert result.exit_code == 0
        assert "LOCALHOST_STR" in result.stdout

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
print(f"reuse_ok:{port}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "reuse_ok:" in result.stdout


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
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "udp:UDP_PING" in result.stdout
        assert "from:127.0.0.1" in result.stdout

    async def test_python_multiple_servers_different_ports(self, scheduler: Scheduler) -> None:
        """Two servers on loopback, different ports, both reachable."""
        code = """\
import asyncio

async def main():
    async def handle_a(reader, writer):
        writer.write(b'SERVER_A')
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def handle_b(reader, writer):
        writer.write(b'SERVER_B')
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    srv1 = await asyncio.start_server(handle_a, '127.0.0.1', 0)
    srv2 = await asyncio.start_server(handle_b, '127.0.0.1', 0)
    port1 = srv1.sockets[0].getsockname()[1]
    port2 = srv2.sockets[0].getsockname()[1]

    r1, w1 = await asyncio.open_connection('127.0.0.1', port1)
    resp1 = await r1.read(1024)
    w1.close()
    await w1.wait_closed()

    r2, w2 = await asyncio.open_connection('127.0.0.1', port2)
    resp2 = await r2.read(1024)
    w2.close()
    await w2.wait_closed()

    srv1.close()
    srv2.close()
    await srv1.wait_closed()
    await srv2.wait_closed()
    print(f"{resp1.decode()},{resp2.decode()}")

asyncio.run(main())
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "SERVER_A" in result.stdout
        assert "SERVER_B" in result.stdout

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
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0
        assert "bun_tcp:BUN_TCP_PONG" in result.stdout


# =============================================================================
# Out of bounds: Privilege and resource limits on loopback
# =============================================================================
class TestLoopbackOutOfBounds:
    """Privilege and resource constraints apply to loopback too."""

    async def test_python_privileged_port_rejected(self, scheduler: Scheduler) -> None:
        """UID 1000 cannot bind to port 80 on loopback (PermissionError/OSError)."""
        code = """\
import socket, errno
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 80))
    s.close()
    print("unexpected_success")
except OSError as e:
    if e.errno in (errno.EACCES, errno.EPERM):
        print("blocked")
    else:
        print(f"unexpected_error:{e.errno}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "blocked" in result.stdout

    async def test_python_ephemeral_port_bind(self, scheduler: Scheduler) -> None:
        """Bind to port 0 (OS-assigned) works on loopback."""
        code = """\
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 0))
port = s.getsockname()[1]
s.close()
print(f"ephemeral:{port}")
"""
        result = await scheduler.run(code=code, language=Language.PYTHON)
        assert result.exit_code == 0
        assert "ephemeral:" in result.stdout
        # Ephemeral ports are OS-assigned from the unprivileged range (> 1024)
        port_str = result.stdout.strip().split("ephemeral:")[1].split()[0]
        assert int(port_str) > 1024

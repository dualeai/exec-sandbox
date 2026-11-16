"""E2E tests for outbound filtering enforcement.

Tests that gvproxy-wrapper OutboundAllow filtering actually works in VMs:
1. Normal: allowed domains connect, blocked domains fail
2. Language defaults (pypi.org for Python)
3. Edge cases: subdomains, exact match, parent/sibling blocked
4. Security: direct IP bypass, plain HTTP blocked, raw TCP blocked
5. DNS behavior: blocked domains fail DNS, allowed domains resolve to real IPs
6. Boundary: large allowlists, IDN domains
"""

import re

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

from .conftest import (
    JS_SAFETY,
    NET_CONNECT_TIMEOUT_S,
    NET_OP_TIMEOUT_S,
    NET_RETRY_BACKOFF_S,
    NET_RETRY_COUNT,
    NET_SAFETY_TIMEOUT_S,
    PY_SAFETY,
)

# Expected stdout of get_connection_test_code: the domain filter blocks a name at DNS (gaierror); the
# TLS filter closes a connection whose SNI does not match (an IP literal has no domain to match).
CONNECTED = "CONNECTED"
DNS_BLOCKED = "BLOCKED:gaierror"
TLS_BLOCKED = "BLOCKED:SSLEOFError"

# =============================================================================
# Normal cases: Basic allow/block behavior
# =============================================================================
OUTBOUND_FILTER_NORMAL_CASES = [
    # Explicitly allowed domain should connect
    pytest.param(["cp.cloudflare.com"], "cp.cloudflare.com", CONNECTED, id="normal-allowed-connects"),
    # Non-allowed domain should be blocked
    pytest.param(["cp.cloudflare.com"], "google.com", DNS_BLOCKED, id="normal-blocked-fails"),
    # Multiple allowed domains
    pytest.param(
        ["google.com", "pypi.org", "cp.cloudflare.com"],
        "cp.cloudflare.com",
        CONNECTED,
        id="normal-multiple-domains-third-connects",
    ),
    # Empty allowed_domains = no filtering (all allowed)
    pytest.param([], "google.com", CONNECTED, id="normal-empty-allowlist-permits-all"),
]

# =============================================================================
# Language defaults: Auto-included package registries
# =============================================================================
OUTBOUND_FILTER_DEFAULTS_CASES = [
    # Python defaults (allowed_domains=None): pypi.org should work
    pytest.param(None, "pypi.org", CONNECTED, id="defaults-python-pypi-connects"),
    # Python defaults: files.pythonhosted.org should work
    pytest.param(None, "files.pythonhosted.org", CONNECTED, id="defaults-python-pythonhosted-connects"),
    # Python defaults: google.com exists but is not in the Python defaults
    pytest.param(None, "google.com", DNS_BLOCKED, id="defaults-python-blocks-others"),
]

# =============================================================================
# Edge cases: Subdomains, deep nesting, special formats
# =============================================================================
OUTBOUND_FILTER_EDGE_CASES = [
    # Subdomain of allowed domain should connect
    pytest.param(["pythonhosted.org"], "files.pythonhosted.org", CONNECTED, id="edge-subdomain-of-allowed-connects"),
    # Parent domain NOT allowed when only subdomain specified
    pytest.param(
        ["sub.cp.cloudflare.com"], "cp.cloudflare.com", DNS_BLOCKED, id="edge-parent-blocked-when-subdomain-allowed"
    ),
    # Sibling subdomain NOT allowed (both names exist: without the filter, test.pypi.org connects)
    pytest.param(["upload.pypi.org"], "test.pypi.org", DNS_BLOCKED, id="edge-sibling-subdomain-blocked"),
    # Exact domain match (not subdomain) — domain itself matches the pattern
    pytest.param(["pypi.org"], "pypi.org", CONNECTED, id="edge-exact-domain-match-connects"),
    # Internationalized domain name in punycode (münchen.de): the allowlist matches the ASCII form
    pytest.param(["xn--mnchen-3ya.de"], "xn--mnchen-3ya.de", CONNECTED, id="edge-idn-punycode-connects"),
]

# =============================================================================
# Security cases: Bypass attempts
# =============================================================================
# NOTE: OutboundAllow filters at both DNS and TLS levels.
# Direct IP connections bypass DNS but still get blocked (no matching pattern).
# This is stronger than DNS-only filtering which could be bypassed with IP literals.
OUTBOUND_FILTER_SECURITY_CASES = [
    # Direct IP connection without a DNS lookup: Python sends no SNI for an IP literal
    # (RFC 6066), so the TLS filter has no name to match and closes the connection.
    pytest.param(["cp.cloudflare.com"], "93.184.216.34", TLS_BLOCKED, id="security-direct-ip-no-dns-blocked"),
]

# Combine all parametrized test cases
OUTBOUND_FILTER_TEST_CASES = (
    OUTBOUND_FILTER_NORMAL_CASES
    + OUTBOUND_FILTER_DEFAULTS_CASES
    + OUTBOUND_FILTER_EDGE_CASES
    + OUTBOUND_FILTER_SECURITY_CASES
)


def get_connection_test_code(test_domain: str, *, retries: bool) -> str:
    """Python code that tries a TLS connection to port 443 of `test_domain`.

    It prints CONNECTED, or BLOCKED:<exception type> (BLOCKED:Timeout from PY_SAFETY when the
    process hangs). When *retries* is True the code retries transient failures (``NET_RETRY_COUNT``
    more attempts, linear backoff): used for allowed-domain tests.
    """
    attempts = NET_RETRY_COUNT + 1 if retries else 1
    return (
        PY_SAFETY
        + f"""
import socket, ssl, time
for attempt in range({attempts}):
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection(("{test_domain}", 443), timeout={NET_CONNECT_TIMEOUT_S}) as sock:
            with ctx.wrap_socket(sock, server_hostname="{test_domain}") as ssock:
                print("CONNECTED")
                break
    except Exception as e:
        if attempt < {attempts - 1}:
            time.sleep({NET_RETRY_BACKOFF_S} * (attempt + 1))
            continue
        print(f"BLOCKED:{{type(e).__name__}}")
"""
    )


@pytest.mark.parametrize("allowed_domains,test_domain,expected", OUTBOUND_FILTER_TEST_CASES)
async def test_outbound_filtering(
    scheduler: Scheduler,
    allowed_domains: list[str] | None,
    test_domain: str,
    expected: str,
) -> None:
    """Python TLS connection to port 443 through the outbound filter.

    OutboundAllow behavior:
    - Allowed domains: DNS resolves, TLS connections succeed (CONNECTED)
    - Blocked domains: DNS resolution fails (gaierror)
    - IP literals: the TLS filter closes the connection (SSLEOFError)
    """
    code = get_connection_test_code(test_domain, retries=expected == CONNECTED)

    result = await run_once(
        scheduler, VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=allowed_domains), code
    )

    assert result.stdout == f"{expected}\n", (
        f"Expected {expected} for a TLS connection to {test_domain}.\n"
        f"allowed_domains={allowed_domains}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


# =============================================================================
# HTTPS allowed (urllib, higher-level than raw socket); a blocked domain: test_outbound_filtering
# =============================================================================
async def test_outbound_filtering_http_allowed(scheduler: Scheduler) -> None:
    """urllib HTTPS GET of an allowed domain gets status 200, and nothing else is printed."""
    code = (
        PY_SAFETY
        + f"""
import urllib.request
try:
    with urllib.request.urlopen("https://pypi.org/simple/", timeout={NET_OP_TIMEOUT_S}) as r:
        print(f"STATUS:{{r.status}}")
except Exception as e:
    print(f"ERROR:{{e}}")
"""
    )

    result = await run_once(
        scheduler, VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=["pypi.org"]), code
    )

    assert result.stdout == "STATUS:200\n", (
        f"HTTPS to allowed domain failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# Network disabled (no gvproxy at all)
# =============================================================================
async def test_network_disabled_no_resolution(scheduler: Scheduler) -> None:
    """With allow_network=False, name resolution fails (the VM has no DNS server to reach)."""
    code = (
        PY_SAFETY
        + """
import socket
try:
    ip = socket.gethostbyname("google.com")
    print(f"RESOLVED:{ip}")
except OSError as e:
    print(f"NO_NETWORK:{type(e).__name__}")
"""
    )

    result = await run_once(
        scheduler,
        VmSpec(
            language=Language.PYTHON,
            allow_network=False,  # Network disabled
        ),
        code,
    )

    assert result.stdout == "NO_NETWORK:gaierror\n", (
        f"Expected network to be disabled.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# RAW language tests (using curl instead of Python)
# =============================================================================
async def test_outbound_filtering_raw_allowed(scheduler: Scheduler) -> None:
    """curl (RAW language) to an allowed domain succeeds: the command prints SUCCESS only.

    Uses a planet-scale connectivity-check (NCSI) endpoint — no rate limits,
    no bot challenges, no scheduled downtime. No ``-f``: any HTTP response
    proves the path works end-to-end (DNS, SNI filter, TLS).
    """
    result = await run_once(
        scheduler,
        VmSpec(language=Language.RAW, allow_network=True, allowed_domains=["cp.cloudflare.com"]),
        f"timeout {NET_SAFETY_TIMEOUT_S} curl -s"
        f" --connect-timeout {NET_CONNECT_TIMEOUT_S} --max-time {NET_OP_TIMEOUT_S}"
        f" --retry {NET_RETRY_COUNT} --retry-connrefused -o /dev/null"
        " https://cp.cloudflare.com/generate_204 && echo 'SUCCESS' || echo 'FAILED'",
    )

    assert result.stdout == "SUCCESS\n", (
        f"curl to allowed domain failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def test_outbound_filtering_raw_blocked(scheduler: Scheduler) -> None:
    """curl to a blocked domain fails at DNS: exit code 6 (could not resolve host)."""
    result = await run_once(
        scheduler,
        VmSpec(
            language=Language.RAW,
            allow_network=True,
            allowed_domains=["cp.cloudflare.com"],  # google.com not allowed
        ),
        f"timeout {NET_SAFETY_TIMEOUT_S} curl -sf --connect-timeout {NET_CONNECT_TIMEOUT_S}"
        f" --max-time {NET_OP_TIMEOUT_S} https://google.com/ && echo 'SUCCESS' || echo \"BLOCKED:$?\"",
    )

    assert result.stdout == "BLOCKED:6\n", (
        f"Expected curl to google.com to fail at DNS.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# JavaScript tests
# =============================================================================
async def test_outbound_filtering_javascript_blocked(scheduler: Scheduler) -> None:
    """Bun fetch to a blocked domain fails at DNS (error code ENOTFOUND)."""
    code = (
        JS_SAFETY
        + f"""
try {{
    const res = await fetch("https://google.com/", {{ signal: AbortSignal.timeout({NET_OP_TIMEOUT_S * 1000}) }});
    console.log("STATUS:" + res.status);
}} catch (e) {{
    console.log("BLOCKED:" + e.code);
}}
"""
    )

    result = await run_once(
        scheduler,
        VmSpec(
            language=Language.JAVASCRIPT,
            allow_network=True,
            allowed_domains=["registry.npmjs.org"],  # google.com not allowed
        ),
        code,
    )

    assert result.stdout == "BLOCKED:ENOTFOUND\n", (
        f"Expected fetch to google.com to fail at DNS.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# Security: SNI bypass attempts
# =============================================================================
async def test_outbound_filtering_plain_http_blocked(scheduler: Scheduler) -> None:
    """Plain HTTP (port 80) is refused when OutboundAllow is active.

    OutboundAllow only allows TLS connections whose SNI matches a pattern.
    Plain HTTP has no TLS/SNI, so it must be blocked even for allowed domains.
    """
    code = (
        PY_SAFETY
        + f"""
import urllib.request
try:
    with urllib.request.urlopen("http://cp.cloudflare.com/", timeout={NET_OP_TIMEOUT_S}) as r:
        print(f"STATUS:{{r.status}}")
except Exception as e:
    print(f"BLOCKED:{{type(getattr(e, 'reason', e)).__name__}}")
"""
    )

    result = await run_once(
        scheduler,
        VmSpec(
            language=Language.PYTHON,
            allow_network=True,
            allowed_domains=["cp.cloudflare.com"],  # Allowed, but only on TLS
        ),
        code,
    )

    assert result.stdout == "BLOCKED:ConnectionRefusedError\n", (
        f"Expected plain HTTP to be refused by OutboundAllow.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def test_outbound_filtering_dns_resolve_then_ip_connect_with_sni(
    scheduler: Scheduler,
) -> None:
    """DNS-resolve allowed domain, then TLS-connect to the IP with correct SNI.

    The proxy resolves the SNI on the host and checks that the destination IP is one of its
    addresses: resolve allowed domain → get IP → TLS to that IP with matching SNI works.
    The TLS handshake (certificate check included) succeeds only on a real IP of the domain, not a sinkhole.
    """
    code = (
        PY_SAFETY
        + f"""
import socket, ssl
try:
    ip = socket.gethostbyname("cp.cloudflare.com")
    print(f"RESOLVED:{{ip}}")
    ctx = ssl.create_default_context()
    with socket.create_connection((ip, 443), timeout={NET_CONNECT_TIMEOUT_S}) as sock:
        with ctx.wrap_socket(sock, server_hostname="cp.cloudflare.com") as ssock:
            print("CONNECTED")
except Exception as e:
    print(f"BLOCKED:{{type(e).__name__}}")
"""
    )

    result = await run_once(
        scheduler, VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=["cp.cloudflare.com"]), code
    )

    # The IP is the one the guest resolved (it varies): an IPv4 address, then the connection
    assert re.fullmatch(r"RESOLVED:\d{1,3}(\.\d{1,3}){3}\nCONNECTED\n", result.stdout), (
        f"TLS to resolved IP with correct SNI should succeed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def test_outbound_filtering_dns_resolve_then_ip_connect_no_sni(
    scheduler: Scheduler,
) -> None:
    """DNS-resolve allowed domain, then raw TCP to the IP without TLS/SNI.

    Even after resolving an allowed domain, a raw TCP connection (no TLS,
    no SNI) to the resulting IP is refused. The proxy requires TLS
    with matching SNI on port 443: raw TCP to another port has no SNI to match.
    """
    code = (
        PY_SAFETY
        + f"""
import socket
try:
    ip = socket.gethostbyname("cp.cloudflare.com")
    print("RESOLVED")
    # Raw TCP to resolved IP on port 80 — no TLS, no SNI
    sock = socket.create_connection((ip, 80), timeout={NET_CONNECT_TIMEOUT_S})
    sock.sendall(b"GET / HTTP/1.1\\r\\nHost: cp.cloudflare.com\\r\\n\\r\\n")
    data = sock.recv(1024)
    sock.close()
    print(f"TCP_CONNECTED:{{len(data)}}")
except Exception as e:
    print(f"BLOCKED:{{type(e).__name__}}")
"""
    )

    result = await run_once(
        scheduler, VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=["cp.cloudflare.com"]), code
    )

    assert result.stdout == "RESOLVED\nBLOCKED:ConnectionRefusedError\n", (
        f"Raw TCP to resolved IP (no TLS/SNI) should be refused.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def test_outbound_filtering_dns_resolve_then_ip_connect_wrong_sni(
    scheduler: Scheduler,
) -> None:
    """DNS-resolve allowed domain, then TLS-connect to its IP with a DIFFERENT SNI.

    SNI spoofing attempt: resolve cp.cloudflare.com → IP, then connect to that IP
    but claim to be pypi.org in the TLS SNI. pypi.org is not in the allowlist, so the
    TLS filter closes the connection (SSLEOFError).
    """
    code = (
        PY_SAFETY
        + f"""
import socket, ssl
try:
    ip = socket.gethostbyname("cp.cloudflare.com")
    print("RESOLVED")
    ctx = ssl.create_default_context()
    # Connect to cp.cloudflare.com's IP but set SNI to pypi.org (not allowed domain)
    with socket.create_connection((ip, 443), timeout={NET_CONNECT_TIMEOUT_S}) as sock:
        with ctx.wrap_socket(sock, server_hostname="pypi.org") as ssock:
            print("CONNECTED")
except Exception as e:
    print(f"BLOCKED:{{type(e).__name__}}")
"""
    )

    result = await run_once(
        scheduler,
        VmSpec(
            language=Language.PYTHON,
            allow_network=True,
            allowed_domains=["cp.cloudflare.com"],  # pypi.org NOT in allowlist
        ),
        code,
    )

    assert result.stdout == f"RESOLVED\n{TLS_BLOCKED}\n", (
        f"TLS with mismatched SNI (pypi.org) to cp.cloudflare.com IP should be blocked.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def test_outbound_filtering_sni_spoof_allowed_domain_wrong_ip(
    scheduler: Scheduler,
) -> None:
    """TLS-connect to a non-allowed IP while spoofing an allowed domain as SNI.

    The proxy's DNS cross-check should catch this: the SNI says "cp.cloudflare.com"
    but the destination IP (8.8.8.8) doesn't match cp.cloudflare.com's DNS records.
    (8.8.8.8 — a non-Cloudflare anycast IP — so the SNI's provider can never
    legitimately serve it.) The TLS filter closes the connection (SSLEOFError).
    """
    code = (
        PY_SAFETY
        + f"""
import socket, ssl
try:
    ctx = ssl.create_default_context()
    # Connect to Google DNS IP but claim to be cp.cloudflare.com
    with socket.create_connection(("8.8.8.8", 443), timeout={NET_CONNECT_TIMEOUT_S}) as sock:
        with ctx.wrap_socket(sock, server_hostname="cp.cloudflare.com") as ssock:
            print("CONNECTED")
except Exception as e:
    print(f"BLOCKED:{{type(e).__name__}}")
"""
    )

    result = await run_once(
        scheduler, VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=["cp.cloudflare.com"]), code
    )

    assert result.stdout == f"{TLS_BLOCKED}\n", (
        f"SNI spoofing (allowed domain to wrong IP) should be blocked.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# Boundary / weird input tests
# =============================================================================
async def test_outbound_filtering_many_domains(scheduler: Scheduler) -> None:
    """Test with many allowed domains (stress test pattern count)."""
    # 50 domains in allowlist
    many_domains = [f"domain{i}.com" for i in range(50)]
    many_domains.append("cp.cloudflare.com")  # Add a real one

    result = await run_once(
        scheduler,
        VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=many_domains),
        get_connection_test_code("cp.cloudflare.com", retries=True),
    )

    assert result.stdout == "CONNECTED\n", f"Large allowlist failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"

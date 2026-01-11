"""E2E tests for DNS filtering enforcement.

Tests that gvproxy-wrapper DNS filtering actually works in VMs:
1. Allowed domains resolve and are accessible
2. Blocked domains return NXDOMAIN / connection fails
3. Language defaults (pypi.org for Python, npmjs.org for JS)
4. Edge cases: subdomains, case sensitivity, IP bypass attempts
"""

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# =============================================================================
# Normal cases: Basic allow/block behavior
# =============================================================================
DNS_FILTER_NORMAL_CASES = [
    # Explicitly allowed domain should resolve
    pytest.param(
        Language.PYTHON,
        ["example.com"],
        "example.com",
        True,
        id="normal-allowed-resolves",
    ),
    # Non-allowed domain should be blocked
    pytest.param(
        Language.PYTHON,
        ["example.com"],
        "google.com",
        False,
        id="normal-blocked-fails",
    ),
    # Multiple allowed domains
    pytest.param(
        Language.PYTHON,
        ["example.com", "google.com", "github.com"],
        "github.com",
        True,
        id="normal-multiple-domains-third-resolves",
    ),
    # Empty allowed_domains = no filtering (all allowed)
    pytest.param(
        Language.PYTHON,
        [],  # Empty list = allow all
        "google.com",
        True,
        id="normal-empty-allowlist-permits-all",
    ),
]

# =============================================================================
# Language defaults: Auto-included package registries
# =============================================================================
DNS_FILTER_DEFAULTS_CASES = [
    # Python defaults: pypi.org should work
    pytest.param(
        Language.PYTHON,
        None,  # Use language defaults
        "pypi.org",
        True,
        id="defaults-python-pypi-resolves",
    ),
    # Python defaults: files.pythonhosted.org should work
    pytest.param(
        Language.PYTHON,
        None,
        "files.pythonhosted.org",
        True,
        id="defaults-python-pythonhosted-resolves",
    ),
    # Python defaults: non-pypi domain should be blocked
    pytest.param(
        Language.PYTHON,
        None,
        "google.com",  # Exists but not in Python defaults, so should be blocked
        False,
        id="defaults-python-blocks-others",
    ),
    # JavaScript defaults: npmjs.org should work
    pytest.param(
        Language.JAVASCRIPT,
        None,
        "registry.npmjs.org",
        True,
        id="defaults-javascript-npm-resolves",
    ),
]

# =============================================================================
# Edge cases: Subdomains, case sensitivity, special formats
# =============================================================================
DNS_FILTER_EDGE_CASES = [
    # Subdomain of allowed domain should resolve
    pytest.param(
        Language.PYTHON,
        ["pythonhosted.org"],
        "files.pythonhosted.org",
        True,
        id="edge-subdomain-of-allowed-resolves",
    ),
    # NOTE: Deep subdomain test removed - DNS filter may not resolve
    # nonexistent domains even if parent is allowed
    # Parent domain NOT allowed when only subdomain specified
    pytest.param(
        Language.PYTHON,
        ["sub.example.com"],
        "example.com",
        False,
        id="edge-parent-blocked-when-subdomain-allowed",
    ),
    # Sibling subdomain NOT allowed
    pytest.param(
        Language.PYTHON,
        ["api.example.com"],
        "www.example.com",
        False,
        id="edge-sibling-subdomain-blocked",
    ),
]

# =============================================================================
# Security cases: Bypass attempts
# =============================================================================
# NOTE: gethostbyname("1.1.1.1") doesn't do DNS lookup - it validates and returns
# the IP directly. DNS filtering cannot block direct IP literals.
# Network firewall rules would be needed to block IP connections.
DNS_FILTER_SECURITY_CASES = [
    # Localhost should be blocked unless explicitly allowed
    pytest.param(
        Language.PYTHON,
        ["example.com"],
        "localhost",
        False,
        id="security-localhost-blocked",
    ),
]

# Combine all test cases
DNS_FILTER_TEST_CASES = (
    DNS_FILTER_NORMAL_CASES + DNS_FILTER_DEFAULTS_CASES + DNS_FILTER_EDGE_CASES + DNS_FILTER_SECURITY_CASES
)


def get_dns_test_code(language: Language, test_domain: str) -> str:
    """Generate language-appropriate DNS test code."""
    if language == Language.PYTHON:
        return f"""
import socket
try:
    ip = socket.gethostbyname("{test_domain}")
    print(f"RESOLVED:{{ip}}")
except socket.gaierror as e:
    print(f"BLOCKED:{{e}}")
"""
    if language == Language.JAVASCRIPT:
        # Use Bun's DNS resolver
        return f"""
const dns = require('dns');
const {{ promisify }} = require('util');
const lookup = promisify(dns.lookup);
(async () => {{
    try {{
        const result = await lookup("{test_domain}");
        console.log("RESOLVED:" + result.address);
    }} catch (e) {{
        console.log("BLOCKED:" + e.message);
    }}
}})();
"""
    # RAW
    return f'getent hosts "{test_domain}" && echo "RESOLVED" || echo "BLOCKED"'


@pytest.mark.parametrize(
    "language,allowed_domains,test_domain,should_resolve",
    DNS_FILTER_TEST_CASES,
)
async def test_dns_filtering(
    scheduler: Scheduler,
    language: Language,
    allowed_domains: list[str] | None,
    test_domain: str,
    should_resolve: bool,
) -> None:
    """Test DNS filtering enforcement in VM using native socket.

    DNS proxy behavior:
    - Allowed domains: resolve to real IP
    - Blocked domains: always return 0.0.0.0 (sinkhole)
    """
    code = get_dns_test_code(language, test_domain)

    result = await scheduler.run(
        code=code,
        language=language,
        allow_network=True,
        allowed_domains=allowed_domains,
    )

    if should_resolve:
        # Domain should resolve to a real IP (not 0.0.0.0)
        assert "RESOLVED:" in result.stdout, (
            f"Expected {test_domain} to resolve but got blocked.\n"
            f"allowed_domains={allowed_domains}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        # Verify it's not the sinkhole IP
        assert "0.0.0.0" not in result.stdout, (
            f"Expected {test_domain} to resolve to real IP but got sinkhole.\n"
            f"allowed_domains={allowed_domains}\n"
            f"stdout: {result.stdout}"
        )
    else:
        # Blocked domains always return 0.0.0.0 (DNS sinkhole)
        assert "RESOLVED:0.0.0.0" in result.stdout, (
            f"Expected {test_domain} to be blocked (0.0.0.0) but got different result.\n"
            f"allowed_domains={allowed_domains}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


async def test_dns_filtering_http_allowed(scheduler: Scheduler) -> None:
    """Test that allowed domain is accessible via HTTP."""
    code = """
import urllib.request
try:
    with urllib.request.urlopen("https://pypi.org/simple/", timeout=10) as r:
        print(f"STATUS:{r.status}")
except Exception as e:
    print(f"ERROR:{e}")
"""

    result = await scheduler.run(
        code=code,
        language=Language.PYTHON,
        allow_network=True,
        allowed_domains=["pypi.org"],
    )

    assert "STATUS:200" in result.stdout, (
        f"HTTP to allowed domain failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def test_dns_filtering_http_blocked(scheduler: Scheduler) -> None:
    """Test that blocked domain fails via HTTP."""
    code = """
import urllib.request
try:
    with urllib.request.urlopen("https://google.com/", timeout=5) as r:
        print(f"STATUS:{r.status}")
except Exception as e:
    print(f"BLOCKED:{type(e).__name__}")
"""

    result = await scheduler.run(
        code=code,
        language=Language.PYTHON,
        allow_network=True,
        allowed_domains=["pypi.org"],  # Only pypi allowed
    )

    assert "BLOCKED:" in result.stdout, (
        f"Expected HTTP to google.com to fail but it succeeded.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def test_dns_filtering_javascript(scheduler: Scheduler) -> None:
    """Test DNS filtering with JavaScript/Bun."""
    # Bun's fetch will fail if DNS is blocked
    code = """
try {
    const res = await fetch("https://registry.npmjs.org/", { signal: AbortSignal.timeout(10000) });
    console.log("STATUS:" + res.status);
} catch (e) {
    console.log("ERROR:" + e.message);
}
"""

    result = await scheduler.run(
        code=code,
        language=Language.JAVASCRIPT,
        allow_network=True,
        allowed_domains=["registry.npmjs.org"],
    )

    assert "STATUS:200" in result.stdout, (
        f"Fetch to allowed domain failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def test_network_disabled_no_resolution(scheduler: Scheduler) -> None:
    """Test that allow_network=False prevents all network access."""
    code = """
import socket
try:
    ip = socket.gethostbyname("google.com")
    print(f"RESOLVED:{ip}")
except socket.gaierror as e:
    print(f"NO_NETWORK:{e}")
except OSError as e:
    print(f"NO_NETWORK:{e}")
"""

    result = await scheduler.run(
        code=code,
        language=Language.PYTHON,
        allow_network=False,  # Network disabled
    )

    assert "NO_NETWORK:" in result.stdout or "RESOLVED:" not in result.stdout, (
        f"Expected network to be disabled.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# RAW language tests (using curl instead of Python)
# =============================================================================
async def test_dns_filtering_raw_allowed(scheduler: Scheduler) -> None:
    """Test DNS filtering with RAW language using curl."""
    result = await scheduler.run(
        code="curl -sf --max-time 10 https://example.com/ && echo 'SUCCESS' || echo 'FAILED'",
        language=Language.RAW,
        allow_network=True,
        allowed_domains=["example.com"],
    )

    assert "SUCCESS" in result.stdout, (
        f"curl to allowed domain failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def test_dns_filtering_raw_blocked(scheduler: Scheduler) -> None:
    """Test that blocked domain fails with RAW language."""
    result = await scheduler.run(
        code="curl -sf --max-time 5 https://google.com/ && echo 'SUCCESS' || echo 'BLOCKED'",
        language=Language.RAW,
        allow_network=True,
        allowed_domains=["example.com"],  # google.com not allowed
    )

    assert "BLOCKED" in result.stdout, (
        f"Expected curl to google.com to fail but it succeeded.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# JavaScript blocked domain test
# =============================================================================
async def test_dns_filtering_javascript_blocked(scheduler: Scheduler) -> None:
    """Test that blocked domain fails with JavaScript."""
    code = """
try {
    const res = await fetch("https://google.com/", { signal: AbortSignal.timeout(5000) });
    console.log("STATUS:" + res.status);
} catch (e) {
    console.log("BLOCKED:" + e.name);
}
"""

    result = await scheduler.run(
        code=code,
        language=Language.JAVASCRIPT,
        allow_network=True,
        allowed_domains=["registry.npmjs.org"],  # google.com not allowed
    )

    assert "BLOCKED:" in result.stdout, (
        f"Expected fetch to google.com to fail but it succeeded.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# Boundary / weird input tests
# =============================================================================
async def test_dns_filtering_many_domains(scheduler: Scheduler) -> None:
    """Test with many allowed domains (stress test)."""
    # 50 domains in allowlist
    many_domains = [f"domain{i}.com" for i in range(50)]
    many_domains.append("example.com")  # Add a real one

    code = """
import socket
try:
    ip = socket.gethostbyname("example.com")
    print(f"RESOLVED:{ip}")
except socket.gaierror as e:
    print(f"BLOCKED:{e}")
"""

    result = await scheduler.run(
        code=code,
        language=Language.PYTHON,
        allow_network=True,
        allowed_domains=many_domains,
    )

    assert "RESOLVED:" in result.stdout, f"Large allowlist failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"


async def test_dns_filtering_unicode_domain(scheduler: Scheduler) -> None:
    """Test with internationalized domain name (IDN)."""
    # IDN domain - should either work or fail gracefully
    code = """
import socket
try:
    # This is a real IDN test domain
    ip = socket.gethostbyname("xn--nxasmq5b.com")  # Punycode for a Greek domain
    print(f"RESOLVED:{ip}")
except socket.gaierror as e:
    print(f"BLOCKED:{e}")
except UnicodeError as e:
    print(f"UNICODE_ERROR:{e}")
"""

    result = await scheduler.run(
        code=code,
        language=Language.PYTHON,
        allow_network=True,
        allowed_domains=["xn--nxasmq5b.com"],
    )

    # Should either resolve or be blocked, not crash
    assert "RESOLVED:" in result.stdout or "BLOCKED:" in result.stdout, (
        f"IDN domain handling failed unexpectedly.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def test_dns_filtering_special_tld(scheduler: Scheduler) -> None:
    """Test with special TLDs (.local, .internal)."""
    code = """
import socket
try:
    ip = socket.gethostbyname("test.local")
    print(f"RESOLVED:{ip}")
except socket.gaierror as e:
    print(f"BLOCKED:{e}")
"""

    result = await scheduler.run(
        code=code,
        language=Language.PYTHON,
        allow_network=True,
        allowed_domains=["example.com"],  # .local not in allowlist
    )

    # .local should be blocked (not in allowlist)
    # Blocked domains always return 0.0.0.0 (DNS sinkhole)
    assert "RESOLVED:0.0.0.0" in result.stdout, (
        f"Expected .local to be blocked (0.0.0.0).\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

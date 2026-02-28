"""E2E tests for ECH/GREASE TLS proxy behavior.

Exercises the full proxy stack (gvproxy -> SNI proxy -> allowlist -> DNS
cross-validation -> outbound dial) against real-world ECH-supporting
domains using multiple TLS backends and protocol versions:

- Python ssl (OpenSSL 3.x on Alpine) -- TLS 1.2 / 1.3, cipher selection
- JavaScript/Bun (BoringSSL) -- default TLS (no version control via fetch)
- curl (OpenSSL 3.x on Alpine) -- TLS 1.2 / 1.3, ciphers, curves, false-start

These tests complement the byte-level Go unit tests for ECH GREASE
parsing (committed in gvisor-tap-vsock). They verify that connections
to ECH-supporting servers (Cloudflare) work end-to-end through the
proxy across different TLS versions and handshake configurations.

See commit 31d11c10 for the proxy fix that stopped rejecting ECH/GREASE.
"""

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# =============================================================================
# Domains
# =============================================================================
# Cloudflare serves ECH configs on all plans. Even without client ECH,
# these negotiate TLS 1.3 (and accept 1.2 fallback).
ECH_DOMAIN_CLOUDFLARE = "cloudflare.com"
ECH_DOMAIN_ONE = "one.one.one.one"
# Non-ECH domains (standard TLS 1.3)
DOMAIN_GITHUB = "github.com"
DOMAIN_PYPI = "pypi.org"

# TLS version constants
TLS_12 = "1.2"
TLS_13 = "1.3"
TLS_DEFAULT = "default"

# Mapping from version constant to curl CLI flags
_CURL_VERSION_FLAGS: dict[str, str] = {
    TLS_12: "--tlsv1.2 --tls-max 1.2",
    TLS_13: "--tlsv1.3",
}


# =============================================================================
# Code generators — per runtime x TLS version
# =============================================================================


def _python_tls_code(domain: str, tls_version: str = TLS_DEFAULT) -> str:
    """Python ssl connection test with optional TLS version forcing."""
    version_lines = ["    ctx = ssl.create_default_context()"]
    if tls_version == TLS_12:
        version_lines.append("    ctx.maximum_version = ssl.TLSVersion.TLSv1_2")
    elif tls_version == TLS_13:
        version_lines.append("    ctx.minimum_version = ssl.TLSVersion.TLSv1_3")
    ctx_setup = "\n".join(version_lines)
    return f"""
import socket
import ssl
try:
{ctx_setup}
    with socket.create_connection(("{domain}", 443), timeout=5) as sock:
        with ctx.wrap_socket(sock, server_hostname="{domain}") as ssock:
            print(f"CONNECTED:{{ssock.version()}}")
except Exception as e:
    print(f"BLOCKED:{{type(e).__name__}}")
"""


def _js_tls_code(domain: str, _tls_version: str = TLS_DEFAULT) -> str:
    """Bun/BoringSSL fetch test with top-level await.

    Bun's fetch does not expose TLS version control; _tls_version is
    accepted for signature compatibility but ignored.
    """
    return f"""try {{
    const res = await fetch("https://{domain}/", {{ signal: AbortSignal.timeout(5000) }});
    console.log("CONNECTED:" + res.status);
}} catch (e) {{
    console.log("BLOCKED:" + e.name);
}}"""


def _curl_tls_code(domain: str, tls_version: str = TLS_DEFAULT, extra_flags: str = "") -> str:
    """curl connection test with optional TLS version forcing and extra flags.

    Uses coreutils ``timeout`` as an outer wall-clock deadline because curl's
    ``--max-time`` cannot interrupt musl's blocking ``getaddrinfo()``.  Alpine
    ships curl without c-ares, so DNS resolution is a synchronous libc call.
    musl's resolver ignores EINTR from SIGALRM (``if (poll(…) <= 0) continue``
    in ``res_msend.c``), which means curl's internal timer never fires while
    the process is stuck in DNS.  Under ARM64 TCG with parallel test load the
    entire operation can be inflated 10-25x by CPU starvation, easily exceeding
    the 120 s guest-agent timeout.  ``timeout 15`` guarantees the command
    terminates and the ``|| echo "BLOCKED"`` fallback always runs.
    """
    parts: list[str] = []
    version_flag = _CURL_VERSION_FLAGS.get(tls_version)
    if version_flag:
        parts.append(version_flag)
    if extra_flags:
        parts.append(extra_flags)
    flags = " ".join(parts)
    flags_str = f"{flags} " if flags else ""
    return (
        f"timeout 15 curl -sf {flags_str}--connect-timeout 5 --max-time 10 "
        f'https://{domain}/ -o /dev/null && echo "CONNECTED" || echo "BLOCKED"'
    )


def _get_connection_code(language: Language, domain: str, tls_version: str = TLS_DEFAULT) -> str:
    """Dispatch to the right code generator."""
    if language == Language.PYTHON:
        return _python_tls_code(domain, tls_version)
    if language == Language.JAVASCRIPT:
        return _js_tls_code(domain, tls_version)
    return _curl_tls_code(domain, tls_version)


# =============================================================================
# Test matrix — allowed connections (runtime x domain x TLS version)
# =============================================================================
ECH_ALLOWED_CASES = [
    # -- Python x TLS versions x ECH domains --
    # NOTE: Python x cloudflare.com covered by test_tls_version_diagnostic (stricter assertions)
    pytest.param(
        Language.PYTHON,
        ECH_DOMAIN_ONE,
        TLS_13,
        [ECH_DOMAIN_ONE],
        id="python-tls13-one",
    ),
    pytest.param(
        Language.PYTHON,
        ECH_DOMAIN_ONE,
        TLS_12,
        [ECH_DOMAIN_ONE],
        id="python-tls12-one",
    ),
    # -- JavaScript (default TLS) x domains --
    pytest.param(
        Language.JAVASCRIPT,
        ECH_DOMAIN_CLOUDFLARE,
        TLS_DEFAULT,
        [ECH_DOMAIN_CLOUDFLARE],
        id="js-cloudflare",
    ),
    pytest.param(
        Language.JAVASCRIPT,
        DOMAIN_GITHUB,
        TLS_DEFAULT,
        [DOMAIN_GITHUB],
        id="js-github",
    ),
    pytest.param(
        Language.JAVASCRIPT,
        DOMAIN_PYPI,
        TLS_DEFAULT,
        [DOMAIN_PYPI],
        id="js-pypi",
    ),
    # NOTE: curl x cloudflare.com covered by test_curl_verbose_tls_negotiation (stricter assertions)
]


@pytest.mark.parametrize("language,domain,tls_version,allowed_domains", ECH_ALLOWED_CASES)
async def test_ech_allowed(
    scheduler: Scheduler,
    language: Language,
    domain: str,
    tls_version: str,
    allowed_domains: list[str],
) -> None:
    """TLS connection to allowed domain succeeds through proxy."""
    code = _get_connection_code(language, domain, tls_version)
    result = await scheduler.run(
        code=code,
        language=language,
        allow_network=True,
        allowed_domains=allowed_domains,
    )
    assert "CONNECTED" in result.stdout, (
        f"{language.value} TLS {tls_version} to {domain} should succeed.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# Test matrix — blocked connections (allowlist enforcement)
# =============================================================================
ECH_BLOCKED_CASES = [
    pytest.param(
        Language.JAVASCRIPT,
        ECH_DOMAIN_CLOUDFLARE,
        TLS_DEFAULT,
        ["example.com"],
        id="js-cloudflare-blocked",
    ),
    pytest.param(
        Language.PYTHON,
        ECH_DOMAIN_CLOUDFLARE,
        TLS_13,
        ["example.com"],
        id="python-tls13-cloudflare-blocked",
    ),
    pytest.param(
        Language.RAW,
        ECH_DOMAIN_CLOUDFLARE,
        TLS_13,
        ["example.com"],
        id="curl-tls13-cloudflare-blocked",
    ),
]


@pytest.mark.parametrize("language,domain,tls_version,allowed_domains", ECH_BLOCKED_CASES)
async def test_ech_blocked(
    scheduler: Scheduler,
    language: Language,
    domain: str,
    tls_version: str,
    allowed_domains: list[str],
) -> None:
    """TLS connection to non-allowed domain is blocked."""
    code = _get_connection_code(language, domain, tls_version)
    result = await scheduler.run(
        code=code,
        language=language,
        allow_network=True,
        allowed_domains=allowed_domains,
    )
    assert "BLOCKED" in result.stdout, (
        f"{language.value} TLS {tls_version} to {domain} should be blocked.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# curl advanced TLS — ciphers, curves, false-start
# =============================================================================
CURL_ADVANCED_CASES = [
    # TLS 1.2 specific cipher
    pytest.param(
        TLS_12,
        "--ciphers ECDHE-ECDSA-CHACHA20-POLY1305",
        ECH_DOMAIN_CLOUDFLARE,
        id="curl-tls12-chacha20-poly1305",
    ),
    # TLS 1.3 specific cipher suite
    pytest.param(
        TLS_DEFAULT,
        "--tls13-ciphers TLS_AES_256_GCM_SHA384",
        ECH_DOMAIN_CLOUDFLARE,
        id="curl-tls13-aes256gcm",
    ),
    # X25519 key exchange curve
    pytest.param(
        TLS_DEFAULT,
        "--curves X25519",
        ECH_DOMAIN_CLOUDFLARE,
        id="curl-x25519",
    ),
    # TLS False Start
    pytest.param(
        TLS_DEFAULT,
        "--false-start",
        ECH_DOMAIN_CLOUDFLARE,
        id="curl-false-start",
    ),
]


@pytest.mark.parametrize("tls_version,extra_flags,domain", CURL_ADVANCED_CASES)
async def test_curl_advanced_tls(
    scheduler: Scheduler,
    tls_version: str,
    extra_flags: str,
    domain: str,
) -> None:
    """curl connects with advanced TLS handshake parameters."""
    code = _curl_tls_code(domain, tls_version, extra_flags)
    result = await scheduler.run(
        code=code,
        language=Language.RAW,
        allow_network=True,
        allowed_domains=[domain],
    )
    assert "CONNECTED" in result.stdout, (
        f"curl ({extra_flags}) to {domain} should succeed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# TLS version diagnostic — verify negotiated version matches request
# =============================================================================
VERSION_DIAGNOSTIC_CASES = [
    pytest.param(ECH_DOMAIN_CLOUDFLARE, TLS_13, "TLSv1.3", id="tls13-cloudflare"),
    pytest.param(ECH_DOMAIN_CLOUDFLARE, TLS_12, "TLSv1.2", id="tls12-cloudflare"),
]


@pytest.mark.parametrize("domain,tls_version,expected_proto", VERSION_DIAGNOSTIC_CASES)
async def test_tls_version_diagnostic(
    scheduler: Scheduler,
    domain: str,
    tls_version: str,
    expected_proto: str,
) -> None:
    """Python ssl reports the expected TLS version was negotiated."""
    code = _python_tls_code(domain, tls_version)
    result = await scheduler.run(
        code=code,
        language=Language.PYTHON,
        allow_network=True,
        allowed_domains=[domain],
    )
    assert f"CONNECTED:{expected_proto}" in result.stdout, (
        f"Expected {expected_proto} with {domain} (forced {tls_version}).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# curl verbose TLS — verify handshake details in verbose output
# =============================================================================
CURL_VERBOSE_CASES = [
    pytest.param(
        TLS_13,
        ECH_DOMAIN_CLOUDFLARE,
        "TLSv1.3",
        "TLS_AES_256_GCM_SHA384",
        id="curl-verbose-tls13",
    ),
    pytest.param(
        TLS_12,
        ECH_DOMAIN_CLOUDFLARE,
        "TLSv1.2",
        "ECDHE",
        id="curl-verbose-tls12",
    ),
]


@pytest.mark.parametrize("tls_version,domain,expected_version,expected_cipher", CURL_VERBOSE_CASES)
async def test_curl_verbose_tls_negotiation(
    scheduler: Scheduler,
    tls_version: str,
    domain: str,
    expected_version: str,
    expected_cipher: str,
) -> None:
    """curl verbose output confirms TLS version and cipher negotiated.

    ``timeout 15`` wraps the entire pipeline so that even if curl hangs in
    musl's blocking ``getaddrinfo()`` (see ``_curl_tls_code`` docstring),
    grep still receives EOF and the ``|| echo "BLOCKED"`` fallback fires
    instead of the pipeline hanging until the guest-agent timeout.
    """
    curl_flags = _CURL_VERSION_FLAGS[tls_version]
    code = (
        f"timeout 15 curl -svf {curl_flags} --connect-timeout 5 --max-time 10 "
        f'https://{domain}/ -o /dev/null 2>&1 | grep "SSL connection using" '
        f'&& echo "CONNECTED" || echo "BLOCKED"'
    )
    result = await scheduler.run(
        code=code,
        language=Language.RAW,
        allow_network=True,
        allowed_domains=[domain],
    )
    assert "CONNECTED" in result.stdout, (
        f"curl TLS {tls_version} should connect to {domain}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert expected_version in result.stdout, f"Expected {expected_version} in verbose output.\nstdout: {result.stdout}"
    assert expected_cipher in result.stdout, f"Expected '{expected_cipher}' in verbose output.\nstdout: {result.stdout}"


# =============================================================================
# Multi-domain sequential — multiple ECH domains in one session
# =============================================================================


async def test_ech_javascript_multiple_ech_domains(
    scheduler: Scheduler,
) -> None:
    """Bun connects to multiple ECH-supporting domains sequentially."""
    code = f"""try {{
    const r1 = await fetch("https://{ECH_DOMAIN_CLOUDFLARE}/", {{ signal: AbortSignal.timeout(5000) }});
    console.log("CONNECTED:cloudflare:" + r1.status);
}} catch (e) {{
    console.log("BLOCKED:cloudflare:" + e.name);
}}
try {{
    const r2 = await fetch("https://{ECH_DOMAIN_ONE}/", {{ signal: AbortSignal.timeout(5000) }});
    console.log("CONNECTED:one:" + r2.status);
}} catch (e) {{
    console.log("BLOCKED:one:" + e.name);
}}"""
    result = await scheduler.run(
        code=code,
        language=Language.JAVASCRIPT,
        allow_network=True,
        allowed_domains=[ECH_DOMAIN_CLOUDFLARE, ECH_DOMAIN_ONE],
    )
    assert "CONNECTED:cloudflare:" in result.stdout, (
        f"Bun should connect to {ECH_DOMAIN_CLOUDFLARE}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "CONNECTED:one:" in result.stdout, (
        f"Bun should connect to {ECH_DOMAIN_ONE}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

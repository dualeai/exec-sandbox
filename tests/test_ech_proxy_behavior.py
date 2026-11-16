"""E2E tests for ECH/GREASE TLS proxy behavior.

Exercises the full proxy stack (gvproxy -> SNI proxy -> allowlist -> DNS
cross-validation -> outbound dial) against real-world ECH-supporting
domains using multiple TLS backends and protocol versions:

- Python ssl (the OpenSSL 3.x of python-build-standalone) -- TLS 1.2 / 1.3
- JavaScript/Bun (BoringSSL) -- default TLS (no version control via fetch)
- curl (OpenSSL 3.x on Alpine) -- TLS 1.2 / 1.3, ciphers, curves, false-start

These tests complement the byte-level Go unit tests for ECH GREASE
parsing (in the dualeai fork of gvisor-tap-vsock). They verify that connections
to ECH-supporting servers (Cloudflare) work end-to-end through the
proxy across different TLS versions and handshake configurations.
"""

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

# =============================================================================
# Domains
# =============================================================================
# Cloudflare serves ECH configs on all plans. Even without client ECH,
# these negotiate TLS 1.3 (and accept 1.2 fallback).
ECH_DOMAIN_CLOUDFLARE = "cloudflare.com"
ECH_DOMAIN_ONE = "one.one.one.one"
# Non-ECH domains (standard TLS 1.3)
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

# Mapping from version constant to the Python line that forces it on `ctx`
_PYTHON_VERSION_LINES: dict[str, str] = {
    TLS_12: "ctx.maximum_version = ssl.TLSVersion.TLSv1_2",
    TLS_13: "ctx.minimum_version = ssl.TLSVersion.TLSv1_3",
}


# =============================================================================
# Code generators — per runtime x TLS version
# =============================================================================


def _python_tls_code(domain: str, tls_version: str, *, retries: bool) -> str:
    """Python ssl connection test that forces a TLS version (1.2 or 1.3): CONNECTED:<version>,
    or BLOCKED:<exception type>.

    When *retries* is True, retries transient failures (``NET_RETRY_COUNT`` more attempts,
    linear backoff): used for allowed-domain tests.
    """
    attempts = NET_RETRY_COUNT + 1 if retries else 1
    return (
        PY_SAFETY
        + f"""
import socket, ssl, time
for attempt in range({attempts}):
    try:
        ctx = ssl.create_default_context()
        {_PYTHON_VERSION_LINES[tls_version]}
        with socket.create_connection(("{domain}", 443), timeout={NET_CONNECT_TIMEOUT_S}) as sock:
            with ctx.wrap_socket(sock, server_hostname="{domain}") as ssock:
                print(f"CONNECTED:{{ssock.version()}}")
                break
    except Exception as e:
        if attempt < {attempts - 1}:
            time.sleep({NET_RETRY_BACKOFF_S} * (attempt + 1))
            continue
        print(f"BLOCKED:{{type(e).__name__}}")
"""
    )


def _js_tls_code(domain: str, *, retries: bool) -> str:
    """Bun/BoringSSL fetch test with top-level await: CONNECTED once a response came (any HTTP
    status: the TLS connection worked).

    Bun's fetch does not expose TLS version control: the TLS version of the
    case does not apply.

    When *retries* is True (allowed-domain tests), makes up to
    ``NET_RETRY_COUNT + 1`` attempts with linear backoff to absorb
    transient connection resets and BoringSSL certificate verification
    flakes observed in CI (see guest-agent ``spawn.rs`` TLS comment for
    full analysis).
    """
    attempts = NET_RETRY_COUNT + 1 if retries else 1
    return (
        JS_SAFETY
        + f"""for (let attempt = 0; attempt < {attempts}; attempt++) {{
    try {{
        await fetch("https://{domain}/", {{ signal: AbortSignal.timeout({NET_OP_TIMEOUT_S * 1000}) }});
        console.log("CONNECTED");
        break;
    }} catch (e) {{
        if (attempt < {attempts - 1}) {{ await new Promise(r => setTimeout(r, {NET_RETRY_BACKOFF_S * 1000} * (attempt + 1))); continue; }}
        console.log("BLOCKED:" + e.name + ":" + e.message);
    }}
}}"""
    )


def _curl_tls_code(domain: str, *, tls_version: str, extra_flags: str, retries: bool) -> str:
    """curl connection test with optional TLS version forcing and extra flags.

    Uses coreutils ``timeout`` as an outer wall-clock deadline because curl's
    ``--max-time`` cannot interrupt musl's blocking ``getaddrinfo()``.
    Although Alpine's curl includes c-ares (``--enable-ares``), certain code
    paths still fall back to musl's synchronous resolver.  musl ignores
    EINTR during ``poll()`` (``if (poll(…) <= 0) continue`` in
    ``res_msend.c``), so curl's internal SIGALRM timer never fires while
    DNS is stuck.  Under ARM64 TCG with parallel test load the entire
    operation can be inflated 10-25x by CPU starvation, easily exceeding
    the exec timeout of the tests (120 s, 240 s under TCG).  ``timeout {NET_SAFETY_TIMEOUT_S}``
    guarantees the command terminates and the ``|| echo "BLOCKED:$?"``
    fallback always runs (``$?`` is the curl exit code, 6 when the name does not resolve,
    or 124 when ``timeout`` stopped it).

    See guest-agent ``spawn.rs`` Raw language comment for full TLS analysis.

    References:
        - Alpine curl APKBUILD (--enable-ares, --with-ca-bundle):
          https://gitlab.alpinelinux.org/alpine/aports/-/blob/master/main/curl/APKBUILD
        - musl res_msend.c poll() loop (ignores EINTR):
          https://www.openwall.com/lists/musl/2017/09/28/1
    """
    version_flags = _CURL_VERSION_FLAGS.get(tls_version, "")
    retry_flags = f"--retry {NET_RETRY_COUNT} --retry-connrefused" if retries else ""
    return (
        f"timeout {NET_SAFETY_TIMEOUT_S} curl -sf {version_flags} {extra_flags}"
        f" --connect-timeout {NET_CONNECT_TIMEOUT_S} --max-time {NET_OP_TIMEOUT_S} {retry_flags}"
        f' https://{domain}/ -o /dev/null && echo "CONNECTED" || echo "BLOCKED:$?"'
    )


# =============================================================================
# Allowed connections: Bun (default TLS) x domains
# =============================================================================
# NOTE: Python x TLS versions x ECH domains: test_tls_version_diagnostic (it also checks the negotiated version)
# NOTE: curl x cloudflare.com covered by test_curl_verbose_tls_negotiation (stricter assertions)
@pytest.mark.parametrize(
    "domain",
    [
        pytest.param(ECH_DOMAIN_CLOUDFLARE, id="js-cloudflare"),
        pytest.param(ECH_DOMAIN_ONE, id="js-one"),
        pytest.param(DOMAIN_PYPI, id="js-pypi"),
    ],
)
async def test_ech_allowed(scheduler: Scheduler, domain: str) -> None:
    """Bun's TLS connection to an allowed domain succeeds through the proxy."""
    code = _js_tls_code(domain, retries=True)
    result = await run_once(
        scheduler, VmSpec(language=Language.JAVASCRIPT, allow_network=True, allowed_domains=[domain]), code
    )
    assert result.stdout == "CONNECTED\n", (
        f"Bun TLS to {domain} should succeed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# Test matrix — blocked connections (allowlist enforcement)
# =============================================================================
# Only example.com is allowed. The filter blocks a non-allowed name at DNS: each runtime reports a name that
# does not resolve.
ECH_BLOCKED_CASES = [
    pytest.param(
        Language.JAVASCRIPT,
        _js_tls_code(ECH_DOMAIN_CLOUDFLARE, retries=False),
        "BLOCKED:TypeError:getaddrinfo ENOTFOUND cloudflare.com\n",
        id="js-cloudflare-blocked",
    ),
    pytest.param(
        Language.PYTHON,
        _python_tls_code(ECH_DOMAIN_CLOUDFLARE, TLS_13, retries=False),
        "BLOCKED:gaierror\n",
        id="python-tls13-cloudflare-blocked",
    ),
    pytest.param(
        Language.RAW,
        _curl_tls_code(ECH_DOMAIN_CLOUDFLARE, tls_version=TLS_13, extra_flags="", retries=False),
        "BLOCKED:6\n",  # curl exit code 6: could not resolve host
        id="curl-tls13-cloudflare-blocked",
    ),
]


@pytest.mark.parametrize("language,code,expected", ECH_BLOCKED_CASES)
async def test_ech_blocked(scheduler: Scheduler, language: Language, code: str, expected: str) -> None:
    """TLS connection to a non-allowed domain (cloudflare.com) fails at DNS."""
    result = await run_once(
        scheduler, VmSpec(language=language, allow_network=True, allowed_domains=["example.com"]), code
    )
    assert result.stdout == expected, (
        f"{language.value} TLS to {ECH_DOMAIN_CLOUDFLARE} should be blocked.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# curl advanced TLS — ciphers, curves, false-start (cloudflare.com)
# =============================================================================
CURL_ADVANCED_CASES = [
    # TLS 1.2 specific cipher
    pytest.param(TLS_12, "--ciphers ECDHE-ECDSA-CHACHA20-POLY1305", id="curl-tls12-chacha20-poly1305"),
    # TLS 1.3 specific cipher suite
    pytest.param(TLS_DEFAULT, "--tls13-ciphers TLS_AES_256_GCM_SHA384", id="curl-tls13-aes256gcm"),
    # X25519 key exchange curve
    pytest.param(TLS_DEFAULT, "--curves X25519", id="curl-x25519"),
    # TLS False Start
    pytest.param(TLS_DEFAULT, "--false-start", id="curl-false-start"),
]


@pytest.mark.parametrize("tls_version,extra_flags", CURL_ADVANCED_CASES)
async def test_curl_advanced_tls(scheduler: Scheduler, tls_version: str, extra_flags: str) -> None:
    """curl connects to cloudflare.com with advanced TLS handshake parameters."""
    code = _curl_tls_code(ECH_DOMAIN_CLOUDFLARE, tls_version=tls_version, extra_flags=extra_flags, retries=True)
    result = await run_once(
        scheduler, VmSpec(language=Language.RAW, allow_network=True, allowed_domains=[ECH_DOMAIN_CLOUDFLARE]), code
    )
    assert result.stdout == "CONNECTED\n", (
        f"curl ({extra_flags}) to {ECH_DOMAIN_CLOUDFLARE} should succeed.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# TLS version diagnostic — verify negotiated version matches request
# =============================================================================
VERSION_DIAGNOSTIC_CASES = [
    pytest.param(ECH_DOMAIN_CLOUDFLARE, TLS_13, "TLSv1.3", id="tls13-cloudflare"),
    pytest.param(ECH_DOMAIN_CLOUDFLARE, TLS_12, "TLSv1.2", id="tls12-cloudflare"),
    pytest.param(ECH_DOMAIN_ONE, TLS_13, "TLSv1.3", id="tls13-one"),
    pytest.param(ECH_DOMAIN_ONE, TLS_12, "TLSv1.2", id="tls12-one"),
]


@pytest.mark.parametrize("domain,tls_version,expected_proto", VERSION_DIAGNOSTIC_CASES)
async def test_tls_version_diagnostic(
    scheduler: Scheduler,
    domain: str,
    tls_version: str,
    expected_proto: str,
) -> None:
    """Python ssl connects to an allowed domain through the proxy and negotiates the forced TLS version."""
    code = _python_tls_code(domain, tls_version, retries=True)
    result = await run_once(
        scheduler, VmSpec(language=Language.PYTHON, allow_network=True, allowed_domains=[domain]), code
    )
    assert result.stdout == f"CONNECTED:{expected_proto}\n", (
        f"Expected {expected_proto} with {domain} (forced {tls_version}).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# =============================================================================
# curl verbose TLS — verify handshake details in verbose output
# =============================================================================
CURL_VERBOSE_CASES = [
    pytest.param(TLS_13, "TLSv1.3", "TLS_AES_256_GCM_SHA384", id="curl-verbose-tls13"),
    pytest.param(TLS_12, "TLSv1.2", "ECDHE", id="curl-verbose-tls12"),
]


@pytest.mark.parametrize("tls_version,expected_version,expected_cipher", CURL_VERBOSE_CASES)
async def test_curl_verbose_tls_negotiation(
    scheduler: Scheduler,
    tls_version: str,
    expected_version: str,
    expected_cipher: str,
) -> None:
    """curl verbose output confirms TLS version and cipher negotiated with cloudflare.com.

    ``timeout {NET_SAFETY_TIMEOUT_S}`` wraps the entire pipeline so that
    even if curl hangs in musl's blocking ``getaddrinfo()`` (see
    ``_curl_tls_code`` docstring), grep still receives EOF and the
    ``|| echo "BLOCKED"`` fallback runs instead of the pipeline hanging
    until the guest-agent timeout.
    """
    curl_flags = _CURL_VERSION_FLAGS[tls_version]
    code = (
        f"timeout {NET_SAFETY_TIMEOUT_S} curl -svf {curl_flags}"
        f" --connect-timeout {NET_CONNECT_TIMEOUT_S} --max-time {NET_OP_TIMEOUT_S} "
        f'https://{ECH_DOMAIN_CLOUDFLARE}/ -o /dev/null 2>&1 | grep "SSL connection using" '
        f'&& echo "CONNECTED" || echo "BLOCKED"'
    )
    result = await run_once(
        scheduler, VmSpec(language=Language.RAW, allow_network=True, allowed_domains=[ECH_DOMAIN_CLOUDFLARE]), code
    )
    # The handshake line, then CONNECTED. The test asks only for a TLS version: the rest of the line
    # (cipher, key exchange group, signature) is the outcome of the handshake, so only the version
    # and a part of the cipher name (expected_cipher) are checked.
    *handshake, connected = result.stdout.splitlines()
    assert connected == "CONNECTED", (
        f"curl TLS {tls_version} should connect to {ECH_DOMAIN_CLOUDFLARE}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    (line,) = handshake
    assert line.startswith(f"* SSL connection using {expected_version} / "), line
    assert expected_cipher in line, line


# =============================================================================
# Multi-domain sequential — multiple ECH domains in one session
# =============================================================================


async def test_ech_javascript_multiple_ech_domains(
    scheduler: Scheduler,
) -> None:
    """Bun connects to multiple ECH-supporting domains sequentially."""
    code = (
        JS_SAFETY
        + f"""try {{
    await fetch("https://{ECH_DOMAIN_CLOUDFLARE}/", {{ signal: AbortSignal.timeout({NET_OP_TIMEOUT_S * 1000}) }});
    console.log("CONNECTED:cloudflare");
}} catch (e) {{
    console.log("BLOCKED:cloudflare:" + e.name);
}}
try {{
    await fetch("https://{ECH_DOMAIN_ONE}/", {{ signal: AbortSignal.timeout({NET_OP_TIMEOUT_S * 1000}) }});
    console.log("CONNECTED:one");
}} catch (e) {{
    console.log("BLOCKED:one:" + e.name);
}}"""
    )
    result = await run_once(
        scheduler,
        VmSpec(
            language=Language.JAVASCRIPT, allow_network=True, allowed_domains=[ECH_DOMAIN_CLOUDFLARE, ECH_DOMAIN_ONE]
        ),
        code,
    )
    assert result.stdout == "CONNECTED:cloudflare\nCONNECTED:one\n", (
        f"Bun should connect to {ECH_DOMAIN_CLOUDFLARE} and {ECH_DOMAIN_ONE}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

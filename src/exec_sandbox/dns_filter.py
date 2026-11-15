"""Outbound allow configuration for gvproxy-wrapper.

Generates OutboundAllow regex patterns for gvproxy outbound filtering.
Patterns are matched against both DNS queries and TLS SNI.
"""

import json
import re
from collections.abc import Sequence

from exec_sandbox.constants import NPM_PACKAGE_DOMAINS, PYTHON_PACKAGE_DOMAINS

# Security: Domain validation pattern (RFC 1035 compliant)
# Prevents ReDoS attacks and regex injection via malicious domain input
# Labels: alphanumeric, can contain hyphens (not at start/end), 1-63 chars each
# TLD: letters only (no digits)
# Total length: max 253 characters
_DOMAIN_LABEL_PATTERN = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$")
_DOMAIN_MAX_LENGTH = 253


def validate_domain(domain: str) -> None:
    """Validate domain is RFC 1035 compliant.

    Prevents regex injection and ReDoS attacks by ensuring domains contain
    only valid characters before using them in regex patterns.

    Args:
        domain: Domain name to validate (e.g., "pypi.org", "example.com")

    Raises:
        ValueError: If domain format is invalid

    Examples:
        >>> validate_domain("pypi.org")  # Valid
        >>> validate_domain("files.pythonhosted.org")  # Valid
        >>> validate_domain("x" * 300)  # Raises ValueError (too long)
        >>> validate_domain("bad..domain")  # Raises ValueError (empty label)
        >>> validate_domain("-invalid.com")  # Raises ValueError (starts with hyphen)
    """
    if not domain:
        raise ValueError("Domain cannot be empty")
    # One final dot is the FQDN form: not part of the 253-character name; a second is
    # an empty label
    name = domain.removesuffix(".")
    if len(name) > _DOMAIN_MAX_LENGTH:
        raise ValueError(f"Domain too long: {len(name)} > {_DOMAIN_MAX_LENGTH}")

    # Split and validate each label
    labels = name.split(".")
    if len(labels) < 2:  # noqa: PLR2004
        raise ValueError(f"Domain must have at least 2 labels (got {len(labels)}): {domain!r}")

    for label in labels:
        if not label:
            raise ValueError(f"Domain contains empty label: {domain!r}")
        if len(label) > 63:  # noqa: PLR2004
            raise ValueError(f"Domain label too long ({len(label)} > 63): {label!r}")
        if not _DOMAIN_LABEL_PATTERN.fullmatch(label):  # match with `$` would let a final newline in
            raise ValueError(f"Invalid domain label format: {label!r} in {domain!r}")

    # TLD must be alphabetic only (no numbers allowed)
    tld = labels[-1]
    if not tld.isalpha():
        raise ValueError(f"TLD must be alphabetic only: {tld!r} in {domain!r}")


def normalize_domain(domain: str) -> str:
    """`domain` as the network filter compares names: checked (validate_domain), with no final
    dot (the FQDN form) and lowercase. gvproxy lowercases DNS names and TLS SNI before its
    regex match, and drops the final dot of a DNS name; SNI never has one.

    Raises:
        ValueError: the domain is not valid (validate_domain)
    """
    validate_domain(domain)
    return domain.removesuffix(".").lower()


def create_outbound_patterns(domains: Sequence[str]) -> list[str]:
    """Create OutboundAllow regex patterns from domain list.

    Each pattern matches the domain and all its subdomains, with optional
    trailing dot (FQDN format).

    Args:
        domains: List of domain names to whitelist

    Returns:
        List of regex pattern strings for OutboundAllow

    Raises:
        ValueError: If any domain has invalid format (see validate_domain; VmSpec
            checks allowed_domains with it when the spec is built)

    Example:
        >>> patterns = create_outbound_patterns(["pypi.org", "example.com"])
        >>> patterns[0]
        '^(.*\\\\.)?pypi\\\\.org\\\\.?$'
    """
    patterns: list[str] = []
    for domain in domains:
        # Security: Validate domain format before constructing regex
        # Prevents ReDoS attacks and regex injection
        name = normalize_domain(domain)
        patterns.append(
            # Match domain AND all subdomains: (.*\.)? makes prefix optional
            # Matches both "pypi.org" and "www.pypi.org"
            # Trailing \.? handles FQDN format (e.g., "google.com.")
            f"^(.*\\.)?{name.replace('.', '\\.')}\\.?$"
        )
    return patterns


def resolved_domains(allowed_domains: Sequence[str] | None, language: str) -> tuple[str, ...]:
    """The domains a VM may reach with the network on: `allowed_domains`, or for None the
    package registries of the language (raw has none). Empty: no filter, all allowed."""
    if allowed_domains is None:
        return tuple({"python": PYTHON_PACKAGE_DOMAINS, "javascript": NPM_PACKAGE_DOMAINS}.get(language, []))
    return tuple(allowed_domains)


def generate_outbound_allow_json(allowed_domains: Sequence[str]) -> str:
    """Generate gvproxy OutboundAllow JSON.

    Args:
        allowed_domains: The domains the guest may reach (resolved_domains); empty for no
            domain filter (all outbound allowed)

    Returns:
        JSON string for gvproxy -outbound-allow flag

    Raises:
        ValueError: A domain is not valid (validate_domain)

    Example:
        >>> "pypi" in generate_outbound_allow_json(resolved_domains(None, "python"))
        True
        >>> generate_outbound_allow_json([])  # No filter: all allowed
        '[]'
    """
    # No domain ("[]") = no domain filter: with allow_network, all outbound is allowed
    # (by design). Blocking all outbound is the -block-outbound flag, set only
    # when ports are exposed without allow_network.
    return json.dumps(create_outbound_patterns(allowed_domains), separators=(",", ":"))

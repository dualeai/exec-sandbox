"""Outbound allow configuration for gvproxy-wrapper.

Generates OutboundAllow regex patterns for gvproxy outbound filtering.
Patterns are matched against both DNS queries and TLS SNI.
"""

import json
import re

from exec_sandbox.constants import NPM_PACKAGE_DOMAINS, PYTHON_PACKAGE_DOMAINS

# Security: Domain validation pattern (RFC 1035 compliant)
# Prevents ReDoS attacks and regex injection via malicious domain input
# Labels: alphanumeric, can contain hyphens (not at start/end), 1-63 chars each
# TLD: must be alphabetic only (no numbers), 2+ chars
# Total length: max 253 characters
_DOMAIN_LABEL_PATTERN = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$")
_DOMAIN_MAX_LENGTH = 253


def _validate_domain(domain: str) -> None:
    """Validate domain is RFC 1035 compliant.

    Prevents regex injection and ReDoS attacks by ensuring domains contain
    only valid characters before using them in regex patterns.

    Args:
        domain: Domain name to validate (e.g., "pypi.org", "example.com")

    Raises:
        ValueError: If domain format is invalid

    Examples:
        >>> _validate_domain("pypi.org")  # Valid
        >>> _validate_domain("files.pythonhosted.org")  # Valid
        >>> _validate_domain("x" * 300)  # Raises ValueError (too long)
        >>> _validate_domain("bad..domain")  # Raises ValueError (empty label)
        >>> _validate_domain("-invalid.com")  # Raises ValueError (starts with hyphen)
    """
    if not domain:
        raise ValueError("Domain cannot be empty")
    if len(domain) > _DOMAIN_MAX_LENGTH:
        raise ValueError(f"Domain too long: {len(domain)} > {_DOMAIN_MAX_LENGTH}")

    # Split and validate each label
    labels = domain.rstrip(".").split(".")
    if len(labels) < 2:  # noqa: PLR2004
        raise ValueError(f"Domain must have at least 2 labels (got {len(labels)}): {domain!r}")

    for label in labels:
        if not label:
            raise ValueError(f"Domain contains empty label: {domain!r}")
        if len(label) > 63:  # noqa: PLR2004
            raise ValueError(f"Domain label too long ({len(label)} > 63): {label!r}")
        if not _DOMAIN_LABEL_PATTERN.match(label):
            raise ValueError(f"Invalid domain label format: {label!r} in {domain!r}")

    # TLD must be alphabetic only (no numbers allowed)
    tld = labels[-1]
    if not tld.isalpha():
        raise ValueError(f"TLD must be alphabetic only: {tld!r} in {domain!r}")


def create_outbound_patterns(domains: list[str]) -> list[str]:
    """Create OutboundAllow regex patterns from domain list.

    Each pattern matches the domain and all its subdomains, with optional
    trailing dot (FQDN format).

    Args:
        domains: List of domain names to whitelist

    Returns:
        List of regex pattern strings for OutboundAllow

    Raises:
        ValueError: If any domain has invalid format (see _validate_domain)

    Example:
        >>> patterns = create_outbound_patterns(["pypi.org", "example.com"])
        >>> patterns[0]
        '^(.*\\\\.)?pypi\\\\.org\\\\.?$'
    """
    patterns: list[str] = []
    for domain in domains:
        # Security: Validate domain format before constructing regex
        # Prevents ReDoS attacks and regex injection
        _validate_domain(domain)
        patterns.append(
            # Match domain AND all subdomains: (.*\.)? makes prefix optional
            # Matches both "pypi.org" and "www.pypi.org"
            # Trailing \.? handles FQDN format (e.g., "google.com.")
            f"^(.*\\.)?{domain.replace('.', '\\.')}\\.?$"
        )
    return patterns


def generate_outbound_allow_json(
    allowed_domains: list[str] | None,
    language: str,
) -> str:
    """Generate gvproxy OutboundAllow JSON.

    Args:
        allowed_domains: Custom allowed domains, empty list to block ALL outbound,
                        or None for language defaults
        language: Programming language (for default package registries)

    Returns:
        JSON string for gvproxy -outbound-allow flag

    Example:
        >>> json_str = generate_outbound_allow_json(None, "python")
        >>> "pypi" in json_str
        True
        >>> json_str = generate_outbound_allow_json([], "python")  # Block all
        >>> json_str
        '[]'
    """
    # Auto-expand package domains if not specified (None = use defaults)
    if allowed_domains is None:
        if language == "python":
            allowed_domains = PYTHON_PACKAGE_DOMAINS.copy()
        elif language == "javascript":
            allowed_domains = NPM_PACKAGE_DOMAINS.copy()
        else:
            # No language-specific defaults, no filtering
            return "[]"

    # Empty list = block ALL outbound (handled by BlockAllOutbound flag,
    # OutboundAllow stays empty so it doesn't interfere)
    if len(allowed_domains) == 0:
        return "[]"

    # Create outbound allow patterns
    patterns = create_outbound_patterns(allowed_domains)
    return json.dumps(patterns, separators=(",", ":"))

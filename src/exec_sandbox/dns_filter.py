"""DNS zone configuration for gvproxy-wrapper.

Generates DNS zones JSON for gvproxy DNS filtering.
"""

import json
import re
from typing import Final

from pydantic import BaseModel, Field

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


class DNSRecord(BaseModel):
    """DNS record with regex matching for gvproxy."""

    name: str = Field(description="Domain name")
    Regexp: str = Field(description="Regex pattern for matching subdomains", alias="Regexp")
    # Omit IP field - gvproxy will forward to upstream DNS


class DNSZone(BaseModel):
    """DNS zone configuration for gvproxy."""

    name: str = Field(description="Zone name")
    records: list[DNSRecord] = Field(description="DNS records to allow")
    defaultIP: str = Field(default="0.0.0.0", description="IP for blocked domains (0.0.0.0 = NXDOMAIN)")  # noqa: N815, S104


# Default package registry domains
PYTHON_PACKAGE_DOMAINS: Final[list[str]] = [
    "pypi.org",
    "files.pythonhosted.org",
]

NPM_PACKAGE_DOMAINS: Final[list[str]] = [
    "registry.npmjs.org",
]


def create_dns_records(domains: list[str]) -> list[DNSRecord]:
    """Create DNS records from domain list.

    Args:
        domains: List of domain names to whitelist

    Returns:
        List of DNSRecord objects with regex patterns

    Raises:
        ValueError: If any domain has invalid format (see _validate_domain)

    Example:
        >>> records = create_dns_records(["pypi.org", "example.com"])
        >>> records[0].name
        'pypi.org'
        >>> records[0].Regexp
        '^(.*\\\\.)?pypi\\\\.org\\\\.?$'
    """
    records: list[DNSRecord] = []
    for domain in domains:
        # Security: Validate domain format before constructing regex
        # Prevents ReDoS attacks and regex injection
        _validate_domain(domain)
        records.append(
            DNSRecord(
                name=domain,
                # Match domain AND all subdomains: (.*\.)? makes prefix optional
                # Matches both "pypi.org" and "www.pypi.org"
                # Trailing \.? handles FQDN format (e.g., "google.com.")
                Regexp=f"^(.*\\.)?{domain.replace('.', '\\.')}\\.?$",
                # No IP field - gvproxy will forward to upstream DNS
            )
        )
    return records


def create_dns_zone(
    domains: list[str],
    zone_name: str = "",  # Empty string creates "." suffix, matches all FQDNs
    block_others: bool = True,
) -> DNSZone:
    """Create DNS zone from domain list.

    Args:
        domains: List of domain names to whitelist
        zone_name: Name for the DNS zone
        block_others: If True, block all non-whitelisted domains

    Returns:
        DNSZone configured for whitelisting

    Example:
        >>> zone = create_dns_zone(["pypi.org"])
        >>> zone.name
        'allowed'
        >>> zone.defaultIP
        '0.0.0.0'
    """
    return DNSZone(
        name=zone_name,
        records=create_dns_records(domains),
        defaultIP="0.0.0.0" if block_others else "8.8.8.8",  # noqa: S104
    )


def generate_dns_zones_json(
    allowed_domains: list[str] | None,
    language: str,
) -> str:
    """Generate gvproxy DNS zones JSON.

    Args:
        allowed_domains: Custom allowed domains, empty list to block ALL DNS,
                        or None for language defaults
        language: Programming language (for default package registries)

    Returns:
        JSON string for gvproxy -dns-zones flag

    Example:
        >>> json_str = generate_dns_zones_json(None, "python")
        >>> "pypi.org" in json_str
        True
        >>> json_str = generate_dns_zones_json([], "python")  # Block all DNS
        >>> "0.0.0.0" in json_str
        True
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

    # Empty list = block ALL DNS (for Mode 1: port-forward only, no internet)
    # This creates a zone with defaultIP=0.0.0.0 and no records = all DNS blocked
    if len(allowed_domains) == 0:
        zone = DNSZone(
            name="",  # Root zone - matches all queries
            records=[],  # No allowed records
            defaultIP="0.0.0.0",  # Block everything  # noqa: S104
        )
        return json.dumps([zone.model_dump()], separators=(",", ":"))

    # Create DNS zone with allowed domains
    zone = create_dns_zone(allowed_domains)

    # Serialize to JSON (gvproxy expects array of zones)
    zones = [zone.model_dump()]
    return json.dumps(zones, separators=(",", ":"))  # Compact JSON


def parse_dns_zones_json(zones_json: str) -> list[DNSZone]:
    """Parse gvproxy DNS zones JSON.

    Args:
        zones_json: JSON string from generate_dns_zones_json

    Returns:
        List of DNSZone objects

    Raises:
        ValueError: If JSON is invalid
    """
    try:
        zones_data = json.loads(zones_json)
        return [DNSZone(**zone) for zone in zones_data]
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid DNS zones JSON: {e}") from e

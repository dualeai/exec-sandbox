"""DNS zone configuration for gvproxy-wrapper.

Generates DNS zones JSON for gvproxy DNS filtering.
"""

import json
from typing import Final

from pydantic import BaseModel, Field


class DNSRecord(BaseModel):
    """DNS record with regex matching for gvproxy."""

    name: str = Field(description="Domain name")
    Regexp: str = Field(description="Regex pattern for matching subdomains", alias="Regexp")
    # Omit IP field - gvproxy will forward to upstream DNS


class DNSZone(BaseModel):
    """DNS zone configuration for gvproxy."""

    name: str = Field(description="Zone name")
    records: list[DNSRecord] = Field(description="DNS records to allow")
    defaultIP: str = Field(default="0.0.0.0", description="IP for blocked domains (0.0.0.0 = NXDOMAIN)")


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

    Example:
        >>> records = create_dns_records(["pypi.org", "example.com"])
        >>> records[0].name
        'pypi.org'
        >>> records[0].Regexp
        '^.*\\.pypi\\.org\\.?$'
    """
    return [
        DNSRecord(
            name=domain,
            # Match domain AND all subdomains: (.*\.)? makes prefix optional
            # Matches both "pypi.org" and "www.pypi.org"
            # Trailing \.? handles FQDN format (e.g., "google.com.")
            Regexp=f"^(.*\\.)?{domain.replace('.', '\\.')}\\.?$",
            # No IP field - gvproxy will forward to upstream DNS
        )
        for domain in domains
    ]


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
        defaultIP="0.0.0.0" if block_others else "8.8.8.8",
    )


def generate_dns_zones_json(
    allowed_domains: list[str] | None,
    language: str,
) -> str:
    """Generate gvproxy DNS zones JSON.

    Args:
        allowed_domains: Custom allowed domains, or None for language defaults
        language: Programming language (for default package registries)

    Returns:
        JSON string for gvproxy -dns-zones flag

    Example:
        >>> json_str = generate_dns_zones_json(None, "python")
        >>> "pypi.org" in json_str
        True
    """
    # Auto-expand package domains if not specified
    if allowed_domains is None:
        if language == "python":
            allowed_domains = PYTHON_PACKAGE_DOMAINS.copy()
        elif language == "javascript":
            allowed_domains = NPM_PACKAGE_DOMAINS.copy()
        else:
            allowed_domains = []

    # No domains = no filtering
    if not allowed_domains:
        return "[]"

    # Create DNS zone
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

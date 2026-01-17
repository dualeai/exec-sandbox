"""Tests for DNS zone configuration (dns_filter.py)."""

import json

import pytest

from exec_sandbox.dns_filter import (
    NPM_PACKAGE_DOMAINS,
    PYTHON_PACKAGE_DOMAINS,
    create_dns_records,
    create_dns_zone,
    generate_dns_zones_json,
    parse_dns_zones_json,
)


def test_create_dns_records():
    """Test DNS record creation from domain list."""
    records = create_dns_records(["pypi.org", "example.com"])

    assert len(records) == 2
    assert records[0].name == "pypi.org"
    # Regexp matches domain and all subdomains, with optional trailing dot
    assert records[0].Regexp == r"^(.*\.)?pypi\.org\.?$"

    assert records[1].name == "example.com"
    assert records[1].Regexp == r"^(.*\.)?example\.com\.?$"


def test_create_dns_zone():
    """Test DNS zone creation."""
    zone = create_dns_zone(["pypi.org", "npm.org"], zone_name="test-zone")

    assert zone.name == "test-zone"
    assert len(zone.records) == 2
    assert zone.defaultIP == "0.0.0.0"  # Block others by default


def test_create_dns_zone_allow_others():
    """Test DNS zone with allow others mode."""
    zone = create_dns_zone(["pypi.org"], block_others=False)

    assert zone.defaultIP == "8.8.8.8"  # Forward others to DNS


def test_generate_dns_zones_json_python():
    """Test JSON generation for Python defaults."""
    zones_json = generate_dns_zones_json(None, "python")

    assert "pypi.org" in zones_json
    assert "files.pythonhosted.org" in zones_json

    # Verify valid JSON
    zones = json.loads(zones_json)
    assert len(zones) == 1
    assert len(zones[0]["records"]) == len(PYTHON_PACKAGE_DOMAINS)


def test_generate_dns_zones_json_javascript():
    """Test JSON generation for JavaScript defaults."""
    zones_json = generate_dns_zones_json(None, "javascript")

    assert "registry.npmjs.org" in zones_json

    zones = json.loads(zones_json)
    assert len(zones) == 1
    assert len(zones[0]["records"]) == len(NPM_PACKAGE_DOMAINS)


def test_generate_dns_zones_json_custom():
    """Test JSON generation with custom domains."""
    zones_json = generate_dns_zones_json(["custom.com"], "python")

    assert "custom.com" in zones_json
    assert "pypi.org" not in zones_json  # Custom overrides defaults

    zones = json.loads(zones_json)
    assert len(zones[0]["records"]) == 1


def test_generate_dns_zones_json_empty():
    """Test JSON generation with no domains."""
    zones_json = generate_dns_zones_json([], "python")

    assert zones_json == "[]"


def test_parse_dns_zones_json():
    """Test parsing DNS zones JSON."""
    zones_json = generate_dns_zones_json(["test.com"], "python")
    zones = parse_dns_zones_json(zones_json)

    assert len(zones) == 1
    assert zones[0].name == ""  # Default zone_name is empty string
    assert len(zones[0].records) == 1
    assert zones[0].records[0].name == "test.com"


def test_parse_dns_zones_json_invalid():
    """Test parsing invalid JSON."""
    with pytest.raises(ValueError, match="Invalid DNS zones JSON"):
        parse_dns_zones_json("invalid json")


def test_regex_pattern_escapes_dots():
    """Test that dots in domains are properly escaped for regex."""
    records = create_dns_records(["example.com"])

    # Should escape dots and match domain + subdomains + optional trailing dot
    assert r"\." in records[0].Regexp
    assert records[0].Regexp == r"^(.*\.)?example\.com\.?$"


# =============================================================================
# Security tests for domain validation
# =============================================================================


class TestDomainValidation:
    """Security tests for domain validation to prevent regex injection and ReDoS."""

    def test_valid_domains(self):
        """Test that valid domains are accepted."""
        valid_domains = [
            "pypi.org",
            "files.pythonhosted.org",
            "registry.npmjs.org",
            "example.com",
            "sub.domain.example.com",
            "a-hyphen.example.com",
            "123.example.com",
        ]
        # Should not raise
        records = create_dns_records(valid_domains)
        assert len(records) == len(valid_domains)

    def test_invalid_domain_empty(self):
        """Test that empty domain is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            create_dns_records([""])

    def test_invalid_domain_too_long(self):
        """Test that overly long domain is rejected."""
        long_domain = "a" * 250 + ".com"  # > 253 chars
        with pytest.raises(ValueError, match="too long"):
            create_dns_records([long_domain])

    def test_invalid_domain_single_label(self):
        """Test that single-label domain is rejected."""
        with pytest.raises(ValueError, match="at least 2 labels"):
            create_dns_records(["localhost"])

    def test_invalid_domain_empty_label(self):
        """Test that domain with empty label is rejected."""
        with pytest.raises(ValueError, match="empty label"):
            create_dns_records(["bad..domain.com"])

    def test_invalid_domain_label_too_long(self):
        """Test that domain with label > 63 chars is rejected."""
        long_label = "a" * 64 + ".com"
        with pytest.raises(ValueError, match="label too long"):
            create_dns_records([long_label])

    def test_invalid_domain_starts_with_hyphen(self):
        """Test that domain label starting with hyphen is rejected."""
        with pytest.raises(ValueError, match="Invalid domain label"):
            create_dns_records(["-invalid.com"])

    def test_invalid_domain_ends_with_hyphen(self):
        """Test that domain label ending with hyphen is rejected."""
        with pytest.raises(ValueError, match="Invalid domain label"):
            create_dns_records(["invalid-.com"])

    def test_invalid_domain_tld_with_numbers(self):
        """Test that TLD with numbers is rejected."""
        with pytest.raises(ValueError, match="TLD must be alphabetic"):
            create_dns_records(["example.123"])

    def test_invalid_domain_special_characters(self):
        """Test that domain with special characters is rejected."""
        invalid_domains = [
            "example.com/path",
            "example.com;rm -rf",
            "example$(whoami).com",
            "example`id`.com",
            "example|cat.com",
            "example&cmd.com",
        ]
        for domain in invalid_domains:
            with pytest.raises(ValueError):
                create_dns_records([domain])

    def test_regex_injection_prevention(self):
        """Test that regex metacharacters in domains are rejected."""
        # These could cause ReDoS or regex injection if not validated
        malicious_domains = [
            ".*",
            "(.*)+evil.com",
            "[a-z]+.com",
            "^start.com",
            "end$.com",
            "a{100}.com",
            "a|b.com",
        ]
        for domain in malicious_domains:
            with pytest.raises(ValueError):
                create_dns_records([domain])

    def test_unicode_domain_rejected(self):
        """Test that Unicode/IDN domains are rejected (must use punycode)."""
        # Unicode domains should be converted to punycode before validation
        with pytest.raises(ValueError):
            create_dns_records(["例え.jp"])  # Japanese characters

    def test_trailing_dot_handled(self):
        """Test that domains with trailing dots are handled correctly."""
        # FQDN format with trailing dot should be handled
        records = create_dns_records(["example.com."])
        assert len(records) == 1

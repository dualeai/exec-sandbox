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

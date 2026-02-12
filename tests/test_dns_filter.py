"""Tests for outbound allow configuration (dns_filter.py)."""

import json

import pytest

from exec_sandbox.dns_filter import (
    NPM_PACKAGE_DOMAINS,
    PYTHON_PACKAGE_DOMAINS,
    create_outbound_patterns,
    generate_outbound_allow_json,
)


def test_create_outbound_patterns():
    """Test outbound pattern creation from domain list."""
    patterns = create_outbound_patterns(["pypi.org", "example.com"])

    assert len(patterns) == 2
    # Regexp matches domain and all subdomains, with optional trailing dot
    assert patterns[0] == r"^(.*\.)?pypi\.org\.?$"
    assert patterns[1] == r"^(.*\.)?example\.com\.?$"


def test_generate_outbound_allow_json_python():
    """Test JSON generation for Python defaults."""
    json_str = generate_outbound_allow_json(None, "python")

    # Verify valid JSON with correct number of patterns
    patterns = json.loads(json_str)
    assert len(patterns) == len(PYTHON_PACKAGE_DOMAINS)
    # Each pattern is a regex string containing the escaped domain
    assert all(isinstance(p, str) for p in patterns)
    assert any("pypi" in p for p in patterns)
    assert any("pythonhosted" in p for p in patterns)


def test_generate_outbound_allow_json_javascript():
    """Test JSON generation for JavaScript defaults."""
    json_str = generate_outbound_allow_json(None, "javascript")

    patterns = json.loads(json_str)
    assert len(patterns) == len(NPM_PACKAGE_DOMAINS)
    assert any("npmjs" in p for p in patterns)


def test_generate_outbound_allow_json_custom():
    """Test JSON generation with custom domains."""
    json_str = generate_outbound_allow_json(["custom.com"], "python")

    patterns = json.loads(json_str)
    assert len(patterns) == 1
    assert "custom" in patterns[0]
    assert "pypi" not in patterns[0]  # Custom overrides defaults


def test_generate_outbound_allow_json_empty_blocks_all():
    """Test JSON generation with empty domains returns empty array.

    When allowed_domains=[] (empty list), OutboundAllow is empty.
    Actual blocking is handled by the BlockAllOutbound flag separately.
    """
    json_str = generate_outbound_allow_json([], "python")
    assert json_str == "[]"


def test_generate_outbound_allow_json_none_uses_defaults_or_no_filter():
    """Test JSON generation with None uses language defaults or no filtering."""
    # For raw language (no defaults), None means no filtering
    json_str = generate_outbound_allow_json(None, "raw")
    assert json_str == "[]"


def test_regex_pattern_escapes_dots():
    """Test that dots in domains are properly escaped for regex."""
    patterns = create_outbound_patterns(["example.com"])

    # Should escape dots and match domain + subdomains + optional trailing dot
    assert r"\." in patterns[0]
    assert patterns[0] == r"^(.*\.)?example\.com\.?$"


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
        patterns = create_outbound_patterns(valid_domains)
        assert len(patterns) == len(valid_domains)

    def test_invalid_domain_empty(self):
        """Test that empty domain is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            create_outbound_patterns([""])

    def test_invalid_domain_too_long(self):
        """Test that overly long domain is rejected."""
        long_domain = "a" * 250 + ".com"  # > 253 chars
        with pytest.raises(ValueError, match="too long"):
            create_outbound_patterns([long_domain])

    def test_invalid_domain_single_label(self):
        """Test that single-label domain is rejected."""
        with pytest.raises(ValueError, match="at least 2 labels"):
            create_outbound_patterns(["localhost"])

    def test_invalid_domain_empty_label(self):
        """Test that domain with empty label is rejected."""
        with pytest.raises(ValueError, match="empty label"):
            create_outbound_patterns(["bad..domain.com"])

    def test_invalid_domain_label_too_long(self):
        """Test that domain with label > 63 chars is rejected."""
        long_label = "a" * 64 + ".com"
        with pytest.raises(ValueError, match="label too long"):
            create_outbound_patterns([long_label])

    def test_invalid_domain_starts_with_hyphen(self):
        """Test that domain label starting with hyphen is rejected."""
        with pytest.raises(ValueError, match="Invalid domain label"):
            create_outbound_patterns(["-invalid.com"])

    def test_invalid_domain_ends_with_hyphen(self):
        """Test that domain label ending with hyphen is rejected."""
        with pytest.raises(ValueError, match="Invalid domain label"):
            create_outbound_patterns(["invalid-.com"])

    def test_invalid_domain_tld_with_numbers(self):
        """Test that TLD with numbers is rejected."""
        with pytest.raises(ValueError, match="TLD must be alphabetic"):
            create_outbound_patterns(["example.123"])

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
                create_outbound_patterns([domain])

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
                create_outbound_patterns([domain])

    def test_unicode_domain_rejected(self):
        """Test that Unicode/IDN domains are rejected (must use punycode)."""
        # Unicode domains should be converted to punycode before validation
        with pytest.raises(ValueError):
            create_outbound_patterns(["例え.jp"])  # Japanese characters

    def test_trailing_dot_handled(self):
        """Test that domains with trailing dots are handled correctly."""
        # FQDN format with trailing dot should be handled
        patterns = create_outbound_patterns(["example.com."])
        assert len(patterns) == 1

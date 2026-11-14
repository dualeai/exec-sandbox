"""Tests for outbound allow configuration (dns_filter.py)."""

import json
import re

import pytest

from exec_sandbox.dns_filter import (
    create_outbound_patterns,
    generate_outbound_allow_json,
    validate_domain,
)


def _allows(patterns_json: str, host: str) -> bool:
    """Whether gvproxy lets `host` through: it lowercases the DNS name or TLS SNI, drops a
    trailing dot, and allows it when one pattern matches (Go regexp MatchString, which
    Python's re.search follows for these patterns)."""
    return any(re.search(pattern, host) for pattern in json.loads(patterns_json))


def test_create_outbound_patterns():
    """Test outbound pattern creation from domain list."""
    patterns = create_outbound_patterns(["pypi.org", "example.com"])

    assert len(patterns) == 2
    # Regexp matches domain and all subdomains, with optional trailing dot
    assert patterns[0] == r"^(.*\.)?pypi\.org\.?$"
    assert patterns[1] == r"^(.*\.)?example\.com\.?$"


@pytest.mark.parametrize(
    ("language", "allowed", "blocked"),
    [
        pytest.param(
            "python",
            ["pypi.org", "files.pythonhosted.org"],
            ["registry.npmjs.org", "example.com"],
            id="python-pypi",
        ),
        pytest.param("javascript", ["registry.npmjs.org"], ["pypi.org", "example.com"], id="javascript-npm"),
    ],
)
def test_none_is_the_package_registries_of_the_language(language: str, allowed: list[str], blocked: list[str]):
    """allowed_domains=None: the package registries of the language only."""
    patterns = generate_outbound_allow_json(None, language)

    assert [host for host in allowed if _allows(patterns, host)] == allowed
    assert [host for host in blocked if _allows(patterns, host)] == []


def test_generate_outbound_allow_json_custom():
    """Custom domains replace the language's registries."""
    patterns = generate_outbound_allow_json(["custom.com"], "python")

    assert _allows(patterns, "custom.com")
    assert _allows(patterns, "api.custom.com")
    assert not _allows(patterns, "pypi.org")


def test_generate_outbound_allow_json_empty_means_no_filter():
    """allowed_domains=[]: no pattern, so no domain filter (with allow_network, all outbound
    is allowed). Blocking all outbound is the separate BlockAllOutbound flag."""
    assert generate_outbound_allow_json([], "python") == "[]"


def test_generate_outbound_allow_json_none_uses_defaults_or_no_filter():
    """raw has no package registry: None means no domain filter."""
    assert generate_outbound_allow_json(None, "raw") == "[]"


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("example.com", True),
        ("www.example.com", True),
        ("a.b.example.com", True),
        ("exampleXcom", False),  # an unescaped dot would match any character
        ("notexample.com", False),  # a subdomain needs its own label
        ("example.com.evil.org", False),
        ("example.co", False),
    ],
)
def test_pattern_matches_the_domain_and_its_subdomains_only(host: str, allowed: bool):
    """The dots are literal: only the domain and its subdomains match."""
    assert _allows(json.dumps(create_outbound_patterns(["example.com"])), host) is allowed


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

    @pytest.mark.parametrize("domain", ["bad..domain.com", "example.com..", "example.com..."])
    def test_invalid_domain_empty_label(self, domain: str):
        """An empty label is refused, also after the one final dot of the FQDN form."""
        with pytest.raises(ValueError, match="empty label"):
            create_outbound_patterns([domain])

    def test_a_253_character_name_with_its_final_dot_is_valid(self):
        """The final dot of the FQDN form is not part of the 253-character limit."""
        name = ".".join(["a" * 63, "b" * 63, "c" * 63, "d" * 57, "com"])
        assert len(name) == 253
        validate_domain(name + ".")
        with pytest.raises(ValueError, match="too long"):
            validate_domain("e" + name)

    def test_a_name_over_253_characters_is_too_long(self):
        """254 characters of short labels: refused by the name limit itself, not by a label
        limit."""
        name = ".".join(["abc"] * 62) + ".commm"
        assert len(name) == 253
        validate_domain(name)
        with pytest.raises(ValueError, match=r"^Domain too long"):
            validate_domain("a" + name)

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

    def test_a_domain_with_capitals_allows_its_lowercase_names(self):
        """gvproxy lowercases DNS names and TLS SNI before its match: "GitHub.com" allows
        github.com."""
        patterns = json.dumps(create_outbound_patterns(["GitHub.com"]))
        assert _allows(patterns, "github.com")
        assert _allows(patterns, "api.github.com")

    def test_a_newline_in_a_label_is_refused(self):
        """`$` of re.match allows a final newline: a label is checked whole (fullmatch)."""
        with pytest.raises(ValueError, match="Invalid domain label"):
            validate_domain("example\n.com")

    def test_trailing_dot_handled(self):
        """A domain in FQDN form (trailing dot) allows the same names as without the dot."""
        patterns = json.dumps(create_outbound_patterns(["example.com."]))
        assert _allows(patterns, "example.com")
        assert _allows(patterns, "www.example.com")
